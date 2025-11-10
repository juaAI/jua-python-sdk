"""In-memory lazy loading cache with bounding box chunking."""

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Any, Sequence

import numpy as np
import pandas as pd

from jua.weather._query_engine import QueryEngine
from jua.weather._types.query_payload_types import (
    ForecastQueryPayload,
    GeoFilter,
    build_init_time_arg,
    build_prediction_timedelta,
)
from jua.weather.models import Models

logger = logging.getLogger(__name__)


@dataclass
class MergedBBox:
    """A merged bounding box region.

    Stores weather data for a contiguous geographic region using integer
    indices for efficient lookups. Float coordinates are kept for info only.

    Attributes:
        lat_min: Minimum latitude value (for info/logging purposes)
        lat_max: Maximum latitude value (for info/logging purposes)
        lon_min: Minimum longitude value (for info/logging purposes)
        lon_max: Maximum longitude value (for info/logging purposes)
        lat_idx_start: Start index in the global latitude array
        lat_idx_end: End index (exclusive) in the global latitude array
        lon_idx_start: Start index in the global longitude array
        lon_idx_end: End index (exclusive) in the global longitude array
        chunks: The chunks contained in this merged bounding box
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    lat_idx_start: int
    lat_idx_end: int
    lon_idx_start: int
    lon_idx_end: int
    chunks: list[tuple[int, int]]

    def extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.lat_min, self.lon_min), (self.lat_max, self.lon_max))

    def id(self) -> str:
        return (
            f"{self.lat_idx_start}_{self.lat_idx_end}_"
            f"{self.lon_idx_start}_{self.lon_idx_end}"
        )


@dataclass
class BBoxCache:
    """Cached data for a merged bounding box region.

    Stores weather data for a contiguous geographic region using integer
    indices for efficient lookups. Float coordinates are kept for info only.

    Attributes:
        init_idx: Index of the initialization time in the global init_times array
        bbox: The merged bounding box contained in this cache
        variables: Dictionary mapping variable names to data arrays with
            shape (lat, lon, pred_td)
    """

    init_idx: int
    bbox: MergedBBox
    variables: dict[str, np.ndarray]

    def extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.bbox.extent()

    def id(self) -> str:
        return f"{self.init_idx}_{self.bbox.id()}"


class ForecastCache:
    """In-memory cache that stores data in grid chunks.

    Each chunk contains all variables and all prediction timedeltas.
    Data is loaded on-demand from the API.

    Thread-safe: Multiple arrays can safely request data concurrently.
    """

    def __init__(
        self,
        *,
        query_engine: QueryEngine,
        model: Models,
        variables: list[str],
        init_times: Sequence[np.datetime64] | Sequence[str] | Sequence[Any],
        prediction_timedeltas: Sequence[np.timedelta64] | Sequence[Any],
        latitudes: Sequence[float],
        longitudes: Sequence[float],
        increasing_lats: bool = False,
        increasing_lons: bool = True,
        original_kwargs: dict[str, Any],
        grid_chunk: int = 8,
    ) -> None:
        """Initialize the forecast cache.

        Args:
            query_engine: QueryEngine instance to fetch data
            model: Model to query
            variables: List of all variable names to fetch
            init_times: Full array of available initialization times
            prediction_timedeltas: Full array of available prediction timedeltas
            latitudes: Full array of available latitudes
            longitudes: Full array of available longitudes
            increasing_lats: Whether the latitudes are sorted in increasing order
            increasing_lons: Whether the longitudes are sorted in increasing order
            original_kwargs: Original query parameters
            grid_chunk: Number of grid points per chunk dimension (default: 8).
                E.g., grid_chunk=8 means 8x8 grid points per chunk.
        """
        self._qe = query_engine
        self._model = model
        self._variables = variables
        self._init_times = np.array(init_times)
        self._prediction_timedeltas = np.array(prediction_timedeltas)
        self._latitudes = np.array(latitudes)
        self._longitudes = np.array(longitudes)
        self._increasing_lats = increasing_lats
        self._increasing_lons = increasing_lons
        self._kwargs = dict(original_kwargs)
        self._grid_chunk = grid_chunk

        # Parsed params for API queries
        self._pred_td_hours = [
            int(td / np.timedelta64(1, "h")) for td in self._prediction_timedeltas
        ]

        # Cache stores merged bboxes: bbox_id -> BBoxCache
        self._bbox_cache: dict[str, BBoxCache] = {}

        # Spatial index: (init_idx, lat_chunk, lon_chunk) -> bbox_id
        self._chunk_to_bbox: dict[tuple[int, int, int], str] = {}

        # Lock for thread safety
        self._lock = RLock()

    def clear(self) -> None:
        """Clear all cached data and indexes."""
        with self._lock:
            self._bbox_cache.clear()
            self._chunk_to_bbox.clear()

    def _positional_to_indices(self, key_any: Any, size: int) -> np.ndarray:
        """Expand a positional indexer (ints/slices/arrays) into integer indices.

        Assumes xarray has already resolved any label-based indexing into
        positional indexers using registered xindexes on the dataset coords.
        """
        if isinstance(key_any, (int, np.integer)):
            idx = int(key_any)
            if idx < 0:
                idx += size
            return np.array([idx], dtype=int)

        if isinstance(key_any, slice):
            start, stop, step = key_any.indices(size)
            return np.arange(start, stop, step, dtype=int)

        arr = np.asarray(key_any)
        if arr.dtype.kind != "i":
            raise TypeError(
                "Indexers must be positional (ints/slices/arrays of ints). "
                "Label-based selection should be handled by xarray before "
                "reaching the backend."
            )

        arr = arr.astype(int)
        arr[arr < 0] += size
        return arr

    def _get_required_grid_cells(
        self,
        init_time_indices: np.ndarray,
        lat_indices: np.ndarray,
        lon_indices: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        """Determine which grid chunks are needed for the requested region.

        Args:
            init_time_indices: Indices of init times being accessed
            lat_indices: Indices of latitudes being accessed
            lon_indices: Indices of longitudes being accessed

        Returns:
            List of (init_time_idx, lat_chunk, lon_chunk) tuples where
            chunk indices are the start positions of grid_chunk-sized chunks
        """
        if len(lat_indices) == 0 or len(lon_indices) == 0:
            return []

        # Determine which chunks are needed
        # Chunk index = (index // grid_chunk) * grid_chunk
        lat_chunks = np.unique((lat_indices // self._grid_chunk) * self._grid_chunk)
        lon_chunks = np.unique((lon_indices // self._grid_chunk) * self._grid_chunk)

        # Generate all combinations
        grid_cells = []
        for init_idx in init_time_indices:
            for lat_chunk in lat_chunks:
                for lon_chunk in lon_chunks:
                    grid_cells.append((int(init_idx), int(lat_chunk), int(lon_chunk)))

        return grid_cells

    def _merge_chunks_into_caches(
        self, spatial_chunks: set[tuple[int, int]]
    ) -> list[MergedBBox]:
        """Merge chunks into larger MergedBBox.

        Args:
            spatial_chunks: Set of (lat_chunk, lon_chunk) tuples where each
                value is the start index (multiple of grid_chunk) in the
                latitude/longitude arrays.

        Returns:
            List of MergedBBox capturing bounds, index ranges and chunk members.
        """
        if not spatial_chunks:
            return []

        g = self._grid_chunk

        # Normalize to unit grid indices so each chunk is size 1x1 in this space
        normalized_cells = {(lat // g, lon // g) for (lat, lon) in spatial_chunks}

        # Row -> set of columns for O(1) membership/removal
        rows: dict[int, set[int]] = {}
        for r, c in normalized_cells:
            rows.setdefault(r, set()).add(c)

        def row_has_run(row_idx: int, c_start: int, c_end: int) -> bool:
            cols = rows.get(row_idx)
            if not cols:
                return False
            for c in range(c_start, c_end + 1):
                if c not in cols:
                    return False
            return True

        groups: list[MergedBBox] = []

        while rows:
            r0 = min(rows)
            c0 = min(rows[r0])

            # Expand to the right on the top row
            c1 = c0
            while (c1 + 1) in rows[r0]:
                c1 += 1

            # Expand downward while subsequent rows contain the full run
            r1 = r0
            while row_has_run(r1 + 1, c0, c1):
                r1 += 1

            # Convert rectangle back to original index space
            lat_idx_start = r0 * g
            lat_idx_end = min((r1 + 1) * g, len(self._latitudes))
            lon_idx_start = c0 * g
            lon_idx_end = min((c1 + 1) * g, len(self._longitudes))

            chunk_lats = self._latitudes[lat_idx_start:lat_idx_end]
            chunk_lons = self._longitudes[lon_idx_start:lon_idx_end]

            if len(chunk_lats) > 0 and len(chunk_lons) > 0:
                # Use min/max to be robust to coordinate order (ascending/descending)
                lat_min, lat_max = _get_padded_extent(chunk_lats)
                lon_min, lon_max = _get_padded_extent(chunk_lons)

                # Record original-space chunk members
                members: list[tuple[int, int]] = []
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        members.append((rr * g, cc * g))

                groups.append(
                    MergedBBox(
                        lat_min=lat_min,
                        lat_max=lat_max,
                        lon_min=lon_min,
                        lon_max=lon_max,
                        lat_idx_start=lat_idx_start,
                        lat_idx_end=lat_idx_end,
                        lon_idx_start=lon_idx_start,
                        lon_idx_end=lon_idx_end,
                        chunks=members,
                    )
                )

            # Remove covered cells from the map
            for r in range(r0, r1 + 1):
                cols = rows.get(r)
                if not cols:
                    continue
                for c in range(c0, c1 + 1):
                    cols.discard(c)
                if not cols:
                    rows.pop(r)

        return groups

    def _fetch_all_chunks(
        self,
        missing_chunks: list[tuple[int, int, int]],
    ) -> None:
        """Fetch all missing chunks in a single API call with multiple bounding boxes.

        This method populates the cache with all variables for the requested chunks.

        Args:
            missing_chunks: List of (init_idx, lat_chunk, lon_chunk) tuples
        """
        if not missing_chunks:
            return

        unique_spatial_chunks = set(
            (lat_chunk, lon_chunk) for _, lat_chunk, lon_chunk in missing_chunks
        )

        # Merge adjacent chunks into larger rectangular groups
        merged_bboxes = self._merge_chunks_into_caches(unique_spatial_chunks)
        bounding_boxes = [
            ((g.lat_min, g.lon_min), (g.lat_max, g.lon_max)) for g in merged_bboxes
        ]

        # Determine which init_times to fetch based on requested chunks
        unique_init_indices = sorted(set(init_idx for init_idx, _, _ in missing_chunks))
        init_times_dt = [
            pd.Timestamp(self._init_times[idx]).to_pydatetime()
            for idx in unique_init_indices
        ]

        df = self._qe.load_raw_forecast(
            payload=ForecastQueryPayload(
                models=[self._model],
                init_time=build_init_time_arg(init_times_dt),
                geo=GeoFilter(type="bounding_box", value=bounding_boxes),
                prediction_timedelta=build_prediction_timedelta(self._pred_td_hours),
                variables=self._variables,
            ),
            stream=True,
            print_progress=False,
        )

        # Process and cache data for each merged bbox group (once per group!)
        with self._lock:
            for init_idx in unique_init_indices:
                for bbox in merged_bboxes:
                    df_bbox = df[
                        (df["init_time"] == self._init_times[init_idx])
                        & (df["latitude"] >= bbox.lat_min)
                        & (df["latitude"] <= bbox.lat_max)
                        & (df["longitude"] >= bbox.lon_min)
                        & (df["longitude"] <= bbox.lon_max)
                    ]
                    if len(df_bbox) == 0:
                        logger.warning(f"No data returned for {bbox.extent()}")
                        continue

                    # Parse bbox data, reverse coordinate order if needed
                    ds = self._qe.transform_dataframe(df_bbox).isel(init_time=0)
                    if not self._increasing_lats:
                        ds = ds.isel(latitude=slice(None, None, -1))
                    if not self._increasing_lons:
                        ds = ds.isel(longitude=slice(None, None, -1))

                    # Check that returned coordinate order matches expected order
                    returned_lats = ds.latitude.values
                    returned_lons = ds.longitude.values
                    expected_lats = self._latitudes[
                        bbox.lat_idx_start : bbox.lat_idx_end
                    ]
                    expected_lons = self._longitudes[
                        bbox.lon_idx_start : bbox.lon_idx_end
                    ]
                    if not np.allclose(returned_lats, expected_lats):
                        raise ValueError(
                            "Failed to fetch lazy-loaded data: latitudes don't match:\n"
                            f"  expected: {expected_lats}\n"
                            f"  returned: {returned_lats}\n"
                        )
                    if not np.allclose(returned_lons, expected_lons):
                        raise ValueError(
                            "Failed to fetch lazy-loaded data: latitudes don't match:\n"
                            f"  expected: {expected_lons}\n"
                            f"  returned: {returned_lons}\n"
                        )

                    # Extract all variables at once
                    cache = BBoxCache(init_idx=init_idx, bbox=bbox, variables={})
                    for var_name in self._variables:
                        if var_name not in ds.data_vars:
                            logger.warning(
                                f"Variable {var_name} not found. "
                                f"Available: {list(ds.data_vars)}"
                            )
                            continue

                        fetched_data = np.asarray(ds[var_name].data)

                        # (pred_td, lat, lon) -> (lat, lon, pred_td)
                        fetched_data = np.transpose(fetched_data, (1, 2, 0))
                        cache.variables[var_name] = fetched_data.astype(np.float32)

                    # Cache the bbox data
                    self._bbox_cache[cache.id()] = cache

                    # Update spatial index for all chunks covered by this bbox
                    for lat_chunk, lon_chunk in cache.bbox.chunks:
                        self._chunk_to_bbox[(init_idx, lat_chunk, lon_chunk)] = (
                            cache.id()
                        )

    def get_variable(self, variable_name: str, key: tuple) -> np.ndarray:
        """Get the numpy array for a specific variable and index key.

        Args:
            variable_name: Name of the variable to retrieve
            key: Indexing tuple (init_time, pred_td, lat, lon)

        Returns:
            4D numpy array subset for the requested indices

        Raises:
            ValueError: If variable_name is not valid
        """
        # Extract indices from key
        init_time_key, pred_td_key, lat_key, lon_key = key

        # Compute indices for each dimension; keys are positional at this point
        init_time_indices = self._positional_to_indices(
            init_time_key, self._init_times.size
        )
        pred_td_indices = self._positional_to_indices(
            pred_td_key, self._prediction_timedeltas.size
        )
        lat_indices = self._positional_to_indices(lat_key, self._latitudes.size)
        lon_indices = self._positional_to_indices(lon_key, self._longitudes.size)

        # Get required grid cells
        grid_cells = self._get_required_grid_cells(
            init_time_indices, lat_indices, lon_indices
        )

        # Find which chunks are missing from cache
        missing_chunks = []
        loaded_cells = {}

        with self._lock:
            for init_idx, lat_chunk, lon_chunk in grid_cells:
                chunk_key = (init_idx, lat_chunk, lon_chunk)
                if chunk_key in self._chunk_to_bbox:
                    # Chunk is cached, get the bbox_id
                    bbox_id = self._chunk_to_bbox[chunk_key]
                    loaded_cells[chunk_key] = bbox_id
                else:
                    missing_chunks.append((init_idx, lat_chunk, lon_chunk))

        # If there are missing chunks, fetch them all in a single query
        if missing_chunks:
            self._fetch_all_chunks(missing_chunks)
            with self._lock:
                for init_idx, lat_chunk, lon_chunk in missing_chunks:
                    chunk_key = (init_idx, lat_chunk, lon_chunk)
                    if chunk_key in self._chunk_to_bbox:
                        bbox_id = self._chunk_to_bbox[chunk_key]
                        loaded_cells[chunk_key] = bbox_id

        # Stitch grid cells together with the global prediction_timedelta indices
        result = self._stitch_grid_cells(
            variable_name,
            loaded_cells,
            init_time_indices,
            pred_td_indices,
            lat_indices,
            lon_indices,
        )

        return result

    def _stitch_grid_cells(
        self,
        variable_name: str,
        loaded_cells: dict[tuple[int, int, int], str],
        init_time_indices: np.ndarray,
        pred_td_indices: np.ndarray,
        lat_indices: np.ndarray,
        lon_indices: np.ndarray,
    ) -> np.ndarray:
        """Stitch grid chunks together to form the requested array.

        Args:
            variable_name: Name of the variable to extract
            loaded_cells: Dict mapping (init_idx, lat_chunk, lon_chunk) to bbox_id
            init_time_indices: Requested init time indices
            pred_td_indices: Requested prediction timedelta indices
            lat_indices: Requested latitude indices
            lon_indices: Requested longitude indices

        Returns:
            4D array with shape (init_times, pred_tds, lats, lons)
        """
        # Initialize output array
        result = np.full(
            (
                len(init_time_indices),
                len(pred_td_indices),
                len(lat_indices),
                len(lon_indices),
            ),
            np.nan,
            dtype=np.float32,
        )

        # Fill in data from bboxes
        for out_init_idx, init_idx in enumerate(init_time_indices):
            for out_lat_idx, lat_idx in enumerate(lat_indices):
                # Determine which chunk this latitude belongs to
                lat_chunk = (lat_idx // self._grid_chunk) * self._grid_chunk

                for out_lon_idx, lon_idx in enumerate(lon_indices):
                    # Determine which chunk this longitude belongs to
                    lon_chunk = (lon_idx // self._grid_chunk) * self._grid_chunk

                    # Get bbox_id for this chunk
                    cell_key = (int(init_idx), lat_chunk, lon_chunk)
                    if cell_key not in loaded_cells:
                        continue

                    bbox_id = loaded_cells[cell_key]
                    if bbox_id not in self._bbox_cache:
                        continue

                    bbox_data = self._bbox_cache[bbox_id]

                    # Check if variable exists in this bbox
                    if variable_name not in bbox_data.variables:
                        continue

                    # Calculate position within bbox
                    lat_idx_in_bbox = lat_idx - bbox_data.bbox.lat_idx_start
                    lon_idx_in_bbox = lon_idx - bbox_data.bbox.lon_idx_start

                    # Verify indices are within bbox bounds
                    bbox_lat_size = (
                        bbox_data.bbox.lat_idx_end - bbox_data.bbox.lat_idx_start
                    )
                    bbox_lon_size = (
                        bbox_data.bbox.lon_idx_end - bbox_data.bbox.lon_idx_start
                    )

                    if not (
                        0 <= lat_idx_in_bbox < bbox_lat_size
                        and 0 <= lon_idx_in_bbox < bbox_lon_size
                    ):
                        continue

                    # Extract data from the bbox
                    var_data = bbox_data.variables[variable_name]

                    # Get all pred_tds for this location
                    cell_values = var_data[lat_idx_in_bbox, lon_idx_in_bbox, :]

                    # Select only requested pred_tds
                    result[out_init_idx, :, out_lat_idx, out_lon_idx] = cell_values[
                        pred_td_indices
                    ]

        return result

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return shape of full coordinate space.

        Shape: (init_time, prediction_timedelta, latitude, longitude)
        """
        return (
            self._init_times.size,
            self._prediction_timedeltas.size,
            self._latitudes.size,
            self._longitudes.size,
        )

    def get_dtype(self) -> np.dtype:
        """Get the dtype of a specific variable.

        Args:
            variable_name: Name of the variable

        Returns:
            Numpy dtype
        """
        return np.dtype("float32")


def _get_padded_extent(values: np.ndarray, n: int = 3) -> tuple[float, float]:
    return (
        round(float(np.min(values)), n) - (10**-n),
        round(float(np.max(values)), n) + (10**-n),
    )
