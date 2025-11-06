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
class BBoxCache:
    """Cached data for a merged bounding box region.

    Stores weather data for a contiguous geographic region using integer
    indices for efficient lookups. Float coordinates are kept for info only.

    Attributes:
        init_idx: Index of the initialization time in the global init_times array
        lat_min: Minimum latitude value (for info/logging purposes)
        lat_max: Maximum latitude value (for info/logging purposes)
        lon_min: Minimum longitude value (for info/logging purposes)
        lon_max: Maximum longitude value (for info/logging purposes)
        lat_idx_start: Start index in the global latitude array
        lat_idx_end: End index (exclusive) in the global latitude array
        lon_idx_start: Start index in the global longitude array
        lon_idx_end: End index (exclusive) in the global longitude array
        variables: Dictionary mapping variable names to data arrays with
            shape (lat, lon, pred_td)
    """

    init_idx: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    lat_idx_start: int
    lat_idx_end: int
    lon_idx_start: int
    lon_idx_end: int
    variables: dict[str, np.ndarray]


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
            original_kwargs: Original query parameters
            grid_chunk: Number of grid points per chunk dimension (default: 8).
                       E.g., grid_chunk=8 means 8×8 grid points per chunk.
        """
        self._qe = query_engine
        self._model = model
        self._variables = variables
        self._init_times = np.array(init_times)
        self._prediction_timedeltas = np.array(prediction_timedeltas)
        self._latitudes = np.array(latitudes)
        self._longitudes = np.array(longitudes)
        self._kwargs = dict(original_kwargs)
        self._grid_chunk = grid_chunk

        # Parsed params for API queries
        self._init_times_list = [
            pd.Timestamp(t).to_pydatetime() for t in self._init_times
        ]
        self._pred_td_hours = [
            int(td / np.timedelta64(1, "h")) for td in self._prediction_timedeltas
        ]

        # Cache stores merged bboxes: bbox_id -> BBoxCache with metadata and data arrays
        self._bbox_cache: dict[str, BBoxCache] = {}

        # Spatial index: (init_idx, lat_chunk, lon_chunk) -> bbox_id
        self._chunk_to_bbox: dict[tuple[int, int, int], str] = {}

        # Lock for thread safety
        self._lock = RLock()

    def _labels_to_indices(self, key_any: Any, coord_values: np.ndarray) -> np.ndarray:
        """Map label/positional keys from xarray into integer indices.

        Supports ints (including negative), positional slices, label scalars,
        label slices (using searchsorted), and arrays/lists of labels or
        positions. Falls back to nearest index for unmatched scalar labels.
        """
        # Integer/slice indices
        if isinstance(key_any, (int, np.integer)):
            idx = int(key_any)
            if idx < 0:
                idx = coord_values.size + idx
            return np.array([idx], dtype=int)

        if isinstance(key_any, slice):
            # If both bounds are positional (ints) or None, treat as positional
            positional = (
                key_any.start is None or isinstance(key_any.start, (int, np.integer))
            ) and (key_any.stop is None or isinstance(key_any.stop, (int, np.integer)))
            if positional:
                start = 0 if key_any.start is None else int(key_any.start)
                stop = coord_values.size if key_any.stop is None else int(key_any.stop)
                step = 1 if key_any.step is None else int(key_any.step)
                return np.arange(start, stop, step, dtype=int)

        # Label-based handling
        # Single scalar label
        if not isinstance(key_any, (slice, list, tuple, np.ndarray)):
            # Find exact match or nearest index
            matches = np.where(coord_values == key_any)[0]
            if matches.size == 0:
                # Fallback to nearest
                idx = int(np.argmin(np.abs(coord_values - key_any)))
                return np.array([idx], dtype=int)
            return matches.astype(int)

        # Slice of labels
        if isinstance(key_any, slice):
            # Determine start/stop using searchsorted; inclusive of stop
            start_val = key_any.start
            stop_val = key_any.stop
            # None means full range
            left = (
                0
                if start_val is None
                else int(np.searchsorted(coord_values, start_val, side="left"))
            )
            right = (
                coord_values.size
                if stop_val is None
                else int(np.searchsorted(coord_values, stop_val, side="right"))
            )
            step = 1 if key_any.step is None else int(key_any.step)
            return np.arange(left, right, step, dtype=int)

        # Array-like of labels/positions
        arr = np.asarray(key_any)
        if arr.dtype.kind in {"i"}:
            # Positional indices
            arr = arr.astype(int)
            arr[arr < 0] += coord_values.size
            return arr

        # Map each label to nearest/exact index
        indices: list[int] = []
        for v in arr:
            matches = np.where(coord_values == v)[0]
            if matches.size:
                indices.append(int(matches[0]))
            else:
                indices.append(int(np.argmin(np.abs(coord_values - v))))
        return np.array(indices, dtype=int)

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
        # Handle empty selections
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

    def _merge_adjacent_chunks(
        self, spatial_chunks: set[tuple[int, int]]
    ) -> list[tuple[float, float, float, float]]:
        """Merge adjacent spatial chunks into larger bounding boxes.

        Uses connected components to group chunks that are horizontally
        or vertically adjacent, then creates one bounding box per group.

        Args:
            spatial_chunks: Set of (lat_chunk, lon_chunk) tuples

        Returns:
            List of (lat_min, lat_max, lon_min, lon_max) tuples
        """
        if not spatial_chunks:
            return []

        # Build adjacency graph using connected components
        # Two chunks are adjacent if they differ by exactly grid_chunk in one dimension
        # and match in the other dimension
        chunk_to_component: dict[tuple[int, int], tuple[int, int]] = {}

        def find_component(chunk: tuple[int, int]) -> tuple[int, int] | None:
            """Find the component ID for a chunk, using path compression."""
            if chunk not in chunk_to_component:
                return None
            root = chunk
            while chunk_to_component[root] != root:
                root = chunk_to_component[root]
            # Path compression
            while chunk_to_component[chunk] != root:
                next_chunk = chunk_to_component[chunk]
                chunk_to_component[chunk] = root
                chunk = next_chunk
            return root

        def union_components(chunk1: tuple[int, int], chunk2: tuple[int, int]) -> None:
            """Union two chunks into the same component."""
            root1 = find_component(chunk1)
            root2 = find_component(chunk2)
            if root1 is None and root2 is None:
                # Both are new, create new component
                chunk_to_component[chunk1] = chunk1
                chunk_to_component[chunk2] = chunk1
            elif root1 is None and root2 is not None:
                # chunk1 is new, add to chunk2's component
                chunk_to_component[chunk1] = root2
            elif root2 is None and root1 is not None:
                # chunk2 is new, add to chunk1's component
                chunk_to_component[chunk2] = root1
            elif root1 is not None and root2 is not None and root1 != root2:
                # Merge components
                chunk_to_component[root2] = root1

        # Process all chunks and build connected components
        sorted_chunks = sorted(spatial_chunks)
        for chunk in sorted_chunks:
            lat_chunk, lon_chunk = chunk
            if chunk not in chunk_to_component:
                chunk_to_component[chunk] = chunk

            # Check for adjacent chunks (4-connected: up, down, left, right)
            neighbors = [
                (lat_chunk - self._grid_chunk, lon_chunk),  # up
                (lat_chunk + self._grid_chunk, lon_chunk),  # down
                (lat_chunk, lon_chunk - self._grid_chunk),  # left
                (lat_chunk, lon_chunk + self._grid_chunk),  # right
            ]

            for neighbor in neighbors:
                if neighbor in spatial_chunks:
                    union_components(chunk, neighbor)

        # Group chunks by their component
        components: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for chunk in spatial_chunks:
            root = find_component(chunk)
            if root is not None:
                if root not in components:
                    components[root] = []
                components[root].append(chunk)

        # Create merged bounding boxes for each component
        merged_bboxes = []
        for component_chunks in components.values():
            # Find the extent of all chunks in this component
            lat_chunks = [lat for lat, lon in component_chunks]
            lon_chunks = [lon for lat, lon in component_chunks]

            min_lat_chunk = min(lat_chunks)
            max_lat_chunk = max(lat_chunks)
            min_lon_chunk = min(lon_chunks)
            max_lon_chunk = max(lon_chunks)

            # Convert chunk indices to actual lat/lon coordinates
            lat_start = min_lat_chunk
            lat_end = min(max_lat_chunk + self._grid_chunk, len(self._latitudes))
            lon_start = min_lon_chunk
            lon_end = min(max_lon_chunk + self._grid_chunk, len(self._longitudes))

            chunk_lats = self._latitudes[lat_start:lat_end]
            chunk_lons = self._longitudes[lon_start:lon_end]

            # Expand bbox slightly to deal with floating point errors
            if len(chunk_lats) > 0 and len(chunk_lons) > 0:
                lat_min = round(chunk_lats[0], 3) - 0.001
                lat_max = round(chunk_lats[-1], 3) + 0.001
                lon_min = round(chunk_lons[0], 3) - 0.001
                lon_max = round(chunk_lons[-1], 3) + 0.001
                merged_bboxes.append((lat_min, lat_max, lon_min, lon_max))

        return merged_bboxes

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

        # Merge adjacent chunks into larger bounding boxes
        merged_bbox_coords = self._merge_adjacent_chunks(unique_spatial_chunks)
        bounding_boxes = [
            ((lat_min, lon_min), (lat_max, lon_max))
            for lat_min, lat_max, lon_min, lon_max in merged_bbox_coords
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

        # Group missing chunks by their merged bbox and init_time
        # This allows us to process each merged bbox only once
        chunk_to_bbox_coords = {}
        spatial_chunks = list(unique_spatial_chunks)
        for init_idx in unique_init_indices:
            for lat_chunk, lon_chunk in spatial_chunks:
                # Find which merged bbox this chunk belongs to
                for lat_min, lat_max, lon_min, lon_max in merged_bbox_coords:
                    # Check if this chunk's coordinates fall within this merged bbox
                    lat_end = min(lat_chunk + self._grid_chunk, len(self._latitudes))
                    lon_end = min(lon_chunk + self._grid_chunk, len(self._longitudes))

                    chunk_lats = self._latitudes[lat_chunk:lat_end]
                    chunk_lons = self._longitudes[lon_chunk:lon_end]

                    if len(chunk_lats) > 0 and len(chunk_lons) > 0:
                        # Check if chunk is within bbox (with small tolerance)
                        if (
                            chunk_lats[0] >= lat_min
                            and chunk_lats[-1] <= lat_max
                            and chunk_lons[0] >= lon_min
                            and chunk_lons[-1] <= lon_max
                        ):
                            chunk_to_bbox_coords[(init_idx, lat_chunk, lon_chunk)] = (
                                lat_min,
                                lat_max,
                                lon_min,
                                lon_max,
                            )
                            break

        # Group chunks by (init_idx, bbox_coords)
        bbox_to_chunks: dict[
            tuple[int, tuple[float, float, float, float]], list[tuple[int, int]]
        ] = {}
        for (
            init_idx,
            lat_chunk,
            lon_chunk,
        ), bbox_coords in chunk_to_bbox_coords.items():
            key = (init_idx, bbox_coords)
            if key not in bbox_to_chunks:
                bbox_to_chunks[key] = []
            bbox_to_chunks[key].append((lat_chunk, lon_chunk))

        # Process and cache data for each merged bbox (once per bbox!)
        with self._lock:
            for (init_idx, bbox_coords), chunks_in_bbox in bbox_to_chunks.items():
                lat_min, lat_max, lon_min, lon_max = bbox_coords

                # Filter dataframe for the entire merged bbox
                df_bbox = df[
                    (df["init_time"] == self._init_times[init_idx])
                    & (df["latitude"] >= lat_min)
                    & (df["latitude"] <= lat_max)
                    & (df["longitude"] >= lon_min)
                    & (df["longitude"] <= lon_max)
                ]

                # Handle empty data
                if len(df_bbox) == 0:
                    logger.warning(
                        f"No data returned for bbox "
                        f"({lat_min}, {lat_max}, {lon_min}, {lon_max})"
                    )
                    continue

                # Transform ONCE for this entire merged bbox
                ds = self._qe.transform_dataframe(df_bbox).isel(init_time=0)

                # Compute index ranges from chunks in this bbox
                lat_chunks = [lat_chunk for lat_chunk, _ in chunks_in_bbox]
                lon_chunks = [lon_chunk for _, lon_chunk in chunks_in_bbox]

                lat_idx_start = min(lat_chunks)
                lat_idx_end = min(
                    max(lat_chunks) + self._grid_chunk, len(self._latitudes)
                )
                lon_idx_start = min(lon_chunks)
                lon_idx_end = min(
                    max(lon_chunks) + self._grid_chunk, len(self._longitudes)
                )

                # Check if returned coordinate order matches expected order
                returned_lats = ds.latitude.values
                returned_lons = ds.longitude.values
                expected_lats = self._latitudes[lat_idx_start:lat_idx_end]
                expected_lons = self._longitudes[lon_idx_start:lon_idx_end]
                if not np.allclose(returned_lats, expected_lats):
                    raise ValueError(
                        "Failed to fetch lazy-loaded data: latitudes did not match:\n"
                        f"  expected: {expected_lats}"
                        f"  returned: {returned_lats}"
                    )
                if not np.allclose(returned_lons, expected_lons):
                    raise ValueError(
                        "Failed to fetch lazy-loaded data: latitudes did not match:\n"
                        f"  expected: {expected_lons}"
                        f"  returned: {returned_lons}"
                    )

                # Create bbox_id
                bbox_id = (
                    f"{init_idx}_{lat_min:.3f}_{lat_max:.3f}_"
                    f"{lon_min:.3f}_{lon_max:.3f}"
                )

                # Create BBoxCache instance
                bbox_cache = BBoxCache(
                    init_idx=init_idx,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    lat_idx_start=lat_idx_start,
                    lat_idx_end=lat_idx_end,
                    lon_idx_start=lon_idx_start,
                    lon_idx_end=lon_idx_end,
                    variables={},
                )

                # Extract all variables at once
                for var_name in self._variables:
                    if var_name not in ds.data_vars:
                        logger.warning(
                            f"Variable {var_name} not found. "
                            f"Available: {list(ds.data_vars)}"
                        )
                        continue

                    fetched_data = np.asarray(ds[var_name].data)

                    # Transpose to (lat, lon, pred_td) if needed
                    if fetched_data.ndim == 3:
                        # (pred_td, lat, lon) -> (lat, lon, pred_td)
                        fetched_data = np.transpose(fetched_data, (1, 2, 0))

                    # Store dynamically-sized array (no padding needed!)
                    bbox_cache.variables[var_name] = fetched_data.astype(np.float32)

                # Cache the bbox data
                self._bbox_cache[bbox_id] = bbox_cache

                # Update spatial index for all chunks covered by this bbox
                for lat_chunk, lon_chunk in chunks_in_bbox:
                    self._chunk_to_bbox[(init_idx, lat_chunk, lon_chunk)] = bbox_id

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

        # Compute indices for each dimension using appropriate coordinate arrays
        init_time_indices = self._labels_to_indices(init_time_key, self._init_times)
        pred_td_indices = self._labels_to_indices(
            pred_td_key, self._prediction_timedeltas
        )
        lat_indices = self._labels_to_indices(lat_key, self._latitudes)
        lon_indices = self._labels_to_indices(lon_key, self._longitudes)

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
                    lat_idx_in_bbox = lat_idx - bbox_data.lat_idx_start
                    lon_idx_in_bbox = lon_idx - bbox_data.lon_idx_start

                    # Verify indices are within bbox bounds
                    bbox_lat_size = bbox_data.lat_idx_end - bbox_data.lat_idx_start
                    bbox_lon_size = bbox_data.lon_idx_end - bbox_data.lon_idx_start

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

    def get_dtype(self, variable_name: str) -> np.dtype:
        """Get the dtype of a specific variable.

        Args:
            variable_name: Name of the variable

        Returns:
            Numpy dtype (defaults to float32 for weather data)
        """
        # Default to float32 for weather data
        return np.dtype("float32")
