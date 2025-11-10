"""Utility functions for weather tests."""

from datetime import datetime, timedelta
from io import BytesIO
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import requests
import xarray as xr

from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather._query_engine import QueryEngine
from jua.weather._types.query_payload_types import ForecastQueryPayload
from jua.weather.models import Models
from jua.weather.statistics import Statistics
from jua.weather.variables import Variables


def create_mock_arrow_response(df: pd.DataFrame) -> requests.Response:
    """Create a mock requests.Response with Arrow IPC stream data.

    Args:
        df: DataFrame with columns: init_time, model, prediction_timedelta,
            latitude, longitude, and variable columns. For point queries, also
            include a 'point' column with integer indices.

    Returns:
        Mock Response object with Arrow data in response body.
    """
    # Convert DataFrame to Arrow Table
    table = pa.Table.from_pandas(df)

    # Serialize to Arrow IPC stream format
    buffer = BytesIO()
    with pa_ipc.new_stream(buffer, table.schema) as writer:
        writer.write_table(table)

    # Get the data and create a fresh buffer for the mock
    arrow_data = buffer.getvalue()

    # Create mock response
    mock_response = Mock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.ok = True
    mock_response.headers = {
        "content-type": "application/vnd.apache.arrow.stream",
        "content-length": str(len(arrow_data)),
    }

    # Create a raw attribute that acts like a proper file-like object
    # PyArrow needs to be able to read from this stream
    raw_buffer = BytesIO(arrow_data)
    raw_buffer.decode_content = True
    mock_response.raw = raw_buffer

    # Also set content for direct access and iter_content
    mock_response.content = arrow_data

    # Mock iter_content for fallback path
    def iter_content(chunk_size=1):
        data = BytesIO(arrow_data)
        while True:
            chunk = data.read(chunk_size)
            if not chunk:
                break
            yield chunk

    mock_response.iter_content = iter_content

    return mock_response


def create_point_dataframe(
    num_points: int,
    num_hours: int,
    init_time: datetime,
    lat_start: float = 50.0,
    lon_start: float = 8.0,
) -> pd.DataFrame:
    """Create a DataFrame for point query mock responses.

    Args:
        num_points: Number of points to generate data for.
        num_hours: Number of forecast hours (0 to num_hours inclusive).
        init_time: Forecast initialization time.
        lat_start: Starting latitude for generating point locations.
        lon_start: Starting longitude for generating point locations.

    Returns:
        DataFrame with columns: init_time, model, prediction_timedelta, point,
        latitude, longitude, air_temperature_at_height_level_2m,
        wind_speed_at_height_level_10m.
    """
    rows = []

    for point_idx in range(num_points):
        # Generate distinct lat/lon for each point
        lat = lat_start + point_idx * 0.5
        lon = lon_start + point_idx * 0.5

        for hour in range(num_hours + 1):  # 0 to num_hours inclusive
            # Generate realistic weather values with some variation
            # Temperature: 280-300K with slight temporal trend
            temp = 285.0 + np.sin(hour * 0.1) * 5.0 + np.random.randn() * 0.5
            # Wind speed: 0-20 m/s with variation
            wind = 10.0 + np.sin(hour * 0.2) * 5.0 + np.random.randn() * 1.0
            wind = max(0.0, wind)  # Wind speed can't be negative

            rows.append(
                {
                    "init_time": init_time,
                    "model": "ept2",
                    "prediction_timedelta": hour * 60,  # in minutes
                    "point": point_idx,
                    "latitude": lat,
                    "longitude": lon,
                    "air_temperature_at_height_level_2m": temp,
                    "wind_speed_at_height_level_10m": wind,
                }
            )

    return pd.DataFrame(rows)


def create_slice_dataframe(
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    num_hours: int,
    init_time: datetime,
    lat_step: float = 0.5,
    lon_step: float = 0.5,
) -> pd.DataFrame:
    """Create a DataFrame for grid/slice query mock responses.

    Args:
        lat_range: Tuple of (min_lat, max_lat) for the grid.
        lon_range: Tuple of (min_lon, max_lon) for the grid.
        num_hours: Number of forecast hours (0 to num_hours inclusive).
        init_time: Forecast initialization time.
        lat_step: Grid resolution in latitude direction.
        lon_step: Grid resolution in longitude direction.

    Returns:
        DataFrame with columns: init_time, model, prediction_timedelta,
        latitude, longitude, air_temperature_at_height_level_2m,
        wind_speed_at_height_level_10m.
        Note: NO 'point' column for slice queries.
    """
    rows = []

    # Generate grid points
    lats = np.arange(lat_range[0], lat_range[1] + lat_step / 2, lat_step)
    lons = np.arange(lon_range[0], lon_range[1] + lon_step / 2, lon_step)

    for hour in range(num_hours + 1):  # 0 to num_hours inclusive
        for lat in lats:
            for lon in lons:
                # Generate realistic weather values with spatial and temporal variation
                # Temperature varies with latitude and time
                temp = 285.0 + (lat - lat_range[0]) * 2.0 + np.sin(hour * 0.1) * 5.0
                temp += np.random.randn() * 0.3

                # Wind speed varies with longitude and time
                wind = 10.0 + (lon - lon_range[0]) * 1.0 + np.sin(hour * 0.2) * 5.0
                wind += np.random.randn() * 0.8
                wind = max(0.0, wind)  # Wind speed can't be negative

                rows.append(
                    {
                        "init_time": init_time,
                        "model": "ept2",
                        "prediction_timedelta": hour * 60,  # in minutes
                        "latitude": lat,
                        "longitude": lon,
                        "air_temperature_at_height_level_2m": temp,
                        "wind_speed_at_height_level_10m": wind,
                    }
                )

    return pd.DataFrame(rows)


VAR_TO_IDX = {v.name: i for i, v in enumerate(Variables)}


class MockQueryEngine:
    """Mock QueryEngine that returns synthetic data."""

    def __init__(
        self,
        max_lead_time: int,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        variables: list[Variables] | list[str],
    ):
        """Initialize mock query engine with coordinate definitions.

        Args:
            init_times: List of initialization times
            max_lead_time: Maximum forecast lead time in hours
            latitudes: Array of latitude values
            longitudes: Array of longitude values
            variables: List of variable names
        """
        # Generate 4 init times per day from 2023-01-01 to 2025-11-01
        init_time_list = []
        next_init_time = datetime(2023, 1, 1, 0)
        while next_init_time < datetime(2025, 11, 1, 0):
            init_time_list.append(next_init_time)
            next_init_time += timedelta(hours=6)

        # Convert to numpy array with datetime64
        self.init_times = np.array(init_time_list, dtype="datetime64[ns]")
        self.prediction_timedeltas = np.arange(0, 60 * (max_lead_time + 1), 60)
        self.latitudes = np.sort(np.asarray(latitudes))
        self.longitudes = np.sort(np.asarray(longitudes))
        self.variables = [v.name if isinstance(v, Variables) else v for v in variables]

        # Call tracking
        self.get_forecast_call_count = 0
        self.get_forecast_index_call_count = 0

    def reset_call_counts(self):
        """Reset call tracking counters."""
        self.get_forecast_call_count = 0
        self.get_forecast_index_call_count = 0

    def get_forecast_index(
        self,
        model: Models,
        init_time: Literal["latest"] | datetime | list[datetime] | slice | None = None,
        variables: list[Variables] | list[str] | None = None,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: slice | None = None,
        longitude: slice | None = None,
    ):
        """Mock get_forecast_index that returns coordinate metadata.

        Returns filtered coordinates based on the query parameters.
        """
        self.get_forecast_index_call_count += 1

        # Filter all dimensions using the appropriate filter methods
        filtered_init_times = self._filter_init_times(init_time)
        filtered_pred_tds = self._filter_prediction_timedeltas(
            to_minutes(prediction_timedelta)
        )
        filtered_lats = self._filter_latitudes(latitude)
        filtered_lons = self._filter_longitudes(longitude)

        if latitude.start > latitude.stop:
            filtered_lats = filtered_lats[::-1]
        if longitude.start > longitude.stop:
            filtered_lons = filtered_lons[::-1]

        # Filter variables
        vars = [v.name if isinstance(v, Variables) else v for v in variables]
        vars = [v for v in vars if v in self.variables]

        # Convert numpy datetime64 to ISO strings for init_time
        init_time_strings = [pd.Timestamp(t).isoformat() for t in filtered_init_times]

        # Convert prediction timedeltas (minutes) to pandas TimedeltaIndex (ns)
        filtered_pred_tds = pd.to_timedelta(filtered_pred_tds, unit="m")

        return {
            "init_time": init_time_strings,
            "prediction_timedelta": filtered_pred_tds,
            "latitude": filtered_lats.tolist(),
            "longitude": filtered_lons.tolist(),
            "variables": vars,
        }

    def get_forecast(
        self,
        model: Models,
        init_time: Literal["latest"] | datetime | list[datetime] | slice | None = None,
        variables: list[Variables] | list[str] | None = None,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | LatLon | None = None,
        statistics: list[str] | list[Statistics] | None = None,
        method: Literal["nearest", "bilinear"] = "nearest",
        stream: bool | None = None,
        print_progress: bool | None = None,
    ) -> xr.Dataset:
        """Mock get_forecast that generates synthetic data on the fly.

        Returns:
            xarray.Dataset with mock data for requested coordinates
        """
        self.get_forecast_call_count += 1

        # Filter dimensions using filter methods
        filtered_init_times = self._filter_init_times(init_time)
        filtered_pred_tds = self._filter_prediction_timedeltas(
            to_minutes(prediction_timedelta)
        )
        filtered_lats = self._filter_latitudes(latitude)
        filtered_lons = self._filter_longitudes(longitude)

        # Use requested variables or all if not specified
        var_list = variables if variables else self.variables
        df = self._generate_dataframe(
            model,
            filtered_init_times,
            filtered_pred_tds,
            filtered_lats,
            filtered_lons,
            var_list,
        )
        return self.transform_dataframe(df, points=None, statistics=None)

    def load_raw_forecast(
        self,
        payload: ForecastQueryPayload,
        stream: bool = False,
        print_progress: bool | None = None,
    ) -> pd.DataFrame:
        """Mock load_raw_forecast that generates data on the fly.

        Args:
            payload: Query payload containing init_time, variables, and geo information
            stream: Whether to stream the response (not used in mock)
            print_progress: Whether to print progress (not used in mock)

        Returns:
            DataFrame with mock forecast data
        """
        self.get_forecast_call_count += 1

        # Parse payload to extract query parameters
        if hasattr(payload, "init_time") and payload.init_time is not None:
            init_time_selection = payload.init_time
            # Convert isoformat strings to datetime objects
            if isinstance(init_time_selection, list):
                init_time_selection = [
                    datetime.fromisoformat(t) if isinstance(t, str) else t
                    for t in init_time_selection
                ]
            elif (
                isinstance(init_time_selection, str) and init_time_selection != "latest"
            ):
                init_time_selection = datetime.fromisoformat(init_time_selection)
        else:
            init_time_selection = None

        # Extract variables
        if hasattr(payload, "variables") and payload.variables is not None:
            variables = [
                v.name if isinstance(v, Variables) else v for v in payload.variables
            ]
        else:
            variables = self.variables

        # Extract prediction_timedelta (if available in payload)
        prediction_timedelta = None
        if (
            hasattr(payload, "prediction_timedelta")
            and payload.prediction_timedelta is not None
        ):
            prediction_timedelta = payload.prediction_timedelta

        # Extract lat/lon from bounding boxes in geo
        # payload.geo.value should contain bounding boxes
        if hasattr(payload, "geo") and payload.geo is not None:
            bboxes = payload.geo.value

            # Collect all latitudes and longitudes from all bounding boxes
            all_lats = []
            all_lons = []
            for bbox in bboxes:
                (lat_min, lon_min), (lat_max, lon_max) = bbox
                # Find coordinates within this bounding box
                lat_mask = (self.latitudes >= lat_min) & (self.latitudes <= lat_max)
                lon_mask = (self.longitudes >= lon_min) & (self.longitudes <= lon_max)
                box_lats = self.latitudes[lat_mask]
                box_lons = self.longitudes[lon_mask]
                all_lats.extend(box_lats)
                all_lons.extend(box_lons)

            # Remove duplicates and sort
            unique_lats = np.array(sorted(set(all_lats)))
            unique_lons = np.array(sorted(set(all_lons)))
        else:
            # Use all coordinates if no geo filter
            unique_lats = self.latitudes
            unique_lons = self.longitudes

        # Filter coordinates using filter methods
        filtered_init_times = self._filter_init_times(init_time_selection)
        filtered_pred_tds = self._filter_prediction_timedeltas(prediction_timedelta)
        filtered_lats = unique_lats
        filtered_lons = unique_lons

        # Extract model (default to ept2 if not specified)
        model = payload.model if hasattr(payload, "model") else Models.EPT2

        # Generate DataFrame using _generate_dataframe
        df = self._generate_dataframe(
            model=model,
            init_times=filtered_init_times,
            prediction_timedelta=filtered_pred_tds,
            latitude=filtered_lats,
            longitude=filtered_lons,
            variables=variables,
        )

        return df

    @classmethod
    def transform_dataframe(
        cls,
        df: pd.DataFrame,
        points: list[LatLon] | LatLon | None = None,
        statistics: list[str] | list[Statistics] | None = None,
    ) -> xr.Dataset:
        """Transform DataFrame back to xarray Dataset.

        Args:
            df: DataFrame with columns: init_time, prediction_timedelta, latitude,
                longitude, and variable columns
            points: Points for point-based queries (not used in mock)
            statistics: Statistics to compute (not used in mock)

        Returns:
            xarray.Dataset with data indexed by coordinates
        """
        return QueryEngine.transform_dataframe(df, points, statistics)

    def _generate_dataframe(
        self,
        model: Models,
        init_times: np.ndarray,
        prediction_timedelta: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        variables: list[str],
    ) -> pd.DataFrame:
        data = []
        for init_time in init_times:
            py_init_time = pd.Timestamp(init_time).to_pydatetime()
            it_enc = py_init_time.day + 10 * py_init_time.hour
            for pt_minutes in prediction_timedelta:
                # Convert minutes to timedelta
                pt_td = pd.Timedelta(minutes=int(pt_minutes))
                for lat in latitude:
                    for lon in longitude:
                        base = it_enc + 10 * pt_minutes + 10 * lat + 10 * lon
                        var_data = {
                            var: base + 100 * VAR_TO_IDX[var] for var in variables
                        }

                        data.append(
                            {
                                "init_time": pd.Timestamp(init_time),
                                "prediction_timedelta": pt_td,
                                "latitude": lat,
                                "longitude": lon,
                                **var_data,
                            }
                        )

        df = pd.DataFrame(data)
        # Ensure proper time dtypes
        df["init_time"] = df["init_time"].astype("datetime64[ns]")
        df["prediction_timedelta"] = df["prediction_timedelta"].astype(
            "timedelta64[ns]"
        )
        return df

    def _filter_init_times(
        self,
        selection: Literal["latest"] | datetime | list[datetime] | slice | None = None,
    ) -> np.ndarray:
        """Filter init times based on selection.

        Args:
            selection: Init time selection criteria

        Returns:
            Filtered numpy array of datetime64 values
        """
        if selection is None:
            return self.init_times
        elif isinstance(selection, str) and selection == "latest":
            return self.init_times[-1:]
        elif isinstance(selection, datetime):
            # Convert to datetime64 for comparison
            target = np.datetime64(selection, "ns")
            mask = self.init_times == target
            return self.init_times[mask]
        elif isinstance(selection, list):
            # Convert list to datetime64 for comparison
            targets = np.array(
                [np.datetime64(t.replace(tzinfo=None), "ns") for t in selection]
            )
            mask = np.isin(self.init_times, targets)
            return self.init_times[mask]
        elif isinstance(selection, slice):
            start = (
                np.datetime64(selection.start, "ns")
                if selection.start
                else self.init_times[0]
            )
            stop = (
                np.datetime64(selection.stop, "ns")
                if selection.stop
                else self.init_times[-1]
            )

            # Ensure start <= stop
            if start > stop:
                start, stop = stop, start

            mask = (self.init_times >= start) & (self.init_times <= stop)
            return self.init_times[mask]

        raise ValueError(f"Incorrect init_time argument: {selection}")

    def _filter_prediction_timedeltas(
        self,
        selection: PredictionTimeDelta | None = None,
    ) -> np.ndarray:
        """Filter prediction timedeltas based on selection.

        Args:
            selection: Prediction timedelta selection in MINUTES (already converted)

        Returns:
            Filtered numpy array of prediction timedeltas (in minutes)
        """
        if selection is None:
            return self.prediction_timedeltas

        if isinstance(selection, slice):
            start = (
                selection.start
                if selection.start is not None
                else self.prediction_timedeltas.min()
            )
            stop = (
                selection.stop
                if selection.stop is not None
                else self.prediction_timedeltas.max()
            )

            # Ensure start <= stop
            if start > stop:
                start, stop = stop, start

            mask = (self.prediction_timedeltas >= start) & (
                self.prediction_timedeltas <= stop
            )
            return self.prediction_timedeltas[mask]
        elif isinstance(selection, (list, tuple, np.ndarray)):
            mask = np.isin(self.prediction_timedeltas, selection)
            return self.prediction_timedeltas[mask]
        else:
            # Single value
            mask = self.prediction_timedeltas == selection
            return self.prediction_timedeltas[mask]

    def _filter_latitudes(
        self,
        selection: SpatialSelection | None = None,
    ) -> np.ndarray:
        """Filter latitudes based on selection.

        Args:
            selection: Latitude selection (value, list, or slice)

        Returns:
            Filtered numpy array of latitudes
        """
        if selection is None:
            return self.latitudes

        if isinstance(selection, slice):
            start = (
                selection.start if selection.start is not None else self.latitudes.min()
            )
            stop = (
                selection.stop if selection.stop is not None else self.latitudes.max()
            )

            # For latitude, convention is start > stop (reversed/descending)
            if start > stop:
                start, stop = stop, start

            mask = (self.latitudes >= start) & (self.latitudes <= stop)
            return self.latitudes[mask]
        elif isinstance(selection, (list, tuple, np.ndarray)):
            mask = np.isin(self.latitudes, selection)
            return self.latitudes[mask]
        else:
            # Single value
            mask = self.latitudes == selection
            return self.latitudes[mask]

    def _filter_longitudes(
        self,
        selection: SpatialSelection | None = None,
    ) -> np.ndarray:
        """Filter longitudes based on selection.

        Args:
            selection: Longitude selection (value, list, or slice)

        Returns:
            Filtered numpy array of longitudes
        """
        if selection is None:
            return self.longitudes

        if isinstance(selection, slice):
            start = (
                selection.start
                if selection.start is not None
                else self.longitudes.min()
            )
            stop = (
                selection.stop if selection.stop is not None else self.longitudes.max()
            )

            # Ensure start <= stop
            if start > stop:
                start, stop = stop, start

            mask = (self.longitudes >= start) & (self.longitudes <= stop)
            return self.longitudes[mask]
        elif isinstance(selection, (list, tuple, np.ndarray)):
            mask = np.isin(self.longitudes, selection)
            return self.longitudes[mask]
        else:
            # Single value
            mask = self.longitudes == selection
            return self.longitudes[mask]


def to_minutes(prediction_timedelta: PredictionTimeDelta | None = None):
    if prediction_timedelta is None:
        return None

    if isinstance(prediction_timedelta, slice):
        return slice(
            prediction_timedelta.start * 60
            if prediction_timedelta.start is not None
            else None,
            prediction_timedelta.stop * 60
            if prediction_timedelta.stop is not None
            else None,
            prediction_timedelta.step,
        )
    elif isinstance(prediction_timedelta, (list, tuple)):
        return [h * 60 for h in prediction_timedelta]

    return prediction_timedelta * 60
