"""Utility functions for weather tests."""

from datetime import datetime
from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import requests


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
