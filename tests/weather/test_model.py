"""Functional tests for Model.get_forecasts() with mocked API responses."""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from jua.client import JuaClient
from jua.types.geo import LatLon
from jua.weather import Model, Models, Variables
from tests.weather.utils import (
    create_mock_arrow_response,
    create_point_dataframe,
    create_slice_dataframe,
)


@pytest.fixture
def mock_client():
    """Create a mock JuaClient with fake credentials."""
    client = JuaClient()
    client.settings.auth.api_key_id = "test_key_id"
    client.settings.auth.api_key_secret = "test_key_secret"
    return client


@pytest.fixture
def ept2_model(mock_client):
    """Create a Model instance for EPT2."""
    return Model(client=mock_client, model=Models.EPT2)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("print_progress", [True, False])
def test_single_point_query(ept2_model, stream, print_progress):
    """Test get_forecasts with a single point."""
    init_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=None)
    requested_point = LatLon(lat=50.0, lon=8.0)

    # Create mock data for 1 point, 7 hours (0-6)
    df = create_point_dataframe(
        num_points=1,
        num_hours=6,
        init_time=init_time,
        lat_start=50.0,
        lon_start=8.0,
    )
    mock_response = create_mock_arrow_response(df)

    # Mock the API call
    with patch.object(
        ept2_model._query_engine._api, "post", return_value=mock_response
    ):
        result = ept2_model.get_forecasts(
            points=requested_point,
            max_lead_time=6,
            variables=[
                Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
                Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            ],
            stream=stream,
            print_progress=print_progress,
        )

    # Convert to xarray for validation
    ds = result.to_xarray()

    # Verify dimensions
    assert "points" in ds.dims
    assert "init_time" in ds.dims
    assert "prediction_timedelta" in ds.dims

    # Verify shapes
    assert ds.sizes["points"] == 1
    assert ds.sizes["init_time"] == 1
    assert ds.sizes["prediction_timedelta"] == 7  # 0-6 hours inclusive

    # Verify coordinates
    # Convert np.datetime64 to datetime for comparison
    returned_init_time = pd.Timestamp(ds.init_time.values[0]).to_pydatetime()
    assert returned_init_time == init_time

    # Verify variables exist
    assert "air_temperature_at_height_level_2m" in ds.data_vars
    assert "wind_speed_at_height_level_10m" in ds.data_vars

    # Verify data is present and within reasonable ranges
    temp_data = ds["air_temperature_at_height_level_2m"].values
    assert temp_data.shape == (1, 1, 7)  # (points, init_time, prediction_timedelta)
    assert 280.0 < temp_data.mean() < 300.0

    wind_data = ds["wind_speed_at_height_level_10m"].values
    assert wind_data.shape == (1, 1, 7)
    assert 0.0 <= wind_data.min()
    assert wind_data.max() < 30.0


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("print_progress", [True, False])
def test_multiple_points_query(ept2_model, stream, print_progress):
    """Test get_forecasts with multiple points."""
    init_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=None)
    requested_points = [
        LatLon(lat=50.0, lon=8.0),
        LatLon(lat=50.5, lon=8.5),
    ]

    # Create mock data for 2 points, 4 hours (0-3)
    df = create_point_dataframe(
        num_points=2,
        num_hours=3,
        init_time=init_time,
        lat_start=50.0,
        lon_start=8.0,
    )
    mock_response = create_mock_arrow_response(df)

    # Mock the API call
    with patch.object(
        ept2_model._query_engine._api, "post", return_value=mock_response
    ):
        result = ept2_model.get_forecasts(
            points=requested_points,
            max_lead_time=3,
            variables=[
                Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
                Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            ],
            stream=stream,
            print_progress=print_progress,
        )

    # Convert to xarray for validation
    ds = result.to_xarray()

    # Verify dimensions
    assert "points" in ds.dims
    assert "init_time" in ds.dims
    assert "prediction_timedelta" in ds.dims

    # Verify shapes
    assert ds.sizes["points"] == 2
    assert ds.sizes["init_time"] == 1
    assert ds.sizes["prediction_timedelta"] == 4  # 0-3 hours inclusive

    # Verify both points are present
    assert len(ds.latitude.values) == 2
    assert len(ds.longitude.values) == 2

    # Verify coordinate mappings
    assert "requested_lat" in ds.coords
    assert "requested_lon" in ds.coords

    # Verify variables exist with correct shapes
    temp_data = ds["air_temperature_at_height_level_2m"].values
    assert temp_data.shape == (2, 1, 4)  # (points, init_time, prediction_timedelta)

    wind_data = ds["wind_speed_at_height_level_10m"].values
    assert wind_data.shape == (2, 1, 4)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("print_progress", [True, False])
def test_grid_slice_query(ept2_model, stream, print_progress):
    """Test get_forecasts with a grid slice."""
    init_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=None)

    # Create mock data for a 5x5 grid (50.0-52.0 lat, 8.0-10.0 lon), 3 hours (0-2)
    df = create_slice_dataframe(
        lat_range=(50.0, 52.0),
        lon_range=(8.0, 10.0),
        num_hours=2,
        init_time=init_time,
        lat_step=0.5,
        lon_step=0.5,
    )
    mock_response = create_mock_arrow_response(df)

    # Mock the API call
    with patch.object(
        ept2_model._query_engine._api, "post", return_value=mock_response
    ):
        result = ept2_model.get_forecasts(
            latitude=slice(50.0, 52.0),
            longitude=slice(8.0, 10.0),
            max_lead_time=2,
            variables=[
                Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
                Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            ],
            stream=stream,
            print_progress=print_progress,
        )

    # Convert to xarray for validation
    ds = result.to_xarray()

    # Verify dimensions (slice queries use lat/lon, not points)
    assert "latitude" in ds.dims
    assert "longitude" in ds.dims
    assert "init_time" in ds.dims
    assert "prediction_timedelta" in ds.dims

    # Verify grid structure
    assert ds.sizes["latitude"] == 5  # 50.0, 50.5, 51.0, 51.5, 52.0
    assert ds.sizes["longitude"] == 5  # 8.0, 8.5, 9.0, 9.5, 10.0
    assert ds.sizes["init_time"] == 1
    assert ds.sizes["prediction_timedelta"] == 3  # 0-2 hours inclusive

    # Verify coordinates
    assert ds.latitude.min() == 50.0
    assert ds.latitude.max() == 52.0
    assert ds.longitude.min() == 8.0
    assert ds.longitude.max() == 10.0

    # Verify variables exist with correct shapes
    temp_data = ds["air_temperature_at_height_level_2m"].values
    assert temp_data.shape == (1, 3, 5, 5)  # (init_time, timedelta, lat, lon)

    wind_data = ds["wind_speed_at_height_level_10m"].values
    assert wind_data.shape == (1, 3, 5, 5)

    # Verify data has spatial variation
    temp_flat = temp_data.flatten()
    assert temp_flat.std() > 0.1  # Should have some variation


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("print_progress", [True, False])
def test_latest_init_time_query(ept2_model, stream, print_progress):
    """Test get_forecasts with init_time='latest'."""
    # Use current time as the "latest" init time
    init_time = datetime.now(tz=None).replace(microsecond=0)
    requested_point = LatLon(lat=50.0, lon=8.0)

    # Create mock data for 1 point, 3 hours (0-2)
    df = create_point_dataframe(
        num_points=1,
        num_hours=2,
        init_time=init_time,
        lat_start=50.0,
        lon_start=8.0,
    )
    mock_response = create_mock_arrow_response(df)

    # Mock the API call
    with patch.object(
        ept2_model._query_engine._api, "post", return_value=mock_response
    ):
        result = ept2_model.get_forecasts(
            init_time="latest",
            points=requested_point,
            max_lead_time=2,
            variables=[
                Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
                Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            ],
            stream=stream,
            print_progress=print_progress,
        )

    # Convert to xarray for validation
    ds = result.to_xarray()

    # Verify dimensions
    assert ds.sizes["points"] == 1
    assert ds.sizes["init_time"] == 1
    assert ds.sizes["prediction_timedelta"] == 3  # 0-2 hours inclusive

    # Verify the init_time is properly set
    returned_init_time = pd.Timestamp(ds.init_time.values[0]).to_pydatetime()

    # The init_time should match what we provided in the mock data (without timezone)
    expected_init_time = init_time.replace(tzinfo=None)
    assert returned_init_time == expected_init_time

    # Verify variables exist
    assert "air_temperature_at_height_level_2m" in ds.data_vars
    assert "wind_speed_at_height_level_10m" in ds.data_vars
