"""Functional tests for forecasts across all available models.

These tests perform actual API calls to verify that forecast retrieval
works correctly for all models with real data.

Note: These tests are excluded from regular CI runs (push/PR) and only
run on PR approval due to API costs and execution time.
"""

from datetime import datetime

import pytest

from jua import JuaClient
from jua.types.geo import LatLon
from jua.weather import Models
from jua.weather._model_meta import get_model_meta_info

# Mark all tests in this module as functional
pytestmark = pytest.mark.functional

# Define test locations
ZURICH = LatLon(lat=47.3769, lon=8.5417, label="Zurich")
GENEVA = LatLon(lat=46.2044, lon=6.1432, label="Geneva")
AMSTERDAM = LatLon(lat=52.3676, lon=4.9041, label="Amsterdam")
ROTTERDAM = LatLon(lat=51.9225, lon=4.47917, label="Rotterdam")
LONDON = LatLon(lat=51.5074, lon=-0.1278, label="London")
MANCHESTER = LatLon(lat=53.4808, lon=-2.2426, label="Manchester")

# Specific forecast date
SPECIFIC_FORECAST_DATE = datetime(2025, 10, 20, 0, 0, 0)

ALL_MODELS = list(Models)
INTERNAL_MODELS = [m for m in Models if get_model_meta_info(m).has_grid_access]


def get_test_cities(model: Models) -> tuple[LatLon, LatLon]:
    """Get appropriate test cities based on the model's regional coverage.

    Args:
        model: The weather model being tested

    Returns:
        Tuple of (city1, city2) appropriate for the model's region
    """
    if model == Models.UKMO_UK_DETERMINISTIC_2KM:
        return MANCHESTER, LONDON
    elif model == Models.KNMI_HARMONIE_AROME_NETHERLANDS:
        return AMSTERDAM, ROTTERDAM
    else:
        # Default to Swiss cities for global/European models
        return ZURICH, GENEVA


@pytest.fixture
def client():
    """Create a JuaClient instance for testing."""
    return JuaClient()


@pytest.mark.parametrize("model", ALL_MODELS)
def test_get_metadata(client: JuaClient, model: Models):
    """Test retrieving the metadata for all models.

    Args:
        client: JuaClient instance
        model: Weather model to test
    """
    try:
        model = client.weather.get_model(model)
        _ = model.get_metadata()
    except Exception as e:
        pytest.fail(f"Failed to retrieve metadata for {model.value}: {e}")


@pytest.mark.parametrize("model", ALL_MODELS)
def test_latest_forecast(client: JuaClient, model: Models):
    """Test retrieving the latest forecast across all models.

    Uses region-appropriate cities based on the model:
    - UK models: Manchester
    - Netherlands models: Amsterdam
    - Other models: Zurich

    Args:
        client: JuaClient instance
        model: Weather model to test
    """
    try:
        # Get appropriate test city for this model
        city1, _ = get_test_cities(model)

        # Get the model instance
        model_instance = client.weather.get_model(model)

        # Get latest forecast for the first city
        forecast = model_instance.get_forecasts(
            init_time="latest",
            points=city1,
            max_lead_time=48,
        )

        # Convert to xarray for validation
        ds = forecast.to_xarray()

        # Basic validations
        assert ds is not None, f"No data returned for {model.value}"
        assert "points" in ds.dims or "latitude" in ds.dims, (
            f"Missing spatial dimensions for {model.value}"
        )
        assert "prediction_timedelta" in ds.dims, (
            f"Missing prediction_timedelta dimension for {model.value}"
        )

        # Verify we have data
        assert ds.sizes.get("points", 1) >= 1, f"No point data for {model.value}"

        print(
            f"✓ {model.value}: Successfully retrieved latest forecast for {city1.label}"
        )

    except Exception as e:
        city1, _ = get_test_cities(model)
        pytest.fail(
            f"Failed to retrieve latest forecast for {model.value}, {city1.label}: {e}"
        )


@pytest.mark.parametrize("model", INTERNAL_MODELS)
def test_specific_forecast(client: JuaClient, model: Models):
    """Test retrieving the 2025-09-29 forecast across all models.

    Uses region-appropriate cities based on the model:
    - UK models: London
    - Netherlands models: Rotterdam
    - Other models: Geneva

    Args:
        client: JuaClient instance
        model: Weather model to test
    """
    try:
        # Get appropriate test city for this model
        _, city2 = get_test_cities(model)

        # Get the model instance
        model_instance = client.weather.get_model(model)

        # Get specific forecast for the second city
        forecast = model_instance.get_forecasts(
            init_time=SPECIFIC_FORECAST_DATE,
            points=city2,
            max_lead_time=48,
        )

        # Convert to xarray for validation
        ds = forecast.to_xarray()

        # Basic validations
        assert ds is not None, f"No data returned for {model.value}"
        assert "points" in ds.dims or "latitude" in ds.dims, (
            f"Missing spatial dimensions for {model.value}"
        )
        assert "prediction_timedelta" in ds.dims, (
            f"Missing prediction_timedelta dimension for {model.value}"
        )
        assert "init_time" in ds.dims, f"Missing init_time dimension for {model.value}"

        # Verify we have data
        assert ds.sizes.get("points", 1) >= 1, f"No point data for {model.value}"

        print(
            f"✓ {model.value}: Successfully retrieved "
            f"{SPECIFIC_FORECAST_DATE.date()} forecast for {city2.label}"
        )

    except Exception as e:
        _, city2 = get_test_cities(model)
        pytest.fail(
            f"Failed to retrieve {SPECIFIC_FORECAST_DATE.date()} forecast "
            f"for {model.value} at {city2.label}: {str(e)}"
        )
