"""Functional tests for market aggregates.

These tests perform actual API calls to verify that market aggregate retrieval
works correctly with real data.
"""

from datetime import datetime

import pytest

from jua import JuaClient
from jua.market_aggregates import AggregateVariables, ModelRuns
from jua.types import MarketZones
from jua.weather import Models

# Mark all tests in this module as functional
pytestmark = pytest.mark.functional

# Define test market zones
TEST_ZONES = {
    "germany": MarketZones.DE,
    "norway": MarketZones.NO_NO1,
    "australia": MarketZones.AU_NSW,
}

# Available aggregate variables
ALL_AGGREGATE_VARIABLES = list(AggregateVariables)

# Specific date for historical queries
HISTORICAL_DATE = datetime(2025, 10, 20, 0, 0, 0)


@pytest.fixture
def client():
    """Create a JuaClient instance for testing."""
    return JuaClient()


@pytest.mark.parametrize("variable", ALL_AGGREGATE_VARIABLES)
def test_latest_run_single_variable(client: JuaClient, variable: AggregateVariables):
    """Test retrieving the latest model run for each aggregate variable.

    Args:
        client: JuaClient instance
        variable: Aggregate variable to test
    """
    try:
        # Create market for Germany
        market = client.market_aggregates.get_market(market_zone=MarketZones.DE)

        # Get latest EPT2 run
        model_runs = [ModelRuns(Models.EPT2, 0)]

        ds = market.compare_runs(
            agg_variable=variable,
            model_runs=model_runs,
            max_lead_time=48,
        )

        # Basic validations
        assert ds is not None, f"No data returned for {variable.name}"

        # Check dimensions (only model_run and time are dimensions)
        assert "model_run" in ds.dims, (
            f"Missing model_run dimension for {variable.name}"
        )
        assert "time" in ds.dims, f"Missing time dimension for {variable.name}"

        # Check data variables
        assert "prediction_timedelta" in ds.data_vars, (
            f"Missing prediction_timedelta data variable for {variable.name}"
        )

        # Verify we have data and attributes
        assert ds.sizes["model_run"] == 1, (
            f"Expected 1 model run, got {ds.sizes['model_run']}"
        )
        assert "market_zone" in ds.attrs, (
            f"Missing market_zone attribute for {variable.name}"
        )
        assert "min_lead_time" in ds.attrs, (
            f"Missing min_lead_time attribute for {variable.name}"
        )
        assert "max_lead_time" in ds.attrs, (
            f"Missing max_lead_time attribute for {variable.name}"
        )

        # Check that the variable data exists
        assert variable.variable.name in ds.data_vars, (
            f"Variable {variable.variable.name} not found in dataset"
        )

        print(f"✓ {variable.name}: Successfully retrieved latest run")

    except Exception as e:
        pytest.fail(f"Failed to retrieve latest run for {variable.name}: {e}")


def test_compare_multiple_models(client: JuaClient):
    """Test comparing multiple models for the same variable.

    Args:
        client: JuaClient instance
    """
    try:
        market = client.market_aggregates.get_market(market_zone=MarketZones.DE)

        # Compare EPT2 and ECMWF IFS
        model_runs = [
            ModelRuns(Models.EPT2, 0),
            ModelRuns(Models.ECMWF_IFS_SINGLE, 0),
        ]

        ds = market.compare_runs(
            agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            model_runs=model_runs,
            max_lead_time=48,
        )

        # Verify we have data for both models
        assert ds is not None
        assert ds.sizes["model_run"] == 2, (
            f"Expected 2 model runs, got {ds.sizes['model_run']}"
        )

        # Check model run coordinates
        model_run_values = ds.coords["model_run"].values
        assert len(model_run_values) == 2

        print("✓ Successfully compared multiple models")
        print(f"  Model runs: {list(model_run_values)}")

    except Exception as e:
        pytest.fail(f"Failed to compare multiple models: {e}")


def test_multiple_model_runs_same_model(client: JuaClient):
    """Test retrieving multiple runs from the same model.

    Args:
        client: JuaClient instance
    """
    try:
        market = client.market_aggregates.get_market(market_zone=MarketZones.GB)

        # Get latest 3 EPT2 runs
        model_runs = [ModelRuns(Models.EPT2, [0, 1, 2])]

        ds = market.compare_runs(
            agg_variable=AggregateVariables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
            model_runs=model_runs,
            max_lead_time=24,
        )

        # Verify we have 3 model runs
        assert ds is not None
        assert ds.sizes["model_run"] == 3, (
            f"Expected 3 model runs, got {ds.sizes['model_run']}"
        )

        # Check that we have init_time coordinate
        assert "init_time" in ds.coords, "Missing init_time coordinate"

        print("✓ Successfully retrieved multiple runs from same model")
        print(f"  Number of runs: {ds.sizes['model_run']}")
        print(f"  Init times: {ds.coords['init_time'].values}")

    except Exception as e:
        pytest.fail(f"Failed to retrieve multiple runs from same model: {e}")


def test_historical_model_runs(client: JuaClient):
    """Test retrieving historical model runs using specific datetimes.

    Args:
        client: JuaClient instance
    """
    try:
        market = client.market_aggregates.get_market(market_zone=MarketZones.DE)

        # Define specific historical initialization times
        historical_init_times = [
            datetime(2025, 10, 20, 0),
            datetime(2025, 10, 20, 6),
            datetime(2025, 10, 20, 12),
        ]

        model_runs = [ModelRuns(Models.EPT2, historical_init_times)]

        ds = market.compare_runs(
            agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            model_runs=model_runs,
            max_lead_time=48,
        )

        # Verify we have the expected number of runs
        assert ds is not None
        assert ds.sizes["model_run"] == len(historical_init_times), (
            f"Expected {len(historical_init_times)} model runs, "
            f"got {ds.sizes['model_run']}"
        )

        # Check that we have the correct init times
        assert "init_time" in ds.coords, "Missing init_time coordinate"

        print("✓ Successfully retrieved historical model runs")
        print(f"  Init times: {ds.coords['init_time'].values}")

    except Exception as e:
        pytest.fail(f"Failed to retrieve historical model runs: {e}")


def test_multiple_zones_aggregation(client: JuaClient):
    """Test retrieving data for multiple market zones simultaneously.

    Args:
        client: JuaClient instance
    """
    try:
        # Create market with multiple European zones
        zones = [MarketZones.DE, MarketZones.FR, MarketZones.NL, MarketZones.BE]
        market = client.market_aggregates.get_market(market_zone=zones)

        model_runs = [ModelRuns(Models.EPT2, 0)]

        ds = market.compare_runs(
            agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            model_runs=model_runs,
            max_lead_time=24,
        )

        # Verify we have data for all zones
        assert ds is not None

        # market_zone is an attribute (metadata), not a data variable
        assert "market_zone" in ds.attrs, "Missing market_zone attribute"

        # Check that the requested zones are in the attribute
        attr_zones = set(ds.attrs["market_zone"])
        expected_zones = {z.zone_name for z in zones}
        assert attr_zones == expected_zones, (
            f"Zone mismatch. Expected: {expected_zones}, Got: {attr_zones}"
        )

        print("✓ Successfully retrieved data for multiple zones")
        print(f"  Zones: {sorted(attr_zones)}")

    except Exception as e:
        pytest.fail(f"Failed to retrieve data for multiple zones: {e}")


def test_lead_time_filtering(client: JuaClient):
    """Test filtering by minimum and maximum lead time.

    Args:
        client: JuaClient instance
    """
    try:
        market = client.market_aggregates.get_market(market_zone=MarketZones.DE)
        model_runs = [ModelRuns(Models.EPT2, 0)]

        # Test with both min and max lead time
        ds = market.compare_runs(
            agg_variable=AggregateVariables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
            model_runs=model_runs,
            min_lead_time=12,
            max_lead_time=36,
        )

        # Verify we have data
        assert ds is not None

        # prediction_timedelta is a data variable, not a dimension
        assert "prediction_timedelta" in ds.data_vars, (
            "Missing prediction_timedelta data variable"
        )

        # Check prediction timedeltas are within expected range
        timedeltas = ds["prediction_timedelta"].values.flatten()
        # Convert to hours for comparison
        min_hours = min(td.astype("timedelta64[h]").astype(int) for td in timedeltas)
        max_hours = max(td.astype("timedelta64[h]").astype(int) for td in timedeltas)

        assert min_hours >= 12, f"Minimum lead time {min_hours}h is less than 12h"
        assert max_hours <= 36, f"Maximum lead time {max_hours}h is greater than 36h"

        print("✓ Successfully filtered by lead time")
        print(f"  Lead time range: {min_hours}h - {max_hours}h")

    except Exception as e:
        pytest.fail(f"Failed to filter by lead time: {e}")


def test_variable_weighting_types(client: JuaClient):
    """Test that different variables use appropriate weighting types.

    Args:
        client: JuaClient instance
    """
    try:
        from jua.market_aggregates.variables import Weighting

        # Verify weighting types are correctly assigned
        assert (
            AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M.weighting
            == Weighting.WIND_CAPACITY
        )
        assert (
            AggregateVariables.SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H.weighting
            == Weighting.SOLAR_CAPACITY
        )
        assert (
            AggregateVariables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.weighting
            == Weighting.POPULATION
        )

        print("✓ All variables have correct weighting types")

        # Test that we can actually retrieve data with different weighting types
        market = client.market_aggregates.get_market(market_zone=MarketZones.DE)
        model_runs = [ModelRuns(Models.EPT2, 0)]

        # Test wind (wind capacity weighting)
        ds_wind = market.compare_runs(
            agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            model_runs=model_runs,
            max_lead_time=24,
        )
        assert ds_wind is not None

        # Test temperature (population weighting)
        ds_temp = market.compare_runs(
            agg_variable=AggregateVariables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
            model_runs=model_runs,
            max_lead_time=24,
        )
        assert ds_temp is not None

        # Test solar (solar capacity weighting)
        ds_solar = market.compare_runs(
            agg_variable=AggregateVariables.SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H,
            model_runs=model_runs,
            max_lead_time=24,
        )
        assert ds_solar is not None

        print("✓ Successfully retrieved data with all weighting types")

    except Exception as e:
        pytest.fail(f"Failed to test variable weighting types: {e}")


def test_dataset_structure(client: JuaClient):
    """Test the structure of the returned xarray dataset.

    Args:
        client: JuaClient instance
    """
    try:
        market = client.market_aggregates.get_market(market_zone=MarketZones.DE)
        model_runs = [ModelRuns(Models.EPT2, [0, 1])]

        ds = market.compare_runs(
            agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            model_runs=model_runs,
            max_lead_time=24,
        )

        # Check required dimensions (only model_run and time)
        required_dims = ["model_run", "time"]
        for dim in required_dims:
            assert dim in ds.dims, f"Missing required dimension: {dim}"

        # Check required coordinates (model and init_time are coords on model_run)
        assert "model" in ds.coords, "Missing model coordinate"
        assert "init_time" in ds.coords, "Missing init_time coordinate"

        # Check required data variables
        assert "prediction_timedelta" in ds.data_vars, (
            "Missing prediction_timedelta data variable"
        )

        # Check required attributes
        assert "market_zone" in ds.attrs, "Missing market_zone attribute"
        assert "min_lead_time" in ds.attrs, "Missing min_lead_time attribute"
        assert "max_lead_time" in ds.attrs, "Missing max_lead_time attribute"

        # Check that the aggregate variable exists
        var_name = AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M.variable.name
        assert var_name in ds.data_vars, f"Missing data variable: {var_name}"

        # Check data variable has correct dimensions (model_run and time)
        data_var = ds[var_name]
        for dim in required_dims:
            assert dim in data_var.dims, f"Data variable missing dimension: {dim}"

        print("✓ Dataset structure is valid")
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Coordinates: {list(ds.coords.keys())}")
        print(f"  Data variables: {list(ds.data_vars.keys())}")

    except Exception as e:
        pytest.fail(f"Failed to validate dataset structure: {e}")
