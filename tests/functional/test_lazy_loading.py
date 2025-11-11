"""Functional tests for lazy-loading

Checks that data loaded via lazy-loading is the same as the one loaded directly.

Note: These tests are excluded from regular CI runs (push/PR) and only run on PR
approval due to API costs and execution time.
"""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from jua import JuaClient
from jua.weather import Models
from jua.weather._query_engine import QueryEngine
from jua.weather.variables import Variables

# Mark all tests in this module as functional
pytestmark = pytest.mark.functional


@pytest.fixture
def client() -> JuaClient:
    """Create a JuaClient instance for testing."""
    return JuaClient(request_credit_limit=1_000_000_000)


@pytest.fixture
def query_engine() -> QueryEngine:
    """Create a JuaClient instance for testing."""
    return QueryEngine(JuaClient(request_credit_limit=1_000_000_000))


@pytest.mark.parametrize(
    "model",
    [
        Models.EPT1_5,
        Models.EPT2,
        Models.EPT2_RR,
    ],
)
@pytest.mark.parametrize(
    "init_time",
    [
        [datetime(2025, 10, 7, 0)],  # list preserves the init_time dim
        slice(
            datetime(2025, 10, 2, 0),
            datetime(2025, 10, 2, 6),
        ),
    ],
)
@pytest.mark.parametrize("lazy_lat", [slice(-180, 180), slice(40, 60)])
@pytest.mark.parametrize("lazy_lon", [slice(-180, 180), slice(0, 10)])
def test_lazy_loading_small_grid(
    client: JuaClient,
    query_engine: QueryEngine,
    model: Models,
    init_time: datetime | slice,
    lazy_lat: slice,
    lazy_lon: slice,
):
    ds_lazy_loaded = None
    ds = None
    try:
        lazy_init_times = slice(datetime(2025, 10, 1, 0), datetime(2025, 10, 12, 0))
        prediction_timedelta = slice(0, 12)
        latitude = slice(42, 45)
        longitude = slice(5, 8)

        variables = [
            Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name,
            Variables.AIR_PRESSURE_AT_MEAN_SEA_LEVEL.name,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M.name,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_100M.name,
        ]

        ds_lazy = xr.open_dataset(
            model,
            engine="jua_query_engine",
            query_engine=query_engine,
            init_time=lazy_init_times,
            prediction_timedelta=prediction_timedelta,
            latitude=lazy_lat,
            longitude=lazy_lon,
            variables=variables,
        )
        ds_lazy_loaded = ds_lazy.sel(
            init_time=init_time,
            latitude=latitude,
            longitude=longitude,
        )

        forecast = client.weather.get_model(model).get_forecasts(
            init_time=init_time,
            latitude=latitude,
            longitude=longitude,
            prediction_timedelta=prediction_timedelta,
            variables=variables,
        )
        ds = forecast.to_xarray()

        print(ds_lazy_loaded.dims, ds.dims)
        print(ds.init_time.equals(ds_lazy_loaded.init_time))
        print(ds.prediction_timedelta.equals(ds_lazy_loaded.prediction_timedelta))
        print(ds.latitude.equals(ds_lazy_loaded.latitude))
        print(ds.longitude.equals(ds_lazy_loaded.longitude))
        for var in ds.data_vars:
            print(var, np.array_equal(ds[var].values, ds_lazy_loaded[var].values))

        xr.testing.assert_allclose(ds_lazy_loaded, ds, rtol=1e-05, atol=1e-08)

    except Exception as e:
        pytest.fail(
            f"Data does not match between direct and lazy loading "
            f"{model.value}, {lazy_lat}, {lazy_lon}: {e}"
            f"\n{ds_lazy_loaded}"
            f"\n{ds}"
        )


@pytest.mark.parametrize("model", [Models.EPT2_RR])
@pytest.mark.parametrize("init_time", [[datetime(2025, 10, 7, 0)]])
@pytest.mark.parametrize("lazy_lat", [slice(-90, 90)])
@pytest.mark.parametrize("lazy_lon", [slice(-180, 180)])
@pytest.mark.parametrize(
    "lat,lon",
    [
        (slice(-12, -14), slice(-10, -9)),
        (slice(-27, -25), slice(5, 9)),
        (slice(35, 36), slice(5, 15)),
    ],
)
def test_lazy_loading_grids(
    client: JuaClient,
    query_engine: QueryEngine,
    model: Models,
    init_time: datetime | slice,
    lazy_lat: slice,
    lazy_lon: slice,
    lat: slice,
    lon: slice,
):
    ds_lazy_loaded = None
    ds = None
    try:
        lazy_init_times = slice(datetime(2025, 10, 1, 0), datetime(2025, 10, 12, 0))
        prediction_timedelta = slice(0, 12)
        variables = [
            Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name,
            Variables.AIR_PRESSURE_AT_MEAN_SEA_LEVEL.name,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M.name,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_100M.name,
        ]

        ds_lazy = xr.open_dataset(
            model,
            engine="jua_query_engine",
            query_engine=query_engine,
            init_time=lazy_init_times,
            prediction_timedelta=prediction_timedelta,
            latitude=lazy_lat,
            longitude=lazy_lon,
            variables=variables,
        )
        ds_lazy_loaded = ds_lazy.sel(
            init_time=init_time,
            latitude=lat,
            longitude=lon,
        )

        forecast = client.weather.get_model(model).get_forecasts(
            init_time=init_time,
            latitude=lat,
            longitude=lon,
            prediction_timedelta=prediction_timedelta,
            variables=variables,
        )
        ds = forecast.to_xarray()

        print(lat)
        print(lon)
        print(ds_lazy_loaded.dims, ds.dims)
        print(ds.init_time.equals(ds_lazy_loaded.init_time))
        print(ds.prediction_timedelta.equals(ds_lazy_loaded.prediction_timedelta))
        print(ds.latitude.equals(ds_lazy_loaded.latitude))
        print(ds.longitude.equals(ds_lazy_loaded.longitude))
        for var in ds.data_vars:
            print(var, np.array_equal(ds[var].values, ds_lazy_loaded[var].values))

        xr.testing.assert_allclose(ds_lazy_loaded, ds, rtol=1e-05, atol=1e-08)

    except Exception as e:
        pytest.fail(
            f"Data does not match between direct and lazy loading "
            f"{model.value}, {lazy_lat}, {lazy_lon}: {e}"
            f"\n{ds_lazy_loaded}"
            f"\n{ds}"
        )


@pytest.mark.parametrize("init_time", [[datetime(2025, 10, 29, 0)]])
@pytest.mark.parametrize("lazy_lat", [slice(30, 72)])
@pytest.mark.parametrize("lazy_lon", [slice(-15, 50)])
def test_lazy_loading_small_grid_hrrr(
    client: JuaClient,
    query_engine: QueryEngine,
    init_time: datetime | slice,
    lazy_lat: slice,
    lazy_lon: slice,
):
    ds_lazy_loaded = None
    ds = None
    try:
        lazy_init_times = slice(datetime(2025, 10, 27, 0), datetime(2025, 10, 31, 0))
        prediction_timedelta = slice(0, 12)
        latitude = slice(42, 45)
        longitude = slice(5, 8)

        model = Models.EPT2_HRRR
        variables = [
            Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name,
            Variables.AIR_PRESSURE_AT_MEAN_SEA_LEVEL.name,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M.name,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_100M.name,
        ]

        ds_lazy = xr.open_dataset(
            model,
            engine="jua_query_engine",
            query_engine=query_engine,
            init_time=lazy_init_times,
            prediction_timedelta=prediction_timedelta,
            latitude=lazy_lat,
            longitude=lazy_lon,
            variables=variables,
        )
        ds_lazy_loaded = ds_lazy.sel(
            init_time=init_time,
            latitude=latitude,
            longitude=longitude,
        )

        forecast = client.weather.get_model(model).get_forecasts(
            init_time=init_time,
            latitude=latitude,
            longitude=longitude,
            prediction_timedelta=prediction_timedelta,
            variables=variables,
        )
        ds = forecast.to_xarray()

        print(ds_lazy_loaded.dims, ds.dims)
        print(ds.init_time.equals(ds_lazy_loaded.init_time))
        print(ds.prediction_timedelta.equals(ds_lazy_loaded.prediction_timedelta))
        print(ds.latitude.equals(ds_lazy_loaded.latitude))
        print(ds.longitude.equals(ds_lazy_loaded.longitude))
        for var in ds.data_vars:
            print(var, np.array_equal(ds[var].values, ds_lazy_loaded[var].values))

        xr.testing.assert_allclose(ds_lazy_loaded, ds, rtol=1e-05, atol=1e-08)

    except Exception as e:
        pytest.fail(
            f"Data does not match between direct and lazy loading "
            f"{model.value}, {lazy_lat}, {lazy_lon}: {e}"
            f"\n{ds_lazy_loaded}"
            f"\n{ds}"
        )


@pytest.mark.parametrize(
    "model",
    [
        Models.EPT1_5,
        Models.EPT2,
        Models.EPT2_RR,
    ],
)
@pytest.mark.parametrize("init_time", [[datetime(2025, 10, 8, 0)]])
@pytest.mark.parametrize("lazy_lat", [slice(0, 90)])
@pytest.mark.parametrize("lazy_lon", [slice(0, 180)])
def test_lazy_loading_multiple_regions(
    client: JuaClient,
    query_engine: QueryEngine,
    model: Models,
    init_time: datetime | slice,
    lazy_lat: slice,
    lazy_lon: slice,
):
    try:
        lazy_init_times = slice(datetime(2025, 10, 1, 0), datetime(2025, 10, 12, 0))
        prediction_timedelta = slice(0, 12)

        lat_r1 = slice(42, 45)
        lon_r1 = slice(5, 8)

        lat_r2 = slice(35, 38)
        lon_r2 = slice(2, 5)

        variables = [
            Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name,
            Variables.AIR_PRESSURE_AT_MEAN_SEA_LEVEL.name,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M.name,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_100M.name,
        ]
        ds_lazy = xr.open_dataset(
            model,
            engine="jua_query_engine",
            query_engine=query_engine,
            init_time=lazy_init_times,
            prediction_timedelta=prediction_timedelta,
            latitude=lazy_lat,
            longitude=lazy_lon,
            variables=variables,
        )

        for lats, lons in [(lat_r1, lon_r1), (lat_r2, lon_r2)]:
            ds_lazy_loaded = ds_lazy.sel(
                init_time=init_time,
                latitude=lats,
                longitude=lons,
            ).load()

            forecast = client.weather.get_model(model).get_forecasts(
                init_time=init_time,
                latitude=lats,
                longitude=lons,
                prediction_timedelta=prediction_timedelta,
                variables=variables,
            )
            ds = forecast.to_xarray()

            print(ds_lazy_loaded.dims, ds.dims)
            print(ds.init_time.equals(ds_lazy_loaded.init_time))
            print(ds.prediction_timedelta.equals(ds_lazy_loaded.prediction_timedelta))
            print(ds.latitude.equals(ds_lazy_loaded.latitude))
            print(ds.longitude.equals(ds_lazy_loaded.longitude))
            for var in ds.data_vars:
                print(var, np.array_equal(ds[var].values, ds_lazy_loaded[var].values))

            xr.testing.assert_allclose(ds_lazy_loaded, ds, rtol=1e-05, atol=1e-08)
    except Exception as e:
        pytest.fail(
            f"Data does not match between direct and lazy loading "
            f"{model.value}, {lazy_lat}, {lazy_lon}: {e}"
        )


def test_lazy_loading_all_ept2_variables_tiny_slice(
    client: JuaClient,
    query_engine: QueryEngine,
):
    """Ensure all variables from EPT2 metadata can be lazy-loaded on a tiny slice."""
    ds_lazy_loaded = None
    ds_direct = None
    try:
        model = Models.EPT2
        model_client = client.weather.get_model(model)
        metadata = model_client.get_metadata()
        # Use standardized variable names from metadata
        variables = [v.name for v in metadata.variables]

        # Small selections to keep data volume minimal
        lazy_init_times = slice(datetime(2025, 10, 1, 0), datetime(2025, 10, 12, 0))
        init_time = [datetime(2025, 10, 7, 0)]  # keep init_time dim
        prediction_timedelta = slice(0, 6)
        latitude = slice(42, 43)
        longitude = slice(5, 6)

        # Lazy-open wider domain but load a tiny slice
        ds_lazy = model_client.get_forecasts(
            init_time=lazy_init_times,
            latitude=latitude,
            longitude=longitude,
            prediction_timedelta=prediction_timedelta,
            variables=variables,
            lazy_load=True,
        ).to_xarray()
        ds_lazy_loaded = ds_lazy.sel(
            init_time=init_time,
            latitude=latitude,
            longitude=longitude,
        ).load()

        # Direct load for the same small slice
        ds = model_client.get_forecasts(
            init_time=init_time,
            latitude=latitude,
            longitude=longitude,
            prediction_timedelta=prediction_timedelta,
            variables=variables,
        ).to_xarray()

        # Ensure all variables are present and data matches
        print(ds_lazy_loaded.dims, ds.dims)
        print(ds.init_time.equals(ds_lazy_loaded.init_time))
        print(ds.prediction_timedelta.equals(ds_lazy_loaded.prediction_timedelta))
        print(ds.latitude.equals(ds_lazy_loaded.latitude))
        print(ds.longitude.equals(ds_lazy_loaded.longitude))
        for var in ds.data_vars:
            print(var, np.array_equal(ds[var].values, ds_lazy_loaded[var].values))

            xr.testing.assert_allclose(ds_lazy_loaded, ds, rtol=1e-05, atol=1e-08)
    except Exception as e:
        pytest.fail(
            f"Failed to lazy-load all EPT2 variables on tiny slice: {e}"
            f"\n{ds_lazy_loaded}"
            f"\n{ds_direct}"
        )
