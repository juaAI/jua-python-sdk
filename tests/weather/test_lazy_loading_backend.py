"""Tests for lazy-loading xarray backend with bounding box chunking."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import indexing

from jua.weather._lazy_loading.backend import JuaQueryEngineArray
from jua.weather._lazy_loading.cache import ForecastCache
from jua.weather.models import Models
from jua.weather.variables import Variables

from .utils import MockQueryEngine

VARIABLES = [
    Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
    Variables.WIND_DIRECTION_AT_HEIGHT_LEVEL_10M,
    Variables.WIND_DIRECTION_AT_HEIGHT_LEVEL_100M,
]


@pytest.fixture
def mock_coordinates():
    """Create mock coordinate arrays for testing."""
    init_times = [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 1, 12, 0),
        datetime(2025, 1, 2, 0, 0),
    ]
    max_lead_time = 12
    prediction_timedeltas = list(range(max_lead_time + 1))
    latitudes = slice(38, 41)
    longitudes = slice(6, 8)
    variables = [v.name for v in VARIABLES]

    return {
        "init_times": init_times,
        "max_lead_time": max_lead_time,
        "prediction_timedeltas": prediction_timedeltas,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "variables": variables,
    }


@pytest.fixture
def mock_query_engine(mock_coordinates):
    """Create mock QueryEngine."""
    return MockQueryEngine(
        max_lead_time=48,
        latitudes=np.linspace(-90, 90, 2160, endpoint=False),
        longitudes=np.linspace(-180, 180, 4320, endpoint=False),
        variables=[v.name for v in VARIABLES],
    )


@pytest.fixture
def lazy_dataset(mock_query_engine, mock_coordinates):
    """Create lazy-loaded dataset using the ForecastCache backend."""
    return xr.open_dataset(
        Models.EPT2,
        engine="jua_query_engine",
        query_engine=mock_query_engine,
        init_time=mock_coordinates["init_times"],
        prediction_timedelta=slice(0, mock_coordinates["max_lead_time"]),
        latitude=slice(-90, 90),
        longitude=slice(-180, 180),
        variables=VARIABLES,
    )


class TestLazyLoadingIndexing:
    """Test various indexing operations on lazy-loaded dataset."""

    @pytest.mark.parametrize(
        "lat_slice,lon_slice",
        [
            (slice(38, 41), slice(6, 8)),
            (slice(35, 37), slice(6, 10)),
            (slice(30, 40), slice(5, 15)),
            (slice(38, 39), slice(6, 7)),
            (slice(38, 45), slice(-10, 10)),
        ],
    )
    def test_spatial_slices(
        self, lazy_dataset, mock_query_engine, lat_slice, lon_slice
    ):
        """Test accessing different spatial slice sizes and shapes."""
        # Get expected data directly from mock query engine
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=[datetime(2025, 1, 1, 0, 0)],
            variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name],
            prediction_timedelta=[0, 1],
            latitude=lat_slice,
            longitude=lon_slice,
        )
        expected_var = (
            expected_ds[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
        ).sel(init_time=datetime(2025, 1, 1, 0, 0))

        # Get data from lazy dataset
        lazy_ds = (
            lazy_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
            .sel(
                init_time=datetime(2025, 1, 1, 0, 0),
                latitude=lat_slice,
                longitude=lon_slice,
            )
            .isel(prediction_timedelta=[0, 1])
        )

        np.testing.assert_array_equal(
            lazy_ds.prediction_timedelta.values,
            expected_var.prediction_timedelta.values,
        )
        np.testing.assert_array_almost_equal(
            lazy_ds.latitude.values, expected_var.latitude.values
        )
        np.testing.assert_array_almost_equal(
            lazy_ds.longitude.values, expected_var.longitude.values
        )
        np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_isel_single_init_time(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test isel with single init_time index (using small spatial slice)."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = (
                lazy_dataset[var]
                .isel(init_time=0)
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
            )
            expected_var = expected_ds[var].isel(init_time=0)
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_isel_init_time_slice(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test isel with init_time slice (using small spatial slice)."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"][0:2],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = (
                lazy_dataset[var]
                .isel(init_time=slice(0, 2))
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
            )
            expected_var = expected_ds[var]
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_isel_latitude_longitude_slices(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test sel with latitude and longitude slices using coordinate values."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            # Use .sel() with coordinate slices from mock_coordinates
            lazy_ds = (
                lazy_dataset[var]
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
                .isel(latitude=slice(2, 8), longitude=slice(3, 10))
            )
            expected_var = expected_ds[var].isel(
                latitude=slice(2, 8), longitude=slice(3, 10)
            )
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_isel_prediction_timedelta_slice(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test isel with prediction_timedelta slice (with spatial constraint)."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"][1:4],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = (
                lazy_dataset[var]
                .isel(prediction_timedelta=slice(1, 4))
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
            )
            lazy_data = lazy_ds.values
            expected_data = expected_ds[var].values
            np.testing.assert_array_equal(
                lazy_ds.init_time.values, expected_ds.init_time.values
            )
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_ds.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_ds.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_ds.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_data, expected_data)

    def test_isel_multiple_dimensions(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test isel with multiple dimensions simultaneously."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = (
                lazy_dataset[var]
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
                .isel(
                    init_time=1,
                    prediction_timedelta=slice(0, 3),
                    latitude=slice(5, 10),
                    longitude=slice(2, 7),
                )
            )
            expected_var = expected_ds[var].isel(
                init_time=1,
                prediction_timedelta=slice(0, 3),
                latitude=slice(5, 10),
                longitude=slice(2, 7),
            )
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_sel_latitude_longitude(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test sel with latitude and longitude coordinate values.

        (with temporal constraint)
        """
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=slice(35.0, 38.0),
            longitude=slice(6.0, 10.0),
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = lazy_dataset[var].sel(
                init_time=mock_coordinates["init_times"],
                latitude=slice(35.0, 38.0),
                longitude=slice(6.0, 10.0),
            )
            expected_var = expected_ds[var]
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_sel_single_point(self, lazy_dataset, mock_query_engine, mock_coordinates):
        """Test sel with single coordinate point (with temporal constraint)."""
        # For single point, we need to query a small area around it
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=slice(35.0, 40.0),
            longitude=slice(6.0, 10.0),
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = lazy_dataset[var].sel(
                init_time=mock_coordinates["init_times"],
                latitude=36.0,
                longitude=8.0,
                method="nearest",
            )
            expected_var = expected_ds[var].sel(
                latitude=36.0, longitude=8.0, method="nearest"
            )
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                np.asarray(lazy_ds.latitude.values),
                np.asarray(expected_var.latitude.values),
            )
            np.testing.assert_array_almost_equal(
                np.asarray(lazy_ds.longitude.values),
                np.asarray(expected_var.longitude.values),
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_sel_init_time(self, lazy_dataset, mock_query_engine, mock_coordinates):
        """Test sel with init_time (with spatial constraint)."""
        target_time = datetime(2025, 1, 1, 12, 0)
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = (
                lazy_dataset[var]
                .sel(init_time=target_time)
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
            )
            expected_var = expected_ds[var].sel(init_time=target_time)
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_sel_prediction_timedelta(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test sel with prediction_timedelta (with spatial constraint)."""
        target_td = pd.Timedelta(hours=12)
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = (
                lazy_dataset[var]
                .sel(prediction_timedelta=target_td)
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
            )
            expected_var = expected_ds[var].sel(prediction_timedelta=target_td)
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_mixed_sel_isel(self, lazy_dataset, mock_query_engine, mock_coordinates):
        """Test mixed sel and isel operations."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=slice(35.0, 37.0),
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            # First isel, then sel with spatial constraints
            lazy_ds = (
                lazy_dataset[var]
                .isel(init_time=0)
                .sel(
                    latitude=slice(35.0, 37.0), longitude=mock_coordinates["longitudes"]
                )
            )
            expected_var = expected_ds[var].isel(init_time=0)
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_sequential_selections(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test sequential selection operations."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=slice(35.0, 38.0),
            longitude=slice(6.0, 9.0),
        )

        for var in [v.name for v in VARIABLES]:
            # Chain multiple selections
            lazy_subset = lazy_dataset[var].sel(latitude=slice(35.0, 38.0))
            lazy_subset = lazy_subset.sel(longitude=slice(6.0, 9.0))
            lazy_subset = lazy_subset.isel(init_time=0)
            expected_var = expected_ds[var].isel(init_time=0)
            np.testing.assert_array_equal(
                lazy_subset.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_subset.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_subset.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_subset.values, expected_var.values
            )

    def test_negative_indexing(self, lazy_dataset, mock_query_engine, mock_coordinates):
        """Test negative indexing with isel (with spatial constraint)."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in [v.name for v in VARIABLES]:
            lazy_ds = (
                lazy_dataset[var]
                .isel(init_time=-1)
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
            )
            expected_var = expected_ds[var].isel(init_time=-1)
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_array_indexing(self, lazy_dataset, mock_query_engine, mock_coordinates):
        """Test indexing with arrays (within spatial constraint)."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        indices = [0, 2, 4, 6]
        for var in [v.name for v in VARIABLES]:
            lazy_ds = (
                lazy_dataset[var]
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
                .isel(latitude=indices)
            )
            expected_var = expected_ds[var].isel(latitude=indices)
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_different_variables_same_selection(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test that different variables return different data for same selection.

        (with temporal constraint)
        """
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=slice(35.0, 37.0),
            longitude=slice(6.0, 8.0),
        )

        selection = {
            "init_time": mock_coordinates["init_times"],
            "latitude": slice(35.0, 37.0),
            "longitude": slice(6.0, 8.0),
        }
        for v in VARIABLES:
            lazy_da = lazy_dataset[v.name].sel(**selection)
            expected_da = expected_ds[v.name]
            np.testing.assert_array_equal(
                lazy_da.prediction_timedelta.values,
                expected_da.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_da.latitude.values, expected_da.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_da.longitude.values, expected_da.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_da.values, expected_da.values)

    def test_edge_coordinates(self, lazy_dataset, mock_query_engine, mock_coordinates):
        """Test selection at edge coordinates (within spatial constraint)."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in VARIABLES:
            # First latitude (within constrained region)
            lazy_ds = (
                lazy_dataset[var.name]
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
                .isel(latitude=0)
            )
            expected_var = expected_ds[var.name].isel(latitude=0)
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

            # Last latitude (within constrained region)
            lazy_ds = (
                lazy_dataset[var.name]
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
                .isel(latitude=-1)
            )
            expected_var = expected_ds[var.name].isel(latitude=-1)
            np.testing.assert_array_equal(
                lazy_ds.prediction_timedelta.values,
                expected_var.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.latitude.values, expected_var.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_ds.longitude.values, expected_var.longitude.values
            )
            np.testing.assert_array_almost_equal(lazy_ds.values, expected_var.values)

    def test_empty_slice(self, lazy_dataset, mock_query_engine, mock_coordinates):
        """Test empty slice selection (within spatial constraint)."""
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=mock_coordinates["latitudes"],
            longitude=mock_coordinates["longitudes"],
        )

        for var in VARIABLES:
            lazy_subset = (
                lazy_dataset[var.name]
                .sel(
                    latitude=mock_coordinates["latitudes"],
                    longitude=mock_coordinates["longitudes"],
                )
                .isel(latitude=slice(0, 0))
            )
            expected_subset = expected_ds[var.name].isel(latitude=slice(0, 0))
            np.testing.assert_array_equal(
                lazy_subset.prediction_timedelta.values,
                expected_subset.prediction_timedelta.values,
            )
            np.testing.assert_array_almost_equal(
                lazy_subset.latitude.values, expected_subset.latitude.values
            )
            np.testing.assert_array_almost_equal(
                lazy_subset.longitude.values, expected_subset.longitude.values
            )
            assert lazy_subset.values.shape == expected_subset.values.shape
            assert lazy_subset.values.shape[2] == 0  # latitude dimension is empty


class TestCaching:
    """Test caching behavior of lazy-loaded dataset."""

    def test_cache_reuse(self, lazy_dataset, mock_query_engine, mock_coordinates):
        """Test that subsequent accesses use cached data (with spatial slice)."""
        # Reset call count
        mock_query_engine.reset_call_counts()

        # First access - should trigger API call(s)
        data1 = (
            lazy_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
            .sel(
                latitude=mock_coordinates["latitudes"],
                longitude=mock_coordinates["longitudes"],
            )
            .values
        )
        first_call_count = mock_query_engine.get_forecast_call_count
        assert first_call_count > 0, "First access should trigger API calls"

        # Second access - should use cache (no new API calls)
        data2 = (
            lazy_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
            .sel(
                latitude=mock_coordinates["latitudes"],
                longitude=mock_coordinates["longitudes"],
            )
            .values
        )
        second_call_count = mock_query_engine.get_forecast_call_count
        assert second_call_count == first_call_count, (
            f"Second access should not trigger new API calls (was {first_call_count}, "
            f"now {second_call_count})"
        )

        # Data should be identical
        np.testing.assert_array_almost_equal(data1, data2)

    def test_different_variables_share_cache(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test different variables from same region share cached boxes.

        (with temporal constraint)
        """
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=mock_coordinates["variables"],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=slice(35.0, 37.0),
            longitude=slice(6.0, 8.0),
        )

        # Access first variable
        temp_data = (
            lazy_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
            .sel(
                init_time=mock_coordinates["init_times"],
                latitude=slice(35.0, 37.0),
                longitude=slice(6.0, 8.0),
            )
            .values
        )

        # Access second variable (should use cached bounding boxes)
        wind_data = (
            lazy_dataset[Variables.WIND_DIRECTION_AT_HEIGHT_LEVEL_10M.name]
            .sel(
                init_time=mock_coordinates["init_times"],
                latitude=slice(35.0, 37.0),
                longitude=slice(6.0, 8.0),
            )
            .values
        )

        # Verify correctness
        temp_expected = expected_ds[
            Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name
        ].values
        wind_expected = expected_ds[
            Variables.WIND_DIRECTION_AT_HEIGHT_LEVEL_10M.name
        ].values

        np.testing.assert_array_almost_equal(temp_data, temp_expected)
        np.testing.assert_array_almost_equal(wind_data, wind_expected)

    def test_overlapping_regions(
        self, lazy_dataset, mock_query_engine, mock_coordinates
    ):
        """Test accessing overlapping regions uses cache efficiently.

        (with temporal constraint)
        """
        expected_ds = mock_query_engine.get_forecast(
            model=Models.EPT2,
            init_time=mock_coordinates["init_times"],
            variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name],
            prediction_timedelta=mock_coordinates["prediction_timedeltas"],
            latitude=slice(36.0, 37.0),
            longitude=slice(7.0, 9.0),
        )

        # Reset call count
        mock_query_engine.reset_call_counts()

        # Access larger region
        _ = (
            lazy_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
            .sel(
                init_time=mock_coordinates["init_times"],
                latitude=slice(35.0, 38.0),
                longitude=slice(6.0, 10.0),
            )
            .values
        )
        first_call_count = mock_query_engine.get_forecast_call_count
        assert first_call_count > 0, "First access should trigger API calls"

        # Access subset (should use cached data - no new API calls)
        data2 = (
            lazy_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
            .sel(
                init_time=mock_coordinates["init_times"],
                latitude=slice(36.0, 37.0),
                longitude=slice(7.0, 9.0),
            )
            .values
        )
        second_call_count = mock_query_engine.get_forecast_call_count
        assert second_call_count == first_call_count, (
            f"Subset access should use cached data (was {first_call_count}, now "
            f"{second_call_count})"
        )

        # Verify subset is correct
        expected_subset = expected_ds[
            Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name
        ].values

        np.testing.assert_array_almost_equal(data2, expected_subset)

    def test_multiple_init_times_batched(self, mock_coordinates):
        """Test that multiple init_times are fetched in a single API call."""
        # Create mock query engine with actual coordinate arrays (not slices)
        test_lats = np.linspace(38, 41, 30)
        test_lons = np.linspace(6, 8, 20)
        mock_qe = MockQueryEngine(
            max_lead_time=48,
            latitudes=test_lats,
            longitudes=test_lons,
            variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name],
        )

        # Wrap load_raw_forecast to capture last call arguments
        original_load_raw = mock_qe.load_raw_forecast
        last_init_times = [None]

        def tracked_load_raw(*args, **kwargs):
            payload = args[0] if args else kwargs.get("payload")
            if payload and hasattr(payload, "init_time"):
                if hasattr(payload.init_time, "__iter__") and not isinstance(
                    payload.init_time, str
                ):
                    last_init_times[0] = list(payload.init_time)
            return original_load_raw(*args, **kwargs)

        mock_qe.load_raw_forecast = tracked_load_raw

        # Create cache
        init_times_array = np.array(
            [pd.Timestamp(t) for t in mock_coordinates["init_times"]],
            dtype="datetime64[ns]",
        )

        pred_td_array = np.array(
            [np.timedelta64(h, "h") for h in mock_coordinates["prediction_timedeltas"]]
        )

        cache = ForecastCache(
            query_engine=mock_qe,
            model=Models.EPT2,
            variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name],
            init_times=init_times_array,
            prediction_timedeltas=pred_td_array,
            latitudes=test_lats,
            longitudes=test_lons,
            increasing_lats=test_lats[1] > test_lats[0],
            increasing_lons=test_lons[1] > test_lons[0],
            original_kwargs={},
        )

        # Create dataset with lazy array
        dims = ("init_time", "prediction_timedelta", "latitude", "longitude")
        coords = {
            "init_time": init_times_array,
            "prediction_timedelta": pred_td_array,
            "latitude": test_lats.astype("float32"),
            "longitude": test_lons.astype("float32"),
        }

        backend_array = JuaQueryEngineArray(
            cache=cache, variable=Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name
        )
        lazy_data = indexing.LazilyIndexedArray(backend_array)

        lazy_dataset = xr.Dataset(
            data_vars={
                Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name: (dims, lazy_data)
            },
            coords=coords,
        )

        # Reset call count
        mock_qe.reset_call_counts()

        # Access small region for ALL init_times (all 3 in same spatial chunk)
        # Should batch all 3 init_times into a single API call
        data = (
            lazy_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
            .isel(latitude=slice(2, 4), longitude=slice(4, 6))
            .values
        )

        # Verify only 1 API call was made (batching all init_times)
        assert mock_qe.get_forecast_call_count == 1, (
            f"Expected 1 API call, got {mock_qe.get_forecast_call_count}"
        )

        # Verify all 3 init_times were fetched together
        assert len(last_init_times[0]) == 3, (
            f"Expected 3 init_times in call, got {len(last_init_times[0])}"
        )

        # Verify data shape includes all init_times
        assert data.shape[0] == 3, f"Expected shape[0]=3, got {data.shape[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
