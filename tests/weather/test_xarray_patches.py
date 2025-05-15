import numpy as np
import pytest
import xarray as xr

import jua.weather._xarray_patches  # # noqa: F401

# Make sure the xarray patches are loaded
from jua.types.geo import LatLon
from jua.weather import Variables
from jua.weather.conversions import to_datetime, to_timedelta

_TIME = to_datetime("2025-05-02")
_TIMEDELTAS = np.array(to_timedelta(list(range(12))))
_LATITUDE = np.arange(-10, 10, 0.1)
_LONGITUDE = np.arange(-10, 10, 0.1)


@pytest.fixture
def mock_dataset():
    time = [_TIME]
    prediction_timedeltas = _TIMEDELTAS

    latitude = _LATITUDE
    longitude = _LONGITUDE

    return xr.Dataset(
        coords={
            "time": time,
            "prediction_timedelta": prediction_timedeltas,
            "latitude": latitude,
            "longitude": longitude,
        },
        data_vars={
            str(var): (
                ("time", "prediction_timedelta", "latitude", "longitude"),
                np.random.rand(
                    len(time), len(prediction_timedeltas), len(latitude), len(longitude)
                ),
            )
            for var in Variables
        },
    )


def test_flip_lat(mock_dataset: xr.Dataset):
    # By default, the lattitude slice must be (upper, lower)
    # but we added support for (lower, upper) too
    lat_slice = slice(-5, 5)

    # Selected dataset should not be empty
    selected = mock_dataset.sel(latitude=lat_slice)
    reference = mock_dataset.sel(latitude=slice(5, -5))

    assert selected.equals(reference)


def test_to_celcius(mock_dataset: xr.Dataset):
    data = mock_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
    assert data.to_celcius().equals(data - 273.15)


def test_to_absolute_time(mock_dataset: xr.Dataset):
    with_absolute_time = mock_dataset.to_absolute_time()

    assert hasattr(with_absolute_time, "absolute_time")
    assert "prediction_timedelta" not in with_absolute_time.dims
    absolute_time = with_absolute_time.absolute_time.values.flatten()
    reference = np.datetime64(_TIME) + _TIMEDELTAS
    assert np.equal(absolute_time, reference).all()


def test_select_single_point(mock_dataset: xr.Dataset):
    point = LatLon(lat=0, lon=0)
    selected = mock_dataset.jua.select_point(point=point, method="nearest")
    reference = mock_dataset.sel(latitude=0, longitude=0, method="nearest")
    air_temp_selected = selected[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
    air_temp_reference = reference[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
    assert np.equal(air_temp_selected.values, air_temp_reference.values).all()


def test_select_multiple_point(mock_dataset: xr.Dataset):
    """Testing that the order of the point is preserved"""
    point = [
        LatLon(lat=np.random.uniform(-10, 10), lon=np.random.uniform(-10, 10))
        for _ in range(10)
    ]
    selected = mock_dataset.sel(point=point, method="nearest")
    data = selected[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
    for i, p in enumerate(point):
        reference = mock_dataset[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M].sel(
            latitude=p.lat, longitude=p.lon, method="nearest"
        )
        assert np.allclose(data.isel(point=i).values, reference.values)
