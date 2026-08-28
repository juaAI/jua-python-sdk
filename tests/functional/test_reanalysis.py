"""Functional tests for the reanalysis (ERA5) surface.

These perform real API calls. Excluded from regular CI runs; see the note in
test_forecasts.py.

The assertions here are the ones a pure unit test cannot make: that the API
still returns the column set the transform expects, and that `time` still
arrives as something we can turn into a naive millisecond index. Both are
untyped wire contracts — no OpenAPI schema describes the Arrow payload.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from jua import JuaClient
from jua.reanalysis import ReanalysisModels
from jua.types.geo import LatLon
from jua.weather import Variables

pytestmark = pytest.mark.functional

UTC = timezone.utc

ZURICH = LatLon(lat=47.3769, lon=8.5417, label="Zurich")
LONDON = LatLon(lat=51.5074, lon=-0.1278, label="London")

# Well inside the published range: ERA5 lags real time by roughly five days, and
# our backfill starts long before this.
REFERENCE_START = datetime(2024, 1, 15, 0, tzinfo=UTC)
REFERENCE_END = datetime(2024, 1, 15, 5, tzinfo=UTC)

TEMPERATURE = Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M


@pytest.fixture
def client() -> JuaClient:
    return JuaClient()


def test_get_metadata(client: JuaClient):
    meta = client.reanalysis.get_metadata()
    assert meta.name == ReanalysisModels.ERA5.value
    assert meta.temporal_resolution_minutes == 60
    assert TEMPERATURE.value.name in meta.variables


def test_every_served_variable_is_in_the_variables_enum(client: JuaClient):
    """Guards the same gap the forecast suite hits via /forecast/meta.

    If ERA5 starts serving a variable the SDK does not know, users cannot
    reference it through the Variables enum. Local tests cannot catch this.
    """
    known = {v.value.name for v in Variables}
    served = client.reanalysis.get_variables()
    assert served, "expected at least one variable"
    unknown = sorted(set(served) - known)
    assert not unknown, f"ERA5 serves variables missing from Variables: {unknown}"


def test_point_query_shape_and_coords(client: JuaClient):
    data = client.reanalysis.get_data(
        time=slice(REFERENCE_START, REFERENCE_END),
        variables=[TEMPERATURE],
        points=[ZURICH, LONDON],
    )
    ds = data.to_xarray()

    assert set(ds.sizes) == {"points", "time"}
    assert ds.sizes["points"] == 2
    assert ds.sizes["time"] == 6  # hourly, both endpoints inclusive

    # Both the requested point and the grid cell it snapped to are preserved.
    zurich = ds.sel(points=str(ZURICH))
    assert float(zurich["requested_lat"]) == pytest.approx(ZURICH.lat)
    assert float(zurich["latitude"]) == pytest.approx(47.5)

    values = ds[TEMPERATURE].values
    assert np.isfinite(values).all()
    # Plausible 2m temperatures in Kelvin.
    assert (values > 200).all() and (values < 330).all()


def test_time_is_naive_millisecond_utc(client: JuaClient):
    """The API sends timestamp[ms, tz=UTC]; the SDK contract is naive ms."""
    data = client.reanalysis.get_data(
        time=slice(REFERENCE_START, REFERENCE_END),
        variables=[TEMPERATURE],
        points=ZURICH,
    )
    time_index = data.to_xarray().time.to_index()
    assert time_index.tz is None
    assert str(data.to_xarray().time.dtype) == "datetime64[ms]"


def test_grid_query_returns_the_expected_cells(client: JuaClient):
    """A 0.5x0.5 degree box is 3x3 cells on the 0.25 degree grid."""
    data = client.reanalysis.get_data(
        time=REFERENCE_START,
        variables=[TEMPERATURE],
        latitude=slice(52.5, 52.0),
        longitude=slice(13.0, 13.5),
    )
    ds = data.to_xarray()

    assert set(ds.sizes) == {"time", "latitude", "longitude"}
    assert ds.sizes["latitude"] == 3
    assert ds.sizes["longitude"] == 3
    assert ds[TEMPERATURE].dtype == np.float32


def test_bilinear_point_query(client: JuaClient):
    data = client.reanalysis.get_data(
        time=REFERENCE_START,
        variables=[TEMPERATURE],
        points=ZURICH,
        method="bilinear",
    )
    ds = data.to_xarray()
    assert ds.sizes["points"] == 1
    # Bilinear reports the requested coordinate rather than a grid cell.
    assert float(ds["latitude"].item()) == pytest.approx(ZURICH.lat, abs=1e-3)


def test_multiple_variables_round_trip(client: JuaClient):
    variables = [
        TEMPERATURE,
        Variables.WIND_SPEED_AT_HEIGHT_LEVEL_100M,
        Variables.ATMOSPHERE_CONVECTIVE_AVAILABLE_POTENTIAL_ENERGY,
    ]
    data = client.reanalysis.get_data(
        time=REFERENCE_START, variables=variables, points=ZURICH
    )
    ds = data.to_xarray()
    for variable in variables:
        assert variable.value.name in ds.data_vars
