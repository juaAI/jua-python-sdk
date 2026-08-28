from datetime import UTC, datetime

import pytest

from jua import JuaClient
from jua.reanalysis import Reanalysis, ReanalysisModels
from jua.types.geo import LatLon
from jua.weather.variables import Variables

# The variables ERA5 serves, from jua-core
# libraries/jua-query-v2/src/jua_query_v2/configs/reanalysis/models/arco_era5.yaml.
# Every one must exist in the Variables enum or the API's own variable names come
# back as strings a user cannot discover from the SDK.
_ERA5_VARIABLES = [
    "air_temperature_at_height_level_2m",
    "surface_temperature",
    "dew_point_temperature_at_height_level_2m",
    "sea_surface_temperature",
    "air_pressure_at_mean_sea_level",
    "surface_air_pressure",
    "wind_speed_at_height_level_10m",
    "wind_direction_at_height_level_10m",
    "wind_speed_at_height_level_100m",
    "wind_direction_at_height_level_100m",
    "wind_speed_of_gust_at_height_level_10m_max",
    "surface_downwelling_shortwave_flux_sum_1h",
    "surface_net_downward_shortwave_flux_sum_1h",
    "surface_direct_downwelling_shortwave_flux_sum_1h",
    "cloud_area_fraction_at_entire_atmosphere",
    "cloud_area_fraction_at_entire_atmosphere_high_type",
    "cloud_area_fraction_at_entire_atmosphere_medium_type",
    "cloud_area_fraction_at_entire_atmosphere_low_type",
    "precipitation_amount_sum_1h",
    "atmosphere_convective_available_potential_energy",
    "predominant_precipitation_type_at_surface",
]


@pytest.mark.parametrize("variable_name", _ERA5_VARIABLES)
def test_every_era5_variable_exists_in_variables_enum(variable_name):
    known = {v.value.name for v in Variables}
    assert variable_name in known, (
        f"{variable_name} is served by ERA5 but missing from the Variables enum"
    )


def test_client_exposes_reanalysis():
    client = JuaClient()
    assert isinstance(client.reanalysis, Reanalysis)


def test_client_reanalysis_is_cached():
    client = JuaClient()
    assert client.reanalysis is client.reanalysis


def test_era5_enum_value_matches_api_name():
    # Must equal the `name:` field of arco_era5.yaml in jua-core.
    assert ReanalysisModels.ERA5.value == "arco_era5"


def test_get_data_rejects_points_and_latlon_together():
    client = JuaClient()
    with pytest.raises(ValueError, match="Cannot provide both points"):
        client.reanalysis.get_data(
            time=datetime(2024, 1, 15, tzinfo=UTC),
            points=LatLon(lat=47.0, lon=8.0),
            latitude=47.0,
            longitude=8.0,
        )


def test_get_data_rejects_oversized_request():
    """A global grid over a year is far past the server's row ceiling."""
    client = JuaClient()
    with pytest.raises(ValueError, match="too large for a single call"):
        client.reanalysis.get_data(
            time=slice(
                datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)
            ),
            variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
            latitude=slice(90, -90),
            longitude=slice(-180, 180),
        )


def test_get_data_rejects_latest_with_an_explanation():
    """`time="latest"` is the most likely wrong input, since the HTTP API
    documents it. The error must say why rather than raising a bare type error.
    """
    client = JuaClient()
    with pytest.raises(ValueError, match='no "latest" selector'):
        client.reanalysis.get_data(
            time="latest",
            variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
            points=LatLon(lat=47.0, lon=8.0),
        )


def test_get_data_requires_a_location():
    client = JuaClient()
    with pytest.raises(ValueError, match="Either both"):
        client.reanalysis.get_data(
            time=datetime(2024, 1, 15, tzinfo=UTC),
            variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
        )
