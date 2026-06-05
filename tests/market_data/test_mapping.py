"""Unit tests for the market_data capability matrix and routing."""

import pytest

from jua.market_data import _mapping
from jua.market_data._mapping import MarketBackend, MarketVariable


def test_supported_zones():
    zones = _mapping.supported_zones()
    assert zones == ["BE", "DE", "FR", "GB", "NL"]


def test_supported_variables_full_vocabulary():
    variables = _mapping.supported_variables()
    assert set(variables) == {v.value for v in MarketVariable}


def test_supported_variables_for_zone():
    gb_vars = _mapping.supported_variables(market_zone="GB")
    # GB exposes renewables + load actual + renewable day-ahead forecasts.
    # Prices and load_forecast are intentionally not advertised (not served).
    assert set(gb_vars) == {
        "solar",
        "wind",
        "load",
        "solar_forecast",
        "wind_forecast",
    }


def test_unsupported_zone_raises():
    with pytest.raises(ValueError, match="Unsupported market zone"):
        _mapping.supported_variables(market_zone="ZZ")


def test_zone_is_case_insensitive():
    assert _mapping.entsoe_zone("de") == "DE_LU"


def test_entsoe_zone_mapping():
    assert _mapping.entsoe_zone("DE") == "DE_LU"
    assert _mapping.entsoe_zone("FR") == "FR"
    assert _mapping.entsoe_zone("GB") == "GB"


def test_eu_zone_routes_to_entsoe():
    cap = _mapping.resolve("DE", "solar")
    assert cap.backend == MarketBackend.ENTSOE
    assert cap.entsoe is not None
    assert cap.entsoe.variable == "generation_actual"
    assert cap.entsoe.psr_types == ("Solar",)


def test_eu_wind_sums_onshore_and_offshore():
    cap = _mapping.resolve("DE", "wind")
    assert cap.entsoe is not None
    assert set(cap.entsoe.psr_types) == {"Wind Onshore", "Wind Offshore"}


def test_gb_renewables_route_to_uk_power():
    for variable in ["solar", "wind", "load", "solar_forecast", "wind_forecast"]:
        cap = _mapping.resolve("GB", variable)
        assert cap.backend == MarketBackend.UK_POWER, variable
        assert cap.uk_power_variable is not None


def test_gb_prices_and_load_forecast_not_supported():
    # These are valid vocabulary but not served for GB, so they must raise a
    # clear "not supported" error rather than silently returning empty data.
    for variable in [
        "day_ahead_prices",
        "imbalance_price_long",
        "imbalance_price_short",
        "load_forecast",
    ]:
        with pytest.raises(ValueError, match="not supported for zone 'GB'"):
            _mapping.resolve("GB", variable)


def test_de_imbalance_overrides_entsoe_zone_to_de():
    # DE imbalance prices are published under the "DE" control area, not the
    # "DE_LU" bidding zone used for the rest of Germany's data. Both direction
    # components carry the override and a single other_type filter.
    for variable, direction in [
        ("imbalance_price_long", "Long"),
        ("imbalance_price_short", "Short"),
    ]:
        cap = _mapping.resolve("DE", variable)
        assert cap.backend == MarketBackend.ENTSOE
        assert cap.entsoe is not None
        assert cap.entsoe.variable == "imbalance_prices"
        assert cap.entsoe.zone_override == "DE"
        assert cap.entsoe.other_type == direction
    # Everything else for DE still uses the default DE_LU zone (no override).
    assert _mapping.resolve("DE", "day_ahead_prices").entsoe.zone_override is None
    assert _mapping.entsoe_zone("DE") == "DE_LU"


def test_unknown_variable_raises():
    with pytest.raises(ValueError, match="Unknown market variable"):
        _mapping.resolve("DE", "not_a_variable")
