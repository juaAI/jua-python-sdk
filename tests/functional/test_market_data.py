"""Functional tests for the market_data module.

These perform real API calls to verify zone-addressed market data retrieval
against live data.
"""

from datetime import datetime, timedelta, timezone

import pytest

from jua import JuaClient

pytestmark = pytest.mark.functional

_UNIFIED_COLUMNS = ["time", "market_zone", "variable", "value", "unit"]


@pytest.fixture
def md():
    return JuaClient().market_data


class TestMetadata:
    def test_get_zones(self, md):
        zones = md.get_zones()
        assert isinstance(zones, list)
        assert "DE" in zones
        assert "GB" in zones

    def test_get_variables(self, md):
        variables = md.get_variables()
        assert "solar" in variables
        assert "wind" in variables
        assert "day_ahead_prices" in variables

    def test_get_variables_for_zone(self, md):
        gb_vars = md.get_variables(market_zone="GB")
        assert "solar" in gb_vars
        assert "day_ahead_prices" in gb_vars


class TestGetData:
    def _window(self):
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=3)
        return start, end

    def test_de_renewables(self, md):
        start, end = self._window()
        df = md.get_data(
            market_zone="DE",
            variables=["solar", "wind"],
            start_time=start,
            end_time=end,
        )
        assert df.columns.tolist() == _UNIFIED_COLUMNS
        assert not df.empty
        assert set(df["variable"]) <= {"solar", "wind"}
        assert set(df["market_zone"]) == {"DE"}

    def test_de_day_ahead_prices(self, md):
        start, end = self._window()
        df = md.get_data(
            market_zone="DE",
            variables=["day_ahead_prices"],
            start_time=start,
            end_time=end,
        )
        assert not df.empty
        assert set(df["unit"]) == {"EUR/MWh"}

    def test_gb_renewables_via_uk_power(self, md):
        start, end = self._window()
        df = md.get_data(
            market_zone="GB",
            variables=["solar", "wind"],
            start_time=start,
            end_time=end,
        )
        assert not df.empty
        assert set(df["market_zone"]) == {"GB"}

    def test_gb_day_ahead_prices_via_entsoe(self, md):
        """GB prices are served from ENTSOE's GB zone (uk-power has none)."""
        start, end = self._window()
        df = md.get_data(
            market_zone="GB",
            variables=["day_ahead_prices"],
            start_time=start,
            end_time=end,
        )
        if df.empty:
            pytest.skip("ENTSOE GB day-ahead prices unavailable for this window")
        assert set(df["variable"]) == {"day_ahead_prices"}

    def test_time_zone_returns_tz_aware(self, md):
        import pandas as pd

        start, end = self._window()
        df = md.get_data(
            market_zone="DE",
            variables=["solar"],
            start_time=start,
            end_time=end,
            time_zone="Europe/Berlin",
        )
        if df.empty:
            pytest.skip("No solar data for this window")
        assert isinstance(df["time"].dtype, pd.DatetimeTZDtype)

    def test_multi_zone(self, md):
        start, end = self._window()
        df = md.get_data(
            market_zone=["DE", "GB"],
            variables=["solar"],
            start_time=start,
            end_time=end,
        )
        assert not df.empty
        assert set(df["market_zone"]) <= {"DE", "GB"}
