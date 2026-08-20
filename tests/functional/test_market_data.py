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
        # GB exposes the transmission/embedded wind split (actual + forecast).
        assert "wind_transmission" in gb_vars
        assert "wind_embedded" in gb_vars
        assert "wind_transmission_forecast" in gb_vars
        assert "wind_embedded_forecast" in gb_vars
        # GB prices/load_forecast are not served and must not be advertised.
        assert "day_ahead_prices" not in gb_vars
        assert "load_forecast" not in gb_vars


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

    def test_gb_wind_split_totals_match_components(self, md):
        """The wind total and its components are served in one SDK call (the
        backend can't return them together, so the SDK splits the request) and
        the components must sum to the total at every timestamp."""
        start, end = self._window()
        df = md.get_data(
            market_zone="GB",
            variables=["wind", "wind_transmission", "wind_embedded"],
            start_time=start,
            end_time=end,
        )
        assert not df.empty
        assert set(df["variable"]) == {"wind", "wind_transmission", "wind_embedded"}

        wide = df.pivot_table(index="time", columns="variable", values="value")
        wide = wide.dropna(subset=["wind", "wind_transmission", "wind_embedded"])
        if wide.empty:
            pytest.skip("No overlapping wind data for this window")
        residual = (
            wide["wind"] - wide["wind_transmission"] - wide["wind_embedded"]
        ).abs()
        # Allow a small tolerance for rounding in the published feeds.
        assert residual.max() <= 1.0

    def test_gb_wind_forecast_split(self, md):
        """The day-ahead wind total and its forecast components also come back
        from a single SDK call despite the same backend batching limit."""
        start, end = self._window()
        df = md.get_data(
            market_zone="GB",
            variables=[
                "wind_forecast",
                "wind_transmission_forecast",
                "wind_embedded_forecast",
            ],
            start_time=start,
            end_time=end,
        )
        assert not df.empty
        assert set(df["variable"]) == {
            "wind_forecast",
            "wind_transmission_forecast",
            "wind_embedded_forecast",
        }

    def test_gb_prices_not_supported(self, md):
        """GB prices/load forecast are not served: raise a clear error."""
        start, end = self._window()
        for variable in [
            "day_ahead_prices",
            "imbalance_price_long",
            "imbalance_price_short",
            "load_forecast",
        ]:
            with pytest.raises(ValueError, match="not supported for zone 'GB'"):
                md.get_data(
                    market_zone="GB",
                    variables=[variable],
                    start_time=start,
                    end_time=end,
                )

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
