"""Functional tests for the power forecast module.

These tests perform actual API calls to verify that power forecast retrieval
works correctly with real data.
"""

import pytest

from jua import JuaClient

pytestmark = pytest.mark.functional


@pytest.fixture
def client():
    """Create a JuaClient instance for testing."""
    return JuaClient()


@pytest.fixture
def pf(client: JuaClient):
    """Create a PowerForecast instance for testing."""
    return client.power_forecast


class TestMetadata:
    """Tests for power forecast metadata endpoints."""

    def test_get_zones(self, pf):
        zones = pf.get_zones()

        assert isinstance(zones, list)
        assert len(zones) > 0
        assert all(isinstance(z, str) for z in zones)
        assert "DE" in zones

    def test_get_psr_types(self, pf):
        psr_types = pf.get_psr_types()

        assert isinstance(psr_types, list)
        assert len(psr_types) > 0
        assert "Solar" in psr_types
        assert "Wind Onshore" in psr_types

    def test_get_psr_types_filtered_by_zone(self, pf):
        psr_types = pf.get_psr_types(zone_key="DE")

        assert isinstance(psr_types, list)
        assert len(psr_types) > 0
        assert all(isinstance(p, str) for p in psr_types)

    def test_get_init_times(self, pf):
        from jua.power_forecast.power_forecast import InitTimeInfo

        init_times = pf.get_init_times()

        assert isinstance(init_times, list)
        assert len(init_times) > 0
        assert all(isinstance(it, InitTimeInfo) for it in init_times)
        assert all(it.max_prediction_timedelta > 0 for it in init_times)

    def test_get_init_times_filtered(self, pf):
        init_times = pf.get_init_times(zone_key="DE", psr_type="Solar")

        assert isinstance(init_times, list)
        assert len(init_times) > 0

    def test_get_init_times_multiple_psr_types(self, pf):
        """Filtering by multiple PSR types returns only init times
        where all requested types are complete (intersection)."""
        all_types = pf.get_psr_types(zone_key="DE")
        filtered = pf.get_init_times(zone_key="DE", psr_type=all_types)
        single = pf.get_init_times(zone_key="DE", psr_type=all_types[0])

        assert len(filtered) <= len(single), (
            "Intersection of multiple PSR types should not exceed a single type"
        )

    def test_get_init_times_time_window(self, pf):
        """Time-window mode returns init times bounded by start/end and is not
        capped at the 1000-item count limit."""
        from datetime import datetime, timedelta, timezone

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)

        init_times = pf.get_init_times(
            zone_key="DE", psr_type="Solar", start_time=start, end_time=end
        )

        assert isinstance(init_times, list)
        assert len(init_times) > 0

        def _as_utc(dt):
            # The endpoint may return naive datetimes; treat them as UTC.
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

        assert all(start <= _as_utc(it.init_time) < end for it in init_times)
        # A 30-day window exceeds the legacy 1000-item count cap.
        assert len(init_times) > 1000


class TestGetData:
    """Tests for power forecast data retrieval."""

    def test_get_data_single_zone(self, pf):
        ds = pf.get_data(
            zone_keys=["DE"],
            init_time="latest",
            max_prediction_timedelta=120,
        )

        assert ds is not None
        assert "value" in ds.data_vars
        assert "zone_key" in ds.dims
        assert "time" in ds.dims

    def test_get_data_with_psr_types(self, pf):
        ds = pf.get_data(
            zone_keys=["DE"],
            psr_types=["Solar", "Wind Onshore"],
            init_time="latest",
            max_prediction_timedelta=120,
        )

        assert ds is not None
        assert "psr_type" in ds.dims
        psr_values = set(ds.coords["psr_type"].values)
        assert psr_values == {"Solar", "Wind Onshore"}

    def test_get_data_multiple_zones(self, pf):
        ds = pf.get_data(
            zone_keys=["DE", "FR"],
            psr_types=["Solar"],
            init_time="latest",
            max_prediction_timedelta=120,
        )

        assert ds is not None
        zone_values = set(ds.coords["zone_key"].values)
        assert zone_values == {"DE", "FR"}

    def test_get_data_latest_resolves_all_psr_types(self, pf):
        """Using init_time='latest' with multiple PSR types should return
        data for every requested type (SDK-side resolution)."""
        psr_types = ["Solar", "Wind Onshore", "Wind Offshore"]
        ds = pf.get_data(
            zone_keys=["DE"],
            psr_types=psr_types,
            init_time="latest",
            max_prediction_timedelta=120,
        )

        assert ds is not None
        df = ds.to_dataframe().reset_index().dropna(subset=["value"])
        actual_psr = set(df["psr_type"].unique())
        assert actual_psr == set(psr_types), (
            f"Expected all PSR types {psr_types}, got {actual_psr}"
        )

    def test_get_data_latest_offset(self, pf):
        ds = pf.get_data(
            zone_keys=["DE"],
            init_time="latest-1",
            max_prediction_timedelta=120,
        )

        assert ds is not None
        assert ds.sizes["time"] > 0

    def test_get_data_integer_offset(self, pf):
        ds = pf.get_data(
            zone_keys=["DE"],
            init_time=0,
            max_prediction_timedelta=120,
        )

        assert ds is not None
        assert ds.sizes["time"] > 0

    def test_get_data_horizon_mode(self, pf):
        ds = pf.get_data(
            zone_keys=["DE"],
            init_time="latest",
            max_prediction_timedelta=240,
        )

        assert ds is not None
        assert ds.sizes["time"] > 0

    def test_get_data_time_range_mode(self, pf):
        from datetime import datetime, timedelta, timezone

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=6)

        ds = pf.get_data(
            zone_keys=["DE"],
            start_time=start,
            end_time=end,
        )

        assert ds is not None

    def test_get_data_mutually_exclusive_modes(self, pf):
        """Cannot combine horizon and time-range parameters."""
        from datetime import datetime, timezone

        with pytest.raises(ValueError):
            pf.get_data(
                zone_keys=["DE"],
                init_time="latest",
                max_prediction_timedelta=120,
                start_time=datetime.now(timezone.utc),
            )
