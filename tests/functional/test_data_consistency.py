"""Functional data-consistency sanity checks against the live API.

These guard the highest-risk failure mode for downstream users: silently
comparing a forecast against the wrong data because of a time-zone, DST, or
stitching error. They hit the real API and are marked ``functional`` so they
run via ``just test-functional`` rather than the default unit run.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from jua import JuaClient

pytestmark = pytest.mark.functional


@pytest.fixture
def client() -> JuaClient:
    return JuaClient()


def _frame(ds) -> pd.DataFrame:
    if "value" not in ds:
        return pd.DataFrame()
    return ds.to_dataframe().reset_index().dropna(subset=["value"])


def test_power_forecast_timezone_invariant_values(client):
    """The same forecast in two zones must carry identical values per instant.

    A tz/DST bug that shifted the data would surface here as differing values
    once both series are aligned on their UTC instant.
    """
    pf = client.power_forecast
    kwargs = dict(
        zone_keys=["DE"],
        psr_types=["Solar", "Wind Onshore"],
        init_time="latest",
        max_prediction_timedelta=1440,
    )
    utc = _frame(pf.get_data(time_zone="UTC", **kwargs))
    berlin = _frame(pf.get_data(time_zone="Europe/Berlin", **kwargs))
    assert not utc.empty and not berlin.empty

    utc["utc"] = pd.to_datetime(utc["time"], utc=True)
    berlin["utc"] = pd.to_datetime(berlin["time"], utc=True)
    merged = utc.merge(
        berlin, on=["zone_key", "psr_type", "utc"], suffixes=("_u", "_b")
    )

    assert len(merged) == len(utc) == len(berlin)
    assert np.allclose(merged["value_u"], merged["value_b"])


def test_day_ahead_stitch_matches_direct_run(client):
    """Each stitched day-ahead value equals a direct query of its source run.

    For ``init_hour=18`` the forecast for valid day D is the D-1 18:00 run.
    Re-fetching that single run directly must reproduce the stitched values.
    """
    pf = client.power_forecast
    tz = ZoneInfo("Europe/Berlin")
    midnight = (
        datetime.now(timezone.utc)
        .astimezone(tz)
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )
    start = midnight - timedelta(days=3)

    stitched = _frame(
        pf.get_day_ahead_timeseries(
            zone_keys=["DE"],
            psr_types=["Solar"],
            init_hour=18,
            time_zone="Europe/Berlin",
            start_date=start,
            end_date=midnight,
        )
    )
    if stitched.empty:
        pytest.skip("no stitched day-ahead data available")

    stitched["utc"] = pd.to_datetime(stitched["time"], utc=True)
    probe_day = stitched["utc"].iloc[len(stitched) // 2].tz_convert(tz).date()
    init_local = datetime(
        probe_day.year, probe_day.month, probe_day.day, 18, tzinfo=tz
    ) - timedelta(days=1)

    direct = _frame(
        pf.get_data(
            zone_keys=["DE"],
            psr_types=["Solar"],
            init_time=init_local.astimezone(timezone.utc),
            max_prediction_timedelta=39 * 60,
            time_zone="Europe/Berlin",
        )
    )
    if direct.empty:
        pytest.skip("source run unavailable for probe day")
    direct["utc"] = pd.to_datetime(direct["time"], utc=True)

    lo = pd.Timestamp(probe_day, tz=tz).tz_convert("UTC")
    hi = lo + pd.Timedelta(days=1)
    merged = stitched.merge(
        direct, on=["zone_key", "psr_type", "utc"], suffixes=("_s", "_d")
    )
    merged = merged[(merged["utc"] >= lo) & (merged["utc"] < hi)]

    assert len(merged) > 0
    assert np.allclose(merged["value_s"], merged["value_d"])


def test_market_forecast_and_actual_share_time_grid(client):
    """Day-ahead forecast and actual must sit on the same timestamps.

    A misaligned grid would invite comparing each forecast point against the
    wrong actual.
    """
    md = client.market_data
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=2)

    fc = md.get_data(
        market_zone="DE",
        variables=["solar_forecast"],
        start_time=start,
        end_time=end,
        time_zone="Europe/Berlin",
    )
    actual = md.get_data(
        market_zone="DE",
        variables=["solar"],
        start_time=start,
        end_time=end,
        time_zone="Europe/Berlin",
    )
    assert not fc.empty and not actual.empty

    grid_fc = set(pd.to_datetime(fc["time"], utc=True))
    grid_actual = set(pd.to_datetime(actual["time"], utc=True))
    overlap = grid_fc & grid_actual
    assert len(overlap) >= 0.9 * min(len(grid_fc), len(grid_actual))


def test_market_solar_actual_is_physically_plausible(client):
    """Solar actual is non-negative and ~0 overnight (catches unit/label slips)."""
    md = client.market_data
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=2)

    df = md.get_data(
        market_zone="DE",
        variables=["solar"],
        start_time=start,
        end_time=end,
        time_zone="Europe/Berlin",
    )
    assert not df.empty

    df = df.copy()
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    night = df[(df["hour"] <= 2) | (df["hour"] >= 23)]["value"]
    midday = df[(df["hour"] >= 11) & (df["hour"] <= 14)]["value"]

    assert (df["value"] >= -1).all()
    assert night.max() < 1000
    assert midday.max() > night.max()
