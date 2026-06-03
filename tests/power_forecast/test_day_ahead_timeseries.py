from datetime import datetime, timedelta, timezone

import pandas as pd
import xarray as xr

from jua import JuaClient
from jua.power_forecast.power_forecast import InitTimeInfo


def _make_ds(zone: str, psr: str, init_times: list[datetime]) -> xr.Dataset:
    """Create a simple dataset with 40 hours of horizon per init."""
    rows = []
    for i, it in enumerate(init_times):
        for h in range(0, 40):  # 0..39h
            rows.append(
                {
                    "zone_key": zone,
                    "psr_type": psr,
                    "init_time": pd.Timestamp(it),
                    "time": pd.Timestamp(it + timedelta(hours=h)),
                    "value": float(i * 1000 + h),
                }
            )
    df = pd.DataFrame(rows)
    return xr.Dataset.from_dataframe(
        df.set_index(["zone_key", "psr_type", "init_time", "time"])
    )


def test_get_day_ahead_timeseries_stitches_across_days(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast

    zone = "GB"
    psr = "Solar"
    t1 = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, 9, 0, tzinfo=timezone.utc)
    init_infos = [
        InitTimeInfo(init_time=t1, max_prediction_timedelta=40 * 60),
        InitTimeInfo(init_time=t2, max_prediction_timedelta=40 * 60),
    ]

    # Patch network methods
    monkeypatch.setattr(
        pf, "get_init_times", lambda zone_key=None, psr_type=None, limit=96: init_infos
    )
    monkeypatch.setattr(pf, "get_data", lambda **kwargs: _make_ds(zone, psr, [t1, t2]))

    stitched = pf.get_day_ahead_timeseries(
        zone_keys=[zone],
        psr_types=[psr],
        init_hour=9,
        time_zone="UTC",
        max_init_times=10,
    )

    assert "time" in stitched.dims
    # Expect 48 hours (two days) starting at midnight after the first init
    assert stitched.sizes["time"] == 48
    first_time = pd.Timestamp(datetime(2025, 1, 2, 0, 0)).tz_localize(None)
    last_time = pd.Timestamp(datetime(2025, 1, 3, 23, 0)).tz_localize(None)
    assert pd.Timestamp(stitched.time.values[0]) == first_time
    assert pd.Timestamp(stitched.time.values[-1]) == last_time


def _make_ds_15min(zone: str, psr: str, init_times: list[datetime]) -> xr.Dataset:
    """Create a 15-minute-resolution dataset with ~40h of horizon per init."""
    rows = []
    for i, it in enumerate(init_times):
        for step in range(0, 40 * 4):  # 0..40h at 15-min steps
            rows.append(
                {
                    "zone_key": zone,
                    "psr_type": psr,
                    "init_time": pd.Timestamp(it),
                    "time": pd.Timestamp(it + timedelta(minutes=15 * step)),
                    "value": float(i * 1000 + step),
                }
            )
    df = pd.DataFrame(rows)
    return xr.Dataset.from_dataframe(
        df.set_index(["zone_key", "psr_type", "init_time", "time"])
    )


def test_get_day_ahead_timeseries_dedupes_overlapping_runs(monkeypatch):
    """Sub-hourly runs sharing a target hour produce overlapping windows.

    The stitched series must stay unique by keeping the most recent init's
    value for each valid time instead of raising on a non-unique index.
    """
    client = JuaClient()
    pf = client.power_forecast

    zone = "GB"
    psr = "Solar"
    # Two runs in the same hour-of-day (09:00 and 09:30) on a 15-min grid.
    t_early = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)
    t_late = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    init_infos = [
        InitTimeInfo(init_time=t_late, max_prediction_timedelta=40 * 60),
        InitTimeInfo(init_time=t_early, max_prediction_timedelta=40 * 60),
    ]

    monkeypatch.setattr(
        pf,
        "get_init_times",
        lambda zone_key=None, psr_type=None, limit=96: init_infos,
    )
    monkeypatch.setattr(
        pf,
        "get_data",
        lambda **kwargs: _make_ds_15min(zone, psr, [t_early, t_late]),
    )

    stitched = pf.get_day_ahead_timeseries(
        zone_keys=[zone],
        psr_types=[psr],
        init_hour=9,
        time_zone="UTC",
        max_init_times=10,
    )

    times = pd.to_datetime(stitched.time.values)
    assert len(times) == len(set(times)), "stitched time index must be unique"


def test_get_day_ahead_timeseries_date_range_builds_inits_and_clips(monkeypatch):
    """Date-range mode constructs daily inits in a single request and clips
    the result to the requested valid-time window."""
    client = JuaClient()
    pf = client.power_forecast

    zone, psr = "DE", "Solar"
    init_hour = 9

    calls = {"n_inits": []}

    def fake_get_data(**kwargs):
        inits = kwargs["init_time"]
        calls["n_inits"].append(len(inits))
        return _make_ds_15min(zone, psr, list(inits))

    # get_init_times must NOT be used in date-range mode.
    def fail_init_times(*a, **k):
        raise AssertionError("get_init_times should not be called in date-range mode")

    monkeypatch.setattr(pf, "get_data", fake_get_data)
    monkeypatch.setattr(pf, "get_init_times", fail_init_times)

    start = datetime(2025, 6, 1, tzinfo=timezone.utc)
    end = datetime(2025, 6, 11, tzinfo=timezone.utc)  # 10 valid days

    ds = pf.get_day_ahead_timeseries(
        zone_keys=[zone],
        psr_types=[psr],
        init_hour=init_hour,
        time_zone="UTC",
        start_date=start,
        end_date=end,
    )

    # Single request with daily inits from (start-1) through end (12 days). The
    # final init's window falls outside [start, end) and is clipped away.
    assert calls["n_inits"] == [12]

    times = pd.to_datetime(ds.time.values)
    assert len(times) == len(set(times)), "time index must be unique"
    # 10 days at 15-min resolution
    assert ds.sizes["time"] == 10 * 96
    lo = pd.Timestamp(start).tz_convert("UTC")
    hi = pd.Timestamp(end).tz_convert("UTC")
    tmin = pd.Timestamp(times.min())
    tmax = pd.Timestamp(times.max())
    tmin = tmin.tz_localize("UTC") if tmin.tzinfo is None else tmin
    tmax = tmax.tz_localize("UTC") if tmax.tzinfo is None else tmax
    assert tmin >= lo and tmax < hi
