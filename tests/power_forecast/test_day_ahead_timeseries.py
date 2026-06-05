from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import xarray as xr

from jua import JuaClient
from jua.power_forecast.power_forecast import InitTimeInfo, PowerForecast


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


# ----------------------------------------------------------------------
# DST / mixed-offset parsing (regression for the day-ahead stitch bug)
# ----------------------------------------------------------------------


def test_to_dataset_parses_mixed_offsets_across_dst():
    """Mixed UTC offsets (across a DST boundary) must parse to a single
    tz-aware dtype, not an ``object`` column.

    Regression test: parsing such a response naively yields ``object`` dtype,
    which breaks the ``time - init_time`` subtraction used by the day-ahead
    stitching with ``TypeError: cannot subtract DatetimeArray from ndarray``.
    """
    # Europe/Berlin springs forward on 2026-03-29: +01:00 before, +02:00 after.
    data = {
        "zone_key": ["GB", "GB"],
        "psr_type": ["Solar", "Solar"],
        "init_time": [
            "2026-03-28T09:00:00+01:00",
            "2026-03-29T09:00:00+02:00",
        ],
        "time": [
            "2026-03-29T00:00:00+01:00",
            "2026-03-30T00:00:00+02:00",
        ],
        "value": [1.0, 2.0],
    }

    ds = PowerForecast._to_dataset(data, time_zone="Europe/Berlin")
    df = ds.to_dataframe().reset_index()

    assert isinstance(df["time"].dtype, pd.DatetimeTZDtype)
    assert isinstance(df["init_time"].dtype, pd.DatetimeTZDtype)
    assert str(df["time"].dt.tz) == "Europe/Berlin"
    # The exact operation that used to raise must now succeed and stay typed.
    lead = df["time"] - df["init_time"]
    assert str(lead.dtype).startswith("timedelta64")
    # On the populated diagonal each value is 15h after its own init.
    valid = df.dropna(subset=["value"])
    assert ((valid["time"] - valid["init_time"]) == pd.Timedelta(hours=15)).all()


def test_to_dataset_defaults_to_utc_for_mixed_offsets():
    """Without a ``time_zone`` the frame is normalized to tz-aware UTC."""
    data = {
        "zone_key": ["GB", "GB"],
        "psr_type": ["Solar", "Solar"],
        "init_time": [
            "2026-03-28T09:00:00+01:00",
            "2026-03-29T09:00:00+02:00",
        ],
        "time": [
            "2026-03-28T10:00:00+01:00",
            "2026-03-29T11:00:00+02:00",
        ],
        "value": [1.0, 2.0],
    }

    ds = PowerForecast._to_dataset(data)
    df = ds.to_dataframe().reset_index()

    assert isinstance(df["time"].dtype, pd.DatetimeTZDtype)
    assert str(df["time"].dt.tz) == "UTC"


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _columnar_response(
    *,
    zone: str,
    psr_types: list[str],
    init_iso_list: list[str],
    tz: ZoneInfo,
    step_minutes: int = 60,
    horizon_hours: int = 40,
):
    """Build a columnar power-forecast response for the requested inits.

    Times and init_times are emitted as zone-local ISO strings (mirroring the
    server), so a range crossing a DST transition naturally yields mixed UTC
    offsets - exactly the shape that exercises the parsing fix.
    """
    cols: dict[str, list] = {
        "zone_key": [],
        "psr_type": [],
        "init_time": [],
        "time": [],
        "value": [],
    }
    steps = int(horizon_hours * 60 / step_minutes)
    for it_iso in init_iso_list:
        it = datetime.fromisoformat(it_iso)  # tz-aware UTC
        for psr in psr_types:
            for s in range(steps):
                t = it + timedelta(minutes=step_minutes * s)
                cols["zone_key"].append(zone)
                cols["psr_type"].append(psr)
                cols["init_time"].append(it.astimezone(tz).isoformat())
                cols["time"].append(t.astimezone(tz).isoformat())
                cols["value"].append(float(s))
    return _FakeResponse(cols)


def _patch_post(monkeypatch, pf, zone, psr_types, tz):
    """Route the data endpoint through a synthetic columnar response."""

    def fake_post(path, data=None, requires_auth=True):
        return _columnar_response(
            zone=zone,
            psr_types=psr_types,
            init_iso_list=list(data["init_time"]),
            tz=tz,
        )

    monkeypatch.setattr(pf._api, "post", fake_post)


def _run_date_range(pf, *, zone, psr_types, init_hour, tz_name, start, end):
    return pf.get_day_ahead_timeseries(
        zone_keys=[zone],
        psr_types=psr_types,
        init_hour=init_hour,
        time_zone=tz_name,
        start_date=start,
        end_date=end,
    )


def test_day_ahead_date_range_handles_spring_forward_dst(monkeypatch):
    """End-to-end stitch across the spring-forward transition (clocks +1h).

    This reproduces Max's GB/init_hour=9 case: the response spans
    2026-03-29 where Europe/Berlin jumps +01:00 -> +02:00.
    """
    pf = JuaClient().power_forecast
    tz_name = "Europe/Berlin"
    tz = ZoneInfo(tz_name)
    _patch_post(monkeypatch, pf, "GB", ["Solar", "Wind"], tz)

    start = datetime(2026, 3, 27, tzinfo=tz)
    end = datetime(2026, 3, 31, tzinfo=tz)
    ds = _run_date_range(
        pf,
        zone="GB",
        psr_types=["Solar", "Wind"],
        init_hour=9,
        tz_name=tz_name,
        start=start,
        end=end,
    )

    times = ds.indexes["time"]
    assert isinstance(times.dtype, pd.DatetimeTZDtype)
    assert times.is_monotonic_increasing
    assert times.is_unique
    # The DST transition day is present and the series is continuous.
    days = {t.date() for t in times}
    assert datetime(2026, 3, 29).date() in days


def test_day_ahead_date_range_handles_fall_back_dst(monkeypatch):
    """End-to-end stitch across the fall-back transition (clocks -1h).

    Europe/Berlin falls back on 2025-10-26 (+02:00 -> +01:00), repeating the
    02:00 hour. The stitched index must stay unique and ordered.
    """
    pf = JuaClient().power_forecast
    tz_name = "Europe/Berlin"
    tz = ZoneInfo(tz_name)
    _patch_post(monkeypatch, pf, "DE", ["Solar"], tz)

    start = datetime(2025, 10, 24, tzinfo=tz)
    end = datetime(2025, 10, 28, tzinfo=tz)
    ds = _run_date_range(
        pf,
        zone="DE",
        psr_types=["Solar"],
        init_hour=9,
        tz_name=tz_name,
        start=start,
        end=end,
    )

    times = ds.indexes["time"]
    assert times.is_unique, "fall-back repeated hour must not duplicate the index"
    assert times.is_monotonic_increasing
    days = {t.date() for t in times}
    assert datetime(2025, 10, 26).date() in days


def test_day_ahead_date_range_includes_leap_day(monkeypatch):
    """A range spanning Feb 29 (2024) builds and stitches the leap day."""
    pf = JuaClient().power_forecast
    tz_name = "UTC"
    tz = ZoneInfo(tz_name)
    _patch_post(monkeypatch, pf, "DE", ["Solar"], tz)

    start = datetime(2024, 2, 28, tzinfo=tz)
    end = datetime(2024, 3, 2, tzinfo=tz)  # valid days: 28, 29, 1 -> 3 days
    ds = _run_date_range(
        pf,
        zone="DE",
        psr_types=["Solar"],
        init_hour=9,
        tz_name=tz_name,
        start=start,
        end=end,
    )

    times = ds.indexes["time"]
    assert times.is_unique
    days = {t.date() for t in times}
    assert datetime(2024, 2, 29).date() in days
    assert days == {
        datetime(2024, 2, 28).date(),
        datetime(2024, 2, 29).date(),
        datetime(2024, 3, 1).date(),
    }


def test_day_ahead_date_range_crosses_year_boundary(monkeypatch):
    """A range spanning New Year stitches across the Dec 31 -> Jan 1 rollover."""
    pf = JuaClient().power_forecast
    tz_name = "UTC"
    tz = ZoneInfo(tz_name)
    _patch_post(monkeypatch, pf, "DE", ["Solar"], tz)

    start = datetime(2025, 12, 31, tzinfo=tz)
    end = datetime(2026, 1, 2, tzinfo=tz)  # valid days: 12-31, 01-01 -> 2 days
    ds = _run_date_range(
        pf,
        zone="DE",
        psr_types=["Solar"],
        init_hour=9,
        tz_name=tz_name,
        start=start,
        end=end,
    )

    times = ds.indexes["time"]
    assert times.is_unique
    assert times.is_monotonic_increasing
    years = {t.year for t in times}
    assert years == {2025, 2026}
