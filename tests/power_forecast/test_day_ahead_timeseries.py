import pandas as pd
import xarray as xr
from datetime import datetime, timezone, timedelta

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
    return xr.Dataset.from_dataframe(df.set_index(["zone_key", "psr_type", "init_time", "time"]))


def test_get_day_ahead_timeseries_stitches_across_days(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast

    zone = "GB"
    psr = "Solar"
    t1 = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, 9, 0, tzinfo=timezone.utc)
    init_infos = [InitTimeInfo(init_time=t1, max_prediction_timedelta=40 * 60),
                  InitTimeInfo(init_time=t2, max_prediction_timedelta=40 * 60)]

    # Patch network methods
    monkeypatch.setattr(pf, "get_init_times", lambda zone_key=None, psr_type=None, limit=96: init_infos)
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
