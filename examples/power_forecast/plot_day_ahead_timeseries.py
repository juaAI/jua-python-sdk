"""Plot stitched day-ahead power forecast time series.

For each zone (DE and GB) two examples are produced:

1. A full year of day-ahead generation, stitched from the 09:00 UTC run each
   day (``plot_yearly_day_ahead``).
2. The last month of day-ahead generation, comparing the 10:00, 12:00 and
   18:00 UTC runs on a single panel (``plot_init_hour_comparison``).

Both rely on ``PowerForecast.get_day_ahead_timeseries`` with ``start_date`` /
``end_date``, which builds one init run per day over the range and stitches the
day-ahead window into a continuous series.
"""

from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from jua import JuaClient

ZONES = ["DE", "GB"]
PSR_COLORS = {
    "Solar": "#F59E0B",
    "Wind": "#10B981",
    "Wind Onshore": "#3B82F6",
    "Wind Offshore": "#06B6D4",
    "Wind Transmission": "#6366F1",
    "Wind Embedded": "#EC4899",
}


def _to_frame(ds):
    """Flatten a day-ahead dataset to a tidy DataFrame."""
    if "value" not in ds:
        return None
    df = ds.to_dataframe().reset_index()
    return df.dropna(subset=["value"]).sort_values("time")


def plot_yearly_day_ahead(pf, zone: str) -> None:
    """Plot a year of day-ahead generation from the 09:00 UTC run."""
    psr_types = pf.get_psr_types(zone_key=zone)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)

    print(f"Fetching yearly day-ahead series for {zone} (09:00 UTC run)...")
    ds = pf.get_day_ahead_timeseries(
        zone_keys=[zone],
        psr_types=psr_types,
        init_hour=9,
        time_zone="UTC",
        start_date=start,
        end_date=end,
    )
    df = _to_frame(ds)
    if df is None or df.empty:
        print("  No data returned.")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    for psr in psr_types:
        psr_df = df[df["psr_type"] == psr]
        if psr_df.empty:
            continue
        ax.plot(
            psr_df["time"],
            psr_df["value"],
            label=psr,
            color=PSR_COLORS.get(psr),
            linewidth=0.8,
            alpha=0.85,
        )

    ax.set_ylabel("Power [MW]", fontsize=12)
    ax.set_title(
        f"{zone} — Yearly day-ahead generation (09:00 UTC run)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    out = f"day_ahead_{zone.lower()}_yearly_09utc.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved {out}")


def plot_init_hour_comparison(pf, zone: str) -> None:
    """Compare the last month of Solar day-ahead across 10/12/18 UTC runs."""
    init_hours = [10, 12, 18]
    psr = "Solar"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)

    fig, ax = plt.subplots(figsize=(16, 6))
    cmap = plt.get_cmap("viridis")

    for idx, init_hour in enumerate(init_hours):
        print(f"Fetching last-month {zone} {psr} day-ahead ({init_hour:02d}:00 UTC)...")
        ds = pf.get_day_ahead_timeseries(
            zone_keys=[zone],
            psr_types=[psr],
            init_hour=init_hour,
            time_zone="UTC",
            start_date=start,
            end_date=end,
        )
        df = _to_frame(ds)
        if df is None or df.empty:
            print(f"  No data for {init_hour:02d}:00 UTC run.")
            continue
        ax.plot(
            df["time"],
            df["value"],
            label=f"{init_hour:02d}:00 UTC run",
            color=cmap(idx / max(len(init_hours) - 1, 1)),
            linewidth=1.0,
            alpha=0.85,
        )

    ax.set_ylabel("Power [MW]", fontsize=12)
    ax.set_title(
        f"{zone} — {psr} day-ahead, last month by init hour",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    fig.autofmt_xdate()

    out = f"day_ahead_{zone.lower()}_last_month_init_hours.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved {out}")


def main():
    client = JuaClient()
    pf = client.power_forecast

    for zone in ZONES:
        plot_yearly_day_ahead(pf, zone)
        plot_init_hour_comparison(pf, zone)


if __name__ == "__main__":
    main()
