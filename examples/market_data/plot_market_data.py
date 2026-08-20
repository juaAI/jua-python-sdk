"""Plot zone-addressed market data: GB wind split and DE load vs forecast.

Produces a single figure with two panels from the unified ``market_data`` API:

1. GB wind generation as total, transmission-connected, and distribution-
   embedded (the components sum to the total).
2. Germany actual demand against the day-ahead load forecast.

Saves the figure to ``market_data_overview.png``.
"""

from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from jua import JuaClient

COLORS = {
    "wind": "#10B981",
    "wind_transmission": "#6366F1",
    "wind_embedded": "#EC4899",
    "load": "#1F2937",
    "load_forecast": "#EF4444",
}


def _series(df, variable):
    """Return (times, values) for one variable, sorted by time."""
    sub = df[df["variable"] == variable].sort_values("time")
    return sub["time"], sub["value"]


def main():
    md = JuaClient().market_data

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=7)

    gb = md.get_data(
        market_zone="GB",
        variables=["wind", "wind_transmission", "wind_embedded"],
        start_time=start,
        end_time=end,
        time_zone="Europe/London",
    )
    de = md.get_data(
        market_zone="DE",
        variables=["load", "load_forecast"],
        start_time=start,
        end_time=end,
        time_zone="Europe/Berlin",
    )
    if gb.empty and de.empty:
        print("No data returned.")
        return

    fig, (ax_wind, ax_load) = plt.subplots(2, 1, figsize=(15, 9))

    # Panel 1: GB wind total + components.
    for variable in ["wind", "wind_transmission", "wind_embedded"]:
        times, values = _series(gb, variable)
        if times.empty:
            continue
        ax_wind.plot(
            times,
            values,
            label=variable,
            color=COLORS[variable],
            linewidth=1.4 if variable == "wind" else 1.0,
            alpha=0.9,
        )
    ax_wind.set_title("GB wind — total vs transmission + embedded", fontweight="bold")
    ax_wind.set_ylabel("Power [MW]")

    # Panel 2: DE load vs day-ahead forecast.
    for variable in ["load", "load_forecast"]:
        times, values = _series(de, variable)
        if times.empty:
            continue
        ax_load.plot(
            times,
            values,
            label=variable,
            color=COLORS[variable],
            linewidth=1.2,
            alpha=0.9,
            linestyle="--" if variable == "load_forecast" else "-",
        )
    ax_load.set_title("DE load — actual vs day-ahead forecast", fontweight="bold")
    ax_load.set_ylabel("Demand [MW]")

    for ax in (ax_wind, ax_load):
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.autofmt_xdate()
    fig.tight_layout()
    out = "market_data_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
