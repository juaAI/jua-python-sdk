"""Plot power forecast data from the Jua API."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from jua import JuaClient


def main():
    client = JuaClient()
    pf = client.power_forecast

    zones = pf.get_zones()
    psr_types = pf.get_psr_types()
    print(f"Zones: {zones}")
    print(f"PSR types: {psr_types}")

    response = pf._api.post(
        "power-forecast/data",
        data={
            "zone_keys": zones,
            "init_time": ["latest-3", "latest-4"],
            "max_prediction_timedelta": 4320,
        },
        requires_auth=True,
    )
    data = response.json()
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df["init_time"] = pd.to_datetime(df["init_time"])

    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    init_times_in_data = sorted(df["init_time"].unique())
    n_inits = len(init_times_in_data)

    colors = {
        "Solar": "#F59E0B",
        "Wind Onshore": "#3B82F6",
        "Wind Offshore": "#06B6D4",
    }
    linestyles = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(
        len(zones),
        1,
        figsize=(14, 5 * len(zones)),
        sharex=True,
        squeeze=False,
    )

    for i, zone in enumerate(zones):
        ax = axes[i, 0]
        zone_df = df[df["zone_key"] == zone]

        for j, init_t in enumerate(init_times_in_data):
            init_df = zone_df[zone_df["init_time"] == init_t]
            init_label = pd.Timestamp(init_t).strftime("%b %d %H:%M")
            ls = linestyles[j % len(linestyles)]

            for psr in psr_types:
                psr_df = init_df[init_df["psr_type"] == psr].sort_values("time")
                if psr_df.empty:
                    continue
                ax.plot(
                    psr_df["time"],
                    psr_df["value"],
                    label=f"{psr} ({init_label})",
                    color=colors.get(psr, None),
                    linewidth=1.4,
                    linestyle=ls,
                    alpha=0.85,
                )

        ax.set_ylabel("Power [MW]", fontsize=12)
        ax.set_title(f"{zone} — Power Forecast", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle="--")
        if not zone_df.empty:
            ax.set_xlim(zone_df["time"].min(), zone_df["time"].max())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

    fig.suptitle(
        f"Jua Power Forecast  ({n_inits} init times)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = "power_forecast_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved plot to {out}")
    plt.show()


if __name__ == "__main__":
    main()
