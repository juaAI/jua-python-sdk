"""Helios SSRD 30min: init time vs valid time grid.

Plots a 6x6 grid where:
- Rows: init times (30min apart)
- Columns: valid times (30min apart)
Each cell shows the forecast for that valid time issued at that init time.
Cells on the same column share the same valid time but have different lead times.
"""

import logging
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jua import JuaClient
from jua.weather import Models, Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = JuaClient(request_credit_limit=10_000)
    helios = client.weather.get_model(Models.EPT2_HELIOS)

    base_init = datetime(2026, 6, 1, 12)
    n = 6
    step = timedelta(minutes=30)

    init_times = [base_init + i * step for i in range(n)]
    valid_times = [base_init + (i + 1) * step for i in range(n)]

    variable = Variables.SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_30MIN
    geo_kwargs = dict(latitude=slice(35, 60), longitude=slice(-15, 30))

    # Fetch data for each init time
    forecasts = {}
    for init_time in init_times:
        # Compute which valid times are reachable from this init
        tds = []
        for vt in valid_times:
            td_minutes = int((vt - init_time).total_seconds() / 60)
            if td_minutes > 0:
                tds.append(np.timedelta64(td_minutes, "m"))

        if not tds:
            continue

        logger.info(
            f"Fetching init={init_time.strftime('%H:%M')}, {len(tds)} lead times..."
        )
        data = helios.get_forecasts(
            init_time=init_time,
            prediction_timedelta=tds,
            variables=[variable],
            stream=False,
            **geo_kwargs,
        )
        forecasts[init_time] = data[variable]

    # Determine shared color scale
    all_vals = [float(f.min()) for f in forecasts.values()] + [
        float(f.max()) for f in forecasts.values()
    ]
    vmin, vmax = min(all_vals), max(all_vals)

    projection = ccrs.PlateCarree()
    fig, axs = plt.subplots(
        n,
        n,
        figsize=(4 * n, 3.5 * n),
        subplot_kw={"projection": projection},
    )

    for r_idx, init_time in enumerate(init_times):
        for c_idx, vt in enumerate(valid_times):
            ax = axs[r_idx, c_idx]
            td_minutes = int((vt - init_time).total_seconds() / 60)

            if td_minutes <= 0 or init_time not in forecasts:
                ax.set_visible(False)
                continue

            td = pd.Timedelta(minutes=td_minutes)
            field = forecasts[init_time]

            try:
                plot_data = field.sel(prediction_timedelta=td, method="nearest")
                plot_data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    add_colorbar=False,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="magma",
                )
            except (KeyError, ValueError):
                ax.set_visible(False)
                continue

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
            ax.set_extent([-15, 30, 35, 60], crs=ccrs.PlateCarree())
            ax.set_title("")

            # Label: lead time in the corner
            ax.text(
                0.05,
                0.95,
                f"td={td_minutes}m",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

    # Row labels (init times)
    for r_idx, init_time in enumerate(init_times):
        axs[r_idx, 0].text(
            -0.3,
            0.5,
            f"init\n{init_time.strftime('%H:%M')}",
            transform=axs[r_idx, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="center",
        )

    # Column labels (valid times)
    for c_idx, vt in enumerate(valid_times):
        axs[0, c_idx].set_title(
            f"valid {vt.strftime('%H:%M')}", fontsize=11, fontweight="bold"
        )

    title = (
        "Helios SSRD 30min — init time vs valid time\n"
        f"(base: {base_init.strftime('%Y-%m-%d %H:%M UTC')})"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plt.savefig("ssrd_helios_valid_time_grid.png", dpi=150, bbox_inches="tight")
    logger.info("Saved to ssrd_helios_valid_time_grid.png")


if __name__ == "__main__":
    main()
