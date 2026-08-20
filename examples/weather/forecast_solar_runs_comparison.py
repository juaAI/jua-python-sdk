"""Compare recent solar-radiation forecast runs at a point, valid-time aligned.

Plots 1h surface downwelling shortwave flux (SSRD) at Zurich for the most recent
runs of Helios and ICON-EU. Every run is drawn against valid time, colored by
model with a light->dark gradient over init time (older runs lighter).

A black "ground truth" line is built from each Helios run's T+1h value (valid at
init + 1h), stitched across runs to approximate the analysis.

This uses cheap single-point queries, so it runs against the live latest
forecasts without needing a raised credit limit.
"""

import logging
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from jua import JuaClient
from jua.types.geo import LatLon
from jua.weather import Models, Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ZURICH = LatLon(lat=47.3769, lon=8.5417, label="Zurich")
VARIABLE = Variables.SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H

# Only keep runs initialized within this many hours of the latest available run.
WINDOW_HOURS = 6

# (label, matplotlib colormap) per model.
MODEL_CONFIG = {
    Models.EPT2_HELIOS: ("Helios", "Oranges"),
    Models.ICON_EU: ("ICON-EU", "Purples"),
}


def _naive_utc(ts) -> pd.Timestamp:
    """Normalize a timestamp to tz-naive UTC."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def available_init_times(model_obj) -> list[pd.Timestamp]:
    """Sorted (ascending) tz-naive init times available for a model."""
    available = model_obj.get_available_forecasts(limit=50)
    return sorted(_naive_utc(f.init_time) for f in available.forecasts)


def ground_truth(model_obj, init_times, day_start, day_end):
    """Stitch each run's T+1h value into a pseudo ground-truth series."""
    inits = [t for t in init_times if day_start <= t + timedelta(hours=1) <= day_end]
    if not inits:
        return None, None

    forecast = model_obj.get_forecasts(
        init_time=[t.to_pydatetime() for t in inits],
        points=ZURICH,
        variables=[VARIABLE],
        prediction_timedelta=[np.timedelta64(60, "m")],
        stream=False,
    )
    da = forecast[VARIABLE].squeeze()
    valid_times = pd.to_datetime(da["init_time"].values) + pd.Timedelta(minutes=60)
    values = np.asarray(da.values).ravel()
    order = np.argsort(valid_times)
    return valid_times[order], values[order]


def main():
    client = JuaClient()

    helios = client.weather.get_model(Models.EPT2_HELIOS)
    helios_inits = available_init_times(helios)
    if not helios_inits:
        logger.warning("No Helios runs available; nothing to plot.")
        return

    # Frame the chart on the day of the latest Helios run.
    latest_init = helios_inits[-1]
    day_start = latest_init.normalize()
    day_end = day_start + timedelta(days=1)

    fig, ax = plt.subplots(figsize=(14, 7))
    legend_handles = []

    for model_enum, (label, cmap_name) in MODEL_CONFIG.items():
        model_obj = client.weather.get_model(model_enum)
        init_times = available_init_times(model_obj)
        runs = [
            t for t in init_times if t >= init_times[-1] - timedelta(hours=WINDOW_HOURS)
        ]
        if not runs:
            logger.warning(f"No runs found for {label}")
            continue

        cmap = plt.get_cmap(cmap_name)
        n = len(runs)
        logger.info(f"{label}: {n} runs from {runs[0]} to {runs[-1]}")

        for i, init_time in enumerate(runs):
            max_lead = int((day_end - init_time).total_seconds() / 3600) + 1
            if max_lead <= 0:
                continue

            forecast = model_obj.get_forecasts(
                init_time=init_time.to_pydatetime(),
                points=ZURICH,
                variables=[VARIABLE],
                max_lead_time=min(48, max_lead),
                stream=False,
            )
            da = forecast[VARIABLE].to_absolute_time().squeeze()
            times = pd.to_datetime(da["time"].values)
            values = np.asarray(da.values).ravel()

            mask = times <= day_end
            times, values = times[mask], values[mask]
            if len(times) == 0:
                continue

            # Older runs lighter, newest darkest.
            shade = 0.35 + 0.6 * (i / max(n - 1, 1))
            ax.plot(times, values, color=cmap(shade), linewidth=1.6, alpha=0.9)

        legend_handles.append(
            Line2D([0], [0], color=cmap(0.85), linewidth=2.5, label=label)
        )

    # Pseudo ground truth: Helios T+1h stitched across runs.
    gt_times, gt_values = ground_truth(helios, helios_inits, day_start, day_end)
    if gt_times is not None:
        ax.plot(gt_times, gt_values, color="black", linewidth=2.8, zorder=10)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.8,
                label="Ground truth (Helios T+1h)",
            )
        )

    ax.set_xlim(day_start, day_end)
    ax.set_xlabel("Valid time (UTC)")
    ax.set_ylabel(VARIABLE.display_name_with_unit)
    ax.set_title(
        f"SSRD 1h at {ZURICH.label} — recent runs (last {WINDOW_HOURS}h), "
        "gradient = init time (light=older)"
    )
    ax.legend(handles=legend_handles)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
