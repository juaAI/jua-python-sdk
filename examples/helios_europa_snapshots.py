#!/usr/bin/env python
"""
Generate snapshot maps for EPT2 Helios and EPT2 Europa at multiple init times and lead times.

Usage:
  - Ensure JUA credentials are available via env:
      JUA_API_KEY_ID, JUA_API_KEY_SECRET
  - Optional: override region via CLI flags in the future; currently fixed to Europe
  - Run:
      python examples/helios_europa_snapshots.py

Outputs:
  - Saves PNG maps under examples/output/
  - Also attempts to save copies under /opt/cursor/artifacts/ if available (for CI/PR embedding)
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import numpy as np
import matplotlib

# Non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jua import JuaClient  # noqa: E402
from jua.weather import Models, Variables  # noqa: E402


def ensure_output_dirs() -> tuple[Path, Path | None]:
    repo_out = Path(__file__).parent / "output"
    repo_out.mkdir(parents=True, exist_ok=True)
    artifacts_root = Path("/opt/cursor/artifacts")
    artifacts_out: Path | None = None
    try:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        artifacts_out = artifacts_root / "helios_europa_maps"
        artifacts_out.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort; not fatal if artifacts path is unavailable
        artifacts_out = None
    return repo_out, artifacts_out


def pick_recent_inits(model, count: int = 3) -> list[datetime]:
    """Return up to `count` most recent init times for the model."""
    # Query the last 48 hours to be safe across different update cadences
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=48)
    af = model.get_available_forecasts(since=since, limit=200)
    inits = [fi.init_time for fi in af.forecasts if hasattr(fi, "init_time")]
    # Ensure uniqueness and sort descending
    uniq = sorted({t for t in inits}, reverse=True)
    return uniq[:count]


def plot_map(
    lat: np.ndarray,
    lon: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    save_paths: Sequence[Path],
) -> None:
    """Render a lat/lon 2D field and save to each provided path."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    # Expect latitude decreasing; pcolormesh expects corner grids; use shading="nearest"
    mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    cb.ax.set_ylabel("Units")
    fig.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def safe_unit_display(var: Variables) -> str:
    try:
        return var.display_name_with_unit
    except Exception:
        return var.name


def snapshot_for_model(
    client: JuaClient,
    model_id: Models,
    variables: Iterable[Variables],
    init_times: Sequence[datetime],
    lead_hours: Sequence[int],
    region: tuple[float, float, float, float],
    repo_out: Path,
    artifacts_out: Path | None,
) -> list[Path]:
    """Fetch slices and write maps; returns list of generated repo paths."""
    model = client.weather.get_model(model_id)

    lat_max, lat_min, lon_min, lon_max = region
    all_paths: list[Path] = []

    for init in init_times:
        ds = model.get_forecasts(
            init_time=[init],
            variables=[v for v in variables],
            latitude=slice(lat_max, lat_min),  # north to south
            longitude=slice(lon_min, lon_max),
            prediction_timedelta=list(lead_hours),
            stream=True,
            print_progress=True,
        ).to_xarray()

        # Dataset dims: (init_time, prediction_timedelta, latitude, longitude)
        for v in variables:
            da = ds[v.name].sel(init_time=init)
            # Choose colormap and scaling heuristics per variable
            if "shortwave" in v.name:
                cmap = "inferno"
                vmin, vmax = 0, float(np.nanpercentile(da.values, 99))
            elif "temperature" in v.name:
                cmap = "coolwarm"
                vmin, vmax = float(np.nanpercentile(da.values, 1)), float(
                    np.nanpercentile(da.values, 99)
                )
            else:
                cmap, vmin, vmax = "viridis", None, None

            for h in lead_hours:
                slice2d = da.sel(prediction_timedelta=h)
                # Convert to numpy array
                field = np.asarray(slice2d.values)
                lats = np.asarray(slice2d.latitude.values)
                lons = np.asarray(slice2d.longitude.values)

                init_str = init.strftime("%Y%m%dT%H%MZ")
                title = f"{model_id.value} | {safe_unit_display(v)} | init {init_str} | +{h}h"

                fname = f"{model_id.value}_{v.name}_init-{init_str}_lead-{h:03d}h.png"
                repo_path = repo_out / fname
                save_paths = [repo_path]
                if artifacts_out is not None:
                    save_paths.append(artifacts_out / fname)

                plot_map(
                    lat=lats,
                    lon=lons,
                    field=field,
                    title=title,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    save_paths=save_paths,
                )
                all_paths.append(repo_path)

    return all_paths


def main() -> None:
    # Fail fast on missing auth
    if not os.getenv("JUA_API_KEY_ID") or not os.getenv("JUA_API_KEY_SECRET"):
        raise SystemExit(
            "Missing credentials: set JUA_API_KEY_ID and JUA_API_KEY_SECRET in env to run."
        )

    repo_out, artifacts_out = ensure_output_dirs()
    client = JuaClient()

    # Region: Europe
    # latitude slice must be north-to-south (max to min)
    region = (72.0, 36.0, -15.0, 35.0)

    # Lead times to visualize (hours)
    lead_hours = [0, 6, 12, 24]

    # Models and variables
    models_and_vars: list[tuple[Models, list[Variables]]] = [
        (
            Models.EPT2_HELIOS,
            [
                Variables.SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_30MIN,
                Variables.SURFACE_DIRECT_DOWNWELLING_SHORTWAVE_FLUX_SUM_30MIN,
            ],
        ),
        (
            Models.EPT2_EUROPA,
            [
                Variables.SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H,
                Variables.SURFACE_DIRECT_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H,
            ],
        ),
    ]

    # For each model, pick the latest 3 init times and render maps
    for model_id, vars_ in models_and_vars:
        model = client.weather.get_model(model_id)
        inits = pick_recent_inits(model, count=3)
        if not inits:
            raise SystemExit(f"No init times available for {model_id.value}")

        print(f"Processing {model_id.value}: {len(inits)} init(s) -> {inits}")
        paths = snapshot_for_model(
            client=client,
            model_id=model_id,
            variables=vars_,
            init_times=inits,
            lead_hours=lead_hours,
            region=region,
            repo_out=repo_out,
            artifacts_out=artifacts_out,
        )
        print(f"Saved {len(paths)} image(s) for {model_id.value} under {repo_out}")


if __name__ == "__main__":
    main()

