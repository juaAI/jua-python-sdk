"""Query ERA5 reanalysis: a point time series and a regional grid.

ERA5 is a reconstruction of past weather, so it has a single `time` dimension —
no `init_time` or `prediction_timedelta`. It is published with a lag of roughly
five days, so pick historical dates.
"""

import logging
from datetime import datetime

import matplotlib.pyplot as plt

from jua import JuaClient
from jua.types.geo import LatLon
from jua.weather import Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMPERATURE = Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M


def point_time_series(client: JuaClient) -> None:
    """One month of hourly 2m temperature for two cities."""
    zurich = LatLon(lat=47.3769, lon=8.5417, label="Zurich")
    london = LatLon(lat=51.5074, lon=-0.1278, label="London")

    data = client.reanalysis.get_data(
        time=slice(datetime(2024, 1, 1), datetime(2024, 2, 1)),
        variables=[TEMPERATURE],
        points=[zurich, london],
    )

    ds = data.to_xarray()
    logger.info("dimensions: %s", dict(ds.sizes))

    fig, ax = plt.subplots(figsize=(15, 6))
    for index in range(ds.sizes["points"]):
        series = data[TEMPERATURE].isel(points=index).to_celcius()
        series.plot(ax=ax, label=str(ds.points.values[index]))
    ax.set_title("ERA5 2m temperature, January 2024")
    ax.legend()
    plt.show()


def regional_grid(client: JuaClient) -> None:
    """A day of hourly data over Germany, averaged and plotted as a map."""
    data = client.reanalysis.get_data(
        time=slice(datetime(2024, 6, 1), datetime(2024, 6, 2)),
        variables=[TEMPERATURE],
        latitude=slice(55, 47),
        longitude=slice(5, 15),
    )

    ds = data.to_xarray()
    logger.info("dimensions: %s", dict(ds.sizes))
    logger.info("dataset size: %.2f MB", data.nbytes / 1e6)

    ds[TEMPERATURE.value.name].mean(dim="time").plot(figsize=(10, 8))
    plt.title("ERA5 mean 2m temperature over Germany, 2024-06-01")
    plt.show()

    # Persist to Zarr for reuse. Defaults to
    # ~/.jua/datasets/arco_era5/<dataset_name>.zarr
    data.save()


def discover(client: JuaClient) -> None:
    """What ERA5 serves."""
    meta = client.reanalysis.get_metadata()
    logger.info(
        "%s: %s grid, %s-minute steps, %d variables",
        meta.display_name,
        meta.grid_resolution,
        meta.temporal_resolution_minutes,
        len(meta.variables),
    )
    for name in meta.variables:
        logger.info("  %s (%s)", name, meta.variable_units.get(name, "?"))


def main() -> None:
    client = JuaClient()
    discover(client)
    point_time_series(client)
    regional_grid(client)


if __name__ == "__main__":
    main()
