import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from jua import JuaClient
from jua.types.geo import LatLon
from jua.weather import Models, Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = JuaClient()
    model = client.weather.get_model(Models.EPT2)

    # Query a historical forecast (hindcast) for January 2024
    # This gets all forecasts run in the month
    zurich = LatLon(lat=47.3769, lon=8.5417)
    hindcast = model.get_forecasts(
        init_time=slice(
            datetime(2024, month=1, day=1, hour=0),
            datetime(2024, month=1, day=31, hour=23, minute=59),
        ),
        variables=[
            Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
            Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
        ],
        max_lead_time=48,
        points=zurich,
        method="nearest",
    )

    # Plot the temperature forecast for the first init time
    air_temp = hindcast[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M].isel(init_time=0)
    air_temp.to_celcius().to_absolute_time().plot()
    plt.title(f"Temperature Analysis - {air_temp.init_time.to_numpy()}")
    plt.show()

    # Plot the wind forecast for the first init time
    wind_speed = hindcast[Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M].isel(init_time=0)
    wind_speed.to_absolute_time().plot()
    plt.title(f"Wind Speed Analysis - {wind_speed.init_time.to_numpy()}")
    plt.show()

    # Save the selected data
    ds_hindcast = hindcast.to_xarray()
    output_path = Path("ept2_hindcast_2024_january.zarr")
    with ProgressBar():
        ds_hindcast.to_zarr(output_path, mode="w", compute=True)


if __name__ == "__main__":
    main()
