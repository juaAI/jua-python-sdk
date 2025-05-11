import logging
from pathlib import Path

import matplotlib.pyplot as plt

from jua.client import JuaClient
from jua.weather.models import Models
from jua.weather.variables import Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = JuaClient()
    model = client.weather.get_model(Models.EPT1_5_EARLY)

    start_date = model.hindcast.metadata.start_date
    end_date = model.hindcast.metadata.end_date
    print(f"Hindcast from {start_date} to {end_date}")
    regions = model.hindcast.metadata.available_regions
    print(f"Regions: {', '.join([r.region for r in regions])}")

    hindcast = model.hindcast.get_hindcast_as_dataset()
    time = "2024-02-01T06:00:00.000000000"
    hindcast.download(
        start_date=time,
        end_date=time,
        always_download=True,
        overwrite=True,
    )

    print(hindcast.to_xarray())
    # get the data for the time
    data = (
        hindcast[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
        .jua.sel(time=time, prediction_timedelta=0, method="nearest")
        # Only plot a small part of the data for faster plotting
        # Note that the latitude must be inverted
        .sel(latitude=slice(71, 36), longitude=slice(0, 50))
    )
    data.plot()
    plt.show()

    # Save the selected data
    output_path = Path(
        "~/data/ept15_early_air_temperature_2024-02-01.zarr"
    ).expanduser()
    data.to_zarr(output_path, mode="w", zarr_format=hindcast.zarr_version)


if __name__ == "__main__":
    main()
