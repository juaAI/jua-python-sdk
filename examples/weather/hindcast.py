import logging
from pathlib import Path

import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from jua import JuaClient
from jua.weather import Models, Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = JuaClient()
    model = client.weather.get_model(Models.EPT1_5)

    time = "2024-02-01T06:00:00.000000000"
    hindcast = model.hindcast.get_hindcast(
        init_time=time,
        variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
        prediction_timedelta=0,
        # Select Europe
        latitude=slice(71, 36),
        longitude=slice(-15, 50),
        method="nearest",
    )

    data = hindcast[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
    data.plot()
    plt.show()

    # Save the selected data
    output_path = Path(
        "~/data/ept15_early_air_temperature_2024-02-01.zarr"
    ).expanduser()
    with ProgressBar():
        data.to_zarr(output_path, mode="w", compute=True)


if __name__ == "__main__":
    main()
