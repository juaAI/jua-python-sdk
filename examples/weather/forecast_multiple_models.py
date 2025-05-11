import logging

import matplotlib.pyplot as plt

from jua.client import JuaClient
from jua.types.weather.weather import Coordinate
from jua.weather.models import Models
from jua.weather.variables import Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    zurich = Coordinate(lat=47.3769, lon=8.5417)

    client = JuaClient()
    models_to_use = [Models.EPT1_5, Models.EPT1_5_EARLY, Models.ECMWF_AIFS025_SINGLE]
    models = [client.weather.get_model(model) for model in models_to_use]

    for model in models:
        forecast = model.forecast.get_latest(
            points=[zurich],
            max_lead_time=480,
        )
        # plot the temperature of the two points
        temp_data = forecast.to_xarray()[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
        temp_data_celsius = temp_data.jua.to_celcius()
        temp_data_celsius.plot(label=model.model_name)

    plt.title("Temperature Forecast Comparison")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
