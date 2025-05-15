import logging

import matplotlib.pyplot as plt

from jua import JuaClient
from jua.types.geo import LatLon
from jua.weather import Models, Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    zurich = LatLon(lat=47.3769, lon=8.5417)

    client = JuaClient()
    models_to_use = [Models.EPT1_5, Models.EPT1_5_EARLY, Models.ECMWF_AIFS025_SINGLE]
    models = [client.weather.get_model(model) for model in models_to_use]

    for model in models:
        forecast = model.forecast.get_forecast(
            points=zurich,
        )
        temp_data = forecast[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
        temp_data_celsius = temp_data.to_absolute_time().to_celcius()
        temp_data_celsius.plot(label=model.name)

    plt.title("Temperature Forecast Comparison")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
