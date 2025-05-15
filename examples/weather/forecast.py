import logging

import matplotlib.pyplot as plt

from jua import JuaClient
from jua.types.geo import LatLon
from jua.weather import Models, Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    client = JuaClient()
    model = client.weather.get_model(Models.EPT1_5)

    # Query the second-to-last forecast for Zurich, Switzerland
    # Note that only JUA's models support querying specific init times
    second_to_last_init_time = model.forecast.get_available_init_times()[1]
    print(f"Querying forecast for {second_to_last_init_time.isoformat()}")
    forecast = model.forecast.get_forecast(
        init_time=second_to_last_init_time,
        latitude=47.3769,
        longitude=8.5417,
    ).to_xarray()

    # Plot the first points's air temperature at height level 2m in Celsius
    forecast[
        Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M
    ].to_absolute_time().to_celcius().plot()
    plt.show()

    # Let's compare Zurichs temperature to that of Cape Town
    zurich = LatLon(lat=47.3769, lon=8.5417, label="Zurich")
    cape_town = LatLon(lat=-33.9249, lon=18.4241, label="Cape Town")
    forecast = model.forecast.get_forecast(
        points=[zurich, cape_town],
    )
    # plot the temperature of the two points
    print(forecast.to_xarray())
    # jua.select_point allows us to use isel(points=0,1,2,...)
    temp_data = forecast[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
    temp_data_celsius = temp_data.to_absolute_time().to_celcius()

    # Plot each points separately using numerical indexes
    temp_data_celsius.sel(points=zurich).plot(label="Zurich")
    temp_data_celsius.sel(points=cape_town).plot(label="Cape Town")

    plt.title("Temperature Forecast Comparison")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
