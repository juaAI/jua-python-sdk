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
    )
    as_xarray = forecast.to_xarray()

    # Plot the first point's air temperature at height level 2m in Celsius
    as_xarray[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M][0].jua.to_celcius().plot()
    plt.show()

    # For the latest forecast, we can also query multiple locations
    # Let's compare Zurichs temperature to that of Cape Town
    zurich = LatLon(lat=47.3769, lon=8.5417)
    cape_town = LatLon(lat=-33.9249, lon=18.4241)
    forecast = model.forecast.get_forecast(
        points=[zurich, cape_town],
    ).to_xarray()

    # plot the temperature of the two points
    print(forecast)
    # jua.select_points allows us to use isel(point=0,1,2,...)
    temp_data = forecast.jua.select_points(points=[zurich, cape_town])[
        Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M
    ]
    temp_data_celsius = temp_data.jua.to_celcius()

    # Plot each point separately
    temp_data_celsius.isel(point=0).plot(label="Zurich")
    temp_data_celsius.isel(point=1).plot(label="Cape Town")

    plt.title("Temperature Forecast Comparison")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
