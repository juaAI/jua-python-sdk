# Jua Python SDK

**Access industry-leading weather forecasts with ease**

The Jua Python SDK provides a simple and powerful interface to Jua's state-of-the-art weather forecasting capabilities. Easily integrate accurate weather data into your applications, research, or analysis workflows.

## Getting Started 🚀

### Prerequisites

- Python 3.11 or higher
- Internet connection for API access

### Installation

Install `jua` with pip:

```
pip install jua
```

Alternatively, checkout [uv](https://docs.astral.sh/uv/) for managing dependencies and Python versions:

```bash
uv init && uv add jua
```

### Authentication

Generate an API key from the [Jua dashboard](https://app.jua.sh/api-keys) and save it to `~/.jua/default/api-key.json`.

_Coming soon: Simply run `jua auth` to authenticate via your web browser._

## Examples

### Access the latest 20-day forecast for a point location

Retrieve temperature forecasts for Zurich and visualize the data:

```python
import matplotlib.pyplot as plt
from jua import JuaClient
from jua.types.geo import LatLon
from jua.weather import Models, Variables

client = JuaClient()
model = client.weather.get_model(Models.EPT1_5)
zurich = LatLon(lat=47.3769, lon=8.5417)
# Get latest forecast
forecast = model.forecast.get_forecast(
    points=[zurich]
)
temp_data = forecast[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
temp_data.to_celcius().to_absolute_time().plot()
plt.show()
```

<details>
<summary>Show output</summary>

![Forecast Zurich 20d](content/readme/forecast_zurich.png)

</details>

### Plot global forecast with 10-hour lead time

Generate a global wind speed visualization:

```python
import matplotlib.pyplot as plt
from jua import JuaClient
from jua.weather import Models, Variables

client = JuaClient()
model = client.weather.get_model(Models.EPT1_5)

lead_time = 10 # hours
dataset = model.forecast.get_forecast(
    prediction_timedelta=lead_time,
    variables=[
        Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
    ],
)
dataset[Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M].plot()
plt.show()
```

<details>
<summary>Show output</summary>

![Global Windspeed 10h](content/readme/global_windspeed_10h.png)

</details>

### Access historical weather data

Retrieve and visualize temperature data for Europe from a specific date:

```python
import matplotlib.pyplot as plt
from jua import JuaClient
from jua.weather import Models, Variables

client = JuaClient()
model = client.weather.get_model(Models.EPT1_5_EARLY)

init_time = "2024-02-01 06:00:00"
hindcast = model.hindcast.get_hindcast(
    variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
    init_time=init_time,
    prediction_timedelta=0,
    # Select Europe
    latitude=slice(71, 36),
    longitude=slice(-15, 50),
    method="nearest",
)

data = hindcast[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
data.plot()
plt.show()
```

<details>
<summary>Show output</summary>

![Europe Hindcast](content/readme/hindcast_europe.png)

</details>

## Documentation

For comprehensive documentation, visit [docs.jua.ai](https://docs.jua.ai).

## Contributing

See the [contribution guide](./CONTRIBUTING.md) to get started.

## Changes

See the [changelog](./CHANGELOG.md) for the latest changes.

## Support

If you encounter any issues or have questions, please:

- Check the [documentation](https://docs.jua.ai)
- Open an issue on GitHub
- Contact support@jua.ai

## License

This project is licensed under the MIT License - see the LICENSE file for details.
