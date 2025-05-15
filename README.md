# Jua Python SDK

Access industry-leading weather forecasts wtih ease

## Getting Started 🚀

### Install

We strongly recommend using [uv](https://docs.astral.sh/uv/) to manage dependencies. `python>=3.11` is required.
TODO: Create PyPI entry

### Authentication

TODO: After installing run `jua auth`. This will open your webbrowser for authentication.

Alternatively, generate an API Key [here](https://app.jua.sh/api-keys) and copy the file to `~/.jua/default/api-key.json`.

### Access the latest 20-day forecast for a specific points

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
temp_data = forecast.to_xarray()[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
temp_data.to_celcius().plot()
```

### Plot the global forecast with 10h lead time

```python
import matplotlib.pyplot as plt
from jua import JuaClient
from jua.types.geo import LatLon
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

### Access historical data with ease

```python
import matplotlib.pyplot as plt
from jua import JuaClient
from jua.types.geo import LatLon
from jua.weather import Models, Variables

client = JuaClient()
model = client.weather.get_model(Models.EPT1_5_EARLY)

time = "2024-02-01T06:00:00.000000000"
hindcast = model.hindcast.get_hindcast(
    variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
    time=time,
    prediction_timedelta=0,
    # Select Europe
    latitude=slice(71, 36),  # Note: slice is inverted
    longitude=slice(-15, 50),
    method="nearest",
)

data = hindcast.to_xarray()[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
data.plot()
plt.show()
```

## Development

To install all dependencies run

```
uv sync --all-extras
```

Enable pre-commit for linting and formatting:

```
uv run pre-commit install && uv run pre-commit install-hooks
```
