# Jua Python SDK

Access industry-leading weather forecasts wtih ease

## Getting Started 🚀

### Install

We strongly recommend using [uv](https://docs.astral.sh/uv/) to manage dependencies. `python>=3.11` is required.
TODO: Create PyPI entry

### Authentication

TODO: After installing run `jua auth`. This will open your webbrowser for authentication.

Alternatively, generate an API Key [here](https://app.jua.sh/api-keys) and copy the file to `~/.jua/default/api-key.json`.

### Access the latest 20-day forecast for a specific point

```python
import matplotlib.pyplot as plt
from jua.client import JuaClient
from jua.types.geo import LatLon
from jua.weather.models import Models
from jua.weather.variables import Variables

client = JuaClient()
model = client.weather.get_model(Models.EPT1_5)
zurich = LatLon(lat=47.3769, lon=8.5417)
forecast = model.forecast.get_latest(
    points=[zurich],
    max_lead_time=480, # Hours
)
temp_data = forecast.to_xarray()[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
temp_data.jua.to_celcius().plot()
```

### Plot the global forecast with 10h lead time

```python
import matplotlib.pyplot as plt
from jua.client import JuaClient
from jua.types.geo import LatLon
from jua.weather.models import Models
from jua.weather.variables import Variables

client = JuaClient()
model = client.weather.get_model(Models.EPT1_5)

lead_time = 10 # hours
dataset = model.forecast.get_latest_forecast_as_dataset()
dataset[Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M].jua.sel(prediction_timedelta=lead_time).plot()
plt.show()
```

### Access historical data with ease

```python
import matplotlib.pyplot as plt
from jua.client import JuaClient
from jua.types.geo import LatLon
from jua.weather.models import Models
from jua.weather.variables import Variables

client = JuaClient()
model = client.weather.get_model(Models.EPT1_5_EARLY)

hindcast = model.hindcast.get_hindcast_as_dataset()
time = "2024-02-01T06:00:00.000000000"
data = (
    hindcast[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
    .jua.sel(time=time, prediction_timedelta=0, method="nearest")
    # Selecting Europe
    # Note that the latitude must be inverted
    .sel(latitude=slice(71, 36), longitude=slice(0, 50))
)
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
