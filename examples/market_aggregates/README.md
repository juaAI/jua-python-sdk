# Market Aggregates Examples

This directory contains examples demonstrating how to use Jua's market aggregate functionality to retrieve spatially aggregated weather forecasts.

## Overview

Market aggregates provide spatially weighted forecasts that are useful for energy market analysis. Aggregate variables automatically uses the appropriate weighting from the `Weighting` enum:

- **Wind variables** use `Weighting.WIND_CAPACITY`
- **Solar variables** use `Weighting.SOLAR_CAPACITY`
- **Temperature** uses `Weighting.POPULATION`

## Available Variables

The `AggregateVariables` enum provides the following variables:

- `WIND_SPEED_AT_HEIGHT_LEVEL_10M` - Wind speed at 10m height (`Weighting.WIND_CAPACITY`)
- `WIND_SPEED_AT_HEIGHT_LEVEL_100M` - Wind speed at 100m height (`Weighting.WIND_CAPACITY`)
- `SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H` - Surface downwelling shortwave flux (`Weighting.SOLAR_CAPACITY`)
- `AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M` - Air temperature at 2m height (`Weighting.POPULATION`)

## Key Features

1. **Automatic Weighting**: Each variable automatically uses the appropriate weighting type based on its characteristics. The weighting is determined when you query a specific variable.

2. **Market-Based Access**: Create an `EnergyMarket` for specific market zones, then query different variables for that market.

3. **ModelRuns Support**: Use `ModelRuns` objects to specify which model forecasts to query. Each ModelRuns can contain:
   - Single or multiple initialization times
   - Explicit datetimes for specific forecast runs
   - Non-negative integers (0, 1, etc.) for recent runs (0 = latest)
   - Mix of datetimes and integers

4. **Efficient API Calls**: Multiple init_times are resolved with minimal API calls:
   - All init_times for the same model are resolved in a single call
   - Results are automatically grouped by init_time for data queries

5. **Geographic Filtering**: Define market zones when creating the `EnergyMarket`:
   - Use `MarketZones` enum (e.g., `[MarketZones.IR, MarketZones.GB_NIR]`) for type safety
   - Use strings (e.g., `["IR", "GB-NIR"]`) for flexibility

6. **Time Filtering**: Control forecast lead times using:
   - `min_lead_time`: Minimum lead time in hours
   - `max_lead_time`: Maximum lead time in hours

## Usage Example

```python
from datetime import datetime
from jua import JuaClient
from jua.market_aggregates import AggregateVariables, ModelRuns
from jua.types import MarketZones
from jua.weather import Models

client = JuaClient()

# Create an energy market for specific zones
germany = client.market_aggregates.get_market(market_zone=MarketZones.DE)

# Create ModelRuns to specify which forecasts to query
# Here, we select the latest 3 EPT2 runs
model_runs = [ModelRuns(Models.EPT2, [0, 1, 2])]

# Retrieve wind data for Germany
wind_data = germany.compare_runs(
    agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
    model_runs=model_runs,
    max_lead_time=48,
)

# Query a different variable for the same market
temp_data = germany.compare_runs(
    agg_variable=AggregateVariables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
    model_runs=model_runs,
    max_lead_time=48,
)

# Or create a market for multiple zones
market_ir_nir = client.market_aggregates.get_market(
    market_zone=[MarketZones.IE, MarketZones.GB_NIR],
)

solar_data = market_ir_nir.compare_runs(
    agg_variable=AggregateVariables.SURFACE_DOWNWELLING_SHORTWAVE_FLUX_SUM_1H,
    model_runs=model_runs,
    max_lead_time=72,
)
```

## ModelRuns

A `ModelRuns` is a simple data container that combines a model with one or more initialization times:

```python
from jua.market_aggregates import ModelRuns
from jua.weather import Models
from datetime import datetime

# Single datetime
run1 = ModelRuns(Models.EPT2, datetime(2024, 8, 5, 0))

# Single integer index
latest = ModelRuns(Models.EPT2, 0)     # Latest run (index 0)

# Multiple integer indices (EFFICIENT - resolved in one API call!)
recent = ModelRuns(Models.EPT2, [0, 1, 2])  # Latest 3 runs

# Multiple datetimes
historical = ModelRuns(Models.EPT2, [
    datetime(2024, 8, 5, 0),
    datetime(2024, 8, 5, 6),
    datetime(2024, 8, 5, 12),
    datetime(2024, 8, 5, 18),
])
```

## API Design

The market aggregates API follows the pattern:

1. **Create an EnergyMarket**: `client.market_aggregates.get_market(market_zone=...)`
2. **Create ModelRuns**: Specify models and init_times (single or multiple per model)
3. **Query data**: Call `compare_runs(agg_variable=..., model_runs=...)`

This ensures that:
- Each query specifies the variable and uses appropriate weighting
- Multiple forecasts can be queried efficiently with minimal API calls
- All init_times for the same model are resolved in one call
- The API automatically handles init_time resolution and grouping
- Results are returned as an xarray Dataset with rich metadata

## MarketZones and Countries Enums

The `MarketZones` and `Countries` enums (located in `jua.types`) provide type-safe access to all 354 market zones and 213 countries.

```python
from jua.types import Countries, MarketZones

# Access zone information (zones are alphabetically ordered)
print(MarketZones.DE.zone_name)  # "DE"
print(MarketZones.DE.country)    # Countries.GERMANY
print(MarketZones.DE.country_name)  # "Germany"
print(MarketZones.NO_NO1.zone_name)  # "NO-NO1"
print(MarketZones.NO_NO1.country)    # Countries.NORWAY
print(MarketZones.NO_NO1.country_name)  # "Norway"

# Filter zones by country
norway_zones = MarketZones.filter_by_country(Countries.NORWAY)
print([z.zone_name for z in norway_zones])  # ['NO-NO1', 'NO-NO2', 'NO-NO3', 'NO-NO4', 'NO-NO5']

# Can also filter using string
germany_zones = MarketZones.filter_by_country("Germany")
print([z.zone_name for z in germany_zones])  # ['DE']

# Create market using filtered zones
norway = client.market_aggregates.get_market(market_zone=norway_zones)

# Use in queries (type-safe with IDE autocomplete)
data = norway.compare_runs(
    agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
    model_runs=[ModelRuns(Models.EPT2, 0)],
)
```

## Files

- `market_aggregate_example.py` - Basic examples of market aggregate queries
- `market_aggregate_example.ipynb` - Jupyter notebook showing use-cases of market aggregates

## Notes

- The returned data is an xarray Dataset with:
  - Dimensions: `model_run` and `time`
  - Data variables: `prediction_timedelta` and the queried variable
  - Extra Coordinates: `model` and `init_time`
  - Attributes: variable metadata, market zones, and lead time parameters
- Each EnergyMarket instance represents a specific set of market zones
- Results include rich metadata in the Dataset attributes for easy data exploration
