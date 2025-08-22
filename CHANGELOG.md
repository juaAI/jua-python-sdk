## v0.13.1 (2025-08-22)

### Fix

- ssrd is 1-hourly

## v0.13.0 (2025-08-21)

### Feat

- adds dew_point_temperature_at_height_level_2m, relative_humidity_at_height_level_2m and precipitation_amount_sum_1h variables

### Perf

- allow querying up to 1000 points in a single api call

## v0.12.0 (2025-08-20)

### Feat

- allow setting the log level of jua. loggers

### Fix

- improve warning message and show only when eagerly loaded

## v0.11.0 (2025-08-18)

### Feat

- convert pandas timedelta objects to_timedelta

## v0.10.0 (2025-08-15)

### Feat

- optional lazy loading
- support lazy loading

### Fix

- statistics might not be none but empty

## v0.9.1 (2025-08-12)

### Fix

- querying slices with ept2-rr
- in the .sel(_) patch, only swap the (start, end) for latitude slices if they are incorrect

## v0.9.0 (2025-08-12)

### Feat

- add ept2-rr and ecmwf_aifs025_ensemble models to the sdk

## v0.8.6 (2025-08-11)

### Fix

- update the `forecast_name_mapping` for ept-1.5 and ept-1.5 early and use the v3 adapter

## v0.8.5 (2025-08-07)

### Fix

- resolves bug where zarr and xarray interfere with each other

## v0.8.4 (2025-08-04)

### Fix

- fix the type annotation for ForecastMetadataResponse.available_ensemble_stats
- set a more verbose error message when the number of values for a variable does not match the number of lead times

## v0.8.3 (2025-07-14)

### Refactor

- **_api.py**: send user-agent header with every request

## v0.8.2 (2025-07-02)

### Fix

- remove trailing slash causing troubles opening dataset

## v0.8.1 (2025-07-01)

### Refactor

- use the jua production domain names

## v0.8.0 (2025-06-30)

### Feat

- Add support to pull statistics from ensemble forecasts through the Jua Client

## v0.7.2 (2025-06-25)

### Fix

- better support for points when requesting forecast

## v0.7.1 (2025-06-25)

### Fix

- missing support for time slices and lists

## v0.7.0 (2025-06-24)

### Feat

- add new models
- add ept2 ensemble
- add new models

### Fix

- model name

## v0.6.0 (2025-05-22)

### Feat

- retrieve hindcast files from api to be always up to date
- get chunk recommendations from api
- get latest hindcast files from api and drop metadata

### Fix

- adjust variable emcwf names
- add warning filter
- rename endpoint and remove debug pring

## v0.5.6 (2025-05-20)

### Fix

- trigger bumping and release

## v0.5.5 (2025-05-20)

### BREAKING CHANGE

- pe of change you are committing docs: Documentation only changes

### Feat

- **Added-commitizen-for-automated-versioning**: added the necessary dependencies

### Fix

- clean up ci testing
- **authentication.py**: remove support to load settings from .env (#3)
