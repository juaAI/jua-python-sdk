## v0.19.2 (2025-11-12)

### Fix

- improve error message for broken connections when pulling data

## v0.19.1 (2025-11-11)

### Fix

- fixes the passing of variables to xr when lazy-loading and adds a test for it

## v0.19.0 (2025-11-10)

### Feat

- add noaa_gfs_single model
- adds the core code for lazy-loading using the new api

### Fix

- fix lazy loading xarray creation

### Refactor

- clear the lazy loaded cache on close
- linting, comments and cleaning code
- clean unused var
- clean lazy-loading indexing
- clean code and improve lazy-loading logic and tests

## v0.18.0 (2025-11-05)

### Feat

- allow to query if a model is a third party model
- add additional fields to meta response

### Fix

- naming convention

## v0.17.1 (2025-11-03)

### Fix

- makes small fixes to statistics access

## v0.17.0 (2025-10-31)

### Feat

- implement statistics in get_forecasts

### Fix

- better error messages when users make malformed queries, and improvements to docstrings

## v0.16.0 (2025-10-30)

### Feat

- add model.is_ready method

## v0.15.6 (2025-10-30)

### Fix

- improve the progress logging to the console, flush the output and close the progress
- set stream to false and log a warning when bilinear interpolation is used for point queries

## v0.15.5 (2025-10-30)

### Fix

- typo and unit
- no need to print progress when not streaming
- replacing all utf8 characters causing issues sometimes

## v0.15.4 (2025-10-30)

### Fix

- use max_prediction_timdelta in legacy function

## v0.15.3 (2025-10-29)

### Fix

- use returned lat and lon instead of requested

## v0.15.2 (2025-10-29)

### Fix

- correctly match point indices

### Refactor

- remove debug print
- remove unused code

## v0.15.1 (2025-10-29)

### Fix

- make sdk compatible with updated api

## v0.15.0 (2025-10-24)

### Feat

- add methods to obtain model metadata (available runs, available variables, latest run)
- add models missing from the sdk
- set the timedelta_unit in requests
- add ept2-hrrr
- adds the possibility to set the credit limit in the jua client
- enable access to market aggregates in the sdk
- add ecmwf ifs support
- updated get_forecasts method allowing to grids for continuous times

### Fix

- use hours for the market aggregates, as updated in the qe
- fix market aggregates and add docs
- fix how point coordinates are added to returned xarray datasets, update examples
- fix tests
- fix xarray patches for init_time
- update the cloud area variable names
- fix multi-point queries and improve returned dataset for point queries
- fix ept2e access through the new sdk method
- better handling when no data is returned

### Refactor

- merge main

## v0.14.5 (2025-10-17)

### Fix

- update model names to use underscores

## v0.14.4 (2025-09-24)

### Fix

- makes ept2 forecasts available back to 2024-04-01

## v0.14.3 (2025-09-24)

### Fix

- fixes access to fdir and dew point variables through the sdk

## v0.14.2 (2025-09-06)

### Fix

- comparing offset-naive and offset-aware datetimes
- dispatching points to api for past data

## v0.14.1 (2025-09-01)

### Fix

- patched latitude selection when a single latitude is returned fails as no len()

## v0.14.0 (2025-08-22)

### Feat

- add auth cli logic

### Fix

- move auth frontend to developer dashboard

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
