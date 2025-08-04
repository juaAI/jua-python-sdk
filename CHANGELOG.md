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
