# models.py

from enum import Enum


class Models(str, Enum):
    """Weather forecast models available through the Jua API.

    This enum defines the set of weather models that can be requested
    when fetching forecasts or hindcasts. Use these constants when
    specifying which model to use with weather data functions.

    Examples:
        >>> from jua.weather.models import Models
        >>> # Request forecast from a specific model
        >>> model = client.weather.get_model(Models.EPT1_5)
        >>> geneva = LatLon(lat=46.2044, lon=6.1432, label="Geneva")
        >>> model.get_forecasts(points=[geneva])  # get the latest forcast for Geneva
    """

    # With Grid Access
    EPT1_5 = "ept1_5"
    EPT1_5_EARLY = "ept1_5_early"
    EPT2 = "ept2"
    EPT2_E = "ept2_e"
    EPT2_EARLY = "ept2_early"
    EPT2_HRRR = "ept2_hrrr"
    EPT2_RR = "ept2_rr"
    AIFS = "aifs"
    AURORA = "aurora"
    ECMWF_IFS_SINGLE = "ecmwf_ifs_single"
    NOAA_GFS_SINGLE = "noaa_gfs_single"

    # Without Grid Access
    ECMWF_AIFS_ENSEMBLE = "ecmwf_aifs025_ensemble"
    ECMWF_AIFS_SINGLE = "ecmwf_aifs025_single"
    ECMWF_IFS_ENSEMBLE = "ecmwf_ifs025_ensemble"
    GFS_GLOBAL_ENSEMBLE = "gfs_global_ensemble"
    GFS_GLOBAL_SINGLE = "gfs_global_single"
    GFS_GRAPHCAST = "gfs_graphcast025"
    ICON_D2 = "icon_d2"
    ICON_EU = "icon_eu"
    KNMI_HARMONIE_AROME_EUROPE = "knmi_harmonie_arome_europe"
    KNMI_HARMONIE_AROME_NETHERLANDS = "knmi_harmonie_arome_netherlands"
    METEOFRANCE_AROME_FRANCE_HD = "meteofrance_arome_france_hd"
    UKMO_GLOBAL_DETERMINISTIC_10KM = "ukmo_global_deterministic_10km"
    UKMO_UK_DETERMINISTIC_2KM = "ukmo_uk_deterministic_2km"
