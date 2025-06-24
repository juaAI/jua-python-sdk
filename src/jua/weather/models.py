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
        >>> model.forecast.get_latest()
    """

    EPT1_5 = "ept1_5"
    EPT1_5_EARLY = "ept1_5_early"
    EPT2 = "ept2"
    EPT2_EARLY = "ept2_early"
    ECMWF_IFS_SINGLE = "ecmwf_ifs025_single"
    ECMWF_IFS_ENSEMBLE = "ecmwf_ifs025_ensemble"
    ECMWF_AIFS_SINGLE = "ecmwf_aifs025_single"
    METEOFRANCE_AROME_FRANCE_HD = "meteofrance_arome_france_hd"
    GFS_GLOBAL_SINGLE = "gfs_global_single"
    GFS_GLOBAL_ENSEMBLE = "gfs_global_ensemble"
    ICON_EU = "icon_eu"
    GFS_GRAPHCAST = "gfs_graphcast025"
    AURORA = "aurora"
    AIFS = "aifs"
    EPT2_E = "ept2-e"
