# models.py

from enum import Enum


class Model(str, Enum):
    EPT1_5 = "ept1_5"
    EPT1_5_EARLY = "ept1_5_early"
    EPT2 = "ept2"
    EPT2_EARLY = "ept2_early"
    ECMWF_IFS025_SINGLE = "ecmwf_ifs025_single"
    ECMWF_IFS025_ENSEMBLE = "ecmwf_ifs025_ensemble"
    ECMWF_AIFS025_SINGLE = "ecmwf_aifs025_single"
    METEOFRANCE_AROME_FRANCE_HD = "meteofrance_arome_france_hd"
    GFS_GLOBAL_SINGLE = "gfs_global_single"
    GFS_GLOBAL_ENSEMBLE = "gfs_global_ensemble"
    ICON_EU = "icon_eu"
    GFS_GRAPHCast025 = "gfs_graphcast025"
