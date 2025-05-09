# models.py

from jua.client import JuaClient
from jua.weather._model import Model


class ModelsMeta(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # Collect all uppercase attributes as model names
        cls._MODEL_NAMES = [
            value
            for key, value in namespace.items()
            if key.isupper() and isinstance(value, str) and not key.startswith("_")
        ]
        return cls

    def __iter__(cls):
        return iter(cls._MODEL_NAMES)

    def __contains__(cls, item):
        return item in cls._MODEL_NAMES


class Models(metaclass=ModelsMeta):
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

    def __init__(self, client: JuaClient) -> None:
        self._client = client
        self._model_cache = {}

    def __getitem__(self, model: str) -> Model:
        if model not in self._model_cache:
            self._model_cache[model] = Model(self._client, model)
        return self._model_cache[model]
