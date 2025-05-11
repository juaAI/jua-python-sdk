from collections import defaultdict

from pydantic.dataclasses import dataclass

from jua.client import JuaClient
from jua.weather._model import Model
from jua.weather.models import Model as ModelEnum


@dataclass
class _ModelInfo:
    has_forecast_file_access: bool = False
    has_hindcast_file_access: bool = False


_MODEL_INFO = defaultdict(_ModelInfo)
_MODEL_INFO[ModelEnum.EPT1_5] = _ModelInfo(
    has_forecast_file_access=True, has_hindcast_file_access=True
)
_MODEL_INFO[ModelEnum.EPT1_5_EARLY] = _ModelInfo(
    has_forecast_file_access=True, has_hindcast_file_access=True
)
_MODEL_INFO[ModelEnum.EPT2] = _ModelInfo(
    has_forecast_file_access=True, has_hindcast_file_access=True
)
_MODEL_INFO[ModelEnum.EPT2_EARLY] = _ModelInfo(
    has_forecast_file_access=True, has_hindcast_file_access=True
)
_MODEL_INFO[ModelEnum.ECMWF_AIFS025_SINGLE] = _ModelInfo(
    has_forecast_file_access=True, has_hindcast_file_access=True
)


class _LayzModelWrapper:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._instance = None

    def get_model(self):
        if self._instance is None:
            self._instance = Model(**self._kwargs)
        return self._instance


class Weather:
    def __init__(self, client: JuaClient) -> None:
        self._client = client
        self._lazy_models = {
            model: _LayzModelWrapper(
                client=client,
                model_name=model,
                has_forecast_file_access=_MODEL_INFO[model].has_forecast_file_access,
                has_hindcast_file_access=_MODEL_INFO[model].has_hindcast_file_access,
            )
            for model in ModelEnum
        }

    def __getitem__(self, model: ModelEnum) -> Model:
        return self.get_model(model)

    def get_model(self, model: ModelEnum) -> Model:
        return self._lazy_models[model].get_model()
