from jua.client import JuaClient
from jua.logging import get_logger
from jua.weather._model import Model
from jua.weather.models import Model as ModelEnum

logger = get_logger(__name__)


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
            model: _LayzModelWrapper(client=client, model_name=model)
            for model in ModelEnum
        }

    def __getitem__(self, model: ModelEnum) -> Model:
        return self.get_model(model)

    def get_model(self, model: ModelEnum) -> Model:
        return self._lazy_models[model].get_model()
