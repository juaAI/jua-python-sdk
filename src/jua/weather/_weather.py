from jua.client import JuaClient
from jua.errors.model_errors import ModelDoesNotExistError
from jua.logging import get_logger
from jua.weather._model import Model
from jua.weather.models import Models

logger = get_logger(__name__)


class _LazyModelWrapper:
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
            model: _LazyModelWrapper(client=client, model=model) for model in Models
        }

    def __getitem__(self, model: Models) -> Model:
        return self.get_model(model)

    def get_model(self, model: Models) -> Model:
        if model not in self._lazy_models:
            raise ModelDoesNotExistError(model)
        return self._lazy_models[model].get_model()
