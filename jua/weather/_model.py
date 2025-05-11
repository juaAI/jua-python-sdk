from jua.client import JuaClient
from jua.weather.forecast import Forecast
from jua.weather.hindcast import Hindcast
from jua.weather.models import Model as ModelEnum


class Model:
    def __init__(
        self,
        client: JuaClient,
        model: ModelEnum,
    ):
        self._client = client
        self._model = model

        self._forecast = Forecast(
            client,
            model=model,
        )

        self._hindcast = Hindcast(
            client,
            model=model,
        )

    @property
    def model_name(self) -> str:
        return self._model.value

    @property
    def forecast(self) -> Forecast:
        return self._forecast

    @property
    def hindcast(self) -> Hindcast:
        return self._hindcast

    def __repr__(self) -> str:
        return f"<Model name='{self.model_name}'>"

    def __str__(self) -> str:
        return self.model_name
