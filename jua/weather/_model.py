from jua.client import JuaClient
from jua.weather.forecast import Forecast


class Model:
    def __init__(self, client: JuaClient, model_name: str):
        self._client = client
        self._model_name = model_name

        self._forecast = Forecast(client, model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def forecast(self) -> Forecast:
        return self._forecast

    def __repr__(self) -> str:
        return f"<Model name='{self.model_name}'>"

    def __str__(self) -> str:
        return self.model_name
