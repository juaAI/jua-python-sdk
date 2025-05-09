from datetime import datetime

from jua.client import JuaClient
from jua.types.weather._api_payload_types import ForecastRequestPayload
from jua.types.weather._api_response_types import ForecastMetadataResponse
from jua.weather._api import WeatherAPI


class Forecast:
    def __init__(self, client: JuaClient, model_name: str):
        self._client = client
        self._model_name = model_name
        self._api = WeatherAPI(client)

    def get_latest(
        self,
        lat: float | None = None,
        lon: float | None = None,
        payload: ForecastRequestPayload | None = None,
    ):
        return self._api.get_latest_forecast(
            model_name=self._model_name, lat=lat, lon=lon, payload=payload
        )

    def get(
        self,
        lat: float | None = None,
        lon: float | None = None,
        payload: ForecastRequestPayload | None = None,
    ):
        return self._api.get_forecast(
            model_name=self._model_name, lat=lat, lon=lon, payload=payload
        )

    def get_metadata(self, init_time: datetime | str):
        return self._api.get_forecast_metadata(
            model_name=self._model_name, init_time=init_time
        )

    def get_available_init_times(self) -> list[datetime]:
        return self._api.get_available_init_times(model_name=self._model_name)

    def get_latest_metadata(self) -> ForecastMetadataResponse:
        return self._api.get_latest_forecast_metadata(model_name=self._model_name)
