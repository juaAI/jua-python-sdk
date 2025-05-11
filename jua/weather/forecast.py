from datetime import datetime

import xarray as xr
from pydantic import validate_call

from jua.client import JuaClient
from jua.types.weather._api_payload_types import ForecastRequestPayload
from jua.types.weather._api_response_types import ForecastMetadataResponse
from jua.weather._api import WeatherAPI
from jua.weather._jua_dataset import JuaDataset
from jua.weather.models import Model


class Forecast:
    _MODEL_NAME_MAPPINGS = {
        Model.EPT2: "ept-2",
        Model.EPT1_5: "ept-1.5",
    }

    def __init__(
        self,
        client: JuaClient,
        model: Model,
        has_forecast_file_access: bool,
    ):
        self._client = client
        self._model = model
        self._model_name = model.value
        self._api = WeatherAPI(client)
        self._has_forecast_file_access = has_forecast_file_access

        self._FORECAST_ADAPTERS = {
            Model.EPT2: self._v3_data_adapter,
        }

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

    def get_latest_metadata(self) -> ForecastMetadataResponse:
        return self._api.get_latest_forecast_metadata(model_name=self._model_name)

    def get_metadata(self, init_time: datetime | str | None = None):
        if init_time is None:
            return self.get_latest_metadata()

        return self._api.get_forecast_metadata(
            model_name=self._model_name, init_time=init_time
        )

    def get_available_init_times(self) -> list[datetime]:
        return self._api.get_available_init_times(model_name=self._model_name)

    def get_latest_forecast_file(self) -> JuaDataset:
        return self.get_forecast_file()

    @validate_call
    def get_forecast_file(self, init_time: datetime | None = None) -> JuaDataset:
        if not self._has_forecast_file_access:
            raise ValueError(
                "This model does not have forecast file access. Please check the model documentation."
            )

        if init_time is None:
            init_time = self.get_latest_metadata().init_time

        return self._FORECAST_ADAPTERS[self._model](init_time)

    def _open_dataset(self, url: str) -> xr.Dataset:
        return xr.open_dataset(
            url,
            engine="zarr",
            decode_timedelta=True,
            storage_options={"auth": self._client.settings.auth.get_basic_auth()},
        )

    def _v3_data_adapter(self, init_time: datetime) -> JuaDataset:
        data_base_url = self._client.settings.data_base_url
        model_name = self._MODEL_NAME_MAPPINGS[self._model]
        init_time_str = init_time.strftime("%Y%m%d%H")
        dataset_name = f"{init_time_str}.zarr"
        data_url = f"{data_base_url}/forecasts/{model_name}/{dataset_name}"

        raw_data = self._open_dataset(data_url)
        # Rename coordinate prediction_timedelta to leadtime
        raw_data = raw_data.rename({"prediction_timedelta": "lead_time"})
        return JuaDataset(dataset_name, raw_data=raw_data, model=self._model)
