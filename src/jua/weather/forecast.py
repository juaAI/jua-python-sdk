from datetime import datetime

import xarray as xr
from pydantic import validate_call

from jua._utils.dataset import open_dataset
from jua.client import JuaClient
from jua.errors.model_errors import ModelDoesNotSupportForecastRawDataAccessError
from jua.logging import get_logger
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather._api import WeatherAPI
from jua.weather._jua_dataset import JuaDataset
from jua.weather._model_meta import get_model_meta_info
from jua.weather._types.api_payload_types import ForecastRequestPayload
from jua.weather._types.api_response_types import ForecastMetadataResponse
from jua.weather._types.forecast import ForecastData
from jua.weather.models import Models
from jua.weather.variables import Variables

logger = get_logger(__name__)


class Forecast:
    def __init__(self, client: JuaClient, model: Models):
        self._client = client
        self._model = model
        self._model_name = model.value
        self._api = WeatherAPI(client)

        self._FORECAST_ADAPTERS = {
            Models.EPT2: self._v3_data_adapter,
            Models.EPT1_5: self._v2_data_adapter,
            Models.EPT1_5_EARLY: self._v2_data_adapter,
        }

    @property
    def zarr_version(self) -> int | None:
        return get_model_meta_info(self._model).forecast_zarr_version

    def is_file_access_available(self) -> bool:
        return self._model in self._FORECAST_ADAPTERS

    def get_latest(
        self,
        lat: float | None = None,
        lon: float | None = None,
        points: list[LatLon] | None = None,
        min_lead_time: int = 0,
        max_lead_time: int = 0,
        variables: list[str] | None = None,
        full: bool = False,
    ) -> ForecastData:
        return self._api.get_latest_forecast(
            model_name=self._model_name,
            lat=lat,
            lon=lon,
            payload=ForecastRequestPayload(
                points=points,
                min_lead_time=min_lead_time,
                max_lead_time=max_lead_time,
                variables=variables,
                full=full,
            ),
        )

    def get(
        self,
        init_time: datetime | str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        points: list[LatLon] | None = None,
        min_lead_time: int = 0,
        max_lead_time: int = 0,
        variables: list[str] | None = None,
        full: bool = False,
    ) -> ForecastData:
        if not init_time:
            return self.get_latest(
                lat=lat,
                lon=lon,
                points=points,
                min_lead_time=min_lead_time,
                max_lead_time=max_lead_time,
                variables=variables,
                full=full,
            )

        return self._api.get_forecast(
            init_time=init_time,
            model_name=self._model_name,
            lat=lat,
            lon=lon,
            payload=ForecastRequestPayload(
                points=points,
                min_lead_time=min_lead_time,
                max_lead_time=max_lead_time,
                variables=variables,
                full=full,
            ),
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

    # TODO: is_ready(init_time, forecast_horizon)

    def get_latest_forecast_as_dataset(
        self,
        print_progress: bool | None = None,
        variables: list[Variables] | list[str] | None = None,
        prediction_timedelta: PredictionTimeDelta = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        method: str | None = None,
    ) -> JuaDataset:
        return self.get_forecast_as_dataset(
            variables=variables,
            print_progress=print_progress,
            prediction_timedelta=prediction_timedelta,
            latitude=latitude,
            longitude=longitude,
            method=method,
        )

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_forecast_as_dataset(
        self,
        variables: list[Variables] | list[str] | None = None,
        init_time: datetime | None = None,
        print_progress: bool | None = None,
        prediction_timedelta: PredictionTimeDelta = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        method: str | None = None,
    ) -> JuaDataset:
        if not self.is_file_access_available():
            raise ModelDoesNotSupportForecastRawDataAccessError(self._model_name)

        if init_time is None:
            init_time = self.get_latest_metadata().init_time

        return self._FORECAST_ADAPTERS[self._model](
            init_time,
            variables=variables,
            print_progress=print_progress,
            prediction_timedelta=prediction_timedelta,
            latitude=latitude,
            longitude=longitude,
            method=method,
        )

    def _open_dataset(
        self, url: str | list[str], print_progress: bool | None = None, **kwargs
    ) -> xr.Dataset:
        model_meta = get_model_meta_info(self._model)

        return open_dataset(
            self._client,
            url,
            should_print_progress=print_progress,
            chunks=model_meta.forecast_chunks,
            **kwargs,
        )

    def _v3_data_adapter(
        self, init_time: datetime, print_progress: bool | None = None, **kwargs
    ) -> JuaDataset:
        data_base_url = self._client.settings.data_base_url
        model_name = get_model_meta_info(self._model).forecast_name_mapping
        init_time_str = init_time.strftime("%Y%m%d%H")
        dataset_name = f"{init_time_str}"
        data_url = f"{data_base_url}/forecasts/{model_name}/{dataset_name}.zarr"

        raw_data = self._open_dataset(data_url, print_progress=print_progress, **kwargs)
        return JuaDataset(
            settings=self._client.settings,
            dataset_name=dataset_name,
            raw_data=raw_data,
            model=self._model,
        )

    def _v2_data_adapter(
        self,
        init_time: datetime,
        print_progress: bool | None = None,
        **kwargs,
    ) -> JuaDataset:
        data_base_url = self._client.settings.data_base_url
        model_name = get_model_meta_info(self._model).forecast_name_mapping
        init_time_str = init_time.strftime("%Y%m%d%H")
        # This is a bit hacky:
        # For EPT1.5, get_metadata will result in an error
        # if the forecast is no longer in cache.
        # For now, we try and if it fails default to 480 hours
        try:
            max_available_hours = self.get_metadata(
                init_time=init_time
            ).available_forecasted_hours
        except Exception:
            max_available_hours = 480

        zarr_urls = [
            f"{data_base_url}/forecasts/{model_name}/{init_time_str}/{hour}.zarr"
            for hour in range(max_available_hours + 1)
        ]

        dataset_name = f"{init_time_str}"
        raw_data = self._open_dataset(
            zarr_urls, print_progress=print_progress, **kwargs
        )
        return JuaDataset(
            settings=self._client.settings,
            dataset_name=dataset_name,
            raw_data=raw_data,
            model=self._model,
        )
