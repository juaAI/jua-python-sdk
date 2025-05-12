from datetime import datetime

from pydantic import validate_call

from jua._api import API
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua.client import JuaClient
from jua.errors.jua_error import JuaError
from jua.types.geo import LatLon
from jua.weather._types.api_payload_types import ForecastRequestPayload
from jua.weather._types.api_response_types import (
    AvailableInitTimesResponse,
    AvailableModelsResponse,
    ForecastMetadataResponse,
    ForecastResponse,
)
from jua.weather._types.forecast import ForecastData
from jua.weather._types.raw_file_access import DirectoryResponse, FileResponse
from jua.weather.conversions import validate_init_time


class WeatherAPI:
    _FORECASTING_METADATA_ENDPOINT = "forecasting"
    _AVAILABLE_INIT_TIMES_ENDPOINT = (
        "forecasting/{model_name}/forecasts/available_init_times"
    )
    _LATEST_FORECAST_ENDPOINT = "forecasting/{model_name}/forecasts/latest"
    _FORECAST_ENDPOINT = "forecasting/{model_name}/forecasts/{init_time}"
    _FORECAST_ENDPOINT_LAT_LON = (
        "forecasting/{model_name}/forecasts/{init_time}/{lat},{lon}"
    )
    _BROWSE_FILES_ENDPOINT = "files/browse"

    def __init__(self, jua_client: JuaClient):
        self._api = API(jua_client)

    def _encode_init_time(self, init_time: datetime) -> str:
        # Format should be "2025-05-11T12:43:14.456Z"
        return init_time.isoformat() + "Z"

    def _get_payload_raise_error(
        self,
        lat: float | None,
        lon: float | None,
        payload: ForecastRequestPayload | None,
    ) -> ForecastRequestPayload:
        if (lat is None and lon is None) and (
            payload is None or payload.points is None
        ):
            raise ValueError("Either lat and lon or payload must be provided")
        if (lat is not None and lon is not None) and (
            payload is not None and payload.points is not None
        ):
            raise ValueError("Only one of lat and lon or payload must be provided")
        if lat is not None and lon is not None:
            if payload is None:
                payload = ForecastRequestPayload(points=[LatLon(lat=lat, lon=lon)])
            else:
                payload.points = [LatLon(lat=lat, lon=lon)]
        # Add this point, payload must be non-None
        if payload is None:
            raise ValueError("Payload must be non-None")
        return payload

    @validate_call
    def get_available_models(self) -> list[str]:
        response = self._api.get(self._FORECASTING_METADATA_ENDPOINT)
        response_json = response.json()
        response = AvailableModelsResponse(**response_json)
        return response.available_models

    @validate_call
    def get_available_init_times(self, model_name: str) -> list[datetime]:
        response = self._api.get(
            self._AVAILABLE_INIT_TIMES_ENDPOINT.format(model_name=model_name)
        )
        response_json = response.json()
        response = AvailableInitTimesResponse(**response_json)
        return response.init_times

    @validate_call
    def get_latest_forecast_metadata(self, model_name: str) -> ForecastMetadataResponse:
        response = self._api.get(
            self._LATEST_FORECAST_ENDPOINT.format(model_name=model_name)
        )
        response_json = response.json()
        response = ForecastMetadataResponse(**response_json)
        return response

    @validate_call
    def get_latest_forecast(
        self,
        model_name: str,
        lat: float | None = None,
        lon: float | None = None,
        payload: ForecastRequestPayload | None = None,
    ) -> ForecastData:
        payload = self._get_payload_raise_error(lat, lon, payload)
        response = self._api.post(
            self._LATEST_FORECAST_ENDPOINT.format(model_name=model_name),
            data=remove_none_from_dict(payload.model_dump()),
        )
        response_json = response.json()
        response = ForecastResponse(**response_json)
        return response.forecast

    @validate_call
    def get_forecast_metadata(
        self, model_name: str, init_time: datetime | str
    ) -> ForecastMetadataResponse:
        init_time = validate_init_time(init_time)
        init_time_str = init_time.isoformat()
        response = self._api.get(
            self._FORECAST_ENDPOINT.format(
                model_name=model_name, init_time=init_time_str
            )
        )
        response_json = response.json()
        response = ForecastMetadataResponse(**response_json)
        return response

    @validate_call
    def get_forecast(
        self,
        model_name: str,
        lat: float | None = None,
        lon: float | None = None,
        payload: ForecastRequestPayload | None = None,
        init_time: datetime | str | None = None,
    ) -> ForecastData:
        if init_time is None:
            return self.get_latest_forecast(model_name, lat, lon, payload)

        payload = self._get_payload_raise_error(lat, lon, payload)
        init_time = validate_init_time(init_time)
        payload_points = payload.points
        if len(payload_points) != 1:
            raise JuaError("Exactly one point is supported for past forecasts")
        lat = payload_points[0].lat
        lon = payload_points[0].lon

        params = remove_none_from_dict(payload.model_dump())
        del params["points"]

        response = self._api.get(
            self._FORECAST_ENDPOINT_LAT_LON.format(
                model_name=model_name, init_time=init_time, lat=lat, lon=lon
            ),
            params=params,
        )
        response_json = response.json()
        response = ForecastResponse(**response_json)
        return response.forecast

    @validate_call
    def browse_files(self, path: str = "") -> list[FileResponse | DirectoryResponse]:
        response = self._api.get(self._BROWSE_FILES_ENDPOINT, params={"path": path})
        response_json = response.json()
        if "contents" in response_json:
            return [
                DirectoryResponse(**content) for content in response_json["contents"]
            ]
        return [FileResponse(**response_json)]
