from datetime import datetime

from pydantic import validate_call

from jua._api import API
from jua._utils.dataset import DatasetConfig
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
    ListDatasetsResponse,
)
from jua.weather._types.forecast import ForecastData
from jua.weather._types.raw_file_access import DirectoryResponse, FileResponse
from jua.weather.conversions import validate_init_time


class WeatherAPI:
    """Internal API client for Jua's weather services.

    Note:
        This class is intended for internal use only and should not be used directly.
        End users should interact with the higher-level classes like Forecast and
        Hindcast.
    """

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
    _HINDCAST_FILES_ENDPOINT = "hindcasts/sdk/files/{model_name}"

    def __init__(self, jua_client: JuaClient):
        """Initialize the weather API client.

        Args:
            jua_client: JuaClient instance for authentication and settings.
        """
        self._api = API(jua_client)

    def _encode_init_time(self, init_time: datetime) -> str:
        """Convert a datetime to the API's expected ISO format.

        Args:
            init_time: The datetime to encode.

        Returns:
            ISO formatted string with 'Z' suffix for UTC.
        """
        # Format should be "2025-05-11T12:43:14.456Z"
        return init_time.isoformat() + "Z"

    def _get_payload_raise_error(
        self,
        lat: float | None,
        lon: float | None,
        payload: ForecastRequestPayload | None,
    ) -> ForecastRequestPayload:
        """Validate and construct a forecast request payload.

        Handles various combinations of input parameters to create a valid payload.

        Args:
            lat: Optional latitude for a point forecast.
            lon: Optional longitude for a point forecast.
            payload: Optional pre-constructed payload object.

        Returns:
            A valid ForecastRequestPayload.

        Raises:
            ValueError: If the inputs cannot produce a valid payload.
        """
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
        # Add this points, payload must be non-None
        if payload is None:
            raise ValueError("Payload must be non-None")
        return payload

    @validate_call
    def get_available_models(self) -> list[str]:
        """Get a list of available weather model names.

        Returns:
            List of weather model names available through the API for the user.
        """
        response = self._api.get(self._FORECASTING_METADATA_ENDPOINT)
        response_json = response.json()
        response = AvailableModelsResponse(**response_json)
        return response.available_models

    @validate_call
    def get_available_init_times(self, model_name: str) -> list[datetime]:
        """Get a list of available forecast initialization times for a model.

        Note some limitations of this function:
        * For EPT1.5 and EPT1.5 Early this only returns the init times that are cached
        * For EPT2 it returns all init times that are available in the Database
        * All other models only support the latest forecast

        Args:
            model_name: The name of the weather model.

        Returns:
            List of datetime objects for available forecast initialization times.
        """
        response = self._api.get(
            self._AVAILABLE_INIT_TIMES_ENDPOINT.format(model_name=model_name)
        )
        response_json = response.json()
        response = AvailableInitTimesResponse(**response_json)
        return response.init_times

    @validate_call
    def get_latest_forecast_metadata(self, model_name: str) -> ForecastMetadataResponse:
        """Get metadata for the latest forecast of a model.

        Args:
            model_name: The name of the weather model.

        Returns:
            Metadata about the latest forecast.
        """
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
        """Get the latest forecast for a model.

        Args:
            model_name: The name of the weather model.
            lat: Optional latitude for a point forecast.
            lon: Optional longitude for a point forecast.
            payload: Optional pre-constructed payload with more detailed parameters.

        Returns:
            Forecast data.

        Raises:
            ValueError: If the location parameters are invalid.
        """
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
        """Get metadata for a specific forecast.

        Args:
            model_name: The name of the weather model.
            init_time: The initialization time of the forecast.

        Returns:
            Metadata about the forecast.
        """
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
        """Get a forecast for a specific model and initialization time.

        Args:
            model_name: The name of the weather model.
            lat: Optional latitude for a point forecast.
            lon: Optional longitude for a point forecast.
            payload: Optional pre-constructed payload with more detailed parameters.
            init_time: The initialization time of the forecast.
                If None, gets the latest.

        Returns:
            Forecast data.

        Raises:
            ValueError: If the location parameters are invalid.
            JuaError: If more than one point is provided for past forecasts.
        """
        if init_time is None:
            return self.get_latest_forecast(model_name, lat, lon, payload)

        payload = self._get_payload_raise_error(lat, lon, payload)
        init_time = validate_init_time(init_time)
        payload_points = payload.points or []
        if len(payload_points) != 1:
            raise JuaError("Exactly one points is supported for past forecasts")
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
        """Browse files available in the Jua storage.

        Note: Browsing files is slow and is currently not used in the higher level
        interfaces.

        Args:
            path: The path to browse, defaults to the root.

        Returns:
            List of file and directory information.
        """
        response = self._api.get(self._BROWSE_FILES_ENDPOINT, params={"path": path})
        response_json = response.json()
        if "contents" in response_json:
            return [
                DirectoryResponse(**content) for content in response_json["contents"]
            ]
        return [FileResponse(**response_json)]

    @validate_call
    def get_hindcast_files(self, model_name: str) -> list[DatasetConfig]:
        """Get the list of hindcast files for the current model.

        Returns:
            List of hindcast file URLs.
        """
        response = self._api.get(
            self._HINDCAST_FILES_ENDPOINT.format(model_name=model_name)
        )
        response_json = response.json()
        response = ListDatasetsResponse(**response_json)
        return response.files
