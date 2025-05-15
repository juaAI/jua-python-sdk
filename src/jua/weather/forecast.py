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
from jua.weather.conversions import timedelta_to_hours, to_datetime
from jua.weather.models import Models
from jua.weather.variables import Variables, rename_to_ept2

logger = get_logger(__name__)


class Forecast:
    _MAX_INIT_TIME_PAST_FOR_API_H = 36
    _MAX_point_FOR_API = 25

    def __init__(self, client: JuaClient, model: Models):
        self._client = client
        self._model = model
        self._model_name = model.value
        self._model_meta = get_model_meta_info(model)
        self._api = WeatherAPI(client)

        self._FORECAST_ADAPTERS = {
            Models.EPT2: self._v3_data_adapter,
            Models.EPT1_5: self._v2_data_adapter,
            Models.EPT1_5_EARLY: self._v2_data_adapter,
        }

    @property
    def zarr_version(self) -> int | None:
        return self._model_meta.forecast_zarr_version

    def is_global_data_available(self) -> bool:
        return self._model in self._FORECAST_ADAPTERS

    def _get_latest_metadata(self) -> ForecastMetadataResponse:
        return self._api.get_latest_forecast_metadata(model_name=self._model_name)

    @validate_call
    def get_metadata(
        self, init_time: datetime | str = "latest"
    ) -> ForecastMetadataResponse | None:
        if init_time == "latest":
            return self._get_latest_metadata()

        if not self._model_meta.is_jua_model:
            logger.warning(
                f"Model {self._model_name} only supports loading the latest metadata"
            )
            return None

        return self._api.get_forecast_metadata(
            model_name=self._model_name, init_time=init_time
        )

    def get_available_init_times(self) -> list[datetime]:
        if not self._model_meta.is_jua_model:
            logger.warning(
                f"Model {self._model_name} only supports loading the latest forecast"
            )
            return []

        return self._api.get_available_init_times(model_name=self._model_name)

    @validate_call
    def is_ready(
        self, forecasted_hours: int, init_time: datetime | str = "latest"
    ) -> bool:
        maybe_metadata = self.get_metadata(init_time)
        if maybe_metadata is None:
            return False

        return maybe_metadata.available_forecasted_hours >= forecasted_hours

    def _rename_variables_for_api(
        self, variables: list[str] | list[Variables]
    ) -> list[str]:
        return [rename_to_ept2(v) for v in variables]

    def _dispatch_to_api(
        self,
        init_time: datetime | str = "latest",
        points: list[LatLon] | None = None,
        min_lead_time: int = 0,
        max_lead_time: int = 0,
        variables: list[str] | list[Variables] | None = None,
    ) -> ForecastData:
        if variables is not None:
            variables = self._rename_variables_for_api(variables)

        if init_time == "latest":
            return self._api.get_latest_forecast(
                model_name=self._model_name,
                payload=ForecastRequestPayload(
                    points=points,
                    min_lead_time=min_lead_time,
                    max_lead_time=max_lead_time,
                    variables=variables,
                ),
            )

        return self._api.get_forecast(
            init_time=init_time,
            model_name=self._model_name,
            payload=ForecastRequestPayload(
                points=points,
                min_lead_time=min_lead_time,
                max_lead_time=max_lead_time,
                variables=variables,
            ),
        )

    def _dispatch_to_data_adapter(
        self,
        init_time: datetime | str = "latest",
        variables: list[Variables] | list[str] | None = None,
        print_progress: bool | None = None,
        prediction_timedelta: PredictionTimeDelta = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        method: str | None = None,
    ):
        if not self.is_global_data_available():
            raise ModelDoesNotSupportForecastRawDataAccessError(self._model_name)

        if init_time == "latest":
            metadata = self.get_metadata()
            if metadata is None:
                raise ValueError("No metadata found for model")
            init_time = metadata.init_time

        return self._FORECAST_ADAPTERS[self._model](
            to_datetime(init_time),
            variables=variables,
            print_progress=print_progress,
            prediction_timedelta=prediction_timedelta,
            latitude=latitude,
            longitude=longitude,
            method=method,
        )

    def _can_be_dispatched_to_api(
        self,
        init_time: datetime | str,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | None = None,
    ) -> bool:
        if init_time == "latest":
            metadata = self.get_metadata()
            if metadata is None:
                raise ValueError("No metadata found for model")
            init_time = metadata.init_time

        init_time_dt = to_datetime(init_time)
        # Make now timezone-aware if init_time is timezone-aware
        now = (
            datetime.now(init_time_dt.tzinfo) if init_time_dt.tzinfo else datetime.now()
        )

        if (
            now - init_time_dt
        ).total_seconds() > self._MAX_INIT_TIME_PAST_FOR_API_H * 3600:
            return False

        if points is None and (latitude is None or longitude is None):
            return False

        if isinstance(latitude, slice) or isinstance(longitude, slice):
            return False

        if latitude is not None and longitude is not None:
            points = self._convert_lat_lon_to_point(latitude, longitude)

        assert points is not None, (
            "points are either provided or determined from latitude and longitude"
        )

        if not self._is_latest_init_time(init_time) and len(points) > 1:
            return False

        if len(points) > self._MAX_point_FOR_API:
            return False

        if prediction_timedelta is not None:
            if isinstance(prediction_timedelta, slice):
                if prediction_timedelta.step is not None:
                    return False

        return True

    def _convert_lat_lon_to_point(
        self,
        latitude: list[float] | float,
        longitude: list[float] | float,
    ) -> list[LatLon]:
        if isinstance(latitude, float):
            latitude = [latitude]
        if isinstance(longitude, float):
            longitude = [longitude]
        return [LatLon(lat, lon) for lat in latitude for lon in longitude]

    def _convert_prediction_timedelta_to_api_call(
        self,
        min_lead_time: int | None,
        max_lead_time: int | None,
        prediction_timedelta: PredictionTimeDelta | None,
    ) -> tuple[int, int]:
        # Default to 480 hours
        min_lead_time = min_lead_time or 0
        max_lead_time = max_lead_time or 480

        if isinstance(prediction_timedelta, slice):
            return prediction_timedelta.start, prediction_timedelta.stop

        if prediction_timedelta is not None:
            # Assume it is a scalar value
            return 0, timedelta_to_hours(prediction_timedelta)
        return min_lead_time, max_lead_time

    def _is_latest_init_time(self, init_time: datetime | str) -> bool:
        if init_time == "latest":
            return True
        init_time_dt = to_datetime(init_time)
        metadata = self.get_metadata()
        if metadata is None:
            return False
        return init_time_dt == metadata.init_time

    def _get_prediction_timedelta_for_adapter(
        self,
        min_lead_time: int | None,
        max_lead_time: int | None,
        prediction_timedelta: PredictionTimeDelta | None,
    ) -> PredictionTimeDelta:
        if prediction_timedelta is not None:
            return prediction_timedelta
        min_lead_time = min_lead_time or 0
        max_lead_time = max_lead_time or 480
        return slice(min_lead_time, max_lead_time)

    def _get_spatial_selection_for_adapter(
        self,
        latitude: SpatialSelection | None,
        longitude: SpatialSelection | None,
        points: list[LatLon] | None,
    ) -> tuple[SpatialSelection | None, SpatialSelection | None]:
        if points is None:
            return latitude, longitude
        lats, lons = zip(*[(p.lat, p.lon) for p in points])
        return list(lats), list(lons)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_forecast(
        self,
        init_time: datetime | str = "latest",
        variables: list[Variables] | list[str] | None = None,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | LatLon | None = None,
        min_lead_time: int | None = None,
        max_lead_time: int | None = None,
        method: str | None = "nearest",
        print_progress: bool | None = None,
    ) -> JuaDataset:
        if points is not None and (latitude is not None or longitude is not None):
            raise ValueError(
                "Cannot provide both points and latitude/longitude. "
                "Please provide either points or latitude/longitude."
            )
        if points is not None and not isinstance(points, list):
            points = [points]

        if self._can_be_dispatched_to_api(
            init_time=init_time,
            latitude=latitude,
            longitude=longitude,
            points=points,
        ):
            # As _can_be_dispatched_to_api has passed, either points are not none
            # or we can convert the latitude and longitude to points
            points = points or self._convert_lat_lon_to_point(latitude, longitude)  # type: ignore
            min_lead_time, max_lead_time = (
                self._convert_prediction_timedelta_to_api_call(
                    min_lead_time, max_lead_time, prediction_timedelta
                )
            )

            data = self._dispatch_to_api(
                init_time=init_time,
                points=points,
                variables=variables,
                min_lead_time=min_lead_time,
                max_lead_time=max_lead_time,
            )
            raw_data = data.to_xarray()
            if points is not None and raw_data is not None:
                raw_data = raw_data.select_point(points=points)
            dataset_name = f"{self._model_name}_{data.init_time.strftime('%Y%m%d%H')}"
            return JuaDataset(
                settings=self._client.settings,
                dataset_name=dataset_name,
                raw_data=raw_data,
                model=self._model,
            )

        logger.warning("Large query, this might take some time.")
        latitude, longitude = self._get_spatial_selection_for_adapter(
            latitude=latitude, longitude=longitude, points=points
        )
        prediction_timedelta = self._get_prediction_timedelta_for_adapter(
            min_lead_time=min_lead_time,
            max_lead_time=max_lead_time,
            prediction_timedelta=prediction_timedelta,
        )

        return self._dispatch_to_data_adapter(
            init_time=init_time,
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
            maybe_metadata = self.get_metadata(init_time=init_time)
            if maybe_metadata is None:
                max_available_hours = 480
            else:
                max_available_hours = maybe_metadata.available_forecasted_hours
        except Exception:
            max_available_hours = 480

        hours_to_load = list(range(max_available_hours + 1))
        prediction_timedelta = kwargs.get("prediction_timedelta", None)
        if prediction_timedelta is not None:
            if isinstance(prediction_timedelta, list):
                hours_to_load = [timedelta_to_hours(td) for td in prediction_timedelta]

            elif isinstance(prediction_timedelta, slice):
                hours_to_load = list(
                    range(
                        timedelta_to_hours(prediction_timedelta.start),
                        timedelta_to_hours(prediction_timedelta.stop),
                        timedelta_to_hours(prediction_timedelta.step or 1),
                    )
                )
            else:
                hours_to_load = [timedelta_to_hours(prediction_timedelta)]

            # Already handled above, remove from kwargs
            del kwargs["prediction_timedelta"]

        zarr_urls = [
            f"{data_base_url}/forecasts/{model_name}/{init_time_str}/{hour}.zarr"
            for hour in hours_to_load
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
