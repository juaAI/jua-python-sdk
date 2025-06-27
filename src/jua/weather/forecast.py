from datetime import datetime

import xarray as xr
from pydantic import validate_call

from jua._utils.dataset import DatasetConfig, open_dataset
from jua.client import JuaClient
from jua.errors.model_errors import ModelDoesNotSupportForecastRawDataAccessError
from jua.logging import get_logger
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather import JuaDataset
from jua.weather._api import WeatherAPI
from jua.weather._model_meta import get_model_meta_info
from jua.weather._types.api_payload_types import ForecastRequestPayload
from jua.weather._types.api_response_types import ForecastMetadataResponse
from jua.weather._types.forecast import ForecastData
from jua.weather.conversions import timedelta_to_hours, to_datetime
from jua.weather.models import Models
from jua.weather.statistics import Statistics
from jua.weather.variables import Variables, rename_to_ept2

logger = get_logger(__name__)


class Forecast:
    """Access to weather forecast data for a specific model.

    This class provides methods to retrieve and query weather forecast data
    from Jua's forecasting models. It supports both point-based queries and
    spatial area selection.

    The class is typically accessed via a Model instance.

    Examples:
        >>> from jua import JuaClient
        >>> from jua.weather import Models, Variables
        >>> from jua.types.geo import LatLon
        >>>
        >>> client = JuaClient()
        >>> model = client.weather.get_model(Models.EPT2)
        >>>
        >>> # Get latest global forecast
        >>> forecast = model.forecast.get_forecast()
        >>>
        >>> # Get forecast for specific points
        >>> zurich = LatLon(lat=47.3769, lon=8.5417)
        >>> london = LatLon(lat=51.5074, lon=-0.1278)
        >>> forecast = model.forecast.get_forecast(
        ...     points=[zurich, london],
        ...     variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
        ... )
    """

    _MAX_INIT_TIME_PAST_FOR_API_H = 36
    _MAX_point_FOR_API = 25

    def __init__(self, client: JuaClient, model: Models):
        """Initialize forecast access for a specific model.

        Args:
            client: JuaClient instance for authentication and settings.
            model: Weather model to access forecast data for.
        """
        self._client = client
        self._model = model
        self._model_name = model.value
        self._model_meta = get_model_meta_info(model)
        self._api = WeatherAPI(client)

        self._FORECAST_ADAPTERS = {
            Models.EPT2: self._v3_data_adapter,
            Models.EPT2_EARLY: self._v3_data_adapter,
            Models.AURORA: self._v3_data_adapter,
            Models.AIFS: self._v3_data_adapter,
            Models.EPT2_E: self._v3_data_adapter,
            Models.EPT1_5: self._v2_data_adapter,
            Models.EPT1_5_EARLY: self._v2_data_adapter,
        }

    def is_global_data_available(self) -> bool:
        """Check if global data access is available for this model.

        Some models only support point forecasts while others allow
        access to global forecast data.

        Returns:
            True if global data can be accessed, False otherwise.
        """
        return self._model in self._FORECAST_ADAPTERS

    def _get_latest_metadata(self) -> ForecastMetadataResponse:
        """Get metadata for the latest forecast of the current model.

        This is an internal helper method that delegates to the API client.

        Returns:
            Metadata about the latest forecast.
        """
        return self._api.get_latest_forecast_metadata(model_name=self._model_name)

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
        statistics: list[str] | list[Statistics] | None = None,
        method: str | None = "nearest",
        print_progress: bool | None = None,
    ) -> JuaDataset:
        """Retrieve forecast data for the model.

        This is the primary method for accessing weather forecast data. It supports
        multiple ways to specify both spatial selection (points, list or slice of
        latitude/longitude) and forecast time periods (prediction_timedelta or
        min/max lead times).

        The method automatically chooses the most efficient way to retrieve the data
        based on the query parameters.

        Args:
            init_time: Forecast initialization time. Use "latest" for the most recent
                forecast, or provide a specific datetime or ISO-format string.

            variables: List of weather variables to retrieve. If None, all available
                variables are included.

            prediction_timedelta: Time period to include in the forecast. Can be:
                - A single value (hours or timedelta) for a specific lead time
                - A slice(start, stop) for a range of lead times
                - A slice(start, stop, step) for lead times at regular intervals

            latitude: Latitude selection. Can be a single value, list of values, or
                a slice(min_lat, max_lat) for a range.

            longitude: Longitude selection. Can be a single value, list of values, or
                a slice(min_lon, max_lon) for a range.

            points: Specific geographic points to get forecasts for. Can be a single
                LatLon object or a list of LatLon objects.

            min_lead_time: Minimum lead time in hours
                (alternative to prediction_timedelta).

            max_lead_time: Maximum lead time in hours
                (alternative to prediction_timedelta).

            method: Interpolation method for selecting points:
                - "nearest" (default): Use nearest grid point
                - All other methods supported by xarray

            print_progress: Whether to display a progress bar during data loading.
                If None, uses the client's default setting.

        Returns:
            JuaDataset containing the requested forecast data.

        Raises:
            ValueError: If both points and latitude/longitude are provided, or if
                other parameter combinations are invalid.

        Examples:
            >>> # Get global forecast for temperature and wind speed
            >>> forecast = model.forecast.get_forecast(
            ...     variables=[
            ...         Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
            ...         Variables.WIND_SPEED_AT_HEIGHT_LEVEL_10M
            ...     ]
            ... )
            >>>
            >>> # Get forecast for a specific region (Europe)
            >>> europe = model.forecast.get_forecast(
            ...     latitude=slice(71, 36),  # North to South
            ...     longitude=slice(-15, 50),  # West to East
            ...     max_lead_time=120  # 5 days
            ... )
            >>>
            >>> # Get forecast for specific cities
            >>> cities = model.forecast.get_forecast(
            ...     points=[
            ...         LatLon(lat=40.7128, lon=-74.0060),  # New York
            ...         LatLon(lat=51.5074, lon=-0.1278),   # London
            ...         LatLon(lat=35.6762, lon=139.6503)   # Tokyo
            ...     ],
            ...     max_lead_time=72  # 3 days
            ... )
        """
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
                statistics=statistics,
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
            points=points,
            method=method,
        )

    @validate_call
    def get_metadata(
        self, init_time: datetime | str = "latest"
    ) -> ForecastMetadataResponse | None:
        """Get metadata about a forecast.

        Retrieves information about a specific forecast, including initialization time,
        available forecasted hours, and other model-specific metadata.

        This metadata is useful for checking availability and planning data
        retrieval operations.

        Args:
            init_time: The initialization time of the forecast. Use "latest" for the
                most recent forecast, or provide a specific datetime or string in
                ISO format (e.g., "2023-01-15T12:00:00"). Must be an exact match.

        Returns:
            Metadata about the forecast or None if not available.

        Examples:
            >>> metadata = model.forecast.get_metadata()
            >>> print(f"Latest forecast from: {metadata.init_time}")
            >>> print(f"Hours available: {metadata.available_forecasted_hours}")
        """
        if init_time == "latest":
            return self._get_latest_metadata()

        if not self._model_meta.has_forecast_file_access:
            logger.warning(
                f"Model {self._model_name} only supports loading the latest metadata"
            )
            return None

        return self._api.get_forecast_metadata(
            model_name=self._model_name, init_time=init_time
        )

    def get_available_init_times(self) -> list[datetime]:
        """Get a list of available forecast initialization times.

        Retrieves the list of initialization times for which forecasts are
        available for the current model.

        Note:
            For non-Jua models, only the latest forecast is typically available.
            For EPT1.5 and EPT1.5 Early models, this returns initialization times
            that are currently cached.
            For EPT2, this returns all initialization times available in the database.

        Returns:
            A list of datetime objects representing available forecast init times,
            sorted from most recent to oldest.

        Examples:
            >>> init_times = model.forecast.get_available_init_times()
            >>> print(f"Most recent forecast: {init_times[0]}")
            >>> print(f"Total forecasts available: {len(init_times)}")
        """
        if not self._model_meta.has_forecast_file_access:
            logger.warning(
                f"Model {self._model_name} only supports loading the latest forecast"
            )
            return []

        return self._api.get_available_init_times(model_name=self._model_name)

    @validate_call
    def is_ready(
        self, forecasted_hours: int, init_time: datetime | str = "latest"
    ) -> bool:
        """Check if a forecast is ready up to a specific lead time.

        This method is useful for checking if a forecast has been processed
        up to a certain number of hours into the future. Forecasts may become
        available incrementally, with longer lead times becoming available
        as processing completes.

        Args:
            forecasted_hours: The number of forecast hours needed.
            init_time: The initialization time of the forecast to check.
                Use "latest" for the most recent forecast, or provide a specific
                datetime or string in ISO format. Must be an exact match.

        Returns:
            True if the forecast is available for the specified hours, False otherwise.

        Examples:
            >>> # Check if 10-day forecast is ready
            >>> is_ten_day_ready = model.forecast.is_ready(240)
            >>> if is_ten_day_ready:
            >>>     # Now we can safely request 10-day forecast data
            >>>     forecast = model.forecast.get_forecast(max_lead_time=240)
        """
        maybe_metadata = self.get_metadata(init_time)
        if maybe_metadata is None:
            return False

        return maybe_metadata.available_forecasted_hours >= forecasted_hours

    def _rename_variables_for_api(
        self, variables: list[str] | list[Variables]
    ) -> list[str]:
        """Convert variable identifiers to the format expected by the API.

        This internal method ensures variables are correctly formatted for API requests
        by standardizing identifiers across model versions.

        Args:
            variables: List of variable identifiers (string names or Variables enums)

        Returns:
            List of variable names formatted for API requests
        """
        return [rename_to_ept2(v) for v in variables]

    def _dispatch_to_api(
        self,
        init_time: datetime | str = "latest",
        points: list[LatLon] | None = None,
        min_lead_time: int = 0,
        max_lead_time: int = 0,
        variables: list[str] | list[Variables] | None = None,
        statistics: list[str] | list[Statistics] | None = None,
    ) -> ForecastData:
        """Send a forecast request to the API.

        This internal method handles direct API communication for point forecasts,
        handling both latest and historical forecasts.

        Args:
            init_time: Forecast initialization time
            points: List of geographic points to get forecasts for
            min_lead_time: Minimum lead time in hours
            max_lead_time: Maximum lead time in hours
            variables: List of weather variables to retrieve
            statistics: List of statistics to retrieve

        Returns:
            Forecast data response from the API
        """
        if variables is not None:
            variables = self._rename_variables_for_api(variables)

        if statistics is not None:
            statistics = [
                s.key if (isinstance(s, Statistics)) else s for s in statistics
            ]

        if init_time == "latest":
            return self._api.get_latest_forecast(
                model_name=self._model_name,
                payload=ForecastRequestPayload(
                    points=points,
                    min_lead_time=min_lead_time,
                    max_lead_time=max_lead_time,
                    variables=variables,
                    ensemble_stats=statistics,
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
                ensemble_stats=statistics,
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
        points: list[LatLon] | None = None,
        method: str | None = None,
    ):
        """Retrieve forecast data using the appropriate data adapter.

        This internal method handles large spatial queries that require direct
        access to forecast data files rather than API requests.

        Args:
            init_time: Forecast initialization time
            variables: List of weather variables to retrieve
            print_progress: Whether to display a progress bar
            prediction_timedelta: Time range to include in the forecast
            latitude: Latitude selection (point, list, or slice)
            longitude: Longitude selection (point, list, or slice)
            method: Interpolation method for selecting points

        Returns:
            JuaDataset containing the requested forecast data

        Raises:
            ModelDoesNotSupportForecastRawDataAccessError:
                If the model doesn't support raw data access
            ValueError:
                If no metadata is found for the model
        """
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
            points=points,
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
        """Determine if the request can be served via the API.

        The API has certain limitations on what kinds of queries it can handle.
        This method checks if the request falls within those limitations.

        Constraints include:
        - Forecast age (must be recent, less than 36 hours)
        - Number of points (limited to _MAX_point_FOR_API)
        - Query structure (slices not supported)
        - Lead time structure (complex lead time slicing not supported)

        Args:
            init_time: Forecast initialization time
            prediction_timedelta: Time range to include in the forecast
            latitude: Latitude selection (point, list, or slice)
            longitude: Longitude selection (point, list, or slice)
            points: List of geographic points to get forecasts for
            statistics: List of statistics to retrieve

        Returns:
            True if the request can be dispatched to the API, False otherwise

        Raises:
            ValueError: If no metadata is found for the model
        """
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
        """Convert separate latitude and longitude values to LatLon objects.

        This internal method creates all possible combinations of lat/lon values
        as individual points.

        Args:
            latitude: Single latitude value or list of values
            longitude: Single longitude value or list of values

        Returns:
            List of LatLon objects representing all combinations
        """
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
        """Convert various time specifications to min/max lead time format.

        This internal method normalizes different ways of specifying forecast periods
        (prediction_timedelta, min/max lead times) to a single format used by the API.

        Args:
            min_lead_time: Minimum lead time in hours
            max_lead_time: Maximum lead time in hours
            prediction_timedelta: Alternative specification of forecast period

        Returns:
            Tuple of (min_lead_time, max_lead_time) in hours
        """
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
        """Check if the specified init time is the latest available.

        This internal method determines if the requested initialization time
        matches the latest available forecast.

        Args:
            init_time: Forecast initialization time to check

        Returns:
            True if it's the latest init time, False otherwise
        """
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
        """Convert lead time parameters to prediction_timedelta format.

        This internal method creates a suitable prediction_timedelta representation
        for the data adapters from various time specifications.

        Args:
            min_lead_time: Minimum lead time in hours
            max_lead_time: Maximum lead time in hours
            prediction_timedelta: Existing prediction timedelta specification

        Returns:
            A PredictionTimeDelta (typically a slice) for data adapters
        """
        if prediction_timedelta is not None:
            return prediction_timedelta
        min_lead_time = min_lead_time or 0
        max_lead_time = max_lead_time or 480
        return slice(min_lead_time, max_lead_time)

    def _open_dataset(
        self, url: str | list[str], print_progress: bool | None = None, **kwargs
    ) -> xr.Dataset:
        """Open a dataset from a URL or list of URLs.

        This internal helper method handles opening datasets with appropriate
        chunking and progress display.

        Args:
            url: URL or list of URLs to open
            print_progress: Whether to display a progress bar
            **kwargs: Additional arguments for dataset opening

        Returns:
            Opened xarray Dataset
        """
        if isinstance(url, str):
            url = [url]
        dataset_configs = [
            DatasetConfig(
                path=u,
            )
            for u in url
        ]
        return open_dataset(
            self._client,
            dataset_config=dataset_configs,
            should_print_progress=print_progress,
            **kwargs,
        )

    def _v3_data_adapter(
        self, init_time: datetime, print_progress: bool | None = None, **kwargs
    ) -> JuaDataset:
        """Adapter for EPT2 (and similar) forecast data access.

        This internal adapter handles retrieving data for models that use
        the v3 Zarr storage format (a single consolidated Zarr store).

        Args:
            init_time: Forecast initialization time
            print_progress: Whether to display a progress bar
            **kwargs: Additional selection parameters

        Returns:
            JuaDataset containing the requested forecast data
        """
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
        """Adapter for EPT1.5 (and similar) forecast data access.

        This internal adapter handles retrieving data for models that use
        the v2 Zarr storage format (separate Zarr stores for each lead time).

        Args:
            init_time: Forecast initialization time
            print_progress: Whether to display a progress bar
            **kwargs: Additional selection parameters including:
                - prediction_timedelta: Time period to include
                - latitude/longitude: Spatial selection
                - method: Interpolation method

        Returns:
            JuaDataset containing the requested forecast data
        """
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
