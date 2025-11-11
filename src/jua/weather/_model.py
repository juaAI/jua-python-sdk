import warnings
from datetime import UTC, datetime
from typing import Literal

import xarray as xr
from pydantic import validate_call

from jua.client import JuaClient
from jua.logging import get_logger
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather import JuaDataset
from jua.weather._model_meta import get_model_meta_info
from jua.weather._query_engine import QueryEngine
from jua.weather._types.pagination import Pagination
from jua.weather._types.query_response_types import (
    AvailableForecasts,
    LatestForecastInfo,
    ModelMetadata,
)
from jua.weather.forecast import Forecast
from jua.weather.hindcast import Hindcast
from jua.weather.models import Models as ModelEnum
from jua.weather.statistics import Statistics
from jua.weather.variables import Variables

logger = get_logger(__name__)


class Model:
    """Represents a specific Jua weather model with access to its data.

    A Model provides unified access to both forecast and hindcast data for a
    specific weather model. Each model has unique characteristics such as spatial
    resolution, update frequency, and forecast horizon.

    Attributes:
        _client: The JuaClient instance used for API communication.
        _model: The model identifier enum value.
        _forecast: Pre-initialized Forecast instance for this model.
        _hindcast: Pre-initialized Hindcast instance for this model.

    Examples:
        >>> from jua import JuaClient
        >>> from jua.weather import Models
        >>> client = JuaClient()
        >>> model = client.weather.get_model(Models.EPT2)
        >>>
        >>> # New method: access a 5-day forecast for all of europe from the model:
        >>> data = model.get_forecasts(
        ...     init_time=datetime(2024, 8, 5, 0),
        ...     latitude=slice(72, 36),
        ...     longitude=slice(-15, 35),
        ...     max_lead_time=5 * 24,
        ...     variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
        ... )
        >>> ds_forecast = data.to_xarray()
        >>>
        >>> # Access forecast data
        >>> forecast = model.forecast.get_forecast()
        >>>
        >>> # Access hindcast (historical) data
        >>> hindcast = model.hindcast.get_hindcast(init_time="2023-05-01")
    """

    def __init__(
        self,
        client: JuaClient,
        model: ModelEnum,
    ):
        """Initialize a weather model instance.

        Args:
            client: JuaClient instance for API communication.
            model: The model identifier (from Models enum).
        """
        self._client = client
        self._model = model

        self._query_engine = QueryEngine(jua_client=self._client)
        self._forecast = Forecast(
            client,
            model=model,
        )
        self._hindcast = Hindcast(
            client,
            model=model,
        )

    @property
    def name(self) -> str:
        """Get the string name of the model.

        Returns:
            The model name as a string.
        """
        return self._model.value

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_forecasts(
        self,
        init_time: Literal["latest"] | datetime | list[datetime] | slice | None = None,
        variables: list[Variables] | list[str] | None = None,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | LatLon | None = None,
        min_lead_time: int | None = None,
        max_lead_time: int | None = None,
        statistics: list[str] | list[Statistics] | None = None,
        method: Literal["nearest", "bilinear"] = "nearest",
        stream: bool | None = None,
        print_progress: bool | None = None,
        lazy_load: bool = False,
    ) -> JuaDataset:
        """Retrieve forecasts for this model.

        This method loads weather data from any model run, allowing to fetch the latest
        forecast as well as obtaining data for analysis of historical forecasts and
        verification against actual observations.

        There is currently no lazy-loading for this method, meaning that all requested
        data will be downloaded once a call is made.

        You can filter the forecasts by:
        - Time period (init_time)
        - Geographic area (latitude/longitude or points)
        - Lead time (prediction_timedelta or min/max_lead_time)
        - Weather variables (variables)

        Args:
            init_time: Filter by forecast initialization time. Can be:
                - None or 'latest' (default): The latest available forecast
                - A single datetime: Specific initialization time
                - A list of datetimes: Multiple specific times
                - A slice(start, end): Range of initialization times

            variables: List of weather variables to include. If None, returns only
                `Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M`.

            prediction_timedelta: Filter by forecast lead time. Can be:
                - None: All available lead times (default)
                - A single value (hours or timedelta): Specific lead time
                - A slice(start, stop): Range of lead times
                - A slice(start, stop, step): Lead times at regular intervals

            latitude: Latitude selection. Can be a single value, list of values, or
                a slice(min_lat, max_lat) for a geographical range.

            longitude: Longitude selection. Can be a single value, list of values, or
                a slice(min_lon, max_lon) for a geographical range.

            points: Specific geographic points to get forecasts for. Can be a single
                LatLon object or a list of LatLon objects (alternative to latitude,
                longitude).

            min_lead_time: Minimum lead time in hours
                (alternative to prediction_timedelta).

            max_lead_time: Maximum lead time in hours
                (alternative to prediction_timedelta).

            statistics: For ensemble models, the statistics to return.

            method: Interpolation method for selecting points:
                - "nearest": Use nearest grid point (default).
                - "bilinear": Bilinear interpolation to the selected point.

            stream: Whether to stream the response content. Recommended when querying
                slices or large amounts of data. Default is set to False for points,
                and True for grid slices. Streaming does not support method="bilinear"
                when requesting points.

            print_progress: Whether to display a progress bar during data loading.
                If None, uses the client's default setting.

            lazy_load: Warning - Experimental feature. Requests using lazy-loading may
                incur higher credit costs than the same request made directly.
                - Does not load data into memory until necessary. The JuaDataset
                    returned contains an xarray Dataset with lazy-loaded data.
                - Only available when loading slices of latitudes/longitudes. Point
                    forecasts cannot use lazy loading.
                - The JuaClient used to load data must have a `request_credit_limit`
                    high enough to load the entire created Dataset at once, even though
                    no credits are charged until data is actually loaded.

        Returns:
            JuaDataset containing the forecast data matching your selection criteria.

        Raises:
            ValueError: If incompatible parameter combinations are provided.

        Examples:
            >>> # Get the 48-hour forecasts for Europe for a week in August 2024
            >>> from datetime import datetime
            >>> model = client.weather.get_model(Models.EPT2)
            >>> europe_august_2024 = model.get_forecasts(
            ...     init_time=slice(
            ...         datetime(2024, 8, 5, 0),
            ...         datetime(2024, 8, 11, 18),
            ...     ),
            ...     latitude=slice(72, 36),
            ...     longitude=slice(-15, 35),
            ...     max_lead_time=48,
            ...     variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
            ... )
            >>>
            >>> # Get forecasts for specific cities with 24-hour lead time
            >>> from datetime import datetime
            >>> cities_data = model.get_forecasts(
            ...     init_time=slice(datetime(2024, 8, 5, 0), datetime(2024, 8, 5, 18)),
            ...     points=[
            ...         LatLon(lat=40.7128, lon=-74.0060),  # New York
            ...         LatLon(lat=51.5074, lon=-0.1278),   # London
            ...     ],
            ...     max_lead_time=24,
            ... )
        """
        if statistics:
            self._check_model_has_stats()

        if variables is None:
            variables = [Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
        else:
            variables = [
                v.name if isinstance(v, Variables) else str(v) for v in variables
            ]

        if prediction_timedelta is None:
            if min_lead_time is not None and max_lead_time is not None:
                prediction_timedelta = slice(min_lead_time, max_lead_time)  # type: ignore
            elif min_lead_time is not None:
                prediction_timedelta = slice(min_lead_time, 60 * 24)  # type: ignore
            elif max_lead_time is not None:
                prediction_timedelta = slice(0, max_lead_time)

        if lazy_load:
            if statistics:
                raise ValueError(f"Cannot `lazy_load` with stats: {statistics}.")
            if points is not None:
                raise ValueError("Cannot `lazy_load` points: load the data directly.")
            if not isinstance(latitude, slice):
                raise ValueError(
                    f"Can only `lazy_load` when latitude is a slice, not {latitude}."
                )
            if not isinstance(longitude, slice):
                raise ValueError(
                    f"Can only `lazy_load` when longitude is a slice, not {longitude}."
                )

            if prediction_timedelta is None and (
                min_lead_time is not None or max_lead_time is not None
            ):
                prediction_timedelta = slice(
                    0 if min_lead_time is None else min_lead_time,
                    24 * 120 if max_lead_time is None else max_lead_time,
                )

            return JuaDataset(
                settings=self._client.settings,
                dataset_name=self._model,
                raw_data=xr.open_dataset(
                    self._model,
                    variables=variables,
                    query_engine=self._query_engine,
                    init_time=init_time,
                    prediction_timedelta=prediction_timedelta,
                    latitude=latitude,
                    longitude=longitude,
                ),
                model=self._model,
            )

        raw_data = self._query_engine.get_forecast(
            model=self._model,
            init_time=init_time,
            variables=variables,
            prediction_timedelta=prediction_timedelta,
            latitude=latitude,
            longitude=longitude,
            points=points,
            statistics=statistics,
            method=method,
            stream=stream,
            print_progress=print_progress,
        )
        return JuaDataset(
            settings=self._client.settings,
            dataset_name=self._model,
            raw_data=raw_data,
            model=self._model,
        )

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
            >>> # Check if 10-day forecast is ready for the latest available init_time
            >>> is_ten_day_ready = model.is_ready(240)
            >>>
            >>> # Check if 10-day forecast is ready for a given init_time
            >>> is_ten_day_ready = model.is_ready(240, datetime(2025, 10, 1, 0))
            >>>
        """
        self._check_model_has_grid_access()

        if init_time == "latest":
            latest = self.get_latest_init_time(min_prediction_timedelta=0)
            return latest.prediction_timedelta >= forecasted_hours
        elif isinstance(init_time, str):
            init_time = datetime.fromisoformat(init_time)

        init_time = init_time.replace(tzinfo=UTC)  # type: ignore[call-arg]
        api_result = self._query_engine.get_available_forecasts(
            model=self._model,
            since=init_time,
            before=init_time,
            limit=1,
            offset=0,
        )
        forecasts = api_result.forecasts_per_model.get(self._model.value, [])
        if forecasts is None or len(forecasts) == 0:
            return False

        if forecasts[0].max_prediction_timedelta is None:
            return False

        return forecasts[0].max_prediction_timedelta >= forecasted_hours

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_available_forecasts(
        self,
        since: datetime | None = None,
        before: datetime | None = None,
        limit: int = 20,
    ) -> AvailableForecasts:
        """Retrieve available forecast initialization times for this model.

        This method queries the available forecast runs, returning their initialization
        times and maximum available lead times. Results are paginated for easy
        navigation through large datasets.

        Args:
            since: Only return forecasts initialized on or after this datetime
                (optional). Useful for finding forecasts within a specific time range.

            before: Only return forecasts initialized on or before before this datetime
                (optional). Useful for filtering historical forecasts.

            limit: Maximum number of results to return per page (default: 20).
                Controls the page size for pagination.

        Returns:
            AvailableForecasts containing the query results with convenient
            pagination support via the `.next()` method.

        Examples:
            >>> from datetime import datetime
            >>> from jua import JuaClient
            >>> from jua.weather import Models
            >>>
            >>> client = JuaClient()
            >>> model = client.weather.get_model(Models.EPT2)
            >>>
            >>> # Get recent forecasts from January 2025
            >>> result = model.get_available_forecasts(
            ...     since=datetime(2025, 1, 1),
            ...     limit=20
            ... )
            >>>
            >>> # Access forecasts - can iterate directly or use .forecasts property
            >>> for forecast_info in result:
            ...     print(f"Init time: {forecast_info.init_time}")
            ...     print(
            ...         f"Max prediction timedelta: "
            ...         f"{forecast_info.max_prediction_timedelta} hours"
            ...     )
            >>>
            >>> # Or access via property
            >>> print(f"Found {len(result.forecasts)} forecasts")
            >>>
            >>> # Paginate through results
            >>> if result.has_more:
            ...     next_page = result.next()
            ...     print(f"Next page has {len(next_page)} forecasts")
            >>>
            >>> # Or iterate through all pages
            >>> result = model.get_available_forecasts(
            >>>     since=datetime(2025, 1, 1), limit=50
            >>> )
            >>> all_forecasts = list(result.forecasts)
            >>> while result.has_more:
            ...     result = result.next()
            ...     all_forecasts.extend(result.forecasts)
        """
        self._check_model_has_grid_access()

        def fetch_page(offset: int) -> AvailableForecasts:
            """Internal helper to fetch a specific page of results."""
            api_result = self._query_engine.get_available_forecasts(
                model=self._model,
                since=since,
                before=before,
                limit=limit,
                offset=offset,
            )
            # Extract the forecasts for this specific model
            model_name = self._model.value
            forecasts = api_result.forecasts_per_model.get(model_name, [])

            # Ensure pagination info exists (should always be present from API)
            pagination = api_result.pagination or Pagination(limit=limit, offset=offset)

            return AvailableForecasts(
                forecasts=forecasts,
                pagination=pagination,
                fetch_next=fetch_page,
            )

        return fetch_page(offset=0)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_latest_init_time(
        self,
        min_prediction_timedelta: int = 0,
    ) -> LatestForecastInfo:
        """Get the latest available forecast initialization time for this model.

        This method retrieves information about the most recent forecast run, including
        its initialization time and maximum available forecast horizon.

        Args:
            min_prediction_timedelta: Minimum required forecast horizon in hours.

        Returns:
            LatestForecastInfo containing the init_time and prediction_timedelta in
            hours.

        Examples:
            >>> from jua import JuaClient
            >>> from jua.weather import Models
            >>>
            >>> client = JuaClient()
            >>> model = client.weather.get_model(Models.EPT2)
            >>>
            >>> # Get the latest forecast info
            >>> latest = model.get_latest_init_time()
            >>> print(f"Latest forecast initialized at: {latest.init_time}")
            >>> print(f"Max lead time: {latest.prediction_timedelta} hours")
            >>> print(f"Max lead time: {latest.prediction_timedelta / 24:.1f} days")
            >>>
            >>> # Get latest forecast with at least 48 hours of lead time
            >>> latest_2day = model.get_latest_init_time(min_prediction_timedelta=48)
            >>> print(f"Forecast horizon: {latest_2day.prediction_timedelta} hours")
        """
        api_result = self._query_engine.get_latest_init_time(
            model=self._model,
            min_prediction_timedelta=min_prediction_timedelta,
        )

        # Extract the info for this specific model
        model_name = self._model.value
        latest_info = api_result.forecasts_per_model.get(model_name)

        if latest_info is None:
            raise ValueError(
                f"No latest forecast information available for model {model_name}"
            )

        return latest_info

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_metadata(self) -> ModelMetadata:
        """Get metadata for this model including available variables and grid.

        This method retrieves comprehensive metadata about the model, including:
        - List of all available weather variables
        - Spatial grid resolution (number of latitude/longitude points)
        - Model identifier

        Returns:
            ModelMetadata containing variables list and grid information.

        Examples:
            >>> from jua import JuaClient
            >>> from jua.weather import Models, Variables
            >>>
            >>> client = JuaClient()
            >>> model = client.weather.get_model(Models.EPT2)
            >>>
            >>> # Get model metadata
            >>> metadata = model.get_metadata()
            >>> print(metadata)
            >>>
            >>> # List all available variables
            >>> print("Available variables:")
            >>> for var in metadata.variables:
            ...     print(f"  - {var.value.name}")
            >>>
            >>> # Check if a specific variable is available
            >>> if Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M in metadata.variables:
            ...     print("Temperature data is available!")
        """
        api_result = self._query_engine.get_meta(
            model=self._model,
        )

        # Extract the metadata for this specific model
        # The API returns a list, and we requested only one model
        if not api_result.models:
            raise ValueError(f"No metadata available for model {self._model.value}")

        # Get the first (and only) model metadata
        return api_result.models[0]

    @property
    def forecast(self) -> Forecast:
        """Access forecast data for this model.

        Returns:
            Forecast instance configured for this model.
        """
        warnings.warn(
            "Accessing .forecast is deprecated and will be removed in a future release."
            " Use model methods directly instead (e.g., model.get_forecasts()). "
            "Check the docs for more information and examples: https://docs.jua.ai",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._forecast

    @property
    def hindcast(self) -> Hindcast:
        """Access historical weather data for this model.

        Returns:
            Hindcast instance configured for this model.
        """
        warnings.warn(
            "Accessing .hindcast is deprecated and will be removed in a future release."
            " Use model methods directly instead (e.g., model.get_forecasts()). "
            "Check the docs for more information and examples: https://docs.jua.ai",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._hindcast

    def __repr__(self) -> str:
        """Get string representation of the model.

        Returns:
            A string representation suitable for debugging.
        """
        return f"<Model name='{self.name}'>"

    def __str__(self) -> str:
        """Get the model name as a string.

        Returns:
            The model name.
        """
        return self.name

    def _check_model_has_grid_access(self) -> None:
        meta = get_model_meta_info(self._model)
        if not meta.has_grid_access:
            raise ValueError(f"This method is not available for {self._model}.")

    def _check_model_has_stats(self) -> None:
        meta = get_model_meta_info(self._model)
        if not meta.has_statistics:
            raise ValueError(f"No statistics are available for {self._model}.")
