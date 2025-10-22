from datetime import datetime
from typing import Literal

from pydantic import validate_call

from jua.client import JuaClient
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather import JuaDataset
from jua.weather._query_engine import QueryEngine
from jua.weather.forecast import Forecast
from jua.weather.hindcast import Hindcast
from jua.weather.models import Models as ModelEnum
from jua.weather.variables import Variables


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
        method: Literal["nearest", "bilinear"] = "nearest",
        stream: bool | None = None,
        print_progress: bool | None = None,
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

            method: Interpolation method for selecting points:
                - "nearest": Use nearest grid point (default).
                - "bilinear": Bilinear interpolation to the selected point.

            stream: Whether to stream the response content. Recommended when querying
                slices or large amounts of data. Default is set to False for points,
                and True for grid slices. Streaming does not support method="bilinear"
                when requesting points.

            print_progress: Whether to display a progress bar during data loading.
                If None, uses the client's default setting.

        Returns:
            JuaDataset containing the hindcast data matching your selection criteria.

        Raises:
            ModelHasNoHindcastData: If the model doesn't support hindcasts.
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

        raw_data = self._query_engine.get_forecast(
            model=self._model,
            init_time=init_time,
            variables=variables,
            prediction_timedelta=prediction_timedelta,
            latitude=latitude,
            longitude=longitude,
            points=points,
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

    @property
    def forecast(self) -> Forecast:
        """Access forecast data for this model.

        Returns:
            Forecast instance configured for this model.
        """
        return self._forecast

    @property
    def hindcast(self) -> Hindcast:
        """Access historical weather data for this model.

        Returns:
            Hindcast instance configured for this model.
        """
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
