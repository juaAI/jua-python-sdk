from dataclasses import dataclass
from datetime import datetime

from pydantic import validate_call

from jua._utils.dataset import DatasetConfig, open_dataset
from jua.client import JuaClient
from jua.errors.model_errors import ModelHasNoHindcastData
from jua.logging import get_logger
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather import JuaDataset
from jua.weather._api import WeatherAPI
from jua.weather.models import Models
from jua.weather.variables import Variables

logger = get_logger(__name__)


@dataclass
class Region:
    """Geographic region with associated coverage information.

    Attributes:
        region: Name of the geographic region (e.g., "Europe", "Global").
        coverage: String description of the region's coordinate boundaries.
    """

    region: str
    coverage: str


@dataclass
class HindcastMetadata:
    """Metadata describing the available hindcast data for a model.

    Attributes:
        start_date: Beginning date of available hindcast data.
        end_date: End date of available hindcast data.
        available_regions: List of geographic regions covered by the hindcast.
    """

    start_date: datetime
    end_date: datetime

    available_regions: list[Region]


class Hindcast:
    """Access to historical weather data (hindcasts) for a specific model.

    This class provides methods to retrieve hindcast data from Jua's archive
    of historical model runs. Hindcasts are past forecasts that can be used
    for model evaluation, training machine learning models, or analyzing
    past weather events.

    Not all models have hindcast data available. Use the is_file_access_available()
    method to check if a model supports hindcasts.

    Examples:
        >>> from jua import JuaClient
        >>> from jua.weather import Models
        >>>
        >>> client = JuaClient()
        >>> model = client.weather.get_model(Models.EPT2)
        >>>
        >>> # Check if hindcast data is available
        >>> if model.hindcast.is_file_access_available():
        >>>
        >>>     # Get hindcast data for specific time period and region
        >>>     data = model.hindcast.get_hindcast(
        >>>         init_time=slice("2023-01-01", "2023-01-31"),
        >>>         latitude=slice(60, 40),  # North to South
        >>>         longitude=slice(-10, 30)  # West to East
        >>>     )
    """

    def __init__(self, client: JuaClient, model: Models):
        """Initialize hindcast access for a specific model.

        Args:
            client: JuaClient instance for authentication and settings.
            model: Weather model to access hindcast data for.
        """
        self._client = client
        self._model = model
        self._model_name = model.value
        self._api = WeatherAPI(client)

    def _raise_if_no_file_access(self):
        """Check for hindcast availability and raise error if unavailable.

        This internal method provides a consistent way to validate hindcast
        availability before performing operations that require it.

        Raises:
            ModelHasNoHindcastData: If the model doesn't support hindcasts.
        """
        if not self.is_file_access_available():
            raise ModelHasNoHindcastData(self._model_name)

    def is_file_access_available(self) -> bool:
        """Check if hindcast data is available for this model.

        Not all models have historical data available. This method allows you
        to check if the current model supports hindcasts before attempting
        to retrieve hindcast data.

        Returns:
            True if hindcast data is available, False otherwise.

        Examples:
            >>> if model.hindcast.is_file_access_available():
            >>>     hindcast_data = model.hindcast.get_hindcast()
            >>> else:
            >>>     print(f"No hindcast data available for {model.name}")
        """
        files = self._get_hindcast_files()
        return len(files) > 0

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_hindcast(
        self,
        init_time: datetime | list[datetime] | slice | None = None,
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
        """Retrieve historical weather data (hindcast) for this model.

        This method loads weather data from past model runs, allowing analysis
        of historical forecasts and verification against actual observations.
        The data is loaded from Jua's archive and not downloaded to your machine.

        You can filter the hindcast data by:
        - Time period (init_time)
        - Geographic area (latitude/longitude or points)
        - Lead time (prediction_timedelta or min/max_lead_time)
        - Weather variables (variables)

        Args:
            init_time: Filter by forecast initialization time. Can be:
                - None: All available initialization times (default)
                - A single datetime: Specific initialization time
                - A list of datetimes: Multiple specific times
                - A slice(start, end): Range of initialization times

            variables: List of weather variables to include. If None, includes all
                available variables. This increases data loading time & memory usage.

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
                LatLon object or a list of LatLon objects.

            min_lead_time: Minimum lead time in hours
                (alternative to prediction_timedelta).

            max_lead_time: Maximum lead time in hours
                (alternative to prediction_timedelta).

            method: Interpolation method for selecting points:
                - "nearest" (default): Use nearest grid point
                - All other methods supported by xarray such as "linear", "cubic"

            print_progress: Whether to display a progress bar during data loading.
                If None, uses the client's default setting.

        Returns:
            JuaDataset containing the hindcast data matching your selection criteria.

        Raises:
            ModelHasNoHindcastData: If the model doesn't support hindcasts.
            ValueError: If incompatible parameter combinations are provided.

        Examples:
            >>> # Get hindcast for Europe in January 2023
            >>> europe_jan_2023 = model.hindcast.get_hindcast(
            ...     init_time=slice("2023-01-01", "2023-01-31"),
            ...     latitude=slice(72, 36),  # North to South
            ...     longitude=slice(-15, 35),  # West to East
            ...     variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M]
            ... )
            >>>
            >>> # Get hindcast for specific cities with 24-hour lead time
            >>> cities_data = model.hindcast.get_hindcast(
            ...     points=[
            ...         LatLon(lat=40.7128, lon=-74.0060),  # New York
            ...         LatLon(lat=51.5074, lon=-0.1278),   # London
            ...     ],
            ...     prediction_timedelta=24,  # 24-hour forecasts
            ... )
        """
        self._raise_if_no_file_access()
        if prediction_timedelta is not None and (
            min_lead_time is not None or max_lead_time is not None
        ):
            raise ValueError(
                "Cannot provide both prediction_timedelta and "
                "min_lead_time/max_lead_time.\nPlease provide "
                "either prediction_timedelta or min_lead_time/max_lead_time."
            )
        if min_lead_time is not None or max_lead_time is not None:
            prediction_timedelta = slice(min_lead_time, max_lead_time)

        return self._open_dataset(
            print_progress=print_progress,
            variables=variables,
            time=init_time,
            prediction_timedelta=prediction_timedelta,
            latitude=latitude,
            longitude=longitude,
            points=points,
            method=method,
        )

    def _get_hindcast_files(self) -> list[DatasetConfig]:
        """Get the list of hindcast files for the current model.

        Returns:
            List of hindcast file URLs.
        """
        files = self._api.get_hindcast_files(self._model_name)
        return [
            DatasetConfig(
                path=f"{self._client.settings.data_base_url}/{file.path}",
                recommended_chunks=file.recommended_chunks,
            )
            for file in files
        ]

    def _open_dataset(
        self,
        print_progress: bool | None = None,
        **kwargs,
    ) -> JuaDataset:
        """Open a dataset from the given URL with appropriate chunking.

        This internal method handles opening datasets with model-specific chunk sizes
        and optional progress display.

        Args:
            url: URL or list of URLs to the dataset files.
            print_progress: Whether to display a progress bar.
            **kwargs: Additional arguments passed to the dataset opening function.

        Returns:
            Opened xarray Dataset.
        """
        dataset_configs = self._get_hindcast_files()
        raw_data = open_dataset(
            self._client,
            dataset_configs,
            should_print_progress=print_progress,
            **kwargs,
        )

        return JuaDataset(
            settings=self._client.settings,
            dataset_name=f"hindcast-{self._model_name}",
            raw_data=raw_data,
            model=self._model,
        )
