from dataclasses import dataclass
from datetime import datetime

import xarray as xr
from pydantic import validate_call

from jua._utils.dataset import open_dataset
from jua.client import JuaClient
from jua.errors.model_errors import ModelHasNoHindcastData
from jua.logging import get_logger
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather import JuaDataset
from jua.weather._api import WeatherAPI
from jua.weather._model_meta import get_model_meta_info
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
        >>>     # Get metadata about available hindcasts
        >>>     meta = model.hindcast.metadata
        >>>     print(f"Date range: {meta.start_date} to {meta.end_date}")
        >>>
        >>>     # Get hindcast data for specific time period and region
        >>>     data = model.hindcast.get_hindcast(
        >>>         init_time=slice("2023-01-01", "2023-01-31"),
        >>>         latitude=slice(60, 40),  # North to South
        >>>         longitude=slice(-10, 30)  # West to East
        >>>     )
    """

    _MODEL_METADATA = {
        Models.EPT2: HindcastMetadata(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 12, 28),
            available_regions=[Region(region="Global", coverage="")],
        ),
        Models.EPT1_5: HindcastMetadata(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2024, 7, 31),
            available_regions=[
                Region(region="Europe", coverage="36°-72°N, -15°-35°E"),
                Region(region="North America", coverage="Various"),
            ],
        ),
        Models.EPT1_5_EARLY: HindcastMetadata(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2024, 7, 31),
            available_regions=[
                Region(region="Europe", coverage=""),
            ],
        ),
        Models.ECMWF_AIFS025_SINGLE: HindcastMetadata(
            start_date=datetime(2023, 1, 2),
            end_date=datetime(2024, 12, 27),
            available_regions=[
                Region(region="Global", coverage=""),
            ],
        ),
    }

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

        self._HINDCAST_ADAPTERS = {
            Models.EPT2: self._ept2_adapter,
            Models.EPT1_5: self._ept15_adapter,
            Models.EPT1_5_EARLY: self._ept_15_early_adapter,
            Models.ECMWF_AIFS025_SINGLE: self._aifs025_adapter,
        }

    def _raise_if_no_file_access(self):
        """Check for hindcast availability and raise error if unavailable.

        This internal method provides a consistent way to validate hindcast
        availability before performing operations that require it.

        Raises:
            ModelHasNoHindcastData: If the model doesn't support hindcasts.
        """
        if not self.is_file_access_available():
            raise ModelHasNoHindcastData(self._model_name)

    @property
    def metadata(self) -> HindcastMetadata:
        """Get metadata about the available hindcast data for this model.

        Retrieves information about the time range and geographic coverage
        of hindcast data available for the current model.

        Returns:
            HindcastMetadata with date ranges and available regions.

        Raises:
            ModelHasNoHindcastData: If the model doesn't support hindcasts.

        Examples:
            >>> metadata = model.hindcast.metadata
            >>> print(
            ...     "Hindcast available from "
            ...     f"{metadata.start_date} to {metadata.end_date}"
            ... )
            >>> for region in metadata.available_regions:
            >>>     print(f"Region: {region.region}, Coverage: {region.coverage}")
        """
        self._raise_if_no_file_access()
        return self._MODEL_METADATA[self._model]

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
        return self._model in self._HINDCAST_ADAPTERS

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

        return self._HINDCAST_ADAPTERS[self._model](
            print_progress=print_progress,
            variables=variables,
            time=init_time,
            prediction_timedelta=prediction_timedelta,
            latitude=latitude,
            longitude=longitude,
            points=points,
            method=method,
        )

    def _open_dataset(
        self,
        url: str | list[str],
        print_progress: bool | None = None,
        **kwargs,
    ) -> xr.Dataset:
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
        chunks = get_model_meta_info(self._model).hindcast_chunks
        return open_dataset(
            self._client,
            url,
            should_print_progress=print_progress,
            chunks=chunks,
            **kwargs,
        )

    def _ept2_adapter(self, print_progress: bool | None = None, **kwargs) -> JuaDataset:
        """Load EPT2 hindcast dataset.

        This adapter handles loading hindcast data for the EPT2 model, which is stored
        in a single consolidated Zarr store covering the global domain.

        Args:
            print_progress: Whether to display a progress bar.
            **kwargs: Selection criteria passed from get_hindcast().

        Returns:
            JuaDataset containing the EPT2 hindcast data.
        """
        data_base_url = self._client.settings.data_base_url
        data_url = (
            f"{data_base_url}/hindcasts/ept-2/v2/global/2023-01-01-to-2024-12-28.zarr"
        )

        raw_data = self._open_dataset(data_url, print_progress=print_progress, **kwargs)
        return JuaDataset(
            settings=self._client.settings,
            dataset_name="hindcast-2023-01-01-to-2024-12-28",
            raw_data=raw_data,
            model=self._model,
        )

    def _ept_15_early_adapter(
        self, print_progress: bool | None = None, **kwargs
    ) -> JuaDataset:
        """Load EPT1.5 Early hindcast dataset.

        This adapter handles loading hindcast data for the EPT1.5 Early model,
        which provides data specifically for the Europe region.

        Args:
            print_progress: Whether to display a progress bar.
            **kwargs: Selection criteria passed from get_hindcast().

        Returns:
            JuaDataset containing the EPT1.5 Early hindcast data.
        """
        data_base_url = self._client.settings.data_base_url
        data_url = f"{data_base_url}/hindcasts/ept-1.5-early/europe/2024.zarr/"

        raw_data = self._open_dataset(data_url, print_progress=print_progress, **kwargs)
        return JuaDataset(
            settings=self._client.settings,
            dataset_name="hindcast-2024-europe",
            raw_data=raw_data,
            model=self._model,
        )

    def _ept15_adapter(
        self, print_progress: bool | None = None, **kwargs
    ) -> JuaDataset:
        """Load EPT1.5 hindcast dataset (multiple regions).

        This adapter handles loading hindcast data for the EPT1.5 model, which
        is stored across multiple Zarr stores covering different regions (Europe
        and North America) and time periods.

        Args:
            print_progress: Whether to display a progress bar.
            **kwargs: Selection criteria passed from get_hindcast().

        Returns:
            JuaDataset containing the combined EPT1.5 hindcast data.
        """
        data_base_url = self._client.settings.data_base_url

        zarr_urls = [
            f"{data_base_url}/hindcasts/ept-1.5/europe/2023.zarr/",
            f"{data_base_url}/hindcasts/ept-1.5/europe/2024.zarr/",
            f"{data_base_url}/hindcasts/ept-1.5/north-america/2023-00H.zarr/",
            f"{data_base_url}/hindcasts/ept-1.5/north-america/2024-00H.zarr/",
            f"{data_base_url}/hindcasts/ept-1.5/north-america/2024.zarr/",
        ]

        raw_data = self._open_dataset(
            zarr_urls, print_progress=print_progress, **kwargs
        )
        return JuaDataset(
            settings=self._client.settings,
            dataset_name="hindcast-ept-1.5-europe-north-america",
            raw_data=raw_data,
            model=self._model,
        )

    def _aifs025_adapter(
        self, print_progress: bool | None = None, **kwargs
    ) -> JuaDataset:
        """Load AIFS025 hindcast dataset.

        This adapter handles loading hindcast data for the ECMWF AIFS025 model,
        which provides global coverage.

        Args:
            print_progress: Whether to display a progress bar.
            **kwargs: Selection criteria passed from get_hindcast().

        Returns:
            JuaDataset containing the AIFS025 hindcast data.
        """
        data_base_url = self._client.settings.data_base_url
        zarr_url = (
            f"{data_base_url}/hindcasts/aifs/v1/global/2023-01-02-to-2024-12-27.zarr/"
        )

        raw_data = self._open_dataset(zarr_url, print_progress=print_progress, **kwargs)

        return JuaDataset(
            settings=self._client.settings,
            dataset_name="hindcast-aifs025-global",
            raw_data=raw_data,
            model=self._model,
        )
