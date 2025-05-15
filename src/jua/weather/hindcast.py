from dataclasses import dataclass
from datetime import datetime

import xarray as xr
from pydantic import validate_call

from jua._utils.dataset import open_dataset
from jua.client import JuaClient
from jua.errors.model_errors import ModelHasNoHindcastData
from jua.logging import get_logger
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather._api import WeatherAPI
from jua.weather._jua_dataset import JuaDataset
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
    of historical model runs.

    Not all models have hindcast data available. Use the is_file_access_available()
    method to check if a model supports hindcasts.
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
        """Check for hindcast availability and raise error if unavailable."""
        if not self.is_file_access_available():
            raise ModelHasNoHindcastData(self._model_name)

    @property
    def metadata(self) -> HindcastMetadata:
        """Get metadata about the available hindcast data for this model.

        Returns:
            HindcastMetadata with date ranges and available regions.

        Raises:
            ModelHasNoHindcastData: If the model doesn't support hindcasts.
        """
        self._raise_if_no_file_access()
        return self._MODEL_METADATA[self._model]

    def is_file_access_available(self) -> bool:
        """Check if hindcast data is available for this model.

        Returns:
            True if hindcast data is available, False otherwise.
        """
        return self._model in self._HINDCAST_ADAPTERS

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_hindcast(
        self,
        print_progress: bool | None = None,
        variables: list[Variables] | list[str] | None = None,
        time: datetime | list[datetime] | slice | None = None,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | LatLon | None = None,
        method: str | None = None,
    ) -> JuaDataset:
        """Retrieve the complete hindcast dataset for this model.

        This method loads the full hindcast dataset for the model, which may
        take some time to load.
        Note: The data is not being downloaded.

        Args:
            print_progress: Whether to display a progress bar during data loading.
                If None, uses the client's default setting.

        Returns:
            JuaDataset containing the hindcast data.

        Raises:
            ModelHasNoHindcastData: If the model doesn't support hindcasts.
        """
        self._raise_if_no_file_access()

        return self._HINDCAST_ADAPTERS[self._model](
            print_progress=print_progress,
            variables=variables,
            time=time,
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

        Args:
            url: URL or list of URLs to the dataset files.
            print_progress: Whether to display a progress bar.

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
        """Load EPT2 hindcast dataset."""
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
        """Load EPT1.5 Early hindcast dataset."""
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
        """Load EPT1.5 hindcast dataset (multiple regions)."""
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
        """Load AIFS025 hindcast dataset."""
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
