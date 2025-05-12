from dataclasses import dataclass
from datetime import datetime

import xarray as xr
from pydantic import validate_call

from jua._utils.dataset import open_dataset
from jua.client import JuaClient
from jua.errors.model_errors import ModelHasNoHindcastData
from jua.logging import get_logger
from jua.weather._api import WeatherAPI
from jua.weather._jua_dataset import JuaDataset, rename_variables
from jua.weather._model_meta import get_model_meta_info
from jua.weather.models import Models

logger = get_logger(__name__)


@dataclass
class Region:
    region: str
    coverage: str


@dataclass
class HindcastMetadata:
    start_date: datetime
    end_date: datetime

    available_regions: list[Region]


class Hindcast:
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
        if not self.is_file_access_available():
            raise ModelHasNoHindcastData(self._model_name)

    @property
    def metadata(self) -> HindcastMetadata:
        self._raise_if_no_file_access()
        return self._MODEL_METADATA[self._model]

    def is_file_access_available(self) -> bool:
        return self._model in self._HINDCAST_ADAPTERS

    @validate_call
    def get_hindcast_as_dataset(
        self,
        print_progress: bool | None = None,
    ) -> JuaDataset:
        self._raise_if_no_file_access()

        return self._HINDCAST_ADAPTERS[self._model](print_progress=print_progress)

    def _open_dataset(
        self, url: str | list[str], print_progress: bool | None = None
    ) -> xr.Dataset:
        chunks = get_model_meta_info(self._model).hindcast_chunks
        return open_dataset(
            self._client,
            url,
            should_print_progress=print_progress,
            chunks=chunks,
        )

    def _ept2_adapter(self, print_progress: bool | None = None) -> JuaDataset:
        data_base_url = self._client.settings.data_base_url
        data_url = (
            f"{data_base_url}/hindcasts/ept-2/v2/global/2023-01-01-to-2024-12-28.zarr"
        )

        raw_data = self._open_dataset(data_url, print_progress=print_progress)
        # Rename coordinate prediction_timedelta to leadtime
        raw_data = rename_variables(raw_data)
        return JuaDataset(
            settings=self._client.settings,
            dataset_name="hindcast-2023-01-01-to-2024-12-28",
            raw_data=raw_data,
            model=self._model,
        )

    def _ept_15_early_adapter(self, print_progress: bool | None = None) -> JuaDataset:
        data_base_url = self._client.settings.data_base_url
        data_url = f"{data_base_url}/hindcasts/ept-1.5-early/europe/2024.zarr/"

        raw_data = self._open_dataset(data_url, print_progress=print_progress)
        # Rename coordinate prediction_timedelta to leadtime
        raw_data = rename_variables(raw_data)
        return JuaDataset(
            settings=self._client.settings,
            dataset_name="hindcast-2024-europe",
            raw_data=raw_data,
            model=self._model,
        )

    def _ept15_adapter(self, print_progress: bool | None = None) -> JuaDataset:
        data_base_url = self._client.settings.data_base_url

        zarr_urls = [
            f"{data_base_url}/hindcasts/ept-1.5/europe/2023.zarr/",
            f"{data_base_url}/hindcasts/ept-1.5/europe/2024.zarr/",
            f"{data_base_url}/hindcasts/ept-1.5/north-america/2023-00H.zarr/",
            f"{data_base_url}/hindcasts/ept-1.5/north-america/2024-00H.zarr/",
            f"{data_base_url}/hindcasts/ept-1.5/north-america/2024.zarr/",
        ]

        raw_data = self._open_dataset(zarr_urls, print_progress=print_progress)
        raw_data = rename_variables(raw_data)
        return JuaDataset(
            settings=self._client.settings,
            dataset_name="hindcast-ept-1.5-europe-north-america",
            raw_data=raw_data,
            model=self._model,
        )

    def _aifs025_adapter(self, print_progress: bool | None = None) -> JuaDataset:
        data_base_url = self._client.settings.data_base_url
        zarr_url = (
            f"{data_base_url}/hindcasts/aifs/v1/global/2023-01-02-to-2024-12-27.zarr/"
        )

        raw_data = self._open_dataset(zarr_url, print_progress=print_progress)
        # Should already have the correct variable names
        return JuaDataset(
            settings=self._client.settings,
            dataset_name="hindcast-aifs025-global",
            raw_data=raw_data,
            model=self._model,
        )
