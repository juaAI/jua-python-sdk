from pathlib import Path

import xarray as xr
from pydantic import validate_call

from jua._utils.optional_progress_bar import OptionalProgressBar
from jua.logging import get_logger
from jua.settings.jua_settings import JuaSettings
from jua.weather._xarray_patches import (
    TypedDataArray,
    TypedDataset,
    as_typed_dataarray,
    as_typed_dataset,
)
from jua.weather.conversions import bytes_to_gb
from jua.weather.models import Model
from jua.weather.variables import Variables

logger = get_logger(__name__)

_MAP_EPT2_TO_NEW_VARIABLES = {
    v.value.name_ept2: v.value.name for v in Variables if v.value.name_ept2
}
_MAP_EPT1_5_TO_NEW_VARIABLES = {
    v.value.emcwf_code: str(v.value) for v in Variables if v.value.emcwf_code
}


def rename_variables_ept2(ds: xr.Dataset) -> xr.Dataset:
    return ds.rename(_MAP_EPT2_TO_NEW_VARIABLES)


def rename_variables_ept1_5(ds: xr.Dataset) -> xr.Dataset:
    return ds.rename(_MAP_EPT1_5_TO_NEW_VARIABLES)


class JuaDataset:
    _DOWLOAD_SIZE_WARNING_THRESHOLD_GB = 20

    def __init__(
        self,
        settings: JuaSettings,
        dataset_name: str,
        raw_data: xr.Dataset,
        model: Model,
    ):
        self._settings = settings
        self._dataset_name = dataset_name
        self._raw_data = raw_data
        self._model = model

    @property
    def nbytes(self) -> int:
        return self._raw_data.nbytes

    @property
    def nbytes_gb(self) -> float:
        return bytes_to_gb(self.nbytes)

    def _get_default_output_path(self) -> Path:
        return (
            Path.home() / ".jua" / "datasets" / self._model.value / self._dataset_name
        )

    def to_xarray(self) -> TypedDataset:
        return as_typed_dataset(self._raw_data)

    def __getitem__(self, key: any) -> TypedDataArray:
        return as_typed_dataarray(self._raw_data[str(key)])

    @validate_call
    def download(
        self,
        output_path: Path | None = None,
        show_progress: bool | None = None,
        overwrite: bool = False,
        always_download: bool = False,
    ) -> None:
        if output_path is None:
            output_path = self._get_default_output_path()

        if output_path.exists() and not overwrite:
            logger.warning(
                f"Dataset {self._dataset_name} already exists at {output_path}. "
                "Skipping download."
            )
            return

        download_size_gb = bytes_to_gb(self._raw_data.nbytes)
        if (
            not always_download
            and download_size_gb > self._DOWLOAD_SIZE_WARNING_THRESHOLD_GB
        ):
            logger.warning(
                f"Dataset {self._dataset_name} is large ({download_size_gb:.2f}GB). "
                "This may take a while to download."
            )
            yn = input("Do you want to continue? (y/N) ")
            if yn.lower() != "y":
                logger.info("Skipping download.")
                return
        logger.info(
            f"Downloading {download_size_gb:.2f}GB dataset "
            f"{self._dataset_name} to {output_path}..."
        )

        with OptionalProgressBar(self._settings, show_progress):
            logger.info("Initializing dataset...")
            delayed = self._raw_data.to_zarr(output_path, mode="w", compute=False)
            logger.info("Downloading dataset...")
            delayed.compute()
        logger.info(f"Dataset {self._dataset_name} downloaded to {output_path}.")
