from datetime import datetime
from pathlib import Path

import xarray as xr
from pydantic import validate_call

from jua._utils.optional_progress_bar import OptionalProgressBar
from jua._utils.spinner import Spinner
from jua.logging import get_logger
from jua.settings.jua_settings import JuaSettings
from jua.weather._model_meta import get_model_meta_info
from jua.weather._xarray_patches import (
    TypedDataArray,
    TypedDataset,
    as_typed_dataarray,
    as_typed_dataset,
)
from jua.weather.conversions import bytes_to_gb
from jua.weather.models import Model
from jua.weather.variables import rename_variable

logger = get_logger(__name__)


def rename_variables(ds: xr.Dataset) -> xr.Dataset:
    output_variable_names = {k: rename_variable(k) for k in ds.variables}
    return ds.rename(output_variable_names)


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

    @property
    def zarr_version(self) -> int:
        return get_model_meta_info(self._model).forecast_zarr_version

    def _get_default_output_path(self) -> Path:
        return Path.home() / ".jua" / "datasets" / self._model.value

    def to_xarray(self) -> TypedDataset:
        return as_typed_dataset(self._raw_data)

    def __getitem__(self, key: any) -> TypedDataArray:
        return as_typed_dataarray(self._raw_data[str(key)])

    @validate_call(config={"arbitrary_types_allowed": True})
    def download(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        output_path: Path | None = None,
        show_progress: bool | None = None,
        overwrite: bool = False,
        always_download: bool = False,
    ) -> None:
        if output_path is None:
            output_path = self._get_default_output_path()

        output_name = self._dataset_name

        if start_date is not None and end_date is not None:
            output_name += f"-from-{start_date.isoformat()}-to-{end_date.isoformat()}"
        elif start_date is not None:
            output_name += f"-since-{start_date.isoformat()}"
        elif end_date is not None:
            output_name += f"-before-{end_date.isoformat()}"

        if output_path.suffix != ".zarr":
            output_path = output_path / f"{output_name}.zarr"

        if output_path.exists() and not overwrite:
            logger.warning(
                f"Dataset {self._dataset_name} already exists at {output_path}. "
                "Skipping download."
            )
            return

        if start_date is not None and end_date is not None:
            data_to_download = self._raw_data.sel(time=slice(start_date, end_date))
        elif start_date is not None:
            data_to_download = self._raw_data.sel(time=slice(start_date, None))
        elif end_date is not None:
            data_to_download = self._raw_data.sel(time=slice(None, end_date))
        else:
            data_to_download = self._raw_data

        download_size_gb = bytes_to_gb(data_to_download.nbytes)
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

        with Spinner("Preparing download. This might take a while..."):
            zarr_version = get_model_meta_info(self._model).forecast_zarr_version
            logger.info(f"Initializing dataset (zarr_format={zarr_version})...")
            delayed = data_to_download.to_zarr(
                output_path, mode="w", zarr_format=zarr_version, compute=False
            )

        with OptionalProgressBar(self._settings, show_progress):
            logger.info("Downloading dataset...")
            delayed.compute()
        logger.info(f"Dataset {self._dataset_name} downloaded to {output_path}.")
