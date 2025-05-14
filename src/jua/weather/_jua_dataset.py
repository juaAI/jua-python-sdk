from pathlib import Path
from typing import Any

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
from jua.weather.models import Models
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
        model: Models,
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
    def zarr_version(self) -> int | None:
        return get_model_meta_info(self._model).forecast_zarr_version

    def _get_default_output_path(self) -> Path:
        return Path.home() / ".jua" / "datasets" / self._model.value

    def to_xarray(self) -> TypedDataset:
        return as_typed_dataset(self._raw_data)

    def __getitem__(self, key: Any) -> TypedDataArray:
        return as_typed_dataarray(self._raw_data[str(key)])

    @validate_call(config={"arbitrary_types_allowed": True})
    def save(
        self,
        output_path: Path | None = None,
        show_progress: bool | None = None,
        overwrite: bool = False,
        ignore_size_warning: bool = False,
    ) -> None:
        if output_path is None:
            output_path = self._get_default_output_path()

        output_name = self._dataset_name
        if output_path.suffix != ".zarr":
            output_path = output_path / f"{output_name}.zarr"

        if output_path.exists() and not overwrite:
            logger.warning(
                f"Dataset {self._dataset_name} already exists at {output_path}. "
                "Skipping download."
            )
            return

        data_to_save = self._raw_data
        data_size = bytes_to_gb(data_to_save.nbytes)
        if (
            not ignore_size_warning
            and data_size > self._DOWLOAD_SIZE_WARNING_THRESHOLD_GB
        ):
            logger.warning(
                f"Dataset {self._dataset_name} is large ({data_size:.2f}GB). "
                "This may take a while to save."
            )
            yn = input("Do you want to continue? (y/N) ")
            if yn.lower() != "y":
                logger.info("Skipping save.")
                return

        logger.info(
            f"Saving a {data_size:.2f}GB dataset "
            f"{self._dataset_name} to {output_path}..."
        )

        with Spinner(
            "Preparing save. This might take a while...",
            disable=not show_progress,
        ):
            zarr_version = get_model_meta_info(self._model).forecast_zarr_version
            logger.info(f"Initializing dataset (zarr_format={zarr_version})...")
            delayed = data_to_save.to_zarr(
                output_path, mode="w", zarr_format=zarr_version, compute=False
            )

        with OptionalProgressBar(self._settings, show_progress):
            logger.info("Saving dataset...")
            delayed.compute()
        logger.info(f"Dataset {self._dataset_name} saved to {output_path}.")
