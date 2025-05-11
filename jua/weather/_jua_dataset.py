from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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
from jua.weather.variables import Variables, rename_variable

logger = get_logger(__name__)


def rename_variables(ds: xr.Dataset) -> xr.Dataset:
    output_variable_names = {k: rename_variable(k) for k in ds.variables}
    return ds.rename(output_variable_names)


def _potential_slice_to_str(maybe_slice: slice | Any) -> str:
    if isinstance(maybe_slice, slice):
        return f"{maybe_slice.start}-{maybe_slice.stop}"
    else:
        return str(maybe_slice)


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

    @validate_call(config={"arbitrary_types_allowed": True})
    def download(
        self,
        variables: list[Variables] | None = None,
        time: datetime | slice | None = None,
        prediction_timedelta: np.timedelta64 | int | slice | None = None,
        latitude: float | slice | None = None,
        longitude: float | slice | None = None,
        output_path: Path | None = None,
        show_progress: bool | None = None,
        overwrite: bool = False,
        always_download: bool = False,
    ) -> None:
        if output_path is None:
            output_path = self._get_default_output_path()

        output_name = self._dataset_name
        if time is not None:
            output_name += f"-time={_potential_slice_to_str(time)}"
        if prediction_timedelta is not None:
            output_name += (
                f"-prediction_timedelta={_potential_slice_to_str(prediction_timedelta)}"
            )
        if latitude is not None:
            output_name += f"-latitude={_potential_slice_to_str(latitude)}"
        if longitude is not None:
            output_name += f"-longitude={_potential_slice_to_str(longitude)}"

        if output_path.suffix != ".zarr":
            output_path = output_path / f"{output_name}.zarr"

        if output_path.exists() and not overwrite:
            logger.warning(
                f"Dataset {self._dataset_name} already exists at {output_path}. "
                "Skipping download."
            )
            return

        if variables is None:
            data_to_download = self._raw_data
        else:
            data_to_download = self._raw_data[[str(v) for v in variables]]

        data_to_download = data_to_download.sel(
            time=time,
            latitude=latitude,
            longitude=longitude,
            prediction_timedelta=prediction_timedelta,
        )

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

        with OptionalProgressBar(self._settings, show_progress):
            logger.info("Initializing dataset...")
            delayed = data_to_download.to_zarr(output_path, mode="w", compute=False)
            logger.info("Downloading dataset...")
            delayed.compute()
        logger.info(f"Dataset {self._dataset_name} downloaded to {output_path}.")
