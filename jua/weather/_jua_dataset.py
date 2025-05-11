from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar
from pydantic import validate_call

from jua.logging import get_logger
from jua.weather.conversions import bytes_to_gb
from jua.weather.models import Model

logger = get_logger(__name__)


class JuaDataset:
    _DOWLOAD_SIZE_WARNING_THRESHOLD_GB = 20

    def __init__(self, dataset_name: str, raw_data: xr.Dataset, model: Model):
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

    def to_xarray(self) -> xr.Dataset:
        return self._raw_data

    @validate_call
    def download(
        self,
        output_path: Path | None = None,
        show_progress: bool = True,
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

        if show_progress:
            logger.info("Initializing dataset...")
            delayed = self._raw_data.to_zarr(output_path, mode="w", compute=False)
            logger.info("Downloading dataset...")
            with ProgressBar():
                delayed.compute()
        else:
            self._raw_data.to_zarr(output_path, mode="w", compute=True)
        logger.info(f"Dataset {self._dataset_name} downloaded to {output_path}.")
