from pathlib import Path

import numpy as np
import xarray as xr
from pydantic import validate_call

from jua._utils.optional_progress_bar import OptionalProgressBar
from jua.logging import get_logger
from jua.settings.jua_settings import JuaSettings
from jua.weather.conversions import bytes_to_gb, to_timedelta
from jua.weather.models import Model

logger = get_logger(__name__)


@xr.register_dataarray_accessor("jua")
@xr.register_dataset_accessor("jua")
class _LeadTimeSelector:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._xarray_obj = xarray_obj

    # Lead time can be int, timedelta or slice
    def sel_lead_time(
        self, lead_time: int | np.timedelta64 | slice
    ) -> xr.DataArray | xr.Dataset:
        if not isinstance(lead_time, slice):
            lead_time = to_timedelta(lead_time)

        if isinstance(lead_time, slice):
            # Make sure the slice values are timedeltas
            start = lead_time.start
            stop = lead_time.stop
            if start is not None:
                start = to_timedelta(start)
            if stop is not None:
                stop = to_timedelta(stop)
            lead_time = slice(start, stop)

        return self._xarray_obj.sel(prediction_timedelta=lead_time)


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

    def to_xarray(self) -> xr.Dataset:
        return self._raw_data

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
