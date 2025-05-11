from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
import xarray as xr
from pydantic import validate_call

from jua._utils.optional_progress_bar import OptionalProgressBar
from jua.logging import get_logger
from jua.settings.jua_settings import JuaSettings
from jua.weather.conversions import bytes_to_gb, to_timedelta
from jua.weather.models import Model

logger = get_logger(__name__)

# Store original sel methods
_original_dataset_sel = xr.Dataset.sel
_original_dataarray_sel = xr.DataArray.sel


# Override Dataset.sel method
def _patched_dataset_sel(self, *args, **kwargs):
    """
    This is a patch to the xarray.Dataset.sel method to convert the prediction_timedelta
    argument to a timedelta.
    """
    # Check if prediction_timedelta is in kwargs
    if "prediction_timedelta" in kwargs and not isinstance(
        kwargs["prediction_timedelta"], slice
    ):
        # Convert to timedelta
        kwargs["prediction_timedelta"] = to_timedelta(kwargs["prediction_timedelta"])
    elif "prediction_timedelta" in kwargs and isinstance(
        kwargs["prediction_timedelta"], slice
    ):
        # Handle slice case
        start = kwargs["prediction_timedelta"].start
        stop = kwargs["prediction_timedelta"].stop
        step = kwargs["prediction_timedelta"].step

        if start is not None:
            start = to_timedelta(start)
        if stop is not None:
            stop = to_timedelta(stop)
        if step is not None:
            step = to_timedelta(step)

        kwargs["prediction_timedelta"] = slice(start, stop, step)

    # Call the original method
    return _original_dataset_sel(self, *args, **kwargs)


# Override DataArray.sel method
def _patched_dataarray_sel(self, *args, **kwargs):
    # Check if prediction_timedelta is in kwargs
    if "prediction_timedelta" in kwargs and not isinstance(
        kwargs["prediction_timedelta"], slice
    ):
        # Convert to timedelta
        kwargs["prediction_timedelta"] = to_timedelta(kwargs["prediction_timedelta"])
    elif "prediction_timedelta" in kwargs and isinstance(
        kwargs["prediction_timedelta"], slice
    ):
        # Handle slice case
        start = kwargs["prediction_timedelta"].start
        stop = kwargs["prediction_timedelta"].stop
        step = kwargs["prediction_timedelta"].step

        if start is not None:
            start = to_timedelta(start)
        if stop is not None:
            stop = to_timedelta(stop)
        if step is not None:
            step = to_timedelta(step)

        kwargs["prediction_timedelta"] = slice(start, stop, step)

    # Call the original method
    return _original_dataarray_sel(self, *args, **kwargs)


# Apply the patches
xr.Dataset.sel = _patched_dataset_sel
xr.DataArray.sel = _patched_dataarray_sel
# TODO: Keeping the code below for reference, might want to remove


# Define the actual implementation
@xr.register_dataarray_accessor("jua")
@xr.register_dataset_accessor("jua")
class _LeadTimeSelector:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._xarray_obj = xarray_obj

    # Lead time can be int, timedelta or slice
    def sel(
        self,
        prediction_timedelta: int | np.timedelta64 | slice | None = None,
        time: np.datetime64 | slice | None = None,
        latitude: float | slice | None = None,
        longitude: float | slice | None = None,
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        jua_args = {}
        if prediction_timedelta is not None:
            jua_args["prediction_timedelta"] = prediction_timedelta
        if time is not None:
            jua_args["time"] = time
        if latitude is not None:
            jua_args["latitude"] = latitude
        if longitude is not None:
            jua_args["longitude"] = longitude
        return self._xarray_obj.sel(**jua_args, **kwargs)


# For type checking only
if TYPE_CHECKING:
    T = TypeVar("T", bound=xr.DataArray | xr.Dataset)

    class JuaAccessorProtocol(Protocol[T]):
        def __init__(self, xarray_obj: T) -> None: ...

        def sel(
            self,
            prediction_timedelta: int | np.timedelta64 | slice | None = None,
            time: np.datetime64 | slice | None = None,
            latitude: float | slice | None = None,
            longitude: float | slice | None = None,
            **kwargs,
        ) -> T: ...

    # Define enhanced types
    class TypedDataArray(xr.DataArray):
        jua: JuaAccessorProtocol["TypedDataArray"]

    class TypedDataset(xr.Dataset):
        jua: JuaAccessorProtocol["TypedDataset"]

        # This is the key addition - make __getitem__ return the TypedDataArray
        def __getitem__(self, key: any) -> "TypedDataArray": ...

    # Monkey patch the xarray types
    xr.DataArray = TypedDataArray  # type: ignore
    xr.Dataset = TypedDataset  # type: ignore


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

    def __getitem__(self, key: any) -> xr.DataArray:
        return self._raw_data[str(key)]

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
