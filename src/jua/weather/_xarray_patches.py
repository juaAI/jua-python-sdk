from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
import xarray as xr

from jua.logging import get_logger
from jua.weather.conversions import to_timedelta
from jua.weather.variables import Variables

logger = get_logger(__name__)

# Store original sel methods
_original_dataset_sel = xr.Dataset.sel
_original_dataarray_sel = xr.DataArray.sel
_original_dataset_getitem = xr.Dataset.__getitem__


def _check_prediction_timedelta(**kwargs):
    if "prediction_timedelta" in kwargs and not isinstance(
        kwargs["prediction_timedelta"], slice
    ):
        # Convert to timedelta
        if isinstance(kwargs["prediction_timedelta"], list):
            kwargs["prediction_timedelta"] = [
                to_timedelta(t) for t in kwargs["prediction_timedelta"]
            ]
        else:
            kwargs["prediction_timedelta"] = to_timedelta(
                kwargs["prediction_timedelta"]
            )
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
    return kwargs


# Override Dataset.sel method
def _patched_dataset_sel(self, *args, **kwargs):
    """
    This is a patch to the xarray.Dataset.sel method to convert the prediction_timedelta
    argument to a timedelta.
    """
    # Check if prediction_timedelta is in kwargs
    kwargs = _check_prediction_timedelta(**kwargs)

    # Call the original method
    return _original_dataset_sel(self, *args, **kwargs)


# Override DataArray.sel method
def _patched_dataarray_sel(self, *args, **kwargs):
    # Check if prediction_timedelta is in kwargs
    kwargs = _check_prediction_timedelta(**kwargs)

    # Call the original method
    return _original_dataarray_sel(self, *args, **kwargs)


# Override Dataset.__getitem__ method
def _patched_dataset_getitem(self, key: Any):
    if isinstance(key, Variables):
        key = str(key)
    return _original_dataset_getitem(self, key)


# Apply the patches
xr.Dataset.sel = _patched_dataset_sel
xr.DataArray.sel = _patched_dataarray_sel
xr.Dataset.__getitem__ = _patched_dataset_getitem


# Define the actual implementation
@xr.register_dataarray_accessor("jua")
@xr.register_dataset_accessor("jua")
class LeadTimeSelector:
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

    def to_celcius(self) -> xr.DataArray:
        if not isinstance(self._xarray_obj, xr.DataArray):
            raise ValueError("This method only works on DataArrays")
        return self._xarray_obj - 273.15


# Tricking python to enable type hints in the IDE
TypedDataArray = Any  # type: ignore
TypedDataset = Any  # type: ignore

# For type checking only
if TYPE_CHECKING:
    T = TypeVar("T", bound=xr.DataArray | xr.Dataset, covariant=True)

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

        def to_celcius(self) -> TypedDataArray: ...

    # Define enhanced types
    class TypedDataArray(xr.DataArray):  # type: ignore
        jua: JuaAccessorProtocol["TypedDataArray"]

        def sel(self, *args, **kwargs) -> "TypedDataArray": ...

        def isel(self, *args, **kwargs) -> "TypedDataArray": ...

    class TypedDataset(xr.Dataset):  # type: ignore
        jua: JuaAccessorProtocol["TypedDataset"]

        # This is the key addition - make __getitem__ return the TypedDataArray
        def __getitem__(self, key: Any) -> "TypedDataArray": ...

        def sel(self, *args, **kwargs) -> "TypedDataset": ...

        def isel(self, *args, **kwargs) -> "TypedDataset": ...

    # Monkey patch the xarray types
    xr.DataArray = TypedDataArray  # type: ignore
    xr.Dataset = TypedDataset  # type: ignore


# Add helper functions that can be used in runtime code
def as_typed_dataset(ds: xr.Dataset) -> "TypedDataset":
    """Mark a dataset as having jua accessors for type checking."""
    return ds


def as_typed_dataarray(da: xr.DataArray) -> "TypedDataArray":
    """Mark a dataarray as having jua accessors for type checking."""
    return da


# In xarray_patches.py
__all__ = ["as_typed_dataset", "as_typed_dataarray", "TypedDataArray", "TypedDataset"]
