from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
import xarray as xr
from pydantic import validate_call

from jua.logging import get_logger
from jua.types.geo import LatLon, PredictionTimeDelta
from jua.weather.conversions import to_timedelta
from jua.weather.variables import Variables

logger = get_logger(__name__)

# Store original sel methods
_original_dataset_sel = xr.Dataset.sel
_original_dataarray_sel = xr.DataArray.sel
_original_dataset_getitem = xr.Dataset.__getitem__


def _check_prediction_timedelta(
    prediction_timedelta: int | np.timedelta64 | slice | None,
):
    if prediction_timedelta is None:
        return None

    if isinstance(prediction_timedelta, slice):
        # Handle slice case
        start = prediction_timedelta.start
        stop = prediction_timedelta.stop
        step = prediction_timedelta.step

        if start is not None:
            start = to_timedelta(start)
        if stop is not None:
            stop = to_timedelta(stop)
        if step is not None:
            step = to_timedelta(step)

        return slice(start, stop, step)

    if isinstance(prediction_timedelta, list):
        return [to_timedelta(t) for t in prediction_timedelta]
    return to_timedelta(prediction_timedelta)


def _patch_args(
    prediction_timedelta: int | np.timedelta64 | slice | None,
    time: np.datetime64 | slice | None,
    latitude: float | slice | None,
    longitude: float | slice | None,
    **kwargs,
):
    prediction_timedelta = _check_prediction_timedelta(prediction_timedelta)
    if isinstance(latitude, slice):
        if latitude.start < latitude.stop:
            latitude = slice(latitude.stop, latitude.start, latitude.step)

    jua_args = {}
    if prediction_timedelta is not None:
        jua_args["prediction_timedelta"] = prediction_timedelta
    if time is not None:
        jua_args["time"] = time
    if latitude is not None:
        jua_args["latitude"] = latitude
    if longitude is not None:
        jua_args["longitude"] = longitude

    return {**jua_args, **kwargs}


# Override Dataset.sel method
def _patched_dataset_sel(
    self,
    *args,
    time: np.datetime64 | slice | None = None,
    prediction_timedelta: PredictionTimeDelta | None = None,
    latitude: float | slice | None = None,
    longitude: float | slice | None = None,
    points: LatLon | list[LatLon] | None = None,
    **kwargs,
):
    """
    This is a patch to the xarray.Dataset.sel method to convert the prediction_timedelta
    argument to a timedelta.
    """
    # Check if prediction_timedelta is in kwargs
    full_kwargs = _patch_args(
        time=time,
        prediction_timedelta=prediction_timedelta,
        latitude=latitude,
        longitude=longitude,
        **kwargs,
    )

    if points is not None:
        return self.jua.select_point(*args, points=points, **full_kwargs)

    # Call the original method
    return _original_dataset_sel(self, *args, **full_kwargs)


# Override DataArray.sel method
def _patched_dataarray_sel(
    self,
    *args,
    time: np.datetime64 | slice | None = None,
    prediction_timedelta: int | np.timedelta64 | slice | None = None,
    latitude: float | slice | None = None,
    longitude: float | slice | None = None,
    points: LatLon | list[LatLon] | None = None,
    **kwargs,
):
    # Check if prediction_timedelta is in kwargs
    full_kwargs = _patch_args(
        time=time,
        prediction_timedelta=prediction_timedelta,
        latitude=latitude,
        longitude=longitude,
        **kwargs,
    )

    if points is not None:
        return self.jua.select_point(*args, points=points, **full_kwargs)

    # Call the original method
    return _original_dataarray_sel(self, *args, **full_kwargs)


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

    @validate_call
    def select_point(
        self,
        points: LatLon | list[LatLon] | str | list[str],
        method: str | None = "nearest",
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        return self._xarray_obj.select_point(points, method, **kwargs)

    def to_celcius(self) -> xr.DataArray:
        if not isinstance(self._xarray_obj, xr.DataArray):
            raise ValueError("This method only works on DataArrays")
        return self._xarray_obj.to_celcius()

    def to_absolute_time(self) -> xr.DataArray | xr.Dataset:
        return self._xarray_obj.to_absolute_time()


@xr.register_dataarray_accessor("to_absolute_time")
@xr.register_dataset_accessor("to_absolute_time")
class ToAbsoluteTimeAccessor:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._xarray_obj = xarray_obj

    def __call__(self) -> xr.DataArray | xr.Dataset:
        if "time" not in self._xarray_obj.dims:
            raise ValueError("time must be a dimension")
        if self._xarray_obj.time.shape != (1,):
            raise ValueError("time must be a single value")
        if "prediction_timedelta" not in self._xarray_obj.dims:
            raise ValueError("prediction_timedelta must be a dimension")

        absolute_time = (
            self._xarray_obj.time[0].values + self._xarray_obj.prediction_timedelta
        )
        ds = self._xarray_obj.copy(deep=True)
        ds = ds.assign_coords({"absolute_time": absolute_time})
        ds = ds.swap_dims({"prediction_timedelta": "absolute_time"})
        return ds


@xr.register_dataarray_accessor("to_celcius")
class ToCelciusAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._xarray_obj = xarray_obj

    def __call__(self) -> xr.DataArray:
        return self._xarray_obj - 273.15


@xr.register_dataarray_accessor("select_point")
@xr.register_dataset_accessor("select_point")
class SelectpointAccessor:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._xarray_obj = xarray_obj

    def __call__(
        self,
        points: LatLon | list[LatLon] | str | list[str],
        method: str | None = "nearest",
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        is_single_point = not isinstance(points, list)
        if is_single_point:
            points = [points]  # type: ignore

        if len(points) == 0:  # type: ignore
            raise ValueError("At least one points must be provided")

        if "points" in self._xarray_obj.dims:
            points = [str(p) for p in points]  # type: ignore
            sel_fn = (
                _original_dataset_sel
                if isinstance(self._xarray_obj, xr.Dataset)
                else _original_dataarray_sel
            )
            data = sel_fn(self._xarray_obj, points=points, **kwargs)
            if is_single_point:
                return data.isel(points=0)
            return data

        # If points is not a dimension, we need to select the points meaning
        # we cannot support strings
        if any(isinstance(p, str) for p in points):  # type: ignore
            raise ValueError("Point must be a LatLon or a list of LatLon")

        point_data = []
        point_keys = []
        for points in points:  # type: ignore
            point_data.append(
                self._xarray_obj.sel(
                    latitude=points.lat,  # type: ignore
                    longitude=points.lon,  # type: ignore
                    method=method,
                    **kwargs,
                )
            )
            point_keys.append(points.key)  # type: ignore

        result = xr.concat(point_data, dim="points")
        # Add the point_keys as coordinates
        result = result.assign_coords(point_key=(["points"], point_keys))
        # create index for key-based selection
        result = result.set_index(points=["point_key"])
        if is_single_point:
            return result.isel(points=0)
        return result


# Tricking python to enable type hints in the IDE
TypedDataArray = Any  # type: ignore
TypedDataset = Any  # type: ignore

# For type checking only
if TYPE_CHECKING:
    T = TypeVar("T", bound=xr.DataArray | xr.Dataset, covariant=True)

    class JuaAccessorProtocol(Protocol[T]):
        def __init__(self, xarray_obj: T) -> None: ...

        def select_point(
            self,
            points: LatLon | list[LatLon],
            method: str | None = "nearest",
            **kwargs,
        ) -> TypedDataArray | TypedDataset: ...

        def to_celcius(self) -> TypedDataArray: ...

        """Convert the dataarray to celcius"""

        def to_absolute_time(self) -> TypedDataArray: ...

        """Add a new dimension to the dataarray with the total time

        The total time is computed as the sum of the time and the prediction_timedelta.
        """

    # Define enhanced types
    class TypedDataArray(xr.DataArray):  # type: ignore
        jua: JuaAccessorProtocol["TypedDataArray"]

        time: xr.DataArray
        prediction_timedelta: xr.DataArray
        latitude: xr.DataArray
        longitude: xr.DataArray

        def sel(
            self,
            *args,
            prediction_timedelta: int | np.timedelta64 | slice | None = None,
            time: np.datetime64 | slice | None = None,
            latitude: float | slice | None = None,
            longitude: float | slice | None = None,
            points: LatLon | list[LatLon] | None = None,
            **kwargs,
        ) -> "TypedDataArray": ...

        def isel(
            self,
            *args,
            prediction_timedelta: int | np.timedelta64 | slice | None = None,
            time: np.datetime64 | slice | None = None,
            latitude: float | slice | None = None,
            longitude: float | slice | None = None,
            points: LatLon | list[LatLon] | None = None,
            **kwargs,
        ) -> "TypedDataArray": ...

        def to_absolute_time(self) -> "TypedDataArray": ...

        def to_celcius(self) -> "TypedDataArray": ...

        def select_point(
            self,
            points: LatLon | list[LatLon],
            method: str | None = "nearest",
            **kwargs,
        ) -> "TypedDataArray": ...

    class TypedDataset(xr.Dataset):  # type: ignore
        jua: JuaAccessorProtocol["TypedDataset"]

        time: xr.DataArray
        prediction_timedelta: xr.DataArray
        latitude: xr.DataArray
        longitude: xr.DataArray

        # This is the key addition - make __getitem__ return the TypedDataArray
        def __getitem__(self, key: Any) -> "TypedDataArray": ...

        def sel(self, *args, **kwargs) -> "TypedDataset": ...

        def isel(self, *args, **kwargs) -> "TypedDataset": ...

        def to_absolute_time(self) -> "TypedDataset": ...

        def select_point(
            self,
            points: LatLon | list[LatLon],
            method: str | None = "nearest",
            **kwargs,
        ) -> "TypedDataset": ...

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
