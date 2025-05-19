from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar

import numpy as np
import xarray as xr
from pydantic import validate_call

from jua.types.geo import LatLon, PredictionTimeDelta
from jua.weather.conversions import to_timedelta
from jua.weather.variables import Variables

"""
This module enhances xarray with Jua-specific functionality by monkey-patching
xarray classes and registering custom accessors.

The key enhancements include:
1. Support for slicing with step for prediction_timedelta
2. Accessors for Jua-specific operations (to_celcius, to_absolute_time, select_point)
3. Enhanced type hints for better IDE support
4. Support for Variables enum members as keys in dictionary-like access

These patches make the SDK's weather data handling more intuitive and user-friendly
by allowing operations specific to weather data to be expressed clearly.
"""

# Store original sel methods
_original_dataset_sel = xr.Dataset.sel
_original_dataarray_sel = xr.DataArray.sel
_original_dataset_getitem = xr.Dataset.__getitem__


def _check_prediction_timedelta(
    prediction_timedelta: int | np.timedelta64 | slice | None,
):
    """Convert various prediction_timedelta formats to a consistent representation.

    This function normalizes different time delta representations to ensure
    they work consistently with xarray operations:
    - None values pass through unchanged
    - Slice objects have their start/stop/step components individually converted
    - Lists of values have each element converted
    - Single values are converted to timedelta format

    Args:
        prediction_timedelta: Time delta value to normalize, which can be:
            - None
            - An integer (interpreted as hours)
            - A numpy.timedelta64 object
            - A slice with int/timedelta start/stop/step values
            - A list of time deltas

    Returns:
        Normalized time delta representation suitable for xarray operations.
    """
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


def _patch_timedelta_slicing(
    available_timedeltas: np.ndarray,
    prediction_timedelta: slice,
    method: str | None = None,
) -> list[int] | slice:
    """Handle complex slicing operations for prediction timedeltas.

    This function handles the case where a user requests a sliced selection
    of prediction time deltas with a non-None step value. It calculates which
    indices in the available timedeltas array correspond to the requested slice.

    Args:
        available_timedeltas: Array of timedelta values available in the dataset
        prediction_timedelta: Slice object specifying start, stop, and step
        method: Selection method ('nearest' or None)

    Returns:
        Either a list of indices or the original slice,
        depending on implementation needs

    Raises:
        ValueError: If start and stop are not provided,
        or if an invalid method is specified
    """
    if prediction_timedelta.step is None:
        return prediction_timedelta
    if prediction_timedelta.start is None or prediction_timedelta.stop is None:
        raise ValueError("start and stop must be provided")

    # Generate all the timedeltas requested by the slice
    requested = [prediction_timedelta.start]
    current = requested[0]
    while True:
        current += prediction_timedelta.step
        if current >= prediction_timedelta.stop:
            break
        requested.append(current)

    # Convert to indices based on the selection method
    if method is None:
        # Exact matching - find the exact indices in the available timedeltas
        return [np.where(available_timedeltas == t)[0][0] for t in requested]
    elif method == "nearest":
        # Nearest matching - find the closest available timedeltas
        indices = [np.argmin(np.abs(available_timedeltas - t)) for t in requested]
        return np.sort(np.unique(indices))
    else:
        raise ValueError(f"Invalid method: {method}")


def _patch_args(
    prediction_timedelta: int | np.timedelta64 | slice | None,
    time: np.datetime64 | slice | None,
    latitude: float | slice | None,
    longitude: float | slice | None,
    **kwargs,
):
    """Process and normalize arguments for patched xarray selection methods.

    This function:
    1. Normalizes prediction_timedelta values
    2. Handles latitude slices
    3. Combines Jua-specific arguments with standard xarray kwargs

    Args:
        prediction_timedelta: Time delta selection parameter
        time: Time selection parameter
        latitude: Latitude selection parameter
        longitude: Longitude selection parameter
        **kwargs: Additional xarray selection parameters

    Returns:
        Dictionary of processed arguments ready for use with xarray selection methods
    """
    prediction_timedelta = _check_prediction_timedelta(prediction_timedelta)

    # Invert latitude slice if needed
    # (since latitude is typically ordered from North to South)
    if isinstance(latitude, slice):
        if latitude.start < latitude.stop:
            latitude = slice(latitude.stop, latitude.start, latitude.step)

    # Collect Jua-specific arguments
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


def _must_use_patch_timedelta_slicing(
    prediction_timedelta: PredictionTimeDelta | None,
) -> bool:
    """Determine if specialized timedelta slicing logic must be used.

    This function checks if the prediction_timedelta requires the specialized
    timedelta slicing logic, which is needed when using slices with non-None steps.

    Args:
        prediction_timedelta: The time delta selection parameter

    Returns:
        True if specialized slicing is needed, False otherwise
    """
    if not isinstance(prediction_timedelta, slice):
        return False
    return prediction_timedelta.step is not None


def _patched_sel(
    original_sel: Callable,
    self,
    *args,
    time: np.datetime64 | slice | None = None,
    prediction_timedelta: PredictionTimeDelta | None = None,
    latitude: float | slice | None = None,
    longitude: float | slice | None = None,
    points: LatLon | list[LatLon] | None = None,
    **kwargs,
):
    """Core implementation of the patched selection method.

    This function is used to patch both Dataset.sel and DataArray.sel.
    It adds support for:
    1. Prediction timedeltas with step values
    2. Point-based selection
    3. Specialized handling of latitude/longitude

    Args:
        original_sel: Original xarray selection method to call
        self: The xarray object (Dataset or DataArray)
        *args: Positional arguments for the selection method
        time: Time selection parameter
        prediction_timedelta: Time delta selection parameter
        latitude: Latitude selection parameter
        longitude: Longitude selection parameter
        points: Points to select (LatLon objects)
        **kwargs: Additional selection parameters

    Returns:
        Selected xarray object (Dataset or DataArray)
    """
    # Process and normalize the arguments
    full_kwargs = _patch_args(
        time=time,
        prediction_timedelta=prediction_timedelta,
        latitude=latitude,
        longitude=longitude,
        **kwargs,
    )

    # If points selection is requested, delegate to the select_point accessor
    if points is not None:
        return self.jua.select_point(*args, points=points, **full_kwargs)

    # Handle special case of timedelta slicing with steps
    prediction_timedelta = full_kwargs.get("prediction_timedelta")
    must_use_patch_timedelta_slicing = _must_use_patch_timedelta_slicing(
        prediction_timedelta
    )

    if must_use_patch_timedelta_slicing:
        # Get indices corresponding to the requested timedeltas
        prediction_timedelta = _patch_timedelta_slicing(
            self.prediction_timedelta.values.flatten(),
            prediction_timedelta,
        )
        # Remove from full_kwargs to avoid passing it to the original method
        del full_kwargs["prediction_timedelta"]

    # Perform the selection using the original method
    data = original_sel(self, *args, **full_kwargs)

    # If needed, apply additional selection based on the timedelta indices
    if must_use_patch_timedelta_slicing:
        data = data.isel(prediction_timedelta=prediction_timedelta)

    return data


# Override Dataset.sel method
def _patched_dataset_sel(
    self,
    *args,
    **kwargs,
):
    """Patched version of xarray.Dataset.sel with Jua-specific enhancements.

    This function adds support for prediction_timedelta, point selection,
    and other Jua-specific selection capabilities to xarray's Dataset.sel method.

    Args:
        *args: Positional arguments for Dataset.sel
        **kwargs: Keyword arguments for Dataset.sel, potentially including:
            - prediction_timedelta: Time delta selection parameter
            - points: Points to select (LatLon objects)

    Returns:
        Selected Dataset
    """
    return _patched_sel(_original_dataset_sel, self, *args, **kwargs)


# Override DataArray.sel method
def _patched_dataarray_sel(
    self,
    *args,
    **kwargs,
):
    """Patched version of xarray.DataArray.sel with Jua-specific enhancements.

    This function adds support for prediction_timedelta, point selection,
    and other Jua-specific selection capabilities to xarray's DataArray.sel method.

    Args:
        *args: Positional arguments for DataArray.sel
        **kwargs: Keyword arguments for DataArray.sel, potentially including:
            - prediction_timedelta: Time delta selection parameter
            - points: Points to select (LatLon objects)

    Returns:
        Selected DataArray
    """
    return _patched_sel(_original_dataarray_sel, self, *args, **kwargs)


# Override Dataset.__getitem__ method
def _patched_dataset_getitem(self, key: Any):
    """Patched version of xarray.Dataset.__getitem__ with Jua-specific enhancements.

    This function adds support for using Variables enum members as keys when
    accessing variables in a Dataset.

    Args:
        key: Key to access (string variable name or Variables enum member)

    Returns:
        Selected DataArray
    """
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
    """Accessor that adds Jua-specific functionality to xarray objects.

    This accessor provides methods for working with weather data, including:
    - Selecting data at specific geographic points
    - Converting temperature units
    - Converting prediction timedeltas to absolute times

    It's accessed via the .jua property on Dataset and DataArray objects.
    """

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        """Initialize the accessor.

        Args:
            xarray_obj: The xarray object (Dataset or DataArray) to enhance
        """
        self._xarray_obj = xarray_obj

    @validate_call
    def select_point(
        self,
        points: LatLon | list[LatLon] | str | list[str],
        method: str | None = "nearest",
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        """Select data at specific geographic points.

        This method delegates to the select_point accessor implementation.

        Args:
            points: One or more geographic points to select
            method: Selection method ('nearest' by default)
            **kwargs: Additional selection parameters

        Returns:
            Selected data for the specified points
        """
        return self._xarray_obj.select_point(points, method, **kwargs)

    def to_celcius(self) -> xr.DataArray:
        """Convert temperature data from Kelvin to Celsius.

        This method delegates to the to_celcius accessor implementation.

        Returns:
            Temperature data in Celsius

        Raises:
            ValueError: If applied to a Dataset rather than a DataArray
        """
        if not isinstance(self._xarray_obj, xr.DataArray):
            raise ValueError("This method only works on DataArrays")
        return self._xarray_obj.to_celcius()

    def to_absolute_time(self) -> xr.DataArray | xr.Dataset:
        """Convert from prediction_timedelta to absolute time coordinates.

        This method delegates to the to_absolute_time accessor implementation.

        Returns:
            Data with absolute_time as a coordinate/dimension
        """
        return self._xarray_obj.to_absolute_time()


@xr.register_dataarray_accessor("to_absolute_time")
@xr.register_dataset_accessor("to_absolute_time")
class ToAbsoluteTimeAccessor:
    """Accessor that adds absolute time conversion functionality to xarray objects.

    This accessor computes absolute time by adding the prediction_timedelta to the
    initialization time, creating a new coordinate and optionally swapping dimensions.

    It's accessed via the .to_absolute_time() method on Dataset and DataArray objects.
    """

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        """Initialize the accessor.

        Args:
            xarray_obj: The xarray object (Dataset or DataArray) to enhance
        """
        self._xarray_obj = xarray_obj

    def __call__(self) -> xr.DataArray | xr.Dataset:
        """Convert from prediction_timedelta to absolute time coordinates.

        This method:
        1. Computes absolute time by adding prediction_timedelta to initialization time
        2. Adds absolute_time as a new coordinate
        3. Optionally swaps the prediction_timedelta dimension with absolute_time

        Returns:
            Data with absolute_time as a coordinate/dimension

        Raises:
            ValueError: If time or prediction_timedelta dimensions are missing
            or invalid
        """
        if not hasattr(self._xarray_obj, "time"):
            raise ValueError("time must be a dimension")
        # empty tuple is also valid
        time = self._xarray_obj.time
        if len(time.shape) != 0 and time.shape[0] != 1:
            raise ValueError("time must be a single value")
        if not hasattr(self._xarray_obj, "prediction_timedelta"):
            raise ValueError("prediction_timedelta must be a dimension")

        # Calculate absolute time by adding prediction_timedelta to init time
        prediction_timedelta = self._xarray_obj.prediction_timedelta
        if len(time.shape) == 0:
            absolute_time = time.values + prediction_timedelta
        else:
            absolute_time = time.values[0] + prediction_timedelta

        # Create a copy and add the new coordinate
        ds = self._xarray_obj.copy(deep=True)
        ds = ds.assign_coords({"absolute_time": absolute_time})

        # If prediction_timedelta is a dimension, swap it with absolute_time
        if len(prediction_timedelta.shape) > 0:
            ds = ds.swap_dims({"prediction_timedelta": "absolute_time"})

        return ds


@xr.register_dataarray_accessor("to_celcius")
class ToCelciusAccessor:
    """Accessor that adds temperature unit conversion to DataArray objects.

    This accessor converts temperature values from Kelvin (the standard unit in
    weather data) to Celsius.

    It's accessed via the .to_celcius() method on DataArray objects.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """Initialize the accessor.

        Args:
            xarray_obj: The DataArray to enhance
        """
        self._xarray_obj = xarray_obj

    def __call__(self) -> xr.DataArray:
        """Convert temperature from Kelvin to Celsius.

        This method applies the K→°C conversion: T(°C) = T(K) - 273.15

        Returns:
            Temperature data in Celsius
        """
        return self._xarray_obj - 273.15


@xr.register_dataarray_accessor("select_point")
@xr.register_dataset_accessor("select_point")
class SelectpointAccessor:
    """Accessor that adds point-based selection to xarray objects.

    This accessor enables selecting data at specific geographic points,
    handling both:
    1. Selection by point dimension (when points are already a dimension)
    2. Selection by latitude/longitude (converting to a points dimension)

    It's accessed via the .select_point() method on Dataset and DataArray objects.
    """

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        """Initialize the accessor.

        Args:
            xarray_obj: The xarray object (Dataset or DataArray) to enhance
        """
        self._xarray_obj = xarray_obj

    def __call__(
        self,
        points: LatLon | list[LatLon] | str | list[str],
        method: str | None = "nearest",
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        """Select data at specific geographic points.

        This method handles two cases:
        1. If 'points' is already a dimension, it selects using point identifiers
        2. Otherwise, it selects by latitude/longitude coordinates

        Args:
            points: One or more geographic points to select, either as:
                - LatLon objects with lat/lon properties
                - String identifiers (if points dimension exists)
            method: Selection method ('nearest' by default)
            **kwargs: Additional selection parameters

        Returns:
            Selected data for the specified points

        Raises:
            ValueError: If no points are provided or if string points are used
                       when no points dimension exists
        """
        # Determine if we have a single point or multiple points
        is_single_point = not isinstance(points, list)
        if is_single_point:
            points = [points]  # type: ignore

        if len(points) == 0:  # type: ignore
            raise ValueError("At least one points must be provided")

        # Case 1: 'points' is already a dimension in the data
        if "points" in self._xarray_obj.dims:
            # Convert points to strings for selection
            points = [str(p) for p in points]  # type: ignore

            # Use the original selection method
            sel_fn = (
                _original_dataset_sel
                if isinstance(self._xarray_obj, xr.Dataset)
                else _original_dataarray_sel
            )
            data = sel_fn(self._xarray_obj, points=points, **kwargs)

            # If only one point was requested, remove the points dimension
            if is_single_point:
                return data.isel(points=0)
            return data

        # Case 2: Need to select by lat/lon coordinates
        # String points aren't supported in this case
        if any(isinstance(p, str) for p in points):  # type: ignore
            raise ValueError("Point must be a LatLon or a list of LatLon")

        # Select each point individually
        point_data = []
        point_keys = []
        for points in points:  # type: ignore
            # Select data at this point's lat/lon
            point_data.append(
                self._xarray_obj.sel(
                    latitude=points.lat,  # type: ignore
                    longitude=points.lon,  # type: ignore
                    method=method,
                    **kwargs,
                )
            )
            point_keys.append(points.key)  # type: ignore

        # Combine the individual point selections
        result = xr.concat(point_data, dim="points")

        # Add the point_keys as coordinates for easier identification
        result = result.assign_coords(point_key=(["points"], point_keys))

        # Create index for key-based selection
        result = result.set_index(points=["point_key"])

        # If only one point was requested, remove the points dimension
        if is_single_point:
            return result.isel(points=0)
        return result


# Need to trick the IDE for proper type hints
TypedDataArray = Any  # type: ignore
TypedDataset = Any  # type: ignore

# Enabling IDE type hints
if TYPE_CHECKING:
    T = TypeVar("T", bound=xr.DataArray | xr.Dataset, covariant=True)

    class JuaAccessorProtocol(Protocol[T]):
        """Protocol defining the interface for the jua accessor.

        This protocol is used for type checking only and defines the methods
        that must be implemented by the jua accessor.
        """

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
        """Enhanced DataArray type with Jua-specific methods and properties.

        This type is used for type checking only and adds Jua-specific
        methods and properties to the standard xarray DataArray.
        """

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
        """Enhanced Dataset type with Jua-specific methods and properties.

        This type is used for type checking only and adds Jua-specific
        methods and properties to the standard xarray Dataset.
        """

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
    """Mark a dataset as having jua accessors for type checking.

    This function is used to cast a standard xarray Dataset to a TypedDataset
    for type checking purposes. It doesn't modify the dataset in any way.

    Args:
        ds: The xarray Dataset to cast

    Returns:
        The same Dataset, but typed as a TypedDataset
    """
    return ds


def as_typed_dataarray(da: xr.DataArray) -> "TypedDataArray":
    """Mark a dataarray as having jua accessors for type checking.

    This function is used to cast a standard xarray DataArray to a TypedDataArray
    for type checking purposes. It doesn't modify the DataArray in any way.

    Args:
        da: The xarray DataArray to cast

    Returns:
        The same DataArray, but typed as a TypedDataArray
    """
    return da


# In xarray_patches.py
__all__ = ["as_typed_dataset", "as_typed_dataarray", "TypedDataArray", "TypedDataset"]
