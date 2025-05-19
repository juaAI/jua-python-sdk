import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass


def _label_to_key(label: str) -> str:
    """Convert a human-readable label to a machine-friendly key.

    Converts spaces to underscores and makes the string lowercase.

    Args:
        label: Human-readable label string

    Returns:
        Machine-friendly key string
    """
    return label.lower().replace(" ", "_")


@dataclass
class LatLon:
    """Geographic coordinate representing a point on Earth's surface.

    This class represents a geographical point with latitude and longitude coordinates.
    It also supports optional labeling for easier identification when working with
    multiple points.

    The class is designed to work seamlessly with Jua's xarray-based data selection
    methods, allowing direct selection of data at specific geographic locations.

    Attributes:
        lat: Latitude in decimal degrees (range: -90 to 90).
        lon: Longitude in decimal degrees (range: -180 to 180).
        label: Optional human-readable name for the location (e.g., "New York").
        key: Machine-friendly identifier derived from label or coordinates.
            Used for indexing and identification in datasets.

    Examples:
        >>> # Create a point representing Tokyo
        >>> tokyo = LatLon(lat=35.6762, lon=139.6503, label="Tokyo")
        >>> print(tokyo.key)  # "tokyo"
        >>>
        >>> # Create multiple city locations
        >>> cities = [
        ...     LatLon(lat=51.5074, lon=-0.1278, label="London"),
        ...     LatLon(lat=40.7128, lon=-74.0060, label="New York"),
        ...     LatLon(lat=35.6762, lon=139.6503, label="Tokyo")
        ... ]
        >>>
        >>> # Use with xarray's selection methods
        >>> from jua import JuaClient
        >>> from jua.weather import Models
        >>>
        >>> client = JuaClient()
        >>> model = client.weather.get_model(Models.EPT2)
        >>> forecast = model.forecast.get_forecast()
        >>>
        >>> # Select data at Tokyo's location
        >>> tokyo_data = forecast.to_xarray().sel(points=tokyo)
        >>>
        >>> # Select data for multiple cities
        >>> cities_data = forecast.to_xarray().sel(points=cities)
    """

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    label: str | None = None
    key: str | None = None

    def __post_init__(self):
        if self.key is not None:
            return
        if self.label is not None:
            self.key = _label_to_key(self.label)
        else:
            self.key = f"point_{self.lat}_{self.lon}"

    def __str__(self):
        return self.key

    def __repr__(self):
        return (
            f"LatLon(lat={self.lat}, lon={self.lon}, "
            f"label={self.label}, key={self.key})"
        )


PredictionTimeDelta = int | np.timedelta64 | slice | list[int] | list[np.timedelta64]
"""Specification for forecast lead times.

This type represents the various ways to specify prediction time deltas (lead times)
when selecting weather forecast data:

- int: Number of hours (e.g., 24 for 24-hour forecast)
- np.timedelta64: NumPy timedelta (e.g., np.timedelta64(12, 'h'))
- slice: Range of lead times (e.g., slice(0, 48) for 0-48 hours)
- list[int]: Specific lead times in hours (e.g., [6, 12, 24, 48])
- list[np.timedelta64]: Specific lead times as timedeltas

Examples:
    >>> # Get forecast for 24-hour lead time
    >>> forecast.sel(prediction_timedelta=24)
    >>>
    >>> # Get forecast for 0-48 hour range
    >>> forecast.sel(prediction_timedelta=slice(0, 48))
    >>>
    >>> # Get forecast for specific lead times
    >>> forecast.sel(prediction_timedelta=[6, 12, 24, 48])
"""

SpatialSelection = float | slice | list[float]
"""Specification for selecting locations by latitude or longitude.

This type represents the various ways to specify latitude or longitude
selections when retrieving weather data:

- float: Single value (e.g., 51.5 for latitude)
- slice: Range of values (e.g., slice(30, 60) for latitude 30°-60°)
- list[float]: List of specific values (e.g., [45.0, 50.0, 55.0])

Examples:
    >>> # Get data for a single point
    >>> forecast.sel(latitude=51.5, longitude=-0.13)
    >>>
    >>> # Get data for a region (Europe)
    >>> forecast.sel(latitude=slice(70, 35), longitude=slice(-10, 30))
    >>>
    >>> # Get data for specific latitudes
    >>> forecast.sel(latitude=[45.0, 50.0, 55.0], longitude=-0.13)
"""
