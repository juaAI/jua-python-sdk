import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class LatLon:
    """Geographic coordinate representing a point on Earth's surface.

    Attributes:
        lat: Latitude in decimal degrees (range: -90 to 90).
        lon: Longitude in decimal degrees (range: -180 to 180).
    """

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


PredictionTimeDelta = (
    int | np.timedelta64 | slice | list[int] | list[np.timedelta64] | None
)
SpatialSelection = float | slice | list[float] | None
