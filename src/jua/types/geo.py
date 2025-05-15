import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass


def _label_to_key(label: str) -> str:
    return label.lower().replace(" ", "_")


@dataclass
class LatLon:
    """Geographic coordinate representing a points on Earth's surface.

    Attributes:
        lat: Latitude in decimal degrees (range: -90 to 90).
        lon: Longitude in decimal degrees (range: -180 to 180).
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
SpatialSelection = float | slice | list[float]
