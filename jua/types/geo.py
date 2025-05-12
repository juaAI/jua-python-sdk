from pydantic.dataclasses import dataclass


@dataclass
class LatLon:
    lat: float
    lon: float
