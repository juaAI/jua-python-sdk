from pydantic.dataclasses import dataclass


@dataclass
class Coordinate:
    lat: float
    lon: float
