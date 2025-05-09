from pydantic import BaseModel

from jua.types.weather.weather import Coordinate


class ForecastRequestPayload(BaseModel):
    points: list[Coordinate]
    min_lead_time: int = 0
    max_lead_time: int = 0
    variables: list[str] | None = None
    full: bool = False
