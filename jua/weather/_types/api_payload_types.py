from pydantic import BaseModel, Field

from jua.types.geo import LatLon


class ForecastRequestPayload(BaseModel):
    points: list[LatLon] = Field(default_factory=list)
    min_lead_time: int = 0
    max_lead_time: int = 0
    variables: list[str] | None = None
    full: bool = False
