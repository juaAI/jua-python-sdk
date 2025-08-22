from pydantic import BaseModel

from jua.types.geo import LatLon


class ForecastRequestPayload(BaseModel):
    points: list[LatLon] | None = None
    min_lead_time: int = 0
    max_lead_time: int = 0
    variables: list[str] | None = None
    full: bool = False
    ensemble_stats: list[str] | None = None
