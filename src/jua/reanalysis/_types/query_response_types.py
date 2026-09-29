from pydantic import BaseModel


class ReanalysisModelInfo(BaseModel):
    """Metadata for one reanalysis dataset, from ``GET /v1/reanalysis/meta``."""

    name: str
    display_name: str
    grid_resolution: str
    temporal_resolution_minutes: int
    variables: list[str]
    variable_units: dict[str, str] = {}


class ReanalysisMetaResult(BaseModel):
    """Response of ``GET /v1/reanalysis/meta``."""

    models: list[ReanalysisModelInfo]
