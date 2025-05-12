from datetime import datetime

from pydantic import BaseModel

from jua.weather._types.forecast import ForecastData
from jua.weather._types.raw_file_access import DirectoryResponse, FileResponse


class AvailableModelsResponse(BaseModel):
    available_models: list[str]


class AvailableInitTimesResponse(BaseModel):
    init_times: list[datetime]


class ForecastMetadataResponse(BaseModel):
    forecast_url: str
    model: str
    init_time: datetime
    available_forecasted_hours: int
    available_variables: list[str]


class ForecastResponse(BaseModel):
    forecast: ForecastData


class BrowseFilesDirectoryResponse(BaseModel):
    contents: list[DirectoryResponse | FileResponse]


BrowseFilesResponse = BrowseFilesDirectoryResponse | FileResponse
