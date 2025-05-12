from datetime import datetime

from pydantic import BaseModel


class FileResponse(BaseModel):
    content_type: str
    last_modified: datetime
    name: str
    parent: str
    path: str
    size: int
    type: str


class DirectoryResponse(BaseModel):
    name: str
    path: str
    type: str
