from abc import ABC, abstractmethod
from datetime import datetime

from jua.client import JuaClient
from jua.errors.jua_error import JuaError
from jua.weather._jua_dataset import JuaDataset
from jua.weather._model import Model


class ModelHasNoRawForecastFileAccess(JuaError):
    def __init__(self, model: Model):
        super().__init__(f"Model {model} has no raw forecast file access")


class ModelHasNoRawHindcastFileAccess(JuaError):
    def __init__(self, model: Model):
        super().__init__(f"Model {model} has no raw hindcast file access")


class FileAccess(ABC):
    def __init__(
        self,
        client: JuaClient,
        model: Model,
        has_forecast_file_access: bool,
        has_hindcast_file_access: bool,
    ):
        self._client = client
        self._model = model
        self._has_forecast_file_access = has_forecast_file_access
        self._has_hindcast_file_access = has_hindcast_file_access

    @property
    def has_forecast_file_access(self) -> bool:
        return self._has_forecast_file_access

    @property
    def has_hindcast_file_access(self) -> bool:
        return self._has_hindcast_file_access

    @abstractmethod
    def _get_hindcast_impl(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list[JuaDataset]:
        pass

    def get_hindcast(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list[JuaDataset]:
        if not self._has_hindcast_file_access:
            raise ModelHasNoRawHindcastFileAccess(self._model)

        return self._get_hindcast_impl(start_date=start_date, end_date=end_date)

    @abstractmethod
    def _get_forecast_impl(self, init_time: datetime | None = None) -> list[JuaDataset]:
        pass

    def get_forecast(self, init_time: datetime | None = None) -> list[JuaDataset]:
        if not self._has_forecast_file_access:
            raise ModelHasNoRawForecastFileAccess(self._model)

        return self._get_forecast_impl(init_time=init_time)
