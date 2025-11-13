from datetime import datetime
from typing import Any, Iterable, Literal

import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core import indexing

from jua import JuaClient
from jua.types.geo import PredictionTimeDelta
from jua.weather._lazy_loading.cache import ForecastCache
from jua.weather._query_engine import QueryEngine
from jua.weather.models import Models
from jua.weather.variables import Variables


class JuaQueryEngineArray(BackendArray):
    """Lazy array that pulls a single variable from a shared cache on demand.

    This uses a shared ForecastCache that loads all variables at once, avoiding
    multiple API calls. It supports BASIC indexing through xarray's
    explicit_indexing_adapter.
    """

    def __init__(
        self,
        *,
        cache: ForecastCache,
        variable: str,
    ) -> None:
        """Initialize the array with a shared cache.

        Args:
            cache: Shared ForecastCache containing all variables
            variable: Name of the specific variable this array represents
        """
        self._cache = cache
        self._variable = variable
        self.shape = cache.shape
        self.dtype = cache.get_dtype()

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        """Get data from the shared cache."""
        arr = self._cache.get_variable(self._variable, key)

        # Squeeze dimensions where integer indexing was used
        squeeze_axes = []
        for i, k in enumerate(key):
            if isinstance(k, (int, np.integer)):
                squeeze_axes.append(i)
        if squeeze_axes:
            arr = np.squeeze(arr, axis=tuple(squeeze_axes))

        return arr


class JuaQueryEngineBackend(BackendEntrypoint):
    """Xarray backend that lazily loads forecast data from Jua Query Engine.

    Usage example:
        from jua.client import JuaClient
        from jua.weather._query_engine import QueryEngine

        client = JuaClient()
        query_engine = QueryEngine(client)
        ds = xr.open_dataset(
            Models.EPT2,
            query_engine=query_engine,
            variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name],
            init_time=slice("2025-01-01", "2025-01-02"),
            latitude=slice(72.0, 36.0),
            longitude=slice(-15.0, 35.0),
        )
    """

    open_dataset_parameters = [
        "filename_or_obj",
        "drop_variables",
        "query_engine",
        "model",
        "variables",
        "init_time",
        "prediction_timedelta",
        "latitude",
        "longitude",
        "points",
        "grid_chunk",
    ]

    description = "Lazy forecast access via Jua Query Engine"
    url = "https://docs.jua.ai"

    def open_dataset(
        self,
        filename_or_obj: Models,
        *,
        query_engine: QueryEngine | None = None,
        variables: list[Variables] | list[str] | None = None,
        init_time: Literal["latest"] | datetime | list[datetime] | slice = "latest",
        prediction_timedelta: PredictionTimeDelta | None = None,
        min_lead_time: int | None = None,
        max_lead_time: int | None = None,
        latitude: slice | None = None,
        longitude: slice | None = None,
        grid_chunk: int = 8,
        drop_variables: Iterable[str] | None = None,
    ) -> xr.Dataset:
        if query_engine is None:
            query_engine = QueryEngine(JuaClient())

        # Parse the args
        model = filename_or_obj
        if variables is None or len(variables) == 0:
            variables = [Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
        variables = [v.name if isinstance(v, Variables) else str(v) for v in variables]

        # Get and parse the forecast index
        index_result = query_engine.get_forecast_index(
            model=model,
            init_time=init_time,
            variables=variables,
            prediction_timedelta=prediction_timedelta,
            latitude=latitude,
            longitude=longitude,
        )
        init_times = np.array(index_result["init_time"], dtype="datetime64[ns]")
        prediction_timedeltas = np.array(index_result["prediction_timedelta"])
        latitudes = np.array(index_result["latitude"], dtype="float32")
        longitudes = np.array(index_result["longitude"], dtype="float32")
        dims = ("init_time", "prediction_timedelta", "latitude", "longitude")
        coords = {
            "init_time": init_times,
            "prediction_timedelta": prediction_timedeltas,
            "latitude": latitudes,
            "longitude": longitudes,
        }

        shared_cache = ForecastCache(
            query_engine=query_engine,
            model=model,
            variables=variables,
            init_times=init_times,
            prediction_timedeltas=prediction_timedeltas,
            latitudes=latitudes,
            longitudes=longitudes,
            increasing_lats=latitudes[1] > latitudes[0],
            increasing_lons=longitudes[1] > longitudes[0],
            original_kwargs=dict(
                init_time=init_time,
                prediction_timedelta=prediction_timedelta,
                latitude=latitude,
                longitude=longitude,
            ),
            grid_chunk=grid_chunk,
        )

        # Create lazy arrays that all share the same cache
        data_vars: dict[str, tuple[tuple[str, ...], Any]] = {}
        for var_name in variables:
            backend_array = JuaQueryEngineArray(
                cache=shared_cache,
                variable=var_name,
            )
            data = indexing.LazilyIndexedArray(backend_array)
            data_vars[var_name] = (dims, data)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        if drop_variables is not None:
            ds = ds.drop_vars(list(drop_variables))

        # Ensure closing the dataset frees the shared cache memory
        def _close_hook() -> None:
            shared_cache.clear()

        ds.set_close(_close_hook)
        return ds

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        if isinstance(filename_or_obj, Models):
            return True
        return False
