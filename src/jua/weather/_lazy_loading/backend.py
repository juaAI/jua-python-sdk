from typing import Any, Iterable

import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core import indexing

from jua import JuaClient
from jua.types.geo import PredictionTimeDelta, SpatialSelection
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

        # Get shape from cache
        self.shape = cache.shape

        # Until first load, assume float64
        self.dtype = np.dtype("float64")

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:  # type: ignore[name-defined]
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:  # type: ignore[name-defined]
        """Get data from the shared cache."""
        # Pass the key to the cache so it knows what subset to load
        arr = self._cache.get_variable(self._variable, key)
        # Update dtype after first load
        self.dtype = arr.dtype

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
            engine="jua_query_engine",
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
        init_time: Any | None = None,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        grid_chunk: int = 8,
        drop_variables: Iterable[str] | None = None,
    ) -> xr.Dataset:
        if query_engine is None:
            query_engine = QueryEngine(JuaClient())

        # Model from filename
        model = filename_or_obj

        # Variables normalization
        if variables is None or len(variables) == 0:
            variables = [Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M.name]
        else:
            variables = [
                v.name if isinstance(v, Variables) else str(v)  # type: ignore[attr-defined]
                for v in variables
            ]

        # Use the more efficient index endpoint
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

        # No open files but provide a close hook for API symmetry
        def _noop_close() -> None:
            return None

        ds.set_close(_noop_close)
        return ds

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        if isinstance(filename_or_obj, Models):
            return True
        return False
