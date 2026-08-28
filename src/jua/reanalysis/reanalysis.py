from datetime import datetime
from typing import Literal

from pydantic import validate_call

from jua.client import JuaClient
from jua.logging import get_logger
from jua.reanalysis._query_engine import ReanalysisQueryEngine
from jua.reanalysis._types.query_response_types import ReanalysisModelInfo
from jua.reanalysis.models import ReanalysisModels
from jua.types.geo import LatLon, SpatialSelection
from jua.weather import JuaDataset
from jua.weather.variables import Variables

logger = get_logger(__name__)


class Reanalysis:
    """Access to Jua's reanalysis datasets (ERA5).

    Reanalysis is a reconstruction of past weather, so it has a single `time`
    dimension: there is no `init_time` or `prediction_timedelta`. Use it for
    historical analysis, for training and validating models, and for scoring
    forecasts against what actually happened.

    Accessed through the `reanalysis` property of a JuaClient.

    Examples:
        >>> from datetime import datetime
        >>> from jua import JuaClient
        >>> from jua.types.geo import LatLon
        >>> from jua.weather import Variables
        >>>
        >>> client = JuaClient()
        >>>
        >>> # A time series for two cities
        >>> zurich = LatLon(lat=47.3769, lon=8.5417)
        >>> london = LatLon(lat=51.5074, lon=-0.1278)
        >>> data = client.reanalysis.get_data(
        ...     time=slice(datetime(2024, 1, 1), datetime(2024, 2, 1)),
        ...     variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
        ...     points=[zurich, london],
        ... )
        >>>
        >>> # A region, then save it to Zarr
        >>> europe = client.reanalysis.get_data(
        ...     time=slice(datetime(2024, 6, 1), datetime(2024, 6, 2)),
        ...     variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
        ...     latitude=slice(71, 36),
        ...     longitude=slice(-15, 50),
        ... )
        >>> europe.save()
    """

    def __init__(self, client: JuaClient) -> None:
        self._client = client
        self._query_engine = ReanalysisQueryEngine(client)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_data(
        self,
        # `str` is accepted only so `build_time_arg` can explain why "latest" is
        # unsupported. Without it, pydantic rejects the value first and the user
        # gets three type errors instead of the reason. The API documents
        # "latest", so this is a mistake worth answering properly.
        time: datetime | list[datetime] | slice | str,
        variables: list[Variables] | list[str] | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | LatLon | None = None,
        method: Literal["nearest", "bilinear"] = "nearest",
        model: ReanalysisModels = ReanalysisModels.ERA5,
        stream: bool | None = None,
        print_progress: bool | None = None,
    ) -> JuaDataset:
        """Retrieve reanalysis data.

        Args:
            time: The time(s) to retrieve. Required - unlike forecasts there is
                no meaningful default. Can be:
                - A slice(start, stop): a range of times (the common case)
                - A single datetime: one timestamp
                - A list of datetimes: specific timestamps

                There is no "latest" selector. ERA5 is published with a lag of
                roughly five days, so pick your times accordingly.

            variables: Weather variables to retrieve. If None, all variables
                available for the dataset are returned.

            latitude: Latitude selection. A single value, a list of values, or a
                slice(min_lat, max_lat) for a bounding box.

            longitude: Longitude selection. A single value, a list of values, or
                a slice(min_lon, max_lon) for a bounding box.

            points: Specific geographic points. A single LatLon or a list.
                Mutually exclusive with latitude/longitude.

            method: How to sample at a point:
                - "nearest" (default): the containing grid cell
                - "bilinear": interpolate from the four surrounding cells

            model: Which reanalysis dataset. Defaults to ERA5.

            stream: Whether to stream the response. Defaults to True for grid
                slices and False for points.

            print_progress: Whether to show a progress bar. If None, uses the
                client's default. Only applies when streaming.

        Returns:
            JuaDataset with dimensions (time, latitude, longitude) for a grid
            query, or (points, time) for a point query.

        Raises:
            ValueError: If both points and latitude/longitude are given, if
                neither is given, or if the request is too large for one call.

        Examples:
            >>> # One month of hourly temperature over Germany
            >>> data = client.reanalysis.get_data(
            ...     time=slice(datetime(2024, 1, 1), datetime(2024, 2, 1)),
            ...     variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],
            ...     latitude=slice(55, 47),
            ...     longitude=slice(5, 15),
            ... )
            >>> ds = data.to_xarray()
            >>> ds.air_temperature_at_height_level_2m.mean(dim="time").plot()
        """
        if points is not None and (latitude is not None or longitude is not None):
            raise ValueError(
                "Cannot provide both points and latitude/longitude. "
                "Please provide either points or latitude/longitude."
            )

        ds = self._query_engine.get_data(
            model=model,
            time=time,
            variables=variables,
            latitude=latitude,
            longitude=longitude,
            points=points,
            method=method,
            stream=stream,
            print_progress=print_progress,
        )
        return JuaDataset(
            settings=self._client.settings,
            dataset_name=f"{model.value}_{_dataset_name_suffix(time)}",
            raw_data=ds,
            model=model,
        )

    @validate_call
    def get_variables(
        self, model: ReanalysisModels = ReanalysisModels.ERA5
    ) -> list[str]:
        """List the variables available for a reanalysis dataset.

        Args:
            model: Which reanalysis dataset. Defaults to ERA5.

        Returns:
            Variable names, as accepted by `get_data(variables=...)`.
        """
        return self._query_engine.get_model_meta(model).variables

    @validate_call
    def get_metadata(
        self, model: ReanalysisModels = ReanalysisModels.ERA5
    ) -> ReanalysisModelInfo:
        """Get metadata for a reanalysis dataset.

        Args:
            model: Which reanalysis dataset. Defaults to ERA5.

        Returns:
            Grid resolution, temporal resolution, variables and their units.
        """
        return self._query_engine.get_model_meta(model)


def _dataset_name_suffix(
    time: datetime | list[datetime] | slice | str,
) -> str:
    """Build a filesystem-friendly suffix describing the time selection.

    `str` cannot reach here in practice — `build_time_arg` rejects it inside the
    query call above — but it stays in the signature so the type matches
    `get_data`'s parameter rather than needing a cast.
    """
    if isinstance(time, str):
        return time
    if isinstance(time, slice):
        return f"{_stamp(time.start)}_{_stamp(time.stop)}"
    if isinstance(time, list):
        return f"{_stamp(time[0])}_{_stamp(time[-1])}" if time else "empty"
    return _stamp(time)


def _stamp(value: datetime | None) -> str:
    if value is None:
        return "none"
    return value.strftime("%Y%m%dT%H%M")
