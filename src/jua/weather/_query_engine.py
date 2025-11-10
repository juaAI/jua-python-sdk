"""Interface with the Query Engine"""

from datetime import datetime
from logging import getLogger
from typing import Literal

import pandas as pd
import xarray as xr
from pydantic import validate_call

from jua._api import QueryEngineAPI
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua.client import JuaClient
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather._model_meta import get_model_meta_info
from jua.weather._stream import process_arrow_streaming_response
from jua.weather._types.forecast import ForecastData
from jua.weather._types.query_payload_types import (
    ForecastIndexQueryPayload,
    ForecastQueryPayload,
    build_geo_filter,
    build_init_time_arg,
    build_prediction_timedelta,
)
from jua.weather._types.query_response_types import (
    AvailableForecastsQueryResult,
    LatestForecastInfoQueryResult,
    MetaQueryResult,
)
from jua.weather.models import Models
from jua.weather.statistics import Statistics
from jua.weather.variables import Variables

logger = getLogger(__name__)


class QueryEngine:
    """Internal API client for Jua's weather services.

    Note:
        This class is intended for internal use only and should not be used directly.
        End users should interact with the higher-level classes.
    """

    _FORECAST_ENDPOINT = "forecast/data"

    def __init__(self, jua_client: JuaClient):
        """Initialize the weather API client.

        Args:
            jua_client: JuaClient instance for authentication and settings.
        """
        self._api = QueryEngineAPI(jua_client)
        self._jua_client = jua_client

        # (30x30 HighRes grid), 1 month, 4 forecasts per day, 49 hours of forecast
        self._MAX_POINTS_PER_REQUEST = (361 * 361) * 31 * 4 * (2 * 24 + 1)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_available_forecasts(
        self,
        model: Models,
        since: datetime | None = None,
        before: datetime | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> AvailableForecastsQueryResult:
        """Get available forecast initialization times for a model.

        Args:
            model: The model to query for available forecasts
            since: Only return forecasts initialized on or after this datetime
            before: Only return forecasts initialized before this datetime
            limit: Maximum number of results to return per page
            offset: Number of results to skip for pagination

        Returns:
            Query result containing available forecast times and pagination info.
            The max_prediction_timedelta values are converted from minutes to hours.

        Examples:
            >>> api = QueryEngine(jua_client)
            >>> result = api.get_available_forecasts(
            ...     model=Models.EPT2,
            ...     since=datetime(2025, 1, 1),
            ...     limit=20
            ... )
            >>> for forecast in result.forecasts_per_model['ept2']:
            ...     print(f"Init time: {forecast.init_time}")
            ...     print(f"Max lead time: {forecast.max_prediction_timedelta} hours")
        """
        params = {
            "models": [model.value],
            "limit": limit,
            "offset": offset,
        }

        if since is not None:
            params["since"] = since.isoformat()

        if before is not None:
            params["before"] = before.isoformat()

        response = self._api.get("forecast/available-forecasts", params=params)
        result = AvailableForecastsQueryResult(**response.json())

        # Convert max_prediction_timedelta from minutes to hours
        for model_forecasts in result.forecasts_per_model.values():
            for forecast_info in model_forecasts:
                if forecast_info.max_prediction_timedelta is not None:
                    forecast_info.max_prediction_timedelta = (
                        forecast_info.max_prediction_timedelta // 60
                    )

        return result

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_latest_init_time(
        self,
        model: Models,
        min_prediction_timedelta: int = 0,
    ) -> LatestForecastInfoQueryResult:
        """Get the latest available forecast initialization time for a model.

        Args:
            model: The model to query for latest forecast info
            min_prediction_timedelta: Minimum required lead time in hours (default: 0)

        Returns:
            Query result containing the latest forecast initialization time and max
            lead time.

        Examples:
            >>> api = QueryEngine(jua_client)
            >>> result = api.get_latest_init_time(
            >>>     model=Models.EPT2,
            >>>     min_prediction_timedelta=48
            >>> )
            >>> latest = result.forecasts_per_model['ept2']
            >>> print(f"Latest init time: {latest.init_time}")
            >>> print(f"Max lead time: {latest.prediction_timedelta} hours")
        """
        # Convert hours to minutes for the API call
        params = {
            "models": [model.value],
            "min_prediction_timedelta": min_prediction_timedelta * 60,
        }

        response = self._api.get("forecast/latest-init-time", params=params)
        result = LatestForecastInfoQueryResult(**response.json())

        # Convert prediction_timedelta from minutes to hours
        for forecast_info in result.forecasts_per_model.values():
            forecast_info.prediction_timedelta = (
                forecast_info.prediction_timedelta // 60
            )

        return result

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_meta(
        self,
        model: Models,
    ) -> MetaQueryResult:
        """Get metadata for a forecast model including available variables and grid.

        Args:
            model: The model to query for metadata

        Returns:
            Query result containing model metadata including variables and grid

        Examples:
            >>> api = QueryEngine(jua_client)
            >>> result = api.get_meta(model=Models.EPT2)
            >>> model_info = result.models[0]
            >>> print(f"Model: {model_info.model}")
            >>> print(f"Variables: {model_info.variables}")
            >>> print(f"Grid: {model_info.grid}")
        """
        params = {
            "models": [model.value],
        }

        response = self._api.get("forecast/meta", params=params)
        return MetaQueryResult.model_validate(response.json())

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_forecast_index(
        self,
        model: Models,
        init_time: Literal["latest"] | datetime | list[datetime] | slice,
        variables: list[Variables] | list[str],
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: slice | None = None,
        longitude: slice | None = None,
    ) -> dict[str, list]:
        """Get the index (coordinates) for a forecast query without loading data.

        This is more efficient than loading actual data when you only need to know
        what coordinates are available for a query.

        Args:
            model: The model to query
            init_time: Filter by forecast initialization time
            variables: List of weather variables
            prediction_timedelta: Filter by forecast lead time
            latitude: Latitude selection
            longitude: Longitude selection

        Returns:
            Dictionary with keys: init_time, prediction_timedelta, latitude,
            longitude, variables. Each containing a list of available coordinate
            values.

        Raises:
            ValueError: If latitude/longitude are not ranges/slices
                (index endpoint requires ranges)
        """
        # Convert latitude/longitude to ranges
        if latitude is None:
            raise ValueError("latitude is required for forecast index")
        if longitude is None:
            raise ValueError("longitude is required for forecast index")

        # Build the payload using the proper type
        init_time_value = build_init_time_arg(init_time)
        prediction_timedelta_value = build_prediction_timedelta(prediction_timedelta)

        # Normalize variables
        variable_names = [
            v.name if isinstance(v, Variables) else str(v) for v in variables
        ]

        payload = ForecastIndexQueryPayload(
            model=model,
            latitude=(latitude.start, latitude.stop),
            longitude=(longitude.start, longitude.stop),
            init_time=init_time_value,
            prediction_timedelta=prediction_timedelta_value,
            variables=variable_names,
        )

        query_params = {}
        if self._jua_client.request_credit_limit is not None:
            query_params["request_credit_limit"] = str(
                self._jua_client.request_credit_limit
            )

        response = self._api.post(
            "forecast/index",
            data=payload.model_dump(exclude_none=True),
            query_params=query_params,
        )

        result = response.json()

        # Convert prediction_timedelta from minutes to hours
        if "prediction_timedelta" in result:
            result["prediction_timedelta"] = pd.to_timedelta(
                result["prediction_timedelta"], unit="m"
            )

        return result

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_forecast(
        self,
        model: Models,
        init_time: Literal["latest"] | datetime | list[datetime] | slice | None = None,
        variables: list[Variables] | list[str] | None = None,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | LatLon | None = None,
        statistics: list[str] | list[Statistics] | None = None,
        method: Literal["nearest", "bilinear"] = "nearest",
        stream: bool | None = None,
        print_progress: bool | None = None,
    ) -> ForecastData:
        """Get a forecast for a specific model and initialization time.

        Args:
            model_name: The name of the model for which to get the forecast.

            init_time: Filter by forecast initialization time. Can be:
                - None: All available initialization times (default)
                - A single datetime: Specific initialization time
                - A list of datetimes: Multiple specific times
                - A slice(start, end): Range of initialization times

            variables: List of weather variables to include.

            prediction_timedelta: Filter by forecast lead time. Can be:
                - None: All available lead times (default)
                - A single value (hours or timedelta): Specific lead time
                - A slice(start, stop): Range of lead times
                - A slice(start, stop, step): Lead times at regular intervals

            latitude: Latitude selection. Can be a single value, list of values, or
                a slice(min_lat, max_lat) for a geographical range.

            longitude: Longitude selection. Can be a single value, list of values, or
                a slice(min_lon, max_lon) for a geographical range.

            points: Specific geographic points to get forecasts for. Can be a single
                LatLon object or a list of LatLon objects.

            statistics: For ensemble models, the statistics to return.

            method: Interpolation method for selecting points:
                - "nearest": Use nearest grid point (default)
                - "bilinear": Bilinear interpolation to the selected point

            stream: Whether to stream the response content. Recommended when querying
                slices or large amounts of data. Default is set to False for points,
                and True for grid slices. Streaming does not support method="bilinear"
                when requesting points.

            print_progress: Whether to display a progress bar during data loading.
                If None, uses the client's default setting. Only works when stream=True.

        Returns:
            Forecast data.

        Raises:
            ValueError: If the location parameters are invalid.
        """
        if isinstance(points, LatLon):
            points = [points]

        geo = build_geo_filter(latitude, longitude, points, method)
        model_meta = get_model_meta_info(model)
        if not model_meta.has_grid_access and geo.type != "point":
            raise ValueError(
                f"There is no access to grid slices with {model}. You can only make "
                "point queries."
            )

        stats: list[Statistics] = []
        if statistics is not None:
            for s in statistics:
                if isinstance(s, str):
                    stats.append(Statistics.from_key(s))
                elif isinstance(s, Statistics):
                    stats.append(s)
                else:
                    raise ValueError(
                        f"`statistics` must be a list of statistics; found {s}"
                    )

        aggregation = [s.agg for s in stats] if stats else None
        df = self.load_raw_forecast(
            payload=ForecastQueryPayload(
                models=[model],
                init_time=build_init_time_arg(init_time),
                geo=geo,
                prediction_timedelta=build_prediction_timedelta(prediction_timedelta),
                variables=variables,
                aggregation=aggregation,
            ),
            stream=geo.type != "point" if stream is None else stream,
            print_progress=print_progress,
        )

        if geo.type == "point":
            if isinstance(points, list):
                points = points
            else:
                points = [LatLon(lat=lat, lon=lon) for lat, lon in geo.value]  # type: ignore

        return self.transform_dataframe(
            df,
            points=points,  # type: ignore
            statistics=stats,
        )

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def load_raw_forecast(
        self,
        payload: ForecastQueryPayload,
        stream: bool = False,
        print_progress: bool | None = None,
    ) -> pd.DataFrame:
        if payload.geo.type == "point" and payload.geo.method == "bilinear" and stream:
            logger.warning(
                "Cannot use streaming responses with bilinear interpolation. Setting "
                "stream=False."
            )
            stream = False

        est_requested_points = payload.num_requested_points()
        if est_requested_points > self._MAX_POINTS_PER_REQUEST:
            raise ValueError(
                "The requested data volume is too large for a single call. "
                f"Estimated size is {est_requested_points} points, which exceeds the "
                f"limit of {self._MAX_POINTS_PER_REQUEST}. The total rows equal "
                "number_of_points x number_of_lead_times x number_of_init_times. "
                "Please split your request into smaller chunks (e.g., fewer points, a "
                "smaller init_time range, or fewer lead times)."
            )

        if print_progress is None:
            print_progress = self._jua_client.settings.print_progress

        data = remove_none_from_dict(payload.model_dump())
        query_params = {"format": "arrow", "stream": str(stream).lower()}
        if self._jua_client.request_credit_limit is not None:
            query_params["request_credit_limit"] = str(
                self._jua_client.request_credit_limit
            )

        response = self._api.post(
            self._FORECAST_ENDPOINT,
            data=data,
            query_params=query_params,
            extra_headers={"Accept": "*/*", "Accept-Encoding": "identity"},
            stream=stream,
        )
        # We can only print progress for streaming responses
        df = process_arrow_streaming_response(response, print_progress and stream)
        if df.empty:
            raise ValueError("No data available for the given parameters.")

        # Parse times to correct units, enforce correct encoding
        df["init_time"] = df["init_time"].astype("datetime64[ns]")
        df["prediction_timedelta"] = pd.to_timedelta(
            df["prediction_timedelta"], unit="m"
        )

        # Remove unused metadata columns
        num_models = len(df["model"].unique())
        if not num_models == 1:
            raise ValueError(f"Unexpected number of models returned: {num_models}")
        cols_to_drop = ["model"]
        if "time" in df.columns:
            cols_to_drop.append("time")
        df.drop(columns=cols_to_drop, inplace=True)
        return df

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def transform_dataframe(
        cls,
        df: pd.DataFrame,
        points: list[LatLon] | None = None,
        statistics: list[Statistics] | None = None,
    ) -> xr.Dataset:
        """Transform a raw DataFrame from the query engine into an xr.Dataset.

        This method converts tabular forecast data into a structured xarray Dataset with
        appropriate dimensions, coordinates, and data types. It handles both grid-based
        and point-based queries, and restructures ensemble statistics when requested.

        Args:
            df: DataFrame to convert. Must have `init_time`, `model`,
                `prediction_timedelta`, `latitude` and `longitude` columns. For ensemble
                models with statistics, columns are named as `{stat}__{variable}`.
            points: The points requested, if a point request was made. When provided,
                creates a "points" dimension and adds both requested and actual grid
                coordinates as metadata.
            statistics: The statistics requested for ensemble models. When provided,
                variables with stat prefixes are combined into a single variable with
                a "stat" dimension.

        Returns:
            An xarray Dataset with the following structure:

            **Grid queries** (points is None):
                - Dimensions: `init_time`, `prediction_timedelta`, `latitude`,
                    `longitude`
                - Variables: Weather variables as float32

            **Point queries** (points is provided):
                - Dimensions: `points`, `init_time`, `prediction_timedelta`
                - Coordinates:
                    - `latitude`: Actual grid latitude for each point
                    - `longitude`: Actual grid longitude for each point
                    - `requested_lat`: Originally requested latitude
                    - `requested_lon`: Originally requested longitude
                - Variables: Weather variables as float32

            **With statistics** (statistics is not None or empty):
                - Adds a `stat` dimension to all variables
                - Variables like `mean__temperature` and `max__temperature` are combined
                  into a single `temperature` variable with shape (..., stat)
                - The `stat` coordinate contains statistic keys (e.g., "mean", "max")

        Note:
            - All data variables are converted to float32 for memory efficiency
            - Init time encoding is set to nanoseconds since epoch
        """
        # Set the correct index
        if points is not None:
            # Map point indices to requested lat/lon and point objects
            df["requested_lat"] = df["point"].apply(lambda idx: points[idx].lat)
            df["requested_lon"] = df["point"].apply(lambda idx: points[idx].lon)

            # Convert point objects to string for use as dimension
            df["points"] = df["point"].apply(lambda idx: str(points[idx]))

            # Keep track of both actual and requested lat/lon for each point
            point_coords = (
                df[
                    [
                        "points",
                        "latitude",
                        "longitude",
                        "requested_lat",
                        "requested_lon",
                    ]
                ]
                .drop_duplicates()
                .set_index("points")
            )

            cols_to_drop = [
                "point",
                "latitude",
                "longitude",
                "requested_lat",
                "requested_lon",
            ]
            df.drop(cols_to_drop, inplace=True, axis=1)
            df.set_index(
                ["points", "init_time", "prediction_timedelta"],
                inplace=True,
            )

            ds = xr.Dataset.from_dataframe(df)
            # Align point_coords with the xarray dataset's points dimension order
            point_coords_aligned = point_coords.loc[ds.points.values]
            ds = ds.assign_coords(
                {
                    "latitude": ("points", point_coords_aligned["latitude"].values),
                    "longitude": ("points", point_coords_aligned["longitude"].values),
                    "requested_lat": (
                        "points",
                        point_coords_aligned["requested_lat"].values,
                    ),
                    "requested_lon": (
                        "points",
                        point_coords_aligned["requested_lon"].values,
                    ),
                }
            )
        else:
            df.set_index(
                ["init_time", "prediction_timedelta", "latitude", "longitude"],
                inplace=True,
            )
            # Remove duplicates, if there are any (remove once duplicates are handeled)
            df = df.loc[~df.index.duplicated()]
            ds = xr.Dataset.from_dataframe(df)

        # Set the dtype for all data_vars to float32
        for var in ds.data_vars:
            ds[var] = ds[var].astype("float32")

        # Set the correct init_time encoding
        ds.init_time.encoding = {
            "dtype": "int64",
            "units": "nanoseconds since 1970-01-01T00:00:00",
        }

        # obtain statistics from the DataVars if they were requested
        if (
            statistics is None
            or len(statistics) == 0
            or statistics == [Statistics.MEAN]
        ):
            rename_dict = {
                var: var.replace("avg__", "") for var in ds.data_vars if "avg__" in var
            }
            if rename_dict:
                ds = ds.rename(rename_dict)
        elif statistics:
            base_stat = statistics[0]
            vars = [  # get var names using the first statistic
                v.split("__")[1]
                for v in ds.data_vars
                if v.startswith(f"{base_stat.agg}__")
            ]
            for var in vars:
                # collect variable data
                arrays = []
                for stat in statistics:
                    da_stat = ds[f"{stat.agg}__{var}"]
                    da_stat["stat"] = stat.key
                    arrays.append(da_stat)

                # drop stat DataArrays from the Dataset
                for stat in statistics:
                    ds = ds.drop_vars(f"{stat.agg}__{var}")

                # add the new DataArray with a combined statistics
                ds[var] = xr.concat(arrays, dim="stat")

        return ds
