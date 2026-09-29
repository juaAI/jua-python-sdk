"""Interface with the Query Engine's reanalysis endpoints.

Deliberately mirrors :mod:`jua.weather._query_engine`, because the two endpoints
share their geo filter, their time-selection shapes, and their Arrow transport.
The three places they genuinely differ are marked DIFFERS below; each one is a
crash or a wrong result if the forecast code is copied verbatim.
"""

from datetime import datetime
from logging import getLogger
from typing import Literal

import pandas as pd
import xarray as xr
from pydantic import validate_call

from jua._api import QueryEngineAPI
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua.client import JuaClient
from jua.reanalysis._types.query_payload_types import (
    ReanalysisQueryPayload,
    build_time_arg,
)
from jua.reanalysis._types.query_response_types import (
    ReanalysisMetaResult,
    ReanalysisModelInfo,
)
from jua.reanalysis.models import ReanalysisModels
from jua.types.geo import LatLon, SpatialSelection, validate_unique_point_keys
from jua.weather._stream import process_arrow_streaming_response
from jua.weather._types.query_payload_types import build_geo_filter
from jua.weather.variables import Variables

logger = getLogger(__name__)

# Mirrors the server's max_rows_returned_reanalysis_arrow. Checked client-side so
# an obviously oversized query fails immediately instead of after a round trip.
_MAX_ROWS_PER_REQUEST = 5_000_000


class ReanalysisQueryEngine:
    """Internal API client for Jua's reanalysis data.

    Note:
        Intended for internal use. End users should go through
        :class:`jua.reanalysis.Reanalysis` (``client.reanalysis``).
    """

    _DATA_ENDPOINT = "reanalysis/data"
    _META_ENDPOINT = "reanalysis/meta"
    # /reanalysis/latest-timestamp and /reanalysis/available-timestamps are
    # deliberately not wrapped: as of 2026-08-20 the former returns HTTP 500 and
    # the latter an empty list, in both dev and prod, despite prod holding
    # 320,976 arco_era5 rows in model_step_status. Wrap them once fixed.

    def __init__(self, jua_client: JuaClient):
        self._api = QueryEngineAPI(jua_client)
        self._jua_client = jua_client

    def get_meta(self) -> ReanalysisMetaResult:
        """Get metadata for every available reanalysis dataset."""
        response = self._api.get(self._META_ENDPOINT)
        return ReanalysisMetaResult(**response.json())

    def get_model_meta(self, model: ReanalysisModels) -> ReanalysisModelInfo:
        """Get metadata for one reanalysis dataset."""
        meta = self.get_meta()
        for model_info in meta.models:
            if model_info.name == model.value:
                return model_info
        available = ", ".join(m.name for m in meta.models)
        raise ValueError(
            f"Model {model.value} is not available. Available models: {available}"
        )

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_data(
        self,
        model: ReanalysisModels,
        # See Reanalysis.get_data for why `str` is accepted here.
        time: datetime | list[datetime] | slice | str,
        variables: list[Variables] | list[str] | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | LatLon | None = None,
        method: Literal["nearest", "bilinear"] = "nearest",
        stream: bool | None = None,
        print_progress: bool | None = None,
    ) -> xr.Dataset:
        """Query reanalysis data and return it as an xarray Dataset.

        See :meth:`jua.reanalysis.Reanalysis.get_data` for the argument docs.
        """
        if isinstance(points, LatLon):
            points = [points]
        if points is not None:
            validate_unique_point_keys(points)

        geo = build_geo_filter(latitude, longitude, points, method)

        df = self.load_raw_data(
            payload=ReanalysisQueryPayload(
                models=[model],
                geo=geo,
                time=build_time_arg(time),
                variables=variables,
            ),
            # Grid slices are large; points are not. Same default as forecasts.
            stream=geo.type != "point" if stream is None else stream,
            print_progress=print_progress,
        )

        if geo.type == "point":
            if not isinstance(points, list):
                points = [LatLon(lat=lat, lon=lon) for lat, lon in geo.value]  # type: ignore

        return self.transform_dataframe(df, points=points)  # type: ignore[arg-type]

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def load_raw_data(
        self,
        payload: ReanalysisQueryPayload,
        stream: bool = False,
        print_progress: bool | None = None,
    ) -> pd.DataFrame:
        """POST the query and return the Arrow response as a DataFrame.

        DIFFERS (1) from the forecast path: no ``group_by`` is set. The forecast
        payload sets it explicitly to make the API emit a ``point`` column, but
        the reanalysis API rejects ``group_by`` without ``aggregation`` and emits
        ``point`` on its own for nearest-neighbour point queries.
        """
        estimated_rows = payload.num_requested_rows()
        if estimated_rows > _MAX_ROWS_PER_REQUEST:
            raise ValueError(
                "The requested data volume is too large for a single call. "
                f"Estimated size is {estimated_rows:,} rows, which exceeds the "
                f"limit of {_MAX_ROWS_PER_REQUEST:,}. Rows equal "
                "number_of_points x number_of_timesteps. Please split your "
                "request into smaller chunks (e.g. a shorter time range or a "
                "smaller area)."
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
            self._DATA_ENDPOINT,
            data=data,
            query_params=query_params,
            extra_headers={"Accept": "*/*", "Accept-Encoding": "identity"},
            stream=stream,
        )
        df = process_arrow_streaming_response(response, print_progress and stream)
        if df.empty:
            raise ValueError("No data available for the given parameters.")

        # DIFFERS (2): the reanalysis endpoint returns `time` as
        # timestamp[ms, tz=UTC] — timezone-AWARE, where the forecast endpoint's
        # init_time is naive. A tz-aware index makes xr.Dataset.from_dataframe
        # produce an object-dtype coordinate, so drop the tz here. The SDK's
        # datetime contract is naive UTC at millisecond resolution.
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column].dtype):
                df[column] = df[column].dt.as_unit("ms")

        # DIFFERS (3): there is no `model` column to drop. The forecast path
        # asserts exactly one model and drops the column; doing that here raises
        # KeyError. Dropped defensively in case the API starts sending one.
        if "model" in df.columns:
            df = df.drop(columns=["model"])
        return df

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def transform_dataframe(
        cls,
        df: pd.DataFrame,
        points: list[LatLon] | None = None,
    ) -> xr.Dataset:
        """Transform a raw reanalysis DataFrame into an xarray Dataset.

        Mirrors :meth:`jua.weather._query_engine.QueryEngine.transform_dataframe`
        with ``time`` in place of ``init_time`` + ``prediction_timedelta``, and no
        ensemble-statistics branch (reanalysis is not an ensemble).

        Args:
            df: Must have `time`, `latitude` and `longitude` columns, plus a
                `point` column when `points` is given.
            points: The points requested, if a point query was made. When
                provided, creates a "points" dimension and keeps both the
                requested and the actual grid coordinates.

        Returns:
            An xarray Dataset.

            **Grid queries** (points is None):
                - Dimensions: `time`, `latitude`, `longitude`

            **Point queries** (points is provided):
                - Dimensions: `points`, `time`
                - Coordinates: `latitude` / `longitude` (actual grid cell) and
                  `requested_lat` / `requested_lon` (what you asked for)
        """
        # Copy up front: the forecast twin mutates its caller's frame in place,
        # which is invisible there because it is only ever handed a freshly built
        # one. This is a classmethod, so don't inherit that trap.
        df = df.copy()

        if points is not None:
            df["requested_lat"] = df["point"].apply(lambda idx: points[idx].lat)
            df["requested_lon"] = df["point"].apply(lambda idx: points[idx].lon)
            df["points"] = df["point"].apply(lambda idx: str(points[idx]))

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

            df = df.drop(
                columns=[
                    "point",
                    "latitude",
                    "longitude",
                    "requested_lat",
                    "requested_lon",
                ]
            )
            df = df.set_index(["points", "time"])
            ds = xr.Dataset.from_dataframe(df)

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
            df = df.set_index(["time", "latitude", "longitude"])
            df = df.loc[~df.index.duplicated()]
            ds = xr.Dataset.from_dataframe(df)

        for var in ds.data_vars:
            ds[var] = ds[var].astype("float32")

        ds.time.encoding = {
            "dtype": "int64",
            "units": "milliseconds since 1970-01-01T00:00:00",
        }
        return ds
