import math
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel

from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather._model_meta import ModelMetaInfo, get_model_meta_info
from jua.weather.models import Models

Point = tuple[float, float]


class TimeSlice(BaseModel):
    start: str
    end: str


class GeoFilter(BaseModel):
    type: Literal["point", "bounding_box"]
    value: list[Point] | list[tuple[Point, Point]]
    method: Literal["nearest", "bilinear"] | None = None


class PredictionTimedeltaSlice(BaseModel):
    start: int
    end: int


class ForecastIndexQueryPayload(BaseModel):
    """Payload for the forecast index endpoint.

    Returns coordinate information without loading actual forecast data.
    More efficient than ForecastQueryPayload when you only need metadata.
    """

    model: Models
    latitude: tuple[float, float]
    longitude: tuple[float, float]
    init_time: str | list[str] | TimeSlice
    prediction_timedelta: int | list[int] | PredictionTimedeltaSlice | None = None
    timedelta_unit: Literal["h", "m", "d"] = "m"
    variables: list[str] | None = None


class ForecastQueryPayload(BaseModel):
    models: list[Models]
    init_time: str | list[str] | TimeSlice
    geo: GeoFilter
    prediction_timedelta: int | list[int] | PredictionTimedeltaSlice | None
    timedelta_unit: Literal["h", "m", "d"] = "m"
    aggregation: list[str] | None = None
    variables: list[str] | None = None

    def num_requested_points(self) -> int:
        """Estimate number of requested data rows for this payload.

        Calculation per model:
          points_count x init_times_count x prediction_timedelta_count

        Notes:
        - For point selection, points_count is the number of points provided.
        - For bounding boxes, points_count is estimated from model grid density.
        - Variables do not multiply rows.
        """
        if len(self.models) == 0:
            return 0

        # Helper: count points for a given model
        def count_points_for_model(meta: ModelMetaInfo) -> int:
            if self.geo.type == "point":
                return len(self.geo.value)  # list[Point]

            # Bounding box: estimate using model grid
            num_lats, num_lons = meta.num_lats, meta.num_lons
            if meta.has_both_poles:
                num_lats -= 1

            total = 0
            for latlon_a, latlon_b in self.geo.value:
                (lat1, lon1), (lat2, lon2) = latlon_a, latlon_b  # type: ignore[misc]
                lat_min, lat_max = (lat1, lat2) if lat1 <= lat2 else (lat2, lat1)
                lon_min, lon_max = (lon1, lon2) if lon1 <= lon2 else (lon2, lon1)
                if lat_max < lat_min or lon_max < lon_min:
                    continue

                min_grid_lat_idx = math.ceil(num_lats * (lat_min + 90) / 180)
                max_grid_lat_idx = math.floor(num_lats * (lat_max + 90) / 180)
                min_grid_lon_idx = math.ceil(num_lons * (lon_min + 180) / 360)
                max_grid_lon_idx = math.floor(num_lons * (lon_max + 180) / 360)

                lat_count = max_grid_lat_idx - min_grid_lat_idx + 1
                lon_count = max_grid_lon_idx - min_grid_lon_idx + 1
                total += max(0, lat_count) * max(0, lon_count)
            return total

        # Helper: count init times for a given model
        def count_init_times_for_model(meta: ModelMetaInfo) -> int:
            if isinstance(self.init_time, str):
                return 1

            if isinstance(self.init_time, list):
                return len(self.init_time)

            step_hours = 24.0 / float(meta.forecasts_per_day)
            start_dt = datetime.fromisoformat(self.init_time.start)  # type: ignore[union-attr]
            end_dt = datetime.fromisoformat(self.init_time.end)  # type: ignore[union-attr]
            total_hours = max(0.0, (end_dt - start_dt).total_seconds() / 3600.0)
            return int(math.floor(total_hours / step_hours)) + 1

        # Helper: count prediction timedeltas for a given model
        def count_prediction_timedeltas_for_model(meta: ModelMetaInfo) -> int:
            tr = meta.temporal_resolution
            if self.prediction_timedelta is None:
                horizon = meta.full_forecasted_hours or 480
                return tr.num_prediction_timedeltas(0, int(horizon))
            if isinstance(self.prediction_timedelta, int):
                return 1
            if isinstance(self.prediction_timedelta, list):
                return len(self.prediction_timedelta)

            # PredictionTimedeltaSlice in minutes; convert to hours
            start_h = int(self.prediction_timedelta.start // 60)
            end_h = int(self.prediction_timedelta.end // 60)
            return tr.num_prediction_timedeltas(start_h, end_h)

        total_rows = 0
        for model in self.models:
            model_meta = get_model_meta_info(model)
            points_count = count_points_for_model(model_meta)
            init_times_count = count_init_times_for_model(model_meta)
            pred_td_count = count_prediction_timedeltas_for_model(model_meta)
            total_rows += points_count * init_times_count * pred_td_count

        return total_rows


def build_geo_filter(
    lat: SpatialSelection | None = None,
    lon: SpatialSelection | None = None,
    points: list[LatLon] | LatLon | None = None,
    method: Literal["nearest", "bilinear"] = "nearest",
) -> GeoFilter:
    """
    Build a geographic filter for weather data queries.

    This function creates a GeoFilter object that defines the spatial selection
    for weather data requests. It supports two main input patterns:
    coordinate-based (lat/lon) and point-based selection.

    Args:
        lat: Latitude spatial selection. Can be:
            - float: Single latitude coordinate
            - slice: Latitude range (e.g., slice(40.0, 45.0) for bounding box)
            - list[float]: Multiple latitude values for multiple points
            - None: Use when specifying points parameter instead
        lon: Longitude spatial selection. Can be:
            - float: Single longitude coordinate
            - slice: Longitude range (e.g., slice(-10.0, 5.0) for bounding box)
            - list[float]: Multiple longitude values for multiple points
            - None: Use when specifying points parameter instead
        points: Geographic points using LatLon objects. Can be:
            - LatLon: Single geographic point
            - list[LatLon]: Multiple geographic points
            - None: Use when specifying lat/lon parameters instead

    Returns:
        GeoFilter: Geographic filter object with:
            - type: "point" for point-based selection or "bounding_box" for area
                selection
            - value: List of coordinate tuples formatted for the API

    Raises:
        ValueError: When:
            - Both (lat, lon) and points are provided simultaneously
            - Neither (lat, lon) nor points are provided
            - lat and lon have mismatched types (e.g., lat is float, lon is slice)
            - lat and lon lists have different lengths
            - Invalid parameter combinations are used
    """
    if lat is None and lon is None and points is not None:
        if isinstance(points, LatLon):
            return GeoFilter(
                type="point", value=[(points.lat, points.lon)], method=method
            )
        else:
            return GeoFilter(
                type="point", value=[(p.lat, p.lon) for p in points], method=method
            )
    elif lat is not None and lon is not None and points is None:
        if not (
            (isinstance(lat, slice) and isinstance(lon, slice))
            or (isinstance(lat, list) and isinstance(lon, list))
            or (isinstance(lat, (float, int)) and isinstance(lon, (float, int)))
        ):
            raise ValueError(
                "The type used for the latitude spatial selection must match the one "
                " found for the longitude spatial selection (e.g. both are slices, or "
                f"both are floats). Found lat={type(lat)}, lon={type(lon)}"
            )

        if isinstance(lat, slice):
            return GeoFilter(
                type="bounding_box",
                value=[
                    (
                        (float(lat.start), float(lon.start)),  # type: ignore
                        (float(lat.stop), float(lon.stop)),  # type: ignore
                    ),
                ],
            )

        elif isinstance(lat, list):
            if len(lat) != len(lon):  # type: ignore
                raise ValueError(
                    f"Number of latitudes ({len(lat)}) must match number of longitudes "
                    f"({len(lon)}) to form points."  # type: ignore
                )

            return GeoFilter(
                type="point",
                value=[(lat_, lon_) for lat_, lon_ in zip(lat, lon)],  # type: ignore
                method=method,
            )

        return GeoFilter(type="point", value=[(lat, lon)], method=method)

    raise ValueError(
        "Either both (lat, lon) must be given (and points=None), or points must be "
        f"given (and lat=None, lon=None). Found (lat={lat}, lon={lon}), points={points}"
    )


def build_init_time_arg(
    time: Literal["latest"] | datetime | list[datetime] | slice | None = None,
) -> str | list[str] | TimeSlice:
    """
    Build init_time argument from various time inputs.

    Args:
        time: Time specification. Can be:
            - None: Returns "latest"
            - datetime: Returns datetime in isoformat with timezone
            - list[datetime]: Returns TimeSlice from first to last datetime
            - slice: Returns TimeSlice from slice.start to slice.stop

    Returns:
        str | TimeSlice: Either "latest" string or TimeSlice object for ranges

    Raises:
        ValueError: If slice has missing start/stop or list is empty
    """
    if time is None or time == "latest":
        return "latest"

    if isinstance(time, datetime):
        # Ensure timezone is set, default to UTC if none
        if time.tzinfo is None:
            time = time.replace(tzinfo=UTC)
        return time.isoformat()

    if isinstance(time, list):
        if len(time) == 0:
            raise ValueError("Time list cannot be empty")

        return [
            t.isoformat()
            for t in map(
                lambda t: t.replace(tzinfo=UTC) if t.tzinfo is None else t,
                time,
            )
        ]

    if isinstance(time, slice):
        if time.start is None or time.stop is None:
            raise ValueError("Time slice must have both start and stop values")

        start_time = time.start
        end_time = time.stop

        # Ensure timezone is set, default to UTC if none
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)

        return TimeSlice(start=start_time.isoformat(), end=end_time.isoformat())

    raise ValueError(
        f"Unsupported time type: {type(time)}. Expected datetime, list[datetime], "
        f"slice, or None."
    )


def build_prediction_timedelta(
    prediction_timedelta: PredictionTimeDelta | None,
) -> int | list[int] | PredictionTimedeltaSlice | None:
    """
    Converts a prediction timedelta for the ForecastQueryPayload.

    Args:
        prediction_timedelta: The prediction timedelta parameter, in hours.

    Returns:
        int: The maximum prediction timedelta to query, in minutes.
        PredictionTimedeltaSlice: The (min, max) prediction timedelta to query, in
            minutes.

    Raises:
        ValueError: If prediction_timedelta cannot be parsed.
    """
    if prediction_timedelta is None:
        return None

    def to_minutes(v: int | np.timedelta64) -> int:
        if isinstance(v, int):
            return 60 * v
        if isinstance(v, np.timedelta64):
            return int(v / np.timedelta64(1, "m"))
        raise ValueError(f"Unknown type to convert to minutes {type(v)}: {v}")

    if isinstance(prediction_timedelta, (int, np.timedelta64)):
        if prediction_timedelta == 0:  # Temporary fix for the query engine
            return [0]

        return to_minutes(prediction_timedelta)

    if isinstance(prediction_timedelta, slice):
        start = (
            to_minutes(prediction_timedelta.start)
            if prediction_timedelta.start is not None
            else 0
        )
        stop = (
            to_minutes(prediction_timedelta.stop)
            if prediction_timedelta.stop is not None
            else None
        )
        if stop is not None and stop < start:
            raise ValueError(
                "prediction_timedelta slice stop must be greater than or equal to start"
            )

        return PredictionTimedeltaSlice(start=start, end=stop)

    if isinstance(prediction_timedelta, list):
        if len(prediction_timedelta) == 0:
            raise ValueError("prediction_timedelta list cannot be empty")

        return [to_minutes(v) for v in prediction_timedelta]

    raise ValueError(
        f"Unsupported prediction_timedelta type {type(prediction_timedelta)}: "
        f"{prediction_timedelta}"
    )
