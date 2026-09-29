import math
from datetime import datetime

from pydantic import BaseModel

from jua.reanalysis.models import ReanalysisModels
from jua.weather._types.query_payload_types import (
    GeoFilter,
    TimeSlice,
    build_init_time_arg,
)

# ERA5 is a 0.25 degree global grid, hourly. Both are fixed properties of the
# dataset, so unlike the forecast payload there is no per-model metadata to look
# up.
_GRID_STEP_DEGREES = 0.25
_TEMPORAL_RESOLUTION_HOURS = 1


class ReanalysisQueryPayload(BaseModel):
    """Request body for ``POST /v1/reanalysis/data``.

    Deliberately narrower than the API accepts. The endpoint also supports
    ``group_by`` / ``aggregation`` / ``weighting`` / ``pagination``, but
    ``group_by`` requires ``aggregation``, and an aggregated response has a
    different column set that would not round-trip into the gridded xarray shape
    this SDK returns. Those knobs stay unexposed until there is a use for them.
    """

    models: list[ReanalysisModels]
    geo: GeoFilter
    time: str | list[str] | TimeSlice
    variables: list[str] | None = None

    def num_requested_rows(self) -> int:
        """Estimate the rows this query would return.

        rows = points x timesteps. Variables are columns, so they do not
        multiply rows. Used to fail fast before spending a request on something
        the server will reject.
        """
        return self._count_points() * self._count_timesteps()

    def _count_points(self) -> int:
        if self.geo.type == "point":
            return len(self.geo.value)

        total = 0
        for corner_a, corner_b in self.geo.value:
            (lat1, lon1), (lat2, lon2) = corner_a, corner_b  # type: ignore[misc]
            lat_count = _grid_count(min(lat1, lat2), max(lat1, lat2))
            lon_count = _grid_count(min(lon1, lon2), max(lon1, lon2))
            total += lat_count * lon_count
        return total

    def _count_timesteps(self) -> int:
        if isinstance(self.time, str):
            return 1
        if isinstance(self.time, list):
            return len(self.time)

        start = datetime.fromisoformat(self.time.start)
        end = datetime.fromisoformat(self.time.end)
        span_hours = max(0.0, (end - start).total_seconds() / 3600.0)
        return int(span_hours // _TEMPORAL_RESOLUTION_HOURS) + 1


def _grid_count(low: float, high: float) -> int:
    """Number of 0.25-degree grid lines in the closed interval [low, high]."""
    first = math.ceil(low / _GRID_STEP_DEGREES)
    last = math.floor(high / _GRID_STEP_DEGREES)
    return max(0, last - first + 1)


def build_time_arg(
    time: datetime | list[datetime] | slice | str,
) -> str | list[str] | TimeSlice:
    """Convert a user-facing time selection into the API's ``time`` field.

    The reanalysis endpoint accepts the same shapes as the forecast endpoint's
    ``init_time`` (they share a pydantic type server-side), so this delegates to
    the forecast builder. It adds two reanalysis-specific rejections:

    - ``None``, because there is no meaningful default timestamp for a reanalysis
      query the way "latest init" is for a forecast.
    - ``"latest"``, because the server's resolution of it currently fails. The
      API documents the value and the type accepts it, but
      ``POST /v1/reanalysis/data`` with ``time="latest"`` returns HTTP 500 in
      both dev and prod, as does ``GET /v1/reanalysis/latest-timestamp``.
      Rejecting it here turns an opaque 500 into an actionable message. Remove
      this branch once the backend is fixed; accepting it later is not a
      breaking change.
    """
    if time is None:
        raise ValueError(
            "`time` is required for reanalysis queries. Pass a datetime, a list "
            "of datetimes, or a slice(start, stop)."
        )
    if isinstance(time, str):
        raise ValueError(
            f'time={time!r} is not supported. Reanalysis has no "latest" '
            "selector: the server-side resolution of it currently returns HTTP "
            "500. Pass an explicit datetime, a list of datetimes, or a "
            "slice(start, stop). Note that ERA5 is published with a lag of "
            "roughly five days, so recent timestamps will not exist yet."
        )
    return build_init_time_arg(time)
