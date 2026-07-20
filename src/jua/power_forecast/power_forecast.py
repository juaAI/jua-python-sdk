"""Power forecast module for the Jua SDK."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import xarray as xr

from jua._api import QueryEngineAPI
from jua._utils.remove_none_from_dict import remove_none_from_dict

if TYPE_CHECKING:
    from jua.client import JuaClient


InitTimeSpec = str | int | datetime | list[str | int | datetime]

# Serving alias or concrete run id from GET /power-forecast/versions.
VersionSpec = str
# Per-(zone, psr) override: {"zone_key": "DE", "psr_type": "Solar", "version": "..."}.
VersionPinSpec = Mapping[str, str]

_LATEST_RE = re.compile(r"^latest(-(\d+))?$")
_INTERNAL_CHANNEL_NAMES = frozenset({"live", "preview"})


@dataclass(frozen=True)
class InitTimeInfo:
    """Information about an available power forecast init time.

    Attributes:
        init_time: The initialization time of the forecast run.
        max_prediction_timedelta: Maximum forecast horizon in minutes.
    """

    init_time: datetime
    max_prediction_timedelta: int


@dataclass(frozen=True)
class VersionInfo:
    """One model version available for a (zone_key, psr_type) cell.

    Attributes:
        model_version: Concrete run id that can be passed as ``version``.
        zone_key: Market zone code (e.g. ``"DE"``).
        psr_type: Production source type (e.g. ``"Solar"``).
        is_stable: True when this run id is the packaged ``stable`` pointer.
        is_latest: True when this run id is the packaged ``latest`` pointer.
        earliest_init_time: Earliest init time with predictions for this version.
        latest_init_time: Latest init time with predictions for this version.
    """

    model_version: str
    zone_key: str
    psr_type: str
    is_stable: bool
    is_latest: bool
    earliest_init_time: datetime
    latest_init_time: datetime


class PowerForecast:
    """Interface for Jua's power forecast services.

    Provides access to renewable energy generation forecasts in MW,
    covering generation types such as Solar, Wind Onshore, and Wind Offshore
    across supported market zones.

    Examples:
        >>> from datetime import datetime
        >>> from jua import JuaClient
        >>>
        >>> client = JuaClient()
        >>> pf = client.power_forecast
        >>>
        >>> # List available zones and PSR types
        >>> zones = pf.get_zones()
        >>> psr_types = pf.get_psr_types(zone_key="DE")
        >>>
        >>> # Horizon mode: latest forecast for German solar (stable serving)
        >>> ds = pf.get_data(
        ...     zone_keys=["DE"],
        ...     psr_types=["Solar"],
        ...     init_time="latest",
        ...     max_prediction_timedelta=2880,
        ... )
        >>>
        >>> # Same query against the packaged latest (preview) pointers
        >>> ds = pf.get_data(
        ...     zone_keys=["DE"],
        ...     psr_types=["Solar"],
        ...     init_time="latest",
        ...     version="latest",
        ... )
        >>>
        >>> # Time range mode
        >>> ds = pf.get_data(
        ...     zone_keys=["DE", "FR"],
        ...     start_time=datetime(2025, 12, 1),
        ...     end_time=datetime(2025, 12, 3),
        ... )
    """

    def __init__(self, client: JuaClient) -> None:
        self._client = client
        self._api = QueryEngineAPI(jua_client=self._client)
        # Per-zone PSR-type cache, used to validate requests and produce
        # actionable errors without repeating the (cheap, unauthenticated)
        # psr-types lookup on every call.
        self._psr_types_cache: dict[str, list[str]] = {}

    def get_zones(self) -> list[str]:
        """Get the list of available power forecast zones.

        Returns:
            List of zone codes (e.g. ``["DE", "FR"]``).

        Raises:
            RuntimeError: If the API request fails.

        Examples:
            >>> zones = client.power_forecast.get_zones()
            >>> print(zones)  # ["DE", "FR"]
        """
        try:
            response = self._api.get("power-forecast/zones", requires_auth=False)
            data = response.json()
            return data["zones"]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch power forecast zones: {e}") from e

    def get_psr_types(self, zone_key: str | None = None) -> list[str]:
        """Get available PSR (Production Source) types.

        Args:
            zone_key: Optional zone code to filter PSR types by.

        Returns:
            List of PSR type names (e.g.
            ``["Solar", "Wind Onshore", "Wind Offshore"]``).

        Raises:
            RuntimeError: If the API request fails.

        Examples:
            >>> psr_types = client.power_forecast.get_psr_types()
            >>> de_types = client.power_forecast.get_psr_types(zone_key="DE")
        """
        params: dict = {}
        if zone_key is not None:
            params["zone_key"] = zone_key

        try:
            response = self._api.get(
                "power-forecast/psr-types",
                params=params or None,
                requires_auth=False,
            )
            data = response.json()
            return data["psr_types"]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch power forecast PSR types: {e}") from e

    def _available_psr_types(self, zone_key: str) -> list[str]:
        """Return (and cache) the PSR types served for a zone."""
        if zone_key not in self._psr_types_cache:
            self._psr_types_cache[zone_key] = self.get_psr_types(zone_key=zone_key)
        return self._psr_types_cache[zone_key]

    def _validate_psr_types(self, zone_keys: list[str], psr_types: list[str]) -> None:
        """Validate requested PSR types per zone, with actionable hints.

        ``power_forecast`` serves Jua-model generation by PSR type (and load
        for the zones that have a fitted demand model). When a requested type
        isn't served for a zone, raise a clear error that lists what *is*
        available and, for demand (``"Load"``), points to the complementary
        Jua product: ``market_aggregates`` exposes predicted demand as
        ``load_mw`` (population weighting) for many more zones.

        Raises:
            ValueError: If any requested PSR type is not available for a zone.
        """
        for zone in zone_keys:
            try:
                available = self._available_psr_types(zone)
            except RuntimeError:
                # Don't block on a metadata lookup failure; let the data call
                # surface the underlying error instead.
                continue
            missing = [p for p in psr_types if p not in available]
            if not missing:
                continue
            message = (
                f"PSR type(s) {missing} not available from power_forecast for "
                f"zone '{zone}'. Available: {available}."
            )
            if "Load" in missing:
                message += (
                    " For predicted demand, market_aggregates exposes load_mw "
                    "via the population weighting: "
                    f"client.market_aggregates.get_market('{zone}')"
                    ".compare_runs_mw(weighting='population')."
                )
            raise ValueError(message)

    def get_init_times(
        self,
        zone_key: str | list[str] | None = None,
        psr_type: str | list[str] | None = None,
        limit: int = 96,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        version: VersionSpec | None = None,
    ) -> list[InitTimeInfo]:
        """Get available forecast init times.

        Two selection modes are supported:

        **Count mode** (default): returns the most recent ``limit`` init times
        (the server caps ``limit`` at 1000).

        **Time-window mode** (``start_time`` and/or ``end_time`` given): returns
        *all* init times whose ``init_time`` falls in ``[start_time, end_time)``,
        regardless of count. ``limit`` is ignored in this mode, so windows
        containing more than 1000 runs are returned in full.

        Args:
            zone_key: Optional zone code(s) to filter by. When multiple
                are given, only init times available for all of them are
                returned (intersection semantics).
            psr_type: Optional PSR type(s) to filter by. When multiple are
                given, only init times available for all of them are returned.
            limit: Maximum number of init times to return in count mode
                (default 96). Ignored when ``start_time``/``end_time`` is set.
            start_time: Inclusive lower bound on ``init_time``. Enables
                time-window mode.
            end_time: Exclusive upper bound on ``init_time``. Enables
                time-window mode.
            version: Serving selection: ``"stable"`` (default when omitted),
                ``"latest"``, or a concrete run id from :meth:`get_versions`.
                Filters init times to that model version.

        Returns:
            List of :class:`InitTimeInfo` objects sorted newest-first.

        Raises:
            ValueError: If ``version`` is an internal channel name
                (``live`` / ``preview``).
            RuntimeError: If the API request fails.

        Examples:
            >>> # Count mode: 10 most recent runs (stable serving)
            >>> init_times = client.power_forecast.get_init_times(
            ...     zone_key="DE", limit=10
            ... )
            >>>
            >>> # Same listing against the latest/preview pointers
            >>> init_times = client.power_forecast.get_init_times(
            ...     zone_key="DE", limit=10, version="latest"
            ... )
            >>>
            >>> # Time-window mode: every run in a date range (can exceed 1000)
            >>> from datetime import datetime, timezone
            >>> init_times = client.power_forecast.get_init_times(
            ...     zone_key="DE",
            ...     start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ...     end_time=datetime(2025, 2, 1, tzinfo=timezone.utc),
            ... )
        """
        self._validate_version(version)
        params: dict = {}
        # Time-window mode: filter by init_time range and let the server return
        # the full window. We omit ``limit`` because the endpoint caps it at
        # 1000, which would silently truncate long ranges.
        if start_time is not None or end_time is not None:
            if start_time is not None:
                params["start_time"] = start_time.isoformat()
            if end_time is not None:
                params["end_time"] = end_time.isoformat()
        else:
            params["limit"] = limit

        if zone_key is not None:
            params["zone_key"] = zone_key
        if psr_type is not None:
            if isinstance(psr_type, str):
                psr_type = [psr_type]
            params["psr_type"] = psr_type
        if version is not None:
            params["version"] = version

        try:
            response = self._api.get(
                "power-forecast/init-times",
                params=params,
                requires_auth=False,
            )
            data = response.json()
            return [
                InitTimeInfo(
                    init_time=datetime.fromisoformat(
                        item["init_time"].replace("Z", "+00:00")
                    ),
                    max_prediction_timedelta=item["max_prediction_timedelta"],
                )
                for item in data["init_times"]
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch power forecast init times: {e}") from e

    def get_versions(
        self,
        zone_key: str | list[str] | None = None,
        psr_type: str | list[str] | None = None,
    ) -> list[VersionInfo]:
        """List pin-able model versions for power forecasts.

        Returns the catalog of concrete run ids per (zone, PSR) cell, with
        ``is_stable`` / ``is_latest`` flags matching the packaged serving
        pointers (``serving.yaml`` on the query-engine). Use a run id from
        this catalog as ``version`` (or in ``version_pins``) to freeze a
        checkpoint; aliases ``stable`` / ``latest`` follow live pointer updates.

        Args:
            zone_key: Optional zone code(s) to filter by.
            psr_type: Optional PSR type(s) to filter by.

        Returns:
            List of :class:`VersionInfo` rows.

        Raises:
            RuntimeError: If the API request fails.

        Examples:
            >>> versions = client.power_forecast.get_versions(
            ...     zone_key="DE", psr_type="Solar"
            ... )
            >>> stable = next(v for v in versions if v.is_stable)
            >>> ds = client.power_forecast.get_data(
            ...     zone_keys=["DE"],
            ...     psr_types=["Solar"],
            ...     init_time="latest",
            ...     version=stable.model_version,
            ... )
        """
        params: dict = {}
        if zone_key is not None:
            params["zone_key"] = zone_key
        if psr_type is not None:
            if isinstance(psr_type, str):
                psr_type = [psr_type]
            params["psr_type"] = psr_type

        try:
            response = self._api.get(
                "power-forecast/versions",
                params=params or None,
                requires_auth=True,
            )
            data = response.json()
            return [
                VersionInfo(
                    model_version=item["model_version"],
                    zone_key=item["zone_key"],
                    psr_type=item["psr_type"],
                    is_stable=bool(item.get("is_stable", False)),
                    is_latest=bool(item.get("is_latest", False)),
                    earliest_init_time=datetime.fromisoformat(
                        item["earliest_init_time"].replace("Z", "+00:00")
                    ),
                    latest_init_time=datetime.fromisoformat(
                        item["latest_init_time"].replace("Z", "+00:00")
                    ),
                )
                for item in data["versions"]
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch power forecast versions: {e}") from e

    def get_data(
        self,
        zone_keys: list[str] | None = None,
        psr_types: list[str] | None = None,
        *,
        init_time: InitTimeSpec | None = None,
        max_prediction_timedelta: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_zone: str | None = None,
        version: VersionSpec | None = None,
        version_pins: Sequence[VersionPinSpec] | None = None,
    ) -> xr.Dataset:
        """Query power forecast data in MW.

        Supports two mutually exclusive query modes:

        **Horizon mode** (init_time-centric):
            Specify ``init_time`` as datetime(s), integer offsets, or
            relative tokens (``"latest"``, ``"latest-N"``). Optionally
            limit the forecast horizon with ``max_prediction_timedelta``.

            Relative tokens and integer offsets are resolved by querying
            available init times filtered by ``zone_keys``,
            ``psr_types``, and ``version``, so ``"latest"`` always refers
            to the most recent run where *all* requested zone/PSR-type
            combinations have data for that serving selection.

        **Time range mode** (time-centric):
            Specify ``start_time`` and/or ``end_time`` to filter by the
            computed forecast valid time.

        Args:
            zone_keys: Zone codes to query (e.g. ``["DE", "FR"]``).
            psr_types: PSR types to query
                (e.g. ``["Solar", "Wind Onshore"]``). If ``None``, returns
                all available types.
            init_time: Init time selection for horizon mode. Accepts
                ``"latest"``, ``"latest-N"``, a ``datetime``, an integer
                offset (0 = latest), or a list of these.
            max_prediction_timedelta: Maximum prediction horizon in
                minutes (horizon mode only).
            start_time: Start of time range, inclusive (time range mode).
            end_time: End of time range, exclusive (time range mode).
            time_zone: IANA time zone name for time formatting
                (e.g. ``"Europe/Berlin"``).
            version: Default model version for all (zone, psr) cells:
                ``"stable"`` (API default when omitted), ``"latest"``, or a
                concrete run id from :meth:`get_versions`. Overridden per
                cell by ``version_pins``.
            version_pins: Optional per-(zone_key, psr_type) overrides. Each
                mapping must include ``zone_key``, ``psr_type``, and
                ``version`` (alias or run id).

        Returns:
            ``xarray.Dataset`` with dimensions ``(zone_key, psr_type, time)``
            and data variable ``value`` in MW.

        Raises:
            ValueError: If both horizon and time-range parameters are given,
                if neither mode is specified, or if ``version`` /
                ``version_pins`` use internal channel names.
            RuntimeError: If the API request fails.

        Examples:
            >>> # Latest solar forecast for Germany (stable serving)
            >>> ds = client.power_forecast.get_data(
            ...     zone_keys=["DE"],
            ...     psr_types=["Solar"],
            ...     init_time="latest",
            ... )
            >>>
            >>> # Preview / latest serving pointers
            >>> ds = client.power_forecast.get_data(
            ...     zone_keys=["DE"],
            ...     psr_types=["Solar"],
            ...     init_time="latest",
            ...     version="latest",
            ... )
            >>>
            >>> # Freeze today's stable run id (promote-safe)
            >>> versions = client.power_forecast.get_versions(
            ...     zone_key="DE", psr_type="Solar"
            ... )
            >>> stable = next(v for v in versions if v.is_stable)
            >>> ds = client.power_forecast.get_data(
            ...     zone_keys=["DE"],
            ...     psr_types=["Solar"],
            ...     init_time="latest",
            ...     version=stable.model_version,
            ... )
            >>>
            >>> # Mix aliases per cell
            >>> ds = client.power_forecast.get_data(
            ...     zone_keys=["DE"],
            ...     psr_types=["Solar", "Load"],
            ...     init_time="latest",
            ...     version="stable",
            ...     version_pins=[
            ...         {"zone_key": "DE", "psr_type": "Solar", "version": "latest"},
            ...     ],
            ... )
            >>>
            >>> # Time range query
            >>> from datetime import datetime
            >>> ds = client.power_forecast.get_data(
            ...     zone_keys=["DE"],
            ...     start_time=datetime(2025, 12, 1),
            ...     end_time=datetime(2025, 12, 3),
            ... )
        """
        has_horizon = init_time is not None or max_prediction_timedelta is not None
        has_time_range = start_time is not None or end_time is not None

        if has_horizon and has_time_range:
            raise ValueError(
                "Cannot mix horizon mode (init_time/max_prediction_timedelta) "
                "with time range mode (start_time/end_time). Choose one."
            )
        if not has_horizon and not has_time_range:
            raise ValueError(
                "Must specify either horizon mode (init_time) "
                "or time range mode (start_time/end_time)."
            )

        self._validate_version(version)
        normalized_pins = self._normalize_version_pins(version_pins)

        if psr_types is not None and zone_keys:
            self._validate_psr_types(zone_keys, psr_types)

        resolved_init_time = init_time
        if init_time is not None:
            resolved_init_time = self._resolve_init_time(
                init_time,
                zone_keys,
                psr_types,
                version=version,
            )

        body = self._build_query_body(
            zone_keys=zone_keys,
            psr_types=psr_types,
            init_time=resolved_init_time,
            max_prediction_timedelta=max_prediction_timedelta,
            start_time=start_time,
            end_time=end_time,
            time_zone=time_zone,
            version=version,
            version_pins=normalized_pins,
        )

        try:
            response = self._api.post(
                "power-forecast/data",
                data=body,
                requires_auth=True,
            )
            data = response.json()
            return self._to_dataset(data, time_zone=time_zone)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch power forecast data: {e}") from e

    # ------------------------------------------------------------------
    # Stitched day-ahead time series
    # ------------------------------------------------------------------
    def get_day_ahead_timeseries(
        self,
        *,
        zone_keys: list[str],
        psr_types: list[str] | None = None,
        init_hour: int,
        init_minute: int = 0,
        time_zone: str = "UTC",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_init_times: int = 365,
        version: VersionSpec | None = None,
        version_pins: Sequence[VersionPinSpec] | None = None,
    ) -> xr.Dataset:
        """Return a continuous day-ahead time series stitched across runs.

        This helper selects forecast runs at a local time of day (interpreted in
        ``time_zone``), takes from each run the day-ahead window, and
        concatenates the results into a single continuous ``time`` axis.

        The day-ahead window is defined by the forecast lead range:
        ``[time until local midnight, +24 hours)`` from the init time.
        For example, for ``init_hour = 9`` the selected window is
        ``[15h, 39h]`` from each init (i.e., 00:00..23:00 of D when the run is
        at D-1 09:00). For an init time at 13:45, it is
        ``[10h15m, 34h15m]``.

        Two selection modes are supported:

        **Date-range mode** (``start_date`` and/or ``end_date`` given):
            The daily runs at the selected time of day spanning the range are
            constructed directly and fetched in a single request. This bypasses
            the init-times listing limit, so arbitrarily long histories (e.g. a
            full year) can be stitched. ``start_date``/``end_date`` bound the
            resulting *valid time* axis. Matching is exact; a missing daily run
            raises ``ValueError`` rather than returning a partial series.

        **Latest mode** (no dates):
            The most recent matching runs are discovered via the init-times
            endpoint (bounded by ``max_init_times``).

        Args:
            zone_keys: Zone codes to query (e.g. ``["DE"]``).
            psr_types: Optional PSR types to include
                (e.g. ``["Solar"]``). If ``None``, returns all available types.
            init_hour: Local hour (0..23) of the runs to stitch.
            init_minute: Local minute (0..59) of the runs to stitch
                (default ``0``).
            time_zone: IANA time zone used to interpret the selected run time
                (default ``"UTC"``).
            start_date: Inclusive lower bound on the valid-time axis. Enables
                date-range mode. Naive datetimes are interpreted in
                ``time_zone``.
            end_date: Exclusive upper bound on the valid-time axis. Enables
                date-range mode. Naive datetimes are interpreted in
                ``time_zone``.
            max_init_times: Upper bound on how many matching init times are
                requested from the server in latest mode (controls history
                depth).
            version: Serving selection forwarded to :meth:`get_data` /
                :meth:`get_init_times` (``stable`` / ``latest`` / run id).
            version_pins: Optional per-(zone, psr) overrides forwarded to
                :meth:`get_data`.

        Returns:
            ``xarray.Dataset`` with dims ``(zone_key, psr_type, time)`` and
            variable ``value`` (MW). The series is continuous across days.
        """
        if not zone_keys or not isinstance(zone_keys, list):
            raise ValueError("zone_keys must be a non-empty list of zone codes")
        if not isinstance(init_hour, int) or not 0 <= init_hour <= 23:
            raise ValueError("init_hour must be in the range 0..23")
        if not isinstance(init_minute, int) or not 0 <= init_minute <= 59:
            raise ValueError("init_minute must be in the range 0..59")

        tz = ZoneInfo(time_zone)

        # Compute lead range in minutes for the day-ahead slice
        init_minutes = init_hour * 60 + init_minute
        start_lead_minutes = (24 * 60 - init_minutes) % (24 * 60)
        end_lead_minutes = start_lead_minutes + 24 * 60

        if start_date is not None or end_date is not None:
            df = self._fetch_day_ahead_by_date_range(
                zone_keys=zone_keys,
                psr_types=psr_types,
                init_hour=init_hour,
                init_minute=init_minute,
                tz=tz,
                time_zone=time_zone,
                start_date=start_date,
                end_date=end_date,
                end_lead_minutes=end_lead_minutes,
                version=version,
                version_pins=version_pins,
            )
        else:
            df = self._fetch_day_ahead_latest(
                zone_keys=zone_keys,
                psr_types=psr_types,
                init_hour=init_hour,
                init_minute=init_minute,
                tz=tz,
                time_zone=time_zone,
                end_lead_minutes=end_lead_minutes,
                max_init_times=max_init_times,
                version=version,
                version_pins=version_pins,
            )

        return self._stitch_day_ahead(
            df,
            start_lead_minutes=start_lead_minutes,
            end_lead_minutes=end_lead_minutes,
            tz=tz,
            start_date=start_date,
            end_date=end_date,
        )

    def _fetch_day_ahead_latest(
        self,
        *,
        zone_keys: list[str],
        psr_types: list[str] | None,
        init_hour: int,
        init_minute: int,
        tz: ZoneInfo,
        time_zone: str,
        end_lead_minutes: int,
        max_init_times: int,
        version: VersionSpec | None = None,
        version_pins: Sequence[VersionPinSpec] | None = None,
    ) -> pd.DataFrame:
        """Fetch day-ahead data for the most recent matching runs."""
        init_infos = self.get_init_times(
            zone_key=zone_keys,
            psr_type=psr_types,
            limit=max_init_times,
            version=version,
        )
        matching_inits: list[str | int | datetime] = []
        for info in init_infos:
            it = info.init_time
            local_init = (
                it if it.tzinfo else it.replace(tzinfo=ZoneInfo("UTC"))
            ).astimezone(tz)
            if (local_init.hour, local_init.minute) == (init_hour, init_minute):
                matching_inits.append(it)

        if not matching_inits:
            raise ValueError(
                "No power forecast runs found at "
                f"{init_hour:02d}:{init_minute:02d} in {time_zone}. "
                "Use get_init_times() to inspect available runs."
            )

        ds = self.get_data(
            zone_keys=zone_keys,
            psr_types=psr_types,
            init_time=matching_inits,
            max_prediction_timedelta=end_lead_minutes,
            time_zone=time_zone,
            version=version,
            version_pins=version_pins,
        )
        if "value" not in ds:
            return pd.DataFrame()
        return ds.to_dataframe().reset_index()

    def _fetch_day_ahead_by_date_range(
        self,
        *,
        zone_keys: list[str],
        psr_types: list[str] | None,
        init_hour: int,
        init_minute: int,
        tz: ZoneInfo,
        time_zone: str,
        start_date: datetime | None,
        end_date: datetime | None,
        end_lead_minutes: int,
        version: VersionSpec | None = None,
        version_pins: Sequence[VersionPinSpec] | None = None,
    ) -> pd.DataFrame:
        """Fetch day-ahead data by constructing daily init runs over a range.

        The day-ahead run for valid day ``D`` is issued on ``D - 1`` at
        ``init_hour`` (except a midnight run, which covers its own day). We
        build exactly the runs whose windows intersect the requested range and
        fetch them in a single request.
        """
        init_times = self._build_day_ahead_inits(
            init_hour=init_hour,
            init_minute=init_minute,
            tz=tz,
            start_date=start_date,
            end_date=end_date,
        )
        if not init_times:
            return pd.DataFrame()

        ds = self.get_data(
            zone_keys=zone_keys,
            psr_types=psr_types,
            init_time=init_times,
            max_prediction_timedelta=end_lead_minutes,
            time_zone=time_zone,
            version=version,
            version_pins=version_pins,
        )
        if "value" not in ds:
            raise ValueError(
                "No power forecast runs found at "
                f"{init_hour:02d}:{init_minute:02d} in {time_zone} for the "
                "requested date range. Use get_init_times() to inspect "
                "available runs."
            )

        df = ds.to_dataframe().reset_index()
        populated = df.dropna(subset=["value"])
        populated = populated.assign(
            _init_time_utc=pd.to_datetime(populated["init_time"], utc=True)
        )
        returned_inits = set(populated["_init_time_utc"])
        requested_inits = set(pd.to_datetime(init_times, utc=True))
        missing_inits = sorted(requested_inits - returned_inits)
        if missing_inits:
            missing_local = ", ".join(
                init.astimezone(tz).isoformat() for init in missing_inits[:5]
            )
            if len(missing_inits) > 5:
                missing_local += f", ... ({len(missing_inits)} total)"
            raise ValueError(
                "No power forecast run found for exact init time(s): "
                f"{missing_local}. Use get_init_times() to inspect available runs."
            )

        if {"zone_key", "psr_type"}.issubset(populated.columns):
            if psr_types is not None:
                expected_cells = {
                    (zone_key, psr_type)
                    for zone_key in zone_keys
                    for psr_type in psr_types
                }
            else:
                expected_cells = set(
                    populated[["zone_key", "psr_type"]].itertuples(
                        index=False, name=None
                    )
                )
            returned = set(
                populated[["zone_key", "psr_type", "_init_time_utc"]].itertuples(
                    index=False, name=None
                )
            )
            missing = sorted(
                (zone_key, psr_type, init)
                for zone_key, psr_type in expected_cells
                for init in requested_inits
                if (zone_key, psr_type, init) not in returned
            )
            if missing:
                details = ", ".join(
                    f"{zone_key}/{psr_type}@{init.astimezone(tz).isoformat()}"
                    for zone_key, psr_type, init in missing[:5]
                )
                if len(missing) > 5:
                    details += f", ... ({len(missing)} total)"
                raise ValueError(
                    "No power forecast run found for exact zone/PSR/init "
                    f"selection(s): {details}. Use get_init_times() to inspect "
                    "available runs."
                )
        return df

    @staticmethod
    def _build_day_ahead_inits(
        *,
        init_hour: int,
        init_minute: int,
        tz: ZoneInfo,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[str | int | datetime]:
        """Construct one ``init_hour`` init datetime per day spanning the range.

        ``start_date``/``end_date`` bound the valid-time axis. Naive bounds are
        interpreted in ``tz``. Returned datetimes are timezone-aware (UTC).
        """
        utc = ZoneInfo("UTC")

        def _localize(value: datetime) -> datetime:
            return value if value.tzinfo else value.replace(tzinfo=tz)

        if end_date is None:
            end_local = datetime.now(utc).astimezone(tz)
        else:
            end_local = _localize(end_date).astimezone(tz)

        if start_date is None:
            # Default to ~1 year of history when only an end is supplied.
            start_local = end_local - timedelta(days=365)
        else:
            start_local = _localize(start_date).astimezone(tz)

        # A non-midnight run produces the following local day's day-ahead
        # window; a midnight run produces its own day. Build only runs whose
        # window intersects [start_date, end_date), avoiding unused boundary
        # requests that could be unavailable.
        init_day_offset = timedelta(
            days=0 if init_hour == 0 and init_minute == 0 else 1
        )
        first_init_day = start_local.date() - init_day_offset
        last_valid_day = (end_local - timedelta(microseconds=1)).date()
        last_init_day = last_valid_day - init_day_offset

        inits: list[str | int | datetime] = []
        day = first_init_day
        while day <= last_init_day:
            local_init = datetime(
                day.year,
                day.month,
                day.day,
                init_hour,
                init_minute,
                tzinfo=tz,
            )
            inits.append(local_init.astimezone(utc))
            day = day + timedelta(days=1)
        return inits

    @staticmethod
    def _stitch_day_ahead(
        df: pd.DataFrame,
        *,
        start_lead_minutes: int,
        end_lead_minutes: int,
        tz: ZoneInfo,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> xr.Dataset:
        """Filter to the day-ahead window, dedupe overlaps, and build a Dataset."""
        if df.empty or "value" not in df.columns:
            return xr.Dataset(attrs={"unit": "MW"})

        df = df.dropna(subset=["value"]).copy()
        if df.empty:
            return xr.Dataset(attrs={"unit": "MW"})

        df["lead_minutes"] = (df["time"] - df["init_time"]) / pd.Timedelta(minutes=1)
        mask = (df["lead_minutes"] >= start_lead_minutes) & (
            df["lead_minutes"] < end_lead_minutes
        )
        df = df.loc[mask].drop(columns=["lead_minutes"])

        if df.empty:
            return xr.Dataset(attrs={"unit": "MW"})

        index_cols = [c for c in ["zone_key", "psr_type", "time"] if c in df.columns]

        # Several matching runs can share the same target hour (e.g. sub-hourly
        # runs like 07:00 and 07:30), so their day-ahead windows overlap and
        # produce duplicate (zone_key, psr_type, time) rows. Keep the value from
        # the most recent init for each valid time so the stitched index stays
        # unique and reflects the freshest forecast.
        sort_cols = [c for c in index_cols if c != "time"] + ["time", "init_time"]
        df = (
            df.sort_values(sort_cols)
            .drop_duplicates(subset=index_cols, keep="last")
            .drop(columns=["init_time"])
        )

        # Clip the valid-time axis to the requested bounds (date-range mode).
        if start_date is not None or end_date is not None:
            df = PowerForecast._clip_time(df, tz, start_date, end_date)
            if df.empty:
                return xr.Dataset(attrs={"unit": "MW"})

        df = df.sort_values(index_cols)
        stitched = xr.Dataset.from_dataframe(df.set_index(index_cols))
        stitched = stitched.assign_attrs(unit="MW")
        return stitched

    @staticmethod
    def _clip_time(
        df: pd.DataFrame,
        tz: ZoneInfo,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pd.DataFrame:
        """Clip ``df`` to ``[start_date, end_date)`` on the ``time`` column.

        Comparison is done in UTC to avoid tz/naive mismatches regardless of how
        the API localized the returned ``time`` values.
        """
        times = pd.DatetimeIndex(df["time"])
        if times.tz is None:
            times_utc = times.tz_localize(tz).tz_convert("UTC")
        else:
            times_utc = times.tz_convert("UTC")

        keep = pd.Series(True, index=df.index)
        if start_date is not None:
            lo = pd.Timestamp(start_date)
            lo = lo.tz_localize(tz) if lo.tzinfo is None else lo
            keep &= times_utc >= lo
        if end_date is not None:
            hi = pd.Timestamp(end_date)
            hi = hi.tz_localize(tz) if hi.tzinfo is None else hi
            keep &= times_utc < hi
        return df.loc[keep.values]

    # ------------------------------------------------------------------
    # Init-time resolution
    # ------------------------------------------------------------------

    def _resolve_init_time(
        self,
        spec: InitTimeSpec,
        zone_keys: list[str] | None,
        psr_types: list[str] | None,
        *,
        version: VersionSpec | None = None,
    ) -> InitTimeSpec:
        """Resolve relative init_time tokens against available init times.

        Queries ``GET /init-times`` filtered by the requested zones,
        PSR types, and serving ``version`` so that ``"latest"`` maps to the
        newest run where all selected combinations have completed data.
        """
        items = spec if isinstance(spec, list) else [spec]

        needs_resolution = any(self._is_relative(v) for v in items)
        if not needs_resolution:
            return spec

        max_offset = 0
        for v in items:
            offset = self._parse_offset(v)
            if offset is not None and offset > max_offset:
                max_offset = offset

        available = self._fetch_available_init_times(
            zone_keys, psr_types, limit=max_offset + 1, version=version
        )

        resolved = [self._resolve_single(v, available) for v in items]

        if not isinstance(spec, list):
            return resolved[0]
        return resolved

    def _fetch_available_init_times(
        self,
        zone_keys: list[str] | None,
        psr_types: list[str] | None,
        limit: int,
        *,
        version: VersionSpec | None = None,
    ) -> list[datetime]:
        """Get available init times filtered by zone and PSR types.

        Delegates intersection semantics to the server by passing all
        zone_keys in a single ``/init-times`` call.
        """
        infos = self.get_init_times(
            zone_key=zone_keys,
            psr_type=psr_types,
            limit=limit,
            version=version,
        )
        return [info.init_time for info in infos]

    @staticmethod
    def _is_relative(value: str | int | datetime) -> bool:
        if isinstance(value, int):
            return True
        if isinstance(value, str) and _LATEST_RE.match(value):
            return True
        return False

    @staticmethod
    def _parse_offset(value: str | int | datetime) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            m = _LATEST_RE.match(value)
            if m:
                return int(m.group(2) or 0)
        return None

    @staticmethod
    def _resolve_single(
        value: str | int | datetime,
        available: list[datetime],
    ) -> str | int | datetime:
        """Resolve a single init_time value to a datetime if relative."""
        offset = PowerForecast._parse_offset(value)
        if offset is None:
            return value

        if offset >= len(available):
            raise ValueError(
                f"Requested init_time offset {offset} but only "
                f"{len(available)} init time(s) available for the "
                f"selected zones/PSR types"
            )
        return available[offset]

    # ------------------------------------------------------------------
    # Query body
    # ------------------------------------------------------------------

    @staticmethod
    def _format_init_time(
        value: InitTimeSpec,
    ) -> str | int | list[str | int]:
        """Serialize init_time values for the API request body."""
        if isinstance(value, list):
            return [PowerForecast._format_single_init(v) for v in value]
        return PowerForecast._format_single_init(value)

    @staticmethod
    def _format_single_init(value: str | int | datetime) -> str | int:
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @staticmethod
    def _validate_version(version: VersionSpec | None) -> None:
        if version in _INTERNAL_CHANNEL_NAMES:
            raise ValueError("use 'stable' or 'latest', not internal channel names")

    @staticmethod
    def _normalize_version_pins(
        version_pins: Sequence[VersionPinSpec] | None,
    ) -> list[dict[str, str]] | None:
        if version_pins is None:
            return None

        normalized: list[dict[str, str]] = []
        for pin in version_pins:
            try:
                zone_key = pin["zone_key"]
                psr_type = pin["psr_type"]
                version = pin["version"]
            except KeyError as exc:
                raise ValueError(
                    "Each version_pins entry must include "
                    "'zone_key', 'psr_type', and 'version'"
                ) from exc
            if not isinstance(zone_key, str) or not zone_key:
                raise ValueError("version_pins.zone_key must be a non-empty string")
            if not isinstance(psr_type, str) or not psr_type:
                raise ValueError("version_pins.psr_type must be a non-empty string")
            if not isinstance(version, str) or not version:
                raise ValueError("version_pins.version must be a non-empty string")
            PowerForecast._validate_version(version)
            normalized.append(
                {
                    "zone_key": zone_key,
                    "psr_type": psr_type,
                    "version": version,
                }
            )
        return normalized

    @staticmethod
    def _build_query_body(
        zone_keys: list[str] | None,
        psr_types: list[str] | None,
        init_time: InitTimeSpec | None,
        max_prediction_timedelta: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        time_zone: str | None,
        version: VersionSpec | None = None,
        version_pins: list[dict[str, str]] | None = None,
    ) -> dict:
        body: dict = {}
        if zone_keys is not None:
            body["zone_keys"] = zone_keys
        if psr_types is not None:
            body["psr_types"] = psr_types

        if init_time is not None:
            body["init_time"] = PowerForecast._format_init_time(init_time)
        if max_prediction_timedelta is not None:
            body["max_prediction_timedelta"] = max_prediction_timedelta

        if start_time is not None:
            body["start_time"] = start_time.isoformat()
        if end_time is not None:
            body["end_time"] = end_time.isoformat()
        if time_zone is not None:
            body["time_zone"] = time_zone
        if version is not None:
            body["version"] = version
        if version_pins is not None:
            body["version_pins"] = version_pins

        return remove_none_from_dict(body)

    @staticmethod
    def _to_dataset(data: dict, time_zone: str | None = None) -> xr.Dataset:
        """Convert columnar JSON response to an xarray Dataset.

        Timestamps are parsed DST-safely: the server emits ISO-8601 strings
        whose UTC offset changes across a daylight-saving transition, so we
        always normalize through UTC first (avoiding an ``object`` column of
        mixed offsets) and then convert to ``time_zone`` when requested. This
        keeps ``time`` / ``init_time`` as proper datetime dtypes so downstream
        arithmetic (e.g. the day-ahead stitching lead computation) works for
        ranges that span a DST boundary.
        """
        if not data or all(len(v) == 0 for v in data.values()):
            return xr.Dataset(attrs={"unit": "MW"})

        df = pd.DataFrame(data)
        for column in ("time", "init_time"):
            if column not in df.columns:
                continue
            parsed = pd.to_datetime(df[column], utc=True, format="ISO8601")
            if time_zone is not None:
                parsed = parsed.dt.tz_convert(ZoneInfo(time_zone))
            df[column] = parsed

        index_cols = ["zone_key", "psr_type", "init_time", "time"]
        present_cols = [c for c in index_cols if c in df.columns]

        ds = xr.Dataset.from_dataframe(df.set_index(present_cols))
        ds = ds.assign_attrs(unit="MW")
        return ds
