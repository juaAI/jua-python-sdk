"""Power forecast module for the Jua SDK."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from zoneinfo import ZoneInfo

from jua._api import QueryEngineAPI
from jua._utils.remove_none_from_dict import remove_none_from_dict

if TYPE_CHECKING:
    from jua.client import JuaClient


InitTimeSpec = str | int | datetime | list[str | int | datetime]

_LATEST_RE = re.compile(r"^latest(-(\d+))?$")


@dataclass(frozen=True)
class InitTimeInfo:
    """Information about an available power forecast init time.

    Attributes:
        init_time: The initialization time of the forecast run.
        max_prediction_timedelta: Maximum forecast horizon in minutes.
    """

    init_time: datetime
    max_prediction_timedelta: int


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
        >>> # Horizon mode: latest forecast for German solar
        >>> ds = pf.get_data(
        ...     zone_keys=["DE"],
        ...     psr_types=["Solar"],
        ...     init_time="latest",
        ...     max_prediction_timedelta=2880,
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

    def get_init_times(
        self,
        zone_key: str | list[str] | None = None,
        psr_type: str | list[str] | None = None,
        limit: int = 96,
    ) -> list[InitTimeInfo]:
        """Get available forecast init times.

        Args:
            zone_key: Optional zone code(s) to filter by. When multiple
                are given, only init times available for all of them are
                returned (intersection semantics).
            psr_type: Optional PSR type(s) to filter by. When multiple are
                given, only init times available for all of them are returned.
            limit: Maximum number of init times to return (default 96).

        Returns:
            List of :class:`InitTimeInfo` objects sorted newest-first.

        Raises:
            RuntimeError: If the API request fails.

        Examples:
            >>> init_times = client.power_forecast.get_init_times(
            ...     zone_key="DE", limit=10
            ... )
            >>> for it in init_times:
            ...     print(it.init_time, it.max_prediction_timedelta)
        """
        params: dict = {"limit": limit}
        if zone_key is not None:
            params["zone_key"] = zone_key
        if psr_type is not None:
            if isinstance(psr_type, str):
                psr_type = [psr_type]
            params["psr_type"] = psr_type

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
    ) -> xr.Dataset:
        """Query power forecast data in MW.

        Supports two mutually exclusive query modes:

        **Horizon mode** (init_time-centric):
            Specify ``init_time`` as datetime(s), integer offsets, or
            relative tokens (``"latest"``, ``"latest-N"``). Optionally
            limit the forecast horizon with ``max_prediction_timedelta``.

            Relative tokens and integer offsets are resolved by querying
            available init times filtered by ``zone_keys`` and
            ``psr_types``, so ``"latest"`` always refers to the most
            recent run where *all* requested zone/PSR-type combinations
            have data.

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

        Returns:
            ``xarray.Dataset`` with dimensions ``(zone_key, psr_type, time)``
            and data variable ``value`` in MW.

        Raises:
            ValueError: If both horizon and time-range parameters are given,
                or if neither mode is specified.
            RuntimeError: If the API request fails.

        Examples:
            >>> # Latest solar forecast for Germany
            >>> ds = client.power_forecast.get_data(
            ...     zone_keys=["DE"],
            ...     psr_types=["Solar"],
            ...     init_time="latest",
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

        resolved_init_time = init_time
        if init_time is not None:
            resolved_init_time = self._resolve_init_time(
                init_time, zone_keys, psr_types
            )

        body = self._build_query_body(
            zone_keys=zone_keys,
            psr_types=psr_types,
            init_time=resolved_init_time,
            max_prediction_timedelta=max_prediction_timedelta,
            start_time=start_time,
            end_time=end_time,
            time_zone=time_zone,
        )

        try:
            response = self._api.post(
                "power-forecast/data",
                data=body,
                requires_auth=True,
            )
            data = response.json()
            return self._to_dataset(data)
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
        time_zone: str = "UTC",
        max_init_times: int = 365,
    ) -> xr.Dataset:
        """Return a continuous day-ahead time series stitched across runs.

        This helper selects all forecast runs whose local init-time hour matches
        ``init_hour`` (interpreted in ``time_zone``), takes from each run the
        day-ahead window, and concatenates the results into a single continuous
        ``time`` axis.

        The day-ahead window is defined by the forecast lead range:
        ``[(24 - init_hour), (24 - init_hour) + 24)`` hours from the init time.
        For example, for ``init_hour = 9`` the selected window is
        ``[15h, 39h]`` from each init (i.e., 00:00..23:00 of D when the run is
        at D-1 09:00).

        Args:
            zone_keys: Zone codes to query (e.g. ``["DE"]``).
            psr_types: Optional PSR types to include
                (e.g. ``["Solar"]``). If ``None``, returns all available types.
            init_hour: Local hour-of-day (0..23) of the runs to stitch together.
            time_zone: IANA time zone used to interpret ``init_hour`` when
                matching runs (default ``\"UTC\"``).
            max_init_times: Upper bound on how many matching init times are
                requested from the server (controls history depth).

        Returns:
            ``xarray.Dataset`` with dims ``(zone_key, psr_type, time)`` and
            variable ``value`` (MW). The series is continuous across days.
        """
        if not (0 <= init_hour <= 23):
            raise ValueError("init_hour must be in the range 0..23")
        if not zone_keys or not isinstance(zone_keys, list):
            raise ValueError("zone_keys must be a non-empty list of zone codes")

        # Compute lead range in minutes for the day-ahead slice
        start_lead_hours = (24 - init_hour) % 24
        end_lead_hours = start_lead_hours + 24
        end_lead_minutes = int(end_lead_hours * 60)

        # Fetch recent init times filtered by zone/PSR and keep only those whose
        # local hour (in time_zone) matches init_hour.
        tz = ZoneInfo(time_zone)
        init_infos = self.get_init_times(
            zone_key=zone_keys, psr_type=psr_types, limit=max_init_times
        )
        matching_inits: list[datetime] = []
        for info in init_infos:
            it = info.init_time
            # Ensure timezone-aware then convert to desired zone for comparison
            local_hour = (it if it.tzinfo else it.replace(tzinfo=ZoneInfo("UTC"))).astimezone(tz).hour
            if local_hour == init_hour:
                matching_inits.append(it)

        if not matching_inits:
            # Nothing to stitch – return empty dataset with expected coords
            return xr.Dataset(attrs={"unit": "MW"})

        # Query data up to the end of the day-ahead window to avoid downloading
        # unnecessarily long horizons. We keep all times up to end_lead and
        # filter the earlier hours client-side.
        ds = self.get_data(
            zone_keys=zone_keys,
            psr_types=psr_types,
            init_time=matching_inits,
            max_prediction_timedelta=end_lead_minutes,
            time_zone=time_zone,
        )

        if "value" not in ds:
            return xr.Dataset(attrs={"unit": "MW"})

        # Convert to a DataFrame to easily compute (time - init_time) and filter
        df = ds.to_dataframe().reset_index()
        if df.empty:
            return xr.Dataset(attrs={"unit": "MW"})

        # Drop NaNs and compute lead in hours from each init to each valid time
        df = df.dropna(subset=["value"])
        # Ensure both are timezone-aware for subtraction
        if pd.api.types.is_datetime64_any_dtype(df["init_time"]):
            pass
        df["lead_hours"] = (df["time"] - df["init_time"]) / pd.Timedelta(hours=1)
        mask = (df["lead_hours"] >= start_lead_hours) & (df["lead_hours"] < end_lead_hours)
        df = df.loc[mask].drop(columns=["lead_hours", "init_time"])

        if df.empty:
            return xr.Dataset(attrs={"unit": "MW"})

        # Sort and reindex to have a clean continuous time series
        df = df.sort_values(["zone_key", "psr_type", "time"])

        index_cols = [c for c in ["zone_key", "psr_type", "time"] if c in df.columns]
        stitched = xr.Dataset.from_dataframe(df.set_index(index_cols))
        # Preserve unit
        stitched = stitched.assign_attrs(unit=ds.attrs.get("unit", "MW"))
        return stitched

    # ------------------------------------------------------------------
    # Init-time resolution
    # ------------------------------------------------------------------

    def _resolve_init_time(
        self,
        spec: InitTimeSpec,
        zone_keys: list[str] | None,
        psr_types: list[str] | None,
    ) -> InitTimeSpec:
        """Resolve relative init_time tokens against available init times.

        Queries ``GET /init-times`` filtered by the requested zones and
        PSR types so that ``"latest"`` maps to the newest run where all
        selected combinations have completed data.
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
            zone_keys, psr_types, limit=max_offset + 1
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
    ) -> list[datetime]:
        """Get available init times filtered by zone and PSR types.

        Delegates intersection semantics to the server by passing all
        zone_keys in a single ``/init-times`` call.
        """
        infos = self.get_init_times(
            zone_key=zone_keys,
            psr_type=psr_types,
            limit=limit,
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
    def _build_query_body(
        zone_keys: list[str] | None,
        psr_types: list[str] | None,
        init_time: InitTimeSpec | None,
        max_prediction_timedelta: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        time_zone: str | None,
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

        return remove_none_from_dict(body)

    @staticmethod
    def _to_dataset(data: dict) -> xr.Dataset:
        """Convert columnar JSON response to an xarray Dataset."""
        if not data or all(len(v) == 0 for v in data.values()):
            return xr.Dataset(attrs={"unit": "MW"})

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df["init_time"] = pd.to_datetime(df["init_time"])

        index_cols = ["zone_key", "psr_type", "init_time", "time"]
        present_cols = [c for c in index_cols if c in df.columns]

        ds = xr.Dataset.from_dataframe(df.set_index(present_cols))
        ds = ds.assign_attrs(unit="MW")
        return ds
