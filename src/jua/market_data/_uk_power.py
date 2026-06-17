"""Internal UK-power backend for the unified market_data interface.

Wraps the Query Engine ``/v1/uk-power`` endpoints (GB actuals from Elexon /
PV_Live and NESO day-ahead forecasts) and normalizes responses into the unified
``[time, market_zone, variable, value, unit]`` schema. Internal module.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from jua._api import QueryEngineAPI
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua.market_data import _mapping
from jua.market_data._frame import UNIFIED_COLUMNS, decode_columnar, parse_time

if TYPE_CHECKING:
    from jua.client import JuaClient

#: UK power is GB only.
_MARKET_ZONE = "GB"

#: Composite uk-power variables that cannot share a request with their own
#: components. Each "total" (e.g. ``wind`` = transmission + embedded) is drawn
#: from the same underlying columns as its parts, and the Query Engine returns a
#: 500 when a total and any of its components are requested together. The two
#: groups are independent: a total only collides with its *own* components
#: (e.g. ``wind`` + ``wind_embedded`` fails, but ``wind`` + ``wind_forecast`` or
#: ``wind_forecast`` + ``wind_embedded`` are fine). We split such requests so
#: callers can ask for a total and its parts in a single SDK call.
_WIND_TOTAL_COMPONENTS: dict[str, frozenset[str]] = {
    "wind": frozenset({"wind_embedded", "wind_transmission"}),
    "wind_forecast": frozenset(
        {"wind_embedded_forecast", "wind_transmission_forecast"}
    ),
}


class _UkPowerBackend:
    """Fetches and normalizes UK power timeseries (GB only)."""

    def __init__(self, client: JuaClient) -> None:
        self._client = client
        self._api = QueryEngineAPI(jua_client=client)

    def fetch(
        self,
        market_zone: str,
        variables: list[_mapping.MarketVariable],
        *,
        start_time: datetime,
        end_time: datetime | None,
        time_zone: str | None,
        temporal_resolution_minutes: int | None = None,
    ) -> pd.DataFrame:
        """Fetch unified ``variables`` for GB from the UK-power feed.

        Returns:
            DataFrame with the unified columns, or empty if no data.
        """
        # unified variable -> native uk-power variable name
        native_by_unified: dict[_mapping.MarketVariable, str] = {}
        for variable in variables:
            native = _mapping.resolve(market_zone, variable.value).uk_power_variable
            assert native is not None  # routed here, so always set
            native_by_unified[variable] = native

        raw = self._fetch_native(
            sorted(set(native_by_unified.values())),
            start_time=start_time,
            end_time=end_time,
            time_zone=time_zone,
            temporal_resolution_minutes=temporal_resolution_minutes,
        )
        if raw.empty:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        frames: list[pd.DataFrame] = []
        for variable, native in native_by_unified.items():
            frame = self._normalize_variable(raw, variable, native)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _fetch_native(
        self,
        natives: list[str],
        *,
        start_time: datetime,
        end_time: datetime | None,
        time_zone: str | None,
        temporal_resolution_minutes: int | None,
    ) -> pd.DataFrame:
        """Request native uk-power variables, splitting incompatible groups.

        Composite totals cannot be queried alongside their own components (see
        :data:`_WIND_TOTAL_COMPONENTS`), so each conflicting total is sent in its
        own request and the columnar responses are concatenated. Everything else
        is fetched together in a single request.
        """
        frames: list[pd.DataFrame] = []
        for batch in self._split_into_compatible_batches(natives):
            body = remove_none_from_dict(
                {
                    "variables": batch,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat() if end_time else None,
                    "temporal_resolution_minutes": temporal_resolution_minutes,
                    "time_zone": time_zone,
                }
            )
            response = self._api.post("uk-power/data", data=body, requires_auth=True)
            frame = parse_time(decode_columnar(response.json()), time_zone)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _split_into_compatible_batches(natives: list[str]) -> list[list[str]]:
        """Split native variables so no total shares a request with its own
        components.

        Each total that is requested together with at least one of its
        components is isolated into a singleton batch; everything else (other
        components, unrelated variables, and totals whose components were not
        requested) is grouped into one final batch. A total left in the grouped
        batch is safe because its components are absent, and components never
        collide with a *different* total. Order within each batch is sorted for
        deterministic requests.
        """
        unique = sorted(set(natives))
        present = set(unique)

        isolated = [
            total
            for total, components in _WIND_TOTAL_COMPONENTS.items()
            if total in present and components & present
        ]
        if not isolated:
            return [unique]

        isolated_set = set(isolated)
        rest = [n for n in unique if n not in isolated_set]
        batches = [[total] for total in sorted(isolated)]
        if rest:
            batches.append(rest)
        return batches

    @staticmethod
    def _normalize_variable(
        raw: pd.DataFrame,
        variable: _mapping.MarketVariable,
        native: str,
    ) -> pd.DataFrame:
        """Select the native variable's rows and rename to the unified schema."""
        if "variable_name" not in raw.columns:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        df = raw[raw["variable_name"] == native].copy()
        if df.empty:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        df["market_zone"] = _MARKET_ZONE
        df["variable"] = variable.value
        if "unit" not in df.columns:
            df["unit"] = None
        return df[UNIFIED_COLUMNS]

    # ------------------------------------------------------------------
    # Discovery passthrough
    # ------------------------------------------------------------------
    def available_sources(self) -> list[str]:
        """Return raw UK-power data source names (diagnostic)."""
        response = self._api.get("uk-power/sources", requires_auth=True)
        return response.json().get("sources", [])
