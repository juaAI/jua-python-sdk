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

        body = remove_none_from_dict(
            {
                "variables": sorted(set(native_by_unified.values())),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat() if end_time else None,
                "temporal_resolution_minutes": temporal_resolution_minutes,
                "time_zone": time_zone,
            }
        )
        response = self._api.post("uk-power/data", data=body, requires_auth=True)
        raw = parse_time(decode_columnar(response.json()), time_zone)
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
