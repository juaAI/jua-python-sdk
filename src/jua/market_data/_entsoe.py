"""Internal ENTSOE backend for the unified market_data interface.

Wraps the Query Engine ``/v1/entsoe`` endpoints and normalizes responses into
the unified ``[time, market_zone, variable, value, unit]`` schema. SDK users do
not interact with this module; they go through :class:`jua.market_data.MarketData`.
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


class _EntsoeBackend:
    """Fetches and normalizes ENTSOE timeseries for a single market zone."""

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
    ) -> pd.DataFrame:
        """Fetch unified ``variables`` for one zone from ENTSOE.

        Variables that resolve to the same native ENTSOE variable (e.g. solar
        and wind both come from ``generation_actual``) are fetched in a single
        request and split apart afterwards. PSR-based variables are summed per
        timestamp (wind = onshore + offshore).

        Returns:
            DataFrame with the unified columns, or empty if no data.
        """
        zone = _mapping.entsoe_zone(market_zone)

        # Group unified variables by their native ENTSOE variable so each
        # native variable is requested at most once.
        groups: dict[str, list[_mapping.MarketVariable]] = {}
        for variable in variables:
            binding = _mapping.resolve(market_zone, variable.value).entsoe
            assert binding is not None  # routed here, so always set
            groups.setdefault(binding.variable, []).append(variable)

        frames: list[pd.DataFrame] = []
        for native_variable, unified_vars in groups.items():
            raw = self._fetch_native(
                native_variable=native_variable,
                unified_vars=unified_vars,
                entsoe_zone=zone,
                start_time=start_time,
                end_time=end_time,
                time_zone=time_zone,
            )
            if raw.empty:
                continue
            for variable in unified_vars:
                frame = self._normalize_variable(raw, market_zone, variable)
                if not frame.empty:
                    frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _fetch_native(
        self,
        *,
        native_variable: str,
        unified_vars: list[_mapping.MarketVariable],
        entsoe_zone: str,
        start_time: datetime,
        end_time: datetime | None,
        time_zone: str | None,
    ) -> pd.DataFrame:
        """Request a single native ENTSOE variable and parse the response."""
        psr_types: list[str] = []
        for variable in unified_vars:
            binding = _mapping._ENTSOE_BINDINGS[variable]
            psr_types.extend(binding.psr_types)

        body = remove_none_from_dict(
            {
                "variables": [native_variable],
                "zone_keys": [entsoe_zone],
                "psr_types": sorted(set(psr_types)) or None,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat() if end_time else None,
                "time_zone": time_zone,
            }
        )
        response = self._api.post("entsoe/data", data=body, requires_auth=True)
        df = decode_columnar(response.json())
        return parse_time(df, time_zone)

    @staticmethod
    def _normalize_variable(
        raw: pd.DataFrame,
        market_zone: str,
        variable: _mapping.MarketVariable,
    ) -> pd.DataFrame:
        """Filter, sum PSR rows, and rename to the unified schema."""
        binding = _mapping._ENTSOE_BINDINGS[variable]

        df = raw
        if binding.psr_types and "psr_type" in df.columns:
            df = df[df["psr_type"].isin(binding.psr_types)]
        if df.empty:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        unit = _first_unit(df)
        # Sum across PSR types per timestamp (identity for single-PSR / non-PSR).
        summed = df.groupby("time", as_index=False, sort=True)["value"].sum(min_count=1)
        summed["market_zone"] = market_zone.upper()
        summed["variable"] = variable.value
        summed["unit"] = unit
        return summed[UNIFIED_COLUMNS]

    # ------------------------------------------------------------------
    # Discovery passthrough (used to validate / surface availability)
    # ------------------------------------------------------------------
    def available_zones(self) -> list[str]:
        """Return raw ENTSOE zone codes that have data (diagnostic)."""
        response = self._api.get("entsoe/zones", requires_auth=True)
        return response.json().get("zones", [])


def _first_unit(df: pd.DataFrame) -> str | None:
    """Return the first non-null unit in the frame, if any."""
    if "unit" not in df.columns:
        return None
    units = df["unit"].dropna()
    return str(units.iloc[0]) if not units.empty else None
