"""Shared response-decoding helpers for the market data backends.

Internal module. Centralizes columnar-JSON decoding and DST-safe parsing of
the ``time`` column so the ENTSOE and UK-power backends behave identically.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

#: Column order of the normalized, unified market-data frame.
UNIFIED_COLUMNS: list[str] = ["time", "market_zone", "variable", "value", "unit"]


def decode_columnar(payload: dict) -> pd.DataFrame:
    """Decode a Query Engine columnar-JSON body into a DataFrame.

    The default ``format=json`` responses are columnar (``{column: [values]}``).
    When ``include_units=true`` the body is wrapped as ``{"data": ..., "units":
    ...}``; we unwrap defensively even though this client never requests it.

    Args:
        payload: Parsed JSON response body.

    Returns:
        A DataFrame (empty when the response carries no rows).
    """
    if isinstance(payload, dict) and "data" in payload and "units" in payload:
        payload = payload["data"]

    if not payload or all(len(v) == 0 for v in payload.values()):
        return pd.DataFrame()

    return pd.DataFrame(payload)


def parse_time(df: pd.DataFrame, time_zone: str | None) -> pd.DataFrame:
    """Parse the ``time`` column to a DST-safe, timezone-aware dtype.

    The server emits ISO-8601 timestamps whose UTC offset changes across a
    daylight-saving transition. Parsing those naively yields an ``object``
    column (mixed offsets), so we always normalize through UTC first and then
    convert to the requested zone, preserving local wall-clock without dropping
    or duplicating the transition hour.

    Args:
        df: Frame containing a ``time`` column (no-op if absent or empty).
        time_zone: IANA zone the caller requested, or ``None`` for UTC.

    Returns:
        The same frame with ``time`` as a tz-aware datetime column.
    """
    if df.empty or "time" not in df.columns:
        return df

    times = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    if time_zone is not None:
        times = times.dt.tz_convert(ZoneInfo(time_zone))
    df = df.copy()
    df["time"] = times
    return df
