"""Zone-addressed market data for the Jua SDK.

``MarketData`` is the single public entry point for observational European
power-market data: renewable generation, load, day-ahead forecasts, and prices.
Callers address data purely by ``market_zone`` and a small, backend-agnostic
vocabulary; the SDK routes each ``(zone, variable)`` to the right underlying
data source and returns a single tidy ``pandas`` DataFrame.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from jua.market_data import _mapping
from jua.market_data._entsoe import _EntsoeBackend
from jua.market_data._frame import UNIFIED_COLUMNS
from jua.market_data._mapping import MarketBackend, MarketVariable
from jua.market_data._uk_power import _UkPowerBackend

if TYPE_CHECKING:
    from jua.client import JuaClient


class MarketData:
    """Interface for Jua's zone-addressed market data.

    Data is addressed by ``market_zone`` (e.g. ``"DE"``, ``"FR"``, ``"GB"``)
    and a curated set of variables that mean the same thing in every zone:

    - ``solar`` / ``wind`` - actual generation (MW)
    - ``load`` - actual demand (MW)
    - ``solar_forecast`` / ``wind_forecast`` - day-ahead generation forecast (MW)
    - ``load_forecast`` - day-ahead demand forecast (MW)
    - ``day_ahead_prices`` - day-ahead market price (EUR/MWh, GBP/MWh for GB)
    - ``imbalance_prices`` - imbalance settlement price

    ``wind`` is the total of all wind sub-types (onshore + offshore). The
    underlying data sources differ by zone and variable, but that is an
    implementation detail - the same call works everywhere.

    Examples:
        >>> from datetime import datetime, timezone
        >>> from jua import JuaClient
        >>>
        >>> client = JuaClient()
        >>> md = client.market_data
        >>>
        >>> md.get_zones()
        ['BE', 'DE', 'FR', 'GB', 'NL']
        >>> md.get_variables(market_zone="GB")
        ['day_ahead_prices', 'imbalance_prices', 'load', 'load_forecast',
         'solar', 'solar_forecast', 'wind', 'wind_forecast']
        >>>
        >>> df = md.get_data(
        ...     market_zone=["DE", "GB"],
        ...     variables=["solar", "wind", "day_ahead_prices"],
        ...     start_time=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ...     end_time=datetime(2025, 12, 3, tzinfo=timezone.utc),
        ... )
        >>> df.columns.tolist()
        ['time', 'market_zone', 'variable', 'value', 'unit']
    """

    def __init__(self, client: JuaClient) -> None:
        self._client = client
        self._entsoe = _EntsoeBackend(client)
        self._uk_power = _UkPowerBackend(client)

    def get_zones(self) -> list[str]:
        """Return the market zones available for querying.

        Returns:
            Sorted list of market zone codes (e.g. ``["BE", "DE", "FR", ...]``).
        """
        return _mapping.supported_zones()

    def get_variables(self, market_zone: str | None = None) -> list[str]:
        """Return the available unified variables, optionally for one zone.

        Args:
            market_zone: When given, restrict to the variables available for
                that zone. When ``None``, return the full vocabulary.

        Returns:
            Sorted list of variable names.

        Raises:
            ValueError: If ``market_zone`` is given but not supported.
        """
        return _mapping.supported_variables(market_zone)

    def get_data(
        self,
        market_zone: str | list[str],
        variables: list[str] | None = None,
        *,
        start_time: datetime,
        end_time: datetime | None = None,
        time_zone: str | None = None,
    ) -> pd.DataFrame:
        """Query market data for one or more zones.

        Args:
            market_zone: Market zone code or list of codes (case-insensitive),
                e.g. ``"DE"`` or ``["DE", "GB"]``.
            variables: Unified variable names to fetch. When ``None``, every
                variable available for each zone is returned.
            start_time: Start of the time range, inclusive. Naive datetimes are
                interpreted as UTC by the server; pass timezone-aware datetimes
                to avoid ambiguity.
            end_time: End of the time range, exclusive. When ``None``, no upper
                bound is applied (useful for day-ahead forecasts that extend
                into tomorrow).
            time_zone: IANA time zone name for the returned ``time`` column
                (e.g. ``"Europe/Berlin"``). Defaults to UTC.

        Returns:
            A long-format ``pandas.DataFrame`` with columns
            ``[time, market_zone, variable, value, unit]``. ``time`` is a
            timezone-aware datetime column. Empty (with those columns) when no
            data matches.

        Raises:
            ValueError: If a zone is unsupported, or a requested variable is not
                available for a requested zone.
            RuntimeError: If an underlying API request fails.

        Examples:
            >>> df = md.get_data(
            ...     market_zone="DE",
            ...     variables=["solar", "wind"],
            ...     start_time=datetime(2025, 12, 1, tzinfo=timezone.utc),
            ...     end_time=datetime(2025, 12, 2, tzinfo=timezone.utc),
            ...     time_zone="Europe/Berlin",
            ... )
        """
        zones = [market_zone] if isinstance(market_zone, str) else list(market_zone)
        if not zones:
            raise ValueError("market_zone must be a non-empty string or list.")

        frames: list[pd.DataFrame] = []
        for zone in zones:
            frames.extend(
                self._fetch_zone(
                    market_zone=zone,
                    variables=variables,
                    start_time=start_time,
                    end_time=end_time,
                    time_zone=time_zone,
                )
            )

        if not frames:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        result = pd.concat(frames, ignore_index=True)
        if result.empty:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)
        return result.sort_values(
            ["market_zone", "variable", "time"]
        ).reset_index(drop=True)

    def _fetch_zone(
        self,
        *,
        market_zone: str,
        variables: list[str] | None,
        start_time: datetime,
        end_time: datetime | None,
        time_zone: str | None,
    ) -> list[pd.DataFrame]:
        """Resolve a single zone's variables and dispatch to the backends."""
        if variables is None:
            requested = [MarketVariable(v) for v in _mapping.supported_variables(
                market_zone
            )]
        else:
            requested = [_mapping._normalize_variable(v) for v in variables]

        # Split requested variables by backend (validates availability).
        by_backend: dict[MarketBackend, list[MarketVariable]] = {}
        for variable in requested:
            capability = _mapping.resolve(market_zone, variable.value)
            by_backend.setdefault(capability.backend, []).append(variable)

        frames: list[pd.DataFrame] = []
        try:
            if MarketBackend.ENTSOE in by_backend:
                frames.append(
                    self._entsoe.fetch(
                        market_zone,
                        by_backend[MarketBackend.ENTSOE],
                        start_time=start_time,
                        end_time=end_time,
                        time_zone=time_zone,
                    )
                )
            if MarketBackend.UK_POWER in by_backend:
                frames.append(
                    self._uk_power.fetch(
                        market_zone,
                        by_backend[MarketBackend.UK_POWER],
                        start_time=start_time,
                        end_time=end_time,
                        time_zone=time_zone,
                    )
                )
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface a uniform error
            raise RuntimeError(
                f"Failed to fetch market data for zone '{market_zone}': {exc}"
            ) from exc

        return [f for f in frames if not f.empty]
