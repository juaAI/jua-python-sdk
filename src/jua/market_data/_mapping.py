"""Unified vocabulary, zone mapping, and capability matrix for market data.

This module is the single source of truth for how the public, zone-addressed
``market_data`` vocabulary maps onto the underlying Query Engine backends
(ENTSOE and UK power). It is internal: SDK users never import from here.

The design goal is that a caller addresses observational market data purely by
``market_zone`` (e.g. ``"DE"``, ``"GB"``) and a small curated set of variable
names that mean the same thing everywhere. The routing from a
``(market_zone, variable)`` pair to a concrete backend query lives here so the
public surface and the backends stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MarketBackend(StrEnum):
    """Underlying data backend a unified variable is served from."""

    ENTSOE = "entsoe"
    UK_POWER = "uk_power"


class MarketVariable(StrEnum):
    """Curated, backend-agnostic market data variables.

    The same name resolves to the appropriate native query for every
    supported zone (see :data:`CAPABILITY_MATRIX`).
    """

    SOLAR = "solar"
    WIND = "wind"
    LOAD = "load"
    SOLAR_FORECAST = "solar_forecast"
    WIND_FORECAST = "wind_forecast"
    LOAD_FORECAST = "load_forecast"
    DAY_AHEAD_PRICES = "day_ahead_prices"
    IMBALANCE_PRICES = "imbalance_prices"


# Market zone -> ENTSOE bidding/control zone code. Most market zones use the
# same code on the ENTSO-E Transparency Platform; Germany is published under
# the joint Germany-Luxembourg bidding zone.
MARKET_ZONE_TO_ENTSOE: dict[str, str] = {
    "DE": "DE_LU",
    "FR": "FR",
    "NL": "NL",
    "BE": "BE",
    "GB": "GB",
}

# ENTSOE PSR types that make up each unified renewable variable. Wind is the
# onshore + offshore total per the agreed "total wind only" vocabulary.
ENTSOE_SOLAR_PSR: tuple[str, ...] = ("Solar",)
ENTSOE_WIND_PSR: tuple[str, ...] = ("Wind Onshore", "Wind Offshore")


@dataclass(frozen=True)
class EntsoeBinding:
    """How a unified variable maps onto an ENTSOE ``/data`` query.

    Attributes:
        variable: Native ``EntsoeVariable`` name (e.g. ``"generation_actual"``).
        psr_types: PSR types to request and sum into the unified value.
            Empty for non-PSR variables (load, prices).
    """

    variable: str
    psr_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class Capability:
    """Resolution of a single ``(market_zone, variable)`` pair.

    Exactly one of ``entsoe`` / ``uk_power_variable`` is populated, matching
    ``backend``.
    """

    backend: MarketBackend
    entsoe: EntsoeBinding | None = None
    uk_power_variable: str | None = None


# ENTSOE bindings for the unified variables, reused across EU zones and the
# GB price/forecast fallbacks.
_ENTSOE_BINDINGS: dict[MarketVariable, EntsoeBinding] = {
    MarketVariable.SOLAR: EntsoeBinding("generation_actual", ENTSOE_SOLAR_PSR),
    MarketVariable.WIND: EntsoeBinding("generation_actual", ENTSOE_WIND_PSR),
    MarketVariable.SOLAR_FORECAST: EntsoeBinding(
        "wind_solar_forecast_da", ENTSOE_SOLAR_PSR
    ),
    MarketVariable.WIND_FORECAST: EntsoeBinding(
        "wind_solar_forecast_da", ENTSOE_WIND_PSR
    ),
    MarketVariable.LOAD: EntsoeBinding("load_actual"),
    MarketVariable.LOAD_FORECAST: EntsoeBinding("load_forecast_da"),
    MarketVariable.DAY_AHEAD_PRICES: EntsoeBinding("day_ahead_prices"),
    MarketVariable.IMBALANCE_PRICES: EntsoeBinding("imbalance_prices"),
}


def _entsoe_capability(variable: MarketVariable) -> Capability:
    return Capability(backend=MarketBackend.ENTSOE, entsoe=_ENTSOE_BINDINGS[variable])


# Every unified variable is available for EU zones via ENTSOE.
_EU_CAPABILITIES: dict[MarketVariable, Capability] = {
    variable: _entsoe_capability(variable) for variable in MarketVariable
}

# GB serves renewables + load actual + renewable day-ahead forecasts from the
# richer UK-power feed (Elexon / PV_Live / NESO), and falls back to ENTSOE's
# GB zone for load_forecast and prices (the /v1/uk-power endpoint exposes no
# prices or load forecast).
_GB_CAPABILITIES: dict[MarketVariable, Capability] = {
    MarketVariable.SOLAR: Capability(MarketBackend.UK_POWER, uk_power_variable="solar"),
    MarketVariable.WIND: Capability(MarketBackend.UK_POWER, uk_power_variable="wind"),
    MarketVariable.LOAD: Capability(MarketBackend.UK_POWER, uk_power_variable="load"),
    MarketVariable.SOLAR_FORECAST: Capability(
        MarketBackend.UK_POWER, uk_power_variable="solar_forecast"
    ),
    MarketVariable.WIND_FORECAST: Capability(
        MarketBackend.UK_POWER, uk_power_variable="wind_forecast"
    ),
    MarketVariable.LOAD_FORECAST: _entsoe_capability(MarketVariable.LOAD_FORECAST),
    MarketVariable.DAY_AHEAD_PRICES: _entsoe_capability(
        MarketVariable.DAY_AHEAD_PRICES
    ),
    MarketVariable.IMBALANCE_PRICES: _entsoe_capability(
        MarketVariable.IMBALANCE_PRICES
    ),
}

# market_zone -> {unified variable -> capability}
CAPABILITY_MATRIX: dict[str, dict[MarketVariable, Capability]] = {
    "DE": dict(_EU_CAPABILITIES),
    "FR": dict(_EU_CAPABILITIES),
    "NL": dict(_EU_CAPABILITIES),
    "BE": dict(_EU_CAPABILITIES),
    "GB": _GB_CAPABILITIES,
}


def supported_zones() -> list[str]:
    """Return the market zones the SDK can serve, sorted."""
    return sorted(CAPABILITY_MATRIX)


def supported_variables(market_zone: str | None = None) -> list[str]:
    """Return the unified variables available, optionally for one zone.

    Args:
        market_zone: When given, restrict to variables available for that
            zone. When ``None``, return the full unified vocabulary.

    Returns:
        Sorted list of unified variable names.

    Raises:
        ValueError: If ``market_zone`` is given but not supported.
    """
    if market_zone is None:
        return sorted(v.value for v in MarketVariable)
    zone = _normalize_zone(market_zone)
    return sorted(v.value for v in CAPABILITY_MATRIX[zone])


def _normalize_zone(market_zone: str) -> str:
    """Validate and canonicalize a market zone code (upper-cased)."""
    zone = market_zone.strip().upper()
    if zone not in CAPABILITY_MATRIX:
        available = ", ".join(supported_zones())
        raise ValueError(
            f"Unsupported market zone '{market_zone}'. "
            f"Available zones: {available}."
        )
    return zone


def _normalize_variable(variable: str) -> MarketVariable:
    """Validate and parse a unified variable name."""
    try:
        return MarketVariable(variable)
    except ValueError as exc:
        available = ", ".join(sorted(v.value for v in MarketVariable))
        raise ValueError(
            f"Unknown market variable '{variable}'. "
            f"Available variables: {available}."
        ) from exc


def resolve(market_zone: str, variable: str) -> Capability:
    """Resolve a ``(market_zone, variable)`` pair to a backend capability.

    Args:
        market_zone: Market zone code (e.g. ``"DE"``, ``"GB"``). Case-insensitive.
        variable: Unified variable name (e.g. ``"solar"``).

    Returns:
        The :class:`Capability` describing which backend and native query to use.

    Raises:
        ValueError: If the zone or variable is unknown, or if the variable is
            not available for that zone.
    """
    zone = _normalize_zone(market_zone)
    market_variable = _normalize_variable(variable)

    zone_caps = CAPABILITY_MATRIX[zone]
    if market_variable not in zone_caps:
        available = ", ".join(sorted(v.value for v in zone_caps))
        raise ValueError(
            f"Variable '{variable}' is not available for zone '{zone}'. "
            f"Available for {zone}: {available}."
        )
    return zone_caps[market_variable]


def entsoe_zone(market_zone: str) -> str:
    """Map a market zone to its ENTSOE zone code (e.g. ``DE -> DE_LU``)."""
    return MARKET_ZONE_TO_ENTSOE[_normalize_zone(market_zone)]
