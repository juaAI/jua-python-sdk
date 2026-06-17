"""Market aggregates module for the Jua SDK."""

from jua._api import QueryEngineAPI
from jua.client import JuaClient
from jua.market_aggregates.energy_market import EnergyMarket
from jua.types import MarketZones


class MarketAggregates:
    """Main interface for market aggregate services in the Jua SDK.

    This class manages access to aggregated forecast data by market zone,
    with automatic weighting appropriate for each variable type.

    Similar to the Weather class, this provides a convenient entry point for
    accessing aggregate data for different market zones.

    Examples:
        >>> from jua import JuaClient
        >>> from jua.market_aggregates import AggregateVariables, ModelRuns
        >>> from jua.weather import Models
        >>> from jua.types import MarketZones
        >>>
        >>> client = JuaClient()
        >>>
        >>> # Get a specific market
        >>> germany = client.market_aggregates.get_market(
        ...     market_zone=[MarketZones.DE]
        ... )
        >>>
        >>> # Query data for that market
        >>> model_runs = [ModelRuns(Models.EPT2, [0, 1])]
        >>> data = germany.compare_runs(
        ...     agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
        ...     model_runs=model_runs,
        ...     max_lead_time=48,
        ... )
    """

    def __init__(self, client: JuaClient) -> None:
        """Initialize the market aggregates interface.

        Args:
            client: JuaClient instance for API communication.
        """
        self._client = client
        self._query_engine_api = QueryEngineAPI(jua_client=self._client)

    def get_market(
        self, market_zone: MarketZones | str | list[MarketZones | str]
    ) -> EnergyMarket:
        """Get an EnergyMarket for querying aggregate data for specific zones.

        Args:
            market_zone: Market zone identifier(s). Can be:
                - Single MarketZones enum (e.g., MarketZones.DE)
                - Single string (e.g., "DE")
                - List of MarketZones (e.g., [MarketZones.DE, MarketZones.FR])
                - List of strings (e.g., ["DE", "FR"])

        Returns:
            An EnergyMarket instance that can be used to query data for the
            specified market zones.

        Examples:
            >>> germany = market_aggregates.get_market(MarketZones.DE)
            >>> ireland_northern_ireland = market_aggregates.get_market(
            >>>     [MarketZones.IE, MarketZones.GB_NIR]
            >>> )
        """
        return EnergyMarket(client=self._client, market_zone=market_zone)

    def get_mw_zones(self) -> dict[str, list[str]]:
        """Get market zones capable of MW output, by output type.

        Returns the zones that can produce predicted MW for each output, i.e.
        the zones that have the data and fitted models required. Generation
        outputs (``"wind"`` / ``"solar"``) need installed capacity and power
        curves; demand (``"load"``) needs a fitted demand model.

        Returns:
            Dictionary mapping each output type to a list of zone codes. Always
            includes ``"wind"``, ``"solar"`` and ``"load"``; the API may also
            return finer-grained wind keys (e.g. ``"wind_combined"``,
            ``"wind_onshore_only"``), which are passed through when present.

        Raises:
            RuntimeError: If the API request fails.

        Examples:
            >>> mw_zones = client.market_aggregates.get_mw_zones()
            >>> print(mw_zones["wind"])   # ["AT", "BE", "DE", "FR", ...]
            >>> print(mw_zones["solar"])  # ["AT", "BE", "DE", "FR", ...]
            >>> print(mw_zones["load"])   # ["AL", "AT", "BE", "DE", "FR", ...]
        """
        try:
            response = self._query_engine_api.get(
                "forecast/market-aggregate/mw-zones",
                requires_auth=False,
            )
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch MW-capable market zones: {e}") from e

        # Pass through every output type the API reports, guaranteeing the
        # documented keys exist even if a future response omits one.
        result = {key: list(values) for key, values in data.items()}
        for key in ("wind", "solar", "load"):
            result.setdefault(key, [])
        return result
