"""Market aggregates module for the Jua SDK."""

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
