import logging

from pydantic import validate_call

from jua.settings.jua_settings import JuaSettings


class JuaClient:
    """Main entry points for the Jua SDK.

    JuaClient provides access to all Jua services through a unified interface.

    Attributes:
        settings: Configuration settings for the API client.
        weather: Property that provides access to weather data services.
        market_aggregates: Property that provides access to market aggregate services.

    Examples:
        >>> from jua import JuaClient
        >>> client = JuaClient()
        >>> # Access weather services
        >>> forecast_model = client.weather.get_model(...)
        >>> # Access market aggregates
        >>> market_data = client.market_aggregates.compare_runs(...)
    """

    @validate_call
    def __init__(
        self,
        settings: JuaSettings = JuaSettings(),
        request_credit_limit: int | None = None,
        jua_log_level: int | None = None,
    ):
        """Initialize a new Jua client.

        Args:
            settings: Optional configuration settings. If not provided,
                default settings will be used.
            request_credit_limit: Sets the maximum number of credits that can be
                consumed by a single request. If None, the default maximum is
                used.
        """
        self.settings = settings
        self.request_credit_limit = request_credit_limit
        self._weather = None
        self._market_aggregates = None

        if jua_log_level is not None:
            logging.getLogger("jua").setLevel(jua_log_level)

    @property
    def weather(self):
        """Access to Jua's weather data services.

        Returns:
            Weather client interface for querying weather data.
        """
        if self._weather is None:
            from jua.weather._weather import Weather

            self._weather = Weather(self)
        return self._weather

    @property
    def market_aggregates(self):
        """Access to Jua's market aggregate data services.

        Returns:
            MarketAggregates client interface for querying market aggregate data.
        """
        if self._market_aggregates is None:
            from jua.market_aggregates import MarketAggregates

            self._market_aggregates = MarketAggregates(self)
        return self._market_aggregates

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
