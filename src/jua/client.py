from pydantic import validate_call

from jua.settings.jua_settings import JuaSettings


class JuaClient:
    """Main entry points for the Jua SDK.

    JuaClient provides access to all Jua services through a unified interface.

    Attributes:
        settings: Configuration settings for the API client.
        weather: Property that provides access to weather data services.

    Examples:
        >>> from jua import JuaClient
        >>> client = JuaClient()
        >>> # Access weather services
        >>> forecast_model = client.weather.get_model(...)
    """

    @validate_call
    def __init__(self, settings: JuaSettings = JuaSettings()):
        """Initialize a new Jua client.

        Args:
            settings: Optional configuration settings. If not provided,
                default settings will be used.
        """
        self.settings = settings
        self._weather = None

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
