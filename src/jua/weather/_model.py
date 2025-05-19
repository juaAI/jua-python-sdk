from jua.client import JuaClient
from jua.weather.forecast import Forecast
from jua.weather.hindcast import Hindcast
from jua.weather.models import Models as ModelEnum


class Model:
    """Represents a specific Jua weather model with access to its data.

    A Model provides unified access to both forecast and hindcast data for a
    specific weather model. Each model has unique characteristics such as spatial
    resolution, update frequency, and forecast horizon.

    Attributes:
        _client: The JuaClient instance used for API communication.
        _model: The model identifier enum value.
        _forecast: Pre-initialized Forecast instance for this model.
        _hindcast: Pre-initialized Hindcast instance for this model.

    Examples:
        >>> from jua import JuaClient
        >>> from jua.weather import Models
        >>> client = JuaClient()
        >>> model = client.weather.get_model(Models.EPT2)
        >>> # Access forecast data
        >>> forecast = model.forecast.get_forecast()
        >>> # Access hindcast (historical) data
        >>> hindcast = model.hindcast.get_hindcast(init_time="2023-05-01")
    """

    def __init__(
        self,
        client: JuaClient,
        model: ModelEnum,
    ):
        """Initialize a weather model instance.

        Args:
            client: JuaClient instance for API communication.
            model: The model identifier (from Models enum).
        """
        self._client = client
        self._model = model

        self._forecast = Forecast(
            client,
            model=model,
        )

        self._hindcast = Hindcast(
            client,
            model=model,
        )

    @property
    def name(self) -> str:
        """Get the string name of the model.

        Returns:
            The model name as a string.
        """
        return self._model.value

    @property
    def forecast(self) -> Forecast:
        """Access forecast data for this model.

        Returns:
            Forecast instance configured for this model.
        """
        return self._forecast

    @property
    def hindcast(self) -> Hindcast:
        """Access historical weather data for this model.

        Returns:
            Hindcast instance configured for this model.
        """
        return self._hindcast

    def __repr__(self) -> str:
        """Get string representation of the model.

        Returns:
            A string representation suitable for debugging.
        """
        return f"<Model name='{self.name}'>"

    def __str__(self) -> str:
        """Get the model name as a string.

        Returns:
            The model name.
        """
        return self.name
