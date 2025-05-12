from jua.errors.jua_error import JuaError
from jua.weather.models import Models


class ModelDoesNotExistError(JuaError):
    """Raised when an invalid or unknown model name is requested."""

    def __init__(self, model_name: str):
        """Initialize with the requested model name.

        Args:
            model_name: The name of the model that doesn't exist.
        """
        available_models = "\n".join(Models)
        super().__init__(
            f"Model {model_name} does not exist.\n"
            "Consider using from `jua.weather.models import Models`.\n"
            f"Available models:\n{available_models}"
        )


class ModelDoesNotSupportForecastRawDataAccessError(JuaError):
    """Raised when attempting to access raw forecast data for an unsupported model."""

    def __init__(self, model_name: str):
        """Initialize with the model name.

        Args:
            model_name: The name of the model that doesn't support raw data access.
        """
        super().__init__(
            f"Model {model_name} does not support forecast raw data access."
        )


class ModelHasNoHindcastData(JuaError):
    """Raised when attempting to access hindcast data for a model without hindcasts."""

    def __init__(self, model_name: str):
        """Initialize with the model name.

        Args:
            model_name: The name of the model that doesn't have hindcast data.
        """
        super().__init__(f"Model {model_name} has no hindcast data available.")
