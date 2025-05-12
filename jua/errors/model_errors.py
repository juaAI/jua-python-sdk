from jua.errors.jua_error import JuaError
from jua.weather.models import Models


class ModelDoesNotExistError(JuaError):
    def __init__(self, model_name: str):
        available_models = "\n".join(Models)
        super().__init__(
            f"Model {model_name} does not exist.\n"
            "Consider using from `jua.weather.models import Models`.\n"
            f"Available models:\n{available_models}"
        )


class ModelDoesNotSupportForecastRawDataAccessError(JuaError):
    def __init__(self, model_name: str):
        super().__init__(
            f"Model {model_name} does not support forecast raw data access."
        )


class ModelHasNoHindcastData(JuaError):
    def __init__(self, model_name: str):
        super().__init__(f"Model {model_name} has no hindcast data available.")
