from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from jua.settings.authentication import AuthenticationSettings


class JuaSettings(BaseSettings):
    api_url: str = Field("https://api.jua.sh", env="JUA_API_URL")
    api_version: str = Field("v1", env="JUA_API_VERSION")

    data_base_url: str = Field("https://data.jua.sh", env="JUA_DATA_BASE_URL")

    auth: AuthenticationSettings = Field(default_factory=AuthenticationSettings)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
