from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from jua.settings.authentication import AuthenticationSettings


class JuaSettings(BaseSettings):
    api_url: str = Field(
        default="https://api.jua.sh", description="Base URL for the JUA API"
    )

    api_version: str = Field(
        default="v1", description="API version to use for requests"
    )

    data_base_url: str = Field(
        default="https://data.jua.sh", description="Base URL for JUA data services"
    )

    auth: AuthenticationSettings = Field(
        default_factory=AuthenticationSettings,
        description="Authentication settings for JUA API",
    )

    print_progress: bool = Field(
        default=True, description="Whether to print progress information"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="JUA_",
    )

    def should_print_progress(self, print_progress: bool | None = None) -> bool:
        """
        Determine if progress should be printed.

        Args:
            print_progress: Optional override for the print_progress setting

        Returns:
            Boolean indicating whether progress should be printed
        """
        if print_progress is None:
            return self.print_progress
        return print_progress
