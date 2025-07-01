from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from jua.settings.authentication import AuthenticationSettings


class JuaSettings(BaseSettings):
    """Settings for configuring the Jua SDK client.

    This class contains all configuration options for the Jua API client,
    including API endpoint, authentication, and behavior preferences.
    Settings can be provided via environment variables prefixed with 'JUA_',
    a .env file, or directly in code.

    Attributes:
        api_url: Base URL for the JUA API endpoint.
        api_version: Version of the API to use (e.g., "v1").
        data_base_url: Base URL for data access services.
        auth: Authentication configuration including API keys.
        print_progress: Whether to display progress bars during operations.

    Examples:
        Create with defaults:
        >>> settings = JuaSettings()

        Override specific settings:
        >>> settings = JuaSettings(
            api_url="https://api.example.com",
            print_progress=False,
        )

        Load from environment variables:
        JUA_API_URL=https://api.example.com JUA_PRINT_PROGRESS=false python script.py
    """

    api_url: str = Field(
        default="https://api.jua.ai", description="Base URL for the JUA API"
    )

    api_version: str = Field(
        default="v1", description="API version to use for requests"
    )

    data_base_url: str = Field(
        default="https://data.jua.ai", description="Base URL for JUA data services"
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
        """Determine if progress information should be displayed.

        This method considers both the global setting and any request-specific
        override to determine if progress information should be displayed.

        Args:
            print_progress: If provided, overrides the instance's print_progress
                setting. When None, uses the instance setting.

        Returns:
            True if progress should be displayed, False otherwise.
        """
        if print_progress is None:
            return self.print_progress
        return print_progress
