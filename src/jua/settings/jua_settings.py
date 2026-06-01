from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from jua.settings.authentication import AuthenticationSettings

DEFAULT_RETRY_STATUS_CODES = [408, 425, 429, 500, 502, 503, 504]


class JuaSettings(BaseSettings):
    """Settings for configuring the Jua SDK client.

    This class contains all configuration options for the Jua API client,
    including API endpoint, authentication, and behavior preferences.
    Settings can be provided via environment variables prefixed with 'JUA_',
    or directly in code.

    Attributes:
        api_url: Base URL for the JUA API endpoint.
        api_version: Version of the API to use (e.g., "v1").
        auth: Authentication configuration including API keys.
        print_progress: Whether to display progress bars during operations.
        max_retries: Number of times a transient HTTP failure is retried at the
            HTTP level before raising. Set to 0 to disable retries.
        retry_backoff_factor: Base factor (in seconds) for the exponential backoff
            applied between retries.
        retry_backoff_max: Upper bound (in seconds) on the time waited between two
            retries. Caps both the exponential backoff and any ``Retry-After``
            header returned by the server.
        retry_status_codes: HTTP status codes that should trigger an automatic
            retry.
        respect_retry_after_header: Whether to honor the ``Retry-After`` header
            returned by the server (capped by ``retry_backoff_max``).

    Examples:
        Create with defaults:
        >>> settings = JuaSettings()

        Override specific settings:
        >>> settings = JuaSettings(
            api_url="https://api.example.com",
            print_progress=False,
        )

        Tune the HTTP retry behavior:
        >>> settings = JuaSettings(max_retries=5, retry_backoff_factor=1.0)

        Load from environment variables:
        JUA_API_URL=https://api.example.com JUA_PRINT_PROGRESS=false python script.py
    """

    frontend_url: str = Field(
        default="https://athena.jua.ai",
        description="Base URL for the JUA developer frontend",
    )

    api_url: str = Field(
        default="https://api.jua.ai", description="Base URL for the JUA API"
    )

    api_version: str = Field(
        default="v1", description="API version to use for requests"
    )

    query_engine_url: str = Field(
        default="https://query.jua.ai", description="Base URL for the JUA Query Engine"
    )

    query_engine_version: str = Field(
        default="v1", description="Query Engine version to use for requests"
    )

    auth: AuthenticationSettings = Field(
        default_factory=AuthenticationSettings,
        description="Authentication settings for JUA API",
    )

    print_progress: bool = Field(
        default=True, description="Whether to print progress information"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        description=(
            "Number of times a transient HTTP failure (connection errors and "
            "retryable status codes such as 502/503/504) is retried at the HTTP "
            "level before raising. Set to 0 to disable retries."
        ),
    )

    retry_backoff_factor: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Base factor in seconds for the exponential backoff applied between "
            "retries. The n-th retry waits "
            "retry_backoff_factor * (2 ** (n - 1)) seconds."
        ),
    )

    retry_backoff_max: float = Field(
        default=60.0,
        ge=0.0,
        description=(
            "Upper bound in seconds on the time waited between two retries. Caps "
            "both the exponential backoff and any Retry-After header returned by "
            "the server."
        ),
    )

    retry_status_codes: list[int] = Field(
        default_factory=lambda: list(DEFAULT_RETRY_STATUS_CODES),
        description="HTTP status codes that should trigger an automatic retry.",
    )

    respect_retry_after_header: bool = Field(
        default=True,
        description=(
            "Whether to honor the Retry-After header returned by the server "
            "(capped by retry_backoff_max)."
        ),
    )

    model_config = SettingsConfigDict(
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
