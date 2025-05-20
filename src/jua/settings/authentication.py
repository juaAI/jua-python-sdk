import json
from pathlib import Path

from aiohttp import BasicAuth
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthenticationSettings(BaseSettings):
    """Authentication settings for JUA API.

    Manages API credentials for authenticating with the Jua platform. Credentials
    can be provided from multiple sources with the following priority order:

    1. Direct initialization via constructor arguments
    2. Environment variables (JUA_API_KEY_ID, JUA_API_KEY_SECRET)
    3. JSON file at the specified secrets_path or default location
       (~/.jua/<environment>/api-key.json)

    Attributes:
        api_key_id: API key identifier for authentication with Jua services.
        api_key_secret: Secret key that pairs with the api_key_id.
        environment: Environment name used to determine configuration settings
            and default secrets file location.
        secrets_path: Optional custom path to load API credentials from a JSON file.
            If provided, overrides the default location.

    Examples:
        Create with explicit credentials:
        >>> auth = AuthenticationSettings(
            api_key_id="your_id",
            api_key_secret="your_secret",
        )

        Create using credentials from environment variables:
        >>> auth = AuthenticationSettings()

        Create with custom JSON file location:
        >>> auth = AuthenticationSettings(secrets_path="/path/to/secrets.json")
    """

    api_key_id: str | None = Field(
        default=None, description="API key identifier for authentication"
    )
    api_key_secret: str | None = Field(
        default=None, description="Secret key for API authentication"
    )
    environment: str = Field(
        default="default",
        description="Environment name to determine configuration settings",
    )
    api_key_path: str | None = Field(
        default=None, description="Custom path to load API credentials from a JSON file"
    )

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="JUA_",
    )

    def __init__(
        self,
        api_key_id: str | None = None,
        api_key_secret: str | None = None,
        environment: str = "default",
        api_key_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if api_key_id is not None:
            self.api_key_id = api_key_id
        if api_key_secret is not None:
            self.api_key_secret = api_key_secret
        if environment is not None:
            self.environment = environment
        if api_key_path is not None:
            self.api_key_path = api_key_path

        # If credentials are not set via env vars, try loading from JSON file
        if not self.api_key_id or not self.api_key_secret:
            self._load_from_json_file()

    def _load_from_json_file(self) -> None:
        """Load credentials from JSON file if needed.

        Attempts to read API credentials from either the specified secrets_path
        or the default location (~/.jua/<environment>/api-key.json). Silently
        fails if the file doesn't exist or contains invalid JSON.
        """
        # Determine the secrets file path
        if self.api_key_path:
            file_path = Path(self.api_key_path)
        else:
            file_path = Path.home() / ".jua" / self.environment / "api-key.json"

        if not file_path.exists():
            return

        try:
            with open(file_path, "r") as f:
                secrets_data = json.load(f)

            self.api_key_id = secrets_data.get("id")
            self.api_key_secret = secrets_data.get("secret")
        except (json.JSONDecodeError, IOError):
            # Silently fail if file cannot be read or parsed
            pass

    @property
    def is_authenticated(self) -> bool:
        """Check if authentication credentials are properly set.

        Returns:
            True if both api_key_id and api_key_secret are non-empty,
            False otherwise.
        """
        return bool(self.api_key_id and self.api_key_secret)

    def get_basic_auth(self) -> BasicAuth:
        """Get BasicAuth object for HTTP authentication.

        Constructs an aiohttp.BasicAuth object using the API credentials.

        Returns:
            BasicAuth object that can be used with aiohttp client requests.
        """
        return BasicAuth(login=self.api_key_id, password=self.api_key_secret)
