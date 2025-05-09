import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthenticationSettings(BaseSettings):
    """
    Authentication settings for JUA API.

    Credential sources (in order of priority):
    1. Environment variables (JUA_API_KEY_ID, JUA_API_KEY_SECRET)
    2. .env file in current directory
    3. JSON file at secrets_path or ~/.jua/api-key.json
    """

    api_key_id: str | None = Field(None, env="JUA_API_KEY_ID")
    api_key_secret: str | None = Field(None, env="JUA_API_KEY_SECRET")
    environment: str = Field("default", env="JUA_ENVIRONMENT")
    secrets_path: str | None = Field(None, env="JUA_SECRETS_PATH")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    def __init__(
        self,
        api_key_id: str | None = None,
        api_key_secret: str | None = None,
        environment: str = "default",
        secrets_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if api_key_id is not None:
            self.api_key_id = api_key_id
        if api_key_secret is not None:
            self.api_key_secret = api_key_secret
        if environment is not None:
            self.environment = environment
        if secrets_path is not None:
            self.secrets_path = secrets_path

        # If credentials are not set via env vars or .env, try loading from JSON file
        if not self.api_key_id or not self.api_key_secret:
            self._load_from_json_file()

    def _load_from_json_file(self) -> None:
        """Load credentials from JSON file if needed."""
        # Determine the secrets file path
        if self.secrets_path:
            file_path = Path(self.secrets_path)
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
        """Check if authentication credentials are properly set."""
        return bool(self.api_key_id and self.api_key_secret)
