from pathlib import Path
from typing import Generator
from unittest.mock import mock_open, patch

import pytest
from aiohttp import BasicAuth
from pytest import MonkeyPatch

from jua.settings.authentication import AuthenticationSettings


@pytest.fixture
def temp_env(monkeypatch: MonkeyPatch) -> Generator[MonkeyPatch, None, None]:
    """Fixture to ensure environment variables are cleared after tests."""
    # Clear relevant environment variables before test
    monkeypatch.delenv("JUA_API_KEY_ID", raising=False)
    monkeypatch.delenv("JUA_API_KEY_SECRET", raising=False)
    yield monkeypatch


@pytest.fixture
def mock_home_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Fixture to mock the home directory for testing file-based credentials."""
    with patch("pathlib.Path.home", return_value=tmp_path):
        yield tmp_path


class TestAuthenticationSettings:
    def test_init_with_explicit_credentials(self) -> None:
        """Test initialization with explicit credentials."""
        auth = AuthenticationSettings(
            api_key_id="test_id", api_key_secret="test_secret"
        )

        assert auth.api_key_id == "test_id"
        assert auth.api_key_secret == "test_secret"
        assert auth.environment == "default"
        assert auth.is_authenticated is True

    def test_load_from_env_vars(self, temp_env: MonkeyPatch) -> None:
        """Test loading credentials from environment variables."""
        temp_env.setenv("JUA_API_KEY_ID", "env_id")
        temp_env.setenv("JUA_API_KEY_SECRET", "env_secret")

        auth = AuthenticationSettings()

        assert auth.api_key_id == "env_id"
        assert auth.api_key_secret == "env_secret"
        assert auth.is_authenticated is True

    def test_custom_environment(self) -> None:
        """Test setting a custom environment."""
        auth = AuthenticationSettings(
            api_key_id="test_id", api_key_secret="test_secret", environment="production"
        )

        assert auth.environment == "production"

    def test_load_from_json_file_with_explicit_path(self) -> None:
        """Test loading credentials from a JSON file at an explicit path."""
        json_content = '{"id": "file_id", "secret": "file_secret"}'

        with patch("builtins.open", mock_open(read_data=json_content)):
            with patch("pathlib.Path.exists", return_value=True):
                auth = AuthenticationSettings(api_key_path="/path/to/secrets.json")

                assert auth.api_key_id == "file_id"
                assert auth.api_key_secret == "file_secret"
                assert auth.is_authenticated is True

    def test_load_from_default_json_location(self, mock_home_dir: Path) -> None:
        """Test loading credentials from the default JSON file location."""
        # Set up default credentials file
        default_dir = mock_home_dir / ".jua" / "default"
        default_dir.mkdir(parents=True)
        creds_file = default_dir / "api-key.json"
        creds_file.write_text('{"id": "default_id", "secret": "default_secret"}')

        auth = AuthenticationSettings()

        assert auth.api_key_id == "default_id"
        assert auth.api_key_secret == "default_secret"
        assert auth.is_authenticated is True

    def test_environment_specific_json_location(self, mock_home_dir: Path) -> None:
        """Test loading credentials from an environment-specific JSON file location."""
        # Set up environment-specific credentials file
        env_dir = mock_home_dir / ".jua" / "staging"
        env_dir.mkdir(parents=True)
        creds_file = env_dir / "api-key.json"
        creds_file.write_text('{"id": "staging_id", "secret": "staging_secret"}')

        auth = AuthenticationSettings(environment="staging")

        assert auth.api_key_id == "staging_id"
        assert auth.api_key_secret == "staging_secret"
        assert auth.is_authenticated is True

    def test_missing_credentials_file(self, mock_home_dir: Path) -> None:
        """Test behavior when credentials file is missing."""
        auth = AuthenticationSettings()

        assert auth.api_key_id is None
        assert auth.api_key_secret is None
        assert auth.is_authenticated is False

    def test_invalid_json_file(self, mock_home_dir: Path) -> None:
        """Test handling of invalid JSON in credentials file."""
        # Set up invalid JSON file
        default_dir = mock_home_dir / ".jua" / "default"
        default_dir.mkdir(parents=True)
        creds_file = default_dir / "api-key.json"
        creds_file.write_text("invalid json content")

        auth = AuthenticationSettings()

        assert auth.api_key_id is None
        assert auth.api_key_secret is None
        assert auth.is_authenticated is False

    def test_get_basic_auth(self) -> None:
        """Test the get_basic_auth method."""
        auth = AuthenticationSettings(
            api_key_id="test_id", api_key_secret="test_secret"
        )

        basic_auth = auth.get_basic_auth()

        assert isinstance(basic_auth, BasicAuth)
        assert basic_auth.login == "test_id"
        assert basic_auth.password == "test_secret"

    def test_credential_loading_priority(
        self, temp_env: MonkeyPatch, mock_home_dir: Path
    ) -> None:
        """Test credential loading priority (explicit over env over file)."""
        # Set up all three sources with different values
        temp_env.setenv("JUA_API_KEY_ID", "env_id")
        temp_env.setenv("JUA_API_KEY_SECRET", "env_secret")

        default_dir = mock_home_dir / ".jua" / "default"
        default_dir.mkdir(parents=True)
        creds_file = default_dir / "api-key.json"
        creds_file.write_text('{"id": "file_id", "secret": "file_secret"}')

        # Explicit parameters should take precedence
        auth1 = AuthenticationSettings(
            api_key_id="explicit_id", api_key_secret="explicit_secret"
        )
        assert auth1.api_key_id == "explicit_id"
        assert auth1.api_key_secret == "explicit_secret"

        # Env vars should take precedence over file
        auth2 = AuthenticationSettings()
        assert auth2.api_key_id == "env_id"
        assert auth2.api_key_secret == "env_secret"

        # Test file priority by removing env vars
        temp_env.delenv("JUA_API_KEY_ID")
        temp_env.delenv("JUA_API_KEY_SECRET")

        auth3 = AuthenticationSettings()
        assert auth3.api_key_id == "file_id"
        assert auth3.api_key_secret == "file_secret"

    def test_secrets_path_overrides_default_path(self, mock_home_dir: Path) -> None:
        """Test that explicit secrets_path takes precedence over default path."""
        # Set up default credentials file with one set of credentials
        default_dir = mock_home_dir / ".jua" / "default"
        default_dir.mkdir(parents=True)
        default_creds_file = default_dir / "api-key.json"
        default_creds_file.write_text(
            '{"id": "default_id", "secret": "default_secret"}'
        )

        # Set up custom credentials file with different credentials
        custom_dir = mock_home_dir / "custom"
        custom_dir.mkdir(parents=True)
        custom_creds_file = custom_dir / "custom-creds.json"
        custom_creds_file.write_text('{"id": "custom_id", "secret": "custom_secret"}')

        # When using secrets_path, it should override the default path
        auth = AuthenticationSettings(api_key_path=str(custom_creds_file))

        assert auth.api_key_id == "custom_id"
        assert auth.api_key_secret == "custom_secret"
        assert auth.is_authenticated is True

    def test_constructor_overrides_env_vars(self, temp_env: MonkeyPatch) -> None:
        """Test that constructor arguments override environment variables."""
        temp_env.setenv("JUA_API_KEY_ID", "env_id")
        temp_env.setenv("JUA_API_KEY_SECRET", "env_secret")

        # Constructor args should take precedence over env vars
        auth = AuthenticationSettings(
            api_key_id="constructor_id", api_key_secret="constructor_secret"
        )

        assert auth.api_key_id == "constructor_id"
        assert auth.api_key_secret == "constructor_secret"
        assert auth.is_authenticated is True
