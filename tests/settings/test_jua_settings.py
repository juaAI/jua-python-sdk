from pytest import MonkeyPatch

from jua.settings.authentication import AuthenticationSettings
from jua.settings.jua_settings import JuaSettings


class TestJuaSettings:
    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        settings = JuaSettings()

        assert settings.api_url == "https://api.jua.sh"
        assert settings.api_version == "v1"
        assert settings.data_base_url == "https://data.jua.sh"
        assert settings.print_progress is True
        assert isinstance(settings.auth, AuthenticationSettings)

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        settings = JuaSettings(
            api_url="https://custom-api.example.com",
            api_version="v2",
            data_base_url="https://custom-data.example.com",
            print_progress=False,
        )

        assert settings.api_url == "https://custom-api.example.com"
        assert settings.api_version == "v2"
        assert settings.data_base_url == "https://custom-data.example.com"
        assert settings.print_progress is False

    def test_load_from_env_vars(self, temp_env: MonkeyPatch) -> None:
        """Test loading settings from environment variables."""
        temp_env.setenv("JUA_API_URL", "https://env-api.example.com")
        temp_env.setenv("JUA_API_VERSION", "v3")
        temp_env.setenv("JUA_DATA_BASE_URL", "https://env-data.example.com")
        temp_env.setenv("JUA_PRINT_PROGRESS", "false")

        settings = JuaSettings()

        assert settings.api_url == "https://env-api.example.com"
        assert settings.api_version == "v3"
        assert settings.data_base_url == "https://env-data.example.com"
        assert settings.print_progress is False

    def test_custom_auth_settings(self) -> None:
        """Test providing custom authentication settings."""
        auth = AuthenticationSettings(
            api_key_id="custom_id",
            api_key_secret="custom_secret",
            environment="production",
        )

        settings = JuaSettings(auth=auth)

        assert settings.auth.api_key_id == "custom_id"
        assert settings.auth.api_key_secret == "custom_secret"
        assert settings.auth.environment == "production"

    def test_should_print_progress_with_default(self) -> None:
        """Test should_print_progress method with default settings."""
        settings = JuaSettings(print_progress=True)

        assert settings.should_print_progress() is True

        settings = JuaSettings(print_progress=False)

        assert settings.should_print_progress() is False

    def test_should_print_progress_with_override(self) -> None:
        """Test should_print_progress method with override parameter."""
        settings = JuaSettings(print_progress=True)

        assert settings.should_print_progress(print_progress=False) is False

        settings = JuaSettings(print_progress=False)

        assert settings.should_print_progress(print_progress=True) is True

    def test_constructor_overrides_env_vars(self, temp_env: MonkeyPatch) -> None:
        """Test that constructor arguments override environment variables."""
        temp_env.setenv("JUA_API_URL", "https://env-api.example.com")
        temp_env.setenv("JUA_PRINT_PROGRESS", "false")

        settings = JuaSettings(
            api_url="https://constructor-api.example.com", print_progress=True
        )

        assert settings.api_url == "https://constructor-api.example.com"
        assert settings.print_progress is True
