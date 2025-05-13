import os
from pathlib import Path
from typing import Generator

import pytest
from pytest import MonkeyPatch


@pytest.fixture
def temp_env(monkeypatch: MonkeyPatch) -> Generator[MonkeyPatch, None, None]:
    """Fixture to ensure environment variables are cleared after tests.

    Clears any JUA_ prefixed environment variables before the test and provides
    the monkeypatch object to set and clear environment variables during tests.

    Yields:
        MonkeyPatch: The pytest monkeypatch object for setting environment variables.
    """
    # Get all environment variables and clear JUA_ ones
    for env_var in list(os.environ.keys()):
        if env_var.startswith("JUA_"):
            monkeypatch.delenv(env_var, raising=False)

    yield monkeypatch


@pytest.fixture
def mock_home_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Fixture to mock the home directory for testing file-based settings.

    Creates a temporary directory and makes Path.home() return it during tests.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Yields:
        Path: The temporary directory path that's being used as the mock home directory.
    """
    from unittest.mock import patch

    with patch("pathlib.Path.home", return_value=tmp_path):
        yield tmp_path
