"""Tests for ForecastWatcher polling logic."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from jua.client import JuaClient
from jua.weather import ForecastWatcher, Models
from jua.weather._types.query_response_types import LatestForecastInfo


@pytest.fixture
def mock_client():
    client = JuaClient()
    client.settings.auth.api_key_id = "test_key_id"
    client.settings.auth.api_key_secret = "test_key_secret"
    return client


def _make_info(init_time: datetime, prediction_timedelta: int = 48):
    return LatestForecastInfo(
        init_time=init_time,
        prediction_timedelta=prediction_timedelta,
    )


class TestForecastWatcherInit:
    def test_rejects_empty_models(self, mock_client):
        with pytest.raises(ValueError, match="At least one model"):
            ForecastWatcher(
                client=mock_client,
                models=[],
                on_new_forecast=lambda *a: None,
            )

    def test_rejects_invalid_interval(self, mock_client):
        with pytest.raises(ValueError, match="interval_seconds must be >= 1"):
            ForecastWatcher(
                client=mock_client,
                models=[Models.EPT2],
                on_new_forecast=lambda *a: None,
                interval_seconds=0,
            )


class TestCheckOnce:
    def test_initial_snapshot_does_not_fire_callback(self, mock_client):
        callback = MagicMock()
        t1 = datetime(2025, 7, 1, 0, tzinfo=timezone.utc)
        info = _make_info(t1)

        watcher = ForecastWatcher(
            client=mock_client,
            models=[Models.EPT2],
            on_new_forecast=callback,
        )

        with patch.object(
            watcher._sdk_models["ept2"],
            "get_latest_init_time",
            return_value=info,
        ):
            result = watcher.check_once()

        callback.assert_not_called()
        assert result == []
        assert watcher.latest_init_times["ept2"] == t1

    def test_fires_callback_on_new_init_time(self, mock_client):
        callback = MagicMock()
        t1 = datetime(2025, 7, 1, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 7, 1, 6, tzinfo=timezone.utc)
        info1 = _make_info(t1)
        info2 = _make_info(t2)

        watcher = ForecastWatcher(
            client=mock_client,
            models=[Models.EPT2],
            on_new_forecast=callback,
        )

        model = watcher._sdk_models["ept2"]

        with patch.object(model, "get_latest_init_time", return_value=info1):
            watcher.check_once()

        with patch.object(model, "get_latest_init_time", return_value=info2):
            result = watcher.check_once()

        callback.assert_called_once_with("ept2", info2)
        assert len(result) == 1
        assert result[0] == ("ept2", info2)
        assert watcher.latest_init_times["ept2"] == t2

    def test_no_callback_when_unchanged(self, mock_client):
        callback = MagicMock()
        t1 = datetime(2025, 7, 1, 0, tzinfo=timezone.utc)
        info = _make_info(t1)

        watcher = ForecastWatcher(
            client=mock_client,
            models=[Models.EPT2],
            on_new_forecast=callback,
        )
        model = watcher._sdk_models["ept2"]

        with patch.object(model, "get_latest_init_time", return_value=info):
            watcher.check_once()
            watcher.check_once()

        callback.assert_not_called()

    def test_handles_api_error_gracefully(self, mock_client):
        callback = MagicMock()

        watcher = ForecastWatcher(
            client=mock_client,
            models=[Models.EPT2],
            on_new_forecast=callback,
        )
        model = watcher._sdk_models["ept2"]

        with patch.object(
            model,
            "get_latest_init_time",
            side_effect=ConnectionError("network down"),
        ):
            result = watcher.check_once()

        callback.assert_not_called()
        assert result == []

    def test_multiple_models(self, mock_client):
        callback = MagicMock()
        t1 = datetime(2025, 7, 1, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 7, 1, 6, tzinfo=timezone.utc)

        watcher = ForecastWatcher(
            client=mock_client,
            models=[Models.EPT2, Models.EPT2_HRRR],
            on_new_forecast=callback,
        )

        model_ept2 = watcher._sdk_models["ept2"]
        model_hrrr = watcher._sdk_models["ept2_hrrr"]

        with (
            patch.object(
                model_ept2, "get_latest_init_time", return_value=_make_info(t1)
            ),
            patch.object(
                model_hrrr, "get_latest_init_time", return_value=_make_info(t1)
            ),
        ):
            watcher.check_once()

        # Only EPT2 advances
        with (
            patch.object(
                model_ept2, "get_latest_init_time", return_value=_make_info(t2)
            ),
            patch.object(
                model_hrrr, "get_latest_init_time", return_value=_make_info(t1)
            ),
        ):
            result = watcher.check_once()

        assert len(result) == 1
        assert result[0][0] == "ept2"
        callback.assert_called_once()


class TestWatch:
    def test_max_cycles(self, mock_client):
        callback = MagicMock()
        t1 = datetime(2025, 7, 1, 0, tzinfo=timezone.utc)

        watcher = ForecastWatcher(
            client=mock_client,
            models=[Models.EPT2],
            on_new_forecast=callback,
            interval_seconds=1,
        )
        model = watcher._sdk_models["ept2"]

        with patch.object(
            model, "get_latest_init_time", return_value=_make_info(t1)
        ) as mock_get:
            watcher.watch(max_cycles=3)
            assert mock_get.call_count == 3

    def test_stop_interrupts_watch(self, mock_client):
        """stop() causes watch() to exit after the current cycle."""
        import threading

        callback = MagicMock()
        t1 = datetime(2025, 7, 1, 0, tzinfo=timezone.utc)

        watcher = ForecastWatcher(
            client=mock_client,
            models=[Models.EPT2],
            on_new_forecast=callback,
            interval_seconds=60,
        )
        model = watcher._sdk_models["ept2"]

        with patch.object(model, "get_latest_init_time", return_value=_make_info(t1)):
            thread = threading.Thread(target=watcher.watch)
            thread.start()
            # Give it a moment to enter the loop
            import time

            time.sleep(0.2)
            watcher.stop()
            thread.join(timeout=5)

        assert not thread.is_alive()


class TestMinPredictionTimedelta:
    def test_passes_min_prediction_timedelta(self, mock_client):
        callback = MagicMock()
        t1 = datetime(2025, 7, 1, 0, tzinfo=timezone.utc)

        watcher = ForecastWatcher(
            client=mock_client,
            models=[Models.EPT2],
            on_new_forecast=callback,
            min_prediction_timedelta=48,
        )
        model = watcher._sdk_models["ept2"]

        with patch.object(
            model, "get_latest_init_time", return_value=_make_info(t1, 72)
        ) as mock_get:
            watcher.check_once()

        mock_get.assert_called_once_with(min_prediction_timedelta=48)
