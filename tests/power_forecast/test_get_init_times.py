from datetime import datetime, timezone

from jua import JuaClient
from jua.power_forecast.power_forecast import InitTimeInfo


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _patch_api(monkeypatch, pf, payload):
    """Capture the params passed to the init-times endpoint."""
    captured: dict = {}

    def fake_get(path, params=None, requires_auth=True):
        captured["path"] = path
        captured["params"] = params
        return _FakeResponse(payload)

    monkeypatch.setattr(pf._api, "get", fake_get)
    return captured


_PAYLOAD = {
    "init_times": [
        {"init_time": "2025-01-02T00:00:00Z", "max_prediction_timedelta": 2400},
        {"init_time": "2025-01-01T00:00:00Z", "max_prediction_timedelta": 2400},
    ]
}


def test_count_mode_sends_limit(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast
    captured = _patch_api(monkeypatch, pf, _PAYLOAD)

    result = pf.get_init_times(zone_key="DE", psr_type="Solar", limit=42)

    assert captured["params"]["limit"] == 42
    assert "start_time" not in captured["params"]
    assert "end_time" not in captured["params"]
    assert captured["params"]["zone_key"] == "DE"
    assert captured["params"]["psr_type"] == ["Solar"]
    assert all(isinstance(it, InitTimeInfo) for it in result)


def test_time_window_mode_omits_limit_and_sends_range(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast
    captured = _patch_api(monkeypatch, pf, _PAYLOAD)

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 2, 1, tzinfo=timezone.utc)
    pf.get_init_times(zone_key="DE", start_time=start, end_time=end)

    params = captured["params"]
    # limit must be omitted so the server returns the full window (>1000 allowed)
    assert "limit" not in params
    assert params["start_time"] == start.isoformat()
    assert params["end_time"] == end.isoformat()


def test_time_window_mode_accepts_only_start(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast
    captured = _patch_api(monkeypatch, pf, _PAYLOAD)

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    pf.get_init_times(zone_key="DE", start_time=start)

    params = captured["params"]
    assert "limit" not in params
    assert params["start_time"] == start.isoformat()
    assert "end_time" not in params
