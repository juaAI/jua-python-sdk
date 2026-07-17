"""Unit tests for power-forecast serving version selection."""

from __future__ import annotations

import pytest

from jua import JuaClient
from jua.power_forecast.power_forecast import VersionInfo


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_get_versions_parses_catalog(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast
    captured: dict = {}

    def fake_get(path, params=None, requires_auth=True):
        captured["path"] = path
        captured["params"] = params
        captured["requires_auth"] = requires_auth
        return _FakeResponse(
            {
                "versions": [
                    {
                        "model_version": "586rhosh",
                        "zone_key": "DE",
                        "psr_type": "Solar",
                        "is_stable": True,
                        "is_latest": False,
                        "earliest_init_time": "2025-05-31T23:45:00Z",
                        "latest_init_time": "2026-07-11T08:30:00Z",
                    },
                    {
                        "model_version": "rv7orbtm",
                        "zone_key": "DE",
                        "psr_type": "Solar",
                        "is_stable": False,
                        "is_latest": True,
                        "earliest_init_time": "2026-07-01T00:00:00Z",
                        "latest_init_time": "2026-07-11T08:30:00Z",
                    },
                ]
            }
        )

    monkeypatch.setattr(pf._api, "get", fake_get)

    rows = pf.get_versions(zone_key="DE", psr_type="Solar")

    assert captured["path"] == "power-forecast/versions"
    assert captured["requires_auth"] is True
    assert captured["params"]["zone_key"] == "DE"
    assert captured["params"]["psr_type"] == ["Solar"]
    assert len(rows) == 2
    assert all(isinstance(row, VersionInfo) for row in rows)
    assert rows[0].is_stable is True
    assert rows[0].is_latest is False
    assert rows[1].model_version == "rv7orbtm"
    assert rows[1].is_latest is True


def test_get_init_times_forwards_version(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast
    captured: dict = {}

    def fake_get(path, params=None, requires_auth=True):
        captured["params"] = params
        return _FakeResponse(
            {
                "init_times": [
                    {
                        "init_time": "2026-07-16T00:00:00Z",
                        "max_prediction_timedelta": 2400,
                    }
                ]
            }
        )

    monkeypatch.setattr(pf._api, "get", fake_get)

    pf.get_init_times(zone_key="DE", psr_type="Solar", version="latest")

    assert captured["params"]["version"] == "latest"


def test_get_data_body_includes_version_and_pins(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast
    captured: dict = {}

    def fake_post(path, data=None, requires_auth=True):
        captured["path"] = path
        captured["data"] = data
        return _FakeResponse(
            {
                "zone_key": [],
                "psr_type": [],
                "init_time": [],
                "time": [],
                "value": [],
            }
        )

    monkeypatch.setattr(pf._api, "post", fake_post)

    pf.get_data(
        zone_keys=["DE"],
        psr_types=["Solar"],
        init_time="2026-07-16T00:00:00+00:00",
        version="latest",
        version_pins=[
            {"zone_key": "DE", "psr_type": "Solar", "version": "rv7orbtm"},
        ],
    )

    assert captured["path"] == "power-forecast/data"
    assert captured["data"]["version"] == "latest"
    assert captured["data"]["version_pins"] == [
        {"zone_key": "DE", "psr_type": "Solar", "version": "rv7orbtm"}
    ]


def test_get_data_rejects_internal_channel_names():
    client = JuaClient()
    pf = client.power_forecast

    with pytest.raises(ValueError, match="stable.*latest"):
        pf.get_data(
            zone_keys=["DE"],
            psr_types=["Solar"],
            init_time="2026-07-16T00:00:00+00:00",
            version="live",
        )


def test_resolve_init_time_passes_version_to_init_times(monkeypatch):
    client = JuaClient()
    pf = client.power_forecast
    captured: dict = {}

    def fake_get(path, params=None, requires_auth=True):
        captured["params"] = params
        return _FakeResponse(
            {
                "init_times": [
                    {
                        "init_time": "2026-07-16T12:00:00Z",
                        "max_prediction_timedelta": 2400,
                    }
                ]
            }
        )

    def fake_post(path, data=None, requires_auth=True):
        captured["data"] = data
        return _FakeResponse(
            {
                "zone_key": [],
                "psr_type": [],
                "init_time": [],
                "time": [],
                "value": [],
            }
        )

    monkeypatch.setattr(pf._api, "get", fake_get)
    monkeypatch.setattr(pf._api, "post", fake_post)

    pf.get_data(
        zone_keys=["DE"],
        psr_types=["Solar"],
        init_time="latest",
        version="latest",
    )

    assert captured["params"]["version"] == "latest"
    assert captured["data"]["version"] == "latest"
    # Relative token resolved to concrete ISO init from the versioned listing.
    assert captured["data"]["init_time"].startswith("2026-07-16T12:00:00")
