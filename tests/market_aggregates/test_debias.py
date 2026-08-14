from datetime import datetime, timezone

import pytest

from jua import JuaClient
from jua.market_aggregates import ModelRuns
from jua.weather import Models


class _FakeResponse:
    def json(self):
        return {}


@pytest.mark.parametrize(("debias", "expected"), [(False, None), (True, True)])
def test_compare_runs_mw_forwards_opt_in_debias(monkeypatch, debias, expected):
    market = JuaClient().market_aggregates.get_market("DE")
    captured: dict = {}

    monkeypatch.setattr(
        market,
        "_resolve_init_times_for_model",
        lambda model, init_times: [
            datetime(2026, 8, 1, tzinfo=timezone.utc),
        ],
    )

    def fake_get(path, params=None, requires_auth=True):
        captured["path"] = path
        captured["params"] = params
        return _FakeResponse()

    monkeypatch.setattr(market._query_engine_api, "get", fake_get)

    market.compare_runs_mw(
        weighting="wind_capacity",
        model_runs=[ModelRuns(Models.EPT2, 0)],
        debias=debias,
    )

    assert captured["path"] == "forecast/market-aggregate"
    assert captured["params"].get("debias") is expected
