"""Unit tests for MarketData routing, normalization, and DST-safe parsing."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from jua import JuaClient
from jua.market_data._frame import parse_time


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _generation_payload():
    """ENTSOE generation_actual rows for Solar + Wind Onshore + Wind Offshore."""
    times = ["2025-12-01T00:00:00Z", "2025-12-01T01:00:00Z"]
    rows_time = []
    rows_psr = []
    rows_value = []
    # solar=10/20, onshore=100/110, offshore=50/60
    values = {
        "Solar": [10.0, 20.0],
        "Wind Onshore": [100.0, 110.0],
        "Wind Offshore": [50.0, 60.0],
    }
    for i, t in enumerate(times):
        for psr in ("Solar", "Wind Onshore", "Wind Offshore"):
            rows_time.append(t)
            rows_psr.append(psr)
            rows_value.append(values[psr][i])
    return {
        "time": rows_time,
        "variable_name": ["generation_actual"] * len(rows_time),
        "zone_key": ["DE_LU"] * len(rows_time),
        "psr_type": rows_psr,
        "value": rows_value,
        "unit": ["MW"] * len(rows_time),
    }


def _prices_payload(zone_key: str):
    times = ["2025-12-01T00:00:00Z", "2025-12-01T01:00:00Z"]
    return {
        "time": times,
        "variable_name": ["day_ahead_prices"] * 2,
        "zone_key": [zone_key] * 2,
        "psr_type": ["", ""],
        "value": [85.5, 82.3],
        "unit": ["EUR/MWh", "EUR/MWh"],
    }


def _imbalance_payload(zone_key="DE"):
    """ENTSOE imbalance_prices rows: two timestamps, each with Long + Short.

    Long != Short (dual-pricing-like) so summing would be detectable, and t0
    carries a duplicate "Long" revision row (identical value) so de-duplication
    can be distinguished from summing.
    """
    return {
        "time": [
            "2025-12-01T00:00:00Z",  # Long
            "2025-12-01T00:00:00Z",  # Long (revision duplicate, same value)
            "2025-12-01T00:00:00Z",  # Short
            "2025-12-01T01:00:00Z",  # Long
            "2025-12-01T01:00:00Z",  # Short
        ],
        "variable_name": ["imbalance_prices"] * 5,
        "zone_key": [zone_key] * 5,
        "psr_type": [""] * 5,
        "other_type": ["Long", "Long", "Short", "Long", "Short"],
        "value": [10.0, 10.0, 99.0, -5.0, 20.0],
        "unit": ["EUR/MWh"] * 5,
    }


def _uk_payload():
    times = ["2025-12-01T00:00:00Z", "2025-12-01T01:00:00Z"]
    return {
        "time": times * 2,
        "variable_name": ["solar", "solar", "wind", "wind"],
        "value": [5.0, 8.0, 200.0, 210.0],
        "unit": ["MW"] * 4,
    }


class _Recorder:
    """Captures posts and returns canned payloads keyed by native variable."""

    def __init__(self):
        self.calls = []

    def make(self, payload_for):
        def fake_post(path, data=None, requires_auth=True, **kwargs):
            self.calls.append({"path": path, "body": data})
            return _FakeResponse(payload_for(data))

        return fake_post


def _patch_entsoe(monkeypatch, md, payload_for):
    rec = _Recorder()
    monkeypatch.setattr(md._entsoe._api, "post", rec.make(payload_for))
    return rec


def _patch_uk(monkeypatch, md, payload_for):
    rec = _Recorder()
    monkeypatch.setattr(md._uk_power._api, "post", rec.make(payload_for))
    return rec


_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
_END = datetime(2025, 12, 2, tzinfo=timezone.utc)


def test_de_routes_to_entsoe_with_mapped_zone(monkeypatch):
    md = JuaClient().market_data

    def payload_for(body):
        return _generation_payload()

    rec = _patch_entsoe(monkeypatch, md, payload_for)

    df = md.get_data(
        market_zone="DE",
        variables=["solar", "wind"],
        start_time=_START,
        end_time=_END,
    )

    # Solar + wind both come from generation_actual -> single request.
    assert len(rec.calls) == 1
    body = rec.calls[0]["body"]
    assert rec.calls[0]["path"] == "entsoe/data"
    assert body["zone_keys"] == ["DE_LU"]
    assert body["variables"] == ["generation_actual"]
    assert set(body["psr_types"]) == {"Solar", "Wind Onshore", "Wind Offshore"}

    assert df.columns.tolist() == ["time", "market_zone", "variable", "value", "unit"]
    assert set(df["variable"]) == {"solar", "wind"}
    assert set(df["market_zone"]) == {"DE"}


def test_entsoe_wind_sums_onshore_and_offshore(monkeypatch):
    md = JuaClient().market_data
    _patch_entsoe(monkeypatch, md, lambda body: _generation_payload())

    df = md.get_data(
        market_zone="DE",
        variables=["wind"],
        start_time=_START,
        end_time=_END,
    )

    wind = df[df["variable"] == "wind"].sort_values("time")
    # onshore + offshore: 100+50=150, 110+60=170
    assert wind["value"].tolist() == [150.0, 170.0]


def test_gb_renewables_route_to_uk_power(monkeypatch):
    md = JuaClient().market_data

    entsoe_rec = _patch_entsoe(monkeypatch, md, lambda body: _prices_payload("GB"))
    uk_rec = _patch_uk(monkeypatch, md, lambda body: _uk_payload())

    df = md.get_data(
        market_zone="GB",
        variables=["solar", "wind"],
        start_time=_START,
        end_time=_END,
    )

    # GB renewables come from uk-power; ENTSOE is not involved for GB.
    assert len(uk_rec.calls) == 1
    assert uk_rec.calls[0]["path"] == "uk-power/data"
    assert set(uk_rec.calls[0]["body"]["variables"]) == {"solar", "wind"}
    assert len(entsoe_rec.calls) == 0

    assert set(df["variable"]) == {"solar", "wind"}
    assert set(df["market_zone"]) == {"GB"}


def test_gb_prices_not_supported_raises(monkeypatch):
    md = JuaClient().market_data
    entsoe_rec = _patch_entsoe(monkeypatch, md, lambda body: _prices_payload("GB"))
    uk_rec = _patch_uk(monkeypatch, md, lambda body: _uk_payload())

    # GB prices/load_forecast are not served: fail clearly, hit no backend.
    for variable in [
        "day_ahead_prices",
        "imbalance_price_long",
        "imbalance_price_short",
        "load_forecast",
    ]:
        with pytest.raises(ValueError, match="not supported for zone 'GB'"):
            md.get_data(
                market_zone="GB",
                variables=[variable],
                start_time=_START,
                end_time=_END,
            )

    assert len(entsoe_rec.calls) == 0
    assert len(uk_rec.calls) == 0


def test_imbalance_long_filters_dedups_and_uses_de_zone(monkeypatch):
    md = JuaClient().market_data
    rec = _patch_entsoe(monkeypatch, md, lambda body: _imbalance_payload("DE"))

    df = md.get_data(
        market_zone="DE",
        variables=["imbalance_price_long"],
        start_time=_START,
        end_time=_END,
    )

    # DE imbalance is published under control area "DE", not the DE_LU bidding
    # zone used for everything else.
    body = rec.calls[0]["body"]
    assert body["zone_keys"] == ["DE"]
    assert body["variables"] == ["imbalance_prices"]

    out = df.sort_values("time")
    # Long only, with the duplicate t0 row collapsed (10.0, NOT 20.0), and the
    # Short rows (99.0/20.0) excluded entirely. Never summed with Short.
    assert out["value"].tolist() == [10.0, -5.0]
    assert set(out["variable"]) == {"imbalance_price_long"}
    assert set(out["unit"]) == {"EUR/MWh"}


def test_imbalance_short_returns_short_side(monkeypatch):
    md = JuaClient().market_data
    _patch_entsoe(monkeypatch, md, lambda body: _imbalance_payload("DE"))

    df = md.get_data(
        market_zone="DE",
        variables=["imbalance_price_short"],
        start_time=_START,
        end_time=_END,
    )
    out = df.sort_values("time")
    assert out["value"].tolist() == [99.0, 20.0]


def test_imbalance_long_and_short_single_fetch_split(monkeypatch):
    md = JuaClient().market_data
    rec = _patch_entsoe(monkeypatch, md, lambda body: _imbalance_payload("DE"))

    df = md.get_data(
        market_zone="DE",
        variables=["imbalance_price_long", "imbalance_price_short"],
        start_time=_START,
        end_time=_END,
    )

    # Both directions share one native variable + zone, so a single request is
    # issued and split locally into two distinct series.
    assert len(rec.calls) == 1
    long = df[df["variable"] == "imbalance_price_long"].sort_values("time")
    short = df[df["variable"] == "imbalance_price_short"].sort_values("time")
    assert long["value"].tolist() == [10.0, -5.0]
    assert short["value"].tolist() == [99.0, 20.0]
    # The two series must not be identical (proves no accidental collapse).
    assert long["value"].tolist() != short["value"].tolist()


def test_multi_zone_fan_out_and_concat(monkeypatch):
    md = JuaClient().market_data
    _patch_entsoe(monkeypatch, md, lambda body: _generation_payload())
    _patch_uk(monkeypatch, md, lambda body: _uk_payload())

    df = md.get_data(
        market_zone=["DE", "GB"],
        variables=["solar"],
        start_time=_START,
        end_time=_END,
    )

    assert set(df["market_zone"]) == {"DE", "GB"}
    assert set(df["variable"]) == {"solar"}


def test_unknown_variable_raises(monkeypatch):
    md = JuaClient().market_data
    # No backend should be called; resolution fails first.
    with pytest.raises(ValueError, match="Unknown market variable"):
        md.get_data(
            market_zone="DE",
            variables=["nonexistent_variable"],
            start_time=_START,
            end_time=_END,
        )


def test_start_time_serialized_isoformat(monkeypatch):
    md = JuaClient().market_data
    rec = _patch_entsoe(monkeypatch, md, lambda body: _generation_payload())

    md.get_data(
        market_zone="DE",
        variables=["solar"],
        start_time=_START,
        end_time=_END,
    )
    assert rec.calls[0]["body"]["start_time"] == _START.isoformat()
    assert rec.calls[0]["body"]["end_time"] == _END.isoformat()


def test_empty_response_returns_empty_unified_frame(monkeypatch):
    md = JuaClient().market_data
    _patch_entsoe(monkeypatch, md, lambda body: {})

    df = md.get_data(
        market_zone="DE",
        variables=["solar"],
        start_time=_START,
        end_time=_END,
    )
    assert df.empty
    assert df.columns.tolist() == ["time", "market_zone", "variable", "value", "unit"]


# ---------------------------------------------------------------------------
# DST-safe time parsing
# ---------------------------------------------------------------------------
def test_parse_time_handles_fall_back_day():
    """A 25-hour fall-back day must parse to a full-length tz-aware column."""
    tz = "Europe/Berlin"
    # 2025-10-26 is the European fall-back day (25 hours): 25 hourly stamps
    # starting at local midnight, crossing the +02:00 -> +01:00 transition.
    local_index = pd.date_range("2025-10-26 00:00", periods=25, freq="h", tz=tz)
    assert len(local_index) == 25  # sanity: the 25-hour day

    # Server emits ISO strings whose offset changes across the transition.
    iso_strings = [ts.isoformat() for ts in local_index]
    assert any("+02:00" in s for s in iso_strings)
    assert any("+01:00" in s for s in iso_strings)

    df = pd.DataFrame({"time": iso_strings, "value": range(25)})
    parsed = parse_time(df, time_zone=tz)

    assert len(parsed) == 25
    assert parsed["time"].notna().all()
    assert isinstance(parsed["time"].dtype, pd.DatetimeTZDtype)
    assert str(parsed["time"].dt.tz) == tz


def test_parse_time_defaults_to_utc():
    df = pd.DataFrame(
        {"time": ["2025-12-01T00:00:00Z", "2025-12-01T01:00:00Z"], "value": [1, 2]}
    )
    parsed = parse_time(df, time_zone=None)
    assert isinstance(parsed["time"].dtype, pd.DatetimeTZDtype)
    assert str(parsed["time"].dt.tz) == "UTC"
