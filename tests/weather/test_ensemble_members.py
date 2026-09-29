"""Model.get_forecasts(include_ensemble_members=True) with mocked API responses."""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from jua.client import JuaClient
from jua.errors.api_errors import UnauthorizedError
from jua.types.geo import LatLon
from jua.weather import Model, Models, Variables
from tests.weather.utils import create_mock_arrow_response

MEMBERS = ["ept2_e_0", "ept2_e_1", "ept2_e_2", "ept2_e_10", "ept2_e_20"]


@pytest.fixture
def mock_client():
    client = JuaClient()
    client.settings.auth.api_key_id = "test_key_id"
    client.settings.auth.api_key_secret = "test_key_secret"
    return client


def _member_frame(init_time: datetime, num_hours: int) -> pd.DataFrame:
    rows = []
    for member in MEMBERS:
        index = int(member.rsplit("_", 1)[1])
        for hour in range(num_hours + 1):
            rows.append(
                {
                    "init_time": init_time,
                    "model": "ept2_e",
                    "prediction_timedelta": hour * 60,
                    "point": 0,
                    "latitude": 52.5,
                    "longitude": 13.5,
                    "ensemble_member": member,
                    "air_temperature_at_height_level_2m": 280.0 + index,
                    # Only the first ten members carry precipitation.
                    "precipitation_amount_sum_1h": 0.1 if index < 10 else np.nan,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("stream", [True, False])
def test_members_become_an_ordered_dimension(mock_client, stream):
    model = Model(client=mock_client, model=Models.EPT2_E)
    response = create_mock_arrow_response(
        _member_frame(datetime(2026, 9, 29), num_hours=3)
    )

    with patch.object(model._query_engine._api, "post", return_value=response) as post:
        ds = model.get_forecasts(
            points=LatLon(lat=52.52, lon=13.405),
            max_lead_time=3,
            variables=[
                Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
                Variables.PRECIPITATION_AMOUNT_SUM_1H,
            ],
            include_ensemble_members=True,
            stream=stream,
            print_progress=False,
        ).to_xarray()

    assert post.call_args.kwargs["data"]["include_ensemble_members"] is True
    assert ds["air_temperature_at_height_level_2m"].dims == (
        "points",
        "init_time",
        "prediction_timedelta",
        "ensemble_member",
    )
    assert list(ds.ensemble_member.values) == MEMBERS
    temperature = ds["air_temperature_at_height_level_2m"].isel(
        points=0, init_time=0, prediction_timedelta=0
    )
    assert temperature.values.tolist() == [280.0, 281.0, 282.0, 290.0, 300.0]
    precipitation = ds["precipitation_amount_sum_1h"].isel(
        points=0, init_time=0, prediction_timedelta=0
    )
    assert np.isnan(precipitation.sel(ensemble_member="ept2_e_10"))
    assert float(precipitation.sel(ensemble_member="ept2_e_2")) == pytest.approx(0.1)


def test_flag_is_not_sent_by_default(mock_client):
    model = Model(client=mock_client, model=Models.EPT2_E)
    frame = _member_frame(datetime(2026, 9, 29), num_hours=1).drop(
        columns="ensemble_member"
    )
    frame = frame.drop_duplicates(subset=["prediction_timedelta"])
    response = create_mock_arrow_response(frame)

    with patch.object(model._query_engine._api, "post", return_value=response) as post:
        ds = model.get_forecasts(
            points=LatLon(lat=52.52, lon=13.405), max_lead_time=1, stream=False
        ).to_xarray()

    assert "include_ensemble_members" not in post.call_args.kwargs["data"]
    assert "ensemble_member" not in ds.dims


def test_rejects_deterministic_models(mock_client):
    model = Model(client=mock_client, model=Models.EPT2)

    with pytest.raises(ValueError, match="ept2 has no ensemble members") as raised:
        model.get_forecasts(
            points=LatLon(lat=52.52, lon=13.405), include_ensemble_members=True
        )
    assert "ept2_e" in str(raised.value)


def _forbidden(body: dict) -> Mock:
    response = Mock(spec=requests.Response)
    response.ok = False
    response.status_code = 403
    response.json.return_value = body
    return response


def test_a_members_refusal_shows_the_servers_reason(mock_client):
    """Without the members feature, query-engine answers 403 with a reason.

    The SDK used to replace it with "check your API key", which sends the
    caller after the wrong fix.
    """
    model = Model(client=mock_client, model=Models.EPT2_E)
    refused = _forbidden(
        {"detail": "Ensemble members are not accessible for model ept2_e."}
    )

    with patch.object(model._query_engine._api._session, "post", return_value=refused):
        with pytest.raises(UnauthorizedError) as raised:
            model.get_forecasts(
                points=LatLon(lat=52.52, lon=13.405),
                include_ensemble_members=True,
                stream=False,
            )

    assert "not accessible for model ept2_e" in str(raised.value)
    assert "API key" not in str(raised.value)


def test_a_403_without_a_reason_keeps_the_api_key_hint(mock_client):
    model = Model(client=mock_client, model=Models.EPT2_E)

    with patch.object(
        model._query_engine._api._session, "post", return_value=_forbidden({})
    ):
        with pytest.raises(UnauthorizedError, match="check your API key"):
            model.get_forecasts(points=LatLon(lat=52.52, lon=13.405), stream=False)


def test_rejects_statistics_together_with_members(mock_client):
    model = Model(client=mock_client, model=Models.EPT2_E)

    with pytest.raises(ValueError, match="cannot be combined with `statistics`"):
        model.get_forecasts(
            points=LatLon(lat=52.52, lon=13.405),
            statistics=["mean"],
            include_ensemble_members=True,
        )


def test_rejects_lazy_load_with_members(mock_client):
    model = Model(client=mock_client, model=Models.EPT2_E)

    with pytest.raises(ValueError, match="include_ensemble_members"):
        model.get_forecasts(
            latitude=slice(55, 50),
            longitude=slice(5, 10),
            lazy_load=True,
            include_ensemble_members=True,
        )
