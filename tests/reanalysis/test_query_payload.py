from datetime import UTC, datetime

import pytest

from jua.reanalysis._types.query_payload_types import (
    ReanalysisQueryPayload,
    build_time_arg,
)
from jua.reanalysis.models import ReanalysisModels
from jua.weather._types.query_payload_types import TimeSlice

_ONE_TIME = datetime(2024, 1, 15, tzinfo=UTC).isoformat()


def _payload(geo: dict, time) -> ReanalysisQueryPayload:
    return ReanalysisQueryPayload(
        models=[ReanalysisModels.ERA5],
        geo=geo,  # type: ignore[arg-type]
        time=time,
        variables=["air_temperature_at_height_level_2m"],
    )


def test_variables_serialize_to_api_names():
    from jua.weather.variables import Variables

    payload = ReanalysisQueryPayload(
        models=[ReanalysisModels.ERA5],
        geo={"type": "point", "value": [(47.0, 8.0)], "method": "nearest"},  # type: ignore[arg-type]
        time=_ONE_TIME,
        variables=[Variables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M],  # type: ignore[list-item]
    )
    assert payload.model_dump()["variables"] == ["air_temperature_at_height_level_2m"]


def test_model_serializes_to_api_name():
    payload = _payload(
        {"type": "point", "value": [(47.0, 8.0)], "method": "nearest"}, _ONE_TIME
    )
    assert payload.model_dump()["models"] == ["arco_era5"]


def test_row_estimate_single_point_single_time():
    payload = _payload(
        {"type": "point", "value": [(47.0, 8.0)], "method": "nearest"}, _ONE_TIME
    )
    assert payload.num_requested_rows() == 1


def test_row_estimate_counts_points_times_timesteps():
    # 3 points x 25 hourly steps (inclusive of both endpoints)
    payload = _payload(
        {
            "type": "point",
            "value": [(47.0, 8.0), (48.0, 9.0), (49.0, 10.0)],
            "method": "nearest",
        },
        TimeSlice(
            start=datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
            end=datetime(2024, 1, 2, tzinfo=UTC).isoformat(),
        ),
    )
    assert payload.num_requested_rows() == 3 * 25


def test_row_estimate_list_of_times():
    payload = _payload(
        {"type": "point", "value": [(47.0, 8.0)], "method": "nearest"},
        [
            datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
            datetime(2024, 1, 2, tzinfo=UTC).isoformat(),
        ],
    )
    assert payload.num_requested_rows() == 2


def test_row_estimate_bounding_box_matches_grid():
    """A 0.5x0.5 degree box on the 0.25 degree grid is 3x3 cells.

    Pinned against the live API, which returns exactly 9 rows for this box at a
    single timestamp.
    """
    payload = _payload(
        {"type": "bounding_box", "value": [((52.0, 13.0), (52.5, 13.5))]},
        datetime(2024, 1, 15, tzinfo=UTC).isoformat(),
    )
    assert payload.num_requested_rows() == 9


def test_row_estimate_bounding_box_ignores_corner_order():
    lower_first = _payload(
        {"type": "bounding_box", "value": [((52.0, 13.0), (52.5, 13.5))]}, _ONE_TIME
    )
    upper_first = _payload(
        {"type": "bounding_box", "value": [((52.5, 13.5), (52.0, 13.0))]}, _ONE_TIME
    )
    assert lower_first.num_requested_rows() == upper_first.num_requested_rows() == 9


def test_row_estimate_handles_negative_longitudes():
    # -1.0 .. -0.5 spans -1.0, -0.75, -0.5 => 3 cells
    payload = _payload(
        {"type": "bounding_box", "value": [((52.0, -1.0), (52.5, -0.5))]}, _ONE_TIME
    )
    assert payload.num_requested_rows() == 9


def test_build_time_arg_accepts_slice():
    result = build_time_arg(
        slice(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))
    )
    assert isinstance(result, TimeSlice)
    assert result.start.startswith("2024-01-01")
    assert result.end.startswith("2024-01-02")


def test_build_time_arg_rejects_latest():
    """The API documents "latest" but returns 500 for it, so we reject early."""
    with pytest.raises(ValueError, match="no \"latest\" selector"):
        build_time_arg("latest")  # type: ignore[arg-type]


def test_build_time_arg_rejects_none():
    with pytest.raises(ValueError, match="`time` is required"):
        build_time_arg(None)  # type: ignore[arg-type]
