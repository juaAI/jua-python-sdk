import pytest

from jua.weather import Model, Models
from jua.weather._model_meta import get_model_meta_info
from jua.weather._types.query_payload_types import ForecastQueryPayload, GeoFilter


@pytest.mark.parametrize("num_points", [1, 2, 5, 10])
@pytest.mark.parametrize("num_timedeltas", [1, 2, 5, 10, 49])
def test_simple_count_rows(num_points: int, num_timedeltas: int):
    for m in Models:
        meta = get_model_meta_info(m)
        if meta.has_forecast_file_access:
            payload = ForecastQueryPayload(
                models=[m],
                init_time="latest",
                geo=GeoFilter(type="point", value=[(8, p) for p in range(num_points)]),
                prediction_timedelta=list(range(num_timedeltas)),
                variables=["air_temperature_at_height_level_2m"],
            )
            computed_rows = payload.num_requested_points()
            assert computed_rows == num_points * num_timedeltas


@pytest.mark.parametrize(
    "bboxes,model,expected_rows",
    [
        [((8, 8), (9, 9)), Models.AIFS, 5 * 5],
        [((8, 8), (9, 9)), Models.AURORA, 13 * 13],
        [((7, 7), (8, 8)), Models.EPT1_5, 12 * 12],
        [((8, 8), (9, 9)), Models.EPT1_5, 13 * 13],
        [((8, 8), (9, 9)), Models.EPT2, 13 * 13],
        [((8, 8), (9, 9)), Models.EPT2_RR, 5 * 5],
        [((8, 8), (9, 9)), Models.EPT2_E, 5 * 5],
    ],
)
@pytest.mark.parametrize("num_timedeltas", [1, 2, 10])
def test_simple_count_rows_from_bbox(
    bboxes: tuple[tuple[float, float], tuple[float, float]],
    model: Model,
    expected_rows: int,
    num_timedeltas: int,
):
    payload = ForecastQueryPayload(
        models=[model],
        init_time="latest",
        geo=GeoFilter(type="bounding_box", value=[bboxes]),
        prediction_timedelta=list(range(num_timedeltas)),
        variables=["air_temperature_at_height_level_2m"],
    )
    computed_rows = payload.num_requested_points()
    assert computed_rows == expected_rows * num_timedeltas
