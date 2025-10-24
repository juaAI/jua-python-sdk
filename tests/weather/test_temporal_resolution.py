import pytest

from jua.weather._model_meta import TemporalResolution


@pytest.mark.parametrize(
    "base,special,test_cases",
    [
        (6, tuple(), [((0, 48), 9)]),
        (6, ((1, 0, 240),), [((0, 48), 49)]),
        (
            6,
            ((1, 0, 24), (3, 24, 48)),
            [((0, 48), 33), ((0, 72), 37)],
        ),
    ],
)
def test_temporal_resolution(
    base: int,
    special: tuple[tuple[int, int, int]],
    test_cases: list[tuple[int, int], int],
) -> None:
    tr = TemporalResolution(base=base, special=special)
    for (from_h, to_h), expected_timedeltas in test_cases:
        num_timedeltas = tr.num_prediction_timedeltas(from_h, to_h)
        assert num_timedeltas == expected_timedeltas, (
            f"{(from_h, to_h)}: {num_timedeltas} != {expected_timedeltas}"
        )
