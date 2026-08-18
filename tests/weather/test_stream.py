from datetime import UTC, datetime

import pandas as pd
import pytest

from jua.weather._stream import process_arrow_streaming_response
from tests.weather.utils import create_mock_arrow_response


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
def test_process_arrow_response_preserves_wire_datetime_resolution(
    time_unit: str,
) -> None:
    instant = datetime(2026, 8, 18, 9, 30, tzinfo=UTC)
    response = create_mock_arrow_response(
        pd.DataFrame({"time": [instant]}),
        timestamp_units={"time": time_unit},
    )

    result = process_arrow_streaming_response(response, print_progress=False)

    assert str(result["time"].dtype) == f"datetime64[{time_unit}, UTC]"
    assert result["time"].tolist() == [pd.Timestamp(instant)]
