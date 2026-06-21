"""Tests for ``RunDefinition.expected_dissemination_datetime``."""

from datetime import datetime, time

from jua.weather._types.query_response_types import RunDefinition


def _make(dissemination_time=None, dissemination_day_offset=None) -> RunDefinition:
    return RunDefinition(
        lead_time_set=[60, 120],
        dissemination_time=dissemination_time,
        dissemination_day_offset=dissemination_day_offset,
    )


class TestExpectedDisseminationDatetime:
    """All the resolution paths for ``expected_dissemination_datetime``."""

    def test_returns_none_when_time_missing(self):
        run = _make(dissemination_time=None)
        result = run.expected_dissemination_datetime(
            datetime(2026, 5, 11, 0, 0), time(0, 0)
        )
        assert result is None

    def test_legacy_same_day_no_rollover(self):
        """00Z init, 09:00 dissemination — same day."""
        run = _make(dissemination_time=time(9, 0))
        result = run.expected_dissemination_datetime(
            datetime(2026, 5, 11, 0, 0), time(0, 0)
        )
        assert result == datetime(2026, 5, 11, 9, 0)

    def test_legacy_implicit_rollover(self):
        """18Z init, 03:00 dissemination — +1 day via legacy rollover."""
        run = _make(dissemination_time=time(3, 0))
        result = run.expected_dissemination_datetime(
            datetime(2026, 5, 11, 18, 0), time(18, 0)
        )
        assert result == datetime(2026, 5, 12, 3, 0)

    def test_legacy_same_time_no_rollover(self):
        """Dissemination == init time-of-day should NOT roll over."""
        run = _make(dissemination_time=time(6, 0))
        result = run.expected_dissemination_datetime(
            datetime(2026, 5, 11, 6, 0), time(6, 0)
        )
        assert result == datetime(2026, 5, 11, 6, 0)

    def test_explicit_offset_no_implicit_rollover(self):
        """``dissemination_day_offset=0`` is absolute — no rollover even
        when time_of_day < init's time_of_day."""
        run = _make(dissemination_time=time(3, 0), dissemination_day_offset=0)
        result = run.expected_dissemination_datetime(
            datetime(2026, 5, 11, 18, 0), time(18, 0)
        )
        assert result == datetime(2026, 5, 11, 3, 0)

    def test_seas5_d_plus_5(self):
        """SEAS5: day-1 00Z init + D+5 12:00 dissemination → day 6 12 UTC.

        This is the canonical example. ECMWF's SEAS5 ``real-time``
        forecasts are released on the 6th of each month at 12 UTC per
        CDS docs.
        """
        run = _make(dissemination_time=time(12, 0), dissemination_day_offset=5)
        result = run.expected_dissemination_datetime(
            datetime(2026, 5, 1, 0, 0), time(0, 0)
        )
        assert result == datetime(2026, 5, 6, 12, 0)

    def test_explicit_offset_one_day(self):
        """``dissemination_day_offset=1`` with same time-of-day."""
        run = _make(dissemination_time=time(0, 0), dissemination_day_offset=1)
        result = run.expected_dissemination_datetime(
            datetime(2026, 5, 11, 0, 0), time(0, 0)
        )
        assert result == datetime(2026, 5, 12, 0, 0)


class TestBackwardCompat:
    """Ensure new field defaults keep old /meta responses parseable."""

    def test_construct_without_new_field(self):
        """Old /meta responses only have ``dissemination_time``."""
        run = RunDefinition(lead_time_set=[60], dissemination_time=time(9, 0))
        assert run.dissemination_day_offset is None

    def test_construct_with_everything_default(self):
        """Full-default construction also works (global defaults)."""
        run = RunDefinition()
        assert run.lead_time_set == []
        assert run.dissemination_time is None
        assert run.dissemination_day_offset is None
