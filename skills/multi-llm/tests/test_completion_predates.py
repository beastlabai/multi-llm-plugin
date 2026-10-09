"""`StateManager.completion_predates` and the one reading of a stamp.

An apply step after its review is re-run on its own must run again: the new
findings are not applied yet. `completion_predates(apply, review)` answers that
from `state.json`'s phase stamps alone: the newer of the apply's completion and
its skip, compared strictly with the review's completion. A date that does not
parse keeps the apply done, as in every other cycle check (absence of a date
is not staleness).

Three parsers date an apply against its review — `state_manager`'s canonical
`_parse_completion_time` and its copies in the scheduler and the prerequisite
script. They are not merged (the scheduler imports `state_manager`, and the
script is a top-level module), so this file holds them to one reading.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
from utils import write_batch_scheduler  # noqa: E402
from utils.state_manager import StateManager, _parse_completion_time  # noqa: E402

APPLY, REVIEW = "apply-suggestions", "review-plan"
_BEFORE = "2026-01-01T10:00:00"
_REVIEWED = "2026-01-02T10:00:00"
_AFTER = "2026-01-03T10:00:00"

#: Not on file at all (a key absent from its map), as opposed to a bad value.
ABSENT = object()


@pytest.fixture
def state(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    return StateManager(plan)


@pytest.fixture
def local_zone_off_utc():
    """A local zone at UTC+05:30, set the POSIX way (no tz database needed).

    A half-hour offset, so a parser that dropped the offset instead of
    converting it, or read the stamp as UTC, cannot pass by coincidence.
    """
    if not hasattr(time, "tzset"):  # pragma: no cover - not on POSIX
        pytest.skip("time.tzset is POSIX-only")
    original = os.environ.get("TZ")
    os.environ["TZ"] = "MLL-5:30"
    time.tzset()
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


def _records(state, *, reviewed=ABSENT, completed=ABSENT, skipped=ABSENT):
    """Write the raw stamps, exactly as given, into the state's maps.

    *skipped* is the whole skip record; a str is wrapped as its `skipped_at`.
    """
    if reviewed is not ABSENT:
        state.state.setdefault("phases_completed", {})[REVIEW] = reviewed
    if completed is not ABSENT:
        state.state.setdefault("phases_completed", {})[APPLY] = completed
    if skipped is not ABSENT:
        record = (
            {"skipped_at": skipped, "reason": "test"}
            if isinstance(skipped, str) else skipped
        )
        state.state.setdefault("phases_skipped", {})[APPLY] = record


class TestOneApplyRecord:

    @pytest.mark.parametrize("stamp,predates", [
        (_BEFORE, True), (_AFTER, False), (_REVIEWED, False),
    ], ids=["before", "after", "equal"])
    def test_a_completion_is_compared_strictly(self, state, stamp, predates):
        _records(state, reviewed=_REVIEWED, completed=stamp)
        assert state.completion_predates(APPLY, REVIEW) is predates

    @pytest.mark.parametrize("stamp,predates", [
        (_BEFORE, True), (_AFTER, False), (_REVIEWED, False),
    ], ids=["before", "after", "equal"])
    def test_an_apply_recorded_only_as_skipped(self, state, stamp, predates):
        """A skip written after the review (`full`, `--skip`) skips the step;
        one written before it is an earlier cycle's, and the step runs."""
        _records(state, reviewed=_REVIEWED, skipped=stamp)
        assert state.completion_predates(APPLY, REVIEW) is predates

    def test_no_apply_record_is_not_a_record_that_predates(self, state):
        """The caller's own case (the apply never ran), answered False here."""
        _records(state, reviewed=_REVIEWED)
        assert state.completion_predates(APPLY, REVIEW) is False

    def test_no_review_completion_predates_nothing(self, state):
        _records(state, completed=_BEFORE, skipped=_BEFORE)
        assert state.completion_predates(APPLY, REVIEW) is False


class TestBothApplyRecordsOnFile:
    """`mark_phase_completed` leaves an older skip in place, so an apply can
    carry both. The NEWER is compared — not `full`'s `_recorded` precedence,
    under which an earlier cycle's completion would hide a later skip."""

    @pytest.mark.parametrize("completed,skipped,predates", [
        # skipped in round one, completed in round two (after the review)
        (_AFTER, _BEFORE, False),
        # completed in round one, skipped after the review: precedence would
        # read the old completion and run the step again
        (_BEFORE, _AFTER, False),
        # both from before the review: the review's findings are unapplied
        (_BEFORE, "2026-01-01T11:00:00", True),
        ("2026-01-01T11:00:00", _BEFORE, True),
        # the newer one equal to the review keeps the apply done
        (_BEFORE, _REVIEWED, False),
        (_REVIEWED, _BEFORE, False),
    ], ids=[
        "skip-r1-complete-r2", "complete-r1-skip-r2", "both-before-skip-newer",
        "both-before-completion-newer", "skip-equal", "completion-equal",
    ])
    def test_the_newer_stamp_is_compared(self, state, completed, skipped, predates):
        _records(state, reviewed=_REVIEWED, completed=completed, skipped=skipped)
        assert state.completion_predates(APPLY, REVIEW) is predates

    def test_through_the_real_writers(self, state):
        """Skipped, then the review is re-run, then the apply completes."""
        state.mark_phase_skipped(APPLY, "no valid findings")
        state.mark_phase_completed(REVIEW)
        # The writers stamp `now`; push the skip a second back so the order
        # does not hang on the clock's resolution.
        skipped_at = datetime.fromisoformat(
            state.state["phases_skipped"][APPLY]["skipped_at"])
        state.state["phases_skipped"][APPLY]["skipped_at"] = (
            skipped_at - timedelta(seconds=1)).isoformat()
        assert state.completion_predates(APPLY, REVIEW) is True

        state.mark_phase_completed(APPLY)
        assert APPLY in state.state["phases_skipped"]
        assert state.completion_predates(APPLY, REVIEW) is False


class TestADateThatDoesNotParseKeepsTheApplyDone:
    """Every case below would answer True on its other, readable record."""

    @pytest.mark.parametrize("bad", ["garbage", "", 12345, None])
    def test_an_unparseable_review_stamp(self, state, bad):
        _records(state, reviewed=bad, completed=_BEFORE, skipped=_BEFORE)
        assert state.completion_predates(APPLY, REVIEW) is False

    @pytest.mark.parametrize("bad", ["garbage", "", 12345, None, ["2026-01-01"]])
    def test_an_unparseable_completion(self, state, bad):
        _records(state, reviewed=_REVIEWED, completed=bad, skipped=_BEFORE)
        assert state.completion_predates(APPLY, REVIEW) is False

    @pytest.mark.parametrize("bad", [
        "2026-01-01T10:00:00", None, ["2026-01-01T10:00:00"], 12345,
    ], ids=["a-bare-string", "none", "a-list", "a-number"])
    def test_a_skip_that_is_not_a_dict(self, state, bad):
        _records(state, reviewed=_REVIEWED, completed=_BEFORE)
        state.state.setdefault("phases_skipped", {})[APPLY] = bad
        assert state.completion_predates(APPLY, REVIEW) is False

    @pytest.mark.parametrize("record", [
        {"reason": "no skipped_at"},
        {"skipped_at": "garbage", "reason": "r"},
        {"skipped_at": "", "reason": "r"},
        {"skipped_at": None, "reason": "r"},
    ], ids=["missing", "garbage", "empty", "none"])
    def test_a_skip_with_no_parseable_skipped_at(self, state, record):
        _records(state, reviewed=_REVIEWED, completed=_BEFORE, skipped=record)
        assert state.completion_predates(APPLY, REVIEW) is False


class TestPhaseWordsOnly:

    def test_the_code_review_is_read_under_its_phase_word(self, state):
        """`state.json` keys the code review `code-review`; the mode word
        `review-code` finds no stamp, so it never reads as re-run."""
        state.state["phases_completed"] = {
            "code-review": _REVIEWED, "apply-code-fixes": _BEFORE}
        assert state.completion_predates("apply-code-fixes", "code-review") is True
        assert state.completion_predates("apply-code-fixes", "review-code") is False


# ---------------------------------------------------------------------------
# One reading of a stamp
# ---------------------------------------------------------------------------

PARSERS = {
    "state_manager._parse_completion_time": _parse_completion_time,
    "write_batch_scheduler._parse_local_timestamp":
        write_batch_scheduler._parse_local_timestamp,
    "check_workflow_prerequisites._parse_timestamp": prereq._parse_timestamp,
}

STAMPS = [
    "2026-10-09T10:00:00",
    "2026-10-09T10:00:00.123456",
    "2026-10-09T04:30:00+00:00",
    "2026-10-09T12:00:00+02:00",
    "2026-10-09T04:30:00Z",
    "",
    12345,
    None,
    "not-a-date",
]


class TestOneReadingOfAStamp:

    @pytest.mark.parametrize("value", STAMPS, ids=repr)
    def test_the_three_parsers_agree(self, local_zone_off_utc, value):
        readings = {name: parse(value) for name, parse in PARSERS.items()}
        assert len(set(readings.values())) == 1, readings

    def test_a_stamp_with_an_offset_is_read_as_local_time(self, local_zone_off_utc):
        for parse in PARSERS.values():
            assert parse("2026-10-09T04:30:00+00:00") == datetime(2026, 10, 9, 10, 0)
            assert parse("2026-10-09T10:00:00") == datetime(2026, 10, 9, 10, 0)
            assert parse("") is None and parse(12345) is None
            assert parse("not-a-date") is None

    @pytest.mark.parametrize("apply_stamp,predates", [
        ("2026-10-09T10:00:00", False),
        ("2026-10-09T09:59:59", True),
        ("2026-10-09T10:00:01", False),
    ], ids=["same-instant", "a-second-before", "a-second-after"])
    def test_a_review_stamp_with_an_offset_and_a_naive_local_apply_stamp(
        self, state, local_zone_off_utc, apply_stamp, predates
    ):
        """10:00 local at UTC+05:30 is 04:30 UTC: the same instant, so equal,
        and equal keeps the apply done."""
        review_utc = datetime(2026, 10, 9, 4, 30, tzinfo=timezone.utc).isoformat()
        _records(state, reviewed=review_utc, completed=apply_stamp)
        assert state.completion_predates(APPLY, REVIEW) is predates

    def test_an_apply_stamp_equal_to_the_reviews_keeps_the_apply_done(self, state):
        _records(state, reviewed=_REVIEWED, completed=_REVIEWED, skipped=_BEFORE)
        assert state.completion_predates(APPLY, REVIEW) is False
