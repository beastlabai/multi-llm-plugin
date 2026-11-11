"""The review-cycle rule, as one table (`state_manager.predates_review`).

A record older than the review's latest completion is an earlier cycle's; a
missing date on either side keeps it. Every check over an apply phase's own
records reads it from here: the scheduler's earlier report rows and reused
snapshots, the apply base's recorded-progress signals and its `--fresh`
refusal, the improve summary's applied files, and the prerequisite script's
stale batches. Two readers take a parameter, each for a reason the table
names: records stamped to the second (`whole_seconds`) and a snapshot
manifest with no date (`undated_predates`).
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import predates_review, stamp_or_mtime  # noqa: E402
from utils.write_batch_scheduler import (  # noqa: E402
    REVIEW_PHASE_BY_WRITE_KIND, review_completed_at,
)
from utils.write_safety import TaskSnapshot  # noqa: E402

#: The review's completion, half a second into 10:00:00.
REVIEWED = datetime(2026, 10, 9, 10, 0, 0, 500000)
SECOND = timedelta(seconds=1)
MICRO = timedelta(microseconds=1)

RULE = [
    # (id, stamped, reviewed_at, options, earlier cycle?)
    ("a second older", REVIEWED - SECOND, REVIEWED, {}, True),
    ("a microsecond older", REVIEWED - MICRO, REVIEWED, {}, True),
    ("the same instant", REVIEWED, REVIEWED, {}, False),
    ("newer", REVIEWED + SECOND, REVIEWED, {}, False),
    ("no record date", None, REVIEWED, {}, False),
    ("no review date", REVIEWED - SECOND, None, {}, False),
    ("no date at all", None, None, {}, False),
    # Records stamped to the second: one in the review's own second is this
    # cycle's, one in the second before is not.
    ("whole second, the review's own", REVIEWED.replace(microsecond=0), REVIEWED,
     {"whole_seconds": True}, False),
    ("same, without whole_seconds", REVIEWED.replace(microsecond=0), REVIEWED,
     {}, True),
    ("whole second, the one before", REVIEWED.replace(microsecond=0) - SECOND,
     REVIEWED, {"whole_seconds": True}, True),
    ("whole second, no review date", REVIEWED - SECOND, None,
     {"whole_seconds": True}, False),
    # A snapshot manifest with no date is set aside, but only against a
    # review that has one.
    ("undated snapshot", None, REVIEWED, {"undated_predates": True}, True),
    ("undated snapshot, no review date", None, None,
     {"undated_predates": True}, False),
    ("dated snapshot, newer", REVIEWED + SECOND, REVIEWED,
     {"undated_predates": True}, False),
]


@pytest.mark.parametrize(
    "stamped, reviewed_at, options, earlier",
    [row[1:] for row in RULE], ids=[row[0] for row in RULE],
)
def test_the_rule(stamped, reviewed_at, options, earlier):
    assert predates_review(stamped, reviewed_at, **options) is earlier


SNAPSHOTS = [
    # (id, taken_at, cutoff, set aside?)
    ("taken a second before", "2026-10-09T09:59:59", REVIEWED, True),
    ("taken in the review's second", "2026-10-09T10:00:00", REVIEWED, False),
    ("taken after", "2026-10-09T10:00:01", REVIEWED, False),
    ("no taken_at", None, REVIEWED, True),
    ("unreadable taken_at", "yesterday", REVIEWED, True),
    ("no review date", "2026-10-09T09:59:59", None, False),
    ("no date at all", None, None, False),
]


@pytest.mark.parametrize(
    "taken_at, cutoff, aside",
    [row[1:] for row in SNAPSHOTS], ids=[row[0] for row in SNAPSHOTS],
)
def test_a_snapshot_reads_the_rule_to_the_second_and_sets_an_undated_one_aside(
    tmp_path, taken_at, cutoff, aside,
):
    snapshot = TaskSnapshot(
        task_id="apply-batch-1", directory=str(tmp_path), root=str(tmp_path),
        taken_at=taken_at,
    )
    assert snapshot.taken_before(cutoff) is aside


def test_a_stamp_dates_a_file_before_its_mtime(tmp_path):
    path = tmp_path / "write_batch_report.json"
    path.write_text("{}", encoding="utf-8")
    mtime = datetime(2026, 10, 9, 8, 0, 0)
    os.utime(path, (mtime.timestamp(), mtime.timestamp()))
    assert stamp_or_mtime("2026-10-09T11:00:00", path) == datetime(2026, 10, 9, 11, 0)
    assert stamp_or_mtime(None, path) == mtime
    assert stamp_or_mtime("not-a-date", path) == mtime
    assert stamp_or_mtime(None, tmp_path / "missing.json") is None


@pytest.mark.parametrize("kind, review", sorted(REVIEW_PHASE_BY_WRITE_KIND.items()))
def test_the_cutoff_is_the_applied_reviews_completion(kind, review):
    recorded = {"phases_completed": {review: "2026-10-09T10:00:00", "implement": "x"}}
    assert review_completed_at(recorded, kind) == datetime(2026, 10, 9, 10, 0)


@pytest.mark.parametrize("recorded, kind", [
    ({"phases_completed": {"implement": "2026-10-09T10:00:00"}}, "implement"),
    ({"phases_completed": {"review-plan": "2026-10-09T10:00:00"}}, None),
    ({"phases_completed": {"review-plan": "nope"}}, "apply_suggestions"),
    ({"phases_completed": {}}, "apply_suggestions"),
    ({"phases_completed": []}, "apply_suggestions"),
    (None, "apply_suggestions"),
], ids=["implement", "no-kind", "unparseable", "no-review", "not-a-map", "no-state"])
def test_no_cutoff(recorded, kind):
    assert review_completed_at(recorded, kind) is None
