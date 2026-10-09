"""`check_workflow_prerequisites.apply_phase_in_progress`: one rule, every reader.

`get_workflow_status` marks an apply phase `in_progress` (the status the
next-step builder spells with `--resume`) when it has progress of the review on
record: a started record or one of its own batch ids recorded `completed`,
neither of them older than the review's latest completion, and only while that
review is `completed`. The rule is a predicate now, so a composite inferring
`--resume` asks exactly what `status` asks. These tests pin the predicate and
hold `status` to it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
from tests.test_workflow_prerequisites import (  # noqa: E402
    APPLY_PHASE_BATCH_ROWS,
    _NEW,
    _OLD,
    _date_cycle,
    _record,
    _status,
    _workflow_plan,
)
from utils.state_manager import StateManager  # noqa: E402

ROW_IDS = [row[0] for row in APPLY_PHASE_BATCH_ROWS]


def _in_progress(plan, phase):
    """The predicate's answer, and `status`'s, read from the same file."""
    answer = prereq.apply_phase_in_progress(StateManager(plan, read_only=True), phase)
    shown = _status(plan)["phases"][phase]["status"]
    return answer, shown


def _start(plan, phase, stamp):
    """The native leg's emit record, stamped *stamp*."""
    state = StateManager(plan)
    state.state.setdefault("phases_started", {})[phase] = stamp
    state.save()


@pytest.mark.parametrize("phase,batch_id,fixture,completed", APPLY_PHASE_BATCH_ROWS,
                         ids=ROW_IDS)
class TestProgressOfThisReview:

    def test_a_started_record_after_the_review(
        self, tmp_path, phase, batch_id, fixture, completed
    ):
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed)
        review = prereq.PHASE_PREREQUISITES[phase][0]
        _date_cycle(plan, phase, review, _OLD)
        _start(plan, phase, _NEW)
        assert _in_progress(plan, phase) == (True, "in_progress")

    def test_a_completed_batch_id_after_the_review(
        self, tmp_path, phase, batch_id, fixture, completed
    ):
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed, tasks={batch_id: "completed"})
        review = prereq.PHASE_PREREQUISITES[phase][0]
        _date_cycle(plan, phase, review, _OLD, metric_at=_NEW)
        assert _in_progress(plan, phase) == (True, "in_progress")

    def test_a_completed_batch_id_with_no_dated_run(
        self, tmp_path, phase, batch_id, fixture, completed
    ):
        """Absence of a date is not staleness."""
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed, tasks={batch_id: "completed"})
        review = prereq.PHASE_PREREQUISITES[phase][0]
        _date_cycle(plan, phase, review, _NEW)
        assert _in_progress(plan, phase) == (True, "in_progress")

    def test_no_progress(self, tmp_path, phase, batch_id, fixture, completed):
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed, tasks={batch_id: "failed"})
        assert _in_progress(plan, phase) == (False, "pending")


@pytest.mark.parametrize("phase,batch_id,fixture,completed", APPLY_PHASE_BATCH_ROWS,
                         ids=ROW_IDS)
class TestProgressOfAnEarlierReviewIsDiscarded:

    @pytest.mark.parametrize("source", ["started_at", "metric_at"])
    def test_batches_older_than_the_review(
        self, tmp_path, phase, batch_id, fixture, completed, source
    ):
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed, tasks={batch_id: "completed"})
        review = prereq.PHASE_PREREQUISITES[phase][0]
        _date_cycle(plan, phase, review, _NEW, **{source: _OLD})
        assert _in_progress(plan, phase) == (False, "pending")

    def test_a_started_record_older_than_the_review(
        self, tmp_path, phase, batch_id, fixture, completed
    ):
        """A native run abandoned before the review was re-run."""
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed)
        review = prereq.PHASE_PREREQUISITES[phase][0]
        _date_cycle(plan, phase, review, _NEW)
        _start(plan, phase, _OLD)
        assert _in_progress(plan, phase) == (False, "pending")


class TestAReviewNotCompletedHasNoCycle:

    def test_a_review_never_run(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(plan, tasks={"apply-batch-0": "completed"})
        _start(plan, "apply-suggestions", _NEW)
        assert prereq.apply_phase_in_progress(
            StateManager(plan), "apply-suggestions") is False

    def test_a_review_recorded_only_as_skipped(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(plan, skipped=("review-plan",), tasks={"apply-batch-0": "completed"})
        assert _in_progress(plan, "apply-suggestions")[0] is False

    def test_a_code_review_stale_after_implement(self, tmp_path):
        """Read as pending (`phase_completion_is_stale`), it takes its
        cycle's batches with it."""
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan", "apply-suggestions",
                                 "generate-tasks", "code-review"),
                tasks={"apply-fix-batch-0": "completed"})
        state = StateManager(plan)
        state.state["phases_completed"]["code-review"] = _OLD
        state.state["phases_completed"]["implement"] = _NEW
        state.state["metrics"] = {"apply-code-fixes": [{"timestamp": _NEW}]}
        state.save()
        assert _in_progress(plan, "apply-code-fixes") == (False, "pending")


class TestTheContract:

    def test_another_phases_batch_never_counts(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan",),
                tasks={"apply-task-batch-0": "completed",
                       "apply-fix-batch-0": "completed"})
        assert prereq.apply_phase_in_progress(
            StateManager(plan), "apply-suggestions") is False

    def test_it_does_not_ask_whether_the_phase_itself_is_done(self, tmp_path):
        """`status` checks `DONE_STATUSES` first; the predicate does not."""
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan", "apply-suggestions"),
                tasks={"apply-batch-0": "completed"})
        assert _in_progress(plan, "apply-suggestions") == (True, "completed")

    @pytest.mark.parametrize("word", ["review-plan", "apply-fixes", "implement"])
    def test_a_word_that_is_not_an_apply_phase_raises(self, tmp_path, word):
        plan = _workflow_plan(tmp_path)
        with pytest.raises(KeyError):
            prereq.apply_phase_in_progress(StateManager(plan), word)
