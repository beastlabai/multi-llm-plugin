"""``write_outcomes``: the partial-application and resume rules around the
verdict table, as pure functions over ``TaskOutcome``
(docs/development/write-scheduler.md, "Retry after edits" and "Declined items
and partial applications").

Pinned here: the write outcome of a dispatch the table never saw, the partial
application a failed dispatch that moved bytes becomes, the checks a
``--human-task <batch>=skip`` answer must pass, and the facts a stop's
``resume_why`` reads, each read only when its rule needs it. The wording each
produces is pinned end to end by the scheduler's own tests
(``test_write_retry_redispatch``, ``test_write_batch_declined``,
``test_resume_why``, ``test_stop_resume_wording``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils import write_outcomes  # noqa: E402
from utils.retry_ladder import STATUS_TIMEOUT  # noqa: E402
from utils.subagent_executor import (  # noqa: E402
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_REJECTED,
    STATUS_SCHEMA_INVALID,
    STATUS_UNPARSEABLE,
    WRITE_CLEAN_FAILURE,
    WRITE_PARTIAL,
    WRITE_POLICY_VIOLATION,
    WRITE_SCHEMA_REJECTED,
    TaskOutcome,
)


def _outcome(status, **fields):
    return TaskOutcome(task_id="apply-batch-0", kind="apply_suggestions",
                       runner="codex", status=status, **fields)


# ===========================================================================
# A dispatch the verdict table never saw
# ===========================================================================

class TestClassifyAttempt:

    @pytest.mark.parametrize("status", [STATUS_SCHEMA_INVALID, STATUS_UNPARSEABLE])
    def test_a_reply_that_never_fit_is_a_schema_rejection(self, status):
        outcome = _outcome(status, reason="prose")
        write_outcomes.classify_attempt(outcome, snapshot_dir="/s")
        assert outcome.write_outcome == WRITE_SCHEMA_REJECTED
        assert outcome.reason == "prose"

    def test_the_pre_retry_checks_partial_application_names_the_snapshot(self):
        outcome = _outcome(STATUS_PARTIAL, reason="PARTIAL APPLICATION: 1 file(s) changed.")
        write_outcomes.classify_attempt(outcome, snapshot_dir="/s/pre_batch/x")
        assert outcome.write_outcome == WRITE_PARTIAL
        assert outcome.reason == (
            "PARTIAL APPLICATION: 1 file(s) changed. The pre-batch snapshot is "
            "kept at /s/pre_batch/x."
        )

    @pytest.mark.parametrize("outcome", [
        _outcome(STATUS_TIMEOUT),
        _outcome(STATUS_OK),
        _outcome(STATUS_UNPARSEABLE, write_outcome=WRITE_CLEAN_FAILURE),
    ], ids=["a transport failure", "ok", "already classified"])
    def test_anything_else_is_left_as_it_is(self, outcome):
        before = (outcome.write_outcome, outcome.reason)
        write_outcomes.classify_attempt(outcome, snapshot_dir="/s")
        assert (outcome.write_outcome, outcome.reason) == before


class TestPartialWhenBytesMoved:

    def test_a_timeout_that_moved_files_is_a_partial_application(self):
        outcome = _outcome(STATUS_TIMEOUT, reason="timed out after 600s",
                           files_touched=[f"f{i}.py" for i in range(6)],
                           unrestorable=["f0.py", "g.py"])
        write_outcomes.partial_when_bytes_moved(outcome, snapshot_dir="/s")
        assert outcome.status == STATUS_PARTIAL
        assert outcome.write_outcome == WRITE_PARTIAL
        assert outcome.reason == (
            "PARTIAL APPLICATION: 7 file(s) changed (f0.py, f1.py, f2.py, "
            "f3.py, f4.py, …) but the batch did not finish: timed out after "
            "600s. The batch loop stops here, the pre-batch snapshot is kept "
            "at /s, and the phase is NOT marked completed — an automatic "
            "revert would destroy the half-applied state a human needs to "
            "inspect."
        )

    def test_a_moved_file_with_no_rollback_point_counts(self):
        outcome = _outcome(STATUS_SCHEMA_INVALID, write_outcome=WRITE_SCHEMA_REJECTED,
                           unrestorable=[".env"])
        write_outcomes.partial_when_bytes_moved(outcome, snapshot_dir="/s")
        assert outcome.write_outcome == WRITE_PARTIAL
        assert "1 file(s) changed (.env) but the batch did not finish: " \
            f"{STATUS_SCHEMA_INVALID}." in outcome.reason

    @pytest.mark.parametrize("outcome", [
        _outcome(STATUS_TIMEOUT),
        _outcome(STATUS_OK, files_touched=["a.py"]),
        _outcome(STATUS_REJECTED, files_touched=["a.py"]),
        _outcome(STATUS_PARTIAL, files_touched=["a.py"], reason="r"),
        _outcome("policy_violation", write_outcome=WRITE_POLICY_VIOLATION,
                 files_touched=["a.py"]),
    ], ids=["nothing moved", "ok", "a rung-7 refusal", "already partial",
            "a verdict of its own"])
    def test_anything_else_keeps_its_verdict(self, outcome):
        before = (outcome.status, outcome.write_outcome, outcome.reason)
        write_outcomes.partial_when_bytes_moved(outcome, snapshot_dir="/s")
        assert (outcome.status, outcome.write_outcome, outcome.reason) == before


# ===========================================================================
# Keeping a stopped batch at the user's request
# ===========================================================================

TASKS = [{"task_id": "apply-batch-0"}, {"task_id": "apply-batch-1"}]


class TestCheckPartialSkips:

    def test_every_answer_that_stands_leaves_nothing_to_say(self):
        assert write_outcomes.check_partial_skips(
            ["apply-batch-0"], kind="apply_code_fixes", phase="apply-fixes",
            tasks=TASKS, evidence=lambda task: None,
        ) == ({}, [])

    def test_each_refusal_is_named_with_its_reason(self):
        refused, lines = write_outcomes.check_partial_skips(
            ["apply-batch-0", "apply-batch-9"], kind="apply_code_fixes",
            phase="apply-fixes", tasks=TASKS,
            evidence=lambda task: f"{task['task_id']} has no result on disk",
        )
        assert refused == {
            "apply-batch-0": "apply-batch-0 has no result on disk",
            "apply-batch-9": "this run holds no batch apply-batch-9 (its "
                             "batches: apply-batch-0, apply-batch-1)",
        }
        assert lines == [
            "",
            "ERROR: apply-fixes cannot take the answer below, so nothing was "
            "dispatched and no file was changed:",
            "  - --human-task apply-batch-0=skip: apply-batch-0 has no result on disk.",
            "  - --human-task apply-batch-9=skip: this run holds no batch "
            "apply-batch-9 (its batches: apply-batch-0, apply-batch-1).",
            write_outcomes.REFUSED_SKIP_WAY_FORWARD,
        ]

    def test_only_an_apply_phase_takes_it_and_nothing_is_read_for_implement(self):
        def evidence(task):
            raise AssertionError("implement is refused before any evidence")

        refused, _lines = write_outcomes.check_partial_skips(
            ["T001"], kind="implement", phase="implement",
            tasks=[{"task_id": "T001"}], evidence=evidence,
        )
        assert refused == {"T001": "only an apply phase's batch takes this answer"}


def _stopped(**over):
    verdict = {
        "write_outcome": WRITE_PARTIAL, "ok": False, "unapplied_ids": ["g2"],
        "task_id": "apply-batch-0", "expected_ids": ["g1", "g2"],
        "inputs_fingerprint": "fp", "sha256_after": {"a.py": "h-a", "b.py": "h-b"},
    }
    verdict.update(over)
    return verdict


class TestPartialSkipRefusal:

    def _refusal(self, verdict, *, digests=None, done=True, fingerprint="fp"):
        reads = []
        digests = digests or {"a.py": "h-a", "b.py": "h-b"}

        def current(relative):
            reads.append(("digest", relative))
            return digests.get(relative)

        why = write_outcomes.partial_skip_refusal(
            "apply-batch-0", verdict, result_name="apply_fix_batch_0.json",
            expected_ids=["g2", "g1"],
            fingerprint=lambda: reads.append("fingerprint") or fingerprint,
            counts_as_done=lambda: reads.append("done") or done,
            current_digest=current,
        )
        return why, reads

    def test_a_partial_application_still_on_disk_is_kept(self):
        why, reads = self._refusal(_stopped())
        assert why is None
        assert reads == ["fingerprint", ("digest", "a.py"), ("digest", "b.py")]

    def test_no_verdict(self):
        why, reads = self._refusal(None)
        assert why == "apply-batch-0's result (apply_fix_batch_0.json) carries no verdict"
        assert reads == []

    def test_another_verdict(self):
        why, reads = self._refusal(_stopped(write_outcome="applied", unapplied_ids=[]))
        assert why.startswith(
            "apply-batch-0 did not stop as a partial application whose report "
            "left items out (its last verdict: applied)")
        assert reads == []

    def test_other_items(self):
        why, _reads = self._refusal(_stopped(expected_ids=["g1"]))
        assert why == ("apply-batch-0 now holds other items than the batch that "
                       "stopped (the selection changed since)")

    def test_other_inputs(self):
        why, reads = self._refusal(_stopped(), fingerprint="other")
        assert why.startswith("the plan, the review or the batching apply-batch-0")
        assert reads == ["fingerprint"]

    def test_a_verdict_with_no_fingerprint_reads_none(self):
        why, reads = self._refusal(_stopped(inputs_fingerprint=""))
        assert why is None and "fingerprint" not in reads

    def test_a_file_restored_since(self):
        why, _reads = self._refusal(_stopped(), digests={"a.py": "h-a", "b.py": "x"})
        assert why == (
            "b.py changed since apply-batch-0 stopped (restored, or edited by "
            "hand), so what it applied is no longer what is on disk"
        )

    def test_an_answer_already_taken_asks_whether_the_loop_keeps_it(self):
        kept = _stopped(ok=True, write_outcome="applied", unapplied_ids=[],
                        skipped_items=[{"id": "g2", "reason": "r"}])
        assert write_outcomes.already_kept(kept)
        why, reads = self._refusal(kept)
        assert why is None and reads == ["fingerprint", "done"]
        why, _reads = self._refusal(kept, done=False)
        assert why.startswith("apply-batch-0's earlier kept result no longer counts")


# ===========================================================================
# What the plain --resume does
# ===========================================================================

class TestResumeWhy:

    def test_a_failed_apply_batch_that_moved_bytes(self):
        failed = _outcome(STATUS_PARTIAL, write_outcome=WRITE_PARTIAL,
                          files_touched=["plans/p.md"])
        done = TaskOutcome(task_id="apply-batch-1", kind="apply_suggestions",
                           runner="codex", status=STATUS_OK)
        reads = []
        why = write_outcomes.resume_why(
            [done, failed], [failed], write_outcomes.ResumeContext(
                kind="apply_suggestions", tasks=TASKS,
                blocking_checks=lambda: reads.append("checks") or [],
                implement_done=lambda: reads.append("done") or 0,
                kept_on=lambda task_id, *, failed_check: reads.append(task_id) or False,
            ),
        )
        assert reads == [], "no check held a task, and this is not implement"
        assert why == (
            "runs the batches that can run; apply-batch-0 is dispatched again; "
            "keeps the 1 batch already done; the half-applied state is left in "
            "place — inspect it, or `util restore-batch` the failed batch, "
            "before resuming"
        )

    def test_implement_judges_each_check_on_the_tree_failed_checks_first(self):
        tasks = [{"task_id": "T001", "declared_files": []},
                 {"task_id": "T002", "declared_files": []},
                 {"task_id": "T003", "declared_files": ["src/a.py"]}]
        failed = TaskOutcome(task_id="T001", kind="implement", runner="codex",
                             status="failed", write_outcome=WRITE_CLEAN_FAILURE)
        reads = []
        why = write_outcomes.resume_why(
            [failed], [failed], write_outcomes.ResumeContext(
                kind="implement", tasks=tasks,
                blocked={"T003": "T002"},
                blocking_checks=lambda: reads.append("checks") or ["T002"],
                dependents_of=lambda task_id: ["T003"] if task_id == "T002" else [],
                implement_done=lambda: reads.append("done") or 2,
                kept_on=lambda task_id, *, failed_check: (
                    reads.append((task_id, failed_check)) or True),
            ),
        )
        assert reads == ["checks", "done", ("T001", True), ("T002", False)]
        assert "T001's check is not run again while the files are unchanged" in why
        assert "T002's check is not run again while the files are unchanged (T003 waits on it)" in why

    def test_undone_with_leaves_out_skipped_manual_and_unknown_tasks(self):
        ctx = write_outcomes.ResumeContext(
            kind="implement",
            tasks=[{"task_id": "T002"}, {"task_id": "T003", "executor_role": "human"},
                   {"task_id": "T004"}],
            skipped={"T004"},
            dependents_of=lambda task_id: ["T002", "T003", "T004", "T009"],
        )
        assert ctx.undone_with("T001") == ["T002"]
