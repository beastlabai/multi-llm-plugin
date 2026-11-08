"""``write_record.record_batch_facts``: the one writer of a write batch's
per-task record, which both legs call (docs/development/write-scheduler.md,
"One record writer for both legs").

Pinned here: a record other than ``completed`` clears every earlier verdict
of the task, an apply-code-fixes batch the executor refused before dispatch
and read as done reports no checks, a resumed batch keeps the verdict its
earlier run recorded, and both legs hand the writer every fact it takes (a
new :class:`~utils.write_record.BatchFacts` field that one leg does not set
fails here unless that leg has a reason not to).
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils.state_manager import (  # noqa: E402
    PROCESSED_LEG_EXECUTOR,
    PROCESSED_LEG_NATIVE,
    TASK_VERDICT_KEYS,
)
from utils.write_record import (  # noqa: E402
    KEEP_CHECK_TREE,
    BatchFacts,
    record_batch_facts,
)

IMPLEMENT_TASK = {
    "task_id": "T001", "task_title": "Add the parser",
    "prompt": "## Task\nAdd the parser.\n",
}
FIX_TASK = {
    "task_id": "apply-fix-batch-0", "prior_changes_id": "pc-0",
    "expected_ids": ["g1", "g2"], "inputs": {"grouped_sha256": "abc"},
}


def _verdict(**overrides):
    facts = dict(
        task_id="T001", kind="implement", status="completed", reason=None,
        leg=PROCESSED_LEG_EXECUTOR, check_findings=["the check fails"],
        check_tree="tree-1", batch_check=["its own test fails"],
        own_check_ran=True, batch_check_accepted={"T000": ["known"]},
        batch_check_repeats={"T002": ["same"]}, note="ran npm test",
    )
    facts.update(overrides)
    return BatchFacts(**facts)


class TestARecordOtherThanCompletedClearsTheVerdict:

    def test_every_verdict_key_is_written_then_cleared(self):
        state = {"task_status": {}}
        record_batch_facts(state, IMPLEMENT_TASK, _verdict())
        written = {key for key in TASK_VERDICT_KEYS if key in state}
        assert written == set(TASK_VERDICT_KEYS)
        record_batch_facts(state, IMPLEMENT_TASK, _verdict(
            status="failed", reason="the runner exited 1"))
        assert [key for key in TASK_VERDICT_KEYS if key in state] == []
        assert state["task_status"] == {"T001": "failed"}
        assert state["task_status_reasons"] == {"T001": "the runner exited 1"}

    def test_the_executor_leaves_a_check_tree_alone(self):
        state = {"task_check_trees": {"T001": "tree-0"}}
        record_batch_facts(state, IMPLEMENT_TASK, _verdict(
            status="failed", check_tree=KEEP_CHECK_TREE))
        assert state["task_check_trees"] == {"T001": "tree-0"}

    def test_the_status_goes_through_the_callers_writer(self):
        calls = []
        state = {}
        record_batch_facts(
            state, IMPLEMENT_TASK, _verdict(),
            set_status=lambda task_id, status, *, reason=None: calls.append(
                (task_id, status, reason)),
        )
        assert calls == [("T001", "completed", None)]
        assert "task_status" not in state


class TestTheCheckReportGate:

    def _facts(self, **overrides):
        facts = dict(
            task_id="apply-fix-batch-0", kind="apply_code_fixes",
            status="completed", reason=None, leg=PROCESSED_LEG_EXECUTOR,
            processed_phase="apply-fixes", batch_check=["pytest fails"],
            listed={"pytest": "failed"},
        )
        facts.update(overrides)
        return BatchFacts(**facts)

    def test_a_batch_that_reports_writes_its_check_report(self):
        state = {}
        record_batch_facts(state, FIX_TASK, self._facts())
        entry = state["apply_fix_batch_checks"]["apply-fix-batch-0"]
        assert entry["findings"] == ["pytest fails"]

    def test_a_refused_batch_read_as_done_writes_none(self):
        state = {}
        record_batch_facts(state, FIX_TASK, self._facts(reports_checks=False))
        assert "apply_fix_batch_checks" not in state
        # Its items are still recorded.
        assert set(state["processed_apply-fixes"]) == {"g1", "g2"}


class TestAResumedBatch:

    def test_keeps_its_verdict_and_records_its_items(self):
        state = {"task_batch_check": {"T001": ["its own test fails"]}}
        record_batch_facts(state, IMPLEMENT_TASK, _verdict(
            status="completed", reason="resumed from T001.json",
            batch_check=None, resumed=True))
        assert state["task_batch_check"] == {"T001": ["its own test fails"]}
        assert state["task_status_reasons"] == {"T001": "resumed from T001.json"}


class TestApplyItems:

    def test_declined_skipped_and_no_op_are_recorded_by_the_leg(self):
        state = {}
        task = dict(FIX_TASK, expected_ids=["g1", "g2", "g3"])
        record = record_batch_facts(state, task, BatchFacts(
            task_id="apply-fix-batch-0", kind="apply_suggestions",
            status="completed", reason=None, leg=PROCESSED_LEG_NATIVE,
            processed_phase="apply-suggestions",
            declined={"g2": "out of scope"}, skipped={"g3": "you skipped it"},
        ))
        assert record.items == ["g1", "g2", "g3"]
        recorded = state["processed_apply-suggestions"]
        assert [recorded[i]["status"] for i in ("g1", "g2", "g3")] == [
            "applied", "declined", "skipped"]
        assert recorded["g2"]["details"]["leg"] == PROCESSED_LEG_NATIVE


# ---------------------------------------------------------------------------
# Both legs hand the writer every fact
# ---------------------------------------------------------------------------

#: The facts a leg leaves at their default, and why.
LEG_DEFAULTS = {
    "native": {
        "no_op": "the native leg has no 'changed nothing' reading",
        "reports_checks": "a native record always reports its checks",
        "resumed": "a native --resume hands out only what it dispatches",
    },
    "executor": {
        "check_tree": "the executor keeps a check's tree on its verdict stamp",
        "resumed": "set on the resume skip's own facts (dataclasses.replace)",
    },
}


def _facts_keywords(path: Path, function: str) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "BatchFacts"):
                    return {kw.arg for kw in call.keywords}
    raise AssertionError(f"no BatchFacts(...) call in {function} of {path.name}")


def test_both_legs_pass_every_fact():
    fields = {field.name for field in dataclasses.fields(BatchFacts)}
    legs = {
        "native": _facts_keywords(SKILL_DIR / "utils" / "batch_done.py", "record_batch"),
        "executor": _facts_keywords(
            SKILL_DIR / "utils" / "write_batch_scheduler.py", "_batch_facts"),
    }
    for leg, keywords in legs.items():
        assert fields - keywords == set(LEG_DEFAULTS[leg]), (
            f"the {leg} leg does not set {sorted(fields - keywords - set(LEG_DEFAULTS[leg]))}: "
            f"set it where that leg builds its BatchFacts, or add it to LEG_DEFAULTS "
            f"with the reason that leg has none"
        )
