"""A review phase reopened for validation, and a round that fails after it.

``reopen_phase_for_validation`` drops a recorded completion while other
commands may be writing ``state.json``; the base lane's ``--reaggregate`` must
not turn a dispatch or merge that raises into a completed phase
(docs/development/review-results.md).
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import review_orchestrator_base as rob  # noqa: E402
from utils import status_file  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import (  # noqa: E402
    StateManager, save_groups_payload, stamp_stable_ids,
)
from utils.validation import prepare_batched_validation_tasks  # noqa: E402

#: The completion an earlier `--reaggregate --skip-validation` recorded.
SKIPPED_AT = "2026-11-08T10:00:00"

#: Another command's locked write of `state.json`, as a separate process: a
#: phase completion (`util batch-done`) and a `record_commit_range`, in one
#: critical section. A zero lock timeout makes it report contention (exit 75)
#: rather than wait, so the interleaving needs no sleeps.
_LOCKED_WRITER = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from utils import metrics
from utils.state_manager import StateManager
metrics.STATE_LOCK_TIMEOUT_S = 0
manager = StateManager(Path(sys.argv[2]))
try:
    with metrics.state_file_lock(manager.state_file):
        document = metrics._read_state(manager.state_file)
        document.setdefault("phases_completed", {})["implement"] = "2026-11-08T12:00:00"
        metrics._write_state(manager.state_file, document)
        manager.record_commit_range({"resolved_at": "raced"})
except metrics.StateLockTimeout:
    sys.exit(75)
"""


def _plan(tmp_path):
    """A plan whose `state.json` records review-plan complete."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")
    manager = StateManager(plan)
    manager.state["phases_completed"] = {"review-plan": SKIPPED_AT}
    manager.save()
    return plan, manager.state_file


def _state(state_file):
    return json.loads(Path(state_file).read_text(encoding="utf-8"))


class TestReopeningIsOneCriticalSection:
    def test_a_locked_writer_between_the_read_and_the_save_keeps_its_records(
        self, tmp_path, monkeypatch
    ):
        """Pause the reopen at its write and try a real competing writer.

        The child's zero-timeout lock attempt observes contention directly. If
        the reopen holds the lock, the child retries once it is released and
        reads the reopened file; a reopen that read outside the lock would let
        the child write first and then save its older snapshot over it.
        """
        plan, state_file = _plan(tmp_path)
        phase_dir = output_dir_path(plan) / "review-plan"
        phase_dir.mkdir(parents=True)

        def record_elsewhere():
            child = subprocess.run(
                [sys.executable, "-c", _LOCKED_WRITER, str(SKILL_ROOT), str(plan)],
                capture_output=True, text=True, timeout=30,
            )
            assert child.returncode in (0, 75), child.stderr
            return child.returncode

        first_attempt = []
        real_write = StateManager._write_atomic

        def write_with_a_competing_record(self, document):
            first_attempt.append(record_elsewhere())
            real_write(self, document)

        monkeypatch.setattr(StateManager, "_write_atomic", write_with_a_competing_record)
        rob.reopen_phase_for_validation(plan, "review-plan", phase_dir)
        if first_attempt == [75]:
            assert record_elsewhere() == 0

        state = _state(state_file)
        assert "review-plan" not in state["phases_completed"]
        assert state["phases_completed"]["implement"] == "2026-11-08T12:00:00"
        assert state["commit_range"]["resolved_at"] == "raced"
        assert first_attempt == [75]
        assert status_file.read_status(str(phase_dir))["state"] == "validation_pending"

    def test_nothing_is_written_when_the_phase_is_not_recorded_complete(self, tmp_path):
        plan, state_file = _plan(tmp_path)
        before = state_file.read_text(encoding="utf-8")
        phase_dir = output_dir_path(plan) / "review-tasks"
        phase_dir.mkdir(parents=True)

        rob.reopen_phase_for_validation(plan, "review-tasks", phase_dir)

        assert state_file.read_text(encoding="utf-8") == before
        assert status_file.read_status(str(phase_dir))["state"] == "validation_pending"


def _round(tmp_path, *, batches_on_disk: bool):
    """A review-plan phase skip-validation completed over a pending round."""
    plan, state_file = _plan(tmp_path)
    out_dir = output_dir_path(plan)
    phase_dir = out_dir / "review-plan"
    phase_dir.mkdir(parents=True)
    suggestions = [
        {"title": f"Finding {i}", "desc": "Some description", "importance": "MEDIUM",
         "reference": "Step 1", "type": "addition"}
        for i in range(3)
    ]
    (phase_dir / "model-a.json").write_text(json.dumps(suggestions), encoding="utf-8")
    grouped = [
        {"theme": s["title"], "category": "x", "models": ["model-a"], "suggestions": [s]}
        for s in suggestions
    ]
    stamp_stable_ids(grouped)
    (phase_dir / "grouped.json").write_text(
        json.dumps(save_groups_payload(grouped)), encoding="utf-8"
    )
    document = prepare_batched_validation_tasks(
        groups=grouped,
        context=plan.read_text(encoding="utf-8"),
        output_dir=str(phase_dir),
        plan_file=str(plan),
    )
    (phase_dir / "validation_tasks.json").write_text(json.dumps(document), encoding="utf-8")
    if batches_on_disk:
        for batch in document["batches"]:
            (phase_dir / f"validation_batch_{batch['batch_index']}.json").write_text(
                "[]", encoding="utf-8"
            )
    status_file.write_status(str(phase_dir), {"phase": "review-plan", "state": "complete"})
    return plan, out_dir, phase_dir, state_file


def _reaggregate(plan, out_dir, phase_dir):
    args = argparse.Namespace(
        skip_validation=False, validation_model="auto", timeout=None, force=False,
    )
    return asyncio.run(rob.reaggregate_from_existing_files(
        prefix="plan", out_dir=str(out_dir), phase_dir=str(phase_dir),
        phase_name="review-plan", plan_path=str(plan), args=args,
        orchestrator_script="review_plan_orchestrator.py",
    ))


class TestARoundThatRaisesLeavesThePhaseOpen:
    """The code-review lane's rule on the base lane: no broad `except` turns a
    failed dispatch or merge into a completed phase with no verdicts."""

    def test_a_dispatch_that_raises_after_the_reopen(self, tmp_path, capsys):
        plan, out_dir, phase_dir, state_file = _round(tmp_path, batches_on_disk=False)

        with patch.object(rob, "dispatch_subagent_tasks",
                          side_effect=OSError("executor pool died")):
            with pytest.raises(OSError, match="executor pool died"):
                _reaggregate(plan, out_dir, phase_dir)

        assert "review-plan" not in _state(state_file)["phases_completed"]
        assert status_file.read_status(str(phase_dir))["state"] == "validation_pending"
        assert not (phase_dir / "validation.json").exists()
        assert "Reaggregation complete!" not in capsys.readouterr().out

    def test_a_merge_that_raises_over_a_whole_round(self, tmp_path, capsys):
        plan, out_dir, phase_dir, state_file = _round(tmp_path, batches_on_disk=True)
        manager = StateManager(plan)
        manager.state["phases_completed"] = {}
        manager.save()

        with patch.object(rob, "merge_validation_batches",
                          side_effect=ValueError("unreadable batch")):
            with pytest.raises(ValueError, match="unreadable batch"):
                _reaggregate(plan, out_dir, phase_dir)

        assert "review-plan" not in _state(state_file)["phases_completed"]
        assert not (phase_dir / "validation.json").exists()
        assert "Reaggregation complete!" not in capsys.readouterr().out
