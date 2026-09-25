"""A no-op ``--reaggregate`` does not re-print the previous run's ending (2026-09-25).

``PhaseProgress.render_line`` reads ``progress`` off ``.status.json``, and the
teardown tick prints it. A ``--reaggregate`` that refused (exit 3: an input or
the validation snapshot changed) or only rematerialized validation (the
``[VALIDATION_BATCHES_PENDING]`` hand-off, nothing dispatched) wrote no
``progress.step`` of its own, so its last line was the previous round's
``[PROGRESS] step: complete | subagents 16/16 done (16 ok)`` — a completion
this invocation never reached, on the line a host reads last.

Now: a refusal prints no teardown line, and a rematerialization writes
``step: validation_pending`` and clears the previous round's executor mirror
before the teardown line reads them.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import progress_tracker  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils import status_file  # noqa: E402
from utils.progress_tracker import PhaseProgress  # noqa: E402

#: The previous round's ending, as the dogfood run's checkpoint held it.
PREVIOUS_ROUND = {
    "step": "complete",
    "subagents": {"kind": "validation", "total": 16, "ok": 16, "counts": {"ok": 16}},
}


def _phase(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    phase_dir = out_dir / "review-plan"
    phase_dir.mkdir(parents=True)
    (phase_dir / "model-a.json").write_text(json.dumps([{
        "title": "Finding", "desc": "Description", "importance": "MEDIUM",
        "reference": "Step 1", "type": "addition",
    }]), encoding="utf-8")
    status_file.write_status(str(phase_dir), {
        "phase": "review-plan", "state": "complete",
        "models_requested": ["model-a"], "models_completed": ["model-a"],
        "models_failed": {}, "plan_hash": "deadbeefcafe",
    })
    return plan, out_dir, phase_dir


def _stamp_previous_round(phase_dir):
    status_file.update_status(str(phase_dir), {"progress": dict(PREVIOUS_ROUND)})


def _reaggregate(plan, out_dir, phase_dir):
    args = argparse.Namespace(
        skip_validation=False, validation_model="auto", timeout=None,
        internal_validation=False, force=False,
    )
    return asyncio.run(rob.reaggregate_from_existing_files(
        prefix="plan", out_dir=str(out_dir), phase_dir=str(phase_dir),
        phase_name="review-plan", plan_path=str(plan), args=args,
        orchestrator_script="review_plan_orchestrator.py",
    ))


@pytest.fixture
def emitted():
    """Every ``[PROGRESS]`` line, with the interval ticks parked so only the
    teardown line — the one under test — can be printed."""
    lines: list = []

    async def parked(_delay):
        await asyncio.sleep(3600)

    with patch.object(progress_tracker, "_emit", lines.append), \
         patch.object(progress_tracker, "_tick_sleep", parked), \
         patch("utils.provider_registry.get_progress_interval", return_value=5):
        yield lines


class TestARematerializationShowsThePendingStep:
    def test_the_last_line_is_validation_pending_without_the_old_counters(
        self, tmp_path, emitted
    ):
        plan, out_dir, phase_dir = _phase(tmp_path)
        _stamp_previous_round(phase_dir)

        _reaggregate(plan, out_dir, phase_dir)

        assert (phase_dir / "validation_tasks.json").exists(), "nothing rematerialized"
        assert emitted, "the rematerialization printed no teardown line"
        assert emitted[-1].startswith("[PROGRESS] step: validation_pending"), emitted
        assert not any("step: complete" in line for line in emitted), emitted
        assert not any("16/16" in line for line in emitted), emitted
        progress = status_file.read_status(str(phase_dir))["progress"]
        assert progress["step"] == "validation_pending"
        assert not progress.get("subagents")


class TestARefusalPrintsNoCompletion:
    def _materialized_round(self, tmp_path):
        plan, out_dir, phase_dir = _phase(tmp_path)
        # First invocation materializes the round, then the host's 16
        # subagents finish it: the checkpoint the refused call inherits.
        with patch.object(progress_tracker, "_emit", lambda _line: None):
            _reaggregate(plan, out_dir, phase_dir)
        assert (phase_dir / "validation_tasks.json").exists()
        _stamp_previous_round(phase_dir)
        return plan, out_dir, phase_dir

    def test_a_changed_snapshot_refusal(self, tmp_path, emitted):
        plan, out_dir, phase_dir = self._materialized_round(tmp_path)

        def snapshot_changed(*_a, **_k):
            print("validation snapshot changed; re-run validation", file=sys.stderr)
            sys.exit(3)

        with patch.object(rob, "verify_round_context", snapshot_changed), \
             pytest.raises(SystemExit) as exited:
            _reaggregate(plan, out_dir, phase_dir)

        assert exited.value.code == 3
        assert not any("step: complete" in line for line in emitted), emitted
        assert not any("16/16" in line for line in emitted), emitted
        # The refusal wrote nothing: the previous round's record is intact.
        assert status_file.read_status(str(phase_dir))["progress"]["step"] == "complete"

    def test_an_edited_plan_refusal(self, tmp_path, emitted):
        plan, out_dir, phase_dir = self._materialized_round(tmp_path)
        plan.write_text("# Plan\n\nDo a different thing.\n", encoding="utf-8")

        with pytest.raises(SystemExit) as exited:
            _reaggregate(plan, out_dir, phase_dir)

        assert exited.value.code == 3
        assert not any("step: complete" in line for line in emitted), emitted


class TestThePhaseTickerSeam:
    def test_final_line_false_drops_only_the_teardown_line(self, tmp_path, emitted):
        status_file.write_status(str(tmp_path), {"progress": dict(PREVIOUS_ROUND)})

        async def run(final_line):
            ticker = PhaseProgress(tmp_path, interval_s=5)
            ticker.start_ticker()
            await asyncio.sleep(0)
            await ticker.stop_ticker(final_line=final_line)

        asyncio.run(run(False))
        assert emitted == []
        asyncio.run(run(True))
        assert len(emitted) == 1 and emitted[0].startswith("[PROGRESS] step: complete")
