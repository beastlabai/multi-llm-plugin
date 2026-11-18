"""`--fresh` voids the earlier runs' resume evidence, not just their progress.

`--fresh` cleared the processed records, the decisions and the prior-changes
log, but every run resumes from the batch result files, which stayed valid.
Whenever the `--fresh` run did not itself replace a batch's result — a
`--fresh --dry-run`, a failure before the first dispatch, or a `--fresh` run
that stopped part-way — the next run kept the reverted batches as done
(measured 2026-11-18). The reset is now stamped in `state.json`, in the save
that clears the records (`utils/fresh_reset.py`): nothing on disk moves.

The executor leg over the mock runner (`_ApplyRig`): three HIGH suggestions in
three sections, so three one-item batches, each appending its `content` line.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.support.write_orchestrators import (  # noqa: E402
    _ApplyRig,
    orchestrator_repo,  # noqa: F401 - the repo fixture
)

RESET_KEY = "fresh_reset_at_apply_suggestions"


def _exit(call) -> int:
    """`--dry-run` ends with `sys.exit(0)`; a run returns its code."""
    try:
        return call()
    except SystemExit as exc:
        return int(exc.code or 0)


def _dispatched(log: Path) -> int:
    if not log.exists():
        return 0
    return sum(1 for line in log.read_text().splitlines() if json.loads(line).get("prompt"))


def _state(plan):
    from utils.state_manager import get_or_create_state
    return get_or_create_state(plan)


class TestFreshVoidsTheEarlierEvidence(_ApplyRig):

    def _completed_then_reverted(self, repo):
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        original = plan.read_text(encoding="utf-8")
        assert self._run(repo, plan, sum(ids, []), content="run1\n") == 0
        plan.write_text(original, encoding="utf-8")
        return plan, ids, original

    def test_a_fresh_preview_then_a_plain_run_applies_every_batch(self, repo):
        plan, ids, original = self._completed_then_reverted(repo)
        assert _exit(lambda: self._run(repo, plan, sum(ids, []), "--fresh", "--dry-run")) == 0
        log = repo.tmp_path / "after-preview.jsonl"
        repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
        assert self._run(repo, plan, sum(ids, []), content="run2\n") == 0
        assert _dispatched(log) == 3
        assert plan.read_text(encoding="utf-8") == original + "run2\n" * 3

    def test_a_fresh_preflight_then_a_resume_applies_every_batch(self, repo):
        plan, ids, original = self._completed_then_reverted(repo)
        assert self._run(repo, plan, sum(ids, []), "--fresh", "--dry-run-subagents") == 0
        assert self._run(repo, plan, sum(ids, []), "--resume", content="run2\n") == 0
        assert plan.read_text(encoding="utf-8") == original + "run2\n" * 3

    def test_a_fresh_run_that_fails_before_dispatch_leaves_nothing_to_resume(self, repo):
        import apply_suggestions_orchestrator as orchestrator

        plan, ids, original = self._completed_then_reverted(repo)

        def boom(self, items):
            raise RuntimeError("killed before the first dispatch")

        with repo.monkeypatch.context() as patch:
            patch.setattr(orchestrator.ApplySuggestionsOrchestrator, "create_batches", boom)
            assert self._run(repo, plan, sum(ids, []), "--fresh", content="never\n") != 0
        assert self._run(repo, plan, sum(ids, []), "--resume", content="run2\n") == 0
        assert plan.read_text(encoding="utf-8") == original + "run2\n" * 3

    def test_a_fresh_run_that_stops_part_way_resumes_only_its_own_batches(self, repo):
        plan, ids, original = self._completed_then_reverted(repo)
        # Batch 0 lands; batch 1 writes its line but attests to nothing it was
        # given, a partial application whose edits are kept: the loop stops.
        assert self._run(repo, plan, ids[0], "--fresh", content="run2\n") == 3
        assert self._run(repo, plan, sum(ids, []), "--resume", content="run3\n") == 0
        # Batch 1 is dispatched again and batch 2, which the --fresh run never
        # reached, for the first time: run 1's result for it no longer counts.
        assert plan.read_text(encoding="utf-8") == original + "run2\n" * 2 + "run3\n" * 2
        rows = self._report(plan)
        assert [(r["task_id"], bool(r.get("from_earlier_run"))) for r in rows] == [
            ("apply-batch-0", True), ("apply-batch-1", False), ("apply-batch-2", False)]

    def test_nothing_on_disk_moves_and_the_reset_is_stamped(self, repo):
        plan, ids, _ = self._completed_then_reverted(repo)
        phase_dir = self._phase_dir(plan)
        before = sorted(p.name for p in phase_dir.iterdir())
        assert _exit(lambda: self._run(repo, plan, sum(ids, []), "--fresh", "--dry-run")) == 0
        assert len(list(phase_dir.glob("apply_batch_*.json"))) == 3
        assert set(before) <= {p.name for p in phase_dir.iterdir()}
        assert not list(phase_dir.glob("*.rerun-*"))
        assert _state(plan).state.get(RESET_KEY)

    def test_the_phase_task_records_go_back_to_pending(self, repo):
        from utils.state_manager import TASK_RESET_AT_KEY

        plan, ids, _ = self._completed_then_reverted(repo)
        state = _state(plan)
        state.update_task_status("apply-batch-1", "failed", "a reason")
        state.update_task_status("T001", "completed")
        state.save()
        assert _exit(lambda: self._run(repo, plan, sum(ids, []), "--fresh", "--dry-run")) == 0
        state = _state(plan)
        statuses = {k: v for k, v in state.get_all_task_statuses().items()
                    if k.startswith("apply-batch-")}
        assert statuses == {f"apply-batch-{n}": "pending" for n in range(3)}
        assert state.get_task_status_reason("apply-batch-1") is None
        assert state.get_task_status("T001") == "completed"
        assert set(state.state[TASK_RESET_AT_KEY]) == set(statuses)

    def test_a_refused_fresh_resets_nothing(self, repo):
        """The plan still carries run 1's edits: `--fresh` is refused before
        anything is cleared, and the next run still keeps every batch."""
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, sum(ids, []), content="run1\n") == 0
        applied = plan.read_text(encoding="utf-8")
        assert _exit(lambda: self._run(repo, plan, sum(ids, []), "--fresh", "--dry-run")) == 3
        assert RESET_KEY not in _state(plan).state
        assert self._run(repo, plan, sum(ids, []), content="run2\n") == 0
        assert plan.read_text(encoding="utf-8") == applied

    def test_a_reset_whose_save_fails_resets_nothing(self, repo):
        """The stamp, the clears and the task records are one save: when it
        fails, none of them happened, and the next plain run keeps every
        batch as the state before it said."""
        from utils import state_manager

        plan, ids, original = self._completed_then_reverted(repo)
        real_save = state_manager.StateManager.save

        def failing_save(self):
            if RESET_KEY in self.state:
                raise IOError("disk full")
            return real_save(self)

        with repo.monkeypatch.context() as patch:
            patch.setattr(state_manager.StateManager, "save", failing_save)
            code = _exit(lambda: self._run(repo, plan, sum(ids, []), "--fresh", content="x\n"))
            assert code != 0
        state = _state(plan)
        assert RESET_KEY not in state.state
        assert state.get_processed_items("apply-suggestions")
        log = repo.tmp_path / "after-failed-reset.jsonl"
        repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
        assert self._run(repo, plan, sum(ids, []), content="run2\n") == 0
        assert _dispatched(log) == 0
        assert plan.read_text(encoding="utf-8") == original

    def test_fresh_beside_resume_resets_nothing(self, repo):
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, ids[0], content="run1\n") == 3
        assert self._run(repo, plan, sum(ids, []), "--resume", "--fresh", content="run2\n") == 0
        assert RESET_KEY not in _state(plan).state
        rows = self._report(plan)
        assert [(r["task_id"], bool(r.get("from_earlier_run"))) for r in rows] == [
            ("apply-batch-0", True), ("apply-batch-1", False), ("apply-batch-2", False)]

    def test_a_phase_never_reset_keeps_its_inputs_binding(self, repo):
        """No reset recorded: the binding hashes exactly the fields it always
        did, so a run in flight across this change resumes as before."""
        from utils.subagent_tasks import declared_file_set

        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, sum(ids, []), content="run1\n") == 0
        phase_dir = self._phase_dir(plan)
        tasks = json.loads((phase_dir / "write_tasks.json").read_text())["batches"]
        recorded = json.loads((phase_dir / ".write_task_inputs.json").read_text())["tasks"]
        assert set(recorded) == {t["task_id"] for t in tasks}
        for task in tasks:
            payload = json.dumps({
                "kind": "apply_suggestions",
                "prompt": str(task.get("prompt") or ""),
                "expected_ids": [str(i) for i in task["expected_ids"]],
                "declared_files": sorted(declared_file_set(task)),
                "batch_index": int(task.get("batch_index", 0)),
            }, sort_keys=True)
            assert recorded[task["task_id"]] == hashlib.sha256(payload.encode()).hexdigest()

    def test_an_earlier_runs_snapshot_is_not_a_reset_runs_rollback_point(self, repo):
        """Run 1's slot for batch 2, which the `--fresh` run never reached,
        is set aside on the resume and batch 2 takes its own snapshot of the
        bytes the resume found. The stamps are backdated so the review, run 1
        and the reset each have a second of their own (snapshot manifests are
        stamped to the second)."""
        plan, ids, original = self._completed_then_reverted(repo)
        state = _state(plan)
        completed = dict(state.state["phases_completed"])
        completed["review-plan"] = "2026-10-01T10:00:00"
        state.set("phases_completed", completed)
        state.save()
        pre_batch = self._phase_dir(plan) / "pre_batch"
        for manifest in pre_batch.glob("*/manifest.json"):
            data = json.loads(manifest.read_text())
            data["taken_at"] = "2026-10-02T10:00:00"
            manifest.write_text(json.dumps(data), encoding="utf-8")

        assert self._run(repo, plan, ids[0], "--fresh", content="run2\n") == 3
        assert self._run(repo, plan, sum(ids, []), "--resume", content="run3\n") == 0
        slot = next((pre_batch / "apply-batch-2" / "files").iterdir())
        assert slot.read_text(encoding="utf-8") == original + "run2\n" * 2 + "run3\n"
        aside = list(pre_batch.glob("apply-batch-2.items-*"))
        assert len(aside) == 1
        assert next((aside[0] / "files").iterdir()).read_text(
            encoding="utf-8") == original + "run1\n" * 2

    def _hold_the_phase_lock(self, plan):
        from utils import detach

        skill = Path(__file__).resolve().parent.parent
        holder = subprocess.Popen([
            sys.executable, "-c",
            "import os, sys, time; sys.path.insert(0, sys.argv[1]);"
            "from utils import process_control as pc;"
            "lock = pc.PhaseLock(sys.argv[2]);"
            "lock.acquire({'pid': os.getpid(), 'holder': 'test'});"
            "print('held', flush=True); time.sleep(60)",
            str(skill), detach.lock_path(str(self._phase_dir(plan))),
        ], stdout=subprocess.PIPE, text=True)
        if holder.stdout.readline().strip() != "held":
            holder.kill()
            raise RuntimeError("the lock holder did not start")
        return holder

    def test_fresh_is_refused_while_another_process_holds_the_phase(self, repo):
        plan, ids, _ = self._completed_then_reverted(repo)
        holder = self._hold_the_phase_lock(plan)
        try:
            code = _exit(lambda: self._run(repo, plan, sum(ids, []), "--fresh", "--dry-run"))
            assert code == 3
            assert RESET_KEY not in _state(plan).state
            assert _state(plan).get_processed_items("apply-suggestions")
        finally:
            holder.kill()
            holder.wait(timeout=10)

    def _bare_run(self, plan, **flags):
        import argparse

        import apply_suggestions_orchestrator as orchestrator

        run = orchestrator.ApplySuggestionsOrchestrator(argparse.Namespace(
            fresh=True, resume=False, **flags))
        run.out_dir = str(plan.parent / plan.stem)
        run.plan_path = str(plan)
        return run

    def test_a_detached_worker_is_not_refused_by_the_lock_its_launcher_handed_it(
        self, repo, monkeypatch,
    ):
        """A detached `--fresh` worker runs `_setup` before it adopts the lock
        its launcher handed it, so the launcher's label is still on the lock:
        that hold is this run's own (POSIX hand-over), matched by the lock
        file's identity."""
        import os

        from utils import detach
        from utils.apply_orchestrator_base import OrchestratorError

        plan, _ids, _ = self._completed_then_reverted(repo)
        holder = self._hold_the_phase_lock(plan)
        handed = os.open(detach.lock_path(str(self._phase_dir(plan))), os.O_RDONLY)
        try:
            run = self._bare_run(plan, internal_worker=True)
            monkeypatch.setattr(detach, "_HANDED_LOCK_FD", handed)
            run._refuse_fresh_under_a_live_run()
            # A descriptor of another file is not this phase's lock.
            other = os.open(str(plan), os.O_RDONLY)
            monkeypatch.setattr(detach, "_HANDED_LOCK_FD", other)
            with pytest.raises(OrchestratorError):
                run._refuse_fresh_under_a_live_run()
            os.close(other)
            # No handed lock at all: the hold is another run's.
            monkeypatch.setattr(detach, "_HANDED_LOCK_FD", None)
            monkeypatch.delenv(detach.LOCK_FD_ENV, raising=False)
            with pytest.raises(OrchestratorError):
                run._refuse_fresh_under_a_live_run()
        finally:
            os.close(handed)
            holder.kill()
            holder.wait(timeout=10)

    def test_a_lost_fingerprint_write_after_a_reset_keeps_the_landed_batch(self, repo):
        """Batch 0 lands after `--fresh`, but the fingerprint file's write is
        lost, so it still holds run 1's entry for `apply-batch-0`. The result's
        own verdict stamp is bound after the reset and wins: batch 0 is kept."""
        from utils import write_resume

        plan, ids, original = self._completed_then_reverted(repo)
        with repo.monkeypatch.context() as patch:
            patch.setattr(write_resume.ResumeEvidence, "record_fingerprint",
                          lambda self, task: None)
            assert self._run(repo, plan, ids[0], "--fresh", content="run2\n") == 3
        assert self._run(repo, plan, sum(ids, []), "--resume", content="run3\n") == 0
        assert plan.read_text(encoding="utf-8") == original + "run2\n" * 2 + "run3\n" * 2


class TestRecordFreshReset:
    """`fresh_reset.record_fresh_reset` and `cycle_cutoff` per apply kind."""

    @pytest.mark.parametrize("kind,task_id,other", [
        ("apply_suggestions", "apply-batch-0", "apply-task-batch-0"),
        ("apply_task_suggestions", "apply-task-batch-0", "apply-batch-0"),
        ("apply_code_fixes", "apply-fix-batch-0", "apply-batch-0"),
    ])
    def test_each_kind_resets_only_its_own_records(self, tmp_path, kind, task_id, other):
        from utils import fresh_reset
        from utils.state_manager import TASK_RESET_AT_KEY, get_or_create_state

        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        state = get_or_create_state(plan)
        state.update_task_status(task_id, "completed")
        state.update_task_status(other, "completed")
        state.update_task_status("T001", "completed")

        assert fresh_reset.record_fresh_reset(state, kind) == [task_id]
        assert state.get_task_status(task_id) == "pending"
        assert state.get_task_status(other) == "completed"
        assert state.get_task_status("T001") == "completed"
        assert list(state.state[TASK_RESET_AT_KEY]) == [task_id]
        state.save()
        assert fresh_reset.reset_stamp(get_or_create_state(plan).state, kind)

    def test_another_kind_records_nothing(self, tmp_path):
        from utils import fresh_reset
        from utils.state_manager import get_or_create_state

        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        state = get_or_create_state(plan)
        assert fresh_reset.record_fresh_reset(state, "implement") == []
        assert fresh_reset.reset_stamp(state.state, "implement") is None

    def test_the_cutoff_is_the_later_of_review_and_reset(self):
        from datetime import datetime
        from utils import fresh_reset

        key = fresh_reset.reset_key("apply_code_fixes")
        early, late = datetime(2026, 10, 1), datetime(2026, 11, 1)
        assert fresh_reset.cycle_cutoff(early, {}, "apply_code_fixes") == early
        assert fresh_reset.cycle_cutoff(None, {}, "apply_code_fixes") is None
        recorded = {key: late.isoformat()}
        assert fresh_reset.cycle_cutoff(early, recorded, "apply_code_fixes") == late
        assert fresh_reset.cycle_cutoff(None, recorded, "apply_code_fixes") == late
        recorded = {key: early.isoformat()}
        assert fresh_reset.cycle_cutoff(late, recorded, "apply_code_fixes") == late
