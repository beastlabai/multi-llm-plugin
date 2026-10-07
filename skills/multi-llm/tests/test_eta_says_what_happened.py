"""The ``[ETA]`` line of a failed task says it failed.

Fix round 24, from live run 10 (2026-10-07):

* #5 (OpenCode racing 06 and 06b): T002 timed out at its 1200 s slot
  budget, and the line read "task 2/11 done in 20m 5s · ~109m 30s
  remaining"; on the resume it timed out again after editing ``index.html``,
  and the line read "task 1/10 done in 20m 5s · ~133m 16s remaining". It
  said "done" for a task that failed, and the remaining time averaged the
  20-minute timeout in. The line of a failed task now says it failed, how,
  and whether its edits were half-applied, on both legs, and no
  remaining-time estimate counts a failed task as progress.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.test_check_gate_fixture import repo  # noqa: F401  (the fixture)
from tests.test_dispatch_counts import _emit, _util
from tests.test_implement_stop_account import _implement, make_repo
from utils import metrics
from utils import provider_registry as registry
from utils.retry_ladder import STATUS_TIMEOUT
from utils.subagent_executor import Attempt, TaskOutcome

#: Three edits, so a failed T002 is neither the first task nor the last.
THREE_EDITS = """# Three edits

## Overview
Three edits.

## Tasks

### T001: Edit: create the module
Create it.
- Depends on: none
- Files to modify: src/main.py

### T002: Edit: extend the module
Extend it.
- Depends on: T001
- Files to modify: src/main.py

### T003: Edit: finish the module
Finish it.
- Depends on: T002
- Files to modify: src/main.py
"""

def _eta_lines(text: str) -> list:
    return [line for line in text.splitlines() if line.startswith("[ETA] implement:")]


def _run(repo_obj, capsys, *flags) -> tuple:
    code = _implement(repo_obj, *flags)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


@pytest.fixture
def three_edits(tmp_path, monkeypatch):
    obj = make_repo(tmp_path, monkeypatch, THREE_EDITS)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001", "T002", "T003"]))
    yield obj
    registry._config = None
    registry._config_key = None


def _time_out(monkeypatch, task_id: str, *, edit: bool) -> None:
    """Make *task_id*'s runner time out after 20 minutes, the way big-pickle
    did in live run 10 (``edit``: after changing ``src/main.py``, the 06b
    partial application)."""
    from utils.write_batch_scheduler import WriteBatchScheduler

    real = WriteBatchScheduler._dispatch_executor

    def dispatch(self, task, prompt, snapshot, profile, declared, **kwargs):
        if task.get("task_id") != task_id:
            return real(self, task, prompt, snapshot, profile, declared, **kwargs)
        if edit:
            path = Path(self.repo_root) / "src" / "main.py"
            path.write_text(path.read_text() + "# half of it\n")
        outcome = TaskOutcome(
            task_id=task_id, kind=self.kind, runner="codex:gpt-5.5",
            status=STATUS_TIMEOUT, profile=profile, duration_seconds=1205.6,
            reason="the runner exceeded its per-task timeout and was killed",
            attempts=[Attempt(index=1, status=STATUS_TIMEOUT, duration_seconds=1205.6)],
        )
        outcome.ran_out_of_time = edit
        return outcome

    monkeypatch.setattr(WriteBatchScheduler, "_dispatch_executor", dispatch)


class TestAFailedTaskSaysSoOnTheExecutor:

    def test_a_timeout_reads_failed_not_done(self, three_edits, capsys, monkeypatch):
        _time_out(monkeypatch, "T002", edit=False)
        code, text = _run(three_edits, capsys)
        assert code == 3, text
        lines = _eta_lines(text)
        assert lines[0].startswith("[ETA] implement: task 1/3 done in "), lines
        assert lines[1] == "[ETA] implement: task 2/3 failed (timeout) after 20m 5s", lines
        assert not any("task 2/3 done" in line for line in lines), lines
        assert lines[-1] == "[ETA] implement: run stopped after task 2/3 · T003 not run", lines

    def test_a_half_applied_timeout_says_its_edits_are_kept(
        self, three_edits, capsys, monkeypatch,
    ):
        _time_out(monkeypatch, "T002", edit=True)
        code, text = _run(three_edits, capsys)
        assert code == 3, text
        assert "partial application" in text, text
        lines = _eta_lines(text)
        assert lines[1] == (
            "[ETA] implement: task 2/3 failed (timeout) after 20m 5s · "
            "half-applied: its edits are kept"
        ), lines

    def test_a_failure_the_runner_reported_names_its_status(
        self, three_edits, capsys,
    ):
        """The runner said it failed and changed nothing: a clean failure."""
        three_edits.monkeypatch.setenv(
            "MOCK_LLM_EDITS_IF_PROMPT_HAS", "Title: Edit: create the module")
        three_edits.monkeypatch.setenv(
            "MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Edit: extend the module")
        code, text = _run(three_edits, capsys)
        assert code == 3, text
        lines = _eta_lines(text)
        assert re.fullmatch(
            r"\[ETA\] implement: task 2/3 failed \(batch failed\) after [^·]+",
            lines[1],
        ), text

    def test_a_runner_that_failed_after_editing_is_half_applied(
        self, three_edits, capsys,
    ):
        three_edits.monkeypatch.setenv(
            "MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Edit: extend the module")
        code, text = _run(three_edits, capsys)
        assert code == 3, text
        lines = _eta_lines(text)
        assert re.fullmatch(
            r"\[ETA\] implement: task 2/3 failed( \([^)]+\))? after [^·]+ · "
            r"half-applied: its edits are kept",
            lines[1],
        ), text
        assert "remaining" not in lines[1], lines


def _native_state(total: int, started_s_ago: float) -> dict:
    state = {"metrics": {}, "task_status": {}}
    metrics.start_phase_in_state(state, "implement", total)
    progress = state["metrics_progress"]["implement"]
    progress["started_at"] = (
        datetime.now() - timedelta(seconds=started_s_ago)
    ).isoformat()
    return state


def _fail(state: dict, task_id: str, reason: str) -> None:
    """What ``util batch-done --status failed --reason …`` records."""
    state["task_status"][task_id] = "failed"
    state.setdefault("task_status_reasons", {})[task_id] = reason
    metrics.mark_failed_in_progress(state, "implement", task_id, failed=True)


class TestAFailedTaskSaysSoOnTheNativeLeg:

    def test_the_record_after_a_failed_batch_done_reads_failed(self):
        state = _native_state(11, 1500)
        metrics.record_metric_in_state(
            state, "implement", "T001: Scaffold", duration_ms=254_000,
            total_batches=11, batch_index=1)
        _fail(state, "T002", "the subagent timed out after 20 minutes")
        line = metrics.record_metric_in_state(
            state, "implement", "T002: Build the track", duration_ms=1_205_600,
            total_batches=11, batch_index=2)
        assert line == (
            "[ETA] implement: task 2/11 failed (the subagent timed out after "
            "20 minutes) after 20m 5s"
        ), line

    def test_a_partial_record_says_half_applied(self):
        state = _native_state(10, 1300)
        # `util batch-done --status failed --partial` prefixes the reason.
        _fail(state, "T002", "half-applied: the subagent timed out")
        line = metrics.record_metric_in_state(
            state, "implement", "T002: Build the track", duration_ms=1_205_600,
            total_batches=10, batch_index=1)
        assert line == (
            "[ETA] implement: task 1/10 failed (the subagent timed out) after "
            "20m 5s · half-applied: its edits are kept"
        ), line

    def test_the_estimate_never_counts_the_failed_task(self):
        """T001 and T003 took a minute each, T002 timed out at 20: two tasks
        are left, so about two minutes, not the 14m 40s an average over the
        timeout gave."""
        state = _native_state(5, 1320)
        metrics.record_metric_in_state(
            state, "implement", "T001: a", duration_ms=60_000,
            total_batches=5, batch_index=1)
        _fail(state, "T002", "timeout")
        metrics.record_metric_in_state(
            state, "implement", "T002: b", duration_ms=1_200_000,
            total_batches=5, batch_index=2)
        line = metrics.record_metric_in_state(
            state, "implement", "T003: c", duration_ms=60_000,
            total_batches=5, batch_index=3)
        assert line == (
            "[ETA] implement: task 3/5 done in 1m 0s · ~2m 0s remaining"
        ), line

    def test_an_apply_batch_is_named_by_the_task_id_flag(self, tmp_path, capsys):
        """An apply phase's label is ``Batch {N} ({key})``: the host passes
        the batch's task id with ``--task-id``."""
        state = {"metrics": {}, "task_status": {}}
        metrics.start_phase_in_state(state, "apply-code-fixes", 2)
        state["task_status"]["apply-fix-batch-0"] = "failed"
        state["task_status_reasons"] = {"apply-fix-batch-0": "the runner exited 1"}
        metrics.mark_failed_in_progress(
            state, "apply-code-fixes", "apply-fix-batch-0", failed=True)
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state), encoding="utf-8")
        code, _out, err = _util(
            capsys, "metrics", "record", "--state-file", str(state_file),
            "--phase", "apply-code-fixes", "--label", "Batch 1 (src/main.py)",
            "--task-id", "apply-fix-batch-0", "--duration-ms", "4000",
            "--total-batches", "2", "--batch-index", "1")
        assert code == 0, err
        assert (
            "[ETA] apply-code-fixes: batch 1/2 failed (the runner exited 1) "
            "after 0m 4s"
        ) in err, err
        entry = json.loads(state_file.read_text())["metrics"]["apply-code-fixes"][0]
        assert entry["task_id"] == "apply-fix-batch-0"
        assert entry["failed"] is True


class TestTheNativeRecordsReadTheBatchDoneRecord:
    """Through the CLI, as ``instructions/implement.md`` step 5 runs it."""

    def test_a_partial_record_reads_half_applied_on_its_line(self, repo, capsys):
        handoff, _out = _emit(repo, capsys)
        state_file = handoff["state_file"]
        total = str(handoff["tasks_to_dispatch"])
        assert _util(capsys, "metrics", "start", "--state-file", state_file,
                     "--phase", "implement", "--total-batches", total)[0] == 0
        code, _o, err = _util(
            capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
            "--task-id", "T001", "--status", "failed", "--partial",
            "--reason", "the subagent timed out after 20 minutes")
        assert code == 0, err
        code, _o, err = _util(
            capsys, "metrics", "record", "--state-file", state_file,
            "--phase", "implement", "--label", "T001: Create the configuration",
            "--task-id", "T001", "--duration-ms", "1205600",
            "--total-batches", total, "--batch-index", "1")
        assert code == 0, err
        # Every other task depends on T001, so the host skips them.
        assert _eta_lines(err) == [
            "[ETA] implement: task 1/1 failed (the subagent timed out after 20 "
            "minutes) after 20m 5s · half-applied: its edits are kept"
        ], err

    def test_the_half_applied_prefix_is_the_schedulers(self):
        from utils.write_batch_scheduler import HALF_APPLIED_PREFIX

        assert metrics.HALF_APPLIED_PREFIX == HALF_APPLIED_PREFIX
