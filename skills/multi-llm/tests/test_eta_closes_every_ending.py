"""Every ending of a write phase closes its `[ETA]` lines, on both legs.

Review of fix round 20, #1: round 20 gave a stop short of its total one
closing line ("run stopped after batch N/M · K task(s) not run"), but the
ending where a check reports findings and holds the last tasks back called
none. Plan T001, T002 (edits) → T003 (a check) → T004, T005 behind it: the
run's last line read "batch 3/5 done in <1s · ~<1s remaining", the write
mirror said 3, and `metrics_progress.implement.total_batches` stayed 5. The
check's own line now leaves the tasks it holds out of the total (the native
leg's `util batch-done` takes them out before `util metrics record`), and
every ending logs one closing line that agrees with the mirror and the stored
total: a stop, a refusal, a check hold, a failed `driver.verify_command`, an
exception out of the loop, a detached worker's abort; a finished phase keeps
its "phase complete" line. On the native leg `util metrics finish` after a
run whose records kept tasks out says so instead of "phase complete".
"""

from __future__ import annotations

import json
import re

import pytest

from tests.test_check_gate_fixture import (  # noqa: F401  (the fixture)
    ALLOW,
    FINDINGS,
    _check,
    _policy,
    repo,
)
from tests.test_dispatch_counts import _emit, _etas, _util
from tests.test_implement_stop_account import _implement, make_repo
from utils import metrics
from utils import provider_registry as registry
from utils.state_manager import get_or_create_state

#: A check last, holding the last two tasks (the reviewer's repro).
TRAILING_CHECK = """# Trailing check

## Overview
A check last.

## Tasks

### T001: Edit: create the module
Create the module file.
- Depends on: none
- Files to modify: src/main.py

### T002: Edit: add tests
Write unit tests.
- Depends on: T001
- Files to modify: src/main.py

### T003: Check the module
Run the validator on the module. Read-only.
- Depends on: T002
- Files to modify: none

### T004: Edit: wire the fallback
Add the fallback.
- Depends on: T003
- Files to modify: src/main.py

### T005: Edit: wire the other fallback
Add the other fallback.
- Depends on: T003
- Files to modify: src/main.py
"""

#: Two edits and nothing else, for the endings that need no check.
TWO_EDITS = """# Two edits

## Overview
Two edits.

## Tasks

### T001: Edit: create the module
Create it.
- Depends on: none
- Files to modify: src/main.py

### T002: Edit: extend the module
Extend it.
- Depends on: T001
- Files to modify: src/main.py
"""


def _eta_lines(text: str) -> list:
    return [line for line in text.splitlines() if line.startswith("[ETA] implement:")]


def _run(repo, capsys, *flags) -> tuple:
    code = _implement(repo, *flags)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _stored_total(repo) -> int:
    state = get_or_create_state(repo.plan).state
    return state["metrics_progress"]["implement"]["total_batches"]


def _mirror(repo) -> dict:
    return json.loads(
        (repo.phase_dir / ".status.json").read_text(encoding="utf-8")
    )["progress"]["write_subagents"]


@pytest.fixture
def trailing(tmp_path, monkeypatch):
    obj = make_repo(tmp_path, monkeypatch, TRAILING_CHECK)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        [f"T00{n}" for n in range(1, 6)]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", "Title: Check the module")
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": ["retries must be 3 (found 2)"]}))
    yield obj
    registry._config = None
    registry._config_key = None


@pytest.fixture
def two_edits(tmp_path, monkeypatch):
    obj = make_repo(tmp_path, monkeypatch, TWO_EDITS)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001", "T002"]))
    yield obj
    registry._config = None
    registry._config_key = None


class TestACheckThatHoldsTheLastTasks:

    def test_its_own_line_and_a_closing_line_agree_with_the_mirror(
        self, trailing, capsys,
    ):
        code, text = _run(trailing, capsys)
        assert code == 3, text
        assert [l for l in text.splitlines() if l.startswith("Organized into")] == [
            "Organized into 5 execution batches: 5 task(s) to dispatch "
            "(T004, T005 wait on a check's verdict)"], text
        # T004 and T005 counted until T003 reported; its own line leaves them
        # out, so the run ends at N/N rather than "3/5 … remaining".
        assert _etas(text) == [(1, 5), (2, 5), (3, 3)], text
        lines = _eta_lines(text)
        assert "ALL BATCHES COMPLETE" in lines[-2], lines
        assert lines[-1] == (
            "[ETA] implement: run stopped after batch 3/3 · T004, T005 not "
            "run: check T003 reported problems"
        ), lines
        assert not any("remaining" in line for line in lines[-2:]), lines
        mirror = _mirror(trailing)
        assert (mirror["total"], mirror["ok"]) == (3, 3)
        assert _stored_total(trailing) == 3

    def test_a_hold_mid_run_closes_after_the_last_line(self, repo, capsys):
        """The check-gate fixture with the policy allowing T005: T007 is
        held, the other tasks run, and the run ends needing a human."""
        _policy(repo, ALLOW)
        _check(repo, passed=False)
        code, text = _run(repo, capsys)
        assert code == 3, text
        # T002's own line already leaves T007 out.
        assert _etas(text) == [(1, 7), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)], text
        assert _eta_lines(text)[-1] == (
            "[ETA] implement: run stopped after batch 6/6 · T007 not run: "
            "check T002 reported problems"
        ), text
        assert _stored_total(repo) == _mirror(repo)["total"] == 6


class TestTheOtherEndings:

    def test_a_failed_verify_command_closes_on_the_command(self, two_edits, capsys):
        (two_edits.tmp_path / "driver.yaml").write_text(
            "driver:\n  subagent_runner: codex:gpt-5.5\n  verify_command: \"false\"\n",
            encoding="utf-8")
        registry._config = None
        registry._config_key = None
        code, text = _run(two_edits, capsys)
        assert code == 3, text
        lines = _eta_lines(text)
        assert "ALL BATCHES COMPLETE" in lines[-2], lines
        assert lines[-1] == (
            "[ETA] implement: run stopped after batch 2/2 · `false` exited 1"
        ), lines
        assert "phase complete" not in text

    def test_a_finished_phase_keeps_its_one_closing_line(self, two_edits, capsys):
        code, text = _run(two_edits, capsys)
        assert code == 0, text
        lines = _eta_lines(text)
        assert lines[-1].startswith("[ETA] implement: phase complete in "), lines
        assert not any("run stopped" in line for line in lines), lines

    def test_a_stop_at_n_over_n_closes_too(self, two_edits, capsys):
        """A failed task that was the last of the total: "ALL BATCHES
        COMPLETE" is not the run's last word."""
        two_edits.monkeypatch.setenv(
            "MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Edit: extend the module")
        code, text = _run(two_edits, capsys)
        assert code == 3, text
        assert _etas(text) == [(1, 2), (2, 2)], text
        assert _eta_lines(text)[-1] == (
            "[ETA] implement: run stopped after batch 2/2"
        ), text

    def test_an_exception_out_of_the_loop_closes_them(
        self, two_edits, capsys, monkeypatch,
    ):
        from utils.write_batch_scheduler import WriteBatchScheduler, WriteSafetyError

        real = WriteBatchScheduler._run_batch

        def run_batch(self, batch_index, tasks, result):
            if any(task.get("task_id") == "T002" for task in tasks):
                raise WriteSafetyError("the phase folder was replaced")
            return real(self, batch_index, tasks, result)

        monkeypatch.setattr(WriteBatchScheduler, "_run_batch", run_batch)
        code, text = _run(two_edits, capsys)
        assert code == 3, text
        assert "Rollback safety refusal" in text
        assert _eta_lines(text)[-1] == (
            "[ETA] implement: run stopped after batch 1/2 · 1 task(s) not run"
        ), text

    def test_an_abort_asks_the_open_run_for_its_line(
        self, two_edits, capsys, monkeypatch,
    ):
        """The detached worker's signal handler cannot reach any ending of
        the run (it ends the process with `os._exit`), so it asks
        `metrics.aborted_eta_line` for the closing line while the run is
        open; once the run ended there is nothing to ask."""
        from utils.write_batch_scheduler import WriteBatchScheduler

        real = WriteBatchScheduler._run_batch
        asked = []

        def run_batch(self, batch_index, tasks, result):
            if any(task.get("task_id") == "T002" for task in tasks):
                asked.append(metrics.aborted_eta_line("SIGTERM"))
            return real(self, batch_index, tasks, result)

        monkeypatch.setattr(WriteBatchScheduler, "_run_batch", run_batch)
        code, _text = _run(two_edits, capsys)
        assert code == 0
        assert asked == [
            "[ETA] implement: run aborted (SIGTERM) after batch 1/2 · "
            "1 task(s) not run"
        ]
        assert metrics.aborted_eta_line("SIGTERM") is None


class TestTheAbortHandlerPrintsIt:

    def test_the_signal_handler_prints_the_open_runs_line(
        self, tmp_path, monkeypatch, capsys,
    ):
        import signal as _signal

        from utils import detach, process_control, status_file, stream_bootstrap

        monkeypatch.setattr(
            process_control, "install_worker_signal_handlers", lambda _cb: None)
        monkeypatch.setattr(stream_bootstrap, "streams_for_a_signal_handler", lambda: None)
        monkeypatch.setattr(detach, "_ACTIVE", None)
        monkeypatch.setattr(detach, "_sweep_live_clis", lambda: None)
        monkeypatch.setattr(detach, "_restore_codex_trust", lambda: None)
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase = tmp_path / "plan" / "implement"
        phase.mkdir(parents=True)
        handle = detach.RunHandle(
            str(phase), run_id="run-abort-eta", mode="implement",
            plan_file=str(plan), detached=False,
        )
        handle.start()

        def closer(cause):
            return f"[ETA] implement: run aborted ({cause}) after batch 2/5 · 3 task(s) not run"

        metrics.open_eta_run(closer)
        exits = []
        monkeypatch.setattr(detach.os, "_exit", exits.append)
        try:
            handle._on_signal(_signal.SIGTERM)
        finally:
            metrics.close_eta_run(closer)
        assert exits == [3]
        err = capsys.readouterr().err
        assert (
            "[ETA] implement: run aborted (SIGTERM) after batch 2/5 · "
            "3 task(s) not run"
        ) in err, err
        assert status_file.read_run(str(phase))["state"] == detach.RUN_STATE_ABORTED


class TestTheNativeFinishLine:
    """`util metrics finish` after the check-gate fixture's native run:
    T002 reported findings and `util batch-done` took T007 out."""

    def test_it_names_the_tasks_the_records_kept_out(self, repo, capsys):
        handoff, _out = _emit(repo, capsys)
        state_file = handoff["state_file"]
        total = str(handoff["tasks_to_dispatch"])
        assert _util(capsys, "metrics", "start", "--state-file", state_file,
                     "--phase", "implement", "--total-batches", total)[0] == 0
        for index, task_id in enumerate(("T001", "T002", "T003", "T004"), 1):
            extra = (
                ("--check", "findings", "--finding", FINDINGS[0])
                if task_id == "T002" else ()
            )
            code, _o, err = _util(
                capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
                "--task-id", task_id, "--status", "completed", *extra)
            assert code == 0, err
            code, _o, err = _util(
                capsys, "metrics", "record", "--state-file", state_file,
                "--phase", "implement", "--label", f"{task_id}: task",
                "--subagent-type", "agent", "--duration-ms", "1000",
                "--total-batches", total, "--batch-index", str(index))
            assert code == 0, err
        assert _etas(err)[-1] == (4, 4)
        code, _o, err = _util(
            capsys, "metrics", "finish", "--state-file", state_file,
            "--phase", "implement")
        assert code == 0, err
        line = [l for l in err.splitlines() if l.startswith("[ETA] implement:")]
        assert len(line) == 1, err
        assert re.fullmatch(
            r"\[ETA\] implement: run stopped after batch 4/4 in [^·]+ · T007 not run",
            line[0],
        ), line
        assert "phase complete" not in err

    def test_a_run_that_kept_nothing_out_still_says_phase_complete(self, tmp_path):
        state = {"metrics": {}}
        metrics.start_phase_in_state(state, "implement", 2)
        _elapsed, line = metrics.finish_phase_in_state(state, "implement")
        assert "phase complete in" in line
