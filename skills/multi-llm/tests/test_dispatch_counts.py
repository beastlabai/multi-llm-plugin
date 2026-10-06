"""The counts implement prints: what a run dispatches, on both legs.

Fix round 18, from live run 7 on the check-gate fixture (a check T002 that
reports problems, T007 behind it, T005 refused by the write policy, T006
after T005):

* Claude Code stop 01 (#2): the native emit said "Organized into 3 execution
  batches: 5 task(s) to dispatch (T007 waits on a check's verdict)" and "pass
  --total-batches 5". The ``[ETA]`` line counts one ``util metrics record``
  per dispatched task, so five was the right unit, but T002 reported problems,
  T007 was held, and the line stopped at "batch 4/5" for good. The executor's
  own ``[ETA]`` lines did the same with an accepted or a held task.
* Codex and OpenCode stop resumes (#3): every resume printed "Organized into 4
  execution batches: 4 task(s) to dispatch" — also when it dispatched nothing
  (01b, 01c) and when it dispatched two (fix 02) — and the status poll read
  "implement 1/4 batches ok, 0/4 items applied" for a resume that ran nothing.

These tests drive the real orchestrators against the mock runner, through the
fixture of ``test_check_gate_fixture.py``.
"""

from __future__ import annotations

import json
import re

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: F401  (the fixture)
    ALLOW,
    DENY,
    FINDINGS,
    _check,
    _first_run,
    _policy,
    repo,
)
from tests.test_implement_stop_account import _dispatched_titles, _implement
from utils.state_manager import get_or_create_state

ETA = re.compile(r"\[ETA\] implement: batch (\d+)/(\d+) done")


def _etas(text: str) -> list:
    """Every ``[ETA] implement: batch N/M`` line's ``(N, M)``, in order."""
    return [(int(a), int(b)) for a, b in ETA.findall(text)]


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _emit(repo, capsys, *flags):
    import implement_orchestrator

    assert implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--yes", *flags]
    ) == 0
    out = capsys.readouterr().out
    path = next(
        line.split(": ", 1)[1] for line in out.splitlines()
        if line.startswith("Output file: ")
    )
    return json.loads(open(path, encoding="utf-8").read()), out


class TestTheNativeEtaEndsAtTheTasksDispatched:
    """The host passes the emit's `tasks_to_dispatch` to `util metrics start`
    and `record`; `util batch-done` takes out of that total each handed-out
    task its record keeps from being dispatched."""

    def _start(self, capsys, handoff) -> str:
        state_file = handoff["state_file"]
        code, _out, err = _util(
            capsys, "metrics", "start", "--state-file", state_file,
            "--phase", "implement", "--total-batches", str(handoff["tasks_to_dispatch"]),
        )
        assert code == 0, err
        return state_file

    def _done(self, repo, capsys, task_id, *extra) -> str:
        code, out, err = _util(
            capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
            "--task-id", task_id, *extra,
        )
        assert code == 0, err
        return out

    def _record(self, capsys, state_file, handoff, task_id, index) -> tuple:
        code, _out, err = _util(
            capsys, "metrics", "record", "--state-file", state_file,
            "--phase", "implement", "--label", f"{task_id}: task",
            "--subagent-type", "agent", "--duration-ms", "1000",
            "--total-batches", str(handoff["tasks_to_dispatch"]),
            "--batch-index", str(index),
        )
        assert code == 0, err
        return _etas(err)[-1], err

    def test_a_task_a_check_holds_leaves_the_total(self, repo, capsys):
        handoff, out = _emit(repo, capsys)
        assert handoff["tasks_to_dispatch"] == 5
        assert "pass --total-batches 5" in out
        state_file = self._start(capsys, handoff)

        self._done(repo, capsys, "T001", "--status", "completed")
        assert self._record(capsys, state_file, handoff, "T001", 1)[0] == (1, 5)
        blocked = self._done(
            repo, capsys, "T002", "--status", "completed", "--check", "findings",
            "--finding", FINDINGS[0])
        assert "[CHECK_BLOCKED] T007" in blocked
        assert self._record(capsys, state_file, handoff, "T002", 2)[0] == (2, 4)
        self._done(repo, capsys, "T003", "--status", "completed")
        assert self._record(capsys, state_file, handoff, "T003", 3)[0] == (3, 4)
        self._done(repo, capsys, "T004", "--status", "completed")
        last, err = self._record(capsys, state_file, handoff, "T004", 4)
        assert last == (4, 4), err
        assert "ALL BATCHES COMPLETE" in err

        progress = get_or_create_state(repo.plan).state["metrics_progress"]["implement"]
        assert progress["total_batches"] == 4
        assert progress["withdrawn"] == {"T007": "T002"}

    def test_the_tasks_after_a_failed_one_leave_it_too(self, repo, capsys):
        _policy(repo, ALLOW)
        handoff, _out = _emit(repo, capsys)
        assert handoff["tasks_to_dispatch"] == 7
        state_file = self._start(capsys, handoff)
        for index, task_id in enumerate(("T001", "T002", "T003", "T004"), 1):
            extra = ("--check", "passed") if task_id == "T002" else ()
            self._done(repo, capsys, task_id, "--status", "completed", *extra)
            self._record(capsys, state_file, handoff, task_id, index)
        self._done(repo, capsys, "T005", "--status", "failed", "--reason", "no stamp")
        assert self._record(capsys, state_file, handoff, "T005", 5)[0] == (5, 6)
        self._done(repo, capsys, "T007", "--status", "completed")
        last, err = self._record(capsys, state_file, handoff, "T007", 6)
        assert last == (6, 6) and "ALL BATCHES COMPLETE" in err, err

    def test_a_task_write_prompt_refused_leaves_it_with_its_dependents(
        self, repo, capsys,
    ):
        _policy(repo, ALLOW)
        handoff, _out = _emit(repo, capsys)
        state_file = self._start(capsys, handoff)
        _policy(repo, DENY)
        for index, task_id in enumerate(("T001", "T002", "T003", "T004"), 1):
            extra = ("--check", "passed") if task_id == "T002" else ()
            self._done(repo, capsys, task_id, "--status", "completed", *extra)
            self._record(capsys, state_file, handoff, task_id, index)
        code, _out, err = _util(
            capsys, "write-prompt", "--phase-dir", str(repo.phase_dir),
            "--task-id", "T005")
        assert code == 3 and "[WRITE_PROMPT_REFUSED]" in err
        self._done(
            repo, capsys, "T005", "--status", "failed", "--reason",
            "refused before dispatch: the write policy (driver.write_paths) "
            "does not allow a file it declares")
        self._done(repo, capsys, "T007", "--status", "completed")
        last, err = self._record(capsys, state_file, handoff, "T007", 5)
        assert last == (5, 5) and "ALL BATCHES COMPLETE" in err, err

    def test_recording_the_check_again_puts_its_held_tasks_back(self, repo, capsys):
        handoff, _out = _emit(repo, capsys)
        self._start(capsys, handoff)
        self._done(repo, capsys, "T002", "--status", "completed", "--check",
                   "findings", "--finding", FINDINGS[0])
        progress = get_or_create_state(repo.plan).state["metrics_progress"]["implement"]
        assert progress["total_batches"] == 4
        self._done(repo, capsys, "T002", "--status", "completed", "--check", "passed")
        progress = get_or_create_state(repo.plan).state["metrics_progress"]["implement"]
        assert progress["total_batches"] == 5
        assert "withdrawn" not in progress

    def test_without_metrics_start_nothing_is_stored(self, repo, capsys):
        _emit(repo, capsys)
        self._done(repo, capsys, "T002", "--status", "completed", "--check",
                   "findings", "--finding", FINDINGS[0])
        assert "metrics_progress" not in get_or_create_state(repo.plan).state


class TestTheExecutorEtaEndsAtTheTasksDispatched:

    def test_the_accept_path_counts_only_the_released_task(self, repo, capsys):
        """Codex and OpenCode accept 02: "batch 1/2 … remaining", the
        accepted check T002 counted in the total."""
        _first_run(repo, capsys)
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        text = capsys.readouterr()
        text = text.out + text.err
        assert _dispatched_titles(repo) == ["Edit: write the release note"]
        assert _etas(text) == [(1, 1)], text
        assert "ALL BATCHES COMPLETE" in text

    def test_a_task_a_check_holds_leaves_the_total(self, repo, capsys):
        _policy(repo, ALLOW)
        _check(repo, passed=False)
        assert _implement(repo) == 3
        text = capsys.readouterr()
        text = text.out + text.err
        etas = _etas(text)
        assert etas[-1] == (6, 6), etas
        assert etas[0] == (1, 7), etas  # T007 counted until T002 reported
        assert "ALL BATCHES COMPLETE" in text
