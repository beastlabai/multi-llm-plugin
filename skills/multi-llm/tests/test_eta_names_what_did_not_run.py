"""The closing `[ETA]` line names what a run did not run, on both legs.

Live run 9 (2026-10-07), the check-gate fixture (T002 a check that reports
findings and holds T007; T005 refused by the write policy, T006 after it):

* #1: the executor (Codex, OpenCode) logged "batch 4/4 … ALL BATCHES
  COMPLETE" and then "run stopped after batch 4/4" with nothing after it,
  while the Claude Code native leg's `util metrics finish` printed "run
  stopped after batch 4/4 … · T007 not run" (T005 and T006 unnamed there).
  "All complete", then "stopped", read as a contradiction. The last line
  before a stop no longer says "complete", and both legs close on the same
  line naming each task the run left: refused, held by a check, not run.
* #1: 01b (`--resume` on unchanged files) and 01c (plain `--implement`
  continuing) dispatched nothing and logged no `[ETA]` line at all, on
  either leg (the native host ran `util metrics finish` with no `start`,
  which printed nothing). Every ending closes, a zero-dispatch one too.
* Units (Claude Code finding 2): "Organized into 3 execution batches: 5
  task(s) to dispatch" beside "--total-batches 5" and "[ETA] implement:
  batch 1/5": the line counts tasks, so on implement it says "task 1/5".
"""

from __future__ import annotations

import json
import re

from tests.test_check_gate_fixture import (  # noqa: F401  (the fixture)
    CHECK_GATE,
    DENY,
    FINDINGS,
    _first_run,
    _policy,
    repo,
)
from tests.test_dispatch_counts import LIVE_LAYOUT, _emit, _util
from tests.test_implement_stop_account import _implement, make_repo
from utils import metrics

#: What the check-gate fixture's stop leaves, on both legs.
LEFT = (
    "T005 refused before dispatch; T006 not run; "
    "T007 not run: check T002 reported problems"
)
TASK_ETA = re.compile(r"\[ETA\] implement: task (\d+)/(\d+) done")


def _eta_lines(text: str) -> list:
    return [line for line in text.splitlines() if line.startswith("[ETA] implement:")]


def _task_etas(text: str) -> list:
    return [(int(a), int(b)) for a, b in TASK_ETA.findall(text)]


def _run(repo, capsys, *flags) -> tuple:
    code = _implement(repo, *flags)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


class TestTheExecutorClosesOnWhatItLeft:
    """The live run's layout: T007 writes its own file, so it sits beside
    T004 before the refused T005 and waits on T002's verdict."""

    @staticmethod
    def _live(tmp_path, monkeypatch):
        repo = make_repo(tmp_path, monkeypatch, LIVE_LAYOUT)
        _policy(repo, DENY)
        monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
            [f"T00{n}" for n in range(1, 8)]))
        monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
        return repo

    def test_the_stop_names_each_task_and_its_last_line_claims_no_completion(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = self._live(tmp_path, monkeypatch)
        text = _first_run(repo, capsys)
        assert _task_etas(text) == [(1, 5), (2, 4), (3, 4), (4, 4)], text
        lines = _eta_lines(text)
        # T004's line is the last dispatched task's, and the run is about to
        # stop on T005: it says what is true, and no more.
        assert "COMPLETE" not in lines[-2], lines
        assert lines[-1] == f"[ETA] implement: run stopped after task 4/4 · {LEFT}", lines
        assert "batch" not in "\n".join(lines), lines

    def test_a_resume_that_dispatches_nothing_still_closes(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = self._live(tmp_path, monkeypatch)
        _first_run(repo, capsys)
        code, text = _run(repo, capsys, "--resume")
        assert code == 3, text
        assert "Organized into 0 execution batches: 0 task(s) to dispatch" in text
        assert _eta_lines(text) == [
            f"[ETA] implement: run stopped with nothing dispatched · {LEFT}"
        ], text

    def test_a_plain_implement_that_continues_and_dispatches_nothing_closes(
        self, tmp_path, monkeypatch, capsys,
    ):
        repo = self._live(tmp_path, monkeypatch)
        _first_run(repo, capsys)
        code, text = _run(repo, capsys)
        assert code == 3, text
        assert "continues it as --resume" in text, text
        assert _eta_lines(text) == [
            f"[ETA] implement: run stopped with nothing dispatched · {LEFT}"
        ], text


class TestTheNativeLegClosesOnTheSameLine:
    """The host runs the emit, `util metrics start` with its
    `tasks_to_dispatch`, one `util batch-done` + `util metrics record` per
    task, and `util metrics finish` (``instructions/implement.md`` 5-7)."""

    def _first_run(self, repo, capsys) -> tuple:
        handoff, _out = _emit(repo, capsys)
        state_file = handoff["state_file"]
        total = str(handoff["tasks_to_dispatch"])
        assert _util(capsys, "metrics", "start", "--state-file", state_file,
                     "--phase", "implement", "--total-batches", total)[0] == 0
        errs = []
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
            errs.append(err)
        return state_file, "".join(errs)

    def _finish(self, capsys, state_file) -> list:
        code, _o, err = _util(
            capsys, "metrics", "finish", "--state-file", state_file,
            "--phase", "implement")
        assert code == 0, err
        return _eta_lines(err)

    def test_the_stop_names_the_same_tasks_as_the_executor(self, repo, capsys):
        state_file, err = self._first_run(repo, capsys)
        assert _task_etas(err) == [(1, 5), (2, 4), (3, 4), (4, 4)], err
        assert "COMPLETE" not in _eta_lines(err)[-1], err
        lines = self._finish(capsys, state_file)
        assert len(lines) == 1, lines
        assert re.fullmatch(
            r"\[ETA\] implement: run stopped after task 4/4 in [^·]+ · "
            + re.escape(LEFT),
            lines[0],
        ), lines

    def test_a_resume_that_hands_out_nothing_closes_without_a_start(
        self, repo, capsys,
    ):
        state_file, _err = self._first_run(repo, capsys)
        self._finish(capsys, state_file)
        handoff, out = _emit(repo, capsys, "--resume")
        assert handoff["tasks_to_dispatch"] == 0, out
        assert "still run step 7's `util metrics finish`" in out, out
        # The host spawns nothing and goes on to step 7.
        assert self._finish(capsys, state_file) == [
            f"[ETA] implement: run stopped with nothing dispatched · {LEFT}"
        ]
        # Once: a second finish has nothing left to close.
        assert self._finish(capsys, state_file) == []

    def test_a_host_that_starts_at_zero_gets_the_one_line_too(self, repo, capsys):
        state_file, _err = self._first_run(repo, capsys)
        self._finish(capsys, state_file)
        handoff, _out = _emit(repo, capsys, "--resume")
        assert _util(capsys, "metrics", "start", "--state-file", state_file,
                     "--phase", "implement", "--total-batches",
                     str(handoff["tasks_to_dispatch"]))[0] == 0
        assert self._finish(capsys, state_file) == [
            f"[ETA] implement: run stopped with nothing dispatched · {LEFT}"
        ]


class TestTheLastTaskFailing:
    """A run whose last counted task failed stops at N/N: no "complete" on
    that line, and the native finish no longer says "phase complete"."""

    def test_the_native_finish_says_stopped(self, tmp_path):
        state = {"metrics": {}}
        metrics.start_phase_in_state(state, "implement", 2)
        metrics.record_metric_in_state(
            state, "implement", "T001: t", duration_ms=10,
            total_batches=2, batch_index=1)
        # `util batch-done … --status failed` for T002, then its record.
        metrics.mark_failed_in_progress(state, "implement", "T002", failed=True)
        line = metrics.record_metric_in_state(
            state, "implement", "T002: t", duration_ms=10,
            total_batches=2, batch_index=2)
        # Fix round 24 (live run 10 #5): it says it failed, never "done".
        assert line.startswith("[ETA] implement: task 2/2 failed after "), line
        assert "COMPLETE" not in line, line
        _elapsed, closing = metrics.finish_phase_in_state(state, "implement")
        assert re.fullmatch(
            r"\[ETA\] implement: run stopped after task 2/2 in [^·]+", closing,
        ), closing


class TestTheTailIsBounded:

    def test_each_group_names_five_and_counts_the_rest(self):
        left = {f"T{n:03d}": "" for n in range(10, 18)}
        left["T009"] = metrics.LEFT_REFUSED
        assert metrics.left_out_tail(left) == (
            "T009 refused before dispatch; "
            "T010, T011, T012, T013, T014 and 3 more not run"
        )

    def test_ids_sort_by_number(self):
        left = {"T10": "", "T9": "", "T2": metrics.left_out_check("T1")}
        assert metrics.left_out_tail(left) == (
            "T2 not run: check T1 reported problems; T9, T10 not run"
        )


class TestTheUnitIsTheOneTheLineCounts:

    def test_implement_counts_tasks_and_the_others_batches(self):
        state = {"metrics": {}}
        metrics.start_phase_in_state(state, "implement", 1)
        line = metrics.record_metric_in_state(
            state, "implement", "T001: t", duration_ms=10,
            total_batches=1, batch_index=1)
        assert line.startswith("[ETA] implement: task 1/1 done in "), line
        assert line.endswith("ALL TASKS COMPLETE"), line
        metrics.start_phase_in_state(state, "apply-suggestions", 1)
        line = metrics.record_metric_in_state(
            state, "apply-suggestions", "b0", duration_ms=10,
            total_batches=1, batch_index=1)
        assert line.startswith("[ETA] apply-suggestions: batch 1/1 done in "), line
        assert line.endswith("ALL BATCHES COMPLETE"), line
