"""`status` heads a native implement that stopped as stopped (fix round 24).

Live run 10, Claude Code stop path 01/01b (FINDINGS-stop-claude-code.md
issue 3): after the native leg stopped — T005 refused by the write policy,
T002's check reporting problems and holding T007 — `status` headed the
implement block "not finished on the host (not marked completed)", while the
Codex and OpenCode routes, whose executor ran the same stop, head it
"stopped and needs you (exit 3)". Round 23 left the native head alone, since
"not finished" also covers a host still working.

`state.json` tells the two apart: the host's loop is closed (no `util
metrics start` without its `finish`), nothing is in flight, something waits
for the user's answer (a task recorded failed — refused before dispatch
included — or a check whose findings hold tasks back), and no task is left
that the host could still run without that answer. Such a run is headed
"stopped and needs you"; a host still working keeps "in progress on the
host" (its loop is open) or "not finished on the host" (a task it can still
run is waiting), and the resume and answer lines below the head do not move.
"""

from __future__ import annotations

import json

import multi_llm
from utils import run_status
from utils.state_manager import get_or_create_state

import tests.test_check_gate_fixture as gate
from tests.test_check_gate_fixture import FINDINGS, repo  # noqa: F401

PLAN = "plans/checked.md"
STOPPED_HEAD = (
    "  implement  stopped and needs you (not marked completed) — no run "
    "record, since the host ran these tasks itself; read from state.json"
)
NOT_FINISHED = "  implement  not finished on the host (not marked completed)"


def _leg() -> "gate.TestTheNativeLeg":
    # Imported through the module so pytest does not collect the class here.
    return gate.TestTheNativeLeg()


def _state_file(repo) -> str:
    return str(get_or_create_state(repo.plan).state_file)


def _metrics(capsys, verb: str, repo, *extra) -> None:
    code, _out, err = _leg()._util(
        capsys, "metrics", verb, "--state-file", _state_file(repo),
        "--phase", "implement", *extra)
    assert code == 0, err


def _record_the_stop(repo, capsys, *, metrics: bool) -> None:
    """The native first stop as the runbook drives it: the emit (T005
    refused and recorded failed), the loop over the tasks it handed out
    (T001, T003, T004 done, T002's check with findings, T007 held) and the
    closing calls, without `--mark-completed`."""
    leg = _leg()
    leg._emit(repo, capsys)
    if metrics:
        _metrics(capsys, "start", repo, "--total-batches", "5")
    phase = str(repo.phase_dir)
    for task_id in ("T001", "T003", "T004"):
        assert leg._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                         task_id, "--status", "completed")[0] == 0
    assert leg._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                     "--status", "completed", "--check", "findings",
                     "--finding", FINDINGS[0])[0] == 0
    if metrics:
        _metrics(capsys, "finish", repo)
    assert leg._util(capsys, "finalize-tracking", "--state-file",
                     _state_file(repo))[0] == 0


def _status_lines(capsys, *flags) -> list:
    capsys.readouterr()
    run_status.main(["--plan-file", PLAN, "--host", "claude-code", *flags])
    return capsys.readouterr().out.splitlines()


def _head(lines) -> str:
    return next(line for line in lines if line.startswith("  implement  "))


def _json_view(capsys) -> dict:
    capsys.readouterr()
    multi_llm.main(["status", PLAN, "--host", "claude-code", "--view", "run",
                    "--json"])
    out = capsys.readouterr().out
    return json.loads(out[: out.index("MULTI_LLM_RESULT")])["implement_tasks"]


class TestAStoppedNativeImplementIsHeadedStopped:

    def test_the_run_view_and_plain_status_head_it_stopped(self, repo, capsys):
        _record_the_stop(repo, capsys, metrics=True)
        for flags in (["--view", "run"], []):
            lines = _status_lines(capsys, *flags)
            assert STOPPED_HEAD in lines, lines
            assert not any(line.startswith(NOT_FINISHED) for line in lines)

    def test_the_resume_and_answers_below_it_do_not_move(self, repo, capsys):
        _record_the_stop(repo, capsys, metrics=True)
        lines = _status_lines(capsys, "--view", "run")
        at = lines.index(STOPPED_HEAD)
        block = lines[at:]
        assert block[1].startswith("      all tasks:      4/7 done;"), block
        resume = next(line for line in block if "resume with:" in line)
        assert resume == f"      resume with: /multi-llm:multi-llm --implement {PLAN} --resume"
        answers = [line for line in block if "or answer the stop with:" in line]
        assert answers[0] == (
            f"      or answer the stop with: /multi-llm:multi-llm --implement {PLAN} "
            "--resume --human-task T002=done --human-task T005=skip")
        assert len(answers) == 6

    def test_a_host_that_ran_no_metrics_calls_reads_the_same(self, repo, capsys):
        """The record decides, not the `[ETA]` calls a host may leave out."""
        _record_the_stop(repo, capsys, metrics=False)
        assert _head(_status_lines(capsys, "--view", "run")) == STOPPED_HEAD

    def test_a_resume_that_dispatched_nothing_stays_stopped(self, repo, capsys):
        """01b: the resume emit hands out nothing (T005 refused, T007 held,
        T006 after T005), the host closes its `[ETA]` lines and stops."""
        _record_the_stop(repo, capsys, metrics=True)
        handoff = _leg()._emit(repo, capsys, "--resume")
        assert not handoff["batches"], handoff["batches"]
        assert _head(_status_lines(capsys, "--view", "run")) == STOPPED_HEAD
        _metrics(capsys, "finish", repo)
        assert _head(_status_lines(capsys, "--view", "run")) == STOPPED_HEAD

    def test_json_says_so(self, repo, capsys):
        _record_the_stop(repo, capsys, metrics=True)
        view = _json_view(capsys)
        assert view["stopped"] is True
        assert view["completed"] is False


class TestAHostStillWorkingIsNotHeadedStopped:

    def test_right_after_the_emit_the_handed_out_tasks_can_still_run(
        self, repo, capsys,
    ):
        """The emit already recorded T005 failed (refused before dispatch),
        but T001-T004 wait for the host: nothing has stopped yet."""
        _leg()._emit(repo, capsys)
        head = _head(_status_lines(capsys, "--view", "run"))
        assert head.startswith(NOT_FINISHED), head
        assert _json_view(capsys)["stopped"] is False

    def test_a_loop_still_open_reads_in_progress(self, repo, capsys):
        leg = _leg()
        leg._emit(repo, capsys)
        _metrics(capsys, "start", repo, "--total-batches", "5")
        leg._util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
                  "--task-id", "T001", "--status", "completed")
        head = _head(_status_lines(capsys, "--view", "run"))
        assert head.startswith("  implement  in progress on the host (started "), head

    def test_a_loop_open_after_its_last_record_still_reads_in_progress(
        self, repo, capsys,
    ):
        """Every task recorded, the closing `util metrics finish` not run
        yet: the host is closing its loop (or died there, which the head's
        own line below names), so it is not called stopped."""
        leg = _leg()
        leg._emit(repo, capsys)
        _metrics(capsys, "start", repo, "--total-batches", "5")
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            leg._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                      task_id, "--status", "completed")
        leg._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                  "--status", "completed", "--check", "findings",
                  "--finding", FINDINGS[0])
        head = _head(_status_lines(capsys, "--view", "run"))
        assert head.startswith("  implement  in progress on the host (started "), head

    def test_a_run_with_nothing_waiting_for_an_answer_is_not_stopped(
        self, repo, capsys,
    ):
        """A loop that closed short with no failure and no check holding
        anything back left tasks a resume runs: not finished, as before."""
        leg = _leg()
        leg._emit(repo, capsys)
        state = get_or_create_state(repo.plan)
        state.update_task_status("T005", "pending")
        state.save()
        _metrics(capsys, "start", repo, "--total-batches", "5")
        leg._util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
                  "--task-id", "T001", "--status", "completed")
        _metrics(capsys, "finish", repo)
        head = _head(_status_lines(capsys, "--view", "run"))
        assert head.startswith(NOT_FINISHED), head

    def test_a_completed_one_reads_done(self, repo, capsys):
        leg = _leg()
        handoff = leg._emit(repo, capsys)
        phase = str(repo.phase_dir)
        for batch in handoff["batches"]:
            for task in batch["tasks"]:
                extra = (["--check", "passed"] if task["id"] == "T002" else [])
                leg._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          task["id"], "--status", "completed", *extra)
        state = get_or_create_state(repo.plan)
        state.update_task_status("T005", "skipped")
        state.update_task_status("T006", "skipped")
        state.mark_phase_completed("implement")
        state.save()
        head = _head(_status_lines(capsys, "--view", "run"))
        assert head.startswith("  implement  done on the host (marked completed)"), head
        assert _json_view(capsys)["stopped"] is False
