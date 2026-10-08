"""A manual task that every code task waits on is a stop, not a finished implement.

Fix round 28 (live run 12, Codex racing 06): generate-tasks made T010 a
manual task that T001-T007, T011 and T012 depended on. Under ``--yes`` the
``skip-dependents`` strategy skipped all nine, and implement returned
``state: done``, exit 0, "Organized into 0 execution batches", with
review-code next on an empty tree. Its manual note said "No later phase
waits for these manual tasks" while nine code tasks waited on T010, and its
command recorded T008, T009 and T010 done at once, though T008 and T009 come
after code nothing had run.

Now:

* a run that dispatches nothing because every code task left waits on a
  manual task not recorded done stops for the user (exit 3, "the run needs
  you"), on the executor leg and the native emit alike, and every reader's
  way on is the resume that records ONLY that manual task done;
* a run that ran some tasks and left others waiting stays done, and its
  ready section leads with that resume, review-code staying the other way on;
* the manual note never says nothing waits when a code task does, and the
  close command records only the manual tasks the user can do now.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from utils import cli_common, detach  # noqa: E402
from utils import next_phase as np_  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import manual_task_gates  # noqa: E402

from tests.test_write_executor_orchestrators import (  # noqa: E402,F401
    IMPLEMENT_WRITE_IDS,
    _seed_generate_tasks,
    repo,
)

#: The Codex racing 06 decomposition, as `state.json` recorded it.
RACING_GRAPH = {
    "T001": ["T010"], "T002": ["T001", "T010"], "T003": ["T001", "T010"],
    "T004": ["T002", "T003", "T010"], "T005": ["T004", "T010"],
    "T006": ["T004", "T010"], "T007": ["T003", "T006", "T010"],
    "T008": ["T001"],
    "T009": ["T002", "T003", "T004", "T005", "T006", "T007", "T008"],
    "T010": [], "T011": ["T002", "T010"], "T012": ["T006", "T010"],
}
RACING_MANUAL = ["T008", "T009", "T010"]
RACING_WAITING = ["T001", "T002", "T003", "T004", "T005", "T006", "T007",
                  "T011", "T012"]


def _run(argv, capsys):
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err, cli_common.parse_result_line(captured.out)


def _manual(repo, *task_ids):
    """The repo fixture's chain T001 → T002 → T003 with *task_ids* manual."""
    import implement_orchestrator

    original = implement_orchestrator.TaskDecomposer

    class SeededDecomposer(original):
        def parse_plan(self, content):
            tasks = super().parse_plan(content)
            for task_id in task_ids:
                self.tasks[task_id].executor_role = "human"
                self.tasks[task_id].subagent_type = "human"
            return tasks

    repo.monkeypatch.setattr(implement_orchestrator, "TaskDecomposer", SeededDecomposer)


def _clean_env(monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)


def _state(repo) -> dict:
    return json.loads((output_dir_path(repo.plan) / "state.json").read_text())


# ---------------------------------------------------------------------------
# The one rule: which manual tasks the user can do now, and what waits
# ---------------------------------------------------------------------------

class TestTheGates:
    def test_the_racing_decomposition_before_anything_ran(self):
        gates = manual_task_gates(
            RACING_GRAPH, human_ids=RACING_MANUAL, done=(), owed=RACING_MANUAL)
        assert gates["closable"] == ["T010"]
        assert gates["waiting"] == {"T010": RACING_WAITING}
        # T008 and T009 come after code nothing has run yet.
        assert gates["after"] == {
            "T008": ["T001"],
            "T009": ["T001", "T002", "T003", "T004", "T005", "T006", "T007"],
        }
        assert gates["gated"] is True

    def test_once_the_code_ran_the_later_manual_tasks_are_closable(self):
        done = [t for t in RACING_GRAPH if t not in ("T008", "T009")]
        gates = manual_task_gates(
            RACING_GRAPH, human_ids=RACING_MANUAL, done=done,
            owed=["T008", "T009"])
        assert gates["closable"] == ["T008", "T009"]
        assert gates["waiting"] == {} and gates["after"] == {}
        assert gates["gated"] is False

    def test_a_code_task_waiting_on_no_manual_task_ungates(self):
        graph = {"T001": [], "T002": ["T001"], "T003": []}
        gates = manual_task_gates(
            graph, human_ids=["T001"], done=(), owed=["T001"])
        assert gates["waiting"] == {"T001": ["T002"]}
        assert gates["gated"] is False  # T003 runs

    def test_a_task_waiting_on_two_manual_tasks_needs_both(self):
        graph = {"T001": [], "T002": [], "T003": ["T001", "T002"]}
        gates = manual_task_gates(
            graph, human_ids=["T001", "T002"], done=(), owed=["T001", "T002"])
        assert gates["closable"] == ["T001", "T002"]
        assert gates["waiting"] == {"T001": ["T003"], "T002": ["T003"]}

    def test_a_task_the_user_skipped_is_not_waiting(self):
        graph = {"T001": [], "T002": ["T001"], "T003": ["T002"]}
        gates = manual_task_gates(
            graph, human_ids=["T001"], done=(), owed=["T001"],
            dropped=["T003"])
        assert gates["waiting"] == {"T001": ["T002"]}


class TestTheNoteNeverSaysNothingWaits:
    """Live run 12: "No later phase waits for these manual tasks" beside nine
    code tasks waiting on T010."""

    def test_a_note_over_waiting_code_tasks(self):
        note = np_.manual_close_lead(1, waiting=9)
        assert "No later phase waits" not in note
        assert note == (
            "Implement has not run the 9 code tasks that wait on this manual "
            "task. Once you have done it by hand, this command records it "
            "done and runs them:"
        )
        plural = np_.manual_close_lead(2, waiting=1)
        assert "No later phase waits" not in plural
        assert "the code task that waits on these manual tasks" in plural

    def test_with_nothing_waiting_the_note_is_as_it_was(self):
        assert np_.manual_close_lead(1) == np_.manual_close_lead(1, waiting=0)
        assert np_.manual_close_lead(1).startswith("No later phase waits")

    def test_a_later_manual_task_names_the_code_it_comes_after(self):
        task = np_.ManualTask("T008", "Verify the CDN module", after=("T001",))
        assert task.line() == (
            "T008 Verify the CDN module (after T001: do it once implement has "
            "run that task)"
        )
        assert task.to_dict()["after"] == ["T001"]


# ---------------------------------------------------------------------------
# The executor leg (the tool routes): exit 3, and every reader's way on
# ---------------------------------------------------------------------------

class TestTheExecutorLegStops:
    def _stop(self, repo, capsys):
        _clean_env(repo.monkeypatch)
        _seed_generate_tasks(repo)
        _manual(repo, "T001")
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        return _run([
            "implement", str(repo.plan), "--yes", "--exec-subagents",
            "--host", "codex",
        ], capsys)

    def test_nothing_ran_is_exit_3_naming_the_manual_task(self, repo, capsys):
        code, out, err, result = self._stop(repo, capsys)
        assert code == 3, err
        assert result["state"] == "needs-human"
        assert "user_next_step" not in result
        error = result["error"]
        assert error.startswith(
            "implement ran nothing: 2 code tasks wait on manual task T001 "
            "(Create the module), which is not recorded done, and no other "
            "code task can run"), error
        assert "--human-task T001=done" in error
        # The log says so, and how many tasks wait.
        assert "T001 (Create the module): 2 code task(s) wait on it (T002, T003)" in err
        assert "[ETA] implement: run stopped with nothing dispatched" in out + err
        assert "T001 left for you (manual); 2 code task(s) wait on it" in out + err
        # Not completed, and recorded started, so every reader offers a resume.
        state = _state(repo)
        assert "implement" not in (state.get("phases_completed") or {})
        assert "implement" in (state.get("phases_started") or {})
        summary = (output_dir_path(repo.plan) / "implement" / "summary.md").read_text()
        assert "## Not Finished: Waiting On A Manual Task" in summary
        assert "No later phase waits" not in summary

    def test_the_tool_routes_resume_records_only_the_manual_task(self, repo, capsys):
        self._stop(repo, capsys)
        code, out, err, result = _run([
            "util", "next-command", str(repo.plan), "--blocked", "implement",
            "--host", "codex",
        ], capsys)
        assert code == 0, err
        resume = result["user_resume"]
        assert resume["command"].endswith(
            "--implement " + str(repo.plan) + " --resume --human-task T001=done")
        assert resume["flags"] == ["--resume", "--human-task", "T001=done"]
        assert resume["why"].startswith(
            "Implement can run no code task until manual task T001 (Create "
            "the module) is recorded done"), resume["why"]
        assert "runs the 2 code tasks that wait on it" in resume["why"]
        markdown = resume["markdown"]
        assert markdown.startswith("To go on with implement:")
        assert resume["command"] in markdown
        assert "--review-code" not in markdown

    def test_status_offers_that_resume(self, repo, capsys):
        """`status --view run`'s resume block of the (detached) run, built
        from its own mirror, as `test_resume_why` reads it."""
        import check_workflow_prerequisites as prereq
        from utils import run_status
        from utils.state_manager import StateManager

        self._stop(repo, capsys)
        phase_dir = output_dir_path(repo.plan) / "implement"
        status = json.loads((phase_dir / ".status.json").read_text())
        view = {"run": {"mode": "implement", "replay_flags": []},
                "phase": "implement", "progress": status["progress"]}
        workflow = prereq.get_workflow_status(
            StateManager(repo.plan), repo.plan, output_dir_path(repo.plan))
        assert workflow["phases"]["implement"]["manual_gate"] is True
        lines = run_status._resume_lines(view, repo.plan, workflow=workflow)
        assert lines[0].strip().startswith("resume with:"), lines
        assert lines[0].rstrip().endswith("--resume --human-task T001=done"), lines
        assert "until manual task T001 (Create the module) is recorded done" in lines[1], lines
        # The workflow trailer's note says what waits, never that nothing does.
        code, out, err, result = _run(["status", str(repo.plan)], capsys)
        assert "No later phase waits" not in out
        assert ("Implement has not run the 2 code tasks that wait on this "
                "manual task.") in out


class TestTheExecutorLegKeepsDoneWhenSomethingRan:
    """T002 manual: T001 runs, T003 waits on T002. The run is done, and its
    ready section leads with the resume that runs T003."""

    def test_the_section_leads_with_the_manual_resume(self, repo, capsys):
        _clean_env(repo.monkeypatch)
        _seed_generate_tasks(repo)
        _manual(repo, "T002")
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        code, out, err, result = _run([
            "implement", str(repo.plan), "--yes", "--exec-subagents",
            "--host", "codex",
        ], capsys)
        assert code == 0, err
        step = result["user_next_step"]
        assert step["phase"] == "implement"
        assert step["command"].endswith("--resume --human-task T002=done")
        assert "Implement has not run the code task that waits on manual task T002" in step["why"]
        assert step["instead"] == [{
            "phase": "code-review",
            "command": step["instead"][0]["command"],
            "why": "Or review the code implement has run so far:",
        }]
        assert "--review-code" in step["instead"][0]["command"]
        markdown = step["markdown"]
        assert markdown.index("--human-task T002=done") < markdown.index("--review-code")
        # The note and its command would repeat the command leading it.
        assert "manual_tasks_command" not in step
        assert "No later phase waits" not in markdown


# ---------------------------------------------------------------------------
# The native leg (Claude Code): the emit stops, and the report's lookup
# ---------------------------------------------------------------------------

class TestTheNativeLegStops:
    def _emit(self, repo, capsys):
        _clean_env(repo.monkeypatch)
        _seed_generate_tasks(repo)
        _manual(repo, "T001")
        return _run([
            "implement", str(repo.plan), "--yes", "--host", "claude-code",
        ], capsys)

    def test_the_emit_hands_out_nothing_and_stops(self, repo, capsys):
        code, out, err, result = self._emit(repo, capsys)
        assert code == 3, err
        assert result["state"] == "needs-human"
        assert result["error"].startswith(
            "implement ran nothing: 2 code tasks wait on manual task T001 "
            "(Create the module)"), result["error"]
        assert "--human-task T001=done" in result["error"]
        assert f"[MANUAL_GATE] {result['error']}" in out
        assert "T001 (Create the module): 2 code task(s) wait on it (T002, T003)" in out
        assert ("[ETA] implement: run stopped with nothing dispatched · T001 left "
                "for you (manual); 2 code task(s) wait on it") in out
        # Never the generic "go on to step 7" of an emit with nothing left,
        # and no summary or second copy of the resume asked for.
        assert "Nothing to dispatch: every task is done" not in out
        assert "Claude Code should now:" not in out
        assert "[MANUAL_TASK_NOTE]" not in out
        assert "Manual Tasks (Deferred) section" not in out
        assert "Tasks to dispatch:" not in out  # no `util metrics start` asked
        state = _state(repo)
        assert state["task_status"]["T001"] == "skipped"
        assert state["task_status"]["T002"] == "skipped"
        assert "implement" in (state.get("phases_started") or {})
        assert "implement" not in (state.get("phases_completed") or {})

    def test_the_report_lookup_leads_with_the_manual_resume(self, repo, capsys):
        self._emit(repo, capsys)
        code, out, err, result = _run([
            "util", "next-command", str(repo.plan), "--after", "implement",
            "--host", "claude-code",
        ], capsys)
        assert code == 0, err
        step = result["user_next_step"]
        assert step["phase"] == "implement"
        assert step["command"].endswith("--resume --human-task T001=done")
        assert "a plain resume runs nothing" in step["why"]
        markdown = step["markdown"]
        assert markdown.startswith("To go on with implement:")
        assert "--review-code" not in markdown
        assert "No later phase waits" not in markdown
        assert ("[NEXT_COMMAND_WHY] Implement can run no code task until manual "
                "task T001") in out
        # The same command is not given twice.
        assert "manual_tasks_command" not in step

    def test_recording_it_done_runs_the_waiting_tasks(self, repo, capsys):
        """The way on works: the next emit hands out T002 and T003."""
        self._emit(repo, capsys)
        code, out, err, result = _run([
            "implement", str(repo.plan), "--yes", "--host", "claude-code",
            "--resume", "--human-task", "T001=done",
        ], capsys)
        assert code == 0, err
        assert "Tasks to dispatch: 2" in out
        assert "[MANUAL_GATE]" not in out


class TestALaterManualTaskIsNotRecordedDoneEarly:
    """Codex racing 06's command recorded T008 and T009 done beside T010,
    though both come after code nothing had run. T001 and T003 manual here:
    T003 comes after T002, which waits on T001."""

    def test_only_the_manual_task_doable_now_is_in_the_command(self, repo, capsys):
        _clean_env(repo.monkeypatch)
        _seed_generate_tasks(repo)
        _manual(repo, "T001", "T003")
        code, out, err, result = _run([
            "implement", str(repo.plan), "--yes", "--host", "claude-code",
        ], capsys)
        assert code == 3, err
        code, out, err, result = _run([
            "util", "next-command", str(repo.plan), "--after", "implement",
            "--host", "claude-code",
        ], capsys)
        assert code == 0, err
        step = result["user_next_step"]
        assert step["command"].endswith("--resume --human-task T001=done")
        assert "T003=done" not in step["markdown"]
        later = next(t for t in step["manual_tasks"] if t["id"] == "T003")
        assert later["after"] == ["T002"]
        assert ("- T003 Add tests (after T002: do it once implement has run "
                "that task)") in step["markdown"]


def test_the_runbook_tells_the_native_host_what_the_gate_means():
    core = (Path(__file__).parent.parent / "instructions" / "implement.md").read_text(
        encoding="utf-8")
    line = next(l for l in core.splitlines() if "`[MANUAL_GATE]`" in l)
    assert "util next-command" in line and "Skip steps 4 to 10" in line


def test_a_gate_over_two_manual_tasks_names_how_many_wait_on_each():
    from utils.write_batch_scheduler import manual_gate_head, manual_gate_way_on

    head = manual_gate_head(
        "implement", {"T010": " (Fixture)", "T011": ""},
        {"T010": ["T001", "T002"], "T011": ["T003"]})
    assert head == (
        "implement ran nothing: every code task left waits on a manual task "
        "not recorded done (T010 (Fixture) holds 2, T011 holds 1)")
    assert manual_gate_way_on(["T010", "T011"]) == (
        "To go on: do T010, T011 by hand, then resume this phase with "
        "--human-task T010=done --human-task T011=done (the code tasks that "
        "wait on them then run).")


def test_a_code_task_behind_a_later_manual_task_is_not_counted_as_released():
    """T001 (manual) -> T002 -> T003 (manual) -> T004: recording T001 done
    runs T002 only; T004 still waits on T003, which comes after T002. The
    gate's words count only what the resume runs and never say every code
    task waits on T001."""
    from utils.write_batch_scheduler import manual_gate_head

    graph = {"T001": [], "T002": ["T001"], "T003": ["T002"], "T004": ["T003"]}
    gates = manual_task_gates(
        graph, human_ids=["T001", "T003"], done=(), owed=["T001", "T003"])
    assert gates["closable"] == ["T001"]
    assert gates["waiting"] == {"T001": ["T002"]}
    assert gates["after"] == {"T003": ["T002"]}
    assert gates["gated"] is True
    head = manual_gate_head("implement", {"T001": ""}, gates["waiting"])
    assert head == (
        "implement ran nothing: 1 code task waits on manual task T001, which "
        "is not recorded done, and no other code task can run")
    why = np_.manual_gate_why({
        "manual_close_ids": ["T001"], "manual_waiting": gates["waiting"],
        "manual_tasks": [{"id": "T001", "title": "Fixture"}]})
    assert why == (
        "Implement can run no code task until manual task T001 (Fixture) is "
        "recorded done, so a plain resume runs nothing. Once you have done "
        "T001 by hand, this command records it done and runs the code task "
        "that waits on it.")
