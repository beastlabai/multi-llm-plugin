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

    def test_the_line_beside_the_section_names_the_gate_and_no_answer(
            self, repo, capsys):
        """Merge of fix round 28 (relay's #5 on implement's #1): the gate's
        stop set its headline itself, recorded no line beside the section,
        and `--blocked` handed the servers none, so a tool route relayed
        "To go on: … --human-task T001=done" beside a section giving the
        same command. The beside line says what stopped and why, and the
        lookup hands both lines back."""
        from utils.write_batch_scheduler import (
            HEADLINE_BESIDE_SECTION_KEY, answers_cover,
        )

        code, out, err, result = self._stop(repo, capsys)
        assert code == 3, err
        phase_dir = output_dir_path(repo.plan) / "implement"
        stop = json.loads((phase_dir / "write_batch_report.json").read_text())["stop"]
        beside = stop[HEADLINE_BESIDE_SECTION_KEY]
        assert beside == (
            "implement ran nothing: 2 code tasks wait on manual task T001 "
            "(Create the module), which is not recorded done, and no other "
            "code task can run. Not run: T002, T003."), beside
        assert answers_cover(stop["headline"], beside, stop)
        mirror = json.loads((phase_dir / ".status.json").read_text())
        assert mirror["progress"]["write_subagents"]["stop"][
            HEADLINE_BESIDE_SECTION_KEY] == beside
        code, out, err, lookup = _run([
            "util", "next-command", str(repo.plan), "--blocked", "implement",
            "--host", "codex",
        ], capsys)
        assert code == 0, err
        resume = lookup["user_resume"]
        assert resume["error_line"] == result["error"] == stop["headline"]
        assert resume["error_beside_section"] == beside
        assert "--human-task T001=done" in resume["markdown"]

    def test_a_recorded_resume_the_section_does_not_carry_withholds_both_lines(
            self, repo, capsys):
        """The beside line drops the gate's answer only while the section's
        own command carries it: a record whose `resume_flags` name another
        manual task (the folder is writable by every jailed batch) keeps
        the full line."""
        self._stop(repo, capsys)
        status = output_dir_path(repo.plan) / "implement" / ".status.json"
        data = json.loads(status.read_text())
        data["progress"]["write_subagents"]["stop"]["resume_flags"] = [
            "--resume", "--human-task", "T002=done"]
        status.write_text(json.dumps(data))
        code, out, err, lookup = _run([
            "util", "next-command", str(repo.plan), "--blocked", "implement",
            "--host", "codex",
        ], capsys)
        assert code == 0, err
        resume = lookup["user_resume"]
        assert "error_line" not in resume and "error_beside_section" not in resume

    def _detached_poll(self, repo, capsys, route=None):
        """The gate's stop as a detached run records it, then the terminal
        `status --view run` poll a tool route (or a shell) makes of it."""
        from utils import run_status, status_file

        _code, _out, _err, result = self._stop(repo, capsys)
        phase_dir = output_dir_path(repo.plan) / "implement"
        progress = json.loads((phase_dir / ".status.json").read_text())["progress"]
        status_file.claim_run(str(phase_dir), {
            "run_id": "run-gate-0028", "state": "failed", "exit_code": 3,
            "error": result["error"], "detached": True, "mode": "implement",
            "started_at": "2026-10-09T10:00:00", "host": "codex"})
        status_file.update_status(str(phase_dir), {"progress": progress})
        capsys.readouterr()
        if route:
            repo.monkeypatch.setenv("MULTI_LLM_ROUTE", route)
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            code = run_status.main([
                "--plan-file", str(repo.plan), "--view", "run", "--phase",
                "implement", "--host", "codex"])
            payload = ctx.payload(code)
        finally:
            cli_common.deactivate_result()
        return code, payload, capsys.readouterr().out

    def test_a_tool_routes_poll_leaves_the_gates_resume_to_the_section(
            self, repo, capsys):
        """The merge of fix round 28: the terminal poll's notice (and the
        text a tool route reads) ended the manual-tasks line with "this
        command records it done and runs them: … --resume --human-task
        T001=done", the very resume the stop's section gives (`--blocked`):
        the way on twice, run 12's #5 again. It names the tasks and leaves
        the command to the section."""
        code, payload, out = self._detached_poll(repo, capsys, route="package")
        assert code == 3, out
        notice = payload.get("notice") or ""
        assert "manual tasks left for you: T001 Create the module" in notice, notice
        assert "--human-task T001=done" not in notice, notice
        assert "this command records it done" not in notice, notice
        trailer = [line for line in out.splitlines()
                   if line.startswith("workflow manual tasks left for you")]
        assert trailer and "--human-task T001=done" not in trailer[0], trailer

    def test_a_shells_text_keeps_the_command(self, repo, capsys):
        code, payload, out = self._detached_poll(repo, capsys)
        assert code == 3, out
        trailer = next(line for line in out.splitlines()
                       if line.startswith("workflow manual tasks left for you"))
        assert trailer.endswith("--resume --human-task T001=done"), trailer
        assert "--human-task T001=done" not in (payload.get("notice") or "")

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
        # Merge of fix round 28: the line the report relays says what
        # stopped and why; the way on is the section's alone.
        gate = next(line for line in out.splitlines()
                    if line.startswith("[MANUAL_GATE] "))
        assert gate == (
            "[MANUAL_GATE] implement ran nothing: 2 code tasks wait on manual "
            "task T001 (Create the module), which is not recorded done, and no "
            "other code task can run."), gate
        assert result["error"].startswith(gate[len("[MANUAL_GATE] "):])
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

    def test_the_report_the_runbook_asks_for_passes_the_stop_hook_and_says_the_way_on_once(
            self, repo, capsys):
        """Runbook step 3: report the `[MANUAL_GATE]` line and the deferred
        tasks, then end with the section `util next-command` prints. The
        Stop hook passes that report, and the resume command is in it once,
        in the section."""
        from tests.test_report_check import Transcript, run_check
        from utils import report_check

        _code, emit, _err, _result = self._emit(repo, capsys)
        code, lookup, err, _result = _run([
            "util", "next-command", str(repo.plan), "--after", "implement",
            "--host", "claude-code",
        ], capsys)
        assert code == 0, err
        gate = next(line for line in emit.splitlines()
                    if line.startswith("[MANUAL_GATE] "))[len("[MANUAL_GATE] "):]
        section = lookup.split(f"{report_check.NEXT_STEP_MARKDOWN}\n", 1)[1].split(
            f"\n{report_check.NEXT_STEP_MARKDOWN_END}", 1)[0]
        message = "\n\n".join([
            "## Implement stopped", gate,
            "Deferred: T001 Create the module; T002 and T003 wait on it.",
            section,
        ])
        assert message.count("--human-task T001=done") == 1, message
        transcript = (
            Transcript().prompt("/multi-llm:multi-llm --implement plans/sample.md")
            .bash(emit).bash(lookup).say(message)
            .write(repo.root.parent / "transcript.jsonl"))
        assert run_check(capsys, transcript) == ""

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
        "wait on them then run); or, for any of them you will not do, give "
        "its answer instead (--human-task T010=skip, --human-task T011=skip): "
        "it and the code tasks that wait on it are skipped.")
    # Fix round 29: one gating task's decline (review of round 28 #1).
    assert manual_gate_way_on(["T010"]) == (
        "To go on: do T010 by hand, then resume this phase with --human-task "
        "T010=done (the code tasks that wait on it then run); or, if you will "
        "not do T010, resume with --human-task T010=skip instead (it and the "
        "code tasks that wait on it are skipped).")
    answers = np_.manual_gate_decline_answers({
        "manual_close_ids": ["T010", "T011"],
        "manual_waiting": {"T010": ["T001", "T002"], "T011": ["T003"]},
        "manual_tasks": [{"id": "T010", "title": "Fixture"}, {"id": "T011"}]})
    assert answers == [
        {"flags": ["--resume", "--human-task", "T010=skip", "--human-task",
                   "T011=skip"],
         "why": ("if you will do none of T010 (Fixture), T011, skip them and "
                 "the 3 code tasks that wait on them, and implement goes on "
                 "without them")},
        {"flags": ["--resume", "--human-task", "T010=skip"],
         "why": ("if you will not do T010 (Fixture), skip it and the 2 code "
                 "tasks that wait on it, and implement goes on without them")},
        {"flags": ["--resume", "--human-task", "T011=skip"],
         "why": ("if you will not do T011, skip it and the code task that "
                 "waits on it, and implement goes on without them")},
    ]


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


# ---------------------------------------------------------------------------
# Fix round 29 (the Fable review of round 28, gate findings 1-4)
# ---------------------------------------------------------------------------

#: T001 and T002 stand alone, T003 waits on T002: the reviewer's plan.
PLAN3 = """# P

## Overview
x

## Tasks

### T001: Alpha task
Do alpha.
- Depends on: none
- Files to modify: src/main.py

### T002: Bravo manual
Do bravo by hand.
- Depends on: none
- Files to modify: src/main.py

### T003: Charlie task
Do charlie.
- Depends on: T002
- Files to modify: src/main.py
"""


def _plan3(repo, *manual, plan=PLAN3):
    _clean_env(repo.monkeypatch)
    _seed_generate_tasks(repo)
    repo.plan.write_text(plan, encoding="utf-8")
    _manual(repo, *manual)
    repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}]))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)


def _section(out: str) -> str:
    from utils import report_check

    return out.split(f"{report_check.NEXT_STEP_MARKDOWN}\n", 1)[1].split(
        f"\n{report_check.NEXT_STEP_MARKDOWN_END}", 1)[0]


class TestAnExplicitSkipIsTheUsersDecline:
    """Review of round 28, gate #1 (MEDIUM): under `--yes` a manual task the
    gate holds on had no decline route. `--resume --human-task T001=skip`
    printed `[HUMAN_TASK_ANSWERED] T001=skip`, recorded T001 deferred again
    and stopped on the same gate, exit 3, forever; only a false `=done`
    completed the run. Under the skip strategies an explicit `=skip` on a
    manual task is now the user's decline: the task is recorded skipped at
    the user's request, the tasks after it skipped with it, nothing is owed,
    the gate no longer holds and the run completes saying what it skipped.
    The gate's stop offers that answer as its second way forward."""

    TASKS = [
        {"task_id": "T001", "executor_role": "human", "depends_on": [],
         "task_title": "Create the module"},
        {"task_id": "T002", "depends_on": ["T001"], "task_title": "Wire it"},
        {"task_id": "T003", "depends_on": ["T002"], "task_title": "Add tests"},
    ]

    def test_the_plan_reads_the_skip_as_a_decline(self):
        from utils.write_batch_scheduler import (
            ANSWERED_SKIP_PREFIX, resolve_human_task_plan,
        )

        plan = resolve_human_task_plan(
            self.TASKS, strategy="skip-dependents", decisions={"T001": "skip"})
        assert plan.skipped["T001"] == (
            f"{ANSWERED_SKIP_PREFIX}T001=skip: you will not do this manual task")
        assert plan.skipped["T002"] == (
            "skipped — depends on skipped task T001 (Create the module)")
        assert plan.deferred() == {}
        assert sorted(plan.declined()) == ["T001", "T002", "T003"]
        # No answer: deferred, as before.
        plan = resolve_human_task_plan(
            self.TASKS, strategy="skip-dependents", decisions={})
        assert plan.skipped["T001"].startswith("deferred — manual task")
        # skip-continue keeps running the code after a manual task: the
        # decline only stops the task being owed.
        plan = resolve_human_task_plan(
            self.TASKS, strategy="skip-continue", decisions={"T001": "skip"})
        assert sorted(plan.skipped) == ["T001"] and plan.deferred() == {}
        assert plan.overrides == {"T002": ["T001"]}

    def test_a_manual_task_after_the_declined_one_is_declined_with_it(self):
        """T001 (manual) -> T002 -> T003 (manual) -> T004: the user will not
        do T001, so T003, after code that now never runs, is skipped with
        it, not owed as a deferral the user could close."""
        from utils.write_batch_scheduler import resolve_human_task_plan

        tasks = [
            {"task_id": "T001", "executor_role": "human", "depends_on": [],
             "task_title": "Fixture"},
            {"task_id": "T002", "depends_on": ["T001"], "task_title": "Code"},
            {"task_id": "T003", "executor_role": "human", "depends_on": ["T002"],
             "task_title": "Play it"},
            {"task_id": "T004", "depends_on": ["T003"], "task_title": "Tune"},
        ]
        plan = resolve_human_task_plan(
            tasks, strategy="skip-dependents", decisions={"T001": "skip"})
        assert plan.deferred() == {}
        assert sorted(plan.declined()) == ["T001", "T002", "T003", "T004"]
        assert plan.skipped["T003"] == (
            "skipped — depends on skipped task T002 (Code)")
        # Declining the later one alone leaves the first owed.
        plan = resolve_human_task_plan(
            tasks, strategy="skip-dependents", decisions={"T003": "skip"})
        assert sorted(plan.deferred()) == ["T001", "T002"]
        assert sorted(plan.declined()) == ["T003", "T004"]

    def _gate(self, repo, capsys, *, native=False):
        _clean_env(repo.monkeypatch)
        _seed_generate_tasks(repo)
        _manual(repo, "T001")
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        leg = ["--host", "claude-code"] if native else [
            "--exec-subagents", "--host", "codex"]
        code, out, err, result = _run(
            ["implement", str(repo.plan), "--yes", *leg], capsys)
        assert code == 3, err
        return leg, result

    def test_the_executor_gate_offers_the_decline(self, repo, capsys):
        from utils.write_batch_scheduler import (
            HEADLINE_BESIDE_SECTION_KEY, answers_cover,
        )

        _leg, result = self._gate(repo, capsys)
        assert "--human-task T001=skip" in result["error"], result["error"]
        phase_dir = output_dir_path(repo.plan) / "implement"
        stop = json.loads((phase_dir / "write_batch_report.json").read_text())["stop"]
        assert stop["answers"] == [{
            "flags": ["--resume", "--human-task", "T001=skip"],
            "why": ("if you will not do T001 (Create the module), skip it and "
                    "the 2 code tasks that wait on it, and implement goes on "
                    "without them"),
        }]
        beside = stop[HEADLINE_BESIDE_SECTION_KEY]
        assert "=skip" not in beside and "=done" not in beside
        assert answers_cover(stop["headline"], beside, stop)
        code, out, err, lookup = _run([
            "util", "next-command", str(repo.plan), "--blocked", "implement",
            "--host", "codex"], capsys)
        assert code == 0, err
        resume = lookup["user_resume"]
        assert resume["flags"] == ["--resume", "--human-task", "T001=done"]
        assert [a["flags"] for a in resume["answers"]] == [
            ["--resume", "--human-task", "T001=skip"]]
        assert resume["answers"][0]["command"].endswith(
            "--resume --human-task T001=skip")
        markdown = resume["markdown"]
        assert markdown.index("T001=done") < markdown.index(
            "Or answer the stop with one of these:") < markdown.index("T001=skip")
        # The servers still swap in the line beside the section.
        assert resume["error_line"] == result["error"]
        assert resume["error_beside_section"] == beside
        # The printed block offers it too.
        assert "if you will not do T001" in _err_block(repo, capsys)

    def test_the_decline_completes_the_run(self, repo, capsys):
        from utils.write_batch_scheduler import ANSWERED_SKIP_PREFIX

        leg, _result = self._gate(repo, capsys)
        code, out, err, result = _run([
            "implement", str(repo.plan), "--yes", *leg, "--resume",
            "--human-task", "T001=skip"], capsys)
        assert code == 0, err
        assert result["state"] == "done", result
        state = _state(repo)
        assert state["task_status"] == {
            "T001": "skipped", "T002": "skipped", "T003": "skipped"}
        reasons = state["task_status_reasons"]
        assert reasons["T001"].startswith(f"{ANSWERED_SKIP_PREFIX}T001=skip")
        assert reasons["T002"].startswith("skipped — depends on skipped task T001")
        assert "implement" in (state.get("phases_completed") or {})
        summary = (output_dir_path(repo.plan) / "implement" / "summary.md").read_text()
        assert "## Skipped at your request" in summary
        assert "## Manual Tasks (Deferred)" not in summary
        assert "Waiting On A Manual Task" not in summary
        if "--exec-subagents" in leg:
            # The closing line says what the run skipped.
            assert ("phase complete in 0s · T001 skipped at your request; "
                    "T002, T003 skipped with a skipped task") in out + err
        # Nothing is owed: no manual task left for the user.
        step = result.get("user_next_step") or {}
        assert not step.get("manual_tasks"), step

    def test_the_native_gate_offers_the_decline(self, repo, capsys):
        leg, result = self._gate(repo, capsys, native=True)
        assert "--human-task T001=skip" in result["error"], result["error"]
        code, out, err, lookup = _run([
            "util", "next-command", str(repo.plan), "--after", "implement",
            "--host", "claude-code"], capsys)
        assert code == 0, err
        section = _section(out)
        assert section.index("--resume --human-task T001=done") < section.index(
            "--resume --human-task T001=skip"), section
        assert "if you will not do T001 (Create the module)" in section
        code, out, err, _r = _run(["status", str(repo.plan)], capsys)
        lines = out.splitlines()
        answer = next(i for i, line in enumerate(lines)
                      if "or answer the stop with:" in line)
        assert lines[answer].rstrip().endswith("--resume --human-task T001=skip")
        assert "if you will not do T001" in lines[answer + 1]
        # A tool route's lookup of the same stop, with no stop record.
        code, out, err, blocked = _run([
            "util", "next-command", str(repo.plan), "--blocked", "implement",
            "--host", "claude-code"], capsys)
        assert code == 0, err
        resume = blocked["user_resume"]
        assert resume["flags"] == ["--resume", "--human-task", "T001=done"]
        assert [a["flags"] for a in resume["answers"]] == [
            ["--resume", "--human-task", "T001=skip"]]
        assert "--human-task T001=skip" in resume["markdown"]

    def test_the_native_decline_completes_the_emit(self, repo, capsys):
        leg, _result = self._gate(repo, capsys, native=True)
        code, out, err, result = _run([
            "implement", str(repo.plan), "--yes", *leg, "--resume",
            "--human-task", "T001=skip"], capsys)
        assert code == 0, err
        assert "[MANUAL_GATE]" not in out
        assert "Skipped at your request" in out
        assert "T001 Create the module: skipped — you answered --human-task T001=skip" in out
        assert "Deferred tasks (" not in out


def _err_block(repo, capsys) -> str:
    """The exit-3 block a re-run of the gated executor prints."""
    code, out, err, _result = _run([
        "implement", str(repo.plan), "--yes", "--exec-subagents",
        "--host", "codex", "--resume"], capsys)
    assert code == 3, err
    return err


class TestOneRecordDoneCommandAtAGate:
    """Review of round 28, gate #2 (LOW): T001 a free manual task, T002 the
    manual task gating T003. The stop's section said `--resume --human-task
    T002=done`; summary.md's note, `status`'s trailer and the lookup's
    manual-task note said `--resume --human-task T001=done --human-task
    T002=done`. While the gate holds there is one command, the gate's
    resume; T001 stays listed."""

    def _stop(self, repo, capsys, *, native=False):
        _plan3(repo, "T001", "T002")
        leg = ["--host", "claude-code"] if native else [
            "--exec-subagents", "--host", "codex"]
        code, out, err, result = _run(
            ["implement", str(repo.plan), "--yes", *leg], capsys)
        assert code == 3, err
        return result

    def test_summary_and_status_give_the_gates_resume_only(self, repo, capsys):
        self._stop(repo, capsys)
        summary = (output_dir_path(repo.plan) / "implement" / "summary.md").read_text()
        assert "| T001 | deferred" in summary
        assert "T001=done" not in summary, summary
        assert "--resume --human-task T002=done" in summary
        code, out, err, _r = _run(["status", str(repo.plan), "--host", "codex"], capsys)
        trailer = next(line for line in out.splitlines()
                       if "manual tasks left for you:" in line)
        assert "T001 Alpha task" in trailer
        assert "T001=done" not in trailer, trailer
        assert trailer.rstrip().endswith("--resume --human-task T002=done"), trailer
        assert "on manual task T002" in trailer, trailer

    def test_a_tool_routes_poll_names_the_tasks_and_no_command(self, repo, capsys):
        """The detached run's terminal poll: its notice names T001 and T002
        and leaves the one command to the section (45efcf5's rule, which
        the free manual task's second command used to bypass)."""
        from utils import run_status, status_file

        result = self._stop(repo, capsys)
        phase_dir = output_dir_path(repo.plan) / "implement"
        progress = json.loads((phase_dir / ".status.json").read_text())["progress"]
        status_file.claim_run(str(phase_dir), {
            "run_id": "run-gate-0029", "state": "failed", "exit_code": 3,
            "error": result["error"], "detached": True, "mode": "implement",
            "started_at": "2026-10-09T10:00:00", "host": "codex"})
        status_file.update_status(str(phase_dir), {"progress": progress})
        capsys.readouterr()
        repo.monkeypatch.setenv("MULTI_LLM_ROUTE", "package")
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            code = run_status.main([
                "--plan-file", str(repo.plan), "--view", "run", "--phase",
                "implement", "--host", "codex"])
            payload = ctx.payload(code)
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert code == 3, out
        notice = payload.get("notice") or ""
        assert "T001 Alpha task" in notice and "T002 Bravo manual" in notice, notice
        assert "=done" not in notice, notice
        trailer = [line for line in out.splitlines()
                   if line.startswith("workflow manual tasks left for you")]
        assert trailer and "=done" not in trailer[0], trailer

    @pytest.mark.parametrize("native", [False, True])
    def test_the_lookup_gives_one_command(self, repo, capsys, native):
        self._stop(repo, capsys, native=native)
        host = "claude-code" if native else "codex"
        code, out, err, result = _run([
            "util", "next-command", str(repo.plan), "--after", "implement",
            "--host", host], capsys)
        assert code == 0, err
        step = result["user_next_step"]
        assert step["command"].endswith("--resume --human-task T002=done")
        assert "manual_tasks_command" not in step, step
        assert [t["id"] for t in step["manual_tasks"]] == ["T001", "T002", "T003"]
        assert "T001=done" not in step["markdown"]
        assert "[MANUAL_TASK_COMMAND]" not in out
