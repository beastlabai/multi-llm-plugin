"""What a native implement (Claude Code) says about itself, fix round 16.

Live run 6 (2026-10-06), Claude Code racing 06: T011 was a manual task,
deferred under ``--yes``'s ``skip-dependents``. Implement's own result and
``implement/summary.md`` said only "T011 … deferred, left for the user"; how
to record it done (``--human-task T011=done``) first appeared in phase 08's
``[WORKFLOW_COMPLETE]`` block, two phases later. The tool routes carry that
how-to-close note (``manual_tasks_note`` / ``[MANUAL_TASK_NOTE]``) in
implement's own result. Now the native leg does too:

* the emit prints ``[MANUAL_TASK_NOTE]`` under its "Deferred tasks" list and
  hands it over as the task JSON's ``manual_tasks_note``, for the host's
  summary and report;
* ``util next-command --after implement``, the report step's lookup, prints
  the deferred tasks and the note beside the next command.

The note's command answers only the manual tasks themselves: a code task
skipped with one is not answerable ``done`` (the parser refuses a code task
that did not fail), and it runs on the resume once its manual task is done.

The fixture is ``test_human_task_answers``' plan with a code task (T006)
after the manual one (T005).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, host_capabilities  # noqa: E402
from utils import next_phase as np_  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402

from tests.test_human_task_answers import PLAN_WITH_CODE_DEPENDENT  # noqa: E402
from tests.test_implement_stop_account import make_repo  # noqa: E402

PLAN_ARG = "plans/checked.md"
MANUAL_TITLE = "Play through by hand"
DEPENDENT_TITLE = "Edit: after the manual step"
#: How T006, skipped with the manual T005, is named (fix round 17): implement
#: runs it on the resume that records T005 done.
WAITS = "(waits on T005: implement runs it once T005 is recorded done)"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    obj = make_repo(tmp_path, monkeypatch, PLAN_WITH_CODE_DEPENDENT)
    yield obj
    registry._config = None
    registry._config_key = None


def _close_command(plan_arg: str, ids) -> str:
    """The close command alone, as the next-step lookup prints it beside the
    note's lead sentence (fix round 22)."""
    template = host_capabilities.next_command_template(
        "claude-code", np_.resolve_route("claude-code", stderr=io.StringIO()))
    return np_.close_command(template, plan_arg, ids)


def _close_note(plan_arg: str, ids) -> str:
    """The note for *ids*, its command spelled the way Claude Code types it."""
    template = host_capabilities.next_command_template(
        "claude-code", np_.resolve_route("claude-code", stderr=io.StringIO()))
    return np_.manual_close_note(np_.close_command(template, plan_arg, ids))


def _emit(capsys, *flags: str, yes: bool = True):
    assert multi_llm.main(
        ["implement", PLAN_ARG, "--host", "claude-code",
         *(["--yes"] if yes else ["--no-confirm"]), *flags]
    ) == 0
    out = capsys.readouterr().out
    path = next(
        line.split(": ", 1)[1] for line in out.splitlines()
        if line.startswith("Output file: ")
    )
    return json.loads(Path(path).read_text(encoding="utf-8")), out


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _finish_the_native_run(repo, capsys, handoff) -> None:
    """What the host does: record each handed-out task, close the phase."""
    phase = str(repo.phase_dir)
    for batch in handoff["batches"]:
        for task in batch["tasks"]:
            extra = ["--check", "passed"] if task["id"] == "T002" else []
            code, _out, err = _util(
                capsys, "batch-done", "--phase-dir", phase, "--task-id", task["id"],
                "--status", "completed", *extra)
            assert code == 0, err
    code, _out, err = _util(
        capsys, "finalize-tracking", "--state-file", handoff["state_file"],
        "--mark-completed")
    assert code == 0, err


class TestTheEmitCarriesTheNote:

    def test_the_deferred_list_ends_with_the_note(self, repo, capsys):
        handoff, out = _emit(capsys)
        assert sorted(handoff["skipped_tasks"]) == ["T005", "T006"]
        note = _close_note(PLAN_ARG, ["T005"])
        assert handoff["manual_tasks_note"] == note
        lines = out.splitlines()
        heading = next(i for i, line in enumerate(lines)
                       if line.startswith("Deferred tasks ("))
        assert (
            "section, with the text of the [MANUAL_TASK_NOTE] line below the "
            "list under it, verbatim; give that text in your report too):"
        ) in lines[heading]
        assert f"[MANUAL_TASK_NOTE] {note}" in lines[heading:]
        # The dependent is not answerable `done`: it runs on that resume.
        assert "--human-task T005=done" in note
        assert "T006=done" not in note

    def test_the_note_is_the_command_the_resume_accepts(self, repo, capsys):
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        assert multi_llm.main(
            ["implement", PLAN_ARG, "--host", "claude-code", "--yes", "--resume",
             "--human-task", "T005=done"]
        ) == 0
        capsys.readouterr()
        statuses = get_or_create_state(repo.plan).implementation_task_statuses()
        assert statuses["T005"] == "completed"

    def test_no_deferred_manual_task_no_note(self, repo, capsys):
        handoff, out = _emit(capsys, "--force-strategy", "pause-and-ask", yes=False)
        assert "manual_tasks_note" not in handoff
        assert "[MANUAL_TASK_NOTE]" not in out


class TestTheReportLookupCarriesIt:

    def test_util_next_command_after_implement_names_them_and_the_note(
        self, repo, capsys,
    ):
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        code, out, err = _util(
            capsys, "next-command", PLAN_ARG, "--after", "implement",
            "--host", "claude-code")
        assert code == 0, err
        command = _close_command(PLAN_ARG, ["T005"])
        lines = [line for line in out.splitlines() if line.startswith("[")]
        assert lines[0].startswith("[NEXT_COMMAND] ")
        assert lines[0].endswith(f"--review-code {PLAN_ARG}")
        assert lines[-4:] == [
            f"[MANUAL_TASK] T005 {MANUAL_TITLE}",
            f"[MANUAL_TASK] T006 {DEPENDENT_TITLE} {WAITS}",
            f"[MANUAL_TASK_NOTE] {np_.manual_close_lead()}",
            f"[MANUAL_TASK_COMMAND] {command}",
        ]
        step = cli_common.parse_result_line(out)["user_next_step"]
        assert step["complete"] is False
        assert step["command"] == lines[0].split(" ", 1)[1]
        assert step["manual_tasks"] == [
            {"id": "T005", "title": MANUAL_TITLE},
            {"id": "T006", "title": DEPENDENT_TITLE, "waits_on": "T005"},
        ]
        assert step["manual_tasks_note"] == np_.manual_close_lead()
        assert step["manual_tasks_command"] == command

    def test_after_another_phase_it_names_them_too(self, repo, capsys):
        """Fix round 17: the two packages of round 16 disagreed here. Every
        command answer carries the tasks implement deferred while they are
        owed (`next_phase_commands`' one writer), which is what the relay
        rule and the shims describe: not only the `--after implement` one."""
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        code, out, err = _util(
            capsys, "next-command", PLAN_ARG, "--after", "generate-tasks",
            "--host", "claude-code")
        assert code == 0, err
        command = _close_command(PLAN_ARG, ["T005"])
        lines = [line for line in out.splitlines() if line.startswith("[")]
        assert lines[0].startswith("[NEXT_COMMAND] ")
        assert lines[-4:] == [
            f"[MANUAL_TASK] T005 {MANUAL_TITLE}",
            f"[MANUAL_TASK] T006 {DEPENDENT_TITLE} {WAITS}",
            f"[MANUAL_TASK_NOTE] {np_.manual_close_lead()}",
            f"[MANUAL_TASK_COMMAND] {command}",
        ]
        step = cli_common.parse_result_line(out)["user_next_step"]
        assert step["complete"] is False
        assert [task["id"] for task in step["manual_tasks"]] == ["T005", "T006"]
        assert step["manual_tasks_command"] == command

    def test_the_complete_answer_closes_only_the_manual_task(self, repo, capsys):
        """The same rule on `[WORKFLOW_COMPLETE]`, both legs' builder: T006,
        skipped with T005, is listed as owed but never answered `done`."""
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        state = get_or_create_state(repo.plan)
        state.mark_phase_completed("code-review")
        state.mark_phase_completed("apply-code-fixes")
        state.save()
        code, out, err = _util(
            capsys, "next-command", PLAN_ARG, "--after", "apply-code-fixes",
            "--host", "claude-code")
        assert code == 0, err
        assert "[WORKFLOW_COMPLETE]" in out
        assert f"[MANUAL_TASK_COMMAND] {_close_command(PLAN_ARG, ['T005'])}" in out


class TestTheExecutorSummaryCarriesItToo:
    """Parity: the `--exec-subagents` leg's own summary.md listed T005 under
    `## Manual Tasks (Deferred)` with no word on how to close it either."""

    def test_the_deferred_table_ends_with_the_note(self, repo, capsys):
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
            ["T001", "T002", "T003", "T004", "T006"]))
        assert multi_llm.main(
            ["implement", PLAN_ARG, "--host", "claude-code", "--yes",
             "--exec-subagents"]
        ) == 0
        capsys.readouterr()
        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        summary = Path(report["summary_file"]).read_text(encoding="utf-8")
        section = summary.split("## Manual Tasks (Deferred)", 1)[1].split("\n## ", 1)[0]
        assert "| T005 |" in section and "| T006 |" in section
        assert _close_note(PLAN_ARG, ["T005"]) in section


class TestATaskSkippedWithAManualOneIsNamedApart:
    """Fix round 17 (review of round 16): T006, a code task skipped because
    it depends on the manual T005, read "deferred manual" on every `status`
    surface, and the relay rule called each `[MANUAL_TASK]` a task "no
    phase will run", though the note's `--resume --human-task T005=done`
    runs it. Each surface now names it for what it is."""

    def _executor_run(self, repo, capsys):
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
            ["T001", "T002", "T003", "T004", "T006"]))
        assert multi_llm.main(
            ["implement", PLAN_ARG, "--host", "claude-code", "--yes",
             "--exec-subagents"]
        ) == 0
        capsys.readouterr()

    def test_the_executor_run_s_progress_and_tally_lines(self, repo, capsys):
        self._executor_run(repo, capsys)
        multi_llm.main(["status", PLAN_ARG, "--host", "claude-code", "--view", "run"])
        out = capsys.readouterr().out
        progress = next(l for l in out.splitlines() if "progress:" in l)
        assert "deferred manual: T005 (skip-dependents)" in progress, progress
        assert "skipped (depends on manual T005): T006" in progress, progress
        assert "T005, T006" not in progress
        tally = next(l for l in out.splitlines() if "all tasks:" in l)
        assert tally.endswith(
            "deferred manual: T005; skipped (depends on manual T005): T006"), tally

    def test_the_workflow_trailer_says_what_it_waits_on(self, repo, capsys):
        self._executor_run(repo, capsys)
        multi_llm.main(["status", PLAN_ARG, "--host", "claude-code"])
        out = capsys.readouterr().out
        assert (
            f"manual tasks left for you: T005 {MANUAL_TITLE}; "
            f"T006 {DEPENDENT_TITLE} {WAITS}"
        ) in out, out

    def test_the_relay_rule_no_longer_says_no_phase_will_run_them(self):
        import modes

        line = modes.TOOL_USER_NEXT_STEP_LINE
        assert "no phase will run" not in line
        # Fix round 22: the `markdown` the line says to copy lists such a
        # task with what it waits on.
        markdown = np_.render_next_step_markdown(np_.NextPhase(
            phase=None, command=None, complete=True,
            manual_tasks=(np_.ManualTask("T006", DEPENDENT_TITLE, "T005"),)))
        assert f"- T006 {DEPENDENT_TITLE} {WAITS}" in markdown
        reference = " ".join(
            (SKILL_DIR / "references" / "summary-format.md").read_text(
                encoding="utf-8").split())
        assert "the plugin will not run them" not in reference
        assert (
            "one with `waits_on` is a task skipped with that manual task, which "
            "implement runs once that one is recorded done"
        ) in reference


class TestTheHostIsToldToCarryIt:

    def test_the_core_and_the_shim_say_where_the_note_goes(self):
        core = (SKILL_DIR / "instructions" / "implement.md").read_text(encoding="utf-8")
        assert core.count(
            "ending with the text of step 2's `[MANUAL_TASK_NOTE]` line (the task "
            "JSON's `manual_tasks_note`), verbatim: it says how to record them done"
        ) == 2
        assert (
            "Each manual task step 2 deferred, as the user's to do, then the "
            "next-step lookup's `[MANUAL_TASK_NOTE]` text verbatim and its "
            "`[MANUAL_TASK_COMMAND]` alone in a fenced block"
        ) in core
        skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        # Fix round 17: after any phase, not only implement (the merged
        # renderer lists them after every command block while owed).
        assert (
            "Those lines also follow a `[NEXT_COMMAND]` block, after any phase, "
            "while implement's manual tasks are owed: list those tasks and give "
            "that note the same way, after the commands, and never write "
            "\"Workflow complete.\" for them."
        ) in skill
        assert "After implement those lines can also follow" not in skill
        reference = " ".join(
            (SKILL_DIR / "references" / "summary-format.md").read_text(
                encoding="utf-8").split())
        assert (
            "A `user_next_step` that names a `command` can carry `manual_tasks` "
            "and `manual_tasks_note` too"
        ) in reference


class TestStatusAfterACompletedNativeImplement:
    """Live run 6 (Claude Code racing 06): after a native implement that
    completed, `status plans/racing-game.md --host claude-code --view run`
    printed "no run recorded for phase implement". The native leg writes no
    run record, report or mirror, so `status` reads its tasks back from
    `state.json` (fix round 14) — but only for a run the host stopped
    part-way; a completed one was left to the workflow view."""

    def _status(self, capsys, *flags):
        multi_llm.main(
            ["status", PLAN_ARG, "--host", "claude-code", "--view", "run", *flags])
        return capsys.readouterr().out

    @pytest.mark.parametrize("flags", [(), ("--phase", "implement")])
    def test_the_run_view_reads_it_back_from_state(self, repo, capsys, flags):
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        assert not (repo.phase_dir / "write_batch_report.json").exists()
        out = self._status(capsys, *flags)
        assert "no run recorded for phase implement" not in out
        assert (
            "  implement  done on the host (marked completed) — no run record, "
            "since the host ran these tasks itself; read from state.json"
        ) in out, out
        line = next(l for l in out.splitlines() if l.strip().startswith("all tasks:"))
        assert line.split("all tasks:", 1)[1].strip() == (
            "4/6 done; completed: T001, T002, T003, T004; "
            "deferred manual: T005; skipped (depends on manual T005): T006"
        )

    def test_json_carries_the_same_tally(self, repo, capsys):
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        multi_llm.main(["status", PLAN_ARG, "--host", "claude-code", "--view", "run",
                        "--json"])
        out = capsys.readouterr().out
        payload = json.loads(out[: out.index("MULTI_LLM_RESULT")])
        tasks = payload["implement_tasks"]
        assert tasks["completed"] is True
        assert tasks["buckets"]["completed"] == ["T001", "T002", "T003", "T004"]
        assert tasks["buckets"]["deferred"] == ["T005"]
        assert tasks["buckets"]["deferred_dependents"] == ["T006"]
        assert tasks["waits_on"] == {"T006": "T005"}

    def test_a_stopped_run_still_reads_not_marked_completed(self, repo, capsys):
        handoff, _out = _emit(capsys)
        _util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
              "--task-id", "T001", "--status", "completed")
        out = self._status(capsys, "--phase", "implement")
        assert "implement  not finished on the host (not marked completed)" in out
        assert "1/6 done" in out

    def test_its_check_findings_follow_the_trailer_s_rule(self, repo, capsys):
        """Shown, like the workflow trailer's, until a review-code that judged
        them completed or was skipped (fix round 11); a completed native
        implement's block must not bring them back after that."""
        handoff, _out = _emit(capsys)
        phase = str(repo.phase_dir)
        for batch in handoff["batches"]:
            for task in batch["tasks"]:
                extra = (["--check", "findings", "--finding", "doctype must be lowercase"]
                         if task["id"] == "T002" else [])
                _util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                      task["id"], "--status", "completed", *extra)
        state = get_or_create_state(repo.plan)
        state.update_task_status("T004", "completed")
        state.save()
        out = self._status(capsys)
        assert "check findings T002: doctype must be lowercase" in out
        state = get_or_create_state(repo.plan)
        state.mark_phase_completed("implement")
        state.mark_phase_skipped("code-review", "not reviewing this one")
        state.save()
        out = self._status(capsys)
        assert "implement  done on the host (marked completed)" in out
        assert "check findings" not in out
