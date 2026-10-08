"""The next step arrives already written, its commands fenced (fix round 22).

Live run 9, the hosts that read `user_next_step` off a tool or file route:

* Codex 04 printed `--implement` and the optional `--apply-task-suggestions`
  as bare paragraphs, though the tool description says each goes in its
  own fenced block;
* the manual-task close command was folded into prose (Codex 07, OpenCode
  06-08) because the rule said to give `manual_tasks_note`, a sentence that
  ended with the command, "verbatim, as plain text", and the result's
  `notice` carried the same sentence again (twice at 07/08).

`user_next_step.markdown` is the step as the report's last section, each
command alone in its own fence, every why and note as plain text, and the
routes say to copy it as it is. The close command is its own key
(`manual_tasks_command`, and `[MANUAL_TASK_COMMAND]` on the native leg, which
the Stop hook holds the report to like every other command), and a terminal
poll's `notice` no longer repeats the note when `user_next_step` carries it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

import modes  # noqa: E402
from utils import cli_common, report_check, run_status, status_file  # noqa: E402
from utils import next_phase as np_  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402

from tests.test_manual_tasks_on_completion import (  # noqa: E402,F401
    P, TITLES, _complete_plan, _run, workdir,
)
from tests.test_report_check import Transcript, decision_of, run_check  # noqa: E402

CLOSE = "$multi-llm:multi-llm --implement plans/r.md --resume --human-task T008=done"
LEAD = np_.manual_close_lead()
FENCE = "```"


def _fenced(command: str) -> str:
    return f"{FENCE}\n{command}\n{FENCE}"


def _answer(**extra) -> np_.NextPhase:
    return np_.NextPhase(
        phase="implement", command="$multi-llm:multi-llm --implement plans/r.md",
        optional_before=(np_.OptionalStep(
            "apply-task-suggestions",
            "$multi-llm:multi-llm --apply-task-suggestions plans/r.md",
            "review-tasks left 9 valid task suggestion groups unapplied."),),
        manual_tasks=(np_.ManualTask("T008", "Manual playthrough"),),
        manual_tasks_note=LEAD, manual_tasks_command=CLOSE, **extra,
    )


class TestTheMarkdown:

    def test_a_command_answer(self):
        payload = _answer().to_dict()
        assert payload["markdown"] == "\n\n".join([
            _fenced("$multi-llm:multi-llm --implement plans/r.md"),
            "Optional, before that:",
            "review-tasks left 9 valid task suggestion groups unapplied.",
            _fenced("$multi-llm:multi-llm --apply-task-suggestions plans/r.md"),
            "Manual task left for you:\n- T008 Manual playthrough",
            LEAD,
            _fenced(CLOSE),
        ])
        assert payload["manual_tasks_note"] == LEAD
        assert payload["manual_tasks_command"] == CLOSE
        assert "Workflow complete." not in payload["markdown"]

    def test_a_complete_answer(self):
        complete = np_.NextPhase(
            phase=None, command=None, complete=True,
            manual_tasks=(np_.ManualTask("T008", "Manual playthrough"),),
            manual_tasks_note=LEAD, manual_tasks_command=CLOSE,
        )
        assert complete.to_dict()["markdown"] == "\n\n".join([
            "Workflow complete.",
            "Manual task left for you:\n- T008 Manual playthrough",
            LEAD,
            _fenced(CLOSE),
        ])
        bare = np_.NextPhase(phase=None, command=None, complete=True)
        assert bare.to_dict()["markdown"] == "Workflow complete."

    def test_the_note_leads_into_the_command_and_holds_none(self):
        assert LEAD.endswith(":")
        assert "--human-task" in LEAD and "$" not in LEAD
        assert np_.manual_close_note(CLOSE) == f"{LEAD} {CLOSE}"

    def test_a_fence_outgrows_any_backtick_run_in_the_command(self):
        assert np_.fenced_command("a ```b``` c") == "````\na ```b``` c\n````"

    def test_the_native_block_prints_the_command_on_its_own_line(self):
        lines = np_.render_next_phase_lines(_answer())
        assert lines[-2:] == [
            f"[MANUAL_TASK_NOTE] {LEAD}", f"[MANUAL_TASK_COMMAND] {CLOSE}"]


class TestTheLookupAndTheStopHook:

    def test_util_next_command_carries_the_command_apart(self, workdir, capsys):
        _complete_plan(workdir)
        code, out, err, result = _run(
            ["util", "next-command", P, "--after", "apply-code-fixes"], capsys)
        assert code == 0, err
        step = result["user_next_step"]
        command = step["manual_tasks_command"]
        assert command.endswith("--resume --human-task T011=done --human-task T012=done")
        assert step["manual_tasks_note"] == LEAD
        assert f"[MANUAL_TASK_COMMAND] {command}" in out.splitlines()
        assert step["markdown"].endswith(_fenced(command))

    def _check(self, tmp_path, capsys, output, message):
        path = (Transcript().prompt("go").bash(output).say(message)
                .write(tmp_path / "t.jsonl"))
        return run_check(capsys, path)

    def test_a_report_that_folds_the_command_into_prose_is_asked_for_it(
        self, workdir, tmp_path, capsys,
    ):
        _complete_plan(workdir)
        _code, out, _err, result = _run(
            ["util", "next-command", P, "--after", "apply-code-fixes"], capsys)
        command = result["user_next_step"]["manual_tasks_command"]
        tasks = f"T011 {TITLES['T011']}, T012 {TITLES['T012']}"
        prose = f"Workflow complete.\n\nLeft for you: {tasks}. {LEAD} {command}\n"
        reason = decision_of(self._check(tmp_path, capsys, out, prose))["reason"]
        assert _fenced(command) in reason
        fenced = f"Workflow complete.\n\nLeft for you: {tasks}. {LEAD}\n\n{_fenced(command)}\n"
        assert self._check(tmp_path, capsys, out, fenced) == ""


class TestTheNoticeDoesNotRepeatTheNote:

    def test_a_terminal_poll_that_carries_the_step_drops_the_note(self, workdir, capsys):
        plan = _complete_plan(workdir)
        phase_dir = output_dir_path(plan) / "apply-fixes"
        phase_dir.mkdir(parents=True, exist_ok=True)
        status_file.write_status(str(phase_dir), {
            "phase": "apply-code-fixes", "state": "complete"})
        status_file.claim_run(str(phase_dir), {
            "run_id": "run-manual-0002", "state": "done", "exit_code": 0,
            "detached": True, "mode": "apply-code-fixes"})
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--phase", "apply-code-fixes"]) == 0
            payload = ctx.payload(0)
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        step = payload["user_next_step"]
        assert step["manual_tasks_command"] in step["markdown"]
        # Fix round 26 (live run 11, Codex 06/08 and OpenCode 06-08): the
        # step's markdown lists the tasks, so the notice no longer names them
        # either; round 22 kept the names, and the hosts relayed them above
        # the copied section.
        notice = payload.get("notice") or ""
        assert "manual tasks left for you" not in notice, notice
        assert "--human-task" not in notice, notice
        assert "No later phase waits" not in notice
        # The status text a person reads still says how to close them.
        assert "--human-task T011=done" in out


class TestTheNoticeDoesNotRepeatTheStep:

    def test_a_poll_that_carries_the_step_drops_suggested_next(
        self, tmp_path, monkeypatch, capsys,
    ):
        """Fix round 24 (live run 10, OpenCode 03 and Codex 01): a terminal
        poll's `notice` said "suggested next: implement (review-tasks /
        apply-task-suggestions optional, not run)" beside `user_next_step`,
        whose `markdown` says it with the commands. The hosts relayed both,
        and OpenCode 03 reordered the step after the notice's words. The
        notice leaves the next step to `user_next_step`; the text view a
        person reads keeps the line."""
        from tests.test_next_phase import RT_ACTIONABLE, S, UP_TO_REVIEW_TASKS, _build

        plan = _build(tmp_path, S(UP_TO_REVIEW_TASKS, files=RT_ACTIONABLE))
        monkeypatch.chdir(tmp_path)
        for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
            monkeypatch.delenv(name, raising=False)
        phase_dir = output_dir_path(plan) / "review-tasks"
        status_file.write_status(str(phase_dir), {
            "phase": "review-tasks", "state": "complete"})
        status_file.claim_run(str(phase_dir), {
            "run_id": "run-rt-0024", "state": "done", "exit_code": 0,
            "detached": True, "mode": "review-tasks"})
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--phase", "review-tasks"]) == 0
            payload = ctx.payload(0)
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        step = payload["user_next_step"]
        assert step["command"].endswith("plans/foo.md"), step
        assert "--implement" in step["command"], step
        assert "suggested next" not in (payload.get("notice") or ""), payload
        # Fix round 26: what the review left unapplied is the optional
        # entry's why, which the step's markdown says; the notice does not.
        assert "review-tasks left 1 valid task suggestion group" in step["markdown"]
        assert "review-tasks left" not in (payload.get("notice") or ""), payload
        assert "workflow suggested next: implement" in out


class TestTheRoutesSayToCopyIt:
    """Every route that reads `user_next_step` off a result: one short rule."""

    @pytest.mark.parametrize("name", [
        "modes.TOOL_USER_NEXT_STEP_LINE",
        "hosts/codex/plugin-skill.md.tmpl",
        "hosts/opencode/plugin-command.md.tmpl",
        "hosts/codex/shim.md.tmpl",
        "hosts/opencode/shim.md.tmpl",
    ])
    def test_the_rule_names_the_markdown(self, name):
        if name.startswith("modes."):
            text = modes.TOOL_USER_NEXT_STEP_LINE
        else:
            text = (SKILL_ROOT / name).read_text(encoding="utf-8")
        text = " ".join(text.split())
        assert "its `markdown`, copied as it is" in text, name
        assert "verbatim, as plain text outside any fenced block" not in text, name

    def test_claude_code_fences_the_native_command(self):
        """Fix round 24: Claude Code copies the lookup's section, which
        fences the close command alone like every other route's."""
        text = " ".join((SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").split())
        assert "`[NEXT_STEP_MARKDOWN]`" in text
        assert "each command alone in its fenced block" in text


def test_a_stops_answers_follow_the_resume_under_their_own_lead():
    """The native lookup's answer (a stopped implement): the resume's why
    and command, then the answers under one line saying what they are, as
    `status` says "or answer the stop with:" — never "Optional, before
    that:", the label OpenCode's accept 01 put on each of them."""
    resume = "/multi-llm:multi-llm --implement plans/r.md --resume"
    answer = np_.NextPhase(
        phase="implement", command=resume, why="runs the tasks that can run",
        answers=(np_.AnswersStep(f"{resume} --human-task T005=skip",
                                 "leave T005 undone", ("T005=skip",)),),
    )
    assert answer.to_dict()["markdown"] == "\n\n".join([
        "runs the tasks that can run",
        _fenced(resume),
        np_.ANSWERS_LEAD,
        "leave T005 undone",
        _fenced(f"{resume} --human-task T005=skip"),
    ])
    assert "Optional, before that:" not in answer.to_dict()["markdown"]
