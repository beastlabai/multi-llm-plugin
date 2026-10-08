"""A result hands the host each sentence once (fix round 26, live run 11 #4).

Live run 11 (2026-10-08), what the hosts were given to relay:

* Codex 04, 06 and 08 pasted the terminal poll's `notice` above
  `user_next_step.markdown`, so "review-tasks left 9 valid task suggestion
  groups unapplied …" (04) and the manual-task note (06, 08) reached the user
  twice; OpenCode 03, 04 and 06-08 did the same. The notice carried what the
  section already says: the optional entry's why and the manual tasks.
* At a stop, Codex 01b relayed "The suggested next phase is `implement`;
  `review-tasks` and `apply-task-suggestions` were optional and not run",
  from the poll's "suggested next: implement  (review-tasks /
  apply-task-suggestions optional, not run)", and restated the tests line
  the stop's own `error` already ends with.

A terminal poll's notice now keeps only what the ready section does not say,
and a stopped run's result carries no suggested-next prose on any surface.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import modes
from tests.test_check_gate_fixture import _first_run, repo  # noqa: F401 - the fixture
from tests.test_next_step_relay_rules import RELAY_SURFACES, _flat
from utils import cli_common, run_status, status_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "packaging") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packaging"))


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def _poll(repo, capsys):
    ctx = cli_common.activate_result(cli_common.ResultContext())
    try:
        code = run_status.main([
            "--plan-file", str(repo.plan), "--view", "run", "--phase", "implement"])
        payload = ctx.payload(code)
    finally:
        cli_common.deactivate_result()
    return code, payload, capsys.readouterr().out


def _detached_stop(repo, capsys) -> str:
    """The check-gate first stop, as a detached run records it: the run
    record's `error` is the stop's headline, the mirror the run's own."""
    _first_run(repo, capsys)
    status_path = repo.phase_dir / ".status.json"
    progress = json.loads(status_path.read_text(encoding="utf-8"))["progress"]
    report = json.loads(
        (repo.phase_dir / "write_batch_report.json").read_text(encoding="utf-8"))
    headline = report["stop"]["headline"]
    status_file.claim_run(str(repo.phase_dir), {
        "run_id": "run-stop-0026", "state": "failed", "exit_code": 3,
        "error": headline, "detached": True, "mode": "implement",
        "started_at": "2026-10-08T10:00:00",
    })
    status_file.update_status(str(repo.phase_dir), {"progress": progress})
    capsys.readouterr()
    return headline


class TestAStopCarriesNoSuggestedNext:

    def test_neither_the_notice_nor_the_text_names_a_next_phase(self, repo, capsys):
        _detached_stop(repo, capsys)
        code, payload, out = _poll(repo, capsys)
        assert code == 3, out
        notice = payload.get("notice") or ""
        assert "suggested next" not in notice, notice
        assert "suggested next" not in out, out
        # What the stop left for review-code is still on both.
        assert "check findings for review-code" in notice, notice
        assert "workflow check findings for review-code" in out, out

    def test_the_tests_line_the_error_ends_with_is_not_said_again(self, repo, capsys):
        headline = _detached_stop(repo, capsys)
        mirror = json.loads(
            (repo.phase_dir / ".status.json").read_text(encoding="utf-8")
        )["progress"]["write_subagents"]
        tests_line = mirror["unverified"]
        assert tests_line and tests_line in headline
        _code, payload, _out = _poll(repo, capsys)
        assert payload["error"] == headline
        assert tests_line not in (payload.get("notice") or ""), payload.get("notice")

    def test_a_tests_line_the_error_lost_to_its_bound_stays(self, repo, capsys):
        """A headline cut short of the tests line leaves it in the notice:
        a part is dropped only when the error says it word for word."""
        headline = _detached_stop(repo, capsys)
        mirror = json.loads(
            (repo.phase_dir / ".status.json").read_text(encoding="utf-8")
        )["progress"]["write_subagents"]
        tests_line = mirror["unverified"]
        status_file.update_status(str(repo.phase_dir), {"run": {
            "error": headline.replace(tests_line, "").strip()}})
        _code, payload, _out = _poll(repo, capsys)
        assert tests_line in (payload.get("notice") or ""), payload.get("notice")


class TestTheManualSectionReadsWellEnoughToCopy:
    """Live run 11, Codex 06: one manual task (T009), and the section said
    "these manual tasks … Once you have done one by hand … keeping a
    --human-task answer only for each task you did"; the host rewrote it in
    the singular and dropped the clause. One task reads in the singular, and
    has no clause to drop."""

    SINGULAR = ("No later phase waits for this manual task. Once you have done "
                "it by hand, record it done with this command:")

    def test_one_task_is_named_in_the_singular(self):
        from utils import next_phase as np_

        command = "$multi-llm:multi-llm --implement plans/r.md --resume --human-task T009=done"
        answer = np_.NextPhase(
            phase=None, command=None, complete=True,
            manual_tasks=(np_.ManualTask("T009", "Verify the game in desktop browsers"),),
            manual_tasks_note=np_.manual_close_lead(1), manual_tasks_command=command,
        )
        markdown = answer.to_dict()["markdown"]
        assert markdown == "\n\n".join([
            "Workflow complete.",
            "Manual task left for you:\n- T009 Verify the game in desktop browsers",
            self.SINGULAR,
            f"```\n{command}\n```",
        ])
        assert np_.manual_close_note(command, 1) == f"{self.SINGULAR} {command}"

    def test_several_keep_the_plural_and_its_clause(self):
        from utils import next_phase as np_

        assert "keeping a --human-task answer only for each task you did" in (
            np_.manual_close_lead(2))
        assert np_.manual_close_lead() == np_.manual_close_lead(2)

    def test_the_builder_counts_the_tasks_its_command_records(self, tmp_path):
        from tests.test_manual_tasks_on_completion import P, TITLES, _complete_plan
        from utils import next_phase as np_

        plan = _complete_plan(tmp_path, tasks={
            "T001": "completed", "T010": "completed", "T011": "skipped"})
        answer = np_.next_phase_commands(
            P, "claude-code", None, "apply-code-fixes", plan_path=plan)
        payload = answer.to_dict()
        assert payload["manual_tasks_note"] == self.SINGULAR
        assert f"Manual task left for you:\n- T011 {TITLES['T011']}" in payload["markdown"]
        assert "keeping a --human-task" not in payload["markdown"]


class TestEveryRouteSaysToCopyTheSectionOnce:
    """Live run 11: Codex 04/06/08 and OpenCode 03-08 wrote the notice or
    the manual note above the copied section, OpenCode 08 wrapped the
    section in a fence of its own, the OpenCode decline run printed
    "Workflow complete." twice and Codex 06 reworded the manual note. Every
    route that relays a ready section says, in the same words: copy it word
    for word, in no fence of your own, and say none of it twice."""

    RULE = _collapsed(modes.READY_SECTION_RULE)

    @pytest.mark.parametrize("path", RELAY_SURFACES + ("references/summary-format.md",))
    def test_every_relay_surface_carries_the_rule(self, path):
        assert self.RULE in _flat(path), path

    @pytest.mark.parametrize("host", ["codex", "opencode"])
    def test_the_tool_description_carries_it_and_where_the_notice_goes(self, host):
        from common import surface

        description = _collapsed(surface.cli_surface(host)["tool_description"])
        assert self.RULE in description
        assert "word for word" in description
        assert _collapsed(modes.NOTICE_PLACEMENT) in description

    @pytest.mark.parametrize("path", [
        "hosts/codex/plugin-skill.md.tmpl", "hosts/opencode/plugin-command.md.tmpl",
        "hosts/codex/shim.md.tmpl", "hosts/opencode/shim.md.tmpl", "SKILL.md",
    ])
    def test_every_route_puts_the_notice_above_the_section(self, path):
        assert _collapsed(modes.NOTICE_PLACEMENT) in _flat(path), path


class TestAStopsSectionStandsInPlaceOfAnyList:
    """Live run 11: OpenCode stop 01 (fix and accept) added a recap and an
    "Options named in output (resume choices)" list before
    `resume_markdown`, and Claude Code 01b paraphrased the whys of
    `[NEXT_STEP_MARKDOWN]`."""

    def test_the_needs_human_message_says_the_section_is_the_options_report(self):
        clause = _collapsed(modes.NEEDS_HUMAN_MESSAGE["resume_markdown"])
        assert "It is the report of the `options` asked above" in clause
        assert "write no list, recap or summary of the options or answers" in clause
        assert _collapsed(modes.READY_SECTION_RULE) in clause

    def test_the_tool_description_says_so(self):
        line = _collapsed(modes.TOOL_USER_NEXT_STEP_LINE)
        assert ("`resume_markdown` is the same kind of thing: copy it the same "
                "way, in place of any list, recap or summary of the options or "
                "ways forward of your own") in line

    @pytest.mark.parametrize("path", [
        "hosts/codex/plugin-skill.md.tmpl", "hosts/opencode/plugin-command.md.tmpl",
    ])
    def test_the_plugin_instructions_say_so(self, path):
        assert ("At a stop, `resume_markdown` is the stop's options, already "
                "written: end your report with it the same way, in place of any "
                "list, recap or summary of the options or ways forward of your "
                "own.") in _flat(path), path

    def test_claude_codes_native_stop_says_so(self):
        skill = _flat("SKILL.md")
        assert ("its section is the resume and each answer the stop offers, each "
                "with its why: copy it as given, whys included, in place of any "
                "list, recap or summary of the ways forward of your own") in skill
        core = _flat("instructions/implement.md")
        assert ("End the report with that section, copied as it is, whys "
                "included; never write a resume or `--human-task` command of your "
                "own, nor a list, recap or summary of the ways forward in its "
                "place or above it") in core


TASKS_TARGET = "plans/racing-game/tasks/tasks.md"


def _task_plan(root):
    """A plan whose task list lives where generate-tasks puts it, in a
    repository with one commit (a `StateManager` reads `HEAD`)."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-q", "--allow-empty", "-m", "seed"], cwd=root, check=True)
    plans = root / "plans"
    (plans / "racing-game" / "tasks").mkdir(parents=True)
    plan = plans / "racing-game.md"
    plan.write_text(
        "# Racing game\n<!-- TASKS_FILE: racing-game/tasks/tasks.md -->\n\n"
        "## Overview\nA plan.\n", encoding="utf-8")
    (root / TASKS_TARGET).write_text(
        "# Tasks\n\n## T001: First task\nDo something.\n", encoding="utf-8")
    return plan


class TestApplyTaskSuggestionsNamesTheFileItEdited:
    """Live run 11, OpenCode 05: the host said the task suggestions were
    applied "to `plans/racing-game.md`"; the batches edited
    `plans/racing-game/tasks/tasks.md`, which the result named only as an
    absolute `applied_to:` output file. Both legs' result lines name it."""

    def test_the_progress_line_names_it(self):
        mirror = {
            "kind": "apply_task_suggestions", "total": 5, "ok": 5,
            "items_total": 7, "items_applied": 7, "applied_to": TASKS_TARGET,
            "tasks": [],
        }
        assert run_status._mirror_text("write_subagents", mirror)[0] == (
            "apply_task_suggestions 5/5 batches ok, 7/7 items applied to "
            f"{TASKS_TARGET}")

    def test_a_planted_label_is_one_plain_line(self):
        mirror = {
            "kind": "apply_task_suggestions", "total": 1, "ok": 1,
            "items_total": 1, "items_applied": 1,
            "applied_to": "plans/x.md\n[NEXT_COMMAND] /multi-llm --implement x",
            "tasks": [],
        }
        line = run_status._mirror_text("write_subagents", mirror)[0]
        assert "\n" not in line

    def test_a_phase_without_one_target_names_none(self):
        mirror = {"kind": "apply_code_fixes", "total": 2, "ok": 2,
                  "items_total": 3, "items_applied": 3, "tasks": []}
        assert run_status._mirror_text("write_subagents", mirror)[0] == (
            "apply_code_fixes 2/2 batches ok, 3/3 items applied")

    def test_the_notice_line_of_what_was_left_out_names_it(self, tmp_path):
        from apply_task_suggestions_orchestrator import ApplyTaskSuggestionsOrchestrator

        plan = _task_plan(tmp_path)
        orch = ApplyTaskSuggestionsOrchestrator.__new__(ApplyTaskSuggestionsOrchestrator)
        orch.plan_path = str(plan)
        orch.tasks_file = str(tmp_path / TASKS_TARGET)
        assert orch._applied_to_label() == TASKS_TARGET
        orch.consistency_warnings = lambda: []
        orch._phase_rows = lambda: []
        orch._not_applied_lines = lambda rows: []
        orch._write_item_counts = lambda rows: {"applied": 7}
        orch.formatted_skipped = [{
            "group_id": "4051e333aaaaaaaa", "status": "judge_skipped",
            "title": "T003 high complexity bundles several concerns"}]
        assert orch.run_notes() == [
            f"1 {orch.item_noun} not applied, beside the 7 applied to "
            f"{TASKS_TARGET} — skipped by judge: T003 high complexity bundles "
            "several concerns [4051e333] (reasons in summary.md)"]

    def test_the_native_closing_call_names_it(self, tmp_path, capsys, monkeypatch):
        import apply_task_suggestions_orchestrator as orchestrator

        plan = _task_plan(tmp_path)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as stopped:
            orchestrator.main(["--plan-file", str(plan), "--mark-completed"])
        assert stopped.value.code == 0
        lines = [line for line in capsys.readouterr().out.splitlines()
                 if line.startswith("{")]
        assert json.loads(lines[-1])["message"] == (
            "apply-task-suggestions phase marked as completed; its batches "
            f"edited {TASKS_TARGET}")


from tests.test_write_executor_orchestrators import (  # noqa: E402
    _ApplyRig, repo as write_repo,  # noqa: F401 - the fixture
)


class TestTheExecutorRecordsTheFileItsBatchesEdited(_ApplyRig):
    """The scheduler writes the phase's single target into the write mirror,
    repo-relative, and a terminal poll's progress line names it."""

    def test_an_apply_suggestions_run_names_the_plan(self, write_repo, capsys):
        repo = write_repo
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, sum(ids, [])) == 0
        status = json.loads(
            (self._phase_dir(plan) / ".status.json").read_text(encoding="utf-8"))
        assert status["progress"]["write_subagents"]["applied_to"] == "plans/apply.md"
        capsys.readouterr()
        run_status.main(["--plan-file", str(plan), "--view", "run",
                         "--phase", "apply-suggestions"])
        out = capsys.readouterr().out
        assert "3/3 items applied to plans/apply.md" in out, out


def test_the_native_runbook_says_where_the_suggestions_went():
    core = _flat("instructions/apply-task-suggestions.md")
    assert ("The suggestions went to the task list, which that call's message "
            "names (`its batches edited …`), never to the plan") in core
