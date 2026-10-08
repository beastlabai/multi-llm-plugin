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

from tests.test_check_gate_fixture import _first_run, repo  # noqa: F401 - the fixture
from utils import cli_common, run_status, status_file


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
