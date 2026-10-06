"""What a stopped implement hands the user, on every surface (fix round 14).

Live run 5 (2026-10-06), the stop-path fixture on the Codex and OpenCode tool
routes and on Claude Code's native leg. Each class pins one finding:

* ``status --view run`` listed the tasks the user skipped with
  ``--human-task T005=skip`` (and the one skipped with it) as "deferred
  manual", beside a summary saying "Skipped at your request";
* after ``--resume --human-task …`` the next stop's ``resume_command`` was a
  plain ``--resume`` while its first option replayed the answers: the
  answers are on record, and a plain ``--resume`` applies them;
* on a first stop ``message`` and ``options`` offered only the bare
  ``--resume``, the answers sitting in ``error``: the stop now records its
  answers as data and ``util next-command --blocked`` hands each back as a
  command in the host's spelling (the servers make them options);
* an apply phase's ``[CONSISTENCY_WARNING]`` and the item its judge skipped
  lived only in the log, and the host reported "Applied 10 of 10";
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_check_gate_fixture import (  # noqa: E402
    NOTE, _check, _first_run, _implement, _policy, repo,  # noqa: F401 - the fixture
)
from tests.test_implement_stop_account import _dispatched_titles, _statuses  # noqa: E402
from utils import run_status  # noqa: E402


def _status_run(repo, capsys) -> str:
    capsys.readouterr()
    run_status.main(["--plan-file", str(repo.plan), "--view", "run"])
    return capsys.readouterr().out


class TestASkipYouChoseIsNotDeferredManualWork:

    def test_status_names_the_skip_and_its_dependent_apart(self, repo, capsys):
        _first_run(repo, capsys)
        # T002's findings still hold T007, so the resume stops again.
        assert _implement(repo, "--resume", "--human-task", "T005=skip") == 3
        mirror = json.loads(
            (repo.phase_dir / ".status.json").read_text()
        )["progress"]["write_subagents"]
        assert mirror["declined_task_ids"] == ["T005"]
        assert mirror["declined_dependent_task_ids"] == ["T006"]
        out = _status_run(repo, capsys)
        assert "skipped at your request: T005" in out, out
        assert "skipped (depend on a task you skipped): T006" in out, out
        assert "deferred manual" not in out, out


class TestAPlainResumeKeepsTheAnswersOnRecord:

    def test_a_skip_answered_once_holds_through_later_resumes(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(repo, "--resume", "--human-task", "T005=skip") == 3
        capsys.readouterr()
        repo.log.unlink(missing_ok=True)
        # Nothing changed and no answer is repeated: T005 is not refused again.
        assert _implement(repo, "--resume") == 3
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert _dispatched_titles(repo) == []
        assert "T005 refused before dispatch" not in text
        assert _statuses(repo)["T005"] == "skipped"
        # Accepting T002 alone runs T007; T005's earlier answer still holds.
        assert _implement(repo, "--resume", "--human-task", "T002=done") == 0
        capsys.readouterr()
        assert _dispatched_titles(repo) == [NOTE]
        statuses = _statuses(repo)
        assert statuses["T005"] == "skipped" and statuses["T006"] == "skipped"


#: The answers the check-gate first stop offers, in the order a user weighs
#: them: the combined go-on-without-fixing command, then T005's (refused before
#: dispatch), then T002's (its findings hold T007).
FIRST_STOP_FLAGS = [
    ["--resume", "--human-task", "T002=done", "--human-task", "T005=skip"],
    ["--resume", "--human-task", "T005=skip"],
    ["--resume", "--human-task", "T005=done"],
    ["--resume", "--human-task", "T002=done"],
    ["--resume", "--human-task", "T002=skip"],
    ["--resume", "--human-task", "T002=retry"],
]


class TestTheStopRecordsItsAnswers:

    def test_the_first_stop_records_every_answer_it_names(self, repo, capsys):
        text = _first_run(repo, capsys)
        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        answers = report["stop"]["answers"]
        assert [a["flags"] for a in answers] == FIRST_STOP_FLAGS
        assert answers[0]["why"].startswith(
            "accept T002's check findings and leave T005 undone, with the "
            "tasks that depend on it (T006)")
        assert "T007 then run" in answers[3]["why"]
        mirror = json.loads(
            (repo.phase_dir / ".status.json").read_text()
        )["progress"]["write_subagents"]["stop"]
        assert mirror["answers"] == answers
        # Every answer recorded is one the stop's own text names.
        for flags in FIRST_STOP_FLAGS:
            assert " ".join(flags[1:]) in text, flags

    def test_util_next_command_spells_each_one_for_the_host(
        self, repo, capsys, monkeypatch,
    ):
        import modes
        from utils import cli_common, next_phase

        _first_run(repo, capsys)
        record = {}
        monkeypatch.setattr(cli_common, "record_user_resume",
                            lambda obj: record.setdefault("v", obj))
        monkeypatch.setenv(modes.ROUTE_ENV, "package")
        plan = "plans/checked.md"
        assert next_phase.main([plan, "--blocked", "implement", "--host", "codex"]) == 0
        out = capsys.readouterr().out
        resume = record["v"]
        prefix = f"$multi-llm:multi-llm --implement {plan}"
        assert out == f"[RESUME_COMMAND] {prefix} --resume\n"
        assert resume["command"] == f"{prefix} --resume"
        assert resume["flags"] == ["--resume"]
        assert [a["flags"] for a in resume["answers"]] == FIRST_STOP_FLAGS
        assert [a["command"] for a in resume["answers"]] == [
            f"{prefix} {' '.join(flags)}" for flags in FIRST_STOP_FLAGS]
        assert all(a["why"] for a in resume["answers"])
        text = resume["answers_text"]
        for answer in resume["answers"]:
            assert f"`{answer['command']}` ({answer['why']})" in text


WARNING = ("T003 depends on T004, listed after it: a task added by an edit "
           "batch belongs where its dependents can follow it")


class TestAnApplyPhasesNotesReachTheResult:

    def _orchestrator(self, tmp_path):
        from tests.test_apply_orchestrator_base import (
            _FakeOutcome, _FakeSchedulerResult, _orchestrator_on,
        )

        orch = _orchestrator_on(tmp_path, yes=True)
        orch.write_tasks = [{"task_id": "apply-batch-0", "expected_ids": ["a", "b"]}]
        orch.write_result = _FakeSchedulerResult([_FakeOutcome("apply-batch-0")])
        orch.formatted_skipped = [
            {"group_id": "934916bf899472b7", "status": "judge_skipped",
             "title": "T002 combines scene, track generation, validation, and helpers",
             "reason": "a scope trade-off"},
        ]
        orch.consistency_warnings = lambda: [WARNING]
        return orch

    def test_the_notes_name_the_warning_and_the_skipped_item(self, tmp_path):
        orch = self._orchestrator(tmp_path)
        notes = orch.run_notes()
        assert notes[0] == f"consistency warning: {WARNING}"
        assert notes[1].startswith(
            f"1 {orch.item_noun} not applied, beside the 2 applied — skipped by "
            "judge: T002 combines scene, track generation, validation, and "
            "helpers [934916bf]")

    def test_they_reach_the_notice_and_the_status_file(self, tmp_path):
        from utils import cli_common

        orch = self._orchestrator(tmp_path)
        (Path(orch.out_dir) / orch.phase_name).mkdir(parents=True, exist_ok=True)
        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            cli_common.record_notice("an earlier notice")
            orch._publish_run_notes()
            notice = context.notice
        finally:
            cli_common.deactivate_result()
        assert notice.startswith("an earlier notice; consistency warning: ")
        assert "1 " + orch.item_noun + " not applied" in notice
        status = json.loads(
            (Path(orch.out_dir) / orch.phase_name / ".status.json").read_text())
        assert status["progress"]["notes"] == orch.run_notes()

    def test_a_terminal_poll_carries_them(self, tmp_path, capsys):
        import multi_llm
        from utils import status_file

        plan = tmp_path / "plans" / "p.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# p\n", encoding="utf-8")
        phase_dir = tmp_path / "plans" / "p" / "apply-task-suggestions"
        phase_dir.mkdir(parents=True)
        notes = [f"consistency warning: {WARNING}",
                 "1 task suggestion not applied, beside the 10 applied — "
                 "skipped by judge: T002 [934916bf] (reasons in summary.md)"]
        status_file.write_status(str(phase_dir), {
            "phase": "apply-task-suggestions", "state": "complete",
            "progress": {"step": "complete", "notes": notes},
            "run": {"run_id": "R1", "state": "done", "exit_code": 0,
                    "mode": "apply-task-suggestions"},
        })
        capsys.readouterr()
        multi_llm.main(["status", str(plan), "--view", "run", "--run-id", "R1"])
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        result = json.loads(line.split(" ", 1)[1])
        assert result["notice"].startswith("; ".join(notes))
        assert [f"note: {note}" for note in notes] == result["progress_lines"][-2:]
        assert f"      note: {notes[1]}" in out

    def test_the_executor_leg_publishes_them_when_it_finishes(self, tmp_path, capsys):
        """The wiring: `_write_outputs` publishes the notes after the RESULT
        block, on the leg that dispatched."""
        from tests.test_apply_orchestrator_base import (
            MockApplyOrchestrator, _FakeOutcome, _FakeSchedulerResult, _make_args,
            _setup_plan_with_output,
        )
        from utils import cli_common

        plan_path = _setup_plan_with_output(tmp_path)

        class WarnedOrch(MockApplyOrchestrator):
            phase_name = "apply-noted-leg"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def run_write_executor(self):
                self.write_tasks = [{"task_id": "apply-batch-0", "expected_ids": ["a"]}]
                self.write_result = _FakeSchedulerResult([_FakeOutcome("apply-batch-0")])
                return 0

            def consistency_warnings(self):
                return [WARNING]

        orch = WarnedOrch(_make_args(plan_file=str(plan_path), yes=True))
        context = cli_common.activate_result(cli_common.ResultContext(
            mode="apply-noted-leg", phase="apply-noted-leg"))
        try:
            assert orch.run() == 0
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert f"consistency warning: {WARNING}" in (context.notice or "")
        status = json.loads(
            (Path(orch.out_dir) / orch.phase_name / ".status.json").read_text())
        assert status["progress"]["notes"][0] == f"consistency warning: {WARNING}"
