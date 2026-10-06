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
* ``status --view run`` listed no task of an implement the Claude Code host
  ran itself (the native leg writes no report and no write mirror);
* a deferred manual task never said how to close it, and the one way —
  ``--human-task <id>=done`` on implement's resume — was ignored under the
  ``--yes`` strategy and, closing nothing, re-stamped implement so the
  review read stale;
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


class TestStatusReadsANativeImplementFromState:
    """The native leg (Claude Code): the host records each task with
    `util batch-done`, and `status --view run` showed none of them."""

    def _native_stop(self, repo, capsys):
        from tests.test_check_gate_fixture import FINDINGS, TestTheNativeLeg

        leg = TestTheNativeLeg()
        leg._emit(repo, capsys)
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            assert leg._util(capsys, "batch-done", "--phase-dir", phase,
                             "--task-id", task_id, "--status", "completed")[0] == 0
        leg._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                  "--status", "completed", "--check", "findings",
                  "--finding", FINDINGS[0])
        assert leg._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                         "T005", "--status", "failed", "--reason",
                         "refused before dispatch: the write policy")[0] == 0
        return leg

    def test_the_run_view_lists_every_task_and_why(self, repo, capsys):
        self._native_stop(repo, capsys)
        assert not (repo.phase_dir / "write_batch_report.json").exists()
        out = _status_run(repo, capsys)
        assert "implement  no run record — the host ran these tasks itself" in out, out
        line = next(l for l in out.splitlines() if l.strip().startswith("tasks:"))
        assert "4/7 done" in line
        assert "completed: T001, T002, T003, T004" in line
        assert "failed: T005" in line
        assert "not run: T006, T007" in line
        assert "T005: refused before dispatch: the write policy" in out
        assert "check findings T002: retries must be 3 (found 2)" in out

    def test_answers_and_skips_read_as_the_user_gave_them(self, repo, capsys):
        leg = self._native_stop(repo, capsys)
        leg._emit(repo, capsys, "--resume", "--human-task", "T002=done",
                  "--human-task", "T005=skip")
        out = _status_run(repo, capsys)
        line = next(l for l in out.splitlines() if l.strip().startswith("tasks:"))
        assert "skipped at your request: T005" in line, line
        assert "skipped (depend on a task you skipped): T006" in line, line
        assert "deferred manual" not in line
        assert "accepted with their check findings or after failing: T002" in line
        assert ("answers on record: --human-task T002=done, "
                "--human-task T005=skip") in out

    def test_an_executor_run_keeps_its_own_view(self, repo, capsys):
        _first_run(repo, capsys)
        out = _status_run(repo, capsys)
        assert "no run record — the host ran these tasks itself" not in out


class TestClosingADeferredManualTask:
    """Codex racing 06/08: T007 was deferred, the workflow completed, and
    nothing said how to mark it done. `--human-task T007=done` on implement's
    resume is that way; it now closes the task without reopening the
    workflow."""

    @pytest.fixture
    def done_repo(self, tmp_path, monkeypatch):
        from tests.test_implement_stop_account import _check_passes, make_repo
        from utils import provider_registry as registry

        obj = make_repo(tmp_path, monkeypatch)
        _check_passes(obj)
        yield obj
        registry._config = None
        registry._config_key = None

    def _finish_with_review(self, repo, capsys):
        from utils.state_manager import get_or_create_state

        assert _implement(repo) == 0
        capsys.readouterr()
        state = get_or_create_state(repo.plan)
        assert state.get_task_status("T005") == "skipped"  # deferred manual
        state.mark_phase_completed("code-review")
        state.save()
        return get_or_create_state(repo.plan).state["phases_completed"]["implement"]

    def test_the_executor_leg_closes_it_and_keeps_the_review_fresh(
        self, done_repo, capsys,
    ):
        import multi_llm
        from utils.state_manager import get_or_create_state

        stamp = self._finish_with_review(done_repo, capsys)
        done_repo.log.unlink(missing_ok=True)
        assert multi_llm.main([
            "implement", str(done_repo.plan), "--yes", "--exec-subagents",
            "--resume", "--human-task", "T005=done",
        ]) == 0
        capsys.readouterr()
        assert _dispatched_titles(done_repo) == []
        state = get_or_create_state(done_repo.plan)
        assert state.get_task_status("T005") == "completed"
        assert state.state["phases_completed"]["implement"] == stamp
        assert state.is_phase_completed("code-review")
        assert state.deferred_implementation_tasks() == []

    def test_the_native_leg_closes_it_without_reopening_implement(
        self, done_repo, capsys, tmp_path,
    ):
        import implement_orchestrator
        from utils.state_manager import get_or_create_state

        stamp = self._finish_with_review(done_repo, capsys)
        assert implement_orchestrator.main([
            "--plan-file", str(done_repo.plan), "--output", str(tmp_path / "h.json"),
            "--yes", "--resume", "--human-task", "T005=done",
        ]) == 0
        assert "Nothing is left to run: implement stays completed" in capsys.readouterr().out
        state = get_or_create_state(done_repo.plan)
        assert state.get_task_status("T005") == "completed"
        assert state.is_phase_completed("implement")
        assert state.is_phase_completed("code-review")
        # A host that closes the phase anyway keeps the date.
        assert multi_llm_util(["finalize-tracking", "--state-file",
                               str(state.state_file), "--mark-completed"]) == 0
        assert get_or_create_state(done_repo.plan).state[
            "phases_completed"]["implement"] == stamp


def multi_llm_util(argv) -> int:
    import multi_llm

    return multi_llm.main(["util", *argv])
