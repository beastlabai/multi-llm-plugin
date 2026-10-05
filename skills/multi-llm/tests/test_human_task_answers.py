"""``--human-task`` answers to a failed task, and how the stop accounts for them.

Fix round 8 (review of round 7, 2026-10-05). Each class pins one finding:

* an answer to a failed code task on the **native** leg (Claude Code, no
  ``--exec-subagents``) was recorded and printed but never applied — the plan
  was only resolved for a plan holding a manual task, and even then the
  accepted task was handed to the host again as ``failed``;
* ``--human-task <id>=skip`` under a persisted ``skip-continue`` ran the
  dependents the stop message said stay undone;
* a ``--resume`` dropped every skipped code task, so the resumed account's
  total was short and "that is every one of the N task(s)" was false;
* a manual task answered ``done`` counted in the total and sat in no bucket;
* review-code's implementation record read an accepted failure as a pass;
* ``status --exit-zero`` dropped the stop's ``error`` from the result line;
* the native emit filed a skip the user chose under "Manual Tasks (Deferred)",
  which ``[WORKFLOW_COMPLETE]`` leaves out.

The fixture is ``test_implement_stop_account.make_repo`` — the five-task plan
with a check in the middle and a manual task at the end — plus two variants:
one with a code task after the manual one, one with no manual task at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import status_file  # noqa: E402
from utils import write_batch_scheduler as scheduler  # noqa: E402
from utils.state_manager import StateManager, get_or_create_state  # noqa: E402
from utils.task_decomposer import Task, TaskStatus  # noqa: E402

from tests.test_implement_stop_account import (  # noqa: E402
    PLAN, _check_fails, _check_passes, _dispatched_titles, _implement,
    _run_one_that_stops, _statuses, make_repo,
)

PLAN_WITH_CODE_DEPENDENT = PLAN + """
### T006: Edit: after the manual step
Needs the manual play-through first.
- Depends on: T005
- Files to modify: src/main.py
"""

PLAN_NO_HUMAN = PLAN.split("### T005", 1)[0]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    yield make_repo(tmp_path, monkeypatch)
    registry._config = None
    registry._config_key = None


@pytest.fixture
def repo_dep(tmp_path, monkeypatch):
    obj = make_repo(tmp_path, monkeypatch, PLAN_WITH_CODE_DEPENDENT)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        ["T001", "T002", "T003", "T004", "T006"]
    ))
    yield obj
    registry._config = None
    registry._config_key = None


@pytest.fixture
def repo_nohuman(tmp_path, monkeypatch):
    yield make_repo(tmp_path, monkeypatch, PLAN_NO_HUMAN)
    registry._config = None
    registry._config_key = None


def _summary(repo) -> str:
    report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
    return Path(report["summary_file"]).read_text()


def _report(repo) -> dict:
    return json.loads((repo.phase_dir / "write_batch_report.json").read_text())


def _account_line(repo) -> str:
    return next(
        line for line in _summary(repo).splitlines()
        if line.startswith("**Tasks in this decomposition**")
    )


def _section(text: str, heading: str) -> str:
    """The body of one ``## heading`` of a summary, up to the next ``## ``."""
    if heading not in text:
        return ""
    body = text.split(heading, 1)[1]
    return body.split("\n## ", 1)[0]


# --- the native leg ----------------------------------------------------------

def _emit(repo, out_json: Path, *flags: str) -> int:
    """``implement`` on the native leg: no ``--exec-subagents``, a handoff file."""
    return implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--output", str(out_json), *flags,
    ])


def _handed(out_json: Path) -> list:
    data = json.loads(out_json.read_text())
    return [(t["id"], t.get("status")) for b in data["batches"] for t in b["tasks"]]


def _fail_the_check(repo) -> None:
    """What a host's ``util batch-done`` leaves after T001 applied and T002 failed."""
    state = get_or_create_state(repo.plan)
    state.update_task_status("T001", "completed")
    state.update_task_status("T002", "failed", reason="the check failed")
    state.save()


class TestAnAnswerOnTheNativeLeg:

    def test_done_is_applied_with_no_manual_task_in_the_plan(
        self, repo_nohuman, tmp_path, capsys
    ):
        """No manual task, so nothing used to resolve the plan: the accept was
        recorded, ``[HUMAN_TASK_ANSWERED]`` printed, and T002 was handed to the
        host again as ``failed``."""
        repo = repo_nohuman
        out_json = tmp_path / "implement_tasks.json"
        assert _emit(repo, out_json, "--yes") == 0
        _fail_the_check(repo)
        capsys.readouterr()
        assert _emit(repo, out_json, "--yes", "--resume", "--human-task", "T002=done") == 0
        out = capsys.readouterr().out
        assert "[HUMAN_TASK_ANSWERED] T002=done" in out
        assert _statuses(repo)["T002"] == "completed"
        reason = get_or_create_state(repo.plan).get_task_status_reason("T002")
        assert reason.startswith("accepted — you answered --human-task T002=done")
        handed = _handed(out_json)
        assert "T002" not in [tid for tid, _ in handed]
        assert ("T004", "pending") in handed, "the accepted task's dependent runs"
        data = json.loads(out_json.read_text())
        assert data["accepted_tasks"] == {"T002": reason}
        assert data["pending_tasks"] == len(handed)
        assert "Accepted after failing (recorded completed in state.json" in out
        assert "T002 Check the module: accepted —" in out
        # A later resume keeps it out: the answer applied once, and the record
        # it left is what `--resume` reads.
        capsys.readouterr()
        assert _emit(repo, out_json, "--yes", "--resume") == 0
        assert "T002" not in [tid for tid, _ in _handed(out_json)]
        assert _statuses(repo)["T002"] == "completed"

    def test_done_with_a_manual_task_present_takes_the_task_out_of_the_batches(
        self, repo, tmp_path, capsys
    ):
        """With a manual task the plan WAS resolved, so state said T002
        completed ("accepted …") while the task JSON still handed it out as
        ``failed`` — and the host's ``batch-done`` then overwrote the accept."""
        out_json = tmp_path / "implement_tasks.json"
        assert _emit(repo, out_json, "--yes") == 0
        _fail_the_check(repo)
        capsys.readouterr()
        assert _emit(repo, out_json, "--yes", "--resume", "--human-task", "T002=done") == 0
        handed = _handed(out_json)
        assert "T002" not in [tid for tid, _ in handed], handed
        assert ("T005", "skipped") in handed
        assert _statuses(repo)["T002"] == "completed"
        data = json.loads(out_json.read_text())
        assert list(data["accepted_tasks"]) == ["T002"]
        assert list(data["skipped_tasks"]) == ["T005"]
        # `write_tasks.json`, which `util write-prompt` reads, has no prompt
        # for a task the host must not dispatch.
        write_tasks = json.loads((repo.phase_dir / "write_tasks.json").read_text())
        assert "T002" not in [t["task_id"] for t in write_tasks["batches"]]

    def test_skip_is_applied_and_takes_the_dependents(
        self, repo_nohuman, tmp_path, capsys
    ):
        repo = repo_nohuman
        out_json = tmp_path / "implement_tasks.json"
        assert _emit(repo, out_json, "--yes") == 0
        _fail_the_check(repo)
        capsys.readouterr()
        assert _emit(repo, out_json, "--yes", "--resume", "--human-task", "T002=skip") == 0
        out = capsys.readouterr().out
        handed = dict(_handed(out_json))
        assert handed["T002"] == "skipped" and handed["T004"] == "skipped"
        assert handed["T003"] == "pending"
        data = json.loads(out_json.read_text())
        assert sorted(data["skipped_tasks"]) == ["T002", "T004"]
        assert data["skipped_tasks"]["T002"].startswith(
            "skipped — you answered --human-task T002=skip"
        )
        statuses = _statuses(repo)
        assert statuses["T002"] == "skipped" and statuses["T004"] == "skipped"
        # Both are the user's decision, not deferred work.
        assert "Skipped at your request" in out
        assert "T002 Check the module: skipped — you answered" in out
        assert "T004 Edit: wire the fallback: skipped — depends on skipped task T002" in out
        assert "Manual Tasks (Deferred) section" not in out.split("Skipped at your request")[0]
        assert get_or_create_state(repo.plan).deferred_implementation_tasks() == []
        # A second resume keeps both skipped rather than handing them out pending.
        capsys.readouterr()
        assert _emit(repo, out_json, "--yes", "--resume") == 0
        handed = dict(_handed(out_json))
        assert handed.get("T002") == "skipped" and handed.get("T004") == "skipped"
        statuses = _statuses(repo)
        assert statuses["T002"] == "skipped" and statuses["T004"] == "skipped"

    def test_a_declined_manual_task_is_named_apart_from_the_deferred_ones(
        self, repo, tmp_path, capsys
    ):
        """A pause-and-ask "Skip" is not owed: the emit header must not tell
        the host to list it under Manual Tasks (Deferred), which is where
        ``[WORKFLOW_COMPLETE]``'s ``[MANUAL_TASK]`` lines would disagree."""
        out_json = tmp_path / "implement_tasks.json"
        assert _emit(
            repo, out_json, "--no-confirm", "--force-strategy", "pause-and-ask",
            "--human-task", "T005=skip",
        ) == 0
        out = capsys.readouterr().out
        assert "Skipped at your request" in out
        assert "T005 Play through by hand: skipped by user during pause-and-ask" in out
        assert "Deferred tasks (" not in out
        assert get_or_create_state(repo.plan).deferred_implementation_tasks() == []


# --- skip takes the dependents under every strategy --------------------------

class TestSkipTakesTheDependentsUnderEveryStrategy:

    def test_skip_continue_still_skips_the_dependents_of_a_skipped_code_task(
        self, repo_dep, capsys
    ):
        """The strategy governs the manual tasks: under ``skip-continue`` a
        manual task's edges are treated as satisfied (T006 runs with an
        override record), while a failed code task answered ``skip`` takes its
        dependents with it, as the stop said and the instructions say."""
        repo = repo_dep
        _check_fails(repo)
        assert _implement(repo, "--force-strategy", "skip-continue") == 3
        text = capsys.readouterr()
        text = text.out + text.err
        skip_line = next(l for l in text.splitlines() if "--human-task T002=skip" in l)
        assert "T004" in skip_line and "stay undone" in skip_line
        # The deferred manual task is not something the skip decides.
        assert "T005" not in skip_line, skip_line
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume", "--human-task", "T002=skip") == 0
        assert _dispatched_titles(repo) == ["Edit: add tests", "Edit: after the manual step"]
        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "skipped", "T003": "completed",
            "T004": "skipped", "T005": "skipped", "T006": "completed",
        }
        assert "depends on skipped task T002" in state.get_task_status_reason("T004")
        overrides = state.get("dependency_overrides") or {}
        assert "T006" in overrides and "T004" not in overrides
        summary = _summary(repo)
        assert "| T006 | T005 |" in _section(summary, "## Dependency Override Warnings")

    def test_the_walk_is_pure_and_strategy_independent(self):
        tasks = [
            {"task_id": "T001", "task_title": "a", "executor_role": "agent", "depends_on": []},
            {"task_id": "T002", "task_title": "check", "executor_role": "agent", "depends_on": ["T001"]},
            {"task_id": "T004", "task_title": "wire", "executor_role": "agent", "depends_on": ["T002"]},
            {"task_id": "T005", "task_title": "manual", "executor_role": "human", "depends_on": ["T004"]},
            {"task_id": "T006", "task_title": "after", "executor_role": "agent", "depends_on": ["T005"]},
        ]
        for strategy in scheduler.STRATEGIES:
            decisions = {"T002": "skip"}
            if strategy == scheduler.STRATEGY_PAUSE:
                decisions["T005"] = "done"
            plan = scheduler.resolve_human_task_plan(
                tasks, strategy=strategy, decisions=decisions,
            )
            assert "T004" in plan.skipped, strategy
            assert plan.chain["T004"] == "T002"
            assert "T004" not in plan.overrides
        plan = scheduler.resolve_human_task_plan(
            tasks, strategy=scheduler.STRATEGY_SKIP_CONTINUE, decisions={"T002": "skip"},
        )
        assert plan.overrides == {"T006": ["T005"]}
        assert sorted(plan.skipped) == ["T002", "T004", "T005"]
        assert sorted(plan.declined()) == ["T002", "T004"]
        assert list(plan.deferred()) == ["T005"]

    def test_the_native_leg_skips_them_too(self, repo_nohuman, tmp_path, capsys):
        repo = repo_nohuman
        out_json = tmp_path / "implement_tasks.json"
        assert _emit(repo, out_json, "--yes", "--force-strategy", "skip-continue") == 0
        _fail_the_check(repo)
        capsys.readouterr()
        assert _emit(repo, out_json, "--yes", "--resume", "--human-task", "T002=skip") == 0
        assert dict(_handed(out_json))["T004"] == "skipped"


# --- the resumed account keeps every skipped task ----------------------------

class TestAResumeKeepsEverySkippedTask:

    def test_a_skipped_code_dependent_stays_in_the_resumed_account(
        self, repo_dep, capsys
    ):
        """T006 depends on the deferred manual task, so ``skip-dependents``
        skips it; a resume dropped it from the materialized list, and the
        resumed summary read ``5 — …`` for a six-task plan."""
        repo = repo_dep
        _check_fails(repo)
        assert _implement(repo) == 3
        first = _account_line(repo)
        assert "6 —" in first and "T006" in first
        main = repo.root / "src" / "main.py"
        main.write_text(main.read_text() + "# fixed\n", encoding="utf-8")
        _check_passes(repo)
        capsys.readouterr()
        assert _implement(repo, "--resume") == 0
        second = _account_line(repo)
        assert "6 —" in second, second
        assert "1 skipped with a skipped task (T006)" in second, second
        assert "1 deferred manual (T005)" in second
        assert sorted(_report(repo)["skipped_tasks"]) == ["T005", "T006"]
        assert "| T006 |" in _section(_summary(repo), "## Manual Tasks (Deferred)")

    def test_a_resume_after_skip_keeps_the_skip_and_dispatches_nothing(
        self, repo, capsys
    ):
        _run_one_that_stops(repo, capsys)
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume", "--human-task", "T002=skip") == 0
        repo.log.unlink(missing_ok=True)
        capsys.readouterr()
        assert _implement(repo, "--resume") == 0
        assert _dispatched_titles(repo) == []
        assert _statuses(repo) == {
            "T001": "completed", "T002": "skipped", "T003": "completed",
            "T004": "skipped", "T005": "skipped",
        }
        line = _account_line(repo)
        assert "5 —" in line
        assert "1 skipped at your request (T002)" in line
        assert "1 skipped with a skipped task (T004)" in line
        assert "1 deferred manual (T005)" in line

    def test_format_batches_for_output_carries_a_skipped_code_task(self):
        done = Task(id="T001", title="Task 1", description="d",
                    status=TaskStatus.COMPLETED)
        skipped = Task(id="T006", title="After the manual step", description="d",
                       status=TaskStatus.SKIPPED)
        result = implement_orchestrator.format_batches_for_output(
            [[done, skipped]], skip_completed=True,
        )
        assert [t["id"] for t in result[0]["tasks"]] == ["T006"]

    def test_a_recorded_skip_still_applies_once_the_task_is_skipped(self, tmp_path):
        """``_applicable_task_decisions`` keeps a ``skip`` for a task whose
        state says ``skipped`` (it is materialized again now) and drops a
        ``done`` once the task is no longer ``failed``."""
        plan = tmp_path / "plans" / "p.md"
        plan.parent.mkdir()
        plan.write_text("# p\n", encoding="utf-8")
        state = StateManager(plan)
        state.set("human_task_decisions", {"T002": "skip", "T003": "done", "T005": "done"})
        state.update_task_status("T002", "skipped", reason="skipped — you answered")
        state.update_task_status("T003", "completed", reason="accepted —")
        applicable = implement_orchestrator._applicable_task_decisions(
            state, ["T005"], {},
        )
        assert applicable == {"T002": "skip", "T005": "done"}


# --- every answer lands in a bucket ------------------------------------------

class TestAManualTaskAnsweredDoneIsCounted:

    def test_the_account_has_a_completed_manual_bucket(self, repo, capsys):
        _check_fails(repo)
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--no-confirm", "--exec-subagents",
            "--force-strategy", "pause-and-ask", "--human-task", "T005=done",
        ]) == 3
        text = capsys.readouterr()
        text = text.out + text.err
        line = _account_line(repo)
        assert line.endswith(
            "5 — 1 done, 1 failed (T002), 2 not run (T003, T004), "
            "1 completed manual (T005)."
        ), line
        assert "Completed by hand (--human-task <id>=done): T005." in text
        assert "That is every one of the 5 task(s)" in text
        assert "| T005 | completed by hand |" in _summary(repo)


class TestTheReviewRecordNamesAnAcceptedFailure:

    def test_an_accepted_check_is_told_to_the_reviewers(self, repo, capsys):
        import code_review_orchestrator

        _run_one_that_stops(repo, capsys)
        assert _implement(repo, "--resume", "--human-task", "T002=done") == 0
        record = code_review_orchestrator.build_implementation_record(repo.plan)
        line = next((l for l in record.splitlines() if "whose failure the user accepted" in l), "")
        assert "T002 (accepted — you answered --human-task T002=done" in line, record
        assert "did not pass" in line


class TestTheStatusPollRelaysTheStopLineUnderExitZero:

    def test_exit_zero_keeps_the_error_on_the_result_line(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "sh"))
        plan = tmp_path / "plans" / "p.md"
        plan.parent.mkdir()
        plan.write_text("# p\n", encoding="utf-8")
        phase = plan.parent / "p" / "implement"
        phase.mkdir(parents=True)
        headline = "implement stopped at T009 (check: x): the check failed. Not run: T012, T013."
        status_file.claim_run(str(phase), {
            "run_id": "run-stop-1", "state": "failed", "exit_code": 3,
            "error": headline, "detached": True, "mode": "implement",
            "replay_flags": ["--host", "codex", "--detach"],
        })
        assert multi_llm.main([
            "status", str(plan), "--view", "run", "--run-id", "run-stop-1", "--exit-zero",
        ]) == 0
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        payload = json.loads(line.split(" ", 1)[1])
        assert payload["exit"] == 0 and payload["state"] == "needs-human"
        assert payload["error"] == headline


# --- a skip the user chose is not deferred work -------------------------------

class TestADeclinedSkipIsNotDeferredWork:

    def test_the_summary_files_declined_skips_under_their_own_heading(
        self, repo, capsys
    ):
        _run_one_that_stops(repo, capsys)
        assert _implement(repo, "--resume", "--human-task", "T002=skip") == 0
        summary = _summary(repo)
        deferred = _section(summary, "## Manual Tasks (Deferred)")
        declined = _section(summary, "## Skipped at your request")
        assert "| T005 |" in deferred
        assert "| T002 |" not in deferred and "| T004 |" not in deferred
        assert "| T002 | skipped — you answered --human-task T002=skip" in declined
        assert "| T004 | skipped — depends on skipped task T002" in declined
        # `[WORKFLOW_COMPLETE]` reads the same decision back from state.json.
        owed = get_or_create_state(repo.plan).deferred_implementation_tasks()
        assert [t["id"] for t in owed] == ["T005"]

    def test_the_two_legs_spell_the_declined_reasons_the_same(self):
        assert StateManager._USER_DECLINED_REASON == scheduler.USER_DECLINED_REASON
        assert StateManager._ANSWERED_SKIP_PREFIX == scheduler.ANSWERED_SKIP_PREFIX
        plan = scheduler.HumanTaskPlan(
            skipped={
                "T002": f"{scheduler.ANSWERED_SKIP_PREFIX}T002=skip after it failed",
                "T004": "skipped — depends on skipped task T002 (check)",
                "T005": scheduler._deferred_reason("skip-dependents"),
                "T006": "skipped — depends on skipped task T005 (manual)",
                "T007": scheduler.USER_DECLINED_REASON,
            },
            chain={"T004": "T002", "T006": "T005"},
        )
        assert sorted(plan.declined()) == ["T002", "T004", "T007"]
        assert sorted(plan.deferred()) == ["T005", "T006"]
