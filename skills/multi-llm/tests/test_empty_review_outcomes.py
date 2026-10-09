#!/usr/bin/env python3
"""An empty review outcome reaches its apply phase as a completion (T3).

Plan improve-plan-and-improve-code §2.3, *Empty apply outcomes*, and the
labelled `full` change of its Goal 4. Two phase-level changes, driven through
the real orchestrators against the mock provider binaries:

* review-plan's two zero-group exits (the fan-out's ``else`` of ``if
  all_suggestions:`` and the base ``--reaggregate``), shared with
  review-tasks, write a zero-group ``grouped.json`` and ``validation.json`` as
  ``[]`` and retire the previous validation round, as code-review's own empty
  branch does. Before them a clean plan review left apply-suggestions nothing
  to load (exit 1, "Validation results not found"), which stopped `full` at
  exit 3, and an empty ``--force`` round reported and handed out the previous
  round's groups.
* apply-code-fixes' no-items exit records ``apply-code-fixes`` completed,
  stamped now, unless ``--dry-run``.

A zero-findings review here is always a review run whose every model answered
``[]``, never hand-written review files (T11). That is also what pins the
fan-out reading ``[]`` as an answer: both of ``run_single_model``'s success
returns used to turn it into ``""``, record each model failed with "Unknown
error" and exit 3 "All models failed" (measured 2026-10-09).

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_empty_review_outcomes.py -v
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import full_workflow_orchestrator as fwo  # noqa: E402
import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from check_workflow_prerequisites import (  # noqa: E402
    check_apply_task_suggestions_prerequisite,
)
from harness import SkillRunner  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

#: Two reviewers, so "every model answered `[]`" is more than one slot.
MODELS = ("--models", "cursor-agent:auto", "gemini:gemini-2.5-flash")


PLAN_BODY = """# Upload Plan

## Overview
Upload the user's files to the archive.

## Step 1
Retry a failed upload.
"""

TASKS_BODY = (
    "# Implementation Tasks\n\n"
    "## T001: Add the upload client\n"
    "Create the client that sends one file.\n\n"
    "## T002: Retry failed uploads\n"
    "Wrap the client in a bounded retry.\n"
)

SUGGESTION = {
    "title": "Bound the upload retry",
    "desc": "Step 1 retries a failed upload without saying how often. Name the "
            "number of attempts and the backoff between them.",
    "type": "clarification",
    "reference": "## Step 1",
    "importance": "MEDIUM",
}

#: Two code-review findings in different files, so they stay two groups.
CODE_ISSUES = [
    {
        "title": "Unchecked upload status",
        "desc": "The client ignores the HTTP status of the upload response.",
        "importance": "medium",
        "file": "src/upload/client.py",
        "line_range": [10, 12],
        "type": "bug",
        "anchor_text": "def send(path):",
    },
    {
        "title": "Retry loop has no bound",
        "desc": "The retry loop never stops on a permanent failure.",
        "importance": "high",
        "file": "src/upload/retry.py",
        "line_range": [4, 9],
        "type": "bug",
        "anchor_text": "while True:",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan(tmp_path: Path, name: str = "upload-plan") -> Path:
    """A plan with a task list beside it, so review-tasks can run on it too."""
    plan = tmp_path / f"{name}.md"
    plan.write_text(PLAN_BODY, encoding="utf-8")
    tasks = tmp_path / name / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    (tasks / "tasks.md").write_text(TASKS_BODY, encoding="utf-8")
    return plan


def _runner(tmp_path: Path, response: Any) -> SkillRunner:
    """Mock providers that each answer *response* to every prompt."""
    runner = SkillRunner(tmp_path / "harness")
    runner.set_config({"response": response})
    return runner


def _phase(plan: Path, phase: str) -> Path:
    return plan.parent / plan.stem / phase


def _state(plan: Path) -> Dict[str, Any]:
    return json.loads(
        (plan.parent / plan.stem / "state.json").read_text(encoding="utf-8")
    )


def _completed_at(plan: Path, phase: str) -> datetime:
    return datetime.fromisoformat(_state(plan)["phases_completed"][phase])


def _groups(phase_dir: Path) -> List[Dict[str, Any]]:
    payload = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
    return payload["groups"]


def _validation(phase_dir: Path) -> Any:
    return json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))


def _batches(plan: Path, phase_dir_name: str) -> List[Any]:
    document = plan.parent / plan.stem / phase_dir_name / "orchestrator_output.json"
    return json.loads(document.read_text(encoding="utf-8"))["batches"]


def _ok(result) -> None:
    assert result.exit_code == 0, (
        f"exit {result.exit_code}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _review(runner: SkillRunner, name: str, plan: Path, *extra: str, models=MODELS):
    return runner.run_orchestrator(name, plan, *models, "--yes", *extra, timeout=120)


# ---------------------------------------------------------------------------
# 1. The zero-group exits write both files
# ---------------------------------------------------------------------------

REVIEW_PHASES = [
    pytest.param("review_plan", "review-plan", id="review-plan"),
    pytest.param("review_tasks", "review-tasks", id="review-tasks"),
]


class TestAZeroFindingsReviewWritesBothFiles:
    @pytest.mark.parametrize("orchestrator, phase_name", REVIEW_PHASES)
    def test_the_fan_out_writes_zero_groups_and_an_empty_validation(
        self, tmp_path, orchestrator, phase_name
    ):
        runner = _runner(tmp_path, [])
        plan = _plan(tmp_path)

        result = _review(runner, orchestrator, plan)

        _ok(result)
        assert result.mock_was_invoked()
        # Every model answered `[]`, and an answer is not a failure.
        assert "All models failed" not in result.stdout
        assert "[PARTIAL_RESULTS" not in result.stdout
        assert "Unknown error" not in result.stdout
        phase = _phase(plan, phase_name)
        for stem in ("cursor-agent_auto", "gemini_gemini-2_5-flash"):
            assert json.loads((phase / f"{stem}.json").read_text(encoding="utf-8")) == []
        assert _groups(phase) == []
        assert _validation(phase) == []
        assert not (phase / "validation_tasks.json").exists()
        report = (phase / "report.md").read_text(encoding="utf-8")
        assert "**Groups:** 0" in report
        assert "_No suggestions found._" in report
        assert phase_name in _state(plan)["phases_completed"]

    @pytest.mark.parametrize("orchestrator, phase_name", REVIEW_PHASES)
    def test_the_reaggregate_writes_them_too(
        self, tmp_path, orchestrator, phase_name
    ):
        """The reaggregate's regroup retires the ``validation.json`` the
        fan-out wrote, so the ``[]`` left on disk is the zero-group exit's own
        write, not the fan-out's survivor."""
        runner = _runner(tmp_path, [])
        plan = _plan(tmp_path)
        _ok(_review(runner, orchestrator, plan))
        phase = _phase(plan, phase_name)

        result = runner.run_orchestrator(
            orchestrator, plan, "--reaggregate", "--yes", timeout=120,
        )

        _ok(result)
        assert "[VALIDATION_RETIRED] validation." in result.stdout
        assert [p.read_text(encoding="utf-8") for p in phase.glob("validation.*.stale.json")] == ["[]\n"]
        assert _groups(phase) == []
        assert _validation(phase) == []
        assert "**Groups:** 0" in (phase / "report.md").read_text(encoding="utf-8")

    def test_apply_suggestions_then_completes_with_no_batch(self, tmp_path):
        """Exit 0, recorded completed after the review, nothing handed out,
        and no ``--exec-subagents`` needed (§2.3: measured exit 1 before)."""
        runner = _runner(tmp_path, [])
        plan = _plan(tmp_path)
        _ok(_review(runner, "review_plan", plan))

        result = runner.run_orchestrator(
            "apply_suggestions", plan, "--yes", "--agent-decide", timeout=120,
        )

        _ok(result)
        assert "Validation results not found" not in result.stderr
        assert _batches(plan, "apply-suggestions") == []
        assert _completed_at(plan, "apply-suggestions") > _completed_at(plan, "review-plan")
        assert "apply-suggestions" not in (_state(plan).get("phases_started") or {})

    def test_a_zero_findings_task_review_reads_as_skipped_no_findings(self, tmp_path):
        """review-tasks shares the branch: its zero-group ``grouped.json`` is
        what the prerequisite counter reads as ``skipped (no findings)``,
        where it used to find no file ("No review-tasks results to apply")."""
        runner = _runner(tmp_path, [])
        plan = _plan(tmp_path)
        _ok(_review(runner, "review_tasks", plan))

        check = check_apply_task_suggestions_prerequisite(
            StateManager(plan), plan.parent / plan.stem
        )

        assert check["met"] is True
        assert check["reason"] == "skipped (no findings)"


# ---------------------------------------------------------------------------
# 2. An empty --force round replaces the previous round's findings
# ---------------------------------------------------------------------------

class TestAnEmptyForceRoundAfterARoundWithFindings:
    def test_the_old_groups_are_neither_reported_nor_handed_out(self, tmp_path):
        runner = _runner(tmp_path, [SUGGESTION])
        plan = _plan(tmp_path)
        phase = _phase(plan, "review-plan")

        # Round 1: one finding, validated by the (simulated) host, merged,
        # handed out as one batch and closed by the native `--mark-completed`.
        first = _review(runner, "review_plan", plan, models=MODELS)
        _ok(first)
        assert "[VALIDATION_PENDING]" in first.stdout
        runner.simulate_subagents(phase / "validation_tasks.json")
        _ok(runner.run_orchestrator(
            "review_plan", plan, "--reaggregate", "--yes", timeout=120,
        ))
        old_theme = _groups(phase)[0]["theme"]
        assert old_theme in (phase / "report.md").read_text(encoding="utf-8")
        _ok(runner.run_orchestrator(
            "apply_suggestions", plan, "--yes", "--agent-decide", timeout=120,
        ))
        assert len(_batches(plan, "apply-suggestions")) == 1
        _ok(runner.run_orchestrator(
            "apply_suggestions", plan, "--mark-completed", timeout=120,
        ))
        first_apply = _completed_at(plan, "apply-suggestions")

        # Round 2: a fresh review round in which every model answers `[]`.
        runner.set_config({"response": []})
        second = _review(
            runner, "review_plan", plan, "--force", "--rerun-all", models=MODELS,
        )

        _ok(second)
        assert _groups(phase) == []
        assert _validation(phase) == []
        # The whole round retires (renamed, never deleted).
        assert not (phase / "validation_tasks.json").exists()
        assert list(phase.glob("validation_tasks.*.stale.json"))
        assert list(phase.glob("validation_batch_0.*.stale.json"))
        report = (phase / "report.md").read_text(encoding="utf-8")
        assert old_theme not in report
        assert "**Groups:** 0" in report

        again = runner.run_orchestrator(
            "apply_suggestions", plan, "--yes", "--agent-decide", timeout=120,
        )

        _ok(again)
        assert _batches(plan, "apply-suggestions") == []
        stamp = _completed_at(plan, "apply-suggestions")
        assert stamp > _completed_at(plan, "review-plan") > first_apply


class TestANativeApplyResumedWithNothingLeft:
    def test_the_completion_restamps_the_plan_its_batches_edited(self, tmp_path):
        """The native host applied every apply-suggestions batch and died
        before `--mark-completed`. Its `--resume` hands out nothing and
        records the completion at emit; the plan edits on disk are the
        plugin's, so the record re-stamps `plan_hash` as `--mark-completed`
        would, and a later `implement --resume` reads no human edit (Fable
        review 2026-10-09)."""
        from utils import batch_done

        runner = _runner(tmp_path, [SUGGESTION])
        plan = _plan(tmp_path)
        phase = _phase(plan, "review-plan")
        _ok(_review(runner, "review_plan", plan))
        runner.simulate_subagents(phase / "validation_tasks.json")
        _ok(runner.run_orchestrator(
            "review_plan", plan, "--reaggregate", "--yes", timeout=120,
        ))
        _ok(runner.run_orchestrator(
            "apply_suggestions", plan, "--yes", "--agent-decide", timeout=120,
        ))
        apply_dir = _phase(plan, "apply-suggestions")
        document = json.loads(
            (apply_dir / "write_tasks.json").read_text(encoding="utf-8")
        )
        tasks = document.get("batches") or document.get("tasks") or []
        assert tasks, document
        # The host's edit, then its per-batch records; no `--mark-completed`.
        with plan.open("a", encoding="utf-8") as handle:
            handle.write("\n## Applied by the host\n")
        for task in tasks:
            assert batch_done.main([
                "--phase-dir", str(apply_dir), "--task-id", task["task_id"],
                "--status", "completed",
            ]) == 0

        resumed = runner.run_orchestrator(
            "apply_suggestions", plan, "--yes", "--agent-decide", "--resume",
            timeout=120,
        )

        _ok(resumed)
        assert _batches(plan, "apply-suggestions") == []
        state = StateManager(plan)
        assert state.is_phase_completed("apply-suggestions")
        assert not state.get("plan_changed")


# ---------------------------------------------------------------------------
# 3. apply-code-fixes' no-items exit records the completion
# ---------------------------------------------------------------------------

def _all_invalid_code_review(
    tmp_path: Path, runner: SkillRunner, verdict: str = "invalid",
) -> Path:
    """A plan whose real code review ends with every finding judged *verdict*.

    The fan-out stops at its validation marker, the (simulated) host judges
    every group ``invalid`` (or *verdict*), and ``--reaggregate`` merges the
    verdicts.
    """
    plan = _plan(tmp_path, "code-plan")
    state = StateManager(plan)
    state.mark_phase_completed("review-plan")
    state.mark_phase_completed("implement")
    state.set("head_at_start", "abc123def")
    for issue in CODE_ISSUES:
        state.track_file(issue["file"], "created", "T001")
    state.save()
    implement = _phase(plan, "implement")
    implement.mkdir(parents=True, exist_ok=True)
    (implement / "summary.json").write_text(
        json.dumps({"status": "completed", "tasks_completed": 2, "tasks_total": 2}),
        encoding="utf-8",
    )
    runner.set_config({"response": CODE_ISSUES})
    first = runner.run_orchestrator(
        "code_review", plan, "--models", "cursor-agent:auto", "--yes", timeout=120,
    )
    _ok(first)
    phase = _phase(plan, "code-review")

    def invalid(task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "groups": [
                {
                    "group_index": index,
                    "group_hash": group_hash,
                    "status": verdict,
                    "reason": f"simulated: judged {verdict}",
                    "confidence": 0.9,
                }
                for index, group_hash in zip(
                    task.get("group_indices", []), task.get("group_ids", [])
                )
            ],
            "metadata": {
                "model": "harness",
                "timestamp": "1970-01-01T00:00:00",
                "schema_version": "2.1",
            },
        }

    runner.simulate_subagents(phase / "validation_tasks.json", result_for=invalid)
    _ok(runner.run_orchestrator(
        "code_review", plan, "--reaggregate", "--yes", timeout=120,
    ))
    assert len(_groups(phase)) == 2
    statuses = {entry["status"] for entry in _validation(phase)["groups"]}
    assert statuses == {verdict}, statuses
    assert "code-review" in _state(plan)["phases_completed"]
    return plan


class TestApplyCodeFixesWithNothingLeftToApply:
    def test_an_all_invalid_review_records_the_completion(self, tmp_path):
        runner = SkillRunner(tmp_path / "harness")
        plan = _all_invalid_code_review(tmp_path, runner)

        result = runner.run_orchestrator("apply_fixes", plan, "--yes", timeout=120)

        _ok(result)
        assert "No fixes to apply." in result.stderr
        assert _batches(plan, "apply-fixes") == []
        completed = _state(plan)["phases_completed"]
        # The word `--mark-completed` persists, never the dir word.
        assert "apply-fixes" not in completed
        assert _completed_at(plan, "apply-code-fixes") > _completed_at(plan, "code-review")
        assert StateManager(plan).is_phase_completed("apply-code-fixes")

    def test_a_rerun_restamps_an_earlier_cycles_completion(self, tmp_path):
        """A completion, never a skip: a skip would leave the older stamp the
        newest record, and completed wins over skipped (§2.3)."""
        runner = SkillRunner(tmp_path / "harness")
        plan = _all_invalid_code_review(tmp_path, runner)
        _ok(runner.run_orchestrator("apply_fixes", plan, "--yes", timeout=120))
        first = _completed_at(plan, "apply-code-fixes")

        _ok(runner.run_orchestrator("apply_fixes", plan, "--yes", timeout=120))

        assert _completed_at(plan, "apply-code-fixes") > first
        assert "apply-code-fixes" not in (_state(plan).get("phases_skipped") or {})

    def test_a_dry_run_records_nothing(self, tmp_path):
        runner = SkillRunner(tmp_path / "harness")
        plan = _all_invalid_code_review(tmp_path, runner)
        before = _state(plan)

        result = runner.run_orchestrator(
            "apply_fixes", plan, "--yes", "--dry-run", timeout=120,
        )

        _ok(result)
        after = _state(plan)
        assert "apply-code-fixes" not in after["phases_completed"]
        assert "apply-code-fixes" not in (after.get("phases_skipped") or {})
        assert after["phases_completed"] == before["phases_completed"]

    def test_a_dry_run_subagents_preflight_records_nothing(self, tmp_path):
        """`--dry-run-subagents` renders every command and starts no child, so
        it records nothing either; a completion here had a later real run read
        the phase done (Fable review 2026-10-09)."""
        runner = SkillRunner(tmp_path / "harness")
        plan = _all_invalid_code_review(tmp_path, runner)
        before = _state(plan)

        result = runner.run_orchestrator(
            "apply_fixes", plan, "--yes", "--exec-subagents", "--dry-run-subagents",
            timeout=120,
        )

        _ok(result)
        assert "No fixes to apply." in result.stderr
        after = _state(plan)
        assert "apply-code-fixes" not in after["phases_completed"]
        assert "apply-code-fixes" not in (after.get("phases_skipped") or {})
        assert after["phases_completed"] == before["phases_completed"]
        assert not StateManager(plan).is_phase_completed("apply-code-fixes")

    def test_an_all_invalid_review_says_nothing_about_tests(self, tmp_path):
        """Nothing was applied, so there is no `[TESTS_NOT_RUN]` line to owe."""
        runner = SkillRunner(tmp_path / "harness")
        plan = _all_invalid_code_review(tmp_path, runner)

        result = runner.run_orchestrator("apply_fixes", plan, "--yes", timeout=120)

        _ok(result)
        assert "[TESTS_NOT_RUN]" not in result.stderr

    def test_a_resume_after_every_batch_landed_closes_like_mark_completed(
        self, tmp_path
    ):
        """The native host died after its last batch and before
        `--mark-completed`; its `--resume` drops every applied group (step 23)
        and reaches this exit, which now records the completion AND says what
        `--mark-completed` says about the repository's tests (Fable review
        2026-10-09), so the record and the `[TESTS_NOT_RUN]` line agree."""
        from utils import batch_done

        runner = SkillRunner(tmp_path / "harness")
        plan = _all_invalid_code_review(tmp_path, runner, verdict="valid")

        emit = runner.run_orchestrator("apply_fixes", plan, "--yes", timeout=120)
        _ok(emit)
        phase_dir = _phase(plan, "apply-fixes")
        document = json.loads(
            (phase_dir / "write_tasks.json").read_text(encoding="utf-8")
        )
        tasks = document.get("batches") or document.get("tasks") or []
        assert tasks, document
        for task in tasks:
            assert batch_done.main([
                "--phase-dir", str(phase_dir), "--task-id", task["task_id"],
                "--status", "completed",
            ]) == 0
        assert not StateManager(plan).is_phase_completed("apply-code-fixes")

        resumed = runner.run_orchestrator(
            "apply_fixes", plan, "--yes", "--resume", timeout=120,
        )

        _ok(resumed)
        assert "No fixes to apply." in resumed.stderr
        assert StateManager(plan).is_phase_completed("apply-code-fixes")
        tests = [
            line for line in resumed.stderr.splitlines()
            if line.startswith("[TESTS_NOT_RUN] ")
        ]
        assert len(tests) == 1, resumed.stderr
        assert "driver.verify_command is not set" in tests[0], tests[0]


# ---------------------------------------------------------------------------
# 4. `full` (the labelled change of Goal 4)
# ---------------------------------------------------------------------------

@pytest.fixture
def full_run(tmp_path, monkeypatch):
    """`full` in-process with the named phases real and the rest stubbed.

    The real phases run against the mock provider binaries, every one of
    which answers *response*; a stubbed phase records the completion a real
    one would, as ``test_full_workflow_orchestrator.stub_phases`` does.
    Returns ``run(plan, real, response, *flags) -> (code, modes called)``.
    """
    import utils.provider_registry as registry

    original = fwo.FullWorkflow.run_mode

    def run(plan: Path, real, response: Any, *flags: str):
        runner = _runner(tmp_path, response)
        for key, value in runner._build_env().items():
            monkeypatch.setenv(key, value)
        registry._config = None
        registry._config_key = None
        calls: List[str] = []

        def run_mode(self, mode_word, extra):
            calls.append(mode_word)
            if mode_word in real:
                return original(self, mode_word, extra)
            phase = mode_table.phase_for(mode_word) or mode_word
            manager = StateManager(self.plan_path)
            manager.mark_phase_completed(phase)
            manager.save()
            return 0

        monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)
        code = multi_llm.main(["full", str(plan), "--yes", *flags])
        return code, calls

    yield run
    registry._config = None
    registry._config_key = None


def _complete(plan: Path, *phases: str) -> None:
    state = StateManager(plan)
    for phase in phases:
        state.mark_phase_completed(phase)
    state.save()


class TestFullOverEmptyReviews:
    def test_a_clean_plan_review_completes_apply_suggestions_and_the_run_goes_on(
        self, tmp_path, full_run, capsys
    ):
        plan = _plan(tmp_path)
        # Phase 3 is pre-recorded: this test is about the two plan phases.
        _complete(plan, "generate-tasks")

        code, called = full_run(
            plan, {"review-plan", "apply-suggestions"}, [],
            "--models", "cursor-agent:auto",
        )

        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "Validation results not found" not in captured.err
        state = StateManager(plan)
        assert state.is_phase_completed("apply-suggestions")
        assert not state.is_phase_skipped("apply-suggestions")
        assert _completed_at(plan, "apply-suggestions") > _completed_at(plan, "review-plan")
        assert called[:2] == ["review-plan", "apply-suggestions"]
        assert "implement" in called and "review-code" in called
        assert state.get(fwo.FullWorkflow.STATE_KEY)["current_phase"] == "complete"

    def test_a_zero_findings_task_review_skips_apply_task_suggestions(
        self, tmp_path, full_run, capsys
    ):
        plan = _plan(tmp_path)
        _complete(plan, "review-plan", "apply-suggestions", "generate-tasks")

        code, called = full_run(
            plan, {"review-tasks"}, [], "--resume", "--models", "cursor-agent:auto",
        )

        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert called[0] == "review-tasks"
        assert "apply-task-suggestions" not in called
        state = StateManager(plan)
        assert state.is_phase_skipped("apply-task-suggestions")
        assert "skipped (no findings)" in state.get_phase_skip_reason(
            "apply-task-suggestions"
        )
        assert "No review-tasks results to apply" not in state.get_phase_skip_reason(
            "apply-task-suggestions"
        )

    def test_an_all_invalid_code_review_completes_apply_code_fixes(
        self, tmp_path, full_run, capsys
    ):
        """It used to stop `full` at exit 3: the phase ran (its
        ``_grouped_has_suggestions`` shortcut ignores verdicts), exited 0 and
        recorded nothing, so ``_verify`` read "records no completion"."""
        plan = _all_invalid_code_review(tmp_path, SkillRunner(tmp_path / "review-harness"))
        _complete(
            plan, "apply-suggestions", "generate-tasks", "review-tasks",
            "apply-task-suggestions",
        )

        code, called = full_run(plan, {"apply-code-fixes"}, [], "--resume")

        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "records no completion" not in captured.err
        assert called == ["apply-code-fixes"]
        assert _completed_at(plan, "apply-code-fixes") > _completed_at(plan, "code-review")
        assert StateManager(plan).get(fwo.FullWorkflow.STATE_KEY)["current_phase"] == "complete"
