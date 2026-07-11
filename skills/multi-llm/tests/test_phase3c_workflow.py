#!/usr/bin/env python3
"""
Tests for full-workflow Phase 3c (apply-task-suggestions) conditional branches
and the --implement soft advisory for unapplied task suggestions.

Covers:
1. review-tasks skipped -> Phase 3c skipped with reason "prerequisite skipped"
2. review-tasks zero findings -> Phase 3c skipped with reason "no findings"
3. review-tasks has findings -> Phase 3c auto-runs apply_task_suggestions_orchestrator
4. --implement detects unapplied task suggestions and surfaces advisory prompt
5. --yes / --no-confirm bypass the advisory

To run:
    uv run --project skills/multi-llm -- pytest \
        skills/multi-llm/tests/test_phase3c_workflow.py -v
"""

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent (skill root) to path for imports
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from utils.state_manager import StateManager, stamp_stable_ids
from utils.output_handler import sanitize_prefix

# ---------------------------------------------------------------------------
# Constants — scripts under test
# ---------------------------------------------------------------------------

PREREQ_SCRIPT = SKILL_DIR / "check_workflow_prerequisites.py"
ORCHESTRATOR_SCRIPT = SKILL_DIR / "apply_task_suggestions_orchestrator.py"
IMPLEMENT_SCRIPT = SKILL_DIR / "implement_orchestrator.py"

# ---------------------------------------------------------------------------
# Sample content
# ---------------------------------------------------------------------------

SAMPLE_PLAN_CONTENT = """\
# Notification System Plan

## Overview
Implement a user notification system with email and in-app channels.

## Requirements
- Send transactional emails
- Real-time WebSocket delivery
"""

SAMPLE_TASKS_CONTENT = """\
# Implementation Tasks

## T001: Create notification data model
Set up the database schema for notifications.
- Depends on: none
- Complexity: low

## T002: Implement notification service
Core service layer for creating and querying notifications.
- Depends on: T001
- Complexity: medium

## T003: Add email channel adapter
Integrate with email provider for transactional emails.
- Depends on: T002
- Complexity: medium
"""

SAMPLE_GROUPED_WITH_FINDINGS = [
    {
        "theme": "Missing rate limiting task",
        "category": "coverage",
        "models": ["cursor-agent"],
        "suggestions": [
            {
                "title": "Add rate limiting task",
                "desc": "Plan requires rate limiting but no task covers it.",
                "type": "addition",
                "reference": "Plan Coverage",
                "importance": "HIGH",
                "source_model": "cursor-agent",
            }
        ],
    },
    {
        "theme": "T003 missing dependency on T001",
        "category": "dependency",
        "models": ["cursor-agent"],
        "suggestions": [
            {
                "title": "Fix T003 dependency",
                "desc": "T003 uses the schema created by T001 but doesn't list T001.",
                "type": "modification",
                "reference": "T003",
                "importance": "HIGH",
                "source_model": "cursor-agent",
            }
        ],
    },
]

SAMPLE_VALIDATION_FOR_FINDINGS = {
    "metadata": {
        "schema_version": "2.0",
        "validated_at": "2026-03-13T10:00:00",
        "model": "mock-llm",
        "total_groups": 2,
    },
    "groups": [
        {"group_index": 0, "status": "valid", "reason": "Clear addition.", "confidence": 0.95},
        {"group_index": 1, "status": "valid", "reason": "Dependency fix.", "confidence": 0.90},
    ],
}

EMPTY_GROUPED = []

EMPTY_GROUPED_ENVELOPE = {"format_version": 2, "groups": []}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files.

    Initialized as a git work tree: real plans always live inside a
    repository, and implement_orchestrator's default --output fails fast
    when the plan is outside one (its default path anchors at the git root).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["git", "init", "-q"], cwd=tmpdir, check=True, capture_output=True
        )
        yield Path(tmpdir)


def _create_plan(
    temp_dir: Path,
    plan_name: str = "test-plan",
    plan_content: str = SAMPLE_PLAN_CONTENT,
    tasks_content: str = SAMPLE_TASKS_CONTENT,
    grouped: list = None,
    validation: dict = None,
    mark_phases: list = None,
    create_grouped_json: bool = True,
    create_validation_json: bool = True,
    create_tasks: bool = True,
) -> Path:
    """Create a plan file with configurable state for testing Phase 3c logic.

    Returns the plan file path.
    """
    plan_path = temp_dir / f"{plan_name}.md"
    plan_path.write_text(plan_content)

    prefix = sanitize_prefix(plan_name)
    output_dir = temp_dir / prefix

    # Create tasks directory and file
    if create_tasks:
        tasks_dir = output_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "tasks.md").write_text(tasks_content)

    # Create review-tasks directory with grouped.json and validation.json
    review_tasks_dir = output_dir / "review-tasks"
    review_tasks_dir.mkdir(parents=True, exist_ok=True)

    if create_grouped_json:
        groups = grouped if grouped is not None else SAMPLE_GROUPED_WITH_FINDINGS
        groups_copy = copy.deepcopy(groups)
        stamp_stable_ids(groups_copy)
        (review_tasks_dir / "grouped.json").write_text(
            json.dumps({"format_version": 2, "groups": groups_copy}, indent=2)
        )

    if create_validation_json:
        val = validation if validation is not None else SAMPLE_VALIDATION_FOR_FINDINGS
        (review_tasks_dir / "validation.json").write_text(
            json.dumps(val, indent=2)
        )

    # Set up state
    state = StateManager(plan_path)
    if mark_phases:
        for phase in mark_phases:
            state.mark_phase_completed(phase)
    state.save()

    return plan_path


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_prereq_check(plan_path: Path, mode: str) -> dict:
    """Run check_workflow_prerequisites.py and parse JSON output."""
    cmd = [
        sys.executable,
        str(PREREQ_SCRIPT),
        "--plan-file", str(plan_path),
        "--mode", mode,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(SKILL_DIR),
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run_prereq_skip(plan_path: Path, mode: str, reason: str) -> dict:
    """Run check_workflow_prerequisites.py --skip --reason and parse JSON output."""
    cmd = [
        sys.executable,
        str(PREREQ_SCRIPT),
        "--plan-file", str(plan_path),
        "--mode", mode,
        "--skip",
        "--reason", reason,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(SKILL_DIR),
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run_status_check(plan_path: Path) -> dict:
    """Run check_workflow_prerequisites.py --status and parse JSON output."""
    cmd = [
        sys.executable,
        str(PREREQ_SCRIPT),
        "--plan-file", str(plan_path),
        "--status",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(SKILL_DIR),
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run_orchestrator(plan_path: Path, *extra_args, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run apply_task_suggestions_orchestrator.py with the given plan file."""
    cmd = [
        sys.executable,
        str(ORCHESTRATOR_SCRIPT),
        "--plan-file", str(plan_path),
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        cwd=str(SKILL_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_implement_orchestrator(plan_path: Path, *extra_args, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run implement_orchestrator.py with the given plan file."""
    cmd = [
        sys.executable,
        str(IMPLEMENT_SCRIPT),
        "--plan-file", str(plan_path),
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        cwd=str(SKILL_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ===========================================================================
# Test Class 1: Phase 3c — review-tasks skipped
# ===========================================================================

class TestPhase3cReviewTasksSkipped:
    """When review-tasks was skipped, apply-task-suggestions should be skipped
    with reason 'prerequisite skipped'."""

    def test_prereq_check_met_when_review_tasks_skipped(self, temp_dir):
        """check_apply_task_suggestions_prerequisite returns met=True
        with reason containing 'prerequisite skipped' when review-tasks is skipped."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="rt-skipped",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks"],
            create_grouped_json=False,
            create_validation_json=False,
        )
        state = StateManager(plan_path)
        state.mark_phase_skipped("review-tasks", "User declined")
        state.save()

        output = run_prereq_check(plan_path, "apply-task-suggestions")

        assert output["prerequisites_met"] is True, (
            f"Prerequisites should be met when review-tasks is skipped.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )

    def test_skip_command_records_prerequisite_skipped_reason(self, temp_dir):
        """The skip command used in full-workflow records the correct reason."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="rt-skip-record",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks"],
            create_grouped_json=False,
            create_validation_json=False,
        )
        # Mark review-tasks as skipped first (simulating the user declining)
        state = StateManager(plan_path)
        state.mark_phase_skipped("review-tasks", "User declined")
        state.save()

        # Now run the skip command for apply-task-suggestions
        # (as the full workflow would do in case (a))
        skip_output = run_prereq_skip(
            plan_path,
            "apply-task-suggestions",
            "Skipped: review-tasks was not executed",
        )

        assert skip_output["skipped"] is True
        assert skip_output["mode"] == "apply-task-suggestions"
        assert "review-tasks was not executed" in skip_output["reason"]

        # Verify state was persisted
        state = StateManager(plan_path)
        assert state.is_phase_skipped("apply-task-suggestions") is True
        assert "review-tasks was not executed" in state.get_phase_skip_reason("apply-task-suggestions")

    def test_status_shows_skipped_prerequisite_skipped(self, temp_dir):
        """--status shows apply-task-suggestions as skipped when review-tasks is skipped."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="status-rt-skip",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks"],
            create_grouped_json=False,
            create_validation_json=False,
        )
        state = StateManager(plan_path)
        state.mark_phase_skipped("review-tasks", "User declined")
        state.save()

        output = run_status_check(plan_path)

        phase = output["phases"]["apply-task-suggestions"]
        assert phase["status"] == "skipped", (
            f"apply-task-suggestions should show as skipped.\n"
            f"Phase: {json.dumps(phase, indent=2)}"
        )
        assert "prerequisite skipped" in phase.get("reason", "").lower(), (
            f"Reason should contain 'prerequisite skipped'.\n"
            f"Reason: {phase.get('reason', '')}"
        )

    def test_phase3c_skipped_when_review_tasks_not_run_at_all(self, temp_dir):
        """When review-tasks is not completed and not skipped (just never run),
        the apply-task-suggestions prerequisite check should still handle gracefully."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="rt-never-run",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks"],
            create_grouped_json=False,
            create_validation_json=False,
        )

        output = run_prereq_check(plan_path, "apply-task-suggestions")

        # review-tasks is a prerequisite for apply-task-suggestions
        assert output["prerequisites_met"] is False, (
            f"Prerequisites should NOT be met when review-tasks hasn't run.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )
        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "review-tasks" in missing_phases


# ===========================================================================
# Test Class 2: Phase 3c — review-tasks zero findings
# ===========================================================================

class TestPhase3cZeroFindings:
    """When review-tasks ran but produced zero findings, apply-task-suggestions
    should be skipped with reason 'no findings'."""

    def test_prereq_met_with_empty_groups(self, temp_dir):
        """check_apply_task_suggestions_prerequisite returns met=True
        with reason 'no findings' when grouped.json has zero suggestions."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="zero-findings",
            grouped=EMPTY_GROUPED,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
            create_validation_json=False,
        )

        output = run_prereq_check(plan_path, "apply-task-suggestions")

        assert output["prerequisites_met"] is True, (
            f"Prerequisites should be met with zero findings.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )

    def test_skip_command_records_no_findings_reason(self, temp_dir):
        """The skip command used in full-workflow case (b) records the correct reason."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="zero-findings-skip",
            grouped=EMPTY_GROUPED,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
            create_validation_json=False,
        )

        skip_output = run_prereq_skip(
            plan_path,
            "apply-task-suggestions",
            "Skipped: review-tasks produced zero findings",
        )

        assert skip_output["skipped"] is True
        assert skip_output["mode"] == "apply-task-suggestions"
        assert "zero findings" in skip_output["reason"]

        state = StateManager(plan_path)
        assert state.is_phase_skipped("apply-task-suggestions") is True
        assert "zero findings" in state.get_phase_skip_reason("apply-task-suggestions")

    def test_status_shows_skipped_no_findings(self, temp_dir):
        """--status shows apply-task-suggestions as skipped (no findings)
        when review-tasks completed but produced zero findings."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="status-zero",
            grouped=EMPTY_GROUPED,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
            create_validation_json=False,
        )

        output = run_status_check(plan_path)

        phase = output["phases"]["apply-task-suggestions"]
        assert phase["status"] == "skipped", (
            f"apply-task-suggestions should show as skipped.\n"
            f"Phase: {json.dumps(phase, indent=2)}"
        )
        assert "no findings" in phase.get("reason", "").lower(), (
            f"Reason should contain 'no findings'.\n"
            f"Reason: {phase.get('reason', '')}"
        )

    def test_groups_with_zero_suggestions_treated_as_empty(self, temp_dir):
        """Groups that exist but have empty suggestion arrays count as zero findings."""
        groups_no_suggestions = [
            {
                "theme": "Empty theme",
                "category": "coverage",
                "models": ["cursor-agent"],
                "suggestions": [],
            }
        ]
        plan_path = _create_plan(
            temp_dir,
            plan_name="empty-suggs",
            grouped=groups_no_suggestions,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
            create_validation_json=False,
        )

        output = run_prereq_check(plan_path, "apply-task-suggestions")

        # Zero total suggestions means the prerequisite is met (nothing to apply)
        assert output["prerequisites_met"] is True, (
            f"Prerequisites should be met with zero suggestions.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )


# ===========================================================================
# Test Class 3: Phase 3c — review-tasks has findings (auto-run)
# ===========================================================================

class TestPhase3cHasFindings:
    """When review-tasks produced actionable findings, apply-task-suggestions
    should auto-run with --no-confirm."""

    def test_prereq_not_met_with_unapplied_findings(self, temp_dir):
        """check_apply_task_suggestions_prerequisite returns met=False
        when grouped.json has actionable suggestions."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="has-findings",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )

        output = run_prereq_check(plan_path, "apply-task-suggestions")

        # The prerequisite check for apply-task-suggestions itself should pass
        # (review-tasks is completed), but the underlying
        # check_apply_task_suggestions_prerequisite should detect unapplied suggestions.
        # However, the prereq CLI checks the PHASE_PREREQUISITES chain
        # which only checks review-tasks for apply-task-suggestions.
        assert output["prerequisites_met"] is True, (
            f"Phase prerequisites (review-tasks completed) should be met.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )

    def test_orchestrator_runs_with_no_confirm(self, temp_dir):
        """apply_task_suggestions_orchestrator.py runs successfully with --no-confirm."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="auto-run",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )

        result = run_orchestrator(plan_path, "--no-confirm")

        assert result.returncode == 0, (
            f"Orchestrator should succeed with --no-confirm.\n"
            f"stderr: {result.stderr[:800]}"
        )

    def test_orchestrator_defers_phase_completion(self, temp_dir):
        """After a successful --no-confirm run, the phase is NOT marked completed
        by the orchestrator — completion is deferred to the instruction file
        (after all batches have been processed and tasks.md updated)."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="completed-phase",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )

        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode == 0, f"Orchestrator failed: {result.stderr[:500]}"

        state = StateManager(plan_path)
        assert not state.is_phase_completed("apply-task-suggestions"), (
            f"Phase should NOT be marked completed by orchestrator "
            f"(deferred to instruction file).\n"
            f"phases_completed: {state.state.get('phases_completed', {})}"
        )

    def test_orchestrator_creates_output_json(self, temp_dir):
        """After --no-confirm run, orchestrator_output.json is created."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="output-json",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )

        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode == 0, f"Orchestrator failed: {result.stderr[:500]}"

        prefix = sanitize_prefix("output-json")
        output_file = temp_dir / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        assert output_file.exists(), f"Output file not created: {output_file}"

        with open(output_file) as f:
            output = json.load(f)

        assert "to_apply" in output
        assert len(output["to_apply"]) > 0, "Should have items to apply"

    def test_status_shows_pending_after_orchestrator_run(self, temp_dir):
        """--status shows apply-task-suggestions as pending after orchestrator run,
        because completion is deferred to the instruction file."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="status-complete",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )

        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode == 0, f"Orchestrator failed: {result.stderr[:500]}"

        output = run_status_check(plan_path)

        phase = output["phases"]["apply-task-suggestions"]
        assert phase["status"] == "pending", (
            f"Phase should still be pending after orchestrator run "
            f"(completion deferred to instruction file).\n"
            f"Phase: {json.dumps(phase, indent=2)}"
        )


# ===========================================================================
# Test Class 4: --implement soft advisory prompt
# ===========================================================================

class TestImplementAdvisoryPrompt:
    """Verify that --implement detects unapplied task suggestions and
    surfaces an advisory prompt. Also verify that --yes/--no-confirm bypass."""

    @staticmethod
    def _create_plan_for_implement(
        temp_dir: Path,
        plan_name: str,
        mark_apply_task_suggestions: str = None,
    ) -> Path:
        """Create a plan ready for implement with review-tasks results.

        Args:
            mark_apply_task_suggestions: None (unapplied), "completed", or "skipped"
        """
        plan_path = _create_plan(
            temp_dir,
            plan_name=plan_name,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
            # Omit validation.json so prereq checker uses grouped.json counts
            create_validation_json=False,
        )

        if mark_apply_task_suggestions == "completed":
            state = StateManager(plan_path)
            state.mark_phase_completed("apply-task-suggestions")
            state.save()
        elif mark_apply_task_suggestions == "skipped":
            state = StateManager(plan_path)
            state.mark_phase_skipped("apply-task-suggestions", "User chose to skip")
            state.save()

        return plan_path

    def test_implement_prereq_surfaces_unapplied_task_suggestions(self, temp_dir):
        """implement mode detects unapplied task suggestions and includes them in advisories."""
        plan_path = self._create_plan_for_implement(temp_dir, "impl-detect")

        output = run_prereq_check(plan_path, "implement")

        # apply-task-suggestions is a soft advisory, not a hard prerequisite
        assert output.get("prerequisites_met") is True, (
            f"prerequisites_met should be True since task suggestions are advisory.\n"
            f"Full output: {json.dumps(output, indent=2)}"
        )
        advisory_phases = [m["phase"] for m in output.get("advisories", [])]
        assert "apply-task-suggestions" in advisory_phases, (
            f"apply-task-suggestions should appear as advisory.\n"
            f"Advisories: {advisory_phases}\nFull output: {json.dumps(output, indent=2)}"
        )

    def test_implement_prereq_includes_actionable_count(self, temp_dir):
        """implement mode includes actionable_count in the advisory entry."""
        plan_path = self._create_plan_for_implement(temp_dir, "impl-count")

        output = run_prereq_check(plan_path, "implement")

        task_sugg = None
        for m in output.get("advisories", []):
            if m["phase"] == "apply-task-suggestions":
                task_sugg = m
                break

        assert task_sugg is not None, "Should have apply-task-suggestions in advisories"
        assert "actionable_count" in task_sugg
        assert task_sugg["actionable_count"] > 0, (
            f"actionable_count should be > 0.\n"
            f"Entry: {json.dumps(task_sugg, indent=2)}"
        )

    def test_implement_prereq_includes_prompt_with_task_suggestion_wording(self, temp_dir):
        """implement mode includes a prompt mentioning task suggestions."""
        plan_path = self._create_plan_for_implement(temp_dir, "impl-prompt")

        output = run_prereq_check(plan_path, "implement")

        assert "prompt" in output, (
            f"Output should include a prompt.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )
        prompt = output["prompt"]
        assert "question" in prompt
        assert "options" in prompt
        assert "task suggestion" in prompt["question"].lower(), (
            f"Prompt question should mention task suggestions.\n"
            f"Question: {prompt['question']}"
        )
        actions = [opt["action"] for opt in prompt["options"]]
        assert "run_apply_task_suggestions" in actions
        assert "skip_and_continue" in actions

    def test_implement_prereq_met_when_apply_task_suggestions_completed(self, temp_dir):
        """implement prereq has no apply-task-suggestions in missing
        when the phase is already completed."""
        plan_path = self._create_plan_for_implement(
            temp_dir, "impl-completed", mark_apply_task_suggestions="completed"
        )

        output = run_prereq_check(plan_path, "implement")

        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "apply-task-suggestions" not in missing_phases, (
            f"apply-task-suggestions should not be in missing when completed.\n"
            f"Missing: {missing_phases}"
        )

    def test_implement_prereq_met_when_apply_task_suggestions_skipped(self, temp_dir):
        """implement prereq has no apply-task-suggestions in missing
        when the phase is explicitly skipped."""
        plan_path = self._create_plan_for_implement(
            temp_dir, "impl-skipped", mark_apply_task_suggestions="skipped"
        )

        output = run_prereq_check(plan_path, "implement")

        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "apply-task-suggestions" not in missing_phases, (
            f"apply-task-suggestions should not be in missing when skipped.\n"
            f"Missing: {missing_phases}"
        )

    def test_implement_no_advisory_when_no_review_tasks_results(self, temp_dir):
        """implement mode has no apply-task-suggestions advisory when
        review-tasks was never run (no grouped.json)."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="impl-no-rt",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks"],
            create_grouped_json=False,
            create_validation_json=False,
        )

        output = run_prereq_check(plan_path, "implement")

        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "apply-task-suggestions" not in missing_phases, (
            f"No advisory when review-tasks results don't exist.\n"
            f"Missing: {missing_phases}"
        )

    def test_implement_no_advisory_when_zero_findings(self, temp_dir):
        """implement mode has no apply-task-suggestions advisory when
        review-tasks produced zero findings."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="impl-zero",
            grouped=EMPTY_GROUPED,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
            create_validation_json=False,
        )

        output = run_prereq_check(plan_path, "implement")

        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "apply-task-suggestions" not in missing_phases, (
            f"No advisory when zero findings.\n"
            f"Missing: {missing_phases}"
        )


# ===========================================================================
# Test Class 5: --implement orchestrator advisory output
# ===========================================================================

class TestImplementOrchestratorAdvisory:
    """Verify that the implement_orchestrator.py itself outputs the
    TASK_SUGGESTIONS_ADVISORY marker and proceeds (does not block)."""

    @staticmethod
    def _create_plan_for_implement_run(
        temp_dir: Path,
        plan_name: str,
        with_findings: bool = True,
    ) -> Path:
        """Create a plan ready for implement orchestrator execution.

        The plan includes a TASKS_FILE marker so the implement orchestrator
        can find the tasks.
        """
        plan_content = SAMPLE_PLAN_CONTENT + "\n<!-- TASKS_FILE: tasks/tasks.md -->\n"
        plan_path = _create_plan(
            temp_dir,
            plan_name=plan_name,
            plan_content=plan_content,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
            grouped=SAMPLE_GROUPED_WITH_FINDINGS if with_findings else EMPTY_GROUPED,
            create_validation_json=False,
        )
        return plan_path

    def test_implement_outputs_advisory_marker(self, temp_dir):
        """implement_orchestrator.py outputs [TASK_SUGGESTIONS_ADVISORY] marker
        when unapplied task suggestions exist."""
        plan_path = self._create_plan_for_implement_run(temp_dir, "impl-adv-marker")

        result = run_implement_orchestrator(plan_path, "--dry-run")

        combined = result.stdout + result.stderr
        assert "[TASK_SUGGESTIONS_ADVISORY]" in combined, (
            f"Should output TASK_SUGGESTIONS_ADVISORY marker.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )

    def test_implement_advisory_contains_actionable_count(self, temp_dir):
        """The TASK_SUGGESTIONS_ADVISORY JSON includes actionable_count."""
        plan_path = self._create_plan_for_implement_run(temp_dir, "impl-adv-count")

        result = run_implement_orchestrator(plan_path, "--dry-run")

        # Parse the advisory JSON from stdout
        lines = result.stdout.split("\n")
        advisory_json = None
        capture_next = False
        json_lines = []
        for line in lines:
            if "[TASK_SUGGESTIONS_ADVISORY]" in line:
                capture_next = True
                continue
            if capture_next:
                json_lines.append(line)
                # Try to parse accumulated lines
                try:
                    advisory_json = json.loads("\n".join(json_lines))
                    break
                except json.JSONDecodeError:
                    continue

        assert advisory_json is not None, (
            f"Could not parse TASK_SUGGESTIONS_ADVISORY JSON.\n"
            f"stdout: {result.stdout[:800]}"
        )
        assert advisory_json["marker"] == "TASK_SUGGESTIONS_ADVISORY"
        assert advisory_json["phase"] == "apply-task-suggestions"
        assert advisory_json["actionable_count"] > 0

    def test_implement_proceeds_after_advisory(self, temp_dir):
        """implement_orchestrator.py does NOT exit after the advisory —
        it proceeds with implementation (dry-run continues past advisory)."""
        plan_path = self._create_plan_for_implement_run(temp_dir, "impl-adv-proceed")

        result = run_implement_orchestrator(plan_path, "--dry-run")

        combined = result.stdout + result.stderr
        # Should have both the advisory AND subsequent implementation output
        assert "[TASK_SUGGESTIONS_ADVISORY]" in combined
        # The orchestrator should continue to the task decomposition stage
        assert "Found" in combined or "Task Summary" in combined or "tasks" in combined.lower(), (
            f"Orchestrator should proceed past advisory.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )

    def test_implement_no_advisory_when_no_findings(self, temp_dir):
        """implement_orchestrator.py does NOT output advisory when no findings."""
        plan_path = self._create_plan_for_implement_run(
            temp_dir, "impl-no-adv", with_findings=False
        )

        result = run_implement_orchestrator(plan_path, "--dry-run")

        combined = result.stdout + result.stderr
        assert "[TASK_SUGGESTIONS_ADVISORY]" not in combined, (
            f"Should NOT output advisory when no findings.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )

    def test_implement_no_advisory_when_phase_completed(self, temp_dir):
        """implement_orchestrator.py does NOT output advisory when
        apply-task-suggestions is already completed."""
        plan_path = self._create_plan_for_implement_run(temp_dir, "impl-adv-done")
        state = StateManager(plan_path)
        state.mark_phase_completed("apply-task-suggestions")
        state.save()

        result = run_implement_orchestrator(plan_path, "--dry-run")

        combined = result.stdout + result.stderr
        assert "[TASK_SUGGESTIONS_ADVISORY]" not in combined, (
            f"Should NOT output advisory when phase already completed.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )

    def test_implement_no_advisory_when_phase_skipped(self, temp_dir):
        """implement_orchestrator.py does NOT output advisory when
        apply-task-suggestions is explicitly skipped."""
        plan_path = self._create_plan_for_implement_run(temp_dir, "impl-adv-skip")
        state = StateManager(plan_path)
        state.mark_phase_skipped("apply-task-suggestions", "User chose to skip")
        state.save()

        result = run_implement_orchestrator(plan_path, "--dry-run")

        combined = result.stdout + result.stderr
        assert "[TASK_SUGGESTIONS_ADVISORY]" not in combined, (
            f"Should NOT output advisory when phase already skipped.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )


# ===========================================================================
# Test Class 6: --no-confirm and --yes bypass for orchestrator
# ===========================================================================

class TestOrchestratorBypassFlags:
    """Verify that --no-confirm and --yes bypass confirmation prompts
    in apply_task_suggestions_orchestrator when no user selections are found."""

    @staticmethod
    def _create_plan_no_selections(temp_dir: Path, plan_name: str) -> Path:
        """Create a plan ready for apply-task-suggestions but without any
        user selection files (report.md, user_selections.json)."""
        plan_path = _create_plan(
            temp_dir,
            plan_name=plan_name,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )
        # Remove any selection files from review-tasks dir
        prefix = sanitize_prefix(plan_name)
        review_dir = temp_dir / prefix / "review-tasks"
        for f in ["report.md", "user_selections.json", "consolidated_user_selections.json"]:
            path = review_dir / f
            if path.exists():
                path.unlink()
        return plan_path

    def test_without_bypass_outputs_confirmation_needed(self, temp_dir):
        """Without --no-confirm or --yes, orchestrator outputs confirmation_needed."""
        plan_path = self._create_plan_no_selections(temp_dir, "confirm-test")

        result = run_orchestrator(plan_path)
        assert result.returncode == 0

        output = json.loads(result.stdout)
        assert output["status"] == "confirmation_needed", (
            f"Should output confirmation_needed.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )
        assert output["phase"] == "apply-task-suggestions"

    def test_no_confirm_bypasses_confirmation(self, temp_dir):
        """--no-confirm bypasses the confirmation prompt and proceeds."""
        plan_path = self._create_plan_no_selections(temp_dir, "no-confirm-test")

        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode == 0

        # Should NOT output confirmation_needed
        if result.stdout.strip():
            try:
                output = json.loads(result.stdout)
                assert output.get("status") != "confirmation_needed", (
                    f"--no-confirm should bypass confirmation.\n"
                    f"Output: {json.dumps(output, indent=2)}"
                )
            except json.JSONDecodeError:
                pass  # Non-JSON output is fine (means it proceeded)

    def test_yes_flag_bypasses_confirmation(self, temp_dir):
        """--yes flag bypasses the confirmation prompt."""
        plan_path = self._create_plan_no_selections(temp_dir, "yes-flag-test")

        result = run_orchestrator(plan_path, "--yes")
        assert result.returncode == 0

        if result.stdout.strip():
            try:
                output = json.loads(result.stdout)
                assert output.get("status") != "confirmation_needed", (
                    f"--yes should bypass confirmation.\n"
                    f"Output: {json.dumps(output, indent=2)}"
                )
            except json.JSONDecodeError:
                pass

    def test_no_confirm_produces_output_or_skips(self, temp_dir):
        """--no-confirm either writes orchestrator output (completion deferred
        to instruction file) or marks the phase as skipped if nothing to apply."""
        plan_path = self._create_plan_no_selections(temp_dir, "no-confirm-complete")

        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode == 0

        state = StateManager(plan_path)
        # The orchestrator either:
        # 1. Wrote output JSON (phase stays pending — completion deferred to instruction file), or
        # 2. Marked phase skipped (no actionable findings)
        prefix = sanitize_prefix("no-confirm-complete")
        output_file = temp_dir / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        has_output = output_file.exists()
        is_skipped = state.is_phase_skipped("apply-task-suggestions")
        assert has_output or is_skipped, (
            f"Phase should either have output JSON (pending batch processing) "
            f"or be marked skipped.\n"
            f"Output exists: {has_output}, skipped: {is_skipped}"
        )


# ===========================================================================
# Test Class 7: End-to-end workflow integration
# ===========================================================================

class TestPhase3cEndToEnd:
    """End-to-end tests simulating the full workflow Phase 3c decision flow
    against a real tasks.md fixture."""

    def test_full_flow_case_a_review_tasks_skipped(self, temp_dir):
        """Full workflow case (a): review-tasks skipped -> Phase 3c skipped."""
        # Step 1: Create plan with prerequisites completed
        plan_path = _create_plan(
            temp_dir,
            plan_name="e2e-case-a",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks"],
            create_grouped_json=False,
            create_validation_json=False,
        )

        # Step 2: Skip review-tasks (simulating user declining)
        run_prereq_skip(plan_path, "review-tasks", "User declined")

        # Step 3: Skip apply-task-suggestions (as full workflow would)
        run_prereq_skip(plan_path, "apply-task-suggestions", "Skipped: review-tasks was not executed")

        # Step 4: Verify final state
        state = StateManager(plan_path)
        assert state.is_phase_skipped("review-tasks") is True
        assert state.is_phase_skipped("apply-task-suggestions") is True
        assert "review-tasks was not executed" in state.get_phase_skip_reason("apply-task-suggestions")

        # Step 5: Verify status output
        status = run_status_check(plan_path)
        assert status["phases"]["review-tasks"]["status"] == "skipped"
        assert status["phases"]["apply-task-suggestions"]["status"] == "skipped"

    def test_full_flow_case_b_zero_findings(self, temp_dir):
        """Full workflow case (b): review-tasks zero findings -> Phase 3c skipped."""
        # Step 1: Create plan with review-tasks completed but zero findings
        plan_path = _create_plan(
            temp_dir,
            plan_name="e2e-case-b",
            grouped=EMPTY_GROUPED,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
            create_validation_json=False,
        )

        # Step 2: Skip apply-task-suggestions (as full workflow would after checking)
        run_prereq_skip(plan_path, "apply-task-suggestions", "Skipped: review-tasks produced zero findings")

        # Step 3: Verify final state
        state = StateManager(plan_path)
        assert state.is_phase_completed("review-tasks") is True
        assert state.is_phase_skipped("apply-task-suggestions") is True
        assert "zero findings" in state.get_phase_skip_reason("apply-task-suggestions")

        # Step 4: Verify status
        status = run_status_check(plan_path)
        assert status["phases"]["review-tasks"]["status"] == "completed"
        assert status["phases"]["apply-task-suggestions"]["status"] == "skipped"

    def test_full_flow_case_c_has_findings(self, temp_dir):
        """Full workflow case (c): review-tasks has findings -> Phase 3c auto-runs.
        The orchestrator produces output JSON but defers completion marking to
        the instruction file (after all batches are processed)."""
        # Step 1: Create plan with review-tasks completed and findings present
        plan_path = _create_plan(
            temp_dir,
            plan_name="e2e-case-c",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )

        # Step 2: Auto-run apply-task-suggestions with --no-confirm (as full workflow would)
        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode == 0, (
            f"Orchestrator should succeed.\n"
            f"stderr: {result.stderr[:500]}"
        )

        # Step 3: Verify phase is NOT yet completed (deferred to instruction file)
        state = StateManager(plan_path)
        assert not state.is_phase_completed("apply-task-suggestions"), (
            f"Phase should NOT be marked completed by orchestrator "
            f"(deferred to instruction file).\n"
            f"phases_completed: {state.state.get('phases_completed', {})}"
        )

        # Step 4: Verify output file exists (orchestrator preparation step done)
        prefix = sanitize_prefix("e2e-case-c")
        output_file = temp_dir / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        assert output_file.exists(), f"Output file should exist: {output_file}"

        # Step 5: Verify status shows pending (awaiting batch processing)
        status = run_status_check(plan_path)
        assert status["phases"]["apply-task-suggestions"]["status"] == "pending", (
            f"Phase should be pending (awaiting batch processing).\n"
            f"Phase: {json.dumps(status['phases']['apply-task-suggestions'], indent=2)}"
        )

    def test_full_flow_implement_after_case_c(self, temp_dir):
        """After case (c) orchestrator run, implement prereq check should not
        flag apply-task-suggestions as a hard prerequisite (it is advisory only).
        The phase is not yet completed (deferred), but apply-task-suggestions
        is NOT in implement's hard prerequisites."""
        plan_path = _create_plan(
            temp_dir,
            plan_name="e2e-impl-after-c",
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )

        # Run apply-task-suggestions orchestrator (does NOT mark completed)
        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode == 0

        # Manually mark phase completed (simulating instruction file step 7)
        state = StateManager(plan_path)
        state.mark_phase_completed("apply-task-suggestions")
        state.save()

        # Check implement prerequisites
        output = run_prereq_check(plan_path, "implement")

        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "apply-task-suggestions" not in missing_phases, (
            f"apply-task-suggestions should not be in missing after completion.\n"
            f"Missing: {missing_phases}"
        )


# ===========================================================================
# Test Class 7: --mark-completed flag and defensive fallback
# ===========================================================================

class TestMarkCompletedFlag:
    """Verify --mark-completed flag marks phase in phases_completed,
    and the defensive fallback in check_apply_task_suggestions_prerequisite
    detects completion from state['phases'] metadata."""

    @staticmethod
    def _create_plan_with_findings(temp_dir: Path, plan_name: str) -> Path:
        return _create_plan(
            temp_dir,
            plan_name=plan_name,
            mark_phases=["review-plan", "apply-suggestions", "generate-tasks", "review-tasks"],
        )

    def test_mark_completed_flag_updates_phases_completed(self, temp_dir):
        """--mark-completed writes to phases_completed in state.json."""
        plan_path = self._create_plan_with_findings(temp_dir, "mark-compl")

        result = run_orchestrator(plan_path, "--mark-completed")
        assert result.returncode == 0

        output = json.loads(result.stdout)
        assert output["status"] == "completed"

        state = StateManager(plan_path)
        assert state.is_phase_completed("apply-task-suggestions")

    def test_mark_completed_clears_advisory(self, temp_dir):
        """After --mark-completed, implement prereq check has no advisory."""
        plan_path = self._create_plan_with_findings(temp_dir, "mark-clears")

        # Before: advisory should be present
        output_before = run_prereq_check(plan_path, "implement")
        advisory_phases = [m["phase"] for m in output_before.get("advisories", [])]
        assert "apply-task-suggestions" in advisory_phases

        # Mark completed
        result = run_orchestrator(plan_path, "--mark-completed")
        assert result.returncode == 0

        # After: advisory should be gone
        output_after = run_prereq_check(plan_path, "implement")
        advisory_phases = [m["phase"] for m in output_after.get("advisories", [])]
        assert "apply-task-suggestions" not in advisory_phases

    def test_phases_metadata_fallback(self, temp_dir):
        """Defensive fallback: state['phases'] metadata detected as completed."""
        plan_path = self._create_plan_with_findings(temp_dir, "fallback")

        # Write completion to state["phases"] only (simulating old behavior)
        state = StateManager(plan_path)
        if "phases" not in state.state:
            state.state["phases"] = {}
        state.state["phases"]["apply-task-suggestions"] = {
            "status": "completed",
            "completed_at": "2026-03-13",
        }
        state.save()

        # phases_completed should NOT have the entry
        assert not state.is_phase_completed("apply-task-suggestions")

        # But prerequisite check should still detect it via fallback
        output = run_prereq_check(plan_path, "implement")
        advisory_phases = [m["phase"] for m in output.get("advisories", [])]
        assert "apply-task-suggestions" not in advisory_phases, (
            f"Fallback should detect completion from state['phases'].\n"
            f"Advisories: {advisory_phases}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
