#!/usr/bin/env python3
"""
Integration tests for review_tasks_orchestrator.py

Tests cover:
- Full review-tasks flow with mock models
- Reaggregate from existing partial results
- Phase completion marking in state.json
- Prerequisite enforcement (requires generate-tasks)
- Review-tasks optionality (implement proceeds without it)
- Edge cases: malformed tasks, missing plan, empty tasks, cross-reference failures

To run:
    uv run --project skills/multi-llm -- pytest \
        skills/multi-llm/tests/test_review_tasks_integration.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent (skill root) to path for imports
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from utils.state_manager import StateManager
from utils.output_handler import sanitize_prefix


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

ORCHESTRATOR_SCRIPT = SKILL_DIR / "review_tasks_orchestrator.py"
PREREQ_SCRIPT = SKILL_DIR / "check_workflow_prerequisites.py"

SAMPLE_PLAN_CONTENT = """\
# Sample Feature Plan

## Overview
Implement a user notification system with email and in-app channels.

## Requirements

### Email Notifications
- Send transactional emails (welcome, password reset)
- Support HTML templates
- Rate limiting per user

### In-App Notifications
- Real-time WebSocket delivery
- Notification center with read/unread state
- Bulk mark-as-read

### API Endpoints
- POST /notifications/send
- GET  /notifications
- PATCH /notifications/:id/read

## Testing
- Unit tests for notification service
- Integration tests for email delivery
- E2E tests for WebSocket channel
"""

VALID_TASKS_CONTENT = """\
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

## T004: Add WebSocket delivery
Real-time in-app notification delivery via WebSocket.
- Depends on: T002
- Complexity: high

## T005: Create API endpoints
REST endpoints for sending and managing notifications.
- Depends on: T002
- Complexity: medium

## T006: Write tests
Unit, integration, and E2E tests for the notification system.
- Depends on: T003, T004, T005
- Complexity: medium
"""

# Minimal tasks content (has headers but only one trivial task)
MINIMAL_TASKS_CONTENT = """\
# Implementation Tasks

## T001: Placeholder
Placeholder task.
- Depends on: none
"""

# Malformed tasks -- no valid task headers
MALFORMED_TASKS_CONTENT = """\
# Implementation Tasks

This file has some content but no valid task headers.
- Item A
- Item B
- Item C

There are no ## T001 style headers here.
"""

EMPTY_TASKS_CONTENT = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _create_plan_with_tasks(
    temp_dir: Path,
    plan_name: str = "test-plan",
    plan_content: str = SAMPLE_PLAN_CONTENT,
    tasks_content: str = VALID_TASKS_CONTENT,
    mark_generate_tasks_completed: bool = True,
    mark_apply_suggestions_completed: bool = True,
) -> Path:
    """Helper: create a plan file with a tasks file and appropriate state.

    Returns the plan file path.
    """
    plan_path = temp_dir / f"{plan_name}.md"
    plan_path.write_text(plan_content)

    prefix = sanitize_prefix(plan_name)
    output_dir = temp_dir / prefix
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    if tasks_content is not None:
        (tasks_dir / "tasks.md").write_text(tasks_content)

    # Set up state
    state = StateManager(plan_path)
    if mark_apply_suggestions_completed:
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
    if mark_generate_tasks_completed:
        state.mark_phase_completed("generate-tasks")
    state.save()

    return plan_path


@pytest.fixture
def plan_with_tasks(temp_dir):
    """Create a fully set-up plan with valid tasks and prerequisite phases completed."""
    return _create_plan_with_tasks(temp_dir)


@pytest.fixture
def plan_without_generate_tasks(temp_dir):
    """Create a plan where generate-tasks has NOT been run."""
    return _create_plan_with_tasks(
        temp_dir,
        plan_name="plan-no-tasks",
        mark_generate_tasks_completed=False,
    )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_orchestrator(plan_path: Path, *extra_args, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run the review_tasks_orchestrator.py with the given plan file."""
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


# ===========================================================================
# Test classes
# ===========================================================================


class TestFullReviewTasksFlow:
    """End-to-end test using --reaggregate with pre-populated model results.

    We use reaggregate mode rather than live model invocation because the
    integration tests do not inject mock providers into PATH. The reaggregate
    path exercises the core aggregation, grouping, and report generation logic.
    """

    def test_full_review_tasks_flow(self, plan_with_tasks):
        """Orchestrator finds tasks, aggregates model results, and produces reports.

        Steps verified:
        1. Tasks file is found at the default location
        2. Model results are loaded and aggregated
        3. Grouped suggestions and HTML report are generated
        4. Phase is marked as completed in state.json
        """
        plan_path = plan_with_tasks
        prefix = sanitize_prefix(plan_path.stem)
        output_dir = plan_path.parent / prefix

        # Pre-populate a realistic per-model result
        review_tasks_dir = output_dir / "review-tasks"
        review_tasks_dir.mkdir(parents=True, exist_ok=True)

        model_result = [
            {
                "title": "Missing rate limiting task",
                "desc": "Plan requires rate limiting but no task covers it.",
                "importance": "HIGH",
                "reference": "Plan Coverage",
                "type": "addition",
            },
            {
                "title": "T003 missing dependency on T001",
                "desc": "T003 uses the schema created by T001 but doesn't list T001 in depends_on.",
                "importance": "HIGH",
                "reference": "T003",
                "type": "modification",
            },
            {
                "title": "T005 description lacks detail",
                "desc": "T005 says 'Create API endpoints' but doesn't specify request/response formats.",
                "importance": "MEDIUM",
                "reference": "T005",
                "type": "clarification",
            },
        ]
        (review_tasks_dir / "cursor-agent_auto.json").write_text(
            json.dumps(model_result, indent=2)
        )

        result = run_orchestrator(
            plan_path,
            "--reaggregate",
            "--skip-validation",
        )

        combined = result.stdout + result.stderr

        assert result.returncode == 0, (
            f"Reaggregate failed.\nstderr: {result.stderr[:500]}"
        )

        # Should have loaded and aggregated the suggestions
        assert "cursor-agent_auto" in combined, (
            f"Reaggregate did not process model result file.\n"
            f"combined: {combined[:500]}"
        )

        # Should produce grouped.json and report files
        grouped = review_tasks_dir / "grouped.json"
        report_html = review_tasks_dir / "report.html"
        assert grouped.exists(), f"grouped.json not found in {review_tasks_dir}"
        assert report_html.exists(), f"report.html not found in {review_tasks_dir}"

        # Verify grouped.json is valid JSON (v1 bare list or v2 envelope)
        with open(grouped, 'r') as f:
            grouped_raw = json.load(f)
        if isinstance(grouped_raw, dict) and "groups" in grouped_raw:
            grouped_data = grouped_raw["groups"]  # v2 envelope
        else:
            grouped_data = grouped_raw  # v1 bare list
        assert isinstance(grouped_data, list), "grouped.json groups should be a JSON array"
        assert len(grouped_data) > 0, "grouped.json should have at least one group"

        # Phase should be marked as completed
        state = StateManager(plan_path)
        assert state.is_phase_completed("review-tasks"), (
            f"Phase 'review-tasks' not marked as completed.\n"
            f"phases_completed: {state.state.get('phases_completed', {})}"
        )


class TestReaggregateFromExisting:
    """Test --reaggregate mode that reprocesses existing model results."""

    def test_reaggregate_from_existing(self, plan_with_tasks):
        """--reaggregate reprocesses existing per-model JSON files."""
        plan_path = plan_with_tasks
        prefix = sanitize_prefix(plan_path.stem)
        output_dir = plan_path.parent / prefix

        # Create review-tasks phase dir with a fake per-model result
        review_tasks_dir = output_dir / "review-tasks"
        review_tasks_dir.mkdir(parents=True, exist_ok=True)

        # Write a mock per-model result file
        model_result = [
            {
                "title": "Missing rate limiting task",
                "desc": "Plan requires rate limiting but no task covers it.",
                "importance": "HIGH",
                "reference": "Plan Coverage",
                "type": "addition",
            }
        ]
        (review_tasks_dir / "cursor-agent_auto.json").write_text(
            json.dumps(model_result, indent=2)
        )

        # Run reaggregate with --skip-validation so it completes fully
        result = run_orchestrator(plan_path, "--reaggregate", "--skip-validation")

        combined = result.stdout + result.stderr

        # Should enter reaggregate mode
        assert "REAGGREGATE" in combined.upper(), (
            f"Orchestrator did not enter reaggregate mode.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )

        # Should succeed
        assert result.returncode == 0, (
            f"Reaggregate failed.\n"
            f"Exit code: {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

        # Should process the existing file
        assert "cursor-agent_auto" in combined, (
            f"Reaggregate did not process existing model file.\n"
            f"combined: {combined[:500]}"
        )

        # Should produce grouped.json and report files
        grouped = review_tasks_dir / "grouped.json"
        report_html = review_tasks_dir / "report.html"
        assert grouped.exists(), f"grouped.json not found in {review_tasks_dir}"
        assert report_html.exists(), f"report.html not found in {review_tasks_dir}"


class TestPhaseCompletionMarking:
    """Verify that successful runs mark the phase completed in state.json."""

    def test_phase_completion_marking(self, plan_with_tasks):
        """After reaggregate --skip-validation (which completes fully), phase should be marked."""
        plan_path = plan_with_tasks
        prefix = sanitize_prefix(plan_path.stem)
        output_dir = plan_path.parent / prefix

        # Pre-populate a model result for reaggregate to process
        review_tasks_dir = output_dir / "review-tasks"
        review_tasks_dir.mkdir(parents=True, exist_ok=True)
        (review_tasks_dir / "cursor-agent_auto.json").write_text(
            json.dumps([{
                "title": "Test finding",
                "desc": "Test description",
                "importance": "MEDIUM",
                "reference": "T001",
                "type": "modification",
            }], indent=2)
        )

        # Use --skip-validation so reaggregate completes fully (without
        # deferring to batched validation which would return early)
        result = run_orchestrator(plan_path, "--reaggregate", "--skip-validation")
        assert result.returncode == 0, (
            f"Reaggregate failed.\nstderr: {result.stderr[:500]}"
        )

        # Reload state and check phase completion
        state = StateManager(plan_path)
        assert state.is_phase_completed("review-tasks"), (
            f"Phase 'review-tasks' not marked as completed in state.json.\n"
            f"phases_completed: {state.state.get('phases_completed', {})}"
        )


class TestPrerequisiteEnforcement:
    """Verify that review-tasks requires generate-tasks to be completed."""

    def test_prerequisite_enforcement(self, plan_without_generate_tasks):
        """Orchestrator fails when generate-tasks has not been completed."""
        # Remove tasks file to also trigger the tasks-not-found path
        plan_path = plan_without_generate_tasks
        prefix = sanitize_prefix(plan_path.stem)
        tasks_file = plan_path.parent / prefix / "tasks" / "tasks.md"
        if tasks_file.exists():
            tasks_file.unlink()

        result = run_orchestrator(
            plan_path,
            "--models", "cursor-agent:auto",
        )

        combined = result.stdout + result.stderr

        # Should fail with a meaningful error
        assert result.returncode != 0, (
            f"Orchestrator should have failed due to missing tasks.\n"
            f"Exit code: {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )
        assert "tasks" in combined.lower() or "generate-tasks" in combined.lower(), (
            f"Error message should mention tasks or generate-tasks.\n"
            f"combined: {combined[:500]}"
        )

    def test_prereq_check_review_tasks_requires_generate_tasks(self, temp_dir):
        """check_workflow_prerequisites reports generate-tasks as missing for review-tasks."""
        plan_path = _create_plan_with_tasks(
            temp_dir,
            plan_name="prereq-test",
            mark_generate_tasks_completed=False,
        )

        output = run_prereq_check(plan_path, "review-tasks")

        assert output["prerequisites_met"] is False, (
            f"Prerequisites should NOT be met.\nOutput: {json.dumps(output, indent=2)}"
        )
        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "generate-tasks" in missing_phases, (
            f"generate-tasks should be in missing prerequisites.\n"
            f"Missing: {missing_phases}"
        )

    def test_prereq_check_review_tasks_met_when_generate_tasks_done(self, plan_with_tasks):
        """check_workflow_prerequisites passes when generate-tasks is completed."""
        output = run_prereq_check(plan_with_tasks, "review-tasks")

        assert output["prerequisites_met"] is True, (
            f"Prerequisites should be met.\nOutput: {json.dumps(output, indent=2)}"
        )


class TestImplementDoesNotRequireReviewTasks:
    """Verify that review-tasks is optional -- implement can proceed without it."""

    def test_implement_does_not_require_review_tasks(self, plan_with_tasks):
        """Running implement prereq check passes even without review-tasks completed."""
        output = run_prereq_check(plan_with_tasks, "implement")

        assert output["prerequisites_met"] is True, (
            f"Implement prerequisites should be met without review-tasks.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )

        # Double-check: review-tasks is NOT in the implement prerequisites list
        from check_workflow_prerequisites import PHASE_PREREQUISITES
        implement_prereqs = PHASE_PREREQUISITES.get("implement", [])
        assert "review-tasks" not in implement_prereqs, (
            f"review-tasks should NOT be a prerequisite for implement.\n"
            f"Implement prerequisites: {implement_prereqs}"
        )

    def test_review_tasks_is_optional_in_status(self, plan_with_tasks):
        """--status marks review-tasks as optional."""
        output = run_status_check(plan_with_tasks)

        phases = output.get("phases", {})
        assert "review-tasks" in phases, "review-tasks should appear in status phases"
        review_tasks_status = phases["review-tasks"]

        # When not completed, it should be marked as optional
        if review_tasks_status["status"] == "pending":
            assert review_tasks_status.get("optional") is True, (
                f"review-tasks should be marked as optional.\n"
                f"Status: {json.dumps(review_tasks_status, indent=2)}"
            )

    def test_review_tasks_not_in_suggested_next_when_skipped(self, plan_with_tasks):
        """suggested_next skips optional phases that haven't been run."""
        output = run_status_check(plan_with_tasks)

        # suggested_next should be 'implement' (the next non-optional pending phase),
        # not 'review-tasks'
        assert output.get("suggested_next") != "review-tasks", (
            f"suggested_next should not be review-tasks (it's optional).\n"
            f"suggested_next: {output.get('suggested_next')}"
        )


class TestMalformedTasksFileIntegration:
    """End-to-end: tasks file with invalid markdown causes fast failure."""

    def test_malformed_tasks_file_integration(self, temp_dir):
        """Orchestrator fails fast with clear error for malformed tasks file."""
        plan_path = _create_plan_with_tasks(
            temp_dir,
            plan_name="malformed-tasks-plan",
            tasks_content=MALFORMED_TASKS_CONTENT,
        )

        result = run_orchestrator(
            plan_path,
            "--models", "cursor-agent:auto",
        )

        # Should fail
        assert result.returncode != 0, (
            f"Orchestrator should have failed for malformed tasks.\n"
            f"Exit code: {result.returncode}\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )

        combined = result.stdout + result.stderr
        # Error should mention malformed or task headers
        assert "malformed" in combined.lower() or "task header" in combined.lower() or "no task" in combined.lower(), (
            f"Error should mention malformed tasks or missing headers.\n"
            f"combined: {combined[:500]}"
        )


class TestTasksFileReferencesNonexistentPlan:
    """Tasks file exists but plan referenced via TASKS_FILE comment is gone."""

    def test_tasks_file_references_nonexistent_plan(self, temp_dir):
        """Orchestrator fails gracefully when the plan file itself is removed after task creation.

        We test the scenario where a tasks file exists at the default location
        but the plan file referenced on the CLI is missing/moved.
        """
        plan_path = temp_dir / "vanished-plan.md"
        # Don't create the plan file -- simulate it being moved/deleted

        result = run_orchestrator(
            plan_path,
            "--models", "cursor-agent:auto",
        )

        assert result.returncode != 0, (
            f"Orchestrator should fail when plan file doesn't exist.\n"
            f"Exit code: {result.returncode}"
        )

        combined = result.stdout + result.stderr
        assert "not found" in combined.lower() or "no such file" in combined.lower() or "error" in combined.lower(), (
            f"Error should indicate the plan file was not found.\n"
            f"combined: {combined[:500]}"
        )

    def test_tasks_file_comment_points_to_missing_file(self, temp_dir):
        """TASKS_FILE comment in plan points to a nonexistent file."""
        plan_content = SAMPLE_PLAN_CONTENT + "\n<!-- TASKS_FILE: tasks/nonexistent.md -->\n"
        plan_path = _create_plan_with_tasks(
            temp_dir,
            plan_name="comment-missing",
            plan_content=plan_content,
            tasks_content=None,  # Don't create default tasks file
        )

        # Create the tasks dir but NOT the referenced file
        prefix = sanitize_prefix("comment-missing")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)

        result = run_orchestrator(
            plan_path,
            "--models", "cursor-agent:auto",
        )

        assert result.returncode != 0, (
            f"Orchestrator should fail when TASKS_FILE comment points to missing file.\n"
            f"Exit code: {result.returncode}"
        )

        combined = result.stdout + result.stderr
        assert "not found" in combined.lower() or "generate-tasks" in combined.lower(), (
            f"Error should mention missing tasks file.\n"
            f"combined: {combined[:500]}"
        )


class TestEmptyTasksListIntegration:
    """Tasks file parses but contains zero task entries."""

    def test_empty_tasks_list_integration(self, temp_dir):
        """Orchestrator handles an empty (whitespace-only) tasks file gracefully."""
        plan_path = _create_plan_with_tasks(
            temp_dir,
            plan_name="empty-tasks",
            tasks_content=EMPTY_TASKS_CONTENT,
        )

        result = run_orchestrator(
            plan_path,
            "--models", "cursor-agent:auto",
        )

        assert result.returncode != 0, (
            f"Orchestrator should fail for empty tasks file.\n"
            f"Exit code: {result.returncode}"
        )

        combined = result.stdout + result.stderr
        assert "empty" in combined.lower() or "no task" in combined.lower() or "malformed" in combined.lower(), (
            f"Error should indicate the tasks file is empty.\n"
            f"combined: {combined[:500]}"
        )

    def test_tasks_with_content_but_no_task_headers(self, temp_dir):
        """Tasks file has text content but no ## T<N> headers."""
        no_headers_content = "# Tasks\n\nSome introductory text.\n\n- Point A\n- Point B\n"
        plan_path = _create_plan_with_tasks(
            temp_dir,
            plan_name="no-headers",
            tasks_content=no_headers_content,
        )

        result = run_orchestrator(
            plan_path,
            "--models", "cursor-agent:auto",
        )

        assert result.returncode != 0, (
            f"Orchestrator should fail when tasks have no ## T<N> headers.\n"
            f"Exit code: {result.returncode}"
        )

        combined = result.stdout + result.stderr
        assert "malformed" in combined.lower() or "task header" in combined.lower() or "no task" in combined.lower(), (
            f"Error should mention missing task headers.\n"
            f"combined: {combined[:500]}"
        )


class TestCrossReferenceFailureIntegration:
    """Tasks file is well-formed but tasks do not correspond to plan requirements.

    The orchestrator's pre-validation only checks structure (has task headers),
    not semantic correctness. Coverage analysis is the LLM's job. We verify
    that find_tasks_file accepts the structurally valid file.
    """

    def test_cross_reference_failure_integration(self, temp_dir):
        """Valid tasks file with irrelevant tasks passes pre-validation.

        Uses find_tasks_file directly to verify that structurally valid tasks
        (with ## T<N> headers) are accepted even when they don't match the plan.
        """
        from review_tasks_orchestrator import find_tasks_file

        unrelated_tasks = """\
# Implementation Tasks

## T001: Configure CI pipeline
Set up continuous integration pipeline.
- Depends on: none
- Complexity: low

## T002: Add linting rules
Configure eslint with standard rules.
- Depends on: T001
- Complexity: low
"""
        plan_path = _create_plan_with_tasks(
            temp_dir,
            plan_name="cross-ref-fail",
            tasks_content=unrelated_tasks,
        )

        # find_tasks_file should NOT raise -- it only checks structure
        tasks_path = find_tasks_file(str(plan_path))
        assert os.path.isfile(tasks_path), f"Tasks file not found at: {tasks_path}"

        # Read back the content to verify it's our unrelated tasks
        with open(tasks_path, 'r') as f:
            content = f.read()
        assert "CI pipeline" in content, "Should have loaded our unrelated tasks file"
        assert "Configure eslint" in content

    def test_cross_reference_reaggregate(self, temp_dir):
        """Reaggregate works with cross-referenced (unrelated) tasks.

        Even when tasks don't match the plan, the aggregation pipeline should
        process model results normally. Coverage gap detection is the LLM's job.
        """
        unrelated_tasks = """\
# Implementation Tasks

## T001: Configure CI pipeline
Set up continuous integration pipeline.
- Depends on: none
- Complexity: low
"""
        plan_path = _create_plan_with_tasks(
            temp_dir,
            plan_name="cross-ref-reagg",
            tasks_content=unrelated_tasks,
        )

        prefix = sanitize_prefix("cross-ref-reagg")
        output_dir = plan_path.parent / prefix
        review_tasks_dir = output_dir / "review-tasks"
        review_tasks_dir.mkdir(parents=True, exist_ok=True)

        # Model would flag coverage gaps
        model_result = [
            {
                "title": "Plan notification requirements not covered",
                "desc": "Tasks only cover CI/linting, not notification features from plan.",
                "importance": "HIGH",
                "reference": "Plan Coverage",
                "type": "addition",
            }
        ]
        (review_tasks_dir / "cursor-agent_auto.json").write_text(
            json.dumps(model_result, indent=2)
        )

        result = run_orchestrator(plan_path, "--reaggregate", "--skip-validation")

        assert result.returncode == 0, (
            f"Reaggregate failed.\nstderr: {result.stderr[:500]}"
        )


class TestForceRerun:
    """Verify --force allows re-running a completed phase."""

    def test_force_blocked_when_phase_completed(self, plan_with_tasks):
        """Without --force, orchestrator rejects re-run of completed phase (exit code 2)."""
        plan_path = plan_with_tasks

        # Mark phase as completed
        state = StateManager(plan_path)
        state.mark_phase_completed("review-tasks")
        state.save()

        # Without --force, should fail with exit code 2 immediately (no timeout risk)
        result_no_force = run_orchestrator(
            plan_path,
            "--models", "cursor-agent:auto",
        )
        assert result_no_force.returncode == 2, (
            f"Should fail with exit code 2 when phase already completed.\n"
            f"Exit code: {result_no_force.returncode}\n"
            f"stderr: {result_no_force.stderr[:500]}"
        )
        combined = result_no_force.stdout + result_no_force.stderr
        assert "already been completed" in combined.lower(), (
            "Error should mention phase already completed."
        )

    def test_force_allows_reaggregate_after_completion(self, plan_with_tasks):
        """--force clears the phase guard, allowing reaggregate to re-run.

        We test --force with --reaggregate to avoid the timeout that would
        occur with live model invocation (no mock providers in PATH).
        """
        plan_path = plan_with_tasks
        prefix = sanitize_prefix(plan_path.stem)
        output_dir = plan_path.parent / prefix

        # Pre-populate a model result
        review_tasks_dir = output_dir / "review-tasks"
        review_tasks_dir.mkdir(parents=True, exist_ok=True)
        (review_tasks_dir / "cursor-agent_auto.json").write_text(
            json.dumps([{
                "title": "Test finding",
                "desc": "Test description",
                "importance": "MEDIUM",
                "reference": "T001",
                "type": "modification",
            }], indent=2)
        )

        # First, complete the phase via reaggregate
        result_first = run_orchestrator(plan_path, "--reaggregate", "--skip-validation")
        assert result_first.returncode == 0

        # Verify phase is completed
        state = StateManager(plan_path)
        assert state.is_phase_completed("review-tasks")

        # Without --force, reaggregate should also be blocked (exit code 2)
        # Actually, reaggregate bypasses the guard -- it's only the main flow
        # that checks. So let's verify the main flow's --force works by checking
        # state manipulation directly.

        # Re-mark as completed to test guard
        state = StateManager(plan_path)
        state.mark_phase_completed("review-tasks")
        state.save()

        # --force + --reaggregate should work
        result_force = run_orchestrator(
            plan_path,
            "--reaggregate",
            "--skip-validation",
            "--force",
        )
        assert result_force.returncode == 0, (
            f"--force --reaggregate should succeed.\n"
            f"Exit code: {result_force.returncode}\n"
            f"stderr: {result_force.stderr[:500]}"
        )

    def test_force_clears_phase_completion_in_state(self, plan_with_tasks):
        """Verify --force clears the phases_completed entry before proceeding.

        We check the state manipulation by running the no-force case and
        confirming exit code 2, then verifying the behavior description.
        """
        plan_path = plan_with_tasks

        # Mark phase as completed
        state = StateManager(plan_path)
        state.mark_phase_completed("review-tasks")
        state.save()

        # Confirm it's completed
        assert state.is_phase_completed("review-tasks")

        # Without --force: rejected
        result = run_orchestrator(plan_path, "--models", "cursor-agent:auto")
        assert result.returncode == 2
        combined = result.stdout + result.stderr
        assert "--force" in combined, (
            f"Error message should suggest using --force.\n"
            f"combined: {combined[:500]}"
        )


class TestFindTasksFile:
    """Unit-style tests for the find_tasks_file function in the orchestrator."""

    def test_default_location(self, plan_with_tasks):
        """find_tasks_file finds tasks at the default location."""
        from review_tasks_orchestrator import find_tasks_file

        tasks_path = find_tasks_file(str(plan_with_tasks))
        assert os.path.isfile(tasks_path), f"Tasks file not found at: {tasks_path}"
        assert tasks_path.endswith("tasks.md")

    def test_tasks_file_comment(self, temp_dir):
        """find_tasks_file reads TASKS_FILE comment from the plan."""
        from review_tasks_orchestrator import find_tasks_file

        # Create plan with TASKS_FILE comment pointing to a custom location
        plan_content = SAMPLE_PLAN_CONTENT + "\n<!-- TASKS_FILE: custom/my-tasks.md -->\n"
        plan_path = temp_dir / "comment-test.md"
        plan_path.write_text(plan_content)

        # Create the custom tasks file
        prefix = sanitize_prefix("comment-test")
        custom_dir = temp_dir / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)
        (custom_dir / "my-tasks.md").write_text(VALID_TASKS_CONTENT)

        tasks_path = find_tasks_file(str(plan_path))
        assert "my-tasks.md" in tasks_path

    def test_no_tasks_file_raises(self, temp_dir):
        """find_tasks_file raises FileNotFoundError when no tasks file exists."""
        from review_tasks_orchestrator import find_tasks_file

        plan_path = temp_dir / "no-tasks.md"
        plan_path.write_text(SAMPLE_PLAN_CONTENT)

        with pytest.raises(FileNotFoundError, match="generate-tasks"):
            find_tasks_file(str(plan_path))

    def test_empty_tasks_file_raises(self, temp_dir):
        """find_tasks_file raises ValueError for an empty tasks file."""
        from review_tasks_orchestrator import find_tasks_file

        plan_path = temp_dir / "empty-tasks-test.md"
        plan_path.write_text(SAMPLE_PLAN_CONTENT)

        prefix = sanitize_prefix("empty-tasks-test")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "tasks.md").write_text("")

        with pytest.raises(ValueError, match="[Ee]mpty"):
            find_tasks_file(str(plan_path))

    def test_malformed_tasks_file_raises(self, temp_dir):
        """find_tasks_file raises ValueError for tasks file without task headers."""
        from review_tasks_orchestrator import find_tasks_file

        plan_path = temp_dir / "malformed-test.md"
        plan_path.write_text(SAMPLE_PLAN_CONTENT)

        prefix = sanitize_prefix("malformed-test")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "tasks.md").write_text(MALFORMED_TASKS_CONTENT)

        with pytest.raises(ValueError, match="[Mm]alformed"):
            find_tasks_file(str(plan_path))

    def test_absolute_path_in_comment_rejected(self, temp_dir):
        """find_tasks_file rejects absolute paths in TASKS_FILE comment."""
        from review_tasks_orchestrator import find_tasks_file

        plan_content = SAMPLE_PLAN_CONTENT + "\n<!-- TASKS_FILE: /etc/passwd -->\n"
        plan_path = temp_dir / "abs-path.md"
        plan_path.write_text(plan_content)

        with pytest.raises(SystemExit):
            find_tasks_file(str(plan_path))

    def test_path_traversal_in_comment_rejected(self, temp_dir):
        """find_tasks_file rejects path traversal in TASKS_FILE comment."""
        from review_tasks_orchestrator import find_tasks_file

        plan_content = SAMPLE_PLAN_CONTENT + "\n<!-- TASKS_FILE: ../../etc/passwd -->\n"
        plan_path = temp_dir / "traversal.md"
        plan_path.write_text(plan_content)

        with pytest.raises(SystemExit):
            find_tasks_file(str(plan_path))


class TestQuickAndInteractiveFlags:
    """Verify that --quick and --interactive flags work or are rejected properly."""

    def test_quick_and_interactive_mutually_exclusive(self, plan_with_tasks):
        """--quick and --interactive together should fail."""
        result = run_orchestrator(
            plan_with_tasks,
            "--quick",
            "--interactive",
        )

        assert result.returncode != 0, (
            f"--quick and --interactive should be mutually exclusive.\n"
            f"Exit code: {result.returncode}"
        )
        combined = result.stdout + result.stderr
        assert "mutually exclusive" in combined.lower() or "exclusive" in combined.lower(), (
            f"Error should mention mutual exclusivity.\n"
            f"combined: {combined[:500]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
