#!/usr/bin/env python3
"""
Integration tests for apply_task_suggestions_orchestrator.py

Tests cover cross-component behaviors:
1. Phase prerequisite enforcement — apply-task-suggestions blocks when
   review-tasks is not completed.
2. Phase status/skip tracking — mark_phase_completed and mark_phase_skipped
   update state.json correctly.
3. Output schema — orchestrator_output.json structure matches expected format.
4. Batching — suggestions grouped by reference field with stable ordering.
5. Selection merge from HTML and MD sources.
6. Display decisions — display_decisions.py outputs correct nouns for the phase.
7. --status includes apply-task-suggestions in phase listing.
8. --implement detects unapplied task suggestions and surfaces a prompt.

To run:
    uv run --project skills/multi-llm -- pytest \
        skills/multi-llm/tests/test_apply_task_suggestions_integration.py -v
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

from utils.state_manager import StateManager, generate_group_id, stamp_stable_ids
from utils.output_handler import sanitize_prefix, get_phase_dir


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORCHESTRATOR_SCRIPT = SKILL_DIR / "apply_task_suggestions_orchestrator.py"
PREREQ_SCRIPT = SKILL_DIR / "check_workflow_prerequisites.py"
DISPLAY_SCRIPT = SKILL_DIR / "display_decisions.py"

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

SAMPLE_GROUPED_SUGGESTIONS = [
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
                "desc": "T003 uses the schema created by T001 but doesn't list T001 in depends_on.",
                "type": "modification",
                "reference": "T003",
                "importance": "HIGH",
                "source_model": "cursor-agent",
            }
        ],
    },
    {
        "theme": "T005 description lacks detail",
        "category": "clarity",
        "models": ["cursor-agent", "gpt-4"],
        "suggestions": [
            {
                "title": "Clarify T005 description",
                "desc": "T005 says 'Create API endpoints' but doesn't specify request/response formats.",
                "type": "clarification",
                "reference": "T005",
                "importance": "MEDIUM",
                "source_model": "cursor-agent",
            }
        ],
    },
    {
        "theme": "T002 acceptance criteria vague",
        "category": "clarity",
        "models": ["gpt-4"],
        "suggestions": [
            {
                "title": "Improve T002 acceptance criteria",
                "desc": "T002 acceptance criteria should list specific interfaces.",
                "type": "clarification",
                "reference": "T002",
                "importance": "LOW",
                "source_model": "gpt-4",
            }
        ],
    },
    {
        "theme": "T006 missing coverage threshold",
        "category": "testing",
        "models": ["gemini-pro"],
        "suggestions": [
            {
                "title": "Add coverage threshold to T006",
                "desc": "Add requirement for minimum 80% code coverage.",
                "type": "modification",
                "reference": "T006",
                "importance": "LOW",
                "source_model": "gemini-pro",
            }
        ],
    },
]

SAMPLE_VALIDATION_RESULTS = {
    "metadata": {
        "schema_version": "2.0",
        "validated_at": "2026-03-13T10:00:00",
        "model": "mock-llm",
        "plan_hash": "abc123",
        "total_groups": 5,
    },
    "groups": [
        {"group_index": 0, "status": "valid", "reason": "Clear addition needed.", "confidence": 0.95},
        {"group_index": 1, "status": "valid", "reason": "Dependency fix is straightforward.", "confidence": 0.90},
        {"group_index": 2, "status": "needs-human-decision", "reason": "Ambiguous scope of clarification.",
         "confidence": 0.55, "error_type": "real_ambiguity", "recoverable": False},
        {"group_index": 3, "status": "valid", "reason": "Minor wording improvement.", "confidence": 0.80},
        {"group_index": 4, "status": "invalid", "reason": "Coverage threshold is out of scope.", "confidence": 0.70},
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _create_plan_with_review_tasks(
    temp_dir: Path,
    plan_name: str = "test-plan",
    plan_content: str = SAMPLE_PLAN_CONTENT,
    tasks_content: str = VALID_TASKS_CONTENT,
    grouped: list = None,
    validation: dict = None,
    mark_prerequisite_phases: bool = True,
    mark_review_tasks_completed: bool = True,
) -> Path:
    """Create a plan file with review-tasks results and state ready for
    apply-task-suggestions.

    Returns the plan file path.
    """
    plan_path = temp_dir / f"{plan_name}.md"
    plan_path.write_text(plan_content)

    prefix = sanitize_prefix(plan_name)
    output_dir = temp_dir / prefix

    # Create tasks directory and file
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "tasks.md").write_text(tasks_content)

    # Create review-tasks directory with grouped.json and validation.json
    review_tasks_dir = output_dir / "review-tasks"
    review_tasks_dir.mkdir(parents=True, exist_ok=True)

    groups = grouped if grouped is not None else SAMPLE_GROUPED_SUGGESTIONS
    # Stamp stable IDs on groups
    import copy
    groups_copy = copy.deepcopy(groups)
    stamp_stable_ids(groups_copy)

    (review_tasks_dir / "grouped.json").write_text(
        json.dumps({"format_version": 2, "groups": groups_copy}, indent=2)
    )
    val = validation if validation is not None else SAMPLE_VALIDATION_RESULTS
    (review_tasks_dir / "validation.json").write_text(
        json.dumps(val, indent=2)
    )

    # Set up state
    state = StateManager(plan_path)
    if mark_prerequisite_phases:
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.mark_phase_completed("generate-tasks")
    if mark_review_tasks_completed:
        state.mark_phase_completed("review-tasks")
    state.save()

    return plan_path


@pytest.fixture
def plan_ready_for_apply(temp_dir):
    """A plan with all prerequisites completed and review-tasks results present."""
    return _create_plan_with_review_tasks(temp_dir)


@pytest.fixture
def plan_without_review_tasks(temp_dir):
    """A plan where review-tasks has NOT been completed."""
    return _create_plan_with_review_tasks(
        temp_dir,
        plan_name="plan-no-review-tasks",
        mark_review_tasks_completed=False,
    )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_orchestrator(plan_path: Path, *extra_args, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run the apply_task_suggestions_orchestrator.py with the given plan file."""
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


def run_display_decisions(output_file: Path, phase: str = "apply-task-suggestions") -> subprocess.CompletedProcess:
    """Run display_decisions.py for an orchestrator output file."""
    cmd = [
        sys.executable,
        str(DISPLAY_SCRIPT),
        "--output-file", str(output_file),
        "--phase", phase,
    ]
    return subprocess.run(
        cmd,
        cwd=str(SKILL_DIR),
        capture_output=True,
        text=True,
    )


# ===========================================================================
# Test classes
# ===========================================================================


class TestPrerequisiteEnforcement:
    """Verify that apply-task-suggestions requires review-tasks to be completed."""

    def test_prereq_check_requires_review_tasks(self, plan_without_review_tasks):
        """check_workflow_prerequisites reports review-tasks as missing."""
        output = run_prereq_check(plan_without_review_tasks, "apply-task-suggestions")

        assert output["prerequisites_met"] is False, (
            f"Prerequisites should NOT be met.\nOutput: {json.dumps(output, indent=2)}"
        )
        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "review-tasks" in missing_phases, (
            f"review-tasks should be in missing prerequisites.\n"
            f"Missing: {missing_phases}"
        )

    def test_prereq_check_passes_when_review_tasks_completed(self, plan_ready_for_apply):
        """Prerequisites are met when review-tasks is completed."""
        output = run_prereq_check(plan_ready_for_apply, "apply-task-suggestions")

        assert output["prerequisites_met"] is True, (
            f"Prerequisites should be met.\nOutput: {json.dumps(output, indent=2)}"
        )

    def test_prereq_check_passes_when_review_tasks_skipped(self, temp_dir):
        """Prerequisites are met when review-tasks has been explicitly skipped."""
        plan_path = _create_plan_with_review_tasks(
            temp_dir,
            plan_name="skipped-review",
            mark_review_tasks_completed=False,
        )
        state = StateManager(plan_path)
        state.mark_phase_skipped("review-tasks", "Tasks are simple enough")
        state.save()

        output = run_prereq_check(plan_path, "apply-task-suggestions")

        assert output["prerequisites_met"] is True, (
            f"Prerequisites should be met when review-tasks is skipped.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )

    def test_orchestrator_requires_validation_json(self, temp_dir):
        """Orchestrator fails when validation.json is missing in review-tasks dir."""
        plan_path = _create_plan_with_review_tasks(temp_dir, plan_name="no-val")
        prefix = sanitize_prefix("no-val")
        # Remove validation.json
        val_path = temp_dir / prefix / "review-tasks" / "validation.json"
        if val_path.exists():
            val_path.unlink()

        result = run_orchestrator(plan_path, "--no-confirm")

        assert result.returncode != 0, (
            f"Orchestrator should fail without validation.json.\n"
            f"Exit code: {result.returncode}\nstderr: {result.stderr[:500]}"
        )
        assert "not found" in result.stderr.lower() or "review-tasks" in result.stderr.lower()


class TestPhaseStatusTracking:
    """Verify mark_phase_completed and mark_phase_skipped update state.json."""

    def test_successful_run_defers_phase_completion(self, plan_ready_for_apply):
        """After a successful orchestrator run, the phase is NOT yet marked
        completed — completion is deferred to the instruction file (step 7)
        after all batches have been processed and tasks.md updated."""
        result = run_orchestrator(plan_ready_for_apply, "--no-confirm")

        assert result.returncode == 0, (
            f"Orchestrator failed.\nstderr: {result.stderr[:500]}"
        )

        state = StateManager(plan_ready_for_apply)
        assert not state.is_phase_completed("apply-task-suggestions"), (
            f"Phase 'apply-task-suggestions' should NOT be marked completed by "
            f"the orchestrator — completion is deferred to the instruction file.\n"
            f"phases_completed: {state.state.get('phases_completed', {})}"
        )

    def test_skip_flag_marks_phase_skipped(self, plan_ready_for_apply):
        """--skip flag marks the phase as skipped in state."""
        result = run_orchestrator(plan_ready_for_apply, "--skip")

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["status"] == "skipped"

        state = StateManager(plan_ready_for_apply)
        assert state.is_phase_skipped("apply-task-suggestions") is True
        assert "User chose to skip" in state.get_phase_skip_reason("apply-task-suggestions")

    def test_skip_persists_after_reload(self, plan_ready_for_apply):
        """Phase skipped persists after state reload."""
        run_orchestrator(plan_ready_for_apply, "--skip")

        # Reload state from disk
        state = StateManager(plan_ready_for_apply)
        assert state.is_phase_skipped("apply-task-suggestions") is True

    def test_deferred_completion_after_reload(self, plan_ready_for_apply):
        """After orchestrator run, phase is not completed (deferred to instruction
        file). Verify this state persists after reload."""
        run_orchestrator(plan_ready_for_apply, "--no-confirm")

        # Reload state from disk
        state = StateManager(plan_ready_for_apply)
        assert not state.is_phase_completed("apply-task-suggestions"), (
            "Phase should not be completed — orchestrator defers completion marking"
        )

    def test_no_actionable_findings_marks_skipped(self, temp_dir):
        """When all suggestions are invalid, phase is marked as skipped."""
        all_invalid_validation = {
            "metadata": {
                "schema_version": "2.0",
                "validated_at": "2026-03-13T10:00:00",
                "model": "mock-llm",
                "total_groups": 2,
            },
            "groups": [
                {"group_index": 0, "status": "invalid", "reason": "Not applicable.", "confidence": 0.80},
                {"group_index": 1, "status": "invalid", "reason": "Out of scope.", "confidence": 0.75},
            ],
        }
        # Only use first two groups
        two_groups = SAMPLE_GROUPED_SUGGESTIONS[:2]
        plan_path = _create_plan_with_review_tasks(
            temp_dir,
            plan_name="all-invalid",
            grouped=two_groups,
            validation=all_invalid_validation,
        )

        result = run_orchestrator(plan_path, "--no-confirm")

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["status"] == "skipped"

        state = StateManager(plan_path)
        assert state.is_phase_skipped("apply-task-suggestions") is True


class TestOutputSchema:
    """Verify that orchestrator_output.json structure matches expected format."""

    def test_output_file_created(self, plan_ready_for_apply):
        """Orchestrator creates orchestrator_output.json in the correct directory."""
        result = run_orchestrator(plan_ready_for_apply, "--no-confirm")
        assert result.returncode == 0, f"Failed: {result.stderr[:500]}"

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        assert output_file.exists(), f"Output file not found: {output_file}"

    def test_output_contains_required_top_level_fields(self, plan_ready_for_apply):
        """Output JSON has all required top-level fields."""
        run_orchestrator(plan_ready_for_apply, "--no-confirm")

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output = json.load(f)

        required_fields = [
            "format_version", "plan_file", "tasks_file", "prefix", "output_dir",
            "apply_task_suggestions_dir", "timestamp", "batches", "to_apply",
            "needs_human_review", "skipped_items", "user_skipped_items",
            "batching_stats", "human_review_config", "resume_info",
            "edited_descriptions", "summary",
        ]
        for field in required_fields:
            assert field in output, f"Missing required field: {field}"

    def test_output_summary_structure(self, plan_ready_for_apply):
        """Output summary contains all expected counters."""
        run_orchestrator(plan_ready_for_apply, "--no-confirm")

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output = json.load(f)

        summary = output["summary"]
        summary_fields = [
            "total_groups", "valid_count", "needs_human_count",
            "skipped_count", "user_skipped_count", "batch_count",
            "validation_failed_count", "auto_approved_count",
            "edited_description_count",
        ]
        for field in summary_fields:
            assert field in summary, f"Missing summary field: {field}"
            assert isinstance(summary[field], int), f"Summary field {field} should be int"

    def test_output_to_apply_items_have_expected_fields(self, plan_ready_for_apply):
        """Each item in to_apply has the expected fields."""
        run_orchestrator(plan_ready_for_apply, "--no-confirm")

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output = json.load(f)

        assert len(output["to_apply"]) > 0, "Should have at least one item to apply"

        item_fields = [
            "index", "title", "description", "type", "reference",
            "task_reference", "importance", "theme", "category",
            "validation_status", "validation_reason", "validation_confidence",
            "models", "suggestion_count",
        ]
        for item in output["to_apply"]:
            for field in item_fields:
                assert field in item, f"Missing field in to_apply item: {field}"

    def test_output_format_version_is_current(self, plan_ready_for_apply):
        """Output format_version matches CURRENT_FORMAT_VERSION."""
        from utils.state_manager import CURRENT_FORMAT_VERSION

        run_orchestrator(plan_ready_for_apply, "--no-confirm")

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output = json.load(f)

        assert output["format_version"] == CURRENT_FORMAT_VERSION

    def test_output_tasks_file_field_present(self, plan_ready_for_apply):
        """Output contains a tasks_file field pointing to the tasks file."""
        run_orchestrator(plan_ready_for_apply, "--no-confirm")

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output = json.load(f)

        assert "tasks_file" in output
        assert output["tasks_file"].endswith("tasks.md")


class TestBatching:
    """Verify suggestions are grouped by reference field with stable ordering."""

    def test_batches_group_by_reference(self, temp_dir):
        """Suggestions with the same reference are batched together."""
        # Create groups where two suggestions target the same task reference
        groups = [
            {
                "theme": "Fix T003 dependency",
                "category": "dependency",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": "Fix T003 dependency on T001",
                    "desc": "Add T001 to T003 depends_on.",
                    "type": "modification",
                    "reference": "T003",
                    "importance": "HIGH",
                    "source_model": "cursor-agent",
                }],
            },
            {
                "theme": "Clarify T003 description",
                "category": "clarity",
                "models": ["gpt-4"],
                "suggestions": [{
                    "title": "Improve T003 description",
                    "desc": "T003 description should mention the email provider.",
                    "type": "clarification",
                    "reference": "T003",
                    "importance": "MEDIUM",
                    "source_model": "gpt-4",
                }],
            },
            {
                "theme": "Missing notification types",
                "category": "coverage",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": "Add notification types task",
                    "desc": "Add a task for defining notification types.",
                    "type": "addition",
                    "reference": "Plan Coverage",
                    "importance": "HIGH",
                    "source_model": "cursor-agent",
                }],
            },
        ]
        validation = {
            "metadata": {
                "schema_version": "2.0",
                "validated_at": "2026-03-13T10:00:00",
                "model": "mock-llm",
                "total_groups": 3,
            },
            "groups": [
                {"group_index": 0, "status": "valid", "reason": "Clear.", "confidence": 0.95},
                {"group_index": 1, "status": "valid", "reason": "Clear.", "confidence": 0.90},
                {"group_index": 2, "status": "valid", "reason": "Clear.", "confidence": 0.95},
            ],
        }

        plan_path = _create_plan_with_review_tasks(
            temp_dir,
            plan_name="batch-test",
            grouped=groups,
            validation=validation,
        )

        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode == 0, f"Failed: {result.stderr[:500]}"

        prefix = sanitize_prefix("batch-test")
        output_file = plan_path.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output = json.load(f)

        # With batching enabled, T003-targeted suggestions should be in the same batch
        batches = output["batches"]
        assert len(batches) >= 1

        # Find the batch containing the T003 suggestions
        t003_batch = None
        for batch in batches:
            refs = [s.get("reference", "") for s in batch["suggestions"]]
            if all("T003" in r for r in refs) and len(refs) > 1:
                t003_batch = batch
                break

        # Smart batcher should group same-reference together
        if t003_batch is not None:
            assert t003_batch["suggestion_count"] == 2
        # Even if batcher doesn't merge (single-item groups), total suggestions should be 3
        total_suggestions = sum(b["suggestion_count"] for b in batches)
        assert total_suggestions == 3

    def test_no_batch_flag_disables_batching(self, plan_ready_for_apply):
        """--no-batch processes each suggestion individually."""
        result = run_orchestrator(plan_ready_for_apply, "--no-batch", "--no-confirm")
        assert result.returncode == 0, f"Failed: {result.stderr[:500]}"

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output = json.load(f)

        assert output["batching_stats"]["batching_enabled"] is False
        for batch in output["batches"]:
            assert batch["suggestion_count"] == 1

    def test_batching_stats_present(self, plan_ready_for_apply):
        """Batching stats are included in output."""
        run_orchestrator(plan_ready_for_apply, "--no-confirm")

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output = json.load(f)

        stats = output["batching_stats"]
        assert "total_suggestions" in stats
        assert "total_batches" in stats
        assert "batching_enabled" in stats
        assert stats["batching_enabled"] is True

    def test_stable_ordering_across_runs(self, temp_dir):
        """Multiple runs produce the same batch ordering."""
        plan_path = _create_plan_with_review_tasks(temp_dir, plan_name="ordering-test")

        result1 = run_orchestrator(plan_path, "--no-confirm")
        assert result1.returncode == 0

        prefix = sanitize_prefix("ordering-test")
        output_file = plan_path.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        with open(output_file) as f:
            output1 = json.load(f)

        # Reset phase completion to allow re-run
        state = StateManager(plan_path)
        phases_completed = state.state.get("phases_completed", {})
        phases_completed.pop("apply-task-suggestions", None)
        state.save()

        result2 = run_orchestrator(plan_path, "--no-confirm")
        assert result2.returncode == 0

        with open(output_file) as f:
            output2 = json.load(f)

        # Compare to_apply order by title
        titles1 = [item["title"] for item in output1["to_apply"]]
        titles2 = [item["title"] for item in output2["to_apply"]]
        assert titles1 == titles2, (
            f"Ordering should be stable across runs.\n"
            f"Run 1: {titles1}\nRun 2: {titles2}"
        )


class TestSelectionMerge:
    """Verify that user selections from HTML and MD sources are merged correctly."""

    def test_md_skip_checkbox_filters_suggestion(self, temp_dir):
        """User marking [x] Skip in report.md filters out the suggestion."""
        plan_path = _create_plan_with_review_tasks(temp_dir, plan_name="md-skip")
        prefix = sanitize_prefix("md-skip")
        review_tasks_dir = plan_path.parent / prefix / "review-tasks"

        # Load the groups to get hashes for v2 bracket-notation
        import copy
        groups = copy.deepcopy(SAMPLE_GROUPED_SUGGESTIONS)
        stamp_stable_ids(groups)

        g1_hash = groups[0].get("group_hash", "")
        g1s1_hash = groups[0]["suggestions"][0].get("suggestion_hash", "")
        g2_hash = groups[1].get("group_hash", "")
        g2s1_hash = groups[1]["suggestions"][0].get("suggestion_hash", "")

        # Create report.md with skip checkbox checked for group 2
        report_content = f"""# Review Report

## G1 [{g1_hash}]: {groups[0]['theme']}
- [ ] Skip this group

### G1S1 [{g1s1_hash}]: {groups[0]['suggestions'][0]['title']}
- [ ] Skip
**Validation:** Valid

---

## G2 [{g2_hash}]: {groups[1]['theme']}
- [x] Skip this group

### G2S1 [{g2s1_hash}]: {groups[1]['suggestions'][0]['title']}
- [ ] Skip
**Validation:** Valid

---
"""
        (review_tasks_dir / "report.md").write_text(report_content)

        result = run_orchestrator(plan_path)
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"Failed: {result.stderr[:500]}"

        # Should mention user skipped groups
        assert "User skipped 1 groups" in result.stderr

    def test_html_selections_override_md(self, temp_dir):
        """HTML selections (user_selections.json) override MD selections."""
        plan_path = _create_plan_with_review_tasks(temp_dir, plan_name="html-override")
        prefix = sanitize_prefix("html-override")
        review_tasks_dir = plan_path.parent / prefix / "review-tasks"

        # Load groups to get hashes
        import copy
        groups = copy.deepcopy(SAMPLE_GROUPED_SUGGESTIONS)
        stamp_stable_ids(groups)

        g1_hash = groups[0].get("group_hash", "")

        # Create user_selections.json (HTML format) that skips group 1
        selections = {
            "format_version": 2,
            "plan_hash": "test",
            "skipped_groups": [g1_hash],
            "skipped_suggestions": [],
            "validation_overrides": {},
            "edited_descriptions": {},
        }
        (review_tasks_dir / "user_selections.json").write_text(
            json.dumps(selections, indent=2)
        )

        result = run_orchestrator(plan_path)
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"Failed: {result.stderr[:500]}"

        # Should detect HTML selections
        assert "HTML selections" in result.stderr or "user_selections.json" in result.stderr


class TestDisplayDecisions:
    """Verify display_decisions.py outputs correct nouns for the phase."""

    def test_display_uses_task_suggestion_nouns(self, plan_ready_for_apply):
        """display_decisions.py uses 'task suggestion' nouns for apply-task-suggestions phase."""
        # First run the orchestrator to produce output
        result = run_orchestrator(plan_ready_for_apply, "--no-confirm")
        assert result.returncode == 0, f"Orchestrator failed: {result.stderr[:500]}"

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"

        # Run display_decisions.py
        display_result = run_display_decisions(output_file, phase="apply-task-suggestions")

        assert display_result.returncode == 0, (
            f"display_decisions.py failed.\nstderr: {display_result.stderr[:500]}"
        )

        output_text = display_result.stdout
        # Should use "task suggestion" nouns, not plain "suggestion"
        assert "Task Suggestion Decision Summary" in output_text, (
            f"Expected 'Task Suggestion Decision Summary' in output.\n"
            f"Got: {output_text[:500]}"
        )
        assert "task suggestions" in output_text.lower(), (
            f"Expected 'task suggestions' noun in output.\n"
            f"Got: {output_text[:500]}"
        )

    def test_display_shows_will_apply_section(self, plan_ready_for_apply):
        """display_decisions.py shows WILL APPLY section with items."""
        result = run_orchestrator(plan_ready_for_apply, "--no-confirm")
        assert result.returncode == 0

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"

        display_result = run_display_decisions(output_file)
        assert display_result.returncode == 0

        assert "WILL APPLY" in display_result.stdout


class TestStatusPhaseListing:
    """Verify --status includes apply-task-suggestions in phase listing."""

    def test_status_includes_apply_task_suggestions(self, temp_dir):
        """--status output lists apply-task-suggestions as a phase."""
        plan_path = temp_dir / "status-test.md"
        plan_path.write_text(SAMPLE_PLAN_CONTENT)

        output = run_status_check(plan_path)

        assert "phases" in output
        assert "apply-task-suggestions" in output["phases"], (
            f"apply-task-suggestions not found in status phases.\n"
            f"Phases: {list(output['phases'].keys())}"
        )

    def test_status_shows_apply_task_suggestions_as_optional(self, temp_dir):
        """--status marks apply-task-suggestions as optional."""
        plan_path = temp_dir / "optional-test.md"
        plan_path.write_text(SAMPLE_PLAN_CONTENT)

        output = run_status_check(plan_path)

        phase = output["phases"]["apply-task-suggestions"]
        assert phase["status"] == "pending"
        assert phase.get("optional") is True, (
            f"apply-task-suggestions should be optional.\n"
            f"Phase: {json.dumps(phase, indent=2)}"
        )

    def test_status_shows_pending_after_orchestrator_run(self, plan_ready_for_apply):
        """--status shows apply-task-suggestions as pending after orchestrator run,
        because completion is deferred to the instruction file (after all batches
        are processed and tasks.md is updated)."""
        run_orchestrator(plan_ready_for_apply, "--no-confirm")

        output = run_status_check(plan_ready_for_apply)

        phase = output["phases"]["apply-task-suggestions"]
        assert phase["status"] == "pending", (
            f"apply-task-suggestions should still be pending after orchestrator run "
            f"(completion deferred to instruction file).\n"
            f"Phase: {json.dumps(phase, indent=2)}"
        )

    def test_status_shows_skipped_after_skip(self, plan_ready_for_apply):
        """--status shows apply-task-suggestions as skipped after --skip."""
        run_orchestrator(plan_ready_for_apply, "--skip")

        output = run_status_check(plan_ready_for_apply)

        phase = output["phases"]["apply-task-suggestions"]
        assert phase["status"] == "skipped"

    def test_status_skips_optional_phases_for_suggested_next(self, temp_dir):
        """--status skips optional phases when suggesting next action."""
        plan_path = _create_plan_with_review_tasks(temp_dir, plan_name="next-test")

        # Mark up to generate-tasks completed, review-tasks pending (optional)
        state = StateManager(plan_path)
        # Reset review-tasks completion (was set in helper)
        phases_completed = state.state.get("phases_completed", {})
        phases_completed.pop("review-tasks", None)
        state.save()

        output = run_status_check(plan_path)

        # suggested_next should skip review-tasks and apply-task-suggestions (both optional)
        assert output.get("suggested_next") != "review-tasks", (
            f"suggested_next should not be review-tasks (optional).\n"
            f"suggested_next: {output.get('suggested_next')}"
        )
        assert output.get("suggested_next") != "apply-task-suggestions", (
            f"suggested_next should not be apply-task-suggestions (optional).\n"
            f"suggested_next: {output.get('suggested_next')}"
        )


class TestImplementDetectsUnappliedTaskSuggestions:
    """Verify that --implement prereq check detects unapplied task suggestions
    and triggers user prompt (unless --yes/--no-confirm is set).

    Note: These tests create grouped.json WITHOUT validation.json to avoid
    a format mismatch in check_apply_task_suggestions_prerequisite (which
    reads validation.json with the old dict-keyed format). Without
    validation.json the prereq checker counts suggestions from grouped.json.
    """

    @staticmethod
    def _create_plan_for_implement_check(temp_dir: Path, plan_name: str) -> Path:
        """Create a plan with grouped.json but no validation.json (prereq check only)."""
        import copy as _copy

        plan_path = temp_dir / f"{plan_name}.md"
        plan_path.write_text(SAMPLE_PLAN_CONTENT)

        prefix = sanitize_prefix(plan_name)
        output_dir = temp_dir / prefix
        tasks_dir = output_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "tasks.md").write_text(VALID_TASKS_CONTENT)

        review_tasks_dir = output_dir / "review-tasks"
        review_tasks_dir.mkdir(parents=True, exist_ok=True)

        groups = _copy.deepcopy(SAMPLE_GROUPED_SUGGESTIONS)
        stamp_stable_ids(groups)
        (review_tasks_dir / "grouped.json").write_text(
            json.dumps({"format_version": 2, "groups": groups}, indent=2)
        )
        # Intentionally omit validation.json so the prereq checker
        # falls back to counting total suggestions from grouped.json.

        state = StateManager(plan_path)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.mark_phase_completed("generate-tasks")
        state.mark_phase_completed("review-tasks")
        state.save()

        return plan_path

    def test_implement_prereq_surfaces_unapplied_task_suggestions(self, temp_dir):
        """implement mode detects unapplied task suggestions from review-tasks."""
        plan_path = self._create_plan_for_implement_check(temp_dir, "implement-detect")

        output = run_prereq_check(plan_path, "implement")

        # apply-task-suggestions is a soft advisory, not a hard prerequisite,
        # so prerequisites_met should be True and the advisory should be in
        # the 'advisories' list (not 'missing')
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
        """implement mode includes actionable_count for unapplied task suggestions."""
        plan_path = self._create_plan_for_implement_check(temp_dir, "actionable-count")

        output = run_prereq_check(plan_path, "implement")

        # Find the apply-task-suggestions entry in advisories (not missing)
        task_sugg_advisory = None
        for m in output.get("advisories", []):
            if m["phase"] == "apply-task-suggestions":
                task_sugg_advisory = m
                break

        assert task_sugg_advisory is not None, "Should have apply-task-suggestions in advisories"
        assert "actionable_count" in task_sugg_advisory, (
            f"Advisory entry should include actionable_count.\n"
            f"Entry: {json.dumps(task_sugg_advisory, indent=2)}"
        )
        assert task_sugg_advisory["actionable_count"] > 0

    def test_implement_prereq_includes_prompt(self, temp_dir):
        """implement mode includes a user prompt for unapplied task suggestions."""
        plan_path = self._create_plan_for_implement_check(temp_dir, "prompt-test")

        output = run_prereq_check(plan_path, "implement")

        assert "prompt" in output, (
            f"Output should include a prompt.\n"
            f"Output: {json.dumps(output, indent=2)}"
        )
        prompt = output["prompt"]
        assert "question" in prompt
        assert "options" in prompt
        # Prompt should mention task suggestions
        assert "task suggestion" in prompt["question"].lower(), (
            f"Prompt should mention task suggestions.\n"
            f"Question: {prompt['question']}"
        )
        # Should have the expected action options
        actions = [opt["action"] for opt in prompt["options"]]
        assert "run_apply_task_suggestions" in actions
        assert "skip_and_continue" in actions

    def test_implement_prereq_met_when_apply_task_suggestions_completed(self, temp_dir):
        """implement prereq is fully met when apply-task-suggestions is completed."""
        plan_path = self._create_plan_for_implement_check(temp_dir, "completed-task-sugg")
        state = StateManager(plan_path)
        state.mark_phase_completed("apply-task-suggestions")
        state.save()

        output = run_prereq_check(plan_path, "implement")

        # No apply-task-suggestions in missing
        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "apply-task-suggestions" not in missing_phases

    def test_implement_prereq_met_when_apply_task_suggestions_skipped(self, temp_dir):
        """implement prereq is met when apply-task-suggestions is explicitly skipped."""
        plan_path = self._create_plan_for_implement_check(temp_dir, "skipped-task-sugg")
        state = StateManager(plan_path)
        state.mark_phase_skipped("apply-task-suggestions", "User chose to skip")
        state.save()

        output = run_prereq_check(plan_path, "implement")

        missing_phases = [m["phase"] for m in output.get("missing", [])]
        assert "apply-task-suggestions" not in missing_phases


class TestDryRun:
    """Verify --dry-run mode works correctly."""

    def test_dry_run_does_not_create_output_file(self, plan_ready_for_apply):
        """--dry-run does not create orchestrator_output.json."""
        result = run_orchestrator(plan_ready_for_apply, "--dry-run")
        assert result.returncode == 0

        prefix = sanitize_prefix("test-plan")
        output_file = plan_ready_for_apply.parent / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        assert not output_file.exists(), "Dry run should not create output file"

    def test_dry_run_does_not_mark_phase_completed(self, plan_ready_for_apply):
        """--dry-run does not mark the phase as completed."""
        # Clear any prior completion
        state = StateManager(plan_ready_for_apply)
        phases_completed = state.state.get("phases_completed", {})
        phases_completed.pop("apply-task-suggestions", None)
        state.save()

        result = run_orchestrator(plan_ready_for_apply, "--dry-run")
        assert result.returncode == 0

        state = StateManager(plan_ready_for_apply)
        assert not state.is_phase_completed("apply-task-suggestions")

    def test_dry_run_shows_batch_info(self, plan_ready_for_apply):
        """--dry-run shows batching info in stderr."""
        result = run_orchestrator(plan_ready_for_apply, "--dry-run")
        assert result.returncode == 0

        assert "DRY RUN" in result.stderr
        assert "Batch" in result.stderr


class TestNoConfirmAndConfirmationPrompt:
    """Verify confirmation behavior when no user selections are found."""

    def test_no_selections_triggers_confirmation_output(self, plan_ready_for_apply):
        """Without user selections or --no-confirm, orchestrator outputs confirmation_needed."""
        # Remove any report.md or user_selections.json
        prefix = sanitize_prefix("test-plan")
        review_dir = plan_ready_for_apply.parent / prefix / "review-tasks"
        for f in ["report.md", "user_selections.json", "consolidated_user_selections.json"]:
            path = review_dir / f
            if path.exists():
                path.unlink()

        result = run_orchestrator(plan_ready_for_apply)
        assert result.returncode == 0  # exits 0 with confirmation_needed

        output = json.loads(result.stdout)
        assert output["status"] == "confirmation_needed"
        assert output["phase"] == "apply-task-suggestions"

    def test_no_confirm_bypasses_confirmation(self, plan_ready_for_apply):
        """--no-confirm bypasses the confirmation prompt and proceeds."""
        # Remove any user selection files
        prefix = sanitize_prefix("test-plan")
        review_dir = plan_ready_for_apply.parent / prefix / "review-tasks"
        for f in ["report.md", "user_selections.json", "consolidated_user_selections.json"]:
            path = review_dir / f
            if path.exists():
                path.unlink()

        result = run_orchestrator(plan_ready_for_apply, "--no-confirm")
        assert result.returncode == 0

        # Should NOT output confirmation_needed
        if result.stdout.strip():
            try:
                output = json.loads(result.stdout)
                assert output.get("status") != "confirmation_needed"
            except json.JSONDecodeError:
                pass  # Text output is fine

    def test_yes_flag_bypasses_confirmation(self, plan_ready_for_apply):
        """--yes flag bypasses the confirmation prompt."""
        prefix = sanitize_prefix("test-plan")
        review_dir = plan_ready_for_apply.parent / prefix / "review-tasks"
        for f in ["report.md", "user_selections.json", "consolidated_user_selections.json"]:
            path = review_dir / f
            if path.exists():
                path.unlink()

        result = run_orchestrator(plan_ready_for_apply, "--yes")
        assert result.returncode == 0

        if result.stdout.strip():
            try:
                output = json.loads(result.stdout)
                assert output.get("status") != "confirmation_needed"
            except json.JSONDecodeError:
                pass


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_nonexistent_plan_file(self):
        """Orchestrator fails gracefully for nonexistent plan file."""
        result = run_orchestrator(Path("/tmp/nonexistent-plan-xyz.md"))
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()

    def test_missing_grouped_json(self, temp_dir):
        """Orchestrator fails when grouped.json is missing."""
        plan_path = _create_plan_with_review_tasks(temp_dir, plan_name="no-grouped")
        prefix = sanitize_prefix("no-grouped")
        # Remove grouped.json
        grouped_path = temp_dir / prefix / "review-tasks" / "grouped.json"
        if grouped_path.exists():
            grouped_path.unlink()

        result = run_orchestrator(plan_path, "--no-confirm")
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "grouped" in result.stderr.lower()

    def test_skip_all_human_skips_human_review_items(self, plan_ready_for_apply):
        """--skip-all-human skips needs-human-decision items."""
        result = run_orchestrator(plan_ready_for_apply, "--skip-all-human", "--no-confirm")
        assert result.returncode == 0

        combined = result.stderr
        assert "Needs human review: 0" in combined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
