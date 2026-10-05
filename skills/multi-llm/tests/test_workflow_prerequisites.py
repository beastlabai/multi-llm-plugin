"""Tests for workflow prerequisite guards functionality."""

import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import StateManager


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_plan_file(temp_dir):
    """Create a sample plan file for testing."""
    plan_content = """# Sample Implementation Plan

## Overview
This is a sample plan for testing.

## Tasks

### T001: Create directory structure
Create the basic directory structure.
- Depends on: none
"""
    plan_path = temp_dir / "test-plan.md"
    plan_path.write_text(plan_content, encoding="utf-8")
    return plan_path


@pytest.fixture
def state_manager(sample_plan_file):
    """Create a StateManager instance for testing."""
    return StateManager(sample_plan_file)


@pytest.fixture
def plan_with_review_results(temp_dir):
    """Create a plan with review-plan results (validation.json)."""
    plan_content = """# Test Plan

## Overview
Test plan with review results.
"""
    plan_path = temp_dir / "plan-with-review.md"
    plan_path.write_text(plan_content, encoding="utf-8")

    # Create output directory structure
    output_dir = temp_dir / "plan-with-review"
    review_dir = output_dir / "review-plan"
    review_dir.mkdir(parents=True)

    # Create validation.json with valid suggestions
    validation = {
        "group_abc123": {"status": "valid", "importance": "HIGH"},
        "group_def456": {"status": "valid", "importance": "MEDIUM"},
        "group_ghi789": {"status": "invalid", "importance": "LOW"},
    }
    (review_dir / "validation.json").write_text(json.dumps(validation), encoding="utf-8")

    return plan_path


@pytest.fixture
def plan_with_tasks(temp_dir):
    """Create a plan with tasks file."""
    plan_content = """# Test Plan

## Overview
Test plan with tasks.
"""
    plan_path = temp_dir / "plan-with-tasks.md"
    plan_path.write_text(plan_content, encoding="utf-8")

    # Create tasks directory and file
    output_dir = temp_dir / "plan-with-tasks"
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True)

    tasks_content = """# Implementation Tasks

## Task 1: Setup
Description here.
"""
    (tasks_dir / "tasks.md").write_text(tasks_content, encoding="utf-8")

    # Mark generate-tasks as completed in state
    from utils.state_manager import StateManager
    state = StateManager(plan_path)
    state.mark_phase_completed("generate-tasks")
    state.save()

    return plan_path


# --- TestStateManagerPhaseCompletion ---

class TestStateManagerPhaseCompletion:
    """Tests for StateManager phase completion tracking methods."""

    def test_mark_phase_completed_stores_timestamp(self, state_manager):
        """mark_phase_completed stores an ISO timestamp."""
        state_manager.mark_phase_completed("apply-suggestions")

        phases = state_manager.state.get("phases_completed", {})
        assert "apply-suggestions" in phases
        # Verify it's an ISO timestamp format
        assert "T" in phases["apply-suggestions"]

    def test_is_phase_completed_returns_false_by_default(self, state_manager):
        """is_phase_completed returns False for uncompleted phases."""
        assert state_manager.is_phase_completed("apply-suggestions") is False
        assert state_manager.is_phase_completed("implement") is False
        assert state_manager.is_phase_completed("nonexistent-phase") is False

    def test_is_phase_completed_returns_true_after_mark(self, state_manager):
        """is_phase_completed returns True after marking complete."""
        state_manager.mark_phase_completed("apply-suggestions")

        assert state_manager.is_phase_completed("apply-suggestions") is True

    def test_multiple_phases_tracked_independently(self, state_manager):
        """Multiple phases can be tracked independently."""
        state_manager.mark_phase_completed("review-plan")
        state_manager.mark_phase_completed("apply-suggestions")

        assert state_manager.is_phase_completed("review-plan") is True
        assert state_manager.is_phase_completed("apply-suggestions") is True
        assert state_manager.is_phase_completed("implement") is False

    def test_phase_completion_persists_after_save_reload(self, sample_plan_file):
        """Phase completion persists after save and reload."""
        sm1 = StateManager(sample_plan_file)
        sm1.mark_phase_completed("apply-suggestions")
        sm1.save()

        sm2 = StateManager(sample_plan_file)
        assert sm2.is_phase_completed("apply-suggestions") is True


# --- TestStateManagerPhaseSkipped ---

class TestStateManagerPhaseSkipped:
    """Tests for StateManager phase skipped tracking methods."""

    def test_mark_phase_skipped_stores_info(self, state_manager):
        """mark_phase_skipped stores skip info with reason and timestamp."""
        state_manager.mark_phase_skipped("apply-suggestions", "User chose to skip")

        phases_skipped = state_manager.state.get("phases_skipped", {})
        assert "apply-suggestions" in phases_skipped
        assert phases_skipped["apply-suggestions"]["reason"] == "User chose to skip"
        assert "skipped_at" in phases_skipped["apply-suggestions"]

    def test_is_phase_skipped_returns_false_by_default(self, state_manager):
        """is_phase_skipped returns False for non-skipped phases."""
        assert state_manager.is_phase_skipped("apply-suggestions") is False
        assert state_manager.is_phase_skipped("implement") is False

    def test_is_phase_skipped_returns_true_after_mark(self, state_manager):
        """is_phase_skipped returns True after marking skipped."""
        state_manager.mark_phase_skipped("apply-suggestions", "No suggestions to apply")

        assert state_manager.is_phase_skipped("apply-suggestions") is True

    def test_get_phase_skip_reason_returns_correct_reason(self, state_manager):
        """get_phase_skip_reason returns the correct reason."""
        state_manager.mark_phase_skipped("apply-suggestions", "User chose to skip")

        reason = state_manager.get_phase_skip_reason("apply-suggestions")
        assert reason == "User chose to skip"

    def test_get_phase_skip_reason_returns_none_for_non_skipped(self, state_manager):
        """get_phase_skip_reason returns None for non-skipped phases."""
        reason = state_manager.get_phase_skip_reason("apply-suggestions")
        assert reason is None

    def test_phase_skipped_persists_after_save_reload(self, sample_plan_file):
        """Phase skipped info persists after save and reload."""
        sm1 = StateManager(sample_plan_file)
        sm1.mark_phase_skipped("apply-suggestions", "Test reason")
        sm1.save()

        sm2 = StateManager(sample_plan_file)
        assert sm2.is_phase_skipped("apply-suggestions") is True
        assert sm2.get_phase_skip_reason("apply-suggestions") == "Test reason"

    def test_skipped_and_completed_are_independent(self, state_manager):
        """Skipped and completed states are tracked independently."""
        state_manager.mark_phase_completed("review-plan")
        state_manager.mark_phase_skipped("apply-suggestions", "Skipped by user")

        assert state_manager.is_phase_completed("review-plan") is True
        assert state_manager.is_phase_skipped("review-plan") is False
        assert state_manager.is_phase_completed("apply-suggestions") is False
        assert state_manager.is_phase_skipped("apply-suggestions") is True


# --- TestCheckWorkflowPrerequisites ---

class TestCheckWorkflowPrerequisites:
    """Tests for check_workflow_prerequisites.py script."""

    def run_prereq_check(self, plan_path: Path, mode: str) -> dict:
        """Helper to run the prerequisite check script and parse output."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(plan_path), "--mode", mode],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )
        return json.loads(result.stdout)

    def test_no_review_plan_results_prerequisites_met(self, sample_plan_file):
        """Prerequisites are met when no review-plan results exist."""
        output = self.run_prereq_check(sample_plan_file, "generate-tasks")

        assert output["prerequisites_met"] is True
        assert output["mode"] == "generate-tasks"
        assert len(output.get("missing", [])) == 0

    def test_apply_suggestions_completed_prerequisites_met(self, plan_with_review_results):
        """Prerequisites are met when apply-suggestions is completed."""
        state = StateManager(plan_with_review_results)
        state.mark_phase_completed("apply-suggestions")
        state.save()

        output = self.run_prereq_check(plan_with_review_results, "generate-tasks")

        assert output["prerequisites_met"] is True

    def test_apply_suggestions_skipped_prerequisites_met(self, plan_with_review_results):
        """Prerequisites are met when apply-suggestions is skipped."""
        state = StateManager(plan_with_review_results)
        state.mark_phase_skipped("apply-suggestions", "User skipped")
        state.save()

        output = self.run_prereq_check(plan_with_review_results, "generate-tasks")

        assert output["prerequisites_met"] is True

    def test_unapplied_suggestions_prerequisites_not_met(self, plan_with_review_results):
        """Prerequisites are NOT met when valid suggestions exist but not applied."""
        output = self.run_prereq_check(plan_with_review_results, "generate-tasks")

        assert output["prerequisites_met"] is False
        assert len(output["missing"]) == 1
        assert output["missing"][0]["phase"] == "apply-suggestions"
        assert output["missing"][0]["valid_count"] == 2  # 2 valid suggestions

    def test_unapplied_suggestions_includes_importance_breakdown(self, plan_with_review_results):
        """Missing prerequisite includes importance breakdown."""
        output = self.run_prereq_check(plan_with_review_results, "generate-tasks")

        assert output["prerequisites_met"] is False
        breakdown = output["missing"][0]["importance_breakdown"]
        assert breakdown["HIGH"] == 1
        assert breakdown["MEDIUM"] == 1
        assert breakdown.get("LOW", 0) == 0  # Invalid suggestion not counted

    def test_prompt_included_when_prerequisites_not_met(self, plan_with_review_results):
        """Prompt is included in output when prerequisites not met."""
        output = self.run_prereq_check(plan_with_review_results, "generate-tasks")

        assert "prompt" in output
        assert "question" in output["prompt"]
        assert "options" in output["prompt"]
        assert len(output["prompt"]["options"]) == 3

    def test_implement_mode_also_checks_apply_suggestions(self, plan_with_review_results):
        """Implement mode also checks apply-suggestions prerequisite."""
        output = self.run_prereq_check(plan_with_review_results, "implement")

        assert output["prerequisites_met"] is False
        # Should have apply-suggestions missing
        phases = [m["phase"] for m in output["missing"]]
        assert "apply-suggestions" in phases

    def test_implement_mode_checks_tasks_exist(self, sample_plan_file):
        """Implement mode checks that tasks exist."""
        output = self.run_prereq_check(sample_plan_file, "implement")

        # Should have generate-tasks missing (no tasks file)
        phases = [m["phase"] for m in output["missing"]]
        assert "generate-tasks" in phases

    def test_implement_mode_tasks_exist_with_file(self, plan_with_tasks):
        """Implement mode passes when tasks file exists."""
        output = self.run_prereq_check(plan_with_tasks, "implement")

        # Should not have generate-tasks missing
        phases = [m["phase"] for m in output.get("missing", [])]
        assert "generate-tasks" not in phases

    def test_phase_flag_works_as_alias_for_mode(self, sample_plan_file):
        """--phase works as an alias for --mode."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(sample_plan_file),
             "--phase", "review-plan"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )
        output = json.loads(result.stdout)

        assert output["prerequisites_met"] is True
        assert output["mode"] == "review-plan"


# --- TestCheckWorkflowPrerequisitesNoValidSuggestions ---

class TestCheckWorkflowPrerequisitesNoValidSuggestions:
    """Tests for when validation.json exists but has no valid suggestions."""

    @pytest.fixture
    def plan_with_all_invalid_suggestions(self, temp_dir):
        """Create a plan with review results where all suggestions are invalid."""
        plan_content = "# Test Plan\n"
        plan_path = temp_dir / "plan-all-invalid.md"
        plan_path.write_text(plan_content, encoding="utf-8")

        output_dir = temp_dir / "plan-all-invalid"
        review_dir = output_dir / "review-plan"
        review_dir.mkdir(parents=True)

        validation = {
            "group_abc123": {"status": "invalid", "importance": "HIGH"},
            "group_def456": {"status": "invalid", "importance": "MEDIUM"},
        }
        (review_dir / "validation.json").write_text(json.dumps(validation), encoding="utf-8")

        return plan_path

    def run_prereq_check(self, plan_path: Path, mode: str) -> dict:
        """Helper to run the prerequisite check script."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(plan_path), "--mode", mode],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )
        return json.loads(result.stdout)

    def test_no_valid_suggestions_prerequisites_met(self, plan_with_all_invalid_suggestions):
        """Prerequisites are met when all suggestions are invalid."""
        output = self.run_prereq_check(plan_with_all_invalid_suggestions, "generate-tasks")

        assert output["prerequisites_met"] is True


# --- TestApplySuggestionsSkipFlag ---

class TestApplySuggestionsSkipFlag:
    """Tests for apply_suggestions_orchestrator.py --skip flag."""

    def test_skip_flag_marks_phase_skipped(self, plan_with_review_results):
        """--skip flag marks the phase as skipped in state."""
        script_path = Path(__file__).parent.parent / "apply_suggestions_orchestrator.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(plan_with_review_results), "--skip"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["status"] == "skipped"

        # Verify state was updated
        state = StateManager(plan_with_review_results)
        assert state.is_phase_skipped("apply-suggestions") is True
        assert "User chose to skip" in state.get_phase_skip_reason("apply-suggestions")

    def test_skip_flag_outputs_json(self, sample_plan_file):
        """--skip flag outputs proper JSON response."""
        script_path = Path(__file__).parent.parent / "apply_suggestions_orchestrator.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(sample_plan_file), "--skip"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["status"] == "skipped"
        assert "message" in output


# --- TestReviewTasksPrerequisites ---

class TestReviewTasksPrerequisites:
    """Tests for review-tasks phase prerequisite checking."""

    def run_prereq_check(self, plan_path: Path, mode: str) -> dict:
        """Helper to run the prerequisite check script and parse output."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(plan_path), "--mode", mode],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )
        return json.loads(result.stdout)

    def test_review_tasks_requires_generate_tasks(self, sample_plan_file):
        """review-tasks fails when generate-tasks has not been run."""
        output = self.run_prereq_check(sample_plan_file, "review-tasks")

        assert output["prerequisites_met"] is False
        phases = [m["phase"] for m in output["missing"]]
        assert "generate-tasks" in phases

    def test_review_tasks_passes_when_generate_tasks_completed(self, plan_with_tasks):
        """review-tasks passes when generate-tasks is completed."""
        output = self.run_prereq_check(plan_with_tasks, "review-tasks")

        assert output["prerequisites_met"] is True
        assert len(output.get("missing", [])) == 0

    def test_review_tasks_passes_when_generate_tasks_skipped(self, sample_plan_file):
        """review-tasks passes when generate-tasks is skipped."""
        state = StateManager(sample_plan_file)
        state.mark_phase_skipped("generate-tasks", "Tasks already exist")
        state.save()

        output = self.run_prereq_check(sample_plan_file, "review-tasks")

        assert output["prerequisites_met"] is True

    def test_review_tasks_accepted_as_mode_value(self, sample_plan_file):
        """review-tasks is accepted as a valid --mode value."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(sample_plan_file),
             "--mode", "review-tasks"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )
        # Should not error on invalid choice
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["mode"] == "review-tasks"

    def test_unapplied_task_suggestions_notice_counts_groups_not_suggestions(
        self, temp_dir
    ):
        """F9.5: `actionable_count` is `valid_count` — one increment per
        validation VERDICT, i.e. per GROUP — never per suggestion inside a
        group. Three groups of two suggestions each (six suggestions total),
        two valid groups: the notice must say "2 unapplied task suggestion
        groups", not "2 unapplied task suggestions" (which reads as a
        suggestion count, and was measured 2026-09-16 on a run where 17 valid
        groups held 26 suggestions)."""
        plan_path = temp_dir / "notice-wording.md"
        plan_path.write_text("# Test Plan\n", encoding="utf-8")

        review_dir = temp_dir / "notice-wording" / "review-tasks"
        review_dir.mkdir(parents=True)

        def _group(n):
            return {
                "category": "addition", "theme": f"group {n}",
                "suggestions": [
                    {"title": f"s{n}a", "desc": "d", "importance": "LOW", "type": "addition"},
                    {"title": f"s{n}b", "desc": "d", "importance": "LOW", "type": "addition"},
                ],
            }

        grouped = {"format_version": 2, "groups": [_group(1), _group(2), _group(3)]}
        (review_dir / "grouped.json").write_text(json.dumps(grouped), encoding="utf-8")
        validation = {
            "format_version": 2,
            "groups": [
                {"group_index": 0, "status": "valid"},
                {"group_index": 1, "status": "valid"},
                {"group_index": 2, "status": "invalid"},
            ],
        }
        (review_dir / "validation.json").write_text(json.dumps(validation), encoding="utf-8")

        state = StateManager(plan_path)
        state.mark_phase_completed("review-tasks")
        state.save()

        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(plan_path), "--status"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(script_path.parent),
        )
        output = json.loads(result.stdout)

        notices = output.get("notices") or []
        assert notices, f"expected a notice on the workflow status, got: {output}"
        assert "2 unapplied task suggestion groups" in notices[0], notices[0]
        # The old wording, without "groups", must be gone — it undercounted
        # by calling 2 groups (6 suggestions) "2 suggestions".
        assert "2 unapplied task suggestions." not in notices[0], notices[0]


# --- TestSkipReasonFlags ---

class TestSkipReasonFlags:
    """Tests for --skip and --reason CLI flags on check_workflow_prerequisites.py."""

    def test_skip_with_reason_marks_phase_skipped(self, sample_plan_file):
        """--skip with --reason marks the phase as skipped in state."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(sample_plan_file),
             "--mode", "review-tasks", "--skip", "--reason", "Not needed for this plan"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["skipped"] is True
        assert output["mode"] == "review-tasks"
        assert output["reason"] == "Not needed for this plan"

        # Verify state was updated
        state = StateManager(sample_plan_file)
        assert state.is_phase_skipped("review-tasks") is True
        assert state.get_phase_skip_reason("review-tasks") == "Not needed for this plan"

    def test_skip_without_reason_errors(self, sample_plan_file):
        """--skip without --reason exits with error."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(sample_plan_file),
             "--mode", "review-tasks", "--skip"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )

        assert result.returncode == 1
        assert "ERROR" in result.stderr or "ERROR" in result.stdout

    def test_skip_with_different_phases(self, sample_plan_file):
        """--skip works with different phase values."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        for phase in ["apply-suggestions", "generate-tasks", "review-tasks", "apply-task-suggestions"]:
            result = subprocess.run(
                [sys.executable, str(script_path), "--plan-file", str(sample_plan_file),
                 "--mode", phase, "--skip", "--reason", f"Skipping {phase}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(script_path.parent)
            )

            assert result.returncode == 0
            output = json.loads(result.stdout)
            assert output["skipped"] is True
            assert output["mode"] == phase

    def test_skip_outputs_valid_json(self, sample_plan_file):
        """--skip outputs valid JSON with expected fields."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(sample_plan_file),
             "--mode", "apply-suggestions", "--skip", "--reason", "Test reason"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "skipped" in output
        assert "mode" in output
        assert "reason" in output

    def test_skip_without_mode_errors(self, sample_plan_file):
        """--skip without --mode exits with error."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(sample_plan_file),
             "--skip", "--reason", "Some reason"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )

        assert result.returncode != 0


# --- TestStatusMode ---

class TestStatusMode:
    """Tests for --status mode."""

    def run_status_check(self, plan_path: Path) -> dict:
        """Helper to run the status check and parse output."""
        script_path = Path(__file__).parent.parent / "check_workflow_prerequisites.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--plan-file", str(plan_path), "--status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(script_path.parent)
        )
        return json.loads(result.stdout)

    def test_status_shows_all_phases_pending_initially(self, sample_plan_file):
        """--status shows all phases as pending for a new plan."""
        output = self.run_status_check(sample_plan_file)

        assert "phases" in output
        assert "suggested_next" in output
        for phase in ["review-plan", "apply-suggestions", "generate-tasks",
                      "review-tasks", "apply-task-suggestions", "implement",
                      "code-review", "apply-code-fixes"]:
            assert phase in output["phases"]
            assert output["phases"][phase]["status"] == "pending"
        # review-tasks and apply-task-suggestions are optional
        assert output["phases"]["review-tasks"].get("optional") is True
        assert output["phases"]["apply-task-suggestions"].get("optional") is True
        assert output["suggested_next"] == "review-plan"

    def test_status_shows_completed_phases(self, sample_plan_file):
        """--status shows completed phases with timestamps."""
        state = StateManager(sample_plan_file)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.save()

        output = self.run_status_check(sample_plan_file)

        assert output["phases"]["review-plan"]["status"] == "completed"
        assert "timestamp" in output["phases"]["review-plan"]
        assert output["phases"]["apply-suggestions"]["status"] == "completed"
        assert output["phases"]["generate-tasks"]["status"] == "pending"
        assert output["suggested_next"] == "generate-tasks"

    def test_status_shows_skipped_phases(self, sample_plan_file):
        """--status shows skipped phases with reason."""
        state = StateManager(sample_plan_file)
        state.mark_phase_completed("review-plan")
        state.mark_phase_skipped("apply-suggestions", "No suggestions to apply")
        state.save()

        output = self.run_status_check(sample_plan_file)

        assert output["phases"]["apply-suggestions"]["status"] == "skipped"
        assert "reason" in output["phases"]["apply-suggestions"]
        assert output["suggested_next"] == "generate-tasks"

    def test_status_suggests_next_action_correctly(self, sample_plan_file):
        """--status suggests the correct next action based on completed phases."""
        state = StateManager(sample_plan_file)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.mark_phase_completed("generate-tasks")
        state.save()

        output = self.run_status_check(sample_plan_file)

        assert output["suggested_next"] == "implement"

    def test_status_includes_plan_path(self, sample_plan_file):
        """--status includes the plan path in output.

        check_workflow_prerequisites emits ``str(Path(plan_file).resolve())``,
        so the reported path is normalized and OS-native: on Windows it uses
        backslashes and expands any 8.3 short components in the temp dir
        (GitHub's runners hand out ``C:\\Users\\RUNNER~1\\...``). A substring
        check against the raw fixture path therefore cannot hold. Compare
        resolved Path objects, which normalizes both sides on every platform.
        """
        output = self.run_status_check(sample_plan_file)

        assert "plan" in output
        assert Path(output["plan"]) == Path(sample_plan_file).resolve()

    def test_status_review_tasks_marked_optional(self, sample_plan_file):
        """--status marks review-tasks as optional."""
        output = self.run_status_check(sample_plan_file)

        assert output["phases"]["review-tasks"]["status"] == "pending"
        assert output["phases"]["review-tasks"].get("optional") is True

    def test_status_skips_optional_phases_for_suggested_next(self, sample_plan_file):
        """--status skips optional phases (review-tasks) when suggesting next action."""
        state = StateManager(sample_plan_file)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.mark_phase_completed("generate-tasks")
        state.save()

        output = self.run_status_check(sample_plan_file)

        # review-tasks is optional, so suggested_next should skip it and suggest implement
        assert output["suggested_next"] == "implement"
        assert output["phases"]["review-tasks"]["status"] == "pending"
        assert output["phases"]["review-tasks"].get("optional") is True

    def test_status_shows_review_tasks_completed(self, sample_plan_file):
        """--status shows review-tasks as completed when marked."""
        state = StateManager(sample_plan_file)
        state.mark_phase_completed("review-tasks")
        state.save()

        output = self.run_status_check(sample_plan_file)

        assert output["phases"]["review-tasks"]["status"] == "completed"
        assert "timestamp" in output["phases"]["review-tasks"]

    def test_status_shows_review_tasks_skipped(self, sample_plan_file):
        """--status shows review-tasks as skipped with reason."""
        state = StateManager(sample_plan_file)
        state.mark_phase_skipped("review-tasks", "Tasks are simple enough")
        state.save()

        output = self.run_status_check(sample_plan_file)

        assert output["phases"]["review-tasks"]["status"] == "skipped"
        assert "reason" in output["phases"]["review-tasks"]

    def test_status_does_not_count_apply_batches_as_implement_tasks(self, sample_plan_file):
        """F4 regression: `implement`'s task progress must come only from the
        decomposer's own `T###` ids, never from apply-suggestions' write-batch
        ids (`apply-batch-N`) that share the same `task_status` map.

        Before the fix, `implement` read as `(1/1 tasks)` here — and, worse,
        would have flipped to `in_progress` on a single failed apply batch —
        despite no implement task ever having been generated.
        """
        state = StateManager(sample_plan_file)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.mark_phase_completed("generate-tasks")
        state.update_task_status("apply-batch-0", "completed")
        state.save()

        output = self.run_status_check(sample_plan_file)

        assert output["phases"]["implement"]["status"] == "pending"
        assert "tasks_done" not in output["phases"]["implement"]
        assert "tasks_total" not in output["phases"]["implement"]

    def test_status_counts_implement_t_ids_and_ignores_apply_batches(self, sample_plan_file):
        """A mixed map (both phases have run) reports progress from the
        implement-owned ids only."""
        state = StateManager(sample_plan_file)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.mark_phase_completed("generate-tasks")
        state.update_task_status("apply-batch-0", "completed")
        state.update_task_status("T001", "completed")
        state.update_task_status("T002", "pending")
        state.save()

        output = self.run_status_check(sample_plan_file)

        assert output["phases"]["implement"]["status"] == "in_progress"
        assert output["phases"]["implement"]["tasks_done"] == 1
        assert output["phases"]["implement"]["tasks_total"] == 2


# --- T1 (plans/next-phase-command.md): a review that left nothing to apply ---

import check_workflow_prerequisites as prereq  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402

#: Every phase before code-review, i.e. "every earlier phase on record" for the
#: apply-code-fixes rows (the two optional phases never block suggested_next).
_BEFORE_CODE_REVIEW = (
    "review-plan", "apply-suggestions", "generate-tasks", "implement",
)

_ONE_FINDING = {"format_version": 2, "groups": [
    {"group_id": "g1", "theme": "t",
     "suggestions": [{"title": "fix it", "importance": "HIGH"}]},
]}


def _workflow_plan(tmp_path: Path, name: str = "t1-plan") -> Path:
    plan = tmp_path / f"{name}.md"
    plan.write_text("# T1 plan\n", encoding="utf-8")
    return plan


def _record(plan: Path, completed=(), skipped=(), tasks=None) -> None:
    state = StateManager(plan)
    for phase in completed:
        state.mark_phase_completed(phase)
    for phase in skipped:
        state.mark_phase_skipped(phase, f"test skip of {phase}")
    for task_id, status in (tasks or {}).items():
        state.update_task_status(task_id, status)
    state.save()


def _put(plan: Path, phase_dir: str, name: str, content) -> Path:
    target = output_dir_path(plan) / phase_dir
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return path


def _status(plan: Path) -> dict:
    return prereq.get_workflow_status(
        StateManager(plan), plan, output_dir_path(plan)
    )


def _fixture_clean_v2(plan):
    _put(plan, "code-review", "grouped.json",
         {"format_version": 2, "groups": [{"group_id": "g1", "suggestions": []}]})


def _fixture_clean_v1(plan):
    _put(plan, "code-review", "grouped.json", [{"group_id": "g1", "suggestions": []}])


def _fixture_missing(plan):
    output_dir_path(plan).mkdir(parents=True, exist_ok=True)


def _fixture_unreadable(plan):
    _put(plan, "code-review", "grouped.json", "{not json")


def _fixture_invalid_finding(plan):
    _put(plan, "code-review", "grouped.json", _ONE_FINDING)
    _put(plan, "code-review", "validation.json", {"format_version": 2, "groups": [
        {"group_index": 0, "status": "invalid"},
    ]})


#: (id, grouped.json fixture, code-review recorded as, apply-code-fixes expected)
CODE_REVIEW_FIXTURES = [
    ("clean-v2", _fixture_clean_v2, "completed", ("skipped", "skipped (no findings)")),
    ("clean-v1", _fixture_clean_v1, "completed", ("skipped", "skipped (no findings)")),
    ("missing", _fixture_missing, "completed", ("skipped", "skipped (no findings)")),
    ("unreadable", _fixture_unreadable, "completed", ("skipped", "skipped (no findings)")),
    ("finding-judged-invalid", _fixture_invalid_finding, "completed", ("pending", None)),
    ("review-skipped-finding-on-disk",
     lambda plan: _put(plan, "code-review", "grouped.json", _ONE_FINDING),
     "skipped", ("skipped", "skipped (prerequisite skipped)")),
]


def _code_review_plan(tmp_path, fixture, review_status) -> Path:
    plan = _workflow_plan(tmp_path)
    fixture(plan)
    if review_status == "completed":
        _record(plan, completed=_BEFORE_CODE_REVIEW + ("code-review",))
    else:
        _record(plan, completed=_BEFORE_CODE_REVIEW, skipped=("code-review",))
    return plan


class TestApplyCodeFixesKnowsWhenReviewLeftNothing:
    @pytest.mark.parametrize(
        "fixture,review_status,expected",
        [row[1:] for row in CODE_REVIEW_FIXTURES],
        ids=[row[0] for row in CODE_REVIEW_FIXTURES],
    )
    def test_apply_code_fixes_entry(self, tmp_path, fixture, review_status, expected):
        plan = _code_review_plan(tmp_path, fixture, review_status)
        entry = _status(plan)["phases"]["apply-code-fixes"]
        status, reason = expected
        assert entry["status"] == status
        assert entry.get("reason") == reason

    @pytest.mark.parametrize(
        "fixture",
        [_fixture_clean_v2, _fixture_missing, _fixture_unreadable],
        ids=["no-non-empty-group", "missing", "unreadable"],
    )
    def test_a_clean_review_leaves_nothing_suggested(self, tmp_path, capsys, fixture):
        from utils import run_status

        plan = _code_review_plan(tmp_path, fixture, "completed")
        payload = _status(plan)
        assert payload["phases"]["apply-code-fixes"]["status"] == "skipped"
        assert payload["suggested_next"] is None

        run_status.main(["--plan-file", str(plan)])
        out = capsys.readouterr().out
        assert "workflow (state.json)" in out
        assert "suggested next:" not in out

    def test_one_non_empty_group_keeps_apply_code_fixes_pending_even_if_invalid(
        self, tmp_path
    ):
        plan = _code_review_plan(tmp_path, _fixture_invalid_finding, "completed")
        payload = _status(plan)
        assert payload["phases"]["apply-code-fixes"]["status"] == "pending"
        assert payload["suggested_next"] == "apply-code-fixes"

    def test_a_pending_review_reads_no_file_and_is_itself_suggested(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _put(plan, "code-review", "grouped.json", _ONE_FINDING)
        _record(plan, completed=_BEFORE_CODE_REVIEW)
        payload = _status(plan)
        assert payload["phases"]["apply-code-fixes"] == {"status": "pending"}
        assert payload["suggested_next"] == "code-review"

    def test_a_pending_review_with_a_clean_file_on_disk_marks_nothing(self, tmp_path):
        """A `grouped.json` from a partial run is never read while the review
        itself is not on record."""
        plan = _workflow_plan(tmp_path)
        _fixture_clean_v2(plan)
        _record(plan, completed=_BEFORE_CODE_REVIEW)
        payload = _status(plan)
        assert payload["phases"]["apply-code-fixes"] == {"status": "pending"}
        assert payload["suggested_next"] == "code-review"

    def test_a_skip_full_recorded_keeps_fulls_reason(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _fixture_clean_v2(plan)
        _record(plan, completed=_BEFORE_CODE_REVIEW + ("code-review",))
        state = StateManager(plan)
        state.mark_phase_skipped(
            "apply-code-fixes", "Skipped: code review produced zero actionable findings"
        )
        state.save()
        entry = _status(plan)["phases"]["apply-code-fixes"]
        assert entry["status"] == "skipped"
        assert entry["reason"] == "Skipped: code review produced zero actionable findings"


class TestFullAndStatusAgreeOnApplyCodeFixes:
    """`full`'s `step_apply_code_fixes` and `get_workflow_status` ask the same
    predicate in the same order, so they give the same answer for every
    completed-review and skipped-review fixture."""

    @pytest.mark.parametrize(
        "fixture,review_status,expected",
        [row[1:] for row in CODE_REVIEW_FIXTURES],
        ids=[row[0] for row in CODE_REVIEW_FIXTURES],
    )
    def test_same_answer(self, tmp_path, capsys, fixture, review_status, expected):
        import full_workflow_orchestrator as fwo

        plan = _code_review_plan(tmp_path, fixture, review_status)
        status_says_skip = _status(plan)["phases"]["apply-code-fixes"]["status"] == "skipped"

        args = fwo.parse_args(["--plan-file", str(plan), "--yes"])
        workflow = fwo.FullWorkflow(args)
        ran = []

        def run_mode(mode_word, extra):
            ran.append(mode_word)
            state = StateManager(plan)
            state.mark_phase_completed("apply-code-fixes")
            state.save()
            return 0

        workflow.run_mode = run_mode  # type: ignore[assignment]
        outcome, _detail, _code = workflow.step_apply_code_fixes()
        capsys.readouterr()

        full_says_skip = outcome == fwo.SKIPPED and not ran
        assert full_says_skip == status_says_skip
        assert full_says_skip == (expected[0] == "skipped")
        if full_says_skip:
            assert StateManager(plan).is_phase_skipped("apply-code-fixes")
        else:
            assert ran == ["apply-code-fixes"]


class TestApplyTaskSuggestionsPrerequisiteReadsEffectiveReviewStatus:
    """A review-tasks that was skipped and later completed is completed:
    ``mark_phase_completed`` leaves the old skip record, so both lists name it."""

    def _plan(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        d = output_dir_path(plan) / "review-tasks"
        d.mkdir(parents=True)
        (d / "grouped.json").write_text(json.dumps([
            {"group_id": "g1", "suggestions": [{"title": "x", "importance": "HIGH"}]},
        ]), encoding="utf-8")
        return plan

    def test_plain_skip_is_still_met_by_skip(self, tmp_path):
        plan = self._plan(tmp_path)
        state = StateManager(plan)
        state.mark_phase_skipped("review-tasks", reason="r")
        result = prereq.check_apply_task_suggestions_prerequisite(state, output_dir_path(plan))
        assert result == {"met": True, "reason": "skipped (prerequisite skipped)"}

    def test_skipped_then_completed_is_not_met_by_skip(self, tmp_path):
        plan = self._plan(tmp_path)
        state = StateManager(plan)
        state.mark_phase_skipped("review-tasks", reason="r")
        state.mark_phase_completed("review-tasks")
        assert state.is_phase_skipped("review-tasks")  # the stale record survives
        result = prereq.check_apply_task_suggestions_prerequisite(state, output_dir_path(plan))
        assert result["reason"] != "skipped (prerequisite skipped)"
        assert result["met"] is False
        assert result["actionable_count"] == 1


class TestApplySuggestionsKnowsWhenReviewLeftNothing:
    def test_every_group_invalid_skips_apply_suggestions(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {"format_version": 2, "groups": [
            {"group_index": 0, "status": "invalid"},
            {"group_index": 1, "status": "invalid"},
        ]})
        _record(plan, completed=("review-plan",))
        payload = _status(plan)
        entry = payload["phases"]["apply-suggestions"]
        assert entry["status"] == "skipped"
        assert entry["reason"] == "skipped (no valid findings)"
        assert payload["suggested_next"] == "generate-tasks"

    def test_every_group_invalid_v1_shape_too(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {
            "g1": {"status": "invalid", "importance": "HIGH"},
        })
        _record(plan, completed=("review-plan",))
        payload = _status(plan)
        assert payload["phases"]["apply-suggestions"]["status"] == "skipped"
        assert payload["suggested_next"] == "generate-tasks"

    def test_needs_human_decision_groups_keep_apply_suggestions_pending(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {"format_version": 2, "groups": [
            {"group_index": 0, "status": "needs-human-decision"},
            {"group_index": 1, "status": "invalid"},
        ]})
        _record(plan, completed=("review-plan",))
        payload = _status(plan)
        assert payload["phases"]["apply-suggestions"] == {"status": "pending"}
        assert payload["suggested_next"] == "apply-suggestions"

    def test_open_count_is_carried_only_by_the_no_valid_return(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        plan_dir = output_dir_path(plan)
        state = StateManager(plan)
        # No validation.json: met, no field.
        assert "open_count" not in prereq.check_apply_suggestions_prerequisite(
            state, plan_dir
        )
        _put(plan, "review-plan", "validation.json", {"format_version": 2, "groups": [
            {"group_index": 0, "status": "needs-human-decision"},
            {"group_index": 1, "status": "invalid"},
            {"group_index": 2, "status": "validation_failed"},
        ]})
        check = prereq.check_apply_suggestions_prerequisite(state, plan_dir)
        assert check["met"] is True
        assert check["reason"] == "No valid suggestions to apply"
        assert check["open_count"] == 2
        _put(plan, "review-plan", "validation.json", "{not json")
        assert "open_count" not in prereq.check_apply_suggestions_prerequisite(
            state, plan_dir
        )

    def test_no_validation_json_leaves_apply_suggestions_pending(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan",))
        payload = _status(plan)
        assert payload["phases"]["apply-suggestions"] == {"status": "pending"}
        assert payload["suggested_next"] == "apply-suggestions"


def _review_tasks_actionable(plan):
    _put(plan, "review-tasks", "grouped.json", {"format_version": 2, "groups": [
        {"group_id": "g1", "theme": "t",
         "suggestions": [{"title": "s", "importance": "LOW"}]},
    ]})
    _put(plan, "review-tasks", "validation.json", {"format_version": 2, "groups": [
        {"group_index": 0, "status": "valid"},
    ]})


#: (phase, its own batch id, fixture that leaves the phase pending, the
#: completed phases that fixture records)
APPLY_PHASE_BATCH_ROWS = [
    ("apply-suggestions", "apply-batch-0",
     lambda plan: _put(plan, "review-plan", "validation.json", {
         "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]}),
     ("review-plan",)),
    ("apply-task-suggestions", "apply-task-batch-0",
     _review_tasks_actionable,
     ("review-plan", "apply-suggestions", "generate-tasks", "review-tasks")),
    ("apply-code-fixes", "apply-fix-batch-0",
     lambda plan: _put(plan, "code-review", "grouped.json", _ONE_FINDING),
     _BEFORE_CODE_REVIEW + ("code-review",)),
]


class TestApplyPhaseInProgress:
    def test_a_partly_implemented_plan_keeps_suggesting_implement(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(
            plan,
            completed=("review-plan", "apply-suggestions", "generate-tasks"),
            tasks={"T001": "completed", "T002": "pending"},
        )
        payload = _status(plan)
        assert payload["phases"]["implement"]["status"] == "in_progress"
        assert payload["suggested_next"] == "implement"

    @pytest.mark.parametrize(
        "phase,batch_id,fixture,completed",
        APPLY_PHASE_BATCH_ROWS,
        ids=[row[0] for row in APPLY_PHASE_BATCH_ROWS],
    )
    def test_own_completed_batch_marks_in_progress(
        self, tmp_path, phase, batch_id, fixture, completed
    ):
        pending_plan = _workflow_plan(tmp_path, "pending")
        fixture(pending_plan)
        _record(pending_plan, completed=completed)
        pending = _status(pending_plan)
        assert pending["phases"][phase]["status"] == "pending"

        plan = _workflow_plan(tmp_path, "partial")
        fixture(plan)
        _record(plan, completed=completed, tasks={batch_id: "completed"})
        partial = _status(plan)
        assert partial["phases"][phase]["status"] == "in_progress"
        assert partial["suggested_next"] == pending["suggested_next"]
        others = {p for p, _b, _f, _c in APPLY_PHASE_BATCH_ROWS} - {phase}
        for other in others:
            assert partial["phases"][other]["status"] != "in_progress"

    @pytest.mark.parametrize(
        "phase,batch_id,fixture,completed",
        APPLY_PHASE_BATCH_ROWS,
        ids=[row[0] for row in APPLY_PHASE_BATCH_ROWS],
    )
    def test_a_batch_not_completed_is_no_progress(
        self, tmp_path, phase, batch_id, fixture, completed
    ):
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed, tasks={batch_id: "failed"})
        assert _status(plan)["phases"][phase]["status"] == "pending"

    def test_another_phases_batch_never_counts(self, tmp_path):
        """`apply-batch-0` belongs to apply-suggestions: completed there, it
        leaves that phase completed and apply-task-suggestions pending."""
        plan = _workflow_plan(tmp_path)
        _record(
            plan,
            completed=("review-plan", "apply-suggestions", "generate-tasks"),
            tasks={"apply-batch-0": "completed"},
        )
        phases = _status(plan)["phases"]
        assert phases["apply-suggestions"]["status"] == "completed"
        assert phases["apply-task-suggestions"]["status"] == "pending"
        assert phases["apply-code-fixes"]["status"] == "pending"
        assert not [p for p, info in phases.items() if info["status"] == "in_progress"]

    def test_ids_are_matched_whole(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(
            plan,
            completed=("review-plan",),
            tasks={"apply-batch-0-extra": "completed", "xapply-batch-1": "completed"},
        )
        assert _status(plan)["phases"]["apply-suggestions"]["status"] == "pending"


def _date_cycle(plan, phase, review, reviewed_at, *, started_at=None, metric_at=None):
    """Rewrite the times `state.json` holds for one apply phase and its review.

    `reviewed_at` stamps the review's completion; `started_at` is the phase's
    `metrics_progress.{phase}.started_at`, `metric_at` one `metrics.{phase}`
    entry's `timestamp`. None leaves that record out.
    """
    state = StateManager(plan)
    state.state["phases_completed"][review] = reviewed_at
    # These cycles are about a review re-run, not about implement finishing
    # after a review: keep implement's completion (stamped "now" by `_record`)
    # older than every date the cycle uses, or a code review stamped _OLD/_NEW
    # would read as stale for that reason instead.
    if "implement" in state.state["phases_completed"]:
        state.state["phases_completed"]["implement"] = "2025-12-31T10:00:00"
    if started_at is not None:
        state.state.setdefault("metrics_progress", {})[phase] = {
            "started_at": started_at}
    if metric_at is not None:
        state.state.setdefault("metrics", {})[phase] = [{"timestamp": metric_at}]
    state.save()


_OLD, _NEW = "2026-01-01T10:00:00", "2026-01-02T10:00:00"


class TestStaleApplyBatchesAfterReviewRerun:
    """Batch ids in the flat `task_status` map outlive a `--force` review re-run.

    (A review re-stamped by `--reaggregate` also reads as newer than its older
    batches; that cost is documented in the helper, not pinned here.)
    """

    @pytest.mark.parametrize(
        "phase,batch_id,fixture,completed",
        APPLY_PHASE_BATCH_ROWS,
        ids=[row[0] for row in APPLY_PHASE_BATCH_ROWS],
    )
    @pytest.mark.parametrize("source", ["started_at", "metric_at"])
    def test_batches_older_than_the_review_do_not_mark_in_progress(
        self, tmp_path, phase, batch_id, fixture, completed, source
    ):
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed, tasks={batch_id: "completed"})
        review = prereq.PHASE_PREREQUISITES[phase][0]
        _date_cycle(plan, phase, review, _NEW, **{source: _OLD})
        assert _status(plan)["phases"][phase]["status"] == "pending"

    @pytest.mark.parametrize(
        "phase,batch_id,fixture,completed",
        APPLY_PHASE_BATCH_ROWS,
        ids=[row[0] for row in APPLY_PHASE_BATCH_ROWS],
    )
    @pytest.mark.parametrize("source", ["started_at", "metric_at"])
    def test_batches_newer_than_the_review_still_mark_in_progress(
        self, tmp_path, phase, batch_id, fixture, completed, source
    ):
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed, tasks={batch_id: "completed"})
        review = prereq.PHASE_PREREQUISITES[phase][0]
        _date_cycle(plan, phase, review, _OLD, **{source: _NEW})
        assert _status(plan)["phases"][phase]["status"] == "in_progress"

    def test_the_newest_of_several_stamps_decides(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {
            "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]})
        _record(plan, completed=("review-plan",),
                tasks={"apply-batch-0": "completed"})
        # started_at predates the review, a later batch's metric entry does not
        _date_cycle(plan, "apply-suggestions", "review-plan", _NEW,
                    started_at=_OLD, metric_at="2026-01-03T10:00:00")
        assert _status(plan)["phases"]["apply-suggestions"]["status"] == "in_progress"

    def test_no_dated_run_keeps_the_mark(self, tmp_path):
        """Absence of a date is not staleness."""
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {
            "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]})
        _record(plan, completed=("review-plan",),
                tasks={"apply-batch-0": "completed"})
        _date_cycle(plan, "apply-suggestions", "review-plan", _NEW)
        assert _status(plan)["phases"]["apply-suggestions"]["status"] == "in_progress"

    @pytest.mark.parametrize("bad", ["not-a-date", "", 12345, None])
    def test_unparseable_run_times_keep_the_mark(self, tmp_path, bad):
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {
            "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]})
        _record(plan, completed=("review-plan",),
                tasks={"apply-batch-0": "completed"})
        _date_cycle(plan, "apply-suggestions", "review-plan", _NEW)
        state = StateManager(plan)
        state.state["metrics_progress"] = {"apply-suggestions": {"started_at": bad}}
        state.state["metrics"] = {"apply-suggestions": [{"timestamp": bad}, "junk"]}
        state.save()
        assert _status(plan)["phases"]["apply-suggestions"]["status"] == "in_progress"

    def test_unparseable_review_time_keeps_the_mark(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {
            "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]})
        _record(plan, completed=("review-plan",),
                tasks={"apply-batch-0": "completed"})
        _date_cycle(plan, "apply-suggestions", "review-plan", "garbage",
                    started_at=_OLD)
        assert _status(plan)["phases"]["apply-suggestions"]["status"] == "in_progress"

    def test_timezone_aware_stamps_compare(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {
            "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]})
        _record(plan, completed=("review-plan",),
                tasks={"apply-batch-0": "completed"})
        _date_cycle(plan, "apply-suggestions", "review-plan",
                    "2026-01-02T10:00:00+00:00",
                    started_at="2026-01-01T10:00:00+00:00")
        assert _status(plan)["phases"]["apply-suggestions"]["status"] == "pending"

    def test_a_review_that_is_not_completed_never_marks_in_progress(self, tmp_path):
        """A review not `completed` has no cycle for the batches to belong to,
        so they are an earlier one's (adversarial review, 2026-10-02; until
        then "staleness is judged only against a completed review" kept the
        mark)."""
        plan = _workflow_plan(tmp_path)
        _put(plan, "code-review", "grouped.json", _ONE_FINDING)
        _record(plan, completed=_BEFORE_CODE_REVIEW,
                tasks={"apply-fix-batch-0": "completed"})
        state = StateManager(plan)
        state.state["metrics_progress"] = {"apply-code-fixes": {"started_at": _OLD}}
        state.save()
        assert _status(plan)["phases"]["apply-code-fixes"]["status"] == "pending"

    def test_a_review_stale_after_implement_takes_its_cycles_batches_with_it(
        self, tmp_path
    ):
        """The Codex run of 2026-10-02 (`stale/` repro): code-review and
        apply-code-fixes completed 2026-10-01T23:30/23:40, implement
        2026-10-02T11:59, `apply-fix-batch-0` completed. `status` showed
        `code-review: pending, apply-code-fixes: in_progress` and
        `next-command --after review-code` answered
        `--apply-code-fixes <plan> --resume`, which replays the stale cycle's
        recorded decisions — exactly what the predate rule exists to prevent."""
        from utils import next_phase
        plan = _workflow_plan(tmp_path)
        _put(plan, "code-review", "grouped.json", _ONE_FINDING)
        _record(plan, completed=_BEFORE_CODE_REVIEW + ("code-review", "apply-code-fixes"),
                tasks={"apply-fix-batch-0": "completed"})
        state = StateManager(plan)
        state.state["phases_completed"].update({
            "code-review": "2026-10-01T23:30:00.000000",
            "apply-code-fixes": "2026-10-01T23:40:00.000000",
            "implement": "2026-10-02T11:59:24.913927",
        })
        state.state["metrics"] = {"apply-code-fixes": [
            {"label": "apply-fix-batch-0", "timestamp": "2026-10-01T23:35:00.000000"}]}
        state.save()
        payload = _status(plan)
        assert payload["phases"]["code-review"]["status"] == "pending"
        assert payload["phases"]["apply-code-fixes"]["status"] == "pending"
        assert payload["suggested_next"] == "code-review"
        # The gate off, both answers name the stale review itself — as the
        # notice beside them does — and neither carries `--resume`:
        # `--after review-code` used to walk past the stale start, and
        # `--after apply-code-fixes` answered `complete` (a stale phase was
        # not "started" to the earlier-phase rule).
        for after in ("review-code", "apply-code-fixes"):
            answer = next_phase.next_phase_commands(
                "plans/x.md", "codex", None, None, after_mode=after, plan_path=plan)
            assert answer.phase == "code-review", after
            assert answer.command.endswith("--review-code plans/x.md"), after
        # Under the gate a stale finished phase is still not on record.
        assert next_phase.next_phase_commands(
            "plans/x.md", "codex", None, "review-code", plan_path=plan) is None


class TestUnappliedGroupsNoticeOnItsPhase:
    def test_notice_is_recorded_on_apply_task_suggestions_and_list_unchanged(
        self, tmp_path
    ):
        plan = _workflow_plan(tmp_path)
        _review_tasks_actionable(plan)
        _record(plan, completed=("review-tasks",))
        payload = _status(plan)
        assert payload["notices"] == [
            "review-tasks produced 1 unapplied task suggestion groups. "
            "Run --apply-task-suggestions to apply them, or proceed to "
            "--implement to skip."
        ]
        assert payload["phases"]["apply-task-suggestions"]["notice"] == payload["notices"][0]

    def test_no_notice_no_entry_key(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-tasks",))
        payload = _status(plan)
        assert "notices" not in payload
        assert "notice" not in payload["phases"]["apply-task-suggestions"]


class TestUpstreamSkipNoticePrecedence:
    """Completed wins over skipped, as everywhere else in the module."""

    def _skip(self, plan: Path, phase: str) -> None:
        state = StateManager(plan)
        state.mark_phase_skipped(phase, f"{prereq.SKIP_PREREQUISITES_REASON} ({phase})")
        state.save()

    def test_plain_skip_keeps_notice_byte_for_byte(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        self._skip(plan, "apply-suggestions")
        reason = f"{prereq.SKIP_PREREQUISITES_REASON} (apply-suggestions)"
        assert prereq.upstream_skip_notices(StateManager(plan)) == [
            f"[PREREQUISITE_SKIPPED_UPSTREAM apply-suggestions: {reason}]"
        ]

    def test_skipped_then_completed_has_no_notice(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        self._skip(plan, "apply-suggestions")
        state = StateManager(plan)
        state.mark_phase_completed("apply-suggestions")
        state.save()
        state = StateManager(plan)
        # Both records remain on file; completed must win.
        assert state.is_phase_skipped("apply-suggestions")
        assert state.is_phase_completed("apply-suggestions")
        assert prereq.upstream_skip_notices(state) == []


class TestReviewStaleAfterImplement:
    """A code review (and the fixes applied from it) that predates implement's
    latest completion judged an earlier state of the code (#26, 2026-10-02:
    the Codex run reviewed an empty repo, then implement ran)."""

    def _stamp(self, plan, **stamps):
        state = StateManager(plan)
        for phase, stamp in stamps.items():
            state.state["phases_completed"][phase.replace("_", "-")] = stamp
        state.save()

    def _plan(self, tmp_path, **stamps):
        plan = _workflow_plan(tmp_path)
        _put(plan, "code-review", "grouped.json", _ONE_FINDING)
        _record(plan, completed=_BEFORE_CODE_REVIEW + ("code-review", "apply-code-fixes"))
        self._stamp(plan, **stamps)
        return plan

    def test_reviews_older_than_implement_read_pending(self, tmp_path):
        plan = self._plan(
            tmp_path, code_review=_OLD, apply_code_fixes=_OLD, implement=_NEW)
        payload = _status(plan)
        assert payload["phases"]["code-review"]["status"] == "pending"
        assert payload["phases"]["apply-code-fixes"]["status"] == "pending"
        assert payload["suggested_next"] == "code-review"
        state = StateManager(plan)
        assert not state.is_phase_completed("code-review")
        assert not state.is_phase_completed("apply-code-fixes")
        assert state.phase_completion_is_stale("code-review")
        # the record itself is untouched
        assert state.state["phases_completed"]["code-review"] == _OLD

    def test_the_next_step_after_implement_is_the_review_not_the_fixes(self, tmp_path):
        from utils import next_phase
        plan = self._plan(
            tmp_path, code_review=_OLD, apply_code_fixes=_OLD, implement=_NEW)
        answer = next_phase.next_phase_commands(
            "plans/x.md", "claude-code", None, "implement", plan_path=plan)
        assert answer.phase == "code-review"

    def test_reviews_newer_than_implement_stay_completed(self, tmp_path):
        plan = self._plan(
            tmp_path, implement=_OLD, code_review=_NEW, apply_code_fixes=_NEW)
        payload = _status(plan)
        assert payload["phases"]["code-review"]["status"] == "completed"
        assert payload["phases"]["apply-code-fixes"]["status"] == "completed"
        assert payload["suggested_next"] is None

    @pytest.mark.parametrize("bad", ["not-a-date", "", None, 12345])
    def test_an_unreadable_date_keeps_the_completion(self, tmp_path, bad):
        plan = self._plan(tmp_path, code_review=bad, implement=_NEW)
        assert StateManager(plan).is_phase_completed("code-review")

    def test_only_the_two_reviewing_phases_go_stale(self, tmp_path):
        plan = self._plan(tmp_path, generate_tasks=_OLD, implement=_NEW)
        state = StateManager(plan)
        assert state.is_phase_completed("generate-tasks")
        assert not state.phase_completion_is_stale("generate-tasks")

    def test_a_plain_review_code_is_not_refused_as_completed(self, tmp_path):
        """The completed-phase guard (exit 2) and its artifact auto-heal both
        used to resurrect the stale review; a plain run takes the --force path."""
        import subprocess
        plan = self._plan(tmp_path, code_review=_OLD, implement=_NEW)
        result = subprocess.run(
            [sys.executable, str(Path(prereq.__file__).parent / "code_review_orchestrator.py"),
             "--plan-file", str(plan), "--models", "test:model"],
            capture_output=True, text=True, timeout=30, encoding="utf-8",
        )
        assert "already been completed" not in result.stdout
        assert "was complete but not marked" not in result.stdout
        assert "re-running it" in result.stdout

    @staticmethod
    def _reaggregate(plan):
        import subprocess
        return subprocess.run(
            [sys.executable, str(Path(prereq.__file__).parent / "code_review_orchestrator.py"),
             "--plan-file", str(plan), "--reaggregate"],
            capture_output=True, text=True, timeout=30, encoding="utf-8",
        )

    def test_a_reaggregate_of_a_stale_review_is_refused_not_restamped(self, tmp_path):
        """Adversarial review, 2026-10-02: the reaggregate branch returned
        before the stale guard and ended in `mark_phase_completed`, so a review
        of the empty repository was re-stamped newer than implement with no
        model run, and apply-code-fixes applied its findings."""
        plan = self._plan(tmp_path, code_review=_OLD, implement=_NEW)
        result = self._reaggregate(plan)
        assert result.returncode == 3, result.stdout + result.stderr
        assert "completed before the latest implement" in result.stdout
        assert "--force" in result.stdout
        assert "Reaggregation complete" not in result.stdout
        state = StateManager(plan)
        assert state.state["phases_completed"]["code-review"] == _OLD
        assert state.phase_completion_is_stale("code-review")

    def test_a_reaggregate_of_a_current_review_still_runs(self, tmp_path):
        plan = self._plan(tmp_path, implement=_OLD, code_review=_NEW)
        result = self._reaggregate(plan)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "completed before the latest implement" not in result.stdout
        assert "=== REAGGREGATE MODE ===" in result.stdout


class _JailHost:
    """A scratch git repo whose runner is the mock codex with a broken
    standard sandbox, a throwaway $CODEX_HOME, and the fake Landlock wrapper
    scripted per test: the early warning is then the dispatch's own decision
    (`subagent_executor.repo_wide_write_refusal`), probed for real through
    the mock, never a resolver stub."""

    def __init__(self, tmp_path, monkeypatch):
        from tests.test_plugin_landlock_executor import FakeWrapper
        from tests.test_plugin_landlock_acceptance import MOCK_LLM, SCENARIO_MATCHER
        from utils import provider_registry as registry
        from utils import subagent_executor as executor
        import shutil as _shutil

        self.root = Path(os.path.realpath(tmp_path))
        self.monkeypatch = monkeypatch
        home = self.root / "home"
        self.codex_home = home / ".codex"
        self.codex_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("CODEX_HOME", str(self.codex_home))
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=t",
                        "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
                        "-m", "root"], check=True)
        monkeypatch.chdir(self.repo)
        self.plan = self.repo / "plans" / "x.md"
        self.plan.parent.mkdir()
        self.plan.write_text("# plan\n", encoding="utf-8")
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        # The mock answers as whichever CLI its launcher is named: codex (the
        # runner under test) and opencode (so the opencode host's session rung
        # resolves to a real opencode spec, which has no sandbox probe).
        for name in ("codex", "opencode"):
            (bin_dir / name).symlink_to(MOCK_LLM)
        _shutil.copyfile(SCENARIO_MATCHER, bin_dir / SCENARIO_MATCHER.name)
        uv = _shutil.which("uv")
        if uv:
            monkeypatch.setenv("MULTI_LLM_UV", uv)
        monkeypatch.setenv("PATH", os.pathsep.join(
            [str(bin_dir), str(Path(sys.executable).parent), "/usr/bin", "/bin"]))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
        monkeypatch.setenv("MOCK_LLM_SANDBOX_BROKEN", "1")
        for name in ("MULTI_LLM_HOST", "MULTI_LLM_HOST_SESSION_MODEL",
                     "MULTI_LLM_PROVIDERS_CONFIG", "MULTI_LLM_CODEX_SKIP_RUNG",
                     "MULTI_LLM_HOST_SANDBOX_POSTURE"):
            monkeypatch.delenv(name, raising=False)
        self.wrapper = FakeWrapper(self.root / "wrapper", monkeypatch)
        self._registry = registry
        executor.reset_version_cache()
        self.reset()

    def reset(self):
        self._registry._config = None
        self._registry._config_key = None

    def configure(self, runner=None, fallback=None):
        lines = []
        if runner:
            lines += ["driver:", f"  subagent_runner: {runner}"]
        if fallback:
            lines += ["providers:", "  codex:", f"    sandbox_fallback: {fallback}"]
        cfg = self.repo / ".multi-llm"
        cfg.mkdir(exist_ok=True)
        (cfg / "providers.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.reset()

    def status(self, **kw):
        self.reset()
        return prereq.get_workflow_status(
            StateManager(self.plan), self.plan, output_dir_path(self.plan), **kw)


@pytest.fixture
def jailhost(tmp_path, monkeypatch):
    host = _JailHost(tmp_path, monkeypatch)
    yield host
    host.reset()


@pytest.mark.config_override  # the scratch repo's own .multi-llm/providers.yaml is the point
class TestEarlySandboxWarning:
    """#1: implement / apply-code-fixes are refused on a codex host whose own
    sandbox cannot start and whose plugin-landlock rung is unusable; the
    workflow view says so from the dispatch's own decision, at init and before
    any phase has run (Codex racing run, 2026-10-02; probe-based since step
    3b, when a recorded `plugin-landlock` rung stopped meaning a refusal)."""

    def test_an_unusable_jail_warns_before_any_phase_ran(self, jailhost):
        jailhost.configure(runner="codex:gpt-5.5")
        jailhost.wrapper.set(probe="absent")
        payload = jailhost.status(host_id="claude-code")
        warning = payload["phases"]["implement"]["warning"]
        assert warning == payload["phases"]["apply-code-fixes"]["warning"]
        assert payload["notices"] == [warning]
        assert warning.startswith("implement and apply-code-fixes will be refused on this machine for driver.subagent_runner 'codex:gpt-5.5': ")
        assert "The plugin's Landlock jail cannot run here either" in warning
        assert "Landlock ABI none (ENOSYS)" in warning
        assert warning.endswith(prereq.REPO_WIDE_REMEDY)
        assert "Nothing was dispatched." not in warning
        assert "Refusing to dispatch" not in warning
        # One standard probe and one rung probe, through the mock, no model.
        assert len(jailhost.wrapper.records(probe=True)) == 1
        assert jailhost.wrapper.records(probe=False) == []

    @pytest.mark.parametrize("leg", ["standard-unreadable", "rung-hang"])
    def test_the_warnings_probe_prints_nothing_whatever_it_decides(
        self, jailhost, monkeypatch, capsys, leg,
    ):
        """`_detect_sandbox` prints the preflight's own notes — on an
        unreadable standard probe "… recording sandbox 'unknown'. Dispatching
        anyway …", which is wrong for a reader that dispatches nothing — so
        the warning's probe swallows both streams. An unreadable standard
        probe draws no warning (a warning that may be wrong is worse than
        none), and so does a rung probe that hangs: the advisory reader runs
        its probes under `ADVISORY_PROBE_BUDGET_S`, not the dispatch's
        `SANDBOX_PROBE_TIMEOUT_S` (10 s), so `status` stays sub-second and
        says nothing it did not measure (adversarial review, 2026-10-02).
        Measured 2026-10-02: the next-command emitters' stderr had to stay
        empty on the codex routes."""
        from utils import subagent_executor as executor
        jailhost.configure(runner="codex:gpt-5.5")
        if leg == "standard-unreadable":
            monkeypatch.setattr(executor, "_run_sandbox_probe", lambda *a, **k: None)
        else:
            jailhost.wrapper.set(probe="hang")
            monkeypatch.setattr(executor, "ADVISORY_PROBE_BUDGET_S", 0.5)
        capsys.readouterr()
        started = time.monotonic()
        payload = jailhost.status(host_id="claude-code")
        elapsed = time.monotonic() - started
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == "", captured
        assert "warning" not in payload["phases"]["implement"]
        if leg == "rung-hang":
            assert elapsed < 4, elapsed  # the budget, never the dispatch's 10 s

    def test_a_budget_cut_read_leaves_nothing_for_the_dispatch_to_inherit(self, jailhost, monkeypatch):
        """The reader's cut-short measurement is cached nowhere: a dispatch
        decision in the same process probes for itself, under its own
        timeout, and takes the rung the wrapper now grants."""
        from utils import subagent_executor as executor
        from utils.providers.base import PROFILE_AGENT_WRITE, WRITE_SCOPE_REPO_WIDE, JailRequest
        jailhost.configure(runner="codex:gpt-5.5")
        jailhost.wrapper.set(probe="hang")
        monkeypatch.setattr(executor, "ADVISORY_PROBE_BUDGET_S", 0.5)
        assert "warning" not in jailhost.status(host_id="claude-code")["phases"]["implement"]
        assert executor._SANDBOX_PROBE_CACHE == {} and executor._SANDBOX_DECISIONS == {}
        jailhost.wrapper.set(probe="deny")
        request = JailRequest(
            repo_root=str(jailhost.repo), kind="implement", write_scope=WRITE_SCOPE_REPO_WIDE,
            output_dir=str(jailhost.plan.parent / "x"),
        )
        assert executor._detect_sandbox(
            "codex", key="driver.subagent_runner", spec="codex:gpt-5.5",
            profile=PROFILE_AGENT_WRITE, anchor=str(jailhost.plan), jail_request=request,
        ) == "plugin-landlock"
        assert len(jailhost.wrapper.records(probe=True)) == 2

    def test_the_advisory_probe_hands_the_canary_no_target_in_the_source_tree(self, jailhost):
        """`status` and `init` are advisory readers: their probe must create
        nothing in the user's source tree outside `.multi-llm/`. The
        repo-wide canary used to take the enumeration's first directory as a
        positive control — `.github/.multi-llm-canary-NNN` on every `status`
        poll (adversarial review, 2026-10-02, strace of one `status`)."""
        (jailhost.repo / ".github").mkdir()
        jailhost.configure(runner="codex:gpt-5.5")
        payload = jailhost.status(host_id="claude-code")
        assert "warning" not in payload["phases"]["implement"]
        [probe] = jailhost.wrapper.records(probe=True)
        assert probe["env"]["MULTI_LLM_CANARY_WRITE_DIR"] is None
        assert probe["env"]["MULTI_LLM_CANARY_WRITE_FILE"] is None
        # Its negative controls live outside the repository (root grant, 2026-10-05).
        workdir = probe["env"]["MULTI_LLM_CANARY_WORKDIR"]
        assert workdir and not workdir.startswith(os.path.realpath(jailhost.repo) + os.sep)

    def test_a_usable_jail_stays_silent(self, jailhost):
        jailhost.configure(runner="codex:gpt-5.5")
        payload = jailhost.status(host_id="claude-code")
        assert "warning" not in payload["phases"]["implement"]
        assert "notices" not in payload
        assert len(jailhost.wrapper.records(probe=True)) == 1

    @pytest.mark.parametrize("case", ["contained", "off", "opt-in-off", "failed-probe"])
    def test_every_other_refusal_of_the_rung_warns_too(self, jailhost, monkeypatch, case):
        if case in ("contained", "off"):
            jailhost.configure(runner="codex:gpt-5.5", fallback=case)
        else:
            jailhost.configure(runner="codex:gpt-5.5")
        if case == "opt-in-off":
            from utils.providers import codex as codex_module
            monkeypatch.setattr(codex_module, "WRITE_RUNG_OPT_IN", False)
        if case == "failed-probe":
            jailhost.wrapper.set(probe="not_denied")
        warning = jailhost.status(host_id="claude-code")["phases"]["implement"]["warning"]
        expected = {
            "contained": "providers.codex.sandbox_fallback=contained",
            "off": "providers.codex.sandbox_fallback=off",
            "opt-in-off": "write jobs are not enabled on it",
            "failed-probe": "failed its self-test",
        }[case]
        assert expected in warning, warning
        assert warning.endswith(prereq.REPO_WIDE_REMEDY)

    def test_a_runner_that_is_not_codex_draws_no_warning_and_no_probe(self, jailhost):
        jailhost.configure(runner="claude-code:sonnet")
        jailhost.wrapper.set(probe="absent")
        payload = jailhost.status(host_id="claude-code")
        assert "warning" not in payload["phases"]["implement"]
        assert jailhost.wrapper.records() == []

    def test_a_finished_implement_and_apply_code_fixes_draw_no_warning(self, jailhost):
        jailhost.configure(runner="codex:gpt-5.5")
        jailhost.wrapper.set(probe="absent")
        _put(jailhost.plan, "code-review", "grouped.json", _ONE_FINDING)
        _record(jailhost.plan, completed=_BEFORE_CODE_REVIEW + ("code-review", "apply-code-fixes"))
        phases = jailhost.status(host_id="claude-code")["phases"]
        assert "warning" not in phases["implement"]
        assert "warning" not in phases["apply-code-fixes"]
        assert jailhost.wrapper.records() == []  # nothing open: nothing probed

    def test_an_unreadable_config_draws_no_warning(self, jailhost, monkeypatch):
        from utils import provider_registry

        def boom(*a, **kw):
            raise RuntimeError("broken override")

        monkeypatch.setattr(provider_registry, "resolve_subagent_runner", boom)
        assert "warning" not in jailhost.status(host_id="claude-code")["phases"]["implement"]

    def test_the_next_step_that_points_at_implement_carries_it(self, jailhost):
        from utils import next_phase
        jailhost.configure(runner="codex:gpt-5.5")
        jailhost.wrapper.set(probe="absent")
        _record(jailhost.plan, completed=("review-plan", "apply-suggestions", "generate-tasks"))
        jailhost.reset()
        answer = next_phase.next_phase_commands(
            "plans/x.md", "claude-code", None, "generate-tasks", plan_path=jailhost.plan)
        assert answer.phase == "implement" and answer.warning
        lines = next_phase.render_next_phase_lines(answer)
        assert lines[0] == f"{next_phase.RUNNER_WARNING_MARKER} {answer.warning}"
        assert lines[1].startswith(next_phase.NEXT_COMMAND_MARKER + " ")
        assert "warning" not in answer.to_dict()  # user_next_step is unchanged

    def test_init_prints_the_warning_for_the_host_it_ran_from(self, jailhost, capsys):
        """A Codex user learns at `init`, not at phase 6: with the jail
        unusable the smoke-test block is followed by one [RUNNER_WARNING]
        line naming the refusal and its remedy; with the jail usable, none."""
        import init_config
        from utils import next_phase
        jailhost.wrapper.set(probe="absent")
        assert init_config.main(["--dir", str(jailhost.repo), "--host", "codex", "--force"]) == 0
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.startswith(next_phase.RUNNER_WARNING_MARKER)]
        assert len(lines) == 1, out
        assert "implement and apply-code-fixes will be refused on this machine" in lines[0]
        assert "'codex:" in lines[0] and lines[0].endswith(prereq.REPO_WIDE_REMEDY)
        assert out.index("Smoke-test it") < out.index(next_phase.RUNNER_WARNING_MARKER)

        jailhost.wrapper.set(probe="deny")
        from utils import subagent_executor as executor
        executor._SANDBOX_DECISIONS.clear()
        executor._SANDBOX_PROBE_CACHE.clear()
        assert init_config.main(["--dir", str(jailhost.repo), "--host", "codex", "--force"]) == 0
        assert next_phase.RUNNER_WARNING_MARKER not in capsys.readouterr().out


@pytest.mark.config_override
class TestEarlySandboxWarningResolvesTheReadersHost:
    """Adversarial review, 2026-10-02 (`repro_warning.py`): the warning
    resolved the runner for `$MULTI_LLM_HOST`-else-claude-code, while every
    server and shim declares the host only as `--host codex` on the argv. On
    the claude-code entry the ladder has no `session` rung, so the codex
    session's `codex:gpt-6-luna` fell to `claude-code:sonnet` and the warning
    never fired on either tool route — while the dispatch, which resolves
    `--host` off the argv, refused. No resolver monkeypatch here: the real
    ladder, the real host entries, the real probe through the mock."""

    @pytest.fixture
    def plan(self, jailhost, monkeypatch):
        monkeypatch.setenv("MULTI_LLM_HOST_SESSION_MODEL", "gpt-6-luna")
        jailhost.wrapper.set(probe="absent")
        _record(jailhost.plan, completed=("review-plan", "apply-suggestions", "generate-tasks"))
        return jailhost

    def test_the_dispatch_and_the_warning_resolve_the_same_runner(self, plan):
        from utils.provider_registry import resolve_subagent_runner
        spec, layer = resolve_subagent_runner("full", anchor=str(plan.plan), host_id="codex")
        assert (spec, layer) == ("codex:gpt-6-luna", "session")
        warning = plan.status(host_id="codex")["phases"]["implement"]["warning"]
        assert "'codex:gpt-6-luna'" in warning and "Landlock jail cannot run here" in warning

    def test_the_opencode_route_resolves_its_own_session_model_and_draws_none(self, plan):
        from utils.provider_registry import resolve_subagent_runner
        spec, layer = resolve_subagent_runner("full", anchor=str(plan.plan), host_id="opencode")
        assert spec == "opencode:gpt-6-luna" and layer == "session"
        assert "warning" not in plan.status(host_id="opencode")["phases"]["implement"]

    def test_the_opencode_route_warns_for_a_configured_codex_runner(self, plan):
        plan.configure(runner="codex:gpt-5.5")
        warning = plan.status(host_id="opencode")["phases"]["implement"]["warning"]
        assert "'codex:gpt-5.5'" in warning

    def test_the_claude_code_host_has_no_session_rung_and_draws_none(self, plan):
        assert "warning" not in plan.status(host_id="claude-code")["phases"]["implement"]

    def test_with_no_host_handed_in_the_argv_decides_as_the_executor_does(
        self, plan, monkeypatch
    ):
        monkeypatch.setattr(sys, "argv", ["multi_llm.py", "status", str(plan.plan), "--host", "codex"])
        assert "warning" in plan.status()["phases"]["implement"]
        monkeypatch.setattr(sys, "argv", ["multi_llm.py", "status", str(plan.plan)])
        assert "warning" not in plan.status()["phases"]["implement"]

    def test_every_reader_hands_its_host_through(self, plan, capsys):
        from utils import next_phase, run_status
        with pytest.raises(SystemExit) as exit_:  # --status sys.exit()s
            prereq.main(["--plan-file", str(plan.plan), "--status", "--host", "codex"])
        assert exit_.value.code == 0
        assert "notices" in json.loads(capsys.readouterr().out)
        assert run_status.main(
            ["--plan-file", str(plan.plan), "--view", "workflow", "--host", "codex"]) == 0
        assert "notices" in json.loads(capsys.readouterr().out)
        assert next_phase.main([str(plan.plan), "--after", "generate-tasks", "--host", "codex"]) == 0
        out = capsys.readouterr().out
        assert out.startswith(next_phase.RUNNER_WARNING_MARKER + " ")
        assert "'codex:gpt-6-luna'" in out


class TestStaleCompletionIsFlagged:
    """S3 (adversarial review, 2026-10-02): a stale completion reads `pending`
    and the payload says so, so `status`'s run view can tell a done run whose
    phase went stale from one not yet marked, and never offers the
    `--reaggregate` that would re-stamp it."""

    def _plan(self, tmp_path, **stamps):
        plan = _workflow_plan(tmp_path)
        _put(plan, "code-review", "grouped.json", _ONE_FINDING)
        _record(plan, completed=_BEFORE_CODE_REVIEW + ("code-review", "apply-code-fixes"))
        state = StateManager(plan)
        for phase, stamp in stamps.items():
            state.state["phases_completed"][phase.replace("_", "-")] = stamp
        state.save()
        return plan

    def test_a_stale_review_and_its_fixes_carry_the_flag_and_both_dates(self, tmp_path):
        plan = self._plan(tmp_path, code_review=_OLD, apply_code_fixes=_OLD, implement=_NEW)
        phases = _status(plan)["phases"]
        for phase in ("code-review", "apply-code-fixes"):
            assert phases[phase] == {
                "status": "pending", "stale": True,
                "completed_at": _OLD, "implemented_at": _NEW,
            }, phase

    def test_a_plain_pending_phase_carries_no_flag(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=_BEFORE_CODE_REVIEW)
        assert _status(plan)["phases"]["code-review"] == {"status": "pending"}

    def test_a_fresh_review_is_completed_not_stale(self, tmp_path):
        plan = self._plan(tmp_path, implement=_OLD, code_review=_NEW, apply_code_fixes=_NEW)
        assert "stale" not in _status(plan)["phases"]["code-review"]


class TestImplementWithEveryTaskDoneIsInProgress:
    """S6 (adversarial review, 2026-10-02): every task completed but the phase
    unmarked — `driver.verify_command` failed, or the run died before
    `mark_phase_completed` — read `pending`, and the earlier-phase rule routed
    a later phase's answer to a bare `--implement <plan>`, which re-dispatches
    every task (`format_batches_for_output` filters only under `--resume`)."""

    def _plan(self, tmp_path, tasks):
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan", "apply-suggestions", "generate-tasks"),
                tasks=tasks)
        return plan

    def test_done_equals_total_reads_in_progress(self, tmp_path):
        entry = _status(self._plan(tmp_path, {"T001": "completed", "T002": "completed"}))
        assert entry["phases"]["implement"]["status"] == "in_progress"
        assert (entry["phases"]["implement"]["tasks_done"],
                entry["phases"]["implement"]["tasks_total"]) == (2, 2)
        assert entry["suggested_next"] == "implement"

    def test_the_answer_after_a_later_phase_carries_resume(self, tmp_path):
        from utils import next_phase
        plan = self._plan(tmp_path, {"T001": "completed", "T002": "completed"})
        answer = next_phase.next_phase_commands(
            "plans/x.md", "claude-code", None, None, after_mode="review-code", plan_path=plan)
        assert answer.phase == "implement"
        assert answer.command.endswith("--implement plans/x.md --resume")

    def test_nothing_done_stays_pending(self, tmp_path):
        """A seeded list with no completed task (a stop before the first
        dispatch) is not progress a bare re-run would redo."""
        entry = _status(self._plan(tmp_path, {"T001": "pending", "T002": "failed"}))
        assert entry["phases"]["implement"]["status"] == "pending"
        assert entry["phases"]["implement"]["tasks_total"] == 2


class TestStartedPhaseReadsInProgress:
    """The native leg's started record (`StateManager.mark_phase_started`).

    A write phase's orchestrator records the phase *started* when it emits its
    batch document; nothing else is on record until the host's first batch
    completes. Until 2026-10-05 apply-suggestions and implement recorded
    themselves *completed* there instead (racing re-run, 2026-10-04: implement
    stamped 10:43:30, six minutes before its last task), and the other two
    read `pending` — a bare re-invocation, not the `--resume` a host that died
    before its first batch needs.
    """

    def _started(self, plan, *phases):
        state = StateManager(plan)
        for phase in phases:
            state.mark_phase_started(phase)
        state.save()

    def test_the_api_round_trips_and_completion_clears_it(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        state = StateManager(plan)
        assert not state.is_phase_started("implement")
        state.mark_phase_started("implement")
        state.save()
        reloaded = StateManager(plan)
        assert reloaded.is_phase_started("implement")
        assert reloaded.phase_started_at("implement")
        reloaded.mark_phase_completed("implement")
        assert not reloaded.is_phase_started("implement")
        assert "implement" not in reloaded.state["phases_started"]
        assert reloaded.is_phase_completed("implement")

    def test_a_skip_clears_it_too(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        state = StateManager(plan)
        state.mark_phase_started("apply-suggestions")
        state.mark_phase_skipped("apply-suggestions", "user chose to skip")
        assert not state.is_phase_started("apply-suggestions")
        assert state.is_phase_skipped("apply-suggestions")

    def test_a_started_implement_with_no_task_done_is_in_progress(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan", "apply-suggestions", "generate-tasks"),
                tasks={"T001": "pending", "T002": "pending"})
        self._started(plan, "implement")
        entry = _status(plan)["phases"]["implement"]
        assert entry["status"] == "in_progress"
        assert entry["started_at"]
        assert _status(plan)["suggested_next"] == "implement"

    @pytest.mark.parametrize(
        "phase,batch_id,fixture,completed",
        APPLY_PHASE_BATCH_ROWS,
        ids=[row[0] for row in APPLY_PHASE_BATCH_ROWS],
    )
    def test_a_started_apply_phase_with_no_batch_done_is_in_progress(
        self, tmp_path, phase, batch_id, fixture, completed
    ):
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed)
        assert _status(plan)["phases"][phase]["status"] == "pending"
        self._started(plan, phase)
        entry = _status(plan)["phases"][phase]
        assert entry["status"] == "in_progress"
        assert entry["started_at"]
        others = {p for p, _b, _f, _c in APPLY_PHASE_BATCH_ROWS} - {phase}
        for other in others:
            assert _status(plan)["phases"][other]["status"] != "in_progress"

    def test_the_next_command_for_a_started_phase_carries_resume(self, tmp_path):
        from utils import next_phase
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {
            "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]})
        _record(plan, completed=("review-plan",))
        self._started(plan, "apply-suggestions")
        answer = next_phase.next_phase_commands(
            "plans/x.md", "claude-code", None, None, plan_path=plan)
        assert answer.phase == "apply-suggestions"
        assert answer.command.endswith("--apply-suggestions plans/x.md --resume")

    @pytest.mark.parametrize(
        "phase,batch_id,fixture,completed",
        APPLY_PHASE_BATCH_ROWS,
        ids=[row[0] for row in APPLY_PHASE_BATCH_ROWS],
    )
    def test_a_start_older_than_the_review_marks_nothing(
        self, tmp_path, phase, batch_id, fixture, completed
    ):
        """An abandoned run from before a review re-run is that cycle's."""
        plan = _workflow_plan(tmp_path)
        fixture(plan)
        _record(plan, completed=completed)
        self._started(plan, phase)
        review = prereq.PHASE_PREREQUISITES[phase][0]
        _date_cycle(plan, phase, review, _NEW)
        state = StateManager(plan)
        state.state["phases_started"][phase] = _OLD
        state.save()
        assert _status(plan)["phases"][phase]["status"] == "pending"
        state.state["phases_started"][phase] = "2026-01-03T10:00:00"
        state.save()
        assert _status(plan)["phases"][phase]["status"] == "in_progress"

    def test_a_start_older_than_the_completion_never_reads_started(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        state = StateManager(plan)
        state.mark_phase_started("apply-suggestions")
        # A stale started key left by hand beside a LATER completion is ignored:
        # the completion is the newer record, so the phase finished.
        state.state["phases_started"]["apply-suggestions"] = "2026-01-02T09:00:00"
        state.state.setdefault("phases_completed", {})["apply-suggestions"] = (
            "2026-01-02T10:00:00")
        state.save()
        assert not StateManager(plan).is_phase_started("apply-suggestions")
        assert StateManager(plan).is_phase_completed("apply-suggestions")
        _put(plan, "review-plan", "validation.json", {
            "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]})
        _record(plan, completed=("review-plan",))
        assert _status(plan)["phases"]["apply-suggestions"]["status"] == "completed"

    def test_a_start_newer_than_an_earlier_completion_reads_started(self, tmp_path):
        """A second review cycle (reviewer repro, 2026-10-05): cycle 1's
        apply-suggestions completed, `review-plan --force` completed later, and
        the host re-runs apply-suggestions natively. The emit stamps a start
        that is NEWER than the cycle-1 completion, so the phase is mid-loop —
        not `completed` on the old stamp, and never `suggested_next:
        generate-tasks` while the host is still applying batches."""
        plan = _workflow_plan(tmp_path)
        _put(plan, "review-plan", "validation.json", {
            "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]})
        _record(plan, completed=("review-plan", "apply-suggestions"))
        time.sleep(0.01)
        _record(plan, completed=("review-plan",))  # cycle 2: --force re-run
        time.sleep(0.01)
        self._started(plan, "apply-suggestions")  # cycle 2: native emit

        state = StateManager(plan, read_only=True)
        assert state.is_phase_started("apply-suggestions")
        assert not state.is_phase_completed("apply-suggestions")
        # The raw cycle-1 record is kept: nothing rewrites history.
        assert "apply-suggestions" in state.state["phases_completed"]
        status = _status(plan)
        assert status["phases"]["apply-suggestions"]["status"] == "in_progress"
        assert status["phases"]["apply-suggestions"]["started_at"]
        assert status["suggested_next"] == "apply-suggestions"

    def test_a_restarted_implement_reads_started_over_its_old_completion(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan", "apply-suggestions",
                                 "generate-tasks", "implement"))
        time.sleep(0.01)
        self._started(plan, "implement")  # implement re-run, native emit
        state = StateManager(plan, read_only=True)
        assert state.is_phase_started("implement")
        assert not state.is_phase_completed("implement")
        status = _status(plan)
        assert status["phases"]["implement"]["status"] == "in_progress"
        assert status["suggested_next"] == "implement"

    def test_a_restarted_apply_code_fixes_staled_by_implement_reads_started(self, tmp_path):
        """reproA/repro_stale_acf.py: cycle 1 complete, implement re-run
        completes later (apply-code-fixes now `stale`), code-review re-runs,
        and the host re-emits apply-code-fixes natively: `in_progress`."""
        plan = _workflow_plan(tmp_path)
        _put(plan, "code-review", "grouped.json", _ONE_FINDING)
        for phase in ("review-plan", "apply-suggestions", "generate-tasks",
                      "implement", "code-review", "apply-code-fixes"):
            _record(plan, completed=(phase,))
            time.sleep(0.01)
        _record(plan, completed=("implement",))
        time.sleep(0.01)
        _record(plan, completed=("code-review",))
        time.sleep(0.01)
        assert _status(plan)["phases"]["apply-code-fixes"].get("stale")
        self._started(plan, "apply-code-fixes")
        state = StateManager(plan, read_only=True)
        assert state.is_phase_started("apply-code-fixes")
        status = _status(plan)
        assert status["phases"]["apply-code-fixes"]["status"] == "in_progress"
        assert status["suggested_next"] == "apply-code-fixes"

    def test_the_completion_write_after_a_restart_is_the_newest_record(self, tmp_path):
        """The closing `--mark-completed` of the re-run wins again."""
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan", "apply-suggestions"))
        time.sleep(0.01)
        self._started(plan, "apply-suggestions")
        state = StateManager(plan)
        assert state.is_phase_started("apply-suggestions")
        state.mark_phase_completed("apply-suggestions")
        state.save()
        reloaded = StateManager(plan, read_only=True)
        assert reloaded.is_phase_completed("apply-suggestions")
        assert not reloaded.is_phase_started("apply-suggestions")


class TestImplementCountsTheWholeDecomposition:
    """Racing re-run 3 (2026-10-05): mid-implement, `status` read
    `implement in_progress (6/6 tasks)` on Claude Code's native leg with ten
    tasks generated — the native host records a task only once it ran — while
    Codex's executor leg, which seeds every task `pending`, read `8/13`. The
    count is now done/total of the decomposition implement ran last
    (`implement_task_ids`) on both legs."""

    TEN = [f"T{n:03d}" for n in range(1, 11)]

    def _plan(self, tmp_path, tasks, *, started=True, ids=None):
        plan = _workflow_plan(tmp_path)
        _record(plan, completed=("review-plan", "apply-suggestions", "generate-tasks"),
                tasks=tasks)
        state = StateManager(plan)
        state.record_task_titles({task_id: task_id for task_id in (ids or self.TEN)})
        if started:
            state.mark_phase_started("implement")
        state.save()
        return plan

    def test_the_native_leg_mid_run_reads_done_of_all(self, tmp_path):
        plan = self._plan(tmp_path, {f"T{n:03d}": "completed" for n in range(1, 7)})
        entry = _status(plan)["phases"]["implement"]
        assert entry["status"] == "in_progress"
        assert (entry["tasks_done"], entry["tasks_total"]) == (6, 10)

    def test_the_status_text_prints_it(self, tmp_path, capsys):
        from utils import run_status

        plan = self._plan(tmp_path, {f"T{n:03d}": "completed" for n in range(1, 7)})
        run_status.main(["--plan-file", str(plan)])
        out = capsys.readouterr().out
        assert "implement               in_progress  (6/10 tasks)" in out

    def test_a_started_emit_with_nothing_run_yet_reads_zero_of_all(self, tmp_path):
        entry = _status(self._plan(tmp_path, {}))["phases"]["implement"]
        assert (entry["status"], entry["tasks_done"], entry["tasks_total"]) == (
            "in_progress", 0, 10,
        )

    def test_the_executor_legs_seeded_count_is_unchanged(self, tmp_path):
        tasks = {task_id: "pending" for task_id in self.TEN}
        tasks.update({"T001": "completed", "T002": "completed", "T010": "skipped"})
        entry = _status(self._plan(tmp_path, tasks, started=False))["phases"]["implement"]
        assert (entry["tasks_done"], entry["tasks_total"]) == (2, 10)

    def test_an_earlier_decompositions_ids_are_not_counted(self, tmp_path):
        plan = self._plan(
            tmp_path, {"T001": "completed", "T020": "completed"},
            ids=["T001", "T002"],
        )
        entry = _status(plan)["phases"]["implement"]
        assert (entry["tasks_done"], entry["tasks_total"]) == (1, 2)

    def test_a_dry_run_record_alone_shows_no_count(self, tmp_path):
        """`implement --dry-run` records the decomposition and starts nothing;
        a count would make the next-step walk read implement as started."""
        entry = _status(self._plan(tmp_path, {}, started=False))["phases"]["implement"]
        assert entry["status"] == "pending"
        assert "tasks_total" not in entry
