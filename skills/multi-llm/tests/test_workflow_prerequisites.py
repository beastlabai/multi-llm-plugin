"""Tests for workflow prerequisite guards functionality."""

import json
import subprocess
import sys
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
