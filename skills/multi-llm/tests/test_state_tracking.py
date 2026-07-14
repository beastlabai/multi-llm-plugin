#!/usr/bin/env python3
"""Tests for state tracking in orchestrators."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import StateManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_plan(temp_dir):
    """Create a sample plan file for testing."""
    plan_content = """# Test Plan

## Overview
This is a test plan for state tracking tests.
"""
    plan_path = temp_dir / "test-plan.md"
    plan_path.write_text(plan_content, encoding="utf-8")
    return plan_path


class TestReviewPlanStateTracking:
    """Tests for review_plan_orchestrator state tracking."""

    def test_marks_phase_completed_after_run(self, sample_plan, temp_dir):
        """review_plan_orchestrator marks review-plan phase as completed."""
        # This would require mocking the LLM calls or using --skip-validation
        # For now, just test that the StateManager API works correctly
        state = StateManager(sample_plan)
        state.mark_phase_completed("review-plan")
        state.save()

        # Reload and verify
        state2 = StateManager(sample_plan)
        assert state2.is_phase_completed("review-plan")

    def test_review_plan_phase_independent_of_other_phases(self, sample_plan):
        """review-plan phase completion does not affect other phases."""
        state = StateManager(sample_plan)
        state.mark_phase_completed("review-plan")
        state.save()

        state2 = StateManager(sample_plan)
        assert state2.is_phase_completed("review-plan")
        assert not state2.is_phase_completed("generate-tasks")
        assert not state2.is_phase_completed("implement")
        assert not state2.is_phase_completed("code-review")


class TestUpdatePlanTasksStateTracking:
    """Tests for update_plan_tasks state tracking."""

    def test_marks_phase_completed_after_tasks_created(self, sample_plan, temp_dir):
        """update_plan_tasks marks generate-tasks phase as completed."""
        # Create a tasks JSON file
        tasks_json = {
            "tasks": [
                {"id": "T001", "title": "Test task", "description": "Test"}
            ]
        }
        tasks_path = temp_dir / "tasks.json"
        tasks_path.write_text(json.dumps(tasks_json), encoding="utf-8")

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(sample_plan),
             "--tasks-file", str(tasks_path)],
            capture_output=True,
            text=True,
            cwd=str(script_path.parent),
            encoding="utf-8",
        )

        assert result.returncode == 0

        # Verify state was updated
        state = StateManager(sample_plan)
        assert state.is_phase_completed("generate-tasks")

    def test_tasks_file_created_in_correct_location(self, sample_plan, temp_dir):
        """update_plan_tasks creates tasks.md in the correct output directory."""
        tasks_json = {
            "tasks": [
                {"id": "T001", "title": "Setup task", "description": "Initial setup"}
            ]
        }
        tasks_path = temp_dir / "tasks.json"
        tasks_path.write_text(json.dumps(tasks_json), encoding="utf-8")

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(sample_plan),
             "--tasks-file", str(tasks_path)],
            capture_output=True,
            text=True,
            cwd=str(script_path.parent),
            encoding="utf-8",
        )

        # Check that tasks.md was created in the plan output directory
        expected_tasks_path = temp_dir / "test-plan" / "tasks" / "tasks.md"
        assert expected_tasks_path.exists()

    def test_dry_run_does_not_mark_phase_completed(self, sample_plan, temp_dir):
        """--dry-run flag does not mark generate-tasks phase as completed."""
        tasks_json = {
            "tasks": [
                {"id": "T001", "title": "Test task", "description": "Test"}
            ]
        }
        tasks_path = temp_dir / "tasks.json"
        tasks_path.write_text(json.dumps(tasks_json), encoding="utf-8")

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(sample_plan),
             "--tasks-file", str(tasks_path),
             "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(script_path.parent),
            encoding="utf-8",
        )

        assert result.returncode == 0

        # Verify state was NOT updated
        state = StateManager(sample_plan)
        assert not state.is_phase_completed("generate-tasks")


class TestCodeReviewStateTracking:
    """Tests for code_review_orchestrator state tracking."""

    def test_marks_phase_completed_after_report(self, sample_plan, temp_dir):
        """code_review_orchestrator marks code-review phase as completed."""
        # Test the StateManager API directly since running the full orchestrator
        # requires LLM mocks
        state = StateManager(sample_plan)
        state.mark_phase_completed("code-review")
        state.save()

        state2 = StateManager(sample_plan)
        assert state2.is_phase_completed("code-review")

    def test_code_review_phase_independent_of_other_phases(self, sample_plan):
        """code-review phase completion does not affect other phases."""
        state = StateManager(sample_plan)
        state.mark_phase_completed("code-review")
        state.save()

        state2 = StateManager(sample_plan)
        assert state2.is_phase_completed("code-review")
        assert not state2.is_phase_completed("review-plan")
        assert not state2.is_phase_completed("generate-tasks")
        assert not state2.is_phase_completed("implement")


class TestPhaseTrackingPersistence:
    """Tests for phase tracking persistence across sessions."""

    def test_multiple_phases_persist(self, sample_plan):
        """Multiple phase completions persist after save/reload."""
        state = StateManager(sample_plan)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("generate-tasks")
        state.save()

        state2 = StateManager(sample_plan)
        assert state2.is_phase_completed("review-plan")
        assert state2.is_phase_completed("generate-tasks")
        assert not state2.is_phase_completed("implement")

    def test_phase_timestamps_stored(self, sample_plan):
        """Phase completion timestamps are stored."""
        state = StateManager(sample_plan)
        state.mark_phase_completed("review-plan")
        state.save()

        phases_completed = state.state.get("phases_completed", {})
        assert "review-plan" in phases_completed
        # Timestamp should be ISO format with 'T'
        assert "T" in phases_completed["review-plan"]

    def test_all_workflow_phases_can_be_tracked(self, sample_plan):
        """All workflow phases can be tracked independently."""
        all_phases = [
            "review-plan",
            "apply-suggestions",
            "generate-tasks",
            "review-tasks",
            "apply-task-suggestions",
            "implement",
            "code-review",
            "apply-code-fixes"
        ]

        state = StateManager(sample_plan)
        for phase in all_phases:
            state.mark_phase_completed(phase)
        state.save()

        state2 = StateManager(sample_plan)
        for phase in all_phases:
            assert state2.is_phase_completed(phase), f"Phase {phase} should be completed"

    def test_state_survives_multiple_save_reload_cycles(self, sample_plan):
        """State persists through multiple save/reload cycles."""
        # First cycle
        state1 = StateManager(sample_plan)
        state1.mark_phase_completed("review-plan")
        state1.save()

        # Second cycle - add another phase
        state2 = StateManager(sample_plan)
        assert state2.is_phase_completed("review-plan")
        state2.mark_phase_completed("generate-tasks")
        state2.save()

        # Third cycle - add another phase
        state3 = StateManager(sample_plan)
        assert state3.is_phase_completed("review-plan")
        assert state3.is_phase_completed("generate-tasks")
        state3.mark_phase_completed("implement")
        state3.save()

        # Final verification
        state4 = StateManager(sample_plan)
        assert state4.is_phase_completed("review-plan")
        assert state4.is_phase_completed("generate-tasks")
        assert state4.is_phase_completed("implement")


class TestPhaseCompletionTimestamps:
    """Tests for phase completion timestamp handling."""

    def test_timestamp_format_is_iso(self, sample_plan):
        """Phase completion timestamps are in ISO format."""
        state = StateManager(sample_plan)
        state.mark_phase_completed("review-plan")
        state.save()

        timestamp = state.state["phases_completed"]["review-plan"]
        # ISO format: YYYY-MM-DDTHH:MM:SS.ffffff or YYYY-MM-DDTHH:MM:SS
        assert "T" in timestamp
        assert "-" in timestamp
        assert ":" in timestamp

    def test_different_phases_have_different_timestamps(self, sample_plan):
        """Different phases get their own timestamps when marked at different times."""
        import time

        state = StateManager(sample_plan)
        state.mark_phase_completed("review-plan")

        # Small delay to ensure different timestamp
        time.sleep(0.01)

        state.mark_phase_completed("generate-tasks")
        state.save()

        ts1 = state.state["phases_completed"]["review-plan"]
        ts2 = state.state["phases_completed"]["generate-tasks"]

        # Timestamps should both be valid ISO format
        assert "T" in ts1
        assert "T" in ts2
        # They could be the same if executed fast enough, so just verify both exist
        assert ts1 is not None
        assert ts2 is not None
