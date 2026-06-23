"""Tests for plan updater module."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plan_updater import (
    PlanUpdater,
    update_plan_status,
    extract_task_list,
    IMPL_START_MARKER,
    IMPL_END_MARKER,
)


class TestPlanUpdater:
    """Tests for PlanUpdater class."""

    def test_load_plan(self, sample_plan):
        """Test loading a plan file."""
        updater = PlanUpdater(sample_plan)
        assert len(updater.content) > 0

    def test_update_task_status(self, sample_plan):
        """Test updating task status."""
        updater = PlanUpdater(sample_plan)

        updater.update_task_status("T001", "completed")
        assert updater.task_statuses["T001"] == "completed"

    def test_mark_task_completed(self, sample_plan):
        """Test marking task as completed."""
        updater = PlanUpdater(sample_plan)

        updater.mark_task_completed("T001")
        assert updater.task_statuses["T001"] == "completed"

    def test_mark_task_in_progress(self, sample_plan):
        """Test marking task as in progress."""
        updater = PlanUpdater(sample_plan)

        updater.mark_task_in_progress("T002")
        assert updater.task_statuses["T002"] == "in_progress"

    def test_apply_updates_adds_markers(self, sample_plan):
        """Test that apply_updates adds status markers."""
        updater = PlanUpdater(sample_plan)

        updater.update_task_status("T001", "completed")
        updater.apply_updates()

        assert "TASK_STATUS:T001:completed" in updater.content

    def test_apply_updates_adds_summary_section(self, sample_plan):
        """Test that apply_updates adds summary section."""
        updater = PlanUpdater(sample_plan)

        updater.update_task_status("T001", "completed")
        updater.update_task_status("T002", "in_progress")
        updater.apply_updates()

        assert IMPL_START_MARKER in updater.content
        assert IMPL_END_MARKER in updater.content
        assert "## Implementation Status" in updater.content

    def test_apply_updates_idempotent(self, sample_plan):
        """Test that apply_updates is idempotent."""
        updater = PlanUpdater(sample_plan)

        updater.update_task_status("T001", "completed")
        updater.apply_updates()
        content_1 = updater.content

        # Apply again - should update, not duplicate
        updater.apply_updates()
        content_2 = updater.content

        # Should have same number of markers
        assert content_1.count(IMPL_START_MARKER) == content_2.count(IMPL_START_MARKER)

    def test_save_writes_to_file(self, sample_plan):
        """Test that save writes content to file."""
        updater = PlanUpdater(sample_plan)

        updater.update_task_status("T001", "completed")
        updater.apply_updates()
        updater.save()

        # Read back and verify
        content = sample_plan.read_text()
        assert "TASK_STATUS:T001:completed" in content

    def test_get_task_status(self, sample_plan):
        """Test getting task status."""
        updater = PlanUpdater(sample_plan)

        updater.update_task_status("T001", "completed")

        assert updater.get_task_status("T001") == "completed"
        assert updater.get_task_status("T999") is None

    def test_get_all_statuses(self, sample_plan):
        """Test getting all statuses."""
        updater = PlanUpdater(sample_plan)

        updater.update_task_status("T001", "completed")
        updater.update_task_status("T002", "pending")

        statuses = updater.get_all_statuses()
        assert len(statuses) == 2
        assert statuses["T001"] == "completed"


class TestUpdatePlanStatus:
    """Tests for update_plan_status convenience function."""

    def test_update_plan_status(self, sample_plan):
        """Test the convenience function."""
        statuses = {
            "T001": "completed",
            "T002": "in_progress"
        }

        content = update_plan_status(sample_plan, statuses, save=False)

        assert "TASK_STATUS:T001:completed" in content
        assert "TASK_STATUS:T002:in_progress" in content


class TestExtractTaskList:
    """Tests for extract_task_list function."""

    def test_extract_tasks(self, sample_plan):
        """Test extracting task list from plan."""
        content = sample_plan.read_text()
        tasks = extract_task_list(content)

        assert len(tasks) == 4
        assert any(t["id"] == "T001" for t in tasks)

    def test_extract_includes_status(self, sample_plan):
        """Test that extracted tasks include status from markers."""
        updater = PlanUpdater(sample_plan)
        updater.update_task_status("T001", "completed")
        updater.apply_updates()
        updater.save()

        content = sample_plan.read_text()
        tasks = extract_task_list(content)

        t001 = next(t for t in tasks if t["id"] == "T001")
        assert t001["status"] == "completed"
