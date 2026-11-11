"""Tests for plan updater module."""

import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plan_updater import (
    PlanUpdater,
    IMPL_START_MARKER,
    IMPL_END_MARKER,
    insert_implementation_summary_reference,
    insert_tasks_file_reference,
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
        content = sample_plan.read_text(encoding="utf-8")
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


@pytest.mark.parametrize("insert", [
    insert_implementation_summary_reference, insert_tasks_file_reference,
])
def test_a_rewritten_reference_keeps_the_path_literally(insert):
    """The second insert rewrites the section through `re.sub`, which read a
    template's backslashes as escapes: a Windows path was `bad escape \\p`
    (`..\\.multi-llm\\p\\implement\\summary.md`, Windows CI 2026-10-23)."""
    first = insert("# Plan\n", "old/summary.md")
    path = "..\\.multi-llm\\p\\implement\\summary.md"
    content = insert(first, path)
    assert f"[{path}]({path})" in content
    assert "old/summary.md" not in content
