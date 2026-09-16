"""Tests for human decision tracking features in state_manager module."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import (
    generate_group_id,
    handle_plan_hash_change,
    StateManager,
)


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
This is a sample plan for testing state management.

## Tasks

### T001: Create directory structure
Create the basic directory structure.
- Depends on: none
"""
    plan_path = temp_dir / "test-plan.md"
    plan_path.write_text(plan_content, encoding="utf-8")
    return plan_path


@pytest.fixture
def sample_group():
    """Create a sample group for testing."""
    return {
        "theme": "Error Handling",
        "suggestions": [
            {
                "type": "addition",
                "section": "Implementation",
                "details": "Add try-catch blocks for error handling"
            },
            {
                "type": "improvement",
                "section": "Testing",
                "details": "Add unit tests for error cases"
            }
        ]
    }


@pytest.fixture
def sample_group_with_issues():
    """Create a sample group using 'issues' key instead of 'suggestions'."""
    return {
        "theme": "Code Quality",
        "issues": [
            {
                "type": "bug",
                "reference": "src/main.py:42",
                "desc": "Missing null check"
            }
        ]
    }


@pytest.fixture
def sample_groups_list():
    """Create a list of sample groups for testing."""
    return [
        {
            "theme": "Error Handling",
            "suggestions": [
                {"type": "addition", "section": "Core", "details": "Add error handling"}
            ]
        },
        {
            "theme": "Performance",
            "suggestions": [
                {"type": "improvement", "section": "Core", "details": "Add caching"}
            ]
        },
        {
            "theme": "Documentation",
            "suggestions": [
                {"type": "addition", "section": "Docs", "details": "Add API docs"}
            ]
        }
    ]


@pytest.fixture
def state_manager(sample_plan_file):
    """Create a StateManager instance for testing."""
    return StateManager(sample_plan_file)


# --- TestGenerateGroupId ---

class TestGenerateGroupId:
    """Tests for generate_group_id function."""

    def test_same_group_content_produces_same_id(self, sample_group):
        """Same group content produces same ID."""
        id1 = generate_group_id(sample_group)
        id2 = generate_group_id(sample_group)
        assert id1 == id2

    def test_different_themes_produce_different_ids(self, sample_group):
        """Different themes produce different IDs."""
        group1 = sample_group.copy()
        group2 = sample_group.copy()
        group2["theme"] = "Different Theme"

        id1 = generate_group_id(group1)
        id2 = generate_group_id(group2)
        assert id1 != id2

    def test_different_suggestions_produce_different_ids(self, sample_group):
        """Different suggestions produce different IDs."""
        group1 = sample_group.copy()
        group2 = {
            "theme": sample_group["theme"],
            "suggestions": [
                {
                    "type": "different",
                    "section": "Different",
                    "details": "Completely different suggestion"
                }
            ]
        }

        id1 = generate_group_id(group1)
        id2 = generate_group_id(group2)
        assert id1 != id2

    def test_works_with_suggestions_key(self, sample_group):
        """Works with 'suggestions' key."""
        group_id = generate_group_id(sample_group)
        assert group_id is not None
        assert len(group_id) == 16

    def test_works_with_issues_key(self, sample_group_with_issues):
        """Works with 'issues' key instead of 'suggestions'."""
        group_id = generate_group_id(sample_group_with_issues)
        assert group_id is not None
        assert len(group_id) == 16

    def test_id_is_16_characters_long(self, sample_group):
        """ID is 16 characters long."""
        group_id = generate_group_id(sample_group)
        assert len(group_id) == 16

    def test_id_is_deterministic(self, sample_group):
        """ID is deterministic (same input = same output)."""
        # Generate ID multiple times
        ids = [generate_group_id(sample_group) for _ in range(10)]
        # All should be identical
        assert all(id == ids[0] for id in ids)

    def test_truncates_long_descriptions_for_stability(self):
        """Truncates long descriptions for stability."""
        # Create two groups with details differing only after 200 chars
        base_details = "A" * 200
        group1 = {
            "theme": "Test",
            "suggestions": [
                {"type": "test", "section": "test", "details": base_details + "XXXXXX"}
            ]
        }
        group2 = {
            "theme": "Test",
            "suggestions": [
                {"type": "test", "section": "test", "details": base_details + "YYYYYY"}
            ]
        }

        # Both should produce the same ID since truncation happens at 200 chars
        id1 = generate_group_id(group1)
        id2 = generate_group_id(group2)
        assert id1 == id2

    def test_empty_group_produces_valid_id(self):
        """Empty group produces a valid ID."""
        empty_group = {}
        group_id = generate_group_id(empty_group)
        assert group_id is not None
        assert len(group_id) == 16

    def test_id_is_hex_string(self, sample_group):
        """ID is a valid hex string."""
        group_id = generate_group_id(sample_group)
        # Should only contain hex characters
        assert all(c in '0123456789abcdef' for c in group_id)


# --- TestHandlePlanHashChange ---

class TestHandlePlanHashChange:
    """Tests for handle_plan_hash_change function."""

    def test_remaps_decisions_when_groups_match(self, sample_groups_list):
        """Remaps decisions when groups match."""
        # Create decisions for the original groups
        old_decisions = {}
        for group in sample_groups_list:
            gid = generate_group_id(group)
            old_decisions[gid] = {
                "decision": "approved",
                "reason": "Looks good",
                "timestamp": "2025-01-01T00:00:00"
            }

        # Same groups in new plan
        new_groups = sample_groups_list.copy()

        remapped = handle_plan_hash_change(old_decisions, sample_groups_list, new_groups)

        # All decisions should be remapped
        assert len(remapped) == len(old_decisions)
        for gid in remapped:
            assert remapped[gid]["remapped"] is True

    def test_drops_decisions_for_removed_groups(self, sample_groups_list):
        """Drops decisions for removed groups."""
        # Create decisions for all groups
        old_decisions = {}
        for group in sample_groups_list:
            gid = generate_group_id(group)
            old_decisions[gid] = {
                "decision": "approved",
                "reason": "Looks good",
                "timestamp": "2025-01-01T00:00:00"
            }

        # New plan has only the first group
        new_groups = [sample_groups_list[0]]

        remapped = handle_plan_hash_change(old_decisions, sample_groups_list, new_groups)

        # Only one decision should remain
        assert len(remapped) == 1

    def test_keeps_only_matching_decisions(self, sample_groups_list):
        """Keeps only matching decisions."""
        # Create decisions for first two groups
        old_decisions = {}
        for group in sample_groups_list[:2]:
            gid = generate_group_id(group)
            old_decisions[gid] = {
                "decision": "approved",
                "reason": "Looks good",
                "timestamp": "2025-01-01T00:00:00"
            }

        # New groups include first and third (not second)
        new_groups = [sample_groups_list[0], sample_groups_list[2]]

        remapped = handle_plan_hash_change(old_decisions, sample_groups_list, new_groups)

        # Only the first group's decision should remain
        assert len(remapped) == 1
        first_gid = generate_group_id(sample_groups_list[0])
        assert first_gid in remapped

    def test_sets_remapped_true_on_remapped_decisions(self, sample_groups_list):
        """Sets remapped=True on remapped decisions."""
        group = sample_groups_list[0]
        gid = generate_group_id(group)
        old_decisions = {
            gid: {
                "decision": "approved",
                "reason": "Looks good",
                "timestamp": "2025-01-01T00:00:00"
            }
        }

        remapped = handle_plan_hash_change(old_decisions, [group], [group])

        assert remapped[gid]["remapped"] is True

    def test_empty_inputs_handled_correctly(self):
        """Empty inputs handled correctly."""
        # Empty old decisions
        remapped = handle_plan_hash_change({}, [], [])
        assert remapped == {}

        # Empty new groups
        old_decisions = {"abc123": {"decision": "approved"}}
        remapped = handle_plan_hash_change(old_decisions, [], [])
        assert remapped == {}

        # Empty old groups
        remapped = handle_plan_hash_change(old_decisions, [], [{"theme": "New"}])
        assert remapped == {}

    def test_preserves_decision_data(self, sample_groups_list):
        """Preserves original decision data except for remapped flag."""
        group = sample_groups_list[0]
        gid = generate_group_id(group)
        original_decision = {
            "decision": "skipped",
            "reason": "Not applicable now",
            "timestamp": "2025-01-01T00:00:00",
            "custom_field": "custom_value"
        }
        old_decisions = {gid: original_decision}

        remapped = handle_plan_hash_change(old_decisions, [group], [group])

        assert remapped[gid]["decision"] == "skipped"
        assert remapped[gid]["reason"] == "Not applicable now"
        assert remapped[gid]["timestamp"] == "2025-01-01T00:00:00"
        assert remapped[gid]["custom_field"] == "custom_value"


# --- TestStateManagerHumanDecisions ---

class TestStateManagerHumanDecisions:
    """Tests for StateManager human decision tracking methods."""

    def test_record_human_decision_stores_decision(self, state_manager):
        """record_human_decision stores decision."""
        state_manager.record_human_decision(
            phase="apply-suggestions",
            group_id="test_group_123",
            decision="approved",
            reason="Looks correct"
        )

        decision = state_manager.get_human_decision("apply-suggestions", "test_group_123")
        assert decision is not None
        assert decision["decision"] == "approved"
        assert decision["reason"] == "Looks correct"
        assert "timestamp" in decision

    def test_record_human_decision_with_batch_context(self, state_manager):
        """record_human_decision with batch_context."""
        batch_context = {
            "batch_number": 1,
            "total_batches": 5,
            "items_in_batch": 3
        }

        state_manager.record_human_decision(
            phase="apply-suggestions",
            group_id="test_group_456",
            decision="approved",
            batch_context=batch_context
        )

        decision = state_manager.get_human_decision("apply-suggestions", "test_group_456")
        assert decision["batch_context"] == batch_context

    def test_get_human_decision_retrieves_correct_decision(self, state_manager):
        """get_human_decision retrieves correct decision."""
        # Record multiple decisions
        state_manager.record_human_decision("phase1", "group_a", "approved")
        state_manager.record_human_decision("phase1", "group_b", "skipped")
        state_manager.record_human_decision("phase2", "group_a", "deferred")

        # Retrieve specific decisions
        decision_a1 = state_manager.get_human_decision("phase1", "group_a")
        decision_b1 = state_manager.get_human_decision("phase1", "group_b")
        decision_a2 = state_manager.get_human_decision("phase2", "group_a")

        assert decision_a1["decision"] == "approved"
        assert decision_b1["decision"] == "skipped"
        assert decision_a2["decision"] == "deferred"

    def test_get_human_decision_returns_none_for_unknown(self, state_manager):
        """get_human_decision returns None for unknown."""
        decision = state_manager.get_human_decision("nonexistent_phase", "unknown_group")
        assert decision is None

    def test_get_all_human_decisions_returns_all(self, state_manager):
        """get_all_human_decisions returns all."""
        state_manager.record_human_decision("test_phase", "group1", "approved")
        state_manager.record_human_decision("test_phase", "group2", "skipped")
        state_manager.record_human_decision("test_phase", "group3", "deferred")

        all_decisions = state_manager.get_all_human_decisions("test_phase")

        assert len(all_decisions) == 3
        assert "group1" in all_decisions
        assert "group2" in all_decisions
        assert "group3" in all_decisions

    def test_clear_human_decisions_removes_all(self, state_manager):
        """clear_human_decisions removes all."""
        state_manager.record_human_decision("test_phase", "group1", "approved")
        state_manager.record_human_decision("test_phase", "group2", "skipped")

        state_manager.clear_human_decisions("test_phase")

        all_decisions = state_manager.get_all_human_decisions("test_phase")
        assert len(all_decisions) == 0

    def test_decisions_persisted_after_save_reload(self, sample_plan_file):
        """Decisions persisted after save/reload."""
        # Create state manager and record decision
        sm1 = StateManager(sample_plan_file)
        sm1.record_human_decision("persist_test", "persist_group", "approved", "Test reason")
        sm1.save()

        # Create new state manager for same plan
        sm2 = StateManager(sample_plan_file)

        # Decision should be persisted
        decision = sm2.get_human_decision("persist_test", "persist_group")
        assert decision is not None
        assert decision["decision"] == "approved"
        assert decision["reason"] == "Test reason"

    def test_clear_human_decisions_only_affects_specified_phase(self, state_manager):
        """clear_human_decisions only affects the specified phase."""
        state_manager.record_human_decision("phase1", "group1", "approved")
        state_manager.record_human_decision("phase2", "group1", "skipped")

        state_manager.clear_human_decisions("phase1")

        # phase1 should be cleared
        assert len(state_manager.get_all_human_decisions("phase1")) == 0
        # phase2 should be unchanged
        assert len(state_manager.get_all_human_decisions("phase2")) == 1


# --- TestStateManagerProcessingProgress ---

class TestStateManagerProcessingProgress:
    """Tests for StateManager processing progress tracking methods."""

    def test_record_processing_progress_stores_progress(self, state_manager):
        """record_processing_progress stores progress."""
        state_manager.record_processing_progress(
            phase="apply-suggestions",
            total_items=10,
            processed_items=5,
            current_batch=2,
            total_batches=4
        )

        progress = state_manager.get_processing_progress("apply-suggestions")
        assert progress is not None
        assert progress["total_items"] == 10
        assert progress["processed_items"] == 5
        assert progress["current_batch"] == 2
        assert progress["total_batches"] == 4

    def test_get_processing_progress_retrieves_correct_progress(self, state_manager):
        """get_processing_progress retrieves correct progress."""
        state_manager.record_processing_progress("phase1", 10, 5, 1, 2)
        state_manager.record_processing_progress("phase2", 20, 15, 3, 4)

        progress1 = state_manager.get_processing_progress("phase1")
        progress2 = state_manager.get_processing_progress("phase2")

        assert progress1["total_items"] == 10
        assert progress2["total_items"] == 20

    def test_progress_includes_all_required_fields(self, state_manager):
        """Progress includes all required fields."""
        state_manager.record_processing_progress("test", 100, 50, 5, 10)

        progress = state_manager.get_processing_progress("test")

        required_fields = ["total_items", "processed_items", "current_batch", "total_batches", "last_updated"]
        for field in required_fields:
            assert field in progress, f"Missing required field: {field}"

    def test_clear_processing_progress_removes_progress(self, state_manager):
        """clear_processing_progress removes progress."""
        state_manager.record_processing_progress("test", 10, 5, 1, 2)

        state_manager.clear_processing_progress("test")

        progress = state_manager.get_processing_progress("test")
        assert progress is None

    def test_progress_persisted_after_save_reload(self, sample_plan_file):
        """Progress persisted after save/reload."""
        sm1 = StateManager(sample_plan_file)
        sm1.record_processing_progress("persist_test", 100, 50, 5, 10)
        sm1.save()

        sm2 = StateManager(sample_plan_file)

        progress = sm2.get_processing_progress("persist_test")
        assert progress is not None
        assert progress["total_items"] == 100
        assert progress["processed_items"] == 50

    def test_get_processing_progress_returns_none_for_unknown_phase(self, state_manager):
        """get_processing_progress returns None for unknown phase."""
        progress = state_manager.get_processing_progress("nonexistent_phase")
        assert progress is None

    def test_progress_updates_overwrite_previous(self, state_manager):
        """Progress updates overwrite previous values."""
        state_manager.record_processing_progress("test", 10, 0, 0, 2)
        state_manager.record_processing_progress("test", 10, 5, 1, 2)

        progress = state_manager.get_processing_progress("test")
        assert progress["processed_items"] == 5
        assert progress["current_batch"] == 1


# --- TestStateManagerProcessedItems ---

class TestStateManagerProcessedItems:
    """Tests for StateManager processed item tracking methods."""

    def test_mark_item_processed_stores_item(self, state_manager):
        """mark_item_processed stores item."""
        state_manager.mark_item_processed(
            phase="apply-suggestions",
            group_id="test_group_123",
            status="applied"
        )

        processed = state_manager.get_processed_items("apply-suggestions")
        assert "test_group_123" in processed
        assert processed["test_group_123"]["status"] == "applied"

    def test_mark_item_processed_with_details(self, state_manager):
        """mark_item_processed with details."""
        details = {
            "files_modified": ["src/main.py", "src/utils.py"],
            "lines_changed": 42,
            "error_message": None
        }

        state_manager.mark_item_processed(
            phase="apply-fixes",
            group_id="test_group_456",
            status="applied",
            details=details
        )

        processed = state_manager.get_processed_items("apply-fixes")
        assert processed["test_group_456"]["details"] == details

    def test_get_processed_items_returns_all(self, state_manager):
        """get_processed_items returns all."""
        state_manager.mark_item_processed("test_phase", "group1", "applied")
        state_manager.mark_item_processed("test_phase", "group2", "skipped")
        state_manager.mark_item_processed("test_phase", "group3", "failed")

        processed = state_manager.get_processed_items("test_phase")

        assert len(processed) == 3
        assert "group1" in processed
        assert "group2" in processed
        assert "group3" in processed

    def test_is_item_processed_returns_true_for_processed(self, state_manager):
        """is_item_processed returns True for processed."""
        state_manager.mark_item_processed("test_phase", "processed_group", "applied")

        assert state_manager.is_item_processed("test_phase", "processed_group") is True

    def test_is_item_processed_returns_false_for_unprocessed(self, state_manager):
        """is_item_processed returns False for unprocessed."""
        assert state_manager.is_item_processed("test_phase", "unprocessed_group") is False

    def test_clear_processed_items_removes_all(self, state_manager):
        """clear_processed_items removes all."""
        state_manager.mark_item_processed("test_phase", "group1", "applied")
        state_manager.mark_item_processed("test_phase", "group2", "skipped")

        state_manager.clear_processed_items("test_phase")

        processed = state_manager.get_processed_items("test_phase")
        assert len(processed) == 0

    def test_items_persisted_after_save_reload(self, sample_plan_file):
        """Items persisted after save/reload."""
        sm1 = StateManager(sample_plan_file)
        sm1.mark_item_processed("persist_test", "persist_group", "applied", {"key": "value"})
        sm1.save()

        sm2 = StateManager(sample_plan_file)

        assert sm2.is_item_processed("persist_test", "persist_group") is True
        processed = sm2.get_processed_items("persist_test")
        assert processed["persist_group"]["status"] == "applied"
        assert processed["persist_group"]["details"]["key"] == "value"

    def test_clear_processed_items_only_affects_specified_phase(self, state_manager):
        """clear_processed_items only affects the specified phase."""
        state_manager.mark_item_processed("phase1", "group1", "applied")
        state_manager.mark_item_processed("phase2", "group1", "applied")

        state_manager.clear_processed_items("phase1")

        # phase1 should be cleared
        assert len(state_manager.get_processed_items("phase1")) == 0
        # phase2 should be unchanged
        assert len(state_manager.get_processed_items("phase2")) == 1

    def test_processed_items_include_timestamp(self, state_manager):
        """Processed items include timestamp."""
        state_manager.mark_item_processed("test_phase", "test_group", "applied")

        processed = state_manager.get_processed_items("test_phase")
        assert "timestamp" in processed["test_group"]

    def test_get_processed_items_returns_copy(self, state_manager):
        """get_processed_items returns a copy, not the original dict."""
        state_manager.mark_item_processed("test_phase", "group1", "applied")

        processed = state_manager.get_processed_items("test_phase")
        processed["new_key"] = "new_value"

        # Original should not be modified
        original = state_manager.get_processed_items("test_phase")
        assert "new_key" not in original


class TestHumanTaskStrategy:
    """Tests for human task strategy persistence."""

    @pytest.fixture
    def state_manager(self, sample_plan_file):
        return StateManager(sample_plan_file)

    def test_save_and_get_strategy(self, state_manager):
        """Strategy can be saved and retrieved."""
        state_manager.save_human_task_strategy("pause-and-ask")
        assert state_manager.get_human_task_strategy() == "pause-and-ask"

    def test_get_strategy_returns_none_when_not_set(self, state_manager):
        """Returns None when no strategy has been saved."""
        assert state_manager.get_human_task_strategy() is None

    def test_strategy_persists_across_save_reload(self, sample_plan_file):
        """Strategy survives save/reload cycle."""
        sm1 = StateManager(sample_plan_file)
        sm1.save_human_task_strategy("skip-continue")
        sm1.save()

        sm2 = StateManager(sample_plan_file)
        assert sm2.get_human_task_strategy() == "skip-continue"

    def test_strategy_overwrite(self, state_manager):
        """Saving a new strategy overwrites the old one."""
        state_manager.save_human_task_strategy("pause-and-ask")
        state_manager.save_human_task_strategy("skip-dependents")
        assert state_manager.get_human_task_strategy() == "skip-dependents"

    def test_all_valid_strategies(self, state_manager):
        """All valid strategy values can be saved and retrieved."""
        for strategy in ["pause-and-ask", "skip-continue", "skip-dependents", "cancel"]:
            state_manager.save_human_task_strategy(strategy)
            assert state_manager.get_human_task_strategy() == strategy


class TestTaskStatusReason:
    """Tests for per-task status reason tracking."""

    @pytest.fixture
    def state_manager(self, sample_plan_file):
        return StateManager(sample_plan_file)

    def test_update_status_with_reason(self, state_manager):
        """Status update with reason stores both status and reason."""
        state_manager.update_task_status("T001", "skipped", reason="deferred — human task")
        assert state_manager.get_task_status("T001") == "skipped"
        assert state_manager.get_task_status_reason("T001") == "deferred — human task"

    def test_update_status_without_reason(self, state_manager):
        """Status update without reason only stores status, no reason."""
        state_manager.update_task_status("T001", "completed")
        assert state_manager.get_task_status("T001") == "completed"
        assert state_manager.get_task_status_reason("T001") is None

    def test_reason_not_set_for_unknown_task(self, state_manager):
        """Reason returns None for unknown task."""
        assert state_manager.get_task_status_reason("T999") is None

    def test_reason_persists_across_save_reload(self, sample_plan_file):
        """Status reason survives save/reload cycle."""
        sm1 = StateManager(sample_plan_file)
        sm1.update_task_status("T001", "skipped", reason="depends on skipped T000")
        sm1.save()

        sm2 = StateManager(sample_plan_file)
        assert sm2.get_task_status_reason("T001") == "depends on skipped T000"

    def test_reason_only_set_when_provided(self, state_manager):
        """Multiple status updates: reason only present for updates that provided one."""
        state_manager.update_task_status("T001", "completed")
        state_manager.update_task_status("T002", "skipped", reason="human task skipped")
        state_manager.update_task_status("T003", "failed")

        assert state_manager.get_task_status_reason("T001") is None
        assert state_manager.get_task_status_reason("T002") == "human task skipped"
        assert state_manager.get_task_status_reason("T003") is None


class TestImplementationTaskStatuses:
    """Tests for `implementation_task_statuses` (F4): `task_status` is one flat
    map shared by implement's own decomposed tasks AND the write-batch ids the
    apply-suggestions / apply-code-fixes / apply-task-suggestions phases assign
    (`apply-batch-N`, `apply-fix-batch-N`, `apply-task-batch-N`) — only the
    former should ever count as implement task progress.
    """

    @pytest.fixture
    def state_manager(self, sample_plan_file):
        return StateManager(sample_plan_file)

    def test_an_apply_batch_only_map_yields_no_implement_tasks(self, state_manager):
        """A run that only ever executed apply-* write batches reports zero
        implement tasks, never `len(task_status)` of them."""
        state_manager.update_task_status("apply-batch-0", "completed")
        state_manager.update_task_status("apply-fix-batch-3", "completed")
        state_manager.update_task_status("apply-task-batch-1", "failed")

        assert state_manager.implementation_task_statuses() == {}

    def test_t000_style_ids_still_count(self, state_manager):
        """The decomposer's own `T###` ids pass through untouched."""
        state_manager.update_task_status("T001", "completed")
        state_manager.update_task_status("T002", "failed")

        assert state_manager.implementation_task_statuses() == {
            "T001": "completed",
            "T002": "failed",
        }

    def test_a_mixed_map_reports_only_the_implement_half(self, state_manager):
        """The exact failure mode from the audit: apply-suggestions batches and
        implement tasks share one state.json after both phases have run."""
        state_manager.update_task_status("apply-batch-0", "completed")
        state_manager.update_task_status("T001", "completed")
        state_manager.update_task_status("T002", "in_progress")

        assert state_manager.implementation_task_statuses() == {
            "T001": "completed",
            "T002": "in_progress",
        }

    def test_get_all_task_statuses_is_unchanged(self, state_manager):
        """The existing unfiltered accessor keeps serving its other callers
        (e.g. implement_orchestrator's own resume loop, which already guards
        with `if task_id in decomposer.tasks`)."""
        state_manager.update_task_status("apply-batch-0", "completed")
        state_manager.update_task_status("T001", "completed")

        assert state_manager.get_all_task_statuses() == {
            "apply-batch-0": "completed",
            "T001": "completed",
        }


class TestDependencyOverrides:
    """Tests for dependency override tracking."""

    @pytest.fixture
    def state_manager(self, sample_plan_file):
        return StateManager(sample_plan_file)

    def test_record_and_get_override(self, state_manager):
        """Record and retrieve a dependency override."""
        state_manager.record_dependency_override("T003", ["T001", "T002"])
        override = state_manager.get_dependency_overrides("T003")

        assert override is not None
        assert override["overridden_deps"] == ["T001", "T002"]
        assert "timestamp" in override

    def test_get_override_returns_none_for_unknown(self, state_manager):
        """Returns None for task with no override."""
        assert state_manager.get_dependency_overrides("T999") is None

    def test_get_all_overrides(self, state_manager):
        """Get all dependency overrides."""
        state_manager.record_dependency_override("T003", ["T001"])
        state_manager.record_dependency_override("T005", ["T004"])

        all_overrides = state_manager.get_all_dependency_overrides()
        assert len(all_overrides) == 2
        assert "T003" in all_overrides
        assert "T005" in all_overrides

    def test_get_all_overrides_returns_copy(self, state_manager):
        """get_all_dependency_overrides returns a copy, not the original."""
        state_manager.record_dependency_override("T003", ["T001"])
        overrides = state_manager.get_all_dependency_overrides()
        overrides["T999"] = {"overridden_deps": ["T998"], "timestamp": "fake"}

        original = state_manager.get_all_dependency_overrides()
        assert "T999" not in original

    def test_overrides_persist_across_save_reload(self, sample_plan_file):
        """Dependency overrides survive save/reload cycle."""
        sm1 = StateManager(sample_plan_file)
        sm1.record_dependency_override("T003", ["T001", "T002"])
        sm1.save()

        sm2 = StateManager(sample_plan_file)
        override = sm2.get_dependency_overrides("T003")
        assert override is not None
        assert override["overridden_deps"] == ["T001", "T002"]


# --- TestReviewPhaseCompletedFlag ---
# F11: `review_phase_completed` was seeded False and never flipped by any
# production caller — a run that fully completed `review-plan` still read
# `review_phase_completed: false` beside `phases_completed: {"review-plan":
# "..."}` in the same file. `mark_phase_completed` now flips it, but only for
# the phase its own docstring names ("the *plan* review phase").

class TestReviewPhaseCompletedFlag:
    def test_starts_false(self, state_manager):
        """A fresh state has not completed the plan review phase."""
        assert state_manager.is_review_phase_completed() is False
        assert state_manager.state["review_phase_completed"] is False

    def test_marking_review_plan_phase_completed_flips_the_flag(self, state_manager):
        """Completing review-plan sets both the map entry and the flag."""
        state_manager.mark_phase_completed("review-plan")

        assert "review-plan" in state_manager.state["phases_completed"]
        assert state_manager.is_review_phase_completed() is True
        assert state_manager.state["review_phase_completed"] is True

    def test_marking_a_different_phase_leaves_the_flag_false(self, state_manager):
        """Only review-plan is 'the plan review phase' — review-tasks and
        code-review track their own completion through phases_completed and
        must not be conflated with it."""
        state_manager.mark_phase_completed("review-tasks")
        state_manager.mark_phase_completed("code-review")

        assert state_manager.is_review_phase_completed() is False

    def test_flag_persists_across_save_reload(self, sample_plan_file):
        """The flag survives a save/reload cycle like the rest of state.json."""
        sm1 = StateManager(sample_plan_file)
        sm1.mark_phase_completed("review-plan")
        sm1.save()

        sm2 = StateManager(sample_plan_file)
        assert sm2.is_review_phase_completed() is True

    def test_missing_key_loads_as_false(self, sample_plan_file):
        """An existing state.json written before F11 (no key at all) still
        loads — the field degrades to False rather than raising."""
        sm = StateManager(sample_plan_file)
        sm.save()
        state_path = sm.state_file
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
        del data["review_phase_completed"]
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        reloaded = StateManager(sample_plan_file)
        assert reloaded.is_review_phase_completed() is False


class TestPlanChangedFlagIsNotSticky:
    """F8: `_load_or_create` only ever SET `plan_changed` on a hash mismatch;
    nothing cleared it when a later load found the hash agreeing again, so
    the flag survived every read-only phase after the run that set it and
    `implement --resume` kept warning about a change that was not there."""

    def test_a_real_edit_sets_the_flag(self, sample_plan_file):
        """Sanity check: editing the plan between loads still sets it."""
        sm1 = StateManager(sample_plan_file)
        sm1.save()

        sample_plan_file.write_text("# Edited\n", encoding="utf-8")

        sm2 = StateManager(sample_plan_file)
        assert sm2.has_plan_changed() is True

    def test_the_flag_clears_once_the_hash_agrees_again(self, sample_plan_file):
        """The residue case: a stale `plan_changed: True` sits in state.json
        (e.g. left over from a run that predates a hash re-stamp) while the
        stored `plan_hash` already matches the plan on disk. A fresh load
        must not keep repeating a change that is not there."""
        sm1 = StateManager(sample_plan_file)
        sm1.save()

        # Hand-craft the residue: the stored hash matches the plan, but a
        # previous (buggy or interrupted) run left `plan_changed: True`
        # sitting beside it.
        state_data = json.loads(sm1.state_file.read_text(encoding="utf-8"))
        assert state_data["plan_hash"] == sm1._compute_plan_hash()
        state_data["plan_changed"] = True
        sm1.state_file.write_text(json.dumps(state_data), encoding="utf-8")

        sm2 = StateManager(sample_plan_file)
        assert sm2.has_plan_changed() is False
        assert "plan_changed" not in sm2.state

    def test_the_cleared_flag_stays_clear_on_the_next_save_reload(
        self, sample_plan_file
    ):
        """The clear happens at load, not only in memory — it has to survive
        a save so the next session (e.g. `implement --resume`) reads it too."""
        sm1 = StateManager(sample_plan_file)
        sm1.save()
        state_data = json.loads(sm1.state_file.read_text(encoding="utf-8"))
        state_data["plan_changed"] = True
        sm1.state_file.write_text(json.dumps(state_data), encoding="utf-8")

        sm2 = StateManager(sample_plan_file)
        sm2.save()

        sm3 = StateManager(sample_plan_file)
        assert sm3.has_plan_changed() is False
