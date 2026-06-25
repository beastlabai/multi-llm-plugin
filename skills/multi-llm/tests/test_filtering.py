"""Tests for filtering module."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.filtering import (
    resolve_bulk_option_conflicts,
    filter_items,
    validate_batch_mode_honored,
    validate_claude_decide_items_honored,
    generate_batch_id,
    should_bypass_no_selection_confirmation,
    ERROR_TYPE_AMBIGUOUS,
)


class TestResolveBulkOptionConflicts:
    """Tests for resolve_bulk_option_conflicts function."""

    def test_skip_all_human_overrides_all_options(self):
        """Test that skip_all_human=True overrides all other options."""
        effective, warnings = resolve_bulk_option_conflicts(
            skip_all_human=True,
            approve_all_human=True,
            approve_all_low=True,
            approve_importance_levels=["HIGH", "MEDIUM"],
            approve_validation_failed=True
        )

        assert effective["skip_all"] is True
        assert effective["approve_all"] is False
        assert effective["approve_importance_levels"] == []
        assert effective["approve_validation_failed"] is False
        assert len(warnings) == 1
        assert "skip-all-human overrides" in warnings[0]

    def test_approve_all_human_overrides_selective_options(self):
        """Test that approve_all_human=True overrides selective options."""
        effective, warnings = resolve_bulk_option_conflicts(
            approve_all_human=True,
            approve_all_low=True,
            approve_importance_levels=["HIGH"]
        )

        assert effective["skip_all"] is False
        assert effective["approve_all"] is True
        assert effective["approve_importance_levels"] == []
        assert len(warnings) == 1
        assert "approve-all overrides" in warnings[0]

    def test_approve_importance_levels_specified(self):
        """Test approve_importance_levels are normalized to uppercase."""
        effective, warnings = resolve_bulk_option_conflicts(
            approve_importance_levels=["high", "Medium", "LOW"]
        )

        assert effective["skip_all"] is False
        assert effective["approve_all"] is False
        assert set(effective["approve_importance_levels"]) == {"HIGH", "MEDIUM", "LOW"}
        assert len(warnings) == 0

    def test_approve_all_low_without_other_options(self):
        """Test approve_all_low sets LOW importance level."""
        effective, warnings = resolve_bulk_option_conflicts(
            approve_all_low=True
        )

        assert effective["skip_all"] is False
        assert effective["approve_all"] is False
        assert effective["approve_importance_levels"] == ["LOW"]
        assert len(warnings) == 0

    def test_approve_validation_failed_flag(self):
        """Test approve_validation_failed is passed through."""
        effective, warnings = resolve_bulk_option_conflicts(
            approve_validation_failed=True
        )

        assert effective["approve_validation_failed"] is True
        assert len(warnings) == 0

    def test_conflict_skip_all_vs_approve_all_skip_wins(self):
        """Test that skip_all wins over approve_all."""
        effective, warnings = resolve_bulk_option_conflicts(
            skip_all_human=True,
            approve_all_human=True
        )

        assert effective["skip_all"] is True
        assert effective["approve_all"] is False
        assert len(warnings) == 1

    def test_conflict_approve_all_vs_approve_importance_approve_all_wins(self):
        """Test that approve_all wins over approve_importance."""
        effective, warnings = resolve_bulk_option_conflicts(
            approve_all_human=True,
            approve_importance_levels=["HIGH"]
        )

        assert effective["approve_all"] is True
        assert effective["approve_importance_levels"] == []
        assert len(warnings) == 1

    def test_warning_approve_importance_without_low_and_approve_all_low(self):
        """Test warning when approve_importance doesn't include LOW but approve_all_low set."""
        effective, warnings = resolve_bulk_option_conflicts(
            approve_all_low=True,
            approve_importance_levels=["HIGH", "MEDIUM"]
        )

        assert effective["approve_importance_levels"] == ["HIGH", "MEDIUM"]
        assert len(warnings) == 1
        assert "does not include LOW" in warnings[0]

    def test_no_options_returns_defaults(self):
        """Test that no options returns all defaults."""
        effective, warnings = resolve_bulk_option_conflicts()

        assert effective["skip_all"] is False
        assert effective["approve_all"] is False
        assert effective["approve_importance_levels"] == []
        assert effective["approve_validation_failed"] is False
        assert len(warnings) == 0


class TestFilterItems:
    """Tests for filter_items function."""

    @pytest.fixture
    def valid_group(self):
        """Create a valid group with HIGH importance."""
        return {
            "group_index": 1,
            "theme": "Test theme",
            "validation_status": "valid",
            "suggestions": [{"importance": "HIGH"}]
        }

    @pytest.fixture
    def invalid_group(self):
        """Create an invalid group."""
        return {
            "group_index": 2,
            "theme": "Invalid theme",
            "validation_status": "invalid",
            "suggestions": [{"importance": "HIGH"}]
        }

    @pytest.fixture
    def needs_human_group(self):
        """Create a needs-human-decision group."""
        return {
            "group_index": 3,
            "theme": "Needs human theme",
            "validation_status": "needs-human-decision",
            "suggestions": [{"importance": "MEDIUM"}]
        }

    @pytest.fixture
    def validation_failed_group(self):
        """Create a validation_failed group with recoverable error."""
        return {
            "group_index": 4,
            "theme": "Validation failed theme",
            "validation_status": "validation_failed",
            "validation_error_type": "parsing_error",
            "suggestions": [{"importance": "LOW"}]
        }

    @pytest.fixture
    def low_importance_valid_group(self):
        """Create a valid group with LOW importance."""
        return {
            "group_index": 5,
            "theme": "Low importance theme",
            "validation_status": "valid",
            "suggestions": [{"importance": "LOW"}]
        }

    def test_filter_valid_items(self, valid_group):
        """Test that valid items go to valid list."""
        valid, needs_human, skipped, report = filter_items([valid_group])

        assert len(valid) == 1
        assert valid[0]["group_index"] == 1
        assert len(needs_human) == 0
        assert len(skipped) == 0

    def test_filter_invalid_items(self, invalid_group):
        """Test that invalid items go to skipped list."""
        valid, needs_human, skipped, report = filter_items([invalid_group])

        assert len(valid) == 0
        assert len(needs_human) == 0
        assert len(skipped) == 1
        assert skipped[0]["group_index"] == 2

    def test_filter_needs_human_items(self, needs_human_group):
        """Test that needs-human-decision items go to needs_human list."""
        valid, needs_human, skipped, report = filter_items([needs_human_group])

        assert len(valid) == 0
        assert len(needs_human) == 1
        assert needs_human[0]["group_index"] == 3
        assert len(skipped) == 0

    def test_filter_validation_failed_items(self, validation_failed_group):
        """Test that validation_failed items go to needs_human unless approved."""
        valid, needs_human, skipped, report = filter_items([validation_failed_group])

        assert len(valid) == 0
        assert len(needs_human) == 1
        assert needs_human[0]["group_index"] == 4
        assert len(skipped) == 0

    def test_skip_all_human_skips_human_review_items(self, needs_human_group, validation_failed_group):
        """Test that skip_all_human skips all human review items."""
        valid, needs_human, skipped, report = filter_items(
            [needs_human_group, validation_failed_group],
            skip_all_human=True
        )

        assert len(valid) == 0
        assert len(needs_human) == 0
        assert len(skipped) == 2

    def test_approve_all_human_approves_all_human_review_items(self, needs_human_group, validation_failed_group):
        """Test that approve_all_human approves all human review items."""
        valid, needs_human, skipped, report = filter_items(
            [needs_human_group, validation_failed_group],
            approve_all_human=True
        )

        assert len(valid) == 2
        assert len(needs_human) == 0
        assert len(skipped) == 0
        for group in valid:
            assert group["auto_approved"] is True
            assert group["auto_approval_reason"] == "--approve-all"

    def test_approve_importance_levels_approves_specific_levels(self, needs_human_group):
        """Test that approve_importance_levels approves specific importance levels."""
        valid, needs_human, skipped, report = filter_items(
            [needs_human_group],
            approve_importance_levels=["MEDIUM"]
        )

        assert len(valid) == 1
        assert valid[0]["auto_approved"] is True
        assert "MEDIUM" in valid[0]["auto_approval_reason"]

    def test_approve_importance_levels_does_not_approve_other_levels(self, needs_human_group):
        """Test that approve_importance_levels doesn't approve items with other levels."""
        valid, needs_human, skipped, report = filter_items(
            [needs_human_group],
            approve_importance_levels=["HIGH"]  # Group has MEDIUM importance
        )

        assert len(valid) == 0
        assert len(needs_human) == 1

    def test_approve_validation_failed_approves_recoverable_errors(self, validation_failed_group):
        """Test that approve_validation_failed approves items with recoverable errors."""
        valid, needs_human, skipped, report = filter_items(
            [validation_failed_group],
            approve_validation_failed=True
        )

        assert len(valid) == 1
        assert valid[0]["auto_approved"] is True
        assert "--approve-validation-failed" in valid[0]["auto_approval_reason"]

    def test_approve_validation_failed_does_not_approve_ambiguous_errors(self):
        """Test that approve_validation_failed doesn't approve ambiguous errors."""
        ambiguous_group = {
            "group_index": 6,
            "theme": "Ambiguous theme",
            "validation_status": "validation_failed",
            "validation_error_type": ERROR_TYPE_AMBIGUOUS,
            "suggestions": [{"importance": "LOW"}]
        }
        valid, needs_human, skipped, report = filter_items(
            [ambiguous_group],
            approve_validation_failed=True
        )

        assert len(valid) == 0
        assert len(needs_human) == 1

    def test_dry_run_mode_does_not_modify_groups(self, needs_human_group):
        """Test that dry_run mode doesn't modify groups."""
        original_status = needs_human_group["validation_status"]
        valid, needs_human, skipped, report = filter_items(
            [needs_human_group],
            approve_all_human=True,
            dry_run=True
        )

        # In dry run, items stay in needs_human
        assert len(valid) == 0
        assert len(needs_human) == 1
        assert needs_human[0]["validation_status"] == original_status
        assert "auto_approved" not in needs_human[0]
        assert len(report["would_auto_approve"]) == 1

    def test_min_priority_low_includes_all_valid_items(self, low_importance_valid_group):
        """Test that min_priority='low' (default) includes LOW importance valid items."""
        valid, needs_human, skipped, report = filter_items(
            [low_importance_valid_group],
            min_priority="low"
        )

        assert len(valid) == 1
        assert len(skipped) == 0

    def test_min_priority_medium_skips_low_importance(self, low_importance_valid_group):
        """Test that min_priority='medium' skips LOW importance valid items."""
        valid, needs_human, skipped, report = filter_items(
            [low_importance_valid_group],
            min_priority="medium"
        )

        assert len(valid) == 0
        assert len(skipped) == 1

    def test_mixed_groups_filtered_correctly(
        self, valid_group, invalid_group, needs_human_group, validation_failed_group
    ):
        """Test that mixed groups are filtered correctly."""
        all_groups = [valid_group, invalid_group, needs_human_group, validation_failed_group]
        valid, needs_human, skipped, report = filter_items(all_groups)

        assert len(valid) == 1  # Only the valid group
        assert len(needs_human) == 2  # needs-human-decision + validation_failed
        assert len(skipped) == 1  # Only the invalid group

    def test_skip_human_review_legacy_flag(self, needs_human_group):
        """Test that skip_human_review (legacy flag) works."""
        valid, needs_human, skipped, report = filter_items(
            [needs_human_group],
            skip_human_review=True
        )

        assert len(valid) == 0
        assert len(needs_human) == 0
        assert len(skipped) == 1

    def test_dry_run_report_contains_warnings(self, needs_human_group):
        """Test that dry_run report contains conflict warnings."""
        valid, needs_human, skipped, report = filter_items(
            [needs_human_group],
            skip_all_human=True,
            approve_all_human=True,
            dry_run=True
        )

        assert len(report["warnings"]) > 0

    def test_importance_added_to_group(self, valid_group):
        """Test that _importance is added to each group."""
        valid, needs_human, skipped, report = filter_items([valid_group])

        assert valid[0]["_importance"] == "HIGH"


class TestMinPriority:
    """Tests for min_priority filtering functionality."""

    @pytest.fixture
    def mixed_importance_groups(self):
        """Create groups with HIGH, MEDIUM, and LOW importance."""
        return [
            {
                "group_index": 1,
                "theme": "High importance",
                "validation_status": "valid",
                "suggestions": [{"importance": "HIGH"}]
            },
            {
                "group_index": 2,
                "theme": "Medium importance",
                "validation_status": "valid",
                "suggestions": [{"importance": "MEDIUM"}]
            },
            {
                "group_index": 3,
                "theme": "Low importance",
                "validation_status": "valid",
                "suggestions": [{"importance": "LOW"}]
            }
        ]

    def test_min_priority_default_is_low(self, mixed_importance_groups):
        """Test that default min_priority includes all valid items."""
        valid, needs_human, skipped, report = filter_items(mixed_importance_groups)

        assert len(valid) == 3
        assert len(skipped) == 0

    def test_min_priority_low_includes_all(self, mixed_importance_groups):
        """Test that min_priority='low' includes HIGH, MEDIUM, and LOW."""
        valid, needs_human, skipped, report = filter_items(
            mixed_importance_groups,
            min_priority="low"
        )

        assert len(valid) == 3
        assert len(skipped) == 0

    def test_min_priority_medium_skips_low(self, mixed_importance_groups):
        """Test that min_priority='medium' skips LOW, includes MEDIUM and HIGH."""
        valid, needs_human, skipped, report = filter_items(
            mixed_importance_groups,
            min_priority="medium"
        )

        assert len(valid) == 2
        assert len(skipped) == 1
        assert skipped[0]["_importance"] == "LOW"

    def test_min_priority_high_skips_low_and_medium(self, mixed_importance_groups):
        """Test that min_priority='high' only includes HIGH importance."""
        valid, needs_human, skipped, report = filter_items(
            mixed_importance_groups,
            min_priority="high"
        )

        assert len(valid) == 1
        assert len(skipped) == 2
        assert valid[0]["_importance"] == "HIGH"
        skipped_importances = {g["_importance"] for g in skipped}
        assert skipped_importances == {"LOW", "MEDIUM"}

    def test_min_priority_case_insensitive(self, mixed_importance_groups):
        """Test that min_priority is case insensitive."""
        valid_lower, _, _, _ = filter_items(mixed_importance_groups, min_priority="high")
        valid_upper, _, _, _ = filter_items(mixed_importance_groups, min_priority="HIGH")
        valid_mixed, _, _, _ = filter_items(mixed_importance_groups, min_priority="High")

        assert len(valid_lower) == len(valid_upper) == len(valid_mixed) == 1

    def test_min_priority_invalid_defaults_to_low(self, mixed_importance_groups):
        """Test that invalid min_priority defaults to including all."""
        valid, needs_human, skipped, report = filter_items(
            mixed_importance_groups,
            min_priority="invalid"
        )

        # Invalid priority defaults to level 0, so all items included
        assert len(valid) == 3
        assert len(skipped) == 0


class TestValidateBatchModeHonored:
    """Tests for validate_batch_mode_honored function."""

    def test_batch_mode_disabled_returns_true(self):
        """Test that disabled batch mode returns True."""
        config = {"batch_enabled": False}
        decisions = {"group_1": {"action": "approve"}}

        is_valid, warnings = validate_batch_mode_honored(config, decisions)

        assert is_valid is True
        assert len(warnings) == 0

    def test_no_decisions_recorded_returns_true(self):
        """Test that no decisions recorded returns True."""
        config = {"batch_enabled": True}
        decisions = {}

        is_valid, warnings = validate_batch_mode_honored(config, decisions)

        assert is_valid is True
        assert len(warnings) == 0

    def test_all_decisions_have_batch_context_returns_true(self):
        """Test that all decisions with batch_context returns True."""
        config = {"batch_enabled": True}
        decisions = {
            "group_1": {
                "action": "approve",
                "batch_context": {"batch_id": "batch_001"}
            },
            "group_2": {
                "action": "skip",
                "batch_context": {"batch_id": "batch_001"}
            }
        }

        is_valid, warnings = validate_batch_mode_honored(config, decisions)

        assert is_valid is True
        assert len(warnings) == 0

    def test_some_decisions_missing_batch_context_returns_false(self):
        """Test that some decisions without batch_context returns False with warnings."""
        config = {"batch_enabled": True}
        decisions = {
            "group_1": {
                "action": "approve",
                "batch_context": {"batch_id": "batch_001"}
            },
            "group_2": {
                "action": "skip"  # Missing batch_context
            },
            "group_3": {
                "action": "approve"  # Missing batch_context
            }
        }

        is_valid, warnings = validate_batch_mode_honored(config, decisions)

        assert is_valid is False
        assert len(warnings) >= 1
        assert "2 decisions" in warnings[0]
        assert "without batch_context" in warnings[0]

    def test_too_many_batch_ids_generates_warning(self):
        """Test that too many batch_ids generates warning."""
        config = {"batch_enabled": True}
        decisions = {
            "group_1": {"action": "approve", "batch_context": {"batch_id": "batch_001"}},
            "group_2": {"action": "skip", "batch_context": {"batch_id": "batch_002"}},
            "group_3": {"action": "approve", "batch_context": {"batch_id": "batch_003"}},
            "group_4": {"action": "skip", "batch_context": {"batch_id": "batch_004"}},
        }

        is_valid, warnings = validate_batch_mode_honored(config, decisions)

        # Still valid, but has warning about many batch IDs
        assert is_valid is True
        assert len(warnings) == 1
        assert "4 different batch_ids" in warnings[0]

    def test_two_batch_ids_no_warning(self):
        """Test that up to 2 batch_ids does not generate warning."""
        config = {"batch_enabled": True}
        decisions = {
            "group_1": {"action": "approve", "batch_context": {"batch_id": "batch_001"}},
            "group_2": {"action": "skip", "batch_context": {"batch_id": "batch_002"}},
        }

        is_valid, warnings = validate_batch_mode_honored(config, decisions)

        assert is_valid is True
        assert len(warnings) == 0


class TestGenerateBatchId:
    """Tests for generate_batch_id function."""

    def test_returns_string_starting_with_batch(self):
        """Test that returned ID starts with 'batch_'."""
        batch_id = generate_batch_id()

        assert isinstance(batch_id, str)
        assert batch_id.startswith("batch_")

    def test_returns_string_containing_timestamp(self):
        """Test that returned ID contains timestamp format."""
        batch_id = generate_batch_id()

        # The format is batch_YYYYMMDD_HHMMSS
        # Extract the part after "batch_"
        timestamp_part = batch_id[6:]  # Remove "batch_"

        # Should have format YYYYMMDD_HHMMSS
        assert len(timestamp_part) == 15  # 8 + 1 + 6
        assert "_" in timestamp_part
        date_part, time_part = timestamp_part.split("_")
        assert len(date_part) == 8
        assert len(time_part) == 6
        assert date_part.isdigit()
        assert time_part.isdigit()

    def test_each_call_returns_unique_id(self):
        """Test that each call returns a unique ID."""
        batch_id1 = generate_batch_id()
        time.sleep(1.1)  # Wait for timestamp to change
        batch_id2 = generate_batch_id()

        assert batch_id1 != batch_id2

    def test_batch_id_format_is_parseable(self):
        """Test that batch ID has a parseable date format."""
        from datetime import datetime

        batch_id = generate_batch_id()
        timestamp_str = batch_id[6:]  # Remove "batch_"

        # Should be parseable as datetime
        parsed = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        assert parsed is not None


class TestShouldBypassNoSelectionConfirmation:
    """Tests for should_bypass_no_selection_confirmation function."""

    def test_no_flags_returns_false(self):
        """No flags set → should NOT bypass (prompt the user)."""
        assert should_bypass_no_selection_confirmation() is False

    def test_no_confirm_flag(self):
        """--no-confirm alone bypasses."""
        assert should_bypass_no_selection_confirmation(no_confirm=True) is True

    def test_yes_flag(self):
        """--yes alone bypasses."""
        assert should_bypass_no_selection_confirmation(yes=True) is True

    def test_force_flag(self):
        """--force alone bypasses."""
        assert should_bypass_no_selection_confirmation(force=True) is True

    def test_approve_all_flag(self):
        """--approve-all alone bypasses."""
        assert should_bypass_no_selection_confirmation(approve_all=True) is True

    def test_skip_all_human_flag(self):
        """--skip-all-human alone bypasses."""
        assert should_bypass_no_selection_confirmation(skip_all_human=True) is True

    def test_approve_all_low_flag(self):
        """--approve-all-low alone bypasses."""
        assert should_bypass_no_selection_confirmation(approve_all_low=True) is True

    def test_approve_importance_flag(self):
        """--approve-importance with levels bypasses."""
        assert should_bypass_no_selection_confirmation(approve_importance=["HIGH"]) is True

    def test_approve_importance_empty_list_does_not_bypass(self):
        """--approve-importance with empty list does NOT bypass."""
        assert should_bypass_no_selection_confirmation(approve_importance=[]) is False

    def test_approve_validation_failed_flag(self):
        """--approve-validation-failed alone bypasses."""
        assert should_bypass_no_selection_confirmation(approve_validation_failed=True) is True

    def test_dry_run_flag(self):
        """--dry-run alone bypasses."""
        assert should_bypass_no_selection_confirmation(dry_run=True) is True

    def test_multiple_flags_still_bypasses(self):
        """Multiple bypass flags together still bypass."""
        assert should_bypass_no_selection_confirmation(
            approve_all=True, yes=True, force=True
        ) is True


class TestClaudeDecideBeatsBulkApproval:
    """Section 4d: a per-item claude_decide marker beats bulk approval."""

    def _group(self, **extra):
        g = {
            "group_hash": "g1",
            "validation_status": "needs-human-decision",
            "suggestions": [{"title": "A", "importance": "HIGH"}],
        }
        g.update(extra)
        return g

    def test_claude_decide_not_auto_approved_by_approve_all(self):
        """A marked group lands in needs_human, not valid, under --approve-all."""
        groups = [self._group(claude_decide=True)]
        valid, needs_human, skipped, _ = filter_items(
            groups, approve_all_human=True
        )
        assert len(needs_human) == 1
        assert needs_human[0]["group_hash"] == "g1"
        assert len(valid) == 0
        # Not flipped to valid/auto_approved.
        assert needs_human[0]["validation_status"] == "needs-human-decision"
        assert needs_human[0].get("auto_approved") is not True

    def test_claude_decide_not_auto_approved_by_approve_importance(self):
        """A marked HIGH group is not auto-approved by --approve-importance HIGH."""
        groups = [self._group(claude_decide=True)]
        valid, needs_human, skipped, _ = filter_items(
            groups, approve_importance_levels=["HIGH"]
        )
        assert len(needs_human) == 1
        assert len(valid) == 0

    def test_unmarked_group_still_auto_approved(self):
        """The short-circuit is scoped to marked groups only."""
        groups = [self._group()]  # no claude_decide
        valid, needs_human, skipped, _ = filter_items(
            groups, approve_all_human=True
        )
        assert len(valid) == 1
        assert valid[0].get("auto_approved") is True
        assert len(needs_human) == 0

    def test_bulk_skip_still_honoured_for_marked_group(self):
        """Bulk *skip* still applies to a marked group (skip is a safety choice)."""
        groups = [self._group(claude_decide=True)]
        valid, needs_human, skipped, _ = filter_items(
            groups, skip_all_human=True
        )
        assert len(skipped) == 1
        assert len(needs_human) == 0


class TestValidateClaudeDecideItemsHonored:
    """Section 4c: post-hoc audit that pre-marked items reached the judge."""

    def test_empty_when_no_marked_items(self):
        is_valid, warnings = validate_claude_decide_items_honored(
            {"claude_decide_item_ids": []}, {}
        )
        assert is_valid is True
        assert warnings == []

    def test_ok_when_all_routed_to_judge(self):
        config = {"claude_decide_item_ids": ["h1", "h2"]}
        decisions = {
            "h1": {"batch_context": {"decision_source": "claude_auto_decide"}},
            "h2": {"batch_context": {"decision_source": "claude_auto_decide_salvage"}},
        }
        is_valid, warnings = validate_claude_decide_items_honored(config, decisions)
        assert is_valid is True
        assert warnings == []

    def test_flags_missing_decision(self):
        config = {"claude_decide_item_ids": ["h1"]}
        is_valid, warnings = validate_claude_decide_items_honored(config, {})
        assert is_valid is False
        assert any("h1" in w for w in warnings)

    def test_flags_wrong_source(self):
        config = {"claude_decide_item_ids": ["h1"]}
        decisions = {
            "h1": {"batch_context": {"decision_source": "interactive"}},
        }
        is_valid, warnings = validate_claude_decide_items_honored(config, decisions)
        assert is_valid is False
        assert any("interactive" in w for w in warnings)

    def test_flags_decision_without_batch_context(self):
        config = {"claude_decide_item_ids": ["h1"]}
        decisions = {"h1": {"decision": "approved"}}
        is_valid, warnings = validate_claude_decide_items_honored(config, decisions)
        assert is_valid is False
        assert len(warnings) == 1
