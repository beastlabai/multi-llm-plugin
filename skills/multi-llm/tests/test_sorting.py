"""Unit tests for priority sorting functions in html_report_generator.

Tests cover:
- compute_max_importance(): extracting highest importance from suggestion lists
- sort_groups_by_priority(): sorting camelCase report-data dicts
- sort_raw_groups_by_priority(): sorting snake_case grouped.json dicts
- sort_consolidated_groups_by_priority(): sorting consolidated group dicts
- derive_aggregate_validation_status(): aggregating statuses by priority
- build_sort_config(): constructing the sortConfig for HTML report injection
- Determinism and ID stability guarantees
"""

import copy
import pytest

from utils.html_report_generator import (
    VALIDATION_ORDER,
    UNKNOWN_STATUS_RANK,
    IMPORTANCE_ORDER,
    UNKNOWN_IMPORTANCE_RANK,
    compute_max_importance,
    sort_groups_by_priority,
    sort_raw_groups_by_priority,
    sort_consolidated_groups_by_priority,
    derive_aggregate_validation_status,
    build_sort_config,
)


# ============================================================================
# compute_max_importance
# ============================================================================


class TestComputeMaxImportance:
    """Tests for compute_max_importance()."""

    def test_returns_high_when_present(self):
        """HIGH should be returned when at least one suggestion has HIGH."""
        suggestions = [
            {"importance": "LOW"},
            {"importance": "HIGH"},
            {"importance": "MEDIUM"},
        ]
        assert compute_max_importance(suggestions) == "HIGH"

    def test_returns_medium_when_highest(self):
        """MEDIUM returned when no HIGH present."""
        suggestions = [
            {"importance": "LOW"},
            {"importance": "MEDIUM"},
        ]
        assert compute_max_importance(suggestions) == "MEDIUM"

    def test_returns_low_when_all_low(self):
        """LOW returned when all suggestions are LOW."""
        suggestions = [
            {"importance": "LOW"},
            {"importance": "LOW"},
        ]
        assert compute_max_importance(suggestions) == "LOW"

    def test_empty_list_returns_empty(self):
        """Empty list should return empty string."""
        assert compute_max_importance([]) == ""

    def test_none_input_returns_empty(self):
        """None input should return empty string."""
        assert compute_max_importance(None) == ""

    def test_mixed_case_normalizes(self):
        """Lowercase 'high' should normalize to 'HIGH' via .upper()."""
        suggestions = [
            {"importance": "low"},
            {"importance": "high"},
        ]
        assert compute_max_importance(suggestions) == "HIGH"

    def test_mixed_case_medium(self):
        """Mixed case 'Medium' normalizes correctly."""
        suggestions = [
            {"importance": "low"},
            {"importance": "Medium"},
        ]
        assert compute_max_importance(suggestions) == "MEDIUM"

    def test_missing_importance_key(self):
        """Suggestions without 'importance' key should be treated as unknown."""
        suggestions = [
            {"title": "No importance field"},
            {"title": "Also missing"},
        ]
        assert compute_max_importance(suggestions) == ""

    def test_none_importance_value(self):
        """importance=None treated as unknown."""
        suggestions = [{"importance": None}]
        assert compute_max_importance(suggestions) == ""

    def test_single_high(self):
        """Single HIGH suggestion returns HIGH."""
        assert compute_max_importance([{"importance": "HIGH"}]) == "HIGH"

    def test_unrecognized_importance_ignored(self):
        """Unrecognized importance values should not win over known ones."""
        suggestions = [
            {"importance": "CRITICAL"},  # Not a recognized value
            {"importance": "LOW"},
        ]
        # LOW is recognized (rank 2), CRITICAL is unknown (rank 3)
        assert compute_max_importance(suggestions) == "LOW"


# ============================================================================
# sort_groups_by_priority (camelCase dicts)
# ============================================================================


class TestSortGroupsByPriority:
    """Tests for sort_groups_by_priority() with camelCase report-data dicts."""

    def _make_group(self, status, importance, index=None):
        """Helper to create a camelCase group dict."""
        g = {
            "validationStatus": status,
            "maxImportance": importance,
            "theme": f"{status}-{importance}",
        }
        if index is not None:
            g["originalIndex"] = index
        return g

    def test_sorts_by_validation_status_rank(self):
        """Groups sort by validation status: needs-human-decision < valid < validation_failed < invalid < pending."""
        groups = [
            self._make_group("pending", "HIGH"),
            self._make_group("valid", "HIGH"),
            self._make_group("needs-human-decision", "HIGH"),
            self._make_group("invalid", "HIGH"),
            self._make_group("validation_failed", "HIGH"),
        ]
        result = sort_groups_by_priority(groups)
        statuses = [g["validationStatus"] for g in result]
        assert statuses == [
            "needs-human-decision",
            "valid",
            "validation_failed",
            "invalid",
            "pending",
        ]

    def test_same_status_sorts_by_importance(self):
        """Within same validation status, sorts by maxImportance: HIGH < MEDIUM < LOW."""
        groups = [
            self._make_group("valid", "LOW"),
            self._make_group("valid", "HIGH"),
            self._make_group("valid", "MEDIUM"),
        ]
        result = sort_groups_by_priority(groups)
        importances = [g["maxImportance"] for g in result]
        assert importances == ["HIGH", "MEDIUM", "LOW"]

    def test_same_status_and_importance_sorts_by_original_index(self):
        """Stable tie-breaking by originalIndex for identical status+importance."""
        groups = [
            self._make_group("valid", "HIGH"),
            self._make_group("valid", "HIGH"),
            self._make_group("valid", "HIGH"),
        ]
        result = sort_groups_by_priority(groups)
        # originalIndex should be stamped 1, 2, 3 and preserved in order
        indices = [g["originalIndex"] for g in result]
        assert indices == [1, 2, 3]

    def test_stamps_original_index_1_based(self):
        """originalIndex is stamped 1-based on each group."""
        groups = [
            self._make_group("valid", "HIGH"),
            self._make_group("invalid", "LOW"),
        ]
        result = sort_groups_by_priority(groups)
        # Group that was at position 0 gets originalIndex=1
        # Group that was at position 1 gets originalIndex=2
        original_indices = sorted(g["originalIndex"] for g in result)
        assert original_indices == [1, 2]

    def test_does_not_restamp_existing_original_index(self):
        """If originalIndex already present, it should not be overwritten."""
        groups = [
            self._make_group("invalid", "LOW", index=42),
            self._make_group("valid", "HIGH", index=7),
        ]
        result = sort_groups_by_priority(groups)
        # valid sorts before invalid
        assert result[0]["originalIndex"] == 7
        assert result[1]["originalIndex"] == 42

    def test_returns_new_list(self):
        """sort_groups_by_priority returns a new list, not the input modified in-place."""
        groups = [
            self._make_group("invalid", "LOW"),
            self._make_group("valid", "HIGH"),
        ]
        result = sort_groups_by_priority(groups)
        # The result list object should be different from the input
        assert result is not groups
        # But the sorted order should differ from input order
        assert result[0]["validationStatus"] == "valid"

    def test_unknown_validation_status_sorts_after_pending(self):
        """Unknown/missing validation status sorts after pending (rank 5 > 4)."""
        groups = [
            self._make_group("pending", "HIGH"),
            self._make_group("unknown_status", "HIGH"),
            self._make_group("valid", "HIGH"),
        ]
        result = sort_groups_by_priority(groups)
        statuses = [g["validationStatus"] for g in result]
        assert statuses == ["valid", "pending", "unknown_status"]

    def test_missing_validation_status_sorts_after_pending(self):
        """Empty/missing validationStatus sorts after pending."""
        groups = [
            self._make_group("pending", "HIGH"),
            {"maxImportance": "HIGH", "theme": "no-status"},
            self._make_group("valid", "HIGH"),
        ]
        result = sort_groups_by_priority(groups)
        # Empty string gets UNKNOWN_STATUS_RANK=5, pending is 4
        # So order: valid(1), pending(4), no-status(5)
        assert result[0]["validationStatus"] == "valid"
        assert result[1]["validationStatus"] == "pending"

    def test_unknown_importance_sorts_after_low(self):
        """Unknown/missing importance sorts after LOW (rank 3)."""
        groups = [
            self._make_group("valid", "LOW"),
            self._make_group("valid", "WEIRD"),
            self._make_group("valid", "HIGH"),
        ]
        result = sort_groups_by_priority(groups)
        importances = [g["maxImportance"] for g in result]
        # HIGH(0), LOW(2), WEIRD(3=UNKNOWN_IMPORTANCE_RANK)
        assert importances == ["HIGH", "LOW", "WEIRD"]

    def test_empty_importance_sorts_after_low(self):
        """Empty string importance sorts after LOW."""
        groups = [
            self._make_group("valid", "LOW"),
            self._make_group("valid", ""),
            self._make_group("valid", "HIGH"),
        ]
        result = sort_groups_by_priority(groups)
        importances = [g["maxImportance"] for g in result]
        assert importances == ["HIGH", "LOW", ""]

    def test_empty_input_returns_empty(self):
        """Empty input list returns empty list."""
        assert sort_groups_by_priority([]) == []

    def test_single_item_returns_with_original_index(self):
        """Single item gets originalIndex stamped and returned."""
        groups = [self._make_group("valid", "HIGH")]
        result = sort_groups_by_priority(groups)
        assert len(result) == 1
        assert result[0]["originalIndex"] == 1
        assert result[0]["validationStatus"] == "valid"


# ============================================================================
# sort_raw_groups_by_priority (snake_case dicts)
# ============================================================================


class TestSortRawGroupsByPriority:
    """Tests for sort_raw_groups_by_priority() with snake_case grouped.json format."""

    def _make_raw_group(self, status, suggestions, index=None):
        """Helper to create a snake_case raw group dict."""
        g = {
            "validation_status": status,
            "suggestions": suggestions,
            "theme": f"raw-{status}",
        }
        if index is not None:
            g["originalIndex"] = index
        return g

    def test_works_with_raw_format(self):
        """Properly reads validation_status and computes importance from suggestions."""
        groups = [
            self._make_raw_group("invalid", [{"importance": "HIGH"}]),
            self._make_raw_group("valid", [{"importance": "LOW"}]),
            self._make_raw_group("needs-human-decision", [{"importance": "MEDIUM"}]),
        ]
        result = sort_raw_groups_by_priority(groups)
        statuses = [g["validation_status"] for g in result]
        assert statuses == ["needs-human-decision", "valid", "invalid"]

    def test_computes_importance_from_suggestions(self):
        """Importance is computed from the suggestions list."""
        groups = [
            self._make_raw_group("valid", [{"importance": "LOW"}, {"importance": "HIGH"}]),
            self._make_raw_group("valid", [{"importance": "MEDIUM"}]),
        ]
        result = sort_raw_groups_by_priority(groups)
        # First group has HIGH (computed from suggestions), should come first
        assert result[0]["suggestions"][0]["importance"] == "LOW"  # The group with HIGH max
        assert result[1]["suggestions"][0]["importance"] == "MEDIUM"

    def test_stamps_original_index(self):
        """originalIndex is stamped on raw groups."""
        groups = [
            self._make_raw_group("valid", [{"importance": "HIGH"}]),
            self._make_raw_group("valid", [{"importance": "LOW"}]),
        ]
        result = sort_raw_groups_by_priority(groups)
        original_indices = sorted(g["originalIndex"] for g in result)
        assert original_indices == [1, 2]

    def test_sorts_correctly_full_ordering(self):
        """Full ordering: status first, then importance, then originalIndex."""
        groups = [
            self._make_raw_group("invalid", [{"importance": "HIGH"}]),
            self._make_raw_group("valid", [{"importance": "LOW"}]),
            self._make_raw_group("valid", [{"importance": "HIGH"}]),
            self._make_raw_group("needs-human-decision", [{"importance": "MEDIUM"}]),
        ]
        result = sort_raw_groups_by_priority(groups)
        expected_order = [
            ("needs-human-decision", "MEDIUM"),
            ("valid", "HIGH"),
            ("valid", "LOW"),
            ("invalid", "HIGH"),
        ]
        actual = [
            (g["validation_status"], compute_max_importance(g["suggestions"]))
            for g in result
        ]
        assert actual == expected_order

    def test_empty_suggestions_sort_last_within_status(self):
        """Groups with empty suggestions get importance="" which sorts after LOW."""
        groups = [
            self._make_raw_group("valid", []),
            self._make_raw_group("valid", [{"importance": "LOW"}]),
        ]
        result = sort_raw_groups_by_priority(groups)
        # Group with LOW importance should come before group with empty
        assert len(result[0]["suggestions"]) == 1
        assert len(result[1]["suggestions"]) == 0

    def test_none_status_treated_as_empty(self):
        """None validation_status should be treated as empty string."""
        groups = [
            {"validation_status": None, "suggestions": [{"importance": "HIGH"}], "theme": "none-status"},
            self._make_raw_group("valid", [{"importance": "HIGH"}]),
        ]
        result = sort_raw_groups_by_priority(groups)
        # valid (rank 1) before None/empty (rank 5)
        assert result[0]["validation_status"] == "valid"


# ============================================================================
# sort_consolidated_groups_by_priority
# ============================================================================


class TestSortConsolidatedGroupsByPriority:
    """Tests for sort_consolidated_groups_by_priority()."""

    def _make_consolidated(self, status, importance, index=None):
        """Helper to create a consolidated group dict."""
        g = {
            "validation_status": status,
            "importance": importance,
            "theme": f"consolidated-{status}-{importance}",
        }
        if index is not None:
            g["originalIndex"] = index
        return g

    def test_sorts_by_validation_status_then_importance(self):
        """Sorts by validation_status rank, then importance."""
        groups = [
            self._make_consolidated("invalid", "HIGH"),
            self._make_consolidated("valid", "MEDIUM"),
            self._make_consolidated("needs-human-decision", "LOW"),
        ]
        result = sort_consolidated_groups_by_priority(groups)
        statuses = [g["validation_status"] for g in result]
        assert statuses == ["needs-human-decision", "valid", "invalid"]

    def test_uses_direct_importance_field(self):
        """Uses the direct 'importance' field, not nested suggestions."""
        groups = [
            self._make_consolidated("valid", "LOW"),
            self._make_consolidated("valid", "HIGH"),
        ]
        result = sort_consolidated_groups_by_priority(groups)
        importances = [g["importance"] for g in result]
        assert importances == ["HIGH", "LOW"]

    def test_stamps_original_index(self):
        """originalIndex is stamped on consolidated groups."""
        groups = [
            self._make_consolidated("valid", "HIGH"),
        ]
        result = sort_consolidated_groups_by_priority(groups)
        assert result[0]["originalIndex"] == 1

    def test_none_status_treated_as_empty(self):
        """None validation_status should sort after pending."""
        groups = [
            {"validation_status": None, "importance": "HIGH", "theme": "none-status"},
            self._make_consolidated("valid", "HIGH"),
        ]
        result = sort_consolidated_groups_by_priority(groups)
        assert result[0]["validation_status"] == "valid"

    def test_none_importance_treated_as_empty(self):
        """None importance should sort after LOW."""
        groups = [
            {"validation_status": "valid", "importance": None, "theme": "none-imp"},
            self._make_consolidated("valid", "LOW"),
        ]
        result = sort_consolidated_groups_by_priority(groups)
        assert result[0]["importance"] == "LOW"


# ============================================================================
# derive_aggregate_validation_status
# ============================================================================


class TestDeriveAggregateValidationStatus:
    """Tests for derive_aggregate_validation_status()."""

    def test_empty_list_returns_empty(self):
        """Empty list returns empty string."""
        assert derive_aggregate_validation_status([]) == ""

    def test_single_status_returned(self):
        """Single-element list returns that status."""
        assert derive_aggregate_validation_status(["valid"]) == "valid"

    def test_needs_human_decision_wins_over_all(self):
        """needs-human-decision (rank 0) wins over all other statuses."""
        statuses = ["valid", "invalid", "needs-human-decision", "pending"]
        assert derive_aggregate_validation_status(statuses) == "needs-human-decision"

    def test_valid_beats_validation_failed(self):
        """valid (rank 1) beats validation_failed (rank 2)."""
        statuses = ["validation_failed", "valid"]
        assert derive_aggregate_validation_status(statuses) == "valid"

    def test_valid_beats_invalid(self):
        """valid (rank 1) beats invalid (rank 3)."""
        statuses = ["invalid", "valid"]
        assert derive_aggregate_validation_status(statuses) == "valid"

    def test_validation_failed_beats_invalid(self):
        """validation_failed (rank 2) beats invalid (rank 3)."""
        statuses = ["invalid", "validation_failed"]
        assert derive_aggregate_validation_status(statuses) == "validation_failed"

    def test_invalid_beats_pending(self):
        """invalid (rank 3) beats pending (rank 4)."""
        statuses = ["pending", "invalid"]
        assert derive_aggregate_validation_status(statuses) == "invalid"

    def test_unknown_statuses_sort_to_lowest_priority(self):
        """Unknown statuses get UNKNOWN_STATUS_RANK=5, lose to all known."""
        statuses = ["unknown_thing", "pending"]
        assert derive_aggregate_validation_status(statuses) == "pending"

    def test_all_unknown_returns_first_by_min(self):
        """When all statuses are unknown, min() still returns one of them."""
        statuses = ["alpha", "beta"]
        # Both have rank 5; min() is stable and returns whichever comes first
        result = derive_aggregate_validation_status(statuses)
        assert result in ("alpha", "beta")

    def test_needs_human_decision_with_all_others(self):
        """needs-human-decision wins even with every other status present."""
        statuses = [
            "valid",
            "invalid",
            "validation_failed",
            "pending",
            "needs-human-decision",
        ]
        assert derive_aggregate_validation_status(statuses) == "needs-human-decision"


# ============================================================================
# build_sort_config
# ============================================================================


class TestBuildSortConfig:
    """Tests for build_sort_config()."""

    def test_contains_validation_order(self):
        """sortConfig has validationOrder matching the module constant."""
        config = build_sort_config()
        assert config["validationOrder"] == VALIDATION_ORDER

    def test_contains_importance_order(self):
        """sortConfig has importanceOrder matching the module constant."""
        config = build_sort_config()
        assert config["importanceOrder"] == IMPORTANCE_ORDER

    def test_unknown_status_rank(self):
        """sortConfig has unknownStatusRank = 5."""
        config = build_sort_config()
        assert config["unknownStatusRank"] == 5

    def test_unknown_importance_rank(self):
        """sortConfig has unknownImportanceRank = 3."""
        config = build_sort_config()
        assert config["unknownImportanceRank"] == 3

    def test_returns_dict(self):
        """build_sort_config returns a dict."""
        config = build_sort_config()
        assert isinstance(config, dict)

    def test_all_expected_keys_present(self):
        """All four expected keys are present."""
        config = build_sort_config()
        expected_keys = {
            "validationOrder",
            "importanceOrder",
            "unknownStatusRank",
            "unknownImportanceRank",
        }
        assert set(config.keys()) == expected_keys


# ============================================================================
# Determinism
# ============================================================================


class TestDeterminism:
    """Sorting the same input twice produces identical output."""

    def test_sort_groups_deterministic(self):
        """sort_groups_by_priority is deterministic across two calls."""
        groups = [
            {"validationStatus": "invalid", "maxImportance": "HIGH", "theme": "a"},
            {"validationStatus": "valid", "maxImportance": "LOW", "theme": "b"},
            {"validationStatus": "needs-human-decision", "maxImportance": "MEDIUM", "theme": "c"},
            {"validationStatus": "valid", "maxImportance": "HIGH", "theme": "d"},
            {"validationStatus": "pending", "maxImportance": "LOW", "theme": "e"},
        ]
        # Deep copy to avoid mutation side-effects on originalIndex
        result1 = sort_groups_by_priority(copy.deepcopy(groups))
        result2 = sort_groups_by_priority(copy.deepcopy(groups))
        assert result1 == result2

    def test_sort_raw_groups_deterministic(self):
        """sort_raw_groups_by_priority is deterministic across two calls."""
        groups = [
            {"validation_status": "invalid", "suggestions": [{"importance": "HIGH"}]},
            {"validation_status": "valid", "suggestions": [{"importance": "LOW"}]},
            {"validation_status": "needs-human-decision", "suggestions": [{"importance": "MEDIUM"}]},
        ]
        result1 = sort_raw_groups_by_priority(copy.deepcopy(groups))
        result2 = sort_raw_groups_by_priority(copy.deepcopy(groups))
        assert result1 == result2

    def test_sort_consolidated_deterministic(self):
        """sort_consolidated_groups_by_priority is deterministic across two calls."""
        groups = [
            {"validation_status": "invalid", "importance": "HIGH"},
            {"validation_status": "valid", "importance": "LOW"},
        ]
        result1 = sort_consolidated_groups_by_priority(copy.deepcopy(groups))
        result2 = sort_consolidated_groups_by_priority(copy.deepcopy(groups))
        assert result1 == result2


# ============================================================================
# ID Stability (originalIndex)
# ============================================================================


class TestOriginalIndexStability:
    """originalIndex values match pre-sort positions, not post-sort positions."""

    def test_original_index_reflects_input_position(self):
        """originalIndex values should reflect 1-based position in the original input."""
        groups = [
            {"validationStatus": "pending", "maxImportance": "LOW", "theme": "first"},
            {"validationStatus": "valid", "maxImportance": "HIGH", "theme": "second"},
            {"validationStatus": "needs-human-decision", "maxImportance": "MEDIUM", "theme": "third"},
        ]
        result = sort_groups_by_priority(groups)

        # After sorting: needs-human-decision (was 3rd), valid (was 2nd), pending (was 1st)
        assert result[0]["theme"] == "third"
        assert result[0]["originalIndex"] == 3  # Was at position 2 (0-based), stamped as 3

        assert result[1]["theme"] == "second"
        assert result[1]["originalIndex"] == 2  # Was at position 1 (0-based), stamped as 2

        assert result[2]["theme"] == "first"
        assert result[2]["originalIndex"] == 1  # Was at position 0 (0-based), stamped as 1

    def test_pre_existing_original_index_preserved(self):
        """Pre-existing originalIndex values should not be overwritten."""
        groups = [
            {"validationStatus": "pending", "maxImportance": "LOW", "originalIndex": 10},
            {"validationStatus": "valid", "maxImportance": "HIGH", "originalIndex": 20},
        ]
        result = sort_groups_by_priority(groups)

        # valid sorts before pending
        assert result[0]["originalIndex"] == 20
        assert result[1]["originalIndex"] == 10

    def test_original_index_stable_across_re_sort(self):
        """Re-sorting an already-sorted list preserves originalIndex values."""
        groups = [
            {"validationStatus": "invalid", "maxImportance": "LOW", "theme": "a"},
            {"validationStatus": "valid", "maxImportance": "HIGH", "theme": "b"},
            {"validationStatus": "needs-human-decision", "maxImportance": "MEDIUM", "theme": "c"},
        ]
        first_sort = sort_groups_by_priority(groups)
        first_indices = [g["originalIndex"] for g in first_sort]

        # Re-sort the already-sorted result
        second_sort = sort_groups_by_priority(first_sort)
        second_indices = [g["originalIndex"] for g in second_sort]

        # originalIndex values should be identical (not re-stamped)
        assert first_indices == second_indices
