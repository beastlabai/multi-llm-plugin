"""Unit tests for the importance utility module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.importance import (
    IMPORTANCE_ORDER,
    compare_importance,
    filter_by_importance,
    get_highest_importance,
    normalize_importance,
)


class TestImportanceOrder:
    """Tests for the IMPORTANCE_ORDER constant."""

    def test_correct_order(self):
        """IMPORTANCE_ORDER has correct order from highest to lowest."""
        assert IMPORTANCE_ORDER == ["HIGH", "MEDIUM", "LOW"]

    def test_high_is_first(self):
        """HIGH is the first (highest) importance level."""
        assert IMPORTANCE_ORDER[0] == "HIGH"

    def test_low_is_last(self):
        """LOW is the last (lowest) importance level."""
        assert IMPORTANCE_ORDER[-1] == "LOW"

    def test_medium_is_middle(self):
        """MEDIUM is in the middle."""
        assert IMPORTANCE_ORDER[1] == "MEDIUM"


class TestGetHighestImportance:
    """Tests for the get_highest_importance function."""

    def test_suggestions_with_high_importance(self):
        """Group with suggestions having HIGH importance returns HIGH."""
        group = {
            "suggestions": [
                {"importance": "HIGH", "text": "Critical fix"},
                {"importance": "LOW", "text": "Minor suggestion"},
            ]
        }
        assert get_highest_importance(group) == "HIGH"

    def test_suggestions_with_only_medium_importance(self):
        """Group with suggestions having only MEDIUM importance returns MEDIUM."""
        group = {
            "suggestions": [
                {"importance": "MEDIUM", "text": "Suggestion 1"},
                {"importance": "MEDIUM", "text": "Suggestion 2"},
            ]
        }
        assert get_highest_importance(group) == "MEDIUM"

    def test_suggestions_with_only_low_importance(self):
        """Group with suggestions having only LOW importance returns LOW."""
        group = {
            "suggestions": [
                {"importance": "LOW", "text": "Minor 1"},
                {"importance": "LOW", "text": "Minor 2"},
            ]
        }
        assert get_highest_importance(group) == "LOW"

    def test_mixed_importance_levels_returns_highest(self):
        """Group with mixed importance levels returns the highest one."""
        group = {
            "suggestions": [
                {"importance": "LOW", "text": "Low priority"},
                {"importance": "HIGH", "text": "High priority"},
                {"importance": "MEDIUM", "text": "Medium priority"},
            ]
        }
        assert get_highest_importance(group) == "HIGH"

    def test_issues_instead_of_suggestions(self):
        """Group with issues instead of suggestions works correctly."""
        group = {
            "issues": [
                {"importance": "HIGH", "text": "Critical issue"},
                {"importance": "LOW", "text": "Minor issue"},
            ]
        }
        assert get_highest_importance(group) == "HIGH"

    def test_empty_group_returns_medium(self):
        """Empty group returns MEDIUM as default."""
        group = {}
        assert get_highest_importance(group) == "MEDIUM"

    def test_empty_suggestions_list_returns_medium(self):
        """Group with empty suggestions list returns MEDIUM."""
        group = {"suggestions": []}
        assert get_highest_importance(group) == "MEDIUM"

    def test_items_missing_importance_field_defaults_to_medium(self):
        """Items missing importance field default to MEDIUM."""
        group = {
            "suggestions": [
                {"text": "No importance field"},
                {"text": "Also no importance"},
            ]
        }
        assert get_highest_importance(group) == "MEDIUM"

    def test_case_insensitive_importance_values(self):
        """Importance values are case insensitive."""
        group = {
            "suggestions": [
                {"importance": "high", "text": "Lowercase high"},
                {"importance": "Low", "text": "Mixed case low"},
            ]
        }
        assert get_highest_importance(group) == "HIGH"

    def test_medium_case_insensitive(self):
        """Medium importance is case insensitive."""
        group = {
            "suggestions": [
                {"importance": "medium", "text": "Lowercase medium"},
            ]
        }
        assert get_highest_importance(group) == "MEDIUM"


class TestFilterByImportance:
    """Tests for the filter_by_importance function."""

    def test_filter_min_low_includes_all(self):
        """Filter with min_importance='LOW' includes all items."""
        items = [
            {"importance": "HIGH", "text": "High"},
            {"importance": "MEDIUM", "text": "Medium"},
            {"importance": "LOW", "text": "Low"},
        ]
        result = filter_by_importance(items, min_importance="LOW")
        assert len(result) == 3

    def test_filter_min_medium_excludes_low(self):
        """Filter with min_importance='MEDIUM' excludes LOW items."""
        items = [
            {"importance": "HIGH", "text": "High"},
            {"importance": "MEDIUM", "text": "Medium"},
            {"importance": "LOW", "text": "Low"},
        ]
        result = filter_by_importance(items, min_importance="MEDIUM")
        assert len(result) == 2
        assert all(item["importance"] in ["HIGH", "MEDIUM"] for item in result)

    def test_filter_min_high_only_includes_high(self):
        """Filter with min_importance='HIGH' only includes HIGH items."""
        items = [
            {"importance": "HIGH", "text": "High"},
            {"importance": "MEDIUM", "text": "Medium"},
            {"importance": "LOW", "text": "Low"},
        ]
        result = filter_by_importance(items, min_importance="HIGH")
        assert len(result) == 1
        assert result[0]["importance"] == "HIGH"

    def test_items_missing_importance_field_default_to_medium(self):
        """Items missing importance field are treated as MEDIUM."""
        items = [
            {"importance": "HIGH", "text": "High"},
            {"text": "No importance (defaults to MEDIUM)"},
            {"importance": "LOW", "text": "Low"},
        ]
        # With min_importance=MEDIUM, should include HIGH and the default MEDIUM
        result = filter_by_importance(items, min_importance="MEDIUM")
        assert len(result) == 2
        assert {"text": "No importance (defaults to MEDIUM)"} in result

    def test_case_insensitive_min_importance(self):
        """min_importance parameter is case insensitive."""
        items = [
            {"importance": "HIGH", "text": "High"},
            {"importance": "MEDIUM", "text": "Medium"},
            {"importance": "LOW", "text": "Low"},
        ]
        result = filter_by_importance(items, min_importance="medium")
        assert len(result) == 2

    def test_empty_list_returns_empty(self):
        """Filtering empty list returns empty list."""
        result = filter_by_importance([], min_importance="LOW")
        assert result == []

    def test_default_min_importance_is_low(self):
        """Default min_importance is LOW (includes all)."""
        items = [
            {"importance": "HIGH", "text": "High"},
            {"importance": "MEDIUM", "text": "Medium"},
            {"importance": "LOW", "text": "Low"},
        ]
        result = filter_by_importance(items)
        assert len(result) == 3

    def test_case_insensitive_item_importance(self):
        """Item importance values are case insensitive."""
        items = [
            {"importance": "high", "text": "Lowercase high"},
            {"importance": "Medium", "text": "Mixed case medium"},
            {"importance": "LOW", "text": "Uppercase low"},
        ]
        result = filter_by_importance(items, min_importance="MEDIUM")
        assert len(result) == 2


class TestCompareImportance:
    """Tests for the compare_importance function."""

    def test_high_vs_medium_returns_negative(self):
        """HIGH vs MEDIUM returns -1 (HIGH is more important)."""
        item1 = {"importance": "HIGH"}
        item2 = {"importance": "MEDIUM"}
        assert compare_importance(item1, item2) == -1

    def test_medium_vs_low_returns_negative(self):
        """MEDIUM vs LOW returns -1 (MEDIUM is more important)."""
        item1 = {"importance": "MEDIUM"}
        item2 = {"importance": "LOW"}
        assert compare_importance(item1, item2) == -1

    def test_high_vs_low_returns_negative(self):
        """HIGH vs LOW returns -1 (HIGH is more important)."""
        item1 = {"importance": "HIGH"}
        item2 = {"importance": "LOW"}
        assert compare_importance(item1, item2) == -1

    def test_low_vs_high_returns_positive(self):
        """LOW vs HIGH returns 1 (LOW is less important)."""
        item1 = {"importance": "LOW"}
        item2 = {"importance": "HIGH"}
        assert compare_importance(item1, item2) == 1

    def test_medium_vs_high_returns_positive(self):
        """MEDIUM vs HIGH returns 1 (MEDIUM is less important)."""
        item1 = {"importance": "MEDIUM"}
        item2 = {"importance": "HIGH"}
        assert compare_importance(item1, item2) == 1

    def test_low_vs_medium_returns_positive(self):
        """LOW vs MEDIUM returns 1 (LOW is less important)."""
        item1 = {"importance": "LOW"}
        item2 = {"importance": "MEDIUM"}
        assert compare_importance(item1, item2) == 1

    def test_same_importance_returns_zero(self):
        """Same importance returns 0."""
        item1 = {"importance": "HIGH"}
        item2 = {"importance": "HIGH"}
        assert compare_importance(item1, item2) == 0

        item1 = {"importance": "MEDIUM"}
        item2 = {"importance": "MEDIUM"}
        assert compare_importance(item1, item2) == 0

        item1 = {"importance": "LOW"}
        item2 = {"importance": "LOW"}
        assert compare_importance(item1, item2) == 0

    def test_invalid_importance_defaults_to_medium(self):
        """Invalid importance values default to MEDIUM."""
        item1 = {"importance": "INVALID"}
        item2 = {"importance": "MEDIUM"}
        assert compare_importance(item1, item2) == 0

    def test_missing_importance_defaults_to_medium(self):
        """Missing importance field defaults to MEDIUM."""
        item1 = {"text": "No importance"}
        item2 = {"importance": "MEDIUM"}
        assert compare_importance(item1, item2) == 0

    def test_both_invalid_returns_zero(self):
        """Both items with invalid importance return 0."""
        item1 = {"importance": "INVALID1"}
        item2 = {"importance": "INVALID2"}
        assert compare_importance(item1, item2) == 0

    def test_case_insensitive(self):
        """Importance comparison is case insensitive."""
        item1 = {"importance": "high"}
        item2 = {"importance": "medium"}
        assert compare_importance(item1, item2) == -1


class TestNormalizeImportance:
    """Tests for the normalize_importance function."""

    def test_valid_uppercase_input(self):
        """Valid uppercase input is returned as-is."""
        assert normalize_importance("HIGH") == "HIGH"
        assert normalize_importance("MEDIUM") == "MEDIUM"
        assert normalize_importance("LOW") == "LOW"

    def test_valid_lowercase_input(self):
        """Valid lowercase input is converted to uppercase."""
        assert normalize_importance("high") == "HIGH"
        assert normalize_importance("medium") == "MEDIUM"
        assert normalize_importance("low") == "LOW"

    def test_valid_mixed_case_input(self):
        """Valid mixed case input is converted to uppercase."""
        assert normalize_importance("High") == "HIGH"
        assert normalize_importance("Medium") == "MEDIUM"
        assert normalize_importance("Low") == "LOW"
        assert normalize_importance("hIgH") == "HIGH"
        assert normalize_importance("mEdIuM") == "MEDIUM"

    def test_input_with_whitespace(self):
        """Input with leading/trailing whitespace is trimmed."""
        assert normalize_importance("  HIGH  ") == "HIGH"
        assert normalize_importance("\tMEDIUM\n") == "MEDIUM"
        assert normalize_importance(" low ") == "LOW"

    def test_invalid_input_returns_medium(self):
        """Invalid input returns MEDIUM as default."""
        assert normalize_importance("INVALID") == "MEDIUM"
        assert normalize_importance("CRITICAL") == "MEDIUM"
        assert normalize_importance("") == "MEDIUM"
        assert normalize_importance("   ") == "MEDIUM"
        assert normalize_importance("HIGHEST") == "MEDIUM"
        assert normalize_importance("123") == "MEDIUM"
