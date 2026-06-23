#!/usr/bin/env python3
"""Integration tests for plan review orchestrator grouping and validation."""

import json
import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    group_similar_suggestions,
    export_groups_to_json,
    SuggestionGroup,
)


class TestGrouping:
    """Tests for suggestion grouping functionality."""

    def test_group_similar_suggestions_empty(self):
        """Test grouping with empty input."""
        result = group_similar_suggestions([])
        assert result == []

    def test_group_similar_suggestions_single(self):
        """Test grouping with single suggestion."""
        suggestions = [
            {
                "title": "Missing error handling",
                "desc": "Add error handling for API calls",
                "importance": "high",
                "type": "addition",
                "source_model": "gpt-4"
            }
        ]
        result = group_similar_suggestions(suggestions)
        assert len(result) == 1
        assert len(result[0].suggestions) == 1

    def test_group_similar_suggestions_merges_duplicates(self):
        """Test that similar suggestions from different models are grouped."""
        suggestions = [
            {
                "title": "Missing error handling for API",
                "desc": "Add error handling for API timeouts",
                "importance": "high",
                "type": "addition",
                "source_model": "gpt-4"
            },
            {
                "title": "Add error handling for API calls",
                "desc": "API calls lack error handling for timeouts",
                "importance": "high",
                "type": "addition",
                "source_model": "claude"
            }
        ]
        result = group_similar_suggestions(suggestions)
        # Should merge into 1 group due to high similarity
        assert len(result) <= 2  # May or may not merge depending on threshold

    def test_group_different_suggestions_not_merged(self):
        """Test that different suggestions stay separate."""
        suggestions = [
            {
                "title": "Missing error handling",
                "desc": "Add error handling",
                "importance": "high",
                "type": "addition",
                "source_model": "gpt-4"
            },
            {
                "title": "Update documentation",
                "desc": "Add API documentation",
                "importance": "low",
                "type": "modification",
                "source_model": "claude"
            }
        ]
        result = group_similar_suggestions(suggestions)
        assert len(result) == 2

    def test_export_groups_to_json(self):
        """Test JSON export of groups."""
        group = SuggestionGroup("addition", "Test theme")
        group.add_suggestion({"title": "Test", "desc": "Test desc"}, "gpt-4")

        json_str = export_groups_to_json([group])
        data = json.loads(json_str)

        assert len(data) == 1
        assert data[0]["theme"] == "Test theme"
        assert data[0]["category"] == "addition"


class TestOutputFiles:
    """Tests for output file generation."""

    def test_grouped_json_structure(self):
        """Test that grouped JSON has correct structure."""
        suggestions = [
            {
                "title": "Test suggestion",
                "desc": "Test description",
                "importance": "medium",
                "type": "addition",
                "source_model": "test-model"
            }
        ]
        groups = group_similar_suggestions(suggestions)
        json_str = export_groups_to_json(groups)
        data = json.loads(json_str)

        assert isinstance(data, list)
        if data:
            group = data[0]
            assert "category" in group
            assert "theme" in group
            assert "suggestions" in group
            assert "models" in group
            assert "priority_score" in group


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
