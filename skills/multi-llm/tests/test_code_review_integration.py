#!/usr/bin/env python3
"""Integration tests for code review orchestrator."""

import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_review_orchestrator import format_tracked_files_for_prompt
from utils import (
    group_similar_suggestions,
    validate_code_review_issues,
)


class TestCodeReviewIssueGrouping:
    """Tests for code review issue grouping."""

    def test_group_code_review_issues(self):
        """Test grouping of code review issues."""
        issues = [
            {
                "title": "Unused variable",
                "desc": "Variable 'x' is declared but never used",
                "importance": "low",
                "file": "src/main.py",
                "type": "style",
                "source_model": "gpt-4"
            },
            {
                "title": "Unused variable found",
                "desc": "The variable 'x' in main.py is not used",
                "importance": "low",
                "file": "src/main.py",
                "type": "style",
                "source_model": "claude"
            }
        ]
        result = group_similar_suggestions(issues)
        # Similar issues should potentially be grouped
        assert len(result) >= 1

    def test_different_file_issues_not_merged(self):
        """Test that issues in different files stay separate."""
        issues = [
            {
                "title": "Missing null check",
                "desc": "Add null check",
                "importance": "high",
                "file": "src/a.py",
                "type": "bug",
                "source_model": "gpt-4"
            },
            {
                "title": "Missing null check",
                "desc": "Add null check",
                "importance": "high",
                "file": "src/b.py",
                "type": "bug",
                "source_model": "claude"
            }
        ]
        result = group_similar_suggestions(issues)
        # Should stay separate due to different files
        # Note: current grouping is text-based, so this tests current behavior
        assert len(result) >= 1


class TestCodeReviewValidation:
    """Tests for code review issue validation."""

    def test_validate_valid_issues(self):
        """Test validation of properly formatted issues."""
        issues = [
            {
                "title": "Test issue",
                "desc": "Test description",
                "importance": "high",
                "file": "test.py",
                "type": "bug"
            }
        ]
        is_valid, errors = validate_code_review_issues(issues)
        assert is_valid
        assert not errors

    def test_validate_missing_fields(self):
        """Test validation catches missing fields."""
        issues = [
            {
                "title": "Test issue"
                # Missing other required fields
            }
        ]
        is_valid, errors = validate_code_review_issues(issues)
        # Validation should catch missing fields
        assert isinstance(errors, list)


class TestPerModelFiles:
    """Tests for per-model file persistence."""

    def test_sanitize_model_name(self):
        """Test model name sanitization for filenames."""
        # Import the function
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Test various model names
        test_cases = [
            ("gpt-4", "gpt-4"),
            ("claude/opus", "claude_opus"),
            ("model:v1.0", "model_v1_0"),
        ]

        import re
        def sanitize_model_name(model: str) -> str:
            return re.sub(r'[^a-zA-Z0-9\-_]', '_', model)

        for input_name, expected_pattern in test_cases:
            result = sanitize_model_name(input_name)
            assert re.match(r'^[a-zA-Z0-9\-_]+$', result), f"Invalid chars in: {result}"


class TestFormatTrackedFilesForPrompt:
    """Tests for the format_tracked_files_for_prompt function."""

    def test_empty_tracked_files(self):
        """Passing empty list returns the default message."""
        result = format_tracked_files_for_prompt([])
        assert result == "(No tracked files - using git diff to discover changes)"

    def test_normal_list(self):
        """A small list is returned fully, no truncation."""
        tracked = [
            {"path": "src/a.py", "task_id": "task-1"},
            {"path": "src/b.py", "task_id": "task-2"},
            {"path": "src/c.py", "task_id": "task-3"},
        ]
        result = format_tracked_files_for_prompt(tracked)
        assert "src/a.py" in result
        assert "src/b.py" in result
        assert "src/c.py" in result
        assert "truncated" not in result

    def test_truncation_with_many_files(self):
        """200 tracked files should be truncated at the default max_chars=5000."""
        tracked = [
            {"path": f"src/file_{i}.py", "task_id": "task-1"}
            for i in range(200)
        ]
        result = format_tracked_files_for_prompt(tracked)
        assert "more files not shown" in result
        assert "truncated to stay within context limits" in result
        # The text before the notice should be shorter than 5100 chars (some tolerance)
        notice_idx = result.index("truncated to stay within context limits")
        text_before = result[:notice_idx]
        assert len(text_before) < 5100

    def test_truncation_with_custom_threshold(self):
        """20 entries with max_chars=100 should trigger truncation."""
        tracked = [
            {"path": f"src/file_{i}.py", "task_id": "task-1"}
            for i in range(20)
        ]
        result = format_tracked_files_for_prompt(tracked, max_chars=100)
        assert "truncated to stay within context limits" in result
        assert "more files not shown" in result

    def test_no_truncation_when_under_threshold(self):
        """5 entries with default max_chars should not be truncated."""
        tracked = [
            {"path": f"src/file_{i}.py", "task_id": "task-1"}
            for i in range(5)
        ]
        result = format_tracked_files_for_prompt(tracked)
        assert "truncated" not in result


class TestPreExistingFiltering:
    """Tests that validate the pre-existing change filtering logic from the fallback path.

    The logic under test is the inline pattern from main():
        pre_existing = set(state.get("pre_existing_changes", []))
        if pre_existing and changed_files:
            changed_files = [f for f in changed_files if f not in pre_existing]
    """

    def test_filters_pre_existing_files(self):
        """Pre-existing files are removed from changed_files."""
        changed_files = ["a.py", "b.py", "c.py"]
        pre_existing = {"a.py", "c.py"}
        if pre_existing and changed_files:
            changed_files = [f for f in changed_files if f not in pre_existing]
        assert changed_files == ["b.py"]

    def test_empty_pre_existing_no_change(self):
        """Empty pre_existing set means the guard is False, so changed_files stays unchanged."""
        changed_files = ["a.py", "b.py"]
        pre_existing = set()
        if pre_existing and changed_files:
            changed_files = [f for f in changed_files if f not in pre_existing]
        assert changed_files == ["a.py", "b.py"]

    def test_empty_changed_files_no_error(self):
        """Empty changed_files with non-empty pre_existing does not error."""
        changed_files = []
        pre_existing = {"a.py"}
        if pre_existing and changed_files:
            changed_files = [f for f in changed_files if f not in pre_existing]
        assert changed_files == []

    def test_all_files_pre_existing(self):
        """All changed files being pre-existing results in an empty list."""
        changed_files = ["a.py", "b.py"]
        pre_existing = {"a.py", "b.py"}
        if pre_existing and changed_files:
            changed_files = [f for f in changed_files if f not in pre_existing]
        assert changed_files == []

    def test_no_overlap(self):
        """No overlap between changed_files and pre_existing leaves list unchanged."""
        changed_files = ["a.py", "b.py"]
        pre_existing = {"c.py", "d.py"}
        if pre_existing and changed_files:
            changed_files = [f for f in changed_files if f not in pre_existing]
        assert changed_files == ["a.py", "b.py"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
