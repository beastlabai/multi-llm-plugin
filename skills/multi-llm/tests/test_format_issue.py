#!/usr/bin/env python3
"""Tests for _format_issue and report generation in code_review_orchestrator."""

import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_review_orchestrator import _format_issue


class TestFormatIssue:
    """Tests for _format_issue function."""

    def test_includes_skip_checkbox(self):
        """Format includes - [ ] Skip checkbox line."""
        issue = {
            "title": "Missing null check",
            "desc": "Add null check before accessing property",
            "type": "bug",
            "file": "src/main.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue)
        result = "\n".join(lines)

        assert "- [ ] Skip" in result

    def test_includes_explicit_validation_valid(self):
        """Format includes explicit validation status for valid."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue, validation_status="valid")
        result = "\n".join(lines)

        assert "**Validation:** ✓ Valid" in result or "**Validation:** Valid" in result

    def test_includes_explicit_validation_invalid(self):
        """Format includes explicit validation status for invalid."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue, validation_status="invalid")
        result = "\n".join(lines)

        assert "**Validation:** ✗ Invalid" in result or "**Validation:** Invalid" in result

    def test_includes_explicit_validation_needs_review(self):
        """Format includes explicit validation status for needs-human-decision."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue, validation_status="needs-human-decision")
        result = "\n".join(lines)

        assert "**Validation:** ? Needs Review" in result or "Needs Review" in result

    def test_includes_explicit_validation_failed(self):
        """Format includes explicit validation status for validation_failed."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue, validation_status="validation_failed")
        result = "\n".join(lines)

        assert "**Validation:** ? Validation Failed" in result or "Validation Failed" in result

    def test_includes_validation_reason_for_invalid(self):
        """Format includes validation reason for invalid issues."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(
            1, issue,
            validation_status="invalid",
            validation_reason="This is a false positive because the code path is unreachable."
        )
        result = "\n".join(lines)

        assert "**Validation Reason:**" in result
        assert "false positive" in result

    def test_includes_validation_reason_for_needs_review(self):
        """Format includes validation reason for needs-human-decision issues."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(
            1, issue,
            validation_status="needs-human-decision",
            validation_reason="Requires understanding of business logic to evaluate."
        )
        result = "\n".join(lines)

        assert "**Validation Reason:**" in result
        assert "business logic" in result

    def test_no_validation_reason_for_valid(self):
        """Format does not include validation reason for valid issues."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(
            1, issue,
            validation_status="valid",
            validation_reason="This is a valid issue."
        )
        result = "\n".join(lines)

        # Reason should NOT be displayed for valid issues
        assert "**Validation Reason:**" not in result

    def test_includes_unknown_for_no_status(self):
        """Format includes Unknown when no validation status provided."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue)
        result = "\n".join(lines)

        assert "**Validation:**" in result
        assert "Unknown" in result or "?" in result

    def test_title_does_not_have_badge(self):
        """Title line should not have compact badge like [✓]."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue, validation_status="valid")

        # First line is the title
        title_line = lines[0]
        assert "[✓]" not in title_line
        assert "[✗]" not in title_line
        assert "[?]" not in title_line

    def test_includes_file_type_model(self):
        """Format includes File, Type, and Model metadata."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "security",
            "file": "src/auth.py",
            "model": "gpt-4",
        }

        lines = _format_issue(1, issue, validation_status="valid")
        result = "\n".join(lines)

        assert "**File:** `src/auth.py`" in result
        assert "**Type:** security" in result
        assert "**Model:** gpt-4" in result

    def test_includes_line_range_in_file(self):
        """Format includes line range in file reference."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/main.py",
            "line_range": [42, 50],
            "model": "test-model",
        }

        lines = _format_issue(1, issue)
        result = "\n".join(lines)

        assert "src/main.py:42-50" in result

    def test_includes_description(self):
        """Format includes issue description."""
        issue = {
            "title": "Test issue",
            "desc": "This is a detailed description of the issue.",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue)
        result = "\n".join(lines)

        assert "This is a detailed description of the issue." in result

    def test_includes_separator(self):
        """Format includes --- separator at the end."""
        issue = {
            "title": "Test issue",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(1, issue)

        assert "---" in lines

    def test_issue_index_in_title(self):
        """Format includes issue index in title."""
        issue = {
            "title": "Missing validation",
            "desc": "Test description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test-model",
        }

        lines = _format_issue(42, issue)
        title_line = lines[0]

        assert "42." in title_line
        assert "### 42. Missing validation" in title_line


class TestFormatIssueStructure:
    """Tests for the structure of formatted output."""

    def test_correct_line_order(self):
        """Verify correct order: title, skip checkbox, validation line, blank, desc, blank, separator."""
        issue = {
            "title": "Test",
            "desc": "Description here",
            "type": "bug",
            "file": "src/test.py",
            "model": "test",
        }

        lines = _format_issue(1, issue, validation_status="valid")

        # Expected structure:
        # 0: ### 1. Test
        # 1: - [ ] Skip
        # 2: **Validation:** ... | **File:** ... | **Type:** ... | **Model:** ...
        # 3: (empty)
        # 4: Description here
        # 5: (empty)
        # 6: ---
        # 7: (empty)

        assert lines[0].startswith("### 1.")
        assert "- [ ] Skip" in lines[1]
        assert "**Validation:**" in lines[2]
        assert lines[3] == ""
        assert "Description here" in lines[4]
        assert "---" in lines


class TestFormatIssueEdgeCases:
    """Tests for edge cases in _format_issue."""

    def test_missing_line_range(self):
        """Works without line_range field."""
        issue = {
            "title": "Test",
            "desc": "Description",
            "type": "bug",
            "file": "src/test.py",
            "model": "test",
        }

        lines = _format_issue(1, issue)
        result = "\n".join(lines)

        # Should just show file without line numbers
        assert "src/test.py" in result
        assert ":-" not in result  # No line range markers

    def test_missing_model(self):
        """Uses 'unknown' for missing model."""
        issue = {
            "title": "Test",
            "desc": "Description",
            "type": "bug",
            "file": "src/test.py",
        }

        lines = _format_issue(1, issue)
        result = "\n".join(lines)

        assert "**Model:** unknown" in result

    def test_missing_type(self):
        """Uses 'unknown' for missing type."""
        issue = {
            "title": "Test",
            "desc": "Description",
            "file": "src/test.py",
            "model": "test",
        }

        lines = _format_issue(1, issue)
        result = "\n".join(lines)

        assert "**Type:** unknown" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
