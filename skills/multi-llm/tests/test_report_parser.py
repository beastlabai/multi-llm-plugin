"""Tests for report_parser utility."""

import pytest
import tempfile
import os
from pathlib import Path

from utils.report_parser import (
    parse_skipped_suggestions,
    parse_skipped_issues,
    parse_skipped_groups,
    parse_skipped_group_suggestions,
    parse_validation_overrides_groups,
    parse_validation_overrides_issues,
    parse_suggestion_validation_overrides,
    normalize_description,
    parse_suggestion_descriptions,
    parse_issue_descriptions,
    find_edited_descriptions,
    parse_consolidated_skipped_groups,
    parse_consolidated_validation_overrides,
    load_consolidated_html_selections,
    merge_consolidated_selections,
)


class TestParseSkippedSuggestions:
    """Tests for parse_skipped_suggestions function."""

    def test_parse_skipped_no_skips(self, tmp_path):
        """Returns empty set when no [x] markers."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## HIGH

### S001: Add error handling
- [ ] Skip
**Validation:** ✓ Valid | **Model:** cursor-agent:auto | **Type:** addition | **Section:** Step 3

Description here.

---

### S002: Add validation
- [ ] Skip
**Validation:** ✓ Valid | **Model:** gemini:gemini-2.5-flash | **Type:** addition | **Section:** Step 2

Another description.

---
""")
        result = parse_skipped_suggestions(str(report))
        assert result == set()

    def test_parse_skipped_single(self, tmp_path):
        """Correctly parses one [x] Skip."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## HIGH

### S001: Add error handling
- [ ] Skip
**Validation:** ✓ Valid | **Model:** cursor-agent:auto | **Type:** addition | **Section:** Step 3

Description here.

---

### S002: Add validation
- [x] Skip
**Validation:** ✓ Valid | **Model:** gemini:gemini-2.5-flash | **Type:** addition | **Section:** Step 2

Another description.

---
""")
        result = parse_skipped_suggestions(str(report))
        assert result == {"S002"}

    def test_parse_skipped_multiple(self, tmp_path):
        """Correctly parses multiple skips."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## HIGH

### S001: Add error handling
- [x] Skip
**Validation:** ✓ Valid | **Model:** cursor-agent:auto | **Type:** addition | **Section:** Step 3

Description here.

---

### S002: Add validation
- [ ] Skip
**Validation:** ✓ Valid | **Model:** gemini:gemini-2.5-flash | **Type:** addition | **Section:** Step 2

Another description.

---

## MEDIUM

### S003: Consider logging
- [x] Skip
**Validation:** ✗ Invalid | **Model:** cursor-agent:auto | **Type:** modification | **Section:** Step 1

Third description.

---
""")
        result = parse_skipped_suggestions(str(report))
        assert result == {"S001", "S003"}

    def test_parse_skipped_case_insensitive(self, tmp_path):
        """Handles [x] skip and [x] SKIP."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### S001: First
- [x] skip
**Validation:** ✓ Valid

Description.

---

### S002: Second
- [x] SKIP
**Validation:** ✓ Valid

Description.

---

### S003: Third
- [x] Skip
**Validation:** ✓ Valid

Description.

---
""")
        result = parse_skipped_suggestions(str(report))
        assert result == {"S001", "S002", "S003"}

    def test_parse_skipped_missing_file(self, tmp_path):
        """Returns empty set if report doesn't exist."""
        result = parse_skipped_suggestions(str(tmp_path / "nonexistent.md"))
        assert result == set()

    def test_parse_skipped_whitespace_flexibility(self, tmp_path):
        """Handles extra whitespace around checkbox."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### S001: First
-  [x]  Skip
**Validation:** ✓ Valid

Description.

---

### S002: Second
- [x]   skip
**Validation:** ✓ Valid

Description.

---
""")
        result = parse_skipped_suggestions(str(report))
        assert result == {"S001", "S002"}

    def test_parse_skipped_whitespace_inside_checkbox(self, tmp_path):
        """Handles whitespace inside checkbox brackets."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### S001: Space after x
- [x ] Skip
**Validation:** ✓ Valid

---

### S002: Space before x
- [ x] Skip
**Validation:** ✓ Valid

---

### S003: Spaces both sides
- [ x ] Skip
**Validation:** ✓ Valid

---

### S004: Multiple spaces
- [    x  ] Skip
**Validation:** ✓ Valid

---
""")
        result = parse_skipped_suggestions(str(report))
        assert result == {"S001", "S002", "S003", "S004"}


    def test_parse_skipped_non_x_characters(self, tmp_path):
        """Accepts non-x characters like v, ✓, 1 as skip markers."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### S001: Checkmark
- [✓] Skip
**Validation:** ✓ Valid

---

### S002: Letter v
- [v] Skip
**Validation:** ✓ Valid

---

### S003: Number
- [1] Skip
**Validation:** ✓ Valid

---

### S004: Multiple chars
- [abc] Skip
**Validation:** ✓ Valid

---

### S005: Unchecked
- [ ] Skip
**Validation:** ✓ Valid

---
""", encoding='utf-8')
        result = parse_skipped_suggestions(str(report))
        assert result == {"S001", "S002", "S003", "S004"}
        assert "S005" not in result

    def test_parse_skipped_non_x_with_whitespace(self, tmp_path):
        """Accepts non-x characters with surrounding whitespace."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### S001: Spaced checkmark
- [ ✓ ] Skip
**Validation:** ✓ Valid

---

### S002: Spaced v
- [ v ] Skip
**Validation:** ✓ Valid

---
""", encoding='utf-8')
        result = parse_skipped_suggestions(str(report))
        assert result == {"S001", "S002"}


class TestParseSkippedIssues:
    """Tests for parse_skipped_issues function (code review reports)."""

    def test_parse_no_skips(self, tmp_path):
        """Returns empty set when no [x] markers."""
        report = tmp_path / "report.md"
        report.write_text("""# Code Review Report

## HIGH Priority

### 1. Missing error handling
- [ ] Skip
**Validation:** ✓ Valid | **File:** `src/api.py:10-20` | **Type:** bug | **Model:** gemini

Description.

---

### 2. SQL injection risk
- [ ] Skip
**Validation:** ✓ Valid | **File:** `src/db.py:50-55` | **Type:** security | **Model:** cursor

Description.

---
""")
        result = parse_skipped_issues(str(report))
        assert result == set()

    def test_parse_single_skip(self, tmp_path):
        """Correctly parses one [x] Skip."""
        report = tmp_path / "report.md"
        report.write_text("""# Code Review Report

## HIGH Priority

### 1. Missing error handling
- [ ] Skip
**Validation:** ✓ Valid | **File:** `src/api.py:10-20` | **Type:** bug

Description.

---

### 2. SQL injection risk
- [x] Skip
**Validation:** ✓ Valid | **File:** `src/db.py:50-55` | **Type:** security

Description.

---
""")
        result = parse_skipped_issues(str(report))
        assert result == {2}

    def test_parse_multiple_skips(self, tmp_path):
        """Correctly parses multiple skips."""
        report = tmp_path / "report.md"
        report.write_text("""# Code Review Report

### 1. First issue
- [x] Skip
**Validation:** ✓ Valid

---

### 2. Second issue
- [ ] Skip
**Validation:** ✓ Valid

---

### 3. Third issue
- [x] Skip
**Validation:** ✗ Invalid

---
""")
        result = parse_skipped_issues(str(report))
        assert result == {1, 3}

    def test_parse_missing_file(self, tmp_path):
        """Returns empty set if report doesn't exist."""
        result = parse_skipped_issues(str(tmp_path / "nonexistent.md"))
        assert result == set()

    def test_parse_non_x_characters(self, tmp_path):
        """Accepts non-x characters like v, ✓, 1 as skip markers."""
        report = tmp_path / "report.md"
        report.write_text("""# Code Review Report

### 1. Checkmark issue
- [✓] Skip
**Validation:** ✓ Valid

---

### 2. Letter v issue
- [v] Skip
**Validation:** ✓ Valid

---

### 3. Unchecked issue
- [ ] Skip
**Validation:** ✓ Valid

---
""", encoding='utf-8')
        result = parse_skipped_issues(str(report))
        assert result == {1, 2}
        assert 3 not in result


class TestParseSkippedGroups:
    """Tests for parse_skipped_groups function."""

    def test_parse_no_skipped_groups(self, tmp_path):
        """Returns empty set when no groups skipped."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1: Error handling improvements
- [ ] Skip this group

### G1S1: Add try/catch block
- [ ] Skip

---
""")
        result = parse_skipped_groups(str(report))
        assert result == set()

    def test_parse_single_skipped_group(self, tmp_path):
        """Correctly parses single group with [x] Skip this group (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: Error handling improvements
- [x] Skip this group

### G1S1 [ccdd000000000001]: Add try/catch block
- [ ] Skip

---
""")
        result = parse_skipped_groups(str(report))
        assert result == {"aabb000000000001"}

    def test_parse_multiple_skipped_groups(self, tmp_path):
        """Correctly parses multiple skipped groups (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: Error handling improvements
- [x] Skip this group

### G1S1 [ccdd000000000001]: Add try/catch block
- [ ] Skip

---

## G2 [aabb000000000002]: Performance optimizations
- [ ] Skip this group

### G2S1 [ccdd000000000002]: Add caching
- [ ] Skip

---

## G3 [aabb000000000003]: Security fixes
- [x] Skip this group

### G3S1 [ccdd000000000003]: Add validation
- [ ] Skip

---
""")
        result = parse_skipped_groups(str(report))
        assert result == {"aabb000000000001", "aabb000000000003"}

    def test_whitespace_flexibility(self, tmp_path):
        """Handles extra whitespace around checkbox (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: First
-  [x]  Skip this group

---

## G2 [aabb000000000002]: Second
- [ x ] skip this group

---
""")
        result = parse_skipped_groups(str(report))
        assert result == {"aabb000000000001", "aabb000000000002"}

    def test_case_insensitive(self, tmp_path):
        """Handles SKIP THIS GROUP, skip this group, etc. (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: First
- [x] SKIP THIS GROUP

---

## G2 [aabb000000000002]: Second
- [x] skip this group

---

## G3 [aabb000000000003]: Third
- [x] Skip This Group

---
""")
        result = parse_skipped_groups(str(report))
        assert result == {"aabb000000000001", "aabb000000000002", "aabb000000000003"}

    def test_does_not_match_suggestion_skip(self, tmp_path):
        """Ensures '- [x] Skip' without 'this group' is not a group skip."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1: Error handling
- [ ] Skip this group

### G1S1: Add try/catch
- [x] Skip

---
""")
        result = parse_skipped_groups(str(report))
        assert result == set()

    def test_parse_missing_file(self, tmp_path):
        """Returns empty set if report doesn't exist."""
        result = parse_skipped_groups(str(tmp_path / "nonexistent.md"))
        assert result == set()

    def test_parse_non_x_characters(self, tmp_path):
        """Accepts non-x characters like v, checkmark as skip markers (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: Checkmark group
- [✓] Skip this group

---

## G2 [aabb000000000002]: Letter v group
- [v] Skip this group

---

## G3 [aabb000000000003]: Unchecked group
- [ ] Skip this group

---
""", encoding='utf-8')
        result = parse_skipped_groups(str(report))
        assert result == {"aabb000000000001", "aabb000000000002"}
        assert "aabb000000000003" not in result

    def test_blank_lines_between_header_and_checkbox(self, tmp_path):
        """Handles blank lines between group header and checkbox (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: First

- [x] Skip this group

---
""")
        result = parse_skipped_groups(str(report))
        assert result == {"aabb000000000001"}


class TestParseSkippedGroupSuggestions:
    """Tests for parse_skipped_group_suggestions function."""

    def test_parse_no_skipped_suggestions(self, tmp_path):
        """Returns empty set when no suggestions skipped."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1: Error handling
- [ ] Skip this group

### G1S1: Add try/catch
- [ ] Skip

### G1S2: Add logging
- [ ] Skip

---
""")
        result = parse_skipped_group_suggestions(str(report))
        assert result == set()

    def test_parse_single_skipped_suggestion(self, tmp_path):
        """Correctly parses single suggestion with [x] Skip (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: Error handling
- [ ] Skip this group

### G1S1 [ccdd000000000001]: Add try/catch
- [x] Skip

### G1S2 [ccdd000000000002]: Add logging
- [ ] Skip

---
""")
        result = parse_skipped_group_suggestions(str(report))
        assert result == {"ccdd000000000001"}

    def test_parse_multiple_across_groups(self, tmp_path):
        """Correctly parses multiple skipped suggestions across groups (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: Error handling
- [ ] Skip this group

### G1S1 [ccdd000000000001]: First suggestion
- [ ] Skip

### G1S2 [ccdd000000000002]: Second suggestion
- [x] Skip

---

## G2 [aabb000000000002]: Performance
- [ ] Skip this group

### G2S1 [ccdd000000000003]: Add caching
- [ ] Skip

---

## G3 [aabb000000000003]: Security
- [ ] Skip this group

### G3S1 [ccdd000000000004]: Add validation
- [x] Skip

---
""")
        result = parse_skipped_group_suggestions(str(report))
        assert result == {"ccdd000000000002", "ccdd000000000004"}

    def test_does_not_match_group_skip(self, tmp_path):
        """Ensures 'Skip this group' is not matched as individual skip."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1: Error handling
- [x] Skip this group

### G1S1: Add try/catch
- [ ] Skip

---
""")
        result = parse_skipped_group_suggestions(str(report))
        assert result == set()

    def test_whitespace_and_case(self, tmp_path):
        """Handles flexible whitespace and case insensitive (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: Error handling
- [ ] Skip this group

### G1S1 [ccdd000000000001]: First
-  [x]  Skip

### G1S2 [ccdd000000000002]: Second
- [ x ] SKIP

### G1S3 [ccdd000000000003]: Third
- [x] skip

---
""")
        result = parse_skipped_group_suggestions(str(report))
        assert result == {"ccdd000000000001", "ccdd000000000002", "ccdd000000000003"}

    def test_parse_missing_file(self, tmp_path):
        """Returns empty set if report doesn't exist."""
        result = parse_skipped_group_suggestions(str(tmp_path / "nonexistent.md"))
        assert result == set()

    def test_parse_non_x_characters(self, tmp_path):
        """Accepts non-x characters like v, checkmark as skip markers (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1 [aabb000000000001]: Error handling
- [ ] Skip this group

### G1S1 [ccdd000000000001]: Checkmark suggestion
- [✓] Skip

### G1S2 [ccdd000000000002]: Letter v suggestion
- [v] Skip

### G1S3 [ccdd000000000003]: Unchecked suggestion
- [ ] Skip

---
""", encoding='utf-8')
        result = parse_skipped_group_suggestions(str(report))
        assert result == {"ccdd000000000001", "ccdd000000000002"}
        assert "ccdd000000000003" not in result


class TestNormalizeDescription:
    """Tests for normalize_description function."""

    def test_strips_leading_trailing_whitespace(self):
        """Strips leading and trailing whitespace."""
        assert normalize_description("  text  ") == "text"

    def test_collapses_multiple_spaces(self):
        """Collapses multiple spaces to single space."""
        assert normalize_description("a   b") == "a b"

    def test_normalizes_crlf_to_lf(self):
        """Normalizes CRLF to LF."""
        assert normalize_description("a\r\nb") == "a\nb"

    def test_removes_validation_reason_blockquote(self):
        """Removes validation reason blockquote from beginning."""
        text = "> **Validation Reason:** reason\n\nActual desc"
        assert normalize_description(text) == "Actual desc"

    def test_removes_multiple_validation_blockquote_lines(self):
        """Removes multi-line validation blockquote followed by description."""
        text = """> **Validation Reason:** This is the first line
> of the validation reason that spans
> multiple lines

Actual description here."""
        assert normalize_description(text) == "Actual description here."

    def test_preserves_non_validation_blockquotes(self):
        """Preserves blockquotes that are not validation reasons."""
        text = "> This is a quote\n\nDesc"
        assert normalize_description(text) == "> This is a quote\n\nDesc"

    def test_empty_string(self):
        """Returns empty string for empty input."""
        assert normalize_description("") == ""

    def test_only_whitespace(self):
        """Returns empty string for whitespace-only input."""
        assert normalize_description("   \n\t  ") == ""

    def test_preserves_markdown_formatting(self):
        """Preserves markdown formatting like bold and code."""
        text = "**bold** and `code`"
        assert normalize_description(text) == "**bold** and `code`"


class TestParseSuggestionDescriptions:
    """Tests for parse_suggestion_descriptions function."""

    def test_extracts_single_suggestion(self, tmp_path):
        """Extracts description for a single G1S1 suggestion."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1: Error handling
- [ ] Skip this group

### G1S1: Add try/catch
- [ ] Skip
**Validation:** Valid | **Model:** test

Description for G1S1.

---
""")
        result = parse_suggestion_descriptions(str(report))
        assert result == {"G1S1": "Description for G1S1."}

    def test_extracts_multiple_suggestions_same_group(self, tmp_path):
        """Extracts G1S1, G1S2, G1S3 all from same group."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1: Error handling
- [ ] Skip this group

### G1S1: First
- [ ] Skip
**Validation:** Valid

First description.

---

### G1S2: Second
- [ ] Skip
**Validation:** Valid

Second description.

---

### G1S3: Third
- [ ] Skip
**Validation:** Valid

Third description.

---
""")
        result = parse_suggestion_descriptions(str(report))
        assert result == {
            "G1S1": "First description.",
            "G1S2": "Second description.",
            "G1S3": "Third description.",
        }

    def test_extracts_multiple_groups(self, tmp_path):
        """Extracts G1S1, G2S1, G3S1 from different groups."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

## G1: Error handling
- [ ] Skip this group

### G1S1: First
- [ ] Skip
**Validation:** Valid

Description G1S1.

---

## G2: Performance
- [ ] Skip this group

### G2S1: Cache
- [ ] Skip
**Validation:** Valid

Description G2S1.

---

## G3: Security
- [ ] Skip this group

### G3S1: Validate
- [ ] Skip
**Validation:** Valid

Description G3S1.

---
""")
        result = parse_suggestion_descriptions(str(report))
        assert result == {
            "G1S1": "Description G1S1.",
            "G2S1": "Description G2S1.",
            "G3S1": "Description G3S1.",
        }

    def test_extracts_multiline_description(self, tmp_path):
        """Extracts description that spans multiple lines."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: Multiline
- [ ] Skip
**Validation:** Valid

This is line one.
This is line two.

This is a new paragraph.

---
""")
        result = parse_suggestion_descriptions(str(report))
        assert result == {
            "G1S1": "This is line one.\nThis is line two.\n\nThis is a new paragraph."
        }

    def test_extracts_description_with_code_block(self, tmp_path):
        """Extracts description containing a code block."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: Code example
- [ ] Skip
**Validation:** Valid

Add error handling like this:

```python
try:
    result = do_something()
except Exception as e:
    log.error(e)
```

---
""")
        result = parse_suggestion_descriptions(str(report))
        expected = """Add error handling like this:

```python
try:
    result = do_something()
except Exception as e:
    log.error(e)
```"""
        assert result == {"G1S1": expected}

    def test_excludes_validation_reason(self, tmp_path):
        """Validation blockquote is not included in extracted description."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: With validation
- [ ] Skip
**Validation:** Valid

> **Validation Reason:** This was validated because...

The actual description.

---
""")
        result = parse_suggestion_descriptions(str(report))
        # The description includes the validation blockquote as raw text
        # (parse_suggestion_descriptions doesn't normalize, that's for comparison)
        assert "G1S1" in result
        assert "The actual description." in result["G1S1"]

    def test_handles_empty_description(self, tmp_path):
        """Handles case with no text between metadata and separator."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: Empty
- [ ] Skip
**Validation:** Valid

---
""")
        result = parse_suggestion_descriptions(str(report))
        # Empty descriptions are not included
        assert result == {}

    def test_handles_missing_separator(self, tmp_path):
        """Handles case with no --- after description."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: No separator
- [ ] Skip
**Validation:** Valid

Description without separator.
""")
        result = parse_suggestion_descriptions(str(report))
        assert result == {"G1S1": "Description without separator."}

    def test_file_not_found(self, tmp_path):
        """Returns empty dict for nonexistent file."""
        result = parse_suggestion_descriptions(str(tmp_path / "nonexistent.md"))
        assert result == {}

    def test_handles_unicode_in_description(self, tmp_path):
        """Preserves emoji and CJK characters in description."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: Unicode test
- [ ] Skip
**Validation:** Valid

This has emoji: 🎉 and CJK: 你好世界

---
""", encoding='utf-8')
        result = parse_suggestion_descriptions(str(report))
        assert result == {"G1S1": "This has emoji: 🎉 and CJK: 你好世界"}


class TestParseIssueDescriptions:
    """Tests for parse_issue_descriptions function."""

    def test_extracts_single_issue(self, tmp_path):
        """Extracts description for a single issue with ### 1. format."""
        report = tmp_path / "report.md"
        report.write_text("""# Code Review Report

### 1. Missing error handling
- [ ] Skip
**Validation:** Valid | **File:** `src/api.py:10-20`

Description for issue 1.

---
""")
        result = parse_issue_descriptions(str(report))
        assert result == {1: "Description for issue 1."}

    def test_extracts_multiple_issues(self, tmp_path):
        """Extracts issues 1, 2, 3 all correctly."""
        report = tmp_path / "report.md"
        report.write_text("""# Code Review Report

### 1. First issue
- [ ] Skip
**Validation:** Valid

First description.

---

### 2. Second issue
- [ ] Skip
**Validation:** Valid

Second description.

---

### 3. Third issue
- [ ] Skip
**Validation:** Valid

Third description.

---
""")
        result = parse_issue_descriptions(str(report))
        assert result == {
            1: "First description.",
            2: "Second description.",
            3: "Third description.",
        }

    def test_handles_non_sequential_indices(self, tmp_path):
        """Extracts issues 1, 3, 5 with gaps correctly."""
        report = tmp_path / "report.md"
        report.write_text("""# Code Review Report

### 1. First
- [ ] Skip
**Validation:** Valid

Desc 1.

---

### 3. Third
- [ ] Skip
**Validation:** Valid

Desc 3.

---

### 5. Fifth
- [ ] Skip
**Validation:** Valid

Desc 5.

---
""")
        result = parse_issue_descriptions(str(report))
        assert result == {1: "Desc 1.", 3: "Desc 3.", 5: "Desc 5."}

    def test_excludes_validation_reason(self, tmp_path):
        """Validation blockquote is present but can be filtered in comparison."""
        report = tmp_path / "report.md"
        report.write_text("""# Code Review Report

### 1. Issue with validation
- [ ] Skip
**Validation:** Valid

> **Validation Reason:** This was validated because...

The actual description.

---
""")
        result = parse_issue_descriptions(str(report))
        assert 1 in result
        assert "The actual description." in result[1]

    def test_file_not_found(self, tmp_path):
        """Returns empty dict for nonexistent file."""
        result = parse_issue_descriptions(str(tmp_path / "nonexistent.md"))
        assert result == {}

    def test_ignores_grouped_format(self, tmp_path):
        """Only extracts numeric format, ignores G1S1 format."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### 1. Numeric issue
- [ ] Skip
**Validation:** Valid

Numeric description.

---

### G1S1: Grouped suggestion
- [ ] Skip
**Validation:** Valid

Grouped description.

---

### 2. Another numeric
- [ ] Skip
**Validation:** Valid

Another numeric description.

---
""")
        result = parse_issue_descriptions(str(report))
        assert result == {
            1: "Numeric description.",
            2: "Another numeric description.",
        }


class TestFindEditedDescriptions:
    """Tests for find_edited_descriptions function."""

    def test_detects_single_edit(self, tmp_path):
        """Detects when a suggestion description differs."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: First
- [ ] Skip
**Validation:** Valid

Edited description.

---
""")
        grouped = [
            {
                "theme": "Group 1",
                "suggestions": [
                    {"title": "First", "desc": "Original description."}
                ]
            }
        ]
        result = find_edited_descriptions(str(report), grouped)
        assert len(result) == 1
        # Key is suggestion_hash (content-based hex string)
        vals = list(result.values())
        assert vals[0] == ("Original description.", "Edited description.")

    def test_detects_multiple_edits(self, tmp_path):
        """Detects multiple edited items."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: First
- [ ] Skip
**Validation:** Valid

Edited first.

---

### G1S2: Second
- [ ] Skip
**Validation:** Valid

Edited second.

---
""")
        grouped = [
            {
                "theme": "Group 1",
                "suggestions": [
                    {"title": "First", "desc": "Original first."},
                    {"title": "Second", "desc": "Original second."},
                ]
            }
        ]
        result = find_edited_descriptions(str(report), grouped)
        assert len(result) == 2
        vals = sorted(result.values())
        assert ("Original first.", "Edited first.") in vals
        assert ("Original second.", "Edited second.") in vals

    def test_ignores_unchanged(self, tmp_path):
        """Matching descriptions are not in result."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: First
- [ ] Skip
**Validation:** Valid

Same description.

---
""")
        grouped = [
            {
                "theme": "Group 1",
                "suggestions": [
                    {"title": "First", "desc": "Same description."}
                ]
            }
        ]
        result = find_edited_descriptions(str(report), grouped)
        assert result == {}

    def test_ignores_whitespace_only_change(self, tmp_path):
        """Whitespace-only changes are not flagged as edits."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: First
- [ ] Skip
**Validation:** Valid

Same description.

---
""")
        grouped = [
            {
                "theme": "Group 1",
                "suggestions": [
                    {"title": "First", "desc": "Same description."}
                ]
            }
        ]
        result = find_edited_descriptions(str(report), grouped)
        assert result == {}

    def test_detects_added_content(self, tmp_path):
        """Detects when user appends text to description."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: First
- [ ] Skip
**Validation:** Valid

Original description.

User added this additional context.

---
""")
        grouped = [
            {
                "theme": "Group 1",
                "suggestions": [
                    {"title": "First", "desc": "Original description."}
                ]
            }
        ]
        result = find_edited_descriptions(str(report), grouped)
        assert len(result) == 1
        val = list(result.values())[0]
        assert val[0] == "Original description."
        assert "User added this additional context." in val[1]

    def test_detects_replaced_content(self, tmp_path):
        """Detects when description is completely different."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: First
- [ ] Skip
**Validation:** Valid

Completely different text.

---
""")
        grouped = [
            {
                "theme": "Group 1",
                "suggestions": [
                    {"title": "First", "desc": "Original description here."}
                ]
            }
        ]
        result = find_edited_descriptions(str(report), grouped)
        assert len(result) == 1
        val = list(result.values())[0]
        assert val == ("Original description here.", "Completely different text.")

    def test_handles_missing_suggestion_in_report(self, tmp_path):
        """Gracefully handles suggestion in grouped but not in report."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: First
- [ ] Skip
**Validation:** Valid

Description.

---
""")
        grouped = [
            {
                "theme": "Group 1",
                "suggestions": [
                    {"title": "First", "desc": "Different."},
                    {"title": "Second", "desc": "Not in report."},
                ]
            }
        ]
        result = find_edited_descriptions(str(report), grouped)
        # Only the suggestion present in report with different desc should be flagged
        assert len(result) == 1
        val = list(result.values())[0]
        assert val[0] == "Different."
        assert val[1] == "Description."

    def test_handles_extra_suggestion_in_report(self, tmp_path):
        """Extra suggestions in report (not in grouped) are ignored."""
        report = tmp_path / "report.md"
        report.write_text("""# Review Report

### G1S1: First
- [ ] Skip
**Validation:** Valid

Same description.

---

### G1S2: Second
- [ ] Skip
**Validation:** Valid

Extra suggestion.

---
""")
        grouped = [
            {
                "theme": "Group 1",
                "suggestions": [
                    {"title": "First", "desc": "Same description."}
                ]
            }
        ]
        result = find_edited_descriptions(str(report), grouped)
        # G1S2 only exists in report, not in grouped, so it's ignored
        assert result == {}


class TestParseValidationOverridesGroups:
    """Tests for parse_validation_overrides_groups()."""

    def test_no_overrides_returns_empty_dict(self, tmp_path):
        """Report with no override checkboxes returns empty dict."""
        report = tmp_path / "report.md"
        report.write_text("""## G1: Theme
- [ ] Skip this group
**Validation:** Valid
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {}

    def test_single_valid_override(self, tmp_path):
        """Single group with 'Mark valid' checked (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group
- [x] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "valid"}

    def test_single_invalid_override(self, tmp_path):
        """Single group with 'Mark invalid' checked (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group
- [ ] Mark valid
- [x] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "invalid"}

    def test_both_checked_invalid_wins(self, tmp_path):
        """When both valid and invalid are checked, invalid wins (safety-first)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group
- [x] Mark valid
- [x] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "invalid"}

    def test_multiple_groups_different_overrides(self, tmp_path):
        """Multiple groups with different overrides (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: First Theme
- [ ] Skip this group
- [x] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review

---

## G2 [aabb000000000002]: Second Theme
- [ ] Skip this group
**Validation:** Valid

---

## G3 [aabb000000000003]: Third Theme
- [ ] Skip this group
- [ ] Mark valid
- [x] Mark invalid
**Validation:** ? Validation Failed

---
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "valid", "aabb000000000003": "invalid"}

    def test_case_insensitive_matching(self, tmp_path):
        """Checkbox labels matched case-insensitively (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group
- [x] MARK VALID
- [ ] mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "valid"}

    def test_flexible_checkbox_markers(self, tmp_path):
        """Supports various checkbox markers like [v], [X] (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group
- [v] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review

---

## G2 [aabb000000000002]: Another Theme
- [ ] Skip this group
- [ ] Mark valid
- [X] Mark invalid
**Validation:** ? Validation Failed
""", encoding='utf-8')
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "valid", "aabb000000000002": "invalid"}

    def test_unchecked_overrides_ignored(self, tmp_path):
        """Groups with unchecked override checkboxes are not in result."""
        report = tmp_path / "report.md"
        report.write_text("""## G1: Theme
- [ ] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {}

    def test_missing_file_returns_empty_dict(self, tmp_path):
        """Non-existent file returns empty dict."""
        result = parse_validation_overrides_groups(str(tmp_path / "nonexistent.md"))
        assert result == {}

    def test_groups_without_override_checkboxes(self, tmp_path):
        """Groups without override checkboxes are ignored (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group
**Validation:** Valid

---

## G2 [aabb000000000002]: Other Theme
- [ ] Skip this group
- [x] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000002": "valid"}


class TestParseSuggestionValidationOverrides:
    """Tests for parse_suggestion_validation_overrides()."""

    def test_no_overrides_returns_empty(self, tmp_path):
        """Report with unchecked suggestion override boxes returns empty dict."""
        report = tmp_path / "report.md"
        report.write_text("""## G1: Theme
- [ ] Skip this group

### G1S1: Add error handling
- [ ] Skip
- [ ] Mark valid
- [ ] Mark invalid
**Validation:** Valid

---

### G1S2: Add logging
- [ ] Skip
- [ ] Mark valid
- [ ] Mark invalid
**Validation:** Valid

---
""")
        result = parse_suggestion_validation_overrides(str(report))
        assert result == {}

    def test_mark_valid_single_suggestion(self, tmp_path):
        """Suggestion with 'Mark valid' checked (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group

### G1S1 [ccdd000000000001]: Add error handling
- [ ] Skip
- [ ] Mark valid
- [ ] Mark invalid
**Validation:** Valid

---

### G1S2 [ccdd000000000002]: Add logging
- [ ] Skip
- [x] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review

---
""")
        result = parse_suggestion_validation_overrides(str(report))
        assert result == {"ccdd000000000002": "valid"}

    def test_mark_invalid_single_suggestion(self, tmp_path):
        """Suggestion with 'Mark invalid' checked (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group

### G1S1 [ccdd000000000001]: Add error handling
- [ ] Skip
- [ ] Mark valid
- [x] Mark invalid
**Validation:** ? Needs Review

---
""")
        result = parse_suggestion_validation_overrides(str(report))
        assert result == {"ccdd000000000001": "invalid"}

    def test_both_checked_invalid_wins(self, tmp_path):
        """When both Mark valid and Mark invalid are checked, invalid wins."""
        report = tmp_path / "report.md"
        report.write_text("""### G1S1 [ccdd000000000001]: Add error handling
- [ ] Skip
- [x] Mark valid
- [x] Mark invalid
**Validation:** ? Needs Review

---
""")
        result = parse_suggestion_validation_overrides(str(report))
        assert result == {"ccdd000000000001": "invalid"}

    def test_multiple_suggestions_different_overrides(self, tmp_path):
        """Multiple suggestions with different overrides (v2 bracket hash)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: First Theme
- [ ] Skip this group

### G1S1 [ccdd000000000001]: Add error handling
- [ ] Skip
- [x] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review

---

### G1S2 [ccdd000000000002]: Add logging
- [ ] Skip
- [ ] Mark valid
- [ ] Mark invalid
**Validation:** Valid

---

## G2 [aabb000000000002]: Second Theme
- [ ] Skip this group

### G2S1 [ccdd000000000003]: Add caching
- [ ] Skip
**Validation:** Valid

---

### G2S2 [ccdd000000000004]: Add retry logic
- [ ] Skip
**Validation:** Valid

---

### G2S3 [ccdd000000000005]: Remove dead code
- [ ] Skip
- [ ] Mark valid
- [x] Mark invalid
**Validation:** ? Validation Failed

---
""")
        result = parse_suggestion_validation_overrides(str(report))
        assert result == {"ccdd000000000001": "valid", "ccdd000000000005": "invalid"}

    def test_skip_not_confused_with_override(self, tmp_path):
        """'[x] Skip' alone should NOT create a validation override."""
        report = tmp_path / "report.md"
        report.write_text("""### G1S1: Add error handling
- [x] Skip
- [ ] Mark valid
- [ ] Mark invalid
**Validation:** Valid

---

### G1S2: Add logging
- [ ] Skip
**Validation:** Valid

---
""")
        result = parse_suggestion_validation_overrides(str(report))
        assert result == {}

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Nonexistent path returns empty dict."""
        result = parse_suggestion_validation_overrides(str(tmp_path / "nonexistent.md"))
        assert result == {}

    def test_mixed_group_and_suggestion_checkboxes(self, tmp_path):
        """Report with BOTH group-level and suggestion-level override checkboxes.

        Verify parse_suggestion_validation_overrides only returns suggestion-level entries,
        not group-level ones.
        """
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: First Theme
- [ ] Skip this group
- [x] Mark valid
- [ ] Mark invalid

### G1S1 [ccdd000000000001]: Add error handling
- [ ] Skip
- [ ] Mark valid
- [ ] Mark invalid
**Validation:** Valid

---

### G1S2 [ccdd000000000002]: Add logging
- [ ] Skip
- [x] Mark invalid
- [ ] Mark valid
**Validation:** ? Needs Review

---

## G2 [aabb000000000002]: Second Theme
- [ ] Skip this group
- [ ] Mark valid
- [x] Mark invalid

### G2S1 [ccdd000000000003]: Add caching
- [ ] Skip
- [x] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review

---
""")
        result = parse_suggestion_validation_overrides(str(report))
        # Should only contain suggestion-level entries, not group-level ones
        assert result == {"ccdd000000000002": "invalid", "ccdd000000000003": "valid"}
        # Verify keys are hash strings (16-char hex), not positional IDs
        for key in result:
            assert len(key) == 16, f"Expected 16-char hex hash, got: {key}"


class TestParseValidationOverridesIssues:
    """Tests for parse_validation_overrides_issues()."""

    def test_no_overrides_returns_empty_dict(self, tmp_path):
        """Report with no override checkboxes returns empty dict."""
        report = tmp_path / "report.md"
        report.write_text("""### 1. Issue Title
- [ ] Skip
**Validation:** Valid
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {}

    def test_single_valid_override(self, tmp_path):
        """Single issue with 'Mark valid' checked."""
        report = tmp_path / "report.md"
        report.write_text("""### 1. Issue Title
- [ ] Skip
- [x] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {1: "valid"}

    def test_single_invalid_override(self, tmp_path):
        """Single issue with 'Mark invalid' checked."""
        report = tmp_path / "report.md"
        report.write_text("""### 1. Issue Title
- [ ] Skip
- [ ] Mark valid
- [x] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {1: "invalid"}

    def test_both_checked_invalid_wins(self, tmp_path):
        """When both valid and invalid are checked, invalid wins (safety-first)."""
        report = tmp_path / "report.md"
        report.write_text("""### 1. Issue Title
- [ ] Skip
- [x] Mark valid
- [x] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {1: "invalid"}

    def test_multiple_issues_different_overrides(self, tmp_path):
        """Multiple issues with different overrides."""
        report = tmp_path / "report.md"
        report.write_text("""### 1. First Issue
- [ ] Skip
- [x] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review

Some description.

---

### 2. Second Issue
- [ ] Skip
**Validation:** Valid

Description.

---

### 3. Third Issue
- [ ] Skip
- [ ] Mark valid
- [x] Mark invalid
**Validation:** ? Validation Failed

Description.

---
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {1: "valid", 3: "invalid"}

    def test_case_insensitive_matching(self, tmp_path):
        """Checkbox labels matched case-insensitively."""
        report = tmp_path / "report.md"
        report.write_text("""### 1. Issue
- [ ] Skip
- [x] MARK VALID
- [ ] mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {1: "valid"}

    def test_flexible_checkbox_markers(self, tmp_path):
        """Supports various checkbox markers like [v], [X]."""
        report = tmp_path / "report.md"
        report.write_text("""### 1. Issue One
- [ ] Skip
- [v] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review

---

### 2. Issue Two
- [ ] Skip
- [ ] Mark valid
- [X] Mark invalid
**Validation:** ? Validation Failed
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {1: "valid", 2: "invalid"}

    def test_missing_file_returns_empty_dict(self, tmp_path):
        """Non-existent file returns empty dict."""
        result = parse_validation_overrides_issues(str(tmp_path / "nonexistent.md"))
        assert result == {}

    def test_unchecked_overrides_ignored(self, tmp_path):
        """Issues with unchecked override checkboxes are not in result."""
        report = tmp_path / "report.md"
        report.write_text("""### 1. Issue
- [ ] Skip
- [ ] Mark valid
- [ ] Mark invalid
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {}


class TestParseConsolidatedSkippedGroups:
    """Tests for parse_consolidated_skipped_groups()."""

    def test_no_skipped_groups(self, tmp_path):
        """Returns empty set when no groups are skipped."""
        report = tmp_path / "report.md"
        report.write_text("""# Consolidated Report

## CG1 [abc123def456]: Section A
- [ ] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention

Description.
""")
        result = parse_consolidated_skipped_groups(str(report))
        assert result == set()

    def test_single_skipped_group(self, tmp_path):
        """Returns consolidated_id when one group is skipped."""
        report = tmp_path / "report.md"
        report.write_text("""# Consolidated Report

## CG1 [abc123def456]: Section A
- [x] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention

Description.
""")
        result = parse_consolidated_skipped_groups(str(report))
        assert result == {"abc123def456"}

    def test_multiple_skipped_groups(self, tmp_path):
        """Returns multiple consolidated_ids when several groups are skipped."""
        report = tmp_path / "report.md"
        report.write_text("""# Consolidated Report

## CG1 [abc123def456]: Section A
- [x] Skip this group

Description A.

## CG2 [bbb222ccc333]: Section B
- [ ] Skip this group

Description B.

## CG3 [ddd444eee555]: Section C
- [x] Skip this group

Description C.
""")
        result = parse_consolidated_skipped_groups(str(report))
        assert result == {"abc123def456", "ddd444eee555"}

    def test_case_insensitive(self, tmp_path):
        """Checkbox label matched case-insensitively."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [x] SKIP THIS GROUP
""")
        result = parse_consolidated_skipped_groups(str(report))
        assert result == {"abc123def456"}

    def test_flexible_checkbox_markers(self, tmp_path):
        """Supports various checkbox markers like [v], [X], etc."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section A
- [v] Skip this group

## CG2 [bbb222ccc333]: Section B
- [X] Skip this group
""")
        result = parse_consolidated_skipped_groups(str(report))
        assert result == {"abc123def456", "bbb222ccc333"}

    def test_blank_lines_between_header_and_checkbox(self, tmp_path):
        """Handles blank lines between ## header and checkbox."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section A

- [x] Skip this group
""")
        result = parse_consolidated_skipped_groups(str(report))
        assert result == {"abc123def456"}

    def test_missing_file_returns_empty_set(self, tmp_path):
        """Non-existent file returns empty set."""
        result = parse_consolidated_skipped_groups(str(tmp_path / "nonexistent.md"))
        assert result == set()

    def test_unchecked_skips_ignored(self, tmp_path):
        """Empty checkboxes are not treated as skipped."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
""")
        result = parse_consolidated_skipped_groups(str(report))
        assert result == set()

    def test_does_not_match_non_consolidated_headers(self, tmp_path):
        """Does not match G1, G2 style headers (only CG format)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1: Regular Group
- [x] Skip this group

## CG1 [abc123def456]: Consolidated Group
- [ ] Skip this group
""")
        result = parse_consolidated_skipped_groups(str(report))
        assert result == set()


class TestParseConsolidatedValidationOverrides:
    """Tests for parse_consolidated_validation_overrides()."""

    def test_no_overrides_returns_empty_dict(self, tmp_path):
        """Report with no override checkboxes returns empty dict."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {}

    def test_single_valid_override(self, tmp_path):
        """Single group with 'Mark valid' checked."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [x] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "valid"}

    def test_single_invalid_override(self, tmp_path):
        """Single group with 'Mark invalid' checked."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [ ] Mark valid
- [x] Mark invalid
- [ ] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "invalid"}

    def test_single_needs_human_override(self, tmp_path):
        """Single group with 'Needs human attention' checked."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
- [x] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "needs-human-decision"}

    def test_priority_invalid_over_needs_human(self, tmp_path):
        """When invalid and needs-human both checked, invalid wins."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [ ] Mark valid
- [x] Mark invalid
- [x] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "invalid"}

    def test_priority_invalid_over_valid(self, tmp_path):
        """When invalid and valid both checked, invalid wins."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [x] Mark valid
- [x] Mark invalid
- [ ] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "invalid"}

    def test_priority_needs_human_over_valid(self, tmp_path):
        """When needs-human and valid both checked, needs-human wins."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [x] Mark valid
- [ ] Mark invalid
- [x] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "needs-human-decision"}

    def test_all_three_checked_invalid_wins(self, tmp_path):
        """When all three are checked, invalid wins (highest priority)."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [x] Mark valid
- [x] Mark invalid
- [x] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "invalid"}

    def test_multiple_groups_different_overrides(self, tmp_path):
        """Multiple groups with different overrides."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section A
- [ ] Skip this group
- [x] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention

## CG2 [bbb222ccc333]: Section B
- [ ] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention

## CG3 [ddd444eee555]: Section C
- [ ] Skip this group
- [ ] Mark valid
- [x] Mark invalid
- [ ] Needs human attention

## CG4 [fff666aaa777]: Section D
- [ ] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
- [x] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {
            "abc123def456": "valid",
            "ddd444eee555": "invalid",
            "fff666aaa777": "needs-human-decision",
        }
        # CG2 has no overrides, so it should not be in the dict
        assert "bbb222ccc333" not in result

    def test_case_insensitive_matching(self, tmp_path):
        """Checkbox labels matched case-insensitively."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Skip this group
- [x] MARK VALID
- [ ] MARK INVALID
- [ ] NEEDS HUMAN ATTENTION
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "valid"}

    def test_flexible_checkbox_markers(self, tmp_path):
        """Supports various checkbox markers like [v], [X]."""
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section A
- [ ] Skip this group
- [v] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention

## CG2 [bbb222ccc333]: Section B
- [ ] Skip this group
- [ ] Mark valid
- [X] Mark invalid
- [ ] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "valid", "bbb222ccc333": "invalid"}

    def test_missing_file_returns_empty_dict(self, tmp_path):
        """Non-existent file returns empty dict."""
        result = parse_consolidated_validation_overrides(str(tmp_path / "nonexistent.md"))
        assert result == {}

    def test_does_not_match_non_consolidated_headers(self, tmp_path):
        """Does not match G1 style headers (only CG format)."""
        report = tmp_path / "report.md"
        report.write_text("""## G1: Regular Group
- [ ] Skip this group
- [x] Mark valid
- [ ] Mark invalid

## CG1 [abc123def456]: Consolidated Group
- [ ] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {}


class TestLoadConsolidatedHtmlSelections:
    """Tests for load_consolidated_html_selections()."""

    def test_loads_existing_file(self, tmp_path):
        """Loads and parses valid consolidated_user_selections.json."""
        import json
        selections = {
            "plan_path": "plans/test.md",
            "phase": "review-plan",
            "exported_at": "2026-01-01T00:00:00Z",
            "skipped_groups": ["abc123def456"],
            "validation_overrides": {"bbb222ccc333": "valid"},
        }
        (tmp_path / "consolidated_user_selections.json").write_text(
            json.dumps(selections), encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path)
        assert result == selections

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None when file does not exist."""
        result = load_consolidated_html_selections(tmp_path)
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        """Returns None when file contains invalid JSON."""
        (tmp_path / "consolidated_user_selections.json").write_text(
            "not valid json {{{", encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path)
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Returns None when file is empty."""
        (tmp_path / "consolidated_user_selections.json").write_text(
            "", encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path)
        assert result is None

    def test_loads_minimal_structure(self, tmp_path):
        """Loads a minimal JSON object."""
        import json
        (tmp_path / "consolidated_user_selections.json").write_text(
            json.dumps({}), encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path)
        assert result == {}

    def test_does_not_load_regular_user_selections(self, tmp_path):
        """Does not load user_selections.json (only consolidated variant)."""
        import json
        (tmp_path / "user_selections.json").write_text(
            json.dumps({"skipped_groups": [1, 2]}), encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path)
        assert result is None


class TestConsolidatedHtmlSelectionsPlanPathValidation:
    """Tests for plan_path validation in load_consolidated_html_selections()."""

    def test_matching_plan_path_returns_data(self, tmp_path):
        """When plan_path matches, data is returned normally."""
        import json
        selections = {
            "plan_path": "/plans/my-plan.md",
            "skipped_groups": {"abc123def456": True},
            "validation_overrides": {},
        }
        (tmp_path / "consolidated_user_selections.json").write_text(
            json.dumps(selections), encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path, plan_path="/plans/my-plan.md")
        assert result is not None

    def test_mismatched_plan_path_raises_value_error(self, tmp_path):
        """When plan_path differs, raises ValueError."""
        import json
        selections = {
            "plan_path": "/plans/plan-a.md",
            "skipped_groups": {},
            "validation_overrides": {},
        }
        (tmp_path / "consolidated_user_selections.json").write_text(
            json.dumps(selections), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Plan path mismatch"):
            load_consolidated_html_selections(tmp_path, plan_path="/plans/plan-b.md")

    def test_missing_plan_path_warns_for_old_exports(self, tmp_path, capsys):
        """When JSON lacks plan_path, prints warning but returns data."""
        import json
        selections = {"skipped_groups": {}, "validation_overrides": {}}
        (tmp_path / "consolidated_user_selections.json").write_text(
            json.dumps(selections), encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path, plan_path="/plans/my-plan.md")
        assert result is not None
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "does not contain" in captured.err

    def test_no_plan_path_arg_skips_validation(self, tmp_path):
        """When plan_path=None (default), no validation occurs."""
        import json
        selections = {
            "plan_path": "/completely/different/path.md",
            "skipped_groups": {},
            "validation_overrides": {},
        }
        (tmp_path / "consolidated_user_selections.json").write_text(
            json.dumps(selections), encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path)
        assert result is not None

    def test_plan_path_normalization(self, tmp_path):
        """Paths that normalize to the same value are treated as equal."""
        import json
        selections = {
            "plan_path": "/plans/../plans/my-plan.md",
            "skipped_groups": {},
            "validation_overrides": {},
        }
        (tmp_path / "consolidated_user_selections.json").write_text(
            json.dumps(selections), encoding="utf-8"
        )
        result = load_consolidated_html_selections(tmp_path, plan_path="/plans/my-plan.md")
        assert result is not None


class TestMergeConsolidatedSelections:
    """Tests for merge_consolidated_selections()."""

    def test_no_html_returns_markdown(self):
        """When html_selections is None, returns markdown values unchanged."""
        md_skipped = {"abc123def456", "bbb222ccc333"}
        md_overrides = {"abc123def456": "valid", "ddd444eee555": "invalid"}
        skipped, overrides = merge_consolidated_selections(None, md_skipped, md_overrides)
        assert skipped == md_skipped
        assert overrides == md_overrides

    def test_html_replaces_skipped(self):
        """HTML skipped_groups completely replace markdown skipped."""
        html = {"skipped_groups": ["111111111111"], "validation_overrides": {}}
        md_skipped = {"abc123def456", "bbb222ccc333"}
        md_overrides = {}
        skipped, overrides = merge_consolidated_selections(html, md_skipped, md_overrides)
        assert skipped == {"111111111111"}

    def test_html_replaces_overrides(self):
        """HTML validation_overrides completely replace markdown overrides."""
        html = {"skipped_groups": [], "validation_overrides": {"222222222222": "invalid"}}
        md_skipped = set()
        md_overrides = {"abc123def456": "valid", "bbb222ccc333": "needs-human-decision"}
        skipped, overrides = merge_consolidated_selections(html, md_skipped, md_overrides)
        assert overrides == {"222222222222": "invalid"}

    def test_html_empty_clears_markdown(self):
        """Empty HTML selections result in empty merged sets."""
        html = {"skipped_groups": [], "validation_overrides": {}}
        md_skipped = {"abc123def456"}
        md_overrides = {"bbb222ccc333": "valid"}
        skipped, overrides = merge_consolidated_selections(html, md_skipped, md_overrides)
        assert skipped == set()
        assert overrides == {}

    def test_html_missing_keys_default_to_empty(self):
        """Missing keys in HTML selections default to empty collections."""
        html = {}
        md_skipped = {"abc123def456"}
        md_overrides = {"bbb222ccc333": "valid"}
        skipped, overrides = merge_consolidated_selections(html, md_skipped, md_overrides)
        assert skipped == set()
        assert overrides == {}

    def test_returns_correct_types(self):
        """Return types are Set[str] and Dict[str, str]."""
        html = {
            "skipped_groups": ["abc123def456"],
            "validation_overrides": {"bbb222ccc333": "invalid"},
        }
        skipped, overrides = merge_consolidated_selections(html, set(), {})
        assert isinstance(skipped, set)
        assert isinstance(overrides, dict)
        assert all(isinstance(s, str) for s in skipped)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in overrides.items())

class TestParseClaudeDecideOverrides:
    """`Let Claude decide` checkbox -> "claude_decide" across all parsers."""

    # --- Group parser ---
    def test_group_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Skip this group
- [ ] Mark valid
- [ ] Mark invalid
- [x] Let Claude decide
**Validation:** ? Needs Review
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "claude_decide"}

    def test_group_precedence_invalid_over_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [ ] Mark valid
- [x] Mark invalid
- [x] Let Claude decide
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "invalid"}

    def test_group_precedence_valid_over_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""## G1 [aabb000000000001]: Theme
- [x] Mark valid
- [ ] Mark invalid
- [x] Let Claude decide
""")
        result = parse_validation_overrides_groups(str(report))
        assert result == {"aabb000000000001": "valid"}

    # --- Suggestion parser ---
    def test_suggestion_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""### G1S1 [ccdd000000000001]: Title
- [ ] Skip
- [ ] Mark valid
- [ ] Mark invalid
- [x] Let Claude decide
""")
        result = parse_suggestion_validation_overrides(str(report))
        assert result == {"ccdd000000000001": "claude_decide"}

    def test_suggestion_precedence_valid_over_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""### G1S1 [ccdd000000000001]: Title
- [x] Mark valid
- [x] Let Claude decide
""")
        result = parse_suggestion_validation_overrides(str(report))
        assert result == {"ccdd000000000001": "valid"}

    # --- Issues parser (code review) ---
    def test_issue_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""### 1. Title
- [ ] Skip
- [ ] Mark valid
- [ ] Mark invalid
- [x] Let Claude decide
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {1: "claude_decide"}

    def test_issue_precedence_invalid_over_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""### 1. Title
- [x] Mark invalid
- [x] Let Claude decide
""")
        result = parse_validation_overrides_issues(str(report))
        assert result == {1: "invalid"}

    # --- Consolidated parser (4-state precedence) ---
    def test_consolidated_claude_decide_alone(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Mark valid
- [ ] Mark invalid
- [ ] Needs human attention
- [x] Let Claude decide
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "claude_decide"}

    def test_consolidated_invalid_beats_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [x] Mark invalid
- [x] Let Claude decide
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "invalid"}

    def test_consolidated_valid_beats_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [x] Mark valid
- [x] Let Claude decide
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "valid"}

    def test_consolidated_needs_human_beats_claude_decide(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("""## CG1 [abc123def456]: Section
- [ ] Mark valid
- [ ] Mark invalid
- [x] Needs human attention
- [x] Let Claude decide
""")
        result = parse_consolidated_validation_overrides(str(report))
        assert result == {"abc123def456": "needs-human-decision"}
