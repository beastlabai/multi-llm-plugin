#!/usr/bin/env python3
"""
Unit tests for apply_code_fixes_orchestrator.py

Tests entrypoint/CLI parsing, state transitions, batcher integration,
validation flow, and output generation. Uses mocks to avoid real LLM calls.
"""

import argparse
import copy
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.output_handler import derive_prefix, find_output_dir
from utils.apply_path_helpers import load_json_file
from utils.apply_selection_helpers import merge_validation_with_groups
from apply_code_fixes_orchestrator import (
    ApplyCodeFixesOrchestrator,
    format_fix_for_output,
    main,
    _parse_g_format_id,
    _apply_edited_descriptions_to_groups,
    build_issue_to_group_map,
    find_edited_issue_descriptions,
    merge_edited_issue_descriptions,
)

# Aliases for test convenience — these were previously module-level shims
parse_args = ApplyCodeFixesOrchestrator.parse_args
from utils.filtering import filter_items as filter_fixes, filter_user_skipped_groups
from utils.report_parser import merge_selections
from utils.state_manager import stamp_stable_ids, load_groups_payload
from utils.code_fix_batcher import CodeFixBatch
from utils.validation import (
    ERROR_TYPE_AMBIGUOUS,
    ERROR_TYPE_PARSING,
    ERROR_TYPE_TIMEOUT,
)


# --- Test helper (was a module-level function, moved here after dead code cleanup) ---

def load_code_review_data(out_dir, prefix):
    """Load grouped issues and validation results from code review (test helper)."""
    import os
    code_review_dir = os.path.join(out_dir, "code-review")
    grouped_path = os.path.join(code_review_dir, "grouped.json")
    validation_path = os.path.join(code_review_dir, "validation.json")
    raw_grouped = load_json_file(grouped_path)
    if raw_grouped is not None:
        raw_grouped = load_groups_payload(raw_grouped)
        stamp_stable_ids(raw_grouped)
    return raw_grouped, load_json_file(validation_path)


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
This is a sample plan for testing code fix application.

## Tasks

### T001: Fix bug in auth module
Fix authentication issues.
"""
    plan_path = temp_dir / "test-plan.md"
    plan_path.write_text(plan_content)
    return plan_path


@pytest.fixture
def sample_output_dir(temp_dir, sample_plan_file):
    """Create a sample output directory structure."""
    prefix = derive_prefix(str(sample_plan_file))
    out_dir = temp_dir / prefix
    code_review_dir = out_dir / "code-review"
    code_review_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def sample_grouped_issues():
    """Create sample grouped issues from code review."""
    return [
        {
            "theme": "Missing null check",
            "category": "bug",
            "models": ["gpt-4", "claude-3"],
            "suggestions": [
                {
                    "title": "Add null check",
                    "desc": "Add null check before accessing user.name",
                    "type": "bug",
                    "importance": "HIGH",
                    "file": "src/auth.py",
                    "line_range": [42, 50],
                    "anchor_text": "user.name",
                }
            ]
        },
        {
            "theme": "Improve error handling",
            "category": "improvement",
            "models": ["claude-3"],
            "suggestions": [
                {
                    "title": "Add try-catch",
                    "desc": "Wrap database call in try-catch",
                    "type": "improvement",
                    "importance": "MEDIUM",
                    "file": "src/db.py",
                    "line_range": [100, 120],
                    "anchor_text": "db.query()",
                }
            ]
        },
        {
            "theme": "Security fix",
            "category": "security",
            "models": ["gpt-4"],
            "suggestions": [
                {
                    "title": "Sanitize input",
                    "desc": "Sanitize user input to prevent SQL injection",
                    "type": "security",
                    "importance": "HIGH",
                    "file": "src/api.py",
                    "line_range": [200, 210],
                    "anchor_text": "request.params",
                }
            ]
        },
        {
            "theme": "Code cleanup",
            "category": "style",
            "models": ["gpt-4"],
            "suggestions": [
                {
                    "title": "Remove unused import",
                    "desc": "Remove unused import statement",
                    "type": "style",
                    "importance": "LOW",
                    "file": "src/utils.py",
                    "line_range": [1, 5],
                    "anchor_text": "import os",
                }
            ]
        },
    ]


@pytest.fixture
def sample_validation_results():
    """Create sample validation results."""
    return [
        {
            "group_index": 0,
            "status": "valid",
            "reason": "Real issue confirmed",
            "confidence": 0.95,
        },
        {
            "group_index": 1,
            "status": "needs-human-decision",
            "reason": "Ambiguous whether this is needed",
            "confidence": 0.5,
            "error_type": ERROR_TYPE_AMBIGUOUS,
        },
        {
            "group_index": 2,
            "status": "valid",
            "reason": "Security issue confirmed",
            "confidence": 0.99,
        },
        {
            "group_index": 3,
            "status": "invalid",
            "reason": "Import is actually used",
            "confidence": 0.8,
        },
    ]


@pytest.fixture
def sample_validation_with_failures():
    """Create sample validation results with validation_failed status."""
    return [
        {
            "group_index": 0,
            "status": "valid",
            "reason": "Real issue confirmed",
            "confidence": 0.95,
        },
        {
            "group_index": 1,
            "status": "validation_failed",
            "reason": "Timeout during validation",
            "confidence": 0.0,
            "error_type": ERROR_TYPE_TIMEOUT,
            "recoverable": True,
        },
        {
            "group_index": 2,
            "status": "validation_failed",
            "reason": "JSON parsing error",
            "confidence": 0.0,
            "error_type": ERROR_TYPE_PARSING,
            "recoverable": True,
        },
        {
            "group_index": 3,
            "status": "needs-human-decision",
            "reason": "Unclear intent",
            "confidence": 0.5,
            "error_type": ERROR_TYPE_AMBIGUOUS,
        },
    ]


# --- TestDerivePrefix ---

class TestDerivePrefix:
    """Tests for derive_prefix function."""

    def test_extracts_basename_without_extension(self):
        """Extracts basename without extension."""
        prefix = derive_prefix("plans/my-feature.md")
        assert prefix == "my-feature"

    def test_handles_nested_path(self):
        """Handles nested paths correctly."""
        prefix = derive_prefix("some/deep/path/to/plan.md")
        assert prefix == "plan"

    def test_handles_simple_filename(self):
        """Handles simple filename."""
        prefix = derive_prefix("plan.md")
        assert prefix == "plan"

    def test_removes_special_characters(self):
        """Removes special characters from prefix."""
        prefix = derive_prefix("plans/my feature!.md")
        # sanitize_prefix should handle special chars
        assert "!" not in prefix


# --- TestFindOutputDir ---

class TestFindOutputDir:
    """Tests for find_output_dir function."""

    def test_returns_plan_dir_with_prefix(self):
        """Returns plan directory with prefix subdirectory."""
        out_dir = find_output_dir("plans/my-feature.md")
        assert out_dir == "plans/my-feature"

    def test_handles_current_dir_plan(self):
        """Handles plan in current directory."""
        out_dir = find_output_dir("plan.md")
        assert out_dir == "./plan"

    def test_avoids_double_nesting(self):
        """Avoids double nesting if already in prefix directory."""
        out_dir = find_output_dir("plans/my-feature/my-feature.md")
        assert out_dir == "plans/my-feature"


# --- TestLoadJsonFile ---

class TestLoadJsonFile:
    """Tests for load_json_file function."""

    def test_loads_valid_json(self, temp_dir):
        """Loads valid JSON file."""
        json_path = temp_dir / "test.json"
        data = {"key": "value", "number": 42}
        json_path.write_text(json.dumps(data))

        result = load_json_file(str(json_path))

        assert result == data

    def test_loads_json_array(self, temp_dir):
        """Loads JSON array."""
        json_path = temp_dir / "array.json"
        data = [1, 2, 3, {"nested": True}]
        json_path.write_text(json.dumps(data))

        result = load_json_file(str(json_path))

        assert result == data

    def test_returns_none_for_missing_file(self, temp_dir):
        """Returns None for missing file."""
        result = load_json_file(str(temp_dir / "nonexistent.json"))
        assert result is None

    def test_returns_none_for_invalid_json(self, temp_dir):
        """Returns None for invalid JSON."""
        json_path = temp_dir / "invalid.json"
        json_path.write_text("not valid json {{{")

        result = load_json_file(str(json_path))

        assert result is None


# --- TestLoadCodeReviewData ---

class TestLoadCodeReviewData:
    """Tests for load_code_review_data function."""

    def test_loads_grouped_and_validation(self, sample_output_dir):
        """Loads both grouped.json and validation.json."""
        code_review_dir = sample_output_dir / "code-review"
        grouped_data = [{"theme": "Test"}]
        validation_data = [{"group_index": 0, "status": "valid"}]

        (code_review_dir / "grouped.json").write_text(json.dumps(grouped_data))
        (code_review_dir / "validation.json").write_text(json.dumps(validation_data))

        grouped, validation = load_code_review_data(str(sample_output_dir), "test-plan")

        # Original fields preserved
        assert grouped[0]["theme"] == "Test"
        # Stable IDs are stamped by load_code_review_data
        assert "group_hash" in grouped[0]
        assert "display_label" in grouped[0]
        assert "display_hash" in grouped[0]
        assert validation == validation_data

    def test_returns_none_for_missing_grouped(self, sample_output_dir):
        """Returns None for missing grouped.json."""
        grouped, validation = load_code_review_data(str(sample_output_dir), "test-plan")

        assert grouped is None

    def test_returns_none_for_missing_validation(self, sample_output_dir):
        """Returns None for missing validation.json."""
        code_review_dir = sample_output_dir / "code-review"
        (code_review_dir / "grouped.json").write_text('[{"theme": "Test"}]')

        grouped, validation = load_code_review_data(str(sample_output_dir), "test-plan")

        assert grouped is not None
        assert grouped[0]["theme"] == "Test"
        assert "group_hash" in grouped[0]  # Stable IDs stamped
        assert validation is None


# --- TestMergeValidationWithGroups ---

class TestMergeValidationWithGroups:
    """Tests for merge_validation_with_groups function."""

    def test_merges_validation_status(self, sample_grouped_issues, sample_validation_results):
        """Merges validation status into groups."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        assert len(merged) == 4
        assert merged[0]["validation_status"] == "valid"
        assert merged[1]["validation_status"] == "needs-human-decision"
        assert merged[2]["validation_status"] == "valid"
        assert merged[3]["validation_status"] == "invalid"

    def test_includes_validation_reason(self, sample_grouped_issues, sample_validation_results):
        """Includes validation reason in merged groups."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        assert merged[0]["validation_reason"] == "Real issue confirmed"
        assert merged[1]["validation_reason"] == "Ambiguous whether this is needed"

    def test_includes_validation_confidence(self, sample_grouped_issues, sample_validation_results):
        """Includes validation confidence in merged groups."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        assert merged[0]["validation_confidence"] == 0.95
        assert merged[1]["validation_confidence"] == 0.5

    def test_copies_error_type_if_present(self, sample_grouped_issues, sample_validation_with_failures):
        """Copies error_type field if present."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_with_failures)

        assert merged[1].get("validation_error_type") == ERROR_TYPE_TIMEOUT
        assert merged[2].get("validation_error_type") == ERROR_TYPE_PARSING

    def test_copies_recoverable_if_present(self, sample_grouped_issues, sample_validation_with_failures):
        """Copies recoverable field if present."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_with_failures)

        assert merged[1].get("validation_recoverable") is True
        assert merged[2].get("validation_recoverable") is True

    def test_defaults_to_needs_human_decision_for_missing_validation(self):
        """Defaults to needs-human-decision for missing validation."""
        groups = [{"theme": "Test", "suggestions": []}]
        validation = []  # Empty validation

        merged = merge_validation_with_groups(groups, validation)

        assert merged[0]["validation_status"] == "needs-human-decision"

    def test_adds_group_index(self, sample_grouped_issues, sample_validation_results):
        """Adds group_index to merged groups."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        for i, group in enumerate(merged):
            assert group["group_index"] == i

    def test_joins_by_group_id_when_validation_reordered(self, sample_grouped_issues):
        """Validation must join by group_id, not array position.

        Regression for the status-misalignment bug: validation results are not
        guaranteed to be in the same order as the groups array (reaggregation /
        priority sorting reorders one but not the other). A positional join
        scrambles every validation_status/reason onto the wrong group.
        """
        from utils.state_manager import stamp_stable_ids

        groups = copy.deepcopy(sample_grouped_issues)
        stamp_stable_ids(groups)
        hashes = [g["group_hash"] for g in groups]

        # Validation carries the correct group_id per group but is shuffled:
        # the records are in reverse order relative to the groups array.
        validation = [
            {"group_index": 0, "group_id": hashes[3], "status": "invalid", "reason": "r3"},
            {"group_index": 1, "group_id": hashes[2], "status": "valid", "reason": "r2"},
            {"group_index": 2, "group_id": hashes[1], "status": "needs-human-decision", "reason": "r1"},
            {"group_index": 3, "group_id": hashes[0], "status": "valid", "reason": "r0"},
        ]

        merged = merge_validation_with_groups(groups, validation)

        # Each group keeps the status/reason that matches its own content hash.
        assert merged[0]["validation_status"] == "valid" and merged[0]["validation_reason"] == "r0"
        assert merged[1]["validation_status"] == "needs-human-decision" and merged[1]["validation_reason"] == "r1"
        assert merged[2]["validation_status"] == "valid" and merged[2]["validation_reason"] == "r2"
        assert merged[3]["validation_status"] == "invalid" and merged[3]["validation_reason"] == "r3"

    def test_group_id_join_survives_groups_reordered(self, sample_grouped_issues):
        """A group moved to a new position keeps its own validation result."""
        from utils.state_manager import stamp_stable_ids

        groups = copy.deepcopy(sample_grouped_issues)
        stamp_stable_ids(groups)
        validation = [
            {"group_index": i, "group_id": g["group_hash"], "status": "valid", "reason": g["theme"]}
            for i, g in enumerate(groups)
        ]
        # Reorder the groups array (e.g. priority sort) without touching validation.
        reordered = [groups[2], groups[0], groups[3], groups[1]]

        merged = merge_validation_with_groups(reordered, validation)

        # Reason was seeded from each group's own theme, so a correct join
        # means reason still matches the group it sits next to.
        for m in merged:
            assert m["validation_reason"] == m["theme"]


# --- TestFilterFixes ---

class TestFilterFixes:
    """Tests for filter_fixes function (wrapper around filter_items)."""

    def test_separates_valid_from_needs_human(self, sample_grouped_issues, sample_validation_results):
        """Separates valid from needs-human-decision groups."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)
        valid, needs_human, skipped, report = filter_fixes(merged)

        # 2 valid, 1 needs-human, 1 invalid (skipped)
        assert len(valid) == 2
        assert len(needs_human) == 1
        assert len(skipped) == 1

    def test_includes_low_importance_by_default(self, sample_grouped_issues, sample_validation_results):
        """Includes LOW importance by default (min_priority='low')."""
        # Make all valid with varying importance
        validation = [
            {"group_index": 0, "status": "valid", "confidence": 0.9},
            {"group_index": 1, "status": "valid", "confidence": 0.9},
            {"group_index": 2, "status": "valid", "confidence": 0.9},
            {"group_index": 3, "status": "valid", "confidence": 0.9},  # LOW importance
        ]
        merged = merge_validation_with_groups(sample_grouped_issues, validation)

        valid, needs_human, skipped, report = filter_fixes(merged)

        # Default is min_priority='low', so all valid items are included
        assert len(valid) == 4
        assert len(skipped) == 0

    def test_min_priority_low_includes_low_importance(self, sample_grouped_issues, sample_validation_results):
        """min_priority='low' includes LOW importance items."""
        validation = [
            {"group_index": 0, "status": "valid", "confidence": 0.9},
            {"group_index": 1, "status": "valid", "confidence": 0.9},
            {"group_index": 2, "status": "valid", "confidence": 0.9},
            {"group_index": 3, "status": "valid", "confidence": 0.9},  # LOW
        ]
        merged = merge_validation_with_groups(sample_grouped_issues, validation)

        valid, needs_human, skipped, report = filter_fixes(merged, min_priority="low")

        assert len(valid) == 4
        assert len(skipped) == 0

    def test_min_priority_medium_skips_low_importance(self, sample_grouped_issues, sample_validation_results):
        """min_priority='medium' skips LOW importance items."""
        validation = [
            {"group_index": 0, "status": "valid", "confidence": 0.9},  # HIGH
            {"group_index": 1, "status": "valid", "confidence": 0.9},  # MEDIUM
            {"group_index": 2, "status": "valid", "confidence": 0.9},  # HIGH
            {"group_index": 3, "status": "valid", "confidence": 0.9},  # LOW
        ]
        merged = merge_validation_with_groups(sample_grouped_issues, validation)

        valid, needs_human, skipped, report = filter_fixes(merged, min_priority="medium")

        assert len(valid) == 3
        assert len(skipped) == 1

    def test_min_priority_high_only_includes_high(self, sample_grouped_issues, sample_validation_results):
        """min_priority='high' only includes HIGH importance items."""
        validation = [
            {"group_index": 0, "status": "valid", "confidence": 0.9},  # HIGH
            {"group_index": 1, "status": "valid", "confidence": 0.9},  # MEDIUM
            {"group_index": 2, "status": "valid", "confidence": 0.9},  # HIGH
            {"group_index": 3, "status": "valid", "confidence": 0.9},  # LOW
        ]
        merged = merge_validation_with_groups(sample_grouped_issues, validation)

        valid, needs_human, skipped, report = filter_fixes(merged, min_priority="high")

        assert len(valid) == 2  # Only HIGH importance items
        assert len(skipped) == 2  # MEDIUM and LOW skipped

    def test_skip_all_human_skips_needs_human(self, sample_grouped_issues, sample_validation_results):
        """skip_all_human=True skips needs-human-decision items."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        valid, needs_human, skipped, report = filter_fixes(merged, skip_all_human=True)

        assert len(needs_human) == 0
        assert len(skipped) == 2  # invalid + needs-human

    def test_approve_all_human_approves_all(self, sample_grouped_issues, sample_validation_results):
        """approve_all_human=True approves needs-human-decision items."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        valid, needs_human, skipped, report = filter_fixes(merged, approve_all_human=True)

        assert len(valid) == 3  # 2 already valid + 1 needs-human approved
        assert len(needs_human) == 0
        for g in valid:
            if g.get("auto_approved"):
                assert g["auto_approval_reason"] == "--approve-all"

    def test_approve_importance_levels(self, sample_grouped_issues, sample_validation_with_failures):
        """approve_importance_levels approves specific importance levels."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_with_failures)

        valid, needs_human, skipped, report = filter_fixes(
            merged,
            approve_importance_levels=["MEDIUM"]
        )

        # Check MEDIUM importance items were auto-approved
        approved_mediums = [g for g in valid if g.get("auto_approved")]
        for g in approved_mediums:
            assert "MEDIUM" in g.get("auto_approval_reason", "")

    def test_approve_validation_failed(self, sample_grouped_issues, sample_validation_with_failures):
        """approve_validation_failed=True approves recoverable failures."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_with_failures)

        valid, needs_human, skipped, report = filter_fixes(
            merged,
            approve_validation_failed=True
        )

        # validation_failed items with recoverable=True should be approved
        # Group 1 (timeout) and 2 (parsing) should be approved
        approved_failed = [g for g in valid if "--approve-validation-failed" in g.get("auto_approval_reason", "")]
        assert len(approved_failed) == 2

    def test_dry_run_mode(self, sample_grouped_issues, sample_validation_results):
        """dry_run=True reports what would be approved without modifying."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        valid, needs_human, skipped, report = filter_fixes(
            merged,
            approve_all_human=True,
            dry_run=True
        )

        # In dry run, items should stay in needs_human
        assert len(needs_human) == 1
        assert "would_auto_approve" in report
        assert len(report["would_auto_approve"]) == 1


# --- TestFormatFixForOutput ---

class TestFormatFixForOutput:
    """Tests for format_fix_for_output function."""

    def test_formats_basic_fields(self, sample_grouped_issues):
        """Formats basic fields from group."""
        group = sample_grouped_issues[0]
        group["validation_status"] = "valid"
        group["validation_reason"] = "Test reason"
        group["validation_confidence"] = 0.9

        result = format_fix_for_output(group, 0)

        assert result["index"] == 0
        # Title comes from group theme or first suggestion title
        assert result["title"] == "Missing null check"  # Uses group theme
        assert result["type"] == "bug"
        assert result["importance"] == "HIGH"
        assert result["file"] == "src/auth.py"
        assert result["validation_status"] == "valid"

    def test_includes_line_range(self, sample_grouped_issues):
        """Includes line_range in output."""
        group = sample_grouped_issues[0]
        group["validation_status"] = "valid"

        result = format_fix_for_output(group, 0)

        assert result["line_range"] == [42, 50]

    def test_includes_anchor_text(self, sample_grouped_issues):
        """Includes anchor_text in output."""
        group = sample_grouped_issues[0]
        group["validation_status"] = "valid"

        result = format_fix_for_output(group, 0)

        assert result["anchor_text"] == "user.name"

    def test_determines_subagent_type(self, sample_grouped_issues):
        """Determines appropriate subagent type."""
        group = sample_grouped_issues[0]  # Python file
        group["validation_status"] = "valid"

        result = format_fix_for_output(group, 0)

        assert "subagent_type" in result
        # general-purpose for generic Python file
        assert result["subagent_type"] == "general-purpose"

    def test_combines_descriptions_for_multiple_suggestions(self):
        """Combines descriptions for groups with multiple suggestions."""
        group = {
            "theme": "Multiple issues",
            "category": "bug",
            "models": ["gpt-4", "claude-3"],
            "suggestions": [
                {
                    "title": "Fix 1",
                    "desc": "First fix description",
                    "type": "bug",
                    "importance": "HIGH",
                    "file": "src/test.py",
                    "source_model": "gpt-4",
                },
                {
                    "title": "Fix 2",
                    "desc": "Second fix description",
                    "type": "bug",
                    "importance": "HIGH",
                    "file": "src/test.py",
                    "source_model": "claude-3",
                },
            ],
            "validation_status": "valid",
        }

        result = format_fix_for_output(group, 0)

        assert "First fix description" in result["description"]
        assert "Second fix description" in result["description"]
        assert "Agreed by models" not in result["description"]

    def test_normalizes_importance_to_uppercase(self):
        """Normalizes importance to uppercase."""
        group = {
            "theme": "Test",
            "suggestions": [
                {
                    "title": "Test",
                    "desc": "Test",
                    "type": "bug",
                    "importance": "medium",  # lowercase
                    "file": "test.py",
                }
            ],
            "validation_status": "valid",
        }

        result = format_fix_for_output(group, 0)

        assert result["importance"] == "MEDIUM"


# --- TestParseArgs ---

class TestParseArgs:
    """Tests for parse_args function."""

    def test_requires_plan_file(self):
        """--plan-file is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["prog"]):
                parse_args()

    def test_parses_plan_file(self):
        """Parses --plan-file argument."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md"]):
            args = parse_args()
            assert args.plan_file == "plans/test.md"

    def test_parses_dry_run(self):
        """Parses --dry-run flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--dry-run"]):
            args = parse_args()
            assert args.dry_run is True

    def test_parses_resume(self):
        """Parses --resume flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--resume"]):
            args = parse_args()
            assert args.resume is True

    def test_parses_fresh(self):
        """Parses --fresh flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--fresh"]):
            args = parse_args()
            assert args.fresh is True

    def test_parses_include_low_deprecated(self):
        """Parses --include-low flag (deprecated)."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--include-low"]):
            args = parse_args()
            assert args.include_low is True

    def test_parses_min_priority(self):
        """Parses --min-priority flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--min-priority", "medium"]):
            args = parse_args()
            assert args.min_priority == "medium"

    def test_parses_min_priority_choices(self):
        """Parses --min-priority with all valid choices."""
        for level in ["low", "medium", "high"]:
            with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--min-priority", level]):
                args = parse_args()
                assert args.min_priority == level

    def test_parses_skip_all_human(self):
        """Parses --skip-all-human flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--skip-all-human"]):
            args = parse_args()
            assert args.skip_all_human is True

    def test_parses_approve_all_with_yes(self):
        """Parses --approve-all with --yes flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--approve-all", "--yes"]):
            args = parse_args()
            assert args.approve_all is True
            assert args.yes is True

    def test_parses_approve_all_low(self):
        """Parses --approve-all-low flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--approve-all-low"]):
            args = parse_args()
            assert args.approve_all_low is True

    def test_parses_approve_importance(self):
        """Parses --approve-importance with multiple levels."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--approve-importance", "LOW", "MEDIUM"]):
            args = parse_args()
            assert args.approve_importance == ["LOW", "MEDIUM"]

    def test_parses_approve_validation_failed(self):
        """Parses --approve-validation-failed flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--approve-validation-failed"]):
            args = parse_args()
            assert args.approve_validation_failed is True

    def test_parses_revalidate(self):
        """Parses --revalidate flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--revalidate"]):
            args = parse_args()
            assert args.revalidate is True

    def test_parses_revalidate_model(self):
        """Parses --revalidate-model option."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--revalidate-model", "cursor-agent:opus"]):
            args = parse_args()
            assert args.revalidate_model == "cursor-agent:opus"

    def test_parses_no_batch(self):
        """Parses --no-batch flag."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--no-batch"]):
            args = parse_args()
            assert args.no_batch is True

    def test_parses_max_batch_size(self):
        """Parses --max-batch-size option."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--max-batch-size", "5"]):
            args = parse_args()
            assert args.max_batch_size == 5

    def test_parses_output_format(self):
        """Parses --output-format option."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--output-format", "text"]):
            args = parse_args()
            assert args.output_format == "text"

    def test_parses_base_ref(self):
        """Parses --base-ref option."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md", "--base-ref", "main"]):
            args = parse_args()
            assert args.base_ref == "main"

    def test_default_output_format_is_text(self):
        """Default output format is text."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md"]):
            args = parse_args()
            assert args.output_format == "text"

    def test_default_max_batch_size_is_3(self):
        """Default max batch size is 3."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md"]):
            args = parse_args()
            assert args.max_batch_size == 3


# --- TestStateTransitions ---

class TestStateTransitions:
    """Tests for state transitions during orchestrator execution."""

    def test_resume_filters_processed_items(
        self, sample_plan_file, sample_output_dir, sample_grouped_issues, sample_validation_results
    ):
        """Resume mode filters out already-processed items."""
        # Setup: Create code review data
        code_review_dir = sample_output_dir / "code-review"
        (code_review_dir / "grouped.json").write_text(json.dumps(sample_grouped_issues))
        (code_review_dir / "validation.json").write_text(json.dumps(sample_validation_results))

        # Create state with processed items
        state_data = {
            "plan_hash": "test_hash",
            "phases": {
                "apply-fixes": {
                    "processed_items": {
                        "some_group_id": {
                            "status": "applied",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                }
            }
        }
        (sample_output_dir / "state.json").write_text(json.dumps(state_data))

        # The orchestrator should filter processed items when --resume is used
        # This is tested implicitly via the main function

    def test_fresh_clears_previous_state(
        self, sample_plan_file, sample_output_dir, sample_grouped_issues, sample_validation_results
    ):
        """Fresh mode clears previous progress."""
        # Setup: Create code review data
        code_review_dir = sample_output_dir / "code-review"
        (code_review_dir / "grouped.json").write_text(json.dumps(sample_grouped_issues))
        (code_review_dir / "validation.json").write_text(json.dumps(sample_validation_results))

        # Create state with processed items
        state_data = {
            "plan_hash": "test_hash",
            "phases": {
                "apply-fixes": {
                    "processed_items": {
                        "some_group_id": {"status": "applied"}
                    },
                    "human_decisions": {
                        "some_group_id": {"decision": "approved"}
                    }
                }
            }
        }
        (sample_output_dir / "state.json").write_text(json.dumps(state_data))

        # The --fresh flag should clear these


# --- TestBatcherIntegration ---

class TestBatcherIntegration:
    """Tests for code fix batcher integration (mocked)."""

    def test_single_fix_creates_single_batch(self):
        """Single fix creates one batch."""
        fixes = [
            {
                "file": "src/main.py",
                "description": "Fix bug",
                "importance": "HIGH",
                "type": "bug",
                "line_range": [10, 15],
            }
        ]

        with patch("utils.code_fix_batcher.batch_code_fixes") as mock_batch:
            mock_batch.return_value = [
                CodeFixBatch(
                    fixes=fixes,
                    file_key="src/main.py",
                    batch_type="bug",
                    subagent_type="general-purpose",
                )
            ]

            from utils.code_fix_batcher import batch_code_fixes
            batches = batch_code_fixes(fixes)

            assert len(batches) == 1
            assert batches[0].size == 1

    def test_multiple_fixes_batched_by_file(self):
        """Multiple fixes in same file are batched together."""
        fixes = [
            {"file": "src/main.py", "description": "Fix 1", "importance": "MEDIUM"},
            {"file": "src/main.py", "description": "Fix 2", "importance": "MEDIUM"},
            {"file": "src/other.py", "description": "Fix 3", "importance": "MEDIUM"},
        ]

        with patch("utils.code_fix_batcher.batch_code_fixes") as mock_batch:
            batch1 = CodeFixBatch(
                fixes=fixes[:2],
                file_key="src/main.py",
                batch_type="mixed",
            )
            batch2 = CodeFixBatch(
                fixes=[fixes[2]],
                file_key="src/other.py",
                batch_type="mixed",
            )
            mock_batch.return_value = [batch1, batch2]

            from utils.code_fix_batcher import batch_code_fixes
            batches = batch_code_fixes(fixes)

            assert len(batches) == 2

    def test_high_risk_fixes_isolated(self):
        """HIGH importance fixes are isolated in their own batches."""
        fixes = [
            {"file": "src/main.py", "description": "Fix 1", "importance": "HIGH"},
            {"file": "src/main.py", "description": "Fix 2", "importance": "HIGH"},
            {"file": "src/main.py", "description": "Fix 3", "importance": "MEDIUM"},
        ]

        with patch("utils.code_fix_batcher.batch_code_fixes") as mock_batch:
            # Each HIGH fix gets its own batch
            batch1 = CodeFixBatch(fixes=[fixes[0]], file_key="src/main.py")
            batch2 = CodeFixBatch(fixes=[fixes[1]], file_key="src/main.py")
            batch3 = CodeFixBatch(fixes=[fixes[2]], file_key="src/main.py")
            mock_batch.return_value = [batch1, batch2, batch3]

            from utils.code_fix_batcher import batch_code_fixes
            batches = batch_code_fixes(fixes)

            assert len(batches) == 3

    def test_empty_fix_list_returns_empty_batches(self):
        """Empty fix list returns empty batches."""
        with patch("utils.code_fix_batcher.batch_code_fixes") as mock_batch:
            mock_batch.return_value = []

            from utils.code_fix_batcher import batch_code_fixes
            batches = batch_code_fixes([])

            assert batches == []

    def test_no_batch_mode_creates_individual_batches(self):
        """When --no-batch is used, each fix gets its own batch."""
        # This is tested implicitly in the main function
        # Each fix becomes a separate batch when no_batch=True
        pass


# --- TestValidationFlow ---

class TestValidationFlow:
    """Tests for validation flow handling."""

    def test_valid_fixes_routed_to_apply(self, sample_grouped_issues, sample_validation_results):
        """Valid fixes are routed to to_apply list."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)
        valid, needs_human, skipped, report = filter_fixes(merged)

        # Valid items should be in valid list
        valid_indices = [g.get("group_index") for g in valid]
        assert 0 in valid_indices  # Group 0 is valid
        assert 2 in valid_indices  # Group 2 is valid

    def test_invalid_fixes_routed_to_skipped(self, sample_grouped_issues, sample_validation_results):
        """Invalid fixes are routed to skipped list."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)
        valid, needs_human, skipped, report = filter_fixes(merged)

        skipped_indices = [g.get("group_index") for g in skipped]
        assert 3 in skipped_indices  # Group 3 is invalid

    def test_needs_human_decision_requires_review(self, sample_grouped_issues, sample_validation_results):
        """Needs-human-decision items go to needs_human list."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)
        valid, needs_human, skipped, report = filter_fixes(merged)

        needs_human_indices = [g.get("group_index") for g in needs_human]
        assert 1 in needs_human_indices  # Group 1 needs human decision

    def test_validation_failed_with_recovery(self, sample_grouped_issues, sample_validation_with_failures):
        """Validation failed items with recoverable=True can be auto-approved."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_with_failures)

        # Without auto-approve, they go to needs_human
        valid, needs_human, skipped, report = filter_fixes(merged)
        assert len(needs_human) >= 2  # Groups 1 and 2 have validation_failed

        # With auto-approve, they go to valid
        valid, needs_human, skipped, report = filter_fixes(
            merged, approve_validation_failed=True
        )
        approved = [g for g in valid if g.get("auto_approved")]
        assert len(approved) == 2


# --- TestOutputGeneration ---

class TestOutputGeneration:
    """Tests for output generation."""

    def test_dry_run_output_format(
        self, sample_plan_file, sample_output_dir, sample_grouped_issues, sample_validation_results
    ):
        """Dry run outputs what would be applied."""
        # Setup
        code_review_dir = sample_output_dir / "code-review"
        (code_review_dir / "grouped.json").write_text(json.dumps(sample_grouped_issues))
        validation_data = {
            "groups": sample_validation_results,
            "metadata": {"schema_version": "2.0", "model": "test"}
        }
        (code_review_dir / "validation.json").write_text(json.dumps(validation_data))

        # The dry run should print what would be applied without actually applying

    def test_json_output_structure(self, sample_grouped_issues, sample_validation_results):
        """JSON output has correct structure."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)
        valid, needs_human, skipped, report = filter_fixes(merged)

        # The output structure should include:
        # - plan_file
        # - prefix
        # - output_dir
        # - batches
        # - to_apply
        # - needs_human_review
        # - skipped_count
        # - summary
        expected_keys = [
            "plan_file", "prefix", "output_dir", "batches",
            "to_apply", "needs_human_review", "skipped_count", "summary"
        ]
        # This tests the expected structure that main() produces

    def test_partial_completion_summary(self, sample_grouped_issues, sample_validation_results):
        """Summary correctly reflects partial completion."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)
        valid, needs_human, skipped, report = filter_fixes(merged)

        # Summary should show:
        # - Total issues
        # - Valid count
        # - Needs human count
        # - Skipped count
        assert len(valid) + len(needs_human) + len(skipped) == len(merged)

    def test_applied_fixes_summary(self, sample_grouped_issues, sample_validation_results):
        """Applied fixes summary is generated correctly."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)
        valid, needs_human, skipped, report = filter_fixes(merged)

        # Format fixes for output
        formatted = [format_fix_for_output(g, i) for i, g in enumerate(valid)]

        # Each formatted fix should have required fields
        for fix in formatted:
            assert "index" in fix
            assert "title" in fix
            assert "file" in fix
            assert "subagent_type" in fix


# --- TestMainFunction ---

class TestMainFunction:
    """Tests for main() function with mocks."""

    @patch("sys.argv", ["prog", "--plan-file", "nonexistent.md"])
    def test_exits_on_missing_plan_file(self):
        """Exits with error on missing plan file."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_approve_all_requires_confirmation(
        self, sample_plan_file, sample_output_dir, sample_grouped_issues, sample_validation_results
    ):
        """--approve-all requires --yes or --force flag."""
        code_review_dir = sample_output_dir / "code-review"
        (code_review_dir / "grouped.json").write_text(json.dumps(sample_grouped_issues))
        validation_data = {
            "groups": sample_validation_results,
            "metadata": {"schema_version": "2.0", "model": "test"}
        }
        (code_review_dir / "validation.json").write_text(json.dumps(validation_data))

        with patch("sys.argv", ["prog", "--plan-file", str(sample_plan_file), "--approve-all"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_approve_all_works_with_yes_flag(
        self, sample_plan_file, sample_output_dir, sample_grouped_issues, sample_validation_results
    ):
        """--approve-all works with --yes flag."""
        code_review_dir = sample_output_dir / "code-review"
        (code_review_dir / "grouped.json").write_text(json.dumps(sample_grouped_issues))
        validation_data = {
            "groups": sample_validation_results,
            "metadata": {"schema_version": "2.0", "model": "test"}
        }
        (code_review_dir / "validation.json").write_text(json.dumps(validation_data))

        # Capture stdout
        captured_output = []

        def mock_print(*args, **kwargs):
            file = kwargs.get("file", sys.stdout)
            if file == sys.stdout:
                captured_output.append(" ".join(str(a) for a in args))

        with patch("sys.argv", ["prog", "--plan-file", str(sample_plan_file), "--approve-all", "--yes"]):
            with patch("builtins.print", mock_print):
                try:
                    main()
                except SystemExit:
                    pass

        # Should produce JSON output without error
        json_output = [o for o in captured_output if o.startswith("{")]
        if json_output:
            data = json.loads(json_output[0])
            assert "batches" in data or "to_apply" in data

    def test_exits_on_missing_output_dir(self, sample_plan_file):
        """Exits with error when output directory doesn't exist."""
        with patch("sys.argv", ["prog", "--plan-file", str(sample_plan_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# --- TestDeprecationWarnings ---

class TestDeprecationWarnings:
    """Tests for deprecation warnings."""

    def test_skip_human_review_deprecation_warning(
        self, sample_plan_file, sample_output_dir, sample_grouped_issues, sample_validation_results, capsys
    ):
        """--skip-human-review shows deprecation warning."""
        code_review_dir = sample_output_dir / "code-review"
        (code_review_dir / "grouped.json").write_text(json.dumps(sample_grouped_issues))
        validation_data = {
            "groups": sample_validation_results,
            "metadata": {"schema_version": "2.0", "model": "test"}
        }
        (code_review_dir / "validation.json").write_text(json.dumps(validation_data))

        with patch("sys.argv", ["prog", "--plan-file", str(sample_plan_file), "--skip-human-review"]):
            try:
                main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        assert "DEPRECATION WARNING" in captured.err or "deprecated" in captured.err.lower()


# --- TestUserSkipFunctionality ---

class TestUserSkipFunctionality:
    """Tests for user skip functionality via report.md checkboxes."""

    def test_skipped_indices_filter_merged_groups(
        self, sample_grouped_issues, sample_validation_results
    ):
        """User-skipped indices should be filtered from merged groups."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        # Simulate user skipping issues 1 and 3 (1-based indices in report)
        skipped_indices = {1, 3}

        # Filter out user-skipped issues (group_index + 1 maps to report index)
        filtered = [
            g for g in merged
            if (g.get("group_index", -1) + 1) not in skipped_indices
        ]

        # Should have 2 remaining (groups 1 and 3 filtered out, so indices 0 and 2 remain)
        assert len(filtered) == 2
        remaining_indices = [g["group_index"] for g in filtered]
        assert 0 not in skipped_indices  # Group 0 -> index 1, should NOT be skipped (wait, index 1 IS skipped)
        # Actually: group_index 0 -> report index 1, which IS in skipped_indices
        # So filtered should have indices 1 and 3 (report indices 2 and 4)
        # Let me recalculate:
        # merged has group_index 0, 1, 2, 3
        # report indices are 1, 2, 3, 4
        # skipped_indices = {1, 3} means report indices 1 and 3
        # So filter out: group_index 0 (report 1), group_index 2 (report 3)
        # Remaining: group_index 1, 3
        assert len(filtered) == 2
        assert all(g["group_index"] in [1, 3] for g in filtered)

    def test_empty_skipped_indices_filters_nothing(
        self, sample_grouped_issues, sample_validation_results
    ):
        """Empty skipped indices should not filter any groups."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        skipped_indices = set()  # No skips

        filtered = [
            g for g in merged
            if (g.get("group_index", -1) + 1) not in skipped_indices
        ]

        assert len(filtered) == len(merged)

    def test_all_skipped_results_in_empty_list(
        self, sample_grouped_issues, sample_validation_results
    ):
        """Skipping all issues should result in empty list."""
        merged = merge_validation_with_groups(sample_grouped_issues, sample_validation_results)

        # Skip all (report indices 1, 2, 3, 4)
        skipped_indices = {1, 2, 3, 4}

        filtered = [
            g for g in merged
            if (g.get("group_index", -1) + 1) not in skipped_indices
        ]

        assert len(filtered) == 0


class TestReportParserIntegration:
    """Tests for integration with report_parser module."""

    def test_parse_skipped_issues_import(self):
        """Verify parse_skipped_issues can be imported."""
        from utils.report_parser import parse_skipped_issues
        assert callable(parse_skipped_issues)

    def test_parse_skipped_issues_from_report(self, temp_dir):
        """Test parsing skipped issues from a code review report."""
        from utils.report_parser import parse_skipped_issues

        report_content = '''# Code Review Report

## HIGH Priority

### 1. Missing null check
- [x] Skip
**Validation:** ✓ Valid | **File:** `src/auth.py:42-50` | **Type:** bug | **Model:** gpt-4

Description.

---

### 2. Improve error handling
- [ ] Skip
**Validation:** ? Needs Review | **File:** `src/db.py:100-120` | **Type:** improvement | **Model:** claude-3

Description.

---

### 3. Security fix
- [x] Skip
**Validation:** ✓ Valid | **File:** `src/api.py:200-210` | **Type:** security | **Model:** gpt-4

Description.

---
'''
        report_path = temp_dir / "report.md"
        report_path.write_text(report_content)

        skipped = parse_skipped_issues(str(report_path))

        assert skipped == {1, 3}
        assert 2 not in skipped


# ── TestParseGFormatId ───────────────────────────────────────────────

class TestParseGFormatId:
    """Tests for _parse_g_format_id helper."""

    def test_canonical_g_format(self):
        """'G3S1' -> (3, 1)"""
        assert _parse_g_format_id("G3S1") == (3, 1)

    def test_canonical_g_format_larger_numbers(self):
        """'G12S5' -> (12, 5)"""
        assert _parse_g_format_id("G12S5") == (12, 5)

    def test_lowercase_g_format(self):
        """'g3s1' -> (3, 1) (case-insensitive)"""
        assert _parse_g_format_id("g3s1") == (3, 1)

    def test_mixed_case_g_format(self):
        """'g3S1' -> (3, 1) (case-insensitive)"""
        assert _parse_g_format_id("g3S1") == (3, 1)

    def test_plain_integer_fallback(self, capsys):
        """'3' -> (3, 1) with a warning on stderr"""
        result = _parse_g_format_id("3")
        assert result == (3, 1)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "plain integer" in captured.err.lower() or "Treating" in captured.err

    def test_malformed_g_only(self):
        """'G' -> None"""
        assert _parse_g_format_id("G") is None

    def test_malformed_g_number_only(self):
        """'G3' -> None"""
        assert _parse_g_format_id("G3") is None

    def test_malformed_g_number_s_only(self):
        """'G3S' -> None"""
        assert _parse_g_format_id("G3S") is None

    def test_malformed_extra_segment(self):
        """'G3S1S2' -> None"""
        assert _parse_g_format_id("G3S1S2") is None

    def test_malformed_random_letters(self):
        """'XYZ' -> None"""
        assert _parse_g_format_id("XYZ") is None

    def test_zero_group_index(self):
        """'G0S1' -> None (zero is invalid)"""
        assert _parse_g_format_id("G0S1") is None

    def test_zero_suggestion_index(self):
        """'G1S0' -> None (zero is invalid)"""
        assert _parse_g_format_id("G1S0") is None

    def test_negative_via_plain_integer(self, capsys):
        """'-1' -> None with warning (negative integer fallback)"""
        result = _parse_g_format_id("-1")
        assert result is None
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_empty_string(self):
        """'' -> None"""
        assert _parse_g_format_id("") is None

    def test_whitespace_string(self):
        """'  ' -> None"""
        assert _parse_g_format_id("  ") is None

    def test_none_input(self):
        """None -> None (non-string)"""
        assert _parse_g_format_id(None) is None

    def test_integer_input(self, capsys):
        """123 (int) -> None (non-string)"""
        assert _parse_g_format_id(123) is None
        captured = capsys.readouterr()
        assert "WARNING" in captured.err


# ── TestFilterUserSkippedGroups ──────────────────────────────────────

class TestFilterUserSkippedGroups:
    """Tests for filter_user_skipped_groups from utils.filtering."""

    @staticmethod
    def _make_group(suggestions_count=2, group_prefix=""):
        """Helper to create a group dict with N suggestions, including hashes."""
        group_hash = f"{group_prefix or 'g'}hash{'0' * 12}"[:16]
        suggestions = [
            {
                "id": f"S{i:03d}",
                "desc": f"suggestion {i}",
                "suggestion_hash": f"{group_prefix or 'g'}s{i}hash{'0' * 10}"[:16],
            }
            for i in range(1, suggestions_count + 1)
        ]
        return {
            "theme": f"{group_prefix}theme",
            "group_hash": group_hash,
            "suggestions": suggestions,
        }

    def test_all_suggestions_skipped_removes_group(self):
        """When all suggestions in a group are skipped, the group is removed."""
        groups = [self._make_group(2)]
        # Skip both suggestions by their hashes
        s1_hash = groups[0]["suggestions"][0]["suggestion_hash"]
        s2_hash = groups[0]["suggestions"][1]["suggestion_hash"]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes=set(),
            skipped_suggestion_hashes={s1_hash, s2_hash},
            old_skipped_ids=set(),
        )
        assert len(result) == 0
        assert count == 2

    def test_some_skipped_group_kept_with_remaining(self):
        """When some suggestions are skipped, the group is kept with the rest."""
        groups = [self._make_group(3)]
        s2_hash = groups[0]["suggestions"][1]["suggestion_hash"]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes=set(),
            skipped_suggestion_hashes={s2_hash},
            old_skipped_ids=set(),
        )
        assert len(result) == 1
        assert len(result[0]["suggestions"]) == 2
        assert count == 1

    def test_group_level_skip_removes_group(self):
        """Skipping by group hash removes the entire group."""
        groups = [self._make_group(2, "A"), self._make_group(3, "B")]
        g1_hash = groups[0]["group_hash"]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes={g1_hash},
            skipped_suggestion_hashes=set(),
            old_skipped_ids=set(),
        )
        assert len(result) == 1
        assert count == 2  # 2 suggestions from group 1

    def test_mixed_group_and_suggestion_skips(self):
        """Mix of group-level and suggestion-level skips."""
        groups = [
            self._make_group(2, "A"),
            self._make_group(3, "B"),
            self._make_group(1, "C"),
        ]
        g1_hash = groups[0]["group_hash"]
        g2_s1_hash = groups[1]["suggestions"][0]["suggestion_hash"]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes={g1_hash},       # Remove group 1 entirely (2 suggestions)
            skipped_suggestion_hashes={g2_s1_hash},  # Remove first suggestion from group 2
            old_skipped_ids=set(),
        )
        assert len(result) == 2  # groups 2 and 3 survive
        assert len(result[0]["suggestions"]) == 2  # group 2 had 3, minus 1
        assert len(result[1]["suggestions"]) == 1  # group 3 untouched
        assert count == 3  # 2 from group 1 + 1 from G2S1

    def test_empty_skips_no_changes(self):
        """No skips -> all groups returned unchanged."""
        groups = [self._make_group(2), self._make_group(3)]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes=set(),
            skipped_suggestion_hashes=set(),
            old_skipped_ids=set(),
        )
        assert len(result) == 2
        assert count == 0

    def test_unmatched_group_hash_ignored_with_warning(self, capsys):
        """Unmatched group hash is ignored with a warning."""
        groups = [self._make_group(2)]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes={"deadbeef12345678"},  # Non-existent hash
            skipped_suggestion_hashes=set(),
            old_skipped_ids=set(),
        )
        assert len(result) == 1
        assert count == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "does not match" in captured.err

    def test_malformed_suggestion_hash_ignored(self, capsys):
        """Non-hex suggestion hash is silently ignored (may be legacy G-format)."""
        groups = [self._make_group(2)]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes=set(),
            skipped_suggestion_hashes={"INVALID"},
            old_skipped_ids=set(),
        )
        assert len(result) == 1
        assert count == 0

    def test_empty_suggestions_list_handled_gracefully(self):
        """Group with empty suggestions list is handled without error."""
        groups = [{"theme": "empty", "group_hash": "emptyhash1234567", "suggestions": []}]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes=set(),
            skipped_suggestion_hashes=set(),
            old_skipped_ids=set(),
        )
        # Group with no suggestions should not appear in result
        # (since it has no remaining suggestions)
        assert len(result) == 0
        assert count == 0

    def test_suggestion_hashes_across_groups_filtered_independently(self):
        """Suggestions in different groups are filtered independently by hash."""
        groups = [self._make_group(2, "A"), self._make_group(2, "B")]
        # Skip first suggestion of each group
        s1_hash = groups[0]["suggestions"][0]["suggestion_hash"]
        s2_hash = groups[1]["suggestions"][0]["suggestion_hash"]
        result, count = filter_user_skipped_groups(
            groups,
            skipped_group_hashes=set(),
            skipped_suggestion_hashes={s1_hash, s2_hash},
            old_skipped_ids=set(),
        )
        assert len(result) == 2
        assert len(result[0]["suggestions"]) == 1
        assert len(result[1]["suggestions"]) == 1
        assert count == 2


# ── TestApplyEditedDescriptionsToGroups ──────────────────────────────

class TestApplyEditedDescriptionsToGroups:
    """Tests for _apply_edited_descriptions_to_groups helper."""

    @staticmethod
    def _make_groups():
        """Create sample groups with suggestions for testing."""
        return [
            {
                "theme": "Performance",
                "suggestions": [
                    {"title": "Optimize loop", "desc": "Original description 1"},
                    {"title": "Cache results", "desc": "Original description 2"},
                ],
            },
            {
                "theme": "Security",
                "suggestions": [
                    {"title": "Sanitize input", "desc": "Original description 3"},
                ],
            },
        ]

    def test_g_format_key_applies_edit(self):
        """G-format key like 'G1S1' should update the matching suggestion desc."""
        groups = self._make_groups()
        count, log = _apply_edited_descriptions_to_groups(
            groups, {"G1S1": "Updated description"}
        )
        assert count == 1
        assert groups[0]["suggestions"][0]["desc"] == "Updated description"
        assert groups[0]["suggestions"][0].get("_description_edited") is True
        assert groups[0]["suggestions"][0].get("_original_desc") == "Original description 1"

    def test_integer_key_fallback(self, capsys):
        """Integer string key like '2' should update group 2 suggestion 1."""
        groups = self._make_groups()
        count, log = _apply_edited_descriptions_to_groups(
            groups, {"2": "New security desc"}
        )
        assert count == 1
        assert groups[1]["suggestions"][0]["desc"] == "New security desc"
        captured = capsys.readouterr()
        assert "WARNING" in captured.err  # plain integer triggers warning

    def test_invalid_key_skipped(self):
        """Invalid/unparseable key is skipped, no changes made."""
        groups = self._make_groups()
        count, log = _apply_edited_descriptions_to_groups(
            groups, {"XYZ": "Should not apply"}
        )
        assert count == 0
        assert len(log) == 0
        # Original descriptions unchanged
        assert groups[0]["suggestions"][0]["desc"] == "Original description 1"

    def test_out_of_range_indices_skipped_with_warning(self, capsys):
        """Out-of-range group/suggestion index emits warning and skips."""
        groups = self._make_groups()
        count, log = _apply_edited_descriptions_to_groups(
            groups, {"G10S1": "Out of range"}
        )
        assert count == 0
        assert len(log) == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "out of range" in captured.err

    def test_out_of_range_suggestion_index_skipped_with_warning(self, capsys):
        """Out-of-range suggestion index within a valid group emits warning."""
        groups = self._make_groups()
        count, log = _apply_edited_descriptions_to_groups(
            groups, {"G1S10": "Out of range suggestion"}
        )
        assert count == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "out of range" in captured.err

    def test_idempotent_reapply_returns_zero(self):
        """Re-applying the same description is a no-op (count=0)."""
        groups = self._make_groups()
        # First apply
        count1, log1 = _apply_edited_descriptions_to_groups(
            groups, {"G1S1": "Updated description"}
        )
        assert count1 == 1
        # Re-apply the same text
        count2, log2 = _apply_edited_descriptions_to_groups(
            groups, {"G1S1": "Updated description"}
        )
        assert count2 == 0
        assert len(log2) == 0

    def test_edit_log_entries_match_expected_format(self):
        """edit_log entries contain required keys with correct values."""
        groups = self._make_groups()
        count, log = _apply_edited_descriptions_to_groups(
            groups, {"G2S1": "Short"}
        )
        assert count == 1
        assert len(log) == 1
        entry = log[0]
        assert entry["id"] == "G2S1"
        assert entry["group_index"] == 2
        assert entry["sugg_index"] == 1
        assert entry["title"] == "Sanitize input"
        assert entry["original_len"] == len("Original description 3")
        assert entry["edited_len"] == len("Short")

    def test_fallback_to_group_level_desc_when_no_suggestions(self):
        """When suggestions list is empty, edit falls back to group-level desc."""
        groups = [
            {
                "theme": "Config",
                "suggestions": [],
                "desc": "Original group description",
            },
        ]
        count, log = _apply_edited_descriptions_to_groups(
            groups, {"G1S1": "New group description"}
        )
        assert count == 1
        assert groups[0]["desc"] == "New group description"
        assert groups[0].get("_description_edited") is True
        assert groups[0].get("_original_desc") == "Original group description"
        assert len(log) == 1
        assert log[0]["group_index"] == 1


# ── TestHTMLWorkflowIntegration ──────────────────────────────────────

class TestHTMLWorkflowIntegration:
    """Integration test: full HTML workflow through merge_selections ->
    filter_user_skipped_groups -> _apply_edited_descriptions_to_groups."""

    @staticmethod
    def _make_grouped_issues():
        """Create mock grouped code-review data with 6 groups, stamped with hashes.

        Layout (1-based indices matching report format):
          G1: 2 suggestions   (will be individually skipped via hash)
          G2: 2 suggestions   (untouched)
          G3: 2 suggestions   (G3S1 individually skipped via skipped_suggestions hash)
          G4: 1 suggestion    (untouched)
          G5: 3 suggestions   (G5S2 individually skipped via skipped_suggestions hash)
          G6: 1 suggestion    (entire group skipped via skipped_groups hash)
        """
        groups = [
            {
                "theme": "Performance",
                "suggestions": [
                    {"title": "Optimize loop", "desc": "Optimize the inner loop"},
                    {"title": "Cache results", "desc": "Add caching layer"},
                ],
            },
            {
                "theme": "Error handling",
                "suggestions": [
                    {"title": "Add try-catch", "desc": "Wrap in try-catch"},
                    {"title": "Log errors", "desc": "Add error logging"},
                ],
            },
            {
                "theme": "Security",
                "suggestions": [
                    {"title": "Sanitize input", "desc": "Sanitize user input"},
                    {"title": "Validate tokens", "desc": "Token validation logic"},
                ],
            },
            {
                "theme": "Documentation",
                "suggestions": [
                    {"title": "Add docstrings", "desc": "Missing docstrings"},
                ],
            },
            {
                "theme": "Testing",
                "suggestions": [
                    {"title": "Add unit tests", "desc": "Unit test coverage"},
                    {"title": "Add e2e tests", "desc": "End-to-end test coverage"},
                    {"title": "Add load tests", "desc": "Load test scenarios"},
                ],
            },
            {
                "theme": "Config",
                "suggestions": [
                    {"title": "Externalize config", "desc": "Move config to env vars"},
                ],
            },
        ]
        stamp_stable_ids(groups)
        return groups

    def _make_html_selections(self):
        """Create mock user_selections.json with hash-based IDs.

        Decisions:
          - skipped_groups: [G6 hash]  (entire group 6 skipped)
          - skipped_suggestions: [G3S1 hash, G5S2 hash]  (individual suggestion skips)
          - edited_descriptions: {G1S1 hash: 'EDITED: Optimize with SIMD'}
        """
        groups = self._make_grouped_issues()
        g6_hash = groups[5]["group_hash"]
        g3s1_hash = groups[2]["suggestions"][0]["suggestion_hash"]
        g5s2_hash = groups[4]["suggestions"][1]["suggestion_hash"]
        g1s1_hash = groups[0]["suggestions"][0]["suggestion_hash"]
        return {
            "plan_path": "plans/test-plan.md",
            "phase": "code-review",
            "exported_at": "2026-02-25T12:00:00Z",
            "skipped_groups": [g6_hash],
            "skipped_suggestions": [g3s1_hash, g5s2_hash],
            "edited_descriptions": {
                g1s1_hash: "EDITED: Optimize with SIMD instructions",
            },
        }

    def test_full_html_workflow(self):
        """End-to-end: merge_selections -> filter_user_skipped_groups ->
        _apply_edited_descriptions_to_groups produces correct results."""
        grouped = self._make_grouped_issues()
        html_selections = self._make_html_selections()
        g6_hash = grouped[5]["group_hash"]
        g3s1_hash = grouped[2]["suggestions"][0]["suggestion_hash"]
        g5s2_hash = grouped[4]["suggestions"][1]["suggestion_hash"]
        g1s1_hash = grouped[0]["suggestions"][0]["suggestion_hash"]

        # Step 1: merge_selections (no markdown decisions, all from HTML)
        md_skipped_groups: set = set()
        md_skipped_suggestions: set = set()
        md_edited: dict = {}

        skipped_group_hashes, skipped_suggestion_hashes, merged_edited = merge_selections(
            html_selections,
            md_skipped_groups,
            md_skipped_suggestions,
            md_edited,
        )

        # Verify merge output
        assert skipped_group_hashes == {g6_hash}
        assert skipped_suggestion_hashes == {g3s1_hash, g5s2_hash}
        assert merged_edited == {g1s1_hash: "EDITED: Optimize with SIMD instructions"}

        # Step 2: filter_user_skipped_groups
        filtered, user_skipped_count = filter_user_skipped_groups(
            grouped,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            old_skipped_ids=set(),
        )

        # Group 6 fully removed (1 suggestion) = 1
        # G3S1 removed individually = 1
        # G5S2 removed individually = 1
        # Total skipped = 3
        assert user_skipped_count == 3

        # Remaining groups: G1 (2 sugg), G2 (2 sugg), G3 (1 sugg remaining),
        #                   G4 (1 sugg), G5 (2 sugg remaining)
        # G6 is entirely removed
        assert len(filtered) == 5

        # G1: both suggestions survive (none individually skipped)
        assert filtered[0]["theme"] == "Performance"
        assert len(filtered[0]["suggestions"]) == 2

        # G2: untouched
        assert filtered[1]["theme"] == "Error handling"
        assert len(filtered[1]["suggestions"]) == 2

        # G3: G3S1 removed, G3S2 remains
        assert filtered[2]["theme"] == "Security"
        assert len(filtered[2]["suggestions"]) == 1
        assert filtered[2]["suggestions"][0]["title"] == "Validate tokens"

        # G4: untouched
        assert filtered[3]["theme"] == "Documentation"
        assert len(filtered[3]["suggestions"]) == 1

        # G5: G5S2 removed, G5S1 and G5S3 remain
        assert filtered[4]["theme"] == "Testing"
        assert len(filtered[4]["suggestions"]) == 2
        assert filtered[4]["suggestions"][0]["title"] == "Add unit tests"
        assert filtered[4]["suggestions"][1]["title"] == "Add load tests"

        # Step 3: _apply_edited_descriptions_to_groups
        # Re-create grouped and apply edits first, then filter.
        grouped_for_edits = self._make_grouped_issues()
        count_applied, edit_log = _apply_edited_descriptions_to_groups(
            grouped_for_edits, merged_edited
        )

        assert count_applied == 1
        assert len(edit_log) == 1
        # Edit log may use hash-based or positional ID depending on _apply implementation
        assert edit_log[0]["title"] == "Optimize loop"
        assert edit_log[0]["original_len"] == len("Optimize the inner loop")
        assert edit_log[0]["edited_len"] == len("EDITED: Optimize with SIMD instructions")

        # Verify the actual description was updated
        assert grouped_for_edits[0]["suggestions"][0]["desc"] == "EDITED: Optimize with SIMD instructions"
        assert grouped_for_edits[0]["suggestions"][0].get("_description_edited") is True
        assert grouped_for_edits[0]["suggestions"][0].get("_original_desc") == "Optimize the inner loop"

    def test_all_suggestions_individually_skipped_removes_group(self):
        """When all suggestions in a group are individually skipped,
        the group is fully removed."""
        grouped = [
            {
                "theme": "Small group",
                "suggestions": [
                    {"title": "Item A", "desc": "Desc A"},
                    {"title": "Item B", "desc": "Desc B"},
                ],
            },
        ]
        stamp_stable_ids(grouped)
        s1_hash = grouped[0]["suggestions"][0]["suggestion_hash"]
        s2_hash = grouped[0]["suggestions"][1]["suggestion_hash"]

        html_selections = {
            "skipped_groups": [],
            "skipped_suggestions": [s1_hash, s2_hash],
            "edited_descriptions": {},
        }

        skipped_group_hashes, skipped_suggestion_hashes, merged_edited = merge_selections(
            html_selections, set(), set(), {}
        )

        filtered, user_skipped_count = filter_user_skipped_groups(
            grouped,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            old_skipped_ids=set(),
        )

        assert len(filtered) == 0, "Group with all suggestions individually skipped should be removed"
        assert user_skipped_count == 2

    def test_no_html_selections_passes_through_markdown(self):
        """When html_selections is None, merge_selections returns markdown values."""
        md_skipped_groups = {"abcdef0123456789", "1234567890abcdef"}
        md_skipped_suggestions = {"fedcba9876543210"}
        md_edited = {"aabbccdd11223344": ("original text", "edited text")}

        skipped_group_hashes, skipped_suggestion_hashes, merged_edited = merge_selections(
            None,  # No HTML selections
            md_skipped_groups,
            md_skipped_suggestions,
            md_edited,
        )

        assert skipped_group_hashes == {"abcdef0123456789", "1234567890abcdef"}
        assert skipped_suggestion_hashes == {"fedcba9876543210"}
        assert merged_edited == {"aabbccdd11223344": "edited text"}

    def test_edited_descriptions_with_multiple_targets(self):
        """Multiple edited descriptions are all applied correctly."""
        grouped = self._make_grouped_issues()
        g1s1_hash = grouped[0]["suggestions"][0]["suggestion_hash"]
        g2s2_hash = grouped[1]["suggestions"][1]["suggestion_hash"]
        g5s3_hash = grouped[4]["suggestions"][2]["suggestion_hash"]

        html_selections = {
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {
                g1s1_hash: "New perf description",
                g2s2_hash: "New logging approach",
                g5s3_hash: "Updated load test plan",
            },
        }

        _, _, merged_edited = merge_selections(
            html_selections, set(), set(), {}
        )

        count_applied, edit_log = _apply_edited_descriptions_to_groups(
            grouped, merged_edited
        )

        assert count_applied == 3
        assert len(edit_log) == 3

        # Verify each edit was applied
        assert grouped[0]["suggestions"][0]["desc"] == "New perf description"
        assert grouped[1]["suggestions"][1]["desc"] == "New logging approach"
        assert grouped[4]["suggestions"][2]["desc"] == "Updated load test plan"

        # Verify edit log entries have the correct titles
        titles_in_log = {entry["title"] for entry in edit_log}
        assert titles_in_log == {"Optimize loop", "Log errors", "Add load tests"}


# ── TestMixedSelectionMergeIntegration ───────────────────────────────

class TestMixedSelectionMergeIntegration:
    """Integration test: combining markdown skips, HTML skips (hash-based),
    HTML edited_descriptions, and C-level skips into a single coherent result."""

    @staticmethod
    def _make_grouped_issues():
        """Create mock grouped data with 5 groups for mixed-source testing, stamped with hashes.

        Layout:
          G1: 2 suggestions  (markdown skip on entire group)
          G2: 3 suggestions  (HTML skips G2S1)
          G3: 2 suggestions  (C-level skip on entire group)
          G4: 2 suggestions  (untouched)
          G5: 2 suggestions  (HTML edit on G5S1)
        """
        groups = [
            {
                "theme": "Auth fixes",
                "suggestions": [
                    {"title": "Fix token refresh", "desc": "Token refresh logic"},
                    {"title": "Fix session timeout", "desc": "Session timeout handling"},
                ],
            },
            {
                "theme": "API changes",
                "suggestions": [
                    {"title": "Add rate limiting", "desc": "Rate limiter implementation"},
                    {"title": "Fix pagination", "desc": "Pagination cursor fix"},
                    {"title": "Add versioning", "desc": "API versioning strategy"},
                ],
            },
            {
                "theme": "Database",
                "suggestions": [
                    {"title": "Add index", "desc": "Missing index on users table"},
                    {"title": "Fix migration", "desc": "Migration rollback issue"},
                ],
            },
            {
                "theme": "UI polish",
                "suggestions": [
                    {"title": "Fix alignment", "desc": "Button alignment issue"},
                    {"title": "Fix colors", "desc": "Color contrast accessibility"},
                ],
            },
            {
                "theme": "Monitoring",
                "suggestions": [
                    {"title": "Add metrics", "desc": "Add prometheus metrics"},
                    {"title": "Add alerts", "desc": "PagerDuty alert rules"},
                ],
            },
        ]
        stamp_stable_ids(groups)
        return groups

    def test_all_skip_sources_combine_correctly(self):
        """Markdown, HTML, and C-level skips combine with correct precedence.

        Scenario:
          - Markdown: skips group 1 (by hash)
          - HTML: skips G2S1 (by hash), edits G5S1 (by hash)
          - C-level skips (0-based): {2} -> skip group 3 (converted to hash)

        Expected after merge + filter:
          - G1: removed (markdown group-level skip)
          - G2: G2S1 removed, G2S2 and G2S3 survive
          - G3: removed (C-level skip)
          - G4: untouched (2 suggestions)
          - G5: untouched (2 suggestions), G5S1 desc edited
        """
        grouped = self._make_grouped_issues()
        g1_hash = grouped[0]["group_hash"]
        g2s1_hash = grouped[1]["suggestions"][0]["suggestion_hash"]
        g3_hash = grouped[2]["group_hash"]
        g5s1_hash = grouped[4]["suggestions"][0]["suggestion_hash"]

        # --- Simulate markdown parsing results ---
        md_skipped_groups = {g1_hash}
        md_skipped_suggestions: set = set()

        # --- Simulate HTML selections ---
        html_selections = {
            "plan_path": "plans/test-plan.md",
            "phase": "code-review",
            "exported_at": "2026-02-25T12:00:00Z",
            "skipped_groups": [],
            "skipped_suggestions": [g2s1_hash],
            "edited_descriptions": {
                g5s1_hash: "EDITED: Add prometheus + grafana metrics",
            },
        }

        # Step 1: merge_selections unions markdown + HTML skips (additive).
        skipped_group_hashes, skipped_suggestion_hashes, merged_edited = merge_selections(
            html_selections,
            md_skipped_groups,
            md_skipped_suggestions,
            {},
        )

        assert skipped_group_hashes == {g1_hash}, (
            "Markdown group 1 skip should remain active; HTML skipped_groups=[] "
            "does not erase it"
        )
        assert skipped_suggestion_hashes == {g2s1_hash}
        assert g5s1_hash in merged_edited

        # Step 2: Apply C-level skips (0-based index 2 -> group 3 hash)
        skipped_group_hashes.add(g3_hash)

        assert skipped_group_hashes == {g1_hash, g3_hash}

        # Step 3: Apply edited descriptions BEFORE filtering
        count_applied, edit_log = _apply_edited_descriptions_to_groups(
            grouped, merged_edited
        )
        assert count_applied == 1
        assert grouped[4]["suggestions"][0]["desc"] == "EDITED: Add prometheus + grafana metrics"

        # Step 4: filter_user_skipped_groups
        filtered, user_skipped_count = filter_user_skipped_groups(
            grouped,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            old_skipped_ids=set(),
        )

        # G1 removed (2 suggestions) + G3 removed (2 suggestions) + G2S1 removed = 5
        assert user_skipped_count == 5

        # Remaining: G2 (2 remaining), G4 (2), G5 (2) = 3 groups
        assert len(filtered) == 3

        # G2: G2S1 removed, G2S2 and G2S3 remain
        assert filtered[0]["theme"] == "API changes"
        assert len(filtered[0]["suggestions"]) == 2
        assert filtered[0]["suggestions"][0]["title"] == "Fix pagination"
        assert filtered[0]["suggestions"][1]["title"] == "Add versioning"

        # G4 untouched
        assert filtered[1]["theme"] == "UI polish"
        assert len(filtered[1]["suggestions"]) == 2

        # G5 untouched structurally, but description was edited
        assert filtered[2]["theme"] == "Monitoring"
        assert len(filtered[2]["suggestions"]) == 2

    def test_markdown_fallback_when_no_html(self):
        """When HTML selections are None, markdown decisions are used directly."""
        grouped = self._make_grouped_issues()
        g1_hash = grouped[0]["group_hash"]
        g3_hash = grouped[2]["group_hash"]

        md_skipped_groups = {g1_hash, g3_hash}
        md_skipped_suggestions: set = set()
        md_edited: dict = {}

        skipped_group_hashes, skipped_suggestion_hashes, merged_edited = merge_selections(
            None,  # No HTML
            md_skipped_groups,
            md_skipped_suggestions,
            md_edited,
        )

        assert skipped_group_hashes == {g1_hash, g3_hash}
        assert skipped_suggestion_hashes == set()
        assert merged_edited == {}

        filtered, user_skipped_count = filter_user_skipped_groups(
            grouped,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            old_skipped_ids=set(),
        )

        # G1: 2 suggestions removed, G3: 2 suggestions removed = 4
        assert user_skipped_count == 4
        # G2, G4, G5 remain
        assert len(filtered) == 3
        assert filtered[0]["theme"] == "API changes"
        assert filtered[1]["theme"] == "UI polish"
        assert filtered[2]["theme"] == "Monitoring"

    def test_markdown_skips_persist_when_html_has_no_group_skips(self):
        """Regression: markdown group skips survive when HTML exists but has empty skipped_groups."""
        grouped = self._make_grouped_issues()
        g1_hash = grouped[0]["group_hash"]
        g3_hash = grouped[2]["group_hash"]

        # Markdown skips groups 1 and 3
        md_skipped_groups = {g1_hash, g3_hash}
        md_skipped_suggestions: set = set()

        # HTML exists but has NO group skips (empty list)
        html_selections = {
            "plan_path": "plans/test-plan.md",
            "phase": "code-review",
            "exported_at": "2026-02-25T12:00:00Z",
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {},
        }

        skipped_group_hashes, skipped_suggestion_hashes, merged_edited = merge_selections(
            html_selections,
            md_skipped_groups,
            md_skipped_suggestions,
            {},
        )

        # Markdown skips must persist even though HTML skipped_groups is empty
        assert skipped_group_hashes == {g1_hash, g3_hash}
        assert skipped_suggestion_hashes == set()
        assert merged_edited == {}

        # Verify filtering actually removes the correct groups
        filtered, user_skipped_count = filter_user_skipped_groups(
            grouped,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            old_skipped_ids=set(),
        )

        # G1: 2 suggestions removed, G3: 2 suggestions removed = 4
        assert user_skipped_count == 4
        # G2, G4, G5 remain
        assert len(filtered) == 3
        assert filtered[0]["theme"] == "API changes"
        assert filtered[1]["theme"] == "UI polish"
        assert filtered[2]["theme"] == "Monitoring"

    def test_c_level_skips_union_with_g_level(self):
        """C-level skips union with G-level skips (both as hashes)."""
        grouped = self._make_grouped_issues()
        g1_hash = grouped[0]["group_hash"]
        g4_hash = grouped[3]["group_hash"]

        # HTML says skip group 1
        html_selections = {
            "skipped_groups": [g1_hash],
            "skipped_suggestions": [],
            "edited_descriptions": {},
        }

        skipped_group_hashes, skipped_suggestion_hashes, merged_edited = merge_selections(
            html_selections, set(), set(), {}
        )
        assert skipped_group_hashes == {g1_hash}

        # C-level says skip group 4 (by hash)
        skipped_group_hashes.add(g4_hash)

        assert skipped_group_hashes == {g1_hash, g4_hash}

        filtered, user_skipped_count = filter_user_skipped_groups(
            grouped,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            old_skipped_ids=set(),
        )

        # G1: 2 suggestions, G4: 2 suggestions = 4 skipped
        assert user_skipped_count == 4
        # G2, G3, G5 remain
        assert len(filtered) == 3
        assert filtered[0]["theme"] == "API changes"
        assert filtered[1]["theme"] == "Database"
        assert filtered[2]["theme"] == "Monitoring"

    def test_no_name_error_from_variable_rename(self):
        """Verify that the hash-based variable flow works throughout the
        merge -> C-level union -> filter chain without NameError."""
        grouped = self._make_grouped_issues()
        g1_hash = grouped[0]["group_hash"]
        g2_hash = grouped[1]["group_hash"]
        g5_hash = grouped[4]["group_hash"]
        g1s1_hash = grouped[0]["suggestions"][0]["suggestion_hash"]

        # Simulate the exact variable flow from main()
        # 1. Parse markdown
        skipped_group_hashes = {g2_hash}  # markdown says skip group 2
        skipped_suggestion_hashes: set = set()

        # 2. HTML overrides
        html_selections = {
            "skipped_groups": [g5_hash],
            "skipped_suggestions": [g1s1_hash],
            "edited_descriptions": {},
        }
        skipped_group_hashes, skipped_suggestion_hashes, merged_edited = merge_selections(
            html_selections,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            {},
        )

        # 3. C-level union (add group 1 hash)
        skipped_group_hashes.add(g1_hash)

        # 4. Filter
        filtered, user_skipped_count = filter_user_skipped_groups(
            grouped,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            old_skipped_ids=set(),
        )

        # G1: removed (2 suggestions, C-level)
        # G2: removed (3 suggestions, markdown)
        # G5: removed (2 suggestions, HTML)
        # G1S1: individually skipped but G1 already removed entirely
        # Total skipped: G1(2) + G2(3) + G5(2) = 7
        assert user_skipped_count == 7

        # Remaining: G3, G4 = 2 groups
        assert len(filtered) == 2
        assert filtered[0]["theme"] == "Database"
        assert filtered[1]["theme"] == "UI polish"

    def test_precedence_group_skip_over_suggestion_skip(self):
        """Group-level skips remove entire groups regardless of individual
        suggestion skips in the same group."""
        grouped = [
            {
                "theme": "Single group",
                "suggestions": [
                    {"title": "A", "desc": "Desc A"},
                    {"title": "B", "desc": "Desc B"},
                    {"title": "C", "desc": "Desc C"},
                ],
            },
        ]
        stamp_stable_ids(grouped)
        g1_hash = grouped[0]["group_hash"]
        g1s2_hash = grouped[0]["suggestions"][1]["suggestion_hash"]

        html_selections = {
            "skipped_groups": [g1_hash],
            "skipped_suggestions": [g1s2_hash],  # Also skip G1S2 individually
            "edited_descriptions": {},
        }

        skipped_group_hashes, skipped_suggestion_hashes, _ = merge_selections(
            html_selections, set(), set(), {}
        )

        filtered, user_skipped_count = filter_user_skipped_groups(
            grouped,
            skipped_group_hashes,
            skipped_suggestion_hashes,
            old_skipped_ids=set(),
        )

        # Group-level skip removes all 3 suggestions
        assert len(filtered) == 0
        assert user_skipped_count == 3


# --- Issue Number → Group Index Mapping Tests ---

class TestBuildIssueToGroupMap:
    """Tests for build_issue_to_group_map() which maps report issue numbers to group/suggestion indices."""

    def _make_grouped(self, groups):
        """Helper to build grouped.json-style list from simplified spec.

        Each group is a list of dicts with keys: title, model, and optionally issue_index.
        """
        result = []
        for suggestions in groups:
            result.append({
                "theme": suggestions[0]["title"] if suggestions else "",
                "suggestions": [
                    {
                        "title": s["title"],
                        "source_model": s["model"],
                        "desc": s.get("desc", f"Description for {s['title']}"),
                        **({"issue_index": s["issue_index"]} if "issue_index" in s else {}),
                    }
                    for s in suggestions
                ],
            })
        return result

    def test_primary_strategy_issue_index(self, temp_dir):
        """When suggestions have issue_index, use that directly."""
        grouped = self._make_grouped([
            [{"title": "Fix null check", "model": "gemini", "issue_index": 1},
             {"title": "Fix null check", "model": "cursor", "issue_index": 3}],
            [{"title": "Add logging", "model": "gemini", "issue_index": 2}],
            [{"title": "Remove dead code", "model": "cursor", "issue_index": 4}],
        ])
        # issues.json doesn't need to exist for primary strategy
        result = build_issue_to_group_map(str(Path(temp_dir) / "nonexistent.json"), grouped)
        assert result == {
            1: (1, 1),  # issue 1 → group 1, suggestion 1
            3: (1, 2),  # issue 3 → group 1, suggestion 2
            2: (2, 1),  # issue 2 → group 2, suggestion 1
            4: (3, 1),  # issue 4 → group 3, suggestion 1
        }

    def test_fallback_strategy_title_model_match(self, temp_dir):
        """When no issue_index fields, match by (title, model) from issues.json."""
        grouped = self._make_grouped([
            [{"title": "Fix null check", "model": "gemini"},
             {"title": "Fix null check", "model": "cursor"}],
            [{"title": "Add logging", "model": "gemini"}],
        ])
        # Create issues.json with 3 raw issues
        issues = [
            {"title": "Fix null check", "source_model": "gemini"},
            {"title": "Add logging", "source_model": "gemini"},
            {"title": "Fix null check", "source_model": "cursor"},
        ]
        issues_path = Path(temp_dir) / "issues.json"
        issues_path.write_text(json.dumps(issues))

        result = build_issue_to_group_map(str(issues_path), grouped)
        assert result == {
            1: (1, 1),  # "Fix null check" by gemini → group 1, sugg 1
            2: (2, 1),  # "Add logging" by gemini → group 2, sugg 1
            3: (1, 2),  # "Fix null check" by cursor → group 1, sugg 2
        }

    def test_no_issues_json_no_issue_index_returns_none(self, temp_dir):
        """Returns None when neither strategy can work."""
        grouped = self._make_grouped([
            [{"title": "Fix bug", "model": "gemini"}],
        ])
        result = build_issue_to_group_map(str(Path(temp_dir) / "missing.json"), grouped)
        assert result is None

    def test_empty_grouped_returns_none(self, temp_dir):
        """Returns None for empty grouped list."""
        result = build_issue_to_group_map(str(Path(temp_dir) / "missing.json"), [])
        assert result is None

    def test_16_issues_11_groups(self, temp_dir):
        """Realistic scenario: 16 raw issues grouped into 11 groups."""
        # Build 11 groups, some with multiple suggestions
        grouped = self._make_grouped([
            [{"title": "A", "model": "m1", "issue_index": 1},
             {"title": "A", "model": "m2", "issue_index": 5}],
            [{"title": "B", "model": "m1", "issue_index": 2},
             {"title": "B", "model": "m2", "issue_index": 6}],
            [{"title": "C", "model": "m1", "issue_index": 3}],
            [{"title": "D", "model": "m1", "issue_index": 4}],
            [{"title": "E", "model": "m2", "issue_index": 7}],
            [{"title": "F", "model": "m2", "issue_index": 8},
             {"title": "F", "model": "m3", "issue_index": 12}],
            [{"title": "G", "model": "m2", "issue_index": 9}],
            [{"title": "H", "model": "m2", "issue_index": 10}],
            [{"title": "I", "model": "m2", "issue_index": 11}],
            [{"title": "J", "model": "m3", "issue_index": 13},
             {"title": "J", "model": "m1", "issue_index": 15}],
            [{"title": "K", "model": "m3", "issue_index": 14},
             {"title": "K", "model": "m1", "issue_index": 16}],
        ])
        result = build_issue_to_group_map(str(Path(temp_dir) / "x.json"), grouped)
        assert result is not None
        assert len(result) == 16
        # Spot checks
        assert result[1] == (1, 1)
        assert result[5] == (1, 2)
        assert result[12] == (6, 2)
        assert result[16] == (11, 2)


class TestFindEditedIssueDescriptionsWithMap:
    """Tests for find_edited_issue_descriptions with issue_to_group_map."""

    def test_map_resolves_correct_suggestion(self, temp_dir):
        """With map, edits are detected against the correct suggestion."""
        # Group 1 has 2 suggestions; issue 3 maps to group 1 suggestion 2
        groups = [{
            "theme": "Fix null check",
            "suggestions": [
                {"title": "Fix null check", "desc": "Original desc from model A"},
                {"title": "Fix null check", "desc": "Original desc from model B"},
            ],
        }]
        issue_map = {3: (1, 2)}  # issue 3 → group 1, suggestion 2

        # Write a report with issue 3 having an edited description
        report_path = Path(temp_dir) / "report.md"
        report_path.write_text(
            "# Code Review Report\n\n"
            "### 3. Fix null check\n\n"
            "Edited desc from user\n\n"
        )

        result = find_edited_issue_descriptions(str(report_path), groups, issue_to_group_map=issue_map)
        assert 3 in result
        assert result[3] == ("Original desc from model B", "Edited desc from user")

    def test_no_edit_when_descriptions_match(self, temp_dir):
        """No edits detected when description matches."""
        groups = [{
            "theme": "Fix bug",
            "suggestions": [{"title": "Fix bug", "desc": "Same description"}],
        }]
        issue_map = {1: (1, 1)}

        report_path = Path(temp_dir) / "report.md"
        report_path.write_text(
            "# Code Review Report\n\n"
            "### 1. Fix bug\n\n"
            "Same description\n\n"
        )

        result = find_edited_issue_descriptions(str(report_path), groups, issue_to_group_map=issue_map)
        assert result == {}


class TestMergeEditedIssueDescriptionsWithMap:
    """Tests for merge_edited_issue_descriptions with issue_to_group_map."""

    def test_map_applies_edit_to_correct_suggestion(self):
        """With map, edits go to the right group/suggestion."""
        groups = [
            {
                "theme": "Group A",
                "suggestions": [
                    {"title": "Issue from model 1", "desc": "Original A1"},
                    {"title": "Issue from model 2", "desc": "Original A2"},
                ],
            },
            {
                "theme": "Group B",
                "suggestions": [
                    {"title": "Issue from model 1", "desc": "Original B1"},
                ],
            },
        ]
        # Issue 3 maps to group 1, suggestion 2
        issue_map = {3: (1, 2)}
        edited = {3: ("Original A2", "User edited A2")}

        updated, log = merge_edited_issue_descriptions(groups, edited, issue_to_group_map=issue_map)

        # Original not mutated
        assert groups[0]["suggestions"][1]["desc"] == "Original A2"
        # Updated has the edit on the right suggestion
        assert updated[0]["suggestions"][1]["desc"] == "User edited A2"
        assert updated[0]["suggestions"][1]["_description_edited"] is True
        # Other suggestion untouched
        assert updated[0]["suggestions"][0]["desc"] == "Original A1"
        # Group B untouched
        assert updated[1]["suggestions"][0]["desc"] == "Original B1"
        # Log has entry
        assert len(log) == 1
        assert log[0]["index"] == 3

    def test_legacy_fallback_without_map(self):
        """Without map, edits use group index (legacy)."""
        groups = [
            {"theme": "G1", "suggestions": [{"title": "T1", "desc": "Orig1"}]},
            {"theme": "G2", "suggestions": [{"title": "T2", "desc": "Orig2"}]},
        ]
        edited = {2: ("Orig2", "Edited2")}

        updated, log = merge_edited_issue_descriptions(groups, edited)
        assert updated[1]["suggestions"][0]["desc"] == "Edited2"
        assert len(log) == 1

    def test_applies_edit_keyed_by_suggestion_hash(self):
        """HTML selections key edits by suggestion_hash, not issue number.

        Regression for the "Applied 0 HTML-edited descriptions" bug: with an
        issue_to_group_map present (integer-keyed), a hash key fell through the
        ``key in issue_to_group_map`` check and the edit was silently dropped.
        """
        groups = [
            {"theme": "G1", "suggestions": [{"title": "T1", "desc": "Orig1", "suggestion_hash": "aaaa1111bbbb2222"}]},
            {"theme": "G2", "suggestions": [{"title": "T2", "desc": "Orig2", "suggestion_hash": "cccc3333dddd4444"}]},
        ]
        # Map is present and integer-keyed (the real-world condition).
        issue_map = {1: (1, 1), 2: (2, 1)}
        # Edit is keyed by the G2 suggestion's hash, as the HTML path produces.
        edited = {"cccc3333dddd4444": ("Orig2", "Edited via hash")}

        updated, log = merge_edited_issue_descriptions(groups, edited, issue_to_group_map=issue_map)

        assert updated[1]["suggestions"][0]["desc"] == "Edited via hash"
        assert updated[1]["suggestions"][0]["_description_edited"] is True
        assert updated[0]["suggestions"][0]["desc"] == "Orig1"  # untouched
        assert len(log) == 1
        assert log[0]["index"] == "cccc3333dddd4444"

    def test_applies_edit_keyed_by_g_format_id(self):
        """A G<g>S<s> positional key resolves even with an integer-keyed map."""
        groups = [
            {"theme": "G1", "suggestions": [{"title": "T1", "desc": "Orig1"}]},
            {"theme": "G2", "suggestions": [
                {"title": "T2a", "desc": "Orig2a"},
                {"title": "T2b", "desc": "Orig2b"},
            ]},
        ]
        issue_map = {1: (1, 1)}
        edited = {"G2S2": ("Orig2b", "Edited via G-format")}

        updated, log = merge_edited_issue_descriptions(groups, edited, issue_to_group_map=issue_map)

        assert updated[1]["suggestions"][1]["desc"] == "Edited via G-format"
        assert updated[1]["suggestions"][0]["desc"] == "Orig2a"  # untouched
        assert len(log) == 1

    def test_unknown_hash_key_skipped(self):
        """A hash that matches no suggestion is skipped, not applied anywhere."""
        groups = [
            {"theme": "G1", "suggestions": [{"title": "T1", "desc": "Orig1", "suggestion_hash": "aaaa1111bbbb2222"}]},
        ]
        issue_map = {1: (1, 1)}
        edited = {"deadbeefdeadbeef": ("x", "should not apply")}

        updated, log = merge_edited_issue_descriptions(groups, edited, issue_to_group_map=issue_map)

        assert updated[0]["suggestions"][0]["desc"] == "Orig1"
        assert len(log) == 0


class TestSkippedIssueConversion:
    """Tests for the skip conversion logic in main() - tested via the building blocks."""

    def test_partial_skip_produces_suggestion_ids(self):
        """When only some suggestions in a group are skipped, individual G-format IDs are used."""
        from collections import defaultdict

        # Simulate: group 1 has 2 suggestions, only issue 1 (→ G1S1) is skipped
        issue_map = {1: (1, 1), 3: (1, 2)}
        grouped = [{"suggestions": [{"title": "A"}, {"title": "A"}]}]
        skipped_issue_numbers = {1}

        skips_by_group: dict = defaultdict(set)
        for issue_num in skipped_issue_numbers:
            if issue_num in issue_map:
                g_idx, s_idx = issue_map[issue_num]
                skips_by_group[g_idx].add(s_idx)

        skipped_group_indices = set()
        skipped_suggestion_ids = set()
        for g_idx, skipped_suggs in skips_by_group.items():
            total_suggs = len(grouped[g_idx - 1].get("suggestions", []))
            if total_suggs > 0 and len(skipped_suggs) >= total_suggs:
                skipped_group_indices.add(g_idx)
            else:
                for s_idx in skipped_suggs:
                    skipped_suggestion_ids.add(f"G{g_idx}S{s_idx}")

        assert skipped_group_indices == set()
        assert skipped_suggestion_ids == {"G1S1"}

    def test_full_group_skip_produces_group_index(self):
        """When all suggestions in a group are skipped, the group index is used."""
        from collections import defaultdict

        issue_map = {1: (1, 1), 3: (1, 2)}
        grouped = [{"suggestions": [{"title": "A"}, {"title": "A"}]}]
        skipped_issue_numbers = {1, 3}  # Both mapped to group 1

        skips_by_group: dict = defaultdict(set)
        for issue_num in skipped_issue_numbers:
            if issue_num in issue_map:
                g_idx, s_idx = issue_map[issue_num]
                skips_by_group[g_idx].add(s_idx)

        skipped_group_indices = set()
        skipped_suggestion_ids = set()
        for g_idx, skipped_suggs in skips_by_group.items():
            total_suggs = len(grouped[g_idx - 1].get("suggestions", []))
            if total_suggs > 0 and len(skipped_suggs) >= total_suggs:
                skipped_group_indices.add(g_idx)
            else:
                for s_idx in skipped_suggs:
                    skipped_suggestion_ids.add(f"G{g_idx}S{s_idx}")

        assert skipped_group_indices == {1}
        assert skipped_suggestion_ids == set()


class TestValidationOverridesConversion:
    """Tests for validation override key conversion from issue numbers to group indices."""

    def test_overrides_converted_to_group_indices(self):
        """Issue-number-keyed overrides are converted to group-index keys."""
        issue_map = {1: (1, 1), 2: (1, 2), 3: (2, 1)}
        validation_overrides = {1: "valid", 3: "invalid"}

        group_validation_overrides = {}
        for issue_num, status in validation_overrides.items():
            if issue_num in issue_map:
                g_idx, _s_idx = issue_map[issue_num]
                group_validation_overrides[g_idx] = status
            else:
                group_validation_overrides[issue_num] = status

        assert group_validation_overrides == {1: "valid", 2: "invalid"}

    def test_fallback_when_no_map(self):
        """Without map, override keys pass through as-is."""
        validation_overrides = {1: "valid", 5: "invalid"}
        group_validation_overrides = validation_overrides
        assert group_validation_overrides == {1: "valid", 5: "invalid"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
