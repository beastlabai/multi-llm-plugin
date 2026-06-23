#!/usr/bin/env python3
"""
Tests for code_fix_batcher.py

Tests batching logic for grouping code review fixes into efficient subagent batches.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.code_fix_batcher import (
    CodeFixBatch,
    determine_subagent_type,
    get_line_start,
    is_high_risk_fix,
    batch_code_fixes,
    format_fix_batch_prompt,
    estimate_batch_processing_stats,
    MAX_FIXES_PER_BATCH,
    MAX_DESCRIPTION_CHARS,
)


def make_fix(
    file: str = "src/main.py",
    description: str = "Fix something",
    importance: str = "MEDIUM",
    fix_type: str = "bug",
    line_range: tuple = (10, 15),
    title: str = "Fix issue",
    anchor_text: str = "some code",
) -> dict:
    """Helper to create a test fix."""
    return {
        "file": file,
        "description": description,
        "importance": importance,
        "type": fix_type,
        "line_range": line_range,
        "title": title,
        "anchor_text": anchor_text,
    }


class TestCodeFixBatch:
    """Tests for CodeFixBatch dataclass."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        batch = CodeFixBatch()

        assert batch.fixes == []
        assert batch.file_key == ""
        assert batch.batch_type == "mixed"
        assert batch.subagent_type == "general-purpose"
        assert batch.total_chars == 0

    def test_initialization_with_values(self):
        """Test initialization with custom values."""
        batch = CodeFixBatch(
            fixes=[{"id": 1}],
            file_key="src/main.py",
            batch_type="bug",
            subagent_type="general-purpose",
            total_chars=100,
        )

        assert len(batch.fixes) == 1
        assert batch.file_key == "src/main.py"
        assert batch.batch_type == "bug"
        assert batch.subagent_type == "general-purpose"
        assert batch.total_chars == 100

    def test_size_property_empty(self):
        """Test size property with empty fixes list."""
        batch = CodeFixBatch()
        assert batch.size == 0

    def test_size_property_with_fixes(self):
        """Test size property with fixes."""
        batch = CodeFixBatch()
        batch.fixes = [{"id": 1}, {"id": 2}, {"id": 3}]
        assert batch.size == 3

    def test_fix_count_alias(self):
        """Test fix_count is an alias for size."""
        batch = CodeFixBatch()
        batch.fixes = [{"id": 1}, {"id": 2}]
        assert batch.fix_count == batch.size
        assert batch.fix_count == 2

    def test_is_full_when_empty(self):
        """Test is_full when batch is empty."""
        batch = CodeFixBatch()
        assert batch.is_full is False

    def test_is_full_at_limit(self):
        """Test is_full when batch reaches MAX_FIXES_PER_BATCH."""
        batch = CodeFixBatch()
        batch.fixes = [{"id": i} for i in range(MAX_FIXES_PER_BATCH)]
        assert batch.is_full is True

    def test_is_full_below_limit(self):
        """Test is_full when batch is below limit."""
        batch = CodeFixBatch()
        batch.fixes = [{"id": i} for i in range(MAX_FIXES_PER_BATCH - 1)]
        assert batch.is_full is False

    def test_priority_score_empty(self):
        """Test priority_score with no fixes."""
        batch = CodeFixBatch()
        assert batch.priority_score == 0.0

    def test_priority_score_high_importance(self):
        """Test priority_score with HIGH importance fix."""
        batch = CodeFixBatch()
        batch.fixes = [{"importance": "HIGH"}]
        assert batch.priority_score == 3.0

    def test_priority_score_medium_importance(self):
        """Test priority_score with MEDIUM importance fix."""
        batch = CodeFixBatch()
        batch.fixes = [{"importance": "MEDIUM"}]
        assert batch.priority_score == 2.0

    def test_priority_score_low_importance(self):
        """Test priority_score with LOW importance fix."""
        batch = CodeFixBatch()
        batch.fixes = [{"importance": "LOW"}]
        assert batch.priority_score == 1.0

    def test_priority_score_multiple_fixes(self):
        """Test priority_score with multiple fixes."""
        batch = CodeFixBatch()
        batch.fixes = [
            {"importance": "HIGH"},  # 3.0
            {"importance": "MEDIUM"},  # 2.0
            {"importance": "LOW"},  # 1.0
        ]
        assert batch.priority_score == 6.0

    def test_priority_score_missing_importance_defaults_medium(self):
        """Test priority_score defaults to MEDIUM weight for missing importance."""
        batch = CodeFixBatch()
        batch.fixes = [{}]  # No importance key
        assert batch.priority_score == 2.0

    def test_priority_score_case_insensitive(self):
        """Test priority_score handles lowercase importance."""
        batch = CodeFixBatch()
        batch.fixes = [{"importance": "high"}]
        assert batch.priority_score == 3.0

    def test_can_add_with_empty_batch(self):
        """Test can_add returns True for empty batch."""
        batch = CodeFixBatch()
        fix = make_fix(description="Short desc")
        assert batch.can_add(fix) is True

    def test_can_add_when_full(self):
        """Test can_add returns False when batch is full."""
        batch = CodeFixBatch()
        batch.fixes = [{"id": i} for i in range(MAX_FIXES_PER_BATCH)]
        fix = make_fix()
        assert batch.can_add(fix) is False

    def test_can_add_exceeds_char_limit(self):
        """Test can_add returns False when adding would exceed char limit."""
        batch = CodeFixBatch()
        batch.total_chars = MAX_DESCRIPTION_CHARS - 10  # Near limit
        fix = make_fix(description="x" * 100)  # Too long
        assert batch.can_add(fix) is False

    def test_can_add_within_char_limit(self):
        """Test can_add returns True when within char limit."""
        batch = CodeFixBatch()
        batch.total_chars = 100
        fix = make_fix(description="Short description")
        assert batch.can_add(fix) is True

    def test_can_add_invalid_fix_type(self):
        """Test can_add returns False for non-dict input."""
        batch = CodeFixBatch()
        assert batch.can_add("not a dict") is False
        assert batch.can_add(None) is False
        assert batch.can_add([]) is False

    def test_can_add_uses_desc_key(self):
        """Test can_add uses 'desc' key if 'description' not present."""
        batch = CodeFixBatch()
        batch.total_chars = MAX_DESCRIPTION_CHARS - 10
        fix = {"desc": "x" * 100}  # Too long via desc key
        assert batch.can_add(fix) is False

    def test_add_updates_fixes_list(self):
        """Test add() appends fix to fixes list."""
        batch = CodeFixBatch()
        fix = make_fix()
        batch.add(fix)
        assert len(batch.fixes) == 1
        assert batch.fixes[0] == fix

    def test_add_updates_total_chars(self):
        """Test add() updates total_chars based on description length."""
        batch = CodeFixBatch()
        fix = make_fix(description="Test description")  # 16 chars
        batch.add(fix)
        assert batch.total_chars == 16

    def test_add_uses_desc_key_for_chars(self):
        """Test add() uses 'desc' key if 'description' not present."""
        batch = CodeFixBatch()
        fix = {"desc": "Short desc"}  # 10 chars
        batch.add(fix)
        assert batch.total_chars == 10

    def test_add_updates_batch_type_single_type(self):
        """Test add() sets batch_type when all fixes have same type."""
        batch = CodeFixBatch()
        batch.add(make_fix(fix_type="bug"))
        assert batch.batch_type == "bug"

        batch.add(make_fix(fix_type="bug"))
        assert batch.batch_type == "bug"

    def test_add_updates_batch_type_mixed(self):
        """Test add() sets batch_type to 'mixed' with different types."""
        batch = CodeFixBatch()
        batch.add(make_fix(fix_type="bug"))
        assert batch.batch_type == "bug"

        batch.add(make_fix(fix_type="improvement"))
        assert batch.batch_type == "mixed"

    def test_to_dict_serialization(self):
        """Test to_dict() returns correct dictionary structure."""
        batch = CodeFixBatch(
            file_key="src/main.py",
            batch_type="bug",
            subagent_type="general-purpose",
        )
        fix = make_fix(importance="HIGH")
        batch.add(fix)

        result = batch.to_dict()

        assert "fixes" in result
        assert "file_key" in result
        assert "batch_type" in result
        assert "subagent_type" in result
        assert "fix_count" in result
        assert "total_chars" in result
        assert "priority_score" in result

        assert result["file_key"] == "src/main.py"
        assert result["batch_type"] == "bug"
        assert result["subagent_type"] == "general-purpose"
        assert result["fix_count"] == 1
        assert result["priority_score"] == 3.0

    def test_to_dict_contains_fixes(self):
        """Test to_dict() includes the fixes list."""
        batch = CodeFixBatch()
        fix1 = make_fix(title="Fix 1")
        fix2 = make_fix(title="Fix 2")
        batch.add(fix1)
        batch.add(fix2)

        result = batch.to_dict()
        assert len(result["fixes"]) == 2


class TestDetermineSubagentType:
    """Tests for determine_subagent_type function."""

    def test_always_returns_general_purpose(self):
        """All fixes route to general-purpose (only available subagent type)."""
        test_cases = [
            {"file": "supabase/schemas/users.sql", "description": "Fix RLS policy"},
            {"file": "src/components/Form.tsx", "description": "Fix react-hook-form validation"},
            {"file": "tests/e2e/login.spec.ts", "description": "Fix playwright locator"},
            {"file": "src/api/handler.ts", "description": "Fix null check"},
        ]
        for fix in test_cases:
            assert determine_subagent_type(fix) == "general-purpose"

    def test_empty_fix(self):
        """Empty fix returns general-purpose."""
        assert determine_subagent_type({}) == "general-purpose"


class TestGetLineStart:
    """Tests for get_line_start function."""

    def test_line_range_tuple(self):
        """Test extraction from line_range tuple."""
        fix = make_fix(line_range=(42, 50))
        assert get_line_start(fix) == 42

    def test_line_range_list(self):
        """Test extraction from line_range list."""
        fix = {"line_range": [100, 110]}
        assert get_line_start(fix) == 100

    def test_line_range_single_element(self):
        """Test extraction from single element line_range."""
        fix = {"line_range": [5]}
        assert get_line_start(fix) == 5

    def test_missing_line_range(self):
        """Test returns 0 when line_range is missing."""
        fix = {"file": "test.py"}
        assert get_line_start(fix) == 0

    def test_none_line_range(self):
        """Test returns 0 when line_range is None."""
        fix = {"line_range": None}
        assert get_line_start(fix) == 0

    def test_empty_line_range(self):
        """Test returns 0 when line_range is empty."""
        fix = {"line_range": []}
        assert get_line_start(fix) == 0

    def test_string_line_number_converted(self):
        """Test string line numbers are converted to int."""
        fix = {"line_range": ["25", "30"]}
        assert get_line_start(fix) == 25


class TestIsHighRiskFix:
    """Tests for is_high_risk_fix function."""

    def test_security_type_returns_true(self):
        """Test security type fix returns True."""
        fix = make_fix(fix_type="security", importance="LOW")
        assert is_high_risk_fix(fix) is True

    def test_security_type_case_insensitive(self):
        """Test security type is case insensitive."""
        fix = make_fix(fix_type="SECURITY", importance="LOW")
        assert is_high_risk_fix(fix) is True

    def test_high_importance_returns_true(self):
        """Test HIGH importance returns True."""
        fix = make_fix(fix_type="bug", importance="HIGH")
        assert is_high_risk_fix(fix) is True

    def test_high_importance_case_insensitive(self):
        """Test HIGH importance is case insensitive."""
        fix = make_fix(fix_type="bug", importance="high")
        assert is_high_risk_fix(fix) is True

    def test_medium_importance_returns_false(self):
        """Test MEDIUM importance returns False."""
        fix = make_fix(fix_type="bug", importance="MEDIUM")
        assert is_high_risk_fix(fix) is False

    def test_low_importance_returns_false(self):
        """Test LOW importance returns False."""
        fix = make_fix(fix_type="bug", importance="LOW")
        assert is_high_risk_fix(fix) is False

    def test_missing_type_not_high_risk(self):
        """Test missing type is not high risk."""
        fix = {"importance": "MEDIUM"}
        assert is_high_risk_fix(fix) is False

    def test_missing_importance_not_high_risk(self):
        """Test missing importance is not high risk."""
        fix = {"type": "bug"}
        assert is_high_risk_fix(fix) is False

    def test_none_values_handled(self):
        """Test None values don't cause errors."""
        fix = {"type": None, "importance": None}
        assert is_high_risk_fix(fix) is False


class TestBatchCodeFixes:
    """Tests for batch_code_fixes function."""

    def test_empty_list_returns_empty(self):
        """Test empty input returns empty list."""
        result = batch_code_fixes([])
        assert result == []

    def test_single_fix(self):
        """Test single fix creates one batch."""
        fixes = [make_fix()]
        result = batch_code_fixes(fixes)

        assert len(result) == 1
        assert result[0].size == 1

    def test_single_high_risk_fix(self):
        """Test single high-risk fix is isolated."""
        fixes = [make_fix(importance="HIGH")]
        result = batch_code_fixes(fixes)

        assert len(result) == 1
        assert result[0].size == 1

    def test_multiple_fixes_same_file_grouped(self):
        """Test multiple fixes in same file are grouped."""
        fixes = [
            make_fix(file="src/main.py", line_range=(10, 15), importance="MEDIUM"),
            make_fix(file="src/main.py", line_range=(20, 25), importance="MEDIUM"),
            make_fix(file="src/main.py", line_range=(30, 35), importance="MEDIUM"),
        ]
        result = batch_code_fixes(fixes)

        assert len(result) == 1
        assert result[0].size == 3
        assert result[0].file_key == "src/main.py"

    def test_fixes_different_files_separate_batches(self):
        """Test fixes in different files get separate batches."""
        fixes = [
            make_fix(file="src/main.py", importance="MEDIUM"),
            make_fix(file="src/utils.py", importance="MEDIUM"),
        ]
        result = batch_code_fixes(fixes)

        assert len(result) == 2
        file_keys = {b.file_key for b in result}
        assert file_keys == {"src/main.py", "src/utils.py"}

    def test_high_risk_fixes_isolated(self):
        """Test high-risk fixes are in their own batches."""
        fixes = [
            make_fix(file="src/main.py", importance="HIGH", description="High 1"),
            make_fix(file="src/main.py", importance="HIGH", description="High 2"),
            make_fix(file="src/main.py", importance="MEDIUM", description="Medium"),
        ]
        result = batch_code_fixes(fixes)

        # 2 HIGH fixes (isolated) + 1 MEDIUM fix
        assert len(result) == 3
        high_batches = [b for b in result if b.priority_score >= 3.0 and b.size == 1]
        assert len(high_batches) == 2

    def test_security_fixes_isolated(self):
        """Test security fixes are isolated."""
        fixes = [
            make_fix(file="src/main.py", fix_type="security", importance="LOW"),
            make_fix(file="src/main.py", fix_type="bug", importance="MEDIUM"),
        ]
        result = batch_code_fixes(fixes)

        assert len(result) == 2

    def test_respects_max_per_batch(self):
        """Test MAX_FIXES_PER_BATCH limit is respected.

        Note: The max_per_batch parameter is accepted but the function
        uses the module constant MAX_FIXES_PER_BATCH (3) via can_add().
        """
        # Create more fixes than MAX_FIXES_PER_BATCH (which is 3)
        fixes = [
            make_fix(file="src/main.py", line_range=(i * 10, i * 10 + 5), importance="MEDIUM")
            for i in range(7)
        ]
        result = batch_code_fixes(fixes)

        # 7 fixes with max 3 per batch = 3 batches (3 + 3 + 1)
        assert len(result) == 3
        assert result[0].size == 3  # First batch full
        assert result[1].size == 3  # Second batch full
        assert result[2].size == 1  # Third batch has remainder

    def test_respects_max_chars(self):
        """Test MAX_DESCRIPTION_CHARS limit is respected.

        Note: The max_chars parameter is accepted but the function
        uses the module constant MAX_DESCRIPTION_CHARS (3000) via can_add().
        """
        # Create fixes with descriptions that exceed the default 3000 char limit when combined
        fixes = [
            make_fix(
                file="src/main.py",
                line_range=(i * 10, i * 10 + 5),
                importance="MEDIUM",
                description="x" * 1500,  # 1500 chars each, 2 fits under 3000
            )
            for i in range(4)
        ]
        result = batch_code_fixes(fixes)

        # 4 fixes: first 2 fit (3000 chars), next 2 fit in second batch
        assert len(result) == 2
        for batch in result:
            assert batch.size == 2  # 2 fixes per batch (1500 + 1500 = 3000)

    def test_batches_sorted_by_priority(self):
        """Test batches are sorted by priority score (highest first)."""
        fixes = [
            make_fix(file="file1.py", importance="LOW"),
            make_fix(file="file2.py", importance="HIGH"),
            make_fix(file="file3.py", importance="MEDIUM"),
        ]
        result = batch_code_fixes(fixes)

        # HIGH should be first
        assert result[0].priority_score >= result[-1].priority_score

    def test_line_ordering_within_file(self):
        """Test fixes within a file are ordered by line number."""
        fixes = [
            make_fix(file="src/main.py", line_range=(50, 55), importance="MEDIUM"),
            make_fix(file="src/main.py", line_range=(10, 15), importance="MEDIUM"),
            make_fix(file="src/main.py", line_range=(30, 35), importance="MEDIUM"),
        ]
        result = batch_code_fixes(fixes)

        assert len(result) == 1
        batch_lines = [get_line_start(f) for f in result[0].fixes]
        assert batch_lines == [10, 30, 50]  # Sorted order

    def test_subagent_type_assigned(self):
        """Test subagent type is assigned based on first fix."""
        fixes = [
            make_fix(file="supabase/migrations/001.sql", importance="MEDIUM"),
        ]
        result = batch_code_fixes(fixes)

        assert result[0].subagent_type == "general-purpose"

    def test_high_risk_only_batch(self):
        """Test batch containing only high-risk fixes."""
        fixes = [
            make_fix(importance="HIGH", description="High 1"),
            make_fix(importance="HIGH", description="High 2"),
        ]
        result = batch_code_fixes(fixes)

        # Each HIGH fix should be isolated
        assert len(result) == 2
        for batch in result:
            assert batch.size == 1

    def test_mixed_file_and_importance(self):
        """Test complex scenario with mixed files and importance."""
        fixes = [
            make_fix(file="src/a.py", importance="HIGH"),
            make_fix(file="src/a.py", importance="MEDIUM"),
            make_fix(file="src/b.py", importance="MEDIUM"),
            make_fix(file="src/b.py", importance="LOW"),
        ]
        result = batch_code_fixes(fixes)

        # 1 HIGH (isolated) + 1 batch for a.py MEDIUM + 1 batch for b.py
        assert len(result) >= 3


class TestFormatFixBatchPrompt:
    """Tests for format_fix_batch_prompt function."""

    def test_single_fix_format(self):
        """Test prompt format for single fix."""
        batch = CodeFixBatch(file_key="src/main.py")
        fix = make_fix(
            file="src/main.py",
            title="Missing null check",
            fix_type="bug",
            importance="HIGH",
            line_range=(42, 50),
            description="Add null check before accessing property",
            anchor_text="user.name",
        )
        batch.add(fix)

        prompt = format_fix_batch_prompt(batch, "plans/feature.md", "HEAD~1")

        assert "**Plan file**: plans/feature.md" in prompt
        assert "**File**: src/main.py" in prompt
        assert "**Lines**: 42-50" in prompt
        assert "**Issue**: Missing null check" in prompt
        assert "**Type**: bug" in prompt
        assert "**Importance**: HIGH" in prompt
        assert "Add null check before accessing property" in prompt
        assert "`user.name`" in prompt
        assert "git diff HEAD~1 -- src/main.py" in prompt

    def test_multiple_fixes_format(self):
        """Test prompt format for multiple fixes."""
        batch = CodeFixBatch(file_key="src/main.py", batch_type="bug")
        batch.add(make_fix(title="Fix 1", line_range=(10, 15)))
        batch.add(make_fix(title="Fix 2", line_range=(20, 25)))
        batch.add(make_fix(title="Fix 3", line_range=(30, 35)))

        prompt = format_fix_batch_prompt(batch, "plans/feature.md")

        assert "Fix the following 3 related issues" in prompt
        assert "**Primary file**: src/main.py" in prompt
        assert "**Batch type**: bug" in prompt
        assert "**Fixes in this batch**: 3" in prompt
        assert "### Fix 1:" in prompt
        assert "### Fix 2:" in prompt
        assert "### Fix 3:" in prompt
        assert "Apply ALL fixes in this batch" in prompt

    def test_empty_batch_raises_value_error(self):
        """Test empty batch raises ValueError."""
        batch = CodeFixBatch()

        with pytest.raises(ValueError, match="Cannot format prompt for empty batch"):
            format_fix_batch_prompt(batch, "plans/feature.md")

    def test_default_base_ref(self):
        """Test default base_ref is HEAD~1."""
        batch = CodeFixBatch(file_key="src/main.py")
        batch.add(make_fix())

        prompt = format_fix_batch_prompt(batch, "plans/feature.md")
        assert "git diff HEAD~1" in prompt

    def test_custom_base_ref(self):
        """Test custom base_ref is used."""
        batch = CodeFixBatch(file_key="src/main.py")
        batch.add(make_fix())

        prompt = format_fix_batch_prompt(batch, "plans/feature.md", base_ref="main")
        assert "git diff main" in prompt

    def test_missing_line_range_shows_unknown(self):
        """Test missing line_range shows 'unknown'."""
        batch = CodeFixBatch(file_key="src/main.py")
        fix = make_fix()
        del fix["line_range"]
        batch.add(fix)

        prompt = format_fix_batch_prompt(batch, "plans/feature.md")
        assert "**Lines**: unknown" in prompt

    def test_single_element_line_range(self):
        """Test single element line_range is handled."""
        batch = CodeFixBatch(file_key="src/main.py")
        fix = make_fix()
        fix["line_range"] = [42]
        batch.add(fix)

        prompt = format_fix_batch_prompt(batch, "plans/feature.md")
        assert "**Lines**: unknown" in prompt  # Needs at least 2 elements

    def test_uses_desc_key_fallback(self):
        """Test prompt uses 'desc' key if 'description' not present."""
        batch = CodeFixBatch(file_key="src/main.py")
        fix = {
            "file": "src/main.py",
            "desc": "Use desc key description",
            "title": "Test",
            "line_range": [1, 5],
            "anchor_text": "code",
        }
        batch.add(fix)

        prompt = format_fix_batch_prompt(batch, "plans/feature.md")
        assert "Use desc key description" in prompt

    def test_single_fix_includes_prior_changes_placeholder(self):
        """Single fix format includes {prior_changes_context} placeholder."""
        batch = CodeFixBatch(file_key="src/main.py")
        batch.add(make_fix())

        prompt = format_fix_batch_prompt(batch, "plans/feature.md")

        assert "{prior_changes_context}" in prompt
        assert "## Changes Applied in Prior Batches" in prompt

    def test_multiple_fixes_includes_prior_changes_placeholder(self):
        """Multiple fix format includes {prior_changes_context} placeholder."""
        batch = CodeFixBatch(file_key="src/main.py", batch_type="bug")
        batch.add(make_fix(title="Fix 1", line_range=(10, 15)))
        batch.add(make_fix(title="Fix 2", line_range=(20, 25)))

        prompt = format_fix_batch_prompt(batch, "plans/feature.md")

        assert "{prior_changes_context}" in prompt
        assert "## Changes Applied in Prior Batches" in prompt


class TestEstimateBatchProcessingStats:
    """Tests for estimate_batch_processing_stats function."""

    def test_empty_batches(self):
        """Test empty batches return zero stats."""
        stats = estimate_batch_processing_stats([])

        assert stats["total_fixes"] == 0
        assert stats["total_batches"] == 0
        assert stats["subagent_calls_saved"] == 0
        assert stats["efficiency_gain_percent"] == 0
        assert stats["average_batch_size"] == 0
        assert stats["max_batch_size"] == 0
        assert stats["single_fix_batches"] == 0
        assert stats["multi_fix_batches"] == 0

    def test_single_batch_single_fix(self):
        """Test stats with one batch, one fix."""
        batch = CodeFixBatch()
        batch.add(make_fix())
        stats = estimate_batch_processing_stats([batch])

        assert stats["total_fixes"] == 1
        assert stats["total_batches"] == 1
        assert stats["subagent_calls_saved"] == 0
        assert stats["efficiency_gain_percent"] == 0.0
        assert stats["single_fix_batches"] == 1
        assert stats["multi_fix_batches"] == 0

    def test_efficiency_gain_calculation(self):
        """Test efficiency_gain_percent is calculated correctly."""
        # 3 fixes in 1 batch = 2 calls saved = 66.7% efficiency
        batch = CodeFixBatch()
        batch.add(make_fix())
        batch.add(make_fix())
        batch.add(make_fix())
        stats = estimate_batch_processing_stats([batch])

        assert stats["total_fixes"] == 3
        assert stats["total_batches"] == 1
        assert stats["subagent_calls_saved"] == 2
        assert stats["efficiency_gain_percent"] == 66.7

    def test_counts_single_vs_multi_fix_batches(self):
        """Test single and multi-fix batch counts."""
        batch1 = CodeFixBatch()
        batch1.add(make_fix())

        batch2 = CodeFixBatch()
        batch2.add(make_fix())
        batch2.add(make_fix())

        batch3 = CodeFixBatch()
        batch3.add(make_fix())
        batch3.add(make_fix())
        batch3.add(make_fix())

        stats = estimate_batch_processing_stats([batch1, batch2, batch3])

        assert stats["single_fix_batches"] == 1
        assert stats["multi_fix_batches"] == 2

    def test_average_batch_size(self):
        """Test average batch size calculation."""
        batch1 = CodeFixBatch()
        batch1.add(make_fix())  # size 1

        batch2 = CodeFixBatch()
        batch2.add(make_fix())
        batch2.add(make_fix())  # size 2

        batch3 = CodeFixBatch()
        batch3.add(make_fix())
        batch3.add(make_fix())
        batch3.add(make_fix())  # size 3

        stats = estimate_batch_processing_stats([batch1, batch2, batch3])

        # Average: (1 + 2 + 3) / 3 = 2.0
        assert stats["average_batch_size"] == 2.0

    def test_max_batch_size(self):
        """Test max batch size is tracked."""
        batch1 = CodeFixBatch()
        batch1.add(make_fix())

        batch2 = CodeFixBatch()
        for _ in range(3):
            batch2.add(make_fix())

        stats = estimate_batch_processing_stats([batch1, batch2])

        assert stats["max_batch_size"] == 3

    def test_batch_type_distribution(self):
        """Test batch type distribution tracking.

        Note: The add() method updates batch_type based on fix types,
        so we need to set fix types to match desired batch_type.
        """
        batch1 = CodeFixBatch(batch_type="bug")
        batch1.add(make_fix(fix_type="bug"))

        batch2 = CodeFixBatch(batch_type="bug")
        batch2.add(make_fix(fix_type="bug"))

        batch3 = CodeFixBatch(batch_type="security")
        batch3.add(make_fix(fix_type="security"))

        stats = estimate_batch_processing_stats([batch1, batch2, batch3])

        assert stats["batch_type_distribution"] == {"bug": 2, "security": 1}

    def test_subagent_distribution(self):
        """Test subagent distribution tracking."""
        batch1 = CodeFixBatch(subagent_type="general-purpose")
        batch1.add(make_fix())

        batch2 = CodeFixBatch(subagent_type="general-purpose")
        batch2.add(make_fix())

        batch3 = CodeFixBatch(subagent_type="general-purpose")
        batch3.add(make_fix())

        stats = estimate_batch_processing_stats([batch1, batch2, batch3])

        assert stats["subagent_distribution"]["general-purpose"] == 3


class TestConstants:
    """Tests for module constants."""

    def test_max_fixes_per_batch(self):
        """Test MAX_FIXES_PER_BATCH value."""
        assert MAX_FIXES_PER_BATCH == 3

    def test_max_description_chars(self):
        """Test MAX_DESCRIPTION_CHARS value."""
        assert MAX_DESCRIPTION_CHARS == 3000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
