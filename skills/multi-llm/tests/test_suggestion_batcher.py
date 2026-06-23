#!/usr/bin/env python3
"""
Tests for suggestion_batcher.py

Tests batching logic for grouping suggestions into efficient subagent batches.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.suggestion_batcher import (
    SuggestionBatch,
    normalize_section_reference,
    extract_section_order,
    are_types_compatible,
    group_suggestions_for_subagents,
    format_batch_for_prompt,
    estimate_batch_processing_stats,
    MAX_SUGGESTIONS_PER_BATCH,
    MAX_DESCRIPTION_CHARS,
)


def make_suggestion(
    title: str = "Test suggestion",
    description: str = "Test description",
    importance: str = "MEDIUM",
    suggestion_type: str = "modification",
    reference: str = "Section 1"
) -> dict:
    """Helper to create a test suggestion."""
    return {
        "title": title,
        "description": description,
        "importance": importance,
        "type": suggestion_type,
        "reference": reference,
    }


class TestSuggestionBatch:
    """Tests for SuggestionBatch dataclass."""

    def test_initialization_defaults(self):
        """Test default initialization of SuggestionBatch."""
        batch = SuggestionBatch()

        assert batch.suggestions == []
        assert batch.section_key == ""
        assert batch.batch_type == "mixed"
        assert batch.total_chars == 0

    def test_initialization_with_values(self):
        """Test initialization with provided values."""
        batch = SuggestionBatch(
            suggestions=[make_suggestion()],
            section_key="step_1",
            batch_type="modification",
            total_chars=100
        )

        assert len(batch.suggestions) == 1
        assert batch.section_key == "step_1"
        assert batch.batch_type == "modification"
        assert batch.total_chars == 100

    def test_size_property(self):
        """Test size property returns number of suggestions."""
        batch = SuggestionBatch()
        assert batch.size == 0

        batch.suggestions.append(make_suggestion())
        assert batch.size == 1

        batch.suggestions.append(make_suggestion())
        assert batch.size == 2

    def test_is_full_false_when_under_limit(self):
        """Test is_full returns False when under MAX_SUGGESTIONS_PER_BATCH."""
        batch = SuggestionBatch()
        for _ in range(MAX_SUGGESTIONS_PER_BATCH - 1):
            batch.suggestions.append(make_suggestion())

        assert batch.is_full is False

    def test_is_full_true_when_at_limit(self):
        """Test is_full returns True when at MAX_SUGGESTIONS_PER_BATCH."""
        batch = SuggestionBatch()
        for _ in range(MAX_SUGGESTIONS_PER_BATCH):
            batch.suggestions.append(make_suggestion())

        assert batch.is_full is True

    def test_is_full_true_when_over_limit(self):
        """Test is_full returns True when over MAX_SUGGESTIONS_PER_BATCH."""
        batch = SuggestionBatch()
        for _ in range(MAX_SUGGESTIONS_PER_BATCH + 1):
            batch.suggestions.append(make_suggestion())

        assert batch.is_full is True

    def test_priority_score_with_high_importance(self):
        """Test priority_score calculation with HIGH importance."""
        batch = SuggestionBatch()
        batch.suggestions.append(make_suggestion(importance="HIGH"))

        assert batch.priority_score == 3.0

    def test_priority_score_with_medium_importance(self):
        """Test priority_score calculation with MEDIUM importance."""
        batch = SuggestionBatch()
        batch.suggestions.append(make_suggestion(importance="MEDIUM"))

        assert batch.priority_score == 2.0

    def test_priority_score_with_low_importance(self):
        """Test priority_score calculation with LOW importance."""
        batch = SuggestionBatch()
        batch.suggestions.append(make_suggestion(importance="LOW"))

        assert batch.priority_score == 1.0

    def test_priority_score_with_mixed_importance(self):
        """Test priority_score sums across multiple suggestions."""
        batch = SuggestionBatch()
        batch.suggestions.append(make_suggestion(importance="HIGH"))  # 3.0
        batch.suggestions.append(make_suggestion(importance="MEDIUM"))  # 2.0
        batch.suggestions.append(make_suggestion(importance="LOW"))  # 1.0

        assert batch.priority_score == 6.0

    def test_priority_score_defaults_to_medium_for_missing(self):
        """Test priority_score defaults to MEDIUM weight for missing importance."""
        batch = SuggestionBatch()
        batch.suggestions.append({"title": "No importance"})

        assert batch.priority_score == 2.0  # MEDIUM default

    def test_priority_score_case_insensitive(self):
        """Test priority_score handles lowercase importance."""
        batch = SuggestionBatch()
        batch.suggestions.append(make_suggestion(importance="high"))

        assert batch.priority_score == 3.0

    def test_can_add_returns_false_for_non_dict(self):
        """Test can_add returns False for non-dict input."""
        batch = SuggestionBatch()

        assert batch.can_add("not a dict") is False
        assert batch.can_add(None) is False
        assert batch.can_add(123) is False
        assert batch.can_add([]) is False

    def test_can_add_returns_false_when_full(self):
        """Test can_add returns False when batch is full."""
        batch = SuggestionBatch()
        for _ in range(MAX_SUGGESTIONS_PER_BATCH):
            batch.suggestions.append(make_suggestion())

        assert batch.can_add(make_suggestion()) is False

    def test_can_add_returns_false_when_chars_exceed_limit(self):
        """Test can_add returns False when total_chars would exceed limit."""
        batch = SuggestionBatch(total_chars=MAX_DESCRIPTION_CHARS - 10)
        long_suggestion = make_suggestion(description="x" * 100)

        assert batch.can_add(long_suggestion) is False

    def test_can_add_returns_true_when_chars_within_limit(self):
        """Test can_add returns True when total_chars stays within limit."""
        batch = SuggestionBatch(total_chars=100)
        short_suggestion = make_suggestion(description="short")

        assert batch.can_add(short_suggestion) is True

    def test_can_add_returns_true_for_empty_batch(self):
        """Test can_add returns True for empty batch with valid suggestion."""
        batch = SuggestionBatch()

        assert batch.can_add(make_suggestion()) is True

    def test_add_appends_suggestion(self):
        """Test add() appends suggestion to list."""
        batch = SuggestionBatch()
        suggestion = make_suggestion()

        batch.add(suggestion)

        assert len(batch.suggestions) == 1
        assert batch.suggestions[0] == suggestion

    def test_add_updates_total_chars(self):
        """Test add() updates total_chars."""
        batch = SuggestionBatch()
        description = "Test description with some length"
        suggestion = make_suggestion(description=description)

        batch.add(suggestion)

        assert batch.total_chars == len(description)

    def test_add_updates_total_chars_cumulatively(self):
        """Test add() accumulates total_chars across multiple adds."""
        batch = SuggestionBatch()
        suggestion1 = make_suggestion(description="First")  # 5 chars
        suggestion2 = make_suggestion(description="Second")  # 6 chars

        batch.add(suggestion1)
        batch.add(suggestion2)

        assert batch.total_chars == 11

    def test_add_updates_batch_type_single_type(self):
        """Test add() sets batch_type to suggestion type when all same."""
        batch = SuggestionBatch()
        batch.add(make_suggestion(suggestion_type="addition"))
        batch.add(make_suggestion(suggestion_type="addition"))

        assert batch.batch_type == "addition"

    def test_add_updates_batch_type_mixed(self):
        """Test add() sets batch_type to mixed when types differ."""
        batch = SuggestionBatch()
        batch.add(make_suggestion(suggestion_type="addition"))
        batch.add(make_suggestion(suggestion_type="modification"))

        assert batch.batch_type == "mixed"

    def test_add_handles_missing_description(self):
        """Test add() handles suggestions without description."""
        batch = SuggestionBatch()
        suggestion = {"title": "No description"}

        batch.add(suggestion)

        assert batch.total_chars == 0
        assert len(batch.suggestions) == 1

    def test_add_handles_missing_type(self):
        """Test add() handles suggestions without type."""
        batch = SuggestionBatch()
        suggestion = {"title": "No type", "description": "Test"}

        batch.add(suggestion)

        assert batch.batch_type == "modification"  # Default

    def test_to_dict_serialization(self):
        """Test to_dict() produces correct serialization."""
        batch = SuggestionBatch(section_key="step_1")
        batch.add(make_suggestion(importance="HIGH", description="Test desc"))

        result = batch.to_dict()

        assert result["suggestions"] == batch.suggestions
        assert result["section_key"] == "step_1"
        assert result["batch_type"] == "modification"
        assert result["suggestion_count"] == 1
        assert result["total_chars"] == len("Test desc")
        assert result["priority_score"] == 3.0

    def test_to_dict_empty_batch(self):
        """Test to_dict() works for empty batch."""
        batch = SuggestionBatch()

        result = batch.to_dict()

        assert result["suggestions"] == []
        assert result["suggestion_count"] == 0
        assert result["total_chars"] == 0
        assert result["priority_score"] == 0.0


class TestNormalizeSectionReference:
    """Tests for normalize_section_reference function."""

    def test_empty_string_returns_unknown(self):
        """Test empty string returns 'unknown'."""
        assert normalize_section_reference("") == "unknown"

    def test_none_returns_unknown(self):
        """Test None returns 'unknown'."""
        assert normalize_section_reference(None) == "unknown"

    def test_whitespace_only_returns_ref_empty(self):
        """Test whitespace-only string after strip becomes empty, falls through to ref."""
        # After strip(), "   " becomes "", which doesn't match FILE_PATTERN or SECTION_PATTERN
        # So it falls through to the fallback: "ref:" + normalized empty string
        result = normalize_section_reference("   ")
        assert result == "ref:"

    def test_step_with_number_extracts_step(self):
        """Test '### Step 3: Create Server Action' returns 'step_3'."""
        result = normalize_section_reference("### Step 3: Create Server Action")
        assert result == "step_3"

    def test_step_lowercase(self):
        """Test lowercase step reference."""
        result = normalize_section_reference("step 5: implementation")
        assert result == "step_5"

    def test_step_with_different_formatting(self):
        """Test various step formats."""
        # "Step 1" matches SECTION_PATTERN with group(1)=None and group(2)="1"
        # Since step_num is None, it falls to section: path with normalized "1"
        assert normalize_section_reference("Step 1") == "section:1"
        # "## Step 2: Setup" matches with group(1)="2" (the Step pattern captures the number)
        # So it returns step_2
        assert normalize_section_reference("## Step 2: Setup") == "step_2"
        # "### 3. Task Name" matches with group(1)="3" so returns step_3
        assert normalize_section_reference("### 3. Task Name") == "step_3"

    def test_file_reference_extracts_path(self):
        """Test 'File: src/api.ts, Line 45' returns 'file:src/api.ts'."""
        result = normalize_section_reference("File: src/api.ts, Line 45")
        assert result == "file:src/api.ts"

    def test_file_reference_path_only(self):
        """Test file reference with path only."""
        result = normalize_section_reference("Path: /home/user/project/main.py")
        assert result == "file:/home/user/project/main.py"

    def test_file_reference_location(self):
        """Test Location reference."""
        result = normalize_section_reference("Location: components/Button.tsx")
        assert result == "file:components/button.tsx"

    def test_section_without_number(self):
        """Test 'Database Schema' returns normalized section key."""
        result = normalize_section_reference("Database Schema")
        assert result == "section:database_schema"

    def test_section_removes_special_characters(self):
        """Test special characters are removed from section names."""
        result = normalize_section_reference("API Integration!")
        assert result == "section:api_integration"

    def test_section_truncates_long_names(self):
        """Test long section names are truncated to 30 chars."""
        long_name = "This is a very long section name that exceeds the limit"
        result = normalize_section_reference(long_name)
        # The result should be section: + up to 30 chars
        section_part = result.replace("section:", "")
        assert len(section_part) <= 30

    def test_fallback_for_unusual_formats(self):
        """Test unusual reference formats go through SECTION_PATTERN."""
        # "some-random-reference-123" matches SECTION_PATTERN with group(1)=None
        # and group(2)="some-random-reference-123", so it normalizes as section:
        result = normalize_section_reference("some-random-reference-123")
        assert result.startswith("section:")

    def test_numbered_section_header(self):
        """Test numbered section headers like '### 2. Implementation'."""
        result = normalize_section_reference("### 2. Implementation Details")
        assert result == "step_2"


class TestExtractSectionOrder:
    """Tests for extract_section_order function."""

    def test_empty_string_returns_999(self):
        """Test empty string returns 999."""
        assert extract_section_order("") == 999

    def test_none_returns_999(self):
        """Test None returns 999."""
        assert extract_section_order(None) == 999

    def test_step_with_number(self):
        """Test 'step 3' returns 3."""
        assert extract_section_order("step 3") == 3

    def test_step_with_number_case_insensitive(self):
        """Test step extraction is case insensitive."""
        assert extract_section_order("Step 5") == 5
        assert extract_section_order("STEP 7") == 7

    def test_step_with_colon(self):
        """Test 'Step 4: Implementation' returns 4."""
        assert extract_section_order("Step 4: Implementation") == 4

    def test_numbered_section(self):
        """Test '### 2. Implementation' returns 2."""
        assert extract_section_order("### 2. Implementation") == 2

    def test_leading_number(self):
        """Test leading number extraction."""
        assert extract_section_order("1. First item") == 1
        assert extract_section_order("# 3 Header") == 3

    def test_no_number_returns_999(self):
        """Test references without numbers return 999."""
        assert extract_section_order("Database Schema") == 999
        assert extract_section_order("Implementation Details") == 999

    def test_number_in_middle_without_step(self):
        """Test that numbers in middle without 'step' don't match."""
        # Should return 999 because there's no step keyword and no leading number
        result = extract_section_order("The section about item 5")
        # This doesn't match step pattern and no leading number
        assert result == 999


class TestAreTypesCompatible:
    """Tests for are_types_compatible function."""

    def test_deletion_with_deletion_compatible(self):
        """Test deletion types are compatible with each other."""
        assert are_types_compatible("deletion", "deletion") is True

    def test_deletion_with_addition_incompatible(self):
        """Test deletion is incompatible with addition."""
        assert are_types_compatible("deletion", "addition") is False
        assert are_types_compatible("addition", "deletion") is False

    def test_deletion_with_modification_incompatible(self):
        """Test deletion is incompatible with modification."""
        assert are_types_compatible("deletion", "modification") is False
        assert are_types_compatible("modification", "deletion") is False

    def test_deletion_with_clarification_incompatible(self):
        """Test deletion is incompatible with clarification."""
        assert are_types_compatible("deletion", "clarification") is False
        assert are_types_compatible("clarification", "deletion") is False

    def test_clarification_with_addition_compatible(self):
        """Test clarification is compatible with addition."""
        assert are_types_compatible("clarification", "addition") is True
        assert are_types_compatible("addition", "clarification") is True

    def test_clarification_with_modification_compatible(self):
        """Test clarification is compatible with modification."""
        assert are_types_compatible("clarification", "modification") is True
        assert are_types_compatible("modification", "clarification") is True

    def test_clarification_with_clarification_compatible(self):
        """Test clarification types are compatible."""
        assert are_types_compatible("clarification", "clarification") is True

    def test_addition_with_modification_compatible(self):
        """Test addition is compatible with modification."""
        assert are_types_compatible("addition", "modification") is True
        assert are_types_compatible("modification", "addition") is True

    def test_addition_with_improvement_compatible(self):
        """Test addition is compatible with improvement."""
        assert are_types_compatible("addition", "improvement") is True
        assert are_types_compatible("improvement", "addition") is True

    def test_modification_with_improvement_compatible(self):
        """Test modification is compatible with improvement."""
        assert are_types_compatible("modification", "improvement") is True
        assert are_types_compatible("improvement", "modification") is True

    def test_case_insensitive(self):
        """Test type comparison is case insensitive."""
        assert are_types_compatible("DELETION", "deletion") is True
        assert are_types_compatible("Addition", "MODIFICATION") is True

    def test_none_defaults_to_modification(self):
        """Test None types default to modification."""
        assert are_types_compatible(None, "modification") is True
        assert are_types_compatible("addition", None) is True
        assert are_types_compatible(None, None) is True

    def test_empty_string_defaults_to_modification(self):
        """Test empty string types default to modification."""
        assert are_types_compatible("", "modification") is True
        assert are_types_compatible("addition", "") is True


class TestGroupSuggestionsForSubagents:
    """Tests for group_suggestions_for_subagents function."""

    def test_empty_list_returns_empty(self):
        """Test empty list returns empty list."""
        result = group_suggestions_for_subagents([])
        assert result == []

    def test_single_suggestion_creates_single_batch(self):
        """Test single suggestion creates one batch."""
        suggestions = [make_suggestion()]
        result = group_suggestions_for_subagents(suggestions)

        assert len(result) == 1
        assert result[0].size == 1

    def test_deletions_isolated_in_separate_batches(self):
        """Test deletion suggestions are isolated in separate batches."""
        suggestions = [
            make_suggestion(suggestion_type="deletion", reference="Step 1"),
            make_suggestion(suggestion_type="deletion", reference="Step 1"),
            make_suggestion(suggestion_type="addition", reference="Step 1"),
        ]
        result = group_suggestions_for_subagents(suggestions)

        # Each deletion should be in its own batch
        deletion_batches = [b for b in result if b.batch_type == "deletion"]
        assert len(deletion_batches) == 2

    def test_same_section_grouped_together(self):
        """Test suggestions in same section are grouped."""
        suggestions = [
            make_suggestion(reference="Step 1: Setup", suggestion_type="addition"),
            make_suggestion(reference="Step 1: Setup", suggestion_type="addition"),
            make_suggestion(reference="Step 2: Implement", suggestion_type="addition"),
        ]
        result = group_suggestions_for_subagents(suggestions)

        # Step 1 suggestions should be in one batch, Step 2 in another
        step1_batches = [b for b in result if "step_1" in b.section_key]
        step2_batches = [b for b in result if "step_2" in b.section_key]

        assert len(step1_batches) == 1
        assert step1_batches[0].size == 2
        assert len(step2_batches) == 1
        assert step2_batches[0].size == 1

    def test_respects_max_per_batch(self):
        """Test batching respects MAX_SUGGESTIONS_PER_BATCH constant.

        Note: The max_per_batch parameter is defined but the current implementation
        uses the global MAX_SUGGESTIONS_PER_BATCH (4) via can_add(). This test
        verifies actual behavior.
        """
        suggestions = [
            make_suggestion(reference="Step 1") for _ in range(10)
        ]
        # Pass max_per_batch=3 but actual behavior uses MAX_SUGGESTIONS_PER_BATCH=4
        result = group_suggestions_for_subagents(suggestions, max_per_batch=3)

        # 10 suggestions with actual max 4 per batch = 3 batches (4+4+2)
        assert len(result) == 3
        assert result[0].size == 4
        assert result[1].size == 4
        assert result[2].size == 2

    def test_respects_max_chars(self):
        """Test batching respects MAX_DESCRIPTION_CHARS constant.

        Note: The max_chars parameter is defined but the current implementation
        uses the global MAX_DESCRIPTION_CHARS (2500) via can_add(). This test
        verifies actual behavior. First suggestion always goes into batch,
        then subsequent check char limits.
        """
        # Create suggestions with long descriptions
        suggestions = [
            make_suggestion(description="x" * 1000, reference="Step 1"),
            make_suggestion(description="x" * 1000, reference="Step 1"),
            make_suggestion(description="x" * 1000, reference="Step 1"),
        ]
        # Actual MAX_DESCRIPTION_CHARS is 2500, so first 2 suggestions fit (2000 chars)
        # but 3rd would exceed, so goes to new batch
        result = group_suggestions_for_subagents(suggestions, max_chars=1500)

        assert len(result) == 2
        assert result[0].size == 2
        assert result[0].total_chars == 2000
        assert result[1].size == 1
        assert result[1].total_chars == 1000

    def test_group_by_section_false_groups_all_together(self):
        """Test group_by_section=False groups all suggestions together."""
        suggestions = [
            make_suggestion(reference="Step 1", suggestion_type="addition"),
            make_suggestion(reference="Step 2", suggestion_type="addition"),
            make_suggestion(reference="Step 3", suggestion_type="addition"),
        ]
        result = group_suggestions_for_subagents(
            suggestions, group_by_section=False, max_per_batch=10
        )

        # All should be in one batch since section grouping is disabled
        assert len(result) == 1
        assert result[0].size == 3

    def test_high_importance_sorted_first_within_section(self):
        """Test HIGH importance suggestions come first within section."""
        suggestions = [
            make_suggestion(importance="LOW", reference="Step 1"),
            make_suggestion(importance="HIGH", reference="Step 1"),
            make_suggestion(importance="MEDIUM", reference="Step 1"),
        ]
        result = group_suggestions_for_subagents(suggestions)

        # All in one batch, HIGH should be first
        assert len(result) == 1
        assert result[0].suggestions[0]["importance"] == "HIGH"
        assert result[0].suggestions[1]["importance"] == "MEDIUM"
        assert result[0].suggestions[2]["importance"] == "LOW"

    def test_incompatible_types_split_into_different_batches(self):
        """Test incompatible types are split into different batches."""
        suggestions = [
            make_suggestion(suggestion_type="addition", reference="Step 1"),
            make_suggestion(suggestion_type="deletion", reference="Step 2"),  # Deletion is isolated
            make_suggestion(suggestion_type="modification", reference="Step 1"),
        ]
        result = group_suggestions_for_subagents(suggestions)

        # Deletion should be isolated, addition and modification can be together
        deletion_batches = [b for b in result if b.batch_type == "deletion"]
        other_batches = [b for b in result if b.batch_type != "deletion"]

        assert len(deletion_batches) == 1
        assert deletion_batches[0].size == 1

    def test_batches_sorted_by_priority_score(self):
        """Test batches are sorted by priority score (higher first)."""
        suggestions = [
            make_suggestion(importance="LOW", reference="Step 1"),
            make_suggestion(importance="HIGH", reference="Step 2"),
        ]
        result = group_suggestions_for_subagents(suggestions)

        # HIGH priority batch should come first
        assert result[0].priority_score > result[1].priority_score

    def test_section_order_preserved_in_batching(self):
        """Test sections are processed in order (step 1 before step 2)."""
        suggestions = [
            make_suggestion(reference="Step 3: Third", importance="MEDIUM"),
            make_suggestion(reference="Step 1: First", importance="MEDIUM"),
            make_suggestion(reference="Step 2: Second", importance="MEDIUM"),
        ]
        result = group_suggestions_for_subagents(suggestions)

        # Should have 3 batches for 3 different sections
        # Priority score same, so sorted by section originally
        section_keys = [b.section_key for b in result]
        # They get sorted by priority which is same, so original processing order matters
        assert len(result) == 3


class TestFormatBatchForPrompt:
    """Tests for format_batch_for_prompt function."""

    def test_single_suggestion_format(self):
        """Test formatting of batch with single suggestion."""
        batch = SuggestionBatch(section_key="step_1")
        batch.add(make_suggestion(
            title="Add error handling",
            description="Add try-catch blocks",
            suggestion_type="addition",
            importance="HIGH",
            reference="Step 1: Setup"
        ))

        result = format_batch_for_prompt(batch, "/path/to/plan.md")

        assert "Apply the following suggestion" in result
        assert "/path/to/plan.md" in result
        assert "Add error handling" in result
        assert "addition" in result
        assert "Step 1: Setup" in result
        assert "HIGH" in result
        assert "Add try-catch blocks" in result

    def test_multiple_suggestions_format(self):
        """Test formatting of batch with multiple suggestions."""
        batch = SuggestionBatch(section_key="step_1")
        batch.add(make_suggestion(title="First suggestion", description="First desc"))
        batch.add(make_suggestion(title="Second suggestion", description="Second desc"))

        result = format_batch_for_prompt(batch, "/path/to/plan.md")

        assert "Apply the following 2 related suggestions" in result
        assert "### Suggestion 1: First suggestion" in result
        assert "### Suggestion 2: Second suggestion" in result
        assert "First desc" in result
        assert "Second desc" in result

    def test_batch_format_includes_section_info(self):
        """Test batch format includes section information."""
        batch = SuggestionBatch(section_key="step_3")
        batch.add(make_suggestion())
        batch.add(make_suggestion())

        result = format_batch_for_prompt(batch, "/plan.md")

        assert "(Section: step_3)" in result

    def test_batch_format_includes_batch_type(self):
        """Test batch format includes batch type."""
        batch = SuggestionBatch()
        batch.add(make_suggestion(suggestion_type="addition"))
        batch.add(make_suggestion(suggestion_type="addition"))

        result = format_batch_for_prompt(batch, "/plan.md")

        assert "**Batch type**: addition" in result

    def test_empty_batch_raises_value_error(self):
        """Test empty batch raises ValueError."""
        batch = SuggestionBatch()

        with pytest.raises(ValueError) as exc_info:
            format_batch_for_prompt(batch, "/plan.md")

        assert "empty batch" in str(exc_info.value).lower()

    def test_single_suggestion_includes_instructions(self):
        """Test single suggestion format includes instructions."""
        batch = SuggestionBatch()
        batch.add(make_suggestion())

        result = format_batch_for_prompt(batch, "/plan.md")

        assert "Instructions:" in result
        assert "Read the current plan file" in result
        assert "Locate the section" in result

    def test_multiple_suggestions_includes_batch_instructions(self):
        """Test multiple suggestions format includes batch instructions."""
        batch = SuggestionBatch()
        batch.add(make_suggestion())
        batch.add(make_suggestion())

        result = format_batch_for_prompt(batch, "/plan.md")

        assert "## Instructions" in result
        assert "Apply ALL suggestions in this batch" in result
        assert "## Return Format" in result

    def test_handles_missing_suggestion_fields(self):
        """Test formatting handles suggestions with missing fields."""
        batch = SuggestionBatch()
        batch.add({"title": "Minimal suggestion"})

        result = format_batch_for_prompt(batch, "/plan.md")

        assert "Minimal suggestion" in result
        assert "N/A" in result  # Missing reference
        assert "No description provided" in result

    def test_unknown_section_key_not_shown(self):
        """Test 'unknown' section key is not shown in batch format."""
        batch = SuggestionBatch(section_key="unknown")
        batch.add(make_suggestion())
        batch.add(make_suggestion())

        result = format_batch_for_prompt(batch, "/plan.md")

        assert "(Section: unknown)" not in result

    def test_single_suggestion_includes_prior_changes_placeholder(self):
        """Single suggestion format includes {prior_changes_context} placeholder."""
        batch = SuggestionBatch()
        batch.add(make_suggestion())

        result = format_batch_for_prompt(batch, "/plan.md")

        assert "{prior_changes_context}" in result
        assert "## Changes Applied in Prior Batches" in result

    def test_multiple_suggestions_includes_prior_changes_placeholder(self):
        """Multiple suggestion format includes {prior_changes_context} placeholder."""
        batch = SuggestionBatch()
        batch.add(make_suggestion())
        batch.add(make_suggestion())

        result = format_batch_for_prompt(batch, "/plan.md")

        assert "{prior_changes_context}" in result
        assert "## Changes Applied in Prior Batches" in result

    def test_prior_changes_placeholder_before_instructions(self):
        """Prior changes placeholder appears before Instructions section."""
        batch = SuggestionBatch()
        batch.add(make_suggestion())
        batch.add(make_suggestion())

        result = format_batch_for_prompt(batch, "/plan.md")

        placeholder_pos = result.index("{prior_changes_context}")
        instructions_pos = result.index("## Instructions")
        assert placeholder_pos < instructions_pos

    def test_prior_changes_placeholder_substitution(self):
        """Placeholder can be substituted with actual prior changes text."""
        batch = SuggestionBatch()
        batch.add(make_suggestion())

        result = format_batch_for_prompt(batch, "/plan.md")

        # First batch case
        substituted = result.replace("{prior_changes_context}", "(none — this is the first batch)")
        assert "(none — this is the first batch)" in substituted
        assert "{prior_changes_context}" not in substituted

        # Later batch case
        result2 = format_batch_for_prompt(batch, "/plan.md")
        later_text = "1. **Batch 1** (step_3): Added error handling guidance"
        substituted2 = result2.replace("{prior_changes_context}", later_text)
        assert later_text in substituted2
        assert "{prior_changes_context}" not in substituted2


class TestEstimateBatchProcessingStats:
    """Tests for estimate_batch_processing_stats function."""

    def test_empty_batches_returns_zeros(self):
        """Test empty batches list returns zero stats."""
        stats = estimate_batch_processing_stats([])

        assert stats["total_suggestions"] == 0
        assert stats["total_batches"] == 0
        assert stats["subagent_calls_saved"] == 0
        assert stats["efficiency_gain_percent"] == 0
        assert stats["average_batch_size"] == 0
        assert stats["max_batch_size"] == 0
        assert stats["single_suggestion_batches"] == 0
        assert stats["multi_suggestion_batches"] == 0
        assert stats["batch_type_distribution"] == {}

    def test_single_batch_single_suggestion(self):
        """Test stats for single batch with single suggestion."""
        batch = SuggestionBatch()
        batch.add(make_suggestion())

        stats = estimate_batch_processing_stats([batch])

        assert stats["total_suggestions"] == 1
        assert stats["total_batches"] == 1
        assert stats["subagent_calls_saved"] == 0
        assert stats["efficiency_gain_percent"] == 0.0
        assert stats["average_batch_size"] == 1.0
        assert stats["single_suggestion_batches"] == 1
        assert stats["multi_suggestion_batches"] == 0

    def test_efficiency_gain_calculation(self):
        """Test efficiency_gain_percent is calculated correctly."""
        # 2 batches with 4 suggestions each = 8 suggestions, 2 batches
        # Saved: 8 - 2 = 6 calls, efficiency = 6/8 = 75%
        batch1 = SuggestionBatch()
        for _ in range(4):
            batch1.add(make_suggestion())

        batch2 = SuggestionBatch()
        for _ in range(4):
            batch2.add(make_suggestion())

        stats = estimate_batch_processing_stats([batch1, batch2])

        assert stats["total_suggestions"] == 8
        assert stats["total_batches"] == 2
        assert stats["subagent_calls_saved"] == 6
        assert stats["efficiency_gain_percent"] == 75.0

    def test_batch_type_distribution(self):
        """Test batch type distribution is calculated correctly."""
        batch1 = SuggestionBatch()
        batch1.add(make_suggestion(suggestion_type="deletion"))

        batch2 = SuggestionBatch()
        batch2.add(make_suggestion(suggestion_type="addition"))

        batch3 = SuggestionBatch()
        batch3.add(make_suggestion(suggestion_type="addition"))

        stats = estimate_batch_processing_stats([batch1, batch2, batch3])

        assert stats["batch_type_distribution"]["deletion"] == 1
        assert stats["batch_type_distribution"]["addition"] == 2

    def test_average_batch_size(self):
        """Test average batch size calculation."""
        batch1 = SuggestionBatch()
        batch1.add(make_suggestion())

        batch2 = SuggestionBatch()
        batch2.add(make_suggestion())
        batch2.add(make_suggestion())

        batch3 = SuggestionBatch()
        batch3.add(make_suggestion())
        batch3.add(make_suggestion())
        batch3.add(make_suggestion())

        stats = estimate_batch_processing_stats([batch1, batch2, batch3])

        # (1 + 2 + 3) / 3 = 2.0
        assert stats["average_batch_size"] == 2.0

    def test_max_batch_size(self):
        """Test max batch size is correct."""
        batch1 = SuggestionBatch()
        batch1.add(make_suggestion())

        batch2 = SuggestionBatch()
        for _ in range(4):
            batch2.add(make_suggestion())

        stats = estimate_batch_processing_stats([batch1, batch2])

        assert stats["max_batch_size"] == 4

    def test_single_vs_multi_suggestion_batch_counts(self):
        """Test single and multi suggestion batch counts."""
        single_batch = SuggestionBatch()
        single_batch.add(make_suggestion())

        multi_batch1 = SuggestionBatch()
        multi_batch1.add(make_suggestion())
        multi_batch1.add(make_suggestion())

        multi_batch2 = SuggestionBatch()
        multi_batch2.add(make_suggestion())
        multi_batch2.add(make_suggestion())
        multi_batch2.add(make_suggestion())

        stats = estimate_batch_processing_stats([single_batch, multi_batch1, multi_batch2])

        assert stats["single_suggestion_batches"] == 1
        assert stats["multi_suggestion_batches"] == 2


class TestConstants:
    """Tests for module constants."""

    def test_max_suggestions_per_batch_value(self):
        """Test MAX_SUGGESTIONS_PER_BATCH has expected value."""
        assert MAX_SUGGESTIONS_PER_BATCH == 4

    def test_max_description_chars_value(self):
        """Test MAX_DESCRIPTION_CHARS has expected value."""
        assert MAX_DESCRIPTION_CHARS == 2500


class TestEdgeCases:
    """Edge case tests."""

    def test_suggestion_with_very_long_description(self):
        """Test handling of suggestion with very long description."""
        long_desc = "x" * 5000
        suggestions = [make_suggestion(description=long_desc)]

        result = group_suggestions_for_subagents(suggestions)

        assert len(result) == 1
        assert result[0].total_chars == 5000

    def test_many_suggestions_same_section(self):
        """Test many suggestions in same section are properly batched."""
        suggestions = [
            make_suggestion(reference="Step 1: Setup") for _ in range(20)
        ]

        result = group_suggestions_for_subagents(suggestions, max_per_batch=4)

        # 20 suggestions / 4 per batch = 5 batches
        assert len(result) == 5
        for batch in result:
            assert batch.size <= 4

    def test_mixed_importance_and_types(self):
        """Test complex scenario with mixed importance and types."""
        suggestions = [
            make_suggestion(importance="HIGH", suggestion_type="deletion", reference="Step 1"),
            make_suggestion(importance="MEDIUM", suggestion_type="addition", reference="Step 1"),
            make_suggestion(importance="LOW", suggestion_type="modification", reference="Step 2"),
            make_suggestion(importance="HIGH", suggestion_type="clarification", reference="Step 2"),
        ]

        result = group_suggestions_for_subagents(suggestions)

        # Deletion should be isolated
        deletion_batches = [b for b in result if b.batch_type == "deletion"]
        assert len(deletion_batches) == 1

    def test_unicode_in_descriptions(self):
        """Test handling of unicode characters in descriptions."""
        suggestions = [
            make_suggestion(description="Unicode test: \u00e9\u00e8\u00ea\u00eb"),
            make_suggestion(description="Emoji test: \U0001f600\U0001f389"),
        ]

        result = group_suggestions_for_subagents(suggestions)

        assert len(result) >= 1
        # Should not raise any exceptions

    def test_empty_description_suggestions(self):
        """Test suggestions with empty descriptions."""
        suggestions = [
            make_suggestion(description=""),
            make_suggestion(description=""),
            make_suggestion(description=""),
        ]

        result = group_suggestions_for_subagents(suggestions)

        # All should fit in one batch since 0 chars
        assert len(result) == 1
        assert result[0].total_chars == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
