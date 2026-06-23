"""Unit tests for validation subagent preparation functions in utils/validation.py.

Tests the new functions added for delegating validation to Claude Code subagents:
- build_validation_subagent_prompt()
- prepare_validation_task()
- prepare_revalidation_task()
"""
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from utils.validation import (
    build_validation_subagent_prompt,
    prepare_validation_task,
    prepare_revalidation_task,
    ERROR_TYPE_AMBIGUOUS,
    ERROR_TYPE_PARSING,
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_UNKNOWN,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_groups() -> List[Dict[str, Any]]:
    """Sample suggestion groups for testing."""
    return [
        {
            "theme": "Error Handling",
            "category": "reliability",
            "models": ["model-a", "model-b"],
            "suggestions": [
                {
                    "title": "Add error handling",
                    "desc": "Add try-catch blocks for network calls",
                    "importance": "HIGH",
                    "type": "addition"
                }
            ]
        },
        {
            "theme": "Performance",
            "category": "optimization",
            "models": ["model-a"],
            "suggestions": [
                {
                    "title": "Add caching",
                    "desc": "Cache database queries",
                    "importance": "MEDIUM",
                    "type": "addition"
                },
                {
                    "title": "Optimize loops",
                    "desc": "Use more efficient loop patterns",
                    "importance": "LOW",
                    "type": "modification"
                }
            ]
        },
        {
            "theme": "Security",
            "category": "security",
            "models": ["model-b"],
            "suggestions": [
                {
                    "title": "Input validation",
                    "desc": "Validate all user inputs",
                    "importance": "HIGH",
                    "type": "addition"
                }
            ]
        }
    ]


@pytest.fixture
def sample_context() -> str:
    """Sample plan context for validation."""
    return """# Implementation Plan

## Overview
Build a REST API with user authentication.

## Tasks
1. Set up database
2. Create user model
3. Implement authentication
4. Add API endpoints

## Technical Details
- Use PostgreSQL for database
- JWT for authentication tokens
"""


@pytest.fixture
def sample_validation_results() -> List[Dict[str, Any]]:
    """Sample validation results with various statuses."""
    return [
        {
            "group_index": 0,
            "status": "valid",
            "reason": "Issue is legitimate",
            "confidence": 0.9,
            "error_type": ERROR_TYPE_UNKNOWN,
            "recoverable": False,
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
            "status": "needs-human-decision",
            "reason": "Requires human judgment",
            "confidence": 0.5,
            "error_type": ERROR_TYPE_AMBIGUOUS,
            "recoverable": False,
        },
    ]


# ============================================================================
# Tests for build_validation_subagent_prompt()
# ============================================================================

class TestBuildValidationSubagentPrompt:
    """Tests for the build_validation_subagent_prompt function."""

    def test_includes_context(self, sample_context):
        """Prompt includes the provided context."""
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path="/tmp/validation.json"
        )
        assert "Implementation Plan" in prompt
        assert "REST API with user authentication" in prompt

    def test_includes_suggestions_json(self, sample_context):
        """Prompt includes the suggestions JSON."""
        suggestions = '[{"index": 0, "theme": "Test"}]'
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json=suggestions,
            output_path="/tmp/validation.json"
        )
        assert '"index": 0' in prompt
        assert '"theme": "Test"' in prompt

    def test_includes_output_path(self, sample_context):
        """Prompt includes the output path."""
        output_path = "/path/to/validation.json"
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path=output_path
        )
        assert output_path in prompt

    def test_includes_valid_status_option(self, sample_context):
        """Prompt includes 'valid' as a status option."""
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path="/tmp/validation.json"
        )
        assert '"valid"' in prompt or "'valid'" in prompt

    def test_includes_invalid_status_option(self, sample_context):
        """Prompt includes 'invalid' as a status option."""
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path="/tmp/validation.json"
        )
        assert '"invalid"' in prompt or "'invalid'" in prompt

    def test_includes_needs_human_decision_status_option(self, sample_context):
        """Prompt includes 'needs-human-decision' as a status option."""
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path="/tmp/validation.json"
        )
        assert "needs-human-decision" in prompt

    def test_includes_json_output_format(self, sample_context):
        """Prompt includes JSON output format instructions."""
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path="/tmp/validation.json"
        )
        assert "group_index" in prompt
        assert "status" in prompt
        assert "reason" in prompt
        assert "confidence" in prompt

    def test_includes_schema_version(self, sample_context):
        """Prompt includes schema version in output format."""
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path="/tmp/validation.json"
        )
        assert "schema_version" in prompt
        assert "2.1" in prompt  # v2.1 includes group_id support

    def test_includes_metadata_section(self, sample_context):
        """Prompt includes metadata section in output format."""
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path="/tmp/validation.json"
        )
        assert "metadata" in prompt
        assert "timestamp" in prompt

    def test_returns_string(self, sample_context):
        """Function returns a string."""
        prompt = build_validation_subagent_prompt(
            context=sample_context,
            suggestions_json="[]",
            output_path="/tmp/validation.json"
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ============================================================================
# Tests for prepare_validation_task()
# ============================================================================

class TestPrepareValidationTask:
    """Tests for the prepare_validation_task function."""

    def test_returns_dict(self, sample_groups, sample_context):
        """Function returns a dictionary."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        assert isinstance(result, dict)

    def test_contains_prompt_key(self, sample_groups, sample_context):
        """Result contains 'prompt' key."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        assert "prompt" in result
        assert isinstance(result["prompt"], str)
        assert len(result["prompt"]) > 0

    def test_contains_output_path_key(self, sample_groups, sample_context):
        """Result contains 'output_path' key."""
        output_path = "/path/to/validation.json"
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path=output_path
        )
        assert "output_path" in result
        assert result["output_path"] == output_path

    def test_contains_groups_count_key(self, sample_groups, sample_context):
        """Result contains 'groups_count' key with correct count."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        assert "groups_count" in result
        assert result["groups_count"] == len(sample_groups)

    def test_contains_suggestions_json_key(self, sample_groups, sample_context):
        """Result contains 'suggestions_json' key."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        assert "suggestions_json" in result
        assert isinstance(result["suggestions_json"], str)
        # Should be valid JSON
        parsed = json.loads(result["suggestions_json"])
        assert isinstance(parsed, list)

    def test_contains_model_hint_key(self, sample_groups, sample_context):
        """Result contains 'model_hint' key."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json",
            model="test-model"
        )
        assert "model_hint" in result
        assert result["model_hint"] == "test-model"

    def test_default_model_is_auto(self, sample_groups, sample_context):
        """Default model is 'auto'."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        assert result["model_hint"] == "auto"

    def test_handles_empty_groups(self, sample_context):
        """Handles empty groups list."""
        result = prepare_validation_task(
            groups=[],
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        assert result["groups_count"] == 0
        parsed = json.loads(result["suggestions_json"])
        assert parsed == []

    def test_extracts_suggestion_details(self, sample_groups, sample_context):
        """Extracts suggestion details into suggestions_json."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        parsed = json.loads(result["suggestions_json"])

        # Should have same number of items as groups
        assert len(parsed) == len(sample_groups)

        # Check first group
        assert parsed[0]["index"] == 0
        assert parsed[0]["theme"] == "Error Handling"
        assert parsed[0]["category"] == "reliability"
        assert parsed[0]["models"] == ["model-a", "model-b"]
        assert len(parsed[0]["suggestions"]) == 1
        assert parsed[0]["suggestions"][0]["title"] == "Add error handling"

    def test_truncates_long_context(self, sample_groups):
        """Truncates context longer than 30000 characters."""
        long_context = "x" * 35000
        result = prepare_validation_task(
            groups=sample_groups,
            context=long_context,
            output_path="/tmp/validation.json"
        )
        # Prompt should not contain full context
        assert len(result["prompt"]) < len(long_context)
        assert "[... truncated ...]" in result["prompt"]

    def test_handles_suggestion_group_objects(self, sample_context):
        """Handles objects with to_dict() method."""
        class MockSuggestionGroup:
            def to_dict(self):
                return {
                    "theme": "Mock Theme",
                    "category": "mock",
                    "models": ["mock-model"],
                    "suggestions": [{"title": "Mock", "desc": "Test", "importance": "LOW", "type": "addition"}]
                }

        groups = [MockSuggestionGroup()]
        result = prepare_validation_task(
            groups=groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        parsed = json.loads(result["suggestions_json"])
        assert parsed[0]["theme"] == "Mock Theme"


# ============================================================================
# Tests for prepare_revalidation_task()
# ============================================================================

class TestPrepareRevalidationTask:
    """Tests for the prepare_revalidation_task function."""

    def test_returns_dict(self, sample_groups, sample_validation_results, sample_context):
        """Function returns a dictionary."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_groups, sample_validation_results, sample_context):
        """Result contains all required keys."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        required_keys = ["prompt", "output_path", "items_to_revalidate", "item_indices", "original_validation", "model_hint"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_identifies_validation_failed_items(self, sample_groups, sample_validation_results, sample_context):
        """Identifies items with validation_failed status for revalidation."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        # Group 1 has validation_failed status
        assert result["items_to_revalidate"] >= 1
        assert 1 in result["item_indices"]

    def test_excludes_valid_items(self, sample_groups, sample_validation_results, sample_context):
        """Excludes items with 'valid' status."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        # Group 0 has valid status - should NOT be revalidated
        assert 0 not in result["item_indices"]

    def test_excludes_ambiguous_needs_human_by_default(self, sample_groups, sample_validation_results, sample_context):
        """Excludes needs-human-decision with ERROR_TYPE_AMBIGUOUS by default."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        # Group 2 has needs-human-decision with ERROR_TYPE_AMBIGUOUS - should NOT be revalidated
        assert 2 not in result["item_indices"]

    def test_includes_all_human_when_flag_set(self, sample_groups, sample_context):
        """Includes non-ambiguous needs-human-decision when include_all_human=True."""
        validation_results = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "needs-human-decision", "reason": "Parse error", "confidence": 0.0, "error_type": ERROR_TYPE_PARSING},
            {"group_index": 2, "status": "needs-human-decision", "reason": "Ambiguous", "confidence": 0.5, "error_type": ERROR_TYPE_AMBIGUOUS},
        ]
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json",
            include_all_human=True
        )
        # Group 1 should be included (parsing error), Group 2 should NOT (ambiguous)
        assert 1 in result["item_indices"]
        assert 2 not in result["item_indices"]

    def test_returns_none_prompt_when_nothing_to_revalidate(self, sample_groups, sample_context):
        """Returns None prompt when no items need revalidation."""
        validation_results = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 2, "status": "invalid", "reason": "False positive", "confidence": 0.8, "error_type": ERROR_TYPE_UNKNOWN},
        ]
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        assert result["prompt"] is None
        assert result["items_to_revalidate"] == 0
        assert result["item_indices"] == []

    def test_preserves_original_validation(self, sample_groups, sample_validation_results, sample_context):
        """Preserves original validation results for later merging."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        assert result["original_validation"] == sample_validation_results

    def test_output_path_in_result(self, sample_groups, sample_validation_results, sample_context):
        """Output path is included in result."""
        output_path = "/custom/path/revalidation.json"
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path=output_path
        )
        assert result["output_path"] == output_path

    def test_model_hint_preserved(self, sample_groups, sample_validation_results, sample_context):
        """Model hint is preserved in result."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json",
            model="custom-model"
        )
        assert result["model_hint"] == "custom-model"

    def test_default_model_is_auto(self, sample_groups, sample_validation_results, sample_context):
        """Default model hint is 'auto'."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        assert result["model_hint"] == "auto"

    def test_handles_mismatched_lengths(self, sample_context):
        """Handles when validation_results length doesn't match groups."""
        groups = [
            {"theme": "A", "category": "a", "models": [], "suggestions": []},
            {"theme": "B", "category": "b", "models": [], "suggestions": []},
            {"theme": "C", "category": "c", "models": [], "suggestions": []},
        ]
        validation_results = [
            {"group_index": 0, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            # Only 1 validation result for 3 groups
        ]
        result = prepare_revalidation_task(
            groups=groups,
            validation_results=validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        # Should only include index 0 (the only one with validation_failed)
        assert 0 in result["item_indices"]
        assert len(result["item_indices"]) == 1

    def test_prompt_contains_only_items_to_revalidate(self, sample_groups, sample_validation_results, sample_context):
        """Prompt only contains items that need revalidation."""
        result = prepare_revalidation_task(
            groups=sample_groups,
            validation_results=sample_validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        if result["prompt"]:
            # Should contain Performance theme (index 1, validation_failed)
            assert "Performance" in result["prompt"] or "Performance" in result.get("suggestions_json", "")
            # Should NOT contain Error Handling theme (index 0, valid)
            # Note: suggestions_json contains the filtered list
            parsed = json.loads(result["suggestions_json"])
            themes = [item.get("theme") for item in parsed]
            assert "Error Handling" not in themes


# ============================================================================
# Tests for Integration with Existing Functions
# ============================================================================

class TestIntegrationWithExistingFunctions:
    """Tests for integration between new and existing functions."""

    def test_suggestions_json_parseable(self, sample_groups, sample_context):
        """suggestions_json from prepare_validation_task is valid JSON."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        # Should not raise
        parsed = json.loads(result["suggestions_json"])
        assert isinstance(parsed, list)

    def test_prompt_includes_all_required_instructions(self, sample_groups, sample_context):
        """Prompt includes all required instructions for the subagent."""
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        prompt = result["prompt"]

        # Should include task instructions
        assert "validating" in prompt.lower()
        assert "suggestion" in prompt.lower()

        # Should include output format
        assert "JSON" in prompt or "json" in prompt

        # Should include status options
        assert "valid" in prompt
        assert "invalid" in prompt
        assert "needs-human-decision" in prompt

    def test_empty_groups_produces_empty_validation_input(self, sample_context):
        """Empty groups list produces empty validation input."""
        result = prepare_validation_task(
            groups=[],
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        assert result["groups_count"] == 0
        parsed = json.loads(result["suggestions_json"])
        assert len(parsed) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handles_unicode_in_context(self, sample_groups):
        """Handles Unicode characters in context."""
        context = "# Plan: Implement 日本語 Support\n\nAdd i18n for emoji: 🎉"
        result = prepare_validation_task(
            groups=sample_groups,
            context=context,
            output_path="/tmp/validation.json"
        )
        assert "日本語" in result["prompt"]

    def test_handles_unicode_in_suggestions(self, sample_context):
        """Handles Unicode characters in suggestions."""
        groups = [
            {
                "theme": "Internationalization 国际化",
                "category": "i18n",
                "models": ["model-a"],
                "suggestions": [
                    {"title": "Add 中文 support", "desc": "Support Chinese language", "importance": "HIGH", "type": "addition"}
                ]
            }
        ]
        result = prepare_validation_task(
            groups=groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        parsed = json.loads(result["suggestions_json"])
        assert "国际化" in parsed[0]["theme"]

    def test_handles_special_characters_in_output_path(self, sample_groups, sample_context):
        """Handles special characters in output path."""
        output_path = "/tmp/path with spaces/validation.json"
        result = prepare_validation_task(
            groups=sample_groups,
            context=sample_context,
            output_path=output_path
        )
        assert output_path in result["prompt"]

    def test_handles_empty_suggestions_list(self, sample_context):
        """Handles groups with empty suggestions list."""
        groups = [
            {
                "theme": "Empty",
                "category": "test",
                "models": [],
                "suggestions": []
            }
        ]
        result = prepare_validation_task(
            groups=groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        assert result["groups_count"] == 1
        parsed = json.loads(result["suggestions_json"])
        assert len(parsed[0]["suggestions"]) == 0

    def test_handles_missing_optional_fields(self, sample_context):
        """Handles groups with missing optional fields."""
        groups = [
            {
                "theme": "Minimal",
                "suggestions": [{"title": "Test"}]
            }
        ]
        result = prepare_validation_task(
            groups=groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        parsed = json.loads(result["suggestions_json"])
        # Should use defaults for missing fields
        assert parsed[0]["category"] == "unknown"
        assert parsed[0]["models"] == []

    def test_handles_very_long_suggestion_descriptions(self, sample_context):
        """Handles very long suggestion descriptions."""
        long_desc = "x" * 10000
        groups = [
            {
                "theme": "Long",
                "category": "test",
                "models": [],
                "suggestions": [{"title": "Test", "desc": long_desc, "importance": "LOW", "type": "addition"}]
            }
        ]
        result = prepare_validation_task(
            groups=groups,
            context=sample_context,
            output_path="/tmp/validation.json"
        )
        # Should complete without error
        assert result["groups_count"] == 1

    def test_context_exactly_at_limit(self, sample_groups):
        """Handles context exactly at truncation limit."""
        context = "x" * 30000
        result = prepare_validation_task(
            groups=sample_groups,
            context=context,
            output_path="/tmp/validation.json"
        )
        # Should not truncate at exactly 30000
        assert "[... truncated ...]" not in result["prompt"]

    def test_context_just_over_limit(self, sample_groups):
        """Handles context just over truncation limit."""
        context = "x" * 30001
        result = prepare_validation_task(
            groups=sample_groups,
            context=context,
            output_path="/tmp/validation.json"
        )
        # Should truncate at 30001
        assert "[... truncated ...]" in result["prompt"]

    def test_revalidation_with_all_validation_failed(self, sample_context):
        """Revalidation with all items having validation_failed status."""
        groups = [
            {"theme": "A", "category": "a", "models": [], "suggestions": []},
            {"theme": "B", "category": "b", "models": [], "suggestions": []},
        ]
        validation_results = [
            {"group_index": 0, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 1, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_PARSING},
        ]
        result = prepare_revalidation_task(
            groups=groups,
            validation_results=validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        assert result["items_to_revalidate"] == 2
        assert 0 in result["item_indices"]
        assert 1 in result["item_indices"]

    def test_revalidation_with_no_validation_failed(self, sample_context):
        """Revalidation with no items having validation_failed status."""
        groups = [
            {"theme": "A", "category": "a", "models": [], "suggestions": []},
            {"theme": "B", "category": "b", "models": [], "suggestions": []},
        ]
        validation_results = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "invalid", "reason": "False positive", "confidence": 0.8, "error_type": ERROR_TYPE_UNKNOWN},
        ]
        result = prepare_revalidation_task(
            groups=groups,
            validation_results=validation_results,
            context=sample_context,
            output_path="/tmp/revalidation.json"
        )
        assert result["items_to_revalidate"] == 0
        assert result["prompt"] is None
