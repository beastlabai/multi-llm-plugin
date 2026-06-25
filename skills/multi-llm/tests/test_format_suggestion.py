#!/usr/bin/env python3
"""Tests for format_suggestion, format_group, and report generation in review_plan_orchestrator."""

import json
import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from review_plan_orchestrator import format_suggestion, format_group, format_suggestion_in_group, aggregate_results


class TestFormatSuggestion:
    """Tests for format_suggestion function."""

    def test_includes_skip_checkbox(self):
        """Format includes - [ ] Skip checkbox line."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion)
        result = "\n".join(lines)

        assert "- [ ] Skip" in result

    def test_includes_explicit_validation_valid(self):
        """Format includes explicit validation status for valid."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion, validation_status="valid")
        result = "\n".join(lines)

        assert "**Validation:** ✓ Valid" in result or "**Validation:** Valid" in result

    def test_includes_explicit_validation_invalid(self):
        """Format includes explicit validation status for invalid."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion, validation_status="invalid")
        result = "\n".join(lines)

        assert "**Validation:** ✗ Invalid" in result or "**Validation:** Invalid" in result

    def test_includes_explicit_validation_needs_review(self):
        """Format includes explicit validation status for needs-human-decision."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion, validation_status="needs-human-decision")
        result = "\n".join(lines)

        assert "**Validation:** ? Needs Review" in result or "Needs Review" in result

    def test_includes_explicit_validation_failed(self):
        """Format includes explicit validation status for validation_failed."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion, validation_status="validation_failed")
        result = "\n".join(lines)

        assert "**Validation:** ? Validation Failed" in result or "Validation Failed" in result

    def test_includes_validation_reason_for_invalid(self):
        """Format includes validation reason for invalid suggestions."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(
            suggestion,
            validation_status="invalid",
            validation_reason="This is a false positive because the schema already handles this case."
        )
        result = "\n".join(lines)

        assert "**Validation Reason:**" in result
        assert "false positive" in result

    def test_includes_validation_reason_for_needs_review(self):
        """Format includes validation reason for needs-human-decision suggestions."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(
            suggestion,
            validation_status="needs-human-decision",
            validation_reason="Requires stakeholder input on the design decision."
        )
        result = "\n".join(lines)

        assert "**Validation Reason:**" in result
        assert "stakeholder input" in result

    def test_no_validation_reason_for_valid(self):
        """Format does not include validation reason for valid suggestions."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(
            suggestion,
            validation_status="valid",
            validation_reason="This is a valid suggestion."
        )
        result = "\n".join(lines)

        # Reason should NOT be displayed for valid suggestions
        assert "**Validation Reason:**" not in result

    def test_includes_unknown_for_no_status(self):
        """Format includes Unknown when no validation status provided."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion)
        result = "\n".join(lines)

        assert "**Validation:**" in result
        assert "Unknown" in result or "?" in result

    def test_title_does_not_have_badge(self):
        """Title line should not have compact badge like [✓]."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion, validation_status="valid")

        # First line is the title
        title_line = lines[0]
        assert "[✓]" not in title_line
        assert "[✗]" not in title_line
        assert "[?]" not in title_line

    def test_includes_model_type_section(self):
        """Format includes Model, Type, and Section metadata."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion, validation_status="valid")
        result = "\n".join(lines)

        assert "**Model:** test-model" in result
        assert "**Type:** addition" in result
        assert "**Section:** Step 1" in result

    def test_includes_description(self):
        """Format includes suggestion description."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "This is a detailed description of the suggestion.",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion)
        result = "\n".join(lines)

        assert "This is a detailed description of the suggestion." in result

    def test_includes_separator(self):
        """Format includes --- separator at the end."""
        suggestion = {
            "id": "S001",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion)

        assert "---" in lines

    def test_suggestion_id_in_title(self):
        """Format includes suggestion ID in title."""
        suggestion = {
            "id": "S042",
            "title": "Test suggestion",
            "desc": "Test description",
            "type": "addition",
            "reference": "Step 1",
            "model": "test-model",
        }

        lines = format_suggestion(suggestion)
        title_line = lines[0]

        assert "S042" in title_line
        assert "### S042:" in title_line


class TestFormatSuggestionStructure:
    """Tests for the structure of formatted output."""

    def test_correct_line_order(self):
        """Verify correct order: title, skip checkbox, validation line, blank, desc, blank, separator."""
        suggestion = {
            "id": "S001",
            "title": "Test",
            "desc": "Description here",
            "type": "addition",
            "reference": "Step 1",
            "model": "test",
        }

        lines = format_suggestion(suggestion, validation_status="valid")

        # Expected structure:
        # 0: ### S001: Test
        # 1: - [ ] Skip
        # 2: **Validation:** ... | **Model:** ... | **Type:** ... | **Section:** ...
        # 3: (empty)
        # 4: Description here
        # 5: (empty)
        # 6: ---
        # 7: (empty)

        assert lines[0].startswith("### S001:")
        assert "- [ ] Skip" in lines[1]
        assert "**Validation:**" in lines[2]
        assert lines[3] == ""
        assert "Description here" in lines[4]
        assert "---" in lines


class TestFormatGroup:
    """Tests for format_group function."""

    def test_includes_group_header_with_theme(self):
        """Format includes group header with theme."""
        group = {
            "theme": "Missing error handling",
            "category": "addition",
            "suggestions": [
                {"title": "Add try/catch", "desc": "Desc 1", "importance": "HIGH", "type": "addition", "reference": "Step 1", "source_model": "model-a"}
            ],
            "models": ["model-a"],
            "priority_score": 5,
            "validation_status": "valid",
        }

        lines = format_group(group, 1)
        result = "\n".join(lines)

        assert "## G1: Missing error handling" in result

    def test_includes_skip_group_checkbox(self):
        """Format includes checkbox to skip the entire group."""
        group = {
            "theme": "Test theme",
            "category": "addition",
            "suggestions": [
                {"title": "Test", "desc": "Desc", "importance": "MEDIUM", "type": "addition", "reference": "Step 1", "source_model": "model-a"}
            ],
            "models": ["model-a"],
            "priority_score": 3,
        }

        lines = format_group(group, 1)
        result = "\n".join(lines)

        assert "- [ ] Skip this group" in result

    def test_includes_group_metadata(self):
        """Format includes group metadata (validation, category, priority, models)."""
        group = {
            "theme": "Test theme",
            "category": "modification",
            "suggestions": [
                {"title": "Test", "desc": "Desc", "importance": "HIGH", "type": "modification", "reference": "Step 1", "source_model": "model-a"}
            ],
            "models": ["model-a", "model-b"],
            "priority_score": 7,
            "validation_status": "valid",
        }

        lines = format_group(group, 1)
        result = "\n".join(lines)

        assert "**Category:** modification" in result
        assert "**Priority:** 7" in result
        assert "**Models:** model-a, model-b" in result
        assert "**Validation:** Valid" in result

    def test_includes_highest_importance(self):
        """Format includes highest importance level in the group."""
        group = {
            "theme": "Test theme",
            "category": "addition",
            "suggestions": [
                {"title": "Low one", "desc": "Desc", "importance": "LOW", "type": "addition", "reference": "Step 1", "source_model": "model-a"},
                {"title": "High one", "desc": "Desc", "importance": "HIGH", "type": "addition", "reference": "Step 2", "source_model": "model-b"},
                {"title": "Medium one", "desc": "Desc", "importance": "MEDIUM", "type": "addition", "reference": "Step 3", "source_model": "model-c"},
            ],
            "models": ["model-a", "model-b", "model-c"],
            "priority_score": 10,
        }

        lines = format_group(group, 1)
        result = "\n".join(lines)

        assert "**Highest Importance:** HIGH" in result

    def test_suggestions_sorted_by_importance_within_group(self):
        """Suggestions within a group are sorted by importance (HIGH first)."""
        group = {
            "theme": "Test theme",
            "category": "addition",
            "suggestions": [
                {"title": "Low suggestion", "desc": "Desc low", "importance": "LOW", "type": "addition", "reference": "Step 1", "source_model": "model-a"},
                {"title": "High suggestion", "desc": "Desc high", "importance": "HIGH", "type": "addition", "reference": "Step 2", "source_model": "model-b"},
                {"title": "Medium suggestion", "desc": "Desc medium", "importance": "MEDIUM", "type": "addition", "reference": "Step 3", "source_model": "model-c"},
            ],
            "models": ["model-a", "model-b", "model-c"],
            "priority_score": 10,
        }

        lines = format_group(group, 1)
        result = "\n".join(lines)

        # Find positions of each suggestion in the output
        high_pos = result.find("High suggestion")
        medium_pos = result.find("Medium suggestion")
        low_pos = result.find("Low suggestion")

        # HIGH should come before MEDIUM, which should come before LOW
        assert high_pos < medium_pos < low_pos, "Suggestions should be sorted HIGH > MEDIUM > LOW"

    def test_includes_validation_reason_for_invalid_group(self):
        """Format includes validation reason for invalid groups."""
        group = {
            "theme": "Test theme",
            "category": "addition",
            "suggestions": [
                {"title": "Test", "desc": "Desc", "importance": "MEDIUM", "type": "addition", "reference": "Step 1", "source_model": "model-a"}
            ],
            "models": ["model-a"],
            "priority_score": 3,
            "validation_status": "invalid",
            "validation_reason": "This is a false positive.",
        }

        lines = format_group(group, 1)
        result = "\n".join(lines)

        assert "**Validation Reason:**" in result
        assert "false positive" in result

    def test_includes_separator_at_end(self):
        """Format includes --- separator at the end of the group."""
        group = {
            "theme": "Test theme",
            "category": "addition",
            "suggestions": [
                {"title": "Test", "desc": "Desc", "importance": "MEDIUM", "type": "addition", "reference": "Step 1", "source_model": "model-a"}
            ],
            "models": ["model-a"],
            "priority_score": 3,
        }

        lines = format_group(group, 1)

        assert "---" in lines


class TestFormatSuggestionInGroup:
    """Tests for format_suggestion_in_group function."""

    def test_uses_group_suggestion_id_format(self):
        """Suggestion ID format is G{group}S{suggestion}."""
        suggestion = {
            "title": "Test suggestion",
            "desc": "Description",
            "importance": "HIGH",
            "type": "addition",
            "reference": "Step 1",
            "source_model": "model-a",
        }

        lines = format_suggestion_in_group(suggestion, group_idx=2, suggestion_idx=3)
        result = "\n".join(lines)

        assert "### G2S3:" in result

    def test_includes_skip_checkbox(self):
        """Format includes individual skip checkbox for suggestion."""
        suggestion = {
            "title": "Test",
            "desc": "Desc",
            "importance": "HIGH",
            "type": "addition",
            "reference": "Step 1",
            "source_model": "model-a",
        }

        lines = format_suggestion_in_group(suggestion, group_idx=1, suggestion_idx=1)
        result = "\n".join(lines)

        assert "- [ ] Skip" in result

    def test_includes_importance_in_metadata(self):
        """Format includes importance level in metadata."""
        suggestion = {
            "title": "Test",
            "desc": "Desc",
            "importance": "HIGH",
            "type": "addition",
            "reference": "Step 1",
            "source_model": "model-a",
        }

        lines = format_suggestion_in_group(suggestion, group_idx=1, suggestion_idx=1)
        result = "\n".join(lines)

        assert "**Importance:** HIGH" in result

    def test_includes_type_section_model(self):
        """Format includes type, section (reference), and model."""
        suggestion = {
            "title": "Test",
            "desc": "Desc",
            "importance": "MEDIUM",
            "type": "modification",
            "reference": "Phase 2",
            "source_model": "gemini-pro",
        }

        lines = format_suggestion_in_group(suggestion, group_idx=1, suggestion_idx=1)
        result = "\n".join(lines)

        assert "**Type:** modification" in result
        assert "**Section:** Phase 2" in result
        assert "**Model:** gemini-pro" in result

    def test_includes_description(self):
        """Format includes the suggestion description."""
        suggestion = {
            "title": "Test",
            "desc": "This is a detailed description of what should be changed.",
            "importance": "LOW",
            "type": "clarification",
            "reference": "Step 5",
            "source_model": "model-a",
        }

        lines = format_suggestion_in_group(suggestion, group_idx=1, suggestion_idx=1)
        result = "\n".join(lines)

        assert "This is a detailed description of what should be changed." in result


class TestAggregateResultsGrouped:
    """Tests for aggregate_results with grouped output."""

    def test_displays_groups_in_grouped_json_order(self, tmp_path):
        """Groups are displayed in grouped.json order (matches HTML report)."""
        phase_dir = tmp_path / "test-plan" / "review-plan"
        phase_dir.mkdir(parents=True)

        # Create grouped.json with groups in a specific order
        groups = [
            {
                "theme": "Low priority group",
                "category": "clarification",
                "suggestions": [{"title": "Low", "desc": "Desc", "importance": "LOW", "type": "clarification", "reference": "Step 1", "source_model": "model-a"}],
                "models": ["model-a"],
                "priority_score": 2,
            },
            {
                "theme": "High priority group",
                "category": "addition",
                "suggestions": [{"title": "High", "desc": "Desc", "importance": "HIGH", "type": "addition", "reference": "Step 2", "source_model": "model-b"}],
                "models": ["model-b"],
                "priority_score": 10,
            },
            {
                "theme": "Medium priority group",
                "category": "modification",
                "suggestions": [{"title": "Medium", "desc": "Desc", "importance": "MEDIUM", "type": "modification", "reference": "Step 3", "source_model": "model-c"}],
                "models": ["model-c"],
                "priority_score": 5,
            },
        ]

        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(groups))

        # Call aggregate_results
        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(tmp_path / "test-plan"),
            phase_dir=str(phase_dir),
            models=["model-a", "model-b", "model-c"],
            failed_models={},
            validated_groups=None,  # Should load from grouped.json
        )

        report_content = Path(report_path).read_text()

        # Find positions of each group in the report
        high_pos = report_content.find("High priority group")
        medium_pos = report_content.find("Medium priority group")
        low_pos = report_content.find("Low priority group")

        # Groups should appear sorted by priority: High, Medium, Low
        assert high_pos < medium_pos < low_pos, "Groups should be sorted by priority (HIGH first, then MEDIUM, then LOW)"

    def test_report_shows_group_count(self, tmp_path):
        """Report header shows number of groups."""
        phase_dir = tmp_path / "test-plan" / "review-plan"
        phase_dir.mkdir(parents=True)

        groups = [
            {"theme": "Group 1", "category": "addition", "suggestions": [{"title": "S1", "desc": "D1", "importance": "HIGH", "type": "addition", "reference": "R1", "source_model": "m1"}], "models": ["m1"], "priority_score": 5},
            {"theme": "Group 2", "category": "modification", "suggestions": [{"title": "S2", "desc": "D2", "importance": "MEDIUM", "type": "modification", "reference": "R2", "source_model": "m2"}], "models": ["m2"], "priority_score": 3},
        ]

        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(groups))

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(tmp_path / "test-plan"),
            phase_dir=str(phase_dir),
            models=["m1", "m2"],
            failed_models={},
        )

        report_content = Path(report_path).read_text()

        assert "**Groups:** 2" in report_content

    def test_report_shows_suggestions_count_by_importance(self, tmp_path):
        """Report header shows suggestion counts by importance level."""
        phase_dir = tmp_path / "test-plan" / "review-plan"
        phase_dir.mkdir(parents=True)

        groups = [
            {
                "theme": "Group 1",
                "category": "addition",
                "suggestions": [
                    {"title": "High 1", "desc": "D1", "importance": "HIGH", "type": "addition", "reference": "R1", "source_model": "m1"},
                    {"title": "High 2", "desc": "D2", "importance": "HIGH", "type": "addition", "reference": "R2", "source_model": "m1"},
                ],
                "models": ["m1"],
                "priority_score": 8,
            },
            {
                "theme": "Group 2",
                "category": "modification",
                "suggestions": [
                    {"title": "Medium 1", "desc": "D3", "importance": "MEDIUM", "type": "modification", "reference": "R3", "source_model": "m2"},
                ],
                "models": ["m2"],
                "priority_score": 4,
            },
            {
                "theme": "Group 3",
                "category": "clarification",
                "suggestions": [
                    {"title": "Low 1", "desc": "D4", "importance": "LOW", "type": "clarification", "reference": "R4", "source_model": "m3"},
                    {"title": "Low 2", "desc": "D5", "importance": "LOW", "type": "clarification", "reference": "R5", "source_model": "m3"},
                    {"title": "Low 3", "desc": "D6", "importance": "LOW", "type": "clarification", "reference": "R6", "source_model": "m3"},
                ],
                "models": ["m3"],
                "priority_score": 2,
            },
        ]

        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(groups))

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(tmp_path / "test-plan"),
            phase_dir=str(phase_dir),
            models=["m1", "m2", "m3"],
            failed_models={},
        )

        report_content = Path(report_path).read_text()

        assert "2 HIGH" in report_content
        assert "1 MEDIUM" in report_content
        assert "3 LOW" in report_content
        assert "6 total" in report_content

    def test_uses_validated_groups_when_provided(self, tmp_path):
        """When validated_groups is provided, uses it instead of loading from file."""
        phase_dir = tmp_path / "test-plan" / "review-plan"
        phase_dir.mkdir(parents=True)

        # Create a different grouped.json that should NOT be used
        file_groups = [
            {"theme": "File group", "category": "addition", "suggestions": [{"title": "From file", "desc": "D1", "importance": "LOW", "type": "addition", "reference": "R1", "source_model": "m1"}], "models": ["m1"], "priority_score": 1},
        ]
        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(file_groups))

        # Provide validated_groups parameter
        validated_groups = [
            {"theme": "Validated group", "category": "modification", "suggestions": [{"title": "From param", "desc": "D2", "importance": "HIGH", "type": "modification", "reference": "R2", "source_model": "m2"}], "models": ["m2"], "priority_score": 5, "validation_status": "valid"},
        ]

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(tmp_path / "test-plan"),
            phase_dir=str(phase_dir),
            models=["m1", "m2"],
            failed_models={},
            validated_groups=validated_groups,
        )

        report_content = Path(report_path).read_text()

        # Should use validated_groups, not file
        assert "Validated group" in report_content
        assert "From param" in report_content
        assert "File group" not in report_content
        assert "From file" not in report_content

    def test_shows_validation_summary(self, tmp_path):
        """Report shows validation summary when groups have validation status."""
        phase_dir = tmp_path / "test-plan" / "review-plan"
        phase_dir.mkdir(parents=True)

        validated_groups = [
            {"theme": "Valid group", "category": "addition", "suggestions": [{"title": "S1", "desc": "D1", "importance": "HIGH", "type": "addition", "reference": "R1", "source_model": "m1"}], "models": ["m1"], "priority_score": 5, "validation_status": "valid"},
            {"theme": "Invalid group", "category": "modification", "suggestions": [{"title": "S2", "desc": "D2", "importance": "MEDIUM", "type": "modification", "reference": "R2", "source_model": "m2"}], "models": ["m2"], "priority_score": 3, "validation_status": "invalid", "validation_reason": "False positive"},
            {"theme": "Needs review group", "category": "clarification", "suggestions": [{"title": "S3", "desc": "D3", "importance": "LOW", "type": "clarification", "reference": "R3", "source_model": "m3"}], "models": ["m3"], "priority_score": 2, "validation_status": "needs-human-decision"},
        ]

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(tmp_path / "test-plan"),
            phase_dir=str(phase_dir),
            models=["m1", "m2", "m3"],
            failed_models={},
            validated_groups=validated_groups,
        )

        report_content = Path(report_path).read_text()

        assert "**Validation:** 1 valid, 1 invalid, 1 needs human review" in report_content

    def test_handles_empty_groups(self, tmp_path):
        """Report handles case with no groups gracefully."""
        phase_dir = tmp_path / "test-plan" / "review-plan"
        phase_dir.mkdir(parents=True)

        # No grouped.json file

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(tmp_path / "test-plan"),
            phase_dir=str(phase_dir),
            models=["m1"],
            failed_models={},
        )

        report_content = Path(report_path).read_text()

        assert "_No suggestions found._" in report_content
        assert "**Groups:** 0" in report_content

    def test_shows_failed_models_section(self, tmp_path):
        """Report includes failed models section."""
        phase_dir = tmp_path / "test-plan" / "review-plan"
        phase_dir.mkdir(parents=True)

        groups = [
            {"theme": "Group 1", "category": "addition", "suggestions": [{"title": "S1", "desc": "D1", "importance": "HIGH", "type": "addition", "reference": "R1", "source_model": "m1"}], "models": ["m1"], "priority_score": 5},
        ]

        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(groups))

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(tmp_path / "test-plan"),
            phase_dir=str(phase_dir),
            models=["m1", "m2", "m3"],
            failed_models={"m2": "Timeout after 300s", "m3": "Rate limited"},
        )

        report_content = Path(report_path).read_text()

        assert "## Models Failed" in report_content
        assert "**m2**: Timeout after 300s" in report_content
        assert "**m3**: Rate limited" in report_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestLetClaudeDecideCheckbox:
    """Markdown emit guards for the 'Let Claude decide' checkbox (Section 5)."""

    def _group(self, status):
        return {
            "theme": "Test theme",
            "category": "addition",
            "suggestions": [
                {"title": "T", "desc": "D", "importance": "MEDIUM",
                 "type": "addition", "reference": "Step 1", "source_model": "m"}
            ],
            "models": ["m"],
            "priority_score": 3,
            "validation_status": status,
        }

    def test_group_emits_checkbox_for_needs_human(self):
        result = "\n".join(format_group(self._group("needs-human-decision"), 1))
        assert "- [ ] Let Claude decide" in result

    def test_group_omits_checkbox_for_validation_failed(self):
        result = "\n".join(format_group(self._group("validation_failed"), 1))
        assert "- [ ] Let Claude decide" not in result
        # The existing valid/invalid checkboxes are still there.
        assert "- [ ] Mark valid" in result

    def test_group_omits_checkbox_for_valid(self):
        result = "\n".join(format_group(self._group("valid"), 1))
        assert "- [ ] Let Claude decide" not in result

    def _sugg(self):
        return {
            "title": "T", "desc": "D", "importance": "LOW",
            "type": "clarification", "reference": "Step 5", "source_model": "m",
        }

    def test_suggestion_emits_checkbox_multi_needs_human(self):
        lines = format_suggestion_in_group(
            self._sugg(), group_idx=1, suggestion_idx=1,
            group_validation_status="needs-human-decision",
            group_suggestion_count=2,
        )
        assert "- [ ] Let Claude decide" in "\n".join(lines)

    def test_suggestion_omits_checkbox_single_suggestion_group(self):
        lines = format_suggestion_in_group(
            self._sugg(), group_idx=1, suggestion_idx=1,
            group_validation_status="needs-human-decision",
            group_suggestion_count=1,
        )
        assert "- [ ] Let Claude decide" not in "\n".join(lines)

    def test_suggestion_omits_checkbox_validation_failed(self):
        lines = format_suggestion_in_group(
            self._sugg(), group_idx=1, suggestion_idx=1,
            group_validation_status="validation_failed",
            group_suggestion_count=2,
        )
        assert "- [ ] Let Claude decide" not in "\n".join(lines)
