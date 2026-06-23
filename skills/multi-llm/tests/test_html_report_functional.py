"""Functional tests for HTML report generation system.

These tests cover realistic data flows and edge cases for the HTML report
generation, including:
- HTML report content validation
- Data transformation pipeline
- Edge cases with missing/malformed data
- Selection file handling
- Template loading
"""

import json
import pytest
from pathlib import Path
import re
import html

from utils.html_report_generator import (
    get_model_metadata,
    extract_section_contexts,
    embed_log_snippets,
    generate_html_report,
    write_html_report,
    _build_suggestion_data,
    _build_group_data,
    string_to_color,
)
from utils.report_parser import load_html_selections, merge_selections


def extract_report_data(html_content: str) -> dict:
    """Extract the embedded reportData JSON from HTML content.

    Uses brace counting to properly extract the complete JSON object,
    handling nested structures and strings with special characters.

    Args:
        html_content: The complete HTML string

    Returns:
        Parsed dict from the reportData JSON

    Raises:
        ValueError: If reportData cannot be found or parsed
    """
    start_marker = 'const reportData = '
    start_idx = html_content.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find reportData in HTML")

    json_start = start_idx + len(start_marker)

    # Find where the JSON object ends by tracking braces
    depth = 0
    in_string = False
    escape_next = False
    end_idx = json_start

    for i, char in enumerate(html_content[json_start:], json_start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    json_str = html_content[json_start:end_idx]
    return json.loads(json_str)


# ============================================================================
# SECTION 1: HTML Report Content Validation
# ============================================================================


class TestHtmlReportStructure:
    """Tests that generated HTML is well-formed and contains expected elements."""

    @pytest.fixture
    def basic_groups(self):
        """Minimal valid groups for testing."""
        return [
            {
                "theme": "Test Theme",
                "category": "test",
                "models": ["claude-3-opus"],
                "suggestions": [
                    {"title": "Test", "desc": "Description", "importance": "HIGH"}
                ],
            }
        ]

    @pytest.fixture
    def sample_plan(self, tmp_path):
        """Create a sample plan file."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest content.")
        return plan_path

    def test_html_has_proper_doctype(self, basic_groups, sample_plan, tmp_path):
        """HTML starts with proper DOCTYPE declaration."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        assert html.strip().startswith("<!DOCTYPE html>")

    def test_html_has_head_and_body(self, basic_groups, sample_plan, tmp_path):
        """HTML contains proper head and body structure."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "</html>" in html

    def test_css_is_inline(self, basic_groups, sample_plan, tmp_path):
        """CSS styles are embedded inline in the HTML."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
            template_style="flat",
        )

        assert "<style>" in html
        # Check for key CSS classes from flat template
        assert ".group-card" in html
        assert ".filter-btn" in html
        assert ".badge" in html

    def test_javascript_is_inline(self, basic_groups, sample_plan, tmp_path):
        """JavaScript is embedded inline in the HTML."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
            template_style="flat",
        )

        assert "<script>" in html
        # Check for key JS function names from flat template
        assert "function renderHeader" in html or "renderHeader" in html
        assert "function applyFilters" in html or "applyFilters" in html
        assert "function exportSelections" in html or "exportSelections" in html

    def test_filters_section_present(self, basic_groups, sample_plan, tmp_path):
        """HTML contains filter controls section."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
            template_style="flat",
        )

        # Filter section elements from flat template
        assert 'id="model-filter"' in html
        assert 'id="importance-filters"' in html
        assert 'id="category-filter"' in html
        assert 'id="validation-filters"' in html

    def test_export_buttons_present(self, basic_groups, sample_plan, tmp_path):
        """HTML contains export action buttons."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
            template_style="flat",
        )

        assert 'id="btn-export"' in html
        assert 'id="btn-copy"' in html
        assert "Export Selections" in html

    def test_report_data_json_is_valid(self, basic_groups, sample_plan, tmp_path):
        """Embedded report data JSON is valid and parseable."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        # Verify the placeholder null was properly replaced (regression test)
        assert "}null;" not in html, "Placeholder 'null' was not replaced"

        # Extract and parse the embedded JSON data
        data = extract_report_data(html)

        assert "planPath" in data
        assert "groups" in data
        assert "summary" in data


# ============================================================================
# SECTION 2: Data Transformation Pipeline
# ============================================================================


class TestDataTransformation:
    """Tests for the full data transformation pipeline."""

    @pytest.fixture
    def multi_group_data(self):
        """Create realistic multi-group data for testing."""
        return [
            {
                "theme": "Error Handling",
                "category": "improvement",
                "models": ["claude-3-opus", "gpt-4"],
                "priority_score": 85,
                "validation_status": "valid",
                "validation_reason": "Identified real issue",
                "validation_confidence": 0.95,
                "suggestions": [
                    {
                        "title": "Add try-catch",
                        "desc": "Add error handling to API call.",
                        "importance": "HIGH",
                        "type": "addition",
                        "reference": "### Step 1",
                        "source_model": "claude-3-opus",
                    },
                    {
                        "title": "Log errors",
                        "desc": "Add error logging.",
                        "importance": "MEDIUM",
                        "type": "enhancement",
                        "reference": "### Step 1",
                        "source_model": "gpt-4",
                    },
                ],
            },
            {
                "theme": "Performance",
                "category": "optimization",
                "models": ["gemini-2.5-flash"],
                "priority_score": 65,
                "validation_status": "needs-human-decision",
                "validation_reason": "May depend on use case",
                "validation_confidence": 0.6,
                "suggestions": [
                    {
                        "title": "Add caching",
                        "desc": "Cache results to improve performance.",
                        "importance": "MEDIUM",
                        "type": "enhancement",
                        "reference": "### Step 2",
                        "source_model": "gemini-2.5-flash",
                    },
                ],
            },
        ]

    @pytest.fixture
    def multi_section_plan(self, tmp_path):
        """Create a plan with multiple sections."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("""# Test Plan

## Overview
This is the overview section.

### Step 1
First step details here.
Additional context for step 1.

### Step 2
Second step details here.
More information about step 2.

### Step 3
Third step content.
""")
        return plan_path

    def test_groups_have_correct_indices(self, multi_group_data, multi_section_plan, tmp_path):
        """Groups retain their original 1-based indices after priority sorting."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=multi_group_data,
            plan_path=multi_section_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus", "gpt-4", "gemini-2.5-flash"],
        )

        # Extract report data
        data = extract_report_data(html)

        # After priority sorting: "Performance" (needs-human-decision, rank 0) comes
        # before "Error Handling" (valid, rank 1).  Each group keeps its original index.
        assert data["groups"][0]["index"] == 2  # Performance (originally group 2)
        assert data["groups"][1]["index"] == 1  # Error Handling (originally group 1)

    def test_suggestion_ids_correctly_generated(self, multi_group_data, multi_section_plan, tmp_path):
        """Suggestion IDs follow G{group}S{suggestion} pattern using original indices."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=multi_group_data,
            plan_path=multi_section_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus", "gpt-4", "gemini-2.5-flash"],
        )

        data = extract_report_data(html)

        # After priority sorting: groups[0] = Performance (originally G2),
        # groups[1] = Error Handling (originally G1).
        # IDs are assigned at build time using the original index, before sorting.
        assert data["groups"][0]["suggestions"][0]["id"] == "G2S1"  # Performance
        # Error Handling suggestions
        assert data["groups"][1]["suggestions"][0]["id"] == "G1S1"
        assert data["groups"][1]["suggestions"][1]["id"] == "G1S2"

    def test_section_contexts_extracted_and_embedded(self, multi_group_data, multi_section_plan, tmp_path):
        """Section contexts are extracted from plan and embedded in report."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=multi_group_data,
            plan_path=multi_section_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus", "gpt-4", "gemini-2.5-flash"],
        )

        data = extract_report_data(html)

        # Section contexts should be present
        assert "sectionContexts" in data
        assert "### Step 1" in data["sectionContexts"]
        assert "First step details" in data["sectionContexts"]["### Step 1"]

    def test_log_snippets_embedded(self, multi_group_data, multi_section_plan, tmp_path):
        """Log snippets are embedded when log files exist."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        # Create a log file
        (phase_dir / "log_claude-3-opus.txt").write_text("Test log line 1\nTest log line 2\n")

        html = generate_html_report(
            groups=multi_group_data,
            plan_path=multi_section_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus", "gpt-4", "gemini-2.5-flash"],
        )

        data = extract_report_data(html)

        assert "logSnippets" in data
        assert "claude-3-opus" in data["logSnippets"]
        assert "Test log line 1" in data["logSnippets"]["claude-3-opus"]

    def test_model_metadata_embedded_for_all_models(self, multi_group_data, multi_section_plan, tmp_path):
        """Model metadata with separate provider/model colors are embedded using hash-based generation."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=multi_group_data,
            plan_path=multi_section_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus", "gpt-4", "gemini-2.5-flash"],
        )

        data = extract_report_data(html)

        assert "modelMetadata" in data
        assert "claude-3-opus" in data["modelMetadata"]
        assert "gpt-4" in data["modelMetadata"]
        assert "gemini-2.5-flash" in data["modelMetadata"]

        # Check metadata has separate provider and model colors
        claude_meta = data["modelMetadata"]["claude-3-opus"]
        assert "model_color" in claude_meta
        assert "provider_color" in claude_meta
        assert "model" in claude_meta
        assert "provider" in claude_meta
        assert "full" in claude_meta

        # Check model colors match hash-generated values
        assert claude_meta["model_color"] == string_to_color("claude-3-opus")
        assert data["modelMetadata"]["gpt-4"]["model_color"] == string_to_color("gpt-4")
        assert data["modelMetadata"]["gemini-2.5-flash"]["model_color"] == string_to_color("gemini-2.5-flash")

        # Verify all model colors are valid hex format and distinct
        colors = {
            data["modelMetadata"]["claude-3-opus"]["model_color"],
            data["modelMetadata"]["gpt-4"]["model_color"],
            data["modelMetadata"]["gemini-2.5-flash"]["model_color"],
        }
        assert len(colors) == 3  # All distinct
        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7

    def test_validation_confidence_as_percentage(self, multi_group_data, multi_section_plan, tmp_path):
        """Validation confidence values are included in report data."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=multi_group_data,
            plan_path=multi_section_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        data = extract_report_data(html)

        # After sorting: groups[0] = Performance (0.6), groups[1] = Error Handling (0.95)
        assert data["groups"][0]["validationConfidence"] == 0.6
        assert data["groups"][1]["validationConfidence"] == 0.95


# ============================================================================
# SECTION 3: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    @pytest.fixture
    def sample_plan(self, tmp_path):
        """Create a sample plan file."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Test Plan\n\nContent here.")
        return plan_path

    def test_groups_with_missing_validation_status(self, sample_plan, tmp_path):
        """Groups without validation_status get 'pending' default."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Test",
                "category": "test",
                "models": ["claude"],
                "suggestions": [{"title": "Test", "desc": "Desc"}],
                # No validation_status, validation_reason, validation_confidence
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        assert data["groups"][0]["validationStatus"] == "pending"

    def test_groups_with_no_suggestions(self, sample_plan, tmp_path):
        """Groups without suggestions are handled gracefully."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Empty Group",
                "category": "test",
                "models": ["claude"],
                # No suggestions key
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        assert data["groups"][0]["suggestions"] == []
        assert data["summary"]["totalSuggestions"] == 0

    def test_empty_models_list(self, sample_plan, tmp_path):
        """Report handles empty models list."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Test",
                "category": "test",
                "models": [],
                "suggestions": [{"title": "Test", "desc": "Desc"}],
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=[],
        )

        data = extract_report_data(html)

        assert data["models"] == []
        assert data["modelMetadata"] == {}

    def test_very_long_descriptions(self, sample_plan, tmp_path):
        """Long descriptions are handled without truncation in the data."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        long_desc = "A" * 10000  # 10,000 characters

        groups = [
            {
                "theme": "Test",
                "category": "test",
                "models": ["claude"],
                "suggestions": [{"title": "Test", "desc": long_desc}],
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        # Description should be preserved fully in data
        assert len(data["groups"][0]["suggestions"][0]["description"]) == 10000

    def test_special_characters_html_escaped(self, sample_plan, tmp_path):
        """Special characters are properly escaped in JSON."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Test <script>alert('xss')</script>",
                "category": "test",
                "models": ["claude"],
                "suggestions": [
                    {
                        "title": "Test \"quoted\" & <special>",
                        "desc": "Description with <html> & \"quotes\" and 'apostrophes'",
                    }
                ],
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        # JSON should parse correctly, preserving special characters
        assert "<script>" in data["groups"][0]["theme"]
        assert '"quoted"' in data["groups"][0]["suggestions"][0]["title"]

    def test_unicode_content_in_suggestions(self, sample_plan, tmp_path):
        """Unicode content is preserved correctly."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Test with unicode",
                "category": "test",
                "models": ["claude"],
                "suggestions": [
                    {
                        "title": "Test with CJK characters",
                        "desc": "Description with Japanese characters.",
                    }
                ],
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        # Unicode should be preserved (either directly or as escape sequences)
        # Note: output uses "description" not "desc"
        assert "Japanese" in data["groups"][0]["suggestions"][0]["description"] or \
               "Japanese" in html

    def test_groups_with_issues_key(self, sample_plan, tmp_path):
        """Groups using 'issues' instead of 'suggestions' are handled."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Code Issues",
                "category": "bug",
                "models": ["claude-3-opus"],
                "issues": [
                    {
                        "title": "Missing null check",
                        "description": "Add null check before accessing property.",
                        "importance": "HIGH",
                        "type": "bug",
                    },
                    {
                        "title": "Unused variable",
                        "desc": "Variable foo is declared but never used.",
                        "importance": "LOW",
                        "type": "cleanup",
                    },
                ],
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude-3-opus"],
        )

        data = extract_report_data(html)

        assert len(data["groups"][0]["suggestions"]) == 2
        assert data["groups"][0]["suggestions"][0]["title"] == "Missing null check"
        # Should use 'description' field when 'desc' not present
        assert "null check" in data["groups"][0]["suggestions"][0]["description"]

    def test_very_large_number_of_groups(self, sample_plan, tmp_path):
        """Report handles large number of groups (100+)."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": f"Group {i}",
                "category": "test",
                "models": ["claude"],
                "suggestions": [{"title": f"Suggestion {i}", "desc": f"Desc {i}"}],
            }
            for i in range(1, 151)  # 150 groups
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        assert len(data["groups"]) == 150
        assert data["summary"]["totalGroups"] == 150
        assert data["groups"][149]["index"] == 150

    def test_nested_sections_in_plan(self, tmp_path):
        """Plan with multiple heading levels is handled correctly."""
        plan_path = tmp_path / "nested-plan.md"
        plan_path.write_text("""# Main Plan

## Phase 1

### Step 1.1
Content for step 1.1.

#### Substep 1.1.1
Deep nested content.

### Step 1.2
Content for step 1.2.

## Phase 2

### Step 2.1
Content for step 2.1.
""")

        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Test",
                "category": "test",
                "models": ["claude"],
                "suggestions": [
                    {"title": "S1", "desc": "D1", "reference": "### Step 1.1"},
                    {"title": "S2", "desc": "D2", "reference": "#### Substep 1.1.1"},
                    {"title": "S3", "desc": "D3", "reference": "### Step 2.1"},
                ],
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        contexts = data["sectionContexts"]
        assert "### Step 1.1" in contexts
        assert "#### Substep 1.1.1" in contexts
        assert "### Step 2.1" in contexts

    def test_missing_plan_file(self, tmp_path):
        """Report handles missing plan file gracefully."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        plan_path = tmp_path / "nonexistent.md"  # Does not exist

        groups = [
            {
                "theme": "Test",
                "category": "test",
                "models": ["claude"],
                "suggestions": [{"title": "Test", "desc": "Desc", "reference": "### Step 1"}],
            }
        ]

        # Should not raise exception
        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        # Section contexts should be empty since plan doesn't exist
        assert data["sectionContexts"] == {}

    def test_suggestion_with_alternative_field_names(self, sample_plan, tmp_path):
        """Suggestions with alternative field names are handled."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Test",
                "category": "test",
                "models": ["claude"],
                "suggestions": [
                    {
                        "title": "Test 1",
                        "desc": "Primary description field",  # 'desc' used
                    },
                    {
                        "title": "Test 2",
                        "description": "Alternative description field",  # 'description' used
                    },
                    {
                        "title": "Test 3",
                        "reference": "### Step 1",  # 'reference' used
                    },
                    {
                        "title": "Test 4",
                        "section": "### Step 2",  # 'section' used
                    },
                ],
            }
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        suggestions = data["groups"][0]["suggestions"]
        assert suggestions[0]["description"] == "Primary description field"
        assert suggestions[1]["description"] == "Alternative description field"
        assert suggestions[2]["sectionRef"] == "### Step 1"
        assert suggestions[3]["sectionRef"] == "### Step 2"


# ============================================================================
# SECTION 4: Selection Files
# ============================================================================


class TestSelectionFiles:
    """Tests for user_selections.json handling."""

    def test_load_selections_all_fields(self, tmp_path):
        """Load selections with all fields present."""
        selections = {
            "plan_path": "/path/to/plan.md",
            "phase": "review-plan",
            "exported_at": "2025-01-15T10:30:00",
            "skipped_groups": [1, 3, 5],
            "skipped_suggestions": ["G1S1", "G2S3", "G4S2"],
            "edited_descriptions": {
                "G1S2": "Updated description for G1S2",
                "G3S1": "Modified text here",
            },
        }
        selections_path = tmp_path / "user_selections.json"
        selections_path.write_text(json.dumps(selections))

        result = load_html_selections(tmp_path)

        assert result is not None
        assert result["plan_path"] == "/path/to/plan.md"
        assert result["skipped_groups"] == [1, 3, 5]
        assert result["skipped_suggestions"] == ["G1S1", "G2S3", "G4S2"]
        assert result["edited_descriptions"]["G1S2"] == "Updated description for G1S2"

    def test_load_selections_partial_fields(self, tmp_path):
        """Load selections with only some fields present."""
        selections = {
            "skipped_groups": [2],
            "skipped_suggestions": [],
            # edited_descriptions missing
        }
        selections_path = tmp_path / "user_selections.json"
        selections_path.write_text(json.dumps(selections))

        result = load_html_selections(tmp_path)

        assert result is not None
        assert result["skipped_groups"] == [2]
        assert result["skipped_suggestions"] == []
        assert "edited_descriptions" not in result

    def test_load_selections_with_invalid_group_indices(self, tmp_path):
        """Load selections with invalid group indices (out of range)."""
        selections = {
            "skipped_groups": [1, 100, 999, -1],  # Mix of valid and invalid
            "skipped_suggestions": ["G1S1", "G100S1"],  # Mix of valid and invalid
            "edited_descriptions": {},
        }
        selections_path = tmp_path / "user_selections.json"
        selections_path.write_text(json.dumps(selections))

        result = load_html_selections(tmp_path)

        # File should load, validation of indices happens elsewhere
        assert result is not None
        assert 100 in result["skipped_groups"]
        assert -1 in result["skipped_groups"]

    def test_merge_overlapping_selections(self):
        """Merge HTML and markdown selections with overlapping items."""
        html = {
            "skipped_groups": [1, 2],  # Groups 1 and 2 skipped in HTML
            "skipped_suggestions": ["G1S1", "G3S1"],
            "edited_descriptions": {"G1S2": "HTML edit for G1S2"},
        }
        md_groups = {2, 3, 4}  # Groups 2, 3, 4 skipped in markdown
        md_suggestions = {"G2S1", "G3S1", "G4S1"}
        md_edited = {
            "G1S2": ("original", "MD edit for G1S2"),  # Will be overwritten
            "G2S1": ("original", "MD edit for G2S1"),  # Will be preserved
        }

        groups, suggestions, edited = merge_selections(
            html, md_groups, md_suggestions, md_edited
        )

        # Markdown skips are the base, HTML skips are unioned on top
        assert groups == {1, 2, 3, 4}  # Union of HTML {1,2} and markdown {2,3,4}
        assert suggestions == {"G1S1", "G2S1", "G3S1", "G4S1"}  # Union of HTML and markdown

        # Edited descriptions are merged with HTML taking precedence
        assert edited["G1S2"] == "HTML edit for G1S2"  # HTML wins
        assert edited["G2S1"] == "MD edit for G2S1"  # MD preserved

    def test_empty_html_preserves_markdown_skips(self):
        """Empty HTML selections should not erase non-empty markdown skips."""
        html = {
            "skipped_groups": [],  # Explicitly empty
            "skipped_suggestions": [],  # Explicitly empty
            "edited_descriptions": {},
        }
        md_groups = {1, 2, 3}  # Non-empty in markdown
        md_suggestions = {"G1S1", "G2S1"}  # Non-empty in markdown
        md_edited = {"G1S1": ("original", "edited")}

        groups, suggestions, edited = merge_selections(
            html, md_groups, md_suggestions, md_edited
        )

        # Markdown skips are preserved (union with empty HTML = markdown)
        assert groups == {1, 2, 3}
        assert suggestions == {"G1S1", "G2S1"}
        # Edited descriptions from markdown are also preserved
        assert edited == {"G1S1": "edited"}

    def test_none_html_falls_back_to_markdown(self):
        """When HTML selections are None, fall back to markdown entirely."""
        md_groups = {1, 2}
        md_suggestions = {"G1S1"}
        md_edited = {"G2S1": ("old", "new")}

        groups, suggestions, edited = merge_selections(
            None, md_groups, md_suggestions, md_edited
        )

        assert groups == {1, 2}
        assert suggestions == {"G1S1"}
        assert edited == {"G2S1": "new"}


# ============================================================================
# SECTION 5: Template Loading
# ============================================================================


class TestTemplateLoading:
    """Tests for HTML template loading behavior."""

    @pytest.fixture
    def basic_groups(self):
        """Minimal valid groups for testing."""
        return [
            {
                "theme": "Test",
                "category": "test",
                "models": ["claude"],
                "suggestions": [{"title": "Test", "desc": "Desc"}],
            }
        ]

    @pytest.fixture
    def sample_plan(self, tmp_path):
        """Create a sample plan file."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Test Plan")
        return plan_path

    def test_template_loads_from_correct_path(self, basic_groups, sample_plan, tmp_path):
        """Template loads successfully from expected path."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        # If template loaded successfully, it should have the expected structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "reportData" in html

    def test_template_missing_returns_error_html(self, basic_groups, sample_plan, tmp_path, monkeypatch):
        """Missing template returns error HTML without crashing."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        # Point module to fake location without template
        fake_path = tmp_path / "fake_module.py"
        fake_path.write_text("")

        import utils.html_report_generator as module
        original_file = module.__file__

        monkeypatch.setattr(module, "__file__", str(fake_path))

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        monkeypatch.setattr(module, "__file__", original_file)

        assert "Error: Template Not Found" in html
        assert "<!DOCTYPE html>" in html  # Still valid HTML

    def test_write_html_report_creates_file(self, basic_groups, sample_plan, tmp_path):
        """write_html_report creates the report file correctly."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        report_path = write_html_report(html, phase_dir)

        assert report_path.exists()
        assert report_path.name == "report.html"
        assert report_path.read_text(encoding='utf-8') == html

    def test_write_html_report_creates_parent_dirs(self, basic_groups, sample_plan, tmp_path):
        """write_html_report creates parent directories if needed."""
        phase_dir = tmp_path / "nested" / "path" / "review-plan"
        # Don't create the directory

        html = "<!DOCTYPE html><html><body>Test</body></html>"

        report_path = write_html_report(html, phase_dir)

        assert report_path.exists()
        assert phase_dir.exists()


# ============================================================================
# SECTION 6: Build Functions Unit Tests
# ============================================================================


class TestBuildFunctions:
    """Unit tests for internal build functions."""

    def test_build_suggestion_data_basic(self):
        """_build_suggestion_data produces correct output."""
        suggestion = {
            "title": "Test Title",
            "desc": "Test description",
            "importance": "high",
            "type": "addition",
            "reference": "### Step 1",
            "source_model": "claude-3-opus",
        }

        result = _build_suggestion_data(suggestion, group_index=2, suggestion_index=3)

        assert result["id"] == "G2S3"
        assert result["title"] == "Test Title"
        assert result["description"] == "Test description"
        assert result["importance"] == "HIGH"  # Uppercase
        assert result["type"] == "addition"
        assert result["sectionRef"] == "### Step 1"
        assert result["model"] == "claude-3-opus"

    def test_build_suggestion_data_defaults(self):
        """_build_suggestion_data handles missing fields with defaults."""
        suggestion = {}

        result = _build_suggestion_data(suggestion, group_index=1, suggestion_index=1)

        assert result["id"] == "G1S1"
        assert result["title"] == "Untitled"
        assert result["description"] == ""
        assert result["importance"] == "MEDIUM"
        assert result["type"] == "unknown"
        assert result["sectionRef"] == ""
        assert result["model"] == "unknown"

    def test_build_group_data_with_validation_results(self):
        """_build_group_data uses validation_results when group lacks validation."""
        group = {
            "theme": "Test Theme",
            "category": "improvement",
            "models": ["claude"],
            "priority_score": 75,
            "suggestions": [{"title": "S1", "desc": "D1"}],
            # No validation_status in group
        }
        validation_results = [
            {"status": "valid", "reason": "Looks correct", "confidence": 0.85}
        ]

        result = _build_group_data(group, index=0, validation_results=validation_results)

        assert result["validationStatus"] == "valid"
        assert result["validationReason"] == "Looks correct"
        assert result["validationConfidence"] == 0.85

    def test_build_group_data_prefers_group_validation(self):
        """_build_group_data prefers validation from group over validation_results."""
        group = {
            "theme": "Test",
            "category": "test",
            "models": ["claude"],
            "validation_status": "invalid",
            "validation_reason": "Group reason",
            "validation_confidence": 0.9,
            "suggestions": [],
        }
        validation_results = [
            {"status": "valid", "reason": "Results reason", "confidence": 0.5}
        ]

        result = _build_group_data(group, index=0, validation_results=validation_results)

        # Group values should take precedence
        assert result["validationStatus"] == "invalid"
        assert result["validationReason"] == "Group reason"
        assert result["validationConfidence"] == 0.9


# ============================================================================
# SECTION 7: Summary Statistics
# ============================================================================


class TestSummaryStatistics:
    """Tests for report summary statistics calculation."""

    @pytest.fixture
    def sample_plan(self, tmp_path):
        """Create a sample plan file."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Test Plan")
        return plan_path

    def test_summary_counts_validation_statuses(self, sample_plan, tmp_path):
        """Summary correctly counts each validation status."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {"theme": "G1", "validation_status": "valid", "suggestions": [{"title": "S1"}]},
            {"theme": "G2", "validation_status": "valid", "suggestions": [{"title": "S2"}]},
            {"theme": "G3", "validation_status": "invalid", "suggestions": [{"title": "S3"}]},
            {"theme": "G4", "validation_status": "needs-human-decision", "suggestions": [{"title": "S4"}]},
            {"theme": "G5", "validation_status": "needs-human-decision", "suggestions": [{"title": "S5"}]},
            {"theme": "G6", "validation_status": "needs-human-decision", "suggestions": [{"title": "S6"}]},
            {"theme": "G7", "validation_status": "validation_failed", "suggestions": [{"title": "S7"}]},
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        assert data["summary"]["totalGroups"] == 7
        assert data["summary"]["totalSuggestions"] == 7
        assert data["summary"]["validCount"] == 2
        assert data["summary"]["invalidCount"] == 1
        assert data["summary"]["needsHumanCount"] == 3
        assert data["summary"]["validationFailedCount"] == 1

    def test_summary_counts_total_suggestions_across_groups(self, sample_plan, tmp_path):
        """Summary correctly counts suggestions across all groups."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {"theme": "G1", "suggestions": [{"title": "S1"}, {"title": "S2"}, {"title": "S3"}]},
            {"theme": "G2", "suggestions": [{"title": "S4"}]},
            {"theme": "G3", "suggestions": []},  # Empty
            {"theme": "G4", "issues": [{"title": "I1"}, {"title": "I2"}]},  # Uses 'issues' key
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        assert data["summary"]["totalGroups"] == 4
        assert data["summary"]["totalSuggestions"] == 6  # 3 + 1 + 0 + 2

    def test_failed_models_included_in_report(self, sample_plan, tmp_path):
        """Failed models information is included in the report."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [{"theme": "G1", "suggestions": [{"title": "S1"}]}]
        failed_models = {
            "gpt-4": "Connection timeout after 30s",
            "gemini-pro": "API rate limit exceeded",
        }

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
            failed_models=failed_models,
        )

        data = extract_report_data(html)

        assert data["failedModels"]["gpt-4"] == "Connection timeout after 30s"
        assert data["failedModels"]["gemini-pro"] == "API rate limit exceeded"


# ============================================================================
# SECTION 8: PR/Flat Parity and Edge Cases
# ============================================================================


class TestZeroSuggestions:
    """Reports generate without errors with empty groups list in both templates."""

    @pytest.fixture
    def sample_plan(self, tmp_path):
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Empty Plan\n")
        return plan_path

    def test_zero_suggestions_pr_template(self, sample_plan, tmp_path):
        """PR template handles zero suggestions without errors."""
        phase_dir = tmp_path / "pr-review"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=[],
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=[],
            template_style="pr",
        )

        assert "<html" in html or "<!DOCTYPE html>" in html
        assert "</html>" in html

        data = extract_report_data(html)
        assert data["groups"] == []
        assert data["summary"]["totalGroups"] == 0
        assert data["summary"]["totalSuggestions"] == 0

    def test_zero_suggestions_flat_template(self, sample_plan, tmp_path):
        """Flat template handles zero suggestions without errors."""
        phase_dir = tmp_path / "flat-review"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=[],
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=[],
            template_style="flat",
        )

        assert "<html" in html or "<!DOCTYPE html>" in html
        assert "</html>" in html

        data = extract_report_data(html)
        assert data["groups"] == []
        assert data["summary"]["totalGroups"] == 0
        assert data["summary"]["totalSuggestions"] == 0

    def test_zero_suggestions_code_review_phase(self, sample_plan, tmp_path):
        """Code review phase with zero groups generates correctly."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=[],
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude-sonnet"],
            template_style="pr",
        )

        data = extract_report_data(html)
        assert data["groups"] == []
        assert data["fileView"]["_global"]["suggestions"] == []
        assert data["fileView"]["_global"]["suggestionCount"] == 0

    def test_zero_suggestions_task_review_phase(self, sample_plan, tmp_path):
        """Task review phase with zero groups generates correctly."""
        phase_dir = tmp_path / "review-tasks"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=[],
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-tasks",
            models=["claude-sonnet"],
            template_style="pr",
        )

        data = extract_report_data(html)
        assert data["groups"] == []
        assert data["taskView"]["_coverageGaps"]["suggestions"] == []


class TestSuggestionsLackingReferences:
    """Suggestions without file/section references routed to global/unanchored."""

    @pytest.fixture
    def sample_plan(self, tmp_path):
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Plan\n\n### Step 1\nContent.\n")
        return plan_path

    def test_no_file_no_section_routes_to_global_file_view(self, sample_plan, tmp_path):
        """Suggestions without file or section ref go to fileView._global."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "General advice",
                "category": "improvement",
                "models": ["claude"],
                "validation_status": "valid",
                "validation_reason": "OK",
                "suggestions": [
                    {
                        "title": "Use TypeScript strict mode",
                        "desc": "Enable strict mode in tsconfig.",
                        "type": "improvement",
                        "importance": "MEDIUM",
                        "source_model": "claude",
                        # No file, no reference
                    },
                ],
            },
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude"],
        )

        data = extract_report_data(html)

        # Should be in _global, not dropped
        assert data["fileView"]["_global"]["suggestionCount"] == 1

        # Should also appear in globalSuggestions
        assert len(data["globalSuggestions"]) == 1
        assert data["globalSuggestions"][0]["title"] == "Use TypeScript strict mode"

    def test_no_section_routes_to_section_view_global(self, sample_plan, tmp_path):
        """Suggestions without section ref go to sectionView._global."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "General thought",
                "category": "improvement",
                "models": ["claude"],
                "validation_status": "valid",
                "validation_reason": "OK",
                "suggestions": [
                    {
                        "title": "Consider architecture",
                        "desc": "Think about architecture.",
                        "type": "improvement",
                        "importance": "LOW",
                        "source_model": "claude",
                        # No reference
                    },
                ],
            },
            {
                "theme": "Section-anchored",
                "category": "improvement",
                "models": ["claude"],
                "validation_status": "valid",
                "validation_reason": "OK",
                "suggestions": [
                    {
                        "title": "Improve Step 1",
                        "desc": "Details.",
                        "type": "improvement",
                        "importance": "HIGH",
                        "source_model": "claude",
                        "reference": "### Step 1",
                    },
                ],
            },
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
        )

        data = extract_report_data(html)

        # The unanchored one should be in sectionView._global
        assert "_global" in data["sectionView"]
        global_themes = [g["theme"] for g in data["sectionView"]["_global"]["suggestions"]]
        assert "General thought" in global_themes
        assert "Section-anchored" not in global_themes

        # The section-anchored one should be under its section
        assert "### Step 1" in data["sectionView"]


class TestMalformedBaseRef:
    """Graceful fallback without crash when base_ref is malformed."""

    @pytest.fixture
    def sample_plan(self, tmp_path):
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Plan\n")
        return plan_path

    @pytest.fixture
    def code_groups(self):
        return [
            {
                "theme": "Fix bug",
                "category": "bug",
                "models": ["claude"],
                "validation_status": "valid",
                "validation_reason": "Clear.",
                "suggestions": [
                    {
                        "title": "Fix null pointer",
                        "desc": "Add null check.",
                        "type": "bug",
                        "file": "src/main.py",
                        "line_range": [10, 15],
                        "importance": "HIGH",
                        "source_model": "claude",
                    },
                ],
            },
        ]

    def test_malformed_base_ref_does_not_crash(self, code_groups, sample_plan, tmp_path):
        """Malformed base_ref triggers graceful fallback, not an exception."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        # This should not raise an exception even with a nonsense base_ref
        # The git_utils.capture_diff_hunks will fail gracefully
        html = generate_html_report(
            groups=code_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude"],
            base_ref="not-a-valid-ref-!!!@#$%",
        )

        # Should still produce valid HTML
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html

        data = extract_report_data(html)
        # Groups should still be present
        assert len(data["groups"]) == 1
        assert data["groups"][0]["theme"] == "Fix bug"

    def test_empty_base_ref_no_crash(self, code_groups, sample_plan, tmp_path):
        """Empty string base_ref is treated as None (no diff capture)."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        # The generate_html_report checks `base_ref is not None`, so empty
        # string will trigger diff capture but should still handle gracefully
        html = generate_html_report(
            groups=code_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude"],
            base_ref="",
        )

        assert "<!DOCTYPE html>" in html or "<html" in html
        data = extract_report_data(html)
        assert len(data["groups"]) == 1

    def test_none_base_ref_skips_diff(self, code_groups, sample_plan, tmp_path):
        """None base_ref skips diff capture entirely."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=code_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude"],
            base_ref=None,
        )

        data = extract_report_data(html)
        # No diffData should be present when base_ref is None
        assert "diffData" not in data
        assert len(data["groups"]) == 1


class TestBinaryFileHandling:
    """Binary files handled with placeholder in diff data."""

    def test_binary_diff_produces_binary_flag(self):
        """Binary file diff text produces binary: true in parsed data."""
        from utils.git_utils import _parse_unified_diff

        binary_diff = """diff --git a/assets/logo.png b/assets/logo.png
index abc123..def456 100644
Binary files a/assets/logo.png and b/assets/logo.png differ
"""
        result = _parse_unified_diff(binary_diff)

        assert result["binary"] is True
        assert result["hunks"] == []

    def test_binary_file_in_capture_diff_hunks_produces_placeholder(self):
        """capture_diff_hunks returns binary: true for binary files."""
        from unittest.mock import patch
        import subprocess

        from utils.git_utils import capture_diff_hunks

        # Mock subprocess.run to return a binary diff
        binary_diff_output = """diff --git a/image.png b/image.png
index abc123..def456 100644
Binary files a/image.png and b/image.png differ
"""
        mock_status = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=""
        )
        mock_diff = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=binary_diff_output
        )

        call_count = 0

        def mock_run(cmd, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            if "status" in cmd_str:
                return mock_status
            if "diff" in cmd_str:
                return mock_diff
            return mock_status

        with patch("subprocess.run", side_effect=mock_run):
            result = capture_diff_hunks("HEAD~1", ["image.png"])

        assert "image.png" in result
        assert result["image.png"]["binary"] is True
        assert result["image.png"]["hunks"] == []


class TestAllFallbackLevelsFail:
    """When all context fallback levels fail, 'Context unavailable' notice rendered."""

    @pytest.fixture
    def sample_plan(self, tmp_path):
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Plan\n")
        return plan_path

    def test_file_context_unavailable_when_no_diff_no_file_no_anchor(self, sample_plan, tmp_path):
        """When diff, file content, and anchor_text all fail, fileContexts shows unavailable."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Fix issue",
                "category": "bug",
                "models": ["claude"],
                "validation_status": "valid",
                "validation_reason": "Clear.",
                "suggestions": [
                    {
                        "title": "Fix something",
                        "desc": "Fix the thing.",
                        "type": "bug",
                        "file": "nonexistent/file.py",
                        "line_range": [10, 15],
                        "importance": "HIGH",
                        "source_model": "claude",
                        # No anchor_text
                    },
                ],
            },
        ]

        # Generate without base_ref and with a nonexistent file
        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude"],
            base_ref=None,  # No diff
        )

        data = extract_report_data(html)

        # fileContexts should have the file with available: False
        assert "fileContexts" in data
        assert "nonexistent/file.py" in data["fileContexts"]
        ctx = data["fileContexts"]["nonexistent/file.py"]
        assert ctx["available"] is False
        assert ctx["source"] == "none"

    def test_anchor_text_used_as_fallback(self, sample_plan, tmp_path):
        """When diff and file fail but anchor_text exists, it's used as fallback."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Fix issue",
                "category": "bug",
                "models": ["claude"],
                "validation_status": "valid",
                "validation_reason": "Clear.",
                "suggestions": [
                    {
                        "title": "Fix something",
                        "desc": "Fix the thing.",
                        "type": "bug",
                        "file": "nonexistent/file.py",
                        "line_range": [10, 15],
                        "importance": "HIGH",
                        "source_model": "claude",
                        "anchor_text": "if user is None:\n    raise Error()",
                    },
                ],
            },
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude"],
            base_ref=None,
        )

        data = extract_report_data(html)

        assert "fileContexts" in data
        ctx = data["fileContexts"]["nonexistent/file.py"]
        assert ctx["available"] is True
        assert ctx["source"] == "anchor"
        assert ctx["anchorText"] == "if user is None:\n    raise Error()"


class TestDiffCaptureTimeout:
    """When timeout is exceeded during diff capture, partial results returned."""

    def test_timeout_returns_partial_results(self):
        """capture_diff_hunks returns partial results on timeout."""
        from unittest.mock import patch
        import subprocess

        from utils.git_utils import capture_diff_hunks

        # Mock subprocess to raise TimeoutExpired on the git diff call
        mock_status = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=""
        )

        call_count = 0

        def mock_run(cmd, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            if "status" in cmd_str:
                return mock_status
            if "diff" in cmd_str:
                raise subprocess.TimeoutExpired(cmd="git diff", timeout=10)
            return mock_status

        with patch("subprocess.run", side_effect=mock_run):
            result = capture_diff_hunks("HEAD~1", ["src/main.py", "src/api.py"])

        # Should return a result dict (not raise)
        assert isinstance(result, dict)

        # Should have notices about timeout
        assert "_notices" in result
        assert any("timed out" in n for n in result["_notices"])

    def test_per_file_timeout_returns_processed_files(self):
        """When per-file processing times out, already-processed files are kept."""
        from unittest.mock import patch
        import subprocess
        import time

        from utils.git_utils import capture_diff_hunks, _GIT_TIMEOUT_SECONDS

        # Prepare a diff with two files
        diff_output = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,4 @@
 line1
+added_line
 line2
 line3
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,4 @@
 line1
+another_added
 line2
 line3
"""
        mock_status = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=""
        )
        mock_diff = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=diff_output
        )

        def mock_run(cmd, *args, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            if "status" in cmd_str:
                return mock_status
            if "diff" in cmd_str:
                return mock_diff
            return mock_status

        # Patch time.monotonic to simulate timeout after first file
        original_monotonic = time.monotonic
        call_index = [0]

        def mock_monotonic():
            call_index[0] += 1
            if call_index[0] <= 2:
                # First calls (before loop and first iteration): return start time
                return 100.0
            else:
                # After first file: return time exceeding timeout
                return 100.0 + _GIT_TIMEOUT_SECONDS + 1

        with patch("subprocess.run", side_effect=mock_run), \
             patch("time.monotonic", side_effect=mock_monotonic):
            result = capture_diff_hunks("HEAD~1", ["file1.py", "file2.py"])

        # file1.py should have been processed
        assert "file1.py" in result

        # Should have a timeout notice
        assert "_notices" in result
        assert any("timed out" in n for n in result["_notices"])
