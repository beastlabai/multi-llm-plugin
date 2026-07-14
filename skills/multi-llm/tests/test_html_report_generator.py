"""Tests for HTML report generator and related selection functions."""

import json
import pytest
from pathlib import Path
import tempfile
import shutil

from unittest.mock import patch

from utils.html_report_generator import (
    get_model_metadata,
    extract_section_contexts,
    embed_log_snippets,
    generate_html_report,
    string_to_color,
    parse_model_string,
    build_file_view,
    build_section_view,
    build_task_view,
    _extract_file_path,
    _build_suggestion_data,
)
from utils.report_parser import load_html_selections, merge_selections


class TestParseModelString:
    """Tests for parse_model_string function."""

    def test_parses_provider_and_model(self):
        """Parses provider:model format correctly."""
        provider, model = parse_model_string("claude-code:opus")
        assert provider == "claude-code"
        assert model == "opus"

    def test_handles_model_only(self):
        """Handles model name without provider."""
        provider, model = parse_model_string("opus")
        assert provider == ""
        assert model == "opus"

    def test_handles_empty_string(self):
        """Handles empty string."""
        provider, model = parse_model_string("")
        assert provider == ""
        assert model == ""

    def test_handles_multiple_colons(self):
        """Handles model strings with multiple colons (takes first split)."""
        provider, model = parse_model_string("kilocode:moonshotai/kimi-k2.5")
        assert provider == "kilocode"
        assert model == "moonshotai/kimi-k2.5"

    def test_common_providers(self):
        """Parses common provider formats."""
        cases = [
            ("cursor-agent:auto", "cursor-agent", "auto"),
            ("cursor-agent:gpt-5.2-high", "cursor-agent", "gpt-5.2-high"),
            ("claude-code:opus", "claude-code", "opus"),
            ("gemini:gemini-2.5-flash", "gemini", "gemini-2.5-flash"),
            ("codex:gpt-5.2-codex", "codex", "gpt-5.2-codex"),
            ("kilocode:moonshotai/kimi-k2.5", "kilocode", "moonshotai/kimi-k2.5"),
        ]
        for model_string, expected_provider, expected_model in cases:
            provider, model = parse_model_string(model_string)
            assert provider == expected_provider
            assert model == expected_model


class TestGetModelMetadata:
    """Tests for get_model_metadata function using hash-based color generation."""

    def test_returns_separate_provider_and_model_colors(self):
        """Returns separate colors for provider and model."""
        result = get_model_metadata("claude-code:opus")
        assert "provider_color" in result
        assert "model_color" in result
        assert result["provider"] == "claude-code"
        assert result["model"] == "opus"
        assert result["full"] == "claude-code:opus"

    def test_same_provider_same_color(self):
        """Same provider always produces the same provider color."""
        result1 = get_model_metadata("cursor-agent:auto")
        result2 = get_model_metadata("cursor-agent:gpt-5.2-high")
        assert result1["provider_color"] == result2["provider_color"]

    def test_same_model_same_color(self):
        """Same model name always produces the same model color."""
        result1 = get_model_metadata("cursor-agent:auto")
        result2 = get_model_metadata("codex:auto")  # Same model, different provider
        assert result1["model_color"] == result2["model_color"]

    def test_different_providers_different_colors(self):
        """Different providers produce different colors."""
        claude_code = get_model_metadata("claude-code:opus")
        cursor = get_model_metadata("cursor-agent:auto")
        gemini = get_model_metadata("gemini:gemini-2.5-flash")

        colors = {claude_code["provider_color"], cursor["provider_color"], gemini["provider_color"]}
        assert len(colors) == 3

    def test_different_models_different_colors(self):
        """Different model names produce different colors."""
        opus = get_model_metadata("claude-code:opus")
        auto = get_model_metadata("cursor-agent:auto")
        flash = get_model_metadata("gemini:gemini-2.5-flash")

        colors = {opus["model_color"], auto["model_color"], flash["model_color"]}
        assert len(colors) == 3

    def test_model_color_is_valid_hex(self):
        """Generated model colors are valid hex format."""
        result = get_model_metadata("claude-code:opus")
        color = result["model_color"]
        assert color.startswith("#")
        assert len(color) == 7
        int(color[1:], 16)  # Should not raise

    def test_provider_color_is_valid_hex(self):
        """Generated provider colors are valid hex format."""
        result = get_model_metadata("cursor-agent:auto")
        color = result["provider_color"]
        assert color.startswith("#")
        assert len(color) == 7
        int(color[1:], 16)  # Should not raise

    def test_no_provider_returns_gray_provider_color(self):
        """Model without provider returns gray for provider_color."""
        result = get_model_metadata("opus")
        assert result["provider"] == ""
        assert result["provider_color"] == "#6B7280"  # Default gray
        assert result["model"] == "opus"
        assert result["model_color"].startswith("#")

    def test_empty_model_string(self):
        """Handles empty model string."""
        result = get_model_metadata("")
        assert result["provider_color"] == "#6B7280"  # Default gray
        assert result["model_color"] == "#6B7280"  # Default gray
        assert result["provider"] == ""
        assert result["model"] == ""
        assert result["full"] == ""

    def test_preserves_full_model_string(self):
        """Full model string is preserved exactly as provided."""
        result = get_model_metadata("Claude-Code:Opus-20240229")
        assert result["full"] == "Claude-Code:Opus-20240229"

    def test_case_sensitive_color_generation(self):
        """Different case produces different colors (case-sensitive hashing)."""
        lower = get_model_metadata("claude-code:opus")
        upper = get_model_metadata("CLAUDE-CODE:OPUS")
        # Case matters in hash - colors will differ
        assert lower["model_color"] != upper["model_color"]
        assert lower["provider_color"] != upper["provider_color"]


class TestStringToColor:
    """Tests for string_to_color function."""

    def test_deterministic(self):
        """Same input always produces same color."""
        assert string_to_color("test") == string_to_color("test")

    def test_empty_string_returns_gray(self):
        """Empty string returns default gray color."""
        assert string_to_color("") == "#6B7280"

    def test_valid_hex_format(self):
        """Returns valid hex color format."""
        color = string_to_color("any-model-name")
        assert color.startswith("#")
        assert len(color) == 7
        # Should be parseable as hex
        int(color[1:], 16)

    def test_different_strings_different_colors(self):
        """Different strings produce different colors."""
        colors = {
            string_to_color("model-a"),
            string_to_color("model-b"),
            string_to_color("model-c"),
            string_to_color("model-d"),
        }
        assert len(colors) == 4

    def test_spread_across_hue_range(self):
        """Colors are spread across the hue range for variety."""
        # Generate colors for many models and verify they're varied
        colors = [string_to_color(f"model-{i}") for i in range(20)]
        unique_colors = set(colors)
        # Should have high variety (at least 15 unique out of 20)
        assert len(unique_colors) >= 15


class TestExtractSectionContexts:
    """Tests for extract_section_contexts function."""

    def test_extracts_simple_context(self):
        """Extracts context around a simple section header."""
        plan_content = """# Plan

## Overview
This is the overview.

### Step 1
Step 1 content here.
More details about step 1.

### Step 2
Step 2 content.
"""
        section_refs = {"### Step 1"}
        result = extract_section_contexts(plan_content, section_refs)

        assert "### Step 1" in result
        assert "Step 1 content" in result["### Step 1"]

    def test_extracts_multiple_sections(self):
        """Extracts context for multiple section references."""
        plan_content = """# Plan

### Step 1
First step content.

### Step 2
Second step content.

### Step 3
Third step content.
"""
        section_refs = {"### Step 1", "### Step 3"}
        result = extract_section_contexts(plan_content, section_refs)

        assert len(result) == 2
        assert "### Step 1" in result
        assert "### Step 3" in result
        assert "First step" in result["### Step 1"]
        assert "Third step" in result["### Step 3"]

    def test_handles_missing_section(self):
        """Gracefully handles section that doesn't exist in plan."""
        plan_content = """# Plan

### Step 1
Step 1 content.
"""
        section_refs = {"### Nonexistent Section"}
        result = extract_section_contexts(plan_content, section_refs)

        # Missing section is not in result
        assert "### Nonexistent Section" not in result

    def test_handles_empty_plan_content(self):
        """Returns empty dict for empty plan content."""
        result = extract_section_contexts("", {"### Step 1"})
        assert result == {}

    def test_handles_empty_section_refs(self):
        """Returns empty dict for empty section refs."""
        result = extract_section_contexts("# Plan\n\n### Step 1\nContent", set())
        assert result == {}

    def test_handles_none_section_refs(self):
        """Returns empty dict for None in section refs."""
        plan_content = "# Plan\n\n### Step 1\nContent"
        section_refs = {"### Step 1", None, ""}
        result = extract_section_contexts(plan_content, section_refs)

        # Only valid ref is included
        assert "### Step 1" in result
        assert len(result) == 1

    def test_context_includes_surrounding_lines(self):
        """Context includes lines before and after the header."""
        lines = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5",
                 "Line 6", "Line 7", "### Target", "Line 8", "Line 9",
                 "Line 10", "Line 11", "Line 12", "Line 13", "Line 14"]
        plan_content = "\n".join(lines)

        result = extract_section_contexts(plan_content, {"### Target"})

        context = result["### Target"]
        assert "### Target" in context
        # Should include context around the header
        assert "Line 8" in context

    def test_partial_match_in_header_text(self):
        """Matches section by text after hash markers."""
        plan_content = """# Plan

### Implementation Details
Here are the details.

More content.
"""
        # Use partial text without the exact hash prefix
        section_refs = {"Implementation Details"}
        result = extract_section_contexts(plan_content, section_refs)

        assert "Implementation Details" in result
        assert "Here are the details" in result["Implementation Details"]

    def test_case_insensitive_matching(self):
        """Section matching is case insensitive."""
        plan_content = """# Plan

### STEP ONE
Content for step one.
"""
        section_refs = {"### step one"}
        result = extract_section_contexts(plan_content, section_refs)

        assert "### step one" in result
        assert "Content for step one" in result["### step one"]


class TestExtractSectionContextsFullSection:
    """Tests for extract_section_contexts with full_section=True."""

    def test_full_section_captures_to_next_header(self):
        """Full section captures all text from header to next same-level header."""
        plan_content = """# Plan

### Step 1
Step 1 line 1.
Step 1 line 2.
Step 1 line 3.

### Step 2
Step 2 content.
"""
        result = extract_section_contexts(plan_content, {"### Step 1"}, full_section=True)

        assert "### Step 1" in result
        context = result["### Step 1"]
        assert "Step 1 line 1" in context
        assert "Step 1 line 2" in context
        assert "Step 1 line 3" in context
        # Should NOT include the next section
        assert "Step 2 content" not in context
        assert "### Step 2" not in context

    def test_full_section_last_section_captures_to_end(self):
        """Last section (no next header) captures to end of content."""
        plan_content = """# Plan

### Step 1
Step 1 content.

### Step 2
Step 2 line 1.
Step 2 line 2.
Final line."""
        result = extract_section_contexts(plan_content, {"### Step 2"}, full_section=True)

        assert "### Step 2" in result
        context = result["### Step 2"]
        assert "Step 2 line 1" in context
        assert "Step 2 line 2" in context
        assert "Final line." in context

    def test_full_section_includes_nested_headers(self):
        """Nested headers (e.g., #### within ###) are included in parent section."""
        plan_content = """# Plan

### Step 1
Overview of step 1.

#### Sub-step 1a
Sub-step 1a details.

#### Sub-step 1b
Sub-step 1b details.

### Step 2
Step 2 content.
"""
        result = extract_section_contexts(plan_content, {"### Step 1"}, full_section=True)

        assert "### Step 1" in result
        context = result["### Step 1"]
        assert "Overview of step 1" in context
        assert "#### Sub-step 1a" in context
        assert "Sub-step 1a details" in context
        assert "#### Sub-step 1b" in context
        assert "Sub-step 1b details" in context
        # Should NOT include the next same-level section
        assert "### Step 2" not in context
        assert "Step 2 content" not in context

    def test_full_section_stopped_by_higher_level_header(self):
        """Section is stopped by a higher-level header (e.g., ## stops ###)."""
        plan_content = """# Plan

### Step 1
Step 1 content.

## New Section
Different content.
"""
        result = extract_section_contexts(plan_content, {"### Step 1"}, full_section=True)

        assert "### Step 1" in result
        context = result["### Step 1"]
        assert "Step 1 content" in context
        assert "## New Section" not in context
        assert "Different content" not in context

    def test_full_section_default_false_preserves_behavior(self):
        """Calling with full_section=False (default) gives same result as no argument."""
        plan_content = """# Plan

### Step 1
Step 1 content here.
More details about step 1.

### Step 2
Step 2 content.
"""
        refs = {"### Step 1"}
        result_default = extract_section_contexts(plan_content, refs)
        result_explicit = extract_section_contexts(plan_content, refs, full_section=False)

        assert result_default == result_explicit

    def test_full_section_multiple_refs(self):
        """Full section works with multiple section references."""
        plan_content = """### Step 1
Step 1 content.

### Step 2
Step 2 content.

### Step 3
Step 3 content.
"""
        result = extract_section_contexts(
            plan_content, {"### Step 1", "### Step 3"}, full_section=True
        )

        assert len(result) == 2
        assert "Step 1 content" in result["### Step 1"]
        assert "Step 2 content" not in result["### Step 1"]
        assert "Step 3 content" in result["### Step 3"]

    def test_full_section_strips_trailing_blank_lines(self):
        """Full section output does not end with trailing blank lines."""
        plan_content = """### Step 1
Content here.



### Step 2
More content.
"""
        result = extract_section_contexts(plan_content, {"### Step 1"}, full_section=True)

        context = result["### Step 1"]
        assert not context.endswith('\n')
        assert context.endswith("Content here.")

    def test_full_section_empty_inputs(self):
        """Full section handles empty inputs same as default mode."""
        assert extract_section_contexts("", {"### Step 1"}, full_section=True) == {}
        assert extract_section_contexts("content", set(), full_section=True) == {}

    def test_full_section_missing_ref(self):
        """Full section handles missing section ref gracefully."""
        plan_content = "### Step 1\nContent."
        result = extract_section_contexts(
            plan_content, {"### Nonexistent"}, full_section=True
        )
        assert "### Nonexistent" not in result


class TestEmbedLogSnippets:
    """Tests for embed_log_snippets function."""

    def test_returns_log_content(self, tmp_path):
        """Returns correct log content for existing log files."""
        log_content = "Line 1\nLine 2\nLine 3\n"
        (tmp_path / "log_claude-3-opus.txt").write_text(log_content, encoding="utf-8")

        result = embed_log_snippets(tmp_path, ["claude-3-opus"])

        assert "claude-3-opus" in result
        assert "Line 1" in result["claude-3-opus"]
        assert "Line 2" in result["claude-3-opus"]

    def test_handles_missing_log_file(self, tmp_path):
        """Gracefully handles missing log files."""
        result = embed_log_snippets(tmp_path, ["nonexistent-model"])

        assert "nonexistent-model" not in result

    def test_handles_missing_log_dir(self, tmp_path):
        """Gracefully handles missing log directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        result = embed_log_snippets(nonexistent_dir, ["claude-3-opus"])

        assert result == {}

    def test_respects_max_lines_parameter(self, tmp_path):
        """Respects the max_lines parameter."""
        # Create log with 10 lines (no trailing newline to avoid empty split element)
        lines = [f"Line {i}" for i in range(1, 11)]
        log_content = "\n".join(lines)
        (tmp_path / "log_test-model.txt").write_text(log_content, encoding="utf-8")

        result = embed_log_snippets(tmp_path, ["test-model"], max_lines=5)

        content = result["test-model"]
        content_lines = content.split("\n")
        # Should only have last 5 lines (6-10)
        assert len(content_lines) == 5
        assert "Line 6" in content_lines[0]
        assert "Line 10" in content_lines[4]
        # First 5 lines should not be present
        assert "Line 5\n" not in content  # Exact line with newline
        assert not content.startswith("Line 1\n")
        assert not content.startswith("Line 5\n")

    def test_returns_all_lines_when_under_max(self, tmp_path):
        """Returns all lines when log is shorter than max_lines."""
        lines = ["Line 1", "Line 2", "Line 3"]
        log_content = "\n".join(lines)
        (tmp_path / "log_test-model.txt").write_text(log_content, encoding="utf-8")

        result = embed_log_snippets(tmp_path, ["test-model"], max_lines=50)

        content = result["test-model"]
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content

    def test_sanitizes_model_name_for_filename(self, tmp_path):
        """Sanitizes model names with special characters."""
        log_content = "Log for model with colons"
        # File uses sanitized name
        (tmp_path / "log_provider_model.txt").write_text(log_content, encoding="utf-8")

        result = embed_log_snippets(tmp_path, ["provider:model"])

        assert "provider:model" in result
        assert "Log for model with colons" in result["provider:model"]

    def test_multiple_models(self, tmp_path):
        """Handles multiple models."""
        (tmp_path / "log_model-a.txt").write_text("Log A", encoding="utf-8")
        (tmp_path / "log_model-b.txt").write_text("Log B", encoding="utf-8")

        result = embed_log_snippets(tmp_path, ["model-a", "model-b"])

        assert len(result) == 2
        assert "Log A" in result["model-a"]
        assert "Log B" in result["model-b"]

    def test_empty_models_list(self, tmp_path):
        """Handles empty models list."""
        result = embed_log_snippets(tmp_path, [])
        assert result == {}


class TestGenerateHtmlReport:
    """Tests for generate_html_report function."""

    @pytest.fixture
    def sample_groups(self):
        """Create sample groups for testing."""
        return [
            {
                "theme": "Error Handling",
                "category": "improvement",
                "models": ["claude-3-opus", "gpt-4"],
                "priority_score": 85,
                "validation_status": "valid",
                "validation_reason": "Identified real issue",
                "validation_confidence": 0.9,
                "suggestions": [
                    {
                        "title": "Add try-catch block",
                        "desc": "Add error handling to the API call.",
                        "importance": "HIGH",
                        "type": "addition",
                        "reference": "### Step 1",
                        "source_model": "claude-3-opus",
                    }
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
                    }
                ],
            },
        ]

    @pytest.fixture
    def sample_plan(self, tmp_path):
        """Create a sample plan file."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("""# Test Plan

## Overview
This is a test plan.

### Step 1
First step details.

### Step 2
Second step details.
""", encoding="utf-8")
        return plan_path

    def test_returns_valid_html(self, sample_groups, sample_plan, tmp_path):
        """Returns a valid HTML string."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=sample_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus", "gpt-4", "gemini-2.5-flash"],
        )

        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<head>" in html
        assert "</body>" in html

    def test_contains_plan_path(self, sample_groups, sample_plan, tmp_path):
        """HTML contains the plan path in the embedded reportData JSON.

        The path is embedded as a JSON string value, so it appears
        JSON-encoded rather than raw.  On Windows the path contains
        backslashes (``C:\\Users\\...``), which ``json.dumps`` escapes to
        ``\\\\`` — the raw ``str(path)`` therefore never appears verbatim.
        Assert against the JSON-encoded form, which is what actually lands
        in the HTML (and what the template decodes back via ``JSON.parse``).
        """
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=sample_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        # json.dumps() yields the quoted, escaped literal exactly as the
        # generator emits it, e.g. "C:\\Users\\...\\test-plan.md" on Windows
        # and "/tmp/.../test-plan.md" on POSIX.
        encoded_plan_path = json.dumps(str(sample_plan))
        assert f'"planPath": {encoded_plan_path}' in html

    def test_contains_model_metadata(self, sample_groups, sample_plan, tmp_path):
        """HTML contains model metadata with separate provider and model colors."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=sample_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-code:opus", "cursor-agent:auto"],
        )

        # Check model and provider colors are in the embedded JSON
        opus_model_color = string_to_color("opus")
        auto_model_color = string_to_color("auto")
        claude_code_provider_color = string_to_color("claude-code")
        cursor_provider_color = string_to_color("cursor-agent")
        assert opus_model_color in html
        assert auto_model_color in html
        assert claude_code_provider_color in html
        assert cursor_provider_color in html

    def test_contains_group_data(self, sample_groups, sample_plan, tmp_path):
        """HTML contains group data."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=sample_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        assert "Error Handling" in html
        assert "Performance" in html
        assert "Add try-catch block" in html

    def test_handles_empty_groups(self, sample_plan, tmp_path):
        """Handles empty groups list."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=[],
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        assert html.startswith("<!DOCTYPE html>")
        assert '"totalGroups": 0' in html

    def test_handles_missing_template(self, sample_groups, sample_plan, tmp_path, monkeypatch):
        """Returns error HTML when template is missing."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        # Create a fake module path that doesn't have the template
        fake_path = tmp_path / "fake_module.py"
        fake_path.write_text("", encoding="utf-8")

        # We can test the error path by checking the function behavior
        # when the template path doesn't exist
        import utils.html_report_generator as module
        original_file = module.__file__

        # Temporarily point to a location without templates
        monkeypatch.setattr(module, "__file__", str(fake_path))

        html = generate_html_report(
            groups=sample_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        # Restore original
        monkeypatch.setattr(module, "__file__", original_file)

        assert "Error: Template Not Found" in html
        assert "<!DOCTYPE html>" in html

    def test_includes_summary_statistics(self, sample_groups, sample_plan, tmp_path):
        """HTML includes summary statistics."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=sample_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
        )

        assert '"totalGroups": 2' in html
        assert '"totalSuggestions": 2' in html
        assert '"validCount": 1' in html
        assert '"needsHumanCount": 1' in html

    def test_includes_failed_models(self, sample_groups, sample_plan, tmp_path):
        """HTML includes failed models info."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        failed_models = {"model-x": "Connection timeout", "model-y": "API error"}

        html = generate_html_report(
            groups=sample_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude-3-opus"],
            failed_models=failed_models,
        )

        assert "model-x" in html
        assert "Connection timeout" in html

    def test_includes_validation_results(self, sample_plan, tmp_path):
        """HTML uses validation results if provided."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Test Group",
                "category": "test",
                "models": ["test-model"],
                "suggestions": [{"title": "Test", "desc": "Desc"}],
            }
        ]
        validation_results = [
            {"status": "invalid", "reason": "Not applicable", "confidence": 0.8}
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["test-model"],
            validation_results=validation_results,
        )

        assert "invalid" in html
        assert "Not applicable" in html

    def test_handles_groups_with_issues_key(self, sample_plan, tmp_path):
        """Handles groups that use 'issues' instead of 'suggestions'."""
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
                    }
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

        assert "Missing null check" in html
        assert "Code Issues" in html

    def test_phase_type_included(self, sample_groups, sample_plan, tmp_path):
        """Phase type is included in the report data."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=sample_groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude-3-opus"],
        )

        assert '"phase": "code-review"' in html


class TestLoadHtmlSelections:
    """Tests for load_html_selections function."""

    def test_returns_none_when_file_missing(self, tmp_path):
        """Returns None when user_selections.json doesn't exist."""
        result = load_html_selections(tmp_path)
        assert result is None

    def test_loads_valid_json(self, tmp_path):
        """Returns parsed dict when file exists with valid JSON."""
        selections = {
            "plan_path": "/path/to/plan.md",
            "phase": "review-plan",
            "exported_at": "2025-01-15T10:30:00",
            "skipped_groups": [1, 2],
            "skipped_suggestions": ["G1S1", "G2S3"],
            "edited_descriptions": {"G1S2": "Updated description"},
        }
        selections_path = tmp_path / "user_selections.json"
        selections_path.write_text(json.dumps(selections), encoding="utf-8")

        result = load_html_selections(tmp_path)

        assert result == selections
        assert result["skipped_groups"] == [1, 2]
        assert result["skipped_suggestions"] == ["G1S1", "G2S3"]

    def test_returns_none_for_invalid_json(self, tmp_path):
        """Returns None for invalid JSON content."""
        selections_path = tmp_path / "user_selections.json"
        selections_path.write_text("not valid json {{{", encoding="utf-8")

        result = load_html_selections(tmp_path)

        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Returns None for empty file."""
        selections_path = tmp_path / "user_selections.json"
        selections_path.write_text("", encoding="utf-8")

        result = load_html_selections(tmp_path)

        assert result is None

    def test_loads_minimal_selections(self, tmp_path):
        """Loads JSON with only required fields."""
        selections = {"skipped_groups": [], "skipped_suggestions": []}
        selections_path = tmp_path / "user_selections.json"
        selections_path.write_text(json.dumps(selections), encoding="utf-8")

        result = load_html_selections(tmp_path)

        assert result == selections

    def test_handles_unicode_content(self, tmp_path):
        """Handles Unicode content in selections."""
        selections = {
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {"G1S1": "Description with emoji and CJK characters"},
        }
        selections_path = tmp_path / "user_selections.json"
        selections_path.write_text(json.dumps(selections, ensure_ascii=False), encoding="utf-8")

        result = load_html_selections(tmp_path)

        assert result["edited_descriptions"]["G1S1"] == "Description with emoji and CJK characters"

    def test_matching_plan_path_returns_data(self, tmp_path):
        """When plan_path matches, data is returned normally."""
        selections = {
            "plan_path": "/home/user/plans/my-plan.md",
            "skipped_groups": [1],
            "skipped_suggestions": [],
        }
        (tmp_path / "user_selections.json").write_text(json.dumps(selections), encoding="utf-8")

        result = load_html_selections(tmp_path, plan_path="/home/user/plans/my-plan.md")

        assert result is not None
        assert result["skipped_groups"] == [1]

    def test_mismatched_plan_path_raises_value_error(self, tmp_path):
        """When plan_path differs, raises ValueError."""
        selections = {
            "plan_path": "/home/user/plans/plan-a.md",
            "skipped_groups": [1],
            "skipped_suggestions": [],
        }
        (tmp_path / "user_selections.json").write_text(json.dumps(selections), encoding="utf-8")

        with pytest.raises(ValueError, match="Plan path mismatch"):
            load_html_selections(tmp_path, plan_path="/home/user/plans/plan-b.md")

    def test_plan_path_normalization(self, tmp_path):
        """Paths that normalize to the same value are treated as equal."""
        selections = {
            "plan_path": "/home/user/plans/../plans/my-plan.md",
            "skipped_groups": [],
            "skipped_suggestions": [],
        }
        (tmp_path / "user_selections.json").write_text(json.dumps(selections), encoding="utf-8")

        result = load_html_selections(tmp_path, plan_path="/home/user/plans/my-plan.md")

        assert result is not None

    def test_missing_plan_path_in_json_warns(self, tmp_path, capsys):
        """When JSON lacks plan_path, prints warning but returns data."""
        selections = {"skipped_groups": [1], "skipped_suggestions": []}
        (tmp_path / "user_selections.json").write_text(json.dumps(selections), encoding="utf-8")

        result = load_html_selections(tmp_path, plan_path="/home/user/plans/my-plan.md")

        assert result is not None
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "does not contain" in captured.err

    def test_no_plan_path_arg_skips_validation(self, tmp_path):
        """When plan_path=None (default), no validation occurs."""
        selections = {
            "plan_path": "/some/completely/different/path.md",
            "skipped_groups": [1],
            "skipped_suggestions": [],
        }
        (tmp_path / "user_selections.json").write_text(json.dumps(selections), encoding="utf-8")

        result = load_html_selections(tmp_path)

        assert result is not None

    def test_error_message_includes_both_paths(self, tmp_path):
        """ValueError message includes both the JSON and current paths."""
        selections = {"plan_path": "/plans/alpha.md", "skipped_groups": []}
        (tmp_path / "user_selections.json").write_text(json.dumps(selections), encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            load_html_selections(tmp_path, plan_path="/plans/beta.md")

        assert "/plans/alpha.md" in str(exc_info.value)
        assert "/plans/beta.md" in str(exc_info.value)


class TestMergeSelections:
    """Tests for merge_selections function."""

    def test_html_unions_with_markdown_for_groups(self):
        """HTML group skips are unioned with markdown group skips."""
        html = {
            "skipped_groups": [1, 3],
            "skipped_suggestions": [],
            "edited_descriptions": {},
        }
        md_groups = {2, 4}  # Different groups in markdown
        md_suggestions = set()
        md_edited = {}

        groups, suggestions, edited = merge_selections(
            html, md_groups, md_suggestions, md_edited
        )

        # Union of HTML {1,3} and markdown {2,4}
        assert groups == {1, 2, 3, 4}

    def test_html_unions_with_markdown_for_suggestions(self):
        """HTML suggestion skips are unioned with markdown suggestion skips."""
        html = {
            "skipped_groups": [],
            "skipped_suggestions": ["G1S1", "G2S1"],
            "edited_descriptions": {},
        }
        md_groups = set()
        md_suggestions = {"G1S2", "G3S1"}  # Different suggestions
        md_edited = {}

        groups, suggestions, edited = merge_selections(
            html, md_groups, md_suggestions, md_edited
        )

        # Union of HTML {"G1S1", "G2S1"} and markdown {"G1S2", "G3S1"}
        assert suggestions == {"G1S1", "G1S2", "G2S1", "G3S1"}

    def test_falls_back_to_markdown_when_html_is_none(self):
        """Falls back to markdown when HTML is None."""
        md_groups = {1, 2}
        md_suggestions = {"G1S1", "G2S1"}
        md_edited = {"G1S2": ("old", "new")}

        groups, suggestions, edited = merge_selections(
            None, md_groups, md_suggestions, md_edited
        )

        assert groups == {1, 2}
        assert suggestions == {"G1S1", "G2S1"}
        assert edited == {"G1S2": "new"}

    def test_merges_edited_descriptions(self):
        """HTML edits overlay markdown edits."""
        html = {
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {"G1S1": "HTML edit"},
        }
        md_groups = set()
        md_suggestions = set()
        md_edited = {
            "G1S1": ("original", "Markdown edit"),  # Will be overwritten
            "G2S1": ("original", "MD only edit"),   # Will be preserved
        }

        groups, suggestions, edited = merge_selections(
            html, md_groups, md_suggestions, md_edited
        )

        # HTML edit overwrites markdown for G1S1
        assert edited["G1S1"] == "HTML edit"
        # Markdown edit preserved for G2S1
        assert edited["G2S1"] == "MD only edit"

    def test_empty_html_preserves_markdown_skips(self):
        """Empty HTML selections do not erase non-empty markdown skips."""
        html = {
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {},
        }
        md_groups = {1, 2, 3}
        md_suggestions = {"G1S1", "G1S2"}
        md_edited = {"G2S1": ("old", "new")}

        groups, suggestions, edited = merge_selections(
            html, md_groups, md_suggestions, md_edited
        )

        # Markdown skips preserved (union with empty HTML = markdown)
        assert groups == {1, 2, 3}
        assert suggestions == {"G1S1", "G1S2"}
        # Edited descriptions are merged (markdown preserved when not in HTML)
        assert edited == {"G2S1": "new"}

    def test_returns_correct_types(self):
        """Returns correct types: Set[int], Set[str], Dict[str, str]."""
        html = {
            "skipped_groups": [1, 2],
            "skipped_suggestions": ["G1S1"],
            "edited_descriptions": {"G2S1": "edited"},
        }
        md_groups = set()
        md_suggestions = set()
        md_edited = {}

        groups, suggestions, edited = merge_selections(
            html, md_groups, md_suggestions, md_edited
        )

        assert isinstance(groups, set)
        assert isinstance(suggestions, set)
        assert isinstance(edited, dict)
        # Check element types
        assert all(isinstance(g, int) for g in groups)
        assert all(isinstance(s, str) for s in suggestions)

    def test_converts_markdown_tuples_to_values(self):
        """Markdown edited dict tuples are converted to just the new value."""
        md_edited = {
            "G1S1": ("original text", "new text"),
            "G1S2": ("old", "updated"),
        }

        groups, suggestions, edited = merge_selections(
            None, set(), set(), md_edited
        )

        assert edited["G1S1"] == "new text"
        assert edited["G1S2"] == "updated"

    def test_handles_missing_edited_descriptions_key(self):
        """Handles HTML selections without edited_descriptions key."""
        html = {
            "skipped_groups": [1],
            "skipped_suggestions": ["G1S1"],
            # Note: edited_descriptions key is missing
        }
        md_edited = {"G2S1": ("old", "new")}

        groups, suggestions, edited = merge_selections(
            html, set(), set(), md_edited
        )

        # Markdown edits should be preserved when HTML has no edits
        assert edited == {"G2S1": "new"}

    def test_all_empty_inputs(self):
        """Handles all empty inputs gracefully."""
        groups, suggestions, edited = merge_selections(
            None, set(), set(), {}
        )

        assert groups == set()
        assert suggestions == set()
        assert edited == {}


# ---------------------------------------------------------------------------
# Helper: build processed groups for index builder tests
# ---------------------------------------------------------------------------

def _make_group(
    index: int,
    suggestions: list,
    theme: str = "Test Theme",
    validation_status: str = "valid",
) -> dict:
    """Build a processed group dict (post _build_group_data()) for testing."""
    return {
        "index": index,
        "originalIndex": index,
        "theme": theme,
        "category": "improvement",
        "models": ["test-model"],
        "priorityScore": 50,
        "validationStatus": validation_status,
        "validationReason": "",
        "validationConfidence": 0.8,
        "maxImportance": "MEDIUM",
        "suggestions": suggestions,
    }


def _make_suggestion(
    suggestion_id: str,
    file_ref: str = "",
    section_ref: str = "",
    importance: str = "MEDIUM",
    title: str = "Test suggestion",
) -> dict:
    """Build a processed suggestion dict for testing."""
    return {
        "id": suggestion_id,
        "title": title,
        "description": "Test description",
        "importance": importance,
        "type": "improvement",
        "sectionRef": section_ref,
        "fileRef": file_ref,
        "model": "test-model",
        "anchorText": None,
        "suggestedFix": None,
        "lineRange": None,
    }


class TestExtractFilePath:
    """Tests for _extract_file_path helper."""

    def test_strips_line_range(self):
        assert _extract_file_path("src/auth/login.py:43-50") == "src/auth/login.py"

    def test_strips_single_line(self):
        assert _extract_file_path("src/auth/login.py:43") == "src/auth/login.py"

    def test_no_line_range(self):
        assert _extract_file_path("src/auth/login.py") == "src/auth/login.py"

    def test_empty_string(self):
        assert _extract_file_path("") == ""

    def test_windows_style_colon_in_path(self):
        """Preserves drive letter colons (non-numeric after colon)."""
        assert _extract_file_path("C:/Users/test.py") == "C:/Users/test.py"

    def test_colon_with_non_numeric(self):
        """Preserves colons followed by non-numeric content."""
        assert _extract_file_path("provider:model") == "provider:model"


class TestBuildFileView:
    """Tests for build_file_view function."""

    def test_groups_by_file_path(self):
        """Groups referencing 3 files produce a dict with 3 file keys plus _global."""
        groups = [
            _make_group(1, [_make_suggestion("S1", file_ref="src/auth/login.py:10-20")]),
            _make_group(2, [_make_suggestion("S2", file_ref="src/api/handler.py:5-8")]),
            _make_group(3, [_make_suggestion("S3", file_ref="tests/test_auth.py:1-3")]),
            _make_group(4, [_make_suggestion("S4")]),  # No fileRef -> _global
        ]

        result = build_file_view(groups)

        assert "src/auth/login.py" in result
        assert "src/api/handler.py" in result
        assert "tests/test_auth.py" in result
        assert "_global" in result
        assert len(result) == 4  # 3 files + _global

    def test_global_for_unanchored_suggestions(self):
        """Suggestions without fileRef go into _global."""
        groups = [
            _make_group(1, [_make_suggestion("S1")]),  # No fileRef
            _make_group(2, [_make_suggestion("S2", file_ref="")]),  # Empty fileRef
        ]

        result = build_file_view(groups)

        assert "_global" in result
        assert len(result["_global"]["suggestions"]) == 2
        assert result["_global"]["suggestionCount"] == 2

    def test_sorts_by_directory_then_alphabetically(self):
        """Files sorted by directory then alphabetically."""
        groups = [
            _make_group(1, [_make_suggestion("S1", file_ref="src/z_module.py:1")]),
            _make_group(2, [_make_suggestion("S2", file_ref="src/a_module.py:1")]),
            _make_group(3, [_make_suggestion("S3", file_ref="lib/utils.py:1")]),
            _make_group(4, [_make_suggestion("S4", file_ref="src/api/handler.py:1")]),
            _make_group(5, [_make_suggestion("S5", file_ref="root_file.py:1")]),
        ]

        result = build_file_view(groups)
        keys = list(result.keys())

        # Expected order by (directory, filename):
        #   root_file.py -> ("", "root_file.py")
        #   lib/utils.py -> ("lib", "utils.py")
        #   src/a_module.py -> ("src", "a_module.py")
        #   src/z_module.py -> ("src", "z_module.py")
        #   src/api/handler.py -> ("src/api", "handler.py")
        #   _global -> ("\xff", "\xff")
        assert keys.index("root_file.py") < keys.index("lib/utils.py")
        assert keys.index("lib/utils.py") < keys.index("src/a_module.py")
        assert keys.index("src/a_module.py") < keys.index("src/z_module.py")
        assert keys.index("src/z_module.py") < keys.index("src/api/handler.py")
        assert keys[-1] == "_global"

    def test_suggestion_count(self):
        """Correctly counts total individual suggestions per file."""
        groups = [
            _make_group(1, [
                _make_suggestion("S1", file_ref="src/auth.py:10"),
                _make_suggestion("S2", file_ref="src/auth.py:20"),
            ]),
            _make_group(2, [
                _make_suggestion("S3", file_ref="src/auth.py:30"),
            ]),
        ]

        result = build_file_view(groups)

        assert result["src/auth.py"]["suggestionCount"] == 3

    def test_max_importance(self):
        """Computes maxImportance correctly per file."""
        groups = [
            _make_group(1, [
                _make_suggestion("S1", file_ref="src/auth.py:10", importance="LOW"),
                _make_suggestion("S2", file_ref="src/auth.py:20", importance="HIGH"),
            ]),
            _make_group(2, [
                _make_suggestion("S3", file_ref="src/api.py:10", importance="MEDIUM"),
            ]),
        ]

        result = build_file_view(groups)

        assert result["src/auth.py"]["maxImportance"] == "HIGH"
        assert result["src/api.py"]["maxImportance"] == "MEDIUM"

    def test_preserves_group_structure(self):
        """Suggestions within each group preserve their original order."""
        sugg1 = _make_suggestion("S1", file_ref="src/auth.py:10", title="First")
        sugg2 = _make_suggestion("S2", file_ref="src/auth.py:20", title="Second")
        group = _make_group(1, [sugg1, sugg2])

        result = build_file_view([group])

        returned_group = result["src/auth.py"]["suggestions"][0]
        assert returned_group["suggestions"][0]["title"] == "First"
        assert returned_group["suggestions"][1]["title"] == "Second"

    def test_empty_groups(self):
        """Handles empty groups list."""
        result = build_file_view([])

        assert "_global" in result
        assert result["_global"]["suggestions"] == []
        assert result["_global"]["suggestionCount"] == 0

    def test_multiple_groups_same_file(self):
        """Multiple groups referencing same file are grouped together."""
        groups = [
            _make_group(1, [_make_suggestion("S1", file_ref="src/auth.py:10")]),
            _make_group(2, [_make_suggestion("S2", file_ref="src/auth.py:30")]),
        ]

        result = build_file_view(groups)

        assert len(result["src/auth.py"]["suggestions"]) == 2
        assert result["src/auth.py"]["suggestionCount"] == 2


class TestBuildSectionView:
    """Tests for build_section_view function."""

    def test_groups_by_section_ref(self):
        """Groups suggestions by sectionRef."""
        groups = [
            _make_group(1, [_make_suggestion("S1", section_ref="### Step 1")]),
            _make_group(2, [_make_suggestion("S2", section_ref="### Step 2")]),
            _make_group(3, [_make_suggestion("S3", section_ref="### Step 1")]),
        ]

        result = build_section_view(groups)

        assert "### Step 1" in result
        assert "### Step 2" in result
        assert len(result["### Step 1"]["suggestions"]) == 2
        assert len(result["### Step 2"]["suggestions"]) == 1

    def test_includes_suggestion_counts(self):
        """Includes suggestion counts per section."""
        groups = [
            _make_group(1, [
                _make_suggestion("S1", section_ref="### Step 1"),
                _make_suggestion("S2", section_ref="### Step 1"),
            ]),
            _make_group(2, [
                _make_suggestion("S3", section_ref="### Step 1"),
            ]),
        ]

        result = build_section_view(groups)

        assert result["### Step 1"]["suggestionCount"] == 3

    def test_max_importance_per_section(self):
        """Computes maxImportance correctly per section."""
        groups = [
            _make_group(1, [
                _make_suggestion("S1", section_ref="### Step 1", importance="LOW"),
            ]),
            _make_group(2, [
                _make_suggestion("S2", section_ref="### Step 1", importance="HIGH"),
            ]),
            _make_group(3, [
                _make_suggestion("S3", section_ref="### Step 2", importance="MEDIUM"),
            ]),
        ]

        result = build_section_view(groups)

        assert result["### Step 1"]["maxImportance"] == "HIGH"
        assert result["### Step 2"]["maxImportance"] == "MEDIUM"

    def test_global_for_no_section_ref(self):
        """Groups without sectionRef go to _global."""
        groups = [
            _make_group(1, [_make_suggestion("S1")]),
            _make_group(2, [_make_suggestion("S2", section_ref="### Step 1")]),
        ]

        result = build_section_view(groups)

        assert "_global" in result
        assert len(result["_global"]["suggestions"]) == 1

    def test_preserves_group_structure(self):
        """Suggestions within groups preserve their original order."""
        sugg1 = _make_suggestion("S1", section_ref="### Step 1", title="First")
        sugg2 = _make_suggestion("S2", section_ref="### Step 1", title="Second")
        group = _make_group(1, [sugg1, sugg2])

        result = build_section_view([group])

        returned_group = result["### Step 1"]["suggestions"][0]
        assert returned_group["suggestions"][0]["title"] == "First"
        assert returned_group["suggestions"][1]["title"] == "Second"

    def test_empty_groups(self):
        """Handles empty groups list."""
        result = build_section_view([])

        assert result == {}


class TestBuildTaskView:
    """Tests for build_task_view function."""

    def test_separates_task_anchored_from_coverage_gaps(self):
        """Correctly separates task-anchored suggestions from coverage gaps."""
        groups = [
            _make_group(1, [_make_suggestion("S1", section_ref="T001")]),
            _make_group(2, [_make_suggestion("S2", section_ref="T002")]),
            _make_group(3, [_make_suggestion("S3", section_ref="Plan Coverage")]),
        ]
        tasks_metadata = {
            "T001": {"id": "T001", "title": "Auth Module", "description": "Implement auth"},
            "T002": {"id": "T002", "title": "API Handler", "description": "Build API"},
        }

        result = build_task_view(groups, tasks_metadata)

        assert "T001" in result
        assert "T002" in result
        assert "_coverageGaps" in result
        assert len(result["T001"]["suggestions"]) == 1
        assert len(result["T002"]["suggestions"]) == 1
        assert len(result["_coverageGaps"]["suggestions"]) == 1

    def test_tasks_metadata_none_produces_null_metadata(self):
        """With tasks_metadata=None, all taskMetadata values are None."""
        groups = [
            _make_group(1, [_make_suggestion("S1", section_ref="T001")]),
            _make_group(2, [_make_suggestion("S2", section_ref="T002")]),
        ]

        result = build_task_view(groups, tasks_metadata=None)

        assert result["T001"]["taskMetadata"] is None
        assert result["T002"]["taskMetadata"] is None

    def test_task_metadata_populated_when_available(self):
        """Task metadata is populated from tasks_metadata dict."""
        groups = [
            _make_group(1, [_make_suggestion("S1", section_ref="T001")]),
        ]
        tasks_metadata = {
            "T001": {
                "id": "T001",
                "title": "Auth Module",
                "description": "Implement authentication",
                "depends_on": ["T003"],
                "context_files": ["src/auth.py"],
                "output_files": ["src/auth_impl.py"],
                "acceptance_criteria": ["Tests pass"],
                "estimated_complexity": "MEDIUM",
            },
        }

        result = build_task_view(groups, tasks_metadata)

        meta = result["T001"]["taskMetadata"]
        assert meta["id"] == "T001"
        assert meta["title"] == "Auth Module"
        assert meta["depends_on"] == ["T003"]

    def test_missing_task_in_metadata_produces_null(self):
        """Task ID not in tasks_metadata dict gets None metadata."""
        groups = [
            _make_group(1, [_make_suggestion("S1", section_ref="T999")]),
        ]
        tasks_metadata = {
            "T001": {"id": "T001", "title": "Auth Module"},
        }

        result = build_task_view(groups, tasks_metadata)

        assert result["T999"]["taskMetadata"] is None

    def test_coverage_gaps_always_present(self):
        """_coverageGaps key is always present even when empty."""
        groups = [
            _make_group(1, [_make_suggestion("S1", section_ref="T001")]),
        ]

        result = build_task_view(groups)

        assert "_coverageGaps" in result
        assert result["_coverageGaps"]["suggestions"] == []
        assert result["_coverageGaps"]["suggestionCount"] == 0

    def test_suggestion_count_per_task(self):
        """Correctly counts suggestions per task."""
        groups = [
            _make_group(1, [
                _make_suggestion("S1", section_ref="T001"),
                _make_suggestion("S2", section_ref="T001"),
            ]),
            _make_group(2, [
                _make_suggestion("S3", section_ref="T001"),
            ]),
        ]

        result = build_task_view(groups)

        assert result["T001"]["suggestionCount"] == 3

    def test_max_importance_per_task(self):
        """Computes maxImportance correctly per task."""
        groups = [
            _make_group(1, [
                _make_suggestion("S1", section_ref="T001", importance="LOW"),
            ]),
            _make_group(2, [
                _make_suggestion("S2", section_ref="T001", importance="HIGH"),
            ]),
        ]

        result = build_task_view(groups)

        assert result["T001"]["maxImportance"] == "HIGH"

    def test_coverage_gaps_max_importance(self):
        """Computes maxImportance for coverage gaps."""
        groups = [
            _make_group(1, [
                _make_suggestion("S1", section_ref="Plan Coverage", importance="HIGH"),
            ]),
            _make_group(2, [
                _make_suggestion("S2", section_ref="Plan Coverage", importance="MEDIUM"),
            ]),
        ]

        result = build_task_view(groups)

        assert result["_coverageGaps"]["maxImportance"] == "HIGH"
        assert result["_coverageGaps"]["suggestionCount"] == 2

    def test_preserves_group_structure(self):
        """Suggestions within groups preserve their original order."""
        sugg1 = _make_suggestion("S1", section_ref="T001", title="First")
        sugg2 = _make_suggestion("S2", section_ref="T001", title="Second")
        group = _make_group(1, [sugg1, sugg2])

        result = build_task_view([group])

        returned_group = result["T001"]["suggestions"][0]
        assert returned_group["suggestions"][0]["title"] == "First"
        assert returned_group["suggestions"][1]["title"] == "Second"

    def test_empty_groups(self):
        """Handles empty groups list."""
        result = build_task_view([])

        assert "_coverageGaps" in result
        assert result["_coverageGaps"]["suggestions"] == []

    def test_unanchored_groups_go_to_unanchored(self):
        """Groups with no reference go to unanchored bucket, not coverage gaps."""
        groups = [
            _make_group(1, [_make_suggestion("S1")]),  # No sectionRef
        ]

        result = build_task_view(groups)

        assert "_unanchored" in result
        assert len(result["_unanchored"]["suggestions"]) == 1
        # Should NOT be in coverage gaps
        assert len(result["_coverageGaps"]["suggestions"]) == 0


# ---------------------------------------------------------------------------
# _build_suggestion_data: PR-style contextual fields
# ---------------------------------------------------------------------------


class TestBuildSuggestionDataPRFields:
    """Tests for _build_suggestion_data PR-style fields (anchorText, suggestedFix, lineRange)."""

    def test_pr_fields_present_when_source_has_them(self):
        """anchorText, suggestedFix, lineRange are populated when source provides them."""
        suggestion = {
            "title": "Fix null check",
            "desc": "Add null guard",
            "importance": "HIGH",
            "type": "bug",
            "reference": "### Auth",
            "source_model": "claude-3-opus",
            "file": "src/auth.py",
            "anchor_text": "if user is not None:",
            "suggested_fix": "if user is not None and user.active:",
            "line_range": [42, 48],
        }

        result = _build_suggestion_data(suggestion, group_index=1, suggestion_index=1)

        assert result["anchorText"] == "if user is not None:"
        assert result["suggestedFix"] == "if user is not None and user.active:"
        assert result["lineRange"] == [42, 48]

    def test_pr_fields_null_when_absent(self):
        """anchorText, suggestedFix, lineRange are None when source lacks them."""
        suggestion = {
            "title": "Generic improvement",
            "desc": "Make things better",
        }

        result = _build_suggestion_data(suggestion, group_index=1, suggestion_index=1)

        assert result["anchorText"] is None
        assert result["suggestedFix"] is None
        assert result["lineRange"] is None

    def test_file_ref_includes_line_range(self):
        """fileRef includes line range when file and line_range are provided."""
        suggestion = {
            "title": "Test",
            "desc": "Desc",
            "file": "src/handler.py",
            "line_range": [10, 25],
        }

        result = _build_suggestion_data(suggestion, group_index=1, suggestion_index=1)

        assert result["fileRef"] == "src/handler.py:10-25"

    def test_file_ref_single_line(self):
        """fileRef includes single line number when line_range has one element."""
        suggestion = {
            "title": "Test",
            "desc": "Desc",
            "file": "src/handler.py",
            "line_range": [10],
        }

        result = _build_suggestion_data(suggestion, group_index=1, suggestion_index=1)

        assert result["fileRef"] == "src/handler.py:10"

    def test_file_ref_without_line_range(self):
        """fileRef is plain file path when no line_range."""
        suggestion = {
            "title": "Test",
            "desc": "Desc",
            "file": "src/handler.py",
        }

        result = _build_suggestion_data(suggestion, group_index=1, suggestion_index=1)

        assert result["fileRef"] == "src/handler.py"

    def test_anchor_text_only(self):
        """Only anchor_text present, others absent."""
        suggestion = {
            "title": "Note",
            "desc": "Something",
            "anchor_text": "def process():",
        }

        result = _build_suggestion_data(suggestion, group_index=1, suggestion_index=1)

        assert result["anchorText"] == "def process():"
        assert result["suggestedFix"] is None
        assert result["lineRange"] is None


# ---------------------------------------------------------------------------
# Fallback chain integration tests
# ---------------------------------------------------------------------------


class TestFallbackChain:
    """Tests for the file context fallback chain in generate_html_report.

    The fallback chain resolves code context for each referenced file:
    1. Diff data available -> source: 'diff'
    2. File content via capture_file_context() -> source: 'file'
    3. anchor_text from suggestion -> source: 'anchor'
    4. None available -> source: 'none', available: False
    """

    @pytest.fixture
    def sample_plan(self, tmp_path):
        """Create a minimal plan file."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n### Step 1\nContent.\n", encoding="utf-8")
        return plan_path

    @pytest.fixture
    def phase_dir(self, tmp_path):
        """Create a phase directory."""
        d = tmp_path / "code-review"
        d.mkdir()
        return d

    def _make_code_review_groups(
        self,
        files_with_fields: list[dict],
    ) -> list[dict]:
        """Build raw groups for code-review style suggestions.

        Each entry in files_with_fields is a dict with keys:
        file, line_range (optional), anchor_text (optional).
        """
        groups = []
        for i, entry in enumerate(files_with_fields, start=1):
            suggestion = {
                "title": f"Issue {i}",
                "desc": f"Description for issue {i}",
                "importance": "MEDIUM",
                "type": "improvement",
                "source_model": "test-model",
                "file": entry.get("file", ""),
            }
            if "line_range" in entry:
                suggestion["line_range"] = entry["line_range"]
            if "anchor_text" in entry:
                suggestion["anchor_text"] = entry["anchor_text"]
            groups.append({
                "theme": f"Group {i}",
                "category": "improvement",
                "models": ["test-model"],
                "priority_score": 50,
                "validation_status": "valid",
                "validation_reason": "",
                "validation_confidence": 0.8,
                "suggestions": [suggestion],
            })
        return groups

    def test_fallback_level1_diff_available(self, sample_plan, phase_dir):
        """When diff data covers a file, file_contexts marks it source='diff'."""
        groups = self._make_code_review_groups([
            {"file": "src/auth.py", "line_range": [10, 20]},
        ])
        diff_data = {
            "src/auth.py": {
                "hunks": [{"header": "@@ -10,5 +10,6 @@", "lines": []}],
                "binary": False,
                "deleted": False,
                "old_path": "src/auth.py",
                "new_path": "src/auth.py",
            }
        }

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["test-model"],
            diff_data=diff_data,
        )

        # The report data should contain fileContexts with diff source
        assert '"source": "diff"' in html
        assert '"available": true' in html

    @patch("utils.git_utils.capture_file_context")
    def test_fallback_level2_file_content(
        self, mock_capture, sample_plan, phase_dir
    ):
        """When no diff but file exists, fallback to capture_file_context."""
        mock_capture.return_value = [
            {"line_number": 10, "content": "def hello():"},
            {"line_number": 11, "content": "    pass"},
        ]

        groups = self._make_code_review_groups([
            {"file": "src/handler.py", "line_range": [10, 11]},
        ])

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["test-model"],
            # No diff_data, no base_ref -> will try file content
        )

        # Should have called capture_file_context
        mock_capture.assert_called_once()
        assert '"source": "file"' in html
        assert '"available": true' in html

    def test_fallback_level3_anchor_text(self, sample_plan, phase_dir):
        """When no diff and no file content, falls back to anchor_text."""
        groups = self._make_code_review_groups([
            {
                "file": "/nonexistent/path/foo.py",
                "anchor_text": "def process_data():",
            },
        ])

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["test-model"],
        )

        # No diff, no file -> anchor_text fallback
        assert '"source": "anchor"' in html
        assert '"available": true' in html
        assert "process_data" in html

    def test_fallback_level4_unavailable(self, sample_plan, phase_dir):
        """When no diff, no file, no anchor_text -> source='none', available=false."""
        groups = self._make_code_review_groups([
            {"file": "/nonexistent/path/bar.py"},
            # No line_range, no anchor_text
        ])

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["test-model"],
        )

        assert '"source": "none"' in html
        assert '"available": false' in html

    @patch("utils.git_utils.capture_file_context")
    def test_fallback_chain_mixed_files(
        self, mock_capture, sample_plan, phase_dir
    ):
        """Multiple files exercise different levels of the fallback chain."""
        # capture_file_context returns lines for file_b but None for others
        def side_effect(path, line_range, **kwargs):
            if "file_b" in str(path):
                return [{"line_number": 5, "content": "content_b"}]
            return None

        mock_capture.side_effect = side_effect

        groups = self._make_code_review_groups([
            # file_a: covered by diff -> level 1
            {"file": "src/file_a.py", "line_range": [1, 5]},
            # file_b: not in diff, but capture_file_context works -> level 2
            {"file": "src/file_b.py", "line_range": [5, 10]},
            # file_c: no diff, capture fails, but has anchor_text -> level 3
            {
                "file": "src/file_c.py",
                "anchor_text": "anchor_for_c",
            },
            # file_d: nothing -> level 4
            {"file": "src/file_d.py"},
        ])

        diff_data = {
            "src/file_a.py": {
                "hunks": [], "binary": False, "deleted": False,
                "old_path": "src/file_a.py", "new_path": "src/file_a.py",
            }
        }

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["test-model"],
            diff_data=diff_data,
        )

        # Parse the embedded JSON to inspect fileContexts
        # Extract JSON between REPORT_DATA markers
        import re
        json_match = re.search(
            r'const\s+reportData\s*=\s*(\{.*?\});\s*\n',
            html,
            re.DOTALL,
        )
        # If the template substitution works, the JSON is embedded
        # Check for expected patterns in the HTML
        assert '"source": "diff"' in html  # file_a
        assert '"source": "file"' in html  # file_b
        assert '"source": "anchor"' in html  # file_c
        assert '"source": "none"' in html  # file_d

    def test_fallback_no_file_ref_skips_context(self, sample_plan, phase_dir):
        """Suggestions without fileRef do not appear in file_contexts."""
        groups = [{
            "theme": "General",
            "category": "improvement",
            "models": ["test-model"],
            "priority_score": 50,
            "validation_status": "valid",
            "validation_reason": "",
            "validation_confidence": 0.8,
            "suggestions": [{
                "title": "General suggestion",
                "desc": "No file reference",
                "importance": "LOW",
                "type": "improvement",
                "source_model": "test-model",
                # No "file" key at all
            }],
        }]

        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["test-model"],
        )

        # No fileContexts key since no files are referenced
        assert '"fileContexts"' not in html

    def test_fallback_diff_data_preempts_base_ref(self, sample_plan, phase_dir):
        """Pre-computed diff_data is used directly, base_ref is ignored."""
        groups = self._make_code_review_groups([
            {"file": "src/main.py", "line_range": [1, 10]},
        ])
        diff_data = {
            "src/main.py": {
                "hunks": [{"header": "@@ -1,3 +1,4 @@", "lines": []}],
                "binary": False, "deleted": False,
                "old_path": "src/main.py", "new_path": "src/main.py",
            }
        }

        # Even though base_ref is provided, diff_data takes precedence
        html = generate_html_report(
            groups=groups,
            plan_path=sample_plan,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["test-model"],
            base_ref="HEAD~1",  # Should be ignored
            diff_data=diff_data,
        )

        # diff_data is used -> file context from diff
        assert '"source": "diff"' in html
        assert '"diffData"' in html
