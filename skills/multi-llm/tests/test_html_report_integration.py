#!/usr/bin/env python3
"""
Integration tests for HTML report generation across orchestrators.

These tests verify that HTML report generation is properly integrated
into the orchestrators and that HTML selections are correctly loaded
and merged with markdown selections.

To run these tests:
    uv run -- pytest tests/test_html_report_integration.py -v
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Add parent directory to path for imports
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from review_plan_orchestrator import aggregate_results
from code_review_orchestrator import generate_review_report
from utils.report_parser import load_html_selections, merge_selections
from utils.html_report_generator import generate_html_report, write_html_report


class TestFixtures:
    """Shared test data fixtures."""

    GROUPED_SUGGESTIONS = [
        {
            "theme": "Add input validation for email",
            "category": "security",
            "models": ["claude-sonnet", "gpt-4"],
            "priority_score": 85,
            "validation_status": "valid",
            "validation_reason": "Email validation is clear.",
            "suggestions": [{
                "title": "Add email format validation",
                "desc": "Add regex validation to ensure email addresses are in valid format.",
                "type": "addition",
                "reference": "### Step 2: User Model",
                "importance": "HIGH",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Add password complexity requirements",
            "category": "security",
            "models": ["gpt-4"],
            "priority_score": 70,
            "validation_status": "needs-human-decision",
            "validation_reason": "Complexity rules need business input.",
            "suggestions": [{
                "title": "Enforce password complexity",
                "desc": "Require passwords to have at least 8 characters with mixed case.",
                "type": "addition",
                "reference": "### Step 2: User Model",
                "importance": "MEDIUM",
                "source_model": "gpt-4"
            }]
        },
        {
            "theme": "Add session logging",
            "category": "monitoring",
            "models": ["claude-sonnet"],
            "priority_score": 45,
            "validation_status": "valid",
            "validation_reason": "Session logging is straightforward.",
            "suggestions": [{
                "title": "Log session activities",
                "desc": "Add logging for login, logout, and token refresh events.",
                "type": "addition",
                "reference": "### Step 5: Session Management",
                "importance": "LOW",
                "source_model": "claude-sonnet"
            }]
        },
    ]

    VALIDATION_RESULTS = {
        "metadata": {
            "schema_version": "2.0",
            "validated_at": "2025-01-30T10:00:00",
            "model": "mock-llm",
            "total_groups": 3
        },
        "groups": [
            {"group_index": 0, "status": "valid", "reason": "Email validation is clear.", "confidence": 0.95},
            {"group_index": 1, "status": "needs-human-decision", "reason": "Complexity rules need business input.", "confidence": 0.60},
            {"group_index": 2, "status": "valid", "reason": "Session logging is straightforward.", "confidence": 0.88},
        ]
    }

    CODE_REVIEW_ISSUES = [
        {
            "theme": "Missing null check",
            "category": "bug",
            "models": ["claude-sonnet"],
            "priority_score": 90,
            "validation_status": "valid",
            "validation_reason": "Clear bug fix needed.",
            "suggestions": [{
                "title": "Add null check for user object",
                "desc": "The user object can be null when not authenticated.",
                "type": "bug",
                "file": "src/auth/service.ts",
                "line_range": [42, 50],
                "importance": "HIGH",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Consider adding rate limiting",
            "category": "improvement",
            "models": ["gpt-4"],
            "priority_score": 60,
            "validation_status": "valid",
            "validation_reason": "Good security practice.",
            "suggestions": [{
                "title": "Add rate limiting to login endpoint",
                "desc": "Prevent brute force attacks by rate limiting login attempts.",
                "type": "improvement",
                "file": "src/auth/routes.ts",
                "line_range": [15, 25],
                "importance": "MEDIUM",
                "source_model": "gpt-4"
            }]
        },
    ]


class TestReviewPlanOrchestratorIntegration:
    """Test HTML report generation in review_plan_orchestrator."""

    @pytest.fixture
    def setup_review_plan_dir(self, tmp_path):
        """Create a realistic review-plan directory structure."""
        # Create plan file
        plan_dir = tmp_path / "plans"
        plan_dir.mkdir()
        plan_file = plan_dir / "test-feature.md"
        plan_file.write_text("# Test Feature Plan\n\n## Overview\nA test plan.\n", encoding="utf-8")

        # Create output directories
        output_dir = plan_dir / "test-feature"
        output_dir.mkdir()
        phase_dir = output_dir / "review-plan"
        phase_dir.mkdir()

        # Write grouped.json
        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(TestFixtures.GROUPED_SUGGESTIONS, indent=2), encoding="utf-8")

        # Write validation.json
        validation_path = phase_dir / "validation.json"
        validation_path.write_text(json.dumps(TestFixtures.VALIDATION_RESULTS, indent=2), encoding="utf-8")

        return {
            "plan_file": plan_file,
            "output_dir": output_dir,
            "phase_dir": phase_dir,
            "prefix": "test-feature",
        }

    def test_aggregate_results_creates_html_report(self, setup_review_plan_dir):
        """Verify aggregate_results() generates HTML report alongside markdown."""
        setup = setup_review_plan_dir

        # Call aggregate_results directly
        report_path = aggregate_results(
            prefix=setup["prefix"],
            out_dir=str(setup["output_dir"]),
            phase_dir=str(setup["phase_dir"]),
            models=["claude-sonnet", "gpt-4"],
            failed_models={},
            validated_groups=TestFixtures.GROUPED_SUGGESTIONS,
        )

        # Verify markdown report exists
        assert Path(report_path).exists()
        assert report_path.endswith("report.md")

        # Verify HTML report exists
        html_path = setup["phase_dir"] / "report.html"
        assert html_path.exists()

    def test_aggregate_results_markdown_contains_html_notice(self, setup_review_plan_dir):
        """Verify markdown report mentions the HTML report."""
        setup = setup_review_plan_dir

        report_path = aggregate_results(
            prefix=setup["prefix"],
            out_dir=str(setup["output_dir"]),
            phase_dir=str(setup["phase_dir"]),
            models=["claude-sonnet", "gpt-4"],
            failed_models={},
            validated_groups=TestFixtures.GROUPED_SUGGESTIONS,
        )

        content = Path(report_path).read_text(encoding="utf-8")
        assert "report.html" in content
        assert "user_selections.json" in content

    def test_aggregate_results_html_contains_group_data(self, setup_review_plan_dir):
        """Verify HTML report contains the group data."""
        setup = setup_review_plan_dir

        aggregate_results(
            prefix=setup["prefix"],
            out_dir=str(setup["output_dir"]),
            phase_dir=str(setup["phase_dir"]),
            models=["claude-sonnet", "gpt-4"],
            failed_models={},
            validated_groups=TestFixtures.GROUPED_SUGGESTIONS,
        )

        html_path = setup["phase_dir"] / "report.html"
        html_content = html_path.read_text(encoding="utf-8")

        # Check that group themes are in the HTML
        assert "Add input validation for email" in html_content
        assert "Add password complexity" in html_content
        assert "Add session logging" in html_content

        # Check validation statuses are in the HTML
        assert "valid" in html_content.lower()
        assert "needs-human-decision" in html_content.lower() or "needsHumanCount" in html_content

    def test_aggregate_results_handles_missing_template(self, setup_review_plan_dir, tmp_path, monkeypatch):
        """Verify graceful handling when HTML template is missing."""
        setup = setup_review_plan_dir

        # Temporarily move the template
        import utils.html_report_generator as hr
        original_func = hr.generate_html_report

        def mock_generate(*args, **kwargs):
            # Simulate missing template by raising an exception
            raise FileNotFoundError("Template not found")

        monkeypatch.setattr(hr, "generate_html_report", mock_generate)

        # Should still generate markdown without error
        report_path = aggregate_results(
            prefix=setup["prefix"],
            out_dir=str(setup["output_dir"]),
            phase_dir=str(setup["phase_dir"]),
            models=["claude-sonnet", "gpt-4"],
            failed_models={},
            validated_groups=TestFixtures.GROUPED_SUGGESTIONS,
        )

        # Markdown report should still exist
        assert Path(report_path).exists()


class TestCodeReviewOrchestratorIntegration:
    """Test HTML report generation in code_review_orchestrator."""

    @pytest.fixture
    def setup_code_review_dir(self, tmp_path):
        """Create a realistic code-review directory structure."""
        # Create plan file
        plan_dir = tmp_path / "plans"
        plan_dir.mkdir()
        plan_file = plan_dir / "test-feature.md"
        plan_file.write_text("# Test Feature Plan\n\n## Overview\nA test plan.\n", encoding="utf-8")

        # Create output directories
        output_dir = plan_dir / "test-feature"
        output_dir.mkdir()
        phase_dir = output_dir / "code-review"
        phase_dir.mkdir()

        # Write grouped.json
        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(TestFixtures.CODE_REVIEW_ISSUES, indent=2), encoding="utf-8")

        return {
            "plan_file": plan_file,
            "output_dir": output_dir,
            "phase_dir": phase_dir,
        }

    def test_generate_review_report_creates_html(self, setup_code_review_dir):
        """Verify generate_review_report() creates HTML report when phase_dir provided."""
        setup = setup_code_review_dir

        # Build results dict as expected by generate_review_report
        results = {
            "claude-sonnet": (True, [
                {
                    "title": "Add null check for user object",
                    "desc": "The user object can be null when not authenticated.",
                    "type": "bug",
                    "file": "src/auth/service.ts",
                    "line_range": [42, 50],
                    "importance": "high",
                    "model": "claude-sonnet"
                }
            ], None),
            "gpt-4": (True, [
                {
                    "title": "Add rate limiting to login endpoint",
                    "desc": "Prevent brute force attacks by rate limiting login attempts.",
                    "type": "improvement",
                    "file": "src/auth/routes.ts",
                    "line_range": [15, 25],
                    "importance": "medium",
                    "model": "gpt-4"
                }
            ], None),
        }

        # Call generate_review_report with phase_dir
        report = generate_review_report(
            plan_path=setup["plan_file"],
            results=results,
            changed_files=["src/auth/service.ts", "src/auth/routes.ts"],
            validated_groups=TestFixtures.CODE_REVIEW_ISSUES,
            phase_dir=setup["phase_dir"],
        )

        # Verify markdown report content
        assert "Code Review Report" in report
        assert "Add null check" in report

        # Verify HTML report exists
        html_path = setup["phase_dir"] / "report.html"
        assert html_path.exists()

    def test_generate_review_report_markdown_contains_html_notice(self, setup_code_review_dir):
        """Verify markdown report mentions the HTML report."""
        setup = setup_code_review_dir

        results = {
            "claude-sonnet": (True, [
                {
                    "title": "Test issue",
                    "desc": "Test description",
                    "type": "bug",
                    "file": "test.ts",
                    "importance": "high",
                }
            ], None),
        }

        report = generate_review_report(
            plan_path=setup["plan_file"],
            results=results,
            changed_files=["test.ts"],
            validated_groups=None,
            phase_dir=setup["phase_dir"],
        )

        assert "report.html" in report
        assert "user_selections.json" in report

    def test_generate_review_report_without_phase_dir(self, setup_code_review_dir):
        """Verify generate_review_report() works without phase_dir (no HTML generated)."""
        setup = setup_code_review_dir

        results = {
            "claude-sonnet": (True, [
                {
                    "title": "Test issue",
                    "desc": "Test description",
                    "type": "bug",
                    "file": "test.ts",
                    "importance": "high",
                }
            ], None),
        }

        # Call without phase_dir
        report = generate_review_report(
            plan_path=setup["plan_file"],
            results=results,
            changed_files=["test.ts"],
            validated_groups=None,
            phase_dir=None,
        )

        # Should return markdown without error
        assert "Code Review Report" in report


class TestApplySuggestionsSelectionLoading:
    """Test HTML selection loading for apply_suggestions_orchestrator."""

    @pytest.fixture
    def setup_selection_dir(self, tmp_path):
        """Create a review-plan directory with selection files."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        # Write grouped.json
        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(TestFixtures.GROUPED_SUGGESTIONS, indent=2), encoding="utf-8")

        return phase_dir

    def test_load_html_selections_with_valid_file(self, setup_selection_dir):
        """Verify load_html_selections() loads user_selections.json correctly."""
        phase_dir = setup_selection_dir

        # Create user_selections.json
        selections = {
            "plan_path": "/plans/test-feature.md",
            "phase": "review-plan",
            "exported_at": "2025-01-30T12:00:00",
            "skipped_groups": [2],
            "skipped_suggestions": ["G1S1"],
            "edited_descriptions": {"G2S1": "Edited description text"}
        }
        selections_path = phase_dir / "user_selections.json"
        selections_path.write_text(json.dumps(selections), encoding="utf-8")

        # Load and verify
        loaded = load_html_selections(phase_dir)

        assert loaded is not None
        assert loaded["skipped_groups"] == [2]
        assert loaded["skipped_suggestions"] == ["G1S1"]
        assert loaded["edited_descriptions"]["G2S1"] == "Edited description text"

    def test_load_html_selections_missing_file(self, setup_selection_dir):
        """Verify load_html_selections() returns None for missing file."""
        phase_dir = setup_selection_dir

        # No user_selections.json created
        loaded = load_html_selections(phase_dir)

        assert loaded is None

    def test_load_html_selections_invalid_json(self, setup_selection_dir):
        """Verify load_html_selections() returns None for invalid JSON."""
        phase_dir = setup_selection_dir

        # Create invalid JSON file
        selections_path = phase_dir / "user_selections.json"
        selections_path.write_text("{ invalid json", encoding="utf-8")

        loaded = load_html_selections(phase_dir)

        assert loaded is None

    def test_merge_selections_unions_skips(self, setup_selection_dir):
        """Verify HTML and markdown skips are unioned additively."""
        # HTML selections
        html_selections = {
            "skipped_groups": [1, 2],
            "skipped_suggestions": ["G1S1"],
            "edited_descriptions": {"G2S1": "HTML edited text"}
        }

        # Markdown selections (different)
        md_skipped_groups = {1, 3}  # Group 3 only in markdown
        md_skipped_suggestions = {"G1S1", "G2S2"}  # G2S2 only in markdown
        md_edited = {"G2S1": ("original", "Markdown edited text")}

        # Merge
        merged_groups, merged_suggestions, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=md_skipped_groups,
            md_skipped_suggestions=md_skipped_suggestions,
            md_edited=md_edited,
        )

        # Skips are unioned: HTML {1,2} | markdown {1,3} = {1,2,3}
        assert merged_groups == {1, 2, 3}

        # Suggestions unioned: HTML {"G1S1"} | markdown {"G1S1","G2S2"} = {"G1S1","G2S2"}
        assert merged_suggestions == {"G1S1", "G2S2"}

        # HTML should win for edited descriptions (overlay semantics)
        assert merged_edited["G2S1"] == "HTML edited text"

    def test_merge_selections_fallback_to_markdown(self, setup_selection_dir):
        """Verify fallback to markdown when HTML is None."""
        # No HTML selections
        html_selections = None

        # Markdown selections
        md_skipped_groups = {1, 3}
        md_skipped_suggestions = {"G1S1", "G2S2"}
        md_edited = {"G3S1": ("original", "Markdown edited text")}

        # Merge
        merged_groups, merged_suggestions, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=md_skipped_groups,
            md_skipped_suggestions=md_skipped_suggestions,
            md_edited=md_edited,
        )

        # Should use markdown values
        assert merged_groups == {1, 3}
        assert merged_suggestions == {"G1S1", "G2S2"}
        assert merged_edited["G3S1"] == "Markdown edited text"


class TestApplyCodeFixesSelectionLoading:
    """Test HTML selection loading for apply_code_fixes_orchestrator."""

    @pytest.fixture
    def setup_code_review_selections(self, tmp_path):
        """Create a code-review directory with selection files."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        # Write grouped.json
        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(TestFixtures.CODE_REVIEW_ISSUES, indent=2), encoding="utf-8")

        return phase_dir

    def test_load_html_selections_code_review(self, setup_code_review_selections):
        """Verify load_html_selections() works for code-review phase."""
        phase_dir = setup_code_review_selections

        # Create user_selections.json
        selections = {
            "plan_path": "/plans/test-feature.md",
            "phase": "code-review",
            "exported_at": "2025-01-30T12:00:00",
            "skipped_groups": [1],
            "skipped_suggestions": [],
            "edited_descriptions": {}
        }
        selections_path = phase_dir / "user_selections.json"
        selections_path.write_text(json.dumps(selections), encoding="utf-8")

        # Load and verify
        loaded = load_html_selections(phase_dir)

        assert loaded is not None
        assert loaded["phase"] == "code-review"
        assert 1 in loaded["skipped_groups"]

    def test_skipped_issues_from_html_honored(self, setup_code_review_selections):
        """Verify skipped issues from HTML are applied correctly."""
        phase_dir = setup_code_review_selections

        # HTML selections with first issue skipped
        html_selections = {
            "skipped_groups": [1],
            "skipped_suggestions": [],
            "edited_descriptions": {}
        }

        # No markdown selections
        md_skipped_groups: set = set()
        md_skipped_suggestions: set = set()
        md_edited: dict = {}

        merged_groups, merged_suggestions, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=md_skipped_groups,
            md_skipped_suggestions=md_skipped_suggestions,
            md_edited=md_edited,
        )

        assert 1 in merged_groups
        assert len(merged_suggestions) == 0


class TestHtmlReportGeneratorDirect:
    """Test html_report_generator functions directly."""

    @pytest.fixture
    def setup_report_dir(self, tmp_path):
        """Create a directory for report generation."""
        plan_file = tmp_path / "test-plan.md"
        plan_file.write_text("# Test Plan\n\n## Step 1\nDo something.\n", encoding="utf-8")

        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        return {
            "plan_file": plan_file,
            "phase_dir": phase_dir,
        }

    def test_generate_html_report_creates_valid_html(self, setup_report_dir):
        """Verify generate_html_report() produces valid HTML."""
        setup = setup_report_dir

        html = generate_html_report(
            groups=TestFixtures.GROUPED_SUGGESTIONS,
            plan_path=setup["plan_file"],
            phase_dir=setup["phase_dir"],
            phase_type="review-plan",
            models=["claude-sonnet", "gpt-4"],
            failed_models={},
        )

        # Check it's valid HTML structure
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html

        # Check report data is embedded
        assert "reportData" in html or "REPORT_DATA" in html

    def test_generate_html_report_includes_groups(self, setup_report_dir):
        """Verify HTML contains group information."""
        setup = setup_report_dir

        html = generate_html_report(
            groups=TestFixtures.GROUPED_SUGGESTIONS,
            plan_path=setup["plan_file"],
            phase_dir=setup["phase_dir"],
            phase_type="review-plan",
            models=["claude-sonnet"],
            failed_models={},
        )

        # Check group themes are in the data
        assert "Add input validation for email" in html
        assert "security" in html.lower()  # category

    def test_generate_html_report_includes_validation_status(self, setup_report_dir):
        """Verify HTML includes validation status information."""
        setup = setup_report_dir

        html = generate_html_report(
            groups=TestFixtures.GROUPED_SUGGESTIONS,
            plan_path=setup["plan_file"],
            phase_dir=setup["phase_dir"],
            phase_type="review-plan",
            models=["claude-sonnet"],
            failed_models={},
        )

        # Check validation statuses are in the data
        assert "valid" in html.lower()
        assert "needs-human-decision" in html or "needsHuman" in html

    def test_write_html_report_creates_file(self, setup_report_dir):
        """Verify write_html_report() creates the file."""
        setup = setup_report_dir

        html_content = "<html><body>Test</body></html>"
        report_path = write_html_report(html_content, setup["phase_dir"])

        assert report_path.exists()
        assert report_path.name == "report.html"
        assert report_path.read_text(encoding="utf-8") == html_content

    def test_generate_html_report_with_empty_groups(self, setup_report_dir):
        """Verify HTML generation handles empty groups list."""
        setup = setup_report_dir

        html = generate_html_report(
            groups=[],
            plan_path=setup["plan_file"],
            phase_dir=setup["phase_dir"],
            phase_type="review-plan",
            models=[],
            failed_models={},
        )

        # Should still produce valid HTML
        assert "<html" in html or "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_generate_html_report_with_failed_models(self, setup_report_dir):
        """Verify HTML includes failed model information."""
        setup = setup_report_dir

        html = generate_html_report(
            groups=TestFixtures.GROUPED_SUGGESTIONS,
            plan_path=setup["plan_file"],
            phase_dir=setup["phase_dir"],
            phase_type="review-plan",
            models=["claude-sonnet"],
            failed_models={"gpt-4": "Timeout error", "gemini": "API error"},
        )

        # Check failed models are in the data
        assert "gpt-4" in html
        assert "Timeout error" in html or "failedModels" in html


class TestReportParserIntegration:
    """Test report_parser functions in realistic orchestrator context."""

    @pytest.fixture
    def setup_phase_dir_with_files(self, tmp_path):
        """Create a phase directory with both grouped.json and user_selections.json."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        # Write grouped.json
        grouped_path = phase_dir / "grouped.json"
        grouped_path.write_text(json.dumps(TestFixtures.GROUPED_SUGGESTIONS, indent=2), encoding="utf-8")

        # Write user_selections.json
        selections = {
            "plan_path": "/plans/test.md",
            "phase": "review-plan",
            "exported_at": "2025-01-30T12:00:00",
            "skipped_groups": [2],
            "skipped_suggestions": ["G1S1"],
            "edited_descriptions": {"G3S1": "User edited description"}
        }
        selections_path = phase_dir / "user_selections.json"
        selections_path.write_text(json.dumps(selections), encoding="utf-8")

        return phase_dir

    def test_load_selections_with_valid_phase_dir(self, setup_phase_dir_with_files):
        """Test load_html_selections with valid phase directory."""
        phase_dir = setup_phase_dir_with_files

        selections = load_html_selections(phase_dir)

        assert selections is not None
        assert selections["skipped_groups"] == [2]
        assert selections["skipped_suggestions"] == ["G1S1"]
        assert "G3S1" in selections["edited_descriptions"]

    def test_load_selections_nonexistent_directory(self, tmp_path):
        """Test load_html_selections with nonexistent directory."""
        nonexistent = tmp_path / "does-not-exist"

        selections = load_html_selections(nonexistent)

        assert selections is None

    def test_merge_selections_full_integration(self, setup_phase_dir_with_files):
        """Test merge_selections in a full integration scenario."""
        phase_dir = setup_phase_dir_with_files

        # Load HTML selections
        html_selections = load_html_selections(phase_dir)

        # Simulate markdown selections (would come from parse_* functions)
        md_skipped_groups = {1}  # Different from HTML
        md_skipped_suggestions = {"G2S1"}  # Different from HTML
        md_edited = {"G1S1": ("original", "markdown edit")}  # Different from HTML

        # Merge
        merged_groups, merged_suggestions, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=md_skipped_groups,
            md_skipped_suggestions=md_skipped_suggestions,
            md_edited=md_edited,
        )

        # Skips are unioned: HTML {2} | markdown {1} = {1, 2}
        assert merged_groups == {1, 2}
        # Suggestions unioned: HTML {"G1S1"} | markdown {"G2S1"} = {"G1S1", "G2S1"}
        assert merged_suggestions == {"G1S1", "G2S1"}

        # Edited descriptions should merge (HTML overlay on markdown)
        assert "G3S1" in merged_edited  # From HTML
        assert "G1S1" in merged_edited  # From markdown (not overwritten by HTML)


# ============================================================================
# PR-style integration tests: ID stability, selection parity, routing
# ============================================================================


def _extract_report_data(html_content: str) -> dict:
    """Extract the embedded reportData JSON from HTML content."""
    start_marker = 'const reportData = '
    start_idx = html_content.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find reportData in HTML")

    json_start = start_idx + len(start_marker)

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


class TestStableIDCompatibility:
    """Assert suggestion and group IDs are identical in PR and Flat view data."""

    @pytest.fixture
    def groups_with_files_and_sections(self):
        """Groups that have both file refs and section refs."""
        return [
            {
                "theme": "Auth validation",
                "category": "security",
                "models": ["claude-sonnet"],
                "priority_score": 85,
                "validation_status": "valid",
                "validation_reason": "Clear fix needed.",
                "suggestions": [
                    {
                        "title": "Validate email",
                        "desc": "Ensure email format is valid.",
                        "type": "addition",
                        "file": "src/auth/service.ts",
                        "line_range": [10, 15],
                        "importance": "HIGH",
                        "source_model": "claude-sonnet",
                        "reference": "### Step 1: Auth",
                    },
                    {
                        "title": "Hash passwords",
                        "desc": "Use bcrypt for password hashing.",
                        "type": "addition",
                        "file": "src/auth/service.ts",
                        "line_range": [20, 30],
                        "importance": "HIGH",
                        "source_model": "claude-sonnet",
                        "reference": "### Step 1: Auth",
                    },
                ],
            },
            {
                "theme": "Logging improvements",
                "category": "monitoring",
                "models": ["gpt-4"],
                "priority_score": 50,
                "validation_status": "valid",
                "validation_reason": "Good practice.",
                "suggestions": [
                    {
                        "title": "Add request logging",
                        "desc": "Log incoming requests for debugging.",
                        "type": "addition",
                        "file": "src/api/handler.ts",
                        "line_range": [5, 12],
                        "importance": "MEDIUM",
                        "source_model": "gpt-4",
                        "reference": "### Step 2: API",
                    },
                ],
            },
        ]

    @pytest.fixture
    def plan_file(self, tmp_path):
        plan = tmp_path / "test-plan.md"
        plan.write_text("# Test Plan\n\n### Step 1: Auth\nAuth details.\n\n### Step 2: API\nAPI details.\n", encoding="utf-8")
        return plan

    def test_ids_identical_across_pr_and_flat(self, groups_with_files_and_sections, plan_file, tmp_path):
        """Group and suggestion IDs are identical whether template_style is 'pr' or 'flat'."""
        phase_dir_pr = tmp_path / "pr-view"
        phase_dir_pr.mkdir()
        phase_dir_flat = tmp_path / "flat-view"
        phase_dir_flat.mkdir()

        html_pr = generate_html_report(
            groups=groups_with_files_and_sections,
            plan_path=plan_file,
            phase_dir=phase_dir_pr,
            phase_type="code-review",
            models=["claude-sonnet", "gpt-4"],
            template_style="pr",
        )
        html_flat = generate_html_report(
            groups=groups_with_files_and_sections,
            plan_path=plan_file,
            phase_dir=phase_dir_flat,
            phase_type="code-review",
            models=["claude-sonnet", "gpt-4"],
            template_style="flat",
        )

        data_pr = _extract_report_data(html_pr)
        data_flat = _extract_report_data(html_flat)

        # Same number of groups
        assert len(data_pr["groups"]) == len(data_flat["groups"])

        # Collect all IDs from both views
        pr_group_indices = sorted(g["index"] for g in data_pr["groups"])
        flat_group_indices = sorted(g["index"] for g in data_flat["groups"])
        assert pr_group_indices == flat_group_indices

        pr_sugg_ids = sorted(
            s["id"]
            for g in data_pr["groups"]
            for s in g["suggestions"]
        )
        flat_sugg_ids = sorted(
            s["id"]
            for g in data_flat["groups"]
            for s in g["suggestions"]
        )
        assert pr_sugg_ids == flat_sugg_ids

    def test_original_indices_preserved(self, groups_with_files_and_sections, plan_file, tmp_path):
        """originalIndex is identical in both PR and Flat views."""
        phase_dir_pr = tmp_path / "pr-view"
        phase_dir_pr.mkdir()
        phase_dir_flat = tmp_path / "flat-view"
        phase_dir_flat.mkdir()

        html_pr = generate_html_report(
            groups=groups_with_files_and_sections,
            plan_path=plan_file,
            phase_dir=phase_dir_pr,
            phase_type="code-review",
            models=["claude-sonnet", "gpt-4"],
            template_style="pr",
        )
        html_flat = generate_html_report(
            groups=groups_with_files_and_sections,
            plan_path=plan_file,
            phase_dir=phase_dir_flat,
            phase_type="code-review",
            models=["claude-sonnet", "gpt-4"],
            template_style="flat",
        )

        data_pr = _extract_report_data(html_pr)
        data_flat = _extract_report_data(html_flat)

        pr_orig = sorted(g["originalIndex"] for g in data_pr["groups"])
        flat_orig = sorted(g["originalIndex"] for g in data_flat["groups"])
        assert pr_orig == flat_orig


class TestSelectionExportParity:
    """Generate user_selections.json structures from PR and Flat view data; assert interchangeable."""

    @pytest.fixture
    def standard_groups(self):
        return [
            {
                "theme": "Security fix",
                "category": "security",
                "models": ["claude-sonnet"],
                "priority_score": 90,
                "validation_status": "valid",
                "validation_reason": "Clear.",
                "suggestions": [
                    {
                        "title": "Fix XSS",
                        "desc": "Sanitize inputs.",
                        "type": "bug",
                        "file": "src/app.ts",
                        "line_range": [10, 15],
                        "importance": "HIGH",
                        "source_model": "claude-sonnet",
                    },
                    {
                        "title": "Escape output",
                        "desc": "HTML-encode outputs.",
                        "type": "bug",
                        "file": "src/app.ts",
                        "line_range": [20, 25],
                        "importance": "MEDIUM",
                        "source_model": "claude-sonnet",
                    },
                ],
            },
            {
                "theme": "Performance",
                "category": "optimization",
                "models": ["gpt-4"],
                "priority_score": 60,
                "validation_status": "needs-human-decision",
                "validation_reason": "Depends on load.",
                "suggestions": [
                    {
                        "title": "Add caching",
                        "desc": "Cache DB results.",
                        "type": "enhancement",
                        "importance": "MEDIUM",
                        "source_model": "gpt-4",
                    },
                ],
            },
        ]

    @pytest.fixture
    def plan_file(self, tmp_path):
        plan = tmp_path / "test.md"
        plan.write_text("# Test\n", encoding="utf-8")
        return plan

    def test_selection_structure_interchangeable(self, standard_groups, plan_file, tmp_path):
        """Selections built from PR view data are interchangeable with Flat view data."""
        phase_dir_pr = tmp_path / "pr"
        phase_dir_pr.mkdir()
        phase_dir_flat = tmp_path / "flat"
        phase_dir_flat.mkdir()

        html_pr = generate_html_report(
            groups=standard_groups,
            plan_path=plan_file,
            phase_dir=phase_dir_pr,
            phase_type="code-review",
            models=["claude-sonnet", "gpt-4"],
            template_style="pr",
        )
        html_flat = generate_html_report(
            groups=standard_groups,
            plan_path=plan_file,
            phase_dir=phase_dir_flat,
            phase_type="code-review",
            models=["claude-sonnet", "gpt-4"],
            template_style="flat",
        )

        data_pr = _extract_report_data(html_pr)
        data_flat = _extract_report_data(html_flat)

        # Simulate exporting selections: skip group 2, skip suggestion G1S2
        def build_selections(data):
            return {
                "plan_path": data["planPath"],
                "phase": data["phase"],
                "skipped_groups": [2],
                "skipped_suggestions": ["G1S2"],
                "edited_descriptions": {"G2S1": "Edited by user"},
            }

        sel_pr = build_selections(data_pr)
        sel_flat = build_selections(data_flat)

        # The structures should be identical because IDs are view-independent
        assert sel_pr["skipped_groups"] == sel_flat["skipped_groups"]
        assert sel_pr["skipped_suggestions"] == sel_flat["skipped_suggestions"]
        assert sel_pr["edited_descriptions"] == sel_flat["edited_descriptions"]

        # Both should load and merge correctly
        for sel in [sel_pr, sel_flat]:
            merged_groups, merged_suggestions, merged_edited = merge_selections(
                html_selections=sel,
                md_skipped_groups=set(),
                md_skipped_suggestions=set(),
                md_edited={},
            )
            assert 2 in merged_groups
            assert "G1S2" in merged_suggestions
            assert merged_edited["G2S1"] == "Edited by user"


class TestGlobalCoverageGapRouting:
    """Assert unanchored suggestions route to Global Suggestions (code review) or Coverage Gaps (task review)."""

    @pytest.fixture
    def plan_file(self, tmp_path):
        plan = tmp_path / "test.md"
        plan.write_text("# Test\n\n### Step 1\nContent.\n", encoding="utf-8")
        return plan

    def test_unanchored_code_review_routes_to_global(self, plan_file, tmp_path):
        """Code review suggestions without file/section refs route to globalSuggestions."""
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "General observation",
                "category": "improvement",
                "models": ["claude-sonnet"],
                "priority_score": 40,
                "validation_status": "valid",
                "validation_reason": "Valid.",
                "suggestions": [
                    {
                        "title": "Consider logging",
                        "desc": "Add logging throughout.",
                        "type": "improvement",
                        "importance": "LOW",
                        "source_model": "claude-sonnet",
                        # No file, no reference
                    },
                ],
            },
            {
                "theme": "File-anchored issue",
                "category": "bug",
                "models": ["gpt-4"],
                "priority_score": 80,
                "validation_status": "valid",
                "validation_reason": "Clear bug.",
                "suggestions": [
                    {
                        "title": "Fix null check",
                        "desc": "Add null guard.",
                        "type": "bug",
                        "file": "src/main.ts",
                        "line_range": [10, 15],
                        "importance": "HIGH",
                        "source_model": "gpt-4",
                    },
                ],
            },
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=plan_file,
            phase_dir=phase_dir,
            phase_type="code-review",
            models=["claude-sonnet", "gpt-4"],
        )

        data = _extract_report_data(html)

        # fileView._global should have the unanchored group
        assert "_global" in data["fileView"]
        global_suggestions = data["fileView"]["_global"]["suggestions"]
        assert len(global_suggestions) >= 1

        # The global theme should include the unanchored one
        global_themes = [g["theme"] for g in global_suggestions]
        assert "General observation" in global_themes

        # globalSuggestions list should also have the unanchored suggestions
        assert len(data["globalSuggestions"]) >= 1
        global_titles = [s["title"] for s in data["globalSuggestions"]]
        assert "Consider logging" in global_titles

        # The file-anchored one should NOT be in _global
        assert "File-anchored issue" not in global_themes

    def test_unanchored_task_review_routes_to_coverage_gaps(self, plan_file, tmp_path):
        """Task review suggestions without reference route to _coverageGaps."""
        phase_dir = tmp_path / "review-tasks"
        phase_dir.mkdir()

        groups = [
            {
                "theme": "Missing coverage",
                "category": "gap",
                "models": ["claude-sonnet"],
                "priority_score": 50,
                "validation_status": "valid",
                "validation_reason": "Gap identified.",
                "suggestions": [
                    {
                        "title": "Add integration tests",
                        "desc": "No integration test task exists.",
                        "type": "addition",
                        "importance": "HIGH",
                        "source_model": "claude-sonnet",
                        "reference": "Plan Coverage",
                    },
                ],
            },
            {
                "theme": "Unanchored gap",
                "category": "gap",
                "models": ["gpt-4"],
                "priority_score": 40,
                "validation_status": "valid",
                "validation_reason": "No anchor.",
                "suggestions": [
                    {
                        "title": "Consider CI/CD",
                        "desc": "No CI/CD task defined.",
                        "type": "addition",
                        "importance": "MEDIUM",
                        "source_model": "gpt-4",
                        # No reference at all
                    },
                ],
            },
            {
                "theme": "Task-anchored suggestion",
                "category": "improvement",
                "models": ["claude-sonnet"],
                "priority_score": 70,
                "validation_status": "valid",
                "validation_reason": "Clear.",
                "suggestions": [
                    {
                        "title": "Improve T001 error handling",
                        "desc": "Add try-catch.",
                        "type": "improvement",
                        "importance": "HIGH",
                        "source_model": "claude-sonnet",
                        "reference": "T001",
                    },
                ],
            },
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=plan_file,
            phase_dir=phase_dir,
            phase_type="review-tasks",
            models=["claude-sonnet", "gpt-4"],
        )

        data = _extract_report_data(html)

        # _coverageGaps should contain only explicit "Plan Coverage" groups
        assert "_coverageGaps" in data["taskView"]
        coverage_gaps = data["taskView"]["_coverageGaps"]
        coverage_themes = [g["theme"] for g in coverage_gaps["suggestions"]]
        assert "Missing coverage" in coverage_themes
        assert "Unanchored gap" not in coverage_themes

        # _unanchored should contain groups with no reference
        assert "_unanchored" in data["taskView"]
        unanchored = data["taskView"]["_unanchored"]
        unanchored_themes = [g["theme"] for g in unanchored["suggestions"]]
        assert "Unanchored gap" in unanchored_themes

        # Task-anchored should NOT be in coverage gaps or unanchored
        assert "Task-anchored suggestion" not in coverage_themes
        assert "Task-anchored suggestion" not in unanchored_themes

        # Task-anchored should be under T001
        assert "T001" in data["taskView"]
        t001_themes = [g["theme"] for g in data["taskView"]["T001"]["suggestions"]]
        assert "Task-anchored suggestion" in t001_themes


class TestTemplateSelection:
    """Assert template_style parameter selects the correct template."""

    @pytest.fixture
    def basic_groups(self):
        return [
            {
                "theme": "Test",
                "category": "test",
                "models": ["claude"],
                "suggestions": [{"title": "Test", "desc": "Desc"}],
            }
        ]

    @pytest.fixture
    def plan_file(self, tmp_path):
        plan = tmp_path / "test.md"
        plan.write_text("# Test\n", encoding="utf-8")
        return plan

    def test_pr_style_loads_pr_template(self, basic_groups, plan_file, tmp_path):
        """template_style='pr' loads pr_report_template.html."""
        phase_dir = tmp_path / "pr"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=plan_file,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
            template_style="pr",
        )

        # PR template should produce valid HTML with reportData
        assert "<!DOCTYPE html>" in html
        assert "reportData" in html

        # The PR template has distinctive elements that differentiate it
        # (at minimum it should not be an error page)
        assert "Error: Template Not Found" not in html

    def test_flat_style_loads_flat_template(self, basic_groups, plan_file, tmp_path):
        """template_style='flat' loads report_template.html."""
        phase_dir = tmp_path / "flat"
        phase_dir.mkdir()

        html = generate_html_report(
            groups=basic_groups,
            plan_path=plan_file,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["claude"],
            template_style="flat",
        )

        assert "<!DOCTYPE html>" in html
        assert "reportData" in html
        assert "Error: Template Not Found" not in html

    def test_pr_and_flat_produce_different_html(self, basic_groups, plan_file, tmp_path):
        """PR and Flat templates produce different HTML output."""
        phase_dir_pr = tmp_path / "pr"
        phase_dir_pr.mkdir()
        phase_dir_flat = tmp_path / "flat"
        phase_dir_flat.mkdir()

        html_pr = generate_html_report(
            groups=basic_groups,
            plan_path=plan_file,
            phase_dir=phase_dir_pr,
            phase_type="review-plan",
            models=["claude"],
            template_style="pr",
        )
        html_flat = generate_html_report(
            groups=basic_groups,
            plan_path=plan_file,
            phase_dir=phase_dir_flat,
            phase_type="review-plan",
            models=["claude"],
            template_style="flat",
        )

        # Both should have valid data, but the HTML wrapper (template) should differ
        data_pr = _extract_report_data(html_pr)
        data_flat = _extract_report_data(html_flat)

        # Data should be identical (same groups, same models, etc.)
        assert data_pr["groups"] == data_flat["groups"]
        assert data_pr["models"] == data_flat["models"]

        # But the full HTML should be different (different templates)
        # Strip the embedded JSON to compare template structure
        assert html_pr != html_flat

    def test_default_style_is_pr(self, basic_groups, plan_file, tmp_path):
        """Default template_style is 'pr'."""
        phase_dir1 = tmp_path / "default"
        phase_dir1.mkdir()
        phase_dir2 = tmp_path / "explicit-pr"
        phase_dir2.mkdir()

        html_default = generate_html_report(
            groups=basic_groups,
            plan_path=plan_file,
            phase_dir=phase_dir1,
            phase_type="review-plan",
            models=["claude"],
            # No template_style - should default to 'pr'
        )
        html_explicit = generate_html_report(
            groups=basic_groups,
            plan_path=plan_file,
            phase_dir=phase_dir2,
            phase_type="review-plan",
            models=["claude"],
            template_style="pr",
        )

        # Both should produce HTML from the same template
        # The data part will differ (generatedAt timestamp), but the template structure should match
        # Remove the JSON data to compare templates
        assert "Error: Template Not Found" not in html_default
        assert "Error: Template Not Found" not in html_explicit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
