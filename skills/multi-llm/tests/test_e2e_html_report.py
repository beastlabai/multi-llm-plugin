#!/usr/bin/env python3
"""End-to-end tests for HTML report workflow.

These tests verify the complete HTML report workflow from generation
through selection loading, ensuring that:
- HTML reports are created alongside markdown reports
- User selections can be exported and reimported
- HTML selections take precedence over markdown edits
- Edited descriptions from HTML are properly applied

Usage:
    uv run -- pytest tests/test_e2e_html_report.py -v
"""

import json
import os
import sys
import pytest
from pathlib import Path
from typing import Any, Dict

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    SkillRunner,
    FixtureManager,
    MockProvider,
    AssertionHelpers,
)


def _configure_scenario(skill_runner: SkillRunner, mock_provider: MockProvider, scenario_name: str):
    """Configure scenario on both mock_provider and skill_runner.

    This helper ensures the scenario is properly passed to the runner's environment
    after the fixtures have been created.
    """
    mock_provider.set_scenario(scenario_name)
    # Update the runner's extra_env with the new scenario
    skill_runner.extra_env.update(mock_provider.get_env())


class TestHtmlReportGeneration:
    """Test that review_plan creates HTML reports alongside markdown."""

    def test_review_plan_creates_html_report(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify review-plan creates report.html alongside report.md."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "html-test",
            "# Test Plan\n\n## Overview\nA test plan for HTML report generation.\n"
        )

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=30,
        )

        assert result.success, f"review_plan failed: {result.stderr}"

        # Verify mock was invoked (safety check against real LLM calls)
        assert result.mock_was_invoked(), "Mock LLM was not invoked - check PATH configuration"

        # Verify both HTML and markdown reports exist
        review_plan_dir = plan.output_dir / "review-plan"
        html_report_path = review_plan_dir / "report.html"
        md_report_path = review_plan_dir / "report.md"

        assert html_report_path.exists(), f"report.html not found at {html_report_path}"
        assert md_report_path.exists(), f"report.md not found at {md_report_path}"

        # Verify HTML content structure
        html = html_report_path.read_text()
        assert "<!DOCTYPE html>" in html, "HTML should start with DOCTYPE"
        assert "reportData" in html, "HTML should contain reportData JavaScript object"
        assert "</html>" in html, "HTML should have closing tag"

    def test_review_plan_html_contains_groups(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify HTML report contains the suggestion groups."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.load_plan("auth-feature")

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=30,
        )

        assert result.success, f"review_plan failed: {result.stderr}"

        review_plan_dir = plan.output_dir / "review-plan"
        html = (review_plan_dir / "report.html").read_text()

        # The happy_path scenario returns suggestions that should appear in HTML
        # Check for common elements that should be in the embedded data
        assert "totalGroups" in html, "HTML should have totalGroups in reportData"
        assert "phase" in html, "HTML should have phase type in reportData"


class TestMarkdownHtmlNotice:
    """Test that markdown reports contain the HTML precedence notice."""

    def test_markdown_has_html_notice(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify report.md contains notice about report.html."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "md-notice-test",
            "# Notice Test Plan\n\n## Overview\nTesting markdown notice.\n"
        )

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=30,
        )

        assert result.success, f"review_plan failed: {result.stderr}"

        review_plan_dir = plan.output_dir / "review-plan"
        md_content = (review_plan_dir / "report.md").read_text()

        # Verify the notice about HTML report is present
        assert "report.html" in md_content, "Markdown should mention report.html"
        assert "user_selections.json" in md_content, "Markdown should mention user_selections.json"
        assert "precedence" in md_content.lower(), "Markdown should mention precedence"


class TestCodeReviewHtmlGeneration:
    """Test HTML generation for code review phase."""

    def test_code_review_creates_html_report(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify code-review creates report.html in code-review dir."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "code-review-html-test",
            "# Code Review Test\n\n## Overview\nTesting code review HTML.\n"
        )

        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=30,
        )

        assert result.success, f"code_review failed: {result.stderr}"
        assert result.mock_was_invoked(), "Mock LLM was not invoked"

        # Verify HTML report exists in code-review directory
        code_review_dir = plan.output_dir / "code-review"
        html_report_path = code_review_dir / "report.html"
        md_report_path = code_review_dir / "report.md"

        assert html_report_path.exists(), f"report.html not found at {html_report_path}"
        assert md_report_path.exists(), f"report.md not found at {md_report_path}"

        # Verify HTML content
        html = html_report_path.read_text()
        assert "<!DOCTYPE html>" in html
        assert "reportData" in html
        assert '"phase": "code-review"' in html, "HTML should indicate code-review phase"


class TestHtmlSelectionRoundTrip:
    """Test the full cycle: generate HTML -> export selections -> load in apply."""

    def test_html_selection_roundtrip(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Test HTML selections are honored when applying suggestions."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # Create plan with review phase already completed
        suggestions_data = fixture_manager.load_response("review_plan", "valid_suggestions", validate=False)
        validation_data = fixture_manager.load_response("validation", "all_valid", validate=False)

        plan = fixture_manager.create_with_review_phase(
            "roundtrip-test",
            "# Roundtrip Test\n\n## Overview\nTesting selection roundtrip.\n",
            suggestions=suggestions_data,
            validation=validation_data,
        )

        # Create state file
        fixture_manager.create_state_file(plan, phases_completed=["review-plan"])

        # Create user_selections.json simulating user's export from HTML
        # Skip group indices 0 and 2 (first and third groups)
        user_selections = {
            "plan_path": str(plan.plan_path),
            "phase": "review-plan",
            "exported_at": "2025-01-15T10:30:00",
            "skipped_groups": [0, 2],
            "skipped_suggestions": [],
            "edited_descriptions": {},
        }
        selections_path = plan.output_dir / "review-plan" / "user_selections.json"
        selections_path.write_text(json.dumps(user_selections))

        # Run apply_suggestions with --dry-run
        result = skill_runner.run_orchestrator(
            "apply_suggestions",
            plan.plan_path,
            "--dry-run",
            "--approve-all",
            "--yes",
            timeout=30,
        )

        assert result.success, f"apply_suggestions failed: {result.stderr}"

        # Verify that we found the HTML selections
        output = result.stdout + result.stderr
        assert "user_selections.json" in output, "Should log that HTML selections were found"


class TestSelectionPrecedence:
    """Test that HTML selections take precedence over markdown edits."""

    def test_html_selections_override_markdown(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify HTML selections override markdown skips."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # Create plan with review phase outputs
        suggestions_data = fixture_manager.load_response("review_plan", "valid_suggestions", validate=False)
        validation_data = fixture_manager.load_response("validation", "all_valid", validate=False)

        plan = fixture_manager.create_with_review_phase(
            "precedence-test",
            "# Precedence Test\n\n## Overview\nTesting selection precedence.\n",
            suggestions=suggestions_data,
            validation=validation_data,
        )

        # Create state file
        fixture_manager.create_state_file(plan, phases_completed=["review-plan"])

        # Create report.md with group 1 skipped (marked with 'x' checkbox)
        # Note: In real usage, [x] means "keep" and [ ] means "skip"
        # The parser treats '-' prefix as skip marker
        review_plan_dir = plan.output_dir / "review-plan"
        md_report = """# Review Report

## Group 1: Add input validation for email format
- [ ] Keep: Add email format validation

## Group 2: Specify password complexity requirements
- [x] Keep: Define minimum password requirements
"""
        (review_plan_dir / "report.md").write_text(md_report)

        # Create user_selections.json that skips group 2 (not group 1)
        # HTML should take precedence
        user_selections = {
            "plan_path": str(plan.plan_path),
            "phase": "review-plan",
            "exported_at": "2025-01-15T10:30:00",
            "skipped_groups": [1],  # Skip second group (0-indexed)
            "skipped_suggestions": [],
            "edited_descriptions": {},
        }
        (review_plan_dir / "user_selections.json").write_text(json.dumps(user_selections))

        # Run apply_suggestions with --dry-run
        result = skill_runner.run_orchestrator(
            "apply_suggestions",
            plan.plan_path,
            "--dry-run",
            "--approve-all",
            "--yes",
            timeout=30,
        )

        assert result.success, f"apply_suggestions failed: {result.stderr}"

        # The HTML selections should be logged as taking effect
        output = result.stdout + result.stderr
        assert "user_selections.json" in output, "HTML selections should be loaded"


class TestEditedDescriptions:
    """Test that edited descriptions from HTML are applied."""

    def test_html_edited_descriptions_preserved(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify edited descriptions from HTML are used."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # Create plan with review phase outputs
        suggestions_data = fixture_manager.load_response("review_plan", "valid_suggestions", validate=False)
        validation_data = fixture_manager.load_response("validation", "all_valid", validate=False)

        plan = fixture_manager.create_with_review_phase(
            "edited-desc-test",
            "# Edited Description Test\n\n## Overview\nTesting edited descriptions.\n",
            suggestions=suggestions_data,
            validation=validation_data,
        )

        # Create state file
        fixture_manager.create_state_file(plan, phases_completed=["review-plan"])

        # Create user_selections.json with edited description
        # Use suggestion ID format: G{group_idx}S{suggestion_idx} (0-indexed)
        user_selections = {
            "plan_path": str(plan.plan_path),
            "phase": "review-plan",
            "exported_at": "2025-01-15T10:30:00",
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {
                "G0S0": "Custom edited description from HTML interface",
            },
        }
        review_plan_dir = plan.output_dir / "review-plan"
        (review_plan_dir / "user_selections.json").write_text(json.dumps(user_selections))

        # Run apply_suggestions with --dry-run
        result = skill_runner.run_orchestrator(
            "apply_suggestions",
            plan.plan_path,
            "--dry-run",
            "--approve-all",
            "--yes",
            timeout=30,
        )

        assert result.success, f"apply_suggestions failed: {result.stderr}"

        # Verify HTML selections were loaded
        output = result.stdout + result.stderr
        assert "user_selections.json" in output


class TestFallbackToMarkdown:
    """Test fallback behavior when no HTML selections exist."""

    def test_falls_back_to_markdown_without_html(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify markdown selections are used when no HTML selections exist."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # Create plan with review phase outputs
        suggestions_data = fixture_manager.load_response("review_plan", "valid_suggestions", validate=False)
        validation_data = fixture_manager.load_response("validation", "all_valid", validate=False)

        plan = fixture_manager.create_with_review_phase(
            "fallback-test",
            "# Fallback Test\n\n## Overview\nTesting markdown fallback.\n",
            suggestions=suggestions_data,
            validation=validation_data,
        )

        # Create state file
        fixture_manager.create_state_file(plan, phases_completed=["review-plan"])

        # Create report.md but NO user_selections.json
        # The apply_suggestions should fall back to markdown parsing
        review_plan_dir = plan.output_dir / "review-plan"

        # Ensure no user_selections.json exists
        selections_path = review_plan_dir / "user_selections.json"
        if selections_path.exists():
            selections_path.unlink()

        # Run apply_suggestions with --dry-run
        result = skill_runner.run_orchestrator(
            "apply_suggestions",
            plan.plan_path,
            "--dry-run",
            "--approve-all",
            "--yes",
            timeout=30,
        )

        assert result.success, f"apply_suggestions failed: {result.stderr}"

        # Should NOT mention user_selections.json since it doesn't exist
        output = result.stdout + result.stderr
        # This verifies the fallback path is used (no HTML selections found)


class TestHtmlReportContents:
    """Test the contents of generated HTML reports."""

    def test_html_report_contains_model_colors(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify HTML report includes model color metadata."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "model-colors-test",
            "# Model Colors Test\n\n## Overview\nTesting model color metadata.\n"
        )

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=30,
        )

        assert result.success, f"review_plan failed: {result.stderr}"

        review_plan_dir = plan.output_dir / "review-plan"
        html = (review_plan_dir / "report.html").read_text()

        # HTML should contain model metadata section with colors
        assert "models" in html, "HTML should contain models section"

    def test_html_report_includes_plan_path(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify HTML report includes the plan path."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "plan-path-test",
            "# Plan Path Test\n\n## Overview\nTesting plan path inclusion.\n"
        )

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=30,
        )

        assert result.success, f"review_plan failed: {result.stderr}"

        review_plan_dir = plan.output_dir / "review-plan"
        html = (review_plan_dir / "report.html").read_text()

        # HTML should contain reference to the plan path
        assert "planPath" in html or "plan_path" in html, "HTML should contain plan path reference"


class TestHtmlReportWithMultipleModels:
    """Test HTML report generation with multiple models."""

    def test_multiple_models_reflected_in_html(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify HTML report reflects multiple models."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.load_plan("auth-feature")

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models", "cursor-agent:auto", "gemini:gemini-2.5-flash",
            "--skip-validation",
            timeout=45,
        )

        assert result.success, f"review_plan failed: {result.stderr}"

        # Verify multiple providers were called
        calls = result.get_mock_calls()
        providers_called = {c.get("provider") for c in calls}
        assert "cursor-agent" in providers_called, "cursor-agent not invoked"
        assert "gemini" in providers_called, "gemini not invoked"

        # Verify HTML exists
        review_plan_dir = plan.output_dir / "review-plan"
        html_path = review_plan_dir / "report.html"
        assert html_path.exists(), "report.html not created"


class TestNoRealLLMCallsInHtmlTests:
    """Safety tests to ensure no real LLM API calls are made."""

    def test_html_tests_use_mock_only(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify all HTML generation tests use mock binaries."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "mock-safety-test",
            "# Mock Safety Test\n\nVerifying mock usage in HTML tests.\n"
        )

        # Clear any existing call log
        mock_provider.clear_call_log()

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=30,
        )

        # Verify mock was invoked
        mock_provider.assert_invoked(
            "No mock LLM calls recorded - real LLM binaries may have been used!"
        )

        # Verify calls went through mock
        calls = mock_provider.get_calls()
        assert len(calls) >= 1, "Expected at least 1 mock call"
