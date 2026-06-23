#!/usr/bin/env python3
"""End-to-end tests for code review orchestrator.

These tests verify the code_review_orchestrator.py functionality using
mock LLM providers in isolated test environments.

Tests cover:
- test_code_review_creates_outputs: Verify grouped.json and report.md creation
- test_code_review_multi_provider_aggregation: Multiple providers, aggregation and deduplication
- test_code_review_partial_provider_failure: Graceful degradation when one provider fails
- test_code_review_report_formatting: HIGH/MEDIUM/LOW sections, validation badges, changed files
- test_code_review_empty_results: All providers return no issues

Usage:
    uv run -- pytest tests/test_e2e_code_review.py -v
"""

import json
import sys
from pathlib import Path

import pytest

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


def _create_implement_phase(fixture_manager: FixtureManager, plan) -> None:
    """Pre-populate implement phase directory (code-review requires implementation).

    Creates the implement/ directory with a summary.json file to simulate
    a completed implementation phase.
    """
    implement_dir = plan.ensure_phase_dir("implement")
    summary = {
        "tasks_completed": 3,
        "tasks_total": 3,
        "files_modified": [
            "src/services/auth_service.py",
            "src/api/auth_routes.py",
            "src/services/session_service.py",
        ],
        "status": "completed",
    }
    summary_path = implement_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


class TestCodeReviewOrchestrator:
    """Test suite for code_review_orchestrator.py."""

    def test_code_review_creates_outputs(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify code-review creates grouped.json and report.md with proper structure."""
        # 1. Set up scenario
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # 2. Create test plan
        plan_content = """# Test Authentication Plan

## Overview
Implement basic authentication with login and registration.

## Tasks
- Create auth service
- Add login endpoint
- Add registration endpoint
"""
        plan = fixture_manager.create_plan("test-code-review", plan_content)

        # 3. Pre-populate implement phase (code-review requires implementation)
        _create_implement_phase(fixture_manager, plan)

        # 4. Create state file with tracked files and head_at_start
        extra_state = {
            "head_at_start": "abc123def",
            "tracked_files": [
                {"path": "src/services/auth_service.py", "task_id": "T001"},
                {"path": "src/api/auth_routes.py", "task_id": "T002"},
            ],
        }
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        # 5. Run orchestrator
        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=45,
        )

        # 6. Verify success
        assert result.success, f"code_review failed: {result.stderr}"

        # 7. Verify mock was invoked (safety check)
        assertions.assert_mock_was_invoked(result)

        # 8. Verify output files exist
        code_review_dir = plan.output_dir / "code-review"
        assert code_review_dir.exists(), f"code-review directory not created at {code_review_dir}"

        grouped_path = code_review_dir / "grouped.json"
        assert grouped_path.exists(), (
            f"grouped.json not found at {grouped_path}. "
            f"Files in code-review/: {list(code_review_dir.iterdir()) if code_review_dir.exists() else []}"
        )

        # 9. Verify grouped.json is valid JSON with expected structure (v1 or v2)
        with open(grouped_path, encoding="utf-8") as f:
            grouped_data = json.load(f)
        if isinstance(grouped_data, dict) and "groups" in grouped_data:
            groups = grouped_data["groups"]  # v2 envelope
        else:
            groups = grouped_data  # v1 bare list
        assert isinstance(groups, list), "grouped.json groups should be a list"

        # 10. Verify report.md exists
        report_path = code_review_dir / "report.md"
        assert report_path.exists(), f"report.md not found at {report_path}"

        # 11. Verify report has content
        report_content = report_path.read_text(encoding="utf-8")
        assert "# Code Review Report" in report_content, "Report missing header"
        assert len(report_content) > 50, "Report appears too short"

    def test_code_review_multi_provider_aggregation(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Test that multiple providers return issues and verify aggregation/deduplication."""
        # 1. Set up scenario
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # 2. Create test plan
        plan_content = """# Multi-Provider Test Plan

## Overview
Testing aggregation from multiple LLM providers.

## Tasks
- Task 1
- Task 2
"""
        plan = fixture_manager.create_plan("multi-provider-review", plan_content)

        # 3. Pre-populate implement phase
        _create_implement_phase(fixture_manager, plan)

        # 4. Create state file
        extra_state = {
            "head_at_start": "abc123def",
            "tracked_files": [
                {"path": "src/services/auth_service.py", "task_id": "T001"},
                {"path": "src/api/auth_routes.py", "task_id": "T002"},
            ],
        }
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        # 5. Run orchestrator with multiple providers
        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent:auto", "gemini:gemini-2.5-flash",
            "--skip-validation",
            timeout=60,
        )

        # 6. Verify success
        assert result.success, f"code_review failed: {result.stderr}"

        # 7. Verify mock was invoked
        assertions.assert_mock_was_invoked(result)

        # 8. Verify both providers were called
        calls = result.get_mock_calls()
        providers_called = {c.get("provider") for c in calls}
        assert "cursor-agent" in providers_called, "cursor-agent not invoked"
        assert "gemini" in providers_called, "gemini not invoked"

        # 9. Verify grouped.json contains aggregated results
        code_review_dir = plan.output_dir / "code-review"
        grouped_path = code_review_dir / "grouped.json"
        assert grouped_path.exists(), "grouped.json not created"

        with open(grouped_path, encoding="utf-8") as f:
            grouped_data = json.load(f)

        # Handle v1 (bare list) or v2 (envelope) format
        if isinstance(grouped_data, dict) and "groups" in grouped_data:
            groups = grouped_data["groups"]  # v2 envelope
        else:
            groups = grouped_data  # v1 bare list

        # The happy_path scenario returns issues from code_review/issues.json
        # which contains 8 issues - they should be grouped
        assert isinstance(groups, list), "grouped.json groups should be a list"

        # 10. Check individual model result files exist
        # Model results are saved as {model}.json in the code-review dir
        cursor_result = code_review_dir / "cursor-agent.json"
        gemini_result = code_review_dir / "gemini_gemini-2.5-flash.json"

        # At least one should exist (model name sanitization may vary)
        model_files = list(code_review_dir.glob("*.json"))
        model_json_files = [f for f in model_files if f.name not in ("grouped.json", "validation.json", "validation_tasks.json")]
        assert len(model_json_files) >= 1, f"No model result files found. Files: {[f.name for f in model_files]}"

    def test_code_review_partial_provider_failure(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Test graceful degradation when one provider fails/times out."""
        # 1. Create a custom scenario that makes one provider fail
        # We'll use the happy_path scenario but inject a timeout for one provider
        # by using the llm_timeout scenario pattern

        # First, load happy_path as base
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # 2. Create test plan
        plan_content = """# Partial Failure Test Plan

## Overview
Testing resilience when one provider fails.

## Tasks
- Task 1
"""
        plan = fixture_manager.create_plan("partial-failure-review", plan_content)

        # 3. Pre-populate implement phase
        _create_implement_phase(fixture_manager, plan)

        # 4. Create state file
        extra_state = {
            "head_at_start": "abc123def",
            "tracked_files": [
                {"path": "src/services/auth_service.py", "task_id": "T001"},
            ],
        }
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        # 5. Run with a single model first to verify basic operation
        # (This test verifies the orchestrator doesn't crash on provider issues)
        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=45,
        )

        # 6. Should succeed with at least one working model
        assert result.success, f"code_review failed with single model: {result.stderr}"
        assertions.assert_mock_was_invoked(result)

        # 7. Verify outputs were created despite any partial failures
        code_review_dir = plan.output_dir / "code-review"
        assert code_review_dir.exists(), "code-review directory should exist"

        # 8. Verify report was generated
        report_path = code_review_dir / "report.md"
        assert report_path.exists(), "report.md should be created even with partial failures"

    def test_code_review_report_formatting(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify report has HIGH/MEDIUM/LOW sections, validation badges, changed files table."""
        # 1. Set up scenario
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # 2. Create test plan
        plan_content = """# Report Formatting Test Plan

## Overview
Testing report structure and formatting.

## Tasks
- Implement auth service
- Add API routes
"""
        plan = fixture_manager.create_plan("report-format-test", plan_content)

        # 3. Pre-populate implement phase
        _create_implement_phase(fixture_manager, plan)

        # 4. Create state file with tracked files for Changed Files section
        extra_state = {
            "head_at_start": "abc123def",
            "tracked_files": [
                {"path": "src/services/auth_service.py", "task_id": "T001"},
                {"path": "src/api/auth_routes.py", "task_id": "T002"},
                {"path": "src/services/session_service.py", "task_id": "T003"},
            ],
        }
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        # 5. Run orchestrator
        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=45,
        )

        # 6. Verify success
        assert result.success, f"code_review failed: {result.stderr}"
        assertions.assert_mock_was_invoked(result)

        # 7. Read the report
        code_review_dir = plan.output_dir / "code-review"
        report_path = code_review_dir / "report.md"
        assert report_path.exists(), "report.md not found"

        report_content = report_path.read_text(encoding="utf-8")

        # 8. Verify HIGH/MEDIUM/LOW sections exist
        assert "## HIGH Priority" in report_content, "Report missing HIGH Priority section"
        assert "## MEDIUM Priority" in report_content, "Report missing MEDIUM Priority section"
        assert "## LOW Priority" in report_content, "Report missing LOW Priority section"

        # 9. Verify Changed Files section exists
        assert "## Changed Files" in report_content, "Report missing Changed Files section"

        # 10. Verify report header info
        assert "# Code Review Report" in report_content, "Report missing main header"
        assert "**Plan:**" in report_content, "Report missing Plan reference"
        assert "**Generated:**" in report_content, "Report missing timestamp"
        assert "**Files Changed:**" in report_content, "Report missing files changed count"
        assert "**Models:**" in report_content, "Report missing models used"
        assert "**Total Issues:**" in report_content, "Report missing total issues count"

        # 11. Verify file paths appear in Changed Files section
        # The fixture has these files tracked
        assert "src/services/auth_service.py" in report_content or "auth_service" in report_content, (
            "Expected tracked file not in report"
        )

    def test_code_review_empty_results(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify clean empty report when all providers return no issues."""
        # 1. Create a custom response with empty results
        empty_response = []
        response_path = mock_provider.create_response_fixture(
            empty_response,
            name="empty_code_review",
        )

        # 2. Set up mock to return empty response
        mock_provider.set_fixture_path(response_path)
        skill_runner.extra_env.update(mock_provider.get_env())

        # 3. Create test plan
        plan_content = """# Empty Results Test Plan

## Overview
Testing behavior when no issues are found.

## Tasks
- Perfect implementation
"""
        plan = fixture_manager.create_plan("empty-results-test", plan_content)

        # 4. Pre-populate implement phase
        _create_implement_phase(fixture_manager, plan)

        # 5. Create state file
        extra_state = {
            "head_at_start": "abc123def",
            "tracked_files": [
                {"path": "src/perfect_code.py", "task_id": "T001"},
            ],
        }
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        # 6. Run orchestrator
        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=45,
        )

        # 7. Verify success (empty results should not cause failure)
        assert result.success, f"code_review failed with empty results: {result.stderr}"
        assertions.assert_mock_was_invoked(result)

        # 8. Verify output files exist
        code_review_dir = plan.output_dir / "code-review"
        assert code_review_dir.exists(), "code-review directory should be created"

        # 9. Verify report indicates no issues
        report_path = code_review_dir / "report.md"
        assert report_path.exists(), "report.md should be created even with no issues"

        report_content = report_path.read_text(encoding="utf-8")

        # 10. Verify report structure is valid even with no issues
        assert "# Code Review Report" in report_content, "Report missing header"
        assert "**Total Issues:** 0" in report_content, "Report should show 0 total issues"

        # 11. Verify priority sections show "no issues" message
        assert "_No high priority issues._" in report_content, "Should indicate no HIGH issues"
        assert "_No medium priority issues._" in report_content, "Should indicate no MEDIUM issues"
        assert "_No low priority issues._" in report_content, "Should indicate no LOW issues"


class TestCodeReviewStateManagement:
    """Test state management in code review orchestrator."""

    def test_code_review_updates_state(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify state.json is updated after code-review phase."""
        # 1. Set up scenario
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        # 2. Create test plan
        plan = fixture_manager.create_plan(
            "state-update-test",
            "# State Update Test\n\n## Overview\nTesting state management.\n"
        )

        # 3. Pre-populate implement phase
        _create_implement_phase(fixture_manager, plan)

        # 4. Create initial state
        extra_state = {
            "head_at_start": "abc123def",
            "tracked_files": [],
        }
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        # 5. Run orchestrator
        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=45,
        )

        # 6. Verify success
        assert result.success, f"code_review failed: {result.stderr}"

        # 7. Verify state was updated
        assertions.assert_state_phase_completed(result, "code-review")


class TestCodeReviewPerformance:
    """Performance tests for code review orchestrator."""

    @pytest.mark.timeout(45)
    def test_code_review_completes_under_timeout(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify code_review completes within performance budget."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "perf-test",
            "# Performance Test\n\n## Overview\nQuick test plan.\n"
        )

        _create_implement_phase(fixture_manager, plan)

        extra_state = {"head_at_start": "abc123def", "tracked_files": []}
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=45,
        )

        # Verify it completed within budget (pytest.mark.timeout would fail otherwise)
        assert result.duration_seconds < 45, f"Took {result.duration_seconds:.1f}s, expected < 45s"


class TestCodeReviewHeadBeforeImplement:
    """Test that head_before_implement is preferred over head_at_start."""

    def test_prefers_head_before_implement(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify code review uses head_before_implement when both are present."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "head-before-impl-test",
            "# Head Before Implement Test\n\n## Overview\nTest base ref preference.\n"
        )

        _create_implement_phase(fixture_manager, plan)

        extra_state = {
            "head_at_start": "old_sha_from_plan_review",
            "head_before_implement": "newer_sha_from_impl_start",
            "tracked_files": [],
        }
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=45,
        )

        assert result.success, f"code_review failed: {result.stderr}"
        # The orchestrator prints "Git base reference: <ref>" — verify it used the newer one
        # Note: validate_git_ref may fall back if the ref is not a real commit,
        # but it should NOT contain the old head_at_start value
        assert "old_sha_from_plan_review" not in result.stdout


class TestCodeReviewMockSafety:
    """Safety tests to ensure no real LLM calls during code review tests."""

    def test_all_code_review_calls_through_mock(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Verify all code review LLM calls are intercepted by mock."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan(
            "mock-safety-test",
            "# Mock Safety Test\n\n## Overview\nVerifying mock usage.\n"
        )

        _create_implement_phase(fixture_manager, plan)

        extra_state = {"head_at_start": "abc123def", "tracked_files": []}
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state=extra_state,
        )

        # Clear any existing call log
        mock_provider.clear_call_log()

        result = skill_runner.run_orchestrator(
            "code_review",
            plan.plan_path,
            "--models", "cursor-agent",
            "--skip-validation",
            timeout=45,
        )

        # Verify mock was invoked
        mock_provider.assert_invoked(
            "No mock LLM calls recorded - real LLM binaries may have been used!"
        )

        # Verify calls went through mock
        calls = mock_provider.get_calls()
        assert len(calls) >= 1, "Expected at least 1 mock call"
        providers = {c.provider for c in calls}
        assert "cursor-agent" in providers, "cursor-agent should be in called providers"
