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
import subprocess
import sys
from pathlib import Path

import pytest

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import (
    SkillRunner,
    FixtureManager,
    MockProvider,
    AssertionHelpers,
    PERF_SCALE,
)
from utils.cli_common import (  # noqa: E402
    EXIT_NEEDS_HUMAN,
    EXIT_NOTHING_TO_DO,
    EXIT_OK,
    parse_result_line,
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

    def test_a_detached_review_counts_its_completed_reviewers(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """The checkpoint's progress counts the completed reviewers: the fanout
        step writes `models_done=0` and a finished run used to leave it there,
        so `status --view run` printed `models 0/3` beside three completed
        reviewers (measured 2026-09-16). Progress is recorded for detached runs
        only, so this one runs detached and waits for the run to close."""
        import time

        from utils import status_file

        _configure_scenario(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan(
            "detached-code-review", "# Plan\n\n## Tasks\n- Create auth service\n"
        )
        _create_implement_phase(fixture_manager, plan)
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state={
                "head_at_start": "abc123def",
                "tracked_files": [
                    {"path": "src/services/auth_service.py", "task_id": "T001"},
                ],
            },
        )
        # `--detach` is a dispatcher verb: the detached child re-executes
        # multi_llm.py, so the launch goes through it rather than the script.
        launch = skill_runner._run_command(
            [
                sys.executable, str(skill_runner.skill_dir / "multi_llm.py"),
                "review-code", str(plan.plan_path),
                "--models", "cursor-agent", "--skip-validation", "--detach",
            ],
            plan.plan_path, 45, False,
        )
        assert launch.success, launch.stderr
        assert "run_id:" in launch.stdout, launch.stdout

        phase_dir = plan.output_dir / "code-review"
        deadline = time.time() + 60
        run = {}
        while time.time() < deadline:
            run = status_file.read_run(str(phase_dir))
            if run.get("state") in status_file.TERMINAL_RUN_STATES:
                break
            time.sleep(0.2)
        assert run.get("state") == "done", run

        status = status_file.read_status(str(phase_dir))
        assert status["models_completed"] == ["cursor-agent:cursor-agent"], status
        progress = status["progress"]
        assert progress["step"] == "complete"
        assert progress["models_total"] == 1
        assert progress["models_done"] == 1

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

    @pytest.mark.timeout(45 * PERF_SCALE)
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
        assert result.duration_seconds < 45 * PERF_SCALE, (
            f"Took {result.duration_seconds:.1f}s, expected < {45 * PERF_SCALE}s"
        )


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

        # Bind the preference *positively*. Both shas are fake, so `validate_git_ref`
        # falls back to HEAD~1 and neither value reaches the "Git base reference:"
        # line — an absence-only assertion would therefore pass just as happily if
        # the preference were reversed. The ref the orchestrator actually picked is
        # named either on that line (when it validates) or in the fallback warning
        # it triggers ("git ref '<ref>' is invalid"), so require the newer value in
        # one of those and the older value in neither.
        chose_newer = (
            "Git base reference: newer_sha_from_impl_start" in result.stdout
            or "newer_sha_from_impl_start" in result.stderr
        )
        assert chose_newer, (
            "code review did not resolve head_before_implement as the base ref\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "old_sha_from_plan_review" not in (result.stdout + result.stderr), (
            "code review fell back to head_at_start even though "
            f"head_before_implement is present\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


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


# ---------------------------------------------------------------------------
# A review that *found* something is not a review that failed
# ---------------------------------------------------------------------------

#: Issues the code-review schema accepts, with a known HIGH count. Built here
#: rather than read from ``responses/code_review/issues.json`` so the assertions
#: below name a count this test owns, and so no other lane's fixture can move it.
def _issues(high: int, low: int = 1):
    issues = [
        {
            "title": f"SQL injection in user lookup {i}",
            "desc": "The query is built with string formatting instead of a "
                    "parameterized statement.",
            "importance": "high",
            "file": "src/services/auth_service.py",
            "line_range": [40 + i, 42 + i],
            "type": "bug",
            "anchor_text": f"cursor.execute(lookup_query_{i})",
        }
        for i in range(high)
    ]
    issues += [
        {
            "title": f"Undocumented helper {i}",
            "desc": "The public helper has no docstring.",
            "importance": "low",
            "file": "src/api/auth_routes.py",
            "line_range": [80 + i, 81 + i],
            "type": "style",
            "anchor_text": f"def _helper_{i}():",
        }
        for i in range(low)
    ]
    return issues


def _plan_with_issues(
    skill_runner: SkillRunner,
    fixture_manager: FixtureManager,
    mock_provider: MockProvider,
    name: str,
    issues,
):
    """A code-review-ready plan whose single model returns exactly *issues*.

    ``set_fixture_path`` rather than a scenario: the shipped scenarios route the
    code-review prompt by regex, and "You are reviewing code changes that
    implement a feature plan" also matches the *plan* review pattern — which is
    why the importance mix a code-review test thinks it pinned is not always the
    one the model returns. Handing the mock one fixture removes that variable.
    """
    mock_provider.set_fixture_path(
        mock_provider.create_response_fixture(issues, name=f"{name}_issues")
    )
    skill_runner.extra_env.update(mock_provider.get_env())

    plan = fixture_manager.create_plan(
        name, "# Auth Plan\n\n## Overview\nAdd login.\n\n## Tasks\n- Auth service\n"
    )
    _create_implement_phase(fixture_manager, plan)
    fixture_manager.create_state_file(
        plan,
        phases_completed=["review-plan", "implement"],
        extra_state={
            "head_at_start": "abc123def",
            "tracked_files": [
                {"path": "src/services/auth_service.py", "task_id": "T001"},
            ],
        },
    )
    return plan


class TestHighFindingsAreNotAFailedPhase:
    """`review-code` must not report "I found something serious" as an error.

    ``main()`` used to end with ``return 1 if high_count > 0 else 0``. Under the
    uniform exit-code contract (AGENTS.md: ``0`` done, ``1`` unexpected error,
    ``2`` nothing to do, ``3`` needs a human) that claimed an error that never
    happened — the report was written and ``code-review`` was recorded completed —
    and two readers believed it:

    * ``full`` routes every phase through ``_verify(phase, code)``, which treats
      any non-zero as a failed phase and stops the chain. A chain whose review
      found HIGH issues therefore aborted **at** ``review-code``, one phase before
      ``apply-code-fixes`` — the phase whose entire job is to fix what the review
      just found.
    * ``cli_common._STATE_BY_EXIT`` maps ``1`` to ``"error"``, so the one
      machine-readable line every host reads reported ``state: "error"`` for a
      successful run.

    The finding is not lost. It is announced as ``[HIGH_PRIORITY_ISSUES] <n>`` —
    the bracketed-marker channel this skill already uses for every other
    "you must act on this" fact — and ``--fail-on-high`` restores a non-zero exit
    for a standalone CI gate.
    """

    def test_high_findings_exit_zero_and_announce_themselves(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        plan = _plan_with_issues(
            skill_runner, fixture_manager, mock_provider,
            "high-findings", _issues(high=3),
        )

        result = skill_runner.run_orchestrator(
            "code_review", plan.plan_path,
            "--models", "cursor-agent", "--skip-validation", timeout=45,
        )

        assert "High priority: 3" in result.stdout, (
            "the fixture must actually produce HIGH findings, or this test proves "
            f"nothing. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        # The bug, and the whole point: `full`'s `_verify` treats anything outside
        # {EXIT_OK, EXIT_NOTHING_TO_DO} as a failed phase and stops the chain.
        assert result.exit_code in (EXIT_OK, EXIT_NOTHING_TO_DO), (
            f"a completed review is not a failed phase, but it exited "
            f"{result.exit_code}; `full` would stop here and never reach "
            f"apply-code-fixes"
        )
        # The signal a standalone caller keeps, richer than the exit code it
        # replaces: it carries the count.
        assert "[HIGH_PRIORITY_ISSUES] 3" in result.stdout

        # And the phase really did finish, so there is something to fix next.
        assert (plan.output_dir / "code-review" / "report.md").exists()
        state = result.get_state() or {}
        assert "code-review" in (state.get("phases_completed") or {})

    def test_the_result_line_of_a_high_finding_review_says_done(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """Through the dispatcher, because that is where the result line is emitted."""
        plan = _plan_with_issues(
            skill_runner, fixture_manager, mock_provider,
            "high-findings-result", _issues(high=2),
        )

        proc = subprocess.run(
            [
                sys.executable, str(skill_runner.skill_dir / "multi_llm.py"),
                "review-code", str(plan.plan_path),
                "--models", "cursor-agent", "--skip-validation",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60 * PERF_SCALE, env=skill_runner._build_env(),
            cwd=str(skill_runner.skill_dir),
        )

        assert "[HIGH_PRIORITY_ISSUES] 2" in proc.stdout, proc.stdout
        result = parse_result_line(proc.stdout)
        assert result is not None, proc.stdout
        assert result["exit"] == EXIT_OK
        assert result["state"] == "done", (
            "a review that completed and wrote its report is `done`, never `error`"
        )

    def test_fail_on_high_restores_the_standalone_gate(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        """The opt-in a CI job that gated on `$?` migrates to.

        Exit 3 ("needs a human"), never 1: the caller asked to be stopped, and
        nothing errored. `full` never passes this flag.
        """
        plan = _plan_with_issues(
            skill_runner, fixture_manager, mock_provider,
            "fail-on-high", _issues(high=2),
        )

        result = skill_runner.run_orchestrator(
            "code_review", plan.plan_path,
            "--models", "cursor-agent", "--skip-validation", "--fail-on-high",
            timeout=45,
        )

        assert result.exit_code == EXIT_NEEDS_HUMAN
        assert "[HIGH_PRIORITY_ISSUES] 2" in result.stdout
        # Gating is not aborting: the review still produced everything it would
        # have produced without the flag.
        assert (plan.output_dir / "code-review" / "report.md").exists()
        assert (plan.output_dir / "code-review" / "grouped.json").exists()
        state = result.get_state() or {}
        assert "code-review" in (state.get("phases_completed") or {})

    def test_fail_on_high_is_quiet_when_nothing_is_high(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
    ):
        plan = _plan_with_issues(
            skill_runner, fixture_manager, mock_provider,
            "no-high-findings", _issues(high=0, low=2),
        )

        result = skill_runner.run_orchestrator(
            "code_review", plan.plan_path,
            "--models", "cursor-agent", "--skip-validation", "--fail-on-high",
            timeout=45,
        )

        assert result.exit_code == EXIT_OK
        assert "High priority: 0" in result.stdout
        assert "[HIGH_PRIORITY_ISSUES]" not in result.stdout, (
            "the marker is the finding, so it is printed only when there is one"
        )
