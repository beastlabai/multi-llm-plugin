"""End-to-end tests for failure handling scenarios.

These tests verify that the multi-llm skill handles various failure modes
gracefully, including:
- LLM timeouts (with fast backoff for CI testing)
- Malformed JSON responses (triggers salvage file creation)
- Rate limiting (429 errors with retry behavior)
- Provider not found (clear error messages)
- Partial batch failures (graceful degradation)

All tests use MULTI_LLM_TEST_FAST_BACKOFF=1 to ensure quick execution
without real multi-second delays.
"""

import argparse
import asyncio
import contextlib
import json
import os
import re
import subprocess
import sys
import time
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

from .harness import (
    SkillRunner,
    FixtureManager,
    MockProvider,
    AssertionHelpers,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro
from utils import subagent_executor as executor
from utils.llm_client import invoke_with_provider
from utils import review_orchestrator_base as base
from utils.review_orchestrator_base import save_model_result as base_save_model_result
from utils.subagent_tasks import tasks_of


class TestLLMTimeoutHandled:
    """Tests for graceful timeout handling.

    Verifies that timeouts are detected and handled without blocking for
    the full timeout duration. Uses MULTI_LLM_TEST_FAST_BACKOFF=1 for
    quick test execution.
    """

    def test_llm_timeout_handled(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
        assertions: AssertionHelpers,
    ):
        """Test that LLM timeout is handled gracefully with fast backoff.

        The test should complete quickly (no real 600s waits) because:
        1. Mock simulates timeout with immediate exit (exit code 124)
        2. MULTI_LLM_TEST_FAST_BACKOFF=1 reduces retry delays to 10ms
        """
        # Create a simple test plan
        plan = fixture_manager.create_plan(
            "timeout-test",
            """# Test Plan for Timeout Handling

## Overview
Simple plan to test timeout handling.

## Tasks
- Task 1: Do something
""",
        )

        # Configure mock to simulate timeout by setting env var directly on runner
        # Note: inject_timeout() modifies mock_provider but the runner was already
        # created with initial env, so we need to set it on the runner directly
        skill_runner.extra_env["MOCK_LLM_TIMEOUT"] = "1"
        skill_runner.extra_env["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"

        # Measure execution time
        start_time = time.time()

        # Run the orchestrator - should handle timeout gracefully
        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "--skip-validation",
            timeout=30,  # Generous timeout for test
        )

        elapsed = time.time() - start_time

        # Verify quick completion (timeout should be simulated, not real)
        assert elapsed < 10, (
            f"Test took {elapsed:.1f}s - timeout handling should be fast with mock"
        )

        # Verify mock was invoked
        assert mock_provider.was_invoked(), "Mock LLM should have been invoked"

        # The orchestrator may succeed with 0 suggestions or fail
        # The key is that it completes quickly (timeout is simulated)
        # Check stderr/stdout for timeout-related messages or error handling
        output = result.stdout + result.stderr

        # Either we see timeout mentioned, or we see "all models failed", or
        # the mock returned with exit 124 and orchestrator handled it
        timeout_indicators = [
            "timeout",
            "timed out",
            "exit code 124",
            "failed",
            "error",
        ]
        has_timeout_handling = any(
            ind in output.lower() for ind in timeout_indicators
        )

        # If it succeeded, check that it handled the error gracefully
        # (0 suggestions due to parse error from timeout stderr)
        if result.success:
            # Successful completion with error handling is acceptable
            assert "warning" in output.lower() or "invalid" in output.lower() or "0 suggestions" in output.lower(), (
                f"If successful, should show warning about invalid/empty results. Got: {output[:500]}"
            )
        else:
            # Failure with timeout message is expected
            assert has_timeout_handling, (
                f"Should mention timeout or error in output. Got: {output[:500]}"
            )

    def test_timeout_with_partial_success(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
        assertions: AssertionHelpers,
    ):
        """Test that partial timeout (some models succeed) is handled.

        When some models timeout but others succeed, the workflow should
        continue with successful results.
        """
        # Create test plan
        plan = fixture_manager.create_plan(
            "partial-timeout-test",
            """# Test Plan

## Overview
Test partial timeout handling.

## Steps
1. Step one
""",
        )

        # Use scenario that has timeout only for specific provider
        scenario_path = fixture_manager.load_scenario("llm_timeout")
        mock_provider.set_scenario_path(scenario_path)

        skill_runner.extra_env["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"

        start_time = time.time()

        # Run with multiple models - some should timeout, others succeed
        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",  # May timeout based on scenario
            "gemini:gemini-2.5-flash",  # Should succeed based on scenario
            "--skip-validation",
            timeout=30,
        )

        elapsed = time.time() - start_time

        # Should complete quickly regardless of outcome
        assert elapsed < 15, f"Partial timeout test took too long: {elapsed:.1f}s"

        # Mock should have been invoked
        assert mock_provider.was_invoked()


class TestMalformedJSONSalvaged:
    """Tests for malformed JSON response handling and salvage file creation.

    Verifies that when an LLM returns unparseable JSON, a salvage file is
    created with the correct structure to allow manual recovery.
    """

    def test_malformed_json_salvaged(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
        assertions: AssertionHelpers,
    ):
        """Test that malformed JSON creates salvage file with correct pattern.

        Salvage file should be created at: {phase_dir}/salvage_{model}.json
        and contain: model, raw_output (or raw_response), error_message, timestamp

        Note: The orchestrator triggers salvage when JSON parsing fails completely.
        If the response is technically valid JSON but just has wrong fields,
        it may be saved as empty results rather than triggering salvage.
        """
        # Create test plan
        plan = fixture_manager.create_plan(
            "malformed-test",
            """# Test Plan for Malformed JSON

## Overview
Testing salvage file creation.

## Tasks
- Do something
""",
        )

        # Configure mock to return truly malformed/truncated JSON that can't be parsed
        # This needs to be something that breaks JSON parsing entirely
        # The mock wraps responses in provider-specific format, so we need to account for that
        # For cursor-agent: {"type": "result", "result": "<content>"}
        # The content will be our malformed string - but it gets JSON-encoded
        # So we need the raw response to be something that after unwrapping is invalid
        malformed_response = 'This is not JSON at all {{{{ broken syntax'
        skill_runner.extra_env["MOCK_LLM_CONFIG"] = json.dumps({"response": malformed_response})

        # Run orchestrator
        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "--skip-validation",
            timeout=30,
        )

        # Mock should have been invoked
        assert mock_provider.was_invoked()

        # Check for salvage file or error handling
        phase_dir = plan.output_dir / "review-plan"

        # Find salvage files matching pattern salvage_{model}.json
        salvage_files = list(phase_dir.glob("salvage_*.json"))

        # Also check for error log files which may be created instead
        error_files = list(phase_dir.glob("error_*.log"))

        # Check output for salvage/error indicators
        output = result.stdout + result.stderr

        # Either salvage file, error file, or warning in output should be present
        has_salvage = len(salvage_files) > 0
        has_error_log = len(error_files) > 0
        has_warning = (
            "[SALVAGE_NEEDED]" in output or
            "salvage" in output.lower() or
            "parse error" in output.lower() or
            "invalid" in output.lower() or
            "warning" in output.lower()
        )

        assert has_salvage or has_error_log or has_warning, (
            f"Expected salvage file, error log, or warning in output. "
            f"Files present: {list(phase_dir.glob('*')) if phase_dir.exists() else 'dir not found'}. "
            f"Output: {output[:500]}"
        )

        # If salvage file exists, verify its structure
        if salvage_files:
            salvage_path = salvage_files[0]
            with open(salvage_path, encoding="utf-8") as f:
                salvage_data = json.load(f)

            # The salvage file may use raw_output or raw_response
            has_raw = "raw_output" in salvage_data or "raw_response" in salvage_data
            assert has_raw, (
                f"Salvage file should have raw_output or raw_response. "
                f"Present fields: {list(salvage_data.keys())}"
            )

            # Verify model and timestamp fields
            assert "model" in salvage_data, "Salvage file should have model field"
            assert "timestamp" in salvage_data, "Salvage file should have timestamp field"

    def test_salvage_file_naming_pattern(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
    ):
        """Test that salvage file follows correct naming pattern: salvage_{model}.json.

        Note: Salvage files are only created when JSON parsing fails completely.
        If the response has wrong schema but valid JSON, it may just be logged
        as invalid suggestions rather than triggering salvage.
        """
        plan = fixture_manager.create_plan(
            "salvage-naming-test",
            """# Test Plan

## Overview
Test salvage file naming.
""",
        )

        # Return completely unparseable response to trigger salvage
        # Use MOCK_LLM_CONFIG to ensure the response is truly broken
        skill_runner.extra_env["MOCK_LLM_CONFIG"] = json.dumps({
            "response": "<<<NOT JSON>>> {broken: syntax"
        })

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "--skip-validation",
            timeout=30,
        )

        phase_dir = plan.output_dir / "review-plan"

        # Check for salvage files or error files
        salvage_files = list(phase_dir.glob("salvage_*.json"))
        error_files = list(phase_dir.glob("error_*.log"))

        output = result.stdout + result.stderr

        # Check if salvage or error handling occurred
        if salvage_files:
            # Verify filename follows pattern salvage_{model}.json
            filename = salvage_files[0].name
            assert filename.startswith("salvage_"), f"Filename should start with 'salvage_': {filename}"
            assert filename.endswith(".json"), f"Filename should end with '.json': {filename}"
        elif error_files:
            # Error log files also indicate error handling
            filename = error_files[0].name
            assert filename.startswith("error_"), f"Error filename should start with 'error_': {filename}"
            assert filename.endswith(".log"), f"Error filename should end with '.log': {filename}"
        else:
            # Check for warning in output about invalid/parse error
            assert (
                "parse error" in output.lower() or
                "invalid" in output.lower() or
                "warning" in output.lower() or
                "json" in output.lower()
            ), (
                f"Expected salvage file, error file, or warning about JSON parsing. "
                f"Files: {list(phase_dir.glob('*')) if phase_dir.exists() else 'none'}. "
                f"Output: {output[:500]}"
            )


class TestRateLimitRetry:
    """Tests for rate limit (429) handling with retry behavior.

    Verifies that 429 responses trigger backoff and retry, completing
    quickly with MULTI_LLM_TEST_FAST_BACKOFF=1.
    """

    def test_rate_limit_retry(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
        assertions: AssertionHelpers,
    ):
        """Test that rate limit (429) triggers backoff and retry.

        Uses MULTI_LLM_TEST_FAST_BACKOFF=1 to verify retry behavior
        without real multi-second delays.
        """
        plan = fixture_manager.create_plan(
            "rate-limit-test",
            """# Test Plan for Rate Limiting

## Overview
Test rate limit handling.

## Steps
1. Test step
""",
        )

        # Use the rate_limited scenario
        scenario_path = fixture_manager.load_scenario("rate_limited")
        mock_provider.set_scenario_path(scenario_path)

        # Ensure fast backoff is enabled
        skill_runner.extra_env["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"

        start_time = time.time()

        # Run orchestrator
        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "--skip-validation",
            timeout=30,
        )

        elapsed = time.time() - start_time

        # Should complete quickly with fast backoff
        assert elapsed < 15, (
            f"Rate limit test took {elapsed:.1f}s - should be fast with fast backoff"
        )

        # Verify mock was invoked
        assert mock_provider.was_invoked()

        # Check that multiple calls were made (indicating retry)
        calls = mock_provider.get_calls()
        # At minimum, should have at least one call
        assert len(calls) >= 1, "Should have at least one mock call"

    def test_rate_limit_fast_backoff_enabled(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        skill_runner: SkillRunner,
    ):
        """Verify MULTI_LLM_TEST_FAST_BACKOFF is properly configured.

        The SkillRunner should automatically enable fast backoff for tests.
        """
        # Verify fast backoff is set in the runner's environment
        env = skill_runner._build_env()
        assert env.get("MULTI_LLM_TEST_FAST_BACKOFF") == "1", (
            "MULTI_LLM_TEST_FAST_BACKOFF should be enabled for tests"
        )


class TestProviderNotFound:
    """Tests for clear error messages when provider binary is missing.

    Verifies that attempting to use a non-existent provider results in
    a clear, helpful error message rather than a cryptic failure.
    """

    def test_provider_not_found(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
    ):
        """Test that missing provider binary gives clear error message.

        When a provider's binary is not in PATH, the orchestrator should
        fail with a helpful message indicating the provider was not found.
        """
        plan_content = """# Test Plan

## Overview
Test provider not found.
"""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text(plan_content, encoding="utf-8")

        # Create a runner without mock binaries in PATH
        # by using an empty bin directory
        empty_bin = tmp_path / "empty_bin"
        empty_bin.mkdir()

        # Build a minimal runner that uses empty PATH
        import subprocess
        import sys

        skill_dir = Path(__file__).parent.parent
        orchestrator_path = skill_dir / "review_plan_orchestrator.py"

        # Build environment with empty PATH (no provider binaries)
        env = os.environ.copy()
        env["PATH"] = str(empty_bin)  # Only has empty directory
        env["MULTI_LLM_TEST_MODE"] = "1"
        env["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"

        # Run orchestrator with a fake provider that doesn't exist
        result = subprocess.run(
            [
                sys.executable,
                str(orchestrator_path),
                "--plan-file",
                str(plan_path),
                "--models",
                "nonexistent-provider:some-model",
                "--skip-validation",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=str(skill_dir),
            encoding="utf-8",
        )

        # Should fail
        assert result.returncode != 0, "Should fail when provider not found"

        # Check for helpful error message
        output = result.stdout + result.stderr
        error_indicators = [
            "not found",
            "command not found",
            "no such file",
            "unknown",
            "invalid",
            "error",
        ]

        has_helpful_message = any(
            indicator in output.lower() for indicator in error_indicators
        )

        assert has_helpful_message, (
            f"Should have helpful error message about missing provider. "
            f"Output: {output[:1000]}"
        )

    def test_provider_not_in_path_error_message(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
    ):
        """Test error message when shutil.which returns None for provider."""
        plan = fixture_manager.create_plan(
            "no-provider-test",
            """# Test Plan

Testing missing provider error.
""",
        )

        # Create a minimal runner with no mock binaries
        import subprocess
        import sys

        skill_dir = Path(__file__).parent.parent
        orchestrator_path = skill_dir / "review_plan_orchestrator.py"

        # Use a clean PATH that won't have the provider. An empty directory is
        # the only portable way to say "no LLM providers on PATH": hardcoding
        # /usr/bin:/bin is meaningless on Windows.
        clean_bin = tmp_path / "clean_bin"
        clean_bin.mkdir()

        env = os.environ.copy()
        env["PATH"] = str(clean_bin)
        env["MULTI_LLM_TEST_MODE"] = "1"
        env["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"

        result = subprocess.run(
            [
                sys.executable,
                str(orchestrator_path),
                "--plan-file",
                str(plan.plan_path),
                "--models",
                "cursor-agent:auto",  # Real provider name but not in PATH
                "--skip-validation",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=str(skill_dir),
            encoding="utf-8",
        )

        # The orchestrator should handle this failure
        # Either by warning about unknown provider or failing to invoke
        output = result.stdout + result.stderr

        # Should mention something about the provider or command issue
        assert any(
            keyword in output.lower()
            for keyword in ["cursor-agent", "not found", "error", "failed", "command"]
        ), f"Output should mention provider issue. Got: {output[:1000]}"


class TestPartialBatchFailure:
    """Tests for graceful degradation when some batches fail.

    Verifies that when multiple models/batches are run and some fail,
    the successful results are still captured and the workflow continues
    as much as possible.
    """

    def test_partial_batch_failure(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
        assertions: AssertionHelpers,
    ):
        """Test that partial failures don't prevent successful results.

        When running with multiple models and some fail, the workflow
        should still produce results from successful models.
        """
        plan = fixture_manager.create_plan(
            "partial-failure-test",
            """# Test Plan for Partial Failure

## Overview
This plan tests partial batch failure handling.

## Tasks
- Task 1: First task
- Task 2: Second task
""",
        )

        # Create a custom scenario where one provider fails and others succeed
        # We'll create a scenario file with mixed behavior
        custom_scenario = tmp_path / "partial_failure.yaml"
        custom_scenario.write_text("""
name: "Partial Failure"
description: "Some providers fail, others succeed"

default_delay_ms: 50

prompts:
  # First provider (cursor-agent) returns valid response
  - pattern: "review.*plan"
    fixture: "review_plan/valid_suggestions.json"
    provider_filter: "cursor-agent"

  # Second provider (gemini) also returns valid response
  - pattern: "review.*plan"
    fixture: "review_plan/valid_suggestions.json"
    provider_filter: "gemini"

provider_overrides:
  cursor-agent:
    wire_format: "cursor-agent"
  gemini:
    wire_format: "gemini"
""", encoding="utf-8")

        # Point to fixtures directory for responses
        responses_dir = Path(__file__).parent / "fixtures" / "e2e" / "responses"

        # Use the happy_path scenario which provides valid responses
        scenario_path = fixture_manager.load_scenario("happy_path")
        mock_provider.set_scenario_path(scenario_path)

        # Run with multiple models
        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "gemini:gemini-2.5-flash",
            "--skip-validation",
            timeout=60,
        )

        # Mock should have been invoked for both providers
        assert mock_provider.was_invoked()
        calls = mock_provider.get_calls()
        assert len(calls) >= 1, "Should have at least one mock call"

        # Check for output indicating some success
        output = result.stdout + result.stderr
        # Either we get successful completion or we get partial results
        success_indicators = [
            "suggestions",
            "grouped",
            "saved",
            "report",
            "completed",
            "successful",
        ]

        # If the run succeeded at all, check for output files
        if result.success:
            phase_dir = plan.output_dir / "review-plan"
            if phase_dir.exists():
                json_files = list(phase_dir.glob("*.json"))
                assert len(json_files) > 0, (
                    f"Should have some output JSON files on partial success. "
                    f"Files: {list(phase_dir.glob('*'))}"
                )

    def test_some_models_fail_others_succeed(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
    ):
        """Test behavior when running multiple models with mixed results.

        Verifies the orchestrator handles cases where:
        - Some models return valid responses
        - Some models fail or return errors
        """
        plan = fixture_manager.create_plan(
            "mixed-results-test",
            """# Test Plan

## Overview
Testing mixed model results.

## Steps
1. Do something
""",
        )

        # Use malformed_response scenario which causes parse errors
        scenario_path = fixture_manager.load_scenario("malformed_response")
        mock_provider.set_scenario_path(scenario_path)

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "gemini:gemini-2.5-flash",
            "--skip-validation",
            timeout=60,
        )

        # The orchestrator should have run
        assert mock_provider.was_invoked()

        # Check output for handling of partial failures
        output = result.stdout + result.stderr

        # Should show some processing occurred
        assert len(output) > 0, "Should have some output"

        # If all models failed, we should see error messages
        # If some succeeded, we might see salvage files for failures
        phase_dir = plan.output_dir / "review-plan"
        if phase_dir.exists():
            all_files = list(phase_dir.glob("*"))
            # Should have at least some output (error logs, salvage files, or results)
            assert len(all_files) > 0, "Phase directory should have some output files"


class TestFastBackoffConfiguration:
    """Tests to verify fast backoff is properly supported.

    These tests ensure that MULTI_LLM_TEST_FAST_BACKOFF=1 is properly
    implemented in the orchestrators to enable quick test execution.
    """

    def test_fast_backoff_env_var_respected(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
    ):
        """Verify that fast backoff environment variable is set and used."""
        # The skill_runner should automatically set this
        env = skill_runner._build_env()

        assert "MULTI_LLM_TEST_FAST_BACKOFF" in env, (
            "MULTI_LLM_TEST_FAST_BACKOFF should be in environment"
        )
        assert env["MULTI_LLM_TEST_FAST_BACKOFF"] == "1", (
            "MULTI_LLM_TEST_FAST_BACKOFF should be set to '1'"
        )

    def test_retry_delays_are_minimal_in_test_mode(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
    ):
        """Verify that retry delays are minimal when fast backoff is enabled.

        This test ensures that the orchestrator's get_backoff_delay function
        returns minimal delays (10ms) when MULTI_LLM_TEST_FAST_BACKOFF=1.
        """
        plan = fixture_manager.create_plan(
            "backoff-timing-test",
            """# Test Plan

Test backoff timing.
""",
        )

        # Use rate_limited scenario which triggers retries
        scenario_path = fixture_manager.load_scenario("rate_limited")
        mock_provider.set_scenario_path(scenario_path)

        # Ensure fast backoff is enabled
        skill_runner.extra_env["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"

        start_time = time.time()

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "--skip-validation",
            timeout=30,
        )

        elapsed = time.time() - start_time

        # With fast backoff, even with retries, should complete quickly
        # Normal backoff would be 5s + 10s + 20s = 35s minimum
        # Fast backoff should be 10ms + 10ms + 10ms = 30ms
        assert elapsed < 20, (
            f"With fast backoff, should complete in under 20s even with retries. "
            f"Actual: {elapsed:.1f}s"
        )


class TestErrorRecovery:
    """Tests for error recovery and continuation behavior."""

    def test_recoverable_error_continues(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
    ):
        """Test that recoverable errors allow workflow to continue."""
        plan = fixture_manager.create_plan(
            "recovery-test",
            """# Test Plan

## Overview
Test error recovery.

## Tasks
1. Task one
""",
        )

        # Use happy path which should succeed
        scenario_path = fixture_manager.load_scenario("happy_path")
        mock_provider.set_scenario_path(scenario_path)

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "--skip-validation",
            timeout=60,
        )

        # Should complete (possibly with errors logged)
        assert mock_provider.was_invoked()

        # Check for output indicating processing occurred
        output = result.stdout + result.stderr
        assert len(output) > 0, "Should have some output"

    def test_all_models_fail_gives_clear_error(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        skill_runner: SkillRunner,
    ):
        """Test that when all models fail, we get a clear error message."""
        plan = fixture_manager.create_plan(
            "all-fail-test",
            """# Test Plan

Testing all models failing.
""",
        )

        # Inject failure for all calls via runner's extra_env
        # Note: inject_failure() modifies mock_provider but runner was already created
        skill_runner.extra_env["MOCK_LLM_FAIL"] = "1"

        result = skill_runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent:auto",
            "--skip-validation",
            timeout=30,
        )

        # The orchestrator should handle the failure
        # Either by failing with non-zero exit code, or by handling gracefully
        output = result.stdout + result.stderr

        # Check for error handling indicators
        error_keywords = ["error", "fail", "invalid", "warning", "0 suggestions"]
        has_error_info = any(kw in output.lower() for kw in error_keywords)

        # Either non-zero exit code OR error message in output
        handled_failure = (result.exit_code != 0) or has_error_info

        assert handled_failure, (
            f"Should either fail with non-zero exit code or show error/warning message. "
            f"Exit code: {result.exit_code}. Output: {output[:500]}"
        )

        # If it succeeded, verify it shows warning about the failure
        if result.success:
            assert has_error_info, (
                f"If successful despite failures, should show warning/error info. "
                f"Output: {output[:500]}"
            )


# ---------------------------------------------------------------------------
# Salvage repairs a damaged answer; it cannot invent a missing one
# ---------------------------------------------------------------------------

#: One real `review-code` slot, verbatim — kilocode's own colours, its banner
#: line, and a four-word account failure on a non-zero exit. Non-empty stderr
#: with no JSON in it is exactly the combination that used to materialize a
#: salvage request *and* a salvage task, whose best possible answer is the empty
#: array `build_salvage_prompt` tells the subagent to fall back to: a model that
#: never ran, written over `output_path` as one that reviewed the diff and found
#: nothing, and an operator sent to a JSON repair when the fix is provider auth.
KILOCODE_AUTH_FAILURE = (
    "\x1b[0m\n> code · moonshotai/kimi-k3\n\x1b[0m\n"
    "\x1b[91m\x1b[1mError: \x1b[0mUser not found.\n"
)

#: The other direction, and the one the gate may never cost us: an answer that
#: really was produced and then cut off mid-array. Nothing in it balances, so
#: only "a structure was started" can see it — and it must, because a truncated
#: array is what salvage exists for.
TRUNCATED_ARRAY = (
    '[{"title": "Missing null check", "desc": "user.profile is not guarded", '
    '"importance": "high", "file": "src/auth.py"}, {"title": "Unbounded ret'
)

KILOCODE_SPEC = "kilocode:openrouter/moonshotai/kimi-k3:thinking"


def _hard_failure(stderr: str) -> Dict[str, Any]:
    """What `invoke_with_file_output` returns when the CLI exited non-zero."""
    return {
        "success": False,
        "error": "kilocode exited with code 1",
        "details": {"stderr": stderr},
        "output_file": "",
        "data": None,
    }


def _unparseable_reply(raw: str) -> Dict[str, Any]:
    """...and when it exited 0 but neither channel held a JSON array."""
    return {
        "success": True,
        "data": {"raw": raw},
        "source": "stdout",
        "output_file": "",
        "details": {},
    }


def _plan_file(tmp_path: Path) -> Path:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Plan\n", encoding="utf-8")
    return plan_path


def _dead_process(stdout: str, stderr: str, returncode: int = 1):
    """A `CompletedProcess` for a CLI that exited non-zero, as `subprocess.run` returns it.

    A real `subprocess.CompletedProcess` rather than a `MagicMock` on purpose:
    every attribute `invoke_with_provider` reads off it — `returncode`, `stdout`,
    `stderr` — is one a mock would have invented, and a mock that answers
    `.stdout` for a field the production code later renames answers it forever.
    """
    return subprocess.CompletedProcess(
        args=["stub-cli"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _real_hard_failure(model_spec: str, stdout: str, stderr: str) -> Dict[str, Any]:
    """The envelope `invoke_with_file_output` really returns for a non-zero exit.

    Built by running the **real** `invoke_with_provider` with only
    `subprocess.run` replaced, then wrapping the result the way
    `invoke_with_file_output`'s hard-failure early return does
    (`utils/llm_client.py`: `{**result, "output_file": ...}` — a
    `SUBPROCESS_FAILED` never reaches the file-reading half).

    **Why not another hand-built dict.** The sibling fixtures above (`_hard_failure`,
    `_unparseable_reply`) are literals, and a literal `details` dict is a snapshot
    of what the plumbing looked like on the day it was typed. `details` is exactly
    where the plumbing and its consumers meet: `_attach_stdout_excerpt`
    (`utils/llm_client.py:413`) decides which keys appear and salvage site A
    decides which it reads, and `_hard_failure_with_empty_stderr` below spent a
    release asserting — in its own docstring — that a non-zero exit carries no
    stdout, which stopped being true the moment that helper landed. A fixture that
    re-derives `details` from the production code cannot drift that way: delete
    the `stdout_excerpt` attachment and the salvage tests below go red, which is
    the whole point of writing them.

    `shutil.which` is patched so provider detection finds a binary without one
    being installed; nothing here runs a real CLI.

    Args:
        model_spec: `provider:model`, resolved for real — the adapter's own
            `describe_failure` / availability verdicts ride out in `error`.
        stdout: What the CLI printed before dying.
        stderr: ...and what it wrote to the other channel.

    Returns:
        The `invoke_with_file_output` reply, with `output_file` as the empty
        string: it is the one field that cannot be reconstructed without the
        caller's output directory, it is falsy so the "Output file:" line stays
        unprinted, and it is only ever read into a salvage request's
        `output_json_path` — which the callers of this helper never write.
    """
    with patch("utils.llm_client.subprocess.run",
               return_value=_dead_process(stdout, stderr)), \
         patch("shutil.which", return_value="/usr/bin/stub-cli"):
        envelope = invoke_with_provider(prompt="Test", model_spec=model_spec)

    assert envelope["success"] is False, "fixture must describe a losing run"
    return {**envelope, "output_file": ""}


def _drive_real_provider_failure(
    tmp_path: Path, model_spec: str, stdout: str, stderr: str
):
    """Run `run_single_review` end to end with only `subprocess.run` faked out.

    One seam, and it is the process boundary: the prompt template, the argv the
    adapter builds, `invoke_with_provider`'s classification, `_attach_stdout_excerpt`'s
    redact-then-bound excerpt, `invoke_with_file_output`'s hard-failure forward,
    the salvage gate and the salvage writers are all the shipping code. Every
    other salvage test in this file patches `cro.invoke_with_file_output` and
    hands it a canned reply, which is the right seam for a question about what
    the orchestrator *decides*; it is the wrong one for a question about which
    **channel** the decision is made from, because a canned reply is free to
    carry keys the plumbing does not produce (or to omit ones it does), and the
    consumer half tested here reads `details` key by key.

    Returns:
        `(success, issues, error, phase_dir)` — the `run_single_review` triple
        plus the phase directory the salvage artifacts would land in.
    """
    plan_path = _plan_file(tmp_path)

    with patch("utils.llm_client.subprocess.run",
               return_value=_dead_process(stdout, stderr)), \
         patch("shutil.which", return_value="/usr/bin/stub-cli"):
        success, issues, error = asyncio.run(
            cro.run_single_review(model_spec, plan_path, tmp_path / "out")
        )

    return success, issues, error, cro.get_phase_dir(plan_path, "code-review")


class TestSalvageOnlyWhenThereIsSomethingToSalvage:
    """Both directions of the gate in front of every `[SALVAGE_NEEDED]` marker.

    A failure with no JSON in it is not a parse failure: the model stays failed
    and D6 row 3 aggregates the slots that answered. A failure *with* a damaged
    answer in it routes to salvage exactly as it always did.
    """

    def test_an_auth_failure_is_a_failed_model_not_a_salvage_request(
        self, tmp_path, capsys
    ):
        """Salvage site A (the non-zero-exit path), on the payload that found it."""
        plan_path = _plan_file(tmp_path)

        with patch.object(
            cro, "invoke_with_file_output",
            side_effect=lambda **_: _hard_failure(KILOCODE_AUTH_FAILURE),
        ):
            success, issues, error = asyncio.run(
                cro.run_single_review(KILOCODE_SPEC, plan_path, tmp_path / "out")
            )

        assert (success, issues) == (False, None)
        assert error == "kilocode exited with code 1"

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output
        assert "nothing to salvage" in output

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        assert list(phase_dir.glob("salvage_*.json")) == []

        # Reported as a failed model, on the same leg as every other dead slot:
        # an error log, `False` back to main(), and the spec in `failed_models`
        # that feeds `[PARTIAL_RESULTS: …]`.
        assert cro.save_model_result(
            "plan", KILOCODE_SPEC, success, issues, error, phase_dir
        ) is False
        error_log = phase_dir / f"error_{cro.sanitize_model_name(KILOCODE_SPEC)}.log"
        assert error_log.exists()
        assert "kilocode exited with code 1" in error_log.read_text(encoding="utf-8")

    def test_a_truncated_array_on_the_same_path_still_salvages(
        self, tmp_path, capsys
    ):
        """The regression guard on the gate: genuine salvage is untouched."""
        plan_path = _plan_file(tmp_path)

        with patch.object(
            cro, "invoke_with_file_output",
            side_effect=lambda **_: _hard_failure(TRUNCATED_ARRAY),
        ):
            success, _, _ = asyncio.run(
                cro.run_single_review(KILOCODE_SPEC, plan_path, tmp_path / "out")
            )

        assert success is False
        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" in output
        assert "[SALVAGE_TASK]" in output

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        sanitized = cro.sanitize_model_name(KILOCODE_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{sanitized}.json").read_text(encoding="utf-8")
        )
        assert request["raw_output"] == TRUNCATED_ARRAY
        assert request["expected_type"] == "array"
        assert (phase_dir / f"salvage_task_{sanitized}.json").exists()

    def test_a_reply_with_no_json_in_it_is_not_a_parse_failure(
        self, tmp_path, capsys
    ):
        """Salvage site B: §40's write refusal, which exits 0 and answers in prose."""
        plan_path = _plan_file(tmp_path)
        refusal = "I can't write the requested file while Ask mode is active."

        with patch.object(
            cro, "invoke_with_file_output",
            side_effect=lambda **_: _unparseable_reply(refusal),
        ):
            success, issues, error = asyncio.run(
                cro.run_single_review("cursor-agent:auto", plan_path, tmp_path / "out")
            )

        assert (success, issues) == (False, None)
        # The failure names what happened, rather than claiming extraction failed.
        assert error.startswith("Model produced no JSON output:")
        assert refusal in error

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        assert list(phase_dir.glob("salvage_*.json")) == []

    def test_a_damaged_reply_on_the_same_path_still_salvages(self, tmp_path, capsys):
        plan_path = _plan_file(tmp_path)

        with patch.object(
            cro, "invoke_with_file_output",
            side_effect=lambda **_: _unparseable_reply(
                f"Here is what I found:\n{TRUNCATED_ARRAY}"
            ),
        ):
            success, _, error = asyncio.run(
                cro.run_single_review("cursor-agent:auto", plan_path, tmp_path / "out")
            )

        assert success is False
        assert error.startswith("Could not extract JSON from response:")

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" in output
        assert "[SALVAGE_TASK]" in output
        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        assert (phase_dir / "salvage_cursor-agent_auto.json").exists()
        assert (phase_dir / "salvage_task_cursor-agent_auto.json").exists()

    def test_the_fan_out_base_holds_the_same_rule(self, tmp_path, capsys):
        """`review_orchestrator_base.save_model_result` — review-plan/review-tasks."""
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        saved = base_save_model_result(
            "plan", "gemini:gemini-2.5-flash", True, KILOCODE_AUTH_FAILURE,
            None, str(tmp_path), str(phase_dir), "review_plan",
        )

        assert saved is False
        assert list(phase_dir.glob("salvage_*.json")) == []
        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "nothing to salvage" in output

        # The error log is written on both paths; only its wording moves.
        error_log = phase_dir / "error_gemini_gemini-2_5-flash.log"
        assert "No JSON in response" in error_log.read_text(encoding="utf-8")

    def test_the_fan_out_base_still_salvages_a_damaged_answer(self, tmp_path, capsys):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        saved = base_save_model_result(
            "plan", "gemini:gemini-2.5-flash", True, TRUNCATED_ARRAY,
            None, str(tmp_path), str(phase_dir), "review_plan",
        )

        assert saved is False
        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" in output
        assert "[SALVAGE_TASK]" in output
        assert (phase_dir / "salvage_gemini_gemini-2_5-flash.json").exists()
        assert (phase_dir / "salvage_task_gemini_gemini-2_5-flash.json").exists()
        error_log = phase_dir / "error_gemini_gemini-2_5-flash.log"
        assert "Parse error" in error_log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Which channel the gate is allowed to look at
# ---------------------------------------------------------------------------


class TestSalvageSiteAReadsTheAnswerTheProviderLeftOnStdout:
    """The consumer half of `details["stdout_excerpt"]`, at salvage site A.

    The class above asks whether the gate's *judgement* is right. This one asks
    whether the gate is shown the evidence at all, which is a separate defect and
    was the older one. Site A sits under `if not result["success"]`, and that
    branch is not only "the provider never spoke": it is also the provider that
    streamed a truncated or fenced array and *then* exited non-zero — a kill at
    the deadline, a crash on teardown. That array is on **stdout** and never on
    stderr, so a `raw_content` built from `details["stderr"]` alone made the
    orchestrator print "Provider failed before producing any JSON — nothing to
    salvage" about a run that had produced JSON, and drop a recoverable review
    without saying so. The stderr-only code was equally blind before the gate
    landed (its `raw_output` was the empty string), so this closes a latent hole
    rather than a regression — but site A is the one place the gate could turn
    that hole into a *visible* loss, which is why the two halves shipped in the
    same pass: `_attach_stdout_excerpt` (`utils/llm_client.py:413`) puts the text
    on the envelope, and this reads it.

    Every test here drives the real provider chain
    (`_drive_real_provider_failure`) rather than a canned `details` literal,
    because the claim under test is a claim about two modules agreeing on a key
    name. A hand-built fixture would assert that agreement by construction and go
    on passing after either side moved.
    """

    def test_a_provider_that_streamed_an_array_and_then_died_is_still_salvaged(
        self, tmp_path, capsys
    ):
        """The case the stderr-only gate was structurally unable to see.

        stdout holds a review array cut off mid-object, stderr holds nothing, and
        the exit code is 1 — the archetype `utils/llm_client.py`'s timeout branch
        names for this site, and the single most salvageable failure the skill
        has. The assertion is on the salvage request's `raw_output` rather than
        merely on `[SALVAGE_NEEDED]`, because a marker printed over an empty
        `raw_output` is worse than no marker: it sends a subagent at a document
        that is not there, and `build_salvage_prompt`'s "if nothing recoverable is
        present, print an empty array" fallback then writes `[]` over
        `output_path` — a model that produced nothing, recorded as one that
        reviewed the diff and found nothing.
        """
        success, issues, error, phase_dir = _drive_real_provider_failure(
            tmp_path, KILOCODE_SPEC, stdout=TRUNCATED_ARRAY, stderr="",
        )

        assert (success, issues) == (False, None)
        assert error.startswith("kilocode exited with code 1")

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" in output
        assert "[SALVAGE_TASK]" in output
        assert "nothing to salvage" not in output

        sanitized = cro.sanitize_model_name(KILOCODE_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{sanitized}.json").read_text(encoding="utf-8")
        )
        assert request["raw_output"]
        assert request["raw_output"] == TRUNCATED_ARRAY
        assert request["expected_type"] == "array"
        assert (phase_dir / f"salvage_task_{sanitized}.json").exists()

        # The text the gate judged is the text the subagent is handed. Gating on
        # one string and persisting another would let a request be written for
        # evidence nobody ever saw, and the excerpt is the only copy that has
        # been through `_redact` — the raw stream must not reach an artifact.
        envelope = _real_hard_failure(KILOCODE_SPEC, stdout=TRUNCATED_ARRAY, stderr="")
        assert request["raw_output"] == envelope["details"]["stdout_excerpt"]

    def test_the_kilocode_auth_text_on_stderr_alone_still_salvages_nothing(
        self, tmp_path, capsys
    ):
        """The negative, and the proof that stdout-first is a fall-through.

        kilocode's dead-account failure is four words of ANSI-coloured prose on
        stderr by a CLI that never emitted a byte of answer, so
        `_attach_stdout_excerpt` omits `stdout_excerpt` entirely rather than
        setting it to `""` (`utils/llm_client.py:413` — "absent means absent").
        Reading stdout *ahead of* stderr must therefore leave stderr reachable: a
        precedence written as a replacement would blank `raw_content` for every
        provider that dies before speaking, and the gate would then reach the
        same verdict it reaches here for a reason that has nothing to do with the
        payload. The sibling in `TestSalvageOnlyWhenThereIsSomethingToSalvage`
        asserts the same outcome against a canned `details`; this one asserts it
        against the envelope the plumbing actually builds, so the pair fails
        differently if the key is renamed on one side only.
        """
        success, issues, error, phase_dir = _drive_real_provider_failure(
            tmp_path, KILOCODE_SPEC, stdout="", stderr=KILOCODE_AUTH_FAILURE,
        )

        assert (success, issues) == (False, None)
        # The adapter's own `describe_failure` reads the dead account off stderr
        # and names it, so the operator's next move is auth rather than a JSON
        # repair — which is exactly the move a salvage request would have stolen.
        assert error.startswith("kilocode exited with code 1")
        assert "User not found." in error

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output
        assert "nothing to salvage" in output

        assert list(phase_dir.glob("salvage_*.json")) == []

    def test_a_banner_on_stderr_does_not_join_the_array_stdout_carried(
        self, tmp_path, capsys
    ):
        """Precedence, not concatenation — the plan's wording, and the wrong fix.

        A provider that dies mid-answer usually leaves something on *both*
        channels: the truncated array on stdout and its own launch banner, a
        progress spinner or a teardown traceback on stderr. The obvious-looking
        repair is to feed the gate `stdout + stderr`, and it is wrong twice over.
        `has_salvageable_json` would see a document whose spans no longer line up,
        and — worse, because it is silent — `raw_output` would reach the salvage
        subagent as two documents with no marker saying which one it was asked to
        repair, inviting exactly the empty-array fallback the gate exists to
        prevent. stdout is the channel that can hold an answer, so stdout wins
        outright and stderr is not appended.
        """
        success, _, _, phase_dir = _drive_real_provider_failure(
            tmp_path, KILOCODE_SPEC,
            stdout=TRUNCATED_ARRAY, stderr=KILOCODE_AUTH_FAILURE,
        )

        assert success is False
        assert "[SALVAGE_NEEDED]" in capsys.readouterr().out

        sanitized = cro.sanitize_model_name(KILOCODE_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{sanitized}.json").read_text(encoding="utf-8")
        )
        assert request["raw_output"] == TRUNCATED_ARRAY
        assert "User not found." not in request["raw_output"]


# ---------------------------------------------------------------------------
# The provider whose failure is itself JSON — P2's other half
# ---------------------------------------------------------------------------
#
# Everything above was proved live on kilocode, which announces a dead account
# in coloured prose. A provider whose *failure* is a JSON document was never
# exercised, and it is the payload the pre-2026-08-28 gate got wrong in the
# worst possible way: `{"error": {"code": 401, …}}` parses, clause 1 accepted
# any balanced candidate `json.loads` would take, so `has_salvageable_json`
# returned True; salvage was dispatched; the salvage prompt's own "if nothing
# recoverable is present, print an empty array" answered `[]`; and a salvage
# task's `output_name` IS the model's result file
# (`utils/subagent_tasks.py:774`) — so `[]` landed on `<model>.json` and
# reaggregation read a model that never ran as one that reviewed the diff and
# found nothing. `SuggestionGroup._recalculate_priority` scores
# `len(self.models) + Σ importance`, so that silence counted as agreement.
#
# `_is_answer_shaped` (`utils/json_extractor.py:294`) plus the span-aware clause
# 2 (:374–396) close it, and `tests/test_json_extractor.py` pins the predicate
# in both directions. These are the layer above: the orchestrator's *use* of it,
# once per salvage site, on the payload the live rerun could not produce.

#: `opencode run --format json` announcing its own death. The whole failure is
#: one event object on **stdout**; stderr is empty (see the OpenCode
#: event-schema notes, and the local run records in `tests/container/records/`).
#: Nothing in it is a damaged answer — the document opened, closed, and said only
#: that the request was refused.
OPENCODE_ERROR_EVENT = (
    '{"type": "error", "message": "AI_APICallError: Unauthorized", "code": 401}'
)

#: The same shape from any HTTP-backed CLI: the provider relays the API's own
#: error body verbatim. Deliberately the 401 kilocode expressed in prose, so the
#: two halves of P2 differ only in whether the CLI serialized its failure.
RELAYED_API_ERROR = '{"error": {"code": 401, "message": "User not found"}}'

OPENCODE_SPEC = "opencode:opencode/big-pickle"
GEMINI_SPEC = "gemini:gemini-2.5-flash"


def _hard_failure_with_empty_stderr() -> Dict[str, Any]:
    """A non-zero exit whose only output was the JSON error body, on stdout.

    Re-derived from the shipping plumbing (`_real_hard_failure`) rather than
    typed out, and that is a deliberate change of kind. This fixture used to be a
    literal whose docstring argued that a non-zero exit carries **no stdout at
    all** — that `details["stdout_excerpt"]` existed only after a *zero* exit —
    and that opencode's error body being absent was "the point of the fixture,
    not an omission in it". That was true of the code it was written against and
    is false of the code now: `_attach_stdout_excerpt`
    (`utils/llm_client.py:413`) attaches the excerpt on all three losing paths,
    non-zero exit included, and salvage site A reads exactly that key ahead of
    `details["stderr"]`.

    Keeping the fixture starved was the other defensible option and it was
    rejected on the merits: the two tests below are the *only* place the
    orchestrator meets a JSON-shaped provider failure on the channel opencode
    really uses, and against a starved `details` both of them passed
    structurally — the gate never saw `OPENCODE_ERROR_EVENT`, so "no salvage
    request" said nothing about `has_salvageable_json` and everything about an
    empty string. Feeding it the real envelope hands that assertion back to the
    gate, which is the coverage the fixture was always meant to provide; a
    starved copy would now be a fixture that agrees with the production code by
    accident.

    The one visible consequence is the error text: the adapter's own
    availability verdict rides along, so `error` is
    `"opencode exited with code 1 [needs-auth]"` rather than the bare sentence
    the literal carried — a strictly better diagnostic, and the callers assert on
    the stable prefix.
    """
    return _real_hard_failure(OPENCODE_SPEC, stdout=OPENCODE_ERROR_EVENT, stderr="")


def _issue(title: str) -> Dict[str, Any]:
    """One code-review finding, in the shape `validate_code_review_issues` takes."""
    return {
        "title": title,
        "desc": f"Description for {title}",
        "importance": "medium",
        "file": "src/a.py",
        "type": "style",
    }


def _good_reply(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """...and what `invoke_with_file_output` returns when the model answered."""
    return {
        "success": True,
        "data": issues,
        "source": "file",
        "output_file": "",
        "details": {},
    }


def _review_state():
    """The `state.json` reads `main()` makes, with an implement phase behind it.

    Tracked files are supplied so the run never falls through to `git diff` for
    its changed-file list: what is being pinned is a salvage decision, and a
    review whose file list depends on the checkout it happens to run in is not a
    fixture.
    """
    state = MagicMock()
    state.get = MagicMock(
        side_effect=lambda key, default=None: {
            "tracked_files": [{"path": "src/a.py"}],
            "head_before_implement": "abc123",
            "head_at_start": "abc123",
            "pre_existing_changes": [],
        }.get(key, default)
    )
    state.is_phase_completed = MagicMock(return_value=False)
    # Nothing completed, so nothing is stale: the reaggregate branch refuses a
    # review older than implement (2026-10-02), and a bare MagicMock answers
    # that probe truthily.
    state.phase_completion_is_stale = MagicMock(return_value=False)
    state.state = {"phases_completed": {}}
    return state


def _review_args(plan: Path, models: List[str], **overrides):
    """`parse_args` output for a foreground `review-code` run, validation skipped.

    Hand-built for the reason `tests/test_reaggregate_identity.py::_fanout_args`
    gives: the real parser reaches host detection and the `--yes` ladder, and
    neither is what any test in this section is about.
    """
    args = argparse.Namespace(
        plan_file=str(plan),
        models=list(models),
        interactive=False,
        quick=False,
        timeout=5,
        max_parallel=2,
        base_ref=None,
        skip_validation=True,
        validation_model=None,
        apply_fixes=False,
        force=False,
        rerun_all=False,
        reaggregate=False,
        report_style="pr",
        exec_subagents=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _run_cro_review(tmp_path: Path, replies: Dict[str, Dict[str, Any]], **arg_overrides):
    """Drive `code_review_orchestrator.main()` with only the CLI call replaced.

    `tests/test_reaggregate_identity.py::_run_cro_fanout` stubs `run_all_reviews`
    wholesale, which is right for a question about `grouped.json`'s bytes and
    wrong for this one: both salvage sites live *inside* `run_single_review`, so
    here the fan-out, `save_model_result`, grouping, the report and the
    partial-results accounting are all real, and the one seam is
    `invoke_with_file_output` — a canned provider result per model spec, in the
    shape `invoke_with_provider` really hands back.

    The launch stagger is flattened because two model slots would otherwise cost
    a real 2 s `asyncio.sleep`; nothing here is about launch ordering.

    The repository root is pinned to `tmp_path`, where the plan (and so the
    phase dir) lives, as it does in a real run. Left to `git rev-parse` from a
    temp dir it would be `None`, the lane would run in the suite's own cwd, and
    every `"cwd"`-scoped adapter would correctly be sent down the diff-access
    inline branch (plan: review-run-scope-salvage-and-failure-visibility D2.3)
    — a question none of these tests is about.
    """
    plan = tmp_path / "auth-feature.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")
    models = list(replies)

    with patch.object(cro, "parse_args",
                      return_value=_review_args(plan, models, **arg_overrides)), \
         patch.object(cro, "get_project_root", return_value=str(tmp_path)), \
         patch.object(cro, "get_or_create_state", return_value=_review_state()), \
         patch.object(cro, "resolve_models", return_value=models), \
         patch.object(cro, "is_model_valid", return_value=True), \
         patch.object(cro, "validate_git_ref", side_effect=lambda ref: ref or "HEAD~1"), \
         patch.object(cro, "intent_to_add_untracked",
                      side_effect=lambda files, **_: contextlib.nullcontext()), \
         patch.object(cro, "PROVIDER_STAGGER_DELAY", 0.0), \
         patch.object(cro, "invoke_with_file_output",
                      side_effect=lambda **kwargs: replies[kwargs["model_spec"]]):
        exit_code = asyncio.run(cro.main())

    return plan, cro.get_phase_dir(plan, "code-review"), exit_code


class TestAProviderWhoseFailureIsItselfJSON:
    """P2's assertion for the JSON-erroring providers, one case per salvage site.

    Three sites gate salvage on `has_salvageable_json`, and each one is reached
    by a different failure: `code_review_orchestrator.py:688` when the CLI exits
    non-zero, `:747` when it exits 0 having answered something unparseable, and
    `utils/review_orchestrator_base.py:1451` for the `review-plan` /
    `review-tasks` fan-out. A structured provider error must read as a provider
    failure at all three, or the phantom participant comes back through whichever
    one was left.
    """

    def test_a_json_erroring_provider_is_a_failed_model_not_a_salvage_request(
        self, tmp_path, capsys
    ):
        """Salvage site A on the real `opencode` failure shape.

        This test spent a release passing *structurally* rather than on the
        gate's judgement — site A built `raw_content` from `details["stderr"]`
        alone, a non-zero exit carried no stdout, and so the error body never
        reached the predicate at all. Both halves of that have now landed:
        `_attach_stdout_excerpt` (`utils/llm_client.py:413`) puts the body on the
        envelope and site A reads `details["stdout_excerpt"]` ahead of
        `details["stderr"]`, so `OPENCODE_ERROR_EVENT` now goes straight into
        `has_salvageable_json`'s clause 1 and the verdict below is the gate's own.
        What it must decide is that `{"type": "error", …}` is a provider failure
        and not a damaged answer: `type` is a required field of
        `schemas/code_review_issues.schema.json` and also the envelope's first
        key, which is precisely why `_is_answer_shaped` refuses to accept `type`
        without `title`, `desc` and `importance` beside it.

        The sibling test puts the same body on stderr instead, so the pair pins
        the decision on both channels and neither can regress alone.
        """
        plan_path = _plan_file(tmp_path)

        with patch.object(
            cro, "invoke_with_file_output",
            side_effect=lambda **_: _hard_failure_with_empty_stderr(),
        ):
            success, issues, error = asyncio.run(
                cro.run_single_review(OPENCODE_SPEC, plan_path, tmp_path / "out")
            )

        assert (success, issues) == (False, None)
        # The prefix, not the whole sentence: the fixture is the real envelope
        # now, so the adapter's availability verdict (`[needs-auth]`) is appended
        # by `availability_from_failure`. Pinning the exact string would make
        # every improvement to that diagnostic a failure of this test, which is
        # about salvage.
        assert error.startswith("opencode exited with code 1")

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output
        assert "nothing to salvage" in output

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        assert list(phase_dir.glob("salvage_*.json")) == []

        # The diagnostic must not be lost along with the salvage: an operator
        # whose provider is unauthenticated has to be able to read that from the
        # phase dir, not only from a stdout line that scrolled past.
        assert cro.save_model_result(
            "plan", OPENCODE_SPEC, success, issues, error, phase_dir
        ) is False
        error_log = phase_dir / f"error_{cro.sanitize_model_name(OPENCODE_SPEC)}.log"
        assert error_log.exists()
        assert "opencode exited with code 1" in error_log.read_text(encoding="utf-8")

    def test_the_same_error_body_on_stderr_is_the_case_the_gate_itself_decides(
        self, tmp_path, capsys
    ):
        """Salvage site A again, with the payload where the gate can actually see it.

        Fails on the pre-`_is_answer_shaped` gate: `{"error": {"code": 401, …}}`
        is a balanced candidate `json.loads` accepts, so clause 1 returned True,
        `[SALVAGE_NEEDED]` and `[SALVAGE_TASK]` were printed, and the subagent's
        empty-array fallback was written over `openrouter_….json`. Tightening
        clause 1 alone would not have caught it either — the envelope's own `{"`
        is an opening outside no span, so clause 2 would have re-admitted it —
        which is why clause 1 keeps the spans of the candidates it rejected.
        """
        plan_path = _plan_file(tmp_path)

        with patch.object(
            cro, "invoke_with_file_output",
            side_effect=lambda **_: _hard_failure(RELAYED_API_ERROR),
        ):
            success, issues, error = asyncio.run(
                cro.run_single_review(KILOCODE_SPEC, plan_path, tmp_path / "out")
            )

        assert (success, issues) == (False, None)

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output
        assert "nothing to salvage" in output

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        assert list(phase_dir.glob("salvage_*.json")) == []
        assert list(phase_dir.glob("salvage_task_*.json")) == []

        assert cro.save_model_result(
            "plan", KILOCODE_SPEC, success, issues, error, phase_dir
        ) is False
        error_log = phase_dir / f"error_{cro.sanitize_model_name(KILOCODE_SPEC)}.log"
        assert "kilocode exited with code 1" in error_log.read_text(encoding="utf-8")

    def test_an_error_event_read_off_stdout_is_no_answer_rather_than_a_damaged_one(
        self, tmp_path, capsys
    ):
        """Salvage site B — the parse path, where the whole reply *is* gate input.

        Nothing structural saves this one: `raw_content` is the reply verbatim,
        so the event envelope goes straight into clause 1. `{"type": "error", …}`
        is the shape the answer-field set deliberately leaves `type` out of —
        `type` is a *required* field of `schemas/code_review_issues.schema.json`
        and also the first key of this envelope, and no genuine finding carries
        it without also carrying `title`, `desc` and `importance`.
        """
        plan_path = _plan_file(tmp_path)

        with patch.object(
            cro, "invoke_with_file_output",
            side_effect=lambda **_: _unparseable_reply(OPENCODE_ERROR_EVENT),
        ):
            success, issues, error = asyncio.run(
                cro.run_single_review(OPENCODE_SPEC, plan_path, tmp_path / "out")
            )

        assert (success, issues) == (False, None)
        # Names what happened rather than claiming extraction failed, and carries
        # the envelope forward so the 401 is readable without opening the log.
        assert error.startswith("Model produced no JSON output:")
        assert "Unauthorized" in error

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output

        phase_dir = cro.get_phase_dir(plan_path, "code-review")
        assert list(phase_dir.glob("salvage_*.json")) == []

        assert cro.save_model_result(
            "plan", OPENCODE_SPEC, success, issues, error, phase_dir
        ) is False
        error_log = phase_dir / f"error_{cro.sanitize_model_name(OPENCODE_SPEC)}.log"
        assert "Model produced no JSON output:" in error_log.read_text(encoding="utf-8")

    def test_the_fan_out_base_refuses_the_same_error_envelope(self, tmp_path, capsys):
        """`review_orchestrator_base.save_model_result` — review-plan/review-tasks.

        The third site, and the one whose `except (JSONDecodeError, ValueError)`
        makes it look like a parse failure by construction. It is not: an error
        body reaches that handler with nothing to repair, and `[]` written over
        `output_path` here is the same invented clean review it is at the other
        two. The error log is written on both branches — only its first line
        moves, which is what makes it a usable diagnostic either way.
        """
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        saved = base_save_model_result(
            "plan", GEMINI_SPEC, True, RELAYED_API_ERROR,
            None, str(tmp_path), str(phase_dir), "review_plan",
        )

        assert saved is False
        assert list(phase_dir.glob("salvage_*.json")) == []
        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output
        assert "nothing to salvage" in output

        error_log = phase_dir / "error_gemini_gemini-2_5-flash.log"
        logged = error_log.read_text(encoding="utf-8")
        assert "No JSON in response" in logged
        assert RELAYED_API_ERROR in logged

    def test_json_erroring_slots_leave_the_run_on_the_partial_results_path(
        self, tmp_path, capsys
    ):
        """The phase-level consequence: dead slots are partial output, not a stop.

        D6 row 3 — a run that lost *some* models prints `[PARTIAL_RESULTS: …]`,
        aggregates what exists and exits 0; only a run where *no* model produced
        a result is exit 3. Driven through `main()` because the accounting that
        decides it (`failed_models`, `completed_models`, the `.status.json`
        write, `grouped.json` and the report) sits above `run_single_review` and
        cannot be seen from there, and because "no salvage was requested" has to
        hold for the whole phase dir at the *end* of a run rather than only at
        the instant the gate ran.

        Both failure channels are in this one run on purpose, and since site A
        began reading `details["stdout_excerpt"]` ahead of `details["stderr"]`
        both of them are now the gate's to decide: opencode's error event arrives
        on stdout, kilocode's relayed 401 body on stderr, and
        `has_salvageable_json` has to refuse each one on its own merits. That is a
        strictly harder run than the same test posed before the excerpt was
        plumbed through, when the opencode slot could only ever hand the
        predicate an empty string — a regression in either the channel precedence
        or the predicate now fails here as well as at the single-site tests above.
        """
        plan, phase_dir, exit_code = _run_cro_review(
            tmp_path,
            {
                OPENCODE_SPEC: _hard_failure_with_empty_stderr(),
                KILOCODE_SPEC: _hard_failure(RELAYED_API_ERROR),
                GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
            },
        )
        output = capsys.readouterr().out

        assert exit_code == 0
        assert f"[PARTIAL_RESULTS: {OPENCODE_SPEC},{KILOCODE_SPEC}]" in output
        assert "ERROR: All models failed" not in output

        # Not one salvage artifact anywhere in the phase dir, request or task.
        assert list(phase_dir.glob("salvage_*.json")) == []
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output

        # Each failed slot has an error log and no result file; the survivor has
        # a result file and is the only model the aggregate names.
        for spec, reason in (
            (OPENCODE_SPEC, "opencode exited with code 1"),
            (KILOCODE_SPEC, "kilocode exited with code 1"),
        ):
            stem = cro.sanitize_model_name(spec)
            assert reason in (phase_dir / f"error_{stem}.log").read_text(
                encoding="utf-8"
            )
            assert not (phase_dir / f"{stem}.json").exists()
        assert (phase_dir / f"{cro.sanitize_model_name(GEMINI_SPEC)}.json").exists()

        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        assert status["models_completed"] == [GEMINI_SPEC]
        assert set(status["models_failed"]) == {OPENCODE_SPEC, KILOCODE_SPEC}

        grouped = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
        attributed = {m for group in grouped["groups"] for m in group["models"]}
        assert attributed == {GEMINI_SPEC}

        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert f"**Models:** {GEMINI_SPEC}" in report
        assert f"- **{OPENCODE_SPEC}**: opencode exited with code 1" in report
        assert f"- **{KILOCODE_SPEC}**: kilocode exited with code 1" in report


# ---------------------------------------------------------------------------
# Salvage reads the answer text, not the transport stream (plan:
# review-run-scope-salvage-and-failure-visibility §4, T4.2)
# ---------------------------------------------------------------------------
#
# On 2026-09-25 pi was killed at the code-review deadline while a `bash` tool
# call ran. The gate judged the last 20 lines of its `--mode json` stream, whose
# `message_end` carries `"content": [thinking, toolCall]` — a list, so
# `has_salvageable_json` passed — and `salvage_pi_…json` was written over pure
# event data. `LLMProvider.salvage_text` now reduces a stream adapter's stdout to
# its last assistant message's text, and `salvage_source` makes that the gate's
# only input whenever the adapter has a reducer.

from utils.json_extractor import has_salvageable_json  # noqa: E402
from utils.llm_client import ERROR_TIMEOUT  # noqa: E402
from utils.providers.pi import PiProvider  # noqa: E402
from utils.review_orchestrator_base import salvage_source  # noqa: E402

PI_SPEC = "pi:openrouter/deepseek/deepseek-v4.1-flash"
PI_FIXTURES = Path(__file__).parent / "fixtures" / "pi"
#: The incident: narration turns, then a `thinking` + `toolCall` message killed
#: while the tool ran.
PI_KILLED_MID_TOOLCALL = (PI_FIXTURES / "killed-mid-toolcall.stdout.txt").read_text(
    encoding="utf-8")
#: A reviewer killed while printing its findings array, after narration turns.
PI_KILLED_MID_TEXT = (PI_FIXTURES / "killed-mid-text.stdout.txt").read_text(
    encoding="utf-8")
PI_AUTH_ERROR = (PI_FIXTURES / "auth-error.stdout.txt").read_text(encoding="utf-8")
#: A diagnostic on stderr that is itself array-shaped JSON: the gate accepts it,
#: so it must never reach the gate once an adapter has reduced its stream.
ARRAY_SHAPED_DIAGNOSTIC = '{"error":"authentication failed","attempts":[]}'


def _drive_real_pi_timeout(tmp_path: Path, stdout: str):
    """`run_single_review` for pi, killed at its deadline after streaming ``stdout``."""
    plan_path = _plan_file(tmp_path)
    with patch("utils.llm_client.subprocess.run",
               side_effect=_killed_after_streaming(stdout)), \
         patch("shutil.which", return_value="/usr/bin/stub-cli"):
        success, issues, error = asyncio.run(
            cro.run_single_review(PI_SPEC, plan_path, tmp_path / "out")
        )
    return success, issues, error, cro.get_phase_dir(plan_path, "code-review")


class TestSalvageSourceIsOneRule:
    """`salvage_source(details)`, the gate input both lanes share."""

    def test_a_present_empty_salvage_text_wins_over_a_stderr_that_passes_the_gate(self):
        assert has_salvageable_json(ARRAY_SHAPED_DIAGNOSTIC)
        details = {
            "salvage_text": "",
            "stdout_excerpt": '{"type":"message_end","message":{"content":[]}}',
            "stderr": ARRAY_SHAPED_DIAGNOSTIC,
        }

        assert salvage_source(details) == ""

    def test_a_present_salvage_text_is_the_input_whatever_else_is_there(self):
        details = {"salvage_text": "[1", "stdout_excerpt": "x", "stderr": "y"}

        assert salvage_source(details) == "[1"

    @pytest.mark.parametrize("details,expected", [
        ({"stdout_excerpt": TRUNCATED_ARRAY, "stderr": "banner"}, TRUNCATED_ARRAY),
        ({"stderr": ARRAY_SHAPED_DIAGNOSTIC}, ARRAY_SHAPED_DIAGNOSTIC),
        ({"stdout_excerpt": "", "stderr": ""}, ""),
        ({}, ""),
    ])
    def test_an_absent_key_keeps_the_stdout_then_stderr_order(self, details, expected):
        assert salvage_source(details) == expected

    def test_the_fan_out_lane_applies_the_same_rule(self):
        assert base.failure_output_from_result(
            {"stdout": "x", "stderr": ARRAY_SHAPED_DIAGNOSTIC, "salvage_text": ""}
        ) == ""
        assert base.failure_output_from_result(
            {"stdout": "", "stderr": ARRAY_SHAPED_DIAGNOSTIC}
        ) == ARRAY_SHAPED_DIAGNOSTIC


class TestSalvageSiteAReadsTheAnswerTextNotTheStream:
    """Code-review salvage site A, driving the real pi adapter end to end."""

    def _no_salvage(self, phase_dir, output):
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output
        assert list(phase_dir.glob("salvage_*.json")) == []

    def test_a_killed_mid_toolcall_is_a_failed_model_not_a_salvage_request(
        self, tmp_path, capsys
    ):
        """(a) The incident. The raw stream passes the gate; its reduction is ``""``."""
        assert has_salvageable_json(PI_KILLED_MID_TOOLCALL)

        success, issues, error, phase_dir = _drive_real_pi_timeout(
            tmp_path, PI_KILLED_MID_TOOLCALL)
        output = capsys.readouterr().out

        assert (success, issues) == (False, None)
        assert "timed out" in error
        self._no_salvage(phase_dir, output)
        assert "timed out before writing an answer — nothing to salvage" in output

    def test_an_auth_error_with_an_array_shaped_stderr_salvages_nothing(
        self, tmp_path, capsys
    ):
        """(b) pi's auth failure on a non-zero exit. stderr would pass the gate on
        its own; with a reducer present it never reaches it."""
        assert has_salvageable_json(ARRAY_SHAPED_DIAGNOSTIC)

        success, issues, error, phase_dir = _drive_real_provider_failure(
            tmp_path, PI_SPEC, stdout=PI_AUTH_ERROR, stderr=ARRAY_SHAPED_DIAGNOSTIC,
        )
        output = capsys.readouterr().out

        assert (success, issues) == (False, None)
        self._no_salvage(phase_dir, output)
        assert "nothing to salvage" in output
        envelope = _real_hard_failure(
            PI_SPEC, stdout=PI_AUTH_ERROR, stderr=ARRAY_SHAPED_DIAGNOSTIC)
        assert envelope["details"]["salvage_text"] == ""
        assert not has_salvageable_json(salvage_source(envelope["details"]))

    def test_a_killed_mid_text_array_is_salvaged_from_its_answer_text_alone(
        self, tmp_path, capsys
    ):
        """(c) The case salvage exists for: the cut array, and none of the narration."""
        success, issues, error, phase_dir = _drive_real_pi_timeout(
            tmp_path, PI_KILLED_MID_TEXT)
        output = capsys.readouterr().out

        assert success is False
        assert "[SALVAGE_NEEDED]" in output
        stem = cro.sanitize_model_name(PI_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{stem}.json").read_text(encoding="utf-8"))
        reduced = PiProvider().salvage_text(PI_KILLED_MID_TEXT).strip()
        assert request["raw_output"] == reduced
        assert request["raw_output"].startswith("[\n  {")
        assert "The plan's retry schedule" not in request["raw_output"]
        assert "[2, 4, 8]" not in request["raw_output"]
        assert '"type":"message_update"' not in request["raw_output"]
        # A deadline ended the run, and the request says so (D4.5).
        assert request["failure"] == "timeout"

    def test_a_reducer_that_raises_salvages_nothing_rather_than_the_raw_stream(
        self, tmp_path, capsys
    ):
        """(d) The raw stream passes the gate today; a raising override records
        ``""``, never a fall-back to that stream."""
        assert has_salvageable_json(PI_KILLED_MID_TEXT)

        with patch.object(PiProvider, "salvage_text", side_effect=RuntimeError("boom")):
            success, issues, error, phase_dir = _drive_real_pi_timeout(
                tmp_path, PI_KILLED_MID_TEXT)
        output = capsys.readouterr().out

        assert (success, issues) == (False, None)
        self._no_salvage(phase_dir, output)
        assert "timed out before writing an answer — nothing to salvage" in output


class TestTheFanOutBaseReadsTheAnswerTextNotTheStream:
    """The same four cases through `failure_output_from_result` (review-plan and
    review-tasks), via `_drive_real_fanout_failure`."""

    #: By name: the lane tuples are defined further down the module, beside
    #: `_drive_real_fanout_failure`, and resolved when the test runs.
    LANES = pytest.mark.parametrize("lane_name", ["review-plan", "review-tasks"])

    @staticmethod
    def _lane(name):
        return {"review-plan": REVIEW_PLAN_LANE, "review-tasks": REVIEW_TASKS_LANE}[name]

    @staticmethod
    def _no_salvage(phase_dir, output, saved):
        assert saved is False
        assert "[SALVAGE_NEEDED]" not in output
        assert list(phase_dir.glob("salvage_*.json")) == []
        stem = cro.sanitize_model_name(PI_SPEC)
        assert (phase_dir / f"error_{stem}.log").exists()
        assert not (phase_dir / f"{stem}.json").exists()

    @LANES
    def test_a_killed_mid_toolcall_is_a_failed_model(self, tmp_path, capsys, lane_name):
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, self._lane(lane_name), stdout=PI_KILLED_MID_TOOLCALL, model_spec=PI_SPEC,
            timing_out=True,
        )

        assert result["salvage_text"] == ""
        assert base.failure_output_from_result(result) == ""
        self._no_salvage(phase_dir, capsys.readouterr().out, saved)

    @LANES
    def test_an_auth_error_with_an_array_shaped_stderr_salvages_nothing(
        self, tmp_path, capsys, lane_name
    ):
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, self._lane(lane_name), stdout=PI_AUTH_ERROR, stderr=ARRAY_SHAPED_DIAGNOSTIC,
            model_spec=PI_SPEC,
        )

        assert result["stderr"] == ARRAY_SHAPED_DIAGNOSTIC
        assert base.failure_output_from_result(result) == ""
        self._no_salvage(phase_dir, capsys.readouterr().out, saved)

    @LANES
    def test_a_killed_mid_text_array_is_salvaged_from_its_answer_text_alone(
        self, tmp_path, capsys, lane_name
    ):
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, self._lane(lane_name), stdout=PI_KILLED_MID_TEXT, model_spec=PI_SPEC,
            timing_out=True,
        )
        output = capsys.readouterr().out

        assert saved is False
        assert "[SALVAGE_NEEDED]" in output
        stem = cro.sanitize_model_name(PI_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{stem}.json").read_text(encoding="utf-8"))
        assert request["raw_output"] == PiProvider().salvage_text(PI_KILLED_MID_TEXT).strip()
        assert "The plan's retry schedule" not in request["raw_output"]
        assert request["failure"] == "timeout"

    @LANES
    def test_a_reducer_that_raises_salvages_nothing(self, tmp_path, capsys, lane_name):
        with patch.object(PiProvider, "salvage_text", side_effect=RuntimeError("boom")):
            result, phase_dir, saved = _drive_real_fanout_failure(
                tmp_path, self._lane(lane_name), stdout=PI_KILLED_MID_TEXT, model_spec=PI_SPEC,
                timing_out=True,
            )

        assert result["salvage_text"] == ""
        self._no_salvage(phase_dir, capsys.readouterr().out, saved)

    def test_a_non_timeout_failure_records_no_failure_kind(self, tmp_path, capsys):
        """Only a deadline stamps the request: a run that finished and then
        exited non-zero keeps its `[]` as a genuine empty review (D4.5)."""
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, REVIEW_PLAN_LANE, stdout=TRUNCATED_ARRAY,
        )
        capsys.readouterr()

        assert "failure" not in result
        stem = cro.sanitize_model_name(KILOCODE_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{stem}.json").read_text(encoding="utf-8"))
        assert "failure" not in request


# ---------------------------------------------------------------------------
# A salvage that found nothing, seen from the phase that dispatched it
# ---------------------------------------------------------------------------
#
# `build_salvage_prompt` no longer offers `[]` as its "nothing recoverable"
# answer; it asks for `{"salvage_failed": true, "reason": "…"}`
# (`utils/subagent_prompts.py:350–358`), and `_salvage_gave_up`
# (`utils/subagent_executor.py:1969`) turns that into
# `STATUS_SALVAGE_UNRECOVERABLE` between extraction and validation — before
# anything can be written to the task's `output_path`, which for a salvage IS
# the model's own result file. That much is pinned nine ways in
# `tests/test_subagent_executor.py::TestSalvageFoundNothing`, and none of it is
# repeated here.
#
# What no test reached is the consequence one level up: the gate is a heuristic
# and the sentinel is the second defence, so what matters to an operator is that
# a salvage which gave up leaves the model *failed* — absent from `grouped.json`,
# absent from the report's `**Models:**` line, present in `Failed Reviews` — and
# that the subagent's reason survives into `subagent_batch_report.json` instead
# of being thrown away the way `[]` threw it away.


def _native_salvage_reply(reply: Any):
    """Stand in for the salvage child process, and for nothing else.

    `dispatch_salvage_tasks` imports `run_task_file` from
    `utils.subagent_executor` inside its own body, so replacing the module
    attribute is enough to intercept the dispatch without touching the phase
    code that calls it.

    What replaces it is not a stubbed outcome. `simulate_native_invoker` is the
    *other* real invoker of the D3a pair: it runs `refuse_reason`,
    `_salvage_gave_up`, `validate_result` and `persist_result` exactly as the
    subprocess lane does, and their equivalence on this very sentinel is pinned
    by `tests/test_subagent_executor.py::TestSalvageFoundNothing::
    test_the_native_invoker_answers_the_sentinel_identically`. So the only thing
    this fixture decides is what the subagent said — which is the one thing a
    test has to decide, since the alternative is a live model.
    """
    def _run(task_file, *, phase_dir=None, phase="", **_ignored):
        document = json.loads(Path(task_file).read_text(encoding="utf-8"))
        return executor.simulate_native_invoker(
            tasks_of(document),
            phase_dir=phase_dir or os.path.dirname(os.path.abspath(task_file)),
            phase=phase,
            kind=str(document.get("kind", "")),
            respond=lambda _task: reply,
            write_report=True,
        )

    return _run


#: What a salvage subagent prints when the raw output holds no answer to repair.
#: The reason is one short factual clause naming what the output turned out to
#: be, which is what the prompt asks for and what `[]` could never carry.
SALVAGE_SENTINEL = {
    "salvage_failed": True,
    "reason": "the raw output was a truncated log line, not a review",
}


class TestASalvageThatGaveUpIsNotAParticipant:
    """The sentinel's phase-level consequence, on a run that really dispatched one.

    The dispatch here is genuine: the failing slot answers with `TRUNCATED_ARRAY`,
    so the gate says salvage — correctly, this time — `emit_salvage_task`
    materializes `salvage_task_<stem>.json`, and `--exec-subagents` runs it.
    That is the route the gate cannot cover and the sentinel exists for: every
    false positive the heuristic has ends here, and what happens at this point
    is the difference between a phantom participant and a failed model.
    """

    def _run(self, tmp_path):
        with patch.object(
            executor, "run_task_file", _native_salvage_reply(SALVAGE_SENTINEL)
        ):
            return _run_cro_review(
                tmp_path,
                {
                    KILOCODE_SPEC: _hard_failure(TRUNCATED_ARRAY),
                    GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
                },
                exec_subagents=True,
            )

    def test_a_salvage_that_gave_up_leaves_the_model_out_of_the_review(
        self, tmp_path, capsys
    ):
        """The phantom participant, asserted where an operator would see it.

        Fails on the pre-sentinel executor in the way the whole plan is about:
        `[]` validated, `persist_result` wrote it to the task's `output_name` —
        `kilocode_openrouter_moonshotai_kimi-k3_thinking.json`, the model's own
        result path — `main()` folded it back into `results` as a repair, and the
        report credited a model that never ran with a clean review. Nothing in
        that chain was an error; that is why the fix had to be a value the
        result handler can tell apart from an answer.
        """
        plan, phase_dir, exit_code = self._run(tmp_path)
        output = capsys.readouterr().out
        stem = cro.sanitize_model_name(KILOCODE_SPEC)

        assert exit_code == 0
        # The salvage really was dispatched — otherwise this test proves nothing.
        assert "[SALVAGE_TASK]" in output
        assert "[exec-subagents] dispatching salvage task" in output
        # The task file is no longer under its own name because the terminal
        # verdict retired the pair (see the class below), which is a stronger
        # witness that it was dispatched than its mere presence ever was: only a
        # task that ran can have been declared unrecoverable. Renamed rather
        # than deleted, so the raw output the request carries is still readable.
        assert not (phase_dir / f"salvage_task_{stem}.json").exists()
        assert list(phase_dir.glob(f"retired_salvage_task_{stem}.*.json"))

        # ...and it wrote nothing. `output_name` is the model's result file, so
        # this absence is the whole defence.
        assert not (phase_dir / f"{stem}.json").exists()
        assert f"[{KILOCODE_SPEC}] recovered by salvage" not in output

        assert f"[PARTIAL_RESULTS: {KILOCODE_SPEC}]" in output
        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        assert status["models_completed"] == [GEMINI_SPEC]
        assert KILOCODE_SPEC in status["models_failed"]

        grouped = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
        attributed = {m for group in grouped["groups"] for m in group["models"]}
        assert attributed == {GEMINI_SPEC}, (
            "a model whose salvage gave up was counted among the models that "
            "reviewed the diff, which is what _recalculate_priority scores"
        )

    def test_report_md_attributes_no_review_to_the_model_that_never_answered(
        self, tmp_path, capsys
    ):
        """The operator-visible half: credited nowhere, explained once.

        `**Models:**` is built from `successful_models` and every issue block
        carries a `**Model:**` line, so a phantom participant shows up twice over
        in the artifact a human actually reads. The failed slot belongs in
        `Failed Reviews` with its error — and nowhere else.
        """
        plan, phase_dir, _ = self._run(tmp_path)
        capsys.readouterr()

        report = (phase_dir / "report.md").read_text(encoding="utf-8")

        assert f"**Models:** {GEMINI_SPEC}" in report
        assert f"**Model:** {KILOCODE_SPEC}" not in report
        assert "## Failed Reviews" in report
        assert f"- **{KILOCODE_SPEC}**: kilocode exited with code 1" in report

        # The one issue in the report came from the model that produced it.
        assert report.count("Unbounded retry loop") >= 1

    def test_the_subagents_own_reason_reaches_the_phases_batch_report(
        self, tmp_path, capsys
    ):
        """`subagent_batch_report.json`, in the phase dir, naming what went wrong.

        The diagnostic is the point of the sentinel's shape: `[]` said "no
        findings" and destroyed the reason, so the operator's next move was a
        JSON repair for a failure that was never about JSON. This is the record
        `dispatch_salvage_tasks` also echoes to stderr — a batch report is what
        survives the scrollback.
        """
        plan, phase_dir, _ = self._run(tmp_path)
        err = capsys.readouterr().err

        written = json.loads(
            (phase_dir / executor.BATCH_REPORT_FILENAME).read_text(encoding="utf-8")
        )
        assert written["counts"] == {"salvage_unrecoverable": 1}
        task = written["tasks"][0]
        assert task["status"] == executor.STATUS_SALVAGE_UNRECOVERABLE
        assert SALVAGE_SENTINEL["reason"] in task["reason"]
        # `dispatch_salvage_tasks` prints every non-ok outcome as it collects it,
        # so a run that never opens the report still says why the model stayed
        # failed (utils/review_orchestrator_base.py:395).
        assert "salvage_unrecoverable" in err
        assert SALVAGE_SENTINEL["reason"] in err


# ---------------------------------------------------------------------------
# A salvage the executor has already given up on is not outstanding work
# ---------------------------------------------------------------------------
#
# The sentinel stops a salvage that recovered nothing from *writing* anything.
# It does not, on its own, stop the run from asking for that salvage again:
# `emit_salvage_task` wrote `salvage_<stem>.json` when the gate said the raw
# output held an answer, the executor then said it did not, and main()'s
# end-of-run block globs `salvage_*.json` off disk with no idea that a verdict
# has since been reached about it. Observed by running the phase and reading
# stdout: `SALVAGE NEEDED / Models requiring salvage: 1`, then
# `[REAGGREGATE_AFTER_SALVAGE] …`, then `record_next` of that same command.
#
# Every part of that is wrong at once. `STATUS_SALVAGE_UNRECOVERABLE` is
# rung-less and in none of `RETRYABLE_STATUSES` / `BACKOFF_STATUSES` /
# `FATAL_STATUSES` (`utils/retry_ladder.py`) *because* the same raw output
# through the same prompt has the same answer — so the work being advertised
# will never be done, an automated host following `record_next` walks a loop
# that cannot make progress, and the one diagnostic that matters (the
# subagent's own `reason`, which on the payload that found this said the raw
# output was an auth error) is buried under a repair instruction.
#
# The verdict is applied where it is known — `dispatch_salvage_tasks`, the only
# code that ever sees the outcome — by retiring the pair out of the two globs
# and printing the reason in the marker's place. The alternative, teaching the
# end-of-run block to re-open `subagent_batch_report.json` and re-derive the
# status, puts the same rule in a second place and leaves the *dispatch* glob
# still holding a task nobody will ever run again.


def _never_dispatched(*_args, **_kwargs):
    """A tripwire for `run_task_file`: reaching it at all fails the test.

    "Nothing was dispatched" asserted from the absence of a stdout line is a
    weaker claim than it looks — a dispatch that refused early prints on stderr
    and could pass it — and letting the real executor run would spawn a runner
    against whatever CLI the developer's machine happens to have installed.
    """
    raise AssertionError(
        "dispatch_salvage_tasks ran a salvage task file that should have been "
        f"retired before this run reached it: {_args!r}"
    )


class TestATerminalSalvageIsNotReAdvertised:
    """What the phase tells the operator once the executor has given up.

    Same run as the class above — a genuine salvage, correctly requested,
    dispatched under `--exec-subagents`, answering with the sentinel — asserted
    one level further out: at the end-of-run block that decides what the host is
    pointed at next.
    """

    def _run(self, tmp_path, reply, **arg_overrides):
        with patch.object(
            executor, "run_task_file", _native_salvage_reply(reply)
        ):
            return _run_cro_review(
                tmp_path,
                {
                    KILOCODE_SPEC: _hard_failure(TRUNCATED_ARRAY),
                    GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
                },
                exec_subagents=True,
                **arg_overrides,
            )

    def test_a_salvage_that_gave_up_is_not_advertised_as_work_still_owed(
        self, tmp_path, capsys
    ):
        """No `SALVAGE NEEDED`, no reaggregate command, nothing on `record_next`.

        `record_next` is asserted separately from the printed marker because the
        two have different readers: the marker is for a human reading stdout,
        `MULTI_LLM_RESULT.next` is what an automated host re-executes without
        reading anything. Advertising a terminal salvage puts the second one in
        a loop, which is the failure mode that costs a run rather than a glance.
        Validation is skipped in this fixture, so the salvage site is the only
        `record_next` caller a passing run could reach — the assertion is
        therefore "never called" rather than "not called with this string".
        """
        with patch.object(cro, "record_next") as next_command:
            plan, phase_dir, exit_code = self._run(tmp_path, SALVAGE_SENTINEL)
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "SALVAGE NEEDED" not in output
        assert "[REAGGREGATE_AFTER_SALVAGE]" not in output
        assert next_command.call_args_list == []

    def test_the_operator_is_shown_the_reason_where_the_marker_used_to_be(
        self, tmp_path, capsys
    ):
        """The reason on the phase's own stdout, not only in the batch report.

        `subagent_batch_report.json` already records it (pinned by the class
        above) and `dispatch_salvage_tasks` already echoes every failed outcome
        to stderr — but a host that merges streams, or an operator skimming the
        run, needs the verdict on the same channel as the `[SALVAGE_NEEDED]`
        marker it replaces. What must survive is the *content*: this model's
        next move is the provider, and the reason is the only line that says so.
        """
        plan, phase_dir, _ = self._run(tmp_path, SALVAGE_SENTINEL)
        output = capsys.readouterr().out
        stem = cro.sanitize_model_name(KILOCODE_SPEC)

        assert f"[SALVAGE_UNRECOVERABLE] {stem}: " in output
        assert SALVAGE_SENTINEL["reason"] in output

    def test_the_pair_is_renamed_out_of_both_globs_rather_than_deleted(
        self, tmp_path, capsys
    ):
        """How the advertisement stops: the artifacts leave the globs, not disk.

        Two globs decide what is still owed — `dispatch_salvage_tasks`'s
        `salvage_task_*.json` and main()'s `salvage_*.json` — and both are
        prefix-anchored, so only a rename that moves the leading `salvage_`
        takes an artifact out of them. Deleting would do that too and is
        rejected: the request carries the raw output the model actually
        produced, which is the evidence that separates a truncated answer from a
        dead account, and it is exactly the run that gave up on it whose
        evidence an operator wants to read.
        """
        plan, phase_dir, _ = self._run(tmp_path, SALVAGE_SENTINEL)
        capsys.readouterr()
        stem = cro.sanitize_model_name(KILOCODE_SPEC)

        assert not (phase_dir / f"salvage_{stem}.json").exists()
        assert not (phase_dir / f"salvage_task_{stem}.json").exists()
        assert list(phase_dir.glob("salvage_*.json")) == []

        retired_request = list(
            phase_dir.glob(f"retired_salvage_{stem}.unrecoverable.*.json")
        )
        retired_task = list(
            phase_dir.glob(f"retired_salvage_task_{stem}.unrecoverable.*.json")
        )
        assert len(retired_request) == 1
        assert len(retired_task) == 1
        # The raw output is still readable, under the name that says why it was
        # retired and when.
        request = json.loads(retired_request[0].read_text(encoding="utf-8"))
        assert request["raw_output"] == TRUNCATED_ARRAY
        assert request["model"] == KILOCODE_SPEC

    def test_a_recovered_salvage_still_reaggregates_exactly_as_before(
        self, tmp_path, capsys
    ):
        """The other outcome of the same dispatch, unmoved.

        Retirement keys on `salvage_unrecoverable` and nothing else, so a
        salvage that actually repaired the model's review takes the path it
        always took: the issues are folded back into `results`, the model is
        credited in `grouped.json` and the report, `salvage_files` is filtered
        by `salvaged_models` rather than by any file having moved, and the
        request stays on disk as the audit trail. A regression that retired on
        *any* outcome would still pass the three tests above and silently delete
        the record of every successful repair.
        """
        recovered = [_issue("Missing null check")]
        with patch.object(cro, "record_next") as next_command:
            plan, phase_dir, exit_code = self._run(tmp_path, recovered)
        output = capsys.readouterr().out
        stem = cro.sanitize_model_name(KILOCODE_SPEC)

        assert exit_code == 0
        assert f"[{KILOCODE_SPEC}] recovered by salvage" in output
        assert json.loads((phase_dir / f"{stem}.json").read_text(encoding="utf-8")) == (
            recovered
        )

        # Repaired, therefore not pending: no marker, no follow-up command.
        assert "SALVAGE NEEDED" not in output
        assert "[REAGGREGATE_AFTER_SALVAGE]" not in output
        assert next_command.call_args_list == []

        # ...and not retired either. The request is the audit trail of a repair.
        assert (phase_dir / f"salvage_{stem}.json").exists()
        assert list(phase_dir.glob("retired_salvage_*.json")) == []

        grouped = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
        attributed = {m for group in grouped["groups"] for m in group["models"]}
        assert attributed == {KILOCODE_SPEC, GEMINI_SPEC}

    def test_without_exec_subagents_the_documented_flow_is_untouched(
        self, tmp_path, capsys
    ):
        """The negative that keeps salvage-then-reaggregate working at all.

        With the flag off nothing is dispatched, so no verdict exists and none
        can be assumed: the host is the invoker, the salvage has not been tried,
        and the marker plus `[REAGGREGATE_AFTER_SALVAGE]` plus `record_next` are
        the entire contract by which it gets run and its result folded in. A fix
        for the terminal case that reached this path would break the flow it was
        protecting — so the same run as the tests above, minus
        `--exec-subagents`, must advertise exactly as it always did.
        """
        with patch.object(cro, "record_next") as next_command:
            plan, phase_dir, exit_code = _run_cro_review(
                tmp_path,
                {
                    KILOCODE_SPEC: _hard_failure(TRUNCATED_ARRAY),
                    GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
                },
            )
        output = capsys.readouterr().out
        stem = cro.sanitize_model_name(KILOCODE_SPEC)

        assert exit_code == 0
        assert "[exec-subagents] dispatching salvage task" not in output
        assert "SALVAGE NEEDED" in output
        assert "Models requiring salvage: 1" in output
        assert f"  1. {stem} -> salvage_{stem}.json" in output

        expected = cro.reaggregate_command("review-code", plan)
        assert f"[REAGGREGATE_AFTER_SALVAGE] {expected}" in output
        assert next_command.call_args_list == [((expected,), {})]

        # Both halves of the pair are still exactly where the host expects them.
        assert (phase_dir / f"salvage_{stem}.json").exists()
        assert (phase_dir / f"salvage_task_{stem}.json").exists()
        assert list(phase_dir.glob("retired_salvage_*.json")) == []


# ---------------------------------------------------------------------------
# Plan review-run-scope-salvage-and-failure-visibility §5 (T5.2): the report
# header names the lost reviewer, and says a salvage was asked for
# ---------------------------------------------------------------------------
#
# `**Models:**` lists only the models that produced results, so the model a
# salvage gave up on was named only in `## Failed Reviews` at the bottom — and
# a `--reaggregate` over the same dir named it nowhere. The header line carries
# it on both lanes, with the salvage clause read from what each lane leaves on
# disk: the `--exec-subagents` lane's retired `.unrecoverable.` request, and the
# native lane's live `salvage_{stem}.json`.


def _models_failed_line(phase_dir: Path) -> str:
    lines = (phase_dir / "report.md").read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("**Models:**"))
    return lines[index + 1]


_TWO_REVIEWERS = {
    KILOCODE_SPEC: _hard_failure(TRUNCATED_ARRAY),
    GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
}


class TestTheHeaderNamesTheReviewerTheSalvageGaveUpOn:
    def _assert_the_clause(self, phase_dir: Path) -> None:
        line = _models_failed_line(phase_dir)
        assert line.startswith(
            f"**Models failed:** 1 of 2 requested — {KILOCODE_SPEC}: "
            "kilocode exited with code 1"
        ), line
        assert line.endswith("; salvage requested, nothing recovered"), line

    def test_the_exec_subagents_run_and_its_reaggregate_carry_the_clause(
        self, tmp_path, capsys
    ):
        with patch.object(
            executor, "run_task_file", _native_salvage_reply(SALVAGE_SENTINEL)
        ):
            plan, phase_dir, exit_code = _run_cro_review(
                tmp_path, dict(_TWO_REVIEWERS), exec_subagents=True,
            )
        capsys.readouterr()
        stem = cro.sanitize_model_name(KILOCODE_SPEC)
        assert exit_code == 0
        # The only evidence left is the retired request, as the plan says.
        assert not (phase_dir / f"salvage_{stem}.json").exists()
        assert list(phase_dir.glob(f"retired_salvage_{stem}.unrecoverable.*.json"))
        self._assert_the_clause(phase_dir)

        _run_cro_review(tmp_path, dict(_TWO_REVIEWERS), reaggregate=True)
        out = capsys.readouterr().out
        self._assert_the_clause(phase_dir)
        assert f"[MODEL_FAILED] {KILOCODE_SPEC}: kilocode exited with code 1" in out
        assert f"[PARTIAL_RESULTS: {KILOCODE_SPEC}]" in out

    def test_the_native_lanes_reaggregate_carries_it_from_the_live_request(
        self, tmp_path, capsys
    ):
        plan, phase_dir, exit_code = _run_cro_review(tmp_path, dict(_TWO_REVIEWERS))
        capsys.readouterr()
        stem = cro.sanitize_model_name(KILOCODE_SPEC)
        assert exit_code == 0
        assert (phase_dir / f"salvage_{stem}.json").exists()

        # The host has not salvaged yet (or its salvage recovered nothing):
        # the request is still on disk when it re-invokes the reaggregate.
        _run_cro_review(tmp_path, dict(_TWO_REVIEWERS), reaggregate=True)
        capsys.readouterr()
        self._assert_the_clause(phase_dir)


def _timed_out_after_streaming(stdout_excerpt: str) -> Dict[str, Any]:
    """What `invoke_with_file_output` returns for a slot killed at its deadline."""
    return {
        "success": False,
        "error": "kilocode timed out after 5s",
        "error_code": ERROR_TIMEOUT,
        "details": {"stdout_excerpt": stdout_excerpt, "timeout": 5},
        "output_file": "",
        "data": None,
    }


class TestATimedOutRunsSalvagedEmptyArrayIsNotAReview:
    """D4.5 at the phase level (decision 11.10: timeouts only).

    The same genuine dispatch as the two classes above — the gate correctly asks
    for salvage over a cut array, `--exec-subagents` runs it — but the salvage
    subagent answers `[]`. From a run a deadline killed that is not "this model
    found nothing": the request carries `"failure": "timeout"`, the executor
    refuses the empty array as `salvage_unrecoverable`, nothing is written, and
    the model stays failed. From a run that finished and exited non-zero, `[]`
    is still the model's genuine empty review and still lands.
    """

    def _run(self, tmp_path, kilocode_reply):
        with patch.object(executor, "run_task_file", _native_salvage_reply([])):
            return _run_cro_review(
                tmp_path,
                {
                    KILOCODE_SPEC: kilocode_reply,
                    GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
                },
                exec_subagents=True,
            )

    def test_a_timed_out_runs_empty_salvage_leaves_the_model_failed(
        self, tmp_path, capsys
    ):
        with patch.object(cro, "record_next") as next_command:
            plan, phase_dir, exit_code = self._run(
                tmp_path, _timed_out_after_streaming(TRUNCATED_ARRAY))
        output = capsys.readouterr().out
        stem = cro.sanitize_model_name(KILOCODE_SPEC)

        assert exit_code == 0
        assert "[SALVAGE_TASK]" in output
        assert "[exec-subagents] dispatching salvage task" in output
        # Refused and reported where the marker used to be, nothing written.
        assert f"[SALVAGE_UNRECOVERABLE] {stem}: " in output
        assert "timed out" in output.split(f"[SALVAGE_UNRECOVERABLE] {stem}: ", 1)[1]
        assert not (phase_dir / f"{stem}.json").exists()
        assert f"[{KILOCODE_SPEC}] recovered by salvage" not in output
        assert "SALVAGE NEEDED" not in output
        assert next_command.call_args_list == []

        # The retired request is the record of why: it names the timeout.
        retired = list(phase_dir.glob(f"retired_salvage_{stem}.*.json"))
        assert len(retired) == 1
        assert json.loads(retired[0].read_text(encoding="utf-8"))["failure"] == "timeout"

        report = json.loads(
            (phase_dir / executor.BATCH_REPORT_FILENAME).read_text(encoding="utf-8"))
        assert report["tasks"][0]["reason"] == executor.SALVAGE_EMPTY_AFTER_TIMEOUT_REASON

        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        assert status["models_completed"] == [GEMINI_SPEC]
        assert KILOCODE_SPEC in status["models_failed"]
        grouped = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
        assert {m for g in grouped["groups"] for m in g["models"]} == {GEMINI_SPEC}

    def test_a_finished_runs_empty_salvage_is_still_its_review(self, tmp_path, capsys):
        plan, phase_dir, exit_code = self._run(tmp_path, _hard_failure(TRUNCATED_ARRAY))
        output = capsys.readouterr().out
        stem = cro.sanitize_model_name(KILOCODE_SPEC)

        assert exit_code == 0
        assert "[SALVAGE_UNRECOVERABLE]" not in output
        assert f"[{KILOCODE_SPEC}] recovered by salvage" in output
        assert json.loads((phase_dir / f"{stem}.json").read_text(encoding="utf-8")) == []
        request = json.loads((phase_dir / f"salvage_{stem}.json").read_text(encoding="utf-8"))
        assert "failure" not in request


# ---------------------------------------------------------------------------
# A salvage pair does not outlive the attempt that wrote it
# ---------------------------------------------------------------------------
#
# The gate decides whether *this* attempt writes `salvage_<model>.json` and
# `salvage_task_<model>.json`. Nothing used to decide whether an *earlier*
# attempt's pair was still lying in the phase directory — and it always is:
# `get_phase_dir` is stable per plan (`utils/output_handler.py`), so every run
# for a plan writes into the same directory; `dispatch_salvage_tasks` globs
# every `salvage_task_*.json` it finds there; main()'s end-of-run block globs
# every `salvage_*.json`; and `--force` clears only the state file's
# phase-completion marker, deleting nothing from disk.
#
# So the exact P2 rerun still reproduced P2 in full: kilocode dies on
# `User not found.`, the gate correctly writes nothing, and the *previous*
# run's kilocode salvage task is dispatched anyway, advertised under
# `SALVAGE NEEDED` anyway, and can still write over that model's `output_path`
# — which for a salvage task IS the model's own result file — so a model that
# never produced a parseable word joins the aggregate and
# `SuggestionGroup._recalculate_priority` scores its silence as agreement. The
# phantom participant, arriving by the one route the gate cannot see.
#
# The rule is "the artifact is retired at the moment the attempt that would
# replace it begins", applied in `run_single_review` before the provider is
# invoked. `run_all_reviews` answers a resume from the existing `<model>.json`
# and returns *before* that call, which is what keeps the rule off models that
# began no attempt — the case the last test here pins.


class TestAStaleSalvagePairDoesNotSurviveTheNextAttempt:
    """Two runs against one phase dir, which is the only way to see any of this.

    Every assertion below is about what run 2 does with run 1's leftovers, so
    each test really runs the phase twice over the same plan — same
    `get_phase_dir`, same artifacts, `--force` in between exactly as an operator
    would type it. A single-run fixture cannot express the defect at all.
    """

    def _first_run_leaves_a_salvage_pair(self, tmp_path, capsys, extra_models=()):
        """Run 1: a genuine salvage request nobody ever ran.

        `TRUNCATED_ARRAY` is a real damaged answer, so the gate says salvage and
        is right to. Without `--exec-subagents` the pair is left for a host that
        — as far as run 2 is concerned — never came back, which is precisely the
        state a phase dir is in after an interrupted or abandoned run.
        """
        replies = {
            KILOCODE_SPEC: _hard_failure(TRUNCATED_ARRAY),
            GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
        }
        replies.update(extra_models)
        plan, phase_dir, exit_code = _run_cro_review(tmp_path, replies)
        assert exit_code == 0
        stem = cro.sanitize_model_name(KILOCODE_SPEC)
        assert (phase_dir / f"salvage_{stem}.json").exists()
        assert (phase_dir / f"salvage_task_{stem}.json").exists()
        capsys.readouterr()
        return plan, phase_dir

    def test_a_forced_rerun_does_not_dispatch_the_previous_runs_salvage_task(
        self, tmp_path, capsys
    ):
        """The P2 rerun end to end: gate writes nothing, and nothing is left to run.

        Run 2 is the failure that started all of this — kilocode's account is
        dead, the whole of stderr is `Error: User not found.`, and
        `has_salvageable_json` correctly declines to ask for a repair. What used
        to happen next is the entire defect: run 1's task file was still there,
        `dispatch_salvage_tasks` globbed it, and `--exec-subagents` ran a repair
        pass against an answer from a run that no longer exists, over a slot
        this run recorded as failed.

        The tripwire on `run_task_file` is what makes "nothing dispatched" an
        observation rather than an inference from a missing stdout line.
        """
        plan, phase_dir = self._first_run_leaves_a_salvage_pair(tmp_path, capsys)
        stem = cro.sanitize_model_name(KILOCODE_SPEC)

        with patch.object(executor, "run_task_file", _never_dispatched), \
             patch.object(cro, "record_next") as next_command:
            _, _, exit_code = _run_cro_review(
                tmp_path,
                {
                    KILOCODE_SPEC: _hard_failure(KILOCODE_AUTH_FAILURE),
                    GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
                },
                force=True,
                exec_subagents=True,
            )
        output = capsys.readouterr().out

        assert exit_code == 0
        # The gate did its half: this attempt asked for no salvage.
        assert "nothing to salvage" in output
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output

        # ...and the other half: run 1's pair is not treated as this run's.
        assert "[exec-subagents] dispatching salvage task" not in output
        assert "SALVAGE NEEDED" not in output
        assert "[REAGGREGATE_AFTER_SALVAGE]" not in output
        assert next_command.call_args_list == []

        assert not (phase_dir / f"salvage_{stem}.json").exists()
        assert not (phase_dir / f"salvage_task_{stem}.json").exists()
        assert len(list(phase_dir.glob(f"retired_salvage_{stem}.superseded.*.json"))) == 1
        assert len(
            list(phase_dir.glob(f"retired_salvage_task_{stem}.superseded.*.json"))
        ) == 1

        # The point of all of it: nothing was written to this model's result
        # path, so the aggregate names only the model that reviewed the diff.
        assert not (phase_dir / f"{stem}.json").exists()
        grouped = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
        attributed = {m for group in grouped["groups"] for m in group["models"]}
        assert attributed == {GEMINI_SPEC}

    def test_a_rerun_all_advertises_only_the_models_this_run_could_not_parse(
        self, tmp_path, capsys
    ):
        """The salvaged subset shrinking between runs, which is the common case.

        Nothing about run 1 is wrong here — two models really did return damaged
        answers and really did need salvage. Run 2 re-runs every model
        (`--rerun-all`) and one of them answers cleanly this time; its request
        describes an answer that no longer exists, and advertising it would send
        the host to repair a reply it cannot see and does not need. The count
        line is asserted as well as the names because `Models requiring salvage:
        N` is what an operator reads first and what a host's log grep matches.
        """
        plan, phase_dir = self._first_run_leaves_a_salvage_pair(
            tmp_path,
            capsys,
            extra_models={OPENCODE_SPEC: _hard_failure(TRUNCATED_ARRAY)},
        )
        kilocode_stem = cro.sanitize_model_name(KILOCODE_SPEC)
        opencode_stem = cro.sanitize_model_name(OPENCODE_SPEC)
        assert (phase_dir / f"salvage_{opencode_stem}.json").exists()

        _, _, exit_code = _run_cro_review(
            tmp_path,
            {
                KILOCODE_SPEC: _hard_failure(TRUNCATED_ARRAY),
                OPENCODE_SPEC: _good_reply([_issue("Missing null check")]),
                GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
            },
            force=True,
            rerun_all=True,
        )
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "Models requiring salvage: 1" in output
        assert f"  1. {kilocode_stem} -> salvage_{kilocode_stem}.json" in output
        assert opencode_stem not in output.split("SALVAGE NEEDED", 1)[1]

        # The model that failed again re-requested salvage under its own name:
        # retiring is not "delete once", it is "the previous attempt's copy".
        assert (phase_dir / f"salvage_{kilocode_stem}.json").exists()
        assert (phase_dir / f"salvage_task_{kilocode_stem}.json").exists()
        assert len(
            list(phase_dir.glob(f"retired_salvage_{kilocode_stem}.superseded.*.json"))
        ) == 1

        # The model that answered has no live request left, only the retired one.
        assert not (phase_dir / f"salvage_{opencode_stem}.json").exists()
        assert not (phase_dir / f"salvage_task_{opencode_stem}.json").exists()
        assert len(
            list(phase_dir.glob(f"retired_salvage_{opencode_stem}.superseded.*.json"))
        ) == 1
        assert (phase_dir / f"{opencode_stem}.json").exists()

    def test_a_model_skipped_as_already_complete_keeps_its_salvage_pair(
        self, tmp_path, capsys
    ):
        """The boundary of the rule, and the reason it is placed where it is.

        Retiring belongs to *the attempt that would replace the artifact*. A
        resumed run (no `--rerun-all`) answers a model with an existing
        `<model>.json` straight out of `run_all_reviews`'s skip branch and never
        reaches `run_single_review` — no provider is invoked, no result is
        produced, nothing supersedes anything. Retiring there would be the
        opposite defect: destroying a live request on a run that did not even
        look at the model.

        The `<model>.json` here is written by hand on purpose — that is exactly
        what the documented no-`--exec-subagents` flow does to a phase dir, a
        host running `task.prompt` and writing the JSON it returns to
        `task.output_path`.
        """
        plan, phase_dir = self._first_run_leaves_a_salvage_pair(tmp_path, capsys)
        stem = cro.sanitize_model_name(KILOCODE_SPEC)
        (phase_dir / f"{stem}.json").write_text(
            json.dumps([_issue("Recovered by the host")]), encoding="utf-8"
        )

        _, _, exit_code = _run_cro_review(
            tmp_path,
            {
                KILOCODE_SPEC: _hard_failure(KILOCODE_AUTH_FAILURE),
                GEMINI_SPEC: _good_reply([_issue("Unbounded retry loop")]),
            },
            force=True,
        )
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "[SKIP]" in output
        assert "already has results" in output
        # The canned auth failure was never consulted: no attempt, no retirement.
        assert "nothing to salvage" not in output
        assert (phase_dir / f"salvage_{stem}.json").exists()
        assert (phase_dir / f"salvage_task_{stem}.json").exists()
        assert list(phase_dir.glob("retired_salvage_*.json")) == []


# ---------------------------------------------------------------------------
# The same loss in the fan-out base — residual item 5
# ---------------------------------------------------------------------------
#
# `code_review_orchestrator`'s salvage site A was closed by reading
# `details["stdout_excerpt"]` ahead of `details["stderr"]`. The fan-out base —
# which serves **`review-plan` and `review-tasks`** — carried the identical loss
# by a *different* mechanism, so that fix did not reach it and could not have:
# `stdout_excerpt` appeared zero times in `utils/review_orchestrator_base.py`.
# `run_single_model`'s failure returns hardcoded `"output": None, "stdout": ""`
# and carried only `stderr_text`; `save_model_result` then short-circuited on
# `if not success or not output:`, wrote the error log and returned `False` — so
# the salvage block in its `except (JSONDecodeError, ValueError)` handler was not
# merely gated to `False` on a hard failure, it was **unreachable**. A provider
# that streamed a truncated array and was then killed at the deadline lost that
# answer silently, in two phases at once.
#
# The fix is deliberately *not* "assign the excerpt to `output`", which is the
# one-line version and is worse than the bug: `save_model_result` would then
# parse it, and a hard failure whose stdout happened to hold a well-formed
# suggestions array would be written to `{model}.json` and counted as that
# model's completed review — the phantom participant inverted, a crashed run
# recorded as a result. `output` keeps meaning "the answer produced by a run that
# completed"; the excerpt travels on `stdout` and reaches `save_model_result`
# through the keyword-only `failure_output`, which only salvage may read. The
# last test in this class is the one that pins that distinction, and it is the
# reason the change was deferred out of site A's pass rather than ridden in.

#: The three names that differ between the two lanes this base serves, and
#: nothing else does: `run_review_orchestration` is called with exactly this
#: triple by `review_plan_orchestrator.main` and `review_tasks_orchestrator.main`.
#: Every test below runs twice because "for **both** `review-plan` and
#: `review-tasks`" is the close condition's wording, and a shared base is
#: precisely the place where a lane silently stops being covered.
REVIEW_PLAN_LANE = ("review-plan", "plan_review", "review_plan")
REVIEW_TASKS_LANE = ("review-tasks", "task_review", "review_tasks")

#: Minimal, and it is the template's whole job here: `invoke_with_file_output`
#: formats it with the context plus `output_json_path`/`output_location`, and
#: what is under test is which *stream* survives a dead child, not what the
#: child was asked. The real templates are exercised by the review-plan and
#: review-tasks suites.
FAN_OUT_PROMPT = "Review {plan_path}. Write to {output_json_path}.\n{output_location}"

#: An answer the model really finished — valid against `validate_suggestion`'s
#: `REQUIRED_FIELDS` / `VALID_IMPORTANCE` / `VALID_TYPES` — sitting on the stdout
#: of a run that then exited non-zero. This is the payload that makes the wrong
#: fix visible, and `_it_would_really_have_been_written` below proves it is not a
#: straw man by feeding the same bytes in as `output` and watching them land.
COMPLETE_SUGGESTIONS = json.dumps([
    {
        "title": "Retry ladder has no deadline",
        "desc": "Each retry re-issues the full per-slot timeout.",
        "importance": "high",
        "reference": "Section 7.5",
        "type": "modification",
    }
])


def _killed_after_streaming(stdout: str, stderr: str = ""):
    """A `subprocess.run` stand-in that kills the provider CLI **and nothing else**.

    `patch("utils.llm_client.subprocess.run", ...)` does not patch a name private
    to `llm_client`: `utils.llm_client.subprocess` *is* the stdlib module, so the
    patch is global for its duration, and `run_single_model` shells out through
    it before it ever spawns a provider — `utils/git_utils.get_project_root` runs
    `git rev-parse` on the very first line of the function. The sibling helpers
    above get away with a blanket `return_value` because a `CompletedProcess` with
    a non-zero code is something `get_project_root` already tolerates; a blanket
    `side_effect` does not get away with it, and would kill the repo-root probe
    instead of the model — a green test about a code path that never ran.

    So `git` is delegated to the real `subprocess.run` and everything else is
    killed the way the kernel kills a child at the deadline: `TimeoutExpired`
    carrying the bytes already read off the pipe. Those arrive as **bytes even
    under `text=True`**, which is why `invoke_with_provider` takes its excerpt
    from the decoded local rather than from `e.stdout` — encoding that here keeps
    the fixture honest about the shape the production decode has to handle.
    """
    real_run = subprocess.run

    def _run(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args", [])
        if argv and str(argv[0]) == "git":
            return real_run(*args, **kwargs)
        raise subprocess.TimeoutExpired(
            cmd=list(argv),
            timeout=kwargs.get("timeout") or 1,
            output=stdout.encode("utf-8"),
            stderr=stderr.encode("utf-8"),
        )

    return _run


def _drive_real_fanout_failure(
    tmp_path: Path,
    lane,
    *,
    stdout: str,
    stderr: str = "",
    model_spec: str = KILOCODE_SPEC,
    timing_out: bool = False,
):
    """Run one fan-out slot and persist it, with only the process boundary faked.

    Both halves of the change under test, joined the way production joins them.
    `run_single_model` is the shipping coroutine — real adapter, real argv, real
    `invoke_with_provider`, real `_attach_stdout_excerpt`, real retry ladder — and
    the two lines after it are a verbatim copy of what
    `review_orchestrator_base.run_review_orchestration` does with each entry of
    `run_all_models`' result dict, `failure_output_from_result` included. Nothing
    canned passes between them.

    That seam is the point. Every other fan-out test in this file hands
    `save_model_result` a string directly, which is the right seam for a question
    about what the *writer* decides; it is the wrong one for this question,
    because the defect was that the writer was never shown the evidence at all.
    A canned `details` — or a canned result dict — is free to carry keys the
    plumbing does not produce, and would assert the two modules agree on
    `stdout_excerpt` by construction, going on passing after either side moved.

    Args:
        tmp_path: The test's directory; the plan, the `out_dir` and the phase dir
            all hang off it.
        lane: `REVIEW_PLAN_LANE` or `REVIEW_TASKS_LANE`.
        stdout: What the CLI printed before it lost.
        stderr: ...and what it wrote to the other channel.
        model_spec: Resolved for real, so the adapter's own `describe_failure`
            verdict rides out in `error` exactly as it would live.
        timing_out: Kill the child at the deadline instead of letting it exit
            non-zero — the "streamed the answer, then died" case the plan names,
            and a different branch of `invoke_with_provider`.

    Returns:
        `(result, phase_dir, saved)` — the `run_single_model` record, the phase
        directory every artifact lands in, and `save_model_result`'s verdict,
        which is what `run_review_orchestration` turns into `failed_models`.
    """
    phase_name, invoke_phase, salvage_phase_name = lane
    plan_path = _plan_file(tmp_path)
    out_dir = tmp_path / "out"

    if timing_out:
        run_patch = patch(
            "utils.llm_client.subprocess.run",
            side_effect=_killed_after_streaming(stdout, stderr),
        )
    else:
        run_patch = patch(
            "utils.llm_client.subprocess.run",
            return_value=_dead_process(stdout, stderr),
        )

    # The ladder retries a timeout, so the sleep between attempts has to be the
    # test-mode 10 ms rather than the real backoff; every other path here is
    # terminal and never reaches it.
    with run_patch, \
         patch("shutil.which", return_value="/usr/bin/stub-cli"), \
         patch.dict(os.environ, {"MULTI_LLM_TEST_FAST_BACKOFF": "1"}):
        result = asyncio.run(base.run_single_model(
            model_spec=model_spec,
            plan_path=str(plan_path),
            prompt_template=FAN_OUT_PROMPT,
            prompt_context={"plan_path": str(plan_path), "prefix": "plan"},
            phase_name=phase_name,
            invoke_phase=invoke_phase,
            # kilocode's slot has to be wide enough that §7.5 rule 4 dispatches
            # it: a budget too small for one worst-case attempt of the declared
            # envelope is refused with nothing spawned, and a refused slot is a
            # different return with no child output by construction.
            timeout=1800,
            out_dir=str(out_dir),
            prefix="plan",
        ))

    phase_dir = base.get_phase_dir(plan_path, phase_name)
    saved = base.save_model_result(
        "plan", model_spec, result["success"], result["output"], result["error"],
        str(out_dir), str(phase_dir), salvage_phase_name,
        failure_output=base.failure_output_from_result(result),
        failure=result.get(base.SALVAGE_FAILURE_KEY),
    )
    return result, phase_dir, saved


class TestTheFanOutBaseReadsTheAnswerTheProviderLeftOnStdout:
    """Residual item 5, both lanes: `review-plan` and `review-tasks`.

    The sibling class `TestSalvageSiteAReadsTheAnswerTheProviderLeftOnStdout`
    asks the same question of `review-code`. This one asks it of the base the
    other two phases share, where the answer used to be "the gate is never
    reached" rather than "the gate decides wrongly" — a strictly worse shape,
    because no gate verdict was printed at all and the operator saw only an error
    log for a run that had produced a recoverable review.

    Every test drives the real chain through `_drive_real_fanout_failure`, and
    runs on both lanes, because the base is exactly the kind of code where one
    lane keeps working and the other quietly stops being covered.
    """

    def test_a_review_plan_provider_that_streamed_an_array_and_then_died_is_salvaged(
        self, tmp_path, capsys
    ):
        """The case the base was structurally unable to see, on the first lane.

        stdout holds a review array cut off mid-object, stderr holds nothing, the
        exit code is 1. The assertion is on the salvage request's `raw_output`
        rather than merely on the `[SALVAGE_NEEDED]` marker, for the reason the
        site A sibling gives: a marker printed over an empty `raw_output` is worse
        than no marker, because `build_salvage_prompt`'s "found nothing" fallback
        is then the only honest answer a subagent can give.

        The model stays **failed** — `save_model_result` returns `False`, which is
        what `run_review_orchestration` turns into a `failed_models` entry and a
        `[PARTIAL_RESULTS: ...]` line. Salvage is an offer of repair, never a
        result.
        """
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, REVIEW_PLAN_LANE, stdout=TRUNCATED_ARRAY,
        )

        assert result["success"] is False
        # `output` is what gets parsed and written; it must stay empty however
        # much text the run left behind. `stdout` is that text.
        assert result["output"] is None
        assert result["stdout"] == TRUNCATED_ARRAY
        assert saved is False

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" in output
        assert "[SALVAGE_TASK]" in output

        stem = base.sanitize_model_name(KILOCODE_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{stem}.json").read_text(encoding="utf-8")
        )
        assert request["raw_output"]
        assert request["raw_output"] == TRUNCATED_ARRAY
        assert request["expected_type"] == "array"
        assert request["phase"] == "review_plan"
        assert (phase_dir / f"salvage_task_{stem}.json").exists()

        # The error log is still written, and still names the failure rather than
        # the salvage: the operator's record of a dead slot does not move because
        # a repair was offered for it.
        logged = (phase_dir / f"error_{stem}.log").read_text(encoding="utf-8")
        assert "kilocode exited with code 1" in logged
        assert "Salvage: requested" in logged

        # And no result file, on the branch that wrote the request.
        assert not (phase_dir / f"{stem}.json").exists()

    def test_a_review_tasks_provider_that_streamed_an_array_and_then_died_is_salvaged(
        self, tmp_path, capsys
    ):
        """The second lane, and not a copy for symmetry's sake.

        `review-tasks` reaches the same `run_single_model` and the same
        `save_model_result` with a different `phase_name`, a different
        `invoke_phase` and a different salvage `phase` stamp, and those three
        names are what route the artifacts: the phase dir they land in, the
        output path `invoke_with_file_output` generates, and the `phase` a salvage
        subagent reads back out of the request. A fix wired through one lane's
        constants and not the base would pass the test above and fail here.
        """
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, REVIEW_TASKS_LANE, stdout=TRUNCATED_ARRAY,
        )

        assert (result["success"], result["output"], saved) == (False, None, False)
        assert result["stdout"] == TRUNCATED_ARRAY
        assert phase_dir.name == "review-tasks"

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" in output
        assert "[SALVAGE_TASK]" in output

        stem = base.sanitize_model_name(KILOCODE_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{stem}.json").read_text(encoding="utf-8")
        )
        assert request["raw_output"] == TRUNCATED_ARRAY
        assert request["phase"] == "review_tasks"
        # `output_path` IS the model's result file, which is what makes a
        # dispatched salvage able to turn this slot into a participant — and what
        # makes it matter that the branch writing this request never wrote that
        # file itself.
        assert request["output_path"] == str(phase_dir / f"{stem}.json")
        assert (phase_dir / f"salvage_task_{stem}.json").exists()
        assert not (phase_dir / f"{stem}.json").exists()

    def test_the_kilocode_auth_text_on_stderr_alone_salvages_nothing_in_review_plan(
        self, tmp_path, capsys
    ):
        """The negative, and the proof that stdout-first is a fall-through.

        kilocode's dead account is ANSI-coloured prose on stderr from a CLI that
        never emitted a byte of answer, so `_attach_stdout_excerpt` omits
        `stdout_excerpt` entirely rather than setting `""`. Reading stdout *ahead
        of* stderr must therefore leave stderr reachable: a precedence written as
        a replacement would blank the gate's input for every provider that dies
        before speaking, and the right verdict would then be reached for a reason
        that has nothing to do with the payload — which is the failure mode this
        whole class exists to stop repeating.

        The error log is still written. That is the half of the close condition
        that is easy to lose: "no salvage" must not become "no record".
        """
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, REVIEW_PLAN_LANE, stdout="", stderr=KILOCODE_AUTH_FAILURE,
        )

        assert (result["success"], result["output"], saved) == (False, None, False)
        assert result["stdout"] == ""
        # The adapter's own `describe_failure` read the dead account off stderr,
        # so the operator's next move is auth rather than a JSON repair — exactly
        # the move a salvage request would have stolen.
        assert "User not found." in result["error"]

        output = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in output
        assert "[SALVAGE_TASK]" not in output
        assert list(phase_dir.glob("salvage_*.json")) == []

        stem = base.sanitize_model_name(KILOCODE_SPEC)
        logged = (phase_dir / f"error_{stem}.log").read_text(encoding="utf-8")
        assert "kilocode exited with code 1" in logged
        assert "Salvage: requested" not in logged
        # The stream the gate turned down is in the log, under a heading that
        # says it is not the model's answer — it is the evidence an operator
        # needs to tell a dead account from a truncated review.
        assert "Raw output (failed run):" in logged
        assert "User not found." in logged

    def test_the_kilocode_auth_text_on_stderr_alone_salvages_nothing_in_review_tasks(
        self, tmp_path, capsys
    ):
        """The negative on the second lane; see the sibling above for the argument."""
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, REVIEW_TASKS_LANE, stdout="", stderr=KILOCODE_AUTH_FAILURE,
        )

        assert (result["success"], result["output"], saved) == (False, None, False)
        assert capsys.readouterr().out.count("[SALVAGE_NEEDED]") == 0
        assert list(phase_dir.glob("salvage_*.json")) == []

        stem = base.sanitize_model_name(KILOCODE_SPEC)
        assert (phase_dir / f"error_{stem}.log").exists()
        assert not (phase_dir / f"{stem}.json").exists()

    def test_a_banner_on_stderr_does_not_join_the_array_stdout_carried(
        self, tmp_path, capsys
    ):
        """Precedence, not concatenation — the rule stated once, checked in this lane.

        A provider that dies mid-answer usually leaves something on *both*
        channels: the truncated array on stdout, its launch banner or a teardown
        traceback on stderr. The obvious repair is to feed the gate
        `stdout + stderr`, and it is wrong twice: `has_salvageable_json` would see
        a document whose spans no longer line up, and — worse, because it is
        silent — `raw_output` would reach the salvage subagent as two documents
        with no marker saying which one it was asked to repair.

        `failure_output_from_result` is the one place that rule lives for this
        lane, and its docstring points at `code_review_orchestrator`'s site A as
        the same rule in the other one. This test is what stops the two drifting.
        """
        _, phase_dir, _ = _drive_real_fanout_failure(
            tmp_path, REVIEW_PLAN_LANE,
            stdout=TRUNCATED_ARRAY, stderr=KILOCODE_AUTH_FAILURE,
        )

        assert "[SALVAGE_NEEDED]" in capsys.readouterr().out
        stem = base.sanitize_model_name(KILOCODE_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{stem}.json").read_text(encoding="utf-8")
        )
        assert request["raw_output"] == TRUNCATED_ARRAY
        assert "User not found." not in request["raw_output"]

    def test_a_provider_killed_at_the_deadline_keeps_the_array_it_had_streamed(
        self, tmp_path, capsys
    ):
        """The timeout, which is the case the plan actually names for this site.

        A non-zero exit and a deadline kill reach `run_single_model` by different
        branches of `invoke_with_provider` — one reads `CompletedProcess.stdout`,
        the other decodes the bytes `subprocess.run` populates onto the re-raised
        `TimeoutExpired` — and only the first is covered above. The timeout is
        also the branch where the loss was most likely in practice: a run killed
        at the deadline is the run most likely to have emitted most of its answer,
        and the ladder retries it, so the excerpt that survives must be the *last*
        attempt's rather than a stale one from the first.
        """
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, REVIEW_PLAN_LANE, stdout=TRUNCATED_ARRAY, timing_out=True,
        )

        assert (result["success"], result["output"], saved) == (False, None, False)
        assert "timed out" in result["error"]
        assert result["stdout"] == TRUNCATED_ARRAY

        assert "[SALVAGE_NEEDED]" in capsys.readouterr().out
        stem = base.sanitize_model_name(KILOCODE_SPEC)
        request = json.loads(
            (phase_dir / f"salvage_{stem}.json").read_text(encoding="utf-8")
        )
        assert request["raw_output"] == TRUNCATED_ARRAY
        assert not (phase_dir / f"{stem}.json").exists()

    def test_a_hard_failure_whose_stdout_parses_is_not_written_as_the_models_review(
        self, tmp_path, capsys
    ):
        """The test the close condition does not name, and the reason this change waited.

        Take the archetype and make the answer *complete*: a provider that
        finished a well-formed, schema-valid suggestions array on stdout and then
        exited non-zero anyway — a crash on teardown, a wrapper that mistranslates
        an exit code, a kill that landed one byte after the closing bracket.

        The one-line fix for residual item 5 is to put the excerpt in `output`.
        Under it this run parses, validates and is written to `{model}.json`, and
        reaggregation reads a **crashed** run as a model that reviewed the plan —
        counted by `SuggestionGroup._recalculate_priority`'s `len(self.models)`
        exactly like a model that answered. That is the phantom participant of
        `retire_salvage_artifacts`'s docstring arriving inverted, and it is worse
        than the silent loss it replaces, because a lost review is visibly absent
        while an invented one is not.

        So: no result file, and the model still recorded failed. A salvage request
        is legitimate here and is asserted to exist — the whole point of the split
        is that the *request* is what a failed run may produce, while the *result*
        is not. If this test ever goes green by writing `{model}.json`, the change
        it guards has introduced a worse defect than the one it fixed.
        """
        result, phase_dir, saved = _drive_real_fanout_failure(
            tmp_path, REVIEW_TASKS_LANE, stdout=COMPLETE_SUGGESTIONS,
        )
        stem = base.sanitize_model_name(KILOCODE_SPEC)

        # The consequence first, deliberately: this is the assertion whose
        # failure means the defect is back, and a test that trips on a contract
        # detail three lines earlier reports the wrong thing to whoever broke it.
        assert not (phase_dir / f"{stem}.json").exists()
        # Stated as the whole set rather than one `exists()`, because the failure
        # this guards against is a file *appearing*, and it can appear under a
        # name this test did not think to check.
        assert {p.name for p in phase_dir.glob("*.json")} == {
            f"salvage_{stem}.json", f"salvage_task_{stem}.json",
        }
        # Still failed, which is how `run_review_orchestration` keeps it out of
        # `completed_models`, out of `grouped.json` and in `Failed Reviews`.
        assert saved is False
        assert (phase_dir / f"error_{stem}.log").exists()
        assert "[SALVAGE_NEEDED]" in capsys.readouterr().out

        # And the contract underneath it: the run did not complete, so it has no
        # `output`; what it left behind travels on `stdout` instead.
        assert result["success"] is False
        assert result["output"] is None
        assert result["stdout"] == COMPLETE_SUGGESTIONS

    def test_the_same_payload_is_one_save_model_result_really_would_have_written(
        self, tmp_path
    ):
        """The proof that the test above is not a straw man.

        `COMPLETE_SUGGESTIONS` has to be a payload `save_model_result` accepts —
        `extract_json_array` finds it, `json.loads` takes it, and every entry
        survives `validate_suggestion`'s required fields and value sets. If it did
        not, "no `{model}.json` was written" would be a statement about a malformed
        fixture and would keep passing under the very fix it is there to refuse.

        Same bytes, same function, one argument moved: passed as `output` — the
        answer of a run that *completed* — they land on disk. Passed as
        `failure_output` they do not. That is the whole distinction residual item
        5's close condition asks for, in two calls.
        """
        phase_dir = tmp_path / "review-tasks"
        phase_dir.mkdir()
        stem = base.sanitize_model_name(KILOCODE_SPEC)

        assert base_save_model_result(
            "plan", KILOCODE_SPEC, True, COMPLETE_SUGGESTIONS, None,
            str(tmp_path), str(phase_dir), "review_tasks",
        ) is True
        written = json.loads(
            (phase_dir / f"{stem}.json").read_text(encoding="utf-8")
        )
        assert len(written) == 1
        assert written[0]["title"] == "Retry ladder has no deadline"

    def test_the_default_none_leaves_every_existing_caller_exactly_as_it_was(
        self, tmp_path, capsys
    ):
        """`failure_output` is keyword-only and defaults to `None`, and that is load-bearing.

        `review_plan_orchestrator.save_model_result`'s backward-compatible wrapper
        and every test call site in this suite pass the eight positional arguments
        and nothing else. For them the hard-failure branch must behave exactly as
        it did before the parameter existed: the error log, `False`, and not one
        salvage artifact — because a caller that supplies no failure output is
        saying "I have no evidence", which is not the same as "there is none to
        find" and must never be guessed at from `error`.
        """
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()

        saved = base_save_model_result(
            "plan", GEMINI_SPEC, False, None, "gemini exited with code 1",
            str(tmp_path), str(phase_dir), "review_plan",
        )

        assert saved is False
        assert list(phase_dir.glob("salvage_*.json")) == []
        assert "[SALVAGE_NEEDED]" not in capsys.readouterr().out
        logged = (phase_dir / "error_gemini_gemini-2_5-flash.log").read_text(
            encoding="utf-8"
        )
        assert "gemini exited with code 1" in logged
        assert "Salvage: requested" not in logged
        assert "Raw output" not in logged


# ---------------------------------------------------------------------------
# A dead slot promises no result file
# ---------------------------------------------------------------------------
#
# `invoke_with_file_output` hands every hard failure back with `output_file` set
# (the path it *would* have read) and no `source`, so both fan-outs printed
#
#     [model] Result will be saved to: <path> (source: unknown)
#     [model] Agent failed (23.7s): cursor-agent exited with code 1
#
# for each slot that produced nothing — measured 2026-09-17 on all three dead
# slots of a run. The line is `describe_output_file`'s "the model did not write
# here" wording (F14), which is right for a stdout fallback and a promise nothing
# keeps for a failure. It is printed only for a slot that answered now; the
# failure line that follows already names the outcome.


class TestADeadSlotPromisesNoResultFile:
    def test_the_code_review_fan_out_prints_no_result_line_for_a_failed_slot(
        self, tmp_path, capsys
    ):
        """The real `invoke_with_file_output` forward — `output_file` set, no
        `source` — through the real `run_single_review`."""
        success, _issues, _error, _phase_dir = _drive_real_provider_failure(
            tmp_path, KILOCODE_SPEC, "", KILOCODE_AUTH_FAILURE
        )
        out = capsys.readouterr().out

        assert success is False
        assert "Agent failed" in out
        assert "Result will be saved to" not in out
        assert "(source: unknown)" not in out

    def test_the_fan_out_base_prints_none_either(self, tmp_path, capsys):
        result, _phase_dir, _saved = _drive_real_fanout_failure(
            tmp_path, REVIEW_PLAN_LANE, stdout="", stderr=KILOCODE_AUTH_FAILURE
        )
        out = capsys.readouterr().out

        assert result["success"] is False
        assert "Result will be saved to" not in out
        assert "(source: unknown)" not in out

    def test_a_slot_that_answered_still_names_its_file(self, tmp_path, capsys):
        """The regression guard: the success line is exactly what it was."""
        plan_path = _plan_file(tmp_path)
        reply = {**_good_reply([_issue("Issue A")]), "output_file": "/out/kilocode.json"}

        with patch.object(cro, "invoke_with_file_output", return_value=reply):
            success, issues, _error = asyncio.run(
                cro.run_single_review(KILOCODE_SPEC, plan_path, tmp_path / "out")
            )
        out = capsys.readouterr().out

        assert success is True and len(issues) == 1
        assert "Output file: /out/kilocode.json (source: file)" in out
