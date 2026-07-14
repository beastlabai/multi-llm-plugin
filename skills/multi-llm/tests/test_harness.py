"""Tests for the test harness package itself.

These tests verify that the harness components work correctly and can be
used to run orchestrators with mock LLM providers.
"""

import json
import os
import sys
import pytest
from pathlib import Path

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    SkillRunner,
    SkillResult,
    FixtureManager,
    TestPlanFixture,
    MockProvider,
    MockLLMCall,
    AssertionHelpers,
)


class TestSkillRunner:
    """Tests for SkillRunner class."""

    def test_creates_mock_symlinks(self, tmp_path):
        """Test that SkillRunner creates provider symlinks in bin directory."""
        runner = SkillRunner(tmp_path)

        # Check bin directory exists
        assert runner.bin_dir.exists()
        assert runner.bin_dir.is_dir()

        # Check symlinks for all providers
        for provider in SkillRunner.PROVIDERS:
            symlink = runner.bin_dir / provider
            assert symlink.exists(), f"Missing symlink for {provider}"
            assert symlink.is_symlink(), f"{provider} is not a symlink"
            # Symlink should point to mock_llm.py
            assert symlink.resolve() == runner.mock_llm_path.resolve()

    def test_builds_correct_environment(self, tmp_path):
        """Test that SkillRunner builds correct environment variables."""
        runner = SkillRunner(tmp_path)
        env = runner._build_env()

        # Check PATH is prepended
        assert str(runner.bin_dir) in env["PATH"]
        assert env["PATH"].startswith(str(runner.bin_dir))

        # Check test mode is enabled
        assert env.get("MULTI_LLM_TEST_MODE") == "1"

        # Check call log is set
        assert env.get("MOCK_LLM_CALL_LOG") == str(runner.call_log_path)

        # Check fast backoff is enabled
        assert env.get("MULTI_LLM_TEST_FAST_BACKOFF") == "1"

    def test_scenario_path_in_environment(self, tmp_path):
        """Test that scenario path is included in environment."""
        scenario_path = tmp_path / "test_scenario.yaml"
        scenario_path.write_text("name: test", encoding="utf-8")

        runner = SkillRunner(tmp_path, scenario_path=scenario_path)
        env = runner._build_env()

        assert env.get("MOCK_LLM_SCENARIO") == str(scenario_path)

    def test_extra_env_included(self, tmp_path):
        """Test that extra environment variables are included."""
        runner = SkillRunner(tmp_path, extra_env={"CUSTOM_VAR": "custom_value"})
        env = runner._build_env()

        assert env.get("CUSTOM_VAR") == "custom_value"

    def test_inject_failure(self, tmp_path):
        """Test that inject_failure sets MOCK_LLM_FAIL."""
        runner = SkillRunner(tmp_path)
        runner.inject_failure()
        env = runner._build_env()

        assert env.get("MOCK_LLM_FAIL") == "1"

    def test_inject_timeout(self, tmp_path):
        """Test that inject_timeout sets MOCK_LLM_TIMEOUT."""
        runner = SkillRunner(tmp_path)
        runner.inject_timeout()
        env = runner._build_env()

        assert env.get("MOCK_LLM_TIMEOUT") == "1"

    def test_clear_injections(self, tmp_path):
        """Test that clear_injections removes injected variables."""
        runner = SkillRunner(tmp_path)
        runner.inject_failure()
        runner.inject_timeout()
        runner.clear_injections()
        env = runner._build_env()

        assert "MOCK_LLM_FAIL" not in env
        assert "MOCK_LLM_TIMEOUT" not in env

    def test_verify_mock_isolation(self, tmp_path):
        """Test that verify_mock_isolation works correctly."""
        runner = SkillRunner(tmp_path)

        # Should return True when call_log is inside tmp_path
        assert runner.verify_mock_isolation() is True

    def test_raises_for_unknown_orchestrator(self, tmp_path):
        """Test that unknown orchestrator names raise ValueError."""
        runner = SkillRunner(tmp_path)

        with pytest.raises(ValueError, match="Unknown orchestrator"):
            runner.run_orchestrator("nonexistent", tmp_path / "plan.md")


class TestFixtureManager:
    """Tests for FixtureManager class."""

    def test_create_plan(self, tmp_path):
        """Test creating a plan fixture."""
        manager = FixtureManager(tmp_path)
        plan = manager.create_plan("test-plan", "# Test Plan\n\nContent here.")

        assert plan.plan_path.exists()
        assert plan.plan_path.read_text(encoding="utf-8") == "# Test Plan\n\nContent here."
        assert plan.name == "test-plan"
        assert plan.output_dir.exists()

    def test_create_plan_with_metadata(self, tmp_path):
        """Test creating a plan fixture with metadata."""
        manager = FixtureManager(tmp_path)
        plan = manager.create_plan(
            "test-plan",
            "# Content",
            metadata={"author": "test", "version": "1.0"},
        )

        assert plan.metadata["author"] == "test"
        assert plan.metadata["version"] == "1.0"

    def test_get_phase_dir(self, tmp_path):
        """Test getting phase directory path."""
        manager = FixtureManager(tmp_path)
        plan = manager.create_plan("test-plan", "# Content")

        phase_dir = plan.get_phase_dir("review-plan")
        assert phase_dir == plan.output_dir / "review-plan"

    def test_ensure_phase_dir(self, tmp_path):
        """Test ensuring phase directory exists."""
        manager = FixtureManager(tmp_path)
        plan = manager.create_plan("test-plan", "# Content")

        phase_dir = plan.ensure_phase_dir("review-plan")
        assert phase_dir.exists()
        assert phase_dir.is_dir()

    def test_create_state_file(self, tmp_path):
        """Test creating a state.json file."""
        manager = FixtureManager(tmp_path)
        plan = manager.create_plan("test-plan", "# Content")

        state_path = manager.create_state_file(
            plan,
            phases_completed=["review-plan"],
            extra_state={"custom_field": "value"},
        )

        assert state_path.exists()
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)

        assert "review-plan" in state["phases_completed"]
        assert state.get("custom_field") == "value"
        assert state["review_phase_completed"] is True

    def test_load_plan_from_fixtures(self, tmp_path):
        """Test loading a plan from the fixtures directory."""
        manager = FixtureManager(tmp_path)

        # This should work if auth-feature.md exists in fixtures/e2e/plans/
        try:
            plan = manager.load_plan("auth-feature")
            assert plan.plan_path.exists()
            assert "auth" in plan.content.lower() or "feature" in plan.content.lower()
        except FileNotFoundError:
            pytest.skip("auth-feature.md fixture not found")

    def test_load_response_fixture(self, tmp_path):
        """Test loading a response fixture."""
        manager = FixtureManager(tmp_path, validate_on_load=False)

        # This should work if valid_suggestions.json exists
        try:
            data = manager.load_response("review_plan", "valid_suggestions")
            assert isinstance(data, (list, dict))
        except FileNotFoundError:
            pytest.skip("valid_suggestions.json fixture not found")

    def test_load_scenario(self, tmp_path):
        """Test loading a scenario path."""
        manager = FixtureManager(tmp_path)

        try:
            scenario_path = manager.load_scenario("happy_path")
            assert scenario_path.exists()
            assert scenario_path.suffix == ".yaml"
        except FileNotFoundError:
            pytest.skip("happy_path.yaml scenario not found")

    def test_create_with_review_phase(self, tmp_path):
        """Test creating a fixture with pre-populated review phase."""
        manager = FixtureManager(tmp_path, validate_on_load=False)

        suggestions = [
            {
                "theme": "Test theme",
                "category": "test",
                "models": ["cursor-agent"],
                "suggestions": [{"title": "Test", "desc": "Description"}],
            }
        ]

        plan = manager.create_with_review_phase(
            "test-plan",
            "# Plan content",
            suggestions=suggestions,
        )

        grouped_path = plan.output_dir / "review-plan" / "grouped.json"
        assert grouped_path.exists()

        with open(grouped_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["theme"] == "Test theme"


class TestMockProvider:
    """Tests for MockProvider class."""

    def test_init(self, tmp_path):
        """Test MockProvider initialization."""
        mock = MockProvider(tmp_path)

        assert mock.tmp_path == tmp_path
        assert mock.call_log_path == tmp_path / "mock_calls.jsonl"

    def test_set_scenario(self, tmp_path):
        """Test setting a scenario."""
        mock = MockProvider(tmp_path)

        # Create a test scenario file
        scenario_path = mock.scenarios_dir / "happy_path.yaml"
        if scenario_path.exists():
            mock.set_scenario("happy_path")
            env = mock.get_env()
            assert env.get("MOCK_LLM_SCENARIO") == str(scenario_path)
        else:
            # Create a temporary scenario
            temp_scenario = tmp_path / "test.yaml"
            temp_scenario.write_text("name: test", encoding="utf-8")
            mock.set_scenario_path(temp_scenario)
            env = mock.get_env()
            assert env.get("MOCK_LLM_SCENARIO") == str(temp_scenario)

    def test_set_response(self, tmp_path):
        """Test setting a direct response."""
        mock = MockProvider(tmp_path)
        mock.set_response({"key": "value"})

        env = mock.get_env()
        assert "MOCK_LLM_CONFIG" in env
        config = json.loads(env["MOCK_LLM_CONFIG"])
        assert config["response"]["key"] == "value"

    def test_inject_failure(self, tmp_path):
        """Test failure injection."""
        mock = MockProvider(tmp_path)
        mock.inject_failure()

        env = mock.get_env()
        assert env.get("MOCK_LLM_FAIL") == "1"
        assert "MOCK_LLM_TIMEOUT" not in env

    def test_inject_timeout(self, tmp_path):
        """Test timeout injection."""
        mock = MockProvider(tmp_path)
        mock.inject_timeout()

        env = mock.get_env()
        assert env.get("MOCK_LLM_TIMEOUT") == "1"
        assert "MOCK_LLM_FAIL" not in env

    def test_method_chaining(self, tmp_path):
        """Test that MockProvider methods support chaining."""
        mock = MockProvider(tmp_path)
        result = mock.set_response({"test": True}).inject_failure()

        assert result is mock

    def test_clear(self, tmp_path):
        """Test clearing all configuration."""
        mock = MockProvider(tmp_path)
        mock.set_response({"test": True}).inject_failure()
        mock.clear()

        env = mock.get_env()
        # Should only have call log path
        assert "MOCK_LLM_CONFIG" not in env
        assert "MOCK_LLM_FAIL" not in env
        assert "MOCK_LLM_CALL_LOG" in env

    def test_get_calls_empty(self, tmp_path):
        """Test getting calls when log is empty."""
        mock = MockProvider(tmp_path)
        calls = mock.get_calls()

        assert calls == []

    def test_get_calls_with_data(self, tmp_path):
        """Test getting calls when log has data."""
        mock = MockProvider(tmp_path)

        # Create a call log
        call_entry = {
            "timestamp": "2025-01-01T00:00:00",
            "provider": "cursor-agent",
            "argv": ["cursor-agent", "--model", "auto", "test prompt"],
            "args": ["--model", "auto", "test prompt"],
            "prompt": "test prompt",
            "env": {"MULTI_LLM_TEST_MODE": "1"},
        }
        with open(mock.call_log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(call_entry) + "\n")

        calls = mock.get_calls()
        assert len(calls) == 1
        assert calls[0].provider == "cursor-agent"
        assert calls[0].prompt == "test prompt"

    def test_was_invoked(self, tmp_path):
        """Test was_invoked check."""
        mock = MockProvider(tmp_path)

        # Should be False with no log
        assert mock.was_invoked() is False

        # Create a call log
        with open(mock.call_log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"provider": "test"}) + "\n")

        assert mock.was_invoked() is True

    def test_create_custom_scenario(self, tmp_path):
        """Test creating a custom scenario."""
        mock = MockProvider(tmp_path)

        prompts = [
            {"pattern": "review.*plan", "fixture": "response.json"},
        ]
        scenario_path = mock.create_custom_scenario(prompts, "custom_test")

        assert scenario_path.exists()
        assert scenario_path.name == "custom_test.yaml"

    def test_create_response_fixture(self, tmp_path):
        """Test creating a custom response fixture."""
        mock = MockProvider(tmp_path)

        data = [{"id": 1, "title": "Test"}]
        fixture_path = mock.create_response_fixture(data, "test_response")

        assert fixture_path.exists()
        with open(fixture_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data


class TestMockLLMCall:
    """Tests for MockLLMCall dataclass."""

    def test_from_dict(self):
        """Test creating MockLLMCall from dictionary."""
        data = {
            "timestamp": "2025-01-01T00:00:00",
            "provider": "gemini",
            "argv": ["gemini", "prompt"],
            "args": ["prompt"],
            "prompt": "test prompt text",
            "env": {"MOCK_LLM_SCENARIO": "/path"},
        }

        call = MockLLMCall.from_dict(data)
        assert call.timestamp == "2025-01-01T00:00:00"
        assert call.provider == "gemini"
        assert call.prompt == "test prompt text"

    def test_prompt_contains(self):
        """Test prompt_contains method."""
        call = MockLLMCall(
            timestamp="",
            provider="test",
            argv=[],
            args=[],
            prompt="Review the implementation plan carefully",
        )

        assert call.prompt_contains("implementation plan")
        assert call.prompt_contains("IMPLEMENTATION")  # case-insensitive
        assert not call.prompt_contains("nonexistent")

    def test_prompt_matches(self):
        """Test prompt_matches method."""
        call = MockLLMCall(
            timestamp="",
            provider="test",
            argv=[],
            args=[],
            prompt="Review the implementation plan carefully",
        )

        assert call.prompt_matches(r"review.*plan")
        assert call.prompt_matches(r"implementation\s+plan")
        assert not call.prompt_matches(r"^plan")


class TestAssertionHelpers:
    """Tests for AssertionHelpers class."""

    def test_assert_output_directory_structure(self, tmp_path):
        """Test assert_output_directory_structure."""
        helpers = AssertionHelpers()

        # Create expected structure
        (tmp_path / "file1.json").write_text("{}", encoding="utf-8")
        (tmp_path / "file2.md").write_text("# Title", encoding="utf-8")
        (tmp_path / "subdir").mkdir()

        # Should pass
        helpers.assert_output_directory_structure(
            tmp_path,
            expected_files=["file1.json", "file2.md"],
            expected_dirs=["subdir"],
        )

        # Should fail with missing file
        with pytest.raises(AssertionError, match="Missing expected files"):
            helpers.assert_output_directory_structure(
                tmp_path,
                expected_files=["nonexistent.json"],
            )

    def test_assert_json_file_valid(self, tmp_path):
        """Test assert_json_file_valid."""
        helpers = AssertionHelpers()

        # Create valid JSON
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}', encoding="utf-8")

        # Should pass
        data = helpers.assert_json_file_valid(
            json_file,
            expected_type=dict,
            expected_keys=["key"],
        )
        assert data["key"] == "value"

        # Should fail with invalid JSON
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not json", encoding="utf-8")

        with pytest.raises(AssertionError, match="Invalid JSON"):
            helpers.assert_json_file_valid(invalid_file)

    def test_assert_stdout_contains(self, tmp_path):
        """Test assert_stdout_contains."""
        helpers = AssertionHelpers()

        result = SkillResult(
            success=True,
            exit_code=0,
            stdout="Successfully completed the operation",
            stderr="",
        )

        # Should pass
        helpers.assert_stdout_contains(result, "Successfully")
        helpers.assert_stdout_contains(result, "complete")  # case-insensitive regex

        # Should fail
        with pytest.raises(AssertionError, match="not found in stdout"):
            helpers.assert_stdout_contains(result, "failed")

    def test_assert_exit_code(self, tmp_path):
        """Test assert_exit_code."""
        helpers = AssertionHelpers()

        result = SkillResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Error occurred",
        )

        # Should pass
        helpers.assert_exit_code(result, 1)

        # Should fail
        with pytest.raises(AssertionError, match="Expected exit code 0"):
            helpers.assert_exit_code(result, 0)

    def test_assert_plan_unchanged(self, tmp_path):
        """Test assert_plan_unchanged."""
        helpers = AssertionHelpers()

        plan_path = tmp_path / "plan.md"
        original_content = "# Original content"
        plan_path.write_text(original_content, encoding="utf-8")

        # Should pass
        helpers.assert_plan_unchanged(plan_path, original_content)

        # Modify the file
        plan_path.write_text("# Modified content", encoding="utf-8")

        # Should fail
        with pytest.raises(AssertionError, match="was modified"):
            helpers.assert_plan_unchanged(plan_path, original_content)

    def test_assert_plan_modified(self, tmp_path):
        """Test assert_plan_modified."""
        helpers = AssertionHelpers()

        plan_path = tmp_path / "plan.md"
        original_content = "# Original content"
        plan_path.write_text(original_content, encoding="utf-8")

        # Should fail when unchanged
        with pytest.raises(AssertionError, match="was not modified"):
            helpers.assert_plan_modified(plan_path, original_content)

        # Modify the file
        plan_path.write_text("# Modified content", encoding="utf-8")

        # Should pass
        helpers.assert_plan_modified(plan_path, original_content)


class TestSkillResult:
    """Tests for SkillResult dataclass."""

    def test_get_state(self, tmp_path):
        """Test get_state method."""
        # Create output directory with state.json
        output_dir = tmp_path / "plan"
        output_dir.mkdir()
        state_file = output_dir / "state.json"
        state_file.write_text('{"phases_completed": {"review-plan": "2025-01-01"}}', encoding="utf-8")

        result = SkillResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            output_dir=output_dir,
        )

        state = result.get_state()
        assert state is not None
        assert "review-plan" in state["phases_completed"]

    def test_get_mock_calls(self, tmp_path):
        """Test get_mock_calls method."""
        call_log = tmp_path / "calls.jsonl"
        call_log.write_text(
            '{"provider": "cursor-agent", "prompt": "test1"}\n'
            '{"provider": "gemini", "prompt": "test2"}\n',
            encoding="utf-8",
        )

        result = SkillResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            call_log_path=call_log,
        )

        calls = result.get_mock_calls()
        assert len(calls) == 2
        assert calls[0]["provider"] == "cursor-agent"
        assert calls[1]["provider"] == "gemini"

    def test_mock_was_invoked(self, tmp_path):
        """Test mock_was_invoked method."""
        call_log = tmp_path / "calls.jsonl"

        # Without call log
        result = SkillResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            call_log_path=call_log,
        )
        assert result.mock_was_invoked() is False

        # With call log
        call_log.write_text('{"provider": "test"}\n', encoding="utf-8")
        assert result.mock_was_invoked() is True
