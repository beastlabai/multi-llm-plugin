"""Tests for mock_llm.py infrastructure.

These tests verify that the mock_llm.py binary works correctly and can be
used reliably for end-to-end integration tests. Tests cover:
- CLI argument parsing for all 5 providers
- Provider detection via sys.argv[0]
- Scenario-based prompt pattern matching
- Output wire format correctness for each provider
- Call logging to JSONL files
- Error injection (timeout, malformed JSON, rate limit)
"""

import json
import os
import subprocess
import sys
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any, List

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Path to the mock_llm.py binary
MOCK_LLM_PATH = Path(__file__).parent / "mocks" / "mock_llm.py"

# All supported providers
PROVIDERS = ["cursor-agent", "gemini", "opencode", "codex", "kilocode"]


def run_mock_llm(
    provider: str,
    args: List[str],
    env_vars: Dict[str, str] = None,
    tmp_dir: Path = None,
) -> subprocess.CompletedProcess:
    """Run mock_llm.py simulating a specific provider.

    Args:
        provider: Provider name to simulate via symlink or env var
        args: Command-line arguments to pass
        env_vars: Additional environment variables to set
        tmp_dir: Temporary directory for creating symlinks

    Returns:
        CompletedProcess with stdout, stderr, and returncode
    """
    env = os.environ.copy()
    env["MULTI_LLM_TEST_MODE"] = "1"

    if env_vars:
        env.update(env_vars)

    if tmp_dir:
        # Create a symlink with the provider name
        symlink_path = tmp_dir / provider
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(MOCK_LLM_PATH)

        # Run via symlink
        cmd = [str(symlink_path)] + args
    else:
        # Run directly with MOCK_LLM_PROVIDER env var
        env["MOCK_LLM_PROVIDER"] = provider
        cmd = [sys.executable, str(MOCK_LLM_PATH)] + args

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )


class TestMockLLMArgumentParsing:
    """Tests for CLI argument parsing for different providers."""

    def test_cursor_agent_args(self, tmp_path):
        """Test cursor-agent CLI argument parsing."""
        result = run_mock_llm(
            "cursor-agent",
            ["--print", "-f", "--output-format", "json", "--model", "auto", "test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        # Output should be in cursor-agent format
        output = json.loads(result.stdout)
        assert output.get("type") == "result"
        assert "result" in output

    def test_gemini_args(self, tmp_path):
        """Test gemini CLI argument parsing."""
        result = run_mock_llm(
            "gemini",
            ["--output-format", "json", "--model", "gemini-2.0-flash", "test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "session_id" in output
        assert "response" in output
        assert "stats" in output

    def test_opencode_run_subcommand(self, tmp_path):
        """Test opencode 'run' subcommand parsing."""
        result = run_mock_llm(
            "opencode",
            ["run", "--format", "json", "--model", "claude-sonnet", "test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        # Output should be NDJSON events
        lines = result.stdout.strip().split("\n")
        assert len(lines) >= 1
        for line in lines:
            event = json.loads(line)
            assert "type" in event

    def test_opencode_requires_run_subcommand(self, tmp_path):
        """Test opencode fails without 'run' subcommand."""
        result = run_mock_llm(
            "opencode",
            ["--format", "json", "test prompt"],
            tmp_dir=tmp_path,
        )

        # Should fail because 'run' subcommand is required
        assert result.returncode != 0

    def test_codex_exec_subcommand(self, tmp_path):
        """Test codex 'exec' subcommand parsing."""
        result = run_mock_llm(
            "codex",
            ["exec", "--full-auto", "--json", "--model", "gpt-4", "test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        for line in lines:
            event = json.loads(line)
            assert "type" in event

    def test_codex_requires_exec_subcommand(self, tmp_path):
        """Test codex fails without 'exec' subcommand."""
        result = run_mock_llm(
            "codex",
            ["--json", "test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode != 0

    def test_kilocode_args(self, tmp_path):
        """Test kilocode CLI argument parsing."""
        result = run_mock_llm(
            "kilocode",
            ["--auto", "--json", "test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        # Output should be valid JSON
        json.loads(result.stdout)


class TestMockLLMProviderDetection:
    """Tests for provider detection via sys.argv[0]."""

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_provider_detected_from_symlink(self, tmp_path, provider):
        """Test that provider is correctly detected from symlink name."""
        call_log = tmp_path / "calls.jsonl"

        # Provider-specific command args
        if provider == "opencode":
            args = ["run", "test prompt"]
        elif provider == "codex":
            args = ["exec", "test prompt"]
        else:
            args = ["test prompt"]

        result = run_mock_llm(
            provider,
            args,
            env_vars={"MOCK_LLM_CALL_LOG": str(call_log)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0

        # Check call log for provider detection
        assert call_log.exists()
        with open(call_log) as f:
            entry = json.loads(f.read().strip())

        assert entry["provider"] == provider

    def test_provider_from_env_var(self, tmp_path):
        """Test provider detection from MOCK_LLM_PROVIDER env var."""
        call_log = tmp_path / "calls.jsonl"

        # Run directly without symlink
        env = os.environ.copy()
        env["MULTI_LLM_TEST_MODE"] = "1"
        env["MOCK_LLM_PROVIDER"] = "gemini"
        env["MOCK_LLM_CALL_LOG"] = str(call_log)

        result = subprocess.run(
            [sys.executable, str(MOCK_LLM_PATH), "test prompt"],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )

        assert result.returncode == 0

        with open(call_log) as f:
            entry = json.loads(f.read().strip())

        assert entry["provider"] == "gemini"

    def test_unknown_provider_defaults_gracefully(self, tmp_path):
        """Test that unknown provider name is handled gracefully."""
        env = os.environ.copy()
        env["MULTI_LLM_TEST_MODE"] = "1"

        result = subprocess.run(
            [sys.executable, str(MOCK_LLM_PATH), "test prompt"],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )

        # Should still succeed with default/unknown provider
        assert result.returncode == 0


class TestMockLLMPatternMatching:
    """Tests for scenario-based prompt pattern matching."""

    def test_pattern_matching_basic(self, tmp_path):
        """Test basic pattern matching from scenario file."""
        # Create a fixture file
        fixture_content = [{"id": "test-1", "message": "Hello from fixture"}]
        fixture_path = tmp_path / "test_fixture.json"
        fixture_path.write_text(json.dumps(fixture_content))

        # Create a scenario file
        scenario = {
            "name": "test_scenario",
            "prompts": [
                {"pattern": "hello.*world", "fixture": "test_fixture.json"},
                {"pattern": "review.*plan", "fixture": "test_fixture.json"},
            ]
        }
        scenario_path = tmp_path / "scenario.yaml"

        import yaml
        with open(scenario_path, "w") as f:
            yaml.safe_dump(scenario, f)

        result = run_mock_llm(
            "cursor-agent",
            ["hello world test"],
            env_vars={"MOCK_LLM_SCENARIO": str(scenario_path)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        # cursor-agent wraps in {"type": "result", "result": "..."}
        result_content = json.loads(output["result"])
        assert result_content[0]["id"] == "test-1"

    def test_pattern_matching_case_insensitive(self, tmp_path):
        """Test that pattern matching is case-insensitive."""
        fixture_content = {"matched": True}
        fixture_path = tmp_path / "matched.json"
        fixture_path.write_text(json.dumps(fixture_content))

        scenario = {
            "name": "case_test",
            "prompts": [
                {"pattern": "UPPERCASE", "fixture": "matched.json"},
            ]
        }
        scenario_path = tmp_path / "scenario.yaml"

        import yaml
        with open(scenario_path, "w") as f:
            yaml.safe_dump(scenario, f)

        # Use lowercase in prompt
        result = run_mock_llm(
            "cursor-agent",
            ["testing uppercase matching"],
            env_vars={"MOCK_LLM_SCENARIO": str(scenario_path)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        result_content = json.loads(output["result"])
        assert result_content["matched"] is True

    def test_pattern_no_match_returns_default(self, tmp_path):
        """Test that unmatched prompts return default response."""
        scenario = {
            "name": "no_match_test",
            "prompts": [
                {"pattern": "specific_pattern_xyz123", "fixture": "not_used.json"},
            ]
        }
        scenario_path = tmp_path / "scenario.yaml"

        import yaml
        with open(scenario_path, "w") as f:
            yaml.safe_dump(scenario, f)

        result = run_mock_llm(
            "cursor-agent",
            ["this will not match anything"],
            env_vars={"MOCK_LLM_SCENARIO": str(scenario_path)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        # Should return default mock response
        output = json.loads(result.stdout)
        assert output["type"] == "result"

    def test_first_matching_pattern_wins(self, tmp_path):
        """Test that first matching pattern takes precedence."""
        first_fixture = {"order": "first"}
        first_path = tmp_path / "first.json"
        first_path.write_text(json.dumps(first_fixture))

        second_fixture = {"order": "second"}
        second_path = tmp_path / "second.json"
        second_path.write_text(json.dumps(second_fixture))

        scenario = {
            "name": "order_test",
            "prompts": [
                {"pattern": "test", "fixture": "first.json"},
                {"pattern": "test.*prompt", "fixture": "second.json"},
            ]
        }
        scenario_path = tmp_path / "scenario.yaml"

        import yaml
        with open(scenario_path, "w") as f:
            yaml.safe_dump(scenario, f)

        result = run_mock_llm(
            "cursor-agent",
            ["test prompt here"],
            env_vars={"MOCK_LLM_SCENARIO": str(scenario_path)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        result_content = json.loads(output["result"])
        assert result_content["order"] == "first"


class TestMockLLMOutputFormats:
    """Tests for correct wire format output for each provider."""

    def test_cursor_agent_format(self, tmp_path):
        """Test cursor-agent outputs correct wire format."""
        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)

        # cursor-agent format: {"type": "result", "result": "<content>"}
        assert output["type"] == "result"
        assert "result" in output
        assert isinstance(output["result"], str)

    def test_gemini_format(self, tmp_path):
        """Test gemini outputs correct wire format."""
        result = run_mock_llm(
            "gemini",
            ["test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)

        # gemini format: {"session_id": "...", "response": "<content>", "stats": {...}}
        assert "session_id" in output
        assert "response" in output
        assert "stats" in output
        assert "input_tokens" in output["stats"]
        assert "output_tokens" in output["stats"]

    def test_opencode_format(self, tmp_path):
        """Test opencode outputs correct NDJSON wire format."""
        result = run_mock_llm(
            "opencode",
            ["run", "test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0

        # opencode format: NDJSON events (step_start, text, step_finish)
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 3

        events = [json.loads(line) for line in lines]
        event_types = [e["type"] for e in events]

        assert event_types[0] == "step_start"
        assert event_types[1] == "text"
        assert event_types[2] == "step_finish"

        # Text event should have part.text
        text_event = events[1]
        assert "part" in text_event
        assert "text" in text_event["part"]

    def test_codex_format(self, tmp_path):
        """Test codex outputs correct NDJSON wire format."""
        result = run_mock_llm(
            "codex",
            ["exec", "test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0

        # codex format: NDJSON events with text type
        lines = result.stdout.strip().split("\n")
        assert len(lines) >= 1

        events = [json.loads(line) for line in lines]

        # Should have text event with part.text
        text_event = events[0]
        assert text_event["type"] == "text"
        assert "part" in text_event
        assert "text" in text_event["part"]

    def test_kilocode_format(self, tmp_path):
        """Test kilocode outputs correct wire format."""
        result = run_mock_llm(
            "kilocode",
            ["test prompt"],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0

        # kilocode format: Direct JSON output
        output = json.loads(result.stdout)
        # Content is either raw JSON array/object or wrapped in {"output": ...}
        assert isinstance(output, (dict, list))

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_output_is_valid_json(self, tmp_path, provider):
        """Test all providers output valid JSON/NDJSON."""
        # Provider-specific command args
        if provider == "opencode":
            args = ["run", "test prompt"]
        elif provider == "codex":
            args = ["exec", "test prompt"]
        else:
            args = ["test prompt"]

        result = run_mock_llm(
            provider,
            args,
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0

        # All output should be valid JSON or NDJSON
        lines = result.stdout.strip().split("\n")
        for line in lines:
            json.loads(line)  # Should not raise


class TestMockLLMCallLogging:
    """Tests for JSONL call logging."""

    def test_call_log_created(self, tmp_path):
        """Test that call log file is created when configured."""
        call_log = tmp_path / "calls.jsonl"

        run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_CALL_LOG": str(call_log)},
            tmp_dir=tmp_path,
        )

        assert call_log.exists()

    def test_call_log_contains_timestamp(self, tmp_path):
        """Test that call log entry contains timestamp."""
        call_log = tmp_path / "calls.jsonl"

        run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_CALL_LOG": str(call_log)},
            tmp_dir=tmp_path,
        )

        with open(call_log) as f:
            entry = json.loads(f.read().strip())

        assert "timestamp" in entry
        # Should be ISO format
        assert "T" in entry["timestamp"]

    def test_call_log_contains_provider(self, tmp_path):
        """Test that call log entry contains provider name."""
        call_log = tmp_path / "calls.jsonl"

        run_mock_llm(
            "gemini",
            ["test prompt"],
            env_vars={"MOCK_LLM_CALL_LOG": str(call_log)},
            tmp_dir=tmp_path,
        )

        with open(call_log) as f:
            entry = json.loads(f.read().strip())

        assert entry["provider"] == "gemini"

    def test_call_log_contains_argv(self, tmp_path):
        """Test that call log entry contains full argv."""
        call_log = tmp_path / "calls.jsonl"

        run_mock_llm(
            "cursor-agent",
            ["--model", "auto", "test prompt"],
            env_vars={"MOCK_LLM_CALL_LOG": str(call_log)},
            tmp_dir=tmp_path,
        )

        with open(call_log) as f:
            entry = json.loads(f.read().strip())

        assert "argv" in entry
        assert isinstance(entry["argv"], list)
        assert "test prompt" in entry["argv"]

    def test_call_log_contains_prompt(self, tmp_path):
        """Test that call log entry contains parsed prompt."""
        call_log = tmp_path / "calls.jsonl"

        run_mock_llm(
            "cursor-agent",
            ["my specific test prompt"],
            env_vars={"MOCK_LLM_CALL_LOG": str(call_log)},
            tmp_dir=tmp_path,
        )

        with open(call_log) as f:
            entry = json.loads(f.read().strip())

        assert entry["prompt"] == "my specific test prompt"

    def test_call_log_contains_env(self, tmp_path):
        """Test that call log entry contains relevant env vars."""
        call_log = tmp_path / "calls.jsonl"

        run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={
                "MOCK_LLM_CALL_LOG": str(call_log),
                "MOCK_LLM_SCENARIO": "/some/path",
            },
            tmp_dir=tmp_path,
        )

        with open(call_log) as f:
            entry = json.loads(f.read().strip())

        assert "env" in entry
        assert entry["env"].get("MULTI_LLM_TEST_MODE") == "1"
        assert entry["env"].get("MOCK_LLM_SCENARIO") == "/some/path"

    def test_call_log_appends_multiple_calls(self, tmp_path):
        """Test that multiple calls are appended to the same log file."""
        call_log = tmp_path / "calls.jsonl"

        # Make 3 calls
        for i in range(3):
            run_mock_llm(
                "cursor-agent",
                [f"prompt {i}"],
                env_vars={"MOCK_LLM_CALL_LOG": str(call_log)},
                tmp_dir=tmp_path,
            )

        # Read all entries
        with open(call_log) as f:
            lines = f.readlines()

        assert len(lines) == 3

        entries = [json.loads(line) for line in lines]
        prompts = [e["prompt"] for e in entries]
        assert prompts == ["prompt 0", "prompt 1", "prompt 2"]

    def test_call_log_creates_parent_directories(self, tmp_path):
        """Test that call log creates parent directories if needed."""
        call_log = tmp_path / "deep" / "nested" / "calls.jsonl"

        run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_CALL_LOG": str(call_log)},
            tmp_dir=tmp_path,
        )

        assert call_log.exists()


class TestMockLLMErrorInjection:
    """Tests for error injection (timeout, failure, malformed responses)."""

    def test_failure_injection(self, tmp_path):
        """Test MOCK_LLM_FAIL=1 causes immediate failure."""
        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_FAIL": "1"},
            tmp_dir=tmp_path,
        )

        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "fail" in result.stderr.lower()

    def test_timeout_injection_exits_quickly(self, tmp_path):
        """Test MOCK_LLM_TIMEOUT=1 exits quickly with timeout error."""
        import time

        start = time.time()
        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_TIMEOUT": "1"},
            tmp_dir=tmp_path,
        )
        elapsed = time.time() - start

        # Should exit quickly (not actually wait for real timeout)
        assert elapsed < 1.0

        # Should return timeout-like exit code
        assert result.returncode == 124  # Standard timeout exit code
        assert "timeout" in result.stderr.lower()

    def test_failure_still_logs_call(self, tmp_path):
        """Test that failed calls are still logged."""
        call_log = tmp_path / "calls.jsonl"

        run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={
                "MOCK_LLM_CALL_LOG": str(call_log),
                "MOCK_LLM_FAIL": "1",
            },
            tmp_dir=tmp_path,
        )

        # Call should still be logged even though it failed
        assert call_log.exists()
        with open(call_log) as f:
            entry = json.loads(f.read().strip())
        assert entry["prompt"] == "test prompt"

    def test_timeout_still_logs_call(self, tmp_path):
        """Test that timeout calls are still logged."""
        call_log = tmp_path / "calls.jsonl"

        run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={
                "MOCK_LLM_CALL_LOG": str(call_log),
                "MOCK_LLM_TIMEOUT": "1",
            },
            tmp_dir=tmp_path,
        )

        # Call should still be logged even though it timed out
        assert call_log.exists()

    def test_test_mode_required(self, tmp_path):
        """Test that mock_llm.py requires MULTI_LLM_TEST_MODE=1."""
        # Create symlink
        symlink_path = tmp_path / "cursor-agent"
        symlink_path.symlink_to(MOCK_LLM_PATH)

        # Run WITHOUT MULTI_LLM_TEST_MODE
        env = os.environ.copy()
        env.pop("MULTI_LLM_TEST_MODE", None)

        result = subprocess.run(
            [str(symlink_path), "test prompt"],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )

        assert result.returncode != 0
        assert "test mode" in result.stderr.lower()


class TestMockLLMLegacyMode:
    """Tests for legacy environment variable mode."""

    def test_legacy_fixture_path(self, tmp_path):
        """Test MOCK_LLM_FIXTURE takes precedence."""
        # Create a fixture file
        fixture_content = {"legacy": True, "message": "from fixture file"}
        fixture_path = tmp_path / "legacy_fixture.json"
        fixture_path.write_text(json.dumps(fixture_content))

        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_FIXTURE": str(fixture_path)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        result_content = json.loads(output["result"])
        assert result_content["legacy"] is True
        assert result_content["message"] == "from fixture file"

    def test_legacy_fixture_over_scenario(self, tmp_path):
        """Test MOCK_LLM_FIXTURE takes precedence over scenario."""
        # Create fixture
        fixture_content = {"source": "legacy_fixture"}
        fixture_path = tmp_path / "legacy.json"
        fixture_path.write_text(json.dumps(fixture_content))

        # Create scenario that would match
        scenario_fixture = {"source": "scenario_fixture"}
        scenario_fixture_path = tmp_path / "scenario_response.json"
        scenario_fixture_path.write_text(json.dumps(scenario_fixture))

        scenario = {
            "name": "test",
            "prompts": [{"pattern": ".*", "fixture": "scenario_response.json"}]
        }
        scenario_path = tmp_path / "scenario.yaml"

        import yaml
        with open(scenario_path, "w") as f:
            yaml.safe_dump(scenario, f)

        # Run with both set - legacy should win
        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={
                "MOCK_LLM_FIXTURE": str(fixture_path),
                "MOCK_LLM_SCENARIO": str(scenario_path),
            },
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        result_content = json.loads(output["result"])
        assert result_content["source"] == "legacy_fixture"

    def test_legacy_output_path(self, tmp_path):
        """Test MOCK_OUTPUT_PATH writes response to file."""
        output_file = tmp_path / "output" / "response.json"

        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_OUTPUT_PATH": str(output_file)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        assert output_file.exists()

        # File should contain the response content
        content = output_file.read_text()
        # Should be valid JSON
        json.loads(content)


class TestMockLLMConfigMode:
    """Tests for dynamic configuration via MOCK_LLM_CONFIG."""

    def test_config_response_string(self, tmp_path):
        """Test MOCK_LLM_CONFIG with string response."""
        config = {"response": "Hello from config"}

        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_CONFIG": json.dumps(config)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["result"] == "Hello from config"

    def test_config_response_dict(self, tmp_path):
        """Test MOCK_LLM_CONFIG with dict response."""
        config = {"response": {"id": "123", "data": "from config"}}

        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_CONFIG": json.dumps(config)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        result_content = json.loads(output["result"])
        assert result_content["id"] == "123"
        assert result_content["data"] == "from config"

    def test_config_invalid_json_ignored(self, tmp_path):
        """Test that invalid MOCK_LLM_CONFIG JSON is handled gracefully."""
        result = run_mock_llm(
            "cursor-agent",
            ["test prompt"],
            env_vars={"MOCK_LLM_CONFIG": "not valid json"},
            tmp_dir=tmp_path,
        )

        # Should still succeed with default response
        assert result.returncode == 0


class TestMockLLMOutputFileParsing:
    """Tests for OUTPUT_FILE directive in prompts."""

    def test_output_file_from_prompt(self, tmp_path):
        """Test OUTPUT_FILE: directive in prompt writes response."""
        output_file = tmp_path / "from_prompt" / "output.json"
        prompt = f"Some task\nOUTPUT_FILE: {output_file}\nMore instructions"

        result = run_mock_llm(
            "cursor-agent",
            [prompt],
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        assert output_file.exists()

    def test_env_output_path_overrides_prompt(self, tmp_path):
        """Test MOCK_OUTPUT_PATH takes precedence over OUTPUT_FILE in prompt."""
        env_output = tmp_path / "env_output.json"
        prompt_output = tmp_path / "prompt_output.json"
        prompt = f"Task\nOUTPUT_FILE: {prompt_output}\nDone"

        result = run_mock_llm(
            "cursor-agent",
            [prompt],
            env_vars={"MOCK_OUTPUT_PATH": str(env_output)},
            tmp_dir=tmp_path,
        )

        assert result.returncode == 0
        assert env_output.exists()
        # Note: Both may exist, but env path should definitely exist
