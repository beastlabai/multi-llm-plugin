"""Comprehensive unit tests for llm_client module.

This module tests the LLM client wrapper functionality including:
- check_cursor_agent_available() - binary availability check
- _save_log() - log file saving
- invoke_with_provider() - primary LLM invocation
- _is_valid_parsed_data() - parsed data validation
- invoke_with_file_output() - LLM with file output
- invoke_subagent() - backward-compatible wrapper with retry logic
- parse_subagent_response() - structured output parsing
- _extract_json_from_text() - JSON extraction from text
- invoke_for_json() - invoke expecting JSON response

All tests use mocking to avoid actual LLM provider calls.
"""

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import (
    LLMClientError,
    SubagentTimeoutError,
    ERROR_TIMEOUT,
    ERROR_PARSE_ERROR,
    ERROR_BINARY_NOT_FOUND,
    ERROR_SUBPROCESS_FAILED,
    ERROR_FILE_NOT_FOUND,
    ERROR_PROMPT_TOO_LONG,
    ERROR_PROMPT_UNSAFE,
    CMDLINE_CAP_UTF16_BATCH,
    CMDLINE_CAP_UTF16_NATIVE,
    CMDLINE_UTF16_HEADROOM,
    check_cursor_agent_available,
    _save_log,
    _resolve_executable,
    _utf16_code_units,
    _prompt_length_error,
    _batch_shim_metachar_error,
    invoke_with_provider,
    _is_valid_parsed_data,
    invoke_with_file_output,
    invoke_subagent,
    parse_subagent_response,
    _extract_json_from_text,
    invoke_for_json,
)


class TestCheckCursorAgentAvailable:
    """Tests for check_cursor_agent_available() function."""

    @patch("shutil.which")
    def test_cursor_agent_binary_exists(self, mock_which):
        """Return True when cursor-agent binary is found in PATH."""
        mock_which.return_value = "/usr/local/bin/cursor-agent"

        result = check_cursor_agent_available()

        assert result is True
        mock_which.assert_called_once_with("cursor-agent")

    @patch("shutil.which")
    def test_cursor_agent_binary_missing(self, mock_which):
        """Return False when cursor-agent binary is not found."""
        mock_which.return_value = None

        result = check_cursor_agent_available()

        assert result is False
        mock_which.assert_called_once_with("cursor-agent")

    @patch("shutil.which")
    def test_cursor_agent_binary_alternative_path(self, mock_which):
        """Return True for cursor-agent in alternative location."""
        mock_which.return_value = "/home/user/.local/bin/cursor-agent"

        result = check_cursor_agent_available()

        assert result is True


class TestSaveLog:
    """Tests for _save_log() function."""

    def test_save_log_successful_write(self, tmp_path):
        """Successfully save log file with all fields."""
        log_file = tmp_path / "test_log.txt"

        result = _save_log(
            log_file=log_file,
            model="cursor-agent:gpt-4",
            prompt="Test prompt content",
            stdout="Test stdout output",
            stderr="Test stderr output",
            returncode=0,
            success=True,
            error=None,
            duration_seconds=5.5,
        )

        assert result is True
        assert log_file.exists()

        content = log_file.read_text()
        assert "CURSOR-AGENT LOG" in content
        assert "cursor-agent:gpt-4" in content
        assert "Test prompt content" in content
        assert "Test stdout output" in content
        assert "Test stderr output" in content
        assert "Success: True" in content
        assert "Return Code: 0" in content
        assert "Duration: 5.5s" in content

    def test_save_log_with_error(self, tmp_path):
        """Save log file with error message."""
        log_file = tmp_path / "error_log.txt"

        result = _save_log(
            log_file=log_file,
            model="gemini:pro",
            prompt="Failed prompt",
            stdout="",
            stderr="Connection error",
            returncode=1,
            success=False,
            error="Provider timed out",
            duration_seconds=30.0,
        )

        assert result is True
        content = log_file.read_text()
        assert "Success: False" in content
        assert "Return Code: 1" in content
        assert "Error: Provider timed out" in content
        assert "(empty)" in content  # For empty stdout

    def test_save_log_creates_parent_directories(self, tmp_path):
        """Create parent directories if they don't exist."""
        log_file = tmp_path / "nested" / "deep" / "log.txt"

        result = _save_log(
            log_file=log_file,
            model="test-model",
            prompt="prompt",
            stdout="output",
            stderr="",
            returncode=0,
            success=True,
        )

        assert result is True
        assert log_file.exists()
        assert log_file.parent.exists()

    def test_save_log_truncates_long_prompt(self, tmp_path):
        """Truncate prompts longer than MAX_LOGGED_PROMPT_LENGTH."""
        log_file = tmp_path / "long_prompt.txt"
        long_prompt = "x" * 10000

        result = _save_log(
            log_file=log_file,
            model="test",
            prompt=long_prompt,
            stdout="",
            stderr="",
            returncode=0,
            success=True,
        )

        assert result is True
        content = log_file.read_text()
        # Should be truncated with "..."
        assert "..." in content
        # Should not contain the full 10000 chars
        assert len(content) < 10000 + 1000  # Some overhead for log formatting

    def test_save_log_io_error(self, tmp_path):
        """Return False on IO error."""
        # Use a path that can't be written to
        log_file = Path("/nonexistent/readonly/path/log.txt")

        result = _save_log(
            log_file=log_file,
            model="test",
            prompt="test",
            stdout="",
            stderr="",
            returncode=0,
            success=True,
        )

        assert result is False

    def test_save_log_empty_outputs(self, tmp_path):
        """Handle empty stdout and stderr gracefully."""
        log_file = tmp_path / "empty.txt"

        result = _save_log(
            log_file=log_file,
            model="test",
            prompt="test prompt",
            stdout="",
            stderr="",
            returncode=0,
            success=True,
        )

        assert result is True
        content = log_file.read_text()
        # Empty values should be replaced with "(empty)"
        assert "(empty)" in content

    def test_save_log_path_as_string(self, tmp_path):
        """Accept string path instead of Path object."""
        log_file = str(tmp_path / "string_path.txt")

        result = _save_log(
            log_file=log_file,
            model="test",
            prompt="test",
            stdout="output",
            stderr="",
            returncode=0,
            success=True,
        )

        assert result is True
        assert Path(log_file).exists()


class TestInvokeWithProvider:
    """Tests for invoke_with_provider() function."""

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock for subprocess.run."""
        with patch("utils.llm_client.subprocess.run") as mock:
            yield mock

    @pytest.fixture
    def mock_provider_available(self):
        """Mock provider to appear available."""
        with patch("shutil.which", return_value="/usr/bin/cursor-agent"):
            yield

    def test_success_returns_parsed_data(self, mock_subprocess, mock_provider_available):
        """Successful invocation returns parsed data with details."""
        inner_data = [{"title": "Test", "importance": "high"}]
        wrapper = {"type": "result", "result": json.dumps(inner_data)}

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(wrapper)
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4",
        )

        assert result["success"] is True
        assert result["data"] == inner_data
        assert result["details"]["provider"] == "cursor-agent"
        assert result["details"]["model"] == "gpt-4"
        assert "duration_seconds" in result["details"]

    def test_timeout_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Timeout returns TIMEOUT error code."""
        exc = subprocess.TimeoutExpired(cmd=["cursor-agent"], timeout=600)
        exc.stdout = "partial output"
        exc.stderr = "stderr"
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4",
            timeout=600,
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert "timed out" in result["error"].lower()
        assert result["details"]["timeout"] == 600

    def test_timeout_with_none_output(self, mock_subprocess, mock_provider_available):
        """Handle timeout when stdout/stderr are None."""
        exc = subprocess.TimeoutExpired(cmd=["cursor-agent"], timeout=300)
        exc.stdout = None
        exc.stderr = None
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:auto",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT

    def test_timeout_with_bytes_output(self, mock_subprocess, mock_provider_available):
        """Handle timeout when stdout/stderr are bytes."""
        exc = subprocess.TimeoutExpired(cmd=["cursor-agent"], timeout=300)
        exc.stdout = b"bytes output"
        exc.stderr = b"bytes error"
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:auto",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT

    def test_parse_error_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Unparseable output returns PARSE_ERROR error code."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is plain text without any JSON"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_PARSE_ERROR

    def test_binary_not_found_returns_error_code(self):
        """Unavailable binary returns BINARY_NOT_FOUND error code."""
        with patch("shutil.which", return_value=None):
            result = invoke_with_provider(
                prompt="Test",
                model_spec="cursor-agent:gpt-4",
            )

            assert result["success"] is False
            assert result["error_code"] == ERROR_BINARY_NOT_FOUND
            assert "not found" in result["error"].lower()

    def test_unknown_provider_returns_error(self):
        """Unknown provider returns BINARY_NOT_FOUND error code."""
        result = invoke_with_provider(
            prompt="Test",
            model_spec="nonexistent-provider:model",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_BINARY_NOT_FOUND
        assert "nonexistent-provider" in result["error"]

    def test_subprocess_failed_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Non-zero exit code returns SUBPROCESS_FAILED error code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Command failed: invalid arguments"
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "exited with code 1" in result["error"]
        assert result["details"]["exit_code"] == 1
        assert result["details"]["stderr"] == "Command failed: invalid arguments"

    def test_subprocess_failed_various_exit_codes(self, mock_subprocess, mock_provider_available):
        """Test SUBPROCESS_FAILED with various exit codes."""
        for exit_code in [1, 2, 127, 255]:
            mock_result = MagicMock()
            mock_result.returncode = exit_code
            mock_result.stdout = ""
            mock_result.stderr = f"exit code {exit_code}"
            mock_subprocess.return_value = mock_result

            result = invoke_with_provider(
                prompt="Test",
                model_spec="cursor-agent:gpt-4",
            )

            assert result["success"] is False
            assert result["error_code"] == ERROR_SUBPROCESS_FAILED
            assert result["details"]["exit_code"] == exit_code

    def test_uses_provider_default_timeout(self, mock_subprocess, mock_provider_available):
        """Use provider's default timeout when not specified."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        call_kwargs = mock_subprocess.call_args
        assert call_kwargs.kwargs["timeout"] == 1200  # cursor-agent default

    def test_uses_custom_timeout(self, mock_subprocess, mock_provider_available):
        """Use custom timeout when specified."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
            timeout=300,
        )

        call_kwargs = mock_subprocess.call_args
        assert call_kwargs.kwargs["timeout"] == 300

    def test_log_file_saved_on_success(self, mock_subprocess, mock_provider_available, tmp_path):
        """Save log file on successful invocation."""
        log_file = tmp_path / "success.log"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4",
            log_file=log_file,
        )

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test prompt" in content

    def test_log_file_saved_on_failure(self, mock_subprocess, mock_provider_available, tmp_path):
        """Save log file on failed invocation."""
        log_file = tmp_path / "failure.log"
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "error output"
        mock_result.stderr = "detailed error"
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
            log_file=log_file,
        )

        assert log_file.exists()
        content = log_file.read_text()
        assert "Success: False" in content

    def test_log_file_saved_on_timeout(self, mock_subprocess, mock_provider_available, tmp_path):
        """Save log file on timeout."""
        log_file = tmp_path / "timeout.log"
        exc = subprocess.TimeoutExpired(cmd=["cursor-agent"], timeout=600)
        exc.stdout = "partial"
        exc.stderr = ""
        mock_subprocess.side_effect = exc

        invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
            log_file=log_file,
        )

        assert log_file.exists()
        content = log_file.read_text()
        assert "timed out" in content.lower()

    def test_duration_tracking(self, mock_subprocess, mock_provider_available):
        """Track duration in result details."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        assert "duration_seconds" in result["details"]
        assert isinstance(result["details"]["duration_seconds"], float)
        assert result["details"]["duration_seconds"] >= 0


class TestIsValidParsedData:
    """Tests for _is_valid_parsed_data() function."""

    def test_valid_dict(self):
        """Return True for non-empty dict."""
        assert _is_valid_parsed_data({"key": "value"}) is True
        assert _is_valid_parsed_data({"a": 1, "b": 2}) is True

    def test_valid_list(self):
        """Return True for list (empty or non-empty)."""
        assert _is_valid_parsed_data([]) is True
        assert _is_valid_parsed_data([1, 2, 3]) is True
        assert _is_valid_parsed_data([{"item": 1}]) is True

    def test_raw_wrapper_is_invalid(self):
        """Return False for raw wrapper dict."""
        assert _is_valid_parsed_data({"raw": "some text"}) is False
        assert _is_valid_parsed_data({"raw": ""}) is False

    def test_none_is_invalid(self):
        """Return False for None."""
        assert _is_valid_parsed_data(None) is False

    def test_empty_string_is_invalid(self):
        """Return False for empty string."""
        assert _is_valid_parsed_data("") is False

    def test_empty_raw_dict_is_invalid(self):
        """Return False for empty raw dict."""
        assert _is_valid_parsed_data({"raw": ""}) is False

    def test_dict_with_raw_and_other_keys(self):
        """Return True for dict with raw and other keys."""
        assert _is_valid_parsed_data({"raw": "text", "other": "value"}) is True

    def test_primitive_types_are_invalid(self):
        """Return False for primitive types."""
        assert _is_valid_parsed_data("string") is False
        assert _is_valid_parsed_data(123) is False
        assert _is_valid_parsed_data(True) is False


class TestInvokeWithFileOutput:
    """Tests for invoke_with_file_output() function."""

    def test_file_written_successfully(self, tmp_path):
        """Return data from file when file is written successfully."""
        output_path = tmp_path / "test-phase" / "cursor-agent_auto.json"

        with patch("utils.llm_client.invoke_with_provider") as mock_invoke, \
             patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_invoke.return_value = {
                "success": True,
                "data": [],
                "details": {"provider": "cursor-agent"},
            }
            mock_read.return_value = {
                "success": True,
                "data": [{"item": 1}],
                "source": "file",
            }

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
            )

            assert result["success"] is True
            assert result["data"] == [{"item": 1}]
            assert result["source"] == "file"

    def test_file_missing_fallback_to_stdout(self, tmp_path):
        """Fall back to stdout when file is missing."""
        with patch("utils.llm_client.invoke_with_provider") as mock_invoke, \
             patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_invoke.return_value = {
                "success": True,
                "data": [{"from": "stdout"}],
                "details": {"provider": "cursor-agent"},
            }
            mock_read.return_value = {
                "success": False,
                "error": "File not found",
                "source": "missing",
            }

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
            )

            assert result["success"] is True
            assert result["data"] == [{"from": "stdout"}]
            assert result["source"] == "stdout_fallback"
            assert result["file_error"] == "File not found"

    def test_both_file_and_stdout_fail(self, tmp_path):
        """Return error when both file and stdout parsing fail."""
        with patch("utils.llm_client.invoke_with_provider") as mock_invoke, \
             patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_invoke.return_value = {
                "success": False,
                "error": "No valid JSON in stdout",
                "data": {"raw": "plain text"},
                "details": {},
            }
            mock_read.return_value = {
                "success": False,
                "error": "No valid JSON in file",
            }

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
            )

            assert result["success"] is False
            assert result["error_code"] == ERROR_PARSE_ERROR
            assert result["file_error"] == "No valid JSON in file"

    def test_hard_failure_propagates(self, tmp_path):
        """Propagate hard failures (timeout, binary not found, subprocess failed)."""
        with patch("utils.llm_client.invoke_with_provider") as mock_invoke:
            mock_invoke.return_value = {
                "success": False,
                "error": "cursor-agent timed out after 600s",
                "error_code": ERROR_TIMEOUT,
                "details": {},
            }

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
            )

            assert result["success"] is False
            assert result["error_code"] == ERROR_TIMEOUT

    @pytest.mark.parametrize("error_code", [ERROR_PROMPT_TOO_LONG, ERROR_PROMPT_UNSAFE])
    def test_prompt_guard_failure_ignores_stale_output_file(self, tmp_path, error_code):
        """Prompt-guard failures must not consume a stale output file from an earlier run.

        PROMPT_TOO_LONG / PROMPT_UNSAFE mean the provider never launched, so a
        pre-existing file at the deterministic per-model output path is stale
        and must not be returned as a successful current result.
        """
        from utils.json_extractor import generate_output_path

        # Simulate a leftover file from an earlier successful run
        stale_path = generate_output_path(tmp_path, "output", "test_phase", "cursor-agent:auto")
        stale_path.write_text('[{"stale": "previous run"}]', encoding="utf-8")

        with patch("utils.llm_client.invoke_with_provider") as mock_invoke:
            mock_invoke.return_value = {
                "success": False,
                "error": "prompt rejected before launch",
                "error_code": error_code,
                "details": {},
            }

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
            )

        assert result["success"] is False
        assert result["error_code"] == error_code
        assert result.get("data") != [{"stale": "previous run"}]

    def test_prompt_template_formatting(self, tmp_path):
        """Format prompt template with context variables."""
        with patch("utils.llm_client.invoke_with_provider") as mock_invoke, \
             patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_invoke.return_value = {
                "success": True,
                "data": [],
                "details": {},
            }
            mock_read.return_value = {
                "success": True,
                "data": [],
                "source": "file",
            }

            invoke_with_file_output(
                prompt_template="Review {plan_name} and write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={"plan_name": "my-plan.md"},
                output_dir=tmp_path,
                phase="test_phase",
            )

            # Verify the prompt was formatted correctly
            call_args = mock_invoke.call_args
            prompt = call_args.kwargs.get("prompt") or call_args[0][0]
            assert "my-plan.md" in prompt

    def test_missing_prompt_variable_error(self, tmp_path):
        """Return error when prompt variable is missing."""
        result = invoke_with_file_output(
            prompt_template="Review {missing_var} and {output_json_path}",
            model_spec="cursor-agent:auto",
            prompt_context={},  # Missing 'missing_var'
            output_dir=tmp_path,
            phase="test_phase",
        )

        assert result["success"] is False
        assert "Missing prompt variable" in result["error"]
        assert result["error_code"] == "PROMPT_FORMAT_ERROR"


class TestInvokeSubagent:
    """Tests for invoke_subagent() function."""

    @pytest.fixture
    def mock_invoke_with_provider(self):
        """Mock invoke_with_provider."""
        with patch("utils.llm_client.invoke_with_provider") as mock:
            yield mock

    def test_success_returns_output(self, mock_invoke_with_provider):
        """Successful invocation returns output in legacy format."""
        mock_invoke_with_provider.return_value = {
            "success": True,
            "data": [{"item": 1}],
            "details": {"stderr": ""},
        }

        result = invoke_subagent(
            prompt="Test prompt",
            model="gpt-4",
        )

        assert result["success"] is True
        assert "output" in result
        # Data should be JSON-serialized
        assert json.loads(result["output"]) == [{"item": 1}]

    def test_failure_returns_error(self, mock_invoke_with_provider):
        """Failed invocation returns error in legacy format."""
        mock_invoke_with_provider.return_value = {
            "success": False,
            "error": "Provider failed",
            "error_code": ERROR_SUBPROCESS_FAILED,
            "details": {"stderr": "error details"},
        }

        result = invoke_subagent(
            prompt="Test prompt",
            model="gpt-4",
            max_retries=0,  # No retries
        )

        assert result["success"] is False
        assert result["error"] == "Provider failed"

    def test_timeout_raises_exception(self, mock_invoke_with_provider):
        """Timeout raises SubagentTimeoutError after retries."""
        mock_invoke_with_provider.return_value = {
            "success": False,
            "error": "cursor-agent timed out",
            "error_code": ERROR_TIMEOUT,
            "details": {},
        }

        with pytest.raises(SubagentTimeoutError):
            invoke_subagent(
                prompt="Test prompt",
                max_retries=0,
            )

    def test_binary_not_found_raises_exception(self, mock_invoke_with_provider):
        """Binary not found raises LLMClientError immediately."""
        mock_invoke_with_provider.return_value = {
            "success": False,
            "error": "cursor-agent CLI not found",
            "error_code": ERROR_BINARY_NOT_FOUND,
            "details": {},
        }

        with pytest.raises(LLMClientError) as exc_info:
            invoke_subagent(prompt="Test")

        assert "cursor-agent CLI not found" in str(exc_info.value)

    def test_retry_logic_with_backoff(self, mock_invoke_with_provider):
        """Retry with exponential backoff on failure."""
        # First two calls fail, third succeeds
        mock_invoke_with_provider.side_effect = [
            {
                "success": False,
                "error": "Temporary failure",
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {},
            },
            {
                "success": False,
                "error": "Temporary failure",
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {},
            },
            {
                "success": True,
                "data": [{"item": 1}],
                "details": {},
            },
        ]

        with patch("utils.llm_client.time.sleep") as mock_sleep:
            result = invoke_subagent(
                prompt="Test",
                max_retries=2,
                retry_backoff=[1, 2],
            )

            assert result["success"] is True
            # Should have slept twice (after first and second failure)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)
            mock_sleep.assert_any_call(2)

    def test_retry_uses_last_backoff_value(self, mock_invoke_with_provider):
        """Use last backoff value when retries exceed backoff list length."""
        mock_invoke_with_provider.side_effect = [
            {
                "success": False,
                "error": "Failure",
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {},
            },
            {
                "success": False,
                "error": "Failure",
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {},
            },
            {
                "success": False,
                "error": "Failure",
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {},
            },
            {
                "success": True,
                "data": [],
                "details": {},
            },
        ]

        with patch("utils.llm_client.time.sleep") as mock_sleep:
            invoke_subagent(
                prompt="Test",
                max_retries=3,
                retry_backoff=[1],  # Only one value
            )

            # All retries should use the same backoff
            assert mock_sleep.call_count == 3
            for call in mock_sleep.call_args_list:
                assert call[0][0] == 1

    def test_log_file_only_on_final_attempt(self, mock_invoke_with_provider, tmp_path):
        """Only save log file on final attempt."""
        log_file = tmp_path / "test.log"

        mock_invoke_with_provider.side_effect = [
            {
                "success": False,
                "error": "Failure",
                "error_code": ERROR_SUBPROCESS_FAILED,
                "details": {},
            },
            {
                "success": True,
                "data": [],
                "details": {},
            },
        ]

        with patch("utils.llm_client.time.sleep"):
            invoke_subagent(
                prompt="Test",
                max_retries=1,
                log_file=log_file,
            )

        # First call should have log_file=None, second should have the path
        calls = mock_invoke_with_provider.call_args_list
        assert calls[0].kwargs["log_file"] is None
        assert calls[1].kwargs["log_file"] == log_file

    def test_model_spec_formatting(self, mock_invoke_with_provider):
        """Format model spec correctly for provider."""
        mock_invoke_with_provider.return_value = {
            "success": True,
            "data": [],
            "details": {},
        }

        invoke_subagent(prompt="Test", model="gpt-4")

        call_kwargs = mock_invoke_with_provider.call_args.kwargs
        assert call_kwargs["model_spec"] == "cursor-agent:gpt-4"

    def test_default_model_spec(self, mock_invoke_with_provider):
        """Use 'cursor-agent:auto' when no model specified."""
        mock_invoke_with_provider.return_value = {
            "success": True,
            "data": [],
            "details": {},
        }

        invoke_subagent(prompt="Test")

        call_kwargs = mock_invoke_with_provider.call_args.kwargs
        assert call_kwargs["model_spec"] == "cursor-agent:auto"


class TestParseSubagentResponse:
    """Tests for parse_subagent_response() function."""

    def test_valid_json_array(self):
        """Parse valid JSON array."""
        result = parse_subagent_response('[{"id": 1}, {"id": 2}]')

        assert result == [{"id": 1}, {"id": 2}]

    def test_cursor_agent_wrapper_format(self):
        """Parse cursor-agent JSON wrapper format."""
        inner_data = [{"title": "Test"}]
        wrapper = {"type": "result", "result": json.dumps(inner_data)}

        result = parse_subagent_response(json.dumps(wrapper))

        assert result == inner_data

    def test_cursor_agent_wrapper_with_object_result(self):
        """Parse cursor-agent wrapper with object in result."""
        inner_data = {"key": "value"}
        wrapper = {"type": "result", "result": json.dumps(inner_data)}

        result = parse_subagent_response(json.dumps(wrapper))

        assert result == inner_data

    def test_code_block_in_wrapper_result(self):
        """Extract JSON from code block in wrapper result."""
        inner_json = [{"item": 1}]
        text_with_block = f"Here is the result:\n```json\n{json.dumps(inner_json)}\n```"
        wrapper = {"type": "result", "result": text_with_block}

        result = parse_subagent_response(json.dumps(wrapper))

        assert result == inner_json

    def test_raw_output_without_json(self):
        """Return raw wrapper for plain text output."""
        result = parse_subagent_response("Just plain text without any JSON")

        assert result == {"raw": "Just plain text without any JSON"}

    def test_direct_json_object(self):
        """Parse direct JSON object (not wrapped)."""
        data = {"key": "value", "nested": {"a": 1}}

        result = parse_subagent_response(json.dumps(data))

        assert result == data

    def test_empty_result_in_wrapper(self):
        """Handle empty result in wrapper."""
        wrapper = {"type": "result", "result": ""}
        wrapper_json = json.dumps(wrapper)

        result = parse_subagent_response(wrapper_json)

        # Empty result triggers _extract_json_from_text on the empty string
        # which returns {"raw": ""} (the empty inner result)
        assert result == {"raw": ""}

    def test_whitespace_handling(self):
        """Handle whitespace around output."""
        data = [1, 2, 3]
        output = f"  \n  {json.dumps(data)}  \n  "

        result = parse_subagent_response(output)

        assert result == data

    def test_json_embedded_in_text(self):
        """Extract JSON embedded in text."""
        json_data = {"found": True}
        text = f"Some prefix text {json.dumps(json_data)} some suffix text"
        wrapper = {"type": "result", "result": text}

        result = parse_subagent_response(json.dumps(wrapper))

        assert result == json_data


class TestExtractJsonFromText:
    """Tests for _extract_json_from_text() function."""

    def test_code_block_with_json_label(self):
        """Extract JSON from labeled code block."""
        text = """Here is the JSON:

```json
[{"id": 1}, {"id": 2}]
```

End of output."""

        result = _extract_json_from_text(text)

        assert result == [{"id": 1}, {"id": 2}]

    def test_code_block_without_json_label(self):
        """Extract JSON from unlabeled code block."""
        text = """Output:
```
{"status": "ok"}
```
"""

        result = _extract_json_from_text(text)

        assert result == {"status": "ok"}

    def test_inline_json_array(self):
        """Extract inline JSON array."""
        text = "The results are: [1, 2, 3] as shown."

        result = _extract_json_from_text(text)

        assert result == [1, 2, 3]

    def test_inline_json_object(self):
        """Extract inline JSON object."""
        text = 'Found: {"key": "value"} in response'

        result = _extract_json_from_text(text)

        assert result == {"key": "value"}

    def test_no_json_returns_raw(self):
        """Return raw wrapper when no JSON found."""
        text = "Just plain text without any JSON structure"

        result = _extract_json_from_text(text)

        assert result == {"raw": text}

    def test_invalid_json_returns_raw(self):
        """Return raw wrapper for malformed JSON."""
        text = "[1, 2, 3"  # Missing closing bracket

        result = _extract_json_from_text(text)

        assert result == {"raw": text}

    def test_prefers_code_block_over_inline(self):
        """Prefer code block JSON over inline JSON."""
        text = """Found [1, 2, 3] inline
```json
[4, 5, 6]
```
"""

        result = _extract_json_from_text(text)

        # Should return code block content
        assert result == [4, 5, 6]

    def test_array_before_object(self):
        """Try array pattern before object pattern."""
        text = '{"obj": 1} and [1, 2] are both valid'

        result = _extract_json_from_text(text)

        # Should find array first (appears after object in text)
        assert result == [1, 2]


class TestInvokeForJson:
    """Tests for invoke_for_json() function."""

    @pytest.fixture
    def mock_invoke_subagent(self):
        """Mock invoke_subagent."""
        with patch("utils.llm_client.invoke_subagent") as mock:
            yield mock

    def test_valid_json_response(self, mock_invoke_subagent):
        """Parse valid JSON response."""
        mock_invoke_subagent.return_value = {
            "success": True,
            "output": json.dumps([{"id": 1}]),
        }

        result = invoke_for_json(
            prompt="Return JSON",
            model="gpt-4",
        )

        assert result["success"] is True
        assert result["data"] == [{"id": 1}]
        assert result["raw_output"] == json.dumps([{"id": 1}])

    def test_parse_failure_propagates(self, mock_invoke_subagent):
        """Propagate subagent failure."""
        mock_invoke_subagent.return_value = {
            "success": False,
            "error": "Subagent failed",
            "output": "",
        }

        result = invoke_for_json(prompt="Test")

        assert result["success"] is False
        assert result["error"] == "Subagent failed"

    def test_parse_cursor_agent_wrapper(self, mock_invoke_subagent):
        """Parse cursor-agent wrapper format."""
        inner_data = {"status": "ok"}
        wrapper = {"type": "result", "result": json.dumps(inner_data)}
        mock_invoke_subagent.return_value = {
            "success": True,
            "output": json.dumps(wrapper),
        }

        result = invoke_for_json(prompt="Test")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_passes_parameters_to_subagent(self, mock_invoke_subagent):
        """Pass parameters to invoke_subagent."""
        mock_invoke_subagent.return_value = {
            "success": True,
            "output": "[]",
        }

        invoke_for_json(
            prompt="Test prompt",
            model="custom-model",
            context_files=["file1.py", "file2.py"],
            timeout=300,
        )

        call_kwargs = mock_invoke_subagent.call_args.kwargs
        assert call_kwargs["prompt"] == "Test prompt"
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["context_files"] == ["file1.py", "file2.py"]
        assert call_kwargs["timeout"] == 300


class TestErrorConstants:
    """Tests for error code constants."""

    def test_error_codes_are_strings(self):
        """Verify error codes are strings."""
        assert isinstance(ERROR_TIMEOUT, str)
        assert isinstance(ERROR_PARSE_ERROR, str)
        assert isinstance(ERROR_BINARY_NOT_FOUND, str)
        assert isinstance(ERROR_SUBPROCESS_FAILED, str)
        assert isinstance(ERROR_FILE_NOT_FOUND, str)
        assert isinstance(ERROR_PROMPT_TOO_LONG, str)

    def test_error_codes_are_unique(self):
        """Verify error codes are unique."""
        codes = [
            ERROR_TIMEOUT,
            ERROR_PARSE_ERROR,
            ERROR_BINARY_NOT_FOUND,
            ERROR_SUBPROCESS_FAILED,
            ERROR_FILE_NOT_FOUND,
            ERROR_PROMPT_TOO_LONG,
        ]
        assert len(codes) == len(set(codes))

    def test_error_codes_values(self):
        """Verify error code values match expected."""
        assert ERROR_TIMEOUT == "TIMEOUT"
        assert ERROR_PARSE_ERROR == "PARSE_ERROR"
        assert ERROR_BINARY_NOT_FOUND == "BINARY_NOT_FOUND"
        assert ERROR_SUBPROCESS_FAILED == "SUBPROCESS_FAILED"
        assert ERROR_FILE_NOT_FOUND == "FILE_NOT_FOUND"
        assert ERROR_PROMPT_TOO_LONG == "PROMPT_TOO_LONG"


class TestExceptions:
    """Tests for exception classes."""

    def test_llm_client_error_is_exception(self):
        """LLMClientError is an Exception subclass."""
        assert issubclass(LLMClientError, Exception)

    def test_subagent_timeout_error_is_llm_client_error(self):
        """SubagentTimeoutError is an LLMClientError subclass."""
        assert issubclass(SubagentTimeoutError, LLMClientError)

    def test_exception_message(self):
        """Exceptions preserve message."""
        error = LLMClientError("Test error message")
        assert str(error) == "Test error message"

        timeout = SubagentTimeoutError("Timeout occurred")
        assert str(timeout) == "Timeout occurred"

    def test_exception_can_be_caught_as_base(self):
        """SubagentTimeoutError can be caught as LLMClientError."""
        try:
            raise SubagentTimeoutError("timeout")
        except LLMClientError as e:
            assert "timeout" in str(e)


class TestResolveExecutable:
    """Tests for _resolve_executable() — npm-shim-aware which-resolution."""

    def test_which_hit_returns_absolute_path(self):
        """A which hit substitutes the absolute path (not a batch shim)."""
        with patch("shutil.which", return_value="/usr/local/bin/cursor-agent"):
            launcher, is_batch_shim = _resolve_executable("cursor-agent")

        assert launcher == ["/usr/local/bin/cursor-agent"]
        assert is_batch_shim is False

    def test_which_miss_keeps_bare_name(self):
        """A which miss keeps the bare name so error text stays meaningful."""
        with patch("shutil.which", return_value=None):
            launcher, is_batch_shim = _resolve_executable("cursor-agent")

        assert launcher == ["cursor-agent"]
        assert is_batch_shim is False

    def test_cmd_shim_prefers_sibling_exe(self, tmp_path):
        """A .cmd shim with a same-stem sibling .exe resolves to the .exe."""
        shim = tmp_path / "codex.cmd"
        shim.write_text("@echo off")
        exe = tmp_path / "codex.exe"
        exe.write_text("")

        with patch("shutil.which", return_value=str(shim)):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(exe)]
        assert is_batch_shim is False

    def test_cmd_shim_without_sibling_exe_is_flagged(self, tmp_path):
        """A .cmd shim with no .exe and no node target is kept but flagged."""
        shim = tmp_path / "codex.cmd"
        shim.write_text("@echo off")

        with patch("shutil.which", return_value=str(shim)):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(shim)]
        assert is_batch_shim is True

    def test_bat_shim_suffix_case_insensitive(self, tmp_path):
        """.BAT (any case) is treated as a batch shim too."""
        shim = tmp_path / "codex.BAT"
        shim.write_text("@echo off")

        with patch("shutil.which", return_value=str(shim)):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(shim)]
        assert is_batch_shim is True

    @staticmethod
    def _write_npm_shim(tmp_path, name="codex", rel=r"node_modules\codex\bin\codex.js"):
        """Write an npm cmd-shim and its dispatched-to cli.js; return both."""
        script = tmp_path.joinpath(*rel.replace("\\", "/").split("/"))
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/usr/bin/env node\n")
        shim = tmp_path / f"{name}.cmd"
        shim.write_text(
            "@ECHO off\r\n"
            "SETLOCAL\r\n"
            'IF EXIST "%dp0%\\node.exe" (\r\n'
            '  SET "_prog=%dp0%\\node.exe"\r\n'
            ") ELSE (\r\n"
            '  SET "_prog=node"\r\n'
            ")\r\n"
            f'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  "%dp0%\\{rel}" %*\r\n'
        )
        return shim, script

    def test_cmd_shim_resolves_node_target_with_sibling_node_exe(self, tmp_path):
        """An npm shim resolves to `node.exe <cli.js>` when node.exe is adjacent."""
        shim, script = self._write_npm_shim(tmp_path)
        node_exe = tmp_path / "node.exe"
        node_exe.write_text("")

        with patch("shutil.which", return_value=str(shim)):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(node_exe), str(script)]
        assert is_batch_shim is False

    def test_cmd_shim_resolves_node_target_via_path_node(self, tmp_path):
        """Without an adjacent node.exe, node is taken from PATH."""
        shim, script = self._write_npm_shim(tmp_path)

        def fake_which(name):
            return {"codex": str(shim), "node": "/usr/bin/node"}.get(name)

        with patch("shutil.which", side_effect=fake_which):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == ["/usr/bin/node", str(script)]
        assert is_batch_shim is False

    def test_cmd_shim_node_target_missing_script_stays_flagged_shim(self, tmp_path):
        """A shim whose js target does not exist falls back to the flagged shim."""
        shim, script = self._write_npm_shim(tmp_path)
        script.unlink()

        def fake_which(name):
            return {"codex": str(shim), "node": "/usr/bin/node"}.get(name)

        with patch("shutil.which", side_effect=fake_which):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(shim)]
        assert is_batch_shim is True

    def test_cmd_shim_node_target_without_node_stays_flagged_shim(self, tmp_path):
        """A parseable shim with no node binary anywhere stays a flagged shim."""
        shim, script = self._write_npm_shim(tmp_path)

        def fake_which(name):
            return {"codex": str(shim)}.get(name)

        with patch("shutil.which", side_effect=fake_which):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(shim)]
        assert is_batch_shim is True

    def test_sibling_exe_wins_over_node_target(self, tmp_path):
        """Step (a1) sibling .exe is preferred over the (a2) node target."""
        shim, script = self._write_npm_shim(tmp_path)
        exe = tmp_path / "codex.exe"
        exe.write_text("")

        with patch("shutil.which", return_value=str(shim)):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(exe)]
        assert is_batch_shim is False


class TestInvokeCommandResolution:
    """invoke_with_provider() resolves cmd[0] before launching."""

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock for subprocess.run."""
        with patch("utils.llm_client.subprocess.run") as mock:
            yield mock

    @staticmethod
    def _success_result():
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        return mock_result

    def test_cmd0_is_which_resolved(self, mock_subprocess):
        """cmd[0] is replaced by the which-resolved absolute path."""
        mock_subprocess.return_value = self._success_result()

        with patch("shutil.which", return_value="/opt/tools/cursor-agent"):
            invoke_with_provider(prompt="Test", model_spec="cursor-agent:auto")

        cmd = mock_subprocess.call_args[0][0]
        assert cmd[0] == "/opt/tools/cursor-agent"
        # The rest of the argv is untouched
        assert cmd[1] == "--print"
        assert cmd[-1] == "Test"

    def test_cmd0_falls_back_to_bare_name(self, mock_subprocess):
        """When which stops resolving (race), the bare name is kept."""
        mock_subprocess.return_value = self._success_result()

        # First which call: is_available() detection; second: resolution.
        with patch("shutil.which", side_effect=["/usr/bin/cursor-agent", None]):
            invoke_with_provider(prompt="Test", model_spec="cursor-agent:auto")

        cmd = mock_subprocess.call_args[0][0]
        assert cmd[0] == "cursor-agent"


class TestLaunchFailureStructuredErrors:
    """Launch-time OSError returns structured errors instead of raising."""

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock for subprocess.run."""
        with patch("utils.llm_client.subprocess.run") as mock:
            yield mock

    @pytest.fixture
    def mock_provider_available(self):
        """Mock provider to appear available."""
        with patch("shutil.which", return_value="/usr/bin/cursor-agent"):
            yield

    def test_launch_race_filenotfound_returns_binary_not_found(
        self, mock_subprocess, mock_provider_available
    ):
        """Detection succeeds but launch raises FileNotFoundError -> BINARY_NOT_FOUND."""
        mock_subprocess.side_effect = FileNotFoundError(2, "No such file or directory")

        result = invoke_with_provider(prompt="Test", model_spec="cursor-agent:gpt-4")

        assert result["success"] is False
        assert result["error_code"] == ERROR_BINARY_NOT_FOUND
        assert "cursor-agent" in result["error"]
        assert result["details"]["provider"] == "cursor-agent"
        assert result["details"]["model"] == "gpt-4"
        assert "duration_seconds" in result["details"]

    def test_launch_oserror_returns_subprocess_failed(
        self, mock_subprocess, mock_provider_available
    ):
        """Any other launch OSError (e.g. EACCES) -> SUBPROCESS_FAILED."""
        mock_subprocess.side_effect = PermissionError(13, "Permission denied")

        result = invoke_with_provider(prompt="Test", model_spec="cursor-agent:gpt-4")

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "cursor-agent" in result["error"]
        assert result["details"]["provider"] == "cursor-agent"
        assert result["details"]["model"] == "gpt-4"

    def test_launch_failure_saves_log(
        self, mock_subprocess, mock_provider_available, tmp_path
    ):
        """Launch failures are logged when a log_file is given."""
        log_file = tmp_path / "launch_failure.log"
        mock_subprocess.side_effect = FileNotFoundError(2, "No such file or directory")

        invoke_with_provider(
            prompt="Test", model_spec="cursor-agent:gpt-4", log_file=log_file
        )

        assert log_file.exists()
        content = log_file.read_text()
        assert "Success: False" in content
        assert "not found" in content


class TestPromptArgvIntegrity:
    """Prompts with shell metacharacters reach the provider's argv intact."""

    @pytest.mark.skipif(
        sys.platform == "win32", reason="child script relies on a POSIX shebang"
    )
    def test_metacharacter_prompt_reaches_child_argv_intact(self, tmp_path):
        """Shell metacharacters survive the real launch path byte-for-byte.

        The provider binary is a real executable script that echoes its argv
        back through the provider JSON protocol; subprocess.run is NOT
        mocked, so this exercises the actual which-resolution + launch path
        and locks it against argv corruption (quoting/reparsing) regressions.
        """
        child = tmp_path / "cursor-agent"
        child.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            'print(json.dumps({"type": "result", "result": json.dumps(sys.argv[1:])}))\n'
        )
        child.chmod(0o755)
        prompt = (
            'quotes " and \' dollar $HOME $(pwd) percent %PATH% 100%\n'
            "amp && single & pipe | semi ; backtick `id` caret ^ bang !\n"
            "redirect > out < in\ttab and a trailing newline\n"
        )

        with patch("shutil.which", return_value=str(child)):
            result = invoke_with_provider(prompt=prompt, model_spec="cursor-agent:auto")

        assert result["success"] is True
        argv = result["data"]
        # The prompt is the final argv element, unchanged.
        assert argv[-1] == prompt
        # The surrounding flags were not disturbed by resolution.
        assert argv[0] == "--print"


class TestSubprocessDecoding:
    """Provider output is decoded as UTF-8 with errors='replace'."""

    @pytest.fixture
    def mock_provider_available(self):
        """Mock provider to appear available."""
        with patch("shutil.which", return_value="/usr/bin/cursor-agent"):
            yield

    @pytest.fixture
    def real_emitter(self, monkeypatch):
        """Route the provider launch to a real python child emitting raw bytes.

        The child replaces the provider command but every subprocess.run
        kwarg (text/encoding/errors/timeout/stdin/...) passes through
        unchanged, exercising the actual decode path end-to-end.
        """
        def install(child_code):
            real_run = subprocess.run

            def fake_run(cmd, **kwargs):
                return real_run([sys.executable, "-c", child_code], **kwargs)

            monkeypatch.setattr("utils.llm_client.subprocess.run", fake_run)

        return install

    def test_run_called_with_utf8_replace(self, mock_provider_available):
        """subprocess.run gets text=True, encoding='utf-8', errors='replace'."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""

        with patch("utils.llm_client.subprocess.run", return_value=mock_result) as mock_run:
            invoke_with_provider(prompt="Test", model_spec="cursor-agent:auto")

        kwargs = mock_run.call_args.kwargs
        assert kwargs["text"] is True
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"

    def test_non_ascii_stdout_decodes_with_replace(
        self, mock_provider_available, real_emitter
    ):
        """Invalid-UTF-8 stdout decodes to U+FFFD instead of raising."""
        # \xe9 is a lone latin-1 byte: invalid UTF-8, valid cp1252 ("é") —
        # exactly the locale-codec hazard the explicit encoding removes.
        payload = b'{"type": "result", "result": "[\\"caf\xe9\\"]"}'
        real_emitter(f"import sys; sys.stdout.buffer.write({payload!r})")

        result = invoke_with_provider(prompt="Test", model_spec="cursor-agent:auto")

        assert result["success"] is True
        assert result["data"] == ["caf�"]

    def test_non_ascii_stderr_decodes_with_replace(
        self, mock_provider_available, real_emitter
    ):
        """Invalid-UTF-8 stderr on failure decodes to U+FFFD instead of raising."""
        payload = b"auth \xe9rror"
        real_emitter(
            f"import sys; sys.stderr.buffer.write({payload!r}); sys.exit(1)"
        )

        result = invoke_with_provider(prompt="Test", model_spec="cursor-agent:auto")

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert result["details"]["stderr"] == "auth �rror"

    def test_timeout_partial_bytes_decode_with_replace(
        self, mock_provider_available, tmp_path
    ):
        """Non-ASCII partial output on timeout decodes as UTF-8/replace."""
        log_file = tmp_path / "timeout.log"
        exc = subprocess.TimeoutExpired(cmd=["cursor-agent"], timeout=1)
        exc.stdout = b"partial caf\xe9"
        exc.stderr = b"stderr \xff"

        with patch("utils.llm_client.subprocess.run", side_effect=exc):
            result = invoke_with_provider(
                prompt="Test", model_spec="cursor-agent:auto", log_file=log_file
            )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        # The log is written as UTF-8; read it the same way so the test
        # measures production behavior, not the host locale (cp1252 on
        # Windows would otherwise mojibake the U+FFFD bytes).
        content = log_file.read_text(encoding="utf-8")
        assert "partial caf�" in content
        assert "stderr �" in content


class TestPromptLengthGuard:
    """Windows command-line length check for prompt-on-argv providers.

    The guard measures the FULL rendered command line (list2cmdline over the
    resolved argv) in UTF-16 code units — not just len(prompt) — because
    cmd.exe/CreateProcess caps apply to the whole command line including the
    executable path, flags, quoting expansion, and astral-character width.
    """

    BATCH_LIMIT = CMDLINE_CAP_UTF16_BATCH - CMDLINE_UTF16_HEADROOM
    NATIVE_LIMIT = CMDLINE_CAP_UTF16_NATIVE - CMDLINE_UTF16_HEADROOM

    @staticmethod
    def _cmd_rendering_exactly(units, exe=r"C:\npm\codex.cmd", flags=("exec",)):
        """Build [exe, *flags, prompt] whose list2cmdline output is exactly
        ``units`` UTF-16 code units, with an all-'x' prompt (no quoting
        expansion)."""
        overhead = _utf16_code_units(
            subprocess.list2cmdline([exe, *flags])
        ) + 1  # separating space before the prompt argument
        prompt = "x" * (units - overhead)
        return [exe, *flags, prompt], prompt

    def test_within_batch_budget_returns_none(self):
        """A command line rendering exactly at the batch-shim limit passes."""
        cmd, prompt = self._cmd_rendering_exactly(self.BATCH_LIMIT)

        assert _prompt_length_error(cmd, prompt, "codex", "gpt-5.5", True) is None

    def test_batch_shim_over_budget_returns_structured_error(self):
        """One unit over the batch-shim limit -> PROMPT_TOO_LONG."""
        cmd, prompt = self._cmd_rendering_exactly(self.BATCH_LIMIT + 1)

        error = _prompt_length_error(cmd, prompt, "codex", "gpt-5.5", True)

        assert error["success"] is False
        assert error["error_code"] == ERROR_PROMPT_TOO_LONG
        assert error["details"]["provider"] == "codex"
        assert error["details"]["model"] == "gpt-5.5"
        assert error["details"]["prompt_chars"] == len(prompt)
        assert error["details"]["cmdline_utf16_units"] == self.BATCH_LIMIT + 1
        assert error["details"]["cmdline_utf16_limit"] == self.BATCH_LIMIT
        # Actionable: names the cause and suggests a fix
        assert "cmd.exe" in error["error"]
        assert "Shorten the prompt" in error["error"]

    def test_native_budget_is_larger(self):
        """Native executables get the larger CreateProcess budget."""
        cmd, prompt = self._cmd_rendering_exactly(
            self.NATIVE_LIMIT, exe=r"C:\bin\codex.exe"
        )
        assert _prompt_length_error(cmd, prompt, "codex", "gpt-5.5", False) is None

        cmd, prompt = self._cmd_rendering_exactly(
            self.NATIVE_LIMIT + 1, exe=r"C:\bin\codex.exe"
        )
        error = _prompt_length_error(cmd, prompt, "codex", "gpt-5.5", False)
        assert error["error_code"] == ERROR_PROMPT_TOO_LONG
        assert error["details"]["cmdline_utf16_limit"] == self.NATIVE_LIMIT

    def test_argv_overhead_counts_against_the_limit(self):
        """A prompt under the limit still trips the guard once the executable
        path and flags push the rendered command line over — the regression
        this guard exists for (len(prompt)-only checks pass this prompt)."""
        prompt = "x" * (self.BATCH_LIMIT - 100)
        long_exe = "C:\\Users\\dev\\AppData\\Roaming\\npm\\" + "a" * 150 + ".cmd"
        cmd = [long_exe, "exec", "--json", prompt]

        error = _prompt_length_error(cmd, prompt, "codex", "gpt-5.5", True)

        assert error is not None
        assert error["error_code"] == ERROR_PROMPT_TOO_LONG

    def test_astral_characters_count_as_two_units(self):
        """Astral chars occupy two UTF-16 code units, so a prompt whose
        code-point count fits can still exceed the cap."""
        # Rendered line: len(prompt) code points but ~2x UTF-16 units.
        prompt = "\U0001F600" * (self.BATCH_LIMIT // 2)
        cmd = [r"C:\npm\codex.cmd", "exec", prompt]

        error = _prompt_length_error(cmd, prompt, "codex", "gpt-5.5", True)

        assert error is not None
        assert error["details"]["prompt_utf16_units"] == 2 * len(prompt)

    def test_quoting_expansion_counts_against_the_limit(self):
        """list2cmdline quoting (spaces force quotes, backslash-doubling
        before quotes) inflates the rendered line beyond len(prompt)."""
        chunk = 'say \\"hi\\" '  # backslashes double + quotes escape + spaces quote
        prompt = chunk * (self.NATIVE_LIMIT // len(chunk))
        cmd = [r"C:\bin\codex.exe", "exec", prompt]
        assert len(prompt) <= self.NATIVE_LIMIT  # raw length fits...

        error = _prompt_length_error(cmd, prompt, "codex", "gpt-5.5", False)

        assert error is not None  # ...but the rendered command line does not
        assert error["details"]["cmdline_utf16_units"] > self.NATIVE_LIMIT

    def test_enforced_on_windows_before_launch(self, tmp_path):
        """On Windows, an over-budget prompt fails fast without launching."""
        shim = tmp_path / "cursor-agent.cmd"
        shim.write_text("@echo off")
        prompt = "x" * (CMDLINE_CAP_UTF16_BATCH + 100)

        with patch("shutil.which", return_value=str(shim)), \
             patch("utils.llm_client._IS_WINDOWS", True), \
             patch("utils.llm_client.subprocess.run") as mock_run:
            result = invoke_with_provider(prompt=prompt, model_spec="cursor-agent:auto")

        assert result["success"] is False
        assert result["error_code"] == ERROR_PROMPT_TOO_LONG
        mock_run.assert_not_called()

    def test_native_executable_allows_longer_prompt_on_windows(self):
        """The same prompt is fine on Windows with a native executable."""
        prompt = "x" * (CMDLINE_CAP_UTF16_BATCH + 100)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""

        with patch("shutil.which", return_value="/usr/bin/cursor-agent"), \
             patch("utils.llm_client._IS_WINDOWS", True), \
             patch("utils.llm_client.subprocess.run", return_value=mock_result) as mock_run:
            result = invoke_with_provider(prompt=prompt, model_spec="cursor-agent:auto")

        assert result["success"] is True
        mock_run.assert_called_once()

    def test_not_enforced_on_posix(self, tmp_path):
        """On POSIX, prompt length is never enforced — even via a .cmd path."""
        shim = tmp_path / "cursor-agent.cmd"
        shim.write_text("@echo off")
        prompt = "x" * (CMDLINE_CAP_UTF16_NATIVE + 100)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""

        with patch("shutil.which", return_value=str(shim)), \
             patch("utils.llm_client._IS_WINDOWS", False), \
             patch("utils.llm_client.subprocess.run", return_value=mock_result) as mock_run:
            result = invoke_with_provider(prompt=prompt, model_spec="cursor-agent:auto")

        assert result["success"] is True
        mock_run.assert_called_once()


class TestBatchShimMetacharGuard:
    """Windows cmd.exe metacharacter guard for batch-shim launches."""

    @staticmethod
    def _success_result():
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        return mock_result

    def test_clean_prompt_returns_none(self):
        """A metacharacter-free prompt passes the guard."""
        prompt = "Review the plan and list issues. Use JSON output, please."

        assert _batch_shim_metachar_error(
            prompt, "codex", "gpt-5.5", r"C:\npm\codex.cmd"
        ) is None

    @pytest.mark.parametrize("char", list('"%!^&|<>()') + ["\r", "\n"])
    def test_each_cmd_metacharacter_is_rejected(self, char):
        """Every cmd.exe metacharacter in the guarded set triggers rejection."""
        error = _batch_shim_metachar_error(
            f"prompt with {char} inside", "codex", "gpt-5.5", r"C:\npm\codex.cmd"
        )

        assert error is not None
        assert error["success"] is False
        assert error["error_code"] == ERROR_PROMPT_UNSAFE
        assert char in error["details"]["unsafe_characters"]

    def test_error_is_structured_and_actionable(self):
        """The rejection names the shim, the class of attack, and a fix."""
        error = _batch_shim_metachar_error(
            'echo "hi" & del *', "codex", "gpt-5.5", r"C:\npm\codex.cmd"
        )

        assert error["details"]["provider"] == "codex"
        assert error["details"]["model"] == "gpt-5.5"
        assert error["details"]["executable"] == r"C:\npm\codex.cmd"
        assert "cmd.exe" in error["error"]
        assert "CVE-2024-24576" in error["error"]
        assert "native" in error["error"]

    def test_enforced_on_windows_before_launch(self, tmp_path):
        """On Windows, a shim launch with a metachar prompt never reaches run()."""
        shim = tmp_path / "cursor-agent.cmd"
        shim.write_text("@echo off")

        with patch("shutil.which", return_value=str(shim)), \
             patch("utils.llm_client._IS_WINDOWS", True), \
             patch("utils.llm_client.subprocess.run") as mock_run:
            result = invoke_with_provider(
                prompt='Say "hello" && exit', model_spec="cursor-agent:auto"
            )

        assert result["success"] is False
        assert result["error_code"] == ERROR_PROMPT_UNSAFE
        mock_run.assert_not_called()

    def test_not_enforced_for_native_executable_on_windows(self):
        """Native (non-shim) launches pass metachar prompts straight through."""
        with patch("shutil.which", return_value="/usr/bin/cursor-agent"), \
             patch("utils.llm_client._IS_WINDOWS", True), \
             patch("utils.llm_client.subprocess.run",
                   return_value=self._success_result()) as mock_run:
            result = invoke_with_provider(
                prompt='Say "hello" && exit', model_spec="cursor-agent:auto"
            )

        assert result["success"] is True
        mock_run.assert_called_once()

    def test_not_enforced_on_posix(self, tmp_path):
        """On POSIX, argv is passed verbatim — no guard even via a .cmd path."""
        shim = tmp_path / "cursor-agent.cmd"
        shim.write_text("@echo off")

        with patch("shutil.which", return_value=str(shim)), \
             patch("utils.llm_client._IS_WINDOWS", False), \
             patch("utils.llm_client.subprocess.run",
                   return_value=self._success_result()) as mock_run:
            result = invoke_with_provider(
                prompt='Say "hello" && exit', model_spec="cursor-agent:auto"
            )

        assert result["success"] is True
        mock_run.assert_called_once()
