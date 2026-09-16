"""Comprehensive unit tests for llm_client module.

This module tests the LLM client wrapper functionality including:
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

from utils import llm_client
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
    MAX_LOG_STREAM_BYTES,
    MAX_LOG_TAIL_LINES,
    _attach_stdout_excerpt,
    _save_log,
    _resolve_executable,
    _utf16_code_units,
    _prompt_length_error,
    _batch_shim_metachar_error,
    invoke_with_provider,
    _is_valid_parsed_data,
    invoke_with_file_output,
    FILE_OUTPUT_LOCATION,
    STDOUT_OUTPUT_LOCATION,
    render_output_location,
    describe_output_file,
    invoke_subagent,
    parse_subagent_response,
    _extract_json_from_text,
    invoke_for_json,
)
from utils.providers.cursor_agent import CursorAgentProvider

# The redaction canary is *synthesized*, never written out — the same table
# `tests/test_redaction.py` draws its `anthropic-api-key` vector from, shared so
# the two suites cannot drift in their fixtures (§4.5.5). A literal
# `sk-ant-api03-...` here would make this file the one committed secret nobody
# scans, and `tests/test_no_committed_secrets.py` would say so.
from .test_no_committed_secrets import _synthesize  # noqa: E402


# `check_cursor_agent_available()` and its three tests are deleted (§9.5): the
# helper had no caller anywhere in the skill, and availability is the adapter's
# own `CursorAgentProvider.is_available()` — which is what
# `tests/test_providers.py` covers. The rest of this module's legacy surface
# (`invoke_subagent`, `parse_subagent_response`, `invoke_for_json`) stays: it is
# live behind the shipped `--internal-validation` flag.


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

        content = log_file.read_text(encoding="utf-8")
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
        content = log_file.read_text(encoding="utf-8")
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
        content = log_file.read_text(encoding="utf-8")
        # Should be truncated with "..."
        assert "..." in content
        # Should not contain the full 10000 chars
        assert len(content) < 10000 + 1000  # Some overhead for log formatting

    def test_save_log_io_error(self, tmp_path, capsys):
        """Swallow an IO error, warn, and return False.

        The write is mocked to raise OSError rather than relying on a
        magic unwritable path: `_save_log` creates parent directories, so
        no portable filesystem path reliably fails on every OS.
        """
        log_file = tmp_path / "log.txt"

        with patch(
            "utils.llm_client.open", side_effect=OSError(13, "Permission denied")
        ) as mock_open_fn:
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
        assert mock_open_fn.called
        assert not log_file.exists()
        assert "Failed to save log" in capsys.readouterr().out

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
        content = log_file.read_text(encoding="utf-8")
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

    def test_save_log_header_is_provider_neutral(self, tmp_path):
        """The header names the provider from the model spec, not a fixed CLI (§Fix1)."""
        log_file = tmp_path / "claude_log.txt"

        result = _save_log(
            log_file=log_file,
            model="claude-code:opus",
            prompt="p",
            stdout="out",
            stderr="",
            returncode=0,
            success=True,
        )

        assert result is True
        content = log_file.read_text(encoding="utf-8")
        assert "CLAUDE-CODE LOG" in content
        assert "CURSOR-AGENT LOG" not in content

    def test_save_log_new_optional_fields(self, tmp_path):
        """``result_source``/``stdout_parse_error`` render when given, and only
        then — except a failed stdout parse beside a FILE_WRITTEN_SOURCES
        ``result_source`` (F9.2): the provider was told not to print JSON on
        stdout at all, so that "failure" is expected, not a fault, and is
        worded to say so instead of being dropped or read as a real error."""
        log_file = tmp_path / "log.txt"

        result = _save_log(
            log_file=log_file,
            model="claude-code:opus",
            prompt="p",
            stdout="out",
            stderr="",
            returncode=0,
            success=True,
            result_source="file",
            stdout_parse_error="No valid JSON found in text",
        )

        assert result is True
        content = log_file.read_text(encoding="utf-8")
        assert "Result source: file" in content
        assert "Stdout: not used (file output)" in content
        assert "Stdout parse: failed" not in content

        # Omitted (the default) renders neither line.
        log_file_2 = tmp_path / "log2.txt"
        _save_log(
            log_file=log_file_2, model="claude-code:opus", prompt="p",
            stdout="out", stderr="", returncode=0, success=True,
        )
        content_2 = log_file_2.read_text(encoding="utf-8")
        assert "Result source:" not in content_2
        assert "Stdout parse:" not in content_2
        assert "Stdout: not used" not in content_2

    def test_save_log_real_stdout_parse_failure_still_reports_as_a_failure(
        self, tmp_path
    ):
        """A genuine double failure (no file, no stdout — ``result_source``
        is ``None``, as `_finalize_file_output_log` passes on that path) must
        keep the honest ``Stdout parse: failed`` wording; only a
        FILE_WRITTEN_SOURCES source excuses it."""
        log_file = tmp_path / "log.txt"

        _save_log(
            log_file=log_file,
            model="claude-code:opus",
            prompt="p",
            stdout="out",
            stderr="",
            returncode=0,
            success=False,
            result_source=None,
            stdout_parse_error="No valid JSON found in text",
        )

        content = log_file.read_text(encoding="utf-8")
        assert "Stdout parse: failed (No valid JSON found in text)" in content
        assert "Stdout: not used" not in content


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

    def test_subprocess_failure_reason_recovered_from_stdout(
        self, mock_subprocess, mock_provider_available
    ):
        """A provider that reports its cause on stdout gets it into the error.

        opencode exits 1 with the real reason on stdout and stderr empty, so
        without describe_failure the caller sees only a bare exit code.
        """
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = json.dumps({
            "type": "error",
            "timestamp": 1785403158514,
            "sessionID": "ses_TEST",
            "error": {
                "name": "ProviderAuthError",
                "data": {"providerID": "google", "message": "API key is missing."},
            },
        })
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test",
            model_spec="opencode:google/gemini-2.5-pro",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "exited with code 1" in result["error"]
        assert "ProviderAuthError: API key is missing." in result["error"]

    def test_subprocess_failure_reason_survives_broken_provider_hook(
        self, mock_subprocess, mock_provider_available
    ):
        """A raising describe_failure must not mask the subprocess failure."""
        mock_result = MagicMock()
        mock_result.returncode = 3
        mock_result.stdout = "whatever"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        with patch.object(
            CursorAgentProvider, "describe_failure", side_effect=RuntimeError("boom")
        ):
            result = invoke_with_provider(
                prompt="Test",
                model_spec="cursor-agent:gpt-4",
            )

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "exited with code 3" in result["error"]

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
        content = log_file.read_text(encoding="utf-8")
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
        content = log_file.read_text(encoding="utf-8")
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
        content = log_file.read_text(encoding="utf-8")
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


class TestHardFailureStdoutExcerpt:
    """``details["stdout_excerpt"]`` on the two paths that used to drop stdout.

    ``invoke_with_provider`` loses a run three ways, and until this change only
    one of them carried what the provider had actually printed:

    * a **zero exit whose output would not parse** attached
      ``details["stdout_excerpt"]`` — the original site, added because "JSON
      parsing failed from both file and stdout" was what a cursor ``--mode ask``
      write refusal looked like from the outside;
    * a **non-zero exit** attached ``exit_code`` and ``stderr`` and nothing else;
    * a **timeout** attached neither stream.

    The two silent ones are the interesting ones, because a non-zero exit is not
    only "the provider never spoke". It is also the provider that wrote a
    truncated or fenced array and *then* died — a kill at the deadline, a crash on
    teardown — and that array is on **stdout**, never on stderr. Salvage gating in
    ``code_review_orchestrator.py`` runs under exactly that branch
    (``if not result["success"]``), so with stderr as its only channel it would
    announce "Provider failed before producing any JSON — nothing to salvage"
    about a run that had produced JSON, and drop a recoverable review without
    saying so. The JSON-erroring providers press the same point from the other
    side: ``opencode run --format json`` puts its failure shape on stdout with
    stderr empty, and a relayed ``{"error": {"code": 401, ...}}`` body is
    indistinguishable from any HTTP-backed CLI — those reached the gate as the
    empty string, so even the verdicts it reached correctly were reached for the
    wrong reason.

    These tests pin the **plumbing half**: that the text arrives on the envelope,
    that it arrives redacted and bounded (this excerpt can reach a salvage
    request, a log *and* a report), and that absence still means absence so no
    consumer needs an empty-string special case. What the orchestrator then
    *decides* with it is site A's own business and is tested there.
    """

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

    #: A review array cut off mid-object: what a provider killed while streaming
    #: leaves behind, and the single most salvageable failure this skill has.
    #: Deliberately not valid JSON — the point is that `has_salvageable_json` gets
    #: to look at it, not that it parses here.
    TRUNCATED_ARRAY = (
        '[{"title": "Retry ladder re-runs a schema violation", '
        '"importance": "high", "file": "utils/retry_ladder.py", '
        '"type": "logic", "desc": "The ladder treats a schema_violation as'
    )

    #: kilocode's measured auth failure, ANSI escapes and all, as captured in
    #: `tests/test_stderr_diagnosis.py` — a provider that died *before* saying
    #: anything, on the stream it was supposed to use. Nothing to salvage, and the
    #: envelope must say so by omission rather than by an empty string.
    KILOCODE_AUTH_STDERR = "\x1b[91m\x1b[1mError: \x1b[0mUser not found.\n"

    #: The opencode error shape: a JSON failure body on **stdout**, stderr empty.
    OPENCODE_ERROR_BODY = json.dumps(
        {
            "type": "error",
            "error": {
                "name": "ProviderAuthError",
                "data": {"providerID": "google", "message": "API key is missing."},
            },
        }
    )

    @staticmethod
    def _failed_run(stdout, stderr, returncode=1):
        """A `CompletedProcess` stand-in for a provider that exited non-zero."""
        mock_result = MagicMock()
        mock_result.returncode = returncode
        mock_result.stdout = stdout
        mock_result.stderr = stderr
        return mock_result

    @staticmethod
    def _timeout(stdout, stderr="", timeout=600):
        """A `TimeoutExpired` carrying whatever the kill managed to recover."""
        exc = subprocess.TimeoutExpired(cmd=["cursor-agent"], timeout=timeout)
        exc.stdout = stdout
        exc.stderr = stderr
        return exc

    # -- the non-zero exit ---------------------------------------------------

    def test_a_non_zero_exit_carries_its_truncated_stdout_array_out_as_stdout_excerpt(
        self, mock_subprocess, mock_provider_available
    ):
        """The answer was on stdout and the exit code was 1; both must survive.

        This is the case the stderr-only salvage gate was structurally unable to
        see: stderr is empty, so the pre-change envelope handed the gate the empty
        string and the array went in the bin. The assertion is on the *text*, not
        merely on the key, because a consumer is going to run
        `has_salvageable_json` over it.
        """
        mock_subprocess.return_value = self._failed_run(self.TRUNCATED_ARRAY, "")

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert result["details"]["exit_code"] == 1
        assert result["details"]["stdout_excerpt"] == self.TRUNCATED_ARRAY

    def test_a_non_zero_exit_with_the_kilocode_auth_text_on_stderr_attaches_no_excerpt(
        self, mock_subprocess, mock_provider_available
    ):
        """The negative of the case above: nothing was printed, so no key appears.

        kilocode's `Error: User not found.` is an auth verdict written to stderr
        by a CLI that never emitted a byte of answer. `details["stderr"]` must
        still carry it verbatim — the stderr diagnosis path reads it — and
        `stdout_excerpt` must be **absent**, not present-and-empty, so that
        `details.get("stdout_excerpt")` reads as "the provider said nothing we
        can use" without every call site adding a second emptiness check.
        """
        mock_subprocess.return_value = self._failed_run("", self.KILOCODE_AUTH_STDERR)

        result = invoke_with_provider(
            prompt="Test",
            model_spec="kilocode:moonshotai/kimi-k3",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "stdout_excerpt" not in result["details"]
        assert result["details"]["stderr"] == self.KILOCODE_AUTH_STDERR

    def test_a_json_erroring_provider_reaches_the_caller_with_its_error_body_attached(
        self, mock_subprocess, mock_provider_available
    ):
        """opencode's failure shape is on stdout with stderr empty.

        `describe_failure` already reads that body for the one-line reason on
        `result["error"]`, but a one-line reason is a verdict, not evidence. The
        body itself now rides out too, so a consumer that wants to reach its own
        conclusion — is this JSON at all? is it salvageable? — has the same text
        the adapter judged.
        """
        mock_subprocess.return_value = self._failed_run(self.OPENCODE_ERROR_BODY, "")

        result = invoke_with_provider(
            prompt="Test",
            model_spec="opencode:google/gemini-2.5-pro",
        )

        assert result["success"] is False
        assert "ProviderAuthError: API key is missing." in result["error"]
        assert result["details"]["stdout_excerpt"] == self.OPENCODE_ERROR_BODY

    # -- the timeout ---------------------------------------------------------

    def test_a_timeout_kill_carries_the_partial_stdout_it_recovered(
        self, mock_subprocess, mock_provider_available
    ):
        """The timeout path *does* have the bytes, and now returns them.

        Worth stating because the opposite would have been a reason to attach
        nothing: `subprocess.run` re-raises a `TimeoutExpired` already populated
        with everything read off the pipe before the deadline (POSIX populates it
        as it raises; Windows kills and re-`communicate()`s), so a provider killed
        mid-stream leaves a real, partial answer here. `_save_log` has always
        written it to the log file; only the returned envelope dropped it, and the
        envelope is the half a caller can act on.
        """
        mock_subprocess.side_effect = self._timeout(self.TRUNCATED_ARRAY)

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
            timeout=600,
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert result["details"]["timeout"] == 600
        assert result["details"]["stdout_excerpt"] == self.TRUNCATED_ARRAY

    def test_a_timeout_decodes_partial_stdout_from_bytes_before_excerpting_it(
        self, mock_subprocess, mock_provider_available
    ):
        """`TimeoutExpired.stdout` arrives as **bytes** even under `text=True`.

        Measured, not assumed: `Popen._communicate` joins the raw chunks it has
        accumulated and hands them to the exception without running them through
        the stream decoder, so the `text=True` guarantee that holds for a
        `CompletedProcess` does not hold here. The excerpt therefore has to sit
        downstream of the module's own utf-8/replace decode, or it would be
        excerpting a `bytes` repr.
        """
        mock_subprocess.side_effect = self._timeout(self.TRUNCATED_ARRAY.encode("utf-8"))

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        assert result["details"]["stdout_excerpt"] == self.TRUNCATED_ARRAY

    @pytest.mark.parametrize("partial", [None, "", "   \n\n  "])
    def test_a_timeout_that_recovered_nothing_usable_attaches_no_excerpt_key(
        self, mock_subprocess, mock_provider_available, partial
    ):
        """A kill before the child's first write, and its whitespace-only cousin.

        All three are "the provider said nothing", and all three must reach a
        consumer identically. The `.strip()` is why the third belongs here: a
        provider that emitted only a newline before dying is not a salvage
        candidate, and an excerpt of `"\\n"` would make `.get("stdout_excerpt")`
        truthy for one.
        """
        mock_subprocess.side_effect = self._timeout(partial)

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        assert result["error_code"] == ERROR_TIMEOUT
        assert "stdout_excerpt" not in result["details"]

    # -- §4.5 then §4.4, on the paths that just started carrying text ---------

    def test_a_credential_in_stdout_does_not_survive_the_non_zero_exit_excerpt(
        self, mock_subprocess, mock_provider_available
    ):
        """Redaction is not inherited from the log writer; this path needs its own.

        The excerpt is a *third* sink for provider output — it reaches a salvage
        request, `failed/*.attemptN.json` and whatever report quotes it — and it
        is the newest, which makes it exactly the sink a redaction change would
        forget. The canary is the shared synthesized `anthropic-api-key` vector,
        so a pattern-set regression names the same shape here that it names in
        `tests/test_redaction.py`.
        """
        secret = _synthesize("anthropic-api-key", "llm-client-non-zero-exit")
        mock_subprocess.return_value = self._failed_run(
            f'{{"error": "invalid x-api-key {secret}"}}', ""
        )

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        excerpt = result["details"]["stdout_excerpt"]
        assert secret not in excerpt
        assert "<redacted:" in excerpt
        # The surrounding sentence survives, or the excerpt stops being evidence.
        assert "invalid x-api-key" in excerpt

    def test_a_credential_in_partial_stdout_does_not_survive_the_timeout_excerpt(
        self, mock_subprocess, mock_provider_available
    ):
        """The same gate on the other new path, asserted rather than inferred.

        Two paths now attach this key and both went through one helper, but "they
        share a helper" is a claim about today's source; the leak this prevents is
        someone tomorrow inlining one of them back.
        """
        secret = _synthesize("anthropic-api-key", "llm-client-timeout")
        mock_subprocess.side_effect = self._timeout(f"streaming... key={secret}")

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        excerpt = result["details"]["stdout_excerpt"]
        assert secret not in excerpt
        assert "<redacted:" in excerpt

    def test_an_over_long_stdout_is_bounded_to_the_tail_on_the_non_zero_exit_path(
        self, mock_subprocess, mock_provider_available
    ):
        """§4.4's caps apply here too — and the run killed at the deadline is the
        run most likely to have emitted megabytes.

        A *successful* `review-plan` once wrote a 6,254,326 B log; nothing stops a
        failing one from producing the same volume, and this excerpt travels into
        a salvage prompt where an unbounded stream is a cost as well as a hazard.
        """
        stdout = "\n".join(f"chunk {i}" for i in range(MAX_LOG_TAIL_LINES * 3))
        mock_subprocess.return_value = self._failed_run(stdout, "")

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        excerpt = result["details"]["stdout_excerpt"]
        lines = excerpt.splitlines()
        assert len(lines) == MAX_LOG_TAIL_LINES + 1, "the marker plus the kept tail"
        assert "earlier line(s) truncated" in lines[0]
        assert "chunk 0" not in excerpt
        assert f"chunk {MAX_LOG_TAIL_LINES * 3 - 1}" in excerpt

    def test_a_single_enormous_stdout_line_is_cut_to_the_per_stream_byte_budget(
        self, mock_subprocess, mock_provider_available
    ):
        """The last-20-lines rule is no cap at all against one huge line.

        A provider streaming JSON without newlines — which is the normal shape for
        a `--format json` CLI — produces exactly one line, so the byte budgets are
        the only thing standing between a 200 KB answer and a 200 KB excerpt.
        """
        mock_subprocess.return_value = self._failed_run("x" * (MAX_LOG_STREAM_BYTES * 4), "")

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        excerpt = result["details"]["stdout_excerpt"]
        assert len(excerpt.encode("utf-8")) <= MAX_LOG_STREAM_BYTES
        assert "stream truncated" in excerpt

    # -- the refactor's own guard rails --------------------------------------

    def test_the_zero_exit_parse_failure_still_attaches_the_excerpt_it_always_did(
        self, mock_subprocess, mock_provider_available
    ):
        """A regression pin on the site the helper was factored *out of*.

        This one predates the change and is the reason the helper had to keep
        `.strip()` and the omit-when-empty rule rather than inventing its own: the
        cursor `--mode ask` refusal that cost three 450 s runs is read out of this
        key by `tests/test_fake_lane_output_channel.py`, and by anyone debugging a
        parse failure.
        """
        refusal = "I can't write the requested file while Ask mode is active."
        mock_subprocess.return_value = self._failed_run(refusal, "", returncode=0)

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_PARSE_ERROR
        assert result["details"]["stdout_excerpt"] == refusal

    def test_a_zero_exit_that_parsed_cleanly_attaches_no_excerpt(
        self, mock_subprocess, mock_provider_available
    ):
        """Success carries data, not a transcript.

        The key exists to explain a loss. Attaching it to a run that succeeded
        would double every successful envelope's payload for no reader, and would
        make `.get("stdout_excerpt")` useless as the "did this go wrong" signal
        the failure paths now use it as.
        """
        wrapper = {"type": "result", "result": json.dumps([{"title": "ok"}])}
        mock_subprocess.return_value = self._failed_run(json.dumps(wrapper), "", returncode=0)

        result = invoke_with_provider(
            prompt="Test",
            model_spec="cursor-agent:gpt-4",
        )

        assert result["success"] is True
        assert "stdout_excerpt" not in result["details"]

    def test_all_three_losing_paths_route_through_the_one_helper(
        self, mock_subprocess, mock_provider_available
    ):
        """The factoring itself, pinned — because three copies drift twice.

        The redact-before-bound order is normative (`_bound_stream` says so: a cut
        applied first leaves a remnant no pattern matches), and a cap tightened or
        a pattern added at two of three copies leaks from the third invisibly,
        until a human reads an artifact. Asserting the call rather than only its
        output is what makes an inlined fourth copy fail here instead of shipping.
        """
        seen = []
        with patch.object(
            llm_client,
            "_attach_stdout_excerpt",
            side_effect=lambda details, stdout: seen.append(stdout),
        ):
            mock_subprocess.return_value = self._failed_run("nonzero stdout", "")
            invoke_with_provider(prompt="T", model_spec="cursor-agent:gpt-4")

            mock_subprocess.return_value = self._failed_run("unparseable", "", returncode=0)
            invoke_with_provider(prompt="T", model_spec="cursor-agent:gpt-4")

            mock_subprocess.return_value = None
            mock_subprocess.side_effect = self._timeout("timeout stdout")
            invoke_with_provider(prompt="T", model_spec="cursor-agent:gpt-4")

        assert seen == ["nonzero stdout", "unparseable", "timeout stdout"]

    def test_the_helper_omits_the_key_rather_than_writing_an_empty_string(self):
        """The contract stated directly, without a subprocess in the way.

        `details.get("stdout_excerpt")` has to mean "there is text worth looking
        at". A key holding `""` would satisfy `in details` for every caller that
        checks membership, and the first one written that way turns "the provider
        printed nothing" into "the provider printed something".
        """
        for nothing in (None, "", "\n\n", "   "):
            details = {"exit_code": 1}
            _attach_stdout_excerpt(details, nothing)
            assert details == {"exit_code": 1}, f"attached a key for {nothing!r}"

        details = {"exit_code": 1}
        _attach_stdout_excerpt(details, "  something  \n")
        assert details["stdout_excerpt"] == "something"


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


class TestInvokeWithFileOutputLogFinalization:
    """§Fix1: the header is provider-neutral, and the one log entry
    ``invoke_with_file_output`` writes describes the FINAL outcome — after the
    file channel and the stdout fallback both had their turn — never the
    stdout extractor's own verdict frozen before that resolution happened.

    Unlike ``TestInvokeWithFileOutput`` above, these drive the real
    ``invoke_with_provider`` (only ``subprocess.run`` is mocked) so the
    deferred-log handoff between the two functions actually runs.
    """

    @pytest.fixture
    def mock_subprocess(self):
        with patch("utils.llm_client.subprocess.run") as mock:
            yield mock

    @pytest.fixture
    def mock_provider_available(self):
        with patch("shutil.which", return_value="/usr/bin/cursor-agent"):
            yield

    def test_log_reflects_file_channel_salvage_after_stdout_parse_failure(
        self, mock_subprocess, mock_provider_available, tmp_path
    ):
        """A run whose stdout never parsed, but whose file did, logs Success: True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is plain text without any JSON"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        log_file = tmp_path / "log_cursor-agent_auto.txt"

        with patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_read.return_value = {
                "success": True,
                "data": [{"suggestion": "ok"}],
                "source": "file",
            }

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
                log_file=log_file,
            )

        assert result["success"] is True
        assert result["source"] == "file"

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "CURSOR-AGENT LOG" in content
        assert "Success: True" in content
        assert "Result source: file" in content
        # F9.2: the file channel is what answered for this run, so a header
        # naming a failed stdout parse beside `Success: True` would read as
        # a fault that never happened — the provider was told not to print
        # JSON on stdout at all.
        assert "Stdout: not used (file output)" in content
        assert "Stdout parse: failed" not in content
        # The stale, pre-resolution verdict must never survive as the log's
        # own claim — this is the exact bug: a completed, salvaged run
        # recorded as failed because the log was written before the file
        # channel got a chance to speak.
        assert "Success: False" not in content

    def test_log_gets_result_source_even_when_stdout_parsed_cleanly(
        self, mock_subprocess, mock_provider_available, tmp_path
    ):
        """F13: a clean stdout parse now defers too, so the log still names
        ``Result source:`` — previously (before F13) a clean parse's log was
        written by ``invoke_with_provider``'s own immediate call, before the
        file-vs-stdout choice existed to report, and ``Result source:`` was
        silently absent for exactly the runs that used the stdout fallback.
        ``invoke_with_file_output`` here takes the stdout_fallback branch,
        since the file read is mocked to fail.
        """
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[1, 2]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        log_file = tmp_path / "log.txt"

        with patch("utils.json_extractor.read_json_from_file") as mock_read, \
             patch("utils.llm_client._save_log", wraps=llm_client._save_log) as spy:
            mock_read.return_value = {"success": False, "error": "no file"}

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
                log_file=log_file,
            )

        assert result["success"] is True
        assert result["source"] == "stdout_fallback"
        # Still written exactly once — now via the deferred path instead of
        # invoke_with_provider's immediate call — so there is no double write.
        assert spy.call_count == 1
        content = log_file.read_text(encoding="utf-8")
        assert "Success: True" in content
        assert "Result source: stdout_fallback" in content

    def test_log_gets_result_source_file_when_stdout_and_file_both_resolve(
        self, mock_subprocess, mock_provider_available, tmp_path
    ):
        """F13's other clean-parse case: the file channel wins over a stdout
        parse that also succeeded, and the log names ``file``, not silence.
        """
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[1, 2]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        log_file = tmp_path / "log.txt"

        with patch("utils.json_extractor.read_json_from_file") as mock_read, \
             patch("utils.llm_client._save_log", wraps=llm_client._save_log) as spy:
            mock_read.return_value = {
                "success": True, "data": [{"a": 1}], "source": "file",
            }

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
                log_file=log_file,
            )

        assert result["success"] is True
        assert result["source"] == "file"
        assert spy.call_count == 1
        content = log_file.read_text(encoding="utf-8")
        assert "Success: True" in content
        assert "Result source: file" in content

    def test_log_reflects_total_failure_when_neither_channel_resolves(
        self, mock_subprocess, mock_provider_available, tmp_path
    ):
        """Neither file nor stdout parsed: the log still says so, exactly once."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is plain text without any JSON"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        log_file = tmp_path / "log.txt"

        with patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_read.return_value = {"success": False, "error": "no file"}

            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
                log_file=log_file,
            )

        assert result["success"] is False
        content = log_file.read_text(encoding="utf-8")
        assert "Success: False" in content
        assert "Error: JSON parsing failed from both file and stdout" in content
        assert "Stdout parse: failed" in content

    def test_log_writes_exactly_once_on_parse_failure_path(
        self, mock_subprocess, mock_provider_available, tmp_path
    ):
        """No stray immediate write races the deferred, final one."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is plain text without any JSON"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        log_file = tmp_path / "log.txt"

        with patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_read.return_value = {
                "success": True, "data": [{"a": 1}], "source": "file",
            }
            with patch(
                "utils.llm_client._save_log", wraps=llm_client._save_log
            ) as spy:
                invoke_with_file_output(
                    prompt_template="Write to {output_json_path}",
                    model_spec="cursor-agent:auto",
                    prompt_context={},
                    output_dir=tmp_path,
                    phase="test_phase",
                    log_file=log_file,
                )
        assert spy.call_count == 1

    def test_log_header_is_provider_neutral_via_deferred_write(
        self, mock_subprocess, mock_provider_available, tmp_path
    ):
        """The deferred write also names the real provider, not a fixed CLI."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is plain text without any JSON"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        log_file = tmp_path / "log.txt"

        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_read.return_value = {
                "success": True, "data": [{"a": 1}], "source": "file",
            }
            invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="claude-code:opus",
                prompt_context={},
                output_dir=tmp_path,
                phase="test_phase",
                log_file=log_file,
            )

        content = log_file.read_text(encoding="utf-8")
        assert "CLAUDE-CODE LOG" in content
        assert "CURSOR-AGENT LOG" not in content


class TestOutputLocationBlock:
    """The `{output_location}` block is rendered from a declared capability.

    An adapter running its CLI in a read-only chat mode (cursor `--mode ask`,
    aider `/ask`) cannot write the result file, and asking it to spends the whole
    turn on a refusal sentence: the measured cursor reply was "I can't write the
    requested file while Ask mode is active", after which the slot failed on
    "JSON parsing failed from both file and stdout". These pin that such an
    adapter is asked for stdout instead — and that every other adapter's prompt
    is byte-for-byte what it was.
    """

    def test_file_capable_provider_is_asked_for_the_file(self):
        block = render_output_location(
            "/tmp/out/claude-code_sonnet.json", "claude-code:sonnet"
        )

        assert block == FILE_OUTPUT_LOCATION.format(
            path="/tmp/out/claude-code_sonnet.json"
        )
        assert "Write your JSON output to this file" in block
        assert "/tmp/out/claude-code_sonnet.json" in block

    @pytest.mark.parametrize(
        "model_spec",
        [
            "cursor-agent:auto",
            "aider:openrouter/z-ai/glm-5.2",
            # D6: codex is not read-only, but its file write goes through a
            # bubblewrap sandbox that a stock Ubuntu 24.04 cannot start, and
            # the array was always its final `agent_message` — so the review
            # asks for stdout and never bets a slot on the sandbox.
            "codex:gpt-5.5",
        ],
    )
    def test_read_only_provider_is_asked_for_stdout(self, model_spec):
        block = render_output_location("/tmp/out/result.json", model_spec)

        assert block == STDOUT_OUTPUT_LOCATION
        assert "Write your JSON output to this file" not in block
        assert "/tmp/out/result.json" not in block
        assert "stdout" in block

    def test_unknown_provider_keeps_the_file_block(self):
        """A config naming an adapter this build no longer ships renders what
        every caller rendered before the capability existed."""
        block = render_output_location("/tmp/out/result.json", "no-such-provider:m")

        assert block == FILE_OUTPUT_LOCATION.format(path="/tmp/out/result.json")

    def test_shipped_review_prompts_carry_the_placeholder(self):
        """The three JSON review prompts delegate their tail to the renderer."""
        from utils.prompt_loader import load_prompt

        for name in ("code_review.txt", "plan_review.txt", "task_review.txt"):
            template = load_prompt(name)
            assert template.rstrip("\n").endswith("{output_location}"), name
            assert "Write your JSON output to this file" not in template, name

    def test_invoke_with_file_output_sends_the_stdout_prompt_to_cursor(self, tmp_path):
        """End-to-end: the prompt cursor's process really receives asks for stdout.

        The output path is still generated and still read first (a provider that
        writes the file anyway is served from it), but nothing in the prompt
        instructs cursor to write one.
        """
        with patch("utils.llm_client.invoke_with_provider") as mock_invoke, \
             patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_invoke.return_value = {
                "success": True,
                "data": [{"title": "from stdout"}],
                "details": {"provider": "cursor-agent"},
            }
            mock_read.return_value = {"success": False, "error": "File not found"}

            result = invoke_with_file_output(
                prompt_template="Review it.\n\n## Output Location\n{output_location}\n",
                model_spec="cursor-agent:auto",
                prompt_context={},
                output_dir=tmp_path,
                phase="code_review",
            )

        call_args = mock_invoke.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[0][0]
        assert STDOUT_OUTPUT_LOCATION in prompt
        assert "Write your JSON output to this file" not in prompt
        assert str(tmp_path) not in prompt

        # The stdout array still lands on the same per-model result path the
        # orchestrator reads — only the channel it arrived on changed.
        assert result["success"] is True
        assert result["data"] == [{"title": "from stdout"}]
        assert result["source"] == "stdout_fallback"
        assert result["output_file"].endswith("cursor-agent_auto.json")

    def test_invoke_with_file_output_still_names_the_file_for_claude_code(
        self, tmp_path
    ):
        """The unchanged half: a file-capable adapter's prompt is untouched.

        claude-code is the example since D6 moved codex to the stdout block.
        """
        with patch("utils.llm_client.invoke_with_provider") as mock_invoke, \
             patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_invoke.return_value = {"success": True, "data": [], "details": {}}
            mock_read.return_value = {"success": True, "data": [], "source": "file"}

            invoke_with_file_output(
                prompt_template="Review it.\n\n## Output Location\n{output_location}\n",
                model_spec="claude-code:sonnet",
                prompt_context={},
                output_dir=tmp_path,
                phase="code_review",
            )

        call_args = mock_invoke.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[0][0]
        assert "Write your JSON output to this file" in prompt
        assert str(tmp_path / "code-review" / "claude-code_sonnet.json") in prompt


class TestDescribeOutputFile:
    """F14: the fan-out's summary line must not claim a model wrote a file it
    only answered on stdout — ``[model] Output file: … (source: stdout_fallback)``
    reads as "the model wrote this" when it did not.
    """

    def test_file_source_says_output_file(self):
        line = describe_output_file("/tmp/out/claude-code_sonnet.json", "file")

        assert line == "Output file: /tmp/out/claude-code_sonnet.json (source: file)"

    def test_file_extracted_source_also_says_output_file(self):
        """A file the model wrote, just wrapped in a code fence, is still a
        file the model wrote."""
        line = describe_output_file("/tmp/out/result.json", "file_extracted")

        assert line.startswith("Output file:")

    def test_stdout_fallback_source_says_result_will_be_saved(self):
        line = describe_output_file(
            "/tmp/out/codex_gpt-5_6-terra.json", "stdout_fallback"
        )

        assert line == (
            "Result will be saved to: /tmp/out/codex_gpt-5_6-terra.json "
            "(source: stdout_fallback)"
        )
        assert "Output file:" not in line

    def test_unknown_source_also_says_result_will_be_saved(self):
        """`source` defaults to `"unknown"` when a caller omits it entirely —
        that is not a claim the model wrote anything either."""
        line = describe_output_file("/tmp/out/result.json", "unknown")

        assert line.startswith("Result will be saved to:")


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
        shim.write_text("@echo off", encoding="utf-8")
        exe = tmp_path / "codex.exe"
        exe.write_text("", encoding="utf-8")

        with patch("shutil.which", return_value=str(shim)):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(exe)]
        assert is_batch_shim is False

    def test_cmd_shim_without_sibling_exe_is_flagged(self, tmp_path):
        """A .cmd shim with no .exe and no node target is kept but flagged."""
        shim = tmp_path / "codex.cmd"
        shim.write_text("@echo off", encoding="utf-8")

        with patch("shutil.which", return_value=str(shim)):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(shim)]
        assert is_batch_shim is True

    def test_bat_shim_suffix_case_insensitive(self, tmp_path):
        """.BAT (any case) is treated as a batch shim too."""
        shim = tmp_path / "codex.BAT"
        shim.write_text("@echo off", encoding="utf-8")

        with patch("shutil.which", return_value=str(shim)):
            launcher, is_batch_shim = _resolve_executable("codex")

        assert launcher == [str(shim)]
        assert is_batch_shim is True

    @staticmethod
    def _write_npm_shim(tmp_path, name="codex", rel=r"node_modules\codex\bin\codex.js"):
        """Write an npm cmd-shim and its dispatched-to cli.js; return both."""
        script = tmp_path.joinpath(*rel.replace("\\", "/").split("/"))
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/usr/bin/env node\n", encoding="utf-8")
        shim = tmp_path / f"{name}.cmd"
        shim.write_text(
            "@ECHO off\r\n"
            "SETLOCAL\r\n"
            'IF EXIST "%dp0%\\node.exe" (\r\n'
            '  SET "_prog=%dp0%\\node.exe"\r\n'
            ") ELSE (\r\n"
            '  SET "_prog=node"\r\n'
            ")\r\n"
            f'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  "%dp0%\\{rel}" %*\r\n',
            encoding="utf-8",
        )
        return shim, script

    def test_cmd_shim_resolves_node_target_with_sibling_node_exe(self, tmp_path):
        """An npm shim resolves to `node.exe <cli.js>` when node.exe is adjacent."""
        shim, script = self._write_npm_shim(tmp_path)
        node_exe = tmp_path / "node.exe"
        node_exe.write_text("", encoding="utf-8")

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
        exe.write_text("", encoding="utf-8")

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

        # First which call: is_available() detection; second: M1's cursor flag
        # gate looking for a binary to read `--help` from (this path does not
        # exist, so the gate skips); third: resolution.
        with patch(
            "shutil.which",
            side_effect=["/usr/bin/cursor-agent", "/usr/bin/cursor-agent", None],
        ):
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
        content = log_file.read_text(encoding="utf-8")
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
            # M1's per-flag gate reads this binary's own --help before the
            # adapter will emit `--trust --mode ask`, so the stand-in answers it
            # the way the real CLI does. Without it the slot is refused
            # (version-too-low) and no launch happens at all.
            "if sys.argv[1:] == ['--help']:\n"
            "    print('  --trust  Trust the current workspace')\n"
            "    print('  --mode <mode>  plan | ask')\n"
            "    sys.exit(0)\n"
            'print(json.dumps({"type": "result", "result": json.dumps(sys.argv[1:])}))\n',
            encoding="utf-8",
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
        shim.write_text("@echo off", encoding="utf-8")
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
        shim.write_text("@echo off", encoding="utf-8")
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
        shim.write_text("@echo off", encoding="utf-8")

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
        shim.write_text("@echo off", encoding="utf-8")

        with patch("shutil.which", return_value=str(shim)), \
             patch("utils.llm_client._IS_WINDOWS", False), \
             patch("utils.llm_client.subprocess.run",
                   return_value=self._success_result()) as mock_run:
            result = invoke_with_provider(
                prompt='Say "hello" && exit', model_spec="cursor-agent:auto"
            )

        assert result["success"] is True
        mock_run.assert_called_once()
