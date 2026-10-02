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

    def test_the_byte_cut_keeps_the_end_of_the_stream_where_the_answer_is(
        self, mock_subprocess, mock_provider_available
    ):
        """2026-10-01: a head-only cut left a codex log ending at the diff read.

        Codex JSONL is a few huge ``aggregated_output`` lines followed by the final
        ``agent_message``; the cut must keep that ending and put the marker between.
        """
        stdout = "\n".join(
            ["h" * 30_000, "m" * 30_000, '{"type":"item.completed","text":"FINAL-VERDICT"}']
        )
        mock_subprocess.return_value = self._failed_run(stdout, "")

        result = invoke_with_provider(prompt="Test", model_spec="cursor-agent:gpt-4")

        excerpt = result["details"]["stdout_excerpt"]
        assert len(excerpt.encode("utf-8")) <= MAX_LOG_STREAM_BYTES
        assert excerpt.startswith("h")
        assert excerpt.endswith("FINAL-VERDICT\"}")
        assert "stream truncated at" in excerpt
        assert excerpt.index("stream truncated") < excerpt.index("FINAL-VERDICT")

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


from utils.llm_client import SALVAGE_TEXT_KEY, _attach_salvage_text  # noqa: E402
from utils.providers.pi import PiProvider  # noqa: E402

PI_SPEC = "pi:openrouter/deepseek/deepseek-v4.1-flash"


def _pi_stream(narration=(), final_text="", *, final_finished=False):
    """A pi ``--mode json`` stream: narration turns, then one last message.

    Each narration turn is a finished assistant message with a text part and a
    ``toolCall`` part. The last message streams ``final_text`` as
    ``text_delta``s and, unless ``final_finished``, is cut there — the reviewer
    killed while printing its answer.
    """
    events = [{"type": "session", "version": 3}]
    for index, text in enumerate(narration):
        call = {"type": "toolCall", "id": f"c{index}", "name": "read",
                "arguments": {"path": f"f{index}.py"}}
        events.append({"type": "message_start", "message": {
            "role": "assistant", "content": [], "stopReason": "pending"}})
        events.append({"type": "message_end", "message": {
            "role": "assistant", "content": [{"type": "text", "text": text}, call],
            "stopReason": "toolUse"}})
        events.append({"type": "tool_execution_end", "toolCallId": f"c{index}",
                       "result": {"content": [{"type": "text", "text": "ok"}]}})
    events.append({"type": "message_start", "message": {
        "role": "assistant", "content": [], "stopReason": "pending"}})
    for cut in range(0, len(final_text), 512):
        events.append({"type": "message_update", "assistantMessageEvent": {
            "type": "text_delta", "contentIndex": 0,
            "delta": final_text[cut:cut + 512]}})
    if final_finished:
        events.append({"type": "message_end", "message": {
            "role": "assistant", "content": [{"type": "text", "text": final_text}],
            "stopReason": "stop"}})
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _truncated_array(approx_bytes):
    """A findings array cut mid-record, about ``approx_bytes`` long."""
    record = json.dumps({"title": "Retry sleeps after the last attempt",
                         "desc": "x" * 200, "importance": "high"})
    count = max(1, approx_bytes // (len(record) + 4))
    return "[\n  " + ",\n  ".join([record] * count) + ',\n  {"title": "cut her'


class TestSalvageTextOnTheLosingPaths:
    """``details["salvage_text"]``: the adapter's answer text, beside the excerpt.

    ``stdout_excerpt`` is the raw stream's last 20 lines, and on a transport
    adapter (pi, omp) those are events whose ``content`` arrays pass the salvage
    gate as findings. ``_attach_salvage_text`` asks the adapter's
    ``salvage_text`` hook for the answer text of the **full** stdout, redacts it,
    then bounds it without ever cutting before the answer's opening bracket
    (plan: review-run-scope-salvage-and-failure-visibility §4, D4.3).
    """

    KILLED_MID_TEXT = (
        Path(__file__).parent / "fixtures" / "pi" / "killed-mid-text.stdout.txt"
    ).read_text(encoding="utf-8")

    @staticmethod
    def _run(stdout, returncode=1):
        return subprocess.CompletedProcess(
            args=["pi"], returncode=returncode, stdout=stdout, stderr=""
        )

    @staticmethod
    def _timeout(stdout):
        exc = subprocess.TimeoutExpired(cmd=["pi"], timeout=600)
        exc.stdout = stdout
        exc.stderr = ""
        return exc

    def _invoke(self, **patch_kwargs):
        with patch("utils.llm_client.subprocess.run", **patch_kwargs), \
             patch("shutil.which", return_value="/usr/bin/pi"):
            return invoke_with_provider(prompt="Test", model_spec=PI_SPEC)

    @pytest.mark.parametrize("path", ["non-zero exit", "exit-0 parse failure", "timeout"])
    def test_present_on_all_three_losing_paths(self, path):
        if path == "non-zero exit":
            result = self._invoke(return_value=self._run(self.KILLED_MID_TEXT))
        elif path == "exit-0 parse failure":
            result = self._invoke(return_value=self._run(self.KILLED_MID_TEXT, 0))
        else:
            result = self._invoke(side_effect=self._timeout(self.KILLED_MID_TEXT))

        assert result["success"] is False
        text = result["details"][SALVAGE_TEXT_KEY]
        assert text.startswith('[\n  {\n    "title": "The retry loop sleeps')
        # Only the cut last message: none of the narration turns before it.
        assert "The plan's retry schedule" not in text
        # The excerpt is still attached, for the log and the diagnosis.
        assert result["details"]["stdout_excerpt"]

    def test_a_killed_tool_call_is_present_and_empty(self):
        stdout = (
            Path(__file__).parent / "fixtures" / "pi" / "killed-mid-toolcall.stdout.txt"
        ).read_text(encoding="utf-8")

        result = self._invoke(side_effect=self._timeout(stdout))

        assert result["details"][SALVAGE_TEXT_KEY] == ""

    def test_a_credential_in_the_answer_text_is_redacted(self):
        secret = _synthesize("anthropic-api-key", "llm-client-salvage-text")
        stdout = _pi_stream(final_text=f'[{{"title": "leaked key {secret}"')

        result = self._invoke(side_effect=self._timeout(stdout))

        text = result["details"][SALVAGE_TEXT_KEY]
        assert secret not in text
        assert "<redacted:" in text
        assert text.startswith('[{"title": "leaked key')

    def test_absent_when_the_hook_has_no_opinion(self):
        """cursor-agent keeps the base hook (``None``): no reducer applies."""
        with patch("utils.llm_client.subprocess.run",
                   return_value=self._run('[{"title": "cut')), \
             patch("shutil.which", return_value="/usr/bin/cursor-agent"):
            result = invoke_with_provider(prompt="Test", model_spec="cursor-agent:gpt-4")

        assert result["success"] is False
        assert SALVAGE_TEXT_KEY not in result["details"]
        assert result["details"]["stdout_excerpt"] == '[{"title": "cut'

    def test_empty_when_the_hook_raises(self):
        """An override that raises still declared its stdout a transport."""
        with patch.object(PiProvider, "salvage_text", side_effect=RuntimeError("boom")):
            result = self._invoke(side_effect=self._timeout(self.KILLED_MID_TEXT))

        assert result["details"][SALVAGE_TEXT_KEY] == ""

    def test_a_non_string_return_is_no_opinion(self):
        details = {}
        stub = MagicMock()
        stub.salvage_text.return_value = ["not", "text"]

        _attach_salvage_text(details, stub, "anything")

        assert details == {}

    def test_all_three_losing_paths_route_through_the_one_helper(self):
        seen = []
        with patch.object(
            llm_client, "_attach_salvage_text",
            # `**_`: the sites now pass the one hook reading as `hook_text`
            # (plans/ask-answer-recovery-and-false-success.md D2.2).
            side_effect=lambda details, provider, stdout, **_: seen.append(stdout),
        ):
            self._invoke(return_value=self._run("nonzero stdout"))
            self._invoke(return_value=self._run("unparseable", 0))
            self._invoke(side_effect=self._timeout("timeout stdout"))

        assert seen == ["nonzero stdout", "unparseable", "timeout stdout"]

    # -- the cap keeps the answer ------------------------------------------

    def test_many_turns_then_a_truncated_array_keep_the_whole_array(self):
        """More than 45 KB of narration, then a 30 KB cut array: the array survives
        whole, opening bracket included, because only the last message counts."""
        narration = [f"Turn {i}: now I'll read the next file. " + "n" * 400
                     for i in range(150)]
        array = _truncated_array(30 * 1024)
        stdout = _pi_stream(narration, array)
        assert len(stdout.encode("utf-8")) > 2 * MAX_LOG_STREAM_BYTES

        details = {}
        _attach_salvage_text(details, PiProvider(), stdout)

        assert details[SALVAGE_TEXT_KEY] == array.strip()
        assert details[SALVAGE_TEXT_KEY].startswith("[\n  {")
        assert "Turn 0" not in details[SALVAGE_TEXT_KEY]

    def test_one_text_over_the_cap_is_windowed_from_the_arrays_bracket(self):
        """Long prose (with a closed ``[see 1]`` and a quote) ahead of a cut array:
        the window starts at the array's never-closed ``[``."""
        prose = ('I checked the "retry" path [see 1] and it\'s fine. ' * 600)
        array = _truncated_array(30 * 1024)
        stdout = _pi_stream(final_text=prose + "\n" + array, final_finished=True)

        details = {}
        _attach_salvage_text(details, PiProvider(), stdout)

        text = details[SALVAGE_TEXT_KEY]
        assert text == array.strip()
        assert len(text.encode("utf-8")) <= MAX_LOG_STREAM_BYTES

    def test_an_answer_longer_than_the_cap_is_cut_at_the_end_only(self):
        array = _truncated_array(80 * 1024)
        stdout = _pi_stream(final_text="Findings:\n" + array)

        details = {}
        _attach_salvage_text(details, PiProvider(), stdout)

        text = details[SALVAGE_TEXT_KEY]
        assert text.startswith("[\n  {")
        assert len(text.encode("utf-8")) <= MAX_LOG_STREAM_BYTES
        assert text.endswith(f"[salvage text truncated at {MAX_LOG_STREAM_BYTES} B]")

    def test_text_with_no_bracket_keeps_its_head(self):
        prose = "".join(f"line {i} of a long refusal\n" for i in range(4000))
        stdout = _pi_stream(final_text=prose, final_finished=True)

        details = {}
        _attach_salvage_text(details, PiProvider(), stdout)

        text = details[SALVAGE_TEXT_KEY]
        assert text.startswith("line 0 of a long refusal")
        assert len(text.encode("utf-8")) <= MAX_LOG_STREAM_BYTES


# --------------------------------------------------------------------------- #
# `details["answer_text"]`: the ask lane's whole answer
# (plans/ask-answer-recovery-and-false-success.md D2.2, T2.1), and the
# `incomplete` verdict riding on `details` (D3.1, T3.1's llm_client half).
# --------------------------------------------------------------------------- #

import re  # noqa: E402
from typing import Any, List, Tuple  # noqa: E402

from utils.json_extractor import looks_like_ndjson  # noqa: E402
from utils.llm_client import (  # noqa: E402
    ANSWER_TEXT_KEY,
    MAX_ANSWER_TEXT_BYTES,
    _attach_answer_text,
    _salvage_hook_text,
    generic_answer_text,
)
from utils.provider_registry import _PROVIDERS, get_provider  # noqa: E402
from utils.providers.base import classify_streams  # noqa: E402
from utils.providers.kilocode import KiloCodeProvider  # noqa: E402

from .test_provider_fixture_rows import MANIFESTS  # noqa: E402
from .test_salvage_text import (  # noqa: E402
    _codex_stdout_captures,
    _opencode_stdout_captures,
)

FIXTURES = Path(__file__).parent / "fixtures"
KILO_SPEC = "kilocode:moonshotai/kimi-k3"
ANSWER_CAP_MARKER = f"\n... [answer text truncated at {MAX_ANSWER_TEXT_BYTES} B]"

#: One spec per adapter that `invoke_with_provider` renders offline.
SPECS = {
    "agy": "agy:stub-model",
    "aider": "aider:openai/stub-model",
    "claude-code": "claude-code:haiku",
    "cline": "cline:stub-model",
    "codex": "codex:gpt-5",
    "copilot": "copilot:gpt-5",
    "cursor-agent": "cursor-agent:gpt-4",
    "gemini": "gemini:gemini-2.5-flash",
    "goose": "goose:stub-model",
    "grok": "grok:stub-model",
    "kilocode": KILO_SPEC,
    "kimi": "kimi:stub-model",
    "muse": "muse:stub-model",
    "omp": "omp:stub/stub-model",
    "opencode": "opencode:openrouter/stub",
    "pi": PI_SPEC,
    "qwen-code": "qwen-code:stub-model",
    "vibe": "vibe:stub-model",
}

#: grok's answer file from `runs/20260922T084502Z/opencode-ask`, byte for byte:
#: the error body the ask fallback once saved as an answer (§3.1). Same bytes as
#: `tests/test_stray_write_paths.py`'s `GROK_ERROR_BODY`.
GROK_ERROR_BODY = (
    '{"type":"error","message":"Not signed in. To authenticate without a '
    'browser, run:\\n  grok login --device-code\\n\\nAlternatively, set the '
    'XAI_API_KEY environment variable or run `grok login` on a machine with '
    'a browser."}'
)

#: Three direct answers a question may ask for, which a `"plain"` adapter
#: returns whole and an `"auto"` one reads as transport.
DIRECT_JSON_ANSWERS = {
    "object": '{"recommendation":"retry"}',
    "array": "[1, 2, 3]",
    "ndjson": (
        '{"type":"a","n":1}\n'
        '{"type":"b","n":2}\n'
        '{"type":"c","n":3}\n'
    ),
}

#: A prose answer about auth and rate limits: the shared classifier tables match
#: it, an adapter's own exit-zero markers do not.
AUTH_PROSE = (
    "Short answer: the service returns 401 Unauthorized when the key is wrong, "
    "and 429 Too Many Requests once you hit the rate limit."
)


def _fixture(relative: str) -> str:
    return (FIXTURES / relative).read_bytes().decode("utf-8")


def _spawn_returning(stdout, *, returncode=0, timeout=False, stderr=""):
    """A stand-in for `_run_provider_cli`: the child printed ``stdout``."""
    def spawn(cmd, **_kwargs):
        if timeout:
            exc = subprocess.TimeoutExpired(cmd=cmd, timeout=600)
            exc.stdout = stdout
            exc.stderr = stderr
            raise exc
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return spawn


def _invoke_answer(spec, stdout, *, returncode=0, timeout=False, want=True, **kwargs):
    """`invoke_with_provider` over the real adapter, only the subprocess stubbed."""
    binary = spec.split(":", 1)[0]
    with patch.object(
        llm_client, "_run_provider_cli",
        side_effect=_spawn_returning(stdout, returncode=returncode, timeout=timeout),
    ), patch("shutil.which", return_value=f"/usr/bin/{binary}"):
        return invoke_with_provider("Q", spec, want_answer_text=want, **kwargs)


def _nemotron_markdown() -> str:
    """A 124-line markdown answer holding a fenced JSON block (§2.1's shape).

    kilocode's parser finds the block, so the parse is ``success: True`` with a
    dict ``data`` and no ``raw`` — the run whose other 104 lines were lost.
    """
    lines = ["# Additional findings beyond the existing audit", ""]
    finding = 0
    while len(lines) < 100:
        finding += 1
        lines += [
            f"## Finding {finding}: session token reuse in path {finding}",
            "",
            f"The handler at step {finding} reads the token before it is refreshed.",
            "- Impact: a stale token is accepted for one more request.",
            "- Fix: refresh before the read, not after.",
            "",
        ]
    lines += ["## Summary as data", "", "```json"]
    lines += ['{"findings": %d, "highest": "medium"}' % finding, "```", ""]
    while len(lines) < 124:
        lines.append(f"- Follow-up {len(lines)}: re-check the audit's scope notes.")
    return "\n".join(lines[:124]) + "\n"


def _long_prose_with_fenced_json(approx_bytes: int) -> str:
    """A prose answer over the salvage window, with a fenced JSON block in it."""
    head = "The first line of a long answer: the plan is sound with two gaps.\n\n"
    paragraph = (
        "The retry ladder is described twice with different budgets, and the "
        "second description is the one the tasks implement. " * 4
    ) + "\n\n"
    body = paragraph * max(1, approx_bytes // (2 * len(paragraph)))
    block = '```json\n{"gaps": 2, "severity": "medium"}\n```\n\n'
    return head + body + block + body + "That is the whole answer."


class TestAnswerTextOnEveryExitThatRanTheChild:
    """``want_answer_text=True`` puts the whole answer on ``details["answer_text"]``."""

    KILLED_MID_TOOLCALL = _fixture("pi/killed-mid-toolcall.stdout.txt")
    KILLED_MID_TEXT = _fixture("pi/killed-mid-text.stdout.txt")

    def test_the_124_line_kilocode_answer_survives_byte_for_byte(self):
        stdout = _nemotron_markdown()
        assert len(stdout.splitlines()) == 124
        parsed = KiloCodeProvider().parse_output(stdout, "")
        assert parsed["success"] is True
        assert isinstance(parsed["data"], dict) and "raw" not in parsed

        result = _invoke_answer(KILO_SPEC, stdout)

        assert result["success"] is True
        assert result["details"][ANSWER_TEXT_KEY] == stdout

    @pytest.mark.parametrize("path", ["exit-0 parse failure", "non-zero exit", "timeout"])
    def test_a_killed_tool_call_gives_empty_answer_text(self, path):
        result = _invoke_answer(
            PI_SPEC, self.KILLED_MID_TOOLCALL,
            returncode=1 if path == "non-zero exit" else 0,
            timeout=path == "timeout",
        )

        assert result["success"] is False
        assert result["details"][ANSWER_TEXT_KEY] == ""

    def test_a_secret_in_a_plain_answer_is_redacted(self):
        secret = _synthesize("anthropic-api-key", "llm-client-answer-text-plain")
        stdout = f"The config you pasted leaks a key: {secret}\nRotate it first.\n"

        result = _invoke_answer(KILO_SPEC, stdout)

        answer = result["details"][ANSWER_TEXT_KEY]
        assert secret not in answer
        assert "<redacted:" in answer
        assert answer.startswith("The config you pasted leaks a key: ")

    def test_a_secret_in_a_hooked_answer_is_redacted(self):
        """pi's success keeps ``raw`` unredacted; ``answer_text`` never is."""
        secret = _synthesize("anthropic-api-key", "llm-client-answer-text-pi")
        answer_in = f"Rotate the key {secret} before anything else."
        stdout = _pi_stream(final_text=answer_in, final_finished=True)

        result = _invoke_answer(PI_SPEC, stdout)

        answer = result["details"][ANSWER_TEXT_KEY]
        assert secret not in answer
        assert "<redacted:" in answer
        assert answer.startswith("Rotate the key ")

    def test_a_2_mib_stdout_keeps_its_first_mib(self):
        stdout = "".join(f"line {i} of a very long answer\n" for i in range(80_000))
        assert len(stdout.encode("utf-8")) > 2 * MAX_ANSWER_TEXT_BYTES

        result = _invoke_answer(KILO_SPEC, stdout)

        answer = result["details"][ANSWER_TEXT_KEY]
        assert len(answer.encode("utf-8")) <= MAX_ANSWER_TEXT_BYTES
        assert answer.startswith("line 0 of a very long answer\n")
        assert answer.endswith(ANSWER_CAP_MARKER)
        assert stdout.startswith(answer[: -len(ANSWER_CAP_MARKER)])

    def test_a_60_kb_plain_answer_keeps_its_first_line(self):
        """The cut ``_bound_salvage_text`` would make — from the block's ``{`` —
        is never made on ``answer_text``."""
        prose = _long_prose_with_fenced_json(60 * 1024)
        assert len(prose.encode("utf-8")) > MAX_LOG_STREAM_BYTES
        assert not llm_client._bound_salvage_text(prose).startswith("The first line")

        assert generic_answer_text(prose) == prose
        result = _invoke_answer(KILO_SPEC, prose)
        assert result["details"][ANSWER_TEXT_KEY] == prose

    def test_a_60_kb_hooked_answer_keeps_its_first_line_and_salvage_keeps_its_window(self):
        prose = _long_prose_with_fenced_json(60 * 1024)
        stdout = _pi_stream(final_text=prose, final_finished=True)

        success = _invoke_answer(PI_SPEC, stdout)
        losing = _invoke_answer(PI_SPEC, stdout, returncode=1)

        assert success["success"] is True
        assert success["details"][ANSWER_TEXT_KEY] == prose
        assert losing["details"][ANSWER_TEXT_KEY] == prose
        # The same hook reading, cut the other way for salvage (D2.2).
        salvage = losing["details"][SALVAGE_TEXT_KEY]
        assert salvage.startswith('{"gaps": 2')
        assert len(salvage.encode("utf-8")) <= MAX_LOG_STREAM_BYTES

    #: Every pi and omp manifest row the adapter itself parses as a success.
    PI_OMP_SUCCESS_ROWS = sorted(
        (provider, row)
        for provider in ("pi", "omp")
        for row, body in MANIFESTS[provider]["rows"].items()
        if body.get("stdout")
        and get_provider(provider).parse_output(
            _fixture(f"{provider}/{body['stdout']}"), ""
        ).get("success")
    )

    def test_pi_and_omp_both_have_success_rows(self):
        assert {provider for provider, _ in self.PI_OMP_SUCCESS_ROWS} == {"pi", "omp"}

    @pytest.mark.parametrize(
        "row", PI_OMP_SUCCESS_ROWS, ids=[f"{p}/{r}" for p, r in PI_OMP_SUCCESS_ROWS]
    )
    def test_pi_and_omp_keep_their_answers_without_raw(self, row):
        """On every row the adapter parses as a success, ``answer_text`` is the
        text it puts in ``raw`` (redaction aside): ask stops reading ``raw``
        (D2.3), so this is what keeps those answers."""
        provider, name = row
        stdout = _fixture(f"{provider}/{MANIFESTS[provider]['rows'][name]['stdout']}")
        parsed = get_provider(provider).parse_output(stdout, "")

        result = _invoke_answer(SPECS[provider], stdout)

        assert result["success"] is True
        assert parsed["raw"].strip()
        assert result["details"][ANSWER_TEXT_KEY] == llm_client._redact(parsed["raw"])

    # -- the one hook reading ------------------------------------------------

    def _pi_path_stdout(self, path):
        if path == "success":
            return _pi_stream(final_text="[]", final_finished=True)
        return self.KILLED_MID_TEXT

    @staticmethod
    def _path_kwargs(path):
        return {
            "returncode": 1 if path == "non-zero exit" else 0,
            "timeout": path == "timeout",
        }

    EXIT_PATHS = ("success", "exit-0 parse failure", "non-zero exit", "timeout")

    @pytest.mark.parametrize("path", EXIT_PATHS)
    def test_a_raising_hook_gives_empty_for_both_keys_and_runs_once(self, path):
        with patch.object(
            PiProvider, "salvage_text", side_effect=RuntimeError("boom")
        ) as hook:
            result = _invoke_answer(
                PI_SPEC, self._pi_path_stdout(path), **self._path_kwargs(path)
            )

        assert hook.call_count == 1
        assert result["details"][ANSWER_TEXT_KEY] == ""
        if path == "success":
            assert result["success"] is True
            assert SALVAGE_TEXT_KEY not in result["details"]
        else:
            assert result["details"][SALVAGE_TEXT_KEY] == ""

    @pytest.mark.parametrize("path", EXIT_PATHS)
    def test_without_the_flag_no_key_and_salvage_text_is_unchanged(self, path):
        stdout = self._pi_path_stdout(path)

        without = _invoke_answer(PI_SPEC, stdout, want=False, **self._path_kwargs(path))
        with_flag = _invoke_answer(PI_SPEC, stdout, **self._path_kwargs(path))

        assert ANSWER_TEXT_KEY not in without["details"]
        assert ANSWER_TEXT_KEY in with_flag["details"]
        assert without["details"].get(SALVAGE_TEXT_KEY) == with_flag["details"].get(
            SALVAGE_TEXT_KEY
        )

    def test_the_hook_reading_helper_reads_the_three_answers(self):
        stub = MagicMock()
        stub.salvage_text.return_value = "text"
        assert _salvage_hook_text(stub, "x") == "text"
        stub.salvage_text.return_value = ["not", "text"]
        assert _salvage_hook_text(stub, "x") is None
        stub.salvage_text.side_effect = RuntimeError("boom")
        assert _salvage_hook_text(stub, "x") == ""
        assert _salvage_hook_text(object(), "x") is None

    # -- the returns that never ran the child --------------------------------

    def test_no_key_when_the_binary_is_missing(self):
        with patch("shutil.which", return_value=None):
            result = invoke_with_provider("Q", PI_SPEC, want_answer_text=True)

        assert result["error_code"] == ERROR_BINARY_NOT_FOUND
        assert ANSWER_TEXT_KEY not in result["details"]

    def test_no_key_on_a_launch_failure(self):
        with patch.object(
            llm_client, "_run_provider_cli", side_effect=PermissionError("denied")
        ), patch("shutil.which", return_value="/usr/bin/pi"):
            result = invoke_with_provider("Q", PI_SPEC, want_answer_text=True)

        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert ANSWER_TEXT_KEY not in result["details"]

    def test_answer_text_never_reaches_the_log(self, tmp_path):
        log = tmp_path / "log.txt"
        stdout = _nemotron_markdown()
        binary = "kilocode"
        with patch.object(
            llm_client, "_run_provider_cli", side_effect=_spawn_returning(stdout)
        ), patch("shutil.which", return_value=f"/usr/bin/{binary}"):
            result = invoke_with_provider(
                "Q", KILO_SPEC, log_file=log, want_answer_text=True
            )

        assert result["details"][ANSWER_TEXT_KEY] == stdout
        text = log.read_text(encoding="utf-8")
        assert "answer_text" not in text
        # The log keeps its own bound: the answer's head is not in it.
        assert "# Additional findings beyond the existing audit" not in text


class TestGenericAnswerText:
    """The reducer for an adapter whose hook has no opinion (D2.2, rules 1-3)."""

    def test_a_clean_claude_code_envelope_unwraps(self):
        stdout = _fixture("claude-code/success.stdout.txt")
        assert generic_answer_text(stdout) == json.loads(stdout)["result"]

    def test_a_clean_cursor_agent_envelope_unwraps(self):
        assert generic_answer_text(_fixture("cursor-agent/success.stdout.txt")) == '["ok"]'

    @pytest.mark.parametrize("relative", [
        "claude-code/error-envelope.stdout.txt",
        "cursor-agent/exit-zero-failure.stdout.txt",
    ])
    def test_an_is_error_envelope_gives_nothing(self, relative):
        stdout = _fixture(relative)
        assert json.loads(stdout)["is_error"] is True
        assert generic_answer_text(stdout) == ""

    def test_groks_error_body_gives_nothing(self):
        assert generic_answer_text(GROK_ERROR_BODY) == ""

    def test_an_ndjson_stream_gives_nothing(self):
        stdout = _fixture("pi/killed-mid-toolcall.stdout.txt")
        assert looks_like_ndjson(stdout)
        assert generic_answer_text(stdout) == ""

    def test_a_gemini_response_unwraps(self):
        stdout = _fixture("gemini/success.stdout.txt")
        assert generic_answer_text(stdout) == json.loads(stdout)["response"]

    def test_an_agy_error_status_without_an_error_key_gives_nothing(self):
        stdout = json.dumps({
            "conversation_id": "", "status": "ERROR",
            "response": "Partial words before the failure.",
        })
        assert generic_answer_text(stdout) == ""

    def test_an_agy_success_envelope_unwraps(self):
        stdout = json.dumps({
            "conversation_id": "conv-1", "status": "SUCCESS",
            "response": "The plan is fine.",
        })
        assert generic_answer_text(stdout) == "The plan is fine."

    def test_an_ansi_coloured_plain_stdout_comes_back_without_escapes(self):
        stdout = "\x1b[1mBold heading\x1b[0m\nThe \x1b[32mplan\x1b[0m is fine.\n"
        assert generic_answer_text(stdout) == "Bold heading\nThe plan is fine.\n"

    @pytest.mark.parametrize("name", sorted(DIRECT_JSON_ANSWERS))
    def test_plain_framing_returns_a_direct_json_answer_whole(self, name):
        stdout = DIRECT_JSON_ANSWERS[name]
        assert generic_answer_text(stdout, framing="plain") == stdout
        assert generic_answer_text(stdout, framing="auto") == ""

    @pytest.mark.parametrize("framing", ["auto", "plain"])
    def test_an_adapter_marker_turns_rule_3_into_nothing(self, framing):
        markers = (re.compile(r"error: 401 unauthorized"),)
        stdout = "\x1b[91mError: 401 Unauthorized\x1b[0m: invalid API key\n"

        assert generic_answer_text(
            stdout, framing=framing, exit_zero_markers=markers
        ) == ""
        assert generic_answer_text(
            "The plan is sound.\n", framing=framing, exit_zero_markers=markers
        ) == "The plan is sound.\n"

    @pytest.mark.parametrize("framing", ["auto", "plain"])
    def test_only_the_adapters_markers_never_the_shared_tables(self, framing):
        """A prose answer about 401s and rate limits trips the shared classifier
        tables, and still comes back whole: they are never consulted here."""
        assert classify_streams(AUTH_PROSE, "") is not None
        markers = (re.compile(r"error: 401 unauthorized"),)

        assert generic_answer_text(
            AUTH_PROSE, framing=framing, exit_zero_markers=markers
        ) == AUTH_PROSE

    def test_whitespace_is_nothing(self):
        assert generic_answer_text("  \n\t\n") == ""
        assert generic_answer_text(None) == ""

    @pytest.mark.parametrize("name", sorted(DIRECT_JSON_ANSWERS))
    def test_a_direct_json_answer_survives_the_real_kilocode_adapter(self, name):
        """kilocode's parser makes each of these non-string ``data`` with no
        ``raw``; ``"plain"`` framing is what keeps the answer (D2.2)."""
        stdout = DIRECT_JSON_ANSWERS[name]
        parsed = KiloCodeProvider().parse_output(stdout, "")
        assert not isinstance(parsed.get("data"), str)
        assert "raw" not in parsed

        result = _invoke_answer(KILO_SPEC, stdout)

        assert result["details"][ANSWER_TEXT_KEY] == stdout


# --------------------------------------------------------------------------- #
# Goal 3 as a sweep: over every adapter's committed stdout captures,
# `answer_text` is never transport.
# --------------------------------------------------------------------------- #

#: Keys only a transport envelope or record carries, never a model's answer.
_ENVELOPE_KEYS = frozenset({
    "is_error", "session_id", "sessionId", "conversation_id", "role",
    "stopReason", "subtype",
})
_NO_JSON = object()


def _whole_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return _NO_JSON


def _stdout_captures() -> List[Tuple[str, str]]:
    """(provider, path under ``tests/fixtures/<provider>/``) for every capture."""
    rows = []
    for directory in sorted(p for p in FIXTURES.iterdir() if p.is_dir()):
        provider = directory.name
        if provider not in _PROVIDERS:
            continue
        if provider == "codex":
            found = set(_codex_stdout_captures())
        elif provider == "opencode":
            found = set(_opencode_stdout_captures())
        else:
            found = {
                body["stdout"]
                for body in ((MANIFESTS.get(provider) or {}).get("rows") or {}).values()
                if body.get("stdout") and "not_applicable" not in body
            }
            found |= {
                path.relative_to(directory).as_posix()
                for path in directory.rglob("*")
                if path.is_file() and ".stdout." in path.name
            }
        rows += [(provider, relative) for relative in sorted(found)]
    return rows


def _transport_records(stdout: str) -> List[Any]:
    """The stdout's own JSON records: the whole value and its elements, or —
    for a stream (two or more lines that are JSON objects) — each line.

    A plain-text stdout has no records: aider prints its reply as a bare line,
    and a reply that is a JSON array is the answer, not a transport record.
    """
    records = []
    whole = _whole_json(stdout)
    if whole is not _NO_JSON:
        records.append(whole)
        if isinstance(whole, list):
            records.extend(whole)
    lines = [_whole_json(line) for line in stdout.splitlines()]
    if sum(isinstance(value, dict) for value in lines) >= 2:
        records.extend(value for value in lines if value is not _NO_JSON)
    return [record for record in records if isinstance(record, (dict, list))]


def _is_envelope(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("type") in ("result", "error") or bool(
            _ENVELOPE_KEYS & value.keys()
        )
    if isinstance(value, list):
        return any(_is_envelope(item) for item in value if isinstance(item, dict))
    return False


GOAL_3_ROWS = _stdout_captures()


@pytest.mark.parametrize(
    "provider,relative", GOAL_3_ROWS, ids=[f"{p}/{r}" for p, r in GOAL_3_ROWS]
)
def test_goal_3_answer_text_is_never_transport(provider, relative):
    """Never an event stream, never the raw stdout when that is one JSON value,
    never an envelope or record object.

    A correct answer may itself be a JSON document (claude-code's and gemini's
    ``success`` rows unwrap to a findings array), so "never parses as JSON"
    would fail on correct output. A ``"plain"``-framed adapter's JSON or NDJSON
    answer *is* its raw stdout, so it is exempt from the first two clauses and
    from matching its own records; the envelope-key clause still holds. No
    ``"plain"`` adapter has a committed capture today (kilocode has none), so
    the exemption covers no row yet.
    """
    stdout = _fixture(f"{provider}/{relative}")
    adapter = get_provider(provider)
    details = {}

    _attach_answer_text(details, adapter, stdout)

    answer = details[ANSWER_TEXT_KEY]
    plain = getattr(adapter, "answer_framing", "auto") == "plain"
    if not plain:
        assert not looks_like_ndjson(answer), f"an event stream: {answer[:200]!r}"
        if _whole_json(stdout) is not _NO_JSON:
            assert answer.strip() != stdout.strip(), "the raw JSON stdout"
    value = _whole_json(answer)
    if value is _NO_JSON:
        return
    assert not _is_envelope(value), f"an envelope or record: {answer[:200]!r}"
    if not plain:
        assert value not in _transport_records(stdout), f"a record: {answer[:200]!r}"


def test_the_goal_3_checks_catch_what_they_guard_against():
    """The sweep's predicates are live: each flags the transport it names."""
    envelope = _fixture("claude-code/success.stdout.txt")
    stream = _fixture("pi/killed-mid-toolcall.stdout.txt")
    first_event = stream.splitlines()[0]

    assert _is_envelope(json.loads(envelope))
    assert _is_envelope(json.loads(GROK_ERROR_BODY))
    assert looks_like_ndjson(stream)
    assert json.loads(first_event) in _transport_records(stream)
    # A plain-text reply line that is JSON is not a record of anything.
    assert _transport_records("Banner\n\n[1, 2]\n\nTokens: 1 sent.\n") == []


def test_the_goal_3_sweep_covers_every_adapter_with_captures():
    swept = {provider for provider, _ in GOAL_3_ROWS}
    with_dirs = {
        p.name for p in FIXTURES.iterdir() if p.is_dir() and p.name in _PROVIDERS
    }
    # kilocode's directory holds only its model catalog (§14).
    assert swept == with_dirs - {"kilocode"}
    # Not vacuous. Since 2026-10-02 opencode commits one release's captures
    # (the 72 rig cases of its pin), not five releases', so the sweep is ~280.
    assert len(GOAL_3_ROWS) > 250


# --------------------------------------------------------------------------- #
# `incomplete` rides on `details` (D3.1).
# --------------------------------------------------------------------------- #

class TestIncompleteRidesOnDetails:
    """``parse_output``'s ``incomplete`` moves to ``details["incomplete"]``.

    It is a completion verdict, not an ``error_code``: pi's ``length`` rows
    still end ``PARSE_ERROR`` with no code of the adapter's own, so
    ``invoke_with_file_output`` still looks for a findings file.
    """

    LENGTH_ROWS = ("length", "length-recovery")

    @staticmethod
    def _pi(row):
        return _fixture(f"pi/{MANIFESTS['pi']['rows'][row]['stdout']}")

    @pytest.mark.parametrize("row", LENGTH_ROWS)
    def test_a_length_stop_is_incomplete_beside_parse_error(self, row):
        result = _invoke_answer(PI_SPEC, self._pi(row), want=False)

        assert result["success"] is False
        assert result["error_code"] == ERROR_PARSE_ERROR
        assert result["details"].get("incomplete") is True
        assert "incomplete" not in result  # moved, not copied

    @pytest.mark.parametrize("case", ["success", "multi-assistant", "prose"])
    def test_a_finished_answer_carries_no_incomplete(self, case):
        if case == "prose":
            stdout = _pi_stream(
                final_text="The plan is sound; ship it.", final_finished=True
            )
        else:
            stdout = self._pi(case)

        result = _invoke_answer(PI_SPEC, stdout, want=False)

        if case == "prose":
            assert result["error_code"] == ERROR_PARSE_ERROR
        else:
            assert result["success"] is True
        assert "incomplete" not in result["details"]

    def test_the_schema_verdict_rebuild_keeps_it(self):
        """The rebuild keeps only ``details``; the key is already there."""
        from types import SimpleNamespace

        parsed = {"success": False, "error": "Truncated output", "raw": "x",
                  "data": None, "incomplete": True}
        verdict = SimpleNamespace(warning=None, failure_reason="schema_violation",
                                  retryable=True, marker=None, data=None)
        with patch.object(PiProvider, "parse_output", return_value=parsed), \
             patch.object(llm_client.structured_output, "schema_for_dispatch",
                          return_value=(None, "unvalidated")), \
             patch.object(llm_client.structured_output, "classify_attempt",
                          return_value=verdict):
            result = _invoke_answer(
                PI_SPEC, "anything", want=False, schema_phase="review-plan"
            )

        assert result["error_code"] == ERROR_PARSE_ERROR
        assert result["details"]["failure_reason"] == "schema_violation"
        assert result["details"]["incomplete"] is True

    @pytest.mark.parametrize("row", LENGTH_ROWS)
    def test_the_file_channel_still_looks_for_a_findings_file(self, row, tmp_path):
        findings = [{"title": "Written before the cut", "desc": "d",
                     "importance": "high"}]
        stdout = self._pi(row)

        def spawn(cmd, **kwargs):
            text = " ".join(map(str, cmd)) + str(kwargs.get("input") or "")
            target = Path(re.search(r"WRITE-TO (\S+\.json)", text).group(1))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(findings), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        with patch.object(llm_client, "_run_provider_cli", side_effect=spawn), \
             patch("shutil.which", return_value="/usr/bin/pi"):
            result = invoke_with_file_output(
                prompt_template="WRITE-TO {output_json_path}",
                model_spec=PI_SPEC,
                prompt_context={},
                output_dir=tmp_path,
                phase="review-plan",
            )

        assert result["success"] is True
        assert result["source"] == "file"
        assert result["data"] == findings
        assert result["details"].get("incomplete") is True

    EXIT_ZERO_LABELLED = sorted(
        (provider, row)
        for provider, manifest in MANIFESTS.items()
        for row, body in (manifest.get("rows") or {}).items()
        if body.get("exit_code") == 0 and body.get("stdout")
        and (body.get("expect") or {}).get("salvage") in ("partial", "answer")
    )

    @pytest.mark.parametrize(
        "provider,row", EXIT_ZERO_LABELLED,
        ids=[f"{p}/{r}" for p, r in EXIT_ZERO_LABELLED],
    )
    def test_the_labels_and_the_key_agree(self, provider, row):
        """Every exit-0 row labelled ``partial`` is incomplete; no ``answer`` is."""
        body = MANIFESTS[provider]["rows"][row]
        stdout = _fixture(f"{provider}/{body['stdout']}")

        result = _invoke_answer(SPECS[provider], stdout, want=False)

        expected = body["expect"]["salvage"] == "partial"
        assert bool(result["details"].get("incomplete")) is expected

    def test_the_label_sweep_has_both_kinds(self):
        labels = {
            MANIFESTS[p]["rows"][r]["expect"]["salvage"]
            for p, r in self.EXIT_ZERO_LABELLED
        }
        assert labels == {"partial", "answer"}


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


class TestWorkingLimitsBlock:
    """The `{working_limits}` block states the attempt's real minutes (D3.2).

    Measured on the 2026-09-25 review run: a pi code-review slot with a 1200 s
    budget started `timeout 1700 python -m pytest tests` and was killed
    mid-suite with nothing written, and a grok lane reported a test failure it
    had never run. The block is advisory, so what these pin is its content: the
    real number, the one-test-file allowance, the test-claim line, and no shell
    command (it reaches lanes that have no shell).
    """

    TEST_CLAIM_LINE = (
        "- Do NOT claim that a test passes or fails unless you ran it during "
        "this review and saw its result."
    )
    TEMPLATE = (
        "Review it.\n\n{working_limits}\n\n## Output Location\n{output_location}\n"
    )

    def test_a_1200_s_slot_reads_20_minutes(self):
        from utils.llm_client import render_working_limits

        block = render_working_limits(1200)

        assert block.startswith("## Working limits\n")
        assert "stopped after about 20 minutes." in block

    def test_a_59_s_slot_reads_1_minute_never_0(self):
        from utils.llm_client import render_working_limits

        block = render_working_limits(59)

        assert "stopped after about 1 minute." in block
        assert "0 minutes" not in block

    def test_minutes_are_floored_so_the_prompt_never_overpromises(self):
        from utils.llm_client import render_working_limits

        assert "about 19 minutes." in render_working_limits(1199.9)

    def test_the_block_allows_one_test_file_and_names_no_shell_command(self):
        from utils.llm_client import render_working_limits

        block = render_working_limits(1200)

        assert self.TEST_CLAIM_LINE in block
        assert "you may run that ONE narrowly scoped test file" in block
        assert "Do NOT run the project's test suite" in block
        # cursor-agent and claude-code have no shell in this lane, and
        # `{diff_access}` is what grants or denies one; a command named here
        # would contradict it in the same prompt.
        for token in ("`", "git ", "pytest", "npm ", "make "):
            assert token not in block, token

    def test_an_echoed_block_trips_no_failure_classifier(self):
        """Four adapters echo our prompt into their stream; `plan_review.txt`'s
        one word "timeout" once classified a clean goose run as a timeout."""
        from utils.llm_client import render_working_limits
        from utils.providers.base import classify_streams

        block = render_working_limits(1200)

        assert classify_streams(block, "") is None
        assert classify_streams("", block) is None

    def test_invoke_with_file_output_renders_the_block_from_its_timeout(
        self, tmp_path
    ):
        from utils.llm_client import render_working_limits

        with patch("utils.llm_client.invoke_with_provider") as mock_invoke, \
             patch("utils.json_extractor.read_json_from_file") as mock_read:
            mock_invoke.return_value = {"success": True, "data": [], "details": {}}
            mock_read.return_value = {"success": True, "data": [], "source": "file"}

            invoke_with_file_output(
                prompt_template=self.TEMPLATE,
                model_spec="claude-code:sonnet",
                prompt_context={},
                output_dir=tmp_path,
                phase="code_review",
                timeout=1200,
            )

        call_args = mock_invoke.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[0][0]
        # The minutes and the kill come from the same number.
        assert call_args[0][2] == 1200
        assert render_working_limits(1200) in prompt
        assert "stopped after about 20 minutes." in prompt
        assert self.TEST_CLAIM_LINE in prompt

    def test_a_missing_timeout_renders_the_providers_own_default(self, tmp_path):
        """`timeout=None` is resolved the way `invoke_with_provider` resolves
        it, so the prompt names the budget the provider default really gives."""
        with patch("utils.llm_client.invoke_with_provider") as mock_invoke, \
             patch("utils.json_extractor.read_json_from_file") as mock_read, \
             patch("utils.llm_client.get_provider_timeout", return_value=300):
            mock_invoke.return_value = {"success": True, "data": [], "details": {}}
            mock_read.return_value = {"success": True, "data": [], "source": "file"}

            invoke_with_file_output(
                prompt_template=self.TEMPLATE,
                model_spec="claude-code:sonnet",
                prompt_context={},
                output_dir=tmp_path,
                phase="code_review",
            )

        call_args = mock_invoke.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[0][0]
        assert "stopped after about 5 minutes." in prompt


class TestDiffAccessBlock:
    """The `{diff_access}` block is rendered from two `generate` declarations.

    Measured on the 2026-09-25 review run (plan: review-run-scope-salvage-and-
    failure-visibility §2.1): the code-review prompt only told reviewers to run
    `git diff`, the two cursor lanes had no shell, kimi returned `[]` without
    ever seeing a diff and grok reported a test failure it could not have run.
    The diff now arrives as files in the phase dir, and this block tells each
    lane how to reach them. These pin the three shell-state texts, their
    partial-coverage additions, the inline branch for a lane that cannot open
    the phase dir, and the unsupported outcome when inlining would be partial
    or would not fit one argv string.
    """

    BASE_REF = "abc1234"
    PLAN_TEXT = "# Feature plan\n\nAdd `frob()` to utils/x.py.\n"
    MANIFEST_TEXT = (
        "# Review scope\n\n"
        "Diff coverage: complete\n\n"
        "## source\n- utils/x.py +2 -1 diff/1-utils-x.py.diff (full, 40 B)\n\n"
        "## tests\n- tests/test_x.py +3 -0 diff/2-tests-test_x.py.diff (full, 30 B)\n"
    )
    SOURCE_DIFF = "--- a/utils/x.py\n+++ b/utils/x.py\n-old\n+def frob(): ...\n"
    TEST_DIFF = "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n+def test_frob(): ...\n"
    #: The `True` text's cross-check sentence, which no other state may carry.
    CROSS_CHECK = (
        "`git diff abc1234`, `git show` and `git log` are an optional cross-check "
        "of the diff files"
    )
    #: Plan D2.3's `False` and `None` first paragraphs, word for word.
    FALSE_HEAD = (
        "You cannot run shell commands. The diff is at {diff_dir}; read it with "
        "your file tools. Do not claim a test passes or fails, because you "
        "cannot run it."
    )
    NONE_HEAD = (
        "The diff files at {diff_dir} are the authoritative record of the "
        "change; read them with your file tools. Do not claim a test passes or "
        "fails unless you actually ran it and saw the result."
    )

    @pytest.fixture
    def art(self, tmp_path):
        """A repo root with a plan and a written code-review diff set inside it."""
        from types import SimpleNamespace

        repo = tmp_path / "repo"
        phase = repo / "plans" / "feature" / "code-review"
        diff_dir = phase / "diff"
        diff_dir.mkdir(parents=True)
        plan = repo / "plans" / "feature.md"
        plan.write_text(self.PLAN_TEXT, encoding="utf-8")
        scope_file = phase / "review-scope.md"
        scope_file.write_text(self.MANIFEST_TEXT, encoding="utf-8")
        (diff_dir / "1-utils-x.py.diff").write_text(self.SOURCE_DIFF, encoding="utf-8")
        (diff_dir / "2-tests-test_x.py.diff").write_text(self.TEST_DIFF, encoding="utf-8")
        (diff_dir / ".gitignore").write_text("*\n", encoding="utf-8")
        return SimpleNamespace(
            root=tmp_path, repo=repo, phase=phase, diff_dir=diff_dir,
            scope_file=scope_file, plan=plan,
        )

    @pytest.fixture
    def stub(self):
        """Register a `stub:` adapter whose two declarations a test sets."""
        from types import SimpleNamespace

        from utils import provider_registry

        provider = SimpleNamespace(runs_shell_in_generate=None, generate_read_scope="cwd")
        with patch.dict(provider_registry._PROVIDERS, {"stub": provider}):
            yield provider

    def render(self, spec, art, *, coverage="complete", cwd=None):
        from utils.llm_client import render_diff_access

        return render_diff_access(
            spec,
            str(art.scope_file),
            str(art.diff_dir),
            str(art.phase),
            str(cwd if cwd is not None else art.repo),
            base_ref=self.BASE_REF,
            coverage=coverage,
            plan_path=str(art.plan),
        )

    # -- the three shell-state texts ---------------------------------------

    def test_true_names_both_paths_and_grants_git_as_a_cross_check(self, art, stub):
        stub.runs_shell_in_generate = True

        access = self.render("stub:m", art)

        assert access.mode == "paths"
        assert (access.runs_shell, access.read_scope) == (True, "cwd")
        assert (access.inline_bytes, access.reason) == (0, "")
        assert str(art.scope_file) in access.text
        assert str(art.diff_dir) in access.text
        assert "authoritative record of the change" in access.text
        assert self.CROSS_CHECK in access.text
        # The guidance the static template used to hold travels with the grant.
        assert "`git diff abc1234 -- path/to/file`" in access.text
        assert "`git diff abc1234 -- <file1> <file2> ...`" in access.text
        assert (
            "`git diff abc1234 -- <path>` is required for any file the manifest "
            "marks `omitted`"
        ) in access.text

    def test_false_is_plan_d23s_text_word_for_word(self, art, stub):
        stub.runs_shell_in_generate = False

        access = self.render("stub:m", art)

        assert access.mode == "paths"
        assert access.text.startswith(self.FALSE_HEAD.format(diff_dir=art.diff_dir))
        assert str(art.scope_file) in access.text
        assert "authoritative record of the change" in access.text
        assert "git " not in access.text

    def test_none_neither_grants_nor_denies_a_shell(self, art, stub):
        stub.runs_shell_in_generate = None

        access = self.render("stub:m", art)

        assert access.mode == "paths"
        assert access.text.startswith(self.NONE_HEAD.format(diff_dir=art.diff_dir))
        assert str(art.scope_file) in access.text
        assert "You cannot run shell commands" not in access.text
        assert self.CROSS_CHECK not in access.text
        assert "git " not in access.text

    @pytest.mark.parametrize("coverage", ["complete", "partial"])
    @pytest.mark.parametrize("shell", [True, False, None])
    def test_only_the_true_text_names_a_git_command(self, art, stub, shell, coverage):
        """`{diff_access}` is the one block that may grant a shell command, and
        only its `True` rendering does. `None`'s partial sentence names the
        command the MANIFEST gives, conditionally — a pointer, not a grant."""
        from utils.llm_client import DIFF_ACCESS_UNMEASURED_PARTIAL

        stub.runs_shell_in_generate = shell

        text = self.render("stub:m", art, coverage=coverage).text

        if shell is True:
            for command in ("git diff abc1234", "`git show`", "`git log`"):
                assert command in text, command
            return
        for command in ("git show", "git log", "git diff abc1234", self.CROSS_CHECK):
            assert command not in text, command
        if shell is None and coverage == "partial":
            assert text.count("git ") == DIFF_ACCESS_UNMEASURED_PARTIAL.count("git ") == 1
            assert "If you can run the `git diff` command the manifest gives" in text
        else:
            assert "git " not in text

    # -- partial coverage ---------------------------------------------------

    def test_partial_false_lists_omitted_files_as_not_reviewed(self, art, stub):
        stub.runs_shell_in_generate = False

        text = self.render("stub:m", art, coverage="partial").text

        assert text.endswith(
            "Files the manifest marks `omitted` have no diff you can read. List "
            "them as not reviewed. Do not review their current contents as if "
            "they were the change."
        )

    def test_partial_true_requires_git_diff_for_omitted_files(self, art, stub):
        stub.runs_shell_in_generate = True

        text = self.render("stub:m", art, coverage="partial").text

        assert "Diff coverage for this change is partial" in text
        assert "Run `git diff abc1234 -- <path>` for each of them" in text
        assert "Do not review their current contents as if they were the change." in text

    def test_partial_none_reviews_from_git_diff_only_if_it_ran_it(self, art, stub):
        stub.runs_shell_in_generate = None

        text = self.render("stub:m", art, coverage="partial").text

        assert text.endswith(
            "Files the manifest marks `omitted` have no diff file. If you can run "
            "the `git diff` command the manifest gives for one, review it from "
            "that output; otherwise list it as not reviewed. Do not review their "
            "current contents as if they were the change."
        )

    @pytest.mark.parametrize("shell", [True, False, None])
    def test_complete_coverage_carries_no_partial_sentence(self, art, stub, shell):
        from utils.llm_client import (
            DIFF_ACCESS_NO_SHELL_PARTIAL,
            DIFF_ACCESS_SHELL_PARTIAL,
            DIFF_ACCESS_UNMEASURED_PARTIAL,
        )

        stub.runs_shell_in_generate = shell

        text = self.render("stub:m", art).text

        assert "Diff coverage for this change is partial" not in text
        assert DIFF_ACCESS_NO_SHELL_PARTIAL not in text
        assert DIFF_ACCESS_UNMEASURED_PARTIAL not in text
        assert DIFF_ACCESS_SHELL_PARTIAL.format(base_ref=self.BASE_REF) not in text

    def test_an_unknown_coverage_word_reads_as_partial(self, art, stub):
        """The conservative reading: only `complete` promises every diff."""
        stub.runs_shell_in_generate = False

        text = self.render("stub:m", art, coverage="").text

        assert "List them as not reviewed." in text

    # -- the adapters' own declarations -------------------------------------

    @pytest.mark.parametrize(
        "spec,shell",
        [("codex:gpt-5.5", True), ("cursor-agent:auto", False), ("grok:grok-4", None)],
    )
    def test_a_real_adapter_renders_from_its_declaration(self, art, spec, shell):
        from utils.provider_registry import get_provider

        access = self.render(spec, art)

        assert access.mode == "paths"
        assert access.runs_shell is shell
        assert access.runs_shell is get_provider(spec.split(":")[0]).runs_shell_in_generate

    def test_an_unknown_adapter_reads_under_cwd_and_asserts_no_shell(self, art):
        access = self.render("no-such-provider:m", art)

        assert (access.mode, access.read_scope, access.runs_shell) == ("paths", "cwd", None)
        assert access.text.startswith(self.NONE_HEAD.format(diff_dir=art.diff_dir))

    # -- which lanes can open the phase dir ---------------------------------

    def test_cwd_with_the_phase_dir_outside_the_repo_goes_inline(self, art, stub):
        """A plan kept outside the repo puts its phase dir where a `"cwd"` lane
        cannot read, so the paths it would be handed are useless to it."""
        elsewhere = art.root / "other-repo"
        elsewhere.mkdir()

        access = self.render("stub:m", art, cwd=elsewhere)

        assert access.mode == "inline"
        assert access.text.startswith(
            "The review's artifacts sit outside the directory you can read, so "
            "everything you need is inline below:"
        )

    def test_any_gets_the_paths_even_outside_the_repo(self, art, stub):
        stub.generate_read_scope = "any"
        elsewhere = art.root / "other-repo"
        elsewhere.mkdir()

        access = self.render("stub:m", art, cwd=elsewhere)

        assert (access.mode, access.read_scope) == ("paths", "any")
        assert str(art.scope_file) in access.text

    def test_cwd_resolves_through_symlinks(self, art, stub, tmp_path):
        """Containment compares resolved paths on both sides, so a repo reached
        through a symlink still contains its own phase dir."""
        link = tmp_path / "repo-link"
        link.symlink_to(art.repo, target_is_directory=True)

        access = self.render("stub:m", art, cwd=link)

        assert access.mode == "paths"

    # -- the inline branch ---------------------------------------------------

    def test_inline_carries_plan_manifest_and_every_diff_under_the_cap(self, art):
        from utils.llm_client import INLINE_DIFF_CAP_BYTES

        access = self.render("qwen-code:qwen/qwen3.8-27b", art)

        assert (access.mode, access.read_scope, access.runs_shell) == ("inline", "none", False)
        assert access.reason == ""
        text = access.text
        assert text.startswith(
            "You cannot open files in this review, so everything you need is "
            "inline below:"
        )
        for body in (self.PLAN_TEXT, self.MANIFEST_TEXT, self.SOURCE_DIFF, self.TEST_DIFF):
            assert body.rstrip("\n") in text
        # Each under a header naming it, the diffs in name (= review) order, and
        # the ignore file is not a diff.
        order = [
            text.index("===== BEGIN plan ====="),
            text.index("===== BEGIN review-scope.md ====="),
            text.index("===== BEGIN diff/1-utils-x.py.diff ====="),
            text.index("===== BEGIN diff/2-tests-test_x.py.diff ====="),
        ]
        assert order == sorted(order)
        assert ".gitignore" not in text
        # The measured size is the block's own, and it fits.
        assert access.inline_bytes == len(text.encode("utf-8"))
        assert access.inline_bytes <= INLINE_DIFF_CAP_BYTES
        # qwen-code has no shell either: its sentence says so, and grants nothing.
        assert "You cannot run shell commands." in text
        assert "git " not in text.split("===== BEGIN plan =====")[0]

    def test_inline_names_no_path_the_lane_cannot_open(self, art):
        access = self.render("qwen-code:qwen/qwen3.8-27b", art)

        assert access.mode == "inline"
        assert str(art.root) not in access.text

    def test_aider_gets_the_change_inline_even_with_the_phase_dir_in_the_repo(self, art):
        """aider's `/ask` has no file tool and its stdin supply declines every
        mention add, so a path prompt would leave it with neither the diff files
        (`{diff_dir}` is a directory, never passed by `--read`) nor the changed
        code. Its slot takes the inline branch whatever the phase dir's place
        (plan: ask-answer-recovery-and-false-success D4.2)."""
        assert str(art.phase).startswith(str(art.repo))

        access = self.render("aider:openrouter/some/model", art)

        assert (access.mode, access.read_scope) == ("inline", "none")
        for body in (self.PLAN_TEXT, self.MANIFEST_TEXT, self.SOURCE_DIFF, self.TEST_DIFF):
            assert body.rstrip("\n") in access.text
        assert str(art.diff_dir) not in access.text

    def test_aider_over_the_cap_is_unsupported_not_a_path_prompt(self, art):
        (art.diff_dir / "3-utils-big.py.diff").write_text(
            "+" + "x" * (100 * 1024) + "\n", encoding="utf-8"
        )

        access = self.render("aider:openrouter/some/model", art)

        assert (access.mode, access.text) == ("unsupported", "")

    def test_the_cap_leaves_room_under_one_argv_string(self):
        from utils.llm_client import INLINE_DIFF_CAP_BYTES

        assert INLINE_DIFF_CAP_BYTES == 96 * 1024
        assert INLINE_DIFF_CAP_BYTES < 131072  # Linux MAX_ARG_STRLEN

    def test_over_the_cap_is_unsupported(self, art):
        from utils.llm_client import INLINE_DIFF_CAP_BYTES

        (art.diff_dir / "3-utils-big.py.diff").write_text(
            "+" + "x" * (100 * 1024) + "\n", encoding="utf-8"
        )

        access = self.render("qwen-code:qwen/qwen3.8-27b", art)

        assert access.mode == "unsupported"
        assert access.text == ""
        assert access.inline_bytes > INLINE_DIFF_CAP_BYTES
        assert access.reason == (
            f"inline context {access.inline_bytes} B exceeds the "
            f"{INLINE_DIFF_CAP_BYTES} B cap"
        )

    def test_partial_coverage_on_a_no_read_lane_is_unsupported(self, art):
        """A partial inline review would read as a review of the whole change."""
        access = self.render("qwen-code:qwen/qwen3.8-27b", art, coverage="partial")

        assert access.mode == "unsupported"
        assert access.text == ""
        assert access.reason == "diff coverage is partial"
        assert 0 < access.inline_bytes <= 96 * 1024

    def test_a_missing_manifest_is_unsupported_not_raised(self, art):
        art.scope_file.unlink()

        access = self.render("qwen-code:qwen/qwen3.8-27b", art)

        assert access.mode == "unsupported"
        assert access.text == ""
        assert "scope manifest not readable" in access.reason

    def test_a_missing_plan_or_diff_dir_is_unsupported_not_raised(self, art):
        import shutil

        art.plan.unlink()
        shutil.rmtree(art.diff_dir)

        access = self.render("qwen-code:qwen/qwen3.8-27b", art)

        assert access.mode == "unsupported"
        assert "plan file not readable" in access.reason
        # The manifest still names two diffs that are no longer there.
        assert "the manifest names 2 diff file(s) that do not exist" in access.reason

    def test_a_diff_the_manifest_names_but_is_gone_is_unsupported(self, art):
        (art.diff_dir / "2-tests-test_x.py.diff").unlink()

        access = self.render("qwen-code:qwen/qwen3.8-27b", art)

        assert access.mode == "unsupported"
        assert access.reason == (
            "the manifest names 1 diff file(s) that do not exist: "
            "2-tests-test_x.py.diff"
        )

    # -- hygiene -------------------------------------------------------------

    @pytest.mark.parametrize("coverage", ["complete", "partial"])
    @pytest.mark.parametrize("shell", [True, False, None])
    def test_an_echoed_block_trips_no_failure_classifier(self, art, stub, shell, coverage):
        """Several adapters echo our prompt into their stream; a block that
        reads as a failure would turn a clean run into a classified one."""
        from utils.providers.base import classify_streams

        stub.runs_shell_in_generate = shell
        text = self.render("stub:m", art, coverage=coverage).text

        assert classify_streams(text, "") is None
        assert classify_streams("", text) is None


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


# ---------------------------------------------------------------------------
# Stage U — `details["usage"]` on every path out of this module
# ---------------------------------------------------------------------------
#
# `_attach_usage` used to run on the losing paths only, so the run that
# *worked* was the one nothing could report the cost of. It now runs on the
# exit-0 path too, which means the key has to survive three more things: the
# schema verdict that REPLACES the parsed dict, the four ways
# `invoke_with_file_output` can end, and an exit-0 run that its adapter calls a
# hard failure — the shape that also has a deferred log nobody was writing.


def _usage_envelope(result_text, *, is_error=False):
    """A claude-code result envelope carrying usage, around any `result`."""
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "result": result_text,
        "usage": {
            "input_tokens": 900,
            "output_tokens": 300,
            "cache_read_input_tokens": 50,
        },
        "total_cost_usd": 0.42,
    })


#: What `ClaudeCodeProvider.parse_usage` makes of the envelope above.
ENVELOPE_USAGE = {
    "input_tokens": 900,
    "output_tokens": 300,
    "cached_input_tokens": 50,
    "cost_usd": 0.42,
    "total_tokens": 1200,
}

from utils.providers.base import LLMProvider as _BaseProvider  # noqa: E402

_USAGE_ISSUES = [{
    "title": "Boundary check missing", "desc": "d", "importance": "HIGH",
    "reference": "Step 1", "type": "bug",
}]


def _usage_completed(stdout, returncode=0, stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


class TestExitZeroCarriesUsage:
    """The exit-0 half of `_attach_usage`, on all three of its outcomes."""

    @staticmethod
    def _invoke(stdout, **kwargs):
        with patch("shutil.which", return_value="/usr/bin/claude"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_usage_completed(stdout)):
            return invoke_with_provider("p", "claude-code:sonnet", **kwargs)

    def test_a_clean_parse_carries_what_the_run_spent(self):
        result = self._invoke(_usage_envelope(json.dumps(_USAGE_ISSUES)))

        assert result["success"] is True
        assert result["details"]["usage"] == ENVELOPE_USAGE

    def test_a_parse_failure_carries_it_too(self):
        """Nothing usable came back; that is not the same as nothing was spent."""
        result = self._invoke(_usage_envelope("The plan looks fine to me."))

        assert result["success"] is False
        assert result["error_code"] == ERROR_PARSE_ERROR
        assert result["details"]["usage"] == ENVELOPE_USAGE

    def test_the_schema_verdict_replaces_the_dict_and_keeps_the_key(self):
        """§4.2.1's refusal builds a NEW envelope; its `details` spread is the
        only reason the numbers attached a few lines earlier survive it."""
        result = self._invoke(
            _usage_envelope(json.dumps(_USAGE_ISSUES)),
            schema_phase="generate-tasks",
        )

        assert result["success"] is False
        assert result["details"]["failure_reason"]
        assert result["details"]["usage"] == ENVELOPE_USAGE

    def test_an_adapter_that_reports_nothing_still_adds_no_key(self):
        """The absent-means-unreported rule, unchanged by the new call site."""
        wrapper = json.dumps({"type": "result", "result": json.dumps(_USAGE_ISSUES)})
        with patch("shutil.which", return_value="/usr/bin/cursor-agent"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_usage_completed(wrapper)):
            result = invoke_with_provider("p", "cursor-agent:auto")

        assert result["success"] is True
        assert "usage" not in result["details"]


class TestFileOutputCarriesUsage:
    """All four ways `invoke_with_file_output` returns, one key to survive."""

    @staticmethod
    def _details():
        return {
            "provider": "claude-code", "model": "sonnet",
            "duration_seconds": 1.0, "usage": dict(ENVELOPE_USAGE),
        }

    def _call(self, tmp_path, stub, **kwargs):
        with patch("utils.llm_client.invoke_with_provider", **stub):
            return invoke_with_file_output(
                prompt_template="write to {output_json_path}",
                model_spec="claude-code:sonnet",
                prompt_context={"prefix": "p"},
                output_dir=tmp_path,
                phase="code_review",
                **kwargs,
            )

    def test_the_file_channel_forwards_it(self, tmp_path):
        def _write_the_file(prompt, *args, **kwargs):
            # The path the caller asked for is in the rendered prompt, which is
            # the only place this stub can learn it.
            path = prompt.rsplit("write to ", 1)[1].strip().splitlines()[0]
            Path(path).write_text(json.dumps(_USAGE_ISSUES), encoding="utf-8")
            return {"success": False, "error": "no stdout JSON", "data": None,
                    "error_code": ERROR_PARSE_ERROR, "details": self._details()}

        out = self._call(tmp_path, {"side_effect": _write_the_file})

        assert out["success"] is True
        assert out["source"] in ("file", "file_extracted")
        assert out["details"]["usage"] == ENVELOPE_USAGE

    def test_the_stdout_fallback_forwards_it(self, tmp_path):
        stub = {"return_value": {
            "success": True, "data": _USAGE_ISSUES, "details": self._details(),
        }}
        out = self._call(tmp_path, stub)

        assert out["source"] == "stdout_fallback"
        assert out["details"]["usage"] == ENVELOPE_USAGE

    def test_both_channels_failing_forwards_it(self, tmp_path):
        stub = {"return_value": {
            "success": False, "error": "unparseable", "data": None,
            "error_code": ERROR_PARSE_ERROR, "details": self._details(),
        }}
        out = self._call(tmp_path, stub)

        assert out["success"] is False
        assert out["error_code"] == ERROR_PARSE_ERROR
        assert out["details"]["usage"] == ENVELOPE_USAGE

    def test_a_hard_failure_forwards_it(self, tmp_path):
        """The return that spreads `**result` rather than rebuilding it."""
        stub = {"return_value": {
            "success": False, "error": "claude exited 1", "data": None,
            "error_code": ERROR_SUBPROCESS_FAILED, "details": self._details(),
        }}
        out = self._call(tmp_path, stub)

        assert out["success"] is False
        assert out["details"]["usage"] == ENVELOPE_USAGE


class TestHardFailureOnAnExitZeroRunWritesItsLog:
    """An adapter may call an exit-0 stream a hard failure. Then the run has
    BOTH a hard error code and a deferred log, which the hard-failure return
    used to drop on the floor — the whole log for that slot, not a line of it.
    """

    class _TerminalProvider(_BaseProvider):
        """The smallest adapter whose exit-0 parse is a hard failure."""

        usage_support = "full"

        @property
        def name(self):
            return "terminalparse"

        @property
        def default_timeout(self):
            return 60

        def is_available(self):
            return True

        def build_command(self, prompt, model, **_kwargs):
            return ["terminalparse-cli", prompt]

        def parse_output(self, stdout, stderr):
            return {
                "success": False,
                "error": "the stream reported a terminal error",
                "error_code": ERROR_SUBPROCESS_FAILED,
                "data": None,
            }

        def parse_usage(self, stdout, stderr):
            return dict(ENVELOPE_USAGE)

    def test_the_log_is_written_and_names_the_failure(self, tmp_path, monkeypatch):
        from utils import provider_registry

        monkeypatch.setitem(
            provider_registry._PROVIDERS, "terminalparse", self._TerminalProvider()
        )
        log_file = tmp_path / "log_terminalparse.txt"

        with patch("shutil.which", return_value="/usr/bin/terminalparse-cli"), \
             patch("utils.llm_client.subprocess.run",
                   return_value=_usage_completed("stream-bytes")):
            out = invoke_with_file_output(
                prompt_template="write to {output_json_path}",
                model_spec="terminalparse:m",
                prompt_context={"prefix": "p"},
                output_dir=tmp_path,
                phase="code_review",
                log_file=log_file,
            )

        assert out["success"] is False
        assert out["error_code"] == ERROR_SUBPROCESS_FAILED
        assert out["details"]["usage"] == ENVELOPE_USAGE
        assert log_file.exists(), "the slot's only log entry was dropped"
        written = log_file.read_text(encoding="utf-8")
        assert "stream-bytes" in written
        assert "the stream reported a terminal error" in written


# --------------------------------------------------------------------------- #
# omp — the production path, with a real fake binary on PATH
#
# These are the three things about omp that a unit test over `build_command` and
# `parse_output` cannot reach, because each of them is a property of the road
# between them: which transport `invoke_with_file_output` actually takes, what a
# pre-spawn refusal does to a slot, and whether a findings file written beside a
# failed parse is read. All three run against a fake `omp` EXECUTABLE rather than
# a patched `subprocess.run`, because the flag gate spawns `--help` itself and a
# patched runner would answer that too.
# --------------------------------------------------------------------------- #

#: Every long option `OmpProvider.build_command` can emit, so the gate passes.
_OMP_FULL_HELP = (
    "Usage: omp [options] [message]\n"
    "  -p, --print              print the answer and exit\n"
    "      --no-session         write no session files\n"
    "      --mode <mode>        output mode: text | json\n"
    "      --config <path>      config overlay\n"
    "      --approval-mode <m>  always-ask | write | yolo\n"
    "      --tools <list>       comma-separated allow-list\n"
    "      --cwd <dir>          working directory\n"
    "      --model <model>      model selector\n"
    "      --thinking <level>   reasoning effort\n"
    "      --max-time <n>       deadline in seconds\n"
)


def _omp_stream(*, stop_text=None, stop_reason="stop", tool_error=None,
                tool_name="write", assistant_content=None):
    """A minimal omp JSONL stream, in the measured event order.

    Only the events `parse_output` reads: `session` (for the version), an
    optional `tool_execution_end`, the terminal assistant `message_end` and
    `agent_end`. The committed fixtures are the real streams; this builds the
    handful of variants these tests need to vary one field at a time.
    """
    events = [{"type": "session", "version": 3, "id": "01a0", "cwd": "/work"}]
    if tool_error is not None:
        events.append({
            "type": "tool_execution_end",
            "toolCallId": "call_1",
            "toolName": tool_name,
            "result": {"content": [{"type": "text", "text": tool_error}], "details": {}},
            "isError": True,
        })
        events.append({
            "type": "tool_execution_end",
            "toolCallId": "call_2",
            "toolName": tool_name,
            "result": {"content": [{"type": "text", "text": "ok"}], "details": {}},
            "isError": False,
        })
    content = assistant_content
    if content is None:
        content = [] if stop_text is None else [{"type": "text", "text": stop_text}]
    events.append({
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": content,
            "stopReason": stop_reason,
            "usage": {"input": 10, "output": 2, "cacheRead": 0, "cacheWrite": 0,
                      "totalTokens": 12,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0,
                               "cacheWrite": 0, "total": 0.01}},
        },
    })
    events.append({"type": "agent_end", "messages": [], "isTerminal": True})
    return "".join(json.dumps(e) + "\n" for e in events)


def _install_fake_omp(tmp_path, monkeypatch, *, script_body, help_text=_OMP_FULL_HELP,
                      version="omp/18.2.2"):
    """Put a fake `omp` on PATH and reset the memoised help probe.

    ``script_body`` is shell run for every invocation that is neither ``--help``
    nor ``--version``; it may read stdin and write to ``$OMP_FAKE_DIR``.
    """
    import os

    from utils.providers import base as providers_base

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "omp"
    fake.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        '  case "$a" in\n'
        f"    --help) cat <<'HELP'\n{help_text}HELP\n"
        "            exit 0 ;;\n"
        f"    --version) echo '{version}'; exit 0 ;;\n"
        "  esac\n"
        "done\n"
        f"{script_body}\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("OMP_FAKE_DIR", str(tmp_path))
    # The gate memoises `read_help_tokens` on (binary, help argv), and the binary
    # path is the same string across tests in one session.
    providers_base.reset_flag_gate_cache()
    return fake


class TestOmpPromptTransport:
    """§2.5 — the prompt reaches omp **once, over stdin**, and none of it on argv.

    The declaration is the whole change, and this is the test that it took:
    `invoke_with_file_output` — the function every Lane A review phase goes
    through — takes no transport argument and forwards none, so an adapter
    default of `argv` would leave every review prompt on the path measured
    failing (200 KB on argv is `E2BIG` from the kernel before omp starts). Only
    the adapter's own `prompt_transport` decides it, and only a run through the
    production entry point can show that it did.
    """

    SENTINEL = "SENTINEL-c0ffee-9f3a"

    def _fake(self, tmp_path, monkeypatch):
        # Reads stdin TO EOF, records the byte count and an `eof` marker, writes
        # a findings array to the --output path the prompt named, and prints a
        # `stop` stream. A stdin that never reached EOF would hang here and the
        # test's own timeout would fail rather than pass quietly.
        body = (
            'printf "%s\\n" "$@" > "$OMP_FAKE_DIR/argv.txt"\n'
            'cat > "$OMP_FAKE_DIR/stdin.txt"\n'
            'printf eof > "$OMP_FAKE_DIR/eof.marker"\n'
            'cat "$OMP_FAKE_DIR/findings.json" > "$OMP_FAKE_DIR/out.json" 2>/dev/null\n'
            'if [ -n "$OMP_FAKE_OUTPUT" ]; then\n'
            '  cp "$OMP_FAKE_DIR/findings.json" "$OMP_FAKE_OUTPUT"\n'
            'fi\n'
            'cat "$OMP_FAKE_DIR/stream.jsonl"\n'
        )
        return _install_fake_omp(tmp_path, monkeypatch, script_body=body)

    def test_a_200kb_prompt_travels_on_stdin_exactly_once(self, tmp_path, monkeypatch):
        self._fake(tmp_path, monkeypatch)
        (tmp_path / "findings.json").write_text(json.dumps([{"title": "t"}]))
        (tmp_path / "stream.jsonl").write_text(_omp_stream(stop_text="[]"))

        # The fake needs the path the caller generated; it is deterministic, so
        # compute it the way `invoke_with_file_output` does and hand it over.
        out_dir = tmp_path / "phase"
        from utils.json_extractor import generate_output_path
        monkeypatch.setenv(
            "OMP_FAKE_OUTPUT",
            generate_output_path(out_dir, "output", "review_plan", "omp:stub/stub-model"),
        )

        padding = "x" * 100_000
        template = padding + "\n" + self.SENTINEL + "\n" + padding + "\n{output_location}"
        result = invoke_with_file_output(
            prompt_template=template,
            model_spec="omp:stub/stub-model",
            prompt_context={},
            output_dir=out_dir,
            phase="review_plan",
            timeout=60,
        )

        assert result["success"] is True, result.get("error")
        assert result["source"] == "file"

        argv = (tmp_path / "argv.txt").read_text().splitlines()
        assert self.SENTINEL not in "\n".join(argv), "the prompt reached argv"
        assert "--" not in argv, "the stdin rendering must drop `--` as well as the prompt"
        assert argv[-2:] == ["--model", "stub/stub-model"], argv

        stdin_text = (tmp_path / "stdin.txt").read_text()
        assert stdin_text.count(self.SENTINEL) == 1, "the prompt arrived twice"
        assert len(stdin_text.encode()) > 200_000
        assert (tmp_path / "eof.marker").exists(), "the child never saw EOF on stdin"

    def test_the_argv_transport_still_renders_the_prompt_last_after_a_dash_dash(
        self, tmp_path, monkeypatch
    ):
        self._fake(tmp_path, monkeypatch)
        (tmp_path / "findings.json").write_text("[]")
        (tmp_path / "stream.jsonl").write_text(_omp_stream(stop_text="[]"))

        result = invoke_with_provider(
            self.SENTINEL, "omp:stub/stub-model", timeout=60,
            prompt_transport="argv",
        )

        assert result["success"] is True, result.get("error")
        argv = (tmp_path / "argv.txt").read_text().splitlines()
        assert argv[-1] == self.SENTINEL
        assert argv[-2] == "--", "`--` is what stops a leading `-` being read as a flag"
        assert (tmp_path / "stdin.txt").read_text() == "", "argv transport wrote stdin"


class TestOmpFlagGateRefusesBeforeSpawning:
    """§2.11 — a build whose help lacks a required flag starts no generation.

    The point of the load-bearing form is that nothing is spawned and nothing is
    retried: the raise escapes `build_command` before an argv exists, propagates
    out of the ladder loop into `run_single_model`'s per-model `except`, and the
    slot ends with that one named error while its siblings run.
    """

    def _fake(self, tmp_path, monkeypatch, *, help_text, version="omp/18.2.2"):
        body = (
            'printf spawned > "$OMP_FAKE_DIR/spawned.marker"\n'
            'cat "$OMP_FAKE_DIR/stream.jsonl" 2>/dev/null\n'
        )
        return _install_fake_omp(
            tmp_path, monkeypatch, script_body=body, help_text=help_text, version=version
        )

    def test_a_help_missing_tools_refuses_and_no_generation_process_starts(
        self, tmp_path, monkeypatch
    ):
        from utils.providers.omp import OmpFlagsUnconfirmed

        self._fake(
            tmp_path, monkeypatch,
            help_text=_OMP_FULL_HELP.replace(
                "      --tools <list>       comma-separated allow-list\n", ""
            ),
            version="omp/17.0.0",
        )
        (tmp_path / "stream.jsonl").write_text(_omp_stream(stop_text="[]"))

        with pytest.raises(OmpFlagsUnconfirmed) as excinfo:
            invoke_with_provider("hello", "omp:stub/stub-model", timeout=60)

        assert excinfo.value.missing == ("--tools",)
        assert "--tools" in str(excinfo.value)
        assert not (tmp_path / "spawned.marker").exists(), (
            "a refused slot spawned a generation process"
        )

    def test_a_build_below_the_verified_version_still_dispatches_on_lane_a(
        self, tmp_path, monkeypatch
    ):
        """The repo's own rule: only BINARY_MISSING refuses a launch.

        A too-old build is a *label* on Lane A — doctor's row, the preflight
        report — and a refusal only in the executor's own version gate. omp adds
        no third gate: what refuses here is the missing flag, by name, not the
        number.
        """
        self._fake(tmp_path, monkeypatch, help_text=_OMP_FULL_HELP, version="omp/0.0.1")
        (tmp_path / "stream.jsonl").write_text(_omp_stream(stop_text="[]"))

        result = invoke_with_provider("hello", "omp:stub/stub-model", timeout=60)

        assert result["success"] is True, result.get("error")
        assert (tmp_path / "spawned.marker").exists()


class TestOmpFileRecoveryHonoursTheTerminalCode:
    """§2.6 — the findings file is read only beside a stream that vouches for it.

    Lane A's review prompts write `{output_json_path}` **before** the model's
    later tool calls, so a run whose findings landed and whose `git diff` was
    then cut by `--max-time` — exit 0, `toolUse`-final — would otherwise be a
    failed parse followed by a recovered file: reported as success, with a
    cut-off review passed downstream as complete. The hard `error_code` is what
    stops `invoke_with_file_output` from looking.
    """

    def _run(self, tmp_path, stream, *, write_file=True, log=True):
        out_dir = tmp_path / "phase"
        out_dir.mkdir(parents=True, exist_ok=True)
        from utils.json_extractor import generate_output_path
        path = Path(generate_output_path(out_dir, "output", "review_plan", "omp:stub/stub-model"))
        path.parent.mkdir(parents=True, exist_ok=True)
        findings = [{"title": "a finding the model really did write"}]
        if write_file:
            path.write_text(json.dumps(findings), encoding="utf-8")
        log_file = tmp_path / "run.log" if log else None

        with patch("utils.llm_client.invoke_with_provider") as mock_invoke:
            from utils.provider_registry import get_provider
            provider = get_provider("omp")
            parsed = provider.parse_output(stream, "")
            # The deferred log is what `invoke_with_provider` hands over on an
            # exit-0 run so the caller can name the channel that answered; the
            # mock stands in for it, because the branch under test is the one
            # that has to settle it on a HARD failure — which, before omp, no
            # exit-0 parse could produce.
            mock_invoke.return_value = {
                **parsed,
                "details": {
                    "provider": "omp",
                    "model": "stub/stub-model",
                    "_deferred_log": {
                        "stdout": stream,
                        "stderr": "",
                        "returncode": 0,
                        "duration_seconds": 0.3,
                    },
                },
            }
            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}",
                model_spec="omp:stub/stub-model",
                prompt_context={},
                output_dir=out_dir,
                phase="review_plan",
                log_file=log_file,
            )
        return result, path, findings

    def test_a_toolUse_final_stream_refuses_a_valid_file(self, tmp_path):
        stream = _omp_stream(stop_reason="toolUse", tool_error="Command aborted",
                             tool_name="bash")
        result, path, findings = self._run(tmp_path, stream)

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert "run ended inside a tool call" in result["error"]
        assert "bash" in result["error"]
        # The tool's result is content, not a verdict: never in the error.
        assert "Command aborted" not in result["error"]
        assert result.get("data") in (None, {}, [])
        assert result["output_file"] == str(path)
        assert json.loads(path.read_text()) == findings, "the file was rewritten"

    def test_the_same_file_beside_a_clean_stop_is_still_recovered(self, tmp_path):
        result, path, findings = self._run(tmp_path, _omp_stream(stop_text="not json"))

        assert result["success"] is True
        assert result["source"] == "file"
        assert result["data"] == findings

    def test_an_approval_denial_refuses_the_file_even_under_a_clean_stop(self, tmp_path):
        stream = _omp_stream(
            stop_text=json.dumps([{"title": "fabricated"}]),
            tool_error="Tool bash requires approval but no interactive UI available",
            tool_name="bash",
        )
        result, path, findings = self._run(tmp_path, stream)

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "was denied approval" in result["error"]
        # The denial notice is tool-channel text: never in the error.
        assert "requires approval" not in result["error"]
        assert json.loads(path.read_text()) == findings

    def test_an_ordinary_tool_error_the_model_recovered_from_is_not_a_denial(self, tmp_path):
        stream = _omp_stream(
            stop_text=json.dumps([{"title": "a real review"}]),
            tool_error="fatal: bad revision 'HEAD~99'",
            tool_name="bash",
        )
        result, path, findings = self._run(tmp_path, stream)

        assert result["success"] is True
        assert result["source"] == "file"
        assert "bad revision" not in (result.get("error") or "")

    def test_an_unrecognized_stop_reason_refuses_the_file(self, tmp_path):
        stream = _omp_stream(stop_text=json.dumps([{"title": "x"}]), stop_reason="maxTokens")
        result, path, findings = self._run(tmp_path, stream)

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "unrecognized stopReason 'maxTokens'" in result["error"]
        assert json.loads(path.read_text()) == findings

    def test_the_run_log_is_written_for_a_hard_exit_zero_failure(self, tmp_path):
        """The comment at the hard-failure return said `deferred_log` is always
        None there, because a hard failure never reached the exit-0 branch that
        sets it. omp is the first adapter for which that is false."""
        stream = _omp_stream(stop_reason="toolUse", tool_error="Command aborted",
                             tool_name="bash")
        result, _path, _findings = self._run(tmp_path, stream)
        log_file = tmp_path / "run.log"

        assert log_file.exists(), "the slot's only log entry was dropped"
        written = log_file.read_text(encoding="utf-8")
        assert "Success: False" in written
        assert "run ended inside a tool call" in written


# =========================================================================== #
# D3.3 (review-run-scope-salvage-and-failure-visibility T3.3): a stopped Lane A
# CLI takes its detached children with it.
#
# The shape of `tests/test_subagent_executor.py`'s "a timed-out dispatch never
# hangs on a surviving grandchild", on Lane A, and with pi's harder shape: the
# grandchildren are in a SESSION of their own (pi's bash tool spawns its shell
# `detached: true`) and `timeout` takes a group of its own again inside it, so no
# signal to the CLI or to its group reaches them. Real processes throughout —
# "is it still running?" is the one question a mock cannot answer.
# =========================================================================== #

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sessions and groups")
class TestAStoppedLaneACliTakesItsTreeWithIt:
    MODEL_SPEC = "pi:stub/stub-model"

    @pytest.fixture
    def fake_pi(self, tmp_path, monkeypatch):
        """The fake CLI as `pi` on PATH, its marker, and a cleanup net."""
        import os
        import shutil

        from tests.harness import cli_tree
        from utils.providers import base as providers_base

        if shutil.which("timeout") is None:
            pytest.skip("GNU `timeout` (its own process group) is the shape under test")

        bin_dir = tmp_path / "bin"
        cli_tree.install_fake_cli(bin_dir, "pi")
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        marker = tmp_path / "tree.pids"
        monkeypatch.setenv("CLI_TREE_MARKER", str(marker))
        # The `--tools` gate memoises one help read per binary path.
        providers_base.reset_flag_gate_cache()
        try:
            yield marker
        finally:
            cli_tree.reap_leftovers(marker)
            providers_base.reset_flag_gate_cache()

    def _everything(self, marker):
        from tests.harness import cli_tree

        pids = cli_tree.tree_pids(marker)
        assert len(pids) == 4, f"the fake recorded {pids}, not sh/timeout/sleep/cat"
        cli = cli_tree.cli_pid(marker)
        assert cli is not None
        return pids, cli

    def test_a_timed_out_slot_leaves_no_descendant_and_keeps_its_partial_stdout(
        self, fake_pi, tmp_path, monkeypatch
    ):
        from tests.harness import cli_tree

        # `cat` holds the CLI's stdout pipe too, so the drain is exercised.
        monkeypatch.setenv("CLI_TREE_SHELL_STDOUT", "inherit")
        log_file = tmp_path / "run.log"
        result = invoke_with_provider(
            "Review this", self.MODEL_SPEC, timeout=1, log_file=log_file,
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        # The partial stdout §4's salvage reads survives the new kill path.
        assert cli_tree.PARTIAL_LINE in result["details"]["stdout_excerpt"]
        pids, cli = self._everything(fake_pi)
        still = cli_tree.await_dead(pids + [cli])
        assert not still, f"outlived the slot's kill: {still}"
        # And the log names what the sweep did, group by group.
        written = log_file.read_text(encoding="utf-8")
        assert "timed out after 1s; cli terminated; process groups:" in written
        assert "SURVIVED" not in written
        # Two groups, both swept: the detached shell's (sh + cat, led by sh) and
        # the one `timeout` made for itself (timeout + sleep).
        roles = cli_tree.tree_roles(fake_pi)
        assert f"{roles['sh']} killed" in written
        assert f"{roles['timeout']} killed" in written

    def test_the_call_returns_within_the_grace_plus_the_bounded_drain(
        self, fake_pi, tmp_path, monkeypatch
    ):
        """A CLI that ignores SIGTERM costs the grace and the SIGKILL, no more.

        `cat` inherited the CLI's stdout pipe, so an unbounded drain would wait
        on it for its whole minute; the sweep ends it and the drain is capped.
        """
        from tests.harness import cli_tree

        monkeypatch.setenv("CLI_TREE_ON_TERM", "ignore")
        monkeypatch.setenv("CLI_TREE_SHELL_STDOUT", "inherit")
        log_file = tmp_path / "run.log"
        started = time.monotonic()
        result = invoke_with_provider(
            "Review this", self.MODEL_SPEC, timeout=1, log_file=log_file,
        )
        elapsed = time.monotonic() - started

        assert result["error_code"] == ERROR_TIMEOUT
        budget = (1 + llm_client.LANE_A_TERMINATE_GRACE_S + llm_client.LANE_A_DRAIN_S)
        assert elapsed < budget, f"{elapsed:.1f}s against a {budget:.1f}s bound"
        assert "cli killed" in log_file.read_text(encoding="utf-8")
        pids, cli = self._everything(fake_pi)
        assert not cli_tree.await_dead(pids + [cli])

    def test_a_cli_that_exits_on_sigterm_is_never_sigkilled(
        self, fake_pi, tmp_path, monkeypatch
    ):
        """SIGTERM first is the point: pi reaps its own shells from that handler."""
        from pathlib import Path as _Path

        from tests.harness import cli_tree

        monkeypatch.setenv("CLI_TREE_ON_TERM", "exit")
        log_file = tmp_path / "run.log"
        result = invoke_with_provider(
            "Review this", self.MODEL_SPEC, timeout=1, log_file=log_file,
        )

        assert result["error_code"] == ERROR_TIMEOUT
        assert _Path(str(fake_pi) + ".term").read_text() == "SIGTERM"
        written = log_file.read_text(encoding="utf-8")
        assert "cli terminated" in written
        assert "cli killed" not in written
        pids, cli = self._everything(fake_pi)
        assert not cli_tree.await_dead(pids + [cli])

    def test_ctrl_c_sweeps_the_tree_and_still_ends_the_run(self, fake_pi):
        """Decision 11.9: the interrupt propagates, and nothing is left behind."""
        import os
        import signal
        import threading

        from tests.harness import cli_tree

        def interrupt_when_ready():
            if cli_tree.await_ready(fake_pi):
                # A moment for the parent to be back inside communicate().
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGINT)

        # The interrupt must arrive as Python's own KeyboardInterrupt, whatever
        # an earlier test installed for SIGINT (a detached worker's abort
        # handler answers it with `os._exit(3)`, which would end pytest itself).
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            threading.Thread(target=interrupt_when_ready, daemon=True).start()
            with pytest.raises(KeyboardInterrupt):
                invoke_with_provider("Review this", self.MODEL_SPEC, timeout=60)
        finally:
            signal.signal(signal.SIGINT, previous)

        pids, cli = self._everything(fake_pi)
        still = cli_tree.await_dead(pids + [cli])
        assert not still, f"outlived the Ctrl-C: {still}"

    def test_a_cli_killed_from_outside_still_has_its_tree_swept(
        self, fake_pi, monkeypatch
    ):
        """A fan-out slot runs on a worker thread, where no KeyboardInterrupt is
        ever raised: a terminal Ctrl-C (or an abort) kills the CLI directly, and
        the call simply returns. Its detached tree is swept from the registry's
        last snapshot, the only thing that still links it to the slot.

        pi's shape (the shell read through pi's own pipe), because a child that
        holds the CALLER's pipe keeps ``communicate()`` waiting until the
        slot's deadline, where the timeout path sweeps it instead.
        """
        from tests.harness import cli_tree

        monkeypatch.setenv("CLI_TREE_SELF_SIGNAL", "SIGTERM")
        monkeypatch.setenv(
            "CLI_TREE_SELF_SIGNAL_AFTER_S",
            str(process_control_interval() * 2 + 0.5),
        )
        result = invoke_with_provider("Review this", self.MODEL_SPEC, timeout=60)

        assert result["success"] is False
        assert result.get("error_code") != ERROR_TIMEOUT
        pids, cli = self._everything(fake_pi)
        still = cli_tree.await_dead(pids + [cli])
        assert not still, f"outlived their CLI: {still}"

    def test_a_cli_whose_own_handler_exits_143_still_has_its_tree_swept(
        self, fake_pi, monkeypatch
    ):
        """pi's real shape of the outside kill: its SIGTERM handler reaps its
        shell and exits 143, so it never *dies* of the signal — and the
        ``timeout`` group inside that shell survives the handler. The 128+N exit
        is the same stop as the ``-N`` death and gets the same sweep.
        """
        from pathlib import Path as _Path

        from tests.harness import cli_tree

        monkeypatch.setenv("CLI_TREE_ON_TERM", "reap-shell")
        monkeypatch.setenv("CLI_TREE_SELF_SIGNAL", "SIGTERM")
        monkeypatch.setenv(
            "CLI_TREE_SELF_SIGNAL_AFTER_S",
            str(process_control_interval() * 2 + 0.5),
        )
        completed = llm_client._run_provider_cli(
            ["pi"], capture_output=True, text=True, timeout=60,
        )

        assert completed.returncode == 143
        assert _Path(str(fake_pi) + ".term").read_text() == "reap-shell"
        pids, cli = self._everything(fake_pi)
        still = cli_tree.await_dead(pids + [cli])
        assert not still, f"outlived their CLI's own handler: {still}"

    def test_the_registry_is_empty_after_every_return(self, fake_pi):
        from utils import process_control

        completed = llm_client._run_provider_cli(
            [sys.executable, "-c", "print('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        assert completed.returncode == 0 and completed.stdout == "ok\n"
        assert len(process_control.live_cli_registry) == 0

        with pytest.raises(subprocess.TimeoutExpired) as caught:
            llm_client._run_provider_cli(
                ["pi"], capture_output=True, text=True, timeout=1,
            )
        assert len(process_control.live_cli_registry) == 0
        assert caught.value.tree_sweep.cli == "terminated"
        assert set(caught.value.tree_sweep.groups.values()) == {"killed"}

    def test_stdin_input_is_still_written_and_closed(self):
        """`input=` exactly as `subprocess.run` took it: written whole, then EOF."""
        text = "x" * 300_000 + "\nEND"
        completed = llm_client._run_provider_cli(
            [sys.executable, "-c",
             "import sys; data = sys.stdin.read(); print(len(data), data[-3:])"],
            capture_output=True, text=True, timeout=30, input=text,
        )
        assert completed.stdout.split() == [str(len(text)), "END"]


def process_control_interval() -> float:
    from utils import process_control

    return process_control.live_cli_registry.interval_s
