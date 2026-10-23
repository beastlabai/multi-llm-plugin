"""What a failed review slot says about itself, pinned to the 2026-09-17 run.

A real ``review-code --quick`` fan-out lost two of five slots and reported both
as a bare ``<provider> exited with code 1``:

* ``claude-code:claude-fable-5-1:high`` — 491 s, four subagents, $25.05 and
  137 348 output tokens, then ``api_error_status: 429`` with *"You've reached
  your Fable limit…"* in ``result``. The orchestrator printed
  ``[SALVAGE_NEEDED]`` over that envelope (``tests/test_json_extractor.py``
  pins the gate), and nothing the run wrote named the status, the sentence or
  a number.
* ``cursor-agent:kimi-k3-high`` — 23.7 s, stdout empty, stderr ending in
  ``RetriableError: [resource_exhausted] Error`` after three reconnects. No
  marker table read it, so the classifier said ``None``.

Each class below is one seam of that report: the envelope renderer both
adapters share, each adapter's ``describe_failure``, the shared rate-limit
table, claude-code's usage extractor on a failed final turn, the fan-out's
usage attach, and the review-code lane's error string end to end.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro
from tests.test_json_extractor import USAGE_CAP_ENVELOPE
from utils import llm_client
from utils.provider_registry import get_provider
from utils.providers.base import STATUS_RATE_LIMITED, classify_streams
from utils.providers.claude_code import USAGE_LIMIT_TAG
from utils.providers.result_envelope import describe_refused_envelope

FIXTURES = Path(__file__).parent / "fixtures" / "claude-code"

#: cursor-agent's stderr on the kimi slot, verbatim but for nothing: no id, no
#: credential, no path — the host name is Cursor's public agent endpoint.
KIMI_STDERR = (
    "Connection lost, reconnecting to https://agentn.global.api5.cursor.sh (attempt 1)...\n"
    "Retry attempt 1...\n"
    "Connection lost, reconnecting to https://agentn.global.api5.cursor.sh (attempt 2)...\n"
    "Retry attempt 2...\n"
    "Connection lost, reconnecting to https://agentn.global.api5.cursor.sh (attempt 3)...\n"
    "Retry attempt 3...\n"
    "RetriableError: [resource_exhausted] Error\n"
)

KIMI_REASON = "RetriableError: [resource_exhausted] Error"
FABLE_SENTENCE = (
    "You've reached your Fable limit. Switch to another model, or manage usage "
    "credits at claude.ai/settings/usage?from=cc_cli_limit_message, to continue."
)


def _capture(row: str) -> str:
    return (FIXTURES / f"{row}.stdout.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The one renderer of a refusing envelope's line
# ---------------------------------------------------------------------------


class TestDescribeRefusedEnvelope:
    def test_the_usage_cap_carries_its_status_and_its_sentence(self):
        assert describe_refused_envelope(USAGE_CAP_ENVELOPE) == f"HTTP 429: {FABLE_SENTENCE}"

    def test_the_committed_rate_limit_row_reads_the_same_way(self):
        line = describe_refused_envelope(_capture("rate-limit"))
        assert line.startswith("HTTP 429: API Error: Request rejected (429)")

    @pytest.mark.parametrize("row", ["truncated", "zero-byte-response", "error-envelope"])
    def test_a_refusal_with_no_http_status_is_the_sentence_alone(self, row):
        """``api_error_status`` is ``null`` where no response arrived."""
        line = describe_refused_envelope(_capture(row))
        assert line and not line.startswith("HTTP")
        assert line.startswith("API Error:")

    @pytest.mark.parametrize("row", ["success", "ephemeral-only", "multi-assistant"])
    def test_a_served_envelope_has_nothing_to_describe(self, row):
        assert describe_refused_envelope(_capture(row)) is None

    @pytest.mark.parametrize("stdout", ["", "not json", "[1, 2]", '{"type":"result"}'])
    def test_anything_that_is_not_a_refusing_envelope_is_none(self, stdout):
        assert describe_refused_envelope(stdout) is None

    def test_a_status_without_a_sentence_still_names_the_status(self):
        assert describe_refused_envelope(
            '{"type":"result","is_error":true,"api_error_status":503}'
        ) == "HTTP 503"


# ---------------------------------------------------------------------------
# claude-code: the sentence reaches the error string, tagged for a usage cap
# ---------------------------------------------------------------------------


class TestClaudeCodeDescribeFailure:
    @pytest.fixture
    def provider(self):
        return get_provider("claude-code")

    def test_the_usage_cap_is_tagged(self, provider):
        assert provider.describe_failure(USAGE_CAP_ENVELOPE, "") == (
            f"{USAGE_LIMIT_TAG} HTTP 429: {FABLE_SENTENCE}"
        )

    def test_a_transient_429_is_not(self, provider):
        """Same status, a busy endpoint: the tag would send a human to buy credits."""
        line = provider.describe_failure(_capture("rate-limit"), "")
        assert line.startswith("HTTP 429:")
        assert USAGE_LIMIT_TAG not in line

    @pytest.mark.parametrize(
        "row,prefix",
        [
            ("auth-error", "HTTP 401: Failed to authenticate."),
            ("invalid-model", "HTTP 404: There's an issue with the selected model"),
            ("context-overflow", "HTTP 400: API Error: 400"),
            ("stdout-error", "HTTP 500: API Error: 500"),
        ],
    )
    def test_every_measured_failure_names_itself(self, provider, row, prefix):
        assert provider.describe_failure(_capture(row), "").startswith(prefix)

    def test_a_served_run_has_nothing_to_say(self, provider):
        assert provider.describe_failure(_capture("success"), "") is None

    def test_the_verdict_does_not_move(self, provider):
        """Diagnostics, never classification: still the status rung's answer."""
        assert provider.classify_failure(USAGE_CAP_ENVELOPE, "", 1) == STATUS_RATE_LIMITED

    def test_the_ladder_lane_reads_the_composed_error_string(self, provider):
        """`run_single_model` classifies ``invoke_with_provider``'s error text.

        With the sentence and status on it, the text ladder lands on the same
        rung the envelope's status rung does; before, the bare exit code
        classified as nothing and the slot was "never retried" for no reason
        anyone could read.
        """
        composed = f"claude-code exited with code 1: {provider.describe_failure(USAGE_CAP_ENVELOPE, '')}"
        assert classify_streams(composed, "") == STATUS_RATE_LIMITED


# ---------------------------------------------------------------------------
# cursor-agent: stderr's last line is the reason, and it classifies
# ---------------------------------------------------------------------------


class TestCursorAgentDescribeFailure:
    @pytest.fixture
    def provider(self):
        return get_provider("cursor-agent")

    def test_the_kimi_slot_names_its_last_stderr_line(self, provider):
        assert provider.describe_failure("", KIMI_STDERR) == KIMI_REASON

    def test_the_reconnect_progress_lines_are_not_the_reason(self, provider):
        assert "Connection lost" not in provider.describe_failure("", KIMI_STDERR)

    def test_a_refusing_envelope_on_stdout_wins_over_stderr(self, provider):
        stdout = json.dumps({"type": "result", "is_error": True, "result": "Model refused"})
        assert provider.describe_failure(stdout, KIMI_STDERR) == "Model refused"

    def test_nothing_on_either_stream_is_none(self, provider):
        assert provider.describe_failure("", "") is None
        assert provider.describe_failure("", "\n  \n") is None

    def test_a_long_last_line_is_capped(self, provider):
        line = provider.describe_failure("", "x" * 1000)
        assert len(line) == 400

    def test_resource_exhausted_is_a_rate_limit(self, provider):
        assert provider.classify_failure("", KIMI_STDERR, 1) == STATUS_RATE_LIMITED

    def test_the_ladder_lane_reads_the_composed_error_string(self, provider):
        composed = f"cursor-agent exited with code 1: {provider.describe_failure('', KIMI_STDERR)}"
        assert classify_streams(composed, "") == STATUS_RATE_LIMITED


class TestResourceExhaustedInTheSharedTable:
    """gRPC/Connect ``RESOURCE_EXHAUSTED`` is a quota, whichever CLI relays it."""

    @pytest.mark.parametrize(
        "stream",
        [
            KIMI_REASON,
            "rpc error: code = ResourceExhausted desc = quota exceeded",
            "status: RESOURCE_EXHAUSTED",
            "resource exhausted, retry later",
        ],
    )
    def test_each_spelling_classifies(self, stream):
        assert classify_streams("", stream) == STATUS_RATE_LIMITED

    def test_a_bare_exit_code_still_classifies_as_nothing(self):
        """The marker is in the table; it is `describe_failure` that gets it
        onto the string the ladder lane reads."""
        assert classify_streams("cursor-agent exited with code 1", "") is None


# ---------------------------------------------------------------------------
# claude-code usage on a failed final turn
# ---------------------------------------------------------------------------


class TestClaudeCodeUsageOnAFailedTurn:
    @pytest.fixture
    def provider(self):
        return get_provider("claude-code")

    def test_a_zeroed_final_turn_defers_to_the_per_model_totals(self, provider):
        usage = provider.parse_usage(USAGE_CAP_ENVELOPE, "")
        assert usage["output_tokens"] == 137348 + 13
        assert usage["input_tokens"] == 1936 + 1944
        assert usage["cached_input_tokens"] == 37205114
        assert usage["cost_usd"] == pytest.approx(25.05, abs=0.01)
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        assert usage["served_model"] == "claude-fable-5-1"

    def test_a_turn_that_reports_tokens_keeps_the_block_the_tests_pin(self, provider):
        """The committed success rows: 1/1 on the turn beside 2/2 in modelUsage."""
        usage = provider.parse_usage(_capture("success"), "")
        assert usage["input_tokens"] == 1 and usage["output_tokens"] == 1

    def test_the_committed_rate_limit_row_has_no_totals_to_defer_to(self, provider):
        """``modelUsage: {}`` on the stub capture: the zeros stay, honestly."""
        usage = provider.parse_usage(_capture("rate-limit"), "")
        assert usage["output_tokens"] == 0 and usage["cost_usd"] == 0


# ---------------------------------------------------------------------------
# The fan-out attaches usage on the losing path, and renders it
# ---------------------------------------------------------------------------


class TestTheFanoutCarriesWhatAFailedSlotSpent:
    def _invoke(self, stdout: str, returncode: int = 1):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

        # The availability gate runs ahead of the spawn and reads PATH, so it is
        # stubbed with the spawn: on a machine without the claude CLI (CI) the
        # slot otherwise ends "CLI not found in PATH" before the envelope is read.
        provider_cls = type(get_provider("claude-code"))
        with mock.patch.object(llm_client.subprocess, "run", fake_run), \
                mock.patch.object(provider_cls, "is_available", lambda self: True), \
                mock.patch.object(llm_client, "_resolve_executable", lambda name: ([name], True)):
            return llm_client.invoke_with_provider(
                "prompt", "claude-code:claude-fable-5-1:high", timeout=5
            )

    def test_the_non_zero_exit_carries_usage_and_the_reason(self):
        result = self._invoke(USAGE_CAP_ENVELOPE)
        assert result["success"] is False
        assert result["error"] == (
            f"claude-code exited with code 1: {USAGE_LIMIT_TAG} HTTP 429: {FABLE_SENTENCE}"
        )
        assert result["details"]["usage"]["output_tokens"] == 137361
        assert result["details"]["usage"]["cost_usd"] == pytest.approx(25.05, abs=0.01)

    def test_a_run_that_reports_no_usage_leaves_the_key_absent(self):
        result = self._invoke("API Error: Connection dropped")
        assert "usage" not in result["details"]

    def test_the_reason_goes_through_the_redactor(self):
        stdout = json.dumps({
            "type": "result", "is_error": True, "api_error_status": 401,
            "result": "Failed to authenticate. api_key: sk-proj-" + "a1b2c3d4" * 8,
        })
        result = self._invoke(stdout)
        assert "sk-proj-" not in result["error"]
        assert "<redacted:" in result["error"]

    @pytest.mark.parametrize(
        "usage,expected",
        [
            (None, ""),
            ({}, ""),
            ({"cost_usd": 25.053, "output_tokens": 137361}, "$25.05 est., 137,361 output tokens"),
            ({"cost_usd": 0.001, "output_tokens": 12}, "12 output tokens"),
            ({"cost_usd": 3.0}, "$3.00 est."),
            ({"output_tokens": 0, "input_tokens": 500}, ""),
        ],
    )
    def test_describe_usage_renders_the_two_numbers_a_human_acts_on(self, usage, expected):
        assert llm_client.describe_usage(usage) == expected


# ---------------------------------------------------------------------------
# review-code, end to end: no salvage, and the error string says it all
# ---------------------------------------------------------------------------


class TestTheReviewCodeLaneOnTheUsageCap:
    def _failure(self):
        """What ``invoke_with_file_output`` hands back for the real slot."""
        details = {
            "exit_code": 1, "stderr": "", "provider": "claude-code",
            "model": "claude-fable-5-1:high", "duration_seconds": 491.1,
        }
        llm_client._attach_stdout_excerpt(details, USAGE_CAP_ENVELOPE)
        llm_client._attach_usage(details, get_provider("claude-code"), USAGE_CAP_ENVELOPE, "")
        return {
            "success": False,
            "error": f"claude-code exited with code 1: {USAGE_LIMIT_TAG} HTTP 429: {FABLE_SENTENCE}",
            "error_code": llm_client.ERROR_SUBPROCESS_FAILED,
            "details": details,
            "output_file": "",
        }

    def test_no_salvage_is_requested_and_the_error_names_the_cost(
        self, tmp_path, monkeypatch, capsys
    ):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        monkeypatch.setattr(cro, "get_project_root", lambda _p: str(tmp_path))
        monkeypatch.setattr(cro, "invoke_with_file_output", lambda **kwargs: self._failure())

        success, issues, error = asyncio.run(cro.run_single_review(
            "claude-code:claude-fable-5-1:high", plan_path, tmp_path,
        ))

        assert success is False and issues is None
        assert error.startswith("claude-code exited with code 1: [USAGE_LIMIT] HTTP 429:")
        assert FABLE_SENTENCE in error
        assert error.endswith("(spent $25.05 est., 137,361 output tokens)")

        out = capsys.readouterr().out
        assert "[SALVAGE_NEEDED]" not in out
        assert "nothing to salvage, model recorded as failed" in out
        phase_dir = tmp_path / "plan" / "code-review"
        assert not list(phase_dir.glob("salvage_*.json"))
