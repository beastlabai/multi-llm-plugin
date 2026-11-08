"""Tests for the OpenCode adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import LLMProvider
from utils.retry_ladder import STATUS_AUTH_FAILED, STATUS_RATE_LIMITED
from utils.providers.goose import GooseProvider
from utils.providers.opencode import OpenCodeProvider
from tests import opencode_fixtures as oc_pin
from tests.provider_test_support import (
    OC_429_EVENT,
    OC_429_STDERR,
    OC_GRADED_CAPTURES,
    _oc_event,
    _oc_stream,
)


class TestOpenCodeProvider:
    """Tests for the OpenCodeProvider class."""

    @pytest.fixture
    def provider(self):
        """Create an OpenCodeProvider instance."""
        return OpenCodeProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "opencode"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_opencode_parse_ndjson_events(self, provider):
        """Parse NDJSON event stream with text events.

        OpenCode outputs newline-delimited JSON events.
        Text content is in events with type="text" and part.type="text".
        """
        json_data = [{"item": 1}, {"item": 2}]
        events = [
            {"type": "step_start", "step": "analyze"},
            {"type": "text", "part": {"type": "text", "text": json.dumps(json_data)}},
            {"type": "step_finish", "step": "analyze"}
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == json_data

    def test_opencode_parse_ndjson_multiple_text_events(self, provider):
        """Parse NDJSON with multiple text events that concatenate."""
        events = [
            {"type": "text", "part": {"type": "text", "text": "[{\"id\":1}"}},
            {"type": "text", "part": {"type": "text", "text": ",{\"id\":2}]"}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"id": 1}, {"id": 2}]

    def test_opencode_parse_ndjson_with_code_block(self, provider):
        """Parse NDJSON where text contains JSON in code block."""
        json_data = {"status": "success", "count": 5}
        text_content = f"""Here's the result:

```json
{json.dumps(json_data)}
```
"""
        events = [
            {"type": "text", "part": {"type": "text", "text": text_content}}
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == json_data

    def test_opencode_parse_ndjson_no_text_events(self, provider):
        """Return error when no text events found in NDJSON stream."""
        events = [
            {"type": "step_start", "step": "init"},
            {"type": "tool_use", "tool": "read_file"},
            {"type": "step_finish", "step": "init"}
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "No text events found" in result["error"]
        assert "raw" in result

    def test_opencode_parse_ndjson_empty_text(self, provider):
        """Handle text events with empty content."""
        events = [
            {"type": "text", "part": {"type": "text", "text": ""}},
            {"type": "text", "part": {"type": "text", "text": "   "}}
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "Empty text response" in result["error"]

    def test_opencode_parse_ndjson_malformed_lines(self, provider):
        """Skip malformed JSON lines gracefully."""
        events = [
            {"type": "text", "part": {"type": "text", "text": "[1, 2, 3]"}},
        ]
        # Mix valid and invalid lines
        stdout = "not json\n" + json.dumps(events[0]) + "\nalso not json"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_opencode_parse_ndjson_no_json_in_text(self, provider):
        """Handle text events that don't contain JSON."""
        events = [
            {"type": "text", "part": {"type": "text", "text": "Just plain text"}}
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "No valid JSON found" in result["error"]

    def test_opencode_parse_empty_stdout(self, provider):
        """Handle empty stdout."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert "No text events found" in result["error"]

    def test_opencode_build_command(self, provider):
        """Test command building for OpenCode CLI."""
        prompt = "Generate code"
        model = "claude-3"

        cmd = provider.build_command(prompt, model)

        assert cmd == [
            "opencode", "run", "--format", "json", "--print-logs", "--log-level", "ERROR",
            "--model", "claude-3", "Generate code",
        ]

    def test_opencode_build_command_requests_error_logs(self, provider):
        """Logging is on and pinned to ERROR, for every model shape.

        Without --print-logs a stream error (an exhausted key, say) makes
        opencode retry internally while writing to neither stream: the run
        burns the full timeout and reports no cause. ERROR keeps a healthy
        run silent, and the logs go to stderr so stdout stays pure NDJSON.
        """
        for model in ("openai/gpt-5.5", "openai/gpt-5.5:high"):
            cmd = provider.build_command("Generate code", model)

            assert "--print-logs" in cmd, f"{model!r} built a command with logging off"
            assert cmd[cmd.index("--log-level") + 1] == "ERROR"
            # The prompt must stay the final element regardless.
            assert cmd[-1] == "Generate code"

    def test_opencode_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the seven valid efforts."""
        from utils.providers.opencode import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

    def test_opencode_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --variant pair after --model <base>, prompt last."""
        cmd = provider.build_command("Generate code", "openai/gpt-5.5:high")

        assert cmd == [
            "opencode",
            "run",
            "--format",
            "json",
            "--print-logs",
            "--log-level",
            "ERROR",
            "--model",
            "openai/gpt-5.5",
            "--variant",
            "high",
            "Generate code",
        ]

    def test_opencode_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --variant form."""
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            cmd = provider.build_command("Generate code", f"openai/gpt-5.5:{effort}")

            assert cmd == [
                "opencode",
                "run",
                "--format",
                "json",
                "--print-logs",
                "--log-level",
                "ERROR",
                "--model",
                "openai/gpt-5.5",
                "--variant",
                effort,
                "Generate code",
            ], f"effort {effort!r} did not build the expected command"

    def test_opencode_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --variant flag."""
        for model in ("openai/gpt-5.5", "opencode/big-pickle"):
            cmd = provider.build_command("Generate code", model)

            assert cmd == [
                "opencode",
                "run",
                "--format",
                "json",
                "--print-logs",
                "--log-level",
                "ERROR",
                "--model",
                model,
                "Generate code",
            ], f"model {model!r} was not passed through verbatim"

    def test_the_effort_suffix_of_a_capped_model_never_reaches_variant(
        self, provider, capsys
    ):
        """The one model measured to HANG on `--variant` is stripped, not dispatched.

        A capped model does not merely ignore the flag — it writes zero bytes and
        burns the whole slot timeout — so the suffix has to die before argv, on
        whichever route it arrived: this is the same code path a user-typed
        `--models opencode:<model>:high` and the host session's propagated picker
        variant both go through.
        """
        from utils.providers import opencode as oc

        oc.reset_hermetic_state()
        capped = "openrouter/deepseek/deepseek-v4-flash-0731"

        cmd = provider.build_command("Generate code", f"{capped}:low")

        assert "--variant" not in cmd
        assert cmd == [
            "opencode", "run", "--format", "json",
            "--print-logs", "--log-level", "ERROR",
            "--model", capped, "Generate code",
        ]
        assert capped in capsys.readouterr().err

    def test_the_dropped_variant_is_announced_once_per_model_and_effort(
        self, provider, capsys
    ):
        """A silently different effort is the defect this propagation exists to fix."""
        from utils.providers import opencode as oc

        oc.reset_hermetic_state()
        capped = "openrouter/deepseek/deepseek-v4-flash-0731"

        for _ in range(3):
            provider.build_command("Generate code", f"{capped}:low")
        provider.build_command("Generate code", f"{capped}:high")

        lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
        assert len(lines) == 2, lines
        assert "--variant low" in lines[0] and capped in lines[0]
        assert "--variant high" in lines[1]

    def test_an_uncapped_model_keeps_the_variant_it_was_given(self, provider, capsys):
        """The cap is a fact about one model; nothing else loses its effort."""
        from utils.providers import opencode as oc

        oc.reset_hermetic_state()

        cmd = provider.build_command(
            "Generate code", "openrouter/~deepseek/deepseek-flash-latest:low"
        )

        assert cmd[cmd.index("--model") + 1] == (
            "openrouter/~deepseek/deepseek-flash-latest"
        )
        assert cmd[cmd.index("--variant") + 1] == "low"
        assert capsys.readouterr().err == ""

    @patch("shutil.which")
    def test_opencode_is_available_true(self, mock_which, provider):
        """Test is_available returns True when opencode is found."""
        mock_which.return_value = "/home/user/.local/bin/opencode"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("opencode")

    @patch("shutil.which")
    def test_opencode_is_available_false(self, mock_which, provider):
        """Test is_available returns False when opencode is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("opencode")


def _oc_text(text: str, part_id: str = "prt_1"):
    """A completed assistant text part — the shape carrying the reply."""
    return _oc_event("text", part={
        "id": part_id,
        "sessionID": "ses_TEST",
        "messageID": "msg_TEST",
        "type": "text",
        "text": text,
        "time": {"start": 1769172934123, "end": 1769172934123},
    })


class TestOpenCodeRealSchema:
    """Tests pinned to the schema `opencode run --format json` really emits.

    The pre-existing tests used hand-rolled events shaped to match the parser
    — the same blind spot that let issue #2 (codex) reach production. These
    use the full envelope (`timestamp`, `sessionID`, `part.time`) taken from
    the CLI's own emitter and from recorded sessions.
    """

    @pytest.fixture
    def provider(self):
        return OpenCodeProvider()

    def test_full_envelope_stream_parses(self, provider):
        """The real event shape, envelope fields and all, still parses."""
        stdout = "\n".join([
            _oc_event("step_start", part={"id": "prt_0", "type": "step-start"}),
            _oc_text('[{"task": "ship", "verdict": "ready"}]'),
            _oc_event("step_finish", part={"id": "prt_2", "type": "step-finish"}),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship", "verdict": "ready"}]

    def test_last_message_wins_over_earlier_draft(self, provider):
        """A draft quoted mid-run must not shadow the final answer.

        OpenCode emits one assistant message per step; recorded sessions reach
        47. Extracting from the concatenation returns the *first* match, so a
        narrated draft would be returned instead of the real answer.
        """
        stdout = "\n".join([
            _oc_text('Here is a first pass:\n\n```json\n[{"draft": true}]\n```\n', "prt_1"),
            _oc_text("Checking the files now...", "prt_2"),
            _oc_text('[{"draft": false, "final": true}]', "prt_3"),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"draft": False, "final": True}]

    def test_separate_messages_are_not_run_together(self, provider):
        """Consecutive messages are separated, not glued end-to-start."""
        stdout = "\n".join([
            _oc_text("No fenced JSON here", "prt_1"),
            _oc_text('trailing prose {"answer": 1} more prose', "prt_2"),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == {"answer": 1}

    def test_reasoning_events_do_not_leak(self, provider):
        """`reasoning` events (only with --thinking) must never be treated as text."""
        stdout = "\n".join([
            _oc_event("reasoning", part={
                "id": "prt_r",
                "type": "reasoning",
                "text": '[{"leaked": "chain of thought"}]',
            }),
            _oc_text('[{"answer": true}]'),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"answer": True}]

    def test_tool_use_parts_do_not_leak(self, provider):
        """`tool_use` carries a part whose type is `tool`, not `text`."""
        stdout = "\n".join([
            _oc_event("tool_use", part={
                "id": "prt_t",
                "type": "tool",
                "tool": "read",
                "state": {"output": '[{"leaked": "tool output"}]'},
            }),
            _oc_text('[{"answer": true}]'),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"answer": True}]

    def test_error_event_has_no_part_key(self, provider):
        """`error` is the one event with no `part`; it must not crash the loop."""
        stdout = _oc_event("error", error={
            "name": "ProviderAuthError",
            "data": {"providerID": "google", "message": "API key is missing."},
        })

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "ProviderAuthError" in result["error"]
        assert "API key is missing." in result["error"]

    def test_error_event_beats_generic_no_text_message(self, provider):
        """A reported error is surfaced instead of "No text events found"."""
        stdout = "\n".join([
            _oc_event("step_start", part={"id": "prt_0", "type": "step-start"}),
            _oc_event("error", error={"name": "UnknownError", "data": {"message": "boom"}}),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "UnknownError: boom" in result["error"]
        assert "No text events found" not in result["error"]

    def test_describe_failure_extracts_message(self, provider):
        """describe_failure recovers the reason from an error event on stdout."""
        stdout = _oc_event("error", error={
            "name": "ProviderAuthError",
            "data": {"providerID": "google", "message": "API key is missing."},
        })

        assert provider.describe_failure(stdout, "") == "ProviderAuthError: API key is missing."

    def test_describe_failure_none_without_error_event(self, provider):
        """A clean stream has nothing to add beyond the exit code."""
        assert provider.describe_failure(_oc_text("[1]"), "") is None

    def test_describe_failure_survives_garbage(self, provider):
        """Non-JSON stdout must not raise — it is a diagnostic path."""
        assert provider.describe_failure("not json at all\n\n", "") is None
        assert provider.describe_failure("", "") is None

    def test_describe_failure_ignores_non_dict_error(self, provider):
        """A malformed `error` payload yields None rather than a stray string."""
        assert provider.describe_failure(_oc_event("error", error="boom"), "") is None

    def test_describe_failure_name_only(self, provider):
        """An error with no nested message still reports its name."""
        stdout = _oc_event("error", error={"name": "ProviderAuthError"})

        assert provider.describe_failure(stdout, "") == "ProviderAuthError"

    def test_real_capture_auth_error(self, provider):
        """The real binary's auth failure, captured offline on the pin.

        The rig's `auth_error` case (a declared built-in `google` provider with
        no key): the CLI writes the error event to *stdout* and exits 1 -- the
        reason describe_failure exists. A live 1.18.8 failure had the same line.
        """
        stdout = oc_pin.read("auth_error")

        assert provider.describe_failure(stdout, "") == (
            "ProviderAuthError: Google Generative AI API key is missing. Pass it using "
            "the 'apiKey' parameter or the GOOGLE_GENERATIVE_AI_API_KEY environment variable."
        )

    def test_real_capture_invalid_model_composes_the_stderr_cause(self, provider):
        """The real binary on `--model opencode/sonnet`, captured offline on the pin.

        stdout carries only a wrapper — an `err_…` ref and no cause — and the
        `ProviderModelNotFoundError` with its "Did you mean" list is on stderr
        under `--print-logs --log-level ERROR`. This pins D5's whole contract:
        the composed **return value** of `describe_failure`. It does not pin the
        exit-3 line for this capture, which D4 owns (the classifier's verdict
        replaces this text there; see `test_subagent_executor.py`).
        """
        stdout = oc_pin.read("invalid_model")
        stderr = oc_pin.read("invalid_model", ".stderr.txt")

        # The suggestion list is the bundled catalog's, read from the capture.
        assert provider.describe_failure(stdout, stderr) == (
            "UnknownError: Unexpected server error. Check server logs for details. "
            "(ProviderModelNotFoundError: Model not found: opencode/sonnet. "
            f"{oc_pin.did_you_mean(stderr)})"
        )
        # The stdout-only behaviour is kept when stderr has no such line.
        assert provider.describe_failure(stdout, "") == (
            "UnknownError: Unexpected server error. Check server logs for details."
        )
        assert provider.describe_failure(stdout, "level=INFO error=\"nope\"\n") == (
            "UnknownError: Unexpected server error. Check server logs for details."
        )
        # And a non-wrapper stdout message is never decorated.
        auth = _oc_event("error", error={"name": "ProviderAuthError",
                                         "data": {"message": "API key is missing."}})
        assert provider.describe_failure(auth, stderr) == (
            "ProviderAuthError: API key is missing."
        )


OPENCODE_FIXTURES = oc_pin.ROOT


#: The rig's stream captures (`capture/<pin>/`) the adapter is graded on.
OC_STREAM_CASES = ("run_review", "denied_turn", "s41_cont_read", "auth_error", "invalid_model")


def _oc_fixture(name):
    return (OPENCODE_FIXTURES / name).read_bytes().decode("utf-8")


#: The same shape for an auth failure, as the rig's `auth_error` case records it.
OC_AUTH_EVENT = _oc_event(
    "error", timestamp=1, sessionID="ses_0",
    error={"name": "ProviderAuthError", "data": {
        "providerID": "openrouter", "message": "API key is missing",
    }},
)


class TestOpenCodeVendorText:
    """T2.1: opencode's classifier haystack is what the VENDOR wrote.

    opencode echoes the repository, not our prompt: `--format json` stdout
    carries every tool call's output. Run b38de8bb's validator grepped a plan
    discussing rate limits, died on an auto-rejected permission, and was
    recorded `rate_limited` off that grep.
    """

    TOOL_WITH_PLAN_TEXT = _oc_event(
        "tool_use", timestamp=1, sessionID="ses_0",
        part={"type": "tool", "tool": "grep", "state": {
            "status": "completed",
            "output": "Line 12: the endpoint returns 429 when the rate limit is hit",
        }},
    )
    PROSE = _oc_event(
        "text", timestamp=2, sessionID="ses_0",
        part={"type": "text", "text": "The limiter returns 429 on a rate_limit hit."},
    )
    REASONING = _oc_event(
        "reasoning", timestamp=3, sessionID="ses_0",
        part={"type": "reasoning", "text": "rate limit exceeded? 429?"},
    )
    STEP_START = _oc_event("step_start", timestamp=0, sessionID="ses_0",
                           part={"type": "step-start"})
    STEP_FINISH = _oc_event("step_finish", timestamp=4, sessionID="ses_0",
                            part={"type": "step-finish", "reason": "tool-calls",
                                  "tokens": {"input": 429, "output": 1}})

    @pytest.fixture
    def provider(self):
        return OpenCodeProvider()

    @pytest.mark.parametrize("event_type", [
        "text", "tool_use", "reasoning", "step_start", "step_finish",
    ])
    def test_non_vendor_event_types_are_dropped(self, provider, event_type):
        line = {
            "text": self.PROSE, "tool_use": self.TOOL_WITH_PLAN_TEXT,
            "reasoning": self.REASONING, "step_start": self.STEP_START,
            "step_finish": self.STEP_FINISH,
        }[event_type]
        stdout = "\n".join([line, OC_429_EVENT])
        assert provider.vendor_text(stdout, "") == (OC_429_EVENT, "")

    def test_error_events_unknown_types_and_non_json_lines_are_kept(self, provider):
        unknown = _oc_event("session_notice", note="429 rate limit")
        banner = "opencode: some non-JSON banner line"
        stdout = "\n".join([
            self.STEP_START, OC_429_EVENT, self.TOOL_WITH_PLAN_TEXT, unknown,
            banner, self.STEP_FINISH,
        ])
        reduced, _ = provider.vendor_text(stdout, "")
        assert reduced.splitlines() == [OC_429_EVENT, unknown, banner]

    def test_a_run_of_only_prose_and_tools_reduces_to_nothing(self, provider):
        """`""` is the accurate reduction of "opencode reported no error"."""
        stdout = "\n".join([self.STEP_START, self.TOOL_WITH_PLAN_TEXT,
                            self.PROSE, self.STEP_FINISH])
        assert provider.vendor_text(stdout, "err") == ("", "err")
        assert provider.classify_failure(stdout, "", 0) is None
        # The pre-T2.1 haystack read the tool output as a rate limit.
        assert LLMProvider.classify_failure(
            type("_Bare", (OpenCodeProvider,), {
                "vendor_text": LLMProvider.vendor_text,
            })(), stdout, "", 0
        ) == STATUS_RATE_LIMITED

    def test_an_end_cut_final_line_is_dropped(self, provider):
        cut = self.TOOL_WITH_PLAN_TEXT[:90]
        assert cut.startswith('{"type":"tool_use"')
        stdout = "\n".join([OC_429_EVENT, cut])
        assert provider.vendor_text(stdout, "") == (OC_429_EVENT, "")

    def test_an_end_cut_line_with_type_in_second_position_is_dropped(self, provider):
        """Key order is not guaranteed: the opener is found in any position."""
        line = json.dumps({
            "timestamp": 1790268175946, "type": "tool_use",
            "part": {"state": {"output": "the endpoint returns 429 on a rate limit"}},
        }, separators=(",", ":"))
        cut = line[:line.index("429") + 3]
        stdout = "\n".join([OC_429_EVENT, cut])
        assert provider.vendor_text(stdout, "") == (OC_429_EVENT, "")

    def test_a_cut_line_that_is_not_last_is_kept(self, provider):
        """Only the FINAL line can be a kill's cut; anything else is kept."""
        cut = self.TOOL_WITH_PLAN_TEXT[:90]
        stdout = "\n".join([cut, OC_429_EVENT])
        reduced, _ = provider.vendor_text(stdout, "")
        assert reduced.splitlines() == [cut, OC_429_EVENT]

    def test_a_beginning_cut_first_line_is_kept(self, provider):
        """A stored tail's first line carries no opener: kept, by design."""
        head_cut = self.TOOL_WITH_PLAN_TEXT[40:]
        stdout = "\n".join([head_cut, self.STEP_FINISH])
        reduced, _ = provider.vendor_text(stdout, "")
        assert reduced == head_cut

    @pytest.mark.parametrize("junk", [
        "", "   ", "not json at all", "Ran into this error: 429 rate limit exceeded",
        '{"unclosed": ', "[1, 2, 3]", "\x00\x01",
    ])
    def test_no_parseable_event_is_the_identity(self, provider, junk):
        assert provider.vendor_text(junk, "err") == (junk, "err")

    def test_a_raised_exception_is_the_identity(self, provider, monkeypatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr("utils.providers.opencode.json.loads", boom)
        stdout = "\n".join([self.TOOL_WITH_PLAN_TEXT, OC_429_EVENT])
        assert provider.vendor_text(stdout, "err") == (stdout, "err")

    def test_stderr_is_never_reduced(self, provider):
        stdout = "\n".join([self.TOOL_WITH_PLAN_TEXT, OC_429_EVENT])
        _, stderr = provider.vendor_text(stdout, OC_429_STDERR)
        assert stderr == OC_429_STDERR

    def test_the_reduction_only_ever_removes(self, provider):
        """Every committed opencode capture: a subsequence, and no stderr."""
        paths = sorted(
            p for p in list(OPENCODE_FIXTURES.iterdir())
            + list(oc_pin.CAPTURE.iterdir())
            if p.is_file() and p.suffix in (".txt", ".jsonl", ".json")
        )
        assert paths
        for path in paths:
            raw = path.read_bytes().decode("utf-8")
            reduced, stderr = provider.vendor_text(raw, "a stderr diagnosis")
            remaining = iter(raw)
            assert all(char in remaining for char in reduced), path.name
            assert stderr == "a stderr diagnosis", path.name


class _OpenCodeBeforeT21(OpenCodeProvider):
    """opencode as it was before T2.1 and T4.3: the base haystack, no structure."""

    vendor_text = LLMProvider.vendor_text
    classify_failure = LLMProvider.classify_failure


class TestOpenCodeCapturesKeepTheirVerdict:
    """T2.4: a later change to `vendor_text` cannot move any committed capture."""

    @pytest.mark.parametrize(
        "row,stdout,stderr,exit_code,before,after", OC_GRADED_CAPTURES,
        ids=[row[0] for row in OC_GRADED_CAPTURES],
    )
    def test_before_and_after(self, row, stdout, stderr, exit_code, before, after):
        out, err = _oc_stream(stdout), _oc_stream(stderr)
        assert _OpenCodeBeforeT21().classify_failure(out, err, exit_code) == before
        assert OpenCodeProvider().classify_failure(out, err, exit_code) == after

    @pytest.mark.parametrize("case", ["empty", "stored", "env_only",
                                      "config_provider_only"])
    def test_the_auth_list_capture_is_not_a_failure_either_way(self, case):
        """`auth_list.<pin>.json` is an auth probe, not a run; graded anyway."""
        data = json.loads(oc_pin.AUTH_LIST.read_text("utf-8"))["cases"][case]
        out, err = data["stdout"], data["stderr"]
        assert _OpenCodeBeforeT21().classify_failure(out, err, 0) is None
        assert OpenCodeProvider().classify_failure(out, err, 0) is None

    @pytest.mark.parametrize(
        "case", sorted(json.loads(oc_pin.DEBUG_CONFIG.read_text("utf-8"))["cases"])
    )
    def test_the_debug_config_capture_is_not_a_failure_either_way(self, case):
        """`debug_config.<pin>.json` (plan §8 T8.1) is a config read, not a
        run, and nothing hands its stdout to `classify_failure` (the reader
        reduces it in memory). Graded anyway, every case exit 0 but
        `builtin_override_filekey_missing`, whose `{file:}` names no file:
        opencode refuses the whole config (exit 1, `bad file reference` on
        stderr), which nothing classifies either. One case reads
        as a failure both before and after T2.1: `builtin_override_keyless`
        carries an `"options": {"timeout": 30000}` and is one pretty-printed
        JSON document, not NDJSON events, so `vendor_text` keeps its lines as
        unrecognised text and the `timeout` pattern matches. Pinned so a
        change is noticed; it would matter only if this output were ever
        classified."""
        data = json.loads(oc_pin.DEBUG_CONFIG.read_text("utf-8"))["cases"][case]
        out, err = data["stdout"], data["stderr"]
        code = data["exit"]
        assert code == (1 if case == "builtin_override_filekey_missing" else 0)
        expected = "timeout" if case == "builtin_override_keyless" else None
        assert _OpenCodeBeforeT21().classify_failure(out, err, code) == expected
        assert OpenCodeProvider().classify_failure(out, err, code) == expected

    def test_every_committed_capture_is_graded(self):
        """Every stream case of the rig joins the table, and nothing but the
        pin's two probe captures sits beside the rig (an older release's
        capture is removed when the pin moves, never kept beside it)."""
        graded = {
            spec[1].partition(".")[0] for row in OC_GRADED_CAPTURES for spec in row[1:3]
            if isinstance(spec, tuple) and spec[0] == "rig"
        }
        assert graded == set(OC_STREAM_CASES)
        for case in graded:
            assert oc_pin.capture(case).is_file(), case
        committed = {
            p.name for p in OPENCODE_FIXTURES.iterdir()
            if p.is_file() and p.suffix in (".txt", ".jsonl", ".json")
        }
        assert committed == {oc_pin.AUTH_LIST.name, oc_pin.DEBUG_CONFIG.name}
        releases = [
            p.name for p in (OPENCODE_FIXTURES / "capture").iterdir()
            if p.is_dir() and p.name[:1].isdigit()
        ]
        assert releases == [oc_pin.PIN]


def _denied_turn():
    return oc_pin.read("denied_turn"), oc_pin.read("denied_turn", ".stderr.txt")


class TestOpenCodeDeniedTurn:
    """T4.3: a turn `run` ended on an auto-rejected permission → `runner_error`.

    Read structurally off `part.reason` / `part.state` (never top-level keys),
    with the reason sentence carried by `classify_structural` because
    `classify_failure` returns a category and nothing else.
    """

    SENTENCE = (
        "opencode ended the turn after an auto-rejected permission "
        "(external_directory /etc/*); no final answer"
    )

    @pytest.fixture
    def provider(self):
        return OpenCodeProvider()

    def test_the_fixture_is_runner_error_naming_the_permission(self, provider):
        stdout, stderr = _denied_turn()
        assert provider.classify_failure(stdout, stderr, 0) == "runner_error"
        status, reason = provider.classify_structural(stdout, stderr, 0)
        assert status == "runner_error"
        assert reason == self.SENTENCE
        assert "external_directory" in reason and "/etc/*" in reason

    def test_the_fixture_is_never_rate_limited(self, provider):
        stdout, stderr = _denied_turn()
        assert "rate limit" in stdout and "rate_limit" in stdout and "429" in stdout
        assert provider.classify_failure(stdout, stderr, 0) != STATUS_RATE_LIMITED

    @pytest.mark.parametrize("event,expected", [
        (OC_429_EVENT, STATUS_RATE_LIMITED),
        (OC_AUTH_EVENT, STATUS_AUTH_FAILED),
    ])
    def test_a_genuine_vendor_error_wins(self, provider, event, expected):
        stdout, stderr = _denied_turn()
        stdout = stdout + event + "\n"
        assert provider.classify_failure(stdout, stderr, 0) == expected

    def test_without_stderr_the_tool_use_alone_fires(self, provider):
        """The pin's PermissionRejectedError carries no fields (read from the
        binary), so without the stderr line the sentence omits the
        parenthetical rather than inventing one."""
        stdout, _ = _denied_turn()
        assert provider.classify_failure(stdout, "", 0) == "runner_error"
        assert provider.classify_structural(stdout, "", 0) == (
            "runner_error",
            "opencode ended the turn after an auto-rejected permission; "
            "no final answer",
        )

    def test_a_tool_use_payload_that_names_the_permission_is_parsed(self, provider):
        """Hypothetical shape (the pinned build does not write it): if the rejected
        `tool_use`'s `part.state.error` ever carries `run`'s own
        `permission requested: <permission> (<pattern>)` words, they are read
        from there when stderr is absent."""
        rejected = _oc_event(
            "tool_use", timestamp=1, sessionID="ses_0",
            part={"type": "tool", "tool": "read", "state": {
                "status": "error",
                "error": "The user rejected permission to use this specific tool "
                         "call. permission requested: external_directory "
                         "(/etc/*); auto-rejecting",
            }},
        )
        finish = _oc_event("step_finish", timestamp=2, sessionID="ses_0",
                           part={"type": "step-finish", "reason": "tool-calls"})
        stdout = "\n".join([rejected, finish])
        assert provider.classify_structural(stdout, "", 0) == (
            "runner_error", self.SENTENCE,
        )

    def test_a_parenthesised_path_keeps_the_parenthetical(self, provider):
        """The pattern runs up to the fixed `); auto-rejecting` terminator, so
        a `)` inside the path does not drop the parenthetical."""
        stdout, _ = _denied_turn()
        stderr = ("! permission requested: external_directory "
                  "(/home/u/My (old) files/*); auto-rejecting\n")
        assert provider.classify_structural(stdout, stderr, 0) == (
            "runner_error",
            "opencode ended the turn after an auto-rejected permission "
            "(external_directory /home/u/My (old) files/*); no final answer",
        )

    def test_continue_loop_on_deny_is_not_a_denied_turn(self, provider):
        """MEASURED: stderr still says `auto-rejecting`, but the turn finished
        `stop` with an answer. stderr alone must never fire the check."""
        stdout = oc_pin.read("s41_cont_read")
        stderr = oc_pin.read("s41_cont_read", ".stderr.txt")
        assert "auto-rejecting" in stderr
        assert provider.classify_structural(stdout, stderr, 0) is None
        assert provider.classify_failure(stdout, stderr, 0) is None

    def test_a_healthy_run_is_not_a_denied_turn(self, provider):
        stdout = oc_pin.read("run_review")
        assert provider.classify_structural(stdout, "", 0) is None
        assert provider.classify_failure(stdout, "", 0) is None

    def test_tool_calls_without_a_rejection_falls_through(self, provider):
        """A stream ending `step_finish tool-calls` with no rejected tool_use and
        no stderr line is today's path: nothing structural, base tables only."""
        stdout, _ = _denied_turn()
        lines = [
            line for line in stdout.splitlines()
            if "rejected permission" not in line
        ]
        stdout = "\n".join(lines) + "\n"
        assert json.loads(lines[-1])["part"]["reason"] == "tool-calls"
        assert provider.classify_structural(stdout, "", 0) is None
        assert provider.classify_failure(stdout, "", 0) is None

    def test_a_non_zero_exit_is_not_a_denied_turn(self, provider):
        stdout, stderr = _denied_turn()
        assert provider.classify_structural(stdout, stderr, 1) is None

    def test_top_level_reason_keys_are_never_read(self, provider):
        """The fields live under `part`; a top-level `reason` means nothing."""
        stdout, stderr = _denied_turn()
        events = [json.loads(line) for line in stdout.splitlines()]
        for event in events:
            if event["type"] == "step_finish":
                event["reason"] = "stop"
        stdout = "\n".join(json.dumps(e) for e in events)
        assert provider.classify_structural(stdout, stderr, 0)[0] == "runner_error"

    def test_the_base_hook_answers_none(self):
        """Every other adapter is unchanged."""
        stdout, stderr = _denied_turn()
        assert LLMProvider.classify_structural(
            GooseProvider(), stdout, stderr, 0
        ) is None


class TestOpenCodeStreamCapturesOnThePin:
    """The parser is graded on the real binary's streams at the pin.

    opencode supports its newest release only: `min_version` and
    `verified_version` are one version, and the rig (`capture/run.sh`) captures
    these streams from that release offline, against a localhost stub, at every
    bump. So a stream-shape break on a new release surfaces here as a red test
    on the bump, rather than as a version number nobody moved.

    `run_review` scripts the shape of a real review turn (the D3 gate's 1.18.30
    `opencode/big-pickle` reviewer, 2026-09-11): four `step_start`/`step_finish`
    pairs, four `tool_use`s, two `text` parts, no `error` event, with that run's
    per-step token counts. The review itself went to a file, so the text is
    prose and `parse_output` reports no JSON — which is the correct reading of
    that stream, not a parser miss.
    """

    @pytest.fixture
    def provider(self):
        from utils.provider_registry import get_provider

        return get_provider("opencode")

    @pytest.fixture
    def review_stream(self):
        return oc_pin.read("run_review")

    def test_the_floor_is_the_capture_release(self, provider):
        assert provider.min_version == provider.verified_version == oc_pin.PIN

    def test_the_review_capture_is_a_real_event_stream(self, review_stream):
        events = [json.loads(line) for line in review_stream.splitlines() if line]
        kinds = [event["type"] for event in events]
        assert kinds.count("step_start") == 4
        assert kinds.count("step_finish") == 4
        assert kinds.count("tool_use") == 4
        assert kinds.count("text") == 2
        assert "error" not in kinds

    def test_usage_is_summed_over_every_step(self, provider, review_stream):
        """Reading the last step alone would under-report a multi-step run."""
        usage = provider.parse_usage(review_stream, "")
        assert usage == {
            "input_tokens": 11489,
            "output_tokens": 1764,
            "cached_input_tokens": 28928,
            "cost_usd": 0.0,
            "total_tokens": 13253,
            "tool_uses": 4,
        }

    def test_the_text_parts_are_read_in_order(self, provider, review_stream):
        result = provider.parse_output(review_stream, "")
        raw = str(result.get("raw") or "")
        assert raw.startswith("Now let me explore the codebase")
        assert "JSON written to" in raw
        # Prose, not an answer: the review went to the file the prompt named.
        assert result["success"] is False
        assert "No valid JSON" in str(result.get("error"))

    def test_a_clean_review_stream_reports_no_failure(self, provider, review_stream):
        assert provider.describe_failure(review_stream, "") is None
        assert provider.classify_failure(review_stream, "", 0) is None

    def test_the_auth_error_is_read(self, provider):
        stdout = oc_pin.read("auth_error")
        assert provider.describe_failure(stdout, "") == (
            "ProviderAuthError: Google Generative AI API key is missing. Pass it using "
            "the 'apiKey' parameter or the GOOGLE_GENERATIVE_AI_API_KEY environment variable."
        )
        assert provider.classify_failure(stdout, "", 1) == STATUS_AUTH_FAILED

    def test_the_denied_turn_is_read(self, provider):
        """T2.3: run b38de8bb's shape, reproduced by the rig's `denied_turn`.

        exit 0, two `step_finish tool-calls`, a rejected bash `tool_use` and no
        final answer. Its tool output quotes plan text about rate limits, which
        is how b38de8bb was recorded `rate_limited`; it is `runner_error` now.
        """
        stdout, stderr = _denied_turn()
        events = [json.loads(line) for line in stdout.splitlines() if line]
        kinds = [event["type"] for event in events]
        assert kinds.count("step_finish") == 2
        assert "error" not in kinds
        assert events[-1]["part"]["reason"] == "tool-calls"
        rejected = [
            event for event in events
            if event["type"] == "tool_use"
            and event["part"]["state"]["status"] == "error"
            and "rejected permission" in event["part"]["state"].get("error", "")
        ]
        assert [event["part"]["tool"] for event in rejected] == ["bash"]
        assert len({event["sessionID"] for event in events}) == 1
        assert stderr.strip().endswith(
            "permission requested: external_directory (/etc/*); auto-rejecting"
        )
        result = provider.parse_output(stdout, stderr)
        assert result["success"] is False
        # The adapter's own denied-turn sentence, not the parser's verdict on
        # the narration (plans/ask-answer-recovery-and-false-success.md D3.1).
        assert result["error"] == (
            "opencode ended the turn after an auto-rejected permission "
            "(external_directory /etc/*); no final answer"
        )
        assert result["incomplete"] is True
        assert provider.classify_failure(stdout, stderr, 0) == "runner_error"
