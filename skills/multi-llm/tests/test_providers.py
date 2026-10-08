"""Unit tests for LLM provider implementations.

This module tests the provider implementations for cursor-agent, Gemini, and OpenCode
CLI tools. It uses mocking to isolate tests from actual CLI tool availability and
focuses on output parsing logic.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.agy import AgyProvider
from utils.providers.aider import AiderProvider
from utils.providers.base import (
    CATEGORY_INVALID_MODEL,
    LLMProvider,
    split_reasoning_effort,
)
from utils.retry_ladder import (
    STATUS_AUTH_FAILED,
    STATUS_CONTEXT_LENGTH_EXCEEDED,
    STATUS_RATE_LIMITED,
)
from utils.providers import base as base_module
from utils.providers.claude_code import ClaudeCodeProvider
from utils.providers.cline import ClineProvider
from utils.providers.codex import CodexProvider
from utils.providers.copilot import CopilotProvider
from utils.providers import cursor_agent as cursor_agent_module
from utils.providers.cursor_agent import CursorAgentProvider
from utils.providers.gemini import GeminiProvider
from utils.providers.goose import GooseProvider
from utils.providers.grok import GrokProvider
from utils.providers.kilocode import KiloCodeProvider
from utils.providers import kimi as kimi_module
from utils.providers.kimi import KimiProvider
from utils.providers.muse import MuseProvider
from utils.providers import omp as omp_module
from utils.providers.omp import (
    OMP_APPROVAL_MODE,
    OMP_MIN_MAX_TIME_S,
    OMP_REQUIRED_FLAGS,
    OMP_SHUTDOWN_MARGIN_S,
    OMP_TOOLS,
    OVERLAY_PATH as OMP_OVERLAY_PATH,
    OmpFlagsUnconfirmed,
    OmpProvider,
    max_time_value,
)
from utils.providers.opencode import OpenCodeProvider
from tests import opencode_fixtures as oc_pin
from utils.providers.pi import PiProvider
from utils.providers import qwen_code as qwen_code_module
from utils.providers.qwen_code import QwenCodeProvider
from utils.providers import vibe as vibe_module
from utils.providers.vibe import VibeProvider


class TestSplitReasoningEffort:
    """Tests for the split_reasoning_effort helper in providers.base."""

    EFFORTS = frozenset({"low", "medium", "high"})

    def test_effort_suffix_split(self):
        """A whitelisted suffix after the last colon is split off."""
        assert split_reasoning_effort("gpt-5.5:high", self.EFFORTS) == ("gpt-5.5", "high")

    def test_no_colon_passthrough(self):
        """A model without a colon passes through verbatim."""
        assert split_reasoning_effort("gpt-5.5", self.EFFORTS) == ("gpt-5.5", None)

    def test_unknown_suffix_passthrough(self):
        """A suffix not in the whitelist keeps the full model string."""
        assert split_reasoning_effort("gpt-5.5:turbo", self.EFFORTS) == ("gpt-5.5:turbo", None)

    def test_empty_base_passthrough(self):
        """An empty base before the colon (":high") passes through verbatim."""
        assert split_reasoning_effort(":high", self.EFFORTS) == (":high", None)

    def test_empty_suffix_passthrough(self):
        """A trailing colon with no suffix passes through verbatim."""
        assert split_reasoning_effort("m:", self.EFFORTS) == ("m:", None)

    def test_multi_colon_splits_on_last(self):
        """Multi-colon models split on the LAST colon only."""
        assert split_reasoning_effort("a:b:high", self.EFFORTS) == ("a:b", "high")

    def test_whitelist_sensitivity(self):
        """The same string splits or passes through depending on the whitelist."""
        assert split_reasoning_effort("m:max", frozenset({"max"})) == ("m", "max")
        assert split_reasoning_effort("m:max", frozenset({"high"})) == ("m:max", None)


class TestClaudeCodeProvider:
    """Tests for the ClaudeCodeProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a ClaudeCodeProvider instance."""
        return ClaudeCodeProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "claude-code"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_claude_code_parse_json_wrapper(self, provider):
        """Parse Claude Code output with JSON wrapper format.

        Claude Code with --output-format json returns responses wrapped in:
        {"type":"result","result":"...","session_id":"...","total_cost_usd":...}
        """
        inner_data = [
            {"title": "Test suggestion", "desc": "Description", "importance": "high"}
        ]
        wrapper = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps(inner_data),
            "session_id": "abc-123",
            "total_cost_usd": 0.05
        }
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_claude_code_parse_json_wrapper_with_object_result(self, provider):
        """Parse Claude Code output when inner result is already an object."""
        inner_data = {"key": "value", "nested": {"a": 1}}
        wrapper = {"type": "result", "result": json.dumps(inner_data)}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_claude_code_parse_code_block(self, provider):
        """Extract JSON from markdown code block in Claude Code output."""
        inner_json = [{"title": "Suggestion 1", "importance": "high"}]
        text_with_codeblock = f"""Here is my analysis:

```json
{json.dumps(inner_json, indent=2)}
```

This concludes the review."""
        wrapper = {"type": "result", "result": text_with_codeblock}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_claude_code_parse_code_block_without_json_label(self, provider):
        """Extract JSON from code block without json language label."""
        inner_json = {"status": "ok", "items": [1, 2, 3]}
        text_with_codeblock = f"""Output:
```
{json.dumps(inner_json)}
```
"""
        wrapper = {"type": "result", "result": text_with_codeblock}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_claude_code_parse_raw_json_in_text(self, provider):
        """Extract raw JSON from text without code blocks."""
        inner_json = [{"id": 1}, {"id": 2}]
        text_with_json = f"The results are: {json.dumps(inner_json)} end of results."
        wrapper = {"type": "result", "result": text_with_json}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_claude_code_parse_direct_array(self, provider):
        """Parse direct JSON array output (no wrapper)."""
        data = [{"item": 1}, {"item": 2}]
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_claude_code_parse_empty_result(self, provider):
        """Handle empty result in JSON wrapper."""
        wrapper = {"type": "result", "result": ""}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_claude_code_parse_no_json_found(self, provider):
        """Handle output with no extractable JSON."""
        wrapper = {"type": "result", "result": "Just plain text with no JSON"}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result
        assert "raw" in result

    def test_claude_code_build_command(self, provider):
        """Test command building for Claude Code CLI."""
        prompt = "Review this code"
        model = "sonnet"

        cmd = provider.build_command(prompt, model)

        assert cmd == [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            "sonnet",
            "--permission-mode",
            "acceptEdits",
            "Review this code",
        ]

    def test_claude_code_build_command_with_a_schema_exact(self, provider):
        """§4.2/§4.2.2's exact argv — the `==` row §14.1 requires for this flag.

        The schema pair lands at the fixed index right after
        `--output-format json`, ahead of `--model`, so the prompt is still
        `cmd[-1]` and the flag is never adjacent to it.
        """
        from unittest.mock import patch

        from utils.providers.base import StructuredSchema
        from utils.providers.claude_code import reset_flag_gate_cache

        reset_flag_gate_cache()
        schema = StructuredSchema(
            document={"type": "object", "properties": {"a": {"type": "string"}}},
            name="probe.schema.json",
        )
        try:
            with patch("shutil.which", return_value="/usr/bin/claude"), patch(
                # M14: one gate, one implementation — so one patch target.
                "utils.providers.base.read_help_tokens",
                lambda binary, **_kwargs: frozenset({"--json-schema"}),
            ):
                cmd = provider.build_command("Review this code", "sonnet",
                                             schema=schema)
        finally:
            reset_flag_gate_cache()

        assert cmd == [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            '{"properties":{"a":{"type":"string"}},"type":"object"}',
            "--model",
            "sonnet",
            "--permission-mode",
            "acceptEdits",
            "Review this code",
        ]

    def test_claude_code_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the five valid efforts."""
        from utils.providers.claude_code import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {"low", "medium", "high", "xhigh", "max"}

    def test_claude_code_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --effort pair after --model <base>."""
        cmd = provider.build_command("Review this code", "fable:max")

        assert cmd == [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            "fable",
            "--effort",
            "max",
            "--permission-mode",
            "acceptEdits",
            "Review this code",
        ]

    def test_claude_code_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --effort <effort> form."""
        for effort in ("low", "medium", "high", "xhigh", "max"):
            cmd = provider.build_command("Review this code", f"fable:{effort}")

            assert cmd == [
                "claude",
                "-p",
                "--output-format",
                "json",
                "--model",
                "fable",
                "--effort",
                effort,
                "--permission-mode",
                "acceptEdits",
                "Review this code",
            ], f"effort {effort!r} did not build the expected command"

    def test_claude_code_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --effort flag."""
        for model in ("fable", "fable:med", "fable:none", "fable:HIGH", ":max"):
            cmd = provider.build_command("Review this code", model)

            assert cmd == [
                "claude",
                "-p",
                "--output-format",
                "json",
                "--model",
                model,
                "--permission-mode",
                "acceptEdits",
                "Review this code",
            ], f"model {model!r} was not passed through verbatim"

    @patch("shutil.which")
    def test_claude_code_is_available_true(self, mock_which, provider):
        """Test is_available returns True when claude is found."""
        mock_which.return_value = "/usr/local/bin/claude"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("claude")

    @patch("shutil.which")
    def test_claude_code_is_available_false(self, mock_which, provider):
        """Test is_available returns False when claude is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("claude")

    def test_get_remove_env_strips_claudecode(self, provider):
        """Test that CLAUDECODE is listed for removal to avoid nested-session guard."""
        remove = provider.get_remove_env()

        assert "CLAUDECODE" in remove

    @patch.dict(os.environ, {"CLAUDECODE": "1", "HOME": "/home/test"})
    def test_get_env_does_not_include_claudecode(self, provider):
        """get_env never re-adds CLAUDECODE (stripping is via get_remove_env).

        It is no longer empty: §7.3/L6 makes it carry the CLI's own retry budget.
        The contents are asserted by ``TestClaudeCodeRetryEnvelope`` below; what
        this one still pins is that the nested-session marker is not among them.
        """
        env = provider.get_env("opus")

        assert "CLAUDECODE" not in env


class TestClaudeCodeRetryEnvelope:
    """§7.3 (task L6) — the retry budget is delegated to the vendor CLI.

    The CLI's own default is **10 retries**, measured at 182 s of wall clock in a
    single rate-limited slot before the harness saw anything at all. So `get_env`
    pins it to §3.6's floor of 3 and sizes the CLI's own per-request timeout to
    sit *under* the outer per-provider slot timeout, in milliseconds.
    """

    @pytest.fixture
    def provider(self):
        return ClaudeCodeProvider()

    def _with_timeout(self, monkeypatch, seconds):
        """Pin the resolved per-provider slot timeout `get_env` derives from."""
        import utils.provider_registry as registry

        monkeypatch.setattr(
            registry,
            "get_provider_timeout",
            lambda name, anchor=None, **kw: seconds,
        )

    def test_max_retries_is_the_floor_exactly(self, provider):
        """3 is §3.6's floor — below it `--fallback-model`'s overload path dies."""
        from utils.providers.claude_code import CLAUDE_CODE_MAX_RETRIES

        assert CLAUDE_CODE_MAX_RETRIES == 3
        assert provider.get_env("sonnet")["CLAUDE_CODE_MAX_RETRIES"] == "3"

    def test_api_timeout_is_milliseconds_at_the_1800s_slot(self, provider, monkeypatch):
        """providers.yaml ships `default_timeout: 1800` → the 300 000 ms ceiling."""
        self._with_timeout(monkeypatch, 1800)

        env = provider.get_env("sonnet")

        # min(300_000, 1_800_000 // 4 - 30_000) = min(300_000, 420_000)
        assert env["API_TIMEOUT_MS"] == "300000"

    def test_api_timeout_is_milliseconds_at_the_600s_slot(self, provider, monkeypatch):
        """The adapter's own `default_timeout` end: the derivation, not the ceiling."""
        self._with_timeout(monkeypatch, 600)

        env = provider.get_env("sonnet")

        # min(300_000, 600_000 // 4 - 30_000) = min(300_000, 120_000)
        assert env["API_TIMEOUT_MS"] == "120000"

    def test_the_value_is_milliseconds_not_seconds(self, provider, monkeypatch):
        """The s→ms conversion is the whole point: the donor plan had it wrong."""
        self._with_timeout(monkeypatch, 600)

        value = int(provider.get_env("sonnet")["API_TIMEOUT_MS"])

        assert value == 120_000
        assert value != 120, "seconds, not milliseconds — this is the §7.3 defect"

    @pytest.mark.parametrize("seconds", [300, 600, 900, 1200, 1800, 3600, 4242])
    def test_the_whole_retry_budget_fits_inside_the_slot(
        self, provider, monkeypatch, seconds
    ):
        """(MAX_RETRIES + 1) × API_TIMEOUT_MS must stay under the outer deadline.

        §9.6.c's margin rule applied one level in: the CLI has to reach its own
        cutoff and exit before the harness kills it, or a live child is killed and
        the run reports a bare timeout instead of the CLI's own reason.
        """
        from utils.providers.claude_code import CLAUDE_CODE_MAX_RETRIES

        self._with_timeout(monkeypatch, seconds)
        env = provider.get_env("sonnet")

        worst_case_ms = (CLAUDE_CODE_MAX_RETRIES + 1) * int(env["API_TIMEOUT_MS"])
        assert worst_case_ms < seconds * 1000, (
            f"a {seconds}s slot would be exceeded by the CLI's own retry budget "
            f"({worst_case_ms} ms)"
        )

    def test_a_slot_too_short_to_carry_a_budget_omits_the_timeout(
        self, provider, monkeypatch
    ):
        """Below ~120 s the derivation goes non-positive; emit nothing, not junk.

        A zero or negative `API_TIMEOUT_MS` would abort every request instantly.
        The retry cap still ships — it is the half that is a bug fix on its own.
        """
        self._with_timeout(monkeypatch, 60)

        env = provider.get_env("sonnet")

        assert env["CLAUDE_CODE_MAX_RETRIES"] == "3"
        assert "API_TIMEOUT_MS" not in env

    def test_the_configured_claude_code_timeout_is_honoured(self, provider):
        """No monkeypatch: the real `providers.yaml` value flows through."""
        from utils.provider_registry import get_provider_timeout
        from utils.providers.claude_code import derive_api_timeout_ms

        resolved = get_provider_timeout("claude-code")
        env = provider.get_env("sonnet")

        assert env["API_TIMEOUT_MS"] == str(derive_api_timeout_ms(resolved))

    def test_the_fictional_anthropic_names_are_never_emitted(self, provider):
        """Measured 2026-08-16: zero occurrences in the binary, provably inert.

        Emitting them would look like a retry policy and be one only on paper.
        """
        env = provider.get_env("sonnet")

        assert "ANTHROPIC_MAX_RETRIES" not in env
        assert "ANTHROPIC_REQUEST_TIMEOUT_SECONDS" not in env

    def test_the_fictional_names_are_absent_from_the_adapter_source(self):
        """Not just from one call — from the module, so nobody re-adds them."""
        import inspect

        import utils.providers.claude_code as adapter

        source = inspect.getsource(adapter)
        for name in ("ANTHROPIC_MAX_RETRIES", "ANTHROPIC_REQUEST_TIMEOUT_SECONDS"):
            # The module names them once each, in the comment that records the
            # measurement; what must never appear is an assignment.
            assert f'"{name}"' not in source, (
                f"{name} is measurably inert — it must not be a key in get_env()"
            )

    def test_only_claude_code_declares_a_retry_shaped_variable(self):
        """§7.6(b)'s converse: no other adapter may set one without an envelope."""
        from utils.provider_registry import _PROVIDERS

        for name, provider in sorted(_PROVIDERS.items()):
            # Written when claude-code was the ONLY adapter with a measured
            # envelope, so it named that one exception directly. The rule it is
            # actually enforcing is the one in its own assertion message --
            # retry-shaped env REQUIRES a declaration -- so the exemption is the
            # declaration, not the name. vibe is the case that forced this:
            # `VIBE_API_TIMEOUT` is retry-shaped by construction (the declared
            # `max_wall_s` is computed from it and from the pinned 2.26.0 retry
            # budget) and pinning it is what makes the envelope true, exactly as
            # `CLAUDE_CODE_MAX_RETRIES` does.
            if provider.internal_retries is not None:
                continue
            try:
                env = provider.get_env("m")
            except TypeError:
                env = provider.get_env()
            retryish = [
                key for key in (env or {})
                if "MAX_RETRIES" in key.upper() or "API_TIMEOUT" in key.upper()
            ]
            assert not retryish, f"{name} sets {retryish} with no declared envelope"


#: §7.6(b)'s guard shape. Two families, both of which move the number §7.5 rule 4
#: subtracts: a *count* of vendor-internal requests (`…MAX_RETRIES`, a backoff or
#: attempt cap) and the *per-request* cutoff those requests are multiplied by
#: (`API_TIMEOUT_MS` — claude-code's `max_wall_s` is literally 4 × this). Both are
#: matched, because either one set without a declared envelope leaves the ladder
#: budgeting for one request against a subprocess that will make several.
#: Deliberately narrow: a generic `*_TIMEOUT` would catch env shaping that says
#: nothing about retries and force a fabricated declaration.
RETRY_SHAPED_ENV_KEY = r"RETRY|RETRIES|BACKOFF|ATTEMPTS?|API_TIMEOUT|REQUEST_TIMEOUT"


def _retry_shaped_env_keys(provider):
    """The retry-shaped variables this adapter's ``get_env`` really sets."""
    import re

    try:
        env = provider.get_env("m")
    except TypeError:  # pragma: no cover - defensive, no adapter needs it today
        env = provider.get_env()
    return sorted(
        key for key in (env or {})
        if re.search(RETRY_SHAPED_ENV_KEY, key.upper())
    )


def _envelope_sweep_names():
    """Every provider id in the registry, resolved at class-definition time.

    A local twin of `_registered_provider_names()` — which is defined further
    down this module, after this class — because `@pytest.mark.parametrize`
    evaluates its argument while the class body runs.
    """
    from utils.provider_registry import _PROVIDERS

    return sorted(_PROVIDERS)


class TestInternalRetryEnvelopeDeclarations:
    """§7.6(b) — every adapter declares its CLI's internal retry envelope.

    §7.5's arithmetic needs one number per adapter and today that number exists
    only in the plan's measurements. The declaration is the input rule 4
    subtracts: a slot whose remaining budget cannot hold one worst-case attempt
    of the adapter's own vendor envelope is `budget_exhausted` *before* a child
    is spawned, and rule 6(b) refuses an identical rate-limit re-dispatch to an
    adapter that already retried 429s internally.

    Nothing reads it yet — M9's ladder is written and pure, and consumes it
    structurally (`retry_ladder.InternalRetryEnvelopeLike`), which is why these
    tests assert the three attribute *names* rather than an isinstance.
    """

    #: Adapters whose vendor envelope was measured against a fail-forever
    #: stub. `kimi` joined on 0.36.1: 10 attempts, exponential backoff to
    #: ~38 s, 429s retried and `Retry-After` honoured exactly — the row
    #: §6.1.1 explicitly warned must be *confirmed*, not assumed `None`.
    MEASURED = ("claude-code", "codex", "kimi")

    @pytest.fixture
    def registry(self):
        from utils.provider_registry import _PROVIDERS

        return _PROVIDERS

    def test_the_sweep_really_covers_every_registered_adapter(self, registry):
        """The done-when is 'all of them' — a shrinking sweep must fail, not pass.

        The count is spelled once, here, and the name no longer carries it: the
        literal "thirteen" went stale the moment the roster grew, and a test
        whose name disagrees with its assertion is worse than one with no name at
        all. Bump the number with the registry entry that moves it.
        """
        assert len(registry) == 18, sorted(registry)

    def test_the_base_default_is_none(self):
        """An adapter that measured nothing declares nothing, by inheriting."""
        assert LLMProvider.internal_retries is None

    @pytest.mark.parametrize("name", _envelope_sweep_names())
    def test_every_adapter_declares_the_attribute(self, registry, name):
        provider = registry[name]

        assert hasattr(provider, "internal_retries")

    @pytest.mark.parametrize("name", _envelope_sweep_names())
    def test_a_declared_envelope_satisfies_the_ladders_protocol(self, registry, name):
        """The seam is structural: three attributes, read off whatever is there."""
        envelope = registry[name].internal_retries
        if envelope is None:
            return

        assert isinstance(envelope.max_requests, int) and envelope.max_requests >= 1
        assert envelope.max_wall_s is None or envelope.max_wall_s > 0
        assert isinstance(envelope.retries_rate_limit, bool)
        assert isinstance(envelope.configurable_by, tuple)

    @pytest.mark.parametrize(
        "name",
        [
            n for n in _envelope_sweep_names()
            if n not in ("claude-code", "codex", "copilot", "gemini",
                         "goose", "kimi",
                         "muse", "omp", "pi", "qwen-code", "vibe")
        ],
    )
    def test_every_unmeasured_adapter_declares_none(self, registry, name):
        """§7.5 rule 6(c): `None` is 'assume one request', never 'assume the worst'.

        Named without a count for the same reason as the sweep above: the
        unmeasured roster shrinks every time somebody runs §6.1.1's experiments,
        and "the other eleven" was already wrong.
        """
        assert registry[name].internal_retries is None

    # -- the measured rows ------------------------------------------------

    def _with_timeout(self, monkeypatch, seconds):
        """Pin the resolved per-provider slot timeout the envelope derives from."""
        import utils.provider_registry as registry_module

        monkeypatch.setattr(
            registry_module,
            "get_provider_timeout",
            lambda name, anchor=None, **kw: seconds,
        )

    def test_claude_code_declares_the_post_l6_request_count(self, registry):
        """11 at the CLI default → 4 once L6 sets `CLAUDE_CODE_MAX_RETRIES=3`.

        The shipped number is the post-L6 one because L6 has landed and every
        dispatch carries its env; declaring 11 would budget for a configuration
        this harness never runs.
        """
        from utils.providers.claude_code import CLAUDE_CODE_MAX_RETRIES

        envelope = registry["claude-code"].internal_retries

        assert envelope is not None
        assert envelope.max_requests == 4
        assert envelope.max_requests == CLAUDE_CODE_MAX_RETRIES + 1

    def test_claude_code_retries_rate_limits_internally(self, registry):
        """Which is what makes §7.5 rule 6(b) forbid an identical re-dispatch."""
        assert registry["claude-code"].internal_retries.retries_rate_limit is True

    def test_claude_code_names_both_knobs_l6_sets(self, registry):
        envelope = registry["claude-code"].internal_retries

        assert envelope.configurable_by == ("CLAUDE_CODE_MAX_RETRIES", "API_TIMEOUT_MS")

    @pytest.mark.parametrize("slot_s,expected", [(1800, 1200.0), (600, 480.0)])
    def test_claude_code_max_wall_tracks_the_resolved_timeout(
        self, registry, monkeypatch, slot_s, expected
    ):
        """4 × `API_TIMEOUT_MS`, and `API_TIMEOUT_MS` is itself derived (§7.3).

        The plan's two measured points: **1200 s** at a 1800 s slot and 480 s at
        a 600 s one. A constant here would be wrong at one of them.
        """
        self._with_timeout(monkeypatch, slot_s)

        assert registry["claude-code"].internal_retries.max_wall_s == expected

    def test_claude_code_max_wall_is_four_times_the_emitted_env(
        self, registry, monkeypatch
    ):
        """The declaration and the env are one derivation, not two copies of it."""
        self._with_timeout(monkeypatch, 1800)
        provider = registry["claude-code"]

        env = provider.get_env("sonnet")

        assert provider.internal_retries.max_wall_s == (
            4 * int(env["API_TIMEOUT_MS"]) / 1000.0
        )

    def test_claude_code_declares_no_bound_when_the_slot_omits_the_timeout(
        self, registry, monkeypatch
    ):
        """Below ~120 s `get_env` emits no `API_TIMEOUT_MS`, so the wall is unknown.

        `None` is the honest answer and the one rule 4 handles (no admission
        test); inventing a bound would fail slots that would have succeeded.
        """
        self._with_timeout(monkeypatch, 60)
        provider = registry["claude-code"]

        assert "API_TIMEOUT_MS" not in provider.get_env("sonnet")
        assert provider.internal_retries.max_wall_s is None
        assert provider.internal_retries.max_requests == 4

    def test_codex_declares_thirty_requests(self, registry):
        """(stream+1) × (request+1), defaults 4 and 5 — measured, not inferred."""
        envelope = registry["codex"].internal_retries

        assert envelope is not None
        assert envelope.max_requests == 30

    def test_codex_does_not_retry_a_rate_limit(self, registry):
        """One request, then `exceeded retry limit, last status: 429`.

        So the harness-side backoff is the only one a 429 gets, and rule 6(b)
        must NOT suppress it for this adapter.
        """
        assert registry["codex"].internal_retries.retries_rate_limit is False

    def test_codex_declares_its_measured_wall(self, registry):
        """~25 s on 5xx/stream failures inside the one awaited subprocess."""
        assert registry["codex"].internal_retries.max_wall_s == 25.0

    # -- the guard --------------------------------------------------------

    @pytest.mark.parametrize("name", _envelope_sweep_names())
    def test_a_retry_shaped_env_var_requires_a_declared_envelope(self, registry, name):
        """§7.6(b)'s converse, and the reason the declaration ships with a test.

        An adapter that grows a retry env without updating its envelope is a
        budget bug: §7.5 rule 4 would subtract one request's worth of wall clock
        from the deadline while the child quietly makes several. Red for
        claude-code from the moment L6's `CLAUDE_CODE_MAX_RETRIES` landed.
        """
        provider = registry[name]
        shaped = _retry_shaped_env_keys(provider)
        if not shaped:
            return

        assert provider.internal_retries is not None, (
            f"{name}: get_env sets {shaped} but declares no internal_retries "
            f"envelope — §7.5's budget math would assume one request"
        )

    def test_the_ladder_stays_free_of_provider_imports(self):
        """The structural read IS the seam (§7.6(b), `retry_ladder`'s docstring).

        The ladder is pure by construction; an import of `utils.providers` would
        make the policy object depend on the adapter registry it is meant to be
        handed a value from.
        """
        import ast
        import inspect

        import utils.retry_ladder as ladder

        tree = ast.parse(inspect.getsource(ladder))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")

        assert not [name for name in imported if "provider" in name], (
            f"retry_ladder must read the envelope structurally, never import it "
            f"(found {imported})"
        )


def _write_fake_claude(tmp_path, *, help_text, body="", name="claude"):
    """Put an executable fake `claude` on a fresh PATH directory.

    The §5.4.1 gate sniffs the CLI's *own* help, so the fake has to answer
    `--help` for real — which is exactly the trap `tests/mocks/mock_llm.py`
    records: a flag missing from the help text is silently dropped one step
    before argparse would have exited 2 over it.
    """
    import stat

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / name
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"HELP = {help_text!r}\n"
        "argv = sys.argv[1:]\n"
        "if '--help' in argv or '-h' in argv:\n"
        "    sys.stdout.write(HELP)\n"
        "    sys.exit(0)\n"
        f"{body or 'sys.exit(0)'}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return bin_dir


#: The real `claude --help` line, verbatim from 2.1.233.
HELP_WITH_FALLBACK = (
    "Usage: claude [options] [command] [prompt]\n"
    "Options:\n"
    "  -p, --print                           Print response and exit\n"
    "  --model <model>                       Model for the session\n"
    "  --fallback-model <model>              Enable automatic fallback to "
    "specified model(s) when the default model is overloaded or not available "
    "(only works with --print)\n"
    "  --allowedTools, --allowed-tools <tools...>\n"
    "  --disallowedTools, --disallowed-tools <tools...>\n"
)
HELP_WITHOUT_FALLBACK = (
    "Usage: claude [options] [command] [prompt]\n"
    "Options:\n"
    "  -p, --print                           Print response and exit\n"
    "  --model <model>                       Model for the session\n"
    "  --allowedTools, --allowed-tools <tools...>\n"
    "  --disallowedTools, --disallowed-tools <tools...>\n"
)


class TestClaudeCodeFallbackModel:
    """§3.6 (task L7) — `--fallback-model`, default off, behind a per-flag gate."""

    PROMPT = "Review this code"

    @pytest.fixture
    def provider(self):
        return ClaudeCodeProvider()

    @pytest.fixture(autouse=True)
    def _clear_gate_cache(self):
        import utils.providers.claude_code as adapter

        adapter.reset_flag_gate_cache()
        yield
        adapter.reset_flag_gate_cache()

    def _configure(self, monkeypatch, provider, mapping, default=None, models=None):
        block = {
            "models": models if models is not None
            else ["sonnet", "opus", "fable", "haiku"],
            "fallback_models": mapping,
        }
        if default is not None:
            block["fallback_models_default"] = default
        monkeypatch.setattr(
            type(provider), "_provider_config", lambda self: block
        )

    def _gate_open(self, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "PATH",
            str(_write_fake_claude(tmp_path, help_text=HELP_WITH_FALLBACK))
            + os.pathsep
            + os.environ["PATH"],
        )

    # ── the off switch ────────────────────────────────────────────────────
    def test_the_shipped_config_emits_no_flag_at_all(self, provider):
        """Nothing configured ⇒ today's argv, byte-for-byte.

        `--permission-mode acceptEdits` is the generate path's own write grant
        (see `claude_code._render_argv`), not a fallback artefact: it is present
        here precisely because nothing about `--fallback-model` is configured.
        """
        cmd = provider.build_command(self.PROMPT, "sonnet")

        assert cmd == [
            "claude", "-p", "--output-format", "json", "--model", "sonnet",
            "--permission-mode", "acceptEdits",
            self.PROMPT,
        ]

    def test_the_base_config_ships_an_empty_fallback_map(self):
        """`providers.yaml` declares the key so it is discoverable, and empty."""
        from utils.provider_registry import load_base_config

        block = load_base_config()["providers"]["claude-code"]

        assert "fallback_models" in block, (
            "the key must exist in base so a user can find it without reading code"
        )
        assert not block["fallback_models"], "the shipped value is EMPTY"

    def test_an_unconfigured_primary_emits_nothing(self, monkeypatch, provider, tmp_path):
        """Order and arity are the user's; nothing is invented per model."""
        self._gate_open(monkeypatch, tmp_path)
        self._configure(monkeypatch, provider, {"opus": ["haiku"]})

        cmd = provider.build_command(self.PROMPT, "sonnet")

        assert "--fallback-model" not in cmd

    # ── the flag itself ───────────────────────────────────────────────────
    def test_a_configured_primary_emits_the_comma_joined_list(
        self, monkeypatch, provider, tmp_path
    ):
        self._gate_open(monkeypatch, tmp_path)
        self._configure(monkeypatch, provider, {"sonnet": ["opus", "haiku"]})

        cmd = provider.build_command(self.PROMPT, "sonnet")

        assert cmd == [
            "claude", "-p", "--output-format", "json", "--model", "sonnet",
            "--permission-mode", "acceptEdits",
            "--fallback-model", "opus,haiku",
            self.PROMPT,
        ]

    def test_the_effort_suffix_and_the_fallback_coexist(
        self, monkeypatch, provider, tmp_path
    ):
        """The lookup key is the base model, not `model:effort`."""
        self._gate_open(monkeypatch, tmp_path)
        self._configure(monkeypatch, provider, {"fable": ["haiku"]})

        cmd = provider.build_command(self.PROMPT, "fable:max")

        assert cmd == [
            "claude", "-p", "--output-format", "json", "--model", "fable",
            "--effort", "max", "--permission-mode", "acceptEdits",
            "--fallback-model", "haiku",
            self.PROMPT,
        ]

    def test_the_default_list_covers_primaries_with_no_entry(
        self, monkeypatch, provider, tmp_path
    ):
        self._gate_open(monkeypatch, tmp_path)
        self._configure(monkeypatch, provider, {}, default=["haiku"])

        cmd = provider.build_command(self.PROMPT, "sonnet")

        assert cmd[-3:] == ["--fallback-model", "haiku", self.PROMPT]

    # ── the splice position (§3.6, §4.2.2, §9.8) ──────────────────────────
    @pytest.mark.parametrize(
        "profile", ["generate", "agent-readonly", "agent-write"]
    )
    @pytest.mark.parametrize("transport", ["argv", "stdin"])
    def test_the_flag_precedes_the_first_variadic_tool_flag(
        self, monkeypatch, provider, tmp_path, profile, transport
    ):
        """All four rows at once — this is the one edit that makes them right.

        `--allowed-tools`/`--disallowed-tools` take `<tools...>`. Anything spliced
        after them is swallowed, and `build_command_for_transport`'s stdin branch
        returns `cmd[:-1]`, so a flag appended after the prompt would ship
        valueless. Hence: inside `build_command`, ahead of the prompt append.
        """
        self._gate_open(monkeypatch, tmp_path)
        self._configure(monkeypatch, provider, {"sonnet": ["haiku"]})

        if profile == "generate":
            argv, stdin_text = provider.build_command_for_transport(
                self.PROMPT, "sonnet", transport
            )
        else:
            argv, stdin_text = provider.build_agent_command_for_transport(
                self.PROMPT, "sonnet", profile, transport
            )

        assert "--fallback-model" in argv
        flag_at = argv.index("--fallback-model")
        assert argv[flag_at + 1] == "haiku"
        for variadic in ("--allowed-tools", "--disallowed-tools"):
            if variadic in argv:
                assert flag_at < argv.index(variadic), (
                    f"{variadic} is variadic — it would swallow the fallback list"
                )
        if transport == "argv":
            assert argv[-1] == self.PROMPT, "the prompt must still be cmd[-1]"
            assert stdin_text is None
        else:
            assert stdin_text == self.PROMPT
            assert self.PROMPT not in argv
            assert argv[-1] != "--fallback-model", "a valueless flag shipped"

    # ── the §5.4.1 per-flag gate, shipped inside this task ────────────────
    def test_help_without_the_flag_drops_it_and_warns(
        self, monkeypatch, provider, tmp_path, capsys
    ):
        """Unsupported ⇒ omit the flag, one WARN, the run proceeds as today.

        Never `unknown option` and a dead slot: emitting a flag the binary may
        reject is the one outcome the gate may not produce.
        """
        monkeypatch.setenv(
            "PATH",
            str(_write_fake_claude(tmp_path, help_text=HELP_WITHOUT_FALLBACK))
            + os.pathsep + os.environ["PATH"],
        )
        self._configure(monkeypatch, provider, {"sonnet": ["haiku"]})

        cmd = provider.build_command(self.PROMPT, "sonnet")

        assert "--fallback-model" not in cmd
        assert cmd[-1] == self.PROMPT
        warning = capsys.readouterr().err
        assert "--fallback-model" in warning
        assert "WARN" in warning.upper()

    def test_an_absent_binary_is_undecidable_and_declines_the_flag(
        self, monkeypatch, provider, tmp_path, capsys
    ):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        self._configure(monkeypatch, provider, {"sonnet": ["haiku"]})

        cmd = provider.build_command(self.PROMPT, "sonnet")

        assert "--fallback-model" not in cmd
        assert "--fallback-model" in capsys.readouterr().err

    def test_a_failing_help_probe_declines_the_flag(
        self, monkeypatch, provider, tmp_path
    ):
        """Non-zero exit / empty help resolve to unsupported, never to a crash."""
        bin_dir = _write_fake_claude(tmp_path, help_text="")
        (bin_dir / "claude").write_text(
            "#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n", encoding="utf-8"
        )
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
        self._configure(monkeypatch, provider, {"sonnet": ["haiku"]})

        cmd = provider.build_command(self.PROMPT, "sonnet")

        assert "--fallback-model" not in cmd

    def test_the_help_is_read_once_per_process(
        self, monkeypatch, provider, tmp_path
    ):
        """One help spawn per CLI per process, memoised on the resolved path."""
        counter = tmp_path / "help-calls"
        bin_dir = _write_fake_claude(
            tmp_path,
            help_text=HELP_WITH_FALLBACK,
            body=(
                "open(os.environ['MLLM_HELP_COUNTER'], 'a').write('x')\n"
                "sys.exit(0)"
            ),
        )
        # Count the help path too, by counting every invocation of the fake.
        (bin_dir / "claude").write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            f"HELP = {HELP_WITH_FALLBACK!r}\n"
            "open(os.environ['MLLM_HELP_COUNTER'], 'a').write('x')\n"
            "sys.stdout.write(HELP)\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MLLM_HELP_COUNTER", str(counter))
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
        self._configure(monkeypatch, provider, {"sonnet": ["haiku"], "opus": ["haiku"]})

        for _ in range(5):
            provider.build_command(self.PROMPT, "sonnet")
            provider.build_command(self.PROMPT, "opus")

        assert counter.read_text(encoding="utf-8") == "x", (
            "the help sniff must be memoised for the process, not re-spawned "
            "per flag question — a 4-model x 10-phase run would spawn 40 of them"
        )

    def test_the_gate_reads_the_flag_as_a_whole_word(
        self, monkeypatch, provider, tmp_path
    ):
        """A help mentioning `--fallback-models-list` is not this flag."""
        monkeypatch.setenv(
            "PATH",
            str(_write_fake_claude(
                tmp_path,
                help_text="Options:\n  --fallback-model-chain <list>  no\n",
            )) + os.pathsep + os.environ["PATH"],
        )
        self._configure(monkeypatch, provider, {"sonnet": ["haiku"]})

        assert "--fallback-model" not in provider.build_command(self.PROMPT, "sonnet")

    # ── load-time config validation ───────────────────────────────────────
    def test_a_primary_in_its_own_fallback_list_is_refused(self, provider):
        """The CLI accepts it silently (measured), so the validation is ours."""
        from utils.provider_registry import ConfigError
        from utils.providers.claude_code import validate_fallback_models

        with pytest.raises(ConfigError) as excinfo:
            validate_fallback_models({
                "models": ["sonnet", "haiku"],
                "fallback_models": {"sonnet": ["haiku", "sonnet"]},
            })

        assert "sonnet" in str(excinfo.value)

    @pytest.mark.parametrize(
        "mapping,needle",
        [
            ({"sonnet": ["haiku", "haiku"]}, "haiku"),      # repeated entry
            ({"sonnet": [""]}, "sonnet"),                   # empty string
            ({"sonnet": ["gpt-4"]}, "gpt-4"),               # undeclared model
            ({"sonnet": "haiku"}, "sonnet"),                # not a list
            ({"sonnet": []}, "sonnet"),                     # empty list
        ],
    )
    def test_every_malformed_list_is_refused_naming_the_key(self, mapping, needle):
        from utils.provider_registry import ConfigError
        from utils.providers.claude_code import validate_fallback_models

        with pytest.raises(ConfigError) as excinfo:
            validate_fallback_models(
                {"models": ["sonnet", "opus", "haiku"], "fallback_models": mapping}
            )

        assert needle in str(excinfo.value)

    def test_a_valid_configuration_passes(self):
        from utils.providers.claude_code import validate_fallback_models

        validate_fallback_models({
            "models": ["sonnet", "opus", "haiku"],
            "fallback_models": {"sonnet": ["opus", "haiku"]},
            "fallback_models_default": ["haiku"],
        })

    def test_validation_runs_on_the_whole_block_not_just_the_dispatched_key(
        self, monkeypatch, provider, tmp_path
    ):
        """`at load, not at dispatch`: a bad entry for B fails a run of A."""
        from utils.provider_registry import ConfigError

        self._gate_open(monkeypatch, tmp_path)
        self._configure(monkeypatch, provider, {"opus": ["opus"]})

        with pytest.raises(ConfigError):
            provider.build_command(self.PROMPT, "sonnet")


class TestClaudeCodeServedModel:
    """§3.6 — attribution is mandatory whenever a fallback is configured."""

    @pytest.fixture
    def provider(self):
        return ClaudeCodeProvider()

    def _envelope(self, model_usage, result="[]"):
        return json.dumps({
            "type": "result",
            "result": result,
            "modelUsage": model_usage,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "total_cost_usd": 0.01,
        })

    def test_served_model_is_read_from_model_usage(self, provider):
        stdout = self._envelope({"claude-haiku-4-5-20251001": {"outputTokens": 5}})

        assert provider.parse_served_model(stdout, "") == "claude-haiku-4-5-20251001"

    def test_served_model_rides_the_usage_channel(self, provider):
        stdout = self._envelope({"claude-haiku-4-5-20251001": {"outputTokens": 5}})

        usage = provider.parse_usage(stdout, "")

        assert usage["served_model"] == "claude-haiku-4-5-20251001"
        # The numeric contract is unchanged: absent means "not reported".
        assert usage["input_tokens"] == 10 and usage["total_tokens"] == 15

    def test_a_fallback_that_fired_is_labelled_and_warned(self, provider):
        stdout = self._envelope({"claude-haiku-4-5-20251001": {"outputTokens": 5}})

        served, is_fallback, warning = provider.resolve_served_model(
            stdout, "sonnet", ["haiku"]
        )

        assert served == "claude-haiku-4-5-20251001"
        assert is_fallback is True
        assert warning and "sonnet" in warning and "haiku" in warning

    def test_the_primary_serving_its_own_turn_is_not_a_fallback(self, provider):
        stdout = self._envelope({"claude-sonnet-4-5-20250929": {"outputTokens": 5}})

        served, is_fallback, warning = provider.resolve_served_model(
            stdout, "sonnet", ["haiku"]
        )

        assert served == "claude-sonnet-4-5-20250929"
        assert is_fallback is False
        assert warning is None

    def test_absent_model_usage_records_the_requested_model_and_warns(self, provider):
        """Read, never depended on: attribution never fails a good attempt."""
        stdout = json.dumps({"type": "result", "result": "[]"})

        served, is_fallback, warning = provider.resolve_served_model(
            stdout, "sonnet", ["haiku"]
        )

        assert served == "sonnet"
        assert is_fallback is False
        assert warning is not None

    def test_a_model_outside_the_configured_set_records_the_request(self, provider):
        stdout = self._envelope({"some-other-model": {"outputTokens": 5}})

        served, is_fallback, warning = provider.resolve_served_model(
            stdout, "sonnet", ["haiku"]
        )

        assert served == "sonnet"
        assert is_fallback is False
        assert warning and "some-other-model" in warning

    def test_unparseable_output_never_raises(self, provider):
        assert provider.parse_served_model("not json at all", "") is None
        assert provider.resolve_served_model("", "sonnet", [])[0] == "sonnet"


class TestClaudeCodeAgentPromptPosition:
    """§9.8 (task L8) — the prompt may not sit after a variadic tool flag.

    `--allowed-tools` / `--disallowed-tools` take `<tools...>`. The adapter used
    to emit the prompt immediately after `--disallowed-tools`, so the variadic
    swallowed it: the adapter's own rendered argv, run verbatim at 2.1.233, exits
    1 with `Error: Input must be provided either through stdin or as a prompt
    argument when using --print` — zero requests.
    """

    PROMPT = "Review this code"

    @pytest.fixture
    def provider(self):
        return ClaudeCodeProvider()

    @pytest.mark.parametrize("profile", ["agent-readonly", "agent-write"])
    def test_the_prompt_is_not_adjacent_to_the_variadic_flag(self, provider, profile):
        argv = provider.build_agent_command(self.PROMPT, "sonnet", profile)

        prompt_at = argv.index(self.PROMPT)
        last_variadic = max(
            argv.index(flag)
            for flag in ("--allowed-tools", "--disallowed-tools")
            if flag in argv
        )
        between = argv[last_variadic + 1:prompt_at]
        assert "--" in between, (
            "`--disallowed-tools` takes <tools...>, so every token after it up to "
            "a separator is part of its VALUE and the prompt is swallowed — this "
            "is the §9.8 defect (exit 1, zero requests, at 2.1.233)"
        )

    @pytest.mark.parametrize("profile", ["agent-readonly", "agent-write"])
    def test_a_separator_shields_the_prompt_and_it_stays_last(self, provider, profile):
        """`--` before the prompt: verified working at 2.1.233 (1 request, exit 0).

        The prompt stays `argv[-1]`, which is what every `cmd[:-1]` override and
        the cross-adapter `agent-write` invariant in this file rely on.
        """
        argv = provider.build_agent_command(self.PROMPT, "sonnet", profile)

        assert argv[-1] == self.PROMPT
        assert argv[-2] == "--"
        assert argv.count("--") == 1

    @pytest.mark.parametrize("profile", ["agent-readonly", "agent-write"])
    def test_the_stdin_argv_drops_the_separator_with_the_prompt(
        self, provider, profile
    ):
        """No positional left ⇒ nothing for the separator to shield.

        This is what keeps `tests/container/expected_argv.yaml`'s two claude-code
        agent rows (both `transport: stdin`) byte-identical to what they already
        assert — the separator exists only on the argv transport.
        """
        from utils.providers.claude_code import (
            READONLY_DENIED_TOOLS,
            WRITE_DENIED_TOOLS,
        )

        argv, stdin_text = provider.build_agent_command_for_transport(
            self.PROMPT, "sonnet", profile, "stdin"
        )

        assert stdin_text == self.PROMPT
        assert self.PROMPT not in argv
        assert "--" not in argv
        assert argv[-1] in (
            ",".join(READONLY_DENIED_TOOLS),
            ",".join(WRITE_DENIED_TOOLS),
        )

    def test_the_generate_profile_is_unaffected(self, provider):
        """`build_command` never had the defect — `--model` is not variadic."""
        cmd = provider.build_command(self.PROMPT, "sonnet")

        assert "--" not in cmd
        assert cmd[-1] == self.PROMPT


class TestClaudeCodeGroupA3:
    """§14.4 group gate A3 — L6 and L7 proved together in one run.

    Either half alone is the inert state the group exists to prevent: a fallback
    flag under fewer than three retries never reaches its overload path, and a
    retry budget with no fallback list buys nothing on a retired model alias.
    """

    PROMPT = "Review this code"

    @pytest.fixture(autouse=True)
    def _clear_gate_cache(self):
        import utils.providers.claude_code as adapter

        adapter.reset_flag_gate_cache()
        yield
        adapter.reset_flag_gate_cache()

    def test_the_fallback_fires_on_a_404_with_the_retry_budget_in_the_child_env(
        self, tmp_path, monkeypatch, capsys
    ):
        """One run, both halves.

        The fake `claude` behaves the way the measured 2.1.233 does on the
        immediate-trigger case: a 404 on the primary falls back on the *first*
        error, and the envelope's `modelUsage` names the model that really served
        the turn. It records the env it was launched with, which is where the L6
        half is observed.
        """
        from utils.llm_client import invoke_with_provider
        from utils.provider_registry import get_provider

        record = tmp_path / "child.json"
        body = (
            "argv = sys.argv[1:]\n"
            "fallbacks = ''\n"
            "if '--fallback-model' in argv:\n"
            "    fallbacks = argv[argv.index('--fallback-model') + 1]\n"
            "json.dump({'argv': argv, 'env': dict(os.environ)},\n"
            "          open(os.environ['MLLM_CHILD_RECORD'], 'w'))\n"
            "if not fallbacks:\n"
            "    sys.stderr.write('API Error: 404 model not found\\n')\n"
            "    sys.exit(1)\n"
            "served = 'claude-' + fallbacks.split(',')[0] + '-4-5-20251001'\n"
            "sys.stdout.write(json.dumps({\n"
            "    'type': 'result', 'result': '[]',\n"
            "    'modelUsage': {served: {'outputTokens': 7}},\n"
            "    'usage': {'input_tokens': 3, 'output_tokens': 7},\n"
            "}))\n"
            "sys.exit(0)\n"
        )
        bin_dir = _write_fake_claude(
            tmp_path, help_text=HELP_WITH_FALLBACK, body=body
        )
        monkeypatch.setenv("MLLM_CHILD_RECORD", str(record))
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])

        provider = get_provider("claude-code")
        monkeypatch.setattr(
            type(provider),
            "_provider_config",
            lambda self: {
                "models": ["sonnet", "haiku"],
                "fallback_models": {"sonnet": ["haiku"]},
            },
        )

        result = invoke_with_provider(self.PROMPT, model_spec="claude-code:sonnet")

        assert result["success"] is True, result
        child = json.loads(record.read_text(encoding="utf-8"))

        # L7: the fallback was offered, and it is what answered.
        assert "--fallback-model" in child["argv"]
        assert child["argv"][child["argv"].index("--fallback-model") + 1] == "haiku"

        # L6: the retry budget was in the child's environment, at or above the
        # floor of 3 — below it the overload path above is silently dead.
        assert int(child["env"]["CLAUDE_CODE_MAX_RETRIES"]) >= 3
        assert int(child["env"]["API_TIMEOUT_MS"]) > 0
        assert "ANTHROPIC_MAX_RETRIES" not in child["env"]
        assert "ANTHROPIC_REQUEST_TIMEOUT_SECONDS" not in child["env"]

        # The attribution the report has to carry.
        stdout_envelope = json.dumps({
            "type": "result", "result": "[]",
            "modelUsage": {"claude-haiku-4-5-20251001": {"outputTokens": 7}},
        })
        served, is_fallback, warning = provider.resolve_served_model(
            stdout_envelope, "sonnet", ["haiku"]
        )
        assert is_fallback and served == "claude-haiku-4-5-20251001"
        assert warning and "sonnet" in warning


class TestCursorAgentProvider:
    """Tests for the CursorAgentProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a CursorAgentProvider instance."""
        return CursorAgentProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "cursor-agent"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_cursor_agent_parse_json_wrapper(self, provider):
        """Parse cursor-agent output with JSON wrapper format.

        Cursor-agent with --output-format json returns responses wrapped in:
        {"type":"result","result":"..."}
        """
        inner_data = [
            {"title": "Test suggestion", "desc": "Description", "importance": "high"}
        ]
        wrapper = {"type": "result", "result": json.dumps(inner_data)}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_cursor_agent_parse_json_wrapper_with_object_result(self, provider):
        """Parse cursor-agent output when inner result is already an object."""
        inner_data = {"key": "value", "nested": {"a": 1}}
        wrapper = {"type": "result", "result": json.dumps(inner_data)}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_cursor_agent_parse_code_block(self, provider):
        """Extract JSON from markdown code block in cursor-agent output."""
        inner_json = [{"title": "Suggestion 1", "importance": "high"}]
        text_with_codeblock = f"""Here is my analysis:

```json
{json.dumps(inner_json, indent=2)}
```

This concludes the review."""
        wrapper = {"type": "result", "result": text_with_codeblock}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_cursor_agent_parse_code_block_without_json_label(self, provider):
        """Extract JSON from code block without json language label."""
        inner_json = {"status": "ok", "items": [1, 2, 3]}
        text_with_codeblock = f"""Output:
```
{json.dumps(inner_json)}
```
"""
        wrapper = {"type": "result", "result": text_with_codeblock}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_cursor_agent_parse_raw_json_in_text(self, provider):
        """Extract raw JSON from text without code blocks."""
        inner_json = [{"id": 1}, {"id": 2}]
        text_with_json = f"The results are: {json.dumps(inner_json)} end of results."
        wrapper = {"type": "result", "result": text_with_json}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_cursor_agent_parse_direct_array(self, provider):
        """Parse direct JSON array output (no wrapper)."""
        data = [{"item": 1}, {"item": 2}]
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_cursor_agent_parse_empty_result(self, provider):
        """Handle empty result in JSON wrapper."""
        wrapper = {"type": "result", "result": ""}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_cursor_agent_parse_no_json_found(self, provider):
        """Handle output with no extractable JSON."""
        wrapper = {"type": "result", "result": "Just plain text with no JSON"}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result
        assert "raw" in result

    def test_cursor_agent_build_command(self, provider):
        """The one shipping argv (M1/§3.1): ``-f`` out, ``--trust --mode ask`` in.

        The two replacement flags are inseparable (atomic group **A5**): without
        ``--trust`` the launch dies on cursor's workspace-trust gate in 0.43 s,
        and ``--trust`` alone still writes files. They are emitted behind
        §5.4.1's per-flag gate, stubbed here so the assertion is about the argv
        rather than about the developer's installed cursor build. The gate's own
        behaviour — including its refusal — is
        ``tests/test_cursor_argv_and_residue.py``.
        """
        prompt = "Review this code"
        model = "gpt-4"

        with patch.object(
            base_module, "read_help_tokens", lambda _b, **_kw: frozenset({"--trust", "--mode"})
        ), patch.object(cursor_agent_module.shutil, "which", lambda _n: sys.executable):
            cmd = provider.build_command(prompt, model)

        assert cmd == [
            "cursor-agent",
            "--print",
            "--trust",
            "--mode",
            "ask",
            "--output-format",
            "json",
            "--model",
            "gpt-4",
            "Review this code",
        ]

    @patch("shutil.which")
    def test_cursor_agent_is_available_true(self, mock_which, provider):
        """Test is_available returns True when cursor-agent is found."""
        mock_which.return_value = "/usr/local/bin/cursor-agent"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("cursor-agent")

    @patch("shutil.which")
    def test_cursor_agent_is_available_false(self, mock_which, provider):
        """Test is_available returns False when cursor-agent is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("cursor-agent")


class TestGeminiProvider:
    """Tests for the GeminiProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a GeminiProvider instance."""
        return GeminiProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "gemini"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 900

    def test_gemini_parse_response_field(self, provider):
        """Extract JSON from Gemini's response field.

        Gemini returns: {"session_id": "...", "response": "...", "stats": {...}}
        """
        inner_data = [
            {"title": "Finding 1", "desc": "Description", "importance": "medium"}
        ]
        wrapper = {
            "session_id": "abc123",
            "response": json.dumps(inner_data),
            "stats": {"tokens": 100}
        }
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_gemini_parse_response_with_object(self, provider):
        """Parse Gemini response containing a JSON object."""
        inner_data = {"status": "complete", "findings": []}
        wrapper = {
            "session_id": "session1",
            "response": json.dumps(inner_data),
            "stats": {}
        }
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_gemini_parse_response_code_block(self, provider):
        """Extract JSON from code block in Gemini response."""
        inner_json = [{"task": "T001", "status": "done"}]
        response_text = f"""Analysis complete:

```json
{json.dumps(inner_json)}
```
"""
        wrapper = {"session_id": "s1", "response": response_text, "stats": {}}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_gemini_parse_response_plain_text_with_json(self, provider):
        """Extract embedded JSON from plain text response."""
        inner_json = {"result": "success"}
        response_text = f"Here is the output: {json.dumps(inner_json)} as requested."
        wrapper = {"session_id": "s2", "response": response_text, "stats": {}}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_gemini_parse_non_json_response(self, provider):
        """Handle response that contains no JSON."""
        wrapper = {
            "session_id": "s3",
            "response": "This is just plain text without any JSON.",
            "stats": {}
        }
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result
        assert "raw" in result

    def test_gemini_parse_malformed_wrapper(self, provider):
        """Handle malformed JSON in stdout."""
        stdout = "not valid json at all"

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_gemini_parse_direct_json_fallback(self, provider):
        """Fallback to direct JSON parsing when wrapper is invalid."""
        # If stdout is invalid JSON but contains extractable JSON
        stdout = "Error prefix [1, 2, 3] suffix"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_gemini_build_command(self, provider):
        """Test command building for Gemini CLI."""
        prompt = "Analyze this"
        model = "gemini-pro"

        cmd = provider.build_command(prompt, model)

        assert cmd == ["gemini", "--output-format", "json", "--model", "gemini-pro", "Analyze this"]

    @patch("shutil.which")
    def test_gemini_is_available_true(self, mock_which, provider):
        """Test is_available returns True when gemini is found."""
        mock_which.return_value = "/usr/bin/gemini"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("gemini")

    @patch("shutil.which")
    def test_gemini_is_available_false(self, mock_which, provider):
        """Test is_available returns False when gemini is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("gemini")


class TestGrokProvider:
    """Tests for the GrokProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a GrokProvider instance."""
        return GrokProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "grok"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_grok_parse_text_field(self, provider):
        """Extract JSON from Grok's text field.

        Grok returns: {"text": "...", "stopReason": "...", "sessionId": "..."}
        """
        inner_data = [
            {"title": "Finding 1", "desc": "Description", "importance": "medium"}
        ]
        wrapper = {
            "text": json.dumps(inner_data),
            "stopReason": "EndTurn",
            "sessionId": "abc123",
        }
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_grok_parse_text_with_object(self, provider):
        """Parse Grok text field containing a JSON object."""
        inner_data = {"status": "complete", "findings": []}
        wrapper = {
            "text": json.dumps(inner_data),
            "stopReason": "EndTurn",
        }
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_grok_parse_text_code_block(self, provider):
        """Extract JSON from code block in Grok text."""
        inner_json = [{"task": "T001", "status": "done"}]
        response_text = f"""Analysis complete:

```json
{json.dumps(inner_json)}
```
"""
        wrapper = {"text": response_text, "stopReason": "EndTurn"}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_grok_parse_text_prose_with_json(self, provider):
        """Extract embedded JSON when prose precedes it in the text field."""
        inner_json = {"result": "success"}
        response_text = f"I'll return only the JSON. {json.dumps(inner_json)}"
        wrapper = {"text": response_text, "stopReason": "EndTurn"}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_grok_parse_non_json_text(self, provider):
        """Handle text that contains no JSON."""
        wrapper = {
            "text": "This is just plain text without any JSON.",
            "stopReason": "EndTurn",
        }
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result
        assert "raw" in result

    def test_grok_parse_empty_text(self, provider):
        """Handle an empty text field."""
        stdout = json.dumps({"text": "", "stopReason": "EndTurn"})

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_grok_parse_malformed_wrapper(self, provider):
        """Handle malformed JSON in stdout."""
        stdout = "not valid json at all"

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_grok_parse_direct_json_fallback(self, provider):
        """Fallback to extraction when wrapper is invalid but JSON is embedded."""
        stdout = "Error prefix [1, 2, 3] suffix"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_grok_build_command(self, provider):
        """Test command building for Grok Build CLI.

        The `-p <prompt>` pair TRAILS (§9.4): grok used to render the prompt at
        index 4, the one adapter that put a flag after it, which silently broke
        the `argv[-1] is the prompt` rule `cmd[:-1]` and
        `build_agent_command_for_transport` rely on.
        """
        prompt = "Analyze this"
        model = "grok-4.5"

        cmd = provider.build_command(prompt, model)

        assert cmd == [
            "grok",
            "--no-auto-update",
            "--always-approve",
            "--output-format",
            "json",
            "-m",
            "grok-4.5",
            "-p",
            "Analyze this",
        ]

    def test_grok_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the seven valid efforts."""
        from utils.providers.grok import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

    def test_grok_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --reasoning-effort pair after -m <base>.

        The effort pair goes BEFORE the trailing `-p <prompt>`, so the prompt is
        still argv[-1] (§9.4).
        """
        cmd = provider.build_command("Analyze this", "grok-4.5:low")

        assert cmd == [
            "grok",
            "--no-auto-update",
            "--always-approve",
            "--output-format",
            "json",
            "-m",
            "grok-4.5",
            "--reasoning-effort",
            "low",
            "-p",
            "Analyze this",
        ]

    def test_grok_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --reasoning-effort form."""
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            cmd = provider.build_command("Analyze this", f"grok-4.5:{effort}")

            assert cmd == [
                "grok",
                "--no-auto-update",
                "--always-approve",
                "--output-format",
                "json",
                "-m",
                "grok-4.5",
                "--reasoning-effort",
                effort,
                "-p",
                "Analyze this",
            ], f"effort {effort!r} did not build the expected command"

    def test_grok_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --reasoning-effort flag."""
        for model in ("grok-4.5", "grok-4.5:turbo"):
            cmd = provider.build_command("Analyze this", model)

            assert cmd == [
                "grok",
                "--no-auto-update",
                "--always-approve",
                "--output-format",
                "json",
                "-m",
                model,
                "-p",
                "Analyze this",
            ], f"model {model!r} was not passed through verbatim"

    @patch("shutil.which")
    def test_grok_is_available_true(self, mock_which, provider):
        """Test is_available returns True when grok is found."""
        mock_which.return_value = "/usr/bin/grok"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("grok")

    @patch("shutil.which")
    def test_grok_is_available_false(self, mock_which, provider):
        """Test is_available returns False when grok is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("grok")


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


def _oc_event(event_type: str, **rest):
    """Build one opencode event with the envelope the real CLI always emits."""
    return json.dumps({
        "type": event_type,
        "timestamp": 1785403656879,
        "sessionID": "ses_TEST",
        **rest,
    })


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


class TestKiloCodeProvider:
    """Tests for the KiloCodeProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a KiloCodeProvider instance."""
        return KiloCodeProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "kilocode"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    # -- Dual binary name (`kilocode` / `kilo`) -----------------------------
    #
    # Upstream ships the CLI under both spellings. This adapter knew only
    # `kilocode`, so on a `kilo`-only machine `is_available()` was False and the
    # provider was invisible to the fan-out — not degraded, absent. The three
    # cases below are the whole contract: canonical present (today's argv,
    # byte-identical), only the alias present (argv[0] moves, nothing else
    # does), and neither present (not available).

    @staticmethod
    def _which_finding(*names):
        """A `shutil.which` stub that resolves exactly `names`."""
        found = set(names)
        return lambda name: f"/usr/local/bin/{name}" if name in found else None

    @patch("shutil.which")
    def test_is_available_true(self, mock_which, provider):
        """Test is_available returns True when kilocode is found."""
        mock_which.return_value = "/usr/local/bin/kilocode"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("kilocode")

    @patch("shutil.which")
    def test_is_available_true_for_the_kilo_spelling(self, mock_which, provider):
        """A `kilo`-only install is available — the defect this pins."""
        mock_which.side_effect = self._which_finding("kilo")

        assert provider.is_available() is True
        assert [c.args[0] for c in mock_which.call_args_list] == ["kilocode", "kilo"]

    @patch("shutil.which")
    def test_is_available_false(self, mock_which, provider):
        """is_available is False only when NEITHER spelling resolves."""
        mock_which.return_value = None

        assert provider.is_available() is False
        assert [c.args[0] for c in mock_which.call_args_list] == ["kilocode", "kilo"]

    @patch("shutil.which")
    def test_build_command_uses_the_kilocode_spelling_when_present(self, mock_which, provider):
        """Canonical install: argv is byte-identical to what shipped before."""
        mock_which.side_effect = self._which_finding("kilocode")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd == [
            "kilocode", "run", "--auto", "-m", "openrouter/z-ai/glm-5.2", "Review this code",
        ]

    @patch("shutil.which")
    def test_build_command_prefers_kilocode_when_both_resolve(self, mock_which, provider):
        """Both installed ⇒ `kilocode` wins, so argv does not move under the user."""
        mock_which.side_effect = self._which_finding("kilocode", "kilo")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd[0] == "kilocode"

    @patch("shutil.which")
    def test_build_command_moves_only_argv0_for_the_kilo_spelling(self, mock_which, provider):
        """`kilo`-only: argv[0] moves and every other element is identical.

        Asserted against the canonical argv element-by-element rather than as a
        literal, so this cannot drift from the row above.
        """
        mock_which.side_effect = self._which_finding("kilocode")
        canonical = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        mock_which.side_effect = self._which_finding("kilo")
        aliased = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert aliased[0] == "kilo"
        assert aliased[1:] == canonical[1:]

    @patch("shutil.which")
    def test_build_command_variant_branch_also_resolves_the_binary(self, mock_which, provider):
        """The `--variant` branch is the second hardcoded literal §2.3 names."""
        mock_which.side_effect = self._which_finding("kilo")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2:xhigh")

        assert cmd == [
            "kilo", "run", "--auto", "-m", "openrouter/z-ai/glm-5.2",
            "--variant", "xhigh", "Review this code",
        ]

    @patch("shutil.which")
    def test_build_command_falls_back_to_the_canonical_name_with_neither_installed(
        self, mock_which, provider
    ):
        """Neither installed ⇒ the canonical spelling, so the argv contract holds.

        `is_available()` is False here, so this argv never reaches a subprocess;
        it is what the argv-contract tests, the housekeeping MODELS table and
        `seed_expected_argv.py` render on a binary-free host.
        """
        mock_which.return_value = None

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd[0] == "kilocode"

    def test_build_command(self, provider):
        """Test command building for Kilo Code CLI."""
        prompt = "Review this code"
        model = "openrouter/moonshotai/kimi-k2.5"

        cmd = provider.build_command(prompt, model)

        assert cmd == ["kilocode", "run", "--auto", "-m", "openrouter/moonshotai/kimi-k2.5", "Review this code"]

    def test_build_command_different_model(self, provider):
        """Test command building with a different model."""
        prompt = "Analyze this"
        model = "kilo/z-ai/glm-5:free"

        cmd = provider.build_command(prompt, model)

        assert cmd == ["kilocode", "run", "--auto", "-m", "kilo/z-ai/glm-5:free", "Analyze this"]

    def test_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the nine valid efforts."""
        from utils.providers.kilocode import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "thinking", "instant",
        }

    def test_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --variant pair after -m <base>, prompt last."""
        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2:xhigh")

        assert cmd == [
            "kilocode",
            "run",
            "--auto",
            "-m",
            "openrouter/z-ai/glm-5.2",
            "--variant",
            "xhigh",
            "Review this code",
        ]

    def test_build_command_thinking_variant_suffix(self, provider):
        """The kilocode-specific :thinking suffix maps to --variant thinking."""
        cmd = provider.build_command("Review this code", "openrouter/moonshotai/kimi-k2.7-code:thinking")

        assert cmd == [
            "kilocode",
            "run",
            "--auto",
            "-m",
            "openrouter/moonshotai/kimi-k2.7-code",
            "--variant",
            "thinking",
            "Review this code",
        ]

    def test_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --variant form."""
        efforts = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "thinking", "instant")
        for effort in efforts:
            cmd = provider.build_command("Review this code", f"openrouter/z-ai/glm-5.2:{effort}")

            assert cmd == [
                "kilocode",
                "run",
                "--auto",
                "-m",
                "openrouter/z-ai/glm-5.2",
                "--variant",
                effort,
                "Review this code",
            ], f"effort {effort!r} did not build the expected command"

    def test_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --variant flag.

        Critically, openrouter ":free" variants are NOT effort suffixes:
        the whole string stays the model.
        """
        for model in ("openrouter/z-ai/glm-5.2", "openrouter/deepseek/deepseek-r1:free"):
            cmd = provider.build_command("Review this code", model)

            assert cmd == [
                "kilocode",
                "run",
                "--auto",
                "-m",
                model,
                "Review this code",
            ], f"model {model!r} was not passed through verbatim"

    def test_parse_output_direct_json_array(self, provider):
        """Parse direct JSON array output."""
        data = [{"item": 1}, {"item": 2}]
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_parse_output_direct_json_object(self, provider):
        """Parse direct JSON object output."""
        data = {"status": "ok", "items": [1, 2, 3]}
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_parse_output_code_block(self, provider):
        """Extract JSON from markdown code block."""
        inner_json = [{"title": "Suggestion 1", "importance": "high"}]
        stdout = f"""Here is my analysis:

```json
{json.dumps(inner_json, indent=2)}
```

This concludes the review."""

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_code_block_without_json_label(self, provider):
        """Extract JSON from code block without json language label."""
        inner_json = {"status": "ok", "items": [1, 2, 3]}
        stdout = f"""Output:
```
{json.dumps(inner_json)}
```
"""

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_embedded_json(self, provider):
        """Extract embedded JSON from plain text."""
        inner_json = [{"id": 1}, {"id": 2}]
        stdout = f"The results are: {json.dumps(inner_json)} end of results."

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_empty(self, provider):
        """Handle empty output."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert "Empty output" in result["error"]

    def test_parse_output_whitespace_only(self, provider):
        """Handle whitespace-only output."""
        result = provider.parse_output("   \n\t  ", "")

        assert result["success"] is False
        assert "Empty output" in result["error"]

    def test_parse_output_no_json(self, provider):
        """Handle output with no extractable JSON."""
        stdout = "Just plain text with no JSON at all."

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_parse_output_whitespace_handling(self, provider):
        """Test that whitespace is handled correctly."""
        data = [1, 2, 3]
        stdout = f"  \n  {json.dumps(data)}  \n  "

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data


class TestKiloCodeDefaultModelSpecs:
    """Every kilocode `:variant` in `defaults.models` must be one the model publishes.

    **This is the only place the mistake can be caught.** `kilocode run
    --variant <name>` SILENTLY IGNORES an unsupported name — no warning, no
    non-zero exit, no marker in the stream — so a fan-out row asking for a
    variant the model does not have runs at the model's default reasoning level
    and reports success. Nothing at run time distinguishes it from a row that
    got what it asked for.

    Four of the six kilocode rows of `defaults.models` were in exactly that
    state (§13.4 #2): `kimi-k3:thinking`, `nemotron-3-ultra-550b-a55b:xhigh`,
    `minimax-m3:xhigh`, `mimo-v2.5-pro:xhigh`.

    The published set is a **captured** fixture, not a live probe: `kilocode
    models <backend> --verbose` is credential-free but it is still a network
    call, and a test that reaches the internet is a test that fails for reasons
    that are not about this repository. `tests/fixtures/kilocode/
    model_variants.json` carries the capture plus its own provenance, and this
    test refuses to pass over a model the capture does not cover — a stale
    fixture must be a failure, not a silent skip, or the guard evaporates the
    first time somebody adds a row.
    """

    VARIANTS_FIXTURE = (Path(__file__).parent / "fixtures" / "kilocode"
                        / "model_variants.json")

    @staticmethod
    def _kilocode_specs():
        """The `kilocode:<model>` rows of the BASE `defaults.models`.

        `load_base_config` and not `load_config`: a developer's own
        `<git-root>/.multi-llm/providers.yaml` deep-merges over base and would
        otherwise decide whether this test passes.
        """
        from utils.provider_registry import load_base_config

        models = (load_base_config().get("defaults") or {}).get("models") or []
        return [spec[len("kilocode:"):] for spec in models
                if isinstance(spec, str) and spec.startswith("kilocode:")]

    def _published(self):
        return json.loads(
            self.VARIANTS_FIXTURE.read_text(encoding="utf-8"))["variants"]

    def test_fixture_covers_every_kilocode_default(self):
        """A spec the capture does not know about fails loudly, naming the refresh."""
        published = self._published()
        specs = self._kilocode_specs()

        assert specs, "defaults.models lists no kilocode rows — did the key move?"
        for spec in specs:
            from utils.providers.kilocode import REASONING_EFFORTS

            base_model, _variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            assert base_model in published, (
                f"no captured `variants` for kilocode model {base_model!r}. "
                f"Refresh {self.VARIANTS_FIXTURE.name} with the commands in its "
                "`_provenance` block (`kilocode models <backend> --verbose`, "
                "credential-free) — an uncovered model is an unchecked variant, "
                "which the CLI would ignore in silence."
            )

    def test_every_default_variant_is_published_by_its_model(self):
        """The L11 #2 assertion: red on four of six before the catalog was corrected."""
        published = self._published()
        from utils.providers.kilocode import REASONING_EFFORTS

        offenders = []
        for spec in self._kilocode_specs():
            base_model, variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            if variant is None:
                continue
            allowed = published.get(base_model)
            if allowed is not None and variant not in allowed:
                offenders.append(
                    f"  kilocode:{spec} — {base_model} publishes {sorted(allowed)}")

        assert not offenders, (
            "defaults.models asks kilocode for a variant the model does not "
            "publish. `kilocode run --variant` ignores an unsupported name "
            "SILENTLY, so these rows run at the model's default effort and "
            "still report success:\n" + "\n".join(offenders)
        )

    @staticmethod
    def _catalog_specs():
        """The `providers.kilocode.models` rows — the *selectable* catalog.

        Separate from `defaults.models` because they fail differently, not
        because the rule differs: a defaults row is silently mis-run on every
        fan-out, while a catalog row is silently mis-run only once a user or
        `--models` picks it. Same silent-ignore mechanism, same guard.
        """
        from utils.provider_registry import load_base_config

        providers = load_base_config().get("providers") or {}
        return list((providers.get("kilocode") or {}).get("models") or [])

    def test_every_catalog_variant_is_published_by_its_model(self):
        """Five catalog rows carried the same defect the defaults did.

        Found while L11 was correcting `defaults.models` and fixed with the
        same rule (the top variant the model actually publishes, preserving the
        row's maximum-reasoning intent): `kimi-k3:thinking` → `:max`,
        `minimax-m3:xhigh` → `:thinking`, `qwen3.7-max:xhigh` → `:max`,
        `hy3:xhigh` → `:high`, `step-3.7-flash:xhigh` → `:high`.
        """
        published = self._published()
        from utils.providers.kilocode import REASONING_EFFORTS

        offenders = []
        uncovered = []
        for spec in self._catalog_specs():
            if not isinstance(spec, str):
                continue
            base_model, variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            if variant is None:
                continue
            allowed = published.get(base_model)
            if allowed is None:
                uncovered.append(f"  {base_model}")
                continue
            if variant not in allowed:
                offenders.append(
                    f"  kilocode:{spec} — {base_model} publishes {sorted(allowed)}")

        assert not uncovered, (
            "providers.kilocode.models carries a `:variant` on a model the "
            f"capture does not cover. Refresh {self.VARIANTS_FIXTURE.name} "
            "with the commands in its `_provenance` block:\n"
            + "\n".join(uncovered)
        )
        assert not offenders, (
            "providers.kilocode.models asks for a variant the model does not "
            "publish; `kilocode run --variant` ignores it SILENTLY:\n"
            + "\n".join(offenders)
        )


class TestKimiProvider:
    """Tests for the KimiProvider class (harness-expansion.md §3.2, H4·kimi).

    Every assertion here is anchored to a capture in `tests/fixtures/kimi/`,
    taken from the real binary at 0.36.1 inside the §6 probe container against
    the committed stub — the `tests/fixtures/kilocode/` precedent — and
    re-captured on 2.1.1 (2026-10-08) in the pinned image, through the wrapper,
    by the same probe servers.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "kimi"

    @pytest.fixture
    def provider(self):
        return KimiProvider()

    def _capture(self, stem):
        """One captured run as (stdout, stderr)."""
        out = self.FIXTURES / f"{stem}.stdout.txt"
        err = self.FIXTURES / f"{stem}.stderr.txt"
        return (
            out.read_text(encoding="utf-8") if out.exists() else "",
            err.read_text(encoding="utf-8") if err.exists() else "",
        )

    # -- identity + argv ---------------------------------------------------

    def test_provider_name(self, provider):
        assert provider.name == "kimi"

    def test_is_available_checks_the_kimi_binary(self, provider):
        with patch("shutil.which", return_value="/usr/bin/kimi") as which:
            assert provider.is_available() is True
        which.assert_called_once_with("kimi")

    def test_is_available_false_when_absent(self, provider):
        with patch("shutil.which", return_value=None):
            assert provider.is_available() is False

    def test_build_command_renders_the_measured_argv(self, provider):
        assert provider.build_command("Review this plan", "moonshotai/kimi-k2-thinking") == [
            "kimi", "--output-format", "stream-json", "-p", "Review this plan",
        ]

    def test_the_model_never_reaches_argv(self, provider):
        """The measured contract: `-m <slug>` is a HARD FAILURE at 0.36.1.

        `kimi -m moonshotai/kimi-k2-thinking` exits 1 with `Model "..." is not
        configured in config.toml`, because the KIMI_MODEL_* family synthesizes
        its model under the fixed literal id `__kimi_env_model__` and
        KIMI_MODEL_NAME carries the upstream wire slug. This is the plan's §8
        item 3 rule inverted, so it is pinned rather than left to a comment.
        """
        argv = provider.build_command("p", "moonshotai/kimi-k2-thinking")

        assert "-m" not in argv
        assert "--model" not in argv
        assert "moonshotai/kimi-k2-thinking" not in argv

    def test_the_prompt_is_last(self, provider):
        """`prompt_argv_index = -1`, and the -p pair is what trails."""
        argv = provider.build_command("the prompt", "moonshotai/kimi-k2-thinking")

        assert provider.prompt_argv_index == -1
        assert argv[-2:] == ["-p", "the prompt"]

    def test_an_effort_suffix_is_part_of_the_model_id(self, provider):
        """kimi DECLINES effort: KIMI_MODEL_THINKING_EFFORT is inert on the wire.

        So a `:high` suffix is not stripped — it stays part of the slug and
        travels to KIMI_MODEL_NAME verbatim, which is what makes the failure a
        legible "model not found" rather than a silently downgraded run.
        """
        env = provider.get_env("moonshotai/kimi-k2-thinking:high")

        assert env["KIMI_MODEL_NAME"] == "moonshotai/kimi-k2-thinking:high"

    def test_the_keywords_are_inert(self, provider):
        """§1's guard: timeout_s/schema render argv byte-identical when unset."""
        base = provider.build_command("p", "m")

        assert provider.build_command("p", "m", timeout_s=None, schema=None) == base
        assert provider.build_command("p", "m", timeout_s=600) == base

    # -- parse_output: §3.4's two-step unwrap ------------------------------

    def test_success_returns_the_models_own_json_document(self, provider):
        """Wave-1 acceptance 1: assert the UNWRAPPED data, not `success is True`.

        The capture is a real review-plan-shaped run whose answer is a suggestion
        list, so this pins the whole chain — JSONL envelope, last assistant
        record, then `extract_json_from_text` — rather than merely that the
        parser did not raise.
        """
        result = provider.parse_output(*self._capture("success"))

        assert result["success"] is True
        assert isinstance(result["data"], list) and len(result["data"]) == 4
        assert result["data"][0]["title"] == "No rollback path for the connection pool"
        assert {s["severity"] for s in result["data"]} == {"high", "medium"}

    def test_the_resume_hint_is_never_mistaken_for_the_answer(self, provider):
        """The trap the plan did not know about (§3.4).

        The trailing `session.resume_hint` meta record ALSO carries a `content`
        key. A parser taking "the last record with a content field" returns
        "To resume this session: kimi -r ..." as the model's answer on every
        successful run, so the `role == "assistant"` filter is load-bearing.
        """
        stdout, stderr = self._capture("success")

        assert "session.resume_hint" in stdout, "fixture no longer covers the trap"
        result = provider.parse_output(stdout, stderr)

        assert "To resume this session" not in json.dumps(result["data"])

    def test_multi_assistant_takes_the_last_message(self, provider):
        """§6.2's `multi-assistant` row: last-wins, never first-wins or concat.

        The muse trap. The capture interleaves an assistant preamble, a tool
        record and the real answer; first-wins would return the preamble and
        concatenation would return prose glued to JSON.
        """
        result = provider.parse_output(*self._capture("multi-assistant"))

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert result["data"][0]["title"] == "No rollback path"
        assert "Let me look at the plan first." not in json.dumps(result["data"])

    def test_an_all_noise_run_is_a_failure_not_an_empty_success(self, provider):
        """§6.2's `ephemeral-only` row.

        5xx-forever emits `system.version` plus nine `turn.step.retrying` meta
        records and no assistant record at all. Reporting that as an empty
        success would hand reaggregation a run that never happened.
        """
        result = provider.parse_output(*self._capture("ephemeral-only"))

        assert result["success"] is False
        assert result["data"] is None
        assert "no assistant record" in result["error"]

    def test_a_truncated_stream_fails_cleanly(self, provider):
        """§6.2's `truncated` row: never a traceback, never a half-parsed object."""
        result = provider.parse_output(*self._capture("truncated"))

        assert result["success"] is False
        assert result["data"] is None
        assert result["raw"]

    def test_an_unparseable_trailing_line_does_not_lose_the_answer(self, provider):
        """A stream cut mid-record keeps every complete record before the cut."""
        stdout, _ = self._capture("success")
        result = provider.parse_output(stdout + '{"role":"assis', "")

        assert result["success"] is True
        assert isinstance(result["data"], list)

    def test_the_failure_branch_has_the_documented_shape(self, provider):
        """`kilocode.py`'s empty-output branch is the shape §3.4 names."""
        result = provider.parse_output("", "")

        assert set(result) == {"success", "error", "raw", "data"}
        assert result["success"] is False and result["data"] is None

    # -- failure classification (§6.2) -------------------------------------

    @pytest.mark.parametrize(
        "stem,expected",
        [
            ("auth-error", STATUS_AUTH_FAILED),
            ("rate-limit", STATUS_RATE_LIMITED),
            ("context-overflow", STATUS_CONTEXT_LENGTH_EXCEEDED),
            ("invalid-model", CATEGORY_INVALID_MODEL),
        ],
    )
    def test_captured_failures_classify(self, provider, stem, expected):
        """Each §6.2 fixture pins the classification it was captured for."""
        stdout, stderr = self._capture(stem)

        assert provider.classify_failure(stdout, stderr, exit_code=1) == expected

    def test_the_invalid_model_failure_names_the_model(self, provider):
        """§6.2: an unnamed model makes the failure unfindable."""
        _stdout, stderr = self._capture("invalid-model")

        assert "no-such-model-xyz" in stderr

    def test_a_masked_error_envelope_is_a_recorded_limitation(self, provider):
        """§6.2's `error-envelope` row — a FINDING, pinned so it cannot regress
        into a silent assumption.

        kimi MASKS a provider error delivered inside a 200 success envelope: the
        vendor's text is replaced by a generic "empty response" message, so it
        never reaches `auth_error_patterns`/`rate_limit_patterns`. The run still
        fails loudly (exit 1 plus a classified line), which is what makes this
        acceptable — but a future reader must not conclude the vendor's own error
        text is available here, because it is not.
        """
        stdout, stderr = self._capture("error-envelope")

        assert "empty response" in stderr
        assert provider.classify_failure(stdout, stderr, exit_code=1) is None
        assert provider.parse_output(stdout, stderr)["success"] is False

    def test_no_exit_zero_markers_are_declared(self, provider):
        """Measured, not omitted: across 27 probe runs the exit code was
        authoritative in every shape (0 success, 1 failure). Declaring a marker
        would be guessing at a failure mode this CLI does not have."""
        assert provider.exit_zero_failure_markers == ()

    # -- §2.2 rule 6: the three guard tests --------------------------------

    B1_SELECTOR_SET = (
        "KIMI_MODEL_NAME",
        "KIMI_MODEL_API_KEY",
        "KIMI_MODEL_BASE_URL",
        "KIMI_MODEL_PROVIDER_TYPE",
    )

    def test_get_env_carries_the_whole_b1_selector_set_by_name(self, provider):
        """§2.2 rule 2 — selection is total, never partial.

        KIMI_MODEL_NAME is the *enable switch*: verified against the binary,
        with no member of the family set, `kimi provider list --json` returns
        empty and the CLI silently reverts to its config-file route. So a
        half-set is not a degraded run, it is a run on somebody else's
        credential.
        """
        env = provider.get_env("moonshotai/kimi-k2-thinking")

        for key in self.B1_SELECTOR_SET:
            assert key in env, key
        assert env["KIMI_MODEL_PROVIDER_TYPE"] == "openai"
        assert env["KIMI_MODEL_BASE_URL"] == "https://openrouter.ai/api/v1"

    def test_the_base_url_is_emitted_even_when_the_key_is_empty(
        self, provider, monkeypatch
    ):
        """§2.2 rule 3: never omit the base URL to "let the CLI decide".

        With no key the run must fail as a legible OpenRouter auth error, not
        quietly resolve to the Moonshot subscription endpoint.
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        env = provider.get_env("moonshotai/kimi-k2-thinking")

        assert env["KIMI_MODEL_API_KEY"] == ""
        assert env["KIMI_MODEL_BASE_URL"] == "https://openrouter.ai/api/v1"

    def test_the_base_url_carries_the_version_segment(self, provider):
        """Measured: kimi appends `/chat/completions` VERBATIM and inserts no
        `/v1` of its own, so a base URL without it 404s."""
        assert provider.get_env("m")["KIMI_MODEL_BASE_URL"].endswith("/v1")

    def test_get_remove_env_strips_every_unset_family_member(self, provider):
        """§8.2's rule verbatim, over the measured 18-name family.

        A `KIMI_MODEL_TEMPERATURE` or `KIMI_MODEL_MAX_CONTEXT_SIZE` left over
        from the user's own kimi session would silently reshape a fan-out request
        nobody configured.
        """
        stripped = set(provider.get_remove_env())
        # The adapter's REAL injected set, not the B1 selector set: the output
        # ceiling is a KIMI_MODEL_* name that is injected without being a
        # backend selector, and reading get_env keeps the two in step.
        injected = set(provider.get_env("moonshotai/kimi-k2-thinking"))

        for name in kimi_module._KIMI_MODEL_VARS:
            if name in injected:
                assert name not in stripped, f"{name} is injected, not stripped"
            else:
                assert name in stripped, f"unset family member not stripped: {name}"

    def test_get_remove_env_strips_the_b3_names(self, provider):
        """The Moonshot subscription route and the legacy aliases (§2.2 rule 3)."""
        stripped = set(provider.get_remove_env())

        for name in kimi_module._STRIPPED_VENDOR_VARS:
            assert name in stripped, name
        assert "KIMI_API_KEY" in stripped
        assert "KIMI_LOOP_MAX_RETRIES_PER_STEP" in stripped

    def test_every_catalog_entry_is_in_the_b1_namespace(self):
        """§2.2 rule 6(c): every shipped `models:` entry is an OpenRouter slug,
        passed verbatim to the model surface (here, KIMI_MODEL_NAME)."""
        from utils.provider_registry import load_base_config

        models = load_base_config()["providers"]["kimi"]["models"]

        assert models, "kimi ships no models: it would be invisible to the picker"
        for slug in models:
            assert "/" in slug, f"not an OpenRouter slug: {slug}"
            assert KimiProvider().get_env(slug)["KIMI_MODEL_NAME"] == slug

    def test_the_default_model_is_models_zero(self):
        """§8 item 3: `--init` injects models[0], so it must be the §6.1 slug
        the §7.4 live smoke actually ran."""
        from utils.provider_registry import load_base_config

        models = load_base_config()["providers"]["kimi"]["models"]

        assert models[0] == "moonshotai/kimi-k2.5"

    def test_defaults_models_gains_no_uncommented_row(self):
        """§8 item 3's rule: the shipped example line is COMMENTED.

        `is_available()` is a bare `shutil.which`, so an active row would put a
        provider the user has no binary for into the default fan-out panel and
        into `--quick` — a dead slot in every review.
        """
        from utils.provider_registry import load_base_config

        defaults = load_base_config()["defaults"]["models"]

        assert not [m for m in defaults if str(m).startswith("kimi:")]

    # -- the pinned retry knob (§6.1.1) ------------------------------------

    def test_the_output_ceiling_is_pinned(self, provider):
        """Measured against the REAL endpoint, which is the only place it shows.

        kimi derives `max_tokens` from its synthesized model's context size and
        sends 131072 on every call. Several OpenRouter upstreams cap output below
        that and REJECT the request rather than clamping it -- the first live
        smoke died on `400 Provider returned error`, raw body "Requested maximum
        tokens of 131072 exceeds the maximum output tokens limit: 102400". So the
        pin is what makes kimi answer at all on those providers, not a spend
        preference. It is also §7.4's per-CLI output ceiling.
        """
        env = provider.get_env("moonshotai/kimi-k2-thinking")

        assert env["KIMI_MODEL_MAX_TOKENS"] == str(kimi_module.KIMI_MODEL_MAX_TOKENS)
        assert kimi_module.KIMI_MODEL_MAX_TOKENS < 102_400

    def test_the_other_token_knobs_stay_stripped(self, provider):
        """Three other names move the same wire field. Only the one that names it
        is injected; the rest are stripped so a stale export cannot quietly raise
        the ceiling back to something an upstream rejects."""
        stripped = set(provider.get_remove_env())

        for name in ("KIMI_MODEL_MAX_OUTPUT_SIZE",
                     "KIMI_MODEL_MAX_COMPLETION_TOKENS",
                     "KIMI_MODEL_MAX_CONTEXT_SIZE"):
            assert name in stripped, name

    def test_the_envelope_declares_the_pinned_number(self, provider):
        """`claude_code.py`'s precedent: pin the knob, declare the PINNED number.

        The bare CLI measured 10 attempts; `get_env` pins
        KIMI_LOOP_MAX_ATTEMPTS_PER_STEP=1, so declaring 10 would budget for a
        configuration this harness never runs.
        """
        envelope = provider.internal_retries

        assert envelope is not None
        assert envelope.max_requests == kimi_module.KIMI_LOOP_MAX_ATTEMPTS_PER_STEP
        assert str(envelope.max_requests) == provider.get_env("m")[
            "KIMI_LOOP_MAX_ATTEMPTS_PER_STEP"
        ]

    def test_the_wall_clock_is_honestly_unbounded(self, provider):
        """`max_wall_s=None` is forced twice over, not unmeasured: a stalled
        request has no client-side timeout (no output within 420 s, and no
        KIMI_* request-timeout knob exists) and a honoured `Retry-After` is
        server-chosen. Rule 4 then runs no admission test, which is honest."""
        assert provider.internal_retries.max_wall_s is None

    def test_rate_limits_are_retried_internally(self, provider):
        """Measured against a 429-forever stub. `True` is what makes §7.5 rule
        6(b) suppress an identical harness-side re-dispatch."""
        assert provider.internal_retries.retries_rate_limit is True


class TestQwenCodeProvider:
    """Tests for QwenCodeProvider (harness-expansion.md §3.3, H4·qwen).

    Every assertion is anchored to a capture in `tests/fixtures/qwen-code/`,
    taken from the real binary at 0.25.0 in the pinned container image
    (re-captured 2026-10-08; first captured at 0.21.13 in the §6 probe) — the
    `tests/fixtures/kilocode/` precedent. The two `*.0.21.13.stdout.json` files
    keep the floor release's exit-0 API-error shape graded.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "qwen-code"
    PROMPT = "Review this implementation plan and reply with a JSON array."
    MODEL = "qwen/qwen3.8-max"

    @pytest.fixture
    def provider(self):
        return QwenCodeProvider()

    def _fixture(self, name):
        return (self.FIXTURES / name).read_text()

    # -- identity ---------------------------------------------------------

    def test_name_is_the_harness_not_the_model_family(self, provider):
        """§8.1: the key names the HARNESS, the binary is `qwen`."""
        assert provider.name == "qwen-code"

    def test_is_available_probes_the_binary_not_the_key(self, provider, monkeypatch):
        monkeypatch.setattr(
            qwen_code_module.shutil, "which", lambda n: "/usr/bin/qwen" if n == "qwen" else None
        )
        assert provider.is_available() is True
        monkeypatch.setattr(qwen_code_module.shutil, "which", lambda n: None)
        assert provider.is_available() is False

    # -- argv -------------------------------------------------------------

    def test_prompt_is_a_bare_trailing_positional(self, provider):
        """`-p` is DEPRECATED at 0.21.13; the positional is the shipped form."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[0] == "qwen"
        assert argv[-1] == self.PROMPT
        assert "-p" not in argv and "--prompt" not in argv

    def test_declared_prompt_index_matches_the_rendered_argv(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)
        assert argv[provider.prompt_argv_index] == self.PROMPT

    def test_the_shipped_tool_tier_is_on_every_invocation(self, provider):
        """§6.5: `--max-tool-calls 0` is a rung BELOW tier 1 and always emitted."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("--max-tool-calls") + 1] == "0"
        # and none of the rungs ABOVE tier 1 ship, because tier 1 answered
        assert "--yolo" not in argv
        assert "--approval-mode" not in argv
        assert "--add-dir" not in argv
        assert "--include-directories" not in argv

    def test_json_output_mode_and_no_chat_recording(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("-o") + 1] == "json"
        assert argv[argv.index("--chat-recording") + 1] == "false"
        assert "--bare" in argv

    def test_model_travels_on_the_m_flag(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)
        assert argv[argv.index("-m") + 1] == self.MODEL

    def test_an_effort_suffix_is_not_split_off(self, provider):
        """qwen has NO effort surface, so the suffix stays part of the model id.

        Measured: `-m qwen3-coder-plus:high` reached the backend verbatim. The
        adapter declining to parse it is what this pins — a silent split would
        send a different model than the user asked for.
        """
        argv = provider.build_command(self.PROMPT, f"{self.MODEL}:high")
        assert argv[argv.index("-m") + 1] == f"{self.MODEL}:high"

    # -- §1's guard: timeout_s / schema are inert when unset ---------------

    def test_timeout_none_renders_no_wall_time_flag(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=None)
        assert "--max-wall-time" not in argv

    def test_timeout_none_is_byte_identical_to_omitting_it(self, provider):
        assert provider.build_command(
            self.PROMPT, self.MODEL, timeout_s=None
        ) == provider.build_command(self.PROMPT, self.MODEL)

    def test_schema_none_is_byte_identical_to_omitting_it(self, provider):
        assert provider.build_command(
            self.PROMPT, self.MODEL, schema=None
        ) == provider.build_command(self.PROMPT, self.MODEL)

    # -- §3.3's wall-time table -------------------------------------------

    @pytest.mark.parametrize("outer,expected", [
        (1200, "1170"),
        (600, "570"),
        (300, "270"),
        (91, "61"),
        (90, "60"),
    ])
    def test_wall_time_derivation_table(self, provider, outer, expected):
        """Bare integer seconds, outer minus the 30 s shutdown margin."""
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=outer)
        assert argv[argv.index("--max-wall-time") + 1] == expected

    @pytest.mark.parametrize("outer", [1200, 600, 300, 91])
    def test_the_inner_bound_is_strictly_under_the_outer(self, provider, outer):
        """agy's INNER_SHUTDOWN_MARGIN_S invariant: qwen must exit first."""
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=outer)
        assert int(argv[argv.index("--max-wall-time") + 1]) < outer

    def test_a_too_small_timeout_clamps_and_warns(self, provider, capsys):
        """qwen REJECTS sub-1s values, so the clamp is a guard, not a taste."""
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=40)

        assert argv[argv.index("--max-wall-time") + 1] == "60"
        assert "WARNING" in capsys.readouterr().err

    # -- structured output (OQ2) ------------------------------------------

    def test_schema_renders_inline_like_claude_code(self, provider):
        from utils.providers.base import StructuredSchema

        schema = StructuredSchema(document={"b": 2, "a": 1}, name="s.json")
        argv = provider.build_command(self.PROMPT, self.MODEL, schema=schema)

        value = argv[argv.index("--json-schema") + 1]
        # sorted keys + compact separators, so one schema is one argv byte string
        assert value == '{"a":1,"b":2}'

    def test_schema_is_declared_inline_not_file(self, provider):
        assert provider.structured_output_support is True
        assert provider.structured_output_transport == "inline"

    def test_the_schema_never_lands_next_to_the_prompt(self, provider):
        from utils.providers.base import StructuredSchema

        schema = StructuredSchema(document={"a": 1}, name="s.json")
        argv = provider.build_command(self.PROMPT, self.MODEL, schema=schema)

        assert argv[-1] == self.PROMPT
        assert argv[-2] != "--json-schema"

    # -- stdin transport (§6.4) -------------------------------------------

    def test_stdin_transport_is_declared(self, provider):
        assert provider.supports_prompt_transport("stdin") is True

    def test_stdin_form_differs_by_exactly_the_prompt(self, provider):
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )
        full = provider.build_command(self.PROMPT, self.MODEL)

        assert stdin_text == self.PROMPT
        assert argv == full[:-1]

    def test_a_long_prompt_stays_under_the_windows_cap_on_stdin(self, provider):
        """§6.4's portability test — measured off the rendered argv, so it runs
        everywhere and needs no Windows host."""
        import subprocess

        long_prompt = "x" * 20000
        assert len(long_prompt) > 8191

        argv, stdin_text = provider.build_command_for_transport(
            long_prompt, self.MODEL, "stdin"
        )
        rendered = subprocess.list2cmdline(argv)

        assert stdin_text == long_prompt
        assert len(rendered.encode("utf-16-le")) // 2 < 8191 - 256

    def test_the_argv_form_of_that_prompt_would_not_fit(self, provider):
        """The other half: this is why the stdin transport is load-bearing."""
        import subprocess

        argv = provider.build_command("x" * 20000, self.MODEL)
        rendered = subprocess.list2cmdline(argv)

        assert len(rendered.encode("utf-16-le")) // 2 > 8191

    # -- parse_output: the real document ----------------------------------

    def test_success_returns_the_models_document_not_the_envelope(self, provider):
        """Wave-1 acceptance 1: assert the UNWRAPPED data."""
        result = provider.parse_output(self._fixture("success-document.stdout.json"), "")

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert [s["id"] for s in result["data"]] == ["S1", "S2"]
        assert result["data"][0]["title"] == "Bound the retry loop"

    def test_structured_output_capture_unwraps_to_the_document(self, provider):
        result = provider.parse_output(
            self._fixture("structured-output-success.stdout.json"), ""
        )

        assert result["success"] is True
        assert result["data"] == {"suggestions": [{"id": 1, "title": "probe"}]}

    def test_a_bare_string_answer_is_not_a_document(self, provider):
        """`success.stdout.json` answers "OK" — no JSON, so the unwrap says so
        rather than inventing one."""
        result = provider.parse_output(self._fixture("success.stdout.json"), "")

        assert result["success"] is False
        assert result["data"] is None

    # -- parse_output: the exit-0 defect ----------------------------------

    def test_exit_zero_failure_is_classified_despite_three_green_signals(self, provider):
        """rc=0, subtype=success, is_error=false — and it is still a failure.

        Captured on 0.21.13, the adapter's `min_version`: 0.25.0 no longer
        writes this shape (see the next test), but the floor still does."""
        raw = self._fixture("exit-zero-failure.0.21.13.stdout.json")
        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["subtype"] == "success" and record["is_error"] is False

        result = provider.parse_output(raw, "")

        assert result["success"] is False
        assert "[API Error: 500" in result["error"]
        assert result["data"] is None

    def test_since_0_25_0_an_api_error_is_an_error_record(self, provider):
        """0.25.0 marks the same 500-forever run as an error: exit 1,
        `is_error: true`, no `result` key, the text under `error.message`, and
        the same text in an `AlreadyReportedError` envelope on stderr. The
        failure must read as that text, not as the error object's repr."""
        raw = self._fixture("exit-zero-failure.stdout.json")
        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["is_error"] is True and "result" not in record
        assert record["error"] == {"message": "[API Error: 500 internal server error]"}
        stderr = self._fixture("exit-zero-failure.stderr.txt")
        assert json.loads(stderr)["error"]["type"] == "AlreadyReportedError"

        result = provider.parse_output(raw, stderr)

        assert result["success"] is False
        assert result["error"] == "[API Error: 500 internal server error]"
        assert result["data"] is None

    @pytest.mark.parametrize("row,expected", [
        ("auth-error", "[API Error: 401 Incorrect API key provided.]"),
        ("invalid-model", "[API Error: 404 The model `stub-model` does not exist]"),
        ("truncated", "[API Error: Model stream ended without a finish reason.]"),
        ("empty-stream", "[API Error: Model stream ended with empty response text.]"),
    ])
    def test_the_error_text_is_the_message_not_the_object(self, provider, row, expected):
        result = provider.parse_output(
            self._fixture(f"{row}.stdout.json"), self._fixture(f"{row}.stderr.txt")
        )
        assert result["success"] is False
        assert result["error"] == expected

    @pytest.mark.parametrize("row,expected", [
        ("auth-error", "auth_failed"),
        ("invalid-model", "invalid_model"),
        ("context-overflow", "context_length_exceeded"),
    ])
    def test_the_0_25_0_streams_classify_with_their_stderr(self, provider, row, expected):
        """Both channels as captured: stdout's error record and stderr's
        `AlreadyReportedError` envelope carry the same upstream sentence."""
        assert provider.classify_failure(
            self._fixture(f"{row}.stdout.json"), self._fixture(f"{row}.stderr.txt"), 1
        ) == expected

    @pytest.mark.parametrize("message", [
        "[API Error: 401 Incorrect API key provided.]",  # the captured stub wording
        "[API Error: 401 No auth credentials found]",    # OpenRouter's missing-key wording
        "[API Error: 401 User not found.]",              # OpenRouter's revoked-key wording
    ])
    def test_a_401_is_an_auth_failure_whatever_the_upstream_wording(self, provider, message):
        """qwen renders every upstream failure as `[API Error: <status> <text>]`.
        The base auth patterns match none of these sentences, so a 401 used to
        classify `server_error` (the `api error` rung) and enter the retry
        ladder as retryable — on 0.21.13 and 0.25.0 alike. The status in qwen's
        own prefix is what names it."""
        assert provider.classify_failure(message, "", 1) == "auth_failed"

    def test_the_ripgrep_advisory_does_not_decide_the_class(self, provider):
        """ADAPTER CHANGE (0.25.0). A root-owned install cannot restore the exec
        bit on qwen's bundled `rg`, so every run's stderr opens with
        `Ripgrep not available: spawn …/rg EACCES. Falling back to built-in
        grep.` Captured that way in the pinned image: a 429-forever run ended by
        `--max-wall-time`. The `EACCES` classified it `permission_denied`,
        hiding the rate limit the next line reports."""
        stderr = self._fixture("rate-limit-wall-time.root-owned-install.stderr.txt")
        first, rest = stderr.split("\n", 1)
        assert first.startswith("Ripgrep not available: ") and "EACCES" in first
        assert "[API Error: 429 rate limit exceeded]" in rest

        assert provider.vendor_text("", stderr) == ("", rest)
        assert provider.classify_failure("", stderr, 55) == "rate_limited"
        assert provider.classify_failure("", stderr, 55) == provider.classify_failure(
            "", self._fixture("rate-limit-wall-time.stderr.txt"), 55
        )

    def test_the_ripgrep_advisory_alone_classifies_nothing(self, provider):
        """The same line ahead of a successful run's empty stderr."""
        line = ("Ripgrep not available: spawn /opt/qwen/vendor/ripgrep/x64-linux/rg "
                "EACCES. Falling back to built-in grep.\n")
        success = self._fixture("success.stdout.json")
        assert provider.classify_failure(success, line, 0) is None

    def test_the_budget_envelope_is_found_after_retry_lines(self, provider):
        """ADAPTER CHANGE (0.25.0). The exit-55 envelope used to be the whole
        of stderr; 0.25.0 prints a `Retrying in …` line per backoff first, so
        a parse from byte 0 failed and the report quoted the retry line instead
        of the abort reason."""
        stderr = self._fixture("rate-limit-wall-time.stderr.txt")
        assert stderr.startswith("Retrying in 60s")

        result = provider.parse_output(self._fixture("rate-limit-wall-time.stdout.txt"), stderr)

        assert result["success"] is False
        assert result["error"] == (
            "qwen aborted with no output — FatalBudgetExceededError: Run aborted: "
            "wall-clock budget of 100s exceeded (--max-wall-time)."
        )
        rooted = provider.parse_output(
            "", self._fixture("rate-limit-wall-time.root-owned-install.stderr.txt")
        )
        assert "FatalBudgetExceededError" in rooted["error"]

    @pytest.mark.parametrize("fixture,needle", [
        ("auth-error.stdout.json", "401"),
        ("invalid-model.stdout.json", "404"),
        ("context-overflow.stdout.json", "400"),
        ("empty-stream.stdout.json", "[API Error:"),
    ])
    def test_every_api_error_class_is_a_failure(self, provider, fixture, needle):
        result = provider.parse_output(self._fixture(fixture), "")

        assert result["success"] is False
        assert needle in result["error"]

    def test_invalid_model_failure_names_the_model(self, provider):
        """§6.2: an unnamed model makes a stray `:effort` suffix unfindable."""
        result = provider.parse_output(self._fixture("invalid-model.stdout.json"), "")
        assert "stub-model" in result["error"]

    def test_concatenated_partial_text_across_retries_is_still_a_failure(self, provider):
        """The muse trap by a different route: the marker is a SUFFIX here, so a
        prefix test would serve `OKOKOKOKOK` as the answer. Captured on 0.21.13
        (`min_version`); 0.25.0 discards each retry's partial text — see
        `test_0_25_0_keeps_only_the_last_attempts_partial_text`."""
        raw = self._fixture("truncated.0.21.13.stdout.json")
        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["result"].startswith("OK")
        assert record["stats"]["models"]["stub-model"]["api"]["totalErrors"] == 0

        result = provider.parse_output(raw, "")

        assert result["success"] is False
        assert "OKOKOKOKOK" not in str(result.get("data"))

    def test_0_25_0_keeps_only_the_last_attempts_partial_text(self, provider):
        """0.25.0 writes one `system`/`retry` record per retry (with
        `preserveText: false`), the assistant record holds only the last
        attempt's `OK` ahead of the marker, and the `result` record is an error
        record — so the parser reads the verdict, not the text."""
        raw = self._fixture("truncated.stdout.json")
        records = json.loads(raw)
        retries = [r for r in records if r["type"] == "system" and r.get("subtype") == "retry"]
        assert retries and all(r["data"]["preserveText"] is False for r in retries)
        texts = [c["text"] for r in records if r["type"] == "assistant"
                 for c in r["message"]["content"] if c["type"] == "text"]
        assert texts == ["OK[API Error: Model stream ended without a finish reason.]"]
        assert records[-1]["is_error"] is True and "result" not in records[-1]

        result = provider.parse_output(raw, "")

        assert result["success"] is False
        assert result["data"] is None

    def test_no_finish_reason_takes_the_same_path(self, provider):
        result = provider.parse_output(self._fixture("no-finish-reason.stdout.json"), "")
        assert result["success"] is False

    def test_the_marker_is_searched_not_prefix_matched(self, provider):
        """The marker arrives as a SUFFIX on the truncated shape, so the
        mechanism has to search. `classify_streams` consumes these as compiled
        regexes via `_matches_marker`, which is `.search` — asserted here
        against the real helper rather than against a re-implementation."""
        from utils.providers.base import _matches_marker

        assert _matches_marker(
            provider.exit_zero_failure_markers,
            "OKOKOKOKOK[API Error: 500 internal server error]",
        )
        assert not _matches_marker(
            provider.exit_zero_failure_markers, "a perfectly ordinary answer"
        )

    # -- parse_output: the error-shaped envelope --------------------------

    def test_an_error_shaped_record_is_not_served_as_the_answer(self, provider):
        """The trap the probe found, and the half of it that is now closed.

        An error record has no `result` key, so `parse_result_envelope`'s
        `if "result" in wrapper` guard missed it and the helper returned the
        record itself as the answer. C5 closed the half this fixture happens to
        exercise: the helper now refuses any envelope whose own `is_error` /
        `terminal_reason` says the run failed, and qwen's error record carries
        `is_error: true` like claude-code's does, so the helper alone would now
        refuse these bytes.

        **The ordering this test exists for is unaffected, and the second half
        below is why it still has to be an ordering.** The helper reads two
        fields; qwen's rule 1 reads three — it also refuses a `subtype` other
        than `"success"`, and a record with no `result` key at all. Strip the
        one field C5 reads and the trap is still there, which is what the
        `is_error`-less derivation pins.
        """
        from utils.providers.result_envelope import parse_result_envelope

        raw = self._fixture("stdout-error.stdout.json")
        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["is_error"] is True and record["subtype"] != "success"

        # the shared helper, on the record as qwen really wrote it
        refused = parse_result_envelope(json.dumps(record))
        assert refused["success"] is False
        assert "is_error: true" in refused["error"]

        # the same record minus that one field: no `result` key, no verdict the
        # helper reads, and the bookkeeping comes back as the model's answer
        blind = dict(record)
        del blind["is_error"]
        naive = parse_result_envelope(json.dumps(blind))
        assert naive["success"] is True and naive["data"]["type"] == "result"

        # what the adapter does, because it classifies FIRST
        result = provider.parse_output(raw, "")
        assert result["success"] is False
        assert result["data"] is None

    def test_multi_assistant_is_last_wins_by_construction(self, provider):
        """qwen avoids the muse trap here: step 1 reads the `result` record
        rather than scanning assistant messages."""
        raw = self._fixture("multi-assistant.stdout.json")
        assert len([r for r in json.loads(raw) if r["type"] == "assistant"]) >= 2

        record = [r for r in json.loads(raw) if r["type"] == "result"][-1]
        assert record["result"] == "LAST-ASSISTANT-TEXT"

    # -- parse_output: degenerate shapes ----------------------------------

    def test_empty_stdout_is_a_classified_failure(self, provider):
        """The exit-55 shape writes zero bytes — distinguishable from an empty
        answer, and never a traceback."""
        result = provider.parse_output(self._fixture("wall-time-exceeded.stdout.txt"), "")

        assert result["success"] is False
        assert result["data"] is None

    def test_an_array_with_no_result_record_is_a_failure(self, provider):
        """Defensive: no run of the real binary produces this, which is why it is
        a unit test rather than a fixture (see ephemeral-only.NOT-APPLICABLE.md)."""
        result = provider.parse_output(
            json.dumps([{"type": "system", "subtype": "init"}]), ""
        )

        assert result["success"] is False
        assert "no result record" in result["error"]

    def test_non_json_stdout_is_a_classified_failure(self, provider):
        result = provider.parse_output("this is not json at all", "")

        assert result["success"] is False
        assert result["data"] is None

    def test_a_json_object_instead_of_an_array_is_a_failure(self, provider):
        result = provider.parse_output(json.dumps({"type": "result"}), "")

        assert result["success"] is False
        assert "array" in result["error"]

    # -- env (§2.2 / §8.2) ------------------------------------------------

    def test_get_env_carries_the_whole_b1_selector_set_by_name(self, provider, monkeypatch):
        """§2.2 rule 6(a): asserted BY NAME, including the base URL when the key
        is empty — a half-set is the bug, not a degraded run."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        env = provider.get_env(self.MODEL)

        assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
        assert env["OPENAI_API_KEY"] == ""
        assert env["OPENAI_MODEL"] == self.MODEL

    def test_the_base_url_carries_v1(self, provider):
        """Measured: qwen appends /chat/completions verbatim and adds no /v1."""
        assert provider.get_env(self.MODEL)["OPENAI_BASE_URL"].endswith("/v1")

    def test_the_key_comes_from_openrouter_only(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        assert provider.get_env(self.MODEL)["OPENAI_API_KEY"] == "sk-or-test"

    def test_get_env_never_mutates_os_environ(self, provider, monkeypatch):
        """§3.3 rule 1: a process-wide write would put qwen's base URL under
        codex's next spawn in the same fan-out."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        before = dict(os.environ)

        provider.get_env(self.MODEL)

        assert dict(os.environ) == before

    def test_telemetry_and_updater_are_opted_out(self, provider):
        env = provider.get_env(self.MODEL)
        assert env["QWEN_TELEMETRY_ENABLED"] == "false"
        assert env["QWEN_USAGE_STATISTICS_ENABLED"] == "false"
        assert env["QWEN_CODE_SKIP_UPDATE_CHECK_ONCE"] == "1"

    def test_get_remove_env_strips_every_b3_name(self, provider):
        """§2.2 rule 6(b), asserted by name."""
        stripped = set(provider.get_remove_env())

        assert {"QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_CODE_MODEL",
                "QWEN_DEFAULT_AUTH_TYPE", "DASHSCOPE_API_KEY",
                "DASHSCOPE_BASE_URL", "MODELSCOPE_API_KEY"} <= stripped

    def test_the_legacy_openai_alias_is_stripped(self, provider):
        """A strip is not a substitute for injecting the base URL — both."""
        assert "OPENAI_API_BASE" in provider.get_remove_env()
        assert "OPENAI_BASE_URL" in provider.get_env(self.MODEL)

    def test_it_never_strips_a_name_it_also_sets(self, provider):
        """A value we strip must never also be a value we branch on."""
        assert set(provider.get_remove_env()).isdisjoint(provider.get_env(self.MODEL))

    def test_unattended_retry_is_not_set(self, provider):
        """Measured inert in both states, so setting it would advertise a knob
        that moves nothing — and would drag §1's guard in for free."""
        assert "QWEN_CODE_UNATTENDED_RETRY" not in provider.get_env(self.MODEL)

    # -- internal retries (§6.1.1) ----------------------------------------

    def test_the_envelope_is_declared_from_the_wire_not_the_self_report(self, provider):
        """qwen's own stats said 7 requests where the stub logged 28."""
        assert provider.internal_retries.max_requests == 28

    def test_max_wall_s_is_none_because_two_paths_are_unbounded(self, provider):
        assert provider.internal_retries.max_wall_s is None

    def test_it_declares_that_it_retries_rate_limits(self, provider):
        """Makes §7.5 rule 6(b) suppress a duplicate harness-side backoff."""
        assert provider.internal_retries.retries_rate_limit is True

    def test_configurable_by_names_only_a_knob_that_moves_a_number(self, provider):
        assert provider.internal_retries.configurable_by == ("QWEN_CODE_API_TIMEOUT_MS",)

    # -- version / usage ---------------------------------------------------

    def test_version_is_a_bare_semver(self, provider):
        """The draft lock pattern guessed a product-name prefix; there is none."""
        assert provider.parse_version("0.21.13\n", "") == "0.21.13"

    def test_usage_stays_none_until_the_live_smoke_earns_it(self, provider):
        assert provider.usage_support == "none"
        assert provider.parse_usage("anything", "") is None


class TestCopilotProvider:
    """Tests for CopilotProvider (harness-expansion.md §3.1, H4·copilot).

    Every assertion is anchored to a capture in `tests/fixtures/copilot/`, taken
    from the real binary at 1.0.80 inside the §6 probe container — the
    `tests/fixtures/kilocode/` precedent — and re-captured at 1.0.93 on
    2026-10-08, except `exit-zero-failure.jsonl`, which only 1.0.80 produces.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
    PROMPT = "Review this implementation plan and reply with a JSON array."
    MODEL = "openai/gpt-4.1-mini"

    #: The document the captured answer really carries, so the success test can
    #: assert the UNWRAPPED data rather than `success is True` alone
    #: (Wave-1 acceptance criterion 1).
    ANSWER = [
        {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},
        {"id": "S002", "title": "Strip the vendor token", "severity": "medium"},
    ]

    @pytest.fixture
    def provider(self):
        return CopilotProvider()

    def _fixture(self, name):
        return (self.FIXTURES / name).read_text()

    # -- argv -------------------------------------------------------------

    def test_build_command_puts_the_prompt_pair_last(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[0] == "copilot"
        assert argv[-2:] == ["-p", self.PROMPT]
        assert argv[provider.prompt_argv_index] == self.PROMPT

    def test_build_command_emits_no_permission_flag(self, provider):
        """§6.5 tier 1, and the help text's own claim is what this refutes.

        `--allow-all-tools` is documented as "required for non-interactive
        mode". Measured, all four rungs produced byte-identical output with a
        clean filesystem canary, so rule 1's ascending stop applies at rung 1 and
        §3.1's bracketed line is struck. Widening this later must be a visible
        test edit, which is the whole point of asserting absence.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL)

        for flag in (
            "--allow-all-tools", "--allow-tool", "--deny-tool", "--add-dir",
            "--yolo", "--allow-all", "--allow-all-paths", "--allow-all-urls",
        ):
            assert flag not in argv, f"{flag} is above the measured tier"

    def test_build_command_carries_the_determinism_flags(self, provider):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        for pair in (
            ["--output-format", "json"],
            ["--stream", "off"],
            ["--log-level", "none"],
            ["--model", self.MODEL],
        ):
            assert pair[0] in argv and argv[argv.index(pair[0]) + 1] == pair[1]
        for flag in ("--no-custom-instructions", "--no-ask-user",
                     "--no-auto-update", "--disable-builtin-mcps"):
            assert flag in argv

    def test_timeout_and_schema_keywords_are_inert(self, provider):
        """§1's guard: copilot has no wall-clock flag and no schema transport."""
        bare = provider.build_command(self.PROMPT, self.MODEL)

        assert provider.build_command(
            self.PROMPT, self.MODEL, timeout_s=None, schema=None
        ) == bare
        assert provider.build_command(self.PROMPT, self.MODEL, timeout_s=600) == bare

    def test_declares_no_structured_output_support(self, provider):
        assert provider.structured_output_support is False

    # -- the :effort suffix ----------------------------------------------

    @pytest.mark.parametrize(
        "effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_every_measured_effort_level_renders(self, provider, effort):
        argv = provider.build_command(f"{self.MODEL}:{effort}", self.MODEL)
        argv = provider.build_command(self.PROMPT, f"{self.MODEL}:{effort}")

        assert argv[argv.index("--model") + 1] == self.MODEL
        assert argv[argv.index("--effort") + 1] == effort
        assert argv[-1] == self.PROMPT

    def test_an_unknown_suffix_passes_through_as_part_of_the_model_id(self, provider):
        argv = provider.build_command(self.PROMPT, "openai/gpt-4.1-mini:free")

        assert argv[argv.index("--model") + 1] == "openai/gpt-4.1-mini:free"
        assert "--effort" not in argv

    def test_the_effort_suffix_keeps_the_prompt_last(self, provider):
        """grok's regression, re-pinned here: the suffix must not move argv[-1]."""
        argv = provider.build_command(self.PROMPT, f"{self.MODEL}:high")

        assert argv[provider.prompt_argv_index] == self.PROMPT

    # -- prompt transport (§6.4) ------------------------------------------

    def test_stdin_transport_drops_the_whole_pair(self, provider):
        """Measured: piped stdin with `-p` OMITTED carries the prompt.

        `-p -` sends the literal one-character prompt `-`, and a bare `-` exits
        1 — so dropping only `argv[-1]` would leave a dangling `-p` and change
        what the CLI is asked to do.
        """
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )

        assert stdin_text == self.PROMPT
        assert "-p" not in argv
        assert self.PROMPT not in argv
        assert argv == provider.build_command(self.PROMPT, self.MODEL)[:-2]

    def test_argv_transport_matches_build_command(self, provider):
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "argv"
        )

        assert stdin_text is None
        assert argv == provider.build_command(self.PROMPT, self.MODEL)

    def test_a_long_prompt_stays_under_the_windows_cap_on_stdin(self, provider):
        """§6.4's portability test — measured off the rendered argv, so it needs
        no Windows host.

        A review-plan prompt is multi-KB by construction, which is above
        `CMDLINE_CAP_UTF16_BATCH` (8191) and would be refused before spawn on the
        `.cmd` shim `npm install -g` leaves on Windows.
        """
        import subprocess

        long_prompt = "x" * 20000
        argv, stdin_text = provider.build_command_for_transport(
            long_prompt, self.MODEL, "stdin"
        )

        assert stdin_text == long_prompt
        rendered = subprocess.list2cmdline(argv)
        assert len(rendered.encode("utf-16-le")) // 2 < 8191 - 256

    # -- parse_output: the two-step unwrap (§3.4) -------------------------

    def test_success_returns_the_models_own_document(self, provider):
        """Wave-1 acceptance 1: assert the UNWRAPPED data, not `success` alone."""
        result = provider.parse_output(self._fixture("success.jsonl"), "")

        assert result["success"] is True
        assert result["data"] == self.ANSWER

    def test_the_streamed_shape_parses_identically(self, provider):
        """One recipe covers both stream modes, so `--stream off` is a
        determinism choice rather than a parsing requirement."""
        result = provider.parse_output(self._fixture("success-streamed.jsonl"), "")

        assert result["success"] is True
        assert result["data"] == self.ANSWER

    def test_the_user_message_is_never_served_as_the_answer(self, provider):
        """THE trap §3.4's recipe walks into.

        `user.message` is non-ephemeral and carries `data.content` — our own
        prompt. Filtering on `ephemeral` alone and taking the first survivor
        returns the prompt as the model's answer, which type-checks and is wrong.
        """
        raw = self._fixture("success.jsonl")
        prompt_record = [
            json.loads(line) for line in raw.splitlines() if line.strip()
        ]
        prompt_text = next(
            r["data"]["content"] for r in prompt_record if r.get("type") == "user.message"
        )

        result = provider.parse_output(raw, "")

        assert len(prompt_text) > 30000, "the fixture's prompt echo is the trap"
        assert result["data"] == self.ANSWER
        assert prompt_text not in json.dumps(result["data"])

    def test_multi_assistant_takes_the_last_message(self, provider):
        """last-wins, not first-wins and not concatenation — the muse trap."""
        raw = self._fixture("multi-assistant.jsonl")

        result = provider.parse_output(raw, "")

        assert "FIRST ANSWER" in raw
        assert result["success"] is True
        assert result["data"] == self.ANSWER

    # -- parse_output: the §6.2 failure set --------------------------------

    def test_an_all_noise_run_is_a_failure(self, provider):
        result = provider.parse_output(self._fixture("ephemeral-only.jsonl"), "")

        assert result["success"] is False
        assert result["data"] is None
        assert result["raw"]

    def test_a_truncated_stream_has_no_result_record(self, provider):
        """The capture is a real kill mid-flight: it ends at `model.call_start`,
        so a parser that assumes a terminal `result` record raises instead of
        classifying."""
        raw = self._fixture("truncated.jsonl")

        result = provider.parse_output(raw, "")

        assert '"type":"result"' not in raw
        assert result["success"] is False
        assert result["data"] is None

    def test_the_exit_zero_failure_is_caught_structurally(self, provider):
        """exit 0, `result.exitCode: 0`, no error record, empty answer.

        There is no text for a marker to match, which is why
        `exit_zero_failure_markers` is empty for this adapter and the detection
        lives here instead.
        """
        raw = self._fixture("exit-zero-failure.jsonl")

        result = provider.parse_output(raw, "")

        assert '"exitCode":0' in raw
        assert result["success"] is False
        assert "empty assistant message" in result["error"]

    def test_empty_output_is_classified_not_crashed(self, provider):
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert result["data"] is None

    def test_empty_stdout_quotes_stderr_when_there_is_any(self, provider):
        """stderr is 0 bytes on every model-path failure, so text here is an
        argument-parse or launch error — exactly the case worth quoting."""
        result = provider.parse_output("", "error: unknown option '--nope'")

        assert result["success"] is False
        assert "--nope" in result["error"]

    @pytest.mark.parametrize(
        "fixture,needle",
        [
            ("auth-error.jsonl", "authentication"),
            ("rate-limit.jsonl", "rate_limit"),
            ("invalid-model.jsonl", "query"),
            ("context-overflow.jsonl", "query"),
        ],
    )
    def test_session_error_becomes_the_reported_failure(
        self, provider, fixture, needle
    ):
        """`session.error` on stdout with stderr empty is §6.2's `stdout-error`
        trap in its exact shape, and it is copilot's ONLY error channel."""
        result = provider.parse_output(self._fixture(fixture), "")

        assert result["success"] is False
        assert needle in result["error"]
        assert result["data"] is None

    def test_the_invalid_model_failure_names_the_model(self, provider):
        """Which is what makes a stray `:effort` suffix findable rather than an
        unattributable 404."""
        result = provider.parse_output(self._fixture("invalid-model.jsonl"), "")

        assert "gpt-5.5" in result["error"]

    # -- classification ----------------------------------------------------

    @pytest.mark.parametrize(
        "fixture,expected",
        [
            ("auth-error.jsonl", "auth_failed"),
            ("rate-limit.jsonl", "rate_limited"),
            ("invalid-model.jsonl", "invalid_model"),
            ("context-overflow.jsonl", "context_length_exceeded"),
        ],
    )
    def test_classify_failure_reads_the_error_record(
        self, provider, fixture, expected
    ):
        """The classifier reads STDOUT here, because that is where copilot's
        errors are and stderr is empty."""
        assert provider.classify_failure(
            self._fixture(fixture), "", 1
        ) == expected

    def test_auth_wording_needed_its_own_pattern(self, provider):
        """copilot says "Authentication failed with provider", which matches none
        of the base set's ten patterns."""
        from utils.providers.base import LLMProvider as Base

        assert "authentication failed" in provider.auth_error_patterns
        assert "authentication failed" not in Base.auth_error_patterns

    def test_the_exit_zero_failure_stays_transport_shaped(self, provider):
        """Deliberate: it reaches the ladder as `None` and takes the bounded
        retry path, which is the right treatment for an empty answer."""
        assert provider.classify_failure(
            self._fixture("exit-zero-failure.jsonl"), "", 0
        ) is None

    def test_exit_zero_markers_are_empty_by_measurement(self, provider):
        assert provider.exit_zero_failure_markers == ()

    # -- the measured internal-retry envelope (§6.1.1) --------------------

    def test_the_envelope_is_the_measured_one_not_the_plans_placeholders(
        self, provider
    ):
        """§3.1 guessed 5 / ~30 s / codex's False. All three were wrong."""
        envelope = provider.internal_retries

        assert envelope is not None
        assert envelope.max_requests == 6
        assert envelope.max_wall_s is None
        assert envelope.retries_rate_limit is True
        assert envelope.configurable_by == ()

    def test_the_rate_limit_fixture_corroborates_the_request_count(self, provider):
        """The CLI's own error text is the independent second source."""
        raw = self._fixture("rate-limit.jsonl")

        assert "retried 5 times" in raw  # 1 + 5 == max_requests

    # -- env (§2.2, §8.2) --------------------------------------------------

    def test_get_env_emits_the_complete_selector_set(self, provider):
        env = provider.get_env(self.MODEL)

        for name in (
            "COPILOT_PROVIDER_BASE_URL", "COPILOT_PROVIDER_TYPE",
            "COPILOT_PROVIDER_API_KEY", "COPILOT_OFFLINE",
            "COPILOT_PROVIDER_MAX_PROMPT_TOKENS",
            "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS",
        ):
            assert name in env, f"{name} missing from the B1 selector set"
        assert env["COPILOT_PROVIDER_BASE_URL"].endswith("/v1")
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"
        assert env["COPILOT_OFFLINE"] == "true"

    def test_the_base_url_is_emitted_even_with_no_key(self, provider, monkeypatch):
        """§2.2 rule 3: never omit the base URL to "let the CLI decide".

        The base URL is the BYOK activation switch, so omitting it does not
        degrade the run — it selects the GitHub seat path instead.
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        env = provider.get_env(self.MODEL)

        assert env["COPILOT_PROVIDER_API_KEY"] == ""
        assert env["COPILOT_PROVIDER_BASE_URL"].startswith("https://openrouter.ai")

    def test_does_not_set_xdg_cache_home(self, provider):
        """A per-slot value would re-extract copilot's 209 MB package cache on
        EVERY spawn, and a hardcoded path would be wrong on a host. The image
        owns this one."""
        assert "XDG_CACHE_HOME" not in provider.get_env(self.MODEL)

    def test_never_mutates_the_parent_environment(self, provider):
        before = dict(os.environ)

        provider.get_env(self.MODEL)

        assert os.environ == before

    @pytest.mark.parametrize(
        "name",
        [
            "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
            "COPILOT_ALLOW_ALL", "COPILOT_PROVIDER_BEARER_TOKEN",
            "OTEL_EXPORTER_OTLP_ENDPOINT", "COPILOT_OTEL_ENABLED",
            "COPILOT_MODEL",
        ],
    )
    def test_get_remove_env_strips_every_measured_name(self, provider, name):
        assert name in provider.get_remove_env()

    def test_the_tier_widening_env_twin_is_stripped(self, provider):
        """Without this, §6.5's recorded tier is unenforceable: COPILOT_ALLOW_ALL
        promotes the run to tier 4 with no argv change at all."""
        assert "COPILOT_ALLOW_ALL" in provider.get_remove_env()

    # -- the token-limit pair (§3.1): the catalog and the floor, full stop.
    # The override rung above them was REMOVED rather than deferred (C9);
    # `TestCopilotHasNoConfigurableTokenLimits` below pins its absence.

    def test_a_catalogued_slug_gets_its_own_pair(self, provider):
        from utils.providers.copilot import OUTPUT_RESERVE_TOKENS, token_limits_for

        prompt_budget, output_budget = token_limits_for("openai/gpt-4o-mini")

        assert prompt_budget == 128000 - OUTPUT_RESERVE_TOKENS
        assert output_budget == OUTPUT_RESERVE_TOKENS

    def test_an_uncatalogued_slug_falls_to_the_floor(self, provider):
        from utils.providers.copilot import (
            FLOOR_MAX_OUTPUT_TOKENS, FLOOR_MAX_PROMPT_TOKENS, token_limits_for,
        )

        assert token_limits_for("some/unknown-model") == (
            FLOOR_MAX_PROMPT_TOKENS, FLOOR_MAX_OUTPUT_TOKENS,
        )

    def test_the_pair_is_resolved_for_the_model_handed_in(self, provider):
        """Never a process-wide constant, and never a pair resolved for a
        different slug."""
        small = provider.get_env("openai/gpt-4o-mini")
        large = provider.get_env("openai/gpt-4.1-mini")

        assert (small["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"]
                != large["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"])

    def test_the_effort_suffix_does_not_break_the_limit_lookup(self, provider):
        """The catalog is keyed by slug, so the suffix has to come off first."""
        plain = provider.get_env("openai/gpt-4o-mini")
        suffixed = provider.get_env("openai/gpt-4o-mini:high")

        assert (suffixed["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"]
                == plain["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"])

    def test_every_shipped_catalog_entry_is_known_to_the_limit_table(self):
        """A slug in `providers.yaml` with no catalog row silently takes the
        floor, which is safe but is not what the block claims."""
        import utils.provider_registry as reg
        from utils.providers.copilot import _MODEL_CATALOG

        config = reg.load_config()
        for slug in config["providers"]["copilot"]["models"]:
            assert slug in _MODEL_CATALOG, f"{slug} has no measured token pair"

    # -- version + usage ---------------------------------------------------

    def test_usage_stays_none_because_no_input_count_exists(self, provider):
        """`result.usage` carries premiumRequests (0 on BYOK by construction),
        durations and codeChanges — no prompt-token field at all, and
        `data.outputTokens` vanishes under `--stream on`. Absent must mean "not
        reported", never 0."""
        assert provider.usage_support == "none"
        assert provider.parse_usage(self._fixture("success.jsonl"), "") is None

    def test_version_parsing_tolerates_the_trailing_update_line(self, provider):
        """`copilot --version` prints "GitHub Copilot CLI 1.0.80." AND a second
        "Run 'copilot update'…" line, so nothing may anchor to end-of-output."""
        raw = "GitHub Copilot CLI 1.0.80.\nRun 'copilot update' to check for updates.\n"

        assert provider.parse_version(raw, "") == "1.0.80"


@pytest.mark.config_override
class TestCopilotHasNoConfigurableTokenLimits:
    """C9: the `providers.yaml` override rung is removed, not deferred.

    `token_limits_for` derives the pair from the committed per-slug catalog and
    the conservative floor, and from nothing else. What is pinned here is the
    **absence** of the affordance, because an absence is exactly what a later
    reader re-adds half-way: the config layer merges an undeclared
    `max_prompt_tokens:` into the copilot block and then nothing reads it, so a
    documented key would have been a silent no-op rather than a feature.

    copilot is the only adapter in the registry that declares a token budget at
    all, which is why the typed-config alternative was rejected instead of held
    open, and why the one condition that reopens it is a *second* such adapter.
    """

    MODEL = "openai/gpt-4o-mini"

    #: Deliberately far from both the catalog pair (128000 - 8000 / 8000) and
    #: the floor (32000 / 8000), so a half-implemented read of either key shows
    #: up as an equality failure rather than hiding behind a coincidence.
    CONFIGURED_PROMPT = 190000
    CONFIGURED_OUTPUT = 32000

    def _with_the_key_configured(self, monkeypatch, tmp_path):
        """A providers.yaml carrying both keys, as the removed rung described."""
        import utils.provider_registry as registry

        override = tmp_path / "providers.yaml"
        override.write_text(
            "providers:\n"
            "  copilot:\n"
            f"    max_prompt_tokens: {self.CONFIGURED_PROMPT}\n"
            f"    max_output_tokens: {self.CONFIGURED_OUTPUT}\n",
            encoding="utf-8",
        )
        # Deterministic layering: project discovery off, env layer on, cache dropped.
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(override))
        registry._config = None
        registry._config_key = None
        return registry

    def test_the_key_really_does_reach_the_merged_provider_block(
        self, monkeypatch, tmp_path
    ):
        """The premise of the removal, measured rather than argued.

        `provider_registry` merges an undeclared key and then ignores it, so the
        key is *present* in the loaded block. That is what makes documenting it
        worse than not having it, and it is the half a reader must see before
        the two assertions below mean anything.
        """
        registry = self._with_the_key_configured(monkeypatch, tmp_path)

        block = registry.load_config()["providers"]["copilot"]

        assert block["max_prompt_tokens"] == self.CONFIGURED_PROMPT
        assert block["max_output_tokens"] == self.CONFIGURED_OUTPUT

    def test_the_same_pair_comes_back_with_and_without_the_block(
        self, monkeypatch, tmp_path
    ):
        from utils.providers.copilot import token_limits_for

        without = token_limits_for(self.MODEL)
        self._with_the_key_configured(monkeypatch, tmp_path)

        assert token_limits_for(self.MODEL) == without

    def test_get_env_emits_the_derived_pair_and_not_the_configured_one(
        self, monkeypatch, tmp_path
    ):
        """The end of the wire: what the CLI is actually told."""
        from utils.providers.copilot import token_limits_for

        derived_prompt, derived_output = token_limits_for(self.MODEL)
        self._with_the_key_configured(monkeypatch, tmp_path)

        env = CopilotProvider().get_env(self.MODEL)

        assert env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] == str(derived_prompt)
        assert env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] == str(derived_output)
        assert env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] != str(self.CONFIGURED_PROMPT)
        assert env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] != str(self.CONFIGURED_OUTPUT)

    def test_the_decision_is_written_where_the_next_reader_will_look(self):
        """A rejected alternative with no reopening condition is just a gap.

        Both halves live in `token_limits_for`'s own docstring, next to the
        behaviour they explain, rather than only in a plan file.
        """
        from utils.providers.copilot import token_limits_for

        doc = (token_limits_for.__doc__ or "").lower()

        assert "second adapter" in doc, (
            "the one reopening condition is not recorded"
        )
        assert "typing the key is" in doc, (
            "the docstring no longer says that a user typing the key does NOT "
            "reopen the decision"
        )
        assert "no config override" in doc, (
            "the docstring no longer states the absence as a decision"
        )


class TestVibeProvider:
    """Tests for VibeProvider (harness-expansion.md §4.2, H6·vibe).

    Every assertion is anchored to a capture in `tests/fixtures/vibe/`, taken
    from the real binary at 2.24.1 inside the §6 probe container and re-taken
    at 2.26.0 in the pinned image on 2026-10-08 (manifest.yaml) — the
    `tests/fixtures/kilocode/` precedent.

    vibe is the row where the plan was most wrong, so several tests here exist
    specifically to pin a **corrected** contract rather than the researched one:
    the permission ladder is inverted at the bottom (§4.2 called the bare form
    tier 1; measured it is DESTRUCTIVE), the model surface is env rather than a
    templated per-run file, and the `emits_finish_reason` retry §4.2 built an
    abort criterion around does not happen at all.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "vibe"
    PROMPT = "Review this implementation plan and reply with a JSON array."
    MODEL = "vibe-default"

    @pytest.fixture
    def provider(self):
        return VibeProvider()

    def _fixture(self, name):
        return (self.FIXTURES / name).read_text()

    # -- identity ---------------------------------------------------------

    def test_name_is_the_binary_not_the_pypi_package(self, provider):
        """§8.1: `mistral-vibe` is a package name, never an identifier."""
        assert provider.name == "vibe"

    def test_is_available_probes_the_binary(self, provider, monkeypatch):
        monkeypatch.setattr(
            vibe_module.shutil, "which",
            lambda n: "/usr/bin/vibe" if n == "vibe" else None,
        )
        assert provider.is_available() is True
        monkeypatch.setattr(vibe_module.shutil, "which", lambda n: None)
        assert provider.is_available() is False

    # -- argv -------------------------------------------------------------

    def test_the_prompt_is_the_last_element(self, provider):
        """§8.3: flags first, the `-p` pair trailing, so `prompt_argv_index` is -1."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[0] == "vibe"
        assert argv[-2:] == ["-p", self.PROMPT]
        assert provider.prompt_argv_index == -1

    def test_the_model_never_reaches_argv(self, provider):
        """vibe has no `--model` flag; the alias rides VIBE_ACTIVE_MODEL."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert self.MODEL not in argv
        assert "--model" not in argv

    def test_the_safe_agent_is_emitted_and_the_wide_ones_are_not(self, provider):
        """§6.5's rung, and the whole reason it is not the bare invocation.

        `default_agent` is `accept-edits`, whose `AgentSafety` is DESTRUCTIVE and
        whose write_file/edit permission is `always` — so omitting `--agent`
        would be the second-widest rung on the ladder, not tier 1.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("--agent") + 1] == "plan"
        for wide in ("--auto-approve", "--yolo", "--trust", "--dangerously-skip"):
            assert wide not in argv

    def test_the_spend_and_turn_caps_are_policy_constants(self, provider):
        """Both are verbatim in the pinned argv table, never placeholders.

        `--max-price` is the roster's only USD cap but is checked BETWEEN turns —
        a single completion under a 0.0000001 cap ran to completion with the full
        answer — so `--max-turns` is the bound that actually stops a runaway.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("--max-price") + 1] == vibe_module.MAX_PRICE_USD
        assert argv[argv.index("--max-turns") + 1] == vibe_module.MAX_TURNS

    def test_timeout_none_renders_byte_identical_argv(self, provider):
        """§1's guard: no default seconds value sneaks in.

        vibe has no wall-clock FLAG at all — its inner bound is the pinned
        `VIBE_API_TIMEOUT` — so a threaded timeout must change nothing here.
        """
        bare = provider.build_command(self.PROMPT, self.MODEL)

        assert provider.build_command(self.PROMPT, self.MODEL, timeout_s=None) == bare
        assert provider.build_command(self.PROMPT, self.MODEL, timeout_s=600) == bare

    def test_schema_none_renders_byte_identical_argv(self, provider):
        bare = provider.build_command(self.PROMPT, self.MODEL)

        assert provider.build_command(self.PROMPT, self.MODEL, schema=None) == bare

    def test_the_effort_suffix_changes_no_argv(self, provider):
        """Effort is a CONFIG key, so it rides env — argv must be untouched."""
        assert provider.build_command(
            self.PROMPT, f"{self.MODEL}:high"
        ) == provider.build_command(self.PROMPT, self.MODEL)

    # -- transport --------------------------------------------------------

    def test_stdin_keeps_the_flag_and_drops_only_the_value(self, provider):
        """vibe's stdin form is the odd one out, and the difference is fatal.

        Every other stdin adapter drops the `-p <prompt>` PAIR. vibe must keep
        `-p` and drop only its value: a bare positional prompt — or no `-p` at
        all — launches the interactive TUI and hangs forever behind an ANSI
        alt-screen, which burns the whole slot rather than degrading.
        """
        argv, stdin = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )

        assert stdin == self.PROMPT
        assert argv[-1] == "-p"
        assert self.PROMPT not in argv

    def test_the_transport_form_differs_only_by_the_prompt(self, provider):
        """§1's guard-shaped test for a second transport."""
        argv_form = provider.build_command(self.PROMPT, self.MODEL)
        stdin_form, _ = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )

        assert stdin_form == argv_form[:-1]

    def test_a_multi_kb_prompt_fits_the_windows_cap_on_stdin(self, provider):
        """§6.4's long-prompt portability test — a unit test on every platform.

        A review-plan prompt is multi-KB by construction, and
        `_prompt_length_error` refuses to spawn above 8,191 UTF-16 units for the
        `.cmd` shim `npm install -g`/`pip install` leaves on Windows. The cap is
        measured off the rendered command line, so this needs no Windows host.
        """
        import subprocess as _sp

        big = "x" * 20000
        assert len(big) > 8191

        argv_form = provider.build_command(big, self.MODEL)
        stdin_form, payload = provider.build_command_for_transport(
            big, self.MODEL, "stdin"
        )

        assert len(_sp.list2cmdline(argv_form)) > 8191
        assert len(_sp.list2cmdline(stdin_form)) < 8191 - 256
        assert payload == big

    # -- parse_output: the §3.4 two-step unwrap ---------------------------

    def test_the_success_fixture_unwraps_to_the_models_document(self, provider):
        """Wave-1 acceptance 1: assert the unwrapped `data`, not `success` alone.

        The answer must be the model's own JSON document — not vibe's transcript
        envelope, and not the answer as a raw string. All three type-check and
        all three break reaggregation.
        """
        result = provider.parse_output(self._fixture("success-json.json"), "")

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert result["data"][0]["id"] == "S001"
        assert len(result["data"]) == 3

    def test_the_streaming_capture_carries_the_same_entries(self, provider):
        """`--output streaming` is the same entry objects as NDJSON.

        The adapter ships `json`, so this pins the *reason* rather than a second
        code path: one parse of one document beats N line-parses when the schema
        is identical.
        """
        raw = self._fixture("success-streaming.jsonl")
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()]

        assert any(
            e.get("role") == "assistant" and e.get("type") == "message"
            for e in entries
        )

    def test_multi_assistant_takes_the_last_entry(self, provider):
        """The muse trap: last-wins, never first-wins and never concatenation."""
        result = provider.parse_output(self._fixture("multi-assistant.json"), "")

        assert result["success"] is True
        assert result["data"][0]["id"] == "S001"

    def test_a_transcript_with_no_assistant_entry_is_a_failure(self, provider):
        """An all-noise run is a failure, not an empty-string success."""
        result = provider.parse_output(self._fixture("ephemeral-only.json"), "")

        assert result["success"] is False
        assert result["data"] is None
        assert "no assistant message" in result["error"]

    def test_a_config_error_on_stdout_is_not_read_as_an_answer(self, provider):
        """§6.2's `stdout-error` trap, in vibe's one measured shape.

        Config-validation errors print on **stdout with stderr empty** and still
        exit 1 — the opencode shape. The reason exists only in that text, so it
        has to be reported rather than handed to the extractor, which would find
        nothing and report an empty answer instead of the cause.
        """
        result = provider.parse_output(self._fixture("stdout-error.stdout.txt"), "")

        assert result["success"] is False
        assert result["data"] is None
        assert "Invalid configuration" in result["error"]

    def test_zero_bytes_on_both_streams_is_classified(self, provider):
        """Distinguishable from a valid empty answer — the silent-hang shape."""
        result = provider.parse_output(
            self._fixture("empty.stdout.txt"), self._fixture("empty.stderr.txt")
        )

        assert result["success"] is False
        assert result["data"] is None
        assert result["raw"] == ""

    def test_the_failure_branch_carries_raw_and_a_null_data(self, provider):
        """`kilocode.py`'s empty-output branch is the shape §6.2 reads."""
        result = provider.parse_output("", "")

        assert set(result) >= {"success", "error", "raw", "data"}
        assert result["success"] is False
        assert result["data"] is None

    # -- failure classification -------------------------------------------

    @pytest.mark.parametrize(
        "fixture,expected",
        [
            ("auth-error", "auth_failed"),
            ("rate-limit", "rate_limited"),
            ("context-overflow", "context_length_exceeded"),
        ],
    )
    def test_the_measured_stderr_shapes_classify(self, provider, fixture, expected):
        """Each pins the vendor wording the declared pattern set was measured on."""
        stderr = self._fixture(f"{fixture}.stderr.txt")

        assert provider.classify_failure("", stderr, 1) == expected

    def test_the_context_overflow_shape_is_terminal(self, provider):
        """It must reach the terminal status, not the bounded-retry path.

        vibe's wording puts the noun first and calls the bound a "limit"
        ("The conversation context exceeds the model's maximum limit"), which the
        pre-existing gateway pattern does not reach — so this is the assertion
        behind the one shared-table marker this adapter added. That sentence is
        2.24.1's (the floor) and is quoted here from the first capture; 2.26.0
        forwards the upstream sentence instead, which the committed fixture
        carries.
        """
        floor_stderr = (
            "Error: The conversation context exceeds the model's maximum limit. "
            "Use /rewind to undo recent actions, then /compact to summarize the "
            "conversation.\n"
        )
        assert provider.classify_failure("", floor_stderr, 1) == "context_length_exceeded"

        stderr = self._fixture("context-overflow.stderr.txt")

        assert "maximum context length" in stderr.lower()
        assert provider.classify_failure("", stderr, 1) == "context_length_exceeded"

    def test_the_invalid_model_failure_names_the_model(self, provider):
        """The shape a stray `:effort` suffix produces; an unnamed model is unfindable."""
        stderr = self._fixture("invalid-model.stderr.txt")

        assert "stub-model" in stderr

    def test_no_exit_zero_failure_markers_are_declared(self, provider):
        """Empty BY MEASUREMENT: every failure mode probed exits 1.

        §6.2's `exit-zero-failure` row is recorded N/A for this CLI rather than
        left untested, and that is what this asserts.
        """
        assert provider.exit_zero_failure_markers == ()

    # -- declarations -----------------------------------------------------

    def test_the_retry_envelope_is_measured_and_pinned(self, provider):
        """§6.1.1, including the experiment no other adapter runs.

        §4.2 predicted a 200 with no `finish_reason` makes vibe retry, and built
        abort criterion 1 on the hazard. Measured, it is one POST and a hard
        error — so the envelope's 3 attempts come from the retryable HTTP set,
        and `max_wall_s` is real (rather than `None`, as copilot's and
        qwen-code's are) precisely because `Retry-After` was NOT honoured at
        2.24.1, so no server-chosen sleep could extend it.

        2.26.0 (2026-10-08) replaced the three tries with a wall-clock retry
        budget and honours `Retry-After` up to a 30 s cap, so the declaration
        now covers both supported releases from the two pins: measured at the
        pinned 150 s budget, a 500-forever chain is 11 requests over 155 s and
        six `Retry-After: 29` 429s followed by a stall end at 267 s, under
        budget + 30 + timeout + 2 = 272 s, the same bound 2.24.1's 3 x 90 + 2
        gives.
        """
        envelope = provider.internal_retries

        assert envelope is not None
        assert envelope.max_requests == 11
        assert envelope.retries_rate_limit is True
        assert envelope.configurable_by == (
            "VIBE_API_TIMEOUT", "VIBE_API_RETRY_MAX_ELAPSED_TIME",
        )
        assert envelope.max_wall_s == max(
            3 * vibe_module.API_TIMEOUT_S + 2,
            vibe_module.API_RETRY_BUDGET_S + 30 + vibe_module.API_TIMEOUT_S + 2,
        ) == 272

    def test_the_declared_bound_sits_under_the_slot_timeout(self, provider):
        """§3.3's invariant for any inner bound."""
        assert provider.internal_retries.max_wall_s < provider.default_timeout

    def test_effort_levels_are_the_validators_own_set(self, provider):
        """Read off the installed pydantic validator, not from documentation."""
        assert vibe_module.REASONING_EFFORTS == frozenset(
            {"off", "low", "medium", "high", "max"}
        )

    def test_usage_stays_none(self, provider):
        """The transcript entries carry no token fields — §3.4's default."""
        assert provider.usage_support == "none"
        assert provider.parse_usage("anything", "") is None

    def test_structured_output_is_not_declared(self, provider):
        assert provider.structured_output_support is False

    def test_the_verified_version_is_the_capture_version(self, provider):
        # Re-captured on 2.26.0 on 2026-10-08; 2.24.1 (the first capture)
        # stays the floor.
        assert provider.verified_version == "2.26.0"
        assert provider.min_version == "2.24.1"


class TestClineProvider:
    """Tests for the ClineProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a ClineProvider instance."""
        return ClineProvider()

    @staticmethod
    def _jsonl(text, finish_reason="completed"):
        """Build representative Cline --json JSONL stdout ending in run_result."""
        lines = [
            json.dumps({"ts": 1, "type": "hook_event", "name": "task_started"}),
            json.dumps({"ts": 2, "type": "agent_event", "kind": "text"}),
            json.dumps({
                "ts": 3,
                "type": "run_result",
                "finishReason": finish_reason,
                "text": text,
            }),
        ]
        return "\n".join(lines)

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "cline"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_cline_parse_run_result_array(self, provider):
        """Extract a JSON array from the run_result text field."""
        inner_data = [
            {"title": "Finding 1", "desc": "Description", "importance": "medium"}
        ]
        stdout = self._jsonl(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_cline_parse_run_result_object(self, provider):
        """Extract a JSON object from the run_result text field."""
        inner_data = {"first_heading": "Probe Widget"}
        stdout = self._jsonl(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_cline_parse_prose_with_json(self, provider):
        """Extract embedded JSON when prose surrounds it in the text field."""
        inner_json = {"result": "success"}
        stdout = self._jsonl(f"Here is the JSON you asked for: {json.dumps(inner_json)}")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_cline_parse_code_block_json(self, provider):
        """Extract JSON from a code block in the text field."""
        inner_json = [{"task": "T001", "status": "done"}]
        text = f"Analysis complete:\n\n```json\n{json.dumps(inner_json)}\n```\n"
        stdout = self._jsonl(text)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_cline_parse_last_run_result_wins(self, provider):
        """Use the LAST run_result line when multiple are present."""
        stdout = "\n".join([
            json.dumps({"type": "run_result", "finishReason": "completed", "text": "[1]"}),
            json.dumps({"type": "run_result", "finishReason": "completed", "text": "[1, 2, 3]"}),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_cline_parse_empty_text(self, provider):
        """Handle an empty text field in run_result."""
        stdout = self._jsonl("")

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_cline_parse_finish_reason_error(self, provider):
        """finishReason == "error" returns a failure carrying the error text."""
        stdout = self._jsonl("Provider quota exceeded", finish_reason="error")

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["error"] == "Provider quota exceeded"
        assert result["data"] is None

    def test_cline_parse_non_jsonl_is_not_extracted_raw(self, provider):
        """No ``run_result``: only the reducer's text is parsed, never the raw
        stdout (plans/ask-answer-recovery-and-false-success.md D4.4).

        This used to pin the opposite — a raw-stdout extraction that found
        ``[1, 2, 3]`` in a banner. The same fallback found the event lines of a
        cut stream and returned them as ``success: True`` data.
        """
        stdout = "Some banner text [1, 2, 3] trailing text"

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["error"] == "No run_result event found in output"
        assert result["incomplete"] is True

    def test_cline_parse_garbage_stdout(self, provider):
        """Handle stdout with no run_result and no extractable JSON."""
        stdout = "not valid json at all"

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result
        assert "raw" in result

    def test_cline_build_command_with_provider_prefix(self, provider):
        """Split <cline-provider>/<model-id> into -P and -m."""
        cmd = provider.build_command("Analyze this", "openrouter/z-ai/glm-5.2")

        assert cmd == [
            "cline",
            "--json",
            "-P",
            "openrouter",
            "-m",
            "z-ai/glm-5.2",
            "Analyze this",
        ]

    def test_cline_build_command_without_provider_prefix(self, provider):
        """A model name without "/" is passed straight to -m (no -P)."""
        cmd = provider.build_command("Analyze this", "some-model")

        assert cmd == ["cline", "--json", "-m", "some-model", "Analyze this"]

    def test_cline_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the five valid efforts."""
        from utils.providers.cline import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {"none", "low", "medium", "high", "xhigh"}

    def test_cline_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --thinking pair right after --json.

        The suffix is stripped off the FULL string first; the remaining base
        then splits on the first "/" into -P/-m as usual.
        """
        cmd = provider.build_command("Analyze this", "openrouter/z-ai/glm-5.2:high")

        assert cmd == [
            "cline",
            "--json",
            "--thinking",
            "high",
            "-P",
            "openrouter",
            "-m",
            "z-ai/glm-5.2",
            "Analyze this",
        ]

    def test_cline_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --thinking form."""
        for effort in ("none", "low", "medium", "high", "xhigh"):
            cmd = provider.build_command("Analyze this", f"openrouter/z-ai/glm-5.2:{effort}")

            assert cmd == [
                "cline",
                "--json",
                "--thinking",
                effort,
                "-P",
                "openrouter",
                "-m",
                "z-ai/glm-5.2",
                "Analyze this",
            ], f"effort {effort!r} did not build the expected command"

    def test_cline_build_command_effort_without_provider_prefix(self, provider):
        """An :effort suffix on a model without "/" still yields --thinking, no -P."""
        cmd = provider.build_command("Analyze this", "some-model:high")

        assert cmd == [
            "cline",
            "--json",
            "--thinking",
            "high",
            "-m",
            "some-model",
            "Analyze this",
        ]

    def test_cline_build_command_effort_passthrough(self, provider):
        """Non-effort suffixes stay part of the model id; no --thinking flag.

        ":free" (openrouter variant) and ":minimal" (not in cline's whitelist)
        must survive intact in the -m argument.
        """
        cases = [
            (
                "openrouter/z-ai/glm-5.2",
                ["cline", "--json", "-P", "openrouter", "-m", "z-ai/glm-5.2", "Analyze this"],
            ),
            (
                "openrouter/deepseek/deepseek-r1:free",
                ["cline", "--json", "-P", "openrouter", "-m", "deepseek/deepseek-r1:free", "Analyze this"],
            ),
            (
                "openrouter/z-ai/glm-5.2:minimal",
                ["cline", "--json", "-P", "openrouter", "-m", "z-ai/glm-5.2:minimal", "Analyze this"],
            ),
        ]
        for model, expected in cases:
            cmd = provider.build_command("Analyze this", model)

            assert cmd == expected, f"model {model!r} was not passed through verbatim"

    @patch("shutil.which")
    def test_cline_is_available_true(self, mock_which, provider):
        """Test is_available returns True when cline is found."""
        mock_which.return_value = "/usr/bin/cline"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("cline")

    @patch("shutil.which")
    def test_cline_is_available_false(self, mock_which, provider):
        """Test is_available returns False when cline is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("cline")


class TestGooseProvider:
    """Tests for the GooseProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a GooseProvider instance."""
        return GooseProvider()

    @staticmethod
    def _envelope(text, total_tokens=100):
        """Build representative goose --output-format json stdout."""
        return json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "the prompt"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            ],
            "metadata": {
                "total_tokens": total_tokens,
                "input_tokens": 50,
                "output_tokens": 50,
                "status": "completed",
            },
        })

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "goose"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_goose_parse_envelope_array(self, provider):
        """Extract a JSON array from the assistant message text."""
        inner_data = [
            {"title": "Finding 1", "desc": "Description", "importance": "medium"}
        ]
        stdout = self._envelope(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_goose_parse_envelope_object(self, provider):
        """Extract a JSON object from the assistant message text."""
        inner_data = {"first_heading": "Probe Widget"}
        stdout = self._envelope(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_goose_parse_code_block_json(self, provider):
        """Extract JSON from a ```json code block in the assistant text."""
        inner_json = [{"task": "T001", "status": "done"}]
        text = f"Analysis complete:\n\n```json\n{json.dumps(inner_json)}\n```\n"
        stdout = self._envelope(text)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_goose_parse_prose_with_json(self, provider):
        """Extract embedded JSON when prose surrounds it in the text."""
        inner_json = {"result": "success"}
        stdout = self._envelope(f"Here is the JSON you asked for: {json.dumps(inner_json)}")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_goose_parse_last_assistant_message_wins(self, provider):
        """Use the LAST assistant message when multiple are present."""
        stdout = json.dumps({
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "[1]"}]},
                {"role": "user", "content": [{"type": "toolResponse", "text": "ok"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "[1, 2, 3]"}]},
            ],
            "metadata": {"total_tokens": 10, "status": "completed"},
        })

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_goose_parse_mixed_content_types(self, provider):
        """Only type == "text" parts are concatenated (thinking is skipped)."""
        stdout = json.dumps({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "text": "let me think {\"wrong\": 1}"},
                        {"type": "toolRequest", "id": "t1"},
                        {"type": "text", "text": "[4, 5, 6]"},
                    ],
                },
            ],
            "metadata": {"total_tokens": 10, "status": "completed"},
        })

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [4, 5, 6]

    def test_goose_parse_provider_error_text(self, provider):
        """goose exits 0 on provider errors; detect "Ran into this error:"."""
        error_text = "Ran into this error: 401 Unauthorized. Please retry if you think this is a transient error."
        stdout = json.dumps({
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": error_text}]},
            ],
            "metadata": {"total_tokens": None, "status": "completed"},
        })

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["error"] == error_text
        assert result["data"] is None

    def test_goose_parse_no_assistant_message(self, provider):
        """Handle an envelope with no assistant message."""
        stdout = json.dumps({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "the prompt"}]},
            ],
            "metadata": {"total_tokens": 10, "status": "completed"},
        })

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_goose_parse_empty_messages(self, provider):
        """Handle an envelope with an empty messages list."""
        stdout = json.dumps({"messages": [], "metadata": {"status": "completed"}})

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_goose_parse_empty_text(self, provider):
        """Handle an empty text field in the assistant message."""
        stdout = self._envelope("")

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_goose_parse_non_json_stdout_fallback(self, provider):
        """Fallback extraction when stdout is not a JSON envelope but embeds JSON."""
        stdout = "Some banner text [1, 2, 3] trailing text"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_goose_parse_garbage_stdout(self, provider):
        """Handle stdout with no envelope and no extractable JSON."""
        stdout = "not valid json at all"

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result
        assert "raw" in result

    def test_goose_build_command_with_provider_prefix(self, provider):
        """Split <goose-provider>/<model-id> into --provider and --model."""
        cmd = provider.build_command("Analyze this", "openrouter/z-ai/glm-5.2")

        assert cmd == [
            "goose",
            "run",
            "--no-session",
            "-q",
            "--output-format",
            "json",
            "--no-profile",
            "--max-turns",
            "25",
            "--with-builtin",
            "developer",
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.2",
            "-t",
            "Analyze this",
        ]

    def test_goose_build_command_without_provider_prefix(self, provider):
        """A model name without "/" goes straight to --model (no --provider)."""
        cmd = provider.build_command("Analyze this", "some-model")

        assert cmd == [
            "goose",
            "run",
            "--no-session",
            "-q",
            "--output-format",
            "json",
            "--no-profile",
            "--max-turns",
            "25",
            "--with-builtin",
            "developer",
            "--model",
            "some-model",
            "-t",
            "Analyze this",
        ]

    def test_goose_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the seven valid efforts."""
        from utils.providers.goose import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {"off", "none", "low", "medium", "high", "max", "xhigh"}

    def test_goose_build_command_effort_suffix_stripped(self, provider):
        """:effort is stripped BEFORE the provider/model split; argv has no effort flag."""
        cmd = provider.build_command("Analyze this", "openrouter/z-ai/glm-5.2:high")

        assert cmd == [
            "goose",
            "run",
            "--no-session",
            "-q",
            "--output-format",
            "json",
            "--no-profile",
            "--max-turns",
            "25",
            "--with-builtin",
            "developer",
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.2",
            "-t",
            "Analyze this",
        ]

    def test_goose_build_command_effort_suffix_without_provider_prefix(self, provider):
        """:effort stripping also applies to models without a provider prefix."""
        cmd = provider.build_command("Analyze this", "some-model:high")

        assert cmd == [
            "goose",
            "run",
            "--no-session",
            "-q",
            "--output-format",
            "json",
            "--no-profile",
            "--max-turns",
            "25",
            "--with-builtin",
            "developer",
            "--model",
            "some-model",
            "-t",
            "Analyze this",
        ]

    def test_goose_build_command_unknown_suffix_passthrough(self, provider):
        """":free" is not an effort: the whole remainder stays the model id."""
        cmd = provider.build_command("Analyze this", "openrouter/deepseek/deepseek-r1:free")

        assert cmd == [
            "goose",
            "run",
            "--no-session",
            "-q",
            "--output-format",
            "json",
            "--no-profile",
            "--max-turns",
            "25",
            "--with-builtin",
            "developer",
            "--provider",
            "openrouter",
            "--model",
            "deepseek/deepseek-r1:free",
            "-t",
            "Analyze this",
        ]

    def test_goose_get_env_with_effort_suffix(self, provider):
        """An :effort suffix is delivered via the GOOSE_THINKING_EFFORT env var."""
        for effort in ("off", "none", "low", "medium", "high", "max", "xhigh"):
            env = provider.get_env(f"openrouter/z-ai/glm-5.2:{effort}")

            assert env.get("GOOSE_THINKING_EFFORT") == effort, (
                f"effort {effort!r} was not mapped to GOOSE_THINKING_EFFORT"
            )

    def test_goose_get_env_without_effort_suffix(self, provider):
        """Without a whitelisted suffix GOOSE_THINKING_EFFORT is left unset."""
        for model in (
            "openrouter/z-ai/glm-5.2",
            "openrouter/deepseek/deepseek-r1:free",
            "some-model",
        ):
            env = provider.get_env(model)

            assert "GOOSE_THINKING_EFFORT" not in env, (
                f"model {model!r} must not set GOOSE_THINKING_EFFORT"
            )

    @patch("shutil.which")
    def test_goose_is_available_true(self, mock_which, provider):
        """Test is_available returns True when goose is found."""
        mock_which.return_value = "/usr/bin/goose"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("goose")

    @patch("shutil.which")
    def test_goose_is_available_false(self, mock_which, provider):
        """Test is_available returns False when goose is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("goose")


class TestAiderProvider:
    """Tests for the AiderProvider class."""

    @pytest.fixture
    def provider(self):
        """Create an AiderProvider instance."""
        return AiderProvider()

    BANNER = (
        "\nAider v0.86.2\n"
        "Model: openrouter/z-ai/glm-5.2 with whole edit format\n"
        "Git repo: .git with 12 files\n"
        "Repo-map: using 4096 tokens, auto refresh\n"
    )

    @classmethod
    def _stdout(cls, answer, thinking=None, tokens_line=True):
        """Build representative aider --no-pretty stdout."""
        parts = [cls.BANNER, "\n"]
        if thinking is not None:
            parts.append(f"--------------\n► **THINKING**\n\n{thinking}\n\n")
        parts.append(f"------------\n► **ANSWER**\n\n{answer}\n")
        if tokens_line:
            parts.append(
                "\nTokens: 210 sent, 703 received. "
                "Cost: $0.0023 message, $0.0023 session.\n"
            )
        return "".join(parts)

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "aider"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_aider_parse_answer_array(self, provider):
        """Extract a JSON array from the text after the ANSWER marker."""
        inner_data = [
            {"title": "Finding 1", "desc": "Description", "importance": "medium"}
        ]
        stdout = self._stdout(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_aider_parse_answer_object(self, provider):
        """Extract a JSON object from the text after the ANSWER marker."""
        inner_data = {"first_heading": "Probe Widget"}
        stdout = self._stdout(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_aider_parse_thinking_decoy_ignored(self, provider):
        """Decoy JSON in the THINKING block is ignored; post-ANSWER wins."""
        stdout = self._stdout(
            "[4, 5, 6]",
            thinking='Draft of the reply:\n[1, 2, 3]\nAlso {"wrong": 1} here.',
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [4, 5, 6]

    def test_aider_parse_multiple_answer_markers_last_wins(self, provider):
        """When several ANSWER markers appear, only the LAST one is parsed."""
        first = "------------\n► **ANSWER**\n\n[1, 2, 3]\n\n"
        stdout = self.BANNER + first + self._stdout("[7, 8, 9]")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [7, 8, 9]

    def test_aider_parse_strips_tokens_cost_line(self, provider):
        """The trailing "Tokens: ... Cost: ..." line does not break parsing."""
        stdout = self._stdout("[1, 2, 3]", tokens_line=True)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_aider_parse_prose_with_json(self, provider):
        """Extract embedded JSON when prose surrounds it in the answer."""
        inner_json = {"result": "success"}
        stdout = self._stdout(
            f"Here is the JSON you asked for: {json.dumps(inner_json)}"
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_aider_parse_code_block_json(self, provider):
        """Extract JSON from a ```json code block in the answer."""
        inner_json = [{"task": "T001", "status": "done"}]
        stdout = self._stdout(
            f"Analysis complete:\n\n```json\n{json.dumps(inner_json)}\n```"
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_aider_parse_file_write_envelope_unwrapped(self, provider):
        """A pseudo file-write envelope ({"file_path", "content"}) is unwrapped.

        In read-only /ask mode some models emulate the requested file write
        by answering with the target path and the JSON as a string content
        field (observed live with kimi-k2.7-code).
        """
        inner = [{"title": "Finding", "desc": "D", "importance": "high"}]
        envelope = {"file_path": "/some/out.json", "content": json.dumps(inner)}
        stdout = self._stdout(json.dumps(envelope))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner

    def test_aider_parse_code_interpreter_envelope_unwrapped(self, provider):
        """A hallucinated code_interpreter call ({"code": ...}) is unwrapped."""
        inner = [{"title": "Finding", "desc": "D", "importance": "high"}]
        code = (
            'import json\nissues = ' + json.dumps(inner, indent=2)
            + '\nwith open("/out.json", "w") as f:\n    json.dump(issues, f)\n'
        )
        stdout = self._stdout(json.dumps({"code": code}))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner

    def test_aider_parse_ordinary_object_not_unwrapped(self, provider):
        """A dict with extra keys (a real answer) is NOT treated as envelope."""
        data = {"content": "[1]", "title": "x", "extra": True}
        stdout = self._stdout(json.dumps(data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_aider_parse_wrapped_json_mid_string_keeps_the_break(self, provider):
        """Hard-wrapped JSON (literal newline INSIDE a string) parses, break kept.

        This documents why get_env() sets COLUMNS=10000: without it aider
        wraps output at ~80 columns and long string values get literal
        newlines inserted. Until 2026-10-05 the strict parse rejected the
        whole document; ``extract_json_from_text`` now tolerates raw control
        characters inside strings (``_loads_reply``), so the answer survives,
        but the inserted break is kept verbatim in the value — the wrap still
        alters what the model wrote, which is why the variable stays.
        """
        wrapped = (
            '[\n  {\n    "title": "Missing Input Validation",\n'
            '    "desc": "The function responsible for parsing user-supplied \n'
            'configuration parameters lacks proper validation checks."\n  }\n]'
        )
        stdout = self._stdout(wrapped)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"][0]["desc"] == (
            "The function responsible for parsing user-supplied \n"
            "configuration parameters lacks proper validation checks."
        )

    def test_aider_parse_wrapped_between_tokens_ok(self, provider):
        """Newlines between JSON tokens (not inside strings) parse fine."""
        stdout = self._stdout('[\n  1,\n  2,\n  3\n]')

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_aider_parse_no_answer_marker_fallback(self, provider):
        """No ANSWER marker: fall back to extraction on the full stdout."""
        stdout = self.BANNER + "\nSome text [1, 2, 3] trailing text\n"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_aider_parse_auth_error_stdout(self, provider):
        """aider exits 0 on auth failure; no extractable JSON -> failure."""
        stdout = (
            self.BANNER
            + "\nlitellm.AuthenticationError: AuthenticationError: "
            "OpenrouterException - Invalid credentials. Check your API key.\n"
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "litellm.AuthenticationError" in result["error"]
        assert result["data"] is None

    def test_aider_parse_empty_answer_after_marker(self, provider):
        """An empty answer after the ANSWER marker is an explicit failure."""
        stdout = self._stdout("", tokens_line=True)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_aider_parse_empty_stdout(self, provider):
        """Empty stdout is an explicit failure."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert "error" in result
        assert result["data"] is None

    def test_aider_build_command(self, provider):
        """Exact argv: model verbatim, /ask prefix, all headless flags."""
        cmd = provider.build_command("Analyze this", "openrouter/z-ai/glm-5.2")

        assert cmd == [
            "aider",
            "--model",
            "openrouter/z-ai/glm-5.2",
            "--message",
            "/ask Analyze this",
            "--no-auto-commits",
            "--no-pretty",
            "--no-stream",
            "--no-check-update",
            "--no-show-model-warnings",
            "--no-analytics",
            "--no-gitignore",
            "--no-show-release-notes",
            "--no-detect-urls",
            "--no-fancy-input",
            "--chat-history-file",
            os.devnull,
            "--input-history-file",
            os.devnull,
        ]

    def test_aider_build_command_read_flags_for_existing_paths(self, provider, tmp_path):
        """Existing absolute file paths in the prompt are passed via --read."""
        plan = tmp_path / "my-plan.md"
        plan.write_text("# Plan", encoding="utf-8")
        missing = tmp_path / "review-plan" / "aider_model.json"  # not yet written
        prompt = (
            f"## Plan File\n{plan}\n\nRead this file.\n"
            f"Write your JSON output to this file: {missing}\n"
            f"Also mentioned twice: {plan}."
        )

        cmd = provider.build_command(prompt, "openrouter/z-ai/glm-5.2")

        assert cmd.count("--read") == 1  # deduped; missing path skipped
        assert cmd[cmd.index("--read") + 1] == str(plan)
        assert str(missing) not in cmd

    def test_aider_build_command_no_read_flags_without_paths(self, provider):
        """No --read args when the prompt mentions no existing files."""
        cmd = provider.build_command(
            "Analyze this plan about /nonexistent/path/plan.md", "some-model"
        )

        assert "--read" not in cmd

    def test_aider_build_command_model_verbatim(self, provider):
        """Model IDs are litellm specs passed verbatim — no splitting."""
        cmd = provider.build_command("Hi", "openrouter/moonshotai/kimi-k2.7-code")

        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "openrouter/moonshotai/kimi-k2.7-code"
        assert "--provider" not in cmd
        assert "-P" not in cmd

    def test_aider_get_env(self, provider):
        """get_env sets BROWSER (webbrowser no-op) and COLUMNS (no wrap)."""
        env = provider.get_env("openrouter/z-ai/glm-5.2")

        assert env == {"BROWSER": "true", "COLUMNS": "10000"}

    def test_aider_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the six valid efforts."""
        from utils.providers.aider import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {"none", "minimal", "low", "medium", "high", "xhigh"}

    def test_aider_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --reasoning-effort pair after --model <base>."""
        cmd = provider.build_command("Analyze this", "openrouter/z-ai/glm-5.2:high")

        assert cmd == [
            "aider",
            "--model",
            "openrouter/z-ai/glm-5.2",
            "--reasoning-effort",
            "high",
            "--message",
            "/ask Analyze this",
            "--no-auto-commits",
            "--no-pretty",
            "--no-stream",
            "--no-check-update",
            "--no-show-model-warnings",
            "--no-analytics",
            "--no-gitignore",
            "--no-show-release-notes",
            "--no-detect-urls",
            "--no-fancy-input",
            "--chat-history-file",
            os.devnull,
            "--input-history-file",
            os.devnull,
        ]

    def test_aider_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --reasoning-effort form."""
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
            cmd = provider.build_command("Analyze this", f"openrouter/z-ai/glm-5.2:{effort}")

            assert cmd == [
                "aider",
                "--model",
                "openrouter/z-ai/glm-5.2",
                "--reasoning-effort",
                effort,
                "--message",
                "/ask Analyze this",
                "--no-auto-commits",
                "--no-pretty",
                "--no-stream",
                "--no-check-update",
                "--no-show-model-warnings",
                "--no-analytics",
                "--no-gitignore",
                "--no-show-release-notes",
                "--no-detect-urls",
                "--no-fancy-input",
                "--chat-history-file",
                os.devnull,
                "--input-history-file",
                os.devnull,
            ], f"effort {effort!r} did not build the expected command"

    def test_aider_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --reasoning-effort flag."""
        for model in (
            "openrouter/z-ai/glm-5.2",
            "openrouter/deepseek/deepseek-r1:free",
            "openrouter/z-ai/glm-5.2:turbo",
        ):
            cmd = provider.build_command("Analyze this", model)

            assert cmd == [
                "aider",
                "--model",
                model,
                "--message",
                "/ask Analyze this",
                "--no-auto-commits",
                "--no-pretty",
                "--no-stream",
                "--no-check-update",
                "--no-show-model-warnings",
                "--no-analytics",
                "--no-gitignore",
                "--no-show-release-notes",
                "--no-detect-urls",
                "--no-fancy-input",
                "--chat-history-file",
                os.devnull,
                "--input-history-file",
                os.devnull,
            ], f"model {model!r} was not passed through verbatim"

    @patch("shutil.which")
    def test_aider_is_available_true(self, mock_which, provider):
        """Test is_available returns True when aider is found."""
        mock_which.return_value = "/usr/bin/aider"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("aider")

    @patch("shutil.which")
    def test_aider_is_available_false(self, mock_which, provider):
        """Test is_available returns False when aider is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("aider")

    def test_aider_build_command_uses_os_devnull(self, provider):
        """History files are routed to os.devnull (portable null device)."""
        cmd = provider.build_command("Analyze this", "some-model")

        assert cmd[cmd.index("--chat-history-file") + 1] == os.devnull
        assert cmd[cmd.index("--input-history-file") + 1] == os.devnull

    def test_aider_abs_path_re_matches_full_backslash_windows_path(self):
        """The regex matches an entire backslash Windows path, not just C:\\."""
        from utils.providers.aider import _ABS_PATH_RE

        matches = _ABS_PATH_RE.findall(r"Read C:\Users\foo\bar.md for context.")

        assert matches == [r"C:\Users\foo\bar.md"]

    def test_aider_abs_path_re_matches_forward_slash_windows_path(self):
        """Drive-letter paths with forward slashes match in full too."""
        from utils.providers.aider import _ABS_PATH_RE

        matches = _ABS_PATH_RE.findall("Read C:/Users/foo/bar.md for context.")

        assert matches == ["C:/Users/foo/bar.md"]

    def test_aider_abs_path_re_posix_branch_unchanged(self):
        """POSIX paths match exactly as with the historical pattern."""
        from utils.providers.aider import _ABS_PATH_RE

        matches = _ABS_PATH_RE.findall(
            "See /home/user/plans/my-plan.md and /tmp/out.json"
        )

        assert matches == ["/home/user/plans/my-plan.md", "/tmp/out.json"]

    def test_aider_extract_read_files_full_windows_path(self, provider, monkeypatch):
        """_extract_read_files yields the whole Windows path, not a truncated C:\\."""
        win_path = r"C:\Users\foo\bar.md"
        monkeypatch.setattr(Path, "is_file", lambda self: str(self) == win_path)

        found = provider._extract_read_files(f"## Plan File\n{win_path}\n\nRead it.")

        assert found == [win_path]

    def test_aider_extract_read_files_posix_unchanged(self, provider, tmp_path):
        """Existing POSIX extraction behavior is unchanged by the Windows branch."""
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan", encoding="utf-8")

        found = provider._extract_read_files(
            f"Read {plan} and the missing /nonexistent/file.md"
        )

        assert found == [str(plan)]


class TestAiderGlossIsNarrowedWhereItIsWritten:
    """C7: aider's own 5xx gloss stops classifying, the vendor's words do not.

    The two moved fixture rows are graded by `test_provider_fixture_rows.py`.
    What is only assertable here is the half no capture covers: that narrowing
    aider's gloss did **not** weaken `overloaded` for the vendor that genuinely
    sends it. There is no Anthropic-through-aider capture in the tree, so the
    529 case below is built from litellm's rendering of one rather than read off
    disk, and it is labelled as such.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "aider"

    #: An Anthropic 529 as litellm renders it into aider's stdout: the vendor's
    #: own error body on one line, aider's gloss on the next. Synthetic (no such
    #: capture is committed); every other assertion in this class reads bytes.
    ANTHROPIC_529 = (
        "\nAider v0.86.2\n"
        "Model: anthropic/claude-sonnet-4-5 with whole edit format\n\n"
        "litellm.InternalServerError: AnthropicException - "
        '{"type":"error","error":{"type":"overloaded_error",'
        '"message":"Overloaded"}}\n'
        "The API provider's servers are down or overloaded.\n"
        "Retrying in 0.2 seconds...\n"
    )

    def _capture(self, stem):
        return (self.FIXTURES / f"{stem}.stdout.txt").read_text(encoding="utf-8")

    def test_the_gloss_alone_no_longer_says_rate_limited(self):
        """The two rows the excision moves, from the committed bytes."""
        provider = AiderProvider()

        assert provider.classify_failure(self._capture("truncated"), "", 0) == \
            "connection_error"
        assert provider.classify_failure(
            self._capture("exit-zero-failure"), "", 0
        ) == "runner_error"

    def test_and_the_gloss_is_still_in_those_bytes(self):
        """The fix is an excision at classification time, not an edited capture."""
        for stem in ("truncated", "exit-zero-failure"):
            assert "The API provider's servers are down or overloaded." in \
                self._capture(stem)

    def test_a_vendor_that_really_says_overloaded_is_still_rate_limited(self):
        """The shared `overloaded` pattern is intact — this is C7's whole point.

        Anthropic's genuine 529 is what `overloaded` carries in
        `base.rate_limit_patterns`, and it survives the excision because the
        vendor writes it on its own line.
        """
        assert AiderProvider().classify_failure(self.ANTHROPIC_529, "", 0) == \
            STATUS_RATE_LIMITED

    def test_aiders_measured_429_is_untouched(self):
        """The committed 429 capture, which the excision must not reach."""
        assert AiderProvider().classify_failure(
            self._capture("rate-limit"), "", 0
        ) == STATUS_RATE_LIMITED

    def test_the_other_glosses_stay_because_they_are_the_only_evidence(self):
        """Why the excision names one sentence rather than aider's gloss shape.

        `not able to authenticate` — this adapter's own declared auth pattern —
        IS the 401 gloss, and no other auth wording appears in that capture. An
        excision keyed on "The API provider …" would turn a rung-1 auth failure
        into a retryable runner error.
        """
        assert AiderProvider().classify_failure(
            self._capture("auth-error"), "", 0
        ) == STATUS_AUTH_FAILED

    def test_the_excision_degrades_to_its_input(self):
        """A capture that is nothing but the gloss must not classify as None.

        An empty haystack is "nothing recognized", i.e. transport-shaped, so an
        excision that consumed a whole capture would launder a failure into a
        retry. The reduction returns its input instead.
        """
        gloss_only = "The API provider's servers are down or overloaded.\n"

        assert AiderProvider()._drop_adapter_gloss(gloss_only) == gloss_only

    def test_the_excision_only_ever_removes(self):
        """Removal-only, the same contract `vendor_text` states for the hook."""
        narrowed = AiderProvider()._drop_adapter_gloss(self._capture("truncated"))
        original = self._capture("truncated")

        assert len(narrowed) < len(original)
        for line in narrowed.splitlines():
            assert line in original


# Per-provider prompt transport table: the transport each adapter's FAN-OUT path
# uses by default. Every row but one declares `argv`, and for those rows what is
# pinned is that the adapter renders the prompt as exactly one argv element, so
# llm_client's Windows argv-length check still applies to that path.
#
# `omp` is the exception and the reason this sentence is no longer "every adapter
# declares argv": it declares `stdin`, because `invoke_with_file_output` — the
# function every Lane A review phase goes through — takes no transport argument
# and forwards none, so an `argv` default would leave every review prompt on the
# path measured failing (200 KB on argv is E2BIG from the kernel before omp
# starts; the same prompt over stdin is answered in one call). A `stdin` row is
# pinned the other way round: the prompt appears on NEITHER argv nor twice.
#
# It is NOT a claim that no other adapter can do better: claude-code, codex,
# opencode and qwen-code each declare
# `supported_prompt_transports = ("argv", "stdin")`, which is the separate member
# a *caller* may ask for. The header used to say "none has a documented stdin or
# prompt-file mode used anywhere in this codebase", which had been false for
# three adapters for some time.
# An adapter that changes its fan-out default must update both its
# prompt_transport and this table.
TRANSPORT_CASES = [
    (AgyProvider, "argv", "Gemini 3.1 Pro (High)"),
    (AiderProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (ClaudeCodeProvider, "argv", "sonnet"),
    (ClineProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (CodexProvider, "argv", "gpt-5.5"),
    # copilot renders the prompt as one argv element like every row here,
    # AND declares a stdin transport a caller may ask for: piped stdin with
    # the `-p` PAIR omitted entirely carried a 30 KB prompt (§6.4, measured).
    (CopilotProvider, "argv", "openai/gpt-4.1-mini"),
    (CursorAgentProvider, "argv", "auto"),
    (GeminiProvider, "argv", "gemini-2.5-pro"),
    # kimi is argv-only by MEASUREMENT, not by default: `-p -` delivers the
    # literal "-" rather than reading stdin, `-p @file` delivers the literal
    # path, and `--prompt-file` does not exist (harness-expansion.md §6.4).
    (KimiProvider, "argv", "moonshotai/kimi-k2-thinking"),
    (GooseProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (GrokProvider, "argv", "grok-4"),
    (KiloCodeProvider, "argv", "openai-native/gpt-5.5"),
    (MuseProvider, "argv", "muse-spark-1.2"),
    # The one `stdin` row, and the first adapter whose fan-out default is not
    # argv. `argv` stays SUPPORTED for a caller that asks for it and renders
    # `-- <prompt>` last; the default is stdin because the harness has no way to
    # ask for it on the path every review prompt takes (§2.5).
    (OmpProvider, "stdin", "openrouter/openai/gpt-5.6-luna"),
    (OpenCodeProvider, "argv", "openai/gpt-5.5"),
    (PiProvider, "argv", "anthropic/claude-opus-4-8"),
    # qwen-code renders the prompt as one argv element like every row here,
    # AND declares a stdin transport a caller may ask for (§6.4, measured).
    (QwenCodeProvider, "argv", "qwen/qwen3.8-max"),
    # vibe renders the prompt as one argv element like every row here, AND
    # declares a stdin transport -- but its stdin form is the odd one out and the
    # difference is load-bearing: `-p` STAYS and merely loses its value, because
    # dropping the flag entirely launches the interactive TUI and hangs forever
    # rather than degrading (§6.4, measured with an 8,427-byte prompt).
    (VibeProvider, "argv", "vibe-default"),
]


class TestPromptTransport:
    """Locks the per-provider prompt transport decision (see TRANSPORT_CASES)."""

    PROMPT = "Analyze this plan carefully"

    @pytest.mark.parametrize(
        "provider_cls,transport,model",
        TRANSPORT_CASES,
        ids=[cls.__name__ for cls, _, _ in TRANSPORT_CASES],
    )
    def test_declared_transport_matches_build_command(
        self, provider_cls, transport, model
    ):
        """prompt_transport is declared as decided AND matches the built argv.

        The assertion is per row rather than the blanket `transport == "argv"`
        it used to be: omp's fan-out default is `stdin`, and grading it against
        argv would have made the first non-argv adapter unrepresentable in the
        table that exists to pin exactly that decision. An `argv` row keeps the
        one-element check; a `stdin` row asserts the mirror image, that the
        prompt reaches argv not at all and `stdin_text` carries it whole —
        omp answers a prompt given on both as TWO user messages, so "not on
        argv" is the assertion and not a tidiness preference.
        """
        provider = provider_cls()

        assert provider.prompt_transport == transport

        if transport == "argv":
            cmd = provider.build_command(self.PROMPT, model)
            # The full prompt travels as exactly ONE argv element (aider
            # prefixes "/ask " but keeps the prompt intact in that element).
            prompt_args = [arg for arg in cmd if self.PROMPT in arg]
            assert len(prompt_args) == 1
            return

        assert transport == "stdin"
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, model, "stdin"
        )
        assert [arg for arg in argv if self.PROMPT in arg] == []
        assert stdin_text == self.PROMPT

    def test_table_covers_every_registered_provider(self):
        """The transport table stays in sync with the provider registry."""
        from utils.provider_registry import _PROVIDERS

        registered = {type(p) for p in _PROVIDERS.values()}

        assert registered == {cls for cls, _, _ in TRANSPORT_CASES}


def _registered_provider_names():
    """Every provider id in the registry, so the sweeps below cannot miss a new one."""
    from utils.provider_registry import _PROVIDERS

    return list(_PROVIDERS)


class TestSubagentRunnerDeclarations:
    """Every adapter's D3 capability declarations must match its members.

    The capability data cannot be allowed to drift from the code in either
    direction: an adapter that gains `build_agent_command()` without declaring
    the profile would be rejected as a runner it can actually serve, and one that
    declares a profile it does not implement would be dispatched and fail at
    runtime on a user's host.
    """

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_declared_profiles_match_implemented_members(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        implements = (
            type(provider).build_agent_command is not LLMProvider.build_agent_command
        )
        assert implements == ("agent-readonly" in provider.agent_profiles), (
            f"{name}: build_agent_command implemented={implements} but "
            f"agent_profiles={provider.agent_profiles}"
        )
        # `generate` is every adapter's build_command and is always declared.
        assert "generate" in provider.agent_profiles

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_declared_write_profile_matches_supports_file_edits(self, name):
        """`agent-write` and `supports_file_edits` are one fact, declared twice.

        Both must agree AND the branch must actually render a command: an adapter
        that declares the profile but falls through to the base raise would be
        chosen as a write runner and fail on a user's repository, and one that
        implements it without declaring it would be refused for work it can do.
        """
        from utils.provider_registry import _PROVIDERS
        from utils.providers.base import PROFILE_AGENT_WRITE, ProfileUnsupported

        provider = _PROVIDERS[name]
        declares = PROFILE_AGENT_WRITE in provider.agent_profiles
        assert declares == provider.supports_file_edits, (
            f"{name}: agent_profiles declares agent-write={declares} but "
            f"supports_file_edits={provider.supports_file_edits}"
        )
        if declares:
            argv = provider.build_agent_command("p", "m", PROFILE_AGENT_WRITE)
            assert argv and argv[-1] == "p"
            # A non-argv transport is mandatory for the edit-capable invocation
            # (task 6 / D2d): apply and implement prompts are the largest the
            # skill builds.
            stdin_argv, stdin_text = provider.build_agent_command_for_transport(
                "p", "m", PROFILE_AGENT_WRITE, "stdin"
            )
            assert stdin_text == "p" and "p" not in stdin_argv
        else:
            with pytest.raises(ProfileUnsupported):
                provider.build_agent_command("p", "m", PROFILE_AGENT_WRITE)

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_declared_usage_matches_an_implemented_extractor(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        implements = type(provider).parse_usage is not LLMProvider.parse_usage
        assert implements == (provider.usage_support != "none"), (
            f"{name}: parse_usage implemented={implements} but "
            f"usage_support={provider.usage_support!r}"
        )
        if not implements:
            assert provider.parse_usage("anything at all", "") is None

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_a_runner_declaring_a_profile_also_declares_its_version_data(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        if "agent-readonly" not in provider.agent_profiles:
            return
        assert provider.version_cmd, f"{name}: no version_cmd"
        assert provider.min_version, f"{name}: no min_version"
        assert provider.verified_version, f"{name}: no verified_version"


CAPABILITIES_PATH = Path(__file__).parent.parent / "hosts" / "capabilities.yaml"


def _declared_host_runner_cells():
    """Every `runners:` cell in `hosts/capabilities.yaml`, as the YAML declares it.

    Read off the document rather than written out as a roster of hosts and
    runners, so a host or a runner cell added later is swept on the day it is
    added rather than on the day someone remembers to extend a list here.
    """
    import yaml

    document = yaml.safe_load(CAPABILITIES_PATH.read_text(encoding="utf-8"))
    return sorted(
        (host_id, runner, cell.get("usage"))
        for host_id, host in (document.get("hosts") or {}).items()
        for runner, cell in (host.get("runners") or {}).items()
    )


HOST_RUNNER_USAGE_CELLS = _declared_host_runner_cells()


class TestHostCapabilityUsageMatchesTheAdapter:
    """W5: the third declaration of one fact, tied to the other two.

    `TestSubagentRunnerDeclarations` above ties an adapter's `usage_support` to
    whether it implements `parse_usage`. Nothing tied either of those to the
    per-host `runners:` cell in `hosts/capabilities.yaml`, and it drifted for
    exactly that reason: the claude-code host declared
    `cursor-agent: {… usage: none}` long after the adapter had moved to
    `USAGE_PARTIAL`.

    Nothing reads the YAML value at run time today, which made the drift silent
    rather than harmless — the cell is rendered into
    `references/host-capabilities.md`, whose per-runner `usage=` line is what a
    human reads before choosing a runner. (`docs/hosts.md` renders the *default
    runner* column only; the repo README carries neither.)
    """

    @pytest.mark.parametrize(
        "host_id,runner,declared",
        HOST_RUNNER_USAGE_CELLS,
        ids=[f"{host}-{runner}" for host, runner, _ in HOST_RUNNER_USAGE_CELLS],
    )
    def test_the_cell_matches_the_named_adapters_declaration(
        self, host_id, runner, declared
    ):
        from utils.provider_registry import _PROVIDERS

        assert runner in _PROVIDERS, (
            f"{host_id}: `runners:` names {runner!r}, which is not an adapter in "
            f"the registry"
        )
        assert declared == _PROVIDERS[runner].usage_support, (
            f"{host_id}/{runner}: capabilities.yaml declares usage={declared!r} "
            f"but the adapter declares "
            f"usage_support={_PROVIDERS[runner].usage_support!r}. The YAML is the "
            f"rendered doc; the adapter is the code — move the cell."
        )

    def test_every_host_that_declares_runners_is_in_the_sweep(self):
        """A loader that quietly returned nothing would make the sweep vacuous."""
        import yaml

        document = yaml.safe_load(CAPABILITIES_PATH.read_text(encoding="utf-8"))
        with_runners = {
            host_id for host_id, host in document["hosts"].items()
            if host.get("runners")
        }

        assert HOST_RUNNER_USAGE_CELLS, "no runner cells were collected at all"
        assert with_runners == {host for host, _, _ in HOST_RUNNER_USAGE_CELLS}


class TestDeclaredOutputChannel:
    """`supports_file_output`: can this adapter write the result file at all?

    Two adapters run their CLI in a read-only chat mode on purpose, and the
    review prompts used to demand a file write from them regardless. Measured
    2026-08-18 on `cursor-agent:gpt-5.6-sol-xhigh`, three identical runs: the
    envelope came back `{"type":"result","subtype":"success","is_error":false,…}`
    with `result` = "I can't write the requested file while Ask mode is active.
    Switch to Agent mode and resend the request." — no file, no JSON, and the
    slot died on "JSON parsing failed from both file and stdout".
    """

    #: Pinned like the `agent-write` roster is pinned, and for the mirror-image
    #: reason: five of the seven are here because of a *mode the adapter chose*, so
    #: an adapter joining or leaving it is a deliberate change to how that CLI is
    #: invoked, never a passing thought. Those five name the argv atom that causes
    #: it, and `TestReadOnlyArgvMatchesTheDeclaration` in
    #: `test_provider_housekeeping.py` checks that atom against the argv the
    #: adapter really renders — this set is the *what*, that test is the *why*.
    #:
    #:   aider       `/ask <prompt>`         read-only chat mode
    #:   copilot     no `--allow-all-tools`  write requested, then denied
    #:   cursor-agent `--mode ask`           write refusal (measured 2026-08-18)
    #:   qwen-code   `--max-tool-calls 0`    zero tool budget, exit 55, no stdout
    #:   vibe        `--agent plan`          write_file/edit permission `never`
    #:
    #: THE SIXTH HAS NO ATOM, and that is why it went unnoticed longest:
    #:
    #:   gemini      (no atom — a DEFAULT)   ships no `write_file` tool at all
    #:
    #: Measured 2026-08-19 (harness-expansion.md §16.3): under this adapter's argv
    #: gemini-cli advertises nine tools and none of them writes. Handed a
    #: `write_file` call it answers `Tool "write_file" not found in registry`,
    #: writes nothing, and **exits 0 printing the model's own "JSON written to …"
    #: confirmation as its answer** — the one member that reports success for a
    #: write that never happened, where the other five fail loudly. An argv guard
    #: cannot see a default, so its adjudication lives in that same file's third
    #: table, `NO_WRITE_TOOL_AT_ALL`, which feeds the same derived-roster
    #: assertion.
    #:
    #: THE SEVENTH IS NEITHER A MODE NOR A MISSING TOOL (D6, 2026-09-11):
    #:
    #:   codex       `--sandbox workspace-write`  writes only through bubblewrap
    #:
    #: codex CAN write a file — wherever bubblewrap can start. On a stock Ubuntu
    #: 24.04 (`kernel.apparmor_restrict_unprivileged_userns=1`) it cannot, and a
    #: review asked for a file spent seven minutes twice, failed three writes
    #: and returned no JSON, with `--ephemeral` keeping nothing to salvage. The
    #: array was always the final `agent_message`, which `parse_output` reads,
    #: so the fan-out asks for stdout and never bets a slot on the sandbox. Its
    #: adjudication is the `STOPS_FILE_WRITES` row under that atom;
    #: `supports_file_edits` stays True (the agent-write profile really edits,
    #: and the sandbox probe that gates it is D6 step 3).
    CANNOT_WRITE_FILES = {"aider", "codex", "copilot", "cursor-agent", "gemini",
                          "qwen-code", "vibe"}

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_adapter_declares_a_bool(self, name):
        from utils.provider_registry import _PROVIDERS

        assert isinstance(_PROVIDERS[name].supports_file_output, bool)

    def test_the_read_only_roster_is_exactly_the_declared_set(self):
        from utils.provider_registry import _PROVIDERS

        declared = {
            name
            for name, provider in _PROVIDERS.items()
            if not provider.supports_file_output
        }
        assert declared == self.CANNOT_WRITE_FILES

    def test_cursor_declares_it_cannot_write_files(self):
        from utils.provider_registry import get_provider

        assert get_provider("cursor-agent").supports_file_output is False

    def test_cursors_declaration_matches_the_argv_that_causes_it(self):
        """The declaration is only true while the write refusal is in the argv.

        `--mode ask` is what refuses the write, and `-f`/`--yolo` is the bypass
        this adapter exists to stop passing (§3.1). Restoring either — "fixing"
        the file write by handing an untrusted prompt write access again — must
        fail here rather than quietly undo the safety decision.

        §5.4.1's per-flag gate is stubbed exactly as in
        `TestCursorAgentProvider.test_cursor_agent_build_command`, so this reads
        the argv rather than the developer's installed cursor build.
        """
        from utils.provider_registry import get_provider

        with patch.object(
            base_module,
            "read_help_tokens",
            lambda _b, **_kw: frozenset({"--trust", "--mode"}),
        ), patch.object(
            cursor_agent_module.shutil, "which", lambda _n: sys.executable
        ):
            cmd = get_provider("cursor-agent").build_command("prompt", "auto")

        assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "ask"
        assert "--trust" in cmd
        assert "-f" not in cmd and "--force" not in cmd and "--yolo" not in cmd

    def test_a_file_capable_adapter_is_still_the_default(self):
        from utils.provider_registry import get_provider

        assert get_provider("claude-code").supports_file_output is True
        assert get_provider("opencode").supports_file_output is True

    def test_codex_moves_the_review_off_its_sandbox_and_keeps_its_edits(self):
        """D6 step 1: the review channel no longer depends on bubblewrap, and
        nothing else about the adapter moves — the agent-write profile still
        edits, and the `--output-schema` sidecar is still a path."""
        from utils.provider_registry import get_provider

        codex = get_provider("codex")

        assert codex.supports_file_output is False
        assert codex.supports_file_edits is True
        assert codex.structured_output_transport == "file"

    @pytest.mark.parametrize("name", sorted(CANNOT_WRITE_FILES))
    def test_a_read_only_adapter_is_asked_for_stdout(self, name):
        """The declaration is what the prompt is rendered from — not a name
        spelled out in an orchestrator."""
        from utils.llm_client import STDOUT_OUTPUT_LOCATION, render_output_location

        block = render_output_location("/out/result.json", f"{name}:some-model")

        assert block == STDOUT_OUTPUT_LOCATION
        assert "/out/result.json" not in block


class TestDeclaredPlanRoute:
    """`wants_inline_plan`: does the ask lane inline the plan's body into this
    adapter's prompt (plans/ask-answer-recovery-and-false-success.md D4.2)?

    Measured 2026-09-26: aider already received the plan by `--read`, so the
    inlined copy only doubled its prompt and fed its mention scan, which
    auto-added every repo path the plan named — a 1.08M-token request.
    """

    #: Pinned like `CANNOT_WRITE_FILES`, and each member names the route that
    #: replaces the inline copy: an adapter declaring `False` without one would
    #: send its model no plan at all. `TestPathOnlyAskPromptHasARoute` in
    #: `test_provider_housekeeping.py` checks the route against the argv the
    #: adapter really renders — this map is the *what*, that test is the *why*.
    #:
    #:   aider   `--read <plan_path>`   every absolute path the prompt names is
    #:                                  attached read-only (`_extract_read_files`)
    PATH_ONLY_PLAN_ROUTES = {"aider": "--read"}

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_adapter_declares_a_bool(self, name):
        from utils.provider_registry import _PROVIDERS

        assert isinstance(_PROVIDERS[name].wants_inline_plan, bool)

    def test_the_path_only_roster_is_exactly_the_declared_set(self):
        from utils.provider_registry import _PROVIDERS

        declared = {
            name
            for name, provider in _PROVIDERS.items()
            if not provider.wants_inline_plan
        }
        assert declared == set(self.PATH_ONLY_PLAN_ROUTES)

    def test_inlining_is_still_the_default(self):
        from utils.provider_registry import get_provider

        assert base_module.LLMProvider.wants_inline_plan is True
        assert get_provider("claude-code").wants_inline_plan is True


class TestDeclaredGenerateShellAndReadScope:
    """`runs_shell_in_generate` and `generate_read_scope`: what a fan-out slot
    can do to reach the change (plan: review-run-scope-salvage-and-failure-
    visibility D2.3).

    Measured on the 2026-09-25 review run: the code-review prompt told every
    lane to run `git diff`, and cursor's `--mode ask` lanes have no shell — one
    returned `[]` without ever seeing a diff. `render_diff_access` now renders
    the prompt's `{diff_access}` block from these two declarations, so a wrong
    value tells a lane it has a shell it lacks, or withholds one it has. Each
    value is pinned to its source here; `tests/test_provider_housekeeping.py`'s
    `TestShellDeclarationMatchesTheEvidence` checks the shell half against the
    argv or the named measurement behind it.
    """

    #: D2.3's table. Every adapter not in it is `None` (not measured), and a
    #: value leaves `None` only with a measured source: a log, a probe or argv.
    #:
    #:   pi           True   the incident's `bash` call running pytest (§3.1)
    #:   omp          True   its argv: `--tools read,grep,glob,bash,edit,write`
    #:   codex        True   §7.3's audited stream ran `pwd && rg --files`
    #:   cursor-agent False  ask mode refused a shell in the kimi and grok logs
    #:   claude-code  False  `acceptEdits` denies headless Bash (3 denials)
    #:   qwen-code    False  `--max-tool-calls 0` aborts before any tool runs
    #:   gemini       False  no `run_shell_command` under this argv (0.25.2)
    SHELL_IN_GENERATE = {
        "pi": True,
        "omp": True,
        "codex": True,
        "cursor-agent": False,
        "claude-code": False,
        "qwen-code": False,
        "gemini": False,
    }
    #: The adapters whose read scope is measured away from the `"cwd"` default.
    #: cursor-agent declares `"cwd"` explicitly (measured, and equal to the
    #: default), so it is not a departure. aider's is read from its argv:
    #: `/ask` has no file tool, and the stdin decline supply refuses the
    #: mention adds that once brought review-code the changed files.
    READ_SCOPE_DEPARTURES = {"qwen-code": "none", "aider": "none"}

    def test_the_base_defaults_assert_nothing(self):
        assert LLMProvider.runs_shell_in_generate is None
        assert LLMProvider.generate_read_scope == "cwd"

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_adapter_declares_true_false_or_none(self, name):
        from utils.provider_registry import _PROVIDERS

        value = _PROVIDERS[name].runs_shell_in_generate
        # `is` and `isinstance(bool)`: a truthy 1 or "yes" is not a declaration.
        assert value is None or isinstance(value, bool), (name, value)

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_read_scope_is_one_of_the_three(self, name):
        from utils.llm_client import GENERATE_READ_SCOPES
        from utils.provider_registry import _PROVIDERS

        assert GENERATE_READ_SCOPES == ("none", "cwd", "any")
        assert _PROVIDERS[name].generate_read_scope in GENERATE_READ_SCOPES, name

    def test_the_shell_values_are_the_measured_table(self):
        from utils.provider_registry import _PROVIDERS

        declared = {
            name: provider.runs_shell_in_generate
            for name, provider in _PROVIDERS.items()
            if provider.runs_shell_in_generate is not None
        }
        assert declared == self.SHELL_IN_GENERATE

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_adapter_outside_the_table_is_unmeasured(self, name):
        from utils.provider_registry import _PROVIDERS

        if name in self.SHELL_IN_GENERATE:
            assert _PROVIDERS[name].runs_shell_in_generate is self.SHELL_IN_GENERATE[name]
        else:
            assert _PROVIDERS[name].runs_shell_in_generate is None, (
                f"{name} declares runs_shell_in_generate="
                f"{_PROVIDERS[name].runs_shell_in_generate!r} with no row in "
                f"D2.3's table — a value leaves None only with a measured source"
            )

    def test_cursor_reads_under_cwd_and_qwen_reads_nothing(self):
        from utils.provider_registry import get_provider

        assert get_provider("cursor-agent").generate_read_scope == "cwd"
        assert get_provider("qwen-code").generate_read_scope == "none"

    def test_every_other_adapter_keeps_the_cwd_default(self):
        from utils.provider_registry import _PROVIDERS

        departures = {
            name: provider.generate_read_scope
            for name, provider in _PROVIDERS.items()
            if provider.generate_read_scope != "cwd"
        }
        assert departures == self.READ_SCOPE_DEPARTURES

    def test_the_two_declarations_are_not_the_agent_readonly_ones(self):
        """codex reads anywhere under `agent-readonly` (`reads_outside_cwd`),
        but nothing measured that for `generate`, so its fan-out slot keeps the
        `"cwd"` default: one profile's measurement never stands in for
        another's."""
        from utils.provider_registry import get_provider

        codex = get_provider("codex")
        assert codex.reads_outside_cwd is True
        assert codex.generate_read_scope == "cwd"


#: `TestClassifyFailure`'s case tables, lifted to module scope so
#: `TestMatchStreams` twins every row through `match_streams` too: a case added
#: here is graded by both tests without anyone copying it.
CLASSIFY_FAILURE_BARE_429_NOT_RATE_LIMIT = [
    '{"type":"step_finish","cost":0.0429,"tokens":{"output":14290}}',
    "total cost 1.0429 for 4290 tokens",
    "429 things were reviewed",
]
CLASSIFY_FAILURE_REAL_429 = [
    "HTTP 429 Too Many Requests",
    '{"error":{"status":429,"message":"slow down"}}',
    "Error: 429",
    "Requests are throttled: status code 429",
]
CLASSIFY_FAILURE_TEXTUAL_RATE = [
    "rate limit exceeded",
    "the provider reported rate limits",
    "API rate-limited, retry later",
    "quota exceeded for this model",
    '{"type":"overloaded_error"}',
]
CLASSIFY_FAILURE_PROSE_AUTH_WORD = [
    "the unauthorized branch in review.py returns early",
    "T004 adds a test for unauthorized callers",
]
CLASSIFY_FAILURE_REAL_LOGIN_NOTICE = [
    "Not logged in. Please log in with `codex login`.",
    "401 Unauthorized",
    "Error: unauthorized",
    "authentication required",
    "invalid api key",
    "invalid_api_key",
    "no credentials found for provider",
]
#: `(stdout, stderr)` pairs `TestClassifyFailure` pins outside a parametrize.
CLASSIFY_FAILURE_PAIRS = [
    ("", ""),
    ("   \n ", ""),
    ("429 too many requests\nnot logged in", ""),
]
CLASSIFY_FAILURE_STREAMS = (
    CLASSIFY_FAILURE_BARE_429_NOT_RATE_LIMIT
    + CLASSIFY_FAILURE_REAL_429
    + CLASSIFY_FAILURE_TEXTUAL_RATE
    + CLASSIFY_FAILURE_PROSE_AUTH_WORD
    + CLASSIFY_FAILURE_REAL_LOGIN_NOTICE
)


class TestClassifyFailure:
    """D3i/D3k: the classifier must read a stream, not find digits in one."""

    @pytest.fixture
    def provider(self):
        from utils.provider_registry import get_provider

        return get_provider("opencode")

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_BARE_429_NOT_RATE_LIMIT)
    def test_a_bare_429_inside_a_number_is_not_a_rate_limit(self, provider, stream):
        """A misread here is retried WITHOUT consuming the retry budget."""
        assert provider.classify_failure(stream, "") is None

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_REAL_429)
    def test_a_real_429_is_still_a_rate_limit(self, provider, stream):
        assert provider.classify_failure("", stream) == "rate_limited"

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_TEXTUAL_RATE)
    def test_the_textual_rate_patterns_keep_matching(self, provider, stream):
        assert provider.classify_failure("", stream) == "rate_limited"

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_PROSE_AUTH_WORD)
    def test_prose_quoting_an_auth_word_is_not_an_auth_failure(self, provider, stream):
        """auth_failed is batch-fatal, so a quoted word must never trigger it."""
        assert provider.classify_failure(stream, "") is None

    @pytest.mark.parametrize("stream", CLASSIFY_FAILURE_REAL_LOGIN_NOTICE)
    def test_a_real_login_notice_is_still_auth_failed(self, provider, stream):
        assert provider.classify_failure("", stream) == "auth_failed"

    def test_an_empty_stream_classifies_nothing(self, provider):
        for stdout, stderr in CLASSIFY_FAILURE_PAIRS[:2]:
            assert provider.classify_failure(stdout, stderr) is None

    def test_auth_wins_over_a_rate_limit_in_the_same_stream(self, provider):
        stdout, stderr = CLASSIFY_FAILURE_PAIRS[2]
        assert provider.classify_failure(stdout, stderr) == "auth_failed"


#: The adapters that override ``describe_failure``, each with **the channel its
#: CLI really writes its diagnosis on**. It is a two-column table rather than a
#: set because the channel is the thing being pinned: the value is what
#: ``test_each_override_reads_only_the_channel_it_declares`` measures against
#: that adapter's committed captures, so a row cannot be added without saying
#: which stream it reads and being held to it.
#:
#: ``opencode`` and ``muse`` report their errors on **stdout** and leave stderr
#: uninformative — the shape the hook was written for. ``pi`` and ``gemini``
#: joined on 2026-08-20 (C4) as the exact inverse: both took ``stderr`` in
#: ``parse_output`` and never mentioned it again, so 100% of the diagnosis was
#: discarded while the CLI exited non-zero with the vendor's full error object
#: sitting unused. The hook is therefore a *channel* rule, not a stream name.
#:
#: ``kilocode`` joined on 2026-09-10 on the stderr side, from a live failure
#: rather than a probe: a `review-code` slot on a dead account whose stderr read
#: ``Error: User not found.`` in ANSI colour and whose operator-visible line read
#: ``kilocode exited with code 1``. It ships **no ``.txt`` capture pairs**, so
#: ``test_each_override_reads_only_the_channel_it_declares`` skips it here and
#: ``tests/test_stderr_diagnosis.py`` makes the same assertion over the literals
#: that file commits — the declared channel is graded either way, and the row is
#: what stops the override landing without declaring one.
#: ``opencode`` is the third shape (D5): ``stdout`` is still the declared
#: channel — blanking it yields ``None`` — but since 1.18.x the stdout event can
#: be a wrapper (``UnknownError: Unexpected server error``) whose cause is a
#: ``level=ERROR … error="…"`` line on stderr, and ``describe_failure`` appends
#: that value in parentheses. So stderr is declared *enrichment* that may change
#: the text: blanking it must leave a non-``None`` answer that is a **prefix** of
#: the composed one. Encoded as a value rather than an exemption, so the row
#: stays graded against the rig's ``invalid_model`` capture.
#:
#: ``claude-code`` and ``cursor-agent`` joined on 2026-09-17, from one live
#: ``review-code`` fan-out whose two failed slots were both recorded as the bare
#: ``<provider> exited with code 1``. ``claude-code`` is plain **stdout**: the
#: envelope carries ``api_error_status`` and the vendor sentence, stderr is 0
#: bytes on all eleven committed captures, and the override renders
#: ``result_envelope.describe_refused_envelope``'s line (a 491 s slot had died
#: on a 429 usage cap with $25.05 spent, and nothing on screen said so).
#: ``cursor-agent`` is the **fourth** shape: a run that reached the backend
#: answers with the same envelope on stdout, so stdout is asked first — but a
#: run that never got one leaves its reason as stderr's **last** line
#: (``RetriableError: [resource_exhausted] Error`` under three reconnect
#: lines, stdout empty), and that channel answers *only when stdout has
#: nothing to say*. The graded rule is therefore precedence, not enrichment:
#: with stdout answering, blanking stderr changes nothing; with stdout blank,
#: the answer is stderr's own. Its committed captures are all stdout-shaped
#: (stderr 0 bytes on both), and ``tests/test_failed_slot_reporting.py``
#: grades the stderr leg over the measured kimi lines.
STDOUT_THEN_STDERR = "stdout-then-stderr"
STDOUT_ELSE_STDERR = "stdout-else-stderr"

_DESCRIBE_FAILURE_CHANNELS = {
    "opencode": STDOUT_THEN_STDERR,
    "muse": "stdout",
    # pi moved from "stderr" with `--mode json` (plan §7, T7.6), to omp's shape
    # and for omp's reason, measured at 0.87.1 (tests/fixtures/pi/manifest.yaml):
    # every in-stream failure is an assistant `message_end` on stdout carrying
    # the vendor's `errorMessage` with ZERO bytes on stderr and exit 0, while a
    # thrown pre-flight (`preflight-no-key`) is a one-line `session` header on
    # stdout — silent — with the diagnosis on stderr and exit 1. Precedence, not
    # enrichment: stderr may never edit an answer the stream already gave.
    "pi": STDOUT_ELSE_STDERR,
    "gemini": "stderr",
    "kilocode": "stderr",
    "claude-code": "stdout",
    "cursor-agent": STDOUT_ELSE_STDERR,
    # omp is cursor-agent's shape, and the asymmetry that makes it precedence
    # rather than enrichment is measured: in `--mode json` an API failure is a
    # JSONL stream on stdout with ZERO bytes on stderr, while a pre-flight fault
    # (bad model, missing key, rejected --tools name, missing overlay) is zero
    # bytes on stdout with a plain-text diagnosis on stderr. STDOUT_THEN_STDERR
    # would fail those pre-flight rows by construction — it requires the
    # declared stdout channel to answer on its own — and composing the two would
    # let stderr edit an answer the stream had already given.
    "omp": STDOUT_ELSE_STDERR,
    # codex names one failure only — the exit-101 panic a globally enabled
    # `use_legacy_landlock` causes on every workspace-write dispatch (landlock
    # rung, constraint 2) — and a Rust panic is printed on stderr.
    "codex": "stderr",
}


def _describe_failure_overrides():
    """The roster read off the CLASSES, never off the table above.

    Parametrizing over the registry rather than over a literal roster is what
    makes a new override fail by name on the day it lands instead of whenever
    somebody remembers to widen a set.
    """
    from utils.provider_registry import _PROVIDERS

    return {
        name
        for name, provider in _PROVIDERS.items()
        if type(provider).describe_failure is not LLMProvider.describe_failure
    }


def _provider_capture_rows(name):
    """``(row, stdout, stderr)`` for every committed ``.txt`` capture pair."""
    directory = Path(__file__).parent / "fixtures" / name
    if not directory.is_dir():
        return []
    rows = []
    for row in sorted({path.name.split(".")[0] for path in directory.glob("*.txt")}):
        streams = []
        for key in ("stdout", "stderr"):
            path = directory / f"{row}.{key}.txt"
            streams.append(path.read_text(encoding="utf-8") if path.is_file() else "")
        rows.append((row, streams[0], streams[1]))
    return rows


class TestDescribeFailureChannels:
    """The opt-in hook that recovers a failure reason from **either** channel.

    Renamed from ``TestDescribeFailure`` on 2026-08-20, with
    ``test_stdout_error_providers_are_the_only_overrides`` renamed with it: both
    names encoded the base docstring's old framing, that this is the hook for *"a
    provider that reports its errors on stdout rather than stderr"*. C4 made that
    false in the most direct way available — by adding the two adapters that are
    its mirror image — so what is pinned here is the channel each override
    declares, measured, rather than one stream's name.

    D5 added the third shape the table has to name: ``stdout-then-stderr``,
    where the declared channel is still one stream (blanking it is ``None``)
    and the other is enrichment that may *extend* the answer but never replace
    it — blanking the enrichment leaves a prefix of the composed text.
    """

    @pytest.mark.parametrize("name", _envelope_sweep_names())
    def test_an_adapter_without_an_override_inherits_none(self, name):
        """No editorial from an adapter that has none to add.

        The base returns ``None`` and most adapters inherit it, so the caller
        sees the exit code plus ``details["stderr"]`` and nothing invented.
        Parametrized over the whole registry, so a *new* adapter is covered on
        the day it is registered.
        """
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        if name in _describe_failure_overrides():
            pytest.skip(f"{name} overrides describe_failure; see the channel test")

        assert provider.describe_failure("some stdout", "some stderr") is None

    def test_the_override_roster_and_the_channel_table_agree(self):
        """Neither half may move without the other.

        The roster is derived from the classes; the table is hand-written and
        carries the channel. Equality between them is what stops an override
        landing with no declared channel — and stops a stale row outliving the
        override it describes.
        """
        assert _describe_failure_overrides() == set(_DESCRIBE_FAILURE_CHANNELS)

    @pytest.mark.parametrize("name", sorted(_DESCRIBE_FAILURE_CHANNELS))
    def test_each_override_reads_only_the_channel_it_declares(self, name):
        """The rule, measured on committed bytes rather than asserted in prose.

        For every capture this adapter ships: blanking the **other** channel
        must not change the answer, and blanking the **declared** channel must
        make it ``None``. That is "stderr is the diagnosis, stdout is the
        answer" — or its mirror — expressed as an assertion, and it is what
        stops a future edit quietly reading the answer channel and reporting a
        model's own words as the failure reason. gemini's ``exit-zero-failure``
        is the row where it bites: 1,174 bytes of (false) answer on stdout and
        193 bytes of real reason on stderr, in one capture.
        """
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        declared = _DESCRIBE_FAILURE_CHANNELS[name]
        rows = _provider_capture_rows(name)
        if not rows:
            pytest.skip(f"{name} ships no .stdout.txt/.stderr.txt capture pairs")

        for row, stdout, stderr in rows:
            both = provider.describe_failure(stdout, stderr)
            if declared == STDOUT_THEN_STDERR:
                # The declared channel alone still answers, and the composed
                # answer only ever extends it; the enrichment alone is nothing.
                stdout_only = provider.describe_failure(stdout, "")
                assert both is not None and stdout_only is not None, (
                    f"{name}/{row}: the declared stdout channel recovered nothing"
                )
                assert both.startswith(stdout_only), (
                    f"{name}/{row}: stderr enrichment replaced the stdout answer "
                    f"instead of extending it"
                )
                assert provider.describe_failure("", stderr) is None, (
                    f"{name}/{row}: a reason was recovered from stderr alone, "
                    f"which is enrichment and not the declared channel"
                )
                continue
            if declared == STDOUT_ELSE_STDERR:
                # Precedence: an answer from stdout is the answer, and stderr
                # may not touch it; with stdout silent, stderr speaks alone.
                stdout_only = provider.describe_failure(stdout, "")
                if stdout_only is not None:
                    assert both == stdout_only, (
                        f"{name}/{row}: stderr changed an answer stdout had given"
                    )
                else:
                    assert both == provider.describe_failure("", stderr), (
                        f"{name}/{row}: with stdout silent the answer is not stderr's own"
                    )
                continue
            if declared == "stdout":
                declared_only, other_only = (stdout, ""), ("", stderr)
            else:
                declared_only, other_only = ("", stderr), (stdout, "")

            assert provider.describe_failure(*declared_only) == both, (
                f"{name}/{row}: the answer changed when the {declared} capture "
                f"was presented on its own — the other channel is being read"
            )
            assert provider.describe_failure(*other_only) is None, (
                f"{name}/{row}: a reason was recovered from the channel "
                f"{name} does NOT declare"
            )

    @pytest.mark.parametrize("name", sorted(_DESCRIBE_FAILURE_CHANNELS))
    def test_every_override_survives_garbage_on_both_channels(self, name):
        """Diagnostics must never raise.

        Both callers wrap this hook in ``try``/``except`` precisely because an
        override that threw would mask the failure it exists to explain — so the
        contract is that it returns a string or ``None`` for any bytes at all.
        """
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]

        for stdout, stderr in (("", ""), ("{", "}"), ("\x00", "\x00"),
                               ("[]", "ApiError: {not json"),
                               ("x" * 40000, "9" * 40000)):
            reason = provider.describe_failure(stdout, stderr)

            assert reason is None or isinstance(reason, str)


class TestAgyProvider:
    """Tests for the AgyProvider class."""

    @pytest.fixture
    def provider(self):
        """Create an AgyProvider instance."""
        return AgyProvider()

    @staticmethod
    def _envelope(response, status="SUCCESS", error=None):
        """Build representative agy --output-format json stdout."""
        envelope = {
            "conversation_id": "conv-123" if status == "SUCCESS" else "",
            "status": status,
            "response": response,
            "duration_seconds": 4.2,
            "num_turns": 1,
            "usage": {"input_tokens": 50, "output_tokens": 50},
        }
        if error is not None:
            envelope["error"] = error
        return json.dumps(envelope)

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "agy"

    def test_default_timeout(self, provider):
        """The fallback twin of `providers.yaml`'s 1200 (§9.6.d).

        It read 600 while the yaml read 1200 and the argv read `20m`: three
        numbers for one fact. Raising it is a no-op on every real path — agy
        ships in base config, so `get_provider_timeout` returns the yaml value
        and never reaches the adapter default — which is what made it safe to
        reconcile here rather than schedule separately.
        """
        assert provider.default_timeout == 1200

    def test_agy_parse_success_array(self, provider):
        """Extract a JSON array from the response field."""
        inner_data = [
            {"title": "Finding 1", "desc": "Description", "importance": "medium"}
        ]
        stdout = self._envelope(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_agy_parse_success_object(self, provider):
        """Extract a JSON object from the response field."""
        inner_data = {"first_heading": "Probe Widget"}
        stdout = self._envelope(json.dumps(inner_data))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_agy_parse_code_block_json(self, provider):
        """Extract JSON from a ```json code block in the response."""
        inner_json = [{"task": "T001", "status": "done"}]
        response = f"Analysis complete:\n\n```json\n{json.dumps(inner_json)}\n```\n"
        stdout = self._envelope(response)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_agy_parse_prose_with_json(self, provider):
        """Extract embedded JSON when prose surrounds it in the response."""
        inner_json = {"result": "success"}
        stdout = self._envelope(f"Here is the JSON you asked for: {json.dumps(inner_json)}")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_agy_parse_trailing_newline_response(self, provider):
        """The response field carries a trailing newline; it is stripped."""
        stdout = self._envelope("[1, 2, 3]\n")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_agy_parse_error_envelope(self, provider):
        """ERROR status (exit 1) -> failure carrying the envelope's error."""
        error_message = "You must sign in to use the Antigravity CLI."
        stdout = self._envelope("", status="ERROR", error=error_message)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["error"] == error_message
        assert result["data"] is None

    def test_agy_parse_non_success_status_without_error(self, provider):
        """A non-SUCCESS status with no error field gets a fallback message."""
        stdout = self._envelope("", status="TIMEOUT")

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "TIMEOUT" in result["error"]
        assert result["data"] is None

    def test_agy_parse_empty_response(self, provider):
        """SUCCESS with an empty response field -> failure."""
        stdout = self._envelope("")

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_agy_parse_envelope_missing_status(self, provider):
        """A JSON dict without a status field is returned as the data itself."""
        stdout = json.dumps({"result": "success", "items": [1, 2]})

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == {"result": "success", "items": [1, 2]}

    def test_agy_parse_non_json_stdout_fallback(self, provider):
        """Fallback extraction when stdout is not a JSON envelope but embeds JSON."""
        stdout = "Some banner text [1, 2, 3] trailing text"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_agy_parse_garbage_stdout(self, provider):
        """Handle stdout with no envelope and no extractable JSON."""
        stdout = "not valid json at all"

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result
        assert "raw" in result

    def test_agy_build_command(self, provider):
        """Exact argv: flags order, display-name model verbatim, -p prompt."""
        cmd = provider.build_command("Analyze this", "Gemini 3.1 Pro (High)")

        assert cmd == [
            "agy",
            "--new-project",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--print-timeout",
            "20m",
            "--model",
            "Gemini 3.1 Pro (High)",
            "-p",
            "Analyze this",
        ]

    # --- L11 #3: agy has a real --effort flag (§13.4 #3) --------------------
    #
    # `agy --help` (agy 1.1.13, offline, no auth):
    #   --effort   Reasoning effort for the current CLI session (low|medium|high)
    # The adapter and providers.yaml both used to state that agy has no
    # `:effort` suffix support, so every effort a user configured was refused by
    # *our* whitelist for a flag agy really has.

    def test_agy_reasoning_efforts_constant(self):
        """REASONING_EFFORTS is exactly what `agy --help` lists."""
        from utils.providers.agy import REASONING_EFFORTS

        assert set(REASONING_EFFORTS) == {"low", "medium", "high"}

    def test_agy_build_command_effort_suffix_exact(self, provider):
        """A `:effort` suffix becomes `--effort <level>`, model name intact.

        agy display names carry no colon of their own, so the last-colon split
        cannot damage one — `"Gemini 3.1 Pro (High)"` survives verbatim as a
        single argv element while `high` moves onto the flag.
        """
        cmd = provider.build_command("Analyze this", "Gemini 3.1 Pro (High):high")

        assert cmd == [
            "agy",
            "--new-project",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--print-timeout",
            "20m",
            "--model",
            "Gemini 3.1 Pro (High)",
            "--effort",
            "high",
            "-p",
            "Analyze this",
        ]

    def test_agy_build_command_all_reasoning_efforts(self, provider):
        """Each accepted level renders the flag; none of them touches the model."""
        for effort in ("low", "medium", "high"):
            cmd = provider.build_command("p", f"Gemini 3.1 Pro (High):{effort}")

            assert cmd[-4:] == ["--effort", effort, "-p", "p"], (
                f"effort {effort!r} did not build the expected command: {cmd}")
            assert "Gemini 3.1 Pro (High)" in cmd

    def test_agy_build_command_effort_passthrough(self, provider):
        """An unknown suffix is not an effort: it stays part of the model name.

        agy rejects an unknown model outright with an ERROR envelope, so
        passing it through is the loud failure — inventing an `--effort xhigh`
        agy does not accept would be the quiet one.
        """
        cmd = provider.build_command("p", "Gemini 3.1 Pro (High):xhigh")

        assert "--effort" not in cmd
        assert cmd[cmd.index("--model") + 1] == "Gemini 3.1 Pro (High):xhigh"

    @patch("shutil.which")
    def test_agy_is_available_true(self, mock_which, provider):
        """Test is_available returns True when agy is found."""
        mock_which.return_value = "/usr/bin/agy"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("agy")

    @patch("shutil.which")
    def test_agy_is_available_false(self, mock_which, provider):
        """Test is_available returns False when agy is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("agy")


class TestMuseProvider:
    """Tests for the MuseProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a MuseProvider instance."""
        return MuseProvider()

    @staticmethod
    def _record(payload_type, payload, seq=1):
        """Wrap a payload in muse's JSONL envelope."""
        return json.dumps({
            "schema_version": 1,
            "id": f"018f0000-0000-7000-8000-{seq:012d}",
            "stream": {"kind": "session", "id": "sess-1"},
            "sequence": seq,
            "recorded_at": 1780531400000000 + seq,
            "record_type": "event",
            "durability": "durable",
            "causation_id": "cmd-1",
            "payload_type": payload_type,
            "payload_schema_version": 1,
            "payload": payload,
        })

    @classmethod
    def _stream(cls, *records):
        """Join envelope records into a JSONL stdout blob."""
        return "\n".join(
            cls._record(payload_type, payload, seq)
            for seq, (payload_type, payload) in enumerate(records, start=1)
        )

    @classmethod
    def _delta(cls, text):
        return ("run.output.delta", {"kind": "run_output_delta", "text": text})

    @classmethod
    def _tool_result(cls, text="Read text file `note.txt`."):
        return ("tool.result", {"kind": "tool_result", "call_id": "call_1", "text": text})

    @classmethod
    def _completed(cls, text):
        return ("run.terminal.completed", {
            "kind": "run_terminal", "terminal": "completed", "reason": None, "text": text,
        })

    @classmethod
    def _failed(cls, reason):
        return ("run.terminal.failed", {
            "kind": "run_terminal", "terminal": "failed", "reason": reason, "text": "",
        })

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "muse"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_muse_parse_terminal_array(self, provider):
        """Extract a JSON array from the terminal record's text."""
        inner_data = [{"title": "Finding 1", "desc": "Description", "importance": "medium"}]
        stdout = self._stream(self._delta(json.dumps(inner_data)),
                              self._completed(json.dumps(inner_data)))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_muse_parse_terminal_object(self, provider):
        """Extract a JSON object from the terminal record's text."""
        inner_data = {"first_heading": "Probe Widget"}
        stdout = self._stream(self._completed(json.dumps(inner_data)))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_muse_parse_code_block_json(self, provider):
        """Extract JSON wrapped in a markdown code block."""
        inner_data = [{"title": "Fenced", "desc": "d", "importance": "low"}]
        text = f"Here you go:\n```json\n{json.dumps(inner_data)}\n```"
        stdout = self._stream(self._completed(text))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_muse_parse_prose_with_json(self, provider):
        """Extract JSON embedded in surrounding prose."""
        inner_data = [{"title": "Inline", "desc": "d", "importance": "high"}]
        stdout = self._stream(self._completed(f"My findings: {json.dumps(inner_data)} Done."))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_data

    def test_muse_parse_closing_message_wins_over_draft(self, provider):
        """The terminal text glues every assistant message together with no
        separator, so an early draft precedes the answer. The deltas after the
        last tool.result isolate the closing message."""
        draft = [{"title": "draft", "desc": "early", "importance": "low"}]
        answer = [{"title": "final", "desc": "answer", "importance": "high"}]
        stdout = self._stream(
            self._delta(f"Draft: {json.dumps(draft)}"),
            self._tool_result(),
            self._delta(json.dumps(answer)),
            self._completed(f"Draft: {json.dumps(draft)}{json.dumps(answer)}"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_split_deltas_reassemble(self, provider):
        """Deltas are token fragments, not messages: a single document arrives
        split mid-token and must be rejoined without a separator."""
        answer = [{"title": "split", "desc": "d", "importance": "low"}]
        blob = json.dumps(answer)
        third = len(blob) // 3
        stdout = self._stream(
            self._tool_result(),
            self._delta(blob[:third]),
            self._delta(blob[third:2 * third]),
            self._delta(blob[2 * third:]),
            self._completed(blob),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_prose_before_answer_same_turn(self, provider):
        """Narration and answer in the closing turn still extract the answer."""
        answer = [{"title": "final", "desc": "answer", "importance": "high"}]
        text = f"Looks good so far.{json.dumps(answer)}"
        stdout = self._stream(self._delta(text), self._completed(text))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_falls_back_to_deltas_without_terminal(self, provider):
        """A truncated stream with no terminal record still yields the deltas."""
        answer = [{"title": "partial", "desc": "d", "importance": "low"}]
        stdout = self._stream(self._delta(json.dumps(answer)))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_terminal_failed(self, provider):
        """A failed terminal record surfaces its reason rather than "no output"."""
        reason = ("API error 402 [request_id=abc]: Billing verification failed. "
                  "Please check your payment method. (billing_error) "
                  "(after 10 provider attempts)")
        stdout = self._stream(self._failed(reason))

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "failed" in result["error"]
        assert "Billing verification failed" in result["error"]

    def test_muse_parse_failed_with_unparseable_partial_text(self, provider):
        """A cancelled/failed run that emitted prose keeps its terminal reason."""
        stdout = self._stream(
            self._delta("I was part way through when"),
            ("run.terminal.cancelled", {
                "kind": "run_terminal", "terminal": "cancelled",
                "reason": "interrupted by signal", "text": "I was part way through when",
            }),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "interrupted by signal" in result["error"]

    def test_muse_parse_no_events(self, provider):
        """Empty output reports a parse failure."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert result["data"] is None

    def test_muse_parse_skips_malformed_lines(self, provider):
        """Malformed JSONL lines are skipped, not fatal."""
        answer = [{"title": "ok", "desc": "d", "importance": "low"}]
        stdout = "not json\n" + self._stream(self._completed(json.dumps(answer))) + "\n{broken"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == answer

    def test_muse_parse_garbage_text(self, provider):
        """Terminal text with no JSON reports a parse failure."""
        stdout = self._stream(self._completed("I could not complete the review."))

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["data"] is None

    def test_muse_describe_failure_reads_terminal_reason(self, provider):
        """muse leaves only "run ended with Failed" on stderr; the cause is on stdout."""
        reason = "API error 402 [request_id=abc]: Billing verification failed."
        stdout = self._stream(self._failed(reason))

        assert provider.describe_failure(stdout, "run ended with Failed\n") == reason

    def test_muse_describe_failure_falls_back_to_task_failures(self, provider):
        """Without a terminal record, task-level failures carry the diagnosis."""
        stdout = self._stream(
            ("task.lifecycle.failed", {
                "kind": "task_lifecycle",
                "event": {"kind": "failed", "reason": "model failed: API error 402"},
            }),
        )

        assert provider.describe_failure(stdout, "") == "model failed: API error 402"

    def test_muse_describe_failure_returns_none_when_silent(self, provider):
        """Nothing to add beyond the exit code returns None."""
        assert provider.describe_failure("", "") is None

    def test_muse_build_command(self, provider):
        """Build the headless command without an effort suffix."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2")

        assert cmd == [
            "muse", "exec", "--json", "--disable-approval", "--disable-sandbox",
            "--model", "muse-spark-1.2",
            "Analyze this",
        ]

    def test_muse_build_command_disables_approval(self, provider):
        """--disable-approval is mandatory: an approval prompt hangs headless runs."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2")

        assert "--disable-approval" in cmd

    def test_muse_build_command_disables_sandbox(self, provider):
        """D6: muse's shell sandbox is bubblewrap, which a stock Ubuntu 24.04
        cannot start, so every sandboxed write died on `bwrap: loopback`.
        `--disable-sandbox` turns it off; `--yolo` would also grant workspace
        trust, which stays the user's own store."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2")

        assert "--disable-sandbox" in cmd
        assert cmd.index("--disable-sandbox") == cmd.index("--disable-approval") + 1
        assert "--yolo" not in cmd
        assert "--trust-workspace" not in cmd

    def test_muse_build_command_effort_suffix(self, provider):
        """A whitelisted :effort suffix becomes --reasoning-effort."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2:xhigh")

        assert cmd == [
            "muse", "exec", "--json", "--disable-approval", "--disable-sandbox",
            "--model", "muse-spark-1.2",
            "--reasoning-effort", "xhigh",
            "Analyze this",
        ]

    def test_muse_build_command_unknown_suffix_passthrough(self, provider):
        """An unrecognised suffix stays part of the model name (muse hard-errors
        on an invalid --reasoning-effort, so it must never be forwarded)."""
        cmd = provider.build_command("Analyze this", "muse-spark-1.2:turbo")

        assert cmd == [
            "muse", "exec", "--json", "--disable-approval", "--disable-sandbox",
            "--model", "muse-spark-1.2:turbo",
            "Analyze this",
        ]
        assert "--reasoning-effort" not in cmd

    def test_muse_reasoning_efforts_constant(self):
        """The whitelist matches the values muse's CLI accepts."""
        from utils.providers.muse import REASONING_EFFORTS

        assert REASONING_EFFORTS == {
            "none", "minimal", "low", "medium", "high", "xhigh", "ultra"
        }

    @patch("shutil.which")
    def test_muse_is_available_true(self, mock_which, provider):
        """Test is_available returns True when muse is found."""
        mock_which.return_value = "/usr/bin/muse"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("muse")

    @patch("shutil.which")
    def test_muse_is_available_false(self, mock_which, provider):
        """Test is_available returns False when muse is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("muse")


class TestOmpProvider:
    """oh-my-pi (`omp`), the eighteenth adapter and the first with a stdin default.

    What this class grades is everything about omp that is *not* a captured
    stream: identity, the setup metadata, the rendered argv, the flag gate that
    refuses to render one, the bypass the guard has to be able to see, the
    file-output declaration and the env delta. The parsing rows — every
    `stopReason`, the denial, the usage sums, `vendor_text`'s reduction — are
    graded from committed bytes under `tests/fixtures/omp/` by the §6.2 manifest
    rig, because a parser graded against strings written beside it is graded
    against its author's memory of the stream.
    """

    MODEL = "openrouter/openai/gpt-5.6-luna"
    PROMPT = "Analyze this plan carefully"

    @pytest.fixture
    def provider(self):
        return OmpProvider()

    @pytest.fixture
    def ungated(self):
        """Render argv from the adapter, never from the developer's PATH.

        omp's §2.11 gate reads the installed binary's `--help` before
        `build_command` will emit anything, which would otherwise make every
        assertion below depend on whether this machine happens to have omp. The
        shared gate resolves its binary through `LLMProvider.gate_binary`, so
        neutralizing that one lookup is enough: with no binary to read,
        `missing_flags` reports nothing missing and the argv is the one the
        adapter would render. The gate itself is graded separately, with the
        help text stubbed rather than the lookup.
        """
        with patch.object(LLMProvider, "gate_binary", lambda self, **_kw: None):
            yield

    # ── identity ─────────────────────────────────────────────────────────
    def test_identity(self, provider):
        assert provider.name == "omp"
        assert provider.default_timeout == 600
        assert provider.version_cmd == ("omp", "--version")
        assert provider.verified_version == "18.8.3"
        assert provider.usage_support == base_module.USAGE_FULL

    def test_availability_is_the_binarys_own_name_on_path(self, provider):
        """`omp` is both the provider key and the binary, so there is no
        `BINARY_TO_CLI` row to keep in step with this."""
        with patch("shutil.which", return_value="/usr/local/bin/omp") as which:
            assert provider.is_available() is True
        assert which.call_args[0][0] == "omp"

        with patch("shutil.which", return_value=None):
            assert provider.is_available() is False

    def test_no_version_floor_is_declared_and_none_is_owed(self, provider):
        """A measured-at version is an upper bound on the floor, never the floor.

        The upstream release notes mention every flag this argv needs without
        dating any of their introductions, so a `min_version` here would be a
        number nobody measured. What the adapter needs is not a version but the
        flags, and `OMP_REQUIRED_FLAGS` refuses by the NAME of the one a build
        lacks. `flags_verified_at_floor` stays empty for the same reason: an
        exemption from the gate requires a declared floor to have been verified
        at, and there is none.
        """
        assert provider.min_version == ""
        assert provider.max_version_exclusive == ""
        assert provider.flags_verified_at_floor == ()

    def test_the_declared_transports(self, provider):
        assert provider.supported_prompt_transports == ("argv", "stdin")
        assert provider.prompt_transport == "stdin"
        assert provider.prompt_argv_index == -1
        assert provider.structured_output_support is False

    # ── setup metadata (§2.1) ────────────────────────────────────────────
    def test_no_auth_probe_is_declared_and_none_is_owed(self, provider):
        """omp has no auth-status subcommand, measured on `omp --help` at 18.2.2.

        The three credential-adjacent commands it does have are each
        disqualified for the probe's job: `omp token <provider>` PRINTS the
        credential, so a probe whose success output is the secret would put the
        key into the captured streams a doctor row records; `omp usage` is a
        live usage-limit snapshot over the network for OAuth accounts (a network
        fact, not a credential read); and `omp auth-broker status` reports the
        credential-vault daemon, not whether an OpenRouter key resolves. A fix
        is owed only where a probe is declared, so declaring none is what makes
        `login_cmd` empty rather than missing.
        """
        assert provider.auth_check == ()
        assert provider.login_cmd == ""
        assert provider.setup_probe == ()

        hints = provider.describe_setup()
        assert hints.status_args == ()
        assert hints.login_args == ()

    def test_the_auth_hint_names_the_credential_and_where_it_lives(self, provider):
        hint = provider.describe_setup().auth_hint

        assert "OPENROUTER_API_KEY" in hint
        assert "~/.omp/agent/.env" in hint

    def test_the_install_hint_names_the_measured_route_and_not_the_npm_one(
        self, provider
    ):
        """The npm package needs Bun ≥ 1.3.14 and does not run under node, which
        is why the hint names the release asset and its checksum file instead —
        and why the container lane's `npm ci` layer could never have carried it.
        """
        hint = provider.describe_setup().install_hint

        assert "omp-linux-x64" in hint
        assert "SHA256SUMS.txt" in hint
        assert "npm" not in hint

    @pytest.mark.parametrize("field", ["auth_hint", "install_hint"])
    def test_a_hint_is_one_stripped_line_carrying_nothing_from_this_machine(
        self, provider, field
    ):
        """Both hints are rendered into `--init` output, a log and a failure
        artifact, so neither may carry a value read from this process's
        environment or an absolute home path. The 12-character floor is what
        keeps the sweep about credentials rather than about `LANG`: shorter than
        that a value is not a secret, and longer than that its appearance here
        would be a leak.
        """
        hint = getattr(provider.describe_setup(), field)

        assert hint == hint.strip() and "\n" not in hint
        assert "/home/" not in hint
        for value in os.environ.values():
            if len(value) >= 12:
                assert value not in hint

    def test_doctors_auth_row_is_unprobed_and_spawns_nothing(self, provider):
        """`unprobed` / ok, and the `_run` tripwire is the half that matters.

        A missing key is not silently assumed good and it is not probed for
        either: it surfaces one dispatch later as omp's own pre-flight `No API
        key found for openrouter.` (exit 1, 0 B stdout, zero requests), which
        `auth_error_patterns` classifies as `auth_failed`. `doctor` is the one
        command allowed to spawn a provider CLI outside a phase, so an adapter
        that declares no probe must cost it no spawn at all.
        """
        from utils import doctor

        def boom(*_args, **_kwargs):
            raise AssertionError("doctor must not spawn anything for omp")

        with patch.object(doctor, "_run", boom):
            row = doctor._probe_auth(provider, timeout=1.0)

        assert row["state"] == "unprobed"
        assert row["detail"] == "no auth_check declared"
        assert row["ok"] is True

    def test_the_executor_preflight_answers_unknown_for_the_same_reason(self):
        """`unknown` is the honest answer for a CLI that declares no probe —
        never an assumption of success, and never a refusal either."""
        from utils import subagent_executor

        subagent_executor._AUTH_CACHE.pop("omp|1", None)
        subagent_executor._AUTH_CACHE.pop("omp|0", None)
        try:
            state = subagent_executor._detect_auth(
                "omp",
                self.MODEL,
                key="k",
                spec=f"omp:{self.MODEL}",
                profile=base_module.PROFILE_AGENT_READONLY,
            )
        finally:
            subagent_executor._AUTH_CACHE.pop("omp|1", None)
            subagent_executor._AUTH_CACHE.pop("omp|0", None)

        assert state == "unknown"

    # ── argv (§2.2) ──────────────────────────────────────────────────────
    def test_build_command_renders_the_measured_argv_in_the_measured_order(
        self, provider, ungated
    ):
        """The order is fixed as written so `expected_argv.yaml`,
        `cli-surface.yaml` and this test agree on ONE rendering."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv == [
            "omp", "--no-session", "-p",
            "--mode", "json",
            "--config", str(OMP_OVERLAY_PATH),
            "--approval-mode", "yolo",
            "--tools", "read,grep,glob,bash,edit,write",
            "--cwd", os.getcwd(),
            "--model", self.MODEL,
            "--", self.PROMPT,
        ]

    def test_the_policy_tokens_are_the_adapters_own_constants(self, provider):
        """Spelled out above AND read from the constants here, so a narrowing of
        either one still has to face the literal."""
        assert OMP_APPROVAL_MODE == "yolo"
        assert OMP_TOOLS == (
            "read", "grep", "glob", "bash", "edit", "write"
        )

    @pytest.mark.parametrize(
        "rung", ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_every_rung_of_the_ladder_renders_thinking_before_model(
        self, provider, ungated, rung
    ):
        """Seven rungs, `off` where pi spells it `none`.

        omp does NOT validate `--thinking` — `none` and `ultra` are accepted
        silently and ignored — so this whitelist is the only guard there is, and
        the suffix has to be stripped from the model before `--model` or omp's
        own native `:level` parsing would see it twice.
        """
        argv = provider.build_command(self.PROMPT, f"{self.MODEL}:{rung}")

        assert "--thinking" in argv
        assert argv[argv.index("--thinking") + 1] == rung
        assert argv.index("--thinking") < argv.index("--model")
        assert argv[argv.index("--model") + 1] == self.MODEL

    def test_no_suffix_renders_no_thinking_flag(self, provider, ungated):
        assert "--thinking" not in provider.build_command(self.PROMPT, self.MODEL)

    @pytest.mark.parametrize(
        "model",
        [
            "openrouter/z-ai/glm-5.2:free",
            "bedrock/anthropic.claude-v1:0",
            "openrouter/openai/gpt-5.6-luna:auto",
        ],
    )
    def test_a_suffix_that_is_not_a_rung_stays_in_the_model(
        self, provider, ungated, model
    ):
        """`:free` ids, bedrock `…-v1:0` version suffixes and omp's own eighth
        level `:auto` all pass through as part of the model string — which is
        exactly how `omp:<model>:auto` keeps working without `auto` being
        declared a rung it is not.
        """
        argv = provider.build_command(self.PROMPT, model)

        assert argv[argv.index("--model") + 1] == model
        assert "--thinking" not in argv

    @pytest.mark.parametrize(
        "timeout_s,expected",
        [(None, None), (1200, "1195"), (6, "1"), (5, None), (25, "20")],
    )
    def test_the_max_time_arithmetic(self, timeout_s, expected):
        """qwen's FORMULA, never qwen's helper: outer deadline minus a shutdown
        margin, `None` in ⇒ `None` out so the flag is absent rather than
        defaulted. `6 → 1` is the floor: `OMP_MIN_MAX_TIME_S` is 1 because
        `--max-time 1` was measured cutting cleanly, and copying qwen's 60 s
        minimum would render a `--max-time` larger than a short slice's own
        deadline and turn a clean `aborted` into a SIGTERM.
        """
        assert max_time_value(timeout_s) == expected
        assert OMP_SHUTDOWN_MARGIN_S == 5
        assert OMP_MIN_MAX_TIME_S == 1

    def test_a_deadline_above_the_margin_renders_the_difference(
        self, provider, ungated
    ):
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=1200)

        assert "--max-time" in argv
        assert argv[argv.index("--max-time") + 1] == "1195"
        assert argv.index("--max-time") < argv.index("--cwd")

    def test_an_undersized_deadline_omits_the_flag_and_warns_once(
        self, provider, ungated, capsys
    ):
        """The undersized case is an OMISSION, not qwen's clamp-and-render.

        The retry ladder produces a sub-margin `timeout_s` legitimately — a
        retry's deadline is the slot's REMAINING wall clock, down to a 1.0 s
        floor — and neither of the other two answers is available: clamping
        would put a `--max-time` larger than the outer deadline on argv, which
        is a token that cannot do its job, and refusing would turn the ladder's
        own short slices into failed slots, because a `build_command` raise
        escapes as the slot's error with no retry. So the flag is dropped and
        the argv is BYTE-IDENTICAL to the `timeout_s=None` rendering, with the
        parent's `subprocess.run` deadline left as the only bound.
        """
        capsys.readouterr()
        argv = provider.build_command(
            self.PROMPT, self.MODEL, timeout_s=OMP_SHUTDOWN_MARGIN_S
        )
        warning = capsys.readouterr().err

        assert argv == provider.build_command(self.PROMPT, self.MODEL, timeout_s=None)
        assert "--max-time" not in argv
        assert warning.count("WARNING") == 1
        assert "omp" in warning
        assert "qwen" not in warning

    @pytest.mark.parametrize("timeout_s", [1, 2, 3, 4, 5])
    def test_nothing_is_clamped_upward(self, provider, ungated, timeout_s):
        """A slice too short for the margin renders no bound at all; it never
        renders one LONGER than the deadline it is supposed to sit inside."""
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=timeout_s)

        assert "--max-time" not in argv

    def test_a_slice_too_short_for_the_retry_chain_is_not_undersized(
        self, provider, ungated
    ):
        """`--max-time` bounds the dispatch; it does not promise omp's own
        twelve-request 429 chain fits inside it. A 25 s slot renders 20 s
        exactly, and the slot had no time for that chain whichever way it ended.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=25)

        assert argv[argv.index("--max-time") + 1] == "20"

    def test_the_overlay_is_absolute_and_ships_with_the_skill(
        self, provider, ungated
    ):
        """A relative `--config` resolves against `--cwd`, not the launch cwd,
        and a missing file is a fast named launch failure (exit 1 in 0.13 s,
        zero requests) — so the path is resolved absolutely here and the file
        has to be in the tree the skill ships.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL)
        rendered = Path(argv[argv.index("--config") + 1])

        assert rendered.is_absolute()
        assert rendered == OMP_OVERLAY_PATH
        assert rendered.is_file()

    def test_cwd_is_rendered_on_every_dispatch(self, provider, ungated):
        """Not a tidiness flag, and not conditional on the spawn directory.

        Launched from `$HOME` without it omp silently relocates to `/tmp` —
        nothing on either channel says so — and a reviewer that moved to `/tmp`
        reads none of the repo's files, has an empty `git diff`, writes
        `{output_json_path}` into `/tmp`, and still exits 0 with a
        plausible-looking answer. The obvious alternative is worse:
        `--allow-home` keeps the cwd AND auto-mounts the MCP servers from
        `~/.cursor/mcp.json`, so it is rendered on no profile.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL, cwd="/srv/project")

        assert argv[argv.index("--cwd") + 1] == "/srv/project"
        assert "--allow-home" not in argv

    def test_cwd_defaults_to_the_process_directory(self, provider, ungated):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("--cwd") + 1] == os.getcwd()

    def test_the_stdin_rendering_is_the_argv_minus_its_last_two_tokens(
        self, provider, ungated
    ):
        """That `[:-2]` is omp's own override and not the base helper's `[:-1]`:
        dropping only the prompt would leave a bare `--` on argv."""
        argv = provider.build_command(self.PROMPT, self.MODEL)
        stdin_argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )

        assert argv[-2:] == ["--", self.PROMPT]
        assert stdin_argv == argv[:-2]
        assert stdin_text == self.PROMPT
        assert stdin_argv[-2:] == ["--model", self.MODEL]

    def test_the_argv_transport_still_renders_the_prompt_last(
        self, provider, ungated
    ):
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "argv"
        )

        assert stdin_text is None
        assert argv == provider.build_command(self.PROMPT, self.MODEL)

    def test_an_unsupported_transport_is_refused_by_name(self, provider, ungated):
        with pytest.raises(ValueError) as excinfo:
            provider.build_command_for_transport(
                self.PROMPT, self.MODEL, "prompt-file"
            )

        assert "prompt-file" in str(excinfo.value)

    @pytest.mark.parametrize("prompt", ["@plans/auth.md", "--help me review this"])
    def test_a_sigil_prefixed_prompt_is_literal_on_stdin(
        self, provider, ungated, prompt
    ):
        """On stdin every prompt is literal whatever its first character: the
        bytes never meet a flag parser or omp's `@` attachment at all, which is
        why this transport is the adapter's default. The two argv tests below are
        the other half — there the same two sigils are made literal by `--`."""
        argv, stdin_text = provider.build_command_for_transport(
            prompt, self.MODEL, "stdin"
        )

        assert stdin_text == prompt
        assert prompt not in argv

    def test_a_dash_prefixed_prompt_is_accepted_on_argv_too(
        self, provider, ungated
    ):
        """`--` ends flag parsing, so a positional beginning with `-` is text."""
        prompt = "--help me review this"
        argv = provider.build_command(prompt, self.MODEL)

        assert argv[-2:] == ["--", prompt]

    def test_an_at_prefixed_prompt_is_accepted_on_argv_behind_the_dash_dash(
        self, provider, ungated
    ):
        """An `@`-prefixed prompt is a prompt, and `--` is what keeps it one.

        Measured at 18.2.2 in both directions: after `--`, `@atfile.txt` reaches
        the model as that literal text and the file is not read; the same prompt
        without `--` has the file's contents inlined into the user message. This
        adapter's argv always ends `["--", prompt]`, so the prompt is accepted
        rather than refused — and the two positional assertions are the
        guarantee, not decoration: tidying the `--` away would turn every
        `@`-prefixed prompt into a silent file read, and it fails here first.
        """
        prompt = "@plan.md"
        argv, stdin_text = provider.build_command_for_transport(
            prompt, self.MODEL, "argv"
        )

        assert stdin_text is None
        assert argv[-1] == prompt
        assert argv[-2] == "--"

    # ── the flag gate (§2.11) ────────────────────────────────────────────
    @staticmethod
    def _with_help(tokens):
        """The gate reading a help text this test wrote, on a binary that
        resolves. Both halves are patched because the gate needs a spawnable
        path before it will read anything, and a machine that happens to have
        omp installed must not be what decides the result.
        """
        return (
            patch.object(LLMProvider, "gate_binary", lambda self, **_kw: sys.executable),
            patch.object(
                base_module, "read_help_tokens", lambda _b, **_kw: frozenset(tokens)
            ),
        )

    def test_a_help_text_missing_one_flag_refuses_by_that_flags_name(self, provider):
        """`--mode` has no safe degraded form: without it the parser reads a text
        stream, so the gate is the load-bearing form and not the optional one.
        """
        tokens = set(OMP_REQUIRED_FLAGS) - {"--mode"}
        binary, help_text = self._with_help(tokens)
        with binary, help_text, pytest.raises(OmpFlagsUnconfirmed) as excinfo:
            provider.build_command(self.PROMPT, self.MODEL)

        refusal = excinfo.value
        assert refusal.missing == ("--mode",)
        assert refusal.state == base_module.VERSION_TOO_LOW
        assert "--mode" in str(refusal)

    def test_an_empty_help_names_every_required_flag(self, provider):
        """An empty help is the undecidable case, and the load-bearing form
        declines toward refusing the SLOT rather than toward a degraded argv:
        without `--tools` or `--approval-mode` the run inherits omp's factory
        `yolo` over every tool it has, and without `--config` the retry floor is
        unbounded.
        """
        binary, help_text = self._with_help(())
        with binary, help_text, pytest.raises(OmpFlagsUnconfirmed) as excinfo:
            provider.build_command(self.PROMPT, self.MODEL)

        assert excinfo.value.missing == OMP_REQUIRED_FLAGS
        for flag in OMP_REQUIRED_FLAGS:
            assert flag in str(excinfo.value)

    def test_p_is_confirmed_through_its_long_spelling(self, provider):
        """The shared token regex reads long options only, so `-p` is gated
        through `--print` — the long spelling on the same help line — while the
        argv keeps `-p`."""
        assert "--print" in OMP_REQUIRED_FLAGS
        assert "-p" not in OMP_REQUIRED_FLAGS

        binary, help_text = self._with_help(OMP_REQUIRED_FLAGS)
        with binary, help_text:
            argv = provider.build_command(self.PROMPT, self.MODEL)

        assert "-p" in argv
        assert "--print" not in argv

    def test_a_flag_this_call_does_not_emit_is_never_probed(self, provider):
        """`--thinking` and `--max-time` are appended to the probed set only when
        the call really renders them, so a build lacking `--max-time` still runs
        the dispatches that do not ask for one."""
        binary, help_text = self._with_help(OMP_REQUIRED_FLAGS)
        with binary, help_text:
            provider.build_command(self.PROMPT, self.MODEL)

            with pytest.raises(OmpFlagsUnconfirmed) as excinfo:
                provider.build_command(self.PROMPT, f"{self.MODEL}:high")

        assert excinfo.value.missing == ("--thinking",)

    def test_a_refusal_names_the_binary_and_the_operators_two_moves(self, provider):
        """Raised BEFORE any argv exists — no call ever gets a partial command —
        which is also what lets the exception propagate out of
        `run_single_model`'s ladder loop into its per-model `except`, recorded as
        that slot's error with `ladder.record` never reached: no retry, by
        construction, and the other slots of the fan-out unaffected. It borrows
        the `version-too-low` word as cursor's does, but the claim it makes is
        the stronger one — the name of the flag, not a number.
        """
        binary, help_text = self._with_help(())
        with binary, help_text, pytest.raises(OmpFlagsUnconfirmed) as excinfo:
            provider.build_command(self.PROMPT, self.MODEL)

        refusal = excinfo.value
        assert refusal.binary == sys.executable
        assert "Upgrade omp" in str(refusal)
        assert "drop omp from --models" in str(refusal)

    def test_an_unresolvable_binary_renders_the_argv_without_a_probe(self, provider):
        """"Not installed" is `is_available()`'s refusal to make, with its own
        precise message; a binary that cannot be spawned cannot reject a flag
        either. It is also the read-only scan's precondition — that guard reads
        this repo's source and must never depend on which CLIs a machine has.
        """
        def boom(*_args, **_kwargs):
            raise AssertionError("an unresolvable binary must not be probed")

        with patch.object(LLMProvider, "gate_binary", lambda self, **_kw: None), \
                patch.object(base_module, "read_help_tokens", boom):
            argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[0] == "omp"
        assert argv[-2:] == ["--", self.PROMPT]

    # ── guard visibility (§2.3) ──────────────────────────────────────────
    def test_the_bypass_is_exactly_one_visible_token_and_it_is_declared(
        self, provider, ungated
    ):
        """The token the scan reports is the VALUE, `yolo`, and never the flag.

        `negative_tokens_in_argv` records whole argv elements: with `yolo` in
        `NEGATIVE_VALUE_TOKENS` the pair `--approval-mode yolo` reports the value
        element once and the flag element never (it is in neither
        `NEGATIVE_FLAG_TOKENS` nor `NEGATIVE_TOKEN_PREFIXES`). Both sides of the
        machinery key on that one string, so a `token: "--approval-mode"` row
        would fail in both directions at once — `yolo` unexcused, and
        `--approval-mode` "never emitted".
        """
        from tests.test_read_only_modes import negative_tokens_in_argv, read_only_modes

        emitted = negative_tokens_in_argv(
            provider.build_command(self.PROMPT, self.MODEL)
        )

        assert emitted == ["yolo"]

        exception = read_only_modes()["omp"]["bypass_exception"]
        assert exception["token"] == "yolo"
        assert exception["profile"] == base_module.PROFILE_GENERATE
        assert read_only_modes()["omp"]["mechanism"] is None

    def test_the_approval_mode_is_on_argv_and_never_in_the_overlay(self):
        """The adapter cannot rely on a safe default because there is none —
        omp's own factory default IS `yolo` over every tool — and it must not
        rely on the user's `config.yml` either. Putting the mode in the overlay
        would make both irrelevant the same way and be invisible to the read-only
        scan, which is the one thing that may not happen. Read as a document
        rather than as text, so the word appearing in the file's own comments is
        not what this asserts about.
        """
        import yaml

        settings = yaml.safe_load(OMP_OVERLAY_PATH.read_text(encoding="utf-8")) or {}

        def _keys(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    yield from _keys(value)
            elif isinstance(node, list):
                for value in node:
                    yield from _keys(value)

        assert "approvalMode" not in set(_keys(settings))
        assert "tools" not in set(_keys(settings))
        # What the overlay IS for: the retry bound and the project-MCP
        # suppression, both of which the argv has no flag for.
        assert settings["retry"]["enabled"] is False
        assert settings["mcp"]["enableProjectConfig"] is False

    def test_no_auto_approve_flag_is_ever_rendered(self, provider, ungated):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert "--auto-approve" not in argv
        assert "--yolo" not in argv

    # ── the file-output declaration (§2.3) ───────────────────────────────
    def test_the_file_channel_is_declared_and_nothing_adjudicates_it(self, provider):
        """`supports_file_output = True` is a MEASUREMENT, not an inference.

        Under `--approval-mode yolo --tools read,grep,glob,bash,edit,write`
        the `write` and `edit` tools really created and edited files — a file
        appeared, and `p1.txt` went alpha→beta — so the declaration records what
        was observed. The empty atom list below is a separate fact and would be
        poor evidence on its own: `RESTRICTIVE_ATOM_RE` matches only
        `ask|plan|read-only|no-write|dry-run|deny|restrict|sandbox|safe` as whole
        words plus a zero budget, and `read,` is not `read-only` while `edit` and
        `write` are not `no-write`. So omp owns NO row in either adjudication
        table, any row added would be stale on arrival, and the regex is
        deliberately not widened to make one fit — a write-ENABLING flag cannot
        cause the failure that table guards. The day a profile renders a
        narrower `--tools` or an `always-ask` mode, that atom IS visible and the
        adjudication is owed then.
        """
        from tests.test_provider_housekeeping import (
            REVIEWED_STILL_WRITES,
            STOPS_FILE_WRITES,
            restrictive_atoms,
        )

        assert provider.supports_file_output is True
        assert restrictive_atoms("omp") == []
        assert not [key for key in STOPS_FILE_WRITES if key[0] == "omp"]
        assert not [key for key in REVIEWED_STILL_WRITES if key[0] == "omp"]

    def test_the_write_profile_is_not_declared(self, provider):
        """`generate` is the only profile this adapter ships. The accepted
        disposition is scoped to it, and an `agent-write` runner would be a
        separate decision on its own measurement.
        """
        assert provider.agent_profiles == (base_module.PROFILE_GENERATE,)
        assert provider.supports_file_edits is False

    # ── env (§2.10) ──────────────────────────────────────────────────────
    SOURCE_ENV = {
        "HOME": "/home/e2e",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "OPENROUTER_API_KEY": "sk-or-from-the-source-env",
        "GEMINI_API_KEY": "gemini-key",
        "OPENAI_API_KEY": "openai-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
    }

    def _composed(self, provider, source):
        """The child env both ways round the allowlist, as `(on, off)`.

        One composition point — `build_child_env` over `resolve_env` — because
        that is what both spawn sites call, and an adapter whose delta survived
        on one lane and not the other would be a credential that is present or
        absent depending on which phase asked.
        """
        out = []
        for enabled in (True, False):
            with patch.object(
                base_module, "env_allowlist_enabled", return_value=enabled
            ):
                out.append(
                    base_module.build_child_env(
                        provider,
                        provider_env=provider.resolve_env(
                            self.MODEL, base_module.PROFILE_GENERATE, base_env=source
                        ),
                        base_env=source,
                    )
                )
        return out[0], out[1]

    def test_the_credential_survives_the_filter_in_both_modes(
        self, provider, monkeypatch
    ):
        """Under `env_allowlist: on` nothing in omp's profile admits
        `OPENROUTER_API_KEY` — the allowlist requires a credential to start with
        the adapter's own prefix and omp's family is `PI_`/`OMP_` — so the one
        auth source measured as "the env alone works" would be silently dropped
        and omp would fall through its `$env` chain to a file, or exit 1.
        Re-emitting it from `get_env` is the route kimi, qwen and copilot already
        take; omp's own spelling happens to be the variable's real name.

        `get_env` reads that value from `os.environ` rather than from the
        `base_env` it is composed over, so both are set here and agree. The two
        lanes' composed envs are then asserted EQUAL, which is the property that
        matters: the allowlist decision may not change what omp is handed.
        """
        monkeypatch.setenv(
            "OPENROUTER_API_KEY", self.SOURCE_ENV["OPENROUTER_API_KEY"]
        )
        on, off = self._composed(provider, self.SOURCE_ENV)

        for env in (on, off):
            assert env["OPENROUTER_API_KEY"] == "sk-or-from-the-source-env"
            assert env["OMP_SKIP_SETUP"] == "1"
            assert env["HOME"] == "/home/e2e"
        assert on == off

    def test_an_absent_key_is_absent_rather_than_injected_empty(
        self, provider, monkeypatch
    ):
        """Conditional, not `os.environ.get(name, "")`.

        An injected empty string would be PRESENT and wrong to omp's `$env`
        chain, and would shadow the `~/.omp/agent/.env` source a user with no
        exported key relies on — turning a working configuration into a
        pre-flight `No API key found for openrouter.`
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        source = {
            key: value
            for key, value in self.SOURCE_ENV.items()
            if key != "OPENROUTER_API_KEY"
        }

        on, off = self._composed(provider, source)

        for env in (on, off):
            assert "OPENROUTER_API_KEY" not in env
            assert env["OMP_SKIP_SETUP"] == "1"
        assert on == off

    def test_every_foreign_vendor_key_is_stripped_in_both_modes(
        self, provider, monkeypatch
    ):
        """"Adapter wins, ambient loses, no fallback."

        This is what closes the measured hazard on a DEFAULT install, not the
        env allowlist — that ships `off`, so with the filter off the child is the
        source environment plus this adapter's delta minus these removals, and
        without the list a stray `GEMINI_API_KEY` reaches omp on every dispatch.
        The two measured rows are the reason it is a list and not a comment: with
        either of them in the process env a run with no `--model` picked
        `google/gemini-3.1-pro-preview` and spun on a real 429.
        """
        monkeypatch.setenv(
            "OPENROUTER_API_KEY", self.SOURCE_ENV["OPENROUTER_API_KEY"]
        )
        source = {
            **self.SOURCE_ENV,
            **{name: "leaked" for name in omp_module._STRIPPED_VENDOR_VARS},
        }

        on, off = self._composed(provider, source)

        for env in (on, off):
            for name in omp_module._STRIPPED_VENDOR_VARS:
                assert name not in env
        assert on == off

    def test_nothing_the_adapter_injects_is_also_stripped(self, provider):
        """An adapter that strips what it injects is a bug, and the two
        declarations sit far enough apart in the file to make it a likely one."""
        injected = set(provider.get_env(self.MODEL))
        stripped = set(provider.get_remove_env())

        assert "OPENROUTER_API_KEY" not in omp_module._STRIPPED_VENDOR_VARS
        assert injected & stripped == set()


class TestPiProvider:
    """Tests for the PiProvider class."""

    @pytest.fixture(autouse=True)
    def tools_gate_declined(self):
        """Every exact argv below is the argv WITHOUT ``--tools`` (T3.1).

        ``build_command`` now asks the shared per-flag gate whether the
        installed pi documents ``--tools``, and that reads the developer's own
        ``pi --help`` when one is on PATH — so these pinned argvs would depend
        on the machine. With no gate binary the optional-flag gate declines, and
        the argv is exactly today's; the WITH-the-flag argv is pinned on its own
        in :class:`TestPiToolAllowlist`.
        """
        with patch.object(LLMProvider, "gate_binary", lambda self, **_kw: None):
            yield

    @pytest.fixture
    def provider(self):
        """Create a PiProvider instance."""
        return PiProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "pi"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    @patch("shutil.which")
    def test_is_available_true(self, mock_which, provider):
        """Test is_available returns True when pi is found."""
        mock_which.return_value = "/usr/local/bin/pi"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("pi")

    @patch("shutil.which")
    def test_is_available_false(self, mock_which, provider):
        """Test is_available returns False when pi is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("pi")

    def test_build_command(self, provider):
        """Model patterns pass VERBATIM to --model; prompt goes last."""
        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd == [
            "pi",
            "--no-session",
            "-p",
            "--mode",
            "json",
            "--model",
            "openrouter/z-ai/glm-5.2",
            "Review this code",
        ]

    def test_build_command_bare_model(self, provider):
        """A bare model pattern without a provider prefix passes through."""
        cmd = provider.build_command("Analyze this", "sonnet")

        assert cmd == [
            "pi", "--no-session", "-p", "--mode", "json", "--model", "sonnet",
            "Analyze this",
        ]

    def test_internal_retry_envelope_is_the_measured_chain(self, provider):
        """T7.14 (decision 12.7), pinned the way copilot's envelope is.

        MEASURED at 0.87.1 (tests/fixtures/pi/manifest.yaml: rate-limit,
        server-error, truncated, error-envelope, zero-body): 4 requests per
        failing model call = 1 + ``retry.maxRetries`` (3), a 429 among the
        retried errors, no outer wall bound, and the two settings keys that
        change it.
        """
        envelope = provider.internal_retries

        assert envelope.max_requests == 4
        assert envelope.retries_rate_limit is True
        assert envelope.max_wall_s is None
        assert envelope.configurable_by == (
            "retry.maxRetries", "retry.provider.maxRetries",
        )

    @pytest.mark.parametrize("model", [
        "openrouter/z-ai/glm-5.2",
        "openrouter/z-ai/glm-5.2:high",
        "openrouter/z-ai/glm-5.2:off",
        "openrouter/deepseek/deepseek-r1:free",
    ])
    def test_mode_json_is_always_rendered_right_after_print(self, provider, model):
        """``--mode json`` on every argv — with and without ``--thinking`` — and
        right after ``-p`` (omp's order), so ``parse_output`` always reads the
        JSONL stream it was written for (plan §7.3, T7.5)."""
        cmd = provider.build_command("Review this code", model)

        assert cmd[:5] == ["pi", "--no-session", "-p", "--mode", "json"]
        assert cmd.count("--mode") == 1

    def test_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the seven valid levels."""
        from utils.providers.pi import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {
            "off", "minimal", "low", "medium", "high", "xhigh", "max",
        }

    def test_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --thinking pair before --model."""
        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2:high")

        assert cmd == [
            "pi",
            "--no-session",
            "-p",
            "--mode",
            "json",
            "--thinking",
            "high",
            "--model",
            "openrouter/z-ai/glm-5.2",
            "Review this code",
        ]

    def test_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --thinking form."""
        efforts = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
        for effort in efforts:
            cmd = provider.build_command("Review this code", f"openrouter/z-ai/glm-5.2:{effort}")

            assert cmd == [
                "pi",
                "--no-session",
                "-p",
                "--mode",
                "json",
                "--thinking",
                effort,
                "--model",
                "openrouter/z-ai/glm-5.2",
                "Review this code",
            ], f"effort {effort!r} did not build the expected command"

    def test_build_command_effort_passthrough(self, provider):
        """Non-effort suffixes stay part of the model name, no --thinking flag.

        Critically, openrouter ":free" variants and bedrock ":0" version
        suffixes are NOT effort suffixes: the whole string stays the model.
        """
        models = (
            "openrouter/deepseek/deepseek-r1:free",
            "amazon-bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        )
        for model in models:
            cmd = provider.build_command("Review this code", model)

            assert cmd == [
                "pi",
                "--no-session",
                "-p",
                "--mode",
                "json",
                "--model",
                model,
                "Review this code",
            ], f"model {model!r} was not passed through verbatim"

    def test_parse_output_direct_json_array(self, provider):
        """Parse direct JSON array output."""
        data = [{"item": 1}, {"item": 2}]
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_parse_output_direct_json_object(self, provider):
        """Parse direct JSON object output."""
        data = {"status": "ok", "items": [1, 2, 3]}
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_parse_output_code_block(self, provider):
        """Extract JSON from markdown code block."""
        inner_json = [{"title": "Suggestion 1", "importance": "high"}]
        stdout = f"""Here is my analysis:

```json
{json.dumps(inner_json, indent=2)}
```

This concludes the review."""

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_embedded_json(self, provider):
        """Extract embedded JSON from plain text."""
        inner_json = [{"id": 1}, {"id": 2}]
        stdout = f"The results are: {json.dumps(inner_json)} end of results."

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_empty(self, provider):
        """Handle empty output."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert "Empty output" in result["error"]

    def test_parse_output_whitespace_only(self, provider):
        """Handle whitespace-only output."""
        result = provider.parse_output("   \n\t  ", "")

        assert result["success"] is False
        assert "Empty output" in result["error"]

    def test_parse_output_no_json(self, provider):
        """Handle output with no extractable JSON."""
        result = provider.parse_output("Just plain text with no JSON at all.", "")

        assert result["success"] is False
        assert "error" in result


class TestPiToolAllowlist:
    """T3.1 (decision 11.6): ``--tools read,grep,find,bash,write``, once.

    pi drops an unknown tool name SILENTLY and parses ``--tools`` as one
    comma-separated value that a repeated flag replaces, so every property
    pinned here is one whose failure is invisible at run time: a misspelled
    name quietly removes a tool, and a wrong separator quietly removes them all.
    """

    PROMPT = "Review this code"
    MODEL = "openrouter/z-ai/glm-5.2"
    #: pi 0.87.1's built-in tool names (``pi --help``: "read, bash, edit, write
    #: tools" by default, plus grep/find/ls as optional built-ins).
    BUILTINS = {"read", "bash", "edit", "write", "grep", "find", "ls"}

    @pytest.fixture
    def gate(self):
        """Answer the ``--tools`` gate without reading any installed help.

        ``state["supported"]`` is the answer (a pi that documents the flag by
        default); ``state["asked"]`` records every flag the adapter asked about.
        """
        state = {"supported": True, "asked": []}

        def supports_flag(_provider, flag):
            state["asked"].append(flag)
            return state["supported"]

        with patch.object(PiProvider, "supports_flag", supports_flag):
            yield state

    def test_the_list_is_the_decided_one(self):
        from utils.providers.pi import PI_TOOLS

        assert PI_TOOLS == ("read", "grep", "find", "bash", "write")

    def test_the_list_is_built_in_vocabulary_only(self):
        """An unknown name would be dropped silently, never refused."""
        from utils.providers.pi import PI_TOOLS

        assert set(PI_TOOLS) <= self.BUILTINS
        assert len(set(PI_TOOLS)) == len(PI_TOOLS)

    def test_edit_is_not_allowed(self):
        """``write`` covers the output file; ``edit`` stays out (D3.1)."""
        from utils.providers.pi import PI_TOOLS

        assert "edit" not in PI_TOOLS

    def test_no_name_carries_a_separator(self):
        """A comma or a space inside a name would split or merge names."""
        from utils.providers.pi import PI_TOOLS

        for name in PI_TOOLS:
            assert name and name == name.strip()
            assert "," not in name
            assert not any(ch.isspace() for ch in name)

    def test_exact_argv_with_the_flag(self, gate):
        cmd = PiProvider().build_command(self.PROMPT, self.MODEL)

        assert cmd == [
            "pi", "--no-session", "-p", "--mode", "json",
            "--tools", "read,grep,find,bash,write", "--no-mcp",
            "--model", self.MODEL, self.PROMPT,
        ]
        assert gate["asked"] == ["--tools", "--no-mcp"]

    def test_exact_argv_with_the_flag_and_an_effort(self, gate):
        cmd = PiProvider().build_command(self.PROMPT, f"{self.MODEL}:high")

        assert cmd == [
            "pi", "--no-session", "-p", "--mode", "json",
            "--tools", "read,grep,find,bash,write", "--no-mcp",
            "--thinking", "high",
            "--model", self.MODEL, self.PROMPT,
        ]

    def test_mcp_is_switched_off_when_the_installed_pi_documents_it(self, gate):
        """pi 1.1.0: ``--tools`` KEEPS MCP tools unless an entry starts with
        ``mcp__``, and every server in ``~/.pi/agent/mcp.json`` connects at
        start-up, so the allow-list alone no longer bounds the reviewer's tools.
        ``--no-mcp`` once, after the allow-list; absent where pi's help does not
        document it (no built-in MCP client to switch off), with no warning."""
        from utils.providers.pi import PI_NO_MCP_FLAG

        assert PI_NO_MCP_FLAG == "--no-mcp"
        cmd = PiProvider().build_command(self.PROMPT, self.MODEL)
        assert cmd.count("--no-mcp") == 1
        assert cmd.index("--no-mcp") == cmd.index("--tools") + 2

        with patch.object(PiProvider, "supports_flag",
                          lambda _provider, flag: flag != "--no-mcp"):
            older = PiProvider().build_command(self.PROMPT, self.MODEL)
        assert "--no-mcp" not in older and "--tools" in older

    def test_exact_argv_without_the_flag_is_todays(self, gate, capsys):
        """A pi whose help does not document ``--tools``: today's argv + one WARN."""
        gate["supported"] = False
        base_module.reset_flag_gate_cache()
        try:
            first = PiProvider().build_command(self.PROMPT, self.MODEL)
            second = PiProvider().build_command(self.PROMPT, self.MODEL)
        finally:
            warned = capsys.readouterr().err
            base_module.reset_flag_gate_cache()

        assert first == second == [
            "pi", "--no-session", "-p", "--mode", "json",
            "--model", self.MODEL, self.PROMPT,
        ]
        assert warned.count("[WARN] pi: --tools") == 1
        assert "read,grep,find,bash,write" in warned

    @pytest.mark.parametrize("model", [
        "openrouter/z-ai/glm-5.2",
        "openrouter/z-ai/glm-5.2:off",
        "openrouter/deepseek/deepseek-r1:free",
    ])
    def test_the_flag_appears_once_with_one_comma_joined_value(self, gate, model):
        """One flag, one value: a second ``--tools`` would REPLACE the first."""
        from utils.providers.pi import PI_TOOLS

        cmd = PiProvider().build_command(self.PROMPT, model)

        assert cmd.count("--tools") == 1
        assert "-t" not in cmd
        value = cmd[cmd.index("--tools") + 1]
        assert value == ",".join(PI_TOOLS)
        assert value.split(",") == list(PI_TOOLS)
        # Right after `--mode json`, the position the argv table pins.
        assert cmd[3:7] == ["--mode", "json", "--tools", value]
        # And the prompt is still the last element, whatever came before it.
        assert cmd[-1] == self.PROMPT

    def test_the_prompt_is_last_without_the_flag_too(self, gate):
        gate["supported"] = False
        cmd = PiProvider().build_command(self.PROMPT, f"{self.MODEL}:high")

        assert cmd[-1] == self.PROMPT
        assert "--tools" not in cmd

    def test_the_hand_edited_argv_row_carries_the_same_value(self):
        """``expected_argv.yaml`` is edited by hand; keep it on the constant."""
        import yaml
        from utils.providers.pi import PI_TOOLS

        table = yaml.safe_load(
            (Path(__file__).parent / "container" / "expected_argv.yaml")
            .read_text(encoding="utf-8")
        )
        rows = [row for row in table["entries"]
                if row["provider"] == "pi" and row["profile"] == "generate"]

        assert rows, "the pi generate row is gone from expected_argv.yaml"
        for row in rows:
            assert row["options"]["--tools"] == ",".join(PI_TOOLS)
        # The gate's own help read is a call-log record too.
        assert any(row["provider"] == "pi" and row["profile"] == "probe"
                   and "--help" in row["options"] for row in table["entries"])

    def test_the_surface_contract_pins_the_flag_and_its_form(self):
        """Presence of ``--tools`` AND its comma-separated form, hand-kept."""
        import yaml

        surface = yaml.safe_load(
            (Path(__file__).parent / "container" / "cli-surface.yaml")
            .read_text(encoding="utf-8")
        )
        rows = {row["cli"]: row for row in surface["clis"]}

        assert "--tools" in rows["pi"]["options"]
        assert rows["pi"]["option_forms"] == {"--tools": "Comma-separated"}
        # A pinned form is only meaningful for a flag the row really emits.
        for cli, row in rows.items():
            for option in (row.get("option_forms") or {}):
                assert option in row["options"], (cli, option)


class TestCodexProvider:
    """Tests for the CodexProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a CodexProvider instance."""
        return CodexProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "codex"

    def test_default_timeout(self, provider):
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

    def test_codex_reasoning_efforts_constant(self):
        """REASONING_EFFORTS is the seven values codex EMITS (L11 #1, §6.1).

        `max` was missing, so `gpt-5.6-sol:max` — a combination the codex
        catalog advertises — had its suffix glued onto the model name. `ultra`
        is deliberately NOT here: it is an inbound alias the harness rewrites,
        never a value this adapter puts on argv.
        """
        from utils.providers.codex import EFFORT_ALIASES, REASONING_EFFORTS

        assert REASONING_EFFORTS == {
            "none", "minimal", "low", "medium", "high", "xhigh", "max",
        }
        assert EFFORT_ALIASES == {"ultra": "max"}

    def test_codex_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: -c pair after --model <base>, prompt last."""
        cmd = provider.build_command("p", "gpt-5.5:high")

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.5",
            "-c",
            "model_reasoning_effort=high",
            "p",
        ]

    def test_codex_build_command_with_a_schema_exact(self, provider, tmp_path):
        """§4.2/§4.2.2's exact argv — codex's half is a FILE, not inline JSON."""
        from utils.providers.base import StructuredSchema

        path = tmp_path / "probe.schema.json"
        path.write_text('{"type": "object"}')
        schema = StructuredSchema(
            document={"type": "object"}, path=str(path), name="probe.schema.json"
        )

        cmd = provider.build_command("p", "gpt-5.5:high", schema=schema)

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--output-schema",
            str(path),
            "--model",
            "gpt-5.5",
            "-c",
            "model_reasoning_effort=high",
            "p",
        ]

    def test_codex_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the -c model_reasoning_effort form."""
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            cmd = provider.build_command("Review this code", f"gpt-5.5:{effort}")

            assert cmd == [
                "codex",
                "exec",
                "--sandbox",
                "workspace-write",
                "--json",
                "-c", "features.plugins=false",
                "--ephemeral",
                "--skip-git-repo-check",
                "--model",
                "gpt-5.5",
                "-c",
                f"model_reasoning_effort={effort}",
                "Review this code",
            ], f"effort {effort!r} did not build the expected command"

    def test_codex_build_command_max_effort_reaches_the_flag(self, provider):
        """L11 #1's live defect, stated as the argv it broke.

        The codex catalog advertises `max` for `gpt-5.6-sol/-terra/-luna`
        (`codex debug models` → `supported_reasoning_levels`). Before L11 the
        adapter's whitelist stopped at `xhigh`, so `split_reasoning_effort`
        found no known suffix and emitted `--model gpt-5.6-sol:max` — a model id
        that does not exist — with no `-c` at all.
        """
        cmd = provider.build_command("p", "gpt-5.6-sol:max")

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.6-sol",
            "-c",
            "model_reasoning_effort=max",
            "p",
        ]

    def test_codex_build_command_ultra_is_normalized_to_max(self, provider):
        """`ultra` is accepted at the harness boundary and EMITTED as `max` (§6.1).

        codex normalizes `ultra` down to `max` client-side, but that backstop is
        deliberately not relied on: `ultra` is not a canonical rung, so the
        rewrite happens here and argv carries `max`.
        """
        cmd = provider.build_command("p", "gpt-5.6-sol:ultra")

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.6-sol",
            "-c",
            "model_reasoning_effort=max",
            "p",
        ]

    def test_codex_build_command_multi_colon_splits_on_last(self, provider):
        """Multi-colon model rsplits on the LAST colon: base foo:bar, effort high."""
        cmd = provider.build_command("p", "foo:bar:high")

        assert cmd == [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--json",
            "-c", "features.plugins=false",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            "foo:bar",
            "-c",
            "model_reasoning_effort=high",
            "p",
        ]

    def test_codex_build_command_passthrough_no_colon(self, provider):
        """A model without a colon is passed verbatim with no -c argument."""
        cmd = provider.build_command("p", "gpt-5.5")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5", "p"]

    def test_codex_build_command_passthrough_unknown_suffix(self, provider):
        """An unknown suffix is not an effort: model passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:turbo")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5:turbo", "p"]

    def test_codex_build_command_passthrough_wrong_case_suffix(self, provider):
        """Effort matching is case-sensitive: :HIGH is passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:HIGH")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5:HIGH", "p"]

    def test_codex_build_command_passthrough_empty_suffix(self, provider):
        """A trailing colon with no effort is passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5:", "p"]

    def test_codex_build_command_passthrough_bare_effort_word(self, provider):
        """A bare effort word as the full model name (no colon) is passed verbatim."""
        cmd = provider.build_command("p", "high")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "high", "p"]

    def test_codex_build_command_passthrough_empty_base(self, provider):
        """An empty base before the colon (:high) is passed verbatim, no -c."""
        cmd = provider.build_command("p", ":high")

        assert cmd == ["codex", "exec", "--sandbox", "workspace-write", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", ":high", "p"]

    # --- M2: --ephemeral + --skip-git-repo-check, unconditional (§3.2) -------
    #
    # Both were re-verified FUNCTIONALLY at the declared floor (`min_version`
    # 0.144.0) as well as at 0.146.0, which is why there is no version gate to
    # test: every codex dispatch carries them, on every profile and every
    # transport. `--ephemeral` stops each fan-out call writing a rollout file
    # into `$CODEX_HOME/sessions` that multi-llm never resumes (a four-model,
    # ten-phase run left 40 dead sessions); `--skip-git-repo-check` opens the
    # trust gate that otherwise makes codex exit 1 with ZERO requests outside a
    # git repo — the measured discriminator.

    M2_FLAGS = ("--ephemeral", "--skip-git-repo-check")

    def test_codex_m2_flags_on_every_profile_and_transport(self, provider):
        """No profile and no transport may drop either flag."""
        from utils.providers.base import PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE

        for transport in ("argv", "stdin"):
            argv, _stdin = provider.build_command_for_transport(
                "p", "gpt-5.5", transport)
            for flag in self.M2_FLAGS:
                assert flag in argv, (
                    f"generate/{transport} lost {flag}: {argv}")
            for profile in (PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE):
                argv, _stdin = provider.build_agent_command_for_transport(
                    "p", "gpt-5.5", profile, transport)
                for flag in self.M2_FLAGS:
                    assert flag in argv, (
                        f"{profile}/{transport} lost {flag}: {argv}")

    def test_codex_m2_flags_are_valueless_and_precede_the_prompt(self, provider):
        """Both are boolean flags, and neither may sit where it swallows argv[-1].

        A boolean flag placed last would take the prompt as its value on the
        real CLI, and `build_command_for_transport`'s stdin leg — which drops
        `cmd[-1]` — would then drop the wrong element.
        """
        cmd = provider.build_command("the prompt", "gpt-5.5:high")

        assert cmd[-1] == "the prompt"
        for flag in self.M2_FLAGS:
            index = cmd.index(flag)
            assert cmd[index + 1].startswith("-"), (
                f"{flag} is followed by {cmd[index + 1]!r}, which a boolean flag "
                "would swallow as its value")
        assert cmd.count("--ephemeral") == 1
        assert cmd.count("--skip-git-repo-check") == 1

    def test_codex_agent_readonly_keeps_m2_flags_and_rewrites_sandbox(self, provider):
        """The read-only rewrite is positional; M2's flags must survive it intact."""
        from utils.providers.base import PROFILE_AGENT_READONLY

        cmd = provider.build_agent_command("p", "gpt-5.5", PROFILE_AGENT_READONLY)

        assert cmd == ["codex", "exec", "--sandbox", "read-only", "--json",
                       "-c", "features.plugins=false",
                       "--ephemeral", "--skip-git-repo-check",
                       "--model", "gpt-5.5", "p"]

    @patch("shutil.which")
    def test_codex_is_available_true(self, mock_which, provider):
        """Test is_available returns True when codex is found."""
        mock_which.return_value = "/usr/bin/codex"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("codex")

    @patch("shutil.which")
    def test_codex_is_available_false(self, mock_which, provider):
        """Test is_available returns False when codex is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("codex")


def _codex_stream(*events) -> str:
    """Join events into the NDJSON stream shape `codex exec --json` writes."""
    return "\n".join(json.dumps(event) for event in events)


def _agent_message(text: str, item_id: str = "item_1") -> dict:
    return {"type": "item.completed",
            "item": {"id": item_id, "type": "agent_message", "text": text}}


class TestCodexParseOutput:
    """Parsing of the `codex exec --json` NDJSON event stream.

    Event shapes here are taken from live codex-cli 0.144.0 output. The reply
    is nested in `item.completed` -> `item.type == "agent_message"`; there is
    no top-level "text"/"message"/"content" event type (issue #2).
    """

    @pytest.fixture
    def provider(self):
        return CodexProvider()

    def test_agent_message_item_is_parsed(self, provider):
        """The advertised happy path: reply nested in item.completed."""
        stdout = _codex_stream(
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            _agent_message('[{"task":"ship","verdict":"ready"}]'),
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship", "verdict": "ready"}]

    def test_reasoning_item_text_never_leaks(self, provider):
        """`reasoning` items also carry `text` — they must not reach the payload."""
        stdout = _codex_stream(
            {"type": "item.completed",
             "item": {"id": "rs_1", "type": "reasoning", "text": "SECRET_CHAIN_OF_THOUGHT"}},
            _agent_message('[{"task":"ship"}]', item_id="item_2"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]
        assert "SECRET_CHAIN_OF_THOUGHT" not in json.dumps(result["data"])

    def test_last_agent_message_wins_over_prose(self, provider):
        """Prose then JSON: take the final message, don't concatenate."""
        stdout = _codex_stream(
            _agent_message("Here is my analysis of the tasks.", item_id="item_1"),
            _agent_message('[{"task":"ship"}]', item_id="item_2"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_streaming_item_events_are_ignored(self, provider):
        """item.started/item.updated are deltas; only item.completed counts."""
        stdout = _codex_stream(
            {"type": "item.started",
             "item": {"id": "item_1", "type": "agent_message", "text": '[{"a":'}},
            {"type": "item.updated",
             "item": {"id": "item_1", "type": "agent_message", "text": '[{"a":1}]'}},
            _agent_message('[{"a":1}]'),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"a": 1}]

    def test_benign_item_error_does_not_fail_a_good_run(self, provider):
        """Nested item errors are routinely non-fatal (unknown model metadata)."""
        stdout = _codex_stream(
            {"type": "item.completed",
             "item": {"id": "item_0", "type": "error",
                      "message": "Model metadata for `x` not found. Defaulting to fallback metadata"}},
            _agent_message('[{"task":"ship"}]', item_id="item_1"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_top_level_error_is_surfaced(self, provider):
        """A quota failure must not be reported as 'No text events found'."""
        stdout = _codex_stream(
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "error", "message": "Quota exceeded. Check your plan and billing details."},
            {"type": "turn.failed",
             "error": {"message": "Quota exceeded. Check your plan and billing details."}},
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "Quota exceeded" in result["error"]

    def test_turn_failed_alone_is_surfaced(self, provider):
        stdout = _codex_stream({"type": "turn.failed", "error": {"message": "upstream 503"}})

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "upstream 503" in result["error"]

    def test_benign_item_error_surfaced_when_nothing_else_parsed(self, provider):
        """With no reply at all, the nested error is better than a generic message."""
        stdout = _codex_stream(
            {"type": "item.completed",
             "item": {"id": "item_0", "type": "error", "message": "sandbox denied"}},
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "sandbox denied" in result["error"]

    def test_no_events_reports_no_text(self, provider):
        stdout = _codex_stream(
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.completed", "usage": {}},
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert result["error"] == "No text events found in output"
        assert result["data"] is None

    def test_empty_agent_message_reports_empty(self, provider):
        result = provider.parse_output(_codex_stream(_agent_message("   ")), "")

        assert result["success"] is False
        assert result["error"] == "Empty text response"

    def test_malformed_and_non_dict_lines_are_skipped(self, provider):
        stdout = "\n".join([
            "not json at all",
            json.dumps(["a", "list", "not", "an", "object"]),
            json.dumps(_agent_message('[{"task":"ship"}]')),
        ])

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_prose_wrapped_json_is_extracted(self, provider):
        """Non-JSON-leading replies still go through the extractor."""
        stdout = _codex_stream(
            _agent_message('Sure! Here you go:\n```json\n[{"task":"ship"}]\n```'))

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_falls_back_to_joined_messages(self, provider):
        """If the final message alone yields nothing, try all of them joined."""
        stdout = _codex_stream(
            _agent_message('[{"task":"ship"}]', item_id="item_1"),
            _agent_message("Let me know if you need anything else.", item_id="item_2"),
        )

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"task": "ship"}]

    def test_legacy_text_events_still_parse(self, provider):
        """Pre-dotted-namespace shapes remain supported as a fallback."""
        stdout = _codex_stream({"type": "text", "part": {"type": "text", "text": '[{"a":1}]'}})

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [{"a": 1}]

    def test_agent_message_beats_legacy_events(self, provider):
        stdout = _codex_stream(
            {"type": "text", "text": '[{"from":"legacy"}]'},
            _agent_message('[{"from":"agent_message"}]'),
        )

        result = provider.parse_output(stdout, "")

        assert result["data"] == [{"from": "agent_message"}]


class TestCodexParseRealCaptures:
    """Regression tests against streams captured from a real codex CLI.

    Both fixtures are verbatim `codex exec --json` output from codex-cli
    0.144.0. They are the streams the pre-fix parser rejected outright.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "codex"

    @pytest.fixture
    def provider(self):
        return CodexProvider()

    def test_real_single_message_capture(self, provider):
        """gpt-5.6-sol at xhigh effort — the model this repo ships by default."""
        stdout = (self.FIXTURES / "gpt5_sol_single_message.jsonl").read_text(encoding="utf-8")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert result["data"], "expected a non-empty findings array"
        assert all("severity" in item for item in result["data"])

    def test_real_agentic_capture_excludes_reasoning(self, provider):
        """Agentic run with real chain-of-thought and two agent messages."""
        stdout = (self.FIXTURES / "agentic_reasoning_multi_message.jsonl").read_text(encoding="utf-8")

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [
            {"entry": "directory listing blocked by sandbox"},
            {"entry": "directory listing blocked by sandbox"},
            {"entry": "directory listing blocked by sandbox"},
        ]

    def test_real_agentic_capture_leaks_no_reasoning_text(self, provider):
        """Every reasoning item's text must be absent from the parsed payload."""
        stdout = (self.FIXTURES / "agentic_reasoning_multi_message.jsonl").read_text(encoding="utf-8")
        reasoning = [
            json.loads(line)["item"]["text"]
            for line in stdout.splitlines()
            if line.strip() and json.loads(line).get("type") == "item.completed"
            and json.loads(line)["item"].get("type") == "reasoning"
        ]
        assert reasoning, "fixture must contain reasoning items to be meaningful"

        payload = json.dumps(provider.parse_output(stdout, "")["data"])

        for text in reasoning:
            assert text[:40] not in payload


class TestProviderBinaryNotFound:
    """Tests for provider availability when binaries are not found."""

    @patch("shutil.which", return_value=None)
    def test_provider_binary_not_found(self, mock_which):
        """Verify is_available() returns False when binary not in PATH.

        This tests all providers to ensure they properly report unavailability
        when their respective CLI tools are not installed.
        """
        providers = [
            ClaudeCodeProvider(),
            CursorAgentProvider(),
            GeminiProvider(),
            KiloCodeProvider(),
            OpenCodeProvider(),
        ]

        for provider in providers:
            assert provider.is_available() is False, (
                f"{provider.name} should report unavailable when binary not found"
            )

    @patch("shutil.which")
    def test_multiple_providers_availability(self, mock_which):
        """Test mixed availability across providers."""
        # Simulate: cursor-agent available, gemini not, opencode available
        def which_side_effect(binary):
            available = {"cursor-agent": "/usr/bin/cursor-agent", "opencode": "/usr/bin/opencode"}
            return available.get(binary)

        mock_which.side_effect = which_side_effect

        cursor = CursorAgentProvider()
        gemini = GeminiProvider()
        opencode = OpenCodeProvider()

        assert cursor.is_available() is True
        assert gemini.is_available() is False
        assert opencode.is_available() is True


class TestProviderEdgeCases:
    """Edge case tests for all providers."""

    def test_cursor_agent_whitespace_handling(self):
        """Test cursor-agent handles whitespace in output."""
        provider = CursorAgentProvider()
        wrapper = {"type": "result", "result": "  \n  [1, 2, 3]  \n  "}
        stdout = f"  \n  {json.dumps(wrapper)}  \n  "

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_gemini_nested_json_in_response(self):
        """Test Gemini handles deeply nested JSON."""
        provider = GeminiProvider()
        nested_data = {
            "level1": {
                "level2": {
                    "level3": [{"deep": True}]
                }
            }
        }
        wrapper = {"session_id": "s1", "response": json.dumps(nested_data), "stats": {}}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == nested_data

    def test_opencode_large_text_concatenation(self):
        """Test OpenCode handles many text events."""
        provider = OpenCodeProvider()
        # Simulate chunked JSON output across many events
        json_str = json.dumps([{"i": i} for i in range(100)])
        chunk_size = 50
        chunks = [json_str[i:i+chunk_size] for i in range(0, len(json_str), chunk_size)]

        events = [{"type": "text", "part": {"type": "text", "text": chunk}} for chunk in chunks]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert len(result["data"]) == 100

    def test_cursor_agent_handles_wrapper_without_type(self):
        """Test cursor-agent handles JSON wrapper missing 'type' field."""
        provider = CursorAgentProvider()
        # Direct JSON object without "type":"result" wrapper
        data = {"result": [1, 2, 3]}
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        # Should parse the inner result
        assert result["data"] == [1, 2, 3]

    def test_gemini_handles_response_as_dict(self):
        """Test Gemini when response field is already a dict (not string)."""
        provider = GeminiProvider()
        # If response is somehow already parsed (edge case)
        wrapper = {"session_id": "s1", "response": {"key": "value"}, "stats": {}}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == {"key": "value"}

    def test_opencode_handles_non_text_part_type(self):
        """Test OpenCode ignores events with non-text part type."""
        provider = OpenCodeProvider()
        events = [
            {"type": "text", "part": {"type": "image", "data": "base64..."}},
            {"type": "text", "part": {"type": "text", "text": "[1, 2]"}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2]


class TestProviderInvocationErrors:
    """Tests for invoke_with_provider() error handling.

    These tests verify the critical orchestration behavior including timeouts,
    subprocess failures, and structured error output by mocking subprocess.run.
    """

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock for subprocess.run."""
        with patch("subprocess.run") as mock:
            yield mock

    @pytest.fixture
    def mock_provider_available(self):
        """Mock shutil.which to make providers appear available."""
        with patch("shutil.which", return_value="/usr/bin/mock-binary"):
            yield

    def test_timeout_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Test that subprocess.TimeoutExpired returns TIMEOUT error code.

        When a provider command times out, invoke_with_provider should return
        a structured error response with error_code='TIMEOUT'.
        """
        from utils.llm_client import invoke_with_provider, ERROR_TIMEOUT
        import subprocess

        # Simulate timeout - TimeoutExpired requires cmd and timeout args,
        # stdout/stderr are set as attributes
        exc = subprocess.TimeoutExpired(
            cmd=["cursor-agent", "--print", "test"],
            timeout=600
        )
        exc.stdout = "partial output"
        exc.stderr = "timeout stderr"
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4",
            timeout=600
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert "timed out" in result["error"].lower()
        assert result["details"]["timeout"] == 600
        assert result["details"]["provider"] == "cursor-agent"
        assert result["details"]["model"] == "gpt-4"

    def test_timeout_with_none_output(self, mock_subprocess, mock_provider_available):
        """Test timeout handling when stdout/stderr are None."""
        from utils.llm_client import invoke_with_provider, ERROR_TIMEOUT
        import subprocess

        exc = subprocess.TimeoutExpired(
            cmd=["gemini", "test"],
            timeout=900
        )
        exc.stdout = None
        exc.stderr = None
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="gemini:gemini-pro",
            timeout=900
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert result["details"]["provider"] == "gemini"

    def test_timeout_with_bytes_output(self, mock_subprocess, mock_provider_available):
        """Test timeout handling when stdout/stderr are bytes."""
        from utils.llm_client import invoke_with_provider, ERROR_TIMEOUT
        import subprocess

        exc = subprocess.TimeoutExpired(
            cmd=["opencode", "run"],
            timeout=600
        )
        exc.stdout = b"bytes output"
        exc.stderr = b"bytes error"
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="opencode:claude-3"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT

    def test_subprocess_failed_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Test that non-zero exit code returns SUBPROCESS_FAILED error code.

        When a provider command fails with a non-zero exit code, invoke_with_provider
        should return a structured error response with error_code='SUBPROCESS_FAILED'.
        """
        from utils.llm_client import invoke_with_provider, ERROR_SUBPROCESS_FAILED
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "error output"
        mock_result.stderr = "command failed: invalid arguments"
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "exited with code 1" in result["error"]
        assert result["details"]["exit_code"] == 1
        assert result["details"]["stderr"] == "command failed: invalid arguments"
        assert result["details"]["provider"] == "cursor-agent"

    def test_subprocess_failed_various_exit_codes(self, mock_subprocess, mock_provider_available):
        """Test SUBPROCESS_FAILED with various exit codes."""
        from utils.llm_client import invoke_with_provider, ERROR_SUBPROCESS_FAILED
        from unittest.mock import MagicMock

        for exit_code in [1, 2, 127, 255]:
            mock_result = MagicMock()
            mock_result.returncode = exit_code
            mock_result.stdout = ""
            mock_result.stderr = f"exit code {exit_code}"
            mock_subprocess.return_value = mock_result

            result = invoke_with_provider(
                prompt="Test prompt",
                model_spec="gemini:gemini-pro"
            )

            assert result["success"] is False
            assert result["error_code"] == ERROR_SUBPROCESS_FAILED
            assert result["details"]["exit_code"] == exit_code

    def test_binary_not_found_returns_error_code(self):
        """Test that unavailable binary returns BINARY_NOT_FOUND error code.

        When a provider's CLI tool is not found in PATH, invoke_with_provider
        should return a structured error response with error_code='BINARY_NOT_FOUND'.
        """
        from utils.llm_client import invoke_with_provider, ERROR_BINARY_NOT_FOUND

        # Mock shutil.which to return None (binary not found)
        with patch("shutil.which", return_value=None):
            result = invoke_with_provider(
                prompt="Test prompt",
                model_spec="cursor-agent:gpt-4"
            )

            assert result["success"] is False
            assert result["error_code"] == ERROR_BINARY_NOT_FOUND
            assert "not found" in result["error"].lower()
            assert result["details"]["provider"] == "cursor-agent"
            assert result["details"]["model"] == "gpt-4"

    def test_unknown_provider_returns_binary_not_found(self):
        """Test that unknown provider returns BINARY_NOT_FOUND error code."""
        from utils.llm_client import invoke_with_provider, ERROR_BINARY_NOT_FOUND

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="nonexistent-provider:model"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_BINARY_NOT_FOUND
        assert "nonexistent-provider" in result["error"]

    def test_parse_error_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Test that parse failures return PARSE_ERROR error code.

        When output cannot be parsed as valid JSON, invoke_with_provider
        should return a structured error response with error_code='PARSE_ERROR'.
        """
        from utils.llm_client import invoke_with_provider, ERROR_PARSE_ERROR
        from unittest.mock import MagicMock

        # Return valid process result but unparseable output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is not JSON at all, just plain text without any brackets"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_PARSE_ERROR
        assert "error" in result

    def test_successful_invocation(self, mock_subprocess, mock_provider_available):
        """Test successful invocation returns data and details."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        # Return valid cursor-agent JSON wrapper format
        inner_data = [{"title": "Test", "importance": "high"}]
        wrapper = {"type": "result", "result": json.dumps(inner_data)}

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(wrapper)
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
        )

        assert result["success"] is True
        assert result["data"] == inner_data
        assert result["details"]["provider"] == "cursor-agent"
        assert result["details"]["model"] == "gpt-4"
        assert "duration_seconds" in result["details"]

    def test_invocation_uses_provider_timeout(self, mock_subprocess, mock_provider_available):
        """Test that invocation uses provider's default timeout when not specified."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
            # No timeout specified - should use provider default
        )

        # Verify subprocess.run was called with correct timeout
        mock_subprocess.assert_called_once()
        call_kwargs = mock_subprocess.call_args
        # cursor-agent default timeout is 1200
        assert call_kwargs.kwargs["timeout"] == 1200

    def test_invocation_uses_custom_timeout(self, mock_subprocess, mock_provider_available):
        """Test that invocation uses custom timeout when specified."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4",
            timeout=300  # Custom timeout
        )

        mock_subprocess.assert_called_once()
        call_kwargs = mock_subprocess.call_args
        assert call_kwargs.kwargs["timeout"] == 300

    def test_duration_tracking(self, mock_subprocess, mock_provider_available):
        """Test that invocation tracks duration in details."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
        )

        assert result["success"] is True
        assert "duration_seconds" in result["details"]
        assert isinstance(result["details"]["duration_seconds"], float)
        assert result["details"]["duration_seconds"] >= 0


class TestClaudeCodeProviderIntegration:
    """Integration tests for Claude Code provider with invoke_with_provider().

    These tests verify the end-to-end flow of invoking Claude Code CLI
    through the LLM client, using mocked subprocess calls.
    """

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock for subprocess.run."""
        with patch("subprocess.run") as mock:
            yield mock

    @pytest.fixture
    def mock_claude_available(self):
        """Mock shutil.which to make claude CLI appear available."""
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            yield

    def test_claude_code_successful_invocation(self, mock_subprocess, mock_claude_available):
        """Test successful Claude Code invocation returns parsed data."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        # Simulate Claude Code JSON output format
        inner_data = [{"title": "Test finding", "importance": "high"}]
        wrapper = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps(inner_data),
            "session_id": "test-session-123",
            "total_cost_usd": 0.05
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(wrapper)
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="claude-code:sonnet"
        )

        assert result["success"] is True
        assert result["data"] == inner_data
        assert result["details"]["provider"] == "claude-code"
        assert result["details"]["model"] == "sonnet"

    def test_claude_code_with_code_block_response(self, mock_subprocess, mock_claude_available):
        """Test Claude Code invocation with markdown code block in response."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        inner_data = {"status": "ok", "items": [1, 2, 3]}
        response_text = f"""Here's the analysis:

```json
{json.dumps(inner_data, indent=2)}
```

Done."""
        wrapper = {
            "type": "result",
            "result": response_text,
            "session_id": "session-456"
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(wrapper)
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="claude-code:opus"
        )

        assert result["success"] is True
        assert result["data"] == inner_data
        assert result["details"]["provider"] == "claude-code"
        assert result["details"]["model"] == "opus"

    def test_claude_code_command_construction(self, mock_subprocess, mock_claude_available):
        """Test that Claude Code command is constructed correctly."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Review this code",
            model_spec="claude-code:haiku"
        )

        # Verify the command was constructed correctly
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]  # First positional arg is the command list

        # cmd[0] is which-resolved to the absolute path before launch
        assert cmd[0] == "/usr/local/bin/claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--model" in cmd
        assert "haiku" in cmd
        assert "Review this code" in cmd

    def test_claude_code_timeout_handling(self, mock_subprocess, mock_claude_available):
        """Test Claude Code timeout returns proper error."""
        from utils.llm_client import invoke_with_provider, ERROR_TIMEOUT
        import subprocess

        exc = subprocess.TimeoutExpired(
            cmd=["claude", "-p", "test"],
            timeout=600
        )
        exc.stdout = "partial"
        exc.stderr = ""
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="claude-code:sonnet",
            timeout=600
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert result["details"]["provider"] == "claude-code"

    def test_claude_code_subprocess_failure(self, mock_subprocess, mock_claude_available):
        """Test Claude Code subprocess failure returns proper error."""
        from utils.llm_client import invoke_with_provider, ERROR_SUBPROCESS_FAILED
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Invalid API key"
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="claude-code:sonnet"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert result["details"]["provider"] == "claude-code"
        assert result["details"]["exit_code"] == 1

    def test_claude_code_binary_not_found(self):
        """Test Claude Code returns error when CLI not found."""
        from utils.llm_client import invoke_with_provider, ERROR_BINARY_NOT_FOUND

        with patch("shutil.which", return_value=None):
            result = invoke_with_provider(
                prompt="Test prompt",
                model_spec="claude-code:sonnet"
            )

        assert result["success"] is False
        assert result["error_code"] == ERROR_BINARY_NOT_FOUND
        assert result["details"]["provider"] == "claude-code"

    def test_claude_code_parse_error(self, mock_subprocess, mock_claude_available):
        """Test Claude Code parse error when response has no JSON."""
        from utils.llm_client import invoke_with_provider, ERROR_PARSE_ERROR
        from unittest.mock import MagicMock

        # Return wrapper with non-JSON result
        wrapper = {
            "type": "result",
            "result": "Just plain text with no JSON anywhere"
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(wrapper)
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="claude-code:sonnet"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_PARSE_ERROR

    def test_claude_code_uses_default_timeout(self, mock_subprocess, mock_claude_available):
        """Test that Claude Code uses 1800s default timeout."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test prompt",
            model_spec="claude-code:sonnet"
            # No timeout specified - should use provider default
        )

        mock_subprocess.assert_called_once()
        call_kwargs = mock_subprocess.call_args
        assert call_kwargs.kwargs["timeout"] == 1800


# ===========================================================================
# L5 (§9.6): agy's `--print-timeout` is derived, not hardcoded
# ===========================================================================
#
# Today the flag is the literal `20m`, which is three numbers pretending to be
# one: `agy.py`'s `20m`, `providers.yaml`'s `default_timeout: 1200` and the
# adapter's own `default_timeout`. The live symptom is two-sided. Configured
# ABOVE 20 minutes the flag is a silent no-op — agy self-terminates at 1200 s
# and returns its own `status: ERROR`, so the user diagnoses agy rather than
# their configuration. Configured BELOW it, the parent's `subprocess.run`
# deadline fires while agy is still alive, and the leader-only kill leaves a
# descendant behind.
#
# The fix is one keyword-only `timeout_s=` on the four command builders,
# threaded from the `effective_timeout` `invoke_with_provider` already
# computes, rendered as whole seconds minus a deliberate shutdown margin.


def parse_go_duration(text):
    """Seconds in a Go duration string, or ``None`` when it is not one.

    Deliberately narrow: only the spellings this adapter may emit
    (``1170s``, ``19m30s``, ``1h0m0s``). `time.ParseDuration` accepts the
    bare-seconds form, which is why §9.6.c renders whole seconds and removes
    the minute-rounding rule instead of specifying one — rounding to minutes
    is exactly what let 1200 s and `20m` become an exact tie.
    """
    import re

    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", str(text))
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


class TestParseGoDurationHelper:
    """The ladder invariant is only as good as the parser it reads with."""

    def test_bare_seconds(self):
        assert parse_go_duration("1170s") == 1170

    def test_minutes_and_seconds(self):
        assert parse_go_duration("19m30s") == 1170

    def test_minutes_only(self):
        assert parse_go_duration("20m") == 1200

    def test_non_duration_is_none(self):
        assert parse_go_duration("soon") is None
        assert parse_go_duration("") is None


class TestAgyPrintTimeoutDerivation:
    """§9.6.e's acceptance table, above and below 20 minutes."""

    PROMPT = "Analyze this"
    MODEL = "Gemini 3.1 Pro (High)"

    @pytest.fixture
    def provider(self):
        return AgyProvider()

    @staticmethod
    def _print_timeout(cmd):
        assert "--print-timeout" in cmd, f"no --print-timeout in {cmd}"
        return cmd[cmd.index("--print-timeout") + 1]

    def test_the_constants_are_named_not_literals(self):
        """§9.6.c: the margin is a named constant, never a literal in the builder."""
        from utils.providers.agy import INNER_SHUTDOWN_MARGIN_S, MIN_INNER_TIMEOUT_S

        assert INNER_SHUTDOWN_MARGIN_S == 30
        assert MIN_INNER_TIMEOUT_S == 60

    @pytest.mark.parametrize("configured,expected", [
        (600, "570s"),
        (1200, "1170s"),
        # ABOVE 20 minutes — the silent no-op. Today this renders `20m`, agy
        # self-terminates at 1200 s and blames itself.
        (2400, "2370s"),
    ])
    def test_derived_from_the_resolved_timeout(self, provider, configured, expected):
        cmd = provider.build_command(self.PROMPT, self.MODEL, timeout_s=configured)

        assert self._print_timeout(cmd) == expected

    def test_below_the_margin_clamps_to_the_floor_and_warns_once(self, provider, capsys):
        """§9.6.c's misconfiguration guard: one WARN naming both numbers."""
        cmd = provider.build_command(self.PROMPT, self.MODEL, timeout_s=45)

        assert self._print_timeout(cmd) == "60s"
        captured = capsys.readouterr()
        assert captured.err.count("WARN") == 1, (
            f"expected exactly one WARN, got: {captured.err!r}")
        assert "45" in captured.err and "60" in captured.err

    # Every rung is at or above MIN_INNER_TIMEOUT_S + INNER_SHUTDOWN_MARGIN_S,
    # so none of them is clamped and the strict invariant must hold on all of
    # them. 90 is the exact boundary; 1200 is the value that ties today.
    LADDER = [90, 91, 120, 300, 600, 900, 1200, 1800, 2400, 3600, 7200]

    def test_the_margin_invariant_holds_over_the_whole_ladder(self, provider):
        from utils.providers.agy import INNER_SHUTDOWN_MARGIN_S

        for configured in self.LADDER:
            rendered = self._print_timeout(
                provider.build_command(self.PROMPT, self.MODEL, timeout_s=configured)
            )
            inner = parse_go_duration(rendered)
            assert inner is not None, (
                f"{rendered!r} is not a Go duration (configured={configured})")
            assert inner > 0, f"non-positive inner timeout for {configured}"
            assert inner + INNER_SHUTDOWN_MARGIN_S <= configured, (
                f"configured={configured} rendered {rendered!r}: agy would reach "
                f"its own cutoff at or after the parent's deadline, which is the "
                f"tie the margin exists to break")

    def test_no_configured_value_still_renders_the_old_literal(self, provider):
        """The hardcoded `20m` no longer decides a live invocation."""
        for configured in (600, 1200, 2400):
            rendered = self._print_timeout(
                provider.build_command(self.PROMPT, self.MODEL, timeout_s=configured)
            )
            assert rendered != "20m", (
                f"configured={configured} still renders the hardcoded literal")

    def test_the_three_numbers_reconcile_to_one(self, provider):
        """§9.6.d: `providers.yaml` and the adapter default agree at 1200."""
        import yaml

        base = Path(__file__).parent.parent / "providers.yaml"
        config = yaml.safe_load(base.read_text(encoding="utf-8"))

        assert config["providers"]["agy"]["default_timeout"] == 1200
        assert provider.default_timeout == 1200
        # And the derived flag is that same single number, minus the margin.
        rendered = self._print_timeout(
            provider.build_command(self.PROMPT, self.MODEL, timeout_s=provider.default_timeout)
        )
        assert rendered == "1170s"

    def test_the_keyword_survives_the_effort_suffix(self, provider):
        """L11 #3's `--effort` and L5's derived timeout are independent."""
        cmd = provider.build_command(
            self.PROMPT, "Gemini 3.1 Pro (High):high", timeout_s=2400
        )

        assert self._print_timeout(cmd) == "2370s"
        assert cmd[cmd.index("--effort") + 1] == "high"
        assert cmd[cmd.index("--model") + 1] == "Gemini 3.1 Pro (High)"

    def test_the_keyword_is_keyword_only(self, provider):
        """§9.6.a: positional passing must not be possible (M7 orders after L5)."""
        with pytest.raises(TypeError):
            provider.build_command(self.PROMPT, self.MODEL, 1200)

    def test_the_transport_builder_threads_it(self, provider):
        """§9.6.b's Lane A leg: `build_command_for_transport` forwards the value."""
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "argv", timeout_s=2400
        )

        assert stdin_text is None
        assert self._print_timeout(argv) == "2370s"


class TestInvokeWithProviderThreadsTheResolvedTimeout:
    """§9.6.b: the value comes from `invoke_with_provider`'s own computation."""

    @pytest.mark.parametrize("configured,expected", [
        (600, "570s"),
        (1200, "1170s"),
        (2400, "2370s"),
    ])
    def test_the_spawned_argv_carries_the_derived_flag(self, configured, expected):
        from unittest.mock import MagicMock
        from utils.llm_client import invoke_with_provider

        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = json.dumps({"status": "SUCCESS", "response": "[]"})
        completed.stderr = ""

        with patch("shutil.which", return_value="/usr/bin/agy"), \
                patch("subprocess.run", return_value=completed) as run:
            invoke_with_provider(
                prompt="Test prompt",
                model_spec="agy:Gemini 3.1 Pro (High)",
                timeout=configured,
            )

        argv = run.call_args.args[0]
        assert argv[argv.index("--print-timeout") + 1] == expected
        # The two deadlines are the same number, one margin apart — never a tie.
        assert run.call_args.kwargs["timeout"] == configured


# --- The off switch: `timeout_s=None` renders today's argv byte-for-byte ----
#
# Captured from the pre-L5 tree, before the keyword existed. This is the whole
# rollback story (§14.4): drop the one call site in `llm_client.py` and every
# adapter is back to exactly these bytes, agy included.
PRE_L5_ARGV_PROMPT = "Analyze this plan carefully"

PRE_L5_ARGV_MODELS = {
    "agy": "Gemini 3.1 Pro (High)",
    "aider": "openrouter/z-ai/glm-5.2",
    "claude-code": "sonnet",
    "cline": "openrouter/z-ai/glm-5.2",
    "codex": "gpt-5.5",
    "copilot": "openai/gpt-4.1-mini",
    "cursor-agent": "auto",
    "gemini": "gemini-2.5-pro",
    "goose": "openrouter/z-ai/glm-5.2",
    "grok": "grok-4",
    "kilocode": "openai-native/gpt-5.5",
    "kimi": "moonshotai/kimi-k2-thinking",
    "muse": "muse-spark-1.2",
    # The real default SELECTOR (providers.yaml), not a placeholder: omp
    # matches on `selector` (openrouter/openai/gpt-5.6-luna) and not on `id`
    # (openai/gpt-5.6-luna), and the row is what keeps the difference rendered.
    "omp": "openrouter/openai/gpt-5.6-luna",
    "opencode": "openai/gpt-5.5",
    "pi": "anthropic/claude-opus-4-8",
    "qwen-code": "qwen/qwen3.8-max",
    # vibe's entry is a SEED ALIAS, not a wire slug: it has no --model flag,
    # so this value never reaches argv at all -- it rides VIBE_ACTIVE_MODEL.
    # It is still the real default from providers.yaml, because the housekeeping
    # sweep renders argv from these and a dummy would exercise nothing.
    "vibe": "vibe-default",
}

PRE_L5_ARGV = {
    "agy": [
        "agy", "--new-project", "--dangerously-skip-permissions",
        "--output-format", "json", "--print-timeout", "20m",
        "--model", "Gemini 3.1 Pro (High)", "-p", PRE_L5_ARGV_PROMPT,
    ],
    "aider": [
        "aider", "--model", "openrouter/z-ai/glm-5.2",
        "--message", f"/ask {PRE_L5_ARGV_PROMPT}",
        # No `--yes-always` since plans/ask-answer-recovery-and-false-success.md
        # T4.2: the decline supply on stdin answers aider's prompts instead.
        "--no-auto-commits", "--no-pretty", "--no-stream",
        "--no-check-update", "--no-show-model-warnings", "--no-analytics",
        "--no-gitignore", "--no-show-release-notes", "--no-detect-urls",
        "--no-fancy-input",
        "--chat-history-file", os.devnull,
        "--input-history-file", os.devnull,
    ],
    # §17's write grant is part of the frozen argv as of 2026-08-20: the
    # generate prompt ends in "write your JSON output to this file: <path>"
    # (`supports_file_output = True`), and headless `claude -p` denied the
    # model's own Write without it. Measured on the credentialed live lane.
    "claude-code": [
        "claude", "-p", "--output-format", "json", "--model", "sonnet",
        "--permission-mode", "acceptEdits",
        PRE_L5_ARGV_PROMPT,
    ],
    "cline": [
        "cline", "--json", "-P", "openrouter", "-m", "z-ai/glm-5.2",
        PRE_L5_ARGV_PROMPT,
    ],
    "codex": [
        "codex", "exec", "--sandbox", "workspace-write", "--json",
        "-c", "features.plugins=false",
        "--ephemeral", "--skip-git-repo-check", "--model", "gpt-5.5",
        PRE_L5_ARGV_PROMPT,
    ],
    # No permission flag and no `--add-dir`: §6.5 tier 1, measured sufficient
    # with a clean canary at all four rungs. This entry is what keeps the help
    # text's false "--allow-all-tools is required for non-interactive mode" from
    # being "fixed" back into the argv by a later edit.
    "copilot": [
        "copilot", "--output-format", "json", "--stream", "off",
        "--log-level", "none", "--no-custom-instructions", "--no-ask-user",
        "--no-auto-update", "--disable-builtin-mcps",
        "--model", "openai/gpt-4.1-mini", "-p", PRE_L5_ARGV_PROMPT,
    ],
    "cursor-agent": [
        "cursor-agent", "--print", "--trust", "--mode", "ask",
        "--output-format", "json", "--model", "auto", PRE_L5_ARGV_PROMPT,
    ],
    "gemini": [
        "gemini", "--output-format", "json", "--model", "gemini-2.5-pro",
        PRE_L5_ARGV_PROMPT,
    ],
    "goose": [
        "goose", "run", "--no-session", "-q", "--output-format", "json",
        "--no-profile", "--max-turns", "25", "--with-builtin", "developer",
        "--provider", "openrouter", "--model", "z-ai/glm-5.2",
        "-t", PRE_L5_ARGV_PROMPT,
    ],
    # §9.4 moved grok's `-p <prompt>` pair to the END (it was at index 4). That
    # is an independent normalization, NOT the timeout keyword: `timeout_s` is
    # still inert here, and the rollback story below is unaffected — dropping the
    # keyword's call site returns grok to exactly this argv.
    "grok": [
        "grok", "--no-auto-update", "--always-approve",
        "--output-format", "json", "-m", "grok-4", "-p", PRE_L5_ARGV_PROMPT,
    ],
    "kilocode": [
        "kilocode", "run", "--auto", "-m", "openai-native/gpt-5.5",
        PRE_L5_ARGV_PROMPT,
    ],
    # No model token: kimi selects it through KIMI_MODEL_NAME, and `-m <slug>`
    # is a hard failure ("Model ... is not configured in config.toml"). This row
    # is what keeps that measured fact from being "fixed" back into argv.
    "kimi": [
        "kimi", "--output-format", "stream-json", "-p", PRE_L5_ARGV_PROMPT,
    ],
    "muse": [
        "muse", "exec", "--json", "--disable-approval", "--disable-sandbox",
        "--model", "muse-spark-1.2", PRE_L5_ARGV_PROMPT,
    ],
    # Two values in this row are NOT literals and cannot be: the overlay lives
    # beside the skill and `--cwd` is the spawn directory. Both are still
    # assertions — `--config` must be the absolute overlay THIS build ships, and
    # `--cwd` must be rendered on every dispatch (without it, launched from
    # $HOME omp silently relocates to /tmp and reviews an empty tree) — so they
    # are taken from the adapter's own constant and from `os.getcwd()`, read
    # here at import for the same reason the argv reads it at build time.
    # Everything else is policy and is spelled out: no `--thinking` and no
    # `--max-time` (the model carries no `:effort` and `timeout_s` is None), the
    # approval mode and the six-name allow-list verbatim, and the prompt last
    # behind `--`.
    "omp": [
        "omp", "--no-session", "-p", "--mode", "json",
        "--config", str(OMP_OVERLAY_PATH),
        "--approval-mode", "yolo",
        "--tools", "read,grep,glob,bash,edit,write",
        "--cwd", os.getcwd(),
        "--model", "openrouter/openai/gpt-5.6-luna",
        "--", PRE_L5_ARGV_PROMPT,
    ],
    "opencode": [
        "opencode", "run", "--format", "json", "--print-logs",
        "--log-level", "ERROR", "--model", "openai/gpt-5.5",
        PRE_L5_ARGV_PROMPT,
    ],
    # `--mode json` since plan §7 (T7.5): the JSONL event stream parse_output
    # reads, rendered right after `-p` like omp's.
    "pi": [
        "pi", "--no-session", "-p", "--mode", "json",
        "--model", "anthropic/claude-opus-4-8",
        PRE_L5_ARGV_PROMPT,
    ],
    # No --max-wall-time element at all: that flag is the ONLY row-derived value
    # qwen renders, and `timeout_s=None` must emit nothing rather than a default
    # (§1's guard, and what keeps expected_argv.yaml free of a "sometimes
    # present" case). The prompt is a bare trailing positional -- `-p` is
    # deprecated at 0.21.13.
    "qwen-code": [
        "qwen", "--bare", "--chat-recording", "false", "-o", "json",
        "--max-tool-calls", "0", "--max-session-turns", "25",
        "-m", "qwen/qwen3.8-max", PRE_L5_ARGV_PROMPT,
    ],
    # Note what is NOT here: the model. vibe has no `--model` flag, so the alias
    # rides VIBE_ACTIVE_MODEL and the argv is model-independent -- which is also
    # why vibe needs no normalize_argv.py entry (no model token ever reaches the
    # call log). `--max-price` and `--max-turns` are POLICY constants and are
    # pinned verbatim; `--agent plan` is the §6.5 rung, and the bare form it
    # replaces is DESTRUCTIVE, not tier 1.
    "vibe": [
        "vibe", "--output", "json", "--max-price", "0.25",
        "--max-turns", "4", "--agent", "plan", "-p", PRE_L5_ARGV_PROMPT,
    ],
}


class TestTimeoutKeywordIsInertWhenUnset:
    """`timeout_s=None` is the seam; assert it rather than assume it."""

    @pytest.fixture(autouse=True)
    def ungated(self):
        """Render argv from the adapters, never from the developer's PATH.

        Two rows in the table are gated: omp's §2.11 probe and cursor's
        `--trust`/`--mode` one both read the *installed* binary's `--help`
        before `build_command` will emit anything. Ungated, a machine carrying
        an older omp raises `OmpFlagsUnconfirmed` here instead of comparing
        argv -- a machine-dependent failure in the one sweep whose whole point
        is that these bytes are machine-independent.
        `TestOmpProvider.ungated` neutralizes exactly this lookup for exactly
        this reason, and the shared gate resolves its binary through
        `LLMProvider.gate_binary`, so neutralizing that one lookup is enough:
        with no binary to read, `missing_flags` reports nothing missing and the
        argv is the one the adapter would render. pi's optional `--tools` gate
        (T3.1) reads the same lookup, so its row is the argv WITHOUT the flag;
        the argv with it is pinned in `TestPiToolAllowlist`. Nothing else in the
        table consults the gate from `build_command` (claude's two optional-flag
        reads are on the `schema`/fallback paths this sweep does not take), and
        both gates are graded on their own, with the help text stubbed rather
        than the lookup.
        """
        with patch.object(LLMProvider, "gate_binary", lambda self, **_kw: None):
            yield

    def test_the_table_covers_every_registered_adapter(self):
        assert set(PRE_L5_ARGV) == set(_registered_provider_names())
        assert len(PRE_L5_ARGV) == 18

    @pytest.mark.parametrize("name", sorted(PRE_L5_ARGV))
    def test_explicit_none_renders_the_pre_l5_argv(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        argv = provider.build_command(
            PRE_L5_ARGV_PROMPT, PRE_L5_ARGV_MODELS[name], timeout_s=None
        )

        assert argv == PRE_L5_ARGV[name]

    @pytest.mark.parametrize("name", sorted(PRE_L5_ARGV))
    def test_omitting_the_keyword_is_the_same_as_passing_none(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        model = PRE_L5_ARGV_MODELS[name]

        assert provider.build_command(PRE_L5_ARGV_PROMPT, model) == provider.build_command(
            PRE_L5_ARGV_PROMPT, model, timeout_s=None
        )

    @pytest.mark.parametrize("name", sorted(PRE_L5_ARGV))
    def test_the_other_three_builders_are_inert_too(self, name):
        """The keyword lands on all four builders and changes none of them."""
        from utils.provider_registry import _PROVIDERS
        from utils.providers.base import PROFILE_GENERATE

        provider = _PROVIDERS[name]
        model = PRE_L5_ARGV_MODELS[name]

        for transport in ("argv", "stdin"):
            if transport != "argv" and not provider.supports_prompt_transport(transport):
                continue
            assert provider.build_command_for_transport(
                PRE_L5_ARGV_PROMPT, model, transport, timeout_s=None
            ) == provider.build_command_for_transport(
                PRE_L5_ARGV_PROMPT, model, transport
            )

        for profile in provider.agent_profiles:
            if profile == PROFILE_GENERATE:
                continue
            assert provider.build_agent_command(
                PRE_L5_ARGV_PROMPT, model, profile, timeout_s=None
            ) == provider.build_agent_command(PRE_L5_ARGV_PROMPT, model, profile)
            for transport in ("argv", "stdin"):
                if transport != "argv" and not provider.supports_prompt_transport(transport):
                    continue
                assert provider.build_agent_command_for_transport(
                    PRE_L5_ARGV_PROMPT, model, profile, transport, timeout_s=None
                ) == provider.build_agent_command_for_transport(
                    PRE_L5_ARGV_PROMPT, model, profile, transport
                )


# --- No child survives the outer timeout — two tests, honestly split -------

MOCK_LLM_PATH = Path(__file__).parent / "mocks" / "mock_llm.py"


def _pid_is_alive(pid):
    """Whether ``pid`` still names a live process (POSIX)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture
def agy_stub(tmp_path, monkeypatch):
    """A real `agy` on PATH, backed by ``mock_llm.py``'s print-timeout stubs."""
    import stat

    launcher = tmp_path / "agy"
    launcher.symlink_to(MOCK_LLM_PATH)
    MOCK_LLM_PATH.chmod(MOCK_LLM_PATH.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MOCK_LLM_PID_FILE", str(tmp_path / "sleeper.pid"))
    return launcher


def _await_pid_file(path, deadline_s=10.0):
    import time as _time

    end = _time.time() + deadline_s
    while _time.time() < end:
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            text = ""
        if text:
            return int(text)
        _time.sleep(0.05)
    raise AssertionError(f"the stub never recorded a pid at {path}")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pid liveness + symlink")
class TestNoChildOutlivesTheOuterDeadline:
    """§9.6.e's survivor check.

    Both stubs run against real processes with the margin scaled down, because
    a test cannot sleep for 20 minutes: the *mechanism* is what is under test,
    and `INNER_SHUTDOWN_MARGIN_S` / `MIN_INNER_TIMEOUT_S` are its parameters.
    """

    MODEL = "Gemini 3.1 Pro (High)"

    @pytest.fixture
    def scaled_margin(self, monkeypatch):
        from utils.providers import agy as agy_module

        monkeypatch.setattr(agy_module, "INNER_SHUTDOWN_MARGIN_S", 1)
        monkeypatch.setattr(agy_module, "MIN_INNER_TIMEOUT_S", 1)

    def test_a_stub_that_honours_the_flag_shuts_itself_down(
        self, agy_stub, scaled_margin, monkeypatch, tmp_path
    ):
        """The child exits on its own and the harness reads agy's own reason."""
        from utils.llm_client import ERROR_TIMEOUT, invoke_with_provider

        monkeypatch.setenv("MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR", "honour")

        with patch("shutil.which", return_value=str(agy_stub)):
            result = invoke_with_provider(
                prompt="Analyze this",
                model_spec=f"agy:{self.MODEL}",
                timeout=4,
            )

        assert result["success"] is False
        # agy's OWN failure reason, not a bare TIMEOUT the parent invented.
        assert result.get("error_code") != ERROR_TIMEOUT
        assert "print-timeout" in result["error"]
        # And nothing it started is still running.
        sleeper = _await_pid_file(tmp_path / "sleeper.pid")
        assert not _pid_is_alive(sleeper), (
            f"pid {sleeper} outlived the run despite the stub honouring its "
            f"own cutoff")

    # This was an `xfail(strict=False)`: the leader-only SIGKILL was the orphan
    # bug itself, and L5 cannot make a disobedient child reapable. The Lane A
    # spawn now snapshots the CLI's descendants before the kill and sweeps
    # every surviving process group (`process_control.terminate_timed_out`,
    # plan review-run-scope-salvage-and-failure-visibility D3.3), so the
    # sleeper is reaped and the test holds as an ordinary assertion.
    def test_a_stub_that_ignores_the_flag_leaves_no_descendant(
        self, agy_stub, scaled_margin, monkeypatch, tmp_path
    ):
        from utils.llm_client import ERROR_TIMEOUT, invoke_with_provider

        monkeypatch.setenv("MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR", "ignore")
        sleeper = None
        try:
            with patch("shutil.which", return_value=str(agy_stub)):
                result = invoke_with_provider(
                    prompt="Analyze this",
                    model_spec=f"agy:{self.MODEL}",
                    timeout=3,
                )

            assert result.get("error_code") == ERROR_TIMEOUT
            sleeper = _await_pid_file(tmp_path / "sleeper.pid")
            assert not _pid_is_alive(sleeper), (
                f"pid {sleeper} outlived the outer deadline — the leader-only "
                f"kill orphaned it")
        finally:
            if sleeper and _pid_is_alive(sleeper):
                try:
                    os.kill(sleeper, 9)
                except OSError:
                    pass


# =========================================================================== #
# C1 — `vendor_text`: the classifier reads what the VENDOR wrote
#
# `classify_failure` built its haystack from the WHOLE capture, and four shipped
# adapters echo our own prompt back inside their envelope. So every marker table
# in `classify_streams` was matching text we wrote: a green goose run classified
# `timeout` (the review prompt's example block contains the word), and green
# copilot and vibe runs classified `rate_limited` (the plan under review
# discusses rate limits). On a *failed* run that is live — `_classify_failed_run`
# forwards exactly `auth_failed` and `rate_limited`, the two classes that fail or
# back off the whole batch.
#
# The fix is a per-adapter reduction hook, `LLMProvider.vendor_text`, applied
# once inside `classify_failure`. Its default is identity, so every adapter that
# does not override it is byte-for-byte what it was.
# =========================================================================== #

C1_FIXTURES = Path(__file__).parent / "fixtures"

#: The whole committed goose corpus and its verdict on both sides of the fix.
#: **This tuple is C1's acceptance test**: fourteen rows, and exactly one cell
#: moves. `stdout` is `None` for the two rows the manifest declares
#: not-applicable (goose exits 0 on every measured failure, and its envelope is
#: never empty), which classify as `None` on empty streams either way. Every
#: goose `*.stderr.txt` is 0 bytes, so stderr is "" throughout.
C1_GOOSE_CORPUS = (
    # row,                stdout file,                    before,          after
    ("success",           "success.stdout.txt",           None,            None),
    ("truncated",         "truncated.stdout.txt",         None,            None),
    ("ephemeral-only",    "ephemeral-only.stdout.txt",    None,            None),
    ("multi-assistant",   "multi-assistant.stdout.txt",   None,            None),
    ("error-envelope",    "error-envelope.stdout.txt",    None,            None),
    ("exit-zero-failure", "exit-zero-failure.stdout.txt", "server_error",  "server_error"),
    ("stdout-error",      None,                           None,            None),
    ("empty",             None,                           None,            None),
    ("auth-error",        "auth-error.stdout.txt",        "auth_failed",   "auth_failed"),
    ("rate-limit",        "rate-limit.stdout.txt",        "rate_limited",  "rate_limited"),
    ("invalid-model",     "invalid-model.stdout.txt",     "invalid_model", "invalid_model"),
    # Review 24 pins #4: the shared table learned goose 1.53.0's `effective`.
    ("context-overflow",  "context-overflow.stdout.txt",  "context_length_exceeded", "context_length_exceeded"),
    ("zero-body",         "zero-body.stdout.txt",         None,            None),
    ("prompt-echo",       "prompt-echo.stdout.txt",       "timeout",       None),
)

#: The four adapters measured to echo our prompt, with the committed capture of a
#: run that SUCCEEDED and the verdict that capture drew before the fix. muse's
#: `None` is the "one plan word away" case: it echoes, nothing in its committed
#: prompt happens to match a table, and its own `auth_error_patterns` already
#: carry `api error 401` — which is why C1 lands before the auth-table widening.
C1_ECHOING_SUCCESS_CAPTURES = (
    ("goose", "prompt-echo.stdout.txt", "timeout"),
    ("copilot", "success.jsonl", "rate_limited"),
    ("vibe", "success-json.json", "rate_limited"),
    ("muse", "success.stdout.txt", None),
)


def _c1_capture(provider, name):
    return (C1_FIXTURES / provider / name).read_bytes().decode("utf-8")


def _c1_adapter(provider):
    from utils.provider_registry import _PROVIDERS

    return _PROVIDERS[provider]


def _c1_without_the_hook(monkeypatch, provider):
    """Put one adapter back on the pre-C1 haystack, in place."""
    monkeypatch.setattr(type(_c1_adapter(provider)), "vendor_text",
                        LLMProvider.vendor_text)


def _c1_goose_with_plan_sentence(sentence):
    """goose's real cut-stream capture, with one sentence added to OUR prompt.

    The bytes stay a real goose envelope: the sentence is appended to the echoed
    user turn, which is exactly where a plan discussing 401s or rate limits would
    have put it. Everything goose itself wrote is untouched.
    """
    envelope = json.loads(_c1_capture("goose", "truncated.stdout.txt"))
    for message in envelope["messages"]:
        if message.get("role") == "user":
            message["content"][0]["text"] += "\n\n" + sentence
            break
    return json.dumps(envelope, indent=2, ensure_ascii=False)


class TestVendorTextContract:
    """The hook itself: identity by default, applied in exactly one place."""

    def test_the_base_hook_is_identity(self):
        """An adapter that does not override it keeps today's bytes exactly."""
        stdout, stderr = LLMProvider.vendor_text(
            AiderProvider(), "some stdout", "some stderr"
        )
        assert (stdout, stderr) == ("some stdout", "some stderr")

    @pytest.mark.parametrize("provider", [
        "agy", "aider", "claude-code", "cline", "cursor-agent",
        "gemini", "grok", "kilocode", "kimi", "qwen-code",
    ])
    def test_every_non_echoing_adapter_still_gets_the_whole_capture(self, provider):
        """The ten adapters no committed capture shows echoing.

        "Non-echoing" means **echoes neither the prompt nor the repo**. opencode
        left this roster (T2.2): it does not echo our prompt, but its
        ``--format json`` stdout carries tool output, and a validator's ``grep``
        of a plan discussing rate limits was recorded ``rate_limited``. pi left
        it with ``--mode json`` (T7.4): its stream echoes our prompt, pi's own
        system prompt and every tool result, so ``PiProvider.vendor_text``
        reduces it (``tests/fixtures/pi/markers-in-tool-result``).

        The default being identity is what makes "fix the class for the adapters
        that echo" a bounded claim rather than a rewrite of every classifier.
        """
        text = "Ran into this error: 429 rate limit exceeded"
        assert _c1_adapter(provider).vendor_text(text, "stderr text") == (
            text, "stderr text"
        )

    def test_the_contract_is_documented_where_an_override_author_will_read_it(self):
        """Both rules that make a reduction safe are in the hook's own docstring."""
        doc = (LLMProvider.vendor_text.__doc__ or "").lower()
        assert "only remove" in doc, "the removal-only rule is not documented"
        assert "unchanged" in doc and "never" in doc, (
            "the degrade-to-its-input rule is not documented"
        )

    @staticmethod
    def _c1_base_source():
        """base.py's text, read from disk rather than through ``inspect``.

        ``inspect.getsource`` resolves a slice by the *imported* object's
        ``co_firstlineno`` against the *current* file, so an unrelated edit to
        this module while the suite runs makes it return the wrong lines. The
        source is parsed here instead, which has no such coupling.
        """
        import ast

        from utils.providers import base as base_module

        text = Path(base_module.__file__).read_text(encoding="utf-8")
        return text, ast.parse(text)

    @staticmethod
    def _c1_body_of(tree, name, cls=None):
        import ast

        scope = tree
        if cls is not None:
            scope = next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == cls
            )
        function = next(
            node for node in scope.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return ast.dump(ast.Module(body=function.body, type_ignores=[]))

    def test_classify_failure_applies_it(self):
        """One call, immediately before the rule runs."""
        _text, tree = self._c1_base_source()
        body = self._c1_body_of(tree, "classify_failure", cls="LLMProvider")
        assert "'vendor_text'" in body, (
            "classify_failure no longer applies the hook"
        )

    def test_nothing_else_applies_it(self):
        """§7.2 depends on `classify_streams` staying the adapter-free rule.

        Exactly two appliers (T2.5): `classify_failure`, to classify, and the
        executor's `_classify_failed_run`, to quote — never to classify.
        """
        import ast

        text, tree = self._c1_base_source()

        assert text.count("self.vendor_text(") == 1, (
            "the hook is applied in more than one place"
        )
        assert "'vendor_text'" not in self._c1_body_of(tree, "classify_streams")
        assert "'vendor_text'" not in self._c1_body_of(tree, "match_streams")
        assert "'vendor_text'" not in self._c1_body_of(
            tree, "classify_auth_probe", cls="LLMProvider"
        )

        utils_dir = Path(base_module.__file__).parent.parent
        executor_path = utils_dir / "subagent_executor.py"
        executor_text = executor_path.read_text(encoding="utf-8")
        assert executor_text.count("vendor_text(") == 1, (
            "subagent_executor.py applies the hook more than once"
        )
        executor_tree = ast.parse(executor_text)
        applier = next(
            node for node in executor_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_classify_failed_run"
        )
        assert "'vendor_text'" in ast.dump(applier), (
            "the executor's one application is not inside _classify_failed_run"
        )

        for path in sorted(utils_dir.rglob("*.py")):
            if path in (executor_path, Path(base_module.__file__)):
                continue
            calls = [
                line for line in path.read_text(encoding="utf-8").splitlines()
                if "vendor_text(" in line
                and "def vendor_text(" not in line
                # An override delegating to its own base is not a new applier.
                and "super().vendor_text(" not in line
            ]
            assert not calls, f"{path.name} applies vendor_text: {calls}"

    def test_a_reduction_that_empties_the_haystack_is_not_what_the_base_does(self):
        """The degrade rule, stated as behaviour rather than as prose.

        An empty haystack classifies as `None`, so a hook that returned "" on a
        parse failure would launder a genuine 401 into a transport-shaped retry.
        This is the shape that must NOT be written; it is here so the reason is
        executable.
        """
        class _Launderer(GooseProvider):
            def vendor_text(self, stdout, stderr):
                return "", ""

        # goose 1.53.0 also tags this 401 with its error `kind`, which
        # `GooseProvider.classify_failure` reads off stdout before any table
        # (review 24, pins #4); `other` takes that away, so the verdict below
        # is the tables' alone, as it was when this test was written.
        genuine_401 = _c1_capture("goose", "auth-error.stdout.txt").replace(
            '"kind": "authentication"', '"kind": "other"'
        )
        assert '"kind": "other"' in genuine_401
        assert GooseProvider().classify_failure(genuine_401, "", 0) == \
            STATUS_AUTH_FAILED
        assert _Launderer().classify_failure(genuine_401, "", 0) is None


class TestVendorTextOnTheFourEchoingAdapters:
    """The measured set: goose, copilot, vibe, muse."""

    @pytest.mark.parametrize("provider,capture,before", C1_ECHOING_SUCCESS_CAPTURES)
    def test_a_successful_run_classifies_as_nothing(self, provider, capture, before):
        """A run that produced an answer is not a transport fault."""
        stdout = _c1_capture(provider, capture)
        assert _c1_adapter(provider).classify_failure(stdout, "", 0) is None

    @pytest.mark.parametrize("provider,capture,before", C1_ECHOING_SUCCESS_CAPTURES)
    def test_and_that_is_a_change(self, provider, capture, before, monkeypatch):
        """The same bytes, on the pre-C1 haystack, drew `before`."""
        stdout = _c1_capture(provider, capture)
        _c1_without_the_hook(monkeypatch, provider)
        assert _c1_adapter(provider).classify_failure(stdout, "", 0) == before

    @pytest.mark.parametrize("provider", ["goose", "copilot", "vibe", "muse"])
    def test_the_reduction_only_ever_removes(self, provider):
        """Every committed capture: a pure deletion, and no stderr.

        "Shorter" is the weak half of the contract and it is not enough on its
        own: three of the four hooks re-render a parsed envelope, so a rendering
        that inserted bytes the capture never carried — muse re-serialized with
        json's default ``", "``/``": "`` separators against a COMPACT stream —
        stays under the length bound purely because the dropped prompt is
        bigger than what the rendering added. The subsequence check is the
        contract as written: every character of the reduction appears in the
        capture, in order, so the hook can only have removed.
        """
        adapter = _c1_adapter(provider)
        for path in sorted((C1_FIXTURES / provider).iterdir()):
            if path.suffix not in (".txt", ".jsonl", ".json"):
                continue
            raw = path.read_bytes().decode("utf-8")
            reduced, stderr = adapter.vendor_text(raw, "a stderr diagnosis")
            assert len(reduced) <= len(raw), (
                f"{provider}/{path.name}: the reduction GREW the capture"
            )
            remaining = iter(raw)
            assert all(char in remaining for char in reduced), (
                f"{provider}/{path.name}: the reduction INSERTED bytes the "
                f"capture never carried"
            )
            assert stderr == "a stderr diagnosis", (
                f"{provider}/{path.name}: no shipped reduction touches stderr"
            )

    @pytest.mark.parametrize("provider", ["goose", "copilot", "vibe", "muse"])
    def test_a_broken_envelope_degrades_to_its_input(self, provider):
        """Not `""` — see `test_a_reduction_that_empties_the_haystack…`."""
        for junk in ("", "   ", "not json at all", '{"messages": "not a list"}',
                     '{"unclosed": ', "\x00\x01"):
            assert _c1_adapter(provider).vendor_text(junk, "err") == (junk, "err")

    @pytest.mark.parametrize("provider,capture,expected", [
        ("goose", "auth-error.stdout.txt", STATUS_AUTH_FAILED),
        ("goose", "rate-limit.stdout.txt", STATUS_RATE_LIMITED),
        ("goose", "exit-zero-failure.stdout.txt", "server_error"),
        ("goose", "invalid-model.stdout.txt", CATEGORY_INVALID_MODEL),
        ("copilot", "auth-error.jsonl", STATUS_AUTH_FAILED),
        ("copilot", "rate-limit.jsonl", STATUS_RATE_LIMITED),
        ("copilot", "invalid-model.jsonl", CATEGORY_INVALID_MODEL),
        ("copilot", "context-overflow.jsonl", STATUS_CONTEXT_LENGTH_EXCEEDED),
    ])
    def test_the_vendors_own_diagnosis_still_reaches_the_tables(
        self, provider, capture, expected
    ):
        """The reduction removes our turn and nothing the adapter said.

        goose's `Ran into this error:`, its quoted `500 Internal Server Error`
        status line and its `401 Unauthorized` gloss all live in the ASSISTANT
        message; copilot's whole diagnosis is a `session.error` record. Both are
        on the far side of the filter.
        """
        stdout = _c1_capture(provider, capture)
        assert _c1_adapter(provider).classify_failure(stdout, "", 0) == expected


OPENCODE_FIXTURES = oc_pin.ROOT
#: The rig's stream captures (`capture/<pin>/`) the adapter is graded on.
OC_STREAM_CASES = ("run_review", "denied_turn", "s41_cont_read", "auth_error", "invalid_model")


def _oc_fixture(name):
    return (OPENCODE_FIXTURES / name).read_bytes().decode("utf-8")


def _oc_event(event_type, **fields):
    """One compact `run --format json` line, `type` first like opencode writes."""
    return json.dumps({"type": event_type, **fields}, separators=(",", ":"))


#: A vendor `error` event carrying a 429, in the shape `describe_failure` reads.
OC_429_EVENT = _oc_event(
    "error", timestamp=1, sessionID="ses_0",
    error={"name": "APIError", "data": {
        "message": "Rate limit exceeded", "statusCode": 429, "isRetryable": True,
    }},
)
#: The same shape for an auth failure, as the rig's `auth_error` case records it.
OC_AUTH_EVENT = _oc_event(
    "error", timestamp=1, sessionID="ses_0",
    error={"name": "ProviderAuthError", "data": {
        "providerID": "openrouter", "message": "API key is missing",
    }},
)
#: A `--print-logs` stderr line for a real opencode 429.
OC_429_STDERR = (
    'timestamp=x level=ERROR message=failed '
    'error="AI_APICallError: Too Many Requests" statusCode=429\n'
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


#: T2.4: every committed opencode capture, graded through `classify_failure`
#: before T2.1/T4.3 (identity `vendor_text`, no structural check) and after.
#: One cell moves: the b38de8bb denied turn, from `rate_limited` to
#: `runner_error`. `stdout`/`stderr` name fixture files (or carry literal
#: synthetic text); `None` is an empty stream.
OC_GRADED_CAPTURES = (
    # id,                       stdout,                                   stderr,                                    exit, before,          after
    ("auth_error",             ("rig", "auth_error"),                    ("rig", "auth_error.stderr"),             1,    "auth_failed",   "auth_failed"),
    ("invalid_model",          ("rig", "invalid_model"),                 ("rig", "invalid_model.stderr"),          1,    "invalid_model", "invalid_model"),
    ("run_review",             ("rig", "run_review"),                    ("rig", "run_review.stderr"),             0,    None,            None),
    ("denied_turn",            ("rig", "denied_turn"),                   ("rig", "denied_turn.stderr"),            0,    "rate_limited",  "runner_error"),
    ("continue_loop_on_deny",  ("rig", "s41_cont_read"),                 ("rig", "s41_cont_read.stderr"),          0,    None,            None),
    ("synthetic 429 event",    ("literal", OC_429_EVENT),                None,                                      1,    "rate_limited",  "rate_limited"),
    ("synthetic 429 stderr",   None,                                     ("literal", OC_429_STDERR),                1,    "rate_limited",  "rate_limited"),
)


def _oc_stream(spec):
    if spec is None:
        return ""
    kind, value = spec
    if kind == "rig":
        case, _, stream = value.partition(".")
        return oc_pin.read(case, ".stderr.txt" if stream == "stderr" else ".jsonl")
    return value


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


class TestMatchStreams:
    """T2.5: the quote-only twin of `classify_streams`, and its offsets."""

    @staticmethod
    def _corpus():
        """Every committed manifest row, through its own adapter's reduction
        and tables, plus the opencode captures and the hand-written
        `classify_streams` cases the suite pins elsewhere."""
        from utils.provider_registry import _PROVIDERS

        rows = []
        for manifest_path in sorted(C1_FIXTURES.glob("*/manifest.yaml")):
            import yaml

            provider = manifest_path.parent.name
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            for name, row in (manifest.get("rows") or {}).items():
                if not isinstance(row, dict) or row.get("not_applicable"):
                    continue
                streams = []
                for key in ("stdout", "stderr"):
                    file_name = row.get(key)
                    streams.append(
                        (manifest_path.parent / file_name).read_bytes().decode("utf-8")
                        if file_name else ""
                    )
                rows.append((f"{provider}/{name}", provider, streams[0], streams[1],
                             int(row.get("exit_code") or 0)))
        for row_id, out, err, exit_code, _before, _after in OC_GRADED_CAPTURES:
            rows.append((f"opencode/{row_id}", "opencode", _oc_stream(out),
                         _oc_stream(err), exit_code))
        # `TestClassifyFailure` grades these through opencode's adapter, so they
        # are twinned through that adapter's reduction and tables here.
        for index, stream in enumerate(CLASSIFY_FAILURE_STREAMS):
            rows.append((f"classify-failure/{index}", "opencode", "", stream, 1))
        for index, (out, err) in enumerate(CLASSIFY_FAILURE_PAIRS):
            rows.append((f"classify-failure-pair/{index}", "opencode", out, err, 1))
        for index, (haystack, _expected) in enumerate(LADDER_ORDER_CASES):
            rows.append((f"ladder-order/{index}", None, haystack, "", 1))
        # Edges no other class pins as a `classify_streams` case.
        literal = [
            ("", "", 127),
            ("", "bwrap: setting up uid map: Permission denied\n", 1),
            ("", "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n", 1),
            ("event: started\n", "Error: 401 Unauthorized\n", 1),
            ("Ran into this error: 429 rate limit exceeded", "", 0),
            ("cost 0.0429 in 14290 ms", "", 0),
            ("cursor-agent exited with code 1", "", 1),
            ("", "", 0),
            ("   ", "\n", 0),
        ]
        for index, (out, err, exit_code) in enumerate(literal):
            rows.append((f"literal/{index}", None, out, err, exit_code))
        assert all(p in _PROVIDERS for _i, p, *_ in rows if p is not None)
        return rows

    def test_the_status_always_equals_classify_streams(self):
        from utils.provider_registry import _PROVIDERS
        from utils.providers.base import classify_streams, match_streams

        for row_id, provider, stdout, stderr, exit_code in self._corpus():
            kwargs = {}
            if provider is not None:
                adapter = _PROVIDERS[provider]
                stdout, stderr = adapter.vendor_text(stdout, stderr)
                kwargs = dict(
                    auth_patterns=adapter.auth_error_patterns,
                    rate_limit_patterns=adapter.rate_limit_patterns,
                    exit_zero_markers=adapter.exit_zero_failure_markers,
                )
            for out, err, code in ((stdout, stderr, exit_code),
                                   (stderr, stdout, exit_code),
                                   (stdout, stderr, 0), (stdout, stderr, 1)):
                expected = classify_streams(out, err, code, **kwargs)
                match = match_streams(out, err, code, **kwargs)
                got = None if match is None else match.status
                assert got == expected, (row_id, code)
                if match is not None and match.pattern is not None:
                    text = (out if match.channel == "stdout" else err).lower()
                    assert 0 <= match.offset < len(text), row_id

    def test_the_offset_points_at_the_matched_text(self):
        from utils.providers.base import match_streams

        match = match_streams("model said nothing", "quota\nError: 429 Too Many Requests", 1)
        assert (match.status, match.channel) == (STATUS_RATE_LIMITED, "stderr")
        text = "quota\nError: 429 Too Many Requests".lower()
        assert text[match.offset:].startswith(("429", "too many requests"))

        match = match_streams("x\nthe provider said: rate limit exceeded", "", 1)
        assert match.channel == "stdout"
        assert "x\nthe provider said: rate limit exceeded"[match.offset:].startswith(
            "rate limit"
        )

    def test_the_patternless_steps_carry_no_position(self):
        import re

        from utils.providers.base import match_streams

        spawn = match_streams("", "", 127)
        assert (spawn.status, spawn.pattern, spawn.channel, spawn.offset) == (
            "spawn_failed", None, None, None,
        )
        marker = match_streams('{"is_error":true}', "", 0,
                               exit_zero_markers=(re.compile(r'"is_error":\s*true'),))
        assert (marker.status, marker.pattern, marker.channel, marker.offset) == (
            "runner_error", None, None, None,
        )

    def test_classify_streams_does_not_delegate_to_it(self):
        import inspect

        from utils.providers import base

        assert "match_streams" not in inspect.getsource(base.classify_streams)


class TestACutInsideCopilotsEchoIsStillDropped:
    """The residual a whole-line parse leaves: a stream cut MID-record.

    copilot's echo is one `user.message` line of 61 751 bytes — the largest in
    the stream by two orders of magnitude — so a kill lands inside it more often
    than anywhere else, and `json.loads` rejects the fragment. A filter that
    dropped only what PARSED would hand the prompt bytes straight back to the
    tables on the failed-run path `_classify_failed_run` reads.
    `truncated.jsonl` cannot show this: that capture's cut fell on a record
    boundary, so nothing committed pinned the shape until
    `truncated-mid-echo.jsonl`.
    """

    CUT = "truncated-mid-echo.jsonl"

    def test_the_committed_mid_echo_cut_classifies_as_nothing(self):
        """A half-written stream says nothing about *why* it stopped."""
        stdout = _c1_capture("copilot", self.CUT)
        assert _c1_adapter("copilot").classify_failure(stdout, "", 137) is None

    def test_and_that_is_a_change(self, monkeypatch):
        """The same bytes on the pre-C1 haystack — and on a line-parse-only one.

        `rate_limited` is one of the two classes the executor forwards, so this
        is the live half: it would have backed the whole batch off on the
        strength of a sentence in the plan under review.
        """
        stdout = _c1_capture("copilot", self.CUT)
        _c1_without_the_hook(monkeypatch, "copilot")
        assert _c1_adapter("copilot").classify_failure(stdout, "", 137) == \
            STATUS_RATE_LIMITED

    def test_the_prompt_bytes_are_gone_from_the_haystack(self):
        """Stated over the bytes, not over the verdict.

        The capture still carries our plan text — the fixture is worthless if a
        later edit removes it from the bytes instead of from the haystack.
        """
        stdout = _c1_capture("copilot", self.CUT)
        assert "rate-limit patterns" in stdout
        reduced, _stderr = _c1_adapter("copilot").vendor_text(stdout, "")
        assert "rate-limit" not in reduced
        assert "user.message" not in reduced

    def test_a_cut_record_of_ANY_OTHER_type_is_still_passed_through(self):
        """Only the echo's opener is dropped, and only at the line start.

        A fragment of an `assistant.message` is the vendor's own half-written
        text and stays in the haystack; so does a line that merely mentions the
        echo's type later on. Nothing about a rejected line is *parsed* — the
        rule is a prefix match used to remove, and never to read evidence.
        """
        adapter = _c1_adapter("copilot")
        vendor_fragment = (
            '{"type":"assistant.message","data":{"content":"429 rate limit'
        )
        mentions_it_later = (
            '{"note":"the record after this one is a "type":"user.message" '
            'and it hit a rate-limit'
        )
        for line in (vendor_fragment, mentions_it_later):
            stdout = _c1_capture("copilot", "success.jsonl") + "\n" + line
            reduced, _stderr = adapter.vendor_text(stdout, "")
            assert line in reduced, "a line that is not the echo was dropped"
            assert adapter.classify_failure(stdout, "", 137) == \
                STATUS_RATE_LIMITED

    def test_a_cut_too_early_to_name_the_type_is_passed_through(self):
        """It carries no prompt bytes, so there is nothing to remove."""
        opener = '{"type":"user.mess'
        reduced, _stderr = _c1_adapter("copilot").vendor_text(opener, "err")
        assert (reduced, _stderr) == (opener, "err")

    def test_the_record_boundary_capture_is_untouched_by_the_new_rule(self):
        """`truncated.jsonl` is the other half of the pair and does not move."""
        stdout = _c1_capture("copilot", "truncated.jsonl")
        adapter = _c1_adapter("copilot")
        assert adapter.classify_failure(stdout, "", 137) is None
        reduced, _stderr = adapter.vendor_text(stdout, "")
        assert "model.call_start" in reduced


class TestTheGooseCorpusMovesExactlyOneCell:
    """C1's acceptance test, run against every committed goose row."""

    @pytest.mark.parametrize("row,capture,before,after", C1_GOOSE_CORPUS)
    def test_each_row_classifies_as_the_table_declares(
        self, row, capture, before, after
    ):
        stdout = _c1_capture("goose", capture) if capture else ""
        assert GooseProvider().classify_failure(stdout, "", 0) == after

    @pytest.mark.parametrize("row,capture,before,after", C1_GOOSE_CORPUS)
    def test_each_row_classified_as_the_table_says_it_used_to(
        self, row, capture, before, after, monkeypatch
    ):
        stdout = _c1_capture("goose", capture) if capture else ""
        _c1_without_the_hook(monkeypatch, "goose")
        assert GooseProvider().classify_failure(stdout, "", 0) == before

    def test_exactly_one_of_the_fourteen_moved(self):
        moved = [row for row, _, before, after in C1_GOOSE_CORPUS if before != after]
        assert len(C1_GOOSE_CORPUS) == 14
        assert moved == ["prompt-echo"], (
            "C1 is a one-cell change to this corpus; anything else is a "
            "reduction that removed more than our prompt"
        )

    def test_the_table_and_the_fixture_manifest_cannot_drift(self):
        """The rig grades the same fourteen verdicts this table asserts."""
        import yaml

        manifest = yaml.safe_load(
            (C1_FIXTURES / "goose" / "manifest.yaml").read_text(encoding="utf-8")
        )
        rows = manifest["rows"]
        assert set(rows) == {row for row, _, _, _ in C1_GOOSE_CORPUS}
        for row, _capture, _before, after in C1_GOOSE_CORPUS:
            declared = rows[row]
            if "not_applicable" in declared:
                continue
            assert declared["expect"]["classify_failure"] == after, (
                f"goose/{row}: the manifest and this table disagree"
            )

    def test_the_prompt_echo_row_was_not_inverted(self):
        """The run always succeeded; only the classifier's answer was wrong."""
        import yaml

        manifest = yaml.safe_load(
            (C1_FIXTURES / "goose" / "manifest.yaml").read_text(encoding="utf-8")
        )
        expect = manifest["rows"]["prompt-echo"]["expect"]
        assert expect["success"] is True
        assert expect["classify_failure"] is None
        raw = _c1_capture("goose", "prompt-echo.stdout.txt")
        assert "address timeout scenarios" in raw, (
            "the row stops pinning anything if the echoed prompt is edited out "
            "of the bytes instead of out of the haystack"
        )


class TestCopilotAndVibeAreWatchedByTheRigFromNowOn:
    """The two halves of this defect that no fixture row watched."""

    @pytest.mark.parametrize("provider,capture", [
        ("copilot", "success.jsonl"),
        ("vibe", "success-json.json"),
    ])
    def test_a_manifest_exists_and_pins_the_success_row(self, provider, capture):
        import yaml

        path = C1_FIXTURES / provider / "manifest.yaml"
        assert path.is_file(), (
            f"{provider} ships captures with no manifest, which is how a green "
            f"run classified `rate_limited` unnoticed"
        )
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        success = manifest["rows"]["success"]
        assert success["stdout"] == capture
        assert success["expect"]["success"] is True
        assert success["expect"]["classify_failure"] is None

    @pytest.mark.parametrize("provider", ["copilot", "vibe"])
    def test_neither_sits_in_the_manifest_pending_table_any_more(self, provider):
        from tests.test_provider_fixture_rows import (
            FIXTURES_PENDING,
            MANIFEST_PENDING,
        )

        assert provider not in MANIFEST_PENDING
        assert provider not in FIXTURES_PENDING


class TestAiderStillComposesWithTheHook:
    """`aider`'s narrowing override is the precedent, and it is unchanged."""

    COST_MAP_NOISE = (
        "Error: Unable to fetch model_prices_and_context_window.json: "
        "Name or service not known\n"
    )

    def test_the_tolerated_line_is_still_dropped(self):
        """A successful aider run on a host with no egress is not a failure."""
        assert AiderProvider().classify_failure(
            self.COST_MAP_NOISE + '["ok"]', "", 0
        ) is None

    def test_a_genuine_failure_still_survives_both_reductions(self):
        assert AiderProvider().classify_failure(
            self.COST_MAP_NOISE + "litellm.AuthenticationError: unauthorized (401)",
            "", 1,
        ) == STATUS_AUTH_FAILED

    def test_the_override_still_delegates_the_rule_to_super(self):
        """Two reductions, both removal-only, so the order does not matter.

        The override narrows, ``super()`` then applies the hook. Read off the
        file rather than through ``inspect``, for the reason
        ``TestVendorTextContract._c1_base_source`` gives.
        """
        import ast
        import sys

        module = sys.modules[AiderProvider.__module__]
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AiderProvider"
        )
        method = next(
            node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "classify_failure"
        )
        dumped = ast.dump(ast.Module(body=method.body, type_ignores=[]))
        assert "'super'" in dumped and "'classify_failure'" in dumped


class TestEchoedPlanTextNoLongerDecidesTheBatch:
    """The integration half: `_classify_failed_run`, on a FAILED run.

    This is the path the green-run measurements above are only a proxy for.
    `_classify_failed_run` is consulted for a run with no usable result, and it
    forwards exactly two of the classifier's answers — `auth_failed` (rung 1,
    the whole batch fails) and `rate_limited` (rung 3, backoff). Those are
    precisely the two an echoed plan sentence could manufacture.
    """

    AUTH_SENTENCE = "The plan says: the API returns 401 unauthorized on error."
    RATE_SENTENCE = "The plan says: retries must cover rate-limit responses."

    def _capture(self, sentence):
        from utils.subagent_executor import _Capture

        return _Capture(
            stdout=_c1_goose_with_plan_sentence(sentence), stderr="", exit_code=0
        )

    @pytest.mark.parametrize("sentence_attr", ["AUTH_SENTENCE", "RATE_SENTENCE"])
    def test_a_cut_stream_stays_transport_shaped(self, sentence_attr):
        from utils.subagent_executor import _classify_failed_run

        capture = self._capture(getattr(self, sentence_attr))
        assert _classify_failed_run(GooseProvider(), capture) is None

    @pytest.mark.parametrize("sentence_attr,was", [
        ("AUTH_SENTENCE", STATUS_AUTH_FAILED),
        ("RATE_SENTENCE", STATUS_RATE_LIMITED),
    ])
    def test_and_before_C1_it_did_not(self, sentence_attr, was, monkeypatch):
        """The same bytes, the same call, on the pre-C1 haystack."""
        from utils.subagent_executor import _classify_failed_run

        capture = self._capture(getattr(self, sentence_attr))
        _c1_without_the_hook(monkeypatch, "goose")
        verdict = _classify_failed_run(GooseProvider(), capture)
        assert verdict is not None and verdict[0] == was

    def test_the_cut_stream_itself_is_unclassified_either_way(self):
        """The control: goose's real `truncated` capture, no plan sentence."""
        from utils.subagent_executor import _Capture, _classify_failed_run

        capture = _Capture(
            stdout=_c1_capture("goose", "truncated.stdout.txt"),
            stderr="",
            exit_code=0,
        )
        assert _classify_failed_run(GooseProvider(), capture) is None


class TestTheNestedProbePromptIsCleanByConstruction:
    """The preflight probe is safe because of what its prompt IS, not the hook.

    `NESTED_PROBE_PROMPT`'s auth verdict raises `RunnerRejected` and refuses the
    host outright, so an echo of it carrying a marker would be the worst
    instance of this defect. It cannot: the prompt is one clean token.
    """

    def test_the_prompt_matches_no_table_of_any_adapter(self):
        from utils.provider_registry import _PROVIDERS
        from utils.subagent_executor import NESTED_PROBE_PROMPT

        for name, adapter in sorted(_PROVIDERS.items()):
            assert adapter.classify_failure(NESTED_PROBE_PROMPT, "", 0) is None, (
                f"{name} reads the nested-exec probe prompt itself as a failure"
            )

    def test_an_adapter_echoing_it_verbatim_still_classifies_nothing(self):
        """The echo, not just the prompt — which is what a runner returns."""
        from utils.subagent_executor import NESTED_PROBE_PROMPT

        echo = f"{NESTED_PROBE_PROMPT}\n{NESTED_PROBE_PROMPT}"
        assert GooseProvider().classify_failure(echo, "", 0) is None

    def test_it_stays_one_token_with_no_marker_vocabulary_in_it(self):
        from utils.providers import base as base_module
        from utils.subagent_executor import NESTED_PROBE_PROMPT

        haystack = NESTED_PROBE_PROMPT.lower()
        tables = (
            base_module.CONTEXT_OVERFLOW_MARKERS,
            base_module.TIMEOUT_MARKERS,
            base_module.CONNECTION_ERROR_MARKERS,
            base_module.SERVER_ERROR_MARKERS,
            base_module.INVALID_MODEL_MARKERS,
            base_module.SHARED_RATE_LIMIT_MARKERS,
        )
        for table in tables:
            assert not base_module._matches_marker(table, haystack)
        for _category, markers in base_module.LAUNCH_FAILURE_MARKERS:
            assert not base_module._matches_marker(markers, haystack)
        for markers in base_module.STDERR_LAUNCH_FAILURE_MARKERS.values():
            assert not base_module._matches_marker(markers, haystack)


class TestTheAuthProbeHaystackIsUntouched:
    """`classify_auth_probe` is deliberately out of scope (base.py's second
    `f"{stdout}\\n{stderr}"`).

    An auth probe never carries our prompt, so narrowing its haystack fixes
    nothing — and it is the one verdict allowed to refuse a run outright.
    """

    @pytest.mark.parametrize("provider,capture", [
        ("goose", "auth-error.stdout.txt"),
        ("copilot", "auth-error.jsonl"),
    ])
    def test_a_real_credential_notice_is_still_read_as_not_authenticated(
        self, provider, capture
    ):
        stdout = _c1_capture(provider, capture)
        assert _c1_adapter(provider).classify_auth_probe(stdout, "", 0) is False

    def test_the_probe_reads_the_whole_capture_where_the_classifier_does_not(self):
        """One set of bytes, two deliberately different answers.

        The auth-shaped sentence is in the ECHOED user turn. `classify_failure`
        no longer sees it; `classify_auth_probe` still does, and that is the
        conservative direction — the probe's only pattern-drawn verdict is the
        one that refuses.
        """
        stdout = _c1_goose_with_plan_sentence(
            "The plan says: the API returns 401 unauthorized on error."
        )
        adapter = GooseProvider()
        assert adapter.classify_failure(stdout, "", 0) is None
        assert adapter.classify_auth_probe(stdout, "", 0) is False


class TestFormatFailureDetailsIsNoLongerTheOneRawStdoutClassifier:
    """Lane A's cosmetic `Reason:` line, routed through the hook (`:523`)."""

    def _reason(self, **kwargs):
        from utils.review_orchestrator_base import _format_failure_details

        return _format_failure_details(
            0, _c1_capture("goose", "prompt-echo.stdout.txt"), "", **kwargs
        )

    def test_with_the_adapter_it_no_longer_calls_a_green_run_timed_out(self):
        assert "Reason: Timed out" not in self._reason(provider=GooseProvider())

    def test_without_one_it_is_byte_for_byte_what_it_was(self):
        """Kept, not made an error: §7.2 needs the adapter-free rule callable."""
        assert "Reason: Timed out" in self._reason()

    def test_the_existing_three_positional_callers_still_work(self):
        from utils.review_orchestrator_base import _format_failure_details

        assert "Reason: Rate limited" in _format_failure_details(
            1, "", "HTTP status 429 returned"
        )


# =========================================================================== #
# T3.1 / T3.2 — the other subagent runners read what the VENDOR wrote
#
# plans/opencode-validator-false-rate-limit-and-denied-turn.md §3. codex does
# not echo our prompt, but it echoes the REPOSITORY: `command_execution`'s
# `aggregated_output` is the `rg`/`cat` the model ran over the plan, and
# `agent_message`/`reasoning` are its prose about it. claude-code's clean
# envelope carries the model's final prose in `result`. Neither is the vendor's
# verdict, and a plan that discusses rate limits or model names must not become
# `rate_limited` (a batch backoff) or `invalid_model` (fatal to the batch).
# =========================================================================== #

#: Every codex capture, graded through `classify_failure` on the pre-T3.1
#: haystack (`before`) and through the hook (`after`), at exit 1 — the exit a
#: failed attempt reaches `_classify_failed_run` with. **This tuple is T3.1's
#: acceptance table**: exactly one cell moves. `agentic_reasoning_multi_message`
#: drew `permission_denied` from the model's own prose ("directory listing
#: blocked by sandbox", in a `reasoning` and an `agent_message` item); nothing
#: codex itself wrote says so. Graded 2026-09-25 while landing T3.1.
T3_CODEX_CORPUS = (
    # capture,                                     before,               after
    ("agentic_reasoning_multi_message.jsonl",      "permission_denied",  None),
    ("gpt5_sol_single_message.jsonl",              None,                 None),
    ("0.156.1/code_mode_mcp.jsonl",                None,                 None),
    ("0.156.1/code_mode_mcp.rollout.jsonl",        None,                 None),
    ("0.156.1/code_mode_writes.jsonl",             None,                 None),
    ("0.156.1/code_mode_writes.rollout.jsonl",     None,                 None),
    ("0.156.1/collab_spawn_agent.jsonl",           None,                 None),
    ("0.160.0/collab_spawn_agent.jsonl",           None,                 None),
    ("0.160.0/quoting_render.jsonl",               None,                 None),
    ("0.160.0/read_only_clean.jsonl",              None,                 None),
    ("0.160.1/collab_spawn_agent.jsonl",           None,                 None),
    ("0.160.1/quoting_render.jsonl",               None,                 None),
    ("0.160.1/read_only_clean.jsonl",              None,                 None),
    ("0.161.0/collab_spawn_agent.jsonl",           None,                 None),
    ("0.161.0/quoting_render.jsonl",               None,                 None),
    ("0.161.0/read_only_clean.jsonl",              None,                 None),
    ("0.156.1/exec_session.jsonl",                 None,                 None),
    ("0.156.1/mcp_tool_call.jsonl",                None,                 None),
    ("0.156.1/quoting_render.jsonl",               None,                 None),
    ("0.156.1/read_only_clean.jsonl",              None,                 None),
    ("0.156.1/run_0c477c4d_partial.jsonl",         None,                 None),
    ("0.156.1/web_search_code_mode.jsonl",         None,                 None),
    ("0.156.1/web_search_code_mode.rollout.jsonl", None,                 None),
    ("0.156.1/web_search_hosted.jsonl",            None,                 None),
    ("0.156.1/web_search_hosted.sse.json",         None,                 None),
    ("0.156.1/web_search_hosted_replay.jsonl",     None,                 None),
    ("0.156.1/web_search_probe_replay.jsonl",      None,                 None),
    ("0.156.1/writes_patches_sessions.jsonl",      None,                 None),
    # Plan codex-sandbox-unavailable (2026-09-30): the preflight's `jail_writes`
    # turn on 0.158.0 inside the plugin-landlock jail — every write denied, so
    # the commands' own output is full of `Permission denied`, none of it the
    # vendor's. Graded `permission_denied` without the hook, nothing with it.
    ("landlock/jail_writes.out.jsonl",             "permission_denied",  None),
)

#: Subdirectories of `tests/fixtures/codex/` that hold rig scripts, schemas and
#: plugin-route data rather than a codex stdout capture.
_T3_CODEX_NOT_CAPTURES = ("capture", "schemas", "plugin-route", "rig")


def _t3_codex_item(item_type, **fields):
    item = {"id": f"item_{item_type}", "type": item_type, **fields}
    return json.dumps({"type": "item.completed", "item": item})


#: The case the plan names: a validator that grepped a plan discussing rate
#: limits and model names, on an attempt that failed for some other reason.
T3_CODEX_REPO_ECHO = "\n".join([
    json.dumps({"type": "thread.started", "thread_id": "0000"}),
    json.dumps({"type": "turn.started"}),
    _t3_codex_item(
        "command_execution",
        command="/bin/bash -lc \"rg -n '429|model' plan.md\"",
        aggregated_output=(
            "12: the client backs off on 429 Too Many Requests\n"
            "31: a typo in the id is reported as model not found\n"
            "40: 401 Unauthorized means the key was revoked\n"
        ),
        exit_code=0,
        status="completed",
    ),
    _t3_codex_item("reasoning", text="The plan says rate limit exceeded is retried."),
    _t3_codex_item("agent_message", text="Section 3 covers the 429 Too Many Requests path."),
]) + "\n"


class TestCodexVendorTextGradesEveryCapture:
    """T3.1 — T2.4's technique: every committed capture, before and after."""

    @pytest.mark.parametrize("capture,before,after", T3_CODEX_CORPUS)
    def test_after(self, capture, before, after):
        stdout = _c1_capture("codex", capture)
        assert _c1_adapter("codex").classify_failure(stdout, "", 1) == after

    @pytest.mark.parametrize("capture,before,after", T3_CODEX_CORPUS)
    def test_before(self, capture, before, after, monkeypatch):
        stdout = _c1_capture("codex", capture)
        _c1_without_the_hook(monkeypatch, "codex")
        assert _c1_adapter("codex").classify_failure(stdout, "", 1) == before

    def test_the_table_covers_every_capture(self):
        """A new capture is graded when it is committed, not when it bites."""
        root = C1_FIXTURES / "codex"
        on_disk = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in (".jsonl", ".json", ".txt")
            and not set(path.relative_to(root).parts) & set(_T3_CODEX_NOT_CAPTURES)
        }
        assert on_disk == {row[0] for row in T3_CODEX_CORPUS}

    def test_every_capture_is_only_ever_reduced(self):
        """Whole lines out, nothing in, stderr untouched (the base contract)."""
        adapter = _c1_adapter("codex")
        for capture, _before, _after in T3_CODEX_CORPUS:
            raw = _c1_capture("codex", capture)
            reduced, stderr = adapter.vendor_text(raw, "a stderr diagnosis")
            assert len(reduced) <= len(raw), capture
            remaining = iter(raw)
            assert all(char in remaining for char in reduced), capture
            assert stderr == "a stderr diagnosis", capture


class TestCodexVendorText:
    """T3.1 — what the hook drops, what it keeps, and how it degrades."""

    def test_repo_text_in_a_failed_attempt_classifies_as_nothing(self):
        """The plan's synthetic stream: `429` and `model not found` quoted."""
        adapter = _c1_adapter("codex")
        assert adapter.classify_failure(T3_CODEX_REPO_ECHO, "", 1) is None
        assert adapter.classify_failure(T3_CODEX_REPO_ECHO, "", 0) is None

    def test_and_that_is_a_change(self, monkeypatch):
        """On the pre-T3.1 haystack the quoted 401 outranked everything."""
        _c1_without_the_hook(monkeypatch, "codex")
        assert _c1_adapter("codex").classify_failure(
            T3_CODEX_REPO_ECHO, "", 1
        ) == STATUS_AUTH_FAILED

    @pytest.mark.parametrize("item_type", [
        "command_execution", "agent_message", "reasoning", "file_change",
        "mcp_tool_call", "web_search", "collab_tool_call",
    ])
    @pytest.mark.parametrize("event_type", [
        "item.started", "item.updated", "item.completed",
    ])
    def test_model_and_repo_items_are_dropped(self, item_type, event_type):
        line = json.dumps({"type": event_type, "item": {
            "id": "item_1", "type": item_type, "text": "rate limit exceeded",
        }})
        stdout = f'{{"type":"turn.started"}}\n{line}\n'
        reduced, _stderr = _c1_adapter("codex").vendor_text(stdout, "")
        assert reduced == '{"type":"turn.started"}\n'

    @pytest.mark.parametrize("line", [
        '{"type":"thread.started","thread_id":"0000"}',
        '{"type":"turn.started"}',
        '{"type":"turn.completed","usage":{"input_tokens":1}}',
        '{"type":"error","message":"exceeded retry limit, last status: 429"}',
        '{"type":"turn.failed","error":{"message":"unexpected status 401"}}',
        # codex's own non-fatal notice is an item, but it is codex's words.
        _t3_codex_item("error", message="Model metadata for `x` not found."),
        # Uncaptured item and event types are kept: the conservative direction.
        _t3_codex_item("todo_list", items=[]),
        '{"type":"some.future.event","message":"x"}',
        "stream error: exceeded retry limit, last status: 429 Too Many Requests",
    ])
    def test_codex_own_events_are_kept(self, line):
        stdout = f"{line}\n" + _t3_codex_item("agent_message", text="hi") + "\n"
        reduced, _stderr = _c1_adapter("codex").vendor_text(stdout, "")
        assert reduced == f"{line}\n"

    def test_a_real_429_is_not_outranked_by_a_quoted_401(self):
        """Auth is asked before rate limit, so a quoted 401 used to win."""
        stdout = T3_CODEX_REPO_ECHO + (
            '{"type":"turn.failed","error":{"message":"stream error: exceeded '
            'retry limit, last status: 429 Too Many Requests"}}\n'
        )
        assert _c1_adapter("codex").classify_failure(stdout, "", 1) == \
            STATUS_RATE_LIMITED

    def test_a_real_401_still_reaches_the_tables(self):
        stdout = T3_CODEX_REPO_ECHO + (
            '{"type":"turn.failed","error":{"message":"unexpected status 401 '
            'Unauthorized: Missing authentication in header"}}\n'
        )
        assert _c1_adapter("codex").classify_failure(stdout, "", 1) == \
            STATUS_AUTH_FAILED

    def test_stderr_is_never_reduced(self):
        stderr = "ERROR: 429 Too Many Requests"
        reduced, kept = _c1_adapter("codex").vendor_text(T3_CODEX_REPO_ECHO, stderr)
        assert kept == stderr
        assert _c1_adapter("codex").classify_failure(
            T3_CODEX_REPO_ECHO, stderr, 1
        ) == STATUS_RATE_LIMITED

    def test_a_cut_final_repo_item_is_dropped(self):
        """A stream killed mid-line: textual, and order-tolerant."""
        adapter = _c1_adapter("codex")
        head = '{"type":"turn.started"}\n'
        for cut in (
            '{"type":"item.completed","item":{"id":"item_7",'
            '"type":"command_execution","aggregated_output":"429 Too Many',
            # `type` is not the first key of either object.
            '{"item":{"id":"item_7","type":"agent_message","text":"x"},'
            '"type":"item.updated","extra":"rate lim',
        ):
            reduced, _stderr = adapter.vendor_text(head + cut, "")
            assert reduced == head, cut

    def test_a_cut_final_error_line_is_kept(self):
        stdout = '{"type":"turn.started"}\n{"type":"turn.failed","error":{"mess'
        assert _c1_adapter("codex").vendor_text(stdout, "") == (stdout, "")

    def test_a_cut_line_that_is_not_last_is_kept(self):
        """Only the END of a stream is cut by a kill; a beginning cut is kept."""
        cut = '"type":"command_execution","aggregated_output":"429 Too Many"}}'
        stdout = cut + '\n{"type":"turn.started"}\n'
        assert _c1_adapter("codex").vendor_text(stdout, "") == (stdout, "")

    @pytest.mark.parametrize("junk", [
        "", "   ", "not json at all", '{"unclosed": ', "\x00\x01",
        # A lone cut repo item: no line parsed, so not an event stream.
        '{"type":"item.completed","item":{"id":"i","type":"reasoning","text":"x',
    ])
    def test_no_parseable_line_is_identity(self, junk):
        assert _c1_adapter("codex").vendor_text(junk, "err") == (junk, "err")

    def test_a_raised_exception_is_identity(self, monkeypatch):
        from utils.providers import codex as codex_module

        def _boom(stdout):
            raise RuntimeError("reduction failed")

        monkeypatch.setattr(codex_module, "_vendor_stdout", _boom)
        assert _c1_adapter("codex").vendor_text(T3_CODEX_REPO_ECHO, "err") == (
            T3_CODEX_REPO_ECHO, "err"
        )

    def test_the_readonly_audit_still_sees_every_item(self):
        """The audit is separate: it judges the commands the hook drops."""
        from utils.providers import codex_stream_audit

        source = Path(codex_stream_audit.__file__).read_text(encoding="utf-8")
        assert "vendor_text" not in source
        assert "_vendor_stdout" not in source


def _t3_claude_envelope(result, *, is_error=False, status=None,
                        terminal_reason="completed"):
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "api_error_status": status,
        "terminal_reason": terminal_reason,
        "result": result,
        "session_id": "00000000-0000-0000-0000-000000000000",
    })


class TestClaudeCodeCleanEnvelopeProseIsNotAVerdict:
    """T3.2 — the prose fallback, audited against the committed captures."""

    @pytest.mark.parametrize("prose,before", [
        ("The plan's retry section says: rate limit exceeded -> back off.",
         STATUS_RATE_LIMITED),
        ("Section 4: a 429 Too Many Requests is retried twice.",
         STATUS_RATE_LIMITED),
        ("The plan says an unknown id is reported as model not found.",
         CATEGORY_INVALID_MODEL),
        ("Per the plan, 401 Unauthorized means the key was revoked.",
         STATUS_AUTH_FAILED),
    ])
    def test_a_green_result_quoting_the_plan_classifies_as_nothing(
        self, prose, before
    ):
        stdout = _t3_claude_envelope(prose)
        adapter = _c1_adapter("claude-code")
        for exit_code in (0, 1):
            assert adapter.classify_failure(stdout, "", exit_code) is None
        # ... and that is a change: the shared ladder over the whole stdout
        # (the base hook's identity reduction, not claude-code's override).
        with patch.object(
            ClaudeCodeProvider, "vendor_text", LLMProvider.vendor_text
        ):
            assert LLMProvider.classify_failure(adapter, stdout, "", 0) == before

    def test_stderr_is_still_read_beside_a_clean_envelope(self):
        stdout = _t3_claude_envelope("rate limit exceeded")
        assert _c1_adapter("claude-code").classify_failure(
            stdout, "Error: 401 Unauthorized", 1
        ) == STATUS_AUTH_FAILED

    def test_a_launch_failure_still_wins(self):
        assert _c1_adapter("claude-code").classify_failure(
            _t3_claude_envelope("rate limit exceeded"), "", 127
        ) is not None

    def test_a_refusing_envelope_without_a_status_is_still_scanned(self):
        """`is_error: true` is the vendor speaking; its sentence is the verdict."""
        stdout = _t3_claude_envelope(
            "API Error: rate limit exceeded", is_error=True,
            terminal_reason="api_error",
        )
        assert _c1_adapter("claude-code").classify_failure(stdout, "", 1) == \
            STATUS_RATE_LIMITED

    def test_a_non_completed_terminal_reason_is_still_scanned(self):
        stdout = _t3_claude_envelope(
            "API Error: rate limit exceeded", terminal_reason="api_error"
        )
        assert _c1_adapter("claude-code").classify_failure(stdout, "", 1) == \
            STATUS_RATE_LIMITED

    def test_a_status_still_answers_first(self):
        stdout = _t3_claude_envelope("anything", is_error=True, status=429,
                                     terminal_reason="api_error")
        assert _c1_adapter("claude-code").classify_failure(stdout, "", 1) == \
            STATUS_RATE_LIMITED

    def test_non_envelope_stdout_takes_the_ladder_as_before(self):
        assert _c1_adapter("claude-code").classify_failure(
            "Error: 429 Too Many Requests", "", 1
        ) == STATUS_RATE_LIMITED

    def test_every_committed_failure_capture_keeps_its_verdict(self):
        """The audit's evidence: no real failure needs a clean envelope's prose.

        All eight refusing rows carry `is_error: true`, and the real 429
        carries `api_error_status: 429`, so the plan's fallback (keep the
        scan, restricted to `is_error: true`) is what the adapter already does.
        """
        import yaml
        from utils.providers.result_envelope import is_clean_envelope

        root = C1_FIXTURES / "claude-code"
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        adapter = _c1_adapter("claude-code")
        graded = 0
        for name, row in manifest["rows"].items():
            if not row.get("stdout"):
                continue
            stdout = (root / row["stdout"]).read_text(encoding="utf-8")
            stderr = (root / row["stderr"]).read_text(encoding="utf-8")
            expected = row["expect"]["classify_failure"]
            assert adapter.classify_failure(
                stdout, stderr, row["exit_code"]
            ) == expected, name
            if expected is not None:
                assert not is_clean_envelope(stdout), name
                graded += 1
        assert graded == 8

    @pytest.mark.parametrize("stdout,clean", [
        (_t3_claude_envelope("ok"), True),
        (_t3_claude_envelope("ok", is_error=True), False),
        (_t3_claude_envelope("ok", terminal_reason="api_error"), False),
        (_t3_claude_envelope("ok", status=400), False),
        ("not an envelope", False),
        ("", False),
    ])
    def test_is_clean_envelope(self, stdout, clean):
        from utils.providers.result_envelope import is_clean_envelope

        assert is_clean_envelope(stdout) is clean


#: `TestClassifyLadderOrder`'s `(haystack, expected)` cases, at module scope so
#: `TestMatchStreams` twins them through `match_streams` as well.
LADDER_MODEL_NAMED_404 = (
    "API error (status 404 Not Found): The model `stub-model` does not "
    "exist or you do not have access to it."
)
LADDER_SERVER_ERRORS = [
    "API error (status 503)",
    "internal server error",
]
LADDER_ORDER_CASES = [(LADDER_MODEL_NAMED_404, "invalid_model")] + [
    (haystack, "server_error") for haystack in LADDER_SERVER_ERRORS
]


class TestClassifyLadderOrder:
    """C2: invalid-model is consulted BEFORE server-error, and stays that way.

    ``\\bapi error\\b`` is a member of ``SERVER_ERROR_MARKERS``, and several
    vendors wrap a model-named 404 in exactly that phrase. While the server
    table was asked first, a *terminal* config error landed on ``server_error``
    — a **retryable** category — and burned the retry budget on a condition no
    retry can fix. A model-named 404 is strictly more specific than "api
    error", so the specific table is asked first.

    These tests pin the ORDER, not one vendor's wording, because the defect was
    an ordering fact rather than a missing pattern: no adapter gained a pattern
    in C2.
    """

    def test_a_model_named_404_wrapped_in_api_error_is_invalid_model(self):
        """The exact collision: both tables match, and the order decides."""
        from utils.providers.base import classify_streams

        assert classify_streams(LADDER_MODEL_NAMED_404, "", 1) == "invalid_model"

    def test_a_server_error_carrying_no_model_name_is_still_server_error(self):
        """Reordering must not launder a genuine 5xx into a config error."""
        from utils.providers.base import classify_streams

        for haystack in LADDER_SERVER_ERRORS:
            assert classify_streams(haystack, "", 1) == "server_error"

    def test_the_two_tables_are_consulted_in_the_declared_order(self):
        """Source-order guard, so a later reshuffle is red rather than silent.

        Asserted against the function's own body rather than against a sample,
        because a sample only catches the collisions somebody thought of.
        """
        import inspect

        from utils.providers import base

        body = inspect.getsource(base.classify_streams)
        invalid_at = body.index("INVALID_MODEL_MARKERS, haystack")
        server_at = body.index("SERVER_ERROR_MARKERS, haystack")
        assert invalid_at < server_at, (
            "C2: INVALID_MODEL_MARKERS must be consulted before "
            "SERVER_ERROR_MARKERS — a model-named 404 is more specific than "
            "'api error', and classifying it as a retryable server_error burns "
            "the retry budget on a terminal condition."
        )

    def test_the_ladder_docstring_still_documents_the_shipped_order(self):
        """The docstring moved with the code in C2; keep them together."""
        from utils.providers.base import classify_streams

        doc = classify_streams.__doc__ or ""
        assert "invalid model, then server" in doc


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


class TestCodexSlotsStartWithoutThePlugins:
    """2026-10-02: a nested ``codex exec`` booted the user's installed plugins, so
    every review slot started multi-llm's own MCP server, whose venv Python wrote
    ``__pycache__/*.pyc`` into the watched ``$CODEX_HOME/plugins`` tree and fired
    ``[CODEX_HOME_CHECK]`` on every slot. ``-c features.plugins=false`` stops it."""

    def test_every_codex_argv_leg_carries_the_plugins_off_pair(self):
        from utils.providers.codex import PLUGINS_OFF_PAIR, CodexProvider

        provider = CodexProvider()
        cmd = provider.build_command("p", "gpt-5.5")
        assert PLUGINS_OFF_PAIR == ("-c", "features.plugins=false")
        at = cmd.index("-c")
        assert tuple(cmd[at:at + 2]) == PLUGINS_OFF_PAIR
        assert at < cmd.index("--ephemeral") < cmd.index("--skip-git-repo-check")
        assert cmd[2:4] == ["--sandbox", cmd[3]], "--sandbox keeps its fixed offset"
