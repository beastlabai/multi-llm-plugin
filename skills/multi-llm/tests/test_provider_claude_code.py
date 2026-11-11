"""Tests for the Claude Code adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import CATEGORY_INVALID_MODEL, LLMProvider
from utils.retry_ladder import STATUS_AUTH_FAILED, STATUS_RATE_LIMITED
from utils.providers.claude_code import ClaudeCodeProvider
from tests.mock_launchers import write_launcher
from tests.provider_test_support import C1_FIXTURES, _c1_adapter


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
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

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
                "utils.providers.version_gate.read_help_tokens",
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


def _write_fake_claude(tmp_path, *, help_text, body="", name="claude"):
    """Put an executable fake `claude` on a fresh PATH directory.

    The §5.4.1 gate sniffs the CLI's *own* help, so the fake has to answer
    `--help` for real — which is exactly the trap `tests/mocks/mock_llm.py`
    records: a flag missing from the help text is silently dropped one step
    before argparse would have exited 2 over it.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    write_launcher(
        bin_dir, name,
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"HELP = {help_text!r}\n"
        "argv = sys.argv[1:]\n"
        "if '--help' in argv or '-h' in argv:\n"
        "    sys.stdout.write(HELP)\n"
        "    sys.exit(0)\n"
        f"{body or 'sys.exit(0)'}\n",
    )
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
        write_launcher(
            bin_dir, "claude", "#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n"
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
        write_launcher(
            bin_dir, "claude",
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            f"HELP = {HELP_WITH_FALLBACK!r}\n"
            "open(os.environ['MLLM_HELP_COUNTER'], 'a').write('x')\n"
            "sys.stdout.write(HELP)\n"
            "sys.exit(0)\n",
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
