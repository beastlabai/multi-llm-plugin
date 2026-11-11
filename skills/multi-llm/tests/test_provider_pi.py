"""Tests for the pi adapter.

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
from utils.providers import base as base_module
from utils.providers.pi import PiProvider


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
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

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
