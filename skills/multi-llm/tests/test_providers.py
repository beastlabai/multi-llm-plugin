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
from utils.providers.base import split_reasoning_effort
from utils.providers.claude_code import ClaudeCodeProvider
from utils.providers.cline import ClineProvider
from utils.providers.codex import CodexProvider
from utils.providers.cursor_agent import CursorAgentProvider
from utils.providers.gemini import GeminiProvider
from utils.providers.goose import GooseProvider
from utils.providers.grok import GrokProvider
from utils.providers.kilocode import KiloCodeProvider
from utils.providers.opencode import OpenCodeProvider
from utils.providers.pi import PiProvider


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
        """Test that get_env returns empty dict (env stripping is via get_remove_env)."""
        env = provider.get_env("opus")

        assert env == {}


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
        """Test command building for cursor-agent CLI."""
        prompt = "Review this code"
        model = "gpt-4"

        cmd = provider.build_command(prompt, model)

        assert cmd == [
            "cursor-agent",
            "--print",
            "-f",
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
        """Test command building for Grok Build CLI."""
        prompt = "Analyze this"
        model = "grok-4.5"

        cmd = provider.build_command(prompt, model)

        assert cmd == [
            "grok",
            "--no-auto-update",
            "--always-approve",
            "-p",
            "Analyze this",
            "--output-format",
            "json",
            "-m",
            "grok-4.5",
        ]

    def test_grok_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the seven valid efforts."""
        from utils.providers.grok import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

    def test_grok_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --reasoning-effort pair after -m <base>."""
        cmd = provider.build_command("Analyze this", "grok-4.5:low")

        assert cmd == [
            "grok",
            "--no-auto-update",
            "--always-approve",
            "-p",
            "Analyze this",
            "--output-format",
            "json",
            "-m",
            "grok-4.5",
            "--reasoning-effort",
            "low",
        ]

    def test_grok_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --reasoning-effort form."""
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            cmd = provider.build_command("Analyze this", f"grok-4.5:{effort}")

            assert cmd == [
                "grok",
                "--no-auto-update",
                "--always-approve",
                "-p",
                "Analyze this",
                "--output-format",
                "json",
                "-m",
                "grok-4.5",
                "--reasoning-effort",
                effort,
            ], f"effort {effort!r} did not build the expected command"

    def test_grok_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --reasoning-effort flag."""
        for model in ("grok-4.5", "grok-4.5:turbo"):
            cmd = provider.build_command("Analyze this", model)

            assert cmd == [
                "grok",
                "--no-auto-update",
                "--always-approve",
                "-p",
                "Analyze this",
                "--output-format",
                "json",
                "-m",
                model,
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

        assert cmd == ["opencode", "run", "--format", "json", "--model", "claude-3", "Generate code"]

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
                "--model",
                model,
                "Generate code",
            ], f"model {model!r} was not passed through verbatim"

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

    @patch("shutil.which")
    def test_is_available_true(self, mock_which, provider):
        """Test is_available returns True when kilocode is found."""
        mock_which.return_value = "/usr/local/bin/kilocode"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("kilocode")

    @patch("shutil.which")
    def test_is_available_false(self, mock_which, provider):
        """Test is_available returns False when kilocode is not found."""
        mock_which.return_value = None

        assert provider.is_available() is False
        mock_which.assert_called_once_with("kilocode")

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

    def test_cline_parse_non_jsonl_fallback(self, provider):
        """Fallback extraction when stdout is not JSONL but embeds JSON."""
        stdout = "Some banner text [1, 2, 3] trailing text"

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

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

    def test_aider_parse_wrapped_json_mid_string_fails(self, provider):
        """Hard-wrapped JSON (literal newline INSIDE a string) is unparseable.

        This documents why get_env() sets COLUMNS=10000: without it aider
        wraps output at ~80 columns and long string values get literal
        newlines inserted, which json.loads rejects (and
        extract_json_from_text does not repair). Newlines BETWEEN tokens
        are fine — only mid-string breaks matter.
        """
        wrapped = (
            '[\n  {\n    "title": "Missing Input Validation",\n'
            '    "desc": "The function responsible for parsing user-supplied \n'
            'configuration parameters lacks proper validation checks."\n  }\n]'
        )
        stdout = self._stdout(wrapped)

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

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
            "--yes-always",
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
        plan.write_text("# Plan")
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
            "--yes-always",
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
                "--yes-always",
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
                "--yes-always",
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
        plan.write_text("# Plan")

        found = provider._extract_read_files(
            f"Read {plan} and the missing /nonexistent/file.md"
        )

        assert found == [str(plan)]


# Per-provider prompt transport table: every current provider CLI receives
# the prompt as a single argv element (none has a documented stdin or
# prompt-file mode used anywhere in this codebase), so all adapters declare
# prompt_transport == "argv" and rely on llm_client's Windows argv-length
# check. An adapter that moves to stdin/file transport must update both its
# prompt_transport and this table.
TRANSPORT_CASES = [
    (AgyProvider, "argv", "Gemini 3.1 Pro (High)"),
    (AiderProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (ClaudeCodeProvider, "argv", "sonnet"),
    (ClineProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (CodexProvider, "argv", "gpt-5.5"),
    (CursorAgentProvider, "argv", "auto"),
    (GeminiProvider, "argv", "gemini-2.5-pro"),
    (GooseProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (GrokProvider, "argv", "grok-4"),
    (KiloCodeProvider, "argv", "openai-native/gpt-5.5"),
    (OpenCodeProvider, "argv", "openai/gpt-5.5"),
    (PiProvider, "argv", "anthropic/claude-opus-4-8"),
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
        """prompt_transport is declared as decided AND matches the built argv."""
        provider = provider_cls()

        assert provider.prompt_transport == transport
        assert transport == "argv"

        cmd = provider.build_command(self.PROMPT, model)
        # The full prompt travels as exactly ONE argv element (aider
        # prefixes "/ask " but keeps the prompt intact in that element).
        prompt_args = [arg for arg in cmd if self.PROMPT in arg]
        assert len(prompt_args) == 1

    def test_table_covers_every_registered_provider(self):
        """The transport table stays in sync with the provider registry."""
        from utils.provider_registry import _PROVIDERS

        registered = {type(p) for p in _PROVIDERS.values()}

        assert registered == {cls for cls, _, _ in TRANSPORT_CASES}


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
        """Test that default_timeout is set correctly."""
        assert provider.default_timeout == 600

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


class TestPiProvider:
    """Tests for the PiProvider class."""

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
            "--model",
            "openrouter/z-ai/glm-5.2",
            "Review this code",
        ]

    def test_build_command_bare_model(self, provider):
        """A bare model pattern without a provider prefix passes through."""
        cmd = provider.build_command("Analyze this", "sonnet")

        assert cmd == ["pi", "--no-session", "-p", "--model", "sonnet", "Analyze this"]

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
        """Module-level REASONING_EFFORTS holds exactly the six valid efforts."""
        from utils.providers.codex import REASONING_EFFORTS

        assert REASONING_EFFORTS == {"none", "minimal", "low", "medium", "high", "xhigh"}

    def test_codex_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: -c pair after --model <base>, prompt last."""
        cmd = provider.build_command("p", "gpt-5.5:high")

        assert cmd == [
            "codex",
            "exec",
            "--full-auto",
            "--json",
            "--model",
            "gpt-5.5",
            "-c",
            "model_reasoning_effort=high",
            "p",
        ]

    def test_codex_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the -c model_reasoning_effort form."""
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
            cmd = provider.build_command("Review this code", f"gpt-5.5:{effort}")

            assert cmd == [
                "codex",
                "exec",
                "--full-auto",
                "--json",
                "--model",
                "gpt-5.5",
                "-c",
                f"model_reasoning_effort={effort}",
                "Review this code",
            ], f"effort {effort!r} did not build the expected command"

    def test_codex_build_command_multi_colon_splits_on_last(self, provider):
        """Multi-colon model rsplits on the LAST colon: base foo:bar, effort high."""
        cmd = provider.build_command("p", "foo:bar:high")

        assert cmd == [
            "codex",
            "exec",
            "--full-auto",
            "--json",
            "--model",
            "foo:bar",
            "-c",
            "model_reasoning_effort=high",
            "p",
        ]

    def test_codex_build_command_passthrough_no_colon(self, provider):
        """A model without a colon is passed verbatim with no -c argument."""
        cmd = provider.build_command("p", "gpt-5.5")

        assert cmd == ["codex", "exec", "--full-auto", "--json", "--model", "gpt-5.5", "p"]

    def test_codex_build_command_passthrough_unknown_suffix(self, provider):
        """An unknown suffix is not an effort: model passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:turbo")

        assert cmd == ["codex", "exec", "--full-auto", "--json", "--model", "gpt-5.5:turbo", "p"]

    def test_codex_build_command_passthrough_wrong_case_suffix(self, provider):
        """Effort matching is case-sensitive: :HIGH is passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:HIGH")

        assert cmd == ["codex", "exec", "--full-auto", "--json", "--model", "gpt-5.5:HIGH", "p"]

    def test_codex_build_command_passthrough_empty_suffix(self, provider):
        """A trailing colon with no effort is passed verbatim, no -c."""
        cmd = provider.build_command("p", "gpt-5.5:")

        assert cmd == ["codex", "exec", "--full-auto", "--json", "--model", "gpt-5.5:", "p"]

    def test_codex_build_command_passthrough_bare_effort_word(self, provider):
        """A bare effort word as the full model name (no colon) is passed verbatim."""
        cmd = provider.build_command("p", "high")

        assert cmd == ["codex", "exec", "--full-auto", "--json", "--model", "high", "p"]

    def test_codex_build_command_passthrough_empty_base(self, provider):
        """An empty base before the colon (:high) is passed verbatim, no -c."""
        cmd = provider.build_command("p", ":high")

        assert cmd == ["codex", "exec", "--full-auto", "--json", "--model", ":high", "p"]

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
