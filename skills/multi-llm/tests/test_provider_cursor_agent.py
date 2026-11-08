"""Tests for the cursor-agent adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import base as base_module, cursor_agent as cursor_agent_module
from utils.providers.cursor_agent import CursorAgentProvider


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
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

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
