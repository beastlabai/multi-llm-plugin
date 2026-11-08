"""Tests for the Grok adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.grok import GrokProvider


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
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

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
