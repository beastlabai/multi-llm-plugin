"""Tests for the Cline adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.cline import ClineProvider


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
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

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
