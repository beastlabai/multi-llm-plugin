"""Tests for the Goose adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.goose import GooseProvider


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
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

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
