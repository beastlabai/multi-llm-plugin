"""Tests for the Gemini adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.gemini import GeminiProvider


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
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

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
