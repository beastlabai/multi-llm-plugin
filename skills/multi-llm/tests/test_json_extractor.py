"""Tests for JSON extractor utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.json_extractor import generate_output_path, sanitize_model_name


class TestSanitizeModelName:
    """Tests for sanitize_model_name function."""

    def test_replaces_colon(self):
        """Test that colons are replaced with underscore."""
        assert sanitize_model_name("cursor-agent:auto") == "cursor-agent_auto"

    def test_replaces_dot(self):
        """Test that dots are replaced with underscore."""
        assert sanitize_model_name("gpt-5.2-high") == "gpt-5_2-high"

    def test_replaces_slash(self):
        """Test that slashes are replaced with underscore."""
        assert sanitize_model_name("moonshotai/kimi-k2.5") == "moonshotai_kimi-k2_5"

    def test_preserves_hyphen(self):
        """Test that hyphens are preserved."""
        assert sanitize_model_name("my-model-name") == "my-model-name"

    def test_preserves_underscore(self):
        """Test that underscores are preserved."""
        assert sanitize_model_name("my_model_name") == "my_model_name"

    def test_preserves_alphanumeric(self):
        """Test that alphanumeric characters are preserved."""
        assert sanitize_model_name("model123ABC") == "model123ABC"

    def test_complex_model_spec(self):
        """Test a complex model specification."""
        assert sanitize_model_name("cursor-agent:gpt-5.2-high") == "cursor-agent_gpt-5_2-high"
        assert sanitize_model_name("kilocode:moonshotai/kimi-k2.5") == "kilocode_moonshotai_kimi-k2_5"


class TestGenerateOutputPath:
    """Tests for generate_output_path function."""

    def test_sanitizes_colon(self, tmp_path):
        """Test that colons are replaced."""
        path = generate_output_path(tmp_path, "test", "review-plan", "cursor-agent:auto")
        assert ":" not in path.name
        assert path.name == "cursor-agent_auto.json"

    def test_sanitizes_dot(self, tmp_path):
        """Test that dots are replaced (consistency with sanitize_model_name)."""
        path = generate_output_path(tmp_path, "test", "review-plan", "cursor-agent:gpt-5.2-high")
        assert "." not in path.stem  # stem excludes .json extension
        assert path.name == "cursor-agent_gpt-5_2-high.json"

    def test_sanitizes_slash(self, tmp_path):
        """Test that slashes are replaced."""
        path = generate_output_path(tmp_path, "test", "review-plan", "kilocode:moonshotai/kimi-k2.5")
        assert "/" not in path.name
        assert path.name == "kilocode_moonshotai_kimi-k2_5.json"

    def test_preserves_hyphen_and_underscore(self, tmp_path):
        """Test that hyphens and underscores are preserved."""
        path = generate_output_path(tmp_path, "test", "review-plan", "my-provider_test")
        assert path.name == "my-provider_test.json"

    def test_uses_sanitize_model_name(self, tmp_path):
        """Test that generate_output_path uses sanitize_model_name consistently."""
        test_cases = [
            "cursor-agent:gpt-5.2-high",
            "kilocode:moonshotai/kimi-k2.5",
            "gemini:gemini-2.5-pro",
            "simple-model",
        ]

        for model_spec in test_cases:
            path = generate_output_path(tmp_path, "test", "review-plan", model_spec)
            expected_name = sanitize_model_name(model_spec) + ".json"
            assert path.name == expected_name, f"Mismatch for {model_spec}: {path.name} != {expected_name}"
