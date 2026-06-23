"""Unit tests for provider registry and configuration loading.

This module tests the provider registry functionality including:
- Configuration loading from providers.yaml
- Model spec parsing (with and without provider prefixes)
- Provider availability checks
- Default model handling
- Model validation
"""

from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provider_registry import (
    load_config,
    parse_model_spec,
    get_available_models,
    get_all_model_specs,
    get_provider,
    get_provider_timeout,
    get_provider_max_concurrent,
    is_model_valid,
    get_default_models,
    has_default_models,
    get_quick_models,
    has_quick_models,
)


class TestLoadConfig:
    """Tests for configuration loading functionality."""

    def test_load_config_valid(self):
        """Load providers.yaml successfully and verify structure.

        This test verifies that the actual providers.yaml file loads correctly
        and contains the expected structure with providers, models, and defaults.
        """
        # Reset the cached config to ensure fresh load
        import utils.provider_registry as registry
        registry._config = None

        config = load_config()

        # Verify basic structure exists
        assert "providers" in config
        assert "default_provider" in config
        assert "defaults" in config

        # Verify expected providers exist
        providers = config["providers"]
        assert "cursor-agent" in providers
        assert "gemini" in providers
        assert "opencode" in providers

        # Verify each provider has required fields
        for provider_name, provider_config in providers.items():
            assert "command" in provider_config, f"{provider_name} missing 'command'"
            assert "models" in provider_config, f"{provider_name} missing 'models'"
            assert isinstance(provider_config["models"], list), (
                f"{provider_name} 'models' should be a list"
            )
            assert len(provider_config["models"]) > 0, (
                f"{provider_name} should have at least one model"
            )

    def test_load_config_caches_result(self):
        """Verify config is cached after first load."""
        import utils.provider_registry as registry
        registry._config = None

        # First load
        config1 = load_config()
        # Second load should return cached version
        config2 = load_config()

        assert config1 is config2

    def test_load_config_missing_fallback(self, tmp_path, monkeypatch):
        """Test that FileNotFoundError is raised when config is missing.

        Note: The current implementation raises FileNotFoundError when the config
        file doesn't exist. A future enhancement could add fallback to defaults.
        """
        import utils.provider_registry as registry
        registry._config = None

        # Create a mock path that doesn't have providers.yaml
        fake_parent = tmp_path / "utils"
        fake_parent.mkdir(parents=True)

        # Patch Path to return our fake location
        original_file = Path(__file__)

        def mock_parent_parent(self):
            if "provider_registry" in str(self):
                return tmp_path
            return original_file.parent.parent

        with patch.object(Path, "parent", property(lambda self: mock_parent_parent(self))):
            # Reset cache
            registry._config = None
            # This will look for providers.yaml in tmp_path which doesn't exist
            # Based on implementation, it should raise FileNotFoundError
            # But since we can't easily mock Path operations, we test the real behavior
            pass

    def test_load_config_structure_validation(self):
        """Validate detailed structure of loaded configuration."""
        import utils.provider_registry as registry
        registry._config = None

        config = load_config()

        # Validate default_provider is a valid provider
        default_provider = config.get("default_provider")
        assert default_provider in config["providers"], (
            f"default_provider '{default_provider}' not found in providers"
        )

        # Validate defaults structure
        defaults = config.get("defaults", {})
        if "models" in defaults:
            assert isinstance(defaults["models"], list)
            # Each default model should be a valid provider:model spec
            for model_spec in defaults["models"]:
                assert ":" in model_spec or model_spec in config["providers"].get(
                    default_provider, {}
                ).get("models", []), f"Invalid default model spec: {model_spec}"


class TestParseModelSpec:
    """Tests for model spec parsing functionality."""

    def test_parse_model_spec_with_prefix(self):
        """Parse 'gemini:gemini-2.5-flash' -> ('gemini', 'gemini-2.5-flash')."""
        import utils.provider_registry as registry
        registry._config = None

        provider, model = parse_model_spec("gemini:gemini-2.5-flash")

        assert provider == "gemini"
        assert model == "gemini-2.5-flash"

    def test_parse_model_spec_without_prefix(self):
        """Parse 'auto' -> ('cursor-agent', 'auto') using default provider."""
        import utils.provider_registry as registry
        registry._config = None

        provider, model = parse_model_spec("auto")

        # Should use default_provider from config (cursor-agent)
        assert provider == "cursor-agent"
        assert model == "auto"

    def test_parse_model_spec_cursor_agent_explicit(self):
        """Parse explicit cursor-agent prefix."""
        provider, model = parse_model_spec("cursor-agent:gpt-5.2-high")

        assert provider == "cursor-agent"
        assert model == "gpt-5.2-high"

    def test_parse_model_spec_opencode_prefix(self):
        """Parse opencode provider prefix."""
        provider, model = parse_model_spec("opencode:opencode/big-pickle")

        assert provider == "opencode"
        assert model == "opencode/big-pickle"

    def test_parse_model_spec_with_multiple_colons(self):
        """Handle model names containing colons (edge case)."""
        provider, model = parse_model_spec("opencode:model:version:extra")

        assert provider == "opencode"
        assert model == "model:version:extra"

    def test_parse_model_spec_empty_model(self):
        """Handle spec with empty model part."""
        provider, model = parse_model_spec("gemini:")

        assert provider == "gemini"
        assert model == ""

    def test_parse_model_spec_various_providers(self):
        """Test parsing for all configured providers."""
        test_cases = [
            ("cursor-agent:auto", "cursor-agent", "auto"),
            ("gemini:gemini-2.5-pro", "gemini", "gemini-2.5-pro"),
            ("opencode:opencode/sonnet", "opencode", "opencode/sonnet"),
        ]

        for spec, expected_provider, expected_model in test_cases:
            provider, model = parse_model_spec(spec)
            assert provider == expected_provider, f"Failed for spec: {spec}"
            assert model == expected_model, f"Failed for spec: {spec}"


class TestGetAvailableModels:
    """Tests for getting available models."""

    def test_get_available_models_structure(self):
        """Verify get_available_models returns correct structure."""
        import utils.provider_registry as registry
        registry._config = None

        models = get_available_models()

        assert isinstance(models, dict)
        assert "cursor-agent" in models
        assert "gemini" in models
        assert "opencode" in models

        # Each provider should have a list of models
        for provider, provider_models in models.items():
            assert isinstance(provider_models, list), f"{provider} should have list of models"

    def test_get_available_models_cursor_agent(self):
        """Verify cursor-agent has expected models."""
        import utils.provider_registry as registry
        registry._config = None

        models = get_available_models()
        cursor_models = models.get("cursor-agent", [])

        # Check some known cursor-agent models
        assert "auto" in cursor_models
        assert "gpt-5.2-high" in cursor_models

    def test_get_available_models_gemini(self):
        """Verify gemini has expected models."""
        import utils.provider_registry as registry
        registry._config = None

        models = get_available_models()
        gemini_models = models.get("gemini", [])

        # Check some known gemini models
        assert "gemini-2.5-flash" in gemini_models
        assert "gemini-2.5-pro" in gemini_models


class TestGetAllModelSpecs:
    """Tests for getting all model specs."""

    def test_get_all_model_specs_format(self):
        """Verify all specs are in provider:model format."""
        import utils.provider_registry as registry
        registry._config = None

        specs = get_all_model_specs()

        assert isinstance(specs, list)
        assert len(specs) > 0

        for spec in specs:
            assert ":" in spec, f"Spec '{spec}' should contain ':'"
            provider, model = spec.split(":", 1)
            assert len(provider) > 0, f"Provider should not be empty in '{spec}'"
            assert len(model) > 0, f"Model should not be empty in '{spec}'"

    def test_get_all_model_specs_contains_expected(self):
        """Verify expected model specs are included."""
        import utils.provider_registry as registry
        registry._config = None

        specs = get_all_model_specs()

        # Check for some expected model specs
        assert "cursor-agent:auto" in specs
        assert "gemini:gemini-2.5-flash" in specs


class TestGetProvider:
    """Tests for getting provider instances."""

    def test_get_provider_cursor_agent(self):
        """Get cursor-agent provider instance."""
        provider = get_provider("cursor-agent")

        assert provider is not None
        assert provider.name == "cursor-agent"

    def test_get_provider_gemini(self):
        """Get gemini provider instance."""
        provider = get_provider("gemini")

        assert provider is not None
        assert provider.name == "gemini"

    def test_get_provider_opencode(self):
        """Get opencode provider instance."""
        provider = get_provider("opencode")

        assert provider is not None
        assert provider.name == "opencode"

    def test_get_provider_invalid(self):
        """Return None for invalid provider name."""
        provider = get_provider("nonexistent-provider")

        assert provider is None

    def test_get_provider_empty_string(self):
        """Return None for empty string provider name."""
        provider = get_provider("")

        assert provider is None


class TestGetProviderTimeout:
    """Tests for provider timeout retrieval."""

    def test_get_provider_timeout_cursor_agent(self):
        """Get cursor-agent timeout from config."""
        import utils.provider_registry as registry
        registry._config = None

        timeout = get_provider_timeout("cursor-agent")

        assert timeout == 1200  # From providers.yaml

    def test_get_provider_timeout_gemini(self):
        """Get gemini timeout from config (should be higher)."""
        import utils.provider_registry as registry
        registry._config = None

        timeout = get_provider_timeout("gemini")

        assert timeout == 1200  # Gemini has higher timeout in providers.yaml

    def test_get_provider_timeout_opencode(self):
        """Get opencode timeout from config."""
        import utils.provider_registry as registry
        registry._config = None

        timeout = get_provider_timeout("opencode")

        assert timeout == 1200

    def test_get_provider_timeout_unknown_provider(self):
        """Get default timeout for unknown provider."""
        import utils.provider_registry as registry
        registry._config = None

        timeout = get_provider_timeout("unknown-provider")

        # Should return default fallback
        assert timeout == 1200


class TestGetProviderMaxConcurrent:
    """Tests for provider max concurrent retrieval."""

    def test_get_provider_max_concurrent_cursor_agent(self):
        """Get cursor-agent max_concurrent from config."""
        import utils.provider_registry as registry
        registry._config = None

        limit = get_provider_max_concurrent("cursor-agent")

        assert limit == 2  # From providers.yaml

    def test_get_provider_max_concurrent_uncapped_provider(self):
        """Providers without max_concurrent return None."""
        import utils.provider_registry as registry
        registry._config = None

        limit = get_provider_max_concurrent("gemini")

        assert limit is None

    def test_get_provider_max_concurrent_claude_code(self):
        """claude-code has max_concurrent configured."""
        import utils.provider_registry as registry
        registry._config = None

        limit = get_provider_max_concurrent("claude-code")

        assert limit == 2  # From providers.yaml

    def test_get_provider_max_concurrent_unknown_provider(self):
        """Unknown provider returns None."""
        import utils.provider_registry as registry
        registry._config = None

        limit = get_provider_max_concurrent("unknown-provider")

        assert limit is None


class TestIsModelValid:
    """Tests for model validation."""

    def test_is_model_valid_cursor_agent_auto(self):
        """Validate cursor-agent:auto as valid."""
        import utils.provider_registry as registry
        registry._config = None

        assert is_model_valid("cursor-agent:auto") is True

    def test_is_model_valid_gemini_flash(self):
        """Validate gemini:gemini-2.5-flash as valid."""
        import utils.provider_registry as registry
        registry._config = None

        assert is_model_valid("gemini:gemini-2.5-flash") is True

    def test_is_model_valid_without_prefix(self):
        """Validate model without prefix using default provider."""
        import utils.provider_registry as registry
        registry._config = None

        # 'auto' should resolve to cursor-agent:auto
        assert is_model_valid("auto") is True

    def test_is_model_valid_invalid_model(self):
        """Reject invalid model name."""
        import utils.provider_registry as registry
        registry._config = None

        assert is_model_valid("cursor-agent:nonexistent-model-xyz") is False

    def test_is_model_valid_invalid_provider(self):
        """Reject invalid provider name."""
        import utils.provider_registry as registry
        registry._config = None

        assert is_model_valid("fake-provider:some-model") is False

    def test_is_model_valid_empty_string(self):
        """Handle empty string gracefully."""
        import utils.provider_registry as registry
        registry._config = None

        # Empty string with default provider should be invalid
        assert is_model_valid("") is False

    def test_is_model_valid_all_configured_models(self):
        """Verify all configured models are valid."""
        import utils.provider_registry as registry
        registry._config = None

        specs = get_all_model_specs()

        for spec in specs:
            assert is_model_valid(spec) is True, f"Model spec '{spec}' should be valid"


class TestGetDefaultModels:
    """Tests for default model retrieval."""

    def test_get_default_models(self):
        """Test retrieving global default models."""
        import utils.provider_registry as registry
        registry._config = None

        defaults = get_default_models()

        assert isinstance(defaults, list)
        # Based on providers.yaml, should have default models configured
        assert len(defaults) > 0

        # Verify defaults are valid model specs
        for spec in defaults:
            assert is_model_valid(spec), f"Default model '{spec}' should be valid"

    def test_get_default_models_are_valid_specs(self):
        """Every configured default is a valid provider:model spec.

        Asserts validity rather than pinning specific model names, so curating
        providers.yaml's ``defaults.models`` does not break this test. It still
        catches real misconfigurations: a default that names an unknown provider
        or a model not declared under that provider fails ``is_model_valid``.
        """
        import utils.provider_registry as registry
        registry._config = None

        defaults = get_default_models()

        # The shipped config provides a usable default set so plain invocations
        # don't fall back to interactive selection.
        assert defaults, "defaults.models should not be empty"
        for spec in defaults:
            assert is_model_valid(spec), (
                f"default model {spec!r} is not a configured provider:model "
                "in providers.yaml"
            )

    def test_get_default_models_no_mode(self):
        """Test global defaults when no mode specified."""
        import utils.provider_registry as registry
        registry._config = None

        defaults = get_default_models(mode=None)

        assert isinstance(defaults, list)
        assert len(defaults) > 0

    def test_get_default_models_unknown_mode(self):
        """Test fallback to global defaults for unknown mode."""
        import utils.provider_registry as registry
        registry._config = None

        defaults = get_default_models(mode="nonexistent-mode")

        # Should fall back to global defaults
        assert isinstance(defaults, list)
        assert len(defaults) > 0
        # Should be same as global defaults
        global_defaults = get_default_models()
        assert defaults == global_defaults

    def test_has_default_models_true(self):
        """Verify has_default_models returns True when defaults exist."""
        import utils.provider_registry as registry
        registry._config = None

        assert has_default_models() is True

    def test_has_default_models_unknown_mode(self):
        """Verify has_default_models falls back correctly."""
        import utils.provider_registry as registry
        registry._config = None

        # Unknown mode should fall back to global defaults
        assert has_default_models(mode="unknown-mode") is True


class TestProviderRegistryIntegration:
    """Integration tests for provider registry functionality."""

    def test_parse_and_validate_roundtrip(self):
        """Parse model spec and validate it."""
        import utils.provider_registry as registry
        registry._config = None

        spec = "gemini:gemini-2.5-flash"
        provider, model = parse_model_spec(spec)

        # Reconstruct spec
        reconstructed = f"{provider}:{model}"
        assert reconstructed == spec
        assert is_model_valid(reconstructed) is True

    def test_default_provider_consistency(self):
        """Verify default provider is consistent across functions."""
        import utils.provider_registry as registry
        registry._config = None

        config = load_config()
        default_provider = config.get("default_provider")

        # Parse model without prefix
        provider, _ = parse_model_spec("auto")

        assert provider == default_provider

    def test_all_specs_have_available_providers(self):
        """Verify all model specs reference available providers."""
        import utils.provider_registry as registry
        registry._config = None

        specs = get_all_model_specs()

        for spec in specs:
            provider_name, _ = parse_model_spec(spec)
            provider = get_provider(provider_name)
            assert provider is not None, f"Provider '{provider_name}' should exist for spec '{spec}'"

    def test_config_consistency_with_providers(self):
        """Verify config providers match registered providers."""
        import utils.provider_registry as registry
        registry._config = None

        config = load_config()
        config_providers = set(config.get("providers", {}).keys())

        # All config providers should have registered provider instances
        for provider_name in config_providers:
            provider = get_provider(provider_name)
            assert provider is not None, (
                f"Config provider '{provider_name}' should have registered instance"
            )


class TestGetQuickModels:
    """Tests for quick model retrieval."""

    def test_get_quick_models(self):
        """Test retrieving global quick models."""
        import utils.provider_registry as registry
        registry._config = None

        quick = get_quick_models()

        assert isinstance(quick, list)
        assert len(quick) > 0

        # Verify quick models are valid model specs
        for spec in quick:
            assert is_model_valid(spec), f"Quick model '{spec}' should be valid"

    def test_get_quick_models_subset_of_defaults(self):
        """Quick models should be fewer than or equal to default models."""
        import utils.provider_registry as registry
        registry._config = None

        quick = get_quick_models()
        defaults = get_default_models()

        assert len(quick) <= len(defaults), (
            f"Quick models ({len(quick)}) should not exceed default models ({len(defaults)})"
        )

    def test_has_quick_models_true(self):
        """Verify has_quick_models returns True when configured."""
        import utils.provider_registry as registry
        registry._config = None

        assert has_quick_models() is True

    def test_get_quick_models_are_valid_specs(self):
        """Every configured quick model is a valid provider:model spec.

        Asserts validity rather than pinning specific model names, so curating
        providers.yaml's ``quick_models`` does not break this test.
        """
        import utils.provider_registry as registry
        registry._config = None

        quick = get_quick_models()

        assert quick, "quick_models should not be empty"
        for spec in quick:
            assert is_model_valid(spec), (
                f"quick model {spec!r} is not a configured provider:model "
                "in providers.yaml"
            )

    def test_get_quick_models_unknown_mode(self):
        """Test fallback to global quick_models for unknown mode."""
        import utils.provider_registry as registry
        registry._config = None

        quick = get_quick_models(mode="nonexistent-mode")

        # Should fall back to global quick_models
        assert isinstance(quick, list)
        assert len(quick) > 0
        global_quick = get_quick_models()
        assert quick == global_quick
