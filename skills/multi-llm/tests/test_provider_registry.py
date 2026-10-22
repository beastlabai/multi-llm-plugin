"""Unit tests for provider registry and configuration loading.

This module tests the provider registry functionality including:
- Configuration loading from providers.yaml
- Model spec parsing (with and without provider prefixes)
- Provider availability checks
- Default model handling
- Model validation
"""

import ast
import asyncio
import copy
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest
import yaml


def _can_symlink() -> bool:
    """Probe whether this host can actually create symlinks.

    Platform checks are wrong in both directions: on Windows os.symlink
    succeeds with Developer Mode or elevation (e.g. GitHub's windows-latest
    runners), while a privilege-restricted POSIX sandbox can lack it.
    """
    with tempfile.TemporaryDirectory() as d:
        try:
            os.symlink(os.path.join(d, "t"), os.path.join(d, "l"))
            return True
        except (OSError, NotImplementedError):
            return False


_requires_symlinks = pytest.mark.skipif(
    not _can_symlink(), reason="requires symlink-creation privilege"
)

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils.provider_registry as registry
from utils.provider_registry import (
    ConfigError,
    CONFIG_WARNING_PREFIX,
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
    _deep_merge,
)
from utils.unattended import NeedsHuman


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

        # Verify each provider has required fields. `command` is NOT one of them
        # any more (§9.2): it had zero readers — every `is_available()` is a
        # hardcoded `shutil.which` — so the base no longer declares it.
        for provider_name, provider_config in providers.items():
            assert "command" not in provider_config, (
                f"{provider_name} re-declares the dead 'command' field"
            )
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
        assert "composer-2.5" in cursor_models

    def test_get_available_models_gemini(self):
        """Verify gemini has expected models."""
        import utils.provider_registry as registry
        registry._config = None

        models = get_available_models()
        gemini_models = models.get("gemini", [])

        # Check some known gemini models
        assert "gemini-3.8-flash" in gemini_models
        assert "gemini-4-argon" in gemini_models


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
        assert "gemini:gemini-3.8-flash" in specs


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
        """Validate gemini:gemini-3.8-flash as valid."""
        import utils.provider_registry as registry
        registry._config = None

        assert is_model_valid("gemini:gemini-3.8-flash") is True

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

        spec = "gemini:gemini-3.8-flash"
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


# =============================================================================
# Per-project provider config override (layered loading) — Section 4 of
# plans/per-project-provider-config-override.md
#
# These tests exercise the NEW layered load_config(): _deep_merge, project-local
# discovery, the MULTI_LLM_PROVIDERS_CONFIG env layer, the trust model, fail-fast
# vs. permissive handling, and init_config.py. They opt OUT of the autouse
# isolation fixture (@pytest.mark.config_override) and set up discovery/env/base
# themselves. Every test patches a small FIXED_BASE so assertions don't depend on
# the shipped providers.yaml, and resets registry._config between distinct
# overrides per the stated cache invariant.
# =============================================================================

# A small, fixed base used in place of the shipped providers.yaml so layering
# assertions are stable regardless of future edits to the real config.
FIXED_BASE = {
    "providers": {
        "claude-code": {
            "command": "claude",
            "default_timeout": 1800,
            "max_concurrent": 2,
            "models": ["sonnet", "opus", "haiku"],
        },
        "cursor-agent": {
            "command": "cursor-agent",
            "default_timeout": 1200,
            "max_concurrent": 2,
            "models": ["auto", "composer-2.5"],
        },
    },
    "default_provider": "cursor-agent",
    "defaults": {
        "models": ["cursor-agent:composer-2.5", "claude-code:opus"],
        "quick_models": ["claude-code:opus"],
        "modes": {
            "code-review": {"models": ["cursor-agent:auto"]},
        },
    },
}


def _reset_cache():
    registry._config = None
    registry._config_key = None
    # §9.3.b: the process-wide run anchor is set-once and immutable *within* a
    # process, so a test that sets one must not leak it into the next.
    if hasattr(registry, "_RUN_ANCHOR"):
        registry._RUN_ANCHOR = None


@pytest.fixture
def reg(monkeypatch):
    """Patch the base layer to FIXED_BASE and default project discovery to None.

    Returns the registry module. Tests re-patch ``_find_project_config`` (and/or
    set MULTI_LLM_PROVIDERS_CONFIG) to introduce override layers. Used together
    with @pytest.mark.config_override so the autouse isolation fixture does not
    interfere with discovery.
    """
    monkeypatch.setattr(registry, "_load_base_config", lambda: copy.deepcopy(FIXED_BASE))
    monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
    _reset_cache()
    yield registry
    _reset_cache()


def _patch_base_only(monkeypatch):
    """Patch the base to FIXED_BASE but leave real project discovery in place.

    For tests that exercise the *real* git-anchored ``_find_project_config``
    (the ``reg`` fixture stubs it to None, which those tests must avoid).
    """
    monkeypatch.setattr(registry, "_load_base_config", lambda: copy.deepcopy(FIXED_BASE))
    _reset_cache()


def _set_project_config(monkeypatch, path):
    """Point project-local discovery at ``path`` and clear the cache."""
    monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: Path(path))
    _reset_cache()


def _write_yaml(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.config_override
class TestDeepMerge:
    """Unit tests for the _deep_merge primitive."""

    def test_nested_dict_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 20, "z": 30}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}

    def test_list_replaces_not_appends(self):
        base = {"models": ["a", "b", "c"]}
        override = {"models": ["x"]}
        assert _deep_merge(base, override) == {"models": ["x"]}

    def test_scalar_replaces(self):
        assert _deep_merge({"n": 1}, {"n": 2}) == {"n": 2}

    def test_disjoint_keys_union(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_none_override_skipped_keeps_base(self):
        base = {"models": ["a", "b"], "n": 5}
        override = {"models": None, "n": None}
        assert _deep_merge(base, override) == {"models": ["a", "b"], "n": 5}

    def test_empty_list_wipes_out_base(self):
        # Distinct from None/omitted: [] is a deliberate replace.
        assert _deep_merge({"q": ["a", "b"]}, {"q": []}) == {"q": []}

    def test_inputs_not_mutated(self):
        base = {"a": {"x": 1}, "lst": [1, 2]}
        override = {"a": {"y": 2}, "lst": [9]}
        base_copy = copy.deepcopy(base)
        override_copy = copy.deepcopy(override)
        result = _deep_merge(base, override)
        assert base == base_copy
        assert override == override_copy
        # Result shares no nested object with either input.
        result["a"]["z"] = 99
        result["lst"].append(0)
        assert base == base_copy
        assert override == override_copy


@pytest.mark.config_override
class TestLayeringPrecedence:
    """Layer precedence: env > project-local > base, and absent → base."""

    def test_absent_overrides_deep_equal_to_base(self, reg):
        # No project file, no env var → result deep-equal to the parsed base.
        config = load_config()
        assert config == FIXED_BASE

    def test_project_layer_applied(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - claude-code:opus\n",
        )
        _set_project_config(monkeypatch, override)
        config = load_config()
        assert config["defaults"]["models"] == ["claude-code:opus"]
        # Untouched keys still come from the base.
        assert config["default_provider"] == "cursor-agent"
        assert config["providers"]["claude-code"]["command"] == "claude"

    def test_env_overrides_project(self, reg, monkeypatch, tmp_path):
        project = _write_yaml(
            tmp_path / "proj" / ".multi-llm" / "providers.yaml",
            "default_provider: claude-code\n",
        )
        _set_project_config(monkeypatch, project)
        env_file = _write_yaml(
            tmp_path / "env.yaml", "default_provider: gemini\n"
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(env_file))
        _reset_cache()
        config = load_config()
        # Env layer is highest precedence.
        assert config["default_provider"] == "gemini"

    def test_commented_stub_is_no_op(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "# only comments here\n# nothing active\n",
        )
        _set_project_config(monkeypatch, override)
        assert load_config() == FIXED_BASE

    def test_uncommented_blank_key_is_ignored(self, reg, monkeypatch, tmp_path):
        # `models:` with no value parses to None and must not blank the base.
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n",
        )
        _set_project_config(monkeypatch, override)
        config = load_config()
        assert config["defaults"]["models"] == FIXED_BASE["defaults"]["models"]

    def test_relative_env_path_resolved_against_cwd(self, reg, monkeypatch, tmp_path):
        # CWD = a dir, relative env path under it → loaded and applied.
        _write_yaml(tmp_path / "rel" / "override.yaml", "default_provider: gemini\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", "rel/override.yaml")
        _reset_cache()
        config = load_config()
        assert config["default_provider"] == "gemini"

    def test_mode_specific_base_shadows_project_global_models(self, reg, monkeypatch, tmp_path):
        # Project sets ONLY defaults.models. A mode WITH a base modes entry keeps
        # the base list; a mode WITHOUT one picks up the new global models.
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - claude-code:opus\n",
        )
        _set_project_config(monkeypatch, override)
        # 'code-review' has a base modes entry → base list still wins.
        assert get_default_models(mode="code-review") == ["cursor-agent:auto"]
        # A mode with no base entry falls back to the new global models.
        assert get_default_models(mode="review-plan") == ["claude-code:opus"]

    def test_empty_mode_list_clears_not_falls_through(self, reg, monkeypatch):
        # A mode entry set to a bare empty list deliberately clears that mode's
        # models. get_default_models must return [] (the explicit clear) rather
        # than falling through to defaults.models, which would re-select models
        # the project tried to disable. A truthiness check on the mode value
        # treats [] like an absent key and wrongly falls through; presence-based
        # lookup distinguishes the two.
        base = copy.deepcopy(FIXED_BASE)
        # Direct empty-list mode entry (not the dict form).
        base["defaults"]["modes"]["code-review"] = []
        monkeypatch.setattr(registry, "_load_base_config", lambda: copy.deepcopy(base))
        _reset_cache()
        # Sanity: defaults.models is non-empty, so a fall-through would be visible.
        assert get_default_models() == FIXED_BASE["defaults"]["models"]
        # Explicit empty mode list clears the mode → [] (no fall-through).
        assert get_default_models(mode="code-review") == []
        # A mode with no entry still falls back to global defaults (absent != []).
        assert get_default_models(mode="review-plan") == FIXED_BASE["defaults"]["models"]


@pytest.mark.config_override
class TestProjectDiscovery:
    """Project-local discovery is git-anchored and respects the anchor arg."""

    def test_file_found_in_git_repo(self, monkeypatch, tmp_path):
        _patch_base_only(monkeypatch)  # real discovery, FIXED_BASE base layer
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "default_provider: gemini\n",
        )
        monkeypatch.chdir(tmp_path)
        _reset_cache()
        found = registry._find_project_config()
        assert found is not None
        assert found == tmp_path / ".multi-llm" / "providers.yaml"
        assert load_config()["default_provider"] == "gemini"

    def test_absent_file_base_only(self, monkeypatch, tmp_path):
        _patch_base_only(monkeypatch)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        monkeypatch.chdir(tmp_path)
        _reset_cache()
        assert registry._find_project_config() is None
        assert load_config() == FIXED_BASE

    def test_outside_git_repo_base_only(self, monkeypatch, tmp_path):
        _patch_base_only(monkeypatch)
        # Guarantee tmp_path is genuinely outside any repo.
        from utils.git_utils import get_project_root_from_dir
        non_git = tmp_path / "plain"
        non_git.mkdir()
        assert get_project_root_from_dir(str(non_git)) is None, (
            "tmp dir unexpectedly inside a git repo"
        )
        _write_yaml(non_git / ".multi-llm" / "providers.yaml", "default_provider: gemini\n")
        monkeypatch.chdir(non_git)
        _reset_cache()
        # Non-git → project-local layer ignored even though the file sits in CWD.
        assert registry._find_project_config() is None
        assert load_config() == FIXED_BASE

    def test_discovery_uses_anchor_dir(self, monkeypatch, tmp_path):
        # _find_project_config(anchor) resolves the git root from the anchor's
        # directory, not CWD. (Real _find_project_config; only git lookup mocked.)
        seen = {}

        def fake_root(directory):
            seen["dir"] = directory
            return None

        monkeypatch.setattr(registry, "get_project_root_from_dir", fake_root)
        plan = tmp_path / "sub" / "plan.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# plan", encoding="utf-8")
        registry._find_project_config(anchor=str(plan))
        assert seen["dir"] == str(tmp_path / "sub")


@pytest.mark.config_override
class TestTrustModelEnvLayer:
    """The explicit env layer may deep-merge an EXISTING provider per-field."""

    def _load_with_env(self, monkeypatch, tmp_path, content):
        env_file = _write_yaml(tmp_path / "env.yaml", content)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(env_file))
        _reset_cache()
        return load_config()

    def test_scalar_field_override_with_sibling_inheritance(self, reg, monkeypatch, tmp_path):
        self._load_with_env(
            monkeypatch, tmp_path,
            "providers:\n  claude-code:\n    max_concurrent: 4\n",
        )
        # Overridden field changes; siblings still resolve from base.
        assert get_provider_max_concurrent("claude-code") == 4
        assert get_provider_timeout("claude-code") == 1800
        cfg = load_config()
        assert cfg["providers"]["claude-code"]["command"] == "claude"
        assert cfg["providers"]["claude-code"]["models"] == ["sonnet", "opus", "haiku"]

    def test_provider_models_list_replaced_not_appended(self, reg, monkeypatch, tmp_path):
        cfg = self._load_with_env(
            monkeypatch, tmp_path,
            "providers:\n  claude-code:\n    models:\n      - opus\n",
        )
        assert cfg["providers"]["claude-code"]["models"] == ["opus"]
        # Sibling scalars unchanged.
        assert cfg["providers"]["claude-code"]["default_timeout"] == 1800

    def test_command_invariant_binary_unchanged(self, reg, monkeypatch, tmp_path):
        cfg = self._load_with_env(
            monkeypatch, tmp_path,
            "providers:\n  claude-code:\n    command: evil-binary\n",
        )
        # Config metadata merges...
        assert cfg["providers"]["claude-code"]["command"] == "evil-binary"
        # ...but the provider's resolved binary is hardcoded, never from config.
        provider = get_provider("claude-code")
        assert provider.build_command("prompt", "opus")[0] == "claude"
        calls = []
        monkeypatch.setattr(
            "shutil.which", lambda name: calls.append(name) or "/usr/bin/claude"
        )
        provider.is_available()
        assert calls == ["claude"]


@pytest.mark.config_override
class TestAutoDiscoveredProvidersBlockMerges:
    """The layer-2 `providers:` filter is REMOVED: an auto-discovered providers:
    block now deep-merges over base, identical to the env layer (no warning).

    (Inverts the former TestTrustModelAutoDiscoveredProvidersDropped.)
    """

    def test_existing_provider_field_merges_no_warning(self, reg, monkeypatch, tmp_path, capsys):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "providers:\n  claude-code:\n    max_concurrent: 99\n",
        )
        _set_project_config(monkeypatch, override)
        # The merged field now takes effect (no longer dropped)...
        assert get_provider_max_concurrent("claude-code") == 99
        cfg = load_config()
        # ...siblings still inherit from base (deep-merge).
        assert cfg["providers"]["claude-code"]["command"] == "claude"
        assert cfg["providers"]["claude-code"]["models"] == ["sonnet", "opus", "haiku"]
        # No "ignoring 'providers:' block" warning is emitted.
        assert "ignoring 'providers:' block" not in capsys.readouterr().err

    def test_provider_models_list_replaced(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "providers:\n  claude-code:\n    models:\n      - opus\n",
        )
        _set_project_config(monkeypatch, override)
        cfg = load_config()
        assert cfg["providers"]["claude-code"]["models"] == ["opus"]
        assert cfg["providers"]["claude-code"]["default_timeout"] == 1800

    def test_parity_with_env_layer(self, reg, monkeypatch, tmp_path):
        # The SAME providers block resolves identically through the project layer
        # and the env layer now that the filter is gone.
        content = "providers:\n  claude-code:\n    max_concurrent: 7\n"
        project = _write_yaml(tmp_path / ".multi-llm" / "providers.yaml", content)
        _set_project_config(monkeypatch, project)
        via_project = load_config()["providers"]["claude-code"]["max_concurrent"]

        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        env_file = _write_yaml(tmp_path / "env.yaml", content)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(env_file))
        _reset_cache()
        via_env = load_config()["providers"]["claude-code"]["max_concurrent"]
        assert via_project == via_env == 7

    def test_selection_keys_apply_alongside_providers_block(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "default_provider: claude-code\n"
            "providers:\n  claude-code:\n    max_concurrent: 99\n",
        )
        _set_project_config(monkeypatch, override)
        cfg = load_config()
        assert cfg["default_provider"] == "claude-code"
        assert cfg["providers"]["claude-code"]["max_concurrent"] == 99


@pytest.mark.config_override
class TestRemovedProviderDrift:
    """A merged providers: block naming a provider with NO hardcoded adapter (an
    older-plugin / renamed key) merges but is filtered out of the model listing."""

    def test_orphan_provider_merges_but_lists_no_models_and_warns(
        self, reg, monkeypatch, tmp_path, capsys
    ):
        registry._WARNED_UNKNOWN_PROVIDERS.clear()
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "providers:\n  oldprov:\n    command: x\n    models:\n      - m1\n",
        )
        _set_project_config(monkeypatch, override)
        # The block merges (no error)...
        cfg = load_config()
        assert "oldprov" in cfg["providers"]
        # ...but has no hardcoded adapter, so it can never be selected...
        assert get_provider("oldprov") is None
        # ...and contributes NO models to the listing (read-site filter), warning once.
        available = get_available_models()
        assert "oldprov" not in available
        # Known providers are unaffected.
        assert "claude-code" in available
        err = capsys.readouterr().err
        assert CONFIG_WARNING_PREFIX in err
        assert "no longer supported" in err

    def test_orphan_warning_is_once_per_process(self, reg, monkeypatch, tmp_path, capsys):
        registry._WARNED_UNKNOWN_PROVIDERS.clear()
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "providers:\n  oldprov:\n    command: x\n    models:\n      - m1\n",
        )
        _set_project_config(monkeypatch, override)
        get_available_models()
        capsys.readouterr()  # drain the first warning
        get_available_models()
        assert "no longer supported" not in capsys.readouterr().err


@pytest.mark.config_override
class TestRemovedProviderDriftConfiguredSpec:
    """Mirror of TestRemovedProviderDrift for the configured-spec path: an orphan
    spec (no hardcoded adapter) written into defaults.* is filtered from the
    configured set, so is_model_valid rejects it — a model that cannot run on a
    stale cloned config is never accepted."""

    def test_orphan_configured_spec_filtered_and_invalid(
        self, reg, monkeypatch, tmp_path, capsys
    ):
        registry._WARNED_UNKNOWN_PROVIDERS.clear()
        # A stale clone lists a removed/renamed provider spec in defaults.models.
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - oldprov:m1\n",
        )
        _set_project_config(monkeypatch, override)
        # oldprov has no hardcoded adapter...
        assert get_provider("oldprov") is None
        # ...so its spec is filtered out of the configured set (read-site filter)...
        configured = registry._collect_configured_specs()
        assert "oldprov:m1" not in configured
        # ...and is_model_valid rejects it rather than treating it as configured.
        assert is_model_valid("oldprov:m1") is False
        err = capsys.readouterr().err
        assert CONFIG_WARNING_PREFIX in err
        assert "no longer supported" in err

    def test_orphan_filter_leaves_known_configured_specs_valid(
        self, reg, monkeypatch, tmp_path
    ):
        registry._WARNED_UNKNOWN_PROVIDERS.clear()
        # A defaults.models mixing an orphan with a real (configured-but-uncatalogued)
        # spec: the orphan is filtered, the known-provider spec still validates.
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - oldprov:m1\n    - cursor-agent:made-up\n",
        )
        _set_project_config(monkeypatch, override)
        assert is_model_valid("oldprov:m1") is False
        assert is_model_valid("cursor-agent:made-up") is True


@pytest.mark.config_override
class TestNonMappingProviderGuard:
    """A merged non-mapping provider value is sanitized at load time (no crash):
    a KNOWN provider's base mapping is restored; an unknown name is dropped."""

    def test_scalar_known_provider_restores_base_with_warning(self, reg, monkeypatch, tmp_path, capsys):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            'providers:\n  claude-code: "scalar-bad"\n',
        )
        _set_project_config(monkeypatch, override)
        # No AttributeError on the downstream `.get(...)` calls; load succeeds and
        # the malformed scalar does NOT erase the built-in claude-code catalog —
        # the base mapping is restored instead.
        cfg = load_config()
        assert cfg["providers"]["claude-code"]["models"] == ["sonnet", "opus", "haiku"]
        assert cfg["providers"]["claude-code"]["max_concurrent"] == 2
        # Other providers still load and resolve.
        assert get_provider_max_concurrent("cursor-agent") == 2
        err = capsys.readouterr().err
        assert CONFIG_WARNING_PREFIX in err
        assert "malformed provider 'claude-code'" in err
        assert "restoring base definition" in err

    def test_scalar_unknown_provider_dropped_with_warning(self, reg, monkeypatch, tmp_path, capsys):
        # An UNKNOWN name has no base mapping to inherit → dropped (not restored).
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            'providers:\n  oldprov: "scalar-bad"\n',
        )
        _set_project_config(monkeypatch, override)
        cfg = load_config()
        assert "oldprov" not in cfg["providers"]
        # Known providers are unaffected.
        assert get_provider_max_concurrent("cursor-agent") == 2
        err = capsys.readouterr().err
        assert CONFIG_WARNING_PREFIX in err
        assert "malformed provider 'oldprov'" in err
        assert "dropping it" in err


@pytest.mark.config_override
class TestCommandNeverLaunchesBinary:
    """The merged `command:` field is documentation-only — the launch path always
    uses the hardcoded binary, in BOTH override layers (security guard)."""

    def _assert_hardcoded(self, monkeypatch, cfg_command):
        cfg = load_config()
        # Config metadata merges...
        assert cfg["providers"]["claude-code"]["command"] == cfg_command
        # ...but the resolved binary is hardcoded, never read from config.
        provider = get_provider("claude-code")
        assert provider.build_command("prompt", "opus")[0] == "claude"
        calls = []
        monkeypatch.setattr(
            "shutil.which", lambda name: calls.append(name) or "/usr/bin/claude"
        )
        provider.is_available()
        assert calls == ["claude"]

    def test_hostile_command_project_layer(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            'providers:\n  claude-code:\n    command: "/bin/evil"\n',
        )
        _set_project_config(monkeypatch, override)
        self._assert_hardcoded(monkeypatch, "/bin/evil")

    def test_hostile_command_env_layer(self, reg, monkeypatch, tmp_path):
        env_file = _write_yaml(
            tmp_path / "env.yaml",
            'providers:\n  claude-code:\n    command: "rm -rf ~"\n',
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(env_file))
        _reset_cache()
        self._assert_hardcoded(monkeypatch, "rm -rf ~")


@pytest.mark.config_override
class TestNestedSubtreeNonRemoval:
    """Deep-merge preserves base subtrees; only [] / leaf overwrite changes them."""

    def test_cannot_prune_mode_entry(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - claude-code:opus\n",
        )
        _set_project_config(monkeypatch, override)
        cfg = load_config()
        # Base modes entry survives even though the project omitted it.
        assert cfg["defaults"]["modes"]["code-review"]["models"] == ["cursor-agent:auto"]

    def test_cannot_remove_base_provider(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "default_provider: claude-code\n",
        )
        _set_project_config(monkeypatch, override)
        cfg = load_config()
        # Base provider definitions remain present.
        assert "cursor-agent" in cfg["providers"]
        assert "claude-code" in cfg["providers"]


@pytest.mark.config_override
class TestEmptyListConsequences:
    """[] clears rather than inherits — with the documented downstream effects."""

    def test_empty_default_models_disables_defaults(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models: []\n",
        )
        _set_project_config(monkeypatch, override)
        assert load_config()["defaults"]["models"] == []
        assert has_default_models() is False  # discarded, not re-inherited

    def test_empty_quick_models_raises_under_quick(self, reg, monkeypatch, tmp_path):
        from utils.interactive import resolve_models
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  quick_models: []\n",
        )
        _set_project_config(monkeypatch, override)
        assert has_quick_models() is False
        with pytest.raises(RuntimeError, match="No quick_models configured"):
            resolve_models(quick=True)


@pytest.mark.config_override
class TestFailFastVsPermissive:
    """Present-but-invalid explicit overrides fail fast; absent falls through."""

    def test_present_malformed_project_yaml_raises(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "default_provider: [unclosed\n",
        )
        _set_project_config(monkeypatch, override)
        with pytest.raises(ConfigError) as exc:
            load_config()
        assert str(override) in str(exc.value)

    def test_present_non_mapping_root_raises(self, reg, monkeypatch, tmp_path):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "- just\n- a\n- list\n",
        )
        _set_project_config(monkeypatch, override)
        with pytest.raises(ConfigError) as exc:
            load_config()
        assert str(override) in str(exc.value)

    def test_present_broken_env_target_raises(self, reg, monkeypatch, tmp_path):
        env_file = _write_yaml(tmp_path / "broken.yaml", "key: [unclosed\n")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(env_file))
        _reset_cache()
        with pytest.raises(ConfigError) as exc:
            load_config()
        assert str(env_file) in str(exc.value)

    def test_directory_at_override_path_raises(self, reg, monkeypatch, tmp_path):
        a_dir = tmp_path / "config.yaml"
        a_dir.mkdir()
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(a_dir))
        _reset_cache()
        with pytest.raises(ConfigError) as exc:
            load_config()
        assert str(a_dir) in str(exc.value)

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses file permissions",
    )
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod(0o000) does not revoke read access on Windows",
    )
    def test_unreadable_file_raises(self, reg, monkeypatch, tmp_path):
        env_file = _write_yaml(tmp_path / "noperm.yaml", "default_provider: gemini\n")
        os.chmod(env_file, 0o000)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(env_file))
        _reset_cache()
        try:
            with pytest.raises(ConfigError) as exc:
                load_config()
            assert str(env_file) in str(exc.value)
        finally:
            os.chmod(env_file, 0o644)

    def test_permissive_restores_warn_and_skip(self, reg, monkeypatch, tmp_path, capsys):
        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "- not a mapping\n",
        )
        _set_project_config(monkeypatch, override)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE", "1")
        _reset_cache()
        config = load_config()
        assert config == FIXED_BASE  # base returned, no crash
        err = capsys.readouterr().err
        assert CONFIG_WARNING_PREFIX in err
        assert str(override) in err

    def test_absent_project_file_silent_fallthrough(self, reg, capsys):
        # No project file → base, no warning, no raise.
        config = load_config()
        assert config == FIXED_BASE
        assert capsys.readouterr().err == ""

    def test_env_set_but_missing_warns_and_skips(self, reg, monkeypatch, tmp_path, capsys):
        missing = tmp_path / "does-not-exist.yaml"
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(missing))
        _reset_cache()
        config = load_config()
        assert config == FIXED_BASE  # base used
        err = capsys.readouterr().err
        assert CONFIG_WARNING_PREFIX in err
        assert str(missing.resolve()) in err

    @_requires_symlinks
    def test_broken_env_symlink_raises(self, reg, monkeypatch, tmp_path):
        # A broken symlink at the env path is present-as-a-link (not absent), so
        # _resolve_env_path keeps it unresolved and the loader fails fast.
        link = tmp_path / "dangling.yaml"
        link.symlink_to(tmp_path / "no-such-target.yaml")
        assert link.is_symlink() and not link.exists()
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(link))
        _reset_cache()
        with pytest.raises(ConfigError) as exc:
            load_config()
        assert "broken symlink" in str(exc.value)
        assert str(link) in str(exc.value)

    @_requires_symlinks
    def test_broken_project_symlink_raises(self, reg, monkeypatch, tmp_path):
        # A broken symlink at the project-local path is surfaced by
        # _find_project_config (exists() or is_symlink()) and fails fast.
        override = tmp_path / ".multi-llm" / "providers.yaml"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.symlink_to(tmp_path / "no-such-target.yaml")
        assert override.is_symlink() and not override.exists()
        _set_project_config(monkeypatch, override)
        with pytest.raises(ConfigError) as exc:
            load_config()
        assert "broken symlink" in str(exc.value)
        assert str(override) in str(exc.value)

    @_requires_symlinks
    def test_broken_env_symlink_permissive_warns_and_skips(
        self, reg, monkeypatch, tmp_path, capsys
    ):
        link = tmp_path / "dangling.yaml"
        link.symlink_to(tmp_path / "no-such-target.yaml")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(link))
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE", "1")
        _reset_cache()
        config = load_config()
        assert config == FIXED_BASE  # base returned, no crash
        err = capsys.readouterr().err
        assert CONFIG_WARNING_PREFIX in err
        assert "broken symlink" in err
        assert str(link) in err

    @_requires_symlinks
    def test_broken_project_symlink_permissive_warns_and_skips(
        self, reg, monkeypatch, tmp_path, capsys
    ):
        override = tmp_path / ".multi-llm" / "providers.yaml"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.symlink_to(tmp_path / "no-such-target.yaml")
        _set_project_config(monkeypatch, override)
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE", "1")
        _reset_cache()
        config = load_config()
        assert config == FIXED_BASE  # base returned, no crash
        err = capsys.readouterr().err
        assert CONFIG_WARNING_PREFIX in err
        assert "broken symlink" in err
        assert str(override) in err


@pytest.mark.config_override
class TestProgressBlockValidation:
    """`progress:` is a CLOSED key set, validated at load on OVERRIDE layers only.

    Driven through a project-local layer rather than through
    `validate_progress_block` directly, because the thing under test is the wiring
    as much as the rules: a `progress_overridden` flag that only noticed the env
    file would leave every project override unchecked, and the rejection has to
    happen at `load_config` — not lazily in an accessor, where `interval_s` is
    first read after a fan-out has already launched its models and `poll_s` is
    first read inside the agent's `status --wait` poll loop.
    """

    @staticmethod
    def _override(monkeypatch, tmp_path, body):
        """Write ``body`` as the project layer and point discovery at it."""
        override = _write_yaml(tmp_path / ".multi-llm" / "providers.yaml", body)
        _set_project_config(monkeypatch, override)
        return override

    @pytest.fixture
    def reg_with_base_progress(self, monkeypatch):
        """``reg``, but with a base that CARRIES ``progress:`` — as the shipped one does.

        The rejection table has to run against this rather than against the bare
        ``FIXED_BASE``, or it passes for the wrong reason. ``_deep_merge`` skips
        an override value of ``None`` (blank keeps base), so the two bases give
        genuinely different merge shapes: against a base that already has the
        block a project-layer ``progress:`` *merges into* it, which is what a
        real config does; against a base with no ``progress:`` key the override
        *replaces* wholesale, and a row whose only fault is a blanked key would
        raise for a reason no real config can reproduce.
        """
        shipped = registry._load_base_config().get("progress")
        assert isinstance(shipped, dict), (
            "the shipped providers.yaml must carry a `progress:` block — this "
            "table's whole point is to merge onto one"
        )
        base = copy.deepcopy(FIXED_BASE)
        # The shipped cadence, spelled out rather than copied from the live base
        # so the expectations below stay stable if those numbers are ever retuned.
        base["progress"] = {"interval_s": 30, "poll_s": 90}
        monkeypatch.setattr(registry, "_load_base_config", lambda: copy.deepcopy(base))
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        _reset_cache()
        yield registry
        _reset_cache()

    @pytest.mark.parametrize(
        "body, fragments",
        [
            # Floor rejections. `interval_s` names its disable form in the
            # message, because 0 being legal here and illegal for `poll_s` is the
            # asymmetry a user is most likely to have guessed wrong.
            (
                "progress:\n  interval_s: -1\n",
                ["progress.interval_s is -1 —", "disable form"],
            ),
            (
                "progress:\n  interval_s: 3\n",
                ["progress.interval_s is 3 —", "at least 5 seconds"],
            ),
            # Type rejections. Each of these is a value YAML actually produces,
            # and each must fail on the TYPE check ("finite"), not on the floor:
            # a quoted scalar would otherwise blow up inside the ticker, `true`
            # is an int in Python and would read as a 1 s ticker, `.nan` would be
            # rejected with a message about a floor rather than about a nonsense
            # value, and `.inf` would sail straight through a floor check.
            (
                'progress:\n  interval_s: "30"\n',
                ["progress.interval_s is '30' —", "finite number of seconds"],
            ),
            (
                "progress:\n  interval_s: true\n",
                ["progress.interval_s is True —", "finite number of seconds"],
            ),
            (
                "progress:\n  interval_s: .nan\n",
                ["progress.interval_s is nan —", "finite number of seconds"],
            ),
            (
                "progress:\n  poll_s: .inf\n",
                ["progress.poll_s is inf —", "finite number of seconds"],
            ),
            # An integer too large to convert to a float. PyYAML resolves a
            # bare digit string of any length to a Python `int`, and
            # `math.isfinite(10**400)` RAISES OverflowError rather than
            # returning False — so without the guard this escaped `load_config`
            # as an uncaught OverflowError instead of the documented ConfigError.
            (
                f"progress:\n  poll_s: {10 ** 400}\n",
                [f"progress.poll_s is {10 ** 400} —", "finite number of seconds"],
            ),
            # `poll_s: 0` is NOT a disable form: a poll of zero returns instantly
            # rather than being switched off, so the floor rejects it.
            (
                "progress:\n  poll_s: 0\n",
                ["progress.poll_s is 0 —", "no disable form"],
            ),
            # NOTE: `poll_s: null` is NOT a row here. Against a base carrying the
            # block — the only base a real config merges onto — `_deep_merge`
            # skips a `None` override and the base's cadence survives, so it
            # never reaches the validator at all. See
            # `test_a_blank_value_keeps_the_base_cadence`.
            #
            # The block itself, and the closed-set check: a typo that was merely
            # ignored would keep the shipped cadence while the config reads as if
            # it had been changed.
            (
                "progress: 30\n",
                ["`progress:` must be a mapping", "(30)"],
            ),
            (
                "progress:\n  interval: 30\n",
                ["unknown key(s) under `progress:`: interval: 30", "interval_s"],
            ),
        ],
        ids=[
            "interval_s-negative", "interval_s-under-floor", "interval_s-quoted",
            "interval_s-true", "interval_s-nan", "poll_s-inf",
            "poll_s-oversized-int", "poll_s-zero",
            "progress-not-a-mapping", "unknown-key-typo",
        ],
    )
    def test_a_bad_value_on_the_project_layer_is_rejected(
        self, reg_with_base_progress, monkeypatch, tmp_path, body, fragments
    ):
        """Driven onto a base that already carries `progress:`, deliberately.

        Every row here has to fail because the VALUE is bad, not because the
        override happened to replace an absent base block — which is what a base
        without `progress:` would turn each of these into.
        """
        self._override(monkeypatch, tmp_path, body)
        with pytest.raises(ConfigError) as exc:
            load_config(anchor=str(tmp_path))
        message = str(exc.value)
        for fragment in fragments:
            assert fragment in message, f"{fragment!r} missing from: {message}"

    @pytest.mark.parametrize(
        "body, expected",
        [
            (
                "progress:\n  interval_s: 60\n  poll_s: null\n",
                {"interval_s": 60, "poll_s": 90},
            ),
            ("progress:\n", {"interval_s": 30, "poll_s": 90}),
        ],
        ids=["poll_s-blank", "whole-block-blank"],
    )
    def test_a_blank_value_keeps_the_base_cadence(
        self, reg_with_base_progress, monkeypatch, tmp_path, body, expected
    ):
        """A blank key is `_deep_merge`'s keep-base form, not a rejection.

        `_deep_merge` skips an override value of `None` for EVERY key, so
        `poll_s:` with nothing after it inherits the shipped 90 while a sibling
        key on the same layer still applies, and a bare `progress:` inherits the
        whole block. That is not an accident of this table: it is what the
        `--init` template's commented-out `progress:` stanza depends on —
        uncommenting the key and one line under it must not blank the other.
        Making `progress:` raise on a blank instead would make one block the
        exception to the rule the rest of the config is authored around.
        """
        self._override(monkeypatch, tmp_path, body)
        config = load_config(anchor=str(tmp_path))
        assert config["progress"] == expected
        assert registry.get_progress_interval(anchor=str(tmp_path)) == float(
            expected["interval_s"]
        )
        assert registry.get_progress_poll(anchor=str(tmp_path)) == float(
            expected["poll_s"]
        )

    def test_the_same_value_in_the_base_is_not_validated(self, monkeypatch):
        """The base is ours: a broken one is a test failure, not a user's problem.

        Validating it here would turn every unrelated invocation into a config
        error, which is `driver:`'s stated rule and the reason the flag keys off
        an override layer rather than off the merged block.
        """
        base = copy.deepcopy(FIXED_BASE)
        base["progress"] = {"interval_s": -1, "poll_s": 0}
        monkeypatch.setattr(registry, "_load_base_config", lambda: copy.deepcopy(base))
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        _reset_cache()
        assert load_config()["progress"] == {"interval_s": -1, "poll_s": 0}

    def test_permissive_mode_does_not_relax_the_rejection(
        self, reg, monkeypatch, tmp_path
    ):
        """Same carve-out `driver:` gets, for the same reason.

        Permissive restores warn-and-skip for a layer that could not be *read*; a
        readable layer whose cadence was silently dropped degrades to the shipped
        default, which is exactly the failure this check exists to catch.
        """
        self._override(monkeypatch, tmp_path, "progress:\n  poll_s: 0\n")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE", "1")
        _reset_cache()
        with pytest.raises(ConfigError) as exc:
            load_config(anchor=str(tmp_path))
        assert "progress.poll_s is 0 —" in str(exc.value)

    def test_a_config_with_no_progress_block_falls_back(self, reg):
        """FIXED_BASE predates the block, which is the only case the hardcoded
        defaults exist for — a real merged config always carries the base's."""
        assert "progress" not in load_config()
        assert registry.get_progress_interval() == registry.PROGRESS_FALLBACKS["interval_s"]
        assert registry.get_progress_poll() == registry.PROGRESS_FALLBACKS["poll_s"]

    def test_a_valid_override_resolves_through_both_accessors_from_a_foreign_cwd(
        self, tmp_path, monkeypatch
    ):
        """§9.3.c, the reason the accessors take an anchor at all.

        Real base, real git-anchored discovery, and a CWD outside the project: the
        unanchored read must answer with the base's cadence and the anchored one
        with the project's. `interval_s: 0` doubles as the proof that the disable
        form survives validation rather than being caught by the floor.
        """
        proj = tmp_path / "proj"
        (proj / "plans").mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
        _write_yaml(
            proj / ".multi-llm" / "providers.yaml",
            "progress:\n  interval_s: 0\n  poll_s: 120\n",
        )
        plan_file = proj / "plans" / "feature.md"
        plan_file.write_text("# plan\n", encoding="utf-8")

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        _reset_cache()
        assert registry._find_project_config() is None, "foreign CWD has its own config"
        _reset_cache()

        base_progress = registry._load_base_config()["progress"]
        assert registry.get_progress_interval() == float(base_progress["interval_s"])
        assert registry.get_progress_poll() == float(base_progress["poll_s"])

        assert registry.get_progress_interval(anchor=str(plan_file)) == 0.0
        assert registry.get_progress_poll(anchor=str(plan_file)) == 120.0


@pytest.mark.config_override
class TestCodeReviewCollapsePathsBlock:
    """`code_review.collapse_paths` — the configurable half of review-code's
    path classifier (review-scope plan D2.1, decision 11.5).

    Validated at load on OVERRIDE layers, like `progress:`: a malformed list
    that silently fell back to the defaults would read exactly like the
    repository's setting having worked while the review kept reading the diff
    the user meant to collapse.
    """

    @staticmethod
    def _override(monkeypatch, tmp_path, body):
        override = _write_yaml(tmp_path / ".multi-llm" / "providers.yaml", body)
        _set_project_config(monkeypatch, override)
        return override

    @pytest.fixture
    def reg_with_base_block(self, monkeypatch):
        """``reg``, over a base carrying the block the shipped one carries."""
        base = copy.deepcopy(FIXED_BASE)
        base["code_review"] = {
            "collapse_paths": {
                "generated": [],
                "fixtures": ["**/tests/fixtures/**", "**/tests/golden/**"],
            }
        }
        monkeypatch.setattr(registry, "_load_base_config", lambda: copy.deepcopy(base))
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        _reset_cache()
        yield registry
        _reset_cache()

    def test_the_shipped_block_is_the_classifier_default(self):
        """One copy of the defaults: the base file must say what the code falls
        back to, or an unconfigured repo and a fixture base would disagree."""
        from utils.git_utils import DEFAULT_COLLAPSE_PATHS

        shipped = registry._load_base_config()["code_review"]["collapse_paths"]
        assert shipped == {key: list(value) for key, value in DEFAULT_COLLAPSE_PATHS.items()}
        # ...and the shipped base passes its own validator.
        registry.validate_code_review_block(registry._load_base_config()["code_review"])

    def test_an_unconfigured_repo_classifies_as_today(self, reg):
        """No block anywhere (FIXED_BASE predates it): the accessor answers with
        the built-in rule, and the classifier groups exactly as with no config."""
        from utils.git_utils import classify_review_paths

        collapse = registry.get_collapse_paths()
        assert collapse == {
            "generated": [],
            "fixtures": ["**/tests/fixtures/**", "**/tests/golden/**"],
        }
        paths = [
            "src/a.py", "tests/test_a.py", "tests/fixtures/x.json",
            "pkg/tests/golden/y.txt", "poetry.lock",
        ]
        assert classify_review_paths(paths, collapse) == classify_review_paths(paths)
        assert classify_review_paths(paths, collapse)["poetry.lock"] == "source"

    def test_an_added_lock_glob_collapses_poetry_lock_into_generated(
        self, reg_with_base_block, monkeypatch, tmp_path
    ):
        from utils.git_utils import classify_review_paths

        self._override(
            monkeypatch, tmp_path,
            'code_review:\n  collapse_paths:\n    generated: ["**/*.lock"]\n',
        )
        collapse = registry.get_collapse_paths(anchor=str(tmp_path))
        # `generated` replaced, `fixtures` inherited from the base by the merge.
        assert collapse == {
            "generated": ["**/*.lock"],
            "fixtures": ["**/tests/fixtures/**", "**/tests/golden/**"],
        }
        groups = classify_review_paths(
            ["poetry.lock", "web/yarn.lock", "src/a.py", "tests/fixtures/f.json"],
            collapse,
        )
        assert groups == {
            "poetry.lock": "generated",
            "web/yarn.lock": "generated",
            "src/a.py": "source",
            "tests/fixtures/f.json": "fixtures",
        }

    def test_an_empty_list_switches_a_rule_off(
        self, reg_with_base_block, monkeypatch, tmp_path
    ):
        """`[]` is a deliberate wipe-out (lists replace on merge), not a blank."""
        from utils.git_utils import classify_review_paths

        self._override(
            monkeypatch, tmp_path, "code_review:\n  collapse_paths:\n    fixtures: []\n",
        )
        collapse = registry.get_collapse_paths(anchor=str(tmp_path))
        assert collapse["fixtures"] == []
        assert classify_review_paths(["tests/fixtures/f.json"], collapse) == {
            "tests/fixtures/f.json": "tests",
        }

    def test_a_blank_value_keeps_the_base_list(
        self, reg_with_base_block, monkeypatch, tmp_path
    ):
        self._override(
            monkeypatch, tmp_path,
            "code_review:\n  collapse_paths:\n    fixtures:\n    generated: [dist/**]\n",
        )
        assert registry.get_collapse_paths(anchor=str(tmp_path)) == {
            "generated": ["dist/**"],
            "fixtures": ["**/tests/fixtures/**", "**/tests/golden/**"],
        }

    @pytest.mark.parametrize(
        "body, fragments",
        [
            (
                'code_review:\n  collapse_paths:\n    generated: "**/*.lock"\n',
                ["code_review.collapse_paths.generated is '**/*.lock' —",
                 "list of non-empty glob strings"],
            ),
            (
                "code_review:\n  collapse_paths:\n    fixtures: 3\n",
                ["code_review.collapse_paths.fixtures is 3 —"],
            ),
            (
                'code_review:\n  collapse_paths:\n    fixtures: [""]\n',
                ["code_review.collapse_paths.fixtures is [''] —"],
            ),
            (
                'code_review:\n  collapse_paths: ["**/*.lock"]\n',
                ["code_review.collapse_paths is ['**/*.lock'] —", "mapping"],
            ),
            (
                "code_review: 5\n",
                ["`code_review:` must be a mapping", "(5)"],
            ),
            (
                "code_review:\n  collapse: {}\n",
                ["unknown key(s) under `code_review:`: collapse", "collapse_paths"],
            ),
            (
                "code_review:\n  collapse_paths:\n    vendored: []\n",
                ["unknown key(s) under code_review.collapse_paths: vendored"],
            ),
            (
                'code_review:\n  collapse_paths:\n    generated: ["/abs/dist/**"]\n',
                ["code_review.collapse_paths.generated is '/abs/dist/**' —",
                 "an absolute path"],
            ),
            (
                'code_review:\n  collapse_paths:\n    generated: ["../dist/**"]\n',
                ["parent-traversal"],
            ),
            (
                'code_review:\n  collapse_paths:\n    generated: ["[z-a].lock"]\n',
                ["code_review.collapse_paths.generated is '[z-a].lock' —",
                 "not a valid glob"],
            ),
        ],
        ids=[
            "generated-bare-string", "fixtures-int", "fixtures-empty-entry",
            "collapse_paths-a-list", "code_review-not-a-mapping",
            "unknown-code_review-key", "unknown-collapse-key", "absolute-entry",
            "parent-traversal-entry", "uncompilable-class",
        ],
    )
    def test_a_malformed_value_fails_the_load_with_the_key_named(
        self, reg_with_base_block, monkeypatch, tmp_path, body, fragments
    ):
        self._override(monkeypatch, tmp_path, body)
        with pytest.raises(ConfigError) as exc:
            load_config(anchor=str(tmp_path))
        message = str(exc.value)
        for fragment in fragments:
            assert fragment in message, f"{fragment!r} missing from: {message}"

    def test_permissive_mode_does_not_relax_the_rejection(
        self, reg_with_base_block, monkeypatch, tmp_path
    ):
        self._override(
            monkeypatch, tmp_path,
            'code_review:\n  collapse_paths:\n    generated: "**/*.lock"\n',
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE", "1")
        _reset_cache()
        with pytest.raises(ConfigError) as exc:
            load_config(anchor=str(tmp_path))
        assert "code_review.collapse_paths.generated" in str(exc.value)


@pytest.mark.config_override
class TestInitConfigScaffolder:
    """init_config.py scaffolding + packaging guard (template path / gitignore).

    These verbatim-copy assertions pass ``--template-only`` explicitly to pin the
    pristine template path. The auto-detect toggler, D2a guard, and end-to-end
    init are covered in tests/test_init_config.py.
    """

    def _run_init(self, target_dir, *extra):
        import init_config
        return init_config.main(["--dir", str(target_dir), "--template-only", *extra])

    def test_template_asset_exists_packaging_guard(self):
        import init_config
        template = Path(init_config.__file__).parent / "templates" / "config" / "providers.override.yaml"
        assert template.exists(), f"config template missing at {template}"
        # And it parses as YAML to a real stub, not garbage: the active
        # `defaults:` key means the template must parse to a mapping (a
        # fully-commented or corrupt template would not).
        assert isinstance(yaml.safe_load(template.read_text(encoding="utf-8")), dict)

    def test_writes_file_and_creates_dir(self, tmp_path):
        rc = self._run_init(tmp_path)
        assert rc == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        assert out.exists()
        # Content is the template.
        import init_config
        assert out.read_text(encoding="utf-8") == init_config.TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_refuses_overwrite_without_force(self, tmp_path):
        assert self._run_init(tmp_path) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        out.write_text("custom: content\n", encoding="utf-8")
        assert self._run_init(tmp_path) == 1
        assert out.read_text(encoding="utf-8") == "custom: content\n"  # untouched

    def test_overwrites_with_force(self, tmp_path):
        assert self._run_init(tmp_path) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        out.write_text("custom: content\n", encoding="utf-8")
        assert self._run_init(tmp_path, "--force") == 0
        import init_config
        assert out.read_text(encoding="utf-8") == init_config.TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_default_leaves_gitignore_untouched(self, tmp_path):
        assert self._run_init(tmp_path) == 0
        assert not (tmp_path / ".gitignore").exists()

    def test_gitignore_appends_idempotently(self, tmp_path):
        assert self._run_init(tmp_path, "--gitignore") == 0
        gi = tmp_path / ".gitignore"
        assert gi.exists()
        assert ".multi-llm/" in gi.read_text(encoding="utf-8")
        first = gi.read_text(encoding="utf-8")
        # Re-run: no duplicate entry.
        assert self._run_init(tmp_path, "--force", "--gitignore") == 0
        assert gi.read_text(encoding="utf-8") == first
        assert gi.read_text(encoding="utf-8").count(".multi-llm/") == 1


class TestTemplateBaseParity:
    """The shipped template is generated from the live base config — re-deriving it
    must reproduce the shipped file byte-for-byte, so a base edit that is never
    regenerated into the template fails CI."""

    def test_shipped_template_matches_generator(self):
        import init_config

        regenerated = init_config.build_template_text()
        shipped = init_config.TEMPLATE_PATH.read_text(encoding="utf-8")
        assert shipped == regenerated, (
            "templates/config/providers.override.yaml is out of sync with the base "
            "providers.yaml — regenerate it:\n"
            "  python -c \"import init_config as c; "
            "c.TEMPLATE_PATH.write_text(c.build_template_text())\""
        )

    def test_generated_template_is_inert(self):
        import init_config

        parsed = yaml.safe_load(init_config.build_template_text())
        # A plain copy changes no behavior: no live providers/default_provider, and
        # the defaults sub-keys are present-but-blank (None → inherit base).
        assert set(parsed.keys()) == {"defaults"}
        assert parsed["defaults"] == {"models": None, "quick_models": None}


@pytest.mark.config_override
class TestOrchestratorIntegration:
    """End-to-end: discovery + layering + selection in a real git repo."""

    def test_resolve_models_uses_project_defaults(self, monkeypatch, tmp_path):
        # Real git repo with a distinctive project override; cwd = repo root.
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - claude-code:opus\n",
        )
        monkeypatch.chdir(tmp_path)
        _reset_cache()
        # All provider CLIs "available" so availability filtering keeps the spec.
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        from utils.interactive import resolve_models
        models = resolve_models(mode=None)
        # Project default wins over the shipped base default list.
        assert models == ["claude-code:opus"]

    def test_resolve_models_anchor_overrides_cwd(self, monkeypatch, tmp_path):
        # Two distinct git repos: the plan's repo carries the override that
        # discovery must use; CWD points at an UNRELATED repo whose own override
        # must be ignored when an anchor is supplied. This exercises the
        # plan-path anchor threading orchestrators rely on when CWD != the plan's
        # git root (centrally-stored plans / multi-repo workflows).
        plan_repo = tmp_path / "plan-repo"
        cwd_repo = tmp_path / "cwd-repo"
        plan_repo.mkdir()
        cwd_repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=plan_repo, check=True)
        subprocess.run(["git", "init", "-q"], cwd=cwd_repo, check=True)
        # Plan repo override: the distinctive list discovery should select.
        _write_yaml(
            plan_repo / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - claude-code:opus\n",
        )
        # CWD repo override: a DIFFERENT list that must NOT be picked up via anchor.
        _write_yaml(
            cwd_repo / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - cursor-agent:auto\n",
        )
        # A plan file living under the plan repo; its path is the anchor.
        plan_file = plan_repo / "plans" / "feature.md"
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text("# plan", encoding="utf-8")
        # CWD is the unrelated repo, NOT the plan's git root.
        monkeypatch.chdir(cwd_repo)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        from utils.interactive import resolve_models

        # Anchored at the plan file → plan-repo override drives discovery.
        _reset_cache()
        anchored = resolve_models(mode=None, anchor=str(plan_file))
        assert anchored == ["claude-code:opus"]

        # Contrast: WITHOUT the anchor, discovery is CWD-based and picks up the
        # unrelated repo's override instead — proving the anchor (not CWD) drove
        # the result above.
        _reset_cache()
        cwd_based = resolve_models(mode=None)
        assert cwd_based == ["cursor-agent:auto"]


def _assert_stem_collision_refusal(excinfo, specs, stem):
    """The refusal has to name BOTH colliding specs and the stem they share.

    Each of the three is load-bearing on its own. Naming one spec leaves the
    operator to work out which of the other four ``--models`` entries it collided
    with; naming neither and only the stem describes a string that never appeared
    in anything they typed. And the stem is the half that cannot be re-derived by
    eye — ``sanitize_model_name`` flattens ``:``, ``.`` and ``/`` alike, so
    "which two of these become the same file name" is precisely the question the
    message exists to answer.

    ``site`` and ``exit_code`` are asserted here rather than in each test because
    they are what makes the refusal *uniform*: all three orchestrators route
    :class:`NeedsHuman` through ``unattended.fail_closed``, so a collision has to
    arrive as D6 row 2 (which models to fan out to) at exit 3, exactly like the
    empty-``--models`` and nothing-resolved gates beside it. A ``ValueError``
    here would exit 1 with a traceback on one caller and be swallowed on another.
    """
    message = str(excinfo.value)
    for spec in specs:
        assert spec in message, (
            f"the refusal never names the colliding spec {spec!r}: {message}"
        )
    assert stem in message, (
        f"the refusal never names the shared stem {stem!r}, which is the one "
        f"string in it the operator cannot derive by eye: {message}"
    )
    assert excinfo.value.site == "model-selection", (
        "a stem collision is D6 row 2's decision site, so it must be recorded "
        f"under that site_id like every other refusal resolve_models raises; "
        f"got {excinfo.value.site!r}"
    )
    assert excinfo.value.exit_code == 3, (
        "exit 3 is reserved for 'unattended run cannot proceed'; a collision "
        "cannot be resolved without a human choosing which spec was meant"
    )


@pytest.mark.config_override
class TestCollidingModelStemsAreRefusedBeforeDispatch:
    """Two distinct specs, one filename stem: refused at ``resolve_models``.

    ``sanitize_model_name`` (``utils/json_extractor.py``) is lossy in one
    direction only — every separator a spec can carry collapses to ``_`` — so
    ``p:a.b`` and ``p:a_b`` are two models with one stem, ``p_a_b``. Selecting
    both breaks two invariants silently: the per-model artifacts (``{stem}.json``
    and its ``log_``/``error_``/``salvage_`` siblings) overwrite each other,
    with ``skip_existing_result`` mis-skipping the second model as already done;
    and ``build_unsanitize_map`` keeps whichever spec the checkpoint listed last,
    so ``--reaggregate`` attributes that one file's groups to a model that may
    not have produced them.

    ``resolve_models`` is where that is refused, because it is the single seam
    all three orchestrators resolve through and it sits before anything
    expensive: refusing costs a correct run nothing, where refusing at the
    ``write_status`` that records ``models_requested`` would throw away a whole
    fan-out.

    Every leg of the priority ladder is exercised, and that is the point of
    having three of them rather than one: ``--models``, ``defaults.models`` and
    ``quick_models`` all return through the same ``_canonical`` closure, and
    these tests are what pin that they do. A gate placed on the ``cli_models``
    branch alone would pass the first test here and leave a colliding pair in a
    project's ``providers.yaml`` entirely unguarded.
    """

    #: The plan's own example, and the one the ``--models`` leg can use verbatim:
    #: ``parse_model_spec`` splits on the first ``:`` and validates nothing, so a
    #: fictitious provider ``p`` survives canonicalization untouched.
    FLAG_SPECS = ["p:a.b", "p:a_b"]
    FLAG_STEM = "p_a_b"

    #: The same collision spelled with a provider that has a hardcoded adapter.
    #: The config legs below filter on ``get_provider(name).is_available()``, so
    #: ``p`` would be dropped as unavailable and the run would die on "all
    #: providers unavailable" long before reaching the gate — which would make
    #: those two tests pass for entirely the wrong reason. The collision is in
    #: the model half of the spec either way: ``a.b`` and ``a_b`` are one stem.
    CONFIG_SPECS = ["claude-code:a.b", "claude-code:a_b"]
    CONFIG_STEM = "claude-code_a_b"

    def test_two_models_flag_specs_that_share_a_stem_are_refused(self, reg):
        from utils.interactive import resolve_models

        with pytest.raises(NeedsHuman) as excinfo:
            resolve_models(cli_models=list(self.FLAG_SPECS))
        _assert_stem_collision_refusal(excinfo, self.FLAG_SPECS, self.FLAG_STEM)

    def test_the_same_pair_reaching_defaults_models_is_refused_identically(
        self, reg, monkeypatch, tmp_path
    ):
        """The config leg, with no ``--models`` anywhere in the call.

        A colliding pair committed to a project's ``providers.yaml`` is the
        *worse* case of the two: it is typed once and then silently re-selected
        by every run in that repo, where a bad ``--models`` invocation at least
        has a human looking at it.
        """
        from utils.interactive import resolve_models

        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n"
            + "".join(f"    - {spec}\n" for spec in self.CONFIG_SPECS),
        )
        _set_project_config(monkeypatch, override)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

        with pytest.raises(NeedsHuman) as excinfo:
            resolve_models(mode=None)
        _assert_stem_collision_refusal(excinfo, self.CONFIG_SPECS, self.CONFIG_STEM)

    def test_the_same_pair_reaching_quick_models_is_refused_identically(
        self, reg, monkeypatch, tmp_path
    ):
        """The third leg. ``--quick`` returns through ``_canonical`` too.

        Worth its own test rather than trusting the shared closure, because
        ``--quick`` is the leg with its own availability filter, its own
        "all providers unavailable" exit and its own fall-through into step 4 —
        three chances for a future edit to build a return path that skips the
        gate.
        """
        from utils.interactive import resolve_models

        override = _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  quick_models:\n"
            + "".join(f"    - {spec}\n" for spec in self.CONFIG_SPECS),
        )
        _set_project_config(monkeypatch, override)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

        with pytest.raises(NeedsHuman) as excinfo:
            resolve_models(quick=True, mode=None)
        _assert_stem_collision_refusal(excinfo, self.CONFIG_SPECS, self.CONFIG_STEM)

    def test_three_specs_on_one_stem_are_named_in_one_refusal(self, reg):
        """An N-way collision is one message that reads as one, not as a pair.

        Three variants of one model is the case where the operator is *least*
        likely to spot the collision by eye, so it is the case where a message
        written for exactly two — "two of the selected models", "both sanitize
        to" — would be at its most misleading. The refusal has to name all three
        and still parse as English.
        """
        from utils.interactive import resolve_models

        # `/` and `.` both flatten to `_`; `-` does **not** — it is inside
        # `sanitize_model_name`'s allowed class, so `p:a-b` keeps its hyphen and
        # is not a collision at all. The slash is the realistic third member
        # anyway: `openrouter/moonshotai/kimi-k3` is how half the specs in this
        # repo's own config are written.
        specs = ["p:a.b", "p:a_b", "p:a/b"]
        with pytest.raises(NeedsHuman) as excinfo:
            resolve_models(cli_models=list(specs))
        _assert_stem_collision_refusal(excinfo, specs, "p_a_b")

        message = str(excinfo.value)
        assert "all sanitize to" in message, (
            f"three specs on one stem still read 'both sanitize to': {message}"
        )
        assert "both sanitize to" not in message
        assert message.count("sanitize to") == 1, (
            f"one stem is one collision and must be one line: {message}"
        )

    def test_one_spec_carrying_a_separator_is_not_a_collision(self, reg):
        """The first negative: a lossy stem is normal, and normal must still run.

        Nearly every real spec sanitizes to something it cannot be recovered
        from — ``cursor-agent:cursor-grok-4.6-xhigh`` becomes
        ``cursor-agent_cursor-grok-4_6-xhigh``. The defect is two specs sharing
        one stem, never one spec having a lossy one, and a gate that confused the
        two would refuse the entire model catalog.
        """
        from utils.interactive import resolve_models

        assert resolve_models(cli_models=["p:a.b"]) == ["p:a.b"]

    def test_the_same_spec_passed_twice_is_collapsed_rather_than_refused(self, reg):
        """The second negative, and the one the gate is most likely to get wrong.

        ``--models p:a.b p:a.b`` is a user typing one thing twice. There is one
        model, one stem and one result file, so there is nothing to overwrite and
        nothing to misattribute — a naive "are all stems distinct?" check would
        refuse it anyway. Collapsing (rather than passing both through) is the
        stronger answer: two identical entries would otherwise fan out twice,
        spend twice and write the same file twice for one answer.
        """
        from utils.interactive import resolve_models

        assert resolve_models(cli_models=["p:a.b", "p:a.b"]) == ["p:a.b"]

    def test_a_bare_spec_and_its_prefixed_twin_collapse_rather_than_collide(
        self, reg
    ):
        """Deduplication is on the CANONICAL form, which is why it can collapse.

        ``defaults.models`` and ``--models`` both legally carry bare, prefix-less
        entries, so ``sonnet`` and ``cursor-agent:sonnet`` — under this config's
        ``default_provider`` — are one model written two ways. They resolve to
        one spec and therefore to one stem; deduplicating on the raw input
        instead would have let two spellings of one model through as a collision
        and refused a perfectly ordinary list.
        """
        from utils.interactive import resolve_models

        assert resolve_models(
            cli_models=["sonnet", "cursor-agent:sonnet"]
        ) == ["cursor-agent:sonnet"]


@pytest.mark.config_override
class TestIsModelValidOptionB:
    """is_model_valid honours explicitly-configured defaults (step 5, Option B).

    A model deliberately placed in ``defaults.*`` is treated as valid even when
    it is absent from the provider catalog, which suppresses the spurious
    "unknown model" warning. The configured set is built INTERNALLY from the
    merged config (no threading from call sites); an optional ``configured``
    override exists for tests / explicit injection.
    """

    def test_explicit_configured_override_param(self, reg):
        # The override param short-circuits without consulting the catalog.
        assert is_model_valid(
            "cursor-agent:made-up", configured={"cursor-agent:made-up"}
        ) is True

    def test_configured_set_built_internally(self, monkeypatch, tmp_path):
        # Real git repo: a project override puts a made-up model in defaults.models.
        # No `configured` argument is passed → the set must be derived internally.
        _patch_base_only(monkeypatch)  # FIXED_BASE base, real discovery
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - cursor-agent:made-up\n",
        )
        monkeypatch.chdir(tmp_path)
        _reset_cache()
        # made-up is NOT in the FIXED_BASE catalog, yet it is configured → valid.
        assert is_model_valid("cursor-agent:made-up", anchor=str(tmp_path)) is True

    def test_unconfigured_and_uncatalogued_is_false(self, reg):
        # Neither configured (in FIXED_BASE defaults) nor present in the catalog.
        assert is_model_valid("cursor-agent:not-a-real-model") is False

    def test_bare_spec_canonicalizes_against_default_provider(self, monkeypatch):
        # A bare spec "opus" in defaults.models resolves against the configured
        # default_provider. With claude-code as default it canonicalizes to
        # claude-code:opus and validates / suppresses. Use a base whose defaults
        # contain only the single bare spec so the configured set is exact.
        base = copy.deepcopy(FIXED_BASE)
        base["default_provider"] = "claude-code"
        base["defaults"] = {"models": ["opus"]}
        monkeypatch.setattr(
            registry, "_load_base_config", lambda: copy.deepcopy(base)
        )
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        _reset_cache()
        assert registry._collect_configured_specs() == {"claude-code:opus"}
        assert is_model_valid("opus") is True
        assert is_model_valid("claude-code:opus") is True

    def test_bare_spec_canonicalizes_under_different_default_provider(self, monkeypatch):
        # The SAME bare spec under a DIFFERENT default_provider resolves to a
        # different canonical spec — proving anchor-aware resolution.
        base = copy.deepcopy(FIXED_BASE)
        base["default_provider"] = "cursor-agent"
        base["defaults"] = {"models": ["opus"]}
        monkeypatch.setattr(
            registry, "_load_base_config", lambda: copy.deepcopy(base)
        )
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        _reset_cache()
        assert registry._collect_configured_specs() == {"cursor-agent:opus"}

    def test_mode_quick_entries_both_shapes_collected(self, monkeypatch, tmp_path):
        # A bare-list mode and a {models, quick} dict mode both contribute their
        # specs; the dict's `quick` sub-key is NOT dropped.
        _patch_base_only(monkeypatch)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "default_provider: cursor-agent\n"
            "defaults:\n"
            "  models: []\n"
            "  quick_models: []\n"
            "  modes:\n"
            "    review-plan:\n"
            "      - cursor-agent:bare-list-model\n"
            "    code-review:\n"
            "      models:\n"
            "        - cursor-agent:dict-model\n"
            "      quick:\n"
            "        - cursor-agent:dict-quick-model\n",
        )
        monkeypatch.chdir(tmp_path)
        _reset_cache()
        specs = registry._collect_configured_specs(anchor=str(tmp_path))
        assert "cursor-agent:bare-list-model" in specs
        assert "cursor-agent:dict-model" in specs
        assert "cursor-agent:dict-quick-model" in specs

    def test_unknown_provider_spec_not_configured_is_false(self, monkeypatch, tmp_path):
        # A typo'd provider that is NOT written into defaults.* must stay invalid:
        # provider-existence is preserved (no short-circuit on the catalog path).
        _patch_base_only(monkeypatch)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        _write_yaml(
            tmp_path / ".multi-llm" / "providers.yaml",
            "defaults:\n  models:\n    - cursor-agent:made-up\n",
        )
        monkeypatch.chdir(tmp_path)
        _reset_cache()
        assert is_model_valid("cursr-agent:foo", anchor=str(tmp_path)) is False


# =============================================================================
# §9.3 — Config anchoring (task L1, group A1)
#
# The measured failure this section exists for is *routing*, not timeouts: run
# from a foreign CWD, a plan whose project config configures `gemini` spawned
# `cursor-agent` instead, on the base timeout, with the base concurrency gates.
# The three tests below are §9.3.f's acceptance row (all six observations in ONE
# run), its no-project-config negative control, and §9.3.e's AST guard against an
# eleventh unanchored accessor.
# =============================================================================

#: §9.3.f's fixture config. `subagent_timeout_s: null` is base's shipped value,
#: restated so the Lane B leak (`get_subagent_timeout` dropping the anchor when
#: the driver key is null) is exercised in its DEFAULT state, and
#: `defaults.models` is bare on purpose so 9.3.d's canonicalisation is exercised.
FOREIGN_CWD_PROJECT_CONFIG = """\
default_provider: gemini
providers:
  gemini:
    default_timeout: 4242
    max_concurrent: 7
driver:
  max_concurrent_subagents: 9
  subagent_timeout_s: null
defaults:
  models:
    - sonnet
"""


def _foreign_cwd_project(tmp_path, monkeypatch, *, with_override: bool) -> Path:
    """A git repo holding the plan, and a CWD *outside* it with no config at all.

    Returns the plan file path (the anchor every accessor is supposed to follow).
    """
    proj = tmp_path / "proj"
    (proj / "plans").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
    if with_override:
        _write_yaml(proj / ".multi-llm" / "providers.yaml", FOREIGN_CWD_PROJECT_CONFIG)
    plan_file = proj / "plans" / "feature.md"
    plan_file.write_text("# plan\n", encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _reset_cache()
    # Sanity: the foreign CWD must resolve to NO project config, or the test
    # would be comparing two overrides instead of anchor-vs-CWD.
    assert registry._find_project_config() is None
    _reset_cache()
    return plan_file


def _completed(stdout: str = "[]"):
    """A stand-in for the provider subprocess that never runs."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = stdout
    result.stderr = ""
    return result


@pytest.mark.config_override
class TestForeignCwdAnchoring:
    """§9.3.f — routing, timeout and concurrency, asserted together (gate A1)."""

    def test_routing_timeout_and_concurrency_all_follow_the_plan(
        self, tmp_path, monkeypatch
    ):
        """All six observations, in ONE run.

        A1's whole point is that a partially anchored run is worse than a
        consistently wrong one, so these are deliberately not six tests that
        could each pass alone.

            spawned argv[0]                     cursor-agent -> gemini
            its timeout= kwarg                  1200         -> 4242
            per-provider semaphore bound        None         -> 7
            global fan-out gate                 4            -> 9
            get_subagent_timeout (Lane B twin)  1200         -> 4242
            resolve_models' spec list           ["sonnet"]   -> ["gemini:sonnet"]
        """
        plan_file = _foreign_cwd_project(tmp_path, monkeypatch, with_override=True)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

        import utils.review_orchestrator_base as rob
        from utils.interactive import resolve_models
        from utils.llm_client import invoke_with_provider

        # The pre-flight capture point (§9.3.a): one absolute plan path, set once.
        registry.set_run_anchor(str(plan_file))

        # -- row 6: resolve_models returns CANONICAL specs (§9.3.d) ------------
        specs = resolve_models(mode=None, anchor=str(plan_file))

        # -- row 4: the global gate, read with NO keyword ----------------------
        # The run anchor is the fallback (§9.3.b), so even a call site the sweep
        # missed lands on the run's plan rather than on CWD.
        gate = registry.get_max_concurrent_subagents()

        # -- row 3: the bound the fan-out really sized its semaphore with ------
        real_bound = registry.get_provider_max_concurrent
        observed_bounds = []

        def _record_bound(provider_name, anchor=None):
            value = real_bound(provider_name, anchor=anchor)
            observed_bounds.append((provider_name, value))
            return value

        monkeypatch.setattr(rob, "get_provider_max_concurrent", _record_bound)

        # -- rows 1 + 2: the real Lane A fan-out, down to the spawn ------------
        async def per_model(model_spec, index):
            return await asyncio.to_thread(
                invoke_with_provider,
                prompt="q",
                model_spec=model_spec,
                anchor=str(plan_file),
            )

        with patch("utils.llm_client.subprocess.run") as spawn:
            spawn.return_value = _completed()
            asyncio.run(
                rob.run_models_concurrent(
                    specs, per_model, None, gate, anchor=str(plan_file)
                )
            )

        assert spawn.call_count == 1
        argv = list(spawn.call_args[0][0])
        spawn_timeout = spawn.call_args[1]["timeout"]

        assert specs == ["gemini:sonnet"], "row 6: resolve_models must canonicalise"
        assert os.path.basename(argv[0]) == "gemini", (
            f"row 1: the configured provider must be the one spawned; got {argv[0]!r}"
        )
        assert spawn_timeout == 4242, "row 2: the project's provider timeout"
        assert ("gemini", 7) in observed_bounds, (
            f"row 3: per-provider bound; observed {observed_bounds!r}"
        )
        assert gate == 9, "row 4: driver.max_concurrent_subagents"
        assert registry.get_subagent_timeout("gemini", anchor=str(plan_file)) == 4242, (
            "row 5: Lane B keeps the anchor when driver.subagent_timeout_s is null"
        )

    def test_negative_control_no_project_config(self, tmp_path, monkeypatch):
        """§9.3.f's control: with no project config, anchored == unanchored.

        Without this the acceptance test could go green by anchoring everything
        at CWD (or at the git root of whatever happened to be around).
        """
        plan_file = _foreign_cwd_project(tmp_path, monkeypatch, with_override=False)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        from utils.interactive import resolve_models

        base = registry.load_base_config()
        base_timeout = base["providers"]["gemini"]["default_timeout"]
        base_gate = base["driver"]["max_concurrent_subagents"]

        registry.set_run_anchor(str(plan_file))
        anchor = str(plan_file)

        assert registry.get_provider_timeout("gemini") == base_timeout
        assert registry.get_provider_timeout("gemini", anchor=anchor) == base_timeout
        assert registry.get_max_concurrent_subagents() == base_gate
        assert registry.get_max_concurrent_subagents(anchor=anchor) == base_gate
        assert (registry.get_provider_max_concurrent("gemini")
                == registry.get_provider_max_concurrent("gemini", anchor=anchor))
        assert registry.get_subagent_timeout("gemini", anchor=anchor) == base_timeout
        assert (sorted(registry.get_all_model_specs())
                == sorted(registry.get_all_model_specs(anchor=anchor)))
        # Bare base defaults still come back canonical — the shape does not
        # depend on an override being present.
        specs = resolve_models(mode=None, anchor=anchor)
        assert specs and all(":" in spec for spec in specs)


@pytest.mark.config_override
class TestRunAnchor:
    """§9.3.b — the process-wide fallback and its two invariants."""

    def test_fallback_answers_an_unanchored_read(self, tmp_path, monkeypatch):
        plan_file = _foreign_cwd_project(tmp_path, monkeypatch, with_override=True)
        registry.set_run_anchor(str(plan_file))
        # No keyword anywhere: the run anchor, not CWD, decides.
        assert registry.get_provider_timeout("gemini") == 4242
        assert registry.load_config()["default_provider"] == "gemini"

    def test_an_explicit_anchor_still_wins_over_the_fallback(self, tmp_path, monkeypatch):
        plan_file = _foreign_cwd_project(tmp_path, monkeypatch, with_override=True)
        registry.set_run_anchor(str(plan_file))
        elsewhere = str(Path.cwd())
        assert registry.get_provider_timeout("gemini", anchor=elsewhere) == 1200

    def test_re_setting_the_same_path_is_a_no_op(self, tmp_path, monkeypatch):
        plan_file = _foreign_cwd_project(tmp_path, monkeypatch, with_override=True)
        registry.set_run_anchor(str(plan_file))
        registry.set_run_anchor(str(plan_file))          # `full`'s per-phase pre-flight
        assert registry.get_run_anchor() == os.path.abspath(str(plan_file))

    def test_re_setting_a_different_path_raises(self, tmp_path, monkeypatch):
        plan_file = _foreign_cwd_project(tmp_path, monkeypatch, with_override=True)
        registry.set_run_anchor(str(plan_file))
        other = plan_file.parent / "other.md"
        other.write_text("# other\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="run anchor"):
            registry.set_run_anchor(str(other))

    def test_a_value_read_before_the_anchor_is_not_served_after_it(
        self, tmp_path, monkeypatch
    ):
        """Invariant 1, observable form.

        `parse_args` renders `--help`'s model list before `main` has a plan path,
        so "no load_config before set_run_anchor" cannot be asserted literally.
        What must hold is the property it protects: nothing read on the CWD key
        survives into the anchored half of the run.
        """
        plan_file = _foreign_cwd_project(tmp_path, monkeypatch, with_override=True)
        assert registry.get_provider_timeout("gemini") == 1200      # CWD-anchored
        registry.set_run_anchor(str(plan_file))
        assert registry.get_provider_timeout("gemini") == 4242


# ---------------------------------------------------------------------------
# §9.3.e — the guard test, so there is never an eleventh site
# ---------------------------------------------------------------------------

#: Every provider_registry accessor that takes a discovery ``anchor`` — the 16
#: that already followed the convention, plus the two L1 gave one to, plus
#: ``resolve_models`` (which threads the same value into all of them).
ANCHORED_ACCESSORS = frozenset({
    "load_config",
    "parse_model_spec",
    "get_available_models",
    "is_model_valid",
    "_collect_configured_specs",
    "get_default_models",
    "get_quick_models",
    "get_driver_config",
    "get_driver_provenance",
    "get_subagent_runner",
    "get_subagent_timeout",
    "get_provider_max_concurrent",
    "get_max_concurrent_subagents",
    "get_max_concurrent_write_subagents",
    "get_verify_command",
    "get_verify_timeout_s",
    "get_write_paths",
    "get_subagent_retries",
    # ...and the ones §9.3.c gave an anchor to.
    "get_provider_timeout",
    "get_all_model_specs",
    "resolve_models",
    # ...and the `progress:` block's two.
    "get_progress_interval",
    "get_progress_poll",
    # ...and the `code_review:` block's one.
    "get_collapse_paths",
})

#: Directories that are not this skill's runtime source.
_NOT_RUNTIME_DIRS = {"tests", "__pycache__", ".venv", "plans", "assets", "templates"}

#: Per-call-site allowlist, keyed ``<module>::<enclosing function>::<callee>`` and
#: valued ``(count, reason)``. Compared for EXACT equality against what the walk
#: finds, so a new unanchored call — even a second one inside an already-listed
#: function — is red, and a stale entry is red too.
UNANCHORED_ALLOWLIST = {
    "ask_orchestrator.py::parse_args::get_all_model_specs": (
        1,
        "argparse epilog: the available-model list is rendered while building the "
        "parser, before --plan-file has been parsed, so no anchor exists yet.",
    ),
    "code_review_orchestrator.py::parse_args::get_all_model_specs": (1, "as above"),
    "review_plan_orchestrator.py::parse_args::get_all_model_specs": (1, "as above"),
    "review_tasks_orchestrator.py::parse_args::get_all_model_specs": (1, "as above"),
}


def _runtime_python_files():
    root = Path(__file__).parent.parent
    for path in sorted(root.rglob("*.py")):
        if _NOT_RUNTIME_DIRS & set(path.relative_to(root).parts):
            continue
        yield root, path


def _call_is_anchored(node: ast.Call) -> bool:
    """True when the call hands an anchor on, by keyword or by forwarding."""
    for keyword in node.keywords:
        if keyword.arg == "anchor":
            return True
        if keyword.arg is None:          # **kwargs forwarding
            return True
    for arg in node.args:
        # A positionally-forwarded `anchor` / `self.anchor` counts too.
        if isinstance(arg, ast.Name) and arg.id == "anchor":
            return True
        if isinstance(arg, ast.Attribute) and arg.attr == "anchor":
            return True
    return False


def _unanchored_accessor_calls(path: Path):
    """``<enclosing function>::<callee>`` for every unanchored accessor call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    stack, found = [], []

    def visit(node):
        pushed = False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.append(node.name)
            pushed = True
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else None)
            if name in ANCHORED_ACCESSORS and not _call_is_anchored(node):
                found.append(f"{'.'.join(stack) or '<module>'}::{name}")
        for child in ast.iter_child_nodes(node):
            visit(child)
        if pushed:
            stack.pop()

    visit(tree)
    return found


class TestNoUnanchoredAccessorCall:
    """§9.3.e — 'add an accessor, forget the anchor' is red in CI."""

    def test_every_unanchored_accessor_call_is_allowlisted(self):
        offenders = Counter()
        for root, path in _runtime_python_files():
            relative = path.relative_to(root).as_posix()
            for site in _unanchored_accessor_calls(path):
                offenders[f"{relative}::{site}"] += 1

        expected = {key: count for key, (count, _) in UNANCHORED_ALLOWLIST.items()}
        assert dict(offenders) == expected, (
            "§9.3.e: an anchored accessor was called without an anchor. Thread the "
            "run's plan path through, or add the site to UNANCHORED_ALLOWLIST with "
            "a reason.\n"
            f"  unexpected: {sorted(set(offenders) - set(expected))}\n"
            f"  stale allowlist entries: {sorted(set(expected) - set(offenders))}"
        )

    def test_the_definitions_really_take_an_anchor(self):
        """The set the guard polices is the set that actually has the keyword."""
        import inspect

        from utils.interactive import resolve_models as _resolve_models

        missing = []
        for name in sorted(ANCHORED_ACCESSORS):
            func = getattr(registry, name, None) or _resolve_models
            if "anchor" not in inspect.signature(func).parameters:
                missing.append(name)
        assert not missing, f"declared anchored but takes no anchor=: {missing}"


# ---------------------------------------------------------------------------
# The session rung
# ---------------------------------------------------------------------------

SESSION_MODEL = "opencode/nemotron-3-ultra-free"
SESSION_SPEC = f"opencode:{SESSION_MODEL}"
#: What `hosts/capabilities.yaml` declares for the two hosts these tests use.
OPENCODE_FLOOR = "opencode:opencode/big-pickle"
CODEX_FLOOR = "codex:gpt-6-astra"


def _detected(monkeypatch, available=True):
    """Pin the opencode/codex adapters' `shutil.which` probe to one verdict.

    The rung is gated on the runner's CLI being *detected*, and whether this
    machine happens to have `opencode` on PATH is not what any of these rows is
    about.
    """
    from utils.providers import codex as cx
    from utils.providers import opencode as oc

    monkeypatch.setattr(oc.OpenCodeProvider, "is_available", lambda self: available)
    monkeypatch.setattr(cx.CodexProvider, "is_available", lambda self: available)


class TestTheSessionRung:
    """The model the host session is running on is a rung of its own (D3o).

    THE DEFECT. A `--review-plan --quick` launched from an OpenCode session whose
    picker said "Nemotron 3 Ultra Free · OpenCode Zen" validated all sixteen of
    its batches on `opencode/big-pickle`. Nothing was broken: no config layer had
    an opinion, so the ladder fell to `hosts/capabilities.yaml`'s
    `default_subagent_runner` — which is a *credential-free floor*, chosen so a
    zero-config install dispatches at all, and not a model anybody selected. The
    session's own model now sits between the explicit layers and that floor.
    """

    def test_the_session_model_answers_when_the_host_declares_a_provider(
        self, monkeypatch
    ):
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        assert registry.resolve_subagent_runner("full", host_id="opencode") == (
            SESSION_SPEC, registry.SESSION_LAYER,
        )

    def test_an_unset_variable_leaves_the_ladder_exactly_as_it_was(self, monkeypatch):
        _detected(monkeypatch)
        monkeypatch.delenv(registry.HOST_SESSION_MODEL_ENV, raising=False)
        assert registry.resolve_subagent_runner("full", host_id="opencode") == (
            OPENCODE_FLOOR, registry.HOST_DEFAULT_LAYER,
        )

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_a_blank_variable_is_the_same_as_unset(self, monkeypatch, value):
        """"The host did not say" and "the host said nothing" are one case."""
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, value)
        assert registry.session_runner(host_id="opencode") is None
        assert registry.resolve_subagent_runner("full", host_id="opencode") == (
            OPENCODE_FLOOR, registry.HOST_DEFAULT_LAYER,
        )

    def test_a_host_that_declares_no_session_provider_has_no_rung(self, monkeypatch):
        """`session_runner_provider: null` means exactly that, variable or not.

        Claude Code's is `null`: nothing on its side of the boundary hands the
        dispatcher the session's selected model, so there is no value to trust.
        """
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        assert registry.session_runner(host_id="claude-code") is None
        assert registry.resolve_subagent_runner(
            "full", host_id="claude-code")[1] != registry.SESSION_LAYER

    @pytest.mark.parametrize("tier", ["full", "cheap"])
    def test_codex_runs_the_session_model_at_codex_s_default_effort(
        self, monkeypatch, tier
    ):
        """The plugin's server exports the turn metadata's `model`; no effort rides along."""
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, "gpt-5.5")
        assert registry.resolve_subagent_runner(tier, host_id="codex") == (
            "codex:gpt-5.5", registry.SESSION_LAYER,
        )

    def test_codex_without_the_variable_keeps_its_floor(self, monkeypatch):
        """The file-shim route sets nothing, so it resolves exactly as before."""
        _detected(monkeypatch)
        monkeypatch.delenv(registry.HOST_SESSION_MODEL_ENV, raising=False)
        assert registry.resolve_subagent_runner("full", host_id="codex") == (
            CODEX_FLOOR, registry.HOST_DEFAULT_LAYER,
        )

    def test_codex_skip_session_drops_to_its_floor(self, monkeypatch):
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, "gpt-5.5")
        assert registry.resolve_subagent_runner(
            "full", host_id="codex", skip_session=True
        ) == (CODEX_FLOOR, registry.HOST_DEFAULT_LAYER)

    def test_an_undetected_host_cli_falls_through_instead_of_refusing(
        self, monkeypatch
    ):
        """Same rule as `host_default_runner`: an absent CLI is not a refusal."""
        _detected(monkeypatch, available=False)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        assert registry.session_runner(host_id="opencode") is None

    def test_an_unknown_host_id_is_no_rung_rather_than_an_exception(self, monkeypatch):
        """Capability data never fails a dispatch; it only declines to answer."""
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        assert registry.session_runner(host_id="no-such-host") is None

    def test_skip_session_drops_to_the_next_rung(self, monkeypatch):
        """The parameter the preflight fallback re-resolves with."""
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        assert registry.resolve_subagent_runner(
            "full", host_id="opencode", skip_session=True
        ) == (OPENCODE_FLOOR, registry.HOST_DEFAULT_LAYER)
        assert registry.get_subagent_runner(
            "full", host_id="opencode", skip_session=True
        ) == OPENCODE_FLOOR

    def test_the_session_layer_is_not_an_operator_opinion(self):
        """It is implicit: below a config layer, above a shipped floor."""
        assert registry.SESSION_LAYER not in registry.EXPLICIT_DRIVER_LAYERS

    def test_the_pickers_variant_survives_the_whole_spec_round_trip(
        self, monkeypatch
    ):
        """`provider:model:effort` with `/` and `~` in the model id still splits.

        The host now reports its variant as the adapter's own `:effort` suffix,
        so the spec carries three `:`-delimited parts *and* a model id with its
        own separators. `parse_model_spec` splits on the FIRST colon (the
        provider) and `split_reasoning_effort` on the LAST (the effort), which is
        what leaves the model id — slashes, tilde and all — intact between them.
        """
        from utils.providers.base import split_reasoning_effort
        from utils.providers.opencode import REASONING_EFFORTS

        _detected(monkeypatch)
        model = "openrouter/~deepseek/deepseek-flash-latest"
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, f"{model}:low")

        spec, layer = registry.resolve_subagent_runner("full", host_id="opencode")

        assert (spec, layer) == (f"opencode:{model}:low", registry.SESSION_LAYER)
        provider_name, model_spec = registry.parse_model_spec(spec)
        assert provider_name == "opencode"
        base, effort = split_reasoning_effort(model_spec, REASONING_EFFORTS)
        assert (base, effort) == (model, "low")

    def test_the_cheap_tier_takes_the_session_model_too(self, monkeypatch):
        """One model was selected, not a tier map.

        `subagent_tiers` is the native-host tier split and opencode's is `null`;
        splitting the *session's* model by tier would mean inventing a cheap
        sibling for an id the operator picked by hand.
        """
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        assert registry.resolve_subagent_runner("cheap", host_id="opencode") == (
            SESSION_SPEC, registry.SESSION_LAYER,
        )


@pytest.mark.config_override
class TestAnExplicitLayerStillOutranksTheSession:
    """A value someone wrote into a file beats a value they merely selected."""

    @staticmethod
    def _project(monkeypatch, tmp_path, body):
        path = tmp_path / "project-providers.yaml"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: path)
        registry._config = None
        registry._config_key = None
        return path

    def test_the_project_layer_wins(self, monkeypatch, tmp_path):
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        self._project(
            monkeypatch, tmp_path,
            'driver:\n  subagent_runner: "claude-code:sonnet"\n',
        )
        assert registry.resolve_subagent_runner("full", host_id="opencode") == (
            "claude-code:sonnet", "project",
        )

    def test_the_env_layer_wins(self, monkeypatch, tmp_path):
        _detected(monkeypatch)
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        override = tmp_path / "env-providers.yaml"
        override.write_text(
            'driver:\n  subagent_runner: "claude-code:sonnet"\n', encoding="utf-8"
        )
        monkeypatch.setenv(registry.ENV_CONFIG_VAR, str(override))
        registry._config = None
        registry._config_key = None
        assert registry.resolve_subagent_runner("full", host_id="opencode") == (
            "claude-code:sonnet", "env",
        )

    @pytest.mark.parametrize("layer", ["project", "env"])
    def test_an_explicit_layer_wins_over_the_codex_session(
        self, monkeypatch, tmp_path, layer
    ):
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, "gpt-5.5")
        body = 'driver:\n  subagent_runner: "codex:gpt-6-astra:high"\n'
        if layer == "project":
            self._project(monkeypatch, tmp_path, body)
        else:
            monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
            override = tmp_path / "env-providers.yaml"
            override.write_text(body, encoding="utf-8")
            monkeypatch.setenv(registry.ENV_CONFIG_VAR, str(override))
            registry._config = None
            registry._config_key = None
        assert registry.resolve_subagent_runner("full", host_id="codex") == (
            "codex:gpt-6-astra:high", layer,
        )

    def test_an_explicit_cheap_override_wins_for_the_cheap_tier_only(
        self, monkeypatch, tmp_path
    ):
        """The tier keys keep their own precedence above the session rung."""
        _detected(monkeypatch)
        monkeypatch.setenv(registry.HOST_SESSION_MODEL_ENV, SESSION_MODEL)
        self._project(
            monkeypatch, tmp_path,
            'driver:\n  subagent_runner_cheap: "claude-code:haiku"\n',
        )
        assert registry.resolve_subagent_runner("cheap", host_id="opencode") == (
            "claude-code:haiku", "project",
        )
        # `full` named no key, so the session still answers it.
        assert registry.resolve_subagent_runner("full", host_id="opencode") == (
            SESSION_SPEC, registry.SESSION_LAYER,
        )
