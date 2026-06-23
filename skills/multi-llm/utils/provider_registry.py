"""Provider registry and configuration loading."""
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .providers.claude_code import ClaudeCodeProvider
from .providers.codex import CodexProvider
from .providers.cursor_agent import CursorAgentProvider
from .providers.gemini import GeminiProvider
from .providers.kilocode import KiloCodeProvider
from .providers.opencode import OpenCodeProvider
from .providers.base import LLMProvider

# Built-in providers
_PROVIDERS: Dict[str, LLMProvider] = {
    "claude-code": ClaudeCodeProvider(),
    "codex": CodexProvider(),
    "cursor-agent": CursorAgentProvider(),
    "gemini": GeminiProvider(),
    "kilocode": KiloCodeProvider(),
    "opencode": OpenCodeProvider(),
}

_config: Optional[dict] = None


def load_config() -> dict:
    """Load providers.yaml configuration."""
    global _config
    if _config is not None:
        return _config

    config_path = Path(__file__).parent.parent / "providers.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"providers.yaml not found at {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    return _config


def get_provider(name: str) -> Optional[LLMProvider]:
    """Get a provider by name."""
    return _PROVIDERS.get(name)


def parse_model_spec(spec: str) -> Tuple[str, str]:
    """Parse 'provider:model' format.

    Returns (provider_name, model_name).
    If no provider prefix, uses default_provider from config.
    """
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider, model

    config = load_config()
    default = config.get("default_provider", "cursor-agent")
    return default, spec


def get_available_models() -> Dict[str, List[str]]:
    """Get all available models grouped by provider."""
    config = load_config()
    result = {}
    for provider_name, provider_config in config.get("providers", {}).items():
        result[provider_name] = provider_config.get("models", [])
    return result


def get_all_model_specs() -> List[str]:
    """Get all models as provider:model specs for interactive selection."""
    models_by_provider = get_available_models()
    specs = []
    for provider, models in models_by_provider.items():
        for model in models:
            specs.append(f"{provider}:{model}")
    return specs


def get_provider_timeout(provider_name: str) -> int:
    """Get timeout for a specific provider."""
    config = load_config()
    providers = config.get("providers", {})
    if provider_name in providers:
        return providers[provider_name].get("default_timeout", 1200)
    # Fallback to provider's built-in default
    provider = get_provider(provider_name)
    return provider.default_timeout if provider else 1200


def get_provider_max_concurrent(provider_name: str) -> Optional[int]:
    """Get max concurrent limit for a specific provider, if configured.

    Returns None if no limit is set, meaning the provider uses only the global semaphore.
    """
    config = load_config()
    providers = config.get("providers", {})
    if provider_name in providers:
        return providers[provider_name].get("max_concurrent")
    return None


def is_model_valid(spec: str) -> bool:
    """Check if a model spec is valid (exists in config)."""
    provider_name, model = parse_model_spec(spec)
    models_by_provider = get_available_models()
    if provider_name not in models_by_provider:
        return False
    return model in models_by_provider[provider_name]


def get_default_models(mode: Optional[str] = None) -> List[str]:
    """Get default models from config.

    Args:
        mode: Optional mode name (e.g., 'review-plan', 'code-review')
              to get mode-specific defaults.

    Returns:
        List of model specs, or empty list if no defaults configured.
    """
    config = load_config()
    defaults = config.get("defaults", {})

    # Check for mode-specific defaults first
    if mode:
        mode_defaults = defaults.get("modes", {}).get(mode)
        if mode_defaults:
            # Support dict format with 'models' key
            if isinstance(mode_defaults, dict):
                return mode_defaults.get("models", [])
            return mode_defaults

    # Fall back to global defaults
    return defaults.get("models", [])


def has_default_models(mode: Optional[str] = None) -> bool:
    """Check if default models are configured."""
    return len(get_default_models(mode)) > 0


def get_quick_models(mode: Optional[str] = None) -> List[str]:
    """Get quick models from config.

    Args:
        mode: Optional mode name (e.g., 'review-plan', 'code-review')
              to get mode-specific quick models.

    Returns:
        List of model specs, or empty list if no quick_models configured.
    """
    config = load_config()
    defaults = config.get("defaults", {})

    # Check for mode-specific quick models first
    if mode:
        mode_defaults = defaults.get("modes", {}).get(mode)
        if isinstance(mode_defaults, dict) and "quick" in mode_defaults:
            return mode_defaults["quick"]

    # Fall back to global quick_models
    return defaults.get("quick_models", [])


def has_quick_models(mode: Optional[str] = None) -> bool:
    """Check if quick models are configured."""
    return len(get_quick_models(mode)) > 0


def get_provider_max_concurrent(provider_name: str) -> Optional[int]:
    """Get the max concurrent limit for a provider, if configured.

    Returns None if no limit is set (unlimited concurrency within the
    global semaphore).
    """
    config = load_config()
    providers = config.get("providers", {})
    if provider_name in providers:
        return providers[provider_name].get("max_concurrent")
    return None
