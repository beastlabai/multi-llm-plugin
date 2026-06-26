"""Provider registry and configuration loading.

Configuration is assembled from up to three layers, lowest → highest precedence,
each deep-merged over the one below:

1. Built-in base   — ``${skill_dir}/providers.yaml`` (always present).
2. Project-local   — ``<git-root>/.multi-llm/providers.yaml`` (auto-discovered,
   restricted to *selection* keys; a ``providers:`` block here is dropped).
3. Env override    — ``MULTI_LLM_PROVIDERS_CONFIG=/path.yaml`` (escape hatch;
   relative values resolve against CWD).

List values *replace* wholesale on merge; only nested dicts deep-merge. A blank
(``None``) override value is skipped (keeps base); an explicit empty list ``[]``
is a deliberate wipe-out. See the per-project override docs in README.md.
"""
import copy
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .git_utils import get_project_root_from_dir
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

# Process-global config cache. Keyed on the resolved discovery anchor + env (see
# _cache_key) so a CWD-anchored result is not reused for a plan-derived root
# within the same process. Cache captures cwd/env at first load for a given key;
# a later os.chdir/setenv is not reflected for an already-cached key.
_config: Optional[dict] = None
_config_key: Optional[tuple] = None

# Env var names.
ENV_CONFIG_VAR = "MULTI_LLM_PROVIDERS_CONFIG"
ENV_PERMISSIVE_VAR = "MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE"

# Stable, greppable prefix for every non-fatal config warning emitted here. Tests
# capsys-assert on this exact prefix; do not vary it per call site. Fail-fast
# raises are errors, not warnings, and do NOT use this prefix.
CONFIG_WARNING_PREFIX = "multi-llm config warning:"


class ConfigError(RuntimeError):
    """An explicitly-provided override config is present but invalid.

    Subclasses ``RuntimeError`` so orchestrators that already wrap model
    resolution in ``except RuntimeError`` surface it as a clean, named
    fail-fast (printing the message and exiting non-zero) instead of an
    unplanned traceback from deep in the YAML parser.
    """
    pass


def _truthy_env(name: str) -> bool:
    """Return True if env var ``name`` is set to a truthy value."""
    val = os.environ.get(name)
    return val is not None and val.strip().lower() in ("1", "true", "yes", "on")


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base``, returning a new independent dict.

    - Nested dicts recurse; every other type (**including lists**) replaces.
    - An override value of ``None`` is **skipped** (keeps base) so a blank /
      uncommented-but-empty stub key (``models:`` → ``None``) does not blank the
      base. An explicit empty list ``[]`` *does* replace (a deliberate wipe-out).
    - Pure: uses ``copy.deepcopy`` for both the seed and each replaced value so
      the result shares no nested dict/list with either input.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if v is None:                                            # blank → keep base
            continue
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)               # nested dicts merge
        else:
            result[k] = copy.deepcopy(v)                        # lists & scalars replace
    return result


def _anchor_start_dir(anchor: Optional[Union[str, Path]]) -> str:
    """Directory to run git discovery from for an optional discovery anchor.

    ``anchor`` may be a plan-file path or a directory; when omitted, falls back
    to CWD. CWD is a *fallback*, not a guarantee — see _find_project_config.
    """
    if anchor is not None:
        anchor_str = str(anchor)
        if os.path.isdir(anchor_str):
            return anchor_str
        return os.path.dirname(anchor_str) or "."
    return os.getcwd()


def _find_project_config(anchor: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Locate ``<git-root>/.multi-llm/providers.yaml`` for the discovery anchor.

    Discovery is anchored on ``anchor`` (a plan-file path or repo directory)
    when supplied — matching the orchestrators' ``get_project_root(plan_path)``
    — and on CWD otherwise. The anchor → CWD precedence makes the plan-derived
    root authoritative when available while keeping CWD as a convenience
    fallback. Returns None outside a git repository or when the file is absent.
    """
    root = get_project_root_from_dir(_anchor_start_dir(anchor))
    if not root:
        return None
    candidate = Path(root) / ".multi-llm" / "providers.yaml"
    # exists() follows symlinks, so a broken symlink at the candidate path reads
    # as absent and would be skipped silently. Surface it (it exists *as a link*)
    # so the loader's is_symlink() branch can fail fast on the present-but-invalid
    # override.
    if candidate.exists() or candidate.is_symlink():
        return candidate
    return None


def _resolve_env_path() -> Optional[Path]:
    """Resolve ``MULTI_LLM_PROVIDERS_CONFIG`` to an absolute path (CWD-anchored).

    Relative values are allowed and resolved against CWD (the repo root in real
    runs), **not** the git root — keeping the env var usable from CI without
    computing a repo root. Returns None when the var is unset/empty.
    """
    value = os.environ.get(ENV_CONFIG_VAR)
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    # A broken symlink must reach the loader's is_symlink() fail-fast branch, but
    # Path.resolve() collapses it to its non-existent target (no longer a link).
    # Detect the link on the unresolved path and keep it intact so the loader can
    # classify it as present-but-invalid rather than silently absent.
    if os.path.islink(p):
        return p
    return p.resolve()


def _override_paths(anchor: Optional[Union[str, Path]] = None) -> List[Tuple[str, Path]]:
    """Override config layers, lowest→highest precedence: ``[project, env]``.

    Each entry is ``(layer_name, path)``. The project-local path is included
    when it exists *or is a broken symlink*; the env path is included whenever
    the var is set (its existence is validated by the loader so a set-but-missing
    target can warn-and-skip rather than fail fast, while a broken symlink in
    either layer reaches the loader's fail-fast branch).
    """
    paths: List[Tuple[str, Path]] = []
    project = _find_project_config(anchor)
    if project is not None:
        paths.append(("project", project))
    env_path = _resolve_env_path()
    if env_path is not None:
        paths.append(("env", env_path))
    return paths


def _cache_key(anchor: Optional[Union[str, Path]], permissive: bool) -> tuple:
    """Cache key identifying the discovery context (anchor dir + relevant env).

    ``permissive`` is the *effective* (resolved) flag, not just the env var: a
    permissive load that warn-and-skips a broken override must not be served to a
    later strict call with the same anchor/env, which would bypass fail-fast.
    """
    return (
        os.path.abspath(_anchor_start_dir(anchor)),
        os.environ.get(ENV_CONFIG_VAR, ""),
        os.environ.get(ENV_PERMISSIVE_VAR, ""),
        permissive,
    )


def _load_base_config() -> dict:
    """Load and parse the built-in base ``providers.yaml`` (strict).

    Raises FileNotFoundError if the base is missing (matches prior behavior).
    """
    config_path = Path(__file__).parent.parent / "providers.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"providers.yaml not found at {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _load_override_layer(layer_name: str, path: Path, permissive: bool) -> Optional[dict]:
    """Load + validate one override layer; return a mapping to merge or None.

    Distinguishes **absent** (optional → skip) from **present-but-invalid** (the
    user asked for this override → fail fast in strict mode, raising
    :class:`ConfigError` that names the offending file). The opt-in permissive
    mode restores warn-and-skip. The base layer is loaded elsewhere and stays
    strict regardless.
    """
    is_env = layer_name == "env"

    if not path.exists():
        # A broken symlink exists *as a link* → present-invalid (fail fast).
        if path.is_symlink():
            msg = f"override config is a broken symlink: {path}"
            if permissive:
                print(f"{CONFIG_WARNING_PREFIX} {msg} — skipping", file=sys.stderr)
                return None
            raise ConfigError(f"multi-llm config error: {msg}")
        # Truly absent. The env layer warns on a set-but-missing target; the
        # project layer never reaches here (only returned when it exists).
        if is_env:
            print(
                f"{CONFIG_WARNING_PREFIX} {ENV_CONFIG_VAR} points at a missing file: "
                f"{path} — skipping",
                file=sys.stderr,
            )
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        if permissive:
            print(
                f"{CONFIG_WARNING_PREFIX} ignoring malformed YAML in override config: "
                f"{path} ({e})",
                file=sys.stderr,
            )
            return None
        raise ConfigError(
            f"multi-llm config error: malformed YAML in override config: {path} ({e})"
        ) from e
    except OSError as e:
        # Covers IsADirectoryError / PermissionError / a file that vanished after
        # the existence check — present-invalid → fail fast (unless permissive).
        if permissive:
            print(
                f"{CONFIG_WARNING_PREFIX} ignoring unreadable override config: "
                f"{path} ({e})",
                file=sys.stderr,
            )
            return None
        raise ConfigError(
            f"multi-llm config error: could not read override config: {path} ({e})"
        ) from e

    if data is None:
        # Empty / all-comments file → no-op merge.
        return None

    if not isinstance(data, dict):
        if permissive:
            print(
                f"{CONFIG_WARNING_PREFIX} ignoring non-mapping root in override config: "
                f"{path} — skipping",
                file=sys.stderr,
            )
            return None
        raise ConfigError(
            f"multi-llm config error: override config must contain a top-level "
            f"mapping, got {type(data).__name__}: {path}"
        )

    # Trust model: an auto-discovered (project-local) layer may NOT introduce or
    # mutate provider *capabilities* — a freshly cloned repo could otherwise ship
    # one. A `providers:` block in the project-local layer is dropped before
    # merge with a warning. The explicit env layer (an out-of-tree path the user
    # chose) keeps the full deep-merge. NOTE: `command`/argv is NEVER honored
    # from config in any layer — provider binaries stay hardcoded in their
    # provider classes; `command` here is documentation-only. Any future change
    # that feeds config into command construction MUST revisit this trust model.
    if not is_env and "providers" in data:
        print(
            f"{CONFIG_WARNING_PREFIX} ignoring 'providers:' block in "
            f"auto-discovered config: {path}",
            file=sys.stderr,
        )
        data = {k: v for k, v in data.items() if k != "providers"}

    return data


def load_base_config() -> dict:
    """Public accessor for the built-in base ``providers.yaml`` (no overrides).

    Used by ``--init`` so the interactive flow reflects the plugin's canonical
    catalog (curated models, provider metadata, declaration order) independent of
    any project-local override or ``MULTI_LLM_PROVIDERS_CONFIG`` that may have
    altered the merged runtime view from :func:`load_config`.
    """
    return _load_base_config()


def load_config(anchor: Optional[Union[str, Path]] = None,
                permissive: Optional[bool] = None) -> dict:
    """Load layered providers configuration (base → project-local → env).

    Args:
        anchor: Optional plan-file path or target-repo directory. When supplied,
            project-local discovery resolves the git root from it (matching the
            orchestrators' plan-derived root); when omitted, falls back to CWD.
        permissive: When True, restores warn-and-skip on a present-but-invalid
            override instead of failing fast. Defaults to the
            ``MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE`` env var (strict otherwise).

    Returns:
        The merged config dict. Cached per resolved anchor/env within a process.
    """
    global _config, _config_key
    # Resolve the effective permissive flag *before* the cache lookup so it is
    # part of the key — otherwise a prior permissive load that degraded past a
    # broken override could be served to a later strict call, bypassing
    # fail-fast.
    if permissive is None:
        permissive = _truthy_env(ENV_PERMISSIVE_VAR)

    key = _cache_key(anchor, permissive)
    if _config is not None and _config_key == key:
        return _config

    config = _load_base_config()
    for layer_name, path in _override_paths(anchor):
        data = _load_override_layer(layer_name, path, permissive)
        if data:
            config = _deep_merge(config, data)

    _config = config
    _config_key = key
    return config


def get_provider(name: str) -> Optional[LLMProvider]:
    """Get a provider by name."""
    return _PROVIDERS.get(name)


def parse_model_spec(spec: str, anchor: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
    """Parse 'provider:model' format.

    Returns (provider_name, model_name).
    If no provider prefix, uses default_provider from config.
    """
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider, model

    config = load_config(anchor=anchor)
    default = config.get("default_provider", "cursor-agent")
    return default, spec


def get_available_models(anchor: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
    """Get all available models grouped by provider.

    Args:
        anchor: Optional discovery anchor (plan path / repo dir) threaded to
                load_config for plan-derived project-local discovery.
    """
    config = load_config(anchor=anchor)
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


def _collect_configured_specs(anchor: Optional[Union[str, Path]] = None) -> set:
    """Canonical set of provider:model specs the user explicitly configured.

    Walks the merged defaults.models + defaults.quick_models + every
    defaults.modes.<mode> entry (both the bare-list shape and the
    {"models": [...], "quick": [...]} dict shape), then canonicalizes each raw
    entry through parse_model_spec so a bare (prefix-less) id resolves against the
    same anchor-aware default_provider and compares equal to a resolved warned spec.
    """
    config = load_config(anchor=anchor)
    defaults = config.get("defaults", {}) or {}
    raw: List[str] = []
    raw += defaults.get("models", []) or []
    raw += defaults.get("quick_models", []) or []
    for mode_val in (defaults.get("modes", {}) or {}).values():
        if isinstance(mode_val, dict):            # {models, quick} shape
            raw += mode_val.get("models", []) or []
            raw += mode_val.get("quick", []) or []
        elif isinstance(mode_val, list):          # bare-list shape
            raw += mode_val
        # any other shape (str/None/etc.) is ignored, not exploded
    canon = set()
    for spec in raw:
        if not isinstance(spec, str) or not spec.strip():
            continue
        provider, model = parse_model_spec(spec, anchor=anchor)
        canon.add(f"{provider}:{model}")
    return canon


def is_model_valid(spec: str, anchor: Optional[Union[str, Path]] = None, *,
                   configured: Optional[set] = None) -> bool:
    """Check if a model spec is valid (exists in config).

    A spec is valid if either (a) the user explicitly configured it in
    ``defaults.*`` — so a model deliberately placed in the config is honoured even
    when it is absent from the provider catalog (suppressing the spurious "unknown
    model" warning) — or (b) it exists in the provider catalog.

    Args:
        spec: Model spec ('provider:model' or a bare, prefix-less model name).
        anchor: Optional discovery anchor (plan path / repo dir) threaded to
                parse_model_spec/get_available_models so a bare spec resolves its
                default_provider against the plan-derived project-local config
                rather than the CWD-anchored one.
        configured: Optional override of the canonical configured set (used by
                tests / explicit injection). When ``None``, the set is built
                internally from the merged ``defaults.*`` via
                ``_collect_configured_specs``.

    The spec is canonicalized through ``parse_model_spec`` *before* the membership
    test so a bare spec compares canonically against the canonical configured set.
    Provider-existence is NOT short-circuited: a spec naming a non-existent
    provider only matches ``configured`` if that exact typo was written into
    ``defaults.*``; otherwise the catalog-membership path still returns False.
    """
    if configured is None:
        configured = _collect_configured_specs(anchor=anchor)
    provider_name, model = parse_model_spec(spec, anchor=anchor)
    if f"{provider_name}:{model}" in configured:   # canonical provider:model
        return True
    models_by_provider = get_available_models(anchor=anchor)
    if provider_name not in models_by_provider:
        return False
    return model in models_by_provider[provider_name]


def get_default_models(mode: Optional[str] = None,
                       anchor: Optional[Union[str, Path]] = None) -> List[str]:
    """Get default models from config.

    Args:
        mode: Optional mode name (e.g., 'review-plan', 'code-review')
              to get mode-specific defaults.
        anchor: Optional discovery anchor (plan path / repo dir) threaded to
                load_config for plan-derived project-local discovery.

    Returns:
        List of model specs, or empty list if no defaults configured.
    """
    config = load_config(anchor=anchor)
    defaults = config.get("defaults", {})

    # Check for mode-specific defaults first. Test for the mode KEY's presence
    # (not truthiness): an explicit empty list `defaults.modes.<mode>: []` is a
    # deliberate clear of that mode entry and must return [] rather than falling
    # through to defaults.models (which would re-select models the project tried
    # to disable). A missing key still falls back to the global defaults.
    modes = defaults.get("modes", {})
    if mode and mode in modes:
        mode_defaults = modes[mode]
        # Support dict format with 'models' key
        if isinstance(mode_defaults, dict):
            return mode_defaults.get("models", [])
        return mode_defaults

    # Fall back to global defaults
    return defaults.get("models", [])


def has_default_models(mode: Optional[str] = None,
                       anchor: Optional[Union[str, Path]] = None) -> bool:
    """Check if default models are configured."""
    return len(get_default_models(mode, anchor=anchor)) > 0


def get_quick_models(mode: Optional[str] = None,
                     anchor: Optional[Union[str, Path]] = None) -> List[str]:
    """Get quick models from config.

    Args:
        mode: Optional mode name (e.g., 'review-plan', 'code-review')
              to get mode-specific quick models.
        anchor: Optional discovery anchor (plan path / repo dir) threaded to
                load_config for plan-derived project-local discovery.

    Returns:
        List of model specs, or empty list if no quick_models configured.
    """
    config = load_config(anchor=anchor)
    defaults = config.get("defaults", {})

    # Check for mode-specific quick models first
    if mode:
        mode_defaults = defaults.get("modes", {}).get(mode)
        if isinstance(mode_defaults, dict) and "quick" in mode_defaults:
            return mode_defaults["quick"]

    # Fall back to global quick_models
    return defaults.get("quick_models", [])


def has_quick_models(mode: Optional[str] = None,
                     anchor: Optional[Union[str, Path]] = None) -> bool:
    """Check if quick models are configured."""
    return len(get_quick_models(mode, anchor=anchor)) > 0


def get_provider_max_concurrent(provider_name: str) -> Optional[int]:
    """Get the max concurrent limit for a provider, if configured.

    Returns None if no limit is set, meaning the provider uses only the global
    semaphore (unlimited concurrency within it).
    """
    config = load_config()
    providers = config.get("providers", {})
    if provider_name in providers:
        return providers[provider_name].get("max_concurrent")
    return None
