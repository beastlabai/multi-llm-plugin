"""Provider registry and configuration loading.

Configuration is assembled from up to three layers, lowest → highest precedence,
each deep-merged over the one below:

1. Built-in base   — ``${skill_dir}/providers.yaml`` (always present).
2. Project-local   — ``<git-root>/.multi-llm/providers.yaml`` (auto-discovered).
3. Env override    — ``MULTI_LLM_PROVIDERS_CONFIG=/path.yaml`` (escape hatch;
   relative values resolve against CWD).

Both override layers deep-merge their **full** contents — including a
``providers:`` block — over base, identical to each other. ``command`` is
documentation-only and is NEVER executed (binaries are hardcoded in
``utils/providers/``), so no execution metadata is honored from config in any
layer. A merged provider name with no hardcoded adapter is ignored at runtime
(filtered out of :func:`get_available_models` *and* configured-spec validation); a
non-mapping provider value is restored from base when it shadows a known provider
(else dropped) at load time. ``--init`` writes layer 2 by uncommenting an inert template.

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
from .providers.agy import AgyProvider
from .providers.aider import AiderProvider
from .providers.claude_code import ClaudeCodeProvider
from .providers.cline import ClineProvider
from .providers.codex import CodexProvider
from .providers.cursor_agent import CursorAgentProvider
from .providers.gemini import GeminiProvider
from .providers.goose import GooseProvider
from .providers.grok import GrokProvider
from .providers.kilocode import KiloCodeProvider
from .providers.muse import MuseProvider
from .providers.opencode import OpenCodeProvider
from .providers.pi import PiProvider
from .providers.base import LLMProvider

# Built-in providers
_PROVIDERS: Dict[str, LLMProvider] = {
    "agy": AgyProvider(),
    "aider": AiderProvider(),
    "claude-code": ClaudeCodeProvider(),
    "cline": ClineProvider(),
    "codex": CodexProvider(),
    "cursor-agent": CursorAgentProvider(),
    "gemini": GeminiProvider(),
    "goose": GooseProvider(),
    "grok": GrokProvider(),
    "kilocode": KiloCodeProvider(),
    "muse": MuseProvider(),
    "opencode": OpenCodeProvider(),
    "pi": PiProvider(),
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

# Provider names already warned about as "no hardcoded adapter" (re-init drift),
# so the read-site filter warns once per name per process instead of every call.
_WARNED_UNKNOWN_PROVIDERS: set = set()


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

    # Trust model: BOTH override layers deep-merge their full contents (including a
    # `providers:` block) over base, identically. The residual surface is
    # availability-only metadata (timeouts/concurrency/model lists) — `command`/argv
    # is NEVER honored from config in any layer (provider binaries stay hardcoded in
    # their provider classes; `command` here is documentation-only), an unknown
    # provider name has no adapter and is ignored at runtime, and a non-mapping
    # provider value is dropped by the load-time guard in load_config. Any future
    # change that feeds config into command construction MUST revisit this.
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

    base_config = _load_base_config()
    config = base_config
    driver_overridden = False
    for layer_name, path in _override_paths(anchor):
        data = _load_override_layer(layer_name, path, permissive)
        if data:
            if "driver" in data:
                driver_overridden = True
            config = _deep_merge(config, data)

    # D3: `driver:` validation is fail-fast, and only for an EXPLICIT override.
    # Base is ours (a broken base is a test failure, not a user's problem), and
    # validating it here would turn every unrelated invocation into a config
    # error. An override layer asked for these values, so a bad one raises
    # ConfigError naming the key, the value and the reason at load time rather
    # than surfacing as a mis-dispatched subagent mid-phase. Permissive mode
    # deliberately does NOT relax this: a silently ignored runner spec degrades
    # to "no runner configured", which is the failure this check exists for.
    if driver_overridden:
        validate_driver_block(config.get("driver"))

    # Load-time guard (post-merge): a clone-authored override can deep-merge a
    # malformed (non-mapping) provider value, e.g. `providers: {claude-code: "x"}`,
    # which REPLACES the base mapping during deep-merge and would later
    # AttributeError on `.get(...)`. Sanitize each such entry so the rest of the
    # config still loads and the residual stays availability-only: for a KNOWN base
    # provider, RESTORE its base mapping — the malformed scalar must not erase the
    # built-in catalog/metadata for the whole repo; for an unknown name (nothing to
    # inherit) drop it. ``base_config`` is the untouched base load (deep-merge seeds
    # from a deepcopy, so it is never mutated), giving the pristine entry to restore.
    base_providers = base_config.get("providers")
    base_providers = base_providers if isinstance(base_providers, dict) else {}
    providers = config.get("providers")
    if isinstance(providers, dict):
        for name in [k for k, v in providers.items() if not isinstance(v, dict)]:
            base_entry = base_providers.get(name)
            restore = isinstance(base_entry, dict)
            print(
                f"{CONFIG_WARNING_PREFIX} ignoring malformed provider '{name}' in "
                f"merged config (expected a mapping, got "
                f"{type(providers[name]).__name__}) — "
                f"{'restoring base definition' if restore else 'dropping it'}",
                file=sys.stderr,
            )
            if restore:
                providers[name] = copy.deepcopy(base_entry)
            else:
                del providers[name]

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


def _provider_has_adapter(provider_name: str) -> bool:
    """Read-site drift filter: True iff ``provider_name`` has a hardcoded adapter.

    A config (e.g. one an OLDER plugin ``--init`` wrote, or a stale clone) can carry
    a ``providers:`` block or a ``defaults.*`` spec for a name the current base
    removed/renamed — one with no hardcoded adapter. Such a name merges but can never
    run (``get_provider`` returns None), so it MUST NOT be surfaced as a usable model
    source NOR validate as an explicitly-configured spec. Returns False (and warns
    once per name per process via ``_WARNED_UNKNOWN_PROVIDERS``) for such names, so
    BOTH read sites — :func:`get_available_models` and :func:`_collect_configured_specs`
    — filter them identically.
    """
    if get_provider(provider_name) is not None:
        return True
    if provider_name not in _WARNED_UNKNOWN_PROVIDERS:
        _WARNED_UNKNOWN_PROVIDERS.add(provider_name)
        print(
            f"{CONFIG_WARNING_PREFIX} provider '{provider_name}' in your "
            f"config is no longer supported — re-init to refresh",
            file=sys.stderr,
        )
    return False


def get_available_models(anchor: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
    """Get all available models grouped by provider.

    Args:
        anchor: Optional discovery anchor (plan path / repo dir) threaded to
                load_config for plan-derived project-local discovery.
    """
    config = load_config(anchor=anchor)
    result = {}
    for provider_name, provider_config in config.get("providers", {}).items():
        # Skip a merged provider name with no hardcoded adapter (warn once) so it
        # contributes no models — see _provider_has_adapter for the rationale.
        if not _provider_has_adapter(provider_name):
            continue
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

    Specs whose provider has **no hardcoded adapter** (a stale clone naming a
    removed/renamed provider) are filtered out via ``_provider_has_adapter`` — the
    same read-site drift filter ``get_available_models`` applies — so an orphan spec
    in ``defaults.*`` is NOT treated as "explicitly configured" and a model that
    cannot run is never validated.
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
        if not _provider_has_adapter(provider):       # orphan provider → filter out
            continue
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
    A spec whose provider has **no hardcoded adapter** (a removed/renamed key or a
    typo) is filtered out of the configured set by ``_collect_configured_specs``
    (warning once), so even an orphan spec written into ``defaults.*`` does NOT
    validate as "explicitly configured"; the catalog-membership path likewise skips
    it, so a model that cannot run is rejected rather than accepted on a stale config.
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


# ---------------------------------------------------------------------------
# driver: — the `--exec-subagents` dispatch config (D3, D3o)
# ---------------------------------------------------------------------------

#: The base `driver:` block's keys, with the types each accepts. This is the
#: closed set: an unknown key under `driver:` is REJECTED rather than ignored, so
#: a typo (`subagent_runners:`) is not silently a no-op.
DRIVER_RUNNER_KEYS = ("subagent_runner", "subagent_runner_cheap")
DRIVER_POSITIVE_INT_KEYS = (
    "max_concurrent_subagents",
    # D3h: write phases get their OWN ceiling, defaulting to 1, because the
    # batchers' partitioning invariant (one shared artifact per apply phase, one
    # file per code-fix batch) is not the read path's fan-out. Raising it only
    # *permits* parallelism; the dispatch-time disjointness check still admits it.
    "max_concurrent_write_subagents",
    "subagent_retries",
)
DRIVER_OPTIONAL_INT_KEYS = ("subagent_timeout_s",)
#: D3l: `allow`/`deny` glob lists resolved against the pinned repo root, enforced
#: pre-dispatch (against the batch's declared file set) and post-run (against the
#: set the snapshot diff actually observed).
DRIVER_PATH_POLICY_KEYS = ("write_paths",)
DRIVER_KEYS = (
    DRIVER_RUNNER_KEYS
    + DRIVER_POSITIVE_INT_KEYS
    + DRIVER_OPTIONAL_INT_KEYS
    + DRIVER_PATH_POLICY_KEYS
)

#: The shipped write policy (D3l). `allow` is the repository; `deny` is what a
#: code-editing batch has no business touching. `.multi-llm/**` is denied to the
#: RUNNER precisely because the EXECUTOR owns those bytes — orchestrator state,
#: task files, snapshots, audit records and reports are written by the invoker
#: under D3f containment, never by a subagent.
DEFAULT_WRITE_PATHS: Dict[str, List[str]] = {
    "allow": ["**"],
    "deny": [
        ".git/**",
        ".multi-llm/**",
        "**/.env",
        "**/.env.*",
        "**/*.pem",
        "**/id_rsa*",
        "**/.ssh/**",
        "**/.aws/**",
        "**/.npmrc",
        "**/node_modules/**",
    ],
}

#: Values used when the merged config carries no `driver:` block at all (a test
#: fixture config, or a base file predating D3). The shipped base is fully
#: populated, so in a real run these are never reached.
DRIVER_FALLBACKS: Dict[str, object] = {
    "subagent_runner": "claude-code:sonnet",
    "subagent_runner_cheap": "claude-code:haiku",
    "max_concurrent_subagents": 4,
    "max_concurrent_write_subagents": 1,
    "subagent_timeout_s": None,
    "subagent_retries": 1,
    "write_paths": DEFAULT_WRITE_PATHS,
}


def _driver_runner_error(key: str, value: object, reason: str) -> str:
    """One rejection surface for a bad `driver.*` runner spec (D3)."""
    return (
        f"multi-llm config error: driver.{key} is {value!r} — {reason}. "
        f"Set it to a provider:model spec whose provider has a built-in adapter "
        f"(e.g. claude-code:sonnet, codex:gpt-5.6-sol, opencode:<model>), or "
        f"remove the key to inherit the default."
    )


def validate_driver_block(driver: object) -> None:
    """Validate a merged ``driver:`` block, raising :class:`ConfigError`.

    Fail-fast and total: every field is checked, unknown keys are rejected, and
    each message names the key, the value and the reason. Called from
    :func:`load_config` whenever an **override** layer carried a ``driver:``
    block, so a bad value surfaces at load time rather than as a mis-dispatched
    subagent mid-phase.

    A runner spec is checked structurally here (parses as ``provider:model``, and
    that provider has a hardcoded adapter). Catalog membership
    (:func:`is_model_valid`) and the profile check (D3e) happen at resolution
    time, because both read the merged config this function is validating.
    """
    if driver is None:
        return
    if not isinstance(driver, dict):
        raise ConfigError(
            f"multi-llm config error: `driver:` must be a mapping, got "
            f"{type(driver).__name__}"
        )

    unknown = [key for key in driver if key not in DRIVER_KEYS]
    if unknown:
        raise ConfigError(
            f"multi-llm config error: unknown key(s) under `driver:`: "
            f"{', '.join(sorted(unknown))}. Known keys: "
            f"{', '.join(DRIVER_KEYS)}. A typo here would otherwise be a silent "
            f"no-op, so it is rejected instead of ignored."
        )

    for key in DRIVER_RUNNER_KEYS:
        if key not in driver:
            continue
        value = driver[key]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(_driver_runner_error(key, value, "expected a non-empty string"))
        provider_name, _, model = value.partition(":")
        if not provider_name.strip() or not model.strip():
            raise ConfigError(_driver_runner_error(
                key, value,
                "it must be an explicit provider:model spec (a bare model name "
                "cannot be resolved for a runner)",
            ))
        # Deliberately NOT warn-and-filter: a filtered runner spec silently
        # degrades to "no runner configured" and resurfaces mid-phase, so the
        # driver validator raises where the fan-out path warns. The fan-out
        # behavior (_provider_has_adapter's warn-once) is untouched.
        if get_provider(provider_name.strip()) is None:
            raise ConfigError(_driver_runner_error(
                key, value,
                f"provider {provider_name.strip()!r} has no built-in adapter "
                f"(reason code: unknown_provider) — run --init to refresh a "
                f"stale config",
            ))

    for key in DRIVER_POSITIVE_INT_KEYS:
        if key not in driver:
            continue
        value = driver[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(
                f"multi-llm config error: driver.{key} is {value!r} — expected a "
                f"positive integer."
            )

    for key in DRIVER_OPTIONAL_INT_KEYS:
        if key not in driver:
            continue
        value = driver[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(
                f"multi-llm config error: driver.{key} is {value!r} — expected a "
                f"positive integer or null (null = the runner provider's "
                f"default_timeout)."
            )

    for key in DRIVER_PATH_POLICY_KEYS:
        if key not in driver:
            continue
        _validate_path_policy(key, driver[key])


def _validate_path_policy(key: str, value: object) -> None:
    """Validate one ``allow``/``deny`` glob-list policy block (D3l).

    Fail-fast for the same reason a bad runner spec is: a policy that silently
    parsed as "no rules" would be indistinguishable from a policy that allows
    everything, and the difference is whether a headless write run is contained.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise ConfigError(
            f"multi-llm config error: driver.{key} is {value!r} — expected a "
            f"mapping with `allow:` and/or `deny:` glob lists."
        )
    unknown = [name for name in value if name not in ("allow", "deny")]
    if unknown:
        raise ConfigError(
            f"multi-llm config error: unknown key(s) under driver.{key}: "
            f"{', '.join(sorted(str(u) for u in unknown))}. Known keys: allow, deny."
        )
    for name in ("allow", "deny"):
        if name not in value or value[name] is None:
            continue
        patterns = value[name]
        if not isinstance(patterns, list) or not all(
            isinstance(item, str) and item.strip() for item in patterns
        ):
            raise ConfigError(
                f"multi-llm config error: driver.{key}.{name} is {patterns!r} — "
                f"expected a list of non-empty glob strings (e.g. "
                f'["src/**", "docs/*.md"]).'
            )


def get_driver_config(anchor: Optional[Union[str, Path]] = None) -> dict:
    """The merged ``driver:`` block, with every key present (D3).

    Reads go through this and the typed accessors below — never ad-hoc
    ``config["driver"]`` indexing — so a config layer that omitted a key still
    yields a complete block.
    """
    config = load_config(anchor=anchor)
    driver = config.get("driver")
    driver = dict(driver) if isinstance(driver, dict) else {}
    merged = dict(DRIVER_FALLBACKS)
    for key in DRIVER_KEYS:
        if key in driver and driver[key] is not None:
            merged[key] = driver[key]
        elif key in driver and key in DRIVER_OPTIONAL_INT_KEYS:
            merged[key] = None
    return merged


def get_driver_provenance(
    key: str, anchor: Optional[Union[str, Path]] = None
) -> str:
    """Which layer supplied ``driver.<key>``: ``env``/``project``/``base``/``default``.

    D3o: `--status` and `install-host --check` print the resolved runner together
    with the layer that supplied it, because "I set it and nothing changed" is
    otherwise a support question. Highest-precedence layer that names the key
    wins, matching the merge order exactly.
    """
    permissive = _truthy_env(ENV_PERMISSIVE_VAR)
    for layer_name, path in reversed(_override_paths(anchor)):
        data = _load_override_layer(layer_name, path, permissive)
        driver = (data or {}).get("driver")
        if isinstance(driver, dict) and driver.get(key) is not None:
            return layer_name
    base_driver = _load_base_config().get("driver")
    if isinstance(base_driver, dict) and key in base_driver:
        return "base"
    return "default"


def get_subagent_runner(
    tier: str = "full", anchor: Optional[Union[str, Path]] = None
) -> str:
    """The configured runner spec for a capability tier (D2e/D3).

    ``cheap`` resolves to ``driver.subagent_runner_cheap`` and falls back to
    ``driver.subagent_runner``; every other tier resolves to
    ``driver.subagent_runner``.
    """
    driver = get_driver_config(anchor=anchor)
    if tier == "cheap":
        cheap = driver.get("subagent_runner_cheap")
        if isinstance(cheap, str) and cheap.strip():
            return cheap
    return str(driver.get("subagent_runner") or DRIVER_FALLBACKS["subagent_runner"])


def get_max_concurrent_subagents(anchor: Optional[Union[str, Path]] = None) -> int:
    """``driver.max_concurrent_subagents`` — the *global* executor gate."""
    return int(get_driver_config(anchor=anchor)["max_concurrent_subagents"])


def get_max_concurrent_write_subagents(
    anchor: Optional[Union[str, Path]] = None
) -> int:
    """``driver.max_concurrent_write_subagents`` — the write gate (default 1, D3h).

    A separate key from the read gate on purpose: the write phases preserve the
    batchers' partitioning invariant (apply phases edit one shared artifact,
    code-fix batches are partitioned by file), so the read path's fan-out would be
    two agents editing one file. Raising this only *permits* parallelism —
    ``can_parallelize`` and the dispatch-time pairwise-disjoint admission check
    still have to agree, and a shared-artifact phase is pinned to 1 regardless.
    """
    return max(1, int(get_driver_config(anchor=anchor)["max_concurrent_write_subagents"]))


def get_write_paths(anchor: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
    """``driver.write_paths`` — the declared write policy (D3l), always complete.

    Returns a dict with both ``allow`` and ``deny`` present, falling back to
    :data:`DEFAULT_WRITE_PATHS` per key, so a config that set only ``deny`` still
    yields an ``allow`` list rather than an accidental deny-all.
    """
    configured = get_driver_config(anchor=anchor).get("write_paths")
    configured = configured if isinstance(configured, dict) else {}
    resolved: Dict[str, List[str]] = {}
    for name in ("allow", "deny"):
        patterns = configured.get(name)
        if isinstance(patterns, list) and patterns:
            resolved[name] = [str(item) for item in patterns]
        else:
            resolved[name] = list(DEFAULT_WRITE_PATHS[name])
    return resolved


def get_subagent_retries(anchor: Optional[Union[str, Path]] = None) -> int:
    """``driver.subagent_retries`` — transport-shaped re-dispatches only."""
    return int(get_driver_config(anchor=anchor)["subagent_retries"])


def get_subagent_timeout(
    provider_name: str, anchor: Optional[Union[str, Path]] = None
) -> int:
    """Per-dispatch timeout: ``driver.subagent_timeout_s`` or the provider default."""
    configured = get_driver_config(anchor=anchor)["subagent_timeout_s"]
    if isinstance(configured, int) and not isinstance(configured, bool):
        return configured
    return get_provider_timeout(provider_name)


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
