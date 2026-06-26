#!/usr/bin/env python3
"""Interactive utilities for terminal-based user input."""

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from .provider_registry import load_config

# Sentinel action ids returned by select_with_actions (see ActionSelection).
ACTION_SHOW_ALL = "show_all"
ACTION_ENTER_MANUAL = "enter_manual"


class _Unavailable:
    """Sentinel: a picker backend (gum/fzf) could not be used at all.

    This distinguishes "the backend tool is not installed / failed to launch"
    (the cascade should fall through to the next backend) from "the backend ran
    and the user made an empty selection / cancelled / hit Esc" (which is a real,
    intentional result and must NOT fall through — it stops the cascade and yields
    an empty selection). Backend helpers return ``UNAVAILABLE`` only in the former
    case; otherwise they return a concrete (possibly empty) selection.

    This mechanism is shared by the multi-select cascade and is reusable by the
    single-select cascade (``select_one`` and its backends).
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNAVAILABLE"

    def __bool__(self) -> bool:
        return False


# Singleton sentinel returned by picker backends that are unavailable.
UNAVAILABLE = _Unavailable()


def is_tty() -> bool:
    """Check if stdin/stdout are connected to a TTY."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _try_gum_choose(options: List[str], prompt: str):
    """Try gum choose for multi-select.

    Returns:
        - ``UNAVAILABLE`` if gum is not installed or the subprocess fails to run
          (the caller should fall through to the next backend).
        - A list of selected items otherwise — possibly **empty** when the user
          made no selection / cancelled / hit Esc (a real result; the caller must
          NOT fall through to another backend).
    """
    if not shutil.which("gum"):
        return UNAVAILABLE

    try:
        # gum choose --no-limit allows multi-select
        # --header provides the prompt text
        result = subprocess.run(
            ["gum", "choose", "--no-limit", "--header", prompt] + options,
            capture_output=True,
            text=True,
            timeout=120,
            check=False  # We handle returncode manually
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return UNAVAILABLE

    # gum ran: a non-zero exit / empty stdout is a deliberate cancel (Esc), which
    # is an empty selection — NOT an "unavailable" fall-through.
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().split("\n")
    return []


def _try_fzf_multi(options: List[str], prompt: str):
    """Try fzf with multi-select.

    Returns:
        - ``UNAVAILABLE`` if fzf is not installed or the subprocess fails to run
          (the caller should fall through to the next backend).
        - A list of selected items otherwise — possibly **empty** when the user
          made no selection / cancelled / hit Esc (a real result; the caller must
          NOT fall through to another backend).
    """
    if not shutil.which("fzf"):
        return UNAVAILABLE

    try:
        # fzf -m enables multi-select with TAB
        # --header provides the prompt
        # --bind 'enter:accept' ensures single enter accepts
        input_text = "\n".join(options)
        result = subprocess.run(
            ["fzf", "-m", "--header", prompt, "--bind", "enter:accept"],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=120,
            check=False  # We handle returncode manually
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return UNAVAILABLE

    # fzf ran: a non-zero exit (Esc → 130) / empty stdout is a deliberate cancel,
    # which is an empty selection — NOT an "unavailable" fall-through.
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().split("\n")
    return []


def _numbered_prompt(options: List[str], prompt: str) -> List[str]:
    """Fallback: numbered list with space-separated input."""
    print(f"\n{prompt}")
    print("-" * 40)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("-" * 40)
    print("Enter numbers separated by spaces (e.g., '1 3 5'):")

    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return []

    if not user_input:
        return []

    selected = []
    for part in user_input.split():
        try:
            idx = int(part)
            if 1 <= idx <= len(options):
                selected.append(options[idx - 1])
        except ValueError:
            continue

    return selected


def select_models_interactive(
    available_models: List[str],
    prompt: str = "Select models for review (multi-select):"
) -> List[str]:
    """
    Interactive model selection with graceful fallbacks.

    Order of attempts:
    1. gum choose --no-limit (if available)
    2. fzf -m (if available)
    3. Numbered prompt with input()

    Args:
        available_models: List of model names to choose from
        prompt: Prompt text to display

    Returns:
        List of selected model names (may be empty if user cancels)

    Raises:
        RuntimeError: If not running in a TTY
    """
    if not is_tty():
        raise RuntimeError(
            "No TTY available for interactive selection. "
            "Use --models flag to specify models explicitly."
        )

    # Cascade gum → fzf → numbered, falling through ONLY when a backend is
    # UNAVAILABLE. A backend that ran returns a (possibly empty) list; an empty
    # list is a deliberate cancel and is returned as-is, not re-prompted.
    result = _try_gum_choose(available_models, prompt)
    if result is not UNAVAILABLE:
        return result

    result = _try_fzf_multi(available_models, prompt)
    if result is not UNAVAILABLE:
        return result

    # Fall back to numbered prompt
    return _numbered_prompt(available_models, prompt)


def select_multi(
    options: List[str],
    prompt: str = "Select options (multi-select):"
) -> List[str]:
    """
    Generic multi-select with graceful fallbacks.

    This is a lower-level utility used by select_models_two_step().

    Order of attempts:
    1. gum choose --no-limit (if available)
    2. fzf -m (if available)
    3. Numbered prompt with input()

    Args:
        options: List of options to choose from
        prompt: Prompt text to display

    Returns:
        List of selected options (may be empty if user cancels)

    Raises:
        RuntimeError: If not running in a TTY
    """
    if not is_tty():
        raise RuntimeError(
            "No TTY available for interactive selection. "
            "Use --models flag to specify models explicitly."
        )

    # Cascade gum → fzf → numbered, but ONLY fall through when a backend is
    # UNAVAILABLE. A backend that actually ran returns a (possibly empty) list;
    # an empty list means the user cancelled / hit Esc / selected nothing, which
    # must be returned as-is — NOT re-prompted via the next backend.
    result = _try_gum_choose(options, prompt)
    if result is not UNAVAILABLE:
        return result

    result = _try_fzf_multi(options, prompt)
    if result is not UNAVAILABLE:
        return result

    # Fall back to numbered prompt
    return _numbered_prompt(options, prompt)


def select_models_two_step(anchor: Optional[str] = None) -> List[str]:
    """
    Two-step interactive model selection: providers first, then models.

    Step 1: User selects which providers to use (only shows available providers)
    Step 2: For each selected provider, user selects which models to use

    Args:
        anchor: Optional discovery anchor (the plan-file path or target-repo
            directory). Threaded into config discovery so per-project overrides
            resolve from the plan-derived git root rather than CWD.

    Returns:
        List of model specs in provider:model format (e.g., ["gemini:gemini-2.5-flash"])
    """
    from .provider_registry import get_provider

    config = load_config(anchor=anchor)
    providers = config.get("providers", {})

    if not providers:
        raise RuntimeError("No providers configured in providers.yaml")

    # Step 1: Select providers (only show available ones)
    provider_choices = []
    provider_name_map = {}
    for name, cfg in providers.items():
        # Check if provider is available
        provider = get_provider(name)
        if provider is None or not provider.is_available():
            continue

        model_count = len(cfg.get("models", []))
        display = f"{name} ({model_count} models)"
        provider_choices.append(display)
        provider_name_map[display] = name

    if not provider_choices:
        raise RuntimeError(
            "No providers are currently available. "
            "Please ensure at least one provider CLI is installed."
        )

    selected_provider_displays = select_multi(
        provider_choices,
        "Select provider(s) to use:"
    )

    if not selected_provider_displays:
        return []

    # Extract provider names from selections
    provider_names = [provider_name_map[p] for p in selected_provider_displays]

    # Step 2: Select models from each provider
    selected_models = []
    for provider_name in provider_names:
        models = providers[provider_name].get("models", [])
        if not models:
            continue

        selected = select_multi(
            models,
            f"Select {provider_name} models:"
        )

        # Add provider prefix to create full model specs
        selected_models.extend(f"{provider_name}:{m}" for m in selected)

    return selected_models


def resolve_models(
    cli_models: Optional[List[str]] = None,
    interactive: bool = False,
    quick: bool = False,
    mode: Optional[str] = None,
    anchor: Optional[str] = None
) -> List[str]:
    """
    Resolve which models to use based on priority order.

    Priority:
    1. CLI --models flag        -> Use specified models
    2. --interactive flag       -> Force two-step interactive selection (ignores defaults)
    3. --quick flag             -> Use quick_models from YAML
    4. YAML defaults.models     -> Use configured defaults (no prompting)
    5. Interactive selection    -> Two-step selection (fallback if no defaults)

    Args:
        cli_models: Models specified via --models CLI flag
        interactive: Whether --interactive flag was passed
        quick: Whether --quick flag was passed
        mode: Optional mode name for mode-specific defaults
        anchor: Optional discovery anchor (the plan-file path or target-repo
            directory). When supplied, per-project config discovery resolves the
            git root from it so the override file used matches the orchestrator's
            plan-derived root; when omitted, discovery falls back to CWD.

    Returns:
        List of model specs in provider:model format
    """
    # Import here to avoid circular imports
    from .provider_registry import (
        get_default_models,
        get_provider,
        get_quick_models,
        has_default_models,
        has_quick_models,
        parse_model_spec,
    )

    # 1. CLI --models flag takes highest priority
    if cli_models and len(cli_models) > 0:
        return cli_models

    # 2. --interactive flag forces interactive selection
    if interactive:
        return select_models_two_step(anchor=anchor)

    # 3. --quick flag uses quick_models from YAML
    if quick:
        if not has_quick_models(mode, anchor=anchor):
            raise RuntimeError(
                "No quick_models configured in providers.yaml. "
                "Add a 'quick_models' list under 'defaults' in providers.yaml."
            )
        quick_models = get_quick_models(mode, anchor=anchor)
        available_models = []
        unavailable_providers = set()

        for model_spec in quick_models:
            provider_name, _ = parse_model_spec(model_spec, anchor=anchor)
            provider = get_provider(provider_name)

            if provider is not None and provider.is_available():
                available_models.append(model_spec)
            else:
                unavailable_providers.add(provider_name)

        for provider_name in unavailable_providers:
            print(
                f"Warning: Provider '{provider_name}' is not available, "
                "skipping its models from quick_models.",
                file=sys.stderr,
            )

        if available_models:
            return available_models
        elif quick_models:
            raise RuntimeError(
                f"All quick_models providers are unavailable: {list(unavailable_providers)}. "
                "Please install at least one provider CLI or use --models to specify models."
            )

    # 4. Check for YAML defaults (with availability filtering)
    if has_default_models(mode, anchor=anchor):
        default_models = get_default_models(mode, anchor=anchor)
        available_models = []
        unavailable_providers = set()

        for model_spec in default_models:
            provider_name, _ = parse_model_spec(model_spec, anchor=anchor)
            provider = get_provider(provider_name)

            if provider is not None and provider.is_available():
                available_models.append(model_spec)
            else:
                unavailable_providers.add(provider_name)

        # Warn about unavailable providers
        for provider_name in unavailable_providers:
            print(
                f"Warning: Provider '{provider_name}' is not available, "
                "skipping its models from defaults.",
                file=sys.stderr,
            )

        # Only fail if ALL default providers are unavailable
        if available_models:
            return available_models
        elif default_models:
            # All default providers are unavailable
            raise RuntimeError(
                f"All default providers are unavailable: {list(unavailable_providers)}. "
                "Please install at least one provider CLI or use --interactive to select models."
            )

    # 5. Fall back to interactive selection
    return select_models_two_step(anchor=anchor)


# ---------------------------------------------------------------------------
# Init-flow picker primitives (sentinel-aware multi-select, free text, single
# select). Composed on top of the gum → fzf → numbered cascade above.
# ---------------------------------------------------------------------------


@dataclass
class ActionSelection:
    """Outcome of a sentinel-aware multi-select pass (see select_with_actions).

    Exactly one of the three states holds:
      * ``cancelled`` — nothing was checked / Esc / empty (skip this provider).
      * ``action`` set — the user triggered a mode-switch sentinel (ordinary picks
        in the same pass are intentionally discarded; see the §4 exclusivity rule).
      * neither — ``selected`` holds the ordinary rows the user picked.
    """

    action: Optional[str] = None          # None | ACTION_SHOW_ALL | ACTION_ENTER_MANUAL
    selected: List[str] = field(default_factory=list)
    cancelled: bool = False


def select_with_actions(
    rows: List[str],
    prompt: str,
    *,
    show_all_label: Optional[str] = None,
    manual_label: Optional[str] = None,
) -> ActionSelection:
    """Multi-select ``rows`` plus optional action *sentinels*, resolving exclusivity.

    Appends the provided sentinel labels (``show_all_label`` /  ``manual_label``)
    to the option list, runs the normal ``select_multi`` cascade, then classifies
    the result per the §4 contract:

    - **Sentinels are exclusive.** If any sentinel is checked, ordinary picks in the
      same pass are ignored and the sentinel's action fires.
    - **"Show all…" wins** over "Enter manually…" when both are checked (a fixed
      precedence so all backends behave identically).
    - **Only ordinary rows** → returned as ``selected``.
    - **Nothing checked** (Esc / empty / non-zero) → ``cancelled=True``.
    """
    options = list(rows)
    sentinel_map = {}
    if show_all_label:
        options.append(show_all_label)
        sentinel_map[show_all_label] = ACTION_SHOW_ALL
    if manual_label:
        options.append(manual_label)
        sentinel_map[manual_label] = ACTION_ENTER_MANUAL

    picked = select_multi(options, prompt)
    if not picked:
        return ActionSelection(cancelled=True)

    actions = [sentinel_map[p] for p in picked if p in sentinel_map]
    if ACTION_SHOW_ALL in actions:                       # fixed precedence
        return ActionSelection(action=ACTION_SHOW_ALL)
    if ACTION_ENTER_MANUAL in actions:
        return ActionSelection(action=ACTION_ENTER_MANUAL)

    ordinary = [p for p in picked if p not in sentinel_map]
    if not ordinary:
        return ActionSelection(cancelled=True)
    return ActionSelection(selected=ordinary)


def prompt_text(prompt: str) -> Optional[str]:
    """Prompt for a single free-text line; return it stripped, or None.

    None is returned on EOF / interrupt / a blank (whitespace-only) entry, so the
    caller can reject empties rather than accept a whitespace spec.
    """
    if not is_tty():
        raise RuntimeError("No TTY available for free-text entry.")
    try:
        value = input(f"{prompt} ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return value or None


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask a yes/no question; return the boolean. Empty/EOF → ``default``."""
    if not is_tty():
        return default
    suffix = " [y/N]" if not default else " [Y/n]"
    try:
        answer = input(f"{prompt}{suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def _try_gum_choose_one(options: List[str], prompt: str, default: Optional[str]):
    """Try gum choose in single-select mode (no --no-limit).

    Mirrors the multi-select ``UNAVAILABLE`` convention so a deliberate cancel is
    distinguished from a missing backend:

    Returns:
        - ``UNAVAILABLE`` if gum is not installed or the subprocess fails to run
          (the caller should fall through to the next backend).
        - ``None`` if gum ran but the user cancelled / hit Esc / made no choice
          (a real "cancelled" result; the caller must NOT fall through — it stops
          the cascade and falls back to the default).
        - The selected row (a ``str``) otherwise.
    """
    if not shutil.which("gum"):
        return UNAVAILABLE
    cmd = ["gum", "choose", "--header", prompt]
    if default in options:
        cmd += ["--selected", default]
    try:
        result = subprocess.run(
            cmd + options, capture_output=True, text=True, timeout=120, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return UNAVAILABLE
    # gum ran: a non-zero exit / empty stdout is a deliberate cancel (Esc) → None,
    # NOT an "unavailable" fall-through.
    if result.returncode == 0 and result.stdout.strip():
        # Single-select returns one line; guard against any stray extra.
        return result.stdout.strip().split("\n")[0]
    return None


def _try_fzf_one(options: List[str], prompt: str):
    """Try fzf in single-select mode (no -m).

    Mirrors the multi-select ``UNAVAILABLE`` convention:

    Returns:
        - ``UNAVAILABLE`` if fzf is not installed or the subprocess fails to run
          (the caller should fall through to the next backend).
        - ``None`` if fzf ran but the user cancelled / hit Esc (a real "cancelled"
          result; the caller must NOT fall through — it stops the cascade and
          falls back to the default).
        - The selected row (a ``str``) otherwise.
    """
    if not shutil.which("fzf"):
        return UNAVAILABLE
    try:
        result = subprocess.run(
            ["fzf", "--header", prompt, "--bind", "enter:accept"],
            input="\n".join(options),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return UNAVAILABLE
    # fzf ran: a non-zero exit (Esc → 130) / empty stdout is a deliberate cancel
    # → None, NOT an "unavailable" fall-through.
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().split("\n")[0]
    return None


def _numbered_prompt_one(options: List[str], prompt: str, default: Optional[str]) -> Optional[str]:
    """Fallback single-choice numbered prompt; empty input → ``default``."""
    print(f"\n{prompt}")
    print("-" * 40)
    for i, opt in enumerate(options, 1):
        marker = "  (default)" if opt == default else ""
        print(f"  {i}. {opt}{marker}")
    print("-" * 40)
    hint = f" [default: {default}]" if default in options else ""
    try:
        user_input = input(f"Enter a single number{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    if not user_input:
        return default
    try:
        idx = int(user_input)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    except ValueError:
        pass
    return default


def select_one(
    rows: List[str],
    prompt: str = "Select one:",
    default: Optional[str] = None,
) -> Optional[str]:
    """Single-select with the gum → fzf → numbered cascade (mirrors select_multi).

    Used for ``default_provider``. Returns the chosen row; on cancel / empty falls
    back to ``default`` (which the caller guarantees is one of ``rows``).

    Cascade rule (mirrors select_multi): fall through to the next backend ONLY when
    a backend is ``UNAVAILABLE`` (not installed / failed to launch). When a backend
    actually ran and the user cancelled (helper returns ``None``), STOP cascading
    and return ``default`` — re-prompting with a different backend would be
    surprising for a deliberate cancel.
    """
    if not is_tty():
        raise RuntimeError("No TTY available for interactive selection.")
    if not rows:
        return default

    result = _try_gum_choose_one(rows, prompt, default)
    if result is not UNAVAILABLE:
        return result if result is not None else default
    result = _try_fzf_one(rows, prompt)
    if result is not UNAVAILABLE:
        return result if result is not None else default
    return _numbered_prompt_one(rows, prompt, default)
