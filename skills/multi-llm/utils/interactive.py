#!/usr/bin/env python3
"""Interactive utilities for terminal-based user input."""

import shutil
import subprocess
import sys
from typing import List, Optional

from .provider_registry import load_config


def is_tty() -> bool:
    """Check if stdin/stdout are connected to a TTY."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _try_gum_choose(options: List[str], prompt: str) -> Optional[List[str]]:
    """Try gum choose for multi-select. Returns None if gum not available."""
    if not shutil.which("gum"):
        return None

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
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _try_fzf_multi(options: List[str], prompt: str) -> Optional[List[str]]:
    """Try fzf with multi-select. Returns None if fzf not available."""
    if not shutil.which("fzf"):
        return None

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
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


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

    # Try gum first (best UX)
    result = _try_gum_choose(available_models, prompt)
    if result is not None:
        return result

    # Try fzf (widely available)
    result = _try_fzf_multi(available_models, prompt)
    if result is not None:
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

    # Try gum first (best UX)
    result = _try_gum_choose(options, prompt)
    if result is not None:
        return result

    # Try fzf (widely available)
    result = _try_fzf_multi(options, prompt)
    if result is not None:
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
