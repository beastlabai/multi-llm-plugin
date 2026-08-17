#!/usr/bin/env python3
"""Interactive utilities for terminal-based user input.

**Interactivity is requested here, never detected** (D6b). This module used to
gate its pickers on a TTY probe; it no longer probes anything, because
a run whose behavior depends on whether stdout is a pipe produces different
phases from the same argv — and the pipe case is exactly the headless invocation
that has nobody to answer a picker. A picker is now reached only when the caller
asked for one (``--interactive``), and :func:`resolve_models` refuses with exit 3
rather than guessing when models cannot be resolved from flags and config.
"""

import shutil
import subprocess
import sys
from typing import List, Optional

from .provider_registry import load_config
from .unattended import NeedsHuman, row


class _Unavailable:
    """Sentinel: a picker backend (gum/fzf) could not be used at all.

    This distinguishes "the backend tool is not installed / failed to launch"
    (the cascade should fall through to the next backend) from "the backend ran
    and the user made an empty selection / cancelled / hit Esc" (which is a real,
    intentional result and must NOT fall through — it stops the cascade and yields
    an empty selection). Backend helpers return ``UNAVAILABLE`` only in the former
    case; otherwise they return a concrete (possibly empty) selection.

    Used by the runtime multi-select cascade (``select_multi`` /
    ``select_models_interactive``) to distinguish a missing backend from a
    deliberate empty selection.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNAVAILABLE"

    def __bool__(self) -> bool:
        return False


# Singleton sentinel returned by picker backends that are unavailable.
UNAVAILABLE = _Unavailable()


class NoDecisionChannel(RuntimeError):
    """A picker was reached with nothing on the other end of stdin.

    Raised instead of returning an empty selection so a run that would have
    blocked fails *visibly*. A ``RuntimeError`` subclass on purpose: every
    existing caller already maps ``RuntimeError`` from model resolution to a
    printed error, so this changes the message and not the control flow.
    """


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
        # UTF-8 + errors="replace": lossy for non-UTF-8 bytes but guarantees a
        # total decode of tool output (vs. locale-codec UnicodeDecodeError).
        result = subprocess.run(
            ["gum", "choose", "--no-limit", "--header", prompt] + options,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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
            encoding="utf-8",
            errors="replace",
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
    except EOFError:
        # stdin is closed (or the user pressed Ctrl-D): there is no channel to
        # answer on, which is a visible failure rather than "selected nothing".
        raise NoDecisionChannel(
            "No interactive channel for model selection: stdin is closed. "
            "Pass --models \"provider:model,...\" (or --quick), or configure "
            "defaults.models in providers.yaml."
        )
    except KeyboardInterrupt:
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
        NoDecisionChannel: If the numbered fallback is reached with stdin closed.
    """
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
        NoDecisionChannel: If the numbered fallback is reached with stdin closed.
    """
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
    anchor: Optional[str] = None,
    unattended: bool = False,
) -> List[str]:
    """
    Resolve which models to use based on priority order.

    Priority:
    1. CLI --models flag        -> Use specified models
    2. --interactive flag       -> Force two-step interactive selection (ignores defaults)
    3. --quick flag             -> Use quick_models from YAML
    4. YAML defaults.models     -> Use configured defaults (no prompting)
    5. Interactive selection    -> Two-step selection (fallback if no defaults)

    D6 row 2: under *unattended* the ladder stops at step 4. ``defaults.models``
    from the resolved config layering is the auto-selected answer; an empty or
    unresolvable set is **exit 3** (:class:`NeedsHuman`) naming ``--models``, and
    step 5's picker is never reached — an unattended run that renders a
    multi-select is a hang, not a prompt. ``--interactive`` together with the
    unattended flag is a contradiction and is likewise exit 3 (the dispatcher
    already rejects it at parse time; this covers direct invocation).

    Args:
        cli_models: Models specified via --models CLI flag
        interactive: Whether --interactive flag was passed
        quick: Whether --quick flag was passed
        mode: Optional mode name for mode-specific defaults
        anchor: Optional discovery anchor (the plan-file path or target-repo
            directory). When supplied, per-project config discovery resolves the
            git root from it so the override file used matches the orchestrator's
            plan-derived root; when omitted, discovery falls back to CWD.
        unattended: Whether --yes / --non-interactive was passed.

    Returns:
        List of model specs in **canonical** ``provider:model`` format (§9.3.d).
        ``--models`` and ``defaults.models`` both legally carry bare,
        prefix-less entries, which every downstream consumer would otherwise
        re-resolve through its *own* ``parse_model_spec`` — only some of them
        anchored. That is the mechanism behind the measured wrong-provider
        spawn: one ``sonnet`` became the project's ``gemini:sonnet`` in
        validation and ``cursor-agent:sonnet`` at the spawn, in one run.
        Resolving once here, against the anchor this function already accepts,
        closes the spec set under ``parse_model_spec`` (idempotent on a
        prefixed spec — it never reads config), so a site the sweep missed can
        still get a *timeout* wrong but can no longer change the **provider**.

        Two consequences, by design: canonical specs feed ``sanitize_model_name``
        (so a run started before this change and resumed after it re-runs its
        bare-spec models once, the resume predicate looking for the old file
        name), and the "available models" / "invalid model" messages now echo
        the resolved form.

    Raises:
        NeedsHuman: When nothing resolves and no human may be asked (exit 3).
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

    def _canonical(specs: List[str]) -> List[str]:
        """Resolve every entry once, against this call's anchor (§9.3.d)."""
        resolved = []
        for spec in specs:
            provider_name, model_name = parse_model_spec(spec, anchor=anchor)
            resolved.append(f"{provider_name}:{model_name}")
        return resolved

    # 0. Row 15, second layer: `--models ""` is never "use the defaults".
    if cli_models is not None:
        for spec in cli_models:
            if not str(spec).strip():
                raise NeedsHuman(
                    "ERROR: --models was passed an empty value. Pass at least "
                    "one provider:model spec, or omit --models to use "
                    "defaults.models from providers.yaml.",
                    site=row(2).site_id,
                )

    # 1. CLI --models flag takes highest priority
    if cli_models and len(cli_models) > 0:
        return _canonical(cli_models)

    # 2. --interactive flag forces interactive selection
    if interactive:
        if unattended:
            raise NeedsHuman(
                "ERROR: --interactive and --yes/--non-interactive are "
                "contradictory: one forces a rendered model picker, the other "
                "forbids prompting. Pass exactly one.",
                site=row(2).site_id,
            )
        # Already canonical by construction (the picker prefixes each choice),
        # but routed through the same leg so there is one definition of the
        # shape resolve_models returns.
        return _canonical(select_models_two_step(anchor=anchor))

    # 3. --quick flag uses quick_models from YAML
    if quick:
        if not has_quick_models(mode, anchor=anchor):
            message = (
                "No quick_models configured in providers.yaml. "
                "Add a 'quick_models' list under 'defaults' in providers.yaml."
            )
            if unattended:
                raise NeedsHuman(f"ERROR: {message}", site=row(2).site_id)
            raise RuntimeError(message)
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
            return _canonical(available_models)
        elif quick_models:
            message = (
                f"All quick_models providers are unavailable: {list(unavailable_providers)}. "
                "Please install at least one provider CLI or use --models to specify models."
            )
            if unattended:
                raise NeedsHuman(f"ERROR: {message}", site=row(2).site_id)
            raise RuntimeError(message)

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
            return _canonical(available_models)
        elif default_models:
            # All default providers are unavailable
            if unattended:
                raise NeedsHuman(
                    "ERROR: every provider in defaults.models is unavailable: "
                    f"{sorted(unavailable_providers)}.\n"
                    "       Install at least one provider CLI, or re-run naming "
                    "the models explicitly:\n"
                    "         --models \"provider:model,...\"",
                    site=row(2).site_id,
                )
            raise RuntimeError(
                f"All default providers are unavailable: {list(unavailable_providers)}. "
                "Please install at least one provider CLI or use --interactive to select models."
            )

    # 5. Nothing resolved. A picker is *requested* (step 2), never inferred, so
    #    this is a fail-closed gate rather than a fallback prompt (D6 rows 2/15).
    raise NeedsHuman(
        "ERROR: no models could be resolved and none may be guessed.\n"
        "       Answer this gate with one of:\n"
        "         --models \"provider:model,...\"   name the models\n"
        "         --quick                          use quick_models from "
        "providers.yaml\n"
        "         defaults.models in providers.yaml (multi_llm.py init "
        "scaffolds it)"
        + ("" if unattended else "\n         --interactive                    "
                                 "pick them now"),
        site=row(2).site_id,
    )
