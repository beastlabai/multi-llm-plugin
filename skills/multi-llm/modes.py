#!/usr/bin/env python3
"""The single owner of the multi-llm mode table.

Every surface that enumerates modes is *rendered* from ``MODES`` rather than
hand-maintained: ``multi_llm.py --help``, ``SKILL.md``'s ``argument-hint`` line,
``check_workflow_prerequisites.py``'s ``--mode``/``--phase`` vocabulary and (with
a later change) each host shim's flag list. Adding a mode is a row here, never an
edit in twelve places, and ``tests/test_instruction_portability.py`` fails if the
dispatcher and ``argument-hint`` disagree.

Three vocabularies meet in this table and are deliberately kept distinct:

* **mode word** — what ``multi_llm.py`` takes as ``argv[1]`` (``review-code``).
* **user flag** — what a user types at the skill (``--review-code``). The mapping
  is mechanical: strip the leading ``--``, verify membership. No shim invents it.
* **phase** — the ``state.json`` / phase-dir name (``code-review``). It differs
  from the mode word for exactly one mode, which is why ``code-review`` is also a
  dispatcher alias.

Nothing here imports an orchestrator: ``target_module`` is a dotted path the
dispatcher resolves lazily, so ``import modes`` never drags in the ~9k lines of
orchestrator code just to list mode names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Positional kinds a mode can declare, in the order they must appear.
POS_PLAN = "<plan>"
POS_QUESTION = "<question>"
POS_HOST = "<host>"


@dataclass(frozen=True)
class Mode:
    """One row of the dispatch table.

    Attributes:
        name: The mode word (``argv[1]``).
        positionals: Positional kinds, in order. Normalized by the dispatcher
            into the flags the target's own argparse parser already accepts.
        summary: One-line ``--help`` text.
        target_module: Dotted module path, imported lazily. ``None`` for modes
            whose implementation has not landed yet.
        target_attr: Callable attribute on that module, called as
            ``main(argv) -> int``.
        phase: ``state.json`` / phase-dir name, or ``None`` for modes that own no
            phase (``full``, ``status``, ``init``, the install modes).
        instruction_file: Core instruction file under ``instructions/``, served
            token-free by ``multi_llm.py util instructions <mode>``.
        user_facing: True for the modes that appear in ``SKILL.md``'s
            ``argument-hint`` as ``--<name>``.
        is_workflow_phase: True for the eight phases
            ``check_workflow_prerequisites.py`` gates on.
        supports_detach: True for the modes ``--detach`` covers: the fan-out modes
            plus, since the write executor landed, every executor-backed mode
            (``implement``, the three apply phases, ``generate-tasks``, ``full``) —
            a write phase that schedules N batches through real agent CLIs outlives
            a host's shell cap exactly as a fan-out does. A mode with ``False``
            **rejects** ``--detach`` with exit 3 rather than accepting a flag that
            silently does nothing. ``generate-tasks`` and ``full`` carry the column
            because their detach support is real *plumbing*, while the modes
            themselves are still ``unavailable`` — the ``unavailable`` exit-3 fires
            first, so neither pretends to work.
        injected_flags: Flags the dispatcher prepends before the user's, because
            the mode is a thin view over a more general script.
        unavailable: When set, the mode is dispatchable but exits 3 with this
            reason instead of running. Keeps the boundary a shim sees stable
            while the implementation is still a markdown sequence.
    """

    name: str
    positionals: Tuple[str, ...]
    summary: str
    target_module: Optional[str] = None
    target_attr: str = "main"
    phase: Optional[str] = None
    instruction_file: Optional[str] = None
    user_facing: bool = True
    is_workflow_phase: bool = False
    supports_detach: bool = False
    injected_flags: Tuple[str, ...] = ()
    unavailable: Optional[str] = None


# Reason strings for the modes that are dispatchable but not yet implemented.
# Kept as constants so the dispatcher's exit-3 text and the tests agree.
#
# Why they are in the table at all: **every mode word is invocable through
# multi_llm.py on every host from day one**, even where the implementation behind
# it is still a markdown sequence. "Instruction-only" describes an
# implementation, never the dispatcher boundary a shim sees — otherwise every
# non-Claude shim has to special-case the remainder and re-derive a
# Claude-authored instruction sequence, which is the coupling the dispatcher
# exists to delete. The dispatcher makes the "not yet here" call, so no shim
# carries a host-specific fallback.
#
# TODO(generate-tasks): replace `unavailable` with
#   target_module="generate_tasks_orchestrator" (gate -> decomposition ->
#   update_plan_tasks) when that script lands.
# TODO(full): replace with target_module="full_workflow_orchestrator" (a
#   resumable state machine over the other modes on the same state.json).
# TODO(install-host/uninstall-host): owned by the install-lifecycle design (D7 /
#   D7b of plans/host-harness-portability.md) — shim rendering, --check,
#   idempotency, rollback. Until then the dispatcher exits 3 rather than
#   pretending the mode does something.
NOT_YET_ON_THIS_HOST = (
    "not yet available on this host - run the instruction sequence instead "
    "(multi_llm.py util instructions {mode})"
)
NOT_YET_AVAILABLE = "not yet available"


# Order is the workflow order, and it is also the order rendered into
# SKILL.md's `argument-hint` line — keep them in sync by rendering, never by
# copying (see render_argument_hint).
MODES: Tuple[Mode, ...] = (
    Mode(
        name="review-plan",
        positionals=(POS_PLAN,),
        summary="Review a plan with every configured model, then group + validate.",
        target_module="review_plan_orchestrator",
        phase="review-plan",
        instruction_file="review-plan.md",
        is_workflow_phase=True,
        supports_detach=True,
    ),
    Mode(
        name="apply-suggestions",
        positionals=(POS_PLAN,),
        summary="Prepare the validated plan-review suggestions for application.",
        target_module="apply_suggestions_orchestrator",
        phase="apply-suggestions",
        instruction_file="apply-suggestions.md",
        is_workflow_phase=True,
        supports_detach=True,
    ),
    Mode(
        name="generate-tasks",
        positionals=(POS_PLAN,),
        summary="Decompose the plan into an implementable task list.",
        phase="generate-tasks",
        instruction_file="generate-tasks.md",
        is_workflow_phase=True,
        supports_detach=True,
        unavailable=NOT_YET_ON_THIS_HOST,
    ),
    Mode(
        name="review-tasks",
        positionals=(POS_PLAN,),
        summary="Review the generated task list with every configured model.",
        target_module="review_tasks_orchestrator",
        phase="review-tasks",
        instruction_file="review-tasks.md",
        is_workflow_phase=True,
        supports_detach=True,
    ),
    Mode(
        name="apply-task-suggestions",
        positionals=(POS_PLAN,),
        summary="Prepare the validated task-review suggestions for application.",
        target_module="apply_task_suggestions_orchestrator",
        phase="apply-task-suggestions",
        instruction_file="apply-task-suggestions.md",
        is_workflow_phase=True,
        supports_detach=True,
    ),
    Mode(
        name="implement",
        positionals=(POS_PLAN,),
        summary="Batch the tasks for implementation and track progress.",
        target_module="implement_orchestrator",
        phase="implement",
        instruction_file="implement.md",
        is_workflow_phase=True,
        supports_detach=True,
    ),
    Mode(
        name="review-code",
        positionals=(POS_PLAN,),
        summary="Review the working diff against the plan with every model.",
        target_module="code_review_orchestrator",
        phase="code-review",
        instruction_file="review-code.md",
        is_workflow_phase=True,
        supports_detach=True,
    ),
    Mode(
        name="apply-code-fixes",
        positionals=(POS_PLAN,),
        summary="Prepare the validated code-review fixes for application.",
        target_module="apply_code_fixes_orchestrator",
        phase="apply-code-fixes",
        instruction_file="apply-code-fixes.md",
        is_workflow_phase=True,
        supports_detach=True,
    ),
    Mode(
        name="full",
        positionals=(POS_PLAN,),
        summary="Run the whole plan-to-code workflow as one resumable sequence.",
        instruction_file="full-workflow.md",
        supports_detach=True,
        unavailable=NOT_YET_ON_THIS_HOST,
    ),
    Mode(
        name="ask",
        positionals=(POS_PLAN, POS_QUESTION),
        summary="Ask every configured model one free-text question about a plan.",
        target_module="ask_orchestrator",
        phase="ask",
        instruction_file="ask.md",
        supports_detach=True,
    ),
    Mode(
        name="status",
        positionals=(POS_PLAN,),
        summary="Report workflow phases (state.json) and detached runs "
                "({phase}/.status.json).",
        # Two views over two files behind one command word. The predecessor was
        # an instruction-only mode word whose markdown wrapper shelled out to
        # `check_workflow_prerequisites.py --status`; that function survives as
        # the workflow view's implementation (still directly callable, still
        # byte-identical under `--view workflow --json`), but the mode itself is
        # Python now, because a detached run with no command to poll it is
        # unusable.
        target_module="utils.run_status",
        instruction_file=None,
    ),
    Mode(
        name="init",
        positionals=(),
        summary="Scaffold a per-project .multi-llm/providers.yaml override.",
        target_module="init_config",
        instruction_file="init-config.md",
    ),
    # Not user-facing: neither is part of the skill's own argument surface, so
    # neither gets a `--flag` in `argument-hint`.
    Mode(
        name="install-host",
        positionals=(POS_HOST,),
        summary="Install this skill's entry shim into another coding harness.",
        user_facing=False,
        unavailable=NOT_YET_AVAILABLE,
    ),
    Mode(
        name="uninstall-host",
        positionals=(POS_HOST,),
        summary="Remove a previously installed host shim.",
        user_facing=False,
        unavailable=NOT_YET_AVAILABLE,
    ),
)


# A closed alias table, shown in `--help`. No prefix or fuzzy matching: an
# unrecognized word is an error, not a guess.
ALIASES: Dict[str, str] = {
    "review": "review-plan",
    "code-review": "review-code",
    "tasks": "generate-tasks",
    # Stopping a detached run has to be discoverable, so the verb is an alias
    # like any other — the only one that expands to a mode word *plus a flag*
    # (see ALIAS_FLAGS). Aliases are a human convenience: generated shims always
    # emit the `status … --abort` spelling so their text stays on one surface.
    "abort": "status",
    "cancel": "status",
    "stop": "status",
}


# The flags an alias contributes on top of its target mode, prepended before the
# user's own. Carried here so `--help` renders the expansion and nothing else
# re-derives it.
ALIAS_FLAGS: Dict[str, Tuple[str, ...]] = {
    "abort": ("--abort",),
    "cancel": ("--abort",),
    "stop": ("--abort",),
}


def alias_flags(word: str) -> Tuple[str, ...]:
    """Flags the dispatcher must inject for *word*, if it is a flag-bearing alias."""
    return ALIAS_FLAGS.get(word, ())


def alias_expansion(word: str) -> str:
    """How `--help` spells an alias: ``abort -> status --abort``."""
    target = ALIASES[word]
    extra = " ".join(ALIAS_FLAGS.get(word, ()))
    return f"{target} {extra}".rstrip()


# The `util` namespace is a mode word for routing purposes but is deliberately
# *not* a mode: it is excluded from `--help`'s mode list and from
# `argument-hint`, and is never rendered into a MULTI_LLM_RESULT.next value.
UTIL_NAMESPACE = "util"


# Legacy direct-invocation script name -> mode word. Used when a printed
# follow-up command has to be built from an orchestrator script name (the
# orchestrators pass their own filename around as `orchestrator_script`).
SCRIPT_TO_MODE: Dict[str, str] = {
    m.target_module + ".py": m.name
    for m in MODES
    if m.target_module and m.target_module.endswith("_orchestrator")
}


_BY_NAME: Dict[str, Mode] = {m.name: m for m in MODES}


def mode_names() -> Tuple[str, ...]:
    """Every mode word, in table order."""
    return tuple(m.name for m in MODES)


def user_facing_modes() -> Tuple[Mode, ...]:
    """The modes that have a ``--<name>`` user flag."""
    return tuple(m for m in MODES if m.user_facing)


def resolve(word: str) -> Optional[Mode]:
    """Resolve a mode word or alias to its :class:`Mode`, else ``None``."""
    return _BY_NAME.get(ALIASES.get(word, word))


def phase_for(word: str) -> Optional[str]:
    """The ``state.json`` phase name for a mode word or alias."""
    mode = resolve(word)
    return mode.phase if mode else None


def mode_for_script(script: str) -> Optional[str]:
    """The mode word that replaces a direct ``*_orchestrator.py`` invocation."""
    return SCRIPT_TO_MODE.get(script.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])


def instruction_path(word: str) -> Optional[str]:
    """Repo-relative path of a mode's core instruction file, if it has one."""
    mode = resolve(word)
    if mode is None or mode.instruction_file is None:
        return None
    return "instructions/" + mode.instruction_file


#: The phase vocabulary ``check_workflow_prerequisites.py`` gates on, in
#: workflow order. Imported there so one table owns it.
PREREQUISITE_PHASES: Tuple[str, ...] = tuple(
    m.phase for m in MODES if m.is_workflow_phase and m.phase
)


#: The modes ``--detach`` covers, in table order. One owner: the dispatcher's
#: rejection, ``utils/detach.py``'s guard and every error message read this.
DETACH_MODES: Tuple[str, ...] = tuple(m.name for m in MODES if m.supports_detach)


def phase_names() -> Tuple[str, ...]:
    """Every phase name any mode owns, in table order (includes ``ask``)."""
    seen: Dict[str, None] = {}
    for mode in MODES:
        if mode.phase:
            seen.setdefault(mode.phase, None)
    return tuple(seen)


def phase_from_selector(word: str) -> Optional[str]:
    """Resolve a ``--phase``/``--mode`` value to a phase name, or ``None``.

    Accepts either vocabulary, because the two differ for exactly one mode
    (``review-code`` → phase ``code-review``) and a caller should not have to know
    which one a given flag wanted. The resolution lives here so ``status``'s
    ``--phase`` selector and ``check_workflow_prerequisites.py``'s ``--mode``
    choices cannot drift apart.
    """
    if word in phase_names():
        return word
    return phase_for(word)


# The non-mode part of `argument-hint` — everything after the mode alternation.
# A literal because it describes flags, not modes; the mode list itself is
# rendered so a new row cannot be forgotten here.
ARGUMENT_HINT_TRAILER = (
    ' <plan_path> ["<question>" (required for --ask)] '
    "[--models provider:model ...] [--interactive] [--quick] [--yes] [--force] "
    "[--dir PATH|--gitignore|--template-only (--init only)]"
)


def render_argument_hint() -> str:
    """Render SKILL.md's ``argument-hint`` value from the table."""
    flags = "|".join(f"--{m.name}" for m in user_facing_modes())
    return f"[{flags}]{ARGUMENT_HINT_TRAILER}"
