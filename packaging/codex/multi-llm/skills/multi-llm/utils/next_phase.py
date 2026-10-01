"""The next-phase command a finished phase ends with (plans/next-phase-command.md §2.1).

One function builds the string, and every emitter (``multi_llm.main``,
``run_status``'s terminal poll, ``util next-command``) only passes it through:
the ``[NEXT_COMMAND]`` block (:func:`render_next_phase_lines`) and
``MULTI_LLM_RESULT.user_next_step`` (:meth:`NextPhase.to_dict`) are two
renderings of the same :class:`NextPhase`.

The contract, briefly:

* :func:`next_phase_commands` is pure apart from reading ``state.json``, and it
  reads that only through ``check_workflow_prerequisites.get_workflow_status``
  (or the payload a caller already computed with it). It adds no status logic
  of its own: which phases are completed, skipped, pending or ``in_progress``
  is that payload's answer, never a second table's.
* It never reads the environment. The route is resolved by the caller, once,
  with :func:`resolve_route` — the only code here that reads a route signal —
  and handed in.
* It holds no prose. The spelling is the host's ``next_command`` template
  (``hosts/capabilities.yaml``, through ``host_capabilities``), the mode word
  comes from ``modes.py``, and an optional entry's ``why`` is its phase row's
  ``Mode.optional_why`` unless the workflow payload recorded a notice on that
  phase's entry.
* ``plan_arg`` is used verbatim — never resolved, rebased or relativized — and
  is quoted for the model that reads the command, never for a shell
  (:func:`quote_plan_arg`).
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

#: The four line kinds of the block (§2.2(2)). The text after a command
#: marker and one space is the builder's ``command`` byte for byte; an
#: optional entry's ``why`` sits on its own line directly above its command.
NEXT_COMMAND_MARKER = "[NEXT_COMMAND]"
NEXT_COMMAND_OPTIONAL_WHY_MARKER = "[NEXT_COMMAND_OPTIONAL_WHY]"
NEXT_COMMAND_OPTIONAL_MARKER = "[NEXT_COMMAND_OPTIONAL]"
WORKFLOW_COMPLETE_MARKER = "[WORKFLOW_COMPLETE]"

#: The one slot every ``next_command`` template ends with (schema pattern
#: ``^[^{}]+\{args\}$``). Filled with ``str.replace``, never ``str.format``.
ARGS_SLOT = "{args}"

#: Appended when the phase a command names is recorded ``in_progress``: a bare
#: re-invocation of a write phase starts it over (§2.1, "``--resume`` is built
#: from state"). Never copied from the finished invocation's argv.
RESUME_FLAG = "--resume"

#: The route key of a plain-skill install, told apart from the plugin by the
#: dispatcher's own skill root rather than by any environment variable.
SKILL_ROUTE = "skill"

#: Whitespace-free path components in ``install_paths`` that stand for a
#: variable prefix (``$HOME``, ``%USERPROFILE%``, ``~``); the tail after the
#: last one is what a skill root is compared against.
_PLACEHOLDER_PREFIXES = ("$", "%", "~")


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionalStep:
    """One ``optional_before`` entry: an optional phase that may run first."""

    phase: str
    command: str
    why: str

    def to_dict(self) -> Dict[str, Any]:
        return {"phase": self.phase, "command": self.command, "why": self.why}


@dataclass(frozen=True)
class NextPhase:
    """The builder's answer: a command to run next, or ``complete``.

    ``optional_before`` lists pending optional phases that precede the primary
    *in workflow order*; every rendering still prints the primary first.
    """

    phase: Optional[str]
    command: Optional[str]
    optional_before: Tuple[OptionalStep, ...] = ()
    complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """The ``user_next_step`` value (§2.1's outcome table).

        Complete is exactly ``{"phase": null, "command": null, "complete":
        true}``; otherwise ``optional_before`` is present only when non-empty.
        """
        if self.complete:
            return {"phase": None, "command": None, "complete": True}
        payload: Dict[str, Any] = {"phase": self.phase, "command": self.command}
        if self.optional_before:
            payload["optional_before"] = [step.to_dict() for step in self.optional_before]
        payload["complete"] = False
        return payload


_COMPLETE = NextPhase(phase=None, command=None, complete=True)


# ---------------------------------------------------------------------------
# Imports from the skill root (tolerating a direct script run)
# ---------------------------------------------------------------------------

def _skill_module(name: str):
    """Import a top-level skill module (``modes``, ``check_workflow_prerequisites``)."""
    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - only when utils/ is sys.path[0]
        sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
        return importlib.import_module(name)


def _host_capabilities():
    # Lazy: host_capabilities imports cli_common, which an emitter living in
    # cli_common may import this module from.
    from . import host_capabilities

    return host_capabilities


# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------

def quote_plan_arg(plan_arg: str) -> str:
    """The plan path as it appears in the command: quoted for a model, not a shell.

    * no whitespace and no quote character → bare;
    * contains ``"`` but not ``'`` → wrapped in single quotes;
    * contains both → wrapped in double quotes as it is (no unescaped spelling
      exists, and no route is claimed to round-trip it);
    * otherwise (whitespace and/or ``'``) → wrapped in double quotes.

    Nothing inside is ever escaped (a Windows path keeps every backslash), and a
    quote character already in ``plan_arg`` is part of the value — it only
    decides which wrapping applies.
    """
    has_double = '"' in plan_arg
    has_single = "'" in plan_arg
    if has_double and not has_single:
        return f"'{plan_arg}'"
    if has_double or has_single or any(ch.isspace() for ch in plan_arg):
        return f'"{plan_arg}"'
    return plan_arg


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

def _path_parts(path: str) -> Tuple[str, ...]:
    """Path components split on either separator, so a Windows path compares too."""
    return tuple(part for part in re.split(r"[\\/]+", path) if part and part != ".")


def _install_tail(entry: str) -> Tuple[str, ...]:
    """An ``install_paths`` entry's components after its last variable prefix."""
    parts = _path_parts(entry)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].startswith(_PLACEHOLDER_PREFIXES):
            return parts[index + 1:]
    return parts


def _is_skill_install(host_id: str, skill_root: Union[str, Path]) -> bool:
    """Whether *skill_root* ends with one of *host_id*'s ``install_paths`` entries.

    Judged by the root's own path, never its ``realpath``: a copy (or a
    symlink) under ``.claude/skills/multi-llm`` is a plain skill whatever it
    points at, and the plugin cache (or a dev checkout linked into it) is not.
    """
    root = _path_parts(str(skill_root))
    install_paths = _host_capabilities().entry(host_id).get("install_paths") or {}
    for entries in install_paths.values():
        for entry in entries or ():
            tail = _install_tail(str(entry))
            if tail and len(root) >= len(tail) and root[-len(tail):] == tail:
                return True
    return False


def resolve_route(
    host_id: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    skill_root: Optional[Union[str, Path]] = None,
    stderr=None,
) -> Optional[str]:
    """The ``next_command`` route key for this process, or ``None`` for ``default``.

    The only reader of a route signal (§2.1, "Knowing the route"); every
    emitter calls it once and passes the result to :func:`next_phase_commands`.

    * A host whose ``next_command`` carries a ``skill`` key (claude-code) is
      told apart by the dispatcher's own skill root (``cli_common._SKILL_DIR``
      unless *skill_root* is given): ``skill`` when that root ends with one of
      the host's ``install_paths`` entries, else ``None`` (the plugin). The
      route variable is ignored there, and saying so costs one stderr line when
      it is set.
    * Any other host reads ``modes.ROUTE_ENV``, which the tool servers set.
      Unset is ``None``, silently. A value is honoured only when it is a member
      of ``modes.TOOL_ROUTES`` **and** a key of the host's ``next_command``
      cell; any other value (empty, a typo, ``skill``, a product name, a tool
      route this host has no cell for) is ``None`` plus one stderr line naming
      the ignored value.

    *environ* defaults to ``os.environ`` and *stderr* to ``sys.stderr``.
    """
    modes = _skill_module("modes")
    env = os.environ if environ is None else environ
    stream = sys.stderr if stderr is None else stderr
    capabilities = _host_capabilities()
    routes = capabilities.next_command_routes(host_id)
    value = env.get(modes.ROUTE_ENV)

    if SKILL_ROUTE in routes:
        if value is not None:
            print(
                f"multi-llm: ignoring {modes.ROUTE_ENV}={value!r}: host "
                f"'{host_id}' takes its next-command route from the skill root, "
                "not the environment",
                file=stream,
            )
        if skill_root is None:
            from . import cli_common

            skill_root = cli_common._SKILL_DIR
        return SKILL_ROUTE if _is_skill_install(host_id, skill_root) else None

    if value is None:
        return None
    if value in modes.TOOL_ROUTES and value in routes:
        return value
    honoured = [route for route in modes.TOOL_ROUTES if route in routes]
    print(
        f"multi-llm: ignoring {modes.ROUTE_ENV}={value!r} for host '{host_id}' "
        f"(honoured here: {', '.join(honoured) or 'none'}); using the "
        f"'{capabilities.DEFAULT_ROUTE_KEY}' next-command spelling",
        file=stream,
    )
    return None


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------

def _workflow_payload(plan_path: Path) -> Dict[str, Any]:
    """``get_workflow_status`` for *plan_path*, as ``run_status`` computes it."""
    from .output_handler import output_dir_path
    from .state_manager import StateManager

    prereq = _skill_module("check_workflow_prerequisites")
    return prereq.get_workflow_status(
        StateManager(plan_path), plan_path, output_dir_path(plan_path)
    )


def next_phase_commands(
    plan_arg: str,
    host_id: str,
    route: Optional[str],
    finished_mode: Optional[str],
    *,
    after_mode: Optional[str] = None,
    plan_path: Optional[Union[str, Path]] = None,
    workflow: Optional[Dict[str, Any]] = None,
) -> Optional[NextPhase]:
    """The next command after *finished_mode*, ``complete``, or ``None``.

    Args:
        plan_arg: The caller's plan argument, printed verbatim (quoted by
            :func:`quote_plan_arg` only when it needs quoting).
        host_id: The host the emitting process resolved.
        route: :func:`resolve_route`'s answer; ``None`` reads the host's
            ``default`` template. A route the host has no template for raises
            ``ValueError`` (a caller bug; emitters treat it as a builder
            failure).
        finished_mode: The mode this invocation ran (a mode word; aliases
            resolve). ``None`` from ``util next-command``.
        after_mode: ``util next-command --after``: where the walk starts, with
            the gate off. Ignored when *finished_mode* is given.
        plan_path: Where ``state.json`` lives — the resolved plan path, used
            for nothing else. Defaults to ``os.path.abspath(plan_arg)``.
        workflow: A ``get_workflow_status`` payload the caller already has
            (``run_status`` computes one). When ``None`` it is computed here.

    Returns:
        ``None`` when there is no answer: *finished_mode* (or *after_mode*) is
        neither a workflow phase nor ``full``, or the finished phase is not on
        record (the gate, *finished_mode* only). Otherwise a :class:`NextPhase`:
        the same phase with ``--resume`` when it is ``in_progress``; else the
        first open required phase after it (``complete`` when none is left),
        with the open optional phases in between as ``optional_before``.
        ``full``, and both modes ``None``, answer from ``suggested_next``
        instead (the fallback).
    """
    modes = _skill_module("modes")
    prereq = _skill_module("check_workflow_prerequisites")

    word = finished_mode if finished_mode is not None else after_mode
    gated = finished_mode is not None
    start: Optional[str] = None
    if word is not None:
        mode = modes.resolve(word)
        if mode is None:
            return None
        if mode.runs_workflow:
            start = None  # `full` names no single phase: the fallback.
        elif mode.is_workflow_phase and mode.phase:
            start = mode.phase
        else:
            return None  # ask, status, init, …: no phase, no answer.

    # Looked up before any state is read, so a (host, route) pair with no
    # template raises for every phase answer, complete included.
    template = _host_capabilities().next_command_template(host_id, route)

    if workflow is None:
        source = plan_arg if plan_path is None else plan_path
        workflow = _workflow_payload(Path(os.path.abspath(str(source))))
    phases: Dict[str, Dict[str, Any]] = workflow.get("phases") or {}
    order: List[str] = list(modes.PREREQUISITE_PHASES)
    done = tuple(prereq.DONE_STATUSES)

    def status_of(phase: str) -> Optional[str]:
        return (phases.get(phase) or {}).get("status")

    def is_open(phase: str) -> bool:
        return status_of(phase) not in done

    def command_for(phase: str) -> str:
        mode_word = _mode_word(modes, phase)
        args = f"--{mode_word} {quote_plan_arg(plan_arg)}"
        if status_of(phase) == "in_progress":
            args += f" {RESUME_FLAG}"
        return template.replace(ARGS_SLOT, args)

    def optional_steps(span: List[str]) -> Tuple[OptionalStep, ...]:
        # *span* is a slice of the phase order, so each phase appears at most
        # once; a notice never adds or removes an entry, it only supplies the
        # `why` of one the walk listed.
        steps: List[OptionalStep] = []
        for phase in span:
            if phase not in prereq.OPTIONAL_PHASES or not is_open(phase):
                continue
            # Offered only once everything it needs is on record: after
            # generate-tasks, review-tasks is offered and apply-task-suggestions
            # (whose review has not run) is not.
            if any(is_open(need) for need in prereq.PHASE_PREREQUISITES.get(phase, ())):
                continue
            steps.append(OptionalStep(
                phase=phase,
                command=command_for(phase),
                why=_optional_why(modes, phase, phases.get(phase) or {}),
            ))
        return tuple(steps)

    if start is None:
        primary = workflow.get("suggested_next")
        if primary is None:
            return _COMPLETE
        return NextPhase(
            phase=primary,
            command=command_for(primary),
            optional_before=optional_steps(order[: order.index(primary)]),
        )

    status = status_of(start)
    if status == "in_progress":
        # A resumable phase is its own answer; the span walk does not run.
        return NextPhase(phase=start, command=command_for(start))
    if gated and status not in done:
        # The finished phase is not on record: exit 0 at a pending marker, a
        # native first call, a fresh detach invoker. Not a finished phase.
        return None

    index = order.index(start)
    primary = next(
        (
            phase for phase in order[index + 1:]
            if phase not in prereq.OPTIONAL_PHASES and is_open(phase)
        ),
        None,
    )
    if primary is None:
        # An optional phase never keeps a workflow open, and one the walk did
        # not reach is never suggested: no `optional_before` on completion.
        return _COMPLETE
    return NextPhase(
        phase=primary,
        command=command_for(primary),
        optional_before=optional_steps(order[index + 1: order.index(primary)]),
    )


def _mode_word(modes, phase: str) -> str:
    """The dispatcher word for workflow *phase* (``code-review`` → ``review-code``)."""
    for mode in modes.MODES:
        if mode.is_workflow_phase and mode.phase == phase:
            return mode.name
    raise ValueError(f"no workflow-phase mode owns phase '{phase}'")


def _optional_why(modes, phase: str, entry: Mapping[str, Any]) -> str:
    """The workflow payload's notice on *phase*'s entry, else its generic why."""
    notice = entry.get("notice")
    if notice:
        return str(notice)
    for mode in modes.MODES:
        if mode.is_workflow_phase and mode.phase == phase and mode.optional_why:
            return mode.optional_why
    raise ValueError(f"optional phase '{phase}' has no Mode.optional_why")


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------

def render_next_phase_lines(answer: NextPhase) -> List[str]:
    """The block's lines, primary first (§2.2(2), §2.1 "Primary first").

    Complete renders only ``[WORKFLOW_COMPLETE]``. Otherwise the
    ``[NEXT_COMMAND]`` line comes first, then each ``optional_before`` entry's
    ``[NEXT_COMMAND_OPTIONAL_WHY]`` / ``[NEXT_COMMAND_OPTIONAL]`` pair in the
    answer's order. A command line holds the command and nothing else.
    """
    if answer.complete:
        return [WORKFLOW_COMPLETE_MARKER]
    lines = [f"{NEXT_COMMAND_MARKER} {answer.command}"]
    for step in answer.optional_before:
        lines.append(f"{NEXT_COMMAND_OPTIONAL_WHY_MARKER} {step.why}")
        lines.append(f"{NEXT_COMMAND_OPTIONAL_MARKER} {step.command}")
    return lines


# ---------------------------------------------------------------------------
# The emitters' shared step (§2.2)
# ---------------------------------------------------------------------------

class LookupFailed(Exception):
    """Raised by :func:`emit_answer` under ``strict=True``, after its stderr line."""


#: How the one stderr line a failed lookup prints opens. ``util next-command``
#: exits 3 with it, and the Claude Code report quotes it (§2.3).
LOOKUP_FAILED_PREFIX = "multi-llm: next-command lookup failed"


def lookup_failure_line(exc: BaseException) -> str:
    """The one stderr line naming a failed workflow read or builder raise."""
    message = " ".join(part.strip() for part in str(exc).splitlines() if part.strip())
    detail = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return f"{LOOKUP_FAILED_PREFIX} ({detail}); no next command printed"


def _state_file(plan_path: Union[str, Path]) -> Path:
    """Where ``StateManager`` keeps *plan_path*'s ``state.json`` (computed only)."""
    from .output_handler import output_dir_path

    return output_dir_path(Path(plan_path).resolve()) / "state.json"


def emit_answer(
    plan_arg: str,
    host_id: str,
    finished_mode: Optional[str],
    *,
    after_mode: Optional[str] = None,
    plan_path: Optional[Union[str, Path]] = None,
    workflow: Optional[Dict[str, Any]] = None,
    stream=None,
    stderr=None,
    strict: bool = False,
) -> Optional[NextPhase]:
    """Build the answer, print its block and set ``user_next_step``: all or nothing.

    The step every emitter shares (``multi_llm.main``, ``run_status``'s terminal
    poll, ``util next-command``), so the block and the key are one rendering of
    one answer. The route is resolved here, once, in the emitting process
    (:func:`resolve_route`); the caller supplies ``plan_arg``, ``host_id`` and
    ``finished_mode`` from its own sources (§2.1, "Where each emitter gets its
    inputs"). The block goes to *stream* (``sys.stdout`` by default; an
    invocation that asked for JSON output passes ``sys.stderr``).

    Outcomes, per §2.1's table: an answer prints the block and records the key
    (``complete`` included); ``None`` prints and records nothing. Any exception
    — the workflow read, the route, the builder — prints and records nothing
    either: one stderr line (:func:`lookup_failure_line`) names it and this
    returns ``None``, or raises :class:`LookupFailed` under *strict*. The block
    is written before the key is recorded, so a failed write leaves no key.

    With *finished_mode* given and no *workflow* handed in, a plan whose
    ``state.json`` does not exist answers ``None`` without reading anything:
    no phase can be on record there, so the gate could only refuse, and the
    workflow read would create the plan's output dir for a run that recorded
    nothing.
    """
    from . import cli_common

    err = sys.stderr if stderr is None else stderr
    try:
        if finished_mode is not None and workflow is None:
            source = plan_arg if plan_path is None else plan_path
            if not _state_file(os.path.abspath(str(source))).is_file():
                return None
        route = resolve_route(host_id, stderr=err)
        answer = next_phase_commands(
            plan_arg, host_id, route, finished_mode,
            after_mode=after_mode, plan_path=plan_path, workflow=workflow,
        )
        if answer is None:
            return None
        payload = answer.to_dict()
        out = sys.stdout if stream is None else stream
        for line in render_next_phase_lines(answer):
            print(line, file=out)
        try:
            out.flush()
        except (AttributeError, ValueError):  # pragma: no cover - closed stream
            pass
    except Exception as exc:  # noqa: BLE001 - advisory: never fails the caller
        print(lookup_failure_line(exc), file=err)
        if strict:
            raise LookupFailed(str(exc)) from exc
        return None
    cli_common.record_user_next_step(payload)
    return answer


# ---------------------------------------------------------------------------
# `multi_llm.py util next-command` (§2.2(3))
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """``util next-command <plan> [--after <mode>] [--host <id>]``.

    Read-only: it resolves the plan path only to find ``state.json`` and hands
    the plan string as received to the builder, with no ``finished_mode`` (the
    gate is off) and ``--after`` as ``after_mode``. Exit 0 with the block on
    stdout and the key set, or with nothing printed when there is no answer;
    exit 3 with one stderr line for an unknown ``--after``, a missing plan, or
    a workflow read / builder that raised.
    """
    import argparse

    modes = _skill_module("modes")
    parser = argparse.ArgumentParser(
        prog="multi_llm.py util next-command",
        description=(
            "Print the workflow's next-phase command for a plan, as the "
            "next-command block (or the workflow-complete line), and set "
            "MULTI_LLM_RESULT.user_next_step. Read-only; reports state.json as "
            "recorded. Prints nothing when there is no answer."
        ),
        epilog=(
            "exit codes: 0 = answered, or nothing to answer | 3 = unknown "
            "--after mode, missing plan, or the lookup failed (one stderr line, "
            "no command printed)."
        ),
    )
    parser.add_argument("plan", help="Path to the plan file, as the user typed it")
    parser.add_argument(
        "--after", metavar="MODE", default=None,
        help="The mode that just ran: the walk starts after its phase "
             "(any mode word or alias; `full` reads suggested_next)",
    )
    parser.add_argument(
        "--host", default=None,
        help="Host whose spelling to print (default: $MULTI_LLM_HOST, else "
             "claude-code)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.after is not None and modes.resolve(args.after) is None:
        aliases = ", ".join(f"{a}={t}" for a, t in sorted(modes.ALIASES.items()))
        print(
            f"ERROR: unknown --after mode '{args.after}'. Valid modes: "
            f"{', '.join(modes.mode_names())} (aliases: {aliases})",
            file=sys.stderr,
        )
        return 3
    plan_path = os.path.abspath(args.plan)
    if not os.path.isfile(plan_path):
        print(f"ERROR: plan file not found: {plan_path}", file=sys.stderr)
        return 3
    host_id = _host_capabilities().resolve_host_id(explicit=args.host)
    try:
        emit_answer(
            args.plan, host_id, None,
            after_mode=args.after, plan_path=plan_path, strict=True,
        )
    except LookupFailed:
        return 3
    return 0
