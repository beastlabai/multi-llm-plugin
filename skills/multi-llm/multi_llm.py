#!/usr/bin/env python3
"""multi_llm.py — the single entry point for every multi-llm mode.

    uv run --project <skill-dir> -- python <skill-dir>/multi_llm.py <mode> [args]

A host shim resolves *one* path, once, and routes nothing itself.

Grammar (one line, no shell preprocessing)::

    multi_llm.py <mode> [<positional>…] [--host <id>] [mode flags…]

* ``<mode>`` is a bare word and must be ``argv[1]``; unknown or missing is exit 3
  listing the valid modes. Users still type ``--review-plan``; the shim maps
  flag → mode word through the closed table in ``modes.py``, so this script never
  has to disambiguate a mode flag from an option.
* **Positionals are normalized into the flags the target already parses** — plan
  modes take ``<plan>`` → ``--plan-file``; ``ask`` takes ``<plan> <question>`` →
  ``--plan-file`` + ``--question`` (falling back to ``--question-file``, written
  *here* and never by the driving agent, when the question outgrows the argv
  budget); ``install-host`` takes ``<host>``; ``init`` takes none. An extra
  positional is exit 3, never silently dropped.
* **Everything after the positionals is forwarded verbatim** to that mode's own
  argparse parser, which stays the single source of truth for its options.
  ``multi_llm.py <mode> --help`` prints that parser's help.
* **Dispatch is in-process**, never ``subprocess``: the target's
  ``main(argv) -> int`` is imported lazily and called, so one ``uv`` resolution
  covers the whole invocation, a plan path containing a space survives by
  construction (the hand-off is a Python list — ``shlex`` never runs on this
  path), stream setup happens once, and foreground Ctrl-C reaches the code that
  owns the phase dir with no forwarding layer.
* **Exit codes are uniform and never remapped**: 0 done, 1 unexpected error,
  2 nothing to do / phase already complete, 3 needs a human. Exit **3** is
  reserved across every mode for "unattended run cannot proceed" (missing
  tooling, unmet prerequisite, contradictory or empty decision flag), so a host
  can tell "needs a human" from "already done" (exit 2) without parsing prose.
  The only codes this script invents are the 3s it raises for an unknown mode, a
  bad positional, a rejected flag combination, an empty/unrecognized decision
  value (D6 row 15) and a missing ``uv`` (D6 row 1 / D6c layer 3) — plus the one
  code it *rewrites*: argparse's usage exit (2) out of a target's own parser
  becomes 3, because 2 already means "phase already complete" and a host must
  never read a typo'd flag as a finished phase.
* **Exactly one machine-readable last stdout line per invocation** —
  ``MULTI_LLM_RESULT: {"mode","phase","state","exit","artifacts":[…],"next":…}``
  — emitted even on a crash, so the contract holds for a host that cannot parse
  prose or is tailing a detached log. ``next`` is ``null`` whenever no follow-up
  is owed, which is the normal case.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Absolute, symlinks preserved — see utils/cli_common.py for why never realpath.
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent))

import modes as mode_table
from utils.cli_common import (
    EXIT_ERROR,
    EXIT_NEEDS_HUMAN,
    EXIT_NOTHING_TO_DO,
    EXIT_OK,
    INTERNAL_FLAG_PREFIX,
    ResultContext,
    activate_result,
    clear_invocation,
    deactivate_result,
    dispatcher_command,
    flag_present,
    flag_value,
    record_invocation,
    render_instruction_text,
    skill_dir,
)
from utils.stream_bootstrap import bootstrap_streams
from utils.unattended import (
    NeedsHuman,
    require_uv,
    validate_decision_flags,
)

#: Above this many characters the ``ask`` question is handed over as a file
#: instead of an argv value. Well under the tightest real ceiling (Windows caps a
#: whole command line at 32767 chars; POSIX caps a single argument at 128 KiB),
#: because the same normalization feeds the detached-worker re-exec later.
MAX_ARGV_QUESTION_CHARS = 4096

#: The hidden re-entry flags this script reserves under the ``--internal-*``
#: prefix. Excluded from ``--help``, never rendered into a follow-up command or a
#: ``MULTI_LLM_RESULT.next`` value (see ``ResultContext.set_next``), and not part
#: of the compatibility surface. Targets own further ``--internal-*`` names of
#: their own (e.g. ``--internal-revalidation``), which is why the dispatcher
#: forwards the prefix rather than validating against this list.
INTERNAL_FLAGS: Tuple[str, ...] = tuple(
    INTERNAL_FLAG_PREFIX + name for name in ("reaggregate", "consolidate", "worker")
)


# ---------------------------------------------------------------------------
# The `util` namespace
# ---------------------------------------------------------------------------
# Utility scripts the instruction layer used to shell out to directly, routed the
# same in-process ``main(argv)`` way as the modes with flags forwarded verbatim to
# each script's existing parser. This is what makes "one substitution site per
# command" true rather than aspirational.
#
# `util` is NOT a user-facing surface: it is excluded from `argument-hint` and
# from `--help`'s mode list, documented only under `multi_llm.py util --help`,
# and — like `--internal-*` — never rendered into a MULTI_LLM_RESULT.next value.
UTIL_COMMANDS: Dict[str, Tuple[Optional[str], str, str]] = {
    "metrics": ("utils.metrics", "main", "Record/report per-subagent metrics in state.json."),
    "prior-changes": ("utils.prior_changes", "main", "Read, append to or clear the prior-batch change log."),
    "report": ("utils.html_report_generator", "main", "Regenerate an HTML review report with decisions overlaid."),
    "decisions": ("display_decisions", "main", "Print the decision summary from an orchestrator output file."),
    "update-tasks": ("update_plan_tasks", "main", "Write generated tasks into a plan's Tasks section."),
    "prereq": ("check_workflow_prerequisites", "main", "Check, skip or report a workflow phase's prerequisites."),
    "finalize-tracking": ("finalize_tracking", "main",
                          "Record the files implementation changed, and link the "
                          "summary from the plan."),
    "backup": ("utils.backup", "main", "Back up a file before it is modified."),
    "restore-batch": ("utils.restore_batch", "main",
                      "Restore a failed write batch from its pre-batch snapshot."),
    "runs": ("utils.run_registry", "main", "List (or --sweep) this project's detached runs."),
    "instructions": (None, "", "Print a mode's core instruction file with every path resolved."),
}


#: The instruction-refactor escape hatch. Set to ``1`` and ``util instructions``
#: serves the frozen pre-refactor core from ``legacy/`` instead of the host-neutral
#: one — a way back from the ~5k-line markdown rewrite that does **not** revert the
#: dispatcher, the executor or the unattended work with it. Read path only: same
#: modes, same flags, same exit codes. See ``legacy/README.md``.
LEGACY_INSTRUCTIONS_ENV = "MULTI_LLM_LEGACY_INSTRUCTIONS"

#: Where the frozen copies live, relative to the skill root.
LEGACY_INSTRUCTIONS_DIR = "legacy"


def _legacy_instruction_path(relative: str) -> Optional[Path]:
    """The frozen pre-refactor file for *relative*, when the hatch is open.

    ``None`` when the variable is unset (the overwhelmingly common case, and the
    only one that costs nothing) or when no frozen copy exists for that mode —
    a partial legacy tree falls back to the current core rather than failing,
    because the hatch is a mitigation and must not become a new failure mode.
    """
    if os.environ.get(LEGACY_INSTRUCTIONS_ENV) != "1":
        return None
    candidate = skill_dir() / LEGACY_INSTRUCTIONS_DIR / relative
    return candidate if candidate.exists() else None


class DispatchError(Exception):
    """A dispatcher-owned failure that maps to a fixed exit code."""

    def __init__(self, message: str, exit_code: int = EXIT_NEEDS_HUMAN):
        super().__init__(message)
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# Help rendering (from modes.py — never a hand-maintained second list)
# ---------------------------------------------------------------------------

def render_help() -> str:
    lines = [
        "usage: multi_llm.py <mode> [<positional>...] [--host <id>] [mode flags...]",
        "",
        "Run a multi-llm workflow phase. The mode is a bare word and must come first;",
        "every flag after it is forwarded verbatim to that mode's own parser, so",
        "`multi_llm.py <mode> --help` prints the authoritative flag list.",
        "",
        "modes:",
    ]
    width = max(len(m.name) for m in mode_table.MODES)
    for mode in mode_table.MODES:
        positionals = " ".join(mode.positionals)
        head = f"  {mode.name.ljust(width)}  {positionals}".rstrip()
        lines.append(f"{head.ljust(width + 22)}{mode.summary}")
        if mode.unavailable:
            lines.append(f"  {''.ljust(width)}  ({mode.unavailable.format(mode=mode.name)})")
    lines += [
        "",
        "aliases:",
    ]
    for alias in sorted(mode_table.ALIASES):
        lines.append(f"  {alias} -> {mode_table.alias_expansion(alias)}")
    lines += [
        "",
        "cross-cutting flags (accepted by every mode, and by each orchestrator",
        "directly): --host ID, --yes/-y/--non-interactive, --interactive/-i,",
        "--detach, --if-not-running, --max-detached N, --exec-subagents,",
        "--skip-prerequisites.",
        "",
        "Unattended is the default everywhere except Claude Code: --yes resolves",
        "every gate from config and flags and renders no question, --interactive",
        "is the opt-in to a rendered gate, and the two together are exit 3 (never",
        "a precedence rule). --yes never implies --skip-prerequisites.",
        "",
        "--detach self-daemonizes the long-running modes "
        f"({', '.join(mode_table.DETACH_MODES)}):",
        "it implies --yes, returns immediately with a log path plus a status",
        "command, and is stopped only by `status <plan> --abort`. Never wrap it in",
        "a host background primitive — pick one mechanism.",
        "",
        "exit codes: 0 done | 1 unexpected error | 2 nothing to do / phase already",
        "complete | 3 needs a human (an unattended run that cannot proceed:",
        "missing uv, unmet prerequisite, contradictory or empty decision flag, or",
        "a command line a parser refused).",
        "",
        "Every invocation ends with one machine-readable line:",
        '  MULTI_LLM_RESULT: {"mode","phase","state","exit","artifacts":[...],"next":...}',
        "",
        "Utility scripts live under a separate namespace: `multi_llm.py util --help`.",
    ]
    return "\n".join(lines)


def render_util_help() -> str:
    lines = [
        "usage: multi_llm.py util <command> [flags...]",
        "",
        "Utility scripts, routed in-process with flags forwarded verbatim to each",
        "script's own parser. Not a user-facing surface: `util` never appears in the",
        "skill's argument-hint, in `multi_llm.py --help`'s mode list, or in a",
        "MULTI_LLM_RESULT.next value.",
        "",
        "commands:",
    ]
    width = max(len(name) for name in UTIL_COMMANDS)
    for name, (_module, _attr, summary) in UTIL_COMMANDS.items():
        lines.append(f"  {name.ljust(width)}  {summary}")
    lines += [
        "",
        "`util instructions <mode>` prints instructions/<mode>.md with every",
        "skill-dir placeholder replaced by the running skill dir, so the text a host",
        "agent reads contains no unexpanded token.",
    ]
    return "\n".join(lines)


def _unknown_mode_message(word: str) -> str:
    return (
        f"ERROR: unknown mode '{word}'.\n"
        f"Valid modes: {', '.join(mode_table.mode_names())}\n"
        f"Aliases: "
        + ", ".join(f"{a}={t}" for a, t in sorted(mode_table.ALIASES.items()))
        + "\nRun `multi_llm.py --help` for the full grammar."
    )


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

def _split_positionals(rest: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Leading bare words are positionals; everything from the first flag on is
    forwarded verbatim.

    Deliberately not argparse: ``parse_known_args`` can reorder or swallow the
    residual argv, and the residual has to reach the target's own parser exactly
    as the user typed it.
    """
    index = 0
    while index < len(rest) and not rest[index].startswith("-"):
        index += 1
    return list(rest[:index]), list(rest[index:])


def _normalize_positionals(
    mode: "mode_table.Mode",
    positionals: Sequence[str],
    cleanup: List[Path],
) -> List[str]:
    """Turn a mode's positionals into the flags its target already parses."""
    if len(positionals) > len(mode.positionals):
        expected = " ".join(mode.positionals) or "(none)"
        raise DispatchError(
            f"ERROR: mode '{mode.name}' takes {len(mode.positionals)} positional "
            f"argument(s) {expected}, got {len(positionals)}: "
            f"{' '.join(positionals)}\n"
            "Flags must follow the positionals; an extra positional is never "
            "silently dropped."
        )

    normalized: List[str] = []
    for kind, value in zip(mode.positionals, positionals):
        if kind == mode_table.POS_PLAN:
            normalized += ["--plan-file", value]
        elif kind == mode_table.POS_QUESTION:
            normalized += _question_flags(value, cleanup)
        elif kind == mode_table.POS_HOST:
            # Forwarded verbatim as a positional: the install-host parser (and
            # with it the host-id vocabulary) is owned by the install design.
            normalized += [value]
        else:  # pragma: no cover - guarded by the table
            raise DispatchError(f"ERROR: unhandled positional kind {kind!r}")
    return normalized


def _question_flags(question: str, cleanup: List[Path]) -> List[str]:
    """``--question`` for a short question, else a dispatcher-written file.

    ``ask_orchestrator.py`` already accepts ``--question-file``; writing it here
    keeps user-controlled free text off every command line, and keeps the
    decision out of the driving agent's hands.
    """
    if len(question) <= MAX_ARGV_QUESTION_CHARS:
        return ["--question", question]
    handle, path = tempfile.mkstemp(prefix="multi_llm_question_", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(question)
    cleanup.append(Path(path))
    return ["--question-file", path]


def _host_capabilities():
    """Import the capability reader lazily (one yaml parse, only when needed)."""
    from utils import host_capabilities

    return host_capabilities


def resolve_host(argv: Sequence[str]) -> str:
    """The host id this invocation runs under: ``--host`` → env → ``claude-code``.

    An omitted flag resolves to the reference host, so a command that predates the
    capability data behaves exactly as it did. Never sniffed: a host is *declared*
    (by the shim that baked the flag in, or by the user), never guessed from the
    environment.
    """
    return _host_capabilities().resolve_host_id(argv)


def _validate_host(argv: Sequence[str]) -> None:
    """An unknown ``--host`` is exit 3 before anything is created (D5b)."""
    caps = _host_capabilities()
    caps.entry(caps.resolve_host_id(argv))


def _reject_incompatible_flags(mode: "mode_table.Mode", argv: Sequence[str]) -> None:
    """Reject contradictory combinations at parse time, never by precedence.

    Enforced here — before any phase dir is touched — because the alternative is
    each mode guessing which flag wins.
    """
    unattended = flag_present(argv, "--yes", "-y", "--non-interactive")
    if unattended and flag_present(argv, "--interactive", "-i"):
        raise DispatchError(
            "ERROR: --yes (or --non-interactive) and --interactive are "
            "contradictory: one forbids prompting, the other forces it. "
            "Pass exactly one."
        )
    if unattended and flag_value(argv, "--force-strategy") == "pause-and-ask":
        raise DispatchError(
            "ERROR: --force-strategy pause-and-ask requires a human at the "
            "per-task gate, which --yes forbids. Pass a non-interactive "
            "strategy (e.g. skip-dependents) or drop --yes."
        )
    _reject_capability_conflicts(mode, argv)


def _reject_capability_conflicts(mode: "mode_table.Mode", argv: Sequence[str]) -> None:
    """The foreground-only gates ``--detach`` is incompatible with.

    A detached worker has ``stdin`` at ``DEVNULL`` and no channel back to the
    host, so anything that would raise a question is a contradiction — and it is
    rejected **at parse time, in the foreground invoker, before the process
    daemonizes**, so the user sees it on their own terminal instead of finding it
    in a log by polling. ``--detach`` therefore implies ``--yes`` (every decision
    point auto-resolves from the documented defaults and the resolution is
    recorded in ``.status.json``), and any flag whose resolution needs a human is
    exit 3.

    ``--exec-subagents`` is rejected the same way, from the capability data rather
    than from a guess here — but in ``_require_executor_capability`` below, which
    runs *after* the implications so ``--detach``'s implied flag is gated exactly
    like a typed one.
    """
    if not flag_present(argv, "--detach"):
        return None
    if flag_present(argv, "--interactive", "-i"):
        raise DispatchError(
            "ERROR: --detach and --interactive are contradictory: a detached run "
            "has no terminal to prompt on (stdin is /dev/null and there is no "
            "channel back to the host), so --detach implies --yes. Pass exactly "
            "one."
        )
    if flag_value(argv, "--force-strategy") == "pause-and-ask":
        raise DispatchError(
            "ERROR: --force-strategy pause-and-ask requires a human at the "
            "per-task gate, which --detach forbids (it implies --yes). Pass a "
            "non-interactive strategy (e.g. skip-dependents) or drop --detach."
        )
    if not mode.supports_detach:
        # A flag that silently does nothing is worse than a refusal: the caller
        # would poll `status` forever for a run that was never detached.
        raise DispatchError(
            f"ERROR: --detach is not available for mode '{mode.name}'.\n"
            "       It covers the long-running fan-out modes today: "
            f"{', '.join(mode_table.DETACH_MODES)}.\n"
            "       Run this mode in the foreground (it emits task JSON and "
            "exits in seconds) or under your host's own background mechanism."
        )
    return None


def _apply_flag_implications(argv: List[str]) -> List[str]:
    """Write ``--detach``'s implications out explicitly rather than leaving them.

    The detached child's argv is rebuilt from the invoker's, so an implication
    that stayed implicit would have to be re-derived in a second place. Making it
    explicit here means the worker's recorded argv says exactly what it ran with.

    Two implications, both D4c's, and both about there being nobody on the other
    end: ``--yes`` because no one is there to *answer*, and ``--exec-subagents``
    because no one is there to *continue* — a detached worker without it would
    print ``[VALIDATION_PENDING]``/``[SALVAGE_NEEDED]`` and exit 0 with the phase
    unfinished, which a poller reads as ``run.state: done``, and the write phases
    would decline the scheduler outright.
    """
    if flag_present(argv, "--detach"):
        if not flag_present(argv, "--yes", "-y", "--non-interactive"):
            argv = list(argv) + ["--yes"]
        if not flag_present(argv, "--exec-subagents"):
            argv = list(argv) + ["--exec-subagents"]
    return argv


def _require_executor_capability(argv: Sequence[str]) -> None:
    """Gate ``--exec-subagents`` on the host's capability data (D3e/D5b).

    Called *after* ``_apply_flag_implications``, so the flag ``--detach`` implies
    is checked exactly like one the caller typed: a host whose entry declares no
    executor-capable runner (the ``agent-readonly``/``agent-write`` profiles)
    would otherwise accept the run, stop at a dispatch marker and wait for a
    subagent nobody starts.
    """
    if flag_present(argv, "--exec-subagents"):
        _host_capabilities().require_executor_runner(resolve_host(argv))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _load_target(module_path: str, attr: str) -> Callable[[Sequence[str]], int]:
    """Import a dispatch target lazily and return its ``main``.

    Import must be side-effect free: every target keeps argparse and all work
    inside ``main(argv)`` so two targets can safely run inside one interpreter
    (see tests/test_instruction_portability.py).
    """
    import importlib

    module = importlib.import_module(module_path)
    target = getattr(module, attr, None)
    if target is None:  # pragma: no cover - guarded by the table + tests
        raise DispatchError(
            f"ERROR: {module_path} has no '{attr}' entry point.",
            exit_code=EXIT_ERROR,
        )
    return target


def _call_target(target: Callable[[Sequence[str]], int], argv: Sequence[str]) -> int:
    """Call ``main(argv)``, awaiting it when the target is async."""
    import asyncio
    import inspect

    if inspect.iscoroutinefunction(target):
        result = asyncio.run(target(list(argv)))
    else:
        result = target(list(argv))
    return EXIT_OK if result is None else int(result)


def _run_mode(word: str, rest: Sequence[str], ctx: ResultContext) -> int:
    mode = mode_table.resolve(word)
    if mode is None:
        raise DispatchError(_unknown_mode_message(word))

    ctx.mode = mode.name
    ctx.phase = mode.phase

    if mode.unavailable:
        if flag_present(rest, "--help", "-h"):
            print(f"{mode.name}: {mode.summary}")
            print(f"  {mode.unavailable.format(mode=mode.name)}")
            return EXIT_OK
        raise DispatchError(
            f"ERROR: mode '{mode.name}' is {mode.unavailable.format(mode=mode.name)}."
        )

    assert mode.target_module is not None  # every available mode has a target
    cleanup: List[Path] = []
    try:
        positionals, flags = _split_positionals(rest)
        if flag_present(flags, "--help", "-h"):
            # Forward to the target's own parser: it is the source of truth.
            return _call_target(
                _load_target(mode.target_module, mode.target_attr), list(flags)
            )
        # The host id first: every rejection below may consult its capability
        # entry, and an unknown id must fail as itself rather than as a confusing
        # consequence.
        _validate_host(flags)
        _reject_incompatible_flags(mode, flags)
        # D6 row 15 then row 1, both before anything is created: an empty or
        # unrecognized decision value and a missing `uv` are the two failures a
        # mode must never discover halfway through a phase dir.
        validate_decision_flags(flags)
        if mode.requires_uv:
            # The two install modes opt out (modes.py): `install-host --check` is
            # layer 1 of the same uv bootstrap and reports a miss as one of its
            # five assertions, and `uninstall-host` must work on a machine whose
            # uv is already gone. Every other mode fails here, before anything is
            # created.
            require_uv()
        argv = (
            list(mode.injected_flags)
            + list(mode_table.alias_flags(word))
            + _normalize_positionals(mode, positionals, cleanup)
            + list(flags)
        )
        argv = _apply_flag_implications(argv)
        # After the implications: --detach's implied --exec-subagents faces the
        # same capability gate as an explicit one, so a host with no executor-
        # capable runner is exit 3 here rather than a marker nobody continues.
        _require_executor_capability(argv)
        # What a detached child re-executes: the mode word plus this normalized
        # argv, never a remembered sys.argv (see utils/detach.build_child_command).
        record_invocation(mode.name, argv)
        return _call_target(
            _load_target(mode.target_module, mode.target_attr), argv
        )
    finally:
        for path in cleanup:
            try:
                path.unlink()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass


def _run_util(rest: Sequence[str], ctx: ResultContext) -> int:
    ctx.mode = mode_table.UTIL_NAMESPACE
    if not rest or rest[0] in ("--help", "-h"):
        print(render_util_help())
        return EXIT_OK
    name, args = rest[0], list(rest[1:])
    if name not in UTIL_COMMANDS:
        raise DispatchError(
            f"ERROR: unknown util command '{name}'.\n"
            f"Valid commands: {', '.join(UTIL_COMMANDS)}\n"
            "Run `multi_llm.py util --help` for details."
        )
    ctx.mode = f"{mode_table.UTIL_NAMESPACE} {name}"
    if name == "instructions":
        return _run_util_instructions(args, ctx)
    module_path, attr, _summary = UTIL_COMMANDS[name]
    assert module_path is not None
    return _call_target(_load_target(module_path, attr), args)


def _run_util_instructions(args: Sequence[str], ctx: ResultContext) -> int:
    """Serve a mode's core instruction file with every placeholder resolved.

    The instruction markdown is static and no installer rewrites it, so a host
    that reads the file directly would have to apply the skill-dir binding by
    hand — an instruction-following step across a 5k-line corpus. Serving the
    rendered text from ``__file__`` removes that step and cannot go stale.
    """
    if not args or args[0] in ("--help", "-h"):
        served = [m.name for m in mode_table.MODES if m.instruction_file]
        print("usage: multi_llm.py util instructions <mode>")
        print(f"modes with a core instruction file: {', '.join(served)}")
        return EXIT_OK
    if len(args) > 1:
        raise DispatchError(
            "ERROR: `util instructions` takes exactly one mode word, got: "
            + " ".join(args)
        )
    word = args[0]
    mode = mode_table.resolve(word)
    if mode is None:
        raise DispatchError(_unknown_mode_message(word))
    relative = mode_table.instruction_path(mode.name)
    if relative is None:
        # Dispatcher-driven by design (`status`, `install-host`,
        # `uninstall-host`): there is no workflow to narrate, only flags. Every
        # shim names this command as its unconditional read step, so erroring
        # here would hand a host agent a failure before it had done anything —
        # observed on OpenCode, where the agent had to improvise past an exit 3.
        # Answer with the one useful thing instead and exit 0; a mode that
        # *declares* a core file which is missing is still a hard error below.
        print(
            f"Mode '{mode.name}' has no core instruction file: it is driven "
            "entirely by the dispatcher, so there are no workflow steps to "
            "follow. Its flags are the whole interface:"
        )
        print()
        print(f"  {dispatcher_command(mode.name)} --help")
        return EXIT_OK
    path = skill_dir() / relative
    legacy = _legacy_instruction_path(relative)
    if legacy is not None:
        # Announced on stderr, never stdout: `util instructions` writes the core
        # itself to stdout and a host may be piping it straight into a prompt.
        print(
            f"[LEGACY_INSTRUCTIONS] serving the pre-refactor core from "
            f"{legacy.parent.name}/{legacy.name} because "
            f"{LEGACY_INSTRUCTIONS_ENV}=1 (see legacy/README.md; Claude Code "
            "only, frozen, unmaintained).",
            file=sys.stderr,
        )
        path = legacy
    if not path.exists():
        raise DispatchError(
            f"ERROR: core instruction file missing: {path}\n"
            "The plugin install looks corrupt or partial; reinstall the skill.",
            exit_code=EXIT_ERROR,
        )
    sys.stdout.write(
        render_instruction_text(path.read_text(encoding="utf-8"), skill_dir())
    )
    ctx.add_artifact(path)
    return EXIT_OK


def _dispatch(argv: Sequence[str], ctx: ResultContext) -> int:
    if not argv:
        raise DispatchError(
            "ERROR: no mode given. The mode is a bare word and must be the first "
            f"argument.\nValid modes: {', '.join(mode_table.mode_names())}\n"
            "Run `multi_llm.py --help` for the full grammar."
        )
    word = argv[0]
    if word in ("--help", "-h", "help"):
        print(render_help())
        return EXIT_OK
    if word == mode_table.UTIL_NAMESPACE:
        return _run_util(argv[1:], ctx)
    if word.startswith("-"):
        raise DispatchError(
            f"ERROR: '{word}' is not a mode. The mode is a bare word and must be "
            "the first argument (e.g. `multi_llm.py review-plan plans/x.md "
            f"{word}`).\n" + _unknown_mode_message(word)
        )
    return _run_mode(word, argv[1:], ctx)


def _exit_code_of(exc: SystemExit) -> int:
    """A target's ``sys.exit`` is its return value; never remap 0/1/2/3.

    The one exception is argparse's usage exit — see ``_is_argparse_usage_exit``,
    which the caller consults; that code was never the target's to spend.
    """
    code = exc.code
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    print(str(code), file=sys.stderr)
    return EXIT_ERROR


def _is_argparse_usage_exit(exc: SystemExit) -> bool:
    """True when this ``SystemExit`` came out of argparse's usage-error path.

    ``ArgumentParser.error`` exits **2**, which is the one code this skill has
    already spent: 2 means "nothing to do / phase already complete" (the
    orchestrators' phase-complete guards, which raise it with a plain
    ``sys.exit(2)``). Passing argparse's 2 through would put
    ``state: nothing-to-do`` on the MULTI_LLM_RESULT line for a typo'd flag or a
    missing ``--plan-file``, and a host reading only the machine contract would
    conclude the phase was done and skip it — the worst possible reading of a
    command that never ran.

    The two are distinguishable only by origin, so the traceback is the
    discriminator: argparse raises from its own module frames, a phase guard from
    the orchestrator's. Matching on frames rather than on the message keeps this
    independent of argparse's (localized, version-dependent) wording.
    """
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_globals.get("__name__") == "argparse":
            return True
        tb = tb.tb_next
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Line buffering + UTF-8/replace stream encoding, once per invocation: there
    # is exactly one process per invocation, so this covers every mode and every
    # in-process continuation.
    bootstrap_streams()
    argv = list(sys.argv[1:] if argv is None else argv)
    ctx = activate_result(ResultContext())
    crash: Optional[str] = None
    exit_code = EXIT_ERROR
    try:
        try:
            exit_code = _dispatch(argv, ctx)
        except DispatchError as exc:
            print(str(exc), file=sys.stderr)
            exit_code = exc.exit_code
        except NeedsHuman as exc:
            # Every unattended gate that cannot resolve lands here with the same
            # code, whether it was raised by the dispatcher's own pre-checks or
            # by the mode it dispatched to.
            print(exc.message, file=sys.stderr)
            ctx.state = "needs-human"
            exit_code = exc.exit_code
        except SystemExit as exc:
            exit_code = _exit_code_of(exc)
            if exit_code == EXIT_NOTHING_TO_DO and _is_argparse_usage_exit(exc):
                # A parser refused the command line: that is a broken invocation
                # a human has to retype, so it lands on exit 3 with every other
                # "cannot proceed" — never on 2, which the phase-complete guards
                # own. argparse has already named the offending token on stderr.
                print(
                    "ERROR: that command line did not parse (see the usage error "
                    "above). Fix the invocation and re-run.\nExit 3, not 2: exit 2 "
                    "is reserved for 'nothing to do / phase already complete', and "
                    "nothing about this phase ran.",
                    file=sys.stderr,
                )
                exit_code = EXIT_NEEDS_HUMAN
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            exit_code = EXIT_ERROR
        except Exception:
            # The machine-readable contract has to hold on a crash too, so the
            # traceback goes to stderr and the result line is still the last
            # stdout line before this returns 1.
            traceback.print_exc()
            crash = _crash_summary()
            exit_code = EXIT_ERROR
        ctx.emit(exit_code)
        return exit_code
    finally:
        # If this process is a detached worker, it is the only thing that can
        # ever observe its own exit status (after the double-spawn nothing is its
        # parent), so it records its own outcome before exiting. A no-op in every
        # other invocation.
        _close_detached_run(exit_code, crash)
        deactivate_result()
        clear_invocation()


def _crash_summary(limit: int = 2000) -> str:
    """Message + traceback tail, for ``run.error`` on a crashed detached run."""
    text = traceback.format_exc()
    return text[-limit:]


def _close_detached_run(exit_code: Optional[int], error: Optional[str]) -> None:
    """Write the terminal ``run`` record when this process owns a detached run."""
    detach = sys.modules.get("utils.detach")
    if detach is None:
        return  # nothing detached ran; do not pay the import
    try:
        detach.finish_active_run(exit_code, error=error)
    except Exception:  # pragma: no cover - close-out must never mask the result
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
