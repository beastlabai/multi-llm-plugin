#!/usr/bin/env python3
"""The CLI surface every multi-llm entry point shares.

Four cross-cutting concerns live here so no entry point re-implements them:

* **The cross-cutting flag set.** ``--host``, ``--yes``/``--non-interactive``,
  ``--interactive``, ``--detach``, ``--exec-subagents`` and
  ``--skip-prerequisites`` are defined exactly
  once, as an ``add_help=False`` *parent* parser that ``multi_llm.py`` and every
  orchestrator parser mix in via ``parents=[...]``. Each orchestrator keeps its
  own parser as the source of truth for its own flags (the dispatcher forwards
  argv verbatim), so a direct
  ``python review_plan_orchestrator.py --plan-file … --host codex`` accepts the
  same flags with the same semantics as the dispatcher route. Sharing the
  definitions is what keeps "direct invocation is supported" from decaying into
  "works until it doesn't".

* **Skill-dir resolution.** Absolute, *symlinks preserved*
  (``os.path.abspath``, never ``realpath``): the dev workflow symlinks the Claude
  plugin-cache entry at a checkout, so resolving would rewrite a user's plugin
  invocation into a path they never named and move ``uv run --project`` off the
  ``pyproject.toml``/``uv.lock`` the host actually loaded. ``realpath`` belongs on
  *both* sides of a containment comparison and nowhere else.

* **Command building.** A follow-up command a host is expected to copy must be
  fully resolved — a ``${CLAUDE_SKILL_DIR}`` or ``{SKILL_DIR}`` token no shell
  expands is the bug this replaces — and it must keep its **allow-rule prefix**:
  Claude Code matches ``Bash(...)`` permissions as glob prefixes, so every
  command built here still starts with ``uv run`` (or
  ``PYTHONUNBUFFERED=1 uv``). Reshaping a printed command so it starts with
  something else (a wrapper script, a leading ``cd``, a shell ``&&`` chain)
  silently turns every follow-up into a permission prompt and must ship its own
  allow rule in the same commit.

* **The ``MULTI_LLM_RESULT`` line.** The single machine-readable last stdout line
  every ``multi_llm.py`` invocation ends with, so a host that cannot reliably
  parse prose (or is tailing a detached log) recovers mode, outcome, artifact
  paths and the one next command without regex-mining the transcript.

Absolute paths built here are **session-scoped, not artifacts**: the Claude
plugin cache is keyed by commit hash, so a skill-dir-absolute command saved in a
transcript or a state file can silently point at a directory that no longer
exists. Print them; never persist them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

# Absolute, symlinks preserved. Module-level because it is a pure string
# computation on ``__file__`` — no I/O at import. Tests that need a fixture skill
# dir (e.g. one containing a space) monkeypatch this attribute.
_SKILL_DIR = Path(os.path.abspath(__file__)).parent.parent

#: Prefix reserved for hidden re-entry flags. Excluded from ``--help``, never
#: rendered into a user-facing follow-up command or a ``MULTI_LLM_RESULT.next``
#: value, and not part of the compatibility surface.
INTERNAL_FLAG_PREFIX = "--internal-"

#: The marker every dispatcher invocation ends its stdout with.
RESULT_MARKER = "MULTI_LLM_RESULT:"

#: Uniform exit codes across every mode. The dispatcher never remaps a target's
#: code and invents one of its own only for an unknown mode or a rejected flag
#: combination.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOTHING_TO_DO = 2
EXIT_NEEDS_HUMAN = 3

_STATE_BY_EXIT = {
    EXIT_OK: "done",
    EXIT_ERROR: "error",
    EXIT_NOTHING_TO_DO: "nothing-to-do",
    EXIT_NEEDS_HUMAN: "needs-human",
}


# ---------------------------------------------------------------------------
# Skill-dir resolution and command building
# ---------------------------------------------------------------------------

def skill_dir() -> Path:
    """The running skill directory: absolute, symlinks preserved (see module doc)."""
    return _SKILL_DIR


def _quote(text: Union[str, Path]) -> str:
    """Encode one argument for the POSIX shell every rendered command targets.

    ``shlex.quote``, never hand-written double quotes: a double-quoted string
    still expands ``$VAR``, a backtick substitution and ``$(cmd)``, and every
    string quoted here is *filesystem-derived* — the skill dir, a plan path, a
    question. A checkout under ``/tmp/$(id).d`` would turn a printed follow-up
    into arbitrary command execution the moment a user ran it, which is not a
    thing a copy-pasteable command may do. Single quotes are inert in every
    POSIX shell, and POSIX is the *only* shell these commands are written for:
    every rendered shim and follow-up is a POSIX-shell command block, which is
    exactly why Git Bash is a hard install-time prerequisite on Windows
    (``host_install.GIT_BASH_MISSING`` — "there is no PowerShell instruction
    path"). So there is no second, ``cmd.exe``-flavoured renderer
    here to keep in sync; adding a native-Windows command surface means adding
    its encoder next to this one, not loosening this one.

    An argument with no shell-special character comes back bare, so an ordinary
    command still reads the way a user would type it.
    """
    return shlex.quote(str(text))


def dispatcher_prefix(base: Optional[Path] = None) -> str:
    """The invariant ``uv run … multi_llm.py`` prefix, fully resolved.

    This is the *one* substitution site a command needs: ``multi_llm.py``
    self-locates via ``__file__``, so the skill dir is needed only for
    ``uv run --project`` and the script path, and both live here.
    """
    base = Path(base) if base is not None else skill_dir()
    return f"uv run --project {_quote(base)} -- python {_quote(base / 'multi_llm.py')}"


def dispatcher_command(
    mode: str,
    *args: Union[str, Path],
    base: Optional[Path] = None,
) -> str:
    """Build ``uv run … multi_llm.py <mode> <args…>`` with every path resolved.

    **Every** element goes through ``_quote`` — the mode word included. A bare
    flag and a plain mode word carry no shell-special character, so they come
    back unchanged and the result still reads the way a user would type it;
    what the uniform encoding removes is the "looks like a flag, so pass it
    through raw" branch, which handed a shell whatever a caller had (an ``ask``
    question, a ``--force-strategy`` value) with no encoding at all.
    """
    parts = [dispatcher_prefix(base), _quote(mode)]
    for arg in args:
        parts.append(_quote(arg))
    return " ".join(parts)


def reaggregate_command(
    mode: str,
    plan_file: Union[str, Path],
    *,
    base: Optional[Path] = None,
) -> str:
    """The one follow-up command a native-subagent host is ever handed.

    Everything downstream of it happens inside that single invocation, which is
    what keeps the staleness window of a printed absolute path to one turn.
    """
    return dispatcher_command(mode, plan_file, "--reaggregate", base=base)


def durable_reaggregate_fields(
    orchestrator_script: str, plan_file: Union[str, Path]
) -> Dict[str, str]:
    """What a *persisted* checkpoint may store about its follow-up command.

    Never the resolved command: the Claude plugin cache is keyed by commit hash,
    so a skill-dir-absolute command written into ``.status.json`` points at a
    directory that no longer exists after the next plugin update. Skill-dir
    movement is **rebindable** — unlike a content hash — so the durable form is the
    mode word plus the plan path, and the reader rebuilds the command from the
    *running* skill dir (``utils/detach.resolved_reaggregate_command``).
    """
    mode = _modes().mode_for_script(orchestrator_script) or orchestrator_script
    return {"reaggregate_mode": mode, "plan_file": str(plan_file)}


def reaggregate_command_for_script(
    orchestrator_script: str,
    plan_file: Union[str, Path],
    *,
    base: Optional[Path] = None,
) -> str:
    """``reaggregate_command`` keyed by a legacy ``*_orchestrator.py`` filename.

    The orchestrators pass their own filename around as ``orchestrator_script``;
    the mode word is looked up in ``modes.py`` so the printed command names the
    single entry point rather than a second, unadvertised one.
    """
    mode = _modes().mode_for_script(orchestrator_script) or orchestrator_script
    return reaggregate_command(mode, plan_file, base=base)


def modes_table():
    """The ``modes`` module, for callers under ``utils/`` (public alias)."""
    return _modes()


def _modes():
    """Import the top-level ``modes`` table, tolerating direct script runs."""
    try:
        import modes  # type: ignore
    except ImportError:  # pragma: no cover - only when utils/ is sys.path[0]
        sys.path.insert(0, str(skill_dir()))
        import modes  # type: ignore
    return modes


# ---------------------------------------------------------------------------
# Instruction rendering (the token-free core `util instructions <mode>` serves)
# ---------------------------------------------------------------------------

#: Every spelling of the skill-dir placeholder that may appear in the corpus,
#: longest first so a substitution cannot eat part of another token.
#: ``${SKILL_DIR}`` and ``$SKILL_DIR`` are deliberately absent: they are never
#: defined and are a test failure wherever they appear, so the brace token can
#: never be mistaken for a variable that merely was not set.
_SKILL_DIR_TOKENS = (
    "${CLAUDE_SKILL_DIR}",
    "${MULTI_LLM_SKILL_DIR}",
    "$MULTI_LLM_SKILL_DIR",
    "{SKILL_DIR}",
)


def render_instruction_text(text: str, base: Union[str, Path]) -> str:
    """Substitute every skill-dir placeholder in instruction text.

    Pure so a test can render against a fixture skill dir containing a space.
    Substituting at read time from the *running* skill dir is also why the served
    core cannot go stale: there is no rendered copy to keep in sync.
    """
    rendered = text
    for token in _SKILL_DIR_TOKENS:
        rendered = rendered.replace(token, str(base))
    return rendered


# ---------------------------------------------------------------------------
# The shared parent parser
# ---------------------------------------------------------------------------

def build_parent_parser(
    *,
    include_yes: bool = True,
    include_interactive: bool = True,
) -> argparse.ArgumentParser:
    """The cross-cutting flags, defined once.

    Mixed into a child parser with ``parents=[build_parent_parser()]``. The two
    toggles exist only for a parser that must *not* offer a flag at all (e.g. a
    utility with no notion of interactivity); they are not an invitation to
    redefine one locally.

    ``--detach`` now has behaviour for the long-running fan-out modes (see
    ``utils/detach.py``); ``--exec-subagents`` is still accepted and stored only.
    The two hidden ``--internal-*`` re-entry flags are defined here rather than
    per orchestrator because the detached worker is re-executed through
    ``multi_llm.py`` and every mode's parser has to accept them.
    """
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("cross-cutting options")
    group.add_argument(
        "--host",
        metavar="ID",
        default=None,
        help="Host harness driving this run (e.g. claude-code, codex, opencode). "
             "Recorded for capability resolution; does not change orchestration.",
    )
    if include_yes:
        group.add_argument(
            "--yes",
            "-y",
            "--non-interactive",
            dest="yes",
            action="store_true",
            help="Run unattended: never prompt, resolve every gate from config "
                 "and flags. Confirms bulk approval operations.",
        )
    if include_interactive:
        group.add_argument(
            "--interactive",
            "-i",
            action="store_true",
            help="Force interactive model selection (ignores YAML defaults)",
        )
    group.add_argument(
        "--detach",
        action="store_true",
        help="Run the phase in a self-daemonized process (fan-out modes only) "
             "and return immediately with its log path and a status command to "
             "poll. Implies --yes. Never combine with a host's own background "
             "primitive.",
    )
    group.add_argument(
        "--if-not-running",
        action="store_true",
        help="With --detach: if a run for this phase is already in flight, print "
             "its run_id/log/status command and exit 0 instead of failing.",
    )
    group.add_argument(
        "--max-detached",
        type=int,
        default=None,
        metavar="N",
        help="Ceiling on concurrent detached runs for this project "
             "(default 4, or $MULTI_LLM_MAX_DETACHED).",
    )
    group.add_argument(
        "--exec-subagents",
        action="store_true",
        help="Run materialized subagent prompts through the provider executor "
             "instead of a host's native subagent tool, and continue past each "
             "marker in the same process (read/analyze phases: validation, "
             "revalidation, consolidation, salvage, judging). Runner and limits "
             "come from providers.yaml `driver:`.",
    )
    # D3l: the recommended first unattended run on a new host, and what CI uses to
    # assert a shim's write path is sane without touching a real repo. Deliberately
    # NOT implied by anything: it changes a run into a preflight.
    group.add_argument(
        "--dry-run-subagents",
        action="store_true",
        help="Preflight every subagent dispatch and start no child: resolve the "
             "runner, render the exact argv, check prompt transport and size, "
             "apply the invocation profile, the write-concurrency admission check "
             "and driver.write_paths — then write "
             "{phase}/subagent_dry_run.json and exit 0 having changed nothing.",
    )
    # D6a: consent to run unattended is not consent to stale input, so this is a
    # separate flag and `--yes` never implies it. Defined here rather than per
    # orchestrator because the two gated modes plus
    # `check_workflow_prerequisites.py` all need the same spelling and the same
    # semantics (loud, recorded, non-fatal).
    group.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Proceed even though a prerequisite phase's output was never "
             "applied. Loud and recorded (the skip lands in state.json and in "
             "every later report); never implied by --yes.",
    )
    # Hidden re-entry flags: the detached worker is re-executed as
    # `multi_llm.py <mode> … --internal-worker --internal-run-id <uuid>`, so every
    # mode's parser must accept them. Suppressed from --help, never rendered into
    # a follow-up command or a MULTI_LLM_RESULT.next value.
    group.add_argument(
        INTERNAL_FLAG_PREFIX + "worker",
        dest="internal_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        INTERNAL_FLAG_PREFIX + "run-id",
        dest="internal_run_id",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parent


# ---------------------------------------------------------------------------
# The MULTI_LLM_RESULT line
# ---------------------------------------------------------------------------

class ResultContext:
    """Collects the one machine-readable line an invocation ends with.

    A target that wants to contribute reaches the *active* context through the
    module-level helpers below rather than being handed one, so adding a recorded
    artifact never changes a ``main(argv)`` signature.
    """

    def __init__(self, mode: Optional[str] = None, phase: Optional[str] = None):
        self.mode = mode
        self.phase = phase
        self.state: Optional[str] = None
        self.artifacts: List[str] = []
        self.next: Optional[str] = None
        self._emitted = False

    # -- recording ------------------------------------------------------
    def add_artifact(self, path: Union[str, Path]) -> None:
        text = str(path)
        if text and text not in self.artifacts:
            self.artifacts.append(text)

    def set_next(self, command: Optional[str]) -> None:
        """Record the single follow-up command, if one is genuinely owed.

        Silently drops a command carrying an ``--internal-*`` flag: those are
        re-entry plumbing and are never handed to a host.
        """
        if command and INTERNAL_FLAG_PREFIX in command:
            return
        self.next = command or None

    # -- emitting -------------------------------------------------------
    def payload(self, exit_code: int) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "phase": self.phase,
            "state": self.state or _STATE_BY_EXIT.get(exit_code, "error"),
            "exit": exit_code,
            "artifacts": list(self.artifacts),
            "next": self.next,
        }

    def emit(self, exit_code: int, stream=None) -> None:
        """Write the terminal line. Idempotent: exactly one line per invocation."""
        if self._emitted:
            return
        self._emitted = True
        out = stream if stream is not None else sys.stdout
        out.write(f"{RESULT_MARKER} {json.dumps(self.payload(exit_code))}\n")
        try:
            out.flush()
        except (AttributeError, ValueError):  # pragma: no cover - closed stream
            pass


_ACTIVE: Optional[ResultContext] = None


def activate_result(context: ResultContext) -> ResultContext:
    """Make ``context`` the one the recording helpers below reach."""
    global _ACTIVE
    _ACTIVE = context
    return context


def deactivate_result() -> None:
    global _ACTIVE
    _ACTIVE = None


def active_result() -> Optional[ResultContext]:
    return _ACTIVE


def record_artifact(path: Union[str, Path]) -> None:
    """Record an artifact path if a dispatcher run is active; else a no-op.

    A no-op under direct invocation is deliberate: an orchestrator run outside
    the dispatcher has no ``MULTI_LLM_RESULT`` line to contribute to, and must
    behave identically either way.
    """
    if _ACTIVE is not None:
        _ACTIVE.add_artifact(path)


def record_artifacts(paths: Iterable[Union[str, Path]]) -> None:
    for path in paths:
        record_artifact(path)


def record_next(command: Optional[str]) -> None:
    """Record the single follow-up command, if a dispatcher run is active."""
    if _ACTIVE is not None:
        _ACTIVE.set_next(command)


# ---------------------------------------------------------------------------
# The invocation record (what a detached child re-executes)
# ---------------------------------------------------------------------------

_INVOCATION: Optional[Dict[str, Any]] = None


def record_invocation(mode: str, argv: Sequence[str]) -> None:
    """Remember the mode word and the **normalized** argv the dispatcher built.

    ``--detach`` rebuilds the child's command line from this rather than echoing
    ``sys.argv``, which is what keeps the detached worker identical whether the
    invoker was reached through a rendered shim, the ``"$MULTI_LLM_SKILL_DIR"``
    spelling or a direct call. Recording it here (instead of threading argv
    through every orchestrator signature) also means a direct
    ``python review_plan_orchestrator.py`` still works — it simply falls back to
    the argv its own ``main`` received.
    """
    global _INVOCATION
    _INVOCATION = {"mode": mode, "argv": list(argv)}


def clear_invocation() -> None:
    global _INVOCATION
    _INVOCATION = None


def invocation() -> Optional[Dict[str, Any]]:
    return _INVOCATION


def parse_result_line(text: str) -> Optional[Dict[str, Any]]:
    """Parse the ``MULTI_LLM_RESULT`` line out of captured stdout, if present."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(RESULT_MARKER):
            return json.loads(stripped[len(RESULT_MARKER):].strip())
    return None


# ---------------------------------------------------------------------------
# argv inspection (used by the dispatcher's parse-time rejection rules)
# ---------------------------------------------------------------------------

def _flag_matches(head: str, name: str) -> bool:
    """Whether the option-name token ``head`` would bind to option ``name``.

    Exact spelling **or an argparse abbreviation**: no parser in the skill passes
    ``allow_abbrev=False``, so argparse resolves ``--y``, ``--intera`` and
    ``--det`` to their full options — and a dispatcher rule that scanned for the
    exact spelling only was therefore bypassable by typing three fewer
    characters. ``--det`` on a mode that does not support ``--detach`` sailed
    past the D6b rejection and ran the mode with a flag that does nothing, which
    is precisely the outcome that rejection exists to prevent.

    ``allow_abbrev`` cannot be fixed from *this* side of the call: it is a
    property of the parser doing the parsing (and is **not** inherited through
    ``parents=[...]``), so the scan is what is taught argparse's rule instead —
    one place rather than one per parser, and it stays correct for the
    orchestrator parsers a direct invocation reaches too.

    Abbreviation is long-option only (argparse never abbreviates a short flag),
    and a token is only ever matched against a *longer* name, so a real
    ``--host-something`` never reads as ``--host``. A prefix that is ambiguous
    across the target's real options (``--d`` for ``--detach`` and
    ``--dry-run-subagents``) is reported present here where argparse would
    instead refuse the whole command line: both end the run at exit 3, and
    over-reporting keeps the guard closed rather than open.
    """
    if head == name:
        return True
    if not name.startswith("--"):
        return False  # short options never abbreviate
    if not head.startswith("--") or len(head) <= 2:
        return False  # bare `--`, a short flag, or a positional
    return name.startswith(head)


def flag_present(argv: Sequence[str], *names: str) -> bool:
    """True if any of ``names`` appears in ``argv``, bare or as ``--flag=value``.

    Matches the way argparse would — see ``_flag_matches`` for why an
    abbreviation counts.
    """
    for token in argv:
        head = token.split("=", 1)[0]
        for name in names:
            if _flag_matches(head, name):
                return True
    return False


def flag_value(argv: Sequence[str], name: str) -> Optional[str]:
    """The value of ``--name value`` or ``--name=value``, else ``None``.

    The **last** occurrence wins, because that is what a ``store`` action does:
    reading the first meant ``--force-strategy skip-dependents --force-strategy
    pause-and-ask`` scanned as ``skip-dependents`` while the run really got
    ``pause-and-ask``. Abbreviations bind here too (``--force-strat=…``); a
    trailing occurrence with no value left to take is argparse's error to report
    and never erases a value an earlier one supplied.
    """
    found: Optional[str] = None
    for index, token in enumerate(argv):
        head, separator, inline = token.partition("=")
        if not _flag_matches(head, name):
            continue
        if separator:
            found = inline
        elif index + 1 < len(argv):
            found = argv[index + 1]
    return found
