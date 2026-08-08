#!/usr/bin/env python3
"""The unattended contract: one disposition per decision point (D6).

Every interactive decision point in this skill resolves **without** a structured
ask tool. This module is the single owner of what that means, so no message, no
instruction file and no test has to re-derive it:

* :data:`ROWS` — the fifteen decision *sites* (not the 51 occurrences: a site
  restated in four mode files is still one decision), each keyed by a stable
  ``site_id``, the modes that reach it, the disposition it lands on and the D5a
  decision flag that answers it. Exactly three dispositions exist —
  :data:`AUTO_SELECT` (a documented default is applied *and recorded*),
  :data:`FAIL_CLOSED` (exit 3, "needs a human", nothing downstream runs) and
  :data:`EXPLICIT_FLAG` (the run proceeds only because the user typed a flag
  naming the choice). **"Ask in free text" is not a fourth option**: a headless
  run renders nothing, so a question is never a fallback — it is the failure this
  module exists to remove.
* :data:`DECISION_FLAGS` — the D5a input contract. No decision may exist only as
  a question: every gate has a flag whose values enumerate exactly the options
  the question would render, validated by ``choices`` (or by
  :func:`validate_decision_flags` for the argv-level pre-check), never guessed.
  An empty or unrecognized value is **exit 3 at parse time** (row 15).
* :func:`require_uv` — the run-time backstop half of the uv bootstrap (D6c layer
  3). The wording lives in ``hosts/uv-missing.txt`` so the shim preflight, the
  ``install-host --check`` prerequisite report and this probe can never disagree.
* :func:`record_resolution` — an auto-resolution is only "auto-select" if it is
  attributable afterwards, so every one of them prints a marker and (under
  ``--detach``) lands in ``{phase}/.status.json`` ``run.resolutions``.

**Nothing here sniffs its environment.** There is no ``isatty()`` call, no ``CI``
read and no ``--print`` detection anywhere in this skill's Python: interactivity
is *requested* (a user-typed ``--interactive``), never detected, so the same argv
produces the same phases in a terminal and in a pipe (D6b). The one former sniff
— ``full-workflow.md``'s "no TTY and no ``--yes`` → skip review-tasks" branch —
is deleted, and ``tests/test_unattended.py`` fails any reintroduction.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # normal import (utils is a package)
    from .cli_common import (
        EXIT_NEEDS_HUMAN,
        EXIT_OK,
        active_result,
        skill_dir,
    )
except ImportError:  # pragma: no cover - direct script runs with utils/ on path
    from cli_common import (  # type: ignore
        EXIT_NEEDS_HUMAN,
        EXIT_OK,
        active_result,
        skill_dir,
    )


# ---------------------------------------------------------------------------
# Dispositions
# ---------------------------------------------------------------------------

#: A documented default is applied and recorded.
AUTO_SELECT = "auto-select"
#: Exit 3, "needs a human"; nothing downstream runs.
FAIL_CLOSED = "fail-closed"
#: The run proceeds only because the user typed a flag naming the choice.
EXPLICIT_FLAG = "explicit-flag"

DISPOSITIONS: Tuple[str, ...] = (AUTO_SELECT, FAIL_CLOSED, EXPLICIT_FLAG)


class NeedsHuman(Exception):
    """A gate that cannot be resolved without a human: always exit 3.

    Distinct from the exit-2 phase-already-complete guard on purpose — a host has
    to be able to tell "needs a human" from "already done" without parsing prose.
    """

    exit_code = EXIT_NEEDS_HUMAN

    def __init__(self, message: str, *, site: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.site = site


# ---------------------------------------------------------------------------
# The fifteen rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    """One decision site.

    Attributes:
        number: D6 table row number (1..15). Stable; quoted by messages.
        site_id: The identifier recorded in ``run.resolutions`` and used by
            tests, so the table stays the single source of truth for what the
            default *was*.
        site: One-line description of the decision.
        modes: Mode words that can reach it.
        dispositions: Where it lands when nobody is watching, in the order the
            D6 disposition table names them.
        flags: The D5a decision flags that answer it, most specific first. Empty
            when the answer is out-of-band by design (rows 1 and 13) or when the
            row *is* the validator (row 15).
        enforced_in: Where the disposition is implemented. A markdown path means
            core-documented; a ``.py`` path means Python-resolved.
    """

    number: int
    site_id: str
    site: str
    modes: Tuple[str, ...]
    dispositions: Tuple[str, ...]
    flags: Tuple[str, ...]
    enforced_in: Tuple[str, ...]


ROWS: Tuple[Row, ...] = (
    Row(
        number=1,
        site_id="uv-missing",
        site="uv is not installed (probe before the first `uv run`)",
        modes=("*",),
        dispositions=(FAIL_CLOSED,),
        flags=(),
        enforced_in=("utils/unattended.py", "hosts/uv-missing.txt",
                     "references/uv-check.md"),
    ),
    Row(
        number=2,
        site_id="model-selection",
        site="which models to fan out to",
        modes=("review-plan", "review-tasks", "review-code", "ask", "full"),
        dispositions=(AUTO_SELECT, FAIL_CLOSED, EXPLICIT_FLAG),
        flags=("--models", "--quick"),
        enforced_in=("utils/interactive.py",),
    ),
    Row(
        number=3,
        site_id="partial-results-recovery",
        site="some models produced no result (partial / resume / rerun-all)",
        modes=("review-plan", "review-tasks", "review-code", "ask"),
        dispositions=(AUTO_SELECT, FAIL_CLOSED),
        flags=("--force", "--reaggregate", "--rerun-all"),
        enforced_in=("utils/review_orchestrator_base.py", "ask_orchestrator.py",
                     "instructions/review-plan.md"),
    ),
    Row(
        number=4,
        site_id="consolidation-offer",
        site="run consolidation after [CONSOLIDATION_RECOMMENDED]?",
        modes=("review-plan", "full"),
        dispositions=(AUTO_SELECT,),
        flags=("--consolidate",),
        enforced_in=("instructions/review-plan.md", "instructions/full-workflow.md"),
    ),
    Row(
        number=5,
        site_id="generate-tasks-prerequisite",
        site="unapplied plan-review suggestions before generate-tasks",
        modes=("generate-tasks", "full"),
        dispositions=(AUTO_SELECT, FAIL_CLOSED, EXPLICIT_FLAG),
        flags=("--skip-prerequisites",),
        enforced_in=("check_workflow_prerequisites.py",),
    ),
    Row(
        number=6,
        site_id="implement-prerequisite",
        site="unapplied plan-review suggestions before implement",
        modes=("implement", "full"),
        dispositions=(AUTO_SELECT, FAIL_CLOSED, EXPLICIT_FLAG),
        flags=("--skip-prerequisites",),
        enforced_in=("check_workflow_prerequisites.py", "implement_orchestrator.py"),
    ),
    Row(
        number=7,
        site_id="task-suggestion-advisory",
        site="unapplied task-review suggestions before implement (advisory)",
        modes=("implement",),
        dispositions=(AUTO_SELECT,),
        flags=("--no-confirm",),
        enforced_in=("implement_orchestrator.py",),
    ),
    Row(
        number=8,
        site_id="human-task-strategy",
        site="how to handle tasks that need a human",
        modes=("implement", "full"),
        dispositions=(AUTO_SELECT, EXPLICIT_FLAG),
        flags=("--force-strategy",),
        enforced_in=("implement_orchestrator.py",),
    ),
    Row(
        number=9,
        site_id="human-task-pause",
        site="per-human-task pause under --force-strategy pause-and-ask",
        modes=("implement",),
        dispositions=(FAIL_CLOSED, EXPLICIT_FLAG),
        flags=("--human-task",),
        enforced_in=("multi_llm.py", "implement_orchestrator.py"),
    ),
    Row(
        number=10,
        site_id="needs-human-decision",
        site="needs-human-decision items (per item, or the batched UI)",
        modes=("apply-suggestions", "apply-task-suggestions", "apply-code-fixes",
               "review-tasks", "full"),
        dispositions=(AUTO_SELECT,),
        flags=("--claude-decide", "--no-confirm"),
        enforced_in=("utils/apply_orchestrator_base.py",),
    ),
    Row(
        number=11,
        site_id="no-selections-confirmation",
        site="no user selections found — apply ALL valid items?",
        modes=("apply-suggestions", "apply-task-suggestions", "apply-code-fixes",
               "full"),
        dispositions=(AUTO_SELECT,),
        flags=("--no-confirm",),
        enforced_in=("utils/filtering.py", "utils/apply_orchestrator_base.py"),
    ),
    Row(
        number=12,
        site_id="review-tasks-continuation",
        site="review the generated tasks before implementing?",
        modes=("full",),
        dispositions=(AUTO_SELECT, EXPLICIT_FLAG),
        flags=("--review-tasks",),
        enforced_in=("instructions/full-workflow.md",),
    ),
    Row(
        number=13,
        site_id="no-tasks-to-review",
        site="review-tasks against a plan that has no task list",
        modes=("review-tasks",),
        dispositions=(FAIL_CLOSED,),
        flags=(),
        enforced_in=("check_workflow_prerequisites.py",),
    ),
    Row(
        number=14,
        site_id="phase-failure-recovery",
        site="a phase failed — retry / run manually / cancel",
        modes=("full", "*"),
        dispositions=(FAIL_CLOSED,),
        flags=("--resume",),
        enforced_in=("instructions/full-workflow.md",),
    ),
    Row(
        number=15,
        site_id="empty-or-ambiguous-selection",
        site="a decision flag that arrives empty or unrecognized",
        modes=("*",),
        dispositions=(FAIL_CLOSED,),
        flags=(),
        enforced_in=("utils/unattended.py", "multi_llm.py"),
    ),
)

_ROWS_BY_NUMBER: Dict[int, Row] = {r.number: r for r in ROWS}
_ROWS_BY_SITE: Dict[str, Row] = {r.site_id: r for r in ROWS}


def row(number: int) -> Row:
    """The D6 row with this number."""
    return _ROWS_BY_NUMBER[number]


def row_for_site(site_id: str) -> Row:
    """The D6 row with this ``site_id``."""
    return _ROWS_BY_SITE[site_id]


def site_ids() -> Tuple[str, ...]:
    return tuple(r.site_id for r in ROWS)


# ---------------------------------------------------------------------------
# D5a — the decision-input contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlagSpec:
    """One non-interactive decision flag.

    Attributes:
        name: The long flag spelling.
        rows: D6 row numbers it answers.
        choices: Legal values, when the flag is valued and enumerable. Empty for
            a boolean flag.
        value_form: ``"none"`` (boolean), ``"one"`` (single value),
            ``"list"`` (``nargs="+"``) or ``"pair"``
            (``<key>=<value>``, repeatable).
        parsers: Files whose argparse parser accepts it. ``"parent"`` means the
            shared cross-cutting parent in ``utils/cli_common.py``, i.e. every
            entry point.
    """

    name: str
    rows: Tuple[int, ...]
    choices: Tuple[str, ...]
    value_form: str
    parsers: Tuple[str, ...]


HUMAN_TASK_ACTIONS: Tuple[str, ...] = ("done", "skip")
HUMAN_TASK_STRATEGIES: Tuple[str, ...] = (
    "pause-and-ask", "skip-continue", "skip-dependents",
)
#: The strategy an unattended run resolves row 8 to: an unattended run cannot
#: perform manual steps, so code tasks depending on them cannot be assumed valid.
UNATTENDED_HUMAN_TASK_STRATEGY = "skip-dependents"

DECISION_FLAGS: Tuple[FlagSpec, ...] = (
    FlagSpec("--models", (2,), (), "list",
             ("review_plan_orchestrator.py", "review_tasks_orchestrator.py",
              "code_review_orchestrator.py", "ask_orchestrator.py")),
    FlagSpec("--quick", (2,), (), "none",
             ("review_plan_orchestrator.py", "review_tasks_orchestrator.py",
              "code_review_orchestrator.py", "ask_orchestrator.py")),
    FlagSpec("--force", (3,), (), "none",
             ("review_plan_orchestrator.py", "review_tasks_orchestrator.py",
              "code_review_orchestrator.py", "ask_orchestrator.py")),
    FlagSpec("--reaggregate", (3,), (), "none",
             ("review_plan_orchestrator.py", "review_tasks_orchestrator.py",
              "code_review_orchestrator.py")),
    FlagSpec("--rerun-all", (3,), (), "none",
             ("review_plan_orchestrator.py", "review_tasks_orchestrator.py",
              "code_review_orchestrator.py", "ask_orchestrator.py")),
    FlagSpec("--consolidate", (4,), (), "none",
             ("review_plan_orchestrator.py",)),
    FlagSpec("--skip-prerequisites", (5, 6), (), "none", ("parent",)),
    FlagSpec("--no-confirm", (7, 10, 11), (), "none",
             ("implement_orchestrator.py", "apply_suggestions_orchestrator.py",
              "apply_task_suggestions_orchestrator.py",
              "apply_code_fixes_orchestrator.py")),
    FlagSpec("--force-strategy", (8,), HUMAN_TASK_STRATEGIES, "one",
             ("implement_orchestrator.py",)),
    FlagSpec("--human-task", (9,), HUMAN_TASK_ACTIONS, "pair",
             ("implement_orchestrator.py",)),
    FlagSpec("--claude-decide", (10,), (), "none",
             ("apply_suggestions_orchestrator.py",
              "apply_task_suggestions_orchestrator.py",
              "apply_code_fixes_orchestrator.py")),
    FlagSpec("--review-tasks", (12,), (), "none", ("multi_llm.py",)),
    FlagSpec("--resume", (14,), (), "none",
             ("implement_orchestrator.py", "apply_suggestions_orchestrator.py",
              "apply_task_suggestions_orchestrator.py",
              "apply_code_fixes_orchestrator.py")),
)

_FLAGS_BY_NAME: Dict[str, FlagSpec] = {f.name: f for f in DECISION_FLAGS}


def flags_for_row(number: int) -> Tuple[FlagSpec, ...]:
    """Every decision flag that answers D6 row *number*."""
    return tuple(f for f in DECISION_FLAGS if number in f.rows)


def flag_spec(name: str) -> Optional[FlagSpec]:
    return _FLAGS_BY_NAME.get(name)


# The valued flags the argv-level pre-check validates. Boolean flags cannot
# arrive "empty", so they need no check beyond argparse's own.
_VALIDATED: Tuple[str, ...] = ("--models", "--force-strategy", "--human-task")


def _values_after(argv: Sequence[str], index: int, *, greedy: bool) -> List[str]:
    """The value tokens that belong to the flag at *index*."""
    values: List[str] = []
    cursor = index + 1
    while cursor < len(argv):
        token = argv[cursor]
        # An empty string is a *value* (the row-15 case), never a flag.
        if token.startswith("-") and token != "-" and token != "":
            break
        values.append(token)
        cursor += 1
        if not greedy:
            break
    return values


def validate_decision_flags(argv: Sequence[str]) -> None:
    """Row 15: an empty or unrecognized decision value is exit 3 at parse time.

    Runs on raw argv *before* any parser, so ``--models ""``,
    ``--human-task T001=`` and an unknown ``--force-strategy`` value all fail the
    same way — with exit 3 rather than argparse's exit 2, which a host would read
    as "nothing to do". ``choices`` on each parser is the second layer, not the
    only one.

    Raises:
        NeedsHuman: naming the flag, the offending value and the legal values.
    """
    argv = list(argv)
    for index, token in enumerate(argv):
        head, _, inline = token.partition("=")
        if head not in _VALIDATED:
            continue
        spec = _FLAGS_BY_NAME[head]
        if token.startswith(head + "="):
            values = [inline]
        else:
            values = _values_after(argv, index, greedy=spec.value_form == "list")
        _validate_values(spec, values)


def _validate_values(spec: FlagSpec, values: Sequence[str]) -> None:
    site = row(spec.rows[0]).site_id
    if not values:
        raise NeedsHuman(
            f"ERROR: {spec.name} was passed with no value. "
            f"{_legal_values(spec)}\n"
            "       A decision flag that arrives empty is never a default "
            "(D6 row 15).",
            site=site,
        )
    for value in values:
        if not value.strip():
            raise NeedsHuman(
                f"ERROR: {spec.name} was passed an empty value. "
                f"{_legal_values(spec)}\n"
                "       A decision flag that arrives empty is never a default "
                "(D6 row 15).",
                site=site,
            )
        if spec.value_form == "pair":
            parse_human_task_decision(value)
        elif spec.choices and value not in spec.choices:
            raise NeedsHuman(
                f"ERROR: {spec.name} got unrecognized value {value!r}. "
                f"{_legal_values(spec)}",
                site=site,
            )


def _legal_values(spec: FlagSpec) -> str:
    if spec.value_form == "pair":
        return f"Expected <id>={'|'.join(spec.choices)}."
    if spec.choices:
        return f"Legal values: {', '.join(spec.choices)}."
    if spec.value_form == "list":
        return "Expected one or more non-empty values."
    return "Expected a non-empty value."


def parse_human_task_decision(value: str) -> Tuple[str, str]:
    """Parse one ``--human-task <task_id>=done|skip`` value (row 9).

    The only genuinely new answer channel in D5a, because it is per-task and
    mid-run: the pause becomes a process boundary (exit, print the two
    re-invocations, be re-invoked) instead of a blocked stdin.
    """
    task_id, sep, action = value.partition("=")
    site = row(9).site_id
    if not sep or not task_id.strip() or not action.strip():
        raise NeedsHuman(
            f"ERROR: --human-task expects <task_id>=done|skip, got {value!r}.\n"
            "       An empty or malformed decision value is never a default "
            "(D6 row 15).",
            site=site,
        )
    if action not in HUMAN_TASK_ACTIONS:
        raise NeedsHuman(
            f"ERROR: --human-task {task_id}={action!r} is not a legal action. "
            f"Legal values: {', '.join(HUMAN_TASK_ACTIONS)}.",
            site=site,
        )
    return task_id.strip(), action


def parse_human_task_decisions(values: Optional[Iterable[str]]) -> Dict[str, str]:
    """Every ``--human-task`` value as ``{task_id: action}``.

    A repeated task id with a *different* action is a contradiction, not a
    precedence rule, so it is exit 3 like every other ambiguous answer.
    """
    decisions: Dict[str, str] = {}
    for value in values or ():
        task_id, action = parse_human_task_decision(value)
        previous = decisions.get(task_id)
        if previous is not None and previous != action:
            raise NeedsHuman(
                f"ERROR: --human-task {task_id} was given twice with different "
                f"actions ({previous!r} and {action!r}). Pass exactly one.",
                site=row(9).site_id,
            )
        decisions[task_id] = action
    return decisions


# ---------------------------------------------------------------------------
# Row 1 / D6c layer 3 — the uv backstop
# ---------------------------------------------------------------------------

#: Where the row-1 wording lives. Interpolated into every rendered shim by
#: ``install-host`` (D7) and read here, so shim and Python can never disagree
#: about how to install uv.
UV_MISSING_FILE = os.path.join("hosts", "uv-missing.txt")

#: Explicit override, also what the hermetic tests point at a stub. Same name
#: ``utils/detach.py`` uses for the worker re-exec, so one env var covers both.
UV_ENV = "MULTI_LLM_UV"

#: Probed in order, after ``PATH``. The Windows ``.exe`` variants are what the
#: official installer creates under Git Bash.
UV_CANDIDATES: Tuple[str, ...] = (
    "~/.local/bin/uv",
    "~/.local/bin/uv.exe",
    "~/.cargo/bin/uv",
    "~/.cargo/bin/uv.exe",
    "/opt/homebrew/bin/uv",
    "/usr/local/bin/uv",
    "/opt/local/bin/uv",
)


def find_uv() -> Optional[str]:
    """The absolute path of a usable ``uv``, or ``None``.

    ``$MULTI_LLM_UV`` → ``PATH`` → :data:`UV_CANDIDATES`. Never installs
    anything, and never falls back to bare ``python``: the skill's dependencies
    are pinned in ``uv.lock`` and resolved by ``uv run`` from ``pyproject.toml``,
    so a bare interpreter turns a clear "install uv" into a
    ``ModuleNotFoundError`` or, worse, a silently wrong dependency version
    mid-fan-out (D6c).
    """
    override = os.environ.get(UV_ENV)
    if override and os.path.isfile(override):
        return os.path.abspath(override)
    found = shutil.which("uv")
    if found:
        return os.path.abspath(found)
    for candidate in UV_CANDIDATES:
        expanded = os.path.expanduser(candidate)
        if os.path.isfile(expanded):
            return expanded
    return None


def uv_missing_text(base: Optional[str] = None) -> str:
    """The row-1 message, read from ``hosts/uv-missing.txt``."""
    path = os.path.join(str(base) if base else str(skill_dir()), UV_MISSING_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().rstrip("\n")
    except OSError:
        # The message file is the single source; a partial install is the only
        # way to get here, so say so instead of duplicating the text.
        return (
            "[UV_MISSING] uv is required and was not found, and this install is "
            f"missing {path}.\n"
            "Install uv (https://docs.astral.sh/uv/getting-started/installation/) "
            "and re-run. There is no bare-`python` fallback."
        )


def require_uv() -> None:
    """Row 1, the run-time backstop (D6c layer 3).

    ``install-host --check`` (install time) and the shim preflight (invocation
    time) cover the common cases; this covers uv removed after install, a host
    invoking ``multi_llm.py`` directly, and a shim edited by hand. Same message,
    same exit 3.
    """
    if find_uv() is None:
        raise NeedsHuman(uv_missing_text(), site=row(1).site_id)


# ---------------------------------------------------------------------------
# Recording an auto-resolution
# ---------------------------------------------------------------------------

#: Prefix of the one stdout line every auto-resolution prints. Greppable in a
#: detached log and in a foreground transcript alike.
RESOLVED_MARKER = "[AUTO_RESOLVED]"


def record_resolution(
    phase_dir: Optional[str],
    site_id: str,
    decision: str,
    *,
    quiet: bool = False,
) -> None:
    """Announce and record one auto-resolved decision point.

    An auto-selection nobody can see afterwards is indistinguishable from a
    silent degradation, so this always prints ``[AUTO_RESOLVED] <site>:
    <decision>`` and, when a detached run owns *phase_dir*, appends the entry to
    ``{phase}/.status.json`` ``run.resolutions`` (D4c) — which is what
    ``status --view run`` reports. A foreground run's ``.status.json`` still gains
    no new keys, so "with ``--detach`` unset, behavior is unchanged" stays
    literally true; the marker line is the foreground record.
    """
    if site_id not in _ROWS_BY_SITE:  # pragma: no cover - guarded by tests
        raise KeyError(f"unknown D6 site id: {site_id!r}")
    if not quiet:
        print(f"{RESOLVED_MARKER} {site_id}: {decision}")
    if not phase_dir:
        return
    try:
        from . import detach as _detach  # local: keep `status` sub-second
    except ImportError:  # pragma: no cover - direct script runs
        import detach as _detach  # type: ignore
    try:
        _detach.record_resolution(str(phase_dir), site_id, decision)
    except Exception:  # pragma: no cover - recording must never break a phase
        pass


# ---------------------------------------------------------------------------
# Fail-closed helpers
# ---------------------------------------------------------------------------

def mark_needs_human(mode: Optional[str] = None) -> None:
    """Set ``MULTI_LLM_RESULT.state`` to ``needs-human`` for this invocation.

    A no-op under direct invocation (no dispatcher run is active), which is why
    it is safe to call from an orchestrator.
    """
    ctx = active_result()
    if ctx is None:
        return
    ctx.state = "needs-human"
    if mode:
        ctx.mode = mode


def fail_closed(
    message: str,
    *,
    site_id: Optional[str] = None,
    mode: Optional[str] = None,
    stream=None,
) -> int:
    """Print a fail-closed message, mark the result line, return exit 3."""
    out = stream if stream is not None else sys.stderr
    print(message, file=out)
    mark_needs_human(mode)
    if site_id and site_id not in _ROWS_BY_SITE:  # pragma: no cover
        raise KeyError(f"unknown D6 site id: {site_id!r}")
    return EXIT_NEEDS_HUMAN


# ---------------------------------------------------------------------------
# Row 3 — partial results are usable output, not a question
# ---------------------------------------------------------------------------

PARTIAL_MARKER = "[PARTIAL_RESULTS:"


def report_partial_results(
    *,
    completed: Sequence[str],
    failed: Sequence[str],
    phase_dir: Optional[str] = None,
    retry_command: Optional[str] = None,
    unattended: bool = False,
) -> int:
    """Row 3's disposition: resume once, then reaggregate what exists.

    Returns the exit code:

    * some models produced results → ``[PARTIAL_RESULTS: a,b]`` on stdout,
      ``MULTI_LLM_RESULT.state: "partial"``, **exit 0**. A partial aggregate is
      usable output, so this is deliberately never exit 3.
    * *no* model produced a result → **exit 3**: there is nothing to aggregate,
      and the retry command is named.

    The marker and the state are unconditional (they describe the *outcome*, which
    a host has to be able to read either way); the ``[AUTO_RESOLVED]`` line is
    recorded only under ``unattended``, because attended runs may still render the
    partial/resume/rerun gate and claiming it was auto-resolved would be a lie.
    """
    failed_list = [str(m) for m in failed]
    if not completed:
        message = (
            "ERROR: no model produced a result, so there is nothing to "
            "aggregate.\n"
            f"       Failed models: {', '.join(failed_list) or 'none recorded'}"
        )
        if retry_command:
            message += f"\n       Retry with: {retry_command}"
        return fail_closed(message, site_id=row(3).site_id)
    if failed_list:
        print(f"{PARTIAL_MARKER} {','.join(failed_list)}]")
        ctx = active_result()
        if ctx is not None:
            ctx.state = "partial"
        if unattended:
            record_resolution(
                phase_dir,
                row(3).site_id,
                f"reaggregated with partial results; failed: {','.join(failed_list)}",
            )
    return EXIT_OK
