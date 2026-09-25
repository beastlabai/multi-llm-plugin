#!/usr/bin/env python3
"""``multi_llm.py status <plan>`` — two views, two files, one command word.

This *replaces* the instruction-only ``--status`` mode rather than sitting beside
it, and extends the surface that mode already exposed instead of inventing one.
Three distinct things existed before, and the distinction is load-bearing:

1. ``--status``, a **mode word** with no Python behind it, whose markdown wrapper
   shelled out to ``check_workflow_prerequisites.py --status`` →
   ``get_workflow_status``, printing a phase map built from **``state.json``**.
   That function survives here as the *workflow view's implementation*, unchanged
   and still directly callable by its existing tests: ``status <plan> --view
   workflow --json`` is byte-identical to what it prints today.
2. ``{phase}/.status.json``, the **orchestrator checkpoint**, whose top-level
   ``state`` is the in-phase machine and whose ``run`` object (added by
   ``--detach``) is the process machine. That is the *run view*.
3. ``state.json`` itself, the durable per-plan workflow record behind (1).

**Two questions, one word.** ``workflow`` answers *what phase comes next?* and
``run`` answers *is the run I launched still alive?*. Giving the heartbeat its own
command word was rejected: the two are read at the same moments (a poller checks
the run, then reads the workflow view to choose the next phase), a second word
would need its own plan/``--phase``/ask selectors and its own exit-code table, and
the pair would drift the first time one gained a selector the other lacked. What a
separate word buys — no ambiguity about which state you are being shown — is
bought instead by ``--view``: no output ever blends the two, and the three state
fields (workflow phase completion, in-phase ``state``, process ``run.state``) are
printed on three separate lines and never collapsed.

**Neither view is authoritative over the other; there is no precedence rule.**
Liveness comes only from the run view, phase completion only from the workflow
view. The one case that looks like a conflict — a finished run under a phase
``state.json`` still calls ``pending`` — is the normal post-run condition, and it
is reported as such rather than resolved by ranking one file over the other.

**This command takes no lock and writes nothing** (bar one exception, below), so
it works *while* another run owns the phase dir — which is the whole point of a
poll loop. The exception is the deliberate close-out of a run whose heartbeat
expired and whose process is provably gone: without it a phase dir would sit at
``running`` forever after a reboot. ``--abort`` and ``--sweep`` are verbs, and they
say so.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import (
    cli_common,
    detach,
    host_capabilities,
    metrics,
    process_control,
    status_file,
)
from .cli_common import (
    EXIT_ERROR,
    EXIT_NEEDS_HUMAN,
    EXIT_NOTHING_TO_DO,
    EXIT_OK,
    STATE_DIED,
    STATE_RUNNING,
    build_parent_parser,
    dispatcher_command,
    record_artifact,
    record_consolidation_recommended,
    record_mode_phase,
    record_validation_context,
    record_next,
    record_notice,
    record_progress,
    record_progress_lines,
    record_state,
)
from .progress_tracker import (
    STATE_DONE,
    STATE_FAILED,
    STATE_QUEUED,
    STATE_RETRYING,
    TERMINAL_STATES,
    entry_budget,
    entry_elapsed_s,
    entry_stagger_remaining_s,
    format_budget,
    format_elapsed,
    one_line_ascii,
    render_progress_line,
    render_roster_line,
)
from .output_handler import (
    PHASE_DIR_NAMES,
    PHASE_DIRECTORIES,
    output_dir_path,
    phase_dir_path,
    slugify_question,
)
from .stream_bootstrap import bootstrap_streams
from .validation_context import (
    pick_worst,
    validation_context_notice,
    validation_context_text,
)

#: ``--wait``'s default. Deliberately under the tightest surveyed host shell cap
#: (OpenCode's ~120 s): it compresses the poll loop into one turn, it never
#: replaces the terminal-state check, and it is never emitted for a host whose
#: shell cap is below its own timeout.
DEFAULT_WAIT_TIMEOUT_S = 90.0

#: How often ``--wait`` re-reads the status file. The heartbeat is 10 s, so a
#: faster cadence learns nothing; 5 s keeps the returned view fresh.
WAIT_POLL_INTERVAL_S = 5.0

#: How far *below* a host's ``timeout_ceiling_s`` a resolved ``--wait`` is
#: clamped. This is the number that explains :data:`DEFAULT_WAIT_TIMEOUT_S`:
#: OpenCode's ceiling is 120 s and 120 - 30 = 90.
#:
#: The margin is **startup, not slack**. A poll spends process start,
#: ``load_config`` and the checkpoint read before ``--wait`` begins sleeping, and
#: ``_wait_for_terminal`` re-checks on a :data:`WAIT_POLL_INTERVAL_S` grid, so it
#: returns a little *after* its own deadline — clamping to the cap itself would
#: put the return on the wrong side of it. The same constant bounds the tool-call
#: ``timeout`` a Claude Code poll has to set (``interval + 30``).
#:
#: **Re-exported, not defined here** (F6, 2026-09-20). The comment this replaces
#: said the number was "named once here rather than twice so the two cannot
#: drift"; it was in fact typed out in six places — this constant, the ``native``
#: *run detached* sentence, ``SKILL.md``, ``references/host-capabilities.md`` and
#: four instruction cores — so "30" could have been moved here and left standing
#: on every surface an agent actually reads. It is owned by
#: :data:`host_capabilities.WAIT_CAP_MARGIN_S`, beside the ceilings it is
#: subtracted from; the verb sentence interpolates it (carrying ``SKILL.md`` and
#: the generated reference with it) and the four cores are pinned against it by
#: ``tests/test_instruction_portability.py``.
WAIT_CAP_MARGIN_S = host_capabilities.WAIT_CAP_MARGIN_S

VIEW_WORKFLOW = "workflow"
VIEW_RUN = "run"
VIEW_ALL = "all"

#: ``progress.step``'s two terminal values. ``progress_tracker.PhaseProgress``
#: owns the pair and names them in its own docstring: ``complete``, and the
#: host-driven ``validation_pending`` early return where the process exits and
#: the agent's own subagents take over. Anything else means the phase is still
#: working.
CHECKPOINT_TERMINAL_STEPS = frozenset({"complete", "validation_pending"})

#: The *phase* machine's terminal words — ``status_file``'s
#: ``models_running -> models_complete/grouping_complete -> validation_pending ->
#: complete`` ladder plus the ways a phase stops without reaching ``complete``
#: (``task_generation_pending``'s host hand-off, ``subagents_failed``, an
#: explicit failure or abort).
#:
#: **The ask lane ends on its own three words**, and they are not the review
#: lanes'. ``ask_orchestrator`` closes a finished fan-out with ``completed``
#: (every model answered), ``partial`` (some did) or ``failed`` (none did) — the
#: same words it reports as ``MULTI_LLM_RESULT.state``, which is why the lane is
#: widened into here rather than made to write ``complete``. Two of the three
#: were missing and ``failed`` was already present by coincidence, so
#: :func:`_checkpoint_in_flight` answered ``True`` for every ask that ever
#: *succeeded*: a finished foreground ask rendered ``running in foreground (5/5
#: models, no detached run record)`` for the rest of the phase dir's life, and
#: ``--wait --question …`` over a detached ask whose ``run.state`` was already
#: ``done`` spent the whole resolved ``poll_s`` on every poll before exiting 0.
#:
#: **A refusal close-out is an ending too**, and the two words it writes were
#: the same gap one lane further on. ``review_orchestrator_base``'s
#: ``_print_executor_refusal`` is the exit-3 block for a run that dispatched
#: nothing, and the one caller it has ``sys.exit``\ s on the very next line — so
#: ``runner_rejected`` (the runner spec itself was rejected) and
#: ``executor_refused`` (every other pre-dispatch refusal) are the last thing
#: that run will ever write. Missing from here, they read as "still going": the
#: head line called a refused phase ``running in foreground (…)`` for the rest
#: of the dir's life and ``--wait`` spent its whole resolved deadline before
#: exiting 2 with a ``next`` that polls it again — F2's loop, over a run that
#: cannot move. The document was already being read both ways at once:
#: :func:`_resume_shape` treats a top-level ``runner_rejected`` as the refusal's
#: durable classification and renders the repair off it.
#:
#: Everything else — including a word this set has never heard of — reads as
#: "still going", which is the safe direction: being wrong that way costs a
#: ``--wait`` its own timeout, never an unbounded loop.
CHECKPOINT_TERMINAL_STATES = frozenset({
    "complete", "completed", "partial", "validation_pending",
    "task_generation_pending", "subagents_failed", "failed", "aborted",
    "stopped", "runner_rejected", "executor_refused",
})


class _WaitTimeoutRefused(Exception):
    """``--wait`` cannot resolve a deadline it would be honest to wait on.

    Its one job is to reach :func:`main` as a *message*, so the refusal prints
    the ``ERROR:`` + exit 3 shape this codebase uses for "needs a human" instead
    of whatever the raising layer's own exception would have become. Raised for
    two unrelated causes with one disposition (F4/F7, 2026-09-20): a
    ``ConfigError`` from anywhere in the config document, and a ``--timeout``
    that is not a positive finite number. Both are a human's edit or a human's
    command line, neither says anything about the run being polled, and exit 1
    — which is what both used to produce — is read by the poll loop as "the run
    state is unreadable, stop and report".
    """


def _checkpoint_in_flight(view: Dict[str, Any]) -> bool:
    """Does the *checkpoint* say this phase is still working?

    Deliberately a different question from the ``run`` record's verdict, and
    asked of a different half of the file. ``run`` is the **process** machine and
    only a detached worker writes it; the top-level ``state`` and
    ``progress.step`` are the **phase** machine and every run writes them,
    foreground included. A phase dir is re-used across runs by design (the same
    ``{plan}/review-plan/`` every time), so the two halves can legitimately
    disagree — and the disagreement that matters is a *predecessor's* terminal
    ``run`` record sitting over a *successor's* live checkpoint.

    F1, 2026-09-20: nothing drops the old ``run`` record on the foreground path,
    so a host-background re-run over a dir an earlier ``--detach`` run used
    inherits ``run.state: done``. ``--wait`` read that as terminal and came back
    in **0.11 s** with exit 0, printing ``done`` directly above a table saying
    ``running``; the instruction then relays nothing and polls again, turning a
    120-300 s loop into a tight sequence of instant tool calls for the whole
    10-30 minute run. Re-running a phase is the common path, not an edge case.

    Evidence-based, never inference: a checkpoint carrying *neither* field is not
    "in flight", so a genuinely finished foreground checkpoint still returns at
    its own deadline with exit 2 (acceptance criterion 1) rather than waiting on
    something that was never there.
    """
    state = view.get("state")
    if isinstance(state, str) and state and state not in CHECKPOINT_TERMINAL_STATES:
        return True
    step = (view.get("progress") or {}).get("step")
    if isinstance(step, str) and step and step not in CHECKPOINT_TERMINAL_STEPS:
        return True
    return False


def _names_this_run(view: Dict[str, Any], run_id: Optional[str]) -> bool:
    """Did the caller pin this poll to *this view's own* ``run`` record?

    ``--run-id`` is the only selector that names a **run**. ``--phase`` and
    ``--question`` name a *directory*, and a phase dir is re-used run after run
    by design (``{plan}/review-plan/`` every time, one ask dir per question), so
    neither can say which run the checkpoint under it belongs to. This one can:
    the caller typed the id the record carries, so the terminal record in front
    of them is the run they asked about rather than a predecessor's.

    Prefix matching, exactly as :func:`_select` resolves the same flag — every
    printed banner truncates the id to eight characters, so a prefix is what a
    caller normally has to hand.
    """
    if not run_id:
        return False
    recorded = str((view.get("run") or {}).get("run_id") or "")
    return bool(recorded) and recorded.startswith(run_id)


# ---------------------------------------------------------------------------
# Phase-dir discovery (read-only: never creates a directory)
# ---------------------------------------------------------------------------

def _ask_dirs(plan_dir: Path) -> List[Path]:
    ask_root = plan_dir / "ask"
    if not ask_root.is_dir():
        return []
    return [
        child for child in sorted(ask_root.iterdir())
        if child.is_dir() and (child / status_file.STATUS_FILENAME).exists()
    ]


def discover_phase_dirs(
    plan_path: Path, *, phase: Optional[str] = None
) -> List[Path]:
    """Every phase dir under a plan that holds a ``.status.json``.

    There is deliberately no phase *auto-detection*: this lists what it found and
    a caller that wants one run names it (``--phase``/``--run-id``/``--question``).
    A shim's poll step is therefore determined by what that shim launched, never by
    directory shape.

    Only children named in ``output_handler.PHASE_DIR_NAMES`` are phase dirs
    (plan §7.4a): a hand-made backup that copied a ``.status.json`` showed up as
    a second, duplicate run. The skipped ones are reported by
    :func:`ignored_phase_dirs`, never silently dropped. Order stays
    alphabetical by directory name.
    """
    plan_dir = output_dir_path(plan_path)
    if not plan_dir.is_dir():
        return []
    if phase:
        if phase == "ask":
            return _ask_dirs(plan_dir)
        target = phase_dir_path(plan_path, phase)
        return [target] if (target / status_file.STATUS_FILENAME).exists() else []
    found: List[Path] = []
    for child in sorted(plan_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name not in PHASE_DIR_NAMES:
            continue
        if child.name == "ask":
            found.extend(_ask_dirs(plan_dir))
            continue
        if (child / status_file.STATUS_FILENAME).exists():
            found.append(child)
    return found


def ignored_phase_dirs(plan_path: Path) -> List[str]:
    """Children of the plan's output dir that hold a ``.status.json`` but are
    not phase dirs (not in ``PHASE_DIR_NAMES``), by name (plan §7.4a2).

    A child without a ``.status.json`` is not listed: it was never going to be
    mistaken for a run.
    """
    plan_dir = output_dir_path(plan_path)
    if not plan_dir.is_dir():
        return []
    return [
        child.name for child in sorted(plan_dir.iterdir())
        if child.is_dir()
        and child.name not in PHASE_DIR_NAMES
        and (child / status_file.STATUS_FILENAME).exists()
    ]


def _dedupe_run_ids(
    views: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Two views with one ``run.run_id`` are one run (plan §7.4b).

    Keeps the view whose directory is the run's own phase dir (its basename is
    the directory word of the view's ``phase``), else the first one listed, and
    returns the others as copies: ``{"dir", "run_id", "copy_of"}``. Views with no
    ``run_id`` (a foreground checkpoint, a ``full`` run's driven phase dirs) are
    never deduped.
    """
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for view in views:
        run_id = str((view.get("run") or {}).get("run_id") or "")
        if run_id:
            by_id.setdefault(run_id, []).append(view)
    dropped: Dict[int, Dict[str, Any]] = {}
    for run_id, group in by_id.items():
        if len(group) < 2:
            continue

        def _own(view: Dict[str, Any]) -> bool:
            phase = view.get("phase") or (view.get("run") or {}).get("mode")
            if not phase:
                return False
            return os.path.basename(view["phase_dir"]) == PHASE_DIRECTORIES.get(
                str(phase), str(phase)
            )

        keep = next((v for v in group if _own(v)), group[0])
        for view in group:
            if view is not keep:
                dropped[id(view)] = {
                    "dir": _view_dir_name(view),
                    "run_id": run_id,
                    "copy_of": _view_dir_name(keep),
                }
    kept = [v for v in views if id(v) not in dropped]
    copies = [dropped[id(v)] for v in views if id(v) in dropped]
    return kept, copies


def _view_dir_name(view: Dict[str, Any]) -> str:
    """A view's directory relative to the plan's output dir (``ask/<q>`` for ask)."""
    path = Path(view["phase_dir"])
    if path.parent.name == "ask":
        return f"ask/{path.name}"
    return path.name


def _ignored_text(ignored: Sequence[str], copies: Sequence[Dict[str, Any]]) -> List[str]:
    """The one-line reports for skipped non-phase dirs and duplicate-run copies."""
    lines: List[str] = []
    if ignored:
        lines.append(f"ignored (not a phase dir): {', '.join(ignored)}")
    for copy in copies:
        lines.append(
            f"ignored (copy of run {str(copy['run_id'])[:8]} in "
            f"{copy['copy_of']}): {copy['dir']}"
        )
    return lines


def _select(
    views: List[Dict[str, Any]],
    *,
    run_id: Optional[str],
    question: Optional[str],
    include_all: bool,
) -> List[Dict[str, Any]]:
    """Apply the selectors, then the ask listing rule."""
    if run_id:
        return [
            v for v in views
            if str(v["run"].get("run_id", "")).startswith(run_id)
        ]
    if question:
        slug = slugify_question(question)
        return [
            v for v in views
            if v.get("question") == question
            or os.path.basename(v["phase_dir"]).startswith(slug)
        ]
    if include_all:
        return views
    # Ask dirs multiply: report every non-terminal run plus the most recently
    # updated terminal one, so a poller sees what is live without a wall of
    # history. Non-ask phases are one dir each, so this is a no-op for them.
    ask_views = [v for v in views if _is_ask(v)]
    other = [v for v in views if not _is_ask(v)]
    live = [v for v in ask_views if v["verdict"] not in detach.TERMINAL_VERDICTS]
    terminal = [v for v in ask_views if v["verdict"] in detach.TERMINAL_VERDICTS]
    terminal.sort(key=lambda v: str(v.get("updated_at") or ""))
    return other + live + terminal[-1:]


def _is_ask(view: Dict[str, Any]) -> bool:
    return os.path.basename(os.path.dirname(view["phase_dir"])) == "ask"


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

def exit_code_for(views: Sequence[Dict[str, Any]]) -> int:
    """``0`` terminal-done, ``2`` non-terminal, ``3`` terminal-needs-human.

    The exit code *is* the loop condition, so a shim never parses JSON just to
    decide whether to keep going. With several runs in view the worst case wins in
    poll-loop order: anything still moving is ``2`` (come back), then needs-human,
    then done. ``2`` keeps its documented "nothing for the caller to do (yet)"
    meaning rather than inventing a status-only code — including the "no run
    recorded" case, which must never be ``0`` (a poll loop would read that as
    finished; the loop's own ceiling is what bounds it).
    """
    if not views:
        return EXIT_NOTHING_TO_DO
    if any(v["verdict"] not in detach.TERMINAL_VERDICTS for v in views):
        return EXIT_NOTHING_TO_DO
    for view in views:
        if view["verdict"] in (detach.VERDICT_FAILED, detach.VERDICT_ABORTED):
            return EXIT_NEEDS_HUMAN
        if view["run"].get("exit_code") not in (EXIT_OK, EXIT_NOTHING_TO_DO, None):
            return EXIT_NEEDS_HUMAN
    return EXIT_OK


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _workflow_payload(plan_path: Path) -> Dict[str, Any]:
    """Today's ``--status`` JSON, from today's implementation."""
    from .state_manager import StateManager

    try:
        import check_workflow_prerequisites as prereq  # type: ignore
    except ImportError:  # pragma: no cover - only when utils/ is sys.path[0]
        sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
        import check_workflow_prerequisites as prereq  # type: ignore

    state = StateManager(plan_path)
    return prereq.get_workflow_status(state, plan_path, output_dir_path(plan_path))


def _workflow_trailer_lines(payload: Dict[str, Any], *, prefix: str = "  ") -> List[str]:
    """The workflow view's ``suggested next``/``notice`` lines, alone.

    Split out of ``render_workflow_text`` so the terminal ``--view run`` carry-
    over (F7, 2026-09-16, below) renders the identical text instead of a second
    hand-typed copy that could drift from it — the whole point is that the two
    channels can never say something different about the same ``state.json``.
    """
    lines: List[str] = []
    if payload.get("suggested_next"):
        # `suggested_next` already skips every optional phase in its way
        # (`check_workflow_prerequisites.get_workflow_status`'s own
        # `OPTIONAL_PHASES` walk), so the bare word reads as "nothing else is
        # outstanding" — a weak model given only this line, and the tool's own
        # `next: null` beside it, asserted exactly that (F8, 2026-09-16). Name
        # the optional phases it stepped over: the ones still `pending` that
        # sit *before* it in phase order (`payload["phases"]` preserves that
        # order), not every pending optional phase regardless of position — an
        # optional phase further down the table has not been skipped for this
        # suggestion, it just has not been reached yet.
        pending_optional = []
        for phase, info in payload.get("phases", {}).items():
            if phase == payload["suggested_next"]:
                break
            if info.get("optional") and info.get("status") == "pending":
                pending_optional.append(phase)
        suffix = (
            f"  ({' / '.join(pending_optional)} optional, not run)"
            if pending_optional else ""
        )
        lines.append(f"{prefix}suggested next: {payload['suggested_next']}{suffix}")
    for notice in payload.get("notices", []) or []:
        lines.append(f"{prefix}notice: {notice}")
    return lines


def render_workflow_text(payload: Dict[str, Any]) -> List[str]:
    lines = ["workflow (state.json) — which phases are done:"]
    width = max((len(p) for p in payload.get("phases", {})), default=0)
    for phase, info in payload.get("phases", {}).items():
        extra = []
        if info.get("optional"):
            extra.append("optional")
        if info.get("reason"):
            extra.append(str(info["reason"]))
        if info.get("tasks_total"):
            extra.append(f"{info.get('tasks_done', 0)}/{info['tasks_total']} tasks")
        if info.get("timestamp"):
            extra.append(str(info["timestamp"]))
        suffix = f"  ({', '.join(extra)})" if extra else ""
        lines.append(f"  {phase.ljust(width)}  {info.get('status', '?')}{suffix}")
    lines.extend(_workflow_trailer_lines(payload))
    return lines


def render_run_text(
    views: Sequence[Dict[str, Any]],
    *,
    plan_path: Path,
    workflow: Optional[Dict[str, Any]],
    scope: Optional[str] = None,
) -> List[str]:
    lines = ["runs ({phase}/.status.json) — is a launched run still alive:"]
    if not views:
        lines.append(f"  no run recorded for {scope or 'this plan'}")
        return lines
    for view in views:
        run = view["run"]
        name = os.path.basename(view["phase_dir"])
        head = f"  {name}"
        if run.get("run_id"):
            head += f"  run {str(run['run_id'])[:8]}"
        head += f"  {_verdict_text(view)}"
        lines.append(head)
        if view.get("question"):
            lines.append(f"      question: {view['question']}")
        # Three state fields, three lines, never collapsed.
        lines.append(f"      run.state:      {_run_state_text(view)}")
        lines.append(f"      phase state:    {view.get('state') or '(none)'}")
        if workflow is not None:
            phase_name = view.get("phase") or name
            wf = (workflow.get("phases") or {}).get(phase_name, {})
            wf_status = wf.get("status", "not tracked")
            # N7, 2026-09-16: `phases_completed` records that a phase finished
            # once — a `--resume` can reopen it (implement picking up tasks a
            # later review sent back), and a live run under an already-
            # `completed` phase is the resulting, entirely valid state. Printed
            # bare, "completed" sits beside `run.state: running` and is the one
            # word on screen a reader is likely to take at face value and stop
            # watching, so a live run under a completed phase says so rather
            # than reading like a stale cache.
            # F4, 2026-09-17: `stale` covers both "slow to beat" and "provably
            # gone but not yet closed out"; only the first is live. A dead run
            # printed "completed (re-running since …)" two lines under its own
            # `died (pid N gone, …)`, which reads as the opposite of the truth.
            live_verdict = view["verdict"] in (
                detach.VERDICT_STARTING, detach.VERDICT_RUNNING,
            ) or (view["verdict"] == detach.VERDICT_STALE and not _died(view))
            if wf_status == "completed" and live_verdict:
                started = run.get("started_at")
                suffix = f" (re-running since {started})" if started else " (re-running)"
                lines.append(f"      workflow phase: completed{suffix}")
            else:
                lines.append(f"      workflow phase: {wf_status}")
            if (
                view["verdict"] == detach.VERDICT_DONE
                and wf_status == "pending"
            ):
                lines.append(
                    "      note: run complete; workflow phase not yet marked "
                    "completed"
                )
                advance = view.get("reaggregate_command") or dispatcher_command(
                    "status", str(plan_path), "--view", "workflow"
                )
                lines.append(f"      advance it with: {advance}")
        if view.get("progress"):
            lines.append(f"      progress:       {_progress_text(view['progress'])}")
            # Between the counts and the table, and INSIDE this guard by
            # decision rather than by accident: the delta is computed from
            # `progress.models` (see `_attach_finished_during_wait`), so a
            # non-empty one proves a non-empty `progress` and this guard can
            # never suppress a line that had something to say. Putting it
            # outside would be dead code that reads like a hedge. Its place in
            # the block is the reading order the answer needs — "3/5 done" then
            # "which three", then the roster they sit in.
            just_finished = _just_finished_text(view)
            if just_finished:
                lines.append(f"      {JUST_FINISHED_LABEL} {just_finished}")
            # ADDITIVE, and deliberately under the same guard: a models-only
            # table that *replaced* the line above would hide the validation and
            # write-phase mirrors, which are the only progress a dispatch-only
            # phase such as `generate-tasks` records. A view whose `progress`
            # carries no `models` map therefore renders exactly as it did before
            # this table existed.
            lines.extend(_model_table_lines(view))
        if run.get("host"):
            # The host that *launched* the run, which is not necessarily the one
            # reading this: without it nothing on screen said whose subagent
            # runner the block below is about (measured 2026-09-16, an opencode
            # run polled from Claude Code).
            lines.append(f"      host:           {run['host']}")
        if run.get("pid"):
            # The identity probe only runs for a non-terminal record — a finished
            # run's pid is history, and probing it would invite exactly the
            # recycled-pid confusion the probe exists to prevent.
            identity = view.get("identity")
            lines.append(
                f"      pid {run['pid']}"
                + (f" ({identity})" if identity else " (recorded)")
                + (f", cwd {run.get('cwd')}" if run.get("cwd") else "")
            )
        if run.get("log_path"):
            lines.append(f"      log: {run['log_path']}")
        if view["output_files"]:
            lines.append("      output_files:")
            for key in sorted(view["output_files"]):
                lines.append(f"        {key}: {view['output_files'][key]}")
        consolidation = _consolidation_text(view.get("consolidation_recommended"))
        if consolidation:
            lines.append(f"      {consolidation}")
        validation_context_line = _validation_context_text(
            view.get("validation_context")
        )
        if validation_context_line:
            lines.append(f"      {validation_context_line}")
        # F4: recorded only for a finished code-review run — `high_priority_groups`
        # is written once, at the very end of main(), so a mid-run poll has
        # nothing to render yet, and the key exists on no other phase.
        if view.get("phase") == "code-review" and view["verdict"] == detach.VERDICT_DONE:
            high_priority = _high_priority_text(view.get("high_priority_groups"))
            if high_priority:
                lines.append(f"      {high_priority}")
        if run.get("resolutions"):
            lines.append("      auto-resolved while you were gone:")
            for entry in run["resolutions"]:
                lines.append(
                    f"        {entry.get('site')}: {entry.get('decision')} "
                    f"({entry.get('at')})"
                )
        if _died(view):
            # The exit-3 block a terminal failure gets, one heartbeat window
            # early: the run is over and no amount of polling will change that,
            # so the resume command is handed over now rather than after the
            # close-out. `reason` carries what the probe saw.
            lines.append(f"      {view.get('reason')}")
            lines.extend(_resume_lines(view, plan_path))
        if view["verdict"] in (detach.VERDICT_FAILED, detach.VERDICT_ABORTED):
            lines.append(f"      exit_code: {run.get('exit_code')}")
            lines.append(f"      error:     {run.get('error')}")
            tail = detach._log_tail(run.get("log_path") or "", 20)
            if tail:
                lines.append("      last log lines:")
                lines.append(tail)
            lines.extend(_resume_lines(view, plan_path))
    return lines


def _result_state(view: Dict[str, Any]) -> Optional[str]:
    """The run's carried result word (``partial``), or ``None``.

    Only a finished run that exited 0 carries one: ``run.result_state`` is
    written by the detached worker's close-out, bounded by
    ``detach.carried_result_state`` there and again here, so a missing, ``null``
    or hand-edited value reads as absent and a clean exit-0 run stays ``done``.
    """
    run = view.get("run") or {}
    if view.get("verdict") != detach.VERDICT_DONE:
        return None
    if run.get("exit_code") != EXIT_OK:
        return None
    return detach.carried_result_state(run.get("result_state"))


def _failed_model_specs(view: Dict[str, Any]) -> List[str]:
    """Which slots failed, for the text view's ``partial`` note.

    ``models_failed`` is the checkpoint's own record of the fan-out's losses
    (read from the file: the run view does not carry it); a phase that keeps
    no such key falls back to the ``failed`` rows of ``progress.models``.
    """
    try:
        failed = status_file.read_status(view["phase_dir"]).get("models_failed")
    except Exception:  # advisory text only; never fail the poll over it
        failed = None
    if isinstance(failed, dict) and failed:
        return sorted(str(spec) for spec in failed)
    if isinstance(failed, list) and failed:
        return [str(spec) for spec in failed]
    models = (view.get("progress") or {}).get("models")
    if isinstance(models, dict):
        return sorted(
            str(spec) for spec, entry in models.items()
            if isinstance(entry, dict) and entry.get("state") == STATE_FAILED
        )
    return []


def _run_state_text(view: Dict[str, Any]) -> str:
    """``run.state``, annotated when the run's result was ``partial`` (plan §5).

    ``run.state`` keeps its process meaning (``done``); the note says what the
    exit code could not: ``done (partial — failed: codex:gpt-6-astra:xhigh)``.
    """
    word = (view.get("run") or {}).get("state") or "(none)"
    result = _result_state(view)
    if result is None:
        return str(word)
    failed = _failed_model_specs(view)
    detail = f" — failed: {', '.join(failed)}" if failed else ""
    return f"{word} ({result}{detail})"


def _died(view: Dict[str, Any]) -> bool:
    """The run is non-terminal on paper and its process is provably gone.

    ``classify_run`` calls this ``stale`` and closes it out as
    ``failed(heartbeat_timeout)`` once the beat expires, which is right — but
    for up to ``HEARTBEAT_STALE_AFTER_S`` the only honest word for a pid the
    identity probe says is dead is "died", not "running (stale)". F2,
    2026-09-16: a write subagent's ``killall -9 … python3`` SIGKILLed the
    detached worker that dispatched it, and every poll for the next minute
    reported the dead run as running.
    """
    return (
        view.get("verdict") == detach.VERDICT_STALE
        and view.get("identity") == process_control.DEAD
    )


def _verdict_text(view: Dict[str, Any]) -> str:
    verdict = view["verdict"]
    age = view.get("heartbeat_age_s")
    if _died(view):
        pid = view["run"].get("pid")
        beat = f"last beat {age:.0f}s ago" if age is not None else "never beat"
        return f"died (pid {pid} gone, {beat})"
    if verdict == detach.VERDICT_STALE:
        return (
            f"running (stale, last beat {age:.0f}s ago)" if age is not None
            else "running (stale, never beat)"
        )
    if verdict == detach.VERDICT_RUNNING and age is not None:
        return f"running (last beat {age:.0f}s ago)"
    if verdict == detach.VERDICT_FAILED and view.get("reason") == "heartbeat_timeout":
        return "failed (heartbeat_timeout — the worker died without recording an outcome)"
    if verdict == detach.VERDICT_NO_RUN:
        # "Checkpoint but no run record" stopped being an oddity the moment a
        # foreground run started writing `progress`: it is the normal Claude
        # Code state, because that host is told never to pass `--detach` and
        # only the detached worker ever calls `claim_run`. The bare word
        # `no_run`, printed beside `run.state: (none)`, reads as "nothing
        # happened here" for a phase that is at that moment dispatching eight
        # models.
        #
        # ONLY THE TEXT CHANGES. The verdict constant stays `no_run`, so
        # `TERMINAL_VERDICTS`, `exit_code_for`'s exit 2 and F1's rule that a
        # `no_run` phase is never the `next` poll target are all untouched —
        # making it terminal is explicitly out of scope, and `status --wait`
        # terminating on a foreground checkpoint depends on it staying
        # non-terminal.
        #
        # F3, 2026-09-20: the gate was `if progress:` alone, so the word
        # "running" survived the phase it described. A finished foreground
        # checkpoint — `state: complete`, `progress.step: complete` — still
        # printed `running in foreground (2/2 models, no detached run record)`
        # with `phase state: complete` two lines below it, and rule 1 tells the
        # agent to relay that head line *verbatim*. Every completed Claude Code
        # phase, and every historical ask dir a bare `status` lists, reported
        # itself as running forever. The tense now comes from the checkpoint,
        # which is the only half of the file a foreground run writes.
        #
        # "finished" claims exactly one thing — this phase is no longer
        # working — and never that it succeeded: how it ended is the
        # `phase state:` line right below, which prints `complete`,
        # `subagents_failed` or whatever the phase wrote.
        progress = view.get("progress") or {}
        if progress:
            tense = "running" if _checkpoint_in_flight(view) else "finished"
            total = progress.get("models_total")
            if isinstance(total, int) and not isinstance(total, bool) and total:
                done = progress.get("models_done", 0)
                return (
                    f"{tense} in foreground ({done}/{total} models, no detached "
                    "run record)"
                )
            return f"{tense} in foreground (no detached run record)"
    return str(verdict)


def _consolidation_text(recommendation: Any) -> Optional[str]:
    """The optional "you may want to consolidate" line, or ``None``.

    The `[CONSOLIDATION_RECOMMENDED]` marker is a log line, and nobody polling a
    detached run reads the log — so this one sentence is where a host learns the
    recommendation exists. It names the command and never *is* one: the phase is
    complete, and `--consolidate` is the caller's decision.
    """
    if not isinstance(recommendation, dict):
        return None
    groups = recommendation.get("valid_groups")
    threshold = recommendation.get("threshold")
    command = recommendation.get("command") or "--consolidate"
    if not isinstance(groups, int) or not isinstance(threshold, int):
        return None
    return (
        f"consolidation recommended: {groups} valid groups >= {threshold} "
        f"— run {command}"
    )


def _validation_context_text(obj: Any) -> Optional[str]:
    """The phase's ``validation_context`` line (plan §2.4), or ``None``.

    Rendered from the published object only — never recomputed — so the poll
    text, ``MULTI_LLM_RESULT`` and ``report.md`` say the same thing. A
    ``mode: path`` source reads "excerpt inline + full snapshot by path",
    never "inlined in full".
    """
    return validation_context_text(obj if isinstance(obj, dict) else None)


def _validation_context_notice(views: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The truncation sentence(s) over every view, in discovery order, or ``None``."""
    sentences: List[str] = []
    for view in views:
        notice = validation_context_notice(view.get("validation_context"))
        if notice and notice not in sentences:
            sentences.append(notice)
    return "; ".join(sentences) or None


def _high_priority_text(high_priority: Any) -> Optional[str]:
    """The one line naming a finished code-review's distinct HIGH groups.

    F4: `[HIGH_PRIORITY_ISSUES] <n>` is a log line, and a detached caller never
    reads the log — this is where a host polling `status` actually learns the
    single most actionable fact a code review produces, the same way
    `_consolidation_text` surfaces `[CONSOLIDATION_RECOMMENDED]`.
    """
    if not isinstance(high_priority, dict):
        return None
    count = high_priority.get("count")
    if not isinstance(count, int):
        return None
    return f"high-priority groups: {count}"


#: What ``_progress_text`` says when the checkpoint's ``progress`` object holds
#: nothing it knows how to render. A sentinel rather than an empty string
#: because the human view prints this line unconditionally and a blank after
#: ``progress:`` reads as a renderer that failed; :func:`_current_state_text`
#: reads it back to mean "there is nothing to show", which is the one caller
#: that has somewhere else to go.
NO_PROGRESS_RECORDED = "(none recorded)"


def _progress_text(progress: Dict[str, Any]) -> str:
    parts = []
    if progress.get("step"):
        parts.append(str(progress["step"]))
    if progress.get("models_total"):
        # `models_done` counts *finished slots* (done + failed), so it only ever
        # moves forwards and reaches `models_total` even on a lossy run — which
        # is what a live line needs. `models_succeeded` carries the other half:
        # before it existed, the post-fan-out restatement recomputed successes
        # into the same key and a run that lost a model ended on `models 3/4`,
        # a numerator walking backwards at the very end. Rendered together the
        # information survives as `8/8 done, 7 ok, 1 failed`.
        #
        # This one line is what Layer 2's rule 1 tells an agent to relay
        # verbatim, so its shape is load-bearing; the clause degrades to today's
        # bare `models 8/8` when `models_succeeded` is absent, which is every
        # checkpoint written before the key existed.
        done = progress.get("models_done", 0)
        clause = f"models {done}/{progress['models_total']}"
        succeeded = progress.get("models_succeeded")
        if isinstance(succeeded, int) and not isinstance(succeeded, bool):
            failed = done - succeeded if isinstance(done, int) else 0
            clause += f" done, {succeeded} ok, {max(failed, 0)} failed"
        parts.append(clause)
    if progress.get("tasks_total"):
        parts.append(f"tasks {progress.get('tasks_done', 0)}/{progress['tasks_total']}")
    if progress.get("batches_total"):
        parts.append(
            f"batch {progress.get('batch_index', 0)}/{progress['batches_total']}"
        )
    if progress.get("failed_task_ids"):
        parts.append(f"failed {', '.join(map(str, progress['failed_task_ids']))}")
    # The executor's mirrors (`subagent_executor._mirror_status` under
    # `subagents`, `write_batch_scheduler._mirror_status` under
    # `write_subagents`) are the only progress a dispatch-only phase such as
    # `generate-tasks` records — it never calls `report_progress(step=…)` before
    # the end — so a line that read only the step keys printed "(none recorded)"
    # beside a mirror that said `ok: 1`. Measured 2026-09-14.
    for key in ("subagents", "write_subagents"):
        parts.extend(_mirror_text(key, progress.get(key)))
    # D6 override (2026-09-17): a detached fan-out records the slots it ran
    # without the runner's own sandbox under `progress.sandbox_bypassed`, and
    # a host polling `status` is told so here — the `[SANDBOX_BYPASSED]` line
    # is in a log a detached caller never reads.
    bypassed = progress.get("sandbox_bypassed")
    if isinstance(bypassed, list) and bypassed:
        parts.append(f"unsandboxed {', '.join(map(str, bypassed))}")
    # Landlock rung (2026-09-22): a slot on a *contained* fallback rung is
    # recorded under `progress.sandbox_fallback` as `{spec: rung}` and rendered
    # by rung name — never as "unsandboxed", which is the one thing it was not.
    contained = progress.get("sandbox_fallback")
    if isinstance(contained, dict) and contained:
        by_rung: Dict[str, List[str]] = {}
        for spec, rung in contained.items():
            by_rung.setdefault(str(rung), []).append(str(spec))
        for rung in sorted(by_rung):
            parts.append(f"sandbox-fallback {rung} {', '.join(sorted(by_rung[rung]))}")
    return ", ".join(parts) or NO_PROGRESS_RECORDED


#: ``source`` values meaning "this model never dispatched on this run — its
#: result came off disk". There is deliberately no ``skipped`` *state*: the
#: terminal set stays ``done | failed`` so a resumed model counts in
#: ``models_done`` **and** ``models_succeeded`` (it has a usable result), and the
#: distinction lives in ``source``. This is the renderer reading it, which is
#: what ``source`` is for.
#:
#: ``salvage`` (``review_orchestrator_base.record_models_salvaged``) is
#: deliberately **not** in here: that slot dispatched, ran, failed to parse and
#: was repaired in-process, so it renders as the plain ``done`` it is.
SKIPPED_SOURCES = frozenset({"existing_file", "resume"})


def _one_line(text: str) -> str:
    """Collapse every run of whitespace, newlines included, into one space.

    Applied to the one field in the ``models`` map that carries provider-authored
    text: ``error``. Everything else in a row is a spec, a state word or a number
    this module produced. An adapter that puts a stderr excerpt or a wrapped
    provider message in there hands the renderers a string with ``\n`` in it, and
    both of them are *line* renderers whose output has a consumer that assumes it:
    Layer 2 rule 1 has the host agent relay ``just finished:`` **verbatim**, so a
    two-line value turns one rule-bound line into two, the second with no label on
    it, and the table's aligned columns come apart at the same time.

    One helper for both sites on purpose — folding it into only the newer
    renderer would leave the two spellings of one field disagreeing, which is the
    drift this file keeps testing against (2026-09-20).
    """
    return " ".join(str(text).split())


def _model_state_label(entry: Dict[str, Any]) -> str:
    """The word in the row's state column."""
    state = str(entry.get("state") or STATE_QUEUED)
    if state == STATE_DONE and str(entry.get("source") or "") in SKIPPED_SOURCES:
        return "skipped"
    return state


def _model_timing_text(entry: Dict[str, Any]) -> str:
    """``2m10s/30m``, or ``""`` when the entry has no clock to read.

    One read-time rule, and :func:`progress_tracker.entry_elapsed_s` owns it:
    ``ended_at`` present -> the stored ``elapsed_s`` (a terminal row is frozen,
    so a table rendered an hour later still shows the slot's real duration);
    absent -> ``now - started_at`` clamped at ``timeout_s``. Nothing stores a
    per-tick ``elapsed_s``, so this number is always fresh — including at
    ``progress.interval_s: 0``, where the ticker prints nothing at all.

    The denominator is the slot's **whole** budget, fixed for its life, never the
    shrinking slice a retry ladder hands back for the next attempt: ``2m10s/30m``
    against a shrinking denominator would overstate the budget left.

    **A clock that cannot be read renders as no clock, never as a crash** —
    and that covers the *denominator* too, which it did not until 2026-09-21:
    the ``try`` below wraps :func:`entry_elapsed_s` and nothing else, so a
    ``timeout_s: Infinity`` (a token ``json.loads`` accepts) sailed past it into
    ``format_budget`` and raised ``OverflowError`` out of the same command this
    paragraph promises will not crash. The budget now reads through
    :func:`progress_tracker.entry_budget`, the one guard the roster and the
    ticker also read, and an unrenderable one costs the denominator rather than
    the row.

    F5, 2026-09-20: ``detach._elapsed_since`` catches ``ValueError`` only, but
    ``datetime.now() - <aware datetime>`` raises ``TypeError``, so a single
    ``started_at: "2026-09-19T00:00:00Z"`` anywhere in the map took the *whole*
    ``status`` command down with a traceback and exit 1 — which rule 2 maps to
    "the run state is unreadable, stop and report", ending a poll loop over a
    perfectly healthy run. Everything else in the map already degrades (garbage
    strings, ints, ``None``, non-dict entries, a missing ``elapsed_s``); this is
    the one shape that did not. The `models` map is the least trustworthy part of
    the checkpoint — it is the only part written per slot, by whichever lane won
    the race — so the catch is deliberately broad and belongs on the *reader*:
    one malformed row costs its own column, never the command. The parser's own
    half of this (``heartbeat_at`` reads the same way) is ``detach``'s to harden.
    """
    try:
        elapsed = entry_elapsed_s(entry)
    except Exception:
        return ""
    if elapsed is None:
        return ""
    budget = entry_budget(entry)
    if budget is not None and budget > 0:
        return f"{format_elapsed(elapsed)}/{format_budget(budget)}"
    return format_elapsed(elapsed)


def _model_detail_text(entry: Dict[str, Any]) -> str:
    """The trailing clause: the attempt, the pending backoff, the failure reason.

    ``attempt`` has no denominator on purpose — the ladder keeps transport
    retries and rate-limit backoffs as two independent counters, so an entry can
    never honestly render ``2/3`` — and it is only worth a column once it is past
    the first. ``error`` is terminal-only, so a row carrying one has already
    stopped.
    """
    details: List[str] = []
    attempt = entry.get("attempt")
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt > 1:
        details.append(f"attempt {attempt}")
    if str(entry.get("state") or "") == STATE_RETRYING:
        retry_after = entry.get("retry_after_s")
        if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
            details.append(f"retry in {retry_after:.0f}s")
    # `starts in 4s`: the launch ramp, named. Slots dispatch
    # `index * PROVIDER_STAGGER_DELAY` apart, and the launch-time poll — taken at
    # second zero on purpose, for the roster — met every row but the first
    # reading `staggering`, which a reader took for the fan-out running its
    # models one at a time (2026-09-21). The state word is the lifecycle's and
    # stays; the clause says how long, in the shape `retry in 20s` already
    # has. Rounded *up*, so a slot 0.4 s from dispatch reads `1s`, never `0s`
    # — and `entry_stagger_remaining_s` is `None` once the stamp is past, so
    # a poll landing on the flip renders no clause rather than a countdown
    # that reached zero and stayed.
    remaining = entry_stagger_remaining_s(entry)
    if remaining is not None:
        details.append(f"starts in {math.ceil(remaining)}s")
    error = entry.get("error")
    if error:
        details.append(_one_line(error))
    return ", ".join(details)


def _model_row_specs(view: Dict[str, Any], models: Dict[str, Any]) -> List[str]:
    """Which specs get a row, in the order the run requested them.

    The tracker seeds every requested spec ``queued`` *before* it creates the
    tasks, so in the normal case the map is already complete and its own keys are
    the answer. The exception is the window between a run's launch and that seed
    — and on a re-run of a phase dir, what a poll lands in that window reads is
    the *predecessor's* map. Dropping the specs that are missing from it would
    make the table quietly disagree with the ``models N/M`` count one line above,
    so they are rendered ``queued`` with no timings instead.

    That makes the row count **at least** ``models_total``, not exactly it: any
    key the map holds that ``models_requested`` does not is appended rather than
    hidden, because the one way a map grows such a key is
    ``FanoutProgress._entry`` seeding a spec some lane transitioned without
    requesting — a wiring mistake, and hiding it in the one table that could show
    it would be the wrong trade. The two lists agreeing is the normal state: the
    three launch writes drop the predecessor's ``progress`` outright (F1,
    2026-09-20), so a map surviving next to a *different* run's
    ``models_requested`` is no longer a shape this can meet.

    The checkpoint's top-level ``models_requested`` is the only record of the
    canonical spec list, and ``classify_run``'s view does not carry it (it is a
    top-level key, not a ``progress`` one), so it is read here — on the one path
    that needs it, gated on the map actually being short. ``render_run_text``
    already reads a file on a branch of its own (``detach._log_tail``), so the
    precedent for a read in the renderer is in the same function.
    """
    specs = [str(spec) for spec in models]
    total = (view.get("progress") or {}).get("models_total")
    if not isinstance(total, int) or isinstance(total, bool) or total <= len(specs):
        return specs
    requested = view.get("models_requested")
    if not isinstance(requested, list):
        try:
            requested = status_file.read_status(
                str(view.get("phase_dir") or "")
            ).get("models_requested")
        except Exception:  # pragma: no cover - an unreadable dir renders the map
            requested = None
    if not isinstance(requested, list):
        return specs
    ordered: List[str] = []
    for raw in requested:
        spec = str(raw)
        if spec not in ordered:
            ordered.append(spec)
    for spec in specs:
        if spec not in ordered:
            ordered.append(spec)
    return ordered


def _model_table_lines(view: Dict[str, Any]) -> List[str]:
    """The per-model rows beneath the one-line ``progress:`` summary.

    Rendered for **every** verdict, ``no_run`` included: a foreground run is the
    normal Claude Code state and holds no ``run`` record at all, so a table
    behind a "live run" guard would be invisible on the one host this feature is
    written for.

    ASCII only. Every host redirects the orchestrator's stdout to a file and a
    poll's output is read back through whatever codec that file took, so a
    separator that survives as ``?`` is worse than one that was ASCII to begin
    with.

    **An absent map is not an absent roster** (2026-09-20). The child writes
    ``models_requested`` and calls ``report_progress(step="fanout",
    models_total=N)`` *before* it constructs the tracker that seeds the map, so
    for the first seconds of every fan-out the view knows the whole roster and
    this rendered nothing at all: the launch-time poll printed ``running in
    foreground (1/5 models, no detached run record)`` over ``models 1/5 done``
    and never said which CLI was running which model — the question the user
    asked for after watching a real run. An absent or non-dict map is therefore
    read as ``{}`` rather than taken as a reason to return, and
    :func:`_model_row_specs` answers from ``models_requested``.

    The *empty* answer is still the empty table, and that is load-bearing in the
    other direction: a dispatch-only phase such as ``generate-tasks`` records no
    ``models_total`` (its only progress is the executor's mirror), so
    ``_model_row_specs`` returns nothing for it and it grows no bare ``models:``
    header — which is what it did before this table existed and what
    ``test_a_progress_object_with_no_models_map_renders_as_it_always_did``
    pins.
    """
    progress = view.get("progress") or {}
    models = progress.get("models")
    if not isinstance(models, dict):
        models = {}
    specs = _model_row_specs(view, models)
    if not specs:
        return []
    rows: List[Tuple[str, str, str, str]] = []
    for spec in specs:
        entry = models.get(spec)
        entry = entry if isinstance(entry, dict) else {}
        rows.append((
            spec,
            _model_state_label(entry),
            _model_timing_text(entry),
            _model_detail_text(entry),
        ))
    spec_width = max(len(row[0]) for row in rows)
    label_width = max(len(row[1]) for row in rows)
    timing_width = max(len(row[2]) for row in rows)
    lines = ["      models:"]
    for spec, label, timing, detail in rows:
        cells = f"{spec.ljust(spec_width)}  {label.ljust(label_width)}"
        if timing_width:
            cells = f"{cells}  {timing.ljust(timing_width)}"
        if detail:
            cells = f"{cells}  {detail}"
        lines.append(f"        {cells.rstrip()}")
    return lines


#: The label on the line that names what crossed the line during the wait.
#:
#: One spelling, two readers: :func:`render_run_text` prints it indented under
#: the ``progress:`` line, and :func:`_progress_lines` prefixes the same
#: renderer's output with it for a host that shows the line on its own, where a
#: bare list of specs and durations would not say what it was a list *of*.
JUST_FINISHED_LABEL = "just finished:"


def _just_finished_text(view: Dict[str, Any]) -> str:
    """``codex:gpt-6-astra:xhigh (4m12s, done)``, or ``""`` when nothing did.

    Renders ``view["finished_during_wait"]``, which only ``--wait`` ever writes
    (see :func:`_attach_finished_during_wait`): the specs that crossed into a
    terminal state *while this very command was waiting*. That window is the
    whole claim — ``just`` is a tense, and a plain poll can prove a row is
    finished but not that it finished since anyone last looked.

    Why it exists at all: the ``progress:`` line one row above carries counts
    (``models 3/5 done, 3 ok, 0 failed``) and Layer 2 rule 1 forbids relaying
    anything else mid-run, so a user watching a fan-out was told *how many* had
    finished and never *which*. A spec is the CLI and the model — nothing has to
    be derived to answer it.

    Three shape decisions, all deliberate:

    * Clauses are joined with ``; `` and **not** ``, ``, because the clause's own
      interior is comma-separated (``4m12s, done``); a comma join would render
      two models as four indistinguishable fields.
    * The state word is :func:`_model_state_label`'s, so this line and the table
      under it use one vocabulary for one row: ``done``, ``failed``, or
      ``skipped`` for a resume that landed mid-wait. ``ok`` is deliberately not
      in it — that is the *counting* line's word for a slot that produced a
      usable result, and a row labelled ``ok`` here would sit directly above the
      same row labelled ``skipped`` in the table.
    * The elapsed is the frozen duration with **no** ``/budget`` denominator.
      Every table row carries one because a live row needs the comparison; a row
      that has stopped does not, and ``4m12s/30m`` invites reading the budget as
      time still available.

    One line and ASCII, through the **same** scrub the roster takes
    (:func:`progress_tracker.one_line_ascii`), applied to the whole composed
    line rather than to one field of it. That claim used to be made here and
    not kept: ``_one_line`` covered ``error`` alone, no ASCII scrub ran at all,
    so ``codex:gpt–5`` (en dash, out of a config file) and ``café`` went
    through, and a spec key carrying a newline put a second, unlabelled line
    into a renderer whose whole contract is one line — which on the OpenCode
    package route, where ``toastMessage`` joins the supplied lines with a
    newline, is a line forged in a display the user reads as the dispatcher's
    own. The two renderers that feed ``progress_lines`` now scrub identically;
    that is the point of the shared helper, since a spec reaches both of them
    by the same route.

    ``_one_line`` still runs on ``error`` here, where it also runs in the table
    (one spelling of one field), and the whole-line pass behind it is what
    covers the spec and every separator this function composes.

    A clause that cannot be built is dropped rather than raising — same rule as
    :func:`_model_timing_text`, for the same reason: ``status`` is the command a
    poll loop runs, so one malformed entry costs its own clause and never the
    command. The elapsed was already reduced to a plain number by the attach
    side, so the only raising left here would be a caller-supplied entry that is
    not a dict.
    """
    entries = view.get("finished_during_wait")
    if not isinstance(entries, list):
        return ""
    clauses: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        spec = str(entry.get("spec") or "")
        if not spec:
            continue
        label = str(entry.get("state") or "") or STATE_DONE
        error = entry.get("error")
        if label == STATE_FAILED and error:
            label = f"{label}: {_one_line(error)}"
        fields: List[str] = []
        elapsed = entry.get("elapsed_s")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            fields.append(format_elapsed(elapsed))
        fields.append(label)
        clauses.append(f"{spec} ({', '.join(fields)})")
    return one_line_ascii("; ".join(clauses))


def _progress_lines(view: Dict[str, Any]) -> List[str]:
    """The run's progress as **rendered text**, for a reader that cannot render.

    The `progress` object rides the result line whole (`record_progress`), which
    is the right shape for a reader that can turn a map into a sentence. The
    OpenCode **package** route cannot: there is no agent in its poll loop at all
    — the plugin's `execute` supervises the run and the model makes one blocking
    tool call — so nothing there reads `instructions/*.md`, nothing relays a
    `[PROGRESS]` line out of a log the plugin never opens, and the plugin's own
    rule forbids it to compose a sentence about a run. Measured 2026-09-21: a
    5 m 40 s four-model fan-out reached the user as silence and then one
    compressed line, with a complete `progress.models` map and twelve ticker
    lines sitting in artifacts nobody on that route reads.

    So the lines are composed here, by the renderers that already own their
    wording, and shipped ready to display:

    * the fan-out roster (:func:`render_roster_line`) — what the run is *over*,
      which never changes, so a reader that shows it once has said it;
    * the current state (:func:`_current_state_text`) — the fan-out ticker's own
      ``[PROGRESS]`` line while a slot is live, the ``progress:`` line the human
      view prints once they are all terminal. It is the line that is there on
      **every** poll, changed or not, which is what keeps a host from falling
      back to a serialized object on the ordinary poll where nothing finished;
    * the ``just finished:`` line (:func:`_just_finished_text`) — what crossed
      into terminal *during this call's own wait*, which is the one thing the
      counts structurally cannot say and the only news a poll carries.

    Order is roster, state, news — what the run is over, what it is doing, what
    moved — which is the reading order and the order they are composed in below.
    The last two are also the order the human view prints them in
    (``progress:`` then ``just finished:``).

    The **roster** is the one this list adds. It used to be claimed that all of
    this was "the order `render_run_text` prints them in", which was false:
    that renderer prints the counts line, then `just finished:`, then the
    per-model table, and it never prints the roster at all —
    `render_roster_line` occurs in this module at its import and in this
    function, and nowhere else. The roster's
    own homes are the orchestrator log (`FanoutProgress.seed` emits it there)
    and the `--detach` launch's result line (`detach.launch` renders it from the
    resolved selection) — both surfaces this route cannot read, which is the
    whole reason the line is composed here as well. So only the second of these
    two lines is also a line a human reading `status` sees; the first is one
    that view never had.

    Nothing is invented for this surface either way: both strings come from
    renderers that exist independently of it, and neither is reworded here.

    The two renderers are called bare, which is the same decision
    :func:`render_run_text` makes two inches away: both answer a checkpoint they
    do not control, so each one type-checks what it reads
    (:func:`render_roster_line` per budget, :func:`_just_finished_text` per
    clause) and a caller that wrapped a second hedge around them would be
    hedging against a defence it can see. A view with no map — ``generate-tasks``,
    an apply phase, a run predating the tracker — simply renders nothing, and
    the key never reaches the result line at all.

    The one thing that **is** checked here is that ``progress`` is a mapping,
    and that check is about the *caller*, not about the file:
    :func:`detach.classify_run` coerces a torn ``progress`` to ``{}`` while it
    builds the view (pinned by
    ``test_the_view_carries_a_mapping_whatever_the_file_says``), so no view this
    module assembles can fail it. It is kept because this is a plain module
    function over a dict and the result-line site calls it *outside* the
    ``if view.get("progress")`` guard the text renderer sits behind — an
    ``AttributeError`` here is exit 1, which rule 2 reads as "the run state is
    unreadable, stop and report", paid for a key that is display-only. Pinned at
    that level, directly, rather than through a checkpoint that can no longer
    produce the shape.
    """
    progress = view.get("progress")
    if not isinstance(progress, dict):
        return []
    lines: List[str] = []
    roster = render_roster_line(progress.get("models"))
    if roster:
        lines.append(roster)
    current = _current_state_text(progress)
    if current:
        lines.append(current)
    just_finished = _just_finished_text(view)
    if just_finished:
        lines.append(f"{JUST_FINISHED_LABEL} {just_finished}")
    return lines


def _has_live_row(models: Any) -> bool:
    """Is any slot still queued, staggering, running or retrying?

    The boundary between the two tickers the orchestrator log already switches
    between: :class:`progress_tracker.FanoutProgress` narrates while this is
    true and :class:`progress_tracker.PhaseProgress` takes over when it is not.
    :func:`_current_state_text` picks the same way, from the checkpoint, so the
    package route sees the same changeover a log reader does.

    A row that is not a mapping is not live — it is a row nothing can be said
    about, which is how every other renderer treats it — and a mapping with no
    ``state`` is, matching :func:`_model_state_label`'s ``queued`` default.
    """
    if not isinstance(models, dict):
        return False
    return any(
        isinstance(entry, dict)
        and str(entry.get("state") or "") not in TERMINAL_STATES
        for entry in models.values()
    )


def _current_state_text(progress: Dict[str, Any]) -> str:
    """One line saying what the run is doing *now*, or ``""`` when nothing is.

    The third thing a reader with no agent in the loop needs, beside the roster
    ("what is this over") and the news ("what just moved"): **the state**, on
    every poll, whether or not anything changed. Without it the plugin's toast
    fell back to ``JSON.stringify(update)`` on any poll whose window contained
    no completion — measured 2026-09-21 on a 363 s run, a multi-KB blob, and
    that is the ordinary case rather than an edge one, since a 90 s window
    containing a completion is the exception.

    Two renderers, picked by the boundary the log already switches at
    (:func:`_has_live_row`) and neither of them new:

    * while a slot is still live, :func:`progress_tracker.render_progress_line`
      — the fan-out ticker's own ``[PROGRESS]`` line, counts plus the live rows
      with their elapsed. That elapsed is derived at read time from
      ``started_at``, so two polls ninety seconds apart really do render
      different strings, which is what the host's shown-once memory needs to
      keep showing it;
    * once every slot is terminal — validation, grouping, a write phase, a
      dispatch-only phase like ``generate-tasks`` that has no map at all —
      :func:`_progress_text`, the ``progress:`` line the human view prints and
      the one Layer 2 rule 1 already has an agent relay verbatim. It carries the
      step and the executor mirrors, which is what moves in that window.

    Neither is reworded here and no third spelling is introduced; ``status``
    simply says which of its own two lines answers the question right now.
    ``(none recorded)`` is a renderer with nothing to say, not a state, so it is
    dropped rather than displayed — that is the one case that still leaves a
    poll with no line of its own.
    """
    models = progress.get("models")
    if _has_live_row(models):
        line = render_progress_line(models)
        if line:
            return line
    text = _progress_text(progress)
    return "" if text == NO_PROGRESS_RECORDED else text


def _terminal_fingerprints(view: Dict[str, Any]) -> Optional[Dict[str, tuple]]:
    """Per terminal spec, *which* ending it is — not merely that it has one.

    ``done | failed`` is the whole terminal set (there is no ``skipped`` state,
    see :data:`SKIPPED_SOURCES`), and a row that is not a dict or whose ``state``
    is a word nothing has ever written reads as not terminal.

    The value is ``(state, ended_at, elapsed_s)`` rather than the bare spec, and
    that is the whole point of this function. A phase dir is re-used run after
    run, so "this spec was already terminal when the wait began" does **not**
    mean "it did not finish during the wait": a re-review of the same plan finds
    the predecessor's finished rows on disk, the new run seeds the same specs
    ``queued``, and each one finishes again with a *new* ``ended_at``. Keyed on
    the spec alone, every one of those completions was swallowed — it was in the
    baseline before this run seeded it, and in the next poll's baseline after it
    really finished, so no poll ever announced it (F1, 2026-09-20; the source of
    the stale map is fixed at the three launch writes, and this is the half that
    still holds when a poll lands in the window *before* one of them).

    ``None`` — distinct from an empty mapping — means the map could not be read
    at all. The caller treats that as "no baseline", because an unreadable map
    that heals into a healthy one mid-wait would otherwise announce every row in
    it as having just finished.

    **A ``models`` key that is simply absent is the honest empty baseline**, not
    an unreadable one, and the two used to collapse into the same ``None``. The
    window is real and it is the launch window: the claim write drops the
    predecessor's ``progress`` outright and ``seed()`` writes this run's roster a
    moment later, so a poll landing between them reads a checkpoint carrying no
    map at all. Nothing had finished yet — which is exactly what ``{}`` says, and
    every row that turns terminal afterwards really did turn terminal during this
    wait. Read as ``None`` the whole first wait was skipped instead, so a
    ``--wait`` that spanned the launch printed no ``just finished:`` line however
    many specs ended inside it (F1, 2026-09-21). ``None`` now means only what the
    paragraph above describes: a ``models`` that is *present and not a dict*.
    """
    progress = view.get("progress") or {}
    if not isinstance(progress, dict):
        return None
    models = progress.get("models")
    if models is None:
        return {}
    if not isinstance(models, dict):
        return None
    fingerprints: Dict[str, tuple] = {}
    for raw_spec, entry in models.items():
        if not isinstance(entry, dict):
            continue
        state = str(entry.get("state") or "")
        if state not in TERMINAL_STATES:
            continue
        fingerprints[str(raw_spec)] = (
            state, str(entry.get("ended_at") or ""), str(entry.get("elapsed_s")),
        )
    return fingerprints


def _attach_finished_during_wait(
    baseline: Dict[str, Optional[Dict[str, tuple]]], views: Sequence[Dict[str, Any]]
) -> None:
    """Record, on each returned view, which specs went terminal during the wait.

    *baseline* is what :func:`_wait_for_terminal` saw when it was entered, keyed
    by ``phase_dir`` — per phase dir and not one flat set, because a poll with no
    selector lists every phase under the plan and two of them can legitimately be
    running the same spec; one flat set would let one phase's model silence the
    other's.

    Order is the map's own key order, which is the tracker's seed order (the
    order the run requested the models in), matching the table below it. It is
    deliberately *not* sorted by ``ended_at``: the stamps are written by whichever
    lane won the race and two rows finishing in the same second would reorder a
    line whose only job is to be read next to the table.

    A view whose ``phase_dir`` is not in the baseline is skipped rather than
    treated as an empty baseline. Such a view was discovered *during* the wait
    (another process opened a phase dir under this plan), so nothing here saw its
    "before" — and every terminal row in it would be announced as having just
    finished on no evidence at all. A dir that appeared mid-wait gets its roster
    from the table like any other first sighting. A dir whose baseline is
    ``None`` is skipped for the same reason: its map was unreadable when the wait
    began, so nothing was observed about it either (F4, 2026-09-20).

    A spec is announced when its terminal *fingerprint* differs from the one the
    baseline holds — a different ending, not merely a second sighting of the same
    one. That is what makes the line survive a re-used phase dir, where the specs
    that finish during the wait are exactly the specs a predecessor already left
    terminal on disk; see :func:`_terminal_fingerprints`.

    **The key is set only when the list is non-empty**, and that is a
    compatibility constraint, not tidiness: ``--json`` serializes these view
    dicts raw (``payload = {"runs": views}``), so an always-present key — even
    ``[]`` — would change the bytes of every JSON poll ever made. Every value
    written here is a ``str``, a ``float`` or ``None`` for the same reason.

    **The key rides ``--json`` like every other view key.** The paragraph two
    above says why — ``payload = {"runs": views}`` serializes these dicts raw —
    so a poll run with ``--wait --json`` really does hand back
    ``runs[0]["finished_during_wait"]``. This paragraph used to claim the
    opposite ("this key itself still never leaves the view"), contradicting its
    own neighbour and in the one direction that costs something: a consumer told
    a key cannot reach it does not handle the day it does.

    What is true is narrower, and it is about ``MULTI_LLM_RESULT`` rather than
    about ``--json``: **these dicts are not put on the result line**. The
    OpenCode package route — the one host with no agent in the loop to read a
    rendered line — gets the *rendered* line there instead:
    :func:`_progress_lines` runs
    :func:`_just_finished_text` over this key and puts the result on the
    ``MULTI_LLM_RESULT`` line beside the roster, through
    :func:`cli_common.record_progress_lines`. That is deliberate and is not the
    same thing as carrying the key: the route has no renderer, so handing it
    these dicts would leave it composing the sentence itself — which the
    plugin's own rule forbids — while handing it the sentence costs it nothing
    but a display. The structured half is still there too: ``progress.models``
    rides the same line whole, states and stamps included, so a consumer that
    wants to compute rather than display never has to parse the text.
    """
    for view in views:
        key = str(view.get("phase_dir") or "")
        if key not in baseline:
            continue
        before = baseline[key]
        if before is None:
            continue
        models = (view.get("progress") or {}).get("models")
        if not isinstance(models, dict):
            continue
        finished: List[Dict[str, Any]] = []
        for raw_spec, raw_entry in models.items():
            spec = str(raw_spec)
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            state = str(entry.get("state") or "")
            if state not in TERMINAL_STATES:
                continue
            now = (
                state, str(entry.get("ended_at") or ""), str(entry.get("elapsed_s")),
            )
            if before.get(spec) == now:
                continue
            try:
                elapsed = entry_elapsed_s(entry)
            except Exception:
                # The `started_at: "...Z"` shape that used to take the whole
                # command down (F5, 2026-09-20). A clause with no clock is worth
                # more than a traceback.
                elapsed = None
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
                elapsed = None
            error = entry.get("error")
            finished.append({
                "spec": spec,
                "state": _model_state_label(entry),
                "elapsed_s": float(elapsed) if elapsed is not None else None,
                "error": str(error) if error else None,
            })
        if finished:
            view["finished_during_wait"] = finished


def _mirror_text(key: str, mirror: Any) -> List[str]:
    """One clause per executor mirror: ``<kind> <ok>/<total> ok`` and, when any
    task did not end ``ok``, the non-ok statuses and the failed ids."""
    if not isinstance(mirror, dict) or not isinstance(mirror.get("total"), int):
        return []
    kind = mirror.get("kind") or key
    total = mirror["total"]
    counts = mirror.get("counts") if isinstance(mirror.get("counts"), dict) else {}
    tasks = mirror.get("tasks") if isinstance(mirror.get("tasks"), list) else []
    if not counts and tasks:
        # The write mirror records one row per task and no `counts`/`ok`
        # summary; read only the summary keys and a finished nine-task run
        # printed `implement 0/9 ok` (measured 2026-09-16).
        counts = _write_mirror_as_counts(mirror)["counts"]
    ok = mirror.get("ok")
    if not isinstance(ok, int):
        ok = counts.get("ok", 0) if isinstance(counts.get("ok"), int) else 0
    items_total = mirror.get("items_total")
    items_applied = mirror.get("items_applied")
    if isinstance(items_total, int) and isinstance(items_applied, int):
        # F9: `ok`/`total` count batches, not the items inside them — a
        # write mirror that recorded both (`write_batch_scheduler`) says so
        # explicitly rather than leaving "ok" to be read as "items applied".
        # "batches" appears only here, precisely because there is now an
        # item count to contrast it with; the read path's `subagents` mirror
        # carries neither key and keeps the plain `<kind> <ok>/<total> ok`.
        parts = [
            f"{kind} {ok}/{total} batches ok, "
            f"{items_applied}/{items_total} items applied"
        ]
    else:
        parts = [f"{kind} {ok}/{total} ok"]
    # F10: `in_flight` (`write_batch_scheduler._mirror_status`/`_begin_sweep`)
    # is what turns "0 dispatched, heartbeat Ns ago" — indistinguishable from
    # a stall — into naming the task that is still running and for how long.
    in_flight = mirror.get("in_flight") if isinstance(mirror.get("in_flight"), list) else []
    entries = []
    for row in in_flight:
        if not isinstance(row, dict) or not row.get("task_id"):
            continue
        # Same defensive read as `_model_timing_text`, same reason (F5,
        # 2026-09-20): `dispatched_at` goes through the same `ValueError`-only
        # parser, so one tz-aware stamp written by a future mirror would take
        # the whole poll down instead of costing this row its `(since …)`.
        try:
            elapsed = detach._elapsed_since(row.get("dispatched_at"))
        except Exception:
            elapsed = None
        since = (
            f" (since {metrics.format_duration(int(elapsed * 1000))})"
            if elapsed is not None else ""
        )
        entries.append(f"{row['task_id']}{since}")
    if entries:
        parts.append(f"in flight: {', '.join(entries)}")
    for status, count in sorted(counts.items()):
        if status != "ok" and count:
            parts.append(f"{status} {count}")
    outcomes: Dict[str, int] = {}
    for row in tasks:
        outcome = str((row or {}).get("write_outcome") or "")
        if outcome:
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    # F14: when every batch's outcome is `applied` the breakdown collapses to
    # a single `applied N` where N is exactly the `ok` count already stated
    # earlier in the line (`<kind> <ok>/<total> [batches] ok`) — printing it
    # again under a different label read as a third, smaller quantity rather
    # than a restatement, and an opencode agent once repeated it verbatim as
    # such. A breakdown with more than one outcome (`applied 7, no-op 2`)
    # still tells the reader something the `ok` count alone does not, so that
    # case is unchanged.
    if not (len(outcomes) == 1 and outcomes.get("applied") == ok):
        for outcome, count in sorted(outcomes.items()):
            parts.append(f"{outcome} {count}")
    failed = mirror.get("failed_task_ids") or []
    if failed:
        parts.append(f"failed {', '.join(map(str, failed))}")
    # F9: `skipped_task_ids`/`human_task_strategy` (resolved before dispatch,
    # `write_batch_scheduler._mirror_status`) were on `.status.json` already
    # but rendered nowhere, so the one task `--yes`'s default strategy defers
    # to a human was invisible in the finished run's own progress line.
    skipped_ids = mirror.get("skipped_task_ids") if isinstance(mirror.get("skipped_task_ids"), list) else []
    if skipped_ids:
        strategy = mirror.get("human_task_strategy")
        suffix = f" ({strategy})" if strategy else ""
        parts.append(f"deferred manual: {', '.join(map(str, skipped_ids))}{suffix}")
    # F4: `driver.verify_command`'s outcome. Without it a run that applied every
    # batch and then failed the repository's own check renders as
    # `apply_code_fixes 3/3 batches ok` with no failed id anywhere — the only
    # sign of the failure being a phase state the counts contradict.
    verification = mirror.get("verification")
    if isinstance(verification, dict) and verification.get("command"):
        verdict = "passed" if verification.get("exit_code") == 0 else (
            f"FAILED (exit {verification.get('exit_code')})"
        )
        parts.append(f"verification {verdict}: `{verification['command']}`")
    return parts


#: The modes whose ``--reaggregate`` is the retry-only-the-failures verb a
#: failed run resumes with (D2 step 4): it re-dispatches the failed batches and
#: re-reads the rest. Every mode in neither table keeps ``--force``, the only
#: resume verb it accepts. ``status`` cannot afford to import each
#: orchestrator's parser to derive this, so these are the two tables in this
#: module that are typed rather than read.
_REAGGREGATE_MODES = ("review-plan", "review-tasks", "review-code")

#: Modes whose retry verb is neither. ``generate-tasks`` has a ``--reaggregate``
#: too, but there it is the native host's second half — it only ever *consumes*
#: an existing ``task_generation_result.json`` and refuses when there is none —
#: so it never re-dispatches a failed task. The plain re-invoke does both:
#: re-dispatches what failed, consumes a valid result if one is on disk. This is
#: the verb that phase's own retry hint prints, and the ``resume with:`` line
#: has to agree with it: measured 2026-09-14, the two lines disagreed, the
#: session model followed this one, and the run ended on ``--reaggregate found
#: no task_generation_result.json`` with nothing re-dispatched.
#:
#: The four write phases resume with ``--resume``, the flag their own exit-3
#: block names ("re-invoke with --resume, which re-dispatches only the
#: unfinished batches and replays the prior-changes log from disk"). ``--force``
#: is not their resume verb at all — the apply parsers define it as an alias
#: for ``--yes`` (bulk approval), and under it the apply base re-ingests every
#: human decision instead of replaying the recorded ones. Measured 2026-09-15
#: on a failed ``apply-task-suggestions``: the log said ``--resume``, this line
#: said ``--force``, and the session model ran ``--claude-decide --force`` twice
#: before a human typed ``--resume`` by hand.
_RESUME_VERB_BY_MODE = {
    "generate-tasks": "--yes",
    "apply-suggestions": "--resume",
    "apply-task-suggestions": "--resume",
    "apply-code-fixes": "--resume",
    "implement": "--resume",
}

#: The modes whose progress mirror is ``write_subagents``
#: (``write_batch_scheduler._mirror_status``) rather than ``subagents``.
_WRITE_MODES = ("apply-suggestions", "apply-task-suggestions",
                "apply-code-fixes", "implement")


def _resume_command(view: Dict[str, Any], plan_path: Path) -> str:
    """The ``resume with:`` command, with the run's own flags spliced back in.

    Flags come from the run record's ``replay_flags`` (the launcher's list,
    D2) through :func:`cli_common.current_replay_flags`, whose live-source rule
    never lets ``status``'s own ``--view run`` argv win here (a different
    mode). ``ask`` re-derives its question positional from the checkpoint's
    ``question`` field — the one place it is stored — and falls back to the
    questionless string when the key is missing.
    """
    run = view["run"]
    mode = str(run.get("mode") or view.get("phase") or "review-plan")
    flags = cli_common.current_replay_flags(run.get("replay_flags"), mode=mode)
    positionals = [str(plan_path)]
    if mode == "ask" and view.get("question"):
        positionals.append(str(view["question"]))
    verb = _RESUME_VERB_BY_MODE.get(mode) or (
        "--reaggregate" if mode in _REAGGREGATE_MODES else "--force"
    )
    return cli_common.retry_command(mode, *positionals, verb, flags=flags)


def _resume_shape(view: Dict[str, Any]) -> "tuple":
    """``(shape, repair, redispatch, kept)`` from the document, in two reads.

    1. the run record's close-out — D4's ``{"state": "runner_rejected",
       "reason_code": …}``, written at the checkpoint's top level and mirrored
       into the view — is the refusal's classification made durable and, after
       a refusal, the only thing on disk ⇒ shape 1;
    2. otherwise ``progress.subagents`` (``_mirror_status``), whose ``counts``
       split by ``FATAL_STATUSES`` ⇒ shape 1 or 2. Its freshness is the
       writer's property: ``claim_run`` drops the predecessor's ``run`` **and**
       ``progress`` together, so a mirror beside a run record is that run's.
    3. neither ⇒ shape 2, the transient default.

    Never ``subagent_batch_report.json``: the report in the phase dir may be
    the previous run's, and it is an arbitrarily large read on a command that
    must stay sub-second.
    """
    run = view["run"]
    reason_code = None
    if run.get("state") == "runner_rejected" or view.get("state") == "runner_rejected":
        reason_code = run.get("reason_code") or view.get("reason_code") or "runner_rejected"
    progress = view.get("progress") or {}
    mirror = progress.get("subagents") or {}
    if not mirror and isinstance(progress.get("write_subagents"), dict):
        # A write phase mirrors per-batch outcomes instead of a counts table
        # (`write_batch_scheduler._mirror_status`); its shape is the transient
        # default and its cost line counts batches.
        mirror = _write_mirror_as_counts(progress["write_subagents"])
    counts = mirror.get("counts") if isinstance(mirror.get("counts"), dict) else {}
    shape = cli_common.retry_shape(reason_code=reason_code, statuses=counts.keys())
    repair = None
    if shape == cli_common.SHAPE_REPAIR:
        repair = cli_common.repair_sentence(
            reason_code=reason_code, explanation=_reason_explanation(reason_code),
            counts=counts,
        )
    redispatch = len(mirror.get("failed_task_ids") or []) or None
    kept = mirror.get("ok") if isinstance(mirror.get("ok"), int) else None
    return shape, repair, redispatch, kept


def _write_mirror_as_counts(mirror: Dict[str, Any]) -> Dict[str, Any]:
    """``progress.write_subagents`` in the shape ``_resume_shape`` reads.

    ``ok`` counts every batch the mirror records as ``ok`` — the ones this run
    applied and the ones a resume skipped because their result file was already
    on disk — and ``failed_task_ids`` is carried as recorded.
    """
    tasks = mirror.get("tasks") if isinstance(mirror.get("tasks"), list) else []
    counts: Dict[str, int] = {}
    for row in tasks:
        status = str((row or {}).get("status") or "")
        if status:
            counts[status] = counts.get(status, 0) + 1
    return {
        "counts": counts,
        "ok": counts.get("ok", 0),
        "failed_task_ids": list(mirror.get("failed_task_ids") or []),
    }


def _reason_explanation(reason_code: Optional[str]) -> Optional[str]:
    """``_REASON_EXPLANATION[reason_code]``, imported only on this failed path."""
    if not reason_code:
        return None
    try:
        from .subagent_executor import _REASON_EXPLANATION
    except Exception:  # pragma: no cover - a broken install still prints the code
        return None
    return _REASON_EXPLANATION.get(reason_code)


def _resume_lines(view: Dict[str, Any], plan_path: Path) -> List[str]:
    """The shaped resume block: the repair first when there is one, then the
    single ``resume with:`` command, then what it costs."""
    shape, repair, redispatch, kept = _resume_shape(view)
    lines: List[str] = []
    if shape == cli_common.SHAPE_REPAIR:
        lines.append(f"      repair first: {repair}")
    lines.append(f"      resume with: {_resume_command(view, plan_path)}")
    if redispatch:
        mode = str(view["run"].get("mode") or view.get("phase") or "")
        if mode in _WRITE_MODES:
            # The write loop stops at the failed batch, so what a resume
            # re-dispatches is that batch and everything after it; the kept
            # count is the batches whose result files it skips over.
            cost = f"re-dispatches the {redispatch} failed batch(es) and the ones never reached"
            if kept:
                cost += f", keeps the {kept} already applied"
            lines.append(
                f"      ({cost}; the half-applied state is left in place — inspect "
                f"it, or `util restore-batch` the failed batch, before resuming)"
            )
        else:
            cost = f"re-dispatches the {redispatch} failed task(s)"
            if kept is not None:
                cost += f" and keeps the {kept} valid result(s) already on disk"
            lines.append(f"      ({cost}; --force is not needed and would re-run the reviewer panel)")
    return lines


def _abort_command(plan_path: Path, view: Dict[str, Any]) -> str:
    """The one selector-carrying ``--abort`` spelling for a single listed run."""
    run_id = str(view["run"].get("run_id") or "")
    if run_id:
        return detach.abort_command_for(str(plan_path), run_id=run_id)
    phase = view.get("phase") or os.path.basename(view["phase_dir"])
    return dispatcher_command("status", str(plan_path), "--abort", "--phase", str(phase))


def _run_runner_facts(
    views: Sequence[Dict[str, Any]]
) -> "tuple":
    """``(host, specs, source)`` the listed runs recorded about their own
    subagents.

    The host is the first one any listed run wrote into ``.status.json``
    (``RunHandle`` records the launcher's ``--host``); the specs are the runner
    keys that same run's executor mirror pinned, i.e. what its dispatches
    actually ran on rather than what a fresh resolution would pick now. Checked
    under both mirror keys — ``subagents`` (the read path) and
    ``write_subagents`` (`write_batch_scheduler._mirror_status`, F7) — a write
    phase carries only the second, and until it recorded a runner at all
    `status` polled outside the launching session fell through to a fresh
    resolution and named the *polling* host's model instead of the run's.

    **A host-bearing view that recorded no spec must not win outright** (F6,
    2026-09-16): with several phases listed, the first one carrying a host can
    be a run whose executor mirror is empty (a failed-before-dispatch run, a
    stale pre-fix record, ``--dry-run-subagents``) while a *later* view holds
    the real answer. Measured on a `generate-tasks` run: the default view's
    ``apply-suggestions`` row sorted first, its mirror had recorded nothing,
    and the search stopped there instead of reaching the `generate-tasks` row
    one below it that had. So an empty-spec view is skipped rather than
    returned, and its host is kept only as the fallback for when no view in
    the whole list ever recorded a spec.

    **Among the views that DID record a spec, the most recently updated one
    wins — not whichever sorts first** (F4, 2026-09-16): ``discover_phase_dirs``
    lists phases in alphabetical directory order, so a `generate-tasks` run's
    pre-`--variant low` record outranked `review-tasks`' own `:low` record
    directly below it, under a line that claimed to be "recorded by this run's
    own dispatches". Every view carries ``updated_at``, so the candidates are
    collected first and the freshest one answers. ``source`` names which run
    the winning spec came from — ``None`` when only one view ever recorded a
    spec, so the caller's existing "this run's own dispatches" wording stays
    exactly as specific as it always was; a phase name, when more than one
    view recorded one, so the line stops asserting an ownership it cannot back.
    """
    fallback_host: Optional[str] = None
    candidates: List[Dict[str, Any]] = []
    for view in views or ():
        run = view.get("run") or {}
        host = run.get("host")
        if not isinstance(host, str) or not host.strip():
            continue
        progress = view.get("progress") or {}
        mirror = progress.get("subagents")
        versions = mirror.get("runner_versions") if isinstance(mirror, dict) else None
        if not isinstance(versions, dict) or not versions:
            mirror = progress.get("write_subagents")
            versions = mirror.get("runner_versions") if isinstance(mirror, dict) else None
        specs = sorted(str(spec) for spec in versions) if isinstance(versions, dict) else []
        if specs:
            name = view.get("phase") or os.path.basename(str(view.get("phase_dir") or "")) or None
            candidates.append({
                "host": host.strip(),
                "specs": specs,
                "name": name,
                "updated_at": str(view.get("updated_at") or ""),
            })
            continue
        if fallback_host is None:
            fallback_host = host.strip()
    if candidates:
        candidates.sort(key=lambda c: c["updated_at"])
        winner = candidates[-1]
        source = winner["name"] if len(candidates) > 1 else None
        return winner["host"], winner["specs"], source
    return fallback_host, [], None


def render_runner_text(
    plan_path: Path,
    host_id: Optional[str] = None,
    *,
    views: Sequence[Dict[str, Any]] = (),
) -> List[str]:
    """The resolved subagent runner, its layer of origin, and who consults it (D3o).

    **The run's host wins over the invoking one.** A poll is routinely typed
    from a different host than the one that launched the run — measured
    2026-09-16, three ``status`` calls on an ``opencode`` run whose eleven
    batches all dispatched to ``opencode:openrouter/…`` each printed
    ``claude-code:sonnet``, because the host came from *this* invocation's argv
    while the run's own sat unread in ``.status.json``. So the run's host is
    what this reports, and with it:

    * the specs the run's executor mirror recorded, when it has dispatched
      anything — the only account on disk of what its subagents really ran on,
      and, when more than one listed view recorded one, the most recently
      updated view's (F4, 2026-09-16), named in the printed line rather than
      claimed as "this run's own";
    * otherwise a fresh resolution for that host, with the **session** rung
      skipped whenever the run was launched elsewhere: that rung reads
      ``MULTI_LLM_HOST_SESSION_MODEL`` off *this* process, which is the polling
      session's model and not the run's. The plugin's own poll passes
      ``--host opencode`` from inside the session that launched the run, so
      there the rung still applies and still answers ``(supplied by session)``.

    "I set ``driver.subagent_runner`` and nothing changed" has three different
    answers — a higher layer won, no layer expressed an opinion so the running
    host's own ``default_subagent_runner`` answered, or this host dispatches
    subagents natively and never consults the key outside ``--exec-subagents`` —
    and a merged value printed on its own distinguishes none of them. So the
    layer is printed with the value, resolved through the *same*
    ``resolve_subagent_runner`` a dispatch takes rather than re-derived here.

    It is reported, never resolved: ``status`` writes nothing, takes no lock and
    must stay sub-second, so a config layer that cannot be read (a ``ConfigError``
    from a broken override, an unknown ``--host``, a partial install) prints
    nothing at all rather than failing a poll. Text only — the ``--json`` views
    stay exactly two keys, ``workflow`` and ``runs``, because this is a config
    fact and blending it into either state view is what ``--view`` exists to
    prevent.
    """
    try:
        from . import host_capabilities as caps
        from .provider_registry import resolve_subagent_runner

        invoking_host = caps.resolve_host_id(sys.argv[1:], explicit=host_id)
        run_host, recorded, recorded_by = _run_runner_facts(views)
        resolved_host = run_host or invoking_host
        if recorded:
            spec = ", ".join(recorded)
            source = (
                f"recorded by {recorded_by}'s own dispatches" if recorded_by
                else "recorded by this run's own dispatches"
            )
        else:
            spec, layer = resolve_subagent_runner(
                "full", anchor=str(plan_path), host_id=resolved_host,
                skip_session=resolved_host != invoking_host,
            )
            source = f"supplied by {layer}"
        native = str(caps.entry(resolved_host).get("subagents")) == "native"
    except Exception:
        return []
    scope = (
        f"{resolved_host} dispatches subagents natively, so it is consulted "
        "only under --exec-subagents"
        if native
        else f"every subagent {resolved_host} dispatches runs through it"
    )
    return [
        "driver.subagent_runner — what a dispatched subagent would run:",
        f"  {spec}  ({source})",
        f"  {scope}",
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="multi_llm.py status",
        description=(
            "Report a plan's workflow phases (state.json) and its detached runs "
            "({phase}/.status.json). One command, two named views, one vocabulary."
        ),
        parents=[build_parent_parser(include_interactive=False)],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes: 0 = terminal and successful | 2 = non-terminal (still "
            "running, or no run recorded) | 3 = terminal and needs a human | 1 = "
            "the phase dir could not be read.\n\n"
            "The exit code IS the poll-loop condition; with --wait the waiting "
            "is already in the call, so poll again on 2 from your own turn loop "
            "and never put `sleep` and `status` in the same shell call."
        ),
    )
    parser.add_argument("--plan-file", required=True, help="Path to the plan file")
    parser.add_argument(
        "--view", choices=[VIEW_WORKFLOW, VIEW_RUN, VIEW_ALL], default=VIEW_ALL,
        help="Which layer to report (default: all)",
    )
    parser.add_argument(
        "--phase", "--mode", dest="phase", default=None,
        help="Scope the run view to one phase dir. Accepts a mode word or a "
             "phase name (review-code and code-review both work).",
    )
    parser.add_argument(
        "--question", default=None,
        help="Select one ask run by its exact question text",
    )
    parser.add_argument(
        "--run-id", dest="run_id", default=None,
        help="Select one run by id (a unique prefix is enough)",
    )
    parser.add_argument(
        "--all", dest="include_all", action="store_true",
        help="Include historical terminal runs (ask dirs list only the latest by default)",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument(
        "--wait", action="store_true",
        help=f"Poll internally until the run is terminal or --timeout elapses "
             f"(default {DEFAULT_WAIT_TIMEOUT_S:.0f}s, deliberately under the "
             "tightest host shell cap), then return the non-terminal view with exit 2",
    )
    parser.add_argument(
        # A SENTINEL, not a value: `None` is what lets `main` tell "the caller
        # asked for this many seconds" from "the caller said nothing", which is
        # the whole of the resolution order below (and of which of the two
        # clamps is loud). Carrying `DEFAULT_WAIT_TIMEOUT_S` here instead would
        # make `progress.poll_s` unreachable — argparse would have already
        # answered the question.
        "--timeout", type=float, default=None,
        help=f"--wait timeout in seconds (default: progress.poll_s, else "
             f"{DEFAULT_WAIT_TIMEOUT_S:.0f}s; clamped under the host's shell cap)",
    )
    parser.add_argument(
        "--abort", action="store_true",
        help="Stop the selected detached run(s): identity probe, process-group "
             "SIGTERM, 10s, SIGKILL, then run.state: aborted. It reaches a run "
             "that recorded one: a foreground run holds no run record, so there "
             "it answers `nothing was running` and the kill path is the host's "
             "own background job handle.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Close out every run under this plan whose process is gone",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="With --abort: stop a run whose identity probe is inconclusive",
    )
    # Accepted so the pre-dispatcher spelling (`--status`) keeps working; the
    # views make it redundant.
    parser.add_argument("--status", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_phase(word: Optional[str]) -> Optional[str]:
    """Resolve ``--phase``/``--mode`` through the mode table (mode word or phase).

    Also accepts the on-disk phase-*directory* word for a phase whose directory
    name differs from its ``modes.py`` phase name (``apply-code-fixes`` is the
    only one today, dir ``apply-fixes``) — a human polling by eye reads the
    directory, not the phase table, and ``PHASE_DIRECTORIES`` is the one place
    that mapping is recorded. Returned as-is: ``phase_dir_path``'s own identity
    fallback resolves a dir word that is already a ``PHASE_DIRECTORIES`` key.

    An unknown selector is a *usage* error, so it exits **3** like every other
    "this run cannot proceed without a human" case (D1a, and this command's own
    epilog) rather than the exit 1 a string-valued ``SystemExit`` would be mapped
    to — a shim must be able to tell a mistyped ``--phase`` from a status command
    that genuinely failed to read the phase dir. The message goes to stderr
    because ``--json``'s stdout has to stay parseable.
    """
    if not word:
        return None
    from .cli_common import _modes

    modes = _modes()
    resolved = modes.phase_from_selector(word)
    # The directory words read the same tuple discovery does
    # (`PHASE_DIR_NAMES`, plan §7.4a), so the two cannot drift.
    if resolved is None and word in PHASE_DIR_NAMES:
        resolved = word
    if resolved is None:
        valid = list(modes.phase_names()) + sorted(
            set(PHASE_DIR_NAMES) - set(modes.phase_names())
        )
        print(
            f"ERROR: unknown phase or mode '{word}'. Valid phases: "
            + ", ".join(valid),
            file=sys.stderr,
        )
        raise SystemExit(EXIT_NEEDS_HUMAN)
    return resolved


def _scope_label(args: argparse.Namespace) -> Optional[str]:
    """What the caller asked about, so "nothing found" names it."""
    if args.run_id:
        return f"run {args.run_id}"
    if args.question:
        return f"question {args.question!r}"
    if args.phase:
        return f"phase {args.phase}"
    return None


def _discovery_report(plan_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """What discovery skipped (plan §7.4a2/b), for the text line and ``--json``.

    ``ignored_dirs``: non-phase children holding a ``.status.json``, listed only
    when no ``--phase`` pins one directory. ``copies``: dirs whose run id another
    phase dir already carries (:func:`_dedupe_run_ids`). Reads the checkpoints
    directly — no classification, no close-out — so it never writes.
    """
    phase = _resolve_phase(args.phase)
    stubs: List[Dict[str, Any]] = []
    for directory in discover_phase_dirs(plan_path, phase=phase):
        try:
            status = status_file.read_status(str(directory))
        except Exception:
            continue
        run = status.get("run") if isinstance(status.get("run"), dict) else {}
        stubs.append(
            {"phase_dir": str(directory), "phase": status.get("phase"), "run": run}
        )
    return {
        "ignored_dirs": [] if phase else ignored_phase_dirs(plan_path),
        "copies": _dedupe_run_ids(stubs)[1],
    }


def _collect(plan_path: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    phase = _resolve_phase(args.phase)
    views = [
        detach.classify_run(str(directory))
        for directory in discover_phase_dirs(plan_path, phase=phase)
    ]
    # Plan §7.4b: two views with one run id are one run.
    views = _dedupe_run_ids(views)[0]
    return _select(
        views,
        run_id=args.run_id,
        question=args.question,
        include_all=args.include_all,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    bootstrap_streams()
    args = parse_args(argv)
    plan_path = Path(os.path.abspath(args.plan_file))

    if not plan_path.exists():
        payload = {"error": f"Plan file not found: {plan_path}"}
        print(json.dumps(payload, indent=2))
        return EXIT_ERROR

    if args.abort:
        return _run_abort(plan_path, args)
    if args.sweep:
        return _run_sweep(plan_path, args)

    workflow = None
    if args.view in (VIEW_WORKFLOW, VIEW_ALL):
        try:
            workflow = _workflow_payload(plan_path)
        except Exception as exc:  # unreadable state.json is exit 1, not a crash
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
            return EXIT_ERROR

    if args.view == VIEW_WORKFLOW:
        # Byte-identical to `check_workflow_prerequisites.py --plan-file X
        # --status`, so a caller that only ever wanted today's phase map keeps
        # today's behavior — including its exit 0.
        print(json.dumps(workflow, indent=2))
        return EXIT_OK

    views = _collect(plan_path, args)
    discovery = _discovery_report(plan_path, args)
    if args.wait:
        # `--timeout` is a sentinel out of argparse, and this is the one place
        # it becomes a number: between `parse_args` and the `--wait` branch,
        # where the resolution can see both the flag and the config. Resolved
        # only under `--wait`, so a plain `status` still pays no `load_config`
        # on a path that would never use the answer.
        try:
            args.timeout = _resolve_wait_timeout(args, plan_path)
        except _WaitTimeoutRefused as exc:
            # Exit 3, and nothing on stdout: rule 2 stops the loop on a `3` and
            # relays the message verbatim because it names the fix, while exit 1
            # would have the agent report the *run* as unreadable. Stderr keeps
            # `--json`'s stdout parseable — what there is of it, which is
            # nothing, because no view was resolved to print (F4/F7, 2026-09-20).
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_NEEDS_HUMAN
        views = _wait_for_terminal(plan_path, args, views)

    # N5, 2026-09-16: the dispatcher stamped `ctx.mode = "status"`,
    # `ctx.phase = None` before this ever ran (`status` is a command, not a
    # phase) — right for a plain multi-run listing, but the opencode plugin's
    # supervised poll pins one run (`--run-id`, sometimes also `--phase`) and
    # reads this leg's own MULTI_LLM_RESULT verbatim. That payload said
    # `{"mode": "status", "phase": null, …}` for a finished `implement` run,
    # and a weaker model than the one that hit this had to infer its phase had
    # finished from `state: done` alone. The watched run already knows both —
    # `.status.json.run.mode`, falling back to the phase dir it lives in — so a
    # poll pinned to exactly one run carries them into its own result.
    # Ambiguous (more than one view, or no selector at all) keeps
    # `status`/`None`: no single run is "the" answer there. F3, 2026-09-17:
    # what makes a poll "the watched run's" is the selector, not whether it
    # also waited, so this sits outside `if args.wait` — a single-view poll
    # without `--wait` was still emitting the old `status`/`null` pair. F8a,
    # 2026-09-20: `--question` joined the pair. The selector list `status`
    # documents is `--phase`/`--question`/`--run-id`, and `ask` is the one mode
    # whose instruction core forbids `--phase` outright (`--phase ask` selects
    # every question ever asked under the plan), so the ask lane polls by
    # `--question` and by nothing else — leaving the one mode that always pins
    # exactly one run as the one mode whose result line carried no `progress`.
    # The rule the test below is written to is "did the caller name one run?",
    # and all three selectors answer it.
    if len(views) == 1 and (args.phase or args.question or args.run_id):
        watched = views[0]
        watched_run = watched.get("run") or {}
        watched_mode = str(
            watched_run.get("mode") or watched.get("phase") or ""
        ) or None
        watched_phase = str(watched.get("phase") or "") or None
        record_mode_phase(watched_mode, watched_phase)
        # Same selector, same reason one step further: a supervised poll that
        # pins one run reads its own `MULTI_LLM_RESULT` verbatim and never opens
        # the rendered text, so the per-model map has to ride the result line or
        # that caller would have to regex a human table to relay anything
        # mid-run. Whole and unreworded — the numbers come from the status file,
        # which is the point of producing them there.
        record_progress(watched.get("progress"))
        # And beside it, the same run rendered (`_progress_lines`). The map
        # alone was only half an answer for the one caller that has no agent in
        # its loop: it could relay nothing without composing a sentence about
        # the run, which is precisely what the OpenCode plugin may not do. A
        # separate key rather than one more key inside `progress`, so that
        # object stays what the status file wrote.
        record_progress_lines(_progress_lines(watched))

    exit_code = exit_code_for(views)

    # Plan §5: a detached run whose result was `partial` exits 0, and
    # `cli_common.emit` would turn exit 0 into `done` — the word the host then
    # relays. The worker carried the word across the process boundary as
    # `run.result_state`; a poll pinned to that one run restates it. The failed
    # specs and reasons are already on `progress.models`, recorded above. Only
    # on exit 0: any other exit already has its own word. `--json`'s
    # `runs[].run.state` stays the process word `done`.
    if (
        exit_code == EXIT_OK
        and len(views) == 1
        and (args.phase or args.question or args.run_id)
    ):
        carried = _result_state(views[0])
        if carried:
            record_state(carried)

    # F7, 2026-09-16: the opencode plugin polls only `--view run`, so a
    # terminal run's real follow-up — the workflow view's `notice:`/`suggested
    # next:` line — lived in text that channel never read. The model saw
    # `next: null`, said "nothing further to run", then invented
    # `--apply-task-suggestions` from general knowledge instead of the notice
    # already sitting one view over. Carried only for a TERMINAL run (every
    # listed view done/failed/aborted) and only on `--view run` itself —
    # `--view all` already renders the workflow section in full, and a
    # still-running poll has nothing to say about what comes after a run that
    # has not finished. A workflow-read failure here is swallowed rather than
    # failing the poll: this line is advisory, the run view is not.
    terminal_run = bool(views) and not any(
        v["verdict"] not in detach.TERMINAL_VERDICTS for v in views
    )
    workflow_trailer: List[str] = []
    terminal_notice: Optional[str] = None
    if args.view == VIEW_RUN and terminal_run:
        try:
            terminal_workflow = _workflow_payload(plan_path)
        except Exception:
            terminal_workflow = None
        if terminal_workflow:
            trailer = _workflow_trailer_lines(terminal_workflow, prefix="")
            if trailer:
                workflow_trailer = [f"workflow {line}" for line in trailer]
                terminal_notice = "; ".join(trailer)

    if args.json:
        payload: Dict[str, Any] = {"runs": views}
        if args.view == VIEW_ALL:
            payload = {"workflow": workflow, "runs": views}
        # Plan §7.4a2/b: what discovery skipped, only when there is something
        # to say, so a plan with none keeps today's payload byte for byte.
        if discovery.get("ignored_dirs"):
            payload["ignored_dirs"] = list(discovery["ignored_dirs"])
        if discovery.get("copies"):
            payload["copied_runs"] = list(discovery["copies"])
        print(json.dumps(payload, indent=2, default=str))
    else:
        lines: List[str] = []
        if args.view == VIEW_ALL and workflow is not None:
            lines += render_workflow_text(workflow)
            lines.append("")
        lines += render_run_text(
            views, plan_path=plan_path, workflow=workflow,
            scope=_scope_label(args),
        )
        # Plan §7.4a2: after the phase views, never silently dropped.
        lines += _ignored_text(
            discovery.get("ignored_dirs") or [], discovery.get("copies") or []
        )
        runner = render_runner_text(
            plan_path, host_id=getattr(args, "host", None), views=views
        )
        if runner:
            lines.append("")
            lines += runner
        if workflow_trailer:
            lines.append("")
            lines += workflow_trailer
        print("\n".join(lines))

    # Plan §2.4: a `mode: truncated` validation source sets the notice on any
    # terminal poll (every view, not only `--view run`); it qualifies the
    # verdicts the workflow step acts on, so it comes first and joins the
    # workflow trailer in ONE record. Either part alone is recorded unchanged;
    # a `mode: path` source sets none.
    truncation_notice = (
        _validation_context_notice(views) if terminal_run else None
    )
    combined_notice = "; ".join(
        part for part in (truncation_notice, terminal_notice) if part
    )
    if combined_notice:
        record_notice(combined_notice)

    for view in views:
        for path in view["output_files"].values():
            if isinstance(path, str):
                record_artifact(path)
        if view["run"].get("log_path"):
            record_artifact(view["run"]["log_path"])
    # The worker recorded this on its own result line, which a detached caller
    # never saw; a poll is the route it reaches the host by, so the poll's line
    # carries it too. One field, so the first view holding one wins — the text
    # above already rendered every phase's.
    recommended = next(
        (v["consolidation_recommended"] for v in views
         if v.get("consolidation_recommended")),
        None,
    )
    if recommended:
        record_consolidation_recommended(recommended)
    # Plan §2.4 step 3: one field, so the view whose object has the worst
    # top-level `mode` wins, first in discovery order on a tie (its `phase`
    # says which); the text above rendered every phase's line.
    validation_context = pick_worst(
        [v.get("validation_context") for v in views]
    )
    if validation_context:
        record_validation_context(validation_context)
    if exit_code == EXIT_NOTHING_TO_DO and views:
        # `views[0]` is `discover_phase_dirs`' alphabetical order, not "the run
        # that made the exit code 2" — with several phases listed, a finished
        # one can sort first while a different one is the live reason for the
        # exit code (F3, 2026-09-16: `apply-suggestions` sorted ahead of the
        # still-running `generate-tasks`, and the poll command handed back
        # polled the finished run straight back to exit 0). Pick the first
        # non-terminal view; only when none is live does the finished one
        # answer, which is the genuine "nothing to do" case.
        # F1, 2026-09-17: "live" is non-terminal *and* actually holding a run.
        # `no_run` is non-terminal only because there is nothing to be terminal
        # about — a phase run in the foreground (every Claude Code host phase)
        # leaves a checkpoint with no `run` record — and counting it live told
        # a host `state: running` about a phase nothing ever launched, and
        # handed back a poll command for it that can never leave exit 2.
        live_views = [
            v for v in views
            if v["verdict"] not in detach.TERMINAL_VERDICTS
            and v["verdict"] != detach.VERDICT_NO_RUN
        ]
        # F2, 2026-09-21: the fallback was a bare `views[0]`, so on a plan whose
        # every phase ran in the foreground it handed back a poll of a `no_run`
        # view — F1's rule broken through the back door, because the guard above
        # only ever chose *between* views. It used to break cheaply
        # (`status_command_for` returned an instant poll); now that the command
        # carries `--wait`, following it spends the whole clamped `poll_s` and
        # still returns exit 2 with the identical `next`, forever. So the
        # fallback skips `no_run` as well, leaving the already-finished run this
        # branch means by "nothing to do" — whose poll exits 0 or 3 and ends the
        # loop. When that leaves nothing at all, *no* `next` is handed back: a
        # plan with no run recorded anywhere has nothing to poll, and a host
        # following `next` mechanically is owed a command that can change its
        # own answer.
        candidates = live_views or [
            v for v in views if v["verdict"] != detach.VERDICT_NO_RUN
        ]
        if candidates:
            first = candidates[0]
            record_next(
                detach.status_command_for(
                    str(plan_path),
                    phase=first.get("phase"),
                    run_id=str(first["run"].get("run_id") or "") or None,
                    # The run's own host, so the poll this hands back reports
                    # the run's runner rather than the reader's (F6,
                    # 2026-09-16).
                    host=str(first["run"].get("host") or "") or None,
                )
            )
        # The exit code stays 2 — the poll loops key off it, never off this
        # word — but "nothing-to-do" is a false alarm when the reason is a run
        # still in flight rather than an already-finished or unrecorded one
        # (F11, 2026-09-16: the payload said `nothing-to-do` on the same
        # screen whose text said `generate-tasks run … running`).
        if live_views:
            # `died` only when every live view is one: a plan holding a dead run
            # beside a genuinely running one is, on the whole, still running.
            record_state(
                STATE_DIED if all(_died(v) for v in live_views) else STATE_RUNNING
            )
    return exit_code


def _wait_clamp_ceiling(host_id: Optional[str]) -> Optional[Tuple[str, int]]:
    """``(host, its foreground command cap)`` a ``--wait`` is clamped against.

    ``None`` means *do not clamp*, and there are exactly two ways to get it:
    the named host declares ``timeout_ceiling_s: 0``, which is the documented
    "no documented cap" value rather than a cap of zero (codex ships it), and
    the capability data could not be read at all. The second follows
    ``render_runner_text`` and ``effective_ceiling``: unreadable capability data
    bounds nothing rather than failing, because ``status`` is the command a poll
    loop runs and a capability read is not what a poll is about.

    With no host *declared at all* — neither ``--host`` nor ``$MULTI_LLM_HOST``,
    see :func:`_declared_host_id` — the clamp falls back to the tightest declared
    cap rather than being skipped: an unrouted poll could be running under any
    host, and overshooting one of their caps by accident is the failure the
    clamp exists for.
    """
    try:
        from . import host_capabilities as caps

        if host_id:
            ceiling = caps.timeout_ceiling_s(host_id)
            return (host_id, ceiling) if ceiling else None
        return caps.tightest_timeout_ceiling()
    except Exception:
        return None


def _declared_host_id(args: argparse.Namespace) -> Optional[str]:
    """The host this invocation *declared*, or ``None`` when neither channel did.

    :func:`host_capabilities.resolve_host_id` owns the ladder — ``--host`` →
    ``$MULTI_LLM_HOST`` → :data:`DEFAULT_HOST` — and it is what
    :func:`render_runner_text` resolves in this same invocation and what the
    dispatcher validates ``--host`` against. The clamp asks it the same question
    and differs only in the last rung: a host nobody named is not a host to
    clamp against, so the default is *not* substituted here and the caller falls
    back to the tightest declared cap instead.

    F3, 2026-09-21: the clamp read ``args.host`` alone, so a poll routed entirely
    through the env var — which is how an installer routes one — was clamped
    against the tightest ceiling on record (90 s) instead of its own host's, and
    the stderr note explaining the clamp told the caller no ``--host`` had been
    given when a host had in fact been declared.
    """
    from . import host_capabilities as caps

    explicit = getattr(args, "host", None)
    if not explicit and not os.environ.get(caps.HOST_ENV):
        return None
    return caps.resolve_host_id(explicit=explicit)


def _resolve_wait_timeout(args: argparse.Namespace, plan_path: Path) -> float:
    """``--wait``'s deadline: resolve the source, then clamp the *resolved value*.

    Order, and nothing else reads ``progress.interval_s`` on the way through:

    1. an explicit ``--timeout`` wins;
    2. else ``progress.poll_s`` — anchored on the plan, so a poll issued from a
       foreign CWD still resolves the *project's* cadence;
    3. else :data:`DEFAULT_WAIT_TIMEOUT_S`. ``poll_s`` itself ships at 90 s, so a
       project that configures nothing keeps exactly today's wait.

    ``interval_s`` is the ticker knob and is deliberately not consulted:
    silencing the ticker lines must not turn a poll into an instant return.

    **The clamp binds whichever source won, an explicit ``--timeout`` included.**
    "Wins" settles precedence, not the ceiling. The rendered poll command Layer 2
    tells an agent to run always passes ``--timeout <interval>``, so a clamp
    guarding only the ``poll_s`` path would leave the one command the
    instructions actually emit unguarded: a ``--timeout 300`` on OpenCode asks
    for a 300 s foreground command under a documented 120 s cap and is killed
    mid-wait with no view rendered at all.

    The two paths differ only in **loudness**. A clamped ``poll_s`` is silent —
    the user configured a cadence, not a deadline, and shortening a cadence
    costs them one extra poll. A clamped explicit ``--timeout`` prints one
    stderr line naming the requested value, the clamped value and the host,
    because that caller asked for a specific number and is getting another.
    Stderr, not stdout, so ``--json``'s output stays parseable.

    **Two ways this refuses, and both wear the same clean exit-3 shape.**

    A ``ConfigError`` — from a rejected ``progress.poll_s`` (``0``, or below the
    5 s floor), *or* from any unrelated block in the same override file, because
    ``load_config`` validates the whole document — is caught and re-raised as
    :class:`_WaitTimeoutRefused` for ``main`` to print. F4, 2026-09-20: it used
    to propagate, and the dispatcher turned it into a traceback plus **exit 1**.
    Measured with ``MULTI_LLM_PROVIDERS_CONFIG`` pointing at a config carrying a
    bogus ``driver:`` key: plain ``status`` exited 2 normally while ``status
    --wait --json`` exited 1 with a stack trace on stderr and no JSON on stdout.
    Rule 2 reads exit 1 as "the run state is unreadable — stop and report", so a
    config edit made *during* a run ended the poll loop on a misdiagnosis of the
    run. Exit 3 is this codebase's "needs a human", which is exactly what a
    rejected config key is; the rejection still surfaces, named, which was the
    point of validating at load.

    And a non-finite or negative explicit ``--timeout`` is refused here rather
    than obeyed. ``--timeout nan``/``inf``/``-1`` all parse as floats, and with
    ``--host codex`` (``timeout_ceiling_s: 0``, so no clamp) they reach
    ``_wait_for_terminal`` intact: ``time.time() >= nan`` is ``False`` forever
    and the poll never returns. ``validate_progress_block`` rejects ``.nan`` and
    ``.inf`` for ``poll_s`` **precisely** to prevent "a ``--wait`` that never
    returns", and the CLI path was undoing that one layer up (F7, 2026-09-20).
    """
    from .provider_registry import ConfigError, get_progress_poll

    explicit = args.timeout is not None
    if explicit:
        requested = float(args.timeout)
        if requested != requested or requested in (float("inf"), float("-inf")):
            raise _WaitTimeoutRefused(
                f"--timeout {args.timeout!r} is not a finite number of seconds. "
                "A non-finite deadline is never reached, so `--wait` would poll "
                "forever with nothing to hand back. Pass a positive number of "
                "seconds, or drop `--timeout` and let `progress.poll_s` answer."
            )
        if requested <= 0:
            raise _WaitTimeoutRefused(
                f"--timeout {requested:g} is not a positive number of seconds. "
                "A zero or negative deadline returns before the first re-read, "
                "which is the instant-return failure `progress.poll_s: 0` is "
                "rejected for. Pass a positive number of seconds, or drop "
                "`--timeout` and let `progress.poll_s` answer."
            )
    try:
        resolved = (
            float(args.timeout) if explicit
            else float(get_progress_poll(anchor=str(plan_path)))
        )
    except ConfigError as exc:
        raise _WaitTimeoutRefused(
            f"{exc}\n`--wait` resolves its deadline through the config layer, so "
            "a config this run cannot read stops the poll rather than guessing a "
            "cadence. Fix the config and poll again; the run itself is untouched "
            "(`status` without `--wait` still reports it)."
        ) from exc

    host_id = _declared_host_id(args)
    ceiling = _wait_clamp_ceiling(host_id)
    if ceiling is None:
        return resolved
    capped_host, cap_s = ceiling
    # A margin below the cap, never the cap itself: the poll spends process
    # start, `load_config` and the checkpoint read before `--wait` begins
    # sleeping, and `_wait_for_terminal` re-checks on a 5 s grid, so it returns a
    # little after its own deadline. The floor is arithmetic hygiene for a
    # hypothetical host whose cap is under the margin — a wait clamped to zero
    # would return instantly, which is exactly what `poll_s: 0`'s rejection
    # exists to prevent.
    clamped = max(float(cap_s) - WAIT_CAP_MARGIN_S, WAIT_POLL_INTERVAL_S)
    if resolved <= clamped:
        return resolved
    if explicit:
        unrouted = (
            "" if host_id
            else " (no --host and no $MULTI_LLM_HOST, so the clamp used the "
                 "tightest declared cap)"
        )
        print(
            f"NOTE: --timeout {resolved:.0f}s is above host {capped_host}'s "
            f"{cap_s}s foreground command cap{unrouted}; waiting "
            f"{clamped:.0f}s instead (the cap less the "
            f"{WAIT_CAP_MARGIN_S:.0f}s startup margin). Poll again to keep "
            "watching.",
            file=sys.stderr,
        )
    return clamped


def _wait_for_terminal(
    plan_path: Path, args: argparse.Namespace, views: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The one convenience, and it is bounded.

    Compresses the poll loop into a single turn for hosts that would rather spend
    one turn than ten; it never replaces the terminal-state check, and at timeout
    it returns the non-terminal view (exit 2) so the caller simply calls again.

    **A view is "still going" when either half of the file says so.** The
    ``run`` record is the process machine and the checkpoint is the phase
    machine, and on a re-used phase dir the two belong to *different runs*: a
    host-background run over a dir an earlier ``--detach`` run used inherits that
    predecessor's terminal ``run`` record, because nothing drops it on the
    foreground path. Keying the loop on the verdict alone therefore returned in
    0.11 s with exit 0 over a live fan-out — see :func:`_checkpoint_in_flight`
    for the incident (F1, 2026-09-20). Erring toward waiting cannot spin: the
    deadline is checked on every pass and a wait that guessed wrong costs its own
    timeout and exits 2, which is the caller's normal "poll again".

    It also takes the **before** picture that ``just finished:`` is a difference
    against. This is the only place that can: ``--wait`` is the one invocation
    that spans two readings of the same file, so it is the one invocation that
    can prove a spec finished *recently* rather than merely that it is finished.
    A loop that exits without iterating (everything was already terminal) leaves
    the views it was handed untouched, the difference is empty, and no key is
    attached — which is the right answer, not a missed case: nothing finished
    during a wait that never waited.

    **The checkpoint's vote is a vote about a *successor*, so a poll that names
    the run does not get it.** The phase half of a failed or aborted run is
    routinely left wherever the failure found it — ``state: models_running``,
    ``progress.step: fanout``, because nothing walks those fields back — so a
    caller polling ``--run-id <that run> --wait`` was held for the whole resolved
    timeout (90 s by default) before being told about an outcome that had been on
    disk since before the call. That is the mirror image of F1, not a softening
    of it: there the terminal record belonged to a *predecessor* and the live
    checkpoint to the run being watched, and the checkpoint deserved the vote
    precisely because the two halves described different runs. Here the caller
    has named the record's own id, so there is no second run left for the stale
    phase fields to be describing. Only an identified run is excused
    (:func:`_names_this_run`) — ``--phase`` and ``--question`` name a re-used
    directory rather than a run, and a bare poll names nothing at all, so every
    shape F1 covers keeps waiting exactly as it did.

    **``no_run`` is not a vote either way, so the checkpoint casts the only one.**
    The verdict is non-terminal purely because there is nothing to be terminal
    about — every foreground phase (all of them on Claude Code) leaves a
    checkpoint and no ``run`` record — and reading that as "still going" made a
    *finished* foreground phase hold every poll for its whole deadline and then
    exit 2 with a ``next`` that polls it again: a loop nothing in it can ever
    end (F2, 2026-09-21). Asking :func:`_checkpoint_in_flight` instead loses
    nothing, because on this lane the checkpoint is the only half of the file
    that was ever written: a live foreground fan-out still holds the wait to its
    deadline (acceptance criterion 1), and a completed one now comes back at
    once.
    """
    baseline = {
        str(view.get("phase_dir") or ""): _terminal_fingerprints(view)
        for view in views
    }
    deadline = time.time() + max(args.timeout, 0.0)

    def _live(view: Dict[str, Any]) -> bool:
        verdict = view["verdict"]
        if (verdict not in detach.TERMINAL_VERDICTS
                and verdict != detach.VERDICT_NO_RUN):
            return True
        if _names_this_run(view, getattr(args, "run_id", None)):
            return False
        return _checkpoint_in_flight(view)

    # `not views` is a waiting state, not a terminal one. A poll issued before
    # the run has written its checkpoint — the first poll of every launch —
    # selects nothing, and a loop keyed on "any view is live" exits instantly
    # over an empty list. Both the exit code and the documented reading of it say
    # otherwise: `2` explicitly covers "no run recorded yet, poll again", so
    # returning in 0.13 s turned the one call that exists to hold a turn into a
    # spin the agent repeats as fast as its host will let it (F4, 2026-09-20).
    # Waiting costs at most this call's own deadline and then still exits 2.
    while not views or any(_live(v) for v in views):
        if time.time() >= deadline:
            break
        time.sleep(min(WAIT_POLL_INTERVAL_S, max(deadline - time.time(), 0.05)))
        views = _collect(plan_path, args)
    _attach_finished_during_wait(baseline, views)
    return views


def _abort_selector_given(args: argparse.Namespace) -> bool:
    """Did the caller name what to abort? (D4h: ``--phase``/``--run-id``/``--all``)."""
    return bool(args.phase or args.question or args.run_id or args.include_all)


def _run_abort(plan_path: Path, args: argparse.Namespace) -> int:
    """``status <plan> --abort`` — idempotent, and it says which case it hit.

    D4h spells the verb with an explicit selector and D4b's rule for a plan whose
    runs are ambiguous is that the candidates are *listed*, never silently
    resolved. A bare ``--abort`` therefore kills a single live run (the case a
    shim's cleanup step means) but refuses to fan out over several: the phase dirs
    under one plan can hold runs a different shim, or a different session,
    launched, and "stop the run I started" must not become "stop everything".
    """
    views = _collect(plan_path, args)
    if not views:
        print("Nothing to abort: no run recorded for this plan.")
        return EXIT_NOTHING_TO_DO
    live = [v for v in views if v["verdict"] not in detach.TERMINAL_VERDICTS]
    if len(live) > 1 and not _abort_selector_given(args):
        print(
            f"{len(live)} live runs under this plan and no selector: naming which "
            "one to abort is the caller's job. Candidates:"
        )
        for view in live:
            name = os.path.basename(view["phase_dir"])
            run_id = str(view["run"].get("run_id") or "")
            head = f"  {name}"
            if run_id:
                head += f"  run {run_id[:8]}"
            print(f"{head}  {_verdict_text(view)}")
            if view.get("question"):
                print(f"      question: {view['question']}")
            print(f"      abort it with: {_abort_command(plan_path, view)}")
        print(
            "  abort every one of them with: "
            + dispatcher_command("status", str(plan_path), "--abort", "--all")
        )
        return EXIT_NEEDS_HUMAN
    aborted = 0
    survived = 0
    needs_force = 0
    for view in views:
        result = detach.abort_run(view["phase_dir"], force=args.force)
        name = os.path.basename(view["phase_dir"])
        run_id = str(view["run"].get("run_id") or "?")
        if result["result"] == "aborted":
            aborted += 1
            print(f"aborted {name} run {run_id[:8]} (process group terminated)")
        elif result["result"] == "survived":
            survived += 1
            print(
                f"REFUSED TO DIE: {name} run {run_id} pid "
                f"{view['run'].get('pid')} survived SIGKILL — "
                f"log: {view['run'].get('log_path')}"
            )
        elif result["result"] == "needs_force":
            needs_force += 1
            print(
                f"{name} run {run_id[:8]}: identity probe inconclusive; re-run "
                "with --force to abort it anyway"
            )
        else:
            print(f"{name} run {run_id[:8]}: nothing was running "
                  f"({result.get('detail', view['verdict'])})")
    if survived:
        return EXIT_NEEDS_HUMAN
    if needs_force:
        return EXIT_NEEDS_HUMAN
    return EXIT_OK if aborted else EXIT_NOTHING_TO_DO


def _run_sweep(plan_path: Path, args: argparse.Namespace) -> int:
    """Close out orphans across every phase dir under this plan, deliberately."""
    phase = _resolve_phase(args.phase)
    closed: List[str] = []
    for directory in discover_phase_dirs(plan_path, phase=phase):
        before = status_file.read_run(str(directory)).get("state")
        view = detach.classify_run(str(directory), close_out=True)
        after = view["run"].get("state")
        if before != after:
            closed.append(f"{os.path.basename(str(directory))}: {before} -> {after}")
    if closed:
        print("swept:")
        for line in closed:
            print(f"  {line}")
    else:
        print("swept: nothing to close out")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - direct invocation
    sys.exit(main(sys.argv[1:]))
