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
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import cli_common, detach, metrics, process_control, status_file
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
    record_next,
    record_notice,
    record_state,
)
from .output_handler import output_dir_path, phase_dir_path, slugify_question
from .stream_bootstrap import bootstrap_streams

#: ``--wait``'s default. Deliberately under the tightest surveyed host shell cap
#: (OpenCode's ~120 s): it compresses the poll loop into one turn, it never
#: replaces the terminal-state check, and it is never emitted for a host whose
#: shell cap is below its own timeout.
DEFAULT_WAIT_TIMEOUT_S = 90.0

#: How often ``--wait`` re-reads the status file. The heartbeat is 10 s, so a
#: faster cadence learns nothing; 5 s keeps the returned view fresh.
WAIT_POLL_INTERVAL_S = 5.0

VIEW_WORKFLOW = "workflow"
VIEW_RUN = "run"
VIEW_ALL = "all"


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
        if child.name == "ask":
            found.extend(_ask_dirs(plan_dir))
            continue
        if (child / status_file.STATUS_FILENAME).exists():
            found.append(child)
    return found


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
        lines.append(f"      run.state:      {run.get('state') or '(none)'}")
        lines.append(f"      phase state:    {view.get('state') or '(none)'}")
        if workflow is not None:
            phase_name = view.get("phase") or name
            wf = (workflow.get("phases") or {}).get(phase_name, {})
            lines.append(
                f"      workflow phase: {wf.get('status', 'not tracked')}"
            )
            if (
                view["verdict"] == detach.VERDICT_DONE
                and wf.get("status") == "pending"
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


def _progress_text(progress: Dict[str, Any]) -> str:
    parts = []
    if progress.get("step"):
        parts.append(str(progress["step"]))
    if progress.get("models_total"):
        parts.append(f"models {progress.get('models_done', 0)}/{progress['models_total']}")
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
    return ", ".join(parts) or "(none recorded)"


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
        elapsed = detach._elapsed_since(row.get("dispatched_at"))
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
            "The exit code IS the poll-loop condition; poll again in ~15s on 2, "
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
        "--timeout", type=float, default=DEFAULT_WAIT_TIMEOUT_S,
        help="--wait timeout in seconds",
    )
    parser.add_argument(
        "--abort", action="store_true",
        help="Stop the selected detached run(s): identity probe, process-group "
             "SIGTERM, 10s, SIGKILL, then run.state: aborted. The only kill path.",
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
    if resolved is None:
        print(
            f"ERROR: unknown phase or mode '{word}'. Valid phases: "
            + ", ".join(modes.phase_names()),
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


def _collect(plan_path: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    phase = _resolve_phase(args.phase)
    views = [
        detach.classify_run(str(directory))
        for directory in discover_phase_dirs(plan_path, phase=phase)
    ]
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
    if args.wait:
        views = _wait_for_terminal(plan_path, args, views)

    exit_code = exit_code_for(views)

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

    if terminal_notice:
        record_notice(terminal_notice)

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
    if exit_code == EXIT_NOTHING_TO_DO and views:
        # `views[0]` is `discover_phase_dirs`' alphabetical order, not "the run
        # that made the exit code 2" — with several phases listed, a finished
        # one can sort first while a different one is the live reason for the
        # exit code (F3, 2026-09-16: `apply-suggestions` sorted ahead of the
        # still-running `generate-tasks`, and the poll command handed back
        # polled the finished run straight back to exit 0). Pick the first
        # non-terminal view; only when none is live does the finished one
        # answer, which is the genuine "nothing to do" case.
        live_views = [v for v in views if v["verdict"] not in detach.TERMINAL_VERDICTS]
        first = live_views[0] if live_views else views[0]
        record_next(
            detach.status_command_for(
                str(plan_path),
                phase=first.get("phase"),
                run_id=str(first["run"].get("run_id") or "") or None,
                # The run's own host, so the poll this hands back reports the
                # run's runner rather than the reader's (F6, 2026-09-16).
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


def _wait_for_terminal(
    plan_path: Path, args: argparse.Namespace, views: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The one convenience, and it is bounded.

    Compresses the poll loop into a single turn for hosts that would rather spend
    one turn than ten; it never replaces the terminal-state check, and at timeout
    it returns the non-terminal view (exit 2) so the caller simply calls again.
    """
    deadline = time.time() + max(args.timeout, 0.0)
    while views and any(v["verdict"] not in detach.TERMINAL_VERDICTS for v in views):
        if time.time() >= deadline:
            break
        time.sleep(min(WAIT_POLL_INTERVAL_S, max(deadline - time.time(), 0.05)))
        views = _collect(plan_path, args)
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
