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

from . import detach, status_file
from .cli_common import (
    EXIT_ERROR,
    EXIT_NEEDS_HUMAN,
    EXIT_NOTHING_TO_DO,
    EXIT_OK,
    build_parent_parser,
    dispatcher_command,
    record_artifact,
    record_next,
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
    if payload.get("suggested_next"):
        lines.append(f"  suggested next: {payload['suggested_next']}")
    for notice in payload.get("notices", []) or []:
        lines.append(f"  notice: {notice}")
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
        if run.get("resolutions"):
            lines.append("      auto-resolved while you were gone:")
            for entry in run["resolutions"]:
                lines.append(
                    f"        {entry.get('site')}: {entry.get('decision')} "
                    f"({entry.get('at')})"
                )
        if view["verdict"] in (detach.VERDICT_FAILED, detach.VERDICT_ABORTED):
            lines.append(f"      exit_code: {run.get('exit_code')}")
            lines.append(f"      error:     {run.get('error')}")
            tail = detach._log_tail(run.get("log_path") or "", 20)
            if tail:
                lines.append("      last log lines:")
                lines.append(tail)
            lines.append(f"      resume with: {_resume_command(view, plan_path)}")
    return lines


def _verdict_text(view: Dict[str, Any]) -> str:
    verdict = view["verdict"]
    age = view.get("heartbeat_age_s")
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
    return ", ".join(parts) or "(none recorded)"


def _resume_command(view: Dict[str, Any], plan_path: Path) -> str:
    mode = view["run"].get("mode") or view.get("phase") or "review-plan"
    return dispatcher_command(mode, str(plan_path), "--force")


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
    """Resolve ``--phase``/``--mode`` through the mode table (mode word or phase)."""
    if not word:
        return None
    from .cli_common import _modes

    modes = _modes()
    resolved = modes.phase_from_selector(word)
    if resolved is None:
        raise SystemExit(
            f"ERROR: unknown phase or mode '{word}'. Valid phases: "
            + ", ".join(modes.phase_names())
        )
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
        print("\n".join(lines))

    for view in views:
        for path in view["output_files"].values():
            if isinstance(path, str):
                record_artifact(path)
        if view["run"].get("log_path"):
            record_artifact(view["run"]["log_path"])
    if exit_code == EXIT_NOTHING_TO_DO and views:
        first = views[0]
        record_next(
            detach.status_command_for(
                str(plan_path),
                phase=first.get("phase"),
                run_id=str(first["run"].get("run_id") or "") or None,
            )
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


def _run_abort(plan_path: Path, args: argparse.Namespace) -> int:
    """``status <plan> --abort`` — idempotent, and it says which case it hit."""
    views = _collect(plan_path, args)
    if not views:
        print("Nothing to abort: no run recorded for this plan.")
        return EXIT_NOTHING_TO_DO
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
