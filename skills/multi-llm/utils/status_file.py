#!/usr/bin/env python3
"""The one writer of ``{phase}/.status.json``.

This module holds the primitives that were previously inline in
``utils/review_orchestrator_base.py`` (``write_status``/``update_status``, which
still re-exports them under the same names, as do
``review_plan_orchestrator._write_status``/``_update_status``). They live in a
module of their own for one reason: ``status`` has to be a *sub-second* command
(a file read plus a liveness probe, polled by hosts with a 2-minute shell cap),
and importing the review orchestration base pulls in the provider client, the
HTML report generator and the validation stack. Nothing about the write path
changed in the move — same atomic temp+rename, same best-effort swallow, same
merge semantics — so every existing consumer and test keeps its meaning.

**The schema is additive, never forked.** ``{phase}/.status.json`` is the
orchestrator checkpoint: ``ask_orchestrator`` reads ``question`` byte-for-byte as
its slug-collision guard and ``plan_hash`` for the stale-answers warning, and
``build_unsanitize_map`` reads ``models_requested``/``models_completed`` to
restore colon-bearing model specs during reaggregation. A detached run therefore
adds keys to the *same* document under two reserved objects rather than writing a
second file with a private shape:

* ``run``: ``{run_id, pid, process_start_time, detached, host, mode, cwd,
  log_path, heartbeat_at, heartbeat_interval_s, state, exit_code, error,
  resolutions}`` — the **process** state machine
  (``starting → running → {done, failed, aborted}``).
* ``progress``: ``{step, batch_index, batches_total, tasks_done, tasks_total,
  failed_task_ids, models_done, models_total}`` — reporting only; no external
  actor advances it.

``run.state`` is **not** the top-level ``state``. The top-level field stays the
*phase* machine (``models_running`` → ``models_complete``/``grouping_complete`` →
``validation_pending`` → ``complete``) that resume and the published instruction
text match verbatim; a crashed fan-out is exactly the pair
``state: models_running`` + ``run.state: failed``, and no reader collapses one
into the other.

After the initial ``write_status`` a detached child only ever calls
``update_status`` (via :func:`update_run`): a heartbeat-shaped ``write_status``
would drop ``question``, ``plan_hash`` and ``models_requested`` and break both
ask resume and the unsanitize map.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

#: The checkpoint file name. Dot-prefixed on purpose: reaggregation globs
#: ``{phase_dir}/*.json`` and Python's ``glob`` skips leading-dot names, which is
#: the only reason this file is not parsed as a model result.
STATUS_FILENAME = ".status.json"

#: Keys whose value is a dict that is *merged* rather than replaced wholesale, so
#: a writer that knows about one sub-key cannot erase another writer's.
#: ``output_files`` was already merged this way; ``run`` and ``progress`` join it
#: on the same path (D4a).
_MERGED_OBJECTS: Tuple[str, ...] = ("output_files", "run", "progress")

#: The objects a detach run owns. A plain ``write_status`` from the phase's own
#: checkpoint path must not erase them (see :func:`write_status`).
_RESERVED_OBJECTS: Tuple[str, ...] = ("run", "progress")

#: The process state machine. ``stale`` is deliberately absent: it is a *view* of
#: a ``running`` record whose heartbeat expired, never a stored state.
RUN_STATE_STARTING = "starting"
RUN_STATE_RUNNING = "running"
RUN_STATE_DONE = "done"
RUN_STATE_FAILED = "failed"
RUN_STATE_ABORTED = "aborted"

TERMINAL_RUN_STATES = (RUN_STATE_DONE, RUN_STATE_FAILED, RUN_STATE_ABORTED)

#: Legal transitions. Anything else is a bug and is refused loudly rather than
#: merged — including any transition *out of* a terminal state, which is what
#: keeps "exactly one terminal state per run" true.
_LEGAL_TRANSITIONS: Dict[Optional[str], Tuple[str, ...]] = {
    None: (RUN_STATE_STARTING, RUN_STATE_RUNNING),
    RUN_STATE_STARTING: (RUN_STATE_STARTING, RUN_STATE_RUNNING) + TERMINAL_RUN_STATES,
    RUN_STATE_RUNNING: (RUN_STATE_RUNNING,) + TERMINAL_RUN_STATES,
    RUN_STATE_DONE: (RUN_STATE_DONE,),
    RUN_STATE_FAILED: (RUN_STATE_FAILED,),
    RUN_STATE_ABORTED: (RUN_STATE_ABORTED,),
}


def status_path(phase_dir: str) -> str:
    """Absolute-or-relative path of a phase dir's checkpoint file."""
    return os.path.join(str(phase_dir), STATUS_FILENAME)


def now_iso() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Write / merge (moved verbatim from review_orchestrator_base)
# ---------------------------------------------------------------------------

def write_status(phase_dir: str, data: dict) -> None:
    """Write a .status.json checkpoint file atomically. Best-effort.

    The reserved ``run``/``progress`` objects are carried over from disk when the
    caller does not mention them: they belong to the process that *owns* the phase
    dir, not to the phase checkpoint being written, and a fan-out phase writing its
    opening ``state: models_running`` must not erase the heartbeat of the very run
    performing it. Passing the key explicitly (including as ``None``) still wins,
    which is how :func:`claim_run` replaces a predecessor's record.
    """
    temp_path = None
    path = status_path(phase_dir)
    try:
        if any(key not in data for key in _RESERVED_OBJECTS):
            existing = read_status(phase_dir)
            for key in _RESERVED_OBJECTS:
                if key not in data and key in existing:
                    data[key] = existing[key]
    except Exception:
        pass
    try:
        data["updated_at"] = now_iso()
        temp_fd, temp_path = tempfile.mkstemp(
            dir=phase_dir, suffix='.tmp', prefix='.status_'
        )
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, path)
    except Exception:
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass


def update_status(phase_dir: str, updates: dict) -> None:
    """Merge updates into existing .status.json. Best-effort.

    ``output_files``, ``run`` and ``progress`` are merged key-by-key rather than
    replaced, so a heartbeat that carries only ``heartbeat_at`` cannot erase
    ``run.run_id`` and a progress update cannot erase a sibling counter. Every
    other key is replaced, exactly as before.
    """
    path = status_path(phase_dir)
    try:
        existing = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        if not isinstance(existing, dict):
            existing = {}
        for key in _MERGED_OBJECTS:
            incoming = updates.get(key)
            current = existing.get(key)
            if isinstance(incoming, dict) and isinstance(current, dict):
                merged = dict(current)
                merged.update(incoming)
                updates = dict(updates)
                updates[key] = merged
        existing.update(updates)
        write_status(phase_dir, existing)
    except Exception:
        try:
            write_status(phase_dir, updates)
        except Exception:
            pass


def read_status(phase_dir: str) -> Dict[str, Any]:
    """Read a phase dir's checkpoint, or ``{}`` when absent/unreadable.

    Never raises: a torn or hand-edited file must not make ``status`` crash, and
    the write path is temp+rename so a torn read is only possible if something
    else wrote the file.
    """
    path = status_path(phase_dir)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, IOError, ValueError):
        return {}


def read_run(phase_dir: str) -> Dict[str, Any]:
    """The ``run`` object of a phase dir's checkpoint, or ``{}``."""
    run = read_status(phase_dir).get("run")
    return run if isinstance(run, dict) else {}


# ---------------------------------------------------------------------------
# The reserved `run` object: compare-and-set on run_id
# ---------------------------------------------------------------------------

class RunConflict(Exception):
    """A ``run`` write was refused: wrong ``run_id`` or an illegal transition."""


def claim_run(phase_dir: str, record: Dict[str, Any]) -> None:
    """Install a brand-new ``run`` record, replacing any predecessor's.

    The *only* place the whole ``run`` object is replaced. Called once, by the
    worker, immediately after it takes ``{phase}/.lock`` — which is what makes
    the replacement safe: the lock holder is by definition the single writer, and
    a predecessor record can only be a terminal run or one whose holder failed
    the liveness+identity probe. Every write after this one is a compare-and-set
    (:func:`update_run`), so a stale process that wakes up later cannot resurrect
    itself over the new run.
    """
    # Drop the predecessor's `run` *and* `progress` wholesale: inheriting either
    # would report another run's exit code or another run's step as this one's.
    update_status(phase_dir, {"run": None, "progress": None})
    update_status(phase_dir, {"run": dict(record)})


def update_run(
    phase_dir: str,
    updates: Dict[str, Any],
    *,
    expect_run_id: str,
    extra: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> bool:
    """Compare-and-set merge into ``run``; ``True`` when the write happened.

    Refuses (returning ``False``) when the on-disk ``run.run_id`` is neither
    absent nor equal to ``expect_run_id``, or when the requested
    ``run.state`` is not a legal successor of the recorded one. Both refusals are
    the same guarantee from two directions: a run that was closed out (by
    ``status``'s heartbeat-timeout sweep, or by ``--abort``) stays closed, and a
    second writer cannot interleave its heartbeat with the holder's.
    """
    current = read_run(phase_dir)
    disk_id = current.get("run_id")
    if disk_id is not None and disk_id != expect_run_id:
        if not quiet:
            print(
                f"NOTE: refusing to update run {expect_run_id} — {phase_dir} is "
                f"now owned by run {disk_id}",
                file=sys.stderr,
            )
        return False
    new_state = updates.get("state")
    if new_state is not None:
        allowed = _LEGAL_TRANSITIONS.get(current.get("state"), ())
        if new_state not in allowed:
            if not quiet:
                print(
                    f"NOTE: refusing illegal run transition "
                    f"{current.get('state')!r} -> {new_state!r} for run "
                    f"{expect_run_id} in {phase_dir}",
                    file=sys.stderr,
                )
            return False
    payload: Dict[str, Any] = dict(extra or {})
    merged = dict(updates)
    merged.setdefault("run_id", expect_run_id)
    payload["run"] = merged
    update_status(phase_dir, payload)
    return True


def append_resolution(
    phase_dir: str, *, run_id: str, site: str, decision: str
) -> bool:
    """Record one auto-resolved decision point (D4c) on the run.

    ``site`` is a D6 table row identifier, so the table stays the single source
    of truth for what the default *was*.
    """
    current = read_run(phase_dir)
    resolutions = list(current.get("resolutions") or [])
    resolutions.append({"site": site, "decision": decision, "at": now_iso()})
    return update_run(
        phase_dir, {"resolutions": resolutions}, expect_run_id=run_id, quiet=True
    )


def update_progress(phase_dir: str, updates: Dict[str, Any]) -> None:
    """Merge into the reserved ``progress`` object (display only)."""
    update_status(phase_dir, {"progress": dict(updates)})


def is_terminal(run: Optional[Dict[str, Any]]) -> bool:
    return bool(run) and run.get("state") in TERMINAL_RUN_STATES
