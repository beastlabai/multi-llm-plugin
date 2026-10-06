#!/usr/bin/env python3
"""``util batch-done``: the native write leg's per-batch progress record.

The four write phases' native leg hands its batches to the host, which
applies them one at a time. Until 2026-10-05 nothing on that leg recorded
which batches had finished: the orchestrator recorded the phase *started*
at emit, ``util next-command`` offered ``--<mode> <plan> --resume`` for it,
and that ``--resume`` — filtering on a ``processed_<phase>`` map nothing had
ever written — printed ``[resume] 0 already processed`` and re-emitted every
batch under the same ``prior_changes_id``, so a host following the offered
command re-applied batches to an already-edited plan (reviewer repro,
fix round 5).

This command is the record. The host runs it once per batch, right after
it verified the batch's outcome and before it appends the batch's summary
to the prior-changes log, and it writes the same ``state.json`` keys the
executor leg's ``WriteBatchScheduler`` writes for the same batch:

* ``task_status[<task id>]`` = ``completed`` | ``failed`` (plus
  ``task_status_reasons`` when ``--reason`` is given) — what
  ``check_workflow_prerequisites`` reads as the phase's progress and what
  ``implement --resume`` filters its tasks on;
* for the three apply kinds, ``processed_<phase dir word>[<member id>]`` for
  every id in the task's ``expected_ids`` (``applied`` | ``failed``) — the
  map ``ApplyOrchestratorBase`` step 23 drops groups on under ``--resume``.
  A ``failed`` record keeps the batch eligible for re-dispatch, as the
  executor leg re-dispatches a failed batch;
* for the two code kinds (``implement``, ``apply_code_fixes``),
  ``tracked_files`` from the dirty-tree census, as the scheduler records it
  after each batch (:func:`_tracked_after_batch`).

It edits ``state.json`` as a file (read, mutate through the state manager's
module-level helpers, atomic replace) under ``metrics.state_file_lock`` —
the host may record same-batch tasks in parallel — never through a
``StateManager`` —
which would read the plan the batch just edited as a human's mid-workflow
edit — and preserves every other key: the host's closing ``--mark-completed``
re-reads the same file.

:func:`unrecorded_logged_batches` is the other half, shared by both native
emit paths (``apply_orchestrator_base``, ``implement_orchestrator``): on a
native ``--resume``, a batch about to be emitted whose ``prior_changes_id``
already has an entry in this phase's log and which no record says finished
is one a host applied without recording. Re-emitting it would apply it
twice, so the resume is refused (exit 3) naming the batch, this command and
``--fresh``.

Usage::

    multi_llm.py util batch-done --phase-dir <dir> --task-id <id> \\
        --status completed|failed [--reason TEXT] [--state-file PATH] \\
        [--check passed|findings|could-not-run [--finding TEXT ...]]

``--check`` is a check task's structured result (an implement task that
declares no files; its reply's ``check`` object): ``findings`` records each
``--finding`` in ``state.json``'s ``task_check_findings`` beside a
``completed`` status, which ``status``, the result notice and review-code's
implementation record read (racing re-run 4, 2026-10-05). It also holds back
the tasks that depend on the check (fix round 11): each one with no record
yet is recorded ``pending`` with ``not run — upstream check <id> reported
problems``, the ``[CHECK_BLOCKED]`` line names them, ``util write-prompt``
refuses them, and ``util finalize-tracking --mark-completed`` refuses while
one is left — the plugin's enforcement of what the instructions tell the
host, as the executor leg enforces it in its loop. Any later record of the
check (``--check passed``, ``--status failed``) releases them.
``could-not-run`` (fix round 12) is a check whose reply said
``check.could_not_run: true``: recorded ``--status failed`` (its reason
defaults to saying so), never as findings; a ``failed`` reply that lists
findings without it ran, and is recorded ``--status completed --check
findings``. A batch that edits code (implement's tasks that declare files,
apply-code-fixes' batches; fix round 14) records the checks it ran on its
own edits the same way, ``--check passed|findings``: a failing one prints a
``[BATCH_CHECK_FAILED]`` line telling the host to list it in summary.md and
report it, and holds nothing back.

Import is side-effect free: argparse and all I/O live inside :func:`main`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from . import prior_changes as _prior_changes
    from .metrics import (
        _read_state, _write_state, restore_to_progress_in_state,
        state_file_lock, withdraw_from_progress_in_state,
    )
    from .state_manager import (
        TASK_RESET_AT_KEY,
        _parse_completion_time,
        mark_item_processed_in_state,
        set_task_batch_check_in_state,
        set_task_check_findings_in_state,
        set_task_check_tree_in_state,
        update_task_status_in_state,
    )
    from .stream_bootstrap import bootstrap_streams
    from .subagent_executor import check_findings_from
    from .subagent_tasks import PHASE_BY_WRITE_KIND
    from .write_phase_bridge import WRITE_TASK_FILENAME
    from .write_batch_scheduler import (
        BATCH_CHECK_MARKER,
        batch_check_line,
        BLOCKED_BY_CHECK_PREFIX,
        CODE_WRITE_KINDS,
        blocked_by_check_reason,
        decomposition_graph,
        is_check_task,
        rerunnable_checks,
        transitive_dependents,
        upstream_held_check,
    )
    from .write_prompt import (
        WritePromptError, changed_since_bound, check_tree_fingerprint,
        discard_check_tree, load_write_task,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from utils import prior_changes as _prior_changes  # type: ignore
    from utils.metrics import (  # type: ignore
        _read_state, _write_state, restore_to_progress_in_state,
        state_file_lock, withdraw_from_progress_in_state,
    )
    from utils.state_manager import (  # type: ignore
        TASK_RESET_AT_KEY,
        _parse_completion_time,
        mark_item_processed_in_state,
        set_task_batch_check_in_state,
        set_task_check_findings_in_state,
        set_task_check_tree_in_state,
        update_task_status_in_state,
    )
    from utils.stream_bootstrap import bootstrap_streams  # type: ignore
    from utils.subagent_executor import check_findings_from  # type: ignore
    from utils.subagent_tasks import PHASE_BY_WRITE_KIND  # type: ignore
    from utils.write_phase_bridge import WRITE_TASK_FILENAME  # type: ignore
    from utils.write_batch_scheduler import (  # type: ignore
        BATCH_CHECK_MARKER,
        batch_check_line,
        BLOCKED_BY_CHECK_PREFIX,
        CODE_WRITE_KINDS,
        blocked_by_check_reason,
        decomposition_graph,
        is_check_task,
        rerunnable_checks,
        transitive_dependents,
        upstream_held_check,
    )
    from utils.write_prompt import (  # type: ignore
        WritePromptError, changed_since_bound, check_tree_fingerprint,
        discard_check_tree, load_write_task,
    )


__all__ = [
    "CHECK_BLOCKED_MARKER",
    "CHECK_RESULTS",
    "MARKER",
    "RESUME_REFUSED_MARKER",
    "STATUSES",
    "record_batch",
    "unrecorded_logged_batches",
    "resume_refusal_lines",
    "main",
]

#: The one stdout line a successful record prints.
MARKER = "[BATCH_DONE]"

#: The line naming the tasks a check with findings holds back (stdout).
CHECK_BLOCKED_MARKER = "[CHECK_BLOCKED]"

#: The first line of a refused native ``--resume`` (stderr).
RESUME_REFUSED_MARKER = "[RESUME_REFUSED]"

#: The two outcomes the host records; the scheduler's own two words.
STATUSES = ("completed", "failed")

#: ``task_status`` values that say a task's outcome was recorded. Anything
#: else (absent, or the ``pending`` the executor seeds / a cancelled run
#: leaves) is "no record".
_RECORDED_TASK_STATUSES = frozenset({"completed", "failed", "skipped"})

#: ``processed_<phase>`` status per recorded outcome.
_ITEM_STATUS = {"completed": "applied", "failed": "failed"}

#: ``--check``'s words: what a check task's reply said in its ``check``
#: object (``passed: true`` and no finding, problems found, or — fix round
#: 12 — ``could_not_run: true``).
CHECK_RESULTS = ("passed", "findings", "could-not-run")

#: The reason ``--check could-not-run`` records when ``--reason`` gives none.
COULD_NOT_RUN_REASON = (
    "its check could not run (the reply said check.could_not_run); fix the "
    "cause and run implement with --resume"
)


def _processed_phase_word(phase_dir: Path) -> str:
    """The key suffix of ``processed_<…>``: the phase DIR word.

    ``ApplyOrchestratorBase`` keys the map on its ``phase_name``, which is
    the directory it writes under (``apply-fixes`` for apply-code-fixes,
    whose mode word the task file carries instead).
    """
    return phase_dir.name


def _tracked_after_batch(
    state: Dict[str, Any], target: Path, kind: str,
) -> Optional[List[Dict[str, str]]]:
    """``tracked_files`` after a code phase's batch, or ``None`` to leave it.

    The native leg's counterpart of the scheduler's per-batch record: a host
    that stops part-way, before the closing ``util finalize-tracking``, has
    still recorded what its finished batches changed, so review-code does
    not review what an earlier phase left there (2026-10-05, Codex racing
    run 3). The census alone — this command is given no file list; the
    closing ``finalize-tracking --touched …`` adds the verified files — and
    only for the two code kinds: the plan and the task list the planning
    phases edit are never tracked. Never fatal: the batch record is this
    command's job.
    """
    try:
        from .write_batch_scheduler import CODE_WRITE_KINDS, load_finalize_tracking
    except ImportError:  # pragma: no cover - direct script invocation
        from utils.write_batch_scheduler import (  # type: ignore
            CODE_WRITE_KINDS, load_finalize_tracking,
        )
    if kind not in CODE_WRITE_KINDS:
        return None
    finalize_tracking = load_finalize_tracking()
    if finalize_tracking is None:
        return None
    try:
        return finalize_tracking.phase_tracked_files(
            state, [], state_path=target, phase=PHASE_BY_WRITE_KIND[kind],
            say=lambda line: None,
        )
    except Exception:  # noqa: BLE001 - bookkeeping must not fail the record
        return None


def record_batch(
    phase_dir: Path,
    task_id: str,
    status: str,
    *,
    reason: Optional[str] = None,
    state_file: Optional[Path] = None,
    check: Optional[str] = None,
    findings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Record *task_id*'s outcome; the ``{task_id, status, items, state_file,
    check_findings}`` written.

    *check* / *findings* are a check task's structured result (its reply's
    ``check`` object, racing re-run 4): ``findings`` records them in
    ``state.json``'s ``task_check_findings`` beside a ``completed`` status,
    as the executor leg does; every other record of an implement task clears
    that task's entry.

    Raises :class:`WritePromptError` (exit 3) for a missing task file, an
    unknown id, a human task, a status outside :data:`STATUSES`, or a check
    result that does not fit the task or the status.
    """
    if status not in STATUSES:
        raise WritePromptError(
            f"--status must be one of {', '.join(STATUSES)}, not {status!r}."
        )
    if check is not None and check not in CHECK_RESULTS:
        raise WritePromptError(
            f"--check must be one of {', '.join(CHECK_RESULTS)}, not {check!r}."
        )
    if findings and check != "findings":
        raise WritePromptError(
            "--finding is recorded only with --check findings: pass "
            "--check findings and one --finding per problem the check reported."
        )
    if check == "findings" and status != "completed":
        raise WritePromptError(
            "A check that ran and reported problems is recorded "
            "--status completed with --check findings, also when its reply "
            "said \"status\": \"failed\" but listed findings and did not say "
            "\"could_not_run\": true: the tasks that do not depend on it go "
            "on and review-code is told about the findings. Use --status "
            "failed --check could-not-run for a check whose reply said "
            "\"could_not_run\": true, and --status failed (without --check) "
            "for one that failed listing no finding or whose subagent changed "
            "a file."
        )
    if check == "could-not-run" and status != "failed":
        raise WritePromptError(
            "A check that could not run is recorded --status failed with "
            "--check could-not-run: it says nothing about the code, so it is "
            "never a pass or a finding, and implement stops on it."
        )
    phase_dir = Path(phase_dir).resolve()
    document, task = load_write_task(phase_dir, task_id)
    kind = str(document.get("kind") or "")
    if kind not in PHASE_BY_WRITE_KIND:
        raise WritePromptError(
            f"{phase_dir / WRITE_TASK_FILENAME} is a {kind!r} task file, not a "
            f"write phase's (known: {', '.join(PHASE_BY_WRITE_KIND)})."
        )
    if task.get("is_human") or str(task.get("executor_role") or "") == "human":
        raise WritePromptError(
            f"Task {task_id} is a human task: it never dispatches to a "
            f"subagent, so there is no batch outcome to record. Answer it "
            f"through the phase's human-task strategy instead."
        )
    checking = is_check_task(kind, task)
    if check is not None and not checking:
        # Fix round 14: a code batch reports the checks it ran on its own
        # edits in the same `check` object; a plan or task-list batch has none.
        if kind not in CODE_WRITE_KINDS:
            raise WritePromptError(
                f"Task {task_id} edits the plan or the task list, not code, "
                f"so it has no check result: record it without --check."
            )
        if check == "could-not-run":
            raise WritePromptError(
                f"Task {task_id} is not a check task (a check task is an "
                f"implement task that declares no files): its reply's `check` "
                f"reports the checks it ran on its own edits, so record --check "
                f"passed, or --check findings with one --finding per failing "
                f"check, or leave --check out when it reported none."
            )
    if checking and status == "completed":
        changed = changed_since_bound(
            phase_dir, task_id,
            [t for t in (document.get("batches") or document.get("tasks") or [])
             if isinstance(t, dict)],
        )
        if changed:
            raise WritePromptError(_changed_a_file_message(phase_dir, task_id, changed))
    check_findings: Optional[List[str]] = None
    batch_check: Optional[List[str]] = None
    if check == "could-not-run":
        reason = reason or COULD_NOT_RUN_REASON
    elif check is not None:
        read = check_findings_from({
            "check": {"passed": check == "passed", "findings": list(findings or [])}
        }) or []
        if checking:
            check_findings = read
        else:
            # A code batch's own check: reported, never a hold (fix round 14).
            batch_check = read
    # The tree a check with findings leaves (fix round 14): a native
    # `--resume` hands the check out again only once the files moved since,
    # as the executor does from its verdict's stamp. `util finalize-tracking`
    # re-stamps it with the tree the run ends on.
    tree = check_tree_fingerprint(phase_dir) if check_findings else None
    target = Path(state_file) if state_file else phase_dir.parent / "state.json"
    if not target.is_file():
        raise WritePromptError(
            f"No state file at {target}: the phase's orchestrator writes it "
            f"when it emits its batches (pass --state-file if it lives elsewhere)."
        )
    # One lock from the read to the write (fix round 8): the host may record
    # same-batch tasks in parallel, and the census below takes seconds, so an
    # unlocked pair read the same file and the second write dropped the
    # first one's `task_status`.
    with state_file_lock(target):
        state = _read_state(target)
        if not isinstance(state, dict):
            raise WritePromptError(f"{target} is not a JSON object.")

        update_task_status_in_state(state, task_id, status, reason)
        blocked: List[str] = []
        if kind == "implement":
            set_task_check_findings_in_state(state, task_id, check_findings)
            set_task_check_tree_in_state(state, task_id, tree)
            # A code task's own failing check, on record beside the check
            # findings and never among them (fix round 15): it holds
            # nothing back, and every other record of the task clears it.
            set_task_batch_check_in_state(state, task_id, batch_check)
            blocked = _hold_or_release_dependents(
                state, document, task_id, held=bool(check_findings),
            )
            _settle_eta_total(state, document, task_id, status, reason, blocked)
        tracked = _tracked_after_batch(state, target, kind)
        if tracked is not None:
            state["tracked_files"] = tracked
        items: List[str] = []
        if kind != "implement":
            phase_word = _processed_phase_word(phase_dir)
            details = {
                "task_id": task_id,
                "prior_changes_id": str(task.get("prior_changes_id") or ""),
            }
            for item_id in task.get("expected_ids") or []:
                item_id = str(item_id)
                if not item_id:
                    continue
                mark_item_processed_in_state(
                    state, phase_word, item_id, _ITEM_STATUS[status], details
                )
                items.append(item_id)
        state["updated_at"] = datetime.now().isoformat()
        _write_state(target, state)
    if checking:
        # One record verifies one dispatch: a later record of this check
        # without a new `util write-prompt` has nothing to compare against.
        discard_check_tree(phase_dir, task_id)
    return {
        "task_id": task_id, "status": status, "items": items,
        "state_file": str(target), "check_findings": check_findings or [],
        "batch_check": batch_check or [],
        "blocked": blocked,
    }


def _changed_a_file_message(phase_dir: Path, task_id: str, changed: List[str]) -> str:
    """The refusal of a ``completed`` record for a check that changed files."""
    shown = ", ".join(changed[:10]) + (
        f" (+{len(changed) - 10} more)" if len(changed) > 10 else ""
    )
    return (
        f"{task_id} is a check task (it declares no files), and since its "
        f"prompt was bound with util write-prompt these files changed, which "
        f"no other task in {WRITE_TASK_FILENAME} declares: {shown}. A check "
        f"whose subagent changed a file did not only check, so it is not "
        f"recorded completed (passed or findings): record it --status failed "
        f"--reason \"its subagent changed {changed[0]}\" and keep or undo the "
        f"change. If something other than its subagent changed them (you, or "
        f"a task running beside it), bind its prompt again with util "
        f"write-prompt --phase-dir \"{phase_dir}\" --task-id {task_id} and run "
        f"the check again. Nothing was recorded."
    )


#: How a ``--reason`` starts when ``util write-prompt`` refused the task, so
#: no subagent ran it (``[WRITE_PROMPT_REFUSED]`` prints the record to make).
REFUSED_BEFORE_DISPATCH_REASON = "refused before dispatch"


def _settle_eta_total(
    state: Dict[str, Any], document: Dict[str, Any], task_id: str,
    status: str, reason: Optional[str], held: List[str],
) -> None:
    """Take out of this run's ``[ETA]`` total the handed-out tasks this record
    keeps from being dispatched (fix round 18, live run 7 #2).

    The native emit's ``tasks_to_dispatch`` — what the host passes as
    ``--total-batches``, one ``util metrics record`` per task it dispatches —
    counts every task in the batches, a task waiting on a check's verdict
    included. A record that settles one of them as never dispatched in this
    run takes it out (``metrics.withdraw_from_progress_in_state``):

    * the tasks a check with findings holds (*held*, the ``[CHECK_BLOCKED]``
      list);
    * after a ``failed`` record, every task that depends on it and has no
      record yet — the host skips them (``instructions/implement.md`` step
      6) — and the task itself when ``util write-prompt`` refused it, since
      no subagent ran.

    Only tasks of this phase's ``write_tasks.json`` (what was handed out) and
    never a manual task (``tasks_to_dispatch`` does not count one). A record
    of the same task again first puts back what its earlier record took out.
    Nothing happens when the host never ran ``util metrics start``.
    """
    phase = PHASE_BY_WRITE_KIND.get(str(document.get("kind") or ""), "implement")
    restore_to_progress_in_state(state, phase, task_id)
    handed = {
        str(task.get("task_id") or ""): task
        for task in (document.get("batches") or document.get("tasks") or [])
        if isinstance(task, dict)
    }

    def manual(task: Dict[str, Any]) -> bool:
        return bool(task.get("is_human")) or str(task.get("executor_role") or "") == "human"

    out = [t for t in held if t in handed and not manual(handed[t])]
    if status == "failed":
        if str(reason or "").strip().lower().startswith(REFUSED_BEFORE_DISPATCH_REASON):
            out.append(task_id)
        statuses = state.get("task_status") or {}
        graph = decomposition_graph(list(handed.values()), state)
        for dependent in transitive_dependents(graph, task_id):
            task = handed.get(dependent)
            if task is None or manual(task):
                continue
            if statuses.get(dependent) in (None, "pending"):
                out.append(dependent)
    withdraw_from_progress_in_state(state, phase, out, task_id)


def _hold_or_release_dependents(
    state: Dict[str, Any], document: Dict[str, Any], task_id: str, *, held: bool,
) -> List[str]:
    """Record (or release) the tasks a check with findings holds back.

    *held*: the check was just recorded with findings, so every transitive
    dependent with no status yet, or a ``pending`` one, is recorded
    ``pending`` with :func:`blocked_by_check_reason` — the reason the
    executor leg writes for the same tasks. Otherwise each dependent still
    ``pending`` under this check's reason loses it: the check passed, failed
    or was recorded again, and is no longer why it waits — unless another
    check with findings is still upstream of it, whose reason it then
    carries (fix round 12: with two checks holding D, recording the first
    passed cleared D's reason while the second's findings stood). The walk
    is over the whole decomposition (``write_batch_scheduler.decomposition_graph``:
    the task file plus ``state.json``'s recorded graph), not only the tasks
    a resume put in the file (fix round 12). Returns the tasks held (empty
    on a release).
    """
    tasks = decomposition_graph(
        document.get("batches") or document.get("tasks") or [], state,
    )
    by_id = {
        str(t.get("task_id") or ""): t for t in tasks if isinstance(t, dict)
    }
    statuses = state.get("task_status") or {}
    reasons = state.get("task_status_reasons") or {}
    mine = f"{BLOCKED_BY_CHECK_PREFIX}{task_id} "
    still_held = [] if held else [c for c in rerunnable_checks(state) if c != task_id]
    held_ids: List[str] = []
    for dependent in transitive_dependents(tasks, task_id):
        task = by_id.get(dependent, {})
        if task.get("is_human") or str(task.get("executor_role") or "") == "human":
            continue
        status = statuses.get(dependent)
        if held:
            if status in (None, "pending"):
                update_task_status_in_state(
                    state, dependent, "pending", blocked_by_check_reason(task_id),
                )
                held_ids.append(dependent)
        elif status == "pending" and str(reasons.get(dependent) or "").startswith(mine):
            other = upstream_held_check(tasks, dependent, still_held)
            update_task_status_in_state(
                state, dependent, "pending",
                blocked_by_check_reason(other) if other else None,
            )
    return held_ids


def _logged_ids(output_dir: str, phase: Optional[str]) -> set:
    return set(_logged_at(output_dir, phase))


def _logged_at(output_dir: str, phase: Optional[str]) -> Dict[str, Optional[str]]:
    """``{entry id: timestamp of its newest line}`` in this phase's log."""
    scoped = _prior_changes.canonical_phase(phase)
    return {
        str(entry.get("id") or ""): entry.get("timestamp")
        for entry in _prior_changes.entries(output_dir)
        if _prior_changes.canonical_phase(entry.get("phase")) == scoped
    }


def _logged_before_reset(logged_at: Any, reset_at: Any) -> bool:
    """Whether a log line was written no later than its task's reset.

    False when either time is missing or unreadable: the line then counts as
    this run's, which keeps the refusal (absence of a date is not a reset).
    """
    logged = _parse_completion_time(logged_at)
    reset = _parse_completion_time(reset_at)
    return logged is not None and reset is not None and logged <= reset


def unrecorded_logged_batches(
    tasks: Iterable[Dict[str, Any]],
    *,
    output_dir: str,
    phase: str,
    state: Dict[str, Any],
    processed_phase: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """The tasks a native ``--resume`` must not re-emit: logged, never recorded.

    *tasks* are the batches the emit path is about to hand out (already
    filtered by the phase's own resume rule); *phase* is the log's phase
    word (``prior_changes.canonical_phase`` folds ``apply-code-fixes`` and
    ``apply-fixes`` together); *processed_phase* is the ``processed_<…>``
    suffix for an apply kind (defaults to *phase*); *state* is the raw
    ``state.json`` dict.

    A task is recorded when ``task_status`` holds a terminal word for its id
    or, for an apply kind, when any of its ``expected_ids`` is in the
    processed map (``failed`` included: that record keeps the batch eligible
    and says the log's failure entry is accounted for). A log line no newer
    than the task's ``state_manager.TASK_RESET_AT_KEY`` stamp is the run's
    before a start-over or a ``--task`` rerun handed the task out again, and
    does not count (fix round 17). A line the run since wrote with the same
    summary is not appended again (``prior_changes.append``), so such a task
    is handed out again rather than refused.
    """
    logged = _logged_at(output_dir, phase)
    if not logged:
        return []
    task_status = state.get("task_status") or {}
    processed = state.get(f"processed_{processed_phase or phase}") or {}
    reset_at = state.get(TASK_RESET_AT_KEY)
    reset_at = reset_at if isinstance(reset_at, dict) else {}
    unrecorded: List[Dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if task.get("is_human") or str(task.get("executor_role") or "") == "human":
            continue
        entry_id = str(task.get("prior_changes_id") or "")
        if not entry_id or entry_id not in logged:
            continue
        task_id = str(task.get("task_id") or "")
        if task_status.get(task_id) in _RECORDED_TASK_STATUSES:
            continue
        if any(str(i) in processed for i in (task.get("expected_ids") or [])):
            continue
        if _logged_before_reset(logged[entry_id], reset_at.get(task_id)):
            # Fix round 17: an earlier run's line, for a task a start-over
            # or a `--task` rerun handed out again since — not one this run
            # applied and left unrecorded.
            continue
        unrecorded.append(task)
    return unrecorded


def resume_refusal_lines(
    unrecorded: List[Dict[str, Any]], *, phase_dir: str, mode_word: str,
) -> List[str]:
    """The stderr block a refused native ``--resume`` prints."""
    lines = [
        f"{RESUME_REFUSED_MARKER} --resume would re-emit {len(unrecorded)} "
        f"batch(es) the prior-changes log says already ran, but no progress "
        f"record says they finished — re-applying them would edit the target "
        f"twice:",
    ]
    for task in unrecorded:
        lines.append(
            f"  - {task.get('task_id')} ({task.get('prior_changes_id')}): "
            f"{task.get('task_title') or ''}".rstrip()
        )
    lines.append(
        "Record each one as the host left it, then re-run with --resume:"
    )
    for task in unrecorded:
        lines.append(
            f"  multi_llm.py util batch-done --phase-dir \"{phase_dir}\" "
            f"--task-id {task.get('task_id')} --status completed|failed"
        )
    lines.append(
        f"Or restore the target to its pre-run bytes and start the phase over "
        f"with `{mode_word} <plan> --fresh`."
    )
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi_llm.py util batch-done",
        description=(
            "Record one write batch's outcome in state.json — the native "
            "leg's counterpart of the executor's per-batch status write, "
            "and what a later --resume skips or retries."
        ),
    )
    parser.add_argument(
        "--phase-dir", required=True,
        help="The phase directory holding write_tasks.json (e.g. "
             "plans/my-plan/apply-suggestions, plans/my-plan/implement).",
    )
    parser.add_argument(
        "--task-id", required=True,
        help="The task id from write_tasks.json (apply-batch-0, "
             "apply-task-batch-0, apply-fix-batch-0, T001, ...).",
    )
    parser.add_argument(
        "--status", required=True, choices=STATUSES,
        help="completed: the batch's edits verified on disk (or a verified "
             "no-op); failed: the subagent failed and the target is unchanged.",
    )
    parser.add_argument(
        "--reason", default=None,
        help="One line on why a failed batch failed (recorded beside the status).",
    )
    parser.add_argument(
        "--state-file", default=None, metavar="PATH",
        help="The plan's state.json; default: <phase-dir>/../state.json.",
    )
    parser.add_argument(
        "--check", default=None, choices=CHECK_RESULTS,
        help="For a check task (an implement task that declares no files): "
             "its reply's check result — passed, findings (the check ran "
             "and reported problems; recorded with --status completed), or "
             "could-not-run (its check said could_not_run: true; recorded "
             "with --status failed). For a batch that edits code (implement, "
             "apply-code-fixes): the checks it ran on its own edits, passed "
             "or findings, with --status completed.",
    )
    parser.add_argument(
        "--finding", action="append", default=None, metavar="TEXT",
        help="One problem a check reported (repeatable; with --check findings).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    bootstrap_streams()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        written = record_batch(
            Path(args.phase_dir), args.task_id, args.status,
            reason=args.reason,
            state_file=Path(args.state_file) if args.state_file else None,
            check=args.check,
            findings=args.finding,
        )
    except WritePromptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return int(getattr(exc, "exit_code", 3) or 3)
    except OSError as exc:
        print(f"ERROR: cannot update state file: {exc}", file=sys.stderr)
        return 1
    items = written["items"]
    suffix = f" ({len(items)} item(s) recorded)" if items else ""
    found = written.get("check_findings") or []
    if found:
        suffix += f" ({len(found)} check finding(s) recorded)"
    print(f"{MARKER} {written['task_id']} {written['status']}{suffix}")
    failing = written.get("batch_check") or []
    if failing:
        # Fix round 14: the native leg's counterpart of the executor's line.
        print(
            f"{batch_check_line(written['task_id'], failing)}. The edits are "
            f"kept. List it under "
            f"## Check Findings in the phase's summary.md, and say in your "
            f"final report that this check fails: never report the phase as a "
            f"clean success."
        )
    blocked = written.get("blocked") or []
    if blocked:
        task_id = written["task_id"]
        print(
            f"{CHECK_BLOCKED_MARKER} {', '.join(blocked)} "
            f"{'depends' if len(blocked) == 1 else 'depend'} on {task_id}, "
            f"whose check reported problems: do not dispatch them (their "
            f"`util write-prompt` refuses) and record nothing for them — they "
            f"stay pending. Run the tasks that do not depend on {task_id}, then "
            f"close the phase without --mark-completed and report the way "
            f"forward: fix what {task_id} reported and re-run implement with "
            f"--resume (the check runs again), or accept its findings with "
            f"--resume --human-task {task_id}=done (they then run, told that "
            f"the findings were accepted), or leave it and them "
            f"undone with --resume --human-task {task_id}=skip."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
