#!/usr/bin/env python3
"""The write-batch scheduler (D3n): a write phase *schedules*, it does not fan out.

D3g says what one write dispatch means, D3h how it stays safe, D3i how a batch of
them is bounded. None of them owns the **loop**, and three facts — all invisible
from the read path — make a write phase a loop with state rather than a fan-out:

1. **Pattern A's "complete prompt" is complete except for one token.** Every apply
   prompt the batchers build carries a literal ``{prior_changes_context}``, whose
   value is a function of the batches that already *finished*. An executor that
   dispatched the stored prompt verbatim would send the model the brace text and
   silently drop every prior-batch summary — exactly the context that keeps batch N
   from re-applying or contradicting batch N−1. So the substitution is **late, in
   Python, per dispatch**, and it produces the *dispatched* prompt without ever
   rewriting the hash-bound task file.
2. **Implement batches are a dependency order, not a list.** Ascending
   ``batch_index``, and batch N+1 is dispatched only after batch N's D3g
   classification has completed and been recorded — ordering and verification are
   the same rule seen twice.
3. **Human tasks sit inside that graph.** ``executor_role: human`` is a per-task
   guard, not a strategy: a phase-wide strategy is resolved once, then skips
   propagate through ``depends_on``.

Everything here is stated **once** and driven by both invokers: the executor under
``--exec-subagents``, and (through ``respond=``) the simulated native invoker the
equivalence tests use. Finalization is part of the loop's contract, not a caller's
courtesy: *a run that dispatched every batch and produced no summary, no
``prior_changes`` entries and no completed phase is a **failed** run, not a partial
one.*
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import shim mirrors the rest of utils/
    from . import git_utils as _git_utils
    from . import landlock_dispatch as _landlock
    from . import prior_changes as _prior_changes
    from . import process_control as _process_control
    from . import subagent_executor as _executor
    from .providers.base import (
        SANDBOX_RUNG_PLUGIN_LANDLOCK,
        host_write_posture_consent,
        WRITE_SCOPE_PLAN_FILE,
        WRITE_SCOPE_REPO_WIDE,
        SANDBOX_RUNG_PLUGIN_LANDLOCK,
        WRITE_SCOPE_TASKS,
        WRITE_SCOPES_ON_RUNG,
        JailRequest,
    )
    from .provider_registry import (
        get_driver_provenance,
        get_max_concurrent_write_subagents,
        get_provider_max_concurrent,
        get_verify_command,
        get_verify_timeout_s,
        parse_model_spec,
    )
    from .subagent_executor import (
        OUTPUT_NAME_PATTERN_BY_KIND,
        STATUS_BATCH_FAILED,
        STATUS_OK,
        STATUS_PARTIAL,
        STATUS_POLICY_VIOLATION,
        STATUS_REJECTED,
        STATUS_SCHEMA_INVALID,
        STATUS_UNPARSEABLE,
        WRITE_APPLIED,
        WRITE_CLEAN_FAILURE,
        WRITE_NO_OP,
        WRITE_PARTIAL,
        WRITE_POLICY_VIOLATION,
        WRITE_SCHEMA_REJECTED,
        BatchReport,
        ExecutorError,
        ResultRejected,
        RunnerRejected,
        TaskOutcome,
        _atomic_write_json,
        _host_ceiling,
        _host_env_guards,
        dispatch_task,
        persist_result,
        preflight_runner,
        profile_for_kind,
        resolve_output_target,
        resolve_task_runner,
        row_outcome_label,
        check_findings_from,
        check_record,
        row_check_findings,
        validate_result,
    )
    from .subagent_tasks import (
        PHASE_BY_WRITE_KIND,
        PRIOR_CHANGES_TOKEN,
        SHARED_ARTIFACT_WRITE_KINDS,
        WRITE_KINDS,
        declared_file_set,
        snapshot_file_set,
        write_output_name,
    )
    from .state_manager import (
        IMPLEMENT_TASK_GRAPH_KEY,
        set_task_check_findings_in_state,
    )
    from .write_phase_bridge import WRITE_TASK_FILENAME
    from .write_safety import (
        DRY_RUN_FILENAME,
        MAX_SNAPSHOT_BYTES,
        AuditLog,
        PolicyViolation,
        contain_under_root,
        TaskSnapshot,
        WritePolicy,
        WriteSafetyError,
        admit_disjoint_groups,
        file_sha256,
        overlapping_pairs,
        relative_to_root,
        resolve_repo_root,
        claim_external_safety_dir,
        safety_dir,
    )
except ImportError:  # pragma: no cover - direct-script import shim
    from utils import git_utils as _git_utils
    from utils import landlock_dispatch as _landlock
    from utils import prior_changes as _prior_changes
    from utils import process_control as _process_control
    from utils import subagent_executor as _executor
    from utils.providers.base import (
        SANDBOX_RUNG_PLUGIN_LANDLOCK,
        host_write_posture_consent,
        WRITE_SCOPE_PLAN_FILE,
        WRITE_SCOPE_REPO_WIDE,
        SANDBOX_RUNG_PLUGIN_LANDLOCK,
        WRITE_SCOPE_TASKS,
        WRITE_SCOPES_ON_RUNG,
        JailRequest,
    )
    from utils.provider_registry import (
        get_driver_provenance,
        get_max_concurrent_write_subagents,
        get_provider_max_concurrent,
        get_verify_command,
        get_verify_timeout_s,
        parse_model_spec,
    )
    from utils.subagent_executor import (
        OUTPUT_NAME_PATTERN_BY_KIND,
        STATUS_BATCH_FAILED,
        STATUS_OK,
        STATUS_PARTIAL,
        STATUS_POLICY_VIOLATION,
        STATUS_REJECTED,
        STATUS_SCHEMA_INVALID,
        STATUS_UNPARSEABLE,
        WRITE_APPLIED,
        WRITE_CLEAN_FAILURE,
        WRITE_NO_OP,
        WRITE_PARTIAL,
        WRITE_POLICY_VIOLATION,
        WRITE_SCHEMA_REJECTED,
        BatchReport,
        ExecutorError,
        ResultRejected,
        RunnerRejected,
        TaskOutcome,
        _atomic_write_json,
        _host_ceiling,
        _host_env_guards,
        dispatch_task,
        persist_result,
        preflight_runner,
        profile_for_kind,
        resolve_output_target,
        resolve_task_runner,
        row_outcome_label,
        check_findings_from,
        check_record,
        row_check_findings,
        validate_result,
    )
    from utils.subagent_tasks import (
        PHASE_BY_WRITE_KIND,
        PRIOR_CHANGES_TOKEN,
        SHARED_ARTIFACT_WRITE_KINDS,
        WRITE_KINDS,
        declared_file_set,
        snapshot_file_set,
        write_output_name,
    )
    from utils.state_manager import (  # type: ignore
        IMPLEMENT_TASK_GRAPH_KEY,
        set_task_check_findings_in_state,
    )
    from utils.write_phase_bridge import WRITE_TASK_FILENAME
    from utils.write_safety import (
        DRY_RUN_FILENAME,
        MAX_SNAPSHOT_BYTES,
        AuditLog,
        PolicyViolation,
        contain_under_root,
        TaskSnapshot,
        WritePolicy,
        WriteSafetyError,
        admit_disjoint_groups,
        file_sha256,
        overlapping_pairs,
        relative_to_root,
        resolve_repo_root,
        claim_external_safety_dir,
        safety_dir,
    )


__all__ = [
    "CHECK_FINDINGS_MARKER",
    "STRATEGIES",
    "STRATEGY_PAUSE",
    "STRATEGY_SKIP_CONTINUE",
    "STRATEGY_SKIP_DEPENDENTS",
    "WRITE_BATCH_REPORT_FILENAME",
    "SchedulerResult",
    "HumanTaskPlan",
    "WriteBatchScheduler",
    "answered_done_on_record",
    "BLOCKED_BY_CHECK_PREFIX",
    "bind_prior_changes",
    "blocked_by_check_reason",
    "decomposition_graph",
    "held_back_by_checks",
    "rerunnable_checks",
    "transitive_dependents",
    "upstream_held_check",
    "mark_no_longer_skipped",
    "persist_human_task_plan",
    "resolve_human_task_plan",
    "run_write_phase",
    "verification_line",
    "with_result_contract",
]


STRATEGY_PAUSE = "pause-and-ask"
STRATEGY_SKIP_CONTINUE = "skip-continue"
STRATEGY_SKIP_DEPENDENTS = "skip-dependents"
STRATEGIES = (STRATEGY_PAUSE, STRATEGY_SKIP_CONTINUE, STRATEGY_SKIP_DEPENDENTS)

#: The scheduler's own report, beside the executor's per-dispatch one. A separate
#: file because it records the *loop*: batch order, the stopping batch, the skip
#: set, the override records and whether finalization ran.
WRITE_BATCH_REPORT_FILENAME = "write_batch_report.json"

#: Write kinds whose ``output_name`` is keyed by the task id
#: (``subagent_tasks.implement_output_name``) rather than a batch position.
#: ``_warn_stale_result_files`` (N6) scopes its stale-file sweep to this set:
#: a positional kind's result filenames have never been renamed, so there is
#: no earlier naming to be stale relative to.
_ID_KEYED_WRITE_KINDS = frozenset({"implement"})

#: The id shape the task decomposer gives implement's own tasks — mirrors
#: ``StateManager._IMPLEMENT_TASK_ID_PATTERN`` for a state object without
#: ``implementation_task_statuses`` (the test fakes), so "which of the flat
#: ``task_status`` map's ids belong to THIS phase" has one answer here too.
_IMPLEMENT_TASK_ID = re.compile(r"^T[0-9]{3}$")

#: The D2g inputs binding for the results already on disk, kept beside them.
#: A **leading dot** on purpose: reaggregation globs ``{phase_dir}/*.json`` and
#: treats every file it matches as a per-model result list, and ``glob`` skips
#: leading-dot names — the same reason ``.status.json`` is invisible to it.
WRITE_FINGERPRINT_FILENAME = ".write_task_inputs.json"

#: Directory *names* the post-run containment sweep never descends into. A VCS
#: metadata dir churns on anything the runner happens to run — ``git status``
#: alone rewrites ``.git/index`` — so attributing that churn to the batch would
#: stop the loop on a violation the runner never committed. They are denied by
#: ``driver.write_paths`` out of the box, which is where the *preventive* half
#: (the runner's own sandbox) applies; the sweep is the detective half and does
#: not guess at VCS internals.
SWEEP_PRUNED_DIR_NAMES = (".git", ".hg", ".svn")

#: Above this many regular files the sweep is abandoned rather than run: it
#: walks the tree twice per dispatch, and a control that silently doubles a
#: phase's wall clock is a control that gets switched off. The degradation is
#: announced once per run, never silent.
SWEEP_MAX_FILES = 60000

#: Where the sweep's rollback bytes live: ``{phase}/rollback_store/ab/<sha256>``,
#: content-addressed so a file that survives ten dispatches is stored once and a
#: reverted edit costs nothing. Under the phase dir like every other executor
#: artifact, which is also what keeps it out of the sweep's own census.
ROLLBACK_STORE_DIRNAME = "rollback_store"

#: The rollback store's byte budget for a whole run. Detection is cheap (a stat
#: per file); *reversibility* costs a read and a copy, so it is bounded and the
#: bound is announced. Past it the sweep still observes, checks and audits an
#: undeclared write — it just cannot undo it, which is the state every release
#: before this one was in for all of them.
SWEEP_RETAIN_MAX_BYTES = 64 * 1024 * 1024

#: The most undeclared paths one dispatch adopts into its snapshot. A runner that
#: ran a package install can touch tens of thousands of files, and a manifest
#: with one entry each helps nobody. Every observed path is still reported and
#: policy-checked; only the rollback points are capped.
SWEEP_ADOPT_MAX_FILES = 1000

#: Write kinds whose deliverable is **source code**, so the repository's own
#: `driver.verify_command` has something to say about them (F4). The other two
#: write kinds edit the plan and the tasks file — markdown a typecheck cannot
#: read — so they skip verification silently rather than logging a notice about a
#: command that was never going to apply to them.
CODE_WRITE_KINDS = frozenset({"apply_code_fixes", "implement"})

#: The line an implement run prints per check task that reported problems
#: (``WriteBatchScheduler._check_findings_lines``, racing re-run 4).
CHECK_FINDINGS_MARKER = "[CHECK_FINDINGS]"

#: How much of the verification command's output travels in the result payload,
#: the summary and `.status.json`. A tail rather than the whole capture: a failing
#: build prints megabytes and what names the failure is the end of it.
VERIFY_TAIL_CHARS = 2000

#: How much of the command's output is read back off disk before redaction and
#: the tail cap above. `capture_output=True` buffered a whole build in this
#: process to hand 2000 characters of it to a report (F6); a bounded read off the
#: end of the capture file is the same tail at a fixed cost.
VERIFY_OUTPUT_CAP_BYTES = 64 * 1024

#: SIGTERM→SIGKILL grace for a timed-out command's process group, then how long
#: the reaping wait holds. The same shape `subagent_executor` uses on a runner.
VERIFY_KILL_GRACE_S = 0.5
VERIFY_REAP_S = 5.0

#: Recorded when the command produced no exit status of its own, in the shell's
#: own vocabulary so a reader of `verification.exit_code` learns nothing private.
VERIFY_EXIT_TIMEOUT = 124
VERIFY_EXIT_NOT_RUN = 127

EXIT_OK = 0
EXIT_NEEDS_HUMAN = 3

#: The review each apply kind applies — the first entry of
#: ``check_workflow_prerequisites.PHASE_PREREQUISITES`` for that phase, which
#: is the prerequisite ``_apply_batches_predate_review`` dates a batch
#: against. Spelled here so the scheduler does not import the prerequisite
#: script (a top-level module) from inside ``utils/``; a test holds the two in
#: agreement. ``implement`` has no entry: its ``T###`` ids are the plan's.
REVIEW_PHASE_BY_WRITE_KIND: Dict[str, str] = {
    "apply_suggestions": "review-plan",
    "apply_task_suggestions": "review-tasks",
    "apply_code_fixes": "code-review",
}


#: The index token of a positional write result's filename — bare
#: (``apply_batch_1.json``) or retired beside the slot by
#: ``_retire_foreign_result`` (``apply_batch_1.items-<sha8>[-N].json``).
_RESULT_TOKEN = re.compile(r"^(\d+)(?:\.items-[0-9a-f]{8}(?:-\d+)?)?$")


def _same_batch_row(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """Whether two ``write_batch_report.json`` rows describe the same batch.

    The items a row was materialized for (``expected_ids``) when both rows
    carry them, else the task id. An apply kind's task ids are positional
    (``apply-batch-N``): a ``--resume`` after the selection changed (a stopped
    batch's items skipped in the report) renumbers its batches, and an id
    match alone replaced the earlier run's ``apply-batch-1`` row, whose edits
    are still on disk, with a row for different items (#28). ``implement``'s
    ``expected_ids`` is ``[task_id]``, so it is unaffected. A row without
    them (refused before any verdict, or a report older than F9) falls back
    to the id.
    """
    left_ids = left.get("expected_ids") or []
    right_ids = right.get("expected_ids") or []
    if left_ids and right_ids:
        return sorted(map(str, left_ids)) == sorted(map(str, right_ids))
    return str(left.get("task_id", "")) == str(right.get("task_id", ""))


def _parse_local_timestamp(value: Any) -> Optional[datetime]:
    """An ISO timestamp as a naive local datetime, or None — the reading
    ``check_workflow_prerequisites._parse_timestamp`` gives ``state.json``."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


# ---------------------------------------------------------------------------
# The write grant on the plugin-landlock rung (plan
# codex-sandbox-unavailable-findings-and-fixes, step 3 option (b), 2026-09-30)
# ---------------------------------------------------------------------------

#: The kind → ``write_scope`` map, as a fallback for an executor that predates
#: ``subagent_executor.WRITE_SCOPE_BY_KIND`` (2026-09-30: the two packages land
#: separately). The executor's table wins whenever it exists
#: (:func:`write_scope_for_kind`), so there is one map in force, never two that
#: can disagree. The scope comes from the phase's kind, never from the task
#: JSON (F1 "How a dispatch receives its grant").
_WRITE_SCOPE_BY_KIND_FALLBACK: Dict[str, str] = {
    "apply_suggestions": WRITE_SCOPE_PLAN_FILE,
    "apply_task_suggestions": WRITE_SCOPE_TASKS,
    "apply_code_fixes": WRITE_SCOPE_REPO_WIDE,
    "implement": WRITE_SCOPE_REPO_WIDE,
}


def write_scope_for_kind(kind: str) -> Optional[str]:
    """The ``write_scope`` a write kind's dispatches carry, or ``None``.

    ``plan-file`` for apply-suggestions, ``tasks`` for apply-task-suggestions,
    ``repo-wide`` for apply-code-fixes and implement (on the rung since plan
    step 3b, 2026-10-02). An unmapped kind gets ``None``, which the executor
    refuses on the rung rather than widening it.
    """
    table = getattr(_executor, "WRITE_SCOPE_BY_KIND", None)
    if not isinstance(table, dict):
        table = _WRITE_SCOPE_BY_KIND_FALLBACK
    return table.get(kind)


def _accepts_keyword(func: Any, name: str) -> bool:
    """Whether ``func`` takes keyword ``name`` (or ``**kwargs``).

    The scheduler hands the executor a ``jail_request=`` / ``write_scope=``
    only where the executor's signature takes one, so it keeps working against
    an executor that predates plan step 3 (2026-09-30) and against a test's
    stand-in with the historical keyword set — the same courtesy
    ``dispatch_task`` already pays ``_spawn`` for ``read_dirs``.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _as_jail_denials(value: Any) -> Any:
    """``after_result``'s ``jail_denials`` as a ``landlock_dispatch.JailDenials``.

    The executor hands the dataclass itself; a dict (its ``to_dict()`` shape,
    e.g. read back from a record) is rebuilt so ``_classify`` reads one type.
    ``None`` — a dispatch that was not on the rung, or carried no write scope —
    stays ``None``.
    """
    if value is None or isinstance(value, _landlock.JailDenials):
        return value
    if hasattr(value, "failed_file_changes") and hasattr(value, "denied_paths"):
        # The same dataclass reached through another import path of the
        # module (a duplicate module object fails `isinstance`, not the shape).
        return value
    if isinstance(value, dict):
        return _landlock.JailDenials(
            failed_file_changes=[
                [str(p) for p in group]
                for group in (value.get("failed_file_changes") or [])
            ],
            in_scope_paths=[str(p) for p in (value.get("in_scope_paths") or [])],
            replacement_paths=[
                str(p) for p in (value.get("replacement_paths") or [])
            ],
            out_of_scope_count=int(value.get("out_of_scope_count") or 0),
            out_of_scope_paths=[
                str(p) for p in (value.get("out_of_scope_paths") or [])
            ],
        )
    return None


# ---------------------------------------------------------------------------
# Human tasks: one strategy, resolved once, then a graph walk (D3n)
# ---------------------------------------------------------------------------

@dataclass
class HumanTaskPlan:
    """The resolved human-task decision for a whole phase.

    Resolved **before** the loop and never re-asked on resume, then applied as a
    graph walk. All three strategies live here rather than in markdown, so the skip
    set, the reasons and the override records are byte-identical on both invokers.
    """

    strategy: Optional[str] = None
    #: ``task_id -> reason``. Every skip carries the sentence the markdown specifies.
    skipped: Dict[str, str] = field(default_factory=dict)
    #: ``task_id -> [human task ids treated as satisfied]`` (skip-continue only).
    overrides: Dict[str, List[str]] = field(default_factory=dict)
    #: Human tasks answered ``done`` via ``--human-task <id>=done``.
    completed: List[str] = field(default_factory=list)
    #: Human tasks still waiting for an answer under ``pause-and-ask``.
    pending: List[str] = field(default_factory=list)
    #: ``child -> parent`` chain, for the summary's skip-chain reporting.
    chain: Dict[str, str] = field(default_factory=dict)
    #: ``task_id -> reason`` for a task the last run recorded ``failed`` and
    #: the user answered ``--human-task <id>=done``: treated as satisfied, not
    #: dispatched again, its dependents run (racing re-run 3, 2026-10-05: a
    #: failed check blocked implement for good).
    accepted: Dict[str, str] = field(default_factory=dict)
    #: Failed tasks answered ``--human-task <id>=retry``: dispatched again even
    #: when nothing in the repository changed since the failure.
    retry: List[str] = field(default_factory=list)
    #: Tasks an earlier run recorded ``skipped`` that this plan no longer
    #: skips (``--resume --force-strategy skip-continue`` after a
    #: ``skip-dependents`` run): :func:`persist_human_task_plan` records them
    #: ``pending`` again, dropping the old reason (:func:`mark_no_longer_skipped`).
    unskipped: List[str] = field(default_factory=list)

    def is_skipped(self, task_id: str) -> bool:
        return task_id in self.skipped

    def declined(self) -> Dict[str, str]:
        """The skips that are the user's own decision, with their reasons.

        A pause-and-ask "Skip", a failed code task answered
        ``--human-task <id>=skip``, and every task skipped *because* it depends
        on one of those (``chain``). They are not deferred work: nobody is
        owed them afterwards, so neither the summary's ``## Manual Tasks
        (Deferred)`` section nor ``[WORKFLOW_COMPLETE]``'s manual-task list
        names them (``StateManager._deferred_to_user`` reads the same two
        reasons back from ``state.json``). Everything else in ``skipped`` is a
        deferral: a manual task under a skip strategy and its dependents.
        """
        declined: Dict[str, str] = {}
        for task_id, reason in self.skipped.items():
            if reason.startswith((USER_DECLINED_REASON, ANSWERED_SKIP_PREFIX)):
                declined[task_id] = reason
        changed = True
        while changed:
            changed = False
            for child, parent in self.chain.items():
                if child not in declined and parent in declined and child in self.skipped:
                    declined[child] = self.skipped[child]
                    changed = True
        return declined

    def deferred(self) -> Dict[str, str]:
        """``skipped`` minus :meth:`declined`: what is still owed by hand."""
        declined = self.declined()
        return {
            task_id: reason for task_id, reason in self.skipped.items()
            if task_id not in declined
        }


#: The recorded reason of a human task the user declined at pause-and-ask.
#: ``StateManager._USER_DECLINED_REASON`` spells the same words, so the
#: workflow's manual-task list reads a declined task as not owed.
USER_DECLINED_REASON = "skipped by user during pause-and-ask"

#: The prefix of the recorded reason of a failed code task the user answered
#: ``--human-task <id>=skip`` (``StateManager._ANSWERED_SKIP_PREFIX`` is its
#: twin). The sentence ends ``<id>=skip after it failed``.
ANSWERED_SKIP_PREFIX = "skipped — you answered --human-task "

#: The prefix of the recorded reason of a failed code task the user answered
#: ``--human-task <id>=done`` (:func:`resolve_human_task_plan` writes the
#: whole sentence into ``HumanTaskPlan.accepted``).
ACCEPTED_REASON_PREFIX = "accepted — "

#: The words in the recorded reason of a check accepted with its findings
#: (``--human-task <id>=done`` on a check that reported problems, fix round
#: 11): the account names it apart from a task accepted after failing.
ACCEPTED_WITH_FINDINGS_MARK = "after its check reported problems"

#: The recorded reason of a manual task answered ``--human-task <id>=done``
#: (:func:`persist_human_task_plan`; the ``<id>`` is literal).
COMPLETED_MANUAL_REASON = "answered --human-task <id>=done"


def answered_done_on_record(
    state: Any, *, exclude: Sequence[str] = (),
) -> Tuple[Dict[str, str], List[str]]:
    """The tasks an earlier run recorded done on the user's answer.

    ``(accepted, completed_manual)``: ``accepted`` maps each failed code task
    the user answered ``--human-task <id>=done`` to its recorded reason, and
    ``completed_manual`` lists the manual tasks answered ``done``. Both are
    ``completed`` in ``state.json`` with the reason the answer wrote, so a
    later ``--resume`` materializes neither and the answer no longer applies
    (it applies to a failed task only); read back from that record, they keep
    their bucket in the resumed account instead of dropping out of it and
    shrinking its total (fix round 9). Scoped to the decomposition implement
    recorded last (``implement_task_ids``, else every ``T###`` id), minus
    *exclude* (the tasks this run materialized, which the run's own plan
    accounts for).
    """
    if state is None or not hasattr(state, "get"):
        return {}, []
    try:
        statuses = dict(state.get("task_status") or {})
        reasons = dict(state.get("task_status_reasons") or {})
        current = state.get("implement_task_ids")
    except Exception:  # noqa: BLE001 - an account read must not fail a phase
        return {}, []
    scope = set(current) if isinstance(current, list) else None
    skip = {str(task_id) for task_id in exclude}
    accepted: Dict[str, str] = {}
    completed_manual: List[str] = []
    for task_id, status in sorted(statuses.items()):
        if status != "completed" or task_id in skip:
            continue
        if not _IMPLEMENT_TASK_ID.match(str(task_id)):
            continue
        if scope is not None and task_id not in scope:
            continue
        reason = str(reasons.get(task_id) or "")
        if reason.startswith(ACCEPTED_REASON_PREFIX):
            accepted[task_id] = reason
        elif reason == COMPLETED_MANUAL_REASON:
            completed_manual.append(task_id)
    return accepted, completed_manual


def rerunnable_checks(state: Any) -> List[str]:
    """The check tasks whose findings hold back the tasks after them.

    Each implement task with findings on record (``task_check_findings``),
    recorded ``completed`` and not accepted (``--human-task <id>=done``
    writes an ``accepted —`` reason and keeps the findings for review-code).
    Fix round 11 (review of round 10, T2): such a task used to be done for
    good — ``--resume`` skipped it and ``--human-task <id>=retry`` answered
    only a task recorded ``failed`` — so the tasks it held could never run
    after the user fixed the code. It is materialized again on a resume
    (``implement_orchestrator.format_batches_for_output``), answerable with
    ``--human-task <id>=done|skip|retry``, and re-dispatched when the files
    changed since it ran (:meth:`WriteBatchScheduler._dispatch_one`).
    """
    if state is None or not hasattr(state, "get"):
        return []
    try:
        recorded = state.get("task_check_findings") or {}
        statuses = dict(state.get("task_status") or {})
        reasons = dict(state.get("task_status_reasons") or {})
    except Exception:  # noqa: BLE001 - a read for the resume, never fatal
        return []
    if not isinstance(recorded, dict):
        return []
    return sorted(
        str(task_id) for task_id, findings in recorded.items()
        if isinstance(findings, list) and findings
        and statuses.get(task_id) == "completed"
        and not str(reasons.get(task_id) or "").startswith(ACCEPTED_REASON_PREFIX)
    )


def decomposition_graph(
    tasks: Sequence[Dict[str, Any]], state: Any,
) -> List[Dict[str, Any]]:
    """*tasks* plus one node per task of the recorded decomposition they lack.

    Fix round 12 (review of round 11): a ``--resume`` materializes only the
    unfinished tasks, so a walk over them alone lost every edge through a
    finished one. With T001 → T002 (a check with findings) → T005 (manual,
    answered done) → T006, the first run held T004 and T006; a plain
    ``--resume`` that changed nothing kept T004 held and dispatched T006,
    whose only path to T002 ran through T005, completed and so not
    materialized — the executor's walk, ``util batch-done``'s and ``util
    write-prompt``'s alike. *state* (a ``StateManager`` or ``state.json``'s
    dict) carries the whole graph implement decomposed
    (:data:`state_manager.IMPLEMENT_TASK_GRAPH_KEY`); each task it names that
    *tasks* lacks is added as ``{"task_id", "depends_on", "on_record_only":
    True}`` (``executor_role: human`` for a manual one), so a walk passes
    through it. *tasks* win for the ids they hold. A state with no graph on
    record (one written before fix round 12) adds nothing.
    """
    nodes = [task for task in tasks if isinstance(task, dict)]
    if state is None or not hasattr(state, "get"):
        return nodes
    try:
        recorded = state.get(IMPLEMENT_TASK_GRAPH_KEY)
        human = {str(item) for item in (state.get("implement_human_task_ids") or [])}
    except Exception:  # noqa: BLE001 - a graph read is never fatal
        return nodes
    if not isinstance(recorded, dict):
        return nodes
    have = {str(task.get("task_id", "")) for task in nodes}
    for task_id in sorted(str(key) for key in recorded):
        deps = recorded.get(task_id)
        if not task_id or task_id in have or not isinstance(deps, list):
            continue
        node: Dict[str, Any] = {
            "task_id": task_id,
            "depends_on": [str(dep) for dep in deps if dep],
            "on_record_only": True,
        }
        if task_id in human:
            node["executor_role"] = "human"
        nodes.append(node)
    return nodes


def held_back_by_checks(
    tasks: Sequence[Dict[str, Any]], state: Any,
) -> List[Tuple[str, str]]:
    """``(task id, check id)`` for each implement task a check with findings
    holds back and that has not run, by task id.

    Computed, never read off a recorded reason (fix round 12: two checks
    held D, the first was recorded again as passed, D's reason was cleared
    while the second's findings stood, and ``util finalize-tracking
    --mark-completed`` completed implement with D unrun): the held checks are
    :func:`rerunnable_checks`, the graph is :func:`decomposition_graph` over
    *tasks* and *state*, and a task has not run when ``task_status`` says
    neither ``completed`` nor ``skipped``. A manual task is never held (no
    subagent runs it), and the walk passes through it. Scoped to the
    decomposition implement recorded last (``implement_task_ids``) when one
    is on record.
    """
    held = rerunnable_checks(state)
    if not held:
        return []
    graph = decomposition_graph(tasks, state)
    try:
        statuses = dict(state.get("task_status") or {})
        scope = state.get("implement_task_ids")
    except Exception:  # noqa: BLE001 - a read for a refusal, never fatal
        return []
    out: List[Tuple[str, str]] = []
    for task in graph:
        task_id = str(task.get("task_id", ""))
        if not task_id or task_id in held:
            continue
        if isinstance(scope, list) and task_id not in scope:
            continue
        if task.get("executor_role") == "human" or task.get("is_human"):
            continue
        if statuses.get(task_id) in ("completed", "skipped"):
            continue
        check_id = upstream_held_check(graph, task_id, held)
        if check_id is not None:
            out.append((task_id, check_id))
    return sorted(out)


def transitive_dependents(tasks: Sequence[Dict[str, Any]], task_id: str) -> List[str]:
    """Every task of *tasks* that depends on *task_id*, transitively, sorted.

    Over each task's ``depends_on``; the one graph walk both legs use for
    the tasks a check with findings holds back (the scheduler's
    ``_dependents_of``, ``util batch-done --check findings``). Pass the
    whole graph (:func:`decomposition_graph`), not a resume's subset.
    """
    dependents: Dict[str, List[str]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for dep in task.get("depends_on") or []:
            dependents.setdefault(str(dep), []).append(str(task.get("task_id", "")))
    seen: List[str] = []
    queue = [str(task_id)]
    while queue:
        parent = queue.pop(0)
        for child in sorted(dependents.get(parent, [])):
            if child and child not in seen:
                seen.append(child)
                queue.append(child)
    return sorted(seen)


def upstream_held_check(
    tasks: Sequence[Dict[str, Any]], task_id: str, held: Sequence[str],
) -> Optional[str]:
    """The check in *held* that *task_id* depends on, transitively, or ``None``.

    *held* is :func:`rerunnable_checks`: checks whose findings hold back the
    tasks after them. Walks ``depends_on`` upward through *tasks*; an id a
    task depends on that *tasks* does not hold is still compared, just not
    walked further — so pass the whole graph (:func:`decomposition_graph`):
    a task a resume did not materialize is exactly where the walk used to
    stop short of the check (fix round 12).
    """
    held_set = {str(item) for item in held}
    if not held_set:
        return None
    by_id = {
        str(task.get("task_id", "")): task for task in tasks if isinstance(task, dict)
    }
    seen: set = set()
    queue = [str(dep) for dep in (by_id.get(str(task_id), {}).get("depends_on") or [])]
    while queue:
        dep = queue.pop(0)
        if dep in seen:
            continue
        seen.add(dep)
        if dep in held_set:
            return dep
        queue.extend(str(d) for d in (by_id.get(dep, {}).get("depends_on") or []))
    return None


def _deferred_reason(strategy: Optional[str]) -> str:
    """Why a manual task did not run, naming the strategy and not a person.

    This used to say "user chose to complete manually", which is false when
    ``--yes`` resolved the strategy to ``skip-dependents`` on its own
    (2026-10-01, OpenCode run): nobody was asked.
    """
    return (
        f"deferred — manual task, left for you to complete after "
        f"implementation (human-task strategy: {strategy})"
    )


def _id_range(ids: Sequence[str]) -> str:
    """``T001–T008`` for a run of consecutive ``T###`` ids, else the list."""
    numbers = []
    for task_id in ids:
        match = _IMPLEMENT_TASK_ID.match(str(task_id))
        if not match:
            return ", ".join(str(i) for i in ids)
        numbers.append(int(str(task_id)[1:]))
    if len(numbers) >= 3 and numbers == list(range(numbers[0], numbers[0] + len(numbers))):
        return f"{ids[0]}–{ids[-1]}"
    return ", ".join(str(i) for i in ids)


#: The start of the recorded reason of a task not run because a check it
#: depends on reported problems (:func:`blocked_by_check_reason`). It starts
#: ``not run``, so review-code's implementation record reads it as a gap
#: (``code_review_orchestrator.build_implementation_record``), and
#: ``util finalize-tracking --mark-completed`` refuses while one is pending.
BLOCKED_BY_CHECK_PREFIX = "not run — upstream check "


def blocked_by_check_reason(check_id: str) -> str:
    """The reason recorded for a task a check with findings kept from running."""
    return (
        f"{BLOCKED_BY_CHECK_PREFIX}{check_id} reported problems; it runs once "
        f"{check_id} passes on a --resume, or after --human-task "
        f"{check_id}=done accepts its findings"
    )


def _accounting_line(account: Dict[str, Any]) -> str:
    """``N tasks in this decomposition: a done, b failed (…), …`` — every
    bucket, named, adding up to the total."""
    parts = [f"{len(account['done'])} done"]
    flagged = account.get("check_findings") or []
    if flagged:
        # Inside the done bucket, so the buckets still add up to the total.
        parts[0] += f" ({len(flagged)} with check findings: {', '.join(flagged)})"
    if account["failed"]:
        parts.append(f"{len(account['failed'])} failed ({', '.join(account['failed'])})")
    if account["not_run"]:
        blocked = account.get("blocked") or {}
        stopped = [tid for tid in account["not_run"] if tid not in blocked]
        named = [", ".join(stopped)] if stopped else []
        for check_id in sorted(set(blocked.values())):
            waiting = [tid for tid in account["not_run"] if blocked.get(tid) == check_id]
            named.append(
                f"{', '.join(waiting)}: upstream check {check_id} reported problems"
            )
        parts.append(f"{len(account['not_run'])} not run ({'; '.join(named)})")
    with_findings = account.get("accepted_with_findings") or []
    after_failing = [tid for tid in account["accepted"] if tid not in with_findings]
    if after_failing:
        parts.append(
            f"{len(after_failing)} accepted after failing "
            f"({', '.join(after_failing)})"
        )
    if with_findings:
        parts.append(
            f"{len(with_findings)} accepted with their check findings "
            f"({', '.join(with_findings)})"
        )
    if account["completed_manual"]:
        parts.append(
            f"{len(account['completed_manual'])} completed manual "
            f"({', '.join(account['completed_manual'])})"
        )
    if account["declined"]:
        parts.append(
            f"{len(account['declined'])} skipped at your request "
            f"({', '.join(account['declined'])})"
        )
    if account["skipped_dependents"]:
        parts.append(
            f"{len(account['skipped_dependents'])} skipped with a skipped task "
            f"({', '.join(account['skipped_dependents'])})"
        )
    if account["deferred"]:
        parts.append(
            f"{len(account['deferred'])} deferred manual ({', '.join(account['deferred'])})"
        )
    total = account["total"]
    return (
        f"**Tasks in this decomposition**: {total} — " + ", ".join(parts) + "."
    )


def verification_line(result: Any) -> str:
    """One line describing what `driver.verify_command` did, for a report (F4).

    Shared by the scheduler's own summary and
    ``apply_orchestrator_base.write_apply_summary_report`` so the two documents a
    code-fix run leaves behind cannot describe the same check differently. Takes
    anything carrying a ``verification`` attribute, so a caller holding only the
    orchestrator's ``write_result`` needs no import of its own.
    """
    record = getattr(result, "verification", None)
    if not isinstance(record, dict):
        return "not configured (`driver.verify_command` is unset)"
    verdict = "pass" if record.get("exit_code") == 0 else (
        f"FAIL (exit {record.get('exit_code')})"
    )
    return (
        f"`{record.get('command', '')}` — {verdict}, "
        f"{record.get('duration_seconds')}s"
    )


def _redacted_text(text: str) -> str:
    """``write_safety.redact_text``, through the module's usual import shim."""
    try:
        from .write_safety import redact_text
    except ImportError:  # pragma: no cover
        from utils.write_safety import redact_text
    return redact_text(text)


def _under_any(path: str, roots: Sequence[str]) -> bool:
    """Whether ``path`` is one of ``roots`` or sits beneath one of them.

    Separator-aware rather than a bare ``startswith``, which would also swallow
    a sibling whose name merely shares a prefix (``plans-old/`` under
    ``plans/``).
    """
    candidate = os.path.abspath(path)
    for root in roots:
        if candidate == root or candidate.startswith(root + os.sep):
            return True
    return False


def resolve_human_task_plan(
    tasks: Sequence[Dict[str, Any]],
    *,
    strategy: Optional[str],
    decisions: Optional[Dict[str, str]] = None,
    check_findings_ids: Sequence[str] = (),
    graph: Optional[Sequence[Dict[str, Any]]] = None,
) -> HumanTaskPlan:
    """Apply ``strategy`` over the task graph and return the resolved plan (D3n).

    * ``pause-and-ask`` routes each human task to the D6 pause: a task answered
      ``done`` is treated as satisfied, one answered ``skip`` seeds the transitive
      propagation, and an unanswered one lands in ``pending`` so the caller can
      exit 3 naming the two ``--human-task`` re-invocations.
    * ``skip-continue`` marks every human task skipped, treats its edges as
      satisfied and records a dependency override per unblocked task.
    * ``skip-dependents`` marks every human task skipped and runs the transitive
      BFS forward walk over ``depends_on``.

    The strategy governs the **manual** tasks. A failed code task answered
    ``--human-task <id>=skip`` always takes its dependents with it, under every
    strategy: ``skip-continue`` means "a manual precondition is not met, run
    the code anyway", and a code task that failed and was then given up is
    not a precondition anyone can meet by hand. Until fix round 8 the
    ``skip-continue`` branch returned before the walk, so the dependents ran
    (with override records) while the stop message and the instructions both
    said they stay undone (reviewer repro, 2026-10-05).

    A check task whose findings hold back the tasks after it
    (*check_findings_ids*, :func:`rerunnable_checks`) is answered the same
    way, and its reason says so: ``done`` accepts the findings (the tasks
    after it run; the findings stay listed for review-code), ``skip`` leaves
    it and its dependents undone, ``retry`` runs the check again on
    unchanged files (fix round 11).

    *graph* is the whole decomposition (:func:`decomposition_graph`) when
    *tasks* are a resume's subset (fix round 12): the walk from a code task
    answered ``skip`` goes through a task this run does not hold (finished
    before the resume, so not skipped) to the tasks after it, which are
    skipped naming the skipped task they depend on — a check answered
    ``skip`` used to leave a task whose chain to it ran through a manual
    task answered done to run. The walk from a manual task stays over
    *tasks*.

    Pure: takes the task list, returns the decision. Persisting it (through
    ``update_task_status`` / ``record_dependency_override``) is the scheduler's job,
    which is what keeps this testable against the native path's expectations.
    """
    decisions = dict(decisions or {})
    with_findings = {str(task_id) for task_id in check_findings_ids}
    by_id: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        by_id[str(task.get("task_id", ""))] = task
    human_ids = [
        task_id for task_id, task in by_id.items()
        if task.get("executor_role") == "human" or task.get("is_human")
    ]
    plan = HumanTaskPlan(strategy=strategy)
    # A decision for a task that is NOT human answers a task the last run
    # recorded `failed` (the orchestrator admits no other): `done` accepts
    # the failure and treats the task as satisfied, `skip` leaves it and what
    # depends on it undone, `retry` re-runs it past the unchanged-tree guard.
    seeds: List[str] = []
    code_seeds: List[str] = []
    for task_id, answer in sorted(decisions.items()):
        if task_id in human_ids or task_id not in by_id:
            continue
        found = task_id in with_findings
        if answer == "done":
            plan.accepted[task_id] = (
                f"{ACCEPTED_REASON_PREFIX}you answered --human-task {task_id}=done "
                + (
                    f"{ACCEPTED_WITH_FINDINGS_MARK}; the findings stand "
                    "(review-code is told about them), the check is not run "
                    "again and the tasks after it run"
                    if found else
                    "after it failed; treated as satisfied and not run again"
                )
            )
        elif answer == "skip":
            plan.skipped[task_id] = (
                f"{ANSWERED_SKIP_PREFIX}{task_id}=skip after "
                + ("its check reported problems" if found else "it failed")
            )
            seeds.append(task_id)
            code_seeds.append(task_id)
        elif answer == "retry":
            plan.retry.append(task_id)
    if not human_ids and not seeds:
        return plan
    if human_ids and strategy is None:
        # Nothing resolved: the caller owns the gate. Reporting the human tasks as
        # pending rather than silently skipping them is the fail-closed half.
        plan.pending = sorted(human_ids)
        return plan
    if strategy is not None and strategy not in STRATEGIES:
        raise ExecutorError(
            f"Unknown human-task strategy {strategy!r}: expected one of "
            f"{', '.join(STRATEGIES)}. Nothing was dispatched."
        )

    if not human_ids:
        pass
    elif strategy == STRATEGY_PAUSE:
        for task_id in sorted(human_ids):
            answer = decisions.get(task_id)
            if answer == "done":
                plan.completed.append(task_id)
            elif answer == "skip":
                plan.skipped[task_id] = USER_DECLINED_REASON
                seeds.append(task_id)
            else:
                plan.pending.append(task_id)
    else:
        for task_id in sorted(human_ids):
            plan.skipped[task_id] = _deferred_reason(strategy)
            seeds.append(task_id)

    # pause-and-ask (for the tasks answered `skip`) and skip-dependents both walk
    # the graph forward: BFS over "who depends on me", so a chain
    # T001(human) -> T003 -> T005 is skipped in full. skip-continue walks it
    # only from a failed code task answered `skip` (see the docstring).
    dependents: Dict[str, List[str]] = {}
    nodes = list(graph) if graph is not None else list(by_id.values())
    for task in nodes:
        if not isinstance(task, dict):
            continue
        for dep in task.get("depends_on") or []:
            dependents.setdefault(str(dep), []).append(str(task.get("task_id", "")))
    code_set = set(code_seeds)
    # (task, the skipped task a dependent names, whether the walk may pass
    # through a task this run does not hold — only from a code seed).
    queue = [
        (seed, seed, seed in code_set)
        for seed in (code_seeds if strategy == STRATEGY_SKIP_CONTINUE else seeds)
    ]
    passed: set = set()
    while queue:
        parent, named, through = queue.pop(0)
        named_title = str(by_id.get(named, {}).get("task_title", "") or named)
        for child in sorted(set(dependents.get(parent, []))):
            if not child:
                continue
            if child not in by_id:
                if through and child not in passed:
                    passed.add(child)
                    queue.append((child, named, through))
                continue
            if child in plan.skipped:
                continue
            plan.skipped[child] = (
                f"skipped — depends on skipped task {named} ({named_title})"
            )
            plan.chain[child] = named
            queue.append((child, child, through))

    if strategy == STRATEGY_SKIP_CONTINUE:
        # A manual task's edges are treated as satisfied: its dependents run,
        # and every one of them gets an override RECORD so "it ran against an
        # unmet precondition" is visible afterwards instead of inferred. A
        # dependent the walk above skipped does not run, so it gets none.
        manual_skipped = {tid for tid in human_ids if tid in plan.skipped}
        for task_id, task in by_id.items():
            if task_id in plan.skipped:
                continue
            overridden = [
                dep for dep in (task.get("depends_on") or [])
                if str(dep) in manual_skipped
            ]
            if overridden:
                plan.overrides[task_id] = sorted(str(dep) for dep in overridden)
    return plan


def mark_no_longer_skipped(
    plan: HumanTaskPlan, task_ids: Sequence[str], recorded: Dict[str, Any],
) -> List[str]:
    """Record on *plan* the tasks *recorded* says ``skipped`` that it runs.

    A ``--resume`` materializes a skipped task again (fix round 8), and only
    a skip the new plan makes used to overwrite its status. After a
    ``skip-dependents`` run, ``--resume --force-strategy skip-continue``
    (``--force-strategy`` outranks the recorded strategy) left a dependent of
    the manual task ``skipped`` with its "depends on skipped task" reason: the
    native host never dispatched it, a stop before it left the record
    standing, and ``[MANUAL_TASK]`` named it owed for good (fix round 9). A
    task of *task_ids* the plan neither skips, completes nor accepts goes in
    ``plan.unskipped``; returns them. A plan with no strategy and manual tasks
    pending resolved nothing (the walk never ran), so it unskips nothing.
    """
    if plan.strategy is None and plan.pending:
        return []
    for task_id in task_ids:
        task_id = str(task_id or "")
        if (
            not task_id or recorded.get(task_id) != "skipped"
            or task_id in plan.skipped or task_id in plan.completed
            or task_id in plan.accepted or task_id in plan.unskipped
        ):
            continue
        plan.unskipped.append(task_id)
    return list(plan.unskipped)


def persist_human_task_plan(state: Any, plan: HumanTaskPlan) -> None:
    """Write a resolved :class:`HumanTaskPlan` into ``state.json`` and save.

    The strategy, a ``skipped`` status (with its reason) for every task the
    walk skipped, ``completed`` for a human task answered ``done`` and for a
    failed code task accepted with ``done``, ``pending`` for a task an
    earlier run skipped that this plan runs (``unskipped``), and one
    dependency override per unblocked task. Shared by the scheduler (before
    its first batch) and by ``implement``'s native emit, which records the
    same plan before it hands the batches to the host: until 2026-10-05 the
    native leg left these writes to the host, which never made them, so a
    task deferred under ``--yes`` never reached ``task_status`` and the
    finished workflow named no manual task (racing re-run 3, Claude Code).
    An unanswered ``pause-and-ask`` task is left without a status here.
    """
    if plan.strategy and state.get("human_task_strategy") != plan.strategy:
        state.set("human_task_strategy", plan.strategy)
    for task_id, reason in sorted(plan.skipped.items()):
        state.update_task_status(task_id, "skipped", reason=reason)
    for task_id in plan.completed:
        state.update_task_status(
            task_id, "completed", reason=COMPLETED_MANUAL_REASON,
        )
    for task_id, reason in sorted(plan.accepted.items()):
        state.update_task_status(task_id, "completed", reason=reason)
    for task_id in plan.unskipped:
        # No reason: the old skip's reason goes with the old status.
        state.update_task_status(task_id, "pending")
    for task_id, overridden in sorted(plan.overrides.items()):
        state.record_dependency_override(task_id, list(overridden))
    state.save()


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------

@dataclass
class SchedulerResult:
    """What one write phase's loop produced. ``exit_code`` is the phase's."""

    exit_code: int = EXIT_OK
    kind: str = ""
    phase: str = ""
    outcomes: List[TaskOutcome] = field(default_factory=list)
    batches_dispatched: List[int] = field(default_factory=list)
    stopped_at_batch: Optional[int] = None
    #: F4: what `driver.verify_command` did, or ``None`` when it never ran (no
    #: command configured, a markdown phase, or no batch applied). Present with a
    #: non-zero ``exit_code`` is the one shape that costs a fully applied run its
    #: completion, so it is a field on the result rather than a log line.
    verification: Optional[Dict[str, Any]] = None
    human_plan: HumanTaskPlan = field(default_factory=HumanTaskPlan)
    finalized: bool = False
    summary_file: Optional[str] = None
    report_path: Optional[str] = None
    dry_run_path: Optional[str] = None
    messages: List[str] = field(default_factory=list)
    #: Plan step 3 (2026-09-30): ``task_id -> {sandbox_rung, jail_denials}``
    #: for a write batch that ran on a fallback rung. §7 step 2 subtask item 3:
    #: from step 3 "a write batch's recorded outcome also names the rung:
    #: applied, failed, partial or timed out" — and ``TaskOutcome`` is the
    #: executor's row, so the scheduler's own report carries it beside that
    #: row rather than in it. Absent for a batch on codex's own sandbox or on
    #: another runner, so those reports are byte-identical to before.
    sandbox_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: #28 (2026-10-02): a ``--resume`` re-materializes only the unfinished tasks,
    #: so its reports described the resumed stretch alone and the first run's
    #: T001-T008 rows were overwritten and gone. These are the previous
    #: ``write_batch_report.json``'s task rows, read before this run replaces it
    #: (and already merged if that run was itself a resume), so the phase's
    #: report and summary cover the whole phase.
    earlier_rows: List[Dict[str, Any]] = field(default_factory=list)
    #: ``task_id -> reason`` for every task of this run that was never
    #: dispatched because the loop stopped first (racing re-run 3, 2026-10-05:
    #: two such tasks were absent from the summary, the counts and the
    #: message). Set by ``_record_stop``; empty on a run that did not stop.
    not_run: Dict[str, str] = field(default_factory=dict)
    #: The stop, as one record: which task, what failed, who was not run and
    #: the way forward — what the summary, the status mirror and the result
    #: line's ``error`` all render from, so they cannot disagree.
    stop: Optional[Dict[str, Any]] = None
    #: ``task_id -> check task id`` for every task this run did not dispatch
    #: because it depends (transitively) on a check task that reported
    #: problems (fix round 11: the dependents ran on code the check had just
    #: rejected). Each is in ``not_run`` too, with
    #: :func:`blocked_by_check_reason` as its reason.
    blocked_by_check: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == EXIT_OK

    @property
    def failed(self) -> List[TaskOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def check_findings(self) -> Dict[str, List[str]]:
        """``{task_id: findings}`` for every check task that finished ``ok``
        with problems, this run's and the earlier runs' rows alike, in row
        order (racing re-run 4)."""
        found: Dict[str, List[str]] = {}
        for row in self.report_rows():
            findings = row_check_findings(row)
            if row.get("status") == STATUS_OK and findings:
                found[str(row.get("task_id"))] = findings
        return found

    def report_rows(self) -> List[Dict[str, Any]]:
        """Every task row of the phase: earlier runs' first, this run's after.

        This run's row replaces an earlier one for the same batch, except a
        resume *skip*: that row only says "already done", and the earlier run's
        row is the one that holds what was done, so it stays (flagged
        ``from_earlier_run``). "The same batch" is :func:`_same_batch_row`'s:
        the same items, not merely the same id, because an apply kind's ids
        are positional and a resume whose selection changed hands
        ``apply-batch-1`` to other items.
        """
        current = [
            {**o.to_dict(), **self.sandbox_records.get(o.task_id, {})}
            for o in self.outcomes
        ]
        rows: List[Dict[str, Any]] = []
        stood_in: set = set()
        for earlier in self.earlier_rows:
            match = next(
                (index for index, row in enumerate(current)
                 if _same_batch_row(earlier, row)),
                None,
            )
            if match is not None and not current[match].get("skipped"):
                continue
            if match is not None:
                stood_in.add(match)
            rows.append({**earlier, "from_earlier_run": True})
        rows.extend(
            row for index, row in enumerate(current) if index not in stood_in
        )
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": 1,
            "kind": self.kind,
            "phase": self.phase,
            # When this report was written, so a later `--resume` can tell
            # whether its rows belong to the current review cycle
            # (`_earlier_report_rows`); the file's mtime is the fallback for
            # a report that predates the key.
            "written_at": datetime.now().isoformat(),
            "exit": self.exit_code,
            "batches_dispatched": list(self.batches_dispatched),
            "stopped_at_batch": self.stopped_at_batch,
            "verification": self.verification,
            "finalized": self.finalized,
            "summary_file": self.summary_file,
            "human_task_strategy": self.human_plan.strategy,
            "skipped_tasks": dict(self.human_plan.skipped),
            "dependency_overrides": {
                k: list(v) for k, v in self.human_plan.overrides.items()
            },
            "pending_human_tasks": list(self.human_plan.pending),
            "accepted_tasks": dict(self.human_plan.accepted),
            "not_run_tasks": dict(self.not_run),
            "blocked_by_check": dict(self.blocked_by_check),
            "stop": copy.deepcopy(self.stop) if self.stop else None,
            "tasks": self.report_rows(),
        }


@dataclass
class UndeclaredWrites:
    """What one dispatch's containment sweep saw outside the declared set.

    Two lists rather than one, because "observed" and "reversible" are different
    facts about the same path and collapsing them is how a rollback point that
    does not exist gets promised. ``paths`` is the whole observed set — it is what
    the post-run ``WritePolicy`` check runs on and what the audit records.
    ``restorable`` is the subset whose pre-run bytes (or recorded absence) made it
    into the task's snapshot, and therefore the only part that may join
    ``files_touched``, which is what ``util restore-batch`` reverts.
    """

    paths: List[str] = field(default_factory=list)
    restorable: List[str] = field(default_factory=list)
    #: Whether a census was taken on BOTH sides of the dispatch. ``False`` is
    #: "the sweep could not look", which is not the same fact as an empty
    #: ``paths`` (adversarial review, 2026-10-02: a verification-only task
    #: read the sweep-off case as "no file changed" and was VERIFIED).
    observed: bool = True

    def __bool__(self) -> bool:
        return bool(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def unrecoverable(self) -> List[str]:
        keepable = set(self.restorable)
        return [path for path in self.paths if path not in keepable]


def bind_prior_changes(
    task: Dict[str, Any], *, output_dir: str, phase: Optional[str]
) -> str:
    """The prompt that actually goes to the runner, bound *now*.

    ``{prior_changes_context}`` is substituted from the log **on disk**, so a
    resumed batch sees the same context an uninterrupted run would have. The
    task JSON is never rewritten: this produces the dispatched prompt only, and
    the audit record stores its hash and length rather than its bytes.

    A prompt that never contained the token is a contract error, not a
    convenience: every batcher embeds it (``utils/suggestion_batcher.py:359``,
    ``utils/code_fix_batcher.py:307``, ``apply_task_suggestions_orchestrator.py:543``,
    ``prompts/implementation_task.txt``), so its absence means the producer
    changed and every later batch would run without prior-batch context while
    looking fine. ``implement`` used to be the one kind exempted — its prompt
    carried no token and this fell through as a no-op — on the theory that its
    inter-task context is the dependency section
    ``implement_orchestrator._build_dependency_section`` renders instead. That
    section names only the *titles* of preceding tasks (F6, 2026-09-16): a
    verified one-line summary of what each batch actually did already exists in
    ``prior_changes.jsonl``, so implement prompts now carry the token too and
    get the same binding, refusal included, as every other kind.

    The render is scoped to *phase* (N8, 2026-09-16): the log is one file
    shared by every write phase this plan has ever run, and an unscoped tail
    let another phase's older batches crowd out this phase's own recent ones.

    Module-level (2026-10-05) because it has two callers that must produce
    byte-identical text for the same state: the scheduler's
    :meth:`WriteBatchScheduler.dispatched_prompt` on the executor leg, and
    ``util write-prompt`` (:mod:`utils.write_prompt`) on the native leg, where
    the host used to substitute the token by hand — and, measured on the
    2026-10-04 Claude Code racing re-run, sometimes did not.
    """
    prompt = str(task.get("prompt") or "")
    if PRIOR_CHANGES_TOKEN not in prompt:
        raise ResultRejected(
            f"Task {task.get('task_id')}: its prompt does not contain the "
            f"{PRIOR_CHANGES_TOKEN} token, so there is nowhere to bind the "
            f"prior-batch context. Every apply/implement prompt must carry it "
            f"(utils/suggestion_batcher.py, utils/code_fix_batcher.py, "
            f"apply_task_suggestions_orchestrator.py, "
            f"prompts/implementation_task.txt). Nothing was dispatched.",
            STATUS_REJECTED,
        )
    context = _prior_changes.render(output_dir, phase=phase)
    return prompt.replace(PRIOR_CHANGES_TOKEN, context)


def with_result_contract(kind: str, task: Dict[str, Any], prompt: str) -> str:
    """Append the write-batch result contract the classification reads.

    The batchers' prompts end in the prose "Return Format" the native
    markdown flow compresses into a one-line summary; the scheduler — on
    both invokers — validates the reply against
    ``write_batch_result.schema.json`` instead, and reads ``ids`` against
    the task's ``expected_ids``. A runner that was never shown that shape
    answers in prose and fails rung 5 with its edits on disk. Measured
    2026-09-15 on ``apply-task-suggestions``: three attempts applied the
    batch to ``tasks.md`` and reported it as asked, every one was recorded
    ``unparseable_result`` / ``schema-rejection``, and a fourth run's
    runner, reading the already-applied file, replied "already addressed"
    in prose and failed the same way.

    The ids are spelled out in the order the items appear in the prompt,
    because the prompt itself never names them; a reply that omits an id
    for an item it found already applied would otherwise read as a partial
    application and stop the loop.

    Module-level for the same reason as :func:`bind_prior_changes`: the
    native leg's ``util write-prompt`` appends the same contract, so a host
    subagent's closing JSON object carries the ``summary`` the prior-changes
    append wants and the ``status``/``files_touched`` the host's verification
    step reads.
    """
    expected = [str(i) for i in (task.get("expected_ids") or [])]
    if kind == "implement":
        id_lines = (
            f"- `ids`: this task's id, `{expected[0]}`, when you applied it "
            f"or found it already implemented; leave it out on `failed`."
            if expected else
            "- `ids`: the task ids you applied."
        )
    else:
        item = "fix" if kind == "apply_code_fixes" else "suggestion"
        listed = ", ".join(
            f"{item} {index} = `{member}`"
            for index, member in enumerate(expected, 1)
        )
        id_lines = (
            f"- `ids`: the ids of the {item}s you applied **or found already "
            f"applied by a prior batch**, from this list (numbered in the "
            f"order they appear above): {listed}. Leave out only the ones "
            f"you could not apply."
        )
    return (
        f"{prompt.rstrip()}\n\n"
        "## Process safety (required)\n\n"
        "Never run `kill`, `pkill`, `killall` or any process-wide signal: "
        "the orchestrator that dispatched you is a sibling process, and a "
        "write subagent's `killall -9 python3` ended a whole phase "
        "(measured 2026-09-16). Do not start background servers or "
        "browsers, and leave no process running when you finish; verify "
        "with static checks only.\n\n"
        "## Scratch files (advisory)\n\n"
        "Keep scratch work (an extracted script to syntax-check, a probe) "
        "out of the repository and out of fixed, shared names such as a "
        "bare `check.mjs` in the system temp directory: create a fresh "
        "directory with `mktemp -d`, work only there and delete it before "
        "you finish. The write gate only sees the repository, so a file "
        "left elsewhere is invisible to it (measured 2026-10-01, OpenCode "
        "runner).\n\n"
        "## Result (required — this is how the batch is verified)\n\n"
        "Do the work above, then end your reply with ONE JSON object (a "
        "fenced ```json block is fine) of exactly this shape:\n\n"
        "```json\n"
        '{"status": "applied", "ids": ["<id>", "..."], '
        '"files_touched": ["<repo-relative path>"], '
        '"summary": "<one line, at most 200 characters>", '
        '"notes": "<the per-item summary requested above>"}\n'
        "```\n\n"
        "- `status`: `applied` if you edited any file; `no-op` if every "
        "change was already present and you edited nothing; `failed` if "
        "you could not apply the batch.\n"
        f"{id_lines}\n"
        "- `files_touched`: repo-root-relative paths you edited (empty for "
        "`no-op` and `failed`).\n"
        "- `summary`: one line; it is recorded verbatim as this batch's "
        "entry in the prior-changes log the next batch reads.\n"
        "- `notes`: the per-item summary the return format above asks for.\n\n"
        "A reply without this JSON object is recorded as a FAILED batch even "
        "when the edits landed, so send it in every case — including when "
        "nothing needed changing.\n"
        + (CHECK_RESULT_CONTRACT if is_check_task(kind, task) else "")
    )


#: Appended to the result contract of a check task (an ``implement`` task that
#: declares no files): the structured check result the verdict reads
#: (``subagent_executor.check_findings_from``). Racing re-run 4 (2026-10-05):
#: T007's ``npx html-validate`` reported two problems, the runner said
#: ``no-op`` and wrote them only in ``notes``, and the task read VERIFIED.
#: Fix round 11: a check that could not run is ``failed`` (it used to be asked
#: for as a finding, so a missing ``npx`` read as findings the run went on
#: past), and the tasks that depend on a check with findings wait for it.
CHECK_RESULT_CONTRACT = (
    "\n## Check result (required for this task)\n\n"
    "This task declares no files: it is a check, and it must not change any "
    "file. Run every check it names.\n\n"
    "If every check ran, answer `status: \"no-op\"` and add a `check` object "
    "to the JSON above:\n\n"
    "```json\n"
    '{"check": {"passed": false, "findings": ["<one problem: the command, '
    'what it reported, the file and line>"]}}\n'
    "```\n\n"
    "- `passed`: the JSON boolean `true` only when every check the task names "
    "ran and met its acceptance criteria, else `false`.\n"
    "- `findings`: one string per problem a check reported; `[]` when "
    "`passed` is `true`.\n\n"
    "A problem a check finds is a finding, not a failure: do not fix it and "
    "do not answer `failed` for it. The findings are recorded for the code "
    "review, and the tasks that depend on this check wait until they are "
    "fixed or accepted.\n\n"
    "If a check could not run — its command is missing, it crashed, it timed "
    "out or it printed no result — answer `status: \"failed\"` and say in "
    "`summary` which check could not run and why. Never list a check that "
    "did not run as a finding: it says nothing about the code, and `failed` "
    "stops the run so the cause can be fixed.\n"
)


def is_check_task(kind: str, task: Dict[str, Any]) -> bool:
    """Whether *task* is a verification-only ``implement`` task (no files)."""
    return kind == "implement" and not snapshot_file_set(task)


class WriteBatchScheduler:
    """Drives one write phase's batches in ``batch_index`` order (D3n).

    Args:
        tasks: Materialized write tasks (any ``batch_index`` order — sorted here).
        phase_dir: Where results, snapshots, the audit log and both reports land.
        kind: One of ``utils.subagent_tasks.WRITE_KINDS``.
        output_dir: The plan's output dir, which owns ``prior_changes.jsonl``.
        repo_root: The pinned root (D3h). Resolved from ``anchor`` when omitted.
        state: A ``StateManager``, for task statuses, override records and the
            phase-completed marker. Optional only so a unit test can drive the loop
            without one; a real phase always has one.
        host_max: The host's ``max_concurrent_subagents``. ``None`` means "look it
            up from ``hosts/capabilities.yaml`` for the host this invocation
            declared" (D5b) — the same seam, with the same meaning, as
            ``subagent_executor.effective_ceiling``'s parameter of the same name.
            No orchestrator passes it; a caller that already knows the number
            still may.
        respond: When given, the **simulated native invoker** runs instead of a
            subprocess: ``(task, dispatched_prompt) -> result JSON``. Same
            containment, same classification, same prior-changes append — which is
            what makes "either invoker, same terminal state" a property of the code.
        dry_run: ``--dry-run-subagents``. Renders every command, applies every
            admission and policy check, writes ``subagent_dry_run.json`` and starts
            **no** child.
        write_target: The phase's one write target — ``write_phase_artifact()``,
            the plan for apply-suggestions and the tasks file for
            apply-task-suggestions (plan step 3, F1 "How a dispatch receives its
            grant", 2026-09-30). ``realpath``-resolved and checked here, at
            construction: it must exist, be a regular file, not be a symlink and
            lie under the repo root, or nothing runs. For the ``plan-file`` and
            ``tasks`` scopes it is the grant's target, and every task's declared
            set must resolve to exactly it, on every rung, before its snapshot.
            ``None`` for implement and apply-code-fixes (``repo-wide`` is one
            dir rule on the repository root, with this run's output folder on
            the request for the developer message, never a target); a
            ``plan-file``/``tasks`` phase
            built without one carries no grant, and the executor refuses it on
            the rung.
    """

    def __init__(
        self,
        *,
        tasks: Sequence[Dict[str, Any]],
        phase_dir: str,
        kind: str,
        output_dir: str,
        repo_root: Optional[str] = None,
        phase: Optional[str] = None,
        state: Any = None,
        plan_file: Optional[str] = None,
        summary_file: Optional[str] = None,
        summary_file_relative: Optional[str] = None,
        anchor: Optional[str] = None,
        host_max: Optional[int] = None,
        runner: Optional[str] = None,
        resume: bool = True,
        dry_run: bool = False,
        human_task_strategy: Optional[str] = None,
        human_task_decisions: Optional[Dict[str, str]] = None,
        respond: Optional[Callable[[Dict[str, Any], str], Any]] = None,
        finalize: bool = True,
        log: Optional[Callable[[str], None]] = None,
        write_target: Optional[str] = None,
        user_commands: Optional[Dict[str, str]] = None,
    ) -> None:
        if kind not in WRITE_KINDS:
            raise ExecutorError(
                f"{kind!r} is not a write kind (known: {', '.join(WRITE_KINDS)}). "
                f"Nothing was dispatched."
            )
        self.tasks = list(tasks)
        self.phase_dir = str(phase_dir)
        self.kind = kind
        self.phase = phase or PHASE_BY_WRITE_KIND[kind]
        self.output_dir = str(output_dir)
        self.anchor = anchor or plan_file
        # An explicit root is checked the way the resolver pins one (review
        # fixes 2026-10-05): a planted `core.worktree` or a redirecting `.git`
        # refuses the phase at construction, before any snapshot or dispatch.
        self.repo_root = (
            self._checked_repo_root(repo_root) if repo_root
            else resolve_repo_root(self.anchor)
        )
        self.state = state
        self.plan_file = plan_file
        self.summary_file = summary_file
        self.summary_file_relative = summary_file_relative
        # `host_max=None` means *look it up*, exactly as it does for the read
        # path's `subagent_executor.effective_ceiling` — never "this phase has no
        # host leg". Reading it as the latter cost the write half its host leg
        # outright: no orchestrator passes `host_max` (both construction sites are
        # keyword-complete without it), so `write_ceiling`'s host branch never
        # ran on a real invocation and every write batch report recorded a null
        # where the read half records a number. Resolved here rather than
        # per-call so `write_ceiling`, the batch report and the dry-run record
        # cannot disagree about which number bound the phase.
        self.host_max = _host_ceiling() if host_max is None else host_max
        self.forced_runner = runner
        self.resume = resume
        self.dry_run = dry_run
        self.human_task_strategy = human_task_strategy
        self.human_task_decisions = dict(human_task_decisions or {})
        self.respond = respond
        self.finalize = finalize
        self.log = log or (lambda line: print(line))
        #: The way forward after a stop, spelled the way this host's user
        #: types it (the orchestrator builds them from the host's
        #: `next_command` template; the scheduler knows no host): `resume`,
        #: `review`, `fix`, and the two `--human-task` variants `accept` /
        #: `skip` with `{task_id}` left for the failed task. Absent (a direct
        #: invocation), the stop message names the flags instead.
        self.user_commands: Dict[str, str] = dict(user_commands or {})

        # Plan step 3 (2026-09-30): the grant a write dispatch on the
        # plugin-landlock rung carries. ONE request for the whole phase — the
        # scope from the kind map, the target from the phase's own artifact —
        # never a per-task path, so nothing a task file says can widen it.
        self.write_scope = write_scope_for_kind(kind)
        self.write_target = self._checked_write_target(write_target)
        self.policy = WritePolicy.load(self.repo_root, anchor=self.anchor)
        self.jail_request = self._build_jail_request()
        self.audit = AuditLog(self.phase_dir)
        #: Whether `_decide_rollback_storage` ran for this phase.
        self._rollback_storage_decided = False
        self.snapshots: Dict[str, TaskSnapshot] = {}
        #: Task ids whose pre-batch snapshot was REUSED from an earlier
        #: dispatch of this same batch (`_snapshot`/`_reusable_snapshot`),
        #: rather than taken fresh at the top of this dispatch. `_classify`
        #: reads this to tell "the diff is a killed attempt's, not this
        #: dispatch's" apart from "this dispatch's own snapshot disagrees
        #: with its no-op claim" (N2, 2026-09-16) — the two read identically
        #: from the bytes alone.
        self._snapshot_reused: set = set()
        self.invoker = "native-simulated" if respond is not None else "executor"
        #: Serializes the per-dispatch bookkeeping (`_record_batch_effects`).
        #: With `driver.max_concurrent_write_subagents` > 1 a disjoint group runs
        #: in a thread pool, and every member reaches the SAME in-memory
        #: `StateManager` and the SAME `prior_changes.jsonl`. Both are
        #: read-modify-write, so two unsynchronized siblings would race on
        #: `state.save()`'s `os.replace` — last writer wins and a sibling's task
        #: status is silently dropped, leaving resume and the report disagreeing
        #: with what actually ran.
        self._effects_lock = threading.Lock()
        #: Dispatched-or-skipped tasks finished so far, the `[ETA]` line's
        #: numerator (F3, 2026-09-17). It counts TASKS because the denominator
        #: beside it does — `len(self.tasks) - human skips`, the same total
        #: `_mirror_status` publishes — and because an `implement` batch is a
        #: dependency level with k tasks in it: `len(batches_dispatched) + 1`
        #: repeated the same ordinal for every member of such a batch, so a
        #: parallelizable plan could never reach `N/N` or print
        #: `ALL BATCHES COMPLETE`. Read and incremented under `_effects_lock`
        #: because a disjoint group finishes on several threads.
        self._tasks_finished = 0
        #: Task ids for which a terminal (`completed`) audit line has been
        #: written, so `_audit_terminal` can close out a dispatch that never
        #: reached `_classify` without ever double-recording one that did.
        self._audited: set = set()
        #: The post-run containment sweep's own state: the pre-dispatch census
        #: per in-flight task, and each in-flight task's OWN declared set —
        #: added on dispatch (`_begin_sweep`), removed on classification
        #: (`_undeclared_writes`). A co-scheduled sibling writes its OWN
        #: declared files while this task is being classified, so those paths
        #: are subtracted before the sweep's result reaches the policy —
        #: otherwise a disjoint group would report each other's legitimate
        #: writes as undeclared. F13: this used to be one set unioned for the
        #: whole PHASE and never pruned, so once any task declared a path, a
        #: LATER task's undeclared write to that same path was invisible for
        #: the rest of the run — the justification above only ever needed the
        #: in-flight sets.
        self._sweep_lock = threading.Lock()
        self._census_before: Dict[str, Optional[Dict[str, Tuple[int, int]]]] = {}
        self._declared_by_task: Dict[str, set] = {}
        # Each census spans the whole dispatch, so a sibling's legitimate
        # writes remain attributable to that sibling even if it finishes first.
        # Keep overlap declarations per census, never for the entire phase.
        self._sweep_ignored: Dict[str, set] = {}
        self._sweep_warned = False
        #: The ignore rules in force when each in-flight dispatch started:
        #: ``source path -> sha256 | None (absent)`` over every `.gitignore`
        #: the census saw, `.git/info/exclude`, `.git/config` and the global
        #: excludes file. `_ignored_and_harmless` drops a path from the sweep
        #: only while these are what they were at dispatch — the runner can
        #: write any of them, and a drop decided on post-run rules hid
        #: everything a runner-written `*` covered (adversarial review,
        #: 2026-10-02).
        self._ignore_sources_before: Dict[str, Dict[str, Optional[str]]] = {}
        self._global_excludes_path: Optional[str] = None
        self._global_excludes_resolved = False
        #: The rollback half of the same sweep. Observing an undeclared write is
        #: worth little if it cannot be undone, and the snapshot only ever held
        #: the declared set, so the pre-run bytes of everything else the runner
        #: could reach are retained here before dispatch:
        #: `_retained` is the run-wide `relative -> (stamp, digest|None)` cache
        #: that keeps a file from being re-hashed while its (size, mtime_ns) is
        #: unchanged, `_retained_before` is the per-in-flight-task view a sibling
        #: dispatch cannot overwrite, and `_retained_bytes` accounts the store
        #: against `SWEEP_RETAIN_MAX_BYTES`. `_counted_digests` is what keeps
        #: that accounting honest across an inherited store (F12): a blob
        #: already on disk — this run's own earlier copy, or one a killed run
        #: left behind — costs real bytes on disk whether or not `_store_blob`
        #: copies anything this call, so each digest is counted once, on
        #: whichever call finds it first, rather than only on the call that
        #: wrote it.
        self._retained: Dict[str, Tuple[Tuple[int, int], Optional[str]]] = {}
        self._retained_before: Dict[str, Dict[str, str]] = {}
        self._retained_bytes = 0
        self._counted_digests: set = set()
        self._retain_warned = False
        #: ``relative -> denied by driver.write_paths?`` for every path the
        #: sweep kept no bytes for (the policy does not change during a run,
        #: so each path is asked once). The denied ones are named once, on the
        #: closing utilisation line, rather than in a warning of their own
        #: (see :meth:`_retain`).
        self._unretained_denied: Dict[str, bool] = {}
        #: ``{spec: version}`` for every distinct runner this run actually
        #: preflighted (F12), populated as ``_dispatch_executor`` dispatches —
        #: the read path fills the same-shaped field from
        #: ``_preflight_batch_runners``; the write path preflights inline, per
        #: dispatch, and this is where its return value was being discarded.
        self._runner_versions: Dict[str, str] = {}
        #: ``task_id -> dispatched_at`` (ISO, `_begin_sweep`'s own clock) for a
        #: dispatch that has started and not yet been classified (F10) — what
        #: lets the mirror report "in flight since Ns" through a batch that
        #: runs for minutes instead of reading `dispatched: 0` the whole time,
        #: indistinguishable from a stall. Popped in step with
        #: `_declared_by_task`/`_census_before` in `_undeclared_writes`, so a
        #: task that never reaches classification (an exception raised before
        #: `_classify`) is the one case this can go stale — the run's next
        #: mirror write self-heals it, or the process is gone and the
        #: heartbeat says so.
        self._dispatch_started: Dict[str, str] = {}
        #: Plan step 3 (2026-09-30), all keyed by task id. The rung each
        #: executor dispatch ran on (``None``: codex's own sandbox or another
        #: runner) — what names the rung on the batch's line, its verdict and
        #: the report whatever the outcome, a timeout included.
        self._dispatch_rungs: Dict[str, Optional[str]] = {}
        #: The write target's ``(st_dev, st_ino)`` recorded right after this
        #: dispatch's pre-batch snapshot — never the first run's, so a
        #: ``util restore-batch`` between runs (whose ``os.replace`` leaves a
        #: new inode) does not read as a replaced target (§8.3 item 14).
        self._target_identity_before: Dict[str, Optional[Tuple[int, int]]] = {}
        #: The jail-denial count that did not change a verdict (out-of-scope
        #: denials, or a replacement the runner recovered from), for the
        #: batch's line and its outcome's reason.
        self._jail_notes: Dict[str, str] = {}
        #: The denials themselves, for the report row.
        self._jail_denials: Dict[str, Dict[str, Any]] = {}
        #: Root grant (2026-10-05): ``task_id -> git_utils.git_metadata_fingerprint``
        #: taken right before a repo-wide dispatch, compared after it for the
        #: warn-only ``[GIT_CONFIG_CHANGED]`` line (never a copy, never a revert).
        self._git_meta_before: Dict[str, Optional[Dict[str, Any]]] = {}
        #: Tasks whose verdict came from the jail (a denied edit or a replaced
        #: target) and failed the batch: no prior-changes entry is appended for
        #: them. ``prior_changes.append`` is idempotent on ``(id, phase)``, so
        #: a "(batch failed — …)" line written now would silently drop the
        #: entry the ``--resume`` that re-dispatches this batch writes.
        self._jail_failed: set = set()
        #: Batch indexes whose stray-writes-only line was already printed.
        self._rung_noticed: set = set()

    # -- the write grant (plan step 3, 2026-09-30) ----------------------

    @property
    def safety_dir(self) -> str:
        """Where this phase's pre-batch snapshots, rollback store and audit
        log live (``write_safety.safety_dir``): the phase dir, or the
        out-of-grant directory a jailed repo-wide run claimed."""
        return safety_dir(self.phase_dir)

    def _decide_rollback_storage(self, task: Dict[str, Any], profile: str) -> None:
        """Once per phase, before its first snapshot: a ``repo-wide`` batch
        that will run on the plugin's Landlock jail keeps its rollback
        evidence outside the grant (review fixes 2026-10-05).

        Under the root grant the phase dir is the batch's to write, and the
        manifest, payloads, rollback blobs and audit log there decided what
        ``util restore-batch`` wrote and where. The jail restricts writes
        only, so no key could be kept from the batch; the evidence goes where
        the jail never grants — ``write_safety.claim_external_safety_dir``,
        under the state home — and every reader resolves it through
        ``safety_dir``, which never falls back to the phase dir once the
        directory exists. The rung is read the way the dispatch reads it,
        from the executor's recorded decision after this runner's preflight
        (cached per process there); a preflight that fails is left for the
        dispatch's own call to report. Every other runner and scope keeps
        the phase dir, unchanged.
        """
        if self._rollback_storage_decided or self.dry_run or self.respond is not None:
            return
        if self.write_scope != WRITE_SCOPE_REPO_WIDE:
            self._rollback_storage_decided = True
            return
        self._rollback_storage_decided = True
        try:
            runner = self.forced_runner or resolve_task_runner(task, anchor=self.anchor)
            self._preflight(runner, profile)
        except Exception:  # noqa: BLE001 - the dispatch's own preflight reports it
            return
        if self._recorded_rung(runner, profile) != SANDBOX_RUNG_PLUGIN_LANDLOCK:
            return
        external = claim_external_safety_dir(self.phase_dir, self.repo_root)
        self.log(
            f"[write-subagents] repo-wide batches run on the plugin's jail, "
            f"which grants the phase dir too: pre-batch snapshots, the rollback "
            f"store and the audit log for this phase live outside it, at "
            f"{external} (a copy inside the phase dir is never read)."
        )

    @staticmethod
    def _checked_repo_root(raw: str) -> str:
        """``raw`` ``realpath``-resolved, refused (exit 3) when
        ``git_utils.pin_repo_root`` says git's view of that tree is
        redirected — the check :func:`resolve_repo_root` makes for a root it
        derives itself, applied to one a caller hands in."""
        root = os.path.realpath(str(raw))
        _pinned, problem = _git_utils.pin_repo_root(root)
        if problem:
            raise WriteSafetyError(
                f"Refusing the repository root {root}: {problem}. Nothing was "
                f"dispatched."
            )
        return root

    def _checked_write_target(self, raw: Optional[str]) -> Optional[str]:
        """``raw`` ``realpath``-resolved, or refuse the phase at construction.

        F1 "How a dispatch receives its grant": the target must exist, be a
        regular file, not be a symlink (checked on the path as given, before
        resolution) and lie under the repo root. A target that fails any of
        these is a phase that cannot be granted, so it fails here with exit 3
        and nothing dispatched, rather than as a jail refusal mid-run.
        """
        if raw is None or not str(raw).strip():
            return None
        given = os.path.abspath(str(raw))
        problem = None
        if os.path.islink(given):
            problem = "is a symlink (a grant is never bound through one)"
        elif not os.path.exists(given):
            problem = "does not exist"
        elif not os.path.isfile(given):
            problem = "is not a regular file"
        resolved = os.path.realpath(given)
        root = os.path.realpath(self.repo_root)
        if problem is None and not resolved.startswith(root.rstrip(os.sep) + os.sep):
            problem = f"lies outside the repository root {root}"
        if problem is not None:
            raise ExecutorError(
                f"The {self.phase} write target {given} {problem}, so this "
                f"phase cannot be granted its one file. Nothing was dispatched."
            )
        return resolved

    def _build_jail_request(self) -> Optional[JailRequest]:
        """The phase's one :class:`JailRequest` (F1 "How a dispatch receives
        its grant").

        ``plan-file`` and ``tasks`` carry the target; ``tasks`` also carries a
        ``target_dir`` — a dir rule — only when the tasks file sits in the
        plan's default ``plans/<stem>/tasks/`` (``default_tasks_dir``), and a
        file rule on the tasks file anywhere else. ``repo-wide`` (step 3b;
        one dir rule on the root since 2026-10-05) carries no target but this
        run's output folder, which the developer message names as hands-off.
        """
        if not self.write_scope:
            return None
        repo_root = os.path.realpath(self.repo_root)
        if self.write_scope == WRITE_SCOPE_REPO_WIDE:
            return JailRequest(
                repo_root=repo_root, kind=self.kind, write_scope=self.write_scope,
                output_dir=os.path.realpath(self.output_dir),
            )
        if self.write_scope not in WRITE_SCOPES_ON_RUNG:
            return JailRequest(
                repo_root=repo_root, kind=self.kind, write_scope=self.write_scope,
            )
        target_dir = None
        if (
            self.write_scope == WRITE_SCOPE_TASKS
            and self.write_target and self.plan_file
        ):
            target_dir = _landlock.default_tasks_dir(
                self.plan_file, self.write_target
            )
        return JailRequest(
            repo_root=repo_root, kind=self.kind, write_scope=self.write_scope,
            target=self.write_target, target_dir=target_dir,
        )


    # -- the repository metadata a repo-wide batch can write (root grant) ----

    def _git_metadata_fingerprint(self) -> Optional[Dict[str, Any]]:
        """``git_utils.git_metadata_fingerprint`` of the pinned root for a
        ``repo-wide`` batch (a hash and a listing, never bytes); ``None`` for
        the narrower scopes, whose grant never reaches ``.git/``, and on any
        failure — a warning line may not fail a dispatch."""
        if self.write_scope != WRITE_SCOPE_REPO_WIDE:
            return None
        try:
            return _git_utils.git_metadata_fingerprint(self.repo_root)
        except Exception:  # noqa: BLE001 - observability must not fail a phase
            return None

    def _warn_git_metadata_changed(self, task_id: str) -> None:
        """One warn-only ``[GIT_CONFIG_CHANGED]`` line when ``.git/config`` or
        the ``.git/hooks/`` listing moved across this dispatch (root grant,
        2026-10-05: the sweep prunes ``.git/`` and nothing is backed up, so
        this line is the one place a planted hook or config key is named).
        The user's own git honours what was planted; the plugin's calls carry
        ``git_utils.GIT_HARDENING_ARGS``. Never a failed batch."""
        before = self._git_meta_before.pop(task_id, None)
        if before is None:
            return
        try:
            change = _git_utils.describe_git_metadata_change(
                before, self._git_metadata_fingerprint()
            )
        except Exception:  # noqa: BLE001 - observability must not fail a phase
            return
        if change:
            self.log(
                f"{_git_utils.GIT_CONFIG_CHANGED_MARKER} {task_id}: the repository's "
                f"git metadata changed during this batch ({change}). The jail "
                f"grants the whole repository and nothing was backed up; "
                f"inspect .git/config and .git/hooks/ before your next git "
                f"command runs what a batch may have planted (warn-only)."
            )

    def _grant_mismatch(self, task: Dict[str, Any]) -> Optional[str]:
        """Why ``task``'s declared set is not this phase's grant, or ``None``.

        Under ``plan-file`` and ``tasks`` the declared set must resolve to
        exactly the target, on every rung (F1 "How a dispatch receives its
        grant"). ``materialize_write_tasks`` always declares exactly the target,
        so only an edited task file differs — and the task file is re-read on
        ``--resume``, so it is checked here and never supplies a grant.
        """
        if self.write_scope not in WRITE_SCOPES_ON_RUNG or not self.write_target:
            return None
        declared = declared_file_set(task)
        resolved = set()
        for path in declared:
            candidate = path if os.path.isabs(path) else os.path.join(
                self.repo_root, path
            )
            resolved.add(os.path.realpath(candidate))
        if resolved == {self.write_target}:
            return None
        shown = ", ".join(declared) or "nothing"
        return (
            f"Refusing to dispatch write task {task.get('task_id')}: it declares "
            f"{shown}, but a {self.kind} batch's grant ({self.write_scope}) is "
            f"exactly this phase's write target {self.write_target}. Only an "
            f"edited task file declares anything else, and a task file never "
            f"supplies a grant. Nothing was snapshotted or dispatched."
        )

    def _identity_path(self) -> Optional[str]:
        """What the replaced-target check watches: the target file under a
        file rule, the tasks directory under the dir rule (F1 "Supported edits
        under a file-only grant", "The target replaced during the dispatch")."""
        request = self.jail_request
        if request is None or request.write_scope not in WRITE_SCOPES_ON_RUNG:
            return None
        return request.target_dir or request.target

    @staticmethod
    def _identity_of(path: Optional[str]) -> Optional[Tuple[int, int]]:
        """``(st_dev, st_ino)`` of ``path`` itself (``lstat``), or ``None``."""
        if not path:
            return None
        try:
            info = os.lstat(path)
        except OSError:
            return None
        return (info.st_dev, info.st_ino)

    def _target_label(self) -> str:
        request = self.jail_request
        if request is not None and request.target_dir:
            return "the tasks directory"
        if self.write_scope == WRITE_SCOPE_TASKS:
            return "the tasks file"
        return "the plan file"

    def _preflight(
        self, runner: str, profile: str, **extra: Any
    ) -> Dict[str, Any]:
        """``preflight_runner`` with this phase's request when it takes one.

        For ``agent-write`` the request's scope joins the executor's sandbox
        decision key, so a ``plan-file`` decision is never replayed for a
        ``repo-wide`` dispatch of the same runner.
        """
        kwargs: Dict[str, Any] = dict(
            profile=profile, anchor=self.anchor, key="driver.subagent_runner",
            **extra,
        )
        if self.jail_request is not None and _accepts_keyword(
            preflight_runner, "jail_request"
        ):
            kwargs["jail_request"] = self.jail_request
        return preflight_runner(runner, **kwargs)

    def _recorded_rung(self, runner: str, profile: str) -> Optional[str]:
        """The fallback rung the preflight recorded for this runner, profile
        and scope, or ``None`` (codex's own sandbox, or another runner)."""
        lookup = getattr(_executor, "sandbox_rung", None)
        if lookup is None:
            return None
        try:
            if self.write_scope and _accepts_keyword(lookup, "write_scope"):
                rung = lookup(runner, profile, write_scope=self.write_scope)
            else:
                rung = lookup(runner, profile)
        except Exception:  # noqa: BLE001 - a lookup must never fail a dispatch
            return None
        return str(rung) if rung else None

    def _announce_rung(self, runner: str, batch_index: Optional[int]) -> None:
        """Print the stray-writes-only line once before a batch on the rung.

        The executor composes it (``write_rung_notice``, the same builder as
        every ``[SANDBOX_INCIDENTAL]`` line): it names the scope, the target
        and the open network, and it is ``None`` for a batch on codex's own
        sandbox, which therefore prints nothing (§8.2 "The stray-writes-only
        line").
        """
        notice_for = getattr(_executor, "write_rung_notice", None)
        if notice_for is None or self.jail_request is None:
            return
        key = batch_index if batch_index is not None else runner
        if key in self._rung_noticed:
            return
        try:
            notice = notice_for(runner, self.jail_request)
        except Exception:  # noqa: BLE001 - a notice must never fail a dispatch
            return
        if notice:
            self._rung_noticed.add(key)
            self.log(str(notice))

    # -- ceilings -------------------------------------------------------

    def write_ceiling(self) -> int:
        """The effective write parallelism (D3h/D3i), and why it is what it is.

        D3i's three-way minimum with the write gate substituted for the read one:
        ``driver.max_concurrent_write_subagents`` (default 1) × the runner
        provider's own ``max_concurrent`` (claude-code and cursor-agent ship 2) ×
        the host's own ceiling — then **pinned to 1** for a shared-artifact phase
        regardless of all three, because that is the batchers' own invariant
        (apply-suggestions/apply-task-suggestions edit one file), enforced here
        rather than trusted from them.

        The provider leg is not optional: raising the driver key above a
        provider's cap would otherwise over-subscribe it on the parallelizable
        implement/code-fix batches, which is exactly the ceiling the user did not
        set and cannot see. Neither is the host leg, and it reaches here through
        ``__init__``'s lookup rather than through a caller: ``self.host_max`` is
        ``None`` only when the capability data could not be read at all, the same
        partial-install case ``_host_ceiling`` bounds nothing over.
        """
        configured = get_max_concurrent_write_subagents(anchor=self.anchor)
        candidates = [configured]
        if self.host_max is not None:
            candidates.append(int(self.host_max))
        limits = [
            limit for limit in self.provider_write_limits().values()
            if limit is not None
        ]
        if limits:
            # The *min* across providers, where the read path's
            # `effective_ceiling` takes the max: that path holds a per-provider
            # semaphore underneath the global gate, so a permissive group ceiling
            # is still bounded per provider. A write group has no second gate —
            # this number is the only thing standing between a mixed batch and an
            # over-subscribed provider — so it binds to the tightest cap.
            candidates.append(min(limits))
        ceiling = max(1, min(candidates))
        if self.kind in SHARED_ARTIFACT_WRITE_KINDS:
            return 1
        return ceiling

    def provider_write_limits(self) -> Dict[str, Optional[int]]:
        """``max_concurrent`` per provider this phase's runners resolve to (D3i).

        Resolution mirrors the dispatch path exactly (``--runner`` override, else
        each task's own chain), and it is computed identically for both invokers
        so the executor and the simulated native invoker admit the same groups. A
        spec naming a provider with no configured cap contributes ``None``, which
        is "no provider leg", never zero.
        """
        specs: List[str] = []
        if self.forced_runner:
            specs.append(self.forced_runner)
        else:
            for task in self.tasks:
                specs.append(resolve_task_runner(task, anchor=self.anchor))
        limits: Dict[str, Optional[int]] = {}
        for spec in dict.fromkeys(specs):
            provider_name, _ = parse_model_spec(spec, anchor=self.anchor)
            limits[provider_name] = get_provider_max_concurrent(
                provider_name, anchor=self.anchor
            )
        return limits

    # -- ordering -------------------------------------------------------

    def batches(self) -> List[Tuple[int, List[Dict[str, Any]]]]:
        """The phase's tasks grouped by ascending ``batch_index``."""
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for task in self.tasks:
            buckets.setdefault(int(task.get("batch_index", 0)), []).append(task)
        return [(index, buckets[index]) for index in sorted(buckets)]

    # -- late prior-changes binding (D3n) -------------------------------

    def dispatched_prompt(self, task: Dict[str, Any]) -> str:
        """The prompt that actually goes to the runner, bound *now*.

        :func:`bind_prior_changes` over this phase's log; see it for the
        contract. Kept as a method so the dispatch site reads as before.
        """
        return bind_prior_changes(
            task, output_dir=self.output_dir, phase=self.phase
        )

    def _with_result_contract(self, task: Dict[str, Any], prompt: str) -> str:
        """Append the write-batch result contract (:func:`with_result_contract`)."""
        return with_result_contract(self.kind, task, prompt)

    # -- the loop -------------------------------------------------------

    def run(self) -> SchedulerResult:
        """Schedule the phase. Returns the phase's :class:`SchedulerResult`.

        The rollback store is staging, not an artifact: every path it made
        reversible has its own copy under ``{phase}/pre_batch/{task_id}/files/``
        by the time a dispatch is classified, so it is dropped on every exit —
        success, stop or crash — rather than leaving up to
        ``SWEEP_RETAIN_MAX_BYTES`` of the repository behind in the plan dir.
        """
        jail_lines_before = self._jail_line_count()
        try:
            result = self._run()
            self._report_resume_tally(result)
            return result
        finally:
            self._report_rollback_store_utilization()
            self._discard_rollback_store()
            self._repeat_jail_lines(jail_lines_before)

    def _report_resume_tally(self, result: SchedulerResult) -> None:
        """One line on a run that kept earlier batches: what it kept and what
        it dispatched.

        Only this loop knows which batches an earlier run finished (their
        result file, ``_done_evidence``); the apply base's own ``[resume]``
        line is printed before any batch exists, and on this leg it used to
        say "0 already processed, 12 remaining" while the run then kept
        batch 0 as an earlier run's (OpenCode racing run 4).
        """
        kept = [o for o in result.outcomes if o.skipped]
        # Every apply run resumes unless `--fresh`, so a first run (nothing
        # kept) says nothing here; its batch lines already say what ran.
        if not self.resume or self.dry_run or not kept:
            return
        dispatched = [o for o in result.outcomes if not o.skipped]
        items = {
            str(task.get("task_id", "")): len(task.get("expected_ids") or [])
            for task in self.tasks
        }

        def counted(outcomes: Sequence[TaskOutcome]) -> str:
            # An implement task is one item; the count would only repeat it.
            if self.kind == "implement":
                return ""
            return f" ({sum(items.get(o.task_id, 0) for o in outcomes)} item(s))"

        names = ", ".join(o.task_id for o in kept[:5]) + (", …" if len(kept) > 5 else "")
        self.log(
            f"[write-subagents] resume: kept {len(kept)} task(s) an earlier run "
            f"finished — {names}{counted(kept)}; "
            f"dispatched {len(dispatched)}{counted(dispatched)}"
            + (f"; not reached {len(result.not_run)}" if result.not_run else "")
        )

    @staticmethod
    def _jail_line_count() -> int:
        try:
            from .subagent_executor import jail_summary_lines
        except Exception:  # noqa: BLE001 - an older executor has no summary
            return 0
        return len(jail_summary_lines())

    def _repeat_jail_lines(self, before: int) -> None:
        """Repeat this phase's ``[JAIL_DENIED]`` / ``[CODEX_HOME_CHECK]`` lines
        at its close (plan codex-sandbox-unavailable §6 F1 item 6, Limits,
        2026-09-30): write batches dispatch through ``dispatch_task`` directly,
        so ``run_tasks``' own closing summary never runs for them, and a change
        to ``$CODEX_HOME`` outlives the run."""
        try:
            from .subagent_executor import jail_summary_lines
        except Exception:  # noqa: BLE001 - an older executor has no summary
            return
        for line in jail_summary_lines()[before:]:
            self.log(f"[write-subagents] closing summary: {line}")

    def _report_rollback_store_utilization(self) -> None:
        """One line naming how much of the reversibility budget this run used.

        The sweep is deliberately generous (F15) — anything under the pinned
        root the run could have touched, not just the phase's declared set,
        because narrowing it would make its *detective* half (catching an
        undeclared write outside that set) vacuous — so on a large enough repo
        it can spend a real share of ``SWEEP_RETAIN_MAX_BYTES`` on files the
        phase never intended to touch. Until now the only visible signal was
        the warning in :meth:`_retain` for the files *already* denied a
        rollback point; this prints once the run is done retaining, so the
        trend toward that budget is visible before it happens rather than
        only once it already has. Skipped when nothing was ever censused
        (``--dry-run-subagents``, or a run that dispatched no batch).
        """
        if not self._retained:
            return
        digests = {digest for _stamp, digest in self._retained.values() if digest}
        shards = {digest[:2] for digest in digests}
        # The files the deny list kept out, named here once rather than in a
        # warning at the start of every write phase (see `_retain`).
        denied = sorted(
            path for path, is_denied in self._unretained_denied.items() if is_denied)
        kept_out = (
            f"; never copied, as driver.write_paths denies "
            f"{'it' if len(denied) == 1 else 'them'} (a write to "
            f"{'it' if len(denied) == 1 else 'one'} fails the batch and "
            f"cannot be undone): {', '.join(denied[:5])}"
            f"{f', … {len(denied) - 5} more' if len(denied) > 5 else ''}"
            if denied else ""
        )
        # The snapshot directory named in full: a jailed repo-wide batch keeps
        # its snapshots under the state home, not the phase dir (Codex racing
        # run 4: "restore points live in pre_batch/" read as the phase dir's
        # copy, which the line before says is never read).
        self.log(
            f"[write-subagents] rollback store peaked at "
            f"{self._retained_bytes / (1024 * 1024):.1f} MB of the "
            f"{SWEEP_RETAIN_MAX_BYTES // (1024 * 1024)} MB budget "
            f"({len(digests)} files, {len(shards)} shards{kept_out}); discarded now "
            f"— the restore points are the pre-batch snapshots in "
            f"{os.path.join(self.safety_dir, 'pre_batch')}/"
        )

    def _discard_rollback_store(self) -> None:
        """Drop the sweep's staging bytes and the index that describes them."""
        shutil.rmtree(
            os.path.join(self.safety_dir, ROLLBACK_STORE_DIRNAME), ignore_errors=True
        )
        with self._sweep_lock:
            self._retained.clear()
            self._retained_bytes = 0

    def _warn_stale_result_files(self) -> None:
        """Name — never delete — result files no current task owns.

        ``implement``'s rename from the positional ``implement_batch_N.json``
        to the task-id-keyed ``implement_batch_T006.json`` (F1) left the old
        files on disk: nothing reads them any more
        (``resolve_output_target`` derives the name from ``output_name``), but
        a human opening the phase dir sees both namings side by side with no
        way to tell which are live (N6, 2026-09-16). Scoped to ``implement``,
        the one write kind whose ``output_name`` is keyed by the task id
        rather than a batch position: a positional kind's filenames have
        never been renamed, and without an id to key on there is no way to
        tell "an earlier run's now-completed batch" apart from "orphaned by a
        renumbering" — flagging those would be noise, not a finding. An
        id-keyed orphan (``implement_batch_T011.json`` from a run whose plan
        had eleven tasks) is the other way a file ends up here, so the
        warning does not call every one of them "an earlier naming".
        """
        if self.kind not in _ID_KEYED_WRITE_KINDS:
            return
        pattern = OUTPUT_NAME_PATTERN_BY_KIND.get(self.kind)
        if pattern is None:
            return
        current_names = {
            str(task.get("output_name") or "") for task in self.tasks
        }
        # "any task id" (D3n's own resume contract): every id this phase has
        # EVER recorded a status for, completed or not — `task_status` is one
        # flat map never trimmed across runs sharing `state.json` — union the
        # ids this run still has materialized, for a state-less caller (e.g.
        # a dry run with no `state=`).
        known_ids = {str(task.get("task_id") or "") for task in self.tasks}
        if self.state is not None and hasattr(self.state, "get_all_task_statuses"):
            try:
                known_ids |= set(self.state.get_all_task_statuses())
            except Exception:  # noqa: BLE001 - advisory only, never fatal
                pass
        known_ids.discard("")
        try:
            candidates = sorted(os.listdir(self.phase_dir))
        except OSError:
            return
        stale = [
            name for name in candidates
            if pattern.match(name)
            and name not in current_names
            and not any(task_id in name for task_id in known_ids)
        ]
        if stale:
            self.log(
                f"[write-subagents] result files no current task owns (an "
                f"earlier naming, or a task the plan no longer has), ignored: "
                f"{', '.join(stale)}"
            )

    def _run(self) -> SchedulerResult:
        result = SchedulerResult(kind=self.kind, phase=self.phase)
        ordered = self.batches()
        if self.resume and not self.dry_run:
            result.earlier_rows = self._earlier_report_rows()

        # 0. Advisory only, never a mutation: name any result file left behind
        # by an earlier naming scheme before the batches that matter start.
        self._warn_stale_result_files()

        # 1. Human tasks: resolve once, apply as a graph walk, persist.
        result.human_plan = resolve_human_task_plan(
            self.tasks,
            strategy=self.human_task_strategy,
            decisions=self.human_task_decisions,
            check_findings_ids=(
                rerunnable_checks(self.state) if self.kind == "implement" else ()
            ),
            graph=(
                decomposition_graph(self.tasks, self.state)
                if self.kind == "implement" else None
            ),
        )
        if self.kind == "implement" and self.state is not None:
            try:
                recorded = dict(self.state.get_all_task_statuses())
            except Exception:  # noqa: BLE001 - a state fake without the map
                recorded = {}
            mark_no_longer_skipped(
                result.human_plan,
                [str(task.get("task_id", "")) for task in self.tasks],
                recorded,
            )
        if not self.dry_run:
            # A dry run resolves the plan (the report says what *would* be
            # skipped) but never persists it: D3l's contract is "no child
            # started, nothing else changed", and persisting would pin the
            # strategy, mark human tasks skipped and record dependency overrides
            # in `state.json` for a run that never dispatched anything.
            self._persist_human_plan(result.human_plan)
        if self.kind == "implement":
            # The answers an earlier run recorded for tasks this resume did
            # not materialize (an accepted failure, a manual task answered
            # done): in no bucket otherwise, and the account's total shrank.
            accepted, completed_manual = answered_done_on_record(
                self.state,
                exclude=[str(task.get("task_id", "")) for task in self.tasks],
            )
            for task_id, reason in accepted.items():
                result.human_plan.accepted.setdefault(task_id, reason)
            for task_id in completed_manual:
                if task_id not in result.human_plan.completed:
                    result.human_plan.completed.append(task_id)

        if result.human_plan.pending:
            # Fail closed: an unanswered human task is exactly the D6 "needs a
            # human" condition, and dispatching around it would run code tasks
            # against unmet preconditions.
            result.exit_code = EXIT_NEEDS_HUMAN
            result.messages.extend(self._pending_human_message(result.human_plan))
            result.report_path = self._write_report(result)
            self._mirror_status(result)
            return result

        # 2. Dry run: full preflight, zero dispatch.
        if self.dry_run:
            result.dry_run_path = self._write_dry_run(ordered, result.human_plan)
            result.messages.append(
                f"[dry-run-subagents] {result.dry_run_path} — every command "
                f"rendered, no child started, nothing changed."
            )
            # No report: `subagent_dry_run.json` IS the dry run's artifact, and
            # writing the zero-outcome batch reports here would clobber a real
            # run's `write_batch_report.json` / `subagent_batch_report.json`.
            return result

        # 3. The batches, in order, each verified before the next is dispatched.
        # Mirrored once before the first dispatch so `progress.step` reads
        # `applying` for the whole stretch rather than only from the first
        # *finished* task — the in-batch mirror runs after a dispatch returns,
        # and a long first batch is exactly when the step matters (F4).
        self._metrics_start(result)
        self._mirror_status(result)
        for batch_index, batch_tasks in ordered:
            runnable = [
                task for task in batch_tasks
                if not result.human_plan.is_skipped(str(task.get("task_id", "")))
                and str(task.get("task_id", "")) not in result.human_plan.accepted
                and str(task.get("task_id", "")) not in result.blocked_by_check
                and task.get("executor_role") != "human"
            ]
            if not runnable:
                continue
            try:
                outcomes = self._run_batch(batch_index, runnable, result)
            except ExecutorError as exc:
                # D3k rung 1: a pre-dispatch refusal (a missing binary, an
                # unauthenticated runner, a profile the adapter cannot serve —
                # `preflight_runner` raising `RunnerRejected`) fails the batch
                # *here*, exactly like every other stop. Left to propagate it
                # reached the dispatcher's generic handler as a traceback and exit
                # 1, so the run produced no `write_batch_report.json`, no
                # `.status.json` mirror and none of the exit-3 wording that names
                # the fix — the read path has converted the same exception to
                # `failure.exit_code` since D3b.
                result.stopped_at_batch = batch_index
                result.exit_code = int(exc.exit_code or EXIT_NEEDS_HUMAN)
                self._record_stop(result, batch_index, [], refusal=exc.message)
                result.messages.extend(
                    self._refusal_message(exc, batch_index, result)
                )
                if self.summary_file and result.outcomes:
                    # Only when an earlier batch really applied something: the
                    # partial summary is that run's account (D3n), and an empty
                    # one for a refusal on the first dispatch would claim a run
                    # that never started.
                    result.summary_file = self._write_summary(result)
                    result.messages.append(
                        f"Partial implementation summary: {result.summary_file}"
                    )
                result.report_path = self._write_report(result)
                self._mirror_status(result)
                self._record_refusal(exc, batch_index)
                return result
            result.outcomes.extend(outcomes)
            result.batches_dispatched.append(batch_index)
            # A check that reported problems keeps every task that depends on
            # it from running (fix round 11); the tasks that do not depend on
            # it go on.
            self._block_dependents_of_checks(outcomes, result)
            self._record_batch_tracking(result)
            self._mirror_status(result)
            if any(o.stops_the_loop for o in outcomes) or any(
                not o.ok for o in outcomes
            ):
                result.stopped_at_batch = batch_index
                result.exit_code = EXIT_NEEDS_HUMAN
                self._record_stop(
                    result, batch_index, [o for o in outcomes if not o.ok],
                )
                result.messages.extend(
                    self._stop_message(batch_index, outcomes, result)
                )
                # The partial summary is the cancellation's own record (D3n,
                # instructions/implement.md): `_finalize` never runs on this
                # path, so without this write the `## Cancelled` section would be
                # unreachable and `--resume` would have no account of what the
                # stopped run applied. The phase is still NOT completed and the
                # plan-side reference is still NOT inserted — only the summary is
                # written.
                if self.summary_file:
                    result.summary_file = self._write_summary(result)
                    result.messages.append(
                        f"Partial implementation summary: {result.summary_file}"
                    )
                result.report_path = self._write_report(result)
                self._mirror_status(result)
                return result

        # 4. The tasks a check with findings kept from running: the phase is
        # not complete while they are unrun, so it ends needing a human, the
        # way a stop does — after every task that could run has run, and
        # with no finalization (no verify command, no completion record).
        if result.blocked_by_check:
            result.exit_code = EXIT_NEEDS_HUMAN
            self._record_check_block(result)
            if self.summary_file:
                result.summary_file = self._write_summary(result)
            result.messages.extend(self._check_findings_lines(result))
            result.messages.extend(self._check_block_message(result))
            if result.summary_file:
                result.messages.append(
                    f"Partial implementation summary: {result.summary_file}"
                )
            result.report_path = self._write_report(result)
            self._mirror_status(result)
            return result

        # 5. Finalization is part of the contract, not the caller's courtesy.
        if self.finalize:
            self._finalize(result)
        result.messages.extend(self._check_findings_lines(result))
        result.report_path = self._write_report(result)
        self._mirror_status(result)
        return result

    def _check_findings_lines(self, result: SchedulerResult) -> List[str]:
        """One ``[CHECK_FINDINGS]`` line per check task that reported
        problems, and one line on what happens to them (racing re-run 4:
        nothing on the run's own output named T007's two problems)."""
        found = result.check_findings()
        if not found:
            return []
        lines = [
            f"{CHECK_FINDINGS_MARKER} {task_id} {len(findings)} problem(s): "
            f"{'; '.join(findings)}"
            for task_id, findings in found.items()
        ]
        where = (
            f"; they are listed in {result.summary_file}"
            if result.summary_file else ""
        )
        blocked = sorted(result.blocked_by_check)
        went_on = (
            f"the tasks that do not depend on them went on, and the ones that "
            f"do were not run ({', '.join(blocked)})"
            if blocked else "no task waits for them, so implement went on"
        )
        lines.append(
            f"Check findings: {', '.join(found)} ran, changed no file and "
            f"reported problems; {went_on}{where}. Review-code is told about "
            f"them, and apply-code-fixes applies the fixes review-code "
            f"validates."
        )
        return lines

    def _block_dependents_of_checks(
        self, outcomes: Sequence[TaskOutcome], result: SchedulerResult,
    ) -> None:
        """Keep every task that depends on a check with findings from running.

        Fix round 11 (review of round 10, T1): a check task that reported
        problems was recorded ``check-findings`` and the loop dispatched the
        tasks that depend on it — on the code the check had just rejected.
        Its transitive dependents (``depends_on``, over the whole
        decomposition — fix round 12: over this run's tasks alone, a resume
        lost the ones whose chain ran through a finished task) that this run
        holds are now recorded in ``result.blocked_by_check`` and left out of every
        later batch; a task already skipped or accepted, and a manual task
        (never dispatched), are not. A resume skip and a check not run again
        because nothing changed carry the earlier findings, so they block
        the same tasks again. ``implement`` only: no other kind has checks.
        """
        if self.kind != "implement":
            return
        plan = result.human_plan
        for outcome in outcomes:
            if not (outcome.ok and outcome.check_findings):
                continue
            for dependent in self._dependents_of(outcome.task_id):
                task = self._task_by_id(dependent)
                if task is None:
                    # Not in this run (a finished task a resume did not
                    # materialize): nothing to hold, but the walk went
                    # through it to the tasks after it (fix round 12).
                    continue
                if (
                    dependent in result.blocked_by_check
                    or plan.is_skipped(dependent)
                    or dependent in plan.accepted
                    or task.get("executor_role") == "human"
                    or task.get("is_human")
                ):
                    continue
                result.blocked_by_check[dependent] = outcome.task_id

    # -- one batch ------------------------------------------------------

    def _run_batch(
        self,
        batch_index: int,
        tasks: Sequence[Dict[str, Any]],
        result: SchedulerResult,
    ) -> List[TaskOutcome]:
        ceiling = self.write_ceiling()
        clashes = overlapping_pairs(tasks, root=self.repo_root)
        groups = admit_disjoint_groups(tasks, ceiling=ceiling, root=self.repo_root)
        if ceiling > 1 and clashes:
            for left, right, shared in clashes:
                self.log(
                    f"[write-subagents] batch {batch_index}: {left} and {right} "
                    f"both declare {', '.join(shared)} — refusing to co-schedule "
                    f"them; they run sequentially instead (D3h)."
                )
        outcomes: List[TaskOutcome] = []
        # `outcomes` is the task file's order (what the report and the stop
        # check read); this one grows in completion order, and is what the
        # mirror reports.
        outcomes_so_far: List[TaskOutcome] = []
        # The mirror below the loop in `_run` moves once per *batch*, so a batch
        # of co-scheduled tasks reported nothing until its last one landed. This
        # is the same counts write, per task: display only, never the report,
        # and taken under a lock because a disjoint group dispatches on threads.
        mirror_lock = threading.Lock()

        def dispatch_and_mirror(task: Dict[str, Any]) -> TaskOutcome:
            outcome = self._dispatch_one(task, batch_index, result)
            if outcome.skipped:
                # A resume-skipped task is still one of the run's finished tasks
                # — the mirror counts it in `completed` — and its branch returns
                # before `_record_batch_effects`, so the `[ETA]` numerator is
                # advanced here instead, or a resumed run's last line stops short
                # of its own total by exactly the batches it skipped (F3).
                with self._effects_lock:
                    self._tasks_finished += 1
            with mirror_lock:
                outcomes_so_far.append(outcome)
                self._mirror_status(result, finished_in_batch=outcomes_so_far)
            return outcome

        for group in groups:
            if len(group) == 1:
                outcomes.append(dispatch_and_mirror(group[0]))
            else:
                with ThreadPoolExecutor(max_workers=len(group)) as pool:
                    futures = [
                        pool.submit(dispatch_and_mirror, task)
                        for task in group
                    ]
                    outcomes.extend(future.result() for future in futures)
            if any(o.stops_the_loop or not o.ok for o in outcomes):
                # Verification gates the next dispatch (D3h): a partial
                # application is never compounded by a concurrent one.
                break
        return outcomes

    def _dispatch_one(
        self, task: Dict[str, Any], batch_index: int, result: SchedulerResult
    ) -> TaskOutcome:
        task_id = str(task.get("task_id", ""))
        profile = profile_for_kind(self.kind)

        # Resume: a batch counts as done when its result file exists, parses and
        # validates (D3i). Its summary is already in the prior-changes log on
        # disk, so a resumed run replays rather than reconstructs.
        evidence = self._done_evidence(task) if self.resume else None
        kept_findings = False
        if evidence is not None and self.kind == "implement":
            rerun = self._check_findings_rerun(task, evidence[0], result)
            if rerun is True:
                evidence = None
            elif rerun is False:
                kept_findings = True
        if evidence is not None:
            output_path, recorded_id = evidence
            self.log(f"[write-subagents] {task_id} already complete — skipping")
            data = json.loads(Path(output_path).read_text(encoding="utf-8"))
            verdict = data.get("verdict")
            if not isinstance(verdict, dict):
                verdict = {}
            recorded = self.audit.completion_for(
                recorded_id,
                expected_ids=[str(i) for i in (task.get("expected_ids") or [])],
            ) or {}
            outcome = TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_OK, profile=profile, skipped=True,
                write_outcome=(verdict.get("write_outcome")
                               or recorded.get("write_outcome")
                               or (WRITE_NO_OP if data.get("status") == "no-op"
                                   else WRITE_APPLIED)),
                summary=data.get("summary"),
                reason=verdict.get("reason") or recorded.get("reason"),
                verification_only=bool(verdict.get("verification_only")),
            )
            if outcome.verification_only and isinstance(verdict.get("check"), dict):
                # The skip keeps the earlier run's `check-findings` label.
                outcome.check_findings = row_check_findings(verdict)
            if kept_findings:
                outcome.reason = (
                    f"NOT RE-RUN: nothing in the repository changed since "
                    f"{task_id}'s check reported problems (the files git sees "
                    f"have the same sizes and timestamps), so it would report "
                    f"them again; its findings stand. Fix what it reported "
                    f"and resume, accept the findings with --human-task "
                    f"{task_id}=done, or run it again anyway with "
                    f"--human-task {task_id}=retry."
                )
            outcome.output_path = output_path
            # The files the earlier run verified for this batch, so a resumed
            # run's finalization (`--touched`) and summary still name them.
            if "files_touched" in verdict:
                outcome.files_touched = list(verdict["files_touched"])
            elif "files_touched" in recorded:
                outcome.files_touched = list(recorded["files_touched"])
            else:
                outcome.files_touched = self._recorded_files_touched(output_path)
            outcome.expected_ids = list(task.get("expected_ids") or [])
            outcome.returned_ids = list(data.get("ids") or [])
            observed = recorded.get("observed") or {}
            outcome.sha256_before = verdict.get("sha256_before") or {
                path: row.get("sha256_before") for path, row in observed.items()
            }
            outcome.sha256_after = verdict.get("sha256_after") or {
                path: row.get("sha256_after") for path, row in observed.items()
            }
            # One line per skip, before the early return: this branch reaches
            # neither `_audit_terminal` nor `_record_batch_effects`, so without it
            # a resumed run's audit log is silent about every task it declared
            # complete — the one place the 2026-09-16 five-task skip would have
            # been visible as it happened (D3l).
            self._audit_resume_skip(outcome, task)
            # The status the earlier run left for this task is not necessarily
            # `completed`: the result file lands before `_record_batch_effects`
            # saves, so a run killed in that window leaves a file resume trusts
            # beside a `pending` nothing ever revisited — a finished phase
            # reporting `(13/14 tasks)` with one task still `pending`
            # (state-tracking F5, 2026-09-16). No metric: the duration was the
            # earlier run's to measure, and a fabricated one would average in.
            if self.state is not None:
                with self._effects_lock:
                    self.state.update_task_status(
                        task_id, "completed",
                        reason=(
                            "not run again: nothing changed since its check "
                            "reported problems" if kept_findings else
                            f"resumed from {Path(outcome.output_path).name}"
                        ),
                    )
                    self.state.save()
            return outcome

        # A resumed verification-only task that failed, with nothing in the
        # repository changed since: not dispatched again (racing re-run 3,
        # 2026-10-05 — `--resume` spent 148 s on the same `html-validate`
        # failure against the same bytes).
        if self.resume:
            refused = self._unchanged_since_failure(task, result)
            if refused is not None:
                self._record_batch_effects(task, refused, batch_index, result)
                return refused

        try:
            prompt = self._with_result_contract(task, self.dispatched_prompt(task))
        except ResultRejected as exc:
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=exc.status, profile=profile, reason=exc.message,
            )
        self._retire_foreign_result(task)

        # D3l pre-dispatch: a batch whose DECLARED set is not wholly inside the
        # policy never starts a runner — and the check runs BEFORE the snapshot,
        # not after it. Snapshotting first would copy the bytes of every denied
        # path the task named (`.env`, `.git/config`, `*.pem`) into
        # `{phase}/pre_batch/{task_id}/files/` and then refuse the dispatch,
        # leaving the payload behind: the deny list exists precisely to keep that
        # material out of executor-owned artifacts, so a refusal that creates one
        # is the control defeating itself.
        declared = [
            relative_to_root(path, self.repo_root)
            for path in declared_file_set(task)
        ]
        try:
            self.policy.check(declared, when="pre-dispatch")
        except PolicyViolation as exc:
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_REJECTED, profile=profile, reason=exc.message,
                write_outcome=WRITE_POLICY_VIOLATION,
            )
        # Plan step 3 (2026-09-30), beside the D3l check and before the
        # snapshot for the same reason: under `plan-file`/`tasks` the declared
        # set must be exactly the phase's write target, on every rung, or the
        # batch is refused with nothing snapshotted and nothing spawned.
        mismatch = self._grant_mismatch(task)
        if mismatch is not None:
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_REJECTED, profile=profile, reason=mismatch,
            )

        try:
            self._decide_rollback_storage(task, profile)
            snapshot = self._snapshot(task)
        except WriteSafetyError as exc:
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_REJECTED, profile=profile, reason=exc.message,
            )
        # The replaced-target check's "before": taken at THIS dispatch's
        # pre-batch snapshot (reused or fresh), so the identity compared after
        # the child is always the one this spawn saw (§8.3 item 14).
        self._target_identity_before[task_id] = self._identity_of(
            self._identity_path()
        )

        try:
            self._begin_sweep(
                task_id, declared,
                verification_only=self._verification_only(snapshot),
            )
        except WriteSafetyError as exc:
            # The census is this task's whole verdict and there is none: a
            # refusal, with nothing spawned (W1, adversarial review 2026-10-02).
            self._target_identity_before.pop(task_id, None)
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_REJECTED, profile=profile, reason=exc.message,
            )
        started = time.time()
        self._git_meta_before[task_id] = self._git_metadata_fingerprint()
        try:
            if self.respond is not None:
                outcome = self._dispatch_native(task, prompt, snapshot, profile)
            else:
                outcome = self._dispatch_executor(
                    task, prompt, snapshot, profile, declared,
                    batch_index=batch_index,
                )
            self._warn_git_metadata_changed(task_id)
            outcome.snapshot_dir = snapshot.directory
            if not outcome.duration_seconds:
                outcome.duration_seconds = time.time() - started
            if outcome.write_outcome is None and not outcome.ok:
                # The four-outcome axis is total, so a dispatch that never reached
                # the classification still says which bucket it fell in. Rungs 5
                # and 6 are both "the runner did not return the batch result
                # contract" — prose and a well-formed wrong object are the same
                # failure to a consumer — so they share the schema-rejection
                # outcome. Rungs 1-4 (no child, no auth, timeout, runner error) are
                # transport failures, not write outcomes, and are deliberately left
                # unlabelled. Either becomes a partial application in
                # `_audit_terminal` when the batch moved bytes (fix round 9).
                if outcome.status in (STATUS_SCHEMA_INVALID, STATUS_UNPARSEABLE):
                    outcome.write_outcome = WRITE_SCHEMA_REJECTED
                elif outcome.status == STATUS_PARTIAL:
                    # T4.8: `dispatch_task`'s pre-retry check found bytes moved
                    # under a turn a denial ended — the same outcome, and the
                    # same stop, as `_classify`'s partial-application branch.
                    outcome.write_outcome = WRITE_PARTIAL
                    outcome.reason = (
                        f"{outcome.reason} The pre-batch snapshot is kept at "
                        f"{snapshot.directory}."
                    )

            self._audit_terminal(outcome, snapshot, profile)
        finally:
            # The census and its retained-byte index are consumed by whichever of
            # `_classify` / `_audit_terminal` got there first; this drops them in
            # the one case neither did (a dispatch that raised out).
            #
            # I11 (plan codex-sandbox-unavailable-findings-and-fixes §5,
            # 2026-09-30): that case is also a batch refused before any child
            # started (`preflight_runner` raising `RunnerRejected` at batch 0),
            # and `_begin_sweep` had already published it in flight. Left
            # registered, the refusal's own `_mirror_status` re-published it,
            # so `.status.json` read `in flight: apply-batch-0 (since 0m 9s)`
            # for the rest of the dir's life. The dispatch is over either way.
            with self._sweep_lock:
                self._census_before.pop(task_id, None)
                self._retained_before.pop(task_id, None)
                self._dispatch_started.pop(task_id, None)
                self._declared_by_task.pop(task_id, None)
                self._sweep_ignored.pop(task_id, None)
                self._ignore_sources_before.pop(task_id, None)
            self._target_identity_before.pop(task_id, None)
        self._record_batch_effects(task, outcome, batch_index, result)
        return outcome

    def _check_findings_rerun(
        self, task: Dict[str, Any], output_path: str, result: SchedulerResult,
    ) -> Optional[bool]:
        """Whether a resumed check whose findings stand is run again.

        ``None`` when the result *output_path* holds is not a check that
        reported problems (the ordinary resume skip applies). Otherwise the
        findings are treated as a failed stamp is (fix round 11, review T2:
        a check with findings was skipped as done on every ``--resume``, so
        the tasks it held could never run): ``True`` — dispatch it again —
        on ``--human-task <id>=retry``, or when the files git sees changed
        since it ran (``verdict.tree_fingerprint``), or when no fingerprint
        can be compared; ``False`` — keep its findings without a dispatch,
        so the tasks after it stay held — when nothing changed.
        """
        try:
            data = json.loads(Path(output_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        verdict = data.get("verdict") if isinstance(data, dict) else None
        if not isinstance(verdict, dict) or not verdict.get("verification_only"):
            return None
        if not row_check_findings(verdict):
            return None
        task_id = str(task.get("task_id", ""))
        if task_id in result.human_plan.retry:
            self.log(
                f"[write-subagents] {task_id}: --human-task {task_id}=retry — "
                f"running its check again whether or not the repository changed."
            )
            return True
        recorded = verdict.get("tree_fingerprint")
        current = self._tree_fingerprint()
        if recorded and current is not None and current == recorded:
            self.log(
                f"[write-subagents] {task_id}: not run again — nothing in the "
                f"repository changed since its check reported problems; its "
                f"findings stand and the tasks after it stay held."
            )
            return False
        self.log(
            f"[write-subagents] {task_id}: its check reported problems and "
            f"the files changed since — running it again."
        )
        return True

    def _unchanged_since_failure(
        self, task: Dict[str, Any], result: SchedulerResult,
    ) -> Optional[TaskOutcome]:
        """The guard a resumed, failed, verification-only task passes first.

        Answers an outcome — not dispatched, ``batch_failed`` / clean failure,
        ``not_rerun`` — when all of: the task declares no files (a check), the
        state records it ``failed``, its result file's verdict carries the
        ``tree_fingerprint`` the failure was stamped with, that fingerprint
        still matches the worktree (``git_utils.worktree_fingerprint``: the
        files git sees, by size and mtime, the plugin's own output trees left
        out), and the user did not answer ``--human-task <id>=retry``. A
        read-only check on unchanged bytes fails the same way; the stop
        message then names the remedy again instead of a dispatch buying
        nothing. ``None`` (dispatch as usual) on any other answer: a changed
        tree, no fingerprint on record, a tree git cannot read.
        """
        task_id = str(task.get("task_id", ""))
        if self.kind != "implement" or declared_file_set(task):
            return None
        if task_id in result.human_plan.retry:
            self.log(
                f"[write-subagents] {task_id}: --human-task {task_id}=retry — "
                f"running the check again whether or not the repository changed."
            )
            return None
        if self.state is not None:
            try:
                if self.state.get_task_status(task_id) != "failed":
                    return None
            except Exception:  # noqa: BLE001 - advisory
                return None
        try:
            own = resolve_output_target(task, self.phase_dir)
        except ResultRejected:
            return None
        if own is None or not Path(own).is_file():
            return None
        try:
            data = json.loads(Path(own).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        verdict = data.get("verdict") if isinstance(data, dict) else None
        if not isinstance(verdict, dict) or verdict.get("ok"):
            return None
        recorded = verdict.get("tree_fingerprint")
        if not recorded or not verdict.get("verification_only"):
            return None
        current = self._tree_fingerprint()
        if current is None or current != recorded:
            return None
        summary = str(data.get("summary") or "").strip()
        outcome = TaskOutcome(
            task_id=task_id, kind=self.kind, runner=self.invoker,
            status=STATUS_BATCH_FAILED, profile=profile_for_kind(self.kind),
            write_outcome=WRITE_CLEAN_FAILURE, verification_only=True,
            not_rerun=True, summary=summary or None,
            runner_notes=(verdict.get("runner_notes") or None),
        )
        outcome.output_path = str(own)
        outcome.expected_ids = [str(i) for i in (task.get("expected_ids") or [])]
        outcome.returned_ids = [str(i) for i in (data.get("ids") or [])]
        outcome.reason = (
            f"NOT RE-RUN: nothing in the repository changed since {task_id} "
            f"failed (the files git sees have the same sizes and timestamps), "
            f"so the same check would fail the same way"
            + (f": {summary}" if summary else "")
            + ". Fix what it reported and resume, accept the failure with "
            f"--human-task {task_id}=done, or run it again anyway with "
            f"--human-task {task_id}=retry."
        )
        self.log(
            f"[write-subagents] {task_id}: not run again — nothing in the "
            f"repository changed since it failed; the stop below names the "
            f"way forward."
        )
        return outcome

    def _tree_fingerprint(self) -> Optional[str]:
        """``git_utils.worktree_fingerprint`` over the pinned root, the
        plugin's own trees (the phase dir, the plan's output dir and
        ``.multi-llm/``) left out."""
        excluded = []
        for path in (self.phase_dir, self.output_dir,
                     os.path.join(self.repo_root, ".multi-llm")):
            try:
                relative = os.path.relpath(os.path.abspath(path), self.repo_root)
            except ValueError:  # pragma: no cover - another drive on Windows
                continue
            if relative and not relative.startswith(".."):
                excluded.append(relative)
        try:
            return _git_utils.worktree_fingerprint(
                self.repo_root, exclude_prefixes=excluded,
            )
        except Exception:  # noqa: BLE001 - advisory, never fatal
            return None

    def _audit_resume_skip(
        self, outcome: TaskOutcome, task: Dict[str, Any]
    ) -> None:
        """Record the one decision a resume skip makes, naming what it trusted."""
        try:
            data = json.loads(
                Path(outcome.output_path).read_text(encoding="utf-8")
            )
            raw = data.get("ids") if isinstance(data, dict) else None
            result_ids = [str(item) for item in raw] if isinstance(raw, list) else []
        except (OSError, ValueError):  # noqa: BLE001 - the skip already happened
            result_ids = []
        self.audit.skipped(
            task_id=outcome.task_id, kind=self.kind,
            output_path=outcome.output_path, result_ids=result_ids,
            files_touched=outcome.files_touched,
        )

    def _audit_terminal(
        self, outcome: TaskOutcome, snapshot: TaskSnapshot, profile: str
    ) -> None:
        """Close the audit record for a dispatch that never reached ``_classify``.

        D3l's contract is *one fsynced line before dispatch and one after
        classification*, but the classifier only runs once rung-6 validation has
        passed. A dispatch that timed out, errored, was rate-limited out or came
        back as prose (rungs 2-5) therefore left a ``dispatched`` line with no
        terminal record — and ``AuditLog.failed_task_ids`` reads only
        ``completed`` events, so ``util restore-batch --all-failed`` silently
        skipped exactly the dispatches most likely to have left half-written
        bytes behind (a timeout mid-edit).

        The record carries the observed post-run hashes, and ``files_touched`` is
        filled from the byte comparison rather than left empty, because that set
        is what the restore path restores and verifies against. The sweep runs
        first on purpose: a timeout mid-edit is exactly the case where the runner
        was writing somewhere it never declared, and adopting those paths into the
        snapshot before ``changed()`` is read is what puts them in reach of
        ``util restore-batch --all-failed``.

        On the plugin-landlock rung the replaced-target check runs here as
        well, once ``files_touched`` is filled
        (:meth:`_replaced_target_unclassified`): a target a user or an editor
        saved over by rename differs from the snapshot, so without it the
        user's new file would land in the restore set.
        """
        if outcome.task_id in self._audited:
            return
        if self.respond is None and not outcome.attempts:
            # `dispatch_task` refused before its first `before_attempt`, so no
            # `dispatched` line was written either. A lone completion here would
            # claim a run that never started.
            return
        undeclared = self._undeclared_writes(outcome.task_id, snapshot)
        observed = self._observed_with_undeclared(snapshot, undeclared)
        if not outcome.files_touched:
            outcome.files_touched = snapshot.changed()
        # After the fill, never before it: the fill is what would put the
        # replaced target back into the restore set.
        if not self._replaced_target_unclassified(outcome, snapshot):
            self._partial_when_bytes_moved(outcome, snapshot)
        last = outcome.attempts[-1] if outcome.attempts else None
        self._audit_completion(
            outcome,
            outcome.runner,
            profile,
            last.index if last is not None else 1,
            last.exit_code if last is not None else None,
            observed,
            outcome.duration_seconds,
        )

    def _partial_when_bytes_moved(
        self, outcome: TaskOutcome, snapshot: TaskSnapshot
    ) -> None:
        """A failed dispatch that never reached ``_classify`` and moved bytes
        is a partial application.

        ``dispatch_task``'s pre-retry check (T4.8, widened in fix round 8)
        makes one of an attempt that would have re-sent the batch, but not of
        the report re-ask's own failure (the re-ask is exempt from that check:
        its retry re-sends the re-ask) nor of a reply rung 6 never retries. So
        a re-ask that replied in prose again, or timed out on every try,
        ended ``unparseable_result`` / ``schema-rejection`` or an unlabelled
        ``timeout`` beside the files the batch had changed. The row read "schema rejected" for work whose
        bytes moved, and the stop's wording was not the partial's (fix round
        9, review of round 8). *outcome*'s ``files_touched`` is the byte
        comparison :meth:`_audit_terminal` just filled; when it names a file,
        the outcome becomes the same partial application ``_classify``
        records, with what failed kept in the reason — the stop, the restore
        command and ``--resume`` then treat it as any partial. A rung-7
        refusal keeps its own verdict.
        """
        if outcome.ok or outcome.status in (STATUS_PARTIAL, STATUS_REJECTED):
            return
        if outcome.write_outcome not in (None, WRITE_SCHEMA_REJECTED):
            return
        moved = [str(path) for path in (outcome.files_touched or [])]
        if not moved:
            return
        what = outcome.reason or outcome.status
        outcome.status = STATUS_PARTIAL
        outcome.write_outcome = WRITE_PARTIAL
        outcome.reason = (
            f"PARTIAL APPLICATION: {len(moved)} file(s) changed "
            f"({', '.join(moved[:5])}{', …' if len(moved) > 5 else ''}) but "
            f"the batch did not finish: {what}. The batch loop stops here, the "
            f"pre-batch snapshot is kept at {snapshot.directory}, and the phase "
            f"is NOT marked completed — an automatic revert would destroy the "
            f"half-applied state a human needs to inspect."
        )

    # -- D3l post-run containment: the OBSERVED set, not the declared one ---

    def _begin_sweep(
        self, task_id: str, declared: Sequence[str], *,
        verification_only: bool = False,
    ) -> None:
        """Take this dispatch's pre-run census, retain its bytes, register its set.

        The census is what makes the post-run policy check non-vacuous; the
        retention pass is what makes its verdict *actionable*. Both happen before
        the runner starts, because neither can be reconstructed afterwards.

        Raises :class:`WriteSafetyError` for a ``verification_only`` task when
        the census is unusable. Every other task still has its declared set to
        compare; this one has nothing but the sweep, and "the sweep could not
        look" read as "nothing moved" (adversarial review, 2026-10-02: with
        the tree over ``SWEEP_MAX_FILES`` — an untracked ``node_modules/`` is
        enough — a file-less task edited ``index.html`` and was VERIFIED at
        exit 0). Never VERIFIED without a census.
        """
        dispatched_at = datetime.now().isoformat()
        with self._sweep_lock:
            census = self._census()
            if census is None and verification_only:
                raise WriteSafetyError(
                    f"Refusing to dispatch write task {task_id}: it declares no "
                    f"files, so the post-run containment sweep would be its "
                    f"whole verdict, and the tree under {self.repo_root} holds "
                    f"more than {SWEEP_MAX_FILES} files, which turns the sweep "
                    f"off (D3l). A verification-only task cannot be verified "
                    f"here. List the files it may change under `Files to "
                    f"modify` in the tasks file and re-run with --resume, or "
                    f"take the untracked trees (node_modules/, build output) "
                    f"out of the checkout. Nothing was dispatched."
                )
            self._census_before[task_id] = census
            self._retained_before[task_id] = self._retain(census)
            self._ignore_sources_before[task_id] = self._ignore_rule_sources(census)
            self._declared_by_task[task_id] = set(declared)
            for ignored in self._sweep_ignored.values():
                ignored.update(declared)
            self._sweep_ignored[task_id] = set().union(
                *self._declared_by_task.values()
            )
            self._dispatch_started[task_id] = dispatched_at
            if census is None and not self._sweep_warned:
                self._sweep_warned = True
                self.log(
                    f"[write-subagents] the tree under {self.repo_root} holds more "
                    f"than {SWEEP_MAX_FILES} files — the post-run containment sweep "
                    f"is skipped for this run and driver.write_paths is enforced "
                    f"against the declared set only (D3l)."
                )
        # F10: the mirror's own writes only happen at a batch's boundaries
        # (`_mirror_status`), which says nothing while a single-task batch is
        # still running — measured 2026-09-16, T001 ran 659s as one batch and
        # `.status.json` read `dispatched: 0` the entire time. This is the one
        # place a dispatch's start time is known, so it patches the mirror's
        # `in_flight` list directly rather than waiting for the next full
        # recompute; that next call (whenever it lands) replaces the whole
        # `write_subagents` object, `in_flight` included, from
        # `self._dispatch_started` afresh, so a stale patch here never lingers.
        self._publish_in_flight()

    def _publish_in_flight(self) -> None:
        """Patch ``progress.write_subagents.in_flight`` alone. Best-effort.

        Reads back whatever :meth:`_mirror_status` last wrote and patches only
        the one key this owns, so a dispatch starting mid-batch can never race
        that fuller write for the other fields — at worst this reads a
        `write_subagents` object one write stale, and the next real
        :meth:`_mirror_status` call (or the next dispatch's own call here)
        corrects it.
        """
        try:
            from . import status_file as _status_file
        except ImportError:  # pragma: no cover
            from utils import status_file as _status_file
        try:
            existing = _status_file.read_status(self.phase_dir)
            mirror = dict(
                ((existing.get("progress") or {}).get("write_subagents")) or {}
            )
            with self._sweep_lock:
                in_flight = sorted(self._dispatch_started.items())
            mirror["in_flight"] = [
                {"task_id": task_id, "dispatched_at": started}
                for task_id, started in in_flight
            ]
            _status_file.update_status(self.phase_dir, {
                "progress": {"write_subagents": mirror}
            })
        except Exception:  # noqa: BLE001 - observability must not fail a phase
            pass

    def _sweep_exclusions(self) -> Tuple[str, ...]:
        """Absolute paths the sweep must never attribute to the runner.

        The phase dir is the *executor's* own artifact tree (results, snapshots,
        the audit log, ``.status.json``), and the two output-dir bookkeeping
        files are written by :meth:`_record_batch_effects` — for a co-scheduled
        sibling, while this task is being classified. The detached-run registry
        is the same story one level up: ``status --sweep`` or a second run's
        claim can rewrite it at any moment. Every one of them would otherwise
        read as an undeclared write, and ``.multi-llm/**`` is denied by default,
        so the sweep would fail batches on the tool's own bookkeeping. Only the
        ``runs/`` subtree is excluded, never all of ``.multi-llm/`` — a runner
        rewriting ``.multi-llm/providers.yaml`` is exactly what the deny rule is
        for and stays observable.
        """
        excluded = [
            os.path.abspath(self.phase_dir),
            os.path.abspath(os.path.join(self.output_dir, _prior_changes.FILENAME)),
            os.path.abspath(os.path.join(self.repo_root, ".multi-llm", "runs")),
        ]
        state_file = getattr(self.state, "state_file", None)
        if state_file:
            excluded.append(os.path.abspath(str(state_file)))
        if self.summary_file:
            excluded.append(os.path.abspath(self.summary_file))
        return tuple(excluded)

    def _census(self) -> Optional[Dict[str, Tuple[int, int]]]:
        """``repo-relative -> (size, mtime_ns)`` for every file under the root.

        ``None`` means *not usable* — the tree is over :data:`SWEEP_MAX_FILES` —
        and a ``None`` on either side of the comparison makes the sweep a no-op
        rather than a guess. ``os.path.relpath`` rather than ``relative_to_root``
        on purpose: the walk starts at the pinned root, so the relative path is
        exact by construction and a per-file ``Path.resolve()`` would put a
        syscall on every entry of a tree this size.
        """
        excluded = self._sweep_exclusions()
        census: Dict[str, Tuple[int, int]] = {}
        stack = [self.repo_root]
        while stack:
            try:
                entries = list(os.scandir(stack.pop()))
            except OSError:
                continue
            for entry in entries:
                if _under_any(entry.path, excluded):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in SWEEP_PRUNED_DIR_NAMES:
                            stack.append(entry.path)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                relative = os.path.relpath(entry.path, self.repo_root)
                census[relative.replace(os.sep, "/")] = (
                    info.st_size, info.st_mtime_ns
                )
                if len(census) > SWEEP_MAX_FILES:
                    return None
        return census

    def _blob_path(self, digest: str) -> str:
        """Where the rollback store keeps ``digest``'s bytes."""
        return os.path.join(
            self.safety_dir, ROLLBACK_STORE_DIRNAME, digest[:2], digest
        )

    def _retain(self, census: Optional[Dict[str, Tuple[int, int]]]) -> Dict[str, str]:
        """Copy every censused file's current bytes into the rollback store.

        ``relative -> sha256`` for what was retained. Three exclusions, each for
        a reason the alternative makes worse:

        * **A path ``driver.write_paths`` denies** (``.env``, ``**/*.pem``,
          ``**/.ssh/**``) is never copied. The deny list exists precisely to keep
          that material out of executor-owned artifacts — it is why the
          pre-dispatch policy check runs *before* ``_snapshot`` — and a rollback
          store that quietly mirrors every secret in the repo into the phase dir
          would be a new vulnerability rather than a control. Such a write is
          still observed, still fails the post-run check and still stops the loop;
          it just gets no automatic undo.
        * **A file over ``MAX_SNAPSHOT_BYTES``**, the same per-file cap a declared
          snapshot refuses to dispatch over.
        * **Anything past ``SWEEP_RETAIN_MAX_BYTES``**, so insurance cannot cost
          more than the run.

        The ``(size, mtime_ns)`` cache is what keeps this from being a full-tree
        read per dispatch: only a file whose stamp moved since the last dispatch
        is hashed again, and the store is content-addressed, so re-storing a
        reverted edit is a no-op.
        """
        if census is None:
            return {}
        kept: Dict[str, str] = {}
        skipped: List[str] = []
        for relative, stamp in sorted(census.items()):
            cached = self._retained.get(relative)
            if cached is None or cached[0] != stamp:
                cached = (stamp, self._store_blob(relative, stamp))
                self._retained[relative] = cached
            if cached[1]:
                kept[relative] = cached[1]
            else:
                skipped.append(relative)
        # A path the deny list keeps out is out by design, and a write to it
        # fails the batch (the post-run policy check) whose reason says it
        # could not be restored. Announced up front it was the same line on
        # every write phase of a run (Codex racing run 4: `.multi-llm/
        # providers.yaml` four times), so it is named once, on the closing
        # utilisation line, and the warning keeps the gaps that are not by
        # design: a file over the per-file cap or past the budget.
        for path in skipped:
            if path not in self._unretained_denied:
                self._unretained_denied[path] = self.policy.violation(
                    os.path.join(self.repo_root, path)) is not None
        skipped = [path for path in skipped if not self._unretained_denied[path]]
        if skipped and not self._retain_warned:
            self._retain_warned = True
            # Named, with the reason each one was skipped: a count alone
            # ("3 file(s) … have no rollback point") left the reader guessing
            # which files and whether that was a problem (measured 2026-09-16:
            # it was the three under `.multi-llm/`, which the deny list keeps
            # out of executor artifacts by design).
            named = ", ".join(
                f"{path} ({self._skip_reason(path)})" for path in skipped[:5]
            ) + (f", … {len(skipped) - 5} more" if len(skipped) > 5 else "")
            self.log(
                f"[write-subagents] {len(skipped)} file(s) under {self.repo_root} "
                f"have no rollback point in "
                f"{os.path.join(self.safety_dir, ROLLBACK_STORE_DIRNAME)}: {named}. "
                f"An undeclared write to one of them is still detected, "
                f"policy-checked and audited — `util restore-batch` just cannot "
                f"undo it (D3l)."
            )
        return kept

    def _skip_reason(self, relative: str) -> str:
        """Why :meth:`_store_blob` kept no bytes for ``relative``."""
        absolute = os.path.join(self.repo_root, relative)
        if self.policy.violation(absolute) is not None:
            return "denied by driver.write_paths"
        try:
            size = os.stat(absolute).st_size
        except OSError:
            return "unreadable"
        if size > MAX_SNAPSHOT_BYTES:
            return f"over the {MAX_SNAPSHOT_BYTES}-byte per-file cap"
        if self._retained_bytes + size > SWEEP_RETAIN_MAX_BYTES:
            return f"past the {SWEEP_RETAIN_MAX_BYTES}-byte store budget"
        return "unreadable"

    def _store_blob(
        self, relative: str, stamp: Tuple[int, int]
    ) -> Optional[str]:
        """Put one file's bytes in the store, or ``None`` when it is not retained."""
        size = stamp[0]
        if size > MAX_SNAPSHOT_BYTES:
            return None
        absolute = os.path.join(self.repo_root, relative)
        if self.policy.violation(absolute) is not None:
            return None
        digest = file_sha256(absolute)
        if digest is None:
            return None
        blob = self._blob_path(digest)
        if os.path.isfile(blob):
            # F12: the blob may be this run's own earlier copy (a second
            # `relative` with identical content) or one a killed run's store
            # left behind — either way its bytes are already on disk, and the
            # budget/utilisation line are both about bytes on disk, not bytes
            # THIS call copied. Returning here without counting is the bug: a
            # run that inherited a 47 MB store from a killed predecessor
            # reported "0.1 MB of 64 MB retained" and the cap went unenforced
            # for the rest of the run. Content-addressed, so it is counted
            # once per digest no matter how many relative paths reach it.
            if digest not in self._counted_digests:
                self._counted_digests.add(digest)
                self._retained_bytes += size
            return digest
        if self._retained_bytes + size > SWEEP_RETAIN_MAX_BYTES:
            return None
        try:
            os.makedirs(os.path.dirname(blob), exist_ok=True)
            # Same temp-then-replace shape every other executor write uses, so a
            # crash mid-copy can never leave a truncated blob that a later
            # restore would write over a good file.
            pending = f"{blob}.{os.getpid()}.{threading.get_ident()}.tmp"
            shutil.copy2(absolute, pending)
            os.replace(pending, blob)
        except OSError:
            return None
        self._counted_digests.add(digest)
        self._retained_bytes += size
        return digest

    def _undeclared_writes(
        self, task_id: str, snapshot: TaskSnapshot
    ) -> UndeclaredWrites:
        """Paths this dispatch moved *outside* any declared set, made reversible.

        This is what makes the post-run policy check non-vacuous.
        ``snapshot.changed()`` can only ever report files the task declared —
        ``write_phase_bridge`` leaves ``snapshot_files`` equal to
        ``declared_files`` — and that exact list already passed the identical
        ``policy.check(..., when="pre-dispatch")``, so enforcing the policy on it
        alone can never raise. A runner is nevertheless free to write anywhere
        its own sandbox allows (the claude-code write profile permits ``Bash``),
        and "it wrote outside what it declared" is precisely the containment
        failure D3l's detective layer exists to observe.

        Observing it is only half the control, so each observed path is adopted
        into this task's snapshot — the retained pre-run bytes for a file that
        existed, the recorded absence for one the runner created — which is what
        lets ``_classify``'s policy-violation branch really restore it and
        ``util restore-batch`` revert it later. A path with no retained bytes
        (denied, oversized, past the budget, or unreadable before the run) stays
        out of ``restorable`` and therefore out of ``files_touched``: promising a
        rollback point that does not exist is the failure this method's caller
        would otherwise ship.

        Deletions count: a declared-set-external file that disappeared was
        touched just as surely as one that grew.
        """
        with self._sweep_lock:
            before = self._census_before.pop(task_id, None)
            retained = self._retained_before.pop(task_id, None) or {}
            sources = self._ignore_sources_before.pop(task_id, None) or {}
            # Include every sibling that overlapped this census, even one
            # already classified. Its edits are still in the before/after
            # difference and must never enter this task's rollback snapshot.
            ignored = self._sweep_ignored.pop(task_id, set())
            # Registering a new census and closing this one share the lock,
            # so a sibling cannot start writing between this read and removal
            # from the overlap bookkeeping.
            after = self._census() if before is not None else None
            self._declared_by_task.pop(task_id, None)
            self._dispatch_started.pop(task_id, None)
        if before is None:
            return UndeclaredWrites(observed=False)
        if after is None:
            # The tree grew past the cap DURING the run (an `npm install`):
            # not "nothing moved", the sweep could not look (W1).
            return UndeclaredWrites(observed=False)
        moved = self._swept_moves(before, after, ignored, retained)
        moved -= self._ignored_and_harmless(task_id, moved, sources, after)
        observed = UndeclaredWrites(paths=sorted(moved))
        adopted = 0
        for relative in observed.paths:
            if adopted >= SWEEP_ADOPT_MAX_FILES:
                break
            stamp = before.get(relative)
            digest = retained.get(relative)
            try:
                entry = snapshot.adopt(
                    relative,
                    existed=stamp is not None,
                    sha256=digest,
                    size=stamp[0] if stamp else 0,
                    payload_source=self._blob_path(digest) if digest else None,
                    save=False,
                )
            except (WriteSafetyError, OSError):
                entry = None
            if entry is not None:
                observed.restorable.append(relative)
                adopted += 1
        if adopted:
            snapshot.save()
        return observed

    def _swept_moves(
        self,
        before: Dict[str, Tuple[int, int]],
        after: Dict[str, Tuple[int, int]],
        ignored: "set[str]",
        retained: Dict[str, str],
    ) -> "set[str]":
        """The paths outside ``ignored`` whose BYTES moved between two censuses.

        The stamps say which files to look at; the bytes decide. A path whose
        ``(size, mtime_ns)`` moved but whose content hashes to the digest the
        retention pass stored is not a write: a formatter no-op, a ``touch``,
        or a checkout of an unchanged file read as a partial application and
        stopped the loop (adversarial review, 2026-10-02; the declared set was
        already judged by sha256 in ``TaskSnapshot.changed``, so this is the
        same rule on the undeclared side). A path with no retained digest —
        denied, oversized, past the budget — stays moved: no bytes to compare
        is no reason to look away.
        """
        moved: set = set()
        for relative, stamp in after.items():
            if relative in ignored:
                continue
            previous = before.get(relative)
            if previous == stamp:
                continue
            digest = retained.get(relative) if previous is not None else None
            if digest and file_sha256(
                os.path.join(self.repo_root, relative)
            ) == digest:
                continue
            moved.add(relative)
        moved.update(
            relative for relative in before
            if relative not in after and relative not in ignored
        )
        return moved

    def _ignore_rule_sources(
        self, census: Optional[Dict[str, Tuple[int, int]]]
    ) -> Dict[str, Optional[str]]:
        """``source -> sha256 | None`` for every file that decides what git ignores.

        Every ``.gitignore`` the census saw (repo-relative), plus the three
        sources the census never covers: ``.git/info/exclude`` and
        ``.git/config`` (``.git`` is pruned from the walk) and the global
        excludes file (outside the repo). ``None`` records an absent file, so
        a source the runner *creates* reads as moved too. Hashed, not stamped,
        because a stamp is what a runner can put back.
        """
        if census is None:
            return {}
        sources = [
            relative for relative in census
            if relative == ".gitignore" or relative.endswith("/.gitignore")
        ]
        sources += [".git/info/exclude", ".git/config"]
        global_excludes = self._global_excludes()
        if global_excludes:
            sources.append(global_excludes)
        out: Dict[str, Optional[str]] = {}
        for source in sources:
            absolute = (
                source if os.path.isabs(source)
                else os.path.join(self.repo_root, source)
            )
            out[source] = file_sha256(absolute) if os.path.isfile(absolute) else None
        return out

    def _global_excludes(self) -> Optional[str]:
        """The absolute path of ``core.excludesFile``, resolved once per run."""
        if self._global_excludes_resolved:
            return self._global_excludes_path
        self._global_excludes_resolved = True
        path = ""
        try:
            result = subprocess.run(
                _git_utils.git_argv(
                    "config", "--path", "--get", "core.excludesFile", cwd=self.repo_root,
                ),
                cwd=self.repo_root, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                path = result.stdout.strip()
        except OSError:
            path = ""
        if not path:
            config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
                os.path.expanduser("~"), ".config"
            )
            path = os.path.join(config_home, "git", "ignore")
        self._global_excludes_path = os.path.abspath(path) if path else None
        return self._global_excludes_path

    def _ignored_and_harmless(
        self,
        task_id: str,
        moved: "set[str]",
        sources_before: Dict[str, Optional[str]],
        after: Optional[Dict[str, Tuple[int, int]]],
    ) -> "set[str]":
        """The moved paths git ignored *at dispatch* and the policy does not deny.

        A gitignored path is not part of the repository's change set: the host's
        own transcript (``logs/*.jsonl``), build output and caches moved during a
        dispatch were read as "changed outside this task's declared set", were
        handed to ``finalize-tracking`` as touched implementation files and from
        there became ``review-code``'s scope (2026-10-02, Codex run). They leave
        the sweep, with two exceptions:

        * a path ``driver.write_paths`` **denies** stays observable even when
          ignored, because the deny list's usual subjects (``.env``,
          ``**/*.pem``) are exactly what ``.gitignore`` hides, and dropping them
          would make the post-run policy check vacuous again. The deny side
          alone (:meth:`WritePolicy.denied`): asking :meth:`violation` also
          failed every ignored path a narrow ``allow`` did not list, which is
          how the fix only ever applied under ``allow: ["**"]`` (adversarial
          review, 2026-10-02);
        * **nothing** is dropped when the ignore rules themselves moved during
          the run. ``git check-ignore`` runs after the run, on rules the runner
          can write — a new ``.gitignore`` holding ``*``, or ``.git/info/exclude``
          (pruned from the census, so never "moved") — and read that way the
          drop hid every file the runner created, and its rule file with them
          (adversarial review, 2026-10-02). The rules in force at dispatch are
          hashed by :meth:`_ignore_rule_sources`; a moved source keeps the
          whole set in the sweep, which is the pre-#22 verdict and the safe one.
        """
        if not moved:
            return set()
        sources_after = self._ignore_rule_sources(after)
        if sources_after != sources_before:
            changed = sorted(
                source for source in set(sources_before) | set(sources_after)
                if sources_before.get(source) != sources_after.get(source)
            )
            self.log(
                f"[write-subagents] {task_id}: the ignore rules changed during "
                f"the run ({', '.join(changed[:5])}"
                f"{', …' if len(changed) > 5 else ''}), so no gitignored path "
                f"leaves the containment sweep for this task — the rules a "
                f"runner can write cannot decide what it hid (D3l)."
            )
            return set()
        ignored = _git_utils.ignored_paths(sorted(moved), cwd=self.repo_root)
        return {path for path in ignored if self.policy.denied(path) is None}

    def _moved_since_dispatch(
        self, task_id: str, snapshot: TaskSnapshot
    ) -> List[str]:
        """Every path this dispatch moved so far, declared or not. Read-only.

        ``snapshot.changed()`` (the declared set, by byte comparison) plus the
        paths the pre-run census says moved outside it — the same two sources
        :meth:`_classify` decides on, but **peeked**, not consumed: the census
        and its retained bytes stay registered for the terminal
        :meth:`_undeclared_writes` that still has to adopt them into the
        rollback snapshot. Siblings' declared paths are ignored exactly as
        there. Raises nothing it can avoid; a raise is read by the caller as
        "could not tell", which fails closed.
        """
        moved = set(snapshot.changed())
        with self._sweep_lock:
            before = self._census_before.get(task_id)
            retained = dict(self._retained_before.get(task_id) or {})
            sources = dict(self._ignore_sources_before.get(task_id) or {})
            ignored = set(self._sweep_ignored.get(task_id, ()))
            after = self._census() if before is not None else None
        if before is not None and after is not None:
            swept = self._swept_moves(before, after, ignored, retained)
            moved.update(
                swept - self._ignored_and_harmless(task_id, swept, sources, after)
            )
        return sorted(moved)

    def _observed_with_undeclared(
        self, snapshot: TaskSnapshot, undeclared: UndeclaredWrites
    ) -> Dict[str, Any]:
        """``snapshot.observed()`` plus a row per undeclared write, for the audit.

        Every observed path is in the audit line whether or not it could be made
        reversible, and each says which it is: an operator reading this log has to
        be able to tell "restore-batch will undo this" from "this one needs a
        human", and silence on the second is how the first gets assumed.
        """
        observed = snapshot.observed()
        restorable = set(undeclared.restorable)
        for relative in undeclared.paths:
            row = observed.get(relative)
            if row is None:
                row = {
                    "sha256_before": None,
                    "sha256_after": file_sha256(
                        os.path.join(self.repo_root, relative)
                    ),
                }
                observed[relative] = row
            row["declared"] = False
            row["rollback"] = "retained" if relative in restorable else "unavailable"
            row["note"] = (
                "observed by the post-run sweep, outside any declared set"
                if relative in restorable else
                "observed by the post-run sweep, outside any declared set; no "
                "pre-run bytes were retained for it, so it cannot be restored"
            )
        return observed

    def _dispatch_executor(
        self,
        task: Dict[str, Any],
        prompt: str,
        snapshot: TaskSnapshot,
        profile: str,
        declared: Sequence[str],
        *,
        batch_index: Optional[int] = None,
    ) -> TaskOutcome:
        """Real subprocess dispatch, with the D3g/D3l hooks bound to this task.

        Plan step 3 (2026-09-30): the phase's :class:`JailRequest` goes to the
        preflight (its scope joins the sandbox decision's key) and to
        ``dispatch_task``, and a batch that will run on the plugin-landlock rung
        is announced first with the stray-writes-only line.
        """
        task_id = str(task.get("task_id", ""))
        runner = self.forced_runner or resolve_task_runner(task, anchor=self.anchor)
        info = self._preflight(runner, profile)
        # F12: `subagent_batch_report.json.runner_versions` was empty on this
        # path — the read half fills it from its own batch-start preflight
        # (`_preflight_batch_runners`); this call already runs one per
        # dispatch and its version was simply never kept.
        self._runner_versions[runner] = str(info.get("version") or "unknown")
        rung = self._recorded_rung(runner, profile)
        self._dispatch_rungs[task_id] = rung
        request = self.jail_request
        self._announce_rung(runner, batch_index)

        def before_attempt(context: Dict[str, Any]) -> None:
            # The prompt this attempt really sends: a report re-ask (the
            # ladder's rule 7) sends its own, never the batch's instructions.
            sent = context.get("prompt") or prompt
            self.audit.dispatched(
                task_id=context["task_id"],
                kind=self.kind,
                attempt=context["attempt"],
                runner=runner,
                profile=profile,
                cwd=context["cwd"],
                sandbox_root=self.repo_root,
                argv=self._render_argv(runner, sent, profile),
                prompt=sent,
                declared_files=list(declared),
                snapshot_dir=snapshot.directory,
                snapshot_before={
                    entry.relative: entry.sha256 for entry in snapshot.entries
                },
            )

        def after_result(context: Dict[str, Any]) -> None:
            # The executor's own record of the rung this attempt ran on wins;
            # an executor that predates plan step 3 sends none, and the
            # preflight's recorded decision stands in for it.
            seen = context["sandbox_rung"] if "sandbox_rung" in context else rung
            self._dispatch_rungs[task_id] = seen
            self._classify(task, context["outcome"], snapshot, context["result"],
                           runner=runner, profile=profile,
                           attempt=context["attempt"],
                           exit_code=context["exit_code"],
                           duration=context["duration_seconds"],
                           sandbox_rung=seen,
                           jail_denials=context.get("jail_denials"))

        def changed_files(context: Dict[str, Any]) -> List[str]:
            # T4.8 (plan opencode-validator-false-rate-limit-and-denied-turn
            # §4.4 step 4): a turn a denial ended has no result, so `_classify`
            # never runs and `dispatch_task` would retry on top of whatever it
            # half-applied. Asked before that retry decision, against the same
            # pre-batch snapshot and pre-run census `_classify` reads.
            return self._moved_since_dispatch(context["task_id"], snapshot)

        def tree_fingerprint(context: Dict[str, Any]) -> Dict[str, Optional[str]]:
            # A report re-ask's guard: the current bytes of every declared file
            # and of every path this dispatch has moved so far (the same two
            # sources `_classify` decides on, peeked). The executor compares the
            # value before and after the re-ask; any difference is an edit the
            # re-ask was told not to make.
            moved = self._moved_since_dispatch(context["task_id"], snapshot)
            paths = set(moved) | {entry.relative for entry in snapshot.entries}
            return {
                relative: file_sha256(os.path.join(self.repo_root, relative))
                for relative in sorted(paths)
            }

        extra: Dict[str, Any] = {}
        if request is not None and _accepts_keyword(dispatch_task, "jail_request"):
            extra["jail_request"] = request
        if _accepts_keyword(dispatch_task, "tree_fingerprint"):
            extra["tree_fingerprint"] = tree_fingerprint
        return dispatch_task(
            task,
            phase_dir=self.phase_dir,
            runner=runner,
            anchor=self.anchor,
            cwd=self.repo_root,
            prompt=prompt,
            before_attempt=before_attempt,
            after_result=after_result,
            changed_files=changed_files,
            **extra,
        )

    def _dispatch_native(
        self,
        task: Dict[str, Any],
        prompt: str,
        snapshot: TaskSnapshot,
        profile: str,
    ) -> TaskOutcome:
        """The simulated native invoker: same persist, same classification (D3a).

        A driving agent runs ``task.prompt`` with its own subagent tool and writes
        the JSON it got back. That write is :func:`persist_result` and the verdict is
        :meth:`_classify` — the *same* functions the executor path uses — so the two
        invokers produce byte-identical result files and identical terminal states
        by construction rather than by convention.
        """
        task_id = str(task.get("task_id", ""))
        outcome = TaskOutcome(
            task_id=task_id, kind=self.kind, runner="native", status=STATUS_OK,
            profile=profile,
        )
        started = time.time()
        self.audit.dispatched(
            task_id=task_id, kind=self.kind, attempt=1, runner="native",
            profile=profile, cwd=self.repo_root, sandbox_root=self.repo_root,
            argv=["<native subagent>"], prompt=prompt,
            declared_files=[
                relative_to_root(p, self.repo_root) for p in declared_file_set(task)
            ],
            snapshot_dir=snapshot.directory,
            snapshot_before={e.relative: e.sha256 for e in snapshot.entries},
        )
        data = self.respond(task, prompt)  # type: ignore[misc]
        if isinstance(data, str):
            try:
                from .json_extractor import extract_json_from_text
            except ImportError:  # pragma: no cover
                from utils.json_extractor import extract_json_from_text
            extracted = extract_json_from_text(data, prefer_arrays=False)
            if not extracted.get("success"):
                outcome.status = STATUS_SCHEMA_INVALID
                outcome.write_outcome = WRITE_SCHEMA_REJECTED
                outcome.reason = extracted.get("error") or "no JSON in the reply"
                return outcome
            data = extracted["data"]
        errors = validate_result(self.kind, data)
        if errors:
            outcome.status = STATUS_SCHEMA_INVALID
            outcome.write_outcome = WRITE_SCHEMA_REJECTED
            outcome.reason = (
                f"the reply does not match the {self.kind} result shape: "
                + "; ".join(errors[:5])
            )
            return outcome
        outcome.output_path = persist_result(task, self.phase_dir, data)
        self._classify(task, outcome, snapshot, data, runner="native",
                       profile=profile, attempt=1, exit_code=0,
                       duration=time.time() - started)
        return outcome

    def _render_argv(self, runner: str, prompt: str, profile: str) -> List[str]:
        """The rendered argv for the audit record / dry run, never executed here.

        Plan step 3 (2026-09-30): when the preflight recorded the
        plugin-landlock rung for this runner and scope, the argv is the wrapped
        shape — the rung's atoms and developer instructions, then
        ``landlock_dispatch.preview_wrap_argv``'s prefix — with placeholder
        paths for the per-attempt scratch directory and ``$CODEX_HOME``. It
        creates nothing: no scratch directory, no context, no allow-list
        resolution (F1 "Task-specific paths stay out of shared state").
        """
        try:
            from .provider_registry import get_provider
        except ImportError:  # pragma: no cover
            from utils.provider_registry import get_provider
        provider_name, model = parse_model_spec(runner, anchor=self.anchor)
        provider = get_provider(provider_name)
        if provider is None:
            return [runner]
        try:
            argv = list(provider.build_agent_command(prompt, model, profile))
        except Exception:  # noqa: BLE001 - a log must never fail a dispatch
            return [provider_name, "<profile unsupported>"]
        rung_id = self._recorded_rung(runner, profile)
        if rung_id and rung_id != SANDBOX_RUNG_PLUGIN_LANDLOCK:
            # Any other recorded rung (F5's uncontained one, plan step 6) is
            # rendered as the dispatch will render it, so the audit record
            # never shows a sandbox the child did not get; a render the adapter
            # refuses leaves today's argv.
            try:
                consent = host_write_posture_consent(write_scope=self.write_scope)
                if consent is not None and _accepts_keyword(
                    provider.apply_sandbox_fallback, "consent"
                ):
                    return list(provider.apply_sandbox_fallback(
                        list(argv), profile, rung_id, consent=consent,
                    ))
                return list(provider.apply_sandbox_fallback(
                    list(argv), profile, rung_id,
                ))
            except Exception:  # noqa: BLE001 - a preview must never fail a dispatch
                return argv
        if rung_id != SANDBOX_RUNG_PLUGIN_LANDLOCK or self.jail_request is None:
            return argv
        try:
            rung = next(
                candidate for candidate in provider.sandbox_fallback_rungs(profile)
                if candidate.name == rung_id
            )
            rendered = provider.apply_sandbox_fallback(
                list(argv), profile, rung_id,
                context=_landlock.preview_context(self.jail_request),
            )
            return _landlock.preview_wrap_argv(rendered, rung, self.jail_request)
        except Exception:  # noqa: BLE001 - a preview must never fail a dispatch
            return argv

    # -- D3g: the classification ----------------------------------------

    def _classify(
        self,
        task: Dict[str, Any],
        outcome: TaskOutcome,
        snapshot: TaskSnapshot,
        result: Any,
        *,
        runner: str,
        profile: str,
        attempt: int,
        exit_code: Optional[int],
        duration: float,
        sandbox_rung: Optional[str] = None,
        jail_denials: Any = None,
    ) -> None:
        """Decide which of the four D3g outcomes this dispatch landed on.

        Precedence, and every branch of it matters:

        * The runner **claimed** something (``status``) and the snapshot **shows**
          something (bytes moved or not). The claim is never the verdict.
        * ``applied`` claimed + empty diff → **clean failure**: recorded, the
          loop continues, and the prior-changes entry takes the documented
          ``(batch failed — …, verified no file changes)`` form.
        * ``no-op`` claimed + empty diff → **no-op**: the claim and the verdict
          agree, so it is recorded under its own name (``(no-op — …, verified
          no file changes)``) and the loop continues. Labelling it a clean
          failure told the next batch the work had failed.
        * ``no-op`` claimed + non-empty diff (and every expected id came back,
          so this is NOT the partial-application row below) → still
          **applied** — that is the verdict ``util restore-batch`` needs — but
          the claim and the bytes disagree, and `_classify` used to fall
          straight through to the plain applied branch with no ``reason`` at
          all (N2, 2026-09-16: T006 read `applied` beside a summary saying
          "already implemented", self-contradicting). ``reason`` now names
          which of two things happened: the pre-batch snapshot was **reused**
          from an earlier dispatch of this same batch (`_snapshot_reused`), so
          the diff is that attempt's and not this one's — or this dispatch
          took its own fresh snapshot and the bytes still moved, which is the
          foreign-or-undeclared-write signature the undeclared-writes sweep
          catches OUTSIDE the declared set, just landing inside it instead;
          that second case also warns on the log.
        * ``failed`` claimed (or a non-zero exit) + empty diff → the batch
          **failed cleanly**: nothing to roll back, but not ``ok`` either — the
          loop stops at the boundary and ``--resume`` re-dispatches it. A runner
          that says it could not do the work has not done it.
        * bytes moved + the batch reported ``failed``, **or** only some of the
          batch's ``expected_ids`` came back, **or** the runner exited non-zero →
          **partial application**: stops the loop, keeps the snapshots, exits 3
          even under ``--yes``.
        * a touched file outside ``driver.write_paths`` → **policy violation**:
          the batch fails, the offending files are restored from the snapshot, the
          loop stops (D3l post-run enforcement). "Touched" is the set the sweep
          observed, declared or not, and the sweep retained its pre-run bytes, so
          "restored" means an undeclared write too — see
          :meth:`_undeclared_writes`.

        A prose result never reaches here at all — it already failed rung 6 as
        ``schema_invalid``, which is the fourth outcome.

        **On the plugin-landlock rung** (plan step 3, 2026-09-30; F1 "A denied
        edit is never recorded as applied" and "Supported edits under a
        file-only grant") two checks sit after the policy-violation branch and
        ahead of every claim-based one, because codex ends a turn whose every
        write the jail refused on ``turn.completed`` with exit 0 and the claim
        alone would read it as applied:

        * **a replaced target** — the target's ``(st_dev, st_ino)`` differs
          from the one recorded at this dispatch's pre-batch snapshot. The
          jail's rule is bound to the inode it opened, so the runner's edits
          were refused or went to the old file. A failed batch with its own
          reason, checked first because it explains any denials; the claim
          never overrides it; nothing is rolled back over the file now at the
          path (it is not the runner's, and may be the user's) and the
          snapshot is kept for comparison; the loop stops at exit 3, even under
          ``--yes``, and ``--resume`` re-dispatches under a fresh rule;
        * **the jail's denials** (``jail_denials``, read off the child's
          stream): a failed ``file_change`` on any path, or a command denial on
          an in-scope path, is a failed batch when no bytes moved and a partial
          application when some did — never ``ok``. Under a file-only grant a
          ``replacement`` denial counts as in scope while the target is
          unchanged, and never makes a partial application once the target
          changed in place (the runner recovered). Out-of-scope denials alone
          leave today's verdict, with their count on the batch's line and the
          outcome. The bytes, never ``changes[]``, say what landed.
        """
        claimed = str((result or {}).get("status") or "")
        summary = str((result or {}).get("summary") or "").strip()
        notes = str((result or {}).get("notes") or "").strip()
        outcome.runner_notes = " ".join(notes.split())[:600] or None
        reported_ids = [str(i) for i in ((result or {}).get("ids") or [])]
        expected_ids = [str(i) for i in (task.get("expected_ids") or [])]
        changed = snapshot.changed()
        undeclared = self._undeclared_writes(outcome.task_id, snapshot)
        observed = self._observed_with_undeclared(snapshot, undeclared)

        # F9: recorded on the outcome itself — every branch below shares this
        # object, so both `write_batch_report.json` (`result.to_dict()`) and
        # this run's own `subagent_batch_report.json` (`BatchReport` built from
        # the same `result.outcomes` in `_write_report`) carry the same
        # verification evidence that used to live only in `subagent_audit.jsonl`
        # and the per-batch result file.
        outcome.expected_ids = expected_ids
        outcome.returned_ids = reported_ids
        # Whatever branch decides it, a file-less implement task's verdict
        # reads VERIFIED or PARTIAL on every surface (`outcome_label`).
        outcome.verification_only = self._verification_only(snapshot)
        outcome.sha256_before = {
            relative: row.get("sha256_before") for relative, row in observed.items()
        }
        outcome.sha256_after = {
            relative: row.get("sha256_after") for relative, row in observed.items()
        }

        # `changed` stays the DECLARED verdict — it is what the four outcomes are
        # decided on below. `files_touched` is the ROLLBACK set, so it also
        # carries the undeclared writes the sweep retained bytes for; an
        # undeclared path with no rollback point is deliberately absent, because
        # `util restore-batch` reads this list and would report a file it cannot
        # actually restore.
        outcome.files_touched = list(changed) + list(undeclared.restorable)
        outcome.summary = summary or None

        # D3l post-run: recompute the touched set and enforce the policy on it.
        # The observed set is the declared files whose bytes moved PLUS whatever
        # else under the pinned root the sweep saw move; enforcing on `changed`
        # alone is vacuous, because every path in it is one the identical
        # pre-dispatch check already passed.
        if undeclared:
            stranded = undeclared.unrecoverable
            self.log(
                f"[write-subagents] {outcome.task_id}: {len(undeclared)} file(s) "
                f"changed outside this task's declared set "
                f"({', '.join(undeclared.paths[:5])}"
                f"{', …' if len(undeclared) > 5 else ''}) — checking them against "
                f"driver.write_paths."
                + (
                    f" {len(stranded)} of them had no retained pre-run bytes and "
                    f"cannot be restored automatically."
                    if stranded else ""
                )
            )
        try:
            self.policy.check(list(changed) + undeclared.paths, when="post-run")
        except PolicyViolation as exc:
            report = snapshot.restore(exc.paths)
            stranded = sorted(
                path for path, state in report.items()
                if state.startswith("skipped:")
            )
            outcome.status = STATUS_POLICY_VIOLATION
            outcome.write_outcome = WRITE_POLICY_VIOLATION
            restored = sorted(
                path for path, state in report.items()
                if not state.startswith("skipped:")
            )
            many = len(exc.paths) != 1
            # The message used to say two things at once: "restored from the
            # pre-batch snapshot (x: skipped: not in this task's snapshot)" and
            # "remain on disk exactly as the runner left them". It also named the
            # runner as the author, which the sweep cannot know: it sees a path
            # that moved during the task's window (2026-10-02: a coordinator's
            # own mid-run edit of `.multi-llm/providers.yaml` failed T009). Each
            # path is now in exactly one of the two lists below.
            outcome.reason = (
                f"{exc.message} "
                f"{'These paths' if many else 'This path'} changed during this "
                f"task's window; the executor sees that they moved, not who "
                f"moved them, so the change may be another process's or your "
                f"own edit and not the runner's."
                + (
                    f" Restored from the pre-batch snapshot: "
                    f"{', '.join(restored)}."
                    if restored else ""
                )
                + (
                    # A path the sweep could not retain bytes for (a policy-denied
                    # one above all: its bytes are deliberately never copied into
                    # an executor artifact) is NOT restored, and says so.
                    f" Not restored, because no rollback point exists "
                    f"({', '.join(stranded[:5])}"
                    f"{', …' if len(stranded) > 5 else ''}): left on disk as "
                    f"{'they are' if len(stranded) != 1 else 'it is'} now, so "
                    f"inspect {'them' if len(stranded) != 1 else 'it'} by hand."
                    if stranded else ""
                )
                + " The loop stops here."
            )
            self.audit.restored(task_id=outcome.task_id, report=report, force=False)
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return

        # Plan step 3 (2026-09-30): the plugin-landlock rung's two verdicts,
        # after the policy branch and ahead of every claim-based one.
        identity_before = self._target_identity_before.pop(outcome.task_id, None)
        on_rung = sandbox_rung == SANDBOX_RUNG_PLUGIN_LANDLOCK
        if on_rung and self._replaced_target(
            outcome, snapshot, identity_before, claimed
        ):
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return
        denials = _as_jail_denials(jail_denials)
        if denials is not None and self._jail_verdict(
            outcome, snapshot, denials, changed, undeclared, claimed,
        ):
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return

        if self._verification_only(snapshot) and self._verification_verdict(
            outcome, snapshot, undeclared, claimed, summary,
            changed=changed, crashed_exit=exit_code,
            check=check_findings_from(result),
        ):
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return

        # `ids` is OPTIONAL in write_batch_result.schema.json, so a batch can move
        # bytes and say nothing at all about *which* of its expected changes
        # landed. The old `if reported_ids` guard emptied `missing_ids` in exactly
        # that case, which made every id-less reply — including a `no-op` claim
        # contradicted by a non-empty diff — read as a full application, and D3n's
        # spot-check ("this batch's expected changes landed, rather than merely
        # that bytes moved") silently never happened. An unattested batch that
        # moved bytes is a partial application until something attests it.
        if expected_ids and not reported_ids:
            missing_ids = list(expected_ids)
        else:
            missing_ids = [i for i in expected_ids if i not in reported_ids]

        # D3k inverts the exit-code rule for write tasks. On the read path a
        # schema-valid result wins over a non-zero exit (CLIs exit non-zero on
        # warnings while still printing the answer, and nothing is at stake but a
        # report); here it fails closed — a non-zero exit never counts as success
        # on its own, and a non-zero exit with a non-empty diff is a partial
        # application, never an applied batch. The case this exists for is a
        # runner that printed an optimistic `{"status": "applied", "ids": [all of
        # them]}` and *then* died mid-edit: the claim is schema-valid and the ids
        # are complete, so claim-and-diff alone read it as a full application and
        # the loop dispatched batch N+1 on top of half-applied bytes.
        # ``None`` is "no exit code observed" (the native invoker), not zero.
        crashed = exit_code not in (0, None)

        if changed and (crashed or claimed == "failed" or missing_ids):
            details = []
            if crashed:
                details.append(
                    f"the runner exited {exit_code} (claiming "
                    f"{claimed or 'no status'!r})"
                )
            if claimed == "failed":
                details.append(f"the batch reported {claimed!r}")
            elif missing_ids and not reported_ids:
                details.append(
                    f"it reported no `ids` at all, so nothing attests that the "
                    f"{len(expected_ids)} expected change(s) "
                    f"({', '.join(expected_ids[:5])}"
                    f"{', …' if len(expected_ids) > 5 else ''}) are what landed"
                )
            elif missing_ids:
                # Counted from the expected side: the reply's own `ids` may
                # name things this batch does not hold. OpenCode racing run 4
                # (2026-10-05): a report re-ask restated "Suggestion 1",
                # "Suggestion 2", and the reason read "only 2 of 2 expected
                # id(s) came back (missing: <both of them>)".
                matched = len(expected_ids) - len(missing_ids)
                unexpected = [i for i in reported_ids if i not in expected_ids]
                details.append(
                    (f"only {matched} of {len(expected_ids)}" if matched
                     else f"none of the {len(expected_ids)}")
                    + f" expected id(s) came back (missing: "
                    f"{', '.join(missing_ids)})"
                    + (
                        f"; the reply named {len(unexpected)} id(s) this batch "
                        f"does not hold ({', '.join(repr(i) for i in unexpected[:5])}"
                        f"{', …' if len(unexpected) > 5 else ''})"
                        if unexpected else ""
                    )
                )
            detail = " and ".join(details)
            outcome.status = STATUS_PARTIAL
            outcome.write_outcome = WRITE_PARTIAL
            outcome.reason = (
                f"PARTIAL APPLICATION: {len(changed)} file(s) changed "
                f"({', '.join(changed)}) but {detail}. The batch loop stops here, "
                f"the pre-batch snapshot is kept at {snapshot.directory}, and the "
                f"phase is NOT marked completed — an automatic revert would destroy "
                f"the half-applied state a human needs to inspect."
            )
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return

        if changed and claimed == "no-op":
            # N2: reachable only past the partial-application guard above, so
            # every expected id came back (or none were expected) and the
            # runner did not crash — the only thing left unexplained is a
            # 'no-op' claim next to bytes that moved. The verdict stays
            # APPLIED (`util restore-batch` needs exactly this outcome), but
            # the two ways a 'no-op' claim can disagree with the bytes are
            # different events and the reason says which one happened here.
            outcome.write_outcome = WRITE_APPLIED
            outcome.status = STATUS_OK
            changed_names = ", ".join(changed)
            if outcome.task_id in self._snapshot_reused:
                outcome.reason = (
                    f"APPLIED (no-op claimed, bytes moved): {changed_names} "
                    f"differ from the pre-batch snapshot, but that snapshot "
                    f"was reused from an earlier dispatch of this batch — "
                    f"the diff is that attempt's work, not this dispatch's. "
                    f"Recorded as applied."
                )
            else:
                outcome.reason = (
                    f"APPLIED (no-op claimed, bytes moved): {changed_names} "
                    f"differ from the snapshot taken at the start of THIS "
                    f"dispatch — a foreign-or-undeclared-write signature, not "
                    f"an artifact of resume. Recorded as applied because the "
                    f"bytes are real; the 'no-op' claim should not be trusted."
                )
                self.log(
                    f"[write-subagents] warning: {outcome.task_id} claimed "
                    f"'no-op' but {changed_names} changed against a snapshot "
                    f"taken this dispatch — treating the bytes as "
                    f"authoritative."
                )
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return

        if not changed:
            declared_names = (
                ", ".join(e.relative for e in snapshot.declared_entries())
                or "any declared file"
            )
            if crashed or claimed == "failed":
                # The runner's own verdict, and the bytes agree. Nothing to
                # restore, but nothing done either: the loop stops here rather
                # than dispatching the next batch on top of the gap, and the
                # status is one `ok` never reads as. D3k: a non-zero exit never
                # counts as success on its own.
                outcome.write_outcome = WRITE_CLEAN_FAILURE
                outcome.status = STATUS_BATCH_FAILED
                what = (
                    f"the runner exited {exit_code} (claiming "
                    f"{claimed or 'no status'!r})"
                    if crashed else f"the batch reported {claimed!r}"
                )
                outcome.reason = (
                    f"FAILED: {what} and the pre-batch snapshot comparison "
                    f"confirms no byte changes in {declared_names}. Nothing to "
                    f"restore; the loop stops here and --resume re-dispatches "
                    f"this batch."
                    + (f" Runner summary: {summary}" if summary else "")
                )
            elif claimed == "no-op":
                # The claim and the verdict agree: nothing needed changing.
                outcome.write_outcome = WRITE_NO_OP
                outcome.status = STATUS_OK
                outcome.reason = (
                    f"NO-OP: the batch reported 'no-op' and the pre-batch "
                    f"snapshot comparison confirms no byte changes in "
                    f"{declared_names}. Recorded; the loop continues."
                )
            else:
                # Success claimed, nothing moved. Recorded as a clean failure
                # and the loop continues: the plan may simply already contain
                # the change.
                outcome.write_outcome = WRITE_CLEAN_FAILURE
                outcome.status = STATUS_OK
                outcome.reason = (
                    f"CLEAN FAILURE: the batch reported {claimed or 'no status'!r} "
                    f"but the pre-batch snapshot comparison found no byte changes "
                    f"in {declared_names}. Recorded; the loop continues."
                )
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return

        # Reachable only on a clean exit: `crashed` with a non-empty diff is the
        # partial branch above and `crashed` with an empty one is the clean
        # failure, so no non-zero exit can land on APPLIED (D3k).
        outcome.write_outcome = WRITE_APPLIED
        outcome.status = STATUS_OK
        self._audit_completion(outcome, runner, profile, attempt, exit_code,
                               observed, duration)

    def _verification_only(self, snapshot: TaskSnapshot) -> bool:
        """Whether this dispatch declared no files at all (``implement`` only)."""
        return self.kind == "implement" and not snapshot.declared_entries()

    def _verification_verdict(
        self,
        outcome: TaskOutcome,
        snapshot: TaskSnapshot,
        undeclared: UndeclaredWrites,
        claimed: str,
        summary: str,
        *,
        changed: Sequence[str],
        crashed_exit: Optional[int],
        check: Optional[List[str]] = None,
    ) -> bool:
        """The verdict for a task that declared no files. ``True`` when decided.

        D3h's guarantee is that the executor, not the runner's claim, says what
        changed. A task with no declared files makes that guarantee *stricter*,
        not void: it promised to change nothing, so every path the sweep saw move
        is a write it never declared.

        * the sweep could not look (``undeclared.observed`` is false: the tree
          grew past ``SWEEP_MAX_FILES`` during the run — a pre-run census that
          is already unusable is refused at ``_begin_sweep``): a failed batch,
          never VERIFIED. There is no evidence either way, and "no evidence"
          read as "nothing moved" was the one way a file-less task could edit
          anything unobserved (adversarial review, 2026-10-02).
        * a path moved: a partial application. The loop stops, the sweep already
          adopted each path into this task's snapshot (so ``util restore-batch``
          can revert it) and nothing is reverted automatically, which is what
          every other unattested write gets. The reason says how to proceed when
          the task is *meant* to fix what it checks: declare those files under
          ``Files to modify`` in the tasks file and ``--resume``. ``changed`` —
          the snapshot's own byte comparison — joins the sweep's paths here:
          under ``--resume`` the reused snapshot still carries the entries the
          earlier dispatch's sweep adopted, so a stray edit the earlier run
          left on disk stays PARTIAL until it is declared or restored, instead
          of being laundered by a second run that touched nothing (adversarial
          review, 2026-10-02).
        * nothing moved and the runner crashed, or claimed ``failed``: falls
          through to the ordinary clean-failure branch (``False``), whatever
          its ``check`` lists. The contract (:data:`CHECK_RESULT_CONTRACT`)
          spells ``failed`` "a check that could not run" — a missing command,
          a crash, a timeout — and a check that did not run says nothing
          about the code, so it is never recorded as findings the run goes
          on past (fix round 11: ``failed`` + ``findings: ["npx: command not
          found"]`` read as check-findings, so any failure could be laundered
          into one). The findings, when the reply listed some and wrote no
          ``notes``, become what the stop says the check reported.
        * nothing moved and its reply's structured ``check`` (*check*,
          :func:`subagent_executor.check_findings_from`) lists findings:
          ``no-op``/``ok`` labelled ``check-findings``. The findings are named
          on every surface and recorded in ``state.json`` for review-code
          (racing re-run 4, 2026-10-05: T007's ``npx html-validate`` reported
          two problems and the task read VERIFIED, the problems only in its
          free-text notes). The tasks that do not depend on the check go on;
          the ones that do are not run (:meth:`_run`).
        * nothing moved and the runner did not fail: ``no-op``/``ok``. The
          check ran and left the repo as it was, which is a verification
          passing, and the loop continues. Only a ``check`` that clearly says
          it passed, or none at all (an older reply), reads so: an empty or
          unreadable one is :data:`subagent_executor.CHECK_RESULT_UNKNOWN`,
          a finding.
        """
        if not undeclared.observed:
            outcome.status = STATUS_BATCH_FAILED
            outcome.write_outcome = WRITE_CLEAN_FAILURE
            outcome.reason = (
                f"BATCH FAILED: {outcome.task_id} declares no files (a "
                f"verification-only task), so the post-run containment sweep "
                f"is its whole verdict, and the sweep could not run: the tree "
                f"under {self.repo_root} holds more than {SWEEP_MAX_FILES} "
                f"files after the run (D3l). Whether the task changed anything "
                f"is unknown, so it is not VERIFIED. Inspect the working tree, "
                f"take the untracked trees (node_modules/, build output) out "
                f"of the checkout or list the files it may change under "
                f"`Files to modify`, and re-run with --resume."
            )
            return True
        moved = sorted(set(undeclared.paths) | set(changed))
        if moved:
            shown = ", ".join(moved[:5]) + (", …" if len(moved) > 5 else "")
            earlier = [path for path in moved if path not in undeclared.paths]
            outcome.status = STATUS_PARTIAL
            outcome.write_outcome = WRITE_PARTIAL
            outcome.reason = (
                f"PARTIAL APPLICATION: {outcome.task_id} declares no files (a "
                f"verification-only task) but {len(moved)} file(s) changed "
                f"during its run ({shown}). "
                + (
                    f"{len(earlier)} of them "
                    f"({', '.join(earlier[:5])}{', …' if len(earlier) > 5 else ''}) "
                    f"still differ from the pre-batch snapshot an earlier "
                    f"dispatch of this task took: left on disk by that run, "
                    f"not restored since. "
                    if earlier and outcome.task_id in self._snapshot_reused
                    else ""
                )
                + f"The loop stops here and the "
                f"pre-batch snapshot is kept at {snapshot.directory}. If this "
                f"task is meant to fix what it checks, list those files under "
                f"`Files to modify` in the tasks file and re-run with --resume; "
                f"otherwise restore them with `util restore-batch`."
            )
            return True
        if crashed_exit not in (0, None):
            return False
        if claimed == "failed":
            if check and not outcome.runner_notes:
                outcome.runner_notes = "; ".join(check)[:600]
            return False
        if check:
            outcome.write_outcome = WRITE_NO_OP
            outcome.status = STATUS_OK
            outcome.check_findings = list(check)
            shown = "; ".join(check[:3]) + (
                f"; +{len(check) - 3} more" if len(check) > 3 else ""
            )
            outcome.reason = (
                f"CHECK FINDINGS: {outcome.task_id}'s check ran and reported "
                f"{len(check)} problem(s): {shown}. No file changed, so the "
                f"loop continues; the findings are listed in the "
                f"implementation summary and go to review-code."
            )
            return True
        outcome.write_outcome = WRITE_NO_OP
        outcome.status = STATUS_OK
        outcome.check_findings = check
        outcome.reason = (
            f"VERIFIED: {outcome.task_id} declares no files and the sweep found "
            f"no file changed during its run"
            + (
                " and its check reported that it passed"
                if check is not None else
                " (its reply reported no `check` result)"
            )
            + ". Recorded; the loop continues."
            + (f" Runner summary: {summary}" if summary else "")
        )
        return True

    def _relative(self, path: str) -> str:
        """``path`` repo-relative when it lies under the root, else as given."""
        return relative_to_root(path, self.repo_root)

    def _in_grant(self, path: str, task_id: Optional[str] = None) -> bool:
        """Whether the resolved ``path`` is inside this phase's write grant —
        under ``repo-wide`` (one dir rule on the root, 2026-10-05), inside
        the repository."""
        request = self.jail_request
        if request is not None and request.write_scope == WRITE_SCOPE_REPO_WIDE:
            root = os.path.realpath(request.repo_root)
            resolved = os.path.realpath(path)
            return resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep)
        if request is None or not request.target:
            return False
        resolved = os.path.realpath(path)
        if request.target_dir:
            base = os.path.realpath(request.target_dir)
            return resolved == base or resolved.startswith(base.rstrip(os.sep) + os.sep)
        return resolved == os.path.realpath(request.target)

    def _replaced_target(
        self,
        outcome: TaskOutcome,
        snapshot: TaskSnapshot,
        before: Optional[Tuple[int, int]],
        claimed: str,
        *,
        ended: Optional[str] = None,
    ) -> bool:
        """The replaced-target verdict (F1 "The target replaced during the
        dispatch"). ``True`` when it decided the outcome.

        Detection only — the plugin never guesses which process swapped the
        file, and never widens the grant to the directory to ride a swap out.
        ``ended`` is the status of a dispatch that never reached
        ``_classify`` (no claim to override), named in the claim's place.
        """
        path = self._identity_path()
        if path is None or before is None:
            return False
        after = self._identity_of(path)
        if after == before:
            return False
        label = self._target_label()
        shown = self._relative(path)
        # No rollback over the new file: `files_touched` is what
        # `util restore-batch` restores, so the target (and, under the dir
        # rule, anything beneath it) leaves it. The pre-batch snapshot stays on
        # disk for comparison.
        outcome.files_touched = [
            relative for relative in outcome.files_touched
            if not self._in_grant(os.path.join(self.repo_root, relative))
        ]
        outcome.status = STATUS_BATCH_FAILED
        outcome.write_outcome = WRITE_CLEAN_FAILURE
        now = "is gone" if after is None else "now names a different file"
        over = (
            f"whatever the runner claimed ({claimed or 'no status'!r})"
            if ended is None
            else f"whatever the dispatch itself ended as ({ended!r})"
        )
        outcome.reason = (
            f"FAILED (target replaced): {label} was replaced while the runner "
            f"was editing it, and the jail's grant stayed on the old file, so "
            f"the runner's edits were refused or lost — {shown} {now} "
            f"(inode {before[1]} before the dispatch, "
            f"{after[1] if after else 'none'} after), {over}. Nothing was "
            f"rolled back over the file now at that path: its bytes are not the "
            f"runner's. The pre-batch snapshot is kept at {snapshot.directory} "
            f"for comparison. The loop stops here; --resume re-dispatches this "
            f"batch against the file now at the path, under a fresh grant."
        )
        self._jail_failed.add(outcome.task_id)
        return True

    def _replaced_target_unclassified(
        self, outcome: TaskOutcome, snapshot: TaskSnapshot
    ) -> bool:
        """The replaced-target verdict for a plugin-landlock dispatch that
        never reached ``_classify``. ``True`` when it decided the outcome.

        ``_classify`` runs only on a schema-valid result. A timeout, a runner
        error, a schema-invalid reply, a ``[JAIL_DENIED]`` rejection or a
        T4.8 partial ends in :meth:`_audit_terminal` instead, whose
        ``files_touched`` is the byte comparison against the snapshot. A target
        saved over by rename differs from that snapshot, so without this check
        the audit record, and the stop message's restore-batch command, would
        overwrite the new file, which may be the user's. F1: a replaced target
        is never restored. The verdict is :meth:`_replaced_target`'s, with the
        dispatch's own failure kept at the end of the reason.
        """
        task_id = outcome.task_id
        if self._dispatch_rungs.get(task_id) != SANDBOX_RUNG_PLUGIN_LANDLOCK:
            return False
        before = self._target_identity_before.pop(task_id, None)
        ended, detail = outcome.status, outcome.reason
        if not self._replaced_target(outcome, snapshot, before, "", ended=ended):
            return False
        if detail:
            outcome.reason = f"{outcome.reason} The dispatch's own failure: {detail}"
        return True

    def _jail_verdict(
        self,
        outcome: TaskOutcome,
        snapshot: TaskSnapshot,
        denials: Any,
        changed: Sequence[str],
        undeclared: UndeclaredWrites,
        claimed: str,
    ) -> bool:
        """The jail-denial verdict (F1 "A denied edit is never recorded as
        applied"). ``True`` when it decided the outcome; ``False`` leaves the
        claim-based verdict, with any denial count noted for the batch's line.
        """
        moved = list(changed) + [
            path for path in undeclared.paths if path not in changed
        ]
        hard = bool(denials.failed_file_changes or denials.in_scope_paths)
        replaced = list(denials.replacement_paths)
        self._jail_denials[outcome.task_id] = denials.to_dict()
        if hard or (replaced and not changed):
            paths = [self._relative(p) for p in denials.denied_paths()]
            in_grant = [
                self._relative(p) for p in denials.denied_paths()
                if self._in_grant(p, outcome.task_id) or p in replaced
            ]
            outside = [p for p in paths if p not in in_grant]
            denied_text = (
                f"the plugin-landlock jail refused the runner's write to "
                f"{', '.join(paths) or 'a path it did not name'}"
            )
            tail = self._jail_scope_sentence(in_grant, outside)
            if hard and moved:
                outcome.status = STATUS_PARTIAL
                outcome.write_outcome = WRITE_PARTIAL
                outcome.reason = (
                    f"PARTIAL APPLICATION (jail-denied edit): {len(moved)} "
                    f"file(s) changed ({', '.join(moved)}) but {denied_text}; "
                    f"the bytes, not the runner's claim ({claimed or 'no status'!r}), "
                    f"say what landed. {tail}The batch loop stops here, the "
                    f"pre-batch snapshot is kept at {snapshot.directory}, and "
                    f"the phase is NOT marked completed — an automatic revert "
                    f"would destroy the half-applied state a human needs to "
                    f"inspect."
                )
            else:
                declared_names = (
                    ", ".join(e.relative for e in snapshot.declared_entries())
                    or "any declared file"
                )
                outcome.status = STATUS_BATCH_FAILED
                outcome.write_outcome = WRITE_CLEAN_FAILURE
                outcome.reason = (
                    f"FAILED (jail-denied edit): {denied_text}, and the "
                    f"pre-batch snapshot comparison confirms no byte changes in "
                    f"{declared_names}. The runner claimed "
                    f"{claimed or 'no status'!r}; a denied edit is never "
                    f"recorded as applied. {tail}Nothing to restore; the loop "
                    f"stops here and --resume re-dispatches this batch under "
                    f"the same grant."
                )
                self._jail_failed.add(outcome.task_id)
            return True
        notes = []
        if replaced:
            notes.append(
                f"{len(replaced)} replacement denial(s) the runner recovered "
                f"from in place ({', '.join(self._relative(p) for p in replaced)})"
            )
        if denials.out_of_scope_count:
            shown = ", ".join(
                self._relative(p) for p in denials.out_of_scope_paths
            )
            notes.append(
                f"{denials.out_of_scope_count} out-of-scope denial(s)"
                + (f" ({shown})" if shown else "")
            )
        if notes:
            self._jail_notes[outcome.task_id] = (
                f"The plugin-landlock jail refused {'; '.join(notes)} — none "
                f"of them changes the verdict."
            )
        return False

    def _jail_scope_sentence(
        self, in_grant: Sequence[str], outside: Sequence[str]
    ) -> str:
        """The reason's second half: why an in-grant path was refused, what to
        do instead, and F6 (the non-Codex runner)."""
        request = self.jail_request
        parts: List[str] = []

        def verb(paths: Sequence[str], one: str, many: str) -> str:
            return one if len(paths) == 1 else many

        if in_grant and request is not None and request.write_scope == WRITE_SCOPE_REPO_WIDE:
            parts.append(
                f"{', '.join(in_grant)} {verb(in_grant, 'lies', 'lie')} inside "
                f"the repository, which the jail granted whole (one rule on "
                f"its root, 2026-10-05): a denial there is the jail's own "
                f"failure, not a write shape the grant withholds."
            )
        elif in_grant:
            target = self._relative(request.target) if request and request.target else "the target"
            if request is not None and request.target_dir:
                parts.append(
                    f"{', '.join(in_grant)} {verb(in_grant, 'lies', 'lie')} "
                    f"inside this batch's grant (the "
                    f"tasks directory {self._relative(request.target_dir)}), so "
                    f"the allow-list may not grant the write shape the runner "
                    f"used (a new sibling file, a rename)."
                )
            else:
                parts.append(
                    f"{', '.join(in_grant)} {verb(in_grant, 'belongs', 'belong')} "
                    f"to this batch's "
                    f"{self.write_scope} grant, which admits in-place edits of "
                    f"{target} only: the allow-list does not grant the write "
                    f"shape the runner used (a new sibling file, a rename — "
                    f"sed -i, perl -i, a temp-file save, mv over it). Edit it in "
                    f"place with apply_patch (or a shell redirect); the grant "
                    f"never widens to its directory."
                )
        if outside:
            parts.append(
                f"{', '.join(outside)} {verb(outside, 'lies', 'lie')} outside "
                f"this batch's grant: the runner tried an edit its scope does "
                f"not grant."
            )
        parts.append(
            "To run this batch without the jail, set `driver.subagent_runner` "
            "to a runner that is not Codex (F6)."
        )
        return " ".join(parts) + " "

    def _audit_completion(
        self,
        outcome: TaskOutcome,
        runner: str,
        profile: str,
        attempt: int,
        exit_code: Optional[int],
        observed: Dict[str, Any],
        duration: float,
    ) -> None:
        self._audited.add(outcome.task_id)
        # Plan step 3: a jail-denial count that left the verdict alone still
        # goes on the outcome (and, through it, the report and the stamp).
        note = self._jail_notes.get(outcome.task_id)
        if note and note not in (outcome.reason or ""):
            outcome.reason = (
                f"{outcome.reason} {note}" if outcome.reason else note
            )
        # Every branch of `_classify` ends here, so this is the one place the
        # verdict reaches the result file that resume reads (F1).
        self._stamp_verdict(outcome)
        # The items, so a set-aside snapshot's record is findable by them
        # once its positional id belongs to another batch; a terminal record
        # (no `_classify`) reads them off the snapshot the dispatch took.
        expected_ids = outcome.expected_ids or (
            (self.snapshots.get(outcome.task_id).expected_ids or [])
            if self.snapshots.get(outcome.task_id) is not None else []
        )
        self.audit.completed(
            task_id=outcome.task_id, kind=self.kind, attempt=attempt,
            runner=runner, profile=profile, status=outcome.status,
            write_outcome=outcome.write_outcome, exit_code=exit_code,
            files_touched=outcome.files_touched, observed=observed,
            duration_seconds=duration, reason=outcome.reason,
            expected_ids=[str(i) for i in expected_ids],
        )

    def _stamp_verdict(self, outcome: TaskOutcome) -> None:
        """Write the classified verdict back onto the batch's result file (F1).

        ``persist_result`` runs BEFORE ``_classify``, so the file on disk
        carries the runner's own ``status`` and nothing else — and the claim
        is never the verdict. A runner that printed ``{"status": "applied",
        "ids": [all of them]}`` and then died mid-edit is a partial
        application: the loop stops at exit 3 promising "--resume
        re-dispatches this batch", and ``_already_done`` then read that same
        file — ``applied``, ids intersecting, a fingerprint recorded — and
        skipped the batch as done (measured 2026-09-16, for the crash and the
        reported-failure shapes both). The ``verdict`` key is additive (the
        result schema admits extra properties, so the file still validates)
        and its ``ok`` is the one bit resume needs: the same bit that decides
        the task's ``completed``/``failed`` status and whether an inputs
        binding is recorded, so the three records cannot disagree.
        """
        if not outcome.output_path:
            return
        path = Path(outcome.output_path)
        if not path.is_file():
            # `output_path` is where the result WOULD go, set before dispatch;
            # a dispatch that persisted nothing (a prose reply, a timeout, a
            # partial application) has no file to stamp, and the audit's
            # `completed` line and `state.json` carry its verdict instead. A
            # warning here named a file that was never meant to exist (fix
            # round 8).
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            data["verdict"] = {
                "write_outcome": outcome.write_outcome,
                "status": outcome.status,
                "ok": outcome.ok,
                "files_touched": outcome.files_touched,
                "sha256_before": outcome.sha256_before,
                "sha256_after": outcome.sha256_after,
                "reason": outcome.reason,
            }
            if outcome.verification_only:
                # The same two keys the report rows carry; a resume skip
                # reads them back so its row keeps the label.
                data["verdict"]["verification_only"] = True
                data["verdict"]["outcome_label"] = outcome.outcome_label
                if outcome.check_findings is not None:
                    # The bounded findings the label rests on, for a resume
                    # skip to carry (the reply's own `check` stays above).
                    data["verdict"]["check"] = check_record(outcome.check_findings)
                if (
                    (not outcome.ok and not outcome.files_touched and not outcome.not_rerun)
                    or (outcome.ok and outcome.check_findings)
                ):
                    # A failed check that moved nothing, or one that reported
                    # problems (fix round 11): what the tree looked like when
                    # it ran, so a `--resume` that changed nothing does not
                    # run it again (`_unchanged_since_failure`,
                    # `_check_findings_rerun`).
                    data["verdict"]["tree_fingerprint"] = self._tree_fingerprint()
            if outcome.runner_notes:
                data["verdict"]["runner_notes"] = outcome.runner_notes
            # The batch's identity beyond its positional name (2026-10-02):
            # the items it was materialized for and the inputs binding they
            # were materialized from, so a resume whose selection renumbered
            # the batch finds this result by its items (`_done_evidence`)
            # and still compares the inputs — the fingerprint file is keyed
            # by the positional id, which the renumbering reassigns.
            task = self._task_by_id(outcome.task_id)
            if task is not None:
                data["verdict"]["task_id"] = outcome.task_id
                data["verdict"]["expected_ids"] = [
                    str(i) for i in (task.get("expected_ids") or [])
                ]
                data["verdict"]["inputs_fingerprint"] = self._task_fingerprint(task)
            # Plan step 3 (2026-09-30): the verdict names the rung it was
            # reached on — additive, and only when there was one.
            rung = self._dispatch_rungs.get(outcome.task_id)
            if rung:
                data["verdict"]["sandbox_rung"] = rung
            temp = path.with_name(path.name + ".tmp")
            temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(str(temp), str(path))
        except (OSError, ValueError) as exc:  # noqa: BLE001 - bookkeeping must not fail a phase
            self.log(
                f"[write-subagents] warning: could not record the verdict on "
                f"{path.name} for {outcome.task_id}: {exc}"
            )

    # -- per-batch bookkeeping (D3n: append BEFORE the next substitution) ---

    def _record_batch_effects(
        self,
        task: Dict[str, Any],
        outcome: TaskOutcome,
        batch_index: int,
        result: SchedulerResult,
    ) -> None:
        """Append prior-changes, record metrics, update task status — in that order.

        The ordering is not cosmetic: the append must land before batch N+1's
        substitution reads the log, or the injected context lags a batch behind.

        The whole body is serialized. A co-scheduled group runs this from several
        threads against one in-memory ``StateManager``, one ``prior_changes.jsonl``
        and one ``state.json`` — all three read-modify-write — so without the lock
        two siblings can interleave and the later ``state.save()`` overwrites the
        earlier sibling's task status, leaving resume and both reports describing
        a run that did not happen. The work here is bookkeeping measured in
        milliseconds, so serializing it costs the batch nothing.
        """
        task_id = str(task.get("task_id", ""))
        entry_id = str(task.get("prior_changes_id") or task_id)

        if outcome.write_outcome == WRITE_APPLIED:
            summary = outcome.summary or f"batch {batch_index} applied"
        elif (outcome.write_outcome == WRITE_NO_OP and outcome.verification_only
              and outcome.check_findings):
            # A check that found problems: never "verified" on the line the
            # next batch and review-code's implementation record read.
            summary = (
                f"(check findings — {len(outcome.check_findings)} problem(s): "
                f"{'; '.join(outcome.check_findings)}; no file changed)"
            )
        elif outcome.write_outcome == WRITE_NO_OP and outcome.verification_only:
            # A check that ran and moved nothing: say it passed, in the word
            # every other surface uses for it.
            summary = (
                f"(verified — {outcome.summary or 'the check passed'}, "
                f"no file changed)"
            )
        elif outcome.write_outcome == WRITE_NO_OP:
            # The claim and the verdict agreed; say so, so the next batch reads
            # "already there" rather than "failed".
            summary = (
                f"(no-op — {outcome.summary or 'nothing needed changing'}, "
                f"verified no file changes)"
            )
        elif outcome.write_outcome == WRITE_CLEAN_FAILURE:
            # The documented clean-failure form, so a later batch reads why the
            # earlier one contributed nothing rather than assuming it succeeded.
            summary = (
                f"(batch failed — {outcome.summary or 'no changes reported'}, "
                f"verified no file changes)"
            )
        else:
            summary = None
        if task_id in self._jail_failed or outcome.not_rerun:
            # Plan step 3: a batch the jail's verdict failed is re-dispatched by
            # --resume, and `append` is idempotent on `(id, phase)` — a
            # "(batch failed — …)" line here would drop that run's real entry.
            # A not-re-run check keeps the line its failure already wrote.
            summary = None

        # Plan step 3 (2026-09-30): the rung the batch ran on, named on its
        # line and in the report whatever the outcome (a timeout included).
        rung = self._dispatch_rungs.get(task_id)
        if rung:
            record: Dict[str, Any] = {"sandbox_rung": rung}
            if task_id in self._jail_denials:
                record["jail_denials"] = self._jail_denials[task_id]
            result.sandbox_records[task_id] = record
        rung_text = f" [on {rung}]" if rung else ""
        note = self._jail_notes.get(task_id)
        if note:
            rung_text += f" {note}"

        with self._effects_lock:
            if summary:
                _prior_changes.append(
                    self.output_dir, entry_id, self.phase, summary[:400]
                )

            # Before the metric, so the `[ETA]` line this task drives carries its
            # own 1-based position in the run (F3). No metric for a check that
            # was not run again: no child ran, and a zero would average in.
            self._tasks_finished += 1
            if not outcome.not_rerun:
                self._record_metrics(task, outcome, result)

            if self.state is not None:
                status = "completed" if outcome.ok else "failed"
                reason = None if outcome.ok else (outcome.reason or outcome.status)
                self.state.update_task_status(task_id, status, reason=reason)
                if self.kind == "implement" and isinstance(
                    getattr(self.state, "state", None), dict
                ):
                    # What this run's check found, for `status` and review-code;
                    # any other outcome clears an earlier run's record.
                    set_task_check_findings_in_state(
                        self.state.state, task_id,
                        outcome.check_findings if outcome.ok else None,
                    )
                # Persists the metric recorded just above too: it is a mutation
                # of this object's dict now, not a second writer's file (F4).
                self.state.save()

            # One line per batch, on the happy path (F7). Every other `self.log`
            # in the loop is exceptional, so a 25-minute five-batch run left an
            # 11-line log — and `status` points the user at that log.
            verdict = outcome.outcome_label or outcome.status
            if outcome.not_rerun:
                verdict = "not run again (nothing changed since it failed)"
            elif outcome.ok and outcome.check_findings:
                verdict = (
                    f"{verdict} ({len(outcome.check_findings)} problem(s): "
                    f"{'; '.join(outcome.check_findings)[:300]})"
                )
            elif outcome.write_outcome and not outcome.ok:
                # Plan step 3: a failed batch whose write outcome reads
                # "clean-failure" says which failure it was.
                verdict = f"{verdict} ({outcome.status})"
            self.log(
                f"[write-subagents] batch {batch_index}: {task_id} "
                f"{verdict} in "
                f"{outcome.duration_seconds:.1f}s — "
                f"{', '.join(outcome.files_touched) or 'no files changed'}"
                f"{rung_text}"
            )

            if outcome.output_path and outcome.ok:
                # A result file only counts as this task's on resume when it was
                # produced from these inputs (D2g/D3i). The binding is written
                # here, under the same lock, because the file is one shared
                # read-modify-write map for the whole phase — and only for an
                # outcome that counts as done: a binding recorded for a failed
                # or partial batch attested that its optimistic result file was
                # current, which is half of how resume skipped it (F1).
                self._record_fingerprint(task)

    def _record_metrics(
        self, task: Dict[str, Any], outcome: TaskOutcome, result: SchedulerResult
    ) -> None:
        """The metric for one batch, with the values the runner reported (D3g).

        Duration is the executor's own wall clock. Tokens and tool uses come from
        the adapter's normalized usage where the provider reports them and are
        simply **omitted** where it does not — ``record_metric`` treats all three as
        optional, and a fabricated zero would average into per-phase totals and read
        as a free run.

        Recorded **through the ``StateManager``**, not through ``metrics.py``'s own
        file-level read/modify/write: the caller's ``state.save()`` follows
        immediately and dumps this object's dict wholesale, so a second writer's
        entry was erased the moment it landed — three audited runs finished with
        no ``metrics`` key at all after every batch had recorded one (F4).

        The ``[ETA]`` line's own ``batch_index``/``total_batches`` are this run's
        own progress and dispatchable count — never ``task["batch_index"]``
        (the plan's global decomposition index, which starts wherever a resumed
        run's first undone task sits) and never bare ``len(self.tasks)`` (which
        still counts a deferred human task). Passing those two straight through
        made every line on a resumed run read `batch 7/6 done in 0m 51s ...
        WARNING: batch_index > total_batches` (N1, 2026-09-16).

        **Both halves count tasks** (F3, 2026-09-17). ``len(result.batches_
        dispatched) + 1`` was the numerator, and it is a *batch* ordinal against
        a *task* denominator: for the three apply kinds one task is one batch and
        the two agree, but an ``implement`` dependency level with k independent
        tasks is one batch of k — every one of those k metric entries recorded
        the same ordinal, so a 14-task plan in 9 batches ended at
        ``batch 9/14 · ~Nm remaining``, never reached ``N/N`` and never printed
        ``ALL BATCHES COMPLETE``. ``self._tasks_finished`` is the count
        ``_mirror_status`` publishes as ``completed`` (resume skips included),
        and ``len(self.tasks) - len(result.human_plan.skipped)`` is the same
        denominator it publishes as ``total``, so the log and ``status`` now mean
        the same thing by the same number.
        """
        if self.state is None or not hasattr(self.state, "record_metric"):
            return
        usage = outcome.usage or {}
        label = (
            f"{task.get('task_id')}: "
            f"{task.get('task_title') or self.kind} "
            f"[{outcome.write_outcome or outcome.status}]"
        ).strip()
        total_batches = len(self.tasks) - len(result.human_plan.skipped)
        run_batch_index = self._tasks_finished
        try:
            eta_line = self.state.record_metric(
                self.phase,
                label,
                subagent_type=outcome.runner,
                token_count=(
                    int(usage["total_tokens"]) if isinstance(
                        usage.get("total_tokens"), (int, float)
                    ) else None
                ),
                tool_uses=(
                    int(usage["tool_uses"]) if isinstance(
                        usage.get("tool_uses"), (int, float)
                    ) else None
                ),
                duration_ms=int(outcome.duration_seconds * 1000),
                total_batches=total_batches or None,
                batch_index=run_batch_index,
            )
        except Exception as exc:  # noqa: BLE001 - metrics must not fail a phase
            self.log(f"[write-subagents] warning: metrics record failed: {exc}")
            return
        if eta_line:
            # The `[ETA]` line `instructions/implement.md` step 5 describes. It
            # goes through `self.log` rather than stderr so it lands in the
            # phase log `status` points at, beside the batch line above it.
            self.log(eta_line)

    # -- snapshots ------------------------------------------------------

    def _snapshot(self, task: Dict[str, Any]) -> TaskSnapshot:
        """Take (or reuse) this task's mandatory pre-batch snapshot (D3h).

        Mandatory, and refused when there is nothing to snapshot: a write task with
        no declared or snapshottable file gives the classification nothing to
        compare, which would make every outcome read as "clean failure".
        """
        task_id = str(task.get("task_id", ""))
        files = snapshot_file_set(task)
        if not files and self.kind == "implement":
            # A verification-only task: generate-tasks emits an agent task with
            # no files to create or modify when the work is a check ("Inline
            # module syntax gate", 2026-10-01, which this refusal used to dead-end
            # on at batch 12, with `--resume` hitting it again). The refusal's
            # reason does not apply to it: the byte comparison has nothing to
            # compare because the task is promised to change nothing, and the
            # undeclared-write sweep is then the whole verdict — see
            # `_verification_only` and `_verification_verdict`. Only `implement`
            # materializes such a task; the three apply kinds always declare
            # their one artifact, so an empty set there is still a bug.
            self.log(
                f"[write-subagents] {task_id}: declares no files, so it runs as "
                f"a verification-only task: any file it changes is a partial "
                f"application, not an edit."
            )
        elif not files:
            raise WriteSafetyError(
                f"Refusing to dispatch write task {task_id}: it declares no files "
                f"to snapshot, so the post-run byte comparison would have nothing "
                f"to compare and every outcome would read as a clean failure. "
                f"Populate `declared_files`/`snapshot_files` on the materialized "
                f"task (D3h). Nothing was dispatched."
            )
        expected_ids = [str(i) for i in (task.get("expected_ids") or [])]
        reused = (
            self._reusable_snapshot(task_id, files, expected_ids)
            if self.resume else None
        )
        if reused is not None:
            self.log(
                f"[write-subagents] {task_id}: reusing the pre-batch snapshot a "
                f"previous run took at {reused.directory} — the rollback point "
                f"stays the bytes from before this batch's FIRST dispatch"
            )
            self.snapshots[task_id] = reused
            self._snapshot_reused.add(task_id)
            return reused
        if not self.resume:
            # `--fresh` (and implement's plain run) reuses no snapshot — but a
            # slot an earlier run filled is that run's rollback point, and
            # `take` would copy today's bytes over it (adversarial review,
            # 2026-10-02: a `--fresh` on an un-reverted target left every
            # slot holding already-applied bytes). Set aside, never deleted.
            existing = TaskSnapshot.load(
                self.phase_dir, task_id, expected_root=self.repo_root,
            )
            if existing is not None:
                moved_to = existing.set_aside()
                self.log(
                    f"[write-subagents] {task_id}: an earlier run's pre-batch "
                    f"snapshot under that id is kept at {moved_to}; this run "
                    f"takes its own."
                )
        snapshot = TaskSnapshot.take(
            phase_dir=self.phase_dir, task_id=task_id, files=files,
            root=self.repo_root, expected_ids=expected_ids,
        )
        self.snapshots[task_id] = snapshot
        return snapshot

    def _reusable_snapshot(
        self, task_id: str, files: Sequence[str],
        expected_ids: Sequence[str] = (),
    ) -> Optional[TaskSnapshot]:
        """A previous run's snapshot for this batch, when it still applies.

        A batch reaching ``_snapshot`` under resume is one that has not
        succeeded yet (``_already_done`` skips the rest), and its half-applied
        edits are deliberately left on disk for a human to inspect. Retaking
        the snapshot over them would make *those* bytes the rollback point:
        measured 2026-09-15 across four resumed runs of one failed batch, each
        run re-snapshotted the previous run's partial edits, so ``util
        restore-batch`` could no longer reach the file as it was before the
        batch, and the "pre-batch snapshots are intact" line was untrue.

        Reused only when the manifest names the same declared set, byte for
        byte; a batch whose file set changed (the D2g resume guard should have
        refused it first) takes a fresh snapshot.

        And only when it was taken for the same items, the rule
        :func:`_same_batch_row` applies to report rows. An apply batch always
        declares its one artifact and its id is positional, so after a
        selection change the batch now holding ``apply-batch-1`` inherited the
        stopped ``apply-batch-1``'s rollback point, and ``util restore-batch``
        of the new batch also reverted the edits the stopped batch left in
        place (#28 follow-up). That snapshot is another batch's: it is set
        aside (renamed, never deleted) and this batch takes its own. A
        manifest that records no items (written before they were recorded)
        falls back to the id, as before.
        """
        existing = TaskSnapshot.load(
            self.phase_dir, task_id, expected_root=self.repo_root,
        )
        # The same cutoff `_earlier_rows` applies: a snapshot taken before
        # the review this phase applies completed belongs to the previous
        # cycle — the bytes before THAT cycle's batch, so restoring from it
        # would roll that cycle's applied edit back too (follow-up to the
        # 2026-10-02 adversarial review). Set aside, never deleted.
        reviewed_at = self._review_completed_at()
        if reviewed_at is not None:
            # The manifest is stamped to the second.
            reviewed_at = reviewed_at.replace(microsecond=0)
        if existing is not None and existing.taken_before(reviewed_at):
            moved_to = existing.set_aside()
            self.log(
                f"[write-subagents] {task_id}: the pre-batch snapshot under "
                f"that id was taken at {existing.taken_at or 'an unknown time'}, "
                f"before this phase's review completed "
                f"({reviewed_at.isoformat(timespec='seconds')}), so it is a "
                f"previous review cycle's rollback point; kept at {moved_to}."
            )
            existing = None
        recorded_ids = (existing.expected_ids or []) if existing else []
        if existing is not None and recorded_ids and expected_ids and (
            sorted(recorded_ids) != sorted(str(i) for i in expected_ids)
        ):
            moved_to = existing.set_aside()
            self.log(
                f"[write-subagents] {task_id}: the pre-batch snapshot under "
                f"that id was taken for other items (an earlier run's batch); "
                f"kept at {moved_to}."
            )
            existing = None
        if existing is None and expected_ids:
            # The items may have held this id before and been set aside when
            # the id went to others (B deselected, C takes `apply-batch-1`,
            # B re-selected): their own pre-first-dispatch snapshot comes
            # back into the slot rather than a fresh one being taken over
            # every run's edits since.
            existing = TaskSnapshot.reinstate(
                self.phase_dir, task_id, [str(i) for i in expected_ids],
                not_before=reviewed_at,
            )
            if existing is not None:
                self.log(
                    f"[write-subagents] {task_id}: the snapshot an earlier run "
                    f"took for these same items was set aside when the id "
                    f"went to other items; moved back to {existing.directory}."
                )
        if existing is None:
            return None
        try:
            wanted = sorted(
                relative_to_root(str(contain_under_root(
                    candidate, self.repo_root, what="snapshot target"
                )), self.repo_root)
                for candidate in files
            )
        except WriteSafetyError:  # a path outside the root: `take` refuses it
            return None
        recorded = sorted(e.relative for e in existing.declared_entries())
        if wanted != recorded:
            return None
        return existing

    # -- resume's inputs binding (D2g/D3i) ------------------------------

    def _task_fingerprint(self, task: Dict[str, Any]) -> str:
        """The D2g binding for one materialized write task, as one hash.

        Over the **materialized task**, deliberately not over its ``inputs``
        block. D2g's hashes name upstream artifacts, and a write phase *writes*
        the artifacts its own hashes cover: ``apply-suggestions`` rewrites the
        plan ``inputs.plan_sha256`` binds to, ``apply-code-fixes`` invalidates
        the ``inputs.diff_sha256`` its own edits were derived from, and
        ``implement`` inserts the summary reference into the plan. Comparing
        those on resume would make a *successful* batch look like changed
        inputs and re-dispatch it — applying it twice, which is the failure the
        resume guard exists to prevent.

        The prompt, the expected ids and the declared set are the derived data
        that actually moves when the plan, the suggestions or the batching
        changed: under ``context_mode: inline`` the payload travels *inside*
        ``prompt``, so an edited suggestion changes the prompt directly.
        ``prompt`` is the **raw** prompt — ``{prior_changes_context}`` is bound
        late and per dispatch — so the text hashed here is stable across runs
        of the same inputs.
        """
        payload = json.dumps(
            {
                "kind": self.kind,
                "prompt": str(task.get("prompt") or ""),
                "expected_ids": [str(i) for i in (task.get("expected_ids") or [])],
                "declared_files": sorted(declared_file_set(task)),
                "batch_index": int(task.get("batch_index", 0)),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _fingerprint_path(self) -> Path:
        return Path(self.phase_dir) / WRITE_FINGERPRINT_FILENAME

    def _load_fingerprints(self) -> Dict[str, str]:
        try:
            data = json.loads(self._fingerprint_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}
        recorded = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(recorded, dict):
            return {}
        return {str(key): str(value) for key, value in recorded.items()}

    def _record_fingerprint(self, task: Dict[str, Any]) -> None:
        """Persist one task's inputs binding beside its result file."""
        recorded = self._load_fingerprints()
        recorded[str(task.get("task_id", ""))] = self._task_fingerprint(task)
        path = self._fingerprint_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".tmp")
            temp.write_text(json.dumps({
                "fingerprint_version": 1,
                "kind": self.kind,
                "tasks": recorded,
            }, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(str(temp), str(path))
        except OSError as exc:  # noqa: BLE001 - bookkeeping must not fail a phase
            self.log(
                f"[write-subagents] warning: could not record the inputs binding "
                f"for {task.get('task_id')}: {exc}"
            )

    @staticmethod
    def _recorded_files_touched(output_path: str) -> List[str]:
        """``files_touched`` as the batch's result file on disk records it."""
        try:
            data = json.loads(Path(output_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        raw = data.get("files_touched") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        seen: List[str] = []
        for item in raw:
            text = str(item).strip()
            if text and text not in seen:
                seen.append(text)
        return seen

    def _task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        for task in self.tasks:
            if str(task.get("task_id", "")) == task_id:
                return task
        return None

    def _sibling_results(
        self, task: Dict[str, Any]
    ) -> List[Tuple[Path, str, int]]:
        """Other result files of this kind that attest to this task's items.

        An apply kind's ids and result names are positional, so a resume
        whose selection changed can hand a batch that already landed as
        ``apply-batch-1`` the id ``apply-batch-2``, with no
        ``apply_batch_2.json`` to find — and re-apply it (adversarial review,
        2026-10-02). The verdict stamp names the items a result was produced
        for; a file naming exactly this task's items is this batch's, under
        the id and index it ran as. ``implement`` keys results by the plan
        task's own id and has no such case.
        """
        if self.kind == "implement":
            return []
        expected = sorted(str(i) for i in (task.get("expected_ids") or []))
        if not expected:
            return []
        try:
            own = resolve_output_target(task, self.phase_dir)
        except ResultRejected:
            own = None
        template = write_output_name(self.kind, 0)
        prefix, suffix = template.split("0", 1)
        found: List[Tuple[Path, str, int]] = []
        for path in sorted(Path(self.phase_dir).glob(f"{prefix}*{suffix}")):
            token = path.name[len(prefix):len(path.name) - len(suffix)]
            index = _RESULT_TOKEN.match(token)
            if index is None or (own is not None and path == own):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            verdict = data.get("verdict") if isinstance(data, dict) else None
            if not isinstance(verdict, dict):
                continue
            recorded = verdict.get("expected_ids")
            if not isinstance(recorded, list) or sorted(map(str, recorded)) != expected:
                continue
            recorded_id = str(verdict.get("task_id") or "")
            if not recorded_id:
                continue
            found.append((path, recorded_id, int(index.group(1))))
        return found

    def _retire_foreign_result(self, task: Dict[str, Any]) -> None:
        """Move another batch's result out from under this task's positional name.

        A result file is named by batch index, so a resume whose selection
        renumbered the batches dispatches B as ``apply-batch-1`` over the
        ``apply_batch_1.json`` C landed under that id in the earlier run —
        and C's only evidence of being done was gone before its own turn
        came (adversarial review, 2026-10-02). A result whose verdict names
        other items is kept beside the slot as ``<stem>.items-<sha8>.json``
        (``-2``, ``-3`` … when taken), where :meth:`_sibling_results` reads
        it; a result for the same items is overwritten as before.
        """
        if self.kind == "implement":
            return
        try:
            target = resolve_output_target(task, self.phase_dir)
        except ResultRejected:
            return
        if not target.is_file():
            return
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        verdict = data.get("verdict") if isinstance(data, dict) else None
        recorded = verdict.get("expected_ids") if isinstance(verdict, dict) else None
        if not isinstance(recorded, list):
            return
        expected = sorted(str(i) for i in (task.get("expected_ids") or []))
        if sorted(map(str, recorded)) == expected:
            return
        stem = target.name[:-len(".json")]
        base = f"{stem}.items-{TaskSnapshot.items_digest(recorded)}"
        kept = target.with_name(f"{base}.json")
        counter = 2
        while kept.exists():
            kept = target.with_name(f"{base}-{counter}.json")
            counter += 1
        try:
            os.replace(str(target), str(kept))
        except OSError as exc:  # noqa: BLE001 - bookkeeping must not fail a phase
            self.log(
                f"[write-subagents] warning: could not keep {target.name} "
                f"(another batch's result) aside before {task.get('task_id')} "
                f"reuses the name: {exc}"
            )
            return
        self.log(
            f"[write-subagents] {task.get('task_id')}: {target.name} holds the "
            f"result of other items (an earlier run's batch under this id); "
            f"kept at {kept.name}."
        )

    def _already_done(self, task: Dict[str, Any]) -> bool:
        """Whether :meth:`_done_evidence` finds a result to resume from."""
        return self._done_evidence(task) is not None

    def _done_evidence(
        self, task: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """The result file and the id it was produced under, when the task is done.

        The task's own positional result first; for an apply kind, a sibling
        result naming the same items next (:meth:`_sibling_results`), judged
        by the same clauses with the id and batch index it ran as — the
        audit, the verdict and the inputs binding are read for THAT batch,
        never for whatever this run put under the renumbered id.
        """
        task_id = str(task.get("task_id", ""))
        try:
            own = resolve_output_target(task, self.phase_dir)
        except ResultRejected:
            own = None
        if own is not None and self._result_counts_as_done(
            task, own, task_id, int(task.get("batch_index", 0)), sibling=False,
        ):
            return str(own), task_id
        for path, recorded_id, index in self._sibling_results(task):
            if self._result_counts_as_done(
                task, path, recorded_id, index, sibling=True,
            ):
                self.log(
                    f"[write-subagents] {task_id}: its items already landed as "
                    f"{recorded_id} ({path.name}) in an earlier run of this "
                    f"phase — the batch was renumbered by a selection change, "
                    f"not re-applied."
                )
                return str(path), recorded_id
        return None

    def _result_counts_as_done(
        self, task: Dict[str, Any], target: Path, task_id: str,
        batch_index: int, *, sibling: bool,
    ) -> bool:
        """Resume's definition of done (D3i), all five clauses of it.

        ``task_id``/``batch_index`` are the ones the result was produced
        under (the task's own, or a renumbered sibling's); ``sibling`` says
        the file was found by its items, in which case the clauses keyed by
        positional id alone (``state.json``'s task status, the fingerprint
        file) are read from the verdict stamp instead, since this run may
        already have put another batch under that id.

        The result file must exist, parse and validate — **and** the task it was
        produced from must not have moved (D2g). The bridge rewrites
        ``write_tasks.json`` before the scheduler runs, so the previous run's own
        hashes are gone by the time resume asks; the binding therefore lives
        beside the results in ``.write_task_inputs.json`` and is compared against
        the *freshly materialized* task (see :meth:`_task_fingerprint`). A
        recorded binding that moved means the plan, the suggestions or the
        batching changed between the partial run and the resume, and the stale
        batch result is re-dispatched rather than silently accepted as current.

        A task with **no** recorded binding keeps the previous behaviour: absence
        of evidence that the inputs moved is not evidence that they did, and
        re-dispatching an apply batch that already landed would apply it twice.

        The fifth clause reads the file's **own** ``ids``: a result that attests
        to none of this task's ``expected_ids`` was produced for a different
        task, whatever its name says. Measured 2026-09-16 — a resumed implement
        phase re-pointed T006 at an ``implement_batch_0.json`` whose ``ids`` said
        ``["T001"]``, and because T006 had never finished it had no recorded
        binding either, so the fourth clause had nothing to compare and five
        never-dispatched tasks were declared complete. The evidence was on disk
        and simply never read. It is a *defence in depth* beside the naming fix
        in ``subagent_tasks.implement_output_name``: neither alone is why the
        skip is safe.
        """
        if not target.is_file() or target.stat().st_size == 0:
            return False
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        if validate_result(self.kind, data):
            return False
        if str(data.get("status") or "") not in (WRITE_APPLIED, WRITE_NO_OP):
            # A schema-valid `{"status": "failed"}` is a batch the previous run
            # recorded as NOT done; resuming past it on the strength of the
            # file existing would skip exactly the batch --resume is for.
            return False
        verdict = data.get("verdict")
        if isinstance(verdict, dict) and not verdict.get("ok", True):
            # The scheduler's own classification, stamped by `_stamp_verdict`
            # once `_classify` decided (F1): the file reads `applied` because
            # that is what the runner printed before it died, and the loop
            # stopped on it promising a re-dispatch.
            self.log(
                f"[write-subagents] {task_id}: {target.name} claims "
                f"{data.get('status')!r} but the previous run classified it "
                f"{verdict.get('write_outcome')} — re-dispatching it rather "
                f"than resuming it (F1)."
            )
            return False
        if sibling and not isinstance(verdict, dict):
            return False
        if (
            not sibling
            and self.state is not None
            and hasattr(self.state, "get_task_status")
        ):
            # The same fact from `state.json`, for a result file that predates
            # the verdict stamp: `_record_batch_effects` wrote `failed` for
            # every outcome that was not ok, and that record outlives the
            # optimistic claim in the file.
            try:
                recorded_status = self.state.get_task_status(task_id)
            except Exception:  # noqa: BLE001 - advisory, never fatal
                recorded_status = None
            if recorded_status == "failed":
                self.log(
                    f"[write-subagents] {task_id}: state.json records it "
                    f"`failed` although {target.name} claims "
                    f"{data.get('status')!r} — re-dispatching it rather than "
                    f"resuming it (F1)."
                )
                return False
        # A crash can occur after persist_result but before classification (or
        # before a new result replaces an older successful one). The last
        # dispatch then has no completion: neither file is evidence of success.
        # Un-audited legacy/native results retain their compatibility behavior.
        expected = [str(item) for item in (task.get("expected_ids") or [])]
        if sibling:
            # Found by its verdict stamp, which only `_classify` writes, so
            # the crash-before-classification shape cannot apply; the audit's
            # own record for these items has to agree it succeeded.
            recorded = self.audit.completion_for(task_id, expected_ids=expected)
            if not recorded or recorded.get("status") != STATUS_OK:
                return False
        else:
            latest = next((
                row for row in reversed(self.audit.records())
                if row.get("task_id") == task_id
                and row.get("event") in ("dispatched", "completed")
            ), None)
            if latest and (
                latest.get("event") == "dispatched"
                or latest.get("status") != STATUS_OK
            ):
                self.log(
                    f"[write-subagents] {task_id}: the last dispatch has no "
                    "successful classification — re-dispatching rather than "
                    "trusting its result."
                )
                return False
        returned = [
            str(item) for item in (data.get("ids") or [])
            if isinstance(data.get("ids"), list)
        ]
        if expected and returned and not (set(expected) & set(returned)):
            self.log(
                f"[write-subagents] {task_id}: {target.name} attests to "
                f"{', '.join(sorted(returned))} and not to "
                f"{', '.join(expected)} — it is another task's result, so this "
                f"one is re-dispatched rather than resumed (D3i)."
            )
            return False
        if sibling:
            recorded_fp = str(verdict.get("inputs_fingerprint") or "")
            current_fp = self._task_fingerprint({**task, "batch_index": batch_index})
        else:
            recorded_fp = self._load_fingerprints().get(task_id)
            current_fp = self._task_fingerprint(task)
        if recorded_fp and recorded_fp != current_fp:
            self.log(
                f"[write-subagents] {task_id}: the inputs it was materialized "
                f"from have changed since {target.name} was written — "
                f"re-dispatching it instead of resuming it (D2g)."
            )
            return False
        return True

    # -- human plan persistence -----------------------------------------

    def _persist_human_plan(self, plan: HumanTaskPlan) -> None:
        if self.state is None:
            return
        persist_human_task_plan(self.state, plan)

    def _pending_human_message(self, plan: HumanTaskPlan) -> List[str]:
        lines = [
            "",
            f"ERROR: {len(plan.pending)} human task(s) need an answer before this "
            f"phase can continue, and an unattended run may not decide for you.",
        ]
        for task_id in plan.pending:
            lines.append(f"  - {task_id}: answer with one of")
            lines.append(f"      --human-task {task_id}=done")
            lines.append(f"      --human-task {task_id}=skip")
        if not plan.strategy:
            lines.append(
                "  (or resolve the whole phase at once with --force-strategy "
                f"{'|'.join(STRATEGIES)})"
            )
        lines.append(
            "Nothing was dispatched, no file was modified, and the phase is not "
            "marked completed."
        )
        return lines

    # -- stop / finalize ------------------------------------------------

    def _stop_message(
        self, batch_index: int, outcomes: Sequence[TaskOutcome],
        result: Optional[SchedulerResult] = None,
    ) -> List[str]:
        """The exit-3 block after a failed batch, in plain words.

        Names the task that failed and what it reported, every task the stop
        left unrun, the tasks that ran, and a way forward that works (racing
        re-run 3, 2026-10-05: a failed read-only check stopped implement, the
        message named only the deferred manual tasks, and ``--resume`` re-ran
        the same check on unchanged code). The commands are the host's own
        spelling when the orchestrator supplied them (``user_commands``), the
        bare flags otherwise.
        """
        failed = [o for o in outcomes if not o.ok]
        stop = (result.stop if result is not None else None) or {}
        lines = [""]
        for outcome in failed:
            what = self._failure_sentence(outcome)
            lines.append(
                f"ERROR: {self.phase} stopped at {outcome.task_id}"
                f"{self._task_label(outcome.task_id)}: {what}"
            )
            if outcome.runner_notes:
                lines.append(f"      what it reported: {outcome.runner_notes}")
            lines.append(f"      details: {outcome.reason or outcome.status}")
            if outcome.snapshot_dir and outcome.files_touched:
                lines.append(
                    f"      restore: multi_llm.py util restore-batch --phase-dir "
                    f'"{self.phase_dir}" --task-id {outcome.task_id}'
                )
        if not failed:
            lines.append(
                f"ERROR: the write loop stopped at batch {batch_index}; no later "
                f"batch was dispatched and the {self.phase} phase is NOT marked "
                f"completed."
            )
        lines.extend(self._account_lines(result))
        lines.extend(self._way_forward_lines(
            failed, stop, plan=result.human_plan if result is not None else None,
        ))
        lines += [
            f"Audit log: {self.audit.path}",
            f"Snapshots: {os.path.join(self.safety_dir, 'pre_batch')}/",
            "The half-applied state is deliberately left in place: what the "
            "stop left on disk stays until you restore the batch or resume.",
        ]
        return lines

    def _task_label(self, task_id: str) -> str:
        """`` (check: <title>)`` for a verification-only task, `` (<title>)`` otherwise."""
        task = self._task_by_id(task_id) or {}
        title = str(task.get("task_title") or "").strip()
        check = self.kind == "implement" and not declared_file_set(task)
        if check and title:
            return f" (check: {title})"
        if check:
            return " (a check)"
        return f" ({title})" if title else ""

    def _failure_sentence(self, outcome: TaskOutcome) -> str:
        """What failed, in one clause: the runner's own summary when it gave one."""
        summary = (outcome.summary or "").strip()
        if outcome.not_rerun:
            return (
                "not run again — nothing in the repository changed since this "
                "check failed, so it would fail the same way"
                + (f" ({summary})" if summary else "")
            )
        check = self._task_by_id(outcome.task_id) or {}
        is_check = self.kind == "implement" and not declared_file_set(check)
        label = outcome.outcome_label or outcome.write_outcome or outcome.status
        if is_check and outcome.write_outcome == WRITE_CLEAN_FAILURE:
            return f"the check failed" + (f" — {summary}" if summary else "")
        if outcome.write_outcome == WRITE_PARTIAL:
            return (
                f"partial application (bytes moved, the batch did not finish)"
                + (f" — {summary}" if summary else "")
            )
        return f"{label}" + (f" — {summary}" if summary else "")

    def _account_lines(self, result: Optional[SchedulerResult]) -> List[str]:
        """Every task of the decomposition, by what became of it."""
        if result is None:
            return []
        account = self._accounting(result)
        lines = []
        done = account["done"]
        if done:
            lines.append(
                f"Done and kept: {len(done)} task(s) ({_id_range(done)})."
            )
        if account.get("check_findings"):
            lines.append(
                f"Of those, checks that reported problems (listed in the "
                f"implementation summary): {', '.join(account['check_findings'])}."
            )
        if account["not_run"]:
            blocked = account.get("blocked") or {}
            for check_id in sorted(set(blocked.values())):
                waiting = [t for t in account["not_run"] if blocked.get(t) == check_id]
                lines.append(
                    f"Not run (they depend on check {check_id}, which reported "
                    f"problems): {', '.join(waiting)}."
                )
            stopped = [t for t in account["not_run"] if t not in blocked]
            if stopped:
                lines.append(
                    f"Not run (the stop came before they were dispatched): "
                    f"{', '.join(stopped)}."
                )
        with_findings = account.get("accepted_with_findings") or []
        after_failing = [t for t in account["accepted"] if t not in with_findings]
        if after_failing:
            lines.append(
                f"Accepted after failing (--human-task <id>=done): "
                f"{', '.join(after_failing)}."
            )
        if with_findings:
            lines.append(
                f"Accepted with their check findings (--human-task <id>=done): "
                f"{', '.join(with_findings)}."
            )
        if account["completed_manual"]:
            lines.append(
                f"Completed by hand (--human-task <id>=done): "
                f"{', '.join(account['completed_manual'])}."
            )
        if account["declined"]:
            lines.append(
                f"Skipped at your request (--human-task <id>=skip): "
                f"{', '.join(account['declined'])}."
            )
        if account["skipped_dependents"]:
            lines.append(
                f"Skipped (depend on a skipped task): "
                f"{', '.join(account['skipped_dependents'])}."
            )
        if account["deferred"]:
            lines.append(f"Deferred manual: {', '.join(account['deferred'])}.")
        lines.append(
            f"That is every one of the {account['total']} task(s) in this "
            f"decomposition; the {self.phase} phase is NOT marked completed."
        )
        return lines

    def _way_forward_lines(
        self, failed: Sequence[TaskOutcome], stop: Dict[str, Any],
        plan: Optional[HumanTaskPlan] = None,
    ) -> List[str]:
        """The remedy, with the host's own command spellings when known.

        The dependents a ``skip`` answer would leave undone are the ones a
        skip can still change: a manual task (never dispatched) and a task
        the run already skipped are not listed, so the sentence does not name
        a deferred manual task as if the answer decided it.
        """
        cmds = self.user_commands
        already = set(plan.skipped) if plan is not None else set()
        manual = {
            str(t.get("task_id", "")) for t in self.tasks
            if t.get("executor_role") == "human" or t.get("is_human")
        }
        resume = cmds.get("resume") or "re-invoke this phase with --resume"
        lines = ["To go on:"]
        acceptable = [
            o for o in failed
            if o.write_outcome == WRITE_CLEAN_FAILURE and not o.files_touched
        ]
        checks = [
            o for o in acceptable
            if self.kind == "implement"
            and not declared_file_set(self._task_by_id(o.task_id) or {})
        ]
        if checks:
            lines.append(
                "  - fix what the check reported in the code — by hand, or run "
                "the review-code phase and then apply-code-fixes"
                + (":" if cmds.get("review") and cmds.get("fix") else "")
            )
            if cmds.get("review") and cmds.get("fix"):
                lines.append(f"        {cmds['review']}")
                lines.append(f"        {cmds['fix']}")
            lines.append(
                "    then resume this phase; the check runs again and the "
                "tasks after it follow:"
            )
            lines.append(f"        {resume}")
        elif acceptable:
            lines.append(
                "  - fix what the task reported, then resume this phase; it is "
                "dispatched again and the tasks after it follow:"
            )
            lines.append(f"        {resume}")
        else:
            lines.append(
                "  - inspect the half-applied state (restore the batch with the "
                "command above, or keep it), fix the cause, then resume this "
                "phase; the failed batch is dispatched again and the ones never "
                "reached follow:"
            )
            lines.append(f"        {resume}")
        for outcome in acceptable:
            task_id = outcome.task_id
            dependents = [
                dep for dep in self._dependents_of(task_id)
                if dep not in already and dep not in manual
                and self._task_by_id(dep) is not None
            ]
            after = sorted(stop.get("not_run") or [])
            accept = cmds.get("accept", "").replace("{task_id}", task_id) or (
                f"add --human-task {task_id}=done to the resume command"
            )
            skip = cmds.get("skip", "").replace("{task_id}", task_id) or (
                f"add --human-task {task_id}=skip to the resume command"
            )
            lines.append(
                f"  - or accept the failure and go on without it: {accept} "
                f"({task_id} is treated as satisfied"
                + (f"; {', '.join(after)} then run" if after else "")
                + f"), or {skip} ({task_id}"
                + (f" and the tasks that depend on it, {', '.join(dependents)},"
                   if dependents else "")
                + " stay undone)."
            )
            if outcome.not_rerun:
                retry = cmds.get("retry", "").replace("{task_id}", task_id) or (
                    f"add --human-task {task_id}=retry to the resume command"
                )
                lines.append(
                    f"  - or run the check again anyway (after installing a tool "
                    f"it needs, say): {retry}."
                )
        return lines

    def _dependents_of(self, task_id: str) -> List[str]:
        """Every task that depends on *task_id*, transitively, in id order.

        Over the whole decomposition (:func:`decomposition_graph`), not only
        this run's tasks: a resume does not materialize a finished task, and
        a walk over the rest stopped at it (fix round 12). A task this run
        does not hold is in the answer; callers that act on a task skip it.
        """
        if self.kind != "implement":
            return transitive_dependents(self.tasks, task_id)
        return transitive_dependents(
            decomposition_graph(self.tasks, self.state), task_id,
        )

    def _accounting(self, result: SchedulerResult) -> Dict[str, Any]:
        """Every task of the decomposition in exactly one bucket.

        ``done`` is this run's ``ok`` outcomes (resume skips included) plus the
        earlier runs' ``ok`` rows; ``failed`` this run's failed outcomes;
        ``not_run`` the tasks the stop left undispatched; ``accepted`` the
        failed tasks answered ``done``; ``completed_manual`` the human tasks
        answered ``done`` (fix round 8: they counted in ``total`` and sat in
        no bucket, so a five-task line summed to four); ``deferred`` the human
        tasks the strategy skipped and ``skipped_dependents`` the tasks
        skipped with a skipped one. The buckets add up to ``total``, which is
        the test ``summary.md``'s accounting line is written to. ``accepted``
        and ``completed_manual`` include the answers an earlier run recorded
        for tasks this resume did not materialize (``answered_done_on_record``,
        fix round 9).
        """
        rows = result.report_rows()
        plan = result.human_plan
        by_id = {str(t.get("task_id", "")): t for t in self.tasks}
        failed = sorted({o.task_id for o in result.outcomes if not o.ok})
        answered = set(plan.accepted) | set(plan.completed)
        done = sorted({
            str(row.get("task_id"))
            for row in rows
            if row.get("status") == STATUS_OK and str(row.get("task_id")) not in failed
            and str(row.get("task_id")) not in answered
        })
        # A skip the user chose (a pause-and-ask "Skip", `--human-task
        # <id>=skip` on a failed task) is its own bucket: the seed is neither
        # deferred work nor "skipped with a skipped task".
        declined = sorted(
            tid for tid, reason in plan.skipped.items()
            if reason.startswith((USER_DECLINED_REASON, ANSWERED_SKIP_PREFIX))
        )
        deferred = sorted(
            tid for tid in plan.skipped
            if tid not in declined
            and ((by_id.get(tid) or {}).get("executor_role") == "human"
                 or (by_id.get(tid) or {}).get("is_human"))
        )
        skipped_dependents = sorted(
            tid for tid in plan.skipped if tid not in deferred and tid not in declined
        )
        accepted = sorted(plan.accepted)
        completed_manual = sorted(plan.completed)
        not_run = sorted(result.not_run)
        ids = set(done) | set(failed) | set(deferred) | set(skipped_dependents) \
            | set(declined) | set(accepted) | set(completed_manual) | set(not_run) \
            | set(by_id)
        check_findings = sorted({
            str(row.get("task_id"))
            for row in rows
            if str(row.get("task_id")) in done and row_check_findings(row)
        })
        return {
            "total": len(ids),
            "done": done,
            "check_findings": check_findings,
            "failed": failed,
            "not_run": not_run,
            "blocked": {
                tid: cid for tid, cid in result.blocked_by_check.items()
                if tid in result.not_run
            },
            "accepted_with_findings": sorted(
                tid for tid in accepted
                if ACCEPTED_WITH_FINDINGS_MARK in str(plan.accepted.get(tid) or "")
            ),
            "accepted": accepted,
            "completed_manual": completed_manual,
            "declined": declined,
            "deferred": deferred,
            "skipped_dependents": skipped_dependents,
        }

    def _record_stop(
        self, result: SchedulerResult, batch_index: int,
        failed: Sequence[TaskOutcome], *, refusal: Optional[str] = None,
    ) -> None:
        """Account for every task the stop left unrun, and keep the stop record.

        Called on both stop paths before the message is built: every
        materialized task that is not human, not skipped, not accepted and has
        no outcome in this run was never dispatched. Each gets a reason on its
        ``pending`` status (``task_status_reasons``), so the implementation
        record review-code reads says why it is pending rather than leaving it
        indistinguishable from a task nobody has looked at.
        """
        failed_ids = [o.task_id for o in failed]
        seen = {o.task_id for o in result.outcomes}
        plan = result.human_plan
        cause = (
            f"{', '.join(failed_ids)} failed" if failed_ids
            else (f"batch {batch_index} was refused" if refusal else
                  f"the loop stopped at batch {batch_index}")
        )
        # A task a check with findings already kept from running keeps that
        # reason: the stop is not why it did not run (fix round 11).
        for task_id, check_id in result.blocked_by_check.items():
            result.not_run[task_id] = blocked_by_check_reason(check_id)
        for task in self.tasks:
            task_id = str(task.get("task_id", ""))
            if (not task_id or task_id in seen or plan.is_skipped(task_id)
                    or task_id in plan.accepted
                    or task_id in result.not_run
                    or task.get("executor_role") == "human"):
                continue
            result.not_run[task_id] = (
                f"not run — {self.phase} stopped at batch {batch_index} "
                f"({cause}) before this task was dispatched"
            )
        self._record_not_run_in_state(result)
        self._restamp_check_trees(result)
        first = failed[0] if failed else None
        is_check = bool(
            first is not None and self.kind == "implement"
            and not declared_file_set(self._task_by_id(first.task_id) or {})
        )
        acceptable = bool(
            first is not None and first.write_outcome == WRITE_CLEAN_FAILURE
            and not first.files_touched
        )
        result.stop = {
            "batch_index": batch_index,
            "failed_task_ids": failed_ids,
            "task_id": first.task_id if first else None,
            "kind": "check" if is_check else ("task" if first else "refusal"),
            "what": (
                self._failure_sentence(first) if first else (refusal or "")
            ),
            "reported": (first.runner_notes if first else None),
            "not_run": sorted(result.not_run),
            "acceptable": acceptable,
            "not_rerun": bool(first is not None and first.not_rerun),
        }
        result.stop["headline"] = self._stop_headline(result.stop)

    def _record_not_run_in_state(self, result: SchedulerResult) -> None:
        """Give each ``result.not_run`` task its reason on a ``pending`` status.

        Only a task with no status or a ``pending`` one: a status an earlier
        run recorded (completed, failed, skipped) is never overwritten.
        """
        if self.state is None or not result.not_run:
            return
        with self._effects_lock:
            for task_id, reason in result.not_run.items():
                current = None
                try:
                    current = self.state.get_task_status(task_id)
                except Exception:  # noqa: BLE001 - advisory
                    current = None
                if current in (None, "pending"):
                    self.state.update_task_status(task_id, "pending", reason=reason)
            try:
                self.state.save()
            except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a phase
                self.log(f"[write-subagents] warning: could not record the not-run tasks: {exc}")

    def _restamp_check_trees(self, result: SchedulerResult) -> None:
        """Stamp the tree as this run left it on each check with findings.

        ``_stamp_verdict`` records the tree when the check ran, and the
        tasks that do not depend on it go on and edit files after it — so a
        ``--resume`` the user changed nothing for would read "the files
        changed" and run the check again. The run's own edits are not a
        change the user made: when the run ends needing a human, each
        check-findings verdict this run holds is re-stamped with the tree as
        the run leaves it, and a resume runs the check again only when
        something moved since (fix round 11). Best-effort.
        """
        if self.kind != "implement":
            return
        current: Optional[str] = None
        for outcome in result.outcomes:
            if not (outcome.ok and outcome.check_findings and outcome.output_path):
                continue
            path = Path(outcome.output_path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                verdict = data.get("verdict") if isinstance(data, dict) else None
                if not isinstance(verdict, dict):
                    continue
                if current is None:
                    current = self._tree_fingerprint()
                    if current is None:
                        return
                if verdict.get("tree_fingerprint") == current:
                    continue
                verdict["tree_fingerprint"] = current
                temp = path.with_name(path.name + ".tmp")
                temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                os.replace(str(temp), str(path))
            except (OSError, ValueError) as exc:  # noqa: BLE001 - bookkeeping
                self.log(
                    f"[write-subagents] warning: could not re-stamp the tree on "
                    f"{path.name} for {outcome.task_id}: {exc}"
                )

    def _blocking_checks(self, result: SchedulerResult) -> List[str]:
        """The check tasks that kept a task from running, in task order."""
        blocking = set(result.blocked_by_check.values())
        order = [str(t.get("task_id", "")) for t in self.tasks]
        return sorted(
            blocking, key=lambda tid: (order.index(tid) if tid in order else len(order), tid),
        )

    def _record_check_block(self, result: SchedulerResult) -> None:
        """Account for the tasks a check with findings kept from running.

        The end-of-loop counterpart of :meth:`_record_stop` (fix round 11):
        every task that could run has run, and the ones in
        ``result.blocked_by_check`` were left out. Each is ``not run`` with
        :func:`blocked_by_check_reason` on its ``pending`` status, and the
        stop record (``kind: "check-findings"``) names the checks, what they
        reported and the tasks waiting for them — the record the summary,
        ``write_batch_report.json``, the mirror and the result line's
        ``error`` render from.
        """
        for task_id, check_id in result.blocked_by_check.items():
            result.not_run[task_id] = blocked_by_check_reason(check_id)
        self._record_not_run_in_state(result)
        self._restamp_check_trees(result)
        checks = self._blocking_checks(result)
        found = result.check_findings()
        first = checks[0]
        findings = found.get(first) or []
        result.stop = {
            "batch_index": None,
            "failed_task_ids": [],
            "task_id": first,
            "kind": "check-findings",
            "check_task_ids": checks,
            "blocked": dict(sorted(result.blocked_by_check.items())),
            "what": (
                f"its check reported {len(findings)} problem(s)"
                if findings else "its check reported problems"
            ),
            "reported": "; ".join(findings)[:600] or None,
            "not_run": sorted(result.not_run),
            "acceptable": True,
            "not_rerun": False,
        }
        result.stop["headline"] = self._stop_headline(result.stop)

    def _check_block_message(self, result: SchedulerResult) -> List[str]:
        """The exit-3 block after a run whose checks kept tasks from running.

        Names each check, what it reported and the tasks waiting for it, the
        account of every task, and the three ways forward: fix and resume
        (the check runs again), accept the findings (``=done``: the waiting
        tasks run, the findings stay listed for review-code) or skip.
        """
        cmds = self.user_commands
        found = result.check_findings()
        blocked = result.blocked_by_check
        checks = self._blocking_checks(result)
        lines = [
            "",
            f"ERROR: {self.phase} did not run {len(blocked)} task(s) that depend "
            f"on a check that reported problems:",
        ]
        for check_id in checks:
            findings = found.get(check_id) or []
            waiting = sorted(tid for tid, cid in blocked.items() if cid == check_id)
            lines.append(
                f"  {check_id}{self._task_label(check_id)} reported "
                f"{len(findings)} problem(s): {'; '.join(findings)[:600]}"
            )
            lines.append(f"      not run: {', '.join(waiting)}")
        lines.extend(self._account_lines(result))
        resume = cmds.get("resume") or "re-invoke this phase with --resume"
        lines.append("To go on:")
        lines.append(
            "  - fix what the check reported in the code — by hand, or run the "
            "review-code phase and then apply-code-fixes"
            + (":" if cmds.get("review") and cmds.get("fix") else "")
        )
        if cmds.get("review") and cmds.get("fix"):
            lines.append(f"        {cmds['review']}")
            lines.append(f"        {cmds['fix']}")
        lines.append(
            "    then resume this phase; the check runs again and, once it "
            "passes, the tasks after it follow (a resume that changes no file "
            "does not run it again):"
        )
        lines.append(f"        {resume}")
        for check_id in checks:
            waiting = sorted(tid for tid, cid in blocked.items() if cid == check_id)
            accept = cmds.get("accept", "").replace("{task_id}", check_id) or (
                f"add --human-task {check_id}=done to the resume command"
            )
            skip = cmds.get("skip", "").replace("{task_id}", check_id) or (
                f"add --human-task {check_id}=skip to the resume command"
            )
            retry = cmds.get("retry", "").replace("{task_id}", check_id) or (
                f"add --human-task {check_id}=retry to the resume command"
            )
            lines.append(
                f"  - or accept {check_id}'s findings and run the tasks after "
                f"it: {accept} ({', '.join(waiting)} then run; the findings "
                f"stay listed for review-code), or {skip} ({check_id} and "
                f"{', '.join(waiting)} stay undone)."
            )
            lines.append(
                f"  - or run {check_id}'s check again on unchanged files: {retry}."
            )
        return lines

    def _stop_headline(self, stop: Dict[str, Any]) -> str:
        """One plain line for the result line's ``error``: what the hosts relay
        as "the last line it left"."""
        task_id = stop.get("task_id")
        if stop.get("kind") == "check-findings":
            checks = stop.get("check_task_ids") or [task_id]
            named = ", ".join(
                f"{cid}{self._task_label(cid)}" for cid in checks
            )
            head = (
                f"{self.phase} did not run {', '.join(sorted(stop.get('blocked') or {}))}: "
                f"they depend on {named}, whose check reported problems"
            )
            if stop.get("reported"):
                head += f" [{stop['reported']}]"
            answer = (
                f"--human-task {task_id}=done" if len(checks) == 1
                else "--human-task <check id>=done for each check"
            )
            head += (
                ". To go on: fix what the check reported (by hand, or "
                "review-code then apply-code-fixes), then resume this phase — "
                f"the check runs again; or accept its findings by adding "
                f"{answer} to the resume command (the tasks after it then run)."
            )
            return head
        if not task_id:
            return (
                f"{self.phase} stopped at batch {stop.get('batch_index')}: "
                f"{stop.get('what') or 'the batch was refused'}"
            )
        head = f"{self.phase} stopped at {task_id}{self._task_label(task_id)}: {stop.get('what')}"
        if stop.get("reported"):
            head += f" [{stop['reported']}]"
        head += "."
        if stop.get("not_run"):
            head += f" Not run: {', '.join(stop['not_run'])}."
        if stop.get("kind") == "check" and stop.get("acceptable"):
            head += (
                " To go on: fix what the check reported (by hand, or review-code "
                "then apply-code-fixes), then resume this phase; or accept the "
                f"failure by adding --human-task {task_id}=done to the resume "
                f"command (the tasks after it then run)."
            )
        elif stop.get("acceptable"):
            head += (
                " To go on: fix what it reported, then resume this phase; or "
                f"accept the failure by adding --human-task {task_id}=done to "
                "the resume command."
            )
        else:
            head += (
                " To go on: inspect the half-applied state, restore or keep it, "
                "fix the cause, then resume this phase."
            )
        return head

    def _refusal_message(
        self, exc: ExecutorError, batch_index: int, result: SchedulerResult
    ) -> List[str]:
        """A refused batch, worded like :meth:`_stop_message` (D3k rung 1).

        The refusal's own text already names the offending key, the spec as
        written and the fix (one rejection formatter, ``_render_runner_rejection``),
        so this frames it rather than restating it: which batch was refused, that
        no child started for it, and what the batches before it left behind.
        """
        lines = [
            "",
            f"ERROR: batch {batch_index} was refused before any child started. No "
            f"later batch was dispatched and the {self.phase} phase is NOT marked "
            f"completed.",
            f"  - {exc.message}",
        ]
        if result.batches_dispatched:
            lines += [
                f"Batches already applied this run: "
                f"{', '.join(str(i) for i in result.batches_dispatched)} — their "
                f"pre-batch snapshots are intact under "
                f"{os.path.join(self.safety_dir, 'pre_batch')}/, and re-invoking "
                f"with --resume re-dispatches only the unfinished batches.",
                f"Audit log: {self.audit.path}",
            ]
        else:
            lines.append(
                "Nothing was dispatched and no file was modified."
            )
        lines.extend(self._account_lines(result))
        return lines

    def _record_refusal(self, exc: ExecutorError, batch_index: int) -> None:
        """I11: name the refusal in ``.status.json``, with nothing in flight.

        Plan codex-sandbox-unavailable-findings-and-fixes §5 I11 (2026-09-30):
        after a batch-0 refusal the status file still listed ``apply-batch-0``
        as "in flight (since 0m 9s)". The in-flight entry is closed by
        ``_dispatch_one``'s ``finally`` before the mirror above is written;
        this records what ended the run in the read path's own close-out
        words (``review_orchestrator_base._print_executor_refusal``):
        ``runner_rejected`` for a refused runner, ``executor_refused`` for any
        other pre-dispatch refusal — both terminal to ``status``, which renders
        the repair off ``reason_code``. Best-effort, like every mirror write.
        """
        try:
            from . import status_file as _status_file
        except ImportError:  # pragma: no cover
            from utils import status_file as _status_file
        rejected = isinstance(exc, RunnerRejected)
        try:
            with self._sweep_lock:
                self._dispatch_started.clear()
            _status_file.update_status(self.phase_dir, {
                "state": "runner_rejected" if rejected else "executor_refused",
                "reason_code": (
                    (getattr(exc, "reason_code", "") or "") if rejected else ""
                ) or type(exc).__name__,
                "runner": (getattr(exc, "spec", "") or "") if rejected else "",
                "runner_key": (getattr(exc, "key", "") or "") if rejected else "",
                "refused_batch": batch_index,
                "progress": {"write_subagents": {
                    **self._mirror_snapshot(),
                    "in_flight": [],
                }},
            })
        except Exception:  # noqa: BLE001 - observability must not fail a phase
            pass

    def _mirror_snapshot(self) -> Dict[str, Any]:
        """The ``write_subagents`` mirror as last written, for a patch."""
        try:
            from . import status_file as _status_file
        except ImportError:  # pragma: no cover
            from utils import status_file as _status_file
        try:
            existing = _status_file.read_status(self.phase_dir)
        except Exception:  # noqa: BLE001 - a missing file is an empty mirror
            return {}
        progress = existing.get("progress") if isinstance(existing, dict) else None
        mirror = (progress or {}).get("write_subagents") if isinstance(progress, dict) else None
        return dict(mirror) if isinstance(mirror, dict) else {}

    # -- verification ---------------------------------------------------

    def _run_verification(self, result: SchedulerResult) -> bool:
        """Run ``driver.verify_command`` once, after the last batch (F4).

        Returns whether the phase may complete. Three ways that is ``True``
        without a command ever running, and only the first is announced:

        * **No command configured.** Logged, because "this repo declares no
          check" and "the check passed" are different facts and a silent skip
          spells them the same way. `instructions/apply-code-fixes.md` used to
          hard-code `pnpm typecheck && pnpm lint:fix`, which is right for one
          repo shape and wrong for the rest — the audited run's target had no
          `package.json` at all.
        * **A markdown phase.** `apply_suggestions`/`apply_task_suggestions`
          edit the plan and the tasks file; a typecheck has nothing to say about
          them, so they skip silently rather than nagging about a command that
          was never going to apply.
        * **Nothing applied.** No batch dispatched by this run **and** none
          recorded applied by an earlier run of this phase
          (:meth:`_phase_has_applied_batches`) means no edit to check.

        The command is a **shell string** (`&&` is in its documented shape), run
        with `cwd` at the pinned repo root — the same root every containment
        check resolves against — and bounded by `driver.verify_timeout_s`, which
        is a bound on the whole process group and not just on `/bin/sh`
        (:meth:`_run_verify_command`). It is logged before it starts and again
        with its exit code once it ends.
        """
        if self.kind not in CODE_WRITE_KINDS:
            return True
        if not result.batches_dispatched and not self._phase_has_applied_batches():
            return True
        command = get_verify_command(anchor=self.anchor)
        if not command:
            self.log(
                "[write-subagents] no driver.verify_command configured — "
                f"{len(result.batches_dispatched)} batch(es) were verified "
                "against their own snapshots, but nothing checked the "
                "repository as a whole."
            )
            return True
        timeout = get_verify_timeout_s(anchor=self.anchor)
        source = get_driver_provenance("verify_command", anchor=self.anchor)
        # Announced BEFORE it runs, which is what `docs/configuration.md`'s
        # safety section already promises of the one config key whose value is
        # executed: the only line used to be written once the command returned,
        # so a 600 s check left a detached run's log silent about what it was
        # waiting on — and naming it is the whole mitigation for an untrusted
        # repo's override (F6/F3a, 2026-09-16). The closing line below still
        # reports the exit code and the duration.
        self.log(
            f"[write-subagents] verification ({source}): running "
            f"`{_redacted_text(command)}` (timeout {timeout}s)"
        )
        started = time.monotonic()
        exit_code, output = self._run_verify_command(command, timeout)
        duration = time.monotonic() - started
        # Redact BEFORE the tail cap, the same order the audit log uses: a build
        # that echoes a token into its log would otherwise put it in
        # `.status.json`, `write_batch_report.json` and `summary.md` at once.
        result.verification = {
            "command": _redacted_text(command),
            "source": source,
            "exit_code": exit_code,
            "duration_seconds": round(duration, 1),
            "output_tail": _redacted_text(output).strip()[-VERIFY_TAIL_CHARS:],
        }
        self.log(
            f"[write-subagents] verification ({source}): "
            f"`{result.verification['command']}` exited {exit_code} "
            f"in {duration:.1f}s"
        )
        if exit_code == 0:
            return True
        result.exit_code = EXIT_NEEDS_HUMAN
        result.messages.extend(self._verification_message(result))
        return False

    def _phase_has_applied_batches(self) -> bool:
        """Whether an earlier run of this phase left applied batches behind (F3).

        The check used to gate on ``result.batches_dispatched`` alone. The
        apply kinds re-materialize their full batch list on ``--resume`` and a
        resume skip counts as dispatched, so their check re-ran; ``implement``
        materializes only the tasks still owed, so after a run in which every
        task applied and only the check failed, the resumed list was EMPTY,
        nothing dispatched, the check never re-ran and ``_finalize`` stamped
        the phase completed against a repository still failing its own check
        — the exact ``--resume`` the exit-3 message had told the user to run
        (2026-09-16). ``state.json``'s task statuses are the record that
        survives that filter, so they answer here.
        """
        if self.state is None or not hasattr(self.state, "get_all_task_statuses"):
            return False
        try:
            statuses = self.state.get_all_task_statuses()
        except Exception:  # noqa: BLE001 - advisory, never fatal
            return False
        candidates = {str(task.get("task_id") or "") for task in self.tasks}
        if self.kind in _ID_KEYED_WRITE_KINDS:
            # The plan's own task ids, materialized this run or not — the flat
            # map also holds the apply kinds' batch ids, which are not this
            # phase's.
            candidates |= {
                task_id for task_id in statuses if _IMPLEMENT_TASK_ID.match(task_id)
            }
        candidates.discard("")
        return any(statuses.get(task_id) == "completed" for task_id in candidates)

    def _run_verify_command(self, command: str, timeout: int) -> Tuple[int, str]:
        """Run the shell string; return ``(exit_code, output)`` (F3/F4, F6).

        Three things the old one-line ``subprocess.run(shell=True, timeout=…)``
        could not do, each measured 2026-09-16:

        * **Stop the check.** ``run``'s timeout SIGKILLs ``/bin/sh`` and nothing
          else, and the documented ``pnpm typecheck && pnpm lint:fix`` shape is
          exactly the one where the shell cannot exec into the command — so the
          phase reported ``exited 124`` while ``node`` kept the CPU, reparented
          to init. The child is its own process-group leader here
          (``process_control.new_process_group_kwargs``) and a timeout takes the
          whole group down through ``kill_tree``, the same facade and the same
          escalation the runner spawn uses.
        * **Bound the wait.** A grandchild that inherited the capture pipes keeps
          ``communicate()`` blocked past the deadline, so ``verify_timeout_s``
          was not a bound at all. Both streams go to one temp file instead:
          ``wait`` bounds on the direct child, and no descriptor this process
          reads is held by anything that outlived the kill.
        * **Bound the memory.** Only the last :data:`VERIFY_OUTPUT_CAP_BYTES` are
          read back — a failing build prints megabytes and what names the failure
          is the end of it, which is why the record keeps a tail in the first
          place.
        """
        env = self._verify_env()
        try:
            sink = tempfile.TemporaryFile(prefix="multi_llm_verify_")
        except OSError as exc:  # pragma: no cover - no writable temp dir
            return VERIFY_EXIT_NOT_RUN, (
                f"verification output could not be captured: {exc}"
            )
        with sink:
            try:
                proc = subprocess.Popen(  # noqa: S602 - a shell string is the contract
                    command, shell=True, cwd=self.repo_root,
                    stdin=subprocess.DEVNULL, stdout=sink,
                    stderr=subprocess.STDOUT, env=env,
                    **_process_control.new_process_group_kwargs(),
                )
            except OSError as exc:  # pragma: no cover - the shell itself is missing
                return VERIFY_EXIT_NOT_RUN, (
                    f"verification command could not be started: {exc}"
                )
            # Read the group while the leader is alive: `os.getpgid` stops
            # answering once it is reaped, and the group is what still holds the
            # build's children.
            pgid = _process_control.process_group_id(proc.pid)
            note = ""
            try:
                exit_code = int(proc.wait(timeout=timeout))
            except subprocess.TimeoutExpired:
                verdict = self._kill_verify_tree(proc, pgid)
                exit_code = VERIFY_EXIT_TIMEOUT
                note = (
                    f"\n(killed after {timeout}s: SIGTERM then SIGKILL to the "
                    f"command's whole process group — verdict {verdict})"
                )
            return exit_code, self._verify_output_tail(sink) + note

    def _kill_verify_tree(self, proc: Any, pgid: Optional[int]) -> str:
        """SIGTERM → SIGKILL the timed-out command *and its group*, then reap it.

        ``start_time=None`` is how an unverified kill is *requested*: this pid is
        a child still held open here, not a record read off disk, so there is
        nothing to corroborate and no recycled pid to protect.
        """
        try:
            verdict = _process_control.kill_tree(
                proc.pid, start_time=None, pgid=pgid,
                escalate_after_s=VERIFY_KILL_GRACE_S, poll_interval_s=0.05,
            )
        except Exception:  # noqa: BLE001 - teardown must never mask the timeout
            try:
                proc.kill()
            except OSError:  # pragma: no cover - already gone
                pass
            verdict = "killed"
        try:
            proc.wait(timeout=VERIFY_REAP_S)
        except subprocess.TimeoutExpired:  # pragma: no cover - survived SIGKILL
            pass
        return verdict

    def _verify_output_tail(self, sink: Any) -> str:
        """The last :data:`VERIFY_OUTPUT_CAP_BYTES` the command wrote, as text."""
        try:
            size = sink.seek(0, os.SEEK_END)
            sink.seek(max(0, size - VERIFY_OUTPUT_CAP_BYTES))
            raw = sink.read()
        except (OSError, ValueError):  # pragma: no cover - sink already closed
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return raw or ""

    def _verify_env(self) -> Optional[Dict[str, str]]:
        """The environment ``driver.verify_command`` runs with (F6).

        Composed by the same function the runner spawn uses
        (``providers.base.build_child_env``) instead of being inherited wholesale
        from ``os.environ``: the running host's own ``env_guards`` — the
        nested-session marker and the session-scoped ids this dispatcher was
        handed — are stripped, and an operator who turned ``env_allowlist`` on
        gets this check reduced to the same deny-by-default set a runner sees.
        There is no provider (this is the repository's own command, not an agent
        CLI), so nothing adapter-specific applies, and ``None`` means "inherit
        unchanged" exactly as it does at the two spawn sites.

        The reason it is worth composing at all: the *string* is config and a run
        cannot change it, but the **scripts it invokes** — `package.json`'s
        `typecheck`, a `Makefile` target — are files the write subagent was just
        allowed to edit, so this command is inside the phase's own blast radius.
        `docs/configuration.md`'s safety section says so too.
        """
        try:
            from .providers.base import build_child_env
        except ImportError:  # pragma: no cover - direct-script import shim
            from utils.providers.base import build_child_env  # type: ignore
        try:
            guards, passthrough = _host_env_guards()
            return build_child_env(
                None, provider_env={}, host_guards=guards,
                host_passthrough=passthrough, anchor=self.anchor,
            )
        except Exception as exc:  # noqa: BLE001 - env policy must not fail a check
            self.log(
                f"[write-subagents] warning: could not compose the verification "
                f"environment ({exc}); inheriting this process's unchanged"
            )
            return None

    def _verification_message(self, result: SchedulerResult) -> List[str]:
        """The exit-3 block for a failed `driver.verify_command` (F4).

        Worded like :meth:`_stop_message`: what happened, what was NOT undone,
        and the one command that moves it forward. Deliberately *not* a rollback
        — every batch was verified against its own snapshot and the half-applied
        state a revert would destroy is exactly what a human needs to read.
        """
        record = result.verification or {}
        lines = [
            "",
            f"ERROR: every batch applied and verified, but the repository's own "
            f"check failed, so the {self.phase} phase is NOT marked completed.",
            f"  - driver.verify_command ({record.get('source', 'unknown')}): "
            f"`{record.get('command', '')}` exited {record.get('exit_code')} "
            f"after {record.get('duration_seconds')}s",
        ]
        tail = str(record.get("output_tail") or "").strip()
        if tail:
            lines.append("Last output:")
            lines += [f"  {line}" for line in tail.splitlines()[-20:]]
        applied = (
            f"batches {', '.join(str(i) for i in result.batches_dispatched)} "
            f"are applied"
            if result.batches_dispatched
            else "every batch an earlier run of this phase applied is still applied"
        )
        lines += [
            f"Nothing was rolled back: {applied} and the pre-batch snapshots "
            f"are intact under {os.path.join(self.safety_dir, 'pre_batch')}/.",
            "Fix the repository (or the fix the batches applied), then re-invoke "
            "with --resume: every batch is already done, so nothing is "
            "re-dispatched and only the check runs again.",
        ]
        return lines

    def _finalize(self, result: SchedulerResult) -> None:
        """``mark_phase_completed`` + tracking + summary + plan reference (D3n).

        Called only after every batch verified. A run that dispatched every batch
        and skipped this is a **failed** run, which is why the flag lands on the
        result rather than being left implicit.

        F4: the repository's own check runs *before* `mark_phase_completed`, and a
        non-zero exit costs the phase its completion. Per-batch verification is a
        byte comparison against a snapshot — it says the runner edited what it
        declared, never that the result still builds — so a phase that applied
        seven code fixes and marked itself completed without once asking the repo
        was reporting on a property it had not checked (measured 2026-09-16).
        """
        if not self._run_verification(result):
            # The stop path's shape, for the same reason: the summary is this
            # run's own account and `--resume` has no other record of what was
            # applied. Nothing is rolled back and nothing is completed — the
            # plan reference, the tracking census and the phase marker all wait
            # for a run whose check passes.
            if self.summary_file:
                result.summary_file = self._write_summary(result)
                result.messages.append(
                    f"Implementation summary: {result.summary_file}"
                )
            return
        if self.state is not None:
            self._metrics_finish()
            self.state.mark_phase_completed(self.phase)
            # F5: re-stamp `plan_hash` to what this run actually left on disk.
            # Every batch's edit is already written by the time `_finalize`
            # runs, so without this the *next* `StateManager` built on this
            # plan reads the run's own edit as a human editing the plan
            # mid-workflow and sets `plan_changed`, which
            # `implement_orchestrator.py` reports as "Plan has changed since
            # last session" for a change the plugin made.
            self.state.refresh_plan_hash()
            self.state.save()
            if self.kind in CODE_WRITE_KINDS:
                # Only a code phase records `tracked_files`: the planning
                # phases edit the plan and the task list, which are never
                # code under review (2026-10-05, Codex racing run 3:
                # apply-task-suggestions' census put both in review-code's
                # scope).
                self._finalize_tracking(result)
                # `finalize_tracking` is a file-level writer: it read the state
                # we just saved, computed `tracked_files` and wrote it back.
                # Adopt its file before touching the dict again, or the `save()`
                # below dumps the pre-finalize dict over it and the phase's
                # only deliverable is not tracked at all (F3, measured
                # 2026-09-16).
                self._reload_state()
        if self.summary_file:
            result.summary_file = self._write_summary(result)
            self._insert_plan_reference(result.summary_file)
            if self.state is not None:
                # `_insert_plan_reference` is the one plan edit a run makes
                # *after* the stamp above (implement's own back-reference
                # insertion) — re-stamp once more so it is not left stale too.
                self.state.refresh_plan_hash()
                self.state.save()
        result.finalized = True

    def _reload_state(self) -> None:
        """Re-read `state.json` after an external writer touched it."""
        reload = getattr(self.state, "reload", None)
        if reload is None:
            return
        try:
            reload()
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a phase
            self.log(f"[write-subagents] warning: state reload failed: {exc}")

    def _metrics_start(self, result: SchedulerResult) -> None:
        """Stamp the phase's ETA baseline before the first dispatch.

        Only the `util metrics start` CLI leg ever called `start_phase`, so on
        the executor path `metrics_progress` never existed and neither did the
        `[ETA]` line `instructions/implement.md` step 5 describes (F4).

        The baseline is the number of **tasks** this run will actually dispatch
        or resume-skip — ``len(self.tasks) - len(result.human_plan.skipped)``,
        the same total ``_record_metrics``/``_mirror_status`` use — never the
        bare materialized count, which still counts a deferred human task that is
        never dispatched (N1, 2026-09-16: with the bare count, the stored
        ``metrics_progress.total_batches`` disagreed with every ``[ETA]`` line
        the run went on to print). Tasks, not batches, because that is the unit
        the numerator counts as well: see :meth:`_record_metrics` (F3).
        """
        if self.state is None or not hasattr(self.state, "start_metrics_phase"):
            return
        total_batches = len(self.tasks) - len(result.human_plan.skipped)
        try:
            self.state.start_metrics_phase(self.phase, total_batches)
            self.state.save()
        except Exception:  # noqa: BLE001 - bookkeeping must not fail a phase
            pass

    def _metrics_finish(self) -> None:
        """Pop the phase's ETA baseline and log its closing line (F7).

        The line goes through ``self.log`` for the same reason every per-batch
        ``[ETA]`` line does — so it lands in the phase log ``status`` points at,
        beside the lines it closes. ``finish_metrics_phase`` used to print it to
        stderr itself, which on a foreground or in-process ``full`` run put the
        closing line somewhere none of the others went.
        """
        if not hasattr(self.state, "finish_metrics_phase"):
            return
        try:
            finished = self.state.finish_metrics_phase(self.phase)
        except Exception:  # noqa: BLE001 - bookkeeping must not fail a phase
            return
        # `(elapsed_ms, line)` when the phase had a baseline, `None` when it
        # never started.
        if isinstance(finished, tuple) and finished[1]:
            self.log(finished[1])

    def _finalize_tracking(self, result: Optional[SchedulerResult] = None) -> None:
        """``finalize_tracking.py --state-file`` in process (D3n).

        Every file a batch verifiably changed travels as ``--touched``.
        ``finalize_tracking`` otherwise subtracts ``pre_existing_changes`` from
        the working tree's current diff, which is right for a file the run
        never touched and wrong for one it did: a file already modified when the
        phase started (measured 2026-09-16, ``index.html`` under ``AM``) is
        subtracted along with the rest, ``tracked_files`` ends up naming only
        the noise, and ``review-code`` reviews the noise.
        """
        touched = self._touched_files(result)
        finalize_tracking = load_finalize_tracking()
        if finalize_tracking is None:
            return
        argv = ["--state-file", str(self.state.state_file)]
        for path in touched:
            argv += ["--touched", path]
        argv += ["--phase", PHASE_BY_WRITE_KIND[self.kind]]
        try:
            code = finalize_tracking.main(argv)
        except SystemExit:  # pragma: no cover - argparse-style exit
            return
        except Exception as exc:  # noqa: BLE001
            self.log(f"[write-subagents] warning: finalize_tracking failed: {exc}")
            return
        # F4 (2026-09-17): `main` *returns* 1 for an unreadable or unparseable
        # state file and for a failed write — it does not raise — and the return
        # value used to be dropped. `_reload_state` then silently kept whatever
        # it could read, the phase finalized, and `tracked_files` stayed whatever
        # an earlier phase had left for `review-code` to review. A census that
        # did not happen has to be visible in both places a reader looks.
        if code:
            message = (
                f"WARNING: finalize_tracking exited {code}; tracked_files was "
                f"NOT updated — `review-code` will read whatever the previous "
                f"phase left. Re-run: util finalize-tracking --state-file "
                f"{self.state.state_file}"
            )
            self.log(f"[write-subagents] {message}")
            if result is not None:
                result.messages.append(message)

    def _touched_files(self, result: Optional[SchedulerResult]) -> List[str]:
        """Every file an ``ok`` batch of this phase verifiably changed.

        This run's outcomes, and on a ``--resume`` the earlier runs' ``ok``
        rows too: a resumed implement materializes only the unfinished tasks,
        so without those rows a file an earlier run edited while it was
        already dirty (the baseline subtracts it) left ``tracked_files`` the
        moment the resumed run recorded its own.
        """
        touched: List[str] = []
        sources: List[Sequence[str]] = []
        for row in (result.earlier_rows if result is not None else []):
            if row.get("status") == STATUS_OK:
                sources.append(row.get("files_touched") or [])
        for outcome in (result.outcomes if result is not None else []):
            if outcome.ok:
                sources.append(outcome.files_touched)
        for paths in sources:
            for path in paths:
                if isinstance(path, str) and path and path not in touched:
                    touched.append(path)
        return touched

    def _record_batch_tracking(self, result: SchedulerResult) -> None:
        """Record what this code phase has changed so far, after each batch.

        ``tracked_files`` used to be written only by ``_finalize``, which a
        run that stops part-way never reaches: the Codex racing run 3
        (2026-10-05) stopped implement at its ninth task with ``index.html``
        created and unrecorded, and review-code reviewed the plan and
        ``tasks.md`` an earlier phase had left there instead. The same census
        ``_finalize`` runs, in memory through this phase's own
        ``StateManager`` (the one writer), quietly, and saved only when the
        list moved.
        """
        if (self.dry_run or not self.finalize or self.state is None
                or self.kind not in CODE_WRITE_KINDS):
            return
        recorded = getattr(self.state, "state", None)
        state_file = getattr(self.state, "state_file", None)
        if not isinstance(recorded, dict) or not state_file:
            return
        finalize_tracking = load_finalize_tracking()
        if finalize_tracking is None:
            return
        try:
            entries = finalize_tracking.phase_tracked_files(
                recorded, self._touched_files(result),
                state_path=Path(str(state_file)),
                phase=PHASE_BY_WRITE_KIND[self.kind],
                say=lambda line: None,
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a phase
            self.log(
                f"[write-subagents] warning: could not record the files this "
                f"batch changed in tracked_files: {exc}"
            )
            return
        if entries == recorded.get("tracked_files"):
            return
        with self._effects_lock:
            self.state.set("tracked_files", entries)
            self.state.save()
        named = ", ".join(
            f"{entry['path']} ({entry['action']})" for entry in entries[:5]
        )
        more = f", and {len(entries) - 5} more" if len(entries) > 5 else ""
        self.log(
            f"[write-subagents] tracked_files now names {len(entries)} "
            f"file(s): {named}{more}"
        )

    def _earlier_report_rows(self) -> List[Dict[str, Any]]:
        """The task rows the previous run left in ``write_batch_report.json``.

        Read before this run writes its own. Anything unreadable is simply no
        earlier rows: the report is evidence, never an input the loop obeys.

        Only rows of *this review cycle*: an apply phase's batch ids
        (``apply-batch-N``) are per cycle and the phase dir outlives the
        review, so a ``--resume`` typed after a ``--force`` review re-run
        carried the previous review's rows as "(earlier run)" rows of a
        different review (adversarial review, 2026-10-02). The rule is
        ``check_workflow_prerequisites._apply_batches_predate_review``'s:
        a report written before its review's latest completion belongs to
        the earlier cycle. A missing date on either side keeps the rows, as
        there — absence of a date is not staleness.
        """
        path = os.path.join(self.phase_dir, WRITE_BATCH_REPORT_FILENAME)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        written_at = _parse_local_timestamp(data.get("written_at"))
        if written_at is None:
            try:
                written_at = datetime.fromtimestamp(os.stat(path).st_mtime)
            except OSError:
                written_at = None
        reviewed_at = self._review_completed_at()
        if written_at is not None and reviewed_at is not None \
                and written_at < reviewed_at:
            self.log(
                f"[write-subagents] the earlier {WRITE_BATCH_REPORT_FILENAME} "
                f"(written {written_at.isoformat(timespec='seconds')}) predates "
                f"this phase's review (completed "
                f"{reviewed_at.isoformat(timespec='seconds')}); its rows belong "
                f"to the previous review cycle and are not carried."
            )
            return []
        rows = data.get("tasks")
        return [
            row for row in (rows or [])
            if isinstance(row, dict) and row.get("task_id")
        ]

    def _review_completed_at(self) -> Optional[datetime]:
        """When the review this apply phase applies last completed, or None.

        ``None`` for ``implement`` (its ``T###`` ids are the plan's, not a
        cycle's) and for a state that records no parseable completion.
        """
        review = REVIEW_PHASE_BY_WRITE_KIND.get(self.kind)
        if review is None or self.state is None:
            return None
        recorded = getattr(self.state, "state", None)
        if not isinstance(recorded, dict):
            return None
        completed = recorded.get("phases_completed")
        if not isinstance(completed, dict):
            return None
        return _parse_local_timestamp(completed.get(review))

    def _write_summary(self, result: SchedulerResult) -> str:
        """The implementation summary, with the three conditional sections (D3n)."""
        plan_name = Path(self.plan_file).name if self.plan_file else "(plan)"
        lines = [
            f"# Implementation Summary: {plan_name}",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            f"**Plan**: {plan_name}",
            f"**Phase**: {self.phase}",
            f"**Invoker**: {self.invoker}",
        ]
        # F4: the template the instructions publish has always promised a
        # `Typecheck status` line, and on this leg nothing filled it. Only for
        # the code-editing kinds — a plan-text phase has no check to report.
        if self.kind in CODE_WRITE_KINDS:
            lines.append(f"**Verification**: {verification_line(result)}")
        rows = result.report_rows()
        earlier = [row for row in rows if row.get("from_earlier_run")]
        account = self._accounting(result)
        lines += [
            "",
            "## Summary",
            "",
            f"{len(result.outcomes)} batch task(s) dispatched across "
            f"{len(result.batches_dispatched)} batch(es) in ascending batch_index "
            f"order, each verified by a pre-batch snapshot comparison before the "
            f"next was dispatched."
            + (
                f" The table below also carries {len(earlier)} task(s) "
                f"recorded by earlier run(s) of this phase (marked), which this "
                f"`--resume` did not dispatch again."
                if earlier else ""
            ),
            "",
            _accounting_line(account),
            "",
            "## Task Results",
            "",
            "| Task | Outcome | Files touched | Notes |",
            "|------|---------|---------------|-------|",
        ]
        for row in rows:
            # `reason` first: it is `_classify`'s own diagnostic (why a
            # partial/clean-failure/no-op-with-bytes-moved outcome landed
            # where it did) and, unlike `summary`, is only ever set when there
            # is something worth explaining beyond the runner's own claim — a
            # plain applied batch has none, so `summary` still carries the
            # row. Before this a self-contradicting `no-op` claim printed its
            # own rosy summary ("T006 already implemented: …") with no sign
            # the bytes disagreed (N2, 2026-09-16).
            task_cell = f"{row['task_id']}" + (
                " (earlier run)" if row.get("from_earlier_run") else ""
            )
            lines.append(
                f"| {task_cell} | {row_outcome_label(row) or row.get('status')} | "
                f"{', '.join(row.get('files_touched') or []) or '—'} | "
                f"{(row.get('reason') or row.get('summary') or '').replace('|', '/')[:160]} |"
            )
        for task_id, reason in sorted(result.human_plan.accepted.items()):
            label = (
                "accepted with its check findings"
                if ACCEPTED_WITH_FINDINGS_MARK in reason else "accepted after failing"
            )
            lines.append(f"| {task_id} | {label} | — | {reason.replace('|', '/')[:160]} |")
        for task_id in sorted(result.human_plan.completed):
            lines.append(f"| {task_id} | completed by hand | — | answered --human-task {task_id}=done |")
        for task_id, reason in sorted(result.not_run.items()):
            lines.append(f"| {task_id} | not run | — | {reason.replace('|', '/')[:160]} |")
        findings_rows = [
            row for row in rows
            if row.get("status") == STATUS_OK and row_check_findings(row)
        ]
        if findings_rows:
            # Racing re-run 4 (2026-10-05): T007's two html-validate problems
            # read VERIFIED here and lived only in the runner's notes. They
            # are listed whole, for the reader and for review-code, which is
            # given this file and state.json's record of them.
            lines += [
                "", "## Check Findings", "",
                "These checks ran and changed no file, but reported problems. "
                + (
                    "The tasks that do not depend on them went on; the tasks "
                    "that do were not run (see below). "
                    if result.blocked_by_check else
                    "No task waits for them, so implement went on. "
                )
                + "Review-code is told about them, and apply-code-fixes "
                "applies the fixes review-code validates.", "",
            ]
            for row in findings_rows:
                lines.append(f"### {row['task_id']}")
                lines.append("")
                lines += [f"- {finding}" for finding in row_check_findings(row)]
                lines.append("")
            lines.pop()
        touched: List[str] = []
        for row in rows:
            for path in row.get("files_touched") or []:
                if path not in touched:
                    touched.append(path)
        lines += ["", "## Files Modified", ""]
        lines += [f"- `{path}`" for path in touched] or ["- (none)"]

        plan = result.human_plan
        deferred = plan.deferred()
        declined = plan.declined()
        if deferred:
            lines += ["", "## Manual Tasks (Deferred)", ""]
            lines += ["| Task | Reason |", "|------|--------|"]
            for task_id, reason in sorted(deferred.items()):
                lines.append(f"| {task_id} | {reason} |")
        if declined:
            # The user's own decision, not work left for them: a pause-and-ask
            # "Skip", a failed task answered `--human-task <id>=skip`, and the
            # tasks skipped with them. Listed apart so the summary and the
            # workflow's `[MANUAL_TASK]` lines (which leave these out) agree.
            lines += ["", "## Skipped at your request", ""]
            lines += ["| Task | Reason |", "|------|--------|"]
            for task_id, reason in sorted(declined.items()):
                lines.append(f"| {task_id} | {reason} |")
        if plan.overrides:
            lines += [
                "", "## Dependency Override Warnings", "",
                "The following tasks ran with skipped human-task dependencies "
                "(skip-continue strategy):", "",
                "| Task | Overridden Dependencies | Risk |",
                "|------|------------------------|------|",
            ]
            for task_id, deps in sorted(plan.overrides.items()):
                lines.append(
                    f"| {task_id} | {', '.join(deps)} | ran against an unmet "
                    f"manual precondition |"
                )
        blocked = result.blocked_by_check
        if blocked:
            # Fix round 11: the tasks a check with findings kept from
            # running, apart from a stop's own not-run list.
            lines += [
                "", "## Not Run: Upstream Check Reported Problems", "",
                "These tasks depend on a check that reported problems (listed "
                "under Check Findings), so they were not run; they stay "
                "`pending`. Fix what the check reported and resume this phase "
                "(the check runs again, and once it passes they run), or "
                "accept the check's findings with `--human-task <check id>=done` "
                "on the resume (they run, and the findings stay listed for "
                "review-code), or leave the check and them undone with "
                "`--human-task <check id>=skip`.", "",
                "| Task | Waits for check |", "|------|-----------------|",
            ]
            for task_id, check_id in sorted(blocked.items()):
                lines.append(f"| {task_id} | {check_id} |")
        if result.stopped_at_batch is not None:
            stop = result.stop or {}
            lines += ["", "## Cancelled", ""]
            if stop.get("headline"):
                lines.append(stop["headline"])
            else:
                lines.append(
                    f"The write loop stopped at batch {result.stopped_at_batch}."
                )
            lines.append("")
            stopped = sorted(t for t in result.not_run if t not in blocked)
            if stopped:
                lines.append(
                    "Not run (the stop came before they were dispatched; they "
                    "stay `pending` and a `--resume` dispatches them after the "
                    "failed task passes or is accepted):"
                )
                lines.append("")
                lines += [f"- {task_id}" for task_id in stopped]
            elif blocked:
                lines.append(
                    "Every other task of the decomposition had run, or waits "
                    "for a check (above)."
                )
            else:
                lines.append("Every other task of the decomposition had run.")
            lines += [
                "",
                f"Snapshots are intact under "
                f"`{os.path.join(self.safety_dir, 'pre_batch')}/`.",
            ]
        text = "\n".join(lines) + "\n"
        target = Path(self.summary_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return str(target)

    def _insert_plan_reference(self, summary_file: Optional[str]) -> None:
        """``insert_implementation_summary_reference`` on the plan (D3n)."""
        if not (self.plan_file and summary_file):
            return
        try:
            from .plan_updater import insert_implementation_summary_reference
        except ImportError:  # pragma: no cover
            from utils.plan_updater import insert_implementation_summary_reference
        relative = self.summary_file_relative or os.path.relpath(
            summary_file, str(Path(self.plan_file).parent)
        )
        try:
            path = Path(self.plan_file)
            content = path.read_text(encoding="utf-8")
            path.write_text(
                insert_implementation_summary_reference(content, relative),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover
            self.log(f"[write-subagents] warning: plan reference failed: {exc}")

    # -- reports and status ---------------------------------------------

    def _write_report(self, result: SchedulerResult) -> str:
        path = os.path.join(self.phase_dir, WRITE_BATCH_REPORT_FILENAME)
        # D3l: a scheduler artifact is written under the same rules as any other
        # executor write — same-directory temp, 0600, ``os.replace`` — so a run
        # killed mid-write leaves either the old report or the new one, never a
        # truncated JSON for `status`/CI to parse, and never a umask-wide mode.
        _atomic_write_json(path, result.to_dict())
        # The executor's own per-dispatch report, so `--status` and the D3k rung
        # accounting look identical on both halves of the executor.
        report = BatchReport(
            phase=self.phase, kind=self.kind, phase_dir=self.phase_dir,
            # N9: the read path fills this from the `task_file` argument
            # `run_task_file` was handed; the write path never takes one —
            # `WriteBatchScheduler` is constructed from an already-materialized
            # task LIST — but `write_task_file_for_phase` always persists that
            # same list at exactly this path (every write orchestrator passes
            # it the identical `phase_dir`), so it is not a guess.
            task_file=os.path.join(self.phase_dir, WRITE_TASK_FILENAME),
            outcomes=list(result.outcomes), invoker=self.invoker,
            # F12: filled the same way the read path fills it
            # (`_preflight_batch_runners`) — every distinct runner this run
            # actually preflighted, with the version it reported.
            runner_versions=dict(self._runner_versions),
            ceilings={"effective": self.write_ceiling(),
                      "driver_max_concurrent_write_subagents":
                          get_max_concurrent_write_subagents(anchor=self.anchor),
                      "provider_max_concurrent": self.provider_write_limits(),
                      "host_max_concurrent_subagents": self.host_max,
                      "shared_artifact_phase":
                          self.kind in SHARED_ARTIFACT_WRITE_KINDS},
        )
        report.write()
        return path

    def _mirror_status(
        self, result: SchedulerResult, *,
        finished_in_batch: Sequence[TaskOutcome] = ()
    ) -> None:
        """Per-task write progress in ``.status.json`` (D3g/D4a). Best-effort.

        ``finished_in_batch`` is the current batch's finished tasks, which are
        not on ``result.outcomes`` until the batch returns; passing them is
        what makes the counts advance *during* a batch instead of at its
        boundary. The rows are appended, never merged, so a task cannot be
        counted twice. (Named to leave ``in_flight`` — the JSON key below —
        for tasks that are the opposite: dispatched and NOT yet finished.)
        """
        try:
            from . import status_file as _status_file
        except ImportError:  # pragma: no cover
            from utils import status_file as _status_file
        try:
            outcomes = list(result.outcomes) + list(finished_in_batch)
            if result.finalized:
                phase_state = "complete"
            elif result.stopped_at_batch is not None or result.exit_code:
                phase_state = "subagents_failed"
            else:
                phase_state = "running"
            # F4: `progress.step` is set before a dispatch and never replaced, so
            # a write phase whose last `report_progress` came from the judging
            # dispatch printed `dispatching human-decision judging` for the whole
            # 288 s of batch application AND in the terminal payload, beside
            # `run.state: done` (measured 2026-09-16). The mirror already writes
            # `progress` on every advance, so it is the one place that can move
            # the step with the work; it is written key-by-key, so every sibling
            # counter survives.
            step = {
                "complete": "complete",
                "subagents_failed": "stopped",
            }.get(phase_state, "applying")
            # F9: the mirror's `total`/`ok` count BATCHES, not the suggestions
            # or fixes inside them — a 15-batch, 29-suggestion apply-suggestions
            # run mirrored (and the model then reported) "15/15 ok, applied 15",
            # 14 suggestions short. `expected_ids` is every write task's own
            # declared item list (`write_phase_bridge`: suggestion/fix member
            # ids for the three apply kinds, the task id itself for implement),
            # so both counts are already on hand with no new bookkeeping.
            #
            # F9 (2026-09-16): both denominators also used to include the human
            # task(s) `human_task_strategy` skips — they never dispatch, so a
            # clean 14-task run with one deferred manual task mirrored
            # `13/14 batches ok`, a run that did everything it could reading as
            # one short. `result.human_plan.skipped` is resolved before this is
            # ever called (`_run`'s step 1), so it is on hand the same way.
            skipped_ids = set(result.human_plan.skipped) | set(result.human_plan.accepted)
            expected_ids_by_task = {
                str(task.get("task_id", "")):
                    [str(i) for i in (task.get("expected_ids") or [])]
                for task in self.tasks
                if str(task.get("task_id", "")) not in skipped_ids
            }
            # Counted over the materialized tasks: an answered task an
            # earlier run recorded (`answered_done_on_record`) is in
            # `accepted` without being one of them.
            total = sum(
                1 for task in self.tasks
                if str(task.get("task_id", "")) not in skipped_ids
            )
            items_total = sum(len(ids) for ids in expected_ids_by_task.values())
            items_applied = sum(
                len(expected_ids_by_task.get(o.task_id, []))
                for o in outcomes if o.write_outcome == WRITE_APPLIED
            )
            # F10: a dispatch that has started and not yet been classified —
            # `_dispatch_started`, added in `_begin_sweep` and removed in
            # `_undeclared_writes`. Read under the same lock those methods take,
            # since a disjoint group dispatches on threads.
            with self._sweep_lock:
                in_flight = sorted(self._dispatch_started.items())
            _status_file.update_status(self.phase_dir, {
                # The read phases record a phase-level `state` beside the run
                # record; without one here `status` printed `phase state: (none)`
                # for every finished write phase (measured 2026-09-16).
                "state": phase_state,
                "phase": self.phase,
                "progress": {
                    "step": step,
                    "write_subagents": {
                        "kind": self.kind,
                        "invoker": self.invoker,
                        # F9: excludes tasks `human_task_strategy` skipped —
                        # they will never dispatch, so counting them here read
                        # a clean run as short by exactly its deferred tasks.
                        "total": total,
                        # F10: was `dispatched`, and counted FINISHED tasks —
                        # for an 11-minute single-task batch it read
                        # `dispatched: 0` the whole time, which a reader
                        # cannot tell from a stall. Renamed to say what it
                        # actually counts; `in_flight` below is the dispatched
                        # count this used to be mistaken for.
                        "completed": len(outcomes),
                        "in_flight": [
                            {"task_id": task_id, "dispatched_at": started}
                            for task_id, started in in_flight
                        ],
                        "items_total": items_total,
                        "items_applied": items_applied,
                        # F7: the run's own resolved runner, so `status` polled
                        # outside the session that launched it names what the
                        # batches actually dispatched to — `render_runner_text`
                        # prefers this over a fresh resolution the same way it
                        # already prefers the read path's `subagents` mirror.
                        "runner_versions": dict(self._runner_versions),
                        "batches_dispatched": list(result.batches_dispatched),
                        "stopped_at_batch": result.stopped_at_batch,
                        # F4: written only once the command really ran, so the
                        # key's presence means "a check happened" rather than
                        # "this version knows about checks". The mirror is
                        # rewritten on every advance and verification is the
                        # last thing a run does, so an absent key on a finished
                        # run is the "not configured" notice in the log.
                        **({"verification": result.verification}
                           if result.verification else {}),
                        "finalized": result.finalized,
                        "human_task_strategy": result.human_plan.strategy,
                        "skipped_task_ids": sorted(result.human_plan.skipped),
                        "pending_human_task_ids": list(result.human_plan.pending),
                        "tasks": [
                            {
                                "task_id": o.task_id,
                                "batch_index": self._batch_index_of(o.task_id),
                                "status": o.status,
                                "write_outcome": o.write_outcome,
                                **({"verification_only": True,
                                    "outcome_label": o.outcome_label}
                                   if o.verification_only else {}),
                                # Racing re-run 4: what a check found, only on
                                # the row of a check that found something.
                                **({"check_findings": list(o.check_findings)}
                                   if o.ok and o.check_findings else {}),
                                "files_touched": list(o.files_touched),
                                # Plan step 3: the rung, only when there was one.
                                **({"sandbox_rung": self._dispatch_rungs[o.task_id]}
                                   if self._dispatch_rungs.get(o.task_id) else {}),
                            }
                            for o in outcomes
                        ],
                        "failed_task_ids": [
                            o.task_id for o in outcomes if not o.ok
                        ],
                        **({"check_findings_task_ids": [
                            o.task_id for o in outcomes
                            if o.ok and o.check_findings
                        ]} if any(o.ok and o.check_findings for o in outcomes)
                           else {}),
                        # Racing re-run 3 (2026-10-05): the tasks a stop left
                        # undispatched, and the failed tasks the user accepted,
                        # so `status` and a terminal poll account for every
                        # task rather than only the dispatched ones.
                        "not_run_task_ids": sorted(result.not_run),
                        # Fix round 11: which of them wait for a check that
                        # reported problems, and for which one.
                        **({"blocked_by_check": dict(sorted(result.blocked_by_check.items()))}
                           if result.blocked_by_check else {}),
                        "accepted_task_ids": sorted(result.human_plan.accepted),
                        **({"stop": copy.deepcopy(result.stop)}
                           if result.stop else {}),
                    }
                }
            })
        except Exception:  # noqa: BLE001 - observability must not fail a phase
            pass

    def _batch_index_of(self, task_id: str) -> Optional[int]:
        for task in self.tasks:
            if str(task.get("task_id", "")) == task_id:
                return int(task.get("batch_index", 0))
        return None

    # -- dry run --------------------------------------------------------

    def _write_dry_run(
        self,
        ordered: Sequence[Tuple[int, List[Dict[str, Any]]]],
        plan: HumanTaskPlan,
    ) -> str:
        """``--dry-run-subagents``: full preflight, zero dispatch (D3l).

        Resolves the runner, renders the exact argv per task, checks the prompt
        transport, applies the profile, the disjointness admission and the write
        policy — then writes the report and starts no child. It is what makes
        "would this batch have been refused?" answerable *before* a model is handed
        edit access.
        """
        try:
            from .subagent_tasks import preferred_prompt_transport, utf16_units
        except ImportError:  # pragma: no cover
            from utils.subagent_tasks import preferred_prompt_transport, utf16_units

        ceiling = self.write_ceiling()
        entries: List[Dict[str, Any]] = []
        for batch_index, batch_tasks in ordered:
            groups = admit_disjoint_groups(
                batch_tasks, ceiling=ceiling, root=self.repo_root
            )
            group_of = {
                str(task.get("task_id", "")): position
                for position, group in enumerate(groups)
                for task in group
            }
            for task in batch_tasks:
                task_id = str(task.get("task_id", ""))
                entry: Dict[str, Any] = {
                    "task_id": task_id,
                    "kind": self.kind,
                    "batch_index": batch_index,
                    "co_schedule_group": group_of.get(task_id),
                    "profile": profile_for_kind(self.kind),
                    "cwd": self.repo_root,
                    "sandbox_root": self.repo_root,
                    "declared_files": [
                        relative_to_root(p, self.repo_root)
                        for p in declared_file_set(task)
                    ],
                    "snapshot_files": [
                        relative_to_root(p, self.repo_root)
                        for p in snapshot_file_set(task)
                    ],
                }
                if plan.is_skipped(task_id) or task.get("executor_role") == "human":
                    entry["would_dispatch"] = False
                    entry["reason"] = plan.skipped.get(
                        task_id, "executor_role: human — never dispatched"
                    )
                    entries.append(entry)
                    continue
                try:
                    prompt = self.dispatched_prompt(task)
                except ResultRejected as exc:
                    entry["would_dispatch"] = False
                    entry["reason"] = exc.message
                    entries.append(entry)
                    continue
                runner = self.forced_runner or resolve_task_runner(
                    task, anchor=self.anchor
                )
                entry["runner"] = runner
                provider_name, _ = parse_model_spec(runner, anchor=self.anchor)
                entry["prompt_transport"] = preferred_prompt_transport(provider_name)
                entry["prompt_utf16_units"] = utf16_units(prompt)
                try:
                    info = self._preflight(
                        runner, entry["profile"],
                        # D3l: this flag renders commands and starts no child, so
                        # the nested-exec preflight keeps the declared verdict
                        # (a `denied` host is still refused) and skips its
                        # throwaway dispatch, which is a real turn.
                        probe_nested=False,
                    )
                    entry["runner_version"] = info["version"]
                    rung = self._recorded_rung(runner, entry["profile"])
                    if rung:
                        entry["sandbox_rung"] = rung
                except ExecutorError as exc:
                    entry["would_dispatch"] = False
                    entry["reason"] = exc.message
                    entries.append(entry)
                    continue
                entry["argv"] = _redacted_argv(
                    self._render_argv(runner, prompt, entry["profile"]), prompt
                )
                try:
                    self.policy.check(entry["declared_files"], when="pre-dispatch")
                except PolicyViolation as exc:
                    entry["would_dispatch"] = False
                    entry["reason"] = exc.message
                    entries.append(entry)
                    continue
                entry["would_dispatch"] = True
                entries.append(entry)

        path = os.path.join(self.phase_dir, DRY_RUN_FILENAME)
        # Same containment as every other executor write (see ``_write_report``).
        _atomic_write_json(path, {
            "dry_run_version": 1,
            "kind": self.kind,
            "phase": self.phase,
            "repo_root": self.repo_root,
            "ceilings": {
                "effective_write_parallelism": ceiling,
                "driver_max_concurrent_write_subagents":
                    get_max_concurrent_write_subagents(anchor=self.anchor),
                "provider_max_concurrent": self.provider_write_limits(),
                "host_max_concurrent_subagents": self.host_max,
                "shared_artifact_phase": self.kind in SHARED_ARTIFACT_WRITE_KINDS,
            },
            "write_paths": self.policy.to_dict(),
            "human_task_strategy": plan.strategy,
            "tasks": entries,
        })
        return path


def _redacted_argv(argv: Sequence[str], prompt: str) -> List[str]:
    try:
        from .write_safety import redact_argv
    except ImportError:  # pragma: no cover
        from utils.write_safety import redact_argv
    return redact_argv(argv, prompt)


# ---------------------------------------------------------------------------
# The one entry point an orchestrator calls
# ---------------------------------------------------------------------------

def load_finalize_tracking():
    """The top-level ``finalize_tracking`` module, or ``None`` if unimportable."""
    try:
        import finalize_tracking  # type: ignore
    except ImportError:  # pragma: no cover - direct-script import shim
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            import finalize_tracking  # type: ignore
        except ImportError:
            return None
    return finalize_tracking


def run_write_phase(args: Any, **kwargs: Any) -> Optional[SchedulerResult]:
    """Drive a write phase through the scheduler, or decline (D3b's write twin).

    Returns ``None`` when neither ``--exec-subagents`` nor ``--dry-run-subagents``
    was requested, so the caller emits its task JSON and its follow-up marker
    exactly as before — **with the flag off, the native path is byte-identical.**
    """
    exec_requested = bool(getattr(args, "exec_subagents", False))
    dry_run = bool(getattr(args, "dry_run_subagents", False))
    if not exec_requested and not dry_run:
        return None
    kwargs.setdefault("dry_run", dry_run)
    scheduler = WriteBatchScheduler(**kwargs)
    return scheduler.run()
