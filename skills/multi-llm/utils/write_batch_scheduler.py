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
``--exec-subagents``, and (through the ``dispatcher=`` seam) the simulated native
invoker the equivalence tests inject (``tests/harness/native_invoker.py``). Finalization is part of the loop's contract, not a caller's
courtesy: *a run that dispatched every batch and produced no summary, no
``prior_changes`` entries and no completed phase is a **failed** run, not a partial
one.*

**One job per module.** This module is the loop: :class:`WriteBatchScheduler`,
its result and :func:`run_write_phase`. The helpers both legs share live in seven
modules below it, lowest first: ``commit_write_guard``, ``check_records``,
``verification_lines``, ``human_task_plan``, ``stop_answers``, ``batch_prompt``
and ``write_outcomes`` (docs/development/write-scheduler.md#where-the-schedulers-helpers-live).
A batch's records are read together, and its verdict stamped, through
``batch_ledger``. The scheduler composes five collaborators over one
``write_context.RunContext``: ``write_verify.VerifyStep`` (the repository's own
check and the tracked-files census), ``write_sweep.ContainmentSweep`` (the
post-run containment sweep and its rollback store), ``write_eta.EtaLines`` (the
``[ETA]`` lines), ``write_reports.WriteReports`` (the mirror, the report, the
summary) and ``write_stop.StopAccount`` (how a stop is worded and recorded).
What a run refused before dispatch is on its result (``result.refusals``).
Every name that moved there is still readable here (:data:`_MOVED`), **for
reading only**: ``write_batch_scheduler.X`` and ``from .write_batch_scheduler
import X`` return the owner's current object, but a patch or assignment here
only shadows the name in this module and reaches no code. Patch a moved name
where it lives; ``tests/test_write_batch_scheduler_moved_names.py`` fails any
test that patches one here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import git_utils as _git_utils
from . import known_checks as _known_checks
from . import landlock_dispatch as _landlock
from . import metrics as _metrics
from . import prior_changes as _prior_changes
from . import process_control as _process_control
from . import subagent_executor as _executor
from . import commit_write_guard as _commit_write_guard
from . import check_records as _check_records
from . import verification_lines as _verification_lines
from . import human_task_plan as _human_task_plan
from . import stop_answers as _stop_answers  # noqa: F401 - `_MOVED` reads it
from . import batch_prompt as _batch_prompt
from . import write_outcomes as _write_outcomes
from . import write_context as _write_context
from . import write_verify as _write_verify
from . import write_sweep as _write_sweep
from . import write_eta as _write_eta
from . import write_reports as _write_reports
from . import write_stop as _write_stop
from . import batch_prompt_builder as _batch_prompt_builder
from .cli_common import EXIT_NEEDS_HUMAN, EXIT_OK
from .batch_ledger import (
    outcome_from_verdict,
    read_verdict_checks,
    verdict_record,
)
from .task_ledger import (
    ACCEPTED_REASON_PREFIX,
    ACCEPTED_WITH_FINDINGS_MARK,
    ANSWERED_SKIP_PREFIX,
    COMPLETED_MANUAL_REASON,
    KEPT_WITH_FINDINGS_LABEL,
    USER_DECLINED_REASON,
)
from .text_clip import (
    clip, with_closing_mark,
)
from .providers.base import (
    SANDBOX_RUNG_PLUGIN_LANDLOCK,
    host_write_posture_consent,
    WRITE_SCOPE_REPO_WIDE,
    WRITE_SCOPE_TASKS,
    WRITE_SCOPES_ON_RUNG,
    JailRequest,
)
from .provider_registry import (
    get_max_concurrent_write_subagents,
    get_provider_max_concurrent,
    load_config_snapshot,
    parse_model_spec,
)
from .subagent_executor import (
    OUTPUT_NAME_PATTERN_BY_KIND,
    STATUS_BATCH_FAILED,
    STATUS_OK,
    STATUS_REJECTED,
    WRITE_APPLIED,
    WRITE_CLEAN_FAILURE,
    WRITE_NO_OP,
    WRITE_POLICY_VIOLATION,
    ExecutorError,
    ResultRejected,
    RunnerRejected,
    TaskOutcome,
    _atomic_write_json,
    _host_ceiling,
    _host_env_guards,
    dispatch_task,
    preflight_runner,
    profile_for_kind,
    resolve_output_target,
    resolve_task_runner,
    batch_check_findings,
    batch_check_note,
    batch_check_words,
    check_findings_from,
    check_record,
    declined_items,
    read_check_reply,
    row_batch_check_accepted,
    row_batch_check_findings,
    row_batch_check_note,
    row_batch_check_repeats,
    row_check_findings,
    row_declined,
    validate_result,
)
from .subagent_tasks import (
    PHASE_BY_WRITE_KIND,
    PRIOR_CHANGES_TOKEN,
    APPLY_WRITE_KINDS,
    SHARED_ARTIFACT_WRITE_KINDS,
    WRITE_KINDS,
    declared_file_set,
    snapshot_file_set,
    write_output_name,
)
from .state_manager import (
    PROCESSED_LEG_EXECUTOR,
    _parse_completion_time,
    predates_review,
    stamp_or_mtime,
)
from .write_safety import (
    CONTEXT_PLAN_RULE,
    PRE_BATCH_DIRNAME,
    DRY_RUN_FILENAME,
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
from .write_record import BatchFacts, BatchRecord, record_batch_facts
from .write_verdict import (
    DECLARED_NAMES_ROWS,
    JailFacts,
    PolicyFacts,
    PolicyRestore,
    TargetReplaced,
    VerdictRule,
    WriteFacts,
    decide,
    guard,
    jail_note,
    reason_for,
    replaced_target_reason,
)


__all__ = [
    "BATCH_CHECK_MARKER",
    "CHECK_FINDINGS_MARKER",
    "check_findings_line",
    "STRATEGIES",
    "STRATEGY_PAUSE",
    "STRATEGY_SKIP_CONTINUE",
    "STRATEGY_SKIP_DEPENDENTS",
    "WRITE_BATCH_REPORT_FILENAME",
    "SchedulerResult",
    "HumanTaskPlan",
    "WriteBatchScheduler",
    "answered_done_on_record",
    "answered_done_label",
    "accepted_checks_on_record",
    "accepted_checks_section",
    "accepted_upstream_checks",
    "BATCH_CHECK_ACCEPTED_MARKER",
    "accepted_findings_with_words_added",
    "BATCH_CHECK_REPEATED_MARKER",
    "batch_check_repeated_line",
    "repeated_findings_for",
    "reported_checks_on_record",
    "split_repeated_findings",
    "move_check_repeats_in_state",
    "batch_check_accepted_line",
    "check_note_clause",
    "restate_findings_problem",
    "words_added_to",
    "split_accepted_findings",
    "accepted_findings_for",
    "task_owns_the_check",
    "task_own_text",
    "BLOCKED_BY_CHECK_PREFIX",
    "bind_prior_changes",
    "last_phase_end",
    "OWN_RUN_CONTEXT_PHASES",
    "blocked_by_check_reason",
    "decomposition_graph",
    "DISPATCHED_PROMPTS_SUBDIR",
    "dispatched_prompt_name",
    "dispatched_prompt_path",
    "held_back_by_checks",
    "rerunnable_checks",
    "transitive_dependents",
    "upstream_held_check",
    "mark_no_longer_skipped",
    "persist_human_task_plan",
    "resolve_human_task_plan",
    "run_write_phase",
    "batch_check_line",
    "batch_check_section",
    "batch_check_summary",
    "declined_words",
    "item_word",
    "quoted_findings",
    "unverified_line",
    "native_stop_unverified_line",
    "native_apply_fixes_unverified_line",
    "write_result_notice",
    "verification_line",
    "verify_command_candidate",
    "with_result_contract",
    "COMMIT_GUARD_KIND",
    "ORCHESTRATOR_OUTPUT_FILENAME",
    "batch_fixes_for",
    "commit_group_files",
    "commit_guard_of",
    "commit_head7",
    "commit_protected_rule",
    "commit_recheck_failing",
    "commit_stale_line",
    "commit_stale_reason",
    "commit_stale_split",
    "load_commit_apply",
    "load_orchestrator_batches",
]


# ---------------------------------------------------------------------------
# The names that moved to the helper modules beside this one
# ---------------------------------------------------------------------------

#: Every top-level name of the ten modules split out of this one, by owner.
#: They are read-only re-exports (see the module docstring): reading one here
#: returns the owner's current binding, and ``dir()`` lists them. Nothing is
#: copied into this module, so a patch of the owner shows through here, and
#: this module's own code reaches each one through its owner at every call.
_MOVED: Dict[str, Tuple[str, ...]] = {
    "commit_write_guard": (
        "COMMIT_GUARD_KIND",
        "ORCHESTRATOR_OUTPUT_FILENAME",
        "batch_fixes_for",
        "commit_group_files",
        "commit_guard_of",
        "commit_head7",
        "commit_protected_rule",
        "commit_recheck_failing",
        "commit_stale_line",
        "commit_stale_reason",
        "commit_stale_split",
        "load_commit_apply",
        "load_orchestrator_batches",
    ),
    "check_records": (
        "ACCEPTED_MATCH_MIN_WORDS",
        "ACCEPTED_NOTE_FINDING_CHARS",
        "ACCEPTED_NOTE_MAX_CHECKS",
        "ACCEPTED_NOTE_MAX_FINDINGS",
        "ACCEPTED_NOTE_TITLE_CHARS",
        "BATCH_CHECK_ACCEPTED_MARKER",
        "BATCH_CHECK_MARKER",
        "BATCH_CHECK_REPEATED_MARKER",
        "BLOCKED_BY_CHECK_PREFIX",
        "CHECK_FINDINGS_MARKER",
        "CHECK_NOTE_SHOWN_CHARS",
        "_BARE_COMMAND_WORDS",
        "_FINDING_ACCEPTANCE_NOTE",
        "_FINDING_COMMAND_END",
        "_FINDING_EXIT",
        "_FINDING_FILLER",
        "_FINDING_QUOTES",
        "_FINDING_WORD",
        "_FIX_ASK",
        "_FIX_NEGATIONS",
        "_FIX_TARGET",
        "_IMPLEMENT_TASK_ID",
        "_PASS_ASK",
        "_SENTENCE_END",
        "_asks_to_fix",
        "_declined_from_a_failed_claim",
        "_finding_parts",
        "_findings_from_a_failed_claim",
        "_id_range",
        "_names_command",
        "_note_sentence",
        "_plain_words",
        "_quoted_finding",
        "_says_only",
        "accepted_checks_on_record",
        "accepted_checks_section",
        "accepted_findings_for",
        "accepted_findings_with_words_added",
        "accepted_upstream_checks",
        "batch_check_accepted_line",
        "batch_check_line",
        "batch_check_repeated_line",
        "batch_check_section",
        "batch_check_summary",
        "blocked_by_check_reason",
        "check_findings_line",
        "check_note_clause",
        "declined_words",
        "decomposition_graph",
        "held_back_by_checks",
        "item_word",
        "move_check_repeats_in_state",
        "quoted_findings",
        "repeated_findings_for",
        "reported_checks_on_record",
        "rerunnable_checks",
        "restate_findings_problem",
        "split_accepted_findings",
        "split_repeated_findings",
        "task_own_text",
        "task_owns_the_check",
        "transitive_dependents",
        "unlanded_on_record",
        "upstream_held_check",
        "words_added_to",
        "write_result_notice",
    ),
    "verification_lines": (
        "CHECKS_RAN_MAX_IDS",
        "NOTHING_RAN_TESTS",
        "PLUGIN_RAN_NO_TESTS",
        "STOP_TALLY_MAX_CHARS",
        "_NPM_PLACEHOLDER_TEST",
        "_check_task_ids_of_plan",
        "implement_checks_ran",
        "native_apply_fixes_unverified_line",
        "native_stop_unverified_line",
        "unverified_line",
        "unverified_with_report",
        "verification_line",
        "verify_command_candidate",
    ),
    "human_task_plan": (
        "HumanTaskPlan",
        "MANUAL_DECLINED_TAIL",
        "STRATEGIES",
        "STRATEGY_PAUSE",
        "STRATEGY_SKIP_CONTINUE",
        "STRATEGY_SKIP_DEPENDENTS",
        "_declined_split",
        "_deferred_reason",
        "_deferred_split",
        "answered_done_label",
        "answered_done_on_record",
        "manual_gate_head",
        "manual_gate_way_on",
        "mark_no_longer_skipped",
        "persist_human_task_plan",
        "plan_manual_gates",
        "resolve_human_task_plan",
        "set_aside_by_plan",
    ),
    "stop_answers": (
        "FailedAnswerable",
        "HALF_APPLIED_LEFT_CLAUSE",
        "HALF_APPLIED_PREFIX",
        "HEADLINE_BESIDE_SECTION_KEY",
        "NATIVE_REFUSED_PREFIX",
        "OUTSIDE_REFUSAL_WORDS",
        "STOP_HEADLINE_LABEL_CAPS",
        "STOP_HEADLINE_QUOTE_MIN",
        "UNLANDED_REASON",
        "_HEADLINE_ANSWER",
        "_accounting_line",
        "_native_stop_facts",
        "answers_cover",
        "check_kept_on_resume",
        "compose_resume_why",
        "compose_stop_answers",
        "executor_check_stamp",
        "fit_stop_headline",
        "implement_tasks_done",
        "native_resume_why",
        "native_stop_answers",
    ),
    "batch_prompt": (
        "BATCH_PROMPT_RULE",
        "CHECK_RESULT_CONTRACT",
        "CODE_WRITE_KINDS",
        "DISPATCHED_PROMPTS_SUBDIR",
        "IMPLEMENT_WHICH_CHECKS",
        "OWN_RUN_CONTEXT_PHASES",
        "RESULT_IDS_HEADING",
        "WRITE_CHECK_CONTRACT",
        "_scope_apply",
        "_state_on_disk",
        "bind_prior_changes",
        "dispatch_line",
        "dispatched_prompt_name",
        "dispatched_prompt_path",
        "is_check_task",
        "last_phase_end",
        "with_result_contract",
    ),
    "write_outcomes": (
        "REFUSED_SKIP_WAY_FORWARD",
        "ResumeContext",
        "already_kept",
        "check_partial_skips",
        "classify_attempt",
        "partial_skip_refusal",
        "partial_when_bytes_moved",
        "resume_why",
    ),
    "write_context": (
        "Refusals",
        "RunContext",
        "_ID_KEYED_WRITE_KINDS",
        "refusals_of",
    ),
    "write_verify": (
        "VERIFY_TAIL_CHARS",
        "VerifyStep",
        "_redacted_text",
        "load_finalize_tracking",
    ),
    "write_sweep": (
        "ContainmentSweep",
        "ROLLBACK_STORE_DIRNAME",
        "SWEEP_ADOPT_MAX_FILES",
        "SWEEP_MAX_FILES",
        "SWEEP_PRUNED_DIR_NAMES",
        "SWEEP_RETAIN_MAX_BYTES",
        "UndeclaredWrites",
    ),
    "write_eta": (
        "EtaLines",
    ),
    "write_reports": (
        "WRITE_BATCH_REPORT_FILENAME",
        "WriteReports",
    ),
    "write_stop": (
        "StopAccount",
    ),
    "batch_prompt_builder": (
        "BatchPromptBuilder",
    ),
}
_MOVED_TO: Dict[str, str] = {
    name: module for module, names in _MOVED.items() for name in names
}


def __getattr__(name: str) -> Any:
    """A moved name, read from the module that defines it now."""
    owner = _MOVED_TO.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(globals()[f"_{owner}"], name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> List[str]:
    """This module's own names and the moved ones it re-exports."""
    return sorted({*globals(), *_MOVED_TO})

#: The D2g inputs binding for the results already on disk, kept beside them.
#: A **leading dot** on purpose: reaggregation globs ``{phase_dir}/*.json`` and
#: treats every file it matches as a per-model result list, and ``glob`` skips
#: leading-dot names — the same reason ``.status.json`` is invisible to it.
WRITE_FINGERPRINT_FILENAME = ".write_task_inputs.json"

#: How much of the command's output is read back off disk before redaction and
#: the tail cap (``write_verify.VERIFY_TAIL_CHARS``). `capture_output=True` buffered a whole build in this
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


def _verify_timeout_note(
    timeout: int, verdict: Optional[str], pid: int, pgid: Optional[int],
) -> str:
    """The line a timed-out verify command's output ends with, from its kill's
    *verdict* (``WriteBatchScheduler._kill_verify_tree``). Only ``killed`` and
    ``gone`` confirm the command stopped (``subagent_executor._unconfirmed_kill``);
    every other verdict, ``None`` for a kill that raised included, names the
    pid and says the command may still be running, never that it was killed
    (docs/development/write-scheduler.md, "When the verify command times out")."""
    reason = _executor._unconfirmed_kill(verdict)
    if reason is None:
        if verdict == "gone":
            return (f"\n(timed out after {timeout}s: the command had already stopped, "
                    f"and nothing of its process group still runs — verdict {verdict})")
        return (f"\n(killed after {timeout}s: SIGTERM then SIGKILL to the command's "
                f"whole process group — verdict {verdict})")
    who = f"pid {pid}" + (f", process group {pgid}" if pgid is not None else "")
    if verdict == "skipped":
        # The kill could not trust the command's pid to name it any more.
        return (f"\n(timed out after {timeout}s: termination attempted, but no signal "
                f"was sent, so the command ({who}) may still be running — "
                f"verdict {verdict})")
    if verdict == "unresolved":
        # What ran in its group that could not be identified as the command's
        # got nothing.
        return (f"\n(timed out after {timeout}s: not every process of the command "
                f"({who}) could be confirmed stopped; what could not be identified as "
                f"the command's was not signalled and may still be running — "
                f"verdict {verdict})")
    return (f"\n(timed out after {timeout}s: termination attempted, but the command "
            f"({who}) could not be confirmed stopped: {reason}, so it may still be "
            f"running — verdict {verdict})")


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


def review_completed_at(recorded: Any, write_kind: Any) -> Optional[datetime]:
    """When the review *write_kind* applies last completed, or None.

    The cutoff between review cycles (:func:`state_manager.predates_review`),
    read from a ``state.json`` dict (*recorded*): the review is
    :data:`REVIEW_PHASE_BY_WRITE_KIND`'s entry. None for ``implement`` (its
    ``T###`` ids are the plan's, not a cycle's), for anything that is not a
    state dict, and for a review with no parseable completion.
    """
    review = REVIEW_PHASE_BY_WRITE_KIND.get(write_kind or "")
    if review is None or not isinstance(recorded, dict):
        return None
    completed = recorded.get("phases_completed")
    if not isinstance(completed, dict):
        return None
    return _parse_completion_time(completed.get(review))


# ---------------------------------------------------------------------------
# The write grant on the plugin-landlock rung (plan
# codex-sandbox-unavailable-findings-and-fixes, step 3 option (b), 2026-09-30)
# ---------------------------------------------------------------------------

def write_scope_for_kind(kind: str) -> Optional[str]:
    """The ``write_scope`` a write kind's dispatches carry, or ``None``.

    ``plan-file`` for apply-suggestions, ``tasks`` for apply-task-suggestions,
    ``repo-wide`` for apply-code-fixes and implement (on the rung since plan
    step 3b, 2026-10-02), read from the executor's one table
    (``subagent_executor.WRITE_SCOPE_BY_KIND``). The scope comes from the
    phase's kind, never from the task JSON (F1 "How a dispatch receives its
    grant"). An unmapped kind gets ``None``, which the executor refuses on the
    rung rather than widening it.
    """
    return _executor.WRITE_SCOPE_BY_KIND.get(kind)


def _tighter_cap(first: Optional[int], second: Optional[int]) -> Optional[int]:
    """The lower of two ``max_concurrent`` caps, ``None`` being "no cap"."""
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


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
    human_plan: _human_task_plan.HumanTaskPlan = field(default_factory=_human_task_plan.HumanTaskPlan)
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
    #: message). Set by ``StopAccount.record_stop``; empty on a run that did not stop.
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
    #: Findings of recorded checks absent from a selected task's handoff.
    #: They must still explain a hold, even after switching from the native
    #: leg, which writes no executor report rows.
    recorded_check_findings: Dict[str, List[str]] = field(default_factory=dict)
    #: The tasks this run refused before any runner started, and why
    #: (``write_context.Refusals``): read by the stop, the ``[ETA]`` lines,
    #: the summary and the mirror. Never in the report (``to_dict``).
    refusals: _write_context.Refusals = field(default_factory=_write_context.Refusals)
    #: Fix round 14: the line saying nothing ran the repository's tests,
    #: when a code phase finished with no ``driver.verify_command`` set
    #: (:func:`unverified_line`); ``None`` when one ran, or none was owed.
    unverified: Optional[str] = None

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
        found.update(self.recorded_check_findings)
        return found

    def batch_check_findings(self) -> Dict[str, List[str]]:
        """``{task_id: findings}`` for every code batch that finished ``ok``
        while a check it ran on its own edits failed (fix round 14), this
        run's and the earlier runs' rows alike, in row order."""
        found: Dict[str, List[str]] = {}
        for row in self.report_rows():
            findings = row_batch_check_findings(row)
            if row.get("status") == STATUS_OK and findings:
                found[str(row.get("task_id"))] = findings
        return found

    def batch_check_accepted(self) -> Dict[str, Dict[str, List[str]]]:
        """``{task_id: {check id: findings}}`` for every code task that
        finished ``ok`` while its own check reported again findings the user
        accepted for a check (fix round 22), in row order."""
        found: Dict[str, Dict[str, List[str]]] = {}
        for row in self.report_rows():
            matched = row_batch_check_accepted(row)
            if row.get("status") == STATUS_OK and matched:
                found[str(row.get("task_id"))] = matched
        return found

    def batch_check_repeats(self) -> Dict[str, Dict[str, List[str]]]:
        """``{task_id: {check id: findings}}`` for every code task that
        finished ``ok`` while its own check reported only what a check task
        of the run reported (fix round 28), in row order."""
        found: Dict[str, Dict[str, List[str]]] = {}
        for row in self.report_rows():
            matched = row_batch_check_repeats(row)
            if row.get("status") == STATUS_OK and matched:
                found[str(row.get("task_id"))] = matched
        return found

    def batch_check_notes(self) -> Dict[str, str]:
        """``{task_id: note}`` for every code task that finished ``ok`` with
        an own-check note (its ``check.note``, fix round 28), in row order."""
        found: Dict[str, str] = {}
        for row in self.report_rows():
            note = row_batch_check_note(row)
            if row.get("status") == STATUS_OK and note:
                found[str(row.get("task_id"))] = note
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
            **({"recorded_check_findings": self.recorded_check_findings}
               if self.recorded_check_findings else {}),
            "stop": copy.deepcopy(self.stop) if self.stop else None,
            **({"unverified": self.unverified} if self.unverified else {}),
            "tasks": self.report_rows(),
        }


#: The label of the summary-path line an ending of a run prints
#: (:meth:`WriteBatchScheduler._end_run`): a stop's summary is partial, a
#: manual gate's covers the run.
PARTIAL_SUMMARY_LINE = "Partial implementation summary"
FULL_SUMMARY_LINE = "Implementation summary"


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
        dispatcher: How each admitted task is dispatched, called as
            ``dispatcher(scheduler, task, prompt, snapshot, profile, declared,
            batch_index=...)``. ``None`` is :meth:`_dispatch_executor`, the
            executor under ``--exec-subagents``; no orchestrator passes another.
            An injected one (the equivalence tests' simulated native invoker)
            names itself in its ``invoker`` attribute and starts no runner, so
            no runner preflight decides rollback storage for it. Containment,
            classification and the prior-changes append are the loop's, the
            same for either — which is what makes "either invoker, same
            terminal state" a property of the code.
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
        dispatcher: Optional[Callable[..., TaskOutcome]] = None,
        finalize: bool = True,
        log: Optional[Callable[[str], None]] = None,
        write_target: Optional[str] = None,
        user_commands: Optional[Dict[str, str]] = None,
        partial_skips: Optional[Sequence[str]] = None,
        scope_policy: Optional[str] = None,
        scope_baseline_id: Optional[str] = None,
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
        # The phase's driver and provider legs of the same minimum, filled on
        # first use and kept as every batch's upper bound: see `_ceiling_legs`.
        self._ceiling_legs_cache: Optional[
            Tuple[int, Dict[str, Optional[int]]]
        ] = None
        # The runner each task of the batch being dispatched resolved to, and
        # what the admitted batches read: see `_batch_ceiling`.
        self._batch_runners: Dict[str, str] = {}
        self._admitted_ceilings: List[int] = []
        self._admitted_driver: Optional[int] = None
        self._admitted_limits: Dict[str, Optional[int]] = {}
        self.forced_runner = runner
        self.resume = resume
        self.dry_run = dry_run
        self.human_task_strategy = human_task_strategy
        self.human_task_decisions = dict(human_task_decisions or {})
        #: The dispatch seam (see ``dispatcher`` above), an unbound callable
        #: taking this scheduler first.
        self._dispatcher: Callable[..., TaskOutcome] = (
            dispatcher or WriteBatchScheduler._dispatch_executor
        )
        #: True when dispatches go through the executor, which spawns runners.
        self._executor_dispatch = dispatcher is None
        self.finalize = finalize
        self.log = log or (lambda line: print(line))
        #: The way forward after a stop, spelled the way this host's user
        #: types it (the orchestrator builds them from the host's
        #: `next_command` template; the scheduler knows no host): `resume`,
        #: `review`, `fix`, and the two `--human-task` variants `accept` /
        #: `skip` with `{task_id}` left for the failed task. Absent (a direct
        #: invocation), the stop message names the flags instead.
        self.user_commands: Dict[str, str] = dict(user_commands or {})
        #: Fix round 18: the apply batches answered ``--human-task <id>=skip``
        #: on this resume — each stopped as a partial application whose report
        #: left items out, kept as it stands, those items recorded as skipped
        #: at the user's request (:meth:`_check_partial_skips`).
        self.partial_skips: List[str] = [str(t) for t in (partial_skips or [])]
        #: The scope policy an apply dispatch enforces (plans/
        #: scope-policy-and-report-tags.md §5): the orchestrator's effective
        #: one, else the round's record in ``state.json`` at dispatch.
        self.scope_policy = scope_policy
        self.scope_baseline_id = scope_baseline_id
        self._kept_at_request: Dict[str, Dict[str, Any]] = {}
        #: The answers :meth:`_check_partial_skips` refused, with why (fix
        #: round 19: recorded as a stop, :meth:`_record_answer_refusal`).
        self._refused_skips: Dict[str, str] = {}

        # Plan step 3 (2026-09-30): the grant a write dispatch on the
        # plugin-landlock rung carries. ONE request for the whole phase — the
        # scope from the kind map, the target from the phase's own artifact —
        # never a per-task path, so nothing a task file says can widen it.
        self.write_scope = write_scope_for_kind(kind)
        self.write_target = self._checked_write_target(write_target)
        self.policy = WritePolicy.load(self.repo_root, anchor=self.anchor)
        #: plans/review-commits.md §2.7: the batches the pre-dispatch
        #: re-check stopped (``task id -> their batch line``), which the loop
        #: goes past instead of stopping on, and this phase's
        #: ``orchestrator_output.json`` batches, whose fixes name every file
        #: their groups do (read once, on the first batch a guard governs).
        self._commit_stale: Dict[str, str] = {}
        self._commit_batches: Optional[List[Any]] = None
        #: The watched paths' disk ids read just before each dispatch
        #: (``task id -> {path: id}``), for its post-run protected check.
        self._commit_watch_before: Dict[str, Dict[str, Optional[str]]] = {}
        # The apply guard's protected set joins the policy as an exact-path
        # deny layer: the pre-dispatch check and the post-run sweep below then
        # fail a batch that writes one, as for a driver.write_paths deny hit.
        self._commit_protect(self._commit_guard())
        self.jail_request = self._build_jail_request()
        self.audit = AuditLog(self.phase_dir)
        #: What the collaborators below may read about this run (WRITE-5):
        #: these same objects, never copies; `log` reaches `self.log` at each
        #: line (`_late`), so quieting it quiets them too.
        self._ctx = _write_context.RunContext(
            kind=self.kind, phase=self.phase, phase_dir=self.phase_dir,
            output_dir=self.output_dir, repo_root=self.repo_root,
            anchor=self.anchor, tasks=self.tasks, state=self.state,
            policy=self.policy, audit=self.audit, log=self._late("log"),
            summary_file=self.summary_file, dry_run=self.dry_run,
        )
        #: ``task id -> the known check commands its prompt listed``
        #: (`BatchPromptBuilder.known_checks_section`), which its reply's
        #: ``listed`` is read against (``_listed_outcomes``).
        self._told_checks: Dict[str, List[str]] = {}
        #: The prompt each dispatch is sent and the copy kept of it
        #: (`batch_prompt_builder.BatchPromptBuilder`). The phase dir the
        #: copies go in is pinned now, before any batch of this run can
        #: replace it (fix round 17).
        self._prompts = _batch_prompt_builder.BatchPromptBuilder(
            self._ctx,
            plan_file=self.plan_file,
            scope_policy=self.scope_policy,
            scope_baseline_id=self.scope_baseline_id,
            told_checks=self._told_checks,
        )
        try:
            self._prompts.pinned_phase_dir()
        except OSError:
            pass  # named on the first prompt kept, with that task's id
        #: Whether `_decide_rollback_storage` ran for the admitted group.
        self._rollback_storage_decided = False
        self._rollback_storage_lock = threading.Lock()
        self._rollback_storage_candidates: Optional[Sequence[Dict[str, Any]]] = None
        self.snapshots: Dict[str, TaskSnapshot] = {}
        #: Task ids whose pre-batch snapshot was REUSED from an earlier
        #: dispatch of this same batch (`_snapshot`/`_reusable_snapshot`),
        #: rather than taken fresh at the top of this dispatch. `_classify`
        #: reads this to tell "the diff is a killed attempt's, not this
        #: dispatch's" apart from "this dispatch's own snapshot disagrees
        #: with its no-op claim" (N2, 2026-09-16) — the two read identically
        #: from the bytes alone.
        self._snapshot_reused: set = set()
        self.invoker = "executor" if dispatcher is None else str(dispatcher.invoker)
        #: Serializes the per-dispatch bookkeeping (`_record_batch_effects`).
        #: With `driver.max_concurrent_write_subagents` > 1 a disjoint group runs
        #: in a thread pool, and every member reaches the SAME in-memory
        #: `StateManager` and the SAME `prior_changes.jsonl`. Both are
        #: read-modify-write, so two unsynchronized siblings would race on
        #: `state.save()`'s `os.replace` — last writer wins and a sibling's task
        #: status is silently dropped, leaving resume and the report disagreeing
        #: with what actually ran.
        self._effects_lock = threading.Lock()
        #: The repository's own check and the tracked-files census
        #: (`write_verify.VerifyStep`), stateless; its `state.json` writes
        #: take `_effects_lock` above.
        self._verify = _write_verify.VerifyStep(
            self._ctx,
            effects_lock=self._effects_lock,
            finalize=self.finalize,
            task_by_id=self._late("_task_by_id"),
            unverified_text=self._late("_unverified_text"),
            set_stop_headline=self._late("_stop.set_stop_headline"),
            run_verify_command=self._late("_run_verify_command"),
        )
        #: How a stop is worded and recorded on the result
        #: (`write_stop.StopAccount`), stateless; its one `state.json` write
        #: takes `_effects_lock` above.
        self._stop = _write_stop.StopAccount(
            self._ctx,
            effects_lock=self._effects_lock,
            user_commands=self.user_commands,
            task_by_id=self._late("_task_by_id"),
            dependents_of=self._late("_dependents_of"),
            keep_partial_clause=self._late("_keep_partial_clause"),
            unapplied_words=self._late("_unapplied_words"),
            restamp_check_trees=self._late("_restamp_check_trees"),
            tree_fingerprint=self._late("_tree_fingerprint"),
            started_a_runner=self._late("_started_a_runner"),
        )
        #: The run's `[ETA]` counts and lines (`write_eta.EtaLines`): the
        #: dispatch walk sets what it will count, the batch bookkeeping marks
        #: each task under `_effects_lock`, and every ending closes them once.
        self._eta = _write_eta.EtaLines(
            self._ctx,
            ran_out_of_time=self._late("_stop.ran_out_of_time"),
            started_a_runner=self._late("_started_a_runner"),
        )
        #: Task ids for which a terminal (`completed`) audit line has been
        #: written, so `_audit_terminal` can close out a dispatch that never
        #: reached `_classify` without ever double-recording one that did.
        self._audited: set = set()
        #: The post-run containment sweep and its rollback store
        #: (`write_sweep.ContainmentSweep`): each in-flight dispatch's census,
        #: declared set and retained bytes, under the sweep's own lock.
        self._sweep = _write_sweep.ContainmentSweep(self._ctx)
        #: ``{spec: version}`` for every distinct runner this run actually
        #: preflighted (F12), populated as ``_dispatch_executor`` dispatches —
        #: the read path fills the same-shaped field from
        #: ``_preflight_batch_runners``; the write path preflights inline, per
        #: dispatch, and this is where its return value was being discarded.
        self._runner_versions: Dict[str, str] = {}
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
        #: target) and failed the batch: never a success line in the
        #: prior-changes log — the failed batch's own line, which the
        #: ``--resume`` that re-dispatches it supersedes (fix round 15).
        self._jail_failed: set = set()
        #: The subset whose target was replaced under the batch: their failure
        #: writes no prior-changes line at all (fix round 15).
        self._replaced_targets: set = set()
        #: Batch indexes whose stray-writes-only line was already printed.
        self._rung_noticed: set = set()
        #: What reading each reply's ``listed`` against ``_told_checks`` gave.
        self._listed_outcomes: Dict[str, Any] = {}
        #: ``task id -> its outcomes so far`` (:meth:`_remember_outcome`).
        self._recorded_outcomes: Dict[str, List[TaskOutcome]] = {}
        #: Whether this run stamped implement's start (:meth:`_mark_implement_started`).
        self._implement_started_marked = False
        #: The run's mirror, report, summary and plan reference
        #: (`write_reports.WriteReports`), read-only over the result; built
        #: last, over the records above that it reads.
        self._reports = _write_reports.WriteReports(
            self._ctx,
            plan_file=self.plan_file,
            summary_file_relative=self.summary_file_relative,
            write_target=self.write_target,
            invoker=self.invoker,
            host_max=self.host_max,
            user_commands=self.user_commands,
            sweep=self._sweep,
            eta=self._eta,
            runner_versions=self._runner_versions,
            dispatch_rungs=self._dispatch_rungs,
            resume_answered_ids=self._late("_resume_answered_ids"),
            resume_kept_rows=self._late("_resume_kept_rows"),
            ceiling_record=self._late("_ceiling_record"),
            check_findings_fate=self._late("_check_findings_fate"),
            task_by_id=self._late("_task_by_id"),
            started_a_runner=self._late("_started_a_runner"),
            accounting=self._late("_stop.accounting"),
            manual_gates=self._late("_stop.manual_gates"),
        )

    def _late(self, name: str) -> Callable[..., Any]:
        """``self.<name>(...)`` as it is bound when called, for a collaborator
        to call back: an instance patch of *name* applies to it too. A dotted
        *name* (``"_stop.accounting"``) reaches a collaborator's method the
        same way, each part looked up at each call."""
        parts = name.split(".")

        def call(*args: Any, **kwargs: Any) -> Any:
            target: Any = self
            for part in parts:
                target = getattr(target, part)
            return target(*args, **kwargs)

        return call

    # -- the write grant (plan step 3, 2026-09-30) ----------------------

    @property
    def safety_dir(self) -> str:
        """Where this phase's pre-batch snapshots, rollback store and audit
        log live (``write_safety.safety_dir``): the phase dir, or the
        out-of-grant directory a jailed repo-wide run claimed."""
        return safety_dir(self.phase_dir)

    def _decide_rollback_storage(self, task: Dict[str, Any], profile: str) -> None:
        """Choose storage for an admitted group's runners before snapshots start.

        The first task can use another runner than a later or co-scheduled
        jailed task. Resolving the group's runner set under this lock prevents
        promotion while a sibling is already writing its snapshot or audit.
        An inconclusive preflight conservatively requires external storage;
        its actual dispatch still reports its own refusal, and a later
        successful preflight cannot strand evidence inside a writable grant.
        """
        if self.dry_run or not self._executor_dispatch:
            return
        with self._rollback_storage_lock:
            if self._rollback_storage_decided:
                return
            if self.write_scope != WRITE_SCOPE_REPO_WIDE:
                self._rollback_storage_decided = True
                return
            needs_external = False
            seen = set()
            candidates = self._rollback_storage_candidates
            for candidate in (candidates if candidates is not None else self.tasks or [task]):
                if candidate.get("is_human") or candidate.get("executor_role") == "human":
                    continue
                if (candidate is not task and self.resume
                        and not _batch_prompt.is_check_task(self.kind, candidate)
                        and self._done_evidence(candidate) is not None):
                    continue
                try:
                    runner = self._runner_for(candidate)
                    candidate_profile = profile_for_kind(str(candidate.get("kind") or self.kind))
                    key = (runner, candidate_profile)
                    if key in seen:
                        continue
                    seen.add(key)
                    self._preflight(runner, candidate_profile)
                    needs_external |= self._recorded_rung(runner, candidate_profile) == SANDBOX_RUNG_PLUGIN_LANDLOCK
                except Exception:  # the task's own dispatch reports the preflight error
                    needs_external = True
            if needs_external:
                external = claim_external_safety_dir(self.phase_dir, self.repo_root)
                for snapshot in self.snapshots.values():
                    snapshot.directory = os.path.join(
                        external, PRE_BATCH_DIRNAME, Path(snapshot.directory).name,
                    )
                self.log(
                    f"[write-subagents] this repo-wide phase may run batches on the "
                    f"plugin's jail, which grants the phase dir too: pre-batch "
                    f"snapshots, the rollback store and the audit log live outside "
                    f"it, at {external} (a copy inside the phase dir is never read)."
                )
            self._rollback_storage_decided = True

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
        """``preflight_runner`` with this phase's request.

        For ``agent-write`` the request's scope joins the executor's sandbox
        decision key, so a ``plan-file`` decision is never replayed for a
        ``repo-wide`` dispatch of the same runner.
        """
        return preflight_runner(
            runner, profile=profile, anchor=self.anchor,
            key="driver.subagent_runner", jail_request=self.jail_request,
            **extra,
        )

    def _recorded_rung(self, runner: str, profile: str) -> Optional[str]:
        """The fallback rung the preflight recorded for this runner, profile
        and scope, or ``None`` (codex's own sandbox, or another runner)."""
        try:
            rung = _executor.sandbox_rung(
                runner, profile, write_scope=self.write_scope or None
            )
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
        if self.jail_request is None:
            return
        key = batch_index if batch_index is not None else runner
        if key in self._rung_noticed:
            return
        try:
            notice = _executor.write_rung_notice(runner, self.jail_request)
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

        This is the phase's number, over every task's runner, resolved once
        (:meth:`_ceiling_legs`). Each batch is admitted under it or lower
        (:meth:`_batch_ceiling`).
        """
        configured, provider_limits = self._ceiling_legs()
        candidates = [configured]
        if self.host_max is not None:
            candidates.append(int(self.host_max))
        limits = [
            limit for limit in provider_limits.values()
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

        Resolution mirrors the dispatch path (``--runner`` override, else each
        task's own chain), and it is computed identically for both invokers so
        the executor and the simulated native invoker admit the same groups. A
        spec naming a provider with no configured cap contributes ``None``, which
        is "no provider leg", never zero. Read from :meth:`_ceiling_legs`, so it
        is the map :meth:`write_ceiling` was computed from.
        """
        return dict(self._ceiling_legs()[1])

    def _ceiling_legs(self) -> Tuple[int, Dict[str, Optional[int]]]:
        """``driver.max_concurrent_write_subagents`` and the provider caps, once.

        Resolved on the first call, every task's runner from one config
        snapshot, and kept as ``host_max`` is in ``__init__``: the phase's upper
        bound. :meth:`_batch_ceiling` reads the current values for each batch
        and binds to any lower one, so a config change can lower a later
        batch's ceiling but never raise it above this one
        (``docs/development/write-safety.md``). A failed resolution caches
        nothing and raises again on the next call.
        """
        if self._ceiling_legs_cache is None:
            snapshot = load_config_snapshot(anchor=self.anchor)
            configured = get_max_concurrent_write_subagents(snapshot=snapshot)
            specs = [self.forced_runner] if self.forced_runner else [
                resolve_task_runner(task, snapshot=snapshot) for task in self.tasks
            ]
            self._ceiling_legs_cache = (
                configured, self._provider_limits(specs, snapshot=snapshot),
            )
        return self._ceiling_legs_cache

    def _provider_limits(
        self, specs: Iterable[str], *, snapshot: Any,
    ) -> Dict[str, Optional[int]]:
        """``max_concurrent`` of each provider ``specs`` name, from ``snapshot``."""
        limits: Dict[str, Optional[int]] = {}
        for spec in dict.fromkeys(specs):
            provider_name, _ = parse_model_spec(spec, snapshot=snapshot)
            limits[provider_name] = get_provider_max_concurrent(
                provider_name, snapshot=snapshot
            )
        return limits

    def _batch_runners_and_ceiling(
        self, tasks: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, str], int]:
        """One batch's runners (:meth:`_resolve_batch_runners`) and the ceiling
        it is admitted under (:meth:`_batch_ceiling`), from one config snapshot.

        The snapshot is loaded when the batch starts, so it sees an edit made
        since the last batch, and both read it: one config load, so one git
        discovery, per batch however many tasks it holds. The snapshot stands
        for ``self.anchor``, so the reads take it alone, never beside an anchor
        (``provider_registry.require_one_config_source``).
        """
        snapshot = load_config_snapshot(anchor=self.anchor)
        runners = self._resolve_batch_runners(tasks, snapshot=snapshot)
        return runners, self._batch_ceiling(runners, snapshot=snapshot)

    def _resolve_batch_runners(
        self, tasks: Sequence[Dict[str, Any]], *, snapshot: Any,
    ) -> Dict[str, str]:
        """The runner spec of each of one batch's tasks, by task id, resolved once.

        ``--runner``, else the task's own chain, read from the batch's config
        ``snapshot``: the configuration or the installed CLIs may have changed
        since an earlier batch. The batch's ceiling, the rollback-storage
        preflight and the dispatch all read this map (:meth:`_runner_for`), so
        they name one runner per task and the ceiling binds the providers that
        really run.
        """
        return {
            str(task.get("task_id", "")):
                self.forced_runner or resolve_task_runner(task, snapshot=snapshot)
            for task in tasks
        }

    def _runner_for(self, task: Dict[str, Any]) -> str:
        """The runner ``task`` dispatches on: its batch's resolution
        (:meth:`_resolve_batch_runners`), else resolved now (a task outside the
        batch being dispatched)."""
        if self.forced_runner:
            return self.forced_runner
        spec = self._batch_runners.get(str(task.get("task_id", "")))
        if spec is None:
            spec = resolve_task_runner(task, anchor=self.anchor)
        return spec

    def _batch_ceiling(self, runners: Mapping[str, str], *, snapshot: Any) -> int:
        """The ceiling one batch is admitted under: never above :meth:`write_ceiling`.

        The phase's number is the upper bound. Under it the driver key and the
        ``max_concurrent`` of each provider this batch's ``runners`` name (the
        specs it dispatches on) are read again, so a lower value in the current
        configuration, or a task that now resolves to a provider with a tighter
        cap, binds this batch; a raised value leaves it at the phase's number.
        Every one of these values comes from the batch's config ``snapshot``
        (:meth:`_batch_runners_and_ceiling`), never a load per key or per
        provider. Called on the loop's thread before the batch's workers
        start; what it read is kept for :meth:`_ceiling_record`.
        """
        phase = self.write_ceiling()
        configured = get_max_concurrent_write_subagents(snapshot=snapshot)
        limits = self._provider_limits(runners.values(), snapshot=snapshot)
        caps = [limit for limit in limits.values() if limit is not None]
        ceiling = max(1, min([phase, configured, *caps]))
        self._admitted_ceilings.append(ceiling)
        self._admitted_driver = configured if self._admitted_driver is None else min(
            self._admitted_driver, configured
        )
        for provider, limit in limits.items():
            self._admitted_limits[provider] = _tighter_cap(
                self._admitted_limits.get(provider), limit
            )
        return ceiling

    def _ceiling_record(self) -> Tuple[int, int, Dict[str, Optional[int]]]:
        """``(effective, driver, providers)`` for the batch report and the
        dry-run record, from what the batches were admitted under.

        ``effective`` is the highest ceiling a batch was admitted under, or
        :meth:`write_ceiling` when none was. The driver value and each
        provider's cap are the tightest the phase's resolution or any batch
        read, and the providers include every one a batch's runners resolved
        to. So when a config change lowered a later batch, the legs can sit
        below ``effective``.
        """
        configured, phase_limits = self._ceiling_legs()
        limits = dict(phase_limits)
        for provider, limit in self._admitted_limits.items():
            limits[provider] = _tighter_cap(limits.get(provider), limit)
        if self._admitted_driver is not None:
            configured = min(configured, self._admitted_driver)
        effective = (
            max(self._admitted_ceilings) if self._admitted_ceilings
            else self.write_ceiling()
        )
        return effective, configured, limits

    # -- ordering -------------------------------------------------------

    def batches(self) -> List[Tuple[int, List[Dict[str, Any]]]]:
        """The phase's tasks grouped by ascending ``batch_index``."""
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for task in self.tasks:
            buckets.setdefault(int(task.get("batch_index", 0)), []).append(task)
        return [(index, buckets[index]) for index in sorted(buckets)]

    # -- late prior-changes binding (D3n) -------------------------------

    def _known_checks_report(
        self, max_chars: Optional[int] = None,
    ) -> Optional[Tuple[str, bool]]:
        """What this phase's batches reported about the known checks they
        were told to run, and whether one ran (:func:`known_checks.checks_report`),
        for the tests line; ``None`` for every other kind, or with no check
        known. Never fatal."""
        if self.kind != "apply_code_fixes":
            return None
        try:
            state = self.state
            if state is None:
                state = _known_checks.read_state(self.output_dir)
            return _known_checks.checks_report(
                self._prompts.known_checks(), state, self.tasks,
                **({} if max_chars is None else {"max_chars": max_chars}),
            )
        except Exception:  # noqa: BLE001 - advisory
            return None

    def _unverified_text(self, *, stop: bool = False) -> str:
        """The tests line this code phase owes with no ``driver.verify_command``
        (:func:`unverified_line`), led by what ran (fix round 24): implement's
        tasks' checks (:func:`implement_checks_ran`), apply-code-fixes' known
        checks (:meth:`_known_checks_report`; at a stop only when a batch
        ran one, cut to :data:`STOP_TALLY_MAX_CHARS`, so the stop's remedy
        and answers keep their room beside it in the bounded ``error``)."""
        if self.kind == "implement":
            return _verification_lines.unverified_line(self.repo_root, ran=self._implement_checks_ran())
        report = self._known_checks_report(_verification_lines.STOP_TALLY_MAX_CHARS if stop else None)
        if stop and not (report and report[1]):
            # A stop's line names the tally only to lead with what ran.
            return _verification_lines.unverified_line(self.repo_root)
        return _verification_lines.unverified_with_report(self.repo_root, report)

    def _implement_checks_ran(self) -> Optional[str]:
        """:func:`implement_checks_ran` over the state this run holds, the
        check tasks read off this run's tasks and the plan's task list (a
        resume's batches leave out the tasks an earlier run finished)."""
        try:
            state = getattr(self.state, "state", None)
            if not isinstance(state, dict):
                state = _known_checks.read_state(self.output_dir)
            checks = {
                str(task.get("task_id") or "")
                for task in self.tasks
                if isinstance(task, dict) and not declared_file_set(task)
                and task.get("executor_role") != "human" and not task.get("is_human")
            }
            if self.plan_file:
                checks |= _verification_lines._check_task_ids_of_plan(Path(self.plan_file))
            return _verification_lines.implement_checks_ran(state, checks)
        except Exception:  # noqa: BLE001 - advisory: the line still stands
            return None

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
        except WriteSafetyError as exc:
            # A batch can replace the phase path after dispatch. Trusted
            # discovery refuses it; report a human-needed stop without reading
            # or writing through that replacement to produce more artifacts.
            # Fix round 21: an ending too, so the `[ETA]` lines say it.
            self._eta.close(self._eta.run_result)
            return SchedulerResult(
                kind=self.kind, phase=self.phase, exit_code=EXIT_NEEDS_HUMAN,
                messages=[f"Rollback safety refusal: {exc.message} The phase was not completed."],
            )
        except KeyboardInterrupt:
            # A foreground run's Ctrl-C (a detached worker's signal goes
            # through `metrics.aborted_eta_line` instead, fix round 21).
            self._eta.close(self._eta.run_result, verb="interrupted")
            raise
        except Exception:
            self._eta.close(self._eta.run_result)
            raise
        finally:
            # Whatever the ending, no abort may close lines a later run owns.
            _metrics.close_eta_run(self._eta.abort_line)
            try:
                self._sweep.report_store_utilization()
                self._sweep.discard_store()
            except WriteSafetyError as exc:
                self.log(f"[write-subagents] rollback evidence retained: {exc.message}")
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
        # Every apply run resumes unless `--fresh`, so a first run (nothing
        # kept) says nothing here; its batch lines already say what ran.
        if not self.resume or self.dry_run:
            return
        kept = self._resume_kept_rows(result, result.outcomes)
        # Fix round 16 (Codex stop-path accept 02): a task answered
        # `--human-task <id>=done` is kept too, and the line says how — the
        # accepted check T002 was missing from "kept 3 … T001, T003, T004".
        answered_kept = [
            f"{task_id} ({_human_task_plan.answered_done_label(result.human_plan, task_id)})"
            for task_id in self._resume_answered_ids(result, result.outcomes, kept)
        ]
        if not kept and not answered_kept:
            return
        # Dispatched: the tasks a runner was started for. A task refused
        # before dispatch and a check not run again started none (fix round
        # 14: the stop fixture's 01b counted the refused T005 as dispatched).
        refused = [
            o for o in result.outcomes
            if not o.skipped and o.task_id in result.refusals.before_dispatch
        ]
        not_rerun = [o for o in result.outcomes if not o.skipped and o.not_rerun]
        dispatched = [
            o for o in result.outcomes
            if not o.skipped and o not in refused and o not in not_rerun
        ]
        items = {
            str(task.get("task_id", "")): len(task.get("expected_ids") or [])
            for task in self.tasks
        }

        def counted(ids: Sequence[str], rows: Sequence[Dict[str, Any]] = ()) -> str:
            # An implement task is one item; the count would only repeat it.
            if self.kind == "implement":
                return ""
            by_row = {
                str(row.get("task_id")): len(row.get("expected_ids") or [])
                for row in rows
            }
            total = sum(by_row.get(i, items.get(i, 0)) for i in ids)
            return f" ({total} item(s))"

        def named(ids: Sequence[str]) -> str:
            return ", ".join(ids[:5]) + (", …" if len(ids) > 5 else "")

        kept_ids = [str(row.get("task_id")) for row in kept]
        dispatched_ids = [o.task_id for o in dispatched]
        # Fix round 20 (live run 8 #6): a check kept with its findings
        # standing says so, as an accepted one does — 01b read "T001, T002,
        # T003, T004" as if T002 had passed.
        with_findings = (
            {o.task_id for o in result.outcomes if o.skipped and o.check_findings}
            | set(_check_records.rerunnable_checks(self.state))
        ) if self.kind == "implement" else set()
        plain_ids = [i for i in kept_ids if i not in with_findings]
        kept_named = ", ".join(
            part for part in (
                named(plain_ids) + counted(plain_ids, kept) if plain_ids else "",
                ", ".join(
                    f"{i} ({KEPT_WITH_FINDINGS_LABEL})"
                    for i in kept_ids if i in with_findings
                ),
                ", ".join(answered_kept),
            ) if part
        )
        self.log(
            f"[write-subagents] resume: kept {len(kept_ids) + len(answered_kept)} "
            f"task(s) an earlier run finished — {kept_named}; "
            f"dispatched {len(dispatched_ids)}{counted(dispatched_ids)}"
            + (
                f"; refused before dispatch {len(refused)} "
                f"({named([o.task_id for o in refused])})" if refused else ""
            )
            + (
                f"; not run again {len(not_rerun)} "
                f"({named([o.task_id for o in not_rerun])})" if not_rerun else ""
            )
            + (f"; not reached {len(result.not_run)}" if result.not_run else "")
        )

    def _resume_kept_rows(
        self, result: SchedulerResult, outcomes: Sequence[TaskOutcome],
    ) -> List[Dict[str, Any]]:
        """The report rows of the tasks a resume kept: every task an earlier
        run finished that this run did not dispatch — a resume skip, and
        (fix round 14) a task the resume never materialized because
        ``state.json`` records it completed: implement materializes only the
        unfinished tasks, so its skips were empty and OpenCode racing 06b,
        which kept 14, printed no line at all. Read off the phase's report
        rows (this run's and the earlier runs'), the account the stop's
        "Done and kept" line is built from; a task answered ``--human-task
        <id>=done`` is its own bucket. ``[]`` on a run that is not a resume.

        The ``resume: kept`` line and the write mirror's ``kept_task_ids``
        (``status``'s ``(<n> kept from an earlier run: …)``) both read it,
        then :meth:`_resume_answered_ids`, so the two name the same tasks:
        fix round 22 (live run 9 #5) — Codex 01b's line kept four, its
        ``status`` said "1 kept". *outcomes* is
        what finished so far (the mirror passes a batch's finished tasks
        before they reach ``result``); a task the run's dispatch walk will
        dispatch is never kept, though an earlier run finished it (a check
        run again once the files changed).
        """
        if not self.resume or self.dry_run:
            return []
        this_run = {o.task_id for o in outcomes if not o.skipped}
        if self._eta.scope is not None:
            this_run |= set(self._eta.scope)
        answered = set(result.human_plan.accepted) | set(result.human_plan.completed)
        kept: List[Dict[str, Any]] = []
        seen: set = set()
        for row in result.report_rows():
            task_id = str(row.get("task_id") or "")
            if (
                not task_id or task_id in seen or task_id in this_run
                or task_id in answered or row.get("status") != STATUS_OK
            ):
                continue
            seen.add(task_id)
            kept.append(row)
        return kept

    def _resume_answered_ids(
        self, result: SchedulerResult, outcomes: Sequence[TaskOutcome],
        kept_rows: Sequence[Dict[str, Any]],
    ) -> List[str]:
        """The tasks a resume keeps on the user's ``--human-task <id>=done``
        answer — an accepted check or failed task, a manual task completed by
        hand — that this run did not dispatch and *kept_rows* do not hold, in
        id order: the last part of the ``resume: kept`` line, and of the write
        mirror's ``kept_task_ids`` (fix round 23, review of round 22 #3: the
        mirror left them out, so ``--json`` read three kept beside the line's
        four). ``[]`` on a run that is not a resume."""
        if not self.resume or self.dry_run:
            return []
        this_run = {o.task_id for o in outcomes if not o.skipped}
        seen = {str(row.get("task_id")) for row in kept_rows}
        answered = set(result.human_plan.accepted) | set(result.human_plan.completed)
        return [
            task_id for task_id in sorted(answered)
            if task_id not in this_run and task_id not in seen
        ]

    @staticmethod
    def _jail_line_count() -> int:
        try:
            from .subagent_executor import jail_summary_count
        except Exception:  # noqa: BLE001 - an older executor has no summary
            return 0
        return jail_summary_count()

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
        for line in jail_summary_lines(before):
            self.log(f"[write-subagents] closing summary: {line}")

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
        if self.kind not in _write_context._ID_KEYED_WRITE_KINDS:
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
        result.human_plan = _human_task_plan.resolve_human_task_plan(
            self.tasks,
            strategy=self.human_task_strategy,
            decisions=self.human_task_decisions,
            check_findings_ids=(
                _check_records.rerunnable_checks(self.state) if self.kind == "implement" else ()
            ),
            graph=(
                _check_records.decomposition_graph(self.tasks, self.state)
                if self.kind == "implement" else None
            ),
        )
        if self.kind == "implement" and self.state is not None:
            try:
                recorded = dict(self.state.get_all_task_statuses())
            except Exception:  # noqa: BLE001 - a state fake without the map
                recorded = {}
            _human_task_plan.mark_no_longer_skipped(
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
            accepted, completed_manual = _human_task_plan.answered_done_on_record(
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
            # Fix round 16: a stop says nothing ran the tests too. No
            # `[ETA]` baseline exists yet, so there is nothing to close.
            return self._end_run(
                result, lambda: self._pending_human_message(result.human_plan),
                close_eta=False, check_lines=False,
            )

        # What this run will dispatch, counted as the loop below decides it
        # (fix round 18, run 7 #3): the orchestrator's line counted every
        # task a resume took up, so "4 task(s) to dispatch" preceded
        # "dispatched 0".
        # Fix round 20 (live run 8 #1): the `[ETA]` total starts from the
        # same walk, so the two never disagree.
        if self.kind == "implement":
            self.log(self._dispatch_preview(ordered, result))

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

        # Fix round 18: a `--human-task <batch>=skip` answer is checked
        # against the batch's stopped verdict and the bytes on disk before
        # anything is dispatched; one that cannot apply stops the run here.
        refusals = self._check_partial_skips()
        if refusals:
            result.exit_code = EXIT_NEEDS_HUMAN
            result.messages.extend(refusals)
            # Fix round 19: on the record the tool routes read, so the
            # result line names the refusal and the refused answer is not
            # offered again from the stop it answered.
            self._record_answer_refusal(result)
            return result

        # 3. The batches, in order, each verified before the next is dispatched.
        # Mirrored once before the first dispatch so `progress.step` reads
        # `applying` for the whole stretch rather than only from the first
        # *finished* task — the in-batch mirror runs after a dispatch returns,
        # and a long first batch is exactly when the step matters (F4).
        self._eta.start(result)
        self._reports.mirror_status(result)
        for batch_index, batch_tasks in ordered:
            self._hold_on_recorded_checks(batch_tasks, result)
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
                self._stop.record_stop(result, batch_index, [], refusal=exc.message)
                # The partial summary only when an earlier batch really
                # applied something: it is that run's account (D3n), and an
                # empty one for a refusal on the first dispatch would claim a
                # run that never started.
                self._end_run(
                    result,
                    lambda: self._stop.refusal_message(exc, batch_index, result),
                    stopped=True,
                    summary=PARTIAL_SUMMARY_LINE if result.outcomes else None,
                )
                self._record_refusal(exc, batch_index)
                return result
            result.outcomes.extend(outcomes)
            result.batches_dispatched.append(batch_index)
            # A check that reported problems keeps every task that depends on
            # it from running (fix round 11); the tasks that do not depend on
            # it go on.
            self._block_dependents_of_checks(outcomes, result)
            self._verify.record_batch_tracking(result)
            self._reports.mirror_status(result)
            # plans/review-commits.md §2.7: a batch the commit re-check
            # stopped was never dispatched and the loop goes on past it.
            stale = self._commit_stale
            stopping = [o for o in outcomes if o.task_id not in stale]
            if any(o.stops_the_loop for o in stopping) or any(
                not o.ok for o in stopping
            ):
                result.stopped_at_batch = batch_index
                result.exit_code = EXIT_NEEDS_HUMAN
                self._stop.record_stop(
                    result, batch_index, [o for o in stopping if not o.ok],
                )
                # The partial summary is the cancellation's own record (D3n,
                # instructions/implement.md): `_finalize` never runs on this
                # path, so without it the `## Cancelled` section would be
                # unreachable and `--resume` would have no account of what the
                # stopped run applied. The phase is still NOT completed and the
                # plan-side reference is still NOT inserted.
                return self._end_run(
                    result,
                    lambda: self._stop.stop_message(batch_index, outcomes, result),
                    stopped=True, summary=PARTIAL_SUMMARY_LINE,
                )

        # 4. The tasks a check with findings kept from running: the phase is
        # not complete while they are unrun, so it ends needing a human, the
        # way a stop does — after every task that could run has run, and
        # with no finalization (no verify command, no completion record).
        if result.blocked_by_check:
            result.exit_code = EXIT_NEEDS_HUMAN
            self._stop.record_check_block(result)
            return self._end_run(
                result, lambda: self._stop.check_block_message(result),
                summary=PARTIAL_SUMMARY_LINE,
            )

        # 4b. Fix round 28 (live run 12, Codex racing 06): a run that
        # dispatched nothing because every code task left waits on a manual
        # task not recorded done is not a finished implement. It stops for
        # the user, naming the manual task and the resume that records it.
        gates = self._stop.manual_gate(result)
        if gates is not None:
            return self._stop_on_manual_gate(result, gates)

        # 5. Finalization is part of the contract, not the caller's courtesy.
        # It closes the `[ETA]` lines ("phase complete"), runs the
        # verification and writes the summary itself.
        if self.finalize:
            self._finalize(result)
        return self._end_run(result, close_eta=False, unverified=False)

    def _end_run(
        self,
        result: SchedulerResult,
        ending: Optional[Callable[[], List[str]]] = None,
        *,
        close_eta: bool = True,
        eta_why: str = "",
        unverified: bool = True,
        check_lines: bool = True,
        stopped: bool = False,
        summary: Optional[str] = None,
    ) -> SchedulerResult:
        """End :meth:`_run` with a report, in the one order every ending keeps.

        1. The ``[ETA]`` lines close (*close_eta*, with *eta_why*): every
           ending short of completion says so, rather than leaving
           "~0m 37s remaining" standing.
        2. The "nothing ran the tests" note (*unverified*,
           ``VerifyStep.note_unverified_on_stop``) is set before the summary is
           written, so the summary's Verification line and the stop's
           headline say it too.
        3. The summary, when the ending names one: *summary* is the label of
           its line, and only a plan with a summary file gets one.
        4. The messages, in order: the ``[CHECK_FINDINGS]`` and batch-check
           lines (*check_lines*, *stopped*; read after the summary, which
           they point at), the note, the ending's own lines (*ending*), the
           summary's path.
        5. The report and the status mirror.

        A dry run never reaches here: its record is ``subagent_dry_run.json``.
        """
        if close_eta:
            self._eta.close(result, why=eta_why)
        unverified_lines = self._verify.note_unverified_on_stop(result) if unverified else []
        if summary and self.summary_file:
            result.summary_file = self._reports.write_summary(result)
        if check_lines:
            result.messages.extend(self._check_findings_lines(result, stopped=stopped))
            result.messages.extend(self._batch_check_lines(result))
        result.messages.extend(unverified_lines)
        if ending is not None:
            result.messages.extend(ending())
        if summary and result.summary_file:
            result.messages.append(f"{summary}: {result.summary_file}")
        result.report_path = self._reports.write_report(result)
        self._reports.mirror_status(result)
        return result

    def _dispatch_preview(
        self, ordered: Sequence[Tuple[int, List[Dict[str, Any]]]],
        result: SchedulerResult,
    ) -> str:
        """The ``Organized into …`` line for what this run will dispatch
        (:meth:`_dispatch_walk`), which also sets the ``[ETA]`` line's task
        set (``EtaLines.scope``, fix round 20): the tasks the line counts. A walk
        that fails counts every task, and the ``[ETA]`` total with it."""
        walk = self._dispatch_walk(ordered, result)
        if walk is None:
            self._eta.scope = None
            self._eta.walk_left = {}
            return _batch_prompt.dispatch_line(
                len([1 for _i, b in ordered if b]),
                sum(len(b) for _i, b in ordered),
            )
        batches, picked_ids, waiting = walk
        self._eta.scope = set(picked_ids)
        return _batch_prompt.dispatch_line(batches, len(picked_ids), waiting=waiting)

    def _dispatch_walk(
        self, ordered: Sequence[Tuple[int, List[Dict[str, Any]]]],
        result: SchedulerResult,
    ) -> Optional[Tuple[int, List[str], List[str]]]:
        """``(batches, task ids, waiting ids)`` this run will dispatch, or
        ``None`` when the walk fails.

        Fix round 18 (live run 7, Codex and OpenCode stop resumes): the line
        the orchestrator printed counted every task a resume took up, so
        01b and 01c read "4 task(s) to dispatch" before "dispatched 0" and
        fix 02 read 4 for the 2 it ran. This walks the batches in the loop's
        order with the loop's own read-only tests, its log muted, once the
        human-task plan is resolved:

        * the plan's skipped and accepted tasks and the manual ones are out;
        * a resume keeps a task with done evidence (``_done_evidence``); a
          check whose findings stand is kept while the files are as it left
          them (``_check_findings_rerun``), and the tasks after it are held;
        * a failed check on unchanged files (``_unchanged_since_failure``) or
          a task the write policy refuses is not dispatched, and the run
          stops there: the tasks after its group in the batch, and the later
          batches, are not reached. A group is what ``_run_batch`` dispatches
          together (``admit_disjoint_groups`` under :meth:`write_ceiling`;
          one task each at the default ceiling of 1), so the tasks dispatched
          beside the refused one count (fix round 19).

        A check this run dispatches is decided by its verdict, which is not
        known yet: the tasks after it count, and are named as waiting on it.
        Once a code task of this run was dispatched ahead of it the files
        have moved, so a check after it counts as run again. A runner that
        fails is not foreseen; the run's own lines and its ``resume:`` line
        say what happened.
        """
        graph = _check_records.decomposition_graph(self.tasks, self.state)
        plan = result.human_plan
        # A dry run never persists the plan, so an accepted check can still
        # read as holding its dependents on record.
        on_record = (
            set(_check_records.rerunnable_checks(self.state)) if self.state is not None else set()
        ) - set(plan.accepted)
        rerun: set = set()
        dispatched_checks: List[str] = []
        waiting: Dict[str, List[str]] = {}
        picked_batches = 0
        picked_ids: List[str] = []
        # Fix round 22: what the walk foresees this run leaves (refused, held
        # by a check on record, not reached) and keeps, for `EtaLines.left_out`.
        left: Dict[str, str] = {}
        kept_ids: set = set()
        candidates: List[str] = []
        edits_dispatched = False
        stopped = False
        quiet_log = self.log
        self.log = lambda *_args, **_kwargs: None
        try:
            ceiling = self.write_ceiling()
            for _index, batch_tasks in ordered:
                candidates.extend(
                    str(task.get("task_id", "")) for task in batch_tasks
                    if not (
                        plan.is_skipped(str(task.get("task_id", "")))
                        or str(task.get("task_id", "")) in plan.accepted
                        or task.get("is_human")
                        or task.get("executor_role") == "human"
                    )
                )
                if stopped:
                    continue
                in_batch = 0
                # The loop's runnable list, as far as a preview can tell, in
                # the groups `_run_batch` admits together (fix round 19,
                # review of round 18, #4): with a write ceiling above 1 a
                # group's tasks are dispatched at once and a stop takes
                # effect only once the whole group settled, so a refused
                # task's siblings in its group run.
                runnable = []
                for task in batch_tasks:
                    task_id = str(task.get("task_id", ""))
                    if (
                        plan.is_skipped(task_id) or task_id in plan.accepted
                        or task.get("is_human")
                        or task.get("executor_role") == "human"
                    ):
                        continue
                    holder = _check_records.upstream_held_check(graph, task_id, on_record - rerun)
                    if holder:
                        left[task_id] = _metrics.left_out_check(holder)
                        continue
                    runnable.append(task)
                for group in admit_disjoint_groups(
                    runnable, ceiling=ceiling, root=self.repo_root,
                ):
                    # A code task this run dispatched before this group moves
                    # the files the two unchanged-tree tests compare; one
                    # dispatched beside it has not moved them yet.
                    moved = edits_dispatched
                    for task in group:
                        task_id = str(task.get("task_id", ""))
                        if self.resume:
                            evidence = self._done_evidence(task)
                            if evidence is not None:
                                again = self._check_findings_rerun(
                                    task, evidence[0], result)
                                if again is not True and not (again is False and moved):
                                    kept_ids.add(task_id)
                                    continue
                                rerun.add(task_id)
                            elif task_id in on_record:
                                rerun.add(task_id)
                            if not moved and self._unchanged_since_failure(
                                task, result,
                            ) is not None:
                                stopped = True
                                left[task_id] = _metrics.LEFT_NOT_RUN
                                continue
                        try:
                            self.policy.check([
                                relative_to_root(path, self.repo_root)
                                for path in declared_file_set(task)
                            ], when="pre-dispatch")
                        except PolicyViolation:
                            stopped = True
                            left[task_id] = _metrics.LEFT_REFUSED
                            continue
                        in_batch += 1
                        picked_ids.append(task_id)
                        upstream = [
                            check for check in dispatched_checks
                            if task_id in _check_records.transitive_dependents(graph, check)
                        ]
                        if upstream:
                            waiting[task_id] = upstream
                        if _batch_prompt.is_check_task(self.kind, task):
                            dispatched_checks.append(task_id)
                        else:
                            edits_dispatched = True
                    if stopped:
                        break
                if in_batch:
                    picked_batches += 1
        except Exception:  # noqa: BLE001 - a count line must not fail a phase
            return None
        finally:
            self.log = quiet_log
        for task_id in candidates:
            if task_id not in left and task_id not in kept_ids and task_id not in picked_ids:
                # After the stop: not reached.
                left[task_id] = _metrics.LEFT_NOT_RUN
        self._eta.walk_left = left
        return picked_batches, picked_ids, list(waiting)

    def _check_findings_lines(
        self, result: SchedulerResult, *, stopped: bool = False,
    ) -> List[str]:
        """One ``[CHECK_FINDINGS]`` line per check task that reported
        problems, and one line on what happens to them (racing re-run 4:
        nothing on the run's own output named T007's two problems).

        *stopped*: the run ended on a stop (fix round 14), where the tasks
        after the stop did not go on whatever they depend on.
        """
        found = result.check_findings()
        if not found:
            return []
        lines = [
            _check_records.check_findings_line(task_id, findings)
            for task_id, findings in found.items()
        ]
        where = (
            f"; they are listed in {result.summary_file}"
            if result.summary_file else ""
        )
        went_on = "; ".join(self._check_findings_fate(result, found, stopped=stopped))
        lines.append(
            f"Check findings: {', '.join(found)} ran, changed no file and "
            f"reported problems; {went_on}{where}. Review-code is told about "
            f"them, and apply-code-fixes applies the fixes review-code "
            f"validates."
        )
        return lines

    def _check_findings_fate(
        self, result: SchedulerResult, found: Dict[str, List[str]], *,
        stopped: bool = False, see_below: bool = False,
    ) -> List[str]:
        """What became of the tasks after the checks in *found*, one clause
        per kind of check (lowercase, no closing stop), for the
        ``[CHECK_FINDINGS]`` summary line and summary.md's ``## Check
        Findings``.

        Fix round 18 (live run 7, accept path): a check whose findings the
        user accepted (``--human-task <id>=done``) read "No task waits for
        them, so implement went on", although T007 had waited on it and ran
        only because the findings were accepted. Such a check is named with
        the answer and the tasks after it that ran; the others keep the hold
        (*stopped*: the run ended on a stop) or "no task waits" wording.
        *see_below*: the summary, which lists the held tasks in a section of
        their own instead of by id.
        """
        reasons = getattr(result.human_plan, "accepted", None) or {}
        accepted = [
            check_id for check_id in found
            if ACCEPTED_WITH_FINDINGS_MARK in str(reasons.get(check_id) or "")
        ]
        others = [check_id for check_id in found if check_id not in accepted]
        clauses: List[str] = []
        if accepted:
            done = {
                str(row.get("task_id")) for row in result.report_rows()
                if row.get("status") == STATUS_OK
            }
            for check_id in accepted:
                ran = [dep for dep in self._dependents_of(check_id) if dep in done]
                clauses.append(
                    f"you accepted {check_id}'s findings (--human-task "
                    f"{check_id}=done), so "
                    + (
                        f"{', '.join(ran)}, which "
                        f"{'waits' if len(ran) == 1 else 'wait'} on it, ran, "
                        f"told that you accepted them"
                        if ran else
                        "the tasks that wait on it are no longer held back by it"
                    )
                )
        if not others:
            return clauses
        them = "them" if not accepted else (
            f"{', '.join(others)}'s findings" if len(others) == 1
            else f"the findings of {', '.join(others)}"
        )
        blocked = sorted(result.blocked_by_check)
        held = "(see below)" if see_below else f"({', '.join(blocked)})"
        if blocked and stopped:
            clauses.append(f"the tasks that depend on {them} are held {held}")
        elif blocked:
            clauses.append(
                f"the tasks that do not depend on {them} went on, and the ones "
                f"that do were not run {held}"
            )
        else:
            clauses.append(
                f"no task waits for {them}" + ("" if stopped else ", so implement went on")
            )
        return clauses

    def _batch_check_lines(self, result: SchedulerResult) -> List[str]:
        """One :data:`BATCH_CHECK_MARKER` line per code batch whose own check
        failed after its edits, and one line saying the run is not a clean
        finish (fix round 14); before them, one
        :data:`BATCH_CHECK_ACCEPTED_MARKER` line per code task whose own
        check reported again only findings the user accepted (fix round 22),
        which never makes the run unclean. Fix round 28: each with the
        task's own-check note, when it gave one."""
        notes = result.batch_check_notes()
        lines = [
            _check_records.batch_check_accepted_line(task_id, matched, note=notes.get(task_id))
            for task_id, matched in result.batch_check_accepted().items()
        ]
        # Fix round 28: what only repeats a check task's finding.
        lines += [
            _check_records.batch_check_repeated_line(task_id, matched, note=notes.get(task_id))
            for task_id, matched in result.batch_check_repeats().items()
        ]
        found = result.batch_check_findings()
        if not found:
            return lines
        lines += [
            _check_records.batch_check_line(task_id, findings, note=notes.get(task_id))
            for task_id, findings in found.items()
        ]
        where = (
            f" They are listed under Check Findings in {result.summary_file}."
            if result.summary_file else ""
        )
        lines.append(
            f"Not a clean finish: {_check_records.batch_check_summary(found)} The edits are "
            f"kept; look at the failure before you go on.{where}"
        )
        return lines

    def _hold_on_recorded_checks(
        self, tasks: Sequence[Dict[str, Any]], result: SchedulerResult,
    ) -> None:
        """Honor upstream findings even when their check is outside this run.

        Read at each dependency level: a check run earlier in this invocation
        can pass and release its dependents, while an omitted check (notably
        under ``--task``) cannot silently lose its hold.
        """
        if self.kind != "implement" or self.state is None:
            return
        held = _check_records.rerunnable_checks(self.state)
        graph = _check_records.decomposition_graph(self.tasks, self.state)
        recorded = self.state.get("task_check_findings") or {}
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            if (task.get("executor_role") == "human" or task.get("is_human")
                    or result.human_plan.is_skipped(task_id)
                    or task_id in result.human_plan.accepted):
                continue
            check_id = _check_records.upstream_held_check(graph, task_id, held)
            if check_id is not None:
                result.blocked_by_check[task_id] = check_id
                result.recorded_check_findings[check_id] = [
                    str(item) for item in recorded.get(check_id, [])
                ]

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
        for outcome in outcomes:
            for dependent in self._check_holds(outcome, result):
                result.blocked_by_check[dependent] = outcome.task_id

    def _check_holds(
        self, outcome: TaskOutcome, result: SchedulerResult,
    ) -> List[str]:
        """The tasks *outcome*, a check that reported findings, holds back
        and that nothing holds yet (:meth:`_block_dependents_of_checks`);
        ``[]`` for any other outcome. Reads only: the check's own ``[ETA]``
        line asks it before its batch returns (fix round 21)."""
        if self.kind != "implement" or not (outcome.ok and outcome.check_findings):
            return []
        plan = result.human_plan
        held: List[str] = []
        for dependent in self._dependents_of(outcome.task_id):
            task = self._task_by_id(dependent)
            if task is None:
                # Not in this run (a finished task a resume did not
                # materialize): nothing to hold, but the walk went
                # through it to the tasks after it (fix round 12).
                continue
            if (
                dependent in result.blocked_by_check
                or dependent in held
                or plan.is_skipped(dependent)
                or dependent in plan.accepted
                or task.get("executor_role") == "human"
                or task.get("is_human")
            ):
                continue
            held.append(dependent)
        return held

    # -- one batch ------------------------------------------------------

    def _run_batch(
        self,
        batch_index: int,
        tasks: Sequence[Dict[str, Any]],
        result: SchedulerResult,
    ) -> List[TaskOutcome]:
        # One resolution of this batch's runners, read by its ceiling, its
        # rollback-storage preflight and its dispatches (`_runner_for`).
        self._batch_runners, ceiling = self._batch_runners_and_ceiling(tasks)
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
                # of its own total by exactly the batches it skipped (F3). An
                # implement run counts only what it dispatches (fix round 20),
                # so there the skip leaves both halves.
                with self._effects_lock:
                    self._eta.mark(outcome.task_id, ran=False)
            with mirror_lock:
                outcomes_so_far.append(outcome)
                self._reports.mirror_status(result, finished_in_batch=outcomes_so_far)
            return outcome

        for group in groups:
            # Earlier groups have settled; the next group's runnable candidates
            # alone can select a new storage location. Do this before any of its
            # threads takes a snapshot. Human/check-held tasks never get probed.
            self._rollback_storage_candidates = group
            self._rollback_storage_decided = False
            if len(group) == 1:
                outcomes.append(dispatch_and_mirror(group[0]))
            else:
                with ThreadPoolExecutor(max_workers=len(group)) as pool:
                    futures = [
                        pool.submit(dispatch_and_mirror, task)
                        for task in group
                    ]
                    outcomes.extend(future.result() for future in futures)
            stale = self._commit_stale
            if any(
                (o.stops_the_loop or not o.ok) and o.task_id not in stale
                for o in outcomes
            ):
                # Verification gates the next dispatch (D3h): a partial
                # application is never compounded by a concurrent one.
                break
        return outcomes

    def _dispatch_one(
        self, task: Dict[str, Any], batch_index: int, result: SchedulerResult
    ) -> TaskOutcome:
        task_id = str(task.get("task_id", ""))
        profile = profile_for_kind(self.kind)

        if task_id in self._kept_at_request:
            # Fix round 18: answered `--human-task <id>=skip` — kept as it
            # stood, never dispatched again.
            return self._keep_at_request(task, batch_index, result)

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
            output_path, _recorded_id, recorded = evidence
            self.log(f"[write-subagents] {task_id} already complete — skipping")
            data = json.loads(Path(output_path).read_text(encoding="utf-8"))
            # The outcome its verdict stamp records (`batch_ledger`, the
            # stamp's round trip): its check's label, what its own checks
            # reported, an apply batch's declined items, the files it moved.
            outcome = outcome_from_verdict(
                data, recorded, task, kind=self.kind, runner=self.invoker,
                profile=profile, output_path=output_path,
                files_on_record=lambda: self._recorded_files_touched(output_path),
            )
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
                    self._remember_outcome(outcome)
                    resumed_reason = (
                        "not run again: nothing changed since its check "
                        "reported problems" if kept_findings else
                        f"resumed from {Path(outcome.output_path).name}"
                    )
                    # Its status and items again; its verdict stays as the
                    # earlier run recorded it.
                    self._write_record(task, replace(self._batch_facts(
                        task, outcome, "completed", resumed_reason,
                        dispatched=False, kept=False), resumed=True))
                    # §2.7: its completion's identities, should the earlier
                    # run have stopped before it saved them.
                    self._commit_repin(task, outcome)
                    self.state.save()
            return outcome

        # Past the resume skips, the task is this run's to handle.
        self._mark_implement_started()

        # A resumed verification-only task that failed, with nothing in the
        # repository changed since: not dispatched again (racing re-run 3,
        # 2026-10-05 — `--resume` spent 148 s on the same `html-validate`
        # failure against the same bytes).
        if self.resume:
            refused = self._unchanged_since_failure(task, result)
            if refused is not None:
                self._record_batch_effects(task, refused, batch_index, result)
                return refused

        # The scope policy's stale-task check (plan §5): a task materialized
        # under another policy, baseline or binding is never sent with the
        # wrong policy's prompt. Never blocks an `expand` task without scope
        # metadata.
        scope_refusal = self._prompts.scope_refusal(task)
        if scope_refusal is not None:
            return self._refused(task, scope_refusal, batch_index, result)
        try:
            prompt = self._prompts.with_result_contract(task, self._prompts.dispatched_prompt(task))
        except ResultRejected as exc:
            return self._refused(
                task, exc.message, batch_index, result, status=exc.status)
        self._retire_foreign_result(task)

        # plans/review-commits.md §2.7, "Every batch is re-checked immediately
        # before it is dispatched": beside the D3l check below and before the
        # snapshot, every fix file the batch declares is held to the content
        # the apply guard saved for it. A batch that fails is never
        # dispatched; the loop goes on to the next one. No-op without a guard.
        stale = self._commit_recheck(task, batch_index, result)
        if stale is not None:
            return stale

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
            # As declared (fix round 21): a file outside the repository is
            # named as the task wrote it, never by its resolved absolute path.
            self.policy.check(declared_file_set(task), when="pre-dispatch")
        except PolicyViolation as exc:
            result.refusals.policy_paths[task_id] = list(exc.paths)
            result.refusals.policy_hits[task_id] = list(exc.hits)
            return self._refused(
                task, exc.message, batch_index, result,
                write_outcome=WRITE_POLICY_VIOLATION,
            )
        # Plan step 3 (2026-09-30), beside the D3l check and before the
        # snapshot for the same reason: under `plan-file`/`tasks` the declared
        # set must be exactly the phase's write target, on every rung, or the
        # batch is refused with nothing snapshotted and nothing spawned.
        mismatch = self._grant_mismatch(task)
        if mismatch is not None:
            return self._refused(task, mismatch, batch_index, result)

        try:
            self._decide_rollback_storage(task, profile)
            snapshot = self._snapshot(task)
        except WriteSafetyError as exc:
            return self._refused(task, exc.message, batch_index, result)
        # The replaced-target check's "before": taken at THIS dispatch's
        # pre-batch snapshot (reused or fresh), so the identity compared after
        # the child is always the one this spawn saw (§8.3 item 14).
        self._target_identity_before[task_id] = self._identity_of(
            self._identity_path()
        )
        # §2.7: the protected set's (and context plan's) "before" for this
        # dispatch's own post-run check, independent of the sweep below.
        self._commit_watch_begin(task_id)

        try:
            self._begin_sweep(
                task_id, declared,
                verification_only=self._verification_only(snapshot),
            )
        except WriteSafetyError as exc:
            # The census is this task's whole verdict and there is none: a
            # refusal, with nothing spawned (W1, adversarial review 2026-10-02).
            self._target_identity_before.pop(task_id, None)
            self._commit_watch_before.pop(task_id, None)
            return self._refused(task, exc.message, batch_index, result)
        started = time.time()
        self._git_meta_before[task_id] = self._git_metadata_fingerprint()
        try:
            outcome = self._dispatcher(
                self, task, prompt, snapshot, profile, declared,
                batch_index=batch_index,
            )
            self._warn_git_metadata_changed(task_id)
            outcome.snapshot_dir = snapshot.directory
            if not outcome.duration_seconds:
                outcome.duration_seconds = time.time() - started
            # A dispatch `_classify` never saw still gets its write outcome
            # (`write_outcomes.classify_attempt`); `_audit_terminal` makes it
            # a partial application when the batch moved bytes.
            _write_outcomes.classify_attempt(outcome, snapshot_dir=snapshot.directory)

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
            # registered, the refusal's own `WriteReports.mirror_status` re-published it,
            # so `.status.json` read `in flight: apply-batch-0 (since 0m 9s)`
            # for the rest of the dir's life. The dispatch is over either way.
            self._sweep.abandon(task_id)
            self._target_identity_before.pop(task_id, None)
            self._commit_watch_before.pop(task_id, None)
        self._record_batch_effects(task, outcome, batch_index, result)
        return outcome

    def _check_partial_skips(self) -> List[str]:
        """Check each ``--human-task <batch>=skip`` answer before any dispatch
        (``write_outcomes.check_partial_skips``, each batch's evidence read by
        :meth:`_partial_skip_evidence`): the refusal lines, or ``[]``. A
        refused answer is kept in ``_refused_skips`` for the stop record."""
        if not self.partial_skips:
            return []
        self._refused_skips, lines = _write_outcomes.check_partial_skips(
            self.partial_skips, kind=self.kind, phase=self.phase,
            tasks=self.tasks, evidence=self._partial_skip_evidence,
        )
        return lines

    def _record_answer_refusal(self, result: SchedulerResult) -> None:
        """Keep a refused ``--human-task <batch>=skip`` where the tool routes
        read a stop (fix round 19).

        Review of fix round 18: the refusal was printed and nothing else, so
        the result line named no ``error`` (the tool's needs-human message
        said "Its result line names no error", run 7's #4 shape) and the
        mirror still held the stop the answer was refused for, whose
        ``answers`` ``util next-command --blocked`` handed back as the way
        forward — the answer just refused. The stop record now says the
        answer was refused, in a headline the orchestrator records as the
        result line's ``error``; it keeps the answers of the earlier stop
        that name no refused batch (an answer typed for the wrong batch
        leaves the right one standing), and the mirror's other rows are the
        earlier run's, untouched: this run dispatched nothing.
        """
        refused = dict(self._refused_skips)
        earlier = self._reports.mirror_snapshot()
        before = earlier.get("stop") if isinstance(earlier.get("stop"), dict) else {}
        refused_flags = {f"{task_id}=skip" for task_id in refused}
        kept = [
            dict(entry) for entry in (before.get("answers") or [])
            if isinstance(entry, dict)
            and not refused_flags.intersection(entry.get("flags") or [])
        ]
        named = "; ".join(
            f"--human-task {task_id}=skip: {why}" for task_id, why in refused.items()
        )
        result.stop = {
            "batch_index": None,
            "failed_task_ids": [],
            "task_id": None,
            "kind": "answer-refused",
            "what": f"cannot take {named}",
            "refused_answers": [
                ["--human-task", f"{task_id}=skip"] for task_id in refused
            ],
            "not_run": [],
            "acceptable": False,
            "not_rerun": False,
            **({"answers": kept} if kept else {}),
        }
        self._stop.set_stop_headline(result.stop)
        if not earlier:
            # No earlier run left a mirror to patch: this run's own.
            self._reports.mirror_status(result)
            return
        from . import status_file as _status_file
        try:
            # The phase-level `state` stays the earlier run's: this run
            # changed nothing a status reader would count.
            _status_file.update_status(self.phase_dir, {
                "progress": {"write_subagents": {
                    **earlier,
                    "in_flight": [],
                    "stop": copy.deepcopy(result.stop),
                }},
            })
        except Exception:  # noqa: BLE001 - observability must not fail a phase
            pass

    def _partial_skip_evidence(self, task: Dict[str, Any]) -> Optional[str]:
        """Why *task* cannot be kept at the user's request, or ``None`` when it
        can (recorded in ``_kept_at_request``) or already was: its result file
        read here, the rules in ``write_outcomes.partial_skip_refusal``."""
        task_id = str(task.get("task_id", ""))
        try:
            path = resolve_output_target(task, self.phase_dir)
        except ResultRejected:
            path = None
        if path is None or not path.is_file():
            return (
                f"{task_id} has no result on disk, so it did not stop as a "
                f"partial application this answer could keep"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return f"{task_id}'s result ({path.name}) cannot be read"
        verdict = data.get("verdict") if isinstance(data, dict) else None
        why = _write_outcomes.partial_skip_refusal(
            task_id, verdict, result_name=path.name,
            expected_ids=task.get("expected_ids") or [],
            fingerprint=lambda: self._task_fingerprint(task),
            counts_as_done=lambda: self._result_counts_as_done(
                task, path, task_id, int(task.get("batch_index", 0)), sibling=False,
            ),
            current_digest=lambda relative: file_sha256(
                os.path.join(self.repo_root, relative)),
        )
        if why is None and not _write_outcomes.already_kept(verdict):
            self._kept_at_request[task_id] = {
                "path": str(path), "data": data, "verdict": verdict,
            }
        return why

    def _keep_at_request(
        self, task: Dict[str, Any], batch_index: int, result: SchedulerResult,
    ) -> TaskOutcome:
        """Keep a stopped batch as it stands, at the user's request (fix round 18).

        Answered ``--human-task <id>=skip`` and checked by
        :meth:`_partial_skip_evidence`. Nothing is dispatched: the outcome is
        the batch's own, ``applied``, with every item its report left out
        recorded as skipped at the user's request (``skipped_items``, never
        counted as applied) beside the items it declined. Its verdict is
        stamped again and the audit gains a completion line, so a later
        resume and ``util restore-batch`` read it as done (the pre-batch
        snapshot stays its rollback point); its summary joins the
        prior-changes log the next batch reads; and the loop goes on.
        """
        task_id = str(task.get("task_id", ""))
        kept = self._kept_at_request.pop(task_id)
        data, verdict = kept["data"], kept["verdict"]
        expected = [str(i) for i in (task.get("expected_ids") or [])]
        unapplied = [
            str(i) for i in (verdict.get("unapplied_ids") or []) if str(i) in expected
        ]
        outcome = TaskOutcome(
            task_id=task_id, kind=self.kind, runner=self.invoker,
            status=STATUS_OK, profile=profile_for_kind(self.kind),
            write_outcome=WRITE_APPLIED,
            summary=(str(data.get("summary") or "").strip() or None),
            runner_notes=(verdict.get("runner_notes") or None),
        )
        outcome.output_path = kept["path"]
        outcome.files_touched = [str(p) for p in (verdict.get("files_touched") or [])]
        outcome.expected_ids = expected
        outcome.returned_ids = [str(i) for i in (data.get("ids") or [])]
        outcome.sha256_before = dict(verdict.get("sha256_before") or {})
        outcome.sha256_after = dict(verdict.get("sha256_after") or {})
        outcome.declined = row_declined(verdict)
        outcome.skipped_items = {
            item: (
                f"--human-task {task_id}=skip: the batch did not report it "
                f"applied, and you chose to keep the batch as it stood"
            )
            for item in unapplied
        }
        read_verdict_checks(outcome, verdict)
        snapshot = os.path.join(self.safety_dir, "pre_batch", task_id)
        if os.path.isdir(snapshot):
            outcome.snapshot_dir = snapshot
        # Fix round 27: the stop after the last reason only where it has no
        # mark of its own (`with_closing_mark`).
        outcome.reason = with_closing_mark(
            f"KEPT AT YOUR REQUEST (--human-task {task_id}=skip): what it "
            f"applied stays as it left it"
            + (f" ({', '.join(outcome.files_touched)})" if outcome.files_touched else "")
            + f"; not applied, skipped at your request: "
            f"{self._declined_words(outcome.skipped_items)}"
            + (
                f"; declined by the batch: "
                f"{self._declined_words(outcome.declined)}"
                if outcome.declined else ""
            )
        ) + " It was not dispatched again."
        recorded = self.audit.completion_for(task_id, expected_ids=expected) or {}
        self.audit.completed(
            task_id=task_id, kind=self.kind, attempt=0, runner=self.invoker,
            profile=outcome.profile, status=STATUS_OK,
            write_outcome=WRITE_APPLIED, exit_code=None,
            files_touched=outcome.files_touched,
            observed=dict(recorded.get("observed") or {}),
            duration_seconds=0.0, reason=outcome.reason, expected_ids=expected,
        )
        self._stamp_verdict(outcome)
        self._record_batch_effects(
            task, outcome, batch_index, result, dispatched=False, kept=True,
        )
        return outcome

    def _refused(
        self, task: Dict[str, Any], reason: str, batch_index: int,
        result: SchedulerResult, *, status: str = STATUS_REJECTED,
        write_outcome: Optional[str] = None,
    ) -> TaskOutcome:
        """:meth:`_refuse_before_dispatch` for a refusal worded *reason*:
        the outcome no runner produced, in this phase's kind and profile."""
        return self._refuse_before_dispatch(task, TaskOutcome(
            task_id=str(task.get("task_id", "")), kind=self.kind,
            runner=self.invoker, status=status, profile=profile_for_kind(self.kind),
            reason=reason, write_outcome=write_outcome,
        ), batch_index, result)

    def _refuse_before_dispatch(
        self, task: Dict[str, Any], outcome: TaskOutcome, batch_index: int,
        result: SchedulerResult,
    ) -> TaskOutcome:
        """Record a task refused before any runner started, and return it.

        Fix round 13 (the check-gate fixture): a task ``driver.write_paths``
        refused before dispatch was never recorded — it stayed ``pending``
        with no reason while the task after it read "T005 failed" — so
        ``--human-task <id>=skip`` was refused as "a code task that did not
        fail" and changing the policy was the only way past it. It goes
        through the normal record path now (``_record_batch_effects``:
        ``failed`` with its reason in ``state.json``, the batch line), minus
        the metric, since no child ran; a ``--resume`` checks it again, so a
        policy the user widened lets it run, and ``--human-task
        <id>=done|skip|retry`` answers it like any failed task.
        """
        result.refusals.before_dispatch.add(outcome.task_id)
        self._record_batch_effects(
            task, outcome, batch_index, result, dispatched=False,
        )
        return outcome

    # -- the commit write policy (plans/review-commits.md §2.7) ---------

    def _commit_guard(self) -> Optional[Dict[str, Any]]:
        """The live apply guard of the state this run holds, or ``None`` (no
        pinned record with a guard, or not an apply-code-fixes phase)."""
        return _commit_write_guard.commit_guard_of(getattr(self.state, "state", None), self.kind)

    def _commit_protect(self, guard: Optional[Mapping[str, Any]]) -> None:
        """Add the guard's protected set to :attr:`policy` as its exact-path
        deny layer (``write_safety.WritePolicy.deny_exact``), with the rule
        text ``changed since reviewed commit <head7>``, and the context plan
        (``guard["context_plan"]``, §2.7: read-only in this review whether or
        not it changed) with :data:`write_safety.CONTEXT_PLAN_RULE`, so a
        batch that declares or writes it fails like a protected write. A
        symlinked context plan's in-repository targets
        (``guard["context_plan_targets"]``, ``commit_apply.
        context_plan_paths``) hold its text, so they get the same rule: the
        layer matches a path by its own name, never through the link.
        Idempotent: run at construction and again whenever a re-check adds
        files to the set."""
        if guard is None:
            return
        protected = guard.get("protected")
        if isinstance(protected, Mapping) and protected:
            self.policy.deny_exact(
                [str(path) for path in protected], _commit_write_guard.commit_protected_rule(guard),
            )
        targets = guard.get("context_plan_targets")
        names = [guard.get("context_plan")] + (
            list(targets) if isinstance(targets, Mapping) else [])
        plan_names = [
            name for name in names
            if isinstance(name, str) and name and not os.path.isabs(name)
            and name != ".." and not name.startswith("../")
        ]
        if plan_names:
            self.policy.deny_exact(plan_names, CONTEXT_PLAN_RULE)

    def _commit_watch_begin(self, task_id: str) -> None:
        """Read the disk ids of every path no batch may change (the
        protected set and the context plan, ``commit_apply.watched_ids``)
        just before *task_id* is dispatched, so its post-run check
        (:meth:`_commit_watch_after`) counts only what moved during its own
        window. No-op without a guard. Cost: one ``git hash-object`` call
        over that set, whatever the tree's size."""
        guard = self._commit_guard()
        if guard is None:
            return
        commit_apply = _commit_write_guard.load_commit_apply()
        with self._effects_lock:
            watched = list(commit_apply.watched_ids(guard))
        if watched:
            self._commit_watch_before[task_id] = commit_apply.disk_blob_ids(
                watched, cwd=self.repo_root,
            )

    def _commit_watch_after(self, task_id: str) -> Tuple[List[str], Any, Optional[Dict[str, Any]]]:
        """The watched paths *task_id*'s window changed: ``(paths,
        commit_apply, guard copy)``, ``([], None, None)`` without a guard.

        plans/review-commits.md §2.7, "Every observed write, after each
        batch": the protected set's (and the context plan's) current disk
        ids against the ones read before the batch (``commit_apply.
        protected_violations``), on its own, so the check never depends on
        the undeclared-write sweep. That sweep is abandoned on a tree over
        ``SWEEP_MAX_FILES`` (a large repository) and keeps no bytes past
        ``SWEEP_RETAIN_MAX_BYTES``; this one costs only the watched set. The
        deny layer gains every watched path first, so ``policy.check`` names
        each hit with its rule."""
        guard = self._commit_guard()
        if guard is None:
            return [], None, None
        commit_apply = _commit_write_guard.load_commit_apply()
        with self._effects_lock:
            # A co-scheduled sibling's re-check can grow the protected set.
            frozen = copy.deepcopy(guard)
            self._commit_protect(guard)
        hits = commit_apply.protected_violations(
            frozen, cwd=self.repo_root,
            baseline=self._commit_watch_before.get(task_id),
        )
        return [str(path) for path in hits or ()], commit_apply, frozen

    def _commit_restore_watched(
        self, task_id: str, report: Dict[str, str], commit_apply: Any,
        guard: Optional[Mapping[str, Any]],
    ) -> Dict[str, str]:
        """*report* (``TaskSnapshot.restore``'s) with each watched path the
        snapshot held no bytes for put back from what the guard kept
        (``commit_apply.restore_watched``: deleted when it was absent before
        the batch, else the guard's copy when that copy is what it held)."""
        if commit_apply is None or guard is None:
            return report
        missing = [
            path for path, state in report.items()
            if state == "skipped: not in this task's snapshot"
        ]
        if not missing:
            return report
        fallback = commit_apply.restore_watched(
            guard, missing, cwd=self.repo_root,
            baseline=self._commit_watch_before.get(task_id),
        )
        return {**report, **fallback}

    def _commit_groups(self, task: Mapping[str, Any]) -> Dict[str, List[str]]:
        """``{group id: [fix files]}`` for *task* (:func:`commit_group_files`
        over its batch's fixes in this phase's ``orchestrator_output.json``,
        which the apply phase writes before the scheduler runs)."""
        if self._commit_batches is None:
            self._commit_batches = _commit_write_guard.load_orchestrator_batches(self.phase_dir) or []
        return _commit_write_guard.commit_group_files(
            task, repo_root=self.repo_root,
            fixes=_commit_write_guard.batch_fixes_for(task, self._commit_batches),
        )

    def _commit_save_guard(self, commit_apply: Any, guard: Mapping[str, Any]) -> None:
        """Save *guard* back to ``state.json`` under ``state_file_lock``
        (``commit_apply.save_guard``). The guard is also this run's live
        dict, so the state's next ``save()`` writes the same decision; a
        failed save is named and the run goes on with it in memory."""
        path = getattr(self.state, "state_file", None)
        if not path:
            return
        try:
            commit_apply.save_guard(path, guard)
        except Exception as exc:  # noqa: BLE001 - the in-memory guard still holds
            self.log(
                f"[write-subagents] could not save the apply guard to {path}: "
                f"{exc}; it is saved with the next state write instead."
            )

    def _commit_recheck(
        self, task: Dict[str, Any], batch_index: int, result: SchedulerResult,
    ) -> Optional[TaskOutcome]:
        """The pre-dispatch re-check of one batch, or ``None`` when it passes
        (or no guard exists).

        Every fix file the batch declares (each group's, not only the
        primary one) is checked against the identity the guard saved for it
        (``commit_apply.recheck_files``: ``head`` by git's three reads,
        ``batch`` by the bytes' sha256), and against the protected set. A
        batch that fails is not dispatched:

        * its groups that name a failing file are skipped with ``file changed
          since reviewed commit <head7>`` (``commit_apply.mark_stale``, which
          also adds the failing files to the protected set), the guard is
          saved back, and the deny layer gains the files before the next
          batch;
        * its other groups are held back: the batch is recorded ``failed``
          with ``held back: batched with <file>`` through
          :meth:`_refuse_before_dispatch`, which a ``--resume`` re-checks
          and re-batches without the skipped groups (a batch every group of
          which is skipped is recorded the same way, with the skip as its
          reason). Every item of the batch is ``failed`` in
          ``processed_<phase>``, which keeps it in a resume's work set; the
          guard's saved skips then take the skipped ones out again and count
          them (``commit_apply.take_guard``);
        * its batch line is the ``[COMMIT_FIXES_STALE]`` line, and the loop
          goes on to the next batch (:meth:`_run` does not stop on it).
        """
        guard = self._commit_guard()
        if guard is None:
            return None
        commit_apply = _commit_write_guard.load_commit_apply()
        task_id = str(task.get("task_id", ""))
        groups = self._commit_groups(task)
        # Under the bookkeeping lock throughout: a co-scheduled sibling's
        # completion re-pins (mutates) the same guard, and its `state.save()`
        # serializes it, both under this lock.
        with self._effects_lock:
            self._commit_protect(guard)
            failing = _commit_write_guard.commit_recheck_failing(
                guard, groups, repo_root=self.repo_root, commit_apply=commit_apply,
            )
            if not failing:
                return None
            skipped, held = _commit_write_guard.commit_stale_split(groups, failing)
            commit_apply.mark_stale(guard, groups, failing, cwd=self.repo_root)
            self._commit_save_guard(commit_apply, guard)
            self._commit_protect(guard)
        line = _commit_write_guard.commit_stale_line(
            task_id, failing, guard,
            marker=getattr(commit_apply, "STALE_MARKER", "[COMMIT_FIXES_STALE]"),
        )
        tail = [f"{len(skipped)} fix(es) skipped"] if skipped else []
        if held:
            tail.append(
                f"{len(held)} held back, recorded failed (a --resume applies "
                f"{'it' if len(held) == 1 else 'them'} in a new batch)"
            )
        self._commit_stale[task_id] = f"{line} — not dispatched: {', '.join(tail)}"
        return self._refuse_before_dispatch(task, TaskOutcome(
            task_id=task_id, kind=self.kind, runner=self.invoker,
            status=STATUS_REJECTED, profile=profile_for_kind(self.kind),
            reason=_commit_write_guard.commit_stale_reason(failing, guard, held=bool(held)),
        ), batch_index, result)

    def _commit_repin(self, task: Dict[str, Any], outcome: TaskOutcome) -> None:
        """Re-pin a completed batch's fix files to the bytes it left
        (``commit_apply.repin_after_batch``, from the ``sha256_after`` of its
        audit record), so a later batch on the same file, or a ``--resume``,
        accepts them by their ``batch`` identity, and a hand edit made after
        it fails. Only the batch's own fix files: a file another batch
        declares that moved during this one's window (an edit made while it
        ran) keeps its expectation, so its own batch is still re-checked
        against it. Caller holds ``_effects_lock``; no-op without a guard."""
        if not outcome.ok:
            return
        guard = self._commit_guard()
        if guard is None:
            return
        own = {path for files in self._commit_groups(task).values() for path in files}
        after = {
            str(path): digest for path, digest in (outcome.sha256_after or {}).items()
            if str(path) in own
        }
        if not after:
            return
        commit_apply = _commit_write_guard.load_commit_apply()
        if commit_apply.repin_after_batch(
            guard, str(task.get("task_id", "")), after, cwd=self.repo_root,
        ):
            self._commit_save_guard(commit_apply, guard)

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
        """Fingerprint source files, including code beside workflow artifacts."""
        from .check_tree import check_tree_exclusions

        excluded = check_tree_exclusions(
            Path(self.phase_dir), self.repo_root, output_dir=self.output_dir,
        )
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
        if self._executor_dispatch and not outcome.attempts:
            # `dispatch_task` refused before its first `before_attempt`, so no
            # `dispatched` line was written either. A lone completion here would
            # claim a run that never started.
            return
        undeclared = self._undeclared_writes(outcome.task_id, snapshot)
        observed = self._observed_with_undeclared(snapshot, undeclared)
        # §2.7: a protected path this window changed is put back here too,
        # before the byte comparison below reads the snapshot.
        self._commit_protected_unclassified(outcome, snapshot)
        if not outcome.files_touched:
            outcome.files_touched = snapshot.changed()
        if not outcome.unrestorable:
            outcome.unrestorable = list(undeclared.unrecoverable)
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

    def _commit_protected_unclassified(
        self, outcome: TaskOutcome, snapshot: TaskSnapshot
    ) -> None:
        """plans/review-commits.md §2.7 on a dispatch that never reached
        ``_classify`` (a timeout, a runner error, a reply that never parsed):
        each watched path its window changed (:meth:`_commit_watch_after`) is
        put back as ``_classify``'s policy branch puts it back, from the
        pre-batch snapshot or else from what the guard kept, and its reason
        names them. Its verdict stays its own: the loop stops on it either
        way. No-op without a guard."""
        watched, commit_apply, guard = self._commit_watch_after(outcome.task_id)
        if not watched or guard is None:
            return
        report = self._commit_restore_watched(
            outcome.task_id, snapshot.restore(watched), commit_apply, guard,
        )
        back = sorted(p for p, state in report.items() if not state.startswith("skipped:"))
        many = len(watched) != 1
        outcome.reason = (
            f"{outcome.reason or outcome.status} It also changed "
            f"{', '.join(watched)}, which the commit review protects (this apply "
            f"never writes {'them' if many else 'it'})."
            + (f" Put back as before the batch: {', '.join(back)}." if back else "")
            # No rollback point, or the restore's own reason: never one for the other.
            + PolicyRestore.from_report(report).not_restored()
        )
        self.audit.restored(task_id=outcome.task_id, report=report, force=False)

    def _partial_when_bytes_moved(
        self, outcome: TaskOutcome, snapshot: TaskSnapshot
    ) -> None:
        """A failed dispatch that never reached ``_classify`` and moved bytes
        is a partial application (``write_outcomes.partial_when_bytes_moved``,
        over the byte comparison :meth:`_audit_terminal` just filled)."""
        _write_outcomes.partial_when_bytes_moved(
            outcome, snapshot_dir=snapshot.directory)

    # -- D3l post-run containment: the OBSERVED set, not the declared one ---

    def _begin_sweep(
        self, task_id: str, declared: Sequence[str], *,
        verification_only: bool = False,
    ) -> None:
        """Open this dispatch's sweep (:meth:`ContainmentSweep.begin`), then
        publish it in flight.

        Raises :class:`WriteSafetyError` for a ``verification_only`` task when
        the census is unusable: never VERIFIED without a census.
        """
        self._sweep.begin(task_id, declared, verification_only=verification_only)
        # F10: the mirror's own writes only happen at a batch's boundaries
        # (`WriteReports.mirror_status`), which says nothing while a single-task batch is
        # still running — measured 2026-09-16, T001 ran 659s as one batch and
        # `.status.json` read `dispatched: 0` the entire time. This is the one
        # place a dispatch's start time is known, so it patches the mirror's
        # `in_flight` list directly rather than waiting for the next full
        # recompute; that next call (whenever it lands) replaces the whole
        # `write_subagents` object, `in_flight` included, from
        # the sweep's in-flight entries afresh, so a stale patch here never lingers.
        self._reports.publish_in_flight()

    def _undeclared_writes(
        self, task_id: str, snapshot: TaskSnapshot
    ) -> _write_sweep.UndeclaredWrites:
        """Close this dispatch's sweep (:meth:`ContainmentSweep.undeclared_writes`).
        The seam ``_classify`` and ``_audit_terminal`` call, so a test can
        stand in for the sweep on a scheduler it built by hand."""
        return self._sweep.undeclared_writes(task_id, snapshot)

    def _observed_with_undeclared(
        self, snapshot: TaskSnapshot, undeclared: _write_sweep.UndeclaredWrites
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
        runner = self._runner_for(task)
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
            self._prompts.keep_dispatched_prompt(
                task_id, int(context["attempt"]), sent, prompt
            )

        def after_result(context: Dict[str, Any]) -> None:
            # The executor's own record of the rung this attempt ran on wins
            # over the preflight's: every `after_result` context carries it,
            # `None` included (`subagent_executor._jail_hook_context`).
            seen = context["sandbox_rung"]
            self._dispatch_rungs[task_id] = seen
            self._classify(task, context["outcome"], snapshot, context["result"],
                           runner=runner, profile=profile,
                           attempt=context["attempt"],
                           exit_code=context["exit_code"],
                           duration=context["duration_seconds"],
                           sandbox_rung=seen,
                           jail_denials=context.get("jail_denials"),
                           resent=bool(context.get("resent")),
                           resend_note=context.get("resend_note"))

        def nothing_changed(context: Dict[str, Any]) -> Optional[str]:
            # Rule 9 (fix round 14): asked before `after_result`, read-only,
            # so a re-send leaves the census for the classification that
            # follows it.
            return self._applied_without_changes(task, snapshot, context)

        def incomplete_report(context: Dict[str, Any]) -> Optional[str]:
            # Fix round 18: asked before `after_result`, read-only, like rule
            # 9's question; a reason sends the one report re-ask.
            return self._incomplete_report(task, snapshot, context)

        def restate_findings(
            context: Dict[str, Any],
        ) -> Optional[Tuple[str, List[str]]]:
            # Fix round 28: asked the same way; never a classification.
            return self._restate_findings(task, context)

        def changed_files(context: Dict[str, Any]) -> List[str]:
            # T4.8 (plan opencode-validator-false-rate-limit-and-denied-turn
            # §4.4 step 4): a turn a denial ended has no result, so `_classify`
            # never runs and `dispatch_task` would retry on top of whatever it
            # half-applied. Asked before that retry decision, against the same
            # pre-batch snapshot and pre-run census `_classify` reads.
            return self._sweep.moved_since_dispatch(context["task_id"], snapshot)

        def tree_fingerprint(context: Dict[str, Any]) -> Dict[str, Optional[str]]:
            # A report re-ask's guard: the current bytes of every declared file
            # and of every path this dispatch has moved so far (the same two
            # sources `_classify` decides on, peeked). The executor compares the
            # value before and after the re-ask; any difference is an edit the
            # re-ask was told not to make.
            moved = self._sweep.moved_since_dispatch(context["task_id"], snapshot)
            paths = set(moved) | {entry.relative for entry in snapshot.entries}
            return {
                relative: file_sha256(os.path.join(self.repo_root, relative))
                for relative in sorted(paths)
            }

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
            jail_request=request,
            tree_fingerprint=tree_fingerprint,
            nothing_changed=nothing_changed,
            incomplete_report=(
                incomplete_report if self.kind in APPLY_WRITE_KINDS else None
            ),
            restate_findings=(
                restate_findings if self.kind == "implement" else None
            ),
        )

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
        from .provider_registry import get_provider
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
                return list(provider.apply_sandbox_fallback(
                    list(argv), profile, rung_id, consent=consent,
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
        resent: bool = False,
        resend_note: Optional[str] = None,
    ) -> None:
        """Decide which D3g outcome this dispatch landed on, and record it.

        The decision is :func:`write_verdict.decide`, a table read in order:
        the write policy over every touched path, then the plugin-landlock
        rung's replaced target and jail denials, then a task that declares no
        files, then the claim rows, where the claim is never the verdict and a
        non-zero exit never lands on APPLIED (D3k). The rows and why:
        ``docs/development/write-scheduler.md``, "The verdict table".

        This method gathers the facts, asks the table and applies the verdict.
        The guards' facts are gathered in the table's order, each only while
        no guard before it has matched (:func:`write_verdict.guard`), so
        reading one a later row needs (the target's inode, a jail path
        resolved) never stands between a matched guard and what its row does.
        Every side effect is here, around the pure call: the containment sweep
        is consumed exactly once, a policy violation's files are restored
        before its verdict is set and audited, the jail's records are kept,
        and one audit line closes the dispatch.
        *resent* says the batch was sent once more after an unchanged
        ``applied`` (the ladder's rule 9), *resend_note* why it was not. A
        prose result never reaches here: it already failed rung 6 as
        ``schema_invalid``, the fourth outcome.
        """
        task_id = outcome.task_id
        claimed = str((result or {}).get("status") or "")
        summary = str((result or {}).get("summary") or "").strip()
        notes = str((result or {}).get("notes") or "").strip()
        outcome.runner_notes = clip(notes, 600) or None
        reported_ids = [str(i) for i in ((result or {}).get("ids") or [])]
        expected_ids = [str(i) for i in (task.get("expected_ids") or [])]
        # An apply batch's declined item is not applied and not missing; an
        # id it put in both `ids` and `declined` is neither. `implement` has
        # no decline: a reply declining its task did not do it (`failed`).
        declined = declined_items(result, expected_ids)
        declined_task: Optional[str] = None
        if self.kind in APPLY_WRITE_KINDS:
            outcome.declined = dict(declined.items)
        elif declined.items:
            declined_task = next(iter(declined.items.values()))
            claimed = "failed"
            outcome.runner_notes = outcome.runner_notes or (
                f"declined: {declined_task}"
            )
        # `changed` is the DECLARED verdict: the byte comparison taken before
        # the sweep adopts this dispatch's undeclared writes into the snapshot.
        changed = snapshot.changed()
        undeclared = self._undeclared_writes(task_id, snapshot)
        observed = self._observed_with_undeclared(snapshot, undeclared)
        # plans/review-commits.md §2.7: under an apply guard, the protected
        # set's (and the context plan's) own post-run check, which never
        # depends on the sweep above. Empty without a guard.
        watched, commit_apply, watched_guard = self._commit_watch_after(task_id)

        # F9: the verification evidence rides the outcome, which both batch
        # reports are built from.
        outcome.expected_ids = expected_ids
        outcome.returned_ids = reported_ids
        # Whatever row decides it, a file-less implement task's verdict reads
        # VERIFIED or PARTIAL on every surface (`outcome_label`).
        outcome.verification_only = self._verification_only(snapshot)
        self._record_reply_checks(task, outcome, result)
        outcome.sha256_before = {
            relative: row.get("sha256_before") for relative, row in observed.items()
        }
        outcome.sha256_after = {
            relative: row.get("sha256_after") for relative, row in observed.items()
        }
        # `files_touched` is the ROLLBACK set: the declared files plus the
        # undeclared writes the sweep retained bytes for. `util restore-batch`
        # reads it, so a path with no rollback point is left out and
        # `unrestorable` carries it (`TaskOutcome.bytes_moved` reads both).
        outcome.files_touched = list(changed) + list(undeclared.restorable)
        outcome.unrestorable = list(undeclared.unrecoverable)
        outcome.summary = summary or None
        if undeclared:
            self._log_undeclared(task_id, undeclared)

        facts = WriteFacts(
            task_id=task_id,
            kind=self.kind,
            claimed=claimed,
            exit_code=exit_code,
            changed=tuple(changed),
            expected_ids=tuple(expected_ids),
            reported_ids=tuple(reported_ids),
            attested_ids=tuple(
                i for i in reported_ids if i not in declined.conflicting
            ),
            declined=dict(outcome.declined),
            declined_task=declined_task,
            declined_problems=tuple(declined.problems),
            undeclared=tuple(undeclared.paths),
            undeclared_own=tuple(undeclared.own),
            siblings=tuple(sorted(undeclared.shared)),
            sweep_observed=undeclared.observed,
            verification_only=outcome.verification_only,
            summary=summary,
            snapshot_dir=str(snapshot.directory),
            snapshot_reused=task_id in self._snapshot_reused,
            resent=resent,
            resend_note=resend_note,
            repo_root=str(self.repo_root),
            sweep_max_files=_write_sweep.SWEEP_MAX_FILES,
        )
        # The guards' facts, in the table's order, each gathered only while no
        # guard before it has matched: reading a later one can fail (its
        # paths are resolved), and must never stand between a matched guard
        # and what its row does, a policy violation's restore above all.
        violation = self._post_run_violation(changed, undeclared, watched)
        if violation is not None:
            facts = replace(
                facts, policy=PolicyFacts(violation.message, tuple(violation.paths)),
            )
        if guard(facts) is None:
            # This dispatch's "before", spent by every row past the policy's.
            before = self._target_identity_before.pop(task_id, None)
            if sandbox_rung == SANDBOX_RUNG_PLUGIN_LANDLOCK:
                facts = replace(facts, replaced=self._target_replacement(before))
        if guard(facts) is None:
            denials = _as_jail_denials(jail_denials)
            if denials is not None:
                # Kept before any of its paths is resolved.
                self._jail_denials[task_id] = denials.to_dict()
                facts = replace(facts, jail=self._jail_facts(task_id, denials, facts))
                note = jail_note(facts.jail) if guard(facts) is None else None
                if note:
                    self._jail_notes[task_id] = note
        reply_check: Optional[_executor.CheckReply] = None
        if guard(facts) is None:
            reply_check = read_check_reply(result)
            facts = replace(
                facts,
                check=(
                    None if reply_check.findings is None
                    else tuple(reply_check.findings)
                ),
                check_listed=reply_check.listed,
                check_could_not_run=reply_check.could_not_run,
                declined_words=self._declined_words(outcome.declined),
                named_files=tuple(
                    str(path) for path in ((result or {}).get("files_touched") or [])
                    if isinstance(path, str)
                ),
            )
        verdict = decide(facts)
        if verdict.rule in DECLARED_NAMES_ROWS:
            facts = replace(facts, declared_names=(
                ", ".join(e.relative for e in snapshot.declared_entries())
                or "any declared file"
            ))

        restore: Optional[PolicyRestore] = None
        if verdict.rule is VerdictRule.POLICY_VIOLATION:
            assert violation is not None
            report = snapshot.restore(violation.paths)
            # A protected path the snapshot holds no bytes for is put back
            # from what the guard kept (§2.7); no-op without a guard.
            report = self._commit_restore_watched(
                task_id, report, commit_apply, watched_guard,
            )
            restore = PolicyRestore.from_report(report, {
                getattr(commit_apply, name, None)
                for name in ("RESTORED_FROM_COPY", "DELETED_ABSENT_BEFORE")
            } - {None})
        elif verdict.rule is VerdictRule.REPLACED_TARGET:
            self._record_replaced_target(outcome)
        elif verdict.rule is VerdictRule.JAIL_FAILED:
            self._jail_failed.add(task_id)
        if (
            verdict.check_not_run and reply_check is not None
            and reply_check.findings and not outcome.runner_notes
        ):
            # A check that did not run: what its reply listed is the stop's
            # account of it, never findings the run goes on past.
            outcome.runner_notes = clip("; ".join(reply_check.findings), 600)
        # The verdict is on the outcome before anything below can fail: a
        # restored violation is never left unclassified by its audit line.
        outcome.status = verdict.status
        outcome.write_outcome = verdict.write_outcome
        if verdict.applied_without_changes:
            outcome.applied_without_changes = True
        if verdict.unapplied_ids:
            outcome.unapplied_ids = list(verdict.unapplied_ids)
        if verdict.rule is VerdictRule.CHECK_FINDINGS:
            outcome.check_findings = list(facts.check or [])
        elif verdict.rule is VerdictRule.VERIFIED:
            outcome.check_findings = None if facts.check is None else list(facts.check)
        reason = reason_for(verdict, facts, restore)
        if reason is not None:
            outcome.reason = reason
        if verdict.rule is VerdictRule.POLICY_VIOLATION:
            self.audit.restored(task_id=task_id, report=report, force=False)
        elif verdict.rule is VerdictRule.APPLIED_NO_OP_CLAIMED and not facts.snapshot_reused:
            self.log(
                f"[write-subagents] warning: {task_id} claimed "
                f"'no-op' but {', '.join(changed)} changed against a snapshot "
                f"taken this dispatch — treating the bytes as "
                f"authoritative."
            )
        self._audit_completion(outcome, runner, profile, attempt, exit_code,
                               observed, duration)

    def _record_reply_checks(
        self, task: Dict[str, Any], outcome: TaskOutcome, result: Any,
    ) -> None:
        """The checks a code batch's reply says it ran on its own edits, on
        the outcome: its findings (fix round 14), its own words about them,
        never a finding (fix round 28), and, for apply-code-fixes, what it
        says each known check it was told to run did (fix round 20)."""
        if self.kind in _batch_prompt.CODE_WRITE_KINDS and not outcome.verification_only:
            outcome.batch_check = batch_check_findings(result)
            outcome.batch_check_note = batch_check_note(result)
            if self.kind == "implement" and outcome.batch_check:
                # What only repeats findings the user accepted for a check is
                # that check's, never this task's own failure: only the checks
                # this task was told were accepted, and none whose failure is
                # its own job (fix rounds 22, 23).
                own, matched = _check_records.split_accepted_findings(
                    outcome.batch_check,
                    _check_records.accepted_findings_for(self.tasks, task, self.state),
                )
                if matched:
                    outcome.batch_check_accepted = matched
                    outcome.batch_check = own or None
        if self.kind == "apply_code_fixes":
            # Read against the commands its prompt listed; a command it lists
            # as failed is a failing check on every surface.
            listed = _known_checks.listed_check_outcomes(
                _known_checks.reply_listed(result),
                self._told_checks.get(outcome.task_id) or (),
            )
            self._listed_outcomes[outcome.task_id] = listed
            outcome.batch_check = _known_checks.with_listed_failures(
                outcome.batch_check, listed,
            )

    def _log_undeclared(self, task_id: str, undeclared: _write_sweep.UndeclaredWrites) -> None:
        stranded = undeclared.unrecoverable
        self.log(
            f"[write-subagents] {task_id}: {len(undeclared)} file(s) "
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

    def _post_run_violation(
        self, changed: Sequence[str], undeclared: _write_sweep.UndeclaredWrites,
        watched: Sequence[str],
    ) -> Optional[PolicyViolation]:
        """D3l post-run: the write policy over the touched set, or ``None``.

        The touched set is the declared files whose bytes moved, whatever
        else under the pinned root the sweep saw move (declared or not: the
        sweep retained its pre-run bytes, so a restore covers an undeclared
        write too, :meth:`_undeclared_writes`), and the watched paths the
        sweep did not see. Enforcing on ``changed`` alone is vacuous: every
        path in it passed the identical pre-dispatch check.
        """
        swept = set(changed) | set(undeclared.paths)
        try:
            self.policy.check(
                list(changed) + undeclared.paths
                + [path for path in watched if path not in swept],
                when="post-run",
            )
        except PolicyViolation as exc:
            return exc
        return None

    def _declined_words(self, declined: Dict[str, str]) -> str:
        """``fix 1 (`<id>`) — <reason>; …``: an apply batch's declined items,
        numbered as its result contract numbers them (fix round 18)."""
        return _check_records.declined_words(self.kind, self._task_ids_in_order(), declined)

    def _unapplied_words(self, items: Sequence[str]) -> str:
        """``fix 1 (`<id>`)`` for each item a partial application left out."""
        positions = self._task_ids_in_order()
        word = _check_records.item_word(self.kind)
        return ", ".join(
            f"{word} {positions[str(i)][1]} (`{i}`)" if str(i) in positions
            else f"`{i}`"
            for i in items
        )

    def _keep_partial_clause(self, outcome: TaskOutcome) -> str:
        """What ``--human-task <batch>=skip`` does, in one clause."""
        return (
            f"keep what {outcome.task_id} applied and go on without "
            f"{self._unapplied_words(outcome.unapplied_ids)}: "
            f"{'it is' if len(outcome.unapplied_ids) == 1 else 'they are'} "
            f"recorded as not applied, skipped at your request, and the "
            f"batches after it run"
        )

    def _task_ids_in_order(self) -> Dict[str, Tuple[str, int]]:
        """``member id -> (task id, 1-based position)`` over this run's tasks."""
        positions: Dict[str, Tuple[str, int]] = {}
        for task in self.tasks:
            for index, member in enumerate(task.get("expected_ids") or [], 1):
                positions.setdefault(str(member), (str(task.get("task_id", "")), index))
        return positions

    def _applied_without_changes(
        self, task: Dict[str, Any], snapshot: TaskSnapshot, context: Dict[str, Any],
    ) -> Optional[str]:
        """Rule 9's measurement: why a schema-valid reply that said ``applied``
        moved not one byte, or ``None`` when that is not what happened.

        Read-only — the census stays registered for the classification —
        and asked before ``_classify`` (fix round 14). ``None`` whenever the
        classification decides on something else, or a re-send could land on
        moved bytes: a claim other than ``applied``, a check task (it changes
        nothing by design, and its verdict is VERIFIED or check-findings), a
        non-zero exit, a jail denial, a replaced target, a sweep that could
        not look (the tree outgrew the census), or any path — declared or
        not — that moved.
        """
        result = context.get("result")
        if not isinstance(result, dict) or str(result.get("status") or "") != "applied":
            return None
        if self._verification_only(snapshot):
            return None
        expected = [str(i) for i in (task.get("expected_ids") or [])]
        declined = declined_items(result, expected).items
        if expected and all(item in declined for item in expected):
            # Fix round 18: it declined every item, so nothing was meant to
            # land; sending it again would ask for what it refused.
            return None
        if context.get("exit_code") not in (0, None):
            return None
        denials = _as_jail_denials(context.get("jail_denials"))
        if denials is not None and denials.total:
            return None
        task_id = str(task.get("task_id", ""))
        looked = self._sweep.has_census(task_id)
        if not looked:
            return None
        if context.get("sandbox_rung") == SANDBOX_RUNG_PLUGIN_LANDLOCK:
            before = self._target_identity_before.get(task_id)
            if before is not None and self._identity_of(self._identity_path()) != before:
                return None
        if snapshot.changed():
            # A declared file moved: applied, whatever else moved beside it.
            # Decided before the census walk below, which could only agree.
            return None
        moved = self._sweep.moved_since_dispatch(task_id, snapshot)
        outside: List[str] = []
        if moved and self.kind in SHARED_ARTIFACT_WRITE_KINDS:
            # Fix round 15: for a plan or task-list batch the declared file
            # is the whole contract, so a stray write beside an unchanged
            # plan does not make it applied, and the plan's own bytes cannot
            # be applied twice by a re-send. Told which files moved instead.
            outside = list(moved)
            context["moved_outside"] = outside
        elif moved:
            return None
        declared = (
            ", ".join(e.relative for e in snapshot.declared_entries())
            or "the files it declares"
        )
        named = [
            str(path) for path in (result.get("files_touched") or [])
            if isinstance(path, str)
        ]
        verb = 'is' if len(snapshot.declared_entries()) == 1 else 'are'
        return (
            f"the reply said 'applied'"
            + (f" and named {', '.join(named[:5])} as edited" if named else "")
            + (
                f", but {declared} {verb} byte-for-byte as before the dispatch "
                f"(only {', '.join(outside[:5])}"
                f"{', …' if len(outside) > 5 else ''} changed, outside it)"
                if outside else
                f", but no file changed: {declared} {verb} byte-for-byte as "
                f"before the dispatch"
            )
        )

    def _incomplete_report(
        self, task: Dict[str, Any], snapshot: TaskSnapshot, context: Dict[str, Any],
    ) -> Optional[str]:
        """Why an apply batch's readable report, after edits that moved
        bytes, leaves some of its items out of both ``ids`` and ``declined``
        — the one problem its report re-ask names — or ``None`` (fix round 18).

        Read-only, asked before ``_classify``. ``None`` whenever the
        classification would not read the reply as a partial application for
        that reason: a ``failed`` claim, a non-zero exit, a jail denial (its
        verdict is read off this stream), every item accounted for, or no byte
        moved (for a plan or task-list batch, its declared file).
        """
        if self.kind not in APPLY_WRITE_KINDS:
            return None
        result = context.get("result")
        if not isinstance(result, dict) or str(result.get("status") or "") == "failed":
            return None
        if context.get("exit_code") not in (0, None):
            return None
        denials = _as_jail_denials(context.get("jail_denials"))
        if denials is not None and denials.total:
            return None
        expected = [str(i) for i in (task.get("expected_ids") or [])]
        raw = result.get("ids")
        reported = [str(i) for i in raw] if isinstance(raw, list) else []
        declined = declined_items(result, expected)
        attested = {i for i in reported if i not in declined.conflicting}
        missing = [
            i for i in expected if i not in attested and i not in declined.items
        ]
        if not missing:
            return None
        task_id = str(task.get("task_id", ""))
        moved = bool(snapshot.changed()) or (
            self.kind not in SHARED_ARTIFACT_WRITE_KINDS
            and bool(self._sweep.moved_since_dispatch(task_id, snapshot))
        )
        if not moved:
            return None
        accounted = len(expected) - len(missing)
        return (
            f"The report accounts for {accounted} of the batch's "
            f"{len(expected)} items: {self._unapplied_words(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} in neither `ids` nor "
            f"`declined`. Every item goes in exactly one of them: `ids` if the "
            f"previous turn applied it (or found it already applied), "
            f"`declined` with the reason it gave if it left the item unapplied "
            f"on purpose."
            + (
                f" Also: {'; '.join(declined.problems[:5])}."
                if declined.problems else ""
            )
        )

    def _restate_findings(
        self, task: Dict[str, Any], context: Dict[str, Any],
    ) -> Optional[Tuple[str, List[str]]]:
        """``(problem, accepted findings)`` when an implement task's readable
        report has an own-check finding that holds a finding the user
        accepted, word for word, with words added
        (:func:`accepted_findings_with_words_added`) — the one problem its
        report re-ask names and what such a finding may be cut down to — or
        ``None`` (fix round 28, live run 12 stop accept 02).

        Read-only, asked before ``_classify``, and never a classification:
        the re-ask's restated report is classified like any. ``None`` for a
        check task, a ``failed`` claim, a non-zero exit, a jail denial, a
        task released by no accepted check, or no such finding."""
        if self.kind != "implement" or _batch_prompt.is_check_task(self.kind, task):
            return None
        result = context.get("result")
        if not isinstance(result, dict) or str(result.get("status") or "") == "failed":
            return None
        if context.get("exit_code") not in (0, None):
            return None
        denials = _as_jail_denials(context.get("jail_denials"))
        if denials is not None and denials.total:
            return None
        findings = batch_check_findings(result)
        if not findings:
            return None
        accepted = _check_records.accepted_findings_for(self.tasks, task, self.state)
        if not accepted:
            return None
        own, _matched = _check_records.split_accepted_findings(findings, accepted)
        flagged = _check_records.accepted_findings_with_words_added(own, accepted)
        if not flagged:
            return None
        task_id = str(task.get("task_id", ""))
        return (
            _check_records.restate_findings_problem(task_id, flagged),
            [accepted_text for _finding, _check, accepted_text in flagged],
        )

    def _verification_only(self, snapshot: TaskSnapshot) -> bool:
        """Whether this dispatch declared no files at all (``implement`` only)."""
        return self.kind == "implement" and not snapshot.declared_entries()

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

    def _target_replacement(
        self, before: Optional[Tuple[int, int]],
    ) -> Optional[TargetReplaced]:
        """The replaced-target check (F1 "The target replaced during the
        dispatch"): the watched target's identity now against *before*, the
        one recorded at this dispatch's pre-batch snapshot. ``None`` when it
        is the same file, or there is nothing to compare.

        Detection only: the plugin never guesses which process swapped the
        file, and never widens the grant to the directory to ride a swap out.
        """
        path = self._identity_path()
        if path is None or before is None:
            return None
        after = self._identity_of(path)
        if after == before:
            return None
        return TargetReplaced(
            label=self._target_label(),
            shown=self._relative(path),
            before_inode=before[1],
            after_inode=None if after is None else after[1],
        )

    def _record_replaced_target(self, outcome: TaskOutcome) -> None:
        """A replaced target's records. No rollback over the new file:
        ``files_touched`` is what ``util restore-batch`` restores, so the
        target (and, under the dir rule, anything beneath it) leaves it; the
        pre-batch snapshot stays on disk for comparison."""
        outcome.files_touched = [
            relative for relative in outcome.files_touched
            if not self._in_grant(os.path.join(self.repo_root, relative))
        ]
        self._jail_failed.add(outcome.task_id)
        self._replaced_targets.add(outcome.task_id)

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
        is never restored. The verdict is the replaced-target row's
        (:func:`write_verdict.replaced_target_reason`), naming the dispatch's
        own ending in the claim's place, with its failure kept at the end.
        """
        task_id = outcome.task_id
        if self._dispatch_rungs.get(task_id) != SANDBOX_RUNG_PLUGIN_LANDLOCK:
            return False
        replaced = self._target_replacement(
            self._target_identity_before.pop(task_id, None)
        )
        if replaced is None:
            return False
        ended, detail = outcome.status, outcome.reason
        self._record_replaced_target(outcome)
        outcome.status = STATUS_BATCH_FAILED
        outcome.write_outcome = WRITE_CLEAN_FAILURE
        outcome.reason = replaced_target_reason(
            replaced, str(snapshot.directory), "", ended=ended,
        )
        if detail:
            outcome.reason = f"{outcome.reason} The dispatch's own failure: {detail}"
        return True

    def _jail_facts(
        self, task_id: str, denials: Any, facts: WriteFacts,
    ) -> JailFacts:
        """The jail's denials (F1 "A denied edit is never recorded as
        applied") as the verdict reads them, every path repo-relative.

        The row's facts name no path. Each path is resolved after them, and
        only for what *facts* with them land on: the denied paths and the
        scope sentence when a jail row decides, the note's when none does.
        """
        replaced = list(denials.replacement_paths)
        jail = JailFacts(
            hard=bool(denials.failed_file_changes or denials.in_scope_paths),
            replacements=len(replaced),
        )
        if guard(replace(facts, jail=jail)) is None:
            return replace(
                jail,
                replacement=tuple(self._relative(p) for p in replaced),
                out_of_scope_count=denials.out_of_scope_count,
                out_of_scope=(
                    tuple(self._relative(p) for p in denials.out_of_scope_paths)
                    if denials.out_of_scope_count else ()
                ),
            )
        denied = denials.denied_paths()
        paths = [self._relative(p) for p in denied]
        in_grant = [
            self._relative(p) for p in denied
            if self._in_grant(p, task_id) or p in replaced
        ]
        return replace(
            jail,
            denied=tuple(paths),
            scope_sentence=self._jail_scope_sentence(
                in_grant, [p for p in paths if p not in in_grant],
            ),
        )

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

    def _remember_outcome(self, outcome: TaskOutcome) -> None:
        """Keep *outcome* by task id for :meth:`_move_repeats_of`, which may
        run before the outcome's batch joins ``result.outcomes`` (a check
        and a code task in one batch). Caller holds the effects lock."""
        self._recorded_outcomes.setdefault(outcome.task_id, []).append(outcome)

    def _split_check_repeats(self, task: Dict[str, Any], outcome: TaskOutcome) -> None:
        """Move out of a finished code task's own failing check what is the
        same finding a check task of the run already reported
        (:func:`repeated_findings_for`, the strict comparison), into
        ``batch_check_repeats``, and re-stamp its verdict (fix round 28, the
        code task recording second). Caller holds the effects lock."""
        if not outcome.ok or outcome.verification_only or not outcome.batch_check:
            return
        repeats = _check_records.repeated_findings_for(self.tasks, task, self.state)
        if not repeats:
            return
        own, matched = _check_records.split_repeated_findings(outcome.batch_check, repeats)
        if not matched:
            return
        outcome.batch_check = own or None
        outcome.batch_check_repeats = {**outcome.batch_check_repeats, **matched}
        self._restamp_own_check(outcome.output_path, outcome.batch_check,
                                outcome.batch_check_repeats)

    def _move_repeats_of(
        self, check_id: str,
        moved: Mapping[str, Tuple[List[str], Dict[str, List[str]]]],
        result: SchedulerResult,
    ) -> None:
        """The check task recording second (fix round 28): what
        :func:`move_check_repeats_in_state` *moved* in ``state.json`` (the
        record ``write_record`` wrote) is moved on the code task's outcome,
        its earlier run's row in *result* and its verdict stamp too, with one
        log line. Caller holds the effects lock."""
        from .subagent_tasks import implement_output_name
        seen = self._recorded_outcomes
        for task_id, (own, merged) in moved.items():
            path: Optional[str] = None
            for outcome in [*result.outcomes, *seen.get(task_id, [])]:
                if outcome.task_id == task_id and outcome.ok:
                    outcome.batch_check = own or None
                    outcome.batch_check_repeats = dict(merged)
                    path = outcome.output_path or path
            for row in result.earlier_rows:
                if isinstance(row, dict) and str(row.get("task_id")) == task_id:
                    if own:
                        row["batch_check"] = check_record(own)
                    else:
                        row.pop("batch_check", None)
                    row["batch_check_repeats"] = {k: list(v) for k, v in merged.items()}
            self._restamp_own_check(
                path or os.path.join(self.phase_dir, implement_output_name(task_id)),
                own or None, merged,
            )
            self.log(
                f"[write-subagents] {task_id}'s own check reported only what check "
                f"task {check_id} reported: recorded as repeating {check_id}'s "
                f"finding, not as {task_id}'s own failure"
            )

    def _restamp_own_check(
        self, output_path: Optional[str], batch_check: Optional[List[str]],
        repeats: Mapping[str, Sequence[str]],
    ) -> None:
        """Re-write a result file's verdict ``batch_check`` and
        ``batch_check_repeats`` keys after :meth:`_split_check_repeats` or
        :meth:`_move_repeats_of` moved findings (fix round 28), so a resume
        skip reads what the records say; every other key is left alone."""
        if not output_path:
            return
        path = Path(output_path)
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            verdict = data.get("verdict") if isinstance(data, dict) else None
            if not isinstance(verdict, dict):
                return
            if batch_check is not None:
                verdict["batch_check"] = check_record(batch_check)
            else:
                verdict.pop("batch_check", None)
            if repeats:
                verdict["batch_check_repeats"] = {
                    str(check_id): list(found) for check_id, found in repeats.items()
                }
            else:
                verdict.pop("batch_check_repeats", None)
            temp = path.with_name(path.name + ".tmp")
            temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(str(temp), str(path))
        except (OSError, ValueError) as exc:  # noqa: BLE001 - bookkeeping
            self.log(
                f"[write-subagents] warning: could not re-record the own check "
                f"on {path.name}: {exc}"
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
            # `batch_ledger.verdict_record`: what the stamp holds and why,
            # read back by a resume skip (`outcome_from_verdict`).
            data["verdict"] = verdict_record(
                outcome,
                task=self._task_by_id(outcome.task_id),
                inputs_fingerprint=self._task_fingerprint,
                tree_fingerprint=self._tree_fingerprint,
                sandbox_rung=self._dispatch_rungs.get(outcome.task_id),
            )
            temp = path.with_name(path.name + ".tmp")
            temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(str(temp), str(path))
        except (OSError, ValueError) as exc:  # noqa: BLE001 - bookkeeping must not fail a phase
            self.log(
                f"[write-subagents] warning: could not record the verdict on "
                f"{path.name} for {outcome.task_id}: {exc}"
            )

    # -- per-batch bookkeeping (D3n: append BEFORE the next substitution) ---

    def _write_record(self, task: Dict[str, Any], facts: BatchFacts) -> BatchRecord:
        """Record *facts* in ``state.json`` through the writer both legs
        share (``write_record``), the status journaled through
        ``StateManager.update_task_status``; a state without a dict (a test
        double) takes the status alone. Caller holds the effects lock."""
        state_dict = getattr(self.state, "state", None)
        if not isinstance(state_dict, dict):
            self.state.update_task_status(facts.task_id, facts.status, reason=facts.reason)
            return BatchRecord()
        return record_batch_facts(
            state_dict, task, facts, tasks=self.tasks,
            set_status=self.state.update_task_status,
        )

    def _batch_facts(
        self, task: Dict[str, Any], outcome: TaskOutcome, status: str,
        reason: Optional[str], *, dispatched: bool, kept: bool,
    ) -> BatchFacts:
        """*outcome* as the facts both legs record (``write_record``): what
        ``util batch-done`` reads from the host's flags, read from the
        outcome. Its own check reported when it passed, failed, or only
        repeated findings that are not its own; an apply-code-fixes batch
        refused before dispatch and read as done reports no checks."""
        task_id = str(task.get("task_id", ""))
        return BatchFacts(
            task_id=task_id, kind=self.kind, status=status, reason=reason,
            leg=PROCESSED_LEG_EXECUTOR,
            processed_phase=os.path.basename(os.path.normpath(self.phase_dir)),
            check_findings=outcome.check_findings,
            batch_check=outcome.batch_check,
            own_check_ran=(
                outcome.batch_check is not None
                or bool(outcome.batch_check_accepted)
                or bool(outcome.batch_check_repeats)
            ),
            batch_check_accepted=outcome.batch_check_accepted or {},
            batch_check_repeats=outcome.batch_check_repeats or {},
            note=outcome.batch_check_note,
            listed=self._listed_outcomes.get(task_id),
            reports_checks=dispatched or kept or not outcome.ok,
            declined=dict(outcome.declined or {}),
            skipped=dict(outcome.skipped_items or {}),
            no_op=outcome.write_outcome == WRITE_NO_OP,
        )

    def _record_batch_effects(
        self,
        task: Dict[str, Any],
        outcome: TaskOutcome,
        batch_index: int,
        result: SchedulerResult,
        *,
        dispatched: bool = True,
        kept: bool = False,
    ) -> None:
        """Append prior-changes, record metrics, update task status — in that order.

        *dispatched* is ``False`` for a task refused before any runner
        started (:meth:`_refuse_before_dispatch`): no metric, and its batch
        line says it was refused. *kept* (with ``dispatched=False``) is an
        apply batch kept as it stood at the user's request
        (:meth:`_keep_at_request`, fix round 18): its summary is appended, its
        status is ``completed`` with that reason, and its line says so.

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
        else:
            summary = None
        if not outcome.ok or task_id in self._jail_failed or outcome.not_rerun:
            # A batch that is not done gets the line the native leg appends
            # for it (fix round 15: the executor wrote none, so later prompts
            # differed between the legs for the same history). Safe since fix
            # round 14 made `append` supersede a changed summary per `(id,
            # phase)`: the --resume that dispatches it again replaces the line
            # with its real entry, and a task answered `--human-task <id>=skip`
            # keeps it on both legs, which tells the tasks after it that the
            # work is not there.
            summary = (
                self._failed_entry(outcome, result.refusals) if dispatched else None
            )

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
                    self.output_dir, entry_id, self.phase, clip(summary, 400),
                    task_id=task_id,
                )

            # Before the metric, so the `[ETA]` line this task drives carries its
            # own 1-based position in the run (F3). No metric for a check that
            # was not run again: no child ran, and a zero would average in.
            ran = bool(not outcome.not_rerun and dispatched)
            self._eta.mark(task_id, ran=ran, ok=outcome.ok)
            # Fix round 21 (review of round 20, #1): a check that reported
            # findings holds its dependents only once its batch returns
            # (`_block_dependents_of_checks`), so its own line and the lines
            # of the tasks beside it in the batch still counted them, and a
            # check that held the last tasks ended the run on "batch 3/5 …
            # remaining". They leave the `[ETA]` total now, as the native
            # leg's `util batch-done` takes them out before `util metrics
            # record`.
            self._eta.held.update({
                held: outcome.task_id for held in self._check_holds(outcome, result)
            })
            if ran:
                self._eta.record(task, outcome, result)

            self._remember_outcome(outcome)
            if self.state is not None:
                if self.kind == "implement" and isinstance(
                    getattr(self.state, "state", None), dict
                ):
                    # Fix round 28: under this lock, so whichever of a code
                    # task and a check task records second sees the other.
                    self._split_check_repeats(task, outcome)
                status = "completed" if outcome.ok else "failed"
                reason = None if outcome.ok else (outcome.reason or outcome.status)
                if kept:
                    reason = (
                        f"kept at your request (--human-task {task_id}=skip): "
                        f"{len(outcome.skipped_items)} item(s) not applied"
                    )
                # The record `util batch-done` writes on the native leg.
                record = self._write_record(task, self._batch_facts(
                    task, outcome, status, reason, dispatched=dispatched, kept=kept))
                if record.repeats_moved:
                    self._move_repeats_of(task_id, record.repeats_moved, result)
                # plans/review-commits.md §2.7: a completed batch re-pins its
                # fix files to the bytes it left (no-op without a guard).
                self._commit_repin(task, outcome)
                # Persists the metric recorded just above too: it is a mutation
                # of this object's dict now, not a second writer's file (F4).
                self.state.save()

            # One line per batch, on the happy path (F7). Every other `self.log`
            # in the loop is exceptional, so a 25-minute five-batch run left an
            # 11-line log — and `status` points the user at that log.
            verdict = outcome.outcome_label or outcome.status
            if outcome.not_rerun:
                verdict = "not run again (nothing changed since it failed)"
            elif kept:
                verdict = (
                    f"kept as it stood at your request, not dispatched again "
                    f"({outcome.outcome_label or outcome.status}; not applied, "
                    f"skipped at your request: "
                    f"{clip(self._declined_words(outcome.skipped_items), 300)})"
                )
            elif not dispatched:
                verdict = "refused before dispatch" + (
                    f" ({outcome.write_outcome})" if outcome.write_outcome else ""
                )
            elif outcome.ok and outcome.check_findings:
                verdict = (
                    f"{verdict} ({len(outcome.check_findings)} problem(s): "
                    f"{clip('; '.join(outcome.check_findings), 300)})"
                )
            elif outcome.ok and outcome.batch_check:
                verdict = (
                    f"{verdict}, but its own report says "
                    f"{batch_check_words(outcome.batch_check)} "
                    f"({_check_records.quoted_findings(outcome.batch_check, 300)})"
                )
            if outcome.ok and outcome.batch_check_accepted:
                # Fix round 22: reported again, never its own failure.
                verdict = (
                    f"{verdict}; its own check reported again only what you "
                    f"accepted for {', '.join(outcome.batch_check_accepted)}"
                )
            if outcome.ok and outcome.batch_check_repeats:
                # Fix round 28: a check task's finding, never its own.
                verdict = (
                    f"{verdict}; its own check reported only what check task "
                    f"{', '.join(outcome.batch_check_repeats)} reported"
                )
            if outcome.ok and outcome.declined:
                # Fix round 18: what it left out on purpose, and why.
                verdict = (
                    f"{verdict}; declined {len(outcome.declined)}, not applied: "
                    f"{clip(self._declined_words(outcome.declined), 300)}"
                )
            elif outcome.write_outcome and not outcome.ok:
                # Plan step 3: a failed batch whose write outcome reads
                # "clean-failure" says which failure it was.
                verdict = f"{verdict} ({outcome.status})"
            stale_line = self._commit_stale.get(task_id)
            if stale_line is not None and not dispatched:
                # plans/review-commits.md §2.7: the pre-dispatch re-check's
                # `[COMMIT_FIXES_STALE]` line is this batch's line.
                self.log(stale_line)
            elif dispatched and not outcome.not_rerun:
                self.log(
                    f"[write-subagents] batch {batch_index}: {task_id} "
                    f"{verdict} in "
                    f"{outcome.duration_seconds:.1f}s — "
                    f"{', '.join(outcome.files_touched) or 'no files changed'}"
                    f"{rung_text}"
                )
            else:
                # Fix round 28 (live run 12, stop fixture on OpenCode and
                # Codex): a task no runner started was named as "batch 3"
                # beside "Organized into 3 execution batches", which counts
                # the batches the run dispatches (0 to 2), so the log read as
                # four. It names no batch: none was dispatched for it.
                self.log(
                    f"[write-subagents] {task_id} {verdict} — "
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

    def _failed_entry(
        self, outcome: TaskOutcome, refusals: _write_context.Refusals,
    ) -> Optional[str]:
        """The prior-changes line a failed batch leaves, in the native leg's
        words (``instructions/implement.md``, the three ``apply-*.md`` cores):
        ``(task failed — …)`` for an implement task, ``(batch failed — …,
        verified no file changes)`` for an apply batch that moved nothing.

        ``None`` where the native leg appends nothing: an apply batch that
        moved bytes (its loop stops before the append), and a check not run
        again (the line of the run that ran it stands). A task refused before
        dispatch gets none either (the caller's ``dispatched``): the native
        emit records it, and no host appends for a task it never ran.
        """
        if outcome.not_rerun or outcome.task_id in self._replaced_targets:
            # A replaced target: the file at the path changed under the batch,
            # and not by it — no line can say what it left.
            return None
        task = self._task_by_id(outcome.task_id) or {}
        is_check = self.kind == "implement" and not declared_file_set(task)
        last = outcome.attempts[-1] if outcome.attempts else None
        if outcome.applied_without_changes:
            what = "said applied, but its change did not land"
        elif outcome.write_outcome == WRITE_CLEAN_FAILURE and not is_check:
            if outcome.task_id in self._jail_failed:
                what = "the write jail refused its edit"
            elif last is not None and last.exit_code not in (0, None):
                what = f"the runner exited {last.exit_code}"
            else:
                # It said it failed: what it said, in its own words.
                what = " ".join(str(outcome.summary or "").split()) or (
                    "it reported that it could not do the task"
                )
            if not outcome.bytes_moved and self.kind == "implement":
                what += ", no file changed"
        else:
            what = " ".join(self._stop.failure_sentence(outcome, refusals).split())
        if self.kind == "implement":
            return f"(task failed — {what})"
        if outcome.bytes_moved:
            return None
        return f"(batch failed — {what}, verified no file changes)"

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
            # `_verification_only` and `write_verdict.decide`. Only `implement`
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
        # The same cutoff `_earlier_report_rows` applies: a snapshot taken
        # before the review this phase applies completed belongs to the
        # previous cycle — the bytes before THAT cycle's batch, so restoring
        # from it would roll that cycle's applied edit back too (follow-up to
        # the 2026-10-02 adversarial review). Set aside, never deleted.
        # `taken_before` applies the rule to the second, as the manifest is
        # stamped.
        reviewed_at = self._review_completed_at()
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
        if existing is None and expected_ids and self.kind != "implement":
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

        Under a commit review's apply guard (:meth:`_commit_guard`) the
        prompt's protected-path clause is left out
        (``code_fix_batcher.without_protected_clause``). That clause lists
        the guard's protected set, and a ``--resume`` grows the set whenever
        its re-check protects a file edited by hand since the stop
        (plans/review-commits.md §2.7, §2.8). Every prompt then changes, and a
        batch the stopped run had completed would read as made from changed
        inputs and be applied a second time. The list only forbids later
        writes, so the batches still to run get it in their prompts and from
        the deny layer. The rest of the prompt is hashed as before.
        """
        prompt = str(task.get("prompt") or "")
        if self._commit_guard() is not None:
            from .code_fix_batcher import without_protected_clause
            prompt = without_protected_clause(prompt)
        payload = json.dumps(
            {
                "kind": self.kind,
                "prompt": prompt,
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
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """The result file, the id it was produced under and the audit log's
        completion record for its items (``{}`` when there is none), when the
        task is done.

        The task's own positional result first; for an apply kind, a sibling
        result naming the same items next (:meth:`_sibling_results`), judged
        by the same clauses with the id and batch index it ran as — the
        audit, the verdict and the inputs binding are read for THAT batch,
        never for whatever this run put under the renumbered id.

        The audit log is read at most once per call, when a result file
        first needs it, and never kept for another task: it is resume
        evidence, and a sibling of a disjoint group may be writing it.
        """
        task_id = str(task.get("task_id", ""))
        expected = [str(item) for item in (task.get("expected_ids") or [])]
        read: List[List[Dict[str, Any]]] = []

        def audit_records() -> List[Dict[str, Any]]:
            if not read:
                read.append(self.audit.records())
            return read[0]

        def completion(recorded_id: str) -> Dict[str, Any]:
            return self.audit.completion_for(
                recorded_id, expected_ids=expected, records=audit_records(),
            ) or {}

        try:
            own = resolve_output_target(task, self.phase_dir)
        except ResultRejected:
            own = None
        if own is not None and self._result_counts_as_done(
            task, own, task_id, int(task.get("batch_index", 0)), sibling=False,
            audit_records=audit_records,
        ):
            return str(own), task_id, completion(task_id)
        for path, recorded_id, index in self._sibling_results(task):
            if self._result_counts_as_done(
                task, path, recorded_id, index, sibling=True,
                audit_records=audit_records,
            ):
                self.log(
                    f"[write-subagents] {task_id}: its items already landed as "
                    f"{recorded_id} ({path.name}) in an earlier run of this "
                    f"phase — the batch was renumbered by a selection change, "
                    f"not re-applied."
                )
                return str(path), recorded_id, completion(recorded_id)
        return None

    def _result_counts_as_done(
        self, task: Dict[str, Any], target: Path, task_id: str,
        batch_index: int, *, sibling: bool,
        audit_records: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    ) -> bool:
        """Resume's definition of done (D3i), all five clauses of it.

        ``task_id``/``batch_index`` are the ones the result was produced
        under (the task's own, or a renumbered sibling's); ``sibling`` says
        the file was found by its items, in which case the clauses keyed by
        positional id alone (``state.json``'s task status, the fingerprint
        file) are read from the verdict stamp instead, since this run may
        already have put another batch under that id. ``audit_records``
        returns the audit log's records as :meth:`_done_evidence` read them;
        without it the log is read now.

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
        verdict = data.get("verdict")
        if (
            str(data.get("status") or "") not in (WRITE_APPLIED, WRITE_NO_OP)
            and not _check_records._findings_from_a_failed_claim(data)
            and not _check_records._declined_from_a_failed_claim(data)
        ):
            # A schema-valid `{"status": "failed"}` is a batch the previous run
            # recorded as NOT done; resuming past it on the strength of the
            # file existing would skip exactly the batch --resume is for —
            # unless the verdict stamped on it read it as a check's findings
            # (fix round 13), whose re-run `_check_findings_rerun` decides.
            return False
        if isinstance(verdict, dict) and (
            not verdict.get("ok", True)
            or verdict.get("write_outcome") == WRITE_CLEAN_FAILURE
        ):
            # The scheduler's own classification, stamped by `_stamp_verdict`
            # once `_classify` decided (F1): the file reads `applied` because
            # that is what the runner printed before it died, and the loop
            # stopped on it promising a re-dispatch. A `clean-failure` stamped
            # `ok` is a run before fix round 14 reading an `applied` claim that
            # moved nothing as done: nothing moved, so it is dispatched again.
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
        if audit_records is None:
            audit_records = self.audit.records
        expected = [str(item) for item in (task.get("expected_ids") or [])]
        if sibling:
            # Found by its verdict stamp, which only `_classify` writes, so
            # the crash-before-classification shape cannot apply; the audit's
            # own record for these items has to agree it succeeded.
            recorded = self.audit.completion_for(
                task_id, expected_ids=expected, records=audit_records())
            if not recorded or recorded.get("status") != STATUS_OK:
                return False
        else:
            latest = next((
                row for row in reversed(audit_records())
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

    def _persist_human_plan(self, plan: _human_task_plan.HumanTaskPlan) -> None:
        if self.state is None:
            return
        _human_task_plan.persist_human_task_plan(self.state, plan)

    def _pending_human_message(self, plan: _human_task_plan.HumanTaskPlan) -> List[str]:
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
                f"{'|'.join(_human_task_plan.STRATEGIES)})"
            )
        lines.append(
            "Nothing was dispatched, no file was modified, and the phase is not "
            "marked completed."
        )
        return lines

    # -- stop / finalize ------------------------------------------------

    def _dependents_of(self, task_id: str) -> List[str]:
        """Every task that depends on *task_id*, transitively, in id order.

        Over the whole decomposition (:func:`decomposition_graph`), not only
        this run's tasks: a resume does not materialize a finished task, and
        a walk over the rest stopped at it (fix round 12). A task this run
        does not hold is in the answer; callers that act on a task skip it.
        """
        if self.kind != "implement":
            return _check_records.transitive_dependents(self.tasks, task_id)
        return _check_records.transitive_dependents(
            _check_records.decomposition_graph(self.tasks, self.state), task_id,
        )

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

    def _stop_on_manual_gate(
        self, result: SchedulerResult, gates: Dict[str, Any],
    ) -> SchedulerResult:
        """End a run that ran nothing because a manual task gates every code
        task left: exit 3, the phase recorded started and not completed (so
        every reader offers its resume), the stop record, the closing
        ``[ETA]`` line, summary.md and the block naming the way on."""
        result.exit_code = EXIT_NEEDS_HUMAN
        self._mark_implement_started()
        self._stop.record_manual_gate(result, gates)
        gating = list(result.stop.get("manual_task_ids") or [])
        held = list(result.stop.get("held_task_ids") or [])
        return self._end_run(
            result, lambda: self._stop.manual_gate_message(result),
            eta_why=(
                f"{', '.join(gating)} left for you (manual); {len(held)} code "
                f"task(s) wait on {'it' if len(gating) == 1 else 'them'}"
            ),
            unverified=False, check_lines=False, summary=FULL_SUMMARY_LINE,
        )

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
        from . import status_file as _status_file
        rejected = isinstance(exc, RunnerRejected)
        try:
            self._sweep.clear_in_flight()
            _status_file.update_status(self.phase_dir, {
                "state": "runner_rejected" if rejected else "executor_refused",
                "reason_code": (
                    (getattr(exc, "reason_code", "") or "") if rejected else ""
                ) or type(exc).__name__,
                "runner": (getattr(exc, "spec", "") or "") if rejected else "",
                "runner_key": (getattr(exc, "key", "") or "") if rejected else "",
                "refused_batch": batch_index,
                "progress": {"write_subagents": {
                    **self._reports.mirror_snapshot(),
                    "in_flight": [],
                }},
            })
        except Exception:  # noqa: BLE001 - observability must not fail a phase
            pass

    # -- verification ---------------------------------------------------

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
          escalation the runner spawn uses. On Windows, which has no groups,
          the command starts in a Job Object of its own
          (``process_control.popen_tree``) and the timeout ends that job.
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
                proc, job = _process_control.popen_tree(  # noqa: S602 - a shell string is the contract
                    command, shell=True, cwd=self.repo_root,
                    stdin=subprocess.DEVNULL, stdout=sink,
                    stderr=subprocess.STDOUT, env=env,
                    **_process_control.new_process_group_kwargs(),
                )
            except OSError as exc:  # pragma: no cover - the shell itself is missing
                return VERIFY_EXIT_NOT_RUN, (
                    f"verification command could not be started: {exc}"
                )
            try:
                # Read the group while the leader is alive: `os.getpgid` stops
                # answering once it is reaped, and the group is what still
                # holds the build's children.
                pgid = _process_control.process_group_id(proc.pid)
                note = ""
                try:
                    exit_code = int(proc.wait(timeout=timeout))
                except subprocess.TimeoutExpired:
                    verdict = self._kill_verify_tree(proc, pgid, job)
                    exit_code = VERIFY_EXIT_TIMEOUT
                    note = _verify_timeout_note(timeout, verdict, proc.pid, pgid)
            finally:
                if job is not None:
                    job.close()
            return exit_code, self._verify_output_tail(sink) + note

    def _kill_verify_tree(
        self, proc: Any, pgid: Optional[int], job: Any = None,
    ) -> Optional[str]:
        """SIGTERM → SIGKILL the timed-out command *and its group*, then reap it.
        Returns ``kill_tree``'s verdict; ``None`` when the kill raised (the
        command was then sent a kill through its job or its ``Popen``,
        unconfirmed).

        ``start_time=None`` is how an unverified kill is *requested*: this pid is
        a child still held open here, not a record read off disk, so there is
        nothing to corroborate and no recycled pid to protect; as ``child``,
        unreaped until the wait below, it anchors its pid and group through the
        kill where the owned kill runs (Linux, glibc or musl; see ``kill_tree``).
        *job* is the command's Windows Job Object, the group's stand-in there.

        Only ``killed`` and ``gone`` confirm the command stopped, the executor's
        rule for its timed-out runner (``subagent_executor._unconfirmed_kill``):
        a ``survived`` that is only the command's own zombie becomes ``killed``
        once its exit is confirmed (``process_control.own_child_stopped``), and
        any other outcome is logged as a warning naming the pid, which a
        detached run's record keeps for a later abort or sweep
        (``detach.note_left_running``; docs/development/write-scheduler.md,
        "When the verify command times out").
        """
        verdict: Optional[str]
        try:
            verdict = _process_control.kill_tree(
                proc.pid, start_time=None, pgid=pgid,
                escalate_after_s=VERIFY_KILL_GRACE_S, poll_interval_s=0.05,
                job=job, child=proc,
            )
            if verdict == "survived" and _process_control.own_child_stopped(proc, pgid):
                verdict = "killed"
        except Exception:  # noqa: BLE001 - teardown must never mask the timeout
            verdict = None
            if job is None or not job.terminate():
                try:
                    proc.kill()
                except OSError:  # pragma: no cover - already gone
                    pass
        try:
            proc.wait(timeout=VERIFY_REAP_S)
        except subprocess.TimeoutExpired:  # pragma: no cover - survived SIGKILL
            pass
        reason = _executor._unconfirmed_kill(verdict)
        if reason is not None:
            detach = sys.modules.get(f"{__package__}.detach") if __package__ else None
            kept = detach is not None and detach.note_left_running(proc.pid, pgid, verdict)
            self.log(
                f"[write-subagents] warning: the timed-out verification command "
                f"pid {proc.pid} (process group {pgid if pgid is not None else '?'}) "
                f"could not be confirmed stopped (kill verdict {verdict}): {reason}, "
                f"so it may still be running. Check it by hand"
                + ("; the run's record keeps it for status --abort." if kept else ".")
            )
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
        from .providers.base import build_child_env
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
        if not self._verify.run(result):
            # The stop path's shape, for the same reason: the summary is this
            # run's own account and `--resume` has no other record of what was
            # applied. Nothing is rolled back and nothing is completed — the
            # plan reference, the tracking census and the phase marker all wait
            # for a run whose check passes. Fix round 21: and its `[ETA]` lines
            # close on the failed command, not on "ALL BATCHES COMPLETE".
            verification = result.verification or {}
            self._eta.close(
                result,
                why=(
                    f"`{verification.get('command')}` exited "
                    f"{verification.get('exit_code')}"
                    if verification.get("command") else ""
                ),
            )
            if self.summary_file:
                result.summary_file = self._reports.write_summary(result)
                result.messages.append(
                    f"Implementation summary: {result.summary_file}"
                )
            return
        if self.state is not None:
            self._eta.finish(result)
            # A resume of a phase already completed that dispatched nothing —
            # `implement --resume --human-task T007=done` closing a deferred
            # manual task — changed no code, so its completion keeps its date:
            # re-stamping it made review-code and apply-code-fixes read stale,
            # and the closing answer cost a review cycle (fix round 14). Fix
            # round 15: "dispatched nothing" is read off the outcomes, not
            # `batches_dispatched`, which a batch of resume skips joins too —
            # a check whose findings stand, materialized again on every
            # resume and kept without a dispatch, re-stamped the date.
            # 2026-10-09: but never a date older than the review this apply
            # phase applies. A `--force` round whose review re-reports every
            # batch byte for byte keeps them all and dispatches nothing; the
            # findings are applied, so the apply is stamped after the review
            # (plan improve-plan-and-improve-code §2.5). Kept, the old date
            # read as "review not applied yet" to `ImproveWorkflow`, and no
            # later re-run of the composite reached `[ALREADY_COMPLETE]`.
            ran = any(not outcome.skipped for outcome in result.outcomes)
            review = REVIEW_PHASE_BY_WRITE_KIND.get(self.kind)
            keeps_date = (
                not ran
                and self.state.is_phase_completed(self.phase)
                and not (
                    review is not None
                    and self.state.completion_predates(self.phase, review)
                )
            )
            if not keeps_date:
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
            if self.kind in _batch_prompt.CODE_WRITE_KINDS:
                # Only a code phase records `tracked_files`: the planning
                # phases edit the plan and the task list, which are never
                # code under review (2026-10-05, Codex racing run 3:
                # apply-task-suggestions' census put both in review-code's
                # scope).
                self._verify.finalize_tracking(result)
                # `finalize_tracking` is a file-level writer: it read the state
                # we just saved, computed `tracked_files` and wrote it back.
                # Adopt its file before touching the dict again, or the `save()`
                # below dumps the pre-finalize dict over it and the phase's
                # only deliverable is not tracked at all (F3, measured
                # 2026-09-16).
                self._reload_state()
        if self.summary_file:
            result.summary_file = self._reports.write_summary(result)
            self._reports.insert_plan_reference(result.summary_file)
            if self.state is not None:
                # `WriteReports.insert_plan_reference` is the one plan edit a run makes
                # *after* the stamp above (implement's own back-reference
                # insertion) — re-stamp once more so it is not left stale too.
                self.state.refresh_plan_hash()
                self.state.save()
        # Fix round 15: a check whose findings stand and hold nothing back
        # lets the run finish; the run's own later edits (the tasks after it,
        # the plan's back-reference just above) are not a change the user
        # made, so its tree is stamped as the run leaves it, as on a stop —
        # or the next resume ran it again on files nobody touched.
        self._restamp_check_trees(result)
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

    def _mark_implement_started(self) -> None:
        """Record implement started, once per run, as the native emit does.

        Fix round 17 (review of fix round 16): the executor leg never wrote
        ``phases_started.implement``, so once an implement had finished, a
        later run that stopped left that completion standing — ``status``
        read implement done, and the next plain ``--implement`` started over
        instead of continuing. Called as a task is handed to this run (past
        the resume skips), so a resume that dispatches nothing — a
        ``--human-task <id>=done`` closing a manual task, a check whose
        findings stand — leaves a completed implement completed, and
        :meth:`_finalize` completes one that ran. A start-over already
        stamped it (``implement_orchestrator._start_over``). Best-effort.
        """
        if (
            self.kind != "implement" or self.dry_run or self.state is None
            or self._implement_started_marked
        ):
            return
        with self._effects_lock:
            if self._implement_started_marked:
                return
            self._implement_started_marked = True
            try:
                if not self.state.is_phase_started(self.phase):
                    self.state.mark_phase_started(self.phase)
                    self.state.save()
            except Exception:  # noqa: BLE001 - bookkeeping must not fail a phase
                pass

    def _earlier_report_rows(self) -> List[Dict[str, Any]]:
        """The task rows the previous run left in ``write_batch_report.json``.

        Read before this run writes its own. Anything unreadable is simply no
        earlier rows: the report is evidence, never an input the loop obeys.

        Only rows of *this review cycle*: an apply phase's batch ids
        (``apply-batch-N``) are per cycle and the phase dir outlives the
        review, so a ``--resume`` typed after a ``--force`` review re-run
        carried the previous review's rows as "(earlier run)" rows of a
        different review (adversarial review, 2026-10-02). The rule is
        :func:`state_manager.predates_review`: a report written (its
        ``written_at``, else its mtime) before its review's latest
        completion belongs to the earlier cycle, and a missing date on
        either side keeps the rows.
        """
        path = os.path.join(self.phase_dir, _write_reports.WRITE_BATCH_REPORT_FILENAME)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        written_at = stamp_or_mtime(data.get("written_at"), path)
        reviewed_at = self._review_completed_at()
        if predates_review(written_at, reviewed_at):
            self.log(
                f"[write-subagents] the earlier {_write_reports.WRITE_BATCH_REPORT_FILENAME} "
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
        """When the review this apply phase applies last completed, or None
        (:func:`review_completed_at`)."""
        return review_completed_at(getattr(self.state, "state", None), self.kind)

    def _started_a_runner(
        self, outcome: TaskOutcome, refusals: _write_context.Refusals,
    ) -> bool:
        """Whether this run started a runner for *outcome*'s task: not a
        resume skip, a task refused before dispatch, a check not run again
        or a batch kept as it stood at the user's request (its left-out
        items are ``skipped_items``, :meth:`_keep_at_request`)."""
        return not (
            outcome.skipped or outcome.not_rerun or outcome.skipped_items
            or outcome.task_id in refusals.before_dispatch
        )

    # -- reports and status ---------------------------------------------

    # -- dry run --------------------------------------------------------

    def _write_dry_run(
        self,
        ordered: Sequence[Tuple[int, List[Dict[str, Any]]]],
        plan: _human_task_plan.HumanTaskPlan,
    ) -> str:
        """``--dry-run-subagents``: full preflight, zero dispatch (D3l).

        Resolves the runner, renders the exact argv per task, checks the prompt
        transport, applies the profile, the disjointness admission and the write
        policy — then writes the report and starts no child. It is what makes
        "would this batch have been refused?" answerable *before* a model is handed
        edit access.
        """
        from .subagent_tasks import preferred_prompt_transport, utf16_units

        entries: List[Dict[str, Any]] = []
        for batch_index, batch_tasks in ordered:
            # As `_run_batch`: one resolution per batch, for its ceiling and
            # each entry's runner.
            runners, ceiling = self._batch_runners_and_ceiling(batch_tasks)
            groups = admit_disjoint_groups(
                batch_tasks, ceiling=ceiling, root=self.repo_root,
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
                    prompt = self._prompts.dispatched_prompt(task)
                except ResultRejected as exc:
                    entry["would_dispatch"] = False
                    entry["reason"] = exc.message
                    entries.append(entry)
                    continue
                runner = runners[task_id]
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
        effective, configured, provider_limits = self._ceiling_record()
        # Same containment as every other executor write (see ``WriteReports.write_report``).
        _atomic_write_json(path, {
            "dry_run_version": 1,
            "kind": self.kind,
            "phase": self.phase,
            "repo_root": self.repo_root,
            "ceilings": {
                "effective_write_parallelism": effective,
                "driver_max_concurrent_write_subagents": configured,
                "provider_max_concurrent": provider_limits,
                "host_max_concurrent_subagents": self.host_max,
                "shared_artifact_phase": self.kind in SHARED_ARTIFACT_WRITE_KINDS,
            },
            "write_paths": self.policy.to_dict(),
            "human_task_strategy": plan.strategy,
            "tasks": entries,
        })
        return path


def _redacted_argv(argv: Sequence[str], prompt: str) -> List[str]:
    from .write_safety import redact_argv
    return redact_argv(argv, prompt)


# ---------------------------------------------------------------------------
# The one entry point an orchestrator calls
# ---------------------------------------------------------------------------

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
