#!/usr/bin/env python3
"""The Python-side subagent executor: `--exec-subagents` (D3, read/analyze half).

D2 materialized every subagent prompt into a task JSON; this module is the
*second* invoker of those tasks, next to a host's native subagent tool. Both run
the identical ``task.prompt`` and both land on the identical ``output_path`` file
(D3a), so a phase driven headlessly and a phase driven by Claude Code's Task tool
reaggregate byte-identical inputs.

One dispatch, in order:

1. **Refuse before spawning** — the D2 pre-dispatch refusals (``inline-json``,
   ``executor_role: human``, ``workspace-read`` against a runner that cannot read
   a workspace) plus the D3f containment check on ``output_name``. Nothing is
   started for a task that cannot legally produce a result.
2. **Pick a profile, never a raw command** (D3e) — every read/analyze kind runs
   under ``agent-readonly``, *derived from the task kind* so a materialized prompt
   (untrusted once it embeds plan text) can never request escalation, with ``cwd``
   pinned to the target repository.
3. **Capture, then parse through the adapter** (D3) — provider CLIs emit
   structured event streams, not plain text, so ``parse_output`` is mandatory
   rather than an optimization: running ``json_extractor`` over raw stdout would
   match an event envelope or a reasoning item and silently write a garbage
   result that reaggregation would accept.
4. **Validate, then write atomically** (D3f) — the extracted JSON is checked
   against its kind's result shape *before* anything is written, under the phase
   dir, 0600, via ``os.replace``.
5. **Classify every failure on exactly one rung** (D3k) and record it in
   ``subagent_batch_report.json`` plus ``.status.json``. A batch with any failed
   or incomplete task refuses to reaggregate and exits 3 — even under ``--yes``,
   and never by falling back to salvage.

The write path (``agent-write``, D3f/D3g/D3h/D3l/D3n) is deliberately not here:
:func:`profile_for_kind` and :func:`dispatch_task` are the seams it extends.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import shim mirrors the rest of utils/
    from . import process_control
    from .json_extractor import extract_json_from_text
    from .llm_client import _CMD_UNSAFE_CHARS, _resolve_executable
    from .provider_registry import (
        get_max_concurrent_subagents,
        get_provider,
        get_provider_max_concurrent,
        get_subagent_retries,
        get_subagent_runner,
        get_subagent_timeout,
        is_model_valid,
        parse_model_spec,
    )
    from .providers.base import (
        PROFILE_AGENT_READONLY,
        PROFILE_AGENT_WRITE,
        ProfileUnsupported,
        matches_any,
    )
    from .schema_validator import (
        validate_human_decision_verdicts,
        validate_with_schema_file,
    )
    from .subagent_tasks import (
        KINDS,  # noqa: F401 - re-exported for callers that read it off this module
        READ_KINDS,
        WRITE_KINDS,
        preferred_prompt_transport,
        refuse_reason,
        tasks_of,
    )
except ImportError:  # pragma: no cover - direct-script import shim
    from utils import process_control
    from utils.json_extractor import extract_json_from_text
    from utils.llm_client import _CMD_UNSAFE_CHARS, _resolve_executable
    from utils.provider_registry import (
        get_max_concurrent_subagents,
        get_provider,
        get_provider_max_concurrent,
        get_subagent_retries,
        get_subagent_runner,
        get_subagent_timeout,
        is_model_valid,
        parse_model_spec,
    )
    from utils.providers.base import (
        PROFILE_AGENT_READONLY,
        PROFILE_AGENT_WRITE,
        ProfileUnsupported,
        matches_any,
    )
    from utils.schema_validator import (
        validate_human_decision_verdicts,
        validate_with_schema_file,
    )
    from utils.subagent_tasks import (
        KINDS,  # noqa: F401 - re-exported for callers that read it off this module
        READ_KINDS,
        WRITE_KINDS,
        preferred_prompt_transport,
        refuse_reason,
        tasks_of,
    )


__all__ = [
    "BATCH_REPORT_FILENAME",
    "FAILED_CAPTURE_DIRNAME",
    "MAX_RESULT_BYTES",
    "MAX_CAPTURE_BYTES",
    "STATUS_OK",
    "STATUS_TIMEOUT",
    "STATUS_RATE_LIMITED",
    "STATUS_RUNNER_ERROR",
    "STATUS_UNPARSEABLE",
    "STATUS_SCHEMA_INVALID",
    "STATUS_REJECTED",
    "STATUS_SPAWN_FAILED",
    "STATUS_AUTH_FAILED",
    "STATUS_INCOMPLETE",
    "STATUS_PARTIAL",
    "STATUS_POLICY_VIOLATION",
    "WRITE_APPLIED",
    "WRITE_CLEAN_FAILURE",
    "WRITE_PARTIAL",
    "WRITE_SCHEMA_REJECTED",
    "WRITE_POLICY_VIOLATION",
    "WRITE_STOP_OUTCOMES",
    "RUNG_BY_STATUS",
    "RETRYABLE_STATUSES",
    "FATAL_STATUSES",
    "ExecutorError",
    "BatchFailed",
    "RunnerRejected",
    "ResultRejected",
    "ProfileUnsupported",
    "Attempt",
    "TaskOutcome",
    "BatchReport",
    "profile_for_kind",
    "resolve_task_runner",
    "resolve_output_target",
    "effective_ceiling",
    "extract_result",
    "validate_result",
    "persist_result",
    "task_is_complete",
    "dispatch_task",
    "run_tasks",
    "run_task_file",
    "simulate_native_invoker",
    "continue_past_marker",
    "dry_run_report",
    "preflight_runner",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The executor-owned bookkeeping file beside the immutable task JSON (D3i).
#: The task file stays a hash-bound snapshot, so attempt counts live here.
BATCH_REPORT_FILENAME = "subagent_batch_report.json"

#: Per-attempt diagnostic captures (D3k). A directory, so the reaggregation glob
#: (``{phase_dir}/*.json``) never sees them.
FAILED_CAPTURE_DIRNAME = "failed"

#: Result size cap (D3f): reject rather than truncate, so a runaway or
#: adversarial runner cannot fill the disk through the executor.
MAX_RESULT_BYTES = 8 * 1024 * 1024

#: Per-stream diagnostic capture cap (D3k): last 64 KiB each of stdout/stderr.
MAX_CAPTURE_BYTES = 64 * 1024

#: Captured-stream cap. Beyond this the capture is refused (rung 5) rather than
#: half-parsed into a plausible-looking result.
MAX_STREAM_BYTES = 32 * 1024 * 1024

#: Grace between the SIGTERM and the SIGKILL of a timed-out dispatch's process
#: group (D3k rung 2). Short on purpose: the child has already spent its entire
#: ``subagent_timeout_s``, so this is teardown, not one more chance to finish.
TIMEOUT_KILL_GRACE_S = 0.5

#: Ceiling on the post-kill drain. A grandchild the kill did not reach still
#: holds the inherited stdout pipe, so the drain must be bounded — an unbounded
#: one is the hang that turns a timed-out task into a wedged batch.
TIMEOUT_DRAIN_S = 5.0

#: The smallest slice of the per-task budget worth *starting* a retry with
#: (D3i). ``subagent_timeout_s`` is the task's total wall-clock budget, so a
#: retry gets what is left of it — and a runner handed under a second would
#: spend it on process start-up and be killed for the same reason the previous
#: attempt was, having only added a spawn and a kill to the batch.
MIN_ATTEMPT_BUDGET_S = 1.0

STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_RUNNER_ERROR = "runner_error"
STATUS_UNPARSEABLE = "unparseable_result"
STATUS_SCHEMA_INVALID = "schema_invalid"
STATUS_REJECTED = "rejected"
STATUS_SPAWN_FAILED = "spawn_failed"
STATUS_AUTH_FAILED = "auth_failed"
#: Not a dispatch outcome: a task with no result file at all (an orphan from a
#: run killed mid-batch). Defined by artifacts, never by memory (D3i).
STATUS_INCOMPLETE = "incomplete"

#: The D3k ladder, first match wins. Every dispatch failure lands on exactly one
#: rung, and the status is what ``subagent_batch_report.json`` records verbatim.
RUNG_BY_STATUS = {
    STATUS_SPAWN_FAILED: 1,
    STATUS_AUTH_FAILED: 1,
    STATUS_TIMEOUT: 2,
    STATUS_RATE_LIMITED: 3,
    STATUS_RUNNER_ERROR: 4,
    STATUS_UNPARSEABLE: 5,
    STATUS_SCHEMA_INVALID: 6,
    STATUS_REJECTED: 7,
}

#: Statuses that consume ``driver.subagent_retries``.
RETRYABLE_STATUSES = frozenset({STATUS_TIMEOUT, STATUS_RUNNER_ERROR, STATUS_UNPARSEABLE})

#: Rate limiting is retried with backoff+jitter and does **not** consume the
#: retry budget (D3i) — it is not a transport failure.
BACKOFF_STATUSES = frozenset({STATUS_RATE_LIMITED})

#: The only statuses ``_classify_failed_run`` may reclassify (D3k). Rungs 4 and 5
#: mean "the run produced nothing usable", which is exactly the condition under
#: which a 429 or a login notice is the *real* explanation. Rung 6 is deliberately
#: absent: a ``schema_invalid`` result **parsed** — it is a deterministic
#: disagreement with the kind's result shape that is never retried — and its
#: streams carry the model's own prose, which routinely quotes "rate limit",
#: "429" or "unauthorized" straight out of the plan or diff under review. Matching
#: those substrings there would launder a schema disagreement into backoff retries
#: or, worse, into a batch-failing ``auth_failed``.
RECLASSIFIABLE_STATUSES = frozenset({STATUS_RUNNER_ERROR, STATUS_UNPARSEABLE})

#: Rungs that fail the whole batch immediately: every sibling would fail
#: identically, so continuing would only burn the retry budget N times.
FATAL_STATUSES = frozenset({STATUS_SPAWN_FAILED, STATUS_AUTH_FAILED})

#: Not a dispatch outcome either: the two D3g **verification** failures. They have
#: no D3k rung by design — the child ran, produced a schema-valid result and the
#: transport worked; what failed is the executor's own byte comparison, so
#: attaching a transport rung would be a category error (and would make a retry
#: look reasonable, which it never is for a half-applied file).
STATUS_PARTIAL = "partial_application"
STATUS_POLICY_VIOLATION = "policy_violation"

#: Every read/analyze kind runs read-only; every write kind runs ``agent-write``.
#: Derived from the kind by the executor (D3e), never read from the task JSON —
#: which is the whole point: a materialized prompt embeds untrusted plan/diff text
#: and must not be able to request escalation.
PROFILE_BY_KIND = {
    **{kind: PROFILE_AGENT_READONLY for kind in READ_KINDS},
    **{kind: PROFILE_AGENT_WRITE for kind in WRITE_KINDS},
}

#: The four D3g write outcomes, plus the D3l containment failure. Recorded on the
#: outcome as its own axis rather than being folded into ``status``, because
#: "applied" and "clean failure" are both successful *dispatches* that the loop
#: continues past, while "partial application" and "policy violation" stop it.
WRITE_APPLIED = "applied"
WRITE_CLEAN_FAILURE = "clean-failure"
WRITE_PARTIAL = "partial-application"
WRITE_SCHEMA_REJECTED = "schema-rejection"
WRITE_POLICY_VIOLATION = "policy-violation"

#: Write outcomes that stop the batch loop at its boundary (D3g/D3n): the phase is
#: not completed, snapshots stay in place, and the run exits 3 naming the restore
#: command — under ``--yes`` too, since an unattended run may not decide that a
#: half-applied plan file is acceptable.
WRITE_STOP_OUTCOMES = frozenset({WRITE_PARTIAL, WRITE_POLICY_VIOLATION})

#: Per-kind ``output_name`` patterns (D3f), so the reaggregation glob can never be
#: fed a foreign file. ``salvage`` is the one kind whose output IS a model result
#: file (it repairs one), so its pattern is "any plain .json basename" and the
#: reserved-name check below is what bounds it.
OUTPUT_NAME_PATTERN_BY_KIND = {
    "validation": re.compile(r"^validation_batch_\d+\.json$"),
    # Revalidation deliberately reuses the validation batch filenames: it is the
    # same producer (``prepare_batched_revalidation_tasks`` delegates to
    # ``prepare_batched_validation_tasks``) and the same merge reads them back, so
    # the pattern accepts both spellings rather than the kind's own name.
    "revalidation": re.compile(r"^(?:re)?validation_batch_\d+\.json$"),
    "consolidation": re.compile(r"^consolidation_batch_\d+\.json$"),
    # A judge verdict file is keyed by the judged item's stable ``group_id``
    # (``human_decisions.verdict_output_name``), not by a batch index, so that a
    # resumed judging phase — which materializes only the *remaining* items —
    # cannot have one group's task adopt another group's already-written verdict
    # as its own completed output. The token is therefore any basename-safe id;
    # older ``_<n>`` names still match.
    "human_decision_judge": re.compile(
        r"^human_decision_batch_[A-Za-z0-9][A-Za-z0-9._-]*\.json$"
    ),
    "task_generation": re.compile(r"^task_generation_result\.json$"),
    "salvage": re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$"),
    # Write kinds (D3g): the batch result JSON. One distinct pattern per kind, so a
    # hand-edited or regenerated task from one phase can never write into another
    # phase's result set even when the phase dirs are siblings.
    "apply_suggestions": re.compile(r"^apply_batch_\d+\.json$"),
    "apply_task_suggestions": re.compile(r"^apply_task_batch_\d+\.json$"),
    "apply_code_fixes": re.compile(r"^apply_fix_batch_\d+\.json$"),
    "implement": re.compile(r"^implement_batch_\d+\.json$"),
}

#: Orchestrator-owned artifacts an executor write may never land on (D3f).
RESERVED_OUTPUT_NAMES = frozenset({
    "grouped.json",
    "validation.json",
    "validation_tasks.json",
    "revalidation.json",
    "revalidation_tasks.json",
    "consolidation_tasks.json",
    "consolidated.json",
    "human_decision_tasks.json",
    "report_data.json",
    "state.json",
    ".status.json",
    "orchestrator-run.log",
    BATCH_REPORT_FILENAME,
    # Write-path artifacts the executor owns (D3g/D3l). A subagent result landing
    # on any of these would overwrite the very evidence the classification, the
    # audit trail and `util restore-batch` depend on.
    "orchestrator_output.json",
    "subagent_audit.jsonl",
    "subagent_dry_run.json",
    "prior_changes.jsonl",
    "write_batch_report.json",
    # The two materialized task files the list above was missing: the write half's
    # (``write_phase_bridge.WRITE_TASK_FILENAME``, spelled out rather than imported
    # only to keep this module's import graph unchanged) and ``generate-tasks``'s
    # single-task envelope. A result landing on either would overwrite the very
    # file the phase is dispatching from, which ``salvage``'s deliberately broad
    # "any plain .json basename" pattern would otherwise permit.
    "write_tasks.json",
    "task_generation_tasks.json",
})

#: Basename prefixes an executor write may never land on (a task file, a salvage
#: request, a rotated log).
RESERVED_OUTPUT_PREFIXES = ("salvage_", "error_", "orchestrator-run.")

#: Basename suffixes an executor write may never land on. Every materialized task
#: file is ``<phase>_tasks.json``, so the suffix — not the enumeration above — is
#: what keeps a later phase's task file safe without a second edit here.
RESERVED_OUTPUT_SUFFIXES = ("_tasks.json",)

#: Statuses mirrored into ``.status.json`` under ``progress`` (display only).
_STATUS_MIRROR_KEY = "subagents"

#: The token D3n substitutes late, per dispatch. Duplicated as a private constant
#: rather than imported at module scope only to keep this module's import graph
#: unchanged; ``tests/test_subagent_tasks.py`` pins the two spellings together.
_PRIOR_CHANGES_TOKEN = "{prior_changes_context}"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ExecutorError(Exception):
    """Something the executor refuses to do. Exit 3 ("needs a human", D6)."""

    exit_code = 3

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RunnerRejected(ExecutorError):
    """A configured runner spec cannot serve this phase (one rejection surface).

    Rendered by :func:`_render_runner_rejection` so the wording cannot drift per
    call site: the offending key, the spec as written, a reason code
    (``unknown_provider`` | ``missing_capability`` | ``missing_profile``), that
    code's explanation, the runners on this host that *would* satisfy the request,
    and the fix.
    """

    def __init__(self, message: str, *, key: str = "", spec: str = "",
                 reason_code: str = "") -> None:
        super().__init__(message)
        self.key = key
        self.spec = spec
        self.reason_code = reason_code


class ResultRejected(ExecutorError):
    """One task's result cannot be used. Carries the D3k status it lands on."""

    def __init__(self, message: str, status: str) -> None:
        super().__init__(message)
        self.status = status


class BatchFailed(ExecutorError):
    """A batch had failed or incomplete tasks, so the phase must not reaggregate."""

    def __init__(self, message: str, report: "BatchReport") -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Attempt:
    """One dispatch attempt of one task."""

    index: int
    status: str
    exit_code: Optional[int] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    capture_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "attempt": self.index,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 3),
        }
        if self.exit_code is not None:
            out["exit_code"] = self.exit_code
        if self.error:
            out["error"] = self.error
        if self.usage:
            out["usage"] = dict(self.usage)
        if self.capture_path:
            out["capture"] = self.capture_path
        return out


@dataclass
class TaskOutcome:
    """The row ``subagent_batch_report.json`` records for one ``task_id``."""

    task_id: str
    kind: str
    runner: str
    status: str
    profile: str = PROFILE_AGENT_READONLY
    attempts: List[Attempt] = field(default_factory=list)
    output_path: Optional[str] = None
    reason: Optional[str] = None
    duration_seconds: float = 0.0
    usage: Optional[Dict[str, Any]] = None
    usage_reported: bool = False
    skipped: bool = False
    #: D3g, write kinds only: which of the four outcomes the snapshot/diff
    #: classification landed on. Its own axis rather than a status, because
    #: `applied` and `clean-failure` are both dispatches the loop continues past.
    write_outcome: Optional[str] = None
    #: The files the post-run byte comparison actually observed as changed — the
    #: set the write policy is enforced against, never the runner's own claim.
    files_touched: List[str] = field(default_factory=list)
    #: The `summary` line lifted from the batch result JSON, for the prior-changes
    #: append. Never a compressed chat message.
    summary: Optional[str] = None
    #: Where this task's pre-batch snapshot lives, so an exit-3 message can name
    #: the restore command instead of describing it.
    snapshot_dir: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def stops_the_loop(self) -> bool:
        """Whether this outcome forbids dispatching the next batch (D3g/D3n)."""
        return self.write_outcome in WRITE_STOP_OUTCOMES

    @property
    def rung(self) -> Optional[int]:
        return RUNG_BY_STATUS.get(self.status)

    @property
    def fatal(self) -> bool:
        return self.status in FATAL_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "task_id": self.task_id,
            "kind": self.kind,
            "runner": self.runner,
            "profile": self.profile,
            "status": self.status,
            "attempt_count": len(self.attempts),
            "duration_seconds": round(self.duration_seconds, 3),
        }
        if self.rung is not None:
            out["rung"] = self.rung
        if self.output_path:
            out["output_path"] = self.output_path
        if self.reason:
            out["reason"] = self.reason
        if self.skipped:
            out["skipped"] = True
        if self.write_outcome:
            out["write_outcome"] = self.write_outcome
            out["files_touched"] = list(self.files_touched)
            if self.summary:
                out["summary"] = self.summary
            if self.snapshot_dir:
                out["snapshot_dir"] = self.snapshot_dir
        if self.usage_reported:
            out["usage"] = dict(self.usage or {})
        else:
            out["usage_reported"] = False
        if self.attempts:
            out["attempts"] = [a.to_dict() for a in self.attempts]
        return out


@dataclass
class BatchReport:
    """Every ``task_id`` in a batch with its status, attempts and runner (D3i)."""

    phase: str = ""
    kind: str = ""
    phase_dir: str = ""
    task_file: Optional[str] = None
    outcomes: List[TaskOutcome] = field(default_factory=list)
    ceilings: Dict[str, Any] = field(default_factory=dict)
    runner_versions: Dict[str, str] = field(default_factory=dict)
    invoker: str = "executor"

    @property
    def ok(self) -> bool:
        """True only when every task produced a schema-valid result file."""
        return bool(self.outcomes) and all(o.ok for o in self.outcomes)

    @property
    def failed(self) -> List[TaskOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        return counts

    def usage_totals(self) -> Dict[str, Any]:
        """Partial sum **plus** the count of tasks with no usage data (D3j).

        Never a total that pretends to be complete: a batch mixing a
        usage-reporting runner with a duration-only one reports both numbers.
        """
        totals: Dict[str, Any] = {}
        missing = 0
        for outcome in self.outcomes:
            if not outcome.usage_reported or not outcome.usage:
                missing += 1
                continue
            for key, value in outcome.usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
        result: Dict[str, Any] = dict(totals)
        result["tasks_without_usage"] = missing
        result["attempts"] = sum(len(o.attempts) for o in self.outcomes)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": 1,
            "invoker": self.invoker,
            "phase": self.phase,
            "kind": self.kind,
            "task_file": self.task_file,
            "ceilings": dict(self.ceilings),
            "runner_versions": dict(self.runner_versions),
            "counts": self.counts(),
            "usage": self.usage_totals(),
            "tasks": [o.to_dict() for o in self.outcomes],
        }

    def write(self, phase_dir: Optional[str] = None) -> str:
        """Write ``subagent_batch_report.json`` atomically; return its path."""
        target_dir = phase_dir or self.phase_dir
        path = os.path.join(target_dir, BATCH_REPORT_FILENAME)
        _atomic_write_json(path, self.to_dict())
        return path

    def failure_summary(self) -> List[str]:
        """One line per failure, for the phase's stdout summary (D3i)."""
        lines = []
        for outcome in self.failed:
            rung = outcome.rung
            rung_text = f"rung {rung}" if rung is not None else "no rung"
            lines.append(
                f"  - {outcome.task_id}: {outcome.status} ({rung_text}, "
                f"{len(outcome.attempts)} attempt(s), runner {outcome.runner}) "
                f"{outcome.reason or ''}".rstrip()
            )
        return lines


# ---------------------------------------------------------------------------
# Profiles, runners, ceilings
# ---------------------------------------------------------------------------

def profile_for_kind(kind: str) -> str:
    """The D3e invocation profile for a task kind.

    Derived here and nowhere else: the profile must never come from the task JSON,
    which embeds untrusted plan/diff/review text and could otherwise request
    escalation to ``agent-write``.
    """
    try:
        return PROFILE_BY_KIND[kind]
    except KeyError:
        raise ExecutorError(
            f"Unknown task kind {kind!r}: the executor dispatches only "
            f"{', '.join(sorted(PROFILE_BY_KIND))}. Nothing was dispatched."
        ) from None


def eligible_runners(profile: str) -> List[str]:
    """Runner specs on this host that could serve ``profile``.

    The intersection of the adapters declaring the profile with the binaries
    actually present on PATH. Printed as advice by every rejection; an empty
    intersection gets its own sentence rather than an empty list.
    """
    try:
        from .provider_registry import _PROVIDERS  # local import: avoids a cycle
    except ImportError:  # pragma: no cover
        from utils.provider_registry import _PROVIDERS
    names = []
    for name, provider in _PROVIDERS.items():
        if not provider.supports_profile(profile):
            continue
        try:
            if provider.is_available():
                names.append(name)
        except Exception:  # noqa: BLE001 - a probe must never break a refusal
            continue
    return sorted(names)


_REASON_EXPLANATION = {
    "unknown_provider": (
        "the provider name has no built-in adapter, so nothing can run it — "
        "this is a stale-config condition"
    ),
    "missing_capability": (
        "the adapter exists and runs fine, but does not declare the capability "
        "this phase needs — this is a capability condition, not a stale config"
    ),
    "missing_profile": (
        "the adapter exists and runs fine, but implements no invocation for this "
        "profile, and a fabricated one would hand an untrusted prompt write access"
    ),
    "binary_absent": (
        "the adapter is implemented but its CLI is not on PATH"
    ),
    "not_authenticated": (
        "the CLI is installed but not logged in — installed is not authenticated, "
        "and no retry can fix a missing credential"
    ),
    "nested_exec_denied": (
        "this host's capability data declares that a third-party agent CLI cannot "
        "be run from inside it at all — the executor refuses the run naming the "
        "field rather than dispatching into a sandbox that eats the failure"
    ),
    "nested_exec_spawn": (
        "the host's sandbox denied the spawn: the throwaway preflight dispatch "
        "could not start the runner as a child of this host's own shell"
    ),
    "nested_exec_network": (
        "the runner started inside this host but could not reach its provider "
        "API — the host's egress rules, not the runner, are what failed"
    ),
    "nested_exec_credentials": (
        "the runner started inside this host but could not read its own "
        "credential store — a login notice from inside a sandbox is a host "
        "condition as often as a logged-out CLI, and neither is retryable"
    ),
}

#: Substrings that identify a *host-side egress* failure in a nested probe's
#: streams (D3c ``nested_exec.network``). Deliberately narrower than the adapters'
#: own classifier sets: these are matched only on a throwaway probe whose entire
#: expected output is one fixed token, and every entry names a transport
#: condition a model answer has no reason to contain. Matched through
#: ``matches_any`` (boundary-aware), never as bare substrings.
NETWORK_DENIAL_PATTERNS: Tuple[str, ...] = (
    "network is unreachable",
    "no route to host",
    "connection refused",
    "connection reset by peer",
    "connection timed out",
    "could not connect",
    "unable to connect",
    "failed to connect",
    "temporary failure in name resolution",
    "name or service not known",
    "getaddrinfo",
    "eai_again",
    "econnrefused",
    "enetunreach",
    "certificate verify failed",
    "self signed certificate",
    "tls handshake",
    "proxy connect",
    "blocked by proxy",
    "egress blocked",
)

#: The throwaway prompt the nested-exec probe dispatches (D3c). Trivial on
#: purpose: it is a real, billed turn, so it asks for one token and nothing that
#: needs the workspace, and it runs under ``agent-readonly`` regardless of the
#: profile the batch itself will use — a probe must never hold write access.
NESTED_PROBE_PROMPT = (
    "Reply with exactly this token and nothing else: MULTI_LLM_NESTED_EXEC_OK"
)

#: Wall-clock budget for that probe. Long enough for a cold agent CLI start plus
#: one trivial turn, short enough that a wedged sandbox fails the preflight
#: rather than consuming the batch's own budget.
NESTED_PROBE_TIMEOUT_S = 120.0


def _render_runner_rejection(
    *, key: str, spec: str, reason_code: str, profile: str, detail: str = ""
) -> RunnerRejected:
    """The single rejection formatter (D3). Exit 3."""
    candidates = eligible_runners(profile)
    if candidates:
        advice = (
            f"Runners on this host that can serve {profile!r}: "
            f"{', '.join(candidates)}."
        )
    else:
        advice = (
            f"No runner on this host can serve {profile!r} — install one of the "
            f"supported CLIs (claude, codex, opencode) and authenticate it."
        )
    explanation = _REASON_EXPLANATION.get(reason_code, reason_code)
    message = (
        f"Refusing to dispatch: {key} is {spec!r} (reason: {reason_code}) — "
        f"{explanation}. {detail} {advice} "
        f"Fix: run --init to refresh a stale provider name, install the missing "
        f"CLI, or set {key} to one of the runners named above."
    ).replace("  ", " ")
    return RunnerRejected(message, key=key, spec=spec, reason_code=reason_code)


def resolve_task_runner(
    task: Dict[str, Any],
    *,
    anchor: Optional[str] = None,
    recorded: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve one task's runner spec, once, at batch start (D2e/D3o).

    Chain: a recorded runner from a previous attempt (so ``--resume`` and
    ``--retry-failed-subagents`` re-dispatch the runner the first attempt used,
    not one a since-edited config would pick) → the task's own ``model_hint`` when
    it is a concrete ``provider:model`` and not ``auto`` → ``capability_tier``
    (``cheap`` → ``driver.subagent_runner_cheap``) → ``driver.subagent_runner``.
    """
    task_id = str(task.get("task_id", ""))
    if recorded and task_id in recorded:
        return recorded[task_id]
    hint = task.get("model_hint")
    if isinstance(hint, str) and hint.strip() and hint.strip() != "auto" and ":" in hint:
        return hint.strip()
    tier = str(task.get("capability_tier") or "full")
    return get_subagent_runner(tier, anchor=anchor)


def preflight_runner(
    spec: str,
    *,
    profile: str = PROFILE_AGENT_READONLY,
    key: str = "driver.subagent_runner",
    anchor: Optional[str] = None,
    check_version: bool = True,
    check_auth: bool = True,
    check_nested: bool = True,
    probe_nested: bool = True,
) -> Dict[str, Any]:
    """Validate one runner spec; detect version, auth and nested-exec (D3c/D3m).

    Checks, earliest wins: the provider has an adapter, the model spec is valid
    for that provider, the adapter declares the requested profile, and (for a
    ``workspace-read`` phase) it declares ``supports_workspace_reads``. Then
    ``version_cmd`` runs once and its result is cached for the process: below
    ``min_version`` is a hard exit-3 refusal before any child is dispatched, above
    ``verified_version`` is a one-line warning (blocking on new releases would rot
    faster than the CLIs do), and unparseable output is recorded ``unknown`` with
    the same warning.

    The adapter's ``auth_check`` runs in the same preflight and is cached the same
    way (D3c): **installed is not authenticated** — every ``is_available()`` is a
    bare ``shutil.which``, so a logged-out CLI passes detection and would
    otherwise be discovered one spawned child at a time, once per task in the
    batch. A recognized login notice is exit 3 here, before anything is
    dispatched; anything the probe cannot read (no ``auth_check`` declared, the
    subcommand missing, the probe itself failing) is recorded ``unknown`` and
    never blocks, leaving the reactive ``auth_error_patterns`` path (rung 1) to
    catch it.

    Last comes the host's own ``nested_exec`` verdict (D3c), because presence,
    version and auth all describe the *runner* and none of them answers whether
    this host lets a third-party CLI run inside it with network and credentials
    intact. A declared ``denied`` leg is exit 3 naming the field — dispatching
    into a sandbox that eats the failure would spend the retry budget on every
    task in the batch — and a leg the data records as ``unknown`` or
    ``needs-config`` is *probed*, once per distinct host/runner/run, with one
    throwaway ``agent-readonly`` dispatch whose recognized denials (a refused
    spawn, an egress block, a credential notice from inside the sandbox) are the
    same exit 3 before anything is scheduled. An inconclusive probe is recorded
    ``unknown`` and never blocks, exactly like the version and auth probes.
    ``probe_nested=False`` keeps the declared verdict and suppresses only that
    throwaway dispatch — it is what ``--dry-run-subagents`` passes, because the
    probe is a real turn and "start no child" (D3l) means it.

    Returns:
        ``{"provider", "model", "version", "auth", "nested_exec",
        "usage_support", "profile"}``.

    Raises:
        RunnerRejected: exit 3, through the one rejection formatter.
    """
    provider_name, model = parse_model_spec(spec, anchor=anchor)
    provider = get_provider(provider_name)
    if provider is None:
        raise _render_runner_rejection(
            key=key, spec=spec, reason_code="unknown_provider", profile=profile
        )
    if not provider.supports_profile(profile):
        raise _render_runner_rejection(
            key=key, spec=spec, reason_code="missing_profile", profile=profile,
            detail=(
                f"{provider_name} declares "
                f"{', '.join(provider.agent_profiles) or 'no profiles'}."
            ),
        )
    if not provider.is_available():
        raise _render_runner_rejection(
            key=key, spec=spec, reason_code="binary_absent", profile=profile,
        )
    if not is_model_valid(spec, anchor=anchor):
        print(
            f"multi-llm warning: {key} names model {model!r}, which is not in "
            f"{provider_name}'s configured catalog — dispatching anyway.",
            file=sys.stderr,
        )

    info = {
        "provider": provider_name,
        "model": model,
        "profile": profile,
        "usage_support": provider.usage_support,
        "version": "unknown",
        "auth": "unknown",
        "nested_exec": "unknown",
    }
    if check_version:
        info["version"] = _detect_version(provider_name, key=key, spec=spec, profile=profile)
    if check_auth:
        info["auth"] = _detect_auth(provider_name, key=key, spec=spec, profile=profile)
    if check_nested:
        info["nested_exec"] = _detect_nested_exec(
            provider_name, model,
            key=key, spec=spec,
            # The version probe already started this CLI as a child of the host's
            # own shell, so a parsed version is a *spawn* observation and the
            # throwaway dispatch has one less leg to answer for.
            spawn_observed=info["version"] != "unknown",
            probe=probe_nested,
        )
    return info


#: Process-local version cache: detected once per *distinct* runner, never per
#: dispatch (D3m).
_VERSION_CACHE: Dict[str, str] = {}

#: The same, for the D3c auth probe: ``ok`` | ``unknown`` per provider. A failed
#: probe is never cached — it raises, and the run ends.
_AUTH_CACHE: Dict[str, str] = {}

#: The same, for the D3c nested-exec verdict, keyed by ``<host id>|<runner spec>``
#: rather than by provider: the question is about the *pair*, and a host that
#: denies one runner's egress may hand another its own bundled CLI. A denial is
#: never cached either — it raises.
_NESTED_CACHE: Dict[str, str] = {}


def reset_version_cache() -> None:
    """Clear the per-process runner-version, auth and nested-exec caches (tests)."""
    _VERSION_CACHE.clear()
    _AUTH_CACHE.clear()
    _NESTED_CACHE.clear()


def reset_auth_cache() -> None:
    """Clear the per-process runner-auth cache (tests)."""
    _AUTH_CACHE.clear()


def reset_nested_exec_cache() -> None:
    """Clear the per-process nested-exec cache (tests)."""
    _NESTED_CACHE.clear()


def _version_tuple(text: str) -> Tuple[int, ...]:
    parts = []
    for chunk in (text or "").split("."):
        match = re.match(r"^(\d+)", chunk)
        parts.append(int(match.group(1)) if match else 0)
    return tuple(parts) or (0,)


def _run_probe(argv: Sequence[str]) -> Optional["subprocess.CompletedProcess"]:
    """Run one cheap preflight argv (``version_cmd`` / ``auth_check``).

    Returns ``None`` when the probe could not be run at all, which every caller
    reads as "unknown": a preflight probe reports, it never blocks on its own
    failure.
    """
    launcher, _ = _resolve_executable(argv[0])
    try:
        return subprocess.run(
            [*launcher, *argv[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _detect_version(
    provider_name: str, *, key: str, spec: str, profile: str
) -> str:
    """Run ``version_cmd`` once per provider and enforce ``min_version`` (D3m)."""
    if provider_name in _VERSION_CACHE:
        return _VERSION_CACHE[provider_name]

    provider = get_provider(provider_name)
    version = "unknown"
    argv = list(getattr(provider, "version_cmd", ()) or ())
    if argv:
        completed = _run_probe(argv)
        if completed is not None:
            parsed = provider.parse_version(completed.stdout, completed.stderr)
            if parsed:
                version = parsed

    minimum = getattr(provider, "min_version", "") or ""
    verified = getattr(provider, "verified_version", "") or ""
    if version == "unknown":
        if argv:
            print(
                f"multi-llm warning: could not determine the {provider_name} "
                f"version from `{' '.join(argv)}` — recording 'unknown'. A CLI "
                f"that dropped --version still answers prompts, so this is not a "
                f"blocker, but a stream-shape failure will not name a build.",
                file=sys.stderr,
            )
    elif minimum and _version_tuple(version) < _version_tuple(minimum):
        raise _render_runner_rejection(
            key=key, spec=spec, reason_code="missing_capability", profile=profile,
            detail=(
                f"{provider_name} {version} is older than the minimum this "
                f"release parses ({minimum}); its event stream shape is not the "
                f"one the adapter was written against. Upgrade "
                f"{provider_name} to >= {minimum}."
            ),
        )
    elif verified and _version_tuple(version) > _version_tuple(verified):
        print(
            f"multi-llm warning: {provider_name} {version} is newer than the "
            f"release the shipped parser fixtures were captured from "
            f"({verified}); stream parsing is best-effort above that.",
            file=sys.stderr,
        )

    _VERSION_CACHE[provider_name] = version
    return version


def _detect_auth(
    provider_name: str, *, key: str, spec: str, profile: str
) -> str:
    """Run ``auth_check`` once per provider and refuse a logged-out CLI (D3c).

    Returns ``"ok"`` or ``"unknown"``; a recognized login/credential notice raises
    instead, because every sibling task in the batch would fail identically and no
    retry can fix a missing credential. ``unknown`` is the honest answer for a CLI
    that declares no probe (claude-code has no non-interactive status subcommand)
    or whose probe could not be read — never an assumption of success.
    """
    if provider_name in _AUTH_CACHE:
        return _AUTH_CACHE[provider_name]

    provider = get_provider(provider_name)
    argv = list(getattr(provider, "auth_check", ()) or ())
    if not argv:
        _AUTH_CACHE[provider_name] = "unknown"
        return "unknown"

    completed = _run_probe(argv)
    verdict: Optional[bool] = None
    if completed is not None:
        try:
            verdict = provider.classify_auth_probe(
                completed.stdout, completed.stderr, completed.returncode
            )
        except Exception:  # noqa: BLE001 - a probe must never break a preflight
            verdict = None
    if verdict is False:
        login = getattr(provider, "login_cmd", "") or f"{argv[0]} login"
        tail = _tail(
            (completed.stderr or completed.stdout) if completed else "", 500
        ).strip()
        raise RunnerRejected(
            f"Refusing to dispatch: {key} is {spec!r} (reason: not_authenticated) — "
            f"{_REASON_EXPLANATION['not_authenticated']}. "
            f"`{' '.join(argv)}` reported: {tail!r}. "
            f"Fix: run `{login}`, then re-run this phase. Nothing was dispatched.",
            key=key,
            spec=spec,
            reason_code="not_authenticated",
        )

    state = "ok" if verdict else "unknown"
    _AUTH_CACHE[provider_name] = state
    return state


def _host_nested_exec() -> Optional[Tuple[str, Dict[str, str], str]]:
    """``(host_id, {leg: state}, notes)`` for the running host, or ``None``.

    ``None`` means the capability data could not be read at all (a partial
    install, a direct script run without ``utils`` importable). That is the one
    case where the nested-exec leg is skipped outright rather than probed: a
    throwaway dispatch is a real billed turn, and "I do not know which host I am
    on" is not a reason to spend one — nor to refuse, since the legs it would
    have answered are exactly the ones the reactive D3k ladder still catches.
    """
    try:
        from . import host_capabilities as caps
    except ImportError:  # pragma: no cover - direct script runs
        try:
            import host_capabilities as caps  # type: ignore
        except ImportError:
            return None
    try:
        host_id = caps.resolve_host_id(sys.argv[1:])
        declared = caps.nested_exec(host_id)
    except Exception:  # pragma: no cover - unknown host / missing data
        return None
    notes = str(declared.pop("notes", "") or "").strip()
    legs = {
        leg: str(declared.get(leg) or "unknown")
        for leg in caps.NESTED_EXEC_LEGS
    }
    return host_id, legs, notes


def _detect_nested_exec(
    provider_name: str,
    model: str,
    *,
    key: str,
    spec: str,
    spawn_observed: bool = False,
    probe: bool = True,
) -> str:
    """Can a runner really run *inside this host*? Once per host/runner (D3c).

    Presence, version and auth are all facts about the runner; this is the one
    preflight leg about the **host**. Three verdicts:

    * a declared ``denied`` leg → exit 3 naming the field and its notes, before
      anything is scheduled. Every task in the batch would fail identically, and
      a sandbox that denies a spawn or an egress usually eats the failure into
      something the D3k ladder reads as generic runner noise and *retries*.
    * every leg ``allowed`` → ``"allowed"``, with no child spawned. The data has
      already answered; a probe would only spend a billed turn re-confirming it.
    * anything ``unknown`` or ``needs-config`` → one throwaway ``agent-readonly``
      dispatch, cached per ``host|spec`` for the process, whose recognized
      denials are the same exit 3 and whose inconclusive outcomes are
      ``"unknown"`` (a preflight probe reports; it never blocks on its own
      failure).
    """
    resolved = _host_nested_exec()
    if resolved is None:
        return "unknown"
    host_id, legs, notes = resolved

    denied = [leg for leg in legs if legs[leg] == "denied"]
    if denied:
        raise RunnerRejected(
            f"Refusing to dispatch: host {host_id!r} declares nested_exec "
            f"{', '.join(denied)} denied (reason: nested_exec_denied) — "
            f"{_REASON_EXPLANATION['nested_exec_denied']}. "
            + (f"{notes} " if notes else "")
            + f"Fix: drop --exec-subagents and let this host's own subagent "
            f"mechanism run the phase, or run it from a host whose nested_exec "
            f"permits a nested CLI. Nothing was dispatched.",
            key=key,
            spec=spec,
            reason_code="nested_exec_denied",
        )

    cache_key = f"{host_id}|{spec}"
    cached = _NESTED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    unresolved = [leg for leg in legs if legs[leg] != "allowed"]
    if spawn_observed and unresolved == ["spawn"]:
        # The version probe already ran this CLI as a child of this host's shell,
        # which is the whole of the spawn leg. Nothing else is open, so there is
        # nothing left for a billed turn to answer.
        unresolved = []
    if not unresolved:
        _NESTED_CACHE[cache_key] = "allowed"
        return "allowed"
    if not probe:
        # Deliberately uncached: the declared verdict has not been answered, and a
        # later real dispatch in this same process must still get its probe.
        return "unknown"

    state = _probe_nested_exec(
        provider_name, model,
        key=key, spec=spec, host_id=host_id, legs=unresolved, notes=notes,
    )
    _NESTED_CACHE[cache_key] = state
    return state


def _probe_nested_exec(
    provider_name: str,
    model: str,
    *,
    key: str,
    spec: str,
    host_id: str,
    legs: Sequence[str],
    notes: str,
) -> str:
    """The throwaway dispatch behind an ``unknown``/``needs-config`` leg (D3c).

    One trivial ``agent-readonly`` turn, through the *same* :func:`_spawn` the
    batch uses — same argv builder, same composed environment, same host env
    guards — because a probe that composed its own child would answer a question
    about a process the batch never runs. Only a recognized denial refuses:

    * the child could not be started at all → the spawn leg, exit 3;
    * a login/credential notice → the credentials leg, exit 3 (from inside a
      sandbox this is as often the host's doing as a logged-out CLI, and neither
      is retryable);
    * an egress failure → the network leg, exit 3.

    Everything else — a timeout, a non-zero exit with nothing recognized, a
    silent zero-byte success, an adapter that will not render the probe — is
    ``unknown`` plus one warning line. The reactive ladder still covers those,
    and blocking a batch on an unreadable probe would be worse than the dispatch
    the probe exists to save.
    """
    provider = get_provider(provider_name)
    if provider is None:  # pragma: no cover - presence is checked before this
        return "unknown"
    opened = "/".join(legs)
    try:
        capture = _spawn(
            provider=provider,
            model=model,
            prompt=NESTED_PROBE_PROMPT,
            # Always read-only, never the batch's own profile: a preflight has no
            # result to write, so it has no reason to hold write access.
            profile=PROFILE_AGENT_READONLY,
            transport=preferred_prompt_transport(provider_name),
            timeout=NESTED_PROBE_TIMEOUT_S,
            cwd=None,
        )
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never blocks
        print(
            f"multi-llm warning: could not render the nested-exec probe for "
            f"{spec} on host {host_id} ({exc}) — recording nested_exec "
            f"'unknown' for {opened}.",
            file=sys.stderr,
        )
        return "unknown"

    trailer = (
        f"Host: {host_id} (nested_exec {opened} was not recorded as allowed). "
        + (f"{notes} " if notes else "")
    )

    if capture.launch_error is not None:
        raise RunnerRejected(
            f"Refusing to dispatch: {key} is {spec!r} (reason: nested_exec_spawn) "
            f"— {_REASON_EXPLANATION['nested_exec_spawn']}: "
            f"{capture.launch_error}. {trailer}"
            f"Fix: grant this host's sandbox permission to start the runner, or "
            f"run the phase without --exec-subagents. Nothing was dispatched.",
            key=key,
            spec=spec,
            reason_code="nested_exec_spawn",
        )

    streams = f"{capture.stdout or ''}\n{capture.stderr or ''}".lower()

    if capture.timed_out:
        print(
            f"multi-llm warning: the nested-exec probe for {spec} on host "
            f"{host_id} did not finish within {NESTED_PROBE_TIMEOUT_S:.0f}s — "
            f"recording nested_exec 'unknown' for {opened}. A wedged sandbox "
            f"will resurface as a per-task timeout (D3k rung 2).",
            file=sys.stderr,
        )
        return "unknown"

    classified = _classify_failed_run(provider, capture)
    if classified is not None and classified[0] == STATUS_AUTH_FAILED:
        login = getattr(provider, "login_cmd", "") or f"{provider_name} login"
        raise RunnerRejected(
            f"Refusing to dispatch: {key} is {spec!r} (reason: "
            f"nested_exec_credentials) — "
            f"{_REASON_EXPLANATION['nested_exec_credentials']}. "
            f"Probe tail: {_tail(capture.stderr or capture.stdout, 500).strip()!r}. "
            f"{trailer}"
            f"Fix: run `{login}` from outside this host, or allow the runner to "
            f"read its credential store from inside it. Nothing was dispatched.",
            key=key,
            spec=spec,
            reason_code="nested_exec_credentials",
        )

    if matches_any(NETWORK_DENIAL_PATTERNS, streams):
        raise RunnerRejected(
            f"Refusing to dispatch: {key} is {spec!r} (reason: "
            f"nested_exec_network) — "
            f"{_REASON_EXPLANATION['nested_exec_network']}. "
            f"Probe tail: {_tail(capture.stderr or capture.stdout, 500).strip()!r}. "
            f"{trailer}"
            f"Fix: allow this host's sandbox to reach the provider API (a domain "
            f"allow-list entry, a proxy setting), or run the phase without "
            f"--exec-subagents. Nothing was dispatched.",
            key=key,
            spec=spec,
            reason_code="nested_exec_network",
        )

    if capture.exit_code == 0 and streams.strip():
        return "allowed"

    print(
        f"multi-llm warning: the nested-exec probe for {spec} on host {host_id} "
        f"was inconclusive (exit {capture.exit_code}, "
        f"{len(streams.strip())} byte(s) of output) — recording nested_exec "
        f"'unknown' for {opened}. Dispatching anyway; a real denial will land on "
        f"the D3k ladder.",
        file=sys.stderr,
    )
    return "unknown"


def _host_env_guards() -> Tuple[List[str], List[str]]:
    """``(strip, passthrough)`` from the running host's ``env_guards`` (D3c/D5b).

    Under ``subagents: executor`` the runner is a *grandchild* — the host's shell
    tool spawned this dispatcher, this dispatcher spawns the runner — so the child
    inherits the host's injected variables on top of the runner's own: the
    nested-session marker that makes a same-binary runner refuse to start, and the
    session/skill ids that make it join or overwrite the parent's session instead
    of running its own turn. The adapter's ``get_remove_env()`` only knows about
    the *runner*, so the strip list is the union of the two, composed here at the
    single spawn site.

    Deliberately non-fatal, like :func:`_host_ceiling`: a partial install that
    cannot answer falls back to the adapter's own list rather than failing a
    dispatch.
    """
    try:
        from . import host_capabilities as caps
    except ImportError:  # pragma: no cover - direct script runs
        try:
            import host_capabilities as caps  # type: ignore
        except ImportError:
            return [], []
    try:
        declared = caps.env_guards(caps.resolve_host_id(sys.argv[1:]))
    except Exception:  # pragma: no cover - unknown host / missing data
        return [], []
    return (
        [*declared["guards"], *declared["session_scoped"]],
        list(declared["passthrough"]),
    )


def _host_ceiling() -> Optional[int]:
    """This host's ``max_concurrent_subagents``, or ``None`` if it cannot be read.

    Deliberately non-fatal: the ceiling is a *bound*, and a partial install that
    cannot answer must not turn a dispatch into a hard failure — the driver and
    provider legs of the minimum still apply.
    """
    try:
        from . import host_capabilities as caps
    except ImportError:  # pragma: no cover - direct script runs
        try:
            import host_capabilities as caps  # type: ignore
        except ImportError:
            return None
    try:
        return caps.max_concurrent_subagents(caps.resolve_host_id())
    except Exception:  # pragma: no cover - unknown host / missing data
        return None


def effective_ceiling(
    runners: Sequence[str],
    *,
    anchor: Optional[str] = None,
    host_max: Optional[int] = None,
) -> Dict[str, Any]:
    """The three-way minimum, computed per dispatch and logged once (D3i).

    ``driver.max_concurrent_subagents`` (user) × the runner provider's
    ``max_concurrent`` × the host's own ceiling from the capability data. Returned
    as a record rather than a bare int so the batch report and the log line can
    both say *which* ceiling bound the batch.

    Args:
        host_max: The host's ``max_concurrent_subagents``. ``None`` means "look it
            up from ``hosts/capabilities.yaml`` for the host this invocation
            declared" (D5b) — the seam is this parameter, not a second code path,
            and a caller that already knows the number still passes it directly.
    """
    if host_max is None:
        host_max = _host_ceiling()
    driver_gate = get_max_concurrent_subagents(anchor=anchor)
    provider_limits: Dict[str, Optional[int]] = {}
    for spec in dict.fromkeys(runners):
        provider_name, _ = parse_model_spec(spec, anchor=anchor)
        provider_limits[provider_name] = get_provider_max_concurrent(
            provider_name, anchor=anchor
        )

    candidates = [driver_gate]
    if host_max is not None:
        candidates.append(host_max)
    limits = [limit for limit in provider_limits.values() if limit is not None]
    if limits:
        candidates.append(max(limits))
    effective = max(1, min(candidates))
    return {
        "effective": effective,
        "driver_max_concurrent_subagents": driver_gate,
        "host_max_concurrent_subagents": host_max,
        "provider_max_concurrent": {
            name: limit for name, limit in provider_limits.items()
        },
    }


# ---------------------------------------------------------------------------
# D3f — containment and the atomic write
# ---------------------------------------------------------------------------

def _is_unsafe_windows_path(name: str) -> bool:
    """Drive-relative (``C:foo``), UNC (``\\\\host\\share``) or an ADS (``a.json:x``)."""
    if name.startswith("\\\\") or name.startswith("//"):
        return True
    if re.match(r"^[A-Za-z]:", name):
        return True
    # An NTFS alternate data stream: a colon anywhere in a *basename*.
    return ":" in name


def resolve_output_target(task: Dict[str, Any], phase_dir: str) -> Path:
    """Resolve and contain a task's ``output_name`` under ``phase_dir`` (D3f).

    ``output_name`` resolved against the dir the task file was actually loaded
    from is authoritative; an absolute ``output_path`` in the JSON is accepted
    only when it resolves to that same file. Task files are persisted in the
    target workspace, survive across runs and (after D2) embed untrusted text, so
    a stale, hand-edited or injected path is an arbitrary-file-overwrite
    primitive.

    Raises:
        ResultRejected: status ``rejected`` (rung 7, no child started), naming the
            task, the offending path and the rule violated.
    """
    task_id = task.get("task_id", "<unnamed>")
    kind = task.get("kind", "")
    name = task.get("output_name")

    def _reject(rule: str, offending: Any) -> "ResultRejected":
        return ResultRejected(
            f"Task {task_id}: refusing the output path {offending!r} — {rule}. "
            f"Phase dir: {phase_dir}. Nothing was written and nothing was "
            f"dispatched.",
            STATUS_REJECTED,
        )

    if not isinstance(name, str) or not name:
        raise _reject("output_name must be a non-empty string", name)
    if os.path.basename(name) != name or _is_unsafe_windows_path(name):
        raise _reject(
            "output_name must be a bare basename (no separators, no drive-relative "
            "or UNC prefix, no alternate data stream)",
            name,
        )
    if (
        name in RESERVED_OUTPUT_NAMES
        or any(name.startswith(prefix) for prefix in RESERVED_OUTPUT_PREFIXES)
        or any(name.endswith(suffix) for suffix in RESERVED_OUTPUT_SUFFIXES)
    ):
        raise _reject(
            "it collides with an orchestrator-owned artifact (the task file, "
            "grouped.json, state, the run log or the batch report)",
            name,
        )
    pattern = OUTPUT_NAME_PATTERN_BY_KIND.get(kind)
    if pattern is None:
        raise _reject(f"kind {kind!r} is not one the executor dispatches", name)
    if not pattern.match(name):
        raise _reject(
            f"it does not match the {kind} output pattern {pattern.pattern!r}, so "
            f"the reaggregation glob could be fed a foreign file",
            name,
        )
    # `salvage`'s pattern is necessarily broad — its output IS a model result file
    # — so it must not be allowed to land on another kind's result. Without this a
    # hand-edited or injected salvage task could overwrite validation_batch_0.json
    # with a suggestion array that the merge would then read as validation.
    if kind == "salvage":
        for other_kind, other_pattern in OUTPUT_NAME_PATTERN_BY_KIND.items():
            if other_kind != "salvage" and other_pattern.match(name):
                raise _reject(
                    f"it is the {other_kind} result filename, which a salvage "
                    f"task may never write",
                    name,
                )

    resolved_dir = Path(phase_dir).resolve()
    # Refuse to write *through* a symlink at the final component, before the
    # containment test — otherwise a link pointing outside the phase dir would be
    # reported as an "outside the phase dir" escape and the fix ("stop symlinking
    # the result file") would not be in the message.
    if (resolved_dir / name).is_symlink():
        raise _reject("the target is a symlink", str(resolved_dir / name))
    # Resolve AFTER joining, so a symlinked path component cannot redirect the
    # write, then require containment beneath the resolved phase dir.
    candidate = (resolved_dir / name).resolve()
    try:
        contained = candidate.is_relative_to(resolved_dir)
    except AttributeError:  # pragma: no cover - Python < 3.9
        contained = os.path.commonpath([str(candidate), str(resolved_dir)]) == str(
            resolved_dir
        )
    if not contained or candidate == resolved_dir:
        raise _reject("it resolves outside the phase directory", str(candidate))

    declared = task.get("output_path")
    if isinstance(declared, str) and declared:
        declared_path = Path(declared)
        if not declared_path.is_absolute():
            declared_path = resolved_dir / declared_path
        # Resolve BOTH sides (D1c: asymmetry is the bug). ``make_task`` derives
        # ``output_path`` from the *unresolved* phase-dir string, so whenever that
        # path crosses a symlink — a symlinked repo root, macOS ``/tmp`` →
        # ``/private/tmp``, a CI workspace link — a normpath-only comparison
        # differs from the resolved candidate and every task in the phase is
        # refused before dispatch. D3f accepts an absolute ``output_path`` that
        # RESOLVES to the same file, which is what this compares.
        try:
            declared_resolved = declared_path.resolve()
        except OSError:  # pragma: no cover - unresolvable path is a mismatch
            declared_resolved = Path(os.path.normpath(str(declared_path)))
        if declared_resolved != candidate:
            raise _reject(
                "the task's absolute output_path does not resolve to "
                f"{candidate} (names are authoritative, paths are a "
                "convenience field)",
                declared,
            )

    return resolved_dir / name


def _atomic_write_json(path: str, data: Any) -> None:
    """Write ``data`` as JSON via a same-directory temp file, 0600, ``os.replace``.

    An executor killed mid-batch (expected on the D4 detached path) therefore
    never leaves a half-written result for the reaggregation glob to pick up.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = json.dumps(data, indent=2)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    handle = os.open(str(tmp), flags, 0o600)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    os.replace(str(tmp), str(target))
    try:
        os.chmod(str(target), 0o600)
    except OSError:  # pragma: no cover - exotic filesystems
        pass


def persist_result(task: Dict[str, Any], phase_dir: str, data: Any) -> str:
    """The invoker-persists leg of the D3a contract. Returns the path written.

    **Shared by both invokers.** The executor calls it with the JSON it extracted
    from captured stdout; :func:`simulate_native_invoker` calls it with the JSON a
    driving agent's subagent returned. Same containment, same serialization, same
    bytes — which is what makes the result-contract equivalence assertion possible
    rather than aspirational.

    Deliberately does **not** branch on ``result_channel``: ``file`` is the only
    behavior the executor has, so a task carrying ``inline-json`` is refused
    before dispatch instead of selecting a second write path here.
    """
    target = resolve_output_target(task, phase_dir)
    encoded = json.dumps(data, indent=2)
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ResultRejected(
            f"Task {task.get('task_id')}: the extracted result is "
            f"{len(encoded.encode('utf-8'))} bytes, over the {MAX_RESULT_BYTES}-byte "
            f"cap. Rejected rather than truncated — a truncated result file would "
            f"still parse and would reaggregate as a real one.",
            STATUS_SCHEMA_INVALID,
        )
    _atomic_write_json(str(target), data)
    return str(target)


# ---------------------------------------------------------------------------
# Extraction and per-kind result validation
# ---------------------------------------------------------------------------

def extract_result(provider: Any, stdout: str, stderr: str, *, prefer_arrays: bool) -> Any:
    """Recover the subagent's answer: ``parse_output`` first, then extraction.

    The ordering is **mandatory, not an optimization** (D3): provider CLIs emit
    structured event streams, so running ``json_extractor`` over raw stdout would
    match an event envelope or a reasoning item instead of the answer, and
    silently write a garbage result that reaggregation would accept.

    Adapters already run ``extract_json_from_text`` on the text they recover, so
    a dict/list ``data`` is the extracted result; a *string* ``data`` (adapters
    that return raw text) goes through ``utils/json_extractor`` here.

    Raises:
        ResultRejected: rung 4 (``runner_error``) when the adapter recovered no
            response text at all, rung 5 (``unparseable_result``) when it recovered
            text but no result object.
    """
    if len(stdout.encode("utf-8", errors="replace")) > MAX_STREAM_BYTES:
        raise ResultRejected(
            f"the runner produced more than {MAX_STREAM_BYTES} bytes on stdout; "
            f"refusing to parse it rather than truncating a stream into a result",
            STATUS_UNPARSEABLE,
        )

    parsed = provider.parse_output(stdout, stderr)
    if not isinstance(parsed, dict):  # pragma: no cover - adapter contract
        raise ResultRejected(
            "the adapter's parse_output did not return a result dict",
            STATUS_RUNNER_ERROR,
        )

    if not parsed.get("success"):
        raw = parsed.get("raw")
        error = parsed.get("error") or "parse_output reported failure"
        if isinstance(raw, str) and raw.strip():
            # Text WAS recovered; it just carries no result object (prose, a
            # refusal, a truncated stream) → rung 5.
            raise ResultRejected(str(error), STATUS_UNPARSEABLE)
        reason = None
        try:
            reason = provider.describe_failure(stdout, stderr)
        except Exception:  # noqa: BLE001 - diagnostics must not raise
            reason = None
        if reason:
            error = f"{error}: {reason}"
        raise ResultRejected(str(error), STATUS_RUNNER_ERROR)

    data = parsed.get("data")
    if isinstance(data, str):
        extracted = extract_json_from_text(data, prefer_arrays=prefer_arrays)
        if not extracted.get("success"):
            raise ResultRejected(
                extracted.get("error") or "no JSON found in the recovered text",
                STATUS_UNPARSEABLE,
            )
        data = extracted.get("data")
    if data is None or (isinstance(data, dict) and set(data.keys()) == {"raw"}):
        raise ResultRejected(
            "the adapter recovered no result object from the stream",
            STATUS_UNPARSEABLE,
        )
    if not isinstance(data, (dict, list)):
        raise ResultRejected(
            f"the recovered result is a {type(data).__name__}, not a JSON object "
            f"or array",
            STATUS_UNPARSEABLE,
        )
    return data


def _validation_result_errors(data: Any) -> List[str]:
    """Structural check for a ``validation_batch_N.json`` / revalidation result.

    There is no shipped JSON Schema for these (the merge deliberately tolerates
    several index shapes and joins by ``group_hash``), so this pins the two things
    the merge cannot recover from: it must be a wrapper object with a ``groups``
    array (or a bare array), and every entry must carry a ``status`` plus the
    ``group_hash``/``group_id`` the merge joins on.
    """
    if isinstance(data, dict):
        groups = data.get("groups")
        if groups is None:
            return ["expected a `groups` array (or a bare array of group results)"]
    elif isinstance(data, list):
        groups = data
    else:
        return [f"expected an object or array, got {type(data).__name__}"]
    if not isinstance(groups, list):
        return [f"`groups` must be an array, got {type(groups).__name__}"]
    errors: List[str] = []
    for index, entry in enumerate(groups):
        if not isinstance(entry, dict):
            errors.append(f"groups[{index}]: expected an object")
            continue
        if not entry.get("status"):
            errors.append(f"groups[{index}]: missing `status`")
        if not (entry.get("group_hash") or entry.get("group_id")):
            errors.append(
                f"groups[{index}]: missing `group_hash`/`group_id` — the merge "
                f"joins by hash, never by position"
            )
    return errors


def _salvage_result_errors(data: Any) -> List[str]:
    """A salvaged model result is the same array shape a model review produces."""
    if not isinstance(data, list):
        return [f"expected a JSON array of suggestions, got {type(data).__name__}"]
    errors = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            errors.append(f"[{index}]: expected an object")
    return errors


def validate_result(kind: str, data: Any) -> List[str]:
    """Validate an extracted result against its kind's result shape (D3f).

    Runs *before* anything is written, so a runner that returned prose or the
    wrong object fails the task instead of writing garbage that reaggregation
    accepts. Shipped JSON Schemas are used where they exist; the two kinds with
    no schema get the structural check their consumer actually depends on.
    """
    if kind in ("validation", "revalidation"):
        return _validation_result_errors(data)
    if kind == "consolidation":
        return validate_with_schema_file(data, "consolidation_batch.schema.json")
    if kind == "human_decision_judge":
        return validate_human_decision_verdicts(data)
    if kind == "task_generation":
        return validate_with_schema_file(data, "generated_tasks.schema.json")
    if kind == "salvage":
        return _salvage_result_errors(data)
    if kind in WRITE_KINDS:
        # D3g's "the batch still returns JSON on stdout" made concrete: the same
        # schema gate a validation result gets, so a runner that narrated its work
        # in prose fails the task instead of being counted as applied.
        return validate_with_schema_file(data, "write_batch_result.schema.json")
    return [f"unknown kind {kind!r}"]


def task_is_complete(task: Dict[str, Any], phase_dir: str) -> bool:
    """Whether a task counts as already done, deterministically (D3i).

    Done **iff** its ``output_name`` exists under the phase dir, parses, and
    validates against its kind's result schema. Anything else is re-dispatched
    under its original ``task_id``. Orphans are defined by artifacts, never by
    memory: there is no in-memory queue to recover and no "running" state to
    trust.
    """
    try:
        target = resolve_output_target(task, phase_dir)
    except ResultRejected:
        return False
    if not target.exists() or target.stat().st_size == 0:
        return False
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return not validate_result(str(task.get("kind", "")), data)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _tail(text: str, limit: int = MAX_CAPTURE_BYTES) -> str:
    """The last ``limit`` bytes of ``text``, decoded leniently for diagnostics."""
    encoded = (text or "").encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text or ""
    return encoded[-limit:].decode("utf-8", errors="replace")


def _safe_task_filename(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", task_id or "task") or "task"


def _write_failure_capture(
    *,
    phase_dir: str,
    task: Dict[str, Any],
    attempt_index: int,
    argv: Sequence[str],
    prompt: str,
    stdout: str,
    stderr: str,
    exit_code: Optional[int],
    duration: float,
    status: str,
    error: str,
) -> Optional[str]:
    """Persist one failed attempt's bounded capture (D3k). Best-effort.

    A detached or CI failure has no chat context and no stdout to scroll back
    through, so the diagnosis has to survive on disk. The rendered argv carries
    the prompt **elided to its sha256 and byte length** — never the prompt bytes,
    which embed the plan/diff under review.
    """
    import hashlib

    try:
        capture_dir = Path(phase_dir) / FAILED_CAPTURE_DIRNAME
        name = f"{_safe_task_filename(str(task.get('task_id', '')))}.attempt{attempt_index}.json"
        digest = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
        rendered = [
            f"<prompt sha256:{digest} bytes:{len((prompt or '').encode('utf-8'))}>"
            if arg == prompt
            else arg
            for arg in argv
        ]
        _atomic_write_json(str(capture_dir / name), {
            "task_id": task.get("task_id"),
            "kind": task.get("kind"),
            "attempt": attempt_index,
            "status": status,
            "rung": RUNG_BY_STATUS.get(status),
            "error": error,
            "exit_code": exit_code,
            "duration_seconds": round(duration, 3),
            "argv": rendered,
            "prompt_sha256": digest,
            "prompt_bytes": len((prompt or "").encode("utf-8")),
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        })
        return str(capture_dir / name)
    except OSError:  # pragma: no cover - diagnostics must never fail a run
        return None


@dataclass
class _Capture:
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    duration: float = 0.0
    timed_out: bool = False
    launch_error: Optional[str] = None
    argv: List[str] = field(default_factory=list)


def _spawn(
    *,
    provider: Any,
    model: str,
    prompt: str,
    profile: str,
    transport: str,
    timeout: float,
    cwd: Optional[str],
) -> _Capture:
    """Run one dispatch: build argv for the profile, spawn, capture everything.

    ``cwd`` is passed explicitly (never inherited) so the runner's sandbox root is
    the intended repository and a task cannot be steered into ``$HOME`` or a
    sibling checkout (D3e). Env composition is the adapter's
    ``get_env``/``get_remove_env`` pair — the same single site the fan-out path and
    the detached child use, so there is never a second hand-rolled env filter.
    """
    capture = _Capture()
    argv, stdin_text = provider.build_agent_command_for_transport(
        prompt, model, profile, transport
    )
    # The batch-shim flag is consumed pre-dispatch by `_prompt_over_budget`, which
    # resolves argv[0] the same way: it applies cmd.exe's 8,191-unit cap and
    # refuses a metacharacter-bearing prompt (rung 7, nothing spawned) before this
    # runs, so by here the prompt is already known to be safe for this launcher.
    launcher, _is_batch_shim = _resolve_executable(argv[0])
    argv = [*launcher, *argv[1:]]
    capture.argv = list(argv)

    provider_env = provider.get_env(model)
    host_guards, passthrough = _host_env_guards()
    # The union, not the adapter's list alone (D3c): the adapter guards against
    # its own nesting marker, the host entry declares what *this* host injects
    # into every child. A name the host sanctions for the runner (`passthrough`,
    # e.g. an API key it owns) is re-admitted last, because over-stripping turns a
    # working runner into an auth failure — which is the outcome the guards exist
    # to avoid, not a safe default.
    remove_env = [
        name
        for name in dict.fromkeys([*(provider.get_remove_env() or ()), *host_guards])
        if name not in passthrough
    ]
    if provider_env or remove_env:
        env = {**os.environ, **provider_env}
        for key in remove_env:
            env.pop(key, None)
    else:
        env = None

    start = time.time()
    # `Popen`, not `subprocess.run`: the timeout has to kill the child *with its
    # process group* (D3k rung 2), and `run` neither hands out the pid nor kills
    # anything but the direct child. An agentic runner is not a leaf — its tool
    # children (a shell, a nested provider CLI) inherit both the group and this
    # stdout pipe — so `run`'s leader-only kill leaves them running against a
    # task nobody is reading, and its own post-kill drain is unbounded on
    # Windows: a timed-out task wedges the whole batch instead of failing on
    # rung 2.
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cwd,
            **process_control.new_process_group_kwargs(),
        )
    except OSError as exc:
        capture.duration = time.time() - start
        capture.launch_error = str(exc)
        return capture

    # Read the group now, while the child is alive: `os.getpgid` stops answering
    # the moment the leader is reaped, and the group — the thing that still holds
    # the tool children — outlives it. `None` on Windows, where `kill_tree` walks
    # the tree instead.
    pgid = process_control.process_group_id(proc.pid)
    try:
        stdout, stderr = proc.communicate(input=stdin_text, timeout=timeout)
        capture.duration = time.time() - start
        capture.stdout = _decode(stdout)
        capture.stderr = _decode(stderr)
        capture.exit_code = proc.returncode
        return capture
    except subprocess.TimeoutExpired as exc:
        _kill_timed_out(proc, pgid)
        stdout, stderr = _drain_after_kill(proc, exc)
        capture.duration = time.time() - start
        capture.timed_out = True
        capture.stdout = stdout
        capture.stderr = stderr
        return capture
    except OSError as exc:
        _kill_timed_out(proc, pgid)
        _drain_after_kill(proc, None)
        capture.duration = time.time() - start
        capture.launch_error = str(exc)
        return capture


def _kill_timed_out(proc: Any, pgid: Optional[int]) -> None:
    """SIGTERM → SIGKILL the child *and its group*, through the one facade.

    ``start_time=None`` is how an unverified kill is *requested* (see
    ``process_control.kill_tree``): this pid is not a record read off disk, it is
    a child this process is still holding open, so there is nothing to
    corroborate and nothing to recycle. The verdict is advisory — the child is
    an unreaped zombie until :func:`_drain_after_kill` collects it, so the probe
    that decides it cannot see the difference; what matters is that the signals
    reached the group.
    """
    try:
        process_control.kill_tree(
            proc.pid,
            start_time=None,
            pgid=pgid,
            escalate_after_s=TIMEOUT_KILL_GRACE_S,
            poll_interval_s=0.05,
        )
    except Exception:  # noqa: BLE001 - teardown must never mask the timeout
        try:
            proc.kill()
        except OSError:
            pass


def _drain_after_kill(
    proc: Any, exc: Optional[subprocess.TimeoutExpired]
) -> Tuple[str, str]:
    """Collect the killed child's output and reap it, without ever hanging.

    A second ``communicate()`` returns everything buffered, including what the
    first one had already read before it timed out. If even *that* blocks, a
    grandchild the kill did not reach is still holding the pipe: fall back to the
    partial capture the ``TimeoutExpired`` carries and let the descriptors go,
    because a diagnostic tail is never worth wedging the batch for.
    """
    try:
        stdout, stderr = proc.communicate(timeout=TIMEOUT_DRAIN_S)
        return _decode(stdout), _decode(stderr)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):  # pragma: no cover - best effort
                pass
        return (
            _decode(getattr(exc, "stdout", None)),
            _decode(getattr(exc, "stderr", None)),
        )


def _decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def dispatch_task(
    task: Dict[str, Any],
    *,
    phase_dir: str,
    runner: str,
    anchor: Optional[str] = None,
    cwd: Optional[str] = None,
    retries: Optional[int] = None,
    timeout: Optional[int] = None,
    sleep: Optional[Callable[[float], None]] = None,
    prompt: Optional[str] = None,
    before_attempt: Optional[Callable[[Dict[str, Any]], None]] = None,
    after_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> TaskOutcome:
    """Dispatch one task and return its :class:`TaskOutcome` (D3k classified).

    Synchronous by design: the concurrency lives in
    ``review_orchestrator_base.run_tasks_concurrent``, which runs this in a
    thread, so one dispatch is a straight line that is readable and unit-testable
    on its own.

    Retries are bounded and only for transport-shaped failures — a timeout, a run
    yielding no parsed text, or a ``parse_output`` failure. A result that parsed
    but failed schema validation is **never** retried: that is a deterministic
    disagreement a retry would only launder. A retried task keeps its ``task_id``,
    so the task-keyed result map and reaggregation are unaffected.

    ``timeout`` (or ``driver.subagent_timeout_s``) is the task's **total**
    wall-clock budget, not a per-attempt one: a monotonic deadline is fixed at
    entry, every attempt is spawned with what is left of it, backoff sleeps are
    spent from it, and a retry is refused once no usable slice remains. So a
    dispatch is bounded by the configured ceiling however many times it retries.

    Args:
        prompt: The prompt to dispatch, overriding ``task["prompt"]``. This is the
            D3n seam: an apply prompt is complete except for
            ``{prior_changes_context}``, whose value is a function of the batches
            that already finished, so the scheduler substitutes it and hands the
            *dispatched* prompt in here. The task JSON stays the immutable
            hash-bound snapshot D2g made it — this never rewrites it.
        before_attempt: Called once per attempt, immediately **before** the spawn,
            with ``{task, profile, runner, cwd, attempt, argv, prompt}``. The write
            path's pre-batch snapshot, its pre-dispatch write-policy check and its
            ``dispatched`` audit line hang here. Raising :class:`ResultRejected`
            refuses the task with nothing spawned (D3l).
        after_result: Called **after** rung-6 validation passed and the result was
            persisted, with ``{task, outcome, result, capture, ...}``. This is the
            D3g seam: the snapshot/diff classification, the post-run policy
            recomputation and the ``completed`` audit line. It mutates
            ``outcome`` (``write_outcome``, ``files_touched``, ``summary``,
            ``status``) — the model's claim is never the verdict.
    """
    task_id = str(task.get("task_id", ""))
    kind = str(task.get("kind", ""))
    profile = profile_for_kind(kind)
    outcome = TaskOutcome(
        task_id=task_id, kind=kind, runner=runner, status=STATUS_REJECTED,
        profile=profile,
    )

    provider_name, model = parse_model_spec(runner, anchor=anchor)
    provider = get_provider(provider_name)
    if provider is None:
        outcome.status = STATUS_SPAWN_FAILED
        outcome.reason = f"runner {runner!r} has no built-in adapter"
        return outcome

    # --- pre-dispatch refusals: nothing is spawned for these ----------------
    reason = refuse_reason(
        task,
        supports_workspace_reads=provider.supports_workspace_reads,
        runner=runner,
    )
    if reason is not None:
        outcome.reason = reason
        return outcome
    if not provider.supports_profile(profile):
        outcome.reason = (
            f"{provider_name} does not implement the {profile!r} profile; a "
            f"read/analyze task is never downgraded to a write-capable "
            f"invocation, so this fails dispatch"
        )
        return outcome
    try:
        target = resolve_output_target(task, phase_dir)
    except ResultRejected as exc:
        outcome.reason = exc.message
        return outcome
    outcome.output_path = str(target)

    prompt = (task.get("prompt") if prompt is None else prompt) or ""
    if not prompt.strip():
        outcome.reason = f"Task {task_id}: carries no prompt"
        return outcome
    # D3n: the literal token must never reach a runner. A write prompt that still
    # carries it means the scheduler's late substitution did not run, which would
    # silently drop every prior-batch summary — the exact context that keeps batch
    # N from re-applying or contradicting batch N-1. Contract error, not a warning.
    if kind in WRITE_KINDS and _PRIOR_CHANGES_TOKEN in prompt:
        outcome.reason = (
            f"Task {task_id}: the dispatched prompt still contains the literal "
            f"{_PRIOR_CHANGES_TOKEN} token, so the prior-batch context was never "
            f"substituted. A runner would receive the brace text and silently lose "
            f"every prior-batch summary. Nothing was dispatched."
        )
        return outcome

    transport = preferred_prompt_transport(provider_name)
    # D2d/D3: refuse an over-budget prompt HERE rather than letting llm_client's
    # Windows command-line guard fail it late with PROMPT_TOO_LONG. A stdin/file
    # transport is unbounded, so this only ever fires for an argv-only runner.
    over_budget = _prompt_over_budget(provider, model, prompt, transport, profile)
    if over_budget is not None:
        outcome.reason = f"Task {task_id}: {over_budget}"
        return outcome

    prefer_arrays = kind == "salvage"
    effective_timeout = timeout or get_subagent_timeout(provider_name, anchor=anchor)
    budget = retries if retries is not None else get_subagent_retries(anchor=anchor)
    napper = sleep or time.sleep

    attempt_index = 0
    transport_retries_left = budget
    rate_limit_attempts = 0
    # D3j: the recorded duration is wall-clock across *every* attempt, including
    # the backoff sleeps between them — not the sum of the children's own
    # runtimes. The sum is also what made D3i's "bounded by a total per-task
    # wall-clock budget" below bound nothing: a task that slept through four
    # exponential backoffs would report only the seconds its children ran and
    # never reach `effective_timeout`. Monotonic, so a clock step cannot rewind
    # the budget mid-task.
    started_at = time.monotonic()
    # D3i: `effective_timeout` is the *task's* wall-clock budget, not one budget
    # per attempt. Every spawn below is given what is LEFT of this deadline and
    # every retry decision is taken against it, so a task with N retries can
    # never run for (N+1) × the configured ceiling plus its backoff sleeps —
    # which is what would push a dispatch past the host's own ceiling and delay
    # a detached run's failure detection by the same multiple.
    deadline = started_at + effective_timeout

    def elapsed() -> float:
        return time.monotonic() - started_at

    def remaining() -> float:
        return deadline - time.monotonic()

    while True:
        attempt_index += 1
        if before_attempt is not None:
            # The write path's snapshot / policy / audit leg. A refusal here means
            # nothing is spawned for this attempt — the whole point of enforcing
            # the policy pre-dispatch rather than only detecting it afterwards.
            try:
                before_attempt({
                    "task": task,
                    "task_id": task_id,
                    "kind": kind,
                    "profile": profile,
                    "runner": runner,
                    "provider": provider_name,
                    "model": model,
                    "cwd": cwd,
                    "attempt": attempt_index,
                    "prompt": prompt,
                    "transport": transport,
                    "outcome": outcome,
                })
            except ResultRejected as exc:
                outcome.status = exc.status
                outcome.reason = exc.message
                outcome.duration_seconds = elapsed()
                return outcome
        attempt_budget = remaining()
        if attempt_budget <= 0:
            # The first attempt always has the whole budget, so this is the
            # `before_attempt` leg (a large pre-batch snapshot) having consumed
            # what a retry was admitted with. Spawning here would hand a runner
            # a budget it cannot use and record the kill as its own timeout;
            # the budget is the task's, so its exhaustion is the task's outcome.
            status, error = STATUS_TIMEOUT, (
                f"the per-task wall-clock budget of {effective_timeout}s was "
                f"exhausted after {attempt_index - 1} attempt(s); attempt "
                f"{attempt_index} was never started"
            )
            break
        capture = _spawn(
            provider=provider,
            model=model,
            prompt=prompt,
            profile=profile,
            transport=transport,
            timeout=attempt_budget,
            cwd=cwd,
        )
        usage = _safe_parse_usage(provider, capture)
        status, error = _classify_pre_parse(capture)

        if status == STATUS_OK:
            status, error, data = _parse_and_validate(
                provider, capture, kind=kind, prefer_arrays=prefer_arrays
            )
            if status == STATUS_OK:
                try:
                    written = persist_result(task, phase_dir, data)
                except ResultRejected as exc:
                    status, error = exc.status, exc.message
                else:
                    outcome.output_path = written
                    outcome.attempts.append(Attempt(
                        index=attempt_index,
                        status=STATUS_OK,
                        exit_code=capture.exit_code,
                        duration_seconds=capture.duration,
                        usage=usage,
                    ))
                    outcome.status = STATUS_OK
                    outcome.duration_seconds = elapsed()
                    _accumulate_usage(outcome, usage)
                    if after_result is not None:
                        # D3g: the model's claim is never the verdict. The
                        # snapshot/diff classification runs HERE — after rung 6, so
                        # a prose result has already failed as `schema_invalid` and
                        # can never be classified as "applied".
                        after_result({
                            "task": task,
                            "task_id": task_id,
                            "kind": kind,
                            "profile": profile,
                            "runner": runner,
                            "cwd": cwd,
                            "attempt": attempt_index,
                            "prompt": prompt,
                            "result": data,
                            "exit_code": capture.exit_code,
                            "stderr_tail": _tail(capture.stderr, 2000),
                            "usage": usage,
                            "duration_seconds": outcome.duration_seconds,
                            "outcome": outcome,
                        })
                        if not outcome.ok:
                            return outcome
                    if capture.exit_code not in (0, None):
                        # Exit code is evidence, never the verdict: for a read
                        # task a schema-valid result wins over a non-zero exit
                        # (several CLIs exit non-zero on a warning while still
                        # printing the answer), and the anomaly stays visible
                        # here rather than being either fatal or invisible.
                        outcome.reason = (
                            f"runner exited {capture.exit_code} but produced a "
                            f"schema-valid result; stderr tail: "
                            f"{_tail(capture.stderr, 2000)!r}"
                        )
                    return outcome
            elif status in RECLASSIFIABLE_STATUSES:
                # Only a run with no usable result may be reclassified as
                # rate-limited or unauthenticated: the classifier matches on the
                # captured streams, and both a successful answer and a
                # schema-invalid one routinely quote plan text that would trip
                # those patterns. Rung 6 therefore never reaches this branch —
                # it is never retried (D3k), so laundering it into rung 3's
                # backoff or rung 1's batch-wide auth failure would be pure loss.
                reclassified = _classify_failed_run(provider, capture)
                if reclassified is not None:
                    status, error = reclassified

        capture_path = _write_failure_capture(
            phase_dir=phase_dir,
            task=task,
            attempt_index=attempt_index,
            argv=capture.argv,
            prompt=prompt,
            stdout=capture.stdout,
            stderr=capture.stderr,
            exit_code=capture.exit_code,
            duration=capture.duration,
            status=status,
            error=error,
        )
        outcome.attempts.append(Attempt(
            index=attempt_index,
            status=status,
            exit_code=capture.exit_code,
            duration_seconds=capture.duration,
            error=error,
            usage=usage,
            capture_path=capture_path,
        ))
        _accumulate_usage(outcome, usage)

        if status in FATAL_STATUSES:
            break
        if status in BACKOFF_STATUSES:
            # Backoff + jitter, bounded by the same per-task wall-clock budget;
            # does NOT consume the transport retry budget (D3i).
            rate_limit_attempts += 1
            delay = _backoff_delay(rate_limit_attempts)
            if (
                rate_limit_attempts > 3
                or remaining() - delay < MIN_ATTEMPT_BUDGET_S
            ):
                # The sleep itself is inside the budget: waiting past the ceiling
                # only to dispatch one more attempt is spending the budget twice.
                # And a slice the next attempt could not use is not worth waiting
                # for either — the sleep would only delay the same outcome.
                break
            napper(delay)
            continue
        if status in RETRYABLE_STATUSES and transport_retries_left > 0:
            if remaining() < MIN_ATTEMPT_BUDGET_S:
                # The retry budget is not the only budget: a retry runs inside
                # the task's remaining wall clock, never on a fresh copy of it.
                error = (
                    f"{error} (no retry: the per-task wall-clock budget of "
                    f"{effective_timeout}s is exhausted, "
                    f"{transport_retries_left} retr"
                    f"{'y' if transport_retries_left == 1 else 'ies'} unused)"
                )
                break
            transport_retries_left -= 1
            continue
        break

    outcome.status = status
    outcome.reason = error
    outcome.duration_seconds = elapsed()
    return outcome


def _prompt_over_budget(
    provider: Any, model: str, prompt: str, transport: str, profile: str
) -> Optional[str]:
    """Why ``prompt`` cannot travel this runner's argv, or ``None`` (D2d).

    The budget is a Windows command-line constraint, so a ``stdin``/``file``
    transport is unbounded and this is a no-op there. For an argv-only runner it
    measures the *rendered* non-prompt argv too, because that is what Windows
    actually limits — and it resolves ``argv[0]`` exactly as :func:`_spawn` does,
    because an npm ``.cmd``/``.bat`` shim is run by cmd.exe and therefore carries
    the 8,191-unit cap rather than ``CreateProcess``'s 32,767. The same resolution
    answers the second cmd.exe hazard: the shim's command line is *reparsed*, so a
    metacharacter-bearing prompt is refused here rather than handed to that
    reparse (BatBadBut / CVE-2024-24576 class), mirroring ``llm_client``'s guard
    on the fan-out path.
    """
    try:
        from .subagent_tasks import prompt_budget_utf16, utf16_units
    except ImportError:  # pragma: no cover
        from utils.subagent_tasks import prompt_budget_utf16, utf16_units

    if transport in ("stdin", "file"):
        return None
    try:
        argv = provider.build_agent_command(prompt, model, profile)
    except Exception:  # noqa: BLE001 - a budget probe must not mask a refusal
        return None
    try:
        launcher, is_batch_shim = _resolve_executable(argv[0])
    except OSError:  # pragma: no cover - PATH probe failure is not a refusal
        launcher, is_batch_shim = [argv[0]], False
    if is_batch_shim:
        unsafe = sorted(set(prompt) & _CMD_UNSAFE_CHARS)
        if unsafe:
            chars = ", ".join(repr(ch) for ch in unsafe)
            return (
                f"{provider.name} resolves to a cmd.exe batch shim "
                f"({launcher[0]}) and delivers the prompt on argv, but the "
                f"prompt contains cmd.exe metacharacters ({chars}). cmd.exe "
                f"reparses the shim's command line, so these characters could "
                f"corrupt the prompt or escape into executed commands "
                f"(BatBadBut / CVE-2024-24576 class). Install a native "
                f"executable for this runner, or configure a runner with a "
                f"stdin/file prompt transport (driver.subagent_runner). "
                f"Nothing was dispatched."
            )
    # Measure the argv `_spawn` will actually run: the resolved launcher, not the
    # adapter's bare command name.
    rendered = [*launcher, *argv[1:]]
    non_prompt = [arg for arg in rendered if arg != prompt]
    ceiling = prompt_budget_utf16(
        transport=transport,
        non_prompt_argv=non_prompt,
        is_batch_shim=is_batch_shim,
    )
    if ceiling is None:
        return None
    units = utf16_units(prompt)
    if units <= ceiling:
        return None
    launcher_kind = (
        "a .cmd/.bat shim run via cmd.exe (8,191-unit command-line cap)"
        if is_batch_shim
        else "a native executable (32,767-unit CreateProcess cap)"
    )
    return (
        f"its prompt is {units} UTF-16 code units but {provider.name} delivers "
        f"the prompt on argv through {launcher_kind}, where this platform allows "
        f"{ceiling}. Configure a runner with a stdin/file prompt transport "
        f"(driver.subagent_runner). The prompt is never truncated. Nothing was "
        f"dispatched."
    )


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff **plus jitter**, honoring the test fast-backoff switch.

    ``get_backoff_delay`` is only the test hook — it returns its argument
    unchanged outside ``MULTI_LLM_TEST_FAST_BACKOFF`` — so the jitter D3i
    specifies has to be added here. Without it the N sibling tasks a shared
    provider throttled at the same moment re-attempt in lockstep and re-trip the
    same limit together, which is the failure the backoff exists to avoid.

    "Equal jitter": half the exponential step is deterministic and half is
    random, so a retry is never immediate and never longer than the step itself.
    The jitter is applied to the *base* and the test hook is the outer call, which
    keeps ``MULTI_LLM_TEST_FAST_BACKOFF=1`` an exact 10 ms — a jittered test suite
    would be both slower and nondeterministic.
    """
    try:
        from .review_orchestrator_base import get_backoff_delay
    except ImportError:  # pragma: no cover
        from utils.review_orchestrator_base import get_backoff_delay
    step = min(2.0 ** attempt, 30.0)
    half = step / 2.0
    return get_backoff_delay(half + random.random() * half)


def _safe_parse_usage(provider: Any, capture: _Capture) -> Optional[Dict[str, Any]]:
    """``parse_usage`` that can never fail a dispatch (D3j)."""
    try:
        usage = provider.parse_usage(capture.stdout, capture.stderr)
    except Exception:  # noqa: BLE001 - metrics must not break the run
        return None
    if not isinstance(usage, dict) or not usage:
        return None
    return usage


def _accumulate_usage(outcome: TaskOutcome, usage: Optional[Dict[str, Any]]) -> None:
    """Accumulate usage across every attempt of one ``task_id`` (D3j).

    A timed-out or rate-limited attempt still burned tokens, so it counts. Absent
    stays absent: a runner that reports nothing records duration only.
    """
    if not usage:
        return
    outcome.usage_reported = True
    totals = dict(outcome.usage or {})
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[key] = totals.get(key, 0) + value
    outcome.usage = totals


def _classify_pre_parse(capture: _Capture) -> Tuple[str, str]:
    """Ladder rungs 1–2, decidable without parsing anything (D3k).

    ``STATUS_OK`` here means only "the child ran to completion"; rungs 3–6 need
    the parse result, because telling a rate limit from "no text at all" from
    "text with no result object" is exactly what the streams say.
    """
    if capture.launch_error is not None:
        return STATUS_SPAWN_FAILED, (
            f"could not start the runner: {capture.launch_error}. Either the CLI "
            f"is missing/not executable, or the host sandbox denied the spawn."
        )
    if capture.timed_out:
        return STATUS_TIMEOUT, (
            "the runner exceeded its per-task timeout and was killed; no partial "
            "result file exists (executor writes are atomic)"
        )
    return STATUS_OK, ""


def _parse_and_validate(
    provider: Any, capture: _Capture, *, kind: str, prefer_arrays: bool
) -> Tuple[str, str, Any]:
    """Ladder rungs 4–6. Returns ``(status, error, data)``."""
    try:
        data = extract_result(
            provider, capture.stdout, capture.stderr, prefer_arrays=prefer_arrays
        )
    except ResultRejected as exc:
        return exc.status, exc.message, None
    errors = validate_result(kind, data)
    if errors:
        return (
            STATUS_SCHEMA_INVALID,
            f"the extracted result does not match the {kind} result shape: "
            + "; ".join(errors[:5]),
            data,
        )
    return STATUS_OK, "", data


def _classify_failed_run(
    provider: Any, capture: _Capture
) -> Optional[Tuple[str, str]]:
    """Ladder rung 3 (plus rung-1 auth), consulted only for a run with no result.

    Classification lives in the adapter — it is the only layer that knows the
    provider's error shape — and is deliberately **not** consulted for a run that
    produced a schema-valid result: the patterns match on captured streams, and a
    successful answer routinely quotes plan text containing "rate limit" or
    "unauthorized".
    """
    try:
        classified = provider.classify_failure(
            capture.stdout, capture.stderr, capture.exit_code or 0
        )
    except Exception:  # noqa: BLE001 - classification must not raise
        return None
    if classified == STATUS_AUTH_FAILED:
        return STATUS_AUTH_FAILED, (
            f"the runner is installed but not authenticated (it printed a "
            f"login/credential notice). No retry can fix this and every sibling "
            f"task would fail identically, so the whole batch fails. Stream tail: "
            f"{_tail(capture.stderr or capture.stdout, 2000)!r}"
        )
    if classified == STATUS_RATE_LIMITED:
        return STATUS_RATE_LIMITED, (
            f"the provider reported a rate limit / quota / overloaded condition. "
            f"Stream tail: {_tail(capture.stderr or capture.stdout, 2000)!r}"
        )
    return None


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def run_tasks(
    tasks: Sequence[Dict[str, Any]],
    *,
    phase_dir: str,
    phase: str = "",
    kind: str = "",
    task_file: Optional[str] = None,
    anchor: Optional[str] = None,
    cwd: Optional[str] = None,
    host_max: Optional[int] = None,
    resume: bool = True,
    only_task_ids: Optional[Iterable[str]] = None,
    write_report: bool = True,
) -> BatchReport:
    """Dispatch a materialized batch and return its report (D3i).

    Runners are resolved **once per task at batch start** and pinned into the
    report, so a re-dispatch under ``--resume`` or ``--retry-failed-subagents``
    uses the runner the first attempt did rather than re-resolving against a
    config that may have changed mid-run. The effective ceilings are logged once
    at batch start alongside each task's runner, so a slow batch is diagnosable
    from the log rather than inferred.
    """
    selected = list(tasks)
    if only_task_ids is not None:
        wanted = set(only_task_ids)
        selected = [t for t in selected if str(t.get("task_id", "")) in wanted]

    report = BatchReport(
        phase=phase, kind=kind, phase_dir=phase_dir, task_file=task_file,
    )
    if not selected:
        report.ceilings = {"effective": 0}
        if write_report:
            report.write()
        return report

    recorded = _recorded_runners(phase_dir)
    runners = {
        str(task.get("task_id", index)): resolve_task_runner(
            task, anchor=anchor, recorded=recorded
        )
        for index, task in enumerate(selected)
    }

    # Preflight, once per DISTINCT runner: presence, profile, version. A refusal
    # here is exit 3 before any child is dispatched.
    profile = profile_for_kind(str(selected[0].get("kind") or kind))
    for spec in dict.fromkeys(runners.values()):
        info = preflight_runner(spec, profile=profile, anchor=anchor)
        report.runner_versions[spec] = info["version"]

    report.ceilings = effective_ceiling(
        list(runners.values()), anchor=anchor, host_max=host_max
    )
    print(
        f"[exec-subagents] {len(selected)} task(s), effective parallelism "
        f"{report.ceilings['effective']} "
        f"(driver={report.ceilings['driver_max_concurrent_subagents']}, "
        f"host={report.ceilings['host_max_concurrent_subagents']}, "
        f"provider={report.ceilings['provider_max_concurrent']})"
    )
    for task_id, spec in runners.items():
        version = report.runner_versions.get(spec, "unknown")
        print(f"[exec-subagents]   {task_id} -> {spec} (version {version})")

    outcomes = _run_sync(_gather(
        selected,
        runners=runners,
        phase_dir=phase_dir,
        anchor=anchor,
        cwd=cwd,
        max_parallel=report.ceilings["effective"],
        resume=resume,
    ))
    report.outcomes = [outcomes[str(t.get("task_id", i))]
                       for i, t in enumerate(selected)
                       if str(t.get("task_id", i)) in outcomes]

    # A task whose coroutine never returned (cancelled by the overall timeout) is
    # *incomplete*, not silently absent: it has no result file, so the batch must
    # not reaggregate.
    for index, task in enumerate(selected):
        task_id = str(task.get("task_id", index))
        if task_id in outcomes:
            continue
        report.outcomes.append(TaskOutcome(
            task_id=task_id,
            kind=str(task.get("kind", "")),
            runner=runners[task_id],
            status=STATUS_INCOMPLETE,
            reason=(
                "the dispatch did not complete (cancelled by the batch timeout or "
                "the run was killed); no result file exists"
            ),
        ))

    if write_report:
        report.write()
    _mirror_status(phase_dir, report)
    return report


def _run_sync(coro):
    """Run ``coro`` to completion whether or not a loop is already running.

    :func:`run_tasks` is a **synchronous** façade on purpose — it is called from
    an async review orchestrator (`run_review_orchestration`), from a sync
    consolidation handler and from the sync apply base, and threading
    async/await through all three call chains would be a much larger change than
    the executor itself. Inside an existing loop the batch runs on a worker
    thread with its own loop; ``asyncio.run`` from within a running loop is a
    ``RuntimeError``, and the concurrency this schedules is subprocess work in
    threads either way.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class _ProviderBackpressure:
    """D3i's rate-limit backpressure, applied at the provider's in-flight slots.

    A 429 belongs to the *provider*, not to the task that happened to draw it:
    every sibling dispatched into the same limit will hit it too, burn its own
    (budget-free but bounded) rate-limit attempts and then fail. D3i's answer is
    to degrade the batch instead — "the executor narrows that provider's
    in-flight slots for the rest of the batch instead of failing sibling tasks".

    This is a *narrowing* gate, never a second dispatch pool: it is acquired
    inside ``run_tasks_concurrent``'s per-provider and global semaphores, so the
    effective concurrency is the minimum of the two layers and the D3i three-way
    ceiling still bounds the batch from above. Acquisition order is therefore the
    same for every task (provider semaphore → global semaphore → this gate) and a
    holder always already owns the outer two, so narrowing can queue tasks but
    cannot deadlock them.
    """

    #: A throttled provider is halved, never closed: one in-flight slot is the
    #: floor, because zero would strand the rest of the batch behind a provider
    #: whose only way back is a successful attempt.
    MIN_SLOTS = 1

    def __init__(self, limits: Dict[str, int]) -> None:
        self._limits = dict(limits)
        self._in_flight: Dict[str, int] = {}
        self._condition = asyncio.Condition()

    async def acquire(self, provider: str) -> None:
        if provider not in self._limits:
            return
        async with self._condition:
            while self._in_flight.get(provider, 0) >= self._limits[provider]:
                await self._condition.wait()
            self._in_flight[provider] = self._in_flight.get(provider, 0) + 1

    async def release(self, provider: str) -> None:
        if provider not in self._limits:
            return
        async with self._condition:
            self._in_flight[provider] = max(0, self._in_flight.get(provider, 0) - 1)
            self._condition.notify_all()

    async def narrow(self, provider: str) -> Optional[int]:
        """Halve ``provider``'s slots. Returns the new count, or ``None`` at the floor."""
        async with self._condition:
            current = self._limits.get(provider)
            if current is None or current <= self.MIN_SLOTS:
                return None
            self._limits[provider] = max(self.MIN_SLOTS, current // 2)
            # Nothing to wake: narrowing only ever removes capacity.
            return self._limits[provider]


def _saw_rate_limit(outcome: TaskOutcome) -> bool:
    """Did any attempt of this task draw a 429/quota response (D3i)?

    A task that was throttled and then *succeeded* on a later attempt still
    proves the provider is throttling this batch, so it narrows the slots too.
    """
    if outcome.status == STATUS_RATE_LIMITED:
        return True
    return any(attempt.status == STATUS_RATE_LIMITED for attempt in outcome.attempts)


async def _gather(
    tasks: Sequence[Dict[str, Any]],
    *,
    runners: Dict[str, str],
    phase_dir: str,
    anchor: Optional[str],
    cwd: Optional[str],
    max_parallel: int,
    resume: bool,
) -> Dict[str, TaskOutcome]:
    """Run the batch through the task-keyed concurrency helper."""
    try:
        from .review_orchestrator_base import run_tasks_concurrent
    except ImportError:  # pragma: no cover
        from utils.review_orchestrator_base import run_tasks_concurrent

    fatal = {"hit": False}

    # The starting slot count per provider: its own `max_concurrent`, bounded by
    # the batch's effective ceiling (a provider limit above it could never be
    # reached through the global gate anyway). Passed to `run_tasks_concurrent` as
    # the explicit `provider_limits` it takes for exactly this purpose, and used to
    # seed the gate that narrows them from there.
    provider_limits: Dict[str, int] = {}
    per_task_budget = 0.0
    for spec in dict.fromkeys(runners.values()):
        provider_name, _ = parse_model_spec(spec, anchor=anchor)
        declared = get_provider_max_concurrent(provider_name, anchor=anchor)
        provider_limits[provider_name] = max(
            1, min(declared or max_parallel, max_parallel)
        )
        per_task_budget = max(
            per_task_budget, get_subagent_timeout(provider_name, anchor=anchor)
        )
    backpressure = _ProviderBackpressure(provider_limits)

    # The batch cap must be derived from the budget a dispatch actually runs on.
    # Left to itself, `run_tasks_concurrent` computes it from
    # `get_provider_timeout` — the *model-call* timeout, which says nothing about
    # `driver.subagent_timeout_s`. Configure a subagent budget above the
    # provider's model timeout (agentic runners routinely need one) and the
    # batch-level `wait_for` cancels tasks that are still healthily inside their
    # own deadline; they come back with no result file, are recorded
    # `incomplete` by `run_tasks` and fail the phase with exit 3 while every
    # runner was behaving. `dispatch_task` bounds one task by
    # `get_subagent_timeout(provider)` **in total** — retries and rate-limit
    # backoff are spent inside that same deadline, so the retry budget no longer
    # multiplies it — and the gate serializes the batch into
    # ceil(tasks / effective parallelism) waves, so that product is the batch's
    # own ceiling. `run_tasks_concurrent` still doubles it and adds the stagger,
    # which is the headroom for queueing, kill grace and drain.
    lanes = max(1, max_parallel)
    waves = (len(tasks) + lanes - 1) // lanes
    batch_timeout = per_task_budget * waves if per_task_budget > 0 else None

    async def per_task(task: Dict[str, Any], runner: str, index: int) -> TaskOutcome:
        if fatal["hit"]:
            return TaskOutcome(
                task_id=str(task.get("task_id", index)),
                kind=str(task.get("kind", "")),
                runner=runner,
                status=STATUS_INCOMPLETE,
                reason=(
                    "not dispatched: an earlier task failed on a rung that fails "
                    "the whole batch (every sibling would fail identically)"
                ),
            )
        provider_name, _ = parse_model_spec(runner, anchor=anchor)
        await backpressure.acquire(provider_name)
        try:
            outcome = await asyncio.to_thread(
                dispatch_task,
                task,
                phase_dir=phase_dir,
                runner=runner,
                anchor=anchor,
                cwd=cwd,
            )
        finally:
            await backpressure.release(provider_name)
        if outcome.fatal:
            fatal["hit"] = True
        if _saw_rate_limit(outcome):
            narrowed = await backpressure.narrow(provider_name)
            if narrowed is not None:
                print(
                    f"[exec-subagents] {provider_name} reported a rate limit — "
                    f"narrowing it to {narrowed} in-flight slot(s) for the rest "
                    f"of the batch"
                )
        return outcome

    def skip_done(task: Dict[str, Any], runner: str, index: int) -> Optional[TaskOutcome]:
        if not resume or not task_is_complete(task, phase_dir):
            return None
        target = resolve_output_target(task, phase_dir)
        print(f"[exec-subagents] {task.get('task_id')} already complete — skipping")
        return TaskOutcome(
            task_id=str(task.get("task_id", index)),
            kind=str(task.get("kind", "")),
            runner=runner,
            status=STATUS_OK,
            output_path=str(target),
            skipped=True,
        )

    return await run_tasks_concurrent(
        list(tasks),
        per_task,
        runner_for_task=lambda task: runners[str(task.get("task_id", ""))],
        timeout=batch_timeout,
        max_parallel=max_parallel,
        skip_predicate=skip_done,
        provider_limits=provider_limits,
    )


def _recorded_runners(phase_dir: str) -> Dict[str, str]:
    """Runners pinned by a previous batch report, for a re-dispatch (D3o)."""
    path = Path(phase_dir) / BATCH_REPORT_FILENAME
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    recorded = {}
    for row in document.get("tasks", []):
        if isinstance(row, dict) and row.get("task_id") and row.get("runner"):
            recorded[str(row["task_id"])] = str(row["runner"])
    return recorded


def _mirror_status(phase_dir: str, report: BatchReport) -> None:
    """Mirror the batch counts into ``.status.json`` (D4a). Best-effort."""
    try:
        from . import status_file as _status_file
    except ImportError:  # pragma: no cover
        from utils import status_file as _status_file
    try:
        _status_file.update_status(phase_dir, {
            "progress": {
                _STATUS_MIRROR_KEY: {
                    "kind": report.kind,
                    "invoker": report.invoker,
                    "total": len(report.outcomes),
                    "ok": sum(1 for o in report.outcomes if o.ok),
                    "counts": report.counts(),
                    "failed_task_ids": [o.task_id for o in report.failed],
                    "runner_versions": dict(report.runner_versions),
                    "effective_parallelism": report.ceilings.get("effective"),
                }
            }
        })
    except Exception:  # noqa: BLE001 - observability must not fail a phase
        pass


#: The provenance a task file records about itself: the document key naming a
#: source file → the ``inputs`` key its hash was recorded under. This is what
#: makes the default read path able to re-bind a file nobody handed it: every
#: producer writes the source paths next to the hashes (validation and
#: consolidation their ``plan_file``/``grouped_file``, salvage its
#: ``salvage_request_path``). ``diff_sha256``/``items_sha256`` are deliberately
#: absent — their source is not a file on disk, so they stay unchecked rather
#: than spuriously refused (:func:`subagent_tasks.check_task_file_inputs`
#: compares only keys present in both blocks).
_INPUT_SOURCE_KEYS = {
    "plan_file": "plan_sha256",
    "grouped_file": "grouped_sha256",
    "salvage_request_path": "salvage_request_sha256",
}


def _recomputed_inputs(
    document: Dict[str, Any], phase_dir: str, anchor: Optional[str]
) -> Dict[str, str]:
    """Re-hash the inputs this task file's own recorded provenance still names.

    Freshly computed hashes are what turn ``read_task_file``'s inputs guard on —
    without them it version- and schema-checks the file and binds nothing. A
    caller that already knows its inputs passes them in; this is the fallback for
    one that only has the file, and it reads the sources *the document itself*
    names so a re-hash can never bind the file to something it was not
    materialized from.

    A source that cannot be read is skipped rather than hashed to ``""``: a
    missing file would compare as "changed" and refuse a phase dir that was
    merely moved, and an input nobody can recompute is exactly the case
    ``check_task_file_inputs`` documents as "simply does not check it". A
    recorded hash that is *already* ``""`` is skipped for the mirror-image
    reason — that is ``file_content_hash``'s "the producer had no such file"
    (every salvage task file records ``plan_sha256: ""``), so re-hashing
    something in its place would refuse a file that was never bound to it.
    """
    try:
        from .subagent_tasks import file_content_hash
    except ImportError:  # pragma: no cover
        from utils.subagent_tasks import file_content_hash

    recorded = document.get("inputs")
    if not isinstance(recorded, dict):
        return {}
    sources: Dict[str, Any] = {
        key: document.get(key) for key in _INPUT_SOURCE_KEYS
    }
    # The salvage files record no `plan_file` (the request is what their prompt is
    # built from), but they do record `plan_sha256`, and the dispatching caller
    # already threads the plan path through as `anchor` — which is a plan file or
    # a repo directory, so it only stands in when it is really a file.
    if not sources.get("plan_file") and anchor and os.path.isfile(anchor):
        sources["plan_file"] = anchor

    current: Dict[str, str] = {}
    for source_key, input_key in _INPUT_SOURCE_KEYS.items():
        source = sources.get(source_key)
        if not recorded.get(input_key) or not isinstance(source, str) or not source:
            continue
        path = source if os.path.isabs(source) else os.path.join(phase_dir, source)
        if not os.path.isfile(path):
            continue
        current[input_key] = file_content_hash(path)
    return current


def run_task_file(
    task_file: str,
    *,
    phase_dir: Optional[str] = None,
    phase: str = "",
    anchor: Optional[str] = None,
    cwd: Optional[str] = None,
    document: Optional[Dict[str, Any]] = None,
    current_inputs: Optional[Dict[str, str]] = None,
    host_max: Optional[int] = None,
    resume: bool = True,
    only_task_ids: Optional[Iterable[str]] = None,
) -> BatchReport:
    """Dispatch every task in a materialized task file.

    ``document`` lets a caller that already read (and input-bound) the file hand
    it in rather than re-reading — the version/inputs guards belong to
    ``utils/subagent_tasks.read_task_file`` and must not be bypassed here.

    ``current_inputs`` is the freshly recomputed hash block for the read path.
    It is optional but the binding is **not**: ``read_task_file`` only enforces
    the inputs guard when it is given one, so passing nothing would silently
    downgrade the default read to a version-and-schema check — and a phase dir
    outlives the run that wrote it, so a plan or a salvage request edited between
    a crash and the resume would produce a confidently stale prompt against text
    nobody is reviewing any more (D2g). When it is omitted the hashes are
    recomputed from the provenance the document records (:func:`_recomputed_inputs`).

    Human tasks are filtered out rather than dispatched: ``executor_role: human``
    means no invoker ever runs them, and they route to the D6 human-task pause.

    Raises:
        ExecutorError: exit 3 — the file is the wrong version, the wrong shape,
            or was materialized from inputs that have since moved. Raised as the
            executor's own refusal so both invokers' handlers (which exit 3 on
            the fan-out path and skip the one non-fatal salvage dispatch) see it
            rather than a bare traceback. Nothing is dispatched either way.
    """
    resolved_dir = phase_dir or os.path.dirname(os.path.abspath(task_file))
    if document is None:
        try:
            from .subagent_tasks import (
                TaskFileError,
                check_task_file_inputs,
                read_task_file,
            )
        except ImportError:  # pragma: no cover
            from utils.subagent_tasks import (
                TaskFileError,
                check_task_file_inputs,
                read_task_file,
            )
        try:
            document = read_task_file(task_file, phase_dir=resolved_dir)
            bound = (
                current_inputs if current_inputs is not None
                else _recomputed_inputs(document, resolved_dir, anchor)
            )
            if bound:
                check_task_file_inputs(document, task_file, resolved_dir, bound)
        except TaskFileError as exc:
            raise ExecutorError(str(exc)) from exc

    tasks = [
        task for task in tasks_of(document)
        if task.get("executor_role") != "human"
    ]
    return run_tasks(
        tasks,
        phase_dir=resolved_dir,
        phase=phase,
        kind=str(document.get("kind", "")),
        task_file=task_file,
        anchor=anchor,
        cwd=cwd,
        host_max=host_max,
        resume=resume,
        only_task_ids=only_task_ids,
    )


# ---------------------------------------------------------------------------
# The native invoker, simulated (D3a equivalence)
# ---------------------------------------------------------------------------

def simulate_native_invoker(
    tasks: Sequence[Dict[str, Any]],
    *,
    phase_dir: str,
    respond: Callable[[Dict[str, Any]], Any],
    phase: str = "",
    kind: str = "",
    write_report: bool = False,
) -> BatchReport:
    """Play the *native* invoker half of the D3a contract as real code.

    On a native-subagent host the driving agent runs ``task.prompt`` and writes the
    JSON its subagent returned to the task's ``output_path``. That write is
    :func:`persist_result` — the same containment, the same per-kind validation and
    the same serialization the executor uses — so the two invokers produce
    byte-identical files by construction rather than by convention, and the
    equivalence assertion has something real to compare.

    Args:
        respond: ``task -> the JSON the subagent returned`` (a dict/list), or the
            raw text it printed (a string, run through the same extraction the
            executor uses).
    """
    report = BatchReport(
        phase=phase, kind=kind, phase_dir=phase_dir, invoker="native-simulated",
    )
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id", index))
        task_kind = str(task.get("kind", kind))
        outcome = TaskOutcome(
            task_id=task_id, kind=task_kind, runner="native", status=STATUS_REJECTED,
            profile=PROFILE_AGENT_READONLY,
        )
        if task.get("executor_role") == "human":
            outcome.reason = "executor_role: human — never dispatched by any invoker"
            report.outcomes.append(outcome)
            continue
        reason = refuse_reason(task, supports_workspace_reads=True, runner="native")
        if reason is not None:
            outcome.reason = reason
            report.outcomes.append(outcome)
            continue
        data = respond(task)
        if isinstance(data, str):
            extracted = extract_json_from_text(
                data, prefer_arrays=task_kind == "salvage"
            )
            if not extracted.get("success"):
                outcome.status = STATUS_UNPARSEABLE
                outcome.reason = extracted.get("error") or "no JSON in the reply"
                report.outcomes.append(outcome)
                continue
            data = extracted["data"]
        errors = validate_result(task_kind, data)
        if errors:
            outcome.status = STATUS_SCHEMA_INVALID
            outcome.reason = "; ".join(errors[:5])
            report.outcomes.append(outcome)
            continue
        outcome.output_path = persist_result(task, phase_dir, data)
        outcome.status = STATUS_OK
        report.outcomes.append(outcome)
    if write_report:
        report.write()
    return report


# ---------------------------------------------------------------------------
# D3l — --dry-run-subagents for the read half
# ---------------------------------------------------------------------------

def dry_run_report(
    task_file: str,
    *,
    phase_dir: str,
    phase: str = "",
    document: Optional[Dict[str, Any]] = None,
    anchor: Optional[str] = None,
    cwd: Optional[str] = None,
) -> str:
    """Render every read/analyze dispatch without starting one. Returns the path.

    The read-half twin of the scheduler's write dry run: resolve the runner, render
    the exact argv (prompt elided to its digest), check the transport and the
    prompt budget, apply the profile, probe runner presence and version — then write
    ``{phase}/subagent_dry_run.json`` and return, having spawned nothing but
    ``--version``.
    """
    try:
        from .write_safety import DRY_RUN_FILENAME, redact_argv
    except ImportError:  # pragma: no cover
        from utils.write_safety import DRY_RUN_FILENAME, redact_argv

    resolved_dir = phase_dir or os.path.dirname(os.path.abspath(task_file))
    if document is None:
        try:
            from .subagent_tasks import read_task_file
        except ImportError:  # pragma: no cover
            from utils.subagent_tasks import read_task_file
        document = read_task_file(task_file, phase_dir=resolved_dir)

    kind = str(document.get("kind", ""))
    entries: List[Dict[str, Any]] = []
    runners: List[str] = []
    for task in tasks_of(document):
        task_id = str(task.get("task_id", ""))
        entry: Dict[str, Any] = {
            "task_id": task_id,
            "kind": str(task.get("kind", kind)),
            "cwd": cwd,
        }
        if task.get("executor_role") == "human":
            entry["would_dispatch"] = False
            entry["reason"] = "executor_role: human — never dispatched by any invoker"
            entries.append(entry)
            continue
        try:
            entry["profile"] = profile_for_kind(str(task.get("kind", kind)))
        except ExecutorError as exc:
            entry["would_dispatch"] = False
            entry["reason"] = exc.message
            entries.append(entry)
            continue
        runner = resolve_task_runner(task, anchor=anchor)
        entry["runner"] = runner
        runners.append(runner)
        provider_name, model = parse_model_spec(runner, anchor=anchor)
        provider = get_provider(provider_name)
        transport = preferred_prompt_transport(provider_name)
        entry["prompt_transport"] = transport
        prompt = str(task.get("prompt") or "")
        try:
            from .subagent_tasks import utf16_units
        except ImportError:  # pragma: no cover
            from utils.subagent_tasks import utf16_units
        entry["prompt_utf16_units"] = utf16_units(prompt)
        reason = refuse_reason(
            task,
            supports_workspace_reads=bool(
                provider is not None and provider.supports_workspace_reads
            ),
            runner=runner,
        )
        if reason is None:
            try:
                resolve_output_target(task, resolved_dir)
            except ResultRejected as exc:
                reason = exc.message
        if reason is None and provider is not None:
            reason = _prompt_over_budget(
                provider, model, prompt, transport, entry["profile"]
            )
        if reason is None:
            try:
                info = preflight_runner(
                    runner, profile=entry["profile"], anchor=anchor,
                    key="driver.subagent_runner",
                    # D3l: a dry run renders commands and starts no child, so the
                    # declared nested_exec verdict still refuses a `denied` host
                    # but the throwaway dispatch behind an `unknown` leg does not
                    # run — it is a real turn, and this flag promises none.
                    probe_nested=False,
                )
                entry["runner_version"] = info["version"]
                entry["nested_exec"] = info["nested_exec"]
            except ExecutorError as exc:
                reason = exc.message
        if reason is not None:
            entry["would_dispatch"] = False
            entry["reason"] = reason
            entries.append(entry)
            continue
        try:
            argv = provider.build_agent_command(prompt, model, entry["profile"])
        except Exception as exc:  # noqa: BLE001 - a preflight must not raise
            entry["would_dispatch"] = False
            entry["reason"] = str(exc)
            entries.append(entry)
            continue
        entry["argv"] = redact_argv(argv, prompt)
        entry["would_dispatch"] = True
        entries.append(entry)

    path = os.path.join(resolved_dir, DRY_RUN_FILENAME)
    _atomic_write_json(path, {
        "dry_run_version": 1,
        "kind": kind,
        "phase": phase,
        "task_file": task_file,
        "cwd": cwd,
        "ceilings": effective_ceiling(runners, anchor=anchor) if runners
        else {"effective": 0},
        "tasks": entries,
    })
    return path


# ---------------------------------------------------------------------------
# D3b — the self-contained continuation past a marker
# ---------------------------------------------------------------------------

def _retry_instruction(
    args: Any, *, failed_ids: str, retry_command: Optional[str]
) -> List[str]:
    """The failed batch's recovery block, addressed to whoever can follow it.

    ``--host`` is already on this run's own namespace — a shim bakes it into
    every command it renders — so the addressee is resolved at the point host
    capability is *already known*, in Python, and the agent is handed a finished
    sentence. Nothing here asks an agent to consult the capability matrix
    mid-run (D5); the matrix stays off every agent read path.

    Two imports, no third fallback: ``host_capabilities`` sits beside this module
    in the same package, so if neither spelling resolves this module's own
    top-level imports have already failed.
    """
    try:
        from . import host_capabilities as caps
    except ImportError:  # pragma: no cover - direct script runs
        import host_capabilities as caps  # type: ignore
    host_id = caps.resolve_host_id(explicit=getattr(args, "host", None))
    return caps.retry_instruction(
        host_id, failed_ids=failed_ids, retry_command=retry_command
    )


def continue_past_marker(
    args: Any,
    *,
    task_file: str,
    phase_dir: str,
    phase: str = "",
    document: Optional[Dict[str, Any]] = None,
    anchor: Optional[str] = None,
    cwd: Optional[str] = None,
    label: str = "subagent",
    retry_command: Optional[str] = None,
) -> bool:
    """Should this phase continue in-process past the marker it just printed?

    Host-driven (the default, and today's behavior byte-for-byte): returns
    ``False`` immediately when ``--exec-subagents`` is off, so the caller prints
    its follow-up command and exits exactly as before. The executor is strictly
    additive until a host asks for it.

    Self-contained (``--exec-subagents``): dispatches the batch, and returns
    ``True`` when every task produced a schema-valid result file, so the caller
    falls through to reaggregation / verdict ingest / consolidation inline and the
    phase completes in one invocation. Markers are still printed — they are
    informational on this path, and the run is finished when the process exits.

    Raises:
        BatchFailed: exit 3 (even under ``--yes``) when any task failed or is
            incomplete. The phase does **not** reaggregate the short set and never
            falls back to salvage.
    """
    if not getattr(args, "exec_subagents", False):
        return False

    if getattr(args, "dry_run_subagents", False):
        # D3l: the flag means "start no child", and it means it on the read half
        # too. Without this branch a read phase would dispatch for real under a
        # flag whose whole promise is that it does not — the surprise being worse
        # here than on the write path precisely because nothing visibly changes.
        report = dry_run_report(
            task_file, phase_dir=phase_dir, phase=phase, document=document,
            anchor=anchor, cwd=cwd,
        )
        print(
            f"[dry-run-subagents] {report} — every {label} command rendered, no "
            f"child started, nothing else changed. This phase still prints its "
            f"marker and its follow-up command."
        )
        return False

    print(f"\n[exec-subagents] dispatching {label} tasks from {task_file}")
    report = run_task_file(
        task_file,
        phase_dir=phase_dir,
        phase=phase,
        document=document,
        anchor=anchor,
        cwd=cwd,
    )
    if report.ok:
        print(
            f"[exec-subagents] {len(report.outcomes)} {label} task(s) complete; "
            f"continuing in-process"
        )
        return True

    failed_ids = ", ".join(o.task_id for o in report.failed)
    lines = [
        f"\nERROR: {len(report.failed)} of {len(report.outcomes)} {label} task(s) "
        f"did not produce a usable result, so this phase refuses to reaggregate a "
        f"short set. Under --yes this is still exit 3: an unattended run may not "
        f"decide that a missing batch is acceptable.",
        *report.failure_summary(),
        f"Report: {os.path.join(phase_dir, BATCH_REPORT_FILENAME)}",
        f"Diagnostics: {os.path.join(phase_dir, FAILED_CAPTURE_DIRNAME)}/",
        # Resume IS the retry-only-the-failures path: a task counts as done iff
        # its result file exists, parses and validates, so re-invoking the phase
        # skips every completed task and re-dispatches exactly the failed set
        # ({failed_ids}) under its original task_id. Who is being asked to
        # re-invoke is the host's `multi_turn` answer, not an assumption: a host
        # that ends its run with this command leaves no agent able to follow the
        # instruction, so it is addressed to the human instead.
        *_retry_instruction(
            args, failed_ids=failed_ids, retry_command=retry_command
        ),
    ]
    raise BatchFailed("\n".join(lines), report)
