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
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import shim mirrors the rest of utils/
    from .json_extractor import extract_json_from_text
    from .llm_client import _resolve_executable
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
        ProfileUnsupported,
    )
    from .schema_validator import (
        validate_human_decision_verdicts,
        validate_with_schema_file,
    )
    from .subagent_tasks import (
        KINDS,
        preferred_prompt_transport,
        refuse_reason,
        tasks_of,
    )
except ImportError:  # pragma: no cover - direct-script import shim
    from utils.json_extractor import extract_json_from_text
    from utils.llm_client import _resolve_executable
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
        ProfileUnsupported,
    )
    from utils.schema_validator import (
        validate_human_decision_verdicts,
        validate_with_schema_file,
    )
    from utils.subagent_tasks import (
        KINDS,
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

#: Rungs that fail the whole batch immediately: every sibling would fail
#: identically, so continuing would only burn the retry budget N times.
FATAL_STATUSES = frozenset({STATUS_SPAWN_FAILED, STATUS_AUTH_FAILED})

#: Every read/analyze kind runs read-only. Derived from the kind by the executor
#: (D3e), never read from the task JSON.
PROFILE_BY_KIND = {kind: PROFILE_AGENT_READONLY for kind in KINDS}

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
    "human_decision_judge": re.compile(r"^human_decision_batch_\d+\.json$"),
    "task_generation": re.compile(r"^task_generation_result\.json$"),
    "salvage": re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$"),
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
})

#: Basename prefixes an executor write may never land on (a task file, a salvage
#: request, a rotated log).
RESERVED_OUTPUT_PREFIXES = ("salvage_", "error_", "orchestrator-run.")

#: Statuses mirrored into ``.status.json`` under ``progress`` (display only).
_STATUS_MIRROR_KEY = "subagents"


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

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

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
}


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
) -> Dict[str, Any]:
    """Validate one runner spec and detect its version, once per run (D3c/D3m).

    Checks, earliest wins: the provider has an adapter, the model spec is valid
    for that provider, the adapter declares the requested profile, and (for a
    ``workspace-read`` phase) it declares ``supports_workspace_reads``. Then
    ``version_cmd`` runs once and its result is cached for the process: below
    ``min_version`` is a hard exit-3 refusal before any child is dispatched, above
    ``verified_version`` is a one-line warning (blocking on new releases would rot
    faster than the CLIs do), and unparseable output is recorded ``unknown`` with
    the same warning.

    Returns:
        ``{"provider", "model", "version", "usage_support", "profile"}``.

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
    }
    if check_version:
        info["version"] = _detect_version(provider_name, key=key, spec=spec, profile=profile)
    return info


#: Process-local version cache: detected once per *distinct* runner, never per
#: dispatch (D3m).
_VERSION_CACHE: Dict[str, str] = {}


def reset_version_cache() -> None:
    """Clear the per-process runner-version cache (tests)."""
    _VERSION_CACHE.clear()


def _version_tuple(text: str) -> Tuple[int, ...]:
    parts = []
    for chunk in (text or "").split("."):
        match = re.match(r"^(\d+)", chunk)
        parts.append(int(match.group(1)) if match else 0)
    return tuple(parts) or (0,)


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
        launcher, _ = _resolve_executable(argv[0])
        try:
            completed = subprocess.run(
                [*launcher, *argv[1:]],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                stdin=subprocess.DEVNULL,
            )
            parsed = provider.parse_version(completed.stdout, completed.stderr)
            if parsed:
                version = parsed
        except (OSError, subprocess.SubprocessError):
            version = "unknown"

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
        host_max: The host's ``max_concurrent_subagents`` from
            ``hosts/capabilities.yaml`` (D5b). ``None`` until that data ships —
            the seam is this parameter, not a second code path.
    """
    driver_gate = get_max_concurrent_subagents(anchor=anchor)
    provider_limits: Dict[str, Optional[int]] = {}
    for spec in dict.fromkeys(runners):
        provider_name, _ = parse_model_spec(spec, anchor=anchor)
        provider_limits[provider_name] = get_provider_max_concurrent(provider_name)

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
    if name in RESERVED_OUTPUT_NAMES or any(
        name.startswith(prefix) for prefix in RESERVED_OUTPUT_PREFIXES
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
        declared_resolved = Path(declared)
        if not declared_resolved.is_absolute():
            declared_resolved = resolved_dir / declared_resolved
        if os.path.normpath(str(declared_resolved)) != os.path.normpath(
            str(resolved_dir / name)
        ):
            raise _reject(
                "the task's absolute output_path does not resolve to "
                f"{resolved_dir / name} (names are authoritative, paths are a "
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
    timeout: int,
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
    launcher, _is_batch_shim = _resolve_executable(argv[0])
    argv = [*launcher, *argv[1:]]
    capture.argv = list(argv)

    provider_env = provider.get_env(model)
    remove_env = provider.get_remove_env()
    if provider_env or remove_env:
        env = {**os.environ, **provider_env}
        for key in remove_env:
            env.pop(key, None)
    else:
        env = None

    start = time.time()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            cwd=cwd,
            **(
                {"input": stdin_text}
                if stdin_text is not None
                else {"stdin": subprocess.DEVNULL}
            ),
        )
        capture.duration = time.time() - start
        capture.stdout = completed.stdout or ""
        capture.stderr = completed.stderr or ""
        capture.exit_code = completed.returncode
        return capture
    except subprocess.TimeoutExpired as exc:
        capture.duration = time.time() - start
        capture.timed_out = True
        capture.stdout = _decode(exc.stdout)
        capture.stderr = _decode(exc.stderr)
        return capture
    except OSError as exc:
        capture.duration = time.time() - start
        capture.launch_error = str(exc)
        return capture


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

    prompt = task.get("prompt") or ""
    if not prompt.strip():
        outcome.reason = f"Task {task_id}: carries no prompt"
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
    total_duration = 0.0

    while True:
        attempt_index += 1
        capture = _spawn(
            provider=provider,
            model=model,
            prompt=prompt,
            profile=profile,
            transport=transport,
            timeout=effective_timeout,
            cwd=cwd,
        )
        total_duration += capture.duration
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
                    outcome.duration_seconds = total_duration
                    _accumulate_usage(outcome, usage)
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
            else:
                # Only a run with no usable result may be reclassified as
                # rate-limited or unauthenticated: the classifier matches on the
                # captured streams, and a successful answer routinely quotes plan
                # text that would trip those patterns.
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
            if (
                rate_limit_attempts > 3
                or total_duration >= effective_timeout
            ):
                break
            napper(_backoff_delay(rate_limit_attempts))
            continue
        if status in RETRYABLE_STATUSES and transport_retries_left > 0:
            transport_retries_left -= 1
            continue
        break

    outcome.status = status
    outcome.reason = error
    outcome.duration_seconds = total_duration
    return outcome


def _prompt_over_budget(
    provider: Any, model: str, prompt: str, transport: str, profile: str
) -> Optional[str]:
    """Why ``prompt`` exceeds the runner's transport budget, or ``None`` (D2d).

    The budget is a Windows command-line constraint, so a ``stdin``/``file``
    transport is unbounded and this is a no-op there. For an argv-only runner it
    measures the *rendered* non-prompt argv too, because that is what Windows
    actually limits.
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
    non_prompt = [arg for arg in argv if arg != prompt]
    ceiling = prompt_budget_utf16(transport=transport, non_prompt_argv=non_prompt)
    if ceiling is None:
        return None
    units = utf16_units(prompt)
    if units <= ceiling:
        return None
    return (
        f"its prompt is {units} UTF-16 code units but {provider.name} delivers "
        f"the prompt on argv, where this platform allows {ceiling}. Configure a "
        f"runner with a stdin/file prompt transport (driver.subagent_runner). "
        f"The prompt is never truncated. Nothing was dispatched."
    )


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, honoring the test fast-backoff switch."""
    try:
        from .review_orchestrator_base import get_backoff_delay
    except ImportError:  # pragma: no cover
        from utils.review_orchestrator_base import get_backoff_delay
    return get_backoff_delay(min(2.0 ** attempt, 30.0))


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
        outcome = await asyncio.to_thread(
            dispatch_task,
            task,
            phase_dir=phase_dir,
            runner=runner,
            anchor=anchor,
            cwd=cwd,
        )
        if outcome.fatal:
            fatal["hit"] = True
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
        max_parallel=max_parallel,
        skip_predicate=skip_done,
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


def run_task_file(
    task_file: str,
    *,
    phase_dir: Optional[str] = None,
    phase: str = "",
    anchor: Optional[str] = None,
    cwd: Optional[str] = None,
    document: Optional[Dict[str, Any]] = None,
    host_max: Optional[int] = None,
    resume: bool = True,
    only_task_ids: Optional[Iterable[str]] = None,
) -> BatchReport:
    """Dispatch every task in a materialized task file.

    ``document`` lets a caller that already read (and input-bound) the file hand
    it in rather than re-reading — the version/inputs guards belong to
    ``utils/subagent_tasks.read_task_file`` and must not be bypassed here.

    Human tasks are filtered out rather than dispatched: ``executor_role: human``
    means no invoker ever runs them, and they route to the D6 human-task pause.
    """
    resolved_dir = phase_dir or os.path.dirname(os.path.abspath(task_file))
    if document is None:
        try:
            from .subagent_tasks import read_task_file
        except ImportError:  # pragma: no cover
            from utils.subagent_tasks import read_task_file
        document = read_task_file(task_file, phase_dir=resolved_dir)

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
# D3b — the self-contained continuation past a marker
# ---------------------------------------------------------------------------

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
        # ({failed_ids}) under its original task_id.
        f"Re-invoke to retry only the failures ({failed_ids}) — completed tasks "
        f"are skipped by their result files:",
        f"  {retry_command}" if retry_command else
        "  (re-run the same command; completed tasks are skipped)",
    ]
    raise BatchFailed("\n".join(lines), report)
