"""The simulated native invoker: the host half of the D3a contract, as test code.

On a native-subagent host the driving agent runs ``task.prompt`` with its own
subagent tool and writes the JSON it got back; no shipped code path plays that
part. These two simulators do, so the equivalence tests can assert "either
invoker, same terminal state" against the production persist and
classification steps:

- :func:`simulate_native_invoker` for a read phase's task file: every write
  goes through :func:`subagent_executor.persist_result`, the executor lane's
  own containment, per-kind validation and serialization.
- :func:`native_dispatcher` for a write phase: injected as
  ``WriteBatchScheduler(dispatcher=...)``, it writes through ``persist_result``
  and is judged by the scheduler's own ``_classify``, the same verdict the
  executor dispatch reaches (docs/development/delegation.md).

The real native lane is not this: the host writes the reply, then
``util check-reply``, ``util write-prompt``, ``util batch-done`` and
``--reaggregate`` take over. Reports from either simulator say
``invoker: native-simulated``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import subagent_executor as _executor  # noqa: E402
from utils.json_extractor import extract_json_from_text  # noqa: E402
from utils.providers.base import PROFILE_AGENT_READONLY  # noqa: E402
from utils.subagent_executor import (  # noqa: E402
    STATUS_OK,
    STATUS_REJECTED,
    STATUS_SALVAGE_UNRECOVERABLE,
    STATUS_SCHEMA_INVALID,
    STATUS_UNPARSEABLE,
    WRITE_SCHEMA_REJECTED,
    BatchReport,
    TaskOutcome,
    _salvage_gave_up,
    persist_result,
    salvage_failure_of,
    the_reply_object,
    validate_result,
)
from utils.subagent_prompts import build_nothing_changed_resend_prompt  # noqa: E402
from utils.subagent_tasks import declared_file_set, refuse_reason  # noqa: E402
from utils.write_safety import relative_to_root  # noqa: E402

#: The ``invoker`` both simulators write into their reports.
INVOKER = "native-simulated"


def simulate_native_invoker(
    tasks: Sequence[Dict[str, Any]],
    *,
    phase_dir: str,
    respond: Callable[[Dict[str, Any]], Any],
    phase: str = "",
    kind: str = "",
    write_report: bool = False,
) -> BatchReport:
    """Play the *native* invoker half of the D3a contract for a read phase.

    The host's write is :func:`persist_result` — the same containment, the
    same per-kind validation and the same serialization the executor uses — so
    the two invokers produce byte-identical files by construction rather than
    by convention, and the equivalence assertion has something real to compare.

    Args:
        respond: ``task -> the JSON the subagent returned`` (a dict/list), or the
            raw text it printed (a string, run through the same extraction the
            executor uses).
    """
    report = BatchReport(
        phase=phase, kind=kind, phase_dir=phase_dir, invoker=INVOKER,
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
            data = the_reply_object(task_kind, extracted["data"], extracted)
        gave_up = _salvage_gave_up(
            task_kind, data, salvage_failure_of(task, phase_dir)
        )
        if gave_up is not None:
            # The executor lane applies the same rule in `_parse_and_validate`,
            # at the same point: after extraction, before validation, and before
            # anything can be written over the model's own result file.
            outcome.status = STATUS_SALVAGE_UNRECOVERABLE
            outcome.reason = gave_up
            report.outcomes.append(outcome)
            continue
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


def native_dispatcher(
    respond: Callable[[Dict[str, Any], str], Any],
) -> Callable[..., TaskOutcome]:
    """A ``WriteBatchScheduler`` dispatcher that plays the native invoker.

    ``respond(task, dispatched_prompt)`` returns the result JSON (or the raw
    text a subagent printed). The write is :func:`persist_result` and the
    verdict is the scheduler's ``_classify``, the same functions the executor
    dispatch uses, so the two invokers produce byte-identical result files and
    identical terminal states by construction. Rule 9 is the executor's too: a
    reply that said ``applied`` while not one byte moved is sent once more.
    """

    def dispatch(
        scheduler: Any,
        task: Dict[str, Any],
        prompt: str,
        snapshot: Any,
        profile: str,
        declared_paths: Sequence[str],
        *,
        batch_index: Optional[int] = None,
    ) -> TaskOutcome:
        task_id = str(task.get("task_id", ""))
        kind = scheduler.kind
        outcome = TaskOutcome(
            task_id=task_id, kind=kind, runner="native", status=STATUS_OK,
            profile=profile,
        )
        started = time.time()
        declared = [
            relative_to_root(p, scheduler.repo_root) for p in declared_file_set(task)
        ]
        allowance = _executor.RESENDS_BY_KIND.get(kind, 0)
        sent = prompt
        attempt = 0
        while True:
            attempt += 1
            scheduler.audit.dispatched(
                task_id=task_id, kind=kind, attempt=attempt, runner="native",
                profile=profile, cwd=scheduler.repo_root,
                sandbox_root=scheduler.repo_root,
                argv=["<native subagent>"], prompt=sent,
                declared_files=declared,
                snapshot_dir=snapshot.directory,
                snapshot_before={e.relative: e.sha256 for e in snapshot.entries},
            )
            scheduler._prompts.keep_dispatched_prompt(task_id, attempt, sent, prompt)
            data = respond(task, sent)
            if isinstance(data, str):
                extracted = extract_json_from_text(data, prefer_arrays=False)
                if not extracted.get("success"):
                    outcome.status = STATUS_SCHEMA_INVALID
                    outcome.write_outcome = WRITE_SCHEMA_REJECTED
                    outcome.reason = extracted.get("error") or "no JSON in the reply"
                    return outcome
                data = the_reply_object(kind, extracted["data"], extracted)
            errors = validate_result(kind, data)
            if errors:
                outcome.status = STATUS_SCHEMA_INVALID
                outcome.write_outcome = WRITE_SCHEMA_REJECTED
                outcome.reason = (
                    f"the reply does not match the {kind} result shape: "
                    + "; ".join(errors[:5])
                )
                return outcome
            outcome.output_path = persist_result(task, scheduler.phase_dir, data)
            asked: Dict[str, Any] = {"result": data, "exit_code": 0}
            if attempt > allowance or not scheduler._applied_without_changes(
                task, snapshot, asked,
            ):
                break
            sent = build_nothing_changed_resend_prompt(
                prompt, declared=declared,
                named=[
                    str(path) for path in (data.get("files_touched") or [])
                    if isinstance(path, str)
                ],
                outside=list(asked.get("moved_outside") or []),
            )
        scheduler._classify(task, outcome, snapshot, data, runner="native",
                            profile=profile, attempt=attempt, exit_code=0,
                            duration=time.time() - started, resent=attempt > 1)
        return outcome

    dispatch.invoker = INVOKER  # type: ignore[attr-defined]
    return dispatch
