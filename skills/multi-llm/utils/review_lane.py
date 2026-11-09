"""What review-plan, review-tasks and review-code share as one review lane.

The review base (``utils/review_orchestrator_base.py``) runs review-plan and
review-tasks; ``code_review_orchestrator.py`` runs review-code. Both fan out to
models, group what came back, resolve a validation round and write a report.
The pieces here are written once for both, so a fix to one lane cannot miss
the other (docs/development/review-results.md, "The shared review lane"):

* :class:`SlotResult`, one model slot's outcome, the shape both fan-outs close
  over;
* :func:`resolve_validation_round`, what a ``--reaggregate`` does with the
  validation round beside its grouping.

``tests/test_review_lane_parity.py`` runs the same offline scenarios on both
lanes and fails on any behaviour that differs between them.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import cli_common
from . import detach as _detach
from . import review_round as _review_round
from . import subagent_tasks as _subagent_tasks
from .llm_client import SALVAGE_FAILURE_KEY
from .state_manager import CURRENT_FORMAT_VERSION
from .subagent_tasks import TaskFileError
from .validation import (
    BoundDocument,
    ValidationRound,
    ensure_prompt_sidecars,
    host_views_lock,
    materialize_validation_round,
    resolved_reaggregate_command,
)


# ---------------------------------------------------------------------------
# One slot's outcome
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotResult:
    """One model slot's outcome, as both review fan-outs read it.

    ``payload`` is the answer of a run that completed: review-plan's output
    text, review-code's issue list. It is ``None`` on every losing path, so a
    crashed provider's parseable stdout is never saved as a completed review
    (``failure_output_from_result`` is the text salvage reads instead).
    ``details`` is the lane's whole record (review-plan's result dict, with the
    served model, sandbox word and usage the checkpoint records); ``failure``
    is the salvage failure kind the slot recorded.

    Each lane's slot still returns its own shape (review-plan a record dict,
    review-code a ``(success, issues, error)`` triple); :func:`slot_results`
    reads either.
    """

    model_spec: str
    success: bool
    payload: Any = None
    error: Optional[str] = None
    source: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)
    failure: Optional[str] = None

    @classmethod
    def from_record(cls, model_spec: str, record: Mapping[str, Any]) -> "SlotResult":
        """review-plan's slot record (``run_single_model``'s return)."""
        return cls(
            model_spec=model_spec,
            success=bool(record.get("success")),
            payload=record.get("output"),
            error=record.get("error"),
            source=record.get("source"),
            details=record,
            failure=record.get(SALVAGE_FAILURE_KEY),
        )

    @classmethod
    def from_review(cls, model_spec: str, value: Any) -> "SlotResult":
        """review-code's slot triple (``run_single_review``'s return)."""
        success, issues, error = value
        return cls(model_spec=model_spec, success=bool(success), payload=issues,
                   error=error)

    @classmethod
    def coerce(cls, model_spec: str, value: Any) -> "SlotResult":
        """*value* as a :class:`SlotResult`, whichever lane's shape it has."""
        if isinstance(value, SlotResult):
            return value
        if isinstance(value, Mapping):
            return cls.from_record(model_spec, value)
        return cls.from_review(model_spec, value)

    def as_review(self) -> tuple:
        """The ``(success, issues, error)`` triple review-code's report reads."""
        return (self.success, self.payload, self.error)


def slot_results(results: Mapping[str, Any]) -> Dict[str, SlotResult]:
    """A fan-out's ``{model_spec: outcome}`` map, each outcome a :class:`SlotResult`."""
    return {spec: SlotResult.coerce(spec, value) for spec, value in results.items()}


# ---------------------------------------------------------------------------
# The validation round of a reaggregate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundSteps:
    """The review base's steps :func:`resolve_validation_round` calls.

    Each lane builds this from the names its own module binds when it calls,
    so a test that patches one of them on that module reaches this function
    too, and this module never imports the review base.
    """

    verify_round_context: Callable[..., None]
    bind_plan_for_validation: Callable[..., BoundDocument]
    reopen_phase_for_validation: Callable[..., None]
    retire_stale_validation_round: Callable[..., Any]
    dispatch_in_round: Callable[..., Any]
    dispatch_subagent_tasks: Callable[..., Any]
    merge_validation_batches: Callable[..., List[Dict[str, Any]]]


@dataclass(frozen=True)
class RoundOutcome:
    """What :func:`resolve_validation_round` left for the report.

    ``groups``: the groups with their verdicts (``[]`` for zero groups), or
    ``None`` when validation was skipped. ``verdicts``: the ``validation.json``
    bytes they came from (``Claim.verdicts``). ``handed_off``: the round is with
    the host, or waits for missing batches; the caller ends here with the
    phase open.
    """

    groups: Optional[List[Dict[str, Any]]] = None
    verdicts: Optional[_review_round.Verdicts] = None
    handed_off: bool = False


async def resolve_validation_round(
    claim: _review_round.Claim,
    args: Any,
    steps: RoundSteps,
    *,
    phase_name: str,
    phase_dir: str,
    plan_path: str,
    grouped: List[Dict[str, Any]],
    validation_path: str,
    current_inputs: Callable[[], Dict[str, Any]],
    build_round: Callable[[BoundDocument], ValidationRound],
    findings: str,
) -> RoundOutcome:
    """Resolve a ``--reaggregate``'s validation round, the same way on every lane.

    * Zero groups: the previous round is retired and ``validation.json`` is
      ``[]``.
    * ``--skip-validation``: nothing is read or written.
    * A ``validation_tasks.json`` on disk: refused (exit 3) when its inputs
      (``current_inputs()``) or snapshots moved; stale prompt sidecars are
      rebound and persisted; a whole round of batch files is merged; otherwise
      the missing batches are dispatched under ``--exec-subagents`` or the
      phase is reopened and left to the host.
    * No task file: a new round is built (``build_round``), stamped and
      published, the phase reopened, then dispatched or handed to the host.

    The merge reads a missing batch as ``validation_failed``, so it is only
    ever handed a whole round. No broad ``except`` surrounds the merge or the
    dispatch: one that raises ends the command with the phase open
    (docs/development/review-results.md, "The shared review lane").

    *findings* names what the review returned ("suggestions", "issues") in the
    zero-group retirement's reason.
    """
    validation_tasks_path = os.path.join(phase_dir, "validation_tasks.json")

    if not grouped:
        # Nothing to validate: a leftover round is bound to the grouping just
        # rewritten, so it retires whole and the empty verdicts take its place.
        steps.retire_stale_validation_round(
            phase_dir, reason=f"review returned no {findings}")
        claim.publish(validation_path, "[]\n", "write validation.json")
        return RoundOutcome(groups=[])

    if args.skip_validation:
        print("\nSkipping validation (--skip-validation flag set)")
        return RoundOutcome()

    if os.path.exists(validation_tasks_path):
        return await _resume_round(
            claim, args, steps, phase_name=phase_name, phase_dir=phase_dir,
            plan_path=plan_path, grouped=grouped, validation_path=validation_path,
            validation_tasks_path=validation_tasks_path, current_inputs=current_inputs,
        )
    return await _materialize_round(
        claim, args, steps, phase_name=phase_name, phase_dir=phase_dir,
        plan_path=plan_path, grouped=grouped, validation_path=validation_path,
        validation_tasks_path=validation_tasks_path, build_round=build_round,
    )


async def _resume_round(
    claim: _review_round.Claim,
    args: Any,
    steps: RoundSteps,
    *,
    phase_name: str,
    phase_dir: str,
    plan_path: str,
    grouped: List[Dict[str, Any]],
    validation_path: str,
    validation_tasks_path: str,
    current_inputs: Callable[[], Dict[str, Any]],
) -> RoundOutcome:
    """The round a task file on disk describes: merged whole, or handed out."""
    print("\nFound validation_tasks.json, checking for completed batches...")
    # A task file whose inputs moved since materialization, or that an older
    # skill version wrote, is refused before anything is dispatched (exit 3,
    # outside any handler, so nothing below can complete the phase).
    try:
        batch_metadata = _subagent_tasks.read_task_file(
            validation_tasks_path, phase_dir=phase_dir, current_inputs=current_inputs(),
        )
    except TaskFileError as task_file_error:
        print(f"ERROR: {task_file_error}", file=sys.stderr)
        sys.exit(task_file_error.exit_code)
    # The snapshots the stored prompts name are re-hashed; a moved phase dir is
    # rebound to its own copy, and a changed or missing one is exit 3.
    steps.verify_round_context(batch_metadata, phase_dir, validation_tasks_path)
    # A stale `prompt_path` is rebound and persisted before any batch is handed
    # out: native hosts reread the JSON, not this in-memory document.
    with host_views_lock(phase_dir):
        if ensure_prompt_sidecars(batch_metadata, phase_dir):
            from .failure_capture import _atomic_write_json

            try:
                with claim.held("rewrite validation_tasks.json"):
                    _atomic_write_json(validation_tasks_path, batch_metadata)
            except OSError as exc:
                print(
                    f"ERROR: Could not persist repaired prompt metadata to "
                    f"{validation_tasks_path}: {exc}. Fix task-file write access "
                    "and retry reaggregation before dispatching native subagents.",
                    file=sys.stderr,
                )
                sys.exit(3)

    total_batches = batch_metadata.get("total_batches", 0)
    batches_found = sum(
        1 for batch in batch_metadata.get("batches", [])
        if os.path.exists(
            os.path.join(phase_dir, f"validation_batch_{batch['batch_index']}.json"))
    )
    if batches_found == total_batches and total_batches > 0:
        print(f"All {total_batches} batch files found, merging...")
        groups = steps.merge_validation_batches(
            phase_dir, batch_metadata, grouped, validation_path, claim=claim,
        )
        return RoundOutcome(groups=groups, verdicts=claim.verdicts)

    # An unfinished round reopens the phase before any batch is handed out: an
    # earlier `--reaggregate --skip-validation` may have recorded it complete.
    claim.check("reopen the phase for validation")
    steps.reopen_phase_for_validation(plan_path, phase_name, phase_dir)
    if await steps.dispatch_in_round(
        claim, steps.dispatch_subagent_tasks,
        args,
        task_file=validation_tasks_path,
        phase_dir=phase_dir,
        phase=phase_name,
        document=batch_metadata,
        plan_path=str(plan_path),
        label="validation",
        retry_command=resolved_reaggregate_command(batch_metadata),
    ):
        # The still-incomplete batches ran in this process (completed ones are
        # skipped by their result files).
        groups = steps.merge_validation_batches(
            phase_dir, batch_metadata, grouped, validation_path, claim=claim,
        )
        return RoundOutcome(groups=groups, verdicts=claim.verdicts)

    _detach.report_progress(phase_dir, step="validation_pending")
    print(f"Only {batches_found}/{total_batches} batch files found.")
    print("Wait for all validation subagents to complete before reaggregating.")
    return RoundOutcome(handed_off=True)


async def _materialize_round(
    claim: _review_round.Claim,
    args: Any,
    steps: RoundSteps,
    *,
    phase_name: str,
    phase_dir: str,
    plan_path: str,
    grouped: List[Dict[str, Any]],
    validation_path: str,
    validation_tasks_path: str,
    build_round: Callable[[BoundDocument], ValidationRound],
) -> RoundOutcome:
    """A new round for *grouped*: published, then dispatched or handed out."""
    print("\nPreparing batched validation for subagent execution...")
    # A new round binds its own snapshot, from one read of the plan as it is now.
    plan_document = steps.bind_plan_for_validation(str(plan_path), phase_dir)
    with host_views_lock(phase_dir):
        batched_tasks = materialize_validation_round(build_round(plan_document))
        batched_tasks["format_version"] = CURRENT_FORMAT_VERSION
        claim.publish(validation_tasks_path, json.dumps(batched_tasks, indent=2),
                      "write validation_tasks.json")

    # The phase owes validation from here (a completion an earlier
    # `--skip-validation` run recorded is dropped), and the executor's mirror of
    # the previous round's batches is cleared: nothing of this one has run yet.
    claim.check("reopen the phase for validation")
    steps.reopen_phase_for_validation(plan_path, phase_name, phase_dir)
    _detach.report_progress(phase_dir, step="validation_pending", subagents=None)

    total_batches = batched_tasks["total_batches"]
    stats = batched_tasks.get("batching_stats", {})
    if total_batches == 1:
        print(f"\n[VALIDATION_PENDING] {validation_tasks_path}")
    else:
        print(f"\n[VALIDATION_BATCHES_PENDING] {validation_tasks_path}")
        print(f"Batches: {total_batches}")
    print("\nValidation prepared for subagent execution.")
    print(f"Groups to validate: {stats.get('total_groups', len(grouped))}")

    # Under `--exec-subagents` this process dispatches the round and merges it;
    # otherwise the host runs it and comes back with `--reaggregate`. The task
    # file stores only the durable mode and plan, so the command is resolved
    # from the running skill dir.
    if await steps.dispatch_in_round(
        claim, steps.dispatch_subagent_tasks,
        args,
        task_file=validation_tasks_path,
        phase_dir=phase_dir,
        phase=phase_name,
        document=batched_tasks,
        plan_path=str(plan_path),
        label="validation",
        retry_command=resolved_reaggregate_command(batched_tasks),
    ):
        groups = steps.merge_validation_batches(
            phase_dir, batched_tasks, grouped, validation_path, claim=claim,
        )
        return RoundOutcome(groups=groups, verdicts=claim.verdicts)

    # The plan as the user typed it, when that names the same file.
    reaggregate_cmd = resolved_reaggregate_command(batched_tasks, typed_plan=True)
    print("\nAfter validation completes, run:")
    print(f"  {reaggregate_cmd or ''}")
    # The one follow-up a native-subagent host is owed, also on the result line.
    cli_common.record_next(reaggregate_cmd or None)
    return RoundOutcome(handed_off=True)
