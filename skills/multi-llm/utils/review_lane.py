"""What review-plan, review-tasks and review-code share as one review lane.

The review base (``utils/review_orchestrator_base.py``) runs review-plan and
review-tasks; ``code_review_orchestrator.py`` runs review-code. Both fan out to
models, group what came back, resolve a validation round and write a report.
The pieces here are written once for both, so a fix to one lane cannot miss
the other (docs/development/review-results.md, "The shared review lane"):

* :class:`ReviewLane`, the hooks a lane supplies where the lanes differ;
* :class:`SlotResult`, one model slot's outcome, the shape both fan-outs close
  over;
* :func:`close_fanout`, what follows a fan-out up to grouping: each slot's
  result saved, the losses named, the sandbox words and salvage recorded, and
  the run either continued or ended at exit 3 when no model answered;
* :func:`bind_grouping`, the grouping a ``--reaggregate`` reports: the bound
  ``grouped.json`` reused, or the results regrouped;
* :func:`resolve_validation_round`, what a ``--reaggregate`` does with the
  validation round beside its grouping: reuse the verdicts bound to it, merge
  or hand out a round on disk, or build a new one.

``tests/test_review_lane_parity.py`` runs the same offline scenarios on both
lanes and fails on any behaviour that differs between them.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from . import cli_common
from . import detach as _detach
from . import review_round as _review_round
from . import scope_policy as _sp
from . import scope_round as _scope_round
from . import status_file as _status_file
from . import subagent_tasks as _subagent_tasks
from . import unattended as _unattended
from .progress_tracker import PhaseProgress
from .llm_client import SALVAGE_FAILURE_KEY
from .state_manager import CURRENT_FORMAT_VERSION, load_groups_payload, save_groups_payload
from .subagent_tasks import TaskFileError
from .validation import (
    BoundDocument,
    ValidationRound,
    apply_validation_to_groups,
    ensure_prompt_sidecars,
    host_views_lock,
    legacy_context_round_reason,
    load_validation_results,
    materialize_validation_round,
    publish_from_task_document,
    resolved_reaggregate_command,
    validation_source_decision,
)


# ---------------------------------------------------------------------------
# The lane's own hooks, and the base's steps it lends the shared stages
# ---------------------------------------------------------------------------

class ReviewLane(Protocol):
    """What one review lane supplies where the lanes legitimately differ.

    The review base's lane (review-plan, review-tasks) and review-code's
    (``code_review_orchestrator.CodeLane``) each build one per run; the shared
    stages here call these hooks and nothing lane-specific besides.

    * ``mode``: the mode word the user's commands name (``review-plan``,
      ``review-code``); ``phase_name`` the phase dir and state key
      (``review-plan``, ``code-review``); ``phase_dir`` and ``plan_path`` this
      run's.
    * ``retry_command(*flags)``: the follow-up command a refusal or a failed
      run names, in the lane's spelling (review-code spells a commit review's
      command in its commit mode).
    * ``save_slot(slot)``: write one slot's ``{model}.json`` (or the salvage
      request of a losing slot); ``False`` when the slot is a failure.
    * ``adopt_salvaged(model_spec, path)``: the repaired ``{model}.json`` an
      in-process salvage wrote, as the slot it makes this model, or ``None``
      to leave the model failed.
    * ``fanout_steps()``: the review base's steps :func:`close_fanout` calls,
      built from the names the lane's own module binds, at call time, so a
      test that patches one there reaches the shared stage too.
    * ``reuse_checks_inputs``: whether a ``--reaggregate`` that reuses bound
      verdicts first holds their task file to the lane's current inputs
      (:func:`resolve_validation_round`). review-code's does: its prompts
      embed the reviewed-code excerpt and a commit review's frozen context
      plan, which no snapshot re-hash covers. The base's does not, and its
      reuse is bound by the round's snapshots alone.
    """

    mode: str
    phase_name: str
    phase_dir: str
    plan_path: str
    reuse_checks_inputs: bool

    def retry_command(self, *flags: str) -> str: ...

    def save_slot(self, slot: "SlotResult") -> bool: ...

    def adopt_salvaged(self, model_spec: str, path: Path) -> Optional["SlotResult"]: ...

    def fanout_steps(self) -> "FanoutSteps": ...


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

    review-code's slot wrapper in ``run_all_reviews`` returns one; the slot
    functions keep their own shapes (``run_single_model`` a record dict,
    ``run_single_review`` a ``(success, issues, error)`` triple), and
    :func:`slot_results` reads any of the three.
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
# Closing a fan-out
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FanoutSteps:
    """The review base's steps :func:`close_fanout` calls, built like
    :class:`RoundSteps` from the calling lane's own module
    (``ReviewLane.fanout_steps``)."""

    backfill_deadline_losses: Callable[..., None]
    record_sandbox_bypassed: Callable[..., Any]
    announce_retired_rungs: Callable[..., Any]
    exec_subagents_requested: Callable[..., bool]
    dispatch_in_round: Callable[..., Any]
    dispatch_salvage_tasks: Callable[..., Any]
    record_models_salvaged: Callable[..., Any]
    sanitize_model_name: Callable[[str], str]
    print_closing_summary: Callable[..., None]


@dataclass(frozen=True)
class FanoutOutcome:
    """What :func:`close_fanout` leaves for grouping.

    ``slots``: every slot that returned, as saved (a slot whose result could
    not be saved is a failure here) and with salvaged models replaced by the
    slot salvage made them. ``failed_models``: ``{model_spec: reason}``, the
    deadline losses included. ``salvaged``: the models this run's in-process
    salvage recovered.
    """

    slots: Dict[str, SlotResult]
    failed_models: Dict[str, str]
    completed_models: List[str]
    salvaged: List[str]


async def close_fanout(
    lane: ReviewLane,
    claim: _review_round.Claim,
    args: Any,
    *,
    model_specs: Sequence[str],
    results: Mapping[str, Any],
    served_models: Dict[str, Dict[str, Any]],
    usage: Dict[str, Dict[str, Any]],
    prior_served: Mapping[str, Any],
    tracker: Any,
    unattended: bool,
) -> FanoutOutcome:
    """Everything between a fan-out's return and grouping, the same on every lane.

    In order: each slot's result is saved (``lane.save_slot``); a model the
    deadline lost gets its failure (``backfill_deadline_losses``, the map
    too); the round is checked, since everything after this belongs to it;
    the slots that ran unsandboxed are mirrored for the poller before any
    ending below, the all-failed exit included; under ``--exec-subagents`` the
    salvage tasks run in this process and each model whose repaired file the
    dispatch names is adopted (``lane.adopt_salvaged``). A run where no model
    answered records ``models_failed`` and ``models_served`` and exits 3
    naming the ``--force`` retry; otherwise the partial results are reported
    and ``models_complete`` is recorded with the served models and usage.

    *served_models* and *usage* are the slots' provenance (``models_served``,
    ``usage``), recorded as given; *prior_served* is what the previous run
    recorded, for the retired-rung notice.
    """
    phase_dir = str(lane.phase_dir)
    steps = lane.fanout_steps()
    slots = slot_results(results)

    failed_models: Dict[str, str] = {}
    for model_spec, slot in list(slots.items()):
        # `payload` is the answer of a run that completed and is `None` on
        # every losing path; the text a losing run left behind stays in
        # `details`, and only salvage may read it (`save_model_result`).
        if not lane.save_slot(slot):
            failed_models[model_spec] = slot.error or "Unknown error"
            # Grouping reads these slots, so one whose result could not be
            # saved is a failure there too.
            slots[model_spec] = replace(
                slot, success=False, payload=None, error=failed_models[model_spec],
            )

    # F11: a model the deadline cancelled has no result and, so far, no failure
    # entry either; one pass writes both, under `DEADLINE_ERROR`.
    steps.backfill_deadline_losses(
        list(model_specs), results, failed_models, tracker=tracker,
    )

    # The model results are saved (a later `--force` reuses them); everything
    # after this belongs to the round, so a run whose round another run
    # replaced while its models ran stops here (utils/review_round.py).
    claim.check("record its models and group their results")

    # The slots that ran without a sandbox, mirrored for the poller before any
    # ending below: every slot failing does not make them sandboxed.
    # `models_served` is the durable copy.
    steps.record_sandbox_bypassed(phase_dir, served_models)
    steps.announce_retired_rungs(prior_served, served_models, anchor=str(lane.plan_path))

    salvaged: List[str] = []
    if failed_models and steps.exec_subagents_requested(args):
        salvaged = await _salvage_in_process(
            lane, claim, args, steps, slots=slots, failed_models=failed_models,
        )

    # D6 row 3: partial results are usable output (exit 0, `state: "partial"`);
    # a run where no model produced a result has nothing to aggregate and is
    # exit 3 naming the retry command, before anything records the phase.
    completed_models = [m for m in model_specs if m not in failed_models]
    if not completed_models:
        # The failure map, so a later `--reaggregate` names each loss and exits
        # 3 too, and `models_served`: a slot that ran without a sandbox did so
        # whether or not it then failed.
        _status_file.update_status(phase_dir, {
            "models_completed": [],
            "models_failed": failed_models,
            **({"models_served": served_models} if served_models else {}),
        })
        print("\nERROR: All models failed")
        for model_spec, reason in failed_models.items():
            print(f"  - {model_spec}: {reason}")
        # The home check's and the jail's lines, repeated: this exit is the
        # run's last word.
        steps.print_closing_summary(phase_dir)
        sys.exit(_unattended.report_partial_results(
            completed=completed_models,
            failed=list(failed_models),
            phase_dir=phase_dir,
            retry_command=lane.retry_command("--force"),
        ))
    _unattended.report_partial_results(
        completed=completed_models,
        failed=list(failed_models),
        phase_dir=phase_dir,
        unattended=unattended,
    )

    _status_file.update_status(phase_dir, {
        "state": "models_complete",
        "models_completed": completed_models,
        "models_failed": failed_models,
        # Added only when something was attributed or reported: no key means
        # nothing was, never that the run was free.
        **({"models_served": served_models} if served_models else {}),
        **({"usage": usage} if usage else {}),
    })
    # Both counters, on the post-salvage truth: `models_done` counts finished
    # slots (every one by now), `models_succeeded` the ones with a result. No
    # `models` key, so the map the fan-out's sweep froze stays as it is.
    _detach.report_progress(
        phase_dir, step="grouping",
        models_total=len(model_specs),
        models_done=len(model_specs),
        models_succeeded=len(model_specs) - len(failed_models),
    )
    return FanoutOutcome(
        slots=slots, failed_models=failed_models,
        completed_models=completed_models, salvaged=salvaged,
    )


async def _salvage_in_process(
    lane: ReviewLane,
    claim: _review_round.Claim,
    args: Any,
    steps: FanoutSteps,
    *,
    slots: Dict[str, SlotResult],
    failed_models: Dict[str, str],
) -> List[str]:
    """D3b: run the salvage tasks here and adopt what they repaired.

    The step is published before the dispatch and a phase ticker narrates it,
    off the loop (``dispatch_in_round``), as the validation window does. Only a
    file this dispatch names as repaired is a recovery: a previous round's
    ``{model}.json`` still beside a failed slot under ``--rerun-all`` is never
    adopted. The per-model map moves with ``failed_models``
    (``record_models_salvaged``), since terminal transitions are idempotent.
    """
    phase_dir = str(lane.phase_dir)
    _detach.report_progress(phase_dir, step="salvage")
    salvage_ticker = PhaseProgress(phase_dir, anchor=str(lane.plan_path))
    salvage_ticker.start_ticker()
    try:
        repaired = await steps.dispatch_in_round(
            claim, steps.dispatch_salvage_tasks,
            args,
            phase_dir=phase_dir,
            phase=lane.phase_name,
            plan_path=str(lane.plan_path),
        )
    finally:
        await salvage_ticker.stop_ticker()
    repaired_paths = {os.path.abspath(path) for path in repaired or ()}
    salvaged: List[str] = []
    for model_spec in list(failed_models):
        recovered = Path(phase_dir) / f"{steps.sanitize_model_name(model_spec)}.json"
        if os.path.abspath(str(recovered)) not in repaired_paths:
            continue
        adopted = lane.adopt_salvaged(model_spec, recovered)
        if adopted is None:
            continue
        print(f"[{model_spec}] recovered by salvage: {recovered}")
        slots[model_spec] = adopted
        del failed_models[model_spec]
        salvaged.append(model_spec)
    if salvaged:
        steps.record_models_salvaged(phase_dir, salvaged)
    return salvaged


# ---------------------------------------------------------------------------
# The validation round of a reaggregate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundSteps:
    """The review base's steps a reaggregate's shared stages call
    (:func:`bind_grouping`, :func:`resolve_validation_round`).

    Each lane builds this from the names its own module binds when it calls,
    so a test that patches one of them on that module reaches this function
    too, and this module never imports the review base.
    """

    group_similar_suggestions: Callable[..., Any]
    stamp_stable_ids: Callable[..., Any]
    grouped_coverage_gap: Callable[..., List[str]]
    verify_round_context: Callable[..., None]
    bind_plan_for_validation: Callable[..., BoundDocument]
    reopen_phase_for_validation: Callable[..., None]
    retire_stale_validation: Callable[..., Any]
    retire_stale_validation_round: Callable[..., Any]
    dispatch_in_round: Callable[..., Any]
    dispatch_subagent_tasks: Callable[..., Any]
    merge_validation_batches: Callable[..., List[Dict[str, Any]]]


@dataclass(frozen=True)
class BoundGrouping:
    """What :func:`bind_grouping` left: the groups, whether they are the bound
    ``grouped.json``'s, and what the lane's *annotate* returned for them."""

    groups: List[Dict[str, Any]]
    reused: bool
    annotation: Any = None


def bind_grouping(
    claim: _review_round.Claim,
    steps: RoundSteps,
    *,
    phase_dir: str,
    grouped_path: str,
    validation_path: str,
    records: List[Dict[str, Any]],
    records_by_model: Mapping[str, List[Dict[str, Any]]],
    skip_validation: bool,
    noun: str,
    annotate: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
) -> BoundGrouping:
    """The grouping a ``--reaggregate`` reports, the same way on every lane.

    A ``grouped.json`` that validation artifacts are bound to is reused, not
    rewritten ("preserves validation alignment"): the task file hashes its
    bytes, so a regroup would make the reaggregate refuse its own round. Bound
    means a ``validation_tasks.json`` beside it (whatever the flags: its
    binding outlives this invocation), or a ``validation.json`` this run will
    use (not under ``--skip-validation``). Zero findings bind nothing.

    Reuse presumes the file still covers the results on disk; a result that
    landed after grouping (a salvage, an out-of-band result) breaks that, so the
    round is retired (renamed) and the results regrouped
    (``grouped_coverage_gap``). A regroup retires the verdicts on disk in the
    same step as the new ``grouped.json``.

    *annotate* is the lane's own pass over the groups (review-code's commit
    attribution): on a reused grouping in memory only, on a regroup before the
    file is written, so the file carries it. *noun* names the findings in the
    printed line.
    """
    validation_tasks_path = os.path.join(phase_dir, "validation_tasks.json")
    has_existing_validation = (
        (os.path.exists(validation_path) and not skip_validation)
        or os.path.exists(validation_tasks_path)
    )
    reuse = bool(records) and has_existing_validation and os.path.exists(grouped_path)
    if reuse:
        with open(grouped_path, 'r', encoding='utf-8') as handle:
            grouped = load_groups_payload(json.load(handle))
        gap = steps.grouped_coverage_gap(grouped, dict(records_by_model))
        if gap:
            print(
                "grouped.json does not cover the model results on disk; "
                "regrouping and retiring the validation round bound to it:"
            )
            for line in gap:
                print(f"  {line}")
            steps.retire_stale_validation_round(
                phase_dir, reason="grouped.json did not cover the results on disk",
            )
            reuse = False

    if reuse:
        print("Using existing grouped.json (preserves validation alignment)")
        return BoundGrouping(
            groups=grouped, reused=True,
            annotation=annotate(grouped) if annotate is not None else None,
        )

    # Plain dicts on both branches (`load_groups_payload` yields them); the
    # stamped ids read `theme` and each record's `type`/`section`/`details`,
    # never the stamped keys, so the conversion moves no group id.
    raw_grouped = steps.group_similar_suggestions(records)
    grouped = [g.to_dict() if hasattr(g, 'to_dict') else g for g in raw_grouped]
    steps.stamp_stable_ids(grouped)
    annotation = annotate(grouped) if annotate is not None else None
    # D7: a re-group orphans any verdicts on disk, so they leave with it, in the
    # same step as the new grouped.json (replaced whole).
    with claim.held("retire the previous validation and regroup"):
        steps.retire_stale_validation(
            phase_dir, reason="grouped.json rewritten by reaggregate",
        )
        _review_round.replace_file(
            grouped_path, json.dumps(save_groups_payload(grouped), indent=2))
    print(f"Grouped {noun} saved to: {grouped_path}")
    return BoundGrouping(groups=grouped, reused=False, annotation=annotation)


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
    grouped_path: str,
    validation_path: str,
    current_inputs: Callable[[], Dict[str, Any]],
    reuse_checks_inputs: bool,
    build_round: Callable[[BoundDocument], ValidationRound],
    scope_round_policy: Any,
    scope_baseline: Optional[Mapping[str, Any]],
    findings: str,
) -> RoundOutcome:
    """Resolve a ``--reaggregate``'s validation round, the same way on every lane.

    * A round built from truncated context (contract 1) is retired whole first.
    * A ``validation.json`` bound to the grouping beside it (and to the batch
      bytes it merged, ``validation_source_decision``) is reused: its
      snapshots are re-verified and its verdicts applied, nothing is merged
      again. One that does not bind is retired (renamed) and the batch files
      decide. On a lane whose reuse checks its inputs
      (*reuse_checks_inputs*, ``ReviewLane.reuse_checks_inputs``), the task
      file is first read against ``current_inputs()`` as a resume reads it
      (exit 3 when one moved), and verdicts no task file binds to every one of
      those inputs are not reused: the round goes on below as if there were
      no ``validation.json``.
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
    zero-group retirement's reason. *scope_round_policy* and *scope_baseline*
    are the round on record's scope policy and baseline, from which reused
    verdicts get their validation-stage decision again.
    """
    validation_tasks_path = os.path.join(phase_dir, "validation_tasks.json")

    # Task 2.10: a round materialized under context contract 1 from an
    # over-limit document judged its head alone. Neither its prompts nor the
    # verdicts reached from them are reused: the whole round is retired
    # (renamed, kept for history) and rematerialized below in the path form.
    legacy_round = (
        None if args.skip_validation else legacy_context_round_reason(phase_dir)
    )
    if legacy_round:
        print(f"\n{legacy_round}")
        steps.retire_stale_validation_round(
            phase_dir, reason="validation built from truncated context (contract 1)",
        )

    # D7: a validation.json is honoured only when it binds to the grouped.json
    # beside it (or, for a legacy file without the binding, when no batch file
    # postdates it); otherwise it is retired (renamed, not deleted) and the
    # batch files decide.
    honour_validation_file = (
        os.path.exists(validation_path) and not args.skip_validation
    )
    use_note = ""
    if honour_validation_file:
        # §7.1: bound to the exact batch bytes it merged (`batch_sha256` +
        # `batch_manifest`), so a rewritten, restored or missing batch retires
        # it by content; only a file without the binding falls back to the
        # legacy mtime test. Both outcomes name what was checked.
        stale_reason, use_note = validation_source_decision(
            Path(validation_path), Path(grouped_path), Path(validation_tasks_path),
        )
        if stale_reason:
            print(f"\nNot using validation.json: {stale_reason}")
            steps.retire_stale_validation(phase_dir, reason=stale_reason)
            honour_validation_file = False

    if grouped and honour_validation_file:
        reuse, reused_round = _round_behind_verdicts(
            phase_dir=phase_dir, validation_tasks_path=validation_tasks_path,
            current_inputs=current_inputs if reuse_checks_inputs else None,
        )
        if reuse:
            return _reuse_verdicts(
                claim, steps, phase_dir=phase_dir, grouped=grouped,
                validation_path=validation_path,
                validation_tasks_path=validation_tasks_path,
                reused_round=reused_round, use_note=use_note,
                scope_round_policy=scope_round_policy,
                scope_baseline=scope_baseline,
            )

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


def _round_behind_verdicts(
    *,
    phase_dir: str,
    validation_tasks_path: str,
    current_inputs: Optional[Callable[[], Dict[str, Any]]],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Whether a bound ``validation.json``'s verdicts may be reused, and the
    task file of the round they came from (``None`` when there is none).

    *current_inputs* is the lane's when its reuse checks them
    (``ReviewLane.reuse_checks_inputs``), else ``None``. The task file is then
    read against them exactly as a resume reads it, so an input that moved is
    refused (exit 3) with the resume's message. Reuse also needs the task file
    to record every one of them: with no task file, or one that records fewer,
    nothing ties the verdicts to those inputs, so they are not reused and the
    round goes on as if no ``validation.json`` were there (merged again, or
    built anew). Without *current_inputs*, the verdicts are bound by their
    round's snapshots alone, and a dir with no task file has none to verify.
    """
    if not os.path.exists(validation_tasks_path):
        if current_inputs is None:
            return True, None
        print(
            "\nNot reusing validation.json: no validation_tasks.json records "
            "the inputs its verdicts were reached on"
        )
        return False, None
    inputs = current_inputs() if current_inputs is not None else None
    try:
        document = _subagent_tasks.read_task_file(
            validation_tasks_path, phase_dir=phase_dir, current_inputs=inputs,
        )
    except TaskFileError as task_file_error:
        print(f"ERROR: {task_file_error}", file=sys.stderr)
        sys.exit(task_file_error.exit_code)
    if inputs is not None:
        recorded = document.get("inputs") or {}
        unrecorded = sorted(key for key in inputs if key not in recorded)
        if unrecorded:
            print(
                f"\nNot reusing validation.json: validation_tasks.json records "
                f"no {', '.join(unrecorded)}"
            )
            return False, None
    return True, document


def _reuse_verdicts(
    claim: _review_round.Claim,
    steps: RoundSteps,
    *,
    phase_dir: str,
    grouped: List[Dict[str, Any]],
    validation_path: str,
    validation_tasks_path: str,
    reused_round: Optional[Dict[str, Any]],
    use_note: str,
    scope_round_policy: Any,
    scope_baseline: Optional[Mapping[str, Any]],
) -> RoundOutcome:
    """The verdicts a bound ``validation.json`` holds, applied to *grouped*.

    *reused_round* is the task file :func:`_round_behind_verdicts` read (its
    inputs already checked when the lane's reuse checks them). The snapshots
    its prompts named are re-hashed (exit 3 on a missing or changed one) and a
    moved phase dir is rebound before the verdicts are reused.
    """
    if reused_round is not None:
        steps.verify_round_context(reused_round, phase_dir, validation_tasks_path)
    print(
        f"\nUsing validation.json"
        f"{f' ({use_note})' if use_note else ''} for report generation..."
    )
    try:
        # Read once: the report built from these verdicts binds these exact
        # bytes (`claim.verdicts`).
        validation_results = load_validation_results(
            Path(validation_path), raw=claim.read_verdicts(validation_path),
        )
        validated_groups = apply_validation_to_groups(
            [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
            validation_results,
        )
        # The reused verdicts' validation-stage decision, re-derived for the
        # report from the round's policy and baseline.
        _scope_round.record_decisions(phase_dir, _scope_round.annotate_decisions(
            validated_groups, scope_round_policy.policy,
            _sp.baseline_id_of(scope_baseline),
        ))
        print(f"Loaded {len(validation_results)} validation results")
        # §2.4: the reused round re-publishes its own recorded context with the
        # reused verdicts, as every merge does.
        publish_from_task_document(phase_dir, reused_round, validation_results)
    except Exception as e:
        # An unreadable file reports the groups without verdicts rather than
        # ending the reaggregate; nothing here dispatches or merges.
        print(f"Warning: Could not load validation.json: {e}")
        return RoundOutcome()
    return RoundOutcome(groups=validated_groups, verdicts=claim.verdicts)


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
