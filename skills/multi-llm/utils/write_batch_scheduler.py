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

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import shim mirrors the rest of utils/
    from . import prior_changes as _prior_changes
    from .provider_registry import (
        get_max_concurrent_write_subagents,
        parse_model_spec,
    )
    from .subagent_executor import (
        STATUS_OK,
        STATUS_PARTIAL,
        STATUS_POLICY_VIOLATION,
        STATUS_REJECTED,
        STATUS_SCHEMA_INVALID,
        STATUS_UNPARSEABLE,
        WRITE_APPLIED,
        WRITE_CLEAN_FAILURE,
        WRITE_PARTIAL,
        WRITE_POLICY_VIOLATION,
        WRITE_SCHEMA_REJECTED,
        BatchReport,
        ExecutorError,
        ResultRejected,
        TaskOutcome,
        dispatch_task,
        persist_result,
        preflight_runner,
        profile_for_kind,
        resolve_output_target,
        resolve_task_runner,
        validate_result,
    )
    from .subagent_tasks import (
        PHASE_BY_WRITE_KIND,
        PRIOR_CHANGES_TOKEN,
        SHARED_ARTIFACT_WRITE_KINDS,
        WRITE_KINDS,
        declared_file_set,
        snapshot_file_set,
    )
    from .write_safety import (
        DRY_RUN_FILENAME,
        AuditLog,
        PolicyViolation,
        TaskSnapshot,
        WritePolicy,
        WriteSafetyError,
        admit_disjoint_groups,
        overlapping_pairs,
        relative_to_root,
        resolve_repo_root,
    )
except ImportError:  # pragma: no cover - direct-script import shim
    from utils import prior_changes as _prior_changes
    from utils.provider_registry import (
        get_max_concurrent_write_subagents,
        parse_model_spec,
    )
    from utils.subagent_executor import (
        STATUS_OK,
        STATUS_PARTIAL,
        STATUS_POLICY_VIOLATION,
        STATUS_REJECTED,
        STATUS_SCHEMA_INVALID,
        STATUS_UNPARSEABLE,
        WRITE_APPLIED,
        WRITE_CLEAN_FAILURE,
        WRITE_PARTIAL,
        WRITE_POLICY_VIOLATION,
        WRITE_SCHEMA_REJECTED,
        BatchReport,
        ExecutorError,
        ResultRejected,
        TaskOutcome,
        dispatch_task,
        persist_result,
        preflight_runner,
        profile_for_kind,
        resolve_output_target,
        resolve_task_runner,
        validate_result,
    )
    from utils.subagent_tasks import (
        PHASE_BY_WRITE_KIND,
        PRIOR_CHANGES_TOKEN,
        SHARED_ARTIFACT_WRITE_KINDS,
        WRITE_KINDS,
        declared_file_set,
        snapshot_file_set,
    )
    from utils.write_safety import (
        DRY_RUN_FILENAME,
        AuditLog,
        PolicyViolation,
        TaskSnapshot,
        WritePolicy,
        WriteSafetyError,
        admit_disjoint_groups,
        overlapping_pairs,
        relative_to_root,
        resolve_repo_root,
    )


__all__ = [
    "STRATEGIES",
    "STRATEGY_PAUSE",
    "STRATEGY_SKIP_CONTINUE",
    "STRATEGY_SKIP_DEPENDENTS",
    "WRITE_BATCH_REPORT_FILENAME",
    "SchedulerResult",
    "HumanTaskPlan",
    "WriteBatchScheduler",
    "resolve_human_task_plan",
    "run_write_phase",
]


STRATEGY_PAUSE = "pause-and-ask"
STRATEGY_SKIP_CONTINUE = "skip-continue"
STRATEGY_SKIP_DEPENDENTS = "skip-dependents"
STRATEGIES = (STRATEGY_PAUSE, STRATEGY_SKIP_CONTINUE, STRATEGY_SKIP_DEPENDENTS)

#: The scheduler's own report, beside the executor's per-dispatch one. A separate
#: file because it records the *loop*: batch order, the stopping batch, the skip
#: set, the override records and whether finalization ran.
WRITE_BATCH_REPORT_FILENAME = "write_batch_report.json"

EXIT_OK = 0
EXIT_NEEDS_HUMAN = 3


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

    def is_skipped(self, task_id: str) -> bool:
        return task_id in self.skipped


_DEFERRED_REASON = (
    "deferred — user chose to complete manually after implementation"
)


def resolve_human_task_plan(
    tasks: Sequence[Dict[str, Any]],
    *,
    strategy: Optional[str],
    decisions: Optional[Dict[str, str]] = None,
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

    Pure: takes the task list, returns the decision. Persisting it (through
    ``update_task_status`` / ``record_dependency_override``) is the scheduler's job,
    which is what keeps this testable against the native path's expectations.
    """
    decisions = dict(decisions or {})
    by_id: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        by_id[str(task.get("task_id", ""))] = task
    human_ids = [
        task_id for task_id, task in by_id.items()
        if task.get("executor_role") == "human" or task.get("is_human")
    ]
    plan = HumanTaskPlan(strategy=strategy)
    if not human_ids:
        return plan
    if strategy is None:
        # Nothing resolved: the caller owns the gate. Reporting the human tasks as
        # pending rather than silently skipping them is the fail-closed half.
        plan.pending = sorted(human_ids)
        return plan
    if strategy not in STRATEGIES:
        raise ExecutorError(
            f"Unknown human-task strategy {strategy!r}: expected one of "
            f"{', '.join(STRATEGIES)}. Nothing was dispatched."
        )

    seeds: List[str] = []
    if strategy == STRATEGY_PAUSE:
        for task_id in sorted(human_ids):
            answer = decisions.get(task_id)
            if answer == "done":
                plan.completed.append(task_id)
            elif answer == "skip":
                plan.skipped[task_id] = (
                    "skipped by user during pause-and-ask"
                )
                seeds.append(task_id)
            else:
                plan.pending.append(task_id)
    else:
        for task_id in sorted(human_ids):
            plan.skipped[task_id] = _DEFERRED_REASON
            seeds.append(task_id)

    if strategy == STRATEGY_SKIP_CONTINUE:
        # Edges are treated as satisfied: dependents run, and every one of them
        # gets an override RECORD so "it ran against an unmet precondition" is
        # visible afterwards instead of inferred.
        skipped_set = set(plan.skipped)
        for task_id, task in by_id.items():
            if task_id in skipped_set:
                continue
            overridden = [
                dep for dep in (task.get("depends_on") or [])
                if str(dep) in skipped_set
            ]
            if overridden:
                plan.overrides[task_id] = sorted(str(dep) for dep in overridden)
        return plan

    # pause-and-ask (for the tasks answered `skip`) and skip-dependents both walk
    # the graph forward: BFS over "who depends on me", so a chain
    # T001(human) -> T003 -> T005 is skipped in full.
    dependents: Dict[str, List[str]] = {}
    for task_id, task in by_id.items():
        for dep in task.get("depends_on") or []:
            dependents.setdefault(str(dep), []).append(task_id)
    queue = list(seeds)
    while queue:
        parent = queue.pop(0)
        parent_title = str(by_id.get(parent, {}).get("task_title", "") or parent)
        for child in sorted(dependents.get(parent, [])):
            if child in plan.skipped:
                continue
            plan.skipped[child] = (
                f"skipped — depends on skipped task {parent} ({parent_title})"
            )
            plan.chain[child] = parent
            queue.append(child)
    return plan


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
    human_plan: HumanTaskPlan = field(default_factory=HumanTaskPlan)
    finalized: bool = False
    summary_file: Optional[str] = None
    report_path: Optional[str] = None
    dry_run_path: Optional[str] = None
    messages: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == EXIT_OK

    @property
    def failed(self) -> List[TaskOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": 1,
            "kind": self.kind,
            "phase": self.phase,
            "exit": self.exit_code,
            "batches_dispatched": list(self.batches_dispatched),
            "stopped_at_batch": self.stopped_at_batch,
            "finalized": self.finalized,
            "summary_file": self.summary_file,
            "human_task_strategy": self.human_plan.strategy,
            "skipped_tasks": dict(self.human_plan.skipped),
            "dependency_overrides": {
                k: list(v) for k, v in self.human_plan.overrides.items()
            },
            "pending_human_tasks": list(self.human_plan.pending),
            "tasks": [o.to_dict() for o in self.outcomes],
        }


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
        respond: When given, the **simulated native invoker** runs instead of a
            subprocess: ``(task, dispatched_prompt) -> result JSON``. Same
            containment, same classification, same prior-changes append — which is
            what makes "either invoker, same terminal state" a property of the code.
        dry_run: ``--dry-run-subagents``. Renders every command, applies every
            admission and policy check, writes ``subagent_dry_run.json`` and starts
            **no** child.
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
        self.repo_root = str(repo_root) if repo_root else resolve_repo_root(self.anchor)
        self.state = state
        self.plan_file = plan_file
        self.summary_file = summary_file
        self.summary_file_relative = summary_file_relative
        self.host_max = host_max
        self.forced_runner = runner
        self.resume = resume
        self.dry_run = dry_run
        self.human_task_strategy = human_task_strategy
        self.human_task_decisions = dict(human_task_decisions or {})
        self.respond = respond
        self.finalize = finalize
        self.log = log or (lambda line: print(line))

        self.policy = WritePolicy.load(self.repo_root, anchor=self.anchor)
        self.audit = AuditLog(self.phase_dir)
        self.snapshots: Dict[str, TaskSnapshot] = {}
        self.invoker = "native-simulated" if respond is not None else "executor"

    # -- ceilings -------------------------------------------------------

    def write_ceiling(self) -> int:
        """The effective write parallelism (D3h/D3i), and why it is what it is.

        ``driver.max_concurrent_write_subagents`` (default 1) bounded by the host's
        own ceiling, then **pinned to 1** for a shared-artifact phase regardless of
        either — that is the batchers' own invariant
        (apply-suggestions/apply-task-suggestions edit one file), enforced here
        rather than trusted from them.
        """
        configured = get_max_concurrent_write_subagents(anchor=self.anchor)
        candidates = [configured]
        if self.host_max is not None:
            candidates.append(int(self.host_max))
        ceiling = max(1, min(candidates))
        if self.kind in SHARED_ARTIFACT_WRITE_KINDS:
            return 1
        return ceiling

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

        ``{prior_changes_context}`` is substituted from the log **on disk**, so a
        resumed batch sees the same context an uninterrupted run would have. The
        task JSON is never rewritten: this produces the dispatched prompt only, and
        the audit record stores its hash and length rather than its bytes.

        For the three **apply** kinds a prompt that never contained the token is a
        contract error, not a convenience: all three batchers embed it
        (``utils/suggestion_batcher.py:359``, ``utils/code_fix_batcher.py:307``,
        ``apply_task_suggestions_orchestrator.py:543``), so its absence means the
        producer changed and every later batch would run without prior-batch
        context while looking fine. ``implement`` prompts legitimately carry no
        token — their inter-task context is the dependency section
        ``implement_orchestrator._build_dependency_section`` renders — so the
        substitution is a no-op there rather than a refusal.
        """
        prompt = str(task.get("prompt") or "")
        if PRIOR_CHANGES_TOKEN not in prompt:
            if self.kind == "implement":
                return prompt
            raise ResultRejected(
                f"Task {task.get('task_id')}: its prompt does not contain the "
                f"{PRIOR_CHANGES_TOKEN} token, so there is nowhere to bind the "
                f"prior-batch context. Every apply/implement prompt must carry it "
                f"(utils/suggestion_batcher.py, utils/code_fix_batcher.py, "
                f"apply_task_suggestions_orchestrator.py). Nothing was dispatched.",
                STATUS_REJECTED,
            )
        context = _prior_changes.render(self.output_dir)
        return prompt.replace(PRIOR_CHANGES_TOKEN, context)

    # -- the loop -------------------------------------------------------

    def run(self) -> SchedulerResult:
        """Schedule the phase. Returns the phase's :class:`SchedulerResult`."""
        result = SchedulerResult(kind=self.kind, phase=self.phase)
        ordered = self.batches()

        # 1. Human tasks: resolve once, apply as a graph walk, persist.
        result.human_plan = resolve_human_task_plan(
            self.tasks,
            strategy=self.human_task_strategy,
            decisions=self.human_task_decisions,
        )
        self._persist_human_plan(result.human_plan)

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
            result.report_path = self._write_report(result)
            return result

        # 3. The batches, in order, each verified before the next is dispatched.
        for batch_index, batch_tasks in ordered:
            runnable = [
                task for task in batch_tasks
                if not result.human_plan.is_skipped(str(task.get("task_id", "")))
                and task.get("executor_role") != "human"
            ]
            if not runnable:
                continue
            outcomes = self._run_batch(batch_index, runnable, result)
            result.outcomes.extend(outcomes)
            result.batches_dispatched.append(batch_index)
            self._mirror_status(result)
            if any(o.stops_the_loop for o in outcomes) or any(
                not o.ok for o in outcomes
            ):
                result.stopped_at_batch = batch_index
                result.exit_code = EXIT_NEEDS_HUMAN
                result.messages.extend(self._stop_message(batch_index, outcomes))
                result.report_path = self._write_report(result)
                self._mirror_status(result)
                return result

        # 4. Finalization is part of the contract, not the caller's courtesy.
        if self.finalize:
            self._finalize(result)
        result.report_path = self._write_report(result)
        self._mirror_status(result)
        return result

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
        for group in groups:
            if len(group) == 1:
                outcomes.append(self._dispatch_one(group[0], batch_index))
            else:
                with ThreadPoolExecutor(max_workers=len(group)) as pool:
                    futures = [
                        pool.submit(self._dispatch_one, task, batch_index)
                        for task in group
                    ]
                    outcomes.extend(future.result() for future in futures)
            if any(o.stops_the_loop or not o.ok for o in outcomes):
                # Verification gates the next dispatch (D3h): a partial
                # application is never compounded by a concurrent one.
                break
        return outcomes

    def _dispatch_one(self, task: Dict[str, Any], batch_index: int) -> TaskOutcome:
        task_id = str(task.get("task_id", ""))
        profile = profile_for_kind(self.kind)

        # Resume: a batch counts as done when its result file exists, parses and
        # validates (D3i). Its summary is already in the prior-changes log on
        # disk, so a resumed run replays rather than reconstructs.
        if self.resume and self._already_done(task):
            self.log(f"[write-subagents] {task_id} already complete — skipping")
            outcome = TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_OK, profile=profile, skipped=True,
                write_outcome=WRITE_APPLIED,
            )
            outcome.output_path = str(resolve_output_target(task, self.phase_dir))
            return outcome

        try:
            prompt = self.dispatched_prompt(task)
        except ResultRejected as exc:
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=exc.status, profile=profile, reason=exc.message,
            )

        try:
            snapshot = self._snapshot(task)
        except WriteSafetyError as exc:
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_REJECTED, profile=profile, reason=exc.message,
            )

        # D3l pre-dispatch: a batch whose DECLARED set is not wholly inside the
        # policy never starts a runner.
        declared = [
            relative_to_root(path, self.repo_root)
            for path in declared_file_set(task)
        ]
        try:
            self.policy.check(declared, when="pre-dispatch")
        except PolicyViolation as exc:
            outcome = TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_REJECTED, profile=profile, reason=exc.message,
                write_outcome=WRITE_POLICY_VIOLATION,
                snapshot_dir=snapshot.directory,
            )
            return outcome

        started = time.time()
        if self.respond is not None:
            outcome = self._dispatch_native(task, prompt, snapshot, profile)
        else:
            outcome = self._dispatch_executor(task, prompt, snapshot, profile, declared)
        outcome.snapshot_dir = snapshot.directory
        if not outcome.duration_seconds:
            outcome.duration_seconds = time.time() - started
        if outcome.write_outcome is None and not outcome.ok:
            # The four-outcome axis is total, so a dispatch that never reached the
            # classification still says which bucket it fell in. Rungs 5 and 6 are
            # both "the runner did not return the batch result contract" — prose and
            # a well-formed wrong object are the same failure to a consumer — so
            # they share the schema-rejection outcome. Rungs 1-4 (no child, no
            # auth, timeout, runner error) are transport failures, not write
            # outcomes, and are deliberately left unlabelled.
            if outcome.status in (STATUS_SCHEMA_INVALID, STATUS_UNPARSEABLE):
                outcome.write_outcome = WRITE_SCHEMA_REJECTED

        self._record_batch_effects(task, outcome, batch_index)
        return outcome

    def _dispatch_executor(
        self,
        task: Dict[str, Any],
        prompt: str,
        snapshot: TaskSnapshot,
        profile: str,
        declared: Sequence[str],
    ) -> TaskOutcome:
        """Real subprocess dispatch, with the D3g/D3l hooks bound to this task."""
        runner = self.forced_runner or resolve_task_runner(task, anchor=self.anchor)
        preflight_runner(runner, profile=profile, anchor=self.anchor,
                         key="driver.subagent_runner")

        def before_attempt(context: Dict[str, Any]) -> None:
            self.audit.dispatched(
                task_id=context["task_id"],
                kind=self.kind,
                attempt=context["attempt"],
                runner=runner,
                profile=profile,
                cwd=context["cwd"],
                sandbox_root=self.repo_root,
                argv=self._render_argv(runner, prompt, profile),
                prompt=prompt,
                declared_files=list(declared),
                snapshot_dir=snapshot.directory,
                snapshot_before={
                    entry.relative: entry.sha256 for entry in snapshot.entries
                },
            )

        def after_result(context: Dict[str, Any]) -> None:
            self._classify(task, context["outcome"], snapshot, context["result"],
                           runner=runner, profile=profile,
                           attempt=context["attempt"],
                           exit_code=context["exit_code"],
                           duration=context["duration_seconds"])

        return dispatch_task(
            task,
            phase_dir=self.phase_dir,
            runner=runner,
            anchor=self.anchor,
            cwd=self.repo_root,
            prompt=prompt,
            before_attempt=before_attempt,
            after_result=after_result,
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
        """The rendered argv for the audit record / dry run, never executed here."""
        try:
            from .provider_registry import get_provider
        except ImportError:  # pragma: no cover
            from utils.provider_registry import get_provider
        provider_name, model = parse_model_spec(runner, anchor=self.anchor)
        provider = get_provider(provider_name)
        if provider is None:
            return [runner]
        try:
            return list(provider.build_agent_command(prompt, model, profile))
        except Exception:  # noqa: BLE001 - a log must never fail a dispatch
            return [provider_name, "<profile unsupported>"]

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
    ) -> None:
        """Decide which of the four D3g outcomes this dispatch landed on.

        Precedence, and every branch of it matters:

        * The runner **claimed** something (``status``) and the snapshot **shows**
          something (bytes moved or not). The claim is never the verdict.
        * ``applied``/``no-op`` claimed + empty diff → **clean failure**: recorded,
          the loop continues, and the prior-changes entry takes the documented
          ``(batch failed — …, verified no file changes)`` form.
        * bytes moved + the batch reported ``failed``, **or** only some of the
          batch's ``expected_ids`` came back → **partial application**: stops the
          loop, keeps the snapshots, exits 3 even under ``--yes``.
        * a touched file outside ``driver.write_paths`` → **policy violation**:
          the batch fails, the offending files are restored from the snapshot, the
          loop stops (D3l post-run enforcement).

        A prose result never reaches here at all — it already failed rung 6 as
        ``schema_invalid``, which is the fourth outcome.
        """
        claimed = str((result or {}).get("status") or "")
        summary = str((result or {}).get("summary") or "").strip()
        reported_ids = [str(i) for i in ((result or {}).get("ids") or [])]
        expected_ids = [str(i) for i in (task.get("expected_ids") or [])]
        changed = snapshot.changed()
        observed = snapshot.observed()

        outcome.files_touched = list(changed)
        outcome.summary = summary or None

        # D3l post-run: recompute the touched set and enforce the policy on it.
        try:
            self.policy.check(changed, when="post-run")
        except PolicyViolation as exc:
            report = snapshot.restore(exc.paths)
            outcome.status = STATUS_POLICY_VIOLATION
            outcome.write_outcome = WRITE_POLICY_VIOLATION
            outcome.reason = (
                f"{exc.message} The offending file(s) were restored from the "
                f"pre-batch snapshot ({', '.join(f'{k}: {v}' for k, v in report.items())}) "
                f"and the loop stops here."
            )
            self.audit.restored(task_id=outcome.task_id, report=report, force=False)
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return

        missing_ids = [i for i in expected_ids if reported_ids and i not in reported_ids]

        if changed and (claimed == "failed" or missing_ids):
            detail = (
                f"the batch reported {claimed!r}" if claimed == "failed"
                else f"only {len(reported_ids)} of {len(expected_ids)} expected "
                     f"id(s) came back (missing: {', '.join(missing_ids)})"
            )
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

        if not changed:
            # Success claimed, nothing moved. Recorded as a clean failure and the
            # loop continues: the plan may simply already contain the change.
            outcome.write_outcome = WRITE_CLEAN_FAILURE
            outcome.status = STATUS_OK
            outcome.reason = (
                f"CLEAN FAILURE: the batch reported {claimed or 'no status'!r} but "
                f"the pre-batch snapshot comparison found no byte changes in "
                f"{', '.join(e.relative for e in snapshot.entries) or 'any declared file'}. "
                f"Recorded; the loop continues."
            )
            self._audit_completion(outcome, runner, profile, attempt, exit_code,
                                   observed, duration)
            return

        outcome.write_outcome = WRITE_APPLIED
        outcome.status = STATUS_OK
        self._audit_completion(outcome, runner, profile, attempt, exit_code,
                               observed, duration)

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
        self.audit.completed(
            task_id=outcome.task_id, kind=self.kind, attempt=attempt,
            runner=runner, profile=profile, status=outcome.status,
            write_outcome=outcome.write_outcome, exit_code=exit_code,
            files_touched=outcome.files_touched, observed=observed,
            duration_seconds=duration, reason=outcome.reason,
        )

    # -- per-batch bookkeeping (D3n: append BEFORE the next substitution) ---

    def _record_batch_effects(
        self, task: Dict[str, Any], outcome: TaskOutcome, batch_index: int
    ) -> None:
        """Append prior-changes, record metrics, update task status — in that order.

        The ordering is not cosmetic: the append must land before batch N+1's
        substitution reads the log, or the injected context lags a batch behind.
        """
        task_id = str(task.get("task_id", ""))
        entry_id = str(task.get("prior_changes_id") or task_id)

        if outcome.write_outcome == WRITE_APPLIED:
            summary = outcome.summary or f"batch {batch_index} applied"
        elif outcome.write_outcome == WRITE_CLEAN_FAILURE:
            # The documented clean-failure form, so a later batch reads why the
            # earlier one contributed nothing rather than assuming it succeeded.
            summary = (
                f"(batch failed — {outcome.summary or 'no changes reported'}, "
                f"verified no file changes)"
            )
        else:
            summary = None

        if summary:
            _prior_changes.append(self.output_dir, entry_id, self.phase, summary[:400])

        self._record_metrics(task, outcome, batch_index)

        if self.state is not None:
            status = "completed" if outcome.ok else "failed"
            reason = None if outcome.ok else (outcome.reason or outcome.status)
            self.state.update_task_status(task_id, status, reason=reason)
            self.state.save()

    def _record_metrics(
        self, task: Dict[str, Any], outcome: TaskOutcome, batch_index: int
    ) -> None:
        """``metrics.py record`` with the values the runner actually reported (D3g).

        Duration is the executor's own wall clock. Tokens and tool uses come from
        the adapter's normalized usage where the provider reports them and are
        simply **omitted** where it does not — ``record_metric`` treats all three as
        optional, and a fabricated zero would average into per-phase totals and read
        as a free run.
        """
        if self.state is None or not getattr(self.state, "state_file", None):
            return
        try:
            from . import metrics as _metrics
        except ImportError:  # pragma: no cover
            from utils import metrics as _metrics
        usage = outcome.usage or {}
        label = (
            f"{task.get('task_id')}: "
            f"{task.get('task_title') or self.kind} "
            f"[{outcome.write_outcome or outcome.status}]"
        ).strip()
        try:
            _metrics.record_metric(
                Path(self.state.state_file),
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
                total_batches=len(self.tasks) or None,
                batch_index=batch_index + 1,
                print_eta=False,
            )
        except Exception as exc:  # noqa: BLE001 - metrics must not fail a phase
            self.log(f"[write-subagents] warning: metrics record failed: {exc}")

    # -- snapshots ------------------------------------------------------

    def _snapshot(self, task: Dict[str, Any]) -> TaskSnapshot:
        """Take (or reuse) this task's mandatory pre-batch snapshot (D3h).

        Mandatory, and refused when there is nothing to snapshot: a write task with
        no declared or snapshottable file gives the classification nothing to
        compare, which would make every outcome read as "clean failure".
        """
        task_id = str(task.get("task_id", ""))
        files = snapshot_file_set(task)
        if not files:
            raise WriteSafetyError(
                f"Refusing to dispatch write task {task_id}: it declares no files "
                f"to snapshot, so the post-run byte comparison would have nothing "
                f"to compare and every outcome would read as a clean failure. "
                f"Populate `declared_files`/`snapshot_files` on the materialized "
                f"task (D3h). Nothing was dispatched."
            )
        snapshot = TaskSnapshot.take(
            phase_dir=self.phase_dir, task_id=task_id, files=files,
            root=self.repo_root,
        )
        self.snapshots[task_id] = snapshot
        return snapshot

    def _already_done(self, task: Dict[str, Any]) -> bool:
        try:
            target = resolve_output_target(task, self.phase_dir)
        except ResultRejected:
            return False
        if not target.is_file() or target.stat().st_size == 0:
            return False
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        return not validate_result(self.kind, data)

    # -- human plan persistence -----------------------------------------

    def _persist_human_plan(self, plan: HumanTaskPlan) -> None:
        if self.state is None:
            return
        if plan.strategy and self.state.get("human_task_strategy") != plan.strategy:
            self.state.set("human_task_strategy", plan.strategy)
        for task_id, reason in sorted(plan.skipped.items()):
            self.state.update_task_status(task_id, "skipped", reason=reason)
        for task_id in plan.completed:
            self.state.update_task_status(
                task_id, "completed", reason="answered --human-task <id>=done",
            )
        for task_id, overridden in sorted(plan.overrides.items()):
            self.state.record_dependency_override(task_id, list(overridden))
        self.state.save()

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
        self, batch_index: int, outcomes: Sequence[TaskOutcome]
    ) -> List[str]:
        failed = [o for o in outcomes if not o.ok]
        lines = [
            "",
            f"ERROR: the write loop stopped at batch {batch_index}. No later batch "
            f"was dispatched, the pre-batch snapshots are intact, and the "
            f"{self.phase} phase is NOT marked completed.",
        ]
        for outcome in failed:
            lines.append(
                f"  - {outcome.task_id}: {outcome.write_outcome or outcome.status} "
                f"— {outcome.reason or ''}".rstrip()
            )
            if outcome.snapshot_dir:
                lines.append(
                    f"      restore: multi_llm.py util restore-batch --phase-dir "
                    f'"{self.phase_dir}" --task-id {outcome.task_id}'
                )
        lines += [
            f"Audit log: {self.audit.path}",
            f"Snapshots: {os.path.join(self.phase_dir, 'pre_batch')}/",
            "The half-applied state is deliberately left in place — inspect it, "
            "then either restore the batch or re-invoke with --resume, which "
            "re-dispatches only the unfinished batches and replays the "
            "prior-changes log from disk.",
        ]
        return lines

    def _finalize(self, result: SchedulerResult) -> None:
        """``mark_phase_completed`` + tracking + summary + plan reference (D3n).

        Called only after every batch verified. A run that dispatched every batch
        and skipped this is a **failed** run, which is why the flag lands on the
        result rather than being left implicit.
        """
        if self.state is not None:
            self._metrics_finish()
            self.state.mark_phase_completed(self.phase)
            self.state.save()
            self._finalize_tracking()
        if self.summary_file:
            result.summary_file = self._write_summary(result)
            self._insert_plan_reference(result.summary_file)
        result.finalized = True

    def _metrics_finish(self) -> None:
        try:
            from . import metrics as _metrics
        except ImportError:  # pragma: no cover
            from utils import metrics as _metrics
        try:
            _metrics.finish_phase(Path(self.state.state_file), self.phase)
        except Exception:  # noqa: BLE001 - bookkeeping must not fail a phase
            pass

    def _finalize_tracking(self) -> None:
        """``finalize_tracking.py --state-file`` in process (D3n)."""
        try:
            import finalize_tracking  # type: ignore
        except ImportError:  # pragma: no cover - direct-script import shim
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            try:
                import finalize_tracking  # type: ignore
            except ImportError:
                return
        try:
            finalize_tracking.main(["--state-file", str(self.state.state_file)])
        except SystemExit:  # pragma: no cover - argparse-style exit
            pass
        except Exception as exc:  # noqa: BLE001
            self.log(f"[write-subagents] warning: finalize_tracking failed: {exc}")

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
            "",
            "## Summary",
            "",
            f"{len(result.outcomes)} batch task(s) dispatched across "
            f"{len(result.batches_dispatched)} batch(es) in ascending batch_index "
            f"order, each verified by a pre-batch snapshot comparison before the "
            f"next was dispatched.",
            "",
            "## Task Results",
            "",
            "| Task | Outcome | Files touched | Notes |",
            "|------|---------|---------------|-------|",
        ]
        for outcome in result.outcomes:
            lines.append(
                f"| {outcome.task_id} | {outcome.write_outcome or outcome.status} | "
                f"{', '.join(outcome.files_touched) or '—'} | "
                f"{(outcome.summary or outcome.reason or '').replace('|', '/')[:160]} |"
            )
        touched: List[str] = []
        for outcome in result.outcomes:
            for path in outcome.files_touched:
                if path not in touched:
                    touched.append(path)
        lines += ["", "## Files Modified", ""]
        lines += [f"- `{path}`" for path in touched] or ["- (none)"]

        plan = result.human_plan
        if plan.skipped:
            lines += ["", "## Manual Tasks (Deferred)", ""]
            lines += ["| Task | Reason |", "|------|--------|"]
            for task_id, reason in sorted(plan.skipped.items()):
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
        if result.stopped_at_batch is not None:
            lines += [
                "", "## Cancelled", "",
                f"The write loop stopped at batch {result.stopped_at_batch}; later "
                f"batches were never attempted and remain pending. Snapshots are "
                f"intact under `{os.path.join(self.phase_dir, 'pre_batch')}/`.",
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
        Path(self.phase_dir).mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result.to_dict(), indent=2)
        Path(path).write_text(payload, encoding="utf-8")
        # The executor's own per-dispatch report, so `--status` and the D3k rung
        # accounting look identical on both halves of the executor.
        report = BatchReport(
            phase=self.phase, kind=self.kind, phase_dir=self.phase_dir,
            outcomes=list(result.outcomes), invoker=self.invoker,
            ceilings={"effective": self.write_ceiling(),
                      "driver_max_concurrent_write_subagents":
                          get_max_concurrent_write_subagents(anchor=self.anchor),
                      "host_max_concurrent_subagents": self.host_max,
                      "shared_artifact_phase":
                          self.kind in SHARED_ARTIFACT_WRITE_KINDS},
        )
        report.write()
        return path

    def _mirror_status(self, result: SchedulerResult) -> None:
        """Per-task write progress in ``.status.json`` (D3g/D4a). Best-effort."""
        try:
            from . import status_file as _status_file
        except ImportError:  # pragma: no cover
            from utils import status_file as _status_file
        try:
            _status_file.update_status(self.phase_dir, {
                "progress": {
                    "write_subagents": {
                        "kind": self.kind,
                        "invoker": self.invoker,
                        "total": len(self.tasks),
                        "dispatched": len(result.outcomes),
                        "batches_dispatched": list(result.batches_dispatched),
                        "stopped_at_batch": result.stopped_at_batch,
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
                                "files_touched": list(o.files_touched),
                            }
                            for o in result.outcomes
                        ],
                        "failed_task_ids": [
                            o.task_id for o in result.outcomes if not o.ok
                        ],
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
                    info = preflight_runner(
                        runner, profile=entry["profile"], anchor=self.anchor,
                        key="driver.subagent_runner",
                    )
                    entry["runner_version"] = info["version"]
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
        Path(self.phase_dir).mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "dry_run_version": 1,
            "kind": self.kind,
            "phase": self.phase,
            "repo_root": self.repo_root,
            "ceilings": {
                "effective_write_parallelism": ceiling,
                "driver_max_concurrent_write_subagents":
                    get_max_concurrent_write_subagents(anchor=self.anchor),
                "host_max_concurrent_subagents": self.host_max,
                "shared_artifact_phase": self.kind in SHARED_ARTIFACT_WRITE_KINDS,
            },
            "write_paths": self.policy.to_dict(),
            "human_task_strategy": plan.strategy,
            "tasks": entries,
        }, indent=2), encoding="utf-8")
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
