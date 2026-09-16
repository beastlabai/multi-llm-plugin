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

import hashlib
import json
import os
import shutil
import threading
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
        get_provider_max_concurrent,
        parse_model_spec,
    )
    from .subagent_executor import (
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
        TaskOutcome,
        _atomic_write_json,
        _host_ceiling,
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
    )
except ImportError:  # pragma: no cover - direct-script import shim
    from utils import prior_changes as _prior_changes
    from utils.provider_registry import (
        get_max_concurrent_write_subagents,
        get_provider_max_concurrent,
        parse_model_spec,
    )
    from utils.subagent_executor import (
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
        TaskOutcome,
        _atomic_write_json,
        _host_ceiling,
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

    def __bool__(self) -> bool:
        return bool(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def unrecoverable(self) -> List[str]:
        keepable = set(self.restorable)
        return [path for path in self.paths if path not in keepable]


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

        self.policy = WritePolicy.load(self.repo_root, anchor=self.anchor)
        self.audit = AuditLog(self.phase_dir)
        self.snapshots: Dict[str, TaskSnapshot] = {}
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
        self._sweep_warned = False
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

        The render is scoped to ``self.phase`` (N8, 2026-09-16): the log is one
        file shared by every write phase this plan has ever run, and an
        unscoped tail let another phase's older batches crowd out this
        phase's own recent ones.
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
        context = _prior_changes.render(self.output_dir, phase=self.phase)
        return prompt.replace(PRIOR_CHANGES_TOKEN, context)

    def _with_result_contract(self, task: Dict[str, Any], prompt: str) -> str:
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
        """
        expected = [str(i) for i in (task.get("expected_ids") or [])]
        if self.kind == "implement":
            id_lines = (
                f"- `ids`: this task's id, `{expected[0]}`, when you applied it "
                f"or found it already implemented; leave it out on `failed`."
                if expected else
                "- `ids`: the task ids you applied."
            )
        else:
            item = "fix" if self.kind == "apply_code_fixes" else "suggestion"
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
        )

    # -- the loop -------------------------------------------------------

    def run(self) -> SchedulerResult:
        """Schedule the phase. Returns the phase's :class:`SchedulerResult`.

        The rollback store is staging, not an artifact: every path it made
        reversible has its own copy under ``{phase}/pre_batch/{task_id}/files/``
        by the time a dispatch is classified, so it is dropped on every exit —
        success, stop or crash — rather than leaving up to
        ``SWEEP_RETAIN_MAX_BYTES`` of the repository behind in the plan dir.
        """
        try:
            return self._run()
        finally:
            self._report_rollback_store_utilization()
            self._discard_rollback_store()

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
        self.log(
            f"[write-subagents] rollback store: "
            f"{self._retained_bytes / (1024 * 1024):.1f} MB of "
            f"{SWEEP_RETAIN_MAX_BYTES // (1024 * 1024)} MB budget retained "
            f"({len(digests)} files, {len(shards)} shards)"
        )

    def _discard_rollback_store(self) -> None:
        """Drop the sweep's staging bytes and the index that describes them."""
        shutil.rmtree(
            os.path.join(self.phase_dir, ROLLBACK_STORE_DIRNAME), ignore_errors=True
        )
        with self._sweep_lock:
            self._retained.clear()
            self._retained_bytes = 0

    def _run(self) -> SchedulerResult:
        result = SchedulerResult(kind=self.kind, phase=self.phase)
        ordered = self.batches()

        # 1. Human tasks: resolve once, apply as a graph walk, persist.
        result.human_plan = resolve_human_task_plan(
            self.tasks,
            strategy=self.human_task_strategy,
            decisions=self.human_task_decisions,
        )
        if not self.dry_run:
            # A dry run resolves the plan (the report says what *would* be
            # skipped) but never persists it: D3l's contract is "no child
            # started, nothing else changed", and persisting would pin the
            # strategy, mark human tasks skipped and record dependency overrides
            # in `state.json` for a run that never dispatched anything.
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
            # No report: `subagent_dry_run.json` IS the dry run's artifact, and
            # writing the zero-outcome batch reports here would clobber a real
            # run's `write_batch_report.json` / `subagent_batch_report.json`.
            return result

        # 3. The batches, in order, each verified before the next is dispatched.
        # Mirrored once before the first dispatch so `progress.step` reads
        # `applying` for the whole stretch rather than only from the first
        # *finished* task — the in-batch mirror runs after a dispatch returns,
        # and a long first batch is exactly when the step matters (F4).
        self._metrics_start()
        self._mirror_status(result)
        for batch_index, batch_tasks in ordered:
            runnable = [
                task for task in batch_tasks
                if not result.human_plan.is_skipped(str(task.get("task_id", "")))
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
                return result
            result.outcomes.extend(outcomes)
            result.batches_dispatched.append(batch_index)
            self._mirror_status(result)
            if any(o.stops_the_loop for o in outcomes) or any(
                not o.ok for o in outcomes
            ):
                result.stopped_at_batch = batch_index
                result.exit_code = EXIT_NEEDS_HUMAN
                result.messages.extend(self._stop_message(batch_index, outcomes))
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
            outcome = self._dispatch_one(task, batch_index)
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
            # The files the earlier run verified for this batch, so a resumed
            # run's finalization (`--touched`) and summary still name them.
            outcome.files_touched = self._recorded_files_touched(outcome.output_path)
            # One line per skip, before the early return: this branch reaches
            # neither `_audit_terminal` nor `_record_batch_effects`, so without it
            # a resumed run's audit log is silent about every task it declared
            # complete — the one place the 2026-09-16 five-task skip would have
            # been visible as it happened (D3l).
            self._audit_resume_skip(outcome, task)
            return outcome

        try:
            prompt = self._with_result_contract(task, self.dispatched_prompt(task))
        except ResultRejected as exc:
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=exc.status, profile=profile, reason=exc.message,
            )

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

        try:
            snapshot = self._snapshot(task)
        except WriteSafetyError as exc:
            return TaskOutcome(
                task_id=task_id, kind=self.kind, runner=self.invoker,
                status=STATUS_REJECTED, profile=profile, reason=exc.message,
            )

        self._begin_sweep(task_id, declared)
        started = time.time()
        try:
            if self.respond is not None:
                outcome = self._dispatch_native(task, prompt, snapshot, profile)
            else:
                outcome = self._dispatch_executor(
                    task, prompt, snapshot, profile, declared
                )
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
                # unlabelled.
                if outcome.status in (STATUS_SCHEMA_INVALID, STATUS_UNPARSEABLE):
                    outcome.write_outcome = WRITE_SCHEMA_REJECTED

            self._audit_terminal(outcome, snapshot, profile)
        finally:
            # The census and its retained-byte index are consumed by whichever of
            # `_classify` / `_audit_terminal` got there first; this drops them in
            # the one case neither did (a dispatch that raised out).
            with self._sweep_lock:
                self._census_before.pop(task_id, None)
                self._retained_before.pop(task_id, None)
        self._record_batch_effects(task, outcome, batch_index)
        return outcome

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

    # -- D3l post-run containment: the OBSERVED set, not the declared one ---

    def _begin_sweep(self, task_id: str, declared: Sequence[str]) -> None:
        """Take this dispatch's pre-run census, retain its bytes, register its set.

        The census is what makes the post-run policy check non-vacuous; the
        retention pass is what makes its verdict *actionable*. Both happen before
        the runner starts, because neither can be reconstructed afterwards.
        """
        census = self._census()
        dispatched_at = datetime.now().isoformat()
        with self._sweep_lock:
            self._census_before[task_id] = census
            self._retained_before[task_id] = self._retain(census)
            self._declared_by_task[task_id] = set(declared)
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
            self.phase_dir, ROLLBACK_STORE_DIRNAME, digest[:2], digest
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
                f"{os.path.join(self.phase_dir, ROLLBACK_STORE_DIRNAME)}: {named}. "
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
            # F13: the union of every task IN FLIGHT right now — this task's
            # own declared set (still present; popped below) plus any
            # co-scheduled sibling's (the justification this method's
            # docstring names). `self._declared_by_task[task_id]` is popped
            # AFTER this union is taken, so a LATER task's undeclared write to
            # a path this task declared is visible to ITS OWN classification —
            # before this fix the set was unioned for the whole phase and
            # never pruned, so a path declared once stayed invisible for
            # every task dispatched afterward.
            ignored: set = set()
            for declared in self._declared_by_task.values():
                ignored.update(declared)
            self._declared_by_task.pop(task_id, None)
            self._dispatch_started.pop(task_id, None)
        if before is None:
            return UndeclaredWrites()
        after = self._census()
        if after is None:
            return UndeclaredWrites()
        moved = {
            relative for relative, stamp in after.items()
            if relative not in ignored and before.get(relative) != stamp
        }
        moved.update(
            relative for relative in before
            if relative not in after and relative not in ignored
        )
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
    ) -> TaskOutcome:
        """Real subprocess dispatch, with the D3g/D3l hooks bound to this task."""
        runner = self.forced_runner or resolve_task_runner(task, anchor=self.anchor)
        info = preflight_runner(runner, profile=profile, anchor=self.anchor,
                                key="driver.subagent_runner")
        # F12: `subagent_batch_report.json.runner_versions` was empty on this
        # path — the read half fills it from its own batch-start preflight
        # (`_preflight_batch_runners`); this call already runs one per
        # dispatch and its version was simply never kept.
        self._runner_versions[runner] = str(info.get("version") or "unknown")

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
        * ``applied`` claimed + empty diff → **clean failure**: recorded, the
          loop continues, and the prior-changes entry takes the documented
          ``(batch failed — …, verified no file changes)`` form.
        * ``no-op`` claimed + empty diff → **no-op**: the claim and the verdict
          agree, so it is recorded under its own name (``(no-op — …, verified
          no file changes)``) and the loop continues. Labelling it a clean
          failure told the next batch the work had failed.
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
        """
        claimed = str((result or {}).get("status") or "")
        summary = str((result or {}).get("summary") or "").strip()
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
            outcome.reason = (
                f"{exc.message} The offending file(s) were restored from the "
                f"pre-batch snapshot ({', '.join(f'{k}: {v}' for k, v in report.items())}) "
                f"and the loop stops here."
                + (
                    # A path the sweep could not retain bytes for (a policy-denied
                    # one above all: its bytes are deliberately never copied into
                    # an executor artifact). Saying "restored" and leaving the
                    # contradiction in the parenthetical is how a violation gets
                    # read as handled.
                    f" No rollback point existed for {len(stranded)} of them "
                    f"({', '.join(stranded[:5])}"
                    f"{', …' if len(stranded) > 5 else ''}); those files remain on "
                    f"disk exactly as the runner left them — inspect them by hand."
                    if stranded else ""
                )
            )
            self.audit.restored(task_id=outcome.task_id, report=report, force=False)
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
                details.append(
                    f"only {len(reported_ids)} of {len(expected_ids)} expected "
                    f"id(s) came back (missing: {', '.join(missing_ids)})"
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

        with self._effects_lock:
            if summary:
                _prior_changes.append(
                    self.output_dir, entry_id, self.phase, summary[:400]
                )

            self._record_metrics(task, outcome, batch_index)

            if self.state is not None:
                status = "completed" if outcome.ok else "failed"
                reason = None if outcome.ok else (outcome.reason or outcome.status)
                self.state.update_task_status(task_id, status, reason=reason)
                # Persists the metric recorded just above too: it is a mutation
                # of this object's dict now, not a second writer's file (F4).
                self.state.save()

            # One line per batch, on the happy path (F7). Every other `self.log`
            # in the loop is exceptional, so a 25-minute five-batch run left an
            # 11-line log — and `status` points the user at that log.
            self.log(
                f"[write-subagents] batch {batch_index}: {task_id} "
                f"{outcome.write_outcome or outcome.status} in "
                f"{outcome.duration_seconds:.1f}s — "
                f"{', '.join(outcome.files_touched) or 'no files changed'}"
            )

            if outcome.output_path:
                # A result file only counts as this task's on resume when it was
                # produced from these inputs (D2g/D3i). The binding is written
                # here, under the same lock, because the file is one shared
                # read-modify-write map for the whole phase.
                self._record_fingerprint(task)

    def _record_metrics(
        self, task: Dict[str, Any], outcome: TaskOutcome, batch_index: int
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
        """
        if self.state is None or not hasattr(self.state, "record_metric"):
            return
        usage = outcome.usage or {}
        label = (
            f"{task.get('task_id')}: "
            f"{task.get('task_title') or self.kind} "
            f"[{outcome.write_outcome or outcome.status}]"
        ).strip()
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
                total_batches=len(self.tasks) or None,
                batch_index=batch_index + 1,
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
        if not files:
            raise WriteSafetyError(
                f"Refusing to dispatch write task {task_id}: it declares no files "
                f"to snapshot, so the post-run byte comparison would have nothing "
                f"to compare and every outcome would read as a clean failure. "
                f"Populate `declared_files`/`snapshot_files` on the materialized "
                f"task (D3h). Nothing was dispatched."
            )
        reused = self._reusable_snapshot(task_id, files) if self.resume else None
        if reused is not None:
            self.log(
                f"[write-subagents] {task_id}: reusing the pre-batch snapshot a "
                f"previous run took at {reused.directory} — the rollback point "
                f"stays the bytes from before this batch's FIRST dispatch"
            )
            self.snapshots[task_id] = reused
            return reused
        snapshot = TaskSnapshot.take(
            phase_dir=self.phase_dir, task_id=task_id, files=files,
            root=self.repo_root,
        )
        self.snapshots[task_id] = snapshot
        return snapshot

    def _reusable_snapshot(
        self, task_id: str, files: Sequence[str]
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
        """
        existing = TaskSnapshot.load(self.phase_dir, task_id)
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

    def _already_done(self, task: Dict[str, Any]) -> bool:
        """Resume's definition of done (D3i), all five clauses of it.

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
        if validate_result(self.kind, data):
            return False
        if str(data.get("status") or "") not in (WRITE_APPLIED, WRITE_NO_OP):
            # A schema-valid `{"status": "failed"}` is a batch the previous run
            # recorded as NOT done; resuming past it on the strength of the
            # file existing would skip exactly the batch --resume is for.
            return False
        task_id = str(task.get("task_id", ""))
        expected = [str(item) for item in (task.get("expected_ids") or [])]
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
        recorded = self._load_fingerprints().get(task_id)
        if recorded and recorded != self._task_fingerprint(task):
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
                f"{os.path.join(self.phase_dir, 'pre_batch')}/, and re-invoking "
                f"with --resume re-dispatches only the unfinished batches.",
                f"Audit log: {self.audit.path}",
            ]
        else:
            lines.append(
                "Nothing was dispatched and no file was modified."
            )
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
            # F5: re-stamp `plan_hash` to what this run actually left on disk.
            # Every batch's edit is already written by the time `_finalize`
            # runs, so without this the *next* `StateManager` built on this
            # plan reads the run's own edit as a human editing the plan
            # mid-workflow and sets `plan_changed`, which
            # `implement_orchestrator.py` reports as "Plan has changed since
            # last session" for a change the plugin made.
            self.state.refresh_plan_hash()
            self.state.save()
            self._finalize_tracking(result)
            # `finalize_tracking` is a file-level writer: it read the state we
            # just saved, computed `tracked_files` and wrote it back. Adopt its
            # file before touching the dict again, or the `save()` below dumps
            # the pre-finalize dict over it and the phase's only deliverable is
            # not tracked at all (F3, measured 2026-09-16).
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

    def _metrics_start(self) -> None:
        """Stamp the phase's ETA baseline before the first dispatch.

        Only the `util metrics start` CLI leg ever called `start_phase`, so on
        the executor path `metrics_progress` never existed and neither did the
        `[ETA]` line `instructions/implement.md` step 5 describes (F4).
        """
        if self.state is None or not hasattr(self.state, "start_metrics_phase"):
            return
        try:
            self.state.start_metrics_phase(self.phase, len(self.tasks))
            self.state.save()
        except Exception:  # noqa: BLE001 - bookkeeping must not fail a phase
            pass

    def _metrics_finish(self) -> None:
        if not hasattr(self.state, "finish_metrics_phase"):
            return
        try:
            self.state.finish_metrics_phase(self.phase)
        except Exception:  # noqa: BLE001 - bookkeeping must not fail a phase
            pass

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
        touched: List[str] = []
        for outcome in (result.outcomes if result is not None else []):
            if not outcome.ok:
                continue
            for path in outcome.files_touched:
                if path not in touched:
                    touched.append(path)
        try:
            import finalize_tracking  # type: ignore
        except ImportError:  # pragma: no cover - direct-script import shim
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            try:
                import finalize_tracking  # type: ignore
            except ImportError:
                return
        argv = ["--state-file", str(self.state.state_file)]
        for path in touched:
            argv += ["--touched", path]
        try:
            finalize_tracking.main(argv)
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
        # D3l: a scheduler artifact is written under the same rules as any other
        # executor write — same-directory temp, 0600, ``os.replace`` — so a run
        # killed mid-write leaves either the old report or the new one, never a
        # truncated JSON for `status`/CI to parse, and never a umask-wide mode.
        _atomic_write_json(path, result.to_dict())
        # The executor's own per-dispatch report, so `--status` and the D3k rung
        # accounting look identical on both halves of the executor.
        report = BatchReport(
            phase=self.phase, kind=self.kind, phase_dir=self.phase_dir,
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
            skipped_ids = set(result.human_plan.skipped)
            expected_ids_by_task = {
                str(task.get("task_id", "")):
                    [str(i) for i in (task.get("expected_ids") or [])]
                for task in self.tasks
                if str(task.get("task_id", "")) not in skipped_ids
            }
            total = len(self.tasks) - len(skipped_ids)
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
                            for o in outcomes
                        ],
                        "failed_task_ids": [
                            o.task_id for o in outcomes if not o.ok
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
                        # D3l: this flag renders commands and starts no child, so
                        # the nested-exec preflight keeps the declared verdict
                        # (a `denied` host is still refused) and skips its
                        # throwaway dispatch, which is a real turn.
                        probe_nested=False,
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
