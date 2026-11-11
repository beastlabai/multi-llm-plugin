"""The write verdict's truth table: every guard against every claim row.

``WriteBatchScheduler._classify`` decides which outcome a schema-valid write
reply lands on (D3g). Four guards come first, in this order:

1. the write policy, over every path the batch touched (D3l), whether the
   offending write is the task's own, one a sibling also saw, one with no
   rollback point, or a path the commit review's guard watches;
2. a replaced target on the plugin-landlock rung (F1);
3. the jail's denials (F1 "a denied edit is never recorded as applied");
4. a verification-only task, which declares no files;

then the claim rows: whether the declared files moved, the exit code (D3k: a
non-zero exit never lands on APPLIED), the claimed status, the ids and
declines, and the writes outside the declared set.

The tables (:data:`TABLES`): every claim input with no guard; a
verification-only task's inputs; every combination of the guards against
every named claim row; each guard alone against every verification-only row
and reply shape; every list a reason prints at the lengths around its cut;
the wording earlier dispatches of the batch change. Many cases land on the
same record, which the golden file stores once: its distinct records and
reasons, not the case count, say how much is pinned.

Each case is one point of that space. :func:`run_case` drives the real
``_classify`` on it and records the verdict (``write_outcome``, ``ok``,
``applied_without_changes``, ``status``, the exact ``reason``) and every side
effect the verdict owns (the sweep consumed once, the paths the policy saw, the
restore, the audit lines, the jail's records, the log), and the I/O boundaries
it crossed, in the order it first crossed each: a guard's facts are read only
while no guard before it has matched. The golden file
``tests/golden/write_verdict_table.json`` holds what the code decided for every
case. Regenerate it with ``MULTI_LLM_UPDATE_GOLDEN=1`` only when a verdict or
its wording changes on purpose, and read the diff: ``status``, ``summary.md``
and the stop messages print these reasons.

The scheduler is built with ``__new__``: its I/O boundaries are fakes (the
pre-batch snapshot, the undeclared-write sweep, the commit watch, the write
policy, the target's inode, the audit log), so a case needs no git, no runner,
and no file but the temporary root the jail paths resolve under. Everything
between those boundaries is production code. The rules themselves:
``docs/development/write-scheduler.md`` ("The verdict table").
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import landlock_dispatch  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_batch_scheduler as wbs  # noqa: E402
from utils import write_commit_leg  # noqa: E402
from utils import write_grant  # noqa: E402
from utils import write_snapshots  # noqa: E402
from utils import write_sweep  # noqa: E402
from utils import write_verdict as verdict  # noqa: E402
from utils.providers.base import SANDBOX_RUNG_PLUGIN_LANDLOCK  # noqa: E402
from utils.write_safety import PolicyViolation  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "write_verdict_table.json"

TASK_ID = "T1"
SIBLING = "T2"
STRAY = "notes.md"
DENIED = "secrets/denied.txt"
WATCHED = "guarded.txt"
SUMMARY = "did the batch"
SNAPSHOT_DIR = "<snapshot-dir>"
ROOT_TOKEN = "<root>"
DECLARED = {
    "apply_suggestions": "plans/p.md",
    "apply_code_fixes": "src/a.py",
    "implement": "src/a.py",
}
KINDS = tuple(DECLARED)

# -- the axes ----------------------------------------------------------------

EXITS = (0, None, 1)
CLAIMS = ("applied", "no-op", "failed", "")
#: The ids and declines a reply carries against the batch's items (``conflict``: an id
#: in both ``ids`` and ``declined``).
ATTEST = ("complete", "missing", "no_ids", "declined_one", "declined_all", "conflict")
#: What the sweep saw outside the declared set.
STRAY_KINDS = ("none", "own", "shared", "stranded", "unobserved")
#: The write policy's hit: none, a denied write the sweep saw (the task's own,
#: one a sibling also saw, one with no retained bytes), or a path the commit
#: review's guard watches. Each adds its own path, so it meets every claim row.
POLICY = ("none", "own", "shared", "stranded", "watched")
#: The target's inode across the dispatch (``off_rung``: it changed, but the
#: dispatch was not on the plugin-landlock rung).
REPLACED = ("none", "swapped", "gone", "off_rung")
#: The jail's denials (``clean``: on the rung, nothing denied).
JAIL = ("none", "clean", "hard", "in_scope", "replacement", "out_of_scope")
#: ``declared``: the task declares files. Otherwise a verification-only
#: implement task and what its reply's ``check`` says, one shape for each
#: reading ``read_check_reply`` gives the verdict: no ``check``; a pass;
#: findings it listed; ``passed: false`` with none listed; ``could_not_run``
#: with none listed; ``could_not_run`` beside findings it listed.
CHECK = ("declared", "vo_none", "vo_passed", "vo_findings", "vo_failed_bare",
         "vo_could_not_run", "vo_could_not_run_listed")
#: This dispatch against earlier ones of the same batch.
HISTORY = ("fresh", "reused", "resent", "resend_note")
#: The lengths a printed list is tried at: an empty batch, one item, and
#: either side of each cut (a path list ends in "…" after its fifth path, a
#: findings list in "+N more" after its third finding).
SIZES = (0, 1, 5, 6)
#: Whose a run of stray writes is: the task's own, all also seen by a sibling
#: running at the same time, or all but the first seen by it.
SPLITS = ("own", "shared", "mixed")

#: The claim rows, by name: (exit code, claim, declared files moved, attest,
#: stray). Every one reaches its row with no guard firing (asserted below).
CLAIM_ROWS: Dict[str, Tuple[Optional[int], str, bool, str, str]] = {
    "partial-crashed": (1, "applied", True, "complete", "none"),
    "partial-failed": (0, "failed", True, "complete", "none"),
    "partial-missing": (0, "applied", True, "missing", "none"),
    "partial-unattested": (0, "applied", True, "no_ids", "none"),
    "partial-all-declined": (0, "applied", True, "declined_all", "none"),
    "applied-no-op-claimed": (0, "no-op", True, "complete", "none"),
    "applied": (0, "applied", True, "complete", "none"),
    "applied-some-declined": (0, "applied", True, "declined_one", "none"),
    "failed-crashed": (1, "applied", False, "complete", "none"),
    "failed-claimed": (0, "failed", False, "complete", "none"),
    "no-op": (0, "no-op", False, "complete", "none"),
    "no-op-all-declined": (0, "applied", False, "declined_all", "none"),
    "applied-outside": (0, "applied", False, "complete", "own"),
    "partial-outside": (0, "applied", False, "missing", "own"),
    "not-applied": (0, "applied", False, "complete", "none"),
    "not-applied-shared": (0, "applied", False, "complete", "shared"),
    "not-applied-unobserved": (0, "applied", False, "complete", "unobserved"),
}


#: A verification-only task's inputs, by name: (exit code, claim, an entry an
#: earlier dispatch's sweep adopted moved, stray). With no guard each lands on
#: the row its name gives, or, for ``replied*``, the one its ``check`` gives.
CHECK_ROWS: Dict[str, Tuple[Optional[int], str, bool, str]] = {
    "unobserved": (0, "applied", False, "unobserved"),
    "moved": (0, "applied", False, "own"),
    "moved-earlier": (0, "applied", True, "none"),
    "replied": (0, "applied", False, "none"),
    "replied-failed": (0, "failed", False, "none"),
    "crashed": (1, "applied", False, "none"),
}
#: Each guard alone, as (policy, replaced, jail), and none. A replaced target
#: is watched under a plan-file or tasks grant only, so on implement's
#: repo-wide grant it never matches: one value shows that.
GUARDS_ALONE: Tuple[Tuple[str, str, str], ...] = (
    (("none", "none", "none"),)
    + tuple((policy, "none", "none") for policy in POLICY[1:])
    + (("none", "swapped", "none"),)
    + tuple(("none", "none", jail) for jail in JAIL[1:])
)


@dataclass(frozen=True)
class Case:
    kind: str
    exit_code: Optional[int]
    claimed: str
    changed: bool
    attest: str = "complete"
    stray: str = "none"
    policy: str = "none"
    replaced: str = "none"
    jail: str = "none"
    check: str = "declared"
    history: str = "fresh"
    #: How many items the batch holds.
    items: int = 2
    #: How many paths the stray writes hold, and the policy's denied writes.
    paths: int = 1
    #: How many findings a reply's ``check`` lists.
    findings: int = 1

    def key(self) -> str:
        return "|".join(f"{name}={value!r}" for name, value in asdict(self).items())

    @property
    def verification_only(self) -> bool:
        return self.check != "declared"


def table_claims() -> Iterator[Case]:
    """Every claim input, no guard firing."""
    for kind, exit_code, claimed, changed, attest, stray in itertools.product(
        KINDS, EXITS, CLAIMS, (False, True), ATTEST, STRAY_KINDS,
    ):
        yield Case(kind, exit_code, claimed, changed, attest, stray)


def table_verification() -> Iterator[Case]:
    """A verification-only implement task against the claim inputs; its
    ``changed`` is an entry an earlier dispatch's sweep adopted."""
    for check, exit_code, claimed, changed, attest, stray in itertools.product(
        CHECK[1:], EXITS, CLAIMS, (False, True),
        ("complete", "no_ids", "declined_all"), STRAY_KINDS,
    ):
        yield Case("implement", exit_code, claimed, changed, attest, stray,
                   check=check)


def _sweep_saw(stray: str, policy: str) -> bool:
    """Whether the policy's hit can meet these stray writes: a sweep that
    could not look saw no path to deny."""
    return not (stray == "unobserved" and policy in ("own", "shared", "stranded"))


def table_guards() -> Iterator[Case]:
    """Every combination of the guards against every named claim row and
    kind: each guard is seen winning over each row and the guards after it."""
    for name, kind, policy, replaced, jail in itertools.product(
        CLAIM_ROWS, KINDS, POLICY, REPLACED, JAIL,
    ):
        exit_code, claimed, changed, attest, stray = CLAIM_ROWS[name]
        if _sweep_saw(stray, policy):
            yield Case(kind, exit_code, claimed, changed, attest, stray,
                       policy=policy, replaced=replaced, jail=jail)


def table_check_guards() -> Iterator[Case]:
    """Each guard alone against every verification-only row and every reply
    shape: a guard wins over the check's findings, its pass and its could
    not run alike."""
    for name, check, (policy, replaced, jail) in itertools.product(
        CHECK_ROWS, CHECK[1:], GUARDS_ALONE,
    ):
        exit_code, claimed, changed, stray = CHECK_ROWS[name]
        if _sweep_saw(stray, policy):
            yield Case("implement", exit_code, claimed, changed, "complete", stray,
                       policy=policy, replaced=replaced, jail=jail, check=check)


def table_sizes() -> Iterator[Case]:
    """Every list a reason prints, at each of :data:`SIZES` or either side of
    its cut: the batch's items against every named claim row; runs of stray
    writes, the task's own, a sibling's or mixed, against every distinct
    claim input; the policy's writes with no rollback point; a
    verification-only task's moved paths and its findings."""
    for name, kind, items in itertools.product(CLAIM_ROWS, KINDS, SIZES):
        exit_code, claimed, changed, attest, stray = CLAIM_ROWS[name]
        yield Case(kind, exit_code, claimed, changed, attest, stray, items=items)
    inputs = sorted({row[:4] for row in CLAIM_ROWS.values()}, key=repr)
    for (exit_code, claimed, changed, attest), kind, split, paths in itertools.product(
        inputs, KINDS, SPLITS, SIZES[2:],
    ):
        yield Case(kind, exit_code, claimed, changed, attest, split, paths=paths)
    for kind, paths in itertools.product(KINDS, SIZES[1:]):
        yield Case(kind, 0, "applied", True, policy="stranded", paths=paths)
    for paths in SIZES[2:]:
        yield Case("implement", 0, "applied", False, stray="own", check="vo_none",
                   paths=paths)
    for name, check, findings in itertools.product(
        ("replied", "replied-failed"), ("vo_findings", "vo_could_not_run_listed"),
        (3, 4),
    ):
        exit_code, claimed, changed, stray = CHECK_ROWS[name]
        yield Case("implement", exit_code, claimed, changed, "complete", stray,
                   check=check, findings=findings)


def table_history() -> Iterator[Case]:
    """The wording that depends on earlier dispatches of the same batch."""
    for history, kind, claimed, changed, stray, check in itertools.product(
        HISTORY, KINDS, CLAIMS, (False, True), STRAY_KINDS, CHECK[:2],
    ):
        if check != "declared" and kind != "implement":
            continue
        yield Case(kind, 0, claimed, changed, "complete", stray,
                   check=check, history=history)


TABLES: Dict[str, Callable[[], Iterator[Case]]] = {
    "claims": table_claims,
    "verification": table_verification,
    "guards": table_guards,
    "check_guards": table_check_guards,
    "sizes": table_sizes,
    "history": table_history,
}


# -- the fakes ---------------------------------------------------------------

class Calls(list):
    """The I/O boundaries ``_classify`` crossed, each named the first time:
    the order it gathered its facts in, and which it never read."""

    def __call__(self, name: str) -> None:
        if name not in self:
            self.append(name)


def _untraced(name: str) -> None:
    pass


class _Entry:
    def __init__(self, relative: str, declared: bool) -> None:
        self.relative = relative
        self.declared = declared


class FakeSnapshot:
    """The slice of ``TaskSnapshot`` ``_classify`` reads. ``moved`` is every
    path whose bytes differ now; ``changed()`` reports the moved entries,
    adopted ones included, as the real one does, so the order of the byte
    comparison and the sweep's adoption shows."""

    def __init__(self, declared: List[str], moved: List[str],
                 adopted: Tuple[str, ...] = ()) -> None:
        self.entries = [_Entry(path, True) for path in declared]
        self.entries += [_Entry(path, False) for path in adopted]
        self.moved = set(moved)
        self.directory = SNAPSHOT_DIR
        self.restored: List[List[str]] = []
        self.calls: Callable[[str], None] = _untraced

    def changed(self) -> List[str]:
        return [e.relative for e in self.entries if e.relative in self.moved]

    def declared_entries(self) -> List[_Entry]:
        self.calls("declarations")
        return [e for e in self.entries if e.declared]

    def adopt(self, relative: str) -> None:
        if all(e.relative != relative for e in self.entries):
            self.entries.append(_Entry(relative, False))

    def observed(self) -> Dict[str, Dict[str, Any]]:
        return {
            e.relative: {
                "sha256_before": f"before:{e.relative}",
                "sha256_after": (
                    f"after:{e.relative}" if e.relative in self.moved
                    else f"before:{e.relative}"
                ),
                "declared": e.declared,
            }
            for e in self.entries
        }

    def restore(self, paths: List[str]) -> Dict[str, str]:
        self.calls("restore")
        self.restored.append(list(paths))
        held = {e.relative for e in self.entries}
        return {
            path: "restored" if path in held
            else "skipped: not in this task's snapshot"
            for path in paths
        }


class FakePolicy:
    def __init__(self, denied: List[str], calls: Callable[[str], None] = _untraced) -> None:
        self.denied = list(denied)
        self.checked: List[List[str]] = []
        self.calls = calls

    def check(self, paths, *, when: str) -> None:
        assert when == "post-run"
        self.calls("policy")
        paths = list(paths)
        self.checked.append(paths)
        hits = [path for path in paths if path in self.denied]
        if hits:
            raise PolicyViolation(
                f"WRITE POLICY: {', '.join(hits)} "
                f"{'is' if len(hits) == 1 else 'are'} denied by driver.write_paths.",
                hits, rule="deny",
            )


class FakeCommitApply:
    RESTORED_FROM_COPY = "restored from the commit review's copy"
    DELETED_ABSENT_BEFORE = "deleted (absent before the batch)"

    def __init__(self, calls: Callable[[str], None] = _untraced) -> None:
        self.calls = calls

    def restore_watched(self, guard, missing, *, cwd, baseline):
        self.calls("guard-restore")
        return {path: self.RESTORED_FROM_COPY for path in missing}


class FakeAudit:
    """``AuditLog``'s two calls ``_classify`` makes, recorded."""

    def __init__(self, calls: Callable[[str], None] = _untraced) -> None:
        self.restores: List[Dict[str, Any]] = []
        self.completions: List[Dict[str, Any]] = []
        self.calls = calls

    def restored(self, **kwargs: Any) -> None:
        self.calls("audit-restore")
        self.restores.append(kwargs)

    def completed(self, **kwargs: Any) -> None:
        self.calls("audit-completion")
        self.completions.append(kwargs)


class Identities(dict):
    """``WriteGrant.target_identity_before``, its spending named."""

    def __init__(self, items: Dict[str, Any], calls: Callable[[str], None]) -> None:
        super().__init__(items)
        self.calls = calls

    def pop(self, *args: Any) -> Any:
        self.calls("identity-spent")
        return super().pop(*args)


class TracedDenials:
    """A ``JailDenials`` whose two reads that are not fields are named:
    keeping its record, and listing its denied paths for resolving."""

    def __init__(self, denials: Any, calls: Callable[[str], None]) -> None:
        self._denials = denials
        self._calls = calls

    def __getattr__(self, name: str) -> Any:
        return getattr(self._denials, name)

    def to_dict(self) -> Dict[str, Any]:
        self._calls("jail-record")
        return self._denials.to_dict()

    def denied_paths(self) -> List[str]:
        self._calls("jail-paths")
        return self._denials.denied_paths()


def _traced(function: Callable[..., Any], calls: Callable[[str], None],
            name: str) -> Callable[..., Any]:
    def traced(*args: Any, **kwargs: Any) -> Any:
        calls(name)
        return function(*args, **kwargs)
    return traced


# -- the harness -------------------------------------------------------------

def _items(case: Case) -> List[str]:
    return [f"s{index}" for index in range(1, case.items + 1)]


def _paths(case: Case, one: str) -> List[str]:
    """*one*, or that many paths beside it."""
    if case.paths == 1:
        return [one]
    stem, dot, suffix = one.rpartition(".")
    return [f"{stem}-{index}{dot}{suffix}" for index in range(1, case.paths + 1)]


def _findings(case: Case) -> List[str]:
    if case.findings == 1:
        return ["lint: 2 errors"]
    return [f"lint {index}: 2 errors." for index in range(1, case.findings + 1)]


def _result(case: Case) -> Dict[str, Any]:
    result: Dict[str, Any] = {"summary": SUMMARY,
                              "files_touched": [DECLARED[case.kind]]}
    if case.claimed:
        result["status"] = case.claimed
    items = _items(case)
    ids = {
        "complete": items, "missing": items[:-1], "no_ids": None,
        "declined_one": items[:-1], "declined_all": None, "conflict": items,
    }[case.attest]
    if ids is not None:
        result["ids"] = ids
    declined = {
        "declined_one": items[-1:], "declined_all": items, "conflict": items[-1:],
    }.get(case.attest)
    if declined:
        result["declined"] = [
            {"id": item, "reason": f"{item} is already covered"} for item in declined
        ]
    check = {
        "vo_passed": {"passed": True},
        "vo_findings": {"passed": False, "findings": _findings(case)},
        "vo_failed_bare": {"passed": False},
        "vo_could_not_run": {"could_not_run": True},
        "vo_could_not_run_listed": {
            "could_not_run": True, "passed": False, "findings": _findings(case),
        },
    }.get(case.check)
    if check is not None:
        result["check"] = check
    return result


def _undeclared(case: Case) -> wbs.UndeclaredWrites:
    if case.stray == "unobserved":
        return wbs.UndeclaredWrites(observed=False)
    paths: List[str] = []
    restorable: List[str] = []
    shared: Dict[str, List[str]] = {}
    for group, split in ((_paths(case, STRAY), case.stray),
                         (_paths(case, DENIED), case.policy)):
        if split not in ("own", "shared", "stranded", "mixed"):
            continue
        for index, path in enumerate(group):
            paths.append(path)
            if split != "stranded":
                restorable.append(path)
            if split == "shared" or (split == "mixed" and index):
                shared.setdefault(SIBLING, []).append(path)
    return wbs.UndeclaredWrites(paths=paths, restorable=restorable, shared=shared)


def _jail(case: Case, root: Path) -> Any:
    declared = str(root / DECLARED[case.kind])
    if case.jail == "none":
        return None
    if case.jail == "clean":
        return landlock_dispatch.JailDenials()
    if case.jail == "hard":
        return landlock_dispatch.JailDenials(failed_file_changes=[[declared]])
    if case.jail == "in_scope":
        return landlock_dispatch.JailDenials(in_scope_paths=[declared])
    if case.jail == "replacement":
        return landlock_dispatch.JailDenials(replacement_paths=[declared])
    return landlock_dispatch.JailDenials(
        out_of_scope_count=1, out_of_scope_paths=[str(root / "src" / "b.py")],
    )


def _scheduler(case: Case, root: Path) -> Tuple[Any, Dict[str, Any]]:
    """A ``WriteBatchScheduler`` with fakes at every I/O boundary, and the
    probes the record reads back."""
    scheduler = wbs.WriteBatchScheduler.__new__(wbs.WriteBatchScheduler)
    calls = Calls()
    probes: Dict[str, Any] = {"sweeps": 0, "log": [], "calls": calls}
    scheduler.kind = case.kind
    scheduler.tasks = [{"task_id": TASK_ID, "expected_ids": _items(case)}]
    scheduler.state = None
    scheduler.repo_root = str(root)
    scheduler.output_dir = str(root / ".multi-llm" / "p")
    scheduler.plan_file = str(root / "plans" / "p.md")
    scheduler.write_scope = wbs.write_scope_for_kind(case.kind)
    scheduler.write_target = (
        str(root / "plans" / "p.md") if case.kind == "apply_suggestions" else None
    )
    scheduler.jail_request = write_grant.build_jail_request(
        kind=scheduler.kind, write_scope=scheduler.write_scope,
        repo_root=scheduler.repo_root, output_dir=scheduler.output_dir,
        write_target=scheduler.write_target, plan_file=scheduler.plan_file,
    )
    scheduler.log = probes["log"].append
    scheduler.audit = FakeAudit(calls)
    scheduler.policy = FakePolicy(
        _paths(case, DENIED) if case.policy in ("own", "shared", "stranded")
        else [WATCHED] if case.policy == "watched" else [],
        calls,
    )
    scheduler._audited = set()
    scheduler._told_checks = {}
    scheduler._listed_outcomes = {}
    # The sweep's audit rows, the commit guard's restore and the write
    # grant's records run for real, over the same root; the sweep, the watch
    # read and the target's inode below are faked.
    run = SimpleNamespace(repo_root=str(root), kind=case.kind, anchor=None,
                          log=scheduler.log)
    scheduler._sweep = write_sweep.ContainmentSweep(run)
    scheduler._commit = write_commit_leg.CommitGuardLeg(
        run, effects_lock=threading.Lock(), refused=None,
    )
    scheduler._grant = write_grant.WriteGrant(
        run, write_scope=scheduler.write_scope, write_target=scheduler.write_target,
        jail_request=scheduler.jail_request, dispatch_rungs={},
    )
    scheduler._snapshots = write_snapshots.PreBatchSnapshots(
        run, resume=False, executor_dispatch=False, grant=scheduler._grant,
        evidence=None, runner_for=None,
    )
    if case.history == "reused":
        scheduler._snapshots.reused.add(TASK_ID)
    grant = scheduler._grant
    grant.target_identity_before = Identities({TASK_ID: (1, 100)}, calls)
    after = {"none": (1, 100), "swapped": (1, 200), "gone": None,
             "off_rung": (1, 200)}[case.replaced]
    grant.identity_of = _traced(lambda path: after, calls, "identity-read")
    # Where a path is resolved, and can fail.
    grant.relative = _traced(grant.relative, calls, "path-resolved")
    grant.in_grant = _traced(grant.in_grant, calls, "path-resolved")

    undeclared = _undeclared(case)

    def sweep(task_id: str, snapshot: FakeSnapshot) -> wbs.UndeclaredWrites:
        calls("sweep")
        probes["sweeps"] += 1
        for path in undeclared.restorable:
            snapshot.adopt(path)
        return undeclared

    scheduler._undeclared_writes = sweep
    if case.policy == "watched":
        commit_apply = FakeCommitApply(calls)
        watch = ([WATCHED], commit_apply, {"protected": [WATCHED]})
    else:
        watch = ([], None, None)
    scheduler._commit.watch_after = _traced(lambda task_id: watch, calls, "commit-watch")
    return scheduler, probes


def _snapshot(case: Case) -> FakeSnapshot:
    declared = DECLARED[case.kind]
    moved = [declared] if case.changed else []
    if case.stray in ("own", "shared", "stranded", "mixed"):
        moved += _paths(case, STRAY)
    if case.policy in ("own", "shared", "stranded"):
        moved += _paths(case, DENIED)
    if case.policy == "watched":
        moved.append(WATCHED)
    if case.verification_only:
        # No declared file: a moved one is an entry an earlier dispatch's
        # sweep adopted, which a reused snapshot still carries.
        return FakeSnapshot([], moved, adopted=(declared,) if case.changed else ())
    return FakeSnapshot([declared], moved)


def classify(case: Case, root: Path,
             edit: Callable[[Any, FakeSnapshot, Any], None] = lambda *_: None,
             ) -> Tuple[Any, FakeSnapshot, Any, Dict[str, Any]]:
    """``_classify`` on *case*, after *edit* (scheduler, snapshot, outcome)
    adjusts the fakes: the scheduler, the snapshot, the outcome and the
    probes, ``decided`` among them, the facts ``decide`` was given."""
    scheduler, probes = _scheduler(case, root)
    calls = probes["calls"]
    snapshot = _snapshot(case)
    snapshot.calls = calls
    profile = executor.profile_for_kind(case.kind)
    outcome = executor.TaskOutcome(
        task_id=TASK_ID, kind=case.kind, runner="native",
        status=executor.STATUS_OK, profile=profile,
    )
    edit(scheduler, snapshot, outcome)
    denials = _jail(case, root)

    def decide(facts: Any) -> Any:
        probes["decided"] = facts
        return verdict.decide(facts)

    with mock.patch.object(wbs, "_as_jail_denials",
                           _traced(wbs._as_jail_denials, calls, "jail-read")), \
            mock.patch.object(wbs, "read_check_reply",
                              _traced(wbs.read_check_reply, calls, "check-read")), \
            mock.patch.object(wbs, "decide", decide, create=True):
        scheduler._classify(
            scheduler.tasks[0], outcome, snapshot, _result(case),
            runner="native", profile=profile, attempt=1, exit_code=case.exit_code,
            duration=0.0,
            sandbox_rung=(
                None if case.replaced == "off_rung" else SANDBOX_RUNG_PLUGIN_LANDLOCK
            ),
            jail_denials=None if denials is None else TracedDenials(denials, calls),
            resent=case.history == "resent",
            resend_note=(
                "not sent again: the census is gone" if case.history == "resend_note"
                else None
            ),
        )
    return scheduler, snapshot, outcome, probes


def run_case(case: Case, root: Path) -> Dict[str, Any]:
    """What ``_classify`` decides on *case*, every root path as ``<root>``."""
    scheduler, snapshot, outcome, probes = classify(case, root)
    completions = scheduler.audit.completions
    record = {
        "write_outcome": outcome.write_outcome,
        "ok": outcome.ok,
        "applied_without_changes": bool(outcome.applied_without_changes),
        "status": outcome.status,
        "reason": outcome.reason,
        "unapplied_ids": list(outcome.unapplied_ids or []),
        "files_touched": list(outcome.files_touched or []),
        "unrestorable": list(outcome.unrestorable or []),
        "runner_notes": outcome.runner_notes,
        "check_findings": outcome.check_findings,
        "verification_only": bool(outcome.verification_only),
        "effects": {
            "sweeps": probes["sweeps"],
            "policy_checked": scheduler.policy.checked,
            "restored": snapshot.restored,
            "audit_restores": [sorted(call["report"].items())
                               for call in scheduler.audit.restores],
            "audit_completions": len(completions),
            "audited_reason_is_final": bool(completions) and (
                completions[-1]["reason"] == outcome.reason
                and completions[-1]["write_outcome"] == outcome.write_outcome
                and completions[-1]["files_touched"] == outcome.files_touched
            ),
            "jail_failed": TASK_ID in scheduler._grant.jail_failed,
            "replaced_target": TASK_ID in scheduler._grant.replaced_targets,
            "jail_denials_recorded": TASK_ID in scheduler._grant.jail_denials,
            "jail_note": scheduler._grant.jail_notes.get(TASK_ID),
            "identity_popped": TASK_ID not in scheduler._grant.target_identity_before,
            "log": probes["log"],
        },
        "calls": list(probes["calls"]),
    }
    return _scrub(record, root)


def _scrub(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        for spelling in {str(root), root.as_posix()}:
            value = value.replace(spelling, ROOT_TOKEN)
        return value
    if isinstance(value, (list, tuple)):
        return [_scrub(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _scrub(item, root) for key, item in value.items()}
    return value


# -- the golden file ---------------------------------------------------------

#: A golden record is a list, in this order; ``reason`` and ``effects`` are
#: indexes into the golden's ``reasons`` and ``effects``, and an effects
#: entry lists :data:`EFFECT_FIELDS`.
RECORD_FIELDS = (
    "write_outcome", "ok", "applied_without_changes", "status", "reason",
    "effects", "unapplied_ids", "files_touched", "unrestorable",
    "runner_notes", "check_findings", "verification_only",
)
EFFECT_FIELDS = (
    "sweeps", "policy_checked", "restored", "audit_restores",
    "audit_completions", "audited_reason_is_final", "jail_failed",
    "replaced_target", "jail_denials_recorded", "jail_note",
    "identity_popped", "log",
)
#: A table's cases are its record indexes, base 36, this many digits each,
#: and the indexes of the boundaries each crossed (``calls``), one digit each.
INDEX_WIDTH = 2
CALLS_WIDTH = 1
_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _encode(indexes: List[int], width: int = INDEX_WIDTH) -> List[str]:
    """*indexes* as fixed-width base-36, 40 cases a line."""
    def one(value: int) -> str:
        out = ""
        for _ in range(width):
            value, digit = divmod(value, len(_DIGITS))
            out = _DIGITS[digit] + out
        assert value == 0, f"more entries than {width} base-36 digit(s) can name"
        return out

    text = "".join(one(i) for i in indexes)
    step = 40 * width
    return [text[i:i + step] for i in range(0, len(text), step)]


def _decode(lines: List[str], width: int = INDEX_WIDTH) -> List[int]:
    text = "".join(lines)
    return [int(text[i:i + width], len(_DIGITS))
            for i in range(0, len(text), width)]


def build_golden(root: Path) -> Dict[str, Any]:
    """Every table's cases as indexes into ``records`` and ``calls``, each
    distinct reason, effects and run of calls stored once."""
    reasons: List[Optional[str]] = []
    effects: List[List[Any]] = []
    calls: List[List[str]] = []
    records: List[List[Any]] = []
    index: Dict[str, int] = {}
    tables: Dict[str, Any] = {}
    for name, cases in TABLES.items():
        column: List[int] = []
        crossed: List[int] = []
        keys: List[str] = []
        for case in cases():
            record = run_case(case, root)
            trace = record.pop("calls")
            if trace not in calls:
                calls.append(trace)
            crossed.append(calls.index(trace))
            key = json.dumps(record, sort_keys=True)
            if key not in index:
                effect = [record["effects"][field] for field in EFFECT_FIELDS]
                if record["reason"] not in reasons:
                    reasons.append(record["reason"])
                if effect not in effects:
                    effects.append(effect)
                index[key] = len(records)
                records.append([
                    reasons.index(record["reason"]) if field == "reason"
                    else effects.index(effect) if field == "effects"
                    else record[field]
                    for field in RECORD_FIELDS
                ])
            column.append(index[key])
            keys.append(case.key())
        tables[name] = {"count": len(column), "cases_sha256": _digest(keys),
                        "records": _encode(column),
                        "calls": _encode(crossed, CALLS_WIDTH)}
    return {
        "about": (
            "What WriteBatchScheduler._classify decides for every case of "
            "tests/test_write_verdict_table.py. tables.<name>.records holds "
            "the record index of each case in order, base 36, "
            f"{INDEX_WIDTH} digits each, and tables.<name>.calls its index "
            f"into calls, {CALLS_WIDTH} digit each: the I/O boundaries it "
            "crossed, each named the first time. A record lists "
            "record_fields, its reason and effects indexes into reasons and "
            "effects; an effects entry lists effect_fields. Regenerate with "
            "MULTI_LLM_UPDATE_GOLDEN=1 and read the diff."
        ),
        "record_fields": list(RECORD_FIELDS),
        "effect_fields": list(EFFECT_FIELDS),
        "reasons": reasons,
        "effects": effects,
        "calls": calls,
        "records": records,
        "tables": tables,
    }


def _digest(keys: List[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def dump_golden(golden: Dict[str, Any]) -> str:
    """JSON with one reason, effect or record per line, so a changed verdict
    is a readable diff."""
    def rows(items: List[Any], indent: str = "  ") -> str:
        if not items:
            return "[]"
        inner = indent + "  "
        body = (",\n" + inner).join(
            json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items
        )
        return f"[\n{inner}{body}\n{indent}]"

    tables = ",\n".join(
        f'    {json.dumps(name)}: {{\n'
        f'      "count": {table["count"]},\n'
        f'      "cases_sha256": {json.dumps(table["cases_sha256"])},\n'
        f'      "records": {rows(table["records"], "      ")},\n'
        f'      "calls": {rows(table["calls"], "      ")}\n'
        f'    }}'
        for name, table in golden["tables"].items()
    )
    return (
        "{\n"
        f'  "about": {json.dumps(golden["about"])},\n'
        f'  "record_fields": {json.dumps(golden["record_fields"])},\n'
        f'  "effect_fields": {json.dumps(golden["effect_fields"])},\n'
        f'  "reasons": {rows(golden["reasons"])},\n'
        f'  "effects": {rows(golden["effects"])},\n'
        f'  "calls": {rows(golden["calls"])},\n'
        f'  "records": {rows(golden["records"])},\n'
        f'  "tables": {{\n{tables}\n  }}\n'
        "}\n"
    )


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("verdict").resolve() / "repo"
    path.mkdir()
    return path


@pytest.fixture(scope="module")
def actual(root) -> Dict[str, Any]:
    golden = build_golden(root)
    if os.environ.get("MULTI_LLM_UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(dump_golden(golden), encoding="utf-8")
    return golden


@pytest.fixture(scope="module")
def expected() -> Dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _resolved(golden: Dict[str, Any], table: str) -> List[Dict[str, Any]]:
    """Each of *table*'s cases' record, as a dict with its reason, effects and
    calls."""
    out = []
    column = golden["tables"][table]
    for i, crossed in zip(_decode(column["records"]),
                          _decode(column["calls"], CALLS_WIDTH)):
        record = dict(zip(golden["record_fields"], golden["records"][i]))
        record["reason"] = golden["reasons"][record["reason"]]
        record["effects"] = dict(zip(golden["effect_fields"],
                                     golden["effects"][record["effects"]]))
        record["calls"] = golden["calls"][crossed]
        out.append(record)
    return out


# -- the tests ---------------------------------------------------------------

@pytest.mark.parametrize("table", list(TABLES))
def test_every_case_decides_what_the_golden_file_records(table, actual, expected):
    assert actual["tables"][table]["cases_sha256"] == expected["tables"][table]["cases_sha256"], (
        f"the {table!r} table's cases changed; regenerate the golden file "
        f"(MULTI_LLM_UPDATE_GOLDEN=1) and read the diff"
    )
    cases = list(TABLES[table]())
    got = _resolved(actual, table)
    want = _resolved(expected, table)
    wrong = [
        (case, mine, theirs)
        for case, mine, theirs in zip(cases, got, want) if mine != theirs
    ]
    shown = "\n\n".join(
        f"{case.key()}\n"
        + "\n".join(
            f"  {field}:\n    got  {mine[field]!r}\n    want {theirs[field]!r}"
            for field in theirs if mine.get(field) != theirs[field]
        )
        for case, mine, theirs in wrong[:5]
    )
    assert not wrong, f"{len(wrong)} of {len(cases)} {table} case(s) differ:\n\n{shown}"


def _all_cases(actual) -> Iterator[Tuple[Case, Dict[str, Any]]]:
    for table, cases in TABLES.items():
        yield from zip(cases(), _resolved(actual, table))


def test_the_sweep_is_consumed_exactly_once(actual):
    assert {record["effects"]["sweeps"] for _, record in _all_cases(actual)} == {1}


def test_every_case_ends_in_one_audit_line_carrying_the_verdict(actual):
    for case, record in _all_cases(actual):
        assert record["effects"]["audit_completions"] == 1, case.key()
        assert record["effects"]["audited_reason_is_final"], case.key()


def test_a_non_zero_exit_never_counts_as_success(actual):
    """D3k: whatever the claim, ids and bytes say."""
    crashed = [(c, r) for c, r in _all_cases(actual) if c.exit_code not in (0, None)]
    assert crashed
    for case, record in crashed:
        assert not record["ok"], case.key()
        assert record["write_outcome"] != executor.WRITE_APPLIED, case.key()


def test_the_policy_runs_before_every_other_branch(actual):
    hits = [(c, r) for c, r in _all_cases(actual) if c.policy != "none"]
    assert hits
    for case, record in hits:
        assert record["write_outcome"] == executor.WRITE_POLICY_VIOLATION, case.key()
        assert record["status"] == executor.STATUS_POLICY_VIOLATION, case.key()
        assert record["effects"]["restored"], case.key()
        assert not record["effects"]["jail_denials_recorded"], case.key()
        assert not record["effects"]["identity_popped"], case.key()


def test_a_replaced_target_wins_over_the_jail_and_every_claim(actual):
    replaced = [
        (c, r) for c, r in _all_cases(actual)
        if c.policy == "none" and c.replaced in ("swapped", "gone")
        and c.kind == "apply_suggestions"
    ]
    assert replaced
    for case, record in replaced:
        assert record["reason"].startswith("FAILED (target replaced)"), case.key()
        assert record["write_outcome"] == executor.WRITE_CLEAN_FAILURE, case.key()
        assert record["effects"]["replaced_target"], case.key()
        assert DECLARED[case.kind] not in record["files_touched"], case.key()


def test_a_hard_jail_denial_is_never_recorded_as_applied(actual):
    """F1, ahead of every claim and of the verification-only verdict."""
    denied = [
        (c, r) for c, r in _all_cases(actual)
        if c.policy == "none" and c.jail in ("hard", "in_scope")
        and not (c.kind == "apply_suggestions" and c.replaced in ("swapped", "gone"))
    ]
    assert denied
    for case, record in denied:
        assert not record["ok"], case.key()
        assert "(jail-denied edit)" in record["reason"], case.key()


def test_a_claim_with_nothing_moved_is_never_applied(actual):
    """The claim is never the verdict: no declared byte, no stray write."""
    still = [
        (c, r) for c, r in _all_cases(actual)
        if not c.changed and c.stray in ("none", "unobserved") and c.policy == "none"
    ]
    assert still
    for case, record in still:
        assert record["write_outcome"] != executor.WRITE_APPLIED, case.key()


def test_the_named_claim_rows_land_where_their_names_say(actual):
    """The rows ``_classify``'s docstring lists, with no guard firing."""
    outcome_by_prefix = {
        "partial": executor.WRITE_PARTIAL,
        "applied": executor.WRITE_APPLIED,
        "failed": executor.WRITE_CLEAN_FAILURE,
        "no-op": executor.WRITE_NO_OP,
        "not-applied": executor.WRITE_CLEAN_FAILURE,
    }
    rows = {
        (c.kind, name): r
        for c, r in zip(table_guards(), _resolved(actual, "guards"))
        for name, inputs in CLAIM_ROWS.items()
        if (c.exit_code, c.claimed, c.changed, c.attest, c.stray) == inputs
        and (c.policy, c.replaced, c.jail, c.check) == ("none", "none", "none", "declared")
    }
    for (kind, name), record in sorted(rows.items()):
        if kind == "apply_suggestions" and name in ("applied-outside", "partial-outside"):
            # A plan batch is judged on its declared file alone (fix round 15).
            assert record["write_outcome"] == executor.WRITE_CLEAN_FAILURE, record
            continue
        if kind == "implement" and "declined" in name:
            # An implement task has no decline: declining it is `failed`.
            assert not record["ok"], record
            continue
        prefix = max((p for p in outcome_by_prefix if name.startswith(p)), key=len)
        assert record["write_outcome"] == outcome_by_prefix[prefix], (kind, name, record)
        assert record["applied_without_changes"] == name.startswith("not-applied"), (kind, name)
    assert len(rows) == len(CLAIM_ROWS) * len(KINDS)


#: The order ``_classify`` first crosses its boundaries in, the snapshot's
#: declarations aside: it reads those for an implement task's kind first, and
#: else only for a reason that names them (``DECLARED_NAMES_ROWS``).
CALL_ORDER = (
    "sweep", "commit-watch", "policy", "restore", "guard-restore",
    "audit-restore", "identity-spent", "identity-read", "jail-read",
    "jail-record", "jail-paths", "path-resolved", "check-read",
    "audit-completion",
)


def test_each_guard_s_facts_are_read_only_while_no_guard_before_it_matched(actual):
    """Reading a later row's facts can fail (a path that does not resolve),
    so it never stands between a matched guard and what its row does."""
    rank = {name: index for index, name in enumerate(CALL_ORDER)}
    for case, record in _all_cases(actual):
        calls = record["calls"]
        ranked = [rank[name] for name in calls if name != "declarations"]
        assert ranked == sorted(ranked), (case.key(), calls)
        after_policy = set(calls[calls.index("policy") + 1:]) - {"declarations"}
        reason = record["reason"] or ""
        if record["write_outcome"] == executor.WRITE_POLICY_VIOLATION:
            assert after_policy <= {"restore", "guard-restore", "audit-restore",
                                    "audit-completion"}, (case.key(), calls)
            assert "declarations" not in calls[calls.index("policy"):], case.key()
        elif reason.startswith("FAILED (target replaced)"):
            assert after_policy <= {"identity-spent", "identity-read", "path-resolved",
                                    "audit-completion"}, (case.key(), calls)
        elif "(jail-denied edit)" in reason:
            assert "check-read" not in calls, (case.key(), calls)
        else:
            assert {"identity-spent", "jail-read", "check-read"} <= set(calls), case.key()


def _skipped_facts(facts: Any, case: Case, root: Path) -> Any:
    """*facts* with every fact ``_classify`` skipped once a guard matched
    filled in from *case*: the later guards' and the check reply's."""
    filled: Dict[str, Any] = {}
    if facts.replaced is None and case.kind == "apply_suggestions" and case.replaced in (
        "swapped", "gone",
    ):
        filled["replaced"] = verdict.TargetReplaced(
            "the plan file", DECLARED[case.kind], 100,
            None if case.replaced == "gone" else 200,
        )
    denials = _jail(case, root)
    if facts.jail is None and denials is not None:
        filled["jail"] = verdict.JailFacts(
            hard=bool(denials.failed_file_changes or denials.in_scope_paths),
            replacements=len(denials.replacement_paths),
        )
    reply = executor.read_check_reply(_result(case))
    filled.update(
        check=None if reply.findings is None else tuple(reply.findings),
        check_listed=reply.listed, check_could_not_run=reply.could_not_run,
    )
    return replace(facts, **filled)


def test_a_matched_guard_is_final_whatever_the_facts_after_it_say(root):
    """Why ``_classify`` may stop gathering at a matched guard: the facts it
    skipped (the later guards', the check reply's) never change what
    ``decide`` says. A table that put a later row, a check's findings say,
    ahead of a guard fails here, though the dispatches above never show it."""
    checked = 0
    for cases in TABLES.values():
        for case in cases():
            _, _, _, probes = classify(case, root)
            gathered = probes["decided"]
            if verdict.guard(gathered) is None:
                continue
            checked += 1
            assert verdict.decide(_skipped_facts(gathered, case, root)) == (
                verdict.decide(gathered)
            ), case.key()
    assert checked


def _resolving_fails(*args: Any, **kwargs: Any) -> Any:
    # What resolving a self-referencing symlink raises before Python 3.13.
    raise RuntimeError("Symlink loop from 'loop'")


def test_a_jail_path_that_fails_to_resolve_never_stops_a_policy_restore(root):
    case = Case("apply_suggestions", 0, "applied", True, policy="own", jail="hard")

    def edit(scheduler: Any, snapshot: FakeSnapshot, outcome: Any) -> None:
        scheduler._grant.jail_facts = _resolving_fails

    scheduler, snapshot, outcome, probes = classify(case, root, edit)
    assert outcome.write_outcome == executor.WRITE_POLICY_VIOLATION
    assert outcome.status == executor.STATUS_POLICY_VIOLATION
    assert snapshot.restored == [[DENIED]]
    assert len(scheduler.audit.restores) == 1
    assert len(scheduler.audit.completions) == 1
    assert "jail-read" not in probes["calls"]


def test_a_jail_path_that_fails_to_resolve_never_stops_a_replaced_target(root):
    case = Case("apply_suggestions", 0, "applied", True, replaced="swapped", jail="hard")

    def edit(scheduler: Any, snapshot: FakeSnapshot, outcome: Any) -> None:
        scheduler._grant.jail_facts = _resolving_fails

    scheduler, snapshot, outcome, probes = classify(case, root, edit)
    assert outcome.reason.startswith("FAILED (target replaced)")
    assert outcome.write_outcome == executor.WRITE_CLEAN_FAILURE
    # Never restored over the file now at the path.
    assert DECLARED[case.kind] not in outcome.files_touched
    assert TASK_ID in scheduler._grant.replaced_targets
    assert len(scheduler.audit.completions) == 1
    assert "jail-read" not in probes["calls"]


def test_a_jail_path_that_fails_to_resolve_with_no_guard_matched_is_raised(root):
    """Nothing before the jail decided, so the failure is the dispatch's: it
    is raised, no verdict is recorded and no audit line claims one. The
    jail's record was kept before any path was resolved."""
    case = Case("apply_code_fixes", 0, "applied", True, jail="hard")
    seen: Dict[str, Any] = {}

    def edit(scheduler: Any, snapshot: FakeSnapshot, outcome: Any) -> None:
        scheduler._grant.jail_facts = _resolving_fails
        seen.update(scheduler=scheduler, outcome=outcome)

    with pytest.raises(RuntimeError, match="Symlink loop"):
        classify(case, root, edit)
    scheduler, outcome = seen["scheduler"], seen["outcome"]
    assert outcome.write_outcome is None
    assert outcome.status == executor.STATUS_OK
    assert scheduler.audit.completions == []
    assert TASK_ID in scheduler._grant.jail_denials
    assert TASK_ID not in scheduler._grant.target_identity_before


def test_a_restore_audit_that_fails_leaves_the_violation_classified(root):
    """The policy's verdict is on the outcome once its paths are restored,
    before the audit line that records the restore."""
    case = Case("apply_code_fixes", 0, "applied", True, policy="own")
    seen: Dict[str, Any] = {}

    def edit(scheduler: Any, snapshot: FakeSnapshot, outcome: Any) -> None:
        def disk_full(**kwargs: Any) -> None:
            raise OSError(28, "No space left on device")

        scheduler.audit.restored = disk_full
        seen.update(snapshot=snapshot, outcome=outcome, scheduler=scheduler)

    with pytest.raises(OSError):
        classify(case, root, edit)
    outcome = seen["outcome"]
    assert seen["snapshot"].restored == [[DENIED]]
    assert outcome.write_outcome == executor.WRITE_POLICY_VIOLATION
    assert outcome.status == executor.STATUS_POLICY_VIOLATION
    assert outcome.reason.startswith("WRITE POLICY:")
    assert f"Restored from the pre-batch snapshot: {DENIED}." in outcome.reason
    assert seen["scheduler"].audit.completions == []
