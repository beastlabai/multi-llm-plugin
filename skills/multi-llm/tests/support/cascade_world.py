"""The cascade archive's world: a plan, its live review and apply dirs, and ``state.json``.

Helpers only (no fixture). :class:`CascadeWorld` lays out one cascade dir
under a plan's composite run dir and writes, stage by stage, what a review
round and its apply leave on disk: ``grouped.json``, ``validation.json``,
``.status.json``, a model's result file, an idle ``.lock`` file;
``write_tasks.json``, ``orchestrator_output.json``, ``write_batch_report.json``;
and the ``state.json`` records a stage writes (``processed_<apply dir>``,
``human_decisions_<apply dir>``, ``phases_completed``, ``scope_policy``),
each stamped as the real writers stamp them. ``utils/cascade_archive.py``
then archives the stage and ``utils/cascade_ledger.py`` reads the archives.

Stamps are fixed (:func:`T`), so every file is the same on every run.
"""

from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils import cascade_ledger as ledger
from utils.output_handler import output_dir_path
from utils.state_manager import stamp_stable_ids
from utils.subagent_tasks import content_hash

BASE = datetime(2026, 11, 13, 9, 0, 0)

PLAN_TEXT = (
    "# Export plan\n\n## Goal\nExport the weekly report as CSV.\n\n"
    "## Step 1: Columns\nList the columns.\n\n## Step 2: Delimiter\nUse a comma.\n"
)

CASCADE_ID = "c0ffee00deadbeef0123456789abcdef"
OTHER_CASCADE_ID = "feedface00000000aaaaaaaabbbbbbbb"


def T(minutes: float) -> str:
    """A naive local stamp *minutes* after a fixed base."""
    return (BASE + timedelta(minutes=minutes)).isoformat()


def member(title: str, desc: str, importance: str = "medium", *,
           reference: str = "## Step 1: Columns", file: Optional[str] = None,
           model: str = "m") -> Dict[str, Any]:
    """One reviewer finding: a plan suggestion, or with *file* a code issue."""
    record: Dict[str, Any] = {"title": title, "desc": desc, "importance": importance,
                              "source_model": model}
    if file is None:
        record.update(type="modification", reference=reference)
    else:
        record.update(type="bug", file=file, line_range=[1, 2], anchor_text="def f():")
    return record


def groups_of(members: Sequence[Dict[str, Any]], *, code: bool = False,
              model: str = "m") -> List[Dict[str, Any]]:
    """One stamped group per member, as ``grouped.json`` holds them."""
    key = "issues" if code else "suggestions"
    groups = [
        {"theme": f"{m['title']} theme", "category": "completeness", "models": [model],
         key: [copy.deepcopy(m)]}
        for m in members
    ]
    stamp_stable_ids(groups)
    return groups


class CascadeWorld:
    """One plan, one cascade dir, and the live dirs a stage leaves."""

    def __init__(self, root: Path, mode: str = "improve-plan", *, name: str = "export",
                 cascade_id: str = CASCADE_ID, of: int = 3,
                 models: Sequence[str] = ("openrouter:cheap", "openrouter:mid",
                                          "openrouter:top")):
        self.root = root
        self.mode = mode
        self.names = ledger.phases(mode)
        self.plan = root / "plans" / f"{name}.md"
        self.plan.parent.mkdir(parents=True, exist_ok=True)
        if not self.plan.exists():
            self.plan.write_text(PLAN_TEXT, encoding="utf-8")
        self.out = output_dir_path(self.plan)
        self.run_dir = self.out / self.names.run_phase
        self.review_dir = self.out / self.names.review_dir
        self.apply_dir = self.out / self.names.apply_dir
        self.cascade_id = cascade_id
        self.of = of
        self.models = list(models)
        self.cascade_dir = self.run_dir / "cascades" / f"20261113T090000-{cascade_id[:8]}"
        self.cascade_dir.mkdir(parents=True, exist_ok=True)
        if mode == "improve-plan":
            (self.cascade_dir / "start.md").write_bytes(self.plan.read_bytes())
        self.write_record()

    # -- the cascade's own files ---------------------------------------------
    def record(self, states: Optional[Mapping[int, str]] = None,
               skip_reasons: Optional[Mapping[int, str]] = None,
               state: str = "running") -> Dict[str, Any]:
        states = dict(states or {})
        return {
            "schema_version": 1, "cascade_id": self.cascade_id, "mode": self.mode,
            "state": state, "started_at": T(0), "scope_policy": None,
            "selectors": {"files_file": None, "base_ref": None},
            "stages": [
                {"index": k, "model": self.models[k - 1], "state": states.get(k, "pending"),
                 "bound_round": None, "scope_policy": None, "started_at": None,
                 "finished_at": None, "skip_reason": (skip_reasons or {}).get(k),
                 "counts": None}
                for k in range(1, self.of + 1)
            ],
            "finished_at": None,
        }

    def write_record(self, **kwargs: Any) -> Dict[str, Any]:
        record = self.record(**kwargs)
        (self.cascade_dir / "record.json").write_text(json.dumps(record, indent=2),
                                                       encoding="utf-8")
        return record

    def freeze(self, stage: int) -> Path:
        return ledger.freeze_inputs(
            cascade_dir=self.cascade_dir, cascade_id=self.cascade_id, mode=self.mode,
            stage=stage, of=self.of, model=self.models[stage - 1], plan_path=self.plan)

    # -- state.json ----------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.out / "state.json"

    def state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"plan_file": str(self.plan), "phases_completed": {}, "phases_skipped": {}}

    def save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # -- a review round ------------------------------------------------------
    def write_review(self, stage: int, groups: List[Dict[str, Any]], *,
                     verdicts: Sequence[Tuple[str, str]] | Sequence[Tuple[str, str, str]],
                     started: float, completed: float, round_id: Optional[str] = None,
                     cost: Optional[float] = 0.04, validation_counts: bool = True,
                     scope_policy: Optional[Dict[str, Any]] = None) -> str:
        """Replace the live review dir with stage *stage*'s round (a fresh
        ``--force --rerun-all`` round retires the old files); its grouped hash.

        *verdicts* is one ``(status, reason[, reraise_of])`` per group.
        """
        model = self.models[stage - 1]
        if self.review_dir.exists():
            shutil.rmtree(self.review_dir)
        self.review_dir.mkdir(parents=True)
        grouped_text = json.dumps({"format_version": 2, "groups": groups}, indent=2)
        (self.review_dir / "grouped.json").write_text(grouped_text, encoding="utf-8")
        records = []
        counts = {"valid": 0, "invalid": 0, "needs_human_decision": 0, "validation_failed": 0}
        for index, (group, verdict) in enumerate(zip(groups, verdicts)):
            status, reason = verdict[0], verdict[1]
            record = {"group_index": index, "group_id": group["group_hash"],
                      "status": status, "reason": reason, "confidence": 0.9}
            if len(verdict) > 2 and verdict[2]:
                record["reraise_of"] = verdict[2]
            records.append(record)
            counts[{"needs-human-decision": "needs_human_decision"}.get(status, status)] += 1
        (self.review_dir / "validation.json").write_text(json.dumps({
            "format_version": 2, "groups": records,
            "metadata": {"model": "host", "grouped_sha256": content_hash(grouped_text)},
        }, indent=2), encoding="utf-8")
        status = {
            "phase": self.names.review_phase, "state": "complete",
            "started_at": T(started), "round_id": round_id or f"{stage:016x}",
            "models_requested": [model], "models_completed": [model], "models_failed": {},
        }
        if validation_counts:
            status["validation_counts"] = counts
        if cost is not None:
            status["usage"] = {model: {"cost_usd": cost, "output_tokens": 1200}}
        (self.review_dir / ".status.json").write_text(json.dumps(status, indent=2),
                                                      encoding="utf-8")
        (self.review_dir / "report.md").write_text(f"# Review stage {stage}\n", encoding="utf-8")
        result = self.review_dir / f"{model.replace(':', '_')}.json"
        result.write_text(json.dumps([m for g in groups for m in ledger._members(g)]),
                          encoding="utf-8")
        prompts = self.review_dir / "prompts"
        prompts.mkdir()
        (prompts / "reviewer.txt").write_text("prompt\n", encoding="utf-8")
        (self.review_dir / ".lock").write_text("", encoding="utf-8")
        (self.review_dir / ".status.lock").write_text("", encoding="utf-8")
        state = self.state()
        state.setdefault("phases_completed", {})[self.names.review_phase] = T(completed)
        state.get("phases_completed", {}).pop(self.names.apply_phase, None)
        table = state.setdefault("scope_policy", {})
        if scope_policy is not None:
            table[self.names.review_phase] = scope_policy
        else:
            table[self.names.review_phase] = {
                "review_policy": "expand", "review_policy_source": "default",
                "recorded_at": T(started)}
        self.save_state(state)
        return content_hash(grouped_text)

    # -- the apply -------------------------------------------------------------
    def write_apply(self, stage: int, grouped_sha: str, *,
                    batches: Sequence[Tuple[str, Sequence[str], Sequence[str]]],
                    outcomes: Mapping[str, Tuple[str, str]],
                    decisions: Optional[Mapping[str, Tuple[str, str, str]]] = None,
                    at: float, completed: Optional[float] = None,
                    report_rows: Optional[Sequence[Dict[str, Any]]] = None,
                    native: bool = True, stamped: bool = True,
                    plan_edit: Optional[str] = None) -> None:
        """Stage *stage*'s apply: its batch files and its ``state.json`` records.

        *batches* are ``(task_id, expected ids, declared files)``; *outcomes*
        ``{item: (status, reason)}`` (``applied``, ``no-op``, ``declined``,
        ``skipped``); *decisions* ``{item: (decision, decision_source,
        reason)}``, recorded at *at*. With *batches* empty the apply handed
        nothing out: ``orchestrator_output.json`` holds no batch and an earlier
        stage's ``write_tasks.json``, if any, is left as it was.
        """
        self.apply_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            {"task_id": task_id, "expected_ids": list(items), "declared_files": list(files),
             "inputs": {"grouped_sha256": grouped_sha} if stamped else {},
             "kind": "agent-write"}
            for task_id, items, files in batches
        ]
        if tasks:
            document: Dict[str, Any] = {"schema_version": 1, "batches": tasks}
            if native:
                document["leg"] = "native"
                document["native_run"] = {
                    "started_at": T(at - 1), "emitted_at": T(at - 1),
                    "items": [i for _t, items, _f in batches for i in items]}
            (self.apply_dir / "write_tasks.json").write_text(json.dumps(document, indent=2),
                                                             encoding="utf-8")
        (self.apply_dir / "orchestrator_output.json").write_text(json.dumps({
            "status": "completed", "batches": [{"task_id": t["task_id"]} for t in tasks],
            "to_apply": [], "skipped_items": [], "skipped_count": 0,
        }, indent=2), encoding="utf-8")
        (self.apply_dir / "summary.md").write_text(f"# Apply stage {stage}\n", encoding="utf-8")
        (self.apply_dir / ".lock").write_text("", encoding="utf-8")
        if report_rows is not None:
            (self.apply_dir / "write_batch_report.json").write_text(json.dumps({
                "written_at": T(at), "tasks": list(report_rows)}, indent=2), encoding="utf-8")
        holder = {i: t for t, items, _f in batches for i in items}
        state = self.state()
        processed = state.setdefault(f"processed_{self.names.apply_dir}", {})
        for item, (status, reason) in outcomes.items():
            details: Dict[str, Any] = {"task_id": holder.get(item, ""), "prior_changes_id": "p",
                                       "grouped_sha256": grouped_sha if stamped else "",
                                       "leg": "native" if native else "executor"}
            if reason:
                details["reason"] = reason
            processed[item] = {"status": status, "details": details, "timestamp": T(at)}
        table = state.setdefault(f"human_decisions_{self.names.apply_dir}", {})
        for item, (decision, source, reason) in (decisions or {}).items():
            table[item] = {"decision": decision, "reason": reason, "timestamp": T(at - 0.5),
                           "batch_context": {"decision_source": source}}
        state.setdefault("phases_completed", {})[self.names.apply_phase] = T(
            completed if completed is not None else at + 1)
        self.save_state(state)
        if plan_edit is not None:
            self.plan.write_text(self.plan.read_text(encoding="utf-8") + plan_edit,
                                 encoding="utf-8")


def ids(groups: Iterable[Mapping[str, Any]]) -> List[str]:
    return [g["group_hash"] for g in groups]


def tree(directory: Path) -> Dict[str, bytes]:
    """``{relative posix path: bytes}`` of every file below *directory*."""
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in sorted(directory.rglob("*")) if p.is_file()
    }
