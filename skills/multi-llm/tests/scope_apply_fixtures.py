"""Shared fixtures for the scope-policy apply tests (plans/scope-policy-and-report-tags.md §5).

One review cycle on disk per phase — ``grouped.json``, a ``validation.json``
whose records carry host-stamped scope verdicts (built with
``scope_policy.stamp_validation_record``, the way the review side stamps them),
the review round's ``state.json`` record, user choices and judge verdict files
— so the apply orchestrators can be driven in-process on both legs with
identical recorded model results and user choices.

Nothing here calls a model: the native leg needs no runner, and the executor
leg runs the ``tests/mocks/mock_llm.py`` runner the ``repo`` fixture installs.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils import scope_policy as sp  # noqa: E402
from utils.state_manager import get_or_create_state, stamp_stable_ids  # noqa: E402

PLAN = """# Feature plan

## Goal
Export the weekly report as CSV. Do not change the public API.

## Step 1: Alpha
Alpha text.

## Step 2: Beta
Beta text.

## Step 3: Gamma
Gamma text.

## Step 4: Delta
Delta text.
"""

BASELINE = sp.build_baseline(
    [sp.baseline_source("plan", sha256="ab" * 32, path="plans/feat.md")],
    phase="review-plan",
)
BID = BASELINE["baseline_id"]

LONG = (
    "Complete the CSV header row so every column the plan names is present, "
    "in the order the plan lists them, with the exact names it gives, and "
    "keep the delimiter a comma as the plan requires for the weekly export."
)
assert len(LONG) > 200


def _member(title, desc, step, section, importance="MEDIUM", type_="addition"):
    return {
        "title": title, "desc": desc, "type": type_,
        "reference": f"## Step {step}: {section}", "importance": importance,
        "source_model": "m",
    }


#: ``key -> (members, validation status, verdict spec)``. The verdict spec is
#: one of ``("bound", relation)``, ``("legacy",)`` (no scope fields, no stamp),
#: ``("malformed",)``, ``("stale", relation)`` (stamped on another proposal),
#: ``("remedy", relation, remedy)`` (a preserve validator's selected remedy).
PLAN_CASES: Dict[str, Tuple[List[Dict[str, Any]], str, Tuple[Any, ...]]] = {
    "in_scope": (
        [_member("Complete the CSV columns", "Add the missing 'total' column.", 1, "Alpha")],
        "valid", ("bound", "in_scope")),
    "extension": (
        [_member("Also export PDF", "Add a PDF export beside the CSV.", 2, "Beta",
                 importance="LOW", type_="improvement")],
        "valid", ("bound", "scope_extension")),
    "unclear": (
        [_member("Clarify the week boundary", "State whether weeks start Monday.",
                 3, "Gamma")],
        "valid", ("bound", "unclear")),
    "legacy": (
        [_member("Handle empty weeks", "Write a header-only file for empty weeks.",
                 4, "Delta", importance="HIGH")],
        "valid", ("legacy",)),
    "malformed": (
        [_member("Tidy wording", "Reword the Alpha paragraph.", 1, "Alpha",
                 importance="LOW", type_="clarification")],
        "valid", ("malformed",)),
    "stale": (
        [_member("Quote fields", "Quote fields that contain commas.", 2, "Beta")],
        "valid", ("stale", "in_scope")),
    "invalid": (
        [_member("Rename the module", "Rename report.py to export.py.", 3, "Gamma")],
        "invalid", ("bound", "in_scope")),
    "user_skipped": (
        [_member("Sort rows", "Sort rows by date.", 4, "Delta")],
        "valid", ("bound", "in_scope")),
    "low_in_scope": (
        [_member("Fix a typo", "Fix the 'colum' typo in Step 3.", 3, "Gamma",
                 importance="LOW", type_="clarification")],
        "valid", ("bound", "in_scope")),
    "edited": (
        [_member("Complete the header", LONG, 1, "Alpha")],
        "valid", ("bound", "in_scope")),
    "member_skip": (
        [_member("Escape quotes", "Escape embedded quotes by doubling them.", 2, "Beta"),
         _member("Escape quotes (and add XLSX)", "Escape quotes, and add an XLSX export.",
                 2, "Beta")],
        "valid", ("bound", "in_scope")),
    "remedy": (
        [_member("Encoding", "Write UTF-8 output.", 3, "Gamma"),
         _member("Encoding (and a settings page)",
                 "Write UTF-8 output and add a settings page to pick the encoding.",
                 3, "Gamma")],
        "valid", ("remedy", "in_scope", "Write the CSV as UTF-8; nothing else.")),
    "judge_approve": (
        [_member("Document the delimiter", "Say the delimiter is a comma.", 4, "Delta")],
        "needs-human-decision", ("bound", "in_scope")),
    "judge_salvage": (
        [_member("Column order", "Fix the column order and add a column picker UI.",
                 1, "Alpha")],
        "needs-human-decision", ("bound", "in_scope")),
    "judge_skip": (
        [_member("Add charts", "Add charts to the report.", 2, "Beta")],
        "needs-human-decision", ("bound", "scope_extension")),
    "judge_extension": (
        [_member("Add an API endpoint", "Expose the export over HTTP.", 4, "Delta")],
        "needs-human-decision", ("bound", "scope_extension")),
    # A reviewer that mislabelled an expanding remedy `in_scope`; the
    # validation verdict is independently `scope_extension`.
    "leaked": (
        [dict(_member("Add caching", "Cache every export in Redis.", 3, "Gamma"),
              scope_relation="in_scope", scope_reason="required for speed")],
        "valid", ("bound", "scope_extension")),
}


def build_groups(cases: Dict[str, Tuple[Any, ...]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """The ``grouped.json`` groups (stamped) and ``key -> group_hash``."""
    groups = [
        {"theme": f"{key} theme", "category": "completeness", "models": ["m"],
         "suggestions": copy.deepcopy(members)}
        for key, (members, _status, _spec) in cases.items()
    ]
    stamp_stable_ids(groups)
    return groups, {key: g["group_hash"] for key, g in zip(cases, groups)}


def build_validation(
    groups: Sequence[Dict[str, Any]], cases: Dict[str, Tuple[Any, ...]],
    *, baseline_id: Optional[str] = BID, preserve_review: bool = True,
) -> Dict[str, Any]:
    """The v2 ``validation.json`` envelope with host-stamped scope verdicts."""
    records = []
    for index, (group, (key, (_members, status, spec))) in enumerate(
            zip(groups, cases.items())):
        record: Dict[str, Any] = {
            "group_id": group["group_hash"], "group_index": index,
            "status": status, "reason": f"{key} reason", "confidence": 0.9,
        }
        kind = spec[0]
        if kind == "legacy":
            records.append(record)
            continue
        if kind == "malformed":
            record.update(scope_relation={"bad": 1}, scope_reason=7)
            records.append(sp.stamp_validation_record(record, group, baseline_id=baseline_id))
            continue
        relation = spec[1]
        record.update(scope_relation=relation, scope_reason=f"{key}: {relation}",
                      scope_reference="Plan: Goal")
        if kind == "stale":
            shown = copy.deepcopy(group)
            shown["suggestions"][0]["desc"] = "A different proposal the validator saw."
            records.append(sp.stamp_validation_record(record, shown, baseline_id=baseline_id))
            continue
        if kind == "remedy":
            record["selected_remedy"] = spec[2]
        records.append(sp.stamp_validation_record(
            record, group, baseline_id=baseline_id,
            allow_selected_remedy=preserve_review))
    return {
        "metadata": {"schema_version": "2.0", "validated_at": "2026-10-31T10:00:00",
                     "model": "m", "plan_hash": "x", "total_groups": len(records)},
        "groups": records,
    }


def write_review(
    root: Path, *, cases=None, review_policy: Optional[str] = "preserve",
    selections: bool = True, verdicts: bool = True, plan_name: str = "feat",
) -> Dict[str, Any]:
    """A completed review-plan cycle for ``plans/<plan_name>.md``; its ids."""
    cases = PLAN_CASES if cases is None else cases
    plan = root / "plans" / f"{plan_name}.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    if not plan.exists():
        plan.write_bytes(PLAN.encode("utf-8"))
    review = root / "plans" / plan_name / "review-plan"
    review.mkdir(parents=True, exist_ok=True)
    groups, ids = build_groups(cases)
    (review / "grouped.json").write_text(
        json.dumps({"format_version": 2, "groups": groups}, indent=2), encoding="utf-8")
    (review / "validation.json").write_text(
        json.dumps(build_validation(groups, cases), indent=2), encoding="utf-8")
    state = get_or_create_state(plan)
    state.mark_phase_completed("review-plan")
    if review_policy is not None:
        sp.record_review_round(state.state, "review-plan", review_policy, "explicit", BASELINE)
    state.save()
    by_key = dict(zip(cases, groups))
    if selections:
        write_selections(review, plan, by_key, cases)
    if verdicts:
        write_verdicts(root / "plans" / plan_name / "apply-suggestions", ids, cases)
    return {"plan": plan, "review": review, "ids": ids, "groups": by_key}


def write_selections(review: Path, plan: Path, by_key, cases) -> None:
    """The user's choices: one group skipped, one member skipped, one edit
    appended past the 200-char id-hash prefix."""
    skipped_groups = [by_key["user_skipped"]["group_hash"]] if "user_skipped" in by_key else []
    skipped_suggestions = []
    if "member_skip" in by_key:
        skipped_suggestions.append(by_key["member_skip"]["suggestions"][1]["suggestion_hash"])
    edited = {}
    if "edited" in by_key:
        member = by_key["edited"]["suggestions"][0]
        edited[member["suggestion_hash"]] = LONG + " Also add a PDF export of the same table."
    (review / "user_selections.json").write_text(json.dumps({
        "format_version": 2, "plan_path": str(plan), "phase": "review-plan",
        "exported_at": "2026-10-31T10:00:00",
        "skipped_groups": skipped_groups, "skipped_suggestions": skipped_suggestions,
        "edited_descriptions": edited,
    }, indent=2), encoding="utf-8")


def write_verdicts(phase_dir: Path, ids: Dict[str, str], cases) -> None:
    """The judge's verdict files, already on disk (the native leg's shape)."""
    phase_dir.mkdir(parents=True, exist_ok=True)
    verdicts = []
    if "judge_approve" in cases:
        verdicts.append({"group_id": ids["judge_approve"], "decision": "approve",
                         "reason": "a real gap"})
    if "judge_salvage" in cases:
        verdicts.append({"group_id": ids["judge_salvage"], "decision": "salvage",
                         "reason": "keep the order fix only",
                         "salvaged_description": "Fix the column order.",
                         "dropped": "the column picker UI"})
    if "judge_skip" in cases:
        verdicts.append({"group_id": ids["judge_skip"], "decision": "skip",
                         "reason": "not needed"})
    if "judge_extension" in cases:
        verdicts.append({"group_id": ids["judge_extension"], "decision": "approve",
                         "reason": "useful"})
    (phase_dir / "human_decision_batch_0.json").write_text(json.dumps(
        {"schema_version": "1.0", "verdicts": verdicts}, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driving the orchestrators in-process
# ---------------------------------------------------------------------------


def run_apply(module, plan: Path, *flags: str, scope_policy: Optional[str] = None) -> int:
    """Parse *flags*, inject ``scope_policy`` on the namespace, run; the exit code.

    The parser flag is agent B's; the namespace attribute is the contract
    (``getattr(args, "scope_policy", None)``), so tests set it there.
    """
    from utils import provider_registry as registry

    # One process runs one plan (`set_run_anchor`); each test run is a new one.
    registry._RUN_ANCHOR = None
    orchestrator_class = _orchestrator_class(module)
    args = orchestrator_class.parse_args(["--plan-file", str(plan), *flags])
    if scope_policy is not None:
        args.scope_policy = scope_policy
    try:
        return orchestrator_class(args).run()
    except SystemExit as exc:  # early exits (confirmation, no items)
        return int(exc.code or 0)


def _orchestrator_class(module):
    for name in ("ApplySuggestionsOrchestrator", "ApplyTaskSuggestionsOrchestrator",
                 "ApplyCodeFixesOrchestrator"):
        if hasattr(module, name):
            return getattr(module, name)
    raise AssertionError(f"no orchestrator class in {module}")


_VOLATILE_KEYS = {
    "timestamp", "emitted_at", "started_at", "recorded_at", "exported_at",
    "validated_at", "decided_at", "apply_recorded_at", "taken_at",
}


def normalize(value: Any, roots: Sequence[str]) -> Any:
    """*value* with temp roots replaced and timestamps dropped (for goldens)."""
    if isinstance(value, dict):
        return {k: normalize(v, roots) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [normalize(v, roots) for v in value]
    if isinstance(value, str):
        for root in roots:
            value = value.replace(root, "<ROOT>")
        value = re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?", "<TIME>", value)
        return value
    return value


def native_snapshot(plan: Path, phase: str, roots: Sequence[str]) -> Dict[str, Any]:
    """What the native leg handed out: decisions, batches, tasks, bound prompts."""
    from utils import write_prompt

    phase_dir = plan.parent / plan.stem / phase
    output = json.loads((phase_dir / "orchestrator_output.json").read_text(encoding="utf-8"))
    snapshot: Dict[str, Any] = {
        "to_apply": [
            {k: item.get(k) for k in ("group_id", "title", "importance", "description",
                                       "validation_status")}
            for item in output.get("to_apply") or []
        ],
        "needs_human_review": [item.get("group_id") for item in output.get("needs_human_review") or []],
        "skipped_items": output.get("skipped_items"),
        "user_skipped_items": output.get("user_skipped_items"),
        "summary": output.get("summary"),
        "batch_prompts": [batch.get("prompt") for batch in output.get("batches") or []],
    }
    tasks_path = phase_dir / "write_tasks.json"
    tasks = []
    prompts = {}
    if tasks_path.exists():
        document = json.loads(tasks_path.read_text(encoding="utf-8"))
        for task in document.get("batches") or []:
            tasks.append({k: task.get(k) for k in (
                "task_id", "expected_ids", "prior_changes_id", "prompt", "batch_index")})
            text, _task, _info = write_prompt.render_write_prompt(phase_dir, task["task_id"])
            prompts[task["task_id"]] = text
    snapshot["tasks"] = tasks
    snapshot["write_prompts"] = prompts
    return normalize(snapshot, roots)


def judge_record(plan: Path, phase: str, roots) -> Dict[str, Any]:
    """The judge decisions ``state.json`` recorded for *phase*."""
    state = json.loads((plan.parent / plan.stem / "state.json").read_text(encoding="utf-8"))
    decisions = state.get(f"human_decisions_{phase}") or {}
    return normalize(decisions, roots)


# ---------------------------------------------------------------------------
# apply-code-fixes: a code-review cycle
# ---------------------------------------------------------------------------


def _issue(title, desc, *, importance="MEDIUM", type_="bug", line=1):
    return {
        "title": title, "desc": desc, "type": type_, "file": "src/main.py",
        "line_range": [line, line], "anchor_text": "x = 1",
        "importance": importance, "source_model": "m",
    }


CODE_CASES: Dict[str, Tuple[List[Dict[str, Any]], str, Tuple[Any, ...]]] = {
    "in_scope": ([_issue("Off-by-one in total", "Use range(n + 1) for the total.")],
                 "valid", ("bound", "in_scope")),
    "extension": ([_issue("Add retries", "Wrap the export in a retry loop with backoff.",
                          type_="improvement", importance="LOW")],
                  "valid", ("bound", "scope_extension")),
    "legacy": ([_issue("Null check", "Guard against a None week.", importance="HIGH")],
               "valid", ("legacy",)),
    # Two members whose descriptions format_fix_for_output concatenates.
    "remedy": ([_issue("Encoding bug", "Open the file with encoding='utf-8'.", line=2),
                _issue("Encoding bug (and config)",
                       "Open the file with utf-8 and add an encoding setting to the config.",
                       line=2)],
               "valid", ("remedy", "in_scope", "Pass encoding='utf-8' to open(); add no setting.")),
}

CODE_BASELINE = sp.build_baseline(
    [sp.baseline_source("plan", sha256="cd" * 32, path="plans/feat.md")],
    phase="code-review",
)
CODE_BID = CODE_BASELINE["baseline_id"]


def write_code_review(root: Path, *, cases=None, review_policy: Optional[str] = "preserve",
                      plan_name: str = "feat") -> Dict[str, Any]:
    """A completed code-review cycle for ``plans/<plan_name>.md``; its ids."""
    cases = CODE_CASES if cases is None else cases
    plan = root / "plans" / f"{plan_name}.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    if not plan.exists():
        plan.write_bytes(PLAN.encode("utf-8"))
    review = root / "plans" / plan_name / "code-review"
    review.mkdir(parents=True, exist_ok=True)
    groups, ids = build_groups(cases)
    (review / "grouped.json").write_text(
        json.dumps({"format_version": 2, "groups": groups}, indent=2), encoding="utf-8")
    (review / "validation.json").write_text(json.dumps(
        build_validation(groups, cases, baseline_id=CODE_BID), indent=2), encoding="utf-8")
    state = get_or_create_state(plan)
    state.mark_phase_completed("code-review")
    if review_policy is not None:
        sp.record_review_round(state.state, "code-review", review_policy, "explicit",
                               CODE_BASELINE)
    state.save()
    return {"plan": plan, "review": review, "ids": ids,
            "groups": dict(zip(cases, groups))}
