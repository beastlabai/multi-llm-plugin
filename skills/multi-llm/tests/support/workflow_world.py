"""Workflow state on disk for ``check_workflow_prerequisites``.

``_workflow_plan(tmp_path)`` writes a plan, ``_record`` marks phases and task
statuses in its ``state.json``, ``_put`` writes a phase file, ``_status`` reads
the workflow status, and ``_date_cycle`` rewrites the times one apply phase and
its review carry. No fixtures.
"""

import json
from pathlib import Path

from utils.state_manager import StateManager
import check_workflow_prerequisites as prereq
from utils.output_handler import output_dir_path


#: Every phase before code-review, i.e. "every earlier phase on record" for the
#: apply-code-fixes rows (the two optional phases never block suggested_next).
_BEFORE_CODE_REVIEW = (
    "review-plan", "apply-suggestions", "generate-tasks", "implement",
)

_ONE_FINDING = {"format_version": 2, "groups": [
    {"group_id": "g1", "theme": "t",
     "suggestions": [{"title": "fix it", "importance": "HIGH"}]},
]}


def _workflow_plan(tmp_path: Path, name: str = "t1-plan") -> Path:
    plan = tmp_path / f"{name}.md"
    plan.write_text("# T1 plan\n", encoding="utf-8")
    return plan


def _record(plan: Path, completed=(), skipped=(), tasks=None) -> None:
    state = StateManager(plan)
    for phase in completed:
        state.mark_phase_completed(phase)
    for phase in skipped:
        state.mark_phase_skipped(phase, f"test skip of {phase}")
    for task_id, status in (tasks or {}).items():
        state.update_task_status(task_id, status)
    state.save()


def _put(plan: Path, phase_dir: str, name: str, content) -> Path:
    target = output_dir_path(plan) / phase_dir
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return path


def _status(plan: Path) -> dict:
    return prereq.get_workflow_status(
        StateManager(plan), plan, output_dir_path(plan)
    )


def _review_tasks_actionable(plan):
    _put(plan, "review-tasks", "grouped.json", {"format_version": 2, "groups": [
        {"group_id": "g1", "theme": "t",
         "suggestions": [{"title": "s", "importance": "LOW"}]},
    ]})
    _put(plan, "review-tasks", "validation.json", {"format_version": 2, "groups": [
        {"group_index": 0, "status": "valid"},
    ]})


#: (phase, its own batch id, fixture that leaves the phase pending, the
#: completed phases that fixture records)
APPLY_PHASE_BATCH_ROWS = [
    ("apply-suggestions", "apply-batch-0",
     lambda plan: _put(plan, "review-plan", "validation.json", {
         "format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]}),
     ("review-plan",)),
    ("apply-task-suggestions", "apply-task-batch-0",
     _review_tasks_actionable,
     ("review-plan", "apply-suggestions", "generate-tasks", "review-tasks")),
    ("apply-code-fixes", "apply-fix-batch-0",
     lambda plan: _put(plan, "code-review", "grouped.json", _ONE_FINDING),
     _BEFORE_CODE_REVIEW + ("code-review",)),
]


def _date_cycle(plan, phase, review, reviewed_at, *, started_at=None, metric_at=None):
    """Rewrite the times `state.json` holds for one apply phase and its review.

    `reviewed_at` stamps the review's completion; `started_at` is the phase's
    `metrics_progress.{phase}.started_at`, `metric_at` one `metrics.{phase}`
    entry's `timestamp`. None leaves that record out.
    """
    state = StateManager(plan)
    state.state["phases_completed"][review] = reviewed_at
    # These cycles are about a review re-run, not about implement finishing
    # after a review: keep implement's completion (stamped "now" by `_record`)
    # older than every date the cycle uses, or a code review stamped _OLD/_NEW
    # would read as stale for that reason instead.
    if "implement" in state.state["phases_completed"]:
        state.state["phases_completed"]["implement"] = "2025-12-31T10:00:00"
    if started_at is not None:
        state.state.setdefault("metrics_progress", {})[phase] = {
            "started_at": started_at}
    if metric_at is not None:
        state.state.setdefault("metrics", {})[phase] = [{"timestamp": metric_at}]
    state.save()


_OLD, _NEW = "2026-01-01T10:00:00", "2026-01-02T10:00:00"
