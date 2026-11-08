"""Workflow states for the next-phase builder: a plan's ``state.json`` and phase files, built from a spec.

``S(...)`` is a state spec (phases completed or skipped, task statuses, phase
files), ``_build(tmp_path, spec)`` writes it under a ``plans/foo.md`` plan with
the real ``StateManager``, and the ``RP_*`` / ``RT_*`` / ``CR_*`` maps are the
``validation.json`` / ``grouped.json`` files the workflow view reads.
``ROUTES`` are the six host/route rows of the next-phase command table, with
the literal text each prints before the arguments. No fixtures.
"""

import json
from pathlib import Path

from utils import next_phase as np_
from utils.next_phase import NextPhase, OptionalStep
from utils.output_handler import output_dir_path
from utils.state_manager import StateManager


#: The six host/route rows of the §2.1 table, with the literal text before
#: ``{args}``.
ROUTES = [
    ("claude-code", None, "/multi-llm:multi-llm "),
    ("claude-code", "skill", "/multi-llm "),
    ("codex", None, "Use the multi-llm skill with exactly these arguments: "),
    ("codex", "package", "$multi-llm:multi-llm "),
    ("opencode", None, "/multi-llm "),
    ("opencode", "package", "/multi-llm "),
]
ROUTE_IDS = [f"{host}.{route or 'default'}" for host, route, _ in ROUTES]

P = "plans/foo.md"

#: Fix round 22: no "optional:" of its own (the host labels the step).
WHY_REVIEW_TASKS = (
    "Have the models review the task list before implementing it; skipping "
    "this implements the tasks as they were generated."
)


NOTICE_ONE_GROUP = (
    "review-tasks left 1 valid task suggestion group unapplied. Run "
    "--apply-task-suggestions first to apply it to the task list; going "
    "straight to --implement skips it, and implement then works from the "
    "task list as it stands, without that change."
)


VALIDATION_VALID = {"format_version": 2, "groups": [
    {"group_index": 0, "status": "valid"},
]}


ONE_GROUP = {"format_version": 2, "groups": [
    {"group_id": "g1", "theme": "t",
     "suggestions": [{"title": "s", "importance": "HIGH"}]},
]}
NO_FINDINGS = {"format_version": 2, "groups": [{"group_id": "g1", "suggestions": []}]}

RP_VALID = {("review-plan", "validation.json"): VALIDATION_VALID}


RT_ACTIONABLE = {
    ("review-tasks", "grouped.json"): ONE_GROUP,
    ("review-tasks", "validation.json"): VALIDATION_VALID,
}
RT_CLEAN = {("review-tasks", "grouped.json"): NO_FINDINGS}
CR_FINDING = {("code-review", "grouped.json"): ONE_GROUP}
CR_CLEAN = {("code-review", "grouped.json"): NO_FINDINGS}

UP_TO_TASKS = ("review-plan", "apply-suggestions", "generate-tasks")
UP_TO_REVIEW_TASKS = UP_TO_TASKS + ("review-tasks",)
UP_TO_IMPLEMENT = UP_TO_TASKS + ("implement",)
UP_TO_CODE_REVIEW = UP_TO_IMPLEMENT + ("code-review",)
ALL_REQUIRED = UP_TO_CODE_REVIEW + ("apply-code-fixes",)


def S(completed=(), skipped=(), tasks=None, files=None):
    """A state spec: phases recorded completed / skipped, task statuses, files."""
    return {
        "completed": tuple(completed),
        "skipped": tuple(skipped),
        "tasks": dict(tasks or {}),
        "files": dict(files or {}),
    }


def _merge(*file_maps):
    merged = {}
    for file_map in file_maps:
        merged.update(file_map)
    return merged


def _build(tmp_path: Path, spec) -> Path:
    plan = tmp_path / "plans" / "foo.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# foo\n", encoding="utf-8")
    for (phase_dir, name), content in spec["files"].items():
        target = output_dir_path(plan) / phase_dir
        target.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else json.dumps(content)
        (target / name).write_text(text, encoding="utf-8")
    state = StateManager(plan)
    for phase in spec["completed"]:
        state.mark_phase_completed(phase)
    for phase in spec["skipped"]:
        state.mark_phase_skipped(phase, f"test skip of {phase}")
    for task_id, status in spec["tasks"].items():
        state.update_task_status(task_id, status)
    state.save()
    return plan


def with_markdown(payload):
    """*payload* (a ``user_next_step`` dict) with the ``markdown`` key the
    builder adds last (fix round 22), rendered from the payload's own fields
    by the one renderer; ``tests/test_next_step_markdown.py`` pins the text."""
    answer = NextPhase(
        phase=payload.get("phase"), command=payload.get("command"),
        optional_before=tuple(
            OptionalStep(step["phase"], step["command"], step["why"])
            for step in payload.get("optional_before") or ()),
        complete=bool(payload.get("complete")),
        manual_tasks=tuple(
            np_.ManualTask(task["id"], task.get("title"), task.get("waits_on"))
            for task in payload.get("manual_tasks") or ()),
        answers=tuple(
            np_.AnswersStep(step["command"], step["why"], tuple(step.get("answers") or ()))
            for step in payload.get("answers") or ()),
        manual_tasks_note=payload.get("manual_tasks_note"),
        manual_tasks_command=payload.get("manual_tasks_command"),
        why=payload.get("why"),
    )
    return dict(payload, markdown=np_.render_next_step_markdown(answer))
