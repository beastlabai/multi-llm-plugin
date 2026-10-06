"""The optional apply-task-suggestions step says what skipping it means (fix round 18).

Live run 7, #11, every host's phase 04: review-tasks left 11 (Claude Code),
9 (Codex) and 7 (OpenCode) valid suggestion groups unapplied, and the next
step read `--implement`, with `--apply-task-suggestions` only as
`optional_before`. The design stays: an optional phase never becomes the
primary, and plans/next-phase-command.md §2.1 ("Primary first") puts the
primary first in every rendering. What changed is the optional entry's `why`,
the workflow notice: it names how many valid groups are waiting and what going
straight to implement does with them.
"""

from __future__ import annotations

import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

import check_workflow_prerequisites as prereq  # noqa: E402
from utils import next_phase  # noqa: E402

from tests.test_workflow_prerequisites import (  # noqa: E402
    _put,
    _record,
    _status,
    _workflow_plan,
)


def _review_tasks_left(plan, groups: int, *, validated: bool) -> None:
    _put(plan, "review-tasks", "grouped.json", {"format_version": 2, "groups": [
        {"group_id": f"g{i}", "theme": "t",
         "suggestions": [{"title": f"s{i}", "importance": "LOW"}]}
        for i in range(groups)
    ]})
    if validated:
        _put(plan, "review-tasks", "validation.json", {"format_version": 2, "groups": [
            {"group_index": i, "status": "valid"} for i in range(groups)
        ]})
    _record(plan, completed=("review-plan", "apply-suggestions", "generate-tasks",
                             "review-tasks"))


class TestTheOptionalStepSaysWhatSkippingMeans:

    def test_it_names_the_valid_groups_waiting_and_what_implement_does(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _review_tasks_left(plan, 11, validated=True)
        why = _status(plan)["phases"]["apply-task-suggestions"]["notice"]
        assert why.startswith(
            "review-tasks left 11 valid task suggestion groups unapplied."), why
        assert "--apply-task-suggestions first" in why
        assert ("going straight to --implement skips them, and implement then "
                "works from the task list as it stands, without those 11 "
                "changes") in why

    def test_unvalidated_groups_are_not_called_valid(self, tmp_path):
        plan = _workflow_plan(tmp_path)
        _review_tasks_left(plan, 3, validated=False)
        why = _status(plan)["phases"]["apply-task-suggestions"]["notice"]
        assert why.startswith("review-tasks left 3 task suggestion groups unapplied.")

    def test_one_group_reads_in_the_singular(self):
        why = prereq.unapplied_task_groups_notice(1, valid=True)
        assert "left 1 valid task suggestion group unapplied" in why
        assert "skips it" in why and "without that change" in why

    def test_the_next_step_carries_it_as_the_optional_why_after_the_primary(
        self, tmp_path,
    ):
        """The primary stays implement (an optional phase is never the
        primary), the block prints it first, and the optional entry's why
        is the notice."""
        plan = _workflow_plan(tmp_path)
        _review_tasks_left(plan, 9, validated=True)
        answer = next_phase.next_phase_commands(
            "plans/x.md", "claude-code", None, None, after_mode="review-tasks",
            plan_path=plan)
        assert answer.phase == "implement"
        (step,) = answer.optional_before
        assert step.phase == "apply-task-suggestions"
        assert "left 9 valid task suggestion groups unapplied" in step.why
        lines = next_phase.render_next_phase_lines(answer)
        assert lines[0].startswith(next_phase.NEXT_COMMAND_MARKER + " ")
        assert lines[1] == f"{next_phase.NEXT_COMMAND_OPTIONAL_WHY_MARKER} {step.why}"
