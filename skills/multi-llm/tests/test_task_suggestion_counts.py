"""Task suggestion notices keep group units even without valid verdicts."""

import json

import pytest

from check_workflow_prerequisites import check_apply_task_suggestions_prerequisite
from utils.output_handler import get_output_dir
from utils.state_manager import StateManager


@pytest.mark.parametrize("validation", [None, {}, {"groups": [{"status": "invalid"}]}])
def test_unvalidated_task_notice_counts_nonempty_groups(tmp_path, validation):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n")
    state = StateManager(plan)
    out = get_output_dir(plan)
    review = out / "review-tasks"
    review.mkdir(parents=True)
    (review / "grouped.json").write_text(json.dumps({"groups": [
        {"suggestions": [{"title": "a"}, {"title": "a duplicate"}]},
        {"suggestions": [{"title": "b"}]},
        {"suggestions": []},
    ]}))
    if validation is not None:
        (review / "validation.json").write_text(json.dumps(validation))
    result = check_apply_task_suggestions_prerequisite(state, out)
    assert result["actionable_count"] == 2
    assert "2 unapplied task suggestion groups" in result["reason"]
