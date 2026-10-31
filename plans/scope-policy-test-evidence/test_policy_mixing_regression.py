"""Failing regression: cached reviewers bind the round, even if deselected."""
from tests.test_scope_prompts_and_validation import _review_plan_main


def test_different_reviewer_cannot_change_an_unfinished_round_policy(tmp_path):
    first = _review_plan_main(tmp_path, 'preserve', models=('cursor-agent:auto',))
    assert first['code'] == 0
    second = _review_plan_main(tmp_path, 'expand', models=('cursor-agent:gpt-5',))
    assert second['code'] == 3, (
        'An unfinished preserve round must refuse an expand reviewer before dispatch'
    )
    assert second['sent'] == []
