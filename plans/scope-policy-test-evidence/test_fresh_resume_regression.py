"""Failing integration regression; invokes implement.main with no model spend.

Run with PYTHONPATH pointing at the staged skills/multi-llm directory and
pytest -p tests.conftest so the repository's isolation fixtures apply.
"""
from tests.test_scope_policy_leaf_modes import (
    repo, handed_out, _implement, _prompts, PRESERVE_HEADING,
)


def test_fresh_default_expand_survives_implement_resume(repo, handed_out):
    assert _implement(repo, '--scope-policy', 'preserve') == 0
    assert _implement(repo, '--fresh', '--yes') == 0
    assert all(PRESERVE_HEADING not in p for p in _prompts(handed_out[-1]))
    assert _implement(repo, '--resume', '--yes') == 0
    assert all(PRESERVE_HEADING not in p for p in _prompts(handed_out[-1])), (
        'The continuation of the fresh expand run restored the older preserve policy'
    )
