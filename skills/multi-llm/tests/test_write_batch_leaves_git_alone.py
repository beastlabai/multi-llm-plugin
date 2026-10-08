"""Every write batch is told to leave staging and committing to the user.

Live run 12 (2026-10-08), OpenCode racing T012 ("Make vendored three
trackable in git"): the task un-ignored ``scripts/vendor/three.module.js``
in ``.gitignore`` — its job — and then ran ``git add`` on it ("A" in ``git
status``; nothing committed), saying "staged (not committed, per no-commit
constraint)". The implement template forbade ``git commit``/``git push``
only, and the apply batches' prompts said nothing about git at all; nothing
in the plugin watches the index after a batch (the run showed the new file
only as ``tracked_files now names … scripts/vendor/three.module.js
(created)``).

Fix round 28: the result contract every write batch carries, on both legs,
says not to stage, commit, stash or push.
"""

from __future__ import annotations

import pytest

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import repo  # noqa: F401 - `repo` is a fixture
from utils.write_batch_scheduler import with_result_contract  # noqa: E402

LINE = (
    "Do not stage, commit, stash or push anything (no `git add`, `git commit`, "
    "`git stash` or `git push`)"
)


@pytest.mark.parametrize("kind, task", [
    ("implement", {"task_id": "T012", "expected_ids": ["T012"],
                   "files_to_modify": [".gitignore"], "files": [".gitignore"]}),
    ("implement", {"task_id": "T002", "expected_ids": ["T002"]}),
    ("apply_code_fixes", {"task_id": "apply-fix-batch-0", "expected_ids": ["a1"],
                          "files_to_modify": ["src/x.js"], "files": ["src/x.js"]}),
    ("apply_suggestions", {"task_id": "apply-batch-0", "expected_ids": ["g1"]}),
    ("apply_task_suggestions", {"task_id": "apply-task-batch-0", "expected_ids": ["g1"]}),
])
def test_every_write_batch_is_told(kind, task):
    prompt = with_result_contract(kind, task, "Do the batch.")
    assert "## Git (required)" in prompt
    assert LINE in prompt
    assert "even when your task makes a file trackable" in prompt


def test_the_native_prompt_says_it_too(repo, capsys):
    import implement_orchestrator

    assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
    capsys.readouterr()
    code = multi_llm.main(["util", "write-prompt", "--phase-dir", str(repo.phase_dir),
                           "--task-id", "T001", "--stdout"])
    prompt = capsys.readouterr().out
    assert code == 0 and LINE in prompt
