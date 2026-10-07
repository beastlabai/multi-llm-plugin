"""The write mirror's `kept_task_ids` is the `resume: kept` line's own set.

Fix round 23 (review of round 22, progress #3): round 22 said the mirror's
`kept_task_ids` equals the resume line's kept set, but it left out the tasks
kept on the user's answer. After `--resume --human-task T002=done` the line
read "kept 4 … T001, T003, T004, T002 (accepted with its check findings)"
while `kept_task_ids` was `['T001', 'T003', 'T004']`: `status`'s text folds
the accepted task back in, a `--json` reader got three. A manual task
answered done read "kept 2 … T001, T005 (completed by hand)" on the line and
"1 kept from an earlier run: T001" on `status`. Both name the same tasks, in
the same words, now.
"""

from __future__ import annotations

import json
import re

import pytest

import implement_orchestrator
import multi_llm
from tests.test_check_gate_fixture import _first_run, repo  # noqa: F401
from tests.test_implement_stop_account import (
    _check_fails,
    _check_passes,
    _implement,
    make_repo,
)
from utils import provider_registry as registry


@pytest.fixture
def manual_repo(tmp_path, monkeypatch):
    """The five-task plan with a manual task at the end (T005)."""
    yield make_repo(tmp_path, monkeypatch)
    registry._config = None
    registry._config_key = None


def _resume_line(text: str) -> tuple:
    """``(count, ids)`` of the run's ``resume: kept`` line."""
    line = next(l for l in text.splitlines() if "resume: kept" in l)
    match = re.search(r"resume: kept (\d+) task\(s\) an earlier run finished — (.*?); dispatched", line)
    assert match, line
    ids = re.findall(r"\b(T\d{3})\b", match.group(2))
    return int(match.group(1)), ids


def _mirror(repo) -> dict:
    status = json.loads((repo.phase_dir / ".status.json").read_text(encoding="utf-8"))
    return status["progress"]["write_subagents"]


def _progress_line(repo, capsys) -> str:
    multi_llm.main(["status", str(repo.plan), "--view", "run", "--phase", "implement"])
    out = capsys.readouterr().out
    return next(l for l in out.splitlines() if "progress:" in l and "implement" in l)


def test_an_accepted_check_is_among_the_kept_ids(repo, capsys):
    _first_run(repo, capsys)
    _implement(repo, "--resume", "--human-task", "T002=done")
    captured = capsys.readouterr()
    count, ids = _resume_line(captured.out + captured.err)
    assert (count, ids) == (4, ["T001", "T003", "T004", "T002"])
    assert _mirror(repo)["kept_task_ids"] == ids
    assert (
        "(4 kept from an earlier run: T001, T003, T004, T002 (accepted with its "
        "check findings))"
    ) in _progress_line(repo, capsys)


def test_a_manual_task_completed_by_hand_is_among_the_kept_ids(manual_repo, capsys):
    repo = manual_repo
    _check_fails(repo)
    assert implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--no-confirm", "--exec-subagents",
        "--force-strategy", "pause-and-ask", "--human-task", "T005=done",
    ]) == 3
    main = repo.root / "src" / "main.py"
    main.write_text(main.read_text() + "# fixed by hand\n", encoding="utf-8")
    _check_passes(repo)
    capsys.readouterr()
    assert implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--no-confirm", "--exec-subagents", "--resume",
    ]) == 0
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "kept 2 task(s) an earlier run finished — T001, T005 (completed by hand)" in text
    mirror = _mirror(repo)
    assert mirror["kept_task_ids"] == ["T001", "T005"]
    assert "(2 kept from an earlier run: T001, T005 (completed by hand))" in (
        _progress_line(repo, capsys))
