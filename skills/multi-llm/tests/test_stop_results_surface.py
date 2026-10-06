"""What a stopped implement hands the user, on every surface (fix round 14).

Live run 5 (2026-10-06), the stop-path fixture on the Codex and OpenCode tool
routes and on Claude Code's native leg. Each class pins one finding:

* ``status --view run`` listed the tasks the user skipped with
  ``--human-task T005=skip`` (and the one skipped with it) as "deferred
  manual", beside a summary saying "Skipped at your request";
"""

from __future__ import annotations

import json

import pytest

from tests.test_check_gate_fixture import (  # noqa: E402
    _check, _first_run, _implement, _policy, repo,  # noqa: F401 - the fixture
)
from utils import run_status  # noqa: E402


def _status_run(repo, capsys) -> str:
    capsys.readouterr()
    run_status.main(["--plan-file", str(repo.plan), "--view", "run"])
    return capsys.readouterr().out


class TestASkipYouChoseIsNotDeferredManualWork:

    def test_status_names_the_skip_and_its_dependent_apart(self, repo, capsys):
        _first_run(repo, capsys)
        # T002's findings still hold T007, so the resume stops again.
        assert _implement(repo, "--resume", "--human-task", "T005=skip") == 3
        mirror = json.loads(
            (repo.phase_dir / ".status.json").read_text()
        )["progress"]["write_subagents"]
        assert mirror["declined_task_ids"] == ["T005"]
        assert mirror["declined_dependent_task_ids"] == ["T006"]
        out = _status_run(repo, capsys)
        assert "skipped at your request: T005" in out, out
        assert "skipped (depend on a task you skipped): T006" in out, out
        assert "deferred manual" not in out, out
