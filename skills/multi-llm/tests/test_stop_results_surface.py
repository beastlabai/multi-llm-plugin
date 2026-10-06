"""What a stopped implement hands the user, on every surface (fix round 14).

Live run 5 (2026-10-06), the stop-path fixture on the Codex and OpenCode tool
routes and on Claude Code's native leg. Each class pins one finding:

* ``status --view run`` listed the tasks the user skipped with
  ``--human-task T005=skip`` (and the one skipped with it) as "deferred
  manual", beside a summary saying "Skipped at your request";
* after ``--resume --human-task …`` the next stop's ``resume_command`` was a
  plain ``--resume`` while its first option replayed the answers: the
  answers are on record, and a plain ``--resume`` applies them;
"""

from __future__ import annotations

import json

import pytest

from tests.test_check_gate_fixture import (  # noqa: E402
    NOTE, _check, _first_run, _implement, _policy, repo,  # noqa: F401 - the fixture
)
from tests.test_implement_stop_account import _dispatched_titles, _statuses  # noqa: E402
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


class TestAPlainResumeKeepsTheAnswersOnRecord:

    def test_a_skip_answered_once_holds_through_later_resumes(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(repo, "--resume", "--human-task", "T005=skip") == 3
        capsys.readouterr()
        repo.log.unlink(missing_ok=True)
        # Nothing changed and no answer is repeated: T005 is not refused again.
        assert _implement(repo, "--resume") == 3
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert _dispatched_titles(repo) == []
        assert "T005 refused before dispatch" not in text
        assert _statuses(repo)["T005"] == "skipped"
        # Accepting T002 alone runs T007; T005's earlier answer still holds.
        assert _implement(repo, "--resume", "--human-task", "T002=done") == 0
        capsys.readouterr()
        assert _dispatched_titles(repo) == [NOTE]
        statuses = _statuses(repo)
        assert statuses["T005"] == "skipped" and statuses["T006"] == "skipped"
