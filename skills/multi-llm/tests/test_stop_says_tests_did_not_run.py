"""A stop says that nothing ran the repository's tests, as a finished run does.

Fix round 16, from the sixth live run (2026-10-06, Codex and OpenCode stop
01 on the check-gate fixture): implement stopped at T005 after T001, T003
and T004 had applied code, and neither the stop's result line, its
``notice`` nor ``summary.md`` said that nothing had run the repository's
tests (``driver.verify_command`` unset, ``npm test`` available). Only the
finished run said it (fix round 14, ``unverified_line``). It is said on a
stop now whenever a code task of the phase completed, and never when one
is configured or no code ran.
"""

from __future__ import annotations

import json

import pytest

import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: E402,F401
    DENY,
    _check,
    _policy,
    repo,
)

UNVERIFIED = "Nothing ran this repository's tests: driver.verify_command is not set."


def _stop(repo, capsys):
    _check(repo, passed=False)
    assert multi_llm.main([
        "implement", str(repo.plan), "--yes", "--exec-subagents",
    ]) == 3
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.startswith("MULTI_LLM_RESULT:")]
    result = json.loads(lines[-1].split(" ", 1)[1])
    return captured.out + captured.err, result


def test_a_stop_after_code_ran_says_nothing_ran_the_tests(repo, capsys):
    text, result = _stop(repo, capsys)
    assert "ERROR: implement stopped at T005" in text
    assert UNVERIFIED in text
    # Above the stop's own block, which stays in one piece.
    assert text.index(UNVERIFIED) < text.index("ERROR: implement stopped")
    assert UNVERIFIED in (result.get("notice") or ""), result
    summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
    assert (
        "**Verification**: not configured (`driver.verify_command` is unset). "
        + UNVERIFIED
    ) in summary
    report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
    assert report["unverified"].startswith(UNVERIFIED)
    mirror = json.loads((repo.phase_dir / ".status.json").read_text())
    assert mirror["progress"]["write_subagents"]["unverified"].startswith(UNVERIFIED)


def test_a_repository_with_a_test_script_is_named_on_the_stop(repo, capsys):
    """The fixture of run 6 had `package.json`'s `test` script: the stop
    names `npm test`, as the finished run did."""
    (repo.root / "package.json").write_text(
        json.dumps({"scripts": {"test": "node --test"}}), encoding="utf-8")
    _text, result = _stop(repo, capsys)
    assert 'add verify_command: "npm test"' in (result.get("notice") or ""), result


def test_a_stop_with_a_verify_command_set_says_nothing(repo, capsys):
    _policy(repo, DENY + "  verify_command: \"true\"\n")
    text, result = _stop(repo, capsys)
    assert "ERROR: implement stopped at T005" in text
    assert "Nothing ran this repository's tests" not in text
    assert "Nothing ran this repository's tests" not in (result.get("notice") or "")


def test_a_stop_before_any_code_ran_says_nothing(repo, capsys):
    """T001, the first task, fails: no code task completed, so there is no
    edit the tests could have judged."""
    repo.monkeypatch.setenv(
        "MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Edit: create the configuration file")
    text, result = _stop(repo, capsys)
    assert "ERROR: implement stopped at T001" in text
    assert "Nothing ran this repository's tests" not in text
    assert "Nothing ran this repository's tests" not in (result.get("notice") or "")


#: Fix round 17 (review of round 16): a check (T001) that reports findings
#: and holds T002, and an independent edit (T003) that fails. The stop comes
#: after the check ran and before any code task applied an edit.
CHECK_FIRST = """# Check first

## Overview
A check that runs before any edit lands.

## Tasks

### T001: Check the module
Run the release check. Read-only.
- Depends on: none
- Files to modify: none

### T002: Edit: write the release note
Write the note.
- Depends on: T001
- Files to modify: src/main.py

### T003: Edit: add the greeting module
Add the greeting.
- Depends on: none
- Files to modify: src/main.py
"""


@pytest.fixture
def check_first(tmp_path, monkeypatch):
    from tests.test_check_gate_fixture import CHECK_GATE, FINDINGS
    from tests.test_implement_stop_account import make_repo

    obj = make_repo(tmp_path, monkeypatch, CHECK_FIRST)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001", "T002", "T003"]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": FINDINGS}))
    # T003 fails having moved nothing: only T002, which never runs, edits.
    monkeypatch.setenv(
        "MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Edit: add the greeting module")
    monkeypatch.setenv("MOCK_LLM_EDITS_IF_PROMPT_HAS", "Title: Edit: write the release note")
    yield obj
    registry._config = None
    registry._config_key = None


def test_a_stop_after_only_a_check_ran_says_nothing(check_first, capsys):
    """The tests line is about edits nothing checked. A check task edits
    nothing (and may have run the tests itself), so a stop after a check
    alone completed must not say "Nothing ran this repository's tests"."""
    assert multi_llm.main([
        "implement", str(check_first.plan), "--yes", "--exec-subagents",
    ]) == 3
    captured = capsys.readouterr()
    text = captured.out + captured.err
    statuses = get_or_create_state(check_first.plan).implementation_task_statuses()
    # The check ran (findings, T002 held) and T003 failed: no edit applied.
    assert statuses["T001"] == "completed", statuses
    assert statuses["T003"] == "failed", statuses
    assert "ERROR: implement stopped at T003" in text, text
    assert "Nothing ran this repository's tests" not in text
    report = json.loads(
        (check_first.phase_dir / "write_batch_report.json").read_text())
    assert not report.get("unverified"), report


def test_a_resume_after_only_a_check_completed_says_nothing_either(
    check_first, capsys,
):
    """The same on a resume: the check is on record `completed` in
    state.json (the earlier run's), which is not an applied edit."""
    assert multi_llm.main([
        "implement", str(check_first.plan), "--yes", "--exec-subagents",
    ]) == 3
    capsys.readouterr()
    assert multi_llm.main([
        "implement", str(check_first.plan), "--yes", "--exec-subagents", "--resume",
    ]) == 3
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "ERROR: implement stopped at T003" in text, text
    assert "Nothing ran this repository's tests" not in text


CHECK_ONLY = """# Check only

## Overview
A plan whose one task is a check.

## Tasks

### T001: Check the module
Run the release check. Read-only.
- Depends on: none
- Files to modify: none
"""


def test_a_finished_run_of_a_check_alone_runs_no_verify_command(
    tmp_path, monkeypatch, capsys,
):
    """`_run_verification` asks the same question as the stop's line: a
    run whose tasks edited nothing has no edit for `driver.verify_command`
    to judge (it used to run on any dispatched batch)."""
    from tests.test_check_gate_fixture import CHECK_GATE
    from tests.test_implement_stop_account import make_repo

    repo = make_repo(tmp_path, monkeypatch, CHECK_ONLY)
    (tmp_path / "driver.yaml").write_text(
        "driver:\n  subagent_runner: codex:gpt-5.5\n"
        "  verify_command: \"touch verify-ran\"\n", encoding="utf-8")
    registry._config = None
    registry._config_key = None
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001"]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": True, "findings": []}))
    try:
        assert multi_llm.main([
            "implement", str(repo.plan), "--yes", "--exec-subagents",
        ]) == 0
    finally:
        registry._config = None
        registry._config_key = None
    captured = capsys.readouterr()
    assert not (repo.root / "verify-ran").exists()
    assert "Nothing ran this repository's tests" not in captured.out + captured.err
    report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
    assert not report.get("verification") and not report.get("unverified"), report
