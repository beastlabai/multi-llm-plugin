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

import multi_llm  # noqa: E402
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
