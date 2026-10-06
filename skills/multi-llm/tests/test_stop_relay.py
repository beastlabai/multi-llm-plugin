"""What a stop hands the host to relay, on every leg (fix round 18).

Live run 7 (2026-10-06), the stop-path fixture and the racing plan:

* On Codex and OpenCode the "Nothing ran this repository's tests … npm test"
  line was in the stop result's ``notice``, and neither host relayed
  ``notice`` at a stop; both relay the stop's own line, the result line's
  ``error``. The line now ends that headline too.
* Codex racing 08 (an apply-code-fixes partial application) left the result
  line with no ``error`` at all: the tool's needs-human message said "Its
  result line names no error", and the host relayed only the resume command,
  not what stopped, what the batch reported or what did not run. An apply
  phase's stop records its headline as ``error``, as implement's does.
* The host relay rule (every route's needs-human sentence and the tool
  result's message) says to give that line whole.
"""

from __future__ import annotations

import json

import pytest

import modes  # noqa: E402
import multi_llm  # noqa: E402
from utils.cli_common import ERROR_MAX_CHARS  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: E402,F401
    DENY,
    _check,
    _policy,
    repo,
)
from tests import test_write_executor_orchestrators as _orchestrators  # noqa: E402
from tests.test_write_executor_orchestrators import _ApplyRig  # noqa: E402

UNVERIFIED = "Nothing ran this repository's tests: driver.verify_command is not set."


@pytest.fixture
def apply_repo(tmp_path, monkeypatch):
    """``test_write_executor_orchestrators``' repo, under its own name here:
    this module's ``repo`` is the check-gate fixture's."""
    yield from _orchestrators.repo.__wrapped__(tmp_path, monkeypatch)


def _result(out: str) -> dict:
    line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
    return json.loads(line.split(" ", 1)[1])


def _stop(repo, capsys):
    _check(repo, passed=False)
    assert multi_llm.main([
        "implement", str(repo.plan), "--yes", "--exec-subagents",
    ]) == 3
    captured = capsys.readouterr()
    return captured.out + captured.err, _result(captured.out)


class TestTheTestsLineTravelsWithTheStop:

    def test_the_stop_s_error_ends_with_it(self, repo, capsys):
        (repo.root / "package.json").write_text(
            json.dumps({"scripts": {"test": "node --test"}}), encoding="utf-8")
        _text, result = _stop(repo, capsys)
        error = result["error"]
        assert error.startswith("implement stopped at T005"), error
        assert UNVERIFIED in error, error
        assert error.endswith(
            'add verify_command: "npm test" under driver: in '
            ".multi-llm/providers.yaml (package.json defines scripts.test)."
        ), error
        # After the stop's own words: the OpenCode stop title, cut at 300
        # characters, still opens on what stopped.
        assert error.index("To go on") < error.index(UNVERIFIED)
        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        assert report["stop"]["headline"].endswith(error[-60:])
        mirror = json.loads((repo.phase_dir / ".status.json").read_text())
        assert UNVERIFIED in mirror["progress"]["write_subagents"]["stop"]["headline"]

    def test_a_long_headline_is_cut_before_it_never_through_it(self, repo, capsys):
        """The result line's `error` is bounded; the tests line is the part
        that must reach the host whole."""
        long_title = "Add the build stamp module " + "with a very long title " * 80
        repo.plan.write_text(repo.plan.read_text(encoding="utf-8").replace(
            "### T005: Edit: add the build stamp module",
            f"### T005: Edit: {long_title.strip()}",
        ), encoding="utf-8")
        _text, result = _stop(repo, capsys)
        error = result["error"]
        assert len(error) <= ERROR_MAX_CHARS, len(error)
        assert error.startswith("implement stopped at T005"), error[:200]
        tail = error[error.index(UNVERIFIED):]
        assert tail.endswith("set to the command that runs your tests or checks your build."), tail
        assert "… " + UNVERIFIED in error

    def test_a_verify_command_keeps_it_out(self, repo, capsys):
        _policy(repo, DENY + "  verify_command: \"true\"\n")
        _text, result = _stop(repo, capsys)
        assert result["error"].startswith("implement stopped at T005")
        assert "Nothing ran this repository's tests" not in result["error"]


class TestAnApplyStopNamesItsLine(_ApplyRig):
    """Codex racing 08: an apply phase's stop recorded no `error`."""

    def test_a_partial_application_records_its_headline_as_error(
        self, apply_repo, capsys,
    ):
        repo = apply_repo
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        capsys.readouterr()
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "plans/apply.md", "mode": "patch", "content": "edit\n"}
        ]))
        # Batch 0 attests its item; batch 1 moves bytes and attests none.
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(ids[0]))
        assert multi_llm.main([
            "apply-suggestions", str(plan), "--yes", "--exec-subagents",
        ]) == 3
        result = _result(capsys.readouterr().out)
        assert result["state"] == "needs-human", result
        error = result.get("error") or ""
        assert error.startswith("apply-suggestions stopped at apply-batch-1"), result
        assert "partial application" in error
        assert "To go on: inspect the half-applied state" in error
        report = json.loads(
            (self._phase_dir(plan) / "write_batch_report.json").read_text())
        assert error == report["stop"]["headline"]


class TestTheRelayRuleSaysToGiveTheLineWhole:
    """Live run 7: the OpenCode accept host cut the stop's line mid-word, and
    the Codex host (racing 08) relayed only the resume command."""

    def test_the_needs_human_rule(self):
        rule = modes.NEEDS_HUMAN_RULE
        assert "the last error line it printed, whole" in rule
        assert (
            "on a stop it says what stopped, what it reported, what did not run "
            "and whether anything ran the repository's tests"
        ) in rule

    def test_the_tool_result_message(self):
        template = modes.NEEDS_HUMAN_MESSAGE["template"]
        assert "that line whole, word for word" in template
        assert "whether anything ran the repository's tests" in template
        # The rule against inventing a command stands beside it.
        assert "do not lift a command out of it for the user" in template
        assert "Never compose a command for the user yourself" in (
            modes.TOOL_NO_COMPOSED_COMMAND_LINE)


@pytest.mark.parametrize("name", ["SKILL.md"])
def test_the_claude_code_shim_carries_the_same_rule(name):
    from pathlib import Path

    text = (Path(modes.__file__).parent / name).read_text(encoding="utf-8")
    assert modes.NEEDS_HUMAN_RULE in " ".join(text.split())
