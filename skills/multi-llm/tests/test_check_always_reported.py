"""A batch that edits code always reports its `check` (fix round 26).

Live run 11 (2026-10-08): Claude Code racing T005 ran `npm test` and replied
with no `check` object (the plugin printed `[OWN_CHECK_NOT_RECORDED]`), and so
did Claude Code's stop-accept T001; OpenCode's free model left it out for
T001 (which had verified its JSON with node) and T004 of the stop fixture.
The prompt asked for `check` in a section of its own, below a result shape
that did not carry it, headed "required when you ran any" and ending "Leave
`check` out when you ran no check".

The result shape now carries `check` for every batch that edits code, the
section says it is required in every reply, and it asks for `ran` (each
command and how its last run ended) or, when no check ran, an empty `ran`
and a `not_run` reason. Both legs read it the same way: an empty `ran` is
no check, never a pass (the executor's `batch_check_findings`; the native
leg's `util batch-done --own-check none`), and a command `ran` maps to
`failed` is a failing check whatever `passed` says.
"""

from __future__ import annotations

import json

import pytest

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import _check, _implement  # noqa: E402
from tests.test_check_gate_fixture import repo as gate_repo  # noqa: E402,F401
from tests.test_check_findings_hold_dependents import repo  # noqa: E402,F401
from tests.test_write_executor import env  # noqa: E402,F401
from utils import batch_done  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.state_manager import (  # noqa: E402
    TASK_BATCH_CHECK_KEY,
    TASK_OWN_CHECK_RAN_KEY,
    get_or_create_state,
)
from utils.subagent_prompts import build_schema_repair_prompt  # noqa: E402
from utils.write_batch_scheduler import (  # noqa: E402
    CHECK_RESULT_CONTRACT,
    WRITE_CHECK_CONTRACT,
    with_result_contract,
)

NONE_RAN = {"ran": {}, "not_run": "the page needs a browser to check"}


def _task(declared):
    return {"task_id": "T005", "expected_ids": ["T005"], "declared_files": declared}


def _shape(text: str) -> str:
    """The one result shape the contract shows (the first fenced json)."""
    start = text.index("```json\n", text.index("## Result")) + len("```json\n")
    return text[start:text.index("```", start)]


class TestThePromptAsksForItFirmly:

    @pytest.mark.parametrize("kind", ["implement", "apply_code_fixes"])
    def test_the_result_shape_carries_check(self, kind):
        text = with_result_contract(kind, _task(["index.html"]), "Do it.")
        shape = json.loads(_shape(text))
        assert shape["check"] == {
            "passed": True, "findings": [],
            "ran": {"<each check command you ran>": "passed"},
        }
        assert "- `check`: required in every reply, also when you ran no check" in text

    def test_the_section_says_always_and_how_to_say_none_ran(self):
        text = " ".join(WRITE_CHECK_CONTRACT.split())
        assert "## Checks you ran (required in every reply)" in WRITE_CHECK_CONTRACT
        assert "Always send the `check` object" in text
        assert "Leave `check` out" not in text
        assert "required when you ran any" not in text
        assert "- `ran`: every check you ran on your edits" in text
        assert '{"check": {"ran": {}, "not_run": "<why no check could run on these edits>"}}' in text

    @pytest.mark.parametrize("kind", ["apply_suggestions", "apply_task_suggestions"])
    def test_a_plan_text_batch_is_not_asked(self, kind):
        text = with_result_contract(kind, _task(["plans/p.md"]), "Do it.")
        assert '"check"' not in _shape(text)
        assert WRITE_CHECK_CONTRACT not in text

    def test_a_check_task_keeps_its_own_contract(self):
        text = with_result_contract("implement", _task([]), "Check.")
        assert '"check"' not in _shape(text)
        assert CHECK_RESULT_CONTRACT in text and WRITE_CHECK_CONTRACT not in text

    def test_the_native_prompt_asks_the_same(self, repo, capsys):
        """`util write-prompt` (the native leg) binds the same contract."""
        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        assert multi_llm.main([
            "util", "write-prompt", "--phase-dir", str(repo.phase_dir),
            "--task-id", "T001", "--stdout"]) == 0
        text = capsys.readouterr().out
        assert '"check": {"passed": true, "findings": [], "ran": ' in _shape(text)
        assert "## Checks you ran (required in every reply)" in text

    def test_a_report_repair_keeps_ran_and_not_run(self):
        prompt = build_schema_repair_prompt(
            "implement", "orig", ["missing summary"],
            previous_reply='{"status": "applied"}', expected_ids=["T005"],
        )
        assert "`ran`, `not_run`," in " ".join(prompt.split())


class TestTheExecutorReadsIt:

    @pytest.mark.parametrize("check", [
        NONE_RAN,
        {**NONE_RAN, "passed": True, "findings": []},
        {"passed": True, "findings": ["npm test: 3 passed"], "ran": []},
        {"passed": False, "findings": [], "ran": {}},
        {"passed": True, "findings": [], "ran": {"npm test": "not run"}},
    ])
    def test_an_empty_ran_is_no_check_never_a_pass(self, check):
        assert executor.batch_check_findings({"check": check}) is None
        assert executor.ran_no_check({"check": check})

    def test_a_failure_it_names_anyway_still_counts(self):
        reply = {"check": {"passed": False, "findings": ["npm test: 1 failing"], "ran": {}}}
        assert executor.batch_check_findings(reply) == ["npm test: 1 failing"]

    @pytest.mark.parametrize("ran", [
        {"npm test": "failed", "node --check src/a.js": "passed"},
        ["npm test: failed", "node --check src/a.js: passed"],
    ])
    def test_a_command_ran_says_failed_fails_whatever_passed_says(self, ran):
        reply = {"check": {"passed": True, "findings": [], "ran": ran}}
        (finding,) = executor.batch_check_findings(reply)
        assert finding.startswith("npm test: failed")

    @pytest.mark.parametrize("check", [
        {"passed": True, "findings": [], "ran": {"npm test": "passed"}},
        {"passed": True, "findings": [], "ran": ["npm test: passed"]},
        # An entry that cannot be read leaves what ran unknown: `passed` decides.
        {"passed": True, "findings": [], "ran": {"npm test": "12 tests ok"}},
        {"passed": True, "findings": []},
    ])
    def test_a_pass_is_still_a_pass(self, check):
        assert executor.batch_check_findings({"check": check}) == []
        assert not executor.ran_no_check({"check": check})

    def test_a_report_repair_keeps_none_ran(self):
        """F4 (eb93ee3) keeps a readable check across a report-only repair;
        "no check ran" is kept too, so the repair cannot make it a pass."""
        assert executor._report_check_evidence({"check": NONE_RAN}) == NONE_RAN

    def test_an_implement_run_records_no_own_check_for_none_ran(
        self, gate_repo, capsys, monkeypatch,
    ):
        """A run whose code tasks all say they ran nothing (with a stray
        `passed: true`) puts none of them on the tests line."""
        from tests.test_check_gate_fixture import ALLOW, _policy

        _policy(gate_repo, ALLOW)
        _check(gate_repo, passed=True)
        monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
            {**NONE_RAN, "passed": True, "findings": []}))
        assert _implement(gate_repo) == 0
        capsys.readouterr()
        state = json.loads((gate_repo.phase_dir.parent / "state.json").read_text())
        assert TASK_OWN_CHECK_RAN_KEY not in state, state.get(TASK_OWN_CHECK_RAN_KEY)
        assert TASK_BATCH_CHECK_KEY not in state


def _record(capsys, phase_dir, task_id, *extra, status="completed", code=0):
    got = multi_llm.main([
        "util", "batch-done", "--phase-dir", str(phase_dir), "--task-id", task_id,
        "--status", status, *extra,
    ])
    captured = capsys.readouterr()
    assert got == code, (captured.out, captured.err)
    return captured.out + captured.err


class TestTheNativeLegRecordsIt:

    def _emit(self, repo, capsys):
        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()

    def test_own_check_none_is_no_check_and_draws_no_nudge(self, repo, capsys):
        self._emit(repo, capsys)
        out = _record(capsys, repo.phase_dir, "T001", "--own-check", "none")
        assert "[BATCH_DONE] T001 completed" in out
        assert batch_done.OWN_CHECK_NOT_RECORDED_MARKER not in out
        state = get_or_create_state(repo.plan).state
        assert state["task_status"]["T001"] == "completed"
        assert TASK_OWN_CHECK_RAN_KEY not in state
        assert TASK_BATCH_CHECK_KEY not in state

    def test_it_clears_an_earlier_pass(self, repo, capsys):
        self._emit(repo, capsys)
        _record(capsys, repo.phase_dir, "T001", "--own-check", "passed")
        assert get_or_create_state(repo.plan).state[TASK_OWN_CHECK_RAN_KEY] == ["T001"]
        _record(capsys, repo.phase_dir, "T001", "--own-check", "none")
        assert TASK_OWN_CHECK_RAN_KEY not in get_or_create_state(repo.plan).state

    def test_it_takes_no_finding(self, repo, capsys):
        self._emit(repo, capsys)
        out = _record(capsys, repo.phase_dir, "T001", "--own-check", "none",
                      "--finding", "npm test: 1 failing", code=3)
        assert "--own-check findings" in out

    def test_the_nudge_names_it(self, repo, capsys):
        self._emit(repo, capsys)
        out = _record(capsys, repo.phase_dir, "T001")
        (line,) = [l for l in out.splitlines()
                   if l.startswith(batch_done.OWN_CHECK_NOT_RECORDED_MARKER)]
        assert "--own-check none when its `ran` is empty" in line
