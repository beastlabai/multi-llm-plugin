"""One reading of a code batch's own `check`, on both legs (fix round 27).

Review of fix round 26 (relay #2, #3; records #3): after round 26 asked every
code batch for `ran` (and `--own-check none` recorded a native reply that ran
no check), the two legs read the same `check` object differently:

* `{"passed": true, "findings": ["npm test: 2 failing"], "ran": {}}` was "no
  check" on the executor leg and a pass on the native leg, whose runbook said
  `--own-check passed` for `passed: true`;
* the result shape pasted unchanged (`ran` holding only the
  `<each check command you ran>` placeholder) was "no check" on the executor
  leg and `--own-check passed` on the native leg, whose runbook said `none`
  only for an empty `ran`;
* `{"passed": false, "findings": [], "ran": {}}`, a failing check before round
  26 (`CHECK_FAILED_WITHOUT_FINDINGS`), became "no check" on both;
* (relay #3) a `ran` list of objects whose `command` is the placeholder read
  as a clean pass.

Now one rule, stated once in `references/own-check.md`, which both runbooks
point to and the executor's `batch_check_findings` follows: a finding a reply
names is a failing check whatever `passed` says, unless `passed: true` comes
with a `ran` that names a command it ran (fix round 15: what it lists beside
a pass is informational); `passed: false` is a failing check, its placeholder
finding kept, also when nothing ran; and only a reply that names no finding,
does not say it failed and ran nothing is no check. The table below is that
rule; each row is read by the executor and recorded by `util batch-done` with
the flags the reference gives for it, and both must land in the same place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from tests.test_check_findings_hold_dependents import repo  # noqa: E402,F401
from tests.test_check_gate_fixture import _check, _implement  # noqa: E402
from tests.test_check_gate_fixture import repo as gate_repo  # noqa: E402,F401
from utils import subagent_executor as executor  # noqa: E402
from utils.state_manager import (  # noqa: E402
    TASK_BATCH_CHECK_KEY,
    TASK_OWN_CHECK_RAN_KEY,
    get_or_create_state,
)

SKILL_DIR = Path(__file__).resolve().parent.parent
REFERENCE = "references/own-check.md"
PLACEHOLDER = "<each check command you ran>"
COULD_NOT_RUN = executor.BATCH_CHECK_COULD_NOT_RUN

NONE = ["--own-check", "none"]
PASSED = ["--own-check", "passed"]


def _findings(*texts):
    flags = ["--own-check", "findings"]
    for text in texts:
        flags += ["--finding", text]
    return flags


#: (the reply's `check`, the flags `references/own-check.md` gives for it,
#: where both legs must put it: "findings", "pass" or "none").
TABLE = [
    # -- the review's three objects --------------------------------------
    pytest.param(
        {"passed": True, "findings": ["npm test: 2 failing"], "ran": {}},
        _findings("npm test: 2 failing"), "findings", id="a-finding-beside-a-pass-of-nothing"),
    pytest.param(
        {"passed": True, "findings": [], "ran": {PLACEHOLDER: "passed"}},
        NONE, "none", id="the-shape-pasted-unchanged"),
    pytest.param(
        {"passed": False, "findings": [], "ran": {}},
        _findings(), "findings", id="said-it-failed-ran-nothing"),
    # -- the placeholder in each spelling ran is read in ------------------
    pytest.param(
        {"passed": True, "findings": [], "ran": [f"{PLACEHOLDER}: passed"]},
        NONE, "none", id="placeholder-as-a-string"),
    pytest.param(
        {"passed": True, "findings": [],
         "ran": [{"command": PLACEHOLDER, "outcome": "passed"}]},
        NONE, "none", id="placeholder-as-an-object"),
    pytest.param(
        {"passed": True, "findings": [],
         "ran": [{"command": PLACEHOLDER, "outcome": "passed"},
                 {"command": "npm test", "outcome": "not run"}]},
        NONE, "none", id="placeholder-object-beside-not-run"),
    # -- it ran nothing and said so ---------------------------------------
    pytest.param(
        {"ran": {}, "not_run": "the page needs a browser to check"},
        NONE, "none", id="nothing-ran-as-asked"),
    pytest.param(
        {"passed": True, "findings": [], "ran": {"npm test": "not run"}},
        NONE, "none", id="every-command-not-run"),
    pytest.param(
        {"passed": True, "findings": "none", "ran": {}},
        NONE, "none", id="a-bare-none-is-no-finding"),
    pytest.param(
        {"passed": True, "findings": ["n/a"], "ran": []},
        NONE, "none", id="an-n/a-entry-is-no-finding"),
    # -- it ran something --------------------------------------------------
    pytest.param(
        {"passed": True, "findings": [], "ran": {"npm test": "passed"}},
        PASSED, "pass", id="a-pass"),
    pytest.param(
        {"passed": True, "findings": ["npm test: 12 passed"], "ran": {"npm test": "passed"}},
        PASSED, "pass", id="fix-round-15-a-pass-listing-what-it-ran"),
    pytest.param(
        {"passed": True, "findings": ["npm test: 12 passed"]},
        PASSED, "pass", id="fix-round-15-an-older-reply-without-ran"),
    pytest.param(
        {"passed": True, "findings": [], "ran": {"npm test": "failed"}},
        _findings("npm test: failed"), "findings", id="a-pass-beside-a-failed-command"),
    pytest.param(
        {"passed": False, "findings": [], "ran": {"npm test": "passed"}},
        _findings(), "findings", id="said-it-failed-beside-passing-commands"),
    pytest.param(
        {"passed": False, "findings": ["npm test: 1 failing"], "ran": {"npm test": "failed"}},
        _findings("npm test: 1 failing"), "findings", id="a-failure-it-names"),
    pytest.param(
        {"ran": {"npm test": "passed"}},
        PASSED, "pass", id="no-passed-key-a-passing-command"),
    pytest.param(
        {"ran": ["npm test: failed"]},
        _findings("npm test: failed"), "findings", id="no-passed-key-a-failed-command"),
    pytest.param(
        {"findings": ["npm test: 1 failing"], "ran": {"npm test": "passed"}},
        _findings("npm test: 1 failing"), "findings", id="no-passed-key-a-finding"),
    pytest.param({}, NONE, "none", id="an-empty-object"),
    pytest.param({"passed": None, "findings": []}, NONE, "none", id="passed-null"),
    # -- a check that could not run ---------------------------------------
    pytest.param(
        {"passed": False, "could_not_run": True, "findings": ["npx: not found"], "ran": {}},
        _findings(f"{COULD_NOT_RUN}: npx: not found"), "findings", id="could-not-run-naming-why"),
    pytest.param(
        {"could_not_run": True, "ran": {}, "not_run": "node is not installed"},
        _findings(COULD_NOT_RUN), "findings", id="could-not-run-naming-nothing"),
]


def _executor_reads(check) -> str:
    found = executor.batch_check_findings({"status": "applied", "check": check})
    return "none" if found is None else "findings" if found else "pass"


class TestTheExecutorLeg:

    @pytest.mark.parametrize("check, _flags, where", TABLE)
    def test_it_reads_each_object_by_the_rule(self, check, _flags, where):
        assert _executor_reads(check) == where

    def test_a_finding_beside_a_pass_of_nothing_is_that_finding(self):
        assert executor.batch_check_findings({"check": {
            "passed": True, "findings": ["npm test: 2 failing"], "ran": {}}}) == [
            "npm test: 2 failing"]

    def test_said_it_failed_keeps_the_placeholder_finding(self):
        for ran in ({}, {"npm test": "passed"}):
            assert executor.batch_check_findings({"check": {
                "passed": False, "findings": [], "ran": ran}}) == [
                executor.CHECK_FAILED_WITHOUT_FINDINGS]

    def test_a_failed_command_stands_in_for_the_placeholder(self):
        """`passed: false` naming nothing beside a command `ran` says failed
        is that command's failure, not the placeholder beside it."""
        (finding,) = executor.batch_check_findings({"check": {
            "passed": False, "findings": [], "ran": {"npm test": "failed"}}})
        assert finding.startswith("npm test: failed")

    def test_a_placeholder_object_says_nothing_ran(self):
        reply = {"check": {"passed": True, "findings": [],
                           "ran": [{"command": PLACEHOLDER, "outcome": "passed"}]}}
        assert executor.ran_no_check(reply)
        # F4 (eb93ee3): a report-only repair keeps it, so the repair cannot
        # turn it into a pass.
        assert executor._report_check_evidence(reply) == reply["check"]

    def test_an_implement_run_records_the_finding(self, gate_repo, capsys, monkeypatch):
        """End to end on the executor leg: a code task that says it passed
        a check it did not run, naming a failure, is on record failing."""
        from tests.test_check_gate_fixture import ALLOW, _policy

        _policy(gate_repo, ALLOW)
        _check(gate_repo, passed=True)
        monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
            {"passed": True, "findings": ["npm test: 2 failing"], "ran": {}}))
        assert _implement(gate_repo) == 0
        capsys.readouterr()
        state = json.loads((gate_repo.phase_dir.parent / "state.json").read_text())
        recorded = state.get(TASK_BATCH_CHECK_KEY) or {}
        assert recorded and all(v == ["npm test: 2 failing"] for v in recorded.values()), recorded


class TestTheNativeLegRecordsTheSame:

    @pytest.mark.parametrize("check, flags, where", TABLE)
    def test_the_reference_s_flags_land_where_the_executor_does(
        self, repo, capsys, check, flags, where,
    ):
        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        got = multi_llm.main([
            "util", "batch-done", "--phase-dir", str(repo.phase_dir),
            "--task-id", "T001", "--status", "completed", *flags,
        ])
        captured = capsys.readouterr()
        assert got == 0, (captured.out, captured.err)
        state = get_or_create_state(repo.plan).state
        native = (
            "findings" if "T001" in (state.get(TASK_BATCH_CHECK_KEY) or {})
            else "pass" if "T001" in (state.get(TASK_OWN_CHECK_RAN_KEY) or [])
            else "none"
        )
        assert native == where == _executor_reads(check)


class TestTheRuleIsStatedOnce:

    def _text(self, name):
        return " ".join((SKILL_DIR / name).read_text(encoding="utf-8").split())

    def test_the_reference_states_each_case(self):
        text = self._text(REFERENCE)
        for clause in (
            "it names a finding: `--own-check findings` and one `--finding` per "
            "finding, whatever `passed` says",
            "else, `passed` is false: `--own-check findings` with no `--finding`",
            "else: `--own-check none` — it ran no check, which is never a pass",
            "what it lists beside a pass, such as `npm test: 12 passed`, is not "
            "a failing check",
            "the result shape's placeholder `<each check command you ran>`",
            '`--own-check findings --finding "its check could not run: ',
            "`subagent_executor.batch_check_findings`",
        ):
            assert clause in text, clause

    @pytest.mark.parametrize("core", ["implement.md", "apply-code-fixes.md"])
    def test_both_runbooks_point_to_it_and_restate_none_of_it(self, core):
        text = self._text(f"instructions/{core}")
        assert text.count(f"`{REFERENCE}`") >= 2, core
        # The round-26 clauses that read the object differently are gone.
        assert "`--own-check none` when its `ran` is empty" not in text
        assert "and `ran` maps no command to `failed`" not in text

    def test_the_record_nudge_points_to_it(self, repo, capsys):
        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        assert multi_llm.main([
            "util", "batch-done", "--phase-dir", str(repo.phase_dir),
            "--task-id", "T001", "--status", "completed"]) == 0
        out = capsys.readouterr().out
        (line,) = [l for l in out.splitlines() if l.startswith("[OWN_CHECK_NOT_RECORDED]")]
        assert REFERENCE in line
        assert "--own-check none when its `ran` is empty" not in line
