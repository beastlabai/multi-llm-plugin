"""A check task whose check reported problems is never a clean "verified".

Measured on the 2026-10-05 racing re-run 4 (Codex, implement T007): a
verification-only task ran ``npx html-validate``, which reported two problems
(doctype case, ``aria-label`` on a div). The runner answered ``no-op`` and put
the problems in its free-text ``notes`` only, so the task was recorded
``verified``, and summary.md, ``status``, the result line and review-code's
implementation record all read it as a check that passed. In run 3 the same
kind of check, worded "must exit 0", claimed ``failed`` and stopped implement
for good — the task's wording decided which of the two happened.

The reply now carries a structured ``check`` object; a check that reported
problems is recorded ``check-findings`` (the loop continues, never a stop), and
the findings live in ``state.json``'s ``task_check_findings``, which both write
legs write (the scheduler; ``util batch-done --check findings`` on the native
leg). This file pins the state-backed surfaces both legs share: the workflow
payload and ``status`` trailer, a terminal poll's ``notice``, implement's own
result-line ``notice`` and review-code's implementation record. The executor
surfaces (reports, summary.md, the batch log line, the mirror) are pinned in
``tests/test_write_executor.py::TestVerificationOnlyImplementTask``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
import code_review_orchestrator as cro  # noqa: E402
import implement_orchestrator  # noqa: E402
from utils import cli_common, run_status, status_file  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import (  # noqa: E402
    StateManager,
    set_task_check_findings_in_state,
)

from tests.test_next_phase import S, UP_TO_CODE_REVIEW, UP_TO_IMPLEMENT, _build  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent

FINDINGS = [
    "npx html-validate index.html: index.html:1 doctype must be lowercase",
    "npx html-validate index.html: index.html:105 aria-label misused on a div",
]
TASKS = {"T006": "completed", "T007": "completed", "T008": "skipped"}


def _plan(tmp_path: Path, completed=UP_TO_IMPLEMENT, findings=None) -> Path:
    here = os.getcwd()
    os.chdir(SKILL_ROOT)  # the first StateManager runs `git rev-parse HEAD`
    try:
        plan = _build(tmp_path, S(completed, tasks=TASKS))
    finally:
        os.chdir(here)
    state = StateManager(plan)
    state.record_task_titles({
        "T006": "Race lifecycle", "T007": "Run static HTML, module, and CDN checks",
        "T008": "Manual playthrough",
    })
    set_task_check_findings_in_state(
        state.state, "T007", FINDINGS if findings is None else findings,
    )
    state.save()
    return plan


def _workflow(plan: Path) -> dict:
    return prereq.get_workflow_status(StateManager(plan), plan, output_dir_path(plan))


class TestTheReplyField:
    """The verdict reads the reply's own `check` object, never its notes."""

    def test_the_readings(self):
        from utils.subagent_executor import (
            CHECK_FAILED_WITHOUT_FINDINGS, CHECK_FINDINGS_MAX, check_findings_from,
        )

        assert check_findings_from({"status": "no-op", "notes": "2 findings"}) is None
        assert check_findings_from({"check": {"passed": True}}) == []
        assert check_findings_from({"check": {"passed": False}}) == [
            CHECK_FAILED_WITHOUT_FINDINGS]
        assert check_findings_from(
            {"check": {"passed": True, "findings": ["  a\n  b "]}}) == ["a b"]
        assert check_findings_from({"check": {"passed": False, "findings": [
            {"command": "npx html-validate", "message": "doctype"}]}}) == [
            "command: npx html-validate; message: doctype"]
        many = check_findings_from(
            {"check": {"passed": False, "findings": [str(i) for i in range(30)]}})
        assert len(many) == CHECK_FINDINGS_MAX
        assert many[-1] == "(+11 more finding(s) in the batch result file)"

    def test_the_schema_accepts_it_and_an_older_reply_without_it(self):
        from utils.schema_validator import validate_with_schema_file

        base = {"status": "no-op", "ids": ["T007"], "files_touched": [],
                "summary": "checks ran"}
        for reply in (
            base,
            {**base, "check": {"passed": False, "findings": FINDINGS}},
            {**base, "check": {"passed": False, "findings": [{"message": "x"}]}},
        ):
            errors = validate_with_schema_file(reply, "write_batch_result.schema.json")
            assert not errors, (reply, errors)

    # Fix round 11 (review of round 10, T3 and repro_check_shapes): a
    # near-miss `check` was a schema_invalid reply, and implement has no
    # report re-ask, so the run stopped for good on a check that had
    # answered; and `check: {}` read as VERIFIED.
    NEAR_MISSES = [
        ({"passed": "false", "findings": ["a"]}, ["a"]),
        ({"passed": False, "findings": "a problem"}, ["a problem"]),
        ({"passed": False, "findings": None}, "failed"),
        ({"passed": 0, "findings": []}, "failed"),
        ({"passed": "FALSE"}, "failed"),
        ({"passed": None, "findings": []}, "unknown"),
        ({}, "unknown"),
        ({"passed": "maybe"}, "unknown"),
        ({"passed": False, "findings": [1, None, "  "]}, ["1"]),
        ({"passed": False, "findings": {"message": "bad doctype"}}, ["message: bad doctype"]),
        ({"passed": False, "findings": [{"a": ""}]}, "failed"),
        ("passed", "unknown"),
        (["a"], "unknown"),
        # Fix round 12 (review item 2): `check: null` read as no result, so
        # a check task's reply carrying it was VERIFIED. Only a missing key
        # is no result.
        (None, "unknown"),
    ]

    @pytest.mark.parametrize("check,expected", NEAR_MISSES)
    def test_a_near_miss_shape_is_valid_and_never_reads_as_a_pass(self, check, expected):
        from utils.schema_validator import validate_with_schema_file
        from utils.subagent_executor import (
            CHECK_FAILED_WITHOUT_FINDINGS, CHECK_RESULT_UNKNOWN,
            check_findings_from, validate_result,
        )

        reply = {"status": "no-op", "ids": ["T007"], "files_touched": [],
                 "summary": "checks ran", "check": check}
        assert validate_with_schema_file(reply, "write_batch_result.schema.json") == []
        assert validate_result("implement", reply) == []
        found = check_findings_from(reply)
        if expected == "failed":
            assert found == [CHECK_FAILED_WITHOUT_FINDINGS]
        elif expected == "unknown":
            assert len(found) == 1 and found[0].startswith(CHECK_RESULT_UNKNOWN), found
        else:
            assert found == expected

    @pytest.mark.parametrize("passed", [True, "true", "Yes", 1, "passed"])
    def test_a_clear_pass_with_nothing_listed_is_a_pass(self, passed):
        from utils.subagent_executor import check_findings_from

        assert check_findings_from({"check": {"passed": passed, "findings": []}}) == []
        assert check_findings_from({"check": {"passed": passed}}) == []

    def test_only_a_missing_check_key_is_no_result(self):
        from utils.subagent_executor import CHECK_RESULT_UNKNOWN, check_findings_from

        assert check_findings_from({"status": "no-op"}) is None
        found = check_findings_from({"status": "no-op", "check": None})
        assert found == [f"{CHECK_RESULT_UNKNOWN} (its `check` was null)"]

    # Fix round 12 (review item 4): a pass that spelled "nothing found" as a
    # bare value was one finding, which held back every task after it.
    @pytest.mark.parametrize("bare", [
        "none", "None", "N/A", "n/a", "No problems found.", "no findings", "",
        "  ", 0, "0",
    ])
    def test_a_pass_whose_bare_findings_say_nothing_found_is_a_pass(self, bare):
        from utils.subagent_executor import check_findings_from

        assert check_findings_from({"check": {"passed": True, "findings": bare}}) == []

    @pytest.mark.parametrize("check,expected", [
        ({"passed": True, "findings": "one problem"}, ["one problem"]),
        ({"passed": True, "findings": "none of the links resolve"},
         ["none of the links resolve"]),
        ({"passed": True, "findings": ["none"]}, ["none"]),
        ({"passed": True, "findings": 3}, ["3"]),
        ({"passed": True, "findings": False}, ["False"]),
        ({"passed": False, "findings": "none"}, ["none"]),
        ({"passed": "maybe", "findings": "n/a"}, ["n/a"]),
    ])
    def test_anything_else_stays_a_finding(self, check, expected):
        from utils.subagent_executor import check_findings_from

        assert check_findings_from({"check": check}) == expected


class TestTheStateRecord:
    def test_set_and_clear(self):
        state: dict = {}
        set_task_check_findings_in_state(state, "T007", FINDINGS)
        assert state["task_check_findings"] == {"T007": FINDINGS}
        set_task_check_findings_in_state(state, "T009", ["one"])
        set_task_check_findings_in_state(state, "T007", None)
        assert state["task_check_findings"] == {"T009": ["one"]}
        set_task_check_findings_in_state(state, "T009", [])
        assert "task_check_findings" not in state

    def test_the_reader_is_scoped_to_completed_tasks_of_the_last_decomposition(
        self, tmp_path
    ):
        plan = _plan(tmp_path)
        state = StateManager(plan)
        assert state.implementation_check_findings() == [{
            "id": "T007", "title": "Run static HTML, module, and CDN checks",
            "findings": FINDINGS,
        }]
        # A task outside the decomposition implement ran last is not named.
        state.record_task_titles({"T006": "Race lifecycle"})
        assert state.implementation_check_findings() == []

    def test_a_task_no_longer_completed_is_not_named(self, tmp_path):
        plan = _plan(tmp_path)
        state = StateManager(plan)
        state.update_task_status("T007", "failed", reason="re-run failed")
        assert state.implementation_check_findings() == []


class TestStatusNamesThem:
    def test_the_workflow_payload_carries_them_on_implement(self, tmp_path):
        plan = _plan(tmp_path)
        entry = _workflow(plan)["phases"]["implement"]["check_findings"]
        assert entry[0]["id"] == "T007" and entry[0]["findings"] == FINDINGS

    def test_no_findings_adds_no_key(self, tmp_path):
        plan = _plan(tmp_path, findings=[])
        assert "check_findings" not in _workflow(plan)["phases"]["implement"]

    def test_the_status_view_names_them_until_review_code_has_run(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        run_status.main(["--plan-file", str(plan)])
        out = capsys.readouterr().out
        assert "check findings for review-code" in out, out
        assert "T007 (2 problem(s): npx html-validate index.html: index.html:1" in out

        reviewed = _plan(tmp_path / "reviewed", completed=UP_TO_CODE_REVIEW)
        run_status.main(["--plan-file", str(reviewed)])
        assert "check findings" not in capsys.readouterr().out

    def test_new_findings_of_an_implement_resume_show_beside_an_older_review(
        self, tmp_path, capsys
    ):
        """Fix round 11 (review finding 5): a review completed before an
        `implement --resume` that recorded new findings hid them until
        implement finished."""
        plan = _plan(tmp_path, completed=UP_TO_CODE_REVIEW)
        state = StateManager(plan)
        state.mark_phase_started("implement")
        state.save()
        workflow = _workflow(plan)
        assert workflow["phases"]["implement"]["status"] == "in_progress"
        assert workflow["phases"]["code-review"]["status"] == "completed"
        run_status.main(["--plan-file", str(plan)])
        assert "check findings for review-code" in capsys.readouterr().out

    def test_a_skipped_review_settles_them(self, tmp_path, capsys):
        """Fix round 11 (review finding 5): a skipped review kept them on
        screen for good."""
        plan = _plan(tmp_path)
        state = StateManager(plan)
        state.mark_phase_skipped("code-review", "not reviewing this one")
        state.save()
        run_status.main(["--plan-file", str(plan)])
        assert "check findings" not in capsys.readouterr().out

    def test_a_terminal_implement_poll_carries_them_into_its_notice(
        self, tmp_path, capsys, monkeypatch
    ):
        """What the codex and opencode tool routes relay: the run-4 poll's
        notice named T008 and nothing about T007's findings."""
        monkeypatch.chdir(tmp_path)
        plan = _plan(tmp_path)
        phase_dir = output_dir_path(plan) / "implement"
        phase_dir.mkdir(parents=True, exist_ok=True)
        status_file.write_status(str(phase_dir), {"phase": "implement", "state": "complete"})
        status_file.claim_run(str(phase_dir), {
            "run_id": "run-check-0001", "state": "done", "exit_code": 0,
            "detached": True, "mode": "implement",
        })
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run", "--phase", "implement",
            ]) == 0
            payload = ctx.payload(0)
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert "check findings for review-code" in payload["notice"]
        assert "T007" in payload["notice"]


class TestImplementsResultLine:
    def test_the_notice_names_each_task_and_its_first_problem(self):
        notice = implement_orchestrator.check_findings_notice({"T007": FINDINGS})
        assert notice.startswith("check findings (review-code is told about them;")
        assert "implement continued" not in notice
        assert f"T007: {FINDINGS[0]} (+1 more)" in notice
        assert "implement/summary.md" in notice


class TestReviewCodeIsTold:
    def test_the_implementation_record_lists_the_findings(self, tmp_path):
        plan = _plan(tmp_path)
        record = cro.build_implementation_record(plan)
        lines = [line for line in record.splitlines() if "Check task T007" in line]
        assert len(lines) == 1, record
        for finding in FINDINGS:
            assert finding in lines[0]
        # A check with findings is not a task "the implement phase was told
        # not to do" nor a gap; it is its own line.
        assert "must not be reported as missing: T007" not in record

    def test_no_findings_no_line(self, tmp_path):
        plan = _plan(tmp_path, findings=[])
        assert "Check task" not in cro.build_implementation_record(plan)


class TestTheNativeLegIsTold:
    def test_the_implement_instructions_pass_the_check_result_to_batch_done(self):
        text = (SKILL_ROOT / "instructions" / "implement.md").read_text(encoding="utf-8")
        assert "--check findings" in text
        assert "--finding" in text
        assert "`Check findings`" in text
        assert "## Check Findings" in text
