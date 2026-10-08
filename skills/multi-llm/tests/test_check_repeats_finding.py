"""A code task that runs a check task's failing check is not blamed for it.

Live run 12 (2026-10-08), OpenCode stop accept 01: T001's runner ran the
release check, ``node scripts/check.mjs`` (T002's check), unprompted. Its
by-design failure was recorded as ``[BATCH_CHECK_FAILED] T001``; the notice
said "not a clean finish" naming T001, ``status`` "own check fails T001" and
summary.md "T001: a check it ran fails", while T002's own stop already
reported the same finding.

Fix round 28, both legs: a code task's own-check finding that is the SAME
finding a check task of the run reported — the existing strict comparison
(``split_accepted_findings``), nothing looser — is recorded as repeating
that check's finding (``batch_check_repeats``), never as the task's own
failure, whichever of the two finished first.
"""

from __future__ import annotations

import json

import pytest

import multi_llm  # noqa: E402
from tests.test_accepted_check_rerun import _result_line, _row  # noqa: E402
from tests.test_check_gate_fixture import repo  # noqa: F401 - `repo` is a fixture
from utils import run_status  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.write_batch_scheduler import (  # noqa: E402
    BATCH_CHECK_MARKER,
    BATCH_CHECK_REPEATED_MARKER,
    implement_checks_ran,
    repeated_findings_for,
    with_result_contract,
)

#: T002's finding (the check task's own record).
CHECKED = "node scripts/check.mjs: check failed: retries must be 3 (found 2)"
#: What a code task's own check reported for the same failure.
SAME = "node scripts/check.mjs exited 1: check failed: retries must be 3 (found 2)"
#: The same command, another failure: the code task's own.
OTHER = "node scripts/check.mjs: check failed: mode must be strict (found loose)"

T001 = "Title: Edit: create the configuration file\n"
T004 = "Title: Edit: add the command-line entry point\n"


def _check_reports(repo) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": [CHECKED]}))


def _code_task_reports(repo, title: str, findings) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps({
        "passed": False, "findings": list(findings),
        "ran": {"node scripts/check.mjs": "failed"}}))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", title)


def _run(repo, capsys) -> str:
    """Run 1 through the dispatcher (its result line): T005 is refused."""
    assert multi_llm.main([
        "implement", str(repo.plan), "--yes", "--exec-subagents",
    ]) == 3
    captured = capsys.readouterr()
    return captured.out + captured.err


def _status_text(repo, capsys, *host) -> str:
    capsys.readouterr()
    run_status.main(["--plan-file", str(repo.plan), *host])
    captured = capsys.readouterr()
    return captured.out + captured.err


# -- the reading -------------------------------------------------------------


class TestTheReading:

    STATE = {
        "task_status": {"T001": "completed", "T002": "completed", "T003": "completed"},
        "task_check_findings": {"T002": [CHECKED]},
    }

    def test_every_check_task_with_findings_on_record_wherever_it_sits(self):
        task = {"task_id": "T001", "task_title": "Create the config", "depends_on": []}
        assert repeated_findings_for([task], task, self.STATE) == {"T002": [CHECKED]}

    def test_never_the_task_itself_nor_a_check_its_own_text_names(self):
        owner = {"task_id": "T003", "task_title": "Fix the release check",
                 "description": "Run `node scripts/check.mjs` and fix what it reports."}
        assert repeated_findings_for([owner], owner, self.STATE) == {}
        check = {"task_id": "T002", "task_title": "Run the release check"}
        assert repeated_findings_for([check], check, self.STATE) == {}

    def test_an_accepted_check_upstream_goes_the_accepted_way(self):
        state = {
            **self.STATE,
            "task_status_reasons": {"T002": "accepted — you answered --human-task T002=done"},
        }
        task = {"task_id": "T007", "task_title": "Write the note", "depends_on": ["T002"]}
        assert repeated_findings_for([task], task, state) == {}

    def test_the_comparison_is_the_strict_one_without_f3s_annotation(self):
        from utils.write_batch_scheduler import split_repeated_findings

        assert split_repeated_findings([SAME, OTHER], {"T002": [CHECKED]}) == (
            [OTHER], {"T002": [SAME]})
        annotated = f"{SAME} (the problem the user accepted from T002)"
        assert split_repeated_findings([annotated], {"T002": [CHECKED]}) == (
            [annotated], {})

    def test_the_tests_line_names_it_apart(self):
        state = {
            "task_status": {"T001": "completed", "T002": "completed"},
            "task_own_check_ran": ["T001"],
            "task_check_findings": {"T002": [CHECKED]},
            "task_batch_check_repeats": {"T001": {"T002": [SAME]}},
        }
        line = implement_checks_ran(state, ["T002"])
        assert line == (
            "The tasks ran checks: reported problems in T002; repeated the "
            "findings of T002 in T001."
        )


# -- which checks are the task's -----------------------------------------------


WHICH = "Which checks to run: the ones this task or its acceptance criteria name"


class TestThePromptSaysWhichChecks:

    def test_an_implement_code_task_is_told(self):
        task = {"task_id": "T001", "expected_ids": ["T001"],
                "files_to_modify": ["src/config.json"], "files": ["src/config.json"]}
        prompt = with_result_contract("implement", task, "Do the task.")
        assert WHICH in prompt
        assert "also a check another task of the plan exists to run" in prompt
        assert "recorded as that task's finding, not as this task's failure" in prompt
        # It never tells a task to leave a check out.
        assert "do not need to run" not in prompt

    def test_a_check_task_and_an_apply_batch_are_not(self):
        check = {"task_id": "T002", "expected_ids": ["T002"]}
        assert WHICH not in with_result_contract("implement", check, "Run it.")
        batch = {"task_id": "apply-fix-batch-0", "expected_ids": ["a1"],
                 "files_to_modify": ["src/x.js"], "files": ["src/x.js"]}
        assert WHICH not in with_result_contract("apply_code_fixes", batch, "Fix it.")

    def test_the_native_prompt_says_it_too(self, repo, capsys):
        import implement_orchestrator

        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        code = multi_llm.main(["util", "write-prompt", "--phase-dir", str(repo.phase_dir),
                               "--task-id", "T001", "--stdout"])
        prompt = capsys.readouterr().out
        assert code == 0 and WHICH in prompt


# -- the executor leg --------------------------------------------------------


class TestTheExecutorLeg:

    def test_the_code_task_recorded_first_is_moved_when_the_check_reports(
        self, repo, capsys,
    ):
        """OpenCode accept 01's order: T001 finishes, then T002 reports."""
        _check_reports(repo)
        _code_task_reports(repo, T001, [SAME])
        text = _run(repo, capsys)
        assert not any(l.startswith(f"{BATCH_CHECK_MARKER} T001")
                       for l in text.splitlines()), text[-3000:]
        line = next(l for l in text.splitlines()
                    if l.startswith(BATCH_CHECK_REPEATED_MARKER))
        assert line.startswith(f"{BATCH_CHECK_REPEATED_MARKER} T001: ")
        assert "only with what check task T002 reported" in line
        assert "not counted as T001's own failure" in line
        assert "not a clean finish" not in str(_result_line(text).get("notice") or "")
        assert "own check fails" not in text
        state = get_or_create_state(repo.plan).state
        assert "T001" not in (state.get("task_batch_check") or {})
        assert state["task_batch_check_repeats"] == {"T001": {"T002": [SAME]}}
        row = _row(repo, "T001")
        assert "batch_check" not in row and row["batch_check_repeats"] == {"T002": [SAME]}
        verdict = json.loads(
            (repo.phase_dir / "implement_batch_T001.json").read_text())["verdict"]
        assert "batch_check" not in verdict
        assert verdict["batch_check_repeats"] == {"T002": [SAME]}
        summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
        assert "a check it ran fails" not in summary
        assert (
            "T001 ran this check too, and its own report says only what T002 "
            "reported: it repeats T002's finding, not counted as T001's own "
            "failure." in summary
        ), summary
        status = _status_text(repo, capsys)
        assert "completed, its check repeated T002's findings: T001" in status, status
        assert "own check fails T001" not in status

    def test_the_code_task_recorded_second_reads_the_checks_record(self, repo, capsys):
        """T004 runs after T002 reported: the split happens at T004's record."""
        _check_reports(repo)
        _code_task_reports(repo, T004, [SAME])
        text = _run(repo, capsys)
        assert not any(l.startswith(f"{BATCH_CHECK_MARKER} T004")
                       for l in text.splitlines()), text[-3000:]
        assert any(l.startswith(f"{BATCH_CHECK_REPEATED_MARKER} T004: ")
                   for l in text.splitlines())
        assert "its own check reported only what check task T002 reported" in text
        state = get_or_create_state(repo.plan).state
        assert state["task_batch_check_repeats"] == {"T004": {"T002": [SAME]}}
        assert "T004" not in (state.get("task_batch_check") or {})

    def test_anything_else_its_check_reports_stays_its_own(self, repo, capsys):
        _check_reports(repo)
        _code_task_reports(repo, T001, [SAME, OTHER])
        text = _run(repo, capsys)
        failed = [l for l in text.splitlines() if l.startswith(f"{BATCH_CHECK_MARKER} T001")]
        assert len(failed) == 1 and "(1 problem(s))" in failed[0] and OTHER in failed[0]
        assert "not a clean finish" in str(_result_line(text).get("notice") or "")
        state = get_or_create_state(repo.plan).state
        assert state["task_batch_check"]["T001"] == [OTHER]
        assert state["task_batch_check_repeats"] == {"T001": {"T002": [SAME]}}

    def test_a_finding_with_words_added_is_not_the_same_finding(self, repo, capsys):
        """The comparison is the strict one: the live T001's commentary
        keeps its finding its own."""
        _check_reports(repo)
        live = f"{SAME} — the task mandates retries 2 while the check expects 3"
        _code_task_reports(repo, T001, [live])
        text = _run(repo, capsys)
        assert any(l.startswith(f"{BATCH_CHECK_MARKER} T001") for l in text.splitlines())
        assert "task_batch_check_repeats" not in get_or_create_state(repo.plan).state


# -- the native leg ----------------------------------------------------------


class TestTheNativeLeg:

    def _util(self, capsys, *argv):
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def _emit(self, repo, capsys) -> str:
        import implement_orchestrator

        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        return str(repo.phase_dir)

    def _own(self, capsys, phase, task_id, *findings):
        argv = ["batch-done", "--phase-dir", phase, "--task-id", task_id,
                "--status", "completed", "--own-check", "findings"]
        for finding in findings:
            argv += ["--finding", finding]
        return self._util(capsys, *argv)

    def _check(self, capsys, phase):
        return self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          "T002", "--status", "completed", "--check", "findings",
                          "--finding", CHECKED)

    def test_the_code_task_recorded_first_is_moved_when_the_check_reports(
        self, repo, capsys,
    ):
        phase = self._emit(repo, capsys)
        code, out, err = self._own(capsys, phase, "T001", SAME)
        assert code == 0, err
        assert any(l.startswith(f"{BATCH_CHECK_MARKER} T001") for l in out.splitlines())
        code, out, err = self._check(capsys, phase)
        assert code == 0, err
        line = next(l for l in out.splitlines() if l.startswith(BATCH_CHECK_REPEATED_MARKER))
        assert line.startswith(f"{BATCH_CHECK_REPEATED_MARKER} T001: ")
        assert "the [BATCH_CHECK_FAILED] line T001's record printed for it no longer stands" in line
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert "T001" not in (state.get("task_batch_check") or {})
        assert state["task_batch_check_repeats"] == {"T001": {"T002": [SAME]}}
        text = _status_text(repo, capsys, "--host", "claude-code")
        assert "completed, its check repeated T002's findings: T001" in text, text
        assert "own check fails T001" not in text

    def test_the_code_task_recorded_second_reads_the_checks_record(self, repo, capsys):
        phase = self._emit(repo, capsys)
        assert self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          "T001", "--status", "completed", "--own-check", "passed")[0] == 0
        assert self._check(capsys, phase)[0] == 0
        assert self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          "T003", "--status", "completed", "--own-check", "passed")[0] == 0
        code, out, err = self._own(capsys, phase, "T004", SAME, OTHER)
        assert code == 0, err
        lines = out.splitlines()
        assert any(l.startswith(f"{BATCH_CHECK_REPEATED_MARKER} T004: ") for l in lines)
        failed = [l for l in lines if l.startswith(f"{BATCH_CHECK_MARKER} T004")]
        assert len(failed) == 1 and OTHER in failed[0] and "(1 problem(s))" in failed[0]
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert state["task_batch_check"]["T004"] == [OTHER]
        assert state["task_batch_check_repeats"] == {"T004": {"T002": [SAME]}}
        # A later record of the task clears it.
        assert self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          "T004", "--status", "completed", "--own-check", "passed")[0] == 0
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert "task_batch_check_repeats" not in state
