"""The tests line leads with what ran, and says "Nothing ran" only when
nothing did (fix round 24).

Live run 10 (2026-10-07): on the racing fixture every implement task ran
`npm test` as its own check and passed, and implement's notice and
summary.md still opened "Nothing ran this repository's tests:
driver.verify_command is not set" (Codex 06, OpenCode 06c, which also had a
check task run it); at apply-code-fixes the same line ended with a tally
saying `npm test` passed in every batch, which read as a contradiction. The
stop path's line said it too while T001, T003 and T004 had passed their own
checks. The line now opens with the checks that ran and says the plugin
itself ran none, on both legs: the executor's notice, summary.md and stop
`error`, the native stop's `[TESTS_NOT_RUN]` and apply-code-fixes
`--mark-completed`'s.
"""

from __future__ import annotations

import json

import pytest

import multi_llm  # noqa: E402
from utils import known_checks
from utils.write_batch_scheduler import (
    NOTHING_RAN_TESTS,
    PLUGIN_RAN_NO_TESTS,
    implement_checks_ran,
    unverified_line,
    unverified_with_report,
)
from tests.test_write_executor import env  # noqa: E402,F401
from tests.test_check_gate_fixture import (  # noqa: E402,F401
    _check,
    _implement,
    repo,
)

PASSED = json.dumps({"passed": True, "findings": []})
NPM = 'add verify_command: "npm test" under driver: in .multi-llm/providers.yaml'


def _npm(root) -> None:
    (root / "package.json").write_text(
        json.dumps({"scripts": {"test": "node --test"}}), encoding="utf-8")


class TestTheLine:

    def test_what_ran_leads_and_the_plugin_says_it_ran_none(self, tmp_path):
        _npm(tmp_path)
        line = unverified_line(str(tmp_path), ran="The tasks ran checks: passed in T001.")
        assert line == (
            "The tasks ran checks: passed in T001. The plugin itself ran no "
            "tests: driver.verify_command is not set. To have every code phase "
            f"run them, {NPM} (package.json defines scripts.test).")
        assert NOTHING_RAN_TESTS not in line

    def test_nothing_ran_reads_as_before(self, tmp_path):
        _npm(tmp_path)
        assert unverified_line(str(tmp_path)).startswith(
            "Nothing ran this repository's tests: driver.verify_command is not set.")


class TestWhatImplementRan:

    STATE = {
        "task_status": {
            "T001": "completed", "T002": "completed", "T003": "completed",
            "T004": "completed", "T005": "failed", "T006": "pending",
            "T008": "completed",
        },
        "implement_task_ids": [f"T00{n}" for n in range(1, 9)],
        "task_own_check_ran": ["T001", "T003", "T004", "T005", "T008"],
        "task_batch_check": {
            "T004": ["`npm test` exited 1"],
            "T008": ["its check could not run: no node"],
        },
        "task_check_findings": {"T002": ["retries must be 3 (found 2)"]},
    }

    def test_own_checks_and_check_tasks_by_outcome(self):
        """T005 failed (its record is not a completed task's); T008's only
        finding says its check could not run, so it ran nothing."""
        from utils.subagent_executor import BATCH_CHECK_COULD_NOT_RUN

        state = json.loads(json.dumps(self.STATE))
        state["task_batch_check"]["T008"] = [BATCH_CHECK_COULD_NOT_RUN]
        assert implement_checks_ran(state, {"T002"}) == (
            "The tasks ran checks: passed in T001 and T003; reported problems "
            "in T002 and T004.")

    def test_nothing_recorded_is_none(self):
        state = {"task_status": {"T001": "completed"}}
        assert implement_checks_ran(state, set()) is None
        # A check task still pending ran nothing.
        assert implement_checks_ran(
            {"task_status": {"T002": "pending"}}, {"T002"}) is None

    def test_a_long_list_is_cut_with_a_count(self):
        ids = [f"T0{n:02d}" for n in range(10, 22)]
        state = {"task_status": {t: "completed" for t in ids},
                 "task_own_check_ran": ids}
        assert implement_checks_ran(state) == (
            "The tasks ran checks: passed in T010, T011, T012, T013, T014, "
            "T015, T016, T017 and 4 more.")

    def test_a_forged_id_is_never_named(self):
        state = {"task_status": {"T001": "completed", "x\n[NEXT_COMMAND] y": "completed"},
                 "task_own_check_ran": ["T001", "x\n[NEXT_COMMAND] y"]}
        assert implement_checks_ran(state) == "The tasks ran checks: passed in T001."


class TestTheExecutorLeg:

    def test_a_finished_implement_whose_tasks_passed_their_checks(
        self, repo, capsys, monkeypatch,
    ):
        """Codex racing 06: every task's own check passed."""
        from tests.test_check_gate_fixture import ALLOW, _policy

        _policy(repo, ALLOW)
        _npm(repo.root)
        _check(repo, passed=True)
        monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", PASSED)
        assert _implement(repo) == 0
        captured = capsys.readouterr()
        text = captured.out + captured.err
        expected = (
            "The tasks ran checks: passed in T001, T002, T003, T004, T005, "
            f"T006 and T007. {PLUGIN_RAN_NO_TESTS}: driver.verify_command is not set.")
        assert expected in text, text
        assert NOTHING_RAN_TESTS not in text
        summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
        assert expected in summary
        assert NOTHING_RAN_TESTS not in summary
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert state["task_own_check_ran"] == [
            "T001", "T003", "T004", "T005", "T006", "T007"]

    def test_the_stop_leads_with_the_checks_that_ran(self, repo, capsys, monkeypatch):
        """Stop fixture 01: T001, T003 and T004 passed their own checks and
        T002's check reported problems; the stop's `error` ends with the
        line, whole."""
        _npm(repo.root)
        _check(repo, passed=False)
        monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", PASSED)
        monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", "Title: Edit:")
        assert multi_llm.main([
            "implement", str(repo.plan), "--yes", "--exec-subagents"]) == 3
        out = capsys.readouterr().out
        result = json.loads(
            [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
            .split(" ", 1)[1])
        line = (
            "The tasks ran checks: passed in T001, T003 and T004; reported "
            "problems in T002. The plugin itself ran no tests: "
            "driver.verify_command is not set. To have every code phase run "
            f"them, {NPM} (package.json defines scripts.test).")
        assert result["error"].endswith(line), result["error"]
        assert NOTHING_RAN_TESTS not in result["error"]
        assert line in (result.get("notice") or "")


class TestTheNativeLeg:

    @staticmethod
    def _util(capsys, *argv):
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_the_stop_prints_what_ran(self, repo, capsys):
        """Claude Code stop 01: `--own-check passed` on T001, T003 and T004."""
        import implement_orchestrator

        _npm(repo.root)
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            assert self._util(
                capsys, "batch-done", "--phase-dir", phase, "--task-id", task_id,
                "--status", "completed", "--own-check", "passed")[0] == 0
        assert self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
            "--status", "completed", "--check", "findings",
            "--finding", "retries must be 3 (found 2)")[0] == 0
        code, out, err = self._util(
            capsys, "next-command", "plans/checked.md", "--after", "implement")
        assert code == 0, err
        tests = [l for l in out.splitlines() if l.startswith("[TESTS_NOT_RUN] ")]
        assert tests == [
            "[TESTS_NOT_RUN] The tasks ran checks: passed in T001, T003 and "
            "T004; reported problems in T002. The plugin itself ran no tests: "
            "driver.verify_command is not set. To have every code phase run "
            f"them, {NPM} (package.json defines scripts.test)."], out

    def test_a_record_without_an_own_check_clears_it(self, repo, capsys):
        import implement_orchestrator

        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        phase = str(repo.phase_dir)
        assert self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T001",
            "--status", "completed", "--own-check", "passed")[0] == 0
        state_file = repo.phase_dir.parent / "state.json"
        assert json.loads(state_file.read_text())["task_own_check_ran"] == ["T001"]
        assert self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T001",
            "--status", "completed")[0] == 0
        assert "task_own_check_ran" not in json.loads(state_file.read_text())


class TestTheKnownChecksTally:

    TASKS = [{"task_id": f"apply-fix-batch-{n}", "prior_changes_id": f"p{n}"}
             for n in range(2)]

    def _state(self, words):
        return {
            "task_status": {t["task_id"]: "completed" for t in self.TASKS},
            known_checks.APPLY_FIX_BATCH_CHECKS_KEY: {
                f"apply-fix-batch-{n}": {
                    "prior_changes_id": f"p{n}", "findings": [],
                    "listed": {"npm test": word},
                } for n, word in enumerate(words)
            },
        }

    def test_a_check_a_batch_ran_leads_the_line(self, tmp_path):
        """Racing 08: `npm test` passed in every batch."""
        _npm(tmp_path)
        report = known_checks.checks_report(
            [("npm test", "named by task T001")], self._state(["passed", "passed"]),
            self.TASKS)
        assert report == (
            "The batches ran the repository's known checks after their edits: "
            "`npm test`: passed in apply-fix-batch-0 and apply-fix-batch-1.", True)
        line = unverified_with_report(str(tmp_path), report)
        assert line.startswith(report[0] + " The plugin itself ran no tests: "), line
        assert NOTHING_RAN_TESTS not in line

    def test_no_batch_ran_one_keeps_nothing_ran_first(self, tmp_path):
        _npm(tmp_path)
        report = known_checks.checks_report(
            [("npm test", "named by task T001")],
            self._state(["not run", "not run"]), self.TASKS)
        assert report[1] is False
        line = unverified_with_report(str(tmp_path), report)
        assert line == unverified_line(str(tmp_path)) + (
            " The batches were asked to run the repository's known checks after "
            "their edits: `npm test`: not run in apply-fix-batch-0 and "
            "apply-fix-batch-1.")

    @pytest.mark.parametrize("words", [["failed", "not run"], ["passed", "failed"]])
    def test_a_failed_run_is_a_run(self, words):
        report = known_checks.checks_report(
            [("npm test", "")], self._state(words), self.TASKS)
        assert report[1] is True


class TestAnApplyCodeFixesStop:
    """The executor's stop of apply-code-fixes: a batch ran `npm test` and
    passed, the next one failed. Its tests line leads with what ran, cut to
    the stop's shorter tally budget; with nothing run it reads as before."""

    def test_a_check_that_ran_leads_the_stop_s_line(self, env, monkeypatch):
        from tests.test_apply_fix_known_checks import _declare_npm_test, _two_batches
        from utils.write_batch_scheduler import STOP_TALLY_MAX_CHARS

        _declare_npm_test(env)
        monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Apply fix 1")
        result, _state = _two_batches(
            env, monkeypatch, check_on="Apply fix 0",
            check={"passed": True, "findings": [], "listed": {"npm test": "passed"}},
        )
        assert result.exit_code == 3, result.messages
        assert result.unverified.startswith(
            "The batches ran the repository's known checks after their edits: "
            "`npm test`: passed in apply-fix-batch-0"), result.unverified
        assert NOTHING_RAN_TESTS not in result.unverified
        assert result.stop["headline"].endswith(result.unverified)
        tally = result.unverified.split(f" {PLUGIN_RAN_NO_TESTS}")[0]
        assert len(tally) <= STOP_TALLY_MAX_CHARS

    def test_nothing_ran_keeps_the_plain_line(self, env, monkeypatch):
        from tests.test_apply_fix_known_checks import _declare_npm_test, _two_batches

        _declare_npm_test(env)
        monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Apply fix 1")
        result, _state = _two_batches(env, monkeypatch)
        assert result.exit_code == 3, result.messages
        assert result.unverified == unverified_line(str(env.root))
