"""A released task that runs the check the user accepted (live run 9, #6).

Stop fixture, accept path on OpenCode (`stop-opencode-accept` 02): the user
resumed with `--human-task T002=done --human-task T005=skip`, accepting T002's
check findings ("node scripts/check.mjs: retries must be 3 (found 2)"). T007,
released by that answer, wrote the release note and ran `node scripts/check.mjs`
again while writing it; its own `check` object reported the same failure, so
the run printed `[BATCH_CHECK_FAILED]`, `status` read "own check fails T007"
and the phase "not a clean finish" over a failure the user had just accepted.

Now (fix round 22, both legs):

* the released task's prompt says the accepted problems are known, not its to
  fix and not a failure of its own check, and that it may cite them;
* a code task's own check that reports only what the user accepted for a check
  (the same command, and nothing the accepted finding does not say) is
  recorded as those accepted findings reported again, never as the task's own
  failure — the phase finishes clean and names them; anything else its check
  reports still counts.
"""

from __future__ import annotations

import json

import pytest

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: F401 - `repo` is a fixture
    NOTE,
    _dispatched_prompt,
    _statuses,
    repo,
)
from tests.test_implement_stop_account import _implement  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.write_batch_scheduler import (  # noqa: E402
    BATCH_CHECK_ACCEPTED_MARKER,
    BATCH_CHECK_MARKER,
    accepted_checks_section,
    split_accepted_findings,
)

#: T002's finding as OpenCode's check recorded it in run 9.
ACCEPTED = "node scripts/check.mjs: retries must be 3 (found 2)"
#: What T007's own check reported when it ran the same check again (run 9).
RERUN = "node scripts/check.mjs: check failed: retries must be 3 (found 2) (exit 1)"
#: The Claude Code host's wording of the same T002 finding (run 9, native leg).
NATIVE = (
    "node scripts/check.mjs exited 1 and printed 'check failed: retries must be "
    "3 (found 2)'. src/config.json has retries=2 (set by T001), but the check "
    "requires 3."
)
#: A problem the user never accepted, from the same command.
NEW = "node scripts/check.mjs: mode must be strict (found loose)"

RESUME = ("--resume", "--human-task", "T002=done", "--human-task", "T005=skip")


# -- the match --------------------------------------------------------------


class TestTheMatch:
    """`split_accepted_findings`: the same command, and words the accepted
    finding already says, in its order; nothing else."""

    @pytest.mark.parametrize("accepted, rerun", [
        (ACCEPTED, RERUN),
        (ACCEPTED, ACCEPTED),
        (NATIVE, RERUN),
        (ACCEPTED, "`node scripts/check.mjs` exited with code 1: retries must be 3 (found 2)"),
        (ACCEPTED, "node scripts/check.mjs printed \"check failed: retries must be 3 (found 2)\""),
    ])
    def test_the_accepted_failure_reported_again_is_matched(self, accepted, rerun):
        own, matched = split_accepted_findings([rerun], {"T002": [accepted]})
        assert own == []
        assert matched == {"T002": [rerun]}

    @pytest.mark.parametrize("rerun", [
        # Another command.
        "npm test: retries must be 3 (found 2)",
        # The same command, other numbers.
        "node scripts/check.mjs: retries must be 2 (found 3)",
        "node scripts/check.mjs: retries must be 3 (found 1)",
        # The accepted problem and one more.
        "node scripts/check.mjs: retries must be 3 (found 2); mode must be strict",
        NEW,
        # No command named.
        "retries must be 3 (found 2)",
        # The command failed, and nothing says how.
        "node scripts/check.mjs failed (exit 1)",
        "node scripts/check.mjs: check failed",
        # A check that could not run says nothing about the code.
        "its check could not run: node scripts/check.mjs: not found",
        # Too little of the accepted finding to tell.
        "node scripts/check.mjs: found 2",
    ])
    def test_anything_else_is_the_tasks_own(self, rerun):
        own, matched = split_accepted_findings([rerun], {"T002": [ACCEPTED]})
        assert own == [rerun]
        assert matched == {}

    def test_only_the_new_finding_stays_the_tasks_own(self):
        own, matched = split_accepted_findings([RERUN, NEW], {"T002": [ACCEPTED]})
        assert own == [NEW]
        assert matched == {"T002": [RERUN]}

    def test_nothing_accepted_matches_nothing(self):
        assert split_accepted_findings([RERUN], {}) == ([RERUN], {})

    def test_two_accepted_findings_quoted_together_stay_the_tasks_own(self):
        """Fix round 23 (records #3): a finding matches one accepted finding,
        never several read one after the other, which also let a finding
        straddling two of them match (test_accepted_findings_scope.py)."""
        accepted = {"T002": [
            "npm test: test parses the config failed",
            "npm test: test greets the user failed",
        ]}
        joined = "npm test: test parses the config failed; test greets the user failed"
        own, matched = split_accepted_findings([joined], accepted)
        assert own == [joined] and matched == {}


# -- the prompt -------------------------------------------------------------


def test_the_released_task_is_told_the_failure_is_accepted_and_not_its_own():
    section = accepted_checks_section([("T002", "Run the release check", [ACCEPTED])])
    assert "They are known and accepted: they are not yours to fix" in section
    assert "not a failure of this task's own check" in section
    assert "you may cite them" in section
    assert "You do not need to run that check again" in section
    two = accepted_checks_section([
        ("T002", "", [ACCEPTED]), ("T003", "", ["npm test: 1 failing"]),
    ])
    assert "You do not need to run those checks again" in two


# -- the executor leg -------------------------------------------------------


def _first_stop(repo, capsys) -> None:
    """Run 1: T002's check reports ACCEPTED (with its command), T005 is
    refused by the write policy, T007 waits on T002."""
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": [ACCEPTED]}))
    assert _implement(repo) == 3
    capsys.readouterr()
    repo.log.unlink(missing_ok=True)


def _own_check(repo, findings) -> None:
    """T007's reply carries its own `check` with *findings*."""
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
        {"passed": False, "findings": list(findings)}))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", f"Title: {NOTE}\n")


def _resume(repo) -> int:
    """The accept answers, through the dispatcher (its result line)."""
    return multi_llm.main([
        "implement", str(repo.plan), "--yes", "--exec-subagents", *RESUME,
    ])


def _result_line(text: str) -> dict:
    line = next(
        line for line in reversed(text.splitlines())
        if line.startswith("MULTI_LLM_RESULT: ")
    )
    return json.loads(line.split(": ", 1)[1])


def _row(repo, task_id: str) -> dict:
    report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
    return next(row for row in report["tasks"] if row["task_id"] == task_id)


class TestTheExecutorLeg:

    def test_a_rerun_reporting_only_the_accepted_failure_is_a_clean_finish(
        self, repo, capsys,
    ):
        _first_stop(repo, capsys)
        _own_check(repo, [RERUN])
        rc = _resume(repo)
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert rc == 0, text[-3000:]
        assert _statuses(repo)["T007"] == "completed"
        assert BATCH_CHECK_MARKER not in text, text[-3000:]
        assert "not a clean finish" not in text.lower()
        assert "own check fails" not in text
        line = next(l for l in text.splitlines() if l.startswith(BATCH_CHECK_ACCEPTED_MARKER))
        assert line.startswith(f"{BATCH_CHECK_ACCEPTED_MARKER} T007: ")
        assert "--human-task T002=done" in line and RERUN in line
        assert "not counted as T007's own failure" in line
        assert "not a clean finish" not in str(_result_line(text).get("notice") or "")
        state = get_or_create_state(repo.plan)
        assert "T007" not in (state.state.get("task_batch_check") or {})
        row = _row(repo, "T007")
        assert "batch_check" not in row, row
        assert row["batch_check_accepted"] == {"T002": [RERUN]}
        verdict = json.loads(
            (repo.phase_dir / "implement_batch_T007.json").read_text())["verdict"]
        assert "batch_check" not in verdict
        assert verdict["batch_check_accepted"] == {"T002": [RERUN]}
        summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
        assert "its check fails" not in summary
        section = summary.split("## Check Findings", 1)[1].split("\n## ", 1)[0]
        assert ACCEPTED in section
        assert (
            "T007 ran this check again, and its own report says only what you "
            "accepted: not counted as T007's own failure." in section
        ), section

    def test_the_prompt_says_the_accepted_failure_is_not_its_own(self, repo, capsys):
        _first_stop(repo, capsys)
        _own_check(repo, [RERUN])
        assert _implement(repo, *RESUME) == 0
        capsys.readouterr()
        prompt = _dispatched_prompt(repo, NOTE)
        assert "they are not yours to fix" in prompt
        assert "not a failure of this task's own check" in prompt

    @pytest.mark.parametrize("findings, own", [
        ([RERUN, NEW], [NEW]),
        ([f"{ACCEPTED}; mode must be strict"], [f"{ACCEPTED}; mode must be strict"]),
    ])
    def test_anything_else_its_check_reports_still_counts(
        self, repo, capsys, findings, own,
    ):
        _first_stop(repo, capsys)
        _own_check(repo, findings)
        assert _resume(repo) == 0
        captured = capsys.readouterr()
        text = captured.out + captured.err
        failed = [l for l in text.splitlines() if l.startswith(BATCH_CHECK_MARKER)]
        assert len(failed) == 1 and failed[0].startswith(f"{BATCH_CHECK_MARKER} T007:")
        assert f"({len(own)} problem(s))" in failed[0]
        assert "not a clean finish" in str(_result_line(text).get("notice") or "")
        state = get_or_create_state(repo.plan)
        assert state.state["task_batch_check"]["T007"] == own
        assert _row(repo, "T007")["batch_check"]["findings"] == own


# -- the native leg ---------------------------------------------------------


class TestTheNativeLeg:

    def _util(self, capsys, *argv):
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def _accept(self, repo, capsys) -> str:
        """The native first stop with ACCEPTED, then the accept emit: T007
        is the one task handed out."""
        import implement_orchestrator

        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            assert self._util(capsys, "batch-done", "--phase-dir", phase,
                              "--task-id", task_id, "--status", "completed")[0] == 0
        assert self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          "T002", "--status", "completed", "--check", "findings",
                          "--finding", NATIVE)[0] == 0
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", *RESUME]
        ) == 0
        capsys.readouterr()
        return phase

    def test_batch_done_records_the_accepted_failure_as_accepted(self, repo, capsys):
        phase = self._accept(repo, capsys)
        code, prompt, _err = self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T007", "--stdout")
        assert code == 0
        assert "they are not yours to fix" in prompt
        code, out, err = self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T007",
            "--status", "completed", "--own-check", "findings", "--finding", RERUN)
        assert code == 0, err
        assert BATCH_CHECK_MARKER not in out
        line = next(l for l in out.splitlines() if l.startswith(BATCH_CHECK_ACCEPTED_MARKER))
        assert line.startswith(f"{BATCH_CHECK_ACCEPTED_MARKER} T007: ")
        assert "not counted as T007's own failure" in line
        assert "## Check Findings" in line and "never report the phase as a clean" not in line
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert "T007" not in (state.get("task_batch_check") or {})
        assert state["task_status"]["T007"] == "completed"

    def test_batch_done_still_counts_anything_else(self, repo, capsys):
        phase = self._accept(repo, capsys)
        code, out, err = self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T007",
            "--status", "completed", "--own-check", "findings",
            "--finding", RERUN, "--finding", NEW)
        assert code == 0, err
        failed = [l for l in out.splitlines() if l.startswith(BATCH_CHECK_MARKER)]
        assert len(failed) == 1 and "(1 problem(s))" in failed[0] and NEW in failed[0]
        assert any(l.startswith(BATCH_CHECK_ACCEPTED_MARKER) for l in out.splitlines())
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert state["task_batch_check"]["T007"] == [NEW]
