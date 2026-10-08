"""The Claude Code Stop hook accepts every correct report of a native stop.

Fix round 21 (review of round 20, stops #4). Round 20 made the native stop's
`util next-command` print each answer alone beside the combined one, and the
Stop hook (`util report-check`) holds the report to every command printed.
These drive the real lookup on the check-gate fixture and hand its output to
the real check, so the shapes the hook sees are the ones the plugin prints:

* one task waiting (T005, refused): two answers and no combined command,
  so a report carrying the resume and the two answers passes, in any order;
* six answers (run 8's first stop): the report passes with them in any
  order, or all in one fenced block, and is never asked for a combined
  command the lookup did not print.

The instruction texts said "first the one command carrying every answer,
then each alone", which reads as a command the one-task stop never prints;
they now say the combined command comes first only when more than one task
waits (`test_check_gate_fixture.TestTheNativeLeg`).
"""

from __future__ import annotations

import json

import pytest

from tests.test_check_gate_fixture import (  # noqa: E402
    CHECK_GATE,
    DENY,
    FINDINGS,
    PLAN,
    _policy,
)
from tests.test_implement_stop_account import make_repo  # noqa: E402
from tests.test_report_check import (  # noqa: E402
    Transcript,
    decision_of,
    report,
    run_check,
)

import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import report_check  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402

RESUME = "/multi-llm:multi-llm --implement plans/checked.md --resume"


def _missing_list(reason: str) -> str:
    """The reason's "Missing command(s):" list alone: a reason with a ready
    section asks for the commands through the section after it (fix round
    28), one without names them under "Carry the block's other …"."""
    listed = reason.split("Missing ", 1)[1]
    for end in ("Carry the block's other", report_check.WHOLE_REPORT):
        listed = listed.split(end, 1)[0]
    return listed


@pytest.fixture
def repo(tmp_path, monkeypatch):
    obj = make_repo(tmp_path, monkeypatch, PLAN)
    _policy(obj, DENY)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        [f"T00{n}" for n in range(1, 8)]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    yield obj
    registry._config = None
    registry._config_key = None


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _native_stop(repo, capsys, *, check: str) -> str:
    """The native first stop (T005 refused by the emit), T002's check
    recorded *check*; returns `util next-command`'s output."""
    import implement_orchestrator

    assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
    capsys.readouterr()
    phase = str(repo.phase_dir)
    for task_id in ("T001", "T003", "T004"):
        assert _util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                     task_id, "--status", "completed")[0] == 0
    extra = (["--check", "findings", "--finding", FINDINGS[0]]
             if check == "findings" else ["--check", "passed"])
    assert _util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                 "--status", "completed", *extra)[0] == 0
    state_file = str(get_or_create_state(repo.plan).state_file)
    assert _util(capsys, "finalize-tracking", "--state-file", state_file)[0] == 0
    code, out, err = _util(capsys, "next-command", "plans/checked.md",
                           "--after", "implement")
    assert code == 0, err
    return out


def _check(tmp_path, capsys, output: str, message: str) -> str:
    path = (Transcript().prompt("implement it").bash(output).say(message)
            .write(tmp_path / "t.jsonl"))
    return run_check(capsys, path)


class TestOneWaitingTask:
    """T002 passed: only T005 waits, so no combined command is printed."""

    def test_the_lookup_prints_two_answers_and_no_combined_one(self, repo, capsys):
        out = _native_stop(repo, capsys, check="passed")
        block = report_check.find_blocks(out)[-1]
        assert block.command == RESUME
        assert [command for _why, command in block.answers] == [
            f"{RESUME} --human-task T005=skip",
            f"{RESUME} --human-task T005=done",
        ]

    @pytest.mark.parametrize("order", ["printed", "reversed"])
    def test_a_report_with_the_resume_and_both_answers_passes(
        self, repo, capsys, tmp_path, order,
    ):
        out = _native_stop(repo, capsys, check="passed")
        answers = [command for _why, command in report_check.find_blocks(out)[-1].answers]
        if order == "reversed":
            answers = answers[::-1]
        assert _check(tmp_path, capsys, out, report(RESUME, *answers)) == ""

    def test_a_report_missing_one_answer_is_asked_for_that_one_alone(
        self, repo, capsys, tmp_path,
    ):
        out = _native_stop(repo, capsys, check="passed")
        reason = decision_of(_check(
            tmp_path, capsys, out, report(RESUME, f"{RESUME} --human-task T005=skip"),
        ))["reason"]
        missing = _missing_list(reason)
        assert f"```\n{RESUME} --human-task T005=done\n```" in missing
        assert "--human-task T005=skip" not in missing
        assert "carrying every answer" not in reason


class TestSixAnswers:
    """Run 8's first stop: T002's findings hold T007, T005 refused."""

    def _answers(self, out: str) -> list:
        return [command for _why, command in report_check.find_blocks(out)[-1].answers]

    def test_any_order_passes(self, repo, capsys, tmp_path):
        out = _native_stop(repo, capsys, check="findings")
        answers = self._answers(out)
        assert len(answers) == 6
        shuffled = [answers[i] for i in (4, 1, 5, 0, 3, 2)]
        assert _check(tmp_path, capsys, out, report(*shuffled, RESUME)) == ""

    def test_all_in_one_fenced_block_passes(self, repo, capsys, tmp_path):
        out = _native_stop(repo, capsys, check="findings")
        message = "## Summary\n\nStopped.\n\n```\n" + "\n".join(
            [RESUME, *self._answers(out)]) + "\n```\n"
        assert _check(tmp_path, capsys, out, message) == ""

    def test_only_what_the_lookup_printed_is_asked_for(self, repo, capsys, tmp_path):
        """The six answers without the resume: only the resume is missing."""
        out = _native_stop(repo, capsys, check="findings")
        reason = decision_of(_check(
            tmp_path, capsys, out, report(*self._answers(out)),
        ))["reason"]
        missing = _missing_list(reason)
        assert f"```\n{RESUME}\n```" in missing
        assert "--human-task" not in missing
