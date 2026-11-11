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


import pytest

from tests.support.transcripts import decision_of, report  # noqa: E402

from utils import report_check  # noqa: E402
from tests.support.stop_answers import RESUME, _check, _native_stop


def _missing_list(reason: str) -> str:
    """The reason's "Missing command(s):" list alone: a reason with a ready
    section asks for the commands through the section after it (fix round
    28), one without names them under "Carry the block's other …"."""
    listed = reason.split("Missing ", 1)[1]
    for end in ("Carry the block's other", report_check.WHOLE_REPORT):
        listed = listed.split(end, 1)[0]
    return listed


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
