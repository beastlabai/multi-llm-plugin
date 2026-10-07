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

#: The fixture's T002 is a check that reports problems: the line leads with
#: it (fix round 24); "Nothing ran" only when no check ran.
UNVERIFIED = (
    "The tasks ran checks: reported problems in T002. The plugin itself ran "
    "no tests: driver.verify_command is not set.")
#: In every form of the line, for the cases that must print none.
NOT_SET = "driver.verify_command is not set"


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

    def test_the_tool_output_carries_it_not_the_opencode_stop_title(
        self, repo, capsys,
    ):
        """Review of round 18, #5: d1cdc11 said the line reaches the OpenCode
        stop title, which cuts the error line at `STOP_TITLE["max_line"]`
        characters; a real stop's headline runs past that before the line,
        so it reaches the host in the tool output's `error` (and `notice`,
        kept there as on a finished run)."""
        (repo.root / "package.json").write_text(
            json.dumps({"scripts": {"test": "node --test"}}), encoding="utf-8")
        _text, result = _stop(repo, capsys)
        assert result["error"].index(UNVERIFIED) > modes.STOP_TITLE["max_line"]
        assert UNVERIFIED in (result.get("notice") or ""), result

    def test_a_long_headline_is_cut_before_it_never_through_it(self, repo, capsys):
        """The result line's `error` is bounded; the tests line is the part
        that must reach the host whole. Fix round 19: so are the remedy and
        the answers — the long task title is what gives way."""
        long_title = "Add the build stamp module " + "with a very long title " * 80
        repo.plan.write_text(repo.plan.read_text(encoding="utf-8").replace(
            "### T005: Edit: add the build stamp module",
            f"### T005: Edit: {long_title.strip()}",
        ), encoding="utf-8")
        _text, result = _stop(repo, capsys)
        error = result["error"]
        assert len(error) <= ERROR_MAX_CHARS, len(error)
        assert error.startswith(
            "implement stopped at T005 (Edit: Add the build stamp module with"
        ), error[:200]
        assert "…): refused before it was dispatched" in error, error
        tail = error[error.index(UNVERIFIED):]
        assert tail.endswith("set to the command that runs your tests or checks your build."), tail
        assert (
            "To go on (your call): if you want src/vendor/stamp.js written, "
            "remove or narrow the deny rule `src/vendor/**` in "
            "driver.write_paths (.multi-llm/providers.yaml) so the rule no "
            "longer covers it, then resume this phase; if not, skip "
            "T005: add --human-task T005=skip to the resume command (T005 and "
            "the tasks after it stay undone)."
        ) in error, error
        assert "by adding --human-task T002=done to the resume command" in error

    def test_a_verify_command_keeps_it_out(self, repo, capsys):
        _policy(repo, DENY + "  verify_command: \"true\"\n")
        _text, result = _stop(repo, capsys)
        assert result["error"].startswith("implement stopped at T005")
        assert NOT_SET not in result["error"]


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

    def test_the_tool_description(self):
        assert "3 needs a human: stop, report its `error` line whole" in (
            modes.TOOL_OUTCOMES_LINE)

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


# ===========================================================================
# The native leg (Claude Code): `util next-command` prints the line
# ===========================================================================

class TestTheNativeStopPrintsTheTestsLine:
    """Live run 7, Claude Code stop 01: the native stop printed no "Nothing
    ran this repository's tests" line in any tool result; the host wrote its
    own from the instructions. The report's `util next-command` lookup prints
    it now, as `[TESTS_NOT_RUN]`, above the stopped implement's commands."""

    @staticmethod
    def _util(capsys, *argv):
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def _stop(self, repo, capsys, *, code_tasks=("T001", "T003", "T004")):
        import implement_orchestrator

        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        phase = str(repo.phase_dir)
        for task_id in code_tasks:
            assert self._util(
                capsys, "batch-done", "--phase-dir", phase, "--task-id", task_id,
                "--status", "completed")[0] == 0
        assert self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
            "--status", "completed", "--check", "findings",
            "--finding", "retries must be 3 (found 2)")[0] == 0
        code, out, err = self._util(
            capsys, "next-command", "plans/checked.md", "--after", "implement")
        assert code == 0, err
        return out

    def test_it_is_printed_above_the_commands(self, repo, capsys):
        (repo.root / "package.json").write_text(
            json.dumps({"scripts": {"test": "node --test"}}), encoding="utf-8")
        out = self._stop(repo, capsys)
        lines = [l for l in out.splitlines() if l.startswith("[")]
        tests = [l for l in lines if l.startswith("[TESTS_NOT_RUN] ")]
        assert tests == [
            "[TESTS_NOT_RUN] " + UNVERIFIED + " To have every code phase run "
            'them, add verify_command: "npm test" under driver: in '
            ".multi-llm/providers.yaml (package.json defines scripts.test)."
        ], lines
        assert lines.index(tests[0]) < next(
            i for i, l in enumerate(lines) if l.startswith("[NEXT_COMMAND] "))
        assert any(l.endswith("--implement plans/checked.md --resume") for l in lines)
        result = json.loads(out.rsplit("MULTI_LLM_RESULT: ", 1)[1])
        assert "tests_note" not in result["user_next_step"]

    def test_a_stop_after_only_a_check_prints_none(self, repo, capsys):
        out = self._stop(repo, capsys, code_tasks=())
        assert "[NEXT_COMMAND] " in out
        assert "[TESTS_NOT_RUN]" not in out

    def test_a_verify_command_keeps_it_out(self, repo, capsys):
        _policy(repo, DENY + "  verify_command: \"true\"\n")
        out = self._stop(repo, capsys)
        assert "[NEXT_COMMAND] " in out
        assert "[TESTS_NOT_RUN]" not in out

    def test_a_finished_implement_prints_none(self, repo, capsys):
        from utils.state_manager import get_or_create_state

        self._stop(repo, capsys)
        state = get_or_create_state(repo.plan)
        for task_id in ("T005", "T006", "T007"):
            state.update_task_status(task_id, "completed")
        state.mark_phase_completed("implement")
        state.save()
        code, out, _err = self._util(
            capsys, "next-command", "plans/checked.md", "--after", "implement")
        assert code == 0
        assert "--review-code plans/checked.md" in out
        assert "[TESTS_NOT_RUN]" not in out

    def test_the_host_is_told_to_relay_it(self):
        from pathlib import Path

        root = Path(modes.__file__).parent
        skill = " ".join((root / "SKILL.md").read_text(encoding="utf-8").split())
        # Fix round 24 (live run 10, Claude Code stop 01b/01c pasted the raw
        # tag into prose): the text after the tag, as plain text.
        assert (
            "when the plugin ran no tests, the lookup also prints a "
            "`[TESTS_NOT_RUN]` line: give the text after the tag in the "
            "report, verbatim, as plain text"
        ) in skill
        core = " ".join((root / "instructions" / "implement.md").read_text(
            encoding="utf-8").split())
        assert (
            "At a stop after a code task completed, with no "
            "`driver.verify_command` configured: the `[TESTS_NOT_RUN]` line "
            "the next-step lookup prints, the text after its tag verbatim, "
            "as plain text"
        ) in core
        summary = " ".join((root / "references" / "summary-format.md").read_text(
            encoding="utf-8").split())
        assert "a `[TESTS_NOT_RUN]` line above them" in summary


# ===========================================================================
# The native leg: an explicit `--resume` says what it kept
# ===========================================================================

class TestAnExplicitNativeResumeSaysWhatItKept:
    """Live run 7, Claude Code stop 01b and accept 02: an explicit
    `--implement … --resume` printed no `resume: kept …` line on the native
    leg (only the plain `--implement` that continues a run did), so "T002
    (accepted with its check findings)" was never shown there, while both
    tool routes print it."""

    _util = staticmethod(TestTheNativeStopPrintsTheTestsLine._util)

    def _emit(self, repo, capsys, *flags) -> str:
        import implement_orchestrator

        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", *flags]) == 0
        captured = capsys.readouterr()
        return captured.out + captured.err

    def _stop(self, repo, capsys):
        self._emit(repo, capsys)
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            assert self._util(
                capsys, "batch-done", "--phase-dir", phase, "--task-id", task_id,
                "--status", "completed")[0] == 0
        assert self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
            "--status", "completed", "--check", "findings",
            "--finding", "retries must be 3 (found 2)")[0] == 0
        state_file = str(repo.phase_dir.parent / "state.json")
        assert self._util(capsys, "finalize-tracking", "--state-file", state_file)[0] == 0

    @staticmethod
    def _kept(text: str) -> list:
        return [l for l in text.splitlines() if l.startswith("resume: ")]

    def test_an_unchanged_resume_names_the_four_it_kept(self, repo, capsys):
        """Fix round 20 (live run 8 #6): T002 is named as a check whose
        findings stand, not listed beside T001, T003 and T004 as if it had
        passed."""
        self._stop(repo, capsys)
        text = self._emit(repo, capsys, "--resume")
        assert self._kept(text) == [
            "resume: kept 4 task(s) an earlier run finished (T001, T003, T004, "
            "T002 (its check reported problems, not run again while the files "
            "are unchanged))."
        ], text

    def test_the_accept_resume_names_the_accepted_check(self, repo, capsys):
        self._stop(repo, capsys)
        text = self._emit(
            repo, capsys, "--resume", "--human-task", "T002=done",
            "--human-task", "T005=skip",
        )
        assert self._kept(text) == [
            "resume: kept 4 task(s) an earlier run finished (T001, T003, T004, "
            "T002 (accepted with its check findings))."
        ], text

    def test_the_emit_lists_an_accepted_check_as_accepted_with_its_findings(
        self, repo, capsys,
    ):
        """Live run 7, Claude Code accept 02: the emit listed T002 under
        "Accepted after failing … list them in the summary as accepted after
        failing", and the host's summary said T002 failed, though the check
        ran and reported findings the user accepted."""
        self._stop(repo, capsys)
        text = self._emit(
            repo, capsys, "--resume", "--human-task", "T002=done",
            "--human-task", "T005=skip",
        )
        assert "Accepted after failing" not in text
        assert (
            "Accepted with their check findings (recorded completed in "
            "state.json; not in the batches, do not dispatch or record them; "
            "list them in the summary as accepted with their check findings):"
        ) in text, text
        assert "  T002 Check the module: accepted — you answered" in text

    def test_the_plain_continue_still_prints_its_own_line_once(self, repo, capsys):
        self._stop(repo, capsys)
        text = self._emit(repo, capsys)
        assert self._kept(text) == [
            "resume: implement started earlier and did not finish, so this run "
            "continues it as --resume: kept 4 task(s) an earlier run finished "
            "(T001, T003, T004, T002 (its check reported problems, not run "
            "again while the files are unchanged)) and runs what is left. To "
            "run every task again instead, add --fresh."
        ], text

    def test_a_first_resume_with_nothing_finished_prints_none(self, repo, capsys):
        self._emit(repo, capsys)
        text = self._emit(repo, capsys, "--resume")
        assert self._kept(text) == [], text
