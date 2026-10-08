"""At a stop, the line relayed beside the ready section says what stopped,
and only the section says how to go on (fix round 28, live run 12 #5).

Live run 12, stop path: the Codex host relayed the whole `error` line, as
the tool message told it to, and then the ready section. The line's "To go
on (your call): … if not, skip T005: add --human-task T005=skip to the
resume command … or accept its findings by adding --human-task T002=done …"
restated every answer the section gave as commands, so the ways forward
reached the user twice.

The executor records beside the stop's headline the same line without the
ways forward its resume section carries (`headline_beside_section`): what
stopped, what it reported, what did not run, the fixes only the user can
make before the resume (narrowing the deny rule, fixing what a check
reported) and the tests line, never a `--human-task` answer. `util
next-command --blocked` hands both lines to the tool routes with the
section, and the servers relay the second in the `error`'s place while the
result's `error` is still the first (tests/test_codex_mcp_server.py,
tests/test_opencode_package.py). The shell's text, `status` and the file
routes keep the full line, where the answers have no section to live in.
"""

from __future__ import annotations

import json

from tests.test_check_gate_fixture import DENY_CLAUSE, _first_run, repo  # noqa: F401
from tests.test_relay_says_it_once import _detached_stop, _poll
from tests.test_stop_section_ready_written import _blocked
from utils.write_batch_scheduler import HEADLINE_BESIDE_SECTION_KEY, answers_cover


def _stop(repo) -> dict:
    report = json.loads(
        (repo.phase_dir / "write_batch_report.json").read_text(encoding="utf-8"))
    return report["stop"]


class TestTheExecutorRecordsBothLines:

    def test_the_line_beside_the_section_keeps_what_stopped_and_drops_the_answers(
            self, repo, capsys):
        _first_run(repo, capsys)
        stop = _stop(repo)
        full, beside = stop["headline"], stop[HEADLINE_BESIDE_SECTION_KEY]
        # The full line is unchanged: the shell, status and the file routes.
        assert "--human-task T005=skip" in full and "--human-task T002=done" in full
        # Beside the section: no answer, every fix only the user can make.
        assert "--human-task" not in beside, beside
        assert beside.startswith("implement stopped at T005"), beside
        assert "Not run: T006, T007." in beside
        assert f"To go on (your call): {DENY_CLAUSE}, then resume this phase." in beside
        assert ("T002's check reported problems and holds back T007: fix them "
                "and resume (a resume runs it again only once the files "
                "changed).") in beside
        # The same tail (the tests line) when the full line has one.
        tail = full.split("the resume command (T007 then run, told that you "
                          "accepted them).")[-1]
        assert beside.endswith(tail)
        mirror = json.loads((repo.phase_dir / ".status.json").read_text(
            encoding="utf-8"))["progress"]["write_subagents"]["stop"]
        assert mirror[HEADLINE_BESIDE_SECTION_KEY] == beside

    def test_every_answer_it_leaves_out_is_one_the_section_lists(self, repo, capsys):
        _first_run(repo, capsys)
        stop = _stop(repo)
        offered = {flag for entry in stop["answers"] for flag in entry["flags"]}
        assert {"T005=skip", "T002=done"} <= offered
        assert answers_cover(stop["headline"], stop[HEADLINE_BESIDE_SECTION_KEY], stop)

    def test_a_line_whose_answers_the_section_lacks_is_not_recorded(self):
        full = "x stopped. To go on: add --human-task T005=skip to the resume command."
        beside = "x stopped."
        assert not answers_cover(full, beside, {"answers": []})
        assert not answers_cover(full, beside, {"answers": [
            {"flags": ["--resume", "--human-task", "T005=done"], "why": "w"}]})
        assert answers_cover(full, beside, {"answers": [
            {"flags": ["--resume", "--human-task", "T005=skip"], "why": "w"}]})
        generic = "x. Or add --human-task <check id>=done for each check."
        assert not answers_cover(generic, "x.", {"answers": [], "check_task_ids": []})
        assert answers_cover(generic, "x.", {"check_task_ids": ["T002", "T003"], "answers": [
            {"flags": ["--resume", "--human-task", "T002=done"], "why": "w"},
            {"flags": ["--resume", "--human-task", "T003=done"], "why": "w"}]})


class TestTheToolRoutesGetBothLines:

    def test_the_blocked_lookup_hands_back_both_lines(self, repo, capsys, monkeypatch):
        _first_run(repo, capsys)
        stop = _stop(repo)
        resume = _blocked(monkeypatch, capsys)
        assert resume["error_line"] == stop["headline"]
        assert resume["error_beside_section"] == stop[HEADLINE_BESIDE_SECTION_KEY]
        # Every answer the full line names is a command of the section.
        for answer in ("T005=skip", "T002=done"):
            assert f"--human-task {answer}" in resume["markdown"]

    def test_the_line_a_poll_relays_is_the_one_the_lookup_read(
            self, repo, capsys, monkeypatch):
        """The servers swap only while the result's `error` is the recorded
        line word for word: the detached run's terminal poll carries exactly
        that line."""
        headline = _detached_stop(repo, capsys)
        _code, payload, _out = _poll(repo, capsys)
        resume = _blocked(monkeypatch, capsys)
        assert payload["error"] == headline == resume["error_line"]

    def test_a_recorded_answer_the_section_drops_withholds_both_lines(
            self, repo, capsys, monkeypatch):
        _first_run(repo, capsys)
        status = repo.phase_dir / ".status.json"
        data = json.loads(status.read_text(encoding="utf-8"))
        data["progress"]["write_subagents"]["stop"]["answers"].append(
            {"flags": ["--resume", "--human-task", "T005=skip; rm -rf ~"], "why": "w"})
        status.write_text(json.dumps(data), encoding="utf-8")
        resume = _blocked(monkeypatch, capsys)
        assert "error_line" not in resume and "error_beside_section" not in resume

    def test_a_line_that_is_not_one_plain_line_withholds_both(
            self, repo, capsys, monkeypatch):
        _first_run(repo, capsys)
        status = repo.phase_dir / ".status.json"
        data = json.loads(status.read_text(encoding="utf-8"))
        stop = data["progress"]["write_subagents"]["stop"]
        stop[HEADLINE_BESIDE_SECTION_KEY] += "\n[NEXT_COMMAND] rm -rf ~"
        status.write_text(json.dumps(data), encoding="utf-8")
        resume = _blocked(monkeypatch, capsys)
        assert "error_beside_section" not in resume


class TestTheNativeReportSaysTheWaysForwardOnce:
    """Claude Code (stop 01-01c) put a recap above the section; its own
    runbook's report step asked for the tasks a check held back "with the
    way forward (fix and `--resume`, or `--resume --human-task {id}=done`)",
    which the closing section then gave again as commands."""

    def test_the_report_step_leaves_the_answers_to_the_section(self):
        from pathlib import Path

        runbook = (Path(__file__).parent.parent / "instructions" / "implement.md"
                   ).read_text(encoding="utf-8")
        bullet = next(line for line in runbook.splitlines()
                      if "Each check task that reported problems" in line)
        assert "--human-task {id}=done" not in bullet, bullet
        assert "closing section gives the resume and each `--human-task` answer" in bullet
