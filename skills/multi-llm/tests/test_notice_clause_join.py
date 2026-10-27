"""A notice joins its clauses with one mark between them (fix round 28,
live run 12 #7).

Live run 12 (2026-10-08):

* Codex decline and accept 02, OpenCode 08: the result's notice read
  ``… (package.json defines scripts.test).; Opponent speed exceeds …`` — the
  tests line keeps its closing stop, and a bare ``"; ".join`` put the
  separator after it.
* OpenCode decline: the notice's line for the declined fix ended
  ``… may exceed MAX_SPEED'`` without the reason's closing stop, which
  summary.md and state.json keep: the line had dropped it to make room for
  a separator, and nothing followed it.

Every notice join goes through ``text_clip.join_clauses``: a clause gives up
its own closing stop to the separator, and keeps it where it ends the line.
"""

from __future__ import annotations

import json
import types

import pytest

from implement_orchestrator import check_findings_notice
from utils import cli_common, status_file
from utils.text_clip import drop_said_clauses, join_clauses
from utils.write_batch_scheduler import write_result_notice

TESTS_LINE = (
    "The tasks ran checks: passed in T001. The plugin itself ran no tests: "
    "driver.verify_command is not set. To have every code phase run them, add "
    'verify_command: "npm test" under driver: in .multi-llm/providers.yaml '
    "(package.json defines scripts.test)."
)
DECLINED_REASON = (
    "the plan's Decisions keep opponent speed unclamped, a decision the "
    "implementation record pins with the test 'a trailing opponent may exceed "
    "MAX_SPEED'."
)


class TestJoinClauses:

    @pytest.mark.parametrize("parts, joined", [
        (["a sentence.", "b"], "a sentence; b"),
        (["a sentence.", "b."], "a sentence; b."),
        (["one;", "two"], "one; two"),
        (["ends in an ellipsis...", "next"], "ends in an ellipsis...; next"),
        (["cut short…", "next"], "cut short…; next"),
        (["(a bracket.)", "next"], "(a bracket.); next"),
        (["", None, "  ", "only."], "only."),
        ([], ""),
        (None, ""),
    ])
    def test_a_clause_gives_its_stop_to_the_separator_only(self, parts, joined):
        assert join_clauses(parts) == joined

    def test_the_last_clause_keeps_its_closing_stop(self):
        assert join_clauses([TESTS_LINE, DECLINED_REASON]).endswith("MAX_SPEED'.")
        assert ".;" not in join_clauses([TESTS_LINE, DECLINED_REASON])


class TestAClauseSaidAlreadyIsNotSaidAgain:
    """Fable review of D2 (2026-10-27): a composite joins its phases' notices,
    and implement and apply-code-fixes both end theirs with the same tests
    line, which a stop's error may carry too."""

    CHECK = 'not a clean finish: 1 batch(es) applied their edits, but ... ("pytest").'

    def test_a_part_that_repeats_one_clause_keeps_the_rest(self):
        assert drop_said_clauses(join_clauses([self.CHECK, TESTS_LINE]), TESTS_LINE) == (
            self.CHECK)

    def test_a_repeated_part_is_dropped_whole_and_a_new_one_kept_as_it_is(self):
        earlier = join_clauses([DECLINED_REASON, TESTS_LINE])
        assert drop_said_clauses(TESTS_LINE, earlier) == ""
        assert drop_said_clauses(self.CHECK, earlier) == self.CHECK
        assert drop_said_clauses(self.CHECK, None) == self.CHECK

    def test_the_kept_clauses_end_their_sentence(self):
        assert drop_said_clauses(
            join_clauses([TESTS_LINE, "one more"]), TESTS_LINE) == "one more."


class TestTheWorkersOwnNotice:

    def test_a_code_phases_notice_has_no_stop_before_its_separator(self):
        result = types.SimpleNamespace(
            batch_check_findings=lambda: {"T004": ["npm test fails."]},
            unverified=TESTS_LINE,
        )
        notice = write_result_notice(result)
        assert ".;" not in notice, notice
        assert notice.endswith("(package.json defines scripts.test)."), notice
        assert '("npm test fails."); The tasks ran checks' in notice, notice

    def test_two_checks_findings_join_without_a_double_mark(self):
        notice = check_findings_notice({
            "T002": ["scripts/check.mjs:11 expects 3, while src/config.json:1 sets 2."],
            "T005": ["lint reported 2 problems."],
        })
        assert ".;" not in notice, notice
        assert "sets 2; T005: lint reported 2 problems." in notice, notice

    def test_a_note_appended_to_a_notice_takes_the_same_join(self):
        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            cli_common.record_notice(TESTS_LINE)
            cli_common.add_notice(
                "Opponent speed exceeds MAX_SPEED [c8fe7191] (fix 1 of "
                f"apply-fix-batch-0) not applied: the batch declined it — "
                f"{DECLINED_REASON}")
            notice = context.notice
        finally:
            cli_common.deactivate_result()
        assert ".;" not in notice, notice
        assert "(package.json defines scripts.test); Opponent speed" in notice
        assert notice.endswith("MAX_SPEED'."), notice


# ===========================================================================
# A declined reason that ends the notice keeps its stop (the real orchestrator)
# ===========================================================================

from tests.test_write_batch_declined import _PairRig  # noqa: E402
from tests.test_write_executor_orchestrators import repo  # noqa: E402,F401


class TestADeclinedReasonEndingTheNoticeKeepsItsStop(_PairRig):

    def test_the_notice_and_the_notes_keep_the_reasons_closing_stop(
            self, repo, capsys):
        plan = self._review_pair(repo)
        ids = self._ids(plan)
        declined_id, applied_id = ids[0]
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_DECLINED", json.dumps(
            [{"id": declined_id, "reason": DECLINED_REASON}]))
        capsys.readouterr()
        context = cli_common.activate_result(cli_common.ResultContext(
            mode="apply-suggestions", phase="apply-suggestions"))
        try:
            assert self._run(repo, plan, [applied_id] + ids[1]) == 0
            notice = context.notice or ""
        finally:
            cli_common.deactivate_result()
        assert notice.endswith(f"the batch declined it — {DECLINED_REASON}"), notice
        notes = json.loads(
            (self._phase_dir(plan) / ".status.json").read_text())["progress"]["notes"]
        assert any(note.endswith(DECLINED_REASON) for note in notes), notes


# ===========================================================================
# A terminal poll's notice (the route the tool servers read)
# ===========================================================================

from tests.test_relay_says_it_once import _detached_stop, _poll  # noqa: E402
from tests.test_check_gate_fixture import repo as gate_repo  # noqa: E402,F401


class TestAPollsNoticeJoinsItsClausesOnce:

    def test_no_clause_keeps_its_stop_before_the_separator(self, gate_repo, capsys):
        repo = gate_repo
        headline = _detached_stop(repo, capsys)
        status_path = repo.phase_dir / ".status.json"
        mirror = json.loads(status_path.read_text(encoding="utf-8"))["progress"]
        tests_line = mirror["write_subagents"]["unverified"]
        assert tests_line.endswith("."), tests_line
        note = ("Opponent speed exceeds MAX_SPEED [c8fe7191] (fix 1 of "
                f"apply-fix-batch-0) not applied: the batch declined it — "
                f"{DECLINED_REASON}")
        # The error cut short of the tests line, so the notice carries it,
        # then the run's own note, then the workflow's trailer.
        status_file.update_status(str(repo.phase_dir), {
            "run": {"error": headline.replace(tests_line, "").strip()},
            "progress": {"notes": [note]},
        })
        _code, payload, _out = _poll(repo, capsys)
        notice = payload.get("notice") or ""
        assert tests_line[:-1] in notice and note[:-1] in notice, notice
        assert ".;" not in notice, notice
