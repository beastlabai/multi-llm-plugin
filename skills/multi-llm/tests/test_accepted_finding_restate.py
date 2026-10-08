"""The backstop for an accepted finding reported with words added (fix round 28).

The prompt asks a released task for the accepted finding copied exactly and
its own words in the check's ``note`` (``test_accepted_finding_note.py``).
A task that copies the finding and still adds words to it — the residue of
live run 12's two notes — is asked once to restate it: on the executor leg
by the existing report re-ask, on the native leg by a ``[FINDING_RESTATE]``
line naming the ``util batch-done`` call. Neither leg ever records such a
finding as accepted itself: the strict match (the user's F3, unchanged)
decides on what the restatement says, and a restatement may only cut a
finding down to the accepted one, word for word.
"""

from __future__ import annotations

import json
import shlex

import pytest

import multi_llm  # noqa: E402
from tests.test_accepted_check_rerun import RESUME, _result_line, _row  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: F401 - `repo` is a fixture
    NOTE,
    repo,
)
from tests.test_implement_stop_account import _implement  # noqa: E402
from utils.batch_done import FINDING_RESTATE_MARKER  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.subagent_executor import merge_restated_check  # noqa: E402
from utils.subagent_prompts import build_schema_repair_prompt  # noqa: E402
from utils.write_batch_scheduler import (  # noqa: E402
    BATCH_CHECK_ACCEPTED_MARKER,
    BATCH_CHECK_MARKER,
    accepted_findings_with_words_added,
    restate_findings_problem,
    split_accepted_findings,
    words_added_to,
)

ACCEPTED = "node scripts/check.mjs: check failed: retries must be 3 (found 2)"
ADDED = f"{ACCEPTED} (known problem the user accepted in T002)"
MORE = f"{ACCEPTED}; mode must be strict (found loose)"
#: Live run 12's Claude Code finding against that run's longer T002 text:
#: it holds no accepted finding whole, so the backstop does not fire.
LIVE_T002 = (
    "node scripts/check.mjs exited 1 and printed 'check failed: retries must be "
    "3 (found 2)'. src/config.json has retries=2, as T001 specified, but the "
    "check expects 3."
)
LIVE_T007 = (
    "node scripts/check.mjs exited 1: 'check failed: retries must be 3 (found "
    "2)' (known problem the user accepted in T002)"
)


# -- the trigger -------------------------------------------------------------


class TestTheTrigger:

    @pytest.mark.parametrize("finding", [ADDED, MORE, f"Again: {ACCEPTED}"])
    def test_an_accepted_finding_held_word_for_word_with_words_added(self, finding):
        assert accepted_findings_with_words_added(
            [finding], {"T002": [ACCEPTED]}) == [(finding, "T002", ACCEPTED)]

    @pytest.mark.parametrize("finding, accepted", [
        (ACCEPTED, ACCEPTED),
        ("  " + ACCEPTED.replace(" ", "  ") + " ", ACCEPTED),
        (LIVE_T007, LIVE_T002),
        ("node scripts/check.mjs: check failed: retries must be 3 (found 1)", ACCEPTED),
        ("its check could not run: " + ACCEPTED, ACCEPTED),
    ])
    def test_anything_else_is_not_asked_about(self, finding, accepted):
        assert accepted_findings_with_words_added([finding], {"T002": [accepted]}) == []

    def test_it_never_moves_the_match(self):
        """The strict match reads the flagged finding as the task's own."""
        assert split_accepted_findings([ADDED], {"T002": [ACCEPTED]}) == ([ADDED], {})

    @pytest.mark.parametrize("finding, words", [
        (ADDED, "known problem the user accepted in T002"),
        (MORE, "mode must be strict (found loose)"),
        (f"Again: {ACCEPTED}", "Again"),
        (f"{ACCEPTED} — the accepted T002 problem (retries 2), noted",
         "the accepted T002 problem (retries 2), noted"),
    ])
    def test_the_words_added(self, finding, words):
        assert words_added_to(finding, ACCEPTED) == words

    def test_the_problem_names_both_ways_to_answer(self):
        problem = restate_findings_problem("T007", [(ADDED, "T002", ACCEPTED)])
        assert json.dumps(ADDED) in problem and json.dumps(ACCEPTED) in problem
        assert "restate that finding as the accepted finding alone" in problem
        assert "keep the finding exactly as it is" in problem

    def test_the_reask_lets_only_the_finding_and_note_change(self):
        prompt = build_schema_repair_prompt(
            "implement", "the batch", ["the problem"], previous_reply='{"status": "applied"}',
            expected_ids=["T007"], restate_findings=True,
        )
        assert "has a check finding\nthat needs restating" in prompt
        assert "a finding may be cut down to the\naccepted finding it names" in prompt
        assert "the words you cut go, as they are, in the `check` object's `note`" in prompt
        assert "include that entire object unchanged" not in prompt


# -- what a restatement may change -------------------------------------------


ORIGINAL = {"passed": False, "findings": [ADDED], "ran": {"node scripts/check.mjs": "failed"}}


class TestTheMerge:

    def test_a_finding_cut_down_to_the_accepted_one_is_taken_with_its_note(self):
        note = "(Known problem the user accepted in T002.)"
        merged = merge_restated_check(
            ORIGINAL, {"passed": False, "findings": [ACCEPTED], "note": note}, [ACCEPTED])
        assert merged["findings"] == [ACCEPTED] and merged["note"] == note
        assert merged["ran"] == ORIGINAL["ran"] and merged["passed"] is False

    @pytest.mark.parametrize("note", [None, "known", "the accepted one"])
    def test_words_cut_and_not_moved_to_the_note_keep_the_finding(self, note):
        """Nothing a finding said leaves the record: a cut whose words are
        not in the note is no restatement."""
        restated = {"passed": False, "findings": [ACCEPTED]}
        if note is not None:
            restated["note"] = note
        merged = merge_restated_check(ORIGINAL, restated, [ACCEPTED])
        assert merged["findings"] == [ADDED]

    @pytest.mark.parametrize("restated", [
        {"passed": True, "findings": []},
        {"passed": False, "findings": []},
        {"passed": False, "findings": ["node scripts/check.mjs: all fine"]},
        {"passed": False, "findings": [ACCEPTED.replace("(found 2)", "(found 1)")]},
        {"passed": False},
        "not an object",
    ])
    def test_anything_else_keeps_the_first_reports_findings(self, restated):
        merged = merge_restated_check(ORIGINAL, restated, [ACCEPTED])
        assert merged["findings"] == [ADDED] and merged["passed"] is False

    def test_a_finding_it_adds_is_kept(self):
        extra = "node scripts/check.mjs: mode must be strict (found loose)"
        merged = merge_restated_check(
            {**ORIGINAL, "findings": [MORE]},
            {"findings": [ACCEPTED, extra],
             "note": "split in two: mode must be strict (found loose)"}, [ACCEPTED])
        assert merged["findings"] == [ACCEPTED, extra]
        own, matched = split_accepted_findings(merged["findings"], {"T002": [ACCEPTED]})
        assert own == [extra] and matched == {"T002": [ACCEPTED]}


# -- the executor leg --------------------------------------------------------


def _stop_with_accepted(repo, capsys) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": [ACCEPTED]}))
    assert _implement(repo) == 3
    capsys.readouterr()
    repo.log.unlink(missing_ok=True)


def _t007_reports(repo, finding: str, restated: dict) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps({
        "passed": False, "findings": [finding], "ran": {"node scripts/check.mjs": "failed"}}))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", f"Title: {NOTE}\n")
    repo.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps({
        "status": "applied", "ids": ["T007"], "files_touched": ["src/main.py"],
        "summary": "wrote the release note", "check": restated,
    }))


def _resume(repo, capsys) -> str:
    assert multi_llm.main([
        "implement", str(repo.plan), "--yes", "--exec-subagents", *RESUME,
    ]) == 0
    captured = capsys.readouterr()
    return captured.out + captured.err


def _reasks(repo) -> list:
    return [
        json.loads(line).get("prompt") or "" for line in repo.log.read_text().splitlines()
        if "You are re-sending the report of a batch of edits." in (
            json.loads(line).get("prompt") or "")
    ]


class TestTheExecutorLeg:

    def test_a_restatement_down_to_the_accepted_finding_is_accepted(self, repo, capsys):
        _stop_with_accepted(repo, capsys)
        _t007_reports(repo, ADDED, {
            "passed": False, "findings": [ACCEPTED],
            "note": "known problem the user accepted in T002",
            "ran": {"node scripts/check.mjs": "failed"}})
        text = _resume(repo, capsys)
        reasks = _reasks(repo)
        assert len(reasks) == 1, text[-3000:]
        assert json.dumps(ADDED) in reasks[0] and json.dumps(ACCEPTED) in reasks[0]
        assert "its own check reports a finding the user accepted with words added" in text
        assert BATCH_CHECK_MARKER not in text, text[-3000:]
        line = next(l for l in text.splitlines() if l.startswith(BATCH_CHECK_ACCEPTED_MARKER))
        assert '(its note: "known problem the user accepted in T002")' in line
        assert "not a clean finish" not in str(_result_line(text).get("notice") or "")
        assert _row(repo, "T007")["batch_check_accepted"] == {"T002": [ACCEPTED]}
        state = get_or_create_state(repo.plan).state
        assert "T007" not in (state.get("task_batch_check") or {})

    def test_a_kept_finding_stays_the_tasks_own(self, repo, capsys):
        _stop_with_accepted(repo, capsys)
        _t007_reports(repo, MORE, {
            "passed": False, "findings": [MORE], "ran": {"node scripts/check.mjs": "failed"}})
        text = _resume(repo, capsys)
        assert len(_reasks(repo)) == 1
        failed = [l for l in text.splitlines() if l.startswith(BATCH_CHECK_MARKER)]
        assert len(failed) == 1 and failed[0].startswith(f"{BATCH_CHECK_MARKER} T007:")
        assert get_or_create_state(repo.plan).state["task_batch_check"]["T007"] == [MORE]

    def test_a_restatement_that_drops_the_failure_changes_nothing(self, repo, capsys):
        _stop_with_accepted(repo, capsys)
        _t007_reports(repo, ADDED, {"passed": True, "findings": [], "ran": {}})
        text = _resume(repo, capsys)
        assert any(l.startswith(f"{BATCH_CHECK_MARKER} T007:") for l in text.splitlines())
        assert get_or_create_state(repo.plan).state["task_batch_check"]["T007"] == [ADDED]

    def test_an_exact_copy_is_never_asked_about(self, repo, capsys):
        _stop_with_accepted(repo, capsys)
        _t007_reports(repo, ACCEPTED, {"passed": True})
        text = _resume(repo, capsys)
        assert _reasks(repo) == []
        assert any(l.startswith(BATCH_CHECK_ACCEPTED_MARKER) for l in text.splitlines())


# -- the native leg ----------------------------------------------------------


class TestTheNativeLeg:

    def _util(self, capsys, *argv):
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def _accept(self, repo, capsys) -> str:
        import implement_orchestrator

        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        capsys.readouterr()
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            assert self._util(capsys, "batch-done", "--phase-dir", phase,
                              "--task-id", task_id, "--status", "completed")[0] == 0
        assert self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                          "T002", "--status", "completed", "--check", "findings",
                          "--finding", ACCEPTED)[0] == 0
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", *RESUME]) == 0
        capsys.readouterr()
        return phase

    def test_the_line_gives_the_call_that_records_it_restated(self, repo, capsys):
        phase = self._accept(repo, capsys)
        code, out, err = self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T007",
            "--status", "completed", "--own-check", "findings", "--finding", ADDED)
        assert code == 0, err
        lines = out.splitlines()
        assert any(l.startswith(f"{BATCH_CHECK_MARKER} T007:") for l in lines)
        restate = next(l for l in lines if l.startswith(FINDING_RESTATE_MARKER))
        assert "holds the finding the user accepted for T002 word for word" in restate
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert state["task_batch_check"]["T007"] == [ADDED]
        call = restate.split("the words as its note: ", 1)[1].split(" (that record", 1)[0]
        argv = shlex.split(call)
        assert argv[:2] == ["util", "batch-done"]
        assert argv[argv.index("--finding") + 1] == ACCEPTED
        assert argv[argv.index("--note") + 1] == "known problem the user accepted in T002"
        code, out, err = self._util(capsys, *argv[1:])
        assert code == 0, err
        assert any(l.startswith(BATCH_CHECK_ACCEPTED_MARKER) for l in out.splitlines())
        assert BATCH_CHECK_MARKER not in out and FINDING_RESTATE_MARKER not in out
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert "T007" not in (state.get("task_batch_check") or {})
        assert state["task_own_check_note"] == {
            "T007": "known problem the user accepted in T002"}

    def test_no_line_for_a_finding_that_holds_no_accepted_one_whole(self, repo, capsys):
        phase = self._accept(repo, capsys)
        code, out, _err = self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T007",
            "--status", "completed", "--own-check", "findings", "--finding",
            ACCEPTED.replace("(found 2)", "(found 1)") + " (new)")
        assert code == 0
        assert FINDING_RESTATE_MARKER not in out
        assert any(l.startswith(f"{BATCH_CHECK_MARKER} T007:") for l in out.splitlines())
