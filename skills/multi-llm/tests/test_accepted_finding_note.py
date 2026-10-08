"""An accepted finding is reported one exact way, and a note goes beside it.

Live run 12 (2026-10-08), stop accept 02 on Claude Code and OpenCode: T007,
released by ``--human-task T002=done``, re-ran T002's release check and
wrote its comment into the finding itself:

* Claude Code: ``node scripts/check.mjs exited 1: 'check failed: retries
  must be 3 (found 2)' (known problem the user accepted in T002)``;
* OpenCode: ``… — the accepted T002 problem (src/config.json has retries 2;
  check requires 3), not caused by this note``.

The strict match (fix rounds 22/23, the user's F3 in ``eb93ee3``) read both
as T007's own failure: ``[BATCH_CHECK_FAILED] T007``, "not a clean finish".
The prompt had said "you may cite them (for example, in a note on the
check's outcome)" and named no place for that note.

Fix round 28, the matcher unchanged: the released task's prompt gives one
exact way to report a repeat (the accepted finding copied character for
character, nothing added) and a ``note`` field for the words around it; both
legs record that note and show it, and never read it as a finding.
"""

from __future__ import annotations

import json

import pytest

import multi_llm  # noqa: E402
from tests.test_accepted_check_rerun import (  # noqa: E402
    RESUME,
    _first_stop,
    _result_line,
    _row,
)
from tests.test_check_gate_fixture import (  # noqa: F401 - `repo` is a fixture
    NOTE,
    _dispatched_prompt,
    repo,
)
from tests.test_implement_stop_account import _implement  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.subagent_executor import batch_check_note  # noqa: E402
from utils.write_batch_scheduler import (  # noqa: E402
    BATCH_CHECK_ACCEPTED_MARKER,
    BATCH_CHECK_MARKER,
    accepted_checks_section,
    split_accepted_findings,
    with_result_contract,
)

#: T002's finding as the run-12 OpenCode check recorded it.
ACCEPTED = (
    "`node scripts/check.mjs` exited 1 and printed 'check failed: retries must "
    "be 3 (found 2)' — src/config.json:1 has \"retries\": 2 while "
    "scripts/check.mjs:13 requires 3."
)
#: The two live run-12 findings, words added to the check's report.
CLAUDE_CODE_LIVE = (
    "node scripts/check.mjs exited 1: 'check failed: retries must be 3 (found "
    "2)' (known problem the user accepted in T002)"
)
OPENCODE_LIVE = (
    "`node scripts/check.mjs` exited 1 and printed 'check failed: retries must "
    "be 3 (found 2)' — the accepted T002 problem (src/config.json has retries "
    "2; check requires 3), not caused by this note"
)
COMMENT = "the accepted T002 problem: src/config.json has retries 2"


# -- the match stays as it is ------------------------------------------------


class TestTheMatchIsUnchanged:
    """The user's decision: no fuzzy matching. The live notes stay the task's
    own; the accepted finding copied as asked is accepted."""

    @pytest.mark.parametrize("finding", [CLAUDE_CODE_LIVE, OPENCODE_LIVE])
    def test_a_finding_with_words_added_stays_the_tasks_own(self, finding):
        assert split_accepted_findings([finding], {"T002": [ACCEPTED]}) == (
            [finding], {})

    def test_the_accepted_finding_copied_exactly_is_accepted(self):
        assert split_accepted_findings([ACCEPTED], {"T002": [ACCEPTED]}) == (
            [], {"T002": [ACCEPTED]})


# -- the prompt --------------------------------------------------------------


class TestThePromptGivesOneWay:

    def test_the_accepted_section_asks_for_the_finding_copied_exactly(self):
        section = accepted_checks_section([("T002", "Run the release check", [ACCEPTED])])
        assert "you may cite them" not in section
        assert "in a note on the check's outcome" not in section
        assert "copied exactly as it is quoted below" in section
        assert "character for character), with nothing added before or after it" in section
        assert "in the `check` object's `note`, never in the finding" in section
        assert "a finding with words added" in section
        # The finding is still quoted as data, once.
        assert json.dumps(ACCEPTED, ensure_ascii=False) in section

    def test_every_code_batch_is_told_where_its_own_words_go(self):
        task = {"task_id": "T007", "expected_ids": ["T007"],
                "files_to_modify": ["docs/release.md"], "files": ["docs/release.md"]}
        prompt = with_result_contract("implement", task, "Do the task.")
        assert "the command and what it reported, nothing else" in prompt
        assert "- `note` (optional): anything you want to say about your checks" in prompt
        assert "never read as a finding" in prompt


# -- the executor leg --------------------------------------------------------


def _own_check(repo, check: dict) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(check))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", f"Title: {NOTE}\n")


def _resume(repo) -> int:
    return multi_llm.main([
        "implement", str(repo.plan), "--yes", "--exec-subagents", *RESUME,
    ])


class TestTheExecutorLeg:

    def test_a_note_is_recorded_and_shown_and_never_a_finding(self, repo, capsys):
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
            {"passed": False, "findings": [ACCEPTED]}))
        assert _implement(repo) == 3
        capsys.readouterr()
        repo.log.unlink(missing_ok=True)
        _own_check(repo, {
            "passed": False, "findings": [ACCEPTED],
            "ran": {"node scripts/check.mjs": "failed"}, "note": COMMENT,
        })
        assert _resume(repo) == 0
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert BATCH_CHECK_MARKER not in text, text[-3000:]
        line = next(l for l in text.splitlines()
                    if l.startswith(BATCH_CHECK_ACCEPTED_MARKER))
        assert line.endswith(f'(its note: "{COMMENT}")'), line
        assert "not a clean finish" not in str(_result_line(text).get("notice") or "")
        state = get_or_create_state(repo.plan).state
        assert state["task_own_check_note"] == {"T007": COMMENT}
        assert "T007" not in (state.get("task_batch_check") or {})
        assert _row(repo, "T007")["batch_check_note"] == COMMENT
        verdict = json.loads(
            (repo.phase_dir / "implement_batch_T007.json").read_text())["verdict"]
        assert verdict["batch_check_note"] == COMMENT
        summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
        assert f"Its note: {COMMENT}." in summary

    def test_a_failing_checks_line_quotes_its_note_after_the_findings(self, repo, capsys):
        _first_stop(repo, capsys)
        own = "node scripts/check.mjs: mode must be strict (found loose)"
        _own_check(repo, {"passed": False, "findings": [own], "note": "new in T007"})
        assert _resume(repo) == 0
        captured = capsys.readouterr()
        text = captured.out + captured.err
        failed = next(l for l in text.splitlines() if l.startswith(BATCH_CHECK_MARKER))
        assert failed.endswith('(its note: "new in T007")'), failed
        summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
        assert "- its note (not a finding): new in T007" in summary

    @pytest.mark.parametrize("check, note", [
        ({"passed": True, "note": "  all\n good  "}, "all good"),
        ({"passed": True, "note": 7}, None),
        ({"passed": True}, None),
        ({"passed": True, "note": "x" * 900}, "x" * 299 + "…"),
    ])
    def test_the_note_reader(self, check, note):
        assert batch_check_note({"check": check}) == note


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

    def test_the_released_tasks_prompt_names_the_one_way(self, repo, capsys):
        phase = self._accept(repo, capsys)
        code, prompt, _err = self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T007", "--stdout")
        assert code == 0
        assert "copied exactly as it is quoted below" in prompt
        assert "- `note` (optional)" in prompt

    def test_batch_done_records_the_note_beside_the_accepted_finding(self, repo, capsys):
        phase = self._accept(repo, capsys)
        code, out, err = self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T007",
            "--status", "completed", "--own-check", "findings",
            "--finding", ACCEPTED, "--note", COMMENT)
        assert code == 0, err
        assert BATCH_CHECK_MARKER not in out
        line = next(l for l in out.splitlines() if l.startswith(BATCH_CHECK_ACCEPTED_MARKER))
        assert f'(its note: "{COMMENT}")' in line
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert state["task_own_check_note"] == {"T007": COMMENT}
        assert "T007" not in (state.get("task_batch_check") or {})
        # Any later record of the task without a note clears it.
        code, _out, err = self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T007",
            "--status", "completed", "--own-check", "passed")
        assert code == 0, err
        state = json.loads((repo.phase_dir.parent / "state.json").read_text())
        assert "task_own_check_note" not in state

    def test_a_note_without_own_check_is_refused_and_nothing_recorded(self, repo, capsys):
        phase = self._accept(repo, capsys)
        before = (repo.phase_dir.parent / "state.json").read_text()
        code, _out, err = self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T007",
            "--status", "completed", "--note", COMMENT)
        assert code == 3
        assert "--note is recorded only with --own-check" in err
        assert (repo.phase_dir.parent / "state.json").read_text() == before
