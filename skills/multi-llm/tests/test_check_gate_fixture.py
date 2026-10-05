"""The check-gate fixture's shape: a policy stop beside a check with findings.

Fix round 13 (review of round 12, executor leg). The live check-gate fixture
(racing re-run 5) decomposes into T001 → T002 (a check, findings) → T007 and
T001 → T003 → T004 → T005 → T006, with T005 declaring a file
``driver.write_paths`` denies, so implement stops on T005 before dispatching
it. Driven offline against the mock runner, three things went wrong:

* a check whose reply said ``failed`` while listing findings was dispatched
  again on every plain ``--resume`` (its result file kept the runner's
  ``failed``, which resume never counted as done) — pinned here and in
  ``test_check_findings_hold_dependents.py``;
* T005, refused before dispatch, was never recorded ``failed`` — it stayed
  ``pending`` with no reason — so ``--human-task T005=skip`` was refused as
  "a code task that did not fail" and the user had no way past it but to
  change the policy;
* the stop offered only ``--resume``, though T007 waited on T002's findings,
  which ``--resume`` alone never releases on unchanged files.

These tests run the real ``implement`` orchestrator through the fixture's
three stories: the accept path, the fix path and a plain resume.
"""

from __future__ import annotations

import json

import pytest

from tests.test_implement_stop_account import (  # noqa: E402
    _dispatched_titles,
    _implement,
    _statuses,
    make_repo,
)

import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402

#: The fixture's graph. Every edit task patches src/main.py (the mock's one
#: edit); T005 also declares the file the policy denies.
PLAN = """# Check gate

## Overview
The check-gate fixture's shape.

## Tasks

### T001: Edit: create the configuration file
Create the configuration.
- Depends on: none
- Files to modify: src/main.py

### T002: Check the module
Run the release check. Read-only.
- Depends on: T001
- Files to modify: none

### T003: Edit: add the greeting module
Add the greeting.
- Depends on: T001
- Files to modify: src/main.py

### T004: Edit: add the command-line entry point
Add the entry point.
- Depends on: T003
- Files to modify: src/main.py

### T005: Edit: add the build stamp module
Add the stamp.
- Depends on: T004
- Files to modify: src/main.py, src/vendor/stamp.js

### T006: Edit: print the build stamp
Print it.
- Depends on: T005
- Files to modify: src/main.py

### T007: Edit: write the release note
Write the note.
- Depends on: T002
- Files to modify: src/main.py
"""

CHECK_GATE = "Title: Check the module"
FINDINGS = ["retries must be 3 (found 2)"]
DENY = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['**']\n    deny: ['.git/**', '.multi-llm/**', 'src/vendor/**']\n"
)
ALLOW = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['**']\n    deny: ['.git/**', '.multi-llm/**']\n"
)
STAMP = "Edit: add the build stamp module"
NOTE = "Edit: write the release note"
PRINT = "Edit: print the build stamp"


def _policy(repo, text: str) -> None:
    (repo.tmp_path / "driver.yaml").write_text(text, encoding="utf-8")
    registry._config = None
    registry._config_key = None


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


def _check(repo, *, passed: bool, reply_status=None) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": passed, "findings": [] if passed else FINDINGS}))
    if reply_status:
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_STATUS", reply_status)
    else:
        repo.monkeypatch.delenv("MOCK_LLM_WRITE_CHECK_STATUS", raising=False)


def _first_run(repo, capsys, reply_status=None) -> str:
    _check(repo, passed=False, reply_status=reply_status)
    assert _implement(repo) == 3
    captured = capsys.readouterr()
    repo.log.unlink(missing_ok=True)
    return captured.out + captured.err


def _reason(repo, task_id: str) -> str:
    return get_or_create_state(repo.plan).get_task_status_reason(task_id) or ""


class TestThePolicyStop:

    def test_the_refused_task_is_recorded_failed_with_its_reason(self, repo, capsys):
        text = _first_run(repo, capsys)
        statuses = _statuses(repo)
        assert {t: statuses[t] for t in ("T001", "T002", "T003", "T004")} == {
            "T001": "completed", "T002": "completed", "T003": "completed",
            "T004": "completed",
        }
        assert statuses["T005"] == "failed", statuses
        assert _reason(repo, "T005").startswith(
            "Write policy violation (pre-dispatch): src/vendor/stamp.js"
        ), _reason(repo, "T005")
        assert statuses["T006"] == "pending" and "T005 failed" in _reason(repo, "T006")
        assert statuses["T007"] == "pending"
        assert _reason(repo, "T007").startswith(
            "not run — upstream check T002 reported problems")
        assert "[write-subagents] batch 3: T005 refused before dispatch" in text

    def test_the_stop_names_every_answer_it_needs(self, repo, capsys):
        text = _first_run(repo, capsys)
        assert (
            "ERROR: implement stopped at T005 (Edit: add the build stamp "
            "module): refused before it was dispatched" in text
        ), text
        # Nothing ran, so there is nothing half-applied to inspect.
        assert "inspect the half-applied state" not in text
        assert "The half-applied state is deliberately left in place" not in text
        assert "Nothing was half-applied: the stopped task never started" in text
        assert "driver.write_paths" in text and ".multi-llm/providers.yaml" in text
        assert "--human-task T005=skip" in text
        assert "(T005 and the tasks that depend on it, T006, stay undone)" in text
        # T007 waits on T002's findings: the stop names their answers too.
        assert "T002's check reported problems and holds back T007" in text
        for answer in ("T002=done", "T002=skip", "T002=retry"):
            assert f"--human-task {answer}" in text, answer
        assert "--human-task T002=done --human-task T005=skip" in text

    def test_the_result_line_names_both_answers(self, repo, capsys):
        _check(repo, passed=False)
        assert multi_llm.main([
            "implement", str(repo.plan), "--yes", "--exec-subagents",
        ]) == 3
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        error = json.loads(line.split(" ", 1)[1])["error"]
        assert error.startswith(
            "implement stopped at T005 (Edit: add the build stamp module): "
            "refused before it was dispatched"
        ), error
        assert "--human-task T005=skip" in error
        assert "--human-task T002=done" in error
        assert "half-applied" not in error

    def test_any_stop_names_the_answers_of_a_check_that_holds_tasks(
        self, repo, capsys,
    ):
        """Not only the policy stop: T005 allowed but failing for real (a
        clean failure: it edits nothing), while T002's findings hold T007,
        also names T002's three answers."""
        _policy(repo, ALLOW)
        repo.plan.write_text(PLAN.replace(
            "### T005: Edit: add the build stamp module",
            "### T005: Stamp: add the build stamp module",
        ), encoding="utf-8")
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Stamp:")
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_SUMMARY", "could not write the stamp")
        text = _first_run(repo, capsys)
        assert _statuses(repo)["T005"] == "failed"
        assert "ERROR: implement stopped at T005" in text
        assert "refused before it was dispatched" not in text
        assert "--human-task T005=done" in text and "--human-task T005=skip" in text
        assert "T002's check reported problems and holds back T007" in text
        for answer in ("T002=done", "T002=skip", "T002=retry"):
            assert f"--human-task {answer}" in text, answer
        assert "--human-task T002=done --human-task T005=skip" in text
        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        assert report["stop"]["blocked"] == {"T007": "T002"}
        assert "by adding --human-task T002=done" in report["stop"]["headline"]

        rc = _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        )
        capsys.readouterr()
        assert rc == 0
        assert _dispatched_titles(repo) == [NOTE]

    @pytest.mark.parametrize("reply_status", [None, "failed"])
    def test_a_plain_resume_refuses_it_again_and_runs_nothing(
        self, repo, capsys, reply_status,
    ):
        _first_run(repo, capsys, reply_status)
        assert _implement(repo, "--resume") == 3
        text = capsys.readouterr()
        text = text.out + text.err
        assert _dispatched_titles(repo) == [], "nothing changed: nothing runs"
        assert _statuses(repo)["T005"] == "failed"
        assert _reason(repo, "T002") == (
            "not run again: nothing changed since its check reported problems")
        assert "--human-task T005=skip" in text


class TestTheAcceptPath:

    @pytest.mark.parametrize("reply_status", [None, "failed"])
    def test_accept_the_findings_and_skip_the_refused_task(
        self, repo, capsys, reply_status,
    ):
        _first_run(repo, capsys, reply_status)
        rc = _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        )
        text = capsys.readouterr()
        assert rc == 0, text.out[-3000:]
        assert _dispatched_titles(repo) == [NOTE]
        statuses = _statuses(repo)
        assert statuses["T007"] == "completed"
        assert statuses["T005"] == "skipped" and statuses["T006"] == "skipped"
        assert _reason(repo, "T005").endswith("T005=skip after it failed")
        assert "depends on skipped task T005" in _reason(repo, "T006")
        state = get_or_create_state(repo.plan)
        assert state.is_phase_completed("implement")
        assert [e["id"] for e in state.implementation_check_findings()] == ["T002"]

    def test_retry_checks_the_policy_again(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(repo, "--resume", "--human-task", "T005=retry") == 3
        capsys.readouterr()
        assert _dispatched_titles(repo) == []
        assert _statuses(repo)["T005"] == "failed"

        _policy(repo, ALLOW)
        assert _implement(repo, "--resume", "--human-task", "T005=retry") == 3
        capsys.readouterr()
        assert _dispatched_titles(repo) == [STAMP, PRINT]
        statuses = _statuses(repo)
        assert statuses["T005"] == "completed" and statuses["T006"] == "completed"
        assert statuses["T007"] == "pending", "T002's findings still hold it"


class TestTheFixPath:

    @pytest.mark.parametrize("reply_status", [None, "failed"])
    def test_fixed_config_and_policy_run_everything_to_completion(
        self, repo, capsys, reply_status,
    ):
        _first_run(repo, capsys, reply_status)
        _policy(repo, ALLOW)
        main = repo.root / "src" / "main.py"
        main.write_text(main.read_text() + "retries = 3\n", encoding="utf-8")
        _check(repo, passed=True)

        rc = _implement(repo, "--resume")
        text = capsys.readouterr()
        assert rc == 0, text.out[-3000:]
        assert sorted(_dispatched_titles(repo)) == sorted(
            ["Check the module", STAMP, PRINT, NOTE])
        assert _dispatched_titles(repo)[0] == "Check the module"
        assert set(_statuses(repo).values()) == {"completed"}, _statuses(repo)
        state = get_or_create_state(repo.plan)
        assert state.is_phase_completed("implement")
        assert "task_check_findings" not in state.state


class TestTheNativeLeg:
    """The same stop on the native leg (Claude Code): the host gets each
    prompt from `util write-prompt`, which handed T005 out although the
    executor refuses it before dispatch. It refuses it now, the host records
    it failed with `util batch-done`, and the accept answers work there too."""

    def _emit(self, repo, capsys, *flags) -> dict:
        import implement_orchestrator

        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", *flags]
        ) == 0
        out = capsys.readouterr().out
        path = next(
            line.split(": ", 1)[1] for line in out.splitlines()
            if line.startswith("Output file: ")
        )
        return json.loads(open(path, encoding="utf-8").read())

    def _util(self, capsys, *argv):
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_write_prompt_refuses_the_task_the_policy_denies(self, repo, capsys):
        self._emit(repo, capsys)
        phase = str(repo.phase_dir)
        code, _out, err = self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T005")
        assert code == 3
        assert (
            "[WRITE_PROMPT_REFUSED] T005 is refused before dispatch: Write "
            "policy violation (pre-dispatch): src/vendor/stamp.js" in err
        ), err
        assert "--task-id T005 --status failed --reason" in err
        assert "--resume --human-task T005=skip" in err
        assert not (repo.phase_dir / "prompts" / "T005.prompt.txt").exists()
        # A task the policy allows is bound as before.
        assert self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T004")[0] == 0

    def test_the_accept_answers_after_the_refusal_is_recorded(self, repo, capsys):
        self._emit(repo, capsys)
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            assert self._util(
                capsys, "batch-done", "--phase-dir", phase, "--task-id", task_id,
                "--status", "completed")[0] == 0
        assert self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
            "--status", "completed", "--check", "findings",
            "--finding", FINDINGS[0])[0] == 0
        assert self._util(
            capsys, "batch-done", "--phase-dir", phase, "--task-id", "T005",
            "--status", "failed", "--reason", "refused before dispatch")[0] == 0

        handoff = self._emit(
            repo, capsys, "--resume", "--human-task", "T002=done",
            "--human-task", "T005=skip",
        )
        emitted = {t["id"]: t for b in handoff["batches"] for t in b["tasks"]}
        assert "T002" not in emitted, "accepted: not handed out again"
        assert emitted["T005"]["status"] == "skipped"
        assert emitted["T006"]["status"] == "skipped"
        assert emitted["T007"]["status"] == "pending"
        assert self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T007")[0] == 0
