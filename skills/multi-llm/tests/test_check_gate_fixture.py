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
from pathlib import Path

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


def _dispatched_prompt(repo, title: str) -> str:
    """The last prompt the mock runner was handed for the task titled *title*."""
    found = ""
    for line in repo.log.read_text().splitlines():
        prompt = json.loads(line).get("prompt") or ""
        if f"Title: {title}\n" in prompt:
            found = prompt
    assert found, f"no dispatch of {title!r} in the mock log"
    return found


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
        # Fix round 14: what `=done` does to the held task, in a few words.
        assert "(T007 then run, told that you accepted the findings" in text

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

    @pytest.mark.parametrize("reply_status", [None, "failed"])
    def test_the_released_task_is_told_the_findings_were_accepted(
        self, repo, capsys, reply_status,
    ):
        """Fix round 14 (rerun 5, every host's accept path): T007's prompt
        showed T002's failure and nothing said the user had accepted it, so
        every runner refused T007 and implement stopped again."""
        _first_run(repo, capsys, reply_status)
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        capsys.readouterr()
        prompt = _dispatched_prompt(repo, NOTE)
        assert "## Check findings the user accepted" in prompt
        assert (
            "T002, a check this task depends on, reported problems, and the "
            "user accepted them (`--human-task T002=done`)" in prompt
        ), prompt
        assert "do not wait for that check to pass" in prompt
        assert "say in your `summary` what you left out and why" in prompt
        assert f'  - "{FINDINGS[0]}"' in prompt
        # After the prior-batch context, before the task's instructions.
        assert prompt.index("## Check findings the user accepted") < prompt.index(
            "## Result (required")

    def test_the_dispatched_prompts_are_kept_beside_the_native_legs(
        self, repo, capsys,
    ):
        """Run 6 (#10b): on the executor leg (both tool routes) the prompt a
        task was dispatched with was in no plan artifact, so T007's prompt
        with the accepted findings could not be audited. It is kept where the
        native leg's `util write-prompt` writes it, byte for byte what the
        runner got, and the audit line's sha256 verifies it."""
        import hashlib

        from utils.write_safety import AUDIT_FILENAME, safety_dir

        prompts = repo.phase_dir / "prompts"
        _first_run(repo, capsys)
        assert sorted(p.name for p in prompts.iterdir()) == [
            "T001.prompt.txt", "T002.prompt.txt", "T003.prompt.txt",
            "T004.prompt.txt",
        ], "T005 was refused before dispatch: no prompt is kept for it"
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        capsys.readouterr()
        kept = (prompts / "T007.prompt.txt").read_text(encoding="utf-8")
        assert kept == _dispatched_prompt(repo, NOTE)
        assert "## Check findings the user accepted" in kept
        audit = Path(safety_dir(str(repo.phase_dir))) / AUDIT_FILENAME
        dispatched = [
            json.loads(line) for line in audit.read_text().splitlines()
            if '"dispatched"' in line
        ]
        last = [r for r in dispatched if r["task_id"] == "T007"][-1]
        assert last["prompt_sha256"] == hashlib.sha256(kept.encode()).hexdigest()

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
        assert "Check findings the user accepted" not in _dispatched_prompt(repo, NOTE)
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

    def _emit_text(self, repo, capsys, *flags):
        import implement_orchestrator

        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", *flags]
        ) == 0
        out = capsys.readouterr().out
        path = next(
            line.split(": ", 1)[1] for line in out.splitlines()
            if line.startswith("Output file: ")
        )
        return json.loads(open(path, encoding="utf-8").read()), out

    @staticmethod
    def _counts(out: str) -> tuple:
        dispatch = next(
            line for line in out.splitlines() if line.startswith("Organized into ")
        )
        pending = next(
            line for line in out.splitlines() if line.startswith("Pending tasks: ")
        )
        return (
            int(dispatch.split(": ", 1)[1].split(" ", 1)[0]),
            int(pending.split(": ", 1)[1]),
        )

    def test_the_emit_hands_out_nothing_the_policy_refuses(self, repo, capsys):
        """Fix round 14: the native emit refuses T005 before handing it out,
        as the executor refuses it before dispatch — recorded failed, out of
        the batches with T006 after it, and the counts agree."""
        handoff, out = self._emit_text(repo, capsys)
        emitted = {t["id"] for b in handoff["batches"] for t in b["tasks"]}
        assert emitted == {"T001", "T002", "T003", "T004", "T007"}
        assert handoff["refused_tasks"]["T005"].startswith(
            "refused before dispatch: Write policy violation (pre-dispatch): "
            "src/vendor/stamp.js"
        )
        assert handoff["not_run_tasks"]["T006"].startswith(
            "not run — depends on T005, which was refused before dispatch"
        )
        assert _statuses(repo)["T005"] == "failed"
        assert _reason(repo, "T005").startswith("refused before dispatch: ")
        written = json.loads((repo.phase_dir / "write_tasks.json").read_text())
        assert {t["task_id"] for t in written["batches"]} == emitted
        assert self._counts(out) == (5, 5)
        assert handoff["pending_tasks"] == 5
        assert "Refused before dispatch (recorded failed in state.json" in out
        assert "without --mark-completed" in out

    @staticmethod
    def _line(out: str, prefix: str) -> str:
        return next(line for line in out.splitlines() if line.startswith(prefix))

    def test_every_count_of_the_first_emit_agrees(self, repo, capsys):
        """Fix round 16 (live run 6, Claude Code stop 01, both repos): the
        first emit said "Organized into 5 execution batches: 5 task(s)"
        beside "Batches: 3" (the decomposition's five layers, counted before
        T005 and T006 were taken out), and the host passed `--total-batches
        5`. Every count is now read off the batches as handed out, the JSON
        carries the one the host passes, and T007 — in the batches, since
        T002's verdict is not known yet — is named as waiting on it, with
        what happens if T002 reports problems."""
        handoff, out = self._emit_text(repo, capsys)
        # Six layers decomposed; T005 and T006 taken out leave four.
        assert [[t["id"] for t in b["tasks"]] for b in handoff["batches"]] == [
            ["T001"], ["T002", "T003"], ["T004"], ["T007"]]
        assert self._line(out, "Organized into ") == (
            "Organized into 4 execution batches: 5 task(s) to dispatch "
            "(T007 waits on a check's verdict)"
        )
        assert self._line(out, "Batches: ") == "Batches: 4"
        assert self._line(out, "Pending tasks: ") == "Pending tasks: 5"
        assert handoff["tasks_to_dispatch"] == 5
        assert self._line(out, "Tasks to dispatch: ") == (
            "Tasks to dispatch: 5 (pass --total-batches 5 to `util metrics "
            "start` and `util metrics record`; the [ETA] line counts tasks, "
            "not batches, and leaves out a task that a `util batch-done` "
            "record keeps from running)"
        )
        assert handoff["waiting_on_checks"] == {
            "T007": (
                "waits on check T002: dispatch it only after T002 is recorded "
                "with `--check passed`; if T002 reports problems, `util "
                "batch-done` prints [CHECK_BLOCKED] for it and it is not run"
            ),
        }
        assert (
            "Waiting on a check's verdict (in the batches after the check they "
            "depend on; if the check reports problems, do not dispatch them — "
            "they stay pending and the phase closes without --mark-completed):"
        ) in out
        assert "  T007 Edit: write the release note: waits on check T002: " in out

    def test_a_resume_that_hands_the_check_out_again_names_its_waiters(
        self, repo, capsys,
    ):
        """On a resume the batches were already right (run 6, 01b/02); a
        resume that hands T002 out again, on changed files, still names T007
        as waiting on it, and its counts agree."""
        self._emit(repo, capsys)
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                       task_id, "--status", "completed")
        self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                   "--status", "completed", "--check", "findings",
                   "--finding", FINDINGS[0])
        main = repo.root / "src" / "main.py"
        main.write_text(main.read_text() + "retries = 3\n", encoding="utf-8")

        handoff, out = self._emit_text(
            repo, capsys, "--resume", "--human-task", "T005=skip")
        assert [[t["id"] for t in b["tasks"]] for b in handoff["batches"]] == [
            ["T002"], ["T007"]]
        assert self._line(out, "Organized into ") == (
            "Organized into 2 execution batches: 2 task(s) to dispatch "
            "(T007 waits on a check's verdict)"
        )
        assert self._line(out, "Batches: ") == "Batches: 2"
        assert handoff["tasks_to_dispatch"] == handoff["pending_tasks"] == 2
        assert list(handoff["waiting_on_checks"]) == ["T007"]

    def test_the_core_passes_the_emitted_count_and_explains_the_waiters(self):
        from pathlib import Path

        core = (Path(__file__).parent.parent / "instructions" / "implement.md").read_text(
            encoding="utf-8")
        assert "--total-batches {total_tasks}" not in core
        assert core.count("--total-batches {tasks_to_dispatch}") == 2
        assert (
            "take `{tasks_to_dispatch}` from the task JSON's `tasks_to_dispatch` "
            "(step 2's `Tasks to dispatch:` line; never count it yourself"
        ) in core
        assert (
            "A task named in `waiting_on_checks` (step 2's \"Waiting on a check's "
            "verdict\" list) is in `batches`, after the check task it depends on"
        ) in core

    def test_a_resume_with_one_task_left_says_one_batch(self, repo, capsys):
        self._emit(repo, capsys)
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                       task_id, "--status", "completed")
        self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                   "--status", "completed", "--check", "findings",
                   "--finding", FINDINGS[0])
        handoff, out = self._emit_text(
            repo, capsys, "--resume", "--human-task", "T002=done",
            "--human-task", "T005=skip",
        )
        assert self._line(out, "Organized into ") == (
            "Organized into 1 execution batch: 1 task(s) to dispatch")
        assert handoff["tasks_to_dispatch"] == 1
        assert "waiting_on_checks" not in handoff
        assert "Waiting on a check's verdict" not in out

    def test_write_prompt_still_refuses_a_task_the_policy_denies_later(
        self, repo, capsys,
    ):
        """The emit's refusal is the gate; `util write-prompt` keeps its own
        for a policy that changed after the batches were handed out."""
        _policy(repo, ALLOW)
        self._emit(repo, capsys)
        _policy(repo, DENY)
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

    def test_the_accept_answers_leave_only_what_runs_in_the_batches(
        self, repo, capsys,
    ):
        """Fix round 14 (stop fixture, Claude Code accept 02): T005 and T006,
        skipped by `--human-task T005=skip`, sat in `batches` 3 and 4 beside
        `skipped_tasks`, and the emit said "4 task(s) to dispatch" against
        "Pending tasks: 3"."""
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

        handoff, out = self._emit_text(
            repo, capsys, "--resume", "--human-task", "T002=done",
            "--human-task", "T005=skip",
        )
        emitted = {t["id"]: t for b in handoff["batches"] for t in b["tasks"]}
        assert list(emitted) == ["T007"]
        assert emitted["T007"]["status"] == "pending"
        assert set(handoff["skipped_tasks"]) == {"T005", "T006"}
        assert "T002" in handoff["accepted_tasks"]
        assert self._counts(out) == (1, 1)
        assert handoff["pending_tasks"] == 1
        written = json.loads((repo.phase_dir / "write_tasks.json").read_text())
        assert [t["task_id"] for t in written["batches"]] == ["T007"]
        assert self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T007")[0] == 0
        # Fix round 14: the released task is told, as on the executor leg.
        code, prompt, _err = self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T007",
            "--stdout")
        assert code == 0
        assert "## Check findings the user accepted" in prompt
        assert "(`--human-task T002=done`)" in prompt
        assert f'  - "{FINDINGS[0]}"' in prompt
        # T006, skipped with T005, is not handed out, so it has no prompt to
        # bind (a task the check never held is told nothing:
        # test_write_prompt.TestAnAcceptedCheckIsNamedOnBothLegs).
        code, prompt, err = self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T006",
            "--stdout")
        assert code == 3, (prompt, err)
        assert "Check findings the user accepted" not in prompt

    def test_next_command_prints_the_resume_and_one_combined_answers_command(
        self, repo, capsys,
    ):
        """Fix round 14 (stop fixture, Claude Code fix/accept 01): the native
        stop printed no command — the host wrote "re-run with --resume" as
        prose, or two separate commands the user had to merge. The report's
        `util next-command` lookup prints the resume and, beside it, one
        command carrying every answer the stop waits for."""
        self._emit(repo, capsys)
        phase = str(repo.phase_dir)
        for task_id in ("T001", "T003", "T004"):
            self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                       task_id, "--status", "completed")
        self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                   "--status", "completed", "--check", "findings",
                   "--finding", FINDINGS[0])
        state_file = str(get_or_create_state(repo.plan).state_file)
        assert self._util(capsys, "finalize-tracking", "--state-file", state_file)[0] == 0

        code, out, err = self._util(
            capsys, "next-command", "plans/checked.md", "--after", "implement",
        )
        assert code == 0, err
        lines = [line for line in out.splitlines() if line.startswith("[NEXT_COMMAND")]
        assert lines[0].startswith("[NEXT_COMMAND] ")
        assert lines[0].endswith("--implement plans/checked.md --resume")
        assert lines[1].startswith("[NEXT_COMMAND_ANSWERS_WHY] or, in one command: ")
        # Fix round 16 (live run 6, Claude Code 01): the released task is
        # told the findings were accepted, as the tool routes' stop says.
        assert (
            "accept the findings of T002 (then T007 runs, told that you "
            "accepted the findings)" in lines[1]
        )
        assert "leave T005 undone" in lines[1]
        assert lines[2].startswith("[NEXT_COMMAND_ANSWERS] ")
        assert lines[2].endswith(
            "--implement plans/checked.md --resume --human-task T002=done "
            "--human-task T005=skip"
        )
        result = json.loads(out.rsplit("MULTI_LLM_RESULT: ", 1)[1])
        step = result["user_next_step"]
        assert step["answers"]["answers"] == ["T002=done", "T005=skip"]
        assert step["answers"]["command"] == lines[2].split(" ", 1)[1]

    def test_a_resume_that_answered_everything_prints_no_answers_line(
        self, repo, capsys,
    ):
        self._emit(repo, capsys)
        state = get_or_create_state(repo.plan)
        for task_id in ("T001", "T002", "T003", "T004"):
            state.update_task_status(task_id, "completed")
        state.save()
        code, out, _err = self._util(
            capsys, "next-command", "plans/checked.md", "--after", "implement",
        )
        assert code == 0
        assert "--human-task T005=skip" in out  # T005 is still recorded failed
        state = get_or_create_state(repo.plan)
        state.update_task_status("T005", "skipped")
        state.save()
        code, out, _err = self._util(
            capsys, "next-command", "plans/checked.md", "--after", "implement",
        )
        assert "[NEXT_COMMAND_ANSWERS" not in out

    def test_the_host_instructions_say_to_relay_both_commands(self):
        from pathlib import Path

        root = Path(__file__).parent.parent
        skill = (root / "SKILL.md").read_text(encoding="utf-8")
        assert (
            "A native `implement` that stopped part-way exited `0`, so look it "
            "up: it prints the `--resume` command and, when tasks wait for an "
            "answer, `[NEXT_COMMAND_ANSWERS]`"
        ) in skill
        summary = (root / "references" / "summary-format.md").read_text(encoding="utf-8")
        assert "`[NEXT_COMMAND_ANSWERS]` (`user_next_step.answers`" in summary
        core = (root / "instructions" / "implement.md").read_text(encoding="utf-8")
        assert (
            "At a stop (step 10 ran without `--mark-completed`) the next-step "
            "lookup of `references/summary-format.md` still runs"
        ) in core
        assert "never write a resume or `--human-task` command of your own" in core

    def test_the_handoff_carries_the_plan_as_typed_for_finalize_tracking(
        self, repo, capsys,
    ):
        """Fix round 14 (racing run 5, Claude Code 06): the closing
        finalize-tracking call carried the absolute plan_file, which the host
        copied; the handoff now carries `plan_arg`, the plan as typed, and the
        instruction passes that."""
        code = multi_llm.main(
            ["implement", "plans/checked.md", "--host", "claude-code", "--yes"]
        )
        assert code == 0
        out = capsys.readouterr().out
        path = next(
            line.split(": ", 1)[1] for line in out.splitlines()
            if line.startswith("Output file: ")
        )
        handoff = json.loads(open(path, encoding="utf-8").read())
        assert handoff["plan_arg"] == "plans/checked.md"
        assert handoff["plan_file"] == str(repo.plan)
        from pathlib import Path

        core = (Path(__file__).parent.parent / "instructions" / "implement.md").read_text(
            encoding="utf-8"
        )
        assert '--plan-file "$PLAN_ARG"' in core
        assert "task JSON's `plan_arg` (the plan as the user typed it)" in core

    def test_a_failed_task_run_again_replaces_its_failure_line(self, repo, capsys):
        """Coordinator note, fix round 14: the native instructions append
        "(task failed — …)" for a failed task, and `append` used to skip the
        resumed run's real entry as a repeated (id, phase), so every later
        prompt kept saying the task failed. The supersede keeps the newest:
        the next task's prompt reads what the task did when it ran again."""
        self._emit(repo, capsys)
        phase = str(repo.phase_dir)
        out = str(repo.plan.parent / repo.plan.stem)

        def append(task_id, summary):
            return self._util(capsys, "prior-changes", "append", "--output-dir", out,
                              "--id", task_id, "--phase", "implement",
                              "--summary", summary)

        self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T001",
                   "--status", "completed")
        append("T001", "Created src/config.json")
        self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T003",
                   "--status", "failed", "--reason", "node --check failed")
        append("T003", "(task failed — node --check failed)")

        handoff = self._emit(repo, capsys, "--resume")
        assert "T003" in {t["id"] for b in handoff["batches"] for t in b["tasks"]}
        self._util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T003",
                   "--status", "completed")
        code, _out, err = append("T003", "Created src/app.js with greet(name)")
        assert code == 0 and "Replaced entry T003/implement" in err

        assert self._util(
            capsys, "write-prompt", "--phase-dir", phase, "--task-id", "T004")[0] == 0
        prompt = (repo.phase_dir / "prompts" / "T004.prompt.txt").read_text(
            encoding="utf-8"
        )
        assert "Created src/app.js with greet(name)" in prompt
        assert "task failed" not in prompt
