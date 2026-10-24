"""A failed check stops ``implement`` with a full account and a way forward.

Racing re-run 3 (2026-10-05, Codex 06/06b, REPORT issues 1 and 2): T009 — a
verification-only task whose criterion was ``npx html-validate`` exiting 0 —
failed on the generated page, implement stopped needs-human, ``--resume``
re-ran the same read-only check on unchanged code (148 s, same failure), and
T012/T013 — never dispatched — were on no surface: not the summary, not the
counts, not the message. These tests drive the real ``implement`` orchestrator
against the mock runner through the same three-run story and pin:

* the stop names the task, what the check reported, every task not run, and a
  way forward that works (fix then ``--resume``; or ``--human-task <id>=done``);
* ``--resume`` with nothing changed does not dispatch the check again;
* after the code is fixed, ``--resume`` re-runs the check, it passes, and the
  tasks after it run to completion;
* the accept and skip answers, and the guards on who may be answered;
* the result line's ``error`` and the status poll that relays it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import run_status  # noqa: E402
from utils import status_file  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.output_handler import get_phase_dir  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from tests.mock_launchers import install_mock_llm, mock_path  # noqa: E402

#: Five tasks: an edit, a read-only check and an independent edit in one
#: batch, an edit that depends on the check, and a manual task after that —
#: the racing plan's T008 / T009 / T012 / T013 / T010 shape.
PLAN = """# Checked plan

## Overview
A plan with a check in the middle.

## Tasks

### T001: Edit: create the module
Create the module file.
- Depends on: none
- Files to modify: src/main.py

### T002: Check the module
Run the validator on the module. Read-only.
- Depends on: T001
- Files to modify: none

### T003: Edit: add tests
Write unit tests.
- Depends on: T001
- Files to modify: src/main.py

### T004: Edit: wire the fallback
Add the fallback.
- Depends on: T002
- Files to modify: src/main.py

### T005: Play through by hand
**Type**: Manual (user action required)
Open the page and race.
- Depends on: T004
"""

EDIT_GATE = "Title: Edit:"
CHECK_GATE = "Title: Check the module"
CHECK_SUMMARY = "html-validate found 2 errors"
CHECK_NOTES = (
    "Ran `npx --yes html-validate index.html` (exit 1): index.html:1:1 "
    "doctype should be uppercase; index.html:128:30 aria-label is not "
    "recommended on this element."
)


def make_repo(tmp_path, monkeypatch, plan_text: str = PLAN):
    """The fixture's body, with the plan text a parameter: shared with
    ``test_human_task_answers.py`` (fix round 8), whose plans add a code task
    after the manual one or drop the manual one."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "plans").mkdir()
    plan = root / "plans" / "checked.md"
    plan.write_text(plan_text, encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True, env=env)

    bin_dir = tmp_path / "bin"
    install_mock_llm(bin_dir, "codex")
    monkeypatch.setenv("PATH", mock_path(bin_dir, ("codex",)))
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    config = tmp_path / "driver.yaml"
    config.write_text("driver:\n  subagent_runner: codex:gpt-5.5\n", encoding="utf-8")
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
    registry._config = None
    registry._config_key = None
    executor.reset_version_cache()
    monkeypatch.chdir(root)

    state = get_or_create_state(plan)
    state.mark_phase_completed("generate-tasks")
    state.mark_phase_completed("apply-suggestions")
    state.save()

    # Every edit task appends to src/main.py; the check edits nothing.
    monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
    ]))
    monkeypatch.setenv("MOCK_LLM_EDITS_IF_PROMPT_HAS", EDIT_GATE)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        ["T001", "T002", "T003", "T004"]
    ))
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))

    class Repo:
        pass

    obj = Repo()
    obj.root = root
    obj.plan = plan
    obj.tmp_path = tmp_path
    obj.monkeypatch = monkeypatch
    obj.log = log
    obj.phase_dir = Path(get_phase_dir(plan, "implement"))
    return obj


@pytest.fixture
def repo(tmp_path, monkeypatch):
    yield make_repo(tmp_path, monkeypatch)
    registry._config = None
    registry._config_key = None


def _check_fails(repo) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", CHECK_GATE)
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_SUMMARY", CHECK_SUMMARY)
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_NOTES", CHECK_NOTES)


def _check_passes(repo) -> None:
    for name in ("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "MOCK_LLM_WRITE_FAIL_SUMMARY",
                 "MOCK_LLM_WRITE_FAIL_NOTES"):
        repo.monkeypatch.delenv(name, raising=False)


def _implement(repo, *flags: str) -> int:
    return implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--yes", "--exec-subagents", *flags,
    ])


def _dispatched_titles(repo) -> list:
    if not repo.log.exists():
        return []
    titles = []
    for line in repo.log.read_text().splitlines():
        prompt = json.loads(line).get("prompt") or ""
        for row in prompt.splitlines():
            if row.startswith("Title: "):
                titles.append(row[len("Title: "):])
                break
    return titles


def _statuses(repo) -> dict:
    return get_or_create_state(repo.plan).implementation_task_statuses()


def _run_one_that_stops(repo, capsys) -> str:
    """Run 1 of the story: T001 applies, the check fails, the rest never runs."""
    _check_fails(repo)
    assert _implement(repo) == 3
    captured = capsys.readouterr()
    return captured.out + captured.err


class TestTheStopAccountsForEveryTask:

    def test_the_stop_names_the_check_what_it_reported_and_who_did_not_run(
        self, repo, capsys
    ):
        text = _run_one_that_stops(repo, capsys)
        # The failed task, by id and kind, and what the check reported.
        assert "implement stopped at T002 (check: Check the module)" in text
        assert CHECK_SUMMARY in text
        assert "what it reported: " + CHECK_NOTES[:40] in text
        # Every task of the decomposition, by what became of it.
        assert "Done and kept: 1 task(s) (T001)." in text
        assert "Not run (the stop came before they were dispatched): T003, T004." in text
        assert "Deferred manual: T005." in text
        assert "That is every one of the 5 task(s) in this decomposition" in text
        # The way forward: fix then resume, or accept.
        assert "fix what the check reported" in text
        assert "review-code phase and then apply-code-fixes" in text
        assert "--human-task T002=done" in text
        assert "--human-task T002=skip" in text
        assert "T004" in text.split("--human-task T002=skip", 1)[1].split("\n", 1)[0]
        # The commands are the host's own spelling (claude-code here), with
        # the plan as typed — never a dispatcher argv.
        assert f"--implement {repo.plan} --resume" in text
        assert f"--review-code {repo.plan}" in text
        assert f"--apply-code-fixes {repo.plan}" in text

    def test_state_summary_report_and_mirror_carry_the_same_account(
        self, repo, capsys
    ):
        _run_one_that_stops(repo, capsys)
        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "failed", "T003": "pending",
            "T004": "pending", "T005": "skipped",
        }
        for task_id in ("T003", "T004"):
            reason = state.get_task_status_reason(task_id)
            assert reason and reason.startswith("not run — implement stopped at batch"), reason
            assert "T002 failed" in reason

        summary = Path(
            json.loads((repo.phase_dir / "write_batch_report.json").read_text())["summary_file"]
        ).read_text(encoding="utf-8")
        assert (
            "**Tasks in this decomposition**: 5 — 1 done, 1 failed (T002), "
            "2 not run (T003, T004), 1 deferred manual (T005)."
        ) in summary
        assert "| T003 | not run |" in summary
        assert "| T004 | not run |" in summary
        assert "## Cancelled" in summary
        cancelled = summary.split("## Cancelled", 1)[1]
        assert "implement stopped at T002 (check: Check the module)" in cancelled
        assert "- T003\n- T004" in cancelled

        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        assert sorted(report["not_run_tasks"]) == ["T003", "T004"]
        assert report["stop"]["task_id"] == "T002"
        assert report["stop"]["kind"] == "check"
        assert report["stop"]["not_run"] == ["T003", "T004"]
        assert report["stop"]["reported"] == CHECK_NOTES

        mirror = json.loads((repo.phase_dir / ".status.json").read_text())
        write = mirror["progress"]["write_subagents"]
        assert write["not_run_task_ids"] == ["T003", "T004"]
        assert write["failed_task_ids"] == ["T002"]
        # ok + failed + not run == total (the deferred manual task is beside
        # it, as before): 1 + 1 + 2 == 4.
        assert write["total"] == 4
        line = ", ".join(run_status._mirror_text("write_subagents", write))
        assert "failed T002" in line
        assert "not run T003, T004" in line
        assert "deferred manual: T005" in line

        verdict = json.loads(
            (repo.phase_dir / "implement_batch_T002.json").read_text()
        )["verdict"]
        assert verdict["ok"] is False
        assert verdict["verification_only"] is True
        assert verdict["runner_notes"] == CHECK_NOTES
        assert isinstance(verdict.get("tree_fingerprint"), str)

    def test_the_result_line_carries_the_stop_as_its_error(self, repo, capsys):
        _check_fails(repo)
        code = multi_llm.main([
            "implement", str(repo.plan), "--yes", "--exec-subagents",
        ])
        assert code == 3
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        payload = json.loads(line.split(" ", 1)[1])
        assert payload["state"] == "needs-human"
        error = payload["error"]
        assert error.startswith("implement stopped at T002 (check: Check the module): the check failed")
        assert CHECK_SUMMARY in error
        assert "Not run: T003, T004." in error
        assert "--human-task T002=done" in error
        assert "review-code then apply-code-fixes" in error


class TestAResumeThatChangesNothingDoesNotRerunTheCheck:

    def test_an_unchanged_tree_is_refused_with_the_remedy(self, repo, capsys):
        _run_one_that_stops(repo, capsys)
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume") == 3
        text = capsys.readouterr()
        text = text.out + text.err
        assert _dispatched_titles(repo) == [], "the check was dispatched again"
        assert "not run again — nothing in the repository changed since this check failed" in text
        assert CHECK_SUMMARY in text
        assert "--human-task T002=done" in text
        assert "--human-task T002=retry" in text
        assert "Not run (the stop came before they were dispatched): T003, T004." in text
        assert _statuses(repo)["T002"] == "failed"
        reason = get_or_create_state(repo.plan).get_task_status_reason("T002")
        assert reason.startswith("NOT RE-RUN:")
        # No metric for a check that never ran.
        metrics = get_or_create_state(repo.plan).get("metrics") or {}
        labels = [m.get("label", "") for m in metrics.get("implement", [])]
        assert not any("T002" in label and "NOT" in label for label in labels)

    def test_retry_runs_the_check_again_on_the_same_tree(self, repo, capsys):
        _run_one_that_stops(repo, capsys)
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume", "--human-task", "T002=retry") == 3
        assert _dispatched_titles(repo) == ["Check the module"]
        # One-shot: the answer is not carried into a later run.
        assert "T002" not in (get_or_create_state(repo.plan).get("human_task_decisions") or {})

    def test_the_fixed_code_resumes_through_the_check_to_completion(self, repo, capsys):
        """The way forward the stop names, end to end: fix, resume, done."""
        _run_one_that_stops(repo, capsys)
        # The user (or apply-code-fixes) fixes the page: bytes move.
        main = repo.root / "src" / "main.py"
        main.write_text(main.read_text() + "# fixed by hand\n", encoding="utf-8")
        _check_passes(repo)
        repo.log.unlink(missing_ok=True)

        assert _implement(repo, "--resume") == 0
        assert _dispatched_titles(repo) == [
            "Check the module", "Edit: add tests", "Edit: wire the fallback",
        ]
        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "completed", "T003": "completed",
            "T004": "completed", "T005": "skipped",
        }
        assert state.is_phase_completed("implement")
        assert main.read_text().count("# edited\n") == 3  # T001, T003, T004
        verdict = json.loads(
            (repo.phase_dir / "implement_batch_T002.json").read_text()
        )["verdict"]
        assert verdict["ok"] is True and verdict["outcome_label"] == "verified"
        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        assert report["finalized"] is True
        assert report["not_run_tasks"] == {}
        summary = Path(report["summary_file"]).read_text(encoding="utf-8")
        assert "**Tasks in this decomposition**: 5 — 4 done, 1 deferred manual (T005)." in summary


class TestAcceptingTheFailure:

    def test_done_treats_the_check_as_satisfied_and_runs_the_rest(self, repo, capsys):
        _run_one_that_stops(repo, capsys)
        repo.log.unlink(missing_ok=True)
        # The check still fails; the user accepts that.
        assert _implement(repo, "--resume", "--human-task", "T002=done") == 0
        assert _dispatched_titles(repo) == ["Edit: add tests", "Edit: wire the fallback"]
        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "completed", "T003": "completed",
            "T004": "completed", "T005": "skipped",
        }
        assert state.get_task_status_reason("T002").startswith(
            "accepted — you answered --human-task T002=done after it failed"
        )
        assert state.is_phase_completed("implement")
        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        assert report["accepted_tasks"] == {"T002": state.get_task_status_reason("T002")}
        summary = Path(report["summary_file"]).read_text(encoding="utf-8")
        assert "| T002 | accepted after failing |" in summary
        assert "1 accepted after failing (T002)" in summary
        mirror = json.loads((repo.phase_dir / ".status.json").read_text())
        write = mirror["progress"]["write_subagents"]
        assert write["accepted_task_ids"] == ["T002"]
        assert write["total"] == 2 and write["completed"] == 2
        # Fix round 20 (run 8 #6): kept from the earlier run, not dispatched,
        # and named in the `resume: kept` line's words; fix round 22: with
        # T001, which the earlier run finished too.
        assert "(2 kept from an earlier run: T001, T002 (accepted after failing))" in (
            ", ".join(run_status._mirror_text("write_subagents", write))
        )

    def test_skip_leaves_the_check_and_its_dependents_undone(self, repo, capsys):
        _run_one_that_stops(repo, capsys)
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume", "--human-task", "T002=skip") == 0
        assert _dispatched_titles(repo) == ["Edit: add tests"]
        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "skipped", "T003": "completed",
            "T004": "skipped", "T005": "skipped",
        }
        assert "depends on skipped task T002" in state.get_task_status_reason("T004")
        assert state.is_phase_completed("implement")

    def test_only_a_failed_code_task_may_be_answered_and_only_on_resume(
        self, repo, capsys
    ):
        _run_one_that_stops(repo, capsys)
        assert _implement(repo, "--resume", "--human-task", "T003=done") == 3
        err = capsys.readouterr().err
        assert "code tasks that did not fail: T003" in err
        assert _implement(repo, "--human-task", "T002=done") == 3
        err = capsys.readouterr().err
        assert "only with --resume" in err
        assert _implement(repo, "--resume", "--human-task", "T005=retry") == 3
        err = capsys.readouterr().err
        assert "human task is answered done or skip: T005" in err
        assert _statuses(repo)["T002"] == "failed"


class TestTheStatusPollRelaysTheStopLine:

    def test_a_terminal_poll_of_a_failed_run_carries_the_recorded_error(
        self, tmp_path, capsys
    ):
        """The tool routes read a terminal poll's ``payload.error`` as "the
        last line it left"; a detached implement's stop line is recorded as
        ``run.error`` and must ride that poll's result line."""
        plan = tmp_path / "plans" / "p.md"
        plan.parent.mkdir()
        plan.write_text("# p\n", encoding="utf-8")
        phase = plan.parent / "p" / "implement"
        phase.mkdir(parents=True)
        headline = (
            "implement stopped at T009 (check: Check Inline Module Syntax): the "
            "check failed — html-validate found 2 errors. Not run: T012, T013."
        )
        status_file.claim_run(str(phase), {
            "run_id": "run-stop-1", "state": "failed", "exit_code": 3,
            "error": headline, "detached": True, "mode": "implement",
            "replay_flags": ["--host", "codex", "--detach"],
        })
        assert multi_llm.main([
            "status", str(plan), "--view", "run", "--run-id", "run-stop-1",
        ]) == 3
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        payload = json.loads(line.split(" ", 1)[1])
        assert payload["error"] == headline
        assert f"error:     {headline}" in out

    def test_the_generic_close_out_word_is_not_relayed_as_an_error(
        self, tmp_path, capsys
    ):
        plan = tmp_path / "plans" / "p.md"
        plan.parent.mkdir()
        plan.write_text("# p\n", encoding="utf-8")
        phase = plan.parent / "p" / "implement"
        phase.mkdir(parents=True)
        status_file.claim_run(str(phase), {
            "run_id": "run-stop-2", "state": "failed", "exit_code": 3,
            "error": "needs_human (exit 3)", "detached": True, "mode": "implement",
        })
        assert multi_llm.main([
            "status", str(plan), "--view", "run", "--run-id", "run-stop-2",
        ]) == 3
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        assert "error" not in json.loads(line.split(" ", 1)[1])


class TestTheReviewRecordReportsNotRunTasksAsGaps:

    def test_a_not_run_task_is_a_gap_not_a_decision(self, repo, capsys):
        """code-review's implementation record lists a `pending` task carrying
        a reason among the tasks "the implement phase was told not to do";
        a task the stop left undispatched carries one now and must stay on
        the gaps line, or its missing work is never reported."""
        import code_review_orchestrator

        _run_one_that_stops(repo, capsys)
        record = code_review_orchestrator.build_implementation_record(repo.plan)
        gaps = [l for l in record.splitlines() if "did not finish" in l]
        decided = [l for l in record.splitlines() if "told not to do" in l]
        assert gaps and "T003 (pending: not run" in gaps[0] and "T004" in gaps[0]
        assert "T002 (failed" in gaps[0]
        assert decided and "T005 (skipped" in decided[0]
        assert "T003" not in decided[0] and "T004" not in decided[0]
