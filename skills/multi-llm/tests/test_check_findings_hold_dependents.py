"""A check that reported problems holds back the tasks that depend on it.

Fix round 11 (review of round 10, T1/T2): a check task whose check ran and
reported problems was recorded ``check-findings`` and the loop dispatched the
tasks that depend on it — on the code the check had just rejected — and the
check could never run again inside implement: ``--resume`` skipped it as done
and ``--human-task <id>=retry`` answered only a task recorded ``failed``.

Now the findings stay non-gating for the tasks that do not depend on the
check, and the check's transitive dependents are not run: each is recorded
``not run — upstream check <id> reported problems`` on every surface, and
implement ends needing a human, naming the way forward — fix the code then
``--resume`` (the check runs again because the files changed), accept the
findings with ``--human-task <id>=done`` (the dependents run), or ``=skip``.

These tests drive the real ``implement`` orchestrator against the mock runner
on the shape a live fixture exercises: T001 edits, T002 checks it (findings),
T003 is independent of the check, T004 depends on T002, T005 is manual.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_implement_stop_account import (  # noqa: E402
    CHECK_GATE,
    PLAN,
    _dispatched_titles,
    _implement,
    _statuses,
    make_repo,
)

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import run_status  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402

FINDINGS = [
    "npx html-validate index.html: index.html:1 doctype must be lowercase",
    "npx html-validate index.html: index.html:9 aria-label misused",
]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    yield make_repo(tmp_path, monkeypatch)
    registry._config = None
    registry._config_key = None


def _check_finds(repo, findings=FINDINGS) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": list(findings)}
    ))


def _check_passes(repo) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": True, "findings": []}
    ))


def _run_one_with_findings(repo, capsys) -> str:
    _check_finds(repo)
    assert _implement(repo) == 3
    captured = capsys.readouterr()
    return captured.out + captured.err


def _report(repo) -> dict:
    return json.loads((repo.phase_dir / "write_batch_report.json").read_text())


class TestTheDependentsAreNotRun:

    def test_the_independent_task_runs_and_the_dependent_does_not(self, repo, capsys):
        text = _run_one_with_findings(repo, capsys)
        assert _dispatched_titles(repo) == [
            "Edit: create the module", "Check the module", "Edit: add tests",
        ], "T004 depends on the check and must not be dispatched"
        assert _statuses(repo) == {
            "T001": "completed", "T002": "completed", "T003": "completed",
            "T004": "pending", "T005": "skipped",
        }
        state = get_or_create_state(repo.plan)
        reason = state.get_task_status_reason("T004")
        assert reason.startswith("not run — upstream check T002 reported problems"), reason
        assert state.get("task_check_findings") == {"T002": FINDINGS}
        assert not state.is_phase_completed("implement")

        # The run's own output names the check, what it found, who waits for
        # it, the account, and the three ways forward in the host's spelling.
        assert f"[CHECK_FINDINGS] T002 2 problem(s): {FINDINGS[0]}" in text
        assert "the ones that do were not run (T004)" in text
        assert (
            "ERROR: implement did not run 1 task(s) that depend on a check "
            "that reported problems:" in text
        )
        assert "T002 (check: Check the module) reported 2 problem(s)" in text
        assert "      not run: T004" in text
        assert "Not run (they depend on check T002, which reported problems): T004." in text
        assert "That is every one of the 5 task(s) in this decomposition" in text
        assert f"--implement {repo.plan} --resume" in text
        assert f"--review-code {repo.plan}" in text
        assert "--human-task T002=done" in text
        assert "--human-task T002=skip" in text
        assert "--human-task T002=retry" in text

    def test_summary_report_and_mirror_carry_the_same_account(self, repo, capsys):
        _run_one_with_findings(repo, capsys)
        report = _report(repo)
        assert report["finalized"] is False
        assert report["exit"] == 3
        assert report["blocked_by_check"] == {"T004": "T002"}
        assert report["not_run_tasks"]["T004"].startswith(
            "not run — upstream check T002 reported problems"
        )
        stop = report["stop"]
        assert stop["kind"] == "check-findings"
        assert stop["task_id"] == "T002" and stop["check_task_ids"] == ["T002"]
        assert stop["blocked"] == {"T004": "T002"}
        assert stop["not_run"] == ["T004"]
        assert stop["reported"] == "; ".join(FINDINGS)

        summary = Path(report["summary_file"]).read_text()
        assert (
            "**Tasks in this decomposition**: 5 — 3 done (1 with check findings: "
            "T002), 1 not run (T004: upstream check T002 reported problems), "
            "1 deferred manual (T005)."
        ) in summary, summary
        assert "## Check Findings" in summary
        assert "the tasks that do were not run (see below)" in summary
        assert "## Not Run: Upstream Check Reported Problems" in summary
        assert "| T004 | T002 |" in summary
        assert "| T004 | not run |" in summary
        assert "## Cancelled" not in summary

        write = json.loads(
            (repo.phase_dir / ".status.json").read_text()
        )["progress"]["write_subagents"]
        assert write["blocked_by_check"] == {"T004": "T002"}
        assert write["not_run_task_ids"] == ["T004"]
        assert write["check_findings_task_ids"] == ["T002"]
        line = ", ".join(run_status._mirror_text("write_subagents", write))
        assert "check findings T002" in line
        assert "not run (upstream check T002 reported problems) T004" in line
        assert "not run T004" not in line

    def test_the_result_line_carries_the_block_as_its_error(self, repo, capsys):
        _check_finds(repo)
        assert multi_llm.main([
            "implement", str(repo.plan), "--yes", "--exec-subagents",
        ]) == 3
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        payload = json.loads(line.split(" ", 1)[1])
        assert payload["state"] == "needs-human"
        error = payload["error"]
        assert error.startswith(
            "implement did not run T004: they depend on T002 (check: Check the "
            "module), whose check reported problems"
        ), error
        assert FINDINGS[0] in error
        assert "--human-task T002=done" in error
        assert "review-code then apply-code-fixes" in error

    def test_review_code_reads_the_held_task_as_a_gap(self, repo, capsys):
        import code_review_orchestrator

        _run_one_with_findings(repo, capsys)
        record = code_review_orchestrator.build_implementation_record(repo.plan)
        gaps = [l for l in record.splitlines() if "did not finish" in l]
        assert gaps and "T004 (pending: not run — upstream check T002" in gaps[0]
        assert any("Check task T002 ran and reported" in l for l in record.splitlines())

    def test_a_check_no_task_depends_on_still_lets_implement_complete(
        self, repo, capsys
    ):
        """The findings stay non-gating when nothing depends on the check."""
        plan = repo.plan.read_text().replace(
            "- Depends on: T002\n- Files to modify: src/main.py",
            "- Depends on: T001\n- Files to modify: src/main.py",
        )
        repo.plan.write_text(plan, encoding="utf-8")
        _check_finds(repo)
        assert _implement(repo) == 0
        captured = capsys.readouterr()
        assert "no task waits for them, so implement went on" in captured.out
        assert get_or_create_state(repo.plan).is_phase_completed("implement")
        assert _report(repo)["blocked_by_check"] == {}


class TestTheCheckRunsAgain:
    """Review T2: a check with findings was done for good — `--resume`
    skipped it and `--human-task <id>=retry` answered only a failed task."""

    def test_a_resume_that_changes_nothing_keeps_the_findings_and_the_hold(
        self, repo, capsys
    ):
        _run_one_with_findings(repo, capsys)
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume") == 3
        text = capsys.readouterr()
        text = text.out + text.err
        assert _dispatched_titles(repo) == [], "nothing changed: nothing runs"
        assert "not run: T004" in text
        state = get_or_create_state(repo.plan)
        assert state.get("task_check_findings") == {"T002": FINDINGS}
        assert _statuses(repo)["T004"] == "pending"
        assert state.get_task_status_reason("T002") == (
            "not run again: nothing changed since its check reported problems"
        )
        report = _report(repo)
        assert report["blocked_by_check"] == {"T004": "T002"}
        row = [r for r in report["tasks"] if r["task_id"] == "T002"][-1]
        assert row["outcome_label"] == "check-findings"

    def test_retry_runs_the_check_again_on_the_same_tree(self, repo, capsys):
        _run_one_with_findings(repo, capsys)
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume", "--human-task", "T002=retry") == 3
        assert _dispatched_titles(repo) == ["Check the module"]
        assert "T002" not in (
            get_or_create_state(repo.plan).get("human_task_decisions") or {}
        )

    def test_the_fixed_code_resumes_through_the_check_to_completion(self, repo, capsys):
        _run_one_with_findings(repo, capsys)
        main = repo.root / "src" / "main.py"
        main.write_text(main.read_text() + "# fixed by hand\n", encoding="utf-8")
        _check_passes(repo)
        repo.log.unlink(missing_ok=True)

        assert _implement(repo, "--resume") == 0
        assert _dispatched_titles(repo) == ["Check the module", "Edit: wire the fallback"]
        state = get_or_create_state(repo.plan)
        assert state.implementation_task_statuses() == {
            "T001": "completed", "T002": "completed", "T003": "completed",
            "T004": "completed", "T005": "skipped",
        }
        assert state.is_phase_completed("implement")
        assert "task_check_findings" not in state.state
        report = _report(repo)
        assert report["finalized"] is True
        assert report["blocked_by_check"] == {} and report["not_run_tasks"] == {}
        verdict = json.loads(
            (repo.phase_dir / "implement_batch_T002.json").read_text()
        )["verdict"]
        assert verdict["outcome_label"] == "verified"


class TestAnsweringACheckWithFindings:

    def test_done_accepts_the_findings_and_runs_the_rest(self, repo, capsys):
        _run_one_with_findings(repo, capsys)
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume", "--human-task", "T002=done") == 0
        assert _dispatched_titles(repo) == ["Edit: wire the fallback"]
        state = get_or_create_state(repo.plan)
        assert _statuses(repo)["T004"] == "completed"
        reason = state.get_task_status_reason("T002")
        assert reason.startswith(
            "accepted — you answered --human-task T002=done after its check "
            "reported problems"
        ), reason
        assert state.is_phase_completed("implement")
        # The findings stand: review-code is still told about them.
        assert [e["id"] for e in state.implementation_check_findings()] == ["T002"]
        report = _report(repo)
        assert report["accepted_tasks"] == {"T002": reason}
        assert report["blocked_by_check"] == {}
        summary = Path(report["summary_file"]).read_text()
        assert "| T002 | accepted with its check findings |" in summary
        assert "1 accepted with their check findings (T002)" in summary
        assert "accepted after failing" not in summary
        # A later resume keeps the answer: the check is not handed out again.
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume") == 0
        assert _dispatched_titles(repo) == []

    def test_skip_leaves_the_check_and_its_dependents_undone(self, repo, capsys):
        _run_one_with_findings(repo, capsys)
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume", "--human-task", "T002=skip") == 0
        assert _dispatched_titles(repo) == []
        state = get_or_create_state(repo.plan)
        statuses = _statuses(repo)
        assert statuses["T002"] == "skipped" and statuses["T004"] == "skipped"
        assert state.get_task_status_reason("T002").endswith(
            "T002=skip after its check reported problems"
        )
        assert "depends on skipped task T002" in state.get_task_status_reason("T004")
        assert state.is_phase_completed("implement")
        # Skipping the check does not make its problems go away.
        assert [e["id"] for e in state.implementation_check_findings()] == ["T002"]

    def test_only_on_resume_and_only_a_check_whose_findings_hold_tasks(
        self, repo, capsys
    ):
        _run_one_with_findings(repo, capsys)
        assert _implement(repo, "--human-task", "T002=done") == 3
        assert "only with --resume" in capsys.readouterr().err
        assert _implement(repo, "--resume", "--human-task", "T003=retry") == 3
        assert "code tasks that did not fail: T003" in capsys.readouterr().err
        assert _statuses(repo)["T002"] == "completed"


class TestTheNativeLegHoldsTheDependents:
    """On the native leg the host dispatches each task itself, so the plugin
    enforces the hold where the host gets and records each task: `util
    batch-done --check findings` records the dependents held, `util
    write-prompt` refuses them, and `util finalize-tracking
    --mark-completed` refuses while one is left."""

    def _emit(self, repo, capsys, *flags) -> dict:
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", *flags]
        ) == 0
        out = capsys.readouterr().out
        path = next(
            line.split(": ", 1)[1] for line in out.splitlines()
            if line.startswith("Output file: ")
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _util(self, capsys, *argv) -> tuple:
        code = multi_llm.main(["util", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def _done(self, repo, capsys, task_id, *extra):
        return self._util(
            capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
            "--task-id", task_id, "--status", "completed", *extra,
        )

    def _prompt(self, repo, capsys, task_id):
        return self._util(
            capsys, "write-prompt", "--phase-dir", str(repo.phase_dir),
            "--task-id", task_id,
        )

    def _findings(self):
        return [arg for finding in FINDINGS for arg in ("--finding", finding)]

    def test_batch_done_holds_write_prompt_refuses_and_completion_is_refused(
        self, repo, capsys,
    ):
        self._emit(repo, capsys)
        assert self._done(repo, capsys, "T001")[0] == 0
        code, out, err = self._done(
            repo, capsys, "T002", "--check", "findings", *self._findings(),
        )
        assert code == 0, err
        assert "[CHECK_BLOCKED] T004 depends on T002, whose check reported problems" in out
        state = get_or_create_state(repo.plan)
        assert state.get_task_status("T004") == "pending"
        assert state.get_task_status_reason("T004").startswith(
            "not run — upstream check T002 reported problems"
        )
        assert state.get_task_status_reason("T003") is None

        assert self._prompt(repo, capsys, "T003")[0] == 0
        code, _out, err = self._prompt(repo, capsys, "T004")
        assert code == 3
        assert f"[WRITE_PROMPT_BLOCKED] T004 depends on T002, whose check reported problems: {FINDINGS[0]}" in err
        assert "--human-task T002=done" in err
        assert not (repo.phase_dir / "prompts" / "T004.prompt.txt").exists()

        state_file = str(state.state_file)
        code, _out, err = self._util(
            capsys, "finalize-tracking", "--state-file", state_file, "--mark-completed",
        )
        assert code == 3
        assert "T004 (waits for T002) did not run" in err
        assert not get_or_create_state(repo.plan).is_phase_completed("implement")
        assert self._util(capsys, "finalize-tracking", "--state-file", state_file)[0] == 0

    def test_a_passed_recheck_releases_the_dependents(self, repo, capsys):
        self._emit(repo, capsys)
        self._done(repo, capsys, "T001")
        self._done(repo, capsys, "T002", "--check", "findings", *self._findings())
        code, out, _err = self._done(repo, capsys, "T002", "--check", "passed")
        assert code == 0 and "[CHECK_BLOCKED]" not in out
        state = get_or_create_state(repo.plan)
        assert state.get_task_status("T004") == "pending"
        assert state.get_task_status_reason("T004") is None
        assert self._prompt(repo, capsys, "T004")[0] == 0

    def test_a_native_resume_hands_the_check_out_again_or_accepts_it(
        self, repo, capsys,
    ):
        self._emit(repo, capsys)
        self._done(repo, capsys, "T001")
        self._done(repo, capsys, "T002", "--check", "findings", *self._findings())
        self._done(repo, capsys, "T003")

        handoff = self._emit(repo, capsys, "--resume")
        emitted = {t["id"]: t for b in handoff["batches"] for t in b["tasks"]}
        assert emitted["T002"]["status"] == "pending", "the check runs again"
        assert "T004" in emitted and "T001" not in emitted

        handoff = self._emit(repo, capsys, "--resume", "--human-task", "T002=done")
        emitted = {t["id"]: t for b in handoff["batches"] for t in b["tasks"]}
        assert "T002" not in emitted
        assert handoff["accepted_tasks"]["T002"].startswith(
            "accepted — you answered --human-task T002=done after its check "
            "reported problems"
        )
        assert self._prompt(repo, capsys, "T004")[0] == 0

    def test_the_native_instructions_say_what_the_plugin_enforces(self):
        text = (Path(__file__).parent.parent / "instructions" / "implement.md").read_text(
            encoding="utf-8"
        )
        assert "[CHECK_BLOCKED]" in text
        assert "[WRITE_PROMPT_BLOCKED]" in text
        assert "## Not Run: Upstream Check Reported Problems" in text
        assert "refuses it while a held task is left" in text
        assert "a check that could not run" in text


#: Fix round 12 (review of round 11, the leak): the PLAN with T005 (manual)
#: depending on the check and a code task T006 after it — the hold reaches
#: T006 only through T005, which a manual answer of `done` completes, so a
#: `--resume` no longer materializes it.
PLAN_THROUGH_A_MANUAL_TASK = (
    PLAN.replace(
        "Open the page and race.\n- Depends on: T004\n",
        "Open the page and race.\n- Depends on: T002\n",
    )
    + """
### T006: Edit: after the manual step
Needs the manual play-through first.
- Depends on: T005
- Files to modify: src/main.py
"""
)


@pytest.fixture
def chain_repo(tmp_path, monkeypatch):
    yield make_repo(tmp_path, monkeypatch, PLAN_THROUGH_A_MANUAL_TASK)
    registry._config = None
    registry._config_key = None


def _paused(repo, *flags) -> int:
    return implement_orchestrator.main([
        "--plan-file", str(repo.plan), "--no-confirm", "--exec-subagents",
        "--force-strategy", "pause-and-ask", *flags,
    ])


class TestTheHoldReachesThroughAFinishedTask:
    """Fix round 12 (review of round 11, HIGH): the walk for the tasks a
    check holds ran over the tasks a resume materialized, and a resume
    leaves a finished task out — so a plain `--resume` that changed nothing
    kept T004 held and dispatched T006, whose only path to the check ran
    through T005, a manual task answered done."""

    def test_a_plain_resume_keeps_holding_the_task_after_the_manual_one(
        self, chain_repo, capsys,
    ):
        repo = chain_repo
        _check_finds(repo)
        assert _paused(repo, "--human-task", "T005=done") == 3
        capsys.readouterr()
        statuses = _statuses(repo)
        assert statuses["T005"] == "completed"
        assert statuses["T004"] == statuses["T006"] == "pending"
        assert "Edit: after the manual step" not in _dispatched_titles(repo)
        recorded = get_or_create_state(repo.plan).get("implement_task_depends_on")
        assert recorded["T006"] == ["T005"] and recorded["T005"] == ["T002"]

        repo.log.unlink(missing_ok=True)
        assert _paused(repo, "--resume") == 3
        captured = capsys.readouterr()
        assert _dispatched_titles(repo) == [], captured.out + captured.err
        assert _statuses(repo)["T006"] == "pending"
        state = get_or_create_state(repo.plan)
        assert state.get_task_status_reason("T006").startswith(
            "not run — upstream check T002 reported problems"
        )
        assert _report(repo)["blocked_by_check"] == {"T004": "T002", "T006": "T002"}
        assert not state.is_phase_completed("implement")

    def test_skipping_the_check_on_a_resume_skips_the_task_after_the_manual_one(
        self, chain_repo, capsys,
    ):
        repo = chain_repo
        _check_finds(repo)
        assert _paused(repo, "--human-task", "T005=done") == 3
        repo.log.unlink(missing_ok=True)
        assert _paused(repo, "--resume", "--human-task", "T002=skip") == 0
        capsys.readouterr()
        assert _dispatched_titles(repo) == []
        statuses = _statuses(repo)
        assert statuses["T002"] == statuses["T004"] == statuses["T006"] == "skipped"
        assert statuses["T005"] == "completed", "a finished task is never re-marked"
        state = get_or_create_state(repo.plan)
        assert state.get_task_status_reason("T006").startswith(
            "skipped — depends on skipped task T002 (Check the module)"
        )
        # Skipped at the user's answer: not owed as manual work.
        assert state.deferred_implementation_tasks() == []


class TestTheNativeLegHoldsThroughAFinishedTask:
    """The same leak on the native leg: `util write-prompt` and `util
    batch-done` walked `write_tasks.json`, which a `--resume` writes
    without the finished tasks."""

    _util = TestTheNativeLegHoldsTheDependents._util
    _done = TestTheNativeLegHoldsTheDependents._done
    _prompt = TestTheNativeLegHoldsTheDependents._prompt
    _findings = TestTheNativeLegHoldsTheDependents._findings

    def _emit(self, repo, capsys, *flags):
        assert implement_orchestrator.main([
            "--plan-file", str(repo.plan), "--no-confirm",
            "--force-strategy", "pause-and-ask", *flags,
        ]) == 0
        capsys.readouterr()
        return json.loads(
            (repo.phase_dir / "write_tasks.json").read_text(encoding="utf-8")
        )

    def test_write_prompt_refuses_and_batch_done_releases_through_it(
        self, chain_repo, capsys,
    ):
        repo = chain_repo
        self._emit(repo, capsys, "--human-task", "T005=done")
        self._done(repo, capsys, "T001")
        code, out, err = self._done(
            repo, capsys, "T002", "--check", "findings", *self._findings(),
        )
        assert code == 0, err
        assert "T004" in out and "T006" in out
        self._done(repo, capsys, "T003")

        written = self._emit(repo, capsys, "--resume")
        assert "T005" not in {t["task_id"] for t in written["batches"]}
        code, _out, err = self._prompt(repo, capsys, "T006")
        assert code == 3, "T006 depends on T002 through the finished T005"
        assert "[WRITE_PROMPT_BLOCKED] T006 depends on T002" in err

        # The check runs again and passes: T006 is released too.
        code, _out, err = self._done(repo, capsys, "T002", "--check", "passed")
        assert code == 0, err
        state = get_or_create_state(repo.plan)
        assert state.get_task_status_reason("T006") is None
        assert self._prompt(repo, capsys, "T006")[0] == 0

