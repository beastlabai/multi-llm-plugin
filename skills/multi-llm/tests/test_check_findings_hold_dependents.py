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
    _dispatched_titles,
    _implement,
    _statuses,
    make_repo,
)

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
        assert "no task depends on them, so implement went on" in captured.out
        assert get_or_create_state(repo.plan).is_phase_completed("implement")
        assert _report(repo)["blocked_by_check"] == {}
