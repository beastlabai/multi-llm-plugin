"""A ``--human-task`` answer answers one stop, never a later one.

Fix round 15 (review of round 14, stop #1). ``implement`` records every
``--human-task <id>=done|skip`` in ``state.json``'s ``human_task_decisions``,
and round 14 made that record load-bearing: a resume command no longer
replays the answers, because a plain ``--resume`` applies them. Nothing ever
removed one, so an answer the user gave to one stop answered a later,
different stop of the same task in silence:

* T002 (a check) reported findings and was answered ``=done``; later the
  user ran it alone with ``--task T002`` and it reported NEW findings; a
  plain ``--resume`` accepted them with no ``[HUMAN_TASK_ANSWERED]`` line;
* a code task answered ``=done`` after failing that failed again when run
  again was accepted again;
* a task answered ``=skip`` stayed skipped when the user ran it again with
  ``--task``.

An answer is consumed by the first status written for its task that the
answer did not produce (``state_manager.update_task_status_in_state``):
running the task again, on either leg, is a new stop. Until then it stands,
so the answers a resume took keep holding through a later stop of another
task: the next resume needs no replay.
"""

from __future__ import annotations

import json

import pytest

from tests.test_check_gate_fixture import (  # noqa: F401 - the fixture
    ALLOW, FINDINGS, NOTE, STAMP, _check, _policy, repo,
)
from tests.test_implement_stop_account import (
    _dispatched_titles, _implement, _statuses,
)

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils.state_manager import (  # noqa: E402
    get_or_create_state,
    update_task_status_in_state,
)
from utils import write_batch_scheduler as scheduler  # noqa: E402


def _emit(repo, capsys, *flags):
    code = implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes", *flags])
    out = capsys.readouterr().out
    assert code == 0, out
    path = next(line.split(": ", 1)[1] for line in out.splitlines()
                if line.startswith("Output file: "))
    return json.loads(open(path, encoding="utf-8").read()), out


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _done(capsys, repo, task_id, *extra):
    code, out, err = _util(
        capsys, "batch-done", "--phase-dir", str(repo.phase_dir), "--task-id",
        task_id, *extra,
    )
    assert code == 0, (out, err)


def _emitted(handoff):
    return [t["id"] for b in handoff["batches"] for t in b["tasks"]]


def _decisions(repo):
    return dict(get_or_create_state(repo.plan).get("human_task_decisions") or {})


def _native_stop_with_findings(repo, capsys):
    """The first native run: T005 refused at emit, T002 records findings."""
    _emit(repo, capsys)
    for task_id in ("T001", "T003", "T004"):
        _done(capsys, repo, task_id, "--status", "completed")
    _done(capsys, repo, "T002", "--status", "completed", "--check", "findings",
          "--finding", FINDINGS[0])


class TestTheStatusUpdateConsumesTheAnswer:

    def test_a_status_the_answer_did_not_produce_drops_it(self):
        state = {"human_task_decisions": {"T002": "done", "T005": "skip"}}
        # What the answers themselves write keeps them.
        update_task_status_in_state(
            state, "T002", "completed",
            scheduler.ACCEPTED_REASON_PREFIX + "you answered --human-task T002=done",
        )
        update_task_status_in_state(
            state, "T005", "skipped",
            scheduler.ANSWERED_SKIP_PREFIX + "T005=skip after it failed",
        )
        assert state["human_task_decisions"] == {"T002": "done", "T005": "skip"}
        # The task ran again: a new stop.
        update_task_status_in_state(state, "T002", "completed")
        assert state["human_task_decisions"] == {"T005": "skip"}
        # Handed out again (a plan that no longer skips it): the skip is gone.
        update_task_status_in_state(state, "T005", "pending")
        assert "human_task_decisions" not in state

    def test_a_first_pending_status_is_no_outcome(self):
        state = {"human_task_decisions": {"T003": "skip"}}
        update_task_status_in_state(state, "T003", "pending")
        update_task_status_in_state(state, "T003", "pending", "not run — held")
        assert state["human_task_decisions"] == {"T003": "skip"}
        update_task_status_in_state(state, "T003", "failed", "boom")
        assert "human_task_decisions" not in state

    def test_a_manual_task_answered_done_keeps_its_answer(self):
        state = {"human_task_decisions": {"T005": "done"}}
        update_task_status_in_state(state, "T005", "pending")
        update_task_status_in_state(
            state, "T005", "completed", scheduler.COMPLETED_MANUAL_REASON,
        )
        assert state["human_task_decisions"] == {"T005": "done"}

    def test_the_twinned_reasons_agree(self):
        from utils import state_manager

        assert state_manager._ACCEPTED_REASON_PREFIX == scheduler.ACCEPTED_REASON_PREFIX
        assert state_manager._COMPLETED_MANUAL_REASON == scheduler.COMPLETED_MANUAL_REASON


class TestTheNativeLeg:

    def test_an_old_done_does_not_accept_a_checks_new_findings(self, repo, capsys):
        _native_stop_with_findings(repo, capsys)
        handoff, _ = _emit(repo, capsys, "--resume", "--human-task", "T002=done",
                           "--human-task", "T005=skip")
        assert "T002" in handoff["accepted_tasks"]
        _done(capsys, repo, "T007", "--status", "completed")
        state_file = str(get_or_create_state(repo.plan).state_file)
        assert _util(capsys, "finalize-tracking", "--state-file", state_file,
                     "--mark-completed")[0] == 0

        # Later the user runs the check alone, and it reports new problems.
        handoff, _ = _emit(repo, capsys, "--task", "T002")
        assert _emitted(handoff) == ["T002"]
        _done(capsys, repo, "T002", "--status", "completed", "--check", "findings",
              "--finding", "NEW: retries must be 4")
        assert "T002" not in _decisions(repo)

        handoff, out = _emit(repo, capsys, "--resume")
        assert "T002" not in (handoff.get("accepted_tasks") or {}), (
            "an answer to the old findings accepted the new ones")
        assert "[HUMAN_TASK_ANSWERED]" not in out
        state = get_or_create_state(repo.plan)
        assert not str(state.get_task_status_reason("T002") or "").startswith("accepted")
        assert state.get("task_check_findings") == {"T002": ["NEW: retries must be 4"]}

    def test_an_old_done_does_not_accept_a_new_failure(self, repo, capsys):
        _emit(repo, capsys)
        _done(capsys, repo, "T001", "--status", "completed")
        _done(capsys, repo, "T003", "--status", "failed", "--reason", "boom")
        handoff, _ = _emit(repo, capsys, "--resume", "--human-task", "T003=done")
        assert "T003" in handoff["accepted_tasks"]

        # Run again by hand, it fails again.
        _emit(repo, capsys, "--task", "T003")
        _done(capsys, repo, "T003", "--status", "failed", "--reason", "boom again")
        handoff, _ = _emit(repo, capsys, "--resume")
        assert "T003" not in (handoff.get("accepted_tasks") or {}), (
            "an answer to the old failure accepted the new one")
        assert "T003" in _emitted(handoff)
        assert _statuses(repo)["T003"] == "failed"

    def test_task_runs_a_task_answered_skip_and_its_new_failure_stands(
        self, repo, capsys,
    ):
        _native_stop_with_findings(repo, capsys)
        assert _statuses(repo)["T005"] == "failed"
        handoff, _ = _emit(repo, capsys, "--resume", "--human-task", "T005=skip")
        assert set(handoff["skipped_tasks"]) == {"T005", "T006"}

        # The user allows its file and runs it alone: it is handed out.
        _policy(repo, ALLOW)
        handoff, _ = _emit(repo, capsys, "--task", "T005")
        assert _emitted(handoff) == ["T005"], handoff.get("skipped_tasks")
        _done(capsys, repo, "T005", "--status", "failed", "--reason", "stamp broke")
        assert "T005" not in _decisions(repo)

        handoff, _ = _emit(repo, capsys, "--resume")
        assert "T005" not in (handoff.get("skipped_tasks") or {}), (
            "an answer to the old stop skipped the task's new failure")
        assert "T005" in _emitted(handoff)

    def test_the_answers_a_resume_took_hold_through_a_later_stop(self, repo, capsys):
        """The premise round 14 relies on: a resume command carries no
        answer, because the answers already given stand until their task
        runs again."""
        _native_stop_with_findings(repo, capsys)
        _emit(repo, capsys, "--resume", "--human-task", "T002=done",
              "--human-task", "T005=skip")
        _done(capsys, repo, "T007", "--status", "failed", "--reason", "note broke")
        assert _decisions(repo) == {"T002": "done", "T005": "skip"}

        handoff, _ = _emit(repo, capsys, "--resume")
        assert _emitted(handoff) == ["T007"]
        assert "T002" in handoff["accepted_tasks"]
        assert set(handoff["skipped_tasks"]) == {"T005", "T006"}
        assert _decisions(repo) == {"T002": "done", "T005": "skip"}


class TestTheExecutorLeg:

    def test_an_old_done_does_not_accept_a_checks_new_findings(self, repo, capsys):
        _check(repo, passed=False)
        assert _implement(repo) == 3
        assert _implement(repo, "--resume", "--human-task", "T002=done",
                          "--human-task", "T005=skip") == 0
        capsys.readouterr()

        # Run the check alone later: it still reports problems.
        _implement(repo, "--task", "T002")
        capsys.readouterr()
        assert "T002" not in _decisions(repo)
        repo.log.unlink(missing_ok=True)
        _implement(repo, "--resume")
        text = capsys.readouterr()
        state = get_or_create_state(repo.plan)
        assert not str(state.get_task_status_reason("T002") or "").startswith(
            "accepted"), "an answer to the old findings accepted the new ones"
        assert "[HUMAN_TASK_ANSWERED]" not in text.out + text.err
        assert [e["id"] for e in state.implementation_check_findings()] == ["T002"]
        assert "T002" in scheduler.rerunnable_checks(state.state)

    def test_task_runs_a_task_answered_skip(self, repo, capsys):
        _check(repo, passed=False)
        assert _implement(repo) == 3
        assert _implement(repo, "--resume", "--human-task", "T002=done",
                          "--human-task", "T005=skip") == 0
        capsys.readouterr()
        assert _statuses(repo)["T005"] == "skipped"

        _policy(repo, ALLOW)
        repo.log.unlink(missing_ok=True)
        _implement(repo, "--task", "T005")
        capsys.readouterr()
        assert _dispatched_titles(repo) == [STAMP], "--task T005 ran nothing"
        assert _statuses(repo)["T005"] == "completed"
        assert "T005" not in _decisions(repo)

    def test_the_answers_a_resume_took_hold_through_a_later_stop(self, repo, capsys):
        _check(repo, passed=False)
        assert _implement(repo) == 3
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", f"Title: {NOTE}")
        assert _implement(repo, "--resume", "--human-task", "T002=done",
                          "--human-task", "T005=skip") == 3
        capsys.readouterr()
        assert _statuses(repo)["T007"] == "failed"
        assert _decisions(repo) == {"T002": "done", "T005": "skip"}

        repo.monkeypatch.delenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS")
        repo.log.unlink(missing_ok=True)
        assert _implement(repo, "--resume") == 0
        capsys.readouterr()
        assert _dispatched_titles(repo) == [NOTE]
        statuses = _statuses(repo)
        assert statuses["T005"] == "skipped" and statuses["T006"] == "skipped"
        assert get_or_create_state(repo.plan).get_task_status_reason(
            "T002").startswith("accepted")
