"""Four-day review F4–F6: selected reruns and checks over colocated source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import implement_orchestrator as implement
from tests.test_implement_stop_account import (
    _dispatched_titles, _implement, make_repo,
)
from utils import provider_registry as registry
from utils.batch_done import record_batch
from utils.state_manager import (
    get_or_create_state, set_task_batch_check_in_state,
    set_task_check_findings_in_state, set_task_check_tree_in_state,
)
from utils.subagent_tasks import implement_output_name
from utils.write_batch_scheduler import WriteBatchScheduler
from utils.write_prompt import (
    WritePromptError, changed_since_bound, check_tree_fingerprint,
    record_check_tree,
)
from utils.write_safety import TaskSnapshot


@pytest.fixture
def repo(tmp_path, monkeypatch):
    yield make_repo(tmp_path, monkeypatch)
    registry._config = None
    registry._config_key = None


def _emit(repo, *flags):
    handoff = repo.tmp_path / "handoff.json"
    code = implement.main([
        "--plan-file", str(repo.plan), "--output", str(handoff), "--yes", *flags,
    ])
    return code, json.loads(handoff.read_text())


def _seed_findings(repo, *, selected_completed=False):
    state = get_or_create_state(repo.plan)
    for task_id in ("T001", "T002", "T003"):
        state.update_task_status(task_id, "completed")
    state.update_task_status("T004", "completed" if selected_completed else "pending")
    set_task_check_findings_in_state(state.state, "T002", ["validator rejects the code"])
    state.save()
    return state


class TestExplicitTaskReruns:
    def test_native_handoff_contains_completed_task_and_retires_old_evidence(self, repo):
        assert _implement(repo, "--task", "T003") == 0
        original = (repo.root / "src/main.py").read_text()
        state = get_or_create_state(repo.plan)
        set_task_batch_check_in_state(state.state, "T003", ["old own-check failure"])
        state.save()

        code, handoff = _emit(repo, "--resume", "--task", "T003")
        assert code == 0
        assert [(t["id"], t["status"]) for b in handoff["batches"] for t in b["tasks"]] == [
            ("T003", "pending")
        ]
        materialized = json.loads((repo.phase_dir / "write_tasks.json").read_text())
        assert [t["task_id"] for t in materialized["batches"]] == ["T003"]
        state = get_or_create_state(repo.plan)
        assert state.get_task_status("T003") == "pending"
        assert state.is_phase_started("implement")
        assert not state.get("task_batch_check")
        result = repo.phase_dir / implement_output_name("T003")
        assert not result.exists()
        assert list(repo.phase_dir.glob(result.name + ".rerun-*"))
        assert TaskSnapshot.load(str(repo.phase_dir), "T003") is None

        # If the native host stops before dispatch, the executor cannot adopt
        # the old success or roll the task back to the previous run's baseline.
        repo.log.unlink()
        assert _implement(repo, "--resume", "--task", "T003") == 0
        assert _dispatched_titles(repo) == ["Edit: add tests"]
        snapshot = TaskSnapshot.load(str(repo.phase_dir), "T003")
        assert snapshot is not None
        snapshot.restore()
        assert (repo.root / "src/main.py").read_text() == original

    def test_completed_executor_task_is_redispatched_and_new_failure_stands(self, repo):
        assert _implement(repo, "--task", "T003") == 0
        before = (repo.root / "src/main.py").read_text()
        repo.log.unlink()
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", "[]")
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Edit:")
        assert _implement(repo, "--resume", "--task", "T003") == 3
        assert _dispatched_titles(repo) == ["Edit: add tests"]
        state = get_or_create_state(repo.plan)
        assert state.get_task_status("T003") == "failed"
        assert not state.is_phase_completed("implement")
        snapshot = TaskSnapshot.load(str(repo.phase_dir), "T003")
        assert snapshot is not None
        snapshot.restore()
        assert (repo.root / "src/main.py").read_text() == before

    def test_retry_of_partial_selected_task_keeps_first_baseline(self, repo):
        before = (repo.root / "src/main.py").read_text()
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Edit:")
        assert _implement(repo, "--task", "T003") == 3
        assert (repo.root / "src/main.py").read_text() != before
        repo.monkeypatch.delenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS")
        assert _implement(repo, "--resume", "--task", "T003") == 0
        snapshot = TaskSnapshot.load(str(repo.phase_dir), "T003")
        assert snapshot is not None
        snapshot.restore()
        assert (repo.root / "src/main.py").read_text() == before


class TestSelectedTasksKeepCheckGates:
    @pytest.mark.parametrize("selected_completed", [False, True])
    @pytest.mark.parametrize("native", [False, True])
    def test_omitted_upstream_check_still_holds_selected_task(
        self, repo, selected_completed, native,
    ):
        _seed_findings(repo, selected_completed=selected_completed)
        if native:
            code, handoff = _emit(repo, "--resume", "--task", "T004")
            assert code == 0
            assert handoff["batches"] == []
            assert "T004" in handoff["held_tasks"]
        else:
            assert _implement(repo, "--resume", "--task", "T004") == 3
            assert _dispatched_titles(repo) == []
            report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
            assert report["blocked_by_check"] == {"T004": "T002"}
            assert report["stop"]["reported"] == "validator rejects the code"
        state = get_or_create_state(repo.plan)
        assert state.get_task_status("T004") == "pending"
        assert not state.is_phase_completed("implement")

    @pytest.mark.parametrize("accepted", [False, True])
    def test_recorded_pass_or_acceptance_releases_selected_task(self, repo, accepted):
        state = _seed_findings(repo)
        if accepted:
            state.update_task_status("T002", "completed", reason="accepted — findings")
        else:
            set_task_check_findings_in_state(state.state, "T002", None)
        state.save()
        assert _implement(repo, "--resume", "--task", "T004") == 0
        assert _dispatched_titles(repo) == ["Edit: wire the fallback"]

    @pytest.mark.parametrize("answer", ["done", "skip"])
    @pytest.mark.parametrize("native", [False, True])
    def test_selected_task_can_answer_its_omitted_check(self, repo, answer, native):
        _seed_findings(repo)
        flags = ("--resume", "--task", "T004", "--human-task", f"T002={answer}")
        if native:
            code, handoff = _emit(repo, *flags)
            assert code == 0
            handed = [t["id"] for b in handoff["batches"] for t in b["tasks"]]
            assert handed == (["T004"] if answer == "done" else [])
        else:
            assert _implement(repo, *flags) == 0
            assert _dispatched_titles(repo) == (
                ["Edit: wire the fallback"] if answer == "done" else []
            )
        state = get_or_create_state(repo.plan)
        assert state.get_task_status("T002") == ("completed" if answer == "done" else "skipped")


class TestColocatedCodeIsInTheCheckTree:
    def _code(self, repo):
        path = repo.phase_dir.parent / "src/app.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 'broken'\n")
        return path

    def _executor_tree(self, repo):
        scheduler = WriteBatchScheduler.__new__(WriteBatchScheduler)
        scheduler.repo_root = str(repo.root)
        scheduler.phase_dir = str(repo.phase_dir)
        scheduler.output_dir = str(repo.phase_dir.parent)
        return scheduler._tree_fingerprint()

    def test_both_legs_see_colocated_source_but_ignore_workflow_bookkeeping(self, repo):
        code = self._code(repo)
        _emit(repo)
        before = check_tree_fingerprint(repo.phase_dir)
        assert before == self._executor_tree(repo)
        (repo.phase_dir / "result.json").write_text('{"new":true}')
        (repo.phase_dir.parent / "prior_changes.jsonl").write_text("new entry\n")
        (repo.phase_dir.parent / ".tracking_new.tmp").write_text("temporary\n")
        state = get_or_create_state(repo.plan)
        state.update_task_status("T001", "completed")
        state.save()
        assert check_tree_fingerprint(repo.phase_dir) == before
        assert self._executor_tree(repo) == before
        code.write_text("value = 'fixed with different bytes'\n")
        assert check_tree_fingerprint(repo.phase_dir) != before
        assert self._executor_tree(repo) == check_tree_fingerprint(repo.phase_dir)

    def test_native_check_cannot_record_success_after_editing_colocated_source(self, repo):
        code = self._code(repo)
        _emit(repo)
        record_check_tree(repo.phase_dir, "T002")
        code.write_text("value = 'changed by the check'\n")
        assert changed_since_bound(repo.phase_dir, "T002", []) == [
            str(code.relative_to(repo.root))
        ]
        with pytest.raises(WritePromptError, match="its subagent changed"):
            record_batch(repo.phase_dir, "T002", "completed", check="passed")

    def test_native_resume_reruns_check_after_fixing_colocated_source(self, repo):
        code = self._code(repo)
        _emit(repo)
        state = _seed_findings(repo)
        stamp = check_tree_fingerprint(repo.phase_dir)
        set_task_check_tree_in_state(state.state, "T002", stamp)
        state.save()
        code.write_text("value = 'fixed'\n")
        rerun, kept = implement._native_check_reruns(state, repo.plan, ["T002"], {})
        assert rerun == ["T002"]
        assert kept == {}
