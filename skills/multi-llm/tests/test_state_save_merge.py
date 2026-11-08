"""`StateManager.save()` merges its own changes into `state.json` under
`metrics.state_file_lock`, so a completion another writer recorded since this
manager loaded the file is kept (docs/development/write-scheduler.md, the
"exactly one writer per run" paragraph).

Every interleaving below is forced with barriers, pipes or a held lock; none
waits on a sleep.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_DIR))

from utils import metrics  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402


@pytest.fixture
def plan(tmp_path):
    """A plan whose `state.json` already exists, as it does mid-workflow."""
    path = tmp_path / "plan.md"
    path.write_text("# Plan\n", encoding="utf-8")
    StateManager(path).save()
    return path


def _on_disk(manager: StateManager) -> dict:
    return json.loads(manager.state_file.read_text(encoding="utf-8"))


def _child(code: str, *args, cwd) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", code, str(SKILL_DIR), *map(str, args)],
        cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )


class TestConcurrentWritersKeepEachOthersRecords:
    def test_two_managers_loaded_together_keep_both_completions(self, plan):
        """The reviewer's repro: two phases of one plan, loaded before either saves."""
        first = StateManager(plan)
        second = StateManager(plan)

        first.mark_phase_completed("review-plan")
        first.save()
        second.mark_phase_completed("review-tasks")
        second.save()

        assert sorted(_on_disk(first)["phases_completed"]) == [
            "review-plan", "review-tasks"]
        # The later saver now holds the earlier one's record too.
        assert second.is_phase_completed("review-plan")

    def test_two_threads_loaded_together_keep_both_completions(self, plan):
        loaded = threading.Barrier(2)
        errors = []

        def run(phase: str) -> None:
            try:
                manager = StateManager(plan)
                loaded.wait(timeout=30)
                manager.mark_phase_completed(phase)
                manager.save()
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(phase,))
                   for phase in ("review-plan", "review-tasks")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not errors
        assert sorted(_on_disk(StateManager(plan))["phases_completed"]) == [
            "review-plan", "review-tasks"]

    def test_another_process_saving_between_load_and_save(self, plan, tmp_path):
        child = _child(
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from utils.state_manager import StateManager; "
            "m = StateManager(sys.argv[2]); print('loaded', flush=True); "
            "sys.stdin.readline(); m.mark_phase_completed('review-tasks'); "
            "m.save(); print('saved', flush=True)",
            plan, cwd=tmp_path,
        )
        try:
            assert child.stdout.readline().strip() == "loaded"
            here = StateManager(plan)
            child.stdin.write("go\n")
            child.stdin.flush()
            assert child.stdout.readline().strip() == "saved"
        finally:
            _out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err

        here.mark_phase_completed("review-plan")
        here.save()

        assert sorted(_on_disk(here)["phases_completed"]) == [
            "review-plan", "review-tasks"]

    def test_a_locked_raw_write_between_load_and_save_survives(self, plan):
        """`util batch-done` records a task while an orchestrator's manager is loaded."""
        manager = StateManager(plan)
        with metrics.state_file_lock(manager.state_file):
            document = metrics._read_state(manager.state_file)
            document.setdefault("task_status", {})["T001"] = "completed"
            metrics._write_state(manager.state_file, document)

        manager.update_task_status("T002", "completed")
        manager.save()

        assert _on_disk(manager)["task_status"] == {
            "T001": "completed", "T002": "completed"}
        assert manager.get_task_status("T001") == "completed"


class TestTheMerge:
    def test_a_value_both_changed_is_this_saves(self, plan):
        first = StateManager(plan)
        second = StateManager(plan)
        first.update_task_status("T001", "failed", reason="boom")
        first.save()
        second.update_task_status("T001", "completed")
        second.save()

        on_disk = _on_disk(second)
        assert on_disk["task_status"]["T001"] == "completed"
        # The reason went with the status that dropped it.
        assert "T001" not in on_disk.get("task_status_reasons", {})

    def test_what_this_manager_deleted_stays_deleted(self, plan):
        manager = StateManager(plan)
        manager.mark_phase_started("implement")
        manager.save()
        other = StateManager(plan)
        other.mark_phase_completed("review-code")
        other.save()

        # Completing implement pops `phases_started` (empty) in memory.
        manager.mark_phase_completed("implement")
        manager.save()

        on_disk = _on_disk(manager)
        assert "phases_started" not in on_disk
        assert sorted(on_disk["phases_completed"]) == ["implement", "review-code"]

    def test_a_dict_one_side_deleted_keeps_what_the_other_added(self, plan):
        manager = StateManager(plan)
        manager.mark_phase_started("implement")
        manager.save()
        other = StateManager(plan)
        other.mark_phase_started("review-code")
        other.save()

        manager.mark_phase_completed("implement")  # drops its `phases_started`
        manager.save()

        assert list(_on_disk(manager)["phases_started"]) == ["review-code"]

    def test_the_plan_changed_flag_set_at_load_is_saved(self, plan):
        plan.write_text("# Plan\n\nEdited by a human.\n", encoding="utf-8")
        manager = StateManager(plan)
        assert manager.has_plan_changed()
        # A raw writer keeps the file's own (unflagged) view meanwhile.
        with metrics.state_file_lock(manager.state_file):
            document = metrics._read_state(manager.state_file)
            document["task_status"] = {"T001": "completed"}
            metrics._write_state(manager.state_file, document)

        manager.save()

        on_disk = _on_disk(manager)
        assert on_disk.get("plan_changed") is True
        assert on_disk["task_status"] == {"T001": "completed"}

    def test_a_reference_held_into_state_stays_the_saved_one(self, plan):
        manager = StateManager(plan)
        record = manager.state.setdefault("commit_range", {"head": "a"})
        manager.save()
        other = StateManager(plan)
        other.state["commit_range"]["base"] = "b"
        other.save()

        manager.save()
        record["guard"] = {"head": "a"}
        manager.save()

        assert _on_disk(manager)["commit_range"] == {
            "head": "a", "base": "b", "guard": {"head": "a"}}

    def test_an_unreadable_file_is_written_whole(self, plan):
        manager = StateManager(plan)
        manager.mark_phase_completed("implement")
        manager.state_file.write_text("{ not json", encoding="utf-8")

        manager.save()

        assert "implement" in _on_disk(manager)["phases_completed"]


class TestTheStateLock:
    def test_a_lock_not_freed_in_time_raises_and_writes_nothing(
        self, plan, tmp_path, monkeypatch,
    ):
        manager = StateManager(plan)
        manager.mark_phase_completed("implement")
        holder = _child(
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from utils import metrics; "
            "lock = metrics.state_file_lock(sys.argv[2]); lock.__enter__(); "
            "d = metrics._read_state(sys.argv[2]); "
            "d.setdefault('task_status', {})['T001'] = 'completed'; "
            "metrics._write_state(metrics.Path(sys.argv[2]), d); "
            "print('held', flush=True); sys.stdin.readline(); "
            "lock.__exit__(None, None, None)",
            manager.state_file, cwd=tmp_path,
        )
        try:
            assert holder.stdout.readline().strip() == "held"
            before = manager.state_file.read_text(encoding="utf-8")
            monkeypatch.setattr(metrics, "STATE_LOCK_TIMEOUT_S", 0.3)

            with pytest.raises(metrics.StateLockTimeout) as raised:
                manager.save()

            assert ".state.lock" in str(raised.value)
            assert "nothing was written" in str(raised.value)
            assert isinstance(raised.value, IOError)  # save()'s documented error
            assert manager.state_file.read_text(encoding="utf-8") == before
        finally:
            holder.stdin.write("release\n")
            holder.stdin.flush()
            _out, err = holder.communicate(timeout=60)
        assert holder.returncode == 0, err

        # The change is still in memory, and the next save carries it beside
        # what the holder wrote.
        manager.save()
        on_disk = _on_disk(manager)
        assert "implement" in on_disk["phases_completed"]
        assert on_disk["task_status"] == {"T001": "completed"}

    def test_save_inside_a_held_state_lock_does_not_wait_on_itself(self, plan):
        """`record_commit_range` and the apply guard save while holding the lock."""
        manager = StateManager(plan)
        manager.mark_phase_completed("implement")
        done = threading.Event()

        def save_under_the_lock() -> None:
            with metrics.state_file_lock(manager.state_file):
                manager.save()
            done.set()

        thread = threading.Thread(target=save_under_the_lock, daemon=True)
        thread.start()
        thread.join(timeout=60)
        assert done.is_set(), "save() deadlocked on a lock its own thread held"
        assert "implement" in _on_disk(manager)["phases_completed"]


class TestMergeUnits:
    """`save()` merges whole units (a task's record, a processed item's
    record, ...), never a field of one writer beside a field of the other;
    the units are listed in its docstring."""

    def test_a_retry_that_keeps_failed_with_a_new_reason_is_one_record(self, plan):
        """Codex review point 1: B's status equalled its baseline, so a
        field-wise merge took A's completion beside B's new failure reason."""
        seed = StateManager(plan)
        seed.update_task_status("T001", "failed", reason="first failure")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.update_task_status("T001", "completed")
        first.save()
        second.update_task_status("T001", "failed", reason="new failure")
        second.save()

        on_disk = _on_disk(second)
        assert on_disk["task_status"]["T001"] == "failed"
        assert on_disk["task_status_reasons"]["T001"] == "new failure"

    def test_the_same_retry_saved_first_loses_to_the_completion(self, plan):
        seed = StateManager(plan)
        seed.update_task_status("T001", "failed", reason="first failure")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        second.update_task_status("T001", "failed", reason="new failure")
        second.save()
        first.update_task_status("T001", "completed")
        first.save()

        on_disk = _on_disk(first)
        assert on_disk["task_status"]["T001"] == "completed"
        assert "T001" not in on_disk.get("task_status_reasons", {})

    def test_a_tasks_findings_go_with_its_record(self, plan):
        from utils.state_manager import set_task_check_findings_in_state

        first = StateManager(plan)
        second = StateManager(plan)
        first.update_task_status("T002", "failed", reason="check failed")
        set_task_check_findings_in_state(first.state, "T002", ["broken link"])
        first.update_task_status("T003", "completed")
        first.save()
        second.update_task_status("T002", "completed")
        second.save()

        on_disk = _on_disk(second)
        assert on_disk["task_status"] == {"T002": "completed", "T003": "completed"}
        assert "task_check_findings" not in on_disk
        assert "T002" not in on_disk.get("task_status_reasons", {})

    def test_a_processed_item_is_one_writers_record(self, plan):
        seed = StateManager(plan)
        seed.mark_item_processed("apply-fixes", "g1", "failed", {"task_id": "B1"})
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.mark_item_processed("apply-fixes", "g1", "applied", {"task_id": "B1"})
        first.save()
        second.state["processed_apply-fixes"]["g1"]["details"]["leg"] = "native"
        second.save()

        record = _on_disk(second)["processed_apply-fixes"]["g1"]
        assert record["status"] == "failed"
        assert record["details"] == {"task_id": "B1", "leg": "native"}

    def test_a_cleared_item_map_leaves_no_fragment(self, plan):
        """Codex review point 2: B's `details.leg` survived A's clear as
        `{"details": {"leg": "native"}}`, which `processed_record_counts`
        reads as done."""
        from utils.state_manager import processed_record_counts

        seed = StateManager(plan)
        seed.mark_item_processed("apply-fixes", "g1", "failed", {"task_id": "B1"})
        seed.save()
        clearing = StateManager(plan)
        other = StateManager(plan)

        other.state["processed_apply-fixes"]["g1"]["details"]["leg"] = "native"
        other.save()
        clearing.clear_processed_items("apply-fixes")
        clearing.save()

        on_disk = _on_disk(clearing)
        assert "processed_apply-fixes" not in on_disk
        assert not processed_record_counts(
            (on_disk.get("processed_apply-fixes") or {}).get("g1"))

    def test_a_clear_drops_what_another_writer_added_meanwhile(self, plan):
        clearing = StateManager(plan)
        other = StateManager(plan)
        other.mark_item_processed("apply-fixes", "g2", "applied", {"task_id": "B2"})
        other.mark_item_processed("apply-suggestions", "s1", "applied")
        other.save()

        clearing.clear_processed_items("apply-fixes")
        clearing.mark_item_processed("apply-fixes", "g3", "applied")
        clearing.save()

        on_disk = _on_disk(clearing)
        assert list(on_disk["processed_apply-fixes"]) == ["g3"]
        # Another phase's map is not the reset's.
        assert list(on_disk["processed_apply-suggestions"]) == ["s1"]

    def test_a_writer_loaded_before_a_clear_cannot_bring_it_back(self, plan):
        seed = StateManager(plan)
        seed.mark_item_processed("apply-fixes", "g1", "failed", {"task_id": "B1"})
        seed.record_human_decision("apply-fixes", "g1", "approved")
        seed.save()
        stale = StateManager(plan)
        fresh = StateManager(plan)

        fresh.clear_human_decisions("apply-fixes")
        fresh.clear_processed_items("apply-fixes")
        fresh.save()
        stale.state["processed_apply-fixes"]["g1"]["details"]["leg"] = "native"
        stale.mark_item_processed("apply-fixes", "g9", "applied")
        stale.record_human_decision("apply-fixes", "g1", "skipped")
        stale.mark_phase_completed("review-code")
        stale.save()

        on_disk = _on_disk(stale)
        assert "processed_apply-fixes" not in on_disk
        assert "human_decisions_apply-fixes" not in on_disk
        # The stale writer's other records are kept, and it now shows the reset.
        assert "review-code" in on_disk["phases_completed"]
        assert "processed_apply-fixes" not in stale.state

    def test_clear_key_is_a_reset_too(self, plan):
        seed = StateManager(plan)
        seed.set("apply_fix_batch_checks", {"B1": ["npm test"]})
        seed.save()
        stale = StateManager(plan)
        fresh = StateManager(plan)

        fresh.clear_key("apply_fix_batch_checks")
        fresh.save()
        stale.state["apply_fix_batch_checks"]["B2"] = ["npm test"]
        stale.save()

        assert "apply_fix_batch_checks" not in _on_disk(stale)

    def test_a_writer_loaded_before_a_task_reset_cannot_undo_it(self, plan):
        """A start-over (`--fresh`) or `--task` rerun stamps the task in
        `implement_task_reset_at`; a stale record of it is not kept."""
        from utils.state_manager import TASK_RESET_AT_KEY, mark_tasks_reset_in_state

        seed = StateManager(plan)
        seed.update_task_status("T001", "completed")
        seed.update_task_status("T002", "completed")
        seed.save()
        stale = StateManager(plan)
        fresh = StateManager(plan)

        fresh.update_task_status("T001", "pending")
        mark_tasks_reset_in_state(fresh.state, ["T001"])
        fresh.save()
        stale.update_task_status("T001", "failed", reason="stale run")
        stale.update_task_status("T002", "failed", reason="real failure")
        stale.save()

        on_disk = _on_disk(stale)
        assert on_disk["task_status"] == {"T001": "pending", "T002": "failed"}
        assert on_disk["task_status_reasons"] == {"T002": "real failure"}
        assert "T001" in on_disk[TASK_RESET_AT_KEY]

    def test_appends_to_one_metrics_phase_are_both_kept(self, plan):
        first = StateManager(plan)
        second = StateManager(plan)
        first.record_metric("implement", "Batch 1", duration_ms=1000)
        first.save()
        second.record_metric("implement", "Batch 2", duration_ms=2000)
        second.save()

        labels = [entry["label"] for entry in _on_disk(second)["metrics"]["implement"]]
        assert labels == ["Batch 1", "Batch 2"]

    def test_an_entry_one_writer_rewrote_keeps_the_others_append(self, plan):
        """`record_metric_in_state` only appends or replaces an entry in
        place, so the merge goes entry by entry: neither change is lost."""
        seed = StateManager(plan)
        seed.record_metric("implement", "Batch 1", duration_ms=1000)
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)
        first.record_metric("implement", "Batch 2", duration_ms=2000)
        first.save()
        second.state["metrics"]["implement"][0]["duration_ms"] = 1500
        second.save()

        entries = _on_disk(second)["metrics"]["implement"]
        assert [(e["label"], e["duration_ms"]) for e in entries] == [
            ("Batch 1", 1500), ("Batch 2", 2000)]

    def test_tracked_files_merge_by_path(self, plan):
        seed = StateManager(plan)
        seed.track_file("shared.py", "modified", "T001")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)
        first.track_file("a.py", "created", "T001")
        first.track_file("shared.py", "deleted", "T001")
        first.save()
        second.track_file("b.py", "created", "T002")
        second.track_file("shared.py", "modified", "T002")
        second.save()

        tracked = {e["path"]: (e["action"], e["task_id"])
                   for e in _on_disk(second)["tracked_files"]}
        assert tracked == {"shared.py": ("modified", "T002"),
                           "a.py": ("created", "T001"), "b.py": ("created", "T002")}


class TestTheBaselineSave:
    def test_a_lock_timeout_keeps_the_dirty_tree_baseline(self, plan, monkeypatch):
        """Codex review point 3: one `StateLockTimeout` sent the populated
        census into the census-failure fallback, which saved `[]`."""
        import contextlib

        import implement_orchestrator

        real_lock = metrics.state_file_lock
        calls = []

        @contextlib.contextmanager
        def timing_out_once(path):
            calls.append(path)
            if len(calls) == 1:
                raise metrics.StateLockTimeout("held by another command")
            with real_lock(path):
                yield

        monkeypatch.setattr(metrics, "state_file_lock", timing_out_once)
        monkeypatch.setattr(implement_orchestrator, "get_modified_files",
                            lambda: ["user-edits.py"])
        monkeypatch.setattr(implement_orchestrator, "get_staged_files", lambda: [])
        state = StateManager(plan)

        with pytest.raises(metrics.StateLockTimeout):
            implement_orchestrator.record_pre_existing_changes(state)

        assert state.get("pre_existing_changes") == ["user-edits.py"]
        assert "pre_existing_changes" not in _on_disk(state)
        state.save()
        assert _on_disk(state)["pre_existing_changes"] == ["user-edits.py"]

    def test_a_failed_census_still_records_an_empty_baseline(self, plan, monkeypatch):
        import implement_orchestrator

        def broken():
            raise RuntimeError("git is gone")

        monkeypatch.setattr(implement_orchestrator, "get_modified_files", broken)
        state = StateManager(plan)

        implement_orchestrator.record_pre_existing_changes(state)

        assert _on_disk(state)["pre_existing_changes"] == []


class TestTheLockDeadline:
    def test_a_thread_holding_the_state_lock_is_bounded_too(self, plan, monkeypatch):
        """Codex review point 4: the process-wide thread lock had no
        timeout, so a save waited forever behind another thread."""
        manager = StateManager(plan)
        manager.mark_phase_completed("implement")
        before = manager.state_file.read_text(encoding="utf-8")
        held = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with metrics._STATE_THREAD_LOCK:
                held.set()
                release.wait(timeout=60)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        try:
            assert held.wait(timeout=60)
            monkeypatch.setattr(metrics, "STATE_LOCK_TIMEOUT_S", 0.05)
            with pytest.raises(metrics.StateLockTimeout) as raised:
                manager.save()
            assert "nothing was written" in str(raised.value)
            assert manager.state_file.read_text(encoding="utf-8") == before
        finally:
            release.set()
            holder.join(timeout=60)

        manager.save()
        assert "implement" in _on_disk(manager)["phases_completed"]

    def test_one_deadline_covers_both_locks(self, tmp_path, monkeypatch):
        import types

        clock = [1000.0]
        file_waits = []

        class ThreadLock:
            def acquire(self, timeout=-1):
                clock[0] += 70.0  # the other thread held it this long
                return True

            def release(self):
                pass

        class FileMutex:
            def __init__(self, path):
                self.path = path

            def acquire(self, *, timeout_s, poll_interval_s):
                file_waits.append(timeout_s)
                return timeout_s > 0

            def release(self):
                pass

        monkeypatch.setattr(metrics, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
        monkeypatch.setattr(metrics, "_STATE_THREAD_LOCK", ThreadLock())
        monkeypatch.setattr(metrics.process_control, "FileMutex", FileMutex)
        monkeypatch.setattr(metrics, "STATE_LOCK_TIMEOUT_S", 120.0)
        state_path = tmp_path / "state.json"

        with metrics.state_file_lock(state_path):
            pass
        assert file_waits == [50.0]

        monkeypatch.setattr(metrics, "STATE_LOCK_TIMEOUT_S", 60.0)
        with pytest.raises(metrics.StateLockTimeout):
            with metrics.state_file_lock(state_path):
                pass
        assert file_waits[-1] == 0.0


class TestTheJournal:
    """`save()` knows a mutator's write from the journal, not from a value
    that differs: a write that repeats the recorded value is still a write
    (Codex second review, point 1)."""

    def test_a_retry_repeating_the_same_failure_is_still_a_write(self, plan):
        seed = StateManager(plan)
        seed.update_task_status("T001", "failed", reason="provider unavailable")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.update_task_status("T001", "completed")
        first.save()
        second.update_task_status("T001", "failed", reason="provider unavailable")
        second.save()

        on_disk = _on_disk(second)
        assert on_disk["task_status"]["T001"] == "failed"
        assert on_disk["task_status_reasons"]["T001"] == "provider unavailable"
        assert second.get_task_status("T001") == "failed"

    def test_setting_a_whole_value_to_what_it_was_is_still_a_write(self, plan):
        seed = StateManager(plan)
        seed.set("pre_existing_changes", ["a.py"])
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        second.set("pre_existing_changes", ["b.py"])
        second.save()
        first.set("pre_existing_changes", ["a.py"])
        first.save()

        assert _on_disk(first)["pre_existing_changes"] == ["a.py"]

    def test_a_repeated_phase_skip_is_still_a_write(self, plan):
        first = StateManager(plan)
        second = StateManager(plan)

        second.mark_phase_completed("apply-suggestions")
        second.save()
        first.mark_phase_skipped("apply-suggestions", "user chose to skip")
        first.save()

        on_disk = _on_disk(first)
        assert "apply-suggestions" not in on_disk["phases_completed"]
        assert on_disk["phases_skipped"]["apply-suggestions"]["reason"] == (
            "user chose to skip")

    def test_an_untouched_record_is_still_the_files(self, plan):
        seed = StateManager(plan)
        seed.update_task_status("T001", "failed", reason="provider unavailable")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.update_task_status("T001", "completed")
        first.save()
        second.update_task_status("T002", "completed")
        second.save()

        assert _on_disk(second)["task_status"] == {
            "T001": "completed", "T002": "completed"}

    def test_a_save_empties_the_journal(self, plan):
        manager = StateManager(plan)
        manager.update_task_status("T001", "failed", reason="boom")
        manager.save()
        other = StateManager(plan)
        other.update_task_status("T001", "completed")
        other.save()

        # T001 was written before the last save, not since: the file's.
        manager.mark_phase_completed("implement")
        manager.save()

        assert _on_disk(manager)["task_status"]["T001"] == "completed"

    def test_reopen_phase_drops_a_completion_recorded_after_the_load(self, plan):
        """The journal op `reopen_phase_for_validation` can use: the phase
        is this manager's even though its view had no completion to drop."""
        reopening = StateManager(plan)
        other = StateManager(plan)
        other.mark_phase_completed("review-plan")
        other.mark_phase_completed("implement")
        other.save()

        assert reopening.reopen_phase("review-plan") is False
        reopening.save()

        completed = _on_disk(reopening)["phases_completed"]
        assert "review-plan" not in completed
        assert "implement" in completed


class TestIndependentRecordsOfOneKey:
    """Codex second review, point 2: `scope_policy` (by review phase),
    `scope_policy_runs` (by mode) and `apply_fix_batch_checks` (by batch)
    have independent writers."""

    def test_the_restated_key_names_match_their_owners(self):
        from utils import known_checks, scope_policy, scope_policy_runs
        from utils import state_manager as sm

        assert sm._APPLY_FIX_BATCH_CHECKS_KEY == known_checks.APPLY_FIX_BATCH_CHECKS_KEY
        assert sm._SCOPE_POLICY_KEY == scope_policy.STATE_KEY
        assert sm._SCOPE_POLICY_RUNS_KEY == scope_policy_runs.RUNS_KEY

    def test_two_review_phases_keep_their_scope_policies(self, plan):
        from utils import scope_policy

        first = StateManager(plan)
        second = StateManager(plan)
        scope_policy.record_review_round(first.state, "code-review", "preserve", "explicit")
        scope_policy.record_review_round(second.state, "review-plan", "expand", "explicit")
        first.save()
        second.save()

        on_disk = _on_disk(second)
        assert scope_policy.round_policies(on_disk, "code-review")["review_policy"] == "preserve"
        assert sorted(on_disk["scope_policy"]) == ["code-review", "review-plan"]

    def test_two_modes_keep_their_scope_policy_runs(self, plan):
        from utils import scope_policy_runs

        first = StateManager(plan)
        second = StateManager(plan)
        scope_policy_runs.record_run_choice(first.state, "implement", "preserve")
        scope_policy_runs.record_run_choice(second.state, "generate-tasks", "expand")
        first.save()
        second.save()

        on_disk = _on_disk(second)
        assert scope_policy_runs.run_record(on_disk, "implement")["policy"] == "preserve"
        assert scope_policy_runs.run_record(on_disk, "generate-tasks")["policy"] == "expand"

    def test_a_scope_record_is_one_writers(self, plan):
        from utils import scope_policy

        seed = StateManager(plan)
        scope_policy.record_review_round(seed.state, "code-review", "expand", "explicit")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)
        scope_policy.record_apply_policy(first.state, "apply-code-fixes", "preserve", "explicit")
        first.save()
        scope_policy.record_review_round(second.state, "code-review", "preserve", "explicit")
        second.save()

        record = _on_disk(second)["scope_policy"]["code-review"]
        assert record["review_policy"] == "preserve"
        assert "apply_policy" not in record  # never a field of each writer

    def test_two_batches_keep_their_check_reports(self, plan):
        from utils import known_checks

        def record(manager, batch, findings):
            manager.update_task_status(batch, "completed")
            known_checks.set_batch_check_report_in_state(
                manager.state, batch, prior_changes_id=f"pc-{batch}",
                findings=findings, listed=None)

        first = StateManager(plan)
        second = StateManager(plan)
        record(first, "apply-fix-batch-1", ["npm test: 1 failing"])
        record(second, "apply-fix-batch-2", ["ruff: E501"])
        first.save()
        second.save()

        reports = _on_disk(second)[known_checks.APPLY_FIX_BATCH_CHECKS_KEY]
        assert sorted(reports) == ["apply-fix-batch-1", "apply-fix-batch-2"]

    def test_a_batch_check_report_goes_with_its_task_record(self, plan):
        from utils import known_checks

        first = StateManager(plan)
        second = StateManager(plan)
        first.update_task_status("apply-fix-batch-1", "completed")
        known_checks.set_batch_check_report_in_state(
            first.state, "apply-fix-batch-1", prior_changes_id="pc-1",
            findings=["npm test: 1 failing"], listed=None)
        first.save()
        second.update_task_status("apply-fix-batch-1", "failed", reason="timed out")
        second.save()

        on_disk = _on_disk(second)
        assert on_disk["task_status"]["apply-fix-batch-1"] == "failed"
        assert known_checks.APPLY_FIX_BATCH_CHECKS_KEY not in on_disk


class TestAFailedWrite:
    """Codex second review, point 3: the merge used to land in `state`
    before the write, so a retry re-asserted what it had adopted."""

    def _fail_next_write(self, manager, monkeypatch):
        real = manager._write_atomic
        calls = []

        def write(document):
            calls.append(document)
            if len(calls) == 1:
                raise IOError("Failed to save state: disk full")
            return real(document)

        monkeypatch.setattr(manager, "_write_atomic", write)

    def test_a_retry_does_not_reassert_what_the_failed_merge_adopted(
        self, plan, monkeypatch,
    ):
        first = StateManager(plan)
        second = StateManager(plan)
        second.update_task_status("T001", "completed")
        second.save()
        first.mark_phase_completed("review-plan")
        self._fail_next_write(first, monkeypatch)

        with pytest.raises(IOError):
            first.save()
        assert first.get_task_status("T001") is None  # nothing adopted

        second.update_task_status("T001", "failed", reason="provider unavailable")
        second.save()
        first.save()

        on_disk = _on_disk(first)
        assert on_disk["task_status"]["T001"] == "failed"
        assert on_disk["task_status_reasons"]["T001"] == "provider unavailable"
        assert "review-plan" in on_disk["phases_completed"]

    def test_a_failed_write_keeps_the_journal(self, plan, monkeypatch):
        seed = StateManager(plan)
        seed.update_task_status("T001", "failed", reason="provider unavailable")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)
        first.update_task_status("T001", "failed", reason="provider unavailable")
        self._fail_next_write(first, monkeypatch)
        with pytest.raises(IOError):
            first.save()

        second.update_task_status("T001", "completed")
        second.save()
        first.save()

        assert _on_disk(first)["task_status"]["T001"] == "failed"

    def test_a_failed_write_leaves_a_held_reference_alone(self, plan, monkeypatch):
        manager = StateManager(plan)
        record = manager.state.setdefault("commit_range", {"head": "a"})
        manager.save()
        other = StateManager(plan)
        other.state["commit_range"]["base"] = "b"
        other.save()
        self._fail_next_write(manager, monkeypatch)

        with pytest.raises(IOError):
            manager.save()
        assert record == {"head": "a"}
        manager.save()
        assert record == {"head": "a", "base": "b"}


class TestOneMetricEvent:
    """Codex second review, point 4: two writers recording one event
    (phase, run_id, label, batch_index) count it once, by
    `record_metric_in_state`'s own rule."""

    @pytest.fixture
    def run(self, plan):
        seed = StateManager(plan)
        seed.start_metrics_phase("implement", 2)
        seed.save()
        return plan

    @staticmethod
    def _record(manager, **numbers):
        manager.record_metric(
            "implement", "T001: Add the page", task_id="T001", batch_index=1,
            total_batches=2, **numbers)

    def test_the_same_record_twice_is_one_entry(self, run, capsys):
        first = StateManager(run)
        second = StateManager(run)
        self._record(first, duration_ms=1000, token_count=10)
        self._record(second, duration_ms=1000, token_count=10)
        first.save()
        second.save()

        entries = _on_disk(second)["metrics"]["implement"]
        assert len(entries) == 1
        assert "[METRICS_DUPLICATE]" in capsys.readouterr().err

    def test_conflicting_records_keep_the_one_saved_first(self, run, capsys):
        first = StateManager(run)
        second = StateManager(run)
        self._record(first, duration_ms=1000)
        self._record(second, duration_ms=2000)
        first.save()
        second.save()

        entries = _on_disk(second)["metrics"]["implement"]
        assert [entry["duration_ms"] for entry in entries] == [1000]
        assert "[METRICS_CONFLICT]" in capsys.readouterr().err

    def test_a_measured_record_replaces_an_empty_one(self, run):
        first = StateManager(run)
        second = StateManager(run)
        self._record(first)
        self._record(second, duration_ms=2000, tool_uses=3)
        first.save()
        second.save()

        entries = _on_disk(second)["metrics"]["implement"]
        assert len(entries) == 1
        assert entries[0]["duration_ms"] == 2000

    def test_two_batches_of_one_run_are_two_entries(self, run):
        first = StateManager(run)
        second = StateManager(run)
        self._record(first, duration_ms=1000)
        second.record_metric(
            "implement", "T002: Style it", task_id="T002", batch_index=2,
            total_batches=2, duration_ms=1500)
        first.save()
        second.save()

        entries = _on_disk(second)["metrics"]["implement"]
        assert [entry["task_id"] for entry in entries] == ["T001", "T002"]


class TestTheDocumentReset:
    """Codex second review, point 5: `reset()` is an authoritative reset
    of the whole document."""

    def test_a_record_made_before_the_reset_does_not_survive_it(self, plan):
        resetting = StateManager(plan)
        other = StateManager(plan)
        other.mark_phase_completed("implement")
        other.update_task_status("T001", "completed")
        other.save()

        resetting.reset()
        resetting.save()

        on_disk = _on_disk(resetting)
        assert on_disk["phases_completed"] == {}
        assert on_disk["task_status"] == {}
        assert list(on_disk["state_resets"]) == ["*"]

    def test_a_writer_loaded_before_the_reset_cannot_bring_it_back(self, plan):
        seed = StateManager(plan)
        seed.mark_phase_completed("review-plan")
        seed.update_task_status("T001", "completed")
        seed.save()
        stale = StateManager(plan)
        resetting = StateManager(plan)

        resetting.reset()
        resetting.save()
        stale.update_task_status("T001", "failed", reason="old run")
        stale.mark_phase_completed("implement")
        stale.save()

        on_disk = _on_disk(stale)
        assert on_disk["phases_completed"] == {}
        assert on_disk["task_status"] == {}
        assert stale.state["phases_completed"] == {}

    def test_a_writer_loaded_after_the_reset_merges_as_usual(self, plan):
        resetting = StateManager(plan)
        resetting.reset()
        resetting.save()
        first = StateManager(plan)
        second = StateManager(plan)
        first.mark_phase_completed("review-plan")
        first.save()
        second.mark_phase_completed("implement")
        second.save()

        assert sorted(_on_disk(second)["phases_completed"]) == [
            "implement", "review-plan"]


class TestThePlanChangeRecord:
    """Codex third review, point 1: `plan_hash` and `plan_changed` are one
    unit, and the edit a load detected is journaled with its hash."""

    def _flagged(self, plan):
        plan.write_text("# Plan\n\nFirst human edit.\n", encoding="utf-8")
        StateManager(plan).save()
        assert StateManager(plan).has_plan_changed()

    def test_a_second_edit_keeps_its_flag_when_the_first_is_consumed(self, plan):
        self._flagged(plan)
        consumer = StateManager(plan)
        plan.write_text("# Plan\n\nSecond human edit.\n", encoding="utf-8")
        detector = StateManager(plan)
        assert detector.has_plan_changed()

        consumer.clear_plan_changed_flag()
        consumer.save()
        detector.save()

        on_disk = _on_disk(detector)
        assert on_disk.get("plan_changed") is True
        assert on_disk["plan_hash"] == detector._compute_plan_hash()
        assert StateManager(plan).has_plan_changed()

    def test_a_consumer_saving_last_leaves_the_newer_edit_detectable(self, plan):
        self._flagged(plan)
        consumer = StateManager(plan)
        plan.write_text("# Plan\n\nSecond human edit.\n", encoding="utf-8")
        detector = StateManager(plan)

        detector.save()
        consumer.clear_plan_changed_flag()
        consumer.save()

        # The consumer saw only the first edit: its hash goes with its
        # cleared flag, and the next load finds the second edit again.
        assert "plan_changed" not in _on_disk(consumer)
        assert StateManager(plan).has_plan_changed()

    def test_an_edit_two_loads_found_and_one_consumed_stays_consumed(self, plan):
        plan.write_text("# Plan\n\nA human edit.\n", encoding="utf-8")
        consumer = StateManager(plan)
        detector = StateManager(plan)
        assert consumer.has_plan_changed() and detector.has_plan_changed()

        consumer.clear_plan_changed_flag()
        consumer.save()
        detector.save()

        assert "plan_changed" not in _on_disk(detector)
        assert not StateManager(plan).has_plan_changed()

    def test_an_edit_the_detector_consumed_itself_is_its_own(self, plan):
        self._flagged(plan)
        other = StateManager(plan)
        plan.write_text("# Plan\n\nSecond human edit.\n", encoding="utf-8")
        detector = StateManager(plan)
        other.mark_phase_completed("review-plan")
        other.save()

        detector.clear_plan_changed_flag()
        detector.save()

        on_disk = _on_disk(detector)
        assert "plan_changed" not in on_disk
        assert on_disk["plan_hash"] == detector._compute_plan_hash()
        assert "review-plan" in on_disk["phases_completed"]


class TestEveryWriterFamilyJournals:
    """Codex third review, point 2: a write that repeats the recorded value
    is still a write, for every writer family that relied on the value
    comparison."""

    def test_a_repeated_human_task_answer_survives_its_consumption(self, plan):
        seed = StateManager(plan)
        seed.update_task_status("T001", "failed", reason="check failed")
        seed.record_human_task_answers({"T001": "done"})
        seed.save()
        consumer = StateManager(plan)
        answerer = StateManager(plan)

        # A new failure is an outcome the answer did not produce: consumed.
        consumer.update_task_status("T001", "failed", reason="check failed again")
        consumer.save()
        assert "human_task_decisions" not in _on_disk(consumer)
        answerer.record_human_task_answers({"T001": "done"})
        answerer.save()

        assert _on_disk(answerer)["human_task_decisions"] == {"T001": "done"}
        assert answerer.get("human_task_decisions") == {"T001": "done"}

    def test_an_answer_journals_only_the_tasks_it_answers(self, plan):
        seed = StateManager(plan)
        seed.record_human_task_answers({"T001": "done", "T002": "skip"})
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.update_task_status("T002", "completed")
        first.save()
        second.record_human_task_answers({"T001": "skip"})
        second.save()

        on_disk = _on_disk(second)
        # T002's record is the first writer's: the answer it consumed stays gone.
        assert on_disk["human_task_decisions"] == {"T001": "skip"}
        assert on_disk["task_status"]["T002"] == "completed"

    def test_a_repeated_scope_round_in_the_same_second(self, plan, monkeypatch):
        from utils import scope_policy

        monkeypatch.setattr(scope_policy, "_now", lambda: "2026-11-08T10:00:00")
        seed = StateManager(plan)
        seed.record_scope_round("code-review", "preserve", "explicit")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.record_scope_round("code-review", "expand", "explicit")
        first.save()
        second.record_scope_round("code-review", "preserve", "explicit")
        second.save()

        record = _on_disk(second)["scope_policy"]["code-review"]
        assert record["review_policy"] == "preserve"

    def test_a_repeated_apply_policy_in_the_same_second(self, plan, monkeypatch):
        from utils import scope_policy

        monkeypatch.setattr(scope_policy, "_now", lambda: "2026-11-08T10:00:00")
        seed = StateManager(plan)
        seed.record_scope_round("code-review", "preserve", "explicit")
        seed.record_scope_apply_policy("apply-code-fixes", "preserve", "explicit")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.record_scope_apply_policy("apply-code-fixes", "expand", "explicit")
        first.save()
        second.record_scope_apply_policy("apply-code-fixes", "preserve", "explicit")
        second.save()

        assert scope_policy.round_policies(
            _on_disk(second), "apply-code-fixes")["apply_policy"] == "preserve"

    def test_repeated_deferrals_and_completion_are_still_writes(self, plan, monkeypatch):
        from utils import scope_apply, scope_policy

        monkeypatch.setattr(scope_policy, "_now", lambda: "2026-11-08T10:00:00")
        monkeypatch.setattr(scope_apply, "_now", lambda: "2026-11-08T10:00:00")
        deferred = [{"group_id": "g1", "title": "an extension"}]
        seed = StateManager(plan)
        seed.record_scope_round("review-plan", "preserve", "explicit")
        seed.record_scope_deferrals("apply-suggestions", deferred)
        seed.record_scope_completion("apply-suggestions", None)
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.record_scope_deferrals("apply-suggestions", [])
        first.save()
        second.record_scope_deferrals("apply-suggestions", deferred)
        second.record_scope_completion("apply-suggestions", None)
        second.save()

        record = _on_disk(second)["scope_policy"]["review-plan"]
        assert record["deferrals"] == deferred
        assert record[scope_apply.COMPLETION_KEY]["deferred"] == 1

    def test_a_baseline_the_round_already_has_journals_nothing(self, plan):
        seed = StateManager(plan)
        seed.record_scope_round("review-plan", "preserve", "explicit",
                                {"baseline_id": "b-1"})
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.record_scope_round("review-plan", "expand", "explicit", {"baseline_id": "b-2"})
        first.save()
        second.attach_scope_baseline("review-plan", {"baseline_id": "b-3"})
        second.save()

        record = _on_disk(second)["scope_policy"]["review-plan"]
        assert record["review_policy"] == "expand"
        assert record["baseline"]["baseline_id"] == "b-2"

    def test_a_repeated_run_choice_is_still_a_write(self, plan, monkeypatch):
        from utils import scope_policy_runs

        monkeypatch.setattr(scope_policy_runs, "_now", lambda: "2026-11-08T10:00:00.000001")
        seed = StateManager(plan)
        assert seed.record_scope_run_start("implement", ("preserve", "explicit"))
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.record_scope_run_start("implement", ("expand", "explicit"))
        first.save()
        second.record_scope_run_start("implement", ("preserve", "explicit"))
        second.save()

        assert scope_policy_runs.run_record(_on_disk(second), "implement")["policy"] == (
            "preserve")

    def test_closing_a_closed_run_journals_nothing(self, plan):
        from utils import scope_policy_runs

        seed = StateManager(plan)
        seed.record_scope_run_start("full", ("preserve", "explicit"))
        seed.close_scope_run("full")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.record_scope_run_start("full", ("expand", "explicit"))
        first.save()
        assert second.close_scope_run("full") is False
        second.save()

        record = scope_policy_runs.run_record(_on_disk(second), "full")
        assert record["policy"] == "expand" and not record.get("closed_at")

    def test_clearing_left_out_tasks_this_manager_never_saw(self, plan):
        """The reviewer's repro: the scheduler starts a metrics run and
        clears the phase's left-out tasks."""
        starter = StateManager(plan)
        emitter = StateManager(plan)
        emitter.set_metrics_left_out("implement", {"T001": metrics.LEFT_NOT_RUN})
        emitter.save()

        starter.start_metrics_phase("implement", 2)
        starter.set_metrics_left_out("implement", {})
        starter.set_metrics_set_aside("implement", {})
        starter.save()

        assert metrics.LEFT_OUT_KEY not in _on_disk(starter)

    def test_repeated_set_aside_tasks_are_still_a_write(self, plan):
        aside = {"T009": "manual"}
        seed = StateManager(plan)
        seed.set_metrics_set_aside("implement", aside)
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        first.set_metrics_set_aside("implement", {})
        first.save()
        second.set_metrics_set_aside("implement", aside)
        second.save()

        assert _on_disk(second)[metrics.SET_ASIDE_KEY] == {"implement": aside}

    def test_a_repeated_total_restamp_is_still_a_write(self, plan):
        seed = StateManager(plan)
        seed.start_metrics_phase("implement", 4)
        seed.save()
        restamper = StateManager(plan)
        with metrics.state_file_lock(seed.state_file):
            document = metrics._read_state(seed.state_file)
            document["metrics_progress"]["implement"]["total_batches"] = 3
            metrics._write_state(seed.state_file, document)

        assert restamper.set_metrics_total("implement", 4) is True
        restamper.save()

        assert _on_disk(restamper)["metrics_progress"]["implement"]["total_batches"] == 4

    def test_a_total_with_no_progress_entry_journals_nothing(self, plan):
        restamper = StateManager(plan)
        starter = StateManager(plan)
        starter.start_metrics_phase("implement", 4)
        starter.save()

        assert restamper.set_metrics_total("implement", 2) is False
        restamper.mark_phase_started("implement")
        restamper.save()

        assert _on_disk(restamper)["metrics_progress"]["implement"]["total_batches"] == 4

    def test_a_repeated_judge_verdict_is_still_a_write(self, plan):
        from utils import human_decisions

        def verdict(decision):
            return human_decisions.VerdictIngest(
                verdicts={"g1": {"group_id": "g1", "decision": decision, "reason": "r",
                                 "salvaged_description": None, "dropped": None}},
                unresolved=["g2"])

        def record(manager, decision):
            human_decisions.record_verdicts(
                state=manager, phase="apply-suggestions", ingest=verdict(decision),
                items_by_group={})

        seed = StateManager(plan)
        record(seed, "approve")
        seed.save()
        first = StateManager(plan)
        second = StateManager(plan)

        record(first, "skip")
        first.set("human_decision_unresolved_apply-suggestions", [])
        first.save()
        record(second, "approve")
        second.save()

        on_disk = _on_disk(second)
        assert on_disk["human_decision_verdicts_apply-suggestions"]["g1"]["decision"] == (
            "approve")
        assert on_disk["human_decision_unresolved_apply-suggestions"] == ["g2"]

    def test_two_writers_keep_their_own_judge_verdicts(self, plan):
        first = StateManager(plan)
        second = StateManager(plan)
        first.record_judge_verdicts("apply-suggestions", {"g1": {"decision": "approve"}})
        second.record_judge_verdicts("apply-suggestions", {"g2": {"decision": "skip"}})
        first.save()
        second.save()

        assert sorted(_on_disk(second)["human_decision_verdicts_apply-suggestions"]) == [
            "g1", "g2"]


class TestFinishingMetrics:
    """Codex third review, point 3: `finish_metrics_phase` closes a run that
    dispatched nothing, as `metrics.finish_phase_in_state` does, and
    journals only what it closed."""

    def test_a_run_that_dispatched_nothing_closes_its_lines(self, plan):
        manager = StateManager(plan)
        manager.set_metrics_left_out("implement", {"T001": metrics.LEFT_NOT_RUN})
        manager.save()
        twin = json.loads(json.dumps(manager.state))

        finished = manager.finish_metrics_phase("implement")
        manager.save()

        assert finished == metrics.finish_phase_in_state(twin, "implement")
        assert "nothing dispatched" in finished[1]
        assert metrics.LEFT_OUT_KEY not in _on_disk(manager)

    def test_it_leaves_a_progress_entry_it_never_had(self, plan):
        finisher = StateManager(plan)
        finisher.set_metrics_left_out("implement", {"T001": metrics.LEFT_NOT_RUN})
        finisher.save()
        starter = StateManager(plan)
        starter.start_metrics_phase("implement", 3)
        starter.save()

        assert finisher.finish_metrics_phase("implement") is not None
        finisher.save()

        on_disk = _on_disk(finisher)
        assert on_disk["metrics_progress"]["implement"]["total_batches"] == 3
        assert metrics.LEFT_OUT_KEY not in on_disk

    def test_nothing_to_close_journals_nothing(self, plan):
        finisher = StateManager(plan)
        emitter = StateManager(plan)
        emitter.set_metrics_left_out("implement", {"T001": metrics.LEFT_NOT_RUN})
        emitter.save()

        assert finisher.finish_metrics_phase("implement") is None
        finisher.mark_phase_started("implement")
        finisher.save()

        assert _on_disk(finisher)[metrics.LEFT_OUT_KEY] == {
            "implement": {"T001": metrics.LEFT_NOT_RUN}}
