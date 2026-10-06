"""Rollback evidence remains trusted across runner and phase-path changes.

All dispatches use the existing offline scheduler harness; no provider CLI runs.
"""

import shutil
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_plugin_landlock_write import Repo, FakeExecutor, RUNG, claim, edit_file, stream
from utils import restore_batch, subagent_executor as executor
from utils import write_batch_scheduler as wbs, write_safety as safety
from utils.providers.base import PROFILE_AGENT_WRITE


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return Repo(tmp_path, monkeypatch)


def tasks_for(repo, *, parallel=False):
    tasks = []
    for index, runner in enumerate(("claude-code:sonnet", "codex:gpt-5.5")):
        target = repo.root / f"source{index}.txt"
        target.write_text(f"original-{index}\n")
        tid = f"T00{index + 1}"
        tasks.append(repo.task(
            0 if parallel else index, kind="implement", target=target,
            task_id=tid, expected_ids=[tid], model_hint=runner,
            can_parallelize=parallel,
        ))
    return tasks


@pytest.mark.parametrize("parallel", [False, True])
def test_mixed_runners_protect_evidence_before_jailed_dispatch(repo, monkeypatch, parallel):
    tasks = tasks_for(repo, parallel=parallel)
    fake = FakeExecutor(repo, monkeypatch, rung=None)
    monkeypatch.setattr(wbs, "resolve_task_runner", lambda task, **kw: task["model_hint"])
    monkeypatch.setattr(wbs.WriteBatchScheduler, "_render_argv", lambda *a, **kw: ["mock"])
    monkeypatch.setattr(
        wbs.WriteBatchScheduler, "_recorded_rung",
        lambda self, spec, profile: RUNG if spec.startswith("codex:") else None,
    )
    barrier = threading.Barrier(2) if parallel else None
    seen = []

    def dispatch(task, *, before_attempt, after_result, phase_dir, runner, cwd, **kwargs):
        external = Path(safety.external_safety_dir(phase_dir))
        if parallel or runner.startswith("codex:"):
            assert (external / "pre_batch" / task["task_id"] / "manifest.json").is_file()
        before_attempt({"task_id": task["task_id"], "attempt": 1, "cwd": cwd})
        if barrier:
            barrier.wait(timeout=5)
        target = Path(task["declared_files"][0])
        target.write_text(target.read_text() + "applied\n")
        data = {"status": "applied", "ids": task["expected_ids"],
                "files_touched": [str(target)], "summary": "Applied"}
        outcome = executor.TaskOutcome(task_id=task["task_id"], kind="implement",
                                       runner=runner, status=executor.STATUS_OK,
                                       profile=PROFILE_AGENT_WRITE)
        outcome.output_path = executor.persist_result(task, phase_dir, data)
        outcome.attempts.append(executor.Attempt(index=1, status=executor.STATUS_OK, exit_code=0))
        after_result({"result": data, "outcome": outcome, "attempt": 1,
                      "exit_code": 0, "duration_seconds": 0.1,
                      "sandbox_rung": RUNG if runner.startswith("codex:") else None})
        seen.append(task["task_id"])
        return outcome

    monkeypatch.setattr(wbs, "dispatch_task", dispatch)
    scheduler = repo.scheduler(tasks, kind="implement", write_target=None, finalize=False)
    if parallel:
        monkeypatch.setattr(scheduler, "write_ceiling", lambda: 2)
    result = scheduler.run()
    assert result.exit_code == 0, result.messages
    assert set(seen) == {"T001", "T002"}
    external = Path(safety.external_safety_dir(str(repo.phase_dir)))
    assert (external / safety.AUDIT_FILENAME).is_file()
    assert {call["spec"] for call in fake.preflights} == {t["model_hint"] for t in tasks}


def test_inconclusive_later_runner_preflight_cannot_leave_writable_evidence(repo, monkeypatch):
    tasks = tasks_for(repo)
    FakeExecutor(repo, monkeypatch, rung=None)
    monkeypatch.setattr(wbs, "resolve_task_runner", lambda task, **kw: task["model_hint"])
    scheduler = repo.scheduler(tasks, kind="implement", write_target=None, finalize=False)

    def preflight(spec, profile):
        if spec.startswith("codex:"):
            raise RuntimeError("probe could not conclude")

    monkeypatch.setattr(scheduler, "_preflight", preflight)
    scheduler._decide_rollback_storage(tasks[0], PROFILE_AGENT_WRITE)
    assert Path(safety.external_safety_dir(str(repo.phase_dir))).is_dir()
    assert scheduler._snapshot(tasks[0]).directory.startswith(scheduler.safety_dir)


def existing_evidence(repo):
    target = repo.root / "source.txt"
    target.write_text("original\n")
    snapshot = safety.TaskSnapshot.take(
        phase_dir=str(repo.phase_dir), task_id="T001", files=[str(target)],
        root=str(repo.root), expected_ids=["T001"],
    )
    target.write_text("partial edit\n")
    record = {"event": "completed", "task_id": "T001", "expected_ids": ["T001"],
              "status": "partial_application", "files_touched": ["source.txt"],
              "observed": {"source.txt": {"sha256_after": safety.file_sha256(str(target))}}}
    safety.AuditLog(str(repo.phase_dir))._append(record)
    blob = repo.phase_dir / "rollback_store" / "ab" / "retained"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"retained bytes")
    return target, snapshot


def test_promotion_preserves_snapshot_audit_and_retained_bytes(repo):
    target, snapshot = existing_evidence(repo)
    original_manifest = Path(snapshot.directory, safety.MANIFEST_NAME).read_bytes()
    original_audit = Path(safety.AuditLog(str(repo.phase_dir)).path).read_bytes()
    external = Path(safety.claim_external_safety_dir(str(repo.phase_dir), str(repo.root)))
    assert (external / "pre_batch" / "T001" / safety.MANIFEST_NAME).read_bytes() == original_manifest
    assert (external / safety.AUDIT_FILENAME).read_bytes() == original_audit
    assert (external / "rollback_store" / "ab" / "retained").read_bytes() == b"retained bytes"
    assert safety.TaskSnapshot.load(str(repo.phase_dir), "T001").entries[0].sha256 == snapshot.entries[0].sha256
    assert restore_batch.restore(phase_dir=str(repo.phase_dir), task_ids=["T001"]) == {
        "T001": {"source.txt": "restored"},
    }
    assert target.read_text() == "original\n"


def test_resumed_runner_switch_reuses_pre_first_dispatch_snapshot(repo, monkeypatch):
    target, snapshot = existing_evidence(repo)
    FakeExecutor(repo, monkeypatch, rung=RUNG)
    task = repo.task(0, kind="implement", target=target, task_id="T001", expected_ids=["T001"])
    scheduler = repo.scheduler([task], kind="implement", write_target=None, finalize=False)
    scheduler._decide_rollback_storage(task, PROFILE_AGENT_WRITE)
    resumed = scheduler._snapshot(task)
    assert resumed.entries[0].sha256 == snapshot.entries[0].sha256
    assert "T001" in scheduler._snapshot_reused


@pytest.mark.parametrize("failure", ["payload", "copy", "symlink"])
def test_failed_promotion_keeps_old_evidence_visible(repo, monkeypatch, failure):
    _, snapshot = existing_evidence(repo)
    payload = Path(snapshot.directory, "files", snapshot.entries[0].payload)
    if failure == "payload":
        payload.write_text("damaged\n")
    elif failure == "symlink":
        payload.unlink()
        payload.symlink_to(repo.plan)
    else:
        monkeypatch.setattr(safety.shutil, "copy2", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    external = Path(safety.external_safety_dir(str(repo.phase_dir)))
    with pytest.raises(safety.WriteSafetyError):
        safety.claim_external_safety_dir(str(repo.phase_dir), str(repo.root))
    assert not external.exists()
    assert safety.safety_dir(str(repo.phase_dir)) == str(repo.phase_dir)
    assert Path(snapshot.directory, safety.MANIFEST_NAME).is_file()
    assert not list(external.parent.glob(".promote-*"))


def replace_phase_with_forgery(repo):
    target, _ = existing_evidence(repo)
    external = Path(safety.claim_external_safety_dir(str(repo.phase_dir), str(repo.root)))
    repo.phase_dir.rename(repo.phase_dir.with_name("old-phase"))
    forged = repo.root / "forged-phase"
    forged.mkdir()
    repo.phase_dir.symlink_to(forged, target_is_directory=True)
    target.write_text("planted baseline\n")
    safety.TaskSnapshot.take(phase_dir=str(forged), task_id="T001", files=[str(target)],
                             root=str(repo.root), expected_ids=["T001"])
    target.write_text("partial edit\n")
    safety.AuditLog(str(forged))._append({
        "event": "completed", "task_id": "T001", "expected_ids": ["T001"],
        "files_touched": ["source.txt"],
        "observed": {"source.txt": {"sha256_after": safety.file_sha256(str(target))}},
    })
    return target, external


def test_library_restore_refuses_a_replaced_phase_path(repo):
    target, external = replace_phase_with_forgery(repo)
    result = restore_batch.restore(phase_dir=str(repo.phase_dir), task_ids=["T001"])
    assert result["T001"]["*"].startswith("skipped:")
    assert "trusted phase identity" in result["T001"]["*"]
    assert target.read_text() == "partial edit\n"
    assert (external / "pre_batch" / "T001" / safety.MANIFEST_NAME).is_file()


@pytest.mark.parametrize("selection", [["--task-id", "T001"], ["--all-failed"]])
def test_cli_restore_preserves_identity_before_resolving_path(repo, selection, capsys):
    target, _ = replace_phase_with_forgery(repo)
    assert restore_batch.main(["--phase-dir", str(repo.phase_dir), *selection]) == 3
    assert "trusted phase identity" in capsys.readouterr().err
    assert target.read_text() == "partial edit\n"


def test_initial_alias_and_later_alias_are_bound_without_changing_evidence(repo):
    target, _ = existing_evidence(repo)
    alias = repo.root / "phase-alias"
    alias.symlink_to(repo.phase_dir, target_is_directory=True)
    external = safety.claim_external_safety_dir(str(alias), str(repo.root))
    assert safety.safety_dir(str(repo.phase_dir)) == external
    assert safety.safety_dir(str(alias)) == external
    second_alias = repo.root / "other-alias"
    second_alias.symlink_to(repo.phase_dir, target_is_directory=True)
    assert safety.safety_dir(str(second_alias)) == external
    alias.unlink()
    alias.symlink_to(repo.root, target_is_directory=True)
    with pytest.raises(safety.WriteSafetyError, match="trusted phase identity"):
        safety.safety_dir(str(alias))
    assert restore_batch.main(["--phase-dir", str(second_alias), "--task-id", "T001"]) == 0
    assert target.read_text() == "original\n"


def test_missing_external_evidence_never_falls_back_to_the_old_copy(repo):
    existing_evidence(repo)
    external = safety.claim_external_safety_dir(str(repo.phase_dir), str(repo.root))
    shutil.rmtree(external)
    with pytest.raises(safety.WriteSafetyError, match="evidence is missing"):
        safety.safety_dir(str(repo.phase_dir))
    with pytest.raises(safety.WriteSafetyError, match="evidence is missing"):
        safety.claim_external_safety_dir(str(repo.phase_dir), str(repo.root))


def test_scheduler_stops_cleanly_when_its_batch_replaces_the_phase_path(repo, monkeypatch):
    target = repo.root / "source.txt"
    target.write_text("original\n")
    task = repo.task(0, kind="implement", target=target, task_id="T001", expected_ids=["T001"])
    external = safety.external_safety_dir(str(repo.phase_dir))

    def replace_path(_repo):
        target.write_text("partial edit\n")
        repo.phase_dir.rename(repo.phase_dir.with_name("old-phase"))
        forged = repo.root / "forged"
        forged.mkdir()
        repo.phase_dir.symlink_to(forged, target_is_directory=True)
        return claim("applied"), stream()

    FakeExecutor(repo, monkeypatch, rung=RUNG).script(replace_path)
    result = repo.scheduler([task], kind="implement", write_target=None, finalize=False).run()
    assert result.exit_code == 3
    assert any("trusted phase identity" in line for line in result.messages)
    assert Path(external, "pre_batch", "T001", safety.MANIFEST_NAME).is_file()


def test_unused_human_runner_is_not_preflighted_or_used_to_select_storage(repo, monkeypatch):
    tasks = tasks_for(repo)
    tasks[1]["executor_role"] = "human"
    tasks[1]["is_human"] = True
    fake = FakeExecutor(repo, monkeypatch, rung=None).script(edit_file(Path(tasks[0]["declared_files"][0])))
    monkeypatch.setattr(wbs, "resolve_task_runner", lambda task, **kw: task["model_hint"])
    monkeypatch.setattr(wbs.WriteBatchScheduler, "_render_argv", lambda *a, **kw: ["mock"])
    # Unavailable external storage must not block a run that has no jailed task.
    monkeypatch.setenv("XDG_STATE_HOME", str(repo.root / "state"))
    scheduler = repo.scheduler(tasks, kind="implement", write_target=None, finalize=False,
                               human_task_strategy=wbs.STRATEGY_SKIP_CONTINUE)
    result = scheduler.run()
    assert result.exit_code == 0, result.messages
    assert {call["spec"] for call in fake.preflights} == {"claude-code:sonnet"}
