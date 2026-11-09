"""The write scheduler's own record writers, through ``utils/atomic_io.py``
(docs/development/runtime.md#atomic-file-writes): the verdict stamp on a
batch's result file (``WriteBatchScheduler._stamp_verdict``), the own-check
re-record on it (``_restamp_own_check``), the re-stamp of a check's tree
(``CheckHolds.restamp_check_trees``) and the inputs binding beside the results
(``ResumeEvidence.record_fingerprint``).

Each once wrote ``<name>.tmp`` opened without ``O_EXCL`` and renamed it in: a
link planted at that name was written through and then renamed into the
record's place, a failed write left the temp behind, and nothing was flushed.
A result file the executor wrote owner-only (``failure_capture``) came back
from its stamp with the umask's mode. These pin that a planted link is left
alone, that a failed write leaves the old record and no temp file, that each
write is flushed, and each writer's mode: a result file stays owner-only, the
inputs binding keeps a new file's mode, as it always had.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_write_executor import env  # noqa: E402,F401
from utils import atomic_io  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.failure_capture import _atomic_write_json  # noqa: E402
from utils.write_check_holds import CheckHolds  # noqa: E402
from utils.write_resume import WRITE_FINGERPRINT_FILENAME, ResumeEvidence  # noqa: E402

POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks and mode bits")

RESULT = {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
          "summary": "Applied"}


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _names(folder: Path):
    return sorted(p.name for p in folder.iterdir())


def _victim(tmp_path: Path) -> Path:
    victim = tmp_path / "victim"
    victim.write_text("victim", encoding="utf-8")
    return victim


def _fsyncs(monkeypatch):
    calls = []
    real_fsync = os.fsync

    def recording(descriptor):
        calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(atomic_io.os, "fsync", recording)
    return calls


def _refuse_replace(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(atomic_io.os, "replace", refuse)


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    yield
    os.umask(previous)


# -- the result file's two writers on the scheduler -------------------------

def _stamped(env, data=None):
    """A scheduler over one apply batch, its result file written as the
    executor writes it (owner-only), and the batch's outcome."""
    task = env.apply_task(0)
    scheduler = env.scheduler([task])
    lines = []
    scheduler.log = lines.append
    path = env.phase_dir / "apply-batch-0.json"
    _atomic_write_json(str(path), dict(data or RESULT))
    outcome = executor.TaskOutcome(
        task_id="apply-batch-0", kind="apply_suggestions", runner="executor",
        status=executor.STATUS_OK, profile=executor.profile_for_kind("apply_suggestions"),
        write_outcome=executor.WRITE_APPLIED,
    )
    outcome.output_path = str(path)
    return scheduler, outcome, path, lines


class TestTheVerdictStamp:
    @POSIX
    def test_the_result_stays_owner_only(self, env, umask_022):
        scheduler, outcome, path, _lines = _stamped(env)
        assert _mode(path) == 0o600
        scheduler._stamp_verdict(outcome)
        assert json.loads(path.read_text(encoding="utf-8"))["verdict"]["ok"] is True
        assert _mode(path) == 0o600

    @POSIX
    def test_a_link_at_the_old_temp_name_is_not_written_through(self, env, tmp_path):
        scheduler, outcome, path, _lines = _stamped(env)
        victim = _victim(tmp_path)
        planted = Path(str(path) + ".tmp")
        planted.symlink_to(victim)
        scheduler._stamp_verdict(outcome)
        assert victim.read_text(encoding="utf-8") == "victim"
        assert not path.is_symlink()
        assert json.loads(path.read_text(encoding="utf-8"))["verdict"]["ok"] is True
        assert planted.is_symlink()  # not this write's file: never removed

    def test_a_failed_write_keeps_the_result_and_leaves_no_temp(self, env, monkeypatch):
        scheduler, outcome, path, lines = _stamped(env)
        before = path.read_bytes()
        _refuse_replace(monkeypatch)
        scheduler._stamp_verdict(outcome)
        monkeypatch.undo()
        assert path.read_bytes() == before
        assert not [n for n in _names(path.parent) if n.endswith(".tmp")]
        assert any("could not record the verdict on apply-batch-0.json" in line
                   for line in lines)

    def test_the_stamp_is_flushed_with_lf_newlines(self, env, monkeypatch):
        scheduler, outcome, path, _lines = _stamped(env)
        calls = _fsyncs(monkeypatch)
        scheduler._stamp_verdict(outcome)
        assert len(calls) == (1 if os.name == "nt" else 2)
        raw = path.read_bytes()
        assert b"\n" in raw and b"\r\n" not in raw


class TestTheOwnCheckReRecord:
    def _verdict_file(self, env):
        data = dict(RESULT, verdict={"ok": True, "batch_check": ["old"]})
        return _stamped(env, data)

    @POSIX
    def test_the_result_stays_owner_only(self, env, umask_022):
        scheduler, _outcome, path, _lines = self._verdict_file(env)
        scheduler._restamp_own_check(str(path), None, {"T001": ["lint: 2 errors"]})
        verdict = json.loads(path.read_text(encoding="utf-8"))["verdict"]
        assert "batch_check" not in verdict
        assert verdict["batch_check_repeats"] == {"T001": ["lint: 2 errors"]}
        assert _mode(path) == 0o600

    @POSIX
    def test_a_link_at_the_old_temp_name_is_not_written_through(self, env, tmp_path):
        scheduler, _outcome, path, _lines = self._verdict_file(env)
        victim = _victim(tmp_path)
        Path(str(path) + ".tmp").symlink_to(victim)
        scheduler._restamp_own_check(str(path), None, {})
        assert victim.read_text(encoding="utf-8") == "victim"
        assert not path.is_symlink()
        assert "batch_check" not in json.loads(path.read_text(encoding="utf-8"))["verdict"]

    def test_a_failed_write_keeps_the_result_and_leaves_no_temp(self, env, monkeypatch):
        scheduler, _outcome, path, lines = self._verdict_file(env)
        before = path.read_bytes()
        _refuse_replace(monkeypatch)
        scheduler._restamp_own_check(str(path), None, {})
        monkeypatch.undo()
        assert path.read_bytes() == before
        assert not [n for n in _names(path.parent) if n.endswith(".tmp")]
        assert any("could not re-record the own check" in line for line in lines)

    def test_the_rewrite_is_flushed(self, env, monkeypatch):
        scheduler, _outcome, path, _lines = self._verdict_file(env)
        calls = _fsyncs(monkeypatch)
        scheduler._restamp_own_check(str(path), None, {})
        assert len(calls) == (1 if os.name == "nt" else 2)


# -- the check tree re-stamp (CheckHolds) ----------------------------------

def _check_holds(tmp_path):
    phase_dir = tmp_path / "implement"
    phase_dir.mkdir()
    path = phase_dir / "T001.json"
    _atomic_write_json(str(path), dict(RESULT, verdict={"ok": True, "tree_fingerprint": "old"}))
    lines = []
    ctx = SimpleNamespace(kind="implement", log=lines.append, repo_root=str(tmp_path))
    holds = CheckHolds(ctx, invoker="executor", task_by_id=lambda task_id: None)
    holds.tree_fingerprint = lambda: "tree-after-the-run"
    outcome = SimpleNamespace(ok=True, check_findings=["lint: 2 errors"],
                              output_path=str(path), task_id="T001")
    return holds, SimpleNamespace(outcomes=[outcome]), path, lines


class TestTheCheckTreeReStamp:
    @POSIX
    def test_the_result_stays_owner_only(self, tmp_path, umask_022):
        holds, result, path, _lines = _check_holds(tmp_path)
        holds.restamp_check_trees(result)
        verdict = json.loads(path.read_text(encoding="utf-8"))["verdict"]
        assert verdict["tree_fingerprint"] == "tree-after-the-run"
        assert _mode(path) == 0o600

    @POSIX
    def test_a_link_at_the_old_temp_name_is_not_written_through(self, tmp_path):
        holds, result, path, _lines = _check_holds(tmp_path)
        victim = _victim(tmp_path)
        Path(str(path) + ".tmp").symlink_to(victim)
        holds.restamp_check_trees(result)
        assert victim.read_text(encoding="utf-8") == "victim"
        assert not path.is_symlink()

    def test_a_failed_write_keeps_the_result_and_leaves_no_temp(self, tmp_path, monkeypatch):
        holds, result, path, lines = _check_holds(tmp_path)
        before = path.read_bytes()
        _refuse_replace(monkeypatch)
        holds.restamp_check_trees(result)
        monkeypatch.undo()
        assert path.read_bytes() == before
        assert _names(path.parent) == ["T001.json"]
        assert any("could not re-stamp the tree on T001.json" in line for line in lines)

    def test_the_re_stamp_is_flushed(self, tmp_path, monkeypatch):
        holds, result, _path, _lines = _check_holds(tmp_path)
        calls = _fsyncs(monkeypatch)
        holds.restamp_check_trees(result)
        assert len(calls) == (1 if os.name == "nt" else 2)


# -- the inputs binding (ResumeEvidence) -----------------------------------

def _evidence(tmp_path):
    phase_dir = tmp_path / "apply-fixes"
    lines = []
    ctx = SimpleNamespace(kind="apply_code_fixes", phase_dir=str(phase_dir), log=lines.append)
    evidence = ResumeEvidence(ctx, resume=True, partial_skips=[], eta=None,
                              commit_guard=lambda: None)
    task = {"task_id": "apply-fix-batch-0", "prompt": "fix it", "expected_ids": ["f0"],
            "declared_files": ["src/a.py"], "batch_index": 0}
    return evidence, task, phase_dir / WRITE_FINGERPRINT_FILENAME, lines


class TestTheInputsBinding:
    @POSIX
    def test_it_keeps_a_new_files_mode(self, tmp_path):
        """A fresh file each time, as it always was: the umask's mode, never
        an earlier binding's."""
        evidence, task, path, _lines = _evidence(tmp_path)
        evidence.record_fingerprint(task)
        os.chmod(path, 0o666)
        previous = os.umask(0o077)
        try:
            evidence.record_fingerprint(task)
        finally:
            os.umask(previous)
        assert _mode(path) == 0o600
        recorded = json.loads(path.read_text(encoding="utf-8"))
        assert recorded["tasks"] == {"apply-fix-batch-0": evidence.task_fingerprint(task)}

    @POSIX
    def test_a_link_at_the_old_temp_name_is_not_written_through(self, tmp_path):
        evidence, task, path, _lines = _evidence(tmp_path)
        path.parent.mkdir(parents=True)
        victim = _victim(tmp_path)
        Path(str(path) + ".tmp").symlink_to(victim)
        evidence.record_fingerprint(task)
        assert victim.read_text(encoding="utf-8") == "victim"
        assert not path.is_symlink()
        assert evidence.load_fingerprints() == {
            "apply-fix-batch-0": evidence.task_fingerprint(task)}

    def test_a_failed_write_keeps_the_binding_and_leaves_no_temp(self, tmp_path, monkeypatch):
        evidence, task, path, lines = _evidence(tmp_path)
        evidence.record_fingerprint(task)
        before = path.read_bytes()
        _refuse_replace(monkeypatch)
        evidence.record_fingerprint(dict(task, prompt="fix it again"))
        monkeypatch.undo()
        assert path.read_bytes() == before
        assert _names(path.parent) == [WRITE_FINGERPRINT_FILENAME]
        assert any("could not record the inputs binding" in line for line in lines)

    def test_the_binding_is_flushed(self, tmp_path, monkeypatch):
        evidence, task, _path, _lines = _evidence(tmp_path)
        calls = _fsyncs(monkeypatch)
        evidence.record_fingerprint(task)
        assert len(calls) == (1 if os.name == "nt" else 2)
