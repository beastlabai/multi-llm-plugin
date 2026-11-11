"""The writers that replace a file whole through ``utils/atomic_io.py``
(docs/development/runtime.md#atomic-file-writes), each with the parameters
it states: the run registry's entries, a detached worker's refusal record,
the checkpoint (``.status.json``), the write-safety records
(``write_safety._atomic_write``) and the Landlock change lists.

Three of them once wrote a temp file of a fixed name opened without
``O_EXCL``: a link planted at that name was written through, and then renamed
into the file's place. These pin that a planted link is left alone, that a
failed write leaves no temp file, and each writer's mode and flush.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import detach, run_registry, status_file, write_safety  # noqa: E402
from utils import landlock_dispatch as ld  # noqa: E402

POSIX_LINKS = pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks and mode bits")


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

    monkeypatch.setattr(os, "fsync", recording)
    return calls


def _refuse_replace(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(os, "replace", refuse)


class TestRunRegistryEntries:
    _ENTRY = {"run_id": "run-1", "pid": 123, "process_start_time": 4.5}

    @POSIX_LINKS
    def test_a_link_at_the_old_temp_name_is_not_written_through(self, tmp_path):
        victim = _victim(tmp_path)
        path = Path(run_registry.entry_path(str(tmp_path), "run-1"))
        path.parent.mkdir(parents=True)
        planted = Path(str(path) + ".tmp")
        planted.symlink_to(victim)
        assert run_registry.write_entry(str(tmp_path), dict(self._ENTRY)) == str(path)
        assert victim.read_text(encoding="utf-8") == "victim"
        assert not path.is_symlink()
        assert json.loads(path.read_text(encoding="utf-8")) == self._ENTRY
        assert planted.is_symlink()  # not this write's file: never removed

    def test_a_failed_write_returns_none_and_leaves_no_temp(self, tmp_path, monkeypatch):
        _refuse_replace(monkeypatch)
        assert run_registry.write_entry(str(tmp_path), dict(self._ENTRY)) is None
        monkeypatch.undo()
        assert _names(Path(run_registry.registry_dir(str(tmp_path)))) == []

    @POSIX_LINKS
    def test_a_rewrite_takes_the_current_umask_not_the_old_entrys_mode(self, tmp_path):
        """Base made a fresh file each time; a cleanup record rewritten over an
        entry left 0666 must not stay that open (Codex's U40b review)."""
        path = Path(run_registry.write_entry(str(tmp_path), dict(self._ENTRY)))
        os.chmod(path, 0o666)
        previous = os.umask(0o077)
        try:
            run_registry.write_entry(str(tmp_path), dict(self._ENTRY))
        finally:
            os.umask(previous)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_the_entry_is_flushed_to_disk(self, tmp_path, monkeypatch):
        """What a sweep after a reboot reconciles: file and folder flushed."""
        calls = _fsyncs(monkeypatch)
        run_registry.write_entry(str(tmp_path), dict(self._ENTRY))
        assert len(calls) == (1 if os.name == "nt" else 2)
        assert run_registry.list_entries(str(tmp_path))[0]["run_id"] == "run-1"


class TestWorkerRefusalRecord:
    @pytest.fixture(autouse=True)
    def _worker(self, monkeypatch):
        monkeypatch.setenv(detach.WORKER_ENV, "1")

    @POSIX_LINKS
    def test_a_link_at_the_old_temp_name_is_not_written_through(self, tmp_path):
        victim = _victim(tmp_path)
        path = detach.worker_refusal_path(str(tmp_path), "run-1")
        planted = Path(f"{path}.{os.getpid()}.tmp")
        planted.symlink_to(victim)
        detach._record_worker_refusal(str(tmp_path), "run-1", "refused at the lock")
        assert victim.read_text(encoding="utf-8") == "victim"
        assert not Path(path).is_symlink()
        record = detach._read_worker_refusal(str(tmp_path), "run-1")
        assert record == {"run_id": "run-1", "pid": os.getpid(),
                          "message": "refused at the lock"}

    @POSIX_LINKS
    def test_a_rewrite_takes_the_current_umask(self, tmp_path):
        path = Path(detach.worker_refusal_path(str(tmp_path), "run-1"))
        detach._record_worker_refusal(str(tmp_path), "run-1", "first")
        os.chmod(path, 0o666)
        previous = os.umask(0o077)
        try:
            detach._record_worker_refusal(str(tmp_path), "run-1", "second")
        finally:
            os.umask(previous)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_a_failed_write_leaves_no_temp(self, tmp_path, monkeypatch):
        _refuse_replace(monkeypatch)
        detach._record_worker_refusal(str(tmp_path), "run-1", "refused")
        monkeypatch.undo()
        assert _names(tmp_path) == []


class TestTheCheckpoint:
    @POSIX_LINKS
    def test_owner_only_whatever_the_umask(self, tmp_path):
        previous = os.umask(0o022)
        try:
            status_file.write_status(str(tmp_path), {"state": "models_running"})
        finally:
            os.umask(previous)
        path = Path(status_file.status_path(str(tmp_path)))
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_lf_newlines_and_no_flush(self, tmp_path, monkeypatch):
        """Rewritten by every heartbeat, under the checkpoint lock: no fsync."""
        calls = _fsyncs(monkeypatch)
        status_file.write_status(str(tmp_path), {"state": "models_running"})
        assert calls == []
        raw = Path(status_file.status_path(str(tmp_path))).read_bytes()
        assert b"\n" in raw and b"\r\n" not in raw
        assert json.loads(raw)["state"] == "models_running"

    def test_a_failed_write_is_swallowed_and_leaves_no_temp(self, tmp_path, monkeypatch):
        status_file.write_status(str(tmp_path), {"state": "old"})
        _refuse_replace(monkeypatch)
        status_file.write_status(str(tmp_path), {"state": "new"})
        monkeypatch.undo()
        assert status_file.read_status(str(tmp_path))["state"] == "old"
        assert _names(tmp_path) == [".status.json", ".status.lock"]


class TestWriteSafetyRecords:
    @POSIX_LINKS
    def test_a_link_at_the_old_temp_name_is_not_written_through(self, tmp_path):
        victim = _victim(tmp_path)
        target = tmp_path / "store" / "manifest.json"
        target.parent.mkdir()
        planted = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        planted.symlink_to(victim)
        write_safety._atomic_write(str(target), '{"a": 1}\n')
        assert victim.read_text(encoding="utf-8") == "victim"
        assert not target.is_symlink()
        assert target.read_bytes() == b'{"a": 1}\n'
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_the_folder_is_made_and_the_write_flushed(self, tmp_path, monkeypatch):
        calls = _fsyncs(monkeypatch)
        target = tmp_path / "a" / "b" / "phase_dir.txt"
        write_safety._atomic_write(str(target), "/canonical\n")
        assert target.read_text(encoding="utf-8") == "/canonical\n"
        assert len(calls) == (1 if os.name == "nt" else 2)

    def test_a_failed_write_leaves_the_old_bytes_and_no_temp(self, tmp_path, monkeypatch):
        target = tmp_path / "manifest.json"
        target.write_bytes(b"old")
        _refuse_replace(monkeypatch)
        with pytest.raises(OSError):
            write_safety._atomic_write(str(target), "new")
        monkeypatch.undo()
        assert target.read_bytes() == b"old"
        assert _names(tmp_path) == ["manifest.json"]


class TestLandlockChangeLists:
    @pytest.fixture(autouse=True)
    def _state_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    @POSIX_LINKS
    def test_a_link_at_the_list_name_is_replaced_never_written_through(self, tmp_path):
        victim = _victim(tmp_path)
        lines = ["plugins/x/f%02d" % n for n in range(3)]
        body = ("\n".join(lines) + "\n").encode("utf-8")
        name = hashlib.sha256(body).hexdigest()[:24] + ".txt"
        directory = tmp_path / "state" / "multi-llm" / ld.HOME_CHANGE_LIST_DIR
        directory.mkdir(parents=True, mode=0o700)
        (directory / name).symlink_to(victim)
        path = ld._write_change_list(lines)
        assert path == str(directory / name)
        assert victim.read_text(encoding="utf-8") == "victim"
        assert not Path(path).is_symlink()
        assert Path(path).read_bytes() == body
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert _names(directory) == [name]

    def test_a_failed_write_names_no_file_and_leaves_no_temp(self, tmp_path, monkeypatch):
        _refuse_replace(monkeypatch)
        assert ld._write_change_list(["plugins/x/a", "plugins/x/b"]) is None
        monkeypatch.undo()
        assert _names(tmp_path / "state" / "multi-llm" / ld.HOME_CHANGE_LIST_DIR) == []
