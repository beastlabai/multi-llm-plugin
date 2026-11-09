"""``utils/atomic_io.py``: the one atomic write (docs/development/runtime.md#atomic-file-writes).

Each guarantee is proven with an injected fault or a watcher at the exact
step: the temp file's mode when its bytes are written, the mode at the
``fsync``, the order of file flush, rename and directory flush, and what a
failure at each step leaves behind.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import atomic_io  # noqa: E402
from utils.atomic_io import atomic_write_bytes, atomic_write_text  # noqa: E402

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _names(folder: Path):
    return sorted(p.name for p in folder.iterdir())


def _record_opens(monkeypatch):
    """Record every descriptor ``os.open`` hands out."""
    opened = []
    real_open = os.open

    def recording(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", recording)
    return opened


_REAL_UMASK = os.umask


def _fail_with(exception):
    def fail(*_args, **_kwargs):
        raise exception
    return fail


class _Watched:
    """A stream that records its file's mode as each write lands."""

    def __init__(self, stream, seen):
        self._stream, self._seen = stream, seen

    def write(self, data):
        self._seen.append(stat.S_IMODE(os.fstat(self._stream.fileno()).st_mode))
        return self._stream.write(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._stream.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _watch_writes(monkeypatch):
    """Record the temp file's mode at the moment its bytes are written."""
    seen = []
    real_fdopen = os.fdopen
    monkeypatch.setattr(os, "fdopen", lambda *a, **k: _Watched(real_fdopen(*a, **k), seen))
    return seen


def _watch_regular_fsyncs(monkeypatch):
    """Record the mode of every regular file at the moment it is fsynced."""
    seen = []
    real_fsync = os.fsync

    def watching(descriptor):
        info = os.fstat(descriptor)
        if stat.S_ISREG(info.st_mode):
            seen.append(stat.S_IMODE(info.st_mode))
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", watching)
    return seen


class TestTheTempFileIsNeverWider:
    @POSIX_ONLY
    def test_replacing_a_file_it_is_owner_only_until_the_mode_is_applied(
            self, tmp_path, monkeypatch, umask_022):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        target.chmod(0o644)
        before_chmod = []
        real_fchmod = os.fchmod

        def watching_fchmod(descriptor, mode):
            before_chmod.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
            return real_fchmod(descriptor, mode)

        monkeypatch.setattr(os, "fchmod", watching_fchmod)
        atomic_write_bytes(target, b"new")
        assert before_chmod == [0o600]
        assert _mode(target) == 0o644

    @POSIX_ONLY
    def test_a_private_file_stays_private_whatever_the_umask(
            self, tmp_path, monkeypatch, umask_022):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        target.chmod(0o600)
        seen = _watch_regular_fsyncs(monkeypatch)
        atomic_write_bytes(target, b"secret")
        assert seen == [0o600]
        assert _mode(target) == 0o600

    @POSIX_ONLY
    def test_an_explicit_mode_starts_owner_only_too(self, tmp_path, monkeypatch, umask_022):
        before_chmod = []
        real_fchmod = os.fchmod

        def watching_fchmod(descriptor, mode):
            before_chmod.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
            return real_fchmod(descriptor, mode)

        monkeypatch.setattr(os, "fchmod", watching_fchmod)
        atomic_write_bytes(tmp_path / "new.txt", b"x", mode=0o640)
        assert before_chmod == [0o600]
        assert _mode(tmp_path / "new.txt") == 0o640

    @POSIX_ONLY
    @pytest.mark.parametrize("umask, expected", [(0o022, 0o644), (0o027, 0o640), (0o077, 0o600)],
                             ids=["umask-022", "umask-027", "umask-077"])
    def test_a_new_file_is_owner_only_while_written_and_ends_at_the_umask_mode(
            self, tmp_path, monkeypatch, umask, expected):
        target = tmp_path / "new.txt"
        at_write = _watch_writes(monkeypatch)
        at_fsync = _watch_regular_fsyncs(monkeypatch)
        previous = os.umask(umask)
        try:
            # The umask is read off the empty file, never set (setting it
            # would race every other thread creating a file).
            monkeypatch.setattr(os, "umask", _fail_with(AssertionError("umask set")))
            atomic_write_bytes(target, b"private pending output")
        finally:
            monkeypatch.setattr(os, "umask", _REAL_UMASK)
            os.umask(previous)
        assert at_write == [0o600]
        assert at_fsync == [expected]
        assert _mode(target) == expected


    @POSIX_ONLY
    def test_a_filesystem_that_keeps_no_modes_still_takes_a_new_file(
            self, tmp_path, monkeypatch, umask_022):
        """FAT and some network mounts refuse every mode change: a new file
        is written at the mode it was created with, as a plain write does."""
        monkeypatch.setattr(os, "fchmod", _fail_with(PermissionError(errno.EPERM, "refused")))
        target = tmp_path / "new.txt"
        atomic_write_bytes(target, b"x")
        assert target.read_bytes() == b"x"
        assert _mode(target) == 0o644
        assert _names(tmp_path) == ["new.txt"]


class TestTheOrderOfFlushes:
    @POSIX_ONLY
    def test_the_mode_is_applied_before_the_fsync(self, tmp_path, monkeypatch, umask_022):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        target.chmod(0o640)
        seen = _watch_regular_fsyncs(monkeypatch)
        atomic_write_bytes(target, b"new")
        assert seen == [0o640]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync")
    def test_file_then_rename_then_directory(self, tmp_path, monkeypatch):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        events = []
        real_fsync, real_replace = os.fsync, os.replace

        def recording_fsync(descriptor):
            info = os.fstat(descriptor)
            events.append("dir" if stat.S_ISDIR(info.st_mode) else "file")
            if stat.S_ISDIR(info.st_mode):
                # The rename is already in the directory being flushed.
                assert target.read_bytes() == b"new"
            return real_fsync(descriptor)

        def recording_replace(source, destination):
            events.append("replace")
            return real_replace(source, destination)

        monkeypatch.setattr(os, "fsync", recording_fsync)
        monkeypatch.setattr(os, "replace", recording_replace)
        atomic_write_bytes(target, b"new")
        assert events == ["file", "replace", "dir"]

    def test_no_fsync_flushes_nothing(self, tmp_path, monkeypatch):
        def no_fsync(_descriptor):
            raise AssertionError("fsync=False must not flush")

        monkeypatch.setattr(os, "fsync", no_fsync)
        atomic_write_bytes(tmp_path / "f.txt", b"x", fsync=False)
        assert (tmp_path / "f.txt").read_bytes() == b"x"


class _Torn:
    """A stream that writes half its data, then fails like a full disk."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        self._stream.write(data[: len(data) // 2])
        self._stream.flush()
        raise OSError(28, "No space left on device")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class TestAFailureLeavesTheOldBytesAndNoTemp:
    @pytest.mark.parametrize("step", ["write", "fchmod", "fsync", "replace", "interrupt"])
    def test_at_every_step(self, tmp_path, monkeypatch, step):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old bytes\n")
        if step == "write":
            real_fdopen = os.fdopen
            monkeypatch.setattr(os, "fdopen", lambda *a, **k: _Torn(real_fdopen(*a, **k)))
        elif step == "fchmod":
            if not hasattr(os, "fchmod"):
                pytest.skip("no fchmod on this platform")
            monkeypatch.setattr(os, "fchmod", _fail_with(OSError("refused")))
        elif step == "fsync":
            monkeypatch.setattr(os, "fsync", _fail_with(OSError("I/O error")))
        elif step == "replace":
            monkeypatch.setattr(os, "replace", _fail_with(OSError("busy")))
        else:
            monkeypatch.setattr(os, "fsync", _fail_with(KeyboardInterrupt()))
        expected = KeyboardInterrupt if step == "interrupt" else OSError
        with pytest.raises(expected):
            atomic_write_bytes(target, b"new bytes that never land\n")
        monkeypatch.undo()
        assert target.read_bytes() == b"old bytes\n"
        assert _names(tmp_path) == ["f.txt"]

    def test_a_new_file_is_not_created_by_a_failed_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os, "replace", _fail_with(OSError("busy")))
        with pytest.raises(OSError):
            atomic_write_bytes(tmp_path / "new.txt", b"x")
        monkeypatch.undo()
        assert _names(tmp_path) == []

    @pytest.mark.parametrize("step", ["fdopen", "fstat"])
    def test_a_failure_before_the_stream_owns_the_descriptor_closes_it(
            self, tmp_path, monkeypatch, step):
        """Before ``os.fdopen`` returns, no stream closes the descriptor."""
        target = tmp_path / "new.txt"
        opened = _record_opens(monkeypatch)
        if step == "fdopen":
            monkeypatch.setattr(os, "fdopen", _fail_with(OSError(errno.EMFILE, "no stream")))
        else:  # the look at the empty file, right after it is created
            monkeypatch.setattr(os, "fstat", _fail_with(OSError(errno.EIO, "I/O error")))
        with pytest.raises(OSError):
            atomic_write_bytes(target, b"x")
        monkeypatch.undo()
        assert len(opened) == 1
        with pytest.raises(OSError):
            os.fstat(opened[0])
        assert _names(tmp_path) == []

    def test_a_descriptor_a_failed_fdopen_closed_is_not_closed_again(
            self, tmp_path, monkeypatch):
        """``fdopen`` closes the descriptor itself when it fails after wrapping
        it; the number may be another file's by then, and stays open."""
        other_path = tmp_path / "other"
        other_path.write_bytes(b"other")
        reused = []

        def closes_then_fails(descriptor, *_args, **_kwargs):
            os.close(descriptor)
            reused.append((descriptor, os.open(other_path, os.O_RDONLY)))
            raise MemoryError("no buffer")

        monkeypatch.setattr(os, "fdopen", closes_then_fails)
        with pytest.raises(MemoryError):
            atomic_write_bytes(tmp_path / "f.txt", b"x")
        monkeypatch.undo()
        (descriptor, other), = reused
        if other != descriptor:
            os.close(other)
            pytest.skip("the descriptor number was not reused")
        os.fstat(other)  # still open: not ours to close
        os.close(other)
        assert _names(tmp_path) == ["other"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
    def test_a_taken_temp_name_is_never_opened_or_removed(self, tmp_path, monkeypatch):
        """``O_EXCL``: a symlink planted at the temp name is not followed, and
        not deleted either: it is not this write's file."""
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        victim = tmp_path / "victim"
        victim.write_bytes(b"victim")
        monkeypatch.setattr(atomic_io.secrets, "token_hex", lambda _n: "cafe")
        planted = Path(atomic_io.temp_path_for(str(target)))
        planted.symlink_to(victim)
        with pytest.raises(FileExistsError):
            atomic_write_bytes(target, b"new")
        assert victim.read_bytes() == b"victim"
        assert target.read_bytes() == b"old"
        assert planted.is_symlink()


def _windows_replace_and_unlink(monkeypatch, *, replace_error=None):
    """Windows' rules, on any platform: ``os.replace`` onto a read-only
    file and ``os.unlink`` of one are refused (Windows deletes no read-only
    file); *replace_error*, when given, refuses every replace (a sharing
    violation: another process holds the target open)."""
    real_replace, real_unlink = os.replace, os.unlink
    unlinked = []

    def read_only(path):
        return not os.stat(path).st_mode & stat.S_IWUSR

    def replace(source, destination):
        if replace_error is not None:
            raise replace_error
        if os.path.exists(destination) and read_only(destination):
            raise PermissionError(errno.EACCES, "Access is denied", str(destination))
        return real_replace(source, destination)

    def unlink(path, *args, **kwargs):
        if read_only(path):
            raise PermissionError(errno.EACCES, "Access is denied", str(path))
        unlinked.append(os.path.basename(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "unlink", unlink)
    return unlinked


class TestWindowsRefusals:
    """The two refusals ``os.replace`` meets on Windows, emulated: the temp
    file is still removed and the old file left as it was."""

    def test_a_read_only_target_leaves_no_read_only_temp(self, tmp_path, monkeypatch):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        target.chmod(0o444)
        try:
            unlinked = _windows_replace_and_unlink(monkeypatch)
            with pytest.raises(PermissionError):
                atomic_write_bytes(target, b"new")
            monkeypatch.undo()
            assert len(unlinked) == 1 and unlinked[0].startswith(".f.txt.")
            assert _names(tmp_path) == ["f.txt"]
            assert target.read_bytes() == b"old"
            assert not os.stat(target).st_mode & stat.S_IWUSR
        finally:
            target.chmod(0o600)

    def test_a_sharing_violation_leaves_no_temp_and_is_the_error_raised(
            self, tmp_path, monkeypatch):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        violation = PermissionError(
            errno.EACCES, "The process cannot access the file because it is "
            "being used by another process", str(target))
        _windows_replace_and_unlink(monkeypatch, replace_error=violation)
        with pytest.raises(PermissionError) as raised:
            atomic_write_bytes(target, b"new")
        monkeypatch.undo()
        assert raised.value is violation
        assert _names(tmp_path) == ["f.txt"]
        assert target.read_bytes() == b"old"

    def test_a_temp_that_cannot_be_removed_does_not_hide_the_error(
            self, tmp_path, monkeypatch):
        """Another process holding the temp open too (an indexer): the
        cleanup's own refusal is swallowed, the replace's error raised."""
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        violation = PermissionError(errno.EACCES, "sharing violation", str(target))
        monkeypatch.setattr(os, "replace", _fail_with(violation))
        monkeypatch.setattr(os, "unlink", _fail_with(PermissionError(errno.EACCES, "in use")))
        with pytest.raises(PermissionError) as raised:
            atomic_write_bytes(target, b"new")
        monkeypatch.undo()
        assert raised.value is violation
        assert target.read_bytes() == b"old"

    @POSIX_ONLY
    def test_on_posix_a_read_only_target_is_replaced_and_stays_read_only(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        target.chmod(0o444)
        try:
            atomic_write_bytes(target, b"new")
            assert target.read_bytes() == b"new"
            assert _mode(target) == 0o444
            assert _names(tmp_path) == ["f.txt"]
        finally:
            target.chmod(0o600)


class TestModes:
    @POSIX_ONLY
    def test_keep_mode_carries_the_old_bits(self, tmp_path, umask_022):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        target.chmod(0o751)
        atomic_write_bytes(target, b"new")
        assert _mode(target) == 0o751

    @POSIX_ONLY
    def test_keep_mode_off_gives_the_umask_mode(self, tmp_path, umask_022):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        target.chmod(0o600)
        atomic_write_bytes(target, b"new", keep_mode=False)
        assert _mode(target) == 0o644

    @POSIX_ONLY
    def test_an_explicit_mode_wins_over_the_old_bits(self, tmp_path, umask_022):
        target = tmp_path / "f.txt"
        target.write_bytes(b"old")
        target.chmod(0o644)
        atomic_write_bytes(target, b"new", mode=0o600)
        assert _mode(target) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
class TestSymlinks:
    def _link(self, tmp_path):
        real = tmp_path / "real" / "plan.md"
        real.parent.mkdir()
        real.write_bytes(b"old")
        real.chmod(0o640)
        link = tmp_path / "plan.md"
        link.symlink_to(real)
        return real, link

    def test_followed_when_asked_the_link_survives(self, tmp_path):
        real, link = self._link(tmp_path)
        atomic_write_bytes(link, b"new", follow_symlinks=True)
        assert link.is_symlink() and os.readlink(link) == str(real)
        assert real.read_bytes() == b"new"
        assert _mode(real) == 0o640
        assert _names(real.parent) == ["plan.md"]

    def test_not_followed_the_link_is_replaced_and_its_target_left_alone(self, tmp_path):
        real, link = self._link(tmp_path)
        atomic_write_bytes(link, b"new")
        assert not link.is_symlink()
        assert link.read_bytes() == b"new"
        assert real.read_bytes() == b"old"
        assert _mode(link) == 0o640  # read through the link


class TestText:
    def test_newlines_are_lf_and_utf8_by_default(self, tmp_path):
        target = tmp_path / "f.txt"
        written = atomic_write_text(target, "a\nb é\n")
        assert written == "a\nb é\n".encode("utf-8")
        assert target.read_bytes() == written

    def test_platform_newlines_match_write_text(self, tmp_path):
        target = tmp_path / "f.txt"
        expected = tmp_path / "expected.txt"
        expected.write_text("a\nb\n", encoding="utf-8")
        atomic_write_text(target, "a\nb\n", newline=os.linesep)
        assert target.read_bytes() == expected.read_bytes()

    def test_temp_prefix_names_the_temp_file(self, tmp_path, monkeypatch):
        sources = []
        real_replace = os.replace

        def recording(source, destination):
            sources.append(os.path.basename(source))
            return real_replace(source, destination)

        monkeypatch.setattr(os, "replace", recording)
        atomic_write_text(tmp_path / "f.txt", "x", temp_prefix=".mine-")
        atomic_write_text(tmp_path / "f.txt", "y")
        assert sources[0].startswith(".mine-") and sources[0].endswith(".tmp")
        assert sources[1].startswith(".f.txt.") and sources[1].endswith(".tmp")
