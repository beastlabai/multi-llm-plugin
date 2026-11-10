"""A restore stages the file it puts back through one descriptor, never by its name.

The rollback restore (``write_safety._replace_with_payload``) runs after a
write-capable runner had the working tree, and whoever can write the file's
folder can swap the staged file's predictable name for a link at any moment:
a hard link or a symlink to a file outside the repository, or another name of
its own. These tests are that writer (docs/development/write-safety.md,
restore). Each adversarial case runs on both POSIX staging ways: an unnamed
``O_TMPFILE`` file, named only once its bytes are checked and its metadata set
(Linux), and the file made by its name exclusively (macOS, a filesystem
without ``O_TMPFILE``). Each must leave the outside file's bytes, mode and
times as they were, put nothing back, say so, and keep the snapshot. Once
the staged file has a name it is never removed (nothing removes a file, only
a name, which can be swapped in between a check and the removal): it is
left and named for the user. The folder the file goes back in is reached
from the root without following a link and must still be the folder at its
path before the file is published and after; BSD file flags go through the
staged file's descriptor or not at all; ``restore_path``'s symlink follows
the same rules.

Then the Windows staging (a read-only payload, Python before 3.12) over the
emulated handle calls, and Codex's U82 round 3 repros, which reproduced on the
base restore and must not any more.

No model runs. POSIX only: Windows runs the same restore cases natively in
``tests/test_write_sweep_sees_the_same.py`` (the ``restoring`` fixture).
"""

from __future__ import annotations

import ast
import errno
import hashlib
import inspect
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tests.test_write_executor import env  # noqa: F401
from tests.test_write_sweep_sees_the_same import (
    _before_bind,
    _HeldCopy,
    _restore_tmp,
    _restoring_by_copy_file2,
    _snapshot_of,
    _staging_by_handle,
    _kept_for_the_user,
    _staging_token_pinned,  # noqa: F401 - autouse: the staging name is known
)
from utils import write_safety

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX staging and the emulated Windows calls; Windows "
                            "runs the restore cases natively in the sweep tests")

#: What the runner left in the file the restore rolls back.
RUNNER_CHANGE = b"runner change\n"
#: The file's mode and times when snapshotted, which the payload keeps.
FILE_MODE = 0o640
FILE_TIMES = (1_600_000_000_000_000_000, 1_650_000_000_987_654_300)


def _tmpfile_here(folder: Path) -> bool:
    """Whether *folder*'s filesystem makes ``O_TMPFILE`` files and
    ``/proc/self/fd`` can name one."""
    flag = getattr(os, "O_TMPFILE", 0)
    if not flag or not os.path.isdir("/proc/self/fd"):
        return False
    try:
        handle = os.open(folder, flag | os.O_RDWR, 0o600)
    except OSError:
        return False
    os.close(handle)
    return True


@pytest.fixture(params=["O_TMPFILE", "named"])
def staging(request, env, monkeypatch):
    """Each POSIX way the staged file is made: unnamed (``O_TMPFILE``), or
    by its name exclusively (``write_safety._O_TMPFILE`` off)."""
    if request.param == "named":
        monkeypatch.setattr(write_safety, "_O_TMPFILE", 0, raising=False)
    elif not _tmpfile_here(env.root / "src"):
        pytest.skip("no O_TMPFILE here (Linux, and not every filesystem)")
    monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
    return request.param


def _rolled_back(env):
    """A snapshot of ``src/a.py`` (mode :data:`FILE_MODE`, times
    :data:`FILE_TIMES`) that the runner then changed; the snapshot, the
    file, its payload and the staged file's name."""
    target = env.root / "src" / "a.py"
    target.chmod(FILE_MODE)
    os.utime(target, ns=FILE_TIMES)
    snapshot, payload = _snapshot_of(env)
    target.write_bytes(RUNNER_CHANGE)
    return snapshot, target, payload, _restore_tmp(target)


def _read(path) -> bytes:
    """*path*'s bytes, read without touching its access time where the
    platform can (``O_NOATIME``)."""
    handle = os.open(path, os.O_RDONLY | getattr(os, "O_NOATIME", 0))
    try:
        return b"".join(iter(lambda: os.read(handle, 1 << 16), b""))
    finally:
        os.close(handle)


def _state(path):
    """What a restore must never change of a file outside the repository:
    its bytes, mode and times (the access time where it reads unchanged)."""
    info = os.stat(path)
    atime = info.st_atime_ns if hasattr(os, "O_NOATIME") else None
    return _read(path), stat.S_IMODE(info.st_mode), info.st_mtime_ns, atime


def _outside(tmp_path, mode=0o600):
    """A file outside the repository, on its filesystem, and its state."""
    victim = tmp_path / "outside.txt"
    victim.write_bytes(b"outside the repository\n")
    victim.chmod(mode)
    os.utime(victim, ns=(1_500_000_000_000_000_000, 1_500_000_000_123_456_700))
    return victim, _state(victim)


def _at_the_flush(monkeypatch, then, *, before=False) -> list:
    """Run *then* (the descriptor flushed) once, at the restore's first
    ``os.fsync``: the staged file's, once its bytes are written. After the
    flush, or before it with *before*. The list holds the descriptor once it
    ran."""
    real = os.fsync
    ran: list = []

    def fsync(handle):
        if ran:
            return real(handle)
        ran.append(handle)
        if before:
            then(handle)
            return real(handle)
        result = real(handle)
        then(handle)
        return result

    monkeypatch.setattr(os, "fsync", fsync)
    return ran


def _reach(tmp: Path, handle: int) -> str:
    """How a writer of the folder reaches the staged file: by its name while
    it has one, else (an unnamed ``O_TMPFILE`` file) through ``/proc``, as
    any process of the same user can."""
    return str(tmp) if os.path.lexists(tmp) else f"/proc/self/fd/{handle}"


def _swap_for(tmp: Path, make) -> None:
    """The staged file's name swapped for what *make* (the name) puts there;
    planted there when the staged file has no name yet."""
    if os.path.lexists(tmp):
        os.unlink(tmp)
    make(tmp)


def _left_when_named(state, tmp, staging) -> None:
    """A staged file that had a name (made by it) is left there and named
    for the user to remove; an unnamed one (``O_TMPFILE``) leaves nothing."""
    if staging == "named":
        assert os.path.lexists(tmp), state
        assert f"was left at {tmp}" in state
        _kept_for_the_user(state)
    else:
        assert not os.path.lexists(tmp), state


def _refused(report, target, payload, snapshot) -> str:
    """The restore put nothing back, said so, and kept the snapshot; its
    state for ``src/a.py``."""
    state = report["src/a.py"]
    assert state.startswith("skipped:"), report
    assert not target.is_symlink() and target.read_bytes() == RUNNER_CHANGE
    entry = snapshot.entry_for("src/a.py")
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == entry.sha256
    return state


# ---------------------------------------------------------------------------
# The staged file swapped, rewritten or linked to (POSIX)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("link", ["hard link", "symlink"])
def test_a_link_swapped_in_before_the_metadata_never_reaches_the_outside_file(
    env, tmp_path, monkeypatch, staging, link,
):
    """The staged file's name swapped, once its bytes are flushed, for a
    link to a file outside the repository: base then set the payload's mode
    and times by that name (``shutil.copystat``, which follows a symlink),
    so on the outside file, and renamed the link over the file, reported
    ``restored``. Now the mode and times go through the staged file's own
    descriptor, and the rename is refused for a name that is not the staged
    file (named: ``lstat`` against ``fstat``) or never made over one already
    there (unnamed: ``linkat``). The outside file keeps its bytes, mode and
    times, the link is left there, named, and the snapshot is kept."""
    snapshot, target, payload, tmp = _rolled_back(env)
    victim, before = _outside(tmp_path)

    def swap(_handle):
        _swap_for(tmp, (lambda name: os.link(victim, name)) if link == "hard link"
                  else (lambda name: os.symlink(victim, name)))

    ran = _at_the_flush(monkeypatch, swap)
    report = snapshot.restore()

    assert ran
    state = _refused(report, target, payload, snapshot)
    assert not os.path.samefile(target, victim)
    assert _state(victim) == before
    assert os.path.samefile(tmp, victim)
    assert os.path.islink(tmp) is (link == "symlink")
    assert "was left as it is" in state
    assert state.endswith("check what is left there and delete it by hand")


def test_the_staged_file_rewritten_before_its_flush_is_refused(env, monkeypatch, staging):
    """A writer reaching the staged file (by its name, or an unnamed one
    through ``/proc``) rewrites it before the flush: base never read it
    back, so it put those bytes back as ``restored``. The bytes read back
    through the descriptor after the flush must be the payload's, so nothing
    is put back; a named staged file is left and named, never removed."""
    snapshot, target, payload, tmp = _rolled_back(env)

    def rewrite(handle):
        with open(_reach(tmp, handle), "r+b") as writer:
            writer.truncate(0)
            writer.write(b"not the hashed payload\n")

    ran = _at_the_flush(monkeypatch, rewrite, before=True)
    report = snapshot.restore()

    assert ran
    state = _refused(report, target, payload, snapshot)
    assert "does not hold the bytes" in state
    _left_when_named(state, tmp, staging)


def test_another_name_given_to_the_staged_file_is_refused(
    env, tmp_path, monkeypatch, staging,
):
    """A writer gives the staged file a second name outside the repository
    (``link(2)``, through ``/proc`` for an unnamed one): base renamed it over
    the file anyway, so the repository's file was then that outside name's
    file too. The staged file must have no name but its own, so nothing is
    put back; the outside name, the writer's, is left, and so is the staged
    file's own where it had one, named."""
    snapshot, target, payload, tmp = _rolled_back(env)
    extra = tmp_path / "another-name"
    refused_link = []

    def link(handle):
        try:
            os.link(_reach(tmp, handle), extra, follow_symlinks=True)
        except OSError as exc:
            # Python before 3.11 links /proc/self/fd/<n> itself rather than
            # following it (EXDEV): the writer this test plays cannot act.
            refused_link.append(exc)

    ran = _at_the_flush(monkeypatch, link)
    report = snapshot.restore()

    assert ran
    if refused_link:
        pytest.skip(f"this Python cannot give an unnamed file a name: {refused_link[0]}")
    state = _refused(report, target, payload, snapshot)
    assert "was given another name" in state
    assert not os.path.samefile(target, extra)
    assert extra.exists()
    _left_when_named(state, tmp, staging)


def test_a_swap_between_the_last_check_and_the_rename_is_reported(
    env, tmp_path, monkeypatch, staging,
):
    """The window left (docs/development/write-safety.md, what restore
    leaves open): between the last check of the staged file's name and
    ``rename(2)``. A hard link to an outside file swapped in there is what
    the rename moves over the file, as the writer could move it itself; the
    outside file keeps its bytes, mode and times (nothing is set by a name),
    and the restore, finding the file is not the one it staged, says so
    instead of ``restored`` (base reported it restored)."""
    snapshot, target, payload, tmp = _rolled_back(env)
    victim, before = _outside(tmp_path)
    real = os.replace
    fired: list = []

    def replace(source, destination, *args, **kwargs):
        if not fired and os.path.basename(os.fspath(destination)) == target.name:
            fired.append(True)
            _swap_for(tmp, lambda name: os.link(victim, name))
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    report = snapshot.restore()

    assert fired
    state = report["src/a.py"]
    assert state.startswith("skipped:") and "did not name the file put back" in state
    assert os.path.samefile(target, victim), "the swap, not the restore, put it there"
    assert _state(victim) == before
    entry = snapshot.entry_for("src/a.py")
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == entry.sha256


def test_a_failed_rename_leaves_the_staged_name_for_the_user(
    env, monkeypatch, staging,
):
    """A rename the file system refuses (a directory put at the path once
    the restore looked: one there before is refused with nothing staged,
    ``test_a_directory_at_the_path_stages_nothing``): the staged file, named
    by then either way, is left at its name and named for the user, never
    removed. Removing it would be a check, then ``unlink(2)`` by the name,
    and a process that swaps its own file in between (here at the removal
    itself) would lose it, as it did on base and in U87's first round. The
    directory is left as it is, the error names both paths in full, and the
    snapshot is kept."""
    snapshot, target, payload, tmp = _rolled_back(env)

    def directory_now(_handle):
        target.unlink()
        target.mkdir()
        (target / "inside.txt").write_bytes(b"kept\n")

    _at_the_flush(monkeypatch, directory_now)
    removals: list = []

    def removing(real):
        def call(path, *args, **kwargs):
            if os.path.basename(os.fsdecode(path)) == tmp.name:
                removals.append(path)
                # Another process's file swapped in just before the removal.
                os.rename(tmp, tmp.with_name("staged-file-moved-aside"))
                tmp.write_bytes(b"another process's file\n")
            return real(path, *args, **kwargs)
        return call

    monkeypatch.setattr(os, "unlink", removing(os.unlink))
    monkeypatch.setattr(os, "remove", removing(os.remove))
    report = snapshot.restore()

    state = report["src/a.py"]
    assert state.startswith("skipped:"), report
    assert str(tmp) in state and str(target) in state
    assert removals == []
    assert f"was left at {tmp}" in state
    _kept_for_the_user(state)
    entry = snapshot.entry_for("src/a.py")
    assert hashlib.sha256(tmp.read_bytes()).hexdigest() == entry.sha256
    assert (target / "inside.txt").read_bytes() == b"kept\n"
    assert payload.exists()


# ---------------------------------------------------------------------------
# How the staged file is made
# ---------------------------------------------------------------------------

def test_an_unnamed_staged_file_has_no_name_until_it_is_checked(env, monkeypatch):
    """Where Linux can, the staged file is made with no name
    (``O_TMPFILE``): while its bytes are written and flushed nothing can
    reach it by a name. It is named only once checked, and the file put back
    holds the payload's bytes, mode and times, with nothing left beside it."""
    if not _tmpfile_here(env.root / "src"):
        pytest.skip("no O_TMPFILE here (Linux, and not every filesystem)")
    monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
    original = (env.root / "src" / "a.py").read_bytes()
    snapshot, target, _payload, tmp = _rolled_back(env)
    seen: list = []

    def look(handle):
        seen.append((os.path.lexists(tmp), os.fstat(handle).st_nlink))

    _at_the_flush(monkeypatch, look)
    assert snapshot.restore() == {"src/a.py": "restored"}

    assert seen == [(False, 0)]
    assert target.read_bytes() == original
    info = os.lstat(target)
    assert stat.S_IMODE(info.st_mode) == FILE_MODE and info.st_nlink == 1
    assert info.st_mtime_ns == FILE_TIMES[1]
    assert not os.path.lexists(tmp)


@pytest.mark.parametrize("why", ["the filesystem refuses it", "no /proc to name it"])
def test_without_an_unnamed_file_the_staged_file_is_made_by_its_name(
    env, tmp_path, monkeypatch, why,
):
    """Where ``O_TMPFILE`` is refused (an old kernel, a filesystem without
    it) or ``/proc`` cannot name such a file, the restore stages it by its
    name instead, exclusively, and puts it back as well."""
    flag = getattr(os, "O_TMPFILE", 0)
    if not flag:
        pytest.skip("no O_TMPFILE on this platform")
    monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
    original = (env.root / "src" / "a.py").read_bytes()
    snapshot, target, _payload, tmp = _rolled_back(env)
    attempts: list = []
    real_open = os.open

    def opening(path, flags, *args, **kwargs):
        if flags & flag == flag:
            attempts.append(path)
            if why == "the filesystem refuses it":
                raise OSError(95, "Operation not supported", path)
        return real_open(path, flags, *args, **kwargs)

    if why == "no /proc to name it":
        if not _tmpfile_here(env.root / "src"):
            pytest.skip("no O_TMPFILE here")
        monkeypatch.setattr(write_safety, "_PROC_SELF_FD", str(tmp_path / "no-proc"),
                            raising=False)
    seen: list = []

    def look(handle):
        seen.append(os.path.samestat(os.lstat(tmp), os.fstat(handle)))

    _at_the_flush(monkeypatch, look)
    with monkeypatch.context() as patched:
        patched.setattr(os, "open", opening)
        report = snapshot.restore()

    assert report == {"src/a.py": "restored"}
    assert len(attempts) == 1 and seen == [True]
    assert target.read_bytes() == original
    assert stat.S_IMODE(os.lstat(target).st_mode) == FILE_MODE
    assert not os.path.lexists(tmp)


@pytest.mark.parametrize("case", [
    "restored", "hard link swapped in", "symlink swapped in", "rewritten",
    "another name",
])
def test_the_staged_file_is_never_reached_by_its_path(
    env, tmp_path, monkeypatch, staging, case,
):
    """Nothing the restore does reaches the staged file by its full path,
    or by its own name other than relative to the folder's descriptor,
    whatever it finds (seamed: the ``os`` and ``shutil`` calls that open,
    examine, change, link, rename or remove a file record any path naming it,
    except the writer's own calls here)."""
    snapshot, target, _payload, tmp = _rolled_back(env)
    victim, _before = _outside(tmp_path)
    full, own = os.path.normcase(str(tmp)), tmp.name
    attacking: list = []

    def attack(handle):
        attacking.append(True)
        try:
            if case == "hard link swapped in":
                _swap_for(tmp, lambda name: os.link(victim, name))
            elif case == "symlink swapped in":
                _swap_for(tmp, lambda name: os.symlink(victim, name))
            elif case == "rewritten":
                with open(_reach(tmp, handle), "r+b") as writer:
                    writer.write(b"#")
            elif case == "another name":
                os.link(_reach(tmp, handle), tmp_path / "extra", follow_symlinks=True)
        finally:
            attacking.pop()

    _at_the_flush(monkeypatch, attack, before=case == "rewritten")
    named: list = []

    def recorder(name, real):
        def call(*args, **kwargs):
            relative = any(kwargs.get(key) is not None
                           for key in ("dir_fd", "src_dir_fd", "dst_dir_fd"))
            for argument in args[:2]:
                if attacking or not isinstance(argument, (str, bytes, os.PathLike)):
                    continue
                text = os.fsdecode(argument)
                if os.path.normcase(text) == full or (text == own and not relative):
                    named.append(name)
            return real(*args, **kwargs)
        return call

    with monkeypatch.context() as patched:
        for module, names in (
            (os, ("open", "stat", "lstat", "chmod", "lchmod", "utime", "truncate",
                  "chflags", "lchflags", "setxattr", "removexattr", "link", "symlink",
                  "unlink", "remove", "rename", "replace")),
            (shutil, ("copystat", "copymode", "copy", "copy2", "copyfile", "move")),
        ):
            for name in names:
                if hasattr(module, name):
                    patched.setattr(module, name, recorder(name, getattr(module, name)))
        report = snapshot.restore()

    assert named == [], named
    if case == "restored":
        assert report == {"src/a.py": "restored"}
    else:
        assert report["src/a.py"].startswith("skipped:"), report
        assert target.read_bytes() == RUNNER_CHANGE


def test_the_staging_names_the_staged_file_only_relative_to_its_folder():
    """Statically, for the functions that hold the staged file or link
    (POSIX): no ``shutil`` call, no ``os`` call that changes a file by its
    path (BSD file flags included: they go through the descriptor), none
    that removes a name, and every ``os`` call that takes a name passes the
    folder's descriptor (``dir_fd``, ``src_dir_fd``, ``dst_dir_fd``)."""
    by_path = {"chmod", "lchmod", "truncate", "chflags", "lchflags", "setxattr",
               "removexattr", "mkdir", "rmdir", "unlink", "remove"}
    named = {"open", "stat", "lstat", "rename", "replace", "link", "symlink",
             "readlink"}
    relative = {"dir_fd", "src_dir_fd", "dst_dir_fd"}
    for function in (write_safety._stage_and_rename, write_safety._publish_staged,
                     write_safety._mode_at_name, write_safety._staged,
                     write_safety._link_unnamed, write_safety._check_staged,
                     write_safety._leave_staged, write_safety._set_flags,
                     write_safety._link_and_rename, write_safety._link_at):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        found = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if not isinstance(owner, ast.Name):
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            if owner.id == "shutil":
                found.add(f"shutil.{node.func.attr}")
            elif owner.id == "os" and node.func.attr in by_path:
                found.add(f"os.{node.func.attr}")
            elif owner.id == "os" and node.func.attr in named and not keywords & relative:
                found.add(f"os.{node.func.attr} without the folder's descriptor")
        assert not found, (function.__name__, found)


# ---------------------------------------------------------------------------
# The folder the file goes back in, replaced (POSIX)
# ---------------------------------------------------------------------------

def _replace_folder(folder: Path, moved: Path) -> None:
    """A writer of the folder above *folder* moves it to *moved* and puts
    another folder at its path, with another ``a.py``."""
    folder.rename(moved)
    folder.mkdir()
    (folder / "a.py").write_bytes(ANOTHER_FOLDERS_FILE)


#: The file in the folder put at the target's folder's path.
ANOTHER_FOLDERS_FILE = b"another folder's file\n"


def _moved_name(moved: Path, name: str) -> str:
    """*name* in the folder moved to *moved*, as the restore finds its path
    now (``/proc/self/fd``, resolved)."""
    return os.path.join(os.path.realpath(moved), name)


def test_the_folder_replaced_during_the_flush_is_refused(
    env, tmp_path, monkeypatch, staging,
):
    """Codex review 1 of U87, point 1: the target's folder moved away
    (outside the repository) and another folder put at its path while the
    staged file is flushed. U87's first round published the file in the
    moved folder through the descriptor it held and reported ``restored``,
    though the repository's file held other bytes. Now the folder must still
    be the one at its path (a fresh walk from the root, without following a
    link) before the file is published: nothing is put back, either folder's
    ``a.py`` keeps its bytes, the snapshot is kept, and a staged file that
    had a name is left in the moved folder, named where it is now."""
    snapshot, target, payload, tmp = _rolled_back(env)
    moved = tmp_path / "moved-folder"
    ran = _at_the_flush(monkeypatch, lambda _handle: _replace_folder(target.parent, moved))

    report = snapshot.restore()

    assert ran
    state = report["src/a.py"]
    assert state.startswith("skipped:"), report
    assert "is no longer the folder the restore opened" in state
    assert target.read_bytes() == ANOTHER_FOLDERS_FILE
    assert (moved / "a.py").read_bytes() == RUNNER_CHANGE
    entry = snapshot.entry_for("src/a.py")
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == entry.sha256
    assert not os.path.lexists(tmp)
    left = moved / tmp.name
    if staging == "named":
        assert os.path.lexists(left)
        assert f"was left at {_moved_name(moved, tmp.name)}" in state
        _kept_for_the_user(state)
    else:
        assert not os.path.lexists(left)


def test_the_folder_replaced_just_before_the_rename_is_reported(
    env, tmp_path, monkeypatch, staging,
):
    """The window left between the last check of the folder and
    ``rename(2)``: the folder replaced there, the rename puts the file back
    in the moved folder. The check after the rename finds the folder at the
    path is not the one it held, so the restore says so, naming where the
    file went, instead of ``restored`` (U87's first round: ``restored``)."""
    snapshot, target, payload, _tmp = _rolled_back(env)
    original = snapshot.entry_for("src/a.py").sha256
    moved = tmp_path / "moved-folder"
    real = os.replace
    fired: list = []

    def replace(source, destination, *args, **kwargs):
        if not fired and os.path.basename(os.fspath(destination)) == target.name:
            fired.append(True)
            _replace_folder(target.parent, moved)
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    report = snapshot.restore()

    assert fired
    state = report["src/a.py"]
    assert state.startswith("skipped:") and "was moved or replaced" in state, report
    assert f"the file put back is at {_moved_name(moved, 'a.py')}" in state
    assert target.read_bytes() == ANOTHER_FOLDERS_FILE
    assert hashlib.sha256((moved / "a.py").read_bytes()).hexdigest() == original
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == original


def test_a_link_planted_above_the_folder_is_never_followed(env, tmp_path, monkeypatch, staging):
    """A folder above the target's (``src`` for ``src/pkg/b.py``) swapped for
    a link to a copy outside the repository once the path was revalidated:
    the folder is reached from the root one component at a time without
    following a link, so the restore refuses and the outside copy keeps its
    bytes. (U87's first round opened the target's folder by its full path,
    refusing a link only at the folder's own name.)"""
    nested = env.root / "src" / "pkg" / "b.py"
    nested.parent.mkdir()
    nested.write_bytes(b"b before\n")
    snapshot, _payload = _snapshot_of(env, "src/pkg/b.py")
    nested.write_bytes(RUNNER_CHANGE)
    outside = tmp_path / "outside-src"
    real = write_safety._revalidated_target

    def revalidated(*args, **kwargs):
        found = real(*args, **kwargs)
        shutil.copytree(env.root / "src", outside, symlinks=True)
        shutil.rmtree(env.root / "src")
        (env.root / "src").symlink_to(outside, target_is_directory=True)
        return found

    monkeypatch.setattr(write_safety, "_revalidated_target", revalidated)
    report = snapshot.restore()

    state = report["src/pkg/b.py"]
    assert state.startswith("skipped:"), report
    assert "without following a link" in state
    assert (outside / "pkg" / "b.py").read_bytes() == RUNNER_CHANGE
    assert not os.path.lexists(_restore_tmp(outside / "pkg" / "b.py"))


# ---------------------------------------------------------------------------
# BSD and macOS file flags, emulated (POSIX)
# ---------------------------------------------------------------------------

class _WithFlags:
    """An ``os.stat_result`` with ``st_flags``, as BSD and macOS report it."""

    def __init__(self, info, flags):
        self._info, self.st_flags = info, flags

    def __getattr__(self, name):
        return getattr(self._info, name)


def _payload_has_flags(monkeypatch, payload: Path, flags: int) -> None:
    """The payload's descriptor reports *flags*; Linux has no file flags."""
    real = os.fstat
    known = os.stat(payload)

    def fstat(handle):
        info = real(handle)
        if (info.st_dev, info.st_ino) == (known.st_dev, known.st_ino):
            return _WithFlags(info, flags)
        return info

    monkeypatch.setattr(os, "fstat", fstat)


def _flags_by_path(monkeypatch, then) -> list:
    """``os.chflags`` and ``os.lchflags`` as BSD and macOS have them (a
    link not followed), running *then* (the path) first; each call's inode
    is recorded. U87's first round set flags this way."""
    reached: list = []

    def chflags(path, flags, *, follow_symlinks=True):
        then(path)
        reached.append(os.stat(path, follow_symlinks=follow_symlinks).st_ino)

    monkeypatch.setattr(os, "chflags", chflags, raising=False)
    monkeypatch.setattr(os, "lchflags", lambda path, flags: chflags(path, flags, follow_symlinks=False),
                        raising=False)
    monkeypatch.setattr(os, "supports_follow_symlinks", {*os.supports_follow_symlinks, chflags})
    return reached


def test_file_flags_go_through_the_staged_files_descriptor(
    env, tmp_path, monkeypatch, staging,
):
    """Codex review 1 of U87, point 2 (emulated: no BSD or macOS here): a
    payload with file flags, and the staged file's name swapped for a hard
    link to an outside file just before the flags are set. U87's first round
    set them by that name (``lchflags``: a hard link is no link to it), so on
    the outside file. Now they go through the staged file's descriptor
    (``fchflags``), reaching the staged file alone; the swap is then refused
    and left, and the outside file keeps its bytes, mode and times."""
    snapshot, target, payload, tmp = _rolled_back(env)
    victim, before = _outside(tmp_path)
    _payload_has_flags(monkeypatch, payload, stat.UF_HIDDEN)

    def swap(_path=None):
        _swap_for(tmp, lambda name: os.link(victim, name))

    by_path = _flags_by_path(monkeypatch, swap)
    by_descriptor: list = []

    def fchflags(handle, flags):
        swap()
        by_descriptor.append((os.fstat(handle).st_ino, flags))

    monkeypatch.setattr(write_safety, "_fchflags", lambda: fchflags, raising=False)
    report = snapshot.restore()

    victim_inode = os.stat(victim).st_ino
    assert by_path == [], "flags were set by the staged file's name"
    assert by_descriptor and all(inode != victim_inode for inode, _ in by_descriptor)
    assert by_descriptor[0][1] == stat.UF_HIDDEN
    _refused(report, target, payload, snapshot)
    assert os.path.samefile(tmp, victim) and _state(victim) == before


@pytest.mark.parametrize("why", ["no libc call", "a libc stub"])
def test_without_a_descriptor_call_the_flags_are_left_off_and_named(
    env, monkeypatch, capsys, staging, why,
):
    """Where ``fchflags`` cannot be called (Python has none, and libc's could
    not be loaded, or is a stub failing with ``ENOSYS``, as glibc's), the
    file goes back without its flags rather than have them set by its name,
    and a warning names them (emulated)."""
    original = (env.root / "src" / "a.py").read_bytes()
    snapshot, target, payload, _tmp = _rolled_back(env)
    _payload_has_flags(monkeypatch, payload, stat.UF_HIDDEN | stat.SF_ARCHIVED)
    by_path = _flags_by_path(monkeypatch, lambda _path: None)

    def stub(_handle, _flags):
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS))

    monkeypatch.setattr(write_safety, "_fchflags",
                        (lambda: None) if why == "no libc call" else (lambda: stub),
                        raising=False)

    report = snapshot.restore()

    assert report == {"src/a.py": "restored"}
    assert by_path == []
    assert target.read_bytes() == original
    warning = capsys.readouterr().err
    assert f"WARNING: {target} is restored without its file flags" in warning
    assert "SF_ARCHIVED|UF_HIDDEN" in warning


# ---------------------------------------------------------------------------
# restore_path's symlink, under the same rules (POSIX)
# ---------------------------------------------------------------------------

#: What the link restore_path puts back reads.
LINK_TEXT = "intended-target"


@pytest.fixture(params=["by descriptor", "by path"])
def link_ways(request, monkeypatch):
    """Each way ``restore_path`` puts a link back: relative to the held
    folder (Linux, macOS), or by full paths (Windows, run here on POSIX)."""
    if request.param == "by path":
        monkeypatch.setattr(write_safety, "_STAGES_BY_DESCRIPTOR", False, raising=False)
    return request.param


def _link_restore(env, tmp_path):
    """A saved link (``restore_path``'s source) for ``src/link``, which the
    runner left as a regular file; the target and its staged name."""
    source = tmp_path / "saved-link"
    source.symlink_to(LINK_TEXT)
    target = env.root / "src" / "link"
    target.write_bytes(RUNNER_CHANGE)
    return str(source), target, _restore_tmp(target)


def test_restore_path_puts_a_link_back(env, tmp_path, link_ways):
    source, target, tmp = _link_restore(env, tmp_path)

    assert write_safety.restore_path(str(env.root), "src/link", source) == "restored"

    assert target.is_symlink() and os.readlink(target) == LINK_TEXT
    assert not os.path.lexists(tmp)


@pytest.mark.parametrize("when", ["once made", "at the rename"])
def test_restore_path_never_reports_a_link_swapped_out_as_restored(
    env, tmp_path, monkeypatch, link_ways, when,
):
    """Codex review 1 of U87, the symlink branch: the link made at the
    staged name swapped for another process's regular file, once made or
    just before the rename. Base and U87's first round renamed that file
    over the target and reported ``restored``. Now the name must still be
    the link made, reading its text, just before the rename (else nothing
    is put back and the file is left, named), and the target must be once
    renamed (else the restore says so)."""
    source, target, tmp = _link_restore(env, tmp_path)
    foreign = b"another process's file\n"

    def swap():
        os.unlink(tmp)
        tmp.write_bytes(foreign)

    fired: list = []
    if when == "once made":
        real_symlink = os.symlink

        def symlink(text, name, *args, **kwargs):
            real_symlink(text, name, *args, **kwargs)
            if os.path.basename(os.fsdecode(name)) == tmp.name and not fired:
                fired.append(True)
                swap()

        monkeypatch.setattr(os, "symlink", symlink)
    else:
        real_replace = os.replace

        def replace(src, dst, *args, **kwargs):
            if os.path.basename(os.fsdecode(src)) == tmp.name and not fired:
                fired.append(True)
                swap()
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", replace)

    state = write_safety.restore_path(str(env.root), "src/link", source)

    assert fired
    assert state.startswith("skipped:"), state
    if when == "once made":
        assert "was no longer the link made there" in state
        assert target.read_bytes() == RUNNER_CHANGE and tmp.read_bytes() == foreign
        assert state.endswith("check what is left there and delete it by hand")
    else:
        assert "did not name the link put back" in state
        # The swap, not the restore, put it there; it was never removed.
        assert target.read_bytes() == foreign


def test_restore_path_refuses_a_link_whose_folder_was_replaced(env, tmp_path, monkeypatch):
    """The link's folder moved away and another put at its path just before
    the rename: nothing is put back in the new folder, and the link made is
    left in the moved one, named where it is now."""
    source, target, tmp = _link_restore(env, tmp_path)
    moved = tmp_path / "moved-folder"
    real_symlink = os.symlink

    def symlink(text, name, *args, **kwargs):
        real_symlink(text, name, *args, **kwargs)
        if os.path.basename(os.fsdecode(name)) == tmp.name:
            _replace_folder(target.parent, moved)

    monkeypatch.setattr(os, "symlink", symlink)
    state = write_safety.restore_path(str(env.root), "src/link", source)

    assert state.startswith("skipped:") and "is no longer the folder" in state, state
    assert not os.path.lexists(target)
    assert (moved / "link").read_bytes() == RUNNER_CHANGE
    assert os.readlink(moved / tmp.name) == LINK_TEXT
    assert f"was left at {_moved_name(moved, tmp.name)}" in state


def test_restore_path_leaves_a_link_it_could_not_rename(env, tmp_path, monkeypatch, link_ways):
    """A rename the file system refuses (a directory put at the path once
    the restore looked, here as the link is made): the link made is left at
    its name and named, never removed by it."""
    source, target, tmp = _link_restore(env, tmp_path)
    real_symlink = os.symlink

    def symlink(text, name, *args, **kwargs):
        real_symlink(text, name, *args, **kwargs)
        if os.path.basename(os.fsdecode(name)) == tmp.name:
            target.unlink()
            target.mkdir()

    monkeypatch.setattr(os, "symlink", symlink)
    removals: list = []
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if os.path.basename(os.fsdecode(path)) == tmp.name:
            removals.append(path)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    state = write_safety.restore_path(str(env.root), "src/link", source)

    assert state.startswith("skipped:"), state
    assert removals == [] and os.readlink(tmp) == LINK_TEXT
    assert f"was left at {tmp}" in state and target.is_dir()


def test_restore_path_leaves_a_name_already_at_the_links(env, tmp_path, link_ways):
    """A name already at the staged one is never replaced or removed: the
    link is not made, and the name is left, named."""
    source, target, tmp = _link_restore(env, tmp_path)
    tmp.write_bytes(b"already here\n")

    state = write_safety.restore_path(str(env.root), "src/link", source)

    assert state.startswith("skipped:") and "was already there" in state, state
    assert tmp.read_bytes() == b"already here\n" and target.read_bytes() == RUNNER_CHANGE


# ---------------------------------------------------------------------------
# Windows staging over the emulated handle calls
# ---------------------------------------------------------------------------

def test_a_failed_rename_clears_the_read_only_attribute_before_deleting(env, monkeypatch):
    """Windows deletes no read-only file, through a handle neither. A
    read-only payload's staged file takes the attribute before its rename,
    so when the rename fails (a directory at the path now) the attribute is
    cleared through the handle, then the file is deleted through it: nothing
    is left beside the file and the error is the rename's."""
    calls = _staging_by_handle(monkeypatch)
    target = env.root / "src" / "a.py"
    target.chmod(stat.S_IREAD)
    try:
        snapshot, payload = _snapshot_of(env)
    finally:
        target.chmod(stat.S_IREAD | stat.S_IWRITE)
    target.unlink()
    target.mkdir()

    report = snapshot.restore()

    assert report["src/a.py"].startswith("skipped:"), report
    assert [name for name, _path in calls.calls] == [
        "create", "set_basic", "rename", "set_basic", "dispose"]
    assert not os.path.lexists(_restore_tmp(target))
    assert target.is_dir() and payload.exists()


def test_a_name_already_at_the_staged_files_is_never_opened_by_the_windows_calls(
    env, tmp_path, monkeypatch,
):
    """The staged file is created through the handle (``CREATE_NEW`` on the
    name itself): a name already there, a hard link to an outside file
    here, fails the creation and is left as it is, named; nothing is bound
    by name and the outside file keeps its bytes, mode and times."""
    calls = _staging_by_handle(monkeypatch)
    snapshot, target, payload, tmp = _rolled_back(env)
    victim, before = _outside(tmp_path)
    os.link(victim, tmp)

    report = snapshot.restore()

    state = _refused(report, target, payload, snapshot)
    assert "was already there" in state
    assert state.endswith("check what is left there and delete it by hand")
    assert [name for name, _path in calls.calls] == []
    assert os.path.samefile(tmp, victim) and _state(victim) == before


def test_without_the_windows_calls_nothing_is_put_back(env, monkeypatch):
    """Where the handle calls cannot be loaded, the restore puts nothing
    back rather than stage the file by its path, and says why."""
    _staging_by_handle(monkeypatch)
    monkeypatch.setattr(write_safety, "_held_copy_calls", lambda: None)
    snapshot, target, payload, tmp = _rolled_back(env)

    report = snapshot.restore()

    state = _refused(report, target, payload, snapshot)
    assert "could not be loaded, so nothing was put back" in state
    assert not os.path.lexists(tmp)


# ---------------------------------------------------------------------------
# Codex's U82 round 3 repros (u82-r3/test_review_adapted.py): each reproduced
# on the base restore for a read-only payload; none may now, for either.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload_mode", ["writable", "read-only"])
def test_codex_overwrite_before_the_flush_is_never_put_back(env, monkeypatch, payload_mode):
    """The file being put back rewritten (same inode) just before its flush:
    a writable payload goes back as ``CopyFile2``'s held copy (emulated), a
    read-only one is staged by the restore; either way it is read back
    after the flush, so nothing is put back."""
    _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    if payload_mode == "read-only":
        target.chmod(0o400)
    try:
        snapshot, payload = _snapshot_of(env)
    finally:
        target.chmod(0o600)
    target.write_bytes(RUNNER_CHANGE)
    tmp = _restore_tmp(target)

    def overwrite(handle):
        with open(_reach(tmp, handle), "r+b") as writer:
            writer.truncate(0)
            writer.write(b"not the hashed payload\n")

    ran = _at_the_flush(monkeypatch, overwrite, before=True)
    report = snapshot.restore()

    assert ran
    assert "does not hold the bytes" in _refused(report, target, payload, snapshot)


@pytest.mark.parametrize("payload_mode", ["writable", "read-only"])
def test_codex_hard_link_swap_never_changes_the_outside_file(
    env, tmp_path, monkeypatch, payload_mode,
):
    """The file being put back swapped for a hard link to an outside file:
    for a writable payload before ``CopyFile2``'s copy is bound (the outside
    file read-only, which the repro made writable), for a read-only one after
    its flush, where base's ``copystat`` made the outside file read-only and
    then installed the link as the repository's file, ``restored``. The
    outside file keeps its bytes, mode and times; nothing is put back."""
    calls = _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    if payload_mode == "read-only":
        target.chmod(0o400)
    try:
        snapshot, payload = _snapshot_of(env)
    finally:
        target.chmod(0o600)
    target.write_bytes(RUNNER_CHANGE)
    tmp = _restore_tmp(target)
    victim, before = _outside(tmp_path, 0o400 if payload_mode == "writable" else 0o600)

    def swap(_name_or_handle):
        _swap_for(tmp, lambda name: os.link(victim, name))

    if payload_mode == "writable":
        ran = _before_bind(monkeypatch, calls, swap)
    else:
        ran = _at_the_flush(monkeypatch, swap)
    try:
        report = snapshot.restore()
        after = _state(victim)
    finally:
        victim.chmod(0o600)

    assert ran
    _refused(report, target, payload, snapshot)
    assert not os.path.samefile(target, victim)
    assert after == before
    assert isinstance(calls, _HeldCopy)


# -- a staged file kept for the user (Codex review 2 of U87) ----------------

#: The restore's own token draw, before this module's autouse fixture pins it.
_DRAWN_TOKEN = write_safety._staging_token


def _failing_rename_once(monkeypatch, staged) -> list:
    """The first rename of a staged file whose name *staged* accepts fails,
    as a transient error would; the list holds the name it refused."""
    real = os.replace
    refused: list = []

    def replace(source, destination, *args, **kwargs):
        name = os.path.basename(os.fsdecode(source))
        if not refused and staged(name):
            refused.append(name)
            raise PermissionError(errno.EACCES, "transient rename failure")
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    return refused


def _kept(env, monkeypatch):
    """``src/a.py`` rolled back with its rename failing, so its staged file
    is kept for the user; the snapshot, the file, the kept name, and the
    restore's state."""
    snapshot, target, _payload, tmp = _rolled_back(env)
    _failing_rename_once(monkeypatch, lambda name: name == tmp.name)
    state = snapshot.restore()["src/a.py"]
    assert state.startswith("skipped:") and os.path.lexists(tmp), state
    return snapshot, target, tmp, state


def _batch(env, task_id, declared):
    """Another batch's dispatch with its containment sweep open: its
    scheduler and pre-batch snapshot."""
    task = env.apply_task(0, task_id=task_id)
    scheduler = env.scheduler([task])
    snapshot = scheduler._snapshots.take(task)
    scheduler._begin_sweep(task_id, declared)
    return scheduler, snapshot


def test_a_kept_staged_file_is_left_out_of_an_overlapping_batch(env, staging, monkeypatch):
    """Codex review 2 of U87, on both staging ways: the staged file a failed
    rename kept for the user while another batch's sweep was open is not
    that batch's write. Its sweep neither adopts nor attributes it, its
    rollback leaves it, and the restore says how to delete it."""
    scheduler, b_snapshot = _batch(env, "T-B", ["plans/p.md", "src/a.py"])

    _snapshot, _target, tmp, state = _kept(env, monkeypatch)

    relative = str(tmp.relative_to(env.root))
    observed = scheduler._undeclared_writes("T-B", b_snapshot)
    assert observed.observed
    assert relative not in observed.paths
    assert relative not in observed.restorable
    assert not any(relative in paths for paths in observed.shared.values())
    assert b_snapshot.entry_for(relative) is None
    assert b_snapshot.restore([relative]) == {
        relative: "skipped: not in this task's snapshot"}
    assert os.path.lexists(tmp)
    _kept_for_the_user(state)
    assert f"(rm -- {tmp})" in state
    assert write_safety.restore_leftovers(str(env.root)).holds(str(tmp))


def test_a_kept_staged_file_a_snapshot_adopted_is_never_deleted(env, staging, monkeypatch):
    """A rollback's delete leg leaves a kept staged file even when a snapshot
    holds it as a file its batch made (adopted before it was recorded, or
    by an older release), as does the commit write policy's delete."""
    _snapshot, _target, tmp, _state = _kept(env, monkeypatch)
    relative = str(tmp.relative_to(env.root))
    _scheduler, b_snapshot = _batch(env, "T-B", ["plans/p.md"])
    assert b_snapshot.adopt(relative, existed=False) is not None

    state = b_snapshot.restore([relative])[relative]

    assert state.startswith(f"skipped: {tmp} is a file a restore in this process "
                            f"left for you"), state
    assert f"(rm -- {tmp})" in state
    assert write_safety.restore_path(str(env.root), relative, None) == state
    assert os.path.lexists(tmp)


@pytest.mark.parametrize("made", ["a look-alike name", "the kept file's own path"])
def test_a_batch_file_named_like_a_kept_one_is_swept(env, monkeypatch, made):
    """Only the kept file itself is left out, by its path and identity, never
    by a name pattern: a file the batch makes with a staging-like name, or at
    the kept file's very path once that is gone, is swept, adopted and rolled
    back as any other, and the record of a kept file that is gone is
    dropped."""
    _snapshot, _target, tmp, _state = _kept(env, monkeypatch)
    kept = write_safety._identity_at(str(tmp))
    scheduler, b_snapshot = _batch(env, "T-B", ["plans/p.md"])
    if made == "a look-alike name":
        batch_file = tmp.with_name(f".a.py.restore.{os.getpid()}.feedface.tmp")
    else:
        os.unlink(tmp)
        batch_file = tmp
    batch_file.write_bytes(b"the batch's file\n")
    assert write_safety._identity_at(str(batch_file)) != kept
    relative = str(batch_file.relative_to(env.root))

    observed = scheduler._undeclared_writes("T-B", b_snapshot)

    assert relative in observed.restorable
    assert b_snapshot.restore([relative]) == {relative: "deleted"}
    assert not os.path.lexists(batch_file)
    leftovers = write_safety.restore_leftovers(str(env.root))
    if made == "a look-alike name":
        assert os.path.lexists(tmp) and leftovers.holds(str(tmp))
    else:
        assert not leftovers


def test_a_retry_after_a_kept_staged_file_puts_the_file_back(env, staging, monkeypatch):
    """Each attempt stages under a name of its own, so a staged file an
    earlier attempt kept never blocks the next: the retry restores, and the
    kept file stays, recorded, for the user."""
    monkeypatch.setattr(write_safety, "_staging_token", _DRAWN_TOKEN)
    snapshot, target, _payload, _tmp = _rolled_back(env)
    original = snapshot.entry_for("src/a.py").sha256
    refused = _failing_rename_once(
        monkeypatch, lambda name: name.startswith(f".a.py.restore.{os.getpid()}."))

    first = snapshot.restore()["src/a.py"]
    second = snapshot.restore()["src/a.py"]

    assert first.startswith("skipped:") and refused, first
    assert second == "restored"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == original
    kept = target.with_name(refused[0])
    assert os.path.lexists(kept)
    assert write_safety.restore_leftovers(str(env.root)).holds(str(kept))
    staged = sorted(p.name for p in target.parent.iterdir() if ".restore." in p.name)
    assert staged == [kept.name]


def test_a_kept_file_is_forgotten_once_it_is_gone(env, monkeypatch):
    """Once the user deletes the kept file, the next look at the set (a
    later restore's included) drops it, and nothing about it is on disk."""
    snapshot, _target, tmp, _state = _kept(env, monkeypatch)
    assert write_safety.restore_leftovers(str(env.root)).holds(str(tmp))
    os.unlink(tmp)

    monkeypatch.setattr(write_safety, "_staging_token", _DRAWN_TOKEN)
    assert snapshot.restore()["src/a.py"] == "restored"

    assert not write_safety.restore_leftovers(str(env.root))
    assert not write_safety._LEFTOVERS


def test_a_kept_link_is_recorded(env, tmp_path, monkeypatch):
    """``restore_path``'s link kept after a failed rename is recorded as the
    restore's own, as its staged file is."""
    source = tmp_path / "saved-link"
    source.symlink_to("elsewhere")
    target = env.root / "src" / "a.py"
    tmp = _restore_tmp(target)
    _failing_rename_once(monkeypatch, lambda name: name == tmp.name)

    state = write_safety.restore_path(str(env.root), "src/a.py", str(source))

    assert os.path.islink(tmp), state
    _kept_for_the_user(state)
    assert write_safety.restore_leftovers(str(env.root)).holds(str(tmp))


def test_the_porcelain_write_check_leaves_a_kept_file_out(env, monkeypatch):
    """The porcelain write check sees what the sweep sees: a kept staged file
    is in no snapshot, so the user deleting it while a slot runs is no
    write of the slot's; a look-alike name is a write."""
    from utils import review_orchestrator_base as base

    _snapshot, _target, tmp, _state = _kept(env, monkeypatch)
    before = base.porcelain_snapshot(str(env.root))
    os.unlink(tmp)
    look_alike = tmp.with_name(f".a.py.restore.{os.getpid()}.feedface.tmp")
    look_alike.write_bytes(b"a slot's file\n")
    after = base.porcelain_snapshot(str(env.root))

    relative = str(tmp.relative_to(env.root))
    assert before is not None and relative not in before.entries
    assert base.unexpected_writes(before, after) == [
        str(look_alike.relative_to(env.root))]


# -- only this process's own restores are trusted (fable review F1) ----------

#: The skill root, for a helper process to import ``utils`` from.
SKILL_ROOT = Path(write_safety.__file__).resolve().parents[1]

#: The uncontained rung's runner, as a separate process with every right the
#: user has: it edits a tracked file, makes an undeclared one, and records both
#: as a restore's leftovers wherever an earlier release kept that record (the
#: state home), plus a copy beside the phase dir, with their real identities.
_FORGING_RUNNER = textwrap.dedent("""
    import hashlib, json, os, sys
    root, phase, record_folder, edited, made = sys.argv[1:]
    with open(edited, "w") as handle:
        handle.write("print('backdoored')\\n")
    with open(made, "w") as handle:
        handle.write("import os\\n")
    left = []
    for path in (edited, made):
        info = os.lstat(path)
        left.append({"path": path, "dev": info.st_dev, "ino": info.st_ino,
                     "ctime_ns": info.st_ctime_ns})
    key = hashlib.sha256(os.path.normcase(root).encode()).hexdigest()[:32]
    for folder in (record_folder, phase):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, key + ".json"), "w") as handle:
            json.dump({"root": root, "left": left}, handle)
""")


def _forge(env, edited: Path, made: Path) -> None:
    """Run :data:`_FORGING_RUNNER` against the test repository."""
    root = os.path.realpath(env.root)
    record_folder = os.path.join(write_safety.state_home(), write_safety.STATE_SUBDIR,
                                 "restore-leftovers")
    subprocess.run([sys.executable, "-c", _FORGING_RUNNER, root, str(env.phase_dir),
                    record_folder, str(edited), str(made)],
                   check=True, timeout=60)


def test_a_record_a_runner_forges_exempts_nothing(env):
    """Fable review F1, on the uncontained rung: a runner with the user's
    rights writes a leftover record naming its own edit and its own new
    file, with their real identities. Nothing on disk is read as a
    leftover, so the porcelain write check reports both, the sweep's census
    counts the new file, and a rollback restores the edit and deletes the
    new file. Before, all four left them out."""
    from utils import review_orchestrator_base as base

    tracked = env.root / "src" / "a.py"
    made = env.root / "src" / "evil.py"
    snapshot = write_safety.TaskSnapshot.take(
        phase_dir=str(env.phase_dir), task_id="T-F", root=str(env.root),
        files=["src/a.py", "src/evil.py"])
    original = tracked.read_bytes()
    scheduler, b_snapshot = _batch(env, "T-B", ["plans/p.md"])
    before = base.porcelain_snapshot(str(env.root))

    _forge(env, tracked, made)

    after = base.porcelain_snapshot(str(env.root))
    assert {"src/a.py", "src/evil.py"} <= set(base.unexpected_writes(before, after))
    assert not write_safety.restore_leftovers(str(env.root))
    observed = scheduler._undeclared_writes("T-B", b_snapshot)
    assert {"src/a.py", "src/evil.py"} <= set(observed.paths)
    assert snapshot.restore() == {"src/a.py": "restored", "src/evil.py": "deleted"}
    assert tracked.read_bytes() == original and not made.exists()


def test_a_file_another_process_kept_is_checked_as_any_other(env):
    """The price of trusting this process alone: a staged file a restore in
    another process kept (an earlier ``util restore-batch``) is an ordinary
    file here, censused, and a rollback holding it deletes it."""
    script = textwrap.dedent("""
        import errno, os, sys
        skill_root, phase, root = sys.argv[1:]
        sys.path.insert(0, skill_root)
        from utils import write_safety as ws
        snapshot = ws.TaskSnapshot.take(phase_dir=phase, task_id="T-O", root=root,
                                        files=["src/a.py"])
        with open(os.path.join(root, "src", "a.py"), "w") as handle:
            handle.write("changed\\n")
        real = os.replace
        def replace(source, destination, *args, **kwargs):
            if ".restore." in os.fsdecode(source):
                raise PermissionError(errno.EACCES, "transient rename failure")
            return real(source, destination, *args, **kwargs)
        os.replace = replace
        state = snapshot.restore()["src/a.py"]
        assert state.startswith("skipped:"), state
        print(next(iter(ws.restore_leftovers(root)._entries)))
    """)
    shown = subprocess.run(
        [sys.executable, "-c", script, str(SKILL_ROOT), str(env.phase_dir),
         os.path.realpath(env.root)],
        check=True, timeout=60, capture_output=True, text=True).stdout.strip()
    kept = Path(shown)
    assert kept.name.startswith(".a.py.restore.") and os.path.lexists(kept)
    relative = str(kept.relative_to(os.path.realpath(env.root)))

    assert not write_safety.restore_leftovers(str(env.root))
    _scheduler, b_snapshot = _batch(env, "T-B", ["plans/p.md"])
    assert b_snapshot.adopt(relative, existed=False) is not None
    assert b_snapshot.restore([relative]) == {relative: "deleted"}
    assert not os.path.lexists(kept)


def test_a_kept_file_a_first_census_saw_is_no_batch_write(env, staging, monkeypatch):
    """Codex review 3 of U87, on both staging ways: batch B's first census
    sees the staged file in the instant before the failed rename's record;
    its final census finds it recorded. The difference leaves it out on both
    sides, so B neither reads it as deleted nor adopts it (before, with
    ``existed=True``, whose rollback replaced the file, changed its identity
    and dropped its record, after which batch C's sweep adopted it and C's
    rollback deleted it). C's sweep leaves it out as well."""
    snapshot, _target, _payload, tmp = _rolled_back(env)
    relative = str(tmp.relative_to(env.root))
    real = os.replace
    opened: dict = {}

    def replace(source, destination, *args, **kwargs):
        if os.path.basename(os.fsdecode(source)) == tmp.name and not opened:
            opened["B"] = _batch(env, "T-B", ["plans/p.md", "src/a.py"])
            raise PermissionError(errno.EACCES, "transient rename failure")
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    state = snapshot.restore()["src/a.py"]
    monkeypatch.setattr(os, "replace", real)
    _kept_for_the_user(state)
    scheduler, b_snapshot = opened["B"]
    assert relative in scheduler._sweep._census_before["T-B"]  # seen before its record
    c_scheduler, c_snapshot = _batch(env, "T-C", ["plans/p.md", "src/a.py"])

    assert relative not in scheduler._sweep.moved_since_dispatch("T-B", b_snapshot)
    observed = scheduler._undeclared_writes("T-B", b_snapshot)
    assert observed.observed
    assert relative not in observed.paths and relative not in observed.restorable
    assert not any(relative in paths for paths in observed.shared.values())
    assert b_snapshot.entry_for(relative) is None
    assert b_snapshot.restore([relative]) == {
        relative: "skipped: not in this task's snapshot"}

    c_observed = c_scheduler._undeclared_writes("T-C", c_snapshot)
    assert relative not in c_observed.paths and c_snapshot.entry_for(relative) is None
    assert os.path.lexists(tmp)
    assert write_safety.restore_leftovers(str(env.root)).holds(str(tmp))


@pytest.mark.parametrize("forced", [False, True])
def test_the_replace_leg_refuses_a_kept_file_a_snapshot_adopted(env, monkeypatch, forced):
    """A rollback's replace leg refuses a kept staged file as its delete leg
    does, even for a snapshot that adopted it as a file its batch changed
    (``existed=True``, with retained bytes) and with --force: the file keeps
    its bytes and identity and stays recorded, and ``restore_path`` with a
    source refuses it too."""
    _snapshot, _target, tmp, _state = _kept(env, monkeypatch)
    relative = str(tmp.relative_to(env.root))
    identity = write_safety._identity_at(str(tmp))
    held = tmp.read_bytes()
    retained = env.tmp_path / "retained"
    retained.write_bytes(b"bytes a batch's sweep retained\n")
    digest = hashlib.sha256(retained.read_bytes()).hexdigest()
    _scheduler, b_snapshot = _batch(env, "T-B", ["plans/p.md"])
    assert b_snapshot.adopt(relative, existed=True, sha256=digest,
                            size=retained.stat().st_size,
                            payload_source=str(retained)) is not None

    overrides = {"expected": {relative: None}, "force": True} if forced else {}
    state = b_snapshot.restore([relative], **overrides)[relative]

    assert state.startswith(f"skipped: {tmp} is a file a restore in this process "
                            f"left for you to check, so no rollback in it replaces "
                            f"or deletes it"), state
    assert f"(rm -- {tmp})" in state
    assert write_safety.restore_path(str(env.root), relative, str(retained)) == state
    assert tmp.read_bytes() == held
    assert write_safety._identity_at(str(tmp)) == identity
    assert write_safety.restore_leftovers(str(env.root)).holds(str(tmp))


# -- a directory at the path stages nothing (fable review F4) ---------------

def test_a_directory_at_the_path_stages_nothing(env, staging):
    """Fable review F4, on both staging ways: a directory at the file's
    path before the restore is refused before anything is staged, so no
    attempt leaves a copy of the pre-batch bytes beside it; the directory
    and the snapshot are kept. Before, each attempt left one staged copy."""
    snapshot, target, payload, _tmp = _rolled_back(env)
    target.unlink()
    target.mkdir()
    (target / "inside.txt").write_bytes(b"kept\n")

    states = [snapshot.restore()["src/a.py"] for _ in range(2)]

    assert states == [f"skipped: {target} is a directory now, so nothing was put "
                      f"back (the snapshot is kept)"] * 2
    assert [p.name for p in target.parent.iterdir() if ".restore." in p.name] == []
    assert (target / "inside.txt").read_bytes() == b"kept\n" and payload.exists()
    assert not write_safety.restore_leftovers(str(env.root))


def test_restore_path_puts_no_link_beside_a_directory(env, tmp_path, link_ways):
    """``restore_path``'s link leg refuses a directory at the path before it
    makes the link, so nothing is left beside it."""
    source, target, tmp = _link_restore(env, tmp_path)
    target.unlink()
    target.mkdir()

    state = write_safety.restore_path(str(env.root), "src/link", source)

    assert state == (f"skipped: {target} is a directory now, so nothing was put "
                     f"back (the snapshot is kept)")
    assert not os.path.lexists(tmp) and target.is_dir()


# -- the delete leg removes relative to its folder held (fable review F3) ----

def _made_by_the_batch(env, tmp_path):
    """``src/new/made.txt``, absent when snapshotted and made by the batch,
    and an outside folder holding the user's own ``made.txt``; the snapshot,
    the outside file and its bytes."""
    (env.root / "src" / "new").mkdir()
    snapshot = write_safety.TaskSnapshot.take(
        phase_dir=str(env.phase_dir), task_id="T-D", root=str(env.root),
        files=["src/new/made.txt"])
    (env.root / "src" / "new" / "made.txt").write_bytes(b"made by the batch\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "made.txt"
    victim.write_bytes(b"the user's own file\n")
    return snapshot, victim


def _swap_new_for_a_link(env, victim: Path) -> None:
    folder = env.root / "src" / "new"
    folder.rename(folder.with_name("new.moved"))
    os.symlink(victim.parent, folder)


@pytest.mark.parametrize("leg", ["snapshot", "restore_path"])
@pytest.mark.parametrize("when", ["before the hold", "once held"])
def test_the_delete_leg_never_removes_through_a_folder_swapped_in(
        env, tmp_path, monkeypatch, leg, when):
    """Fable review F3: between the delete leg's revalidation and its
    removal, the file's folder is swapped for a link to an outside folder
    holding a file of the same name (a provider tree still running). The
    removal is made relative to the folder reached without following a link
    and still at its path, so it is refused, and the outside file stays.
    Before, the name was unlinked by path and the outside file was deleted."""
    snapshot, victim = _made_by_the_batch(env, tmp_path)
    if when == "before the hold":
        real = write_safety._revalidated_target

        def racing(*args, **kwargs):
            target = real(*args, **kwargs)
            _swap_new_for_a_link(env, victim)
            return target

        monkeypatch.setattr(write_safety, "_revalidated_target", racing)
    else:
        real_lstat = os.lstat

        def lstat(path, *args, **kwargs):
            if kwargs.get("dir_fd") is not None and os.fsdecode(path) == "made.txt" \
                    and not os.path.islink(env.root / "src" / "new"):
                _swap_new_for_a_link(env, victim)
            return real_lstat(path, *args, **kwargs)

        monkeypatch.setattr(os, "lstat", lstat)

    if leg == "snapshot":
        state = snapshot.restore()["src/new/made.txt"]
    else:
        state = write_safety.restore_path(str(env.root), "src/new/made.txt", None)

    assert state.startswith("skipped:"), state
    assert victim.read_bytes() == b"the user's own file\n"
    assert (env.root / "src" / "new.moved" / "made.txt").exists()


@pytest.mark.parametrize("leg", ["snapshot", "restore_path"])
@pytest.mark.parametrize("change", ["removed", "replaced by a file", "unreadable"])
def test_a_folder_changed_between_the_checks(env, tmp_path, monkeypatch, leg, change):
    """Codex review of F01, point 1: the file's folder removed by another
    process between the delete leg's revalidation and its hold leaves
    nothing to delete, so the leg says ``deleted (already absent)``, as it
    did before the hold. A folder replaced by a file, or one that cannot be
    opened, is not absence: still refused, and nothing is removed."""
    if change == "unreadable" and hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root opens a folder whatever its mode")
    snapshot, _victim = _made_by_the_batch(env, tmp_path)
    folder = env.root / "src" / "new"
    real = write_safety._revalidated_target

    def racing(*args, **kwargs):
        target = real(*args, **kwargs)
        if change == "unreadable":
            folder.chmod(0)
        else:
            shutil.rmtree(folder)
            if change == "replaced by a file":
                folder.write_bytes(b"another process's file\n")
        return target

    monkeypatch.setattr(write_safety, "_revalidated_target", racing)
    try:
        if leg == "snapshot":
            state = snapshot.restore()["src/new/made.txt"]
        else:
            state = write_safety.restore_path(str(env.root), "src/new/made.txt", None)
    finally:
        if folder.is_dir():
            folder.chmod(0o755)

    if change == "removed":
        assert state == "deleted (already absent)"
    else:
        assert state.startswith("skipped:") and "could not be reached" in state, state
    if change == "replaced by a file":
        assert folder.read_bytes() == b"another process's file\n"
    if change == "unreadable":
        assert (folder / "made.txt").exists()


def test_the_delete_leg_still_deletes_and_leaves_a_directory(env, tmp_path):
    """Without a swap the delete leg removes the batch's file, a second
    pass finds it gone, and a directory at the path is left on either leg."""
    snapshot, _victim = _made_by_the_batch(env, tmp_path)
    made = env.root / "src" / "new" / "made.txt"

    assert snapshot.restore() == {"src/new/made.txt": "deleted"}
    assert not made.exists()
    assert snapshot.restore() == {"src/new/made.txt": "deleted (already absent)"}
    made.mkdir()
    assert snapshot.restore() == {
        "src/new/made.txt": "skipped: a directory is at the path now"}
    assert write_safety.restore_path(str(env.root), "src/new/made.txt", None) == (
        "skipped: a directory is at the path now")
    assert made.is_dir()
