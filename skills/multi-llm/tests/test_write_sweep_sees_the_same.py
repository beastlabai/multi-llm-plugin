"""The containment sweep's verdicts, pinned on one planted tree.

The sweep (``_begin_sweep`` -> runner -> ``_undeclared_writes``) and rule 9's
"said applied, changed nothing" measurement (``_applied_without_changes``) are
the detective half of write safety (docs/development/write-safety.md). Their
census, retention and audit reads are tuned for speed, so this file plants one
tree that exercises every rule they decide on and asserts the exact answer:

* undeclared edits, creations, deletions and a moved file are all observed;
  the ones that existed before the run are restorable from retained bytes;
* a path ``driver.write_paths`` denies, a symlink and a symlinked directory
  never enter the rollback store, and the deny-listed write is still observed;
* the executor's own folders are left out, a sibling folder whose name only
  shares their prefix is not;
* an ``applied`` reply over an unchanged tree is named, a declared change is
  not, and a stray write beside an unchanged plan is named with what moved;
* the retention pass reads each file once, through the descriptor the policy
  was asked about, on both ways a file is opened (by descriptor walk, by path
  as on Windows): a link swapped in mid-pass is never followed and a blob
  always holds the bytes its name is the digest of;
* a restore puts a payload that is not read-only back as ``CopyFile2``'s
  copy on Windows (emulated elsewhere), checked and swapped in through one
  handle held on it alone, keeping its streams, attributes and encryption
  and refusing a copy of anything else; every other restore stages the
  payload's bytes itself, through one descriptor or handle from creation to
  rename (the adversarial cases: tests/test_restore_staged_file.py); and the
  store's discard names what it cannot remove, never trying it again.

No model runs; the scheduler is the real one over a real git repository.
"""

from __future__ import annotations

import errno
import hashlib
import os
import pathlib
import shutil
import stat
import subprocess
import threading
import types
from pathlib import Path

import pytest

from tests.test_write_executor import env  # noqa: F401
from utils import write_batch_scheduler as wbs
from utils import write_safety


def _plant(env, tmp_path: Path) -> None:
    """The tree the sweep walks, beyond `WriteEnv`'s plan and `src/`."""
    (env.root / "src" / "c.py").write_text("c = 1\n", encoding="utf-8")
    (env.root / "src" / "deep" / "er").mkdir(parents=True)
    (env.root / "src" / "deep" / "er" / "d.py").write_text("d = 1\n", encoding="utf-8")
    (env.root / ".env").write_text("NAME=value-before\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("outside the repository\n", encoding="utf-8")
    (env.root / "link.txt").symlink_to(outside / "private.txt")
    (env.root / "linked-dir").symlink_to(outside, target_is_directory=True)
    # Shares the phase folder's name as a prefix: walked like any folder.
    sibling = env.out_dir / (env.phase_dir.name + "-old")
    sibling.mkdir(parents=True)
    (sibling / "notes.md").write_text("old\n", encoding="utf-8")


def _store_bytes(scheduler) -> set:
    store = Path(scheduler.safety_dir) / wbs.ROLLBACK_STORE_DIRNAME
    if not store.is_dir():
        return set()
    return {path.read_bytes() for path in store.rglob("*") if path.is_file()}


def test_the_census_is_the_tree_without_links_vcs_or_the_executors_own_files(env, tmp_path):
    _plant(env, tmp_path)
    scheduler = env.scheduler([env.apply_task(0)])
    (env.phase_dir / "result.json").write_text("{}", encoding="utf-8")
    (env.out_dir / "prior_changes.jsonl").write_text("", encoding="utf-8")
    runs = env.root / ".multi-llm" / "runs"
    runs.mkdir(parents=True)
    (runs / "r.json").write_text("{}", encoding="utf-8")

    census = scheduler._sweep.census()

    expected = {}
    for folder, dirs, files in os.walk(env.root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            path = Path(folder) / name
            if path.is_symlink():
                continue
            relative = path.relative_to(env.root).as_posix()
            if relative.startswith(".multi-llm/p/apply-suggestions/") or relative in (
                    ".multi-llm/p/prior_changes.jsonl", ".multi-llm/runs/r.json"):
                continue
            info = path.stat()
            expected[relative] = (info.st_size, info.st_mtime_ns)
    assert census == expected
    assert ".multi-llm/p/apply-suggestions-old/notes.md" in census
    assert "link.txt" not in census
    assert not any(key.startswith("linked-dir/") for key in census)


def test_undeclared_writes_are_observed_and_the_existing_ones_restorable(env, tmp_path):
    _plant(env, tmp_path)
    task = env.apply_task(0, task_id="T-A")
    scheduler = env.scheduler([task])
    snapshot = scheduler._snapshots.take(task)
    before = {
        "src/a.py": (env.root / "src" / "a.py").read_bytes(),
        "src/b.py": (env.root / "src" / "b.py").read_bytes(),
        "src/c.py": (env.root / "src" / "c.py").read_bytes(),
    }
    scheduler._begin_sweep("T-A", ["plans/p.md"])

    # Nothing denied or linked was copied, whatever its bytes.
    store = _store_bytes(scheduler)
    assert b"NAME=value-before\n" not in store
    assert b"outside the repository\n" not in store
    assert set(before.values()) <= store

    # The runner: its declared plan, and everything else it should not touch.
    env.plan.write_text("# Plan\n\n## Step 1\nedited\n", encoding="utf-8")
    (env.root / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
    (env.root / "src" / "c.py").unlink()
    (env.root / "src" / "new.py").write_text("new = 1\n", encoding="utf-8")
    (env.root / "src" / "moved").mkdir()
    os.replace(env.root / "src" / "b.py", env.root / "src" / "moved" / "b.py")
    (env.root / ".env").write_text("NAME=value-after\n", encoding="utf-8")
    (env.phase_dir / "scratch.json").write_text("{}", encoding="utf-8")
    old = env.out_dir / (env.phase_dir.name + "-old") / "notes.md"
    old.write_text("rewritten\n", encoding="utf-8")

    undeclared = scheduler._undeclared_writes("T-A", snapshot)

    assert undeclared.observed
    assert sorted(undeclared.paths) == [
        ".env", ".multi-llm/p/apply-suggestions-old/notes.md",
        "src/a.py", "src/b.py", "src/c.py", "src/moved/b.py", "src/new.py",
    ]
    # Restorable: retained bytes for what existed, recorded absence for what
    # the runner created. Denied paths (`.env`, `.multi-llm/**`) have
    # neither: observed and policy-checked, never copied.
    assert sorted(undeclared.restorable) == [
        "src/a.py", "src/b.py", "src/c.py", "src/moved/b.py", "src/new.py",
    ]
    assert sorted(undeclared.unrecoverable) == [
        ".env", ".multi-llm/p/apply-suggestions-old/notes.md"]
    with pytest.raises(write_safety.PolicyViolation):
        scheduler.policy.check(undeclared.paths, when="post-run")

    snapshot.restore(undeclared.restorable)
    for relative, data in before.items():
        assert (env.root / relative).read_bytes() == data, relative
    assert not (env.root / "src" / "new.py").exists()
    assert not (env.root / "src" / "moved" / "b.py").exists()


def _applied(scheduler, task, snapshot, **result):
    context = {
        "result": {"status": "applied", "ids": list(task["expected_ids"]),
                   "files_touched": [], "summary": "done", **result},
        "exit_code": 0,
    }
    return scheduler._applied_without_changes(task, snapshot, context), context


def test_an_applied_reply_is_named_only_when_its_files_did_not_move(env, tmp_path):
    _plant(env, tmp_path)
    tasks = [env.apply_task(index, task_id=f"T-{index}") for index in range(3)]
    scheduler = env.scheduler(tasks)
    # Each snapshot is taken just before its own dispatch, as `_dispatch_one`
    # takes it (an earlier one of the same batch would be reused).
    snapshots = {"T-0": scheduler._snapshots.take(tasks[0])}

    # Declared change: applied, nothing to name.
    scheduler._begin_sweep("T-0", ["plans/p.md"])
    env.plan.write_text("# Plan\n\n## Step 1\nedited\n", encoding="utf-8")
    why, context = _applied(scheduler, tasks[0], snapshots["T-0"])
    assert why is None and "moved_outside" not in context
    scheduler._undeclared_writes("T-0", snapshots["T-0"])

    # Nothing moved at all: named.
    snapshots["T-1"] = scheduler._snapshots.take(tasks[1])
    scheduler._begin_sweep("T-1", ["plans/p.md"])
    why, context = _applied(scheduler, tasks[1], snapshots["T-1"],
                            files_touched=["plans/p.md"])
    assert why is not None and "no file changed" in why
    assert "named plans/p.md as edited" in why
    scheduler._undeclared_writes("T-1", snapshots["T-1"])

    # A plan batch whose plan did not move but whose runner wrote elsewhere:
    # named, with the stray paths it moved, deletions included.
    snapshots["T-2"] = scheduler._snapshots.take(tasks[2])
    scheduler._begin_sweep("T-2", ["plans/p.md"])
    (env.root / "src" / "a.py").write_text("a = 3\n", encoding="utf-8")
    (env.root / "src" / "deep" / "er" / "d.py").unlink()
    why, context = _applied(scheduler, tasks[2], snapshots["T-2"])
    assert why is not None and "outside it" in why
    assert context["moved_outside"] == ["src/a.py", "src/deep/er/d.py"]
    # Read-only: the census is still registered for the classification.
    assert sorted(scheduler._undeclared_writes("T-2", snapshots["T-2"]).paths) == [
        "src/a.py", "src/deep/er/d.py"]


def test_a_code_task_with_a_stray_write_is_not_named(env, tmp_path):
    """For a code task the bytes decide: a write outside the declared files
    is a change, so the reply is not "applied over nothing"."""
    _plant(env, tmp_path)
    task = env.apply_task(
        0, kind="implement", task_id="T009", expected_ids=["T009"],
        declared_files=[str(env.root / "src" / "a.py")],
        prompt=f"Do T009\n\n## Changes Applied in Prior Batches\n\n{wbs.PRIOR_CHANGES_TOKEN}\n",
    )
    scheduler = env.scheduler([task], kind="implement", finalize=False)
    snapshot = scheduler._snapshots.take(task)
    scheduler._begin_sweep("T009", ["src/a.py"])
    (env.root / "src" / "c.py").write_text("c = 2\n", encoding="utf-8")
    why, _context = _applied(scheduler, task, snapshot)
    assert why is None
    assert scheduler._undeclared_writes("T009", snapshot).paths == ["src/c.py"]


def test_a_folder_swapped_for_a_link_after_the_census_is_not_copied_through(env, tmp_path):
    """Between the census and the retention pass a sibling batch can replace a
    folder with a link to one outside the repository: the pass must judge the
    path where it now leads, never copy those bytes into the store."""
    _plant(env, tmp_path)
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    assert "src/deep/er/d.py" in census
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "d.py").write_text("bytes from outside the repository\n", encoding="utf-8")
    (env.root / "src" / "deep" / "er" / "d.py").unlink()
    (env.root / "src" / "deep" / "er").rmdir()
    (env.root / "src" / "deep" / "er").symlink_to(elsewhere, target_is_directory=True)

    kept = scheduler._sweep.retain(census)

    assert "src/deep/er/d.py" not in kept
    assert b"bytes from outside the repository\n" not in _store_bytes(scheduler)
    assert "src/a.py" in kept


def _after_first_resolve(monkeypatch, path, then) -> list:
    """Run *then* right after *path* is first resolved (``realpath``); the
    returned list is non-empty once it ran."""
    real = os.path.realpath
    wanted = os.path.normcase(os.path.normpath(real(path)))
    fired: list = []

    def resolve(target, *args, **kwargs):
        resolved = real(target, *args, **kwargs)
        if not fired and os.path.normcase(os.path.normpath(os.fspath(target))) == wanted:
            fired.append(True)
            then()
        return resolved

    monkeypatch.setattr(os.path, "realpath", resolve)
    # Python 3.10's `Path.resolve` reaches `realpath` through its accessor.
    accessor = getattr(pathlib, "_NormalAccessor", None)
    if accessor is not None:
        monkeypatch.setattr(accessor, "realpath", staticmethod(resolve))
    return fired


def test_a_file_swapped_for_a_link_inside_the_retention_pass_is_not_copied(
    env, tmp_path, monkeypatch,
):
    """Inside the pass, a sibling batch can replace a file with a link to one
    outside the repository right after the pass first resolved it. The
    policy check resolves the path again for its allow side, so under an
    allow list that does not admit the link's target those bytes never
    enter the store (docs/development/write-safety.md)."""
    _plant(env, tmp_path)
    env.use(write_paths={"allow": ["src/**"], "deny": ["**/.env"]})
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    outside = tmp_path / "outside" / "secret.txt"
    outside.write_text("bytes the deny list keeps out\n", encoding="utf-8")
    target = env.root / "src" / "b.py"

    def swap() -> None:
        target.unlink()
        target.symlink_to(outside)

    fired = _after_first_resolve(monkeypatch, target, swap)
    kept = scheduler._sweep.retain(census)

    assert fired
    assert "src/b.py" not in kept
    assert b"bytes the deny list keeps out\n" not in _store_bytes(scheduler)
    assert "src/a.py" in kept


@pytest.mark.parametrize("undone", [False, True], ids=["persistent", "temporary"])
def test_a_phase_path_swapped_inside_the_retention_pass_stops_it_at_the_next_file(
    env, tmp_path, monkeypatch, undone,
):
    """With the rollback evidence outside the repository, the phase path is
    what names it, and a runner can replace that path with a link while the
    pass copies. Each copy re-validates the store's identity
    (`write_safety.safety_dir`), so the pass stops before the next file's
    bytes are copied, even when the swap is undone right after it."""
    scheduler = env.scheduler([env.apply_task(0)])
    write_safety.claim_external_safety_dir(str(env.phase_dir), str(env.root))
    census = scheduler._sweep.census()
    saved = env.phase_dir.with_name("saved-phase")
    forged = tmp_path / "forged"
    forged.mkdir()
    store = scheduler._sweep.store_blob
    copied = []

    def store_then_swap(relative, stamp):
        result = store(relative, stamp)
        name = Path(relative).name
        copied.append(name)
        if name == "a.py":
            env.phase_dir.rename(saved)
            env.phase_dir.symlink_to(forged, target_is_directory=True)
        elif name == "b.py" and undone:
            env.phase_dir.unlink()
            saved.rename(env.phase_dir)
        return result

    monkeypatch.setattr(scheduler._sweep, "store_blob", store_then_swap)
    with pytest.raises(write_safety.WriteSafetyError, match="trusted phase identity"):
        scheduler._sweep.retain(census)
    assert copied == ["p.md", "a.py"]
    assert not any(forged.iterdir())


def _copy_file2_off_windows(source, destination, flags):
    """``CopyFile2`` as Windows runs it, for the platforms without it: a link
    at the source's own name copied as a link (``COPY_FILE_COPY_SYMLINK``),
    any other source read by path through any link above it, the destination
    created new, failing on any name there (``COPY_FILE_FAIL_IF_EXISTS``)
    but a link to nothing, which is followed and its target created with
    the copy, as the Windows CI showed (2026-11-12) though ``CopyFileEx``
    documents that ``COPY_FILE_COPY_SYMLINK`` makes it fail on any link
    there, then ``copy2``'s metadata, taken after the read."""
    assert flags & 0x809 == 0x809, (
        "shutil.copy2's flag, an exclusive creation, and a link copied as one")
    if os.path.islink(source):
        os.symlink(os.readlink(source), destination)
        return
    if os.path.islink(destination) and not os.path.exists(destination):
        destination = os.path.realpath(destination)
    handle = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with open(source, "rb") as reader, os.fdopen(handle, "wb") as writer:
        shutil.copyfileobj(reader, writer)
    shutil.copystat(source, destination)


@pytest.fixture(params=["descriptor walk", "by path", "by path, plain copy"])
def opening(request, monkeypatch):
    """Each way the retention pass opens and copies a file: one component at
    a time by descriptor (Linux, macOS); by path with the identity check and
    copied by ``CopyFile2`` (Windows on Python 3.12+, emulated elsewhere);
    by path and copied from the bytes read (Windows before 3.12)."""
    if request.param == "descriptor walk":
        if not getattr(write_safety, "OPENS_WITHOUT_FOLLOWING", True):
            pytest.skip("this platform opens by path only")
        return request.param
    monkeypatch.setattr(write_safety, "OPENS_WITHOUT_FOLLOWING", False, raising=False)
    if request.param == "by path, plain copy":
        monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
    elif getattr(write_safety, "_COPY_FILE2", None) is None:
        monkeypatch.setattr(write_safety, "_COPY_FILE2", _copy_file2_off_windows,
                            raising=False)
    return request.param


def _copies_by_path(opening) -> bool:
    return opening == "by path"


def _by_path_with_copy_file2(monkeypatch) -> None:
    """Windows on Python 3.12+: opened by path, copied by ``CopyFile2``."""
    monkeypatch.setattr(write_safety, "OPENS_WITHOUT_FOLLOWING", False, raising=False)
    if getattr(write_safety, "_COPY_FILE2", None) is None:
        monkeypatch.setattr(write_safety, "_COPY_FILE2", _copy_file2_off_windows,
                            raising=False)


def _blobs(scheduler) -> list:
    store = Path(scheduler.safety_dir) / wbs.ROLLBACK_STORE_DIRNAME
    return [path for path in store.rglob("*") if path.is_file()] if store.is_dir() else []


@pytest.mark.parametrize("swapped", ["the file", "a folder above it"])
def test_a_link_swapped_in_after_the_first_resolve_is_not_followed_under_the_default_allow(
    env, tmp_path, monkeypatch, opening, swapped,
):
    """Under the default ``allow: ["**"]`` the resolving policy check asks
    about the root on its first resolution only, and its allow side matches
    any path: a file, or a folder above it, swapped for a link to one outside
    the repository right after that resolution used to be read through the
    link into the store. The policy is asked again about the file actually
    opened, so those bytes never enter it (docs/development/write-safety.md)."""
    _plant(env, tmp_path)
    scheduler = env.scheduler([env.apply_task(0)])
    assert scheduler.policy.allow == ["**"]
    census = scheduler._sweep.census()
    secret = b"bytes from outside, linked in mid-pass\n"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if swapped == "the file":
        relative = "src/b.py"
        (elsewhere / "secret.txt").write_bytes(secret)
        target = env.root / relative

        def swap() -> None:
            target.unlink()
            target.symlink_to(elsewhere / "secret.txt")
    else:
        relative = "src/deep/er/d.py"
        (elsewhere / "d.py").write_bytes(secret)
        target = env.root / relative
        folder = target.parent

        def swap() -> None:
            target.unlink()
            folder.rmdir()
            folder.symlink_to(elsewhere, target_is_directory=True)

    fired = _after_first_resolve(monkeypatch, target, swap)
    kept = scheduler._sweep.retain(census)

    assert fired
    assert relative not in kept
    assert secret not in _store_bytes(scheduler)
    assert "src/a.py" in kept


@pytest.mark.parametrize("change", ["rewritten in place", "replaced by a link"])
def test_a_file_changed_between_its_hash_and_its_copy_stores_the_bytes_hashed(
    env, tmp_path, monkeypatch, opening, change,
):
    """The pass used to read a file twice, by path: once to hash it, once to
    copy it. A change in between stored bytes under the digest of others,
    and a link swapped in there copied an outside file's bytes. Read once,
    every blob holds exactly the bytes its name is the digest of: the ones
    the file held when it was read (copied by path, a copy that read other
    bytes is refused). A leftover pending copy would fail that check too."""
    if change == "replaced by a link" and os.name == "nt":
        pytest.skip("Windows refuses to remove a file held open, so it cannot be "
                    "replaced while it is read")
    _plant(env, tmp_path)
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    target = env.root / "src" / "a.py"
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    secret = tmp_path / "outside" / "between.txt"
    secret.write_bytes(b"bytes linked in between the hash and the copy\n")
    blob_path = scheduler._sweep.blob_path
    fired = []

    def name_the_blob(name: str) -> str:
        # The seam between the digest and the copy, in both shapes of the pass.
        if name == digest and not fired:
            fired.append(True)
            if change == "rewritten in place":
                with open(target, "r+b") as stream:
                    stream.write(b"z = 9\n")
            else:
                target.unlink()
                target.symlink_to(secret)
        return blob_path(name)

    monkeypatch.setattr(scheduler._sweep, "blob_path", name_the_blob)
    kept = scheduler._sweep.retain(census)

    assert fired
    if _copies_by_path(opening):
        # Copied by path, the copy read the new bytes: checked against the
        # bytes hashed, it is refused and removed.
        assert "src/a.py" not in kept
    else:
        assert kept["src/a.py"] == digest
    blobs = _blobs(scheduler)
    assert blobs
    for blob in blobs:
        assert hashlib.sha256(blob.read_bytes()).hexdigest() == blob.name, blob
    assert secret.read_bytes() not in _store_bytes(scheduler)


def test_by_path_a_link_there_only_for_the_open_is_refused(env, tmp_path, monkeypatch):
    """Opened by path (Windows), a link at the file's own name that is there
    for the open and gone again for the resolving re-check is still refused:
    the path's own entry is then not the file the descriptor reads."""
    monkeypatch.setattr(write_safety, "OPENS_WITHOUT_FOLLOWING", False, raising=False)
    _plant(env, tmp_path)
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    target = env.root / "src" / "b.py"
    saved = target.with_name("b.saved")
    secret = tmp_path / "outside" / "for-the-open.txt"
    secret.write_bytes(b"bytes linked in for the open only\n")
    wanted = os.path.normcase(os.path.realpath(target))
    real_open = os.open
    fired = []

    def open_through_a_link(path, flags, *args, **kwargs):
        if fired or os.path.normcase(os.fspath(path)) != wanted:
            return real_open(path, flags, *args, **kwargs)
        fired.append(True)
        os.rename(target, saved)
        os.symlink(secret, target)
        try:
            return real_open(path, flags, *args, **kwargs)
        finally:
            os.unlink(target)
            os.rename(saved, target)

    monkeypatch.setattr(os, "open", open_through_a_link)
    kept = scheduler._sweep.retain(census)

    assert fired
    assert "src/b.py" not in kept
    assert secret.read_bytes() not in _store_bytes(scheduler)
    assert "src/a.py" in kept


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs here")
def test_a_fifo_swapped_in_after_the_census_is_refused_without_waiting(
    env, tmp_path, opening,
):
    """A file replaced by a FIFO after the census is opened without blocking
    and refused as not a regular file: opened by path for reading, the pass
    used to wait for a writer that never comes."""
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    target = env.root / "src" / "b.py"
    target.unlink()
    os.mkfifo(target)

    kept = scheduler._sweep.retain(census)

    assert "src/b.py" not in kept
    assert "src/a.py" in kept


def test_a_blob_keeps_what_copy2_kept_of_the_file(env, opening):
    """Restore copies a blob's mode and times onto the file it puts back
    (``_replace_with_payload``), so the blob keeps the source's, taken from
    the descriptor it was read through, and its extended attributes where
    the file system has them. The times are the source's after the read, as
    ``copy2`` stats after copying: an access time older than the change
    time moves when the file is read (``relatime``)."""
    target = env.root / "src" / "a.py"
    os.chmod(target, 0o751)
    os.utime(target, ns=(1_600_000_000_123_456_700, 1_650_000_000_987_654_300))
    xattr = None
    if hasattr(os, "setxattr"):
        try:
            os.setxattr(target, "user.multi-llm-test", b"kept")
            xattr = b"kept"
        except OSError:
            pass
    scheduler = env.scheduler([env.apply_task(0)])

    kept = scheduler._sweep.retain(scheduler._sweep.census())

    source = os.stat(target)
    blob = os.stat(scheduler._sweep.blob_path(kept["src/a.py"]))
    assert stat.S_IMODE(blob.st_mode) == stat.S_IMODE(source.st_mode)
    assert blob.st_mtime_ns == source.st_mtime_ns
    if not _copies_by_path(opening):
        # CopyFile2 sets the copy's access time itself, and reading the copy
        # back to check its bytes can move it.
        assert blob.st_atime_ns == source.st_atime_ns
    if xattr is not None:
        assert os.getxattr(scheduler._sweep.blob_path(kept["src/a.py"]),
                           "user.multi-llm-test") == xattr


@pytest.mark.parametrize("planted", ["to a file", "dangling"])
def test_a_link_planted_at_the_pending_name_is_replaced_never_written_through(
    env, tmp_path, opening, planted,
):
    """The pending copy's name is predictable (digest, pid, thread), and a
    runner that can write the phase dir can plant a link there before the
    pass runs. The copy is created exclusively and never through a link
    (the name cleared first; ``CopyFile2`` copies to a name drawn for the
    copy and the copy is renamed there; the plain copy is created without
    following a link, through one handle on Windows, where ``os.open``'s
    ``O_EXCL`` follows a dangling one), so the link's target is neither
    written, re-moded nor, for a dangling link, created."""
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    source = env.root / "src" / "a.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    blob = Path(scheduler._sweep.blob_path(digest))
    pending = Path(f"{blob}.{os.getpid()}.{threading.get_ident()}.tmp")
    pending.parent.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    if planted == "to a file":
        victim.write_bytes(b"not the executor's to write\n")
        os.chmod(victim, 0o600)
        mode = stat.S_IMODE(victim.stat().st_mode)
    pending.symlink_to(victim)

    kept = scheduler._sweep.retain(census)

    assert kept["src/a.py"] == digest
    if planted == "to a file":
        assert victim.read_bytes() == b"not the executor's to write\n"
        assert stat.S_IMODE(victim.stat().st_mode) == mode
    else:
        assert not os.path.lexists(victim)
    assert not blob.is_symlink() and blob.read_bytes() == source.read_bytes()
    assert not os.path.lexists(pending)


def _opens_as_windows_does(monkeypatch) -> None:
    """``os.open`` as Windows' C runtime runs it: ``O_CREAT | O_EXCL`` is
    ``CreateFile``'s ``CREATE_NEW``, which follows a link at the name to a
    target that does not exist and creates that target (the Windows CI,
    2026-11-12), and ``O_NOFOLLOW`` stops nothing, as Windows has none."""
    real = os.open

    def open_(path, flags, mode=0o777, *, dir_fd=None):
        if (dir_fd is None and flags & os.O_CREAT and flags & os.O_EXCL
                and os.path.islink(path) and not os.path.exists(path)):
            path = os.path.realpath(path)
        return real(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_)


@pytest.mark.parametrize("planted", ["to a file", "dangling"])
def test_the_plain_copy_on_windows_is_made_through_one_handle_never_through_a_link(
    env, tmp_path, monkeypatch, planted,
):
    """Windows without ``CopyFile2`` (Python before 3.12, a refused
    ``CopyFile2``): ``os.open``'s ``O_EXCL`` there follows a dangling link
    planted at the pending name and creates its target with the file's bytes
    (the Windows CI, 2026-11-12; emulated off Windows). The plain copy is
    made through one handle instead (``CREATE_NEW`` on the name itself,
    never followed), the name cleared when anything is there, and the
    source's times set through that handle: the link's target is neither
    written nor created, and the blob holds the file's bytes and
    modification time."""
    if os.name != "nt":
        calls = _HeldCopy()
        monkeypatch.setattr(write_safety, "_held_copy_calls", lambda: calls)
        monkeypatch.setattr(write_safety, "_EXCLUSIVE_OPEN_FOLLOWS_LINKS", True)
        _opens_as_windows_does(monkeypatch)
    monkeypatch.setattr(write_safety, "OPENS_WITHOUT_FOLLOWING", False, raising=False)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
    source = env.root / "src" / "a.py"
    os.utime(source, ns=(1_600_000_000_000_000_000, 1_650_000_000_987_654_300))
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    blob = Path(scheduler._sweep.blob_path(digest))
    pending = Path(f"{blob}.{os.getpid()}.{threading.get_ident()}.tmp")
    pending.parent.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    if planted == "to a file":
        victim.write_bytes(b"not the executor's to write\n")
    pending.symlink_to(victim)

    kept = scheduler._sweep.retain(census)

    assert kept["src/a.py"] == digest
    if planted == "to a file":
        assert victim.read_bytes() == b"not the executor's to write\n"
    else:
        assert not os.path.lexists(victim)
    assert not blob.is_symlink() and blob.read_bytes() == source.read_bytes()
    assert blob.stat().st_mtime_ns == 1_650_000_000_987_654_300
    assert not os.path.lexists(pending)
    if os.name != "nt":
        assert ("create", str(pending)) in calls.calls


@pytest.mark.skipif(os.name == "nt", reason="Windows will not move a file, or a "
                    "folder above it, while the file is held open")
@pytest.mark.parametrize("swap", ["folder redirected for the copy alone",
                                  "file replaced by one with the same bytes"])
def test_by_path_a_copy_of_another_file_is_refused_and_removed(
    env, tmp_path, monkeypatch, swap,
):
    """``CopyFile2`` copies by path, so what it reads is checked afterwards:
    the path must still name the file the descriptor read, and the copy must
    hold the bytes read through it. A folder above the file redirected to an
    outside one for the copy and put back (a junction's two flips) fails the
    bytes; the file replaced by another with the same bytes fails the
    identity. Either way nothing is retained and the pending copy is gone."""
    _by_path_with_copy_file2(monkeypatch)
    _plant(env, tmp_path)
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    relative = "src/deep/er/d.py"
    target = env.root / relative
    folder = target.parent
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    secret = b"bytes from outside, linked in for the copy alone\n"
    (elsewhere / "d.py").write_bytes(secret)
    copy = write_safety._COPY_FILE2
    fired = []

    def copy_another(source, destination, flags):
        if os.path.normcase(os.path.realpath(source)) != os.path.normcase(
                os.path.realpath(target)) or fired:
            return copy(source, destination, flags)
        fired.append(destination)
        if swap.startswith("folder"):
            saved = folder.with_name("er.saved")
            folder.rename(saved)
            folder.symlink_to(elsewhere, target_is_directory=True)
            try:
                return copy(source, destination, flags)
            finally:
                folder.unlink()
                saved.rename(folder)
        twin = target.with_name("d.twin")
        twin.write_bytes(target.read_bytes())
        os.replace(twin, target)
        return copy(source, destination, flags)

    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_another)
    kept = scheduler._sweep.retain(census)

    assert fired
    assert relative not in kept
    assert "src/a.py" in kept
    assert secret not in _store_bytes(scheduler)
    assert not os.path.lexists(fired[0])
    for blob in _blobs(scheduler):
        assert hashlib.sha256(blob.read_bytes()).hexdigest() == blob.name, blob


def _copying(target, then):
    """A ``CopyFile2`` that runs *then* (source, destination, flags, copy) in
    place of the copy of *target*, once; the returned list holds the
    destination once it ran."""
    copy = write_safety._COPY_FILE2
    wanted = os.path.normcase(os.path.realpath(target))
    fired: list = []

    def copy_file2(source, destination, flags):
        if fired or os.path.normcase(os.path.realpath(source)) != wanted:
            return copy(source, destination, flags)
        fired.append(destination)
        return then(source, destination, flags, copy)

    return copy_file2, fired


def _copy_then_change(source, destination, flags, copy) -> None:
    """The copy made, then changed: it no longer holds the bytes read."""
    copy(source, destination, flags)
    with open(destination, "r+b") as stream:
        stream.write(b"#")


_WINDOWS_COPY_FILE2 = pytest.mark.skipif(
    os.name != "nt" or getattr(write_safety, "_COPY_FILE2", None) is None,
    reason="CopyFile2, the read-only attribute and its destination-link "
           "behaviour are Windows'")


@_WINDOWS_COPY_FILE2
def test_a_refused_read_only_copy_is_removed(env, monkeypatch):
    """``CopyFile2`` copies the source's read-only attribute, and Windows
    will not delete a read-only file: a refused copy that is read-only is
    still removed (the attribute cleared on its own entry first), and no
    pending copy is left in the store."""
    lines: list = []
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
    census = scheduler._sweep.census()

    def copy_changed_read_only(source, destination, flags, copy) -> None:
        _copy_then_change(source, destination, flags, copy)
        os.chmod(destination, stat.S_IREAD)
        assert os.stat(destination).st_file_attributes & stat.FILE_ATTRIBUTE_READONLY

    copy_file2, fired = _copying(env.root / "src" / "a.py", copy_changed_read_only)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    kept = scheduler._sweep.retain(census)

    assert fired
    assert "src/a.py" not in kept
    assert not os.path.lexists(fired[0])
    assert not [blob for blob in _blobs(scheduler) if blob.name.endswith(".tmp")]
    assert not [line for line in lines if "WARNING" in line], lines


@pytest.mark.parametrize("failure", [
    "copied by path, then refused",
    pytest.param("the copy failed midway", marks=pytest.mark.skipif(
        os.name == "nt", reason="Windows makes its plain copy through one handle: "
        "the handle row")),
    "the handle copy failed midway",
])
def test_a_refused_copy_that_cannot_be_removed_is_reported(
    env, monkeypatch, failure,
):
    """A refused copy can hold another file's bytes, so one that cannot be
    removed is not dropped like an ordinary failed copy: the file has no
    rollback point, a warning names what is left, the retention warning
    gives it as the reason, and its bytes count against the store budget.
    The handle row is Windows' plain copy (``_written_through_handle``,
    emulated off Windows): its delete through the handle is refused too."""
    target = env.root / "src" / "a.py"
    wanted = os.stat(target)
    fired: list = []

    def first_for_the_target(info) -> bool:
        if fired or (info.st_dev, info.st_ino) != (wanted.st_dev, wanted.st_ino):
            return False
        fired.append(True)
        return True

    if failure.startswith("copied by path"):
        _by_path_with_copy_file2(monkeypatch)
        copy_file2, fired = _copying(target, _copy_then_change)
        monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    elif failure == "the copy failed midway":
        monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
        copy_xattrs = write_safety._copy_xattrs

        def fail_midway(source, copy) -> None:
            if first_for_the_target(os.fstat(source)):
                raise OSError(errno.EIO, "failed midway")
            return copy_xattrs(source, copy)

        monkeypatch.setattr(write_safety, "_copy_xattrs", fail_midway)
    else:
        if os.name != "nt":
            calls = _HeldCopy()
            monkeypatch.setattr(write_safety, "_held_copy_calls", lambda: calls)
            monkeypatch.setattr(write_safety, "_EXCLUSIVE_OPEN_FOLLOWS_LINKS", True)
            _opens_as_windows_does(monkeypatch)
        monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
        attributes = write_safety._copied_attributes

        def fail_as_it_sets_them(made, info):
            if first_for_the_target(info):
                raise OSError(errno.EIO, "failed midway")
            return attributes(made, info)

        def refuse(self, held):
            raise PermissionError(errno.EACCES, "refused")

        monkeypatch.setattr(write_safety, "_copied_attributes", fail_as_it_sets_them)
        monkeypatch.setattr(type(write_safety._held_copy_calls()), "dispose", refuse)
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if os.fspath(path).endswith(".tmp"):
            raise PermissionError(errno.EACCES, "refused", os.fspath(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    lines: list = []
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)

    kept = scheduler._sweep.retain(scheduler._sweep.census())

    assert fired
    assert "src/a.py" not in kept
    left = [blob for blob in _blobs(scheduler) if blob.name.endswith(".tmp")]
    assert len(left) == 1
    warnings = [line for line in lines if "WARNING: src/a.py has no rollback point" in line]
    assert len(warnings) == 1 and str(left[0]) in warnings[0], lines
    assert "delete it by hand" in warnings[0]
    assert any(f"src/a.py (its refused copy is left at {left[0]})" in line
               for line in lines), lines
    blobs = [blob for blob in _blobs(scheduler) if not blob.name.endswith(".tmp")]
    assert scheduler._sweep._retained_bytes == (
        sum(blob.stat().st_size for blob in blobs) + left[0].stat().st_size)


def test_a_link_at_the_pending_name_once_copied_is_refused_and_named(
    env, tmp_path, monkeypatch,
):
    """Whatever ``CopyFile2`` makes of a link put at the name it copies to
    (which needs that name's token, drawn for the copy), a copy that is a
    link once made is refused: the link is removed, never followed, nothing
    is retained, and the warning names where the copy may have been
    written, which the pass does not remove (a path someone else chose is
    not its to delete)."""
    _by_path_with_copy_file2(monkeypatch)
    followed = tmp_path / "followed.txt"

    def copy_through_a_link(source, destination, flags, copy) -> None:
        # What a copy that followed a dangling link planted there leaves.
        os.symlink(followed, destination)
        followed.write_bytes(Path(source).read_bytes())

    copy_file2, fired = _copying(env.root / "src" / "a.py", copy_through_a_link)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    lines: list = []
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)

    kept = scheduler._sweep.retain(scheduler._sweep.census())

    assert fired
    assert "src/a.py" not in kept
    assert not os.path.lexists(fired[0])
    assert followed.exists()
    warnings = [line for line in lines if "WARNING: src/a.py has no rollback point" in line]
    assert len(warnings) == 1 and followed.name in warnings[0], lines
    assert "was a link to" in warnings[0]


def test_a_dangling_link_planted_before_the_copy_is_not_followed(
    env, tmp_path, monkeypatch,
):
    """The pending name is cleared, then the file is copied. A dangling link
    planted at that predictable name in between was followed on Windows when
    ``CopyFile2`` copied to it: its target was created with the file's bytes,
    ``COPY_FILE_FAIL_IF_EXISTS`` with ``COPY_FILE_COPY_SYMLINK``
    notwithstanding, though ``CopyFileEx`` documents that the pair fails on
    any link there (the Windows CI, 2026-11-12; emulated so elsewhere). So
    ``CopyFile2`` never copies to the pending name: it copies to a name
    beside it drawn for that copy, and the checked copy is renamed over
    whatever is at the pending name by then, never through it. The link's
    target is never created, and the blob holds the file's bytes."""
    _by_path_with_copy_file2(monkeypatch)
    source = env.root / "src" / "a.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scheduler = env.scheduler([env.apply_task(0)])
    blob = Path(scheduler._sweep.blob_path(digest))
    pending = Path(f"{blob}.{os.getpid()}.{threading.get_ident()}.tmp")
    absent = tmp_path / "never-created.txt"

    def plant_then_copy(source, destination, flags, copy) -> None:
        pending.symlink_to(absent)
        copy(source, destination, flags)

    copy_file2, fired = _copying(source, plant_then_copy)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)

    kept = scheduler._sweep.retain(scheduler._sweep.census())

    assert fired
    assert Path(fired[0]).parent == pending.parent and Path(fired[0]) != pending
    assert not os.path.lexists(absent)
    assert kept["src/a.py"] == digest
    assert not blob.is_symlink() and blob.read_bytes() == source.read_bytes()
    assert not os.path.lexists(pending) and not os.path.lexists(fired[0])
    for kept_blob in _blobs(scheduler):
        assert hashlib.sha256(kept_blob.read_bytes()).hexdigest() == kept_blob.name


#: A token pinned for the name ``CopyFile2`` copies to, so a test can plant
#: something there beforehand, as only someone who knew the token could.
COPY_TOKEN = "ab" * 8


def test_a_link_planted_at_the_copys_own_name_is_removed_never_followed(
    env, tmp_path, monkeypatch,
):
    """The name ``CopyFile2`` copies to (``.copy.<pid>.<token>.tmp`` beside the
    pending name) is drawn for each copy; were it known, as here with the
    token pinned, a dangling link planted there before the pass is removed
    like one at the pending name, never followed: its target is never
    created, and the blob holds the file's bytes."""
    _by_path_with_copy_file2(monkeypatch)
    monkeypatch.setattr(write_safety, "_copy_token", lambda: COPY_TOKEN)
    source = env.root / "src" / "a.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._sweep.census()
    blob = Path(scheduler._sweep.blob_path(digest))
    copied = blob.with_name(f".copy.{os.getpid()}.{COPY_TOKEN}.tmp")
    copied.parent.mkdir(parents=True)
    absent = tmp_path / "never-created.txt"
    copied.symlink_to(absent)

    kept = scheduler._sweep.retain(census)

    assert not os.path.lexists(absent)
    assert kept["src/a.py"] == digest
    assert not blob.is_symlink() and blob.read_bytes() == source.read_bytes()
    assert not os.path.lexists(copied)


def test_copy_file2_is_given_a_name_drawn_for_each_copy(env, monkeypatch):
    """Every name ``CopyFile2`` is given lies beside its pending name, in the
    same folder (so the rename that follows stays on one filesystem), is
    never that predictable name, and holds a token drawn for that copy, so
    no two copies share one and none can be known before the copy."""
    _by_path_with_copy_file2(monkeypatch)
    copy = write_safety._COPY_FILE2
    given: list = []

    def recording(source, destination, flags):
        given.append(Path(destination))
        return copy(source, destination, flags)

    monkeypatch.setattr(write_safety, "_COPY_FILE2", recording)
    draw = write_safety._copy_token
    drawn: list = []
    monkeypatch.setattr(write_safety, "_copy_token",
                        lambda: drawn.append(draw()) or drawn[-1])
    scheduler = env.scheduler([env.apply_task(0)])

    kept = scheduler._sweep.retain(scheduler._sweep.census())

    blobs = {Path(scheduler._sweep.blob_path(digest)) for digest in kept.values()}
    assert len(given) == len(drawn) == len(blobs) >= 2
    assert len(set(drawn)) == len(drawn)
    for name, token in zip(given, drawn):
        assert name.name == f".copy.{os.getpid()}.{token}.tmp" and len(token) == 16
        assert name.parent in {blob.parent for blob in blobs}
        assert not os.path.lexists(name)


def test_a_copy_a_killed_process_left_is_reclaimed_once_its_pid_is_gone(
    tmp_path, monkeypatch,
):
    """A process killed between ``CopyFile2`` and the rename leaves its copy
    under a name no later run can derive. The first copy into the folder in
    each process removes those whose pid no process has any more, each entry
    itself (a link there removed, never followed), and nothing else: not a
    live pid's (a copy in flight elsewhere), not this process's, not a name
    of another shape. Once per folder: a second copy does not look again."""
    dead = 999_999
    monkeypatch.setattr(write_safety, "process_alive", lambda pid: pid != dead)
    write_safety._RECLAIMED.clear()
    folder = tmp_path / "store"
    folder.mkdir()
    target = tmp_path / "elsewhere.txt"
    target.write_bytes(b"not the pass's")
    left = folder / f".copy.{dead}.{'ab' * 8}.tmp"
    left.write_bytes(b"a whole copy")
    left.chmod(0o444)
    linked = folder / f".copy.{dead}.{'ef' * 8}.tmp"
    linked.symlink_to(target)
    kept = [
        folder / f".copy.{os.getppid()}.{'cd' * 8}.tmp",
        folder / f".copy.{os.getpid()}.{'12' * 8}.tmp",
        folder / f".copy.{dead}.not-a-token.tmp",
        folder / "blob.tmp",
    ]
    for path in kept:
        path.write_bytes(b"kept")

    name = write_safety._copy_name(str(folder / "blob.123.456.tmp"))

    assert Path(name).parent == folder
    assert not os.path.lexists(left) and not os.path.lexists(linked)
    assert target.read_bytes() == b"not the pass's", "the link was followed"
    assert all(path.exists() for path in kept)
    left.write_bytes(b"left after the look")
    write_safety._copy_name(str(folder / "blob.123.456.tmp"))
    assert left.exists(), "a folder is looked through once per process"


def test_a_name_put_at_the_pending_name_before_the_rename_is_reported(
    tmp_path, monkeypatch,
):
    """Something put at the pending name between its clearing and the rename
    (here a folder) refuses ``os.replace``: the copy is removed, and what
    cannot be removed at the pending name is ``CopyNotContained``, never a
    plain ``OSError`` its caller would let pass in silence."""
    _by_path_with_copy_file2(monkeypatch)
    write_safety._RECLAIMED.clear()
    copy = write_safety._COPY_FILE2
    folder = tmp_path / "store"
    folder.mkdir()
    pending = folder / "blob.1.2.tmp"
    made: list = []

    def copy_then_plant(source, destination, flags):
        copy(source, destination, flags)
        made.append(Path(destination))
        pending.mkdir()

    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_then_plant)
    source_path = tmp_path / "a.py"
    source_path.write_bytes(b"print(1)\n")
    source = os.open(source_path, os.O_RDONLY)
    try:
        with pytest.raises(write_safety.CopyNotContained):
            write_safety.write_copy(str(pending), b"print(1)\n", source,
                                    path=str(source_path))
    finally:
        os.close(source)
    assert made and not os.path.lexists(made[0]), "the copy stayed"
    assert pending.is_dir()


@pytest.mark.parametrize("winerror", [5, 32], ids=["access denied", "sharing violation"])
def test_by_path_a_refused_copy_falls_back_as_copy2_does(env, monkeypatch, winerror):
    """Where ``CopyFile2`` refuses access, ``shutil.copy2`` falls back to a
    plain copy and so does the pass, from the bytes it read; any other
    ``CopyFile2`` failure leaves the file without a rollback point, as
    ``copy2`` raising did. Either way, what the failed copy left at the
    pending name (part of a copy, which can be another file's bytes) is
    removed."""
    _by_path_with_copy_file2(monkeypatch)

    def refuse(source, destination, flags):
        Path(destination).write_bytes(b"part of a copy")
        error = OSError(errno.EACCES, "refused", source)
        error.winerror = winerror
        raise error

    monkeypatch.setattr(write_safety, "_COPY_FILE2", refuse)
    scheduler = env.scheduler([env.apply_task(0)])
    source = env.root / "src" / "a.py"

    kept = scheduler._sweep.retain(scheduler._sweep.census())

    if winerror == 5:
        blob = Path(scheduler._sweep.blob_path(kept["src/a.py"]))
        assert blob.read_bytes() == source.read_bytes()
        assert blob.stat().st_mtime_ns == source.stat().st_mtime_ns
    else:
        assert "src/a.py" not in kept
        assert not _blobs(scheduler)


@pytest.mark.parametrize("allow", [["**"], ["src/**"]], ids=["default allow", "narrow allow"])
def test_the_unresolved_check_answers_as_the_resolving_one_for_a_path_without_links(
    env, tmp_path, allow,
):
    """``violation_relative`` is ``violation(path, exact=False)`` for a path
    that holds no link, rule text included, and ``open_permitted`` refuses
    what either refuses: the walk asks the one, Windows asks the other."""
    _plant(env, tmp_path)
    (env.root / "node_modules" / "pkg").mkdir(parents=True)
    (env.root / "node_modules" / "pkg" / "i.js").write_text("i\n", encoding="utf-8")
    policy = write_safety.WritePolicy.load(str(env.root), anchor=str(env.plan))
    policy = write_safety.WritePolicy(root=policy.root, allow=allow, deny=policy.deny)
    for relative in ("src/a.py", "src/deep/er/d.py", "plans/p.md", ".env",
                     "node_modules/pkg/i.js", ".multi-llm/p/apply-suggestions-old/notes.md"):
        absolute = str(env.root / relative)
        expected = policy.violation(absolute, exact=False)
        assert policy.violation_relative(relative) == expected, relative
        opened = policy.open_permitted(absolute)
        if opened is not None:
            os.close(opened[0])
        assert (opened is None) == (expected is not None), relative
    assert policy.violation_relative("../outside/private.txt") is not None
    assert policy.open_permitted(str(tmp_path / "outside" / "private.txt")) is None
    assert policy.open_permitted(str(env.root / "link.txt")) is None


@pytest.mark.parametrize("walk", [True, False], ids=["descriptor walk", "by path"])
def test_a_path_spelled_with_dots_or_a_trailing_separator_is_never_opened(
    env, monkeypatch, walk,
):
    """``src/a.py/`` names no file, and ``src/link/../a.py`` names what lies
    beside the link's target, yet the resolving check permits both spellings
    of ``src/a.py`` and the walk would open ``src/a.py`` for them. Such a
    spelling is refused outright, so what is opened never depends on how the
    caller spells the path (the census never spells one so)."""
    if walk and not write_safety.OPENS_WITHOUT_FOLLOWING:
        pytest.skip("this platform opens by path only")
    if not walk:
        monkeypatch.setattr(write_safety, "OPENS_WITHOUT_FOLLOWING", False)
    policy = write_safety.WritePolicy.load(str(env.root), anchor=str(env.plan))
    plain = os.path.join(str(env.root), "src", "a.py")
    opened = policy.open_permitted(plain)
    assert opened is not None
    os.close(opened[0])
    for spelled in (
        plain + os.sep,
        os.path.join(str(env.root), "src", "..", "src", "a.py"),
        os.path.join(str(env.root), "src", ".", "a.py"),
        "src/a.py/",
        "src/../src/a.py",
    ):
        absolute = spelled if os.path.isabs(spelled) else os.path.join(str(env.root), spelled)
        assert policy.violation(absolute, exact=False) is None, spelled
        assert policy.open_permitted(spelled) is None, spelled


# ---------------------------------------------------------------------------
# Restore and the store's discard (docs/development/write-safety.md, restore):
# a payload that is not read-only goes back on Windows as ``CopyFile2``'s
# copy, checked and swapped in through one handle held on it alone; every
# other restore stages the payload's bytes itself, through one descriptor or
# handle; and the store is removed whole, or what is left is named.
# ---------------------------------------------------------------------------

#: The ``restoring`` ways, by what writes the file put back.
STAGED, STAGED_BY_HANDLE, COPY_FILE2 = (
    "staged copy", "staged copy, Windows calls", "CopyFile2")


@pytest.fixture(params=[STAGED, STAGED_BY_HANDLE, COPY_FILE2])
def restoring(request, monkeypatch):
    """Each way a restore writes the file it puts back: the payload's bytes
    staged by the restore itself, through descriptors (Linux, macOS) or
    through one handle (Windows: a read-only payload, Python before 3.12;
    emulated elsewhere), or ``CopyFile2``'s copy held through one handle
    (Windows on Python 3.12+ for a payload that is not read-only, emulated
    elsewhere)."""
    if request.param == STAGED:
        monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
    elif request.param == STAGED_BY_HANDLE:
        _staging_by_handle(monkeypatch)
    else:
        _restoring_by_copy_file2(monkeypatch)
    return request.param


def _winerror(code: int, message: str, path) -> PermissionError:
    error = PermissionError(errno.EACCES, message, os.fspath(path))
    error.winerror = code
    return error


class _HeldCopy:
    """The Windows calls a restore makes on the one handle it holds on the
    file it puts back (``write_safety._HeldCopyCalls``), over POSIX.

    ``bind`` opens the name itself: a link as the link, never followed (as
    ``FILE_FLAG_OPEN_REPARSE_POINT`` does); a read-only file is refused, as
    Windows opens one for writing to nobody, and so is a file in *writers*,
    one another writer holds open (a sharing violation). ``create`` makes the
    name as a new file, failing on any name there, a link included. Windows
    keeps the bound name on the file while it is held (no delete sharing), so
    ``rename`` and ``dispose`` act on that name, checked to still be the
    file bound; ``dispose`` refuses a read-only file, as Windows deletes
    none. *calls* records each call with the name it reached."""

    def __init__(self) -> None:
        self.writers: set = set()
        self.bound: dict = {}
        self.calls: list = []
        # Taken now: a test that records calls naming the copy patches `os`,
        # and one that runs `os.open` as Windows does follows a link with it.
        self._open, self._lstat, self._fstat = os.open, os.lstat, os.fstat
        self._unlink, self._replace = os.unlink, os.replace
        self._utime, self._fchmod = os.utime, getattr(os, "fchmod", None)

    def bind(self, path):
        named = self._lstat(path)
        if stat.S_ISLNK(named.st_mode):
            if hasattr(os, "O_PATH"):
                held = os.open(path, os.O_PATH | os.O_NOFOLLOW)
            elif hasattr(os, "O_SYMLINK"):  # pragma: no cover - macOS
                held = os.open(path, os.O_SYMLINK)
            else:  # pragma: no cover
                pytest.skip("this platform cannot open a link itself")
        elif (named.st_dev, named.st_ino) in self.writers:
            raise _winerror(32, "it is being used by another process", path)
        elif not named.st_mode & stat.S_IWUSR:
            raise _winerror(5, "access is denied", path)
        else:
            held = self._open(path, os.O_RDWR | os.O_NOFOLLOW)
        self.bound[held] = os.fspath(path)
        self.calls.append(("bind", os.fspath(path)))
        return held

    def create(self, path):
        held = self._open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.bound[held] = os.fspath(path)
        self.calls.append(("create", os.fspath(path)))
        return held

    def _name_of(self, held) -> str:
        path = self.bound[held]
        named, opened = self._lstat(path), self._fstat(held)
        assert (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino), (
            "Windows keeps the bound name on the file while it is held")
        return path

    def set_basic(self, held, attributes, atime_ns, mtime_ns):
        self.calls.append(("set_basic", self._name_of(held)))
        self._utime(held, ns=(atime_ns, mtime_ns))
        mode = stat.S_IMODE(self._fstat(held).st_mode)
        if attributes & stat.FILE_ATTRIBUTE_READONLY:
            self._fchmod(held, mode & ~0o222)
        elif not mode & stat.S_IWUSR:
            self._fchmod(held, mode | stat.S_IWUSR)

    def rename(self, held, target):
        path = self._name_of(held)
        self.calls.append(("rename", path))
        self._replace(path, target)
        self.bound[held] = os.fspath(target)

    def dispose(self, held):
        path = self._name_of(held)
        self.calls.append(("dispose", path))
        if not self._fstat(held).st_mode & stat.S_IWUSR:
            raise _winerror(5, "access is denied", path)
        self._unlink(path)


def _restoring_by_copy_file2(monkeypatch):
    """``CopyFile2`` and the calls on the handle a restore holds on its copy:
    Windows' own on Windows on Python 3.12+, emulated elsewhere. Returns the
    handle calls."""
    if getattr(write_safety, "_COPY_FILE2", None) is None:
        monkeypatch.setattr(write_safety, "_COPY_FILE2", _copy_file2_off_windows,
                            raising=False)
    if os.name == "nt":
        calls = write_safety._held_copy_calls()
        assert calls is not None, "the handle calls load on Windows"
        return calls
    calls = _HeldCopy()
    monkeypatch.setattr(write_safety, "_held_copy_calls", lambda: calls, raising=False)
    return calls


def _staging_by_handle(monkeypatch):
    """The restore Windows makes where ``CopyFile2`` does not restore (a
    read-only payload, Python before 3.12): the payload's bytes written into
    a file it creates and holds through one handle. Emulated off Windows;
    on Windows it is the ``staged copy`` way. Returns the handle calls."""
    if os.name == "nt":
        pytest.skip("on Windows the staged copy is held through these calls already")
    monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
    monkeypatch.setattr(write_safety, "_STAGES_BY_DESCRIPTOR", False, raising=False)
    calls = _HeldCopy()
    monkeypatch.setattr(write_safety, "_held_copy_calls", lambda: calls, raising=False)
    settable = write_safety._settable_attributes

    def as_on_windows(info):
        # Windows' fstat reports a file without write permission as read-only.
        read_only = 0 if info.st_mode & stat.S_IWUSR else stat.FILE_ATTRIBUTE_READONLY
        return settable(info) & ~0x80 | read_only or settable(info)

    monkeypatch.setattr(write_safety, "_settable_attributes", as_on_windows)
    return calls


def _before_bind(monkeypatch, calls, then) -> list:
    """Run *then* (the copy's name) once, just before the restore binds its
    copy; the returned list holds that name once it ran."""
    bind = calls.bind
    ran: list = []

    def binding(path):
        if not ran:
            ran.append(path)
            then(path)
        return bind(path)

    monkeypatch.setattr(calls, "bind", binding)
    return ran


def _snapshot_of(env, *relatives):
    """A snapshot of *relatives* (default ``src/a.py``), and the payload path
    of the first."""
    relatives = relatives or ("src/a.py",)
    snapshot = write_safety.TaskSnapshot.take(
        phase_dir=str(env.phase_dir), task_id="T-R",
        files=[str(env.root / relative) for relative in relatives], root=str(env.root),
    )
    entry = snapshot.entry_for(relatives[0])
    payload = Path(snapshot.directory) / write_safety.PAYLOAD_DIRNAME / (entry.payload or "")
    return snapshot, payload


#: The token in every restore's staging name here, so a test can name the
#: staged file, or plant something at its name, beforehand; a restore draws a
#: new one for each attempt (``write_safety._staging_token``).
STAGING_TOKEN = "0123abcd"


@pytest.fixture(autouse=True)
def _staging_token_pinned(monkeypatch):
    """Every restore here stages under :func:`_restore_tmp`'s name."""
    monkeypatch.setattr(write_safety, "_staging_token", lambda: STAGING_TOKEN)


def _restore_tmp(target: Path) -> Path:
    return target.with_name(f".{target.name}.restore.{os.getpid()}.{STAGING_TOKEN}.tmp")


def _tmpfile_here(folder: Path) -> bool:
    """Whether a restore staging through descriptors in *folder* makes its
    file with no name, as ``write_safety._staged`` decides: it tries
    ``O_TMPFILE`` (Linux), *folder*'s filesystem makes such files, and
    ``/proc/self/fd`` can name one. Elsewhere (macOS, a filesystem without
    ``O_TMPFILE``) the staged file is made by its name."""
    flag = write_safety._O_TMPFILE
    if not flag or not os.path.isdir(write_safety._PROC_SELF_FD):
        return False
    try:
        handle = os.open(folder, flag | os.O_RDWR, 0o600)
    except OSError:
        return False
    os.close(handle)
    return True


#: How a restore's ``skipped`` state ends once it left a staged file of its
#: own for the user, recorded in this process (``write_safety.restore_leftovers``).
KEPT_FOR_THE_USER = ("until it is gone, no write check in this process counts it "
                     "as a batch's file and no rollback in it replaces or deletes "
                     "it; another process checks it as any other file")


def _kept_for_the_user(state: str) -> None:
    """*state* names a staged file the restore kept, says how to delete it
    and that the write checks leave it alone meanwhile."""
    assert "check it and delete it by hand (" in state, state
    assert state.endswith(KEPT_FOR_THE_USER), state


def test_a_restore_puts_back_the_payloads_bytes_and_times(env, monkeypatch, restoring):
    """Every way the file put back holds the payload's bytes and its
    modification time, and no temp file is left beside it; where
    ``CopyFile2`` copies, it copied the payload, and the copy was bound,
    given the payload's attributes and times, and renamed over the file,
    each through the handle; where the restore stages the bytes through the
    Windows calls, the file was created by them instead of bound."""
    target = env.root / "src" / "a.py"
    os.utime(target, ns=(1_600_000_000_000_000_000, 1_650_000_000_987_654_300))
    original = target.read_bytes()
    snapshot, payload = _snapshot_of(env)
    copies: list = []
    copy = getattr(write_safety, "_COPY_FILE2", None)
    if copy is not None:
        def recording(source, destination, flags):
            copies.append(source)
            return copy(source, destination, flags)

        monkeypatch.setattr(write_safety, "_COPY_FILE2", recording)
    target.write_bytes(b"a = 2\n")

    assert snapshot.restore(["src/a.py"]) == {"src/a.py": "restored"}

    assert target.read_bytes() == original
    assert target.stat().st_mtime_ns == 1_650_000_000_987_654_300
    assert not os.path.lexists(_restore_tmp(target))
    assert [os.path.normcase(source) for source in copies] == (
        [os.path.normcase(str(payload))] if restoring == COPY_FILE2 else [])
    calls = write_safety._held_copy_calls() if restoring != STAGED else None
    if isinstance(calls, _HeldCopy):
        assert [name for name, _path in calls.calls] == [
            "bind" if restoring == COPY_FILE2 else "create", "set_basic", "rename"]


def test_the_refusal_rules_hold_either_way(env, restoring):
    """A file changed after the run is refused without ``--force`` and put
    back with it, and a file the batch created is deleted."""
    target = env.root / "src" / "a.py"
    original = target.read_bytes()
    snapshot, _payload = _snapshot_of(env, "src/a.py", "src/new.py")
    target.write_bytes(b"a = 2\n")
    (env.root / "src" / "new.py").write_text("new = 1\n", encoding="utf-8")
    recorded = {"src/a.py": write_safety.file_sha256(str(target))}
    target.write_bytes(b"a = 3, by hand\n")

    refused = snapshot.restore(["src/a.py"], expected=recorded)
    assert refused["src/a.py"].startswith("skipped: changed after the run")
    assert target.read_bytes() == b"a = 3, by hand\n"

    forced = snapshot.restore(expected=recorded, force=True)
    assert forced == {"src/a.py": "restored", "src/new.py": "deleted"}
    assert target.read_bytes() == original
    assert not (env.root / "src" / "new.py").exists()


def test_a_symlink_now_at_the_path_is_replaced_as_the_link_itself(env, restoring):
    """The file put back replaces the link's own entry (``os.replace``, or
    the held copy's rename), never written through it: the link's target
    keeps its bytes."""
    target = env.root / "src" / "a.py"
    original = target.read_bytes()
    snapshot, _payload = _snapshot_of(env)
    victim = env.tmp_path / "victim.txt"
    victim.write_bytes(b"not the restore's to write\n")
    target.unlink()
    target.symlink_to(victim)

    assert snapshot.restore(force=True) == {"src/a.py": "restored"}

    assert not target.is_symlink() and target.read_bytes() == original
    assert victim.read_bytes() == b"not the restore's to write\n"


def test_a_read_only_file_comes_back_read_only(env, restoring):
    """The payload keeps the file's mode (on Windows its read-only
    attribute), and so does the file put back."""
    target = env.root / "src" / "a.py"
    os.chmod(target, stat.S_IREAD)
    original = target.read_bytes()
    try:
        snapshot, _payload = _snapshot_of(env)
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        target.write_bytes(b"a = 2\n")

        assert snapshot.restore(["src/a.py"]) == {"src/a.py": "restored"}

        assert target.read_bytes() == original
        assert not stat.S_IMODE(target.stat().st_mode) & stat.S_IWRITE
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)


def test_a_read_only_payload_is_staged_by_the_restore_itself(env, monkeypatch, restoring):
    """A read-only payload is never copied by ``CopyFile2``, even where it
    copies: its copy would be read-only, which Windows opens for writing to
    nobody, so it could not be held. The restore writes the payload's bytes
    into a file it holds from its creation instead: never bound by name
    afterwards, flushed before it takes the payload's mode, and read-only
    when it replaces the file (seamed: ``os.fsync`` and the rename record
    the file each reaches, in order; the Windows calls record themselves)."""
    copies: list = []
    bound: list = []
    calls = write_safety._held_copy_calls() if restoring != STAGED else None
    if restoring == COPY_FILE2:
        copy = write_safety._COPY_FILE2

        def recording(source, destination, flags):
            copies.append(source)
            return copy(source, destination, flags)

        monkeypatch.setattr(write_safety, "_COPY_FILE2", recording)
    if calls is not None:
        bind = calls.bind

        def binding(path):
            bound.append(path)
            return bind(path)

        monkeypatch.setattr(calls, "bind", binding)
    target = env.root / "src" / "a.py"
    os.chmod(target, stat.S_IREAD)
    original = target.read_bytes()
    events: list = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(handle):
        info = os.fstat(handle)
        events.append(("fsync", (info.st_dev, info.st_ino)))
        return real_fsync(handle)

    def replace(source, destination, *args, **kwargs):
        if os.path.basename(os.fspath(destination)) == target.name:
            named = os.lstat(source, dir_fd=kwargs.get("src_dir_fd"))
            events.append(("replace", (named.st_dev, named.st_ino),
                           bool(stat.S_IMODE(named.st_mode) & stat.S_IWRITE)))
        return real_replace(source, destination, *args, **kwargs)

    try:
        snapshot, _payload = _snapshot_of(env)
        # The snapshot's own copy (``TaskSnapshot.take`` copies the file as
        # ``shutil.copy2`` does where ``CopyFile2`` copies) is not the restore's.
        copies.clear()
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        target.write_bytes(b"a = 2\n")
        with monkeypatch.context() as patched:
            patched.setattr(os, "fsync", fsync)
            patched.setattr(os, "replace", replace)
            assert snapshot.restore(["src/a.py"]) == {"src/a.py": "restored"}

        assert target.read_bytes() == original
        assert not stat.S_IMODE(target.stat().st_mode) & stat.S_IWRITE
        assert not os.path.lexists(_restore_tmp(target))
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)

    assert copies == [] and bound == []
    # Windows stages it through the handle calls (emulated: STAGED_BY_HANDLE);
    # elsewhere through descriptors, whatever CopyFile2 would do.
    by_handle = restoring == STAGED_BY_HANDLE or os.name == "nt"
    if isinstance(calls, _HeldCopy):
        assert [name for name, _path in calls.calls] == (
            ["create", "set_basic", "rename"] if by_handle else [])
    if by_handle:
        return
    replaced = [event for event in events if event[0] == "replace"]
    assert len(replaced) == 1, events
    _, put_back, writable = replaced[0]
    assert not writable, "the copy is read-only when it replaces the file"
    assert ("fsync", put_back) in events[:events.index(replaced[0])], events


def test_the_copy_is_checked_flushed_read_back_and_renamed_through_its_handle(
    env, monkeypatch,
):
    """Through the one handle held on the copy, in this order: what it is
    (``fstat``: a regular file with one name), flushed, read back after the
    flush and hashed, its attributes and times set, renamed over the file.
    No other descriptor reaches it (seamed: ``os.fstat``, ``os.fsync`` and
    ``os.read`` on the held descriptor, and the handle calls, record
    themselves)."""
    calls = _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    original = target.read_bytes()
    snapshot, _payload = _snapshot_of(env)
    target.write_bytes(b"a = 2\n")
    held: list = []
    events: list = []
    bind = calls.bind

    def binding(path):
        handle = bind(path)
        held.append(handle)
        return handle

    def recorded(name, real):
        def call(handle, *args, **kwargs):
            if held and handle == held[0]:
                events.append(name)
            return real(handle, *args, **kwargs)
        return call

    monkeypatch.setattr(calls, "bind", binding)
    for name in ("set_basic", "rename", "dispose"):
        monkeypatch.setattr(calls, name, recorded(name, getattr(calls, name)))
    with monkeypatch.context() as patched:
        for name in ("fstat", "fsync", "read"):
            patched.setattr(os, name, recorded(name, getattr(os, name)))
        assert snapshot.restore(["src/a.py"]) == {"src/a.py": "restored"}

    assert target.read_bytes() == original
    once = [name for index, name in enumerate(events)
            if not index or name != events[index - 1]]
    assert once == ["fstat", "fsync", "read", "set_basic", "rename"], events


def test_a_restore_copy_that_does_not_hold_the_payloads_bytes_is_refused(env, monkeypatch):
    """``CopyFile2`` copies by path, so its copy is read back before it
    replaces anything: a copy that does not hash to the payload's bytes is
    deleted through its handle and the file is left as it is."""
    _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    snapshot, payload = _snapshot_of(env)
    target.write_bytes(b"a = 2\n")
    copy_file2, fired = _copying(payload, _copy_then_change)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)

    report = snapshot.restore(["src/a.py"])

    assert fired
    assert report["src/a.py"].startswith("skipped:")
    assert "does not hold the bytes" in report["src/a.py"]
    assert target.read_bytes() == b"a = 2\n"
    assert not os.path.lexists(fired[0])


def test_a_copy_changed_before_the_bind_is_refused(env, monkeypatch):
    """The copy rewritten in place between ``CopyFile2`` and the bind is
    bound (still one name, a regular file), flushed and read back through the
    handle: its bytes no longer hash to the payload's, so nothing is put
    back, the copy is deleted through the handle and the snapshot is kept."""
    calls = _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    snapshot, payload = _snapshot_of(env)
    target.write_bytes(b"runner change\n")

    def rewrite(name):
        with open(name, "r+b") as stream:
            stream.write(b"not the payload's bytes\n")

    ran = _before_bind(monkeypatch, calls, rewrite)
    report = snapshot.restore()

    assert ran
    assert report["src/a.py"].startswith("skipped:")
    assert "does not hold the bytes" in report["src/a.py"]
    assert target.read_bytes() == b"runner change\n"
    assert not os.path.lexists(_restore_tmp(target))
    assert payload.exists()
    if isinstance(calls, _HeldCopy):
        assert [name for name, _path in calls.calls] == ["bind", "dispose"]


@pytest.mark.parametrize("outside", ["read-only", "writable"])
def test_a_hard_link_swapped_in_before_the_bind_is_refused_and_left_alone(
    env, tmp_path, monkeypatch, outside,
):
    """The copy's name swapped for a hard link to a file outside the
    repository before the bind: a read-only one cannot be bound (Windows
    opens it for writing to nobody), and is named, left as it is; a
    writable one is bound and refused for its second name, and only the
    link at the temp name is deleted, through the handle. Either way nothing
    is put back, and the outside file keeps its bytes, its mode and its own
    name."""
    calls = _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    snapshot, payload = _snapshot_of(env)
    target.write_bytes(b"runner change\n")
    victim = tmp_path / "outside.txt"
    victim.write_bytes(b"outside the repository\n")
    os.chmod(victim, stat.S_IREAD if outside == "read-only" else stat.S_IREAD | stat.S_IWRITE)
    mode = stat.S_IMODE(victim.stat().st_mode)
    unlink, link = os.unlink, os.link

    def swap(name):
        unlink(name)
        link(victim, name)

    ran = _before_bind(monkeypatch, calls, swap)
    try:
        report = snapshot.restore()
        after = stat.S_IMODE(victim.stat().st_mode)
    finally:
        os.chmod(victim, stat.S_IREAD | stat.S_IWRITE)

    assert ran
    assert report["src/a.py"].startswith("skipped:")
    assert target.read_bytes() == b"runner change\n"
    assert victim.read_bytes() == b"outside the repository\n"
    assert after == mode
    assert payload.exists()
    if outside == "read-only":
        assert "could not be held for its checks" in report["src/a.py"]
        assert os.path.samefile(_restore_tmp(target), victim)
    else:
        assert "was not a plain new file" in report["src/a.py"]
        assert not os.path.lexists(_restore_tmp(target))
        assert victim.stat().st_nlink == 1
    assert report["src/a.py"].endswith("check what is left there and delete it by hand")


def test_a_copy_a_writer_holds_open_is_refused(env, monkeypatch):
    """A handle already open on the copy to write refuses the bind (a
    sharing violation): nothing is put back and the copy is left as it is,
    named; what the writer writes stays in the copy alone."""
    calls = _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    snapshot, payload = _snapshot_of(env)
    target.write_bytes(b"runner change\n")
    writers: list = []

    def hold(name):
        writer = open(name, "r+b")
        writers.append(writer)
        if isinstance(calls, _HeldCopy):
            info = os.fstat(writer.fileno())
            calls.writers.add((info.st_dev, info.st_ino))

    _before_bind(monkeypatch, calls, hold)
    try:
        report = snapshot.restore()
        assert writers
        writers[0].write(b"the writer's bytes\n")
        writers[0].flush()
    finally:
        for writer in writers:
            writer.close()

    assert report["src/a.py"].startswith("skipped:")
    assert "could not be held for its checks" in report["src/a.py"]
    _kept_for_the_user(report["src/a.py"])
    assert target.read_bytes() == b"runner change\n"
    assert _restore_tmp(target).read_bytes().startswith(b"the writer's bytes\n")
    assert payload.exists()


@_WINDOWS_COPY_FILE2
def test_the_held_copy_cannot_be_written_renamed_or_removed_meanwhile(
    env, tmp_path, monkeypatch,
):
    """Windows' own sharing rules, which the held handle relies on: while it
    is held (here once the copy is hashed, as its attributes are about to be
    set), nobody can open the copy to write, rename it or remove it, and
    the file put back holds the payload's bytes."""
    calls = _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    original = target.read_bytes()
    snapshot, _payload = _snapshot_of(env)
    target.write_bytes(b"runner change\n")
    copy = _restore_tmp(target)
    attempts = (
        lambda: open(copy, "r+b").close(),
        lambda: os.rename(copy, tmp_path / "moved"),
        lambda: os.unlink(copy),
    )
    refused: list = []
    set_basic = calls.set_basic

    def try_then_set(held, *args):
        for attempt in attempts:
            try:
                attempt()
            except PermissionError:
                refused.append("refused")
            else:
                refused.append("allowed")
        return set_basic(held, *args)

    monkeypatch.setattr(calls, "set_basic", try_then_set)

    assert snapshot.restore(["src/a.py"]) == {"src/a.py": "restored"}
    assert refused == ["refused"] * 3
    assert target.read_bytes() == original
    assert not os.path.lexists(copy)


@pytest.mark.parametrize("case", [
    "restored", "bytes differ", "hard link swapped in", "held by a writer",
    "copy failed, something left",
])
def test_the_copy_is_never_changed_removed_or_renamed_by_its_name(
    env, tmp_path, monkeypatch, case,
):
    """Every change to the copy goes through the handle held on it: no
    ``chmod``, ``utime``, removal, truncation or rename by path names the
    temp file, whatever the restore finds (seamed: those calls record any
    path they are given, except inside ``CopyFile2`` itself)."""
    calls = _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    snapshot, _payload = _snapshot_of(env)
    target.write_bytes(b"runner change\n")
    copy = os.path.normcase(str(_restore_tmp(target)))
    victim = tmp_path / "outside.txt"
    victim.write_bytes(b"outside the repository\n")
    unlink, link = os.unlink, os.link
    writers: list = []

    def rewrite(name):
        with open(name, "r+b") as stream:
            stream.write(b"#")

    def swap(name):
        unlink(name)
        link(victim, name)

    def hold(name):
        writers.append(open(name, "r+b"))
        if isinstance(calls, _HeldCopy):
            info = os.fstat(writers[0].fileno())
            calls.writers.add((info.st_dev, info.st_ino))

    if case in ("bytes differ", "hard link swapped in", "held by a writer"):
        _before_bind(monkeypatch, calls, {
            "bytes differ": rewrite, "hard link swapped in": swap,
            "held by a writer": hold}[case])
    copying: list = []
    copy_file2 = write_safety._COPY_FILE2

    def copied(source, destination, flags):
        copying.append(True)
        try:
            if case == "copy failed, something left":
                Path(destination).write_bytes(b"part of a copy")
                raise _winerror(112, "there is not enough space on the disk", destination)
            return copy_file2(source, destination, flags)
        finally:
            copying.pop()

    monkeypatch.setattr(write_safety, "_COPY_FILE2", copied)
    named: list = []

    def recorder(name, real):
        def call(*args, **kwargs):
            for argument in args[:2]:
                if (not copying and isinstance(argument, (str, bytes, os.PathLike))
                        and os.path.normcase(os.fsdecode(argument)) == copy):
                    named.append(name)
            return real(*args, **kwargs)
        return call

    try:
        with monkeypatch.context() as patched:
            for module, names in (
                (os, ("chmod", "lchmod", "utime", "unlink", "remove", "rename",
                      "replace", "truncate", "rmdir")),
                (shutil, ("copystat", "copymode", "copy", "copy2", "copyfile",
                          "move", "rmtree")),
            ):
                for name in names:
                    if hasattr(module, name):
                        patched.setattr(module, name, recorder(name, getattr(module, name)))
            report = snapshot.restore()
    finally:
        for writer in writers:
            writer.close()

    assert named == [], named
    if case == "restored":
        assert report == {"src/a.py": "restored"}
    else:
        assert report["src/a.py"].startswith("skipped:"), report
        assert target.read_bytes() == b"runner change\n"
    assert victim.read_bytes() == b"outside the repository\n"


def test_the_held_restore_calls_nothing_that_changes_a_file_by_name():
    """Statically, for what a Windows restore runs once its payload is open
    (``CopyFile2``'s, and the one staging the bytes through a handle): no
    ``os`` or ``shutil`` call that changes, removes or renames a file by its
    path, and none of the helpers that do (the store's ``_discard`` and its
    read-only clearing, ``_copy_file2``'s clearing of the name first)."""
    import ast
    import inspect
    import textwrap

    by_path = {"chmod", "lchmod", "utime", "unlink", "remove", "rename", "replace",
               "truncate", "rmdir", "copystat", "copymode", "copy", "copy2",
               "copyfile", "move", "rmtree"}
    helpers = {"_discard", "_discard_after", "_unlink", "_clear_readonly",
               "_copy_file2", "_copy_times_and_mode"}
    for function in (write_safety._restored_by_copy_file2, write_safety._publish_held,
                     write_safety._publish_bound, write_safety._dispose_held,
                     write_safety._restored_through_handle):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        found = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if (isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name)
                    and called.value.id in ("os", "shutil") and called.attr in by_path):
                found.add(f"{called.value.id}.{called.attr}")
            elif isinstance(called, ast.Name) and called.id in helpers:
                found.add(called.id)
        assert not found, (function.__name__, found)


def test_a_payload_rewritten_after_its_check_is_never_put_back(env, monkeypatch, restoring):
    """The payload is hashed by path against the manifest, then copied. One
    rewritten in between is caught every way: the bytes read through the
    descriptor the restore holds on it must hash to the manifest's digest,
    so nothing is put back. Nothing is left beside the file either, except
    where the restore staged by descriptors and made its staged file by its
    name (macOS, a filesystem without ``O_TMPFILE``): a name it made is never
    removed, so that one is left there and named for the user, as every
    other refusal after the staged file was named leaves it
    (``tests/test_restore_staged_file.py``, ``_left_when_named``)."""
    target = env.root / "src" / "a.py"
    snapshot, payload = _snapshot_of(env)
    target.write_bytes(b"a = 2\n")
    real = write_safety.file_sha256
    fired: list = []

    def then_rewrite(path):
        digest = real(path)
        if not fired and os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                str(payload)):
            fired.append(True)
            payload.write_bytes(b"planted rollback\n")
        return digest

    monkeypatch.setattr(write_safety, "file_sha256", then_rewrite)
    report = snapshot.restore(["src/a.py"])

    assert fired
    state = report["src/a.py"]
    assert state.startswith("skipped:")
    assert "no longer matches the manifest" in state
    assert target.read_bytes() == b"a = 2\n"
    tmp = _restore_tmp(target)
    if (restoring == STAGED and write_safety._STAGES_BY_DESCRIPTOR
            and not _tmpfile_here(target.parent)):
        assert os.path.lexists(tmp), state
        assert f"was left at {tmp}" in state
        _kept_for_the_user(state)
    else:
        assert not os.path.lexists(tmp), state


@pytest.mark.skipif(os.name == "nt", reason="Windows will not move a file, or a "
                    "folder above it, while the file is held open")
@pytest.mark.parametrize("swap", ["folder redirected for the copy alone",
                                  "payload replaced by one with the same bytes"])
def test_a_restore_copy_of_another_file_is_refused(env, tmp_path, monkeypatch, swap):
    """What ``CopyFile2`` read by path is checked against the payload the
    descriptor holds: the payload's folder redirected for the copy alone (a
    junction's two flips) fails the bytes, the payload replaced by a file
    with the same bytes fails the identity. Either way nothing is put back
    and the temp file is gone."""
    _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    snapshot, payload = _snapshot_of(env)
    target.write_bytes(b"a = 2\n")
    folder = payload.parent
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    secret = b"bytes from outside, linked in for the copy alone\n"
    (elsewhere / payload.name).write_bytes(secret)

    def copy_another(source, destination, flags, copy) -> None:
        if swap.startswith("folder"):
            saved = folder.with_name("files.saved")
            folder.rename(saved)
            folder.symlink_to(elsewhere, target_is_directory=True)
            try:
                copy(source, destination, flags)
            finally:
                folder.unlink()
                saved.rename(folder)
            return
        twin = payload.with_name("twin")
        twin.write_bytes(payload.read_bytes())
        os.replace(twin, payload)
        copy(source, destination, flags)

    copy_file2, fired = _copying(payload, copy_another)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    report = snapshot.restore(["src/a.py"])

    assert fired
    assert report["src/a.py"].startswith("skipped:")
    assert target.read_bytes() == b"a = 2\n"
    assert not os.path.lexists(fired[0])


def test_a_link_at_the_restore_copys_name_once_copied_is_refused_and_named(
    env, tmp_path, monkeypatch,
):
    """A temp name that is a link once ``CopyFile2`` copied may have taken
    the copy elsewhere: the link is bound as itself and deleted through the
    handle, never followed, nothing is put back, and the file's state names
    where the copy may be, which the restore does not delete (a path someone
    else chose is not its to)."""
    _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    snapshot, payload = _snapshot_of(env)
    target.write_bytes(b"a = 2\n")
    followed = tmp_path / "followed.txt"

    def copy_through_a_link(source, destination, flags, copy) -> None:
        os.symlink(followed, destination)
        followed.write_bytes(Path(source).read_bytes())

    copy_file2, fired = _copying(payload, copy_through_a_link)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    report = snapshot.restore(["src/a.py"])

    assert fired
    assert "was a link to" in report["src/a.py"] and followed.name in report["src/a.py"]
    assert report["src/a.py"].endswith("check what is left there and delete it by hand")
    assert target.read_bytes() == b"a = 2\n"
    assert not os.path.lexists(fired[0])
    assert followed.exists()


@pytest.mark.parametrize("planted", ["to a file", "dangling"])
def test_a_link_planted_at_the_restore_copys_name_is_never_written_through(
    env, tmp_path, restoring, planted,
):
    """The temp name holds a token drawn for each attempt, pinned here as
    though a runner knew it. A link it planted there is never written
    through nor removed: the staged copy's exclusive creation (``O_EXCL``,
    the ``linkat`` naming an ``O_TMPFILE`` file, Windows' ``CREATE_NEW`` on
    the name itself) refuses the name, and the ``CopyFile2`` restore looks
    first and refuses it too, since ``CopyFile2`` follows a dangling link
    whatever its flags (the Windows CI, 2026-11-12; emulated so elsewhere),
    so the file is skipped, the link's target is neither written nor, for a
    dangling link, created, and the link stays, named for the user to look
    at."""
    target = env.root / "src" / "a.py"
    snapshot, _payload = _snapshot_of(env)
    target.write_bytes(b"a = 2\n")
    victim = tmp_path / "victim.txt"
    if planted == "to a file":
        victim.write_bytes(b"not the restore's to write\n")
    _restore_tmp(target).symlink_to(victim)

    report = snapshot.restore(["src/a.py"])

    if planted == "to a file":
        assert victim.read_bytes() == b"not the restore's to write\n"
    else:
        assert not os.path.lexists(victim)
    assert report["src/a.py"].startswith("skipped:")
    assert target.read_bytes() == b"a = 2\n"
    assert os.path.islink(_restore_tmp(target))
    assert report["src/a.py"].endswith(
        "was left as it is; check what is left there and delete it by hand")


@pytest.mark.parametrize("left", ["nothing", "part of a copy"])
@pytest.mark.parametrize("winerror", [5, 112], ids=["access denied", "disk full"])
def test_a_failed_restore_copy_takes_the_staged_restore_or_is_named(
    env, monkeypatch, winerror, left,
):
    """A ``CopyFile2`` that fails and leaves nothing at the temp name hands
    the file to the staged restore, which puts back the bytes it writes
    itself, with ``copystat``'s times. One that leaves something there skips
    the file and names it: what is there is never removed by its name, and
    may not be the restore's."""
    _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    os.utime(target, ns=(1_600_000_000_000_000_000, 1_650_000_000_987_654_300))
    original = target.read_bytes()
    snapshot, _payload = _snapshot_of(env)
    target.write_bytes(b"a = 2\n")

    def refuse(source, destination, flags):
        if left != "nothing":
            Path(destination).write_bytes(b"part of a copy")
        raise _winerror(winerror, "refused", source)

    monkeypatch.setattr(write_safety, "_COPY_FILE2", refuse)
    report = snapshot.restore(["src/a.py"])

    if left == "nothing":
        assert report == {"src/a.py": "restored"}
        assert target.read_bytes() == original
        assert target.stat().st_mtime_ns == 1_650_000_000_987_654_300
        assert not os.path.lexists(_restore_tmp(target))
    else:
        assert report["src/a.py"].startswith("skipped: CopyFile2 could not copy")
        _kept_for_the_user(report["src/a.py"])
        assert target.read_bytes() == b"a = 2\n"
        assert _restore_tmp(target).read_bytes() == b"part of a copy"


def test_restore_path_checks_its_copy_as_a_snapshot_restore_does(env, tmp_path, monkeypatch):
    """The commit write policy's restore (``restore_path``) has no digest to
    check against; its ``CopyFile2`` copy must hold the bytes read through
    the descriptor it holds of the copy it was handed."""
    _restoring_by_copy_file2(monkeypatch)
    target = env.root / "src" / "a.py"
    kept = tmp_path / "kept-a.py"
    kept.write_bytes(b"a = 0, as kept\n")

    assert write_safety.restore_path(str(env.root), "src/a.py", str(kept)) == "restored"
    assert target.read_bytes() == b"a = 0, as kept\n"

    target.write_bytes(b"a = 2\n")
    copy_file2, fired = _copying(kept, _copy_then_change)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    state = write_safety.restore_path(str(env.root), "src/a.py", str(kept))
    assert fired and state.startswith("skipped:")
    assert target.read_bytes() == b"a = 2\n"
    assert not os.path.lexists(fired[0])


def test_a_folder_that_became_a_junction_is_never_restored_through(env, monkeypatch):
    """``lstat`` reports a Windows directory junction as a directory, so a
    folder above the file that is a junction now (a reparse point naming
    another path) is refused like a symlink: neither the file put back nor
    the one deleted is reached through it. Seamed: the folder's ``lstat``
    carries a junction's reparse tag."""
    snapshot, _payload = _snapshot_of(env, "src/a.py", "src/new.py")
    target = env.root / "src" / "a.py"
    target.write_bytes(b"a = 2\n")
    (env.root / "src" / "new.py").write_text("new = 1\n", encoding="utf-8")
    real_lstat = os.lstat
    folder = os.path.normcase(str(env.root / "src"))

    def lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if os.path.normcase(os.fspath(path)) != folder:
            return info
        return types.SimpleNamespace(
            st_mode=info.st_mode, st_dev=info.st_dev, st_ino=info.st_ino,
            st_reparse_tag=_IO_REPARSE_TAG_MOUNT_POINT)

    with monkeypatch.context() as patched:
        patched.setattr(os, "lstat", lstat)
        report = snapshot.restore()

    for relative in ("src/a.py", "src/new.py"):
        assert "is a directory junction or another link now" in report[relative], report
    assert target.read_bytes() == b"a = 2\n"
    assert (env.root / "src" / "new.py").exists()


#: WinNT.h's tags (``stat`` defines them on Windows alone).
_IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003
_IO_REPARSE_TAG_SYMLINK = 0xA000000C


@pytest.mark.parametrize("tag, names_another", [
    (_IO_REPARSE_TAG_MOUNT_POINT, True),
    (_IO_REPARSE_TAG_SYMLINK, True),
    (0x9000001A, False),   # a OneDrive placeholder (IO_REPARSE_TAG_CLOUD_1)
    (0x80000013, False),   # a deduplicated file (IO_REPARSE_TAG_DEDUP)
    (0, False),
    (None, False),
], ids=["junction", "symlink", "cloud placeholder", "dedup", "none", "no field"])
def test_only_a_reparse_point_naming_another_path_counts_as_a_link(tag, names_another):
    info = types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o755)
    if tag is not None:
        info.st_reparse_tag = tag
    assert write_safety._names_another_path(info) is names_another


@pytest.mark.skipif(os.name != "nt" or not hasattr(write_safety._winapi, "CreateJunction"),
                    reason="directory junctions are Windows'")
def test_a_real_junction_above_the_file_is_refused(env, tmp_path):
    """A folder above the file swapped for a directory junction to a folder
    outside the repository: the restore refuses the path, and the file
    outside keeps its bytes."""
    snapshot, _payload = _snapshot_of(env)
    folder = env.root / "src"
    outside = tmp_path / "outside-src"
    outside.mkdir()
    (outside / "a.py").write_bytes(b"outside the repository\n")
    folder.rename(env.root / "src.saved")
    write_safety._winapi.CreateJunction(str(outside), str(folder))

    report = snapshot.restore(force=True)

    assert "is a directory junction or another link now" in report["src/a.py"], report
    assert (outside / "a.py").read_bytes() == b"outside the repository\n"


def _file_attributes(path) -> int:
    return os.stat(path).st_file_attributes


def _system32(name: str) -> str:
    """A Windows tool by its full path: the ``env`` fixture's PATH holds no
    system folder."""
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", name)


@_WINDOWS_COPY_FILE2
def test_a_restored_file_keeps_its_alternate_data_streams_and_attributes(env):
    """The snapshot's ``shutil.copy2`` (``CopyFile2``) keeps a file's NTFS
    alternate data streams and attributes in its payload, and the restore's
    ``CopyFile2`` copy puts them back on a file the runner replaced with a
    plain one; a copy of the payload's bytes alone dropped both. (A file
    that is not read-only: a read-only one takes the plain restore.)"""
    target = env.root / "src" / "a.py"
    try:
        with open(f"{target}:multi-llm-test", "wb") as stream:
            stream.write(b"kept in a stream\n")
    except OSError:
        pytest.skip("this file system has no alternate data streams")
    subprocess.run([_system32("attrib.exe"), "+h", str(target)], check=True,
                   capture_output=True)
    assert _file_attributes(target) & stat.FILE_ATTRIBUTE_HIDDEN
    original = target.read_bytes()
    snapshot, _payload = _snapshot_of(env)
    target.unlink()
    target.write_bytes(b"a = 2\n")  # the runner's file: no stream, not hidden

    assert snapshot.restore(["src/a.py"]) == {"src/a.py": "restored"}

    assert target.read_bytes() == original
    with open(f"{target}:multi-llm-test", "rb") as stream:
        assert stream.read() == b"kept in a stream\n"
    assert _file_attributes(target) & stat.FILE_ATTRIBUTE_HIDDEN


def _encrypted(path) -> bool:
    return bool(_file_attributes(path) & stat.FILE_ATTRIBUTE_ENCRYPTED)


@_WINDOWS_COPY_FILE2
def test_an_encrypted_file_comes_back_encrypted(env):
    """A file EFS-encrypted on its own, in a folder that is not encrypted,
    and replaced by the runner with a plaintext one comes back encrypted:
    the payload is encrypted (``CopyFile2``) and so is the restore's copy of
    it. Written as plain bytes, it came back as plaintext."""
    target = env.root / "src" / "a.py"
    try:
        subprocess.run([_system32("cipher.exe"), "/e", "/a", str(target)],
                       capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass
    if not _encrypted(target):
        pytest.skip("EFS is not available here: cipher /e did not encrypt the file")
    if _encrypted(target.parent):
        pytest.skip("the folder is encrypted, so every new file in it is too")
    original = target.read_bytes()
    snapshot, payload = _snapshot_of(env)
    if not _encrypted(payload):
        pytest.skip("the snapshot's copy is not encrypted here, so there is no "
                    "encryption to put back")
    target.unlink()
    target.write_bytes(b"a = 2\n")
    if _encrypted(target):
        pytest.skip("every new file is encrypted here")

    assert snapshot.restore(["src/a.py"]) == {"src/a.py": "restored"}

    assert target.read_bytes() == original
    assert _encrypted(target)


def _store(scheduler) -> Path:
    return Path(scheduler.safety_dir) / wbs.ROLLBACK_STORE_DIRNAME


def test_the_discarded_store_is_gone_whole_and_says_nothing(env, tmp_path):
    """The store is removed whole, a link planted in it as the link itself
    (its target keeps its bytes), and a store already gone says nothing."""
    lines: list = []
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
    assert scheduler._sweep.retain(scheduler._sweep.census())
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"not the discard's to remove\n")
    (_store(scheduler) / "planted").symlink_to(victim)

    scheduler._sweep.discard_store()
    scheduler._sweep.discard_store()

    assert not os.path.lexists(_store(scheduler))
    assert victim.read_bytes() == b"not the discard's to remove\n"
    assert not scheduler._sweep._retained and scheduler._sweep._retained_bytes == 0
    assert not [line for line in lines if "WARNING" in line], lines


def test_what_the_discard_cannot_remove_is_named_and_never_tried_again(env, monkeypatch):
    """A blob whose removal is refused (on Windows, a read-only one) is
    named in a warning, never left silently, and never tried again on any
    platform: no attribute is cleared and its name is not unlinked a second
    time. The folders left only because it is are not named again."""
    lines: list = []
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
    kept = scheduler._sweep.retain(scheduler._sweep.census())
    blob = scheduler._sweep.blob_path(kept["src/a.py"])
    tries: list = []
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if os.path.basename(os.fspath(path)) == kept["src/a.py"]:
            tries.append(path)
            raise PermissionError(errno.EACCES, "refused", os.fspath(path))
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(os, "unlink", unlink)
        patched.setattr(write_safety, "_clear_readonly",
                        lambda *args: pytest.fail("no attribute is cleared"))
        scheduler._sweep.discard_store()

    assert len(tries) == 1
    assert os.path.lexists(blob)
    warnings = [line for line in lines if "WARNING: the rollback store" in line]
    assert len(warnings) == 1, lines
    assert f"{blob} could not be removed (" in warnings[0]
    assert warnings[0].count(" could not be removed (") == 1
    assert "It was not tried again; left there it costs disk space only." in warnings[0]
    assert "delete it by hand" in warnings[0]
    assert not scheduler._sweep._retained and scheduler._sweep._retained_bytes == 0


@pytest.mark.skipif(
    os.name == "nt" or not shutil.rmtree.avoids_symlink_attacks
    or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="shutil.rmtree's descriptor walk, and a folder that refuses a "
           "non-root user's unlink")
def test_a_refused_blob_is_never_tried_again_through_a_folder_swapped_for_a_link(
    env, tmp_path, monkeypatch,
):
    """``shutil.rmtree`` removes each entry through its folder's descriptor,
    so once it opened ``rollback_store/ab/`` a link swapped in at that name
    leads nowhere. A refused removal is named, never tried again by path:
    the path would follow the link and remove the file of the same name
    outside the store. (The blob's removal is refused by its folder's mode,
    and the folder is swapped once ``rmtree`` holds it.)"""
    lines: list = []
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
    kept = scheduler._sweep.retain(scheduler._sweep.census())
    blob = Path(scheduler._sweep.blob_path(kept["src/a.py"]))
    folder, saved = blob.parent, blob.parent.with_name(blob.parent.name + ".saved")
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / blob.name
    victim.write_bytes(b"not the discard's to remove\n")
    real_unlink = os.unlink
    swapped: list = []

    def unlink(path, *args, **kwargs):
        if not swapped and os.path.basename(os.fspath(path)) == blob.name:
            assert kwargs.get("dir_fd") is not None, "rmtree's descriptor walk"
            swapped.append(True)
            folder.rename(saved)
            folder.symlink_to(outside, target_is_directory=True)
        return real_unlink(path, *args, **kwargs)

    os.chmod(folder, stat.S_IREAD | stat.S_IEXEC)
    try:
        with monkeypatch.context() as patched:
            patched.setattr(os, "unlink", unlink)
            scheduler._sweep.discard_store()
    finally:
        os.chmod(saved if saved.exists() else folder, stat.S_IRWXU)

    assert swapped
    assert victim.read_bytes() == b"not the discard's to remove\n"
    assert (saved / blob.name).exists()
    warnings = [line for line in lines if "WARNING: the rollback store" in line]
    assert len(warnings) == 1 and f"{blob} could not be removed (" in warnings[0], lines


@pytest.mark.skipif(os.name != "nt" or not hasattr(write_safety._winapi, "CreateJunction"),
                    reason="directory junctions are Windows'")
def test_a_store_folder_swapped_for_a_junction_clears_nothing_outside(
    env, tmp_path, monkeypatch,
):
    """The blob's folder swapped for a junction to an outside folder once
    ``rmtree`` listed it: ``rmtree``'s own removal reaches the read-only file
    of the same name there and is refused, and nothing tries it again, so
    its attribute and bytes stay; the blob is named."""
    lines: list = []
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
    kept = scheduler._sweep.retain(scheduler._sweep.census())
    blob = Path(scheduler._sweep.blob_path(kept["src/a.py"]))
    folder = blob.parent
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / blob.name
    victim.write_bytes(b"not the discard's to remove\n")
    os.chmod(victim, stat.S_IREAD)
    real_unlink = os.unlink
    swapped: list = []

    def unlink(path, *args, **kwargs):
        if not swapped and os.path.basename(os.fspath(path)) == blob.name:
            swapped.append(True)
            folder.rename(folder.with_name(folder.name + ".saved"))
            write_safety._winapi.CreateJunction(str(outside), str(folder))
        return real_unlink(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patched:
            patched.setattr(os, "unlink", unlink)
            scheduler._sweep.discard_store()

        assert swapped
        assert victim.read_bytes() == b"not the discard's to remove\n"
        assert _file_attributes(victim) & stat.FILE_ATTRIBUTE_READONLY
        warnings = [line for line in lines if "WARNING: the rollback store" in line]
        assert len(warnings) == 1 and f"{blob} could not be removed" in warnings[0], lines
    finally:
        os.chmod(victim, stat.S_IREAD | stat.S_IWRITE)


@_WINDOWS_COPY_FILE2
def test_a_read_only_blob_is_named_and_left_by_the_discard(env):
    """``CopyFile2`` keeps a read-only file's attribute in its blob, and
    Windows will not delete a read-only file: the discard names the blob,
    leaves it read-only in the store (it costs disk space only) and removes
    the rest."""
    lines: list = []
    target = env.root / "src" / "a.py"
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
    os.chmod(target, stat.S_IREAD)
    try:
        kept = scheduler._sweep.retain(scheduler._sweep.census())
        blob = scheduler._sweep.blob_path(kept["src/a.py"])
        assert _file_attributes(blob) & stat.FILE_ATTRIBUTE_READONLY

        scheduler._sweep.discard_store()

        assert os.path.lexists(blob)
        assert _file_attributes(blob) & stat.FILE_ATTRIBUTE_READONLY
        warnings = [line for line in lines if "WARNING: the rollback store" in line]
        assert len(warnings) == 1 and f"{blob} could not be removed (" in warnings[0], lines
        assert "costs disk space only" in warnings[0]
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        for left in _store(scheduler).rglob("*"):
            if left.is_file():
                os.chmod(left, stat.S_IREAD | stat.S_IWRITE)


# ---------------------------------------------------------------------------
# The pre-batch snapshot reads each file once, without following a link
# (fable review F9)
# ---------------------------------------------------------------------------

#: An outside file's bytes, which no rollback storage may ever hold.
OUTSIDE_SECRET = b"OUTSIDE SECRET\n"


def _payloads(env) -> list:
    return [path.read_bytes() for path in Path(env.phase_dir).rglob("*")
            if path.is_file() and write_safety.PAYLOAD_DIRNAME in path.parts]


@pytest.mark.skipif(not write_safety.OPENS_WITHOUT_FOLLOWING,
                    reason="Windows opens by path, then checks the name (lstat)")
@pytest.mark.parametrize("swapped", ["the file", "its folder"])
def test_a_declared_file_swapped_for_a_link_is_never_snapshotted(
        env, tmp_path, monkeypatch, swapped):
    """Fable review F9: a declared file, or its folder, swapped for a link
    to an outside file between ``take``'s containment check and its read (a
    sibling batch still running) is refused, nothing dispatched, and no
    payload holds the outside bytes. Before, ``copy2`` followed the link."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_bytes(OUTSIDE_SECRET)
    target = env.root / "src" / "a.py"
    real = write_safety.contain_under_root

    def racing(*args, **kwargs):
        resolved = real(*args, **kwargs)
        if swapped == "the file":
            target.unlink()
            target.symlink_to(outside / "a.py")
        else:
            target.parent.rename(target.parent.with_name("src.moved"))
            target.parent.symlink_to(outside)
        return resolved

    monkeypatch.setattr(write_safety, "contain_under_root", racing)
    with pytest.raises(write_safety.WriteSafetyError, match="without following a link"):
        write_safety.TaskSnapshot.take(
            phase_dir=str(env.phase_dir), task_id="T-S", files=["src/a.py"],
            root=str(env.root))

    assert OUTSIDE_SECRET not in _payloads(env)


def test_a_snapshot_holds_the_bytes_it_hashed(env):
    """Each payload is the bytes read once and hashed, with the file's mode
    and times; an absent file and a directory are recorded absent."""
    target = env.root / "src" / "a.py"
    target.chmod(0o640)
    os.utime(target, ns=(1_600_000_000_000_000_000, 1_650_000_000_000_000_000))
    (env.root / "src" / "dir").mkdir()
    snapshot = write_safety.TaskSnapshot.take(
        phase_dir=str(env.phase_dir), task_id="T-H", root=str(env.root),
        files=["src/a.py", "src/new.py", "src/dir"])

    entry = snapshot.entry_for("src/a.py")
    payload = Path(snapshot.directory) / write_safety.PAYLOAD_DIRNAME / entry.payload
    assert entry.existed and entry.size == len(target.read_bytes())
    assert entry.sha256 == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert payload.read_bytes() == target.read_bytes()
    # The file's mode as the platform keeps it: Windows keeps only the
    # read-only bit, so 0o640 reads back there as 0o666.
    assert stat.S_IMODE(payload.stat().st_mode) == stat.S_IMODE(target.stat().st_mode)
    if os.name != "nt":
        assert stat.S_IMODE(payload.stat().st_mode) == 0o640
    assert payload.stat().st_mtime_ns == 1_650_000_000_000_000_000
    for absent in ("src/new.py", "src/dir"):
        assert not snapshot.entry_for(absent).existed
        assert snapshot.entry_for(absent).payload is None


def test_adopt_copies_the_bytes_it_hashed_and_never_through_a_link(env, tmp_path, monkeypatch):
    """Fable review F9: ``adopt`` reads the retained blob once and copies only
    bytes that hash to its digest. A blob swapped for a link to an outside
    file once its hash was checked (here, where the old code checked it) is
    never copied; a blob that is a link is no rollback point."""
    snapshot = write_safety.TaskSnapshot.take(
        phase_dir=str(env.phase_dir), task_id="T-A", files=["src/a.py"],
        root=str(env.root))
    blob = tmp_path / "blob"
    blob.write_bytes(b"retained\n")
    digest = hashlib.sha256(b"retained\n").hexdigest()
    secret = tmp_path / "secret"
    secret.write_bytes(OUTSIDE_SECRET)
    real_hash = write_safety.file_sha256

    def hash_then_swap(path):
        answer = real_hash(path)
        if os.fsdecode(path) == str(blob):
            blob.unlink()
            blob.symlink_to(secret)
        return answer

    monkeypatch.setattr(write_safety, "file_sha256", hash_then_swap)
    entry = snapshot.adopt("src/b.py", existed=True, sha256=digest, size=9,
                           payload_source=str(blob))

    assert OUTSIDE_SECRET not in _payloads(env)
    if entry is not None:
        payload = Path(snapshot.directory) / write_safety.PAYLOAD_DIRNAME / entry.payload
        assert payload.read_bytes() == b"retained\n"
    linked = tmp_path / "linked-blob"
    linked.symlink_to(blob)
    assert snapshot.adopt("src/c.py", existed=True,
                          sha256=hashlib.sha256(linked.read_bytes()).hexdigest(),
                          size=9, payload_source=str(linked)) is None
    assert snapshot.adopt("src/d.py", existed=True, sha256=None, size=9,
                          payload_source=str(tmp_path / "blob")) is None


@pytest.mark.parametrize("copying", ["CopyFile2", "plain copy"])
def test_adopt_by_path_refuses_a_blob_name_that_is_a_link(env, tmp_path, monkeypatch, copying):
    """Codex review of F01, point 2: where a link cannot be refused at the
    open (Windows: no ``O_NOFOLLOW``, emulated here, ``CopyFile2`` or the
    plain copy), ``adopt`` opens the blob by path and its name must be the
    file opened (``lstat`` against ``fstat``): a blob name that is a link
    is no rollback point, before anything is copied, though its target's
    bytes hash right. A blob that is a plain file is adopted as before."""
    snapshot = write_safety.TaskSnapshot.take(
        phase_dir=str(env.phase_dir), task_id="T-W", files=["src/a.py"],
        root=str(env.root))
    blob = tmp_path / "blob"
    blob.write_bytes(b"retained\n")
    linked = tmp_path / "linked-blob"
    linked.symlink_to(blob)
    digest = hashlib.sha256(b"retained\n").hexdigest()
    monkeypatch.setattr(write_safety, "OPENS_WITHOUT_FOLLOWING", False, raising=False)
    if copying == "plain copy":
        monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
    elif getattr(write_safety, "_COPY_FILE2", None) is None:
        monkeypatch.setattr(write_safety, "_COPY_FILE2", _copy_file2_off_windows,
                            raising=False)
    payloads = Path(snapshot.directory) / write_safety.PAYLOAD_DIRNAME
    before = sorted(payloads.iterdir())

    with monkeypatch.context() as windows:
        if hasattr(os, "O_NOFOLLOW"):
            windows.delattr(os, "O_NOFOLLOW")
        refused = snapshot.adopt("src/b.py", existed=True, sha256=digest, size=9,
                                 payload_source=str(linked), save=False)
        adopted = snapshot.adopt("src/c.py", existed=True, sha256=digest, size=9,
                                 payload_source=str(blob), save=False)

    assert refused is None
    assert adopted is not None
    assert sorted(payloads.iterdir()) == sorted(before + [payloads / adopted.payload])
    assert (payloads / adopted.payload).read_bytes() == b"retained\n"
