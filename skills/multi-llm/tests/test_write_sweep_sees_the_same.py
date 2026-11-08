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
  always holds the bytes its name is the digest of.

No model runs; the scheduler is the real one over a real git repository.
"""

from __future__ import annotations

import errno
import hashlib
import os
import pathlib
import shutil
import stat
import threading
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

    census = scheduler._census()

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
    snapshot = scheduler._snapshot(task)
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
    snapshots = {"T-0": scheduler._snapshot(tasks[0])}

    # Declared change: applied, nothing to name.
    scheduler._begin_sweep("T-0", ["plans/p.md"])
    env.plan.write_text("# Plan\n\n## Step 1\nedited\n", encoding="utf-8")
    why, context = _applied(scheduler, tasks[0], snapshots["T-0"])
    assert why is None and "moved_outside" not in context
    scheduler._undeclared_writes("T-0", snapshots["T-0"])

    # Nothing moved at all: named.
    snapshots["T-1"] = scheduler._snapshot(tasks[1])
    scheduler._begin_sweep("T-1", ["plans/p.md"])
    why, context = _applied(scheduler, tasks[1], snapshots["T-1"],
                            files_touched=["plans/p.md"])
    assert why is not None and "no file changed" in why
    assert "named plans/p.md as edited" in why
    scheduler._undeclared_writes("T-1", snapshots["T-1"])

    # A plan batch whose plan did not move but whose runner wrote elsewhere:
    # named, with the stray paths it moved, deletions included.
    snapshots["T-2"] = scheduler._snapshot(tasks[2])
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
    snapshot = scheduler._snapshot(task)
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
    census = scheduler._census()
    assert "src/deep/er/d.py" in census
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "d.py").write_text("bytes from outside the repository\n", encoding="utf-8")
    (env.root / "src" / "deep" / "er" / "d.py").unlink()
    (env.root / "src" / "deep" / "er").rmdir()
    (env.root / "src" / "deep" / "er").symlink_to(elsewhere, target_is_directory=True)

    kept = scheduler._retain(census)

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
    census = scheduler._census()
    outside = tmp_path / "outside" / "secret.txt"
    outside.write_text("bytes the deny list keeps out\n", encoding="utf-8")
    target = env.root / "src" / "b.py"

    def swap() -> None:
        target.unlink()
        target.symlink_to(outside)

    fired = _after_first_resolve(monkeypatch, target, swap)
    kept = scheduler._retain(census)

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
    census = scheduler._census()
    saved = env.phase_dir.with_name("saved-phase")
    forged = tmp_path / "forged"
    forged.mkdir()
    store = scheduler._store_blob
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

    monkeypatch.setattr(scheduler, "_store_blob", store_then_swap)
    with pytest.raises(write_safety.WriteSafetyError, match="trusted phase identity"):
        scheduler._retain(census)
    assert copied == ["p.md", "a.py"]
    assert not any(forged.iterdir())


def _copy_file2_off_windows(source, destination, flags):
    """``CopyFile2`` as Windows runs it, for the platforms without it: a link
    at the source's own name copied as a link (``COPY_FILE_COPY_SYMLINK``),
    any other source read by path through any link above it, the destination
    created new, failing on any name there, a dangling link included
    (``COPY_FILE_FAIL_IF_EXISTS`` with ``COPY_FILE_COPY_SYMLINK``, as
    ``CopyFileEx`` documents them), then ``copy2``'s metadata, taken after
    the read."""
    assert flags & 0x809 == 0x809, (
        "shutil.copy2's flag, an exclusive creation, and no link followed")
    if os.path.islink(source):
        os.symlink(os.readlink(source), destination)
        return
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
    census = scheduler._census()
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
    kept = scheduler._retain(census)

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
    census = scheduler._census()
    target = env.root / "src" / "a.py"
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    secret = tmp_path / "outside" / "between.txt"
    secret.write_bytes(b"bytes linked in between the hash and the copy\n")
    blob_path = scheduler._blob_path
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

    monkeypatch.setattr(scheduler, "_blob_path", name_the_blob)
    kept = scheduler._retain(census)

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
    census = scheduler._census()
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
    kept = scheduler._retain(census)

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
    census = scheduler._census()
    target = env.root / "src" / "b.py"
    target.unlink()
    os.mkfifo(target)

    kept = scheduler._retain(census)

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

    kept = scheduler._retain(scheduler._census())

    source = os.stat(target)
    blob = os.stat(scheduler._blob_path(kept["src/a.py"]))
    assert stat.S_IMODE(blob.st_mode) == stat.S_IMODE(source.st_mode)
    assert blob.st_mtime_ns == source.st_mtime_ns
    if not _copies_by_path(opening):
        # CopyFile2 sets the copy's access time itself, and reading the copy
        # back to check its bytes can move it.
        assert blob.st_atime_ns == source.st_atime_ns
    if xattr is not None:
        assert os.getxattr(scheduler._blob_path(kept["src/a.py"]),
                           "user.multi-llm-test") == xattr


@pytest.mark.parametrize("planted", ["to a file", "dangling"])
def test_a_link_planted_at_the_pending_name_is_replaced_never_written_through(
    env, tmp_path, opening, planted,
):
    """The pending copy's name is predictable (digest, pid, thread), and a
    runner that can write the phase dir can plant a link there before the
    pass runs. The copy is created exclusively and never through a link
    (the name cleared first, then ``COPY_FILE_FAIL_IF_EXISTS`` where
    ``CopyFile2`` copies), so the link's target is neither written, re-moded
    nor, for a dangling link, created."""
    scheduler = env.scheduler([env.apply_task(0)])
    census = scheduler._census()
    source = env.root / "src" / "a.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    blob = Path(scheduler._blob_path(digest))
    pending = Path(f"{blob}.{os.getpid()}.{threading.get_ident()}.tmp")
    pending.parent.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    if planted == "to a file":
        victim.write_bytes(b"not the executor's to write\n")
        os.chmod(victim, 0o600)
        mode = stat.S_IMODE(victim.stat().st_mode)
    pending.symlink_to(victim)

    kept = scheduler._retain(census)

    assert kept["src/a.py"] == digest
    if planted == "to a file":
        assert victim.read_bytes() == b"not the executor's to write\n"
        assert stat.S_IMODE(victim.stat().st_mode) == mode
    else:
        assert not os.path.lexists(victim)
    assert not blob.is_symlink() and blob.read_bytes() == source.read_bytes()
    assert not os.path.lexists(pending)


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
    census = scheduler._census()
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
    kept = scheduler._retain(census)

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
    census = scheduler._census()

    def copy_changed_read_only(source, destination, flags, copy) -> None:
        _copy_then_change(source, destination, flags, copy)
        os.chmod(destination, stat.S_IREAD)
        assert os.stat(destination).st_file_attributes & stat.FILE_ATTRIBUTE_READONLY

    copy_file2, fired = _copying(env.root / "src" / "a.py", copy_changed_read_only)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    kept = scheduler._retain(census)

    assert fired
    assert "src/a.py" not in kept
    assert not os.path.lexists(fired[0])
    assert not [blob for blob in _blobs(scheduler) if blob.name.endswith(".tmp")]
    assert not [line for line in lines if "WARNING" in line], lines


@pytest.mark.parametrize("failure", ["copied by path, then refused",
                                     "the copy failed midway"])
def test_a_refused_copy_that_cannot_be_removed_is_reported(
    env, monkeypatch, failure,
):
    """A refused copy can hold another file's bytes, so one that cannot be
    removed is not dropped like an ordinary failed copy: the file has no
    rollback point, a warning names what is left, the retention warning
    gives it as the reason, and its bytes count against the store budget."""
    target = env.root / "src" / "a.py"
    if failure.startswith("copied by path"):
        _by_path_with_copy_file2(monkeypatch)
        copy_file2, fired = _copying(target, _copy_then_change)
        monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    else:
        monkeypatch.setattr(write_safety, "_COPY_FILE2", None, raising=False)
        copy_xattrs = write_safety._copy_xattrs
        wanted = os.stat(target)
        fired = []

        def fail_midway(source, copy) -> None:
            opened = os.fstat(source)
            if fired or (opened.st_dev, opened.st_ino) != (wanted.st_dev, wanted.st_ino):
                return copy_xattrs(source, copy)
            fired.append(True)
            raise OSError(errno.EIO, "failed midway")

        monkeypatch.setattr(write_safety, "_copy_xattrs", fail_midway)
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if os.fspath(path).endswith(".tmp"):
            raise PermissionError(errno.EACCES, "refused", os.fspath(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    lines: list = []
    scheduler = env.scheduler([env.apply_task(0)], log=lines.append)

    kept = scheduler._retain(scheduler._census())

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
    assert scheduler._retained_bytes == (
        sum(blob.stat().st_size for blob in blobs) + left[0].stat().st_size)


def test_a_link_at_the_pending_name_once_copied_is_refused_and_named(
    env, tmp_path, monkeypatch,
):
    """Whatever ``CopyFile2`` makes of a link put at the pending name after
    it was cleared, a pending name that is a link once the copy was made is
    refused: the link is removed, never followed, nothing is retained, and
    the warning names where the copy may have been written, which the pass
    does not remove (a path someone else chose is not its to delete)."""
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

    kept = scheduler._retain(scheduler._census())

    assert fired
    assert "src/a.py" not in kept
    assert not os.path.lexists(fired[0])
    assert followed.exists()
    warnings = [line for line in lines if "WARNING: src/a.py has no rollback point" in line]
    assert len(warnings) == 1 and followed.name in warnings[0], lines
    assert "was a link to" in warnings[0]


@_WINDOWS_COPY_FILE2
def test_a_dangling_link_planted_before_the_copy_is_not_followed(
    env, tmp_path, monkeypatch,
):
    """The pending name is cleared, then ``CopyFile2`` creates it. A dangling
    link planted in between: ``COPY_FILE_FAIL_IF_EXISTS`` alone lets
    ``CopyFileEx`` follow it and create its target; with
    ``COPY_FILE_COPY_SYMLINK`` it is documented to fail on any link there.
    The link's target is never created, and the store holds only blobs that
    hold the bytes their names are the digest of."""
    absent = tmp_path / "never-created.txt"

    def plant_then_copy(source, destination, flags, copy) -> None:
        os.symlink(absent, destination)
        copy(source, destination, flags)

    copy_file2, fired = _copying(env.root / "src" / "a.py", plant_then_copy)
    monkeypatch.setattr(write_safety, "_COPY_FILE2", copy_file2)
    scheduler = env.scheduler([env.apply_task(0)])

    kept = scheduler._retain(scheduler._census())

    assert fired
    assert not os.path.lexists(absent)
    if "src/a.py" in kept:
        blob = Path(scheduler._blob_path(kept["src/a.py"]))
        assert blob.read_bytes() == (env.root / "src" / "a.py").read_bytes()
    for blob in _blobs(scheduler):
        if not blob.name.endswith(".tmp"):
            assert hashlib.sha256(blob.read_bytes()).hexdigest() == blob.name, blob


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

    kept = scheduler._retain(scheduler._census())

    if winerror == 5:
        blob = Path(scheduler._blob_path(kept["src/a.py"]))
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
