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
  not, and a stray write beside an unchanged plan is named with what moved.

No model runs; the scheduler is the real one over a real git repository.
"""

from __future__ import annotations

import os
import pathlib
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
    copy = wbs.shutil.copy2
    copied = []

    def copy_then_swap(source, destination, *args, **kwargs):
        result = copy(source, destination, *args, **kwargs)
        name = Path(source).name
        copied.append(name)
        if name == "a.py":
            env.phase_dir.rename(saved)
            env.phase_dir.symlink_to(forged, target_is_directory=True)
        elif name == "b.py" and undone:
            env.phase_dir.unlink()
            saved.rename(env.phase_dir)
        return result

    monkeypatch.setattr(wbs.shutil, "copy2", copy_then_swap)
    with pytest.raises(write_safety.WriteSafetyError, match="trusted phase identity"):
        scheduler._retain(census)
    assert copied == ["p.md", "a.py"]
    assert not any(forged.iterdir())
