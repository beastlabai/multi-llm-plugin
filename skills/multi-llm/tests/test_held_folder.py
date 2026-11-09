"""``utils/held_folder.py``: the plan directories a run writes below, and the
writes made through a folder held from one
(docs/development/write-safety.md#plan-output-containment).

Each writer's own swap test is in ``tests/test_plan_dir_writers_held.py``;
here are the root (anchored once), the refusals, the fallbacks, and that a
normal write is byte for byte, mode for mode and flush for flush the write
each writer made before.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import atomic_io, failure_capture, held_folder, output_handler  # noqa: E402
from utils import state_file, status_file, validation, write_safety  # noqa: E402
from utils.held_folder import (  # noqa: E402
    OutsidePlanDir,
    anchor_plan_dir,
    plan_dir_of,
    remove_in_plan_dir,
    write_in_plan_dir,
)

POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks, modes and dir_fd")
HELD = pytest.mark.skipif(not held_folder._WRITES_HELD,
                          reason="writes through a held folder (Linux, macOS)")

VICTIM = b"outside the plan's directory\n"


@pytest.fixture
def plan(tmp_path):
    """``repo/plans/feature.md``, its run started (outputs located); the
    output folder ``feature/`` and its ``implement/`` phase folder exist."""
    plans = tmp_path / "repo" / "plans"
    plans.mkdir(parents=True)
    plan = plans / "feature.md"
    plan.write_text("# Feature\n", encoding="utf-8")
    output_handler.phase_dir_path(plan, "implement").mkdir(parents=True)
    return plan


def _outside(tmp_path: Path, *names: str) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    for name in names:
        (outside / name).write_bytes(VICTIM)
    return outside


def _unchanged(outside: Path, *names: str) -> None:
    assert sorted(p.name for p in outside.iterdir()) == sorted(names)
    for name in names:
        assert (outside / name).read_bytes() == VICTIM


def _swap(folder: Path, to: Path) -> None:
    folder.rename(folder.with_name(folder.name + ".moved"))
    os.symlink(to, folder)


# ---------------------------------------------------------------------------
# The root: the plan's directory, resolved once
# ---------------------------------------------------------------------------

class TestTheRoot:
    def test_locating_the_outputs_anchors_the_plans_directory_resolved(self, plan):
        assert plan_dir_of(plan.parent / "feature" / "state.json") == os.path.realpath(plan.parent)
        assert plan_dir_of(plan.parent.parent / "elsewhere.json") is None

    @POSIX
    def test_it_is_resolved_once_and_both_spellings_find_it(self, tmp_path):
        real = tmp_path / "real" / "plans"
        real.mkdir(parents=True)
        via = tmp_path / "via"
        os.symlink(tmp_path / "real", via)
        assert anchor_plan_dir(via / "plans") == str(real)
        os.unlink(via)
        os.symlink(tmp_path / "other", via)
        assert anchor_plan_dir(via / "plans") == str(real)
        assert plan_dir_of(via / "plans" / "feature" / "state.json") == str(real)
        assert plan_dir_of(real / "feature" / "state.json") == str(real)

    def test_the_deepest_anchored_directory_wins(self, tmp_path):
        anchor_plan_dir(tmp_path)
        anchor_plan_dir(tmp_path / "plans")
        assert plan_dir_of(tmp_path / "plans" / "x" / "s.json") == os.path.realpath(tmp_path / "plans")
        assert plan_dir_of(tmp_path / "x" / "s.json") == os.path.realpath(tmp_path)

    @POSIX
    @HELD
    def test_a_link_above_the_plans_directory_repointed_during_the_run_is_refused(
            self, tmp_path):
        for side in ("real", "other"):
            (tmp_path / side / "plans" / "feature").mkdir(parents=True)
        via = tmp_path / "via"
        os.symlink(tmp_path / "real", via)
        anchor_plan_dir(via / "plans")
        os.unlink(via)
        os.symlink(tmp_path / "other", via)
        with pytest.raises(OutsidePlanDir):
            write_in_plan_dir(via / "plans" / "feature" / "state.json", b"{}")
        assert list((tmp_path / "other" / "plans" / "feature").iterdir()) == []

    def test_the_restore_and_the_plan_writers_share_one_walk(self):
        assert write_safety._HeldFolder is held_folder._HeldFolder
        assert write_safety._open_below_root is held_folder._open_below_root


# ---------------------------------------------------------------------------
# The write
# ---------------------------------------------------------------------------

@POSIX
class TestTheWrite:
    @HELD
    def test_it_goes_through_the_folder_held(self, plan, monkeypatch):
        opened = []
        real_open = held_folder._HeldFolder.open

        def spy(*args, **kwargs):
            folder = real_open(*args, **kwargs)
            opened.append(folder.parts)
            return folder

        monkeypatch.setattr(held_folder._HeldFolder, "open", spy)
        target = plan.parent / "feature" / "implement" / ".status.json"
        write_in_plan_dir(target, b"{}")
        assert opened == [("feature", "implement")]
        assert target.read_bytes() == b"{}"

    def test_a_folder_linked_out_of_the_plans_directory_is_refused(self, plan, tmp_path):
        outside = _outside(tmp_path, "state.json")
        _swap(plan.parent / "feature", outside)
        with pytest.raises(OutsidePlanDir, match="outside the plan's directory"):
            write_in_plan_dir(plan.parent / "feature" / "state.json", b"{}", make=True)
        _unchanged(outside, "state.json")

    def test_a_folder_linked_inside_the_plans_directory_is_written_where_it_leads(
            self, plan):
        elsewhere = plan.parent / "archive"
        elsewhere.mkdir()
        _swap(plan.parent / "feature", elsewhere)
        write_in_plan_dir(plan.parent / "feature" / "state.json", b"{}")
        assert (elsewhere / "state.json").read_bytes() == b"{}"
        assert os.path.islink(plan.parent / "feature")

    def test_a_link_at_the_files_own_name_is_replaced_never_followed(self, plan, tmp_path):
        outside = _outside(tmp_path, "state.json")
        target = plan.parent / "feature" / "state.json"
        os.symlink(outside / "state.json", target)
        write_in_plan_dir(target, b"{}")
        assert not os.path.islink(target) and target.read_bytes() == b"{}"
        _unchanged(outside, "state.json")

    @HELD
    def test_a_folder_swapped_after_the_walk_is_never_renamed_into(
            self, plan, tmp_path, monkeypatch):
        outside = _outside(tmp_path, ".status.json")
        phase = plan.parent / "feature" / "implement"
        real_write = held_folder.atomic_write_bytes

        def swap_then_write(*args, **kwargs):
            _swap(phase, outside)
            return real_write(*args, **kwargs)

        monkeypatch.setattr(held_folder, "atomic_write_bytes", swap_then_write)
        with pytest.raises(OSError, match="no longer the folder"):
            write_in_plan_dir(phase / ".status.json", b"{}")
        _unchanged(outside, ".status.json")
        assert list(phase.with_name("implement.moved").iterdir()) == []

    @HELD
    def test_a_folder_moved_once_the_file_is_in_says_where_the_file_is(
            self, plan, tmp_path, monkeypatch):
        phase = plan.parent / "feature" / "implement"
        real_write = held_folder.atomic_write_bytes

        def write_then_move(*args, **kwargs):
            real_write(*args, **kwargs)
            phase.rename(phase.with_name("implement.moved"))
            phase.mkdir()

        monkeypatch.setattr(held_folder, "atomic_write_bytes", write_then_move)
        with pytest.raises(OSError, match="implement.moved"):
            write_in_plan_dir(phase / ".status.json", b"{}")
        assert list(phase.iterdir()) == []

    @HELD
    def test_a_missing_folder_is_made_never_through_a_link_planted_on_the_way(
            self, plan, tmp_path, monkeypatch):
        outside = _outside(tmp_path)
        missing = plan.parent / "feature" / "review-plan"
        real_realpath = os.path.realpath

        def realpath_then_plant(path, *args, **kwargs):
            resolved = real_realpath(path, *args, **kwargs)
            if not os.path.lexists(missing):
                os.symlink(outside, missing)
            return resolved

        monkeypatch.setattr(held_folder.os.path, "realpath", realpath_then_plant)
        with pytest.raises(OSError, match="without following a link"):
            write_in_plan_dir(missing / "r.json", b"[]", make=True)
        _unchanged(outside)

    def test_a_file_below_no_anchored_directory_is_written_by_its_path(self, tmp_path):
        """A tool or a test handed a folder directly: the writer's own write,
        a link at the folder followed as it always was."""
        outside = _outside(tmp_path)
        folder = tmp_path / "loose"
        os.symlink(outside, folder)
        write_in_plan_dir(folder / "x.json", b"{}")
        assert (outside / "x.json").read_bytes() == b"{}"

    def test_windows_writes_by_the_checked_path(self, plan, tmp_path, monkeypatch):
        monkeypatch.setattr(held_folder, "_WRITES_HELD", False)
        target = plan.parent / "feature" / "implement" / ".status.json"
        write_in_plan_dir(target, b"{}")
        assert target.read_bytes() == b"{}"
        outside = _outside(tmp_path, ".status.json")
        _swap(plan.parent / "feature" / "implement", outside)
        with pytest.raises(OutsidePlanDir):
            write_in_plan_dir(target, b"[]")
        _unchanged(outside, ".status.json")


@POSIX
class TestTheRemoval:
    def test_a_name_is_removed_in_its_folder_held(self, plan):
        target = plan.parent / "feature" / "state.json"
        target.write_bytes(b"{}")
        remove_in_plan_dir(target)
        remove_in_plan_dir(target)  # already gone: nothing to remove
        assert not target.exists()

    def test_a_name_behind_a_folder_linked_out_is_never_removed(self, plan, tmp_path):
        outside = _outside(tmp_path, "state.json")
        _swap(plan.parent / "feature", outside)
        with pytest.raises(OutsidePlanDir):
            remove_in_plan_dir(plan.parent / "feature" / "state.json")
        _unchanged(outside, "state.json")


# ---------------------------------------------------------------------------
# Normal runs: the same bytes, mode and flushes as each writer's own write
# ---------------------------------------------------------------------------

def _fsyncs(monkeypatch):
    calls = []
    real = os.fsync

    def recording(descriptor):
        calls.append(descriptor)
        return real(descriptor)

    monkeypatch.setattr(os, "fsync", recording)
    return calls


def _state(path: Path, document):
    state_file.write_document(path, document)


def _state_before(path: Path, document):
    atomic_io.atomic_write_text(path, json.dumps(document, indent=2), mode=0o600,
                                fsync=False, temp_prefix=state_file.STATE_TEMP_PREFIX)


def _status(path: Path, document):
    status_file.write_status(str(path.parent), dict(document))


def _status_before(path: Path, document):
    atomic_io.atomic_write_text(path, json.dumps({**document, "updated_at": "T"}, indent=2),
                                mode=0o600, fsync=False,
                                temp_prefix=status_file.STATUS_TEMP_PREFIX)


def _sidecar(path: Path, document):
    validation._write_one_sidecar(str(path), document["prompt"])


def _sidecar_before(path: Path, document):
    atomic_io.atomic_write_text(path, document["prompt"])


def _record(path: Path, document):
    failure_capture._atomic_write_json(str(path), document)


def _record_before(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_io.atomic_write_text(path, json.dumps(document, indent=2), mode=0o600)


def _restore(path: Path, document):
    write_in_plan_dir(path, json.dumps(document).encode("utf-8"))


def _restore_before(path: Path, document):
    atomic_io.atomic_write_bytes(path, json.dumps(document).encode("utf-8"))


WRITERS = {
    "state.json": (_state, _state_before, "state.json"),
    ".status.json": (_status, _status_before, "implement/.status.json"),
    "prompt sidecar": (_sidecar, _sidecar_before, "implement/validation_batch_0.prompt.txt"),
    "executor record": (_record, _record_before, "review-plan/validation_batch_0.json"),
    "restore's state.json": (_restore, _restore_before, "state.json"),
}


@POSIX
@pytest.mark.parametrize("existing", [None, 0o640])
@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_normal_write_is_the_writers_own_write(tmp_path, monkeypatch, writer, existing):
    write, before, relative = WRITERS[writer]
    monkeypatch.setattr(status_file, "now_iso", lambda: "T")
    document = {"prompt": "line one\nline two\n", "é": [1, 2]}
    previous = os.umask(0o022)
    try:
        results = []
        for side, writes in (("held", write), ("before", before)):
            plans = tmp_path / side / "plans"
            (plans / "feature" / "implement").mkdir(parents=True)
            plan = plans / "feature.md"
            plan.write_text("# Feature\n", encoding="utf-8")
            if side == "held":
                output_handler.output_dir_path(plan)
            target = plans / "feature" / relative
            if existing is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"old")
                os.chmod(target, existing)
            held = []
            real_open = held_folder._HeldFolder.open
            with monkeypatch.context() as patched:
                calls = _fsyncs(patched)
                patched.setattr(held_folder._HeldFolder, "open", classmethod(
                    lambda cls, *a, **k: held.append(a) or real_open(*a, **k)))
                writes(target, document)
            assert bool(held) is (side == "held" and held_folder._WRITES_HELD)
            info = os.lstat(target)
            # The checkpoint's lock file is the checkpoint lock's, not the write's.
            names = sorted(p.name for p in target.parent.iterdir()
                           if not p.name.endswith(".lock"))
            results.append((target.read_bytes(), stat.S_IMODE(info.st_mode), len(calls),
                            names))
    finally:
        os.umask(previous)
    assert results[0] == results[1]
