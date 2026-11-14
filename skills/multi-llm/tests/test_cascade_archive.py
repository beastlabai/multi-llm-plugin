"""A cascade stage's close-out: the archive, its recovery, its locks and its counts.

``utils/cascade_archive.py`` copies a finished stage's live review and apply
dirs into ``stage-<k>/artifacts/`` with the stage's own ``state.json``
entries (``records.json``), seals it with ``.complete.json`` and counts the
per-round table from the archive alone. The worlds come from
``tests/support/cascade_world.py``: real files in the shapes the phases write,
no phase run, no provider.

Contract: docs/development/workflows.md, "A cascade (``--cascade``)".
"""

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import cascade_archive as archive  # noqa: E402
from utils import cascade_ledger as ledger  # noqa: E402
from utils import detach, process_control  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from tests.support.cascade_world import (  # noqa: E402
    OTHER_CASCADE_ID,
    T,
    CascadeWorld,
    groups_of,
    ids,
    member,
    tree,
)

SKILL_DIR = Path(__file__).resolve().parent.parent
PLAN, CODE = "improve-plan", "improve-code"


def _yes():
    return True


def close(world: CascadeWorld, stage: int, recheck=_yes, round_id=None):
    return archive.close_out(
        plan_path=world.plan, mode=world.mode, cascade_dir=world.cascade_dir,
        cascade_id=world.cascade_id, stage=stage, model=world.models[stage - 1],
        round_id=round_id, recheck=recheck)


def stage_one(world: CascadeWorld):
    """Stage 1 of improve-plan: one finding of each outcome; its group ids."""
    world.freeze(1)
    groups = groups_of([
        member("Add a total column", "The plan never lists the total column.", "high"),
        member("Rename the module", "Rename report.py to export.py.", "low",
               reference="## Step 2: Delimiter"),
        member("Document the delimiter", "Say the delimiter is a comma.", "medium"),
        member("Add charts", "Add charts to the weekly report.", "medium"),
        member("Quote fields", "Quote fields that contain commas.", "medium"),
        member("Columns are listed", "Step 1 lists every column already.", "low"),
        member("Broken batch", "A group whose validation failed.", "low"),
    ])
    sha = world.write_review(1, groups, verdicts=[
        ("valid", "real gap"), ("invalid", "the module name is out of scope"),
        ("needs-human-decision", "ask"), ("needs-human-decision", "ask"),
        ("valid", "fine"), ("valid", "fine"), ("validation_failed", "batch failed"),
    ], started=0, completed=5)
    g = ids(groups)
    world.write_apply(1, sha, batches=[
        ("apply-batch-0", [g[0], g[2]], [str(world.plan)]),
        ("apply-batch-1", [g[4], g[5]], [str(world.plan)]),
    ], outcomes={g[0]: ("applied", ""), g[2]: ("applied", ""),
                 g[4]: ("declined", "conflicts with Step 2"), g[5]: ("no-op", "")},
        decisions={g[2]: ("approved", "judge_auto_decide", "a real gap"),
                   g[3]: ("skipped", "judge_auto_decide", "charts are not in the goal")},
        at=10, plan_edit="\n## Step 3: Totals\nAdd a total column.\n")
    return groups, sha


STAGE_ONE_COUNTS = {
    "findings": 7, "valid": 3, "invalid": 1, "needs_decision": 2, "reraises": 0,
    "applied": 2, "no_op": 1, "declined": 1, "skipped_by_user": 0, "held_back": 0,
    "judge_approved": 1, "judge_salvaged": 0, "judge_skipped": 1,
    "nothing_to_apply": False, "time_s": 660.0, "cost": "$0.04 est.",
}


def unlocked(files):
    return {name: data for name, data in files.items() if not name.endswith(".lock")}


def artifacts_of(world: CascadeWorld, stage: int) -> Path:
    return world.cascade_dir / f"stage-{stage}" / "artifacts"


# ---------------------------------------------------------------------------
# Publication: copies, derived files, the mark
# ---------------------------------------------------------------------------

class TestPublication:
    def test_close_out_copies_both_dirs_and_writes_the_derived_files(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        live_review, live_apply = tree(world.review_dir), tree(world.apply_dir)
        counts = close(world, 1)
        assert counts == STAGE_ONE_COUNTS
        published = artifacts_of(world, 1)
        files = tree(published)
        assert not (world.cascade_dir / "stage-1" / "artifacts.partial").exists()
        # Every live file but the locks, byte for byte.
        for name, data in live_review.items():
            if name.endswith(".lock"):
                assert f"review-plan/{name}" not in files
            else:
                assert files[f"review-plan/{name}"] == data
        for name, data in live_apply.items():
            if name.endswith(".lock"):
                assert f"apply-suggestions/{name}" not in files
            else:
                assert files[f"apply-suggestions/{name}"] == data
        assert "review-plan/prompts/reviewer.txt" in files
        for derived in ("records.json", "plan.after.md", "stage.diff", ".complete.json"):
            assert derived in files
        assert "files-touched.txt" not in files
        # The live dirs are as the phases left them (a lock file keeps the
        # label of its last holder, which is the close-out's own).
        assert unlocked(tree(world.review_dir)) == unlocked(live_review)
        assert unlocked(tree(world.apply_dir)) == unlocked(live_apply)

    def test_the_mark_is_written_last_and_names_every_file(self, tmp_path):
        world = CascadeWorld(tmp_path)
        _, sha = stage_one(world)
        close(world, 1, round_id="ab" * 8)
        published = artifacts_of(world, 1)
        mark = json.loads((published / ".complete.json").read_text(encoding="utf-8"))
        assert mark["cascade_id"] == world.cascade_id
        assert (mark["stage"], mark["model"], mark["mode"]) == (1, "openrouter:cheap", PLAN)
        assert mark["round_id"] == "ab" * 8
        assert mark["grouped_sha256"] == sha
        assert mark["files"] == ledger.hash_tree(published)
        assert ".complete.json" not in mark["files"]

    def test_round_id_defaults_to_the_reviews_own(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        close(world, 1)
        mark = json.loads((artifacts_of(world, 1) / ".complete.json").read_text("utf-8"))
        assert mark["round_id"] == "0000000000000001"

    def test_plan_after_and_the_stage_diff(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        before = (world.cascade_dir / "stage-1" / "inputs" / "plan.before.md").read_text("utf-8")
        close(world, 1)
        published = artifacts_of(world, 1)
        assert (published / "plan.after.md").read_bytes() == world.plan.read_bytes()
        diff = (published / "stage.diff").read_text(encoding="utf-8")
        assert diff.startswith("--- a/plan.before.md\n+++ b/plan.after.md\n")
        assert "+## Step 3: Totals" in diff
        assert before != world.plan.read_text(encoding="utf-8")

    def test_copies_keep_modes_and_mtimes(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        status = world.review_dir / ".status.json"
        os.chmod(status, 0o600)
        os.utime(status, (1_700_000_000, 1_700_000_000))
        close(world, 1)
        copy = artifacts_of(world, 1) / "review-plan" / ".status.json"
        assert int(copy.stat().st_mtime) == 1_700_000_000
        if os.name != "nt":
            assert (copy.stat().st_mode & 0o777) == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_links_in_a_live_dir_are_not_followed(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        secret = tmp_path / "outside.txt"
        secret.write_text("not the phase's", encoding="utf-8")
        os.symlink(secret, world.review_dir / "linked.txt")
        close(world, 1)
        assert not (artifacts_of(world, 1) / "review-plan" / "linked.txt").exists()

    def test_the_deepest_archived_path_stays_short(self, tmp_path):
        """Windows' path limit: a typical layout's deepest archived file, from
        the plan's output dir, stays well inside 240 characters."""
        world = CascadeWorld(tmp_path, CODE, of=12, models=[f"openrouter:m{k}" for k in range(1, 13)])
        groups = groups_of([member("Fix f", "f is wrong.", file="src/app/report.py")], code=True)
        sha = world.write_review(12, groups, verdicts=[("valid", "ok")], started=0, completed=1)
        (world.review_dir / "openrouter_deepseek-deepseek-v4.1-flash-0731.json").write_text("[]")
        snap = (world.apply_dir / "pre_batch" / "apply-fix-batch-12" / "skills" / "multi-llm"
                / "utils" / "review_orchestrator_base.py")
        snap.parent.mkdir(parents=True)
        snap.write_text("x = 1\n", encoding="utf-8")
        world.write_apply(12, sha, batches=[("apply-fix-batch-12", ids(groups), ["src/app/report.py"])],
                          outcomes={ids(groups)[0]: ("applied", "")}, at=3)
        close(world, 12)
        deepest = max(
            (p for p in artifacts_of(world, 12).rglob("*") if p.is_file()),
            key=lambda p: len(str(p.relative_to(world.out))))
        for name in ("artifacts", "artifacts.partial"):
            shown = str(deepest.relative_to(world.out)).replace("artifacts", name, 1)
            assert len(shown) < 240, shown

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_a_link_swapped_in_for_the_stage_dir_is_refused(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        stage_dir = world.cascade_dir / "stage-1"
        inputs = tree(stage_dir)
        for path in sorted(stage_dir.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        stage_dir.rmdir()
        os.symlink(outside, stage_dir)
        with pytest.raises(archive.ArchiveRefusal) as caught:
            close(world, 1)
        assert "could not be written" in caught.value.message
        assert list(outside.rglob("*")) == []
        assert inputs  # the stage had inputs before the swap


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
class TestALinkAtALiveDirIsRefused:
    """``os.walk`` follows a link given as its root, and a lock file opened
    through one lands at its target: a live dir (or the output dir above it)
    that is a link stops the close-out before any lock or copy."""

    @staticmethod
    def _files(root: Path) -> Dict[str, Any]:
        """Every file below *root* (links not followed): its bytes and mtime."""
        return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file() and not p.is_symlink()}

    def _swap_for_link(self, tmp_path: Path, live: Path) -> Path:
        """Move *live* out of the plan's folder and leave a link in its place."""
        outside = tmp_path / "outside" / live.name
        outside.parent.mkdir(exist_ok=True)
        live.rename(outside)
        (outside / "private.txt").write_text("PRIVATE-SIBLING-DATA", encoding="utf-8")
        os.symlink(outside, live, target_is_directory=True)
        return outside

    @pytest.mark.parametrize("which", ["review_dir", "apply_dir", "run_dir"])
    def test_a_live_dir_swapped_for_a_link_refuses_locking_and_copying_nothing(
            self, tmp_path, acquired, which):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        outside = self._swap_for_link(tmp_path, getattr(world, which))
        outside_before = self._files(outside)
        with pytest.raises(archive.ArchiveRefusal) as caught:
            close(world, 1)
        assert "is a link" in caught.value.message
        assert "Nothing was locked, copied or archived" in caught.value.message
        assert acquired == [], "no lock was taken"
        assert self._files(outside) == outside_before, "nothing was written outside"
        assert not artifacts_of(world, 1).exists()
        assert not (world.cascade_dir / "stage-1" / "artifacts.partial").exists()

    def test_an_output_dir_that_is_a_link_refuses(self, tmp_path, acquired):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        outside = tmp_path / "outside-out"
        world.out.rename(outside)
        os.symlink(outside, world.out, target_is_directory=True)
        locks_before = {p: p.read_bytes() for p in outside.rglob(".lock")}
        with pytest.raises(archive.ArchiveRefusal) as caught:
            archive.close_out(
                plan_path=world.plan, mode=world.mode,
                cascade_dir=outside / world.cascade_dir.relative_to(world.out),
                cascade_id=world.cascade_id, stage=1, model=world.models[0],
                round_id=None, recheck=_yes)
        assert f"{world.out} is a link" in caught.value.message
        assert acquired == []
        assert {p: p.read_bytes() for p in outside.rglob(".lock")} == locks_before

    @pytest.mark.skipif(not archive._HOLDS_LIVE, reason="no descriptor-relative walk here")
    def test_the_locks_are_opened_through_the_folders_held(self, tmp_path, monkeypatch):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        original = process_control.PhaseLock.acquire
        seen = {}

        def acquire(self, payload=None, **kwargs):
            seen[self.path] = kwargs.get("dir_fd")
            return original(self, payload, **kwargs)

        monkeypatch.setattr(process_control.PhaseLock, "acquire", acquire)
        assert close(world, 1) == STAGE_ONE_COUNTS
        assert set(seen) == {_lock(world.run_dir), _lock(world.review_dir),
                             _lock(world.apply_dir)}
        assert all(isinstance(fd, int) for fd in seen.values())

    @pytest.mark.skipif(not archive._HOLDS_LIVE, reason="no descriptor-relative walk here")
    def test_a_dir_swapped_after_it_was_held_is_never_entered(self, tmp_path, monkeypatch):
        """The tree is walked from the folder held: a folder below it swapped
        for a link during the close-out is not followed, and the dir itself
        moved away after its lock was taken refuses."""
        world = CascadeWorld(tmp_path)
        stage_one(world)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "private.txt").write_text("PRIVATE", encoding="utf-8")
        original = archive._take

        def take_then_swap(stack, plan, mode, phase, directory, live):
            original(stack, plan, mode, phase, directory, live)
            if directory == world.review_dir:
                directory.rename(directory.with_name("moved-away"))
                os.symlink(outside, directory, target_is_directory=True)

        monkeypatch.setattr(archive, "_take", take_then_swap)
        with pytest.raises(archive.ArchiveRefusal) as caught:
            close(world, 1)
        assert "no longer the folder its lock was taken in" in caught.value.message
        assert sorted(p.name for p in outside.iterdir()) == ["private.txt"]
        assert not artifacts_of(world, 1).exists()


# ---------------------------------------------------------------------------
# The recovery table
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_a_partial_archive_is_removed_and_redone(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        partial = world.cascade_dir / "stage-1" / "artifacts.partial"
        (partial / "review-plan").mkdir(parents=True)
        (partial / "review-plan" / "half.json").write_text("{", encoding="utf-8")
        (partial / ".complete.json").write_text("{}", encoding="utf-8")
        assert close(world, 1) == STAGE_ONE_COUNTS
        assert not partial.exists()
        assert not (artifacts_of(world, 1) / "review-plan" / "half.json").exists()

    def test_a_published_archive_is_read_and_never_copied_again(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        first = close(world, 1)
        before = tree(artifacts_of(world, 1))
        # The live dirs move on (the next stage's round); the archive does not.
        (world.review_dir / "grouped.json").write_text('{"groups": []}', encoding="utf-8")

        def must_not_run():
            raise AssertionError("a published stage needs no recheck and no lock")

        assert close(world, 1, recheck=must_not_run) == first
        assert tree(artifacts_of(world, 1)) == before
        assert archive.published_counts(world.cascade_dir, 1, world.cascade_id) == first

    @pytest.mark.parametrize("damage", ["missing", "not-json", "other-cascade",
                                        "other-stage", "edited-file", "deleted-file"])
    def test_an_archive_whose_mark_does_not_check_out_stops(self, tmp_path, damage):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        close(world, 1)
        published = artifacts_of(world, 1)
        mark_path = published / ".complete.json"
        mark = json.loads(mark_path.read_text(encoding="utf-8"))
        if damage == "missing":
            mark_path.unlink()
        elif damage == "not-json":
            mark_path.write_text("{", encoding="utf-8")
        elif damage == "other-cascade":
            mark["cascade_id"] = OTHER_CASCADE_ID
            mark_path.write_text(json.dumps(mark), encoding="utf-8")
        elif damage == "other-stage":
            mark["stage"] = 2
            mark_path.write_text(json.dumps(mark), encoding="utf-8")
        elif damage == "edited-file":
            (published / "records.json").write_text("{}", encoding="utf-8")
        else:
            (published / "review-plan" / "report.md").unlink()
        before = tree(published)
        with pytest.raises(archive.ArchiveRefusal) as caught:
            close(world, 1)
        assert str(published) in caught.value.message
        assert caught.value.code == 3
        # Restoring it or a new cascade, never moving it aside: a closed
        # stage would drop out of the ledger without a word.
        assert "Restore the archive as it was published" in caught.value.message
        assert "--force --cascade" in caught.value.message
        assert "aside" not in caught.value.message
        assert tree(published) == before
        assert archive.published_counts(world.cascade_dir, 1, world.cascade_id) is None

    def test_published_counts_without_an_archive(self, tmp_path):
        world = CascadeWorld(tmp_path)
        assert archive.published_counts(world.cascade_dir, 1, world.cascade_id) is None

    def test_a_race_seen_after_locking_releases_and_raises(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        seen = []

        def recheck():
            seen.append([process_control.held_phase_lock(detach.lock_path(str(d))) is not None
                         for d in (world.run_dir, world.review_dir, world.apply_dir)])
            return False

        with pytest.raises(archive.CloseOutRaced):
            close(world, 1, recheck=recheck)
        assert seen == [[True, True, True]]
        for directory in (world.run_dir, world.review_dir, world.apply_dir):
            assert process_control.held_phase_lock(detach.lock_path(str(directory))) is None
            assert process_control.lock_is_free(detach.lock_path(str(directory)))
        assert not (world.cascade_dir / "stage-1").joinpath("artifacts").exists()
        assert not (world.cascade_dir / "stage-1").joinpath("artifacts.partial").exists()

    def test_an_unreadable_state_file_stops(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        world.state_path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(archive.ArchiveRefusal) as caught:
            close(world, 1)
        assert "state.json" in caught.value.message
        assert not artifacts_of(world, 1).exists()


# ---------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------

@pytest.fixture
def acquired(monkeypatch):
    """Record the path of every ``PhaseLock.acquire`` call, in order."""
    calls = []
    original = process_control.PhaseLock.acquire

    def acquire(self, payload=None, **kwargs):
        calls.append(self.path)
        return original(self, payload, **kwargs)

    monkeypatch.setattr(process_control.PhaseLock, "acquire", acquire)
    return calls


def _lock(directory: Path) -> str:
    return detach.lock_path(str(directory))


class TestLocks:
    def test_run_dir_then_review_then_apply_and_all_released(self, tmp_path, acquired):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        close(world, 1)
        assert acquired == [_lock(world.run_dir), _lock(world.review_dir),
                            _lock(world.apply_dir)]
        for directory in (world.run_dir, world.review_dir, world.apply_dir):
            assert process_control.held_phase_lock(_lock(directory)) is None
            assert process_control.lock_is_free(_lock(directory))

    def test_a_held_run_dir_lock_is_used_as_it_is_and_stays_held(self, tmp_path, acquired):
        """The detached worker holds its run dir's lock for the whole cascade."""
        world = CascadeWorld(tmp_path)
        stage_one(world)
        lock = process_control.PhaseLock(_lock(world.run_dir))
        lock.acquire({"holder": "improve-plan worker", "pid": os.getpid()})
        try:
            acquired.clear()
            assert close(world, 1) == STAGE_ONE_COUNTS
            assert acquired == [_lock(world.review_dir), _lock(world.apply_dir)]
            assert lock.held
            assert process_control.held_phase_lock(_lock(world.run_dir)) is lock
        finally:
            lock.release()

    def test_a_lock_handed_over_through_holding_is_reentered(self, tmp_path, acquired):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        lock = process_control.PhaseLock(_lock(world.review_dir))
        lock.acquire({"holder": "front door", "pid": os.getpid()})
        try:
            with iw.holding(world.review_dir, lock):
                acquired.clear()
                close(world, 1)
            assert acquired == [_lock(world.run_dir), _lock(world.apply_dir)]
            assert lock.held
        finally:
            lock.release()

    def test_each_lock_is_registered_for_the_recheck(self, tmp_path):
        """A live-owner check inside ``recheck`` does not read the close-out's
        own locks as another run's."""
        world = CascadeWorld(tmp_path)
        stage_one(world)
        owners = []

        def recheck():
            owners.append(iw.live_owner(PLAN, world.plan, skip_own_run_dir=False))
            return True

        close(world, 1, recheck=recheck)
        assert owners == [None]

    def test_a_lock_briefly_held_by_a_status_poll_is_retried(self, tmp_path, monkeypatch):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        original = process_control.PhaseLock.acquire
        refusals = {"left": 2}
        tries = []

        def acquire(self, payload=None, **kwargs):
            if self.path == _lock(world.review_dir):
                tries.append(1)
                if refusals["left"]:
                    refusals["left"] -= 1
                    raise process_control.LockUnavailable(self.path, {"holder": "status"})
            return original(self, payload, **kwargs)

        monkeypatch.setattr(process_control.PhaseLock, "acquire", acquire)
        assert close(world, 1) == STAGE_ONE_COUNTS
        assert len(tries) == 3

    def test_a_live_holder_refuses_and_nothing_is_archived(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        code = textwrap.dedent("""
            import os, sys
            sys.path.insert(0, sys.argv[2])
            from utils import process_control
            lock = process_control.PhaseLock(sys.argv[1])
            lock.acquire({"holder": "review-plan (foreground)", "mode": "review-plan",
                          "pid": os.getpid(),
                          "process_start_time": process_control.process_start_time(os.getpid())})
            print("held", os.getpid(), flush=True)
            sys.stdin.read()
        """)
        holder = subprocess.Popen(
            [sys.executable, "-c", code, _lock(world.review_dir), str(SKILL_DIR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            # Its own pid, not Popen.pid: on Windows sys.executable is the venv's
            # launcher, which starts the interpreter that takes the lock as a child.
            head, _, pid = holder.stdout.readline().strip().partition(" ")
            assert head == "held" and pid.isdigit(), (head, pid)
            with pytest.raises(archive.ArchiveRefusal) as caught:
                close(world, 1)
        finally:
            holder.stdin.close()
            holder.wait(timeout=30)
        message = caught.value.message
        assert message.startswith("ERROR: improve-plan: a live run owns")
        assert f"pid {pid}" in message
        assert "Watch it:" in message
        assert not (world.cascade_dir / "stage-1" / "artifacts").exists()
        assert not (world.cascade_dir / "stage-1" / "artifacts.partial").exists()
        assert process_control.lock_is_free(_lock(world.run_dir))

    def test_an_unconfirmed_teardown_refuses_and_releases(self, tmp_path, monkeypatch):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        checked = []

        def refuse(phase_dir, *, plan_file, phase, lock=None):
            checked.append(phase)
            if phase == "apply-suggestions":
                if lock is not None:
                    lock.release()
                raise detach.DetachError("ERROR: an earlier run's teardown is not confirmed")

        monkeypatch.setattr(detach, "refuse_unconfirmed_cleanup", refuse)
        with pytest.raises(archive.ArchiveRefusal) as caught:
            close(world, 1)
        assert "teardown is not confirmed" in caught.value.message
        assert checked == ["improve-plan", "review-plan", "apply-suggestions"]
        for directory in (world.run_dir, world.review_dir, world.apply_dir):
            assert process_control.lock_is_free(_lock(directory))

    def test_a_missing_apply_dir_is_not_made_to_be_locked(self, tmp_path, acquired):
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        groups = groups_of([member("Wrong", "Not a problem.")])
        world.write_review(1, groups, verdicts=[("invalid", "no")], started=0, completed=1)
        state = world.state()
        state["phases_skipped"] = {"apply-suggestions": {"skipped_at": T(2), "reason": "none"}}
        world.save_state(state)
        counts = close(world, 1)
        assert not world.apply_dir.exists()
        assert acquired == [_lock(world.run_dir), _lock(world.review_dir)]
        assert counts["nothing_to_apply"] is True
        assert counts["time_s"] == 120.0


# ---------------------------------------------------------------------------
# records.json: only the stage's own entries
# ---------------------------------------------------------------------------

class TestRecords:
    def _two_stages(self, world: CascadeWorld):
        groups1, sha1 = stage_one(world)
        close(world, 1)
        world.freeze(2)
        g1 = ids(groups1)
        # Stage 2 repeats stage 1's first finding word for word (same group
        # id, so its record overwrites stage 1's) and adds one of its own.
        groups2 = groups_of([
            member("Add a total column", "The plan never lists the total column.", "high"),
            member("Name the encoding", "Say the CSV is UTF-8.", "medium"),
        ])
        sha2 = world.write_review(2, groups2, verdicts=[("valid", "x"), ("valid", "y")],
                                  started=20, completed=25)
        g2 = ids(groups2)
        assert g2[0] == g1[0]
        world.write_apply(2, sha2, batches=[("apply-batch-0", g2, [str(world.plan)])],
                          outcomes={g2[0]: ("no-op", ""), g2[1]: ("applied", "")},
                          decisions={g2[1]: ("approved", "judge_auto_decide", "fine")},
                          at=30)
        return g1, g2, sha2

    def test_only_the_stages_entries_when_state_holds_earlier_ones(self, tmp_path):
        world = CascadeWorld(tmp_path)
        g1, g2, sha2 = self._two_stages(world)
        close(world, 2)
        records = json.loads((artifacts_of(world, 2) / "records.json").read_text("utf-8"))
        assert sorted(records["processed"]) == sorted(g2)
        assert records["processed"][g2[0]]["status"] == "no-op"
        assert sorted(records["human_decisions"]) == [g2[1]]
        assert records["grouped_sha256"] == sha2
        assert records["phases_completed"] == {"review-plan": T(25), "apply-suggestions": T(31)}
        assert records["scope_policy"]["review_policy"] == "expand"
        assert records["nothing_to_apply"] is False
        # Stage 1's archive kept its own, from before stage 2 overwrote state.json.
        first = json.loads((artifacts_of(world, 1) / "records.json").read_text("utf-8"))
        assert first["processed"][g1[0]]["status"] == "applied"
        assert sorted(first["human_decisions"]) == sorted([g1[2], g1[3]])

    def test_a_later_stage_with_the_same_grouping_takes_none_of_the_earlier_entries(
            self, tmp_path):
        """Group ids are content hashes: a stage whose findings repeat the
        earlier stage's word for word has the same ``grouped.json`` hash. Its
        apply handed nothing out, so the earlier stage's entries (recorded
        before this stage's review completed) are not its own: no phantom
        Applied count and no ``A2-<n>`` ledger entry."""
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        groups = groups_of([member("A thing", "Fix a."), member("B thing", "Fix b.")])
        sha1 = world.write_review(1, groups, verdicts=[("valid", "x"), ("valid", "y")],
                                  started=0, completed=2)
        g = ids(groups)
        world.write_apply(1, sha1, batches=[("apply-batch-0", g, [str(world.plan)])],
                          outcomes={g[0]: ("applied", ""), g[1]: ("applied", "")}, at=5)
        assert close(world, 1)["applied"] == 2
        world.freeze(2)
        sha2 = world.write_review(2, groups, verdicts=[("valid", "x"), ("valid", "y")],
                                  started=20, completed=25)
        assert sha2 == sha1
        world.write_apply(2, sha2, batches=[], outcomes={}, at=30)
        counts = close(world, 2)
        records = json.loads((artifacts_of(world, 2) / "records.json").read_text("utf-8"))
        assert records["processed"] == {}
        assert records["run_items"] == [] and records["batches"] == []
        assert (counts["applied"], counts["nothing_to_apply"]) == (0, True)
        built = ledger.build_ledger(world.cascade_dir, 2, PLAN)
        assert [e.id for e in built.applied] == ["A1-1", "A1-2"]

    def _repeated_stage(self, world: CascadeWorld, first_row: Dict[str, Any]):
        """Stage 1 applies A and B in one batch; stage 2 repeats all of stage 1's
        findings word for word plus C, so its first batch is stage 1's and its
        report row for it is *first_row*; C's batch is dispatched."""
        world.freeze(1)
        groups = groups_of([member("A thing", "Fix a."), member("B thing", "Fix b."),
                            member("C thing", "Fix c.")])
        sha1 = world.write_review(1, groups, verdicts=[("valid", "x")] * 3,
                                  started=0, completed=2)
        g = ids(groups)
        world.write_apply(1, sha1, batches=[("apply-batch-0", g[:2], [str(world.plan)])],
                          outcomes={g[0]: ("applied", ""), g[1]: ("applied", "")}, at=5)
        assert close(world, 1)["applied"] == 2
        world.freeze(2)
        sha2 = world.write_review(2, groups, verdicts=[("valid", "x")] * 3,
                                  started=20, completed=25)
        assert sha2 == sha1
        row = {"task_id": "apply-batch-0", "status": "ok", "expected_ids": g[:2],
               "write_outcome": "applied", "files_touched": ["plans/export.md"], **first_row}
        world.write_apply(
            2, sha2, batches=[("apply-batch-0", g[:2], [str(world.plan)]),
                              ("apply-batch-1", [g[2]], [str(world.plan)])],
            outcomes={g[0]: ("applied", ""), g[1]: ("applied", ""), g[2]: ("applied", "")},
            at=30, native=False, report_rows=[
                row,
                {"task_id": "apply-batch-1", "status": "ok", "expected_ids": [g[2]],
                 "write_outcome": "applied", "files_touched": ["plans/export.md"]},
            ])
        return g

    @pytest.mark.parametrize("first_row", [
        {"skipped": True},
        # Carried from an earlier run of stage 2 that had skipped it already.
        {"skipped": True, "from_earlier_run": True},
        # Its result file is on disk, written by stage 1 (before stage 2's review).
        {"skipped": True, "output_path": "/live/apply-suggestions/apply_batch_0.json"},
    ], ids=["skipped", "carried", "old-result-file"])
    def test_a_repeated_stages_reused_batches_are_no_op_not_applied(self, tmp_path, first_row):
        """Stage 2 repeats stage 1's findings word for word, so its batches
        are stage 1's: the write scheduler's resume finds their result files
        (``already complete — skipping``), dispatches nothing for them and
        stamps the items ``processed`` again after stage 2's review, inside
        its window. Only the batch stage 2 really dispatched counts as Applied
        and gets an ``A2-<n>`` entry; the reused ones are No-op."""
        world = CascadeWorld(tmp_path)
        if "output_path" in first_row:
            world.apply_dir.mkdir(parents=True, exist_ok=True)
            result = world.apply_dir / "apply_batch_0.json"
            result.write_text("{}", encoding="utf-8")
            at = datetime.fromisoformat(T(6)).timestamp()
            os.utime(result, (at, at))
        g = self._repeated_stage(world, first_row)
        counts = close(world, 2)
        records = json.loads((artifacts_of(world, 2) / "records.json").read_text("utf-8"))
        assert sorted(records["processed"]) == sorted(g), "the window keeps all three"
        assert records["reused_items"] == sorted(g[:2])
        assert (counts["applied"], counts["no_op"], counts["nothing_to_apply"]) == (1, 2, False)
        built = ledger.build_ledger(world.cascade_dir, 2, PLAN)
        assert [e.id for e in built.applied] == ["A1-1", "A1-2", "A2-1"]
        assert built.applied[-1].title == "C thing"
        table = archive.render_summary(world.cascade_dir, world.record({1: "done", 2: "done"}))
        cells = [line for line in table.splitlines() if line.startswith("| 2 |")][0].split(" | ")
        assert (cells[7], cells[8]) == ("1", "2"), cells
        assert "dispatched nothing" in table

    @pytest.mark.parametrize("first_row", [
        # A resume of stage 2: its earlier run's own row stands in for the skip.
        {"from_earlier_run": True},
        # A resume whose earlier run left no report: the result file is stage 2's.
        {"skipped": True, "output_path": "/live/apply-suggestions/apply_batch_0.json"},
    ], ids=["earlier-run-row", "new-result-file"])
    def test_a_stages_own_resumed_batches_stay_applied(self, tmp_path, first_row):
        world = CascadeWorld(tmp_path)
        if "output_path" in first_row:
            world.apply_dir.mkdir(parents=True, exist_ok=True)
            result = world.apply_dir / "apply_batch_0.json"
            result.write_text("{}", encoding="utf-8")
            at = datetime.fromisoformat(T(27)).timestamp()
            os.utime(result, (at, at))
        self._repeated_stage(world, first_row)
        counts = close(world, 2)
        records = json.loads((artifacts_of(world, 2) / "records.json").read_text("utf-8"))
        assert records["reused_items"] == []
        assert (counts["applied"], counts["no_op"]) == (3, 0)
        built = ledger.build_ledger(world.cascade_dir, 2, PLAN)
        assert [e.id for e in built.applied][-3:] == ["A2-1", "A2-2", "A2-3"]

    def test_unstamped_entries_count_when_their_item_was_handed_out(self, tmp_path):
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        groups = groups_of([member("A thing", "Fix a."), member("B thing", "Fix b.")])
        sha = world.write_review(1, groups, verdicts=[("valid", "x"), ("valid", "y")],
                                 started=0, completed=2)
        g = ids(groups)
        world.write_apply(1, sha, batches=[("apply-batch-0", [g[0]], [str(world.plan)])],
                          outcomes={g[0]: ("applied", "")}, at=5, stamped=False)
        state = world.state()
        processed = state["processed_apply-suggestions"]
        processed["stray0000000000"] = {"status": "applied", "details": {}, "timestamp": T(5)}
        processed[g[1]] = {"status": "applied", "details": {"grouped_sha256": sha},
                           "timestamp": T(3.5)}  # before native_run.started_at (T(4))
        world.save_state(state)
        close(world, 1)
        records = json.loads((artifacts_of(world, 1) / "records.json").read_text("utf-8"))
        assert list(records["processed"]) == [g[0]]
        assert records["run_items"] == [g[0]]

    def test_an_earlier_stages_batch_file_is_not_this_stages(self, tmp_path):
        """An apply that handed nothing out leaves the earlier stage's
        ``write_tasks.json`` in place; its batches are not this review's."""
        world = CascadeWorld(tmp_path)
        stage_one(world)
        close(world, 1)
        world.freeze(2)
        groups = groups_of([member("Maybe add charts", "Charts would help.", "low")])
        sha = world.write_review(2, groups, verdicts=[("needs-human-decision", "ask")],
                                 started=20, completed=22)
        world.write_apply(2, sha, batches=[], outcomes={},
                          decisions={ids(groups)[0]: ("skipped", "judge_auto_decide",
                                                      "not in the goal")}, at=24)
        counts = close(world, 2)
        records = json.loads((artifacts_of(world, 2) / "records.json").read_text("utf-8"))
        assert records["run_items"] == [] and records["batches"] == []
        assert records["processed"] == {}
        assert counts["nothing_to_apply"] is True
        assert (counts["applied"], counts["no_op"], counts["declined"]) == (0, 0, 0)
        assert (counts["judge_approved"], counts["judge_salvaged"],
                counts["judge_skipped"]) == (0, 0, 1)


# ---------------------------------------------------------------------------
# improve-code: files-touched.txt
# ---------------------------------------------------------------------------

class TestImproveCode:
    def _review(self, world):
        world.freeze(1)
        groups = groups_of([
            member("Off by one", "The loop skips the last row.", "high", file="src/a.py"),
            member("Leak", "The file is never closed.", "medium", file="src/b.py"),
        ], code=True)
        sha = world.write_review(1, groups, verdicts=[("valid", "x"), ("valid", "y")],
                                 started=0, completed=2)
        return groups, sha

    def test_files_touched_come_from_the_write_report(self, tmp_path):
        world = CascadeWorld(tmp_path, CODE)
        groups, sha = self._review(world)
        g = ids(groups)
        world.write_apply(1, sha, batches=[("apply-fix-batch-0", [g[0]], ["src/a.py"]),
                                           ("apply-fix-batch-1", [g[1]], ["src/b.py"])],
                          outcomes={g[0]: ("applied", ""), g[1]: ("applied", "")},
                          at=5, native=False, report_rows=[
                              {"task_id": "apply-fix-batch-0", "status": "ok",
                               "files_touched": ["src/a.py", "src/a_helpers.py"]},
                              {"task_id": "apply-fix-batch-1", "status": "failed",
                               "files_touched": ["src/b.py"]},
                          ])
        close(world, 1)
        published = artifacts_of(world, 1)
        assert (published / "files-touched.txt").read_text("utf-8") == "src/a.py\nsrc/a_helpers.py\n"
        assert not (published / "plan.after.md").exists()
        assert (published / "code-review" / "grouped.json").is_file()
        assert (published / "apply-fixes" / "write_batch_report.json").is_file()

    def test_without_a_report_the_applied_batches_declared_files(self, tmp_path):
        world = CascadeWorld(tmp_path, CODE)
        groups, sha = self._review(world)
        g = ids(groups)
        world.write_apply(1, sha, batches=[("apply-fix-batch-0", [g[0]], ["src/a.py"]),
                                           ("apply-fix-batch-1", [g[1]], ["src/b.py"])],
                          outcomes={g[0]: ("applied", ""), g[1]: ("declined", "risky")}, at=5)
        close(world, 1)
        assert (artifacts_of(world, 1) / "files-touched.txt").read_text("utf-8") == "src/a.py\n"

    def test_a_report_older_than_the_review_is_an_earlier_cycles(self, tmp_path):
        world = CascadeWorld(tmp_path, CODE)
        groups, sha = self._review(world)
        g = ids(groups)
        world.write_apply(1, sha, batches=[("apply-fix-batch-0", [g[0]], ["src/a.py"])],
                          outcomes={g[0]: ("applied", "")}, at=5, report_rows=[
                              {"task_id": "apply-fix-batch-0", "status": "ok",
                               "files_touched": ["src/old.py"]}])
        report = world.apply_dir / "write_batch_report.json"
        document = json.loads(report.read_text("utf-8"))
        document["written_at"] = T(-30)
        report.write_text(json.dumps(document), encoding="utf-8")
        close(world, 1)
        assert (artifacts_of(world, 1) / "files-touched.txt").read_text("utf-8") == "src/a.py\n"


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

class TestCounting:
    def test_every_column_of_a_full_stage(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        assert close(world, 1) == STAGE_ONE_COUNTS

    def test_user_skips_salvages_and_reraises(self, tmp_path):
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        groups = groups_of([member(f"Item {n}", f"Fix item {n}.") for n in range(4)])
        sha = world.write_review(1, groups, verdicts=[
            ("valid", "x", "R0-1"), ("invalid", "repeat", "R0-2"),
            ("needs-human-decision", "ask"), ("valid", "y")], started=0, completed=1)
        g = ids(groups)
        world.write_apply(1, sha, batches=[("apply-batch-0", [g[0], g[2], g[3]], ["p"])],
                          outcomes={g[0]: ("skipped", "skipped at your request: no"),
                                    g[2]: ("applied", ""), g[3]: ("declined", "no")},
                          decisions={g[2]: ("approved", "claude_auto_decide_salvage", "trim")},
                          at=3)
        counts = close(world, 1)
        assert counts["reraises"] == 2
        assert (counts["applied"], counts["declined"], counts["skipped_by_user"]) == (1, 1, 1)
        assert (counts["judge_approved"], counts["judge_salvaged"]) == (0, 1)

    def test_judge_only_outcomes_with_no_batch(self, tmp_path):
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        groups = groups_of([member(f"Ask {n}", f"Decide {n}.") for n in range(3)])
        sha = world.write_review(1, groups, verdicts=[("needs-human-decision", "ask")] * 3,
                                 started=0, completed=1)
        g = ids(groups)
        world.write_apply(1, sha, batches=[], outcomes={}, decisions={
            g[0]: ("approved", "judge_auto_decide", "a"),
            g[1]: ("approved", "judge_auto_decide_salvage", "b"),
            g[2]: ("skipped", "judge_auto_decide", "c"),
        }, at=3)
        counts = close(world, 1)
        assert (counts["judge_approved"], counts["judge_salvaged"],
                counts["judge_skipped"]) == (1, 1, 1)
        assert counts["nothing_to_apply"] is True
        assert counts["applied"] == counts["no_op"] == counts["declined"] == 0

    def test_validation_counts_fall_back_to_the_validation_file(self, tmp_path):
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        groups = groups_of([member("A", "a."), member("B", "b."), member("C", "c.")])
        sha = world.write_review(1, groups, verdicts=[
            ("valid", "x"), ("invalid", "y"), ("needs-human-decision", "z")],
            started=0, completed=1, validation_counts=False, cost=None)
        world.write_apply(1, sha, batches=[], outcomes={}, at=2)
        counts = close(world, 1)
        assert (counts["findings"], counts["valid"], counts["invalid"],
                counts["needs_decision"]) == (3, 1, 1, 1)
        assert counts["cost"] is None

    def test_scope_deferrals_are_counted_held_back(self, tmp_path):
        world = CascadeWorld(tmp_path)
        world.freeze(1)
        groups = groups_of([member("Also PDF", "Add PDF."), member("Columns", "List them."),
                            member("Also XLSX", "Add XLSX.")])
        g = ids(groups)
        sha = world.write_review(
            1, groups, verdicts=[("valid", "x")] * 3, started=0, completed=1,
            scope_policy={"review_policy": "preserve", "review_policy_source": "flag",
                          "recorded_at": T(0), "deferrals": [
                              {"group_id": g[0], "reason": "the goal names CSV only"},
                              {"group_id": g[2], "reason": "the goal names CSV only"}]})
        world.write_apply(1, sha, batches=[("apply-batch-0", [g[1]], ["p"])],
                          outcomes={g[1]: ("applied", "")}, at=2)
        counts = close(world, 1)
        assert (counts["held_back"], counts["applied"]) == (2, 1)

    def test_a_retried_stage_counts_from_the_records_start(self, tmp_path):
        """A retry fan-out resets the review's ``started_at``; the stage's
        start in ``record.json`` is set once, so the time takes it."""
        world = CascadeWorld(tmp_path)
        stage_one(world)
        status_path = world.review_dir / ".status.json"
        status = json.loads(status_path.read_text("utf-8"))
        status["started_at"] = T(8)  # the retry's opening
        status_path.write_text(json.dumps(status), encoding="utf-8")
        record = world.record(states={1: "running"})
        record["stages"][0]["started_at"] = T(0)
        (world.cascade_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
        assert close(world, 1)["time_s"] == 660.0
        assert archive.published_counts(world.cascade_dir, 1, world.cascade_id)["time_s"] == 660.0

    def test_without_a_record_the_reviews_start_stands_in(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        status_path = world.review_dir / ".status.json"
        status = json.loads(status_path.read_text("utf-8"))
        status["started_at"] = T(8)
        status_path.write_text(json.dumps(status), encoding="utf-8")
        (world.cascade_dir / "record.json").unlink()
        assert close(world, 1)["time_s"] == 180.0

    def test_time_is_not_recorded_without_a_start(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        status_path = world.review_dir / ".status.json"
        status = json.loads(status_path.read_text("utf-8"))
        del status["started_at"]
        status_path.write_text(json.dumps(status), encoding="utf-8")
        assert close(world, 1)["time_s"] is None

    def test_counts_are_json_and_carry_every_key(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        counts = close(world, 1)
        assert tuple(counts) == archive.COUNT_KEYS
        assert json.loads(json.dumps(counts)) == counts


# ---------------------------------------------------------------------------
# The per-round table
# ---------------------------------------------------------------------------

class TestSummary:
    def _cascade(self, tmp_path):
        world = CascadeWorld(tmp_path)
        stage_one(world)
        close(world, 1)
        # Stage 2's review produced nothing: recorded skipped, no archive.
        world.freeze(3)
        groups = groups_of([member("Totals rounding", "Round totals to cents.", "high"),
                            member("Add charts", "Add charts to the weekly report.")])
        sha = world.write_review(3, groups, verdicts=[
            ("invalid", "repeats R1-2", "R1-2"), ("needs-human-decision", "ask")],
            started=40, completed=45, cost=None)
        world.write_apply(3, sha, batches=[], outcomes={}, decisions={
            ids(groups)[1]: ("skipped", "judge_auto_decide", "still not in the goal")}, at=50)
        close(world, 3)
        record = world.record(states={1: "done", 2: "skipped", 3: "done"},
                              skip_reasons={2: "openrouter:mid: rate limited"},
                              state="finished")
        return world, record

    def test_the_table_rows(self, tmp_path):
        world, record = self._cascade(tmp_path)
        text = archive.render_summary(world.cascade_dir, record)
        lines = text.splitlines()
        assert lines[0] == archive.TABLE_HEADER
        assert lines[1] == archive.TABLE_RULE
        assert lines[2] == ("| 1 | openrouter:cheap | 7 | 3 | 1 | 2 | — | 2 | 1 | 1 | 0 "
                            "| 1/0/1 | 11m 0s | $0.04 est. |")
        assert lines[3] == ("| 2 | openrouter:mid | skipped: openrouter:mid: rate limited "
                            "|  |  |  |  |  |  |  |  |  |  |  |")
        assert lines[4] == ("| 3 | openrouter:top | 2 | 0 | 1 | 1 | 1 | 0 (nothing to apply) "
                            "| 0 | 0 | 0 | 0/0/1 | 11m 0s | not recorded |")
        assert all(line.count("|") == lines[0].count("|") for line in lines[:5])
        assert "Applied + No-op" in text
        assert "Only the last round's can still be applied" in text

    def test_the_time_cell_reads_the_records_start_and_finish(self, tmp_path):
        world, record = self._cascade(tmp_path)
        record["stages"][0].update(started_at=T(0), finished_at=T(12.5))
        lines = archive.render_summary(world.cascade_dir, record).splitlines()
        assert "| 12m 30s |" in lines[2]
        # Stage 3 has no start on record: the counts' time stands.
        assert "| 11m 0s |" in lines[4]

    def test_held_back_shows_in_its_column(self, tmp_path):
        world = CascadeWorld(tmp_path, of=1, models=["openrouter:only"])
        world.freeze(1)
        groups = groups_of([member("Also PDF", "Add PDF."), member("Columns", "List.")])
        g = ids(groups)
        sha = world.write_review(
            1, groups, verdicts=[("valid", "x")] * 2, started=0, completed=1,
            scope_policy={"review_policy": "preserve", "review_policy_source": "flag",
                          "recorded_at": T(0), "deferrals": [
                              {"group_id": g[0], "reason": "out of scope"}]})
        world.write_apply(1, sha, batches=[("apply-batch-0", [g[1]], ["p"])],
                          outcomes={g[1]: ("applied", "")}, at=2)
        close(world, 1)
        row = archive.render_summary(world.cascade_dir,
                                     world.record(states={1: "done"})).splitlines()[2]
        cells = [c.strip() for c in row.strip("|").split("|")]
        header = [c.strip() for c in archive.TABLE_HEADER.strip("|").split("|")]
        assert cells[header.index("Held back")] == "1"
        assert cells[header.index("Applied")] == "1"

    def test_summary_and_ledger_views_are_written(self, tmp_path):
        world, record = self._cascade(tmp_path)
        text = archive.render_summary(world.cascade_dir, record)
        summary = (world.cascade_dir / "summary.md").read_text("utf-8")
        assert summary.startswith(f"# Cascade summary\n\nCascade `{world.cascade_id}`")
        assert summary.endswith(text)
        view = (world.cascade_dir / "ledger.md").read_text("utf-8")
        expected = ledger.render_full(ledger.build_ledger(world.cascade_dir, 3, PLAN))
        assert view == expected
        assert "- R1-1 " in view and "- R3-1 " in view and "- A1-1 " in view

    def test_declined_shows_user_skips(self, tmp_path):
        world = CascadeWorld(tmp_path, of=1, models=["openrouter:only"])
        world.freeze(1)
        groups = groups_of([member("A", "a."), member("B", "b.")])
        sha = world.write_review(1, groups, verdicts=[("valid", "x"), ("valid", "y")],
                                 started=0, completed=1)
        g = ids(groups)
        world.write_apply(1, sha, batches=[("apply-batch-0", g, ["p"])], outcomes={
            g[0]: ("declined", "no"), g[1]: ("skipped", "skipped at your request")}, at=2)
        close(world, 1)
        text = archive.render_summary(world.cascade_dir, world.record(states={1: "done"}))
        assert "| 2 (1 skipped) |" in text.splitlines()[2]

    def test_stages_not_closed_yet(self, tmp_path):
        world = CascadeWorld(tmp_path)
        text = archive.render_summary(world.cascade_dir, world.record(states={1: "running"}))
        rows = text.splitlines()[2:5]
        assert rows[0].startswith("| 1 | openrouter:cheap | in progress |")
        assert rows[1].startswith("| 2 | openrouter:mid | not started |")
        assert (world.cascade_dir / "ledger.md").read_text("utf-8") == ""
