"""Every writer under a plan's directory publishes through its folder held from
there (docs/development/write-safety.md#plan-output-containment).

The run locates its outputs first (``output_handler.output_dir_path``, the
run's start), and then, at a seam just before each writer's write, a link
leading out of the plan's directory is swapped in for the phase folder or the
file's own folder. Nothing may be written, renamed or removed outside: the
writer refuses (or, where it is best-effort, skips) instead. On the base
these writers wrote by path and followed the link.

The file imports nothing the base lacked, so each case can be run against it.
The mechanism's own tests are in ``tests/test_held_folder.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import failure_capture, output_handler, review_round, state_file  # noqa: E402
from utils import status_file, task_result, validation, write_reports  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX: Windows writes by the checked path (documented)")

VICTIM = b"outside the plan's directory\n"


class Tree:
    """A plan in ``repo/plans/``, its run started (the output folder located),
    and a folder outside the plan's directory holding a victim file at every
    name a writer could be redirected to."""

    def __init__(self, tmp_path: Path, phase: str = "implement") -> None:
        self.plans = tmp_path / "repo" / "plans"
        self.plans.mkdir(parents=True)
        self.plan = self.plans / "feature.md"
        self.plan.write_text("# Feature\n", encoding="utf-8")
        # The run's start: it locates its outputs.
        self.output = output_handler.output_dir_path(self.plan)
        self.phase = output_handler.phase_dir_path(self.plan, phase)
        self.phase.mkdir(parents=True)
        self.outside = tmp_path / "outside"
        self.outside.mkdir()

    def plant(self, *names: str) -> None:
        for name in names:
            victim = self.outside / name
            victim.parent.mkdir(parents=True, exist_ok=True)
            victim.write_bytes(VICTIM)

    def swap_for_link(self, folder: Path) -> None:
        """Move *folder* aside and put a link to the outside folder at its path."""
        folder.rename(folder.with_name(folder.name + ".moved"))
        os.symlink(self.outside, folder)

    def outside_untouched(self, *names: str) -> None:
        listing = sorted(str(p.relative_to(self.outside)) for p in self.outside.rglob("*"))
        expected = set()
        for name in names:
            parts = Path(name).parts
            expected.update(str(Path(*parts[:i])) for i in range(1, len(parts) + 1))
        assert listing == sorted(expected), listing
        for name in names:
            assert (self.outside / name).read_bytes() == VICTIM, name


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


def _refusal(call, *args):
    """What *call* raised (``None`` when it returned): the outside folder is
    checked first, so a write that went through fails on that."""
    try:
        call(*args)
    except Exception as exc:  # noqa: BLE001 - returned for the assertion
        return exc
    return None


class _JsonThatSwaps:
    """``state_file``'s ``json``, with a swap made as the document is encoded:
    just before the write."""

    def __init__(self, swap):
        self._swap = swap

    def dumps(self, *args, **kwargs):
        self._swap()
        return json.dumps(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(json, name)


# ---------------------------------------------------------------------------
# state.json: StateManager and every raw writer go through state_file
# ---------------------------------------------------------------------------

def test_state_manager_save_never_writes_through_a_swapped_output_folder(tree, monkeypatch):
    manager = StateManager(tree.plan)
    tree.plant("state.json")
    manager.state["review_scope_files"] = ["src/a.py"]
    manager._touched.add("review_scope_files")
    monkeypatch.setattr(state_file, "json",
                        _JsonThatSwaps(lambda: tree.swap_for_link(tree.output)))
    refused = _refusal(manager.save)
    tree.outside_untouched("state.json")
    assert isinstance(refused, OSError)


def test_a_state_write_handed_the_file_alone_never_writes_through_a_swapped_output_folder(
        tmp_path, monkeypatch):
    """``util batch-done`` / ``util metrics`` / ``finalize-tracking`` get the
    state file by ``--state-file`` and never locate the outputs themselves."""
    plans = tmp_path / "repo" / "plans"
    output = plans / "feature"
    output.mkdir(parents=True)
    path = output / "state.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "state.json").write_bytes(VICTIM)

    def swap():
        output.rename(plans / "feature.moved")
        os.symlink(outside, output)

    monkeypatch.setattr(state_file, "json", _JsonThatSwaps(swap))

    def record():
        with state_file.update(path) as document:
            document["a"] = 2

    refused = _refusal(record)
    assert sorted(p.name for p in outside.iterdir()) == ["state.json"]
    assert (outside / "state.json").read_bytes() == VICTIM
    assert isinstance(refused, OSError)


@pytest.mark.parametrize("swapped", ["plans", "repo"])
def test_a_state_only_command_keeps_the_root_it_read_from(tmp_path, monkeypatch, swapped):
    """``util metrics start --state-file`` records the plan's directory before
    its locked read, so the plan's directory (or a folder above it) swapped for
    a link between that read and the write never becomes the root."""
    from utils import metrics

    repo = tmp_path / "repo"
    path = repo / "plans" / "feature" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    folder = repo if swapped == "repo" else repo / "plans"
    outside = tmp_path / "outside"
    victim = outside / path.relative_to(folder)
    victim.parent.mkdir(parents=True)
    victim.write_bytes(VICTIM)
    real_read = state_file.read_document_strict

    def read_then_swap(state_path):
        document = real_read(state_path)
        folder.rename(folder.with_name(folder.name + ".moved"))
        os.symlink(outside, folder)
        return document

    monkeypatch.setattr(state_file, "read_document_strict", read_then_swap)
    refused = _refusal(metrics.main, ["start", "--state-file", str(path),
                                      "--phase", "implement", "--total-batches", "1"])
    assert victim.read_bytes() == VICTIM
    assert sorted(p.name for p in victim.parent.iterdir()) == ["state.json"]
    assert isinstance(refused, OSError)


#: A state file at the old hash-named location, as an old StateManager left it.
LEGACY_STATE = b'{\n  "schema_version": "2.0",\n  "task_status": {"T1": "completed"}\n}'


def _legacy_state_file(tree: Tree, monkeypatch) -> Path:
    """The plan's state file at the old location (``<skill>/state/<hash>.json``,
    found from ``state_manager``'s ``__file__``), in a scratch skill folder."""
    from utils import state_manager

    skill = tree.outside.parent / "skill"
    monkeypatch.setattr(state_manager, "__file__", str(skill / "utils" / "state_manager.py"))
    digest = state_manager.hashlib.sha256(str(tree.plan.resolve()).encode()).hexdigest()[:16]
    legacy = skill / "state" / f"{digest}.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(LEGACY_STATE)
    return legacy


class _JsonThatSwapsAfterLoad:
    """``state_manager``'s ``json``, with a swap made once the first load
    returns: the migration's check of the old file, just before its write."""

    def __init__(self, swap):
        self._swap = swap

    def _then_swap(self, real, *args, **kwargs):
        result = real(*args, **kwargs)
        if self._swap is not None:
            swap, self._swap = self._swap, None
            swap()
        return result

    def load(self, *args, **kwargs):
        return self._then_swap(json.load, *args, **kwargs)

    def loads(self, *args, **kwargs):
        return self._then_swap(json.loads, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(json, name)


@pytest.mark.parametrize("planted", [True, False], ids=["over-a-file", "new-file"])
def test_the_old_location_move_never_writes_through_a_swapped_output_folder(
        tree, monkeypatch, planted):
    """``StateManager`` moves a state file from its old location into the
    output folder; one swapped for a link leading out after the old file is
    checked is refused, and the old file is kept."""
    from utils import state_manager

    legacy = _legacy_state_file(tree, monkeypatch)
    if planted:
        tree.plant("state.json")
    monkeypatch.setattr(state_manager, "json",
                        _JsonThatSwapsAfterLoad(lambda: tree.swap_for_link(tree.output)))
    _refusal(StateManager, tree.plan)  # the move is best-effort: a refusal is logged
    tree.outside_untouched(*(["state.json"] if planted else []))
    assert legacy.read_bytes() == LEGACY_STATE
    assert not (tree.output.with_name(tree.output.name + ".moved") / "state.json").exists()


def test_the_old_location_move_writes_the_old_bytes_under_the_lock(tree, monkeypatch):
    legacy = _legacy_state_file(tree, monkeypatch)
    held = []
    real_write = state_file.write_in_plan_dir

    def write_in_plan_dir(path, data, **kwargs):
        held.append(bool(state_file._HELD_STATE_LOCKS))
        real_write(path, data, **kwargs)

    monkeypatch.setattr(state_file, "write_in_plan_dir", write_in_plan_dir)
    manager = StateManager(tree.plan)
    written = tree.output / "state.json"
    assert written.read_bytes() == LEGACY_STATE
    assert written.stat().st_mode & 0o777 == 0o600  # every state.json write's mode
    assert held == [True]
    assert not legacy.exists()
    assert manager.state["task_status"] == {"T1": "completed"}


def test_the_old_location_move_rechecks_under_the_lock(tree, monkeypatch):
    """A ``state.json`` another process wrote before this one took the lock
    is kept, and so is the old file."""
    legacy = _legacy_state_file(tree, monkeypatch)
    real_locked = state_file.locked
    theirs = b'{"task_status": {"T2": "completed"}}'

    @contextlib.contextmanager
    def locked(path):
        (tree.output / "state.json").write_bytes(theirs)
        with real_locked(path):
            yield

    monkeypatch.setattr(state_file, "locked", locked)
    StateManager(tree.plan)
    assert (tree.output / "state.json").read_bytes() == theirs
    assert legacy.read_bytes() == LEGACY_STATE


@pytest.mark.parametrize("bad", [b"\xff", b"{not json"], ids=["bad-utf8", "bad-json"])
def test_the_old_location_move_keeps_an_unreadable_old_file(tree, monkeypatch, bad):
    """An old state file that is not UTF-8 JSON is logged and kept: the
    constructor goes on with a new state rather than raising."""
    legacy = _legacy_state_file(tree, monkeypatch)
    legacy.write_bytes(bad)
    manager = StateManager(tree.plan)
    assert legacy.read_bytes() == bad
    assert manager.state["task_status"] == {}


class _PathThatSwapsOnExists(type(Path())):
    """``state.json``'s path, with a swap made as its existence is checked:
    just before the removal."""

    def exists(self, *args, **kwargs):
        swap = getattr(self, "swap", None)
        if swap is not None:
            self.swap = None
            swap()
        return super().exists(*args, **kwargs)


def test_deleting_the_state_never_removes_through_a_swapped_output_folder(tree):
    manager = StateManager(tree.plan)
    manager.save()
    tree.plant("state.json")
    manager.state_file = _PathThatSwapsOnExists(manager.state_file)
    manager.state_file.swap = lambda: tree.swap_for_link(tree.output)
    refused = _refusal(manager.delete)
    tree.outside_untouched("state.json")
    assert isinstance(refused, OSError)
    assert (tree.output.with_name(tree.output.name + ".moved") / "state.json").exists()


# ---------------------------------------------------------------------------
# .status.json
# ---------------------------------------------------------------------------

def test_the_checkpoint_never_writes_through_a_swapped_phase_folder(tree, monkeypatch):
    tree.plant(".status.json")
    real_now = status_file.now_iso

    def swap_then_stamp():
        tree.swap_for_link(tree.phase)
        return real_now()

    monkeypatch.setattr(status_file, "now_iso", swap_then_stamp)
    status_file.write_status(str(tree.phase), {"state": "running"})  # best-effort: no raise
    tree.outside_untouched(".status.json")


# ---------------------------------------------------------------------------
# The prompt sidecar (validation._write_one_sidecar)
# ---------------------------------------------------------------------------

class _TaskThatSwaps(dict):
    """A task whose prompt is read just before its sidecar is written."""

    def __init__(self, swap, **fields):
        super().__init__(**fields)
        self._swap = swap

    def __getitem__(self, key):
        if key == "prompt" and self._swap is not None:
            swap, self._swap = self._swap, None
            swap()
        return super().__getitem__(key)


def test_a_prompt_sidecar_never_writes_through_a_swapped_phase_folder(tree, capsys):
    target = validation.sidecar_path(str(tree.phase), 0)
    tree.plant(os.path.basename(target))
    task = _TaskThatSwaps(lambda: tree.swap_for_link(tree.phase),
                          prompt="the prompt\n", prompt_path=target, prompt_lines=1)
    written = validation.write_prompt_sidecars([task], str(tree.phase))
    tree.outside_untouched(os.path.basename(target))
    assert written is False
    assert "[PROMPT_SIDECAR_SKIPPED]" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The executor's result (task_result.persist_result)
# ---------------------------------------------------------------------------

def test_a_result_never_writes_through_a_phase_folder_swapped_after_its_check(
        tree, monkeypatch):
    name = "validation_batch_0.json"
    tree.plant(name)
    task = {"task_id": "validation-batch-0", "kind": "validation", "output_name": name}

    @contextlib.contextmanager
    def swap_then_write(_phase_dir, _name):
        tree.swap_for_link(tree.phase)
        yield True

    monkeypatch.setattr(review_round, "persisting_result", swap_then_write)
    refused = _refusal(task_result.persist_result, task, str(tree.phase),
                       [{"verdict": "valid"}])
    tree.outside_untouched(name)
    assert isinstance(refused, OSError)


# ---------------------------------------------------------------------------
# The write scheduler's batch report (write_reports.write_report)
# ---------------------------------------------------------------------------

class _Stop(Exception):
    """Raised after the batch report's write, ending the call there (only
    when the write went through)."""


def test_the_batch_report_never_writes_through_a_swapped_phase_folder(tree):
    tree.plant(write_reports.WRITE_BATCH_REPORT_FILENAME)
    reports = object.__new__(write_reports.WriteReports)
    reports._ctx = types.SimpleNamespace(phase_dir=str(tree.phase))

    def stop():
        raise _Stop

    reports._ceiling_record = stop

    def swap_then_dict():
        tree.swap_for_link(tree.phase)
        return {"batches": []}

    result = types.SimpleNamespace(to_dict=swap_then_dict)
    refused = _refusal(reports.write_report, result)
    tree.outside_untouched(write_reports.WRITE_BATCH_REPORT_FILENAME)
    assert isinstance(refused, OSError)


# ---------------------------------------------------------------------------
# The failure captures (failure_capture._write_failure_capture)
# ---------------------------------------------------------------------------

def _capture(phase_dir: Path):
    return failure_capture._write_failure_capture(
        phase_dir=str(phase_dir), task={"task_id": "t1", "kind": "validation"},
        attempt_index=1, argv=["runner", "the prompt"], prompt="the prompt",
        stdout="out", stderr="err", exit_code=1, duration=0.5,
        status="nonzero_exit", error="it failed")


def test_a_failure_capture_is_held_from_the_plans_directory_not_the_phase_folder(tree):
    """The phase folder swapped after the run's start check and before the
    capture: its own check resolves both sides through the link, so only a
    walk from the plan's directory refuses it."""
    capture = "failed/t1.attempt1.json"
    tree.plant(capture)
    tree.swap_for_link(tree.phase)
    written = _capture(tree.phase)
    tree.outside_untouched(capture)
    assert written is None


@pytest.mark.parametrize("swapped", ["plans", "repo"])
def test_a_failure_capture_keeps_the_recorded_root(tree, swapped):
    """The plan's directory (or a folder above it) swapped after the run's
    start and before the capture: the capture's own check resolves the phase
    folder through the link, so only a check against the root recorded at the
    start, kept unresolved through the move aside and the write, refuses it."""
    folder = tree.plans if swapped == "plans" else tree.plans.parent
    capture = str((tree.phase / "failed" / "t1.attempt1.json").relative_to(folder))
    tree.plant(capture)
    tree.swap_for_link(folder)
    written = _capture(tree.phase)
    tree.outside_untouched(capture)
    assert written is None


def test_a_failure_capture_never_writes_through_a_swapped_capture_folder(tree, monkeypatch):
    (tree.phase / "failed").mkdir()
    tree.plant("t1.attempt1.json")
    real_retire = failure_capture._retire_capture

    def swap_then_retire(target, root, **kwargs):
        tree.swap_for_link(tree.phase / "failed")
        return real_retire(target, root, **kwargs)

    monkeypatch.setattr(failure_capture, "_retire_capture", swap_then_retire)
    written = _capture(tree.phase)
    tree.outside_untouched("t1.attempt1.json")
    assert written is None


# ---------------------------------------------------------------------------
# The generate-tasks restore's state.json (generate_tasks_orchestrator._Snapshot)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("before", ["written back", "removed"])
def test_the_restore_never_puts_state_back_through_a_swapped_output_folder(
        tree, monkeypatch, capsys, before):
    import generate_tasks_orchestrator as gto

    state = gto._state_file_for(tree.plan)
    tasks_target = output_handler.output_file_path(tree.plan, "tasks")
    if before == "written back":
        state.write_text('{"before": true}', encoding="utf-8")
    snapshot = gto._Snapshot(tree.plan, state, tasks_target)
    state.write_text('{"after": true}', encoding="utf-8")
    tree.plant("state.json")
    real_read = gto._Snapshot._read

    def swap_then_read(path):
        if path == state and not os.path.islink(tree.output):
            tree.swap_for_link(tree.output)
        return real_read(path)

    monkeypatch.setattr(gto._Snapshot, "_read", staticmethod(swap_then_read))
    snapshot.restore()
    tree.outside_untouched("state.json")
    assert "could not restore" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The write scheduler's rewrites of a result it persisted, and its inputs
# binding (fable review F5)
# ---------------------------------------------------------------------------

#: An outside JSON object of a result's name: the only file a rewrite by path
#: through a swapped phase folder would change (it reads, then rewrites, it).
OUTSIDE_RESULT = json.dumps({"status": "applied", "verdict": {"ok": True}}).encode()


def _persisted_then_swapped(tree, name: str) -> Path:
    """A result persisted as the executor persists it (held), then the phase
    folder swapped for a link to a folder holding :data:`OUTSIDE_RESULT` at
    that name; the result's path."""
    path = tree.phase / name
    failure_capture._atomic_write_json(
        str(path), {"status": "applied", "verdict": {"ok": True, "batch_check": ["x"]}})
    (tree.outside / name).write_bytes(OUTSIDE_RESULT)
    tree.swap_for_link(tree.phase)
    return path


def _outside_result_untouched(tree, name: str) -> None:
    assert (tree.outside / name).read_bytes() == OUTSIDE_RESULT
    assert sorted(p.name for p in tree.outside.iterdir()) == [name]


def test_the_verdict_stamp_never_writes_through_a_swapped_phase_folder(tree):
    from utils.task_outcome import TaskOutcome
    from utils.write_batch_scheduler import WriteBatchScheduler

    name = "implement_batch_1.json"
    path = _persisted_then_swapped(tree, name)
    lines: list = []
    stub = types.SimpleNamespace(
        _task_by_id=lambda task_id: None,
        _evidence=types.SimpleNamespace(task_fingerprint=lambda task: "f"),
        _checks=types.SimpleNamespace(tree_fingerprint=lambda: None),
        _dispatch_rungs={}, log=lines.append)
    outcome = TaskOutcome(task_id="implement-batch-1", kind="implement", runner="x",
                          status="completed", output_path=str(path))

    WriteBatchScheduler._stamp_verdict(stub, outcome)

    _outside_result_untouched(tree, name)
    assert any("could not record the verdict" in line for line in lines), lines


def test_the_own_check_rerecord_never_writes_through_a_swapped_phase_folder(tree):
    from utils.write_batch_scheduler import WriteBatchScheduler

    name = "implement_batch_2.json"
    path = _persisted_then_swapped(tree, name)
    lines: list = []
    stub = types.SimpleNamespace(log=lines.append)

    WriteBatchScheduler._restamp_own_check(stub, str(path), None, {"T9": ["x"]})

    _outside_result_untouched(tree, name)
    assert any("could not re-record the own check" in line for line in lines), lines


def test_the_tree_restamp_never_writes_through_a_swapped_phase_folder(tree):
    from utils.write_check_holds import CheckHolds

    name = "implement_batch_3.json"
    path = _persisted_then_swapped(tree, name)
    lines: list = []
    holds = object.__new__(CheckHolds)
    holds._ctx = types.SimpleNamespace(kind="implement", log=lines.append)
    holds.tree_fingerprint = lambda: "the tree now"
    outcome = types.SimpleNamespace(ok=True, check_findings=["x"], output_path=str(path),
                                    task_id="implement-batch-3")

    holds.restamp_check_trees(types.SimpleNamespace(outcomes=[outcome]))

    _outside_result_untouched(tree, name)
    assert any("could not re-stamp the tree" in line for line in lines), lines


def test_the_inputs_binding_never_writes_through_a_swapped_phase_folder(tree):
    from utils.write_resume import WRITE_FINGERPRINT_FILENAME, ResumeEvidence

    (tree.outside / WRITE_FINGERPRINT_FILENAME).write_bytes(OUTSIDE_RESULT)
    tree.swap_for_link(tree.phase)
    lines: list = []
    evidence = object.__new__(ResumeEvidence)
    evidence._ctx = types.SimpleNamespace(phase_dir=str(tree.phase), kind="implement",
                                          log=lines.append)
    evidence.task_fingerprint = lambda task: "f"

    evidence.record_fingerprint({"task_id": "implement-batch-1"})

    _outside_result_untouched(tree, WRITE_FINGERPRINT_FILENAME)
    assert any("could not record the inputs binding" in line for line in lines), lines


def test_the_inputs_binding_is_written_and_its_folder_made(tree):
    """Without a swap the binding is written, its phase folder made when
    missing, with a new file's mode as before."""
    from utils.write_resume import WRITE_FINGERPRINT_FILENAME, ResumeEvidence

    phase = output_handler.phase_dir_path(tree.plan, "apply-code-fixes")
    evidence = object.__new__(ResumeEvidence)
    evidence._ctx = types.SimpleNamespace(phase_dir=str(phase), kind="implement",
                                          log=print)
    evidence.task_fingerprint = lambda task: "f"

    evidence.record_fingerprint({"task_id": "implement-batch-1"})

    data = json.loads((phase / WRITE_FINGERPRINT_FILENAME).read_text(encoding="utf-8"))
    assert data["tasks"] == {"implement-batch-1": "f"}


# ---------------------------------------------------------------------------
# The write audit log and the pre-batch snapshot's manifest (fable review F6)
# ---------------------------------------------------------------------------

def test_the_audit_log_is_appended_owner_only(tree):
    from utils.write_safety import AUDIT_FILENAME, AuditLog

    audit = AuditLog(str(tree.phase))
    audit._append({"event": "one"})
    audit._append({"event": "two"})

    path = tree.phase / AUDIT_FILENAME
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == [
        "one", "two"]
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("planted", ["symlink", "dangling symlink", "fifo",
                                     "swapped phase folder"])
def test_the_audit_log_is_never_appended_through_a_link(tree, planted):
    """Fable review F6: a symlink at the audit log's name (one leading
    nowhere included, which the append would have created), something that
    is not a file there, or the phase folder swapped for a link never
    carries the line outside, nor blocks: the append is refused. Before, it
    was appended there."""
    from utils.write_safety import AUDIT_FILENAME, AuditLog

    tree.plant(AUDIT_FILENAME)
    victim = tree.outside / AUDIT_FILENAME
    if planted == "symlink":
        os.symlink(victim, tree.phase / AUDIT_FILENAME)
    elif planted == "dangling symlink":
        os.symlink(tree.outside / "made-by-the-append", tree.phase / AUDIT_FILENAME)
    elif planted == "fifo":
        os.mkfifo(tree.phase / AUDIT_FILENAME)
    else:
        tree.swap_for_link(tree.phase)

    refused = _refusal(AuditLog(str(tree.phase))._append, {"event": "completed"})

    assert isinstance(refused, OSError), refused
    tree.outside_untouched(AUDIT_FILENAME)


def test_a_snapshot_manifest_never_writes_through_a_swapped_phase_folder(tree):
    from utils.write_safety import MANIFEST_NAME, PAYLOAD_DIRNAME, PRE_BATCH_DIRNAME
    from utils.write_safety import TaskSnapshot

    repo = tree.plans.parent
    (repo / "a.txt").write_text("a\n")
    snapshot = TaskSnapshot.take(phase_dir=str(tree.phase), task_id="T1",
                                 files=["a.txt"], root=str(repo))
    folder = Path(PRE_BATCH_DIRNAME) / "T1"
    tree.plant(str(folder / MANIFEST_NAME))
    (tree.outside / folder / PAYLOAD_DIRNAME).mkdir()
    tree.swap_for_link(tree.phase)

    refused = _refusal(snapshot.save)

    assert isinstance(refused, OSError), refused
    assert (tree.outside / folder / MANIFEST_NAME).read_bytes() == VICTIM
