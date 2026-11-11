"""The host's views of the task files: ``<stem>.index.json`` and the apply summary.

A native host dispatches validation, revalidation and consolidation batches from
the index beside each task file, and drives an apply phase's batch loop from
``orchestrator_summary.json`` beside ``orchestrator_output.json``, reading one
batch's detail file (``orchestrator_batches/<task_id>.json``) at a time. All of
them are views: the full files stay the source of truth (the executor, ``util
check-reply``, the audit, the container digest, ``util write-prompt``),
byte-identical, and so do the prompt sidecars. A view is published only after
its source is written, removed before its source changes, and rebuilt from the
source at every hand-back, each sequence under the phase dir's views lock; one
already right is left alone, and an obsolete one that cannot be removed stops
the run (docs/development/delegation.md#the-host-index,
docs/development/write-scheduler.md#the-host-summary).
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils.validation as validation_module  # noqa: E402
from utils.apply_output_helpers import (  # noqa: E402
    ORCHESTRATOR_BATCHES_DIRNAME,
    ORCHESTRATOR_SUMMARY_FILENAME,
    batch_detail_path,
    build_batch_detail,
    build_orchestrator_summary,
    detail_pointer_lines,
    invalidate_orchestrator_views,
    publish_orchestrator_views,
    write_and_emit_output,
)
from utils.consolidation import prepare_consolidation_tasks  # noqa: E402
from utils.review_orchestrator_base import (  # noqa: E402
    dispatch_subagent_tasks,
    retire_stale_validation_round,
)
from utils.subagent_executor import (  # noqa: E402
    RESERVED_OUTPUT_NAMES,
    RESERVED_OUTPUT_SUFFIXES,
    ResultRejected,
    resolve_output_target,
)
from utils.validation import (  # noqa: E402
    HOST_INDEX_TASK_KEYS,
    build_host_index,
    ensure_prompt_sidecars,
    host_index_path,
    prepare_batched_revalidation_tasks,
    prepare_batched_validation_tasks,
    refresh_host_index,
    task_file_for_kind,
)

from tests.scope_apply_fixtures import run_apply, write_review  # noqa: E402
from tests.support.write_orchestrators import (  # noqa: E402
    orchestrator_repo,  # noqa: F401 - the repo fixture
)

SKILL_DIR = Path(__file__).resolve().parent.parent

#: Directory permissions decide these; a root run or Windows ignores them.
_posix_permissions = pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX directory permissions, as a non-root user",
)


# ---------------------------------------------------------------------------
# A real-shaped round: the 2026-09-17 measurement behind the sidecars was a
# 57,748-byte plan validated in fourteen batches of ~65 KB prompts.
# ---------------------------------------------------------------------------

def _real_shaped_plan() -> str:
    section = (
        "Each step names the files it touches, the tests that prove it and the "
        "order it lands in, so a reviewer can check the plan against the code "
        "without reading the history behind it. "
    ) * 19
    return "# Plan\n\n" + "".join(
        f"## Step {n}: Part {n}\n\n{section}\n\n" for n in range(1, 18)
    )


def _group(n: int) -> dict:
    return {
        "theme": f"Theme {n}",
        "category": "modification",
        "models": ["codex:gpt-6", "opencode:deepseek"],
        "suggestions": [{
            "title": f"Tighten step {n % 17 + 1}",
            "desc": (
                f"Step {n % 17 + 1} says what changes but not how it is verified; "
                "name the test that fails before the change and passes after it, "
                "and the file the fixture lives in. "
            ) * 3,
            "importance": ("HIGH", "MEDIUM", "LOW")[n % 3],
            "type": "modification",
            "reference": f"## Step {n % 17 + 1}",
        }],
    }




def _prepare(phase_dir: Path, groups: int = 54, batch_size: int = 4) -> dict:
    """A round materialized and not yet written: its caller's `json.dump` is next."""
    plan = _real_shaped_plan()
    assert 55_000 < len(plan) < 60_000
    return prepare_batched_validation_tasks(
        groups=[_group(n) for n in range(groups)],
        context=plan,
        output_dir=str(phase_dir),
        plan_file=str(phase_dir / "plan.md"),
        max_per_batch=batch_size,
    )


def _commit(phase_dir: Path, document: dict, name: str = "validation_tasks.json") -> None:
    with open(phase_dir / name, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def _hand_back(phase_dir: Path, name: str = "validation_tasks.json",
               label: str = "validation") -> None:
    """The hand-back every lane makes once the task file is written."""
    assert dispatch_subagent_tasks(
        SimpleNamespace(), task_file=str(phase_dir / name), phase_dir=str(phase_dir),
        phase="review-plan", label=label,
    ) is False


def _materialize(phase_dir: Path, groups: int = 54, batch_size: int = 4) -> dict:
    document = _prepare(phase_dir, groups, batch_size)
    _commit(phase_dir, document)
    _hand_back(phase_dir)
    return document


def _index(phase_dir: Path, name: str = "validation_tasks.index.json") -> dict:
    return json.loads((phase_dir / name).read_text(encoding="utf-8"))


def _full(phase_dir: Path, name: str = "validation_tasks.json") -> dict:
    return json.loads((phase_dir / name).read_text(encoding="utf-8"))


def _shrunk(document: dict) -> dict:
    """*document* without its last task: a source its old index no longer describes."""
    document = copy.deepcopy(document)
    document["batches"] = document["batches"][:-1]
    if "tasks" in document:
        document["tasks"] = document["batches"]
    document["total_batches"] = len(document["batches"])
    return document


def _describes(index: dict, document: dict) -> bool:
    """Whether *index* is the view of *document*, entry for entry."""
    return index["total_batches"] == document["total_batches"] and index["batches"] == [
        {key: task[key] for key in HOST_INDEX_TASK_KEYS if key in task}
        for task in document["batches"]
    ]


class TestTheValidationIndex:
    def test_it_lists_every_task_the_host_dispatches_and_no_prompt(self, tmp_path):
        document = _materialize(tmp_path)
        index = _index(tmp_path)
        assert index["index_of"] == "validation_tasks.json"
        assert index["total_batches"] == document["total_batches"] >= 14
        assert index["reaggregate_mode"] == document["reaggregate_mode"]
        assert index["plan_file"] == document["plan_file"]
        assert _describes(index, document)
        for entry in index["batches"]:
            assert set(entry) >= {"task_id", "prompt_path", "prompt_lines", "output_path"}
        assert all("prompt" not in entry for entry in index["batches"])
        assert "prompt" not in index

    def test_it_is_a_small_fraction_of_the_full_file(self, tmp_path):
        """The PERF-3 measurement: on real files the index is ~1.5% of the task file.

        Both files spell absolute paths, so the temp dir's own length is taken
        out of both before they are compared.
        """
        _materialize(tmp_path)
        prefix = json.dumps(str(tmp_path))[1:-1]

        def size(name: str) -> int:
            text = (tmp_path / name).read_text(encoding="utf-8")
            return len(text.replace(prefix, "/p").encode("utf-8"))

        full, index = size("validation_tasks.json"), size("validation_tasks.index.json")
        assert full > 500_000
        assert index < 8_000
        assert index / full < 0.015

    def test_the_full_file_and_the_sidecars_are_what_they_were(self, tmp_path):
        """A view, not a contract: the document and the file on disk are untouched."""
        document = _materialize(tmp_path)
        before = copy.deepcopy(document)
        on_disk = (tmp_path / "validation_tasks.json").read_bytes()
        build_host_index(document, str(tmp_path / "validation_tasks.json"))
        assert refresh_host_index(str(tmp_path / "validation_tasks.json")) is True
        assert document == before
        assert (tmp_path / "validation_tasks.json").read_bytes() == on_disk
        for task in document["batches"]:
            assert Path(task["prompt_path"]).read_bytes() == task["prompt"].encode("utf-8")

    def test_a_skipped_sidecar_leaves_its_entry_without_a_pointer(self, tmp_path, monkeypatch):
        real = validation_module._write_one_sidecar

        def refuse_sidecars(target, text):
            if target.endswith(".prompt.txt"):
                raise OSError(28, "No space left on device")
            return real(target, text)

        monkeypatch.setattr(validation_module, "_write_one_sidecar", refuse_sidecars)
        document = _materialize(tmp_path, groups=8)
        entries = _index(tmp_path)["batches"]
        assert len(entries) == document["total_batches"]
        assert all("prompt_path" not in e and "prompt_lines" not in e for e in entries)
        assert all(e["output_path"] for e in entries)

    def test_a_failed_write_leaves_no_index(self, tmp_path, monkeypatch, capsys):
        _materialize(tmp_path, groups=8)
        assert (tmp_path / "validation_tasks.index.json").exists()
        _commit(tmp_path, _shrunk(_full(tmp_path)))  # the index is now obsolete

        def refuse(target, text):
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(validation_module, "_write_one_sidecar", refuse)
        assert refresh_host_index(str(tmp_path / "validation_tasks.json")) is False
        assert not (tmp_path / "validation_tasks.index.json").exists()
        err = capsys.readouterr().err
        assert "validation_tasks.index.json not written" in err
        assert "a host reads the file it describes instead" in err

    def test_a_missing_task_file_leaves_no_index(self, tmp_path):
        _materialize(tmp_path, groups=8)
        (tmp_path / "validation_tasks.json").unlink()
        assert refresh_host_index(str(tmp_path / "validation_tasks.json")) is False
        assert not (tmp_path / "validation_tasks.index.json").exists()


class TestTheIndexNeverOutlivesItsFile:
    """Published after the file, removed before it changes, rebuilt at hand-back."""

    def test_an_interrupted_rematerialisation_leaves_no_index(self, tmp_path):
        """The new round's sidecars are written, its task file is not."""
        old = _materialize(tmp_path, groups=12, batch_size=4)
        new = _prepare(tmp_path, groups=12, batch_size=1)
        assert new["total_batches"] != old["total_batches"]
        assert not (tmp_path / "validation_tasks.index.json").exists()

    def test_the_reaggregate_hand_back_rebuilds_it_from_the_full_file(self, tmp_path):
        old = _materialize(tmp_path, groups=12, batch_size=4)
        _prepare(tmp_path, groups=12, batch_size=1)  # interrupted before its write
        loaded = _full(tmp_path)
        changed = ensure_prompt_sidecars(loaded, str(tmp_path))
        assert not (tmp_path / "validation_tasks.index.json").exists()
        if changed:
            _commit(tmp_path, loaded)
        _hand_back(tmp_path)
        index = _index(tmp_path)
        assert _describes(index, _full(tmp_path))
        assert index["total_batches"] == old["total_batches"]
        for entry, task in zip(index["batches"], loaded["batches"]):
            assert Path(entry["prompt_path"]).read_bytes() == task["prompt"].encode("utf-8")

    def test_the_base_lane_reaggregate_hands_back_the_full_files_index(self, tmp_path, capsys):
        """Through `reaggregate_from_existing_files`, host-driven (no executor)."""
        from tests.test_reopen_for_validation import _reaggregate, _round
        from utils.state_manager import load_groups_payload

        plan, out_dir, phase_dir, _state = _round(tmp_path, batches_on_disk=False)
        document = _full(phase_dir)
        groups = load_groups_payload(
            json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8")))
        rematerialized = prepare_batched_validation_tasks(  # interrupted before its write
            groups=groups, context=plan.read_text(encoding="utf-8"),
            output_dir=str(phase_dir), plan_file=str(plan), max_per_batch=1,
        )
        assert rematerialized["total_batches"] != document["total_batches"]

        _reaggregate(plan, out_dir, phase_dir)

        assert _describes(_index(phase_dir), _full(phase_dir))
        assert _index(phase_dir)["total_batches"] == document["total_batches"]

    def test_retiring_a_round_removes_its_index(self, tmp_path):
        _materialize(tmp_path, groups=3)
        retire_stale_validation_round(str(tmp_path), reason="host index test")
        assert not (tmp_path / "validation_tasks.json").exists()
        assert not (tmp_path / "validation_tasks.index.json").exists()

    def test_every_hand_back_replaces_an_obsolete_one(self, tmp_path):
        """An index edited, or left over from anything, is replaced at hand-back."""
        document = _materialize(tmp_path, groups=8)
        (tmp_path / "validation_tasks.index.json").write_text(
            json.dumps({"index_of": "validation_tasks.json", "total_batches": 99,
                        "batches": []}), encoding="utf-8")
        _hand_back(tmp_path)
        assert _describes(_index(tmp_path), document)

    @_posix_permissions
    def test_a_read_only_phase_dir_stops_the_rematerialisation(self, tmp_path, capsys):
        """The old index cannot be removed, so nothing goes on beside it."""
        phase = tmp_path / "phase"
        phase.mkdir()
        old = _materialize(phase, groups=3)
        phase.chmod(0o555)
        try:
            with pytest.raises(SystemExit) as stop:
                _prepare(phase, groups=12)
        finally:
            phase.chmod(0o755)
        assert stop.value.code == 3
        err = capsys.readouterr().err
        assert f"ERROR: Cannot remove {phase / 'validation_tasks.index.json'}" in err
        assert _describes(_index(phase), _full(phase))
        assert _full(phase)["total_batches"] == old["total_batches"]

    @_posix_permissions
    def test_an_index_that_cannot_be_replaced_stops_the_hand_back(self, tmp_path, capsys):
        phase = tmp_path / "phase"
        phase.mkdir()
        _materialize(phase, groups=3)
        _commit(phase, _shrunk(_full(phase)))  # the index is now obsolete
        phase.chmod(0o555)
        try:
            with pytest.raises(SystemExit) as stop:
                _hand_back(phase)
        finally:
            phase.chmod(0o755)
        assert stop.value.code == 3
        assert str(phase / "validation_tasks.index.json") in capsys.readouterr().err


def _lock_the_index(monkeypatch) -> None:
    """Simulated Windows sharing violations: the index can be neither replaced nor removed."""
    real_write, real_unlink = validation_module._write_one_sidecar, os.unlink

    def locked_write(target, text):
        if target.endswith(".index.json"):
            raise PermissionError(13, "[WinError 32] file in use")
        return real_write(target, text)

    def locked_unlink(path, *args, **kwargs):
        if str(path).endswith(".index.json"):
            raise PermissionError(13, "[WinError 32] file in use")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(validation_module, "_write_one_sidecar", locked_write)
    monkeypatch.setattr(os, "unlink", locked_unlink)


class TestRetirementRemovesTheIndexFirst:
    def test_an_interrupted_retirement_leaves_no_index(self, tmp_path, monkeypatch):
        """Interrupted right after the task file moved aside: the index went first."""
        import utils.review_orchestrator_base as rob

        _materialize(tmp_path, groups=3)
        real = rob._retire_stale_file

        def retire(directory, name, **kwargs):
            moved = real(directory, name, **kwargs)
            if name == "validation_tasks.json":
                raise KeyboardInterrupt("interrupted after the task file")
            return moved

        monkeypatch.setattr(rob, "_retire_stale_file", retire)
        with pytest.raises(KeyboardInterrupt):
            retire_stale_validation_round(str(tmp_path), reason="host index test")
        assert not (tmp_path / "validation_tasks.json").exists()
        assert not (tmp_path / "validation_tasks.index.json").exists()

    def test_an_index_that_cannot_be_removed_retires_nothing(self, tmp_path, monkeypatch, capsys):
        _materialize(tmp_path, groups=3)
        before = sorted(path.name for path in tmp_path.iterdir())
        _lock_the_index(monkeypatch)
        with pytest.raises(SystemExit) as stop:
            retire_stale_validation_round(str(tmp_path), reason="host index test")
        assert stop.value.code == 3
        assert f"Cannot remove {tmp_path / 'validation_tasks.index.json'}" in capsys.readouterr().err
        assert sorted(path.name for path in tmp_path.iterdir()) == before
        assert not list(tmp_path.glob("*.stale*"))


def _called_name(node):
    """The function a call node names, bare (`f(...)`) or as an attribute
    (`steps.f(...)`); ``None`` for anything else."""
    import ast

    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class TestTheViewsLock:
    """One mutex per phase dir orders every invalidate → change → publish sequence."""

    def test_a_refresh_paused_after_its_read_cannot_publish_after_a_repair(
        self, tmp_path, monkeypatch,
    ):
        """A refresh reads the task file and stalls; a `--reaggregate` repairs a
        missing sidecar it cannot rewrite, commits the pointer-free file and is
        interrupted before its own hand-back. The stalled refresh then resumes:
        it must not publish the old pointer beside the committed file."""
        import utils.review_orchestrator_base as rob
        from tests.test_reopen_for_validation import _reaggregate, _round

        plan, out_dir, phase_dir, _state = _round(tmp_path, batches_on_disk=False)
        source = phase_dir / "validation_tasks.json"
        old = _full(phase_dir)
        read, resume = threading.Event(), threading.Event()
        real_build = validation_module.build_host_index

        def stalled_build(document, task_file):
            if not read.is_set():  # the first refresh only: it has read the file
                read.set()
                resume.wait(10)
            return real_build(document, task_file)

        monkeypatch.setattr(validation_module, "build_host_index", stalled_build)
        refresh = threading.Thread(target=refresh_host_index, args=(str(source),))
        refresh.start()
        assert read.wait(10)

        Path(old["batches"][0]["prompt_path"]).unlink()
        real_write = validation_module._write_one_sidecar

        def refuse_sidecars(target, text):
            if target.endswith(".prompt.txt"):
                raise PermissionError(13, "sidecar in use")
            return real_write(target, text)

        class Interrupted(Exception):
            pass

        def interrupted_hand_back(*args, **kwargs):
            raise Interrupted()

        monkeypatch.setattr(validation_module, "_write_one_sidecar", refuse_sidecars)
        monkeypatch.setattr(rob, "dispatch_subagent_tasks", interrupted_hand_back)
        outcome: list = []

        def repair():
            try:
                _reaggregate(plan, out_dir, phase_dir)
            except Interrupted:
                outcome.append("interrupted before its hand-back")
            except BaseException as exc:  # noqa: BLE001 - reported below
                outcome.append(exc)

        repairer = threading.Thread(target=repair)
        repairer.start()
        repairer.join(1.0)  # under the views lock it is still waiting here
        resume.set()
        refresh.join(10)
        repairer.join(10)
        assert not refresh.is_alive() and not repairer.is_alive()
        assert outcome == ["interrupted before its hand-back"]
        full = _full(phase_dir)
        assert "prompt_path" not in full["batches"][0]
        index = phase_dir / "validation_tasks.index.json"
        assert not index.exists() or _describes(_index(phase_dir), full)

    @pytest.mark.parametrize("module", [
        "utils/review_orchestrator_base.py",
        "utils/review_lane.py", "utils/apply_orchestrator_base.py",
    ])
    def test_every_caller_that_changes_a_round_holds_the_lock(self, module):
        """A materialization with its caller's write of the task file, a repair
        with its commit: each call sits inside `with host_views_lock(...)`,
        whether it names the function or reaches it through a lane's steps
        (`steps.materialize_validation_round`). review-code materializes
        through the shared lane (utils/review_lane.py) and has no site of its
        own (`test_review_code_changes_no_round_itself`)."""
        import ast

        guarded = {"materialize_validation_round", "prepare_batched_revalidation_tasks",
                   "ensure_prompt_sidecars"}
        tree = ast.parse((SKILL_DIR / module).read_text(encoding="utf-8"))
        unguarded, seen = [], 0

        def holds_lock(node: ast.With) -> bool:
            return any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "id", None) == "host_views_lock"
                for item in node.items
            )

        def walk(node: ast.AST, held: bool) -> None:
            nonlocal seen
            if isinstance(node, (ast.With, ast.AsyncWith)) and holds_lock(node):
                held = True
            name = _called_name(node)
            if name in guarded:
                seen += 1
                if not held:
                    unguarded.append(f"{module}:{node.lineno} {name}")
            for child in ast.iter_child_nodes(node):
                walk(child, held)

        walk(tree, False)
        assert seen, f"no round-changing call found in {module}"
        assert unguarded == []

    def test_review_code_changes_no_round_itself(self):
        """review-code's round writes are the shared lane's, which the test
        above holds to the lock: a call of its own would need the lock too."""
        import ast

        guarded = {"materialize_validation_round", "prepare_batched_revalidation_tasks",
                   "ensure_prompt_sidecars"}
        tree = ast.parse((SKILL_DIR / "code_review_orchestrator.py").read_text(
            encoding="utf-8"))
        calls = [node.lineno for node in ast.walk(tree) if _called_name(node) in guarded]
        assert calls == []

    def test_a_refresh_waits_for_another_holder_of_the_lock_file(self, tmp_path):
        """The interprocess half: a second open file description stands in for a
        second process holding `.views.lock`."""
        from utils import process_control

        _materialize(tmp_path, groups=3)
        _commit(tmp_path, _shrunk(_full(tmp_path)))
        old_index = (tmp_path / "validation_tasks.index.json").read_bytes()
        holder = process_control.FileMutex(str(tmp_path / ".views.lock"))
        assert holder.acquire(timeout_s=1)
        done = threading.Event()

        def refresh():
            refresh_host_index(str(tmp_path / "validation_tasks.json"))
            done.set()

        worker = threading.Thread(target=refresh)
        try:
            worker.start()
            assert not done.wait(0.5)
            assert (tmp_path / "validation_tasks.index.json").read_bytes() == old_index
        finally:
            holder.release()
        worker.join(10)
        assert done.is_set()
        assert _describes(_index(tmp_path), _full(tmp_path))

    def test_a_holder_that_never_lets_go_stops_the_run(self, tmp_path, monkeypatch, capsys):
        from utils import process_control

        _materialize(tmp_path, groups=3)
        monkeypatch.setattr(validation_module, "HOST_VIEWS_LOCK_TIMEOUT_S", 0.2)
        holder = process_control.FileMutex(str(tmp_path / ".views.lock"))
        assert holder.acquire(timeout_s=1)
        try:
            with pytest.raises(SystemExit) as stop:
                refresh_host_index(str(tmp_path / "validation_tasks.json"))
        finally:
            holder.release()
        assert stop.value.code == 3
        assert f"{tmp_path / '.views.lock'} was not free" in capsys.readouterr().err

    def test_a_lock_file_that_cannot_be_opened_publishes_no_index(self, tmp_path, capsys):
        """It still removes an obsolete one; the host reads the full file."""
        _materialize(tmp_path, groups=3)
        _commit(tmp_path, _shrunk(_full(tmp_path)))
        (tmp_path / ".views.lock").unlink()
        (tmp_path / ".views.lock").mkdir()
        assert refresh_host_index(str(tmp_path / "validation_tasks.json")) is False
        assert not (tmp_path / "validation_tasks.index.json").exists()
        assert "views lock .views.lock cannot be taken" in capsys.readouterr().err


class TestAMatchingIndexIsLeftAlone:
    """An index already right is neither replaced nor removed, so a host holding
    it open (a Windows sharing violation, simulated here) cannot stop a run."""

    def test_through_the_front_door_hand_back(self, tmp_path, monkeypatch, capsys):
        import multi_llm

        _materialize(tmp_path, groups=3)
        source = (tmp_path / "validation_tasks.json").read_bytes()
        view = (tmp_path / "validation_tasks.index.json").read_bytes()
        _lock_the_index(monkeypatch)

        def dispatch(argv, ctx):
            dispatch_subagent_tasks(
                SimpleNamespace(), task_file=str(tmp_path / "validation_tasks.json"),
                phase_dir=str(tmp_path), phase="review-plan",
            )
            return 0

        monkeypatch.setattr(multi_llm, "_dispatch", dispatch)
        monkeypatch.setattr(multi_llm, "_remember_plan_spelling", lambda ctx: None)
        monkeypatch.setattr(multi_llm, "_emit_user_next_step", lambda ctx, code: None)
        assert multi_llm.main([]) == 0
        output = capsys.readouterr()
        assert '"state": "needs-human"' not in output.out
        assert "Cannot remove" not in output.err
        assert (tmp_path / "validation_tasks.json").read_bytes() == source
        assert (tmp_path / "validation_tasks.index.json").read_bytes() == view

    def test_through_a_reaggregate_with_nothing_to_repair(self, tmp_path, monkeypatch, capsys):
        from tests.test_reopen_for_validation import _reaggregate, _round

        plan, out_dir, phase_dir, _state = _round(tmp_path, batches_on_disk=False)
        _hand_back(phase_dir)
        view = (phase_dir / "validation_tasks.index.json").read_bytes()
        _lock_the_index(monkeypatch)
        _reaggregate(plan, out_dir, phase_dir)
        assert "Cannot remove" not in capsys.readouterr().err
        assert (phase_dir / "validation_tasks.index.json").read_bytes() == view
        assert _describes(_index(phase_dir), _full(phase_dir))


class TestTheOtherTaskFiles:
    def test_revalidation_hands_back_its_own_index(self, tmp_path):
        groups = [_group(n) for n in range(6)]
        results = [{"group_index": n, "status": "validation_failed",
                    "error_type": "parsing_error"} for n in range(6)]
        document = prepare_batched_revalidation_tasks(
            groups=groups, validation_results=results, context=_real_shaped_plan(),
            output_dir=str(tmp_path), plan_file=str(tmp_path / "plan.md"),
        )
        _commit(tmp_path, document, "revalidation_tasks.json")
        _hand_back(tmp_path, "revalidation_tasks.json", label="revalidation")
        index = _index(tmp_path, "revalidation_tasks.index.json")
        assert index["index_of"] == "revalidation_tasks.json"
        assert _describes(index, document)
        assert not (tmp_path / "validation_tasks.index.json").exists()

    def test_consolidation_writes_its_own_index_after_its_file(self, tmp_path, monkeypatch):
        groups = [dict(_group(n), suggestions=[dict(_group(n)["suggestions"][0],
                                                    reference="Task 1")])
                  for n in range(4)]
        (tmp_path / "consolidation_tasks.index.json").write_text("{}", encoding="utf-8")
        seen = []
        real = validation_module.write_prompt_sidecars

        def sidecars(*args, **kwargs):
            seen.append((tmp_path / "consolidation_tasks.index.json").exists())
            return real(*args, **kwargs)

        monkeypatch.setattr(validation_module, "write_prompt_sidecars", sidecars)
        document = prepare_consolidation_tasks(
            groups, {"task 1": [0, 1, 2, 3]}, str(tmp_path), "plan.md"
        )
        assert seen == [False], "the old index goes before a sidecar changes"
        index = _index(tmp_path, "consolidation_tasks.index.json")
        assert index["index_of"] == "consolidation_tasks.json"
        assert _describes(index, _full(tmp_path, "consolidation_tasks.json"))
        assert index["total_batches"] == document["total_batches"] >= 1

    def test_the_file_names_follow_the_task_file(self):
        assert host_index_path("/p/validation_tasks.json") == "/p/validation_tasks.index.json"
        for kind in ("validation", "revalidation", "consolidation", "task_generation"):
            assert task_file_for_kind("/p", kind) == os.path.join("/p", f"{kind}_tasks.json")

    def test_a_task_file_without_an_index_is_left_alone(self, tmp_path):
        (tmp_path / "task_generation_tasks.json").write_text("{}", encoding="utf-8")
        assert refresh_host_index(str(tmp_path / "task_generation_tasks.json")) is False
        assert not (tmp_path / "task_generation_tasks.index.json").exists()

    @pytest.mark.parametrize("name", [
        "validation_tasks.index.json", "revalidation_tasks.index.json",
        "consolidation_tasks.index.json", ORCHESTRATOR_SUMMARY_FILENAME,
        ORCHESTRATOR_BATCHES_DIRNAME,
    ])
    def test_no_subagent_result_may_land_on_a_view(self, name, tmp_path):
        assert name in RESERVED_OUTPUT_NAMES or any(
            name.endswith(suffix) for suffix in RESERVED_OUTPUT_SUFFIXES
        )
        with pytest.raises(ResultRejected):
            resolve_output_target(
                {"task_id": "probe", "kind": "salvage", "output_name": name}, str(tmp_path)
            )

    @pytest.mark.parametrize("name", [
        "validation_tasks.index.json", "revalidation_tasks.index.json",
        "consolidation_tasks.index.json",
    ])
    def test_no_reaggregation_glob_reads_an_index_as_a_model_result(self, name):
        from utils.result_files import is_model_result_name

        assert not is_model_result_name(name)


# ---------------------------------------------------------------------------
# The apply phases' summary and batch details
# ---------------------------------------------------------------------------

def _batch(section: str, change: str, group: str = "g1") -> dict:
    return {
        "section_key": section, "prior_changes_id": f"{section}#aaaa",
        "batch_type": "same_section", "prompt": "x" * 4000,
        "suggestions": [{"group_id": group, "title": "The total column",
                         "description": change, "reference": f"## {section}"}],
    }


_OUTPUT = {
    "format_version": 3,
    "plan_file": "/r/plans/feat.md",
    "timestamp": "2026-11-08T00:00:00",
    "batches": [
        dict(_batch("step_2", "y" * 2000), prior_changes_id="step_2#aaaa"),
        dict(_batch("step_5", "y" * 2000, "g2"), prior_changes_id="step_5#bbbb"),
    ],
    "to_apply": [{"group_id": "g1", "description": "y" * 2000}],
    "needs_human_review": [{"group_id": "g3", "title": "Decide"}],
    "skipped_items": [{"group_id": "g4"}],
    "user_skipped_items": [],
    "skipped_count": 1,
    "human_review_config": {"mode": "by-importance", "decision_mode": "interactive"},
    "summary": {"valid_count": 2, "batch_count": 2},
}


def _tasks(n: int, first: int = 0) -> list:
    return [
        {"task_id": f"apply-batch-{first + k}", "batch_index": first + k,
         "task_title": "t", "expected_ids": [f"g{k + 1}"], "prompt": "p"}
        for k in range(n)
    ]


class TestTheApplySummary:
    def test_it_keeps_the_outcome_and_the_ids_and_drops_the_records(self):
        summary = build_orchestrator_summary(_OUTPUT)
        assert summary["summary"] == _OUTPUT["summary"]
        assert summary["needs_human_review"] == _OUTPUT["needs_human_review"]
        assert summary["human_review_config"] == _OUTPUT["human_review_config"]
        assert summary["skipped_items"] == _OUTPUT["skipped_items"]
        assert summary["batches"] == [
            {"batch_index": 0, "section_key": "step_2", "prior_changes_id": "step_2#aaaa"},
            {"batch_index": 1, "section_key": "step_5", "prior_changes_id": "step_5#bbbb"},
        ]
        assert "to_apply" not in summary
        assert "prompt" not in json.dumps(summary["batches"])

    def test_the_write_tasks_add_their_ids_and_detail_paths_in_order(self, tmp_path):
        output = tmp_path / "orchestrator_output.json"
        batches = build_orchestrator_summary(_OUTPUT, _tasks(2, 3), output)["batches"]
        assert [b["task_id"] for b in batches] == ["apply-batch-3", "apply-batch-4"]
        assert [b["batch_index"] for b in batches] == [3, 4]
        assert [b["detail_path"] for b in batches] == [
            str(tmp_path / ORCHESTRATOR_BATCHES_DIRNAME / f"apply-batch-{n}.json")
            for n in (3, 4)
        ]
        assert all("prompt" not in b for b in batches)
        # A task list that does not pair one to one names no task id at all.
        unpaired = build_orchestrator_summary(_OUTPUT, _tasks(1), output)["batches"]
        assert all("task_id" not in b and "detail_path" not in b for b in unpaired)

    def test_a_batch_detail_is_its_record_without_the_prompt(self):
        batch, task = _OUTPUT["batches"][0], _tasks(1)[0]
        detail = build_batch_detail(batch, task)
        assert detail["task_id"] == "apply-batch-0"
        assert detail["suggestions"] == batch["suggestions"]
        for key, value in batch.items():
            if key != "prompt":
                assert detail[key] == value
        assert "prompt" not in detail

    def test_opposite_edits_with_the_same_titles_and_ids_are_told_apart(self, tmp_path):
        """Verification needs the requested edit, not only its title and ids."""
        add, remove = tmp_path / "add", tmp_path / "remove"
        for phase, change in ((add, "Add the 'total' column"),
                              (remove, "Remove the 'total' column")):
            output = dict(_OUTPUT, batches=[_batch("step_2", change)])
            phase.mkdir()
            assert publish_orchestrator_views(
                output, phase / "orchestrator_output.json", _tasks(1)) is not None

        def summary(phase):
            text = (phase / ORCHESTRATOR_SUMMARY_FILENAME).read_text(encoding="utf-8")
            return json.loads(text.replace(json.dumps(str(phase))[1:-1], "/p"))

        assert summary(add) == summary(remove)
        for phase, change in ((add, "Add the 'total' column"),
                              (remove, "Remove the 'total' column")):
            entry = json.loads((phase / ORCHESTRATOR_SUMMARY_FILENAME).read_text())["batches"][0]
            detail = json.loads(Path(entry["detail_path"]).read_text(encoding="utf-8"))
            assert detail["suggestions"][0]["description"] == change
            assert detail["task_id"] == entry["task_id"]

    def test_no_views_without_a_write_task_for_every_batch(self, tmp_path, capsys):
        output = tmp_path / "orchestrator_output.json"
        assert publish_orchestrator_views(_OUTPUT, output) is None
        assert publish_orchestrator_views(_OUTPUT, output, _tasks(1)) is None
        bad_id = [dict(t, task_id="../escape") for t in _tasks(2)]
        assert publish_orchestrator_views(_OUTPUT, output, bad_id) is None
        assert list(tmp_path.iterdir()) == []
        assert "a host reads orchestrator_output.json instead" in capsys.readouterr().err

    def test_every_empty_output_write_writes_its_summary(self, tmp_path, capsys):
        path = write_and_emit_output(dict(_OUTPUT, batches=[]), str(tmp_path))
        summary = json.loads((tmp_path / ORCHESTRATOR_SUMMARY_FILENAME).read_text(encoding="utf-8"))
        assert summary["batches"] == []
        assert json.loads(path.read_text(encoding="utf-8"))["to_apply"] == _OUTPUT["to_apply"]

    def test_rewriting_the_output_first_removes_the_views_of_the_last_one(self, tmp_path, capsys):
        output = tmp_path / "orchestrator_output.json"
        assert publish_orchestrator_views(_OUTPUT, output, _tasks(2)) is not None
        assert len(list((tmp_path / ORCHESTRATOR_BATCHES_DIRNAME).iterdir())) == 2
        write_and_emit_output(dict(_OUTPUT, batches=[]), str(tmp_path))
        assert list((tmp_path / ORCHESTRATOR_BATCHES_DIRNAME).iterdir()) == []
        assert json.loads((tmp_path / ORCHESTRATOR_SUMMARY_FILENAME).read_text())["batches"] == []

    def test_a_failed_write_leaves_no_summary(self, tmp_path, capsys):
        output = tmp_path / "orchestrator_output.json"
        assert publish_orchestrator_views(_OUTPUT, output, _tasks(2)) is not None
        invalidate_orchestrator_views(output)
        bad = dict(_OUTPUT, summary={"unserializable": object()})
        assert publish_orchestrator_views(bad, output, _tasks(2)) is None
        assert not (tmp_path / ORCHESTRATOR_SUMMARY_FILENAME).exists()
        assert list((tmp_path / ORCHESTRATOR_BATCHES_DIRNAME).iterdir()) == []
        assert "a host reads orchestrator_output.json instead" in capsys.readouterr().err

    @_posix_permissions
    def test_a_summary_that_cannot_be_removed_stops_the_rewrite(self, tmp_path, capsys):
        output = tmp_path / "orchestrator_output.json"
        assert publish_orchestrator_views(_OUTPUT, output, _tasks(2)) is not None
        tmp_path.chmod(0o555)
        try:
            with pytest.raises(SystemExit) as stop:
                write_and_emit_output(dict(_OUTPUT, batches=[]), str(tmp_path))
        finally:
            tmp_path.chmod(0o755)
        assert stop.value.code == 3
        assert f"Cannot remove {tmp_path / ORCHESTRATOR_SUMMARY_FILENAME}" in (
            capsys.readouterr().err)
        assert not output.exists()

    def test_the_pointer_names_the_summary_only_when_it_is_there(self, tmp_path):
        output = tmp_path / "orchestrator_output.json"
        assert detail_pointer_lines(output) == [f"Batches and their items: {output}"]
        publish_orchestrator_views(dict(_OUTPUT, batches=[]), output)
        assert detail_pointer_lines(output)[1].endswith(ORCHESTRATOR_SUMMARY_FILENAME)

    @pytest.mark.parametrize("name, verify", [
        ("apply-suggestions.md", "4. **Wait for completion**"),
        ("apply-task-suggestions.md", "4. **Wait for completion**"),
        ("apply-code-fixes.md", "3. **Wait for completion**"),
    ])
    def test_the_instructions_read_the_detail_once_before_applying(self, name, verify):
        text = (SKILL_DIR / "instructions" / name).read_text(encoding="utf-8")
        bind = text.split("1. **Bind the batch prompt**", 1)[1].split("\n", 1)[0]
        assert "detail_path" in bind and bind.index("detail_path") < bind.index("Then run")
        verify_step = text.split(verify, 1)[1].split("**Partial application handling**", 1)[0]
        assert ("its detail file lists: you read it at step 1; read it again only if it is "
                "no longer in your context") in verify_step
        assert ("When there is no summary, read `orchestrator_output.json` and "
                "`write_tasks.json`") in text


# ---------------------------------------------------------------------------
# End to end: a native apply phase
# ---------------------------------------------------------------------------

def _apply_dir(plan: Path) -> Path:
    return plan.parent / plan.stem / "apply-suggestions"


def test_a_native_apply_publishes_views_that_match_its_write_tasks(repo, capsys):  # noqa: F811
    """The ids a host loops over are exactly `write_tasks.json`'s."""
    import apply_suggestions_orchestrator as aso

    plan = write_review(repo.root)["plan"]
    capsys.readouterr()
    assert run_apply(aso, plan, "--yes") == 0
    out = capsys.readouterr().out
    phase_dir = _apply_dir(plan)
    output = json.loads((phase_dir / "orchestrator_output.json").read_text(encoding="utf-8"))
    tasks = json.loads((phase_dir / "write_tasks.json").read_text(encoding="utf-8"))["batches"]
    summary = json.loads((phase_dir / ORCHESTRATOR_SUMMARY_FILENAME).read_text(encoding="utf-8"))

    assert len(summary["batches"]) == len(output["batches"]) == len(tasks) > 1
    for entry, batch, task in zip(summary["batches"], output["batches"], tasks):
        assert entry["task_id"] == task["task_id"]
        assert entry["batch_index"] == task["batch_index"]
        assert entry["expected_ids"] == task["expected_ids"]
        assert entry["prior_changes_id"] == batch["prior_changes_id"] == task["prior_changes_id"]
        detail = json.loads(Path(entry["detail_path"]).read_text(encoding="utf-8"))
        assert Path(entry["detail_path"]) == batch_detail_path(
            phase_dir / "orchestrator_output.json", task["task_id"])
        assert detail["suggestions"] == batch["suggestions"]
        assert "prompt" not in detail
    assert summary["summary"] == output["summary"]
    # The command's output stops at the counts and names both files.
    assert "--- Suggestion" not in out
    assert f"Batches and their items: {phase_dir / 'orchestrator_output.json'}" in out
    assert f"Batch ids, counts and items to decide: {phase_dir / ORCHESTRATOR_SUMMARY_FILENAME}" in out
    # What the host reads for one batch, against the two files it read before.
    largest = max(Path(e["detail_path"]).stat().st_size for e in summary["batches"])
    read = (phase_dir / ORCHESTRATOR_SUMMARY_FILENAME).stat().st_size + largest
    before = sum((phase_dir / n).stat().st_size
                 for n in ("orchestrator_output.json", "write_tasks.json"))
    assert read < before / 3


def test_an_apply_interrupted_after_its_output_leaves_no_view(repo, monkeypatch):  # noqa: F811
    """A rewritten output with the last run's summary beside it is never on disk."""
    import apply_suggestions_orchestrator as aso
    import utils.apply_orchestrator_base as base

    plan = write_review(repo.root)["plan"]
    with contextlib.redirect_stdout(io.StringIO()):
        assert run_apply(aso, plan, "--yes") == 0
    phase_dir = _apply_dir(plan)
    first = json.loads((phase_dir / ORCHESTRATOR_SUMMARY_FILENAME).read_text(encoding="utf-8"))

    def interrupt(self):
        raise KeyboardInterrupt("interrupted after the output was rewritten")

    monkeypatch.setattr(base.ApplyOrchestratorBase, "write_target_backup", interrupt)
    with pytest.raises(KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
        run_apply(aso, plan, "--yes", "--min-priority", "high")

    output = json.loads((phase_dir / "orchestrator_output.json").read_text(encoding="utf-8"))
    assert len(output["batches"]) != len(first["batches"])
    assert not (phase_dir / ORCHESTRATOR_SUMMARY_FILENAME).exists()
    assert list((phase_dir / ORCHESTRATOR_BATCHES_DIRNAME).iterdir()) == []


def test_an_apply_interrupted_after_its_write_tasks_leaves_no_summary(repo, monkeypatch):  # noqa: F811
    """No summary without the task ids `write_tasks.json` gives its batches."""
    import apply_suggestions_orchestrator as aso
    import utils.write_phase_bridge as bridge

    real = bridge.write_task_file_for_phase

    def write_then_interrupt(**kwargs):
        real(**kwargs)
        raise KeyboardInterrupt("interrupted after write_tasks.json")

    monkeypatch.setattr(bridge, "write_task_file_for_phase", write_then_interrupt)
    plan = write_review(repo.root)["plan"]
    with pytest.raises(KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
        run_apply(aso, plan, "--yes")

    phase_dir = _apply_dir(plan)
    tasks = json.loads((phase_dir / "write_tasks.json").read_text(encoding="utf-8"))["batches"]
    assert tasks and all(task["task_id"] for task in tasks)
    assert not (phase_dir / ORCHESTRATOR_SUMMARY_FILENAME).exists()
