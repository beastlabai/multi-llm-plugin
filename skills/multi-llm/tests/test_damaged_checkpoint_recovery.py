"""A fresh review run over a damaged ``.status.json`` recovers it and completes with a warning.

The checkpoint withholds the answers whose safety checks failed or never
finished, so a damaged one used to stop every review run at exit 3 until the
whole phase folder was moved aside. A fresh run now recovers it at its
startup (``review_lane.recover_damaged_checkpoint``): under the phase lock,
with no earlier run left in a registry it can read whole, it sets every earlier
model result and salvage pair aside and retires the validation round under the
views lock **while the damaged checkpoint still refuses the folder**, copies
the damaged bytes to ``.status.json.unreadable-<stamp>``, renames a fresh
checkpoint over the damaged one, and warns (``[CHECKPOINT_RECOVERED]``). The
fresh checkpoint records ``result_files.RECOVERED_KEY``: a provider the
crashed run left behind may still write any file of the folder, so the phase
never trusts a result file on disk again. Every model is withheld, selected,
unselected or never seen; a later run reuses nothing and says so once,
salvage never runs, a ``--reaggregate`` refuses, the round groups the answers
it holds in memory and hands no validation round to the host. A read that
fails is not damage, nothing blocks on a FIFO, a live run is never reset, and
damage found mid-run stops with the shared advice
(docs/development/review-results.md, "Recovering a damaged one").

Offline, on both review lanes, through their real fan-outs, with a fault
injected at each recovery step; only the provider call is stubbed. The late
writers, the registry failures, the refresh race, the FIFOs and the windows
between a check and a later read are Codex's U90c reviews' scenarios; the ask
rows run ``ask_orchestrator.main`` with ``test_ask``'s ``FakeInvoker`` over a
seeded record (ask's own recovery rows are in ``test_ask_withheld_answers``).

Every other phase's checkpoint holds run and progress records only, so the
run that owns the dir replaces a damaged one at its start: the detached
worker before it claims its run (``RunHandle.start``), the foreground run at
its boundary (``detach.settle_foreground_checkpoint``).
"""

from __future__ import annotations

import builtins
import errno
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence
from unittest.mock import patch

import pytest

from tests.test_rejected_slot_answers import (
    OTHER,
    SPEC,
    STEM,
    _failing_invoke,
    _lane,
    _module,
    _nothing_grouped,
    _probe,
    _reject_and_leave_the_file,
    _repairer,
    _run,
    _titles,
)
from tests.test_review_lane_parity import CodeLane, PlanLane
import code_review_orchestrator as cro
import review_tasks_orchestrator as rto
from utils import detach
from utils import held_folder
from utils import process_control
from utils import result_files
from utils import resume_state
from utils import review_lane
from utils import review_orchestrator_base as rob
from utils import review_round
from utils import run_registry
from utils import status_file
from utils import validation
from utils.state_manager import get_or_create_state

LANES = pytest.mark.parametrize("lane_class", [PlanLane, CodeLane],
                                ids=["review-plan", "review-code"])

OTHER_STEM = rob.sanitize_model_name(OTHER)
#: The titles of every answer the folder held before its checkpoint was damaged.
EARLIER = {"Rejected file", "Unselected answer", "Salvaged earlier"}
#: Validation round files a fresh round must never expose as its own.
ROUND_FILES = ("validation.json", "validation_tasks.json", "validation_batch_0.json",
               "validation_batch_0.prompt.txt")


class Killed(BaseException):
    """The process dying at a step: nothing after it runs, no ``except Exception`` catches it."""


def _answers(lane, by_spec: Dict[str, Sequence[str]], calls: List[str]) -> Callable[..., Any]:
    """A provider that writes each spec's findings to its requested file and succeeds."""
    def invoke(**kwargs):
        spec = kwargs["model_spec"]
        calls.append(spec)
        findings = lane.findings(list(by_spec[spec]))
        destination = lane.phase_dir / f"{rob.sanitize_model_name(spec)}.json"
        destination.write_text(json.dumps(findings))
        return {"success": True, "data": findings, "source": "file",
                "output_file": str(destination), "details": {}}
    return invoke


def _damaged_folder(lane, monkeypatch) -> bytes:
    """A rejected answer whose set-aside failed, an unselected model's answer,
    an outstanding salvage pair, a materialized validation round with its
    host index and views lock, and then the checkpoint damaged (truncated).
    Returns the damaged bytes."""
    _reject_and_leave_the_file(lane, monkeypatch)
    phase = lane.phase_dir
    (phase / f"{OTHER_STEM}.json").write_text(json.dumps(lane.findings(["Unselected answer"])))
    target = phase / f"{STEM}.json"
    (phase / f"salvage_{STEM}.json").write_text(json.dumps({
        "model": SPEC, "output_path": str(target), "raw_output": "[]"}))
    (phase / f"salvage_task_{STEM}.json").write_text(json.dumps({"tasks": []}))
    for name in ROUND_FILES:
        (phase / name).write_text("{}" if name.endswith(".json") else "prompt")
    (phase / "validation_tasks.index.json").write_text("{}")
    (phase / ".views.lock").write_text("")
    status = phase / ".status.json"
    status.write_bytes(status.read_bytes()[:-7])
    return status.read_bytes()


def _archives(lane) -> List[Path]:
    return sorted(lane.phase_dir.glob(".status.json.unreadable-*"))


def _quarantined(lane) -> List[str]:
    return sorted(p.name.split(review_lane.QUARANTINED_INFIX)[0]
                  for p in lane.phase_dir.glob(f"*{review_lane.QUARANTINED_INFIX}*"))


def _no_earlier_answer_read(lane) -> bool:
    return _nothing_grouped(lane) or not (set(_titles(lane)) & EARLIER)


def _recovered(lane) -> bool:
    """Whether the phase's checkpoint carries a fresh recovery record."""
    return lane.status().get(result_files.RECOVERED_KEY) == result_files.recovery_record()


#: The one warning a later run of a recovered phase prints.
LATER_RUN_WARNING = "an earlier run replaced this phase's damaged checkpoint"


def _refused_reaggregate(lane, capsys) -> None:
    """A reaggregate of the recovered phase stops at exit 3 with its advice,
    every file left as it was."""
    def files() -> Dict[str, bytes]:
        # The locks are the reaggregate's own.
        return {p.name: p.read_bytes() for p in lane.phase_dir.iterdir()
                if p.is_file() and not p.name.endswith(".lock")}
    before = files()
    capsys.readouterr()
    assert lane.reaggregate("--skip-validation") == 3
    err = capsys.readouterr().err
    assert review_lane.recovered_phase_advice(str(lane.phase_dir)) in err
    assert files() == before


# ---------------------------------------------------------------------------
# Recovery completes, reuses nothing, retires the views and warns
# ---------------------------------------------------------------------------

@LANES
def test_a_fresh_run_recovers_a_damaged_checkpoint_and_reuses_nothing(tmp_path, monkeypatch,
                                                                      capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    damaged = _damaged_folder(lane, monkeypatch)
    capsys.readouterr()

    calls: List[str] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Clean answer"], OTHER: ["Fresh other"]}, calls))
    result = _run(lane, "--force")
    captured = capsys.readouterr()

    assert result == 0, captured.out + captured.err
    assert "[CHECKPOINT_RECOVERED]" in captured.out
    assert "[SKIP]" not in captured.out and calls == [SPEC]
    assert _titles(lane) == ["Clean answer"]
    # The damaged bytes are kept, never deleted; every earlier answer of every
    # model, selected or not, is set aside, the salvage pair included.
    [archive] = _archives(lane)
    assert archive.read_bytes() == damaged
    assert _quarantined(lane) == sorted([
        f"{STEM}.json", f"{OTHER_STEM}.json", f"salvage_{STEM}.json",
        f"salvage_task_{STEM}.json"])
    # The validation round and its host index are retired; the views lock stays.
    for name in ROUND_FILES + ("validation_tasks.index.json",):
        assert not (lane.phase_dir / name).exists(), name
    assert (lane.phase_dir / ".views.lock").exists()
    # The fresh checkpoint is readable, records the warning, and this round's
    # checked answer took the selected model out of `models_unchecked`.
    status = lane.status()
    assert status["round_id"] and status["models_requested"] == [SPEC]
    assert SPEC not in (status.get(result_files.UNCHECKED_MODELS_KEY) or {})
    assert not status.get(result_files.REJECTED_MODELS_KEY)
    [note] = [n for n in status["result_notes"] if archive.name in n]
    assert "reuses none of them" in note
    # Every model stays withheld, this round's included: the record is the
    # fresh one, and nothing in it vouches for a file.
    assert _recovered(lane)
    assert lane.phase_completed()
    # The warning reaches the report's **Result files:** line.
    assert archive.name in (lane.phase_dir / "report.md").read_text(encoding="utf-8")

    # A reaggregate refuses the folder, every file left as it was.
    _refused_reaggregate(lane, capsys)
    assert _titles(lane) == ["Clean answer"]
    # A later run reuses nothing, this round's checked answer included, and
    # says so once; the record survives its opening.
    calls.clear()
    assert _run(lane, "--force", models=[SPEC, OTHER]) == 0
    captured = capsys.readouterr()
    assert "[SKIP]" not in captured.out and "re-running" not in captured.out
    assert sorted(calls) == sorted([SPEC, OTHER])
    assert captured.err.count(LATER_RUN_WARNING) == 1
    assert _titles(lane) == ["Clean answer", "Fresh other"]
    assert _recovered(lane)
    assert not resume_state._kept_on_rerun(lane.phase_dir, lane.phase, SPEC)


@LANES
def test_a_reaggregate_still_refuses_a_damaged_checkpoint(tmp_path, monkeypatch, capsys,
                                                          lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    damaged = _damaged_folder(lane, monkeypatch)
    capsys.readouterr()

    assert lane.reaggregate("--skip-validation") == 3
    err = capsys.readouterr().err
    assert status_file.unreadable_checkpoint_advice(str(lane.phase_dir)) in err
    assert _nothing_grouped(lane)
    # It can only reuse, so it recovers nothing.
    assert (lane.phase_dir / ".status.json").read_bytes() == damaged
    assert _archives(lane) == [] and _quarantined(lane) == []
    assert (lane.phase_dir / f"{STEM}.json").exists()


# ---------------------------------------------------------------------------
# An interruption or a failure at each step leaves a refusal, never a reuse
# ---------------------------------------------------------------------------

def _fault(monkeypatch, step: str, kind: str) -> Callable[[], None]:
    """Make recovery *step* die (``killed``) or fail (``failed``); returns the
    undo, so the next run is healthy."""
    undo: List[Callable[[], None]] = []

    def patch(owner: Any, name: str, value: Any) -> None:
        original = getattr(owner, name)
        monkeypatch.setattr(owner, name, value)
        undo.append(lambda: setattr(owner, name, original))

    def fail(what: str) -> BaseException:
        return Killed(what) if kind == "killed" else OSError(f"injected: {what}")

    if step == "quarantine":
        real = held_folder.rename_in_plan_dir
        moved: List[int] = []

        def rename(path, name):
            if moved:
                raise fail("rename")
            moved.append(1)
            return real(path, name)
        patch(held_folder, "rename_in_plan_dir", rename)
    elif step == "retire":
        def retire(*_a, **_k):
            if kind == "killed":
                raise Killed("retire")
            return []
        patch(rob, "retire_stale_validation_round", retire)
        patch(cro, "retire_stale_validation_round", retire)
    elif step == "archive":
        def archive(*_a, **_k):
            raise fail("archive")
        patch(status_file, "archive_damaged_checkpoint", archive)
    elif step == "publish":
        def publish(*_a, **_k):
            raise fail("publish")
        patch(status_file, "publish_checkpoint", publish)
    elif step == "after-publish":
        real_publish = status_file.publish_checkpoint

        def published(*args, **kwargs):
            real_publish(*args, **kwargs)
            raise Killed("after publish")
        patch(status_file, "publish_checkpoint", published)
    else:  # pragma: no cover
        raise AssertionError(step)
    return lambda: [restore() for restore in reversed(undo)]


STEPS = [
    *[pytest.param(step, kind, id=f"{step}-{kind}")
      for step in ("quarantine", "retire", "archive", "publish")
      for kind in ("killed", "failed")],
    pytest.param("after-publish", "killed", id="after-publish-killed"),
]


@pytest.mark.parametrize("step,kind", STEPS)
@LANES
def test_a_recovery_cut_short_at_any_step_reads_no_earlier_answer(tmp_path, monkeypatch, capsys,
                                                                  lane_class, step, kind):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    damaged = _damaged_folder(lane, monkeypatch)
    capsys.readouterr()
    calls: List[str] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Clean answer"], OTHER: ["Fresh other"]}, calls))

    undo = _fault(monkeypatch, step, kind)
    if kind == "killed":
        with pytest.raises(Killed):
            _run(lane, "--force")
    else:
        assert _run(lane, "--force") == 3
        err = capsys.readouterr().err
        assert status_file.unreadable_checkpoint_advice(str(lane.phase_dir)) in err
    undo()
    capsys.readouterr()
    assert calls == []
    status_path = lane.phase_dir / ".status.json"
    if step in ("quarantine", "retire", "archive"):
        # Before the copy: the damaged checkpoint is still the refusal.
        assert status_path.read_bytes() == damaged and _archives(lane) == []
    elif step == "publish":
        # Copied, never moved: the damaged checkpoint is the refusal until
        # the fresh one replaces it.
        assert status_path.read_bytes() == damaged
        assert [p.read_bytes() for p in _archives(lane)] == [damaged]
    else:
        assert len(_archives(lane)) == 1
        assert lane.status()[result_files.RECOVERED_KEY] == result_files.recovery_record()

    # Neither a reaggregate nor a resume reads an earlier answer.
    assert lane.reaggregate("--skip-validation") in (0, 3)
    assert _no_earlier_answer_read(lane)
    capsys.readouterr()
    assert _run(lane, "--force") == 0
    out = capsys.readouterr().out
    assert "[SKIP]" not in out and calls == [SPEC]
    assert _titles(lane) == ["Clean answer"]
    assert _recovered(lane)
    _refused_reaggregate(lane, capsys)
    assert _titles(lane) == ["Clean answer"]


# ---------------------------------------------------------------------------
# A live run is never reset
# ---------------------------------------------------------------------------

#: A live process holding the lock ``sys.argv[2]`` until its stdin closes.
_HOLDER = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from utils import process_control as pc
lock = pc.PhaseLock(sys.argv[2])
lock.acquire({"holder": "review-code (foreground)", "pid": os.getpid(),
              "process_start_time": pc.process_start_time(os.getpid())})
print("held", flush=True)
sys.stdin.readline()
"""


def _untouched(lane, damaged: bytes) -> None:
    assert (lane.phase_dir / ".status.json").read_bytes() == damaged
    assert _archives(lane) == [] and _quarantined(lane) == []
    assert (lane.phase_dir / f"{STEM}.json").exists()
    assert (lane.phase_dir / "validation_tasks.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="the holder's lock is POSIX advisory here")
@LANES
def test_a_folder_another_live_run_holds_is_never_recovered(tmp_path, monkeypatch, capsys,
                                                            lane_class):
    import subprocess

    lane = _lane(lane_class, tmp_path, monkeypatch)
    damaged = _damaged_folder(lane, monkeypatch)
    skill_root = Path(rob.__file__).resolve().parents[1]
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(skill_root), detach.lock_path(str(lane.phase_dir))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        capsys.readouterr()
        assert _run(lane, "--force") == 3
    finally:
        holder.communicate("\n", timeout=30)
    assert "another run of this phase is in progress" in capsys.readouterr().out
    _untouched(lane, damaged)


@LANES
def test_an_earlier_run_the_registry_keeps_stops_the_recovery(tmp_path, monkeypatch, capsys,
                                                              lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    damaged = _damaged_folder(lane, monkeypatch)
    root = run_registry.resolve_project_root(str(lane.plan))
    run_registry.write_entry(root, {"run_id": "run-earlier", "pid": 1,
                                    "phase_dir": str(lane.phase_dir)})
    capsys.readouterr()

    assert _run(lane, "--force") == 3
    err = capsys.readouterr().err
    assert "run-earlier" in err and "runs --sweep" in err
    _untouched(lane, damaged)


def test_the_recovery_never_runs_without_the_phase_lock(tmp_path, capsys):
    phase = tmp_path / "phase"
    phase.mkdir()
    (phase / ".status.json").write_text('{"models_rejected": ')
    (phase / f"{STEM}.json").write_text("[]")
    with pytest.raises(SystemExit) as stopped:
        review_lane.recover_damaged_checkpoint(
            phase_dir=str(phase), phase_name="review-plan",
            plan_file=str(tmp_path / "plan.md"),
            retire_validation_round=rob.retire_stale_validation_round)
    assert stopped.value.code == 3
    assert "does not hold" in capsys.readouterr().err
    assert (phase / ".status.json").read_text() == '{"models_rejected": '
    assert (phase / f"{STEM}.json").exists()


def _args(**flags):
    import argparse

    values = {"detach": False, "internal_worker": False, "internal_run_id": None,
              "host": None, "force": False, "if_not_running": False, **flags}
    return argparse.Namespace(**values)


def test_a_worker_claims_its_run_in_the_recovered_checkpoint(tmp_path):
    """Run identity and the heartbeat's compare-and-set live in the checkpoint:
    recovered before the worker's claim, the claim lands (over the damaged
    file, ``claim_run`` writes nothing at all)."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    phase = tmp_path / "plan" / "review-plan"
    phase.mkdir(parents=True)
    (phase / ".status.json").write_text("")
    (phase / f"{STEM}.json").write_text("[]")
    with detach.PhaseRun() as run:
        run.take(str(phase), mode="review-plan", plan_file=str(plan),
                 args=_args(internal_worker=True, internal_run_id="run-w"),
                 phase="review-plan")
        recovery = review_lane.recover_damaged_checkpoint(
            phase_dir=str(phase), phase_name="review-plan",
            plan_file=str(plan), run_id="run-w",
            retire_validation_round=rob.retire_stale_validation_round)
        assert recovery is not None
        handle = detach.RunHandle(str(phase), run_id="run-w", mode="review-plan",
                                  plan_file=str(plan), detached=False)
        handle.start()
        try:
            assert status_file.read_run(str(phase))["run_id"] == "run-w"
            status = status_file.read_status(str(phase))
            assert status[result_files.RECOVERED_KEY] == result_files.recovery_record()
            assert not (phase / f"{STEM}.json").exists()
        finally:
            handle.finish(0)
    assert status_file.read_run(str(phase))["state"] == "done"


def test_without_the_recovery_a_worker_refuses_a_damaged_checkpoint_at_once(tmp_path,
                                                                           monkeypatch):
    """Why the recovery comes before the worker's claim, not in ``start_round``:
    over a damaged checkpoint of a phase that withholds answers the claim
    could write nothing, so the worker stops there, the file untouched, and
    its launch reports why at once instead of waiting out the handshake."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    phase = tmp_path / "plan" / "review-plan"
    phase.mkdir(parents=True)
    (phase / ".status.json").write_text("{")
    monkeypatch.setenv(detach.WORKER_ENV, "1")
    handle = detach.RunHandle(str(phase), run_id="run-x", mode="review-plan",
                              plan_file=str(plan), detached=False)
    with pytest.raises(detach.DetachError) as refused:
        handle.start()
    assert ".status.json is damaged (not JSON, 1 bytes)" in str(refused.value)
    assert status_file.unreadable_checkpoint_advice(str(phase)) in str(refused.value)
    assert (phase / ".status.json").read_text() == "{"
    assert status_file.read_run(str(phase)) == {}
    assert not _copies(phase)
    assert not detach.holds_phase_lock(str(phase))
    monkeypatch.setattr(detach, "handshake_timeout", lambda: 0.2)
    with pytest.raises(detach.DetachError) as launch:
        detach._await_handshake(str(phase), "run-x")
    assert ".status.json is damaged" in str(launch.value)


# ---------------------------------------------------------------------------
# Every other phase: the run that owns the dir replaces a damaged checkpoint
# ---------------------------------------------------------------------------

TORN = '{"phase": "generate-tasks", "state": "comp'


def _copies(phase) -> List[Path]:
    """The damaged checkpoint's copies in *phase* (a directory)."""
    return sorted(p for p in Path(phase).iterdir()
                  if p.name.startswith(".status.json" + status_file.UNREADABLE_ARCHIVE_INFIX))


def _phase_dir(tmp_path, name):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    phase = tmp_path / "plan" / name
    phase.mkdir(parents=True)
    return plan, phase


def test_replacing_a_damaged_checkpoint_keeps_its_bytes_and_reads_none(tmp_path):
    _plan, phase = _phase_dir(tmp_path, "generate-tasks")
    (phase / ".status.json").write_text(TORN)
    damage = status_file.checkpoint_damage(str(phase))
    archived = status_file.replace_damaged_checkpoint(str(phase), damage,
                                                      phase="generate-tasks")
    assert [Path(archived)] == _copies(phase) and Path(archived).read_text() == TORN
    status = status_file.load_status(str(phase))
    assert set(status) == {"phase", "updated_at"} and status["phase"] == "generate-tasks"


@pytest.mark.parametrize("phase_name", [*status_file.WITHHELD_RECORD_PHASES, None])
def test_a_phase_that_withholds_answers_is_never_replaced_whole(tmp_path, phase_name):
    (tmp_path / ".status.json").write_text("{")
    damage = status_file.checkpoint_damage(str(tmp_path))
    with pytest.raises(ValueError):
        status_file.replace_damaged_checkpoint(str(tmp_path), damage, phase=phase_name)
    assert (tmp_path / ".status.json").read_text() == "{" and not _copies(tmp_path)


def test_bytes_that_changed_after_the_check_are_not_replaced(tmp_path):
    (tmp_path / ".status.json").write_text("{")
    damage = status_file.checkpoint_damage(str(tmp_path))
    (tmp_path / ".status.json").write_text("[")
    with pytest.raises(OSError):
        status_file.replace_damaged_checkpoint(str(tmp_path), damage, phase="implement")
    assert (tmp_path / ".status.json").read_text() == "[" and not _copies(tmp_path)


def test_the_withheld_phases_are_the_phases_of_the_review_and_ask_modes():
    import modes

    review = {m.phase for m in modes.MODES
              if m.name in ("review-plan", "review-tasks", "review-code",
                            "review-commits", "ask")}
    assert set(status_file.WITHHELD_RECORD_PHASES) == review


@pytest.mark.parametrize("mode,phase_name", [
    ("generate-tasks", "generate-tasks"), ("implement", "implement"),
    ("apply-suggestions", "apply-suggestions"), ("apply-code-fixes", "apply-code-fixes"),
    ("full", "full"), ("improve-plan", "improve-plan"),
])
def test_a_worker_replaces_a_damaged_checkpoint_and_claims_its_run(tmp_path, capsys,
                                                                  mode, phase_name):
    """SR-1: a detached run of any other phase claims its run over a damaged
    checkpoint (it used to claim nothing, and its launch waited out the
    handshake while the worker ran the phase)."""
    plan, phase = _phase_dir(tmp_path, phase_name)
    (phase / ".status.json").write_text(TORN)
    handle = detach.RunHandle(str(phase), run_id="run-g", mode=mode,
                              plan_file=str(plan), detached=False)
    handle.start()
    try:
        assert status_file.read_run(str(phase))["run_id"] == "run-g"
        assert [p.read_text() for p in _copies(phase)] == [TORN]
        assert status_file.read_status(str(phase))["phase"] == phase_name
    finally:
        handle.finish(0)
    assert status_file.read_run(str(phase))["state"] == "done"
    assert ".status.json is damaged (not JSON" in capsys.readouterr().err


def test_a_worker_refuses_a_checkpoint_it_cannot_read_at_once(tmp_path, monkeypatch):
    """Not proven damaged (a directory at the name): left as it is, and the
    launch is told why instead of waiting out the handshake."""
    plan, phase = _phase_dir(tmp_path, "generate-tasks")
    (phase / ".status.json").mkdir()
    monkeypatch.setenv(detach.WORKER_ENV, "1")
    handle = detach.RunHandle(str(phase), run_id="run-u", mode="generate-tasks",
                              plan_file=str(plan), detached=False)
    with pytest.raises(detach.DetachError) as refused:
        handle.start()
    message = str(refused.value)
    assert "could not be read" in message and "is not a file" in message
    assert status_file.unreadable_checkpoint_advice(
        str(phase), phase="generate-tasks") in message
    assert (phase / ".status.json").is_dir() and not _copies(phase)
    monkeypatch.setattr(detach, "handshake_timeout", lambda: 0.2)
    with pytest.raises(detach.DetachError):
        detach._await_handshake(str(phase), "run-u")


def _foreground(**flags):
    return _args(**flags)


def test_a_foreground_run_replaces_a_damaged_checkpoint_at_its_boundary(tmp_path, capsys):
    plan, phase = _phase_dir(tmp_path, "generate-tasks")
    (phase / ".status.json").write_text(TORN)
    assert rob.maybe_detach(args=_foreground(), mode="generate-tasks", phase_dir=str(phase),
                            plan_file=str(plan), phase="generate-tasks") is None
    assert [p.read_text() for p in _copies(phase)] == [TORN]
    status_file.update_status(str(phase), {"state": "complete"})
    assert status_file.load_status(str(phase))["state"] == "complete"
    assert "a fresh checkpoint replaced it" in capsys.readouterr().err


@pytest.mark.parametrize("mode,phase_name", [
    ("review-plan", "review-plan"), ("review-tasks", "review-tasks"),
    ("review-code", "code-review"), ("ask", "ask"), ("review-code", None),
])
def test_the_foreground_boundary_leaves_a_withheld_phase_to_its_recovery(tmp_path, capsys,
                                                                        mode, phase_name):
    plan, phase = _phase_dir(tmp_path, phase_name or "code-review")
    (phase / ".status.json").write_text("{")
    assert rob.maybe_detach(args=_foreground(), mode=mode, phase_dir=str(phase),
                            plan_file=str(plan), phase=phase_name) is None
    assert (phase / ".status.json").read_text() == "{" and not _copies(phase)
    assert capsys.readouterr().err == ""


#: A detached worker of generate-tasks in a process of its own: it starts its
#: run over the phase dir and, when it holds the lock and has found the
#: checkpoint damaged, says so and waits for a line before it replaces it.
_WORKER = """
import sys
from utils import detach
phase, plan, run_id = sys.argv[1:]
real = detach._replace_damaged
def paused(*args, **kwargs):
    print("WORKER_HAS_LOCK_AND_DAMAGE", flush=True)
    sys.stdin.readline()
    return real(*args, **kwargs)
detach._replace_damaged = paused
handle = detach.RunHandle(phase, run_id=run_id, mode="generate-tasks", plan_file=plan,
                          detached=False)
try:
    handle.start()
except detach.DetachError as exc:
    print("WORKER_REFUSED: " + " ".join(str(exc).split()), flush=True)
else:
    print("WORKER_PUBLISHED", flush=True)
    handle.finish(0)
"""


def _worker(phase, plan, run_id):
    import subprocess

    return subprocess.Popen(
        [sys.executable, "-c", _WORKER, str(phase), str(plan), run_id],
        cwd=str(Path(__file__).resolve().parents[1]), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_the_foreground_holds_the_phase_lock_from_its_decision_to_its_replacement(
        tmp_path, monkeypatch):
    """Codex F04 review 1: a worker that starts between the foreground's
    decision to replace a damaged checkpoint and the replacement must never
    find the lock free and the damage still there. The probe the foreground
    used dropped its lock at once: the worker took the lock and found the
    damage, the foreground replaced the file under it, and the worker refused
    ("changed after it was found damaged") and never published its run."""
    plan, phase = _phase_dir(tmp_path, "generate-tasks")
    (phase / ".status.json").write_text(TORN)
    seen: List[str] = []
    children = []
    real = detach._replace_damaged

    def replace_while_a_worker_starts(*args, **kwargs):
        child = _worker(phase, plan, "run-early")
        children.append(child)
        seen.append(child.stdout.readline().strip())
        return real(*args, **kwargs)

    monkeypatch.setattr(detach, "_replace_damaged", replace_while_a_worker_starts)
    try:
        detach.settle_foreground_checkpoint(str(phase), phase="generate-tasks")
        children[0].communicate("\n", timeout=30)
        assert seen and seen[0] != "WORKER_HAS_LOCK_AND_DAMAGE", seen
        assert seen[0].startswith("WORKER_REFUSED: ERROR: another run owns this phase dir")
        assert [p.read_text() for p in _copies(phase)] == [TORN]
        assert process_control.lock_is_free(detach.lock_path(str(phase)))
        later = _worker(phase, plan, "run-later")
        children.append(later)
        out, err = later.communicate("\n", timeout=30)
        assert "WORKER_PUBLISHED" in out, (out, err)
        assert status_file.read_run(str(phase))["run_id"] == "run-later"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


def test_the_foreground_leaves_a_damaged_checkpoint_to_the_process_holding_its_lock(
        tmp_path, capsys):
    """A worker in another process holds the lock and has found the damage:
    the foreground leaves the file to it, and the worker publishes its run."""
    plan, phase = _phase_dir(tmp_path, "generate-tasks")
    (phase / ".status.json").write_text(TORN)
    worker = _worker(phase, plan, "run-w")
    try:
        assert worker.stdout.readline().strip() == "WORKER_HAS_LOCK_AND_DAMAGE"
        detach.settle_foreground_checkpoint(str(phase), phase="generate-tasks")
        assert (phase / ".status.json").read_text() == TORN and not _copies(phase)
        err = capsys.readouterr().err
        assert "another process holds" in err and "This run goes on" in err
        out, _err = worker.communicate("\n", timeout=30)
        assert "WORKER_PUBLISHED" in out
        assert [p.read_text() for p in _copies(phase)] == [TORN]
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_a_foreground_run_goes_on_over_a_checkpoint_it_cannot_read(tmp_path, capsys):
    plan, phase = _phase_dir(tmp_path, "implement")
    (phase / ".status.json").mkdir()
    detach.settle_foreground_checkpoint(str(phase), phase="implement")
    assert (phase / ".status.json").is_dir()
    assert "could not be read" in capsys.readouterr().err


def test_a_foreground_implement_replaces_a_damaged_checkpoint(tmp_path):
    import implement_orchestrator

    plan, phase = _phase_dir(tmp_path, "implement")
    (phase / ".status.json").write_text(TORN)
    assert implement_orchestrator.maybe_detach_implement_phase(
        _foreground(), plan_path=plan) is None
    assert [p.read_text() for p in _copies(phase)] == [TORN]


@pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry-run"])
def test_a_foreground_apply_phase_replaces_a_damaged_checkpoint_but_not_on_a_dry_run(
        tmp_path, dry_run):
    from utils.apply_orchestrator_base import ApplyOrchestratorBase

    plan, phase = _phase_dir(tmp_path, "apply-suggestions")
    (phase / ".status.json").write_text(TORN)
    fake = SimpleNamespace(args=_foreground(dry_run=dry_run), out_dir=str(phase.parent),
                           phase_name="apply-suggestions", confirmation_phase_name=None,
                           plan_path=str(plan))
    fake._is_detach_invoker = lambda: ApplyOrchestratorBase._is_detach_invoker(fake)
    assert ApplyOrchestratorBase.maybe_detach_write_phase(fake) is None
    assert len(_copies(phase)) == (0 if dry_run else 1)


# ---------------------------------------------------------------------------
# Damage found mid-run stops with the shared advice; it is not recovered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doing", ["check", "opening", "publish"])
def test_mid_run_damage_gets_the_shared_advice_not_force(tmp_path, capsys, doing):
    claim = review_round.start_round(tmp_path)
    status = tmp_path / ".status.json"
    status.write_bytes(status.read_bytes()[:-3])
    damaged = status.read_bytes()
    with pytest.raises(review_round.RoundTakenOver) as stopped:
        if doing == "check":
            claim.check("write grouped.json")
        elif doing == "opening":
            with claim.opening():
                pass
        else:
            claim.publish(tmp_path / "report.md", "x", "write report.md")
    assert stopped.value.code == 3
    captured = capsys.readouterr()
    assert status_file.unreadable_checkpoint_advice(str(tmp_path)) in captured.err
    assert "--force" not in captured.out
    assert status.read_bytes() == damaged and not (tmp_path / "report.md").exists()


def test_a_readable_checkpoint_without_the_round_is_still_a_takeover(tmp_path, capsys):
    claim = review_round.start_round(tmp_path)
    status_file.remove_status_keys(str(tmp_path), [review_round.ROUND_KEY])
    with pytest.raises(review_round.RoundTakenOver):
        claim.check("write grouped.json")
    assert "--force" in capsys.readouterr().out


@LANES
def test_damage_during_the_fanout_stops_the_run_and_is_left_in_place(tmp_path, monkeypatch,
                                                                     capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    answer = _answers(lane, {SPEC: ["Mid-run answer"]}, [])
    seen: Dict[str, bytes] = {}

    def damaging(**kwargs):
        result = answer(**kwargs)
        status = lane.phase_dir / ".status.json"
        status.write_bytes(status.read_bytes()[:-5])
        seen["damaged"] = status.read_bytes()
        return result
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output", damaging)

    assert _run(lane) == 3
    captured = capsys.readouterr()
    assert status_file.unreadable_checkpoint_advice(str(lane.phase_dir)) in captured.err
    assert "re-run with --force" not in captured.out
    assert "[CHECKPOINT_RECOVERED]" not in captured.out
    assert (lane.phase_dir / ".status.json").read_bytes() == seen["damaged"]
    assert _archives(lane) == [] and _quarantined(lane) == []
    assert not lane.phase_completed()


# ---------------------------------------------------------------------------
# Controls: healthy runs, readable older checkpoints, transient read failures
# ---------------------------------------------------------------------------

@LANES
def test_a_healthy_checkpoint_is_not_recovered(tmp_path, monkeypatch, capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    calls: List[str] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Clean answer"]}, calls))
    assert _run(lane) == 0
    assert _run(lane, "--force") == 0
    out = capsys.readouterr().out
    assert "[CHECKPOINT_RECOVERED]" not in out
    assert "[SKIP]" in out and calls == [SPEC]
    assert _archives(lane) == [] and _quarantined(lane) == []


@LANES
def test_a_readable_older_checkpoint_resumes_as_before(tmp_path, monkeypatch, capsys,
                                                       lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    calls: List[str] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Clean answer"]}, calls))
    assert _run(lane) == 0
    # A checkpoint from before round ids and the withheld maps.
    status_file.remove_status_keys(str(lane.phase_dir), [
        review_round.ROUND_KEY, result_files.REJECTED_MODELS_KEY,
        result_files.UNCHECKED_MODELS_KEY])
    capsys.readouterr()
    assert _run(lane, "--force") == 0
    out = capsys.readouterr().out
    assert "[CHECKPOINT_RECOVERED]" not in out
    assert "[SKIP]" in out and calls == [SPEC]
    assert _titles(lane) == ["Clean answer"]
    assert _archives(lane) == []


def _failing_reads(monkeypatch, failures: int) -> List[str]:
    """The next *failures* binary reads of a ``.status.json`` fail with EIO."""
    failed: List[str] = []
    real_open = builtins.open

    def flaky(file, mode="r", *args, **kwargs):
        if str(file).endswith(".status.json") and mode == "rb" and len(failed) < failures:
            failed.append(str(file))
            raise OSError(5, "Input/output error")
        return real_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", flaky)
    return failed


def test_a_read_that_fails_once_is_not_damage(tmp_path, monkeypatch):
    (tmp_path / ".status.json").write_text("{}")
    failed = _failing_reads(monkeypatch, 1)
    assert status_file.checkpoint_damage(str(tmp_path)) is None
    assert failed


@LANES
def test_a_read_that_keeps_failing_stops_the_run_and_changes_nothing(tmp_path, monkeypatch,
                                                                     capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    damaged = _damaged_folder(lane, monkeypatch)
    capsys.readouterr()
    calls: List[str] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Clean answer"]}, calls))
    failed = _failing_reads(monkeypatch, status_file.CHECKPOINT_READ_ATTEMPTS)

    assert _run(lane, "--force") == 3
    err = capsys.readouterr().err
    assert len(failed) == status_file.CHECKPOINT_READ_ATTEMPTS
    assert "nothing shows that its bytes are damaged" in err
    assert status_file.unreadable_checkpoint_advice(str(lane.phase_dir)) in err
    assert calls == []
    _untouched(lane, damaged)


# ---------------------------------------------------------------------------
# The pieces: damage told apart, the archive, the publication
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content,detail", [
    (b"", "empty"),
    (b'{"models_rejected": ', "not JSON"),
    (b"[1, 2]", "a JSON list, not an object"),
    (b"null", "a JSON NoneType, not an object"),
    (b"\xff\xfe{}", "not UTF-8 text"),
])
def test_bytes_that_are_not_a_json_object_are_damage(tmp_path, content, detail):
    (tmp_path / ".status.json").write_bytes(content)
    damage = status_file.checkpoint_damage(str(tmp_path))
    assert damage is not None and damage.detail == detail and damage.size == len(content)


def test_absent_and_readable_checkpoints_are_not_damage(tmp_path):
    assert status_file.checkpoint_damage(str(tmp_path)) is None
    (tmp_path / ".status.json").write_text('{"state": "complete"}')
    assert status_file.checkpoint_damage(str(tmp_path)) is None


@pytest.mark.skipif(os.name == "nt", reason="links and FIFOs")
@pytest.mark.parametrize("shape", ["link", "fifo", "folder"])
def test_anything_but_a_file_is_never_damage(tmp_path, shape):
    status = tmp_path / ".status.json"
    if shape == "link":
        (tmp_path / "elsewhere").write_text("{")
        status.symlink_to(tmp_path / "elsewhere")
    elif shape == "fifo":
        os.mkfifo(status)
    else:
        status.mkdir()
    with pytest.raises(status_file.CheckpointUnreadable):
        status_file.checkpoint_damage(str(tmp_path))


def test_bytes_that_change_between_reads_are_not_damage(tmp_path, monkeypatch):
    status = tmp_path / ".status.json"
    status.write_text("{")
    real_open = builtins.open
    reads: List[int] = []

    def changing(file, mode="r", *args, **kwargs):
        if str(file) == str(status) and mode == "rb":
            reads.append(1)
            status.write_text("{" * (len(reads) + 1))
        return real_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", changing)
    with pytest.raises(status_file.CheckpointUnreadable):
        status_file.checkpoint_damage(str(tmp_path))


def test_the_archive_copies_the_bytes_and_never_a_changed_file(tmp_path):
    status = tmp_path / ".status.json"
    status.write_text("{")
    damage = status_file.checkpoint_damage(str(tmp_path))
    status.write_text("{{")
    with pytest.raises(OSError):
        status_file.archive_damaged_checkpoint(str(tmp_path), damage, "T1")
    assert status.read_text() == "{{" and not list(tmp_path.glob(".status.json.unreadable-*"))

    damage = status_file.checkpoint_damage(str(tmp_path))
    (tmp_path / ".status.json.unreadable-T1").write_text("earlier")
    archived = status_file.archive_damaged_checkpoint(str(tmp_path), damage, "T1")
    assert Path(archived).name == ".status.json.unreadable-T1-2"
    # A copy: the damaged checkpoint stays, refusing the folder, until the
    # fresh one is renamed over it.
    assert Path(archived).read_text() == "{{" and status.read_text() == "{{"
    assert (tmp_path / ".status.json.unreadable-T1").read_text() == "earlier"


def test_a_publication_that_does_not_read_back_raises(tmp_path, monkeypatch):
    document = status_file.publish_checkpoint(str(tmp_path), {"state": "models_running"})
    assert status_file.load_status(str(tmp_path)) == document
    monkeypatch.setattr(status_file, "load_status", lambda *_a, **_k: {"other": 1})
    with pytest.raises(OSError):
        status_file.publish_checkpoint(str(tmp_path), {"state": "models_running"})


def test_the_fresh_checkpoint_replaces_only_the_damaged_bytes(tmp_path):
    status = tmp_path / ".status.json"
    status.write_text("{")
    damage = status_file.checkpoint_damage(str(tmp_path))
    status.write_text("{{")
    with pytest.raises(OSError):
        status_file.publish_checkpoint(str(tmp_path), {"state": "x"}, replacing=damage)
    assert status.read_text() == "{{"
    damage = status_file.checkpoint_damage(str(tmp_path))
    published = status_file.publish_checkpoint(str(tmp_path), {"state": "x"},
                                               replacing=damage)
    assert status_file.load_status(str(tmp_path)) == published


def test_the_rename_below_a_plan_dir_goes_through_its_held_folder(tmp_path):
    plan_dir = tmp_path / "plans"
    phase = plan_dir / "feature" / "review-plan"
    phase.mkdir(parents=True)
    (phase / "a.json").write_text("[]")
    held_folder.anchor_plan_dir(plan_dir)
    renamed = held_folder.rename_in_plan_dir(phase / "a.json", "a.json.quarantined-T")
    assert renamed == phase / "a.json.quarantined-T" and renamed.read_text() == "[]"
    with pytest.raises(OSError):
        held_folder.rename_in_plan_dir(phase / "missing.json", "x")


def test_the_advice_tells_a_fresh_run_recovers_and_keeps_the_folder_fallback():
    advice = status_file.unreadable_checkpoint_advice("/work/out/review-plan")
    assert "run the phase again without --reaggregate" in advice
    assert ".status.json.unreadable-<time>" in advice
    assert "move the whole phase folder /work/out/review-plan aside" in advice
    assert "Never move .status.json aside on its own" in advice


def test_the_lock_check_reads_this_processes_hold(tmp_path):
    lock = process_control.PhaseLock(detach.lock_path(str(tmp_path)))
    assert not detach.holds_phase_lock(str(tmp_path))
    lock.acquire({"pid": os.getpid()})
    try:
        assert detach.holds_phase_lock(str(tmp_path))
    finally:
        lock.release()


def test_a_detached_workers_stop_is_reported_to_its_launch(tmp_path, monkeypatch, capsys):
    phase = tmp_path / "phase"
    phase.mkdir()
    (phase / ".status.json").write_text("{")
    monkeypatch.setenv(detach.WORKER_ENV, "1")
    with pytest.raises(SystemExit) as stopped:
        review_lane.recover_damaged_checkpoint(
            phase_dir=str(phase), phase_name="review-plan",
            plan_file=str(tmp_path / "plan.md"), run_id="run-launched")
    assert stopped.value.code == 3
    capsys.readouterr()
    refusal = detach._read_worker_refusal(str(phase), "run-launched")
    assert refusal and "does not hold" in refusal["message"]


def test_a_phase_without_a_validation_round_keeps_its_own_keys(tmp_path):
    """The helper a phase with more in its checkpoint than a review round can
    call (ask's ``question``): its keys go into the fresh checkpoint, and no
    validation round is retired."""
    phase = tmp_path / "plan" / "ask" / "q"
    phase.mkdir(parents=True)
    (phase / ".status.json").write_text("[]")
    (phase / f"{STEM}.json").write_text("[]")
    (phase / "validation.json").write_text("{}")
    lock = process_control.PhaseLock(detach.lock_path(str(phase)))
    lock.acquire({"pid": os.getpid()})
    try:
        recovery = review_lane.recover_damaged_checkpoint(
            phase_dir=str(phase), phase_name="ask",
            plan_file=str(tmp_path / "plan.md"), fields={"question": "Why?"})
    finally:
        lock.release()
    assert recovery is not None and recovery.retired == ()
    status = status_file.read_status(str(phase))
    assert status["question"] == "Why?" and status["phase"] == "ask"
    assert (phase / "validation.json").exists()
    assert not (phase / f"{STEM}.json").exists()


@LANES
def test_a_plain_run_recovers_too(tmp_path, monkeypatch, capsys, lane_class):
    """No ``--force``: review-code's partial-completion guard leaves a
    checkpoint it cannot read to the startup's recovery instead of naming
    the earlier round's task file another round's."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _damaged_folder(lane, monkeypatch)
    capsys.readouterr()
    calls: List[str] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Clean answer"]}, calls))
    assert _run(lane) == 0
    out = capsys.readouterr().out
    assert "[CHECKPOINT_RECOVERED]" in out and "Partial completion" not in out
    assert calls == [SPEC] and _titles(lane) == ["Clean answer"]


# ---------------------------------------------------------------------------
# A provider the crashed run left behind writes after the recovery (Codex 1)
# ---------------------------------------------------------------------------

def _late_writer(lane, monkeypatch, timing: str, title: str) -> Callable[[], None]:
    """An orphaned provider of the crashed run writing the unselected
    model's answer: right after the recovery set the folder's answers aside
    (``after-quarantine``), or once the recovery's run is over (the returned
    call, for ``after-recovery``)."""
    def write() -> None:
        (lane.phase_dir / f"{OTHER_STEM}.json").write_text(
            json.dumps(lane.findings([title])))
    if timing == "after-quarantine":
        real = review_lane._quarantine_earlier_answers

        def quarantine(*args, **kwargs):
            moved = real(*args, **kwargs)
            write()
            return moved
        monkeypatch.setattr(review_lane, "_quarantine_earlier_answers", quarantine)
        return lambda: None
    return write


@pytest.mark.parametrize("timing", ["after-quarantine", "after-recovery"])
@pytest.mark.parametrize("earlier_file", [True, False], ids=["file-at-enumeration",
                                                             "no-file-at-enumeration"])
@LANES
def test_a_late_answer_of_a_model_the_recovery_did_not_select_is_never_reused(
        tmp_path, monkeypatch, capsys, lane_class, timing, earlier_file):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    calls: List[str] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output", _answers(
        lane, {SPEC: ["Clean answer"], OTHER: ["Fresh other"]}, calls))
    # An earlier attempt of the unselected model, which review-code's scope
    # ledger records; then the crash: its file there or not yet written, and
    # the checkpoint damaged.
    assert _run(lane, models=[OTHER]) == 0
    if not earlier_file:
        (lane.phase_dir / f"{OTHER_STEM}.json").unlink()
    status = lane.phase_dir / ".status.json"
    status.write_bytes(status.read_bytes()[:-7])
    # (A title of another length: the fresh answer must show a new file stamp.)
    late = _late_writer(lane, monkeypatch, timing, "Late answer of the crashed run")
    capsys.readouterr()
    calls.clear()

    assert _run(lane, "--force") == 0
    assert "[CHECKPOINT_RECOVERED]" in capsys.readouterr().out
    late()
    assert (lane.phase_dir / f"{OTHER_STEM}.json").exists()

    # `util resume-state` and a later run's resume test agree: not reusable.
    assert not resume_state._kept_on_rerun(lane.phase_dir, lane.phase, OTHER)
    # Nor is the answer this round checked: a later run dispatches its model
    # again, and the record survives that round's opening.
    capsys.readouterr()
    calls.clear()
    assert _run(lane, "--force") == 0
    assert "[SKIP]" not in capsys.readouterr().out and calls == [SPEC]
    assert _recovered(lane)
    calls.clear()
    assert _run(lane, "--force", models=[OTHER]) == 0
    out = capsys.readouterr().out
    assert "[SKIP]" not in out and calls == [OTHER]
    assert _titles(lane) == ["Fresh other"]
    # A clean, checked run of the model makes no file of it reusable either.
    for spec in (SPEC, OTHER):
        assert not resume_state._kept_on_rerun(lane.phase_dir, lane.phase, spec)


def _fresh_checkpoint_write_fails(monkeypatch) -> None:
    """The fresh checkpoint's write fails; every other write (the copy of the
    damaged bytes among them) lands."""
    real = status_file.write_in_plan_dir

    def write(path, *args, **kwargs):
        if os.path.basename(os.fspath(path)) == status_file.STATUS_FILENAME:
            raise OSError(errno.ENOSPC, "injected: no space left", os.fspath(path))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(status_file, "write_in_plan_dir", write)


@LANES
def test_a_failed_publication_keeps_the_damaged_checkpoint_refusing_late_answers(
        tmp_path, monkeypatch, capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    damaged = _damaged_folder(lane, monkeypatch)
    calls: List[str] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Clean answer"]}, calls))
    _fresh_checkpoint_write_fails(monkeypatch)
    capsys.readouterr()

    assert _run(lane, "--force") == 3
    monkeypatch.undo()
    lane = _lane(lane_class, tmp_path, monkeypatch)
    assert calls == []
    assert (lane.phase_dir / ".status.json").read_bytes() == damaged
    assert [p.read_bytes() for p in _archives(lane)] == [damaged]
    # The selected model's provider, left running by the crash, writes now.
    (lane.phase_dir / f"{STEM}.json").write_text(json.dumps(lane.findings(["Late answer"])))
    capsys.readouterr()
    assert lane.reaggregate("--skip-validation") == 3
    assert status_file.unreadable_checkpoint_advice(str(lane.phase_dir)) in (
        capsys.readouterr().err)
    assert _nothing_grouped(lane)
    assert (lane.phase_dir / f"{STEM}.json").exists()


# ---------------------------------------------------------------------------
# The recovered checkpoint's disposition, as every reader reads it
# ---------------------------------------------------------------------------

def _recovered_phase(tmp_path, **status) -> Path:
    phase = tmp_path / "phase"
    phase.mkdir()
    status_file.write_status(str(phase), {
        "phase": "review-plan", result_files.RECOVERED_KEY: result_files.recovery_record(),
        **status})
    return phase


def test_a_recovered_checkpoint_withholds_every_model_whatever_its_files_hold(tmp_path):
    phase = _recovered_phase(tmp_path, **{
        result_files.REJECTED_MODELS_KEY: {"re:jected": "home check"}})
    withheld = result_files.withheld_models(phase)
    assert withheld and not withheld.unreadable
    for spec in (SPEC, OTHER, "never:seen"):
        assert spec in withheld and withheld.recorded(spec)
        assert withheld[spec].key == result_files.RECOVERED_KEY
        assert "damaged checkpoint" in withheld[spec].phrase
    # A map's record names the model first.
    assert withheld["re:jected"].rejected
    assert withheld.owner("never_seen_stem") == "never_seen_stem"
    assert withheld.owner(STEM) == STEM

    # Clearing `models_unchecked` (a guard's pass) lets nothing through.
    assert result_files.clear_unchecked_model(phase, SPEC)
    assert SPEC in result_files.withheld_models(phase)
    # Every file is set aside, whatever it holds.
    answer = phase / f"{STEM}.json"
    answer.write_text('[{"title": "Checked"}]')
    late = phase / f"{OTHER_STEM}.json"
    late.write_text("[]")
    assert result_files.withhold_results(phase, [str(answer), str(late)]) == []
    assert (phase / f"{STEM}.json.unchecked").exists()
    assert (phase / f"{OTHER_STEM}.json.unchecked").exists()
    # The operator's acceptance is the one way through, for that model only.
    assert result_files.clear_withheld_model(phase, SPEC)
    withheld = result_files.withheld_models(phase)
    assert SPEC not in withheld and withheld.owner(STEM) is None
    assert OTHER in withheld and withheld.owner(OTHER_STEM) == OTHER_STEM
    assert result_files.recorded_recovery(phase) == {
        "reason": result_files.RECOVERED_REASON, result_files.ACCEPTED_FIELD: [SPEC]}


def test_a_recovery_record_of_the_wrong_shape_still_withholds_every_model(tmp_path):
    phase = _recovered_phase(tmp_path, **{result_files.RECOVERED_KEY: "garbled"})
    withheld = result_files.withheld_models(phase)
    assert SPEC in withheld and withheld.owner(STEM) == STEM


def test_a_salvage_task_of_a_model_no_slot_checked_is_never_dispatched(tmp_path, monkeypatch,
                                                                        capsys):
    phase = _recovered_phase(tmp_path)
    (phase / f"salvage_task_{OTHER_STEM}.json").write_text("{}")
    dispatched: List[str] = []
    from utils import task_file_runner

    monkeypatch.setattr(task_file_runner, "run_task_file",
                        lambda task_file, **_k: dispatched.append(task_file))
    args = SimpleNamespace(exec_subagents=True, dry_run_subagents=False)
    monkeypatch.setattr(rob, "exec_subagents_requested", lambda _args: True)
    assert rob.dispatch_salvage_tasks(args, phase_dir=str(phase), phase="review_plan",
                                      plan_path=None) == []
    assert dispatched == []
    assert f"[SALVAGE_WITHHELD] {OTHER_STEM}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The registry is read once, strictly, for the ownership decision (Codex 2)
# ---------------------------------------------------------------------------

def _locked_damaged_phase(tmp_path) -> Path:
    phase = tmp_path / "plan" / "review-plan"
    phase.mkdir(parents=True)
    (phase / ".status.json").write_text('{"models_rejected": ')
    (phase / f"{STEM}.json").write_text("[]")
    return phase


def _recover_holding_the_lock(tmp_path, phase: Path, run_id=None):
    lock = process_control.PhaseLock(detach.lock_path(str(phase)))
    lock.acquire({"pid": os.getpid()})
    try:
        return review_lane.recover_damaged_checkpoint(
            phase_dir=str(phase), phase_name="review-plan",
            plan_file=str(tmp_path / "plan.md"), run_id=run_id,
            retire_validation_round=rob.retire_stale_validation_round)
    finally:
        lock.release()


def _registry(tmp_path) -> Path:
    directory = Path(run_registry.registry_dir(run_registry.resolve_project_root(
        str(tmp_path / "plan.md"))))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _stopped_untouched(tmp_path, phase: Path, capsys, *, run_id=None) -> str:
    with pytest.raises(SystemExit) as stopped:
        _recover_holding_the_lock(tmp_path, phase, run_id=run_id)
    assert stopped.value.code == 3
    assert (phase / ".status.json").read_text() == '{"models_rejected": '
    assert (phase / f"{STEM}.json").exists()
    assert not list(phase.glob(".status.json.unreadable-*"))
    return capsys.readouterr().err


def test_a_registry_listing_that_fails_after_the_first_never_authorises_recovery(
        tmp_path, monkeypatch, capsys):
    phase = _locked_damaged_phase(tmp_path)
    registry = _registry(tmp_path)
    (registry / "run-earlier.json").write_text(json.dumps(
        {"run_id": "run-earlier", "pid": 1, "phase_dir": str(phase)}))
    real_listdir = os.listdir
    listed: List[str] = []

    def listdir(path="."):
        if os.path.realpath(os.fspath(path)) == os.path.realpath(registry):
            listed.append(os.fspath(path))
            if len(listed) > 1:
                raise OSError(errno.EIO, "injected: listing failed", os.fspath(path))
        return real_listdir(path)
    monkeypatch.setattr(os, "listdir", listdir)
    err = _stopped_untouched(tmp_path, phase, capsys)
    assert "run-earlier" in err and len(listed) == 1


def test_a_registry_that_cannot_be_listed_stops_the_recovery(tmp_path, monkeypatch, capsys):
    phase = _locked_damaged_phase(tmp_path)
    registry = _registry(tmp_path)
    real_listdir = os.listdir

    def listdir(path="."):
        if os.path.realpath(os.fspath(path)) == os.path.realpath(registry):
            raise PermissionError(errno.EACCES, "injected", os.fspath(path))
        return real_listdir(path)
    monkeypatch.setattr(os, "listdir", listdir)
    assert "run registry could not be read" in _stopped_untouched(tmp_path, phase, capsys)


@pytest.mark.parametrize("entry", ["unopenable", "not-json", "a-list", "no-phase-dir"])
def test_a_registry_entry_whose_phase_cannot_be_told_stops_the_recovery(
        tmp_path, monkeypatch, capsys, entry):
    phase = _locked_damaged_phase(tmp_path)
    path = _registry(tmp_path) / "run-unknown.json"
    path.write_text({
        "unopenable": json.dumps({"run_id": "run-unknown", "phase_dir": str(phase)}),
        "not-json": '{"run_id": "run-unk',
        "a-list": "[]",
        "no-phase-dir": json.dumps({"run_id": "run-unknown", "pid": 1}),
    }[entry])
    if entry == "unopenable":
        real_open = builtins.open

        def unopenable(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, os.PathLike)) and os.fspath(file) == str(path):
                raise OSError(errno.EIO, "injected: entry read failed", os.fspath(file))
            return real_open(file, mode, *args, **kwargs)
        monkeypatch.setattr(builtins, "open", unopenable)
    err = _stopped_untouched(tmp_path, phase, capsys)
    assert str(path) in err and "runs --sweep" in err


def test_registry_entries_of_other_phases_and_this_run_do_not_stop_the_recovery(tmp_path):
    phase = _locked_damaged_phase(tmp_path)
    registry = _registry(tmp_path)
    other = tmp_path / "plan" / "review-tasks"
    other.mkdir()
    (registry / "run-other.json").write_text(json.dumps(
        {"run_id": "run-other", "pid": 1, "phase_dir": str(other)}))
    (registry / "run-self.json").write_text("not an entry yet")
    assert _recover_holding_the_lock(tmp_path, phase, run_id="run-self") is not None


@LANES
def test_an_unreadable_registry_entry_stops_both_lanes_recovery(tmp_path, monkeypatch, capsys,
                                                                 lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    damaged = _damaged_folder(lane, monkeypatch)
    root = run_registry.resolve_project_root(str(lane.plan))
    os.makedirs(run_registry.registry_dir(root), exist_ok=True)
    Path(run_registry.entry_path(root, "run-torn")).write_text('{"run_id": "run-torn", "phase')
    capsys.readouterr()

    assert _run(lane, "--force") == 3
    err = capsys.readouterr().err
    assert "run-torn.json" in err and "runs --sweep" in err
    _untouched(lane, damaged)


# ---------------------------------------------------------------------------
# A host-index refresh cannot slip between the retirement and its check (Codex 3)
# ---------------------------------------------------------------------------

def _materialized_round(lane, monkeypatch) -> None:
    """A real validation round handed to the host: its task file, its host
    index and its prompt sidecars; then the checkpoint damaged."""
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Validated answer"]}, []))
    assert lane.run(
        "--models", SPEC, "--timeout", "120", "--max-parallel", "1",
        patches=(
            patch("utils.interactive.resolve_models", return_value=[SPEC]),
            patch("utils.interactive.is_model_valid", return_value=True),
        ),
    ) == 0
    phase = lane.phase_dir
    assert (phase / "validation_tasks.json").exists()
    assert (phase / "validation_tasks.index.json").exists()
    assert list(phase.glob("validation_batch_*.prompt.txt"))
    status = phase / ".status.json"
    status.write_bytes(status.read_bytes()[:-7])


@LANES
def test_a_concurrent_index_refresh_never_outlives_the_retired_round(tmp_path, monkeypatch,
                                                                     capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _materialized_round(lane, monkeypatch)
    task_file = str(lane.phase_dir / "validation_tasks.json")
    real_invalidate = rob.invalidate_host_index
    refreshes: List[threading.Thread] = []

    def invalidate(path):
        real_invalidate(path)
        if refreshes or os.path.abspath(path) != os.path.abspath(task_file):
            return
        # A host's refresh, on its own thread, the moment the index is gone:
        # given a second to land before the retirement goes on.
        refresh = threading.Thread(target=validation.refresh_host_index, args=(task_file,))
        refreshes.append(refresh)
        refresh.start()
        refresh.join(timeout=1.0)
    monkeypatch.setattr(rob, "invalidate_host_index", invalidate)
    # What the folder holds the moment the recovery hands the run on: the
    # run's later stages would hide an index left behind.
    real_recover = review_lane.recover_damaged_checkpoint
    left: Dict[str, bool] = {}

    def recover(**kwargs):
        recovery = real_recover(**kwargs)
        if recovery is not None:
            for refresh in refreshes:
                refresh.join(timeout=30)
            left.update({name: (lane.phase_dir / name).exists() for name in (
                "validation_tasks.index.json", "validation_tasks.json")})
        return recovery
    monkeypatch.setattr(review_lane, "recover_damaged_checkpoint", recover)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answers(lane, {SPEC: ["Clean answer"]}, []))
    capsys.readouterr()

    assert _run(lane, "--force") == 0
    assert refreshes and not refreshes[0].is_alive()
    assert "[CHECKPOINT_RECOVERED]" in capsys.readouterr().out
    assert left == {"validation_tasks.index.json": False, "validation_tasks.json": False}
    assert not (lane.phase_dir / "validation_tasks.index.json").exists()
    assert _titles(lane) == ["Clean answer"]


# ---------------------------------------------------------------------------
# Nothing blocks on a FIFO at the checkpoint's name (Codex 4)
# ---------------------------------------------------------------------------

def _finishes_without_a_writer(call: Callable[[], Any], fifo: Path) -> Dict[str, Any]:
    """Run *call* on a thread; when it is still running after a few seconds
    (blocked opening the FIFO for reading), open the FIFO's write end to
    release it. ``{"blocked": bool, "result" | "error": ...}``."""
    outcome: Dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = call()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            outcome["error"] = exc
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=3.0)
    outcome["blocked"] = worker.is_alive()
    if outcome["blocked"]:
        descriptor = os.open(str(fifo), os.O_WRONLY | os.O_NONBLOCK)
        os.close(descriptor)
        worker.join(timeout=10)
    return outcome


FIFO_ONLY = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs")


@FIFO_ONLY
def test_a_link_to_a_fifo_is_no_damage_and_never_blocks(tmp_path):
    fifo = tmp_path / "elsewhere"
    os.mkfifo(fifo)
    (tmp_path / ".status.json").symlink_to(fifo)
    outcome = _finishes_without_a_writer(lambda: status_file.checkpoint_damage(str(tmp_path)),
                                         fifo)
    assert not outcome["blocked"]
    assert isinstance(outcome.get("error"), status_file.CheckpointUnreadable)


@FIFO_ONLY
def test_a_file_swapped_for_a_fifo_after_its_lstat_never_blocks(tmp_path, monkeypatch):
    status = tmp_path / ".status.json"
    os.mkfifo(status)
    regular = tmp_path / "regular"
    regular.write_text("{")
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        if os.fspath(path) == str(status):
            return real_lstat(regular)
        return real_lstat(path, *args, **kwargs)
    monkeypatch.setattr(os, "lstat", lstat)
    outcome = _finishes_without_a_writer(lambda: status_file.checkpoint_damage(str(tmp_path)),
                                         status)
    assert not outcome["blocked"]
    assert isinstance(outcome.get("error"), status_file.CheckpointUnreadable)


@FIFO_ONLY
@pytest.mark.parametrize("shape", ["fifo", "link-to-fifo"])
def test_the_generic_checkpoint_readers_never_block_on_a_fifo(tmp_path, shape):
    status = tmp_path / ".status.json"
    fifo = tmp_path / "elsewhere" if shape == "link-to-fifo" else status
    os.mkfifo(fifo)
    if shape == "link-to-fifo":
        status.symlink_to(fifo)
    loaded = _finishes_without_a_writer(lambda: status_file.load_status(str(tmp_path)), fifo)
    assert not loaded["blocked"]
    assert isinstance(loaded.get("error"), status_file.CheckpointUnreadable)
    read = _finishes_without_a_writer(lambda: status_file.read_status(str(tmp_path)), fifo)
    assert not read["blocked"] and read.get("result") == {}
    # A link to a readable checkpoint is still read through.
    if shape == "link-to-fifo":
        status.unlink()
        (tmp_path / "real.json").write_text('{"state": "complete"}')
        status.symlink_to(tmp_path / "real.json")
        assert status_file.load_status(str(tmp_path)) == {"state": "complete"}
        assert status_file.checkpoint_damage(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# A recovered phase never trusts a result file on disk (Codex rounds 2 and 3)
# ---------------------------------------------------------------------------

def _invokes(lane, by_spec: Dict[str, Any], calls: List[str]) -> Callable[..., Any]:
    """A provider per spec: a list of titles writes them to the requested file
    and succeeds; ``None`` fails with no answer, no file and no salvage text,
    and still passes the slot's checks."""
    def invoke(**kwargs):
        spec = kwargs["model_spec"]
        calls.append(spec)
        titles = by_spec[spec]
        if titles is None:
            return {"success": False, "error": "provider exited 1", "details": {}}
        findings = lane.findings(list(titles))
        destination = lane.phase_dir / f"{rob.sanitize_model_name(spec)}.json"
        destination.write_text(json.dumps(findings))
        return {"success": True, "data": findings, "source": "file",
                "output_file": str(destination), "details": {}}
    return invoke


LATE = "Late answer of the crashed run"


def _recovered_run(lane, monkeypatch, capsys, by_spec: Dict[str, Any], calls: List[str],
                   *flags: str, models: Sequence[str] = (SPEC,)) -> int:
    """The fresh run that recovers the damaged folder, its providers *by_spec*."""
    _damaged_folder(lane, monkeypatch)
    capsys.readouterr()
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _invokes(lane, by_spec, calls))
    return _run(lane, "--force", *flags, models=models)


@pytest.mark.parametrize("reader", ["resume", "reaggregate"])
@pytest.mark.parametrize("replacement", ["failed-file-less", "successful"])
@LANES
def test_a_late_answer_after_a_checked_replacement_is_never_reused(
        tmp_path, monkeypatch, capsys, lane_class, replacement, reader):
    """Codex's eight: the recovery's run checks a slot of the model (one that
    fails with nothing, or one that answers), then the crashed run's provider
    writes its late answer at the model's path. A resume dispatches every
    model again and a reaggregate refuses: nothing on disk is read."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    calls: List[str] = []
    first = None if replacement == "failed-file-less" else ["Clean answer"]
    assert _recovered_run(lane, monkeypatch, capsys, {SPEC: first, OTHER: ["Other answer"]},
                          calls, models=[SPEC, OTHER]) == 0
    assert sorted(calls) == sorted([SPEC, OTHER])
    grouped = _titles(lane)
    late = lane.phase_dir / f"{STEM}.json"
    late.write_text(json.dumps(lane.findings([LATE])))
    assert not resume_state._kept_on_rerun(lane.phase_dir, lane.phase, SPEC)
    calls.clear()

    if reader == "resume":
        monkeypatch.setattr(_module(lane), "invoke_with_file_output", _invokes(
            lane, {SPEC: ["Fresh answer"], OTHER: ["Other again"]}, calls))
        assert _run(lane, "--force", models=[SPEC, OTHER]) == 0
        out = capsys.readouterr().out
        assert sorted(calls) == sorted([SPEC, OTHER]) and "[SKIP]" not in out
        assert _titles(lane) == ["Fresh answer", "Other again"]
    else:
        _refused_reaggregate(lane, capsys)
        assert _titles(lane) == grouped and LATE not in grouped
        assert late.exists() and calls == []


@pytest.mark.parametrize("timing", ["after-the-save", "after-the-close"])
@LANES
def test_an_answer_replaced_after_its_save_is_never_read(tmp_path, monkeypatch, capsys,
                                                         lane_class, timing):
    """Codex round 3, 1 and 2: a predecessor replaces the answer the fan-out's
    close has just saved, before the close's record (``after-the-save``) or
    before the round groups it (``after-the-close``). The round groups the
    answer it holds; a reaggregate refuses and a later run asks again."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    calls: List[str] = []

    def replace_it() -> None:
        (lane.phase_dir / f"{STEM}.json").write_text(json.dumps(lane.findings([LATE])))
    if timing == "after-the-save":
        real_record = review_lane.withheld_record

        def record(*args, **kwargs):
            replace_it()
            return real_record(*args, **kwargs)
        monkeypatch.setattr(review_lane, "withheld_record", record)
    else:
        real_close = review_lane.close_fanout

        async def close(*args, **kwargs):
            closed = await real_close(*args, **kwargs)
            replace_it()
            return closed
        monkeypatch.setattr(review_lane, "close_fanout", close)

    assert _recovered_run(lane, monkeypatch, capsys, {SPEC: ["Checked answer"]}, calls) == 0
    assert calls == [SPEC]
    assert _titles(lane) == ["Checked answer"]
    assert LATE not in (lane.phase_dir / "report.md").read_text(encoding="utf-8")
    monkeypatch.undo()
    lane = _lane(lane_class, tmp_path, monkeypatch)
    assert not resume_state._kept_on_rerun(lane.phase_dir, lane.phase, SPEC)
    _refused_reaggregate(lane, capsys)
    assert _titles(lane) == ["Checked answer"]

    calls.clear()
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _invokes(lane, {SPEC: ["Fresh answer"]}, calls))
    assert _run(lane, "--force") == 0
    assert calls == [SPEC] and _titles(lane) == ["Fresh answer"]


@LANES
def test_a_salvage_pair_published_late_is_never_dispatched(tmp_path, monkeypatch, capsys,
                                                           lane_class):
    """Codex round 3, 1: after an answer-less slot's checks pass, a predecessor
    publishes a salvage pair for it. No salvage runs in a recovered phase,
    even under ``--exec-subagents``: the model stays failed and the run
    completes partial with a warning."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    real_clear = review_lane.clear_unchecked_slot

    def checked_then_published(model_spec, result_path, *args, **kwargs):
        real_clear(model_spec, result_path, *args, **kwargs)
        if model_spec == SPEC:
            rob._write_salvage_request(
                model=SPEC, sanitized_model=STEM,
                raw_output=json.dumps(lane.findings([LATE])),
                phase_dir=str(lane.phase_dir), phase_name="review_plan")
    monkeypatch.setattr(review_lane, "clear_unchecked_slot", checked_then_published)
    from utils import task_file_runner
    repaired: List[Path] = []
    monkeypatch.setattr(task_file_runner, "run_task_file",
                        _repairer(lane, lane.findings([LATE]), repaired))
    calls: List[str] = []

    assert _recovered_run(lane, monkeypatch, capsys, {SPEC: None, OTHER: ["Other answer"]},
                          calls, "--exec-subagents", models=[SPEC, OTHER]) == 0
    captured = capsys.readouterr()
    assert repaired == [] and "recovered by salvage" not in captured.out
    assert "no salvage runs" in captured.err
    assert _titles(lane) == ["Other answer"]
    assert SPEC in lane.status()["models_failed"]
    assert (lane.phase_dir / f"salvage_task_{STEM}.json").exists()


@LANES
def test_a_recovered_phase_writes_no_salvage_request(tmp_path, monkeypatch, capsys, lane_class):
    """Codex round 3, 2: nothing a salvage would read or write is trusted in a
    recovered phase, so a failed slot whose output holds an answer writes no
    request, and none is dispatched."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _damaged_folder(lane, monkeypatch)
    capsys.readouterr()
    _probe(monkeypatch, None)
    answering = _invokes(lane, {OTHER: ["Other answer"]}, [])
    failing = _failing_invoke(lane, lane.findings(["Salvageable answer"]))
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        lambda **kw: failing(**kw) if kw["model_spec"] == SPEC else answering(**kw))
    from utils import task_file_runner
    dispatched: List[Any] = []
    monkeypatch.setattr(task_file_runner, "run_task_file",
                        lambda *a, **_k: dispatched.append(a))

    assert _run(lane, "--force", "--exec-subagents", models=[SPEC, OTHER]) == 0
    out = capsys.readouterr().out
    assert f"[{SPEC}] No salvage request written" in out
    assert "[SALVAGE_NEEDED]" not in out and "[SALVAGE_TASK]" not in out
    assert sorted(p.name for p in lane.phase_dir.glob("salvage_*.json")) == []
    assert dispatched == []
    assert _titles(lane) == ["Other answer"]


@LANES
def test_a_host_written_repair_in_a_recovered_phase_is_never_read(tmp_path, monkeypatch,
                                                                   capsys, lane_class):
    """Codex round 3, 4: a repair written outside the run is a file on disk
    like any other. A reaggregate refuses with it left in place; a later run
    asks the model again."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    calls: List[str] = []
    assert _recovered_run(lane, monkeypatch, capsys, {SPEC: None}, calls) == 3
    repair = lane.phase_dir / f"{STEM}.json"
    repair.write_text(json.dumps(lane.findings(["Host repair"])))
    assert not resume_state._kept_on_rerun(lane.phase_dir, lane.phase, SPEC)
    _refused_reaggregate(lane, capsys)
    assert repair.exists()
    calls.clear()
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _invokes(lane, {SPEC: ["Fresh answer"]}, calls))
    assert _run(lane, "--force") == 0
    assert calls == [SPEC] and _titles(lane) == ["Fresh answer"]


@LANES
def test_a_recovered_round_hands_no_validation_to_the_host(tmp_path, monkeypatch, capsys,
                                                           lane_class):
    """Its ``--reaggregate`` would refuse to read the verdicts back, so the
    round completes unvalidated, as under ``--skip-validation``, and says how
    to get a validated review."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _damaged_folder(lane, monkeypatch)
    capsys.readouterr()
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _invokes(lane, {SPEC: ["Clean answer"]}, []))
    result = lane.run(
        "--models", SPEC, "--timeout", "120", "--max-parallel", "1", "--force",
        patches=(
            patch("utils.interactive.resolve_models", return_value=[SPEC]),
            patch("utils.interactive.is_model_valid", return_value=True),
        ),
    )
    captured = capsys.readouterr()
    assert result == 0, captured.out + captured.err
    assert "validation was not run" in captured.err and "--exec-subagents" in captured.err
    assert "[VALIDATION_PENDING]" not in captured.out
    assert not (lane.phase_dir / "validation_tasks.json").exists()
    assert lane.phase_completed() and _titles(lane) == ["Clean answer"]
    assert any("validation was not run" in note for note in lane.status()["result_notes"])


@LANES
def test_util_resume_state_finds_nothing_reusable(tmp_path, monkeypatch, capsys, lane_class):
    """Whatever a recovered phase that ``state.json`` does not record complete
    holds, a validation round of its own included, the answer is a re-run of
    every requested model."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    assert _recovered_run(lane, monkeypatch, capsys, {SPEC: ["Clean answer"]}, []) == 0
    state = get_or_create_state(lane.plan)
    state.state.get("phases_completed", {}).pop(lane.phase, None)
    state.save()
    review_round_id = lane.status()[review_round.ROUND_KEY]
    (lane.phase_dir / "validation_tasks.json").write_text(json.dumps({
        review_round.ROUND_KEY: review_round_id, "total_batches": 1,
        "batches": [{"batch_index": 0}]}))
    (lane.phase_dir / "validation_batch_0.json").write_text("{}")
    answer = resume_state.classify(lane.plan, lane.phase)
    assert answer.action == "rerun" and answer.models == (SPEC,)
    assert "damaged checkpoint" in answer.note
    # The same folder without the record: this round's validation is reused.
    status_file.remove_status_keys(str(lane.phase_dir), [result_files.RECOVERED_KEY])
    assert resume_state.classify(lane.plan, lane.phase).action != "rerun"


@pytest.mark.parametrize("ending", ["complete", "stale"])
@LANES
def test_util_resume_state_keeps_a_completed_recovered_phase_with_the_note(
        tmp_path, monkeypatch, capsys, lane_class, ending):
    """``state.json`` decides first: a phase completed after its recovery
    reported the answers that run checked in memory, so it stays ``report``
    (a code review older than the latest implement stays ``run``), and the
    note says so while the record is present."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    assert _recovered_run(lane, monkeypatch, capsys, {SPEC: ["Clean answer"]}, []) == 0
    expected = ("a", "report")
    if ending == "stale":
        if lane.phase != "code-review":
            pytest.skip("only a code review is staled by implement")
        state = get_or_create_state(lane.plan)
        state.state["phases_completed"]["implement"] = "2999-01-01T00:00:00"
        state.save()
        expected = ("e", "run")
    answer = resume_state.classify(lane.plan, lane.phase)
    assert (answer.case, answer.action) == expected
    assert "damaged checkpoint" in answer.note and "Move the whole phase folder" in answer.note
    capsys.readouterr()
    assert resume_state.main([lane.mode, str(lane.plan)]) == 0
    captured = capsys.readouterr()
    assert f"case={expected[0]} action={expected[1]}" in captured.out
    assert answer.note in captured.err
    # The same folder without the record: the same answer, no note.
    status_file.remove_status_keys(str(lane.phase_dir), [result_files.RECOVERED_KEY])
    plain = resume_state.classify(lane.plan, lane.phase)
    assert (plain.case, plain.action, plain.note) == (*expected, "")


class TaskLane(PlanLane):
    """review-tasks, the base lane's other phase, over a generated task list."""

    mode, phase, module = "review-tasks", "review-tasks", rto

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        super().__init__(root, monkeypatch)
        tasks = self.plan.parent / self.plan.stem / "tasks"
        tasks.mkdir(parents=True, exist_ok=True)
        (tasks / "tasks.md").write_text(
            "# Tasks\n\n## T001: Change handler\nImplement the handler.\n", encoding="utf-8")
        state = get_or_create_state(self.plan)
        state.mark_phase_completed("generate-tasks")
        state.save()


BASE_LANES = pytest.mark.parametrize("lane_class", [PlanLane, TaskLane],
                                     ids=["review-plan", "review-tasks"])
SKIPPING = pytest.mark.parametrize("skipping", ["automatic", "--skip-validation"])


def _run_over_a_damaged_checkpoint(lane, monkeypatch, titles: Sequence[str], *flags: str) -> int:
    """A fresh run, not validated in its own process, over a phase folder whose
    checkpoint is damaged (it recovers it), its one model answering *titles*."""
    lane.phase_dir.mkdir(parents=True, exist_ok=True)
    (lane.phase_dir / ".status.json").write_text("{broken")
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _invokes(lane, {SPEC: list(titles)}, []))
    return lane.run(
        "--models", SPEC, "--timeout", "120", "--max-parallel", "1", "--force", *flags,
        patches=(
            patch("utils.interactive.resolve_models", return_value=[SPEC]),
            patch("utils.interactive.is_model_valid", return_value=True),
        ),
    )


@SKIPPING
def test_a_recovered_round_without_validation_still_fails_on_high(tmp_path, monkeypatch,
                                                                  capsys, skipping):
    """Codex's U90c review 4: a recovered review-code round that runs no
    validation reports and counts the groups it holds, so ``--fail-on-high``
    exits 3 over a HIGH one whether the skip was automatic or typed."""
    lane = _lane(CodeLane, tmp_path, monkeypatch)
    findings = lane.findings
    lane.findings = lambda names: findings(names, high=names)
    flags = ["--fail-on-high"] + ([skipping] if skipping != "automatic" else [])
    result = _run_over_a_damaged_checkpoint(lane, monkeypatch, ["Checked answer"], *flags)
    captured = capsys.readouterr()
    assert result == 3, captured.out + captured.err
    assert "[HIGH_PRIORITY_ISSUES] 1" in captured.out
    assert result_files.phase_recovered(lane.phase_dir)
    assert _titles(lane) == ["Checked answer"]
    report = (lane.phase_dir / "report.md").read_text()
    assert "Checked answer" in report
    if skipping == "automatic":
        assert "validation was not run" in report


@SKIPPING
@pytest.mark.parametrize("titles", [["Checked answer"], []], ids=["findings", "no-findings"])
@BASE_LANES
def test_a_recovered_report_never_reads_grouped_json_back(tmp_path, monkeypatch, capsys,
                                                         lane_class, titles, skipping):
    """Codex's U90c review 4: a grouped.json written between the grouping and
    the report (by a provider the crashed run left behind) never becomes the
    report or its HIGH-group count, an empty grouping included: the report
    shows the groups the round holds."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    real = rob.aggregate_results

    def replaced_first(*args, **kwargs):
        name = "Unchecked replacement"
        (lane.phase_dir / "grouped.json").write_text(json.dumps([{
            "group_hash": "1111222233334444", "importance": "HIGH",
            "suggestions": lane.findings([name], high=[name])}]))
        return real(*args, **kwargs)

    monkeypatch.setattr(rob, "aggregate_results", replaced_first)
    flags = [skipping] if skipping != "automatic" else []
    result = _run_over_a_damaged_checkpoint(lane, monkeypatch, titles, *flags)
    captured = capsys.readouterr()
    assert result == 0, captured.out + captured.err
    assert result_files.phase_recovered(lane.phase_dir)
    report = (lane.phase_dir / "report.md").read_text()
    assert "Unchecked replacement" not in report
    if titles:
        assert "Checked answer" in report
    else:
        assert "_No suggestions found._" in report
    assert not lane.status().get("high_priority_groups")


def test_a_report_that_trusts_no_file_never_reads_grouped_json(tmp_path):
    """``[]`` is zero groups, never a missing answer; and a round that trusts
    no file on disk refuses to report without its groups."""
    phase = tmp_path / "review-plan"
    phase.mkdir()
    (phase / "grouped.json").write_text(json.dumps([{
        "group_hash": "1111222233334444", "importance": "HIGH",
        "suggestions": [{"title": "On disk", "importance": "HIGH"}]}]))
    report = rob.aggregate_results("feature", str(tmp_path), str(phase), "review-plan",
                                   [SPEC], {}, [], plan_path=str(tmp_path / "feature.md"))
    text = Path(report).read_text()
    assert "On disk" not in text and "_No suggestions found._" in text
    with pytest.raises(ValueError, match="trusts no file on disk"):
        rob.aggregate_results("feature", str(tmp_path), str(phase), "review-plan",
                              [SPEC], {}, None, read_grouped_file=False)
    assert "On disk" in Path(rob.aggregate_results(
        "feature", str(tmp_path), str(phase), "review-plan", [SPEC], {}, None,
        plan_path=str(tmp_path / "feature.md"))).read_text()


def test_a_checked_attempt_drops_the_operators_acceptance(tmp_path):
    phase = _recovered_phase(tmp_path)
    assert result_files.clear_withheld_model(phase, SPEC)
    assert result_files.recorded_recovery(phase)[result_files.ACCEPTED_FIELD] == [SPEC]
    assert SPEC not in result_files.withheld_models(phase)
    # The model's next checked attempt is its answer now, the acceptance gone.
    record = review_lane.withheld_record(str(phase), {
        SPEC: review_lane.SlotResult(SPEC, False, checked=True)})
    assert record[result_files.RECOVERED_KEY][result_files.ACCEPTED_FIELD] == []
    status_file.update_status(str(phase), record)
    assert SPEC in result_files.withheld_models(phase)
    # An attempt that did not pass its checks leaves it.
    assert result_files.clear_withheld_model(phase, OTHER)
    record = review_lane.withheld_record(str(phase), {
        OTHER: review_lane.SlotResult(OTHER, False)})
    assert record[result_files.RECOVERED_KEY][result_files.ACCEPTED_FIELD] == [OTHER]


# ---------------------------------------------------------------------------
# Registry entries never block on a FIFO (Codex round 2, candidate unit)
# ---------------------------------------------------------------------------

@FIFO_ONLY
@pytest.mark.parametrize("shape", ["fifo", "link-to-fifo"])
def test_a_registry_fifo_never_blocks_either_reader(tmp_path, shape):
    directory = _registry(tmp_path)
    root = run_registry.resolve_project_root(str(tmp_path / "plan.md"))
    entry = directory / "run-entry.json"
    fifo = tmp_path / "elsewhere" if shape == "link-to-fifo" else entry
    os.mkfifo(fifo)
    if shape == "link-to-fifo":
        entry.symlink_to(fifo)
    listed = _finishes_without_a_writer(lambda: run_registry.list_entries(root), fifo)
    assert not listed["blocked"]
    assert [e.get("corrupt") for e in listed["result"]] == [True]
    strict = _finishes_without_a_writer(
        lambda: run_registry.read_entries_strictly(root), fifo)
    assert not strict["blocked"]
    assert isinstance(strict.get("error"), run_registry.UnreadableEntries)
    assert strict["error"].paths == [str(entry)]
    for reader in (lambda: run_registry.entry_cleanup_open(root, "run-entry"),):
        outcome = _finishes_without_a_writer(reader, fifo)
        assert not outcome["blocked"] and outcome.get("result") is False


def test_a_regular_registry_entry_is_still_read(tmp_path):
    directory = _registry(tmp_path)
    root = run_registry.resolve_project_root(str(tmp_path / "plan.md"))
    (directory / "run-1.json").write_text(json.dumps({
        "run_id": "run-1", "phase_dir": str(tmp_path / "x"),
        "cleanup": {"state": "pending"}}))
    [entry] = run_registry.read_entries_strictly(root)
    assert entry["run_id"] == "run-1"
    assert run_registry.list_entries(root)[0]["run_id"] == "run-1"
    assert run_registry.entry_cleanup_open(root, "run-1")


# ---------------------------------------------------------------------------
# Ask keeps the recovery record, and the operator's acceptance is explicit
# (Codex round 2, 2; ask recovery itself is not wired)
# ---------------------------------------------------------------------------

@pytest.fixture
def ask_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    monkeypatch.setattr(rob, "sandbox_refusal",
                        lambda *_a, **_k: rob.SandboxVerdict(rob.SANDBOX_DISPATCH, None))
    plan = tmp_path / "my-feature.md"
    plan.write_text("# My Feature\n\nBuild a widget.\n", encoding="utf-8")
    return plan


def _ask_recovered(question_dir: Path) -> None:
    status_file.update_status(str(question_dir), {
        result_files.RECOVERED_KEY: result_files.recovery_record()})


def test_an_asks_launch_carries_the_recovery_record(ask_plan, capsys):
    from tests.test_ask import FakeInvoker, ask_argv, qdir, read_status, run_ask
    from tests.test_ask_run import SPEC as ASKED

    question = qdir(ask_plan, "Q")
    assert run_ask(ask_argv(ask_plan, ASKED), FakeInvoker()) == 0
    _ask_recovered(question)
    earlier = question / f"answer_{rob.sanitize_model_name(ASKED)}.md"
    earlier_text = earlier.read_text()
    capsys.readouterr()
    asked = FakeInvoker()
    assert run_ask(ask_argv(ask_plan, ASKED), asked) == 0
    out = capsys.readouterr().out
    # The answer on disk is neither reused nor counted: the slot sets it aside
    # as `.unchecked` and asks the model again, and the record stays.
    assert f"[SKIP] {ASKED}" not in out and asked.calls == [ASKED]
    assert "whose damaged checkpoint was replaced" in out
    assert Path(f"{earlier}.unchecked").read_text() == earlier_text
    status = read_status(question)
    assert result_files.RECOVERED_KEY in status
    assert status["models_completed"] == [ASKED]


def test_an_asks_closing_scan_counts_no_answer_this_run_did_not_produce(ask_plan, monkeypatch,
                                                                        capsys):
    """Codex round 3, 3: the sandbox refuses the dispatch, so this run
    produces nothing, and the answer on disk is not counted."""
    from tests.test_ask import FakeInvoker, answer_of, ask_argv, qdir, read_answers, run_ask
    from tests.test_ask_run import SPEC as ASKED

    question = qdir(ask_plan, "Q")
    question.mkdir(parents=True)
    status_file.write_status(str(question), {
        "question": "Q", "phase": "ask", "models_requested": [ASKED],
        result_files.RECOVERED_KEY: result_files.recovery_record()})
    answer_of(question, ASKED).write_text("Unchecked surviving answer")
    monkeypatch.setattr(rob, "sandbox_refusal", lambda *_a, **_k: rob.SandboxVerdict(
        rob.SANDBOX_REFUSE, "[SANDBOX_UNAVAILABLE] test refusal"))
    asked = FakeInvoker()
    assert run_ask(ask_argv(ask_plan, ASKED), asked) == 3
    assert asked.calls == []
    assert "Unchecked surviving answer" not in read_answers(question)
    status = status_file.read_status(str(question))
    assert ASKED in status["models_failed"] and not status["models_completed"]
    assert result_files.RECOVERED_KEY in status


def test_the_closing_scan_withholds_by_the_record_but_this_runs_answers_and_acceptances():
    from ask_orchestrator import withheld_after_run

    maps = {result_files.REJECTED_MODELS_KEY: {}, result_files.UNCHECKED_MODELS_KEY: {},
            result_files.RECOVERED_KEY: {**result_files.recovery_record(),
                                         result_files.ACCEPTED_FIELD: ["acc:epted"]}}
    results = {
        "pro:duced": {"success": True, review_lane.CHECKED_KEY: True},
        "fail:ed": {"success": False, review_lane.CHECKED_KEY: True},
        "un:checked": {"success": True},
    }
    specs = ["pro:duced", "fail:ed", "un:checked", "acc:epted", "never:run"]
    withheld = withheld_after_run(specs, results, maps)
    assert sorted(withheld) == ["fail:ed", "never:run", "un:checked"]
    assert {w.key for w in withheld.values()} == {result_files.RECOVERED_KEY}
    # Without the record nothing changes.
    del maps[result_files.RECOVERED_KEY]
    assert withheld_after_run(specs, results, maps) == {}


def _accept_over_a_recovered_partial(ask_plan):
    from tests.test_ask import FakeInvoker, ask_argv, qdir, run_ask, timeout_result
    from tests.test_ask_run import SPEC as ASKED

    question = qdir(ask_plan, "Q")
    timing_out = FakeInvoker()
    timing_out.configure(ASKED, write_file=False,
                         result=timeout_result(text="Half an answer."))
    assert run_ask(ask_argv(ask_plan, ASKED), timing_out) == 3
    _ask_recovered(question)
    return question, ASKED, lambda: run_ask(
        ask_argv(ask_plan, ASKED, extra=("--accept-partial", ASKED)), FakeInvoker())


def test_accepting_a_partial_in_a_recovered_folder_records_the_operators_choice(
        ask_plan, capsys):
    from tests.test_ask import partial_of, read_answers, read_status

    question, asked, accept = _accept_over_a_recovered_partial(ask_plan)
    capsys.readouterr()
    assert accept() == 0
    out = capsys.readouterr().out
    assert f"[ACCEPTED] {asked}" in out
    assert "counted only because --accept-partial names it" in out
    status = read_status(question)
    assert status["models_completed"] == [asked]
    record = status[result_files.RECOVERED_KEY]
    # The operator's acceptance, never a passed check.
    assert record == {"reason": result_files.RECOVERED_REASON,
                      result_files.ACCEPTED_FIELD: [asked]}
    assert "Half an answer." in read_answers(question)
    assert not partial_of(question, asked).exists()


def test_an_acceptance_that_does_not_read_back_is_never_reported(ask_plan, monkeypatch, capsys):
    from tests.test_ask import partial_of, read_status

    question, asked, accept = _accept_over_a_recovered_partial(ask_plan)
    real = status_file.update_status

    def update(phase_dir, updates, *args, **kwargs):
        if result_files.RECOVERED_KEY in updates:
            return None
        return real(phase_dir, updates, *args, **kwargs)
    monkeypatch.setattr(status_file, "update_status", update)
    capsys.readouterr()
    accept()
    out = capsys.readouterr().out
    assert "[ACCEPTED]" not in out and f"could not clear {asked}" in out
    assert partial_of(question, asked).exists()
    assert asked not in (read_status(question).get("models_completed") or [])
