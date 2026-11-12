"""An answer from a review slot whose safety checks failed, or never finished, never re-enters the review.

A slot whose working-tree, mode or home check fails is failed with the guard's
reason, and ``review_lane.reject_slot_answer`` refuses its answer for good: no
salvage request, no adoption of a repaired answer, its ``<stem>.json`` set
aside as ``.json.rejected`` and the model recorded under ``models_rejected``,
so neither ``--reaggregate`` nor a resume's ``[SKIP]`` reads it back, even when
the set-aside fails. A slot is recorded under ``models_unchecked`` before it is
dispatched and taken out when its checks pass, so one the deadline cancels
mid-check, or whose rejection cannot be recorded, stays withheld too. Either
record lasts until a checked attempt of the model supersedes it, and with no
record and the files still readable the phase stops
(docs/development/review-results.md, "Withheld slot answers"). Offline, on
both review lanes, through their real fan-outs: only the provider call
(``invoke_with_file_output``), the guard's ``finish`` where a test cancels it,
and the salvage executor are stubbed. The first scenarios are Codex's U37e
repros, inverted, and the later ones its U90 review's.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import errno
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

import pytest

from tests.test_review_lane_parity import CodeLane, PlanLane, _clean_env
from tests.test_slot_guard_parity import CLEAN, HOME_CHANGE, NOT_CLEAN, PROBES
import code_review_orchestrator as cro
from utils import llm_client
from utils import result_files
from utils import resume_state
from utils import review_lane
from utils import review_orchestrator_base as rob
from utils import slot_guard
from utils import status_file
from utils import task_file_runner

SPEC = "codex:gpt-5.5"
STEM = rob.sanitize_model_name(SPEC)
#: Another reviewer, selected while :data:`SPEC` is not.
OTHER = "cursor-agent:auto"
MARKERS = {"write": "[SANDBOX_BYPASSED]", "mode": "[MODE_CHECK]", "home": "[CODEX_HOME_CHECK]"}


def _lane(lane_class, tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    return lane_class(tmp_path, monkeypatch)


def _module(lane) -> Any:
    """The module whose ``invoke_with_file_output`` the lane's slot calls."""
    return rob if isinstance(lane, PlanLane) else cro


def _run(lane, *flags: str, models: Sequence[str] = (SPEC,)) -> int:
    return lane.run(
        "--models", *models, "--timeout", "120", "--max-parallel", "1",
        "--skip-validation", *flags,
        patches=(
            patch("utils.interactive.resolve_models", return_value=list(models)),
            patch("utils.interactive.is_model_valid", return_value=True),
        ),
    )


def _probe(monkeypatch, check: str) -> None:
    """The runner's sandbox for *check*: uncontained for the write check, the
    plugin's jail for the mode and home checks, a working sandbox for none."""
    name = {"write": "bypass", "mode": "jail", "home": "jail", None: "plain"}[check]
    monkeypatch.setattr(rob, "_fanout_sandbox_verdict", lambda *_: PROBES[name])


def _violate(lane, check: str, details: Dict[str, Any]) -> None:
    """Make the slot fail *check*: a tree change, a mode change, a home change."""
    if check == "write":
        (lane.repo / "src/app.py").write_text('print("concurrent change")\n')
    elif check == "mode":
        (lane.repo / "src/app.py").chmod(0o755)
    else:
        details["home_check"] = HOME_CHANGE
    details["sandbox"] = "unsandboxed" if check == "write" else "plugin-landlock"


def _answering_invoke(lane, findings, *, check=None, audit=NOT_CLEAN, calls=None):
    """A provider that writes *findings* to its result file and succeeds,
    after failing *check* when one is named."""
    def invoke(**_kwargs):
        if calls is not None:
            calls.append(1)
        details: Dict[str, Any] = {"readonly_audit": audit}
        if check is not None:
            _violate(lane, check, details)
        destination = lane.phase_dir / f"{STEM}.json"
        destination.write_text(json.dumps(findings))
        return {"success": True, "data": findings, "source": "file",
                "output_file": str(destination), "details": details}
    return invoke


def _failing_invoke(lane, findings, *, check=None):
    """A provider that exits 1 with *findings* on stdout, after failing
    *check* when one is named: a run salvage could repair."""
    def invoke(**_kwargs):
        details: Dict[str, Any] = {"stdout_excerpt": json.dumps(findings),
                                   "readonly_audit": NOT_CLEAN}
        if check is not None:
            _violate(lane, check, details)
        return {"success": False, "error": "provider exited 1", "details": details}
    return invoke


def _repairer(lane, findings, calls: List[Path]):
    """The salvage executor: writes *findings* where the request says."""
    def repair(task_file, **_kwargs):
        request = json.loads((lane.phase_dir / f"salvage_{STEM}.json").read_text())
        destination = Path(request["output_path"])
        destination.write_text(json.dumps(findings))
        calls.append(destination)
        return SimpleNamespace(outcomes=[SimpleNamespace(ok=True, output_path=str(destination))])
    return repair


def _titles(lane) -> List[str]:
    """The titles of every finding the round grouped."""
    return sorted(
        finding["title"]
        for group in lane.groups("grouped.json")
        for finding in group.get("suggestions") or [group]
    )


def _salvage_requests(lane) -> List[str]:
    return sorted(p.name for p in lane.phase_dir.glob("salvage_*.json"))


def _nothing_grouped(lane) -> bool:
    return not (lane.phase_dir / "grouped.json").exists()


def _leftover_salvage_pair(lane, findings, monkeypatch) -> None:
    """An earlier attempt's salvage pair, which no retirement moves."""
    lane.phase_dir.mkdir(parents=True, exist_ok=True)
    rob._write_salvage_request(
        model=SPEC, sanitized_model=STEM, raw_output=json.dumps(findings),
        phase_dir=str(lane.phase_dir), phase_name="review_plan",
    )
    monkeypatch.setattr(rob, "retire_salvage_artifacts", lambda *_a, **_k: [])
    monkeypatch.setattr(cro, "retire_salvage_artifacts", lambda *_a, **_k: [])


def _reject_and_leave_the_file(lane, monkeypatch) -> None:
    """A first run whose slot fails its home check and whose file stays."""
    _set_aside_fails(monkeypatch)
    _probe(monkeypatch, "home")
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Rejected file"]), check="home"))
    assert _run(lane) == 3
    assert (lane.phase_dir / f"{STEM}.json").exists()
    assert SPEC in lane.status()[result_files.REJECTED_MODELS_KEY]


# ---------------------------------------------------------------------------
# No salvage of a rejected answer (Codex's three base-lane repros)
# ---------------------------------------------------------------------------

#: The ``mode`` row flips an owner-execute bit under the plugin's jail. Windows
#: has neither: ``chmod`` there toggles the read-only flag alone, ``lstat``
#: derives the execute bit from the file's extension, and git for Windows sets
#: ``core.fileMode`` false, which turns ``mode_snapshot`` off. The row cannot
#: be expressed there (tests/test_slot_guard_parity.py skips the same way).
_CHECKS = ["write", pytest.param("mode", marks=pytest.mark.skipif(
    os.name == "nt", reason="POSIX file modes (the mode check); Windows runs no plugin jail",
)), "home"]


@pytest.mark.parametrize("check", _CHECKS)
def test_a_rejected_slot_is_never_salvaged(tmp_path, monkeypatch, capsys, check):
    lane = _lane(PlanLane, tmp_path, monkeypatch)
    findings = lane.findings(["Rejected answer"])
    repaired: List[Path] = []
    _probe(monkeypatch, check)
    monkeypatch.setattr(rob, "invoke_with_file_output", _failing_invoke(lane, findings, check=check))
    monkeypatch.setattr(task_file_runner, "run_task_file", _repairer(lane, findings, repaired))

    result = _run(lane, "--exec-subagents")
    out = capsys.readouterr().out

    assert result == 3, out
    assert MARKERS[check] in out
    assert "[SALVAGE_NEEDED]" not in out and "recovered by salvage" not in out
    assert _salvage_requests(lane) == [] and repaired == []
    assert not lane.phase_completed()
    status = lane.status()
    assert SPEC in status["models_failed"]
    assert SPEC in status[result_files.REJECTED_MODELS_KEY]
    # The diagnostics stay: the error log names the rejection and keeps the text.
    log = (lane.phase_dir / f"error_{STEM}.log").read_text()
    assert "Rejected: the slot failed its safety checks" in log
    assert "Rejected answer" in log


# ---------------------------------------------------------------------------
# The model-written file, through --reaggregate (Codex's two file repros)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("check", ["write", "home"])
@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_rejected_file_never_reenters_a_reaggregate(tmp_path, monkeypatch, capsys,
                                                      lane_class, check):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    findings = lane.findings(["Rejected file"])
    _probe(monkeypatch, check)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answering_invoke(lane, findings, check=check))

    result = _run(lane)
    out = capsys.readouterr().out
    assert result == 3, out
    assert _salvage_requests(lane) == []
    assert not (lane.phase_dir / f"{STEM}.json").exists()
    assert json.loads((lane.phase_dir / f"{STEM}.json.rejected").read_text()) == findings
    assert SPEC in lane.status()["models_failed"]

    result = lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert result == 3, out
    assert not lane.phase_completed()
    assert SPEC in lane.status()["models_failed"]


# ---------------------------------------------------------------------------
# The model-written file, through a resume's [SKIP]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_rejected_file_is_never_reused_by_a_resume(tmp_path, monkeypatch, capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    # The home check: it changes nothing a review-code scope id is cut from.
    _probe(monkeypatch, "home")
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Rejected file"]), check="home"))
    assert _run(lane) == 3
    capsys.readouterr()

    calls: List[int] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Clean answer"]), calls=calls))
    result = _run(lane, "--force")
    out = capsys.readouterr().out

    assert result == 0, out
    assert "[SKIP]" not in out
    assert calls == [1]
    assert _titles(lane) == ["Clean answer"]
    assert lane.status()[result_files.REJECTED_MODELS_KEY] == {}


# ---------------------------------------------------------------------------
# Fail closed: the set-aside or the retirement fails
# ---------------------------------------------------------------------------

def _set_aside_fails(monkeypatch) -> None:
    monkeypatch.setattr(result_files, "set_aside_rejected_result", lambda *_a, **_k: None)


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_file_the_set_aside_left_is_still_refused(tmp_path, monkeypatch, capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    _set_aside_fails(monkeypatch)
    _probe(monkeypatch, "home")
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Rejected file"]), check="home"))
    assert _run(lane) == 3
    captured = capsys.readouterr()
    assert (lane.phase_dir / f"{STEM}.json").exists()
    assert "keeps them out of --reaggregate and a resume" in captured.err

    # The reaggregate leaves it out and the model stays failed.
    result = lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert result == 3, out
    assert "left in place and not read" in out
    assert not lane.phase_completed()

    # A resume re-runs the model instead of reusing the file, and the
    # accepted answer clears the record.
    calls: List[int] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Clean answer"]), calls=calls))
    result = _run(lane, "--force")
    out = capsys.readouterr().out
    assert result == 0, out
    assert "failed its safety checks; re-running" in out
    assert "[SKIP]" not in out and calls == [1]
    assert _titles(lane) == ["Clean answer"]
    assert lane.status()[result_files.REJECTED_MODELS_KEY] == {}


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_salvage_pair_the_retirement_left_is_never_dispatched(tmp_path, monkeypatch, capsys,
                                                               lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    findings = lane.findings(["Rejected answer"])
    _leftover_salvage_pair(lane, findings, monkeypatch)
    repaired: List[Path] = []
    _probe(monkeypatch, "write")
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _failing_invoke(lane, findings, check="write"))
    monkeypatch.setattr(task_file_runner, "run_task_file", _repairer(lane, findings, repaired))

    result = _run(lane, "--exec-subagents")
    captured = capsys.readouterr()

    assert result == 3, captured.out
    assert f"salvage_task_{STEM}.json" in captured.err
    # The recorded rejection keeps the leftover task from running at all.
    assert repaired == []
    assert f"[SALVAGE_WITHHELD] {STEM}" in captured.out
    assert "recovered by salvage" not in captured.out
    assert not (lane.phase_dir / f"{STEM}.json").exists()
    assert SPEC in lane.status()["models_failed"]
    assert not lane.phase_completed()


# ---------------------------------------------------------------------------
# A rejection lasts until a checked attempt supersedes it (Codex U90 #1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_refused_retry_keeps_the_rejection(tmp_path, monkeypatch, capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _reject_and_leave_the_file(lane, monkeypatch)

    # The retry is refused before anything runs: it checked nothing.
    monkeypatch.setattr(rob, "_fanout_sandbox_verdict", lambda *_: PROBES["refuse"])
    assert _run(lane, "--force") == 3
    assert SPEC in lane.status()[result_files.REJECTED_MODELS_KEY]
    capsys.readouterr()

    result = lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert result == 3, out
    assert "left in place and not read" in out
    assert _nothing_grouped(lane)


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_refused_retry_never_dispatches_a_leftover_salvage(tmp_path, monkeypatch, capsys,
                                                            lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    findings = lane.findings(["Rejected salvage"])
    _leftover_salvage_pair(lane, findings, monkeypatch)
    _probe(monkeypatch, "home")
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answering_invoke(lane, findings, check="home"))
    assert _run(lane) == 3
    assert SPEC in lane.status()[result_files.REJECTED_MODELS_KEY]
    capsys.readouterr()

    calls: List[Path] = []
    monkeypatch.setattr(task_file_runner, "run_task_file", _repairer(lane, findings, calls))
    monkeypatch.setattr(rob, "_fanout_sandbox_verdict", lambda *_: PROBES["refuse"])
    result = _run(lane, "--force", "--exec-subagents")
    out = capsys.readouterr().out

    assert result == 3, out
    assert calls == []
    assert f"[SALVAGE_WITHHELD] {STEM}" in out
    assert "recovered by salvage" not in out
    assert SPEC in lane.status()[result_files.REJECTED_MODELS_KEY]
    assert _nothing_grouped(lane)


# ---------------------------------------------------------------------------
# A verdict nothing could record (Codex U90 #2)
# ---------------------------------------------------------------------------

def _status_writes_fail_after_the_answer(lane, monkeypatch, invoke):
    """*invoke*, after which every ``.status.json`` write fails until the
    returned function restores them."""
    real = status_file.write_in_plan_dir

    def run(**kwargs):
        result = invoke(**kwargs)

        def fail(*_a, **_k):
            raise OSError("simulated unavailable checkpoint storage")

        monkeypatch.setattr(status_file, "write_in_plan_dir", fail)
        return result

    monkeypatch.setattr(_module(lane), "invoke_with_file_output", run)
    return lambda: monkeypatch.setattr(status_file, "write_in_plan_dir", real)


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_rejection_storage_could_not_record_is_still_withheld(tmp_path, monkeypatch, capsys,
                                                                lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _set_aside_fails(monkeypatch)
    _probe(monkeypatch, "home")
    restore = _status_writes_fail_after_the_answer(
        lane, monkeypatch,
        _answering_invoke(lane, lane.findings(["Rejected file"]), check="home"),
    )

    assert _run(lane) == 3
    captured = capsys.readouterr()
    status = lane.status()
    # The rejection never landed; the record written before the dispatch did.
    assert not status.get(result_files.REJECTED_MODELS_KEY)
    assert SPEC in status[result_files.UNCHECKED_MODELS_KEY]
    assert ".status.json keeps them out" in captured.err
    assert (lane.phase_dir / f"{STEM}.json").exists()

    # Storage is back. The model's failure never landed either, so nothing
    # says it failed; but its file is not read.
    restore()
    lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert f"{STEM}.json is not {SPEC}'s result" in out
    assert "No model result files found" in out
    assert _nothing_grouped(lane)


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_slot_whose_withholding_cannot_be_recorded_is_never_dispatched(
        tmp_path, monkeypatch, capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _probe(monkeypatch, "home")
    calls: List[int] = []
    real_record = result_files.record_unchecked_model
    monkeypatch.setattr(result_files, "record_unchecked_model", lambda *_a, **_k: False)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Unrecorded answer"]),
                                          check="home", calls=calls))

    result = _run(lane)
    out = capsys.readouterr().out

    # Refused before the provider ran: nothing exists that a later
    # invocation could read back.
    assert result == 3, out
    assert calls == []
    assert f"[{SPEC}] {review_lane.UNRECORDED_REASON}" in out
    assert lane.status()["models_failed"][SPEC] == review_lane.UNRECORDED_REASON
    assert not list(lane.phase_dir.glob(f"{STEM}.json*"))

    monkeypatch.setattr(result_files, "record_unchecked_model", real_record)
    assert lane.reaggregate("--skip-validation") == 3
    assert _nothing_grouped(lane)


def test_a_slot_is_armed_only_over_a_checkpoint_it_can_read(tmp_path):
    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    result_path = phase_dir / f"{STEM}.json"
    # A direct call: no round opened the folder, so nothing is recorded.
    assert review_lane.arm_slot(SPEC, result_path) is None
    assert not (phase_dir / ".status.json").exists()
    # A checkpoint that cannot be read: refused, and kept as it is.
    (phase_dir / ".status.json").write_text('{"models_rejected": {"a:b"')
    assert review_lane.arm_slot(SPEC, result_path) is False
    assert (phase_dir / ".status.json").read_text() == '{"models_rejected": {"a:b"'
    # A readable one: recorded and read back.
    status_file.write_status(str(phase_dir), {"phase": "review-plan"})
    assert review_lane.arm_slot(SPEC, result_path) is True
    assert result_files.unchecked_models(phase_dir) == {SPEC: review_lane.ARMED_REASON}


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_the_phase_stops_when_a_withheld_record_vanishes(tmp_path, monkeypatch, capsys,
                                                         lane_class):
    # The last resort: the record written before the dispatch is gone (an
    # outside edit), and neither the rejection nor the set-aside lands.
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _set_aside_fails(monkeypatch)
    _probe(monkeypatch, "home")
    answer = _answering_invoke(lane, lane.findings(["Rejected file"]), check="home")

    def erase_then_answer(**kwargs):
        status_path = lane.phase_dir / ".status.json"
        status = json.loads(status_path.read_text())
        status.pop(result_files.UNCHECKED_MODELS_KEY)
        status_path.write_text(json.dumps(status))
        return answer(**kwargs)

    _status_writes_fail_after_the_answer(lane, monkeypatch, erase_then_answer)

    result = _run(lane)
    captured = capsys.readouterr()

    assert result == 3, captured.out
    assert "nothing durable records the rejection" in captured.err
    assert "could not record that they are withheld" in captured.err
    assert f"{SPEC}: {STEM}.json" in captured.err
    assert "The phase stops before grouping" in captured.err
    # It stopped before the ending a run with no result reaches.
    assert "ERROR: All models failed" not in captured.out
    assert _nothing_grouped(lane)


# ---------------------------------------------------------------------------
# Only a shown retirement or replacement supersedes (Codex U90 round 2, #1)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _stat_fails_for(target: Path):
    """``os.stat`` of *target* fails with EIO; every other path is stat'ed."""
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(target):
            raise OSError(errno.EIO, "simulated metadata read failure", os.fspath(path))
        return real_stat(path, *args, **kwargs)

    with patch.object(os, "stat", stat):
        yield


def _silent_failure(**_kwargs):
    """A provider that exits 1 and writes nothing at all."""
    return {"success": False, "error": "provider exited 1",
            "details": {"readonly_audit": NOT_CLEAN}}


@pytest.mark.parametrize("unavailable", ["at-the-start", "after-the-checks"])
@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_an_unavailable_file_stamp_never_clears_a_rejection(tmp_path, monkeypatch, capsys,
                                                            lane_class, unavailable):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    _reject_and_leave_the_file(lane, monkeypatch)
    capsys.readouterr()

    # A clean retry that writes nothing, while one of the slot's two stamps
    # of the rejected file it left cannot be read.
    real_baseline = module.requested_result_baseline
    target = lane.phase_dir / f"{STEM}.json"
    reads: List[int] = []

    def baseline(*args):
        reads.append(1)
        if len(reads) == (1 if unavailable == "at-the-start" else 2):
            with _stat_fails_for(target):
                return real_baseline(*args)
        return real_baseline(*args)

    monkeypatch.setattr(module, "requested_result_baseline", baseline)
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output", _silent_failure)

    assert _run(lane, "--force") == 3
    assert len(reads) == 2
    assert target.exists()
    assert SPEC in lane.status()[result_files.REJECTED_MODELS_KEY]
    capsys.readouterr()

    result = lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert result == 3, out
    assert "left in place and not read" in out
    assert _nothing_grouped(lane)


def test_the_requested_result_stamp_tells_absence_from_an_unread_stamp(tmp_path):
    from utils import llm_client

    phase = "plan_review"
    target = Path(rob.generate_output_path(tmp_path, "output", phase, SPEC))
    assert llm_client.requested_result_baseline(tmp_path, phase, SPEC) == ()
    with _stat_fails_for(target):
        # Nothing there, but nothing shows it either.
        assert llm_client.requested_result_baseline(tmp_path, phase, SPEC) is None
    target.write_text("[]")
    stamp = llm_client.requested_result_baseline(tmp_path, phase, SPEC)
    assert stamp and stamp == llm_client._result_file_stamp(target)
    with _stat_fails_for(target):
        assert llm_client.requested_result_baseline(tmp_path, phase, SPEC) is None


# ---------------------------------------------------------------------------
# A file no readable stamp shows to be new is never the answer (Codex U90
# round 3, #1): below the reader's mock boundary, and the slot's own rule
# ---------------------------------------------------------------------------

class _FlakyStat:
    """``os.stat`` of one path fails with EIO while armed: every call until
    :meth:`disarm`, or the next one after :meth:`fail_next`."""

    def __init__(self, monkeypatch, target: Path):
        self.target = os.path.realpath(target)
        self.name = os.path.basename(target)
        self.remaining: Optional[int] = 0  # None: until disarmed
        self.failed = 0
        real_stat = os.stat

        def stat(path, *args, **kwargs):
            if (self.remaining != 0 and isinstance(path, (str, os.PathLike))
                    and os.fspath(path).endswith(self.name)
                    and os.path.realpath(path) == self.target):
                if self.remaining is not None:
                    self.remaining -= 1
                self.failed += 1
                raise OSError(errno.EIO, "simulated metadata read failure", self.target)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", stat)

    def fail_until_disarmed(self) -> None:
        self.remaining = None

    def disarm(self) -> None:
        self.remaining = 0

    def fail_next(self) -> None:
        self.remaining = 1


def _exits_zero_writing_nothing(on_call):
    """``invoke_with_provider`` of a provider that exits 0 with no answer:
    nothing on stdout, no file. *on_call* runs as it is called."""
    def provider(*_args, **_kwargs):
        on_call()
        return {"success": True, "data": None, "details": {"readonly_audit": NOT_CLEAN}}
    return provider


def _unreadable_stamp(stat: _FlakyStat, where: str):
    """Arm *stat* for *where*; the provider's *on_call*."""
    if where == "until-the-provider-ran":
        stat.fail_until_disarmed()
        return stat.disarm
    return stat.fail_next


UNREADABLE_STAMPS = ["until-the-provider-ran", "once-after-it-ran"]


@pytest.mark.parametrize("where", UNREADABLE_STAMPS)
@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_file_no_readable_stamp_shows_new_is_never_the_answer(tmp_path, monkeypatch, capsys,
                                                                lane_class, where):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    _reject_and_leave_the_file(lane, monkeypatch)
    capsys.readouterr()

    # A clean retry through the real file reader, whose provider exits 0
    # without an answer, while the rejected file's metadata cannot be read:
    # by any stamp taken before the provider ran, or by the first one after.
    target = lane.phase_dir / f"{STEM}.json"
    stat = _FlakyStat(monkeypatch, target)
    on_call = _unreadable_stamp(stat, where)
    monkeypatch.setattr(module, "invoke_with_file_output", llm_client.invoke_with_file_output)
    monkeypatch.setattr(llm_client, "invoke_with_provider", _exits_zero_writing_nothing(on_call))
    _probe(monkeypatch, None)

    result = _run(lane, "--force")
    out = capsys.readouterr().out

    assert stat.failed
    assert result == 3, out
    assert _nothing_grouped(lane)
    status = lane.status()
    assert SPEC in status["models_failed"]
    if where == "once-after-it-ran":
        # Every stamp the slot took itself was read: the file never moved.
        assert target.exists()
        assert SPEC in status[result_files.REJECTED_MODELS_KEY]

    stat.disarm()
    assert lane.reaggregate("--skip-validation") == 3
    assert _nothing_grouped(lane)


@pytest.mark.parametrize("where", UNREADABLE_STAMPS)
def test_the_file_reader_reads_no_file_it_cannot_show_is_new(tmp_path, monkeypatch, where):
    _clean_env(monkeypatch)
    earlier_answer = [{"title": "Earlier answer"}]
    earlier = rob.generate_output_path(tmp_path, "output", "plan_review", SPEC)
    earlier.write_text(json.dumps(earlier_answer))
    stat = _FlakyStat(monkeypatch, earlier)
    on_call = _unreadable_stamp(stat, where)
    # The slot's stamp, taken before its first attempt.
    baseline = llm_client.requested_result_baseline(tmp_path, "plan_review", SPEC)
    assert (baseline is None) is (where == "until-the-provider-ran")

    with patch.object(llm_client, "invoke_with_provider", _exits_zero_writing_nothing(on_call)):
        result = llm_client.invoke_with_file_output(
            "Write to {output_json_path}", SPEC, {}, tmp_path, "plan_review",
            timeout=30, run_model_specs=[SPEC], requested_baseline=baseline,
        )

    assert stat.failed
    assert result["success"] is False and result["data"] is None
    assert "could not be shown to have been written since this slot started" in result["file_error"]
    # Left in place, neither read nor renamed: it may be this call's answer too.
    assert json.loads(earlier.read_text()) == earlier_answer
    assert sorted(p.name for p in earlier.parent.iterdir()) == [earlier.name]


@pytest.mark.parametrize("source", ["file", "stdout_fallback"])
@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_slot_accepts_no_answer_read_from_the_file_it_started_beside(tmp_path, monkeypatch,
                                                                       lane_class, source):
    # The slot's own rule, behind the reader's: a reader that takes the
    # unchanged earlier file for this call's answer is not believed. An answer
    # on stdout is this call's whatever the old file does.
    lane = _lane(lane_class, tmp_path, monkeypatch)
    lane.phase_dir.mkdir(parents=True, exist_ok=True)
    earlier = lane.phase_dir / f"{STEM}.json"
    earlier.write_text(json.dumps(lane.findings(["Earlier answer"])))
    answer = lane.findings(["Earlier answer" if source == "file" else "Stdout answer"])

    def invoke(**_kwargs):
        return {"success": True, "data": answer, "source": source,
                "output_file": str(earlier), "details": {"readonly_audit": NOT_CLEAN}}

    _probe(monkeypatch, None)
    outcome = _slot(lane, invoke)

    # The earlier file is still in place either way, so nothing supersedes it.
    assert not outcome.checked
    if source == "file":
        assert not outcome.success and outcome.payload is None
        assert outcome.error == review_lane.UNSHOWN_ANSWER_REASON
        assert outcome.rejected is None
        if lane_class is PlanLane:
            assert rob.failure_output_from_result(outcome.details) == ""
        assert lane.save_slot(outcome) is False
        assert _salvage_requests(lane) == []
    else:
        assert outcome.success and outcome.payload


# ---------------------------------------------------------------------------
# An answer whose checks never finished (Codex U90 #3)
# ---------------------------------------------------------------------------

def _cancel_the_checks(monkeypatch):
    """Cancel the guard's checks; returns what puts the real ones back."""
    real = slot_guard.SlotGuard.finish

    async def cancelled(*_a, **_k):
        raise asyncio.CancelledError()
    monkeypatch.setattr(slot_guard.SlotGuard, "finish", cancelled)
    return lambda: monkeypatch.setattr(slot_guard.SlotGuard, "finish", real)


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_slot_cancelled_before_its_checks_is_withheld(tmp_path, monkeypatch, capsys,
                                                        lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    _probe(monkeypatch, "home")
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Unchecked file"]), check="home"))
    restore_the_checks = _cancel_the_checks(monkeypatch)

    assert _run(lane) == 3
    status = lane.status()
    assert "deadline" in status["models_failed"][SPEC]
    assert SPEC in status[result_files.UNCHECKED_MODELS_KEY]
    assert not status.get(result_files.REJECTED_MODELS_KEY)
    assert not (lane.phase_dir / f"{STEM}.json").exists()
    assert (lane.phase_dir / f"{STEM}.json.unchecked").exists()
    capsys.readouterr()

    # A file put back by hand is still not read.
    (lane.phase_dir / f"{STEM}.json.unchecked").rename(lane.phase_dir / f"{STEM}.json")
    result = lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert result == 3, out
    assert "ended before its safety checks passed" in out
    assert _nothing_grouped(lane)

    # A checked replacement supersedes it, and a resume does not reuse it.
    restore_the_checks()
    calls: List[int] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Clean answer"]), calls=calls))
    result = _run(lane, "--force")
    out = capsys.readouterr().out
    assert result == 0, out
    assert "[SKIP]" not in out and calls == [1]
    assert _titles(lane) == ["Clean answer"]
    assert lane.status()[result_files.UNCHECKED_MODELS_KEY] == {}


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_the_fanout_deadline_withholds_a_slot_mid_checks(tmp_path, monkeypatch, capsys,
                                                         lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _probe(monkeypatch, "home")
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Unchecked file"]), check="home"))
    entered: List[bool] = []

    async def checks_that_never_end(*_a, **_k):
        entered.append(True)
        await asyncio.Future()
    monkeypatch.setattr(slot_guard.SlotGuard, "finish", checks_that_never_end)
    # The fan-out's own deadline (2 x --timeout), shortened, and counted from
    # the moment the slot's checks begin: on a loaded runner the slot alone
    # took longer than the 0.2 s to get there, so the deadline fired before the
    # checks it is meant to interrupt (windows-latest, 2026-11-12).
    wait_for = asyncio.wait_for

    async def shorter(awaitable, timeout):
        if timeout != 240:
            return await wait_for(awaitable, timeout)
        fanout = asyncio.ensure_future(awaitable)
        for _ in range(3000):
            if entered or fanout.done():
                break
            await asyncio.sleep(0.01)
        return await wait_for(fanout, 0.2)
    monkeypatch.setattr(asyncio, "wait_for", shorter)

    assert _run(lane) == 3
    assert entered
    status = lane.status()
    assert "deadline" in status["models_failed"][SPEC]
    assert SPEC in status[result_files.UNCHECKED_MODELS_KEY]
    assert not (lane.phase_dir / f"{STEM}.json").exists()
    capsys.readouterr()

    result = lane.reaggregate("--skip-validation")
    assert result == 3, capsys.readouterr().out
    assert _nothing_grouped(lane)


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_slot_that_raises_after_its_dispatch_is_withheld(tmp_path, monkeypatch, capsys,
                                                           lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Unchecked file"])))

    def broken(self, run):
        raise RuntimeError("simulated failure reading the last attempt")
    monkeypatch.setattr(rob.BaseSlot if lane_class is PlanLane else cro.CodeSlot,
                        "after_ladder", broken)

    assert _run(lane) == 3
    assert not (lane.phase_dir / f"{STEM}.json").exists()
    assert (lane.phase_dir / f"{STEM}.json.unchecked").exists()
    status = lane.status()
    assert SPEC in status["models_failed"]
    assert status[result_files.UNCHECKED_MODELS_KEY] == {SPEC: review_lane.RAISED_REASON}
    capsys.readouterr()

    (lane.phase_dir / f"{STEM}.json.unchecked").rename(lane.phase_dir / f"{STEM}.json")
    result = lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert result == 3, out
    assert "ended before its safety checks passed" in out
    assert _nothing_grouped(lane)


def _close_steps():
    return SimpleNamespace(sanitize_model_name=rob.sanitize_model_name,
                           print_closing_summary=lambda _phase_dir: None)


def _opened_phase(tmp_path) -> Path:
    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    status_file.write_status(str(phase_dir), {"phase": "review-plan",
                                              "models_requested": [SPEC]})
    return phase_dir


@pytest.mark.parametrize("settled", [True, False], ids=["joined", "join-timed-out"])
def test_a_deadline_loss_is_withheld_before_its_file_exists(tmp_path, monkeypatch, settled):
    phase_dir = _opened_phase(tmp_path)
    monkeypatch.setattr(review_lane._detach, "stop_phase_work", lambda _tag: settled)

    assert review_lane._withhold_lost_slots(str(phase_dir), _close_steps(), [SPEC]) is settled
    assert result_files.unchecked_models(phase_dir) == {SPEC: review_lane.DEADLINE_ERROR}

    # What a provider the join missed writes afterwards is refused everywhere.
    late = phase_dir / f"{STEM}.json"
    late.write_text("[]")
    assert result_files.withhold_results(phase_dir, [str(late)]) == []
    assert not resume_state._kept_on_rerun(phase_dir, "plan-review", SPEC)
    review_lane._stop_unless_withheld(
        str(phase_dir), _close_steps(), {}, [SPEC], settled=settled,
        failed_models={}, served_models={}, withheld={},
    )


def test_an_unjoined_deadline_loss_nothing_records_stops_the_phase(tmp_path, monkeypatch,
                                                                   capsys):
    phase_dir = _opened_phase(tmp_path)
    monkeypatch.setattr(review_lane._detach, "stop_phase_work", lambda _tag: False)
    monkeypatch.setattr(result_files, "record_unchecked_model", lambda *_a, **_k: False)

    settled = review_lane._withhold_lost_slots(str(phase_dir), _close_steps(), [SPEC])
    assert settled is False
    with pytest.raises(SystemExit) as stopped:
        review_lane._stop_unless_withheld(
            str(phase_dir), _close_steps(), {}, [SPEC], settled=settled,
            failed_models={}, served_models={}, withheld={},
        )
    assert stopped.value.code == 3
    err = capsys.readouterr().err
    assert f"{SPEC}: {STEM}.json (its provider may still write it)" in err


# ---------------------------------------------------------------------------
# A checkpoint read that fails never passes for no record (Codex U90 round 2, #3)
# ---------------------------------------------------------------------------

def _fail_the_next_checkpoint_read(monkeypatch) -> List[str]:
    """The next ``open`` of a ``.status.json`` for reading fails, once;
    returns the list the failed path is appended to."""
    import builtins

    real_open = builtins.open
    failed: List[str] = []

    def flaky(file, mode="r", *args, **kwargs):
        if not failed and "r" in mode and os.fspath(file).endswith(".status.json"):
            failed.append(os.fspath(file))
            raise OSError(errno.EIO, "simulated checkpoint read failure", os.fspath(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky)
    return failed


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_failed_checkpoint_read_never_erases_a_rejection(tmp_path, monkeypatch, capsys,
                                                          lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _reject_and_leave_the_file(lane, monkeypatch)
    before = lane.status()
    capsys.readouterr()

    # A later progress update cannot read the checkpoint it merges into.
    failed = _fail_the_next_checkpoint_read(monkeypatch)
    status_file.update_progress(str(lane.phase_dir), {"step": "fanout"})
    assert failed
    after = lane.status()
    assert after == before
    assert SPEC in after[result_files.REJECTED_MODELS_KEY]

    result = lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert result == 3, out
    assert _nothing_grouped(lane)


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_an_unreadable_checkpoint_withholds_every_answer_from_a_reaggregate(
        tmp_path, monkeypatch, capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _reject_and_leave_the_file(lane, monkeypatch)
    status_path = lane.phase_dir / ".status.json"
    status_path.write_text(status_path.read_text()[:-7])
    capsys.readouterr()

    result = lane.reaggregate("--skip-validation")
    captured = capsys.readouterr()
    assert result == 3, captured.out
    assert "cannot be read, so nothing shows which result files" in captured.err
    assert _nothing_grouped(lane)
    # Left in place and unread, not renamed: the read failure may pass.
    assert (lane.phase_dir / f"{STEM}.json").exists()


def _flaky_opening_reads(monkeypatch, *, every: bool) -> List[str]:
    """``open`` of ``.status.json`` fails while ``open_round`` reads the
    withheld maps to carry them over (``result_files.recorded_withheld``):
    the first such ``open`` only, or every one. Its other reads, the round
    check among them (``Claim.held``), succeed. Returns the failed paths."""
    import builtins

    real_open = builtins.open
    real_load = result_files.recorded_withheld
    real_open_round = review_lane.open_round
    state = {"opening": False, "loading": False}
    failed: List[str] = []

    def flaky_open(file, mode="r", *args, **kwargs):
        if (state["loading"] and (every or not failed) and "r" in mode
                and isinstance(file, (str, os.PathLike))
                and os.fspath(file).endswith(".status.json")):
            failed.append(os.fspath(file))
            raise OSError(errno.EIO, "simulated checkpoint read failure", os.fspath(file))
        return real_open(file, mode, *args, **kwargs)

    def load_status(*args, **kwargs):
        state["loading"] = state["opening"]
        try:
            return real_load(*args, **kwargs)
        finally:
            state["loading"] = False

    def open_round(*args, **kwargs):
        state["opening"] = True
        try:
            return real_open_round(*args, **kwargs)
        finally:
            state["opening"] = False

    monkeypatch.setattr(builtins, "open", flaky_open)
    monkeypatch.setattr(result_files, "recorded_withheld", load_status)
    monkeypatch.setattr(review_lane, "open_round", open_round)
    return failed


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_failed_opening_read_keeps_an_unselected_models_rejection(tmp_path, monkeypatch, capsys,
                                                                   lane_class):
    # Codex's sequence: a rejected answer whose set-aside failed; a round of
    # another model only, whose opening read fails once and whose slot fails;
    # then the rejected model again.
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    _reject_and_leave_the_file(lane, monkeypatch)
    rejection = lane.status()[result_files.REJECTED_MODELS_KEY][SPEC]
    capsys.readouterr()

    failed = _flaky_opening_reads(monkeypatch, every=False)
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output", _silent_failure)
    assert _run(lane, "--force", models=[OTHER]) == 3
    assert failed
    assert lane.status()[result_files.REJECTED_MODELS_KEY] == {SPEC: rejection}
    capsys.readouterr()

    calls: List[int] = []
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Clean answer"]), calls=calls))
    result = _run(lane, "--force")
    out = capsys.readouterr().out

    assert result == 0, out
    assert "[SKIP]" not in out and calls == [1]
    assert _titles(lane) == ["Clean answer"]


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_round_never_opens_over_a_checkpoint_it_cannot_read(tmp_path, monkeypatch, capsys,
                                                             lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    module = _module(lane)
    _reject_and_leave_the_file(lane, monkeypatch)
    before = lane.status()
    capsys.readouterr()

    failed = _flaky_opening_reads(monkeypatch, every=True)
    calls: List[int] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(module, "invoke_with_file_output",
                        _answering_invoke(lane, lane.findings(["Clean answer"]), calls=calls))
    result = _run(lane, "--force", models=[OTHER])
    captured = capsys.readouterr()

    # Stopped before anything was written or dispatched: no maps rebuilt
    # from OTHER alone, nothing of OTHER's run. It read again before it gave up.
    assert result == 3, captured.out
    assert calls == []
    after = lane.status()
    assert after[result_files.REJECTED_MODELS_KEY] == before[result_files.REJECTED_MODELS_KEY]
    assert after["models_requested"] == [SPEC]
    assert (lane.phase_dir / f"{STEM}.json").exists()
    assert "could not be read while this run opened its round" in captured.err
    assert status_file.unreadable_checkpoint_advice(str(lane.phase_dir)) in captured.err
    assert len(failed) == status_file.CHECKPOINT_READ_ATTEMPTS


def test_an_unreadable_checkpoint_withholds_every_model(tmp_path):
    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    status_path = phase_dir / ".status.json"
    status_path.write_text('{"models_rejected": {"codex:gpt-5.5": "home"')
    result = phase_dir / f"{STEM}.json"
    result.write_text("[]")

    withheld = result_files.withheld_models(phase_dir)
    assert withheld and SPEC in withheld and not withheld.recorded(SPEC)
    assert withheld[SPEC] == result_files.UNREADABLE
    assert result_files.withhold_results(phase_dir, [str(result)]) == []
    assert result.exists()
    assert not resume_state._kept_on_rerun(phase_dir, "plan-review", SPEC)
    assert cro.existing_result_reuse(phase_dir, STEM, None).verdict == "withheld"
    # Writers refuse to rebuild the maps from a read that failed.
    assert result_files.record_rejected_model(phase_dir, SPEC, "home") is False
    assert result_files.record_unchecked_model(phase_dir, SPEC, "armed") is False
    assert result_files.clear_unchecked_model(phase_dir, SPEC) is False
    assert review_lane.withheld_record(str(phase_dir), {}) is None
    assert status_path.read_text() == '{"models_rejected": {"codex:gpt-5.5": "home"'


def test_a_record_never_drops_the_entries_a_failed_read_missed(tmp_path, monkeypatch):
    phase_dir = _opened_phase(tmp_path)
    assert result_files.record_rejected_model(phase_dir, "a:one", "home")
    _fail_the_next_checkpoint_read(monkeypatch)
    # The write's own read fails: nothing is written, and it says so.
    assert result_files.record_rejected_model(phase_dir, "b:two", "mode") is False
    assert result_files.rejected_models(phase_dir) == {"a:one": "home"}


# ---------------------------------------------------------------------------
# One safe advice at every stop over an unreadable checkpoint
# ---------------------------------------------------------------------------

def test_the_unreadable_checkpoint_advice_names_the_recovery_and_the_folder_fallback():
    advice = status_file.unreadable_checkpoint_advice("/work/out/review-plan")
    assert advice == (
        "If the read failure has passed, running again is enough. If the file "
        "is damaged, run the phase again without --reaggregate once no run of "
        "this phase is live (`status`): a fresh run (a review fan-out, or the "
        "same ask question) keeps the damaged file as "
        ".status.json.unreadable-<time> and reuses no earlier result in the "
        "folder: every model runs again. If the file cannot be opened at all, or "
        "that recovery stops, move the whole phase folder /work/out/review-plan "
        "aside, or delete it, and run the phase again. Never move .status.json "
        "aside on its own: the result files left beside it would then be read "
        "as results, answers whose safety checks failed among them."
    )
    for phase in status_file.WITHHELD_RECORD_PHASES:
        assert status_file.unreadable_checkpoint_advice(
            "/work/out/review-plan", phase=phase) == advice


def test_a_phase_without_withheld_answers_is_told_its_run_replaces_the_file():
    """Its checkpoint holds run and progress records only: the run replaces a
    damaged one, and moving the file aside alone is safe there."""
    advice = status_file.unreadable_checkpoint_advice("/work/out/implement", phase="implement")
    assert "run the phase again: its run keeps the damaged file" in advice
    # The checkpoint's path joined as the platform joins it (`status_path`): a
    # real phase dir on Windows, `C:\...\implement`, names `C:\...\implement\.status.json`.
    checkpoint = os.path.join("/work/out/implement", ".status.json")
    assert f"move {checkpoint} aside" in advice
    assert "Never move" not in advice and "--reaggregate" not in advice


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_every_stop_over_a_damaged_checkpoint_gives_the_same_advice(tmp_path, monkeypatch,
                                                                    capsys, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _reject_and_leave_the_file(lane, monkeypatch)
    status_path = lane.phase_dir / ".status.json"
    status_path.write_text(status_path.read_text()[:-7])
    damaged = status_path.read_text()
    advice = status_file.unreadable_checkpoint_advice(str(lane.phase_dir))
    capsys.readouterr()

    # A reaggregate, which can only reuse, stops and recovers nothing.
    assert lane.reaggregate("--skip-validation") == 3
    captured = capsys.readouterr()
    assert advice in captured.err
    assert "move it aside" not in captured.err
    assert _nothing_grouped(lane)
    assert status_path.read_text() == damaged

    # A round's start reached over it (a fresh run recovers it before, in its
    # startup) and a round whose checkpoint was damaged after it started stop
    # the same way, never with the takeover's --force.
    from utils import review_round

    with pytest.raises(review_round.RoundTakenOver):
        review_round.start_round(lane.phase_dir)
    claim = review_round.Claim(str(lane.phase_dir), "0123456789abcdef", started=True)
    with pytest.raises(review_round.RoundTakenOver):
        claim.check("write grouped.json")
    captured = capsys.readouterr()
    assert captured.err.count(advice) == 2
    assert "--force" not in captured.out
    assert status_path.read_text() == damaged
    # A fresh run recovering it is test_damaged_checkpoint_recovery.py's.


def test_the_last_resort_stop_over_an_unreadable_checkpoint_gives_the_same_advice(tmp_path,
                                                                                  capsys):
    phase_dir = _opened_phase(tmp_path)
    (phase_dir / f"{STEM}.json").write_text("[]")
    (phase_dir / ".status.json").write_text('{"models_rejected": ')
    rejected = review_lane.SlotResult(SPEC, False, error="home", rejected="home")
    with pytest.raises(SystemExit) as stopped:
        review_lane._stop_unless_withheld(
            str(phase_dir), _close_steps(), {SPEC: rejected}, [], settled=True,
            failed_models={SPEC: "home"}, served_models={}, withheld={},
        )
    assert stopped.value.code == 3
    err = capsys.readouterr().err
    assert f"{SPEC}: {STEM}.json" in err
    assert status_file.unreadable_checkpoint_advice(str(phase_dir)) in err
    assert "Move these files out of the folder" not in err


# ---------------------------------------------------------------------------
# Controls: what is not rejected is not touched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_an_ordinary_failed_slot_is_still_salvaged_and_adopted(tmp_path, monkeypatch, capsys,
                                                               lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    findings = lane.findings(["Salvaged answer"])
    repaired: List[Path] = []
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output", _failing_invoke(lane, findings))
    monkeypatch.setattr(task_file_runner, "run_task_file", _repairer(lane, findings, repaired))

    result = _run(lane, "--exec-subagents")
    out = capsys.readouterr().out

    assert result == 0, out
    assert "[SALVAGE_NEEDED]" in out and "recovered by salvage" in out
    assert repaired
    assert lane.phase_completed()
    assert _titles(lane) == ["Salvaged answer"]
    status = lane.status()
    assert SPEC not in status.get("models_failed", {})
    assert not status.get("models_rejected")


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_an_ordinary_failed_slot_salvaged_by_hand_is_reaggregated(tmp_path, monkeypatch, capsys,
                                                                  lane_class):
    # Its checks passed, so the record written before its dispatch is gone
    # and the documented salvage-then-reaggregate flow reads the repair.
    lane = _lane(lane_class, tmp_path, monkeypatch)
    findings = lane.findings(["Salvaged answer"])
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output", _failing_invoke(lane, findings))

    assert _run(lane) == 3
    assert lane.status()[result_files.UNCHECKED_MODELS_KEY] == {}
    assert _salvage_requests(lane) == sorted([f"salvage_{STEM}.json", f"salvage_task_{STEM}.json"])
    _repairer(lane, findings, [])(str(lane.phase_dir / f"salvage_task_{STEM}.json"))
    capsys.readouterr()

    result = lane.reaggregate("--skip-validation")
    out = capsys.readouterr().out
    assert result == 0, out
    assert _titles(lane) == ["Salvaged answer"]


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_bypassed_slot_its_audit_cleared_is_not_rejected(tmp_path, monkeypatch, capsys,
                                                           lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    findings = lane.findings(["Kept answer"])
    _probe(monkeypatch, "write")
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _answering_invoke(lane, findings, check="write", audit=CLEAN))

    result = _run(lane)
    out = capsys.readouterr().out

    assert result == 0, out
    assert "not attributed" in out
    assert (lane.phase_dir / f"{STEM}.json").exists()
    assert not (lane.phase_dir / f"{STEM}.json.rejected").exists()
    assert _titles(lane) == ["Kept answer"]
    assert not lane.status().get("models_rejected")
    assert not lane.status().get("models_unchecked")


# ---------------------------------------------------------------------------
# The slot's own outcome, on both lanes
# ---------------------------------------------------------------------------

def _slot(lane, invoke) -> review_lane.SlotResult:
    """The lane's slot function on :data:`SPEC`, its provider *invoke*."""
    lane.phase_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(lane, PlanLane):
        call = rob.run_single_model(
            SPEC, str(lane.plan), "{prefix}{output_location}{working_limits}", {},
            "review-plan", "plan_review", 30, out_dir=str(lane.phase_dir.parent),
            prefix="feature", run_model_specs=[SPEC],
        )
    else:
        call = cro.run_single_review(SPEC, lane.plan, lane.phase_dir.parent, timeout=30,
                                     usage={}, run_model_specs=[SPEC])
    with patch.object(_module(lane), "invoke_with_file_output", invoke):
        return review_lane.SlotResult.coerce(SPEC, asyncio.run(call))


@pytest.mark.parametrize("lane_class", [PlanLane, CodeLane], ids=["review-plan", "review-code"])
def test_a_rejected_slot_carries_its_rejection(tmp_path, monkeypatch, lane_class):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    findings = lane.findings(["Rejected answer"])
    _probe(monkeypatch, "home")
    outcome = _slot(lane, _failing_invoke(lane, findings, check="home"))

    assert outcome.rejected and outcome.rejected == outcome.error
    assert not outcome.success and outcome.payload is None
    if lane_class is PlanLane:
        # The record suppresses the stdout fallback with an explicit "".
        assert outcome.details[review_lane.REJECTED_KEY] == outcome.rejected
        assert rob.failure_output_from_result(outcome.details) == ""
        assert "Rejected answer" in outcome.details["stdout"]
    assert lane.save_slot(outcome) is False
    assert _salvage_requests(lane) == []


def test_a_review_triple_reads_as_a_plain_triple():
    triple = review_lane.SlotResult(SPEC, False, error="checks", rejected="checks").as_review()
    assert triple == (False, None, "checks")
    success, issues, error = triple
    assert (success, issues, error) == (False, None, "checks")
    checked = review_lane.SlotResult(SPEC, True, payload=[], checked=True).as_review()
    assert checked == (True, [], None)
    for copied in (copy.deepcopy(triple), pickle.loads(pickle.dumps(triple))):
        assert copied == triple
        assert review_lane.SlotResult.from_review(SPEC, copied).rejected == "checks"
    for copied in (copy.deepcopy(checked), pickle.loads(pickle.dumps(checked))):
        assert review_lane.SlotResult.from_review(SPEC, copied).checked is True
    assert type(review_lane.SlotResult(SPEC, True, payload=[]).as_review()) is tuple


def test_only_a_checked_attempt_supersedes_a_withheld_answer(tmp_path):
    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    status_file.write_status(str(phase_dir), {"phase": "review-plan"})
    specs = ("ok:checked", "ok:failed", "no:refused", "no:raised", "re:jected", "c:lost",
             "gone:unselected")
    for spec in specs:
        assert result_files.record_rejected_model(phase_dir, spec, f"earlier {spec}")
        assert result_files.record_unchecked_model(phase_dir, spec, f"armed {spec}")
    slots = {
        "ok:checked": review_lane.SlotResult("ok:checked", True, payload=[], checked=True),
        "ok:failed": review_lane.SlotResult("ok:failed", False, error="parse", checked=True),
        "no:refused": review_lane.SlotResult("no:refused", False, error="refused"),
        "no:raised": review_lane.SlotResult("no:raised", False, error="boom"),
        "re:jected": review_lane.SlotResult("re:jected", False, error="now", rejected="now"),
    }
    record = review_lane.withheld_record(str(phase_dir), slots)
    # Checked attempts supersede both records, failed or not; a refused or
    # raised slot, a deadline loss and a model no longer selected supersede
    # nothing; a rejection now replaces the model's unchecked entry.
    kept = {spec for spec in specs if not spec.startswith("ok:")}
    assert set(record[result_files.REJECTED_MODELS_KEY]) == kept
    assert record[result_files.REJECTED_MODELS_KEY]["re:jected"] == "now"
    assert set(record[result_files.UNCHECKED_MODELS_KEY]) == kept - {"re:jected"}


def test_a_checkpoint_without_the_records_withholds_nothing(tmp_path):
    # Older checkpoints, written before either record existed, resume as before.
    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    status_file.write_status(str(phase_dir), {"phase": "review-plan", "models_failed": {}})
    assert not result_files.withheld_models(phase_dir)
    (phase_dir / f"{STEM}.json").write_text("[]")
    assert result_files.withhold_results(phase_dir, [str(phase_dir / f"{STEM}.json")]) == [
        str(phase_dir / f"{STEM}.json")
    ]
