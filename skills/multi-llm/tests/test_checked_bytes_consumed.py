"""Review and ask consumers use the bytes they checked, never the file again.

Code health U91. A concurrent writer (a crashed run's surviving provider, say)
can replace a file between the moment a run checked or saved it and the moment
the run used to read it back. Each test here injects that replacement at the
old re-read, on a healthy phase (no recovery record), and shows it never
reaches the report, the result file, the gate or ``answers.md``; the controls
show that without the replacement the run reports what it always did
(docs/development/review-results.md, "Consuming checked bytes").

The windows: the fan-out's grouping (1), a reaggregate's verdicts, task file
and grouping (2), the validation merge (3), review-code's heal of a lost
completion (4), ask's closing scan (5) and the salvage dispatch (6). A
recovered phase's validation round (3, cont.) merges only what its own
dispatch wrote.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
import review_tasks_orchestrator as rto  # noqa: E402
from utils import review_lane, review_round, task_file_runner, task_result  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.subagent_tasks import content_hash, tasks_of  # noqa: E402
from tests.test_damaged_checkpoint_recovery import _invokes  # noqa: E402
from tests.test_rejected_slot_answers import (  # noqa: E402
    SPEC, STEM, _failing_invoke, _lane, _module, _probe, _titles,
)
from tests.test_review_lane_parity import CodeLane, PlanLane  # noqa: E402

#: What the concurrent writer puts in place of the checked file.
REPLACED = "Unchecked replacement"
CHECKED = "Checked answer"
CHECKED_VERDICT = "Checked verdict"
REPLACED_VERDICT = "Unchecked replacement verdict"


class TaskLane(PlanLane):
    """review-tasks, on the review base's lane like review-plan."""

    mode, phase, module = "review-tasks", "review-tasks", rto

    def __init__(self, root, monkeypatch):
        super().__init__(root, monkeypatch)
        tasks = self.plan.parent / self.plan.stem / "tasks"
        tasks.mkdir(parents=True, exist_ok=True)
        (tasks / "tasks.md").write_text(
            "# Tasks\n\n## T001: Change handler\nImplement the handler.\n")
        state = get_or_create_state(self.plan)
        state.mark_phase_completed("generate-tasks")
        state.save()


BASE_LANES = pytest.mark.parametrize("lane_class", [PlanLane, TaskLane])
BOTH_LANES = pytest.mark.parametrize("lane_class", [PlanLane, CodeLane])
ALL_LANES = pytest.mark.parametrize("lane_class", [PlanLane, TaskLane, CodeLane])


def _healthy(lane_class, tmp_path, monkeypatch, *, high: bool = False):
    """A lane on a fresh phase whose provider answers :data:`CHECKED` and
    passes its checks; its finding is HIGH when *high*."""
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _probe(monkeypatch, None)
    if high:
        findings = lane.findings
        lane.findings = lambda names: findings(names, high=names)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _invokes(lane, {SPEC: [CHECKED]}, []))
    return lane


def _go(lane, *flags: str) -> int:
    return lane.run(
        "--models", SPEC, "--timeout", "120", "--max-parallel", "1", "--force", *flags,
        patches=(
            patch("utils.interactive.resolve_models", return_value=[SPEC]),
            patch("utils.interactive.is_model_valid", return_value=True),
        ),
    )


def _report(lane) -> str:
    return (lane.phase_dir / "report.md").read_text(encoding="utf-8")


#: The report's verdict counts when the one group's checked verdict, valid,
#: is the one shown (a replacement says invalid).
CHECKED_COUNTS = "**Validation:** 1 valid, 0 invalid"


def _verdicts(document: Dict[str, Any], status: str, reason: str) -> Dict[int, List[Dict]]:
    """Each batch's verdicts on its own groups, by batch index."""
    return {
        batch["batch_index"]: [
            {"group_index": index, "group_hash": gid, "status": status,
             "reason": reason, "confidence": 0.9}
            for index, gid in zip(batch["group_indices"], batch["group_ids"])
        ]
        for batch in document["batches"]
    }


def _writing_dispatch(lane, status: str = "valid", reason: str = CHECKED_VERDICT):
    """A host-style validation dispatch: it writes every batch file itself,
    with no executor record (the merge reads each file once)."""
    async def dispatch(claim, steps, *args, **kwargs):
        for index, rows in _verdicts(kwargs["document"], status, reason).items():
            (lane.phase_dir / f"validation_batch_{index}.json").write_text(json.dumps(rows))
        return True
    return dispatch


# ---------------------------------------------------------------------------
# 1. The fan-out groups the answers it saved
# ---------------------------------------------------------------------------

@ALL_LANES
def test_the_fanout_groups_and_reports_the_answers_it_saved(
        tmp_path, monkeypatch, lane_class):
    """A model's result file replaced after the close saved it: grouping and
    the report use the answer the slot returned. (review-code always did;
    review-plan and review-tasks read the saved files back.)"""
    lane = _healthy(lane_class, tmp_path, monkeypatch)
    real = review_lane.close_fanout

    async def close_then_replace(*args, **kwargs):
        closed = await real(*args, **kwargs)
        (lane.phase_dir / f"{STEM}.json").write_text(
            json.dumps(lane.findings([REPLACED])))
        return closed

    monkeypatch.setattr(review_lane, "close_fanout", close_then_replace)
    assert _go(lane, "--skip-validation") == 0
    assert _titles(lane) == [CHECKED]
    report = _report(lane)
    assert CHECKED in report and REPLACED not in report


@ALL_LANES
def test_control_a_fanout_with_no_writer_groups_its_answer(tmp_path, monkeypatch, lane_class):
    lane = _healthy(lane_class, tmp_path, monkeypatch)
    assert _go(lane, "--skip-validation") == 0
    assert _titles(lane) == [CHECKED]
    assert json.loads((lane.phase_dir / f"{STEM}.json").read_text())[0]["title"] == CHECKED
    assert CHECKED in _report(lane)


# ---------------------------------------------------------------------------
# 6. (with 1) A salvaged model joins the round as its dispatch persisted it
# ---------------------------------------------------------------------------

def _repairing_executor(lane, repaired: List[Dict], *, replace_after: bool):
    """The salvage executor's persist leg, for real (``persist_result``, bound
    to the round's claim), then a concurrent writer over its output."""
    calls: List[str] = []

    def run_task_file(task_file, *, phase_dir, **_kwargs):
        task = tasks_of(json.loads(Path(task_file).read_text(encoding="utf-8")))[0]
        written = task_result.persist_result(task, phase_dir, repaired)
        calls.append(written)
        if replace_after:
            Path(written).write_text(json.dumps(lane.findings([REPLACED])))
        return SimpleNamespace(outcomes=[SimpleNamespace(ok=True, output_path=written)])

    return run_task_file, calls


@BOTH_LANES
@pytest.mark.parametrize("replace_after", [True, False])
def test_a_salvaged_model_joins_as_its_dispatch_persisted_it(
        tmp_path, monkeypatch, lane_class, replace_after):
    lane = _lane(lane_class, tmp_path, monkeypatch)
    _probe(monkeypatch, None)
    monkeypatch.setattr(_module(lane), "invoke_with_file_output",
                        _failing_invoke(lane, lane.findings(["Checked raw answer"])))
    executor, calls = _repairing_executor(
        lane, lane.findings(["Checked repair"]), replace_after=replace_after)
    monkeypatch.setattr(task_file_runner, "run_task_file", executor)
    assert _go(lane, "--exec-subagents", "--skip-validation") == 0
    assert len(calls) == 1
    assert _titles(lane) == ["Checked repair"]
    report = _report(lane)
    assert "Checked repair" in report and REPLACED not in report


def test_a_salvage_request_replaced_after_its_check_refuses_an_empty_repair(tmp_path):
    """The executor checked the request against the digest the task binds;
    the failure kind is then read from the same request: a request replaced
    in between (one that no longer says the run timed out) is not the one
    the task was bound to, so a salvaged ``[]`` is refused."""
    phase_dir = tmp_path / "review-plan"
    phase_dir.mkdir()
    request_path = rob._write_salvage_request(
        model=SPEC, sanitized_model=STEM, raw_output='[{"title": "x"',
        phase_dir=str(phase_dir), phase_name="review_plan", failure="timeout",
    )
    task_path = phase_dir / f"salvage_task_{STEM}.json"
    task = tasks_of(json.loads(task_path.read_text(encoding="utf-8")))[0]
    assert task_result.salvage_failure_of(task, str(phase_dir)) == "timeout"
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    del request["failure"]
    Path(request_path).write_text(json.dumps(request, indent=2))
    assert task_result.salvage_failure_of(task, str(phase_dir)) == "timeout"
    # Control: a request the task binds, with no failure kind, has none.
    del task["inputs"]["salvage_request_sha256"]
    assert task_result.salvage_failure_of(task, str(phase_dir)) is None


# ---------------------------------------------------------------------------
# 3. The validation merge reads the batches its dispatch persisted
# ---------------------------------------------------------------------------

def _persisting_dispatch(lane, *, replace_after: bool):
    """The executor's persist leg for every validation batch, for real
    (``persist_result`` under the dispatch's claim), then a late writer over
    each batch file."""
    def dispatch_subagent_tasks(_args, *, document, phase_dir, **_kwargs):
        checked = _verdicts(document, "valid", CHECKED_VERDICT)
        late = _verdicts(document, "invalid", REPLACED_VERDICT)
        for task in tasks_of(document):
            index = task["batch_index"]
            written = task_result.persist_result(task, phase_dir, checked[index])
            if replace_after:
                Path(written).write_text(json.dumps(late[index]))
        return True
    return dispatch_subagent_tasks


@BOTH_LANES
@pytest.mark.parametrize("replace_after", [True, False])
def test_the_merge_reads_the_batches_its_dispatch_persisted(
        tmp_path, monkeypatch, lane_class, replace_after):
    lane = _healthy(lane_class, tmp_path, monkeypatch)
    monkeypatch.setattr(_module(lane), "dispatch_subagent_tasks",
                        _persisting_dispatch(lane, replace_after=replace_after))
    assert _go(lane, "--exec-subagents") == 0
    validation = lane.json("validation.json")
    assert {g["status"] for g in validation["groups"]} == {"valid"}
    assert {g["reason"] for g in validation["groups"]} == {CHECKED_VERDICT}
    assert CHECKED_COUNTS in _report(lane)


@BOTH_LANES
def test_the_merge_binds_the_bytes_it_merged(tmp_path, monkeypatch, lane_class):
    """With no executor record (a host wrote the batches), each batch is read
    once: the digest validation.json records is of the bytes whose verdicts it
    holds, even when the file is replaced right after a digest is taken."""
    from utils import subagent_tasks

    lane = _healthy(lane_class, tmp_path, monkeypatch)
    monkeypatch.setattr(_module(lane), "dispatch_in_round", _writing_dispatch(lane))
    real_digest = subagent_tasks.file_bytes_sha256
    replaced: List[str] = []

    def digest_then_replace(path):
        digest = real_digest(path)
        if path and Path(path).name.startswith("validation_batch_") and not replaced:
            rows = json.loads(Path(path).read_text())
            for row in rows:
                row["status"], row["reason"] = "invalid", REPLACED_VERDICT
            Path(path).write_text(json.dumps(rows))
            replaced.append(path)
        return digest

    monkeypatch.setattr(subagent_tasks, "file_bytes_sha256", digest_then_replace)
    assert _go(lane, "--exec-subagents") == 0
    validation = lane.json("validation.json")
    assert {g["reason"] for g in validation["groups"]} == {CHECKED_VERDICT}
    for index, digest in validation["metadata"]["batch_sha256"].items():
        merged = (lane.phase_dir / f"validation_batch_{index}.json").read_bytes()
        assert hashlib.sha256(merged).hexdigest() == digest


def test_a_bound_dispatch_remembers_what_it_persisted_and_found_complete(tmp_path):
    """The two points the executor checks a result, persisted after its
    checks or found complete under resume, leave those bytes on the claim the
    dispatch is bound to; outside a dispatch nothing is remembered."""
    claim = review_round.start_round(tmp_path)
    task = {"task_id": "validation-batch-0", "kind": "validation",
            "output_name": "validation_batch_0.json", "batch_index": 0,
            "group_ids": ["1111222233334444"], "group_indices": [0]}
    rows = [{"group_index": 0, "group_hash": "1111222233334444",
             "status": "valid", "reason": "ok", "confidence": 0.9}]
    with review_round.results_bound_to(claim):
        task_result.persist_result(task, str(tmp_path), rows)
    persisted = claim.checked_result("validation_batch_0.json")
    assert persisted == (tmp_path / "validation_batch_0.json").read_bytes()
    (tmp_path / "validation_batch_0.json").write_text(json.dumps(rows))
    with review_round.results_bound_to(claim):
        assert claim.checked_result("validation_batch_0.json") is None  # a new dispatch
        assert task_result.task_is_complete(task, str(tmp_path))
    assert claim.checked_result("validation_batch_0.json") == json.dumps(rows).encode()
    other = review_round.start_round(tmp_path / "elsewhere")
    task_result.task_is_complete(task, str(tmp_path))
    assert other.checked_result("validation_batch_0.json") is None


# ---------------------------------------------------------------------------
# 2. A reaggregate reports the verdicts and the grouping it checked
# ---------------------------------------------------------------------------

@BOTH_LANES
def test_a_reaggregate_reports_the_verdicts_it_checked(tmp_path, monkeypatch, lane_class):
    lane = _healthy(lane_class, tmp_path, monkeypatch)
    monkeypatch.setattr(_module(lane), "dispatch_in_round", _writing_dispatch(lane))
    assert _go(lane, "--exec-subagents") == 0
    real = review_lane.validation_source_decision

    def judge_then_replace(validation_path, *args, **kwargs):
        decision = real(validation_path, *args, **kwargs)
        data = json.loads(Path(validation_path).read_text(encoding="utf-8"))
        for group in data["groups"]:
            group["status"], group["reason"] = "invalid", REPLACED_VERDICT
        Path(validation_path).write_text(json.dumps(data))
        return decision

    monkeypatch.setattr(review_lane, "validation_source_decision", judge_then_replace)
    assert lane.reaggregate() == 0
    assert CHECKED_COUNTS in _report(lane)


@BOTH_LANES
def test_a_reaggregate_checks_the_grouping_it_holds(tmp_path, monkeypatch, lane_class):
    """grouped.json changes on disk after the reaggregate read it: the
    verdicts are judged against the grouping it read and reports (bound, so
    reused), never against a second read of the file."""
    lane = _healthy(lane_class, tmp_path, monkeypatch)
    monkeypatch.setattr(_module(lane), "dispatch_in_round", _writing_dispatch(lane))
    assert _go(lane, "--exec-subagents") == 0
    real = review_lane.load_groups_payload
    seen: List[bool] = []

    def read_then_replace(payload):
        groups = real(payload)
        if not seen:
            seen.append(True)
            data = json.loads((lane.phase_dir / "grouped.json").read_text())
            data_groups = data["groups"] if isinstance(data, dict) else data
            data_groups[0]["theme"] = "changed after the read"
            (lane.phase_dir / "grouped.json").write_text(json.dumps(data))
        return groups

    monkeypatch.setattr(review_lane, "load_groups_payload", read_then_replace)
    assert lane.reaggregate() == 0
    assert seen
    assert not list(lane.phase_dir.glob("validation*.stale.json"))
    assert CHECKED_COUNTS in _report(lane)


@BASE_LANES
def test_an_unvalidated_reaggregate_reports_the_grouping_it_holds(
        tmp_path, monkeypatch, lane_class):
    """``--reaggregate --skip-validation`` regroups and reports: the report
    shows the groups it wrote, never grouped.json read back."""
    lane = _healthy(lane_class, tmp_path, monkeypatch)
    assert _go(lane, "--skip-validation") == 0
    real = review_lane.bind_grouping

    def bind_then_replace(*args, **kwargs):
        bound = real(*args, **kwargs)
        data = json.loads((lane.phase_dir / "grouped.json").read_text())
        groups = data["groups"] if isinstance(data, dict) else data
        groups[0]["suggestions"][0]["title"] = REPLACED
        (lane.phase_dir / "grouped.json").write_text(json.dumps(data))
        return bound

    monkeypatch.setattr(review_lane, "bind_grouping", bind_then_replace)
    assert lane.reaggregate("--skip-validation") == 0
    report = _report(lane)
    assert CHECKED in report and REPLACED not in report


# ---------------------------------------------------------------------------
# 4. review-code's heal counts the grouping its seal check bound
# ---------------------------------------------------------------------------

def _gated_round(tmp_path, monkeypatch, capsys, *, swap_after_grouping=False):
    """A review-code round with a HIGH finding, validated in-process and
    gated (exit 3), whose completion record is then lost. With
    *swap_after_grouping* a writer replaces grouped.json, every finding LOW,
    right after the round wrote it."""
    lane = _healthy(CodeLane, tmp_path, monkeypatch, high=True)
    monkeypatch.setattr(cro, "dispatch_in_round", _writing_dispatch(lane))
    if swap_after_grouping:
        real = review_lane.group_findings

        def group_then_swap(*args, **kwargs):
            grouping = real(*args, **kwargs)
            _lower_everything(lane)
            return grouping

        monkeypatch.setattr(review_lane, "group_findings", group_then_swap)
    assert _go(lane, "--exec-subagents", "--fail-on-high") == 3
    state = get_or_create_state(lane.plan)
    state.state["phases_completed"].pop("code-review")
    state.save()
    capsys.readouterr()
    return lane


def _lower_everything(lane) -> str:
    data = json.loads((lane.phase_dir / "grouped.json").read_text(encoding="utf-8"))
    groups = data["groups"] if isinstance(data, dict) else data
    for group in groups:
        for suggestion in group.get("suggestions") or []:
            suggestion["importance"] = "LOW"
        if "importance" in group:
            group["importance"] = "LOW"
    text = json.dumps(data)
    (lane.phase_dir / "grouped.json").write_text(text)
    return text


def test_control_the_heal_finishes_a_lost_completion_and_gates_it(
        tmp_path, monkeypatch, capsys):
    lane = _gated_round(tmp_path, monkeypatch, capsys)
    assert lane.run("--models", SPEC, "--fail-on-high") == 3
    out = capsys.readouterr().out
    assert "Auto-healed" in out and "High priority: 1 group(s)" in out


def test_a_pair_replaced_after_the_seal_check_never_heals(tmp_path, monkeypatch, capsys):
    """Codex's repro: after the guard checked the report's seal, a writer
    replaces grouped.json (every finding LOW) and validation.json (its digest
    updated to match). The heal reads the grouping digest from the bytes the
    seal bound, so the new grouping does not check out: exit 3, no heal."""
    lane = _gated_round(tmp_path, monkeypatch, capsys)
    real = cro.heal_unrecorded_completion

    def replace_pair(*args, **kwargs):
        text = _lower_everything(lane)
        validation = lane.json("validation.json")
        validation["metadata"]["grouped_sha256"] = content_hash(text)
        (lane.phase_dir / "validation.json").write_text(json.dumps(validation))
        return real(*args, **kwargs)

    monkeypatch.setattr(cro, "heal_unrecorded_completion", replace_pair)
    assert lane.run("--models", SPEC, "--fail-on-high") == 3
    out = capsys.readouterr().out
    assert "Auto-healed" not in out
    assert "High priority: 0" not in out


def test_a_grouping_replaced_after_its_write_is_never_bound(tmp_path, monkeypatch, capsys):
    """grouped.json replaced (every finding LOW) right after the round wrote
    it: the task file and validation.json bind the grouping the round wrote,
    so the later heal refuses the replacement instead of healing to zero."""
    lane = _gated_round(tmp_path, monkeypatch, capsys, swap_after_grouping=True)
    tasks = lane.json("validation_tasks.json")
    validation = lane.json("validation.json")
    on_disk = content_hash((lane.phase_dir / "grouped.json").read_text(encoding="utf-8"))
    assert tasks["inputs"]["grouped_sha256"] != on_disk
    assert validation["metadata"]["grouped_sha256"] == tasks["inputs"]["grouped_sha256"]
    assert lane.run("--models", SPEC, "--fail-on-high") == 3
    out = capsys.readouterr().out
    assert "Auto-healed" not in out and "High priority: 0" not in out


# ---------------------------------------------------------------------------
# 5. Ask counts and writes the answer each checked slot read or wrote
# ---------------------------------------------------------------------------

from tests.test_ask import FakeInvoker, answer_of, ask_argv, qdir, read_answers, run_ask  # noqa: E402
from tests.test_ask_withheld_answers import plan  # noqa: E402,F401
import ask_orchestrator as ask  # noqa: E402

#: Answers through its file (``claude-code``), and on stdout (``codex``, whose
#: answer the slot persists itself).
FILE_SPEC = "claude-code:haiku"


@pytest.mark.parametrize("spec", [FILE_SPEC, SPEC], ids=["file", "stdout"])
@pytest.mark.parametrize("replace_after", [True, False])
def test_ask_counts_and_writes_the_answer_its_checked_slot_held(
        plan, monkeypatch, spec, replace_after):  # noqa: F811
    d = qdir(plan, "Q")
    real = ask.withheld_after_run
    seen: List[Dict[str, Any]] = []

    def replace_then_close(specs, results, maps):
        seen.append(results[spec])
        if replace_after:
            answer_of(d, spec).write_text(REPLACED, encoding="utf-8")
        return real(specs, results, maps)

    monkeypatch.setattr(ask, "withheld_after_run", replace_then_close)
    invoker = FakeInvoker()
    invoker.configure(spec, write_file=True, content=CHECKED)
    assert run_ask(ask_argv(plan, spec), invoker) == 0
    assert seen and seen[0]["success"] and seen[0][ask.CHECKED_KEY]
    answers = read_answers(d)
    assert CHECKED in answers and REPLACED not in answers
    assert "**Answered:** 1/1" in answers


# ---------------------------------------------------------------------------
# 2. (cont.) A validation.json unreadable at first is never reused unchecked
# ---------------------------------------------------------------------------

@BOTH_LANES
def test_a_validation_file_unreadable_at_first_is_never_reused_unchecked(
        tmp_path, monkeypatch, capsys, lane_class):
    """The reaggregate's one read of validation.json fails (a transient
    error); a later read could see a replacement nothing checked. Nothing
    was checked, so the file is retired as one the decision cannot read, the
    round's batches are merged again, and the replacement never reaches the
    report."""
    lane = _healthy(lane_class, tmp_path, monkeypatch)
    monkeypatch.setattr(_module(lane), "dispatch_in_round", _writing_dispatch(lane))
    assert _go(lane, "--exec-subagents") == 0
    validation = lane.phase_dir / "validation.json"
    real_read = Path.read_bytes
    failed: List[bool] = []

    def fail_once(path):
        if Path(path) == validation and not failed:
            failed.append(True)
            raise OSError("temporary read failure")
        return real_read(path)

    real_judge = review_lane.validation_source_decision

    def judge_then_replace(validation_path, *args, **kwargs):
        decision = real_judge(validation_path, *args, **kwargs)
        data = json.loads(Path(validation_path).read_text(encoding="utf-8"))
        for group in data["groups"]:
            group["status"], group["reason"] = "invalid", REPLACED_VERDICT
        Path(validation_path).write_text(json.dumps(data))
        return decision

    monkeypatch.setattr(Path, "read_bytes", fail_once)
    monkeypatch.setattr(review_lane, "validation_source_decision", judge_then_replace)
    capsys.readouterr()
    assert lane.reaggregate() == 0
    assert failed
    report = _report(lane)
    assert CHECKED_COUNTS in report and REPLACED_VERDICT not in report
    assert "Not using validation.json: could not read validation.json" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 2. (cont.) A reused round is judged on one read of its task file
# ---------------------------------------------------------------------------

def _round_swapped_after_one_read(monkeypatch, lane, later: Dict[str, Any]) -> List[bool]:
    """Replace validation_tasks.json with *later* (atomically, so a handle
    already open keeps reading the old file) right after the first time it is
    opened once the reaggregate has read validation.json's bytes to judge
    them: the moment a reaggregate that reads the task file twice has judged
    the verdicts' round on one copy and has yet to check inputs and snapshots
    on the other."""
    import builtins
    import io
    import os as _os

    taskfile = lane.phase_dir / "validation_tasks.json"
    validation = lane.phase_dir / "validation.json"
    real_open = io.open
    judged: List[bool] = []
    swapped: List[bool] = []

    def watching_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if isinstance(file, (str, Path)) and set(mode) <= set("rb"):
            if Path(file) == validation and "b" in mode:
                judged.append(True)
            elif Path(file) == taskfile and judged and not swapped:
                swapped.append(True)
                staged = taskfile.with_name("validation_tasks.json.swap")
                staged.write_text(json.dumps(later), encoding="utf-8")
                _os.replace(staged, taskfile)
        return handle

    monkeypatch.setattr(io, "open", watching_open)
    monkeypatch.setattr(builtins, "open", watching_open)
    return swapped


def _another_round(document: Dict[str, Any]) -> Dict[str, Any]:
    """*document* as another validation round's (its manifest differs)."""
    later = json.loads(json.dumps(document))
    for task in later["batches"]:
        task["task_id"] += "-next-round"
    return later


@BOTH_LANES
@pytest.mark.parametrize("swap", [True, False], ids=["swapped", "control"])
def test_a_reused_round_is_judged_on_one_read_of_its_task_file(
        tmp_path, monkeypatch, capsys, lane_class, swap):
    """What the verdicts were reached on moves (review-plan: the round's
    large plan snapshot; review-code: a reviewed line), so the reaggregate
    refuses (exit 3). A writer replaces the task file, after the verdicts'
    round was judged on it, with another round's that records the moved
    value: the inputs and snapshots are still checked on the copy the round
    was judged on, so it still refuses, and the old verdicts are not reused."""
    from tests.test_validation_context_by_path import big_doc

    lane = _healthy(lane_class, tmp_path, monkeypatch)
    if lane_class is PlanLane:
        lane.plan.write_text(big_doc("Original"), encoding="utf-8")
    monkeypatch.setattr(_module(lane), "dispatch_in_round", _writing_dispatch(lane))
    assert _go(lane, "--exec-subagents") == 0
    document = lane.json("validation_tasks.json")
    later = _another_round(document)
    if lane_class is PlanLane:
        snapshot = lane.phase_dir / "validation-snapshot.md"
        snapshot.write_text(big_doc("Later context"), encoding="utf-8")
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        sources = [source for task in later["batches"]
                   for source in task["context_sources"] if source["mode"] == "path"]
        assert sources
        for source in sources:
            source["sha256"] = digest
    else:
        from utils.subagent_tasks import build_inputs

        lane.edit_a_reviewed_line()
        later["inputs"]["reviewed_code_sha256"] = build_inputs(
            reviewed_code=lane.reviewed_code())["reviewed_code_sha256"]
    merged = (lane.phase_dir / "validation.json").read_bytes()
    swapped = _round_swapped_after_one_read(monkeypatch, lane, later) if swap else []
    capsys.readouterr()
    code = lane.reaggregate()
    captured = capsys.readouterr()
    assert bool(swapped) is swap
    assert code == 3
    assert "Using validation.json" not in captured.out
    assert (lane.phase_dir / "validation.json").read_bytes() == merged


# ---------------------------------------------------------------------------
# 6. (cont.) A bound salvage request that cannot be verified refuses `[]`
# ---------------------------------------------------------------------------

def _base_salvage_request(phase: Path) -> Path:
    return Path(rob._write_salvage_request(
        model=SPEC, sanitized_model=STEM, raw_output='[{"title": "partial"',
        phase_dir=str(phase), phase_name="review_plan", failure="timeout",
    ))


def _code_salvage_request(phase: Path) -> Path:
    request = {
        "model": SPEC, "phase": "code_review", "raw_output": '[{"title": "partial"',
        "expected_type": "array", "expected_schema": {"fields": ["title"]},
        "output_path": str(phase / f"{STEM}.json"), "failure": "timeout",
    }
    path = phase / f"salvage_{STEM}.json"
    path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    cro.emit_salvage_task(request, path, phase, phase / "plan.md")
    return path


#: How a checked request can stop being readable or verifiable.
_BAD_REQUESTS = {
    "malformed": b"{",
    "empty": b"",
    "missing": None,
    "invalid-utf8": b"\xff\xfe",
    "another-request": json.dumps({"model": SPEC, "phase": "review_plan"}).encode(),
}


@pytest.mark.parametrize("write_request", [_base_salvage_request, _code_salvage_request],
                         ids=["review-plan", "review-code"])
@pytest.mark.parametrize("replacement", list(_BAD_REQUESTS), ids=list(_BAD_REQUESTS))
def test_a_bound_salvage_request_that_cannot_be_verified_refuses_an_empty_repair(
        tmp_path, write_request, replacement):
    """A timed-out run's request, checked when its task was bound, is then
    truncated, deleted, made undecodable or replaced: it is no longer the
    request the task binds, so its failure kind is unknown and read as a
    timeout. A salvaged ``[]`` is refused, a real repair still persists."""
    phase = tmp_path / "phase"
    phase.mkdir()
    request = write_request(phase)
    task = tasks_of(json.loads(
        (phase / f"salvage_task_{STEM}.json").read_text(encoding="utf-8")))[0]
    assert task["inputs"]["salvage_request_sha256"]
    assert task_result.salvage_failure_of(task, str(phase)) == "timeout"
    data = _BAD_REQUESTS[replacement]
    if data is None:
        request.unlink()
    else:
        request.write_bytes(data)
    failure = task_result.salvage_failure_of(task, str(phase))
    assert failure == "timeout"
    assert (task_result._salvage_gave_up("salvage", [], failure=failure)
            == task_result.SALVAGE_EMPTY_AFTER_TIMEOUT_REASON)
    assert task_result._salvage_gave_up(
        "salvage", [{"title": "repaired"}], failure=failure) is None


def test_a_bound_request_that_is_not_an_object_refuses_an_empty_repair(tmp_path):
    """Even when its text is the one the task binds, a request that is not a
    JSON object says nothing about how the run ended: read as a timeout."""
    phase = tmp_path / "phase"
    phase.mkdir()
    request = _base_salvage_request(phase)
    task = tasks_of(json.loads(
        (phase / f"salvage_task_{STEM}.json").read_text(encoding="utf-8")))[0]
    request.write_text("[]", encoding="utf-8")
    task["inputs"]["salvage_request_sha256"] = content_hash("[]")
    assert task_result.salvage_failure_of(task, str(phase)) == "timeout"


def test_control_an_unbound_salvage_task_keeps_the_old_reading(tmp_path):
    """A task that binds no request (written before the binding, or with no
    request on disk then): an unreadable request is no failure kind, and a
    salvaged ``[]`` persists as it always did."""
    phase = tmp_path / "phase"
    phase.mkdir()
    request = _base_salvage_request(phase)
    task = tasks_of(json.loads(
        (phase / f"salvage_task_{STEM}.json").read_text(encoding="utf-8")))[0]
    del task["inputs"]["salvage_request_sha256"]
    assert task_result.salvage_failure_of(task, str(phase)) == "timeout"
    for data in (None, b"{"):
        if data is None:
            request.unlink()
        else:
            request.write_bytes(data)
        failure = task_result.salvage_failure_of(task, str(phase))
        assert failure is None
        assert task_result._salvage_gave_up("salvage", [], failure=failure) is None


# ---------------------------------------------------------------------------
# 3. (cont.) A recovered round merges only the batches its dispatch wrote
# ---------------------------------------------------------------------------

#: The reason a surviving earlier run's batches carry.
LATE_VERDICT = "Late earlier verdict"


def _recovered(lane_class, tmp_path, monkeypatch, *, high: bool = False):
    """A healthy lane over a phase whose checkpoint is damaged, so the fresh
    run recovers it (``result_files.RECOVERED_KEY``) before its round."""
    lane = _healthy(lane_class, tmp_path, monkeypatch, high=high)
    lane.phase_dir.mkdir(parents=True, exist_ok=True)
    (lane.phase_dir / ".status.json").write_text("{broken")
    return lane


def _write_late_batches(phase_dir, document) -> None:
    for index, rows in _verdicts(document, "invalid", LATE_VERDICT).items():
        Path(phase_dir, f"validation_batch_{index}.json").write_text(json.dumps(rows))


def _gate_flags(lane_class) -> List[str]:
    return ["--exec-subagents"] + (["--fail-on-high"] if lane_class is CodeLane else [])


@BOTH_LANES
def test_a_recovered_round_dispatches_every_batch_again(tmp_path, monkeypatch, lane_class):
    """Through the real executor: a surviving earlier run's batch files land
    before the dispatch looks for completed batches. A recovered phase takes
    none of them as done: every batch is dispatched, and the verdicts this
    run's dispatch wrote are the ones merged and reported."""
    from tests.conftest import stand_in
    from utils import runner_selection, task_batch, task_dispatch
    from utils import result_files
    from utils import subagent_executor as executor

    lane = _recovered(lane_class, tmp_path, monkeypatch)
    dispatched: List[str] = []
    real_run_tasks = task_batch.run_tasks

    def late_writer_first(tasks, *, phase_dir, **kwargs):
        assert result_files.phase_recovered(phase_dir)
        _write_late_batches(phase_dir, {"batches": list(tasks)})
        return real_run_tasks(tasks, phase_dir=phase_dir, **kwargs)

    @stand_in
    def dispatch_task(task, *, phase_dir, runner):
        dispatched.append(task["task_id"])
        rows = _verdicts({"batches": [task]}, "valid", CHECKED_VERDICT)[task["batch_index"]]
        written = task_result.persist_result(task, phase_dir, rows)
        return executor.TaskOutcome(
            task_id=task["task_id"], kind=task["kind"], runner=runner,
            status=executor.STATUS_OK, output_path=written,
        )

    monkeypatch.setattr(task_batch, "run_tasks", late_writer_first)
    monkeypatch.setattr(task_dispatch, "dispatch_task", dispatch_task)
    monkeypatch.setattr(runner_selection, "_resolve_task_source",
                        lambda task, **kwargs: (SPEC, "env", "env"))
    monkeypatch.setattr(task_batch, "_preflight_batch_runners",
                        lambda resolved, *a, **k: ({SPEC: "1.0"}, None))
    assert _go(lane, "--exec-subagents") == 0
    tasks = lane.json("validation_tasks.json")
    assert sorted(dispatched) == sorted(t["task_id"] for t in tasks["batches"])
    assert {g["reason"] for g in lane.json("validation.json")["groups"]} == {CHECKED_VERDICT}
    report = _report(lane)
    assert CHECKED_COUNTS in report and LATE_VERDICT not in report


def _skipping_dispatch(lane):
    """A dispatch that finds the late batches complete and skips them (the
    executor's resume skip, run as it is), writing nothing itself."""
    import asyncio

    from utils import task_batch

    def dispatch(_args, *, document, phase_dir, **_kwargs):
        tasks = tasks_of(document)
        _write_late_batches(phase_dir, document)
        outcomes = asyncio.run(task_batch._gather(
            tasks, runners={t["task_id"]: SPEC for t in tasks}, phase_dir=phase_dir,
            anchor=str(lane.plan), cwd=str(lane.repo), max_parallel=1, resume=True,
        ))
        assert all(o.skipped and o.ok for o in outcomes.values())
        return True
    return dispatch


def _host_style_dispatch(lane):
    """A dispatch that writes the late batches itself, with no executor record."""
    def dispatch(_args, *, document, phase_dir, **_kwargs):
        _write_late_batches(phase_dir, document)
        return True
    return dispatch


@BOTH_LANES
@pytest.mark.parametrize("dispatch", [_skipping_dispatch, _host_style_dispatch],
                         ids=["resume-skip", "unrecorded-write"])
def test_a_recovered_round_never_merges_a_batch_its_dispatch_did_not_write(
        tmp_path, monkeypatch, capsys, lane_class, dispatch):
    """The dispatch reports success but holds no bytes it wrote itself for
    the batches: validation is not run, with the recovered phase's warning;
    the report and ``--fail-on-high`` count the groups the run holds (one
    HIGH group on review-code), and no late verdict reaches either."""
    from utils import task_dispatch

    lane = _recovered(lane_class, tmp_path, monkeypatch, high=(lane_class is CodeLane))
    monkeypatch.setattr(_module(lane), "dispatch_subagent_tasks", dispatch(lane))
    monkeypatch.setattr(task_dispatch, "dispatch_task",
                        lambda *a, **k: pytest.fail("unexpected provider dispatch"))
    capsys.readouterr()
    code = _go(lane, *_gate_flags(lane_class))
    captured = capsys.readouterr()
    assert LATE_VERDICT not in _report(lane)
    assert not (lane.phase_dir / "validation.json").exists()
    assert code == (3 if lane_class is CodeLane else 0)
    assert "WARNING: validation was not run:" in captured.err
    assert "a batch file it did not write is never merged there" in captured.err
    if lane_class is CodeLane:
        assert "High priority: 1 group(s)" in captured.out
