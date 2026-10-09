"""`status`'s native apply count is the phase run's, across a `--resume`,
and a bare re-run counts only its own records (fix round 29, review of
round 28, records #1 and #2).

Fix round 28 put "2/3 items applied; 1 declined by its batch …" on the
native apply view, counting the batches of the current `write_tasks.json`.
A native `--resume` re-materializes that file with only the batches still
to run, so the count dropped the items an earlier part of the run applied:
right after the resume's emit the block read "1 declined by its batch (not
applied): …" with no count before it (live run 12 #4 again), and once the
resumed batch was recorded "1/1 items applied; 1 declined …" for a phase
that applied 3 of its 4 items. A bare re-run of a completed native phase
(no `--resume`, no `--fresh`) keeps the earlier run's records, so before
the new run recorded anything the block read the earlier run's "4/4 items
applied" under the new run's "not finished" heading.

The native emit now records the phase run its batches belong to in
`write_tasks.json` (`native_run`: when it started, every item it handed
out, when this file was written); a `--resume` of the same review carries
it, anything else starts a new one. `status` counts that run's items and
the records written since it started.

Records #2: a batch recorded failed and nothing else read "0/4 items
applied" and named nothing; the block now names the failed batch and its
recorded reason after the count, as the executor's line names it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import apply_suggestions_orchestrator  # noqa: E402
from utils import batch_done, prior_changes, run_status  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.write_phase_bridge import WRITE_TASK_FILENAME  # noqa: E402

from tests.test_apply_suggestions_integration import TestFixtures  # noqa: E402

TYPED = "plans/test-plan.md"
FORBIDS = "the plan forbids it"
DECLINED = f"1 declined by its batch (not applied): 797ad707 in apply-batch-1 ({FORBIDS})"


@pytest.fixture
def plan(tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    path = root / "plans" / "test-plan.md"
    review = root / "plans" / "test-plan" / "review-plan"
    review.mkdir(parents=True)
    path.write_text(TestFixtures.PLAN_CONTENT, encoding="utf-8")
    (review / "grouped.json").write_text(
        json.dumps(TestFixtures.GROUPED_SUGGESTIONS, indent=2), encoding="utf-8")
    (review / "validation.json").write_text(
        json.dumps(TestFixtures.VALIDATION_RESULTS, indent=2), encoding="utf-8")
    (review / "backup.md").write_text(TestFixtures.PLAN_CONTENT, encoding="utf-8")
    state = StateManager(path)
    state.mark_phase_completed("review-plan")
    state.save()
    monkeypatch.chdir(root)
    return path


def _phase_dir(plan: Path) -> Path:
    return plan.parent / "test-plan" / "apply-suggestions"


def _emit(plan, capsys, *flags):
    """The native emit (no `--exec-subagents`); its batches."""
    code = apply_suggestions_orchestrator.main([
        "--plan-file", str(plan), "--yes", "--output-format", "json", *flags,
    ])
    err = capsys.readouterr().err
    assert code == 0, err
    document = json.loads((_phase_dir(plan) / WRITE_TASK_FILENAME).read_text(encoding="utf-8"))
    return document["batches"]


def _record(plan, capsys, task_id, *flags, status="completed"):
    assert batch_done.main([
        "--phase-dir", str(_phase_dir(plan)), "--task-id", task_id,
        "--status", status, *flags,
    ]) == 0
    capsys.readouterr()


def _finish(plan):
    state = StateManager(plan)
    state.mark_phase_completed("apply-suggestions")
    state.save()


def _status(capsys, *flags) -> str:
    capsys.readouterr()
    run_status.main(["--plan-file", TYPED, "--host", "claude-code", *flags])
    return capsys.readouterr().out


def _block(out: str) -> list:
    """The native apply block's lines under its head."""
    lines = out.splitlines()
    head = next(i for i, line in enumerate(lines)
                if line.startswith("  apply-suggestions  ") and " on the host " in line)
    block = []
    for line in lines[head + 1:]:
        if not line.startswith("      "):
            break
        block.append(line.strip())
    return block


def _first_run_then_resume(plan, capsys):
    """4 items in 3 batches: batch 0 (2 items) applied, batch 1 declined its
    one item, batch 2 never recorded; then `--resume`."""
    first = _emit(plan, capsys)
    assert [len(t["expected_ids"]) for t in first] == [2, 1, 1]
    assert first[1]["expected_ids"] == ["797ad7073ba1bef9"]
    _record(plan, capsys, "apply-batch-0")
    _record(plan, capsys, "apply-batch-1", "--declined", f"797ad7073ba1bef9={FORBIDS}")
    for task in first[:2]:
        prior_changes.append(str(plan.parent / "test-plan"), task["prior_changes_id"],
                             "apply-suggestions", "batch applied")
    before = _block(_status(capsys, "--view", "run"))
    second = _emit(plan, capsys, "--resume")
    assert [t["expected_ids"] for t in second] == [first[2]["expected_ids"]]
    return before, second


class TestTheCountIsThePhaseRuns:

    def test_before_the_resume(self, plan, capsys):
        before, _second = _first_run_then_resume(plan, capsys)
        assert before[0] == f"2/4 items applied to {TYPED}; {DECLINED}", before

    def test_right_after_the_resume_emit_the_count_stays(self, plan, capsys):
        _first_run_then_resume(plan, capsys)
        for flags in ((), ("--view", "run")):
            block = _block(_status(capsys, *flags))
            assert block[0] == f"2/4 items applied to {TYPED}; {DECLINED}", block

    def test_once_the_resumed_batch_is_recorded(self, plan, capsys):
        _before, second = _first_run_then_resume(plan, capsys)
        _record(plan, capsys, second[0]["task_id"])
        _finish(plan)
        for flags in ((), ("--view", "run")):
            block = _block(_status(capsys, *flags))
            assert block[0] == f"3/4 items applied to {TYPED}; {DECLINED}", block
        entry = json.loads(_status(capsys, "--view", "run", "--json"))["apply_phases"][0]
        assert (entry["items_applied"], entry["items_total"]) == (3, 4), entry
        assert [row["id"] for row in entry["declined_items"]] == ["797ad7073ba1bef9"]

    def test_a_resume_carries_the_run_a_bare_rerun_starts_one(self, plan, capsys):
        first = _emit(plan, capsys)
        run = json.loads((_phase_dir(plan) / WRITE_TASK_FILENAME).read_text(
            encoding="utf-8"))["native_run"]
        assert run["items"] == [i for t in first for i in t["expected_ids"]]
        assert run["started_at"] == run["emitted_at"]
        _record(plan, capsys, "apply-batch-0")
        _emit(plan, capsys, "--resume")
        resumed = json.loads((_phase_dir(plan) / WRITE_TASK_FILENAME).read_text(
            encoding="utf-8"))["native_run"]
        assert resumed["started_at"] == run["started_at"]
        assert resumed["items"] == run["items"]
        assert resumed["emitted_at"] > run["emitted_at"]
        _emit(plan, capsys)
        again = json.loads((_phase_dir(plan) / WRITE_TASK_FILENAME).read_text(
            encoding="utf-8"))["native_run"]
        assert again["started_at"] > run["started_at"]
        assert again["started_at"] == again["emitted_at"]


class TestABareRerunCountsOnlyItsOwnRecords:

    def _completed_then_rerun(self, plan, capsys):
        first = _emit(plan, capsys)
        for task in first:
            _record(plan, capsys, task["task_id"])
        _finish(plan)
        done = _block(_status(capsys, "--view", "run"))
        assert done[0] == f"4/4 items applied to {TYPED}", done
        return _emit(plan, capsys)

    def test_nothing_recorded_yet_counts_nothing(self, plan, capsys):
        self._completed_then_rerun(plan, capsys)
        for flags in ((), ("--view", "run")):
            out = _status(capsys, *flags)
            assert "not finished on the host" in out, out
            assert "items applied" not in out, out
        entry = json.loads(_status(capsys, "--view", "run", "--json"))["apply_phases"][0]
        assert "items_applied" not in entry and "items_total" not in entry, entry

    def test_its_own_record_counts_from_zero(self, plan, capsys):
        second = self._completed_then_rerun(plan, capsys)
        _record(plan, capsys, second[0]["task_id"])
        block = _block(_status(capsys, "--view", "run"))
        assert block[0] == f"2/4 items applied to {TYPED}", block

    def test_the_earlier_runs_decline_is_not_this_runs(self, plan, capsys):
        first = _emit(plan, capsys)
        _record(plan, capsys, "apply-batch-0")
        _record(plan, capsys, "apply-batch-1", "--declined", f"797ad7073ba1bef9={FORBIDS}")
        _record(plan, capsys, first[2]["task_id"])
        _finish(plan)
        _emit(plan, capsys)
        out = _status(capsys, "--view", "run")
        assert "declined by its batch" not in out, out


class TestAFailedBatchIsNamed:

    def test_a_failed_record_alone_names_its_batch_and_reason(self, plan, capsys):
        _emit(plan, capsys)
        _record(plan, capsys, "apply-batch-0", "--reason", "timeout, no file changed.",
                status="failed")
        expected = (f"0/4 items applied to {TYPED}; "
                    "1 batch failed: apply-batch-0 (timeout, no file changed)")
        for flags in ((), ("--view", "run")):
            block = _block(_status(capsys, *flags))
            assert block[0] == expected, block
        entry = json.loads(_status(capsys, "--view", "run", "--json"))["apply_phases"][0]
        assert entry["failed_batches"] == [
            {"task_id": "apply-batch-0", "reason": "timeout, no file changed."}]

    def test_beside_what_applied(self, plan, capsys):
        _emit(plan, capsys)
        _record(plan, capsys, "apply-batch-0")
        _record(plan, capsys, "apply-batch-1", "--reason", "the subagent timed out",
                status="failed")
        _record(plan, capsys, "apply-batch-2", "--reason", "no reply", status="failed")
        block = _block(_status(capsys, "--view", "run"))
        assert block[0] == (
            f"2/4 items applied to {TYPED}; 2 batches failed: "
            "apply-batch-1 (the subagent timed out); apply-batch-2 (no reply)"), block

    def test_a_resume_hands_it_out_again_so_it_is_no_longer_named(self, plan, capsys):
        first = _emit(plan, capsys)
        _record(plan, capsys, "apply-batch-0")
        _record(plan, capsys, "apply-batch-1", "--reason", "the subagent timed out",
                status="failed")
        prior_changes.append(str(plan.parent / "test-plan"), first[0]["prior_changes_id"],
                             "apply-suggestions", "batch applied")
        second = _emit(plan, capsys, "--resume")
        assert first[1]["expected_ids"][0] in [i for t in second for i in t["expected_ids"]]
        block = _block(_status(capsys, "--view", "run"))
        assert block[0] == f"2/4 items applied to {TYPED}", block
        assert "failed" not in " ".join(block), block


class TestWhatAResumeCarries:
    """`native_run_record` on a `--resume`: only the native run of the same
    review is carried; anything else leaves the start unknown, so every
    record of the review counts, as before this record existed."""

    def _tasks(self, tmp_path, review="cycle 1"):
        from utils.write_phase_bridge import materialize_write_tasks

        plan = tmp_path / "p.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        grouped = tmp_path / f"grouped-{review.replace(' ', '-')}.json"
        grouped.write_text(json.dumps([{"theme": review}]), encoding="utf-8")
        tasks = materialize_write_tasks(
            kind="apply_suggestions", phase_dir=str(tmp_path / "apply-suggestions"),
            plan_file=str(plan), grouped_file=str(grouped),
            batches=[{"section_key": "s1", "prompt": "Apply.",
                      "suggestions": [{"group_id": "g1"}]}])
        return tasks, plan, grouped

    def _write(self, tmp_path, *, native=True):
        from utils.write_phase_bridge import native_run_record, write_task_file_for_phase

        tasks, plan, grouped = self._tasks(tmp_path)
        phase_dir = tmp_path / "apply-suggestions"
        write_task_file_for_phase(
            kind="apply_suggestions", tasks=tasks, phase_dir=str(phase_dir),
            plan_file=str(plan), grouped_file=str(grouped),
            native_host="claude-code" if native else None,
            native_run=native_run_record(
                str(phase_dir), tasks, resume=False, now="2026-10-09T01:00:00.000001"))
        return phase_dir, tasks

    def test_the_same_reviews_native_run_is_carried(self, tmp_path):
        from utils.write_phase_bridge import native_run_record

        phase_dir, tasks = self._write(tmp_path)
        assert native_run_record(str(phase_dir), tasks, resume=True, now="T") == {
            "started_at": "2026-10-09T01:00:00.000001", "items": ["g1"], "emitted_at": "T"}

    def test_another_reviews_run_is_not(self, tmp_path):
        from utils.write_phase_bridge import native_run_record

        phase_dir, _cycle1 = self._write(tmp_path)
        cycle2, _plan, _grouped = self._tasks(tmp_path, review="cycle 2")
        assert native_run_record(str(phase_dir), cycle2, resume=True, now="T") == {
            "started_at": None, "items": ["g1"], "emitted_at": "T"}

    def test_an_executor_runs_file_is_not(self, tmp_path):
        from utils.write_phase_bridge import native_run_record

        phase_dir, tasks = self._write(tmp_path, native=False)
        assert "native_run" not in json.loads(
            (phase_dir / WRITE_TASK_FILENAME).read_text(encoding="utf-8"))
        assert native_run_record(str(phase_dir), tasks, resume=True, now="T") == {
            "started_at": None, "items": ["g1"], "emitted_at": "T"}

    def test_a_bare_run_starts_one(self, tmp_path):
        from utils.write_phase_bridge import native_run_record

        phase_dir, tasks = self._write(tmp_path)
        assert native_run_record(str(phase_dir), tasks, resume=False, now="T") == {
            "started_at": "T", "items": ["g1"], "emitted_at": "T"}
