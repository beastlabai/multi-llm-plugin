"""``processed_<phase>`` carries one record per apply item on both legs
(fix round 22, live run 9 #3).

The decline fixture (an apply-code-fixes batch holding one fix to apply and
one the plan forbids) ran on all three hosts. On Claude Code's native leg
``util batch-done --declined`` recorded the forbidden fix in
``processed_apply-fixes`` as ``declined`` with its reason, and the two
applied fixes as ``applied``; on Codex and OpenCode, whose write executor
drove the same batches, ``state.json`` held no ``processed_apply-fixes`` at
all — the decline lived only in the batch's result file, the run report and
summary.md.

Both legs now write the same record through one writer
(``state_manager.record_batch_items_in_state``): ``applied`` / ``no-op``,
``declined`` / ``skipped`` with the reason (never ``applied``), ``failed``
for a batch that did not finish, each stamped with its batch, the review it
applied (``grouped_sha256``) and the leg. What the record means on a
``--resume`` (``processed_record_counts``) is pinned here too: the executor
leg's own records never renumber its batches, and a later review's run
offers again an item an earlier review's batch declined, on both legs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import apply_suggestions_orchestrator  # noqa: E402
from utils import batch_done  # noqa: E402
from utils.schema_validator import validate_state_file  # noqa: E402
from utils.state_manager import (  # noqa: E402
    PROCESSED_LEG_EXECUTOR,
    PROCESSED_LEG_NATIVE,
    StateManager,
    processed_record_counts,
)
from utils.write_phase_bridge import WRITE_TASK_FILENAME  # noqa: E402

from tests.test_write_executor import FakeState, env  # noqa: E402,F401
from tests.test_write_executor_orchestrators import repo  # noqa: E402,F401
from tests.test_write_batch_declined import REASON, _PairRig  # noqa: E402

#: Every record's details, on both legs; a declined or skipped one adds
#: its `reason`.
DETAIL_KEYS = {"task_id", "prior_changes_id", "grouped_sha256", "leg"}


def _processed(state_dict, phase="apply-suggestions"):
    return state_dict.get(f"processed_{phase}") or {}


# ===========================================================================
# The executor leg writes the record
# ===========================================================================

class TestTheSchedulerRecordsEveryItem:

    def _declining_batch_then_one(self, env, *, second_fails=False):
        def respond(task, prompt):
            if task["task_id"] == "apply-batch-0":
                env.plan.write_text(env.plan.read_text() + "s1\n")
                return {"status": "applied", "ids": ["s1"],
                        "declined": [{"id": "s0", "reason": REASON}],
                        "files_touched": ["plans/p.md"], "summary": "applied s1"}
            if second_fails:
                return {"status": "failed", "ids": [], "files_touched": [],
                        "summary": "could not apply s2"}
            env.plan.write_text(env.plan.read_text() + "s2\n")
            return {"status": "applied", "ids": ["s2"],
                    "files_touched": ["plans/p.md"], "summary": "applied s2"}
        return respond

    def _tasks(self, env):
        return [env.apply_task(0, expected_ids=["s0", "s1"]),
                env.apply_task(1, expected_ids=["s2"])]

    def test_a_declined_item_is_declined_with_its_reason_and_the_rest_applied(self, env):
        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler(
            self._tasks(env), respond=self._declining_batch_then_one(env),
            state=state).run()
        assert result.exit_code == 0, result.messages

        on_disk = json.loads((env.tmp_path / "state.json").read_text())
        processed = _processed(on_disk)
        assert {item: record["status"] for item, record in processed.items()} == {
            "s0": "declined", "s1": "applied", "s2": "applied"}
        declined = processed["s0"]["details"]
        assert declined["reason"] == REASON
        assert declined["task_id"] == "apply-batch-0"
        assert declined["prior_changes_id"] == "sec-0"
        assert declined["leg"] == PROCESSED_LEG_EXECUTOR
        assert set(declined) == DETAIL_KEYS | {"reason"}
        assert set(processed["s1"]["details"]) == DETAIL_KEYS

    def test_a_batch_that_did_not_finish_records_its_items_failed(self, env):
        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler(
            self._tasks(env),
            respond=self._declining_batch_then_one(env, second_fails=True),
            state=state).run()
        assert result.exit_code == 3
        processed = _processed(json.loads((env.tmp_path / "state.json").read_text()))
        assert processed["s2"]["status"] == "failed"
        assert processed["s0"]["status"] == "declined"

    def test_a_resume_that_keeps_the_batch_records_it_too(self, env):
        """A batch kept from its result file (an earlier run that wrote no
        record, say) is recorded by the resume that keeps it."""
        state = FakeState(env.tmp_path / "state.json")
        env.scheduler(
            self._tasks(env), respond=self._declining_batch_then_one(env),
            state=state).run()
        state.state.pop("processed_apply-suggestions")
        state.save()

        resumed = env.scheduler(
            self._tasks(env), respond=self._declining_batch_then_one(env),
            state=state, resume=True).run()
        assert resumed.exit_code == 0, resumed.messages
        assert all(outcome.skipped for outcome in resumed.outcomes)
        processed = _processed(json.loads((env.tmp_path / "state.json").read_text()))
        assert processed["s0"]["status"] == "declined"
        assert processed["s0"]["details"]["reason"] == REASON
        assert processed["s1"]["status"] == processed["s2"]["status"] == "applied"


class TestTheApplyPhaseRecordsEveryItem(_PairRig):

    def test_the_executor_writes_what_the_native_leg_writes(self, repo):
        plan = self._review_pair(repo)
        ids = self._ids(plan)
        declined_id, applied_id = ids[0]
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_DECLINED", json.dumps(
            [{"id": declined_id, "reason": REASON}]))
        assert self._run(repo, plan, [applied_id] + ids[1]) == 0

        state = json.loads((plan.parent / plan.stem / "state.json").read_text())
        processed = _processed(state)
        assert processed[declined_id]["status"] == "declined"
        assert processed[declined_id]["details"]["reason"] == REASON
        assert processed[applied_id]["status"] == "applied"
        assert processed[ids[1][0]]["status"] == "applied"
        tasks = json.loads(
            (self._phase_dir(plan) / WRITE_TASK_FILENAME).read_text())["batches"]
        details = processed[declined_id]["details"]
        assert details["task_id"] == tasks[0]["task_id"]
        assert details["prior_changes_id"] == tasks[0]["prior_changes_id"]
        assert details["grouped_sha256"] == tasks[0]["inputs"]["grouped_sha256"] != ""
        assert details["leg"] == PROCESSED_LEG_EXECUTOR
        assert validate_state_file(state)[0], validate_state_file(state)


# ===========================================================================
# The native leg writes the same shape
# ===========================================================================

@pytest.fixture
def native_repo(tmp_path, monkeypatch):
    """A repo holding a review-plan cycle, review-plan recorded completed."""
    import subprocess

    from tests.test_apply_suggestions_integration import TestFixtures

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    plan = root / "plans" / "test-plan.md"
    review = root / "plans" / "test-plan" / "review-plan"
    review.mkdir(parents=True)
    plan.write_text(TestFixtures.PLAN_CONTENT, encoding="utf-8")
    (review / "grouped.json").write_text(
        json.dumps(TestFixtures.GROUPED_SUGGESTIONS, indent=2), encoding="utf-8")
    (review / "validation.json").write_text(
        json.dumps(TestFixtures.VALIDATION_RESULTS, indent=2), encoding="utf-8")
    state = StateManager(plan)
    state.mark_phase_completed("review-plan")
    state.save()
    monkeypatch.chdir(root)
    return plan


def _emit(plan, capsys, *flags):
    """One native apply-suggestions run; its exit, stderr and the task list."""
    code = apply_suggestions_orchestrator.main([
        "--plan-file", str(plan), "--yes", "--output-format", "json", *flags,
    ])
    captured = capsys.readouterr()
    task_file = plan.parent / "test-plan" / "apply-suggestions" / WRITE_TASK_FILENAME
    tasks = (json.loads(task_file.read_text(encoding="utf-8"))["batches"]
             if task_file.exists() else [])
    return code, captured.err, tasks


def _record(plan, task_id, *flags):
    return batch_done.main([
        "--phase-dir", str(plan.parent / "test-plan" / "apply-suggestions"),
        "--task-id", task_id, "--status", "completed", *flags,
    ])


def _native_state(plan):
    return json.loads((plan.parent / "test-plan" / "state.json").read_text())


class TestTheNativeLegWritesTheSameShape:

    def test_batch_done_stamps_the_batch_the_review_and_the_leg(self, native_repo, capsys):
        plan = native_repo
        code, err, tasks = _emit(plan, capsys)
        assert code == 0, err
        first = tasks[0]
        declined = first["expected_ids"][0]
        assert _record(plan, first["task_id"], "--declined", f"{declined}={REASON}") == 0
        record = _processed(_native_state(plan))[declined]
        assert record["status"] == "declined"
        assert set(record["details"]) == DETAIL_KEYS | {"reason"}
        assert record["details"]["leg"] == PROCESSED_LEG_NATIVE
        assert record["details"]["grouped_sha256"] == first["inputs"]["grouped_sha256"] != ""


# ===========================================================================
# What a record means on a --resume
# ===========================================================================

def _rec(status="applied", **details):
    return {"status": status, "details": details, "timestamp": "2026-10-07T13:34:29"}


class TestWhatARecordMeansOnAResume:

    @pytest.mark.parametrize("record,stamp,executor_leg,counts", [
        (_rec(), "", False, True),
        (_rec("declined", reason="r"), "", False, True),
        (_rec("skipped", reason="r"), "", True, True),
        (_rec("failed"), "", False, False),
        # A record with no stamp (written before fix round 22) counts as before.
        (_rec(), "aaaa", False, True),
        # This review's record counts; an earlier review's decline or skip
        # does not, and its applied item still does (never handed out twice).
        (_rec(grouped_sha256="aaaa"), "aaaa", False, True),
        (_rec("declined", grouped_sha256="aaaa", reason="r"), "aaaa", False, True),
        (_rec("declined", grouped_sha256="aaaa", reason="r"), "bbbb", False, False),
        (_rec("skipped", grouped_sha256="aaaa", reason="r"), "bbbb", True, False),
        (_rec(grouped_sha256="aaaa"), "bbbb", False, True),
        # On the executor leg its own record is its result file's business;
        # a native host's record still keeps the item out.
        (_rec(leg=PROCESSED_LEG_EXECUTOR), "", True, False),
        (_rec(leg=PROCESSED_LEG_EXECUTOR), "", False, True),
        (_rec(leg=PROCESSED_LEG_NATIVE), "", True, True),
        ("not a record", "", False, False),
    ])
    def test_the_rule(self, record, stamp, executor_leg, counts):
        assert processed_record_counts(
            record, review_stamp=stamp, executor_leg=executor_leg) is counts


class TestTheExecutorsOwnRecordsDoNotRenumberItsBatches(_PairRig):

    def test_a_resume_dispatches_the_stopped_batch_under_its_own_id(self, repo, capsys):
        """Batch 0 applied, batch 1 a partial application: the resume keeps
        batch 0 from its result file and re-dispatches `apply-batch-1` —
        the id a stop's `--human-task apply-batch-1=skip` names — rather
        than handing what is left out again as `apply-batch-0`."""
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, ids[0]) == 3
        state = json.loads((plan.parent / plan.stem / "state.json").read_text())
        processed = _processed(state)
        assert processed[ids[0][0]]["status"] == "applied"
        assert processed[ids[1][0]]["status"] == "failed"
        capsys.readouterr()

        assert self._run(repo, plan, sum(ids, []), "--resume") == 0
        err = capsys.readouterr().err
        assert "already processed" not in err
        assert "the write executor keeps every batch an earlier run finished" in err
        rows = {row["task_id"]: row for row in self._report(plan)}
        assert rows["apply-batch-1"]["expected_ids"] == ids[1]
        assert rows["apply-batch-0"]["expected_ids"] == ids[0]


class TestALaterReviewOffersADeclinedItemAgain:

    def test_on_the_native_leg(self, native_repo, capsys):
        plan = native_repo
        code, err, first = _emit(plan, capsys)
        assert code == 0, err
        batch = first[0]
        declined = batch["expected_ids"][0]
        assert _record(plan, batch["task_id"], "--declined", f"{declined}={REASON}") == 0
        capsys.readouterr()

        # The same review: the declined item is done, never handed out again.
        code, err, same = _emit(plan, capsys, "--resume")
        assert code == 0, err
        assert declined not in {i for t in same for i in t["expected_ids"]}

        # A later review reports it again (another grouped.json): the resume
        # offers it again, as the executor leg does, and nothing records it
        # applied.
        grouped = plan.parent / "test-plan" / "review-plan" / "grouped.json"
        grouped.write_text(
            json.dumps(json.loads(grouped.read_text()), indent=4), encoding="utf-8")
        code, err, later = _emit(plan, capsys, "--resume")
        assert code == 0, err
        later_items = {i for t in later for i in t["expected_ids"]}
        assert declined in later_items
        assert _processed(_native_state(plan))[declined]["status"] == "declined"
        # What the batch applied is still never handed out twice.
        applied = set(batch["expected_ids"]) - {declined}
        assert applied and not applied & later_items


class TestALaterReviewOffersADeclinedItemAgainOnTheExecutorLeg(_PairRig):

    def test_its_batch_is_dispatched_again_and_it_is_recorded_declined(self, repo):
        """The executor leg's own behaviour, which the native leg now
        matches: a later review that reports the declined item again (the
        applied one fixed, so not reported) makes a batch the earlier run
        never ran, which is dispatched, and a second decline is recorded as
        one — never as applied."""
        from tests.test_write_batch_declined import _alpha_pair_groups
        from tests.test_write_executor_orchestrators import _validation
        from utils.state_manager import generate_group_id, get_or_create_state

        plan = self._review_pair(repo)
        ids = self._ids(plan)
        declined_id, applied_id = ids[0]
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_DECLINED", json.dumps(
            [{"id": declined_id, "reason": REASON}]))
        assert self._run(repo, plan, [applied_id] + ids[1]) == 0

        # The next review: the declined finding and the other section's.
        review = plan.parent / plan.stem / "review-plan"
        groups = [g for g in _alpha_pair_groups() if generate_group_id(g) != applied_id]
        assert len(groups) == 2
        (review / "grouped.json").write_text(json.dumps(groups), encoding="utf-8")
        (review / "validation.json").write_text(
            json.dumps(_validation(["valid"] * 2)), encoding="utf-8")
        state = get_or_create_state(plan)
        state.mark_phase_completed("review-plan")
        state.save()
        # The batch that declines everything edits nothing.
        repo.monkeypatch.setenv("MOCK_LLM_EDITS_IF_PROMPT_HAS", "Do B.")
        assert self._run(repo, plan, ids[1], "--resume") == 0

        rows = [row for row in self._report(plan) if declined_id in row["expected_ids"]
                and not row.get("from_earlier_run")]
        assert len(rows) == 1 and not rows[0].get("skipped"), rows
        assert rows[0]["expected_ids"] == [declined_id]
        state_json = json.loads((plan.parent / plan.stem / "state.json").read_text())
        record = _processed(state_json)[declined_id]
        assert record["status"] == "declined"
        assert record["details"]["task_id"] == rows[0]["task_id"]


def test_the_native_no_op_asymmetry_is_documented():
    """Fix round 23 (records #5): the native leg has no way to record a
    verified no-op (`util batch-done` takes no flag for it; the runbooks say
    a verified no-op is `--status completed`), so the same batch outcome is
    `applied` on Claude Code and `no-op` on Codex/OpenCode. No reader tells
    them apart (`processed_record_counts` treats both as applied), so it is
    documented rather than given a host flag."""
    schema = json.loads((SKILL_DIR / "schemas" / "state_file.schema.json").read_text())
    text = schema["patternProperties"]["^processed_"]["description"]
    assert "the native leg records a verified no-op batch as applied" in text, text
    assert "no reader tells the two apart" in text, text
    agents = (SKILL_DIR / "AGENTS.md").read_text(encoding="utf-8")
    assert "the native leg records a verified no-op batch as `applied`" in agents
