"""A write batch can decline an item with a reason (fix round 18).

Codex racing run 7 (2026-10-06), apply-code-fixes 08: write batch 1 held two
fixes. It applied fix 2 and left fix 1 alone on purpose — "Fix 1: Not
applied; T011 records the clamp removal as deliberate" — which it could say
only in ``notes``, because the result contract had ``ids`` for applied items
and nothing for an item declined with a reason. The runner recorded a partial
application ("only 1 of 2 expected id(s) came back"), the phase stopped
needs-human, and the resume re-dispatched the batch and stopped the same way,
offering only the same ``--resume``.

These tests pin the contract's ``declined`` list on the apply kinds, the
runner's reading of it (a declined item is neither applied nor missing; a
batch whose every item is applied or declined is complete), the contract
violations that stay partial applications, and a declined batch that a resume
does not dispatch again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from tests.test_write_executor import FakeState, _patch_edit, env  # noqa: E402,F401
from utils import subagent_executor as executor  # noqa: E402
from utils import subagent_prompts  # noqa: E402
from utils import subagent_tasks  # noqa: E402
from utils.schema_validator import validate_with_schema_file  # noqa: E402
from utils.write_batch_scheduler import with_result_contract  # noqa: E402

APPLY_KINDS = ("apply_suggestions", "apply_task_suggestions", "apply_code_fixes")
REASON = "the implementation record says T011 removed the clamp on purpose"


def _prompts(log: Path):
    if not log.exists():
        return []
    return [json.loads(line)["prompt"] for line in log.read_text().splitlines()]


# ===========================================================================
# The contract the batch is shown
# ===========================================================================

class TestTheContract:

    @pytest.mark.parametrize("kind", APPLY_KINDS)
    def test_an_apply_batch_is_asked_for_the_items_it_declines(self, kind):
        item = "fix" if kind == "apply_code_fixes" else "suggestion"
        text = with_result_contract(kind, {"expected_ids": ["aa", "bb"]}, "P")
        assert '"declined": [{"id": "<id>", "reason": "<why you did not apply it>"}]' in text
        assert f"- `declined`: each {item} from that list you deliberately left unapplied" in text
        assert "one the implementation record says was a deliberate decision" in text
        assert f"Every {item} goes in exactly one of `ids` and `declined`" in text
        assert "the run stops for a human to inspect it" in text
        # `no-op` covers a batch that declined everything and edited nothing.
        assert "because every change was already present or you declined it" in text
        # The old "leave out only the ones you could not apply" invited a
        # silent omission; it is gone.
        assert "Leave out only the ones you could not apply" not in text

    def test_implement_is_not_asked_for_declined_items(self):
        """A task its runner will not do is `failed`, for the user to answer
        with ``--human-task <id>=skip|done`` — never a silent decline."""
        text = with_result_contract(
            "implement", {"expected_ids": ["T001"], "declared_files": ["x"]}, "P",
        )
        assert "declined" not in text

    def test_the_schema_reads_declined_as_a_list_of_objects(self):
        base = {"status": "applied", "summary": "s"}
        ok = dict(base, declined=[{"id": "a", "reason": "r"}])
        assert validate_with_schema_file(ok, "write_batch_result.schema.json") == []
        # An entry without a reason is a named contract problem for the
        # classification, never a schema rejection of the whole reply.
        bare = dict(base, declined=[{"id": "a"}])
        assert validate_with_schema_file(bare, "write_batch_result.schema.json") == []
        wrong = dict(base, declined="fix 1 was not applied")
        assert validate_with_schema_file(wrong, "write_batch_result.schema.json")

    @pytest.mark.parametrize("kind", APPLY_KINDS)
    def test_the_report_re_ask_restates_declined_items(self, kind):
        prompt = subagent_prompts.build_schema_repair_prompt(
            kind, "Q", ["x"], previous_reply="Fix 1: not applied. Fix 2: done.",
            expected_ids=["aa", "bb"],
        )
        assert '"declined": [{"id": "<id>", "reason": "<why it was not applied>"}]' in prompt
        assert "goes in `declined`, with the\nreason it gave" in prompt
        assert "`ids` must use these spellings, and so must `declined`" in prompt
        example = subagent_prompts.SCHEMA_REPAIR_EXAMPLES[kind]
        assert example["declined"] == []

    def test_implements_re_ask_names_no_declined(self):
        prompt = subagent_prompts.build_schema_repair_prompt(
            "implement", "Q", ["x"], previous_reply="done", expected_ids=["T001"],
        )
        assert "declined" not in prompt

    def test_the_two_lists_of_apply_kinds_agree(self):
        assert subagent_prompts.APPLY_REPORT_KINDS == subagent_tasks.APPLY_WRITE_KINDS
        assert set(APPLY_KINDS) == set(subagent_tasks.APPLY_WRITE_KINDS)


class TestTheReader:

    def test_a_decline_needs_an_expected_id_and_a_reason(self):
        read = executor.declined_items(
            {
                "ids": ["b"],
                "declined": [
                    {"id": "a", "reason": "  deliberate \n per T011 "},
                    {"id": "b", "reason": "also applied?"},
                    {"id": "z", "reason": "not this batch's"},
                    {"id": "c"},
                    {"reason": "no id"},
                    "d",
                ],
            },
            ["a", "b", "c", "d"],
        )
        assert read.items == {"a": "deliberate per T011"}
        assert read.conflicting == ["b"]
        assert read.problems == [
            "b is in both `ids` and `declined`",
            "`declined` names 'z', which this batch does not hold",
            "`declined` gives no reason for c",
            "`declined` entry 5 names no id",
            "`declined` entry 6 is not an object",
        ]

    def test_no_declined_key_reads_as_nothing_declined(self):
        read = executor.declined_items({"ids": ["a"]}, ["a"])
        assert (read.items, read.conflicting, read.problems) == ({}, [], [])


# ===========================================================================
# The runner's reading: a declined item is neither applied nor missing
# ===========================================================================

def _two_item_task(env, index=0, ids=("s0", "s1"), **over):
    return env.apply_task(index, expected_ids=list(ids), **over)


def _respond_with(env, reply, *, edit=True):
    seen = []

    def respond(task, prompt):
        seen.append(prompt)
        if edit:
            env.plan.write_text(env.plan.read_text() + f"edit {len(seen)}\n")
        return dict(reply)

    respond.seen = seen  # type: ignore[attr-defined]
    return respond


class TestABatchThatDeclinesAnItem:

    def test_one_applied_one_declined_is_complete_and_the_next_batch_runs(self, env):
        """The Codex 08 batch, answered in the new contract."""
        calls = []

        def respond(task, prompt):
            calls.append(task["task_id"])
            env.plan.write_text(env.plan.read_text() + f"{task['task_id']}\n")
            if task["task_id"] == "apply-batch-0":
                return {"status": "applied", "ids": ["s1"],
                        "declined": [{"id": "s0", "reason": REASON}],
                        "files_touched": ["plans/p.md"], "summary": "applied s1"}
            return {"status": "applied", "ids": ["s2"],
                    "files_touched": ["plans/p.md"], "summary": "applied s2"}

        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler(
            [_two_item_task(env), env.apply_task(1, expected_ids=["s2"])],
            respond=respond, state=state,
        ).run()

        assert result.exit_code == 0, result.messages
        assert calls == ["apply-batch-0", "apply-batch-1"]
        first = result.outcomes[0]
        assert first.ok and first.write_outcome == executor.WRITE_APPLIED
        assert first.declined == {"s0": REASON}
        assert first.unapplied_ids == []
        assert first.reason.startswith("APPLIED 1 of 2 item(s); the batch declined the rest")
        assert f"suggestion 1 (`s0`) — {REASON}" in first.reason
        row = first.to_dict()
        assert row["declined"] == [{"id": "s0", "reason": REASON}]
        assert state.get_task_status("apply-batch-0") == "completed"
        # The verdict stamp carries it, for a resume to keep.
        stamped = json.loads((env.phase_dir / "apply_batch_0.json").read_text())
        assert stamped["verdict"]["declined"] == [{"id": "s0", "reason": REASON}]
        assert stamped["verdict"]["ok"] is True

    @pytest.mark.parametrize("reply,problem", [
        ({"ids": ["s1", "s0"], "declined": [{"id": "s0", "reason": REASON}]},
         "s0 is in both `ids` and `declined`"),
        ({"ids": ["s1"], "declined": [{"id": "s9", "reason": REASON}]},
         "`declined` names 's9', which this batch does not hold"),
        ({"ids": ["s1"], "declined": [{"id": "s0"}]},
         "`declined` gives no reason for s0"),
        ({"ids": ["s1"], "declined": [{"id": "s0", "reason": "   "}]},
         "`declined` gives no reason for s0"),
    ])
    def test_a_contract_violation_is_a_partial_application(self, env, reply, problem):
        respond = _respond_with(env, dict(
            reply, status="applied", files_touched=["plans/p.md"], summary="s",
        ))
        result = env.scheduler([_two_item_task(env)], respond=respond).run()

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_PARTIAL, outcome.reason
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert "expected id(s) came back (missing: s0)" in outcome.reason
        assert problem in outcome.reason
        assert outcome.declined == {}
        assert result.exit_code == 3

    def test_an_item_left_out_of_both_lists_is_a_partial_application(self, env):
        """The Codex 08 shape itself: fix 2 in `ids`, fix 1 declined in
        `notes` only. Still a partial application — the notes are never read
        as a decline — with the unaccounted item named for the way forward."""
        respond = _respond_with(env, {
            "status": "applied", "ids": ["s1"], "files_touched": ["plans/p.md"],
            "summary": "applied s1",
            "notes": "Suggestion 1: Not applied; T011 records it as deliberate.",
        })
        result = env.scheduler([_two_item_task(env)], respond=respond).run()

        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert "only 1 of 2 expected id(s) came back (missing: s0)" in outcome.reason
        assert outcome.unapplied_ids == ["s0"]
        assert result.exit_code == 3

    def test_every_item_declined_and_nothing_changed_is_complete(self, env):
        """Whatever word it put in `status`: a batch that declined every item
        and edited nothing is not sent again (rule 9) and does not stop."""
        for status in ("no-op", "applied"):
            respond = _respond_with(env, {
                "status": status, "ids": [],
                "declined": [{"id": "s0", "reason": REASON},
                             {"id": "s1", "reason": "needs a plan decision"}],
                "files_touched": [], "summary": "declined both",
            }, edit=False)
            result = env.scheduler(
                [_two_item_task(env)], respond=respond, resume=False,
            ).run()
            outcome = result.outcomes[0]
            assert outcome.ok, (status, outcome.reason)
            assert outcome.write_outcome == executor.WRITE_NO_OP
            assert outcome.reason.startswith("NO-OP: the batch declined every item it held")
            assert len(respond.seen) == 1, "a batch that declined everything is not re-sent"
            assert result.exit_code == 0

    def test_every_item_declined_while_the_file_changed_is_a_partial_application(self, env):
        respond = _respond_with(env, {
            "status": "applied", "ids": [],
            "declined": [{"id": "s0", "reason": REASON},
                         {"id": "s1", "reason": "needs a plan decision"}],
            "files_touched": ["plans/p.md"], "summary": "declined both",
        })
        result = env.scheduler([_two_item_task(env)], respond=respond).run()
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_PARTIAL
        assert "the batch declined every item it held" in outcome.reason
        assert result.exit_code == 3

    def test_an_implement_task_that_declines_itself_failed(self, env):
        """`implement` has no decline: its batch is a task of the plan. A
        reply declining it did not do it — a failure the user answers."""
        task = env.apply_task(0, kind="implement", task_id="T001",
                              expected_ids=["T001"],
                              output_name="implement_batch_0.json")
        for status in ("no-op", "applied"):
            respond = _respond_with(env, {
                "status": status, "ids": [],
                "declined": [{"id": "T001", "reason": "the plan contradicts itself"}],
                "files_touched": [], "summary": "not done",
            }, edit=False)
            result = env.scheduler(
                [task], kind="implement", respond=respond, finalize=False,
                resume=False,
            ).run()
            outcome = result.outcomes[0]
            assert outcome.status == executor.STATUS_BATCH_FAILED, outcome.reason
            assert outcome.write_outcome == executor.WRITE_CLEAN_FAILURE
            assert "the runner declined the task (the plan contradicts itself)" in outcome.reason
            assert outcome.declined == {}
            assert len(respond.seen) == 1
            assert result.exit_code == 3


class TestAResumeKeepsADeclinedBatch:

    def test_a_declined_batch_is_not_dispatched_again(self, env):
        calls = []

        def first(task, prompt):
            calls.append(task["task_id"])
            if task["task_id"] == "apply-batch-0":
                env.plan.write_text(env.plan.read_text() + "s1\n")
                return {"status": "applied", "ids": ["s1"],
                        "declined": [{"id": "s0", "reason": REASON}],
                        "files_touched": ["plans/p.md"], "summary": "applied s1"}
            return {"status": "failed", "ids": [], "files_touched": [],
                    "summary": "could not apply s2"}

        tasks = [_two_item_task(env), env.apply_task(1, expected_ids=["s2"])]
        state = FakeState(env.tmp_path / "state.json")
        stopped = env.scheduler(tasks, respond=first, state=state).run()
        assert stopped.exit_code == 3
        assert calls == ["apply-batch-0", "apply-batch-1"]

        def second(task, prompt):
            calls.append(task["task_id"])
            env.plan.write_text(env.plan.read_text() + "s2\n")
            return {"status": "applied", "ids": ["s2"],
                    "files_touched": ["plans/p.md"], "summary": "applied s2"}

        resumed = env.scheduler(
            [_two_item_task(env), env.apply_task(1, expected_ids=["s2"])],
            respond=second, state=state, resume=True,
        ).run()
        assert resumed.exit_code == 0, resumed.messages
        assert calls == ["apply-batch-0", "apply-batch-1", "apply-batch-1"]
        kept = resumed.outcomes[0]
        assert kept.skipped and kept.ok
        assert kept.declined == {"s0": REASON}
        rows = {row["task_id"]: row for row in resumed.report_rows()}
        assert rows["apply-batch-0"]["declined"] == [{"id": "s0", "reason": REASON}]


# ===========================================================================
# Every surface names a declined item and its reason (the real orchestrator)
# ===========================================================================

from tests.test_write_executor_orchestrators import (  # noqa: E402
    APPLY_PLAN, _ApplyRig, _validation, repo,  # noqa: F401
)


def _alpha_pair_groups():
    """Two suggestions on one section (one batch) and one on another."""
    def group(n, section, step):
        return {"theme": f"theme {n}", "category": "c", "models": ["m"],
                "suggestions": [{
                    "title": f"suggestion {n}", "desc": f"Do {n}.",
                    "type": "addition", "reference": f"## Step {step}: {section}",
                    "importance": "MEDIUM", "source_model": "m",
                }]}
    return [group("A1", "Alpha", 1), group("A2", "Alpha", 1), group("B", "Beta", 2)]


class TestADeclinedItemIsNamedOnEverySurface(_ApplyRig):

    def _review_pair(self, repo):
        from utils.state_manager import get_or_create_state

        plan = repo.root / "plans" / "apply.md"
        plan.write_text(APPLY_PLAN, encoding="utf-8")
        review = repo.root / "plans" / "apply" / "review-plan"
        review.mkdir(parents=True, exist_ok=True)
        (review / "grouped.json").write_text(
            json.dumps(_alpha_pair_groups()), encoding="utf-8")
        (review / "validation.json").write_text(
            json.dumps(_validation(["valid"] * 3)), encoding="utf-8")
        state = get_or_create_state(plan)
        state.mark_phase_completed("review-plan")
        state.save()
        return plan

    def test_summary_md_the_notice_the_mirror_and_status_name_it(self, repo, capsys):
        import multi_llm
        from utils import cli_common

        plan = self._review_pair(repo)
        ids = self._ids(plan)
        assert [len(batch) for batch in ids] == [2, 1], ids
        declined_id, applied_id = ids[0]
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_DECLINED", json.dumps(
            [{"id": declined_id, "reason": REASON}]))
        capsys.readouterr()
        context = cli_common.activate_result(cli_common.ResultContext(
            mode="apply-suggestions", phase="apply-suggestions"))
        try:
            assert self._run(repo, plan, [applied_id] + ids[1]) == 0
            notice = context.notice or ""
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out

        # summary.md: never counted as applied, named with its reason.
        summary = self._summary(plan)
        assert "- Suggestions applied: 2" in summary
        assert "- Suggestions declined by their batch (not applied): 1" in summary
        titles = {"A1": "suggestion A1", "A2": "suggestion A2"}
        lines = [line for line in summary.splitlines() if "**suggestion A" in line]
        declined_title = next(
            title for title in titles.values()
            if any(f"**{title}** - Not applied: the batch declined it — {REASON}"
                   in line for line in lines)
        )
        assert any(line.endswith("- Applied") for line in lines), lines
        # The RESULT block and the result line's notice say it in plain words.
        expected = (
            f"{declined_title} [{declined_id[:8]}] (suggestion 1 of apply-batch-0) "
            f"not applied: the batch declined it — {REASON}"
        )
        assert "Suggestions declined by their batch (not applied): 1" in out
        assert f"  - {expected}" in out
        assert expected in notice
        # The mirror `status` reads, and `status` itself.
        mirror = json.loads(
            (self._phase_dir(plan) / ".status.json").read_text())["progress"]
        assert mirror["write_subagents"]["declined_items"] == [
            {"task_id": "apply-batch-0", "id": declined_id, "reason": REASON}]
        assert mirror["write_subagents"]["items_applied"] == 2
        assert expected in mirror["notes"]
        capsys.readouterr()
        multi_llm.main(["status", str(plan), "--view", "run", "--phase",
                        "apply-suggestions"])
        view = capsys.readouterr().out
        assert (f"not applied, declined by its batch: {declined_id[:8]} in "
                f"apply-batch-0 ({REASON})") in view
        assert f"note: {expected}" in view
