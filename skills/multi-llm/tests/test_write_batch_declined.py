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


class _PairRig(_ApplyRig):
    """The apply-suggestions orchestrator over a review whose first batch
    holds two items."""

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


class TestADeclinedItemIsNamedOnEverySurface(_PairRig):

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


# ===========================================================================
# The way forward from a partial application a resume would only repeat
# ===========================================================================

CODEX_08_REPLY = {
    # Codex racing run 7, apply-code-fixes 08: fix 2 applied and attested,
    # fix 1 left alone on purpose and said so only in `notes`.
    "status": "applied", "ids": ["s1"], "files_touched": ["plans/p.md"],
    "summary": "applied suggestion 2; kept the deliberate T011 decision",
    "notes": "Suggestion 1: Not applied; T011 records it as deliberate. "
             "Suggestion 2: applied.",
}
SKIP_FLAGS = ["--resume", "--human-task", "apply-batch-0=skip"]
SKIP_WHY = (
    "keep what apply-batch-0 applied and go on without suggestion 1 (`s0`): "
    "it is recorded as not applied, skipped at your request, and the batches "
    "after it run"
)


def _codex_08_respond(env, calls):
    def respond(task, prompt):
        calls.append(task["task_id"])
        env.plan.write_text(env.plan.read_text() + f"{task['task_id']}\n")
        if task["task_id"] == "apply-batch-0":
            return dict(CODEX_08_REPLY)
        return {"status": "applied", "ids": ["s2"],
                "files_touched": ["plans/p.md"], "summary": "applied s2"}
    return respond


def _codex_08_tasks(env):
    return [_two_item_task(env), env.apply_task(1, expected_ids=["s2"])]


class TestThePartialApplicationOffersAWayForward:

    def test_the_stop_offers_keeping_the_batch_and_skipping_what_it_left_out(self, env):
        calls = []
        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls), state=state,
        ).run()

        assert result.exit_code == 3
        assert calls == ["apply-batch-0"]
        assert result.stop["unapplied_ids"] == ["s0"]
        assert result.stop["answers"] == [{"flags": SKIP_FLAGS, "why": SKIP_WHY}]
        assert ("Or keep what it applied and go on without suggestion 1 (`s0`): "
                "add --human-task apply-batch-0=skip to the resume command"
                ) in result.stop["headline"]
        text = "\n".join(result.messages)
        assert (f"  - or {SKIP_WHY}: add --human-task apply-batch-0=skip to the "
                f"resume command") in text
        # The mirror carries the answer, which `util next-command --blocked`
        # hands the tool routes as an option (`stop_answers_on_record`).
        mirror = json.loads((env.phase_dir / ".status.json").read_text())
        assert mirror["progress"]["write_subagents"]["stop"]["answers"] == [
            {"flags": SKIP_FLAGS, "why": SKIP_WHY}]

    def test_the_answer_keeps_the_batch_and_the_phase_finishes(self, env):
        calls = []
        state = FakeState(env.tmp_path / "state.json")
        env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls), state=state,
        ).run()
        lines = []

        resumed = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls),
            state=state, resume=True, partial_skips=["apply-batch-0"],
            log=lines.append,
        ).run()

        assert resumed.exit_code == 0, resumed.messages
        # Kept, never dispatched again; the batch after it ran.
        assert calls == ["apply-batch-0", "apply-batch-1"]
        kept = resumed.outcomes[0]
        assert kept.ok and kept.write_outcome == executor.WRITE_APPLIED
        assert not kept.skipped
        assert list(kept.skipped_items) == ["s0"]
        assert kept.reason.startswith(
            "KEPT AT YOUR REQUEST (--human-task apply-batch-0=skip)")
        assert "not applied, skipped at your request: suggestion 1 (`s0`)" in kept.reason
        assert any(
            "apply-batch-0 kept as it stood at your request, not dispatched again"
            in line for line in lines
        ), lines
        assert state.get_task_status("apply-batch-0") == "completed"
        assert state.get_task_status_reason("apply-batch-0").startswith(
            "kept at your request (--human-task apply-batch-0=skip)")
        # Its summary is in the prior-changes log the next batch read.
        log = (env.out_dir / "prior_changes.jsonl").read_text()
        assert CODEX_08_REPLY["summary"] in log
        stamped = json.loads((env.phase_dir / "apply_batch_0.json").read_text())
        assert stamped["verdict"]["ok"] is True
        assert [e["id"] for e in stamped["verdict"]["skipped_items"]] == ["s0"]
        rows = {row["task_id"]: row for row in resumed.report_rows()}
        assert [e["id"] for e in rows["apply-batch-0"]["skipped_items"]] == ["s0"]

        # A later resume reads it as done: nothing is dispatched again.
        again = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls),
            state=state, resume=True,
        ).run()
        assert again.exit_code == 0, again.messages
        assert calls == ["apply-batch-0", "apply-batch-1"]
        assert list(again.outcomes[0].skipped_items) == ["s0"]

    def test_an_answer_over_changed_files_is_refused_and_nothing_runs(self, env):
        calls = []
        state = FakeState(env.tmp_path / "state.json")
        env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls), state=state,
        ).run()
        env.plan.write_text(env.plan.read_text() + "the user's own edit\n")

        resumed = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls),
            state=state, resume=True, partial_skips=["apply-batch-0"],
        ).run()

        assert resumed.exit_code == 3
        assert calls == ["apply-batch-0"], "nothing is dispatched"
        text = "\n".join(resumed.messages)
        assert "--human-task apply-batch-0=skip: plans/p.md changed since apply-batch-0 stopped" in text
        assert "nothing was dispatched and no file was changed" in text

    @pytest.mark.parametrize("answer,why", [
        ("apply-batch-1", "apply-batch-1 has no result on disk"),
        ("apply-batch-7", "this run holds no batch apply-batch-7"),
    ])
    def test_an_answer_for_a_batch_that_did_not_stop_so_is_refused(self, env, answer, why):
        calls = []
        env.scheduler(_codex_08_tasks(env), respond=_codex_08_respond(env, calls)).run()
        resumed = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls),
            resume=True, partial_skips=[answer],
        ).run()
        assert resumed.exit_code == 3
        assert calls == ["apply-batch-0"]
        assert why in "\n".join(resumed.messages)

    def test_a_batch_that_stopped_otherwise_is_not_offered_the_answer(self, env):
        """Nothing attested (no `ids`), or the runner said it failed: the
        bytes it left are not "what it applied", so only the resume and the
        restore are offered — and an answer for it is refused."""
        for reply in (
            {"status": "applied", "ids": [], "files_touched": ["plans/p.md"],
             "summary": "s"},
            {"status": "failed", "ids": ["s1"], "files_touched": ["plans/p.md"],
             "summary": "s"},
        ):
            respond = _respond_with(env, reply)
            result = env.scheduler(
                [_two_item_task(env)], respond=respond, resume=False,
            ).run()
            assert result.exit_code == 3
            assert result.outcomes[0].write_outcome == executor.WRITE_PARTIAL
            assert "answers" not in result.stop
            assert "unapplied_ids" not in result.stop
            assert "--human-task" not in result.stop["headline"]
            refused = env.scheduler(
                [_two_item_task(env)], respond=respond, resume=True,
                partial_skips=["apply-batch-0"],
            ).run()
            assert refused.exit_code == 3
            assert ("did not stop as a partial application whose report left "
                    "items out") in "\n".join(refused.messages)
            env.plan.write_text("# Plan\n\n## Step 1\nbody\n", encoding="utf-8")


class TestTheCodex08ShapeThroughTheApplyPhase(_PairRig):
    """The whole loop on the real orchestrator: the batch leaves an item out
    of its report, the phase stops needs-human offering the answer (on the
    record the tool routes read), and the answer finishes the phase."""

    def test_the_stop_names_the_answer_and_the_answer_finishes_the_phase(
        self, repo, capsys,
    ):
        from utils import next_phase

        plan = self._review_pair(repo)
        ids = self._ids(plan)
        left_out, applied_id = ids[0]
        attested = [applied_id] + ids[1]

        assert self._run(repo, plan, attested) == 3
        answers = next_phase.stop_answers_on_record(plan, "apply-suggestions")
        assert [a["flags"] for a in answers] == [
            ["--resume", "--human-task", "apply-batch-0=skip"]]
        assert f"(`{left_out}`)" in answers[0]["why"]
        spelled = next_phase.resume_answers(
            "plans/apply.md", "claude-code", None, "apply-suggestions",
            plan_path=plan,
        )
        assert spelled[0]["command"].endswith(
            "--apply-suggestions plans/apply.md --resume --human-task apply-batch-0=skip")
        # Resuming without the answer only repeats the stop (08b).
        assert self._run(repo, plan, attested, "--resume") == 3

        capsys.readouterr()
        assert self._run(
            repo, plan, attested, "--resume", "--human-task", "apply-batch-0=skip",
        ) == 0
        summary = self._summary(plan)
        assert "- Suggestions applied: 2" in summary
        assert "- Suggestions skipped at your request (not applied): 1" in summary
        assert "Not applied: skipped at your request — --human-task apply-batch-0=skip" in summary
        out = capsys.readouterr().out
        assert f"[{left_out[:8]}] (suggestion 1 of apply-batch-0) not applied: skipped at your request" in out

    @pytest.mark.parametrize("flags,said", [
        (["--resume", "--human-task", "apply-batch-0=done"],
         "an apply batch takes only =skip"),
        (["--human-task", "apply-batch-0=skip"], "it needs --resume"),
    ])
    def test_an_answer_the_phase_cannot_take_is_exit_3_before_any_work(
        self, repo, capsys, flags, said,
    ):
        plan = self._review_pair(repo)
        capsys.readouterr()
        assert self._run(repo, plan, [], *flags) == 3
        err = capsys.readouterr().err
        assert said in err
        assert not (self._phase_dir(plan) / "write_tasks.json").exists()

    def test_the_native_leg_is_told_to_record_it_itself(self, repo, capsys):
        import apply_suggestions_orchestrator as orchestrator

        plan = self._review_pair(repo)
        capsys.readouterr()
        assert orchestrator.main([
            "--plan-file", str(plan), "--yes", "--resume",
            "--human-task", "apply-batch-0=skip",
        ]) == 3
        assert "util batch-done --skipped" in capsys.readouterr().err
