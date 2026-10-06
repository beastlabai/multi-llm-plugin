"""A refused ``--human-task`` answer is on the record the tool routes read
(fix round 19, the review of fix round 18's decline package, #1).

A batch stopped as a partial application whose report left an item out, and
offered ``--resume --human-task apply-batch-0=skip``. The user edited the file
by hand, answered ``=skip``, and the answer was refused (exit 3, nothing
dispatched) — on stderr only. The result line named no ``error``, so the tool
routes said "Its result line names no error", and the stop record the answer
was refused for still sat in the mirror with that answer, so
``util next-command --blocked`` offered the refused answer again as the way
forward. The refusals ``_setup`` makes before any work (``=done``, no
``--resume``) and implement's own refused answers left no ``error`` either.
"""

from __future__ import annotations

import json

import pytest

from tests.test_write_executor import FakeState, env  # noqa: F401
from tests.test_write_batch_declined import (
    SKIP_FLAGS,
    _codex_08_respond,
    _codex_08_tasks,
    _PairRig,
)
from tests.test_write_executor_orchestrators import repo  # noqa: F401


def _mirror(phase_dir):
    return json.loads((phase_dir / ".status.json").read_text())["progress"][
        "write_subagents"]


class TestTheSchedulerRecordsTheRefusal:

    def _stopped(self, env, calls):
        state = FakeState(env.tmp_path / "state.json")
        first = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls), state=state,
        ).run()
        assert first.exit_code == 3
        assert _mirror(env.phase_dir)["stop"]["answers"][0]["flags"] == SKIP_FLAGS
        return state

    def test_the_refused_answer_is_a_stop_that_offers_it_no_more(self, env):
        calls = []
        state = self._stopped(env, calls)
        env.plan.write_text(env.plan.read_text() + "the user's own edit\n")

        refused = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls),
            state=state, resume=True, partial_skips=["apply-batch-0"],
        ).run()

        assert refused.exit_code == 3
        assert calls == ["apply-batch-0"], "nothing is dispatched"
        stop = refused.stop
        assert stop["kind"] == "answer-refused"
        assert "answers" not in stop
        assert stop["refused_answers"] == [["--human-task", "apply-batch-0=skip"]]
        assert stop["headline"].startswith(
            "apply-suggestions cannot take --human-task apply-batch-0=skip: "
            "plans/p.md changed since apply-batch-0 stopped"
        ), stop["headline"]
        assert "Nothing was dispatched and no file was changed." in stop["headline"]
        assert stop["headline"].endswith(
            "Resume without it to dispatch the batch again, or restore the "
            "batch first (util restore-batch) and then resume.")
        # The mirror the tool routes read: the refusal, and no answer.
        mirror = _mirror(env.phase_dir)
        assert mirror["stop"]["kind"] == "answer-refused"
        assert "answers" not in mirror["stop"]
        # The earlier run's rows stand: this run dispatched nothing.
        assert mirror["failed_task_ids"] == ["apply-batch-0"]
        assert [row["task_id"] for row in mirror["tasks"]] == ["apply-batch-0"]

    def test_an_answer_for_another_batch_leaves_the_right_one_offered(self, env):
        calls = []
        state = self._stopped(env, calls)

        refused = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls),
            state=state, resume=True, partial_skips=["apply-batch-7"],
        ).run()

        assert refused.exit_code == 3
        assert refused.stop["kind"] == "answer-refused"
        assert "this run holds no batch apply-batch-7" in refused.stop["headline"]
        assert [a["flags"] for a in refused.stop["answers"]] == [SKIP_FLAGS]
        assert [a["flags"] for a in _mirror(env.phase_dir)["stop"]["answers"]] == [
            SKIP_FLAGS]


class TestTheResultLineNamesTheRefusal(_PairRig):
    """The real apply-suggestions orchestrator, as the tool routes see it."""

    def _context(self):
        from utils import cli_common

        return cli_common.activate_result(cli_common.ResultContext(
            mode="apply-suggestions", phase="apply-suggestions"))

    def test_a_refused_skip_is_the_error_and_is_not_offered_again(self, repo, capsys):
        from utils import cli_common, next_phase

        plan = self._review_pair(repo)
        ids = self._ids(plan)
        _left_out, applied_id = ids[0]
        attested = [applied_id] + ids[1]
        assert self._run(repo, plan, attested) == 3
        assert [a["flags"] for a in next_phase.stop_answers_on_record(
            plan, "apply-suggestions")] == [SKIP_FLAGS]
        plan.write_text(plan.read_text() + "\nhand edit\n", encoding="utf-8")

        context = self._context()
        try:
            code = self._run(
                repo, plan, attested, "--resume", "--human-task", "apply-batch-0=skip")
            error = context.error
        finally:
            cli_common.deactivate_result()

        assert code == 3
        assert (error or "").startswith(
            "apply-suggestions cannot take --human-task apply-batch-0=skip: "
            "plans/apply.md changed since apply-batch-0 stopped"), error
        assert next_phase.stop_answers_on_record(plan, "apply-suggestions") == []

    @pytest.mark.parametrize("flags,said", [
        (["--resume", "--human-task", "apply-batch-0=done"],
         "an apply batch takes only =skip"),
        (["--human-task", "apply-batch-0=skip"], "so it needs --resume"),
    ])
    def test_a_refusal_before_any_work_is_the_error(self, repo, capsys, flags, said):
        from utils import cli_common

        plan = self._review_pair(repo)
        context = self._context()
        try:
            assert self._run(repo, plan, [], *flags) == 3
            error = context.error
        finally:
            cli_common.deactivate_result()
        assert (error or "").startswith(
            "apply-suggestions cannot take this --human-task answer: "), error
        assert said in error


    def test_the_native_leg_s_refusal_is_the_error(self, repo, capsys):
        import apply_suggestions_orchestrator as orchestrator
        from utils import cli_common

        plan = self._review_pair(repo)
        context = self._context()
        try:
            assert orchestrator.main([
                "--plan-file", str(plan), "--yes", "--resume",
                "--human-task", "apply-batch-0=skip",
            ]) == 3
            error = context.error
        finally:
            cli_common.deactivate_result()
        assert "the --exec-subagents leg stopped" in (error or ""), error
        assert "util batch-done --skipped <id>" in error


class TestImplementNamesARefusedAnswer:

    def test_an_answer_for_a_task_that_did_not_fail_is_the_error(
        self, tmp_path, monkeypatch, capsys,
    ):
        from tests.test_check_gate_fixture import repo as gate_repo  # noqa: F401
        import multi_llm

        fixture = gate_repo.__wrapped__(tmp_path, monkeypatch)
        rig = next(fixture)
        try:
            capsys.readouterr()
            code = multi_llm.main([
                "implement", str(rig.plan), "--yes", "--exec-subagents",
                "--resume", "--human-task", "T002=done",
            ])
            out = capsys.readouterr().out
        finally:
            fixture.close()
        assert code == 3
        line = [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        result = json.loads(line.split(" ", 1)[1])
        assert result["error"].startswith(
            "--human-task answers a human task, a code task the last run "
            "recorded failed"), result
        assert "T002" in result["error"]


class TestARefusalWithNoEarlierRun:

    def test_the_mirror_is_this_run_s_own_and_status_reads_it(self, env):
        from utils import run_status

        refused = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, []),
            resume=True, partial_skips=["apply-batch-0"],
        ).run()

        assert refused.exit_code == 3
        mirror = _mirror(env.phase_dir)
        assert mirror["stop"]["kind"] == "answer-refused"
        assert mirror["kind"] == "apply_suggestions"
        assert mirror["total"] == 2
        assert "apply-batch-0 has no result on disk" in mirror["stop"]["headline"]
        assert run_status._mirror_text("write_subagents", mirror) == [
            "apply_suggestions 0/2 batches ok, 0/3 items applied"]
