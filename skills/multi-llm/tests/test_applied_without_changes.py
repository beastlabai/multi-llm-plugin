"""A write batch that says `applied` and changes nothing is never done.

Fix round 14, from the fifth live run (OpenCode racing 06, 2026-10-06): T016's
runner replied `status: applied` and named `index.html`, and `index.html` was
byte-for-byte unchanged. The verdict was `clean-failure` at `status: ok`, so
`state.json` recorded T016 `completed`, the stop listed it under "Done and
kept", the `--resume` reported "2/2 batches ok", and T013/T014 ran on top of a
function (`checkOpponentFinish`) that never existed. Elsewhere `clean-failure`
stopped implement (a `failed` claim over an empty diff), so the word had two
meanings.

Now: nothing moved, so the batch is sent once more, told that its previous
attempt changed no file (the retry ladder's rule 9 — rule 8 forbids re-sending
only an attempt that moved bytes). A re-send that lands the change is applied;
one that honestly answers `no-op` is a no-op; one that again says `applied`
over an unchanged tree is a failed batch: never `ok`, never `completed`, the
loop stops, the tasks after it are not run, and `--resume` dispatches it again.
A check task (no declared files) changes nothing by design and is untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import implement_orchestrator  # noqa: E402
from tests.test_write_executor import _patch_edit, env  # noqa: E402,F401
from tests.test_write_executor_orchestrators import (  # noqa: E402,F401
    IMPLEMENT_WRITE_IDS,
    _phase_dir,
    _seed_generate_tasks,
    repo,
)
from utils import subagent_executor as executor  # noqa: E402
from utils import subagent_prompts  # noqa: E402
from utils.retry_ladder import RESEND_NOTHING_CHANGED, RetryLadder  # noqa: E402

HEADING = subagent_prompts.NOTHING_CHANGED_RESEND_HEADING

APPLIED_S0 = {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
              "summary": "added the step"}


def _dispatch_prompts(log: Path):
    rows = [json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines() if line]
    return [r["prompt"] for r in rows if "--version" not in r.get("argv", [])]


# ===========================================================================
# The scheduler, one apply batch
# ===========================================================================

class TestTheReSend:

    def test_a_re_send_that_lands_the_change_is_applied(self, env):
        """The common recovery: the runner's edit tool failed silently, and the
        re-send, told so, makes the edit."""
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)], result=APPLIED_S0)
        env.monkeypatch.setenv("MOCK_LLM_EDITS_IF_PROMPT_HAS", HEADING)

        result = env.scheduler([env.apply_task(0), env.apply_task(1)]).run()

        first = result.outcomes[0]
        assert first.status == executor.STATUS_OK, first.reason
        assert first.write_outcome == executor.WRITE_APPLIED
        assert first.files_touched == ["plans/p.md"]
        assert [(a.status, a.repair) for a in first.attempts] == [
            (executor.STATUS_BATCH_FAILED, None),
            (executor.STATUS_OK, RESEND_NOTHING_CHANGED),
        ]
        assert "no file changed" in first.attempts[0].error
        assert first.attempts[0].capture_path, "the attempt that changed nothing is kept"
        prompts = _dispatch_prompts(log)
        assert HEADING not in prompts[0] and HEADING in prompts[1]
        assert "Apply batch 0" in prompts[1], "the re-send is the batch, not a re-ask"
        # The loop went on.
        assert [o.task_id for o in result.outcomes] == ["apply-batch-0", "apply-batch-1"]

    def test_the_batch_and_its_re_send_are_kept_as_separate_prompt_files(self, env):
        """Run 6 (#10b): the executor leg keeps what each dispatch sent under
        `{phase}/prompts/`, the native leg's layout. The re-send never replaces
        the batch's own file; it is the attempt's own, and the next dispatch
        of the task clears it."""
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)], result=APPLIED_S0)
        env.monkeypatch.setenv("MOCK_LLM_EDITS_IF_PROMPT_HAS", HEADING)

        env.scheduler([env.apply_task(0)]).run()

        prompts = env.phase_dir / "prompts"
        sent = _dispatch_prompts(log)
        assert sorted(p.name for p in prompts.iterdir()) == [
            "apply-batch-0.attempt-2.prompt.txt", "apply-batch-0.prompt.txt"]
        assert (prompts / "apply-batch-0.prompt.txt").read_text(
            encoding="utf-8") == sent[0]
        assert (prompts / "apply-batch-0.attempt-2.prompt.txt").read_text(
            encoding="utf-8") == sent[1]
        assert HEADING not in sent[0] and HEADING in sent[1]

        # A fresh dispatch of the same task: its first attempt rewrites the
        # batch's file and clears the earlier dispatch's re-send.
        (prompts / "apply-batch-0.prompt.txt").write_text("stale", encoding="utf-8")
        (env.phase_dir / "apply_batch_0.json").unlink()
        env.monkeypatch.delenv("MOCK_LLM_EDITS_IF_PROMPT_HAS")
        again = env.scheduler([env.apply_task(0)]).run()
        assert len(again.outcomes[0].attempts) == 1, "dispatched once, no re-send"
        assert sorted(p.name for p in prompts.iterdir()) == ["apply-batch-0.prompt.txt"]
        assert (prompts / "apply-batch-0.prompt.txt").read_text(
            encoding="utf-8") == _dispatch_prompts(log)[-1]

    def test_a_planted_prompts_symlink_is_never_followed(self, env, tmp_path):
        """A jailed repo-wide batch can write the phase dir; the scheduler is
        not jailed, so a `prompts` link it planted must not carry the prompt
        out of the repository. The dispatch goes ahead, with one warning."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (env.phase_dir / "prompts").symlink_to(outside, target_is_directory=True)
        env.edits([_patch_edit(env.plan)], result=APPLIED_S0)
        lines = []

        result = env.scheduler([env.apply_task(0)], log=lines.append).run()

        assert result.outcomes[0].status == executor.STATUS_OK
        assert list(outside.iterdir()) == []
        assert any("could not keep apply-batch-0's dispatched prompt" in line
                   and "symlink" in line for line in lines), lines

    def test_a_re_send_that_finds_the_change_present_is_a_no_op(self, env):
        """Told that nothing changed, an honest runner whose change is already
        there answers `no-op`: the claim and the bytes agree."""
        env.edits([], result=APPLIED_S0)
        env.monkeypatch.setenv(
            "MOCK_LLM_WRITE_RESULT", json.dumps(APPLIED_S0),
        )
        real = executor._spawn
        prompts = []

        def spawn(**kwargs):
            prompts.append(kwargs.get("prompt") or "")
            if HEADING in prompts[-1]:
                env.monkeypatch.setenv("MOCK_LLM_WRITE_RESULT", json.dumps({
                    "status": "no-op", "ids": ["s0"], "files_touched": [],
                    "summary": "already in the plan"}))
            return real(**kwargs)

        env.monkeypatch.setattr(executor, "_spawn", spawn)
        result = env.scheduler([env.apply_task(0)]).run()

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_OK, outcome.reason
        assert outcome.write_outcome == executor.WRITE_NO_OP
        assert not outcome.applied_without_changes
        assert result.exit_code == 0

    def test_a_second_unchanged_claim_stops_the_loop_and_is_recorded_failed(self, env):
        from tests.test_write_executor import FakeState

        state = FakeState(env.tmp_path / "state.json")
        env.edits([], result=APPLIED_S0)

        result = env.scheduler(
            [env.apply_task(0), env.apply_task(1)], state=state,
        ).run()

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert outcome.write_outcome == executor.WRITE_CLEAN_FAILURE
        assert outcome.applied_without_changes
        assert result.exit_code == 3
        assert len(result.outcomes) == 1
        state.reload()
        assert state.state["task_status"]["apply-batch-0"] == "failed"
        row = result.to_dict()["tasks"][0]
        assert row["applied_without_changes"] is True
        assert row["status"] == executor.STATUS_BATCH_FAILED
        stop = result.stop
        assert stop["applied_without_changes"] is True
        assert "its change did not land" in stop["headline"]
        # `--human-task` is implement's flag: an apply phase has no answer to
        # name (fix round 15), only the resume that dispatches it again.
        assert "--human-task" not in stop["headline"]
        assert "--human-task" not in "\n".join(result.messages)
        assert "resume this phase" in stop["headline"]
        assert "answers" not in stop

    def test_a_check_task_that_changes_nothing_is_verified_without_a_re_send(self, env):
        """A check declares no files and changes nothing by design: its verdict
        is VERIFIED (or check-findings), never this one."""
        log = env.enable_call_log()
        env.edits([], result={"status": "applied", "ids": ["T009"],
                              "files_touched": [], "summary": "node --check ok"})
        task = env.apply_task(
            0, kind="implement", task_id="T009", expected_ids=["T009"],
            output_name="implement_batch_T009.json",
            declared_files=[], snapshot_files=[],
        )
        result = env.scheduler([task], kind="implement", finalize=False).run()

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_OK
        assert outcome.outcome_label == "verified"
        assert len(_dispatch_prompts(log)) == 1

    @staticmethod
    def _code_task(env, task_id="T009"):
        from utils import write_batch_scheduler as wbs

        return env.apply_task(
            0, kind="implement", task_id=task_id, expected_ids=[task_id],
            declared_files=[str(env.root / "src" / "a.py")],
            prompt=f"Do {task_id}\n\n## Changes Applied in Prior Batches\n\n"
                   f"{wbs.PRIOR_CHANGES_TOKEN}\n",
        )

    def test_a_change_outside_the_declared_files_is_judged_by_its_ids(self, env):
        """An implement task: declared files unchanged, another file moved:
        the bytes decide, as when a declared file moved too — applied with
        every id attested, a partial application without (it used to be a
        clean failure at `ok`)."""
        other = env.root / "src" / "b.py"
        env.edits([_patch_edit(other, "# moved\n")],
                  result={**APPLIED_S0, "ids": ["T009"], "files_touched": ["src/b.py"]})
        attested = env.scheduler(
            [self._code_task(env)], kind="implement", finalize=False,
        ).run().outcomes[0]
        assert attested.status == executor.STATUS_OK, attested.reason
        assert attested.write_outcome == executor.WRITE_APPLIED
        assert attested.reason.startswith("APPLIED outside the declared files")
        assert attested.files_touched == ["src/b.py"]

    def test_the_same_change_with_no_ids_is_a_partial_application(self, env):
        other = env.root / "src" / "b.py"
        env.edits([_patch_edit(other, "# moved\n")],
                  result={**APPLIED_S0, "ids": []})
        outcome = env.scheduler(
            [self._code_task(env)], kind="implement", finalize=False,
        ).run().outcomes[0]
        assert outcome.status == executor.STATUS_PARTIAL, outcome.reason
        assert outcome.write_outcome == executor.WRITE_PARTIAL

    @pytest.mark.parametrize("ids", [["s0"], []])
    def test_a_plan_batch_whose_plan_did_not_change_is_not_applied(self, env, ids):
        """Fix round 15 (review of round 14): for a plan or task-list batch the
        declared file is the whole contract. One stray write beside an
        untouched plan read APPLIED ("the bytes decide"), which routed the T016
        shape around rule 9 — and could upgrade a sibling's stray file into
        this batch's "applied". It is sent once more, told that the plan did
        not change, and then recorded failed."""
        from tests.test_write_executor import FakeState

        log = env.enable_call_log()
        state = FakeState(env.tmp_path / "state.json")
        stray = env.root / "scratch-notes.md"
        env.edits([{"path": str(stray), "mode": "create", "content": "notes\n"}],
                  result={**APPLIED_S0, "ids": ids})
        plan_before = env.plan.read_bytes()

        result = env.scheduler(
            [env.apply_task(0), env.apply_task(1)], state=state,
        ).run()

        assert env.plan.read_bytes() == plan_before
        outcome = result.outcomes[0]
        assert not outcome.ok, outcome.reason
        assert outcome.status == executor.STATUS_BATCH_FAILED
        assert outcome.write_outcome == executor.WRITE_CLEAN_FAILURE
        assert outcome.applied_without_changes
        assert outcome.reason.startswith("NOT APPLIED"), outcome.reason
        assert "scratch-notes.md" in outcome.reason
        assert "no other file changed" not in outcome.reason
        assert outcome.files_touched == ["scratch-notes.md"], "the stray stays restorable"
        assert len(result.outcomes) == 1, "the next batch was dispatched"
        state.reload()
        assert state.state["task_status"]["apply-batch-0"] == "failed"
        prompts = _dispatch_prompts(log)
        assert len(prompts) == 2, "rule 9 sends it once more"
        assert HEADING in prompts[1]
        assert "plans/p.md is byte-for-byte as before it started" in prompts[1]
        assert "scratch-notes.md" in prompts[1]
        # The stop does not say nothing changed: a file outside the plan did.
        assert result.stop["bytes_moved"] is True
        assert "Nothing changed" not in result.stop["headline"]
        joined = "\n".join(result.messages)
        assert "Nothing was half-applied" not in joined

    def test_a_plan_batch_whose_re_send_lands_the_plan_edit_is_applied(self, env):
        stray = env.root / "scratch-notes.md"
        env.edits([{"path": str(stray), "mode": "create", "content": "notes\n"}],
                  result=APPLIED_S0)
        prompts = []
        real = executor._spawn

        def spawn(**kwargs):
            prompts.append(kwargs.get("prompt") or "")
            if HEADING in prompts[-1]:
                env.edits([_patch_edit(env.plan)], result=APPLIED_S0)
            return real(**kwargs)

        env.monkeypatch.setattr(executor, "_spawn", spawn)
        outcome = env.scheduler([env.apply_task(0)]).run().outcomes[0]
        assert outcome.status == executor.STATUS_OK, outcome.reason
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert "plans/p.md" in outcome.files_touched

    def test_a_run_before_the_fix_is_not_resumed_as_done(self, env):
        """A result file stamped `clean-failure` at `ok` (a run before fix
        round 14) is dispatched again: nothing moved, so that is safe."""
        stale = dict(APPLIED_S0, verdict={
            "write_outcome": "clean-failure", "status": "ok", "ok": True,
            "files_touched": [], "reason": "CLEAN FAILURE: …",
        })
        (env.phase_dir / "apply_batch_0.json").write_text(
            json.dumps(stale), encoding="utf-8")
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)], result=APPLIED_S0)

        result = env.scheduler([env.apply_task(0)], resume=True).run()

        outcome = result.outcomes[0]
        assert not outcome.skipped
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert len(_dispatch_prompts(log)) == 1

    def test_the_native_invoker_re_sends_the_same_way(self, env):
        """`respond=` (the simulated native invoker) classifies with the same
        code, so it re-sends the same way."""
        prompts = []

        def respond(task, prompt):
            prompts.append(prompt)
            if HEADING in prompt:
                with env.plan.open("a", encoding="utf-8") as handle:
                    handle.write("edit\n")
            return dict(APPLIED_S0)

        result = env.scheduler([env.apply_task(0)], respond=respond).run()

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_OK, outcome.reason
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert len(prompts) == 2 and HEADING in prompts[1]


# ===========================================================================
# The answers an applied-without-changes stop offers (fix round 15)
# ===========================================================================

class TestTheAnswersOfAChangeThatDidNotLand:
    """Review of round 14 (R1): the stop's message said "once you have made
    its change yourself", while the recorded answers offered "accept T009's
    failure: T009 counts as done and the tasks after it run" first — relayed
    verbatim by `util next-command` and the tool routes. A change that did
    not land is not a failure to accept: going on without it comes first,
    and `done` says it is for a change the user made by hand."""

    @staticmethod
    def _run(env):
        from utils import write_batch_scheduler as wbs

        task = env.apply_task(
            0, kind="implement", task_id="T009", expected_ids=["T009"],
            declared_files=[str(env.root / "src" / "a.py")],
            prompt="Do T009\n\n## Changes Applied in Prior Batches\n\n"
                   f"{wbs.PRIOR_CHANGES_TOKEN}\n",
        )
        env.edits([], result={**APPLIED_S0, "ids": ["T009"],
                              "files_touched": ["src/a.py"]})
        result = env.scheduler([task], kind="implement", finalize=False).run()
        assert result.outcomes[0].applied_without_changes
        return result

    def test_skip_comes_first_and_done_is_for_a_change_made_by_hand(self, env):
        answers = self._run(env).stop["answers"]
        assert [a["flags"] for a in answers] == [
            ["--resume", "--human-task", "T009=skip"],
            ["--resume", "--human-task", "T009=done"],
        ]
        done = answers[1]["why"]
        assert done.startswith("once you have made T009's change yourself"), done
        assert "accept" not in done and "failure" not in done
        assert all("accept T009's failure" not in a["why"] for a in answers)

    def test_the_stop_message_offers_them_in_the_same_order(self, env):
        result = self._run(env)
        joined = "\n".join(result.messages)
        assert "accept the failure" not in joined
        assert joined.index("T009=skip") < joined.index("T009=done"), joined
        head = result.stop["headline"]
        assert "T009=done" in head and head.index("T009=skip") < head.index("T009=done")

    def test_a_clean_failure_keeps_its_accept_answer(self, env):
        """A task that said `failed`: accepting the failure is still the
        answer the stop offers first."""
        from utils import write_batch_scheduler as wbs

        task = env.apply_task(
            0, kind="implement", task_id="T009", expected_ids=["T009"],
            declared_files=[str(env.root / "src" / "a.py")],
            prompt="Do T009\n\n## Changes Applied in Prior Batches\n\n"
                   f"{wbs.PRIOR_CHANGES_TOKEN}\n",
        )
        env.edits([], result={"status": "failed", "ids": [], "files_touched": [],
                              "summary": "could not"})
        answers = env.scheduler(
            [task], kind="implement", finalize=False,
        ).run().stop["answers"]
        assert answers[0]["flags"] == ["--resume", "--human-task", "T009=done"]
        assert answers[0]["why"].startswith("accept T009's failure")


# ===========================================================================
# implement, end to end: the measured T016 shape
# ===========================================================================

def test_an_implement_task_that_changed_nothing_is_not_done_and_holds_the_rest(repo, capsys):
    """T002 says `applied` twice and changes nothing: it is recorded failed,
    T003 (after it) is not run, the stop does not call it done, and the
    `--resume` dispatches it again — and records its real summary."""
    from utils import prior_changes
    from utils.output_handler import get_output_dir
    from utils.state_manager import get_or_create_state

    _seed_generate_tasks(repo)
    repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
    ]))
    repo.monkeypatch.setenv("MOCK_LLM_EDITS_IF_PROMPT_HAS", "Title: Create the module")
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_RESULT", json.dumps({
        "status": "applied", "ids": ["T001", "T002", "T003"],
        "files_touched": ["src/main.py"], "summary": "did the task",
    }))

    code = implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--yes", "--exec-subagents"]
    )
    text = "".join(capsys.readouterr())

    assert code == 3, text
    state = get_or_create_state(repo.plan)
    assert state.get_task_status("T001") == "completed"
    assert state.get_task_status("T002") == "failed"
    assert state.get_task_status("T003") == "pending"
    assert "implement stopped at T002 (Implement the core): its change did not land" in text
    assert "Done and kept: 1 task(s) (T001)." in text
    assert "Not run (the stop came before they were dispatched): T003." in text
    assert "--human-task T002=done" in text
    report = json.loads(
        (_phase_dir(repo) / "write_batch_report.json").read_text(encoding="utf-8"))
    rows = {row["task_id"]: row for row in report["tasks"]}
    assert rows["T002"]["status"] == executor.STATUS_BATCH_FAILED
    assert rows["T002"]["applied_without_changes"] is True
    entries = prior_changes.entries(str(get_output_dir(repo.plan)))
    assert [e["id"] for e in entries] == ["T001", "T002"], entries
    # Fix round 15: the native leg's failure line, written on this leg too.
    assert entries[1]["summary"] == "(task failed — said applied, but its change did not land)"

    # The resume dispatches T002 again; this time the change lands.
    repo.monkeypatch.delenv("MOCK_LLM_EDITS_IF_PROMPT_HAS")
    code = implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--yes", "--exec-subagents", "--resume"]
    )
    text = "".join(capsys.readouterr())
    assert code == 0, text
    state = get_or_create_state(repo.plan)
    assert state.get_task_status("T002") == "completed"
    entries = prior_changes.entries(str(get_output_dir(repo.plan)))
    by_id = {e["id"]: e["summary"] for e in entries}
    assert by_id["T002"] == "did the task", "the resumed task's real entry was dropped"


def test_a_task_a_run_before_round_14_recorded_done_is_dispatched_again(repo, capsys):
    """Fix round 15 (review of round 14, #5): a run before fix round 14
    recorded an `applied` reply that moved nothing `completed` (OpenCode
    racing run 5's T016: its result file is stamped `clean-failure` at
    `ok`). Implement's `--resume` materializes only unfinished tasks, so the
    scheduler's re-dispatch of that verdict never got the chance: the task
    is handed out again, and the resume dispatches it."""
    from utils.state_manager import get_or_create_state

    _seed_generate_tasks(repo)
    repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
        {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
    ]))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
    assert implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--yes", "--exec-subagents"]) == 0
    capsys.readouterr()
    # What a run before fix round 14 left for T002.
    result_file = _phase_dir(repo) / "implement_batch_T002.json"
    data = json.loads(result_file.read_text(encoding="utf-8"))
    data["verdict"].update({"write_outcome": "clean-failure", "status": "ok",
                            "ok": True, "files_touched": []})
    result_file.write_text(json.dumps(data), encoding="utf-8")
    assert get_or_create_state(repo.plan).get_task_status("T002") == "completed"

    log = repo.tmp_path / "calls.jsonl"
    repo.monkeypatch.setenv("MOCK_LLM_CALL_LOG", str(log))
    code = implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--yes", "--exec-subagents", "--resume"])
    text = "".join(capsys.readouterr())
    assert code == 0, text
    prompts = _dispatch_prompts(log)
    titles = [line[len("Title: "):] for p in prompts for line in p.splitlines()
              if line.startswith("Title: ")]
    assert titles == ["Implement the core"], titles
    assert "T002" in text and "never landed" in text, text
    state = get_or_create_state(repo.plan)
    assert state.get_task_status("T002") == "completed"
    verdict = json.loads(result_file.read_text(encoding="utf-8"))["verdict"]
    assert verdict["write_outcome"] == executor.WRITE_APPLIED


# ===========================================================================
# The ladder's rule 9 and the prompt
# ===========================================================================

class TestRuleNine:

    def _ladder(self, **over):
        clock = {"now": 0.0}
        kwargs = dict(timeout=100, retries=1, now=lambda: clock["now"])
        kwargs.update(over)
        return RetryLadder(**kwargs), clock

    def test_no_grant_by_default(self):
        ladder, _ = self._ladder()
        assert ladder.resends_left == 0
        action = ladder.record_resend()
        assert action.retry is False and action.resend is False

    def test_one_grant_then_none(self):
        ladder, _ = self._ladder(resends=1)
        assert ladder.next_attempt() is not None
        first = ladder.record_resend()
        assert first.retry and first.resend and first.reason == RESEND_NOTHING_CHANGED
        assert ladder.resends_left == 0 and ladder.resends_used == 1
        assert ladder.next_attempt() is not None, "the granted re-send is admitted"
        assert ladder.record_resend().retry is False
        # Neither the transport budget nor the re-ask allowance was spent.
        assert ladder.transport_retries_left == 1

    def test_a_re_send_the_deadline_cannot_hold_is_refused_with_a_note(self):
        ladder, clock = self._ladder(resends=1)
        ladder.next_attempt()
        clock["now"] = 99.5
        action = ladder.record_resend()
        assert action.retry is False
        assert action.note and action.note.startswith("no re-send:")
        assert ladder.resends_left == 1

    def test_the_grant_table_covers_every_write_kind(self):
        from utils.subagent_tasks import WRITE_KINDS

        assert set(executor.RESENDS_BY_KIND) == set(WRITE_KINDS)
        assert all(value == 1 for value in executor.RESENDS_BY_KIND.values())


def test_the_resend_prompt_carries_the_batch_and_says_what_happened():
    prompt = subagent_prompts.build_nothing_changed_resend_prompt(
        "Implement the opponent finish check.\n\n## Result (required)\n…",
        declared=["index.html"], named=["index.html"],
    )
    assert prompt.startswith("Implement the opponent finish check.")
    assert prompt.index("## Result (required)") < prompt.index(HEADING)
    assert 'ended with `"status": "applied"` and named index.html as edited' in prompt
    assert "no file changed: index.html is byte-for-byte as before it started" in prompt
    assert '`"status": "no-op"`' in prompt and '`"status": "failed"`' in prompt
    assert "only for edits you made in this turn" in prompt


@pytest.mark.parametrize("name", [
    "implement.md", "apply-suggestions.md", "apply-task-suggestions.md",
    "apply-code-fixes.md",
])
def test_the_native_leg_is_told_the_same_rule(name):
    """The native host classifies its own batches: an `applied` reply over
    unchanged files is sent once more and then recorded failed, as on the
    executor leg."""
    text = (Path(__file__).parent.parent / "instructions" / name).read_text(
        encoding="utf-8")
    assert "answered `applied` but changed no file" in text
    assert "never `completed`" in text or "never a success" in text
