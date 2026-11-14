#!/usr/bin/env python3
"""The Stop hook on a native ``--improve-plan`` / ``--improve-code`` run (Fable
review 2026-10-09).

On Claude Code the composite is a core that runs ``util improve-steps`` first
(it ends with ``[IMPROVE_RERUN] <command>``, the composite's continuation),
then each phase's own core, then one closing ``util next-command``. Each phase
still prints its own standalone block: review-plan's ``--reaggregate`` names
``--apply-suggestions <plan>``. The check used to hold the report to the
turn's last block, so when the apply then stopped (the core reports the stop,
gives the stopped phase's answer and ``[IMPROVE_RERUN]``, and makes no
lookup), it blocked every correct report with "Missing command:
/multi-llm:multi-llm --apply-suggestions plans/x.md", and the forced turn
pasted a bare apply, which on the native leg re-emits every batch.

Now the planner call's ``[IMPROVE_RERUN]`` line is a block, and no later
block without a ready section (an inner phase's) replaces it; a lookup's,
which always carries one, does. A run that never ran the planner is read as
before.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.support.transcripts import Transcript, decision_of, run_check  # noqa: E402
from tests.support.wakeup_world import DISPATCH, call  # noqa: E402
from utils import detach, improve_steps, report_check, wakeup_check  # noqa: E402


@pytest.fixture(autouse=True)
def _short_settle(monkeypatch):
    monkeypatch.setattr(report_check, "SETTLE_BUDGET_S", 0.3, raising=False)


PLAN = "plans/x.md"
RERUN = {
    "improve-plan": "/multi-llm:multi-llm --improve-plan plans/x.md --quick --resume",
    "improve-code": "/multi-llm:multi-llm --improve-code plans/x.md --quick --resume",
}
REVIEW = {"improve-plan": "review-plan", "improve-code": "review-code"}
APPLY = {"improve-plan": "apply-suggestions", "improve-code": "apply-code-fixes"}
APPLY_DIR = {"improve-plan": "apply-suggestions", "improve-code": "apply-fixes"}
#: What each review's `--reaggregate` names: its standalone next step.
BARE_APPLY = {mode: f"/multi-llm:multi-llm --{APPLY[mode]} {PLAN}" for mode in RERUN}
#: The stopped apply's own answer, which the core gives beside the continuation.
ANSWER = {mode: f"{BARE_APPLY[mode]} --resume --human-task apply-batch-1=skip"
          for mode in RERUN}
GENERATE = f"/multi-llm:multi-llm --generate-tasks {PLAN}"
MODES = sorted(RERUN)


def _result(mode: str, exit_code: int = 0, phase: str = "null") -> str:
    return (f'MULTI_LLM_RESULT: {{"mode": "{mode}", "phase": {phase}, '
            f'"state": "done", "exit": {exit_code}, "artifacts": [], "next": null}}\n')


def _steps(mode: str) -> str:
    """``util improve-steps``' text block for a plan nothing has run on."""
    return (
        f"[IMPROVE_STEP] 1 {REVIEW[mode]} run (not reviewed yet)\n"
        "[IMPROVE_STEP_FLAGS] 1 --yes --quick\n"
        f"[IMPROVE_STEP] 2 {APPLY[mode]} run (review not applied yet)\n"
        "[IMPROVE_STEP_FLAGS] 2 --yes --agent-decide --no-confirm\n"
        f"{improve_steps.RERUN_MARKER} {RERUN[mode]}\n" + _result("util")
    )


def _util(t: Transcript, words: str, output: str) -> Transcript:
    return call(t, f"{DISPATCH} util {words}", output)


def _phase(t: Transcript, mode: str, flags: str, output: str) -> Transcript:
    return call(t, f'{DISPATCH} {mode} "{PLAN}" --host claude-code {flags}', output)


def _planned(mode: str) -> Transcript:
    """The typed prompt, the core loaded, then the planner's call."""
    t = Transcript().prompt(f"/multi-llm:multi-llm --{mode} {PLAN} --quick")
    t.meta("Base directory for this skill: /p/skills/multi-llm")
    core = (SKILL_ROOT / "instructions" / f"{mode}.md").read_text(encoding="utf-8")
    _util(t, f"instructions {mode}", core + _result("util"))
    _util(t, f'improve-steps "{PLAN}" --mode {mode} --host claude-code --quick',
          _steps(mode))
    return t


def _reviewed(mode: str) -> Transcript:
    """Step 1 done: the review's `--reaggregate` prints its standalone block."""
    t = _planned(mode)
    _util(t, f"instructions {REVIEW[mode]}", "# Review core\n" + _result("util"))
    _phase(t, REVIEW[mode], "--reaggregate",
           "Validation merged: 4 valid, 1 invalid.\n"
           f"[NEXT_COMMAND] {BARE_APPLY[mode]}\n" + _result(REVIEW[mode]))
    return t


def _apply_stopped(mode: str) -> Transcript:
    """Step 2's apply emitted its batches and one was recorded failed: the
    reviewer's reproduction, up to the report."""
    t = _reviewed(mode)
    _util(t, f"instructions {APPLY[mode]}", "# Apply core\n" + _result("util"))
    _phase(t, APPLY[mode], "--yes --agent-decide --no-confirm",
           f"[WRITE_TASKS_FILE] plans/x/{APPLY_DIR[mode]}/write_tasks.json\n"
           + _result(APPLY[mode]))
    _util(t, f"batch-done --phase-dir plans/x/{APPLY_DIR[mode]} "
             "--task-id apply-batch-1 --status failed",
          "recorded apply-batch-1: failed\n" + _result("util"))
    return t


def _stop_report(mode: str, *commands: str) -> str:
    parts = [f"## {mode}: stopped at step 2", f"{APPLY[mode]} stopped: batch "
             "apply-batch-1 failed (the edit did not apply)."]
    parts += [f"```\n{command}\n```" for command in commands]
    return "\n\n".join(parts)


LOOKUP = (
    f"[NEXT_COMMAND] {GENERATE}\n[NEXT_STEP_MARKDOWN]\n```\n{GENERATE}\n```\n"
    "[NEXT_STEP_MARKDOWN_END]\n" + _result("util next-command")
)


def _finished(mode: str) -> Transcript:
    """Both steps done: the apply's `--mark-completed`, then section 4's lookup."""
    t = _reviewed(mode)
    _phase(t, APPLY[mode], "--yes --agent-decide --no-confirm",
           f"[WRITE_TASKS_FILE] plans/x/{APPLY_DIR[mode]}/write_tasks.json\n"
           + _result(APPLY[mode]))
    _phase(t, APPLY[mode], "--mark-completed",
           f"[NEXT_COMMAND] {GENERATE}\n" + _result(APPLY[mode]))
    _util(t, f'next-command "{PLAN}" --after {mode} --host claude-code', LOOKUP)
    return t


def _turn(t: Transcript, tmp_path: Path):
    return report_check.read_turn(t.write(tmp_path / "turn.jsonl"))


# ---------------------------------------------------------------------------
# A stop is held to the composite's continuation
# ---------------------------------------------------------------------------

class TestAStopIsHeldToTheContinuation:

    @pytest.mark.parametrize("mode", MODES)
    def test_the_reviewers_stop_transcript_passes(self, tmp_path, capsys, mode):
        path = (_apply_stopped(mode)
                .say(_stop_report(mode, ANSWER[mode], RERUN[mode]))
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    @pytest.mark.parametrize("mode", MODES)
    def test_the_block_held_is_the_planners_not_the_reviews(self, tmp_path, mode):
        block = report_check.last_block(_turn(_apply_stopped(mode), tmp_path))
        assert block is not None and block.continuation
        assert block.commands() == [RERUN[mode]]

    @pytest.mark.parametrize("mode", MODES)
    def test_a_report_without_the_continuation_still_blocks(self, tmp_path, capsys, mode):
        path = (_apply_stopped(mode).say(_stop_report(mode, ANSWER[mode]))
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert f"```\n{RERUN[mode]}\n```" in reason
        assert "[IMPROVE_RERUN]" in reason
        # Never the review's standalone next step: that is what the forced
        # turn used to paste.
        assert f"```\n{BARE_APPLY[mode]}\n```" not in reason
        assert report_check.WHOLE_REPORT in reason

    @pytest.mark.parametrize("mode", MODES)
    def test_the_bare_apply_in_its_place_blocks(self, tmp_path, capsys, mode):
        path = (_apply_stopped(mode).say(_stop_report(mode, BARE_APPLY[mode]))
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert f"```\n{RERUN[mode]}\n```" in reason

    def test_a_stop_in_step_1_is_held_to_the_continuation_too(self, tmp_path, capsys):
        """The review exits 3 (no reviewer produced a result): no phase printed
        a block, and the report owes the continuation."""
        mode = "improve-plan"
        t = _planned(mode)
        _phase(t, REVIEW[mode], "--yes --quick",
               "ERROR: no model produced a result.\n" + _result(REVIEW[mode], 3))
        with_rerun = t.say(_stop_report(mode, RERUN[mode])).write(tmp_path / "a.jsonl")
        assert run_check(capsys, with_rerun) == ""
        t.entries.pop()
        without = t.say(_stop_report(mode)).write(tmp_path / "b.jsonl")
        reason = decision_of(run_check(capsys, without))["reason"]
        assert f"```\n{RERUN[mode]}\n```" in reason

    def test_a_planner_run_again_at_the_stop_is_read_the_same(self, tmp_path, capsys):
        """A second `util improve-steps` call (its step 2 then shows the
        inferred --resume) prints the same continuation: one command held."""
        mode = "improve-plan"
        t = _apply_stopped(mode)
        _util(t, f'improve-steps "{PLAN}" --mode {mode} --host claude-code --quick',
              _steps(mode))
        path = t.say(_stop_report(mode, ANSWER[mode], RERUN[mode])).write(tmp_path / "t.jsonl")
        assert run_check(capsys, path) == ""


# ---------------------------------------------------------------------------
# A finished run, and every run that never ran the planner, read as before
# ---------------------------------------------------------------------------

class TestEverythingElseIsReadAsBefore:

    @pytest.mark.parametrize("mode", MODES)
    def test_a_finished_run_is_held_to_its_lookup(self, tmp_path, capsys, mode):
        report = f"## {mode} Summary\n\nItems applied: 4\n\n```\n{GENERATE}\n```\n"
        path = _finished(mode).say(report).write(tmp_path / "t.jsonl")
        assert run_check(capsys, path) == ""

    @pytest.mark.parametrize("mode", MODES)
    def test_a_finished_run_without_the_lookups_command_blocks_on_it(
        self, tmp_path, capsys, mode
    ):
        report = f"## {mode} Summary\n\nItems applied: 4\n\n```\n{RERUN[mode]}\n```\n"
        path = _finished(mode).say(report).write(tmp_path / "t.jsonl")
        reason = decision_of(run_check(capsys, path))["reason"]
        assert f"```\n{GENERATE}\n```" in reason
        assert report_check.SECTION_REQUEST in reason
        assert RERUN[mode] not in reason

    def test_already_complete_prints_no_continuation(self, tmp_path, capsys):
        """`[ALREADY_COMPLETE]`, exit 2: no `[IMPROVE_RERUN]`, so the lookup
        alone is held, as for any run."""
        t = Transcript().prompt(f"/multi-llm:multi-llm --improve-plan {PLAN}")
        _util(t, f'improve-steps "{PLAN}" --mode improve-plan --host claude-code',
              "[IMPROVE_STEP] 1 review-plan skip (reviewed)\n"
              "[IMPROVE_STEP_FLAGS] 1 --yes\n"
              "[IMPROVE_STEP] 2 apply-suggestions skip (applied)\n"
              "[IMPROVE_STEP_FLAGS] 2 --yes --agent-decide --no-confirm\n"
              "[ALREADY_COMPLETE] improve-plan: nothing to do.\n" + _result("util", 2))
        _util(t, f'next-command "{PLAN}" --after improve-plan --host claude-code', LOOKUP)
        block = report_check.last_block(_turn(t, tmp_path))
        assert block is not None and not block.continuation
        path = t.say(f"Nothing to do.\n\n```\n{GENERATE}\n```").write(tmp_path / "t.jsonl")
        assert run_check(capsys, path) == ""

    def test_a_refused_planner_leaves_no_block(self, tmp_path, capsys):
        t = Transcript().prompt(f"/multi-llm:multi-llm --improve-plan {PLAN} -i")
        _util(t, f'improve-steps "{PLAN}" --mode improve-plan --host claude-code -i',
              "improve-plan runs unattended; run --review-plan, then "
              "--apply-suggestions.\n" + _result("util", 3))
        path = t.say("Refused: run the two phases.").write(tmp_path / "t.jsonl")
        assert report_check.last_block(_turn(t, tmp_path)) is None
        assert run_check(capsys, path) == ""

    def test_a_quoted_line_outside_the_planners_result_is_no_block(self, tmp_path, capsys):
        """A standalone review of a plan that quotes the planner's output (this
        repository's own plan does): the line is the plan's text, not the
        run's continuation, and the review's block is held as always."""
        quoted = ("```text\n[IMPROVE_RERUN] /multi-llm:multi-llm --improve-plan "
                  "plans/other.md --resume\n```\n")
        t = Transcript().prompt(f"/multi-llm:multi-llm --review-plan {PLAN}")
        call(t, "", quoted, name="Read", tool_input={"file_path": "/r/plans/x.md"})
        call(t, f"grep -n IMPROVE_RERUN {PLAN}", quoted)
        _util(t, "instructions improve-plan",
              "[IMPROVE_RERUN] <the command that continues this run>\n")
        _phase(t, "review-plan", "--reaggregate",
               f"[NEXT_COMMAND] {BARE_APPLY['improve-plan']}\n" + _result("review-plan"))
        block = report_check.last_block(_turn(t, tmp_path))
        assert block is not None and not block.continuation
        assert block.command == BARE_APPLY["improve-plan"]
        path = t.say(f"Done.\n\n```\n{BARE_APPLY['improve-plan']}\n```").write(
            tmp_path / "t.jsonl")
        assert run_check(capsys, path) == ""

    def test_a_subagent_quoting_the_planners_output_is_no_block(self, tmp_path):
        t = Transcript().prompt("go")
        call(t, "", _steps("improve-plan"), name="Agent",
             tool_input={"prompt": "util improve-steps"})
        assert report_check.last_block(_turn(t, tmp_path)) is None

    def test_without_continuations_the_line_is_not_read(self):
        """`util wakeup-check` reads `find_blocks` bare: the planner's line,
        printed before any phase runs, never ends a phase's hold."""
        assert report_check.find_blocks(_steps("improve-plan")) == []
        blocks = report_check.find_blocks(_steps("improve-plan"), continuations=True)
        assert [(b.command, b.continuation) for b in blocks] == [
            (RERUN["improve-plan"], True)]

    def test_a_planner_line_after_a_phase_call_keeps_the_wakeup_hold(self, tmp_path):
        t = _reviewed("improve-plan")
        _phase(t, "apply-suggestions", "--yes --agent-decide --no-confirm",
               "[WRITE_TASKS_FILE] plans/x/apply-suggestions/write_tasks.json\n"
               + _result("apply-suggestions"))
        _util(t, f'improve-steps "{PLAN}" --mode improve-plan --host claude-code',
              _steps("improve-plan"))
        assert wakeup_check.phase_in_flight(_turn(t, tmp_path)) == "apply-suggestions"


# ---------------------------------------------------------------------------
# A cascade: every `util improve-cascade next` call is a planner call
# ---------------------------------------------------------------------------
# Under --cascade, `util improve-steps` hands off ([IMPROVE_CASCADE_NEXT] and
# [IMPROVE_RERUN]), and each stage comes from a `util improve-cascade next`
# call that prints the stage's steps, [IMPROVE_CASCADE_CONTINUE] and
# [IMPROVE_RERUN] again. A continue call carries no selector, so its
# continuation is the bare `--cascade --resume`: the report is held to the
# latest one printed, never to an inner phase's [NEXT_COMMAND].

#: The hand-off's and the first `next` call's continuation (the typed --quick),
#: and every continue call's.
CASCADE_FIRST = {mode: f"/multi-llm:multi-llm --{mode} {PLAN} --quick --cascade --resume"
                 for mode in MODES}
CASCADE_RERUN = {mode: f"/multi-llm:multi-llm --{mode} {PLAN} --cascade --resume"
                 for mode in MODES}
CASCADE_ID = "c0ffee00deadbeef0123456789abcdef"
SUMMARY = "plans/x/improve-plan/cascades/20261113T090000-c0ffee00/summary.md"


def _next_call(mode: str, *flags: str) -> str:
    return f'improve-cascade next "{PLAN}" --mode {mode} --host claude-code ' + " ".join(flags)


def _continue(mode: str) -> str:
    return f"{DISPATCH} util {_next_call(mode, '--cascade-id', CASCADE_ID)}"


def _stage(mode: str, k: int, of: int, rerun: str) -> str:
    """One `next` call's stage block."""
    return (
        f"[IMPROVE_CASCADE] stage {k} of {of}: codex:m{k}\n"
        f"[IMPROVE_STEP] 1 {REVIEW[mode]} run (new round, --force)\n"
        "[IMPROVE_STEP_FRESH] 1\n"
        f"[IMPROVE_STEP_FLAGS] 1 --yes --models codex:m{k} --force --rerun-all "
        f"--cascade-stage-file /r/plans/x/stage-{k}/inputs/manifest.json\n"
        f"[IMPROVE_STEP] 2 {APPLY[mode]} run (review not applied yet)\n"
        "[IMPROVE_STEP_FLAGS] 2 --yes --agent-decide --no-confirm\n"
        f"[IMPROVE_CASCADE_CONTINUE] {_continue(mode)}\n"
        f"{improve_steps.RERUN_MARKER} {rerun}\n" + _result("util improve-cascade")
    )


def _cascade_stage_done(t: Transcript, mode: str) -> None:
    """A stage's review and apply, each phase printing its own standalone block."""
    _phase(t, REVIEW[mode], "--reaggregate",
           f"[NEXT_COMMAND] {BARE_APPLY[mode]}\n" + _result(REVIEW[mode]))
    _phase(t, APPLY[mode], "--mark-completed",
           f"[NEXT_COMMAND] {GENERATE}\n" + _result(APPLY[mode]))


def _cascade_started(mode: str) -> Transcript:
    """The hand-off, the first `next` call, and stage 1 run."""
    t = Transcript().prompt(f"/multi-llm:multi-llm --{mode} {PLAN} --cascade --quick")
    _util(t, f'improve-steps "{PLAN}" --mode {mode} --host claude-code --cascade --quick',
          f"[IMPROVE_CASCADE_NEXT] {DISPATCH} util {_next_call(mode, '--cascade', '--quick')}\n"
          f"{improve_steps.RERUN_MARKER} {CASCADE_FIRST[mode]}\n" + _result("util"))
    _util(t, _next_call(mode, "--cascade", "--quick"), _stage(mode, 1, 2, CASCADE_FIRST[mode]))
    _cascade_stage_done(t, mode)
    return t


def _cascade_stopped(mode: str) -> Transcript:
    """Stage 2 handed out by a continue call, and its apply stopped."""
    t = _cascade_started(mode)
    call(t, _continue(mode), _stage(mode, 2, 2, CASCADE_RERUN[mode]))
    _phase(t, REVIEW[mode], "--reaggregate",
           f"[NEXT_COMMAND] {BARE_APPLY[mode]}\n" + _result(REVIEW[mode]))
    _phase(t, APPLY[mode], "--yes --agent-decide --no-confirm",
           f"[WRITE_TASKS_FILE] plans/x/{APPLY_DIR[mode]}/write_tasks.json\n"
           + _result(APPLY[mode]))
    return t


def _cascade_finished(mode: str) -> Transcript:
    """Both stages run, `next` exits 2 with [CASCADE_COMPLETE], then the lookup."""
    t = _cascade_started(mode)
    call(t, _continue(mode), _stage(mode, 2, 2, CASCADE_RERUN[mode]))
    _cascade_stage_done(t, mode)
    call(t, _continue(mode), f"[CASCADE_COMPLETE] 2 stages; table: {SUMMARY}\n"
         + _result("util improve-cascade", 2))
    _util(t, f'next-command "{PLAN}" --after {mode} --host claude-code', LOOKUP)
    return t


class TestACascade:

    @pytest.mark.parametrize("mode", MODES)
    def test_a_stop_mid_stage_is_held_to_the_latest_continuation(self, tmp_path, capsys,
                                                                 mode):
        t = _cascade_stopped(mode)
        block = report_check.last_block(_turn(t, tmp_path))
        assert block is not None and block.continuation
        assert block.commands() == [CASCADE_RERUN[mode]]
        path = t.say(_stop_report(mode, ANSWER[mode], CASCADE_RERUN[mode])).write(
            tmp_path / "t.jsonl")
        assert run_check(capsys, path) == ""

    @pytest.mark.parametrize("mode", MODES)
    def test_the_hand_offs_continuation_alone_no_longer_passes(self, tmp_path, capsys, mode):
        path = (_cascade_stopped(mode)
                .say(_stop_report(mode, ANSWER[mode], CASCADE_FIRST[mode]))
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert f"```\n{CASCADE_RERUN[mode]}\n```" in reason
        assert "util improve-cascade next" in reason

    @pytest.mark.parametrize("mode", MODES)
    def test_an_inner_next_command_is_never_the_block(self, tmp_path, capsys, mode):
        path = (_cascade_stopped(mode).say(_stop_report(mode, BARE_APPLY[mode]))
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert f"```\n{CASCADE_RERUN[mode]}\n```" in reason
        assert f"```\n{BARE_APPLY[mode]}\n```" not in reason

    @pytest.mark.parametrize("mode", MODES)
    def test_a_completed_cascade_is_held_to_its_lookup(self, tmp_path, capsys, mode):
        t = _cascade_finished(mode)
        block = report_check.last_block(_turn(t, tmp_path))
        assert block is not None and not block.continuation
        report = f"## {mode} Summary\n\n### Cascade\n\n```\n{GENERATE}\n```\n"
        assert run_check(capsys, t.say(report).write(tmp_path / "t.jsonl")) == ""

    @pytest.mark.parametrize("mode", MODES)
    def test_a_completed_cascade_without_the_lookups_command_blocks(self, tmp_path, capsys,
                                                                    mode):
        report = f"## {mode} Summary\n\n```\n{CASCADE_RERUN[mode]}\n```\n"
        path = _cascade_finished(mode).say(report).write(tmp_path / "t.jsonl")
        reason = decision_of(run_check(capsys, path))["reason"]
        assert f"```\n{GENERATE}\n```" in reason

    @pytest.mark.parametrize("mode", MODES)
    def test_a_completed_cascade_whose_lookup_failed_keeps_the_latest_continuation(
            self, tmp_path, capsys, mode):
        """No section: the last stage's apply printed `[NEXT_COMMAND]` after the
        last `next` call's continuation, and that inner block never replaces it."""
        t = _cascade_finished(mode)
        t.entries.pop()  # the lookup's result
        t.entries.pop()  # and its call
        block = report_check.last_block(_turn(t, tmp_path))
        assert block is not None and block.continuation
        assert block.commands() == [CASCADE_RERUN[mode]]
        report = f"## {mode} Summary\n\n```\n{CASCADE_RERUN[mode]}\n```\n"
        assert run_check(capsys, t.say(report).write(tmp_path / "t.jsonl")) == ""

    def test_a_stopped_last_stage_prints_the_continuation_on_its_own(self, tmp_path, capsys):
        """The continue call exits 3 (the last stage's review produced nothing):
        its stdout is the [IMPROVE_RERUN] line alone, which is the block."""
        mode = "improve-plan"
        t = _cascade_started(mode)
        call(t, _continue(mode), _stage(mode, 2, 2, CASCADE_RERUN[mode]))
        _phase(t, REVIEW[mode], "--yes --models codex:m2",
               "ERROR: no model produced a result.\n" + _result(REVIEW[mode], 3))
        call(t, _continue(mode),
             "ERROR: cascade c0ffee00: stage 2 of 2 (codex:m2) produced no review result\n"
             f"{improve_steps.RERUN_MARKER} {CASCADE_RERUN[mode]}\n"
             + _result("util improve-cascade", 3))
        block = report_check.last_block(_turn(t, tmp_path))
        assert block is not None and block.commands() == [CASCADE_RERUN[mode]]
        path = t.say(_stop_report(mode, CASCADE_RERUN[mode])).write(tmp_path / "t.jsonl")
        assert run_check(capsys, path) == ""


@pytest.mark.parametrize("command, planner", [
    (f'{DISPATCH} util improve-steps "{PLAN}" --mode improve-plan --host claude-code', True),
    (f'{DISPATCH} util improve-cascade next "{PLAN}" --mode improve-plan --host claude-code '
     f'--cascade-id {CASCADE_ID}', True),
    ("uv run --project /p -- python '/p q/multi_llm.py' util improve-cascade next plans/x.md "
     "--mode improve-code --cascade", True),
    ("uv run --project /p -- python /p/multi_llm.py util improve-steps plans/x.md "
     "--mode improve-code", True),
    ("python 'multi_llm.py' 'util' 'improve-steps' plans/x.md --mode improve-plan", True),
    (f"{DISPATCH} util instructions improve-plan", False),
    (f"grep -rn improve-steps {PLAN}", False),
    (f"{DISPATCH} util next-command {PLAN} --after improve-plan", False),
])
def test_which_calls_are_the_planner(tmp_path, command, planner):
    t = Transcript().prompt("go")
    call(t, command, "x")
    assert bool(report_check.planner_call_ids(_turn(t, tmp_path))) is planner


# ---------------------------------------------------------------------------
# The real printer, and the docs a run reads
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository as the cwd, with no host declared."""
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("mode", MODES)
def test_the_planners_own_line_is_read_byte_for_byte(repo, capsys, mode):
    """`util improve-steps` itself, on a plan path with a space: the block's
    command is the `[IMPROVE_RERUN]` value exactly, quotes included."""
    plan = repo / "plans" / "my plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Plan\n\n## Step 1\nDo the thing.\n", encoding="utf-8")
    code = improve_steps.main(["plans/my plan.md", "--mode", mode,
                               "--host", "claude-code", "--quick"])
    out = capsys.readouterr().out
    assert code == 0, out
    rerun = [line[len(improve_steps.RERUN_MARKER) + 1:] for line in out.splitlines()
             if line.startswith(improve_steps.RERUN_MARKER + " ")]
    assert len(rerun) == 1 and '"plans/my plan.md"' in rerun[0]
    blocks = report_check.find_blocks(out, continuations=True)
    assert [(b.command, b.continuation) for b in blocks] == [(rerun[0], True)]
    assert report_check.IMPROVE_RERUN.strip() == improve_steps.RERUN_MARKER


def test_no_doc_a_run_reads_starts_a_block():
    """The cores show the planner's output indented: a core loaded through
    `util instructions` in the same call as the planner would otherwise hand
    the check a placeholder as the run's continuation."""
    docs = [SKILL_ROOT / "SKILL.md",
            *sorted((SKILL_ROOT / "instructions").glob("*.md")),
            *sorted((SKILL_ROOT / "references").glob("*.md"))]
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        assert report_check.find_blocks(text, continuations=True) == [], doc.name
