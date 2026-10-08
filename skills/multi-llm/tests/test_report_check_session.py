#!/usr/bin/env python3
"""The Stop hook's report check across a headless phase's several turns, and
every command of the ready section (fix round 26, live run 11 #3).

Claude Code 2.1.293 runs a phase's subagents in the background, so a headless
``claude -p`` phase spans several turns: the typed prompt's, one opened by each
subagent's ``task-notification`` entry, and, in racing 04, one opened by a
``ScheduleWakeup`` prompt (an ``isMeta`` entry). 04 printed the ready section
in a notification turn, its report there copied it, and the run then ended on
the wakeup turn's reply, which carried none of its commands. The check read the
run from the notification only, and read the wakeup reply together with the
report before it, so it passed.

Now the run is everything since the prompt the user typed (a harness entry is
not a prompt), the final message is the reply after the run's last user entry
of any kind, and every command the section fences is held to the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.test_report_check import (  # noqa: E402
    COMMAND,
    OPTIONAL,
    WHY,
    Transcript,
    block_output,
    decision_of,
    report,
    run_check,
)
from utils import report_check  # noqa: E402



@pytest.fixture(autouse=True)
def _short_settle(monkeypatch):
    """A reply the hook input names that the transcript never gets costs this
    long here, not the hook's 2 s (fix round 27, ``settled_turn``)."""
    monkeypatch.setattr(report_check, "SETTLE_BUDGET_S", 0.3, raising=False)


IMPLEMENT = "/multi-llm:multi-llm --implement plans/racing-game.md"
APPLY = "/multi-llm:multi-llm --apply-task-suggestions plans/racing-game.md"
APPLY_WHY = (
    "review-tasks left 5 valid task suggestion groups unapplied. Run "
    "--apply-task-suggestions first to apply them to the task list; going "
    "straight to --implement skips them, and implement then works from the "
    "task list as it stands, without those 5 changes."
)
SECTION = (
    f"```\n{IMPLEMENT}\n```\n\nOptional, before that:\n\n{APPLY_WHY}\n\n"
    f"```\n{APPLY}\n```"
)
#: Racing 04's `util next-command` output, byte for byte up to the result line.
LOOKUP_04 = (
    f"[NEXT_COMMAND] {IMPLEMENT}\n"
    f"[NEXT_COMMAND_OPTIONAL_WHY] {APPLY_WHY}\n"
    f"[NEXT_COMMAND_OPTIONAL] {APPLY}\n"
    "[NEXT_STEP_MARKDOWN]\n"
    f"{SECTION}\n"
    "[NEXT_STEP_MARKDOWN_END]\n"
    'MULTI_LLM_RESULT: {"mode": "util next-command", "phase": null, '
    '"state": "done", "exit": 0, "artifacts": [], "next": null}\n'
)
REPORT_04 = (
    "The task review of `plans/racing-game.md` is finished. 9 suggestions, "
    "grouped into 7.\n\nThe five valid groups are not yet applied to "
    f"`tasks.md`.\n\n{SECTION}\n"
)
WAKEUP_REPLY_04 = (
    "There's nothing left to do for that step. The earlier wakeup fired after "
    "the work was finished.\n\nThe review-tasks phase is complete. The "
    "next-step commands are in my last message."
)


def notification(transcript: Transcript, summary: str = "Validation batch 1") -> Transcript:
    """A background task's completion, as Claude Code 2.1.293 writes it."""
    return transcript._add({
        "type": "user", "promptSource": "system", "turnOrigin": "task_notification",
        "origin": {"kind": "task-notification", "producer": "session-task"},
        "message": {"role": "user", "content": (
            "<task-notification>\n<task-id>a40458d35e4c51dd9</task-id>\n"
            f"<status>completed</status>\n<summary>Agent \"{summary}\" finished"
            "</summary>\n</task-notification>")},
    })


def wakeup(transcript: Transcript) -> Transcript:
    """A `ScheduleWakeup` prompt firing: an `isMeta` entry, no origin."""
    return transcript._add({
        "type": "user", "isMeta": True, "promptSource": "system",
        "turnOrigin": "scheduled",
        "message": {"role": "user", "content": (
            "Continue review-tasks validation: write subagent replies to "
            "output_path files")},
    })


def typed(transcript: Transcript, text: str) -> Transcript:
    """A prompt the user typed in an interactive session."""
    return transcript._add({
        "type": "user", "promptSource": "typed", "origin": {"kind": "human"},
        "message": {"role": "user", "content": text},
    })


def racing_04(transcript: Transcript) -> Transcript:
    """Racing 04 up to its report: launch, four batches back, the lookup."""
    transcript.prompt("/multi-llm:multi-llm --review-tasks plans/racing-game.md --quick --yes")
    transcript.bash("[VALIDATION_PENDING] plans/racing-game/review-tasks/validation_tasks.json\n")
    transcript.say("All four validation batches are still running.")
    notification(transcript)
    transcript.bash("[REPLY_OK] validation-batch-0\n")
    transcript.bash(LOOKUP_04)
    return transcript.say(REPORT_04)


class TestTheRunEndsOnTheReplyAfterAWakeup:

    def test_racing_04_blocks_naming_both_commands(self, tmp_path, capsys):
        path = wakeup(racing_04(Transcript())).say(WAKEUP_REPLY_04).write(
            tmp_path / "t.jsonl")
        reason = decision_of(run_check(capsys, path))["reason"]
        missing = reason.split("Carry the block's other", 1)[0]
        assert "Missing commands:" in missing
        assert f"```\n{IMPLEMENT}\n```" in missing
        assert f"```\n{APPLY}\n```" in missing

    def test_racing_04_as_the_hook_sees_it(self, tmp_path, capsys):
        """At Stop time the transcript has not written the reply yet; the
        hook input's `last_assistant_message` is the wakeup turn's reply."""
        path = wakeup(racing_04(Transcript())).write(tmp_path / "t.jsonl")
        hook = {"transcript_path": str(path), "stop_hook_active": False,
                "last_assistant_message": WAKEUP_REPLY_04}
        reason = decision_of(run_check(capsys, path, hook=hook, tmp_path=tmp_path))["reason"]
        assert APPLY in reason and IMPLEMENT in reason

    def test_a_wakeup_reply_that_prints_the_section_again_passes(self, tmp_path, capsys):
        path = wakeup(racing_04(Transcript())).say(
            "Nothing left to do.\n\n" + SECTION).write(tmp_path / "t.jsonl")
        assert run_check(capsys, path) == ""

    def test_the_report_turn_itself_passes(self, tmp_path, capsys):
        path = racing_04(Transcript()).write(tmp_path / "t.jsonl")
        assert run_check(capsys, path) == ""


class TestATaskNotificationIsNotAPrompt:

    def test_a_late_notification_turn_without_the_commands_blocks(self, tmp_path, capsys):
        transcript = (Transcript().prompt("go").bash(block_output())
                      .say(report(COMMAND)))
        notification(transcript, "status poll").say("The background poll finished.")
        decision_of(run_check(capsys, transcript.write(tmp_path / "t.jsonl")))

    def test_the_block_printed_in_a_notification_turn_is_checked(self, tmp_path, capsys):
        """Racing 03: the typed prompt's turn ended waiting, the lookup ran in
        the notification's turn."""
        transcript = Transcript().prompt("go").bash("[TASK_GENERATION_PENDING] x\n")
        transcript.say("The subagent is still running.")
        notification(transcript, "Generate implementation tasks")
        transcript.bash(block_output(optional=[(WHY, OPTIONAL)]))
        ok = transcript.say(report(COMMAND, OPTIONAL)).write(tmp_path / "ok.jsonl")
        assert run_check(capsys, ok) == ""

    def test_an_interim_stop_before_any_block_passes(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash("[TASK_GENERATION_PENDING] x\n")
                .say("Waiting for the subagent.").write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_a_prompt_the_user_typed_still_starts_a_new_run(self, tmp_path, capsys):
        transcript = (Transcript().prompt("go").bash(block_output())
                      .say(report(COMMAND)))
        typed(transcript, "what does T003 do?").say("T003 builds the track.")
        assert run_check(capsys, transcript.write(tmp_path / "t.jsonl")) == ""


class TestEveryCommandTheSectionFences:

    EXTRA = "/multi-llm:multi-llm --implement 'plans/my plan.md' --resume --human-task T005=skip"

    def _output(self) -> str:
        section = f"```\n{COMMAND}\n```\n\nOr answer the stop:\n\n```\n{self.EXTRA}\n```"
        return (f"[NEXT_COMMAND] {COMMAND}\n[NEXT_STEP_MARKDOWN]\n{section}\n"
                "[NEXT_STEP_MARKDOWN_END]\n"
                'MULTI_LLM_RESULT: {"mode": "util next-command", "exit": 0}\n')

    def test_the_block_holds_the_section_and_its_commands(self):
        block = report_check.find_blocks(self._output())[-1]
        assert block.section is not None and block.section.startswith("```")
        assert block.commands() == [COMMAND, self.EXTRA]

    def test_a_report_without_a_section_command_blocks_naming_it(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(self._output()).say(report(COMMAND))
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        missing = reason.split("Carry the block's other", 1)[0]
        assert f"```\n{self.EXTRA}\n```" in missing

    def test_a_report_with_every_command_passes(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(self._output())
                .say(report(COMMAND, self.EXTRA)).write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_the_reason_asks_for_the_section_as_given(self, tmp_path, capsys):
        path = wakeup(racing_04(Transcript())).say(WAKEUP_REPLY_04).write(
            tmp_path / "t.jsonl")
        reason = decision_of(run_check(capsys, path))["reason"]
        assert report_check.SECTION_REQUEST in reason
        assert reason.endswith(SECTION)
        # The section closes the reason, after every command list.
        assert reason.index(report_check.SECTION_REQUEST) > reason.index("Missing commands:")

    def test_a_section_that_never_closes_is_not_kept(self):
        text = f"[NEXT_COMMAND] {COMMAND}\n[NEXT_STEP_MARKDOWN]\n```\n/x\n```\n"
        block = report_check.find_blocks(text)[-1]
        assert block.section is None and block.commands() == [COMMAND]

    def test_a_block_without_a_section_reads_as_before(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert report_check.SECTION_REQUEST not in reason


class TestTheStopsSectionToo:
    """A stop's section: the resume and each answer, all fenced."""

    RESUME = "/multi-llm:multi-llm --implement plans/check-gate.md --resume"
    ANSWER = f"{RESUME} --human-task T005=skip"
    MANUAL = f"{RESUME} --human-task T010=done"

    def _output(self) -> str:
        section = (
            "To go on with implement:\n\nruns the tasks that can run\n\n"
            f"```\n{self.RESUME}\n```\n\nOr answer the stop with one of these:\n\n"
            f"skip T005\n\n```\n{self.ANSWER}\n```\n\nManual tasks left for you:\n"
            f"- T010 Play it\n\nRecord it done with:\n\n```\n{self.MANUAL}\n```"
        )
        return (
            f"[NEXT_COMMAND_WHY] runs the tasks that can run\n"
            f"[NEXT_COMMAND] {self.RESUME}\n"
            f"[NEXT_COMMAND_ANSWERS_WHY] skip T005\n[NEXT_COMMAND_ANSWERS] {self.ANSWER}\n"
            "[MANUAL_TASK] T010 Play it\n[MANUAL_TASK_NOTE] Record it done with:\n"
            f"[MANUAL_TASK_COMMAND] {self.MANUAL}\n"
            f"[NEXT_STEP_MARKDOWN]\n{section}\n[NEXT_STEP_MARKDOWN_END]\n"
            'MULTI_LLM_RESULT: {"mode": "util next-command", "exit": 0}\n'
        )

    def test_the_answer_and_the_manual_command_are_held_after_a_notification(
        self, tmp_path, capsys,
    ):
        transcript = Transcript().prompt("go").bash("[WRITE_TASKS_FILE] x\n")
        transcript.say("Waiting for T001.")
        notification(transcript, "T001")
        transcript.bash(self._output())
        path = transcript.say(report(self.RESUME) + "\n\nT010 is yours.").write(
            tmp_path / "t.jsonl")
        reason = decision_of(run_check(capsys, path))["reason"]
        missing = reason.split("Carry the block's other", 1)[0]
        assert f"```\n{self.ANSWER}\n```" in missing
        assert f"```\n{self.MANUAL}\n```" in missing
        assert reason.endswith("```\n" + self.MANUAL + "\n```")
