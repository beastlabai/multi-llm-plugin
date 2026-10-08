#!/usr/bin/env python3
"""The Stop hook's block reason says once how the report ends (fix round 28).

Live run 12 (2026-10-08), Claude Code racing 03: the report printed the
implement command bare, the hook blocked, and its reason asked for two
endings: "end it with its `### Next Steps` section, with each command in its
own fenced code block …", then "End it with the ready section the lookup
printed …, printed as given: …, no caption of your own". The resent report
put a `### Next Steps` heading of its own above the section.

A reason with a section now asks for the whole report ending with the
section, in the words SKILL.md uses for it (``modes.READY_SECTION_RULE``, the
relay rule every route carries), and no reason asks for a `### Next Steps`
section of any kind.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import modes  # noqa: E402
from tests.test_next_step_relay_rules import MARKDOWN_RULE  # noqa: E402
from tests.test_report_check import (  # noqa: E402
    COMPLETE_OUTPUT,
    Transcript,
    block_output,
    decision_of,
    run_check,
)
from utils import report_check  # noqa: E402


@pytest.fixture(autouse=True)
def _short_settle(monkeypatch):
    monkeypatch.setattr(report_check, "SETTLE_BUDGET_S", 0.3, raising=False)


IMPLEMENT = "/multi-llm:multi-llm --implement plans/racing-game.md"
REVIEW_TASKS = "/multi-llm:multi-llm --review-tasks plans/racing-game.md"
REVIEW_WHY = (
    "Have the models review the task list before implementing it; skipping "
    "this implements the tasks as they were generated."
)
SECTION_03 = (
    f"```\n{IMPLEMENT}\n```\n\nOptional, before that:\n\n{REVIEW_WHY}\n\n"
    f"```\n{REVIEW_TASKS}\n```"
)
#: Racing 03's `util next-command` output, up to its result line.
LOOKUP_03 = (
    f"[NEXT_COMMAND] {IMPLEMENT}\n"
    f"[NEXT_COMMAND_OPTIONAL_WHY] {REVIEW_WHY}\n"
    f"[NEXT_COMMAND_OPTIONAL] {REVIEW_TASKS}\n"
    f"[NEXT_STEP_MARKDOWN]\n{SECTION_03}\n[NEXT_STEP_MARKDOWN_END]\n"
    'MULTI_LLM_RESULT: {"mode": "util next-command", "phase": null, '
    '"state": "done", "exit": 0, "artifacts": [], "next": null}\n'
)
#: Racing 03's first report: the implement command printed bare.
REPORT_03 = (
    "Task generation produced 10 tasks, written to "
    "`plans/racing-game/tasks/tasks.md`.\n\n"
    f"{IMPLEMENT}\n\nOptional, before that:\n\n{REVIEW_WHY}\n\n"
    f"```\n{REVIEW_TASKS}\n```"
)

MANUAL_RESUME = "/multi-llm:multi-llm --implement plans/racing-game.md --resume --human-task T010=done"
MANUAL_NOTE = (
    "No later phase waits for this manual task. Once you have done it by "
    "hand, record it done with this command:"
)
SECTION_COMPLETE = (
    "Workflow complete.\n\nManual task left for you:\n"
    "- T010 Manual playthrough against acceptance criteria\n\n"
    f"{MANUAL_NOTE}\n\n```\n{MANUAL_RESUME}\n```"
)
LOOKUP_COMPLETE = (
    "[WORKFLOW_COMPLETE]\n"
    "[MANUAL_TASK] T010 Manual playthrough against acceptance criteria\n"
    f"[MANUAL_TASK_NOTE] {MANUAL_NOTE}\n"
    f"[MANUAL_TASK_COMMAND] {MANUAL_RESUME}\n"
    f"[NEXT_STEP_MARKDOWN]\n{SECTION_COMPLETE}\n[NEXT_STEP_MARKDOWN_END]\n"
    'MULTI_LLM_RESULT: {"mode": "util next-command", "exit": 0}\n'
)


def _reason(tmp_path, capsys, output: str, message: str) -> str:
    path = (Transcript().prompt("go").bash(output).say(message)
            .write(tmp_path / "t.jsonl"))
    return decision_of(run_check(capsys, path))["reason"]


def _endings(reason: str) -> int:
    """How many times the reason says what the report ends with."""
    return len(re.findall(r"\bend it with\b", reason, re.IGNORECASE))


class TestRacing03:

    def test_the_reason_asks_for_one_ending_and_no_heading(self, tmp_path, capsys):
        reason = _reason(tmp_path, capsys, LOOKUP_03, REPORT_03)
        assert "### Next Steps" not in reason
        assert _endings(reason) == 1, reason
        assert f"Missing command:\n\n```\n{IMPLEMENT}\n```" in reason
        assert reason.endswith(f"{report_check.SECTION_REQUEST}\n\n{SECTION_03}")

    def test_the_ending_is_the_section_in_skill_md_words(self, tmp_path, capsys):
        reason = _reason(tmp_path, capsys, LOOKUP_03, REPORT_03)
        flat = " ".join(reason.split())
        assert MARKDOWN_RULE in flat
        assert modes.READY_SECTION_RULE in flat
        # Said once: the section carries the commands, the why and the
        # optional step's lead line, so the reason describes none of them.
        assert flat.count("no caption of your own") == 1
        assert "Carry the block's other" not in reason
        assert '"Optional, before that:"' not in reason

    def test_the_resent_report_of_racing_03_passes(self, tmp_path, capsys):
        resent = REPORT_03.replace(f"{IMPLEMENT}\n\nOptional", f"```\n{IMPLEMENT}\n```\n\nOptional")
        path = (Transcript().prompt("go").bash(LOOKUP_03).say(resent)
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""


class TestEveryReasonEndsOneWay:

    @pytest.mark.parametrize("output,message", [
        (LOOKUP_03, "Done."),
        (LOOKUP_COMPLETE, "Done."),
        (LOOKUP_COMPLETE, "Workflow complete.\n\nT010 is yours."),
        (LOOKUP_COMPLETE, f"T010 is yours.\n\n```\n{MANUAL_RESUME}\n```"),
    ])
    def test_with_a_section(self, tmp_path, capsys, output, message):
        reason = _reason(tmp_path, capsys, output, message)
        assert "### Next Steps" not in reason
        assert _endings(reason) == 1, reason
        assert report_check.SECTION_REQUEST in reason

    def test_a_complete_section_names_what_the_report_lacks(self, tmp_path, capsys):
        reason = _reason(tmp_path, capsys, LOOKUP_COMPLETE, "Workflow complete.")
        assert "does not name the task T010" in reason
        assert reason.endswith(SECTION_COMPLETE)

    @pytest.mark.parametrize("output,message", [
        (block_output(), "Done."),
        (COMPLETE_OUTPUT, "Done."),
        ("[WORKFLOW_COMPLETE]\n[MANUAL_TASK] T011 Play it by hand\n"
         'MULTI_LLM_RESULT: {"mode": "util next-command", "exit": 0}\n', "Done."),
        (block_output() + "[MANUAL_TASK] T011 Play it by hand\n",
         "```\n/multi-llm:multi-llm --generate-tasks 'plans/my plan.md'\n```"),
    ])
    def test_without_a_section(self, tmp_path, capsys, output, message):
        reason = _reason(tmp_path, capsys, output, message)
        assert "### Next Steps" not in reason
        assert _endings(reason) == 1, reason
        assert report_check.SECTION_REQUEST not in reason


class TestTheRuleIsTheRoutesOwn:

    def test_the_literal_is_modes_constant(self):
        assert report_check.READY_SECTION_RULE == modes.READY_SECTION_RULE

    def test_the_request_carries_the_markdown_rule(self):
        assert MARKDOWN_RULE in " ".join(report_check.SECTION_REQUEST.split())
