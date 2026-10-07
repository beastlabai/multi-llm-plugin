#!/usr/bin/env python3
"""``MULTI_LLM_RESULT.user_next_step`` (plans/next-phase-command.md T4).

The key is the workflow's next phase, as a command for the *user* to type. It
is optional and present only when set, so a result line on which it is not set
stays byte-for-byte what it was. A completed workflow is an answer, not the
lack of one: ``{"phase": null, "command": null, "complete": true}`` rides the
line, and only ``None`` leaves it absent (§2.1, the outcome table). The payload
also guards it a second time: never beside a non-null ``next`` (a follow-up the
caller owes, which a tool route auto-follows), never on exit 1 or 3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import multi_llm  # noqa: E402
from utils import cli_common  # noqa: E402
from utils.cli_common import (  # noqa: E402
    EXIT_ERROR,
    EXIT_NEEDS_HUMAN,
    EXIT_NOTHING_TO_DO,
    EXIT_OK,
    RESULT_MARKER,
    ResultContext,
)

#: The six fixed keys, in the order every result line has carried them.
FIXED_KEYS = ["mode", "phase", "state", "exit", "artifacts", "next"]

COMPLETE = {"phase": None, "command": None, "complete": True}

ANSWER = {
    "phase": "generate-tasks",
    "command": "/multi-llm:multi-llm --generate-tasks 'plans/my plan.md'",
    "optional_before": [
        {
            "phase": "review-plan",
            "command": "/multi-llm:multi-llm --review-plan 'plans/my plan.md'",
            "why": "Have the models review the plan again; skipping this keeps the plan as it stands.",
        }
    ],
    "complete": False,
}

ALL_EXITS = [EXIT_OK, EXIT_ERROR, EXIT_NOTHING_TO_DO, EXIT_NEEDS_HUMAN]


def _context(**fields) -> ResultContext:
    ctx = ResultContext(mode="generate-tasks", phase="generate-tasks")
    ctx.add_artifact("plans/my plan/tasks/tasks.md")
    for name, value in fields.items():
        setattr(ctx, name, value)
    return ctx


def _pre_change_payload(ctx: ResultContext, exit_code: int) -> dict:
    """The line as it was before the key existed: the fixed keys only."""
    return {
        "mode": ctx.mode,
        "phase": ctx.phase,
        "state": ctx.state or cli_common._STATE_BY_EXIT.get(exit_code, "error"),
        "exit": exit_code,
        "artifacts": list(ctx.artifacts),
        "next": ctx.next,
    }


class TestTheCompletionCase:
    """§2.1: a complete workflow is an answer, never the lack of one."""

    def test_a_completed_answer_rides_the_line_with_command_null(self):
        ctx = _context()
        ctx.set_user_next_step(dict(COMPLETE))
        payload = ctx.payload(EXIT_OK)
        assert "user_next_step" in payload
        assert payload["user_next_step"] == {
            "phase": None, "command": None, "complete": True,
        }

    def test_a_none_answer_leaves_the_key_absent(self):
        ctx = _context()
        ctx.set_user_next_step(None)
        assert "user_next_step" not in ctx.payload(EXIT_OK)

    def test_the_two_payloads_differ_so_a_consumer_can_tell_them_apart(self):
        complete, none = _context(), _context()
        complete.set_user_next_step(dict(COMPLETE))
        none.set_user_next_step(None)
        assert complete.payload(EXIT_OK) != none.payload(EXIT_OK)
        line = json.dumps(complete.payload(EXIT_OK))
        assert '"user_next_step": {"phase": null, "command": null, "complete": true}' in line

    def test_the_completed_answer_rides_exit_2_as_well(self):
        """A re-run of a finished phase exits 2 and still carries the key (§2.2)."""
        ctx = _context()
        ctx.set_user_next_step(dict(COMPLETE))
        assert ctx.payload(EXIT_NOTHING_TO_DO)["user_next_step"] == COMPLETE


class TestPresentOnlyWhenSet:

    @pytest.mark.parametrize("exit_code", ALL_EXITS)
    def test_absent_unless_set(self, exit_code):
        assert "user_next_step" not in _context().payload(exit_code)

    def test_present_when_set(self):
        ctx = _context()
        ctx.set_user_next_step(ANSWER)
        assert ctx.payload(EXIT_OK)["user_next_step"] == ANSWER

    def test_none_clears_a_value_set_earlier(self):
        ctx = _context()
        ctx.set_user_next_step(ANSWER)
        ctx.set_user_next_step(None)
        assert "user_next_step" not in ctx.payload(EXIT_OK)

    def test_it_is_stored_as_a_copy(self):
        """A caller that keeps mutating its dict cannot move the line."""
        answer = json.loads(json.dumps(ANSWER))
        ctx = _context()
        ctx.set_user_next_step(answer)
        answer["command"] = "something else"
        answer["optional_before"][0]["command"] = "something else"
        assert ctx.payload(EXIT_OK)["user_next_step"] == ANSWER

    def test_the_payload_hands_out_a_copy(self):
        ctx = _context()
        ctx.set_user_next_step(ANSWER)
        ctx.payload(EXIT_OK)["user_next_step"]["optional_before"].clear()
        assert ctx.payload(EXIT_OK)["user_next_step"] == ANSWER


class TestNeverBesideNextOrOnAFailure:

    def test_never_alongside_a_non_null_next(self):
        """A mechanical continuation (`next` owed) never carries the key."""
        ctx = _context()
        ctx.set_user_next_step(ANSWER)
        ctx.set_next("uv run --frozen --no-dev --project /s -- python /s/multi_llm.py "
                     "review-plan plans/x.md --reaggregate")
        payload = ctx.payload(EXIT_OK)
        assert payload["next"]
        assert "user_next_step" not in payload

    def test_the_completed_answer_is_guarded_the_same_way(self):
        ctx = _context()
        ctx.set_user_next_step(dict(COMPLETE))
        ctx.set_next("uv run x")
        assert "user_next_step" not in ctx.payload(EXIT_OK)

    def test_a_null_next_lets_it_through(self):
        ctx = _context()
        ctx.set_user_next_step(ANSWER)
        ctx.set_next(None)
        assert ctx.payload(EXIT_OK)["user_next_step"] == ANSWER

    @pytest.mark.parametrize("exit_code", [EXIT_ERROR, EXIT_NEEDS_HUMAN])
    def test_never_on_exit_1_or_3(self, exit_code):
        for value in (ANSWER, COMPLETE):
            ctx = _context()
            ctx.set_user_next_step(value)
            assert "user_next_step" not in ctx.payload(exit_code)

    @pytest.mark.parametrize("exit_code", [EXIT_OK, EXIT_NOTHING_TO_DO])
    def test_present_on_exit_0_and_2(self, exit_code):
        ctx = _context()
        ctx.set_user_next_step(ANSWER)
        assert ctx.payload(exit_code)["user_next_step"] == ANSWER


class TestAnUnsetLineIsByteForByteUnchanged:
    """A run the key is not set on (a non-phase mode, a failed run, an excluded
    invocation, a run the gate stops) emits exactly the pre-change line."""

    @pytest.mark.parametrize("exit_code", ALL_EXITS)
    def test_the_fixed_keys_in_the_fixed_order(self, exit_code):
        ctx = _context()
        payload = ctx.payload(exit_code)
        assert list(payload) == FIXED_KEYS
        assert json.dumps(payload) == json.dumps(_pre_change_payload(ctx, exit_code))

    def test_with_a_next_owed(self):
        ctx = _context()
        ctx.set_next("uv run x --reaggregate")
        assert json.dumps(ctx.payload(EXIT_OK)) == json.dumps(
            _pre_change_payload(ctx, EXIT_OK)
        )

    @pytest.mark.parametrize("exit_code", [EXIT_ERROR, EXIT_NEEDS_HUMAN])
    def test_a_guarded_out_key_leaves_no_trace(self, exit_code):
        ctx = _context()
        ctx.set_user_next_step(ANSWER)
        assert json.dumps(ctx.payload(exit_code)) == json.dumps(
            _pre_change_payload(ctx, exit_code)
        )

    def test_other_optional_keys_keep_their_place(self):
        """Setting the key appends it; every other key stays where it was."""
        without = _context(notice="run the next phase")
        with_key = _context(notice="run the next phase")
        with_key.set_user_next_step(ANSWER)
        before = without.payload(EXIT_OK)
        after = with_key.payload(EXIT_OK)
        assert list(after) == list(before) + ["user_next_step"]
        assert {k: after[k] for k in before} == before

    def test_the_emitted_line(self):
        import io

        ctx = _context()
        stream = io.StringIO()
        ctx.emit(EXIT_OK, stream=stream)
        assert stream.getvalue() == (
            f"{RESULT_MARKER} {json.dumps(_pre_change_payload(ctx, EXIT_OK))}\n"
        )


class TestTheRecordingHelper:

    def test_a_no_op_with_no_active_context(self):
        cli_common.deactivate_result()
        cli_common.record_user_next_step(ANSWER)  # must not raise
        assert cli_common.active_result() is None

    def test_it_reaches_the_active_context(self):
        ctx = cli_common.activate_result(_context())
        try:
            cli_common.record_user_next_step(dict(COMPLETE))
            assert ctx.payload(EXIT_OK)["user_next_step"] == COMPLETE
            cli_common.record_user_next_step(None)
            assert "user_next_step" not in ctx.payload(EXIT_OK)
        finally:
            cli_common.deactivate_result()


class TestTheDispatcherSaysSo:
    """The two places the dispatcher spells out the contract name the key."""

    def test_render_help_names_user_next_step(self):
        help_text = multi_llm.render_help()
        assert "user_next_step" in help_text
        clause = help_text.split("user_next_step", 1)[1]
        assert "user to" in clause and "never a" in clause and "`next`" in clause
        # The fixed-key line is untouched.
        assert (
            '  MULTI_LLM_RESULT: {"mode","phase","state","exit","artifacts":[...],"next":...}'
            in help_text
        )

    def test_the_module_docstring_names_user_next_step(self):
        doc = multi_llm.__doc__ or ""
        assert "user_next_step" in doc
        assert "never a follow-up for the caller" in doc
