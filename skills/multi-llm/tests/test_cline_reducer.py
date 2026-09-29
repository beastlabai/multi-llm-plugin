"""cline's answer reducer, and the no-``run_result`` branch that reads it.

Plan: plans/ask-answer-recovery-and-false-success.md §4.4, task T4.4 (D4.4),
with D2.5's five steps and D3.1's ``incomplete`` key.

**Why.** With no ``run_result`` in the stream, ``ClineProvider.parse_output``
used to run ``extract_json_from_text`` over the whole event stream, which finds
the event lines themselves. Measured by the preflight survey: all nine failure
fixtures with their ``run_result`` removed came back ``success: True`` with the
``error`` event as ``data``, and the 2026-09-26 incident's cut tail with a tool
call's ``input`` list. The review lanes took that as a review. The branch now
parses only :func:`read_cline_stream`'s text.

The incident tail below is **synthetic**: the same event sequence as the last
20 lines of the incident's log (usage, the end of a reasoning block, two
``run_commands`` tool calls, ``iteration_end`` / ``iteration_start``, then
reasoning deltas and no ``run_result``), with neutral content. The log itself
lives outside this repository.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.cline import (  # noqa: E402
    ClineProvider,
    cline_answer_text,
    read_cline_stream,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cline"

ANSWER = [
    {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},
    {"id": "S002", "title": "Strip the vendor token", "severity": "medium"},
]

#: The two rows whose stream holds an answer; every other row is a failure.
ANSWER_ROWS = ("success", "multi-assistant")
FAILURE_ROWS = (
    "truncated", "ephemeral-only", "error-envelope", "auth-error", "rate-limit",
    "invalid-model", "context-overflow", "zero-body", "server-error",
)


def _stdout(row: str) -> str:
    return (FIXTURES / f"{row}.stdout.txt").read_text(encoding="utf-8")


def _events(stdout: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def _join(events: List[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _inner_type(event: Dict[str, Any]) -> Any:
    inner = event.get("event")
    return inner.get("type") if isinstance(inner, dict) else None


def _without_run_result(row: str) -> str:
    return _join([e for e in _events(_stdout(row)) if e.get("type") != "run_result"])


def _without_run_result_or_done(row: str) -> str:
    return _join([
        e for e in _events(_stdout(row))
        if e.get("type") != "run_result" and _inner_type(e) != "done"
    ])


def _agent(event: Dict[str, Any]) -> Dict[str, Any]:
    return {"ts": "2026-09-26T16:38:32.700Z", "type": "agent_event", "event": event}


def _hook(name: str) -> Dict[str, Any]:
    return {"ts": "2026-09-26T16:38:32.700Z", "type": "hook_event",
            "hookEventName": name, "agentId": "agent_1", "taskId": "conv_1"}


def _tool_call(call_id: str) -> List[Dict[str, Any]]:
    """One ``run_commands`` call as cline streams it: start, updates, end."""
    base = {"contentType": "tool", "toolName": "run_commands", "toolCallId": call_id}
    return [
        _agent({"type": "content_start", **base,
                "input": {"commands": ["sed -n '1,40p' config.yaml"]}}),
        _agent({"type": "content_update", **base,
                "update": {"stream": "stdout", "chunk": "", "commandIndex": 0}}),
        _agent({"type": "content_update", **base,
                "update": {"stream": "stdout", "chunk": "key: [1, 2]\n",
                           "commandIndex": 0}}),
        _hook("tool_result"),
        _agent({"type": "content_end", **base,
                "output": [{"query": "sed -n '1,40p' config.yaml",
                            "result": "key: [1, 2]\n"}]}),
    ]


def incident_tail() -> str:
    """SYNTHETIC: reasoning after two tool calls, cut with no ``run_result``.

    The shape of the incident's last 20 lines (see the module docstring), cut
    mid-way through the last reasoning delta, as a kill leaves it.
    """
    events = [
        _agent({"type": "usage", "inputTokens": 112079, "outputTokens": 1260}),
        _agent({"type": "content_end", "contentType": "reasoning",
                "reasoning": "The config forwards headers; check the endpoint list."}),
        _hook("tool_call"),
        _hook("tool_call"),
        *_tool_call("call_a1"),
        *_tool_call("call_b2"),
        _agent({"type": "iteration_end", "iteration": 28, "hadToolCalls": True,
                "toolCallCount": 2}),
        _agent({"type": "iteration_start", "iteration": 29}),
        _agent({"type": "content_start", "contentType": "reasoning",
                "reasoning": "Interesting", "redacted": False}),
        _agent({"type": "content_start", "contentType": "reasoning",
                "reasoning": ":", "redacted": False}),
        _agent({"type": "content_start", "contentType": "reasoning",
                "reasoning": " usage.py:1", "redacted": False}),
    ]
    torn = json.dumps(_agent({"type": "content_start", "contentType": "reasoning",
                              "reasoning": "22 hashes the", "redacted": False}))
    return _join(events) + torn[: len(torn) // 2]


class TestTheNoRunResultBranch:
    """D4.4: the branch parses the reducer's text, never the raw stream."""

    @pytest.mark.parametrize("row", FAILURE_ROWS)
    def test_a_failure_stream_without_its_run_result_fails(self, row: str) -> None:
        stdout = _without_run_result(row)
        # The pin is live: the stream still carries the event objects the old
        # raw-stream extraction returned as data.
        assert '"type": "agent_event"' in stdout
        result = ClineProvider().parse_output(stdout, "")
        assert result["success"] is False
        assert result["error"] == "No run_result event found in output"
        assert result["data"] is None
        assert result["incomplete"] is True

    @pytest.mark.parametrize("row", ANSWER_ROWS)
    def test_an_answer_stream_keeps_its_answer_through_done(self, row: str) -> None:
        result = ClineProvider().parse_output(_without_run_result(row), "")
        assert result["success"] is True
        assert result["data"] == ANSWER
        # A completed `done` closed the turn: the answer is finished.
        assert "incomplete" not in result

    def test_the_incident_tail_fails(self) -> None:
        result = ClineProvider().parse_output(incident_tail(), "")
        assert result["success"] is False
        assert result["error"] == "No run_result event found in output"
        assert result["incomplete"] is True

    def test_a_cut_text_turn_is_parsed_from_its_text_and_marked_incomplete(self) -> None:
        # `success` with `done` and `run_result` removed: the text event is the
        # answer, but nothing says the turn ended (D3.1).
        result = ClineProvider().parse_output(_without_run_result_or_done("success"), "")
        assert result["success"] is True
        assert result["data"] == ANSWER
        assert result["incomplete"] is True

    @pytest.mark.parametrize("row", ANSWER_ROWS + FAILURE_ROWS)
    def test_a_whole_capture_is_parsed_as_before(self, row: str) -> None:
        # With its run_result the branch is not reached: no `incomplete` key.
        result = ClineProvider().parse_output(_stdout(row), "")
        assert result["success"] is (row in ANSWER_ROWS)
        assert "incomplete" not in result


class TestTheReducer:
    """D2.5's five steps, called directly."""

    @pytest.mark.parametrize("row", FAILURE_ROWS)
    def test_a_finish_reason_error_run_gives_nothing(self, row: str) -> None:
        stdout = _stdout(row)
        assert '"finishReason":"error"' in stdout
        reading = read_cline_stream(stdout)
        assert reading.text == ""
        assert reading.run_result is not None and reading.finished is True

    def test_the_incident_tail_gives_nothing(self) -> None:
        reading = read_cline_stream(incident_tail())
        assert reading.text == ""
        assert reading.run_result is None and reading.finished is False

    def test_success_without_its_run_result_answers_through_done(self) -> None:
        reading = read_cline_stream(_without_run_result("success"))
        assert json.loads(reading.text) == ANSWER
        assert reading.finished is True

    def test_success_without_done_too_answers_through_the_text_event(self) -> None:
        stdout = _without_run_result_or_done("success")
        assert '"done"' not in stdout
        reading = read_cline_stream(stdout)
        assert json.loads(reading.text) == ANSWER
        assert reading.finished is False

    @pytest.mark.parametrize("row", ANSWER_ROWS)
    def test_the_last_run_result_is_the_answer(self, row: str) -> None:
        # multi-assistant carries a decoy first run_result.
        assert json.loads(cline_answer_text(_stdout(row))) == ANSWER

    def test_a_torn_final_line_is_ignored(self) -> None:
        stdout = _without_run_result_or_done("success")
        cut = stdout + '{"ts":"2026-08-19T16:17:18.140Z","type":"agent_ev'
        assert json.loads(cline_answer_text(cut)) == ANSWER

    def test_a_line_separator_inside_the_answer_does_not_tear_its_line(self) -> None:
        # JSON leaves U+2028 unescaped inside a string; `str.splitlines` would
        # break the run_result line there and lose the answer.
        answer = "First line second line"
        stdout = json.dumps(
            {"type": "run_result", "finishReason": "completed", "text": answer},
            ensure_ascii=False,
        )
        assert " " in stdout
        assert cline_answer_text(stdout) == answer
        assert ClineProvider().parse_output(stdout, "")["raw"] == answer

    def test_text_in_an_iteration_that_used_a_tool_is_narration(self) -> None:
        events = [
            _agent({"type": "iteration_start", "iteration": 1}),
            _agent({"type": "content_start", "contentType": "text",
                    "text": 'I will check [{"title": "example"}] first.',
                    "accumulated": 'I will check [{"title": "example"}] first.'}),
            *_tool_call("call_c3"),
        ]
        assert cline_answer_text(_join(events)) == ""

    def test_only_the_last_iteration_counts(self) -> None:
        # A text answer in iteration 1, then iteration 2 starts and runs a
        # tool: the earlier text is dropped, not carried over.
        events = [
            _agent({"type": "iteration_start", "iteration": 1}),
            _agent({"type": "content_end", "contentType": "text",
                    "text": json.dumps(ANSWER)}),
            _agent({"type": "iteration_start", "iteration": 2}),
            *_tool_call("call_d4"),
        ]
        assert cline_answer_text(_join(events)) == ""

    def test_accumulated_outranks_the_delta(self) -> None:
        events = [
            _agent({"type": "iteration_start", "iteration": 1}),
            _agent({"type": "content_start", "contentType": "text",
                    "text": "The answer", "accumulated": "The answer"}),
            _agent({"type": "content_update", "contentType": "text",
                    "text": " is 42.", "accumulated": "The answer is 42."}),
        ]
        assert cline_answer_text(_join(events)) == "The answer is 42."

    def test_a_done_that_did_not_complete_supplies_no_text(self) -> None:
        events = [
            _agent({"type": "iteration_start", "iteration": 1}),
            _agent({"type": "done", "reason": "aborted", "text": "[1]"}),
        ]
        reading = read_cline_stream(_join(events))
        assert reading.text == "" and reading.finished is False

    @pytest.mark.parametrize("stdout", [
        "", "   \n", "not json", "[1, 2, 3]", "null\n42\n\"s\"", None, 7,
        '{"type": "agent_event", "event": ["x"]}',
        '{"type": "agent_event", "event": {"type": "content_start", '
        '"contentType": "text", "text": 5, "accumulated": ["x"]}}',
        '{"type": "run_result", "finishReason": "completed", "text": ["x"]}',
        '{"type": "agent_event", "event": {"type": "done", '
        '"reason": "completed", "text": {"k": 1}}}',
    ])
    def test_never_raises_and_finds_nothing_in_garbage(self, stdout: Any) -> None:
        assert read_cline_stream(stdout).text == ""
