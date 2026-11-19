"""An executor-leg writer that goes quiet is stopped and read, and its tool
calls are counted (docs/development/delegation.md#an-idle-writer,
docs/development/delegation.md#tool-use-counts-on-the-executor-leg).

The incident: a ``codex exec --json`` writer edited the plan, printed its
closing ``agent_message`` in prose and then printed nothing, with no
``turn.completed`` and no exit, until it was killed by hand; the run recovered
through the report re-ask. These rows hold the pieces of that recovery apart:
the capture's idle clock, codex's in-flight tracker, the stop through the
executor's real kill path against stand-in CLIs, and the scheduler's three
outcomes against the mock codex. No real codex runs here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from tests.test_write_executor import FakeState, _patch_edit, env  # noqa: E402,F401
from utils import attempt_classifier, dispatch_spawn  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.process_capture import (  # noqa: E402
    IdleStopped,
    PipeCapture,
    StreamStopped,
    provider_idle_watch,
)
from utils.provider_registry import ConfigError, _PROVIDERS  # noqa: E402
from utils.providers.base import PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE  # noqa: E402
from utils.providers.codex import (  # noqa: E402
    CODEX_TOOL_ITEM_TYPES,
    IN_FLIGHT_LINE_MAX,
    ToolsInFlight,
)
from utils.retry_ladder import STATUS_OK, STATUS_RUNNER_ERROR, STATUS_TIMEOUT  # noqa: E402

CODEX_FIXTURES = SKILL_DIR / "tests" / "fixtures" / "codex"


def _codex_streams():
    found = [p for p in CODEX_FIXTURES.rglob("*.jsonl")
             if "rollout" not in p.name and "requests" not in p.name]
    found += list((CODEX_FIXTURES / "rows").glob("*.stdout.txt"))
    return sorted(found)


CODEX_STREAMS = _codex_streams()


def _event(kind, item_id=None, item_type=None, **fields):
    if item_id is None:
        return json.dumps({"type": kind, **fields}) + "\n"
    return json.dumps({"type": kind, "item": {"id": item_id, "type": item_type, **fields}}) + "\n"


# --------------------------------------------------------------------------- #
# codex's in-flight tracker
# --------------------------------------------------------------------------- #

class TestToolsInFlight:

    def test_a_tool_is_in_flight_from_its_start_to_its_completion(self):
        tracker = ToolsInFlight()
        assert tracker(_event("thread.started", thread_id="t").encode()) is False
        assert tracker(_event("item.started", "item_1", "command_execution",
                              command="pytest -q").encode()) is True
        assert tracker(_event("item.completed", "item_1", "command_execution",
                              exit_code=0).encode()) is False

    @pytest.mark.parametrize("item_type", ["agent_message", "reasoning", "todo_list", "error"])
    def test_the_items_that_run_no_tool_never_open_one(self, item_type):
        assert ToolsInFlight()(_event("item.started", "x", item_type).encode()) is False

    def test_an_item_type_it_does_not_know_counts_as_in_flight(self):
        """The safe direction: a later codex's new tool keeps the clock off."""
        tracker = ToolsInFlight()
        assert tracker(_event("item.started", "x", "image_generation").encode()) is True
        assert tracker(_event("item.completed", "x", "image_generation").encode()) is False

    @pytest.mark.parametrize("ending", ["turn.completed", "turn.failed"])
    def test_the_end_of_the_turn_closes_every_call(self, ending):
        tracker = ToolsInFlight()
        tracker(_event("item.started", "a", "command_execution").encode())
        tracker(_event("item.started", "b", "command_execution").encode())
        assert tracker(_event(ending).encode()) is False

    def test_an_unreadable_line_changes_nothing(self):
        tracker = ToolsInFlight()
        assert tracker(_event("item.started", "a", "file_change").encode()) is True
        assert tracker(b"not json at all\n") is True
        assert tracker(b'{"type":"item.completed","item":{"id"\n') is True

    def test_lines_cut_anywhere_read_as_whole_lines(self):
        data = (_event("item.started", "a", "command_execution")
                + _event("item.completed", "a", "command_execution")
                + _event("item.started", "b", "mcp_tool_call")).encode()
        for size in (1, 2, 7, 64):
            tracker = ToolsInFlight()
            answers = [tracker(data[i:i + size]) for i in range(0, len(data), size)]
            assert answers[-1] is True, size

    @pytest.mark.parametrize("compact", [True, False], ids=["as-codex-prints-it", "spaced"])
    def test_an_over_long_completion_is_read_from_its_head_and_not_kept(self, compact):
        tracker = ToolsInFlight()
        tracker(_event("item.started", "item_9", "command_execution").encode())
        event = {"type": "item.completed", "item": {
            "id": "item_9", "type": "command_execution",
            "aggregated_output": "x" * (IN_FLIGHT_LINE_MAX * 3)}}
        huge = (json.dumps(event, separators=(",", ":")) if compact
                else json.dumps(event)).encode() + b"\n"
        for start in range(0, len(huge) - 1, 4096):
            tracker(huge[start:start + 4096])
            assert len(tracker._partial) <= IN_FLIGHT_LINE_MAX
        assert tracker(huge[-1:]) is False

    def test_exec_sessions_left_open_keep_it_busy_until_the_turn_ends(self):
        """0.156.1's two sessions still running at turn end (SCHEMA.md)."""
        lines = (CODEX_FIXTURES / "0.156.1" / "writes_patches_sessions.jsonl").read_bytes()
        body, _, last = lines.rstrip(b"\n").rpartition(b"\n")
        assert b"turn.completed" in last
        tracker = ToolsInFlight()
        assert tracker(body + b"\n") is True
        assert tracker(last + b"\n") is False

    @pytest.mark.parametrize("path", CODEX_STREAMS, ids=lambda p: str(p.relative_to(CODEX_FIXTURES)))
    def test_every_committed_stream_that_ends_its_turn_ends_idle(self, path):
        data = path.read_bytes()
        whole = ToolsInFlight()(data if data.endswith(b"\n") else data + b"\n")
        if b'"turn.completed"' in data or b'"turn.failed"' in data:
            assert whole is False
        chunked = ToolsInFlight()
        answer = False
        for start in range(0, len(data), 333):
            answer = chunked(data[start:start + 333])
        assert answer == ToolsInFlight()(data)

    def test_only_codex_declares_a_tracker(self):
        declaring = sorted(name for name, adapter in _PROVIDERS.items()
                           if provider_idle_watch(adapter) is not None)
        assert declaring == ["codex"]

    def test_each_dispatch_gets_a_fresh_tracker(self):
        adapter = _PROVIDERS["codex"]
        assert adapter.idle_watch() is not adapter.idle_watch()


# --------------------------------------------------------------------------- #
# After an idle stop: had the runner finished speaking?
# --------------------------------------------------------------------------- #

#: The incident's stream, event for event (its `failed/` capture): the opening
#: preamble, a read, the edit, then a prose message and nothing more.
_INCIDENT = (_event("thread.started", thread_id="t") + _event("turn.started")
             + _event("item.completed", "item_0", "agent_message",
                      text="I'll read the current plan first, then update it.")
             + _event("item.started", "item_1", "command_execution")
             + _event("item.completed", "item_1", "command_execution")
             + _event("item.started", "item_2", "file_change")
             + _event("item.completed", "item_2", "file_change")
             + _event("item.completed", "item_3", "agent_message",
                      text="The plan now backfills before switching reads."))


#: A report re-ask's whole stream, its turn never ended: no tool call, one
#: message holding the restated report.
_REASK_REPLY = (_event("thread.started", thread_id="t") + _event("turn.started")
                + _event("item.completed", "item_0", "agent_message",
                         text='{"status": "applied", "ids": ["s0"]}'))


def _turn_ends(path):
    return b'"turn.completed"' in path.read_bytes()


class TestTurnFinished:

    def _finished(self, text):
        return _PROVIDERS["codex"].turn_finished(text)

    def test_the_incidents_closing_message_after_its_edit_is_finished(self):
        assert self._finished(_INCIDENT) is True

    def test_an_ended_turn_is_finished(self):
        assert self._finished(_INCIDENT + _event("turn.completed", usage={})) is True

    @pytest.mark.parametrize(
        "path", [p for p in CODEX_STREAMS if _turn_ends(p)],
        ids=lambda p: str(p.relative_to(CODEX_FIXTURES)))
    def test_every_committed_stream_that_ends_its_turn_reads_finished(self, path):
        """Cut before its `turn.completed`, a committed stream still reads
        finished when it ran a tool before its last message, and as a stall
        after a preamble when it ran none."""
        text = path.read_text(encoding="utf-8")
        assert self._finished(text) is True
        lines = text.rstrip("\n").split("\n")
        cut = "\n".join(line for line in lines if '"turn.completed"' not in line)
        ran_a_tool = any(
            (json.loads(line).get("item") or {}).get("type")
            not in (None, "agent_message", "reasoning", "todo_list", "error")
            for line in lines if line.startswith("{")
        )
        assert self._finished(cut) is ran_a_tool

    @pytest.mark.parametrize("tail", [
        # The last word a tool call: an edit just ended.
        _event("item.started", "t", "file_change")
        + _event("item.completed", "t", "file_change"),
        # Still thinking after the message.
        _event("item.completed", "r", "reasoning", text="Next, the tests."),
        _event("item.started", "r", "reasoning"),
        # The model stream reconnecting, or the turn failing.
        _event("error", message="Reconnecting... 1/5"),
        _event("turn.failed", error={"message": "stream disconnected"}),
        # A plan update after the message.
        _event("item.updated", "p", "todo_list"),
    ])
    def test_anything_after_the_closing_message_is_mid_turn(self, tail):
        assert self._finished(_INCIDENT + tail) is False

    def test_a_line_that_is_no_codex_event_changes_nothing(self):
        assert self._finished(_INCIDENT + "not json\n" + '{"other": 1}\n' + "[1]\n") is True

    @pytest.mark.parametrize("text", [
        "",
        _event("thread.started", thread_id="t") + _event("turn.started"),
        # Only the opening preamble.
        _event("turn.started")
        + _event("item.completed", "p", "agent_message", text="I'll read the plan first."),
        # The preamble and a tool call, nothing after it.
        _event("item.completed", "p", "agent_message", text="I'll read the plan first.")
        + _event("item.started", "t", "command_execution")
        + _event("item.completed", "t", "command_execution"),
        # Messages, but no tool call before them: nothing after the preamble.
        _event("item.completed", "a", "agent_message", text="one")
        + _event("item.completed", "b", "agent_message", text="two"),
    ])
    def test_a_stall_before_any_reply_after_a_tool_is_mid_turn(self, text):
        assert self._finished(text) is False

    def test_the_committed_stream_cut_mid_tool_is_mid_turn(self):
        text = (CODEX_FIXTURES / "0.156.1" / "run_0c477c4d_partial.jsonl").read_text()
        assert self._finished(text) is False

    def test_commentary_between_two_tool_calls_reads_finished(self):
        """The limit (docs/development/delegation.md#an-idle-writer): a
        message between two tool calls, then a stall, cannot be told from a
        closing message."""
        text = (_event("item.started", "t1", "file_change")
                + _event("item.completed", "t1", "file_change")
                + _event("item.completed", "m", "agent_message",
                         text="That section is done; now the tests."))
        assert self._finished(text) is True

    def test_an_adapter_that_cannot_tell_reads_mid_turn(self):
        assert _PROVIDERS["claude-code"].turn_finished(_INCIDENT) is False
        assert _PROVIDERS["claude-code"].turn_finished(_REASK_REPLY, reask=True) is False

    def test_a_report_re_asks_one_message_is_its_reply(self):
        """Fable review 2 of 2026-11-19, N4: a report re-ask runs no tool, so
        its report is a lone message; read as the opening preamble it was, a
        re-ask that answered and then hung waited out the idle timeout and
        was a timeout. On a re-ask a finished message is the reply."""
        assert self._finished(_REASK_REPLY) is False
        assert _PROVIDERS["codex"].turn_finished(_REASK_REPLY, reask=True) is True
        assert _PROVIDERS["codex"].turn_finished(_INCIDENT, reask=True) is True

    @pytest.mark.parametrize("text", [
        "",
        _event("thread.started", thread_id="t") + _event("turn.started"),
        # A read before any reply, and nothing after it.
        _event("turn.started") + _event("item.started", "t", "command_execution")
        + _event("item.completed", "t", "command_execution"),
        _event("turn.started") + _event("item.started", "t", "command_execution"),
        # Something after the reply: still thinking, reconnecting, failing.
        _REASK_REPLY + _event("item.completed", "r", "reasoning", text="Next."),
        _REASK_REPLY + _event("error", message="Reconnecting... 1/5"),
        _REASK_REPLY + _event("turn.failed", error={"message": "stream disconnected"}),
        _REASK_REPLY + _event("item.updated", "p", "todo_list"),
        # A message still being written.
        _event("turn.started") + _event("item.started", "m", "agent_message"),
    ], ids=["empty", "no-item", "tool-last", "tool-open", "reasoning-after",
            "error-after", "turn-failed-after", "todo-after", "message-open"])
    def test_a_re_ask_stopped_before_its_reply_ended_is_still_mid_turn(self, text):
        assert _PROVIDERS["codex"].turn_finished(text, reask=True) is False

    def test_the_spawn_reads_it_fail_closed(self):
        codex = _PROVIDERS["codex"]
        read = dispatch_spawn._idle_turn_finished
        assert read(codex, dispatch_spawn._Capture(stdout=_INCIDENT)) is True
        capped = dispatch_spawn._Capture(stdout=_INCIDENT, capped_streams=frozenset({"stdout"}))
        assert read(codex, capped) is False
        lone = dispatch_spawn._Capture(stdout=_REASK_REPLY)
        assert read(codex, lone) is False
        assert read(codex, lone, reask=True) is True
        capped_reask = dispatch_spawn._Capture(
            stdout=_REASK_REPLY, capped_streams=frozenset({"stdout"}))
        assert read(codex, capped_reask, reask=True) is False

        class Raises:
            def turn_finished(self, stdout, *, reask=False):
                raise RuntimeError("boom")

        class Truthy:
            def turn_finished(self, stdout, *, reask=False):
                return "yes"

        class TakesNoReask:
            def turn_finished(self, stdout):
                return True

        for adapter in (Raises(), Truthy(), TakesNoReask(), object()):
            for reask in (False, True):
                assert read(adapter, dispatch_spawn._Capture(stdout=_INCIDENT),
                            reask=reask) is False


# --------------------------------------------------------------------------- #
# codex's tool-use count
# --------------------------------------------------------------------------- #

class TestCodexToolUses:

    def _usage(self, text):
        return _PROVIDERS["codex"].parse_usage(text, "")

    @pytest.mark.parametrize("name,count", [
        ("0.163.0/writes_patches_sessions.jsonl", 10),
        ("0.156.1/code_mode_writes.jsonl", 3),
        ("0.156.1/mcp_tool_call.jsonl", 1),
        ("0.156.1/collab_spawn_agent.jsonl", 2),
        ("0.156.1/web_search_hosted.jsonl", 3),
        ("gpt5_sol_single_message.jsonl", 0),
    ])
    def test_each_tool_item_counts_once(self, name, count):
        usage = self._usage((CODEX_FIXTURES / name).read_text(encoding="utf-8"))
        assert usage["tool_uses"] == count
        assert "input_tokens" in usage

    def test_the_count_is_the_distinct_tool_items(self):
        text = (CODEX_FIXTURES / "0.163.0" / "writes_patches_sessions.jsonl").read_text()
        ids = set()
        for line in text.splitlines():
            event = json.loads(line)
            item = event.get("item") or {}
            if item.get("type") in CODEX_TOOL_ITEM_TYPES:
                ids.add(item["id"])
        assert self._usage(text)["tool_uses"] == len(ids)

    def test_a_stream_a_stop_cut_short_keeps_its_count_without_tokens(self):
        text = (_event("thread.started", thread_id="t")
                + _event("item.started", "a", "file_change")
                + _event("item.completed", "a", "file_change")
                + _event("item.started", "b", "command_execution"))
        assert self._usage(text) == {"tool_uses": 2}

    def test_a_finished_turn_with_no_tool_reports_zero(self):
        text = (_event("turn.started")
                + _event("item.completed", "m", "agent_message", text="{}")
                + _event("turn.completed", usage={"input_tokens": 3, "output_tokens": 4}))
        assert self._usage(text)["tool_uses"] == 0

    def test_a_stream_with_neither_reports_nothing(self):
        assert self._usage(_event("thread.started", thread_id="t")) is None
        assert self._usage("") is None

    @pytest.mark.parametrize("stream", [
        # Cut short after one tool item: no `turn.completed`, so no tokens.
        _event("item.started", "a", "file_change") + _event("item.completed", "a", "file_change"),
        # A finished turn whose `turn.completed` carries no usage block.
        _event("item.completed", "m", "agent_message", text="{}") + _event("turn.completed"),
    ], ids=["cut-short", "no-usage-block"])
    def test_a_tool_count_alone_is_kept_but_is_no_reported_usage(self, stream):
        """Fable review 2026-11-19, finding 12: a tool count is no token figure, so the
        task still counts under `tasks_without_usage`, its count kept."""
        usage = self._usage(stream)
        assert set(usage) == {"tool_uses"}
        outcome = executor.TaskOutcome("t", "apply_suggestions", "codex:x", executor.STATUS_OK)
        executor._accumulate_usage(outcome, usage)
        assert outcome.usage == usage and outcome.usage_reported is False
        row = outcome.to_dict()
        assert row["usage"] == usage and row["usage_reported"] is False
        totals = executor.BatchReport(outcomes=[outcome]).usage_totals()
        assert totals["tasks_without_usage"] == 1
        assert totals["tool_uses"] == usage["tool_uses"]

    @pytest.mark.parametrize("usage", [
        {"tool_uses": 3}, {"tool_uses": 0}, {"served_model": "m"}, {"cost_usd": 0.5},
    ])
    def test_only_a_token_figure_makes_usage_reported(self, usage):
        outcome = executor.TaskOutcome("t", "apply_suggestions", "codex:x", executor.STATUS_OK)
        executor._accumulate_usage(outcome, usage)
        assert outcome.usage_reported is False
        executor._accumulate_usage(outcome, {"input_tokens": 5, "output_tokens": 0,
                                             "total_tokens": 5, "tool_uses": 1})
        assert outcome.usage_reported is True
        assert outcome.usage["input_tokens"] == 5
        assert outcome.usage.get("tool_uses") == usage.get("tool_uses", 0) + 1
        row = outcome.to_dict()
        assert "usage_reported" not in row and row["usage"] == outcome.usage
        totals = executor.BatchReport(outcomes=[outcome]).usage_totals()
        assert totals["tasks_without_usage"] == 0 and totals["input_tokens"] == 5

    def test_a_task_with_no_usage_at_all_is_unchanged(self):
        outcome = executor.TaskOutcome("t", "apply_suggestions", "gemini:x", executor.STATUS_OK)
        executor._accumulate_usage(outcome, None)
        row = outcome.to_dict()
        assert "usage" not in row and row["usage_reported"] is False

    def test_plan_updates_are_not_counted(self):
        text = (_event("item.started", "p", "todo_list")
                + _event("item.updated", "p", "todo_list")
                + _event("item.completed", "p", "todo_list")
                + _event("turn.completed", usage={"input_tokens": 1, "output_tokens": 1}))
        assert self._usage(text)["tool_uses"] == 0


# --------------------------------------------------------------------------- #
# The capture's idle clock, against real child processes
# --------------------------------------------------------------------------- #

def _popen(script: str):
    return subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _stop(proc):
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait(timeout=10)


_PRINT_THEN_HANG = "import sys,time; print('hello', flush=True); time.sleep(30)"

#: The clock starts at the spawn, so every window here is longer than a child's
#: start (a ``.cmd`` shim, then Python, on Windows), and a silence the test
#: means to be ignored is longer than the window it must outlast.
WINDOW_S = 3.0


class TestTheIdleClock:

    def test_a_silent_cli_is_stopped_once_the_clock_runs_out(self):
        proc = _popen(_PRINT_THEN_HANG)
        capture = PipeCapture(proc, idle_s=WINDOW_S, in_flight=lambda chunk: False)
        started = time.monotonic()
        try:
            with pytest.raises(IdleStopped) as caught:
                capture.communicate(timeout=30)
            assert time.monotonic() - started < 20
            assert not isinstance(caught.value, StreamStopped)
            assert "printed nothing for 3s" in caught.value.reason
            assert capture.idle_stop == caught.value.reason
            assert "hello" in caught.value.output
        finally:
            _stop(proc)
            capture.close()

    def test_a_tool_in_flight_keeps_the_clock_off(self):
        proc = _popen("import sys,time; print('busy', flush=True); time.sleep(4.5); print('done')")
        capture = PipeCapture(proc, idle_s=WINDOW_S, in_flight=lambda chunk: b"busy" in chunk)
        try:
            out, _err = capture.communicate(timeout=30)
            assert proc.returncode == 0 and "done" in out
            assert capture.idle_stop is None
        finally:
            capture.close()

    def test_output_on_stderr_is_activity_too(self):
        proc = _popen(textwrap.dedent("""\
            import sys, time
            for _ in range(5):
                sys.stderr.write('retrying\\n'); sys.stderr.flush(); time.sleep(1)
        """))
        capture = PipeCapture(proc, idle_s=WINDOW_S, in_flight=lambda chunk: False)
        try:
            capture.communicate(timeout=30)
            assert proc.returncode == 0 and capture.idle_stop is None
        finally:
            capture.close()

    def test_a_tracker_that_raises_turns_the_clock_off(self):
        def broken(chunk):
            raise ValueError("unreadable")

        proc = _popen("import sys,time; print('x', flush=True); time.sleep(4.5)")
        capture = PipeCapture(proc, idle_s=WINDOW_S, in_flight=broken)
        try:
            capture.communicate(timeout=30)
            assert proc.returncode == 0 and capture.idle_stop is None
        finally:
            capture.close()

    def test_no_tracker_no_clock(self):
        proc = _popen("import time; time.sleep(1)")
        capture = PipeCapture(proc, idle_s=0.1)
        try:
            capture.communicate(timeout=30)
            assert proc.returncode == 0 and capture.idle_stop is None
        finally:
            capture.close()

    def test_the_deadline_still_wins_when_it_comes_first(self):
        proc = _popen(_PRINT_THEN_HANG)
        capture = PipeCapture(proc, idle_s=60, in_flight=lambda chunk: False)
        try:
            with pytest.raises(subprocess.TimeoutExpired) as caught:
                capture.communicate(timeout=0.5)
            assert not isinstance(caught.value, IdleStopped)
            assert capture.idle_stop is None
        finally:
            _stop(proc)
            capture.close()


# --------------------------------------------------------------------------- #
# The executor's spawn, through its real kill path
# --------------------------------------------------------------------------- #

#: A codex stand-in: ``hang`` prints an edit and a prose reply and then never
#: ends its turn; ``midwork`` prints its opening preamble and an edit, then
#: goes quiet in mid-turn; ``lone`` prints one message and no tool call (a
#: report re-ask's reply), then never ends its turn; ``busy`` runs one silent
#: tool call longer than the clock and then finishes cleanly; ``silent``
#: prints nothing at all.
_CODEX_STAND_IN = textwrap.dedent("""\
    import json, os, sys, time
    mode, pidfile = sys.argv[1], sys.argv[2]
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
    def emit(event):
        sys.stdout.write(json.dumps(event) + "\\n"); sys.stdout.flush()
    if mode == "silent":
        time.sleep(60); sys.exit(0)
    emit({"type": "thread.started", "thread_id": "t"})
    emit({"type": "turn.started"})
    if mode == "lone":
        emit({"type": "item.completed", "item": {"id": "m", "type": "agent_message",
              "text": "The plan now backfills before switching reads."}})
        time.sleep(60); sys.exit(0)
    if mode == "midwork":
        emit({"type": "item.completed", "item": {"id": "p", "type": "agent_message",
              "text": "I'll read the plan first, then update the sections."}})
    if mode == "busy":
        emit({"type": "item.started", "item": {"id": "c", "type": "command_execution",
              "command": "pytest -q", "aggregated_output": "", "exit_code": None,
              "status": "in_progress"}})
        time.sleep(4.5)
        emit({"type": "item.completed", "item": {"id": "c", "type": "command_execution",
              "command": "pytest -q", "aggregated_output": "ok", "exit_code": 0,
              "status": "completed"}})
    emit({"type": "item.started", "item": {"id": "f", "type": "file_change", "changes": [],
          "status": "in_progress"}})
    emit({"type": "item.completed", "item": {"id": "f", "type": "file_change", "changes": [],
          "status": "completed"}})
    if mode == "midwork":
        time.sleep(60); sys.exit(0)
    emit({"type": "item.completed", "item": {"id": "m", "type": "agent_message",
          "text": "The plan now backfills before switching reads."}})
    if mode == "hang":
        time.sleep(60); sys.exit(0)
    emit({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 5}})
""")


@pytest.fixture
def codex_stand_in(tmp_path, monkeypatch):
    script = tmp_path / "codex_stand_in.py"
    script.write_text(_CODEX_STAND_IN, encoding="utf-8")
    pidfile = tmp_path / "pid"

    def spawn(mode, *, provider="codex", profile=PROFILE_AGENT_WRITE, idle=3, timeout=60,
              **extra):
        adapter = _PROVIDERS[provider]
        argv = [sys.executable, str(script), mode, str(pidfile)]
        monkeypatch.setitem(vars(adapter), "build_agent_command_for_transport",
                            lambda *a, **k: (list(argv), None))
        monkeypatch.setattr(dispatch_spawn, "dispatch_child_env", lambda *a, **k: None)
        return dispatch_spawn._spawn_in_context(
            provider=adapter, model="m", prompt="hi", profile=profile,
            transport="argv", timeout=timeout, cwd=str(tmp_path), idle_timeout=idle,
            **extra,
        )

    spawn.pidfile = pidfile
    return spawn


def _gone(pidfile):
    if os.name == "nt":
        return True
    try:
        os.kill(int(pidfile.read_text()), 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


class TestTheExecutorsStop:

    def test_a_writer_silent_after_its_reply_is_stopped_and_read_as_an_exit(
        self, codex_stand_in,
    ):
        started = time.monotonic()
        capture = codex_stand_in("hang")
        assert time.monotonic() - started < 30
        assert capture.idle_stop and "printed nothing for 3s" in capture.idle_stop
        assert capture.timed_out is False
        assert capture.exit_code not in (0, None)
        assert capture.kill_verdict in ("killed", "gone"), capture.kill_verdict
        assert "backfills" in capture.stdout
        assert attempt_classifier._classify_pre_parse(capture) == (STATUS_OK, "")
        assert dispatch_spawn.runner_exit_code(capture) is None
        assert "so it was stopped (exit" in dispatch_spawn.idle_stop_note(capture)
        assert capture.idle_turn_finished is True
        assert dispatch_spawn.idle_stop_case(capture) == executor.IDLE_STOP_FINISHED
        assert _gone(codex_stand_in.pidfile)

    def test_a_writer_silent_in_mid_turn_stays_a_timeout(self, codex_stand_in):
        """Its preamble, an edit, then silence: stopped as soon, but it is the
        timeout it would have been at the deadline, never read as an exit."""
        started = time.monotonic()
        capture = codex_stand_in("midwork")
        assert time.monotonic() - started < 30
        assert capture.idle_stop and "printed nothing for 3s" in capture.idle_stop
        assert capture.kill_verdict in ("killed", "gone"), capture.kill_verdict
        assert capture.timed_out is True and capture.exit_code is None
        assert capture.idle_turn_finished is False
        assert dispatch_spawn.idle_stop_case(capture) == executor.IDLE_STOP_MID_TURN
        assert dispatch_spawn.idle_stop_note(capture) is None
        assert dispatch_spawn.runner_exit_code(capture) is None
        status, line = attempt_classifier._classify_pre_parse(capture)
        assert status == STATUS_TIMEOUT
        assert line.startswith("the runner printed nothing for 3s")
        assert "in mid-turn" in line and "could not be confirmed" not in line
        assert "no completed turn or closing reply was found in its output" in line
        assert "read the plan first" in capture.stdout
        assert _gone(codex_stand_in.pidfile)

    @pytest.mark.parametrize("reask, finished", [(False, False), (True, True)],
                             ids=["first-dispatch", "report-re-ask"])
    def test_a_lone_message_then_silence_is_a_reply_only_on_a_re_ask(
        self, codex_stand_in, reask, finished,
    ):
        """One message and no tool call: an opening preamble on a writer's
        dispatch (a stall in mid-turn), the reply on a report re-ask."""
        capture = codex_stand_in("lone", reask=reask)
        assert capture.idle_stop and capture.kill_verdict in ("killed", "gone")
        assert capture.idle_turn_finished is finished
        assert capture.timed_out is (not finished)
        assert dispatch_spawn.idle_stop_case(capture) == (
            executor.IDLE_STOP_FINISHED if finished else executor.IDLE_STOP_MID_TURN)
        status, line = attempt_classifier._classify_pre_parse(capture)
        assert status == (STATUS_OK if finished else STATUS_TIMEOUT)
        if not finished:
            # Fable review 2 of 2026-11-19, N6: no tool call was made, so the
            # words must not say a reply failed to follow one.
            said = line.partition(", in mid-turn: ")[2]
            assert said.startswith("no completed turn or closing reply was found")
            assert "tool call" not in said
        assert _gone(codex_stand_in.pidfile)

    def test_a_silent_tool_call_longer_than_the_clock_is_not_stopped(self, codex_stand_in):
        capture = codex_stand_in("busy")
        assert capture.idle_stop is None and capture.timed_out is False
        assert capture.exit_code == 0
        assert _PROVIDERS["codex"].parse_usage(capture.stdout, "")["tool_uses"] == 2

    def test_a_read_dispatch_has_no_clock(self, codex_stand_in):
        capture = codex_stand_in("hang", profile=PROFILE_AGENT_READONLY, timeout=5)
        assert capture.idle_stop is None and capture.timed_out is True
        assert attempt_classifier._classify_pre_parse(capture)[0] == STATUS_TIMEOUT

    def test_an_adapter_that_prints_at_exit_has_no_clock(self, codex_stand_in):
        capture = codex_stand_in("silent", provider="claude-code", timeout=5)
        assert capture.idle_stop is None and capture.timed_out is True

    def test_the_clock_off_means_no_stop(self, codex_stand_in):
        capture = codex_stand_in("hang", idle=None, timeout=5)
        assert capture.idle_stop is None and capture.timed_out is True

    def test_an_unconfirmed_stop_fails_closed_as_a_runner_error(
        self, codex_stand_in, monkeypatch,
    ):
        real = dispatch_spawn._kill_timed_out

        def unconfirmed(proc, pgid, job=None, **kwargs):
            real(proc, pgid, job, **kwargs)
            return "unresolved"

        monkeypatch.setattr(dispatch_spawn, "_kill_timed_out", unconfirmed)
        capture = codex_stand_in("hang")
        assert capture.idle_stop and capture.timed_out is True
        assert dispatch_spawn.idle_stop_note(capture) is None
        assert capture.idle_turn_finished is None
        assert dispatch_spawn.idle_stop_case(capture) == executor.IDLE_STOP_UNCONFIRMED
        status, line = attempt_classifier._classify_pre_parse(capture)
        assert status == STATUS_RUNNER_ERROR
        assert line.startswith("the runner printed nothing for 3s")
        assert "could not be confirmed stopped" in line


@pytest.mark.parametrize("stdout, capped", [
    ("", False),
    (_REASK_REPLY, False),
    (_REASK_REPLY + _event("turn.failed", error={"message": "stream disconnected"}), False),
    (_INCIDENT, True),
], ids=["empty", "lone-message", "turn-failed", "capped"])
def test_the_mid_turn_words_hold_for_every_mid_turn_case(stdout, capped):
    """Fable review 2 of 2026-11-19, N6: the line said "no reply followed its
    last tool call" for an empty stream, a lone message and a failed turn
    too. It claims only what every mid-turn reading shares."""
    capture = dispatch_spawn._Capture(
        stdout=stdout, timed_out=True, kill_pid=1, kill_verdict="killed",
        idle_stop="the runner printed nothing for 3s with no tool call in flight",
        capped_streams=frozenset({"stdout"}) if capped else frozenset(),
    )
    capture.idle_turn_finished = dispatch_spawn._idle_turn_finished(_PROVIDERS["codex"], capture)
    assert capture.idle_turn_finished is False
    status, line = attempt_classifier._classify_pre_parse(capture)
    assert status == STATUS_TIMEOUT
    assert line == (
        "the runner printed nothing for 3s with no tool call in flight, in mid-turn: "
        "no completed turn or closing reply was found in its output, so it was stopped "
        "as at its per-task timeout and nothing it printed was read")


# --------------------------------------------------------------------------- #
# The scheduler's outcomes, against the mock codex
# --------------------------------------------------------------------------- #

REASK_MARKER = "You are re-sending the report of a batch of edits."
GOOD = {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
        "summary": "added the step"}


def _prompts(log: Path):
    if not log.exists():
        return []
    return [json.loads(line)["prompt"] for line in log.read_text().splitlines()]


def _report(env):
    return json.loads((env.phase_dir / "write_batch_report.json").read_text(encoding="utf-8"))


class TestTheRecovery:

    @pytest.fixture(autouse=True)
    def _idle(self, env):
        # Longer than the mock launcher's start, so the writer always edits and
        # replies before the clock can run out.
        env.use("codex:gpt-5.5", subagent_idle_timeout_s=4)
        env.monkeypatch.setenv("MOCK_LLM_HANG_AFTER_REPLY", "1")
        # The incident's shape: a tool item before the closing message. The
        # mock's edit lands either way; the item is what the stream shows.
        env.monkeypatch.setenv("MOCK_LLM_CODEX_TOOL_ITEMS", "1")

    def test_edits_then_prose_then_silence_get_the_report_re_ask(self, env, capsys):
        """The incident: an edit, then a prose message, then silence. The
        runner had finished speaking, so it is read as an exit, re-asked, and
        applied on the moved bytes."""
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)], prose=True)
        env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps(GOOD))
        started = time.monotonic()

        result = env.scheduler([env.apply_task(0)]).run()

        assert time.monotonic() - started < 60
        first, second = _prompts(log)
        assert "Apply batch 0" in first and second.startswith(REASK_MARKER)
        outcome = result.outcomes[0]
        assert [(a.status, a.repair) for a in outcome.attempts] == [
            (executor.STATUS_UNPARSEABLE, None),
            (executor.STATUS_OK, executor.REPAIR_SCHEMA_INVALID),
        ]
        stopped = outcome.attempts[0]
        assert stopped.exit_code not in (0, None)
        assert stopped.error.startswith("the runner printed nothing for 4s")
        assert stopped.idle_stop == executor.IDLE_STOP_FINISHED
        assert outcome.attempts[1].idle_stop is None
        assert outcome.write_outcome == executor.WRITE_APPLIED
        assert result.exit_code == 0
        assert env.plan.read_text(encoding="utf-8").count("edit\n") == 1
        assert "printed nothing for 4s" in capsys.readouterr().out
        rows = _report(env)["tasks"][0]["attempts"]
        assert rows[0]["idle_stop"] == "finished_turn" and "idle_stop" not in rows[1]

    def test_edits_then_a_valid_report_then_silence_are_applied(self, env):
        env.edits([_patch_edit(env.plan)], result=GOOD)

        result = env.scheduler([env.apply_task(0)]).run()

        outcome = result.outcomes[0]
        assert [a.status for a in outcome.attempts] == [executor.STATUS_OK]
        assert outcome.attempts[0].exit_code not in (0, None)
        assert outcome.attempts[0].idle_stop == executor.IDLE_STOP_FINISHED
        assert outcome.write_outcome == executor.WRITE_APPLIED, outcome.reason
        assert "printed nothing for 4s" in outcome.reason
        assert result.exit_code == 0

    def test_a_mid_turn_stall_stops_as_a_partial_without_a_re_ask(self, env, capsys):
        """Edits, then silence after a tool call, with only the opening
        preamble said: the timeout it would have been at the deadline, only
        sooner. Nothing it printed is read or re-asked, its edits are kept,
        and the batch stops as a partial application."""
        log = env.enable_call_log()
        env.monkeypatch.setenv("MOCK_LLM_HANG_AFTER_REPLY", "midwork")
        env.edits([_patch_edit(env.plan)])
        started = time.monotonic()

        result = env.scheduler([env.apply_task(0)]).run()

        assert time.monotonic() - started < 60
        prompts = _prompts(log)
        assert len(prompts) == 1 and "Apply batch 0" in prompts[0]
        outcome = result.outcomes[0]
        [stopped] = outcome.attempts
        assert stopped.status == executor.STATUS_PARTIAL
        assert stopped.idle_stop == executor.IDLE_STOP_MID_TURN
        assert stopped.exit_code is None
        assert stopped.error.startswith("PARTIAL APPLICATION")
        assert "in mid-turn" in stopped.error
        assert outcome.write_outcome == executor.WRITE_PARTIAL, outcome.reason
        assert outcome.ran_out_of_time is True
        assert result.exit_code == 3
        assert env.plan.read_text(encoding="utf-8").count("edit\n") == 1
        assert "went quiet in mid-turn" in capsys.readouterr().out
        row = _report(env)["tasks"][0]
        assert row["attempts"][0]["idle_stop"] == "mid_turn"
        # Its tool call is counted; no token figure was read, so the task
        # counts under `tasks_without_usage`.
        assert row["usage"] == {"tool_uses": 1} and row["usage_reported"] is False
        totals = json.loads(
            (env.phase_dir / "subagent_batch_report.json").read_text(encoding="utf-8"))["usage"]
        assert totals["tasks_without_usage"] == 1 and totals["tool_uses"] == 1
        # The stop names the idle timeout, and a longer time limit is no remedy.
        assert result.stop["timed_out"] is True and result.stop["idle_stalled"] is True
        assert "went quiet in mid-turn" in result.stop["what"]
        assert "subagent_timeout_s" not in result.stop["headline"]
        assert "run it on another model" in result.stop["headline"]

    def test_a_mid_turn_stall_cannot_be_applied_by_a_re_ask_claiming_every_id(self, env):
        """Fable review 2026-11-19, finding 3: a re-ask after a mid-turn stall that
        attested every id was applied with edits missing. No re-ask is sent
        after one now, so the batch stops as a partial application, as it did
        at the slot deadline before the idle timeout."""
        log = env.enable_call_log()
        env.monkeypatch.setenv("MOCK_LLM_HANG_AFTER_REPLY", "midwork")
        env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps(GOOD))
        env.edits([_patch_edit(env.plan)])

        result = env.scheduler([env.apply_task(0)]).run()

        assert not any(p.startswith(REASK_MARKER) for p in _prompts(log))
        outcome = result.outcomes[0]
        assert [(a.status, a.repair, a.idle_stop) for a in outcome.attempts] == [
            (executor.STATUS_PARTIAL, None, executor.IDLE_STOP_MID_TURN),
        ]
        assert outcome.write_outcome == executor.WRITE_PARTIAL, outcome.reason
        assert result.exit_code == 3

    def test_a_message_with_no_tool_call_before_it_is_mid_turn(self, env):
        """Nothing after the preamble: a lone message cannot be told from an
        opening one, so even with bytes moved (the mock's edit lands without
        a tool item) it is never read or re-asked."""
        log = env.enable_call_log()
        env.monkeypatch.setenv("MOCK_LLM_CODEX_TOOL_ITEMS", "0")
        env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps(GOOD))
        env.edits([_patch_edit(env.plan)], prose=True)

        result = env.scheduler([env.apply_task(0)]).run()

        assert len(_prompts(log)) == 1
        outcome = result.outcomes[0]
        assert [(a.status, a.idle_stop) for a in outcome.attempts] == [
            (executor.STATUS_PARTIAL, executor.IDLE_STOP_MID_TURN),
        ]
        assert outcome.write_outcome == executor.WRITE_PARTIAL, outcome.reason
        assert result.exit_code == 3

    def test_a_re_ask_silent_after_its_report_is_applied_on_it(self, env):
        """Fable review 2 of 2026-11-19, N4: the report re-ask runs no tool, so
        its report is a lone message. Read as an opening preamble, a re-ask
        that answered and then hung was a timeout, re-sent while the budget
        lasted, and the batch stopped as a partial application with a valid
        report in hand. On a re-ask the message is the reply: it is read,
        and the batch is applied on it and the moved bytes."""
        log = env.enable_call_log()
        env.edits([_patch_edit(env.plan)], prose=True)
        env.monkeypatch.setenv("MOCK_LLM_REPAIR_RESULT", json.dumps(GOOD))
        env.monkeypatch.setenv("MOCK_LLM_HANG_AFTER_REASK", "1")
        real = dispatch_spawn._spawn
        reasks = []

        def spawn(**kwargs):
            reasks.append(kwargs.get("reask"))
            return real(**kwargs)

        env.monkeypatch.setattr(dispatch_spawn, "_spawn", spawn)

        result = env.scheduler([env.apply_task(0)]).run()

        first, second = _prompts(log)
        assert "Apply batch 0" in first and second.startswith(REASK_MARKER)
        assert reasks == [False, True]
        outcome = result.outcomes[0]
        assert [(a.status, a.repair, a.idle_stop) for a in outcome.attempts] == [
            (executor.STATUS_UNPARSEABLE, None, executor.IDLE_STOP_FINISHED),
            (executor.STATUS_OK, executor.REPAIR_SCHEMA_INVALID, executor.IDLE_STOP_FINISHED),
        ]
        assert outcome.write_outcome == executor.WRITE_APPLIED, outcome.reason
        assert result.exit_code == 0
        assert env.plan.read_text(encoding="utf-8").count("edit\n") == 1

    @pytest.mark.parametrize("tool_items,status,case", [
        ("1", executor.STATUS_UNPARSEABLE, "finished_turn"),
        ("0", STATUS_TIMEOUT, "mid_turn"),
    ], ids=["finished", "mid-turn"])
    def test_silence_with_nothing_moved_is_retried_as_the_batch(
        self, env, tool_items, status, case,
    ):
        log = env.enable_call_log()
        env.monkeypatch.setenv("MOCK_LLM_CODEX_TOOL_ITEMS", tool_items)
        env.edits([], prose=True)

        result = env.scheduler([env.apply_task(0)]).run()

        prompts = _prompts(log)
        assert len(prompts) == 2 and all("Apply batch 0" in p for p in prompts)
        outcome = result.outcomes[0]
        assert outcome.status == status
        assert all(a.error.startswith("the runner printed nothing") for a in outcome.attempts)
        assert [a.idle_stop for a in outcome.attempts] == [case, case]
        assert result.exit_code == 3


# --------------------------------------------------------------------------- #
# The executor records codex's count where the native leg records the host's
# --------------------------------------------------------------------------- #

def test_the_tool_count_reaches_state_json_and_the_batch_report(env):
    env.monkeypatch.setenv("MOCK_LLM_CODEX_TOOL_ITEMS", "3")
    env.edits([_patch_edit(env.plan)], result=GOOD)
    state = FakeState(env.tmp_path / "state.json")

    result = env.scheduler([env.apply_task(0)], state=state, finalize=False).run()

    assert result.exit_code == 0
    assert result.outcomes[0].usage["tool_uses"] == 3
    saved = json.loads((env.tmp_path / "state.json").read_text(encoding="utf-8"))
    [entry] = saved["metrics"]["apply-suggestions"]
    assert entry["tool_uses"] == 3
    report = json.loads((env.phase_dir / "write_batch_report.json").read_text())
    [row] = report["tasks"]
    assert row["usage"]["tool_uses"] == 3
    assert row["attempts"][0]["usage"]["tool_uses"] == 3


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class TestTheKey:

    @pytest.fixture(autouse=True)
    def _fresh(self):
        registry._config = None
        yield
        registry._config = None

    def _driver(self, tmp_path, monkeypatch, value):
        path = tmp_path / "providers.yaml"
        path.write_text(f"driver:\n  subagent_idle_timeout_s: {value}\n", encoding="utf-8")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
        registry._config = None

    def test_the_base_default_is_on(self, tmp_path, monkeypatch):
        path = tmp_path / "providers.yaml"
        path.write_text("driver:\n  subagent_retries: 1\n", encoding="utf-8")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
        assert registry.get_subagent_idle_timeout() == 900
        assert registry.DRIVER_FALLBACKS["subagent_idle_timeout_s"] == 900

    def test_zero_turns_it_off(self, tmp_path, monkeypatch):
        self._driver(tmp_path, monkeypatch, 0)
        assert registry.get_subagent_idle_timeout() is None

    def test_a_blank_value_keeps_the_base(self, tmp_path, monkeypatch):
        self._driver(tmp_path, monkeypatch, "")
        assert registry.get_subagent_idle_timeout() == 900

    def test_a_number_sets_it(self, tmp_path, monkeypatch):
        self._driver(tmp_path, monkeypatch, 120)
        assert registry.get_subagent_idle_timeout() == 120

    @pytest.mark.parametrize("value", [-1, 1.5, True, "'900'"])
    def test_anything_else_is_refused(self, tmp_path, monkeypatch, value):
        self._driver(tmp_path, monkeypatch, value)
        with pytest.raises(ConfigError, match="subagent_idle_timeout_s"):
            registry.get_subagent_idle_timeout()
