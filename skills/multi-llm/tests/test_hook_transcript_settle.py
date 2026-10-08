#!/usr/bin/env python3
"""The Claude Code plugin's hooks read a transcript Claude Code is still
writing (fix round 27, the Fable review of fix round 26).

Claude Code 2.1.293 writes the session transcript in batches. Measured against
a localhost Messages-API stub (2026-10-08): when the ``Stop`` hook starts, the
reply is never on file yet and appears 0.1-0.2 s later; when the reply came at
once after the turn's last tool result, that result was not on file either
(in so young a session the whole file was missing), so a ``[NEXT_COMMAND]``
block printed by the turn's last call was not seen, ``hooks/report-check.sh``
exited at its marker grep, and a report without the commands passed. The
``PreToolUse`` hook had the same blind spot: the ``ScheduleWakeup`` call, a
phase call in the same message and a phase result just printed can be
unwritten when it starts.

So both hooks wait, at most ``report_check.SETTLE_BUDGET_S``, for what they
need to be on file: the ``Stop`` check for the reply its input names
(``last_assistant_message``), the ``PreToolUse`` check for the call it decides
and the results of the phase calls its message made before it. These tests
write the missing entries from a background thread while the hook waits.

The same round tightens what ends ``util wakeup-check``'s hold: a subagent's
report quoting the block ends nothing, and a background phase that failed or
was stopped ends it on Claude Code's ``task-notification``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
REPORT_CHECK_SH = REPO_ROOT / "hooks" / "report-check.sh"
WAKEUP_CHECK_SH = REPO_ROOT / "hooks" / "wakeup-check.sh"

sys.path.insert(0, str(SKILL_ROOT))

from tests.test_report_check import (  # noqa: E402
    COMMAND,
    OPTIONAL,
    Transcript,
    _fake_uv,
    block_output,
    decision_of,
    report,
)
from tests.test_report_check_session import notification  # noqa: E402
from tests.test_wakeup_check import DISPATCH, PLAN, OWN_CALL, own_call_entry  # noqa: E402
from utils import report_check, wakeup_check  # noqa: E402

#: How long after the hook starts the background thread writes.
LATE_S = 0.3

#: The `util next-command` call a stopped implement's report step runs.
LOOKUP = f'{DISPATCH} util next-command "{PLAN}" --after implement --host claude-code'
#: A phase launched in the background with its log redirected, as SKILL.md says.
BACKGROUND_PHASE = (f'{DISPATCH} review-tasks "{PLAN}" --host claude-code --yes '
                    "> plans/racing-game/review-tasks/orchestrator-run.log 2>&1")
LAUNCHED = ("Command running in background with ID: b1. Output is being written "
            "to: /tmp/claude-1000/tasks/b1.output. You will be notified when it "
            "completes.")


@pytest.fixture(autouse=True)
def _the_hooks_budget(monkeypatch):
    """The hooks' own budget, in finer steps than theirs."""
    monkeypatch.setattr(report_check, "SETTLE_BUDGET_S", 2.0, raising=False)
    monkeypatch.setattr(report_check, "SETTLE_STEP_S", 0.05, raising=False)


def line(entry: Dict[str, Any]) -> str:
    return json.dumps(entry) + "\n"


def write_later(path: Path, text: str, delay: float = LATE_S) -> threading.Thread:
    """Append *text* to *path* after *delay* seconds, from a thread: Claude
    Code writing the transcript while the hook reads it."""

    def append() -> None:
        time.sleep(delay)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(text)

    thread = threading.Thread(target=append, daemon=True)
    thread.start()
    return thread


def tool_use(tool_id: str, command: str, message: str, name: str = "Bash",
             **tool_input: Any) -> Dict[str, Any]:
    tool_input.setdefault("command", command)
    return {"type": "assistant", "message": {"id": message, "role": "assistant", "content": [
        {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}]}}


def tool_result(tool_id: str, text: str) -> Dict[str, Any]:
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": text}]}}


def reply(text: str) -> Dict[str, Any]:
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": text}]}}


STOP_SUMMARY = {"type": "system", "subtype": "stop_hook_summary", "hookCount": 1}


def stop_input(path: Path, last_message: Optional[str], **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "session_id": "s1", "transcript_path": str(path), "cwd": str(path.parent),
        "hook_event_name": "Stop", "stop_hook_active": False,
    }
    if last_message is not None:
        payload["last_assistant_message"] = last_message
    payload.update(extra)
    return payload


def run_report_check(capsys, tmp_path: Path, payload: Dict[str, Any]) -> str:
    hook_file = tmp_path / "hook-input.json"
    hook_file.write_text(json.dumps(payload), encoding="utf-8")
    assert report_check.main(["--hook-input", str(hook_file)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out


def wakeup_input(path: Path, tool: str = "ScheduleWakeup") -> Dict[str, Any]:
    return {"session_id": "s1", "transcript_path": str(path), "cwd": str(path.parent),
            "hook_event_name": "PreToolUse", "tool_name": tool,
            "tool_input": {"delaySeconds": 60, "prompt": "check the phase",
                           "reason": "waiting"},
            "tool_use_id": OWN_CALL}


def run_wakeup_check(capsys, tmp_path: Path, payload: Dict[str, Any]) -> str:
    hook_file = tmp_path / "hook-input.json"
    hook_file.write_text(json.dumps(payload), encoding="utf-8")
    assert wakeup_check.main(["--hook-input", str(hook_file)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out


def denied(out: str) -> str:
    decision = json.loads(out)
    specific = decision["hookSpecificOutput"]
    assert specific["permissionDecision"] == "deny"
    return specific["permissionDecisionReason"]


def run_script(script: Path, tmp_path: Path, payload: Dict[str, Any]):
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
        "CLAUDE_PLUGIN_ROOT": str(REPO_ROOT), "ARGV_LOG": str(tmp_path / "argv.txt"),
        "REAL_PYTHON": sys.executable, "MULTI_LLM_UV": str(_fake_uv(tmp_path)),
    }
    return subprocess.run(["/bin/sh", str(script)], input=json.dumps(payload), env=env,
                          capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# The Stop hook: the reply and the result before it, written late
# ---------------------------------------------------------------------------

class TestTheStopCheckWaitsForTheReply:

    def test_a_block_the_last_call_printed_is_seen_once_written(self, tmp_path, capsys):
        """The reviewer's race: the lookup is the turn's last call, its block
        and the reply land on file after the hook started."""
        path = tmp_path / "t.jsonl"
        path.write_text(line(Transcript().prompt("go").entries[0])
                        + line(tool_use("toolu_1", LOOKUP, "msg_1")), encoding="utf-8")
        write_later(path, line(tool_result("toolu_1", block_output()))
                    + line(reply("Implement stopped. Done.")))
        decision = decision_of(run_report_check(
            capsys, tmp_path, stop_input(path, "Implement stopped. Done.")))
        assert f"```\n{COMMAND}\n```" in decision["reason"]

    def test_a_reply_line_still_being_written_is_waited_for(self, tmp_path, capsys):
        """A half-written last line used to make the transcript unreadable,
        which passes silently."""
        path = Transcript().prompt("go").bash(block_output()).write(tmp_path / "t.jsonl")
        whole = line(reply("Done."))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(whole[:25])
        write_later(path, whole[25:])
        decision_of(run_report_check(capsys, tmp_path, stop_input(path, "Done.")))

    def test_an_earlier_stop_s_reply_worded_the_same_is_not_this_one(self, tmp_path, capsys):
        """A notification turn ends on the same words as the turn before it:
        the earlier reply has a `stop_hook_summary` after it, so the hook
        waits for this turn's block and reply."""
        newer = "/multi-llm:multi-llm --implement 'plans/my plan.md' --resume"
        transcript = Transcript().prompt("go").bash(block_output()).say("Done.")
        path = transcript.write(tmp_path / "t.jsonl")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line(STOP_SUMMARY))
        later = notification(Transcript()).bash(block_output(newer)).say("Done.")
        write_later(path, "".join(line(entry) for entry in later.entries))
        reason = decision_of(run_report_check(capsys, tmp_path, stop_input(path, "Done.")))["reason"]
        assert f"```\n{newer}\n```" in reason

    def test_a_copy_the_settled_transcript_holds_passes(self, tmp_path, capsys):
        """Once the reply is on file its earlier text blocks count too: the
        field alone is the reply's last block."""
        path = tmp_path / "t.jsonl"
        transcript = Transcript().prompt("go").bash(block_output())
        path.write_text("".join(line(entry) for entry in transcript.entries), encoding="utf-8")
        write_later(path, line(reply(report(COMMAND))) + line(reply("Anything else?")))
        assert run_report_check(capsys, tmp_path, stop_input(path, "Anything else?")) == ""

    def test_a_settled_transcript_costs_one_read(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).say(report(COMMAND))
                .write(tmp_path / "t.jsonl"))
        started = time.monotonic()
        assert run_report_check(capsys, tmp_path, stop_input(path, report(COMMAND))) == ""
        assert time.monotonic() - started < 0.5

    def test_a_reply_never_written_is_waited_for_no_longer_than_the_budget(
            self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(report_check, "SETTLE_BUDGET_S", 0.4, raising=False)
        path = Transcript().prompt("go").bash(block_output()).write(tmp_path / "t.jsonl")
        started = time.monotonic()
        decision_of(run_report_check(capsys, tmp_path, stop_input(path, "Done.")))
        assert time.monotonic() - started < 1.5

    def test_a_turn_already_blocked_once_is_not_waited_for(self, tmp_path, capsys):
        path = Transcript().prompt("go").bash(block_output()).write(tmp_path / "t.jsonl")
        started = time.monotonic()
        assert run_report_check(capsys, tmp_path, stop_input(
            path, "Done.", stop_hook_active=True)) == ""
        assert time.monotonic() - started < 0.5

    def test_with_no_field_it_waits_for_the_file_to_stop_growing(
            self, tmp_path, capsys, monkeypatch):
        """A reply with no text hands the hook no field to wait for; the file
        counts as written once it held its size for one step."""
        monkeypatch.setattr(report_check, "SETTLE_STEP_S", 0.2, raising=False)
        path = (Transcript().prompt("go").bash(block_output())
                .write(tmp_path / "t.jsonl"))
        write_later(path, line(reply("Done.")), delay=0.05)
        decision_of(run_report_check(capsys, tmp_path, stop_input(path, None)))

    def test_a_transcript_named_on_the_command_line_is_read_as_it_stands(
            self, tmp_path, capsys):
        path = Transcript().prompt("go").bash(block_output()).write(tmp_path / "t.jsonl")
        started = time.monotonic()
        assert report_check.main(["--transcript", str(path)]) == 0
        assert capsys.readouterr().out == ""
        assert time.monotonic() - started < 0.5


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")
class TestTheStopScriptDoesNotExitBeforeTheBlockIsWritten:

    def test_a_dispatcher_call_whose_block_is_not_on_file_yet_goes_on(self, tmp_path):
        """Before fix round 27 the script's marker grep exited here in a few
        milliseconds, and the util never ran."""
        path = tmp_path / "t.jsonl"
        path.write_text(line(Transcript().prompt("go").entries[0])
                        + line(tool_use("toolu_1", LOOKUP, "msg_1")), encoding="utf-8")
        write_later(path, line(tool_result("toolu_1", block_output())) + line(reply("Done.")))
        completed = run_script(REPORT_CHECK_SH, tmp_path, stop_input(path, "Done."))
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["decision"] == "block"

    def test_a_transcript_file_written_after_the_hook_started(self, tmp_path):
        """In a session that young the whole file was missing at `Stop`."""
        path = tmp_path / "t.jsonl"
        transcript = (Transcript().prompt("go").bash(block_output())
                      .say("Implement stopped. Done."))
        write_later(path, "".join(line(entry) for entry in transcript.entries))
        completed = run_script(REPORT_CHECK_SH, tmp_path,
                               stop_input(path, "Implement stopped. Done."))
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["decision"] == "block"

    def test_a_bash_call_without_the_dispatcher_still_exits_before_uv(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(line(Transcript().prompt("list the files").entries[0])
                        + line(tool_use("toolu_1", "ls", "msg_1"))
                        + line(tool_result("toolu_1", "a.txt\nb.txt\n"))
                        + line(reply("Two files.")), encoding="utf-8")
        completed = run_script(REPORT_CHECK_SH, tmp_path, stop_input(path, "Two files."))
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not (tmp_path / "argv.txt").exists()


# ---------------------------------------------------------------------------
# The PreToolUse hook: the call being decided, written late
# ---------------------------------------------------------------------------

class TestTheWakeupCheckWaitsForTheCall:

    def test_a_phase_result_still_being_written_ends_the_hold(self, tmp_path, capsys):
        """Finding 2: the phase printed its block, but its result was not on
        file when the hook started, so the phase read as running."""
        path = tmp_path / "t.jsonl"
        path.write_text(line(Transcript().prompt("go").entries[0])
                        + line(tool_use("toolu_1", f'{DISPATCH} generate-tasks "{PLAN}" '
                                        "--host claude-code --yes --reaggregate", "msg_1")),
                        encoding="utf-8")
        write_later(path, line(tool_result("toolu_1", block_output()))
                    + line(own_call_entry()))
        assert run_wakeup_check(capsys, tmp_path, wakeup_input(path)) == ""

    def test_a_phase_launched_in_the_same_message_is_seen(self, tmp_path, capsys):
        """Finding 3c: a phase call and the wakeup in one assistant message,
        neither on file when the hook started."""
        path = tmp_path / "t.jsonl"
        path.write_text(line(Transcript().prompt("go").entries[0]), encoding="utf-8")
        launch = tool_use("toolu_1", BACKGROUND_PHASE, "msg_1", run_in_background=True)
        own = own_call_entry()
        own["message"]["id"] = "msg_1"
        write_later(path, line(launch) + line(own) + line(tool_result("toolu_1", LAUNCHED)))
        assert "review-tasks phase" in denied(run_wakeup_check(capsys, tmp_path, wakeup_input(path)))

    def test_a_same_message_phase_call_that_finished_is_waited_for(self, tmp_path, capsys):
        """The phase ran in the foreground before the wakeup was asked for;
        its block lands on file after the hook started."""
        path = tmp_path / "t.jsonl"
        own = own_call_entry()
        own["message"]["id"] = "msg_1"
        path.write_text(line(Transcript().prompt("go").entries[0])
                        + line(tool_use("toolu_1", f'{DISPATCH} generate-tasks "{PLAN}" '
                                        "--host claude-code --yes", "msg_1"))
                        + line(own), encoding="utf-8")
        write_later(path, line(tool_result("toolu_1", block_output())))
        assert run_wakeup_check(capsys, tmp_path, wakeup_input(path)) == ""

    def test_a_phase_call_after_it_in_the_message_is_not_waited_for(self, tmp_path, capsys):
        """A call after the wakeup in the same message has not run yet: the
        wakeup is decided without waiting for its result."""
        path = tmp_path / "t.jsonl"
        own = own_call_entry()
        own["message"]["id"] = "msg_1"
        path.write_text(line(Transcript().prompt("go").entries[0]) + line(own)
                        + line(tool_use("toolu_2", f'{DISPATCH} implement "{PLAN}" '
                                        "--host claude-code --yes", "msg_1")),
                        encoding="utf-8")
        started = time.monotonic()
        assert "implement phase" in denied(run_wakeup_check(capsys, tmp_path, wakeup_input(path)))
        assert time.monotonic() - started < 1.0

    def test_a_same_message_phase_still_running_is_running(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(report_check, "SETTLE_BUDGET_S", 0.3, raising=False)
        path = tmp_path / "t.jsonl"
        own = own_call_entry()
        own["message"]["id"] = "msg_1"
        path.write_text(line(Transcript().prompt("go").entries[0])
                        + line(tool_use("toolu_1", f'{DISPATCH} implement "{PLAN}" '
                                        "--host claude-code --yes", "msg_1"))
                        + line(own), encoding="utf-8")
        assert "implement phase" in denied(run_wakeup_check(capsys, tmp_path, wakeup_input(path)))

    def test_the_refusal_says_what_it_knows(self, tmp_path, capsys):
        """It knows no next step was printed, not that the process runs."""
        path = tmp_path / "t.jsonl"
        transcript = Transcript().prompt("go")
        path.write_text("".join(line(entry) for entry in transcript.entries)
                        + line(tool_use("toolu_1", BACKGROUND_PHASE, "msg_1",
                                        run_in_background=True))
                        + line(tool_result("toolu_1", LAUNCHED))
                        + line(own_call_entry()), encoding="utf-8")
        reason = denied(run_wakeup_check(capsys, tmp_path, wakeup_input(path)))
        assert ("ScheduleWakeup is refused while the review-tasks phase you started "
                "has not printed its next step in this session.") in reason
        assert "still running" not in reason


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")
class TestTheWakeupScriptWaitsForTheCall:

    def test_a_phase_launched_in_the_same_message_reaches_the_util(self, tmp_path):
        """Before fix round 27 the script's `multi_llm.py` grep exited at once
        on a transcript that did not hold the launch yet."""
        path = tmp_path / "t.jsonl"
        path.write_text(line(Transcript().prompt("go").entries[0]), encoding="utf-8")
        launch = tool_use("toolu_1", BACKGROUND_PHASE, "msg_1", run_in_background=True)
        own = own_call_entry()
        own["message"]["id"] = "msg_1"
        write_later(path, line(launch) + line(own) + line(tool_result("toolu_1", LAUNCHED)))
        completed = run_script(WAKEUP_CHECK_SH, tmp_path, wakeup_input(path))
        assert completed.returncode == 0, completed.stderr
        assert "review-tasks phase" in denied(completed.stdout)


# ---------------------------------------------------------------------------
# What ends the hold, and what does not
# ---------------------------------------------------------------------------

def background_notification(tool_id: str, status: str, summary: str) -> Dict[str, Any]:
    """A background command's end, as Claude Code 2.1.293 writes it (read off
    a stub session, 2026-10-08)."""
    return {
        "type": "user", "promptSource": "system", "turnOrigin": "task_notification",
        "origin": {"kind": "task-notification", "producer": "session-task", "runId": "r1"},
        "message": {"role": "user", "content": (
            f"<task-notification>\n<task-id>b1</task-id>\n<tool-use-id>{tool_id}</tool-use-id>\n"
            "<output-file>/tmp/claude-1000/tasks/b1.output</output-file>\n"
            f"<status>{status}</status>\n<summary>{summary}</summary>\n</task-notification>")},
    }


def launched(path: Path, *after: Dict[str, Any]) -> Path:
    """A background review-tasks launch, *after* it, then the wakeup call."""
    transcript = Transcript().prompt("/multi-llm:multi-llm --review-tasks plans/racing-game.md")
    body = "".join(line(entry) for entry in transcript.entries)
    body += line(tool_use("toolu_1", BACKGROUND_PHASE, "msg_1", run_in_background=True))
    body += line(tool_result("toolu_1", LAUNCHED)) + line(reply("Launched."))
    body += "".join(line(entry) for entry in after)
    path.write_text(body + line(own_call_entry()), encoding="utf-8")
    return path


class TestABackgroundPhaseThatEnded:

    def test_one_that_failed_ends_the_hold(self, tmp_path, capsys):
        """Finding 3b: its exit 1 reached the session only as a notification,
        and every later wakeup of the session was refused."""
        path = launched(tmp_path / "t.jsonl", background_notification(
            "toolu_1", "failed", 'Background command "review-tasks" failed with exit code 1'))
        assert run_wakeup_check(capsys, tmp_path, wakeup_input(path)) == ""

    @pytest.mark.parametrize("status,summary", [
        ("killed", 'Background command "review-tasks" was stopped'),
        ("failed", 'Background command "review-tasks" failed with exit code 3'),
        ("completed", 'Background command "review-tasks" completed (exit code 2)'),
    ])
    def test_one_stopped_or_that_exited_non_zero_ends_it(self, tmp_path, capsys,
                                                         status, summary):
        path = launched(tmp_path / "t.jsonl", background_notification("toolu_1", status, summary))
        assert run_wakeup_check(capsys, tmp_path, wakeup_input(path)) == ""

    @pytest.mark.parametrize("summary", [
        'Background command "review-tasks" completed (exit code 0)',
        'Background command "retry on exit code 5" completed (exit code 0)',
    ])
    def test_one_that_exited_0_has_not_printed_its_next_step(self, tmp_path, capsys, summary):
        """A phase whose first call exits 0 hands its subagents to the host;
        the command's own description does not speak for its exit."""
        path = launched(tmp_path / "t.jsonl",
                        background_notification("toolu_1", "completed", summary))
        assert "review-tasks phase" in denied(run_wakeup_check(capsys, tmp_path, wakeup_input(path)))

    def test_another_call_s_failure_ends_nothing(self, tmp_path, capsys):
        path = launched(tmp_path / "t.jsonl", background_notification(
            "toolu_9", "failed", 'Background command "npm test" failed with exit code 1'))
        assert "review-tasks phase" in denied(run_wakeup_check(capsys, tmp_path, wakeup_input(path)))

    def test_a_notification_quoted_by_a_tool_ends_nothing(self, tmp_path, capsys):
        """Only an entry the harness sent says a call ended."""
        quoted = background_notification(
            "toolu_1", "failed", 'Background command "review-tasks" failed with exit code 1')
        text = quoted["message"]["content"]
        path = launched(tmp_path / "t.jsonl",
                        tool_use("toolu_2", "cat notes.txt", "msg_2"),
                        tool_result("toolu_2", text))
        assert "review-tasks phase" in denied(run_wakeup_check(capsys, tmp_path, wakeup_input(path)))


class TestWhatPrintsThePhasesNextStep:

    @pytest.mark.parametrize("tool", ["Agent", "Task", "SendMessage"])
    def test_a_subagent_s_report_quoting_the_block_ends_nothing(self, tmp_path, capsys, tool):
        """Finding 3a: the subagent's words, not the phase's output."""
        path = launched(tmp_path / "t.jsonl",
                        tool_use("toolu_2", "", "msg_2", name=tool,
                                 prompt="Validate batch 0", description="batch 0"),
                        tool_result("toolu_2", "Done. The next command is:\n"
                                    f"[NEXT_COMMAND] {COMMAND}\n"))
        assert "review-tasks phase" in denied(run_wakeup_check(capsys, tmp_path, wakeup_input(path)))

    def test_a_read_of_the_phase_s_log_ends_it(self, tmp_path, capsys):
        path = launched(tmp_path / "t.jsonl",
                        tool_use("toolu_2", "", "msg_2", name="Read",
                                 file_path="plans/racing-game/review-tasks/orchestrator-run.log"),
                        tool_result("toolu_2", block_output()))
        assert run_wakeup_check(capsys, tmp_path, wakeup_input(path)) == ""

    def test_a_tail_of_the_phase_s_log_ends_it(self, tmp_path, capsys):
        path = launched(tmp_path / "t.jsonl",
                        tool_use("toolu_2", "tail -n 20 plans/racing-game/review-tasks/"
                                 "orchestrator-run.log", "msg_2"),
                        tool_result("toolu_2", block_output(OPTIONAL)))
        assert run_wakeup_check(capsys, tmp_path, wakeup_input(path)) == ""
