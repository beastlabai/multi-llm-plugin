#!/usr/bin/env python3
"""The Claude Code plugin's hooks read a transcript Claude Code is still
writing (fix round 27, the Fable review of fix round 26).

Claude Code 2.1.293 writes the session transcript in batches. Measured against
a localhost Messages-API stub (2026-10-08): when the ``Stop`` hook starts, the
reply is never on file yet and appears 0.1-0.2 s later; when the reply came at
once after the turn's last tool result, that result was not on file either
(in so young a session the whole file was missing), so a ``[NEXT_COMMAND]``
block printed by the turn's last call was not seen, ``hooks/report-check.sh``
exited at its marker grep, and a report without the commands passed.

So the ``Stop`` check waits, at most ``report_check.SETTLE_BUDGET_S``, for the
reply its input names (``last_assistant_message``) to be on file. These tests
write the missing entries from a background thread while the hook waits.
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

sys.path.insert(0, str(SKILL_ROOT))

from tests.test_report_check import (  # noqa: E402
    COMMAND,
    Transcript,
    _fake_uv,
    block_output,
    decision_of,
    report,
)
from tests.test_report_check_session import notification  # noqa: E402
from tests.test_wakeup_check import DISPATCH, PLAN  # noqa: E402
from utils import report_check  # noqa: E402

#: How long after the hook starts the background thread writes.
LATE_S = 0.3

#: The `util next-command` call a stopped implement's report step runs.
LOOKUP = f'{DISPATCH} util next-command "{PLAN}" --after implement --host claude-code'


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
