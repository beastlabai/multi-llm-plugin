"""Transcript entries that land after the hook reads them: the settle window's inputs.

``tool_use`` / ``tool_result`` / ``reply`` build transcript entries and
``line`` serializes one; ``write_later`` appends text from a thread after a
delay, as Claude Code does; ``stop_input`` and ``wakeup_input`` are the hook
inputs, and ``denied`` reads a block decision. No fixtures.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tests.support.wakeup_world import DISPATCH, OWN_CALL, PLAN


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


def stop_input(path: Path, last_message: Optional[str], **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "session_id": "s1", "transcript_path": str(path), "cwd": str(path.parent),
        "hook_event_name": "Stop", "stop_hook_active": False,
    }
    if last_message is not None:
        payload["last_assistant_message"] = last_message
    payload.update(extra)
    return payload


def wakeup_input(path: Path, tool: str = "ScheduleWakeup") -> Dict[str, Any]:
    return {"session_id": "s1", "transcript_path": str(path), "cwd": str(path.parent),
            "hook_event_name": "PreToolUse", "tool_name": tool,
            "tool_input": {"delaySeconds": 60, "prompt": "check the phase",
                           "reason": "waiting"},
            "tool_use_id": OWN_CALL}


def denied(out: str) -> str:
    decision = json.loads(out)
    specific = decision["hookSpecificOutput"]
    assert specific["permissionDecision"] == "deny"
    return specific["permissionDecisionReason"]
