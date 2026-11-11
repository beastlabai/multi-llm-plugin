"""Transcripts and hook inputs for the wakeup check: a phase launched in the background and its scheduling call.

Built on ``tests.support.transcripts.Transcript``: ``call`` adds one tool call
and its result, ``phase`` a phase's run, ``racing_03`` the racing run 03
transcript up to its ``ScheduleWakeup`` call; ``own_call_entry`` is that call
as the transcript holds it, and ``hook_input`` the hook input for it. No
fixtures: the hook's settle budget is each test file's own choice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from tests.support.transcripts import Transcript


DISPATCH = ('uv run --frozen --no-dev --project "/p/skills/multi-llm" -- python '
            '"/p/skills/multi-llm/multi_llm.py"')
PLAN = "plans/racing-game.md"
PENDING = (
    "[TASK_GENERATION_PENDING] plans/racing-game/generate-tasks/task_generation_tasks.json\n"
    'MULTI_LLM_RESULT: {"mode": "generate-tasks", "phase": "generate-tasks", '
    '"state": "done", "exit": 0, "artifacts": [], "next": "uv run … --reaggregate"}\n'
)


WAKEUP_INPUT = {"delaySeconds": 1200, "reason": "Fallback heartbeat",
                "prompt": f"/multi-llm:multi-llm --generate-tasks {PLAN} --yes"}


def call(transcript: Transcript, command: str, output: str, name: str = "Bash",
         tool_input: Optional[Dict[str, Any]] = None) -> Transcript:
    """One tool call and its result."""
    tool_id = f"toolu_{transcript._n + 1}"
    transcript._add({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": tool_id, "name": name,
         "input": tool_input if tool_input is not None else {"command": command}}]}})
    return transcript._add({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": output}]}})


def phase(transcript: Transcript, mode: str, output: str, extra: str = "") -> Transcript:
    return call(transcript, f'{DISPATCH} {mode} "{PLAN}" --host claude-code --yes{extra}',
                output)


def util(transcript: Transcript, words: str, output: str) -> Transcript:
    return call(transcript, f"{DISPATCH} util {words}", output)


def racing_03(transcript: Transcript) -> Transcript:
    """Racing 03 up to its `ScheduleWakeup` call."""
    transcript.prompt("/multi-llm:multi-llm --generate-tasks plans/racing-game.md --yes")
    call(transcript, "command -v uv", "/usr/bin/uv")
    util(transcript, "instructions generate-tasks", "=== READ THIS FIRST ===\n")
    phase(transcript, "generate-tasks", PENDING)
    call(transcript, "", "Async agent launched successfully.", name="Agent",
         tool_input={"description": "Generate implementation tasks",
                     "subagent_type": "general-purpose", "prompt": "Read the file …"})
    return transcript.say("I've started a subagent to write the task list.")


#: The id of the scheduling call each test's hook input asks about.
OWN_CALL = "toolu_x"


def own_call_entry(tool: str = "ScheduleWakeup",
                   tool_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The scheduling call itself, as the transcript holds it once written."""
    return {"type": "assistant", "message": {"id": "msg_own", "role": "assistant", "content": [
        {"type": "tool_use", "id": OWN_CALL, "name": tool,
         "input": tool_input if tool_input is not None else WAKEUP_INPUT}]}}


def hook_input(path: Path, tool: str = "ScheduleWakeup", **extra: Any) -> Dict[str, Any]:
    """The hook input for a scheduling call, which is appended to the
    transcript at *path* first when it is not on file: Claude Code writes the
    call before the hook decides (fix round 27: the hook waits for it)."""
    payload = {"session_id": "s1", "transcript_path": str(path), "cwd": str(path.parent),
               "permission_mode": "bypassPermissions", "hook_event_name": "PreToolUse",
               "tool_name": tool, "tool_input": WAKEUP_INPUT, "tool_use_id": OWN_CALL}
    payload.update(extra)
    if path.is_file() and f'"{OWN_CALL}"' not in path.read_text(encoding="utf-8"):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(own_call_entry(tool, payload["tool_input"])) + "\n")
    return payload
