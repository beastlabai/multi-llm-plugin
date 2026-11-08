"""Claude Code transcripts and the report-check hook, driven in-process.

``Transcript`` builds a JSONL transcript one entry per call; ``block_output``
is a phase result's stdout carrying ``[NEXT_COMMAND]`` lines, ``report`` a
final message with the commands in fenced blocks, ``run_check`` runs
``report_check.main`` on a transcript or a hook input, and ``decision_of``
parses the one block decision it prints. ``_fake_uv`` is a ``uv`` stand-in that
logs its argv and runs the rest on this test's Python. No fixtures: the hook's
settle budget is each test file's own choice (an autouse fixture there).
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import report_check


COMMAND = "/multi-llm:multi-llm --generate-tasks 'plans/my plan.md'"


OPTIONAL = "/multi-llm:multi-llm --review-plan 'plans/my plan.md'"
WHY = "Have the models review the plan again; skipping this keeps the plan as it stands."


class Transcript:
    """Builds a Claude Code JSONL transcript, one entry per call."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self._n = 0

    def _add(self, entry: Dict[str, Any]) -> "Transcript":
        self._n += 1
        entry.setdefault("uuid", f"u{self._n}")
        entry.setdefault("sessionId", "s1")
        self.entries.append(entry)
        return self

    def prompt(self, text: str, as_list: bool = False) -> "Transcript":
        content: Any = [{"type": "text", "text": text}] if as_list else text
        return self._add({"type": "user", "message": {"role": "user", "content": content}})

    def meta(self, text: str) -> "Transcript":
        return self._add({"type": "user", "isMeta": True,
                          "message": {"role": "user", "content": text}})

    def say(self, text: str) -> "Transcript":
        return self._add({"type": "assistant", "message": {
            "role": "assistant", "content": [{"type": "text", "text": text}]}})

    def think(self) -> "Transcript":
        return self._add({"type": "assistant", "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm", "signature": "x"}]}})

    def bash(self, output: str, as_list: bool = False) -> "Transcript":
        tool_id = f"toolu_{self._n + 1}"
        self._add({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": "Bash",
             "input": {"command": "uv run … multi_llm.py util next-command"}}]}})
        inner: Any = [{"type": "text", "text": output}] if as_list else output
        return self._add({"type": "user", "toolUseResult": {"stdout": output},
                          "message": {"role": "user", "content": [
                              {"type": "tool_result", "tool_use_id": tool_id,
                               "content": inner}]}})

    def other(self, kind: str) -> "Transcript":
        return self._add({"type": kind, "content": "noise"})

    def write(self, path: Path) -> Path:
        path.write_text(
            "".join(json.dumps(entry) + "\n" for entry in self.entries),
            encoding="utf-8",
        )
        return path


def block_output(command: str = COMMAND, optional: Optional[List[tuple]] = None) -> str:
    """A phase result's stdout: the block above the result line."""
    lines = ["Tasks written to plans/my plan/tasks/tasks.md", f"[NEXT_COMMAND] {command}"]
    for why, opt in optional or []:
        lines += [f"[NEXT_COMMAND_OPTIONAL_WHY] {why}", f"[NEXT_COMMAND_OPTIONAL] {opt}"]
    lines.append('MULTI_LLM_RESULT: {"mode": "util next-command", "phase": null, '
                 '"state": "done", "exit": 0, "artifacts": [], "next": null}')
    return "\n".join(lines) + "\n"


COMPLETE_OUTPUT = (
    "[WORKFLOW_COMPLETE]\n"
    'MULTI_LLM_RESULT: {"mode": "util next-command", "phase": null, "state": "done", '
    '"exit": 0, "artifacts": [], "next": null}\n'
)


def report(*commands: str, prose: str = "") -> str:
    """A final message ending with each command in its own fenced block."""
    parts = ["## Summary", "Generated 7 tasks.", prose, "### Next Steps"]
    for command in commands:
        parts.append(f"```\n{command}\n```")
    return "\n\n".join(part for part in parts if part)


def run_check(capsys, transcript: Path, hook: Optional[Dict[str, Any]] = None,
              tmp_path: Optional[Path] = None) -> str:
    """``report_check.main`` on a transcript (or a hook input); returns stdout."""
    argv = ["--transcript", str(transcript)]
    if hook is not None:
        assert tmp_path is not None
        hook_file = tmp_path / "hook-input.json"
        hook_file.write_text(json.dumps(hook), encoding="utf-8")
        argv = ["--hook-input", str(hook_file)]
    assert report_check.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out


def decision_of(out: str) -> Dict[str, Any]:
    lines = out.splitlines()
    assert len(lines) == 1, out
    assert lines[0].startswith(report_check.DECISION_PREFIX)
    decision = json.loads(lines[0])
    assert list(decision) == ["decision", "reason"]
    assert decision["decision"] == "block"
    return decision


def _fake_uv(tmp_path: Path) -> Path:
    """A ``uv`` that logs its argv and runs the rest on this test's Python."""
    fake = tmp_path / "bin" / "uv"
    fake.parent.mkdir()
    fake.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$ARGV_LOG\"\n"
        "while [ \"$#\" -gt 0 ] && [ \"$1\" != \"--\" ]; do shift; done\n"
        "shift\n"   # --
        "shift\n"   # python
        "exec \"$REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake
