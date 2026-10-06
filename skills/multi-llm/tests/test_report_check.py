#!/usr/bin/env python3
"""The Claude Code report check: the plugin's ``Stop`` hook
(plans/next-phase-command.md T11).

``hooks/report-check.sh`` reads the hook input, lets a turn through at once
when ``stop_hook_active`` is true or the transcript never printed the block,
and otherwise runs ``multi_llm.py util report-check``
(``utils/report_check.py``). That compares the turn's final message with the
last ``[NEXT_COMMAND]`` / ``[WORKFLOW_COMPLETE]`` block the turn's tool results
printed and, on a miss, prints one ``{"decision": "block", "reason": …}`` line.
Every error is a silent pass.

The transcripts here are fixtures in the Claude Code JSONL shape: one object per
line, ``type`` ``user``/``assistant``, ``message.content`` a string or a list of
``text`` / ``tool_use`` / ``tool_result`` blocks.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
HOOKS_JSON = REPO_ROOT / "hooks" / "hooks.json"
REPORT_CHECK_SH = REPO_ROOT / "hooks" / "report-check.sh"

sys.path.insert(0, str(SKILL_ROOT))

import multi_llm  # noqa: E402
from utils import cli_common, report_check, unattended  # noqa: E402

PLAN = "plans/my plan.md"
COMMAND = "/multi-llm:multi-llm --generate-tasks 'plans/my plan.md'"
#: The same command with one character dropped: the closing quote.
ALTERED = "/multi-llm:multi-llm --generate-tasks 'plans/my plan.md"
OPTIONAL = "/multi-llm:multi-llm --review-plan 'plans/my plan.md'"
WHY = "optional: have the models review the plan again"


# ---------------------------------------------------------------------------
# Fixture transcripts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

class TestCopiedExactlyPasses:

    def test_the_command_copied_exactly(self, tmp_path, capsys):
        path = (Transcript().prompt("/multi-llm:multi-llm --generate-tasks 'plans/my plan.md'")
                .bash(block_output()).think().say(report(COMMAND))
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_primary_and_optional_copied(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output(optional=[(WHY, OPTIONAL)]))
                .say(report(COMMAND, OPTIONAL)).write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_a_fence_nested_in_a_list_item_counts(self, tmp_path, capsys):
        message = f"### Next Steps\n\n- Run:\n  ```bash\n  {COMMAND}\n  ```\n"
        path = (Transcript().prompt("go").bash(block_output()).say(message)
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_a_tool_result_given_as_text_blocks(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output(), as_list=True)
                .say(report(COMMAND)).write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_the_final_message_spans_several_entries(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).think()
                .say("## Summary\n\nDone.").say(report(COMMAND))
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""


class TestAMissOrAnAlteredCopyBlocks:

    def test_missing_blocks_and_names_the_command(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output())
                .say("## Summary\n\nGenerated 7 tasks.").write(tmp_path / "t.jsonl"))
        decision = decision_of(run_check(capsys, path))
        assert f"```\n{COMMAND}\n```" in decision["reason"]
        assert "copied character for character" in decision["reason"]
        assert "do not run it" in decision["reason"]

    def test_a_dropped_quote_blocks_and_names_the_real_command(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output())
                .say(report(ALTERED)).write(tmp_path / "t.jsonl"))
        decision = decision_of(run_check(capsys, path))
        assert f"```\n{COMMAND}\n```" in decision["reason"]

    def test_a_command_in_prose_is_not_a_fenced_copy(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output())
                .say(f"Next, run `{COMMAND}`.").write(tmp_path / "t.jsonl"))
        decision_of(run_check(capsys, path))

    def test_a_missing_optional_is_named_alone(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output(optional=[(WHY, OPTIONAL)]))
                .say(report(COMMAND)).write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        missing, _, others = reason.partition("Carry the block's other command")
        assert "Missing command:" in missing
        assert f"```\n{OPTIONAL}\n```" in missing
        assert COMMAND not in missing
        # The reply replaces the final message, so the one already copied is
        # asked for again, unchanged.
        assert f"```\n{COMMAND}\n```" in others

    def test_a_why_line_is_never_a_command(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output(optional=[(WHY, OPTIONAL)]))
                .say(report(COMMAND, OPTIONAL)).write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""
        blocks = report_check.find_blocks(block_output(optional=[(WHY, OPTIONAL)]))
        assert blocks[-1].commands() == [COMMAND, OPTIONAL]

    def test_the_last_block_of_the_turn_is_the_one_checked(self, tmp_path, capsys):
        later = "/multi-llm:multi-llm --review-tasks 'plans/my plan.md'"
        path = (Transcript().prompt("go").bash(block_output()).bash(block_output(later))
                .say(report(COMMAND)).write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert later in reason and COMMAND not in reason

    def test_a_meta_entry_mid_turn_does_not_end_the_turn(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output())
                .meta("Base directory for this skill: /x").other("attachment")
                .say("Done.").write(tmp_path / "t.jsonl"))
        decision_of(run_check(capsys, path))


class TestOnlyThisTurnIsChecked:

    def test_a_block_printed_only_in_an_earlier_turn_passes(self, tmp_path, capsys):
        path = (Transcript().prompt("generate the tasks").bash(block_output())
                .say(report(COMMAND)).prompt("now explain task 3")
                .say("Task 3 adds the retry loop.").write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_an_earlier_uncopied_block_passes_too(self, tmp_path, capsys):
        path = (Transcript().prompt("generate the tasks").bash(block_output())
                .say("Done.").prompt("thanks", as_list=True)
                .say("You're welcome.").write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_an_indented_quote_of_the_block_is_not_a_phase_result(self, tmp_path, capsys):
        quoted = f"   ```\n   [NEXT_COMMAND] {COMMAND}\n   ```\n"
        path = (Transcript().prompt("read the plan").bash(quoted)
                .say("The plan describes the block.").write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""


class TestWorkflowComplete:

    def test_with_the_sentence_it_passes(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(COMPLETE_OUTPUT)
                .say("## Summary\n\n### Next Steps\n\nWorkflow complete.")
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_without_it_blocks(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(COMPLETE_OUTPUT)
                .say("## Summary\n\nAll phases are done.").write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert "[WORKFLOW_COMPLETE]" in reason
        assert '"Workflow complete."' in reason


class TestTheHookInput:

    def test_stop_hook_active_passes_whatever_the_message(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "t.jsonl"))
        hook = {"session_id": "s1", "transcript_path": str(path),
                "hook_event_name": "Stop", "stop_hook_active": True}
        assert run_check(capsys, path, hook=hook, tmp_path=tmp_path) == ""

    def test_the_transcript_path_comes_from_the_hook_input(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "my session.jsonl"))
        hook = {"session_id": "s1", "transcript_path": str(path),
                "hook_event_name": "Stop", "stop_hook_active": False}
        decision_of(run_check(capsys, path, hook=hook, tmp_path=tmp_path))

    def test_last_assistant_message_covers_a_stale_transcript(self, tmp_path, capsys):
        """The hook can fire before the transcript's last line is written."""
        path = (Transcript().prompt("go").bash(block_output())
                .write(tmp_path / "t.jsonl"))
        hook = {"transcript_path": str(path), "stop_hook_active": False,
                "last_assistant_message": report(COMMAND)}
        assert run_check(capsys, path, hook=hook, tmp_path=tmp_path) == ""

    def test_a_missing_copy_in_both_still_blocks(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "t.jsonl"))
        hook = {"transcript_path": str(path), "stop_hook_active": False,
                "last_assistant_message": "Done."}
        decision_of(run_check(capsys, path, hook=hook, tmp_path=tmp_path))

    def test_a_transcript_that_has_not_written_the_reply_passes_without_the_field(
            self, tmp_path, capsys):
        """Measured live on 2.1.289 (2026-10-05): when the Stop hook runs, the
        transcript still ends at the turn's last tool result. With no
        ``last_assistant_message`` the check cannot see the reply at all, and a
        reply it cannot see is never a miss."""
        path = (Transcript().prompt("go").bash(block_output())
                .write(tmp_path / "t.jsonl"))
        hook = {"transcript_path": str(path), "stop_hook_active": False}
        assert run_check(capsys, path, hook=hook, tmp_path=tmp_path) == ""

    def test_a_reply_written_only_up_to_its_thinking_passes_too(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).think()
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_the_field_alone_decides_while_the_reply_is_unwritten(self, tmp_path, capsys):
        """The live input, field for field (2.1.289, headless ``claude -p``)."""
        path = (Transcript().prompt("go").bash(block_output())
                .write(tmp_path / "t.jsonl"))
        hook = {"session_id": "s1", "transcript_path": str(path), "cwd": str(tmp_path),
                "prompt_id": "p1", "permission_mode": "auto",
                "effort": {"level": "medium"}, "hook_event_name": "Stop",
                "stop_hook_active": False, "last_assistant_message": "done",
                "background_tasks": [], "session_crons": []}
        decision = decision_of(run_check(capsys, path, hook=hook, tmp_path=tmp_path))
        assert f"```\n{COMMAND}\n```" in decision["reason"]
        hook["last_assistant_message"] = report(COMMAND)
        assert run_check(capsys, path, hook=hook, tmp_path=tmp_path) == ""

    def test_a_copy_in_an_earlier_block_of_a_split_reply_blocks_once(
            self, tmp_path, capsys):
        """The documented limit (hooks review of fix round 10): the field is
        the reply's last text block alone, so a command copied into an earlier
        block reads as missing. The block costs one turn, no more: the hook's
        next run carries ``stop_hook_active`` and passes."""
        path = (Transcript().prompt("go").bash(block_output())
                .write(tmp_path / "t.jsonl"))
        hook = {"transcript_path": str(path), "stop_hook_active": False,
                "last_assistant_message": "Let me know if you want anything else."}
        decision_of(run_check(capsys, path, hook=hook, tmp_path=tmp_path))
        hook["stop_hook_active"] = True
        assert run_check(capsys, path, hook=hook, tmp_path=tmp_path) == ""

    def test_the_hook_input_on_stdin(self, tmp_path, capsys, monkeypatch):
        import io

        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "t.jsonl"))
        payload = json.dumps({"transcript_path": str(path), "stop_hook_active": False})
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload.encode())))
        assert report_check.main(["--hook-input", "-"]) == 0
        decision_of(capsys.readouterr().out)


class TestEveryErrorIsASilentPass:

    @pytest.mark.parametrize("body", [
        "this is not json\n",
        '{"type": "user", "message": {"content": "go"}}\n[1, 2, 3]\n',
        '{"type": "user", "message": {"content": "go"}}\n{"type": "user", "mess\n',
        "\x00\xff garbage [NEXT_COMMAND] x\n",
    ])
    def test_a_malformed_transcript(self, tmp_path, capsys, body):
        path = tmp_path / "t.jsonl"
        path.write_text(body, encoding="utf-8", errors="replace")
        assert run_check(capsys, path) == ""

    def test_a_malformed_line_inside_the_turn(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).write(tmp_path / "t.jsonl"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write("{not json\n")
        assert run_check(capsys, path) == ""

    def test_a_missing_transcript(self, tmp_path, capsys):
        assert run_check(capsys, tmp_path / "nope.jsonl") == ""

    @pytest.mark.parametrize("hook_text", ["not json", "[1]", "{}",
                                           '{"transcript_path": 7}'])
    def test_a_bad_hook_input(self, tmp_path, capsys, hook_text):
        hook_file = tmp_path / "hook.json"
        hook_file.write_text(hook_text, encoding="utf-8")
        assert report_check.main(["--hook-input", str(hook_file)]) == 0
        assert capsys.readouterr() == ("", "")

    @pytest.mark.parametrize("argv", [[], ["--bogus"], ["--transcript"],
                                      ["--hook-input", "/nonexistent/x.json"]])
    def test_bad_arguments(self, capsys, argv):
        assert report_check.main(argv) == 0
        assert capsys.readouterr() == ("", "")

    def test_an_api_error_ending_is_not_checked(self, tmp_path, capsys):
        transcript = Transcript().prompt("go").bash(block_output())
        transcript._add({"type": "assistant", "isApiErrorMessage": True, "message": {
            "role": "assistant", "content": [{"type": "text", "text": "API Error: 500"}]}})
        assert run_check(capsys, transcript.write(tmp_path / "t.jsonl")) == ""

    def test_import_is_side_effect_free_and_help_prints(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            report_check.main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "report-check" in out and "--hook-input" in out


class TestThroughTheDispatcher:

    def test_the_decision_line_is_selectable_beside_the_result_line(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "t.jsonl"))
        assert multi_llm.main(["util", "report-check", "--transcript", str(path)]) == 0
        lines = capsys.readouterr().out.splitlines()
        decisions = [line for line in lines if line.startswith(report_check.DECISION_PREFIX)]
        assert len(decisions) == 1
        assert json.loads(decisions[0])["decision"] == "block"
        assert lines[-1].startswith(cli_common.RESULT_MARKER)
        result = json.loads(lines[-1][len(cli_common.RESULT_MARKER):])
        assert result["exit"] == 0 and result["mode"] == "util report-check"
        assert "user_next_step" not in result

    def test_a_pass_prints_only_the_result_line(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(block_output()).say(report(COMMAND))
                .write(tmp_path / "t.jsonl"))
        assert multi_llm.main(["util", "report-check", "--transcript", str(path)]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1 and lines[0].startswith(cli_common.RESULT_MARKER)

    def test_it_is_a_registered_util_command(self):
        module, attr, summary = multi_llm.UTIL_COMMANDS["report-check"]
        assert (module, attr) == ("utils.report_check", "main")
        assert summary


# ---------------------------------------------------------------------------
# The hook wiring
# ---------------------------------------------------------------------------

#: The SessionStart entry as it shipped before the Stop hook joined it.
SESSION_START = [
    {
        "matcher": "startup",
        "hooks": [
            {
                "type": "command",
                "command": "\"${CLAUDE_PLUGIN_ROOT}/hooks/doctor.sh\"",
                "timeout": 30,
            }
        ],
    }
]


class TestTheHookFile:

    def test_it_parses_with_session_start_unchanged_and_a_stop_entry(self):
        data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
        assert list(data["hooks"]) == ["SessionStart", "Stop"]
        assert data["hooks"]["SessionStart"] == SESSION_START

    def test_the_session_start_text_is_byte_identical(self):
        text = HOOKS_JSON.read_text(encoding="utf-8")
        assert (
            '    "SessionStart": [\n'
            "      {\n"
            '        "matcher": "startup",\n'
            '        "hooks": [\n'
            "          {\n"
            '            "type": "command",\n'
            '            "command": "\\"${CLAUDE_PLUGIN_ROOT}/hooks/doctor.sh\\"",\n'
            '            "timeout": 30\n'
            "          }\n"
            "        ]\n"
            "      }\n"
            "    ]"
        ) in text

    def test_the_stop_entry_runs_the_report_check_script(self):
        entries = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))["hooks"]["Stop"]
        assert len(entries) == 1
        hooks = entries[0]["hooks"]
        assert len(hooks) == 1
        hook = hooks[0]
        assert hook["type"] == "command"
        assert hook["command"] == "\"${CLAUDE_PLUGIN_ROOT}/hooks/report-check.sh\""
        assert isinstance(hook.get("timeout"), int) and hook["timeout"] >= 10


class TestTheScript:

    def test_it_is_executable(self):
        if os.name == "nt":
            pytest.skip("POSIX mode bits")
        assert REPORT_CHECK_SH.stat().st_mode & stat.S_IXUSR

    def test_the_cheap_exits_come_before_the_uv_lookup(self):
        text = REPORT_CHECK_SH.read_text(encoding="utf-8")
        loop_guard = text.index('"stop_hook_active"')
        prefilter = text.index("grep -q -F -e '[NEXT_COMMAND' -e '[WORKFLOW_COMPLETE]'")
        uv_lookup = text.index('uv=""')
        assert loop_guard < prefilter < uv_lookup

    def test_it_runs_the_util_with_the_owned_uv_flags(self):
        text = REPORT_CHECK_SH.read_text(encoding="utf-8")
        assert f"run {' '.join(unattended.DEFAULT_UV_RUN_ARGS)} --project" in text
        assert "util report-check --hook-input -" in text
        assert "grep '^{\"decision\"'" in text
        assert "python3 -m" not in text and "python -m" not in text


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


def _run_hook(tmp_path: Path, hook_input: str, fake_uv: Optional[Path] = None):
    log = tmp_path / "argv.txt"
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CLAUDE_PLUGIN_ROOT": str(REPO_ROOT),
        "ARGV_LOG": str(log),
        "REAL_PYTHON": sys.executable,
    }
    if fake_uv is not None:
        env["MULTI_LLM_UV"] = str(fake_uv)
    completed = subprocess.run(
        ["/bin/sh", str(REPORT_CHECK_SH)], input=hook_input, env=env,
        capture_output=True, text=True, timeout=60,
    )
    return completed, log


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")
class TestTheScriptEndToEnd:

    def test_stop_hook_active_exits_0_silently_before_uv(self, tmp_path):
        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "t.jsonl"))
        fake = _fake_uv(tmp_path)
        hook = json.dumps({"session_id": "s1", "transcript_path": str(path),
                           "hook_event_name": "Stop", "stop_hook_active": True})
        completed, log = _run_hook(tmp_path, hook, fake)
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists(), "uv must not run on a turn already blocked once"

    def test_a_transcript_without_the_marker_exits_before_uv(self, tmp_path):
        path = (Transcript().prompt("hello").say("Hi.").write(tmp_path / "t.jsonl"))
        fake = _fake_uv(tmp_path)
        hook = json.dumps({"transcript_path": str(path), "stop_hook_active": False})
        completed, log = _run_hook(tmp_path, hook, fake)
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    @pytest.mark.parametrize("hook", [
        "", "not json", '{"stop_hook_active": false}',
        '{"transcript_path": "/nonexistent/t.jsonl", "stop_hook_active": false}',
    ])
    def test_no_transcript_exits_0_silently(self, tmp_path, hook):
        fake = _fake_uv(tmp_path)
        completed, log = _run_hook(tmp_path, hook, fake)
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    def test_a_miss_prints_exactly_one_json_object(self, tmp_path):
        """The whole of stdout is the decision: Claude Code parses it as JSON
        only when it is nothing else, so the result line must not reach it."""
        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "my session.jsonl"))
        fake = _fake_uv(tmp_path)
        hook = json.dumps({"session_id": "s1", "transcript_path": str(path),
                           "cwd": str(tmp_path), "hook_event_name": "Stop",
                           "stop_hook_active": False})
        completed, log = _run_hook(tmp_path, hook, fake)
        assert completed.returncode == 0, completed.stderr
        decision = json.loads(completed.stdout)
        assert decision["decision"] == "block"
        assert COMMAND in decision["reason"]
        assert cli_common.RESULT_MARKER not in completed.stdout
        skill = str(REPO_ROOT / "skills" / "multi-llm")
        assert log.read_text(encoding="utf-8").splitlines() == [
            "run", *unattended.DEFAULT_UV_RUN_ARGS, "--project", skill, "--",
            "python", f"{skill}/multi_llm.py", "util", "report-check",
            "--hook-input", "-",
        ]

    def test_a_copied_command_prints_nothing(self, tmp_path):
        path = (Transcript().prompt("go").bash(block_output()).say(report(COMMAND))
                .write(tmp_path / "t.jsonl"))
        fake = _fake_uv(tmp_path)
        hook = json.dumps({"transcript_path": str(path), "stop_hook_active": False})
        completed, log = _run_hook(tmp_path, hook, fake)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == ""
        assert log.exists(), "the check itself must have run"

    def test_a_failing_util_exits_0_silently(self, tmp_path):
        path = (Transcript().prompt("go").bash(block_output()).say("Done.")
                .write(tmp_path / "t.jsonl"))
        broken = tmp_path / "broken-uv"
        broken.write_text("#!/bin/sh\necho 'boom' >&2\necho 'not json'\nexit 7\n",
                          encoding="utf-8")
        broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
        hook = json.dumps({"transcript_path": str(path), "stop_hook_active": False})
        completed, _log = _run_hook(tmp_path, hook, broken)
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""


class TestTheAnswersCommandIsCarriedToo:
    """Fix round 14: a stopped implement's `util next-command` prints, beside
    the resume, one command carrying every answer the stop waits for; the
    report has to carry it like the others."""

    RESUME = "/multi-llm:multi-llm --implement plans/check-gate.md --resume"
    ANSWERS = (
        "/multi-llm:multi-llm --implement plans/check-gate.md --resume "
        "--human-task T002=done --human-task T005=skip"
    )
    ANSWERS_WHY = "or, in one command: accept the findings of T002 (then T007 runs)"

    def _output(self) -> str:
        return (
            f"[NEXT_COMMAND] {self.RESUME}\n"
            f"[NEXT_COMMAND_ANSWERS_WHY] {self.ANSWERS_WHY}\n"
            f"[NEXT_COMMAND_ANSWERS] {self.ANSWERS}\n"
            'MULTI_LLM_RESULT: {"mode": "util next-command", "exit": 0}\n'
        )

    def test_the_block_holds_it_and_its_why_is_no_command(self):
        block = report_check.find_blocks(self._output())[-1]
        assert block.commands() == [self.RESUME, self.ANSWERS]

    def test_a_report_without_it_blocks_naming_it(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(self._output())
                .say(report(self.RESUME)).write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        missing, _, _others = reason.partition("Carry the block's other command")
        assert f"```\n{self.ANSWERS}\n```" in missing
        assert "[NEXT_COMMAND_ANSWERS]" in reason

    def test_both_copied_passes(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(self._output())
                .say(report(self.RESUME, self.ANSWERS)).write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""
