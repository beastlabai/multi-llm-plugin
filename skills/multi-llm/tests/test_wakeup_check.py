#!/usr/bin/env python3
"""The Claude Code plugin's ``PreToolUse`` hook: no scheduled wakeup while a
phase runs (fix round 26, live run 11 #2).

Racing 03: after an interim "still in progress" reply, the headless host
called ``ScheduleWakeup`` (1200 s) with a prompt re-running
``--generate-tasks … --yes``; ``claude -p`` stayed alive about 16 minutes after
its final result until the driver killed it, and had the wakeup fired the
phase would have run again. Racing 04 called it mid-run too. The hook
(``hooks/wakeup-check.sh`` → ``util wakeup-check``) refuses ``ScheduleWakeup``
and ``CronCreate`` while the session's transcript shows a phase it started that
has printed no next step yet, and lets every other use of them through.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
HOOKS_JSON = REPO_ROOT / "hooks" / "hooks.json"
WAKEUP_CHECK_SH = REPO_ROOT / "hooks" / "wakeup-check.sh"

sys.path.insert(0, str(SKILL_ROOT))

import multi_llm  # noqa: E402
from tests.test_report_check import Transcript, block_output  # noqa: E402
from tests.test_report_check_session import notification, typed  # noqa: E402
from utils import cli_common, unattended, wakeup_check  # noqa: E402

DISPATCH = ('uv run --frozen --no-dev --project "/p/skills/multi-llm" -- python '
            '"/p/skills/multi-llm/multi_llm.py"')
PLAN = "plans/racing-game.md"
PENDING = (
    "[TASK_GENERATION_PENDING] plans/racing-game/generate-tasks/task_generation_tasks.json\n"
    'MULTI_LLM_RESULT: {"mode": "generate-tasks", "phase": "generate-tasks", '
    '"state": "done", "exit": 0, "artifacts": [], "next": "uv run … --reaggregate"}\n'
)
EMIT = ('MULTI_LLM_RESULT: {"mode": "implement", "phase": "implement", '
        '"state": "done", "exit": 0, "artifacts": [], "next": null}\n')
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


def hook_input(path: Path, tool: str = "ScheduleWakeup", **extra: Any) -> Dict[str, Any]:
    payload = {"session_id": "s1", "transcript_path": str(path), "cwd": str(path.parent),
               "permission_mode": "bypassPermissions", "hook_event_name": "PreToolUse",
               "tool_name": tool, "tool_input": WAKEUP_INPUT, "tool_use_id": "toolu_x"}
    payload.update(extra)
    return payload


def decide(capsys, tmp_path: Path, payload: Dict[str, Any]) -> str:
    hook_file = tmp_path / "hook-input.json"
    hook_file.write_text(json.dumps(payload), encoding="utf-8")
    assert wakeup_check.main(["--hook-input", str(hook_file)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out


def denial(out: str) -> str:
    lines = out.splitlines()
    assert len(lines) == 1, out
    assert lines[0].startswith(wakeup_check.DECISION_PREFIX)
    decision = json.loads(lines[0])
    assert list(decision) == ["hookSpecificOutput"]
    specific = decision["hookSpecificOutput"]
    assert specific["hookEventName"] == "PreToolUse"
    assert specific["permissionDecision"] == "deny"
    return specific["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# While a phase runs
# ---------------------------------------------------------------------------

class TestRefusedWhileAPhaseRuns:

    def test_racing_03_is_refused_with_a_plain_reason(self, tmp_path, capsys):
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        reason = denial(decide(capsys, tmp_path, hook_input(path)))
        assert "ScheduleWakeup is refused while the generate-tasks phase" in reason
        assert "Keep waiting for the phase's subagents" in reason
        assert "completion notification arrives on its own" in reason
        assert "never schedule a phase command to run again" in reason

    def test_cron_create_is_refused_too(self, tmp_path, capsys):
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        reason = denial(decide(capsys, tmp_path, hook_input(
            path, tool="CronCreate",
            tool_input={"cron": "*/5 * * * *", "prompt": "check the subagents"})))
        assert reason.startswith("multi-llm: CronCreate is refused")

    def test_a_subagent_s_own_call_is_refused_too(self, tmp_path, capsys):
        """A subagent's hook input names the main session's transcript."""
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        denial(decide(capsys, tmp_path, hook_input(path, agent_id="a6d24cd3")))

    def test_racing_04_mid_run_is_refused(self, tmp_path, capsys):
        """The review fan-out launched in the background, its validation
        batches dispatched after a notification: still running."""
        transcript = Transcript().prompt("/multi-llm:multi-llm --review-tasks plans/racing-game.md")
        util(transcript, "prereq --plan-file plans/racing-game.md --mode review-tasks",
             '{"prerequisites_met": true}\nMULTI_LLM_RESULT: {"exit": 0}\n')
        phase(transcript, "review-tasks", "Command running in background with ID: bzfwwie97.",
              extra=" --quick > plans/racing-game/review-tasks/orchestrator-run.log 2>&1")
        transcript.say("Launched.")
        notification(transcript, "review-tasks fan-out")
        call(transcript, "cat plans/racing-game/review-tasks/orchestrator-run.log",
             "[VALIDATION_PENDING] validation_tasks.json\n")
        path = transcript.write(tmp_path / "t.jsonl")
        assert "review-tasks phase" in denial(decide(capsys, tmp_path, hook_input(path)))

    def test_a_native_write_phase_handing_its_batches_out_is_running(self, tmp_path, capsys):
        """Implement's native emit exits 0 with `next` null and prints no
        block: the phase runs until the host's closing lookup."""
        transcript = Transcript().prompt("/multi-llm:multi-llm --implement plans/racing-game.md")
        phase(transcript, "implement", "[WRITE_TASKS_FILE] write_tasks.json\n" + EMIT)
        util(transcript, "write-prompt --phase-dir plans/racing-game/implement --task-id T001",
             "[WRITE_PROMPT] T001.prompt.txt\n")
        util(transcript, "batch-done --phase-dir plans/racing-game/implement --task-id T001",
             "[BATCH_DONE] T001 completed\n")
        path = transcript.write(tmp_path / "t.jsonl")
        assert "implement phase" in denial(decide(capsys, tmp_path, hook_input(path)))

    def test_an_alias_and_a_quoted_mode_word_count(self, tmp_path, capsys):
        transcript = Transcript().prompt("go")
        call(transcript, f"PYTHONUNBUFFERED=1 {DISPATCH} 'tasks' {PLAN}", PENDING)
        path = transcript.write(tmp_path / "t.jsonl")
        assert "generate-tasks phase" in denial(decide(capsys, tmp_path, hook_input(path)))

    def test_a_retry_after_a_failed_call_runs_again(self, tmp_path, capsys):
        """Racing 03's `--reaggregate` failed from the wrong directory and the
        host ran it again: the second call is a phase running."""
        transcript = racing_03(Transcript())
        phase(transcript, "generate-tasks", 'ERROR: Plan file not found\nMULTI_LLM_RESULT: '
              '{"mode": "generate-tasks", "exit": 1, "next": null}\n', " --reaggregate")
        call(transcript, f"cd /r && {DISPATCH} generate-tasks {PLAN} --reaggregate",
             "Command running in background with ID: b1.")
        path = transcript.write(tmp_path / "t.jsonl")
        denial(decide(capsys, tmp_path, hook_input(path)))


# ---------------------------------------------------------------------------
# Every other use passes
# ---------------------------------------------------------------------------

class TestEveryOtherUsePasses:

    def test_a_session_that_never_ran_the_dispatcher(self, tmp_path, capsys):
        path = (Transcript().prompt("remind me in an hour to deploy").say("Sure.")
                .write(tmp_path / "t.jsonl"))
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    def test_once_the_phase_printed_its_next_step(self, tmp_path, capsys):
        transcript = racing_03(Transcript())
        notification(transcript, "Generate implementation tasks")
        phase(transcript, "generate-tasks", block_output(), " --reaggregate")
        path = transcript.write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    def test_once_the_lookup_printed_a_stop_s_next_step(self, tmp_path, capsys):
        transcript = Transcript().prompt("/multi-llm:multi-llm --implement plans/racing-game.md")
        phase(transcript, "implement", EMIT)
        util(transcript, "next-command plans/racing-game.md --after implement",
             "[NEXT_COMMAND] /multi-llm:multi-llm --implement plans/racing-game.md --resume\n")
        path = transcript.write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    def test_once_the_workflow_is_complete(self, tmp_path, capsys):
        transcript = Transcript().prompt("go")
        phase(transcript, "apply-code-fixes", "[WORKFLOW_COMPLETE]\nMULTI_LLM_RESULT: {\"exit\": 0}\n",
              " --mark-completed")
        path = transcript.write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    @pytest.mark.parametrize("code", [1, 3])
    def test_a_phase_that_failed_or_stopped_before_it_ran(self, tmp_path, capsys, code):
        transcript = Transcript().prompt("go")
        phase(transcript, "implement",
              f'[PREREQUISITE_UNMET] x\nMULTI_LLM_RESULT: {{"exit": {code}, "next": null}}\n')
        path = transcript.write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    def test_an_answered_question(self, tmp_path, capsys):
        """`ask` prints no next step: its result line ends it."""
        transcript = Transcript().prompt("go")
        phase(transcript, "ask", 'answers written\nMULTI_LLM_RESULT: {"exit": 0, "next": null}\n')
        path = transcript.write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    def test_after_a_prompt_the_user_typed(self, tmp_path, capsys):
        """The user's own request for a wakeup, after the phase they ran."""
        transcript = racing_03(Transcript())
        typed(transcript, "never mind; remind me in an hour to look at it")
        path = transcript.write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    def test_util_and_status_calls_alone_start_nothing(self, tmp_path, capsys):
        transcript = Transcript().prompt("how far is it?")
        call(transcript, f"{DISPATCH} status {PLAN} --host claude-code --view run",
             'MULTI_LLM_RESULT: {"mode": "review-plan", "exit": 0}\n')
        util(transcript, f"next-command {PLAN} --after review-plan", "no next step\n")
        call(transcript, "grep -n multi_llm.py implement plans/x.md", "")
        path = transcript.write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    @pytest.mark.parametrize("flags", [" --help", " -h", " --dry-run", " --dry-run-subagents"])
    def test_a_preview_of_a_phase_starts_nothing(self, tmp_path, capsys, flags):
        transcript = Transcript().prompt("what would implement do?")
        phase(transcript, "implement", "usage: …\n", flags)
        path = transcript.write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    def test_a_preview_chained_before_a_real_call_does_not_hide_it(self):
        command = (f"{DISPATCH} implement {PLAN} --help; "
                   f"{DISPATCH} generate-tasks {PLAN} --yes")
        assert wakeup_check.phase_mode(command) == "generate-tasks"

    @pytest.mark.parametrize("tool", ["Bash", "CronDelete", "Agent"])
    def test_a_tool_this_does_not_watch(self, tmp_path, capsys, tool):
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        assert decide(capsys, tmp_path, hook_input(path, tool=tool)) == ""


class TestEveryErrorAllows:

    @pytest.mark.parametrize("body", [
        "this is not json\n",
        '{"type": "user", "message": {"content": "go"}}\n[1, 2, 3]\n',
    ])
    def test_a_malformed_transcript(self, tmp_path, capsys, body):
        path = tmp_path / "t.jsonl"
        path.write_text(body + f'"{DISPATCH} implement {PLAN}"\n', encoding="utf-8")
        assert decide(capsys, tmp_path, hook_input(path)) == ""

    def test_a_missing_transcript(self, tmp_path, capsys):
        assert decide(capsys, tmp_path, hook_input(tmp_path / "nope.jsonl")) == ""

    def test_a_line_still_being_written_is_left_out(self, tmp_path, capsys):
        """The hook reads the transcript while the session appends to it: a
        last line with no newline yet is not an unreadable transcript."""
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{"type": "assistant", "message": {"content": [{"type": "tool_')
        denial(decide(capsys, tmp_path, hook_input(path)))

    @pytest.mark.parametrize("hook_text", ["not json", "[1]", "{}",
                                           '{"tool_name": "ScheduleWakeup"}',
                                           '{"tool_name": "ScheduleWakeup", "transcript_path": 7}'])
    def test_a_bad_hook_input(self, tmp_path, capsys, hook_text):
        hook_file = tmp_path / "hook.json"
        hook_file.write_text(hook_text, encoding="utf-8")
        assert wakeup_check.main(["--hook-input", str(hook_file)]) == 0
        assert capsys.readouterr() == ("", "")

    @pytest.mark.parametrize("argv", [[], ["--bogus"], ["--hook-input"],
                                      ["--hook-input", "/nonexistent/x.json"]])
    def test_bad_arguments(self, capsys, argv):
        assert wakeup_check.main(argv) == 0
        assert capsys.readouterr() == ("", "")

    def test_help_prints(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            wakeup_check.main(["--help"])
        assert excinfo.value.code == 0
        assert "wakeup-check" in capsys.readouterr().out


class TestThroughTheDispatcher:

    def test_it_is_a_registered_util_command(self):
        module, attr, summary = multi_llm.UTIL_COMMANDS["wakeup-check"]
        assert (module, attr) == ("utils.wakeup_check", "main")
        assert summary

    def test_the_decision_line_is_selectable_beside_the_result_line(self, tmp_path, capsys):
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        hook_file = tmp_path / "hook.json"
        hook_file.write_text(json.dumps(hook_input(path)), encoding="utf-8")
        assert multi_llm.main(["util", "wakeup-check", "--hook-input", str(hook_file)]) == 0
        lines = capsys.readouterr().out.splitlines()
        decisions = [line for line in lines if line.startswith(wakeup_check.DECISION_PREFIX)]
        assert len(decisions) == 1
        assert lines[-1].startswith(cli_common.RESULT_MARKER)


# ---------------------------------------------------------------------------
# The host is told first
# ---------------------------------------------------------------------------

class TestTheShimSaysSo:
    """SKILL.md is what the Claude Code host reads on every phase: it says to
    wait for the subagents in the session and never to schedule a way back,
    naming the tools the hook refuses."""

    @staticmethod
    def _paragraph() -> str:
        text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        paragraphs = [p for p in text.split("\n\n") if "ScheduleWakeup" in p]
        assert len(paragraphs) == 1, "one SKILL.md paragraph names the scheduling tools"
        return " ".join(paragraphs[0].split())

    def test_it_names_every_tool_the_hook_refuses_and_the_loop(self):
        paragraph = self._paragraph()
        for tool in wakeup_check.SCHEDULING_TOOLS:
            assert f"`{tool}`" in paragraph
        assert "`/loop`" in paragraph
        assert "never schedule a phase command to run again" in paragraph

    def test_it_says_how_to_wait(self):
        paragraph = self._paragraph()
        assert "wait in this session for what it started" in paragraph
        assert "completion notification arrives on its own" in paragraph

    def test_it_sends_a_batch_back_rather_than_editing_it(self):
        """Racing 08's host sent a batch back with `SendMessage` when its
        diff broke the game; the shim makes that the rule."""
        paragraph = self._paragraph()
        assert "send the batch back to the subagent that ran it (`SendMessage`)" in paragraph
        assert "never edit a batch's files yourself" in paragraph


# ---------------------------------------------------------------------------
# The hook wiring
# ---------------------------------------------------------------------------

class TestTheHookFile:

    def test_the_pre_tool_use_entry_runs_the_script_for_the_two_tools(self):
        data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
        entries = data["hooks"]["PreToolUse"]
        assert len(entries) == 1
        assert entries[0]["matcher"] == "|".join(wakeup_check.SCHEDULING_TOOLS)
        hooks = entries[0]["hooks"]
        assert len(hooks) == 1
        assert hooks[0]["type"] == "command"
        assert hooks[0]["command"] == "\"${CLAUDE_PLUGIN_ROOT}/hooks/wakeup-check.sh\""
        assert isinstance(hooks[0].get("timeout"), int) and hooks[0]["timeout"] >= 10

    def test_the_script_checks_the_same_two_tools(self):
        text = WAKEUP_CHECK_SH.read_text(encoding="utf-8")
        assert '"(' + "|".join(wakeup_check.SCHEDULING_TOOLS) + ')"' in text


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


def _run_hook(tmp_path: Path, hook_text: str, fake_uv: Optional[Path] = None):
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
        ["/bin/sh", str(WAKEUP_CHECK_SH)], input=hook_text, env=env,
        capture_output=True, text=True, timeout=60,
    )
    return completed, log


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")
class TestTheScriptEndToEnd:

    def test_it_is_executable(self):
        assert WAKEUP_CHECK_SH.stat().st_mode & stat.S_IXUSR

    def test_a_refusal_is_exactly_one_json_object(self, tmp_path):
        path = racing_03(Transcript()).write(tmp_path / "my session.jsonl")
        completed, log = _run_hook(tmp_path, json.dumps(hook_input(path)), _fake_uv(tmp_path))
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr == ""
        decision = json.loads(completed.stdout)
        assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert cli_common.RESULT_MARKER not in completed.stdout
        skill = str(REPO_ROOT / "skills" / "multi-llm")
        assert log.read_text(encoding="utf-8").splitlines() == [
            "run", *unattended.DEFAULT_UV_RUN_ARGS, "--project", skill, "--",
            "python", f"{skill}/multi_llm.py", "util", "wakeup-check",
            "--hook-input", "-",
        ]

    def test_a_finished_phase_prints_nothing(self, tmp_path):
        transcript = racing_03(Transcript())
        phase(transcript, "generate-tasks", block_output(), " --reaggregate")
        path = transcript.write(tmp_path / "t.jsonl")
        completed, log = _run_hook(tmp_path, json.dumps(hook_input(path)), _fake_uv(tmp_path))
        assert completed.returncode == 0 and completed.stdout == ""
        assert log.exists(), "the check itself must have run"

    def test_another_tool_exits_before_uv(self, tmp_path):
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        completed, log = _run_hook(
            tmp_path, json.dumps(hook_input(path, tool="Bash")), _fake_uv(tmp_path))
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    def test_a_session_that_never_ran_the_dispatcher_exits_before_uv(self, tmp_path):
        path = Transcript().prompt("remind me later").say("Sure.").write(tmp_path / "t.jsonl")
        completed, log = _run_hook(tmp_path, json.dumps(hook_input(path)), _fake_uv(tmp_path))
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    @pytest.mark.parametrize("hook_text", [
        "", "not json", '{"tool_name": "ScheduleWakeup"}',
        '{"tool_name": "ScheduleWakeup", "transcript_path": "/nonexistent/t.jsonl"}',
    ])
    def test_no_transcript_exits_0_silently(self, tmp_path, hook_text):
        completed, log = _run_hook(tmp_path, hook_text, _fake_uv(tmp_path))
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    def test_without_uv_it_allows_silently(self, tmp_path):
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        completed, _log = _run_hook(tmp_path, json.dumps(hook_input(path)))
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""

    def test_a_failing_util_allows_silently(self, tmp_path):
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        broken = tmp_path / "broken-uv"
        broken.write_text("#!/bin/sh\necho 'boom' >&2\necho 'not json'\nexit 7\n",
                          encoding="utf-8")
        broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
        completed, _log = _run_hook(tmp_path, json.dumps(hook_input(path)), broken)
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
