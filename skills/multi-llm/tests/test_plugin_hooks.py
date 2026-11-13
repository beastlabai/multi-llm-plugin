#!/usr/bin/env python3
"""The Claude Code plugin's `hooks/hooks.json` (D6 step 5).

One `SessionStart` command hook runs `multi_llm.py doctor --warn-only` through
`hooks/doctor.sh` — advisory, never blocking: `--warn-only` exits 0 whatever it
finds, and the script itself exits 0 on every path, including a machine with no
`uv` (a SessionStart hook that exited 2 would block the session). The manifest
at `.claude-plugin/plugin.json` points at the file explicitly. The script drops
the dispatcher's `MULTI_LLM_RESULT` line and forwards everything else, and
`hooks/wakeup-check.sh` leaves uv alone unless a Bash call ran the dispatcher.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
HOOKS_JSON = REPO_ROOT / "hooks" / "hooks.json"
DOCTOR_SH = REPO_ROOT / "hooks" / "doctor.sh"
MANIFEST = REPO_ROOT / ".claude-plugin" / "plugin.json"

sys.path.insert(0, str(SKILL_ROOT))

import multi_llm  # noqa: E402
from tests.support.transcripts import Transcript  # noqa: E402
from tests.support.wakeup_world import hook_input, phase  # noqa: E402
from utils import cli_common, doctor, unattended  # noqa: E402


def _hook_entries():
    data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    return data["hooks"]["SessionStart"]


class TestTheHookFile:

    def test_it_is_valid_json_naming_session_start(self):
        data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
        assert list(data) == ["hooks"]
        # `Stop` is the report check (plans/next-phase-command.md T11), pinned
        # in tests/test_report_check.py; `PreToolUse` refuses a scheduled
        # wakeup while a phase runs (fix round 26), pinned in
        # tests/test_wakeup_check.py.
        assert list(data["hooks"]) == ["SessionStart", "Stop", "PreToolUse"]

    def test_the_one_hook_is_a_command_running_the_doctor_script(self):
        entries = _hook_entries()
        assert len(entries) == 1
        hooks = entries[0]["hooks"]
        assert len(hooks) == 1
        hook = hooks[0]
        assert hook["type"] == "command"
        assert "${CLAUDE_PLUGIN_ROOT}/hooks/doctor.sh" in hook["command"]
        assert isinstance(hook.get("timeout"), int) and hook["timeout"] >= 10

    def test_the_manifest_points_at_it(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["hooks"] == "./hooks/hooks.json"
        assert (REPO_ROOT / manifest["hooks"]).resolve() == HOOKS_JSON.resolve()


class TestTheDoctorScript:

    def test_it_calls_doctor_warn_only_with_the_owned_uv_flags(self):
        text = DOCTOR_SH.read_text(encoding="utf-8")
        assert 'multi_llm.py" doctor --warn-only 2>/dev/null' in text
        assert f"run {' '.join(unattended.DEFAULT_UV_RUN_ARGS)} --project" in text
        assert "MULTI_LLM_UV" in text
        # Never a bare-python fallback (D6c).
        assert "python3 -m" not in text and "python -m" not in text

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
    def test_it_is_executable(self):
        assert DOCTOR_SH.stat().st_mode & stat.S_IXUSR

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")
    def test_it_is_silent_and_exits_0_without_uv(self, tmp_path):
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "CLAUDE_PLUGIN_ROOT": str(REPO_ROOT),
        }
        completed = subprocess.run(
            ["/bin/sh", str(DOCTOR_SH)], env=env, capture_output=True, text=True,
            timeout=30,
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")
    def test_it_is_silent_and_exits_0_outside_a_plugin_tree(self, tmp_path):
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
               "CLAUDE_PLUGIN_ROOT": str(tmp_path / "nowhere")}
        completed = subprocess.run(
            ["/bin/sh", str(DOCTOR_SH)], env=env, capture_output=True, text=True,
            timeout=30,
        )
        assert completed.returncode == 0 and completed.stdout == ""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")
    def test_it_runs_doctor_through_the_resolved_uv(self, tmp_path):
        """`$MULTI_LLM_UV` first, and the argv is the dispatcher's own shape."""
        log = tmp_path / "argv.txt"
        fake_uv = tmp_path / "uv"
        fake_uv.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ARGV_LOG\"\nexit 0\n", encoding="utf-8"
        )
        fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "CLAUDE_PLUGIN_ROOT": str(REPO_ROOT),
            "MULTI_LLM_UV": str(fake_uv),
            "ARGV_LOG": str(log),
        }
        completed = subprocess.run(
            ["/bin/sh", str(DOCTOR_SH)], env=env, capture_output=True, text=True,
            timeout=30,
        )
        assert completed.returncode == 0
        skill = str(REPO_ROOT / "skills" / "multi-llm")
        assert log.read_text(encoding="utf-8").splitlines() == [
            "run", *unattended.DEFAULT_UV_RUN_ARGS, "--project", skill, "--",
            "python", f"{skill}/multi_llm.py", "doctor", "--warn-only",
        ]


# ---------------------------------------------------------------------------
# What the doctor hook forwards, and what the wakeup hook's early exit reads
# ---------------------------------------------------------------------------
# A SessionStart hook's stdout enters the model's context in every session, and the
# transcript stores it as a `hook_success` attachment. Both rows run the real hook
# scripts from the source tree and from the generated plugin tree, which is what
# users run.

#: The source tree and the generated one (``make claude-regen``).
PLUGIN_ROOTS = {
    "source": REPO_ROOT,
    "generated": REPO_ROOT / "packaging" / "claude" / "multi-llm",
}
posix_shell = pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")

#: Two problem lines in master's `--warn-only` shape (`doctor.problem_lines`).
WARN_LINE = ("WARN: codex: sandbox BROKEN (bwrap: loopback: Failed RTM_NEWADDR: "
             "Operation not permitted) [sandbox_fallback: auto]")
NOTE_LINE = "NOTE: cursor-agent: auth checking was incomplete (no verdict matched)"
#: The report doctor's probes are replaced by: nothing here depends on the machine.
DOCTOR_REPORT = {
    "present": 2, "configured": 19, "host": "claude-code", "elapsed_s": 0.5,
    "failures": ["codex"],
}

#: Runs the real dispatcher with doctor's probes replaced, so the summary line, the
#: problem lines and the result line after them are the code's own.
DRIVER = f'''
import os, sys
script, args = sys.argv[1], sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
import multi_llm
from utils import doctor
doctor.collect = lambda **_kwargs: dict({DOCTOR_REPORT!r})
doctor.problem_lines = lambda _report: [{WARN_LINE!r}, {NOTE_LINE!r}]
sys.exit(multi_llm.main(args))
'''


def _recording_uv(tmp_path: Path, body: str = "") -> Path:
    """A ``uv`` that records its argv in ``$ARGV_LOG``, then runs *body*."""
    stub = tmp_path / "bin" / "uv"
    stub.parent.mkdir(exist_ok=True)
    stub.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ARGV_LOG\"\n" + body, encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return stub


def _dispatching_uv(tmp_path: Path) -> Path:
    """A recording ``uv`` that runs the dispatcher through :data:`DRIVER`."""
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    return _recording_uv(tmp_path, (
        'while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done\n'
        "shift\nshift\n"   # -- python
        f'exec "$REAL_PYTHON" "{driver}" "$@"\n'
    ))


def _hook_env(tmp_path: Path, root: Path, stub: Path) -> dict:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CLAUDE_PLUGIN_ROOT": str(root),
        "MULTI_LLM_UV": str(stub),
        "ARGV_LOG": str(tmp_path / "argv.txt"),
        "REAL_PYTHON": sys.executable,
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _run_script(tmp_path: Path, root: Path, script: str, stub: Path, text: str = ""):
    completed = subprocess.run(
        ["/bin/sh", str(root / "hooks" / script)], input=text,
        env=_hook_env(tmp_path, root, stub),
        capture_output=True, encoding="utf-8", timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return completed


def _summary(monkeypatch) -> str:
    """The doctor's `--warn-only` text for :data:`DOCTOR_REPORT`, as the hook prints it."""
    monkeypatch.setattr(doctor, "problem_lines", lambda _report: [WARN_LINE, NOTE_LINE])
    return "\n".join(doctor.render_problems(dict(DOCTOR_REPORT))) + "\n"


class TestADirectDoctorCall:

    def test_it_still_ends_with_exactly_one_result_line(self, monkeypatch, capsys):
        """The hook drops the protocol line; nobody else may lose it."""
        monkeypatch.setattr(doctor, "collect", lambda **_kwargs: dict(DOCTOR_REPORT))
        monkeypatch.setattr(doctor, "problem_lines", lambda _report: [WARN_LINE, NOTE_LINE])
        assert multi_llm.main(["doctor", "--warn-only"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert [l for l in lines if l.startswith(cli_common.RESULT_MARKER)] == [lines[-1]]
        assert WARN_LINE in lines and NOTE_LINE in lines


@posix_shell
@pytest.mark.parametrize("root", PLUGIN_ROOTS.values(), ids=PLUGIN_ROOTS.keys())
class TestTheDoctorHookOutput:

    def test_it_keeps_master_s_argv_and_filters_one_line(self, root):
        """Nothing renders this shell file, so the argv and the filter are pinned here."""
        text = (root / "hooks" / "doctor.sh").read_text(encoding="utf-8")
        assert 'multi_llm.py" doctor --warn-only 2>/dev/null' in text
        assert "--brief" not in text
        assert "LC_ALL=C sed '/^MULTI_LLM_RESULT: {/d'" in text
        assert "|| filtered=$out" in text

    def test_a_byte_the_locale_cannot_decode_never_drops_the_warnings(self, root, tmp_path):
        """`grep -v` turns a line it cannot decode into "binary file matches" and
        prints nothing; the hook filters byte-wise, so the warning survives."""
        warn = b"WARN: x: sandbox \xff\xfe broken"
        body = (
            "printf '%s\\n' 'multi-llm doctor: 1 problem(s)'\n"
            "printf 'WARN: x: sandbox \\377\\376 broken\\n'\n"
            "printf '%s\\n' 'MULTI_LLM_RESULT: {\"mode\": \"doctor\", \"exit\": 0}'\n"
        )
        env = {**_hook_env(tmp_path, root, _recording_uv(tmp_path, body)), "LC_ALL": "C.UTF-8"}
        completed = subprocess.run(
            ["/bin/sh", str(root / "hooks" / "doctor.sh")], input=b"", env=env,
            capture_output=True, timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.splitlines() == [b"multi-llm doctor: 1 problem(s)", warn]

    def test_the_result_line_stops_here_and_the_warnings_go_on(
        self, root, tmp_path, monkeypatch,
    ):
        completed = _run_script(tmp_path, root, "doctor.sh", _dispatching_uv(tmp_path))
        argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
        assert argv[-2:] == ["doctor", "--warn-only"]
        assert cli_common.RESULT_MARKER not in completed.stdout
        # Everything the dispatcher printed above its result line, unchanged.
        assert completed.stdout == _summary(monkeypatch)
        assert completed.stdout.splitlines()[1:] == [WARN_LINE, NOTE_LINE]

    def test_only_a_line_that_starts_with_the_marker_is_dropped(self, root, tmp_path):
        printed = [
            "multi-llm doctor: 1 problem(s)",
            "WARN: x: a detail quoting MULTI_LLM_RESULT: {} mid-line",
            "  MULTI_LLM_RESULT: {indented, so not the protocol line}",
            'MULTI_LLM_RESULT: {"mode": "doctor", "exit": 0}',
        ]
        body = "".join(f"printf '%s\\n' '{line}'\n" for line in printed)
        completed = _run_script(tmp_path, root, "doctor.sh", _recording_uv(tmp_path, body))
        assert completed.stdout.splitlines() == printed[:3]


#: The doctor hook's `hook_success` attachment as Claude Code 2.1.3xx stores it, with
#: the text a machine with a warning prints. Its summary names `multi_llm.py`.
def _doctor_attachment(stdout: str) -> dict:
    return {
        "parentUuid": None, "isSidechain": False, "type": "attachment",
        "attachment": {
            "type": "hook_success", "hookName": "SessionStart:startup",
            "toolUseID": "SessionStart", "hookEvent": "SessionStart",
            "content": stdout, "stdout": stdout, "stderr": "", "exitCode": 0,
            "command": '"${CLAUDE_PLUGIN_ROOT}/hooks/doctor.sh"', "durationMs": 900,
        },
    }


@posix_shell
@pytest.mark.parametrize("root", PLUGIN_ROOTS.values(), ids=PLUGIN_ROOTS.keys())
class TestTheWakeupHookEarlyExit:
    """`hooks/wakeup-check.sh` exits before uv unless a Bash call ran the dispatcher.
    It used to look for the bare word, which the doctor's own output carries."""

    def _transcript(self, tmp_path: Path, monkeypatch, *, dispatched: bool) -> str:
        transcript = Transcript().prompt("remind me in an hour")
        transcript._add(_doctor_attachment(_summary(monkeypatch)))
        if dispatched:
            phase(transcript, "review-plan", "[PROGRESS] fanout\n")
        transcript.say("I'll check back later.")
        return json.dumps(hook_input(transcript.write(tmp_path / "t.jsonl")))

    def test_a_doctor_attachment_alone_never_starts_uv(self, root, tmp_path, monkeypatch):
        text = self._transcript(tmp_path, monkeypatch, dispatched=False)
        assert "multi_llm.py" in (tmp_path / "t.jsonl").read_text(encoding="utf-8")
        completed = _run_script(tmp_path, root, "wakeup-check.sh", _recording_uv(tmp_path), text)
        assert completed.stdout == "" and completed.stderr == ""
        assert not (tmp_path / "argv.txt").exists()

    def test_a_bash_dispatcher_call_does(self, root, tmp_path, monkeypatch):
        """The control: without it the row above could pass by exiting for another reason."""
        text = self._transcript(tmp_path, monkeypatch, dispatched=True)
        _run_script(tmp_path, root, "wakeup-check.sh", _recording_uv(tmp_path), text)
        argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
        assert argv[-4:] == ["util", "wakeup-check", "--hook-input", "-"]

    @pytest.mark.parametrize(("script", "util"), [("wakeup-check.sh", "wakeup-check"), ("report-check.sh", "report-check")])
    def test_a_dispatcher_call_in_either_key_order_does(self, root, tmp_path, monkeypatch, script, util):
        """JSON keys have no order: with `input` written before `name`, as a sorted dump
        writes them, the prefilter still finds the call. Both hooks share the prefilter."""
        text = self._transcript(tmp_path, monkeypatch, dispatched=True)
        path = tmp_path / "t.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("".join(json.dumps(json.loads(line), sort_keys=True) + "\n" for line in lines), encoding="utf-8")
        # The control: the old one-regex prefilter, name before the dispatcher, misses this file.
        assert not re.search(r'"name"\s*:\s*"Bash".*multi_llm\.py', path.read_text(encoding="utf-8"))
        _run_script(tmp_path, root, script, _recording_uv(tmp_path), text)
        argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
        assert util in argv
