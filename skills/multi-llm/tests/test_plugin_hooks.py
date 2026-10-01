#!/usr/bin/env python3
"""The Claude Code plugin's `hooks/hooks.json` (D6 step 5).

One `SessionStart` command hook runs `multi_llm.py doctor --warn-only` through
`hooks/doctor.sh` — advisory, never blocking: `--warn-only` exits 0 whatever it
finds, and the script itself exits 0 on every path, including a machine with no
`uv` (a SessionStart hook that exited 2 would block the session). The manifest
at `.claude-plugin/plugin.json` points at the file explicitly.
"""

from __future__ import annotations

import json
import os
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

from utils import unattended  # noqa: E402


def _hook_entries():
    data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    return data["hooks"]["SessionStart"]


class TestTheHookFile:

    def test_it_is_valid_json_naming_session_start(self):
        data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
        assert list(data) == ["hooks"]
        # `Stop` is the report check (plans/next-phase-command.md T11), pinned
        # in tests/test_report_check.py.
        assert list(data["hooks"]) == ["SessionStart", "Stop"]

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
        assert "doctor --warn-only" in text
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
