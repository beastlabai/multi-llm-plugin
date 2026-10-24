"""fx-specific rig contracts plus a real, offline wrapper/tool smoke."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from . import harness, stub_server
from .docker_probe import requires_docker


def test_code_review_verdicts_cover_original_and_fx_backtick_normalization():
    from utils.state_manager import generate_group_id

    responses = Path(__file__).parents[1] / "fixtures/e2e/responses"
    issues = json.loads((responses / "code_review/container_issues.json").read_text())
    sheet = json.loads((responses / "validation/code_review_valid.json").read_text())
    verdicts = {row["group_hash"]: row for row in sheet["groups"]}
    changed = 0
    for issue in issues:
        normalized = {**issue, "desc": issue["desc"].replace("`", "")}
        original_hash, fx_hash = [
            generate_group_id({"theme": issue["title"], "suggestions": [entry]})
            for entry in (issue, normalized)
        ]
        for key in (original_hash, fx_hash):
            assert verdicts[key]["_theme"] == issue["title"]
            assert verdicts[key]["status"] == "valid"
            assert verdicts[key]["_reviewers"] == 1
        if original_hash != fx_hash:
            changed += 1
            assert verdicts[original_hash]["reason"] == verdicts[fx_hash]["reason"]
    assert changed == 2


@pytest.mark.parametrize("tool", [False, True], ids=["text", "tool"])
def test_stream_usage_is_requested_and_has_a_consistent_total(tool):
    script = stub_server.ScriptedToolCall({"name": "read_file", "arguments": {"path": "x"}})
    render = (lambda include: stub_server.chat_call_stream("stub-model", script, include)) if tool else (
        lambda include: stub_server.chat_stream("stub-model", '["ok"]', include))
    original = list(render(False))
    with_usage = list(render(True))
    assert with_usage[:-1] == original[:-1]
    chunk = with_usage[-1][1]
    assert chunk["choices"] == original[-1][1]["choices"]
    assert len(chunk["choices"]) == 1
    assert chunk["choices"][0]["index"] == 0
    usage = chunk["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] == usage["completion_tokens"] == 1


def _review_script(**extra):
    return stub_server.ScriptedToolCall({
        "name": "shell", "surfaces": ["chat"],
        "arguments": {"request": {"action": "run", "command": "printf probe; pwd"}},
        **extra,
    })


def test_permission_reviewer_is_opted_in_and_matches_only_the_exact_shell_command():
    body = {"tools": [{"type": "function", "function": {"name": "permission_decision"}}],
            "messages": [{"role": "system", "content": "<permission_review>"},
                         {"role": "user", "content": "command: printf probe; pwd\ncwd: /tmp"}]}
    assert stub_server.fx_permission_review(body, _review_script()) is None
    script = _review_script(fx_permission_decision="clear")
    assert stub_server.fx_permission_review(body, script) == "clear"
    body["messages"][1]["content"] = "command: printf probe; rm x\ncwd: /tmp"
    assert stub_server.fx_permission_review(body, script) is None
    body["messages"][1]["content"] = "command: printf probe; pwd\ncwd: /tmp"
    body["tools"].append({"type": "function", "function": {"name": "shell"}})
    assert stub_server.fx_permission_review(body, script) is None


@pytest.mark.parametrize("block", [
    {"name": "read_file", "arguments": {"path": "x"}},
    {"name": "shell", "arguments": {"request": "wrong-shape"}},
    {"name": "shell", "arguments": {"request": {"action": "run", "command": "a\nb"}}},
])
def test_permission_fixture_cannot_approve_an_unspecified_command(block):
    with pytest.raises(ValueError, match="fx_permission_decision"):
        stub_server.ScriptedToolCall({"surfaces": ["chat"],
                                     "fx_permission_decision": "clear", **block})


def test_fx_has_no_timeout_derived_argv_policy():
    from utils.providers.fx import FX_REQUIRED_FLAGS
    from ..golden.normalize_argv import _ROW_DERIVED

    assert set(FX_REQUIRED_FLAGS) & set(_ROW_DERIVED) == {"--model"}


def test_live_seed_uses_the_same_profile_with_private_modes(monkeypatch, tmp_path):
    import importlib
    import sys
    from utils.providers.fx import render_settings

    rig = Path(__file__).parent
    monkeypatch.syspath_prepend(str(rig))
    monkeypatch.setattr(sys, "path", list(sys.path))
    runner = importlib.import_module("live_provider")
    monkeypatch.setattr(runner, "Path", lambda value: rig / "home_seed"
                        if value == "/opt/mllm/rig/home_seed" else Path(value))
    home = tmp_path / "home"
    endpoint = "https://openrouter.ai/api/v1"
    runner.render_seed("fx", home, endpoint, "OPENROUTER_API_KEY", "stub-model")
    settings = home / ".fx/settings.json"
    assert json.loads(settings.read_text()) == render_settings(base_url=endpoint)
    if os.name != "nt":  # POSIX mode bits: Windows keeps only a read-only flag
        assert settings.parent.stat().st_mode & 0o777 == 0o700
        assert settings.stat().st_mode & 0o777 == 0o600
    assert runner.seed_guards(home, endpoint) == []


@pytest.mark.container
@requires_docker()
def test_real_fx_wrapper_read_write_and_reviewed_shell():
    script = Path(__file__).with_name("fx_smoke.py").read_text()
    completed = subprocess.run([
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--tmpfs", "/tmp:rw,exec,mode=1777", "--user", "10001:10001",
        "--entrypoint", "/opt/mllm/skill/.venv/bin/python",
        "-i", harness.DEFAULT_IMAGE, "-",
    ], input=script, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["ok"]
    assert [r["case"] for r in report["results"]] == ["happy", "read", "write", "shell"]
    assert report["results"][-1]["reviewers"] == 1
