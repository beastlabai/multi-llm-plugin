"""fx-specific rig contracts plus a real, offline wrapper/tool smoke."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from . import harness, stub_server
from .docker_probe import requires_docker


RESPONSES = Path(__file__).parents[1] / "fixtures/e2e/responses"


def _fx_render(answer: str) -> str:
    """fx's ``final_output`` for a final message with no fence and no fx
    self-introduction line: trim, delete every ``*`` and backtick, trim the end.
    Checked against a real capture by the test below."""
    return "".join(c for c in answer.strip(" \t\r\n") if c not in "*`").rstrip(" \t\r\n")


def _fx_stdout(served: str) -> str:
    """What fx prints when the stub serves ``served`` as the one completed turn:
    ``output`` is those bytes verbatim, ``final_output`` their rendering.
    Measured on runs/20261106T160333Z chain-fx (fx 0.0.16), for both the
    code-review fan-out and the ask answer."""
    return json.dumps({
        "output": served, "final_output": _fx_render(served), "exit_code": 0,
        "model": "stub-model", "resolved_provider": None, "session_id": "", "steps": 0,
        "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1},
    })


def test_the_render_model_matches_a_real_fx_capture():
    captured = json.loads((Path(__file__).parents[1] / "fixtures/fx/markdown-markers.stdout.txt").read_text())
    _narration, answer = captured["output"].split("\n\n", 1)
    assert "`" in answer and "*" in answer
    assert _fx_render(answer) == captured["final_output"]


@pytest.mark.parametrize("fixture", ["code_review/container_issues.json", "ask/plan_question.md"])
def test_an_fx_member_returns_the_served_answer_byte_for_byte(fixture):
    """fx's rendering strips both fixtures' backticks (and the ask answer's
    emphasis). The adapter must hand back what the stub served."""
    from utils.providers.fx import FxProvider

    served = (RESPONSES / fixture).read_text()
    stdout = _fx_stdout(served)
    assert json.loads(stdout)["final_output"] != served.strip()
    assert FxProvider().salvage_text(stdout) == served.strip()
    if fixture.endswith(".json"):
        assert FxProvider().parse_output(stdout, "") == {"success": True, "data": json.loads(served)}


def _panel_groups(members: int, with_fx: bool = False) -> list[dict]:
    """The groups a ``members``-strong panel forms over the code-review fan-out.

    Grouped through the product's own grouper. An fx member's copy goes
    through the real adapter from the envelope fx prints for the stub's answer.
    """
    from utils.providers.fx import FxProvider
    from utils.state_manager import stamp_stable_ids
    from utils.suggestion_processor import group_similar_suggestions

    served = (RESPONSES / "code_review/container_issues.json").read_text()
    copies = [(f"plain-{n}:stub-model", json.loads(served)) for n in range(members - with_fx)]
    if with_fx:
        copies.append(("fx:stub-model", FxProvider().parse_output(_fx_stdout(served), "")["data"]))
    panel = [{**issue, "source_model": model} for model, issues in copies for issue in issues]
    groups = [group.to_dict() for group in group_similar_suggestions(panel)]
    stamp_stable_ids(groups)
    return groups


def _verdicts() -> dict:
    sheet = json.loads((RESPONSES / "validation/code_review_valid.json").read_text())
    return {row["group_hash"]: row for row in sheet["groups"]}


@pytest.mark.parametrize("with_fx", [False, True], ids=["plain", "one-fx"])
@pytest.mark.parametrize("members", range(1, 7))
def test_code_review_verdicts_join_every_panel_with_or_without_an_fx_member(members, with_fx):
    """Every group a 1-6 member panel forms has a verdict, fx or no fx.

    runs/20261106T142827Z: the opencode host's panel carries one fx member.
    While the adapter read fx's rendered ``final_output``, groups 2 and 4 hashed
    to ids the sheet lacked, landed ``validation_failed``, and apply-code-fixes
    stopped on human-decision turns the scenario does not answer. An fx member
    now groups exactly like a plain one.
    """
    verdicts = _verdicts()
    groups = _panel_groups(members, with_fx)
    assert len(groups) == len(json.loads((RESPONSES / "code_review/container_issues.json").read_text()))
    if with_fx:
        assert [g["group_hash"] for g in groups] == [g["group_hash"] for g in _panel_groups(members)]
    for group in groups:
        row = verdicts.get(group["group_hash"])
        assert row, f"no verdict for {group['theme']!r} ({group['group_hash']})"
        assert row["_theme"] == group["theme"]
        assert row["status"] == "valid"
        assert row["_reviewers"] == len(group["suggestions"]) == members


def test_every_verdict_row_is_joined_by_some_panel():
    """No dead rows: each row is a group some 1-6 member panel forms."""
    formed = {g["group_hash"] for members in range(1, 7) for g in _panel_groups(members)}
    assert set(_verdicts()) == formed


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
