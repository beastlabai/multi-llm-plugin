"""The codex host tells its model to keep waiting on a long call (fix round 16).

Codex racing run 6: under codex's code mode the model called ``multi_llm``
from a JavaScript ``exec`` cell, the cell handed control back after its default
30 s ``yield_time_ms`` while the call was still running, and after four
"still running" messages the model ended its turn. ``codex exec`` exited and
its teardown aborted the implement run at 3/10. Measured offline on codex
0.160.1: a cell whose first line is ``// @exec: {"yield_time_ms": 900000}``
waits for the call and returns only with its result. The rule that says so is
rendered into the tool description and the plugin's skill, with the wait
derived from the same ``tool_timeout_sec`` the server enforces.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packaging"))

import modes as mode_table  # noqa: E402
from common import surface  # noqa: E402

MCP_JSON = REPO_ROOT / "packaging" / "codex" / "mcp.json"
SERVER = REPO_ROOT / "packaging" / "codex" / "mcp_server.py"


def _budget_s() -> int:
    (entry,) = json.loads(MCP_JSON.read_text(encoding="utf-8"))["mcpServers"].values()
    return entry["tool_timeout_sec"]


def _server_default_budget() -> int:
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_TOOL_TIMEOUT_S"
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("mcp_server.py defines no DEFAULT_TOOL_TIMEOUT_S")


class TestTheRuleAgreesWithTheBudget:

    def test_the_wait_is_the_whole_tool_budget(self):
        budget = _budget_s()
        rule = surface.long_call_rule("codex")
        assert f'`// @exec: {{"yield_time_ms": {budget * 1000}}}`' in rule
        assert f"up to its {budget // 60}-minute budget" in rule
        # The shipped numbers, so a change to either is a visible decision.
        assert budget == 900
        assert '`// @exec: {"yield_time_ms": 900000}`' in rule

    def test_the_server_s_fallback_budget_is_the_same_number(self):
        """The server falls back to this when `.mcp.json` is unreadable."""
        assert _server_default_budget() == _budget_s()

    def test_the_rule_is_the_mode_table_s_text(self):
        rule = surface.long_call_rule("codex")
        expected = (mode_table.TOOL_LONG_CALL_RULE
                    .replace("{BUDGET_MIN}", "15").replace("{YIELD_MS}", "900000"))
        assert rule == expected
        for words in ('"Script running"', "call `wait` until this call returns",
                      "never end your turn while it is running"):
            assert words in rule


class TestWhereTheRuleIsRendered:

    def test_the_codex_tool_description_carries_it(self):
        rendered = surface.cli_surface("codex")["tool_description"]
        assert rendered.count(surface.long_call_rule("codex")) == 1

    def test_no_other_host_carries_it(self):
        assert surface.long_call_rule("opencode") is None
        rendered = surface.cli_surface("opencode")["tool_description"]
        assert "yield_time_ms" not in rendered

    def test_the_codex_plugin_skill_carries_it(self):
        # Loaded by path, as test_codex_package does: a bare `import build`
        # can return the OpenCode package's build module.
        spec = importlib.util.spec_from_file_location(
            "codex_build_long_call", str(REPO_ROOT / "packaging" / "codex" / "build.py"))
        codex_build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codex_build)

        text = codex_build.skill_md()
        assert text.count(surface.long_call_rule("codex")) == 1
        assert codex_build.LONG_CALL_SLOT not in text
