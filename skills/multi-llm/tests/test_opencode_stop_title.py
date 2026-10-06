"""The OpenCode package titles its tool part at a stop (fix round 16).

OpenCode racing run 6: implement stopped at T006 and the tool result carried
the stop and ``resume_command``, but the session's free model never wrote its
reply — it sat idle for 28 minutes until it was killed — so the user got
nothing. A plugin tool's title is what opencode shows of a call without the
model: ``opencode run`` prints it on the tool's own line the moment the call
returns (``⚙ multi_llm <title>``, in place of the arguments), ``--format json``
carries it as ``state.title`` (the TUI's tool header reads the same title, not
measured) — measured offline on opencode 1.18.34 with a stub provider and a model that
never replies, through ``tool.execute.after``, the hook this plugin sets it in.

Every word of the title is the surface's ``stop_title`` (``modes.STOP_TITLE``)
or the result's own; the rows grade the JS against that Python owner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import modes as mode_table  # noqa: E402
from test_opencode_package import built_package, node_json  # noqa: E402,F401

STOP_LINE = (
    "implement stopped at T006 (Track player progress): partial application "
    "(bytes moved, the batch did not finish). Not run: T007. To go on: inspect "
    "the half-applied state, restore or keep it, fix the cause, then resume "
    "this phase."
)
RESUME = "/multi-llm --implement plans/racing-game.md --resume"


def _after(package: Path, tool: str | None, output: object) -> dict:
    """Run the plugin's own ``tool.execute.after`` on one tool output."""
    return node_json(f"""
    const hooks = await MultiLlmPlugin({{}})
    const output = {{ title: "", output: {json.dumps(output)}, metadata: {{}} }}
    await hooks["tool.execute.after"](
      {{ tool: {json.dumps(tool)} ?? surface.tool_name, sessionID: "s", callID: "c", args: {{}} }},
      output,
    )
    emit({{ title: output.title, output: output.output, name: surface.tool_name }})
    """, package)


def _stop(**extra) -> str:
    outcome = {"ok": False, "state": "needs-human", "user_next_step": None,
               "resume_command": RESUME, "message": "Stop here: …",
               "exit": 3, "result": {"mode": "implement", "state": "needs-human",
                                     "exit": 3, "error": STOP_LINE}}
    outcome.update(extra)
    return json.dumps(outcome)


class TestTheStopTitle:

    def test_a_stop_with_a_resume_command_is_titled_with_both(self, built_package):
        got = _after(built_package, None, _stop())
        text = mode_table.STOP_TITLE
        assert got["title"] == (
            text["template"].replace("{line}", STOP_LINE)
            + text["resume"].replace("{command}", RESUME)
        )
        assert json.loads(got["output"])["resume_command"] == RESUME

    def test_a_stop_with_no_command_names_none(self, built_package):
        outcome = json.loads(_stop())
        del outcome["resume_command"]
        outcome["no_command_reason"] = "There is no command for the user to type."
        del outcome["result"]
        got = _after(built_package, None, json.dumps(outcome))
        assert got["title"] == mode_table.STOP_TITLE["template"].replace(
            "{line}", mode_table.STOP_TITLE["no_line"])
        assert "Resume with" not in got["title"]

    def test_a_finished_phase_keeps_its_title(self, built_package):
        done = json.dumps({"ok": True, "state": "done", "user_next_step": {
            "phase": "code-review", "command": "/multi-llm --review-code p.md"}})
        assert _after(built_package, None, done)["title"] == ""

    def test_a_run_left_alive_is_not_a_stop(self, built_package):
        alive = json.dumps({"ok": False, "state": "needs-human", "exit": None,
                            "run_id": "RID", "abort_command": {"mode": "status"}})
        assert _after(built_package, None, alive)["title"] == ""

    def test_another_tool_and_unreadable_output_are_left_alone(self, built_package):
        assert _after(built_package, "bash", _stop())["title"] == ""
        assert _after(built_package, None, "not json at all")["title"] == ""
