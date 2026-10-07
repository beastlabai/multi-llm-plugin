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


# ---------------------------------------------------------------------------
# Fix round 17 (the Fable review of round 16, abort #2): the error line carries
# a task title from tasks.md and the runner's own summary, so an ANSI sequence
# or another control character in either reached the tool line, where it could
# erase or rewrite the resume command; and a 1200-character headline took the
# whole line.
# ---------------------------------------------------------------------------

CONTROLS = "\x1b[2K\x1b[1G\x07\x08\x00\x7f\x9b31m‮⁦"
FORGED = f"T006 (Track{CONTROLS} progress): fake resume /multi-llm --implement x.md"


def _controls_in(text: str) -> list:
    return [ch for ch in text if ord(ch) < 0x20 or 0x7f <= ord(ch) <= 0x9f
            or ch in "‎‏‪‫‬‭‮⁦⁧⁨⁩"]


class TestTheTitleIsPlainAndBounded:

    def test_control_characters_never_reach_the_tool_line(self, built_package):
        outcome = json.loads(_stop())
        outcome["result"]["error"] = FORGED
        outcome["resume_command"] = RESUME + "\x1b[2K"
        got = _after(built_package, None, json.dumps(outcome))
        assert _controls_in(got["title"]) == [], repr(got["title"])
        assert "Track[2K[1G" in got["title"] and got["title"].endswith(RESUME + "[2K")
        # The tool output itself is untouched: the model still reads it whole.
        assert json.loads(got["output"])["result"]["error"] == FORGED

    def test_a_long_line_is_cut_and_the_resume_command_kept_whole(self, built_package):
        outcome = json.loads(_stop())
        outcome["result"]["error"] = "x" * 1200
        got = _after(built_package, None, json.dumps(outcome))
        text = mode_table.STOP_TITLE
        head = text["template"].split("{line}")[0]
        tail = text["resume"].replace("{command}", RESUME)
        assert got["title"].startswith(head) and got["title"].endswith(tail)
        line = got["title"][len(head):-len(tail)]
        assert len(line) == text["max_line"] and line.endswith(text["cut"])
        # A line within the limit is shown as it is.
        assert _after(built_package, None, _stop())["title"].count(STOP_LINE) == 1


def test_the_result_lines_error_carries_no_control_character():
    """The same line reaches the codex tool result, ``run.error`` and
    ``status``, so the dispatcher keeps it plain at the source."""
    from utils import cli_common

    context = cli_common.ResultContext(mode="implement", phase="implement")
    context.set_error(f"implement stopped at {FORGED}\n\tNot run: T007.")
    assert _controls_in(context.error) == []
    assert context.error == (
        "implement stopped at T006 (Track[2K[1G31m progress): fake resume "
        "/multi-llm --implement x.md Not run: T007."
    )


# ---------------------------------------------------------------------------
# Fix round 22 (live run 9, #4): the title cut the error line mid-word. It
# cuts by the dispatcher's own rule now (`utils.text_clip.clip`), so the tool
# line and every other preview of the model's words read alike.
# ---------------------------------------------------------------------------

_CUT_CORPUS = [
    "implement stopped at T002 (Release check): its check reported 1 problem(s) "
    "[node scripts/check.mjs exited 1: check failed: retries must be 3 (found 2); "
    "src/config.json has retries 2, but scripts/check.mjs requires 3.]. Not run: "
    "T007. To go on (your call): fix them and resume, or accept its findings by "
    "adding --human-task T002=done to the resume command.",
    "apply-code-fixes stopped: \"Implementation record T001 and the plan's "
    "Decisions deliberately leave nextSpeed unclamped (catch-up may reach 60 "
    "m/s), and test 'a trailing opponent may exceed MAX_SPEED' pins it\"; "
    "adding `Math.min(..., MAX_SPEED)` would break it — " + "x" * 40 + " end.",
    "word " * 90,
    "/" + "very-long-path-segment/" * 20 + "file.js and then words",
    # Fix round 23 (review of round 22, #5): nothing a word boundary keeps,
    # cut inside on both sides rather than down to the bare mark.
    "(" * 400,
    "( [ " * 100,
    '"' * 30 + "(" * 300,
    "-" * 400,
]


def test_the_title_cuts_by_the_dispatchers_rule(built_package):
    from utils.text_clip import clip

    lines = json.dumps(_CUT_CORPUS)
    got = node_json(f"""
    const out = []
    for (const line of {lines}) {{
      for (const max of [40, 80, 120, 200, 300]) {{
        out.push(T.capTitleLine(line, {{ max_line: max, cut: surface.stop_title.cut }}))
      }}
    }}
    emit(out)
    """, built_package)
    want = [clip(line, limit) for line in _CUT_CORPUS for limit in (40, 80, 120, 200, 300)]
    assert got == want


def test_the_tool_line_ends_on_a_whole_word(built_package):
    """Through the hook itself: a 1200-character stop line is cut to
    ``max_line`` by the dispatcher's rule, never mid-word."""
    from utils.text_clip import clip

    line = "T002: " + " ".join([_CUT_CORPUS[0]] * 6)
    outcome = json.loads(_stop())
    outcome["result"]["error"] = line
    got = _after(built_package, None, json.dumps(outcome))
    text = mode_table.STOP_TITLE
    head = text["template"].split("{line}")[0]
    tail = text["resume"].replace("{command}", RESUME)
    shown = got["title"][len(head):-len(tail)]
    assert shown == clip(line, text["max_line"])
    assert shown.split(text["cut"])[0].split()[-1] in line.split()
