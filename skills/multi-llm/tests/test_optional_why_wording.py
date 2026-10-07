"""An optional next step's ``why`` carries no "optional:" of its own.

Fix round 22 (live run 9, #8; OpenCode racing 03): the host printed
"Optional, before that: optional: have the models review the task list…",
because ``Mode.optional_why`` began "optional:" while every host rule already
labels the step ("Optional, before that:", which ``user_next_step.markdown``
carries since fix round 22, and the shims). The why is a sentence that reads
after that label and says what the step would do and what skipping it means. These rows read it off
every surface that carries it: the mode table, ``user_next_step.optional_before``,
the ``[NEXT_COMMAND_OPTIONAL_WHY]`` line, ``util next-command`` and a terminal
``status`` poll (both build it through the one builder), and the generated CLI
surface both tool servers ship (``optional_why`` per mode row). The codex MCP
server and the OpenCode plugin pass ``user_next_step`` through unworded.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL.parent.parent / "packaging"))

import modes  # noqa: E402
import multi_llm  # noqa: E402
from tests.test_next_phase import S, UP_TO_TASKS, _build  # noqa: E402
from utils import next_phase as np_  # noqa: E402

#: What a why must never start with: the label is the host's to print.
_LABELLED = re.compile(r"^\W*optional\b", re.IGNORECASE)


def _reads_after_the_label(why: str) -> None:
    assert why and not _LABELLED.match(why), why
    assert why[0].isupper(), why
    assert why.endswith("."), why


def test_the_mode_table():
    whys = [mode.optional_why for mode in modes.MODES if mode.optional_why]
    assert len(whys) == 2
    for why in whys:
        _reads_after_the_label(why)


@pytest.fixture
def plan(tmp_path):
    return _build(tmp_path, S(UP_TO_TASKS))


def test_the_markdown_still_labels_the_step(plan):
    """Fix round 22 moved the label out of the tool routes' rule into
    ``user_next_step.markdown``, which the rule says to copy as it is: the
    markdown carries "Optional, before that:" and right after it the step's
    why, which reads after the label with no "optional" of its own."""
    assert "`markdown`, copied as it is" in modes.TOOL_USER_NEXT_STEP_LINE
    answer = np_.next_phase_commands(
        "plans/foo.md", "opencode", "package", "generate-tasks", plan_path=plan)
    payload = answer.to_dict()
    (step,) = payload["optional_before"]
    blocks = payload["markdown"].split("\n\n")
    at = blocks.index(np_.OPTIONAL_LEAD)
    assert np_.OPTIONAL_LEAD == "Optional, before that:"
    assert blocks[at + 1] == step["why"], blocks
    _reads_after_the_label(blocks[at + 1])
    assert payload["markdown"].lower().count("optional") == 1, payload["markdown"]


def test_user_next_step_and_its_block(plan):
    answer = np_.next_phase_commands(
        "plans/foo.md", "opencode", "package", "generate-tasks", plan_path=plan)
    steps = answer.to_dict()["optional_before"]
    assert [step["phase"] for step in steps] == ["review-tasks"]
    for step in steps:
        _reads_after_the_label(step["why"])
    whys = [line.split(" ", 1)[1] for line in np_.render_next_phase_lines(answer)
            if line.startswith(np_.NEXT_COMMAND_OPTIONAL_WHY_MARKER)]
    assert whys == [steps[0]["why"]]


def test_util_next_command(plan, capsys, monkeypatch):
    monkeypatch.chdir(plan.parent.parent)
    assert multi_llm.main(
        ["util", "next-command", "plans/foo.md", "--after", "generate-tasks",
         "--host", "codex"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines()
             if line.startswith(np_.NEXT_COMMAND_OPTIONAL_WHY_MARKER)]
    assert lines
    for line in lines:
        _reads_after_the_label(line.split(" ", 1)[1])
    result = json.loads(out.rsplit("MULTI_LLM_RESULT: ", 1)[1].splitlines()[0])
    for step in result["user_next_step"]["optional_before"]:
        _reads_after_the_label(step["why"])
    assert "optional: " not in out.lower()


@pytest.mark.parametrize("host", ["codex", "opencode"])
def test_the_cli_surface_the_tool_servers_ship(host):
    from common import surface

    rows = surface.cli_surface(host)["modes"]
    whys = [row["optional_why"] for row in rows if row.get("optional_why")]
    assert whys
    for why in whys:
        _reads_after_the_label(why)
