"""The manual-task note ends with its command (live run 8, #8).

Codex racing 08: the note read "… record it done with $multi-llm:multi-llm
--implement plans/racing-game.md --resume --human-task T007=done (keep a
--human-task answer only for each task you did)." and the host put the whole
tail, parenthetical and full stop included, inside the command's fence, so a
copy of the fence carries prose. The advice now comes before the command and
nothing follows it, on every host and route, in `[MANUAL_TASK_NOTE]`, and in
a `status` trailer or terminal-poll notice, where the manual line is the last
part.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from utils import host_capabilities, run_status  # noqa: E402
from utils import next_phase as np_  # noqa: E402

from tests.test_manual_tasks_on_completion import (  # noqa: E402,F401
    P, _complete_plan, _run, workdir,
)

ADVICE = "keeping a --human-task answer only for each task you did"


def _routes():
    for host in host_capabilities.host_ids():
        for route in host_capabilities.next_command_routes(host):
            yield host, route


@pytest.mark.parametrize("host,route", list(_routes()))
def test_every_route_spells_a_note_that_ends_with_its_command(host, route):
    template = host_capabilities.next_command_template(host, route)
    command = np_.close_command(template, "plans/racing game.md", ["T007", "T011"])
    note = np_.manual_close_note(command)
    assert note.endswith(command), note
    assert note.index(ADVICE) < note.index(command)
    assert note.count(command) == 1


def test_the_marker_line_ends_with_the_command(workdir, capsys):
    _complete_plan(workdir)
    code, out, err, _ = _run(
        ["util", "next-command", P, "--after", "apply-code-fixes"], capsys
    )
    assert code == 0, err
    # Fix round 22: the command on its own line, right after the note.
    lines = out.splitlines()
    notes = [i for i, line in enumerate(lines) if line.startswith("[MANUAL_TASK_NOTE] ")]
    assert len(notes) == 1
    assert lines[notes[0]] == f"[MANUAL_TASK_NOTE] {np_.manual_close_lead()}"
    command = lines[notes[0] + 1]
    assert command.startswith("[MANUAL_TASK_COMMAND] ")
    assert command.endswith("--resume --human-task T011=done --human-task T012=done")


def test_the_manual_line_is_the_trailers_last_so_a_joined_notice_ends_on_it():
    """A terminal poll's `notice` joins the trailer lines with "; ": the line
    carrying the command comes last, after the check findings lines that can
    sit beside it while implement is unfinished."""
    command = "/multi-llm --implement plans/x.md --resume --human-task T011=done"
    payload = {
        "suggested_next": "implement",
        "notices": [],
        "phases": {
            "implement": {
                "status": "in_progress",
                "manual_tasks": [{"id": "T011", "title": "Manual playthrough"}],
                "check_findings": [{"id": "T002", "findings": ["retries must be 3"]}],
                "batch_check_findings": [{"id": "T004", "findings": ["lint fails"]}],
            },
            "code-review": {"status": "pending"},
        },
    }
    lines = run_status._workflow_trailer_lines(
        payload, prefix="", close_note=np_.manual_close_note(command),
    )
    assert len(lines) == 4, lines
    assert lines[-1].startswith("manual tasks left for you: T011")
    assert "; ".join(lines).endswith(command)


@pytest.mark.parametrize("route,spelling", [
    ("package", "$multi-llm:multi-llm --implement "),
    (None, "Use the multi-llm skill with exactly these arguments: --implement "),
])
def test_codex_status_spells_the_route_it_runs_on(
    workdir, capsys, monkeypatch, route, spelling,
):
    """Live run 8 (Codex racing 06): the driver's `status --host codex`, run
    from a plain shell, spelled the file shim's sentence while the tool route
    said `$multi-llm:multi-llm`. Kept on purpose: `MULTI_LLM_ROUTE` is the one
    route signal (AGENTS.md), the plugin's tool server sets it on every call,
    status included, and a shell without it is the file-shim route by
    definition. A Codex host reads status through its own route; only a
    driver's shell reads it outside one (set `MULTI_LLM_ROUTE=package` there
    to see the plugin's spelling). Fix round 24: unless a run of the plan
    recorded the tool route it was started on (`run.route`), which this plan
    has none of (`test_status_recorded_route.py`)."""
    from utils import run_status as rs

    if route:
        monkeypatch.setenv("MULTI_LLM_ROUTE", route)
    plan = _complete_plan(workdir)
    capsys.readouterr()
    rs.main(["--plan-file", str(plan), "--host", "codex"])
    out = capsys.readouterr().out
    assert f"{ADVICE}: {spelling}" in out, out
