"""`utils/phase_walk.py`: the one "what is next" rule (code health ENTRY-2).

``get_workflow_status`` (``suggested_next``), ``next_phase.next_phase_commands``
(the walk from a finished phase) and ``next_phase.gate_resume_suggested`` each
used to walk ``modes.PREREQUISITE_PHASES`` with their own copy of the rule. The
table below was produced by those three walkers before they were folded into
:func:`phase_walk.next_step`, and pins every answer for each representative
state and every start: the walk itself, and the next-step builder's answer
from the same payload, gated (a finished phase) and not (``--after``).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
import modes  # noqa: E402
from utils import next_phase as np_  # noqa: E402
from utils import phase_walk  # noqa: E402

ORDER = list(modes.PREREQUISITE_PHASES)

#: One letter per phase, in ``ORDER``: the entry ``get_workflow_status`` gives it.
ENTRY = {
    "C": {"status": "completed"},
    "S": {"status": "skipped"},
    "P": {"status": "pending"},
    "I": {"status": "in_progress"},
    # An implement that stopped with tasks seeded and none completed.
    "T": {"status": "pending", "tasks_total": 4, "tasks_done": 0},
    # Recorded completed before implement's latest completion.
    "X": {"status": "pending", "stale": True},
}

CODE = {
    "review-plan": "rp", "apply-suggestions": "as", "generate-tasks": "gt",
    "review-tasks": "rt", "apply-task-suggestions": "ats", "implement": "im",
    "code-review": "cr", "apply-code-fixes": "acf",
}
KIND = {
    phase_walk.RESUME: "r", phase_walk.STARTED: "s", phase_walk.STALE: "x",
    phase_walk.NEXT: "n",
}

#: (state, the walk with no start then from each phase in ORDER, the next-step
#: builder's answer for each phase as a finished mode). A walk answer is
#: ``<kind>:<phase>`` (r resume, s started, x stale, n next) or ``-`` for
#: complete; a gated answer is the phase, ``-`` for complete, or ``.`` for no
#: answer (the finished phase is not on record).
TABLE = [
    ("PPPPPPPP", "n:rp n:as n:gt n:im n:im n:im n:cr n:acf -", ". . . . . . . ."),
    ("CPPPPPPP", "n:as n:as n:gt n:im n:im n:im n:cr n:acf -", "as . . . . . . ."),
    ("CCPPPPPP", "n:gt n:gt n:gt n:im n:im n:im n:cr n:acf -", "gt gt . . . . . ."),
    ("CCCPPPPP", "n:im n:im n:im n:im n:im n:im n:cr n:acf -", "im im im . . . . ."),
    ("CCCCPPPP", "n:im n:im n:im n:im n:im n:im n:cr n:acf -", "im im im im . . . ."),
    ("CCCSSPPP", "n:im n:im n:im n:im n:im n:im n:cr n:acf -", "im im im im im . . ."),
    ("CCCPPCPP", "n:cr n:cr n:cr n:cr n:cr n:cr n:cr n:acf -", "cr cr cr . . cr . ."),
    ("CCCPPCCP", "n:acf n:acf n:acf n:acf n:acf n:acf n:acf n:acf -", "acf acf acf . . acf acf ."),
    ("CCCPPCCC", "- - - - - - - - -", "- - - . . - - -"),
    ("CCCPPCCS", "- - - - - - - - -", "- - - . . - - -"),
    ("CCCPPIPP", "s:im n:im n:im n:im n:im n:im r:im s:im s:im", "im im im . . im . ."),
    ("CCCPPTPP", "s:im n:im n:im n:im n:im n:im n:cr s:im s:im", "im im im . . . . ."),
    ("CCCPPCXP", "s:cr n:cr n:cr n:cr n:cr n:cr n:cr x:cr s:cr", "cr cr cr . . cr . ."),
    ("XCCPPCPP", "s:rp x:rp s:rp s:rp s:rp s:rp s:rp s:rp s:rp", ". rp rp . . rp . ."),
    ("CCCPPCCI", "s:acf n:acf n:acf n:acf n:acf n:acf n:acf n:acf r:acf", "acf acf acf . . acf acf acf"),
    ("PPPPPPCP", "n:acf n:as n:gt n:im n:im n:im n:acf n:acf -", ". . . . . . acf ."),
    ("PPPPPCPP", "n:cr n:as n:gt n:cr n:cr n:cr n:cr n:acf -", ". . . . . cr . ."),
    ("CICPPPPP", "s:as n:as r:as s:as s:as s:as s:as s:as s:as", "as as as . . . . ."),
    ("CCCPIPPP", "n:im n:im n:im n:im n:im r:ats n:cr n:acf -", "im im im . ats . . ."),
    ("CCCIPPPP", "n:im n:im n:im n:im r:rt n:im n:cr n:acf -", "im im im rt . . . ."),
    ("CCCPPIPI", "s:im n:im n:im n:im n:im n:im r:im s:im r:acf", "im im im . . im . acf"),
    ("SSSSSSSS", "- - - - - - - - -", "- - - - - - - -"),
    ("CSPPPPPP", "n:gt n:gt n:gt n:im n:im n:im n:cr n:acf -", "gt gt . . . . . ."),
    ("CCCPPTCP", "s:im n:im n:im n:im n:im n:im n:acf s:im s:im", "im im im . . . im ."),
    ("PPPPPPPI", "s:acf n:as n:gt n:im n:im n:im n:cr n:acf r:acf", ". . . . . . . acf"),
    ("CCCXPCPP", "n:cr n:cr n:cr n:cr x:rt n:cr n:cr n:acf -", "cr cr cr . . cr . ."),
    ("CCCPPCXC", "s:cr n:cr n:cr n:cr n:cr n:cr n:cr x:cr s:cr", "cr cr cr . . cr . cr"),
    ("PPPPPPXP", "s:cr n:as n:gt n:im n:im n:im n:cr x:cr s:cr", ". . . . . . . ."),
]

MODE_WORD = {m.phase: m.name for m in modes.MODES if m.is_workflow_phase and m.phase}


def _phases(spec):
    assert len(spec) == len(ORDER)
    return {phase: dict(ENTRY[letter]) for phase, letter in zip(ORDER, spec)}


def _walk_token(step):
    if step.phase is None:
        assert step.kind == phase_walk.COMPLETE
        return "-"
    return f"{KIND[step.kind]}:{CODE[step.phase]}"


def _answer_token(answer):
    if answer is None:
        return "."
    return "-" if answer.complete else CODE[answer.phase]


def _workflow(phases):
    return {"phases": phases,
            "suggested_next": phase_walk.next_step(phases, ORDER).phase}


def test_the_table_covers_every_phase_and_entry_kind():
    assert len(CODE) == len(ORDER) and set(CODE) == set(ORDER)
    letters = {letter for spec, _, _ in TABLE for letter in spec}
    assert letters == set(ENTRY)
    for spec, walk, gated in TABLE:
        assert len(walk.split()) == len(ORDER) + 1, spec
        assert len(gated.split()) == len(ORDER), spec


@pytest.mark.parametrize("spec,walk,gated", TABLE, ids=[row[0] for row in TABLE])
def test_the_walk(spec, walk, gated):
    phases = _phases(spec)
    got = [_walk_token(phase_walk.next_step(phases, ORDER, after=start))
           for start in [None, *ORDER]]
    assert " ".join(got) == walk


@pytest.mark.parametrize("spec,walk,gated", TABLE, ids=[row[0] for row in TABLE])
def test_the_next_step_builder_answers_as_the_walk(spec, walk, gated):
    """``--after <phase>`` names the walk's phase; a finished phase not on
    record has no answer; otherwise the gated answer is the walk's too."""
    workflow = _workflow(_phases(spec))
    after, finished = [], []
    for phase in ORDER:
        kwargs = dict(plan_path="/nonexistent/plans/foo.md", workflow=workflow)
        after.append(_answer_token(np_.next_phase_commands(
            "plans/foo.md", "claude-code", None, None, after_mode=MODE_WORD[phase], **kwargs)))
        finished.append(_answer_token(np_.next_phase_commands(
            "plans/foo.md", "claude-code", None, MODE_WORD[phase], **kwargs)))
    assert " ".join(after) == " ".join(token.split(":")[-1] for token in walk.split()[1:])
    assert " ".join(finished) == gated


@pytest.mark.parametrize("spec,walk,gated", TABLE, ids=[row[0] for row in TABLE])
def test_full_falls_back_to_suggested_next(spec, walk, gated):
    workflow = _workflow(_phases(spec))
    answer = np_.next_phase_commands(
        "plans/foo.md", "claude-code", None, "full",
        plan_path="/nonexistent/plans/foo.md", workflow=workflow)
    assert _answer_token(answer) == walk.split()[0].split(":")[-1]


#: An implement entry with a code task waiting on a manual task the user can do.
GATED_IMPLEMENT = {
    "status": "completed",
    "manual_tasks": [{"id": "T009", "title": "play it"}],
    "manual_close_ids": ["T009"],
    "manual_waiting": {"T009": ["T010"]},
}


@pytest.mark.parametrize("spec,suggested,expected", [
    # The suggestion lies after implement: its resume leads.
    ("CCCPPCPP", "code-review", ["--resume", "--human-task", "T009=done"]),
    ("CCCPPCCP", "apply-code-fixes", ["--resume", "--human-task", "T009=done"]),
    # A suggestion before implement, implement itself, or none: no gate.
    ("XCCPPCPP", "review-plan", []),
    ("CCCPPCPP", "implement", []),
    ("CCCPPCCC", None, []),
])
def test_the_gate_resume_reads_the_suggestion_position(spec, suggested, expected):
    phases = _phases(spec)
    phases["implement"] = dict(GATED_IMPLEMENT)
    assert np_.gate_resume_suggested(
        {"phases": phases, "suggested_next": suggested}) == expected


def test_the_gate_needs_a_completed_implement():
    phases = _phases("CCCPPIPP")
    phases["implement"] = dict(GATED_IMPLEMENT, status="in_progress")
    assert np_.gate_resume_suggested(
        {"phases": phases, "suggested_next": "code-review"}) == []


def test_comes_after():
    assert phase_walk.comes_after(ORDER, "code-review", "implement")
    assert not phase_walk.comes_after(ORDER, "implement", "implement")
    assert not phase_walk.comes_after(ORDER, "review-plan", "implement")
    assert not phase_walk.comes_after(ORDER, "ask", "implement")
    assert not phase_walk.comes_after(ORDER, "code-review", "nope")


def test_a_missing_entry_is_open():
    """The builder's payloads always carry every phase; a hand-built one that
    does not reads the missing phase as open, never done."""
    assert phase_walk.next_step({}, ORDER) == phase_walk.Step("review-plan", phase_walk.NEXT)
    assert phase_walk.is_open({})
    assert not phase_walk.is_started({})


def test_prerequisites_re_exports_the_walk_constants():
    """One owner: the module the walk lives in, read through either name."""
    assert prereq.OPTIONAL_PHASES is phase_walk.OPTIONAL_PHASES
    assert prereq.DONE_STATUSES is phase_walk.DONE_STATUSES
    optional_rows = {m.phase for m in modes.MODES if m.optional_why}
    assert optional_rows == phase_walk.OPTIONAL_PHASES
