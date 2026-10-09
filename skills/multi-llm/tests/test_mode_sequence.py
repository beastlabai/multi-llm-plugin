"""The ``sequence`` column of ``modes.py`` and the three spellings it resolves to.

A composite row lists the *mode words* it runs. No reader uses them raw: the
surface emitter and the next step want phase words (``sequence_phases``,
``last_phase``), and ``--detach``'s close-out wants on-disk dir words. Each
spelling differs from the mode word for one phase (``review-code`` →
``code-review`` → ``code-review/``; ``apply-code-fixes`` → ``apply-code-fixes``
→ ``apply-fixes/``), which is what a reader using the raw words would miss.

``full``'s row carries its eight words as a literal, because a frozen row cannot
derive them from the table it sits in. Two other lists name the same phases —
``PREREQUISITE_PHASES`` (from ``is_workflow_phase``) and
``FullWorkflow.steps()`` (hardcoded) — and these asserts hold all three
together, so a phase added to one and not the others fails here rather than
silently changing ``full``'s ``writes_worktree`` or leaving a map unretired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import full_workflow_orchestrator as fwo  # noqa: E402
import modes as mode_table  # noqa: E402
from utils import detach  # noqa: E402

FULL = mode_table.resolve("full")


def test_full_runs_every_workflow_phase_in_table_order():
    assert FULL.is_composite
    assert FULL.runs_workflow
    assert FULL.sequence == tuple(
        m.name for m in mode_table.MODES if m.is_workflow_phase
    )
    assert FULL.sequence_phases == mode_table.PREREQUISITE_PHASES


def test_fulls_sequence_is_the_phases_its_state_machine_steps_through(tmp_path):
    """Read without running anything: ``steps()`` only builds the list."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    workflow = fwo.FullWorkflow(
        fwo.parse_args(["--plan-file", str(plan), "--yes"])
    )
    phases = tuple(step.phase for step in workflow.steps() if step.phase)
    assert phases == FULL.sequence_phases


def test_sequence_phases_resolves_mode_words_to_phase_words():
    assert "review-code" in FULL.sequence
    assert "code-review" in FULL.sequence_phases
    assert "review-code" not in FULL.sequence_phases
    assert FULL.sequence_phases[FULL.sequence.index("review-code")] == "code-review"


def test_last_phase_is_the_phase_a_composite_ends_on():
    assert FULL.last_phase == "apply-code-fixes"
    assert mode_table.resolve("review-plan").last_phase is None


def test_every_composite_word_names_a_workflow_phase():
    """A misspelt word raises in ``sequence_phases``; this finds it first."""
    for mode in mode_table.MODES:
        for word in mode.sequence:
            row = mode_table.resolve(word)
            assert row is not None and row.is_workflow_phase, (mode.name, word)


@pytest.mark.parametrize("mode", [m for m in mode_table.MODES if not m.sequence])
def test_a_row_without_a_sequence_is_not_composite(mode):
    assert not mode.is_composite
    assert mode.sequence_phases == ()
    assert mode.last_phase is None


def test_an_unresolvable_sequence_word_raises_rather_than_yield_none():
    row = mode_table.Mode(
        name="broken", positionals=(), summary="x", sequence=("no-such-mode",)
    )
    with pytest.raises(ValueError, match="no-such-mode"):
        row.sequence_phases


def test_the_detach_walk_resolves_fulls_dirs_to_the_on_disk_words():
    assert detach._driven_dir_names("full") == (
        "review-plan", "apply-suggestions", "generate-tasks", "review-tasks",
        "apply-task-suggestions", "implement", "code-review", "apply-fixes",
    )


@pytest.mark.parametrize("word", ["review-plan", "ask", "status", "", None,
                                  "no-such-mode"])
def test_the_detach_walk_resolves_nothing_for_a_mode_that_runs_no_other(word):
    assert detach._driven_dir_names(word) == ()


def test_the_detach_walk_opens_only_the_driven_dirs(tmp_path):
    """A checkpointed dir outside the sequence is never a candidate."""
    out = tmp_path / "a_plan"
    for name in ("full", "code-review", "apply-fixes", "ask", "implement"):
        (out / name).mkdir(parents=True)
    for name in ("full", "code-review", "apply-fixes", "ask"):
        (out / name / ".status.json").write_text("{}", encoding="utf-8")
    found = detach._driven_phase_dirs(str(out / "full"), "full")
    # Its own run dir is skipped; `implement/` holds no checkpoint; `ask/` is
    # not in the sequence.
    assert [Path(p).name for p in found] == ["code-review", "apply-fixes"]
