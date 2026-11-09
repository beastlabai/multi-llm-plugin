"""``util resume-state``: one fixture phase dir per resume case, a to f.

The review cores' resume step runs the command and does what its line says, so
each case of ``utils/resume_state.CASES`` is built on disk here and read back,
through ``classify`` and through the dispatcher route a host types. The
reference's case list is rendered from the same table (``--check`` below).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from unittest.mock import patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import multi_llm  # noqa: E402
from tests.tools import render_resume_cases  # noqa: E402
from utils import resume_state  # noqa: E402
from utils.cli_common import RESULT_MARKER  # noqa: E402
from utils.json_extractor import sanitize_model_name  # noqa: E402

MODELS = ["codex:gpt-6-astra:xhigh", "pi:openrouter/deepseek/deepseek-v4.1-flash"]
EARLY, LATE = "2026-11-01T10:00:00", "2026-11-02T10:00:00"


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plans" / "feat.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Feat\n\nDo it.\n", encoding="utf-8")
    return plan


def _phase_dir(plan: Path, phase: str) -> Path:
    path = plan.parent / plan.stem / phase
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _state(plan: Path, completed: Dict[str, str]) -> None:
    folder = plan.parent / plan.stem
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / "state.json", {"phases_completed": completed})


def _status(phase_dir: Path, **fields) -> None:
    _write_json(phase_dir / ".status.json", fields)


def _result(phase_dir: Path, spec: str, value=None) -> None:
    _write_json(phase_dir / f"{sanitize_model_name(spec)}.json",
                [{"title": "x"}] if value is None else value)


def _tasks(phase_dir: Path, batches: List[int]) -> None:
    _write_json(phase_dir / "validation_tasks.json",
                {"batches": [{"batch_index": index} for index in batches]})


#: How far back the previous round's files are dated, against a run started now.
PREVIOUS_ROUND_S = 2 * 24 * 3600


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def _previous_round(phase_dir: Path) -> None:
    """A finished round's files, as a `--force` re-run leaves them until it
    regroups: report, verdicts, task file and a batch, all two days old."""
    _tasks(phase_dir, [1])
    _write_json(phase_dir / "validation_batch_1.json", [])
    _write_json(phase_dir / "validation.json", [])
    (phase_dir / "report.md").write_text("# Old report\n", encoding="utf-8")
    for name in ("validation_tasks.json", "validation_batch_1.json",
                 "validation.json", "report.md"):
        _age(phase_dir / name, PREVIOUS_ROUND_S)


def _started(seconds_ago: float = 0.0) -> str:
    """A `.status.json` `started_at`, as the fan-out's opening write stamps it."""
    return (datetime.now() - timedelta(seconds=seconds_ago)).isoformat()


#: The scope id the review set on disk records (``review-scope.md``).
SCOPE = "5c0be1d0c0de"
OTHER_SCOPE = "0ther5c0be00"


def _review_set(phase_dir: Path, scope_id: str = SCOPE) -> None:
    """A whole ``review-scope.md`` + ``diff/`` pair naming *scope_id*."""
    (phase_dir / "diff").mkdir(exist_ok=True)
    (phase_dir / "review-scope.md").write_text(
        f"# Review scope\n\n- Scope id: `{scope_id}`\n\nDiff coverage: complete\n",
        encoding="utf-8")


def _ledger(phase_dir: Path, entries: Dict[str, str]) -> None:
    _write_json(phase_dir / ".review-scope.json",
                {sanitize_model_name(spec): scope for spec, scope in entries.items()})


def _build(case: str, plan: Path, phase: str) -> Optional[Path]:
    """Lay out *case* for *phase* of *plan*; the phase dir (``None``: no folder)."""
    if case == "e-empty":
        return None
    phase_dir = _phase_dir(plan, phase)
    if case == "a-recorded":
        _state(plan, {phase: LATE})
    elif case == "c-report":
        # Verdicts and a report, but no completion recorded and no checkpoint.
        _write_json(phase_dir / "validation.json", [])
        (phase_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    elif case == "c-no-report":
        _write_json(phase_dir / "validation.json", [])
    elif case == "b-all":
        _tasks(phase_dir, [1, 2])
        _write_json(phase_dir / "validation_batch_1.json", [])
        _write_json(phase_dir / "validation_batch_2.json", [])
        _status(phase_dir, state="validation_pending")
    elif case == "b-some":
        _tasks(phase_dir, [1, 2, 3])
        _write_json(phase_dir / "validation_batch_2.json", [])
        # A retired round's batch is not this round's.
        _write_json(phase_dir / "validation_batch_1.20261101T1000.stale.json", [])
        _status(phase_dir, state="validation_pending")
    elif case == "c-tasks":
        # A task file the checkpoint does not vouch for.
        _tasks(phase_dir, [1, 2])
        _status(phase_dir, state="models_complete")
    elif case == "c-batches":
        # Batch files with no checkpoint at all.
        _tasks(phase_dir, [1, 2])
        _write_json(phase_dir / "validation_batch_1.json", [])
    elif case == "c-unreadable-tasks":
        (phase_dir / "validation_tasks.json").write_text("{not json", encoding="utf-8")
        _status(phase_dir, state="validation_pending", models_requested=MODELS)
    elif case == "c-short-tasks":
        # A task file whose own count disagrees with its batch list.
        _write_json(phase_dir / "validation_tasks.json",
                    {"total_batches": 3, "batches": [{"batch_index": 1}]})
        _status(phase_dir, state="validation_pending", models_requested=MODELS)
    elif case == "f":
        _tasks(phase_dir, [1])
        _status(phase_dir, state="validation_pending")
    elif case == "d-all":
        _status(phase_dir, state="models_running", models_requested=MODELS)
        for spec in MODELS:
            _result(phase_dir, spec)
    elif case == "d-some":
        _status(phase_dir, state="models_running", models_requested=MODELS)
        _result(phase_dir, MODELS[0])
        # Not a list: the fan-outs' [SKIP] test does not keep it.
        _result(phase_dir, MODELS[1], value={"error": "cut off"})
    elif case == "d-grouped":
        _write_json(phase_dir / "grouped.json", {"groups": []})
    elif case == "e-stale":
        _state(plan, {phase: EARLY, "implement": LATE})
        _write_json(phase_dir / "validation.json", [])
        (phase_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    elif case == "e-unrelated":
        (phase_dir / "orchestrator-run.log").write_text("", encoding="utf-8")
    elif case == "rerun-old-round":
        # A `--force` re-run interrupted before any model answered: the
        # previous round's files are older than the run's start.
        _previous_round(phase_dir)
        _status(phase_dir, state="models_running", started_at=_started(),
                models_requested=MODELS)
    elif case == "rerun-old-round-no-start":
        _previous_round(phase_dir)
        _status(phase_dir, state="models_running", models_requested=MODELS)
    elif case == "rerun-old-round-one-result":
        _previous_round(phase_dir)
        _status(phase_dir, state="models_running", started_at=_started(),
                models_requested=MODELS)
        _result(phase_dir, MODELS[0])
    elif case == "c-dated-this-round":
        # Verdicts dated after the run's start: still no proof they are its own.
        _write_json(phase_dir / "validation.json", [])
        (phase_dir / "report.md").write_text("# Report\n", encoding="utf-8")
        _status(phase_dir, state="grouping_complete", started_at=_started(3600),
                models_requested=MODELS)
    elif case == "c-all-kept":
        # Every model's result kept, but the verdicts beside them may be an
        # earlier round's: a reaggregate would reuse them, a re-run regroups.
        _previous_round(phase_dir)
        _status(phase_dir, state="models_complete", started_at=_started(),
                models_requested=MODELS)
        for spec in MODELS:
            _result(phase_dir, spec)
        if phase == "code-review":
            _review_set(phase_dir)
            _ledger(phase_dir, {spec: SCOPE for spec in MODELS})
    elif case == "d-status-only":
        _status(phase_dir, state="models_running", started_at=_started(),
                models_requested=MODELS)
    elif case == "e-status-complete":
        _status(phase_dir, state="complete", started_at=_started(),
                models_requested=MODELS)
    elif case == "d-scope-kept":
        _review_set(phase_dir)
        _status(phase_dir, state="models_running", started_at=_started(),
                models_requested=MODELS)
        for spec in MODELS:
            _result(phase_dir, spec)
        _ledger(phase_dir, {spec: SCOPE for spec in MODELS})
    elif case == "d-scope-unrecorded":
        # Results with no ledger entry: reviewed scope unknown.
        _review_set(phase_dir)
        _status(phase_dir, state="models_running", started_at=_started(),
                models_requested=MODELS)
        for spec in MODELS:
            _result(phase_dir, spec)
    elif case == "d-scope-mixed":
        _review_set(phase_dir)
        _status(phase_dir, state="models_running", started_at=_started(),
                models_requested=MODELS)
        for spec in MODELS:
            _result(phase_dir, spec)
        _ledger(phase_dir, {MODELS[0]: SCOPE, MODELS[1]: OTHER_SCOPE})
    elif case == "d-scope-no-set":
        # A ledger, but no review set on disk to name the scope.
        _status(phase_dir, state="models_running", started_at=_started(),
                models_requested=MODELS)
        for spec in MODELS:
            _result(phase_dir, spec)
        _ledger(phase_dir, {spec: SCOPE for spec in MODELS})
    else:  # pragma: no cover - a typo in the table below
        raise AssertionError(case)
    return phase_dir


FORCE = ("--force",)
#: Case c's answer: validation files of an uncertain round.
UNCERTAIN = "uncertain"

#: (fixture, phase) -> (case, action, models, batches, flags, ownership).
EXPECTED = {
    ("a-recorded", "review-plan"): ("a", "report", (), (), (), "-"),
    ("b-all", "review-plan"): ("b", "reaggregate", (), (), (), "-"),
    ("b-some", "code-review"): ("b", "validate", (), (1, 3), (), "-"),
    ("c-report", "review-tasks"): ("c", "rerun", (), (), FORCE, UNCERTAIN),
    ("c-no-report", "code-review"): ("c", "rerun", (), (), FORCE, UNCERTAIN),
    ("c-tasks", "review-plan"): ("c", "rerun", (), (), FORCE, UNCERTAIN),
    ("c-batches", "review-tasks"): ("c", "rerun", (), (), FORCE, UNCERTAIN),
    ("c-unreadable-tasks", "review-plan"): ("c", "rerun", tuple(MODELS), (), FORCE, UNCERTAIN),
    ("c-short-tasks", "code-review"): ("c", "rerun", tuple(MODELS), (), FORCE, UNCERTAIN),
    ("c-dated-this-round", "review-plan"): ("c", "rerun", tuple(MODELS), (), FORCE, UNCERTAIN),
    ("d-all", "review-plan"): ("d", "reaggregate", (), (), (), "-"),
    ("d-some", "review-tasks"): ("d", "rerun", tuple(MODELS), (), (), "-"),
    ("d-some", "code-review"): ("d", "rerun", tuple(MODELS), (), FORCE, "-"),
    ("d-grouped", "code-review"): ("d", "reaggregate", (), (), (), "-"),
    ("e-empty", "review-plan"): ("e", "run", (), (), (), "-"),
    ("e-unrelated", "review-tasks"): ("e", "run", (), (), (), "-"),
    ("e-stale", "code-review"): ("e", "run", (), (), (), "-"),
    ("f", "review-tasks"): ("f", "validate", (), (1,), (), "-"),
    ("e-status-complete", "code-review"): ("e", "run", (), (), (), "-"),
}

#: The interrupted fan-outs: (fixture, phase) -> (case, action, models,
#: batches, flags, ownership), each also checked against the orchestrator's
#: own decision (``TestTheAnswerAgreesWithTheOrchestrator``).
INTERRUPTED = {
    **{
        (fixture, phase): ("c", "rerun", tuple(MODELS), (), FORCE, UNCERTAIN)
        for fixture in ("rerun-old-round", "rerun-old-round-no-start",
                        "rerun-old-round-one-result", "c-all-kept")
        for phase in ("review-plan", "review-tasks", "code-review")
    },
    **{
        ("d-status-only", phase): ("d", "rerun", tuple(MODELS), (),
                                   FORCE if phase == "code-review" else (), "-")
        for phase in ("review-plan", "review-tasks", "code-review")
    },
    ("d-scope-kept", "code-review"): ("d", "reaggregate", (), (), (), "-"),
    ("d-scope-unrecorded", "code-review"): ("d", "rerun", tuple(MODELS), (), FORCE, "-"),
    ("d-scope-mixed", "code-review"): ("d", "rerun", tuple(MODELS), (), FORCE, "-"),
    ("d-scope-no-set", "code-review"): ("d", "rerun", tuple(MODELS), (), FORCE, "-"),
}


def _answer(state: "resume_state.ResumeState"):
    return (state.case, state.action, state.models, state.batches, state.flags,
            state.ownership)


def _tree(root: Path) -> Dict[str, int]:
    return {str(path.relative_to(root)): path.stat().st_mtime_ns
            for path in sorted(root.rglob("*"))}


@pytest.mark.parametrize("fixture,phase", sorted({**EXPECTED, **INTERRUPTED}))
def test_each_case_is_read_from_its_fixture_dir(tmp_path, fixture, phase):
    plan = _plan(tmp_path)
    _build(fixture, plan, phase)
    before = _tree(tmp_path)
    got = resume_state.classify(plan.resolve(), phase)
    expected = {**EXPECTED, **INTERRUPTED}[(fixture, phase)]
    assert _answer(got) == expected
    assert _tree(tmp_path) == before, "util resume-state wrote something"


def test_the_fixtures_cover_every_case_and_every_action():
    cases = {expected[0] for expected in EXPECTED.values()}
    actions = {expected[1] for expected in EXPECTED.values()}
    assert cases == {case.letter for case in resume_state.CASES} == set("abcdef")
    assert actions == set(resume_state.ACTIONS)


def test_no_answer_is_ever_rerun_all(tmp_path):
    for index, (fixture, phase) in enumerate(sorted({**EXPECTED, **INTERRUPTED})):
        plan = _plan(tmp_path / str(index))
        _build(fixture, plan, phase)
        line = resume_state.classify(plan.resolve(), phase).line()
        assert "rerun-all" not in line, line


def test_the_line_spells_every_key_and_dashes_an_empty_one():
    state = resume_state.ResumeState("d", "rerun", models=("a:b", "c:d"), flags=("--force",))
    assert state.line() == (
        "[RESUME_STATE] case=d action=rerun models=a:b,c:d batches=- flags=--force "
        "ownership=-")
    assert resume_state.ResumeState("e", "run").line() == (
        "[RESUME_STATE] case=e action=run models=- batches=- flags=- ownership=-")
    uncertain = resume_state.ResumeState("c", "rerun", flags=("--force",),
                                         ownership="uncertain", note="why")
    assert uncertain.line() == (
        "[RESUME_STATE] case=c action=rerun models=- batches=- flags=--force "
        "ownership=uncertain")


# ---------------------------------------------------------------------------
# The answer against the orchestrator's own decision
# ---------------------------------------------------------------------------

def _primary_guard_completed(plan: Path, phase: str) -> bool:
    """The completed-phase guard each review orchestrator runs first."""
    from utils.state_manager import StateManager

    if not (plan.parent / plan.stem / "state.json").is_file():
        return False
    return StateManager(plan, read_only=True).is_phase_completed(phase)


def _dispatched(plan: Path, phase: str, models, scope_id: Optional[str]) -> Set[str]:
    """The models the phase's real fan-out hands to a provider when launched
    with *models*: ``run_all_reviews`` or the base ``run_all_models``, with
    only the per-model provider call stubbed (nothing reaches a provider).
    Both run on the base's ``run_models_concurrent``, whose provider caps and
    launch spacing are switched off here."""
    from utils import review_orchestrator_base as base

    called: List[str] = []
    out_dir = plan.parent / plan.stem
    mechanics = (
        patch.object(base, "get_provider_max_concurrent", return_value=None),
        patch.object(base, "plan_launch_limiter", return_value=(None, 0)),
        patch.object(base, "slot_launch",
                     side_effect=lambda *a, **k: contextlib.nullcontext()),
    )
    with contextlib.ExitStack() as stack:
        for mechanic in mechanics:
            stack.enter_context(mechanic)
        if phase == "code-review":
            import code_review_orchestrator as review_code

            async def fake_review(spec, *_args, **_kwargs):
                called.append(spec)
                return True, [], None

            with patch.object(review_code, "run_single_review", fake_review):
                asyncio.run(review_code.run_all_reviews(
                    list(models), plan, out_dir, 10, 2, scope_id=scope_id))
        else:
            async def fake_model(spec, *_args, **_kwargs):
                called.append(spec)
                return {"success": True, "output": "[]", "error": None, "prompt": "",
                        "stdout": "", "stderr": "", "duration_seconds": 0.0,
                        "model_spec": spec}

            with patch.object(base, "run_single_model", fake_model):
                asyncio.run(base.run_all_models(
                    list(models), str(plan), "", lambda spec: {}, phase, "plan_review",
                    10, 2, out_dir=str(out_dir), prefix=plan.stem))
    return set(called)


class TestTheAnswerAgreesWithTheOrchestrator:
    """Each interrupted fan-out's answer, checked against what the orchestrator
    itself does: its primary guard (not complete, so a launch fans out),
    review-code's partial-completion guard (which a ``rerun`` must pass with
    ``--force``), and the fan-out's own reuse test, run for real with the
    provider call stubbed and the scope the review set on disk records (the
    one an unchanged change's re-run computes)."""

    @pytest.mark.parametrize("fixture,phase", sorted(
        key for key in INTERRUPTED if key[0] != "d-scope-no-set"))
    def test_the_fanout_runs_exactly_the_models_the_answer_does_not_keep(
            self, tmp_path, fixture, phase):
        import code_review_orchestrator as review_code

        plan = _plan(tmp_path)
        phase_dir = _build(fixture, plan, phase)
        got = resume_state.classify(plan.resolve(), phase)
        kept = {spec for spec in MODELS
                if resume_state._kept_on_rerun(phase_dir, phase, spec)}  # noqa: SLF001
        assert not _primary_guard_completed(plan, phase)
        assert got.action in ("rerun", "reaggregate")
        if phase == "code-review" and got.action == "rerun":
            refused_plain = (
                (phase_dir / "validation_tasks.json").exists()
                or (phase_dir / "grouped.json").exists()
                or review_code.leftover_model_results(phase_dir))
            assert "--force" in got.flags or not refused_plain
        dispatched = _dispatched(plan.resolve(), phase, MODELS,
                                 SCOPE if phase == "code-review" else None)
        assert dispatched == set(MODELS) - kept
        if got.action == "reaggregate":
            assert dispatched == set(), "a reaggregate would report without these"
        else:
            # The original run's full selection, never a new one.
            assert got.models == tuple(MODELS)
            if got.case == "d":
                assert dispatched, "nothing to re-run: the answer should be reaggregate"
            else:
                # Case c re-runs even with every result kept: the re-run
                # regroups, where a reaggregate would reuse the verdicts.
                assert got.flags == ("--force",)

    def test_with_no_review_set_a_rerun_is_the_safe_answer(self, tmp_path):
        """No ``review-scope.md``: the re-run's scope is not on disk, so the
        answer is ``rerun``, which loses nothing whichever scope it computes."""
        for index, scope in enumerate((SCOPE, OTHER_SCOPE)):
            plan = _plan(tmp_path / str(index))
            _build("d-scope-no-set", plan, "code-review")
            got = resume_state.classify(plan.resolve(), "code-review")
            assert (got.action, got.models, got.flags) == ("rerun", tuple(MODELS), ("--force",))
            dispatched = _dispatched(plan.resolve(), "code-review", got.models, scope)
            assert dispatched == (set() if scope == SCOPE else set(MODELS))

    @pytest.mark.parametrize("phase", resume_state.REVIEW_PHASES)
    def test_a_previous_round_s_report_is_not_the_interrupted_one_s(self, tmp_path, phase):
        """Case a reads ``state.json`` alone: the old report would have been
        answered ``report`` while the primary guard reads the phase as pending
        and a launch re-runs every model."""
        plan = _plan(tmp_path)
        _build("rerun-old-round", plan, phase)
        _state(plan, {})
        assert not _primary_guard_completed(plan, phase)
        assert resume_state.classify(plan.resolve(), phase).action == "rerun"


# ---------------------------------------------------------------------------
# No file date decides a round (Codex review 2 of U62b)
# ---------------------------------------------------------------------------

def _date_all(phase_dir: Path, stamp: float) -> None:
    for path in phase_dir.iterdir():
        if path.is_file() and path.name != ".status.json":
            os.utime(path, (stamp, stamp))


#: How each variant dates the files against the run's recorded start:
#: (started_at seconds ago or None, the files' mtime as seconds from now).
DATINGS = {
    # The clock was set back after the earlier round: its files look newer.
    "clock-rollback": (3600, 0.0),
    # Copied from a machine whose clock ran a day fast, dates kept.
    "faster-clock-transfer": (0, 24 * 3600.0),
    # A coarse file system: the file and the start share one stamp.
    "equal-timestamps": ("equal", None),
    # A copy that kept old dates, or a file system that rounds down.
    "preserved-older": (0, -PREVIOUS_ROUND_S),
    # No start recorded at all.
    "no-started-at": (None, -PREVIOUS_ROUND_S),
}


def _dated_layout(plan: Path, phase: str, layout: str, dating: str) -> Path:
    """*layout* for *phase*, its files dated as *dating* says.

    ``old-round-<state>``: a finished round's report, verdicts, task file and
    batch, under a re-run's checkpoint at ``<state>``. ``current-partial``:
    this round's two-batch task file under ``validation_pending``, batch 1
    done and batch 2 missing.
    """
    phase_dir = _phase_dir(plan, phase)
    ago, offset = DATINGS[dating]
    now = time.time()
    fields: Dict[str, object] = {"models_requested": MODELS}
    if ago == "equal":
        started = datetime.now().replace(microsecond=0)
        fields["started_at"] = started.isoformat()
        stamp = started.timestamp()
    else:
        if ago is not None:
            fields["started_at"] = _started(ago)
        stamp = now + offset
    if layout == "current-partial":
        _tasks(phase_dir, [1, 2])
        _write_json(phase_dir / "validation_batch_1.json", [])
        fields["state"] = "validation_pending"
    else:
        _previous_round(phase_dir)
        fields["state"] = layout.removeprefix("old-round-")
    _write_json(phase_dir / ".status.json", fields)
    _date_all(phase_dir, stamp)
    return phase_dir


#: layout -> the answer every dating gives.
DATED_ANSWERS = {
    **{f"old-round-{state}": ("c", "rerun", tuple(MODELS), (), FORCE, UNCERTAIN)
       for state in ("models_running", "models_complete", "grouping_complete")},
    "current-partial": ("b", "validate", (), (2,), (), "-"),
}


class TestNoFileDateDecidesARound:
    """Codex's variants: a clock set back, a transfer from a faster clock,
    equal stamps, rounded-down or preserved dates, and no recorded start.
    Each layout gives one answer whatever the dates, and no layout whose
    primary guard reads the phase as pending is answered ``report``."""

    @pytest.mark.parametrize("phase", resume_state.REVIEW_PHASES)
    @pytest.mark.parametrize("dating", sorted(DATINGS))
    @pytest.mark.parametrize("layout", sorted(DATED_ANSWERS))
    def test_the_answer_ignores_the_dates(self, tmp_path, layout, dating, phase):
        plan = _plan(tmp_path)
        _dated_layout(plan, phase, layout, dating)
        got = resume_state.classify(plan.resolve(), phase)
        assert _answer(got) == DATED_ANSWERS[layout]
        assert not _primary_guard_completed(plan, phase)

    @pytest.mark.parametrize("phase", resume_state.REVIEW_PHASES)
    def test_the_reopened_phase_is_not_reported(self, tmp_path, phase):
        """Codex's first repro: a completion older than the phase's newer
        start (the primary guard reads it pending), the earlier round's
        verdicts and report on disk, and no checkpoint at all."""
        plan = _plan(tmp_path)
        phase_dir = _phase_dir(plan, phase)
        _write_json(plan.parent / plan.stem / "state.json", {
            "phases_completed": {phase: EARLY}, "phases_started": {phase: LATE}})
        _write_json(phase_dir / "validation.json", [])
        (phase_dir / "report.md").write_text("# Old report\n", encoding="utf-8")
        assert not _primary_guard_completed(plan, phase)
        got = resume_state.classify(plan.resolve(), phase)
        assert _answer(got) == ("c", "rerun", (), (), FORCE, UNCERTAIN)


def test_report_only_ever_follows_the_primary_guard(tmp_path):
    """Every fixture, on every phase: ``report`` exactly when the
    orchestrator's own completed-phase guard says complete."""
    fixtures = sorted({fixture for fixture, _phase in {**EXPECTED, **INTERRUPTED}})
    seen = set()
    for phase in resume_state.REVIEW_PHASES:
        for fixture in fixtures:
            plan = _plan(tmp_path / phase / fixture)
            _build(fixture, plan, phase)
            got = resume_state.classify(plan.resolve(), phase)
            assert (got.action == "report") == _primary_guard_completed(plan, phase), (
                fixture, phase, got.line())
            seen.add(got.action)
        for layout in DATED_ANSWERS:
            for dating in DATINGS:
                plan = _plan(tmp_path / phase / f"{layout}-{dating}")
                _dated_layout(plan, phase, layout, dating)
                assert resume_state.classify(plan.resolve(), phase).action != "report"
    assert "report" in seen


class TestTheSharedReusePredicate:
    """``existing_result_reuse``: review-code's one reuse test, which
    ``run_all_reviews`` and ``util resume-state`` both call."""

    def _verdict(self, phase_dir: Path, scope_id: Optional[str]) -> str:
        import code_review_orchestrator as review_code

        return review_code.existing_result_reuse(
            phase_dir, sanitize_model_name(MODELS[0]), scope_id).verdict

    def test_each_verdict(self, tmp_path):
        phase_dir = tmp_path
        target = phase_dir / f"{sanitize_model_name(MODELS[0])}.json"
        assert self._verdict(phase_dir, SCOPE) == "absent"
        target.write_bytes(b"")
        assert self._verdict(phase_dir, SCOPE) == "absent"
        target.write_bytes(b"\xff\xfe not utf-8")
        assert self._verdict(phase_dir, SCOPE) == "corrupt"
        target.write_text("{not json", encoding="utf-8")
        assert self._verdict(phase_dir, SCOPE) == "corrupt"
        target.write_text('{"error": "cut off"}', encoding="utf-8")
        assert self._verdict(phase_dir, SCOPE) == "not_a_list"
        target.write_text("[]", encoding="utf-8")
        assert self._verdict(phase_dir, SCOPE) == "scope_changed"
        assert self._verdict(phase_dir, None) == "reuse"
        _ledger(phase_dir, {MODELS[0]: OTHER_SCOPE})
        assert self._verdict(phase_dir, SCOPE) == "scope_changed"
        _ledger(phase_dir, {MODELS[0]: SCOPE})
        assert self._verdict(phase_dir, SCOPE) == "reuse"


class TestTheDispatcherRoute:
    def _run(self, argv, capsys):
        code = multi_llm.main(["util", "resume-state", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_review_code_and_its_commit_door_read_the_code_review_dir(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _build("d-some", plan, "code-review")
        for mode in ("review-code", "code-review", "review-commits"):
            code, out, _err = self._run([mode, str(plan)], capsys)
            assert code == 0, out
            lines = [line for line in out.splitlines() if line.startswith("[RESUME_STATE]")]
            assert lines == [
                f"[RESUME_STATE] case=d action=rerun models={','.join(MODELS)} "
                "batches=- flags=--force ownership=-"
            ]
            assert out.count(RESULT_MARKER) == 1

    def test_an_uncertain_round_says_why_on_stderr(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _build("rerun-old-round", plan, "review-plan")
        code, out, err = self._run(["review-plan", str(plan)], capsys)
        assert code == 0
        assert (f"[RESUME_STATE] case=c action=rerun models={','.join(MODELS)} "
                "batches=- flags=--force ownership=uncertain") in out
        assert "may be an earlier round's" in err
        assert ".status.json reads `models_running`" in err
        assert "[RESUME_STATE]" not in err

    def test_a_plan_with_no_output_folder_is_case_e_and_gets_none(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        code, out, _err = self._run(["review-plan", str(plan)], capsys)
        assert code == 0
        assert "[RESUME_STATE] case=e action=run" in out
        assert not (plan.parent / plan.stem).exists()

    @pytest.mark.parametrize("mode", ["implement", "apply-suggestions", "no-such-mode"])
    def test_a_mode_without_a_resume_step_is_refused(self, tmp_path, capsys, mode):
        plan = _plan(tmp_path)
        code, out, err = self._run([mode, str(plan)], capsys)
        assert code == 3
        assert "[RESUME_STATE]" not in out
        assert "has no resume step" in err

    def test_a_missing_plan_is_refused(self, tmp_path, capsys):
        code, out, err = self._run(["review-plan", str(tmp_path / "nope.md")], capsys)
        assert code == 3
        assert "plan not found" in err


class TestTheReferenceIsRenderedFromTheTable:
    def test_the_committed_block_is_fresh(self):
        assert render_resume_cases.main(["--check"]) == 0

    def test_every_case_and_action_word_is_in_the_reference(self):
        text = render_resume_cases.REFERENCE.read_text(encoding="utf-8")
        for case in resume_state.CASES:
            assert f"\n{case.letter}. If {case.when}:" in text
        for word in resume_state.ACTIONS:
            assert f"- `{word}`:" in text

    def test_a_stale_block_fails_the_check(self, tmp_path, monkeypatch, capsys):
        copy = tmp_path / "resume-detection.md"
        copy.write_text(render_resume_cases.REFERENCE.read_text(encoding="utf-8")
                        .replace("a. If", "a. When"), encoding="utf-8")
        monkeypatch.setattr(render_resume_cases, "REFERENCE", copy)
        assert render_resume_cases.main(["--check"]) == 1
        assert "is stale" in capsys.readouterr().err
        assert render_resume_cases.main([]) == 0
        assert render_resume_cases.main(["--check"]) == 0

    def test_lost_markers_fail_the_check(self, tmp_path, monkeypatch):
        copy = tmp_path / "resume-detection.md"
        copy.write_text("# Resume Detection\n", encoding="utf-8")
        monkeypatch.setattr(render_resume_cases, "REFERENCE", copy)
        assert render_resume_cases.main(["--check"]) == 1


def test_the_three_review_cores_run_the_command():
    for core, mode in (("review-plan", "review-plan"), ("review-tasks", "review-tasks"),
                       ("review-code", "review-code")):
        text = (SKILL_ROOT / "instructions" / f"{core}.md").read_text(encoding="utf-8")
        assert f'multi_llm.py" util resume-state {mode} "$PLAN_PATH"' in text, core
        for word in resume_state.ACTIONS:
            assert f"`action={word}`" in text, (core, word)
