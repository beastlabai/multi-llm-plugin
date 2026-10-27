"""§9.3.a/b — every plan-file pre-flight pins the process run anchor (W4).

Three properties, one file:

1. **Every plan-file mode pins.** Parametrized over ``modes.MODES`` rather than
   over a literal roster (M0.a's rule), so a mode added to the table later is
   either covered here on the day it arrives or listed in ``EXEMPT`` with a
   reason. The exemptions are declared, never implicit. A commit mode
   (``POS_REVS``) takes a range rather than a plan and pins through the phase
   it fronts, on the commit brief the dispatcher hands it as ``--plan-file``;
   that is driven here too, from the table.
2. **The six pinners agree on one spelling.** ``set_run_anchor`` compares
   strings, so two pre-flights that normalize a symlinked plan path differently
   pin two different strings and the second one raises mid-run. ``full`` runs
   several modes in one interpreter against one ``--plan-file``, which is
   exactly that shape.
3. **``_setup``'s ``--skip`` / ``--mark-completed`` early exits still exit
   without pinning**, asserted directly so that block and the pin below it
   cannot quietly reorder each other.

The pre-flights are driven for real — the mode table's own entry point, the
dispatcher's own positional normalization — and stopped at the pin by an
instrumented ``set_run_anchor`` that performs the real pin and then raises. The
real pin matters: a stubbed one could not collide, and (2) is a collision test.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import modes as mode_table                                          # noqa: E402
import multi_llm                                                    # noqa: E402
import utils.provider_registry as registry                          # noqa: E402
from utils.output_handler import sanitize_prefix                    # noqa: E402


def _can_symlink() -> bool:
    """Probe whether this host can actually create symlinks.

    Platform checks are wrong in both directions (see the same helper in
    tests/test_provider_registry.py).
    """
    with tempfile.TemporaryDirectory() as d:
        try:
            os.symlink(os.path.join(d, "t"), os.path.join(d, "l"))
            return True
        except (OSError, NotImplementedError):
            return False


_requires_symlinks = pytest.mark.skipif(
    not _can_symlink(), reason="requires symlink-creation privilege"
)


# The real function, captured once at import time so an instrumented recorder
# never ends up wrapping a previous recorder.
_REAL_SET_RUN_ANCHOR = registry.set_run_anchor


# ---------------------------------------------------------------------------
# The exemptions, declared
# ---------------------------------------------------------------------------

#: Modes that take a ``<plan>`` positional and still do not pin, each with the
#: reason it is not a gap. Every other plan-taking mode in ``modes.MODES`` must
#: pin, and a mode that takes no plan file at all is exempt structurally (it has
#: no ``--plan-file`` to resolve), which is derived from the table below rather
#: than listed.
EXEMPT: dict = {
    "full": (
        "pins through the sub-mode it dispatches — every phase "
        "full_workflow_orchestrator.run_mode runs is one of the modes asserted "
        "here, against the same --plan-file; demonstrated by "
        "test_every_composite_pins_through_the_sub_mode_it_dispatches below"
    ),
    "improve-plan": (
        "pins through the sub-mode it dispatches — the composite owns no "
        "pre-flight pin; the first phase ImproveWorkflow.run_mode dispatches "
        "against the same --plan-file (review-plan, or apply-suggestions when "
        "the review is recorded) is a mode asserted here, and nothing before "
        "that dispatch reads provider config (only argv, state.json and the "
        "typed scope); a run with nothing to do exits 2 before any dispatch "
        "and starts no run to anchor. Demonstrated by "
        "test_every_composite_pins_through_the_sub_mode_it_dispatches below"
    ),
    "improve-code": (
        "pins through the sub-mode it dispatches — the composite owns no "
        "pre-flight pin; the first phase ImproveWorkflow.run_mode dispatches "
        "against the same --plan-file (review-code, or apply-code-fixes when "
        "the review is recorded) is a mode asserted here, and nothing before "
        "that dispatch reads provider config (only argv, state.json and the "
        "typed scope); a run with nothing to do exits 2 before any dispatch "
        "and starts no run to anchor. Demonstrated by "
        "test_every_composite_pins_through_the_sub_mode_it_dispatches below"
    ),
    "status": (
        "a reporting view over state.json and {phase}/.status.json: it starts "
        "no run and invokes no model, and its one provider read "
        "(resolve_subagent_runner, for the runner line) already passes an "
        "explicit anchor=plan_path, so it never consults the process pin"
    ),
}


def _plan_modes() -> List[mode_table.Mode]:
    """Every mode whose pre-flight resolves a ``--plan-file``."""
    return [m for m in mode_table.MODES if mode_table.POS_PLAN in m.positionals]


def _pinning_modes() -> List[mode_table.Mode]:
    return [m for m in _plan_modes() if m.name not in EXEMPT]


def _front_doors() -> List[mode_table.Mode]:
    """The commit modes (plans/review-commits.md §2.1): a range, not a plan."""
    return [m for m in mode_table.MODES if mode_table.POS_REVS in m.positionals]


def _fronted(door: mode_table.Mode) -> mode_table.Mode:
    """The phase mode a front door runs first on its brief: the row it fronts
    (``commit_view_of``), or a composite's first step."""
    word = door.commit_view_of or (door.sequence[0] if door.sequence else None)
    assert word, f"{door.name} fronts no mode"
    return mode_table.resolve(word)


def test_every_declared_exemption_is_still_a_mode():
    """A renamed or deleted mode must not leave a stale free pass behind."""
    names = {m.name for m in mode_table.MODES}
    unknown = sorted(set(EXEMPT) - names)
    assert not unknown, (
        f"EXEMPT names modes that are not in the table any more: {unknown}. "
        "Delete the entry, or fix the name."
    )
    not_plan_modes = sorted(set(EXEMPT) - {m.name for m in _plan_modes()})
    assert not not_plan_modes, (
        f"{not_plan_modes} take no <plan> positional, so they are exempt "
        "structurally and do not need a declared reason."
    )


# ---------------------------------------------------------------------------
# Driving a pre-flight, and stopping it at its pin
# ---------------------------------------------------------------------------

class _Pinned(BaseException):
    """Raised by the instrumented pin to stop a run the moment it anchors.

    Deliberately a ``BaseException``: ``ApplyOrchestratorBase.run`` catches
    ``Exception`` and turns it into exit 1, which would hide the stop (and, in
    the collision test, would hide a real ``RuntimeError`` too).
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.path = path


class _PinRecorder:
    """Instrumentation for every bound name of ``set_run_anchor``.

    Patched and unpatched by hand rather than through ``monkeypatch``, because
    the interesting bindings are made *during* the test: an orchestrator
    imported by ``_load_target`` runs ``from utils.provider_registry import
    set_run_anchor`` at that moment and binds whatever the registry attribute
    then holds. ``monkeypatch`` never saw that assignment and so would not undo
    it, which would leave a dead recorder bound for every later test in the
    session — including tests in other files. :meth:`restore` sweeps instead.
    """

    def __init__(self):
        self.calls: List[str] = []

    def __call__(self, path):
        self.calls.append(str(path))
        # The REAL pin: invariant 2's collision is the thing under test in
        # test 2, so it has to be able to happen.
        _REAL_SET_RUN_ANCHOR(path)
        raise _Pinned(str(path))

    @staticmethod
    def _bound_names():
        for module in list(sys.modules.values()):
            if module is None:
                continue
            try:
                bound = getattr(module, "set_run_anchor", None)
            except Exception:                    # noqa: BLE001 — lazy __getattr__
                continue
            if bound is not None:
                yield module, bound

    def sync(self) -> None:
        """Re-point every module that imported the name directly.

        Called again after each ``_load_target``: a module imported later picks
        the recorder up from the registry attribute by itself, but one that was
        already imported holds a direct reference that has to be rebound here.
        """
        registry.set_run_anchor = self
        for module, bound in self._bound_names():
            if module is registry or bound is self:
                continue
            if bound is _REAL_SET_RUN_ANCHOR or isinstance(bound, _PinRecorder):
                setattr(module, "set_run_anchor", self)

    def restore(self) -> None:
        registry.set_run_anchor = _REAL_SET_RUN_ANCHOR
        for module, bound in self._bound_names():
            if isinstance(bound, _PinRecorder):
                setattr(module, "set_run_anchor", _REAL_SET_RUN_ANCHOR)


@pytest.fixture
def pins():
    recorder = _PinRecorder()
    recorder.sync()
    try:
        yield recorder
    finally:
        recorder.restore()


def _drive(mode: mode_table.Mode, positionals: Sequence[str], pins: _PinRecorder):
    """Run *mode*'s real entry point on *positionals*, through the dispatcher's
    own normalization and its own async/sync call convention.

    Raises ``_Pinned`` when the pre-flight reaches its anchor pin.
    """
    cleanup: List[Path] = []
    try:
        argv = multi_llm._normalize_positionals(mode, list(positionals), cleanup)
        assert mode.target_module is not None
        target = multi_llm._load_target(mode.target_module, mode.target_attr)
        pins.sync()          # the target's module tree is loaded now
        return multi_llm._call_target(target, argv)
    finally:
        for path in cleanup:
            try:
                path.unlink()
            except OSError:                      # pragma: no cover
                pass


def _drive_flags(mode: mode_table.Mode, argv: Sequence[str], pins: _PinRecorder):
    """Like :func:`_drive`, but with an argv the caller built itself."""
    assert mode.target_module is not None
    target = multi_llm._load_target(mode.target_module, mode.target_attr)
    pins.sync()
    return multi_llm._call_target(target, list(argv))


def _write_plan(directory: Path, name: str = "plan.md") -> Path:
    """A plan file plus the task list the review-tasks pre-flight looks for.

    The tasks file is not what is under test — it is a precondition the
    review-tasks pre-flight checks *before* it reaches the pin, and without it
    that mode fails closed at exit 3 and never gets there.
    """
    directory.mkdir(parents=True, exist_ok=True)
    plan = directory / name
    plan.write_text(
        "# Plan\n\n## Overview\nOne plan, one anchor.\n", encoding="utf-8"
    )
    tasks_dir = directory / sanitize_prefix(plan.name) / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "tasks.md").write_text(
        "# Tasks\n\n## T001: Pin the anchor\n- Depends on: none\n", encoding="utf-8"
    )
    return plan


@pytest.fixture
def plan_file(tmp_path) -> Path:
    return _write_plan(tmp_path)


# ---------------------------------------------------------------------------
# 1. All six pre-flights pin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode", _pinning_modes(), ids=lambda m: m.name
)
def test_every_plan_mode_pins_the_run_anchor(mode, plan_file, pins, capsys):
    """Six call sites, nine modes, one assertion — from the mode table.

    The pin has to happen inside the pre-flight, after the plan file has been
    resolved and confirmed to exist and before anything that can read provider
    config to decide routing. Driving the real entry point and stopping at the
    pin asserts both halves at once: reaching ``_Pinned`` means the mode pinned,
    and reaching it *here* means nothing further down the run had to happen
    first.
    """
    assert registry.get_run_anchor() is None, "fixture leak: already pinned"

    positionals = [str(plan_file)]
    if mode_table.POS_QUESTION in mode.positionals:
        positionals.append("Does this pre-flight pin the run anchor?")

    with pytest.raises(_Pinned):
        _drive(mode, positionals, pins)

    assert len(pins.calls) == 1, (
        f"{mode.name} pinned {len(pins.calls)} times: {pins.calls}"
    )
    anchor = registry.get_run_anchor()
    assert anchor is not None
    assert os.path.realpath(anchor) == os.path.realpath(str(plan_file)), (
        f"{mode.name} pinned {anchor!r}, not the run's plan"
    )


@pytest.mark.parametrize(
    "composite",
    [m for m in mode_table.MODES
     if m.is_composite and mode_table.POS_PLAN in m.positionals],
    ids=lambda m: m.name,
)
def test_every_composite_pins_through_the_sub_mode_it_dispatches(
    composite, plan_file, pins
):
    """A composite's exemption, demonstrated rather than asserted in prose.

    A composite (``full``, ``improve-plan``, ``improve-code``) resolves
    ``--plan-file`` and owns no pre-flight pin of its own; it pins because the
    first phase it dispatches is one of the modes above, against that same
    plan. If that ever stopped being true this fails, and its entry in
    ``EXEMPT`` stops being free.
    """
    with pytest.raises(_Pinned):
        _drive(composite, [str(plan_file)], pins)

    assert registry.get_run_anchor() == os.path.realpath(str(plan_file))


@pytest.mark.parametrize("door", _front_doors(), ids=lambda m: m.name)
def test_every_front_door_pins_through_the_phase_it_fronts(
    door, tmp_path, pins, monkeypatch
):
    """A commit front door owns no pin and takes no plan: it resolves its
    range to a commit brief and calls the phase it fronts in-process with
    ``--plan-file <brief>`` (improve-commits through its composite, whose
    first step is review-code). Driven from the table: the brief path a user
    types (a root range's pinned spelling) goes through the dispatcher's own
    normalization, which makes it that ``--plan-file``, and the fronted
    phase's real pre-flight pins exactly that brief. The front door's own
    reads before that call (the commit-reviews folder) pass an explicit
    anchor, as ``status``'s runner line does.

    ``utils.commit_review.is_commit_brief`` is the brief rule's one owner and
    another task's module; a stand-in that accepts this file keeps the row
    about the pin.
    """
    import types

    import utils

    brief = tmp_path / "plans" / "commit-reviews" / "root-e4f5a6b7c8d9.md"
    _write_plan(brief.parent, brief.name)
    stand_in = types.ModuleType("utils.commit_review")
    stand_in.is_commit_brief = lambda path, *, cwd=None: (
        os.path.realpath(str(path)) == os.path.realpath(str(brief)))
    monkeypatch.setitem(sys.modules, "utils.commit_review", stand_in)
    monkeypatch.setattr(utils, "commit_review", stand_in, raising=False)

    argv = multi_llm._normalize_positionals(door, [str(brief)], [])
    assert argv == ["--plan-file", str(brief)]
    fronted = _fronted(door)
    assert fronted.name in {m.name for m in _pinning_modes()}
    with pytest.raises(_Pinned):
        _drive_flags(fronted, argv, pins)
    assert registry.get_run_anchor() == os.path.realpath(str(brief))


# ---------------------------------------------------------------------------
# 2. The symlink re-pin regression
# ---------------------------------------------------------------------------

@_requires_symlinks
def test_two_pre_flights_pin_once_across_two_spellings_of_one_plan(
    tmp_path, pins
):
    """One plan, two spellings, one interpreter: pin once, raise nothing.

    ``set_run_anchor`` compares strings, and the pre-flights normalize with
    three different functions (``os.path.realpath``, ``Path(...).resolve()``,
    ``os.path.abspath``). Before the normalization moved inside
    ``set_run_anchor`` the second pin here raised
    ``RuntimeError: … already pinned …``.
    """
    real_dir = tmp_path / "real"
    plan = _write_plan(real_dir)
    link_dir = tmp_path / "link"
    os.symlink(real_dir, link_dir)
    link_plan = link_dir / plan.name
    assert os.path.abspath(str(link_plan)) != os.path.realpath(str(link_plan))

    review_plan = mode_table.resolve("review-plan")
    review_code = mode_table.resolve("review-code")

    with pytest.raises(_Pinned):
        _drive(review_plan, [str(link_plan)], pins)     # the link spelling
    with pytest.raises(_Pinned):
        _drive(review_code, [str(plan)], pins)          # the real spelling

    assert pins.calls == [str(link_plan), str(plan)], (
        "both pre-flights must have been reached"
    )
    assert registry.get_run_anchor() == os.path.realpath(str(plan))


@_requires_symlinks
def test_a_full_runs_phases_do_not_collide_on_a_symlinked_plan(tmp_path, pins):
    """The measured defect, in the shape ``full`` actually produces.

    ``full`` passes ONE ``--plan-file`` to every phase it runs in one
    interpreter. When that path contains a symlinked component the phases still
    disagreed, because ``review-plan`` normalized it with ``os.path.abspath``
    (the link spelling) and ``review-code`` with ``Path(...).resolve()`` (the
    real one), so phase 5 re-pointed the anchor phase 1 had set and raised
    mid-run.
    """
    real_dir = tmp_path / "real"
    plan = _write_plan(real_dir)
    link_plan = tmp_path / "link" / plan.name
    os.symlink(real_dir, tmp_path / "link")

    with pytest.raises(_Pinned):
        _drive(mode_table.resolve("review-plan"), [str(link_plan)], pins)
    with pytest.raises(_Pinned):
        _drive(mode_table.resolve("review-code"), [str(link_plan)], pins)

    assert pins.calls[0] != pins.calls[1], (
        "one --plan-file, and the two pre-flights still hand over two "
        f"different strings — that disagreement is the defect: {pins.calls}"
    )
    assert [os.path.realpath(c) for c in pins.calls] == [
        os.path.realpath(str(link_plan))
    ] * 2, "…and both of them are that one file"
    assert registry.get_run_anchor() == os.path.realpath(str(link_plan))


# ---------------------------------------------------------------------------
# 3. `_setup`'s early exits still exit without pinning
# ---------------------------------------------------------------------------

#: ``(mode word, flag, reported status)`` — the two ``_setup`` early exits that
#: precede the pin, on modes whose parser actually offers the flag.
_EARLY_EXITS = [
    ("apply-suggestions", "--skip", "skipped"),
    ("apply-task-suggestions", "--skip", "skipped"),
    ("apply-task-suggestions", "--mark-completed", "completed"),
    ("apply-code-fixes", "--mark-completed", "completed"),
]


@pytest.mark.parametrize(
    "mode_word,flag,status",
    _EARLY_EXITS,
    ids=[f"{word}{flag}" for word, flag, _ in _EARLY_EXITS],
)
def test_setup_early_exits_exit_without_pinning(
    mode_word, flag, status, plan_file, pins, capsys
):
    """``--skip`` and ``--mark-completed`` record a phase word and stop.

    They run before step 3, so they must reach ``sys.exit(0)`` with **no**
    anchor pinned: a process that records a phase state and exits has started
    no run to anchor. Asserted directly so neither this pin nor R1's edit to
    that same early-exit block can be reordered past the other unnoticed.
    """
    import json

    mode = mode_table.resolve(mode_word)
    with pytest.raises(SystemExit) as exc:
        _drive_flags(mode, ["--plan-file", str(plan_file), flag], pins)

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == status, "the early exit must have run in full"
    assert pins.calls == [], f"{mode_word} {flag} pinned: {pins.calls}"
    assert registry.get_run_anchor() is None


@pytest.mark.parametrize(
    "mode_word", sorted({word for word, _, _ in _EARLY_EXITS})
)
def test_the_same_setup_pins_when_no_early_exit_fires(mode_word, plan_file, pins):
    """The positive control for the ordering above.

    Without the flag, the very same ``_setup`` falls through to step 3 and
    pins — so the assertion above is about *ordering*, not about an apply mode
    that never pins at all.
    """
    mode = mode_table.resolve(mode_word)
    with pytest.raises(_Pinned):
        _drive_flags(mode, ["--plan-file", str(plan_file)], pins)
    assert registry.get_run_anchor() == os.path.realpath(str(plan_file))
