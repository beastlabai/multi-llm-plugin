"""``--improve-commits``: the composite's rules, its executor leg and its native planner.

plans/review-commits.md §2.8 and §4 T7. improve-commits is improve-code's
sequence (``review-code`` then ``apply-code-fixes``) behind a commit front
door, which resolves the range, writes the commit brief that stands in for
the plan and hands ``improve_workflow.entry_point`` ``--plan-file <brief>
--base-ref <base> --head-ref <head>``. These tests pin what
``utils/improve_workflow.py``, ``utils/improve_steps.py`` and
``utils/report_check.py`` add for it:

* the composites, the scoped review and the step methods are read off the
  mode table, and every existing row's derived value is what it was;
* ``--head-ref`` is the improve-commits parser's only, reaches the review, and
  the apply always gets the pinned base (the empty-tree id on a root range);
* the continuation is ``--improve-commits <base12>..<head12> --resume`` plus
  the typed selectors, never a flag the front door owns;
* the row-keyed mapping: ``--interactive``'s refusal, ``[ALREADY_COMPLETE]``
  and the executor's step labels name ``review-commits`` and
  ``apply-commit-fixes``, while ``[IMPROVE_STEP]`` keeps the phase words;
* ``util improve-steps --mode improve-commits`` plans a commit brief from its
  ``commit_range`` (the pinned refs, ``(commit brief changed)``, the refusals).

The executor tests run the real front door and the real composite with each
phase's ``run_mode`` stubbed (``tests/test_improve_orchestrator.py``'s
``stub``), so no model runs. The end-to-end half, with the real phases and the
mock providers, is in ``tests/test_e2e_improve_modes.py``.
"""

from __future__ import annotations

import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, commit_review, commit_spelling, detach, git_utils  # noqa: E402
from utils import improve_steps, process_control  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils import report_check  # noqa: E402
from utils.commit_review import output_dir_for_brief  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

from tests.test_improve_orchestrator import (  # noqa: E402,F401
    APPLY_FLAGS,
    T,
    _argv_of,
    _live_run,
    stub,
)

IMPROVE = "improve-commits"
FOLDER = "plans/commit-reviews"
PLUGIN = "/multi-llm:multi-llm "

#: A command a commit mode prints never holds one of these (§2.8).
FRONT_DOOR_FLAGS = ("--plan-file", "--base-ref", "--head-ref", "--focus",
                    "--context-plan", "--commit-stage", "--revs")


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """No host or route declared, no recorded invocation left over."""
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    cli_common.clear_invocation()
    yield
    cli_common.clear_invocation()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(root), check=True,
                          capture_output=True, text=True).stdout


def _commit(root: Path, path: str, text: str, message: str) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD").strip()


def _init(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--quiet", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgsign", "false")
    return Path(os.path.realpath(root))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Three commits (c1 a.py, c2 b.py, c3 a.py); the cwd is the root."""
    root = _init(tmp_path / "repo")
    _commit(root, "a.py", "a = 1\n", "c1: start")
    _commit(root, "b.py", "b = 2\n", "c2: add b")
    _commit(root, "a.py", "a = 3\n", "c3: change a")
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def root_repo(tmp_path, monkeypatch):
    """One commit: ``HEAD`` is the root, so the default range is a root range."""
    root = _init(tmp_path / "one")
    _commit(root, "a.py", "a = 1\n", "only")
    monkeypatch.chdir(root)
    return root


def _sha(root: Path, rev: str) -> str:
    return _git(root, "rev-parse", rev).strip()


def _range(root: Path) -> Tuple[str, str]:
    """``last 2 commits``: ``(base, head)``."""
    return _sha(root, "HEAD~2"), _sha(root, "HEAD")


def _brief_of(root: Path, base: Optional[str], head: str) -> Path:
    return root / FOLDER / f"{'root' if base is None else base[:12]}-{head[:12]}.md"


def _state(brief: Path) -> dict:
    return json.loads((output_dir_for_brief(brief) / "state.json").read_text(encoding="utf-8"))


def _pinned(base: str, head: str) -> str:
    return f"{base[:12]}..{head[:12]}"


def _door(*argv: str) -> int:
    return commit_review.front_door(IMPROVE, list(argv))


def _round_started(brief: Path) -> None:
    """What review-code writes when a round starts (T3): the header it answers."""
    manager = StateManager(brief)
    record = manager.state["commit_range"]
    record["reviewed_header_sha256"] = record["header_sha256"]
    record.pop("guard", None)
    manager.save()


def _phases(*hooks):
    """A ``stub`` hook mimicking what the real phases record beyond completion:
    review-code's round start, and an apply that stops after its first batch
    (recorded started, then failed through ``stub(fail=...)``)."""
    def hook(workflow, mode_word, _extra):
        if mode_word == "review-code" and "round" in hooks:
            _round_started(workflow.plan_path)
        if mode_word == "apply-code-fixes" and "apply-started" in hooks:
            manager = StateManager(workflow.plan_path)
            manager.mark_phase_started("apply-code-fixes")
            manager.save()
    return hook


def _args(plan: Path, *argv: str):
    return iw.parse_args(IMPROVE, ["--plan-file", str(plan), *argv])


def _fresh(stub, **shape):
    """``stub(**shape)`` with an empty call log (``configure`` keeps one list)."""
    calls = stub(**shape)
    calls.clear()
    return calls


# ===========================================================================
# 1. The table: every composite, the scoped review, the step methods
# ===========================================================================


class TestTheTable:
    def test_the_composites_are_the_tables_review_then_apply_rows(self):
        assert [row.name for row in iw.COMPOSITE_ROWS] == [
            "improve-plan", "improve-code", IMPROVE]
        assert improve_steps.MODES == ("improve-plan", "improve-code", IMPROVE)
        assert all(not row.runs_workflow for row in iw.COMPOSITE_ROWS)

    def test_every_existing_rows_derived_values_are_unchanged(self):
        """§2.8 "improve-plan and improve-code stay byte-identical": the values
        the literals held before the table derivation, in the same order."""
        assert iw._SCOPED_REVIEW == "review-code"
        assert list(iw._STEP_METHODS.items()) == [
            ("review-plan", "step_review_plan"),
            ("apply-suggestions", "step_apply_suggestions"),
            ("review-code", "step_review_code"),
            ("apply-code-fixes", "step_apply_code_fixes"),
        ]
        assert improve_steps.MODES[:2] == ("improve-plan", "improve-code")
        assert improve_steps._SCOPED_MODE == "improve-code"
        assert not iw._reviews_code("improve-plan")
        assert iw._reviews_code("improve-code")

    def test_improve_commits_is_a_scoped_composite_as_improve_code_is(self):
        assert iw._reviews_code(IMPROVE)
        assert improve_steps._SCOPED_MODES == ("improve-code", IMPROVE)
        assert improve_steps._PINNED_MODES == (IMPROVE,)
        assert iw.step_indices(IMPROVE) == [("1", "review-code"), ("2", "apply-code-fixes")]
        assert iw.step_indices(IMPROVE) == iw.step_indices("improve-code")

    def test_every_step_method_is_a_workflow_method(self):
        for word, method in iw._STEP_METHODS.items():
            assert callable(getattr(iw.ImproveWorkflow, method, None)), (word, method)

    def test_only_improve_commits_pins_its_range(self):
        assert [row.name for row in iw.COMPOSITE_ROWS if iw.pins_commit_range(row.name)] == [
            IMPROVE]
        assert mode_table.resolve(IMPROVE).positionals == (mode_table.POS_REVS,)

    def test_no_step_flags_branch_names_a_mode(self):
        """§2.8: the apply's base comes through the typed-base branch on every
        composite, never through a branch on the mode's name."""
        source = inspect.getsource(iw.step_flags)
        body = source.split('"""', 2)[2]
        assert "improve-" not in body and "IMPROVE" not in body


# ===========================================================================
# 2. The parser
# ===========================================================================


def _own_options(parser) -> set:
    return {
        (tuple(a.option_strings), a.dest, a.nargs, repr(a.default), type(a).__name__)
        for a in parser._actions
        if a.option_strings and a.dest not in (
            "help", "plan_file", "mode", "json", "commit_stage")
    }


class TestTheParser:
    def test_improve_commits_takes_head_ref_and_the_hidden_stage(self, tmp_path):
        parser = iw.build_parser(IMPROVE)
        for flag in ("--head-ref", "--base-ref", "--files", "--files-from",
                     "--commit-stage", "--models", "--quick", "--resume", "--force"):
            assert flag in parser._option_string_actions, flag
        for flag in ("--focus", "--context-plan", "--rerun-all", "--reaggregate"):
            assert flag not in parser._option_string_actions, flag
        args = _args(tmp_path / "b.md", "--base-ref", "a" * 40, "--head-ref", "b" * 40)
        assert (args.base_ref, args.head_ref) == ("a" * 40, "b" * 40)

    def test_head_ref_on_improve_code_stays_a_usage_error(self, tmp_path, capsys):
        assert "--head-ref" not in iw.build_parser("improve-code")._option_string_actions
        with pytest.raises(SystemExit) as excinfo:
            iw.parse_args("improve-code", ["--plan-file", str(tmp_path / "p.md"),
                                           "--head-ref", "HEAD"])
        assert excinfo.value.code == 2
        assert "unrecognized arguments: --head-ref HEAD" in capsys.readouterr().err

    def test_the_planners_flags_are_the_composites_own(self):
        """As ``test_improve_steps.py`` holds for the other two: the namespace
        ``step_flags`` reads is the same on both legs (less the hidden stage)."""
        assert _own_options(improve_steps.build_parser(IMPROVE)) == _own_options(
            iw.build_parser(IMPROVE))
        assert "--head-ref" not in improve_steps.build_parser(
            "improve-code")._option_string_actions

    def test_its_help_names_the_commit_modes_and_a_range(self):
        text = iw.build_parser(IMPROVE).format_help()
        assert "Runs review-commits, then apply-commit-fixes" in text
        assert "improve-commits last 2 commits --quick" in text
        assert "plans/my-plan.md" not in text
        assert "--commit-stage" not in text
        code_help = iw.build_parser("improve-code").format_help()
        assert "Runs review-code, then apply-code-fixes, both unattended:" in code_help


# ===========================================================================
# 3. The rules, pure: Records built by hand
# ===========================================================================

B, H = "1" * 40, "2" * 40


class TestTheRules:
    def test_a_fresh_run_pins_both_refs_on_the_review_and_the_base_on_the_apply(self, tmp_path):
        args = _args(tmp_path / "b.md", "--models", "A", "--base-ref", B, "--head-ref", H)
        records = iw.Records(mode=IMPROVE)
        assert iw.step_flags(IMPROVE, "review-code", args, records) == [
            "--yes", "--models", "A", "--base-ref", B, "--head-ref", H]
        assert iw.step_flags(IMPROVE, "apply-code-fixes", args, records) == APPLY_FLAGS + [
            "--base-ref", B]

    def test_a_passed_over_review_resumes_the_apply_on_the_pinned_base(self, tmp_path):
        args = _args(tmp_path / "b.md", "--base-ref", B, "--head-ref", H)
        records = iw.Records(mode=IMPROVE, review_completed=True, review_on_record=True,
                             apply_started=True, apply_in_progress=True,
                             recorded_base_ref="0" * 40)
        assert iw.skip_reason(IMPROVE, "review-code", args, records) == "completed"
        assert iw.step_flags(IMPROVE, "apply-code-fixes", args, records) == APPLY_FLAGS + [
            "--resume", "--base-ref", B], "the front door's base, never the manifest's"

    def test_a_changed_commit_brief_is_a_new_round(self, tmp_path):
        args = _args(tmp_path / "b.md", "--base-ref", B, "--head-ref", H,
                     "--files", "a.py")
        records = iw.Records(mode=IMPROVE, review_completed=True, review_on_record=True,
                             apply_word="completed", apply_word_at=T(5),
                             requested_scope=("a.py",), recorded_scope=("b.py",),
                             header_changed=True)
        assert iw.new_round(args, records)
        assert not iw.scope_changed(IMPROVE, args, records)
        review = iw.step_flags(IMPROVE, "review-code", args, records)
        assert review == ["--yes", "--force", "--rerun-all", "--files", "a.py",
                          "--base-ref", B, "--head-ref", H]
        assert iw.skip_reason(IMPROVE, "apply-code-fixes", args, records) is None
        assert "--resume" not in iw.step_flags(IMPROVE, "apply-code-fixes", args, records)
        step = iw.plan_steps(IMPROVE, args, records)[0]
        assert (step.decision, step.reason, step.fresh) == ("run", "commit brief changed", True)
        # A typed --force with it is still one pair, and the reason is the typed flag.
        forced = _args(tmp_path / "b.md", "--force", "--base-ref", B, "--head-ref", H)
        assert iw.step_flags(IMPROVE, "review-code", forced, records).count("--force") == 1
        assert iw.plan_steps(IMPROVE, forced, records)[0].reason == "new round, --force"

    def test_the_continuation_carries_no_front_door_flag(self, tmp_path):
        listing = tmp_path / "list.txt"
        args = _args(tmp_path / "b.md", "--models", "A", "B", "--files", "a.py",
                     "--files-from", str(listing), "--base-ref", B, "--head-ref", H,
                     "--force")
        assert iw.continuation_flags(IMPROVE, args) == [
            "--models", "A", "B", "--files", "a.py", "--files-from", str(listing),
            "--resume"]
        stdin = _args(tmp_path / "b.md", "--quick", "--files-from", "-",
                      "--base-ref", B, "--head-ref", H)
        assert iw.continuation_flags(IMPROVE, stdin) == ["--quick", "--resume"]
        # improve-code's typed --base-ref is the user's, and stays.
        code = iw.parse_args("improve-code", ["--plan-file", "p.md", "--base-ref", "r"])
        assert iw.continuation_flags("improve-code", code) == ["--base-ref", "r", "--resume"]

    def test_a_run_records_flags_continue_without_force(self):
        assert iw.recorded_continuation(IMPROVE, [
            "--models", "A", "--detach", "--force", "--host", "codex", "--yes",
            "--exec-subagents"]) == ["--models", "A", "--resume"]
        assert iw.recorded_continuation(IMPROVE, None) == ["--resume"]


# ===========================================================================
# 4. The continuation as a command (commit_spelling)
# ===========================================================================


class TestTheContinuationCommand:
    def test_it_names_the_pinned_range_and_the_typed_selectors(self, repo, capsys):
        base, head = _range(repo)
        assert commit_review.commit_brief_main(["last", "2", "commits", "--focus", "X"]) == 0
        brief = _brief_of(repo, base, head)
        args = _args(brief, "--models", "A", "--base-ref", base, "--head-ref", head)
        assert iw.continuation_args(IMPROVE, brief, args) == (
            IMPROVE, [_pinned(base, head), "--models", "A", "--resume"])
        expected = cli_common.dispatcher_command(
            IMPROVE, _pinned(base, head), "--models", "A", "--resume")
        assert iw.continuation_command(IMPROVE, brief, args) == expected
        assert iw.ImproveWorkflow(args, IMPROVE).resume_command() == expected
        assert iw.ImproveWorkflow(args, IMPROVE).resume_flags() == ["--models", "A", "--resume"]
        # The generic hook's own composition gives the same command (T8b's site).
        assert commit_spelling.spell_command(
            IMPROVE, brief, *iw.continuation_flags(IMPROVE, args),
            view=commit_spelling.commit_view(brief)) == expected

    def test_a_root_range_names_the_brief(self, root_repo, capsys):
        head = _sha(root_repo, "HEAD")
        assert commit_review.commit_brief_main([]) == 0
        brief = _brief_of(root_repo, None, head)
        args = _args(brief, "--base-ref", git_utils.empty_tree_id(str(root_repo)),
                     "--head-ref", head)
        word, rest = iw.continuation_args(IMPROVE, brief, args, plan_arg="typed/brief.md")
        assert (word, rest) == (IMPROVE, ["typed/brief.md", "--resume"])

    def test_a_brief_with_no_commit_range_yet_stands_as_the_range(self, tmp_path):
        """A new brief whose stage no worker has applied: still a valid command."""
        brief = tmp_path / "plans" / "commit-reviews" / "x.md"
        args = _args(brief, "--base-ref", B, "--head-ref", H)
        assert iw.continuation_args(IMPROVE, brief, args) == (IMPROVE, [str(brief), "--resume"])

    def test_improve_code_on_a_plan_is_unchanged(self, tmp_path):
        plan = tmp_path / "p.md"
        args = iw.parse_args("improve-code", ["--plan-file", str(plan), "--base-ref", "r"])
        assert iw.continuation_command("improve-code", plan, args) == (
            cli_common.dispatcher_command("improve-code", plan, "--base-ref", "r", "--resume"))


# ===========================================================================
# 5. The row-keyed mapping: the lines a user reads name the commit modes
# ===========================================================================


class TestTheCommitWords:
    def test_the_interactive_refusal(self):
        assert iw.interactive_refusal(IMPROVE) == (
            "ERROR: improve-commits runs unattended and does not take --interactive (-i). "
            "Run --review-commits --interactive, then --apply-commit-fixes.")
        assert iw.interactive_refusal("improve-code").endswith(
            "Run --review-code --interactive, then --apply-code-fixes.")
        assert iw.interactive_refusal("improve-plan").endswith(
            "Run --review-plan --interactive, then --apply-suggestions.")

    def test_already_complete(self, tmp_path):
        args = _args(tmp_path / "b.md", "--base-ref", B, "--head-ref", H)
        records = iw.Records(mode=IMPROVE, review_completed=True, review_on_record=True,
                             review_completed_at=T(0), apply_word="completed",
                             apply_word_at=T(5))
        assert iw.nothing_to_do(IMPROVE, args, records)
        assert iw.already_complete_line(IMPROVE, args, records) == (
            f"[ALREADY_COMPLETE] improve-commits: review-commits completed {T(0)}, "
            f"apply-commit-fixes completed {T(5)}; nothing to do. "
            "--force runs review-commits again.")

    def test_the_executor_labels_its_steps_with_them(self, tmp_path):
        plan = tmp_path / "b.md"
        plan.write_text("# brief\n", encoding="utf-8")
        steps = iw.ImproveWorkflow(_args(plan), IMPROVE).steps()
        assert [(s.phase, s.label) for s in steps] == [
            ("code-review", "Step 1: review-commits"),
            ("apply-code-fixes", "Step 2: apply-commit-fixes")]
        code = iw.ImproveWorkflow(
            iw.parse_args("improve-code", ["--plan-file", str(plan)]), "improve-code").steps()
        assert [s.label for s in code] == ["Step 1: review-code", "Step 2: apply-code-fixes"]

    def test_the_planners_step_lines_keep_the_phase_words(self, tmp_path):
        """``[IMPROVE_STEP]``'s word is the phase core the native core loads."""
        args = _args(tmp_path / "b.md", "--base-ref", B, "--head-ref", H)
        assert [s.step for s in iw.plan_steps(IMPROVE, args, iw.Records(mode=IMPROVE))] == [
            "review-code", "apply-code-fixes"]

    def test_the_front_door_refuses_interactive_with_them(self, repo, capsys):
        assert _door("--revs", "last 2 commits", "--interactive") == 3
        captured = capsys.readouterr()
        assert captured.err.strip() == iw.interactive_refusal(IMPROVE)
        assert not (repo / FOLDER).exists()


# ===========================================================================
# 6. The executor leg: the front door, the composite, the phases stubbed
# ===========================================================================


class TestTheExecutorLeg:
    def test_a_same_invocation_review_hands_the_apply_the_pinned_base(
        self, repo, stub, capsys
    ):
        calls = stub(hook=_phases("round"))
        base, head = _range(repo)
        assert _door("--revs", "last 2 commits", "--models", "A") == 0
        assert [word for word, _argv, _held in calls] == ["review-code", "apply-code-fixes"]
        assert _argv_of(calls, "review-code") == [
            "--yes", "--models", "A", "--base-ref", base, "--head-ref", head]
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS + ["--base-ref", base]
        out = capsys.readouterr().out
        assert "=== Step 1: review-commits ===" in out
        assert "=== Step 2: apply-commit-fixes ===" in out
        brief = _brief_of(repo, base, head)
        state = _state(brief)
        assert set(state["phases_completed"]) == {"code-review", "apply-code-fixes"}
        assert state["improve_commits"]["failed_phase"] is None
        assert not (output_dir_for_brief(brief) / "improve-code").exists()

    def test_the_front_doors_review_lock_is_held_through_the_apply(self, repo, stub, capsys):
        """§2.1: the front door holds the code-review lock through the
        composite. Its preflight finds no other owner, the review step runs
        under that lock instead of taking it again, and the apply step reads
        the round with it still held, so no other invocation can rewrite or
        archive that round meanwhile. Released when the door returns."""
        base, head = _range(repo)
        review_lock = detach.lock_path(
            str(output_dir_for_brief(_brief_of(repo, base, head)) / "code-review"))
        seen = {}

        def hook(workflow, mode_word, _extra):
            holder = process_control.read_lock_holder(review_lock) or {}
            seen[mode_word] = (process_control.lock_is_free(review_lock), holder.get("holder"))
            if mode_word == "review-code":
                _round_started(workflow.plan_path)

        calls = stub(hook=hook)
        assert _door("--revs", "last 2 commits", "--models", "A") == 0
        assert [word for word, _argv, _held in calls] == ["review-code", "apply-code-fixes"]
        door = f"--{IMPROVE} front door"
        assert seen == {"review-code": (False, door), "apply-code-fixes": (False, door)}
        assert process_control.lock_is_free(review_lock)
        assert iw._HELD_LOCKS == {}

    def test_a_root_range_hands_both_steps_the_empty_tree(self, root_repo, stub, capsys):
        calls = stub(hook=_phases("round"))
        head = _sha(root_repo, "HEAD")
        empty = git_utils.empty_tree_id(str(root_repo))
        assert _door() == 0
        assert _argv_of(calls, "review-code") == [
            "--yes", "--base-ref", empty, "--head-ref", head]
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS + ["--base-ref", empty]

    def test_a_stop_after_the_first_batch_then_the_continuation(self, repo, stub, capsys):
        """§2.8 "When a step fails": exit 3 naming apply-commit-fixes; the
        continuation calls no review, resumes the apply on the pinned base, and
        leaves the brief, its identity and the focus as they were."""
        calls = stub(fail={"apply-code-fixes": 3}, hook=_phases("round", "apply-started"))
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        ctx = cli_common.activate_result(cli_common.ResultContext(mode=IMPROVE))
        try:
            assert _door("--revs", "last 2 commits", "--focus", "X", "--models", "A") == 3
        finally:
            cli_common.deactivate_result()
        err = capsys.readouterr().err
        assert ("ERROR: Step 2: apply-commit-fixes did not complete (exited 3). The "
                "workflow stopped here; no later phase ran.") in err
        assert ctx.error.startswith("Step 2: apply-commit-fixes did not complete")
        assert ctx.state == "needs-human"
        # The command the stop names carries none of the front door's flags.
        stop = shlex.split(ctx.next)
        assert stop[stop.index(IMPROVE) + 2:] == ["--models", "A", "--resume"]
        for flag in FRONT_DOOR_FLAGS + ("last", "X"):
            assert flag not in stop, (flag, stop)
        assert _state(brief)["improve_commits"]["failed_phase"] == "apply-code-fixes"

        before = brief.read_bytes(), _state(brief)["commit_range"]
        calls = _fresh(stub, hook=_phases("round"))
        assert _door("--revs", _pinned(base, head), "--models", "A", "--resume") == 0
        assert [word for word, _argv, _held in calls] == ["apply-code-fixes"], (
            "the completed review was run again")
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS + [
            "--resume", "--base-ref", base]
        after = brief.read_bytes(), _state(brief)["commit_range"]
        assert after[0] == before[0]
        for key in ("header_sha256", "reviewed_header_sha256", "focus", "base", "head"):
            assert after[1][key] == before[1][key], key
        assert after[1]["focus"] == "X"

        # A second re-run: nothing to do, in the commit modes' words.
        calls = _fresh(stub)
        assert _door("--revs", _pinned(base, head), "--models", "A", "--resume") == 2
        out = capsys.readouterr().out
        assert "[ALREADY_COMPLETE] improve-commits: review-commits completed" in out
        assert ", apply-commit-fixes completed " in out
        assert "--force runs review-commits again." in out
        assert calls == []

    def test_the_stop_names_the_pinned_range(self, repo, stub, capsys):
        stub(fail={"apply-code-fixes": 3}, hook=_phases("round", "apply-started"))
        base, head = _range(repo)
        expected = cli_common.dispatcher_command(
            IMPROVE, _pinned(base, head), "--models", "A", "--resume")
        ctx = cli_common.activate_result(cli_common.ResultContext(mode=IMPROVE))
        try:
            assert _door("--revs", "last 2 commits", "--focus", "X", "--models", "A") == 3
        finally:
            cli_common.deactivate_result()
        assert expected in [line.strip() for line in capsys.readouterr().err.splitlines()]
        assert ctx.next == expected

    def test_a_typed_focus_after_the_stop_runs_the_review_again(self, repo, stub, capsys):
        stub(fail={"apply-code-fixes": 3}, hook=_phases("round", "apply-started"))
        base, head = _range(repo)
        assert _door("--revs", "last 2 commits", "--focus", "X") == 3
        calls = _fresh(stub, hook=_phases("round"))
        assert _door("--revs", _pinned(base, head), "--focus", "Y") == 0
        review = _argv_of(calls, "review-code")
        assert review.count("--force") == 1 and review.count("--rerun-all") == 1, review
        assert review[-4:] == ["--base-ref", base, "--head-ref", head]
        apply = _argv_of(calls, "apply-code-fixes")
        assert "--resume" not in apply, "a new round starts a new apply cycle"
        assert apply == APPLY_FLAGS + ["--base-ref", base]
        assert _state(_brief_of(repo, base, head))["commit_range"]["focus"] == "Y"

    def test_a_restart_cut_off_before_its_round_still_restarts(self, repo, stub, capsys):
        """The header was rewritten (focus Y) but the review never started its
        round: the next plain re-run reviews again rather than pass over the
        old round's completion (``Records.header_changed``)."""
        stub(hook=_phases("round"))
        base, head = _range(repo)
        assert _door("--revs", "last 2 commits", "--focus", "X") == 0
        stub(fail={"review-code": 1})          # the restarted review dies at once
        assert _door("--revs", _pinned(base, head), "--focus", "Y") == 3
        brief = _brief_of(repo, base, head)
        record = _state(brief)["commit_range"]
        assert record["header_sha256"] != record["reviewed_header_sha256"]
        calls = _fresh(stub, hook=_phases("round"))
        assert _door("--revs", _pinned(base, head)) == 0
        review = _argv_of(calls, "review-code")
        assert review[:3] == ["--yes", "--force", "--rerun-all"], review

    def test_detach_crosses_the_boundary_as_improve_commits(self, repo, monkeypatch, capsys):
        """The recorded-mode check passes (the front door recorded the
        composite itself) and the run claims ``<slug>/improve-commits/``."""
        seen = {}

        def handle(**kwargs):
            seen.update(kwargs)
            seen["recorded"] = dict(cli_common.invocation())
            return 0

        monkeypatch.setattr(detach, "handle_phase_detach", handle)
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert not brief.exists()
        assert _door("--revs", "last 2 commits", "--models", "A", "--detach") == 0
        assert seen["mode"] == IMPROVE and seen["phase"] == IMPROVE
        assert seen["phase_dir"] == str(output_dir_for_brief(brief) / IMPROVE)
        assert seen["recorded"]["mode"] == IMPROVE
        recorded = seen["recorded"]["argv"]
        assert recorded[:2] == ["--models", "A"]
        assert recorded[recorded.index("--revs") + 1] == _pinned(base, head)
        for flag in ("--plan-file", "--base-ref", "--head-ref", "--focus"):
            assert flag not in recorded, (flag, recorded)
        assert "--commit-stage" in recorded


# ===========================================================================
# 7. The native leg: util commit-brief, then util improve-steps
# ===========================================================================


def _plan_steps(capsys, brief, *argv) -> Tuple[int, str, str]:
    target = brief if isinstance(brief, str) else os.path.relpath(brief)
    code = improve_steps.main([target, "--mode", IMPROVE, *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _flags(out: str, index: str) -> List[str]:
    prefix = f"{improve_steps.FLAGS_MARKER} {index} "
    [line] = [line for line in out.splitlines() if line.startswith(prefix)]
    return shlex.split(line)[2:]


def _line(out: str, marker: str) -> str:
    """The one line starting with *marker*, without it (``""`` for a bare marker)."""
    [line] = [line for line in out.splitlines()
              if line == marker or line.startswith(marker + " ")]
    return line[len(marker) + 1:]


def _remaining(out: str) -> List[str]:
    return shlex.split(_line(out, commit_review.COMMIT_REMAINING_FLAGS))


class TestTheNativeLeg:
    def test_commit_brief_then_improve_steps_plans_the_pinned_refs(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert not brief.exists()
        assert commit_review.commit_brief_main(["last", "2", "commits", "--quick"]) == 0
        out = capsys.readouterr().out
        assert "[COMMIT_RANGE] " in out and "[COMMIT_RANGE_END]" in out
        assert _line(out, commit_review.COMMIT_BRIEF) == f"{FOLDER}/{brief.name}"
        assert _line(out, commit_review.COMMIT_RANGE_PINNED) == f"{base}..{head}"
        assert _remaining(out) == ["--quick"]
        assert brief.is_file() and _state(brief)["commit_range"]["head"] == head

        code, out, err = _plan_steps(capsys, f"{FOLDER}/{brief.name}", *_remaining(
            "[COMMIT_REMAINING_FLAGS] --quick"))
        assert code == 0, err
        assert "[IMPROVE_STEP] 1 review-code run (not reviewed yet)" in out
        assert "[IMPROVE_STEP] 2 apply-code-fixes run (review not applied yet)" in out
        assert improve_steps.FRESH_MARKER not in out
        assert _flags(out, "1") == ["--yes", "--quick", "--base-ref", base, "--head-ref", head]
        assert _flags(out, "2") == APPLY_FLAGS + ["--base-ref", base]
        assert _line(out, improve_steps.RERUN_MARKER) == (
            f"{PLUGIN}--improve-commits {_pinned(base, head)} --quick --resume")

    def test_another_focus_after_a_round_makes_step_1_fresh(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert commit_review.commit_brief_main(["last", "2", "commits", "--focus", "X"]) == 0
        _round_started(brief)
        manager = StateManager(brief)
        manager.mark_phase_completed("code-review")
        manager.save()
        capsys.readouterr()
        code, out, err = _plan_steps(capsys, brief)
        assert code == 0, err
        assert "[IMPROVE_STEP] 1 review-code skip (completed " in out

        assert commit_review.commit_brief_main(["last", "2", "commits", "--focus", "Y"]) == 0
        out = capsys.readouterr().out
        assert "[COMMIT_REVIEW_RESTARTED]" in out
        assert _remaining(out) == []
        code, out, err = _plan_steps(capsys, brief)
        assert code == 0, err
        assert "[IMPROVE_STEP] 1 review-code run (commit brief changed)" in out
        assert "[IMPROVE_STEP_FRESH] 1" in out
        assert _flags(out, "1") == ["--yes", "--force", "--rerun-all", "--base-ref", base,
                                    "--head-ref", head]
        assert _flags(out, "2") == APPLY_FLAGS + ["--base-ref", base]
        assert _line(out, improve_steps.RERUN_MARKER) == (
            f"{PLUGIN}--improve-commits {_pinned(base, head)} --resume"), (
            "the continuation carries no --focus: the brief inherits it")

    def test_a_passed_over_review_resumes_the_apply_on_the_pinned_base(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert commit_review.commit_brief_main(["last", "2", "commits"]) == 0
        _round_started(brief)
        manager = StateManager(brief)
        manager.mark_phase_completed("code-review")
        manager.save()
        manager = StateManager(brief)
        manager.mark_phase_started("apply-code-fixes")
        manager.save()
        capsys.readouterr()
        code, out, err = _plan_steps(capsys, brief)
        assert code == 0, err
        assert "[IMPROVE_STEP] 2 apply-code-fixes run (apply in progress for this review)" in out
        assert _flags(out, "2") == APPLY_FLAGS + ["--resume", "--base-ref", base]

    def test_the_brief_bytes_equal_the_executor_front_doors(self, repo, monkeypatch, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert commit_review.commit_brief_main(["last", "2", "commits", "--focus", "X"]) == 0
        native, native_record = brief.read_bytes(), _state(brief)["commit_range"]
        shutil.rmtree(repo / FOLDER)
        monkeypatch.setattr(iw, "entry_point", lambda mode, argv: 0)
        assert _door("--revs", "last 2 commits", "--focus", "X") == 0
        assert brief.read_bytes() == native
        executor_record = _state(brief)["commit_range"]
        for key in ("base", "head", "focus", "header_sha256", "phrase", "brief"):
            assert executor_record[key] == native_record[key], key

    def test_another_focus_while_a_review_is_live_changes_nothing(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert commit_review.commit_brief_main(["last", "2", "commits", "--focus", "X"]) == 0
        state_file = output_dir_for_brief(brief) / "state.json"
        before = brief.read_bytes(), state_file.read_bytes()
        lock = _live_run(output_dir_for_brief(brief) / "code-review", "run-1", "review-code")
        try:
            capsys.readouterr()
            assert commit_review.commit_brief_main(
                ["last", "2", "commits", "--focus", "Y"]) == 3
            refused = capsys.readouterr()
            code, out, err = _plan_steps(capsys, brief)
        finally:
            lock.release()
        assert (brief.read_bytes(), state_file.read_bytes()) == before
        assert "a live run owns" in refused.err
        assert code == 3 and out == ""
        assert "a live run owns" in err

    def test_a_root_range_plans_the_empty_tree_and_continues_on_the_brief(
        self, root_repo, capsys
    ):
        head = _sha(root_repo, "HEAD")
        brief = _brief_of(root_repo, None, head)
        assert commit_review.commit_brief_main([]) == 0
        capsys.readouterr()
        typed = f"{FOLDER}/{brief.name}"
        code, out, err = _plan_steps(capsys, typed, "--models", "A")
        assert code == 0, err
        empty = git_utils.empty_tree_id(str(root_repo))
        assert _flags(out, "1") == ["--yes", "--models", "A", "--base-ref", empty,
                                    "--head-ref", head]
        assert _flags(out, "2") == APPLY_FLAGS + ["--base-ref", empty]
        assert _line(out, improve_steps.RERUN_MARKER) == (
            f"{PLUGIN}--improve-commits {typed} --models A --resume")

    def test_a_finished_brief_is_already_complete_in_commit_words(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert commit_review.commit_brief_main(["last", "2", "commits"]) == 0
        _round_started(brief)
        manager = StateManager(brief)
        manager.mark_phase_completed("code-review")
        manager.save()
        manager = StateManager(brief)
        manager.mark_phase_completed("apply-code-fixes")
        manager.save()
        capsys.readouterr()
        code, out, err = _plan_steps(capsys, brief)
        assert code == 2, err
        line = _line(out, improve_steps.ALREADY_COMPLETE_MARKER)
        assert line.startswith("improve-commits: review-commits completed ")
        assert ", apply-commit-fixes completed " in line
        assert line.endswith("--force runs review-commits again.")
        assert improve_steps.RERUN_MARKER not in out

    def test_json_carries_the_same_refs_and_the_pinned_rerun(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert commit_review.commit_brief_main(["last", "2", "commits"]) == 0
        capsys.readouterr()
        code, out, err = _plan_steps(capsys, brief, "--json")
        assert code == 0, err
        document = json.loads(out)
        assert document["mode"] == IMPROVE
        assert [step["step"] for step in document["steps"]] == ["review-code", "apply-code-fixes"]
        assert document["steps"][0]["argv"] == ["--yes", "--base-ref", base, "--head-ref", head]
        assert document["steps"][1]["argv"] == APPLY_FLAGS + ["--base-ref", base]
        assert document["rerun"].endswith(f"--improve-commits {_pinned(base, head)} --resume")
        assert document["rerun_argv"] == ["--resume"]

    def test_through_the_dispatcher(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert multi_llm.main(["util", "commit-brief", "last", "2", "commits"]) == 0
        capsys.readouterr()
        assert multi_llm.main(["util", "improve-steps", f"{FOLDER}/{brief.name}",
                               "--mode", IMPROVE]) == 0
        out = capsys.readouterr().out
        assert "[IMPROVE_STEP_FLAGS] 1 --yes --base-ref " in out
        assert json.loads(out.rsplit(cli_common.RESULT_MARKER, 1)[1])["exit"] == 0

    def test_its_interactive_refusal_names_the_commit_modes(self, repo, capsys):
        assert commit_review.commit_brief_main(["last", "2", "commits"]) == 0
        base, head = _range(repo)
        capsys.readouterr()
        code, out, err = _plan_steps(capsys, _brief_of(repo, base, head), "--interactive")
        assert (code, out) == (3, "")
        assert err.strip() == iw.interactive_refusal(IMPROVE)


class TestTheNativeLegsRefusals:
    def test_a_brief_with_no_commit_range_is_exit_3(self, repo, capsys):
        brief = repo / FOLDER / "a1b2c3d4e5f6-e4f5a6b7c8d9.md"
        brief.parent.mkdir(parents=True)
        brief.write_text(f"{commit_review.BRIEF_MARKER}\n# brief\n", encoding="utf-8")
        code, out, err = _plan_steps(capsys, brief)
        assert (code, out) == (3, "")
        assert err.startswith(f"{improve_steps.COMMIT_REVIEW_NOT_FOUND} {FOLDER}/{brief.name}: "
                              "no commit range recorded")
        assert not output_dir_for_brief(brief).exists(), "the planner wrote nothing"

    def test_recorded_shas_that_no_longer_resolve_are_exit_3(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert commit_review.commit_brief_main(["last", "2", "commits"]) == 0
        manager = StateManager(brief)
        manager.state["commit_range"]["head"] = "f" * 40
        manager.save()
        capsys.readouterr()
        code, out, err = _plan_steps(capsys, brief)
        assert (code, out) == (3, "")
        assert err.startswith(f"[RANGE_UNRESOLVED] head {'f' * 12} recorded in ")
        assert "no longer names a commit" in err

    @pytest.mark.parametrize("flag", ["--base-ref", "--head-ref"])
    def test_a_typed_ref_is_refused(self, repo, capsys, flag):
        base, head = _range(repo)
        assert commit_review.commit_brief_main(["last", "2", "commits"]) == 0
        capsys.readouterr()
        code, out, err = _plan_steps(capsys, _brief_of(repo, base, head), flag, "HEAD")
        assert (code, out) == (3, "")
        assert err.startswith(f"{improve_steps.COMMIT_FLAG_CONFLICT} {flag}: ")

    def test_an_unreadable_state_is_the_executors_refusal(self, repo, capsys):
        base, head = _range(repo)
        brief = _brief_of(repo, base, head)
        assert commit_review.commit_brief_main(["last", "2", "commits"]) == 0
        state_file = output_dir_for_brief(brief) / "state.json"
        state_file.write_text("{not json", encoding="utf-8")
        capsys.readouterr()
        code, out, err = _plan_steps(capsys, brief)
        assert (code, out) == (3, "")
        assert err.startswith("ERROR: improve-commits: cannot read ")
        assert state_file.read_text(encoding="utf-8") == "{not json"

    def test_a_missing_brief_is_exit_1_naming_the_commit_brief(self, repo, capsys):
        code, out, err = _plan_steps(capsys, f"{FOLDER}/nope.md")
        assert (code, out) == (1, "")
        assert err.startswith(f"ERROR: Commit brief not found: {FOLDER}/nope.md")


# ===========================================================================
# 8. report_check's composite wording
# ===========================================================================


def test_report_check_names_improve_commits_in_its_composite_wording():
    assert "--improve-commits" in report_check.__doc__
    command = f"{PLUGIN}--improve-commits a1b2c3d4e5f6..e4f5a6b7c8d9 --resume"
    block = report_check.Block(False, command, continuation=True)
    reason = report_check._continuation_reason(block)
    assert "this --improve-plan, --improve-code or --improve-commits run's next step" in reason
    assert f"```\n{command}\n```" in reason
