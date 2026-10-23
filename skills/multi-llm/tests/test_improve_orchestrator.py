"""``--improve-plan`` and ``--improve-code``: the executor leg and the shared rules.

The two composites run a review, then the apply of its findings, unattended.
Every rule both legs follow is a function in ``utils/improve_workflow.py``
(:func:`skip_reason`, :func:`step_flags`, :func:`continuation_flags`,
:func:`read_records`), and the executor leg is ``ImproveWorkflow``,
``full``'s state machine with the composite's identity.

Most tests stub ``FullWorkflow.run_mode`` as ``tests/test_full_workflow_
orchestrator.py`` does: the stub records the argv each phase receives and the
completion a real phase would record. ``ImproveWorkflow.run_mode`` wraps the
stub, so the step lock it holds is exercised for real. Asserting on the argv
``run_mode`` receives (never on ``step_flags`` alone) is what catches a step
that builds its own argv.

The native planner (``util improve-steps``) is the next agent's; its parity
tests build on :func:`improve_workflow.plan_steps`, which these tests pin for a
few states.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
import full_workflow_orchestrator as fwo  # noqa: E402
import improve_code_orchestrator as ico  # noqa: E402
import improve_plan_orchestrator as ipo  # noqa: E402
import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, detach, process_control, status_file  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils import state_manager as sm  # noqa: E402
from utils import write_batch_scheduler as wbs  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

#: A `--files-from` listing whose name carries both quote kinds. Windows forbids
#: `"` in a filename, so there it keeps only the `'`; the `--files` names,
#: which never touch the disk, still carry both.
LISTING_NAME = "list 'x' \"y\".txt" if os.name != "nt" else "list 'x' y.txt"


PLAN = "improve-plan"
CODE = "improve-code"
MODES = [PLAN, CODE]
ENTRY = {PLAN: ipo, CODE: ico}

#: The review and apply phase word of each composite, and their dirs.
REVIEW_PHASE = {PLAN: "review-plan", CODE: "code-review"}
APPLY_PHASE = {PLAN: "apply-suggestions", CODE: "apply-code-fixes"}
APPLY_DIR = {PLAN: "apply-suggestions", CODE: "apply-fixes"}

APPLY_FLAGS = ["--yes", "--agent-decide", "--no-confirm"]
TAIL = ["--host", "codex", "--exec-subagents", "--dry-run-subagents"]

BASE = datetime(2026, 10, 1, 12, 0, 0)


def T(minutes: float) -> str:
    """A naive local stamp *minutes* after a fixed base."""
    return (BASE + timedelta(minutes=minutes)).isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository the plan lives in, and the cwd: review-code's scope
    reader resolves ``--files`` against the plan's repository root."""
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _plan(root: Path, name: str = "imp-plan") -> Path:
    plan = root / "plans" / f"{name}.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Improve\n\n## Step 1\nDo the thing.\n", encoding="utf-8")
    return plan


def _out(plan: Path) -> Path:
    return output_dir_path(plan)


def _write_state(plan: Path, *, completed=None, skipped=None, started=None,
                 extra=None) -> Path:
    """``state.json`` with exactly these records (stamps as given)."""
    state = StateManager(plan)
    state.state["phases_completed"] = dict(completed or {})
    state.state["phases_skipped"] = {
        phase: ({"skipped_at": value, "reason": "test"}
                if isinstance(value, str) or value is None else value)
        for phase, value in (skipped or {}).items()
    }
    if started:
        state.state["phases_started"] = dict(started)
    for key, value in (extra or {}).items():
        state.state[key] = value
    state.save()
    return state.state_file


def _args(mode: str, plan: Path, *argv: str) -> argparse.Namespace:
    return iw.parse_args(mode, ["--plan-file", str(plan), *argv])


def _records(mode: str, plan: Path, *argv: str) -> iw.Records:
    return iw.read_records(mode, plan, _args(mode, plan, *argv))


def _write_manifest(plan: Path, base_ref: str) -> Path:
    """A ``code-review/review-scope.md`` header ``read_review_scope`` accepts."""
    phase_dir = _out(plan) / "code-review"
    phase_dir.mkdir(parents=True, exist_ok=True)
    path = phase_dir / "review-scope.md"
    path.write_text(
        "# Review scope\n\n"
        "- Scope id: `0123abcd`\n"
        f"- Base ref: `{base_ref}` (commit `abc1234`)\n"
        "- Written by: review-code\n\n"
        "Diff coverage: complete\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def stub(monkeypatch):
    """Stub every phase ``run_mode`` dispatches; return ``configure``.

    ``configure(fail={mode: code}, silent={mode})`` shapes it as
    ``test_full_workflow_orchestrator``'s ``stub_phases``. Each call is logged
    as ``(mode word, argv, the step lock's holder while it ran)``; a consolidation
    pass (``--consolidate``) records nothing.
    """
    calls = []
    shape = {"fail": {}, "silent": set(), "hook": None}

    def run_mode(self, mode_word, extra):
        phase = mode_table.phase_for(mode_word) or mode_word
        lock = detach.lock_path(str(self.plan_dir / {
            "apply-code-fixes": "apply-fixes"}.get(phase, phase)))
        held = None
        if not process_control.lock_is_free(lock):
            held = process_control.read_lock_holder(lock)
        calls.append((mode_word, list(extra), held))
        if shape["hook"] is not None:
            shape["hook"](self, mode_word, list(extra))
        if mode_word in shape["fail"]:
            return shape["fail"][mode_word]
        if mode_word in shape["silent"] or "--consolidate" in extra:
            return 0
        manager = StateManager(self.plan_path)
        manager.mark_phase_completed(phase)
        manager.save()
        return 0

    monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)

    def configure(fail=None, silent=None, hook=None):
        shape["fail"] = dict(fail or {})
        shape["silent"] = set(silent or ())
        shape["hook"] = hook
        return calls

    configure()
    return configure


def _argv_of(calls, mode_word):
    found = [argv for word, argv, _held in calls if word == mode_word
             and "--consolidate" not in argv]
    assert len(found) == 1, (mode_word, calls)
    return found[0]


def _main(mode: str, plan: Path, *argv: str) -> int:
    return ENTRY[mode].main(["--plan-file", str(plan), *argv])


@pytest.fixture
def no_boundary(monkeypatch):
    """Fail the test if the composite reaches its detach boundary."""
    def refuse(**kwargs):
        pytest.fail(f"maybe_detach was reached: {kwargs}")

    monkeypatch.setattr(rob, "maybe_detach", refuse)


# ---------------------------------------------------------------------------
# 1. The rows and the resolvers
# ---------------------------------------------------------------------------

class TestTheRows:
    def test_the_two_rows_are_the_plans_literals(self):
        plan = mode_table.resolve(PLAN)
        assert plan.summary == (
            "Review the plan with every model, then apply the validated "
            "suggestions to it (unattended)."
        )
        assert (plan.target_module, plan.phase, plan.instruction_file) == (
            "improve_plan_orchestrator", "improve-plan", "improve-plan.md",
        )
        assert plan.sequence == ("review-plan", "apply-suggestions")
        code = mode_table.resolve(CODE)
        assert code.summary == (
            "Review the code changes with every model, then apply the "
            "validated fixes (unattended)."
        )
        assert (code.target_module, code.phase, code.instruction_file) == (
            "improve_code_orchestrator", "improve-code", "improve-code.md",
        )
        assert code.sequence == ("review-code", "apply-code-fixes")
        for row in (plan, code):
            assert row.supports_detach and row.is_composite and row.user_facing
            assert not row.is_workflow_phase and not row.runs_workflow

    def test_each_sits_right_after_the_apply_phase_it_ends_on(self):
        names = list(mode_table.mode_names())
        assert names.index(PLAN) == names.index("apply-suggestions") + 1
        assert names.index(CODE) == names.index("apply-code-fixes") + 1
        assert PLAN not in mode_table.PREREQUISITE_PHASES
        assert CODE not in mode_table.PREREQUISITE_PHASES

    def test_one_orchestrator_script_per_mode(self):
        assert mode_table.mode_for_script("improve_plan_orchestrator.py") == PLAN
        assert mode_table.mode_for_script("improve_code_orchestrator.py") == CODE

    def test_improve_codes_sequence_resolves_to_phase_and_dir_words(self):
        code = mode_table.resolve(CODE)
        assert code.sequence_phases == ("code-review", "apply-code-fixes")
        assert code.last_phase == "apply-code-fixes"
        assert detach._driven_dir_names(CODE) == ("code-review", "apply-fixes")
        assert detach._driven_dir_names(PLAN) == ("review-plan", "apply-suggestions")

    @pytest.mark.parametrize("mode", MODES)
    def test_the_review_pairing_comes_from_the_scheduler_map(self, mode):
        row = mode_table.resolve(mode)
        assert iw.review_phase_for(row.sequence[-1]) == mode_table.resolve(
            row.sequence[0]).phase == REVIEW_PHASE[mode]

    def test_improve_codes_review_reads_code_review_never_review_code(self):
        assert iw.review_phase_for("apply-code-fixes") == "code-review"
        with pytest.raises(KeyError):
            wbs.REVIEW_PHASE_BY_WRITE_KIND["apply-code-fixes"]
        with pytest.raises(KeyError):
            iw.review_phase_for("review-code")

    @pytest.mark.parametrize("mode", MODES)
    def test_the_parser_carries_the_two_attributes_full_reads(self, mode, tmp_path):
        args = _args(mode, tmp_path / "p.md")
        assert args.consolidate is False and args.review_tasks is False

    def test_full_is_not_a_review_then_apply_composite(self):
        with pytest.raises(ValueError):
            iw.step_indices("full")


# ---------------------------------------------------------------------------
# 2. The argv each step receives
# ---------------------------------------------------------------------------

class TestStepArgv:
    def test_a_fresh_improve_plan_runs_both_steps_with_their_flags(
        self, tmp_path, stub, capsys
    ):
        calls = stub()
        plan = _plan(tmp_path)
        assert _main(PLAN, plan) == 0
        assert [word for word, _argv, _held in calls] == [
            "review-plan", "apply-suggestions",
        ], "no consolidation pass: the review recommended none"
        assert _argv_of(calls, "review-plan") == ["--yes"]
        assert _argv_of(calls, "apply-suggestions") == APPLY_FLAGS

    def test_a_fresh_improve_code_runs_both_steps_with_their_flags(
        self, repo, stub, capsys
    ):
        calls = stub()
        plan = _plan(repo)
        assert _main(CODE, plan) == 0
        assert [word for word, _argv, _held in calls] == [
            "review-code", "apply-code-fixes",
        ]
        assert _argv_of(calls, "review-code") == ["--yes"]
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("selectors, review_gets", [
        (["--quick"], ["--quick"]),
        (["--models", "a:b", "c:d"], ["--models", "a:b", "c:d"]),
        (["--models", "a:b", "--quick"], ["--models", "a:b"]),
    ], ids=["quick", "models", "models-win"])
    def test_model_selectors_reach_the_review_and_never_the_apply(
        self, repo, stub, capsys, mode, selectors, review_gets
    ):
        calls = stub()
        plan = _plan(repo)
        assert _main(mode, plan, *selectors) == 0
        review, apply = mode_table.resolve(mode).sequence
        assert _argv_of(calls, review) == ["--yes", *review_gets]
        assert _argv_of(calls, apply) == APPLY_FLAGS

    @pytest.mark.parametrize("mode", MODES)
    def test_the_cross_cutting_tail_is_the_literal_list(
        self, repo, stub, capsys, mode
    ):
        calls = stub()
        plan = _plan(repo)
        assert _main(mode, plan, "--host", "codex", "--exec-subagents",
                     "--dry-run-subagents", "--yes", "--skip-prerequisites") == 0
        assert calls
        for word, argv, _held in calls:
            assert argv[-len(TAIL):] == TAIL, (word, argv)
            assert argv.count("--yes") == 1, (word, argv)
            assert "--skip-prerequisites" not in argv
            assert "--interactive" not in argv and "--detach" not in argv

    @pytest.mark.parametrize("mode", MODES)
    def test_judge_and_agent_decide_spellings_are_accepted_and_change_nothing(
        self, repo, stub, capsys, mode
    ):
        for spelling in ("--agent-decide", "--claude-decide", "--let-claude-decide",
                         "--no-confirm"):
            calls = stub()
            calls.clear()
            plan = _plan(repo, name=f"p{spelling.strip('-')}")
            assert _main(mode, plan, spelling) == 0, spelling
            review, apply = mode_table.resolve(mode).sequence
            assert _argv_of(calls, review) == ["--yes"]
            assert _argv_of(calls, apply) == APPLY_FLAGS

    @pytest.mark.parametrize("scope, review_gets", [
        (["--files", "b.py", "--files", "a.py"],
         ["--files", "b.py", "--files", "a.py"]),
        (["--files-from", "LIST"], ["--files-from", "LIST"]),
        (["--base-ref", "origin/release"], ["--base-ref", "origin/release"]),
        (["--files", "b.py", "--files", "a.py", "--files-from", "LIST",
          "--base-ref", "origin/release"],
         ["--files", "b.py", "--files", "a.py", "--files-from", "LIST",
          "--base-ref", "origin/release"]),
    ], ids=["files", "files-from", "base-ref", "all-together"])
    def test_each_scope_flag_reaches_review_code_as_given(
        self, repo, stub, capsys, scope, review_gets
    ):
        listing = repo / "scope list.txt"
        listing.write_text("c.py\n", encoding="utf-8")
        scope = [str(listing) if token == "LIST" else token for token in scope]
        review_gets = [str(listing) if t == "LIST" else t for t in review_gets]
        calls = stub()
        plan = _plan(repo)
        assert _main(CODE, plan, *scope) == 0
        assert _argv_of(calls, "review-code") == ["--yes", *review_gets]
        typed_ref = "--base-ref" in scope
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS + (
            ["--base-ref", "origin/release"] if typed_ref else []
        ), "the apply gets --base-ref only when typed, while step 1 runs"

    def test_a_step_one_that_runs_reads_nothing_from_the_manifest(
        self, repo, stub, capsys
    ):
        calls = stub()
        plan = _plan(repo)
        _write_manifest(plan, "origin/old")
        assert _main(CODE, plan) == 0
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS


class TestTheRecordedDiffBase:
    """An apply-only re-run of improve-code keeps the review's diff base."""

    def _reviewed(self, repo, base_ref="origin/release"):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0)})
        if base_ref is not None:
            _write_manifest(plan, base_ref)
        return plan

    def test_the_apply_gets_the_ref_the_review_used(self, repo, stub, capsys):
        calls = stub()
        plan = self._reviewed(repo)
        assert _main(CODE, plan) == 0
        assert [word for word, _a, _h in calls] == ["apply-code-fixes"]
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS + [
            "--base-ref", "origin/release",
        ]

    def test_a_typed_ref_wins_and_is_not_doubled(self, repo, stub, capsys):
        calls = stub()
        plan = self._reviewed(repo)
        assert _main(CODE, plan, "--base-ref", "HEAD~3") == 0
        argv = _argv_of(calls, "apply-code-fixes")
        assert argv == APPLY_FLAGS + ["--base-ref", "HEAD~3"]

    @pytest.mark.parametrize("manifest", ["missing", "unreadable", "empty"])
    def test_no_usable_manifest_forwards_nothing(self, repo, stub, capsys, manifest):
        calls = stub()
        plan = self._reviewed(repo, base_ref=None if manifest == "missing" else
                              ("" if manifest == "empty" else "x"))
        if manifest == "unreadable":
            (_out(plan) / "code-review" / "review-scope.md").write_text(
                "not a manifest\n", encoding="utf-8")
        assert _main(CODE, plan) == 0
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS

    def test_the_record_carries_the_ref(self, repo):
        plan = self._reviewed(repo)
        assert _records(CODE, plan).recorded_base_ref == "origin/release"
        assert _records(PLAN, plan).recorded_base_ref is None


class TestConsolidation:
    """improve-plan's advisory pass, from a ``parse_args``-built namespace."""

    def _review_files(self, plan, actionable, consolidated=False):
        phase_dir = _out(plan) / "review-plan"
        phase_dir.mkdir(parents=True, exist_ok=True)
        statuses = ["valid", "needs-human-decision"]
        (phase_dir / "validation.json").write_text(json.dumps({"groups": [
            {"group_index": i, "status": statuses[i % 2]} for i in range(actionable)
        ]}), encoding="utf-8")
        grouped = json.dumps({"groups": []})
        (phase_dir / "grouped.json").write_text(grouped, encoding="utf-8")
        if consolidated:
            import hashlib

            (phase_dir / "consolidated.json").write_text(json.dumps({"metadata": {
                "grouped_hash": hashlib.sha256(grouped.encode()).hexdigest(),
            }}), encoding="utf-8")

    def test_a_recommended_pass_runs_between_the_review_and_the_apply(
        self, tmp_path, stub, capsys
    ):
        calls = stub()
        plan = _plan(tmp_path)
        _write_state(plan, completed={"review-plan": T(0)})
        self._review_files(plan, 15)
        plans = iw.plan_steps(PLAN, _args(PLAN, plan), _records(PLAN, plan))
        assert [(p.index, p.decision) for p in plans][1] == ("1b", "run")
        assert plans[1].flags == ("--consolidate", "--yes")
        assert _main(PLAN, plan) == 0
        assert [(word, argv) for word, argv, _h in calls] == [
            ("review-plan", ["--consolidate", "--yes"]),
            ("apply-suggestions", APPLY_FLAGS),
        ]

    @pytest.mark.parametrize("actionable, consolidated, reason", [
        (15, True, "already ran on these review groups"),
        (14, False, "not recommended by review-plan"),
        (0, False, "not recommended by review-plan"),
    ], ids=["consolidated", "fourteen", "zero"])
    def test_no_pass_when_not_recommended(
        self, tmp_path, stub, capsys, actionable, consolidated, reason
    ):
        calls = stub()
        plan = _plan(tmp_path)
        _write_state(plan, completed={"review-plan": T(0)})
        self._review_files(plan, actionable, consolidated)
        records = _records(PLAN, plan)
        assert records.consolidation_skip == reason
        assert _main(PLAN, plan) == 0
        assert [word for word, _a, _h in calls] == ["apply-suggestions"]
        assert f"[SKIPPED] Step 1b: consolidation: {reason}" in capsys.readouterr().out
        step = iw.plan_steps(PLAN, _args(PLAN, plan), records)[1]
        assert (step.index, step.decision, step.reason) == ("1b", "skip", reason)

    def test_a_failed_pass_warns_and_the_apply_still_runs(
        self, tmp_path, stub, capsys
    ):
        def fail_the_pass(workflow, mode_word, argv):
            if "--consolidate" in argv:
                raise_code.append(1)

        raise_code = []
        calls = stub()

        def run_mode(self, mode_word, extra):
            calls.append((mode_word, list(extra), None))
            if "--consolidate" in extra:
                return 1
            manager = StateManager(self.plan_path)
            manager.mark_phase_completed(mode_table.phase_for(mode_word))
            manager.save()
            return 0

        plan = _plan(tmp_path)
        _write_state(plan, completed={"review-plan": T(0)})
        self._review_files(plan, 15)
        original = fwo.FullWorkflow.run_mode
        fwo.FullWorkflow.run_mode = run_mode
        try:
            assert _main(PLAN, plan) == 0
        finally:
            fwo.FullWorkflow.run_mode = original
        assert [word for word, _a, _h in calls] == ["review-plan", "apply-suggestions"]
        assert "consolidation exited 1" in capsys.readouterr().err

    def test_a_finished_run_with_an_open_recommendation_is_already_complete(
        self, tmp_path, stub, capsys
    ):
        calls = stub()
        plan = _plan(tmp_path)
        _write_state(plan, completed={"review-plan": T(0),
                                      "apply-suggestions": T(5)})
        self._review_files(plan, 15)
        assert _main(PLAN, plan) == 2
        assert calls == []
        plans = iw.plan_steps(PLAN, _args(PLAN, plan), _records(PLAN, plan))
        assert [p.index for p in plans] == ["1", "2"], "no 1b under [ALREADY_COMPLETE]"

    def test_step_one_running_reads_after_review(self, tmp_path):
        plan = _plan(tmp_path)
        plans = iw.plan_steps(PLAN, _args(PLAN, plan), _records(PLAN, plan))
        assert [(p.index, p.step, p.decision) for p in plans] == [
            ("1", "review-plan", "run"),
            ("1b", "consolidate", "after-review"),
            ("2", "apply-suggestions", "run"),
        ]

    def test_improve_code_has_no_consolidation_step(self, repo, stub, capsys):
        plan = _plan(repo)
        assert [w for _i, w in iw.step_indices(CODE)] == [
            "review-code", "apply-code-fixes"]
        plans = iw.plan_steps(CODE, _args(CODE, plan), _records(CODE, plan))
        assert [p.index for p in plans] == ["1", "2"]
        calls = stub()
        assert _main(CODE, plan) == 0
        assert not any("--consolidate" in argv for _w, argv, _h in calls)


# ---------------------------------------------------------------------------
# 3. Skip logic
# ---------------------------------------------------------------------------

class TestSkipLogic:
    @pytest.mark.parametrize("mode", MODES)
    def test_a_recorded_review_runs_only_the_apply(self, repo, stub, capsys, mode):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
        assert _main(mode, plan) == 0
        assert [w for w, _a, _h in calls] == [mode_table.resolve(mode).sequence[1]]

    @pytest.mark.parametrize("mode, line", [
        (PLAN, "[ALREADY_COMPLETE] improve-plan: review-plan completed "
               "2026-10-01T12:00:00, apply-suggestions completed "
               "2026-10-01T12:05:00; nothing to do. --force runs review-plan again."),
        (CODE, "[ALREADY_COMPLETE] improve-code: review-code completed "
               "2026-10-01T12:00:00, apply-code-fixes completed "
               "2026-10-01T12:05:00; nothing to do. --force runs review-code again."),
    ], ids=MODES)
    def test_both_recorded_is_exit_2_with_the_pinned_line(
        self, repo, stub, capsys, mode, line
    ):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0),
                                      APPLY_PHASE[mode]: T(5)})
        ctx = cli_common.activate_result(cli_common.ResultContext(mode=mode))
        try:
            assert _main(mode, plan) == 2
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert line in out.splitlines()
        assert calls == []
        assert ctx.next is None
        title = "Improve Plan Summary" if mode == PLAN else "Improve Code Summary"
        assert f"## {title}" in out

    def test_an_apply_on_record_only_as_skipped_says_skipped(self, repo):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0)},
                     skipped={"apply-suggestions": T(5)})
        args = _args(PLAN, plan)
        records = iw.read_records(PLAN, plan, args)
        assert iw.nothing_to_do(PLAN, args, records)
        assert iw.already_complete_line(PLAN, args, records) == (
            "[ALREADY_COMPLETE] improve-plan: review-plan completed "
            "2026-10-01T12:00:00, apply-suggestions skipped 2026-10-01T12:05:00; "
            "nothing to do. --force runs review-plan again."
        )

    @pytest.mark.parametrize("mode", MODES)
    def test_an_apply_older_than_its_review_runs(self, repo, stub, capsys, mode):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={APPLY_PHASE[mode]: T(0),
                                      REVIEW_PHASE[mode]: T(5)})
        assert _main(mode, plan) == 0
        assert [w for w, _a, _h in calls] == [mode_table.resolve(mode).sequence[1]]

    def test_a_code_review_stale_after_implement_runs_both(self, repo, stub, capsys):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0),
                                      "apply-code-fixes": T(5),
                                      "implement": T(10)})
        records = _records(CODE, plan)
        assert records.review_stale and not records.review_completed
        assert _main(CODE, plan) == 0
        assert [w for w, _a, _h in calls] == ["review-code", "apply-code-fixes"]
        assert _argv_of(calls, "review-code") == ["--yes"], (
            "a stale review forces itself; the composite adds nothing"
        )

    @pytest.mark.parametrize("mode", MODES)
    def test_a_review_on_record_only_as_skipped_runs(self, repo, stub, capsys, mode):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, skipped={REVIEW_PHASE[mode]: T(0)})
        assert _main(mode, plan) == 0
        assert [w for w, _a, _h in calls] == list(mode_table.resolve(mode).sequence)
        step = iw.plan_steps(mode, _args(mode, plan),
                             iw.Records(mode=mode, review_skipped=True))[0]
        assert (step.decision, step.reason) == ("run", "on record only as skipped")

    def test_full_passes_over_the_same_skip_record(self, repo):
        """The deliberate divergence: ``full`` keeps a skipped review skipped."""
        plan = _plan(repo)
        _write_state(plan, skipped={"review-plan": T(0)})
        workflow = fwo.FullWorkflow(fwo.parse_args(["--plan-file", str(plan), "--yes"]))
        review = workflow.steps()[0]
        assert workflow.skip_reason(review) == "skipped"

    def test_a_review_completed_then_skipped_is_skipped_as_completed(self, repo):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0), "apply-suggestions": T(9)},
                     skipped={"review-plan": T(5)})
        args = _args(PLAN, plan)
        records = iw.read_records(PLAN, plan, args)
        assert iw.skip_reason(PLAN, "review-plan", args, records) == "completed"
        assert "review-plan completed 2026-10-01T12:00:00" in (
            iw.already_complete_line(PLAN, args, records))

    @pytest.mark.parametrize("mode", MODES)
    def test_an_unfinished_review_runs_both_from_one_read(
        self, repo, stub, capsys, monkeypatch, mode
    ):
        calls = stub()
        plan = _plan(repo)
        # An earlier cycle's apply, and a review whose completion a --force cleared.
        _write_state(plan, completed={APPLY_PHASE[mode]: T(0)})
        reads = []
        real = iw.read_records
        monkeypatch.setattr(iw, "read_records",
                            lambda *a, **k: reads.append(a) or real(*a, **k))
        assert _main(mode, plan) == 0
        assert [w for w, _a, _h in calls] == list(mode_table.resolve(mode).sequence)
        assert len(reads) == 1, "the driver reads records once"

    @pytest.mark.parametrize("skipped_at, runs", [(T(10), False), (T(-10), True)],
                             ids=["after-review", "before-review"])
    def test_an_apply_recorded_only_as_skipped(self, repo, skipped_at, runs):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0)},
                     skipped={"apply-suggestions": skipped_at})
        args = _args(PLAN, plan)
        said = iw.skip_reason(PLAN, "apply-suggestions", args,
                              iw.read_records(PLAN, plan, args))
        assert said == (None if runs else "skipped")

    @pytest.mark.parametrize("completed_at, skipped_at, word, runs", [
        (T(10), T(-10), "completed", False),  # skipped round one, completed two
        (T(-10), T(10), "skipped", False),    # the reverse
        (T(-10), T(-5), "skipped", True),     # both before the review
    ], ids=["skip-then-complete", "complete-then-skip", "both-older"])
    def test_both_apply_records_compare_the_newer(
        self, repo, completed_at, skipped_at, word, runs
    ):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0),
                                      "apply-suggestions": completed_at},
                     skipped={"apply-suggestions": skipped_at})
        args = _args(PLAN, plan)
        records = iw.read_records(PLAN, plan, args)
        assert records.apply_word == word
        said = iw.skip_reason(PLAN, "apply-suggestions", args, records)
        assert said == (None if runs else word)

    @pytest.mark.parametrize("case", [
        "unparseable-completion", "skip-without-skipped-at", "skip-not-a-dict",
        "unparseable-review-stamp",
    ])
    def test_a_date_that_does_not_parse_keeps_the_apply_done(self, repo, case):
        plan = _plan(repo)
        completed = {"review-plan": T(0)}
        skipped = {}
        if case == "unparseable-completion":
            completed["apply-suggestions"] = "not a date"
        elif case == "skip-without-skipped-at":
            skipped["apply-suggestions"] = {"reason": "x"}
        elif case == "skip-not-a-dict":
            skipped["apply-suggestions"] = "yesterday-ish"
        else:
            completed["review-plan"] = "garbled"
            completed["apply-suggestions"] = T(-10)
        _write_state(plan, completed=completed, skipped=skipped)
        args = _args(PLAN, plan)
        records = iw.read_records(PLAN, plan, args)
        if case == "unparseable-review-stamp":
            # `is_phase_completed` holds whatever the stamp says.
            assert records.review_completed
        assert not records.apply_predates_review
        assert iw.skip_reason(PLAN, "apply-suggestions", args, records) is not None

    def test_one_instant_across_an_offset_and_an_equal_stamp_keeps_done(self, repo):
        plan = _plan(repo)
        local = BASE.astimezone()
        review_utc = local.astimezone(timezone.utc).isoformat()
        _write_state(plan, completed={"review-plan": review_utc,
                                      "apply-suggestions": BASE.isoformat()})
        args = _args(PLAN, plan)
        records = iw.read_records(PLAN, plan, args)
        assert not records.apply_predates_review, "an equal instant keeps it done"
        assert iw.skip_reason(PLAN, "apply-suggestions", args, records) == "completed"

    def test_three_stamp_readers_agree(self):
        cases = [T(0), BASE.astimezone(timezone.utc).isoformat(), "", None, 7, "nope"]
        for value in cases:
            assert (
                sm._parse_completion_time(value)
                == wbs._parse_local_timestamp(value)
                == prereq._parse_timestamp(value)
            ), value

    @pytest.mark.parametrize("started, completed_at", [(T(10), T(5)), (T(10), None)],
                             ids=["start-newer-than-completion", "start-beside-a-skip"])
    def test_a_started_apply_runs(self, repo, stub, capsys, started, completed_at):
        calls = stub()
        plan = _plan(repo)
        completed = {"review-plan": T(0)}
        skipped = {}
        if completed_at:
            completed["apply-suggestions"] = completed_at
        else:
            skipped["apply-suggestions"] = T(5)
        _write_state(plan, completed=completed, skipped=skipped,
                     started={"apply-suggestions": started})
        assert _main(PLAN, plan) == 0
        assert [w for w, _a, _h in calls] == ["apply-suggestions"]


# ---------------------------------------------------------------------------
# 4. --resume, inferred and typed; --force, a new round
# ---------------------------------------------------------------------------

class TestResumeAndForce:
    def test_an_apply_interrupted_natively_gets_resume_untyped(
        self, repo, stub, capsys
    ):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0)},
                     started={"apply-suggestions": T(5)})
        step = iw.plan_steps(PLAN, _args(PLAN, plan), _records(PLAN, plan))[-1]
        assert (step.decision, step.reason) == ("run", "apply in progress for this review")
        assert step.flags == tuple(APPLY_FLAGS + ["--resume"])
        assert _main(PLAN, plan) == 0
        assert _argv_of(calls, "apply-suggestions") == APPLY_FLAGS + ["--resume"]

    def test_an_executor_interruption_with_a_completed_batch_resumes(
        self, repo, stub, capsys
    ):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0)})
        assert prereq.APPLY_BATCH_ID_PATTERNS["apply-code-fixes"].match(
            "apply-fix-batch-1")
        state = StateManager(plan)
        state.update_task_status("apply-fix-batch-1", "completed")
        state.save()
        assert _records(CODE, plan).apply_in_progress
        assert _main(CODE, plan) == 0
        assert _argv_of(calls, "apply-code-fixes")[:4] == APPLY_FLAGS + ["--resume"]

    def test_a_typed_resume_is_not_doubled(self, repo, stub, capsys):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0)},
                     started={"apply-suggestions": T(5)})
        assert _main(PLAN, plan, "--resume") == 0
        assert _argv_of(calls, "apply-suggestions").count("--resume") == 1

    def test_progress_older_than_the_review_infers_nothing(self, repo, stub, capsys):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={"apply-suggestions": T(-20), "review-plan": T(0)},
                     started={"apply-suggestions": T(-30)})
        assert not _records(PLAN, plan).apply_in_progress
        assert _main(PLAN, plan) == 0
        assert _argv_of(calls, "apply-suggestions") == APPLY_FLAGS

    @pytest.mark.parametrize("mode", MODES)
    def test_force_on_a_finished_plan_is_a_new_round(self, repo, stub, capsys, mode):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0),
                                      APPLY_PHASE[mode]: T(5)},
                     started={APPLY_PHASE[mode]: T(1)})
        assert _main(mode, plan, "--force", "--resume") == 0
        review, apply = mode_table.resolve(mode).sequence
        assert _argv_of(calls, review) == ["--yes", "--force", "--rerun-all"]
        assert _argv_of(calls, apply) == APPLY_FLAGS, (
            "no --resume, typed or inferred, after a review that ran"
        )

    def test_a_typed_resume_reaches_the_apply_only_when_step_one_is_skipped(
        self, repo
    ):
        plan = _plan(repo)
        fresh = _args(PLAN, plan, "--resume")
        records = _records(PLAN, plan, "--resume")
        assert iw.step_flags(PLAN, "apply-suggestions", fresh, records) == APPLY_FLAGS
        _write_state(plan, completed={"review-plan": T(0)})
        records = _records(PLAN, plan, "--resume")
        assert iw.step_flags(PLAN, "apply-suggestions", fresh, records) == (
            APPLY_FLAGS + ["--resume"])
        assert "--resume" not in iw.step_flags(PLAN, "review-plan", fresh, records)

    def test_the_plan_for_a_new_round(self, repo):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0), "apply-suggestions": T(5)})
        args = _args(PLAN, plan, "--force")
        plans = iw.plan_steps(PLAN, args, iw.read_records(PLAN, plan, args))
        assert [(p.index, p.decision, p.reason, p.fresh, p.flags) for p in plans] == [
            ("1", "run", "new round, --force", True, ("--yes", "--force", "--rerun-all")),
            ("1b", "after-review", "runs if the review recommends it", False,
             ("--consolidate", "--yes")),
            ("2", "run", "review not applied yet", False, tuple(APPLY_FLAGS)),
        ]


# ---------------------------------------------------------------------------
# 5. The code review's scope
# ---------------------------------------------------------------------------

class TestScope:
    def _reviewed(self, repo, scope=None, apply_done=False):
        plan = _plan(repo)
        completed = {"code-review": T(0)}
        if apply_done:
            completed["apply-code-fixes"] = T(5)
        extra = {"review_scope_files": list(scope)} if scope is not None else None
        _write_state(plan, completed=completed, extra=extra)
        return plan

    def test_a_typed_scope_over_the_default_one_reviews_again(
        self, repo, stub, capsys
    ):
        calls = stub()
        plan = self._reviewed(repo, apply_done=True)
        assert _main(CODE, plan, "--files", "a.py") == 0
        assert _argv_of(calls, "review-code") == ["--yes", "--force", "--files", "a.py"]
        out = capsys.readouterr().out
        assert (
            "[IMPROVE_SCOPE_CHANGED] improve-code: the code review on record "
            "reviewed the default scope (tracked_files, else the git diff; no "
            "review_scope_files recorded); this run asks for 1 files "
            "(--files/--files-from). Reviewing again (as --force)."
        ) in out.splitlines()
        assert [w for w, _a, _h in calls] == ["review-code", "apply-code-fixes"]
        step = iw.plan_steps(CODE, _args(CODE, plan, "--files", "a.py"),
                             _records(CODE, plan, "--files", "a.py"))[0]
        assert (step.decision, step.reason, step.fresh) == (
            "run", "review scope changed", True)

    def test_with_resume_typed_the_apply_after_it_gets_none(self, repo, stub, capsys):
        calls = stub()
        plan = self._reviewed(repo, apply_done=True)
        assert _main(CODE, plan, "--files", "a.py", "--resume") == 0
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS

    def test_the_same_set_in_another_order_skips_the_review(self, repo, stub, capsys):
        calls = stub()
        plan = self._reviewed(repo, scope=["a.py", "b.py"])
        assert _main(CODE, plan, "--files", "b.py", "--files", "a.py") == 0
        assert [w for w, _a, _h in calls] == ["apply-code-fixes"]

    def test_a_different_set_runs_the_review(self, repo, stub, capsys):
        calls = stub()
        plan = self._reviewed(repo, scope=["a.py", "b.py"])
        assert _main(CODE, plan, "--files", "a.py") == 0
        assert _argv_of(calls, "review-code") == ["--yes", "--force", "--files", "a.py"]
        assert "2 files given with --files (state.json review_scope_files)" in (
            capsys.readouterr().out)

    def test_force_with_scope_flags_compares_nothing_and_forces_once(
        self, repo, stub, capsys
    ):
        calls = stub()
        plan = self._reviewed(repo, scope=["a.py"], apply_done=True)
        assert _main(CODE, plan, "--force", "--files", "a.py") == 0
        argv = _argv_of(calls, "review-code")
        assert argv == ["--yes", "--force", "--rerun-all", "--files", "a.py"]
        assert "[IMPROVE_SCOPE_CHANGED]" not in capsys.readouterr().out

    def test_no_scope_flag_leaves_a_completed_review_standing(self, repo, stub, capsys):
        calls = stub()
        plan = self._reviewed(repo, scope=["a.py"])
        assert _main(CODE, plan) == 0
        assert [w for w, _a, _h in calls] == ["apply-code-fixes"]

    def test_an_unfinished_review_resumes_its_recorded_scope(self, repo, stub, capsys):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, extra={"review_scope_files": ["b.py", "a.py", "b.py", ""]})
        assert _main(CODE, plan) == 0
        assert _argv_of(calls, "review-code") == [
            "--yes", "--files", "b.py", "--files", "a.py"]
        assert iw.continuation_flags(CODE, _args(CODE, plan)) == ["--resume"]

    def test_a_stale_review_and_a_new_round_carry_no_recorded_scope(self, repo):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0), "implement": T(5)},
                     extra={"review_scope_files": ["a.py"]})
        stale = _records(CODE, plan)
        assert stale.review_stale
        assert iw.step_flags(CODE, "review-code", _args(CODE, plan), stale) == ["--yes"]
        forced = _args(CODE, plan, "--force")
        assert iw.step_flags(CODE, "review-code", forced,
                             iw.read_records(CODE, plan, forced)) == [
            "--yes", "--force", "--rerun-all"]

    def test_files_from_stdin_on_a_completed_review_is_not_read(
        self, repo, stub, capsys, monkeypatch
    ):
        class NoStdin:
            def read(self, *a):
                pytest.fail("stdin was read")

            readline = read

        monkeypatch.setattr(sys, "stdin", NoStdin())
        calls = stub()
        plan = self._reviewed(repo)
        assert _main(CODE, plan, "--files-from", "-") == 0
        assert [w for w, _a, _h in calls] == ["apply-code-fixes"]
        assert (
            "NOTE: improve-code does not compare --files-from - with the scope on "
            "record; the completed code review stands. Pass the list with --files "
            "or --files-from <file>, or add --force, to review it."
        ) in capsys.readouterr().out.splitlines()

    @pytest.mark.parametrize("extra", [[], ["--files", "a.py"]], ids=["alone", "beside-files"])
    @pytest.mark.parametrize("force", [False, True], ids=["fresh", "force"])
    def test_files_from_stdin_for_a_review_that_runs_under_detach_is_refused(
        self, repo, capsys, monkeypatch, no_boundary, extra, force
    ):
        class NoStdin:
            def read(self, *a):
                pytest.fail("stdin was read")

        monkeypatch.setattr(sys, "stdin", NoStdin())
        plan = _plan(repo)
        if force:
            _write_state(plan, completed={"code-review": T(0),
                                          "apply-code-fixes": T(5)})
        before = sorted(p.name for p in _out(plan).iterdir()) if _out(plan).exists() else None
        argv = ["--detach", "--files-from", "-", *extra, *(["--force"] if force else [])]
        assert _main(CODE, plan, *argv) == 1
        err = capsys.readouterr().err
        assert (
            "ERROR: --files-from - reads stdin, which a --detach worker does not "
            "inherit; write the list to a file and pass its path instead"
        ) in err.splitlines()
        after = sorted(p.name for p in _out(plan).iterdir()) if _out(plan).exists() else None
        assert after == before, "no run dir, nothing written"

    def test_the_worker_backstop_refuses_a_review_that_turned_run(
        self, repo, stub, capsys, monkeypatch
    ):
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0), "implement": T(5)})
        args = _args(CODE, plan, "--files-from", "-", "--internal-worker")
        workflow = iw.ImproveWorkflow(args, CODE)
        assert workflow.run() == 1
        assert calls == []
        assert "--files-from - reads stdin" in capsys.readouterr().err

    def test_in_the_foreground_the_phase_reads_stdin_itself(self, repo, stub, capsys):
        calls = stub()
        plan = _plan(repo)
        assert _main(CODE, plan, "--files-from", "-") == 0
        assert _argv_of(calls, "review-code") == ["--yes", "--files-from", "-"]

    @pytest.mark.parametrize("detach_flag", [[], ["--detach"]], ids=["foreground", "detach"])
    @pytest.mark.parametrize("scope, says", [
        (["--files-from", "missing.txt"], "--files-from missing.txt: cannot read it"),
        (["--files", "/elsewhere/a.py"], "--files path is outside the repository"),
    ], ids=["unreadable-list", "outside"])
    def test_a_scope_that_cannot_be_read_is_exit_1_before_any_step(
        self, repo, stub, capsys, no_boundary, detach_flag, scope, says
    ):
        calls = stub()
        plan = _plan(repo)
        assert _main(CODE, plan, *scope, *detach_flag) == 1
        assert says in capsys.readouterr().err
        assert calls == []
        assert not _out(plan).exists(), "nothing written"


# ---------------------------------------------------------------------------
# 6. A validation.json that outlived its completion
# ---------------------------------------------------------------------------

class TestLeftoverValidation:
    @pytest.mark.parametrize("mode", MODES)
    def test_a_review_stopped_before_its_regroup_starts_fresh(
        self, repo, stub, capsys, mode
    ):
        calls = stub()
        plan = _plan(repo)
        # A completed round, then a --force that cleared the completion and
        # died before its regroup: the earlier validation.json is on disk.
        _write_state(plan, completed={APPLY_PHASE[mode]: T(5)})
        review_dir = _out(plan) / REVIEW_PHASE[mode]
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "validation.json").write_text("[]", encoding="utf-8")
        records = _records(mode, plan)
        assert records.leftover_validation
        step = iw.plan_steps(mode, _args(mode, plan), records)[0]
        assert (step.decision, step.fresh) == ("run", True)
        assert _main(mode, plan) == 0
        review = mode_table.resolve(mode).sequence[0]
        expected = ["--yes", "--force"] if mode == CODE else ["--yes"]
        assert _argv_of(calls, review) == expected
        assert [w for w, _a, _h in calls] == list(mode_table.resolve(mode).sequence)

    def test_a_retired_validation_round_does_not_count(self, repo):
        plan = _plan(repo)
        review_dir = _out(plan) / "code-review"
        review_dir.mkdir(parents=True)
        (review_dir / "validation.20261001T120000.stale.json").write_text("[]")
        assert not _records(CODE, plan).leftover_validation

    def test_a_stale_review_over_the_same_files_gets_no_force(self, repo):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0), "implement": T(5)})
        review_dir = _out(plan) / "code-review"
        review_dir.mkdir(parents=True)
        (review_dir / "validation.json").write_text("[]")
        records = _records(CODE, plan)
        assert iw.step_flags(CODE, "review-code", _args(CODE, plan), records) == ["--yes"]

    def test_the_stopped_round_rerun_with_force_is_another_round(self, repo):
        plan = _plan(repo)
        review_dir = _out(plan) / "code-review"
        review_dir.mkdir(parents=True)
        (review_dir / "validation.json").write_text("[]")
        args = _args(CODE, plan, "--force")
        flags = iw.step_flags(CODE, "review-code", args, iw.read_records(CODE, plan, args))
        assert flags == ["--yes", "--force", "--rerun-all"]


# ---------------------------------------------------------------------------
# 7. A live run is refused, never joined
# ---------------------------------------------------------------------------

def _live_run(directory: Path, run_id: str, mode: str) -> process_control.PhaseLock:
    """A held ``.lock`` plus a live ``run`` record in *directory* (this process)."""
    directory.mkdir(parents=True, exist_ok=True)
    status_file.claim_run(str(directory), {
        "run_id": run_id, "pid": os.getpid(),
        "process_start_time": process_control.process_start_time(os.getpid()),
        "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
        "state": "running", "detached": True, "mode": mode,
    })
    lock = process_control.PhaseLock(detach.lock_path(str(directory)))
    lock.acquire({"run_id": run_id, "pid": os.getpid(),
                  "process_start_time": process_control.process_start_time(os.getpid())})
    return lock


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class TestALiveRun:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("where", ["review", "apply", "full", "own"])
    @pytest.mark.parametrize("flags", [[], ["--force"], ["--resume"]],
                             ids=["bare", "force", "resume"])
    def test_a_live_owner_is_exit_3_before_any_step(
        self, repo, stub, capsys, no_boundary, mode, where, flags
    ):
        calls = stub()
        plan = _plan(repo)
        state_file = _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
        before = state_file.read_bytes()
        dirs = {"review": REVIEW_PHASE[mode], "apply": APPLY_DIR[mode],
                "full": "full", "own": mode}
        lock = _live_run(_out(plan) / dirs[where], "run-live-0001", "review-plan")
        try:
            assert _main(mode, plan, *flags) == 3
        finally:
            lock.release()
        err = capsys.readouterr().err
        assert f"ERROR: {mode}: a live run owns " in err
        assert f"{dirs[where]}/ (run run-live-0001, pid {os.getpid()}, running" in err
        assert "Watch it: " in err and "--run-id run-live-0001" in err
        assert "Abort it: " in err and "--abort --run-id run-live-0001" in err
        assert calls == []
        assert state_file.read_bytes() == before, "nothing written"
        assert not (_out(plan) / mode).exists() or where == "own"

    @pytest.mark.parametrize("mode", MODES)
    def test_under_detach_too_with_no_worker(self, repo, capsys, no_boundary, mode):
        plan = _plan(repo)
        lock = _live_run(_out(plan) / REVIEW_PHASE[mode], "run-live-0002", "x")
        try:
            assert _main(mode, plan, "--detach") == 3
        finally:
            lock.release()
        assert "a live run owns" in capsys.readouterr().err

    def test_a_dead_holder_blocks_nothing(self, repo, stub, capsys):
        calls = stub()
        plan = _plan(repo)
        review_dir = _out(plan) / "review-plan"
        review_dir.mkdir(parents=True)
        status_file.claim_run(str(review_dir), {
            "run_id": "run-dead-0001", "pid": _dead_pid(),
            "process_start_time": "12345", "state": "running", "detached": True,
            "heartbeat_at": (datetime.now() - timedelta(minutes=10)).isoformat(),
        })
        assert iw.live_owner(PLAN, plan, skip_own_run_dir=False) is None
        assert _main(PLAN, plan) == 0
        assert [w for w, _a, _h in calls] == ["review-plan", "apply-suggestions"]
        assert status_file.read_run(str(review_dir))["state"] == "running", (
            "read-only: classify_run(close_out=False) closed nothing out"
        )

    def test_the_composites_own_run_dir_is_skipped_only_when_asked(self, repo):
        """``skip_own_run_dir`` passes over the own run dir and the step locks
        its own run holds; the entry point asks for it only under
        ``--detach --if-not-running``, where ``launch`` re-attaches (below)."""
        plan = _plan(repo)
        lock = _live_run(_out(plan) / PLAN, "run-own-0001", PLAN)
        step = process_control.PhaseLock(detach.lock_path(str(_out(plan) / "review-plan")))
        (_out(plan) / "review-plan").mkdir()
        step.acquire({"holder": "improve-plan step review-plan", "run_id": "run-own-0001",
                      "run_phase": PLAN, "pid": os.getpid()})
        try:
            assert iw.live_owner(PLAN, plan, skip_own_run_dir=True) is None, (
                "a re-attach is launch's to answer"
            )
            owner = iw.live_owner(PLAN, plan, skip_own_run_dir=False)
            assert owner is not None and owner.directory.endswith("review-plan")
            assert "--phase improve-plan" in owner.status_command, (
                "a step lock's status command watches the composite's run dir"
            )
        finally:
            step.release()
            lock.release()

    @staticmethod
    def _own_live_run(plan: Path, mode: str, where: str):
        """The composite's own detached run, live: mid-step (it holds the
        review's step lock too) or between steps (its run dir alone)."""
        locks = [_live_run(_out(plan) / mode, "run-own-0002", mode)]
        if where == "mid-step":
            review_dir = _out(plan) / REVIEW_PHASE[mode]
            review_dir.mkdir(parents=True, exist_ok=True)
            step = process_control.PhaseLock(detach.lock_path(str(review_dir)))
            step.acquire({
                "holder": f"{mode} step {mode_table.resolve(mode).sequence[0]}",
                "mode": mode, "run_id": "run-own-0002", "run_phase": mode,
                "pid": os.getpid(),
                "process_start_time": process_control.process_start_time(os.getpid()),
            })
            locks.append(step)
        return locks

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("where", ["mid-step", "between-steps"])
    @pytest.mark.parametrize("flags", [["--detach"], ["--detach", "--force"],
                                       ["--detach", "--resume"]],
                             ids=["detach", "detach-force", "detach-resume"])
    def test_a_detach_over_its_own_live_run_is_refused_never_taken_over(
        self, repo, stub, capsys, no_boundary, mode, where, flags
    ):
        """Fable review 2026-10-09: the own run dir and its step locks used to
        be skipped on every ``--detach``, so ``--detach --force`` fell through
        to ``launch``'s take-over of a live worker mid-step. ``--force`` and
        ``--resume`` change nothing (§2.5): exit 3, the composite's refusal,
        no boundary, nothing written."""
        calls = stub()
        plan = _plan(repo)
        state_file = _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
        status_path = Path(status_file.status_path(str(_out(plan) / mode)))
        locks = self._own_live_run(plan, mode, where)
        before = (state_file.read_bytes(), status_path.read_bytes())
        try:
            assert _main(mode, plan, *flags) == 3
        finally:
            for lock in reversed(locks):
                lock.release()
        err = capsys.readouterr().err
        assert f"ERROR: {mode}: a live run owns " in err
        assert "(run run-own-0002, pid " in err
        assert "--run-id run-own-0002" in err and "--abort --run-id run-own-0002" in err
        assert "taking" not in err and "take the dir over" not in err
        assert calls == []
        assert (state_file.read_bytes(), status_path.read_bytes()) == before
        assert not (_out(plan) / mode / detach.LOG_NAME).exists()

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("where", ["mid-step", "between-steps"])
    def test_detach_if_not_running_still_reattaches_to_its_own_run(
        self, repo, stub, capsys, monkeypatch, mode, where
    ):
        """Through the real dispatcher and ``launch``: the one invocation that
        passes over its own live run is the re-attach, which ``launch`` answers
        with exit 0 and the run's poll as ``next``."""
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0)})
        monkeypatch.setenv(detach.UV_ENV, sys.executable)
        # The composite implies `--exec-subagents`, so the launcher checks the
        # runner's binary is on PATH first: tested on its own elsewhere, and
        # here only a question about this machine's claude CLI.
        monkeypatch.setattr(detach, "missing_runner_binaries", lambda *a, **k: {})
        for name in ("MULTI_LLM_HOST", mode_table.ROUTE_ENV,
                     detach.PLAN_ARG_ENV, detach.WORKER_ENV):
            monkeypatch.delenv(name, raising=False)
        locks = self._own_live_run(plan, mode, where)
        try:
            code = multi_llm.main([mode, str(plan), "--detach", "--if-not-running"])
        finally:
            for lock in reversed(locks):
                lock.release()
        out = capsys.readouterr().out
        assert code == 0, out
        assert "A detached run is already in flight" in out
        result = cli_common.parse_result_line(out)
        assert "--run-id run-own-0002" in result["next"]
        assert calls == []

    @pytest.mark.parametrize("fail", [False, True], ids=["success", "stop"])
    def test_the_step_lock_is_held_for_the_step_and_released_after(
        self, repo, stub, capsys, fail
    ):
        calls = stub(fail={"review-plan": 1} if fail else None)
        plan = _plan(repo)
        assert _main(PLAN, plan) == (3 if fail else 0)
        for word, _argv, held in calls:
            assert held is not None, f"{word} ran without its step lock"
            assert held["mode"] == PLAN and held["holder"] == f"{PLAN} step {word}"
            assert held["pid"] == os.getpid()
        for name in ("review-plan", "apply-suggestions"):
            path = detach.lock_path(str(_out(plan) / name))
            assert process_control.lock_is_free(path), f"{name} lock left held"

    def test_a_standalone_detach_launch_of_the_phase_stops_at_the_lock(
        self, repo, stub, capsys, monkeypatch
    ):
        plan = _plan(repo)
        outcomes = []

        def launch_beside(workflow, mode_word, argv):
            monkeypatch.setenv(detach.UV_ENV, sys.executable)
            monkeypatch.setattr(detach, "missing_provider_binaries", lambda *a, **k: {})
            try:
                detach.launch(mode=mode_word, argv=["--plan-file", str(plan)],
                              phase_dir=str(_out(plan) / "review-plan"),
                              plan_file=str(plan), phase="review-plan")
            except detach.DetachError as exc:
                outcomes.append(exc.exit_code)

        stub(hook=lambda wf, word, argv: launch_beside(wf, word, argv)
             if word == "review-plan" else None)
        assert _main(PLAN, plan) == 0
        assert outcomes == [3]

    def test_a_step_lock_another_run_holds_stops_the_step(self, repo, capsys):
        plan = _plan(repo)
        review_dir = _out(plan) / "review-plan"
        review_dir.mkdir(parents=True)
        held = process_control.PhaseLock(detach.lock_path(str(review_dir)))
        held.acquire({"holder": "someone else", "pid": os.getpid(),
                      "process_start_time": process_control.process_start_time(os.getpid())})
        try:
            workflow = iw.ImproveWorkflow(_args(PLAN, plan), PLAN)
            assert workflow.run_mode("review-plan", ["--yes"]) == 3
        finally:
            held.release()
        err = capsys.readouterr().err
        assert "a live run owns" in err and "someone else" in err

    def test_a_foreground_step_lock_adopts_no_run_record(self, repo):
        """Fable review 2026-10-09: a step lock whose payload names no
        ``run_id`` is a foreground composite in another shell, and the ``run``
        record in the composite's run dir is some earlier detached run's. It
        lends the refusal no heartbeat, host, abort or poll of that run."""
        plan = _plan(repo)
        own_dir = _out(plan) / PLAN
        own_dir.mkdir(parents=True)
        status_file.claim_run(str(own_dir), {
            "run_id": "run-old-0001", "pid": _dead_pid(), "state": "done",
            "exit_code": 0, "detached": True, "mode": PLAN, "host": "codex",
            "heartbeat_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            "finished_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        })
        review_dir = _out(plan) / "review-plan"
        review_dir.mkdir()
        step = process_control.PhaseLock(detach.lock_path(str(review_dir)))
        step.acquire({
            "holder": f"{PLAN} step review-plan", "mode": PLAN,
            "step": "review-plan", "run_phase": PLAN, "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
        })
        try:
            owner = iw.live_owner(PLAN, plan, skip_own_run_dir=False)
        finally:
            step.release()
        assert owner is not None and owner.directory.endswith("review-plan")
        assert owner.holder == f"{PLAN} step review-plan"
        assert owner.heartbeat_age_s is None
        assert owner.abort_command is None
        assert "--phase review-plan" in owner.status_command, (
            "the poll watches the locked step's own dir, where a foreground "
            "phase writes its progress"
        )
        for absent in ("--phase improve-plan", "--run-id", "--host"):
            assert absent not in owner.status_command
        refusal = iw.live_owner_refusal(PLAN, owner)
        assert "last heartbeat never" in refusal and "Abort it" not in refusal


# ---------------------------------------------------------------------------
# 8. --interactive is refused by the composite
# ---------------------------------------------------------------------------

REFUSAL = {
    PLAN: ("ERROR: improve-plan runs unattended and does not take --interactive "
           "(-i). Run --review-plan --interactive, then --apply-suggestions."),
    CODE: ("ERROR: improve-code runs unattended and does not take --interactive "
           "(-i). Run --review-code --interactive, then --apply-code-fixes."),
}


class TestInteractiveIsRefused:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("spelling", ["--interactive", "-i", "--interac"])
    @pytest.mark.parametrize("beside", [[], ["--agent-decide"], ["--claude-decide"],
                                        ["--let-claude-decide"], ["--no-confirm"]],
                             ids=["alone", "agent-decide", "claude-decide",
                                  "let-claude-decide", "no-confirm"])
    def test_main_refuses_every_spelling(self, tmp_path, capsys, mode, spelling, beside):
        plan = _plan(tmp_path)
        assert _main(mode, plan, spelling, *beside) == 3
        assert capsys.readouterr().err.strip().splitlines() == [REFUSAL[mode]]
        assert not _out(plan).exists(), "a refusal writes nothing"

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("spelling", ["--interactive", "-i"])
    def test_the_dispatchers_mode_route_lets_it_through_to_the_refusal(
        self, tmp_path, capsys, mode, spelling
    ):
        plan = _plan(tmp_path)
        assert multi_llm.main([mode, str(plan), spelling]) == 3
        assert REFUSAL[mode] in capsys.readouterr().err.splitlines()
        assert not _out(plan).exists()

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("beside, says", [
        ("--yes", "--yes (or --non-interactive) and --interactive are contradictory"),
        ("--detach", "--detach and --interactive are contradictory"),
    ], ids=["yes", "detach"])
    def test_a_typed_yes_or_detach_meets_the_dispatchers_conflict_first(
        self, tmp_path, capsys, monkeypatch, mode, beside, says
    ):
        monkeypatch.setattr(iw, "entry_point",
                            lambda *a, **k: pytest.fail("the entry point was called"))
        plan = _plan(tmp_path)
        assert multi_llm.main([mode, str(plan), "--interactive", beside]) == 3
        err = capsys.readouterr().err
        assert says in err and REFUSAL[mode] not in err

    def test_the_parser_still_needs_the_plan(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            ipo.parse_args(["--interactive"])


# ---------------------------------------------------------------------------
# 9. A step that does not finish (row 14), and the continuation
# ---------------------------------------------------------------------------

class TestTheContinuation:
    @pytest.mark.parametrize("mode", MODES)
    def test_with_no_selector_the_row_14_command_is_pinned(
        self, repo, stub, capsys, mode
    ):
        stub(fail={mode_table.resolve(mode).sequence[0]: 1})
        plan = _plan(repo)
        ctx = cli_common.activate_result(cli_common.ResultContext(mode=mode))
        try:
            assert _main(mode, plan) == 3
        finally:
            cli_common.deactivate_result()
        expected = cli_common.dispatcher_command(mode, plan, "--resume")
        lines = [line.strip() for line in capsys.readouterr().err.splitlines()]
        assert expected in lines
        assert "--yes" not in expected
        assert ctx.next == expected and ctx.mode == mode and ctx.state == "needs-human"

    @pytest.mark.parametrize("mode", MODES)
    def test_the_stop_is_the_result_lines_error(self, repo, stub, capsys, mode):
        """Fable review 2026-10-09 (inherited from `full`): the stop is the
        result line's ``error``, which a detached close-out writes as
        ``run.error`` and both tool routes relay as the run's last line."""
        apply = mode_table.resolve(mode).sequence[1]
        stub(fail={apply: 1})
        plan = _plan(repo)
        ctx = cli_common.activate_result(cli_common.ResultContext(mode=mode))
        try:
            assert _main(mode, plan) == 3
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert ctx.error == (
            f"Step 2: {apply} did not complete (exited 1). The workflow "
            "stopped here; no later phase ran."
        )

    def test_the_forced_rounds_continuation_keeps_selectors_and_drops_force(
        self, repo, stub, capsys
    ):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0), "apply-code-fixes": T(5)})
        typed = ["--force", "--models", "a,b", "--base-ref", "origin/release"]
        expected_flags = ["--models", "a,b", "--base-ref", "origin/release", "--resume"]
        for stopped_in in ("review-code", "apply-code-fixes"):
            stub(fail={stopped_in: 1})
            assert _main(CODE, plan, *typed) == 3
            err = capsys.readouterr().err
            assert cli_common.dispatcher_command(CODE, plan, *expected_flags) in err
        assert iw.continuation_flags(CODE, _args(CODE, plan, *typed)) == expected_flags

    def test_running_the_continuation_after_a_stop_in_the_review(
        self, repo, stub, capsys
    ):
        plan = _plan(repo)
        # The forced round cleared the review's completion and stopped in it.
        _write_state(plan, completed={"apply-code-fixes": T(5)})
        calls = stub()
        assert _main(CODE, plan, "--models", "a,b", "--base-ref", "origin/release",
                     "--resume") == 0
        assert _argv_of(calls, "review-code") == [
            "--yes", "--models", "a,b", "--base-ref", "origin/release"]
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS + [
            "--base-ref", "origin/release"], "a new cycle: no --resume"

    def test_running_the_continuation_after_a_stop_in_the_apply(
        self, repo, stub, capsys
    ):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0)},
                     started={"apply-code-fixes": T(3)})
        calls = stub()
        assert _main(CODE, plan, "--models", "a,b", "--base-ref", "origin/release",
                     "--resume") == 0
        assert [w for w, _a, _h in calls] == ["apply-code-fixes"]
        assert _argv_of(calls, "apply-code-fixes") == APPLY_FLAGS + [
            "--resume", "--base-ref", "origin/release"]

    def test_the_row_14_command_is_token_safe(self, repo, stub, capsys):
        names = ["a b.py", "it's.py", 'say "hi".py']
        listing = repo / LISTING_NAME
        listing.write_text("c.py\n", encoding="utf-8")
        stub(fail={"review-code": 1})
        plan = _plan(repo, name="spaced plan")
        argv = []
        for name in names:
            argv += ["--files", name]
        argv += ["--files-from", str(listing)]
        assert _main(CODE, plan, *argv) == 3
        command = next(
            line.strip() for line in capsys.readouterr().err.splitlines()
            if line.strip().startswith("uv run") and "improve-code" in line
        )
        tokens = shlex.split(command)
        after_plan = tokens[tokens.index(str(plan)) + 1:]
        assert after_plan == argv + ["--resume"]

    def test_recorded_continuation_parses_a_run_records_flags(self):
        assert iw.recorded_continuation(PLAN, None) == ["--resume"]
        assert iw.recorded_continuation(PLAN, []) == ["--resume"]
        assert iw.recorded_continuation(CODE, [
            "--force", "--models", "a", "b", "--detach", "--host", "codex",
            "--yes", "--exec-subagents", "--files", "x.py", "--base-ref", "r",
        ]) == ["--models", "a", "b", "--files", "x.py", "--base-ref", "r", "--resume"]
        with pytest.raises(iw.ContinuationError):
            iw.recorded_continuation(PLAN, ["--files", "x.py"])


# ---------------------------------------------------------------------------
# 10. Identity, --detach, and an unreadable state.json
# ---------------------------------------------------------------------------

class TestIdentity:
    @pytest.mark.parametrize("mode", MODES)
    def test_the_composites_own_record_never_full_workflow(
        self, repo, stub, capsys, mode
    ):
        stub(fail={mode_table.resolve(mode).sequence[1]: 1})
        plan = _plan(repo)
        assert _main(mode, plan) == 3
        state = StateManager(plan)
        key = mode.replace("-", "_")
        assert state.get(key)["failed_phase"] == APPLY_PHASE[mode]
        assert state.get("full_workflow") is None
        workflow = iw.ImproveWorkflow(_args(mode, plan), mode)
        assert (workflow.MODE_NAME, workflow.RUN_DIR_PHASE, workflow.STATE_KEY) == (
            mode, mode, key)
        assert workflow.run_dir == _out(plan) / mode

    @pytest.mark.parametrize("mode", MODES)
    def test_detach_claims_the_composites_run_dir(self, repo, monkeypatch, mode):
        seen = {}

        def boundary(**kwargs):
            seen.update(kwargs)
            return 0

        monkeypatch.setattr(rob, "maybe_detach", boundary)
        plan = _plan(repo)
        assert _main(mode, plan, "--detach") == 0
        assert seen["mode"] == mode and seen["phase"] == mode
        assert seen["phase_dir"] == str(_out(plan) / mode)
        assert not (_out(plan) / "full").exists()

    @pytest.mark.parametrize("mode", MODES)
    def test_detach_on_a_finished_plan_exits_2_before_the_boundary(
        self, repo, capsys, no_boundary, mode
    ):
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0),
                                      APPLY_PHASE[mode]: T(5)})
        ctx = cli_common.activate_result(cli_common.ResultContext(mode=mode))
        try:
            assert _main(mode, plan, "--detach") == 2
        finally:
            cli_common.deactivate_result()
        assert "[ALREADY_COMPLETE]" in capsys.readouterr().out
        assert not (_out(plan) / mode).exists(), "no run dir, no run record"
        assert ctx.next is None

    @pytest.mark.parametrize("mode", MODES)
    def test_detach_on_a_finished_plan_carries_the_next_step(
        self, repo, capsys, monkeypatch, no_boundary, mode
    ):
        """Through the real dispatcher, as a tool route re-runs a finished
        composite (it always appends `--detach`): exit 2 in the foreground with
        `[ALREADY_COMPLETE]`, no worker, no `next`, and `user_next_step` on the
        result line, answered as the phase the composite ends on (§2.6)."""
        plan = _plan(repo)
        _write_state(plan, completed={REVIEW_PHASE[mode]: T(0),
                                      APPLY_PHASE[mode]: T(5)})
        for name in ("MULTI_LLM_HOST", mode_table.ROUTE_ENV,
                     detach.PLAN_ARG_ENV, detach.WORKER_ENV):
            monkeypatch.delenv(name, raising=False)
        typed = "plans/imp-plan.md"
        assert multi_llm.main([mode, typed, "--detach"]) == 2
        out = capsys.readouterr().out
        result = cli_common.parse_result_line(out)
        assert "[ALREADY_COMPLETE]" in out
        assert result["state"] == "nothing-to-do" and result["next"] is None
        step = result["user_next_step"]
        if mode == PLAN:
            assert step["command"] == f"/multi-llm:multi-llm --generate-tasks {typed}"
        else:
            assert step["complete"] is True
        assert not (_out(plan) / mode).exists(), "no run dir, no run record"

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("route", ["entry", "dispatcher-detach"])
    def test_a_finished_run_writes_nothing_even_over_legacy_spellings(
        self, repo, capsys, monkeypatch, no_boundary, mode, route
    ):
        """Fable review 2026-10-09: the exit-2 report read ``tracked_files``
        through a writing ``StateManager``, whose load writes the judge-name
        upgrade back, so a finished plan's ``state.json`` changed (and a
        ``.state.lock`` appeared) on a run that promises to write nothing
        (§2.3, §2.4)."""
        plan = _plan(repo)
        apply_phase = APPLY_PHASE[mode]
        legacy = {"g1": {"decision": "approved",
                         "batch_context": {"decision_source": "claude_auto_decide"}}}
        state_file = _write_state(
            plan, completed={REVIEW_PHASE[mode]: T(0), apply_phase: T(5)},
            extra={
                f"human_decisions_{APPLY_DIR[mode]}": legacy,
                "tracked_files": [{"path": "src/app.py", "action": "modified",
                                   "task_id": "apply-code-fixes"}],
            },
        )
        # `_write_state` saves through a StateManager, which upgrades on save:
        # put the legacy spelling back on disk as an old run left it.
        text = state_file.read_text(encoding="utf-8").replace(
            '"judge_auto_decide"', '"claude_auto_decide"')
        assert '"claude_auto_decide"' in text
        state_file.write_text(text, encoding="utf-8")

        def tree():
            return {
                path.relative_to(repo): (path.read_bytes() if path.is_file() else None)
                for path in repo.rglob("*") if ".git" not in path.relative_to(repo).parts
            }

        before = tree()
        if route == "entry":
            assert _main(mode, plan) == 2
        else:
            for name in ("MULTI_LLM_HOST", mode_table.ROUTE_ENV,
                         detach.PLAN_ARG_ENV, detach.WORKER_ENV):
                monkeypatch.delenv(name, raising=False)
            assert multi_llm.main([mode, "plans/imp-plan.md", "--detach"]) == 2
        assert "[ALREADY_COMPLETE]" in capsys.readouterr().out
        assert tree() == before, "the exit-2 run wrote nothing, legacy spelling kept"

    def test_the_report_names_the_files_the_mode_lists(self, repo, stub, capsys):
        stub()
        plan = _plan(repo)
        (_out(plan) / "review-plan").mkdir(parents=True)
        (_out(plan) / "review-plan" / "report.html").write_text("x")
        (_out(plan) / "apply-suggestions").mkdir()
        (_out(plan) / "apply-suggestions" / "summary.md").write_text("x")
        assert _main(PLAN, plan) == 0
        out = capsys.readouterr().out
        assert "## Improve Plan Summary" in out
        files = out.split("### Output files", 1)[1]
        for path in (_out(plan) / "review-plan" / "report.html",
                     _out(plan) / "apply-suggestions" / "summary.md", plan):
            assert f"- {path}" in files


STATE_FAULTS = {
    "not-json": ("{\"phases_completed\": {,}", "Expecting property name"),
    "top-level-list": ("[1, 2]", "the top-level value is a list, not an object"),
    "phases-completed-list": ('{"phases_completed": ["review-plan"]}',
                              "phases_completed is a list, not an object"),
    # Fable review 2026-10-09: every other object the records read is checked
    # too. A `task_status` list passed and then raised in
    # `apply_phase_in_progress` (`.items()`), a traceback at exit 1.
    "task-status-list": ('{"phases_completed": {}, "task_status": ["T001"]}',
                         "task_status is a list, not an object"),
    "task-status-null": ('{"task_status": null}', "task_status is null, not an object"),
    "phases-started-string": ('{"phases_started": "review-plan"}',
                              "phases_started is a string, not an object"),
    "metrics-progress-list": ('{"metrics_progress": []}',
                              "metrics_progress is a list, not an object"),
    "metrics-number": ('{"metrics": 7}', "metrics is a number, not an object"),
}


class TestAnUnreadableStateFile:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("fault", list(STATE_FAULTS))
    @pytest.mark.parametrize("detach_flag", [[], ["--detach"]], ids=["foreground", "detach"])
    def test_it_is_refused_and_left_as_it_was(
        self, repo, stub, capsys, no_boundary, mode, fault, detach_flag
    ):
        calls = stub()
        plan = _plan(repo)
        state_file = _out(plan) / "state.json"
        state_file.parent.mkdir(parents=True)
        text, detail = STATE_FAULTS[fault]
        state_file.write_text(text, encoding="utf-8")
        assert _main(mode, plan, *detach_flag) == 3
        err = capsys.readouterr().err
        line = next(l for l in err.splitlines() if l.startswith("ERROR:"))
        assert line.startswith(f"ERROR: {mode}: cannot read ")
        assert "state.json (" in line and detail in line
        assert line.endswith(
            "; nothing was run. Repair the file, or move it aside to start the "
            "plan's records over, then run the command again.")
        assert "Traceback" not in err
        assert state_file.read_text(encoding="utf-8") == text
        assert calls == [] and not (_out(plan) / mode).exists()

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("shape", ["no-phase-keys", "missing"])
    def test_missing_keys_and_a_missing_file_are_no_records(
        self, repo, stub, capsys, mode, shape
    ):
        calls = stub()
        plan = _plan(repo)
        if shape == "no-phase-keys":
            (_out(plan)).mkdir(parents=True)
            (_out(plan) / "state.json").write_text('{"tracked_files": []}')
        assert _records(mode, plan) == iw.Records(
            mode=mode,
            consolidation_skip="not recommended by review-plan" if mode == PLAN else None,
        )
        assert _main(mode, plan) == 0
        assert [w for w, _a, _h in calls] == list(mode_table.resolve(mode).sequence)


# ---------------------------------------------------------------------------
# 11. One rule, both legs
# ---------------------------------------------------------------------------

class TestOneRule:
    def test_the_driver_and_the_planner_ask_the_same_skip_reason(
        self, repo, stub, capsys, monkeypatch
    ):
        plan = _plan(repo)
        _write_state(plan, completed={"review-plan": T(0), "apply-suggestions": T(5)})
        args = _args(PLAN, plan)
        records = iw.read_records(PLAN, plan, args)
        assert [p.decision for p in iw.plan_steps(PLAN, args, records)] == ["skip", "skip"]
        monkeypatch.setattr(iw, "skip_reason", lambda *a, **k: None)
        assert [p.decision for p in iw.plan_steps(PLAN, args, records)] == [
            "run", "after-review", "run"]
        calls = stub()
        assert _main(PLAN, plan) == 0
        assert [w for w, _a, _h in calls] == ["review-plan", "apply-suggestions"]

    def test_the_hook_maps_a_steps_phase_word_to_its_sequence_word(self, repo):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0)})
        workflow = iw.ImproveWorkflow(_args(CODE, plan), CODE)
        steps = workflow.steps()
        assert [s.phase for s in steps] == ["code-review", "apply-code-fixes"]
        assert [workflow.skip_reason(s) for s in steps] == ["completed", None]

    def test_step_argv_is_step_flags_then_the_tail(self, repo):
        plan = _plan(repo)
        args = _args(PLAN, plan, "--host", "codex", "--exec-subagents")
        workflow = iw.ImproveWorkflow(args, PLAN)
        assert workflow.step_argv("consolidate") == [
            "--consolidate", "--yes", "--host", "codex", "--exec-subagents"]
        assert iw.cross_cutting_flags(args, include_host=False) == ["--exec-subagents"]
        with pytest.raises(ValueError):
            workflow.step_argv("implement")
