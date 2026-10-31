"""``--scope-policy`` through the composites and ``full``: resolved once, one value per step.

plans/scope-policy-and-report-tags.md §4: a composite or ``full`` resolves the
policy once and hands that one value to every step as ``--scope-policy
<value>`` (the argv each phase receives, never a reconstructed string). Only
a user's choice is forwarded: the typed flag, or the live choice an earlier
run of the same composite recorded (``state.json`` → ``scope_policy_runs``).
With no choice no step gets the flag and each resolves from its own review
round, so omission and an explicit ``expand`` differ exactly there. An
explicit choice survives a later resume that omits it, on both legs.

The executor-leg fixtures are ``tests/test_improve_orchestrator.py``'s and
``tests/test_full_workflow_orchestrator.py``'s (the stubbed ``run_mode`` logs
each phase's argv and records the completion a real phase would).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import full_workflow_orchestrator as fwo  # noqa: E402
import generate_tasks_orchestrator as gto  # noqa: E402
import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import improve_steps  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils import scope_policy as sp  # noqa: E402
from utils import scope_policy_runs as runs  # noqa: E402
from utils import unattended as un  # noqa: E402
from utils.cli_common import RESULT_MARKER  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

from tests.test_improve_orchestrator import (  # noqa: E402,F401
    APPLY_FLAGS,
    CODE,
    MODES,
    PLAN,
    T,
    _args,
    _argv_of,
    _main,
    _plan,
    _write_state,
    repo,
    stub,
)
from tests.test_full_workflow_orchestrator import (  # noqa: E402,F401
    _plan as _full_plan,
    stub_phases,
)

PRESERVE = ["--scope-policy", "preserve"]
EXPAND = ["--scope-policy", "expand"]


def _state_doc(plan: Path) -> dict:
    return json.loads((output_dir_path(plan) / "state.json").read_text(encoding="utf-8"))


def _run_record(plan: Path, key: str):
    return runs.run_record(_state_doc(plan), key)


def _count(argv, value):
    return sum(1 for i, t in enumerate(argv[:-1])
               if t == "--scope-policy" and argv[i + 1] == value)


# ---------------------------------------------------------------------------
# 1. The improve composites, executor leg
# ---------------------------------------------------------------------------

class TestImproveExecutorLeg:
    @pytest.mark.parametrize("mode", MODES)
    def test_a_typed_policy_reaches_both_steps_once(self, repo, stub, capsys, mode):
        calls = stub()
        plan = _plan(repo)
        assert _main(mode, plan, *PRESERVE) == 0
        review, apply = mode_table.resolve(mode).sequence
        assert _argv_of(calls, review) == ["--yes", *PRESERVE]
        assert _argv_of(calls, apply) == [*APPLY_FLAGS, *PRESERVE]

    @pytest.mark.parametrize("mode", MODES)
    def test_omission_forwards_nothing(self, repo, stub, capsys, mode):
        """Each step resolves from its own round: a new round's default
        expand, the apply's inherited review policy. Exactly today's argv."""
        calls = stub()
        plan = _plan(repo)
        assert _main(mode, plan) == 0
        review, apply = mode_table.resolve(mode).sequence
        assert _argv_of(calls, review) == ["--yes"]
        assert _argv_of(calls, apply) == APPLY_FLAGS
        assert _run_record(plan, mode) is None

    @pytest.mark.parametrize("mode", MODES)
    def test_explicit_expand_is_forwarded_unlike_omission(self, repo, stub, capsys, mode):
        calls = stub()
        plan = _plan(repo)
        assert _main(mode, plan, *EXPAND) == 0
        for _word, argv, _held in calls:
            assert _count(argv, "expand") == 1, argv

    @pytest.mark.parametrize("mode", MODES)
    def test_a_finished_run_closes_its_record(self, repo, stub, capsys, mode):
        stub()
        plan = _plan(repo)
        assert _main(mode, plan, *PRESERVE) == 0
        record = _run_record(plan, mode)
        assert record["policy"] == "preserve" and record["source"] == "explicit"
        assert record.get("closed_at")
        assert runs.live_run_choice(_state_doc(plan), mode) is None

    @pytest.mark.parametrize("mode", MODES)
    def test_the_override_survives_a_resume_that_omits_it(self, repo, stub, capsys, mode):
        calls_log = stub(fail={mode_table.resolve(mode).sequence[1]: 3})
        plan = _plan(repo)
        review, apply = mode_table.resolve(mode).sequence
        assert _main(mode, plan, *PRESERVE) == 3
        err = capsys.readouterr().err
        # The row-14 command carries the typed policy with the other selectors.
        assert f"{mode} " in err and "--scope-policy preserve --resume" in err
        assert runs.live_run_choice(_state_doc(plan), mode) == "preserve"

        calls_log.clear()
        calls = stub()
        assert _main(mode, plan, "--resume") == 0
        out = capsys.readouterr().out
        assert [word for word, _a, _h in calls] == [apply]
        assert _count(_argv_of(calls, apply), "preserve") == 1
        assert (f"[AUTO_RESOLVED] scope-policy: preserve (kept from this {mode} run's "
                "recorded --scope-policy)") in out

    @pytest.mark.parametrize("mode", MODES)
    def test_a_forced_new_round_starts_without_the_old_choice(self, repo, stub, capsys, mode):
        stub(fail={mode_table.resolve(mode).sequence[1]: 3})
        plan = _plan(repo)
        assert _main(mode, plan, *PRESERVE) == 3
        calls = stub(fail={mode_table.resolve(mode).sequence[1]: 3})
        calls.clear()
        assert _main(mode, plan, "--force") == 3
        for _word, argv, _held in calls:
            assert "--scope-policy" not in argv, argv
        calls = stub()
        assert _main(mode, plan, "--resume") == 0
        for _word, argv, _held in calls:
            assert "--scope-policy" not in argv, argv

    @pytest.mark.parametrize("mode", MODES)
    def test_a_typed_value_replaces_the_recorded_one(self, repo, stub, capsys, mode):
        stub(fail={mode_table.resolve(mode).sequence[1]: 3})
        plan = _plan(repo)
        assert _main(mode, plan, *PRESERVE) == 3
        calls = stub()
        calls.clear()
        assert _main(mode, plan, *EXPAND, "--resume") == 0
        for _word, argv, _held in calls:
            assert _count(argv, "expand") == 1 and "preserve" not in argv

    def test_an_explicit_policy_releases_a_completed_applys_deferrals(self, repo, stub, capsys):
        """plan §4: a completed bounded apply that still holds scope deferrals
        is offered them again under a changed policy, never \"already complete\"."""
        calls = stub()
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0), "apply-code-fixes": T(5)},
                     extra={sp.STATE_KEY: {"code-review": {
                         "review_policy": "preserve", "review_policy_source": "explicit",
                         "recorded_at": T(0),
                         "deferrals": [{"group_id": "abcd", "relation": "scope_extension"}],
                     }}})
        assert _main(CODE, plan) == 2
        assert "[ALREADY_COMPLETE]" in capsys.readouterr().out
        assert _main(CODE, plan, *PRESERVE) == 2, "the same policy releases nothing"
        capsys.readouterr()
        assert _main(CODE, plan, *EXPAND) == 0
        assert [word for word, _a, _h in calls] == ["apply-code-fixes"]
        assert _count(_argv_of(calls, "apply-code-fixes"), "expand") == 1

    def test_a_released_override_survives_a_run_that_omits_it(self, repo, stub, capsys):
        """plan §4: "The same override must survive subsequent omission" — an
        `expand` release that stopped is offered again by the next run without
        the flag (the composite's live recorded choice), not "already complete"."""
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0), "apply-code-fixes": T(5)},
                     extra={sp.STATE_KEY: {"code-review": {
                         "review_policy": "preserve", "review_policy_source": "explicit",
                         "recorded_at": T(0), "apply_policy": "preserve",
                         "apply_policy_source": "explicit",
                         "deferrals": [{"group_id": "abcd", "relation": "scope_extension"}],
                     }}})
        stub(fail={"apply-code-fixes": 3})
        assert _main(CODE, plan, *EXPAND) == 3
        # The stub never clears the deferrals the way a real `expand` apply
        # does, so each later run stops again until the last one succeeds.
        for extra, fail, code in (([], {"apply-code-fixes": 3}, 3), (["--resume"], {}, 0)):
            calls = stub(fail=fail)
            calls.clear()
            capsys.readouterr()
            assert _main(CODE, plan, *extra) == code, extra
            assert [word for word, _a, _h in calls] == ["apply-code-fixes"], extra
            assert _count(_argv_of(calls, "apply-code-fixes"), "expand") == 1

    def test_expand_offers_a_blocked_apply_again(self, repo, stub, capsys):
        """A completed `preserve` apply that stopped on a scope blocker is not
        done for `expand`, which is the recovery the blocker message names."""
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0), "apply-code-fixes": T(5)},
                     extra={sp.STATE_KEY: {"code-review": {
                         "review_policy": "preserve", "review_policy_source": "explicit",
                         "recorded_at": T(0), "apply_policy": "preserve",
                         "apply_policy_source": "explicit", "deferrals": [],
                         "blockers": [{"group_id": "g1", "task_id": "apply-fix-batch-1",
                                       "reason": "needs the new API"}],
                         "completion": {"outcome": "blocked", "policy": "preserve",
                                        "deferred": 0, "blockers": 1, "at": T(5)},
                     }}})
        calls = stub()
        assert _main(CODE, plan, *PRESERVE) == 2, "preserve would only block again"
        capsys.readouterr()
        assert _main(CODE, plan, *EXPAND) == 0
        assert [word for word, _a, _h in calls] == ["apply-code-fixes"]

    def test_the_consolidation_pass_gets_it_only_with_its_review(self, tmp_path):
        plan = _plan(tmp_path)
        args = _args(PLAN, plan, *PRESERVE)
        fresh = iw.Records(mode=PLAN)
        assert iw.step_flags(PLAN, iw.CONSOLIDATE, args, fresh) == [
            "--consolidate", "--yes", *PRESERVE]
        reviewed = iw.Records(mode=PLAN, review_completed=True, review_on_record=True)
        assert iw.step_flags(PLAN, iw.CONSOLIDATE, args, reviewed) == ["--consolidate", "--yes"]

    @pytest.mark.parametrize("mode", MODES)
    def test_a_recorded_continuation_reparses_it(self, mode):
        """A detached run's ``replay_flags`` give the same continuation."""
        assert iw.recorded_continuation(mode, ["--scope-policy", "preserve", "--detach"]) == [
            *PRESERVE, "--resume"]
        assert iw.recorded_continuation(mode, ["--detach"]) == ["--resume"]

    @pytest.mark.parametrize("mode", MODES)
    def test_the_tool_routes_resume_array_keeps_it(self, tmp_path, mode):
        """``util next-command --blocked <composite>``'s ``user_resume.flags``
        (the array both tool routes build their option calls from) is the
        composite's run record replayed through its own parser."""
        from utils import next_phase, status_file
        from utils.cli_common import replay_flags

        plan = _plan(tmp_path)
        run_dir = output_dir_path(plan) / mode_table.resolve(mode).phase
        run_dir.mkdir(parents=True)
        status_file.write_status(str(run_dir), {"phase": mode})
        launched = ["--plan-file", str(plan), "--detach", *PRESERVE, "--quick"]
        status_file.claim_run(str(run_dir), {
            "run_id": "r-1", "state": "failed", "exit_code": 3, "detached": True,
            "mode": mode, "replay_flags": replay_flags(mode, launched)})
        assert next_phase._composite_continuation(mode, plan) == [
            "--quick", *PRESERVE, "--resume"]


# ---------------------------------------------------------------------------
# 2. util improve-steps, the native leg
# ---------------------------------------------------------------------------

def _planner(repo_root: Path, mode: str, plan: Path, *argv: str, capsys) -> dict:
    code = multi_llm.main(["util", "improve-steps", str(plan.relative_to(repo_root)),
                           "--mode", mode, "--json", *argv])
    out = capsys.readouterr().out
    assert code == 0, out
    return json.loads(out.rsplit(RESULT_MARKER, 1)[0])


class TestImproveSteps:
    @pytest.mark.parametrize("mode", MODES)
    def test_every_step_and_the_rerun_carry_one_value(self, repo, capsys, mode):
        plan = _plan(repo)
        payload = _planner(repo, mode, plan, *PRESERVE, capsys=capsys)
        assert payload["steps"]
        for step in payload["steps"]:
            if step["step"] == iw.CONSOLIDATE and step["decision"] == "skip":
                continue
            assert _count(step["argv"], "preserve") == 1, step
        assert payload["rerun_argv"][-3:] == [*PRESERVE, "--resume"]
        assert "--scope-policy preserve --resume" in payload["rerun"]

    @pytest.mark.parametrize("mode", MODES)
    def test_omission_plans_no_flag(self, repo, capsys, mode):
        plan = _plan(repo)
        payload = _planner(repo, mode, plan, capsys=capsys)
        for step in payload["steps"]:
            assert "--scope-policy" not in step["argv"], step
        assert "--scope-policy" not in payload["rerun_argv"]

    def test_it_forwards_a_live_executor_record_and_writes_none(self, repo, capsys):
        plan = _plan(repo)
        _write_state(plan, completed={"code-review": T(0)})
        state = StateManager(plan)
        runs.record_run_choice(state.state, CODE, "preserve")
        state.save()
        before = (output_dir_path(plan) / "state.json").read_bytes()
        payload = _planner(repo, CODE, plan, capsys=capsys)
        apply = next(s for s in payload["steps"] if s["step"] == "apply-code-fixes")
        assert apply["decision"] == "run"
        assert _count(apply["argv"], "preserve") == 1
        assert (output_dir_path(plan) / "state.json").read_bytes() == before, "read-only"

    def test_the_planner_and_the_executor_agree(self, repo):
        plan = _plan(repo)
        args = improve_steps.parse_args([str(plan), "--mode", CODE, *PRESERVE])
        records = iw.read_records(CODE, plan, args)
        planned = {s.step: list(s.flags) for s in iw.plan_steps(CODE, args, records)}
        executor = iw.ImproveWorkflow(_args(CODE, plan, *PRESERVE), CODE, records=records)
        for word, flags in planned.items():
            assert flags == iw.step_flags(CODE, word, executor.args, records)
            assert executor.step_argv(word)[:len(flags)] == flags


# ---------------------------------------------------------------------------
# 3. full: common_flags, the prerequisite apply runner, generate-tasks
# ---------------------------------------------------------------------------

@pytest.fixture
def generation(monkeypatch):
    """Stub generate-tasks (``full`` calls it as a function, not through
    ``run_mode``): record the child namespace's policy and the completion."""
    seen = []

    def run_generate_tasks(child, *, in_workflow, apply_runner):
        seen.append(getattr(child, "scope_policy", "<absent>"))
        manager = StateManager(Path(child.plan_file).resolve())
        manager.mark_phase_completed("generate-tasks")
        manager.save()
        return 0

    monkeypatch.setattr(gto, "run_generate_tasks", run_generate_tasks)
    return seen


def _full(plan: Path, *argv: str) -> int:
    return fwo.main(["--plan-file", str(plan), "--yes", *argv])


class TestFullWorkflow:
    @pytest.mark.parametrize("flags", [[], EXPAND])
    def test_a_forced_dry_run_keeps_the_previous_policy_record(self, tmp_path, flags):
        plan = _full_plan(tmp_path)
        state = StateManager(plan)
        runs.record_run_choice(state.state, "full", "preserve")
        state.save()
        before = _run_record(plan, "full")
        workflow = fwo.FullWorkflow(fwo.parse_args(
            ["--plan-file", str(plan), "--force", "--dry-run-subagents", *flags]))
        workflow._record_scope_policy()
        assert _run_record(plan, "full") == before

    def test_every_phase_gets_the_one_typed_value(self, tmp_path, stub_phases, generation,
                                                  capsys):
        calls = stub_phases()
        plan = _full_plan(tmp_path)
        assert _full(plan, *PRESERVE) == 0
        assert calls, "the stub saw no phase"
        for mode, argv in calls:
            assert _count(argv, "preserve") == 1, (mode, argv)
        assert generation == ["preserve"]
        out = capsys.readouterr().out
        assert "generate-tasks" in out and "--scope-policy preserve" in out

    def test_omission_forwards_nothing(self, tmp_path, stub_phases, generation, capsys):
        calls = stub_phases()
        plan = _full_plan(tmp_path)
        assert _full(plan) == 0
        for mode, argv in calls:
            assert "--scope-policy" not in argv, (mode, argv)
        assert generation == [None]

    def test_the_prerequisite_apply_runner_carries_it(self, tmp_path, stub_phases, capsys):
        calls = stub_phases()
        plan = _full_plan(tmp_path)
        workflow = fwo.FullWorkflow(fwo.parse_args(
            ["--plan-file", str(plan), "--yes", *PRESERVE]))
        runner = workflow.apply_runner(un.row(5).site_id)
        assert runner("apply-suggestions") == 0
        [(mode, argv)] = calls
        assert mode == "apply-suggestions"
        assert argv[:2] == [un.AGENT_DECIDE_FLAG, "--no-confirm"]
        assert _count(argv, "preserve") == 1

    def test_common_flags_end_with_it_and_only_when_chosen(self, tmp_path):
        plan = _full_plan(tmp_path)
        typed = fwo.FullWorkflow(fwo.parse_args(["--plan-file", str(plan), *PRESERVE]))
        assert typed.common_flags()[-2:] == PRESERVE
        bare = fwo.FullWorkflow(fwo.parse_args(["--plan-file", str(plan)]))
        assert "--scope-policy" not in bare.common_flags()

    def test_the_override_survives_full_resume_and_closes_at_the_end(
        self, tmp_path, stub_phases, generation, capsys
    ):
        calls = stub_phases(fail={"implement": 3})
        plan = _full_plan(tmp_path)
        assert _full(plan, *PRESERVE) == 3
        assert runs.live_run_choice(_state_doc(plan), "full") == "preserve"
        calls.clear()
        calls = stub_phases()
        generation.clear()
        assert _full(plan, "--resume") == 0
        out = capsys.readouterr().out
        assert [mode for mode, _a in calls][0] == "implement"
        for mode, argv in calls:
            assert _count(argv, "preserve") == 1, (mode, argv)
        assert ("[AUTO_RESOLVED] scope-policy: preserve (kept from this full run's "
                "recorded --scope-policy)") in out
        assert runs.live_run_choice(_state_doc(plan), "full") is None
        assert _run_record(plan, "full").get("closed_at")

    def _seed_completed(self, plan, round_record):
        state = StateManager(plan)
        for phase in ("review-plan", "apply-suggestions", "generate-tasks", "review-tasks",
                      "implement", "code-review", "apply-code-fixes"):
            state.mark_phase_completed(phase)
        state.state[sp.STATE_KEY] = {"code-review": round_record}
        state.save()
        review_dir = plan.parent / plan.stem / "code-review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "grouped.json").write_text(json.dumps(
            [{"theme": "t", "suggestions": [{"title": "x", "desc": "y"}]}]))

    def test_a_released_override_survives_full_resume(
        self, tmp_path, stub_phases, generation, capsys
    ):
        plan = _full_plan(tmp_path)
        self._seed_completed(plan, {
            "review_policy": "preserve", "review_policy_source": "explicit",
            "recorded_at": T(0), "apply_policy": "preserve", "apply_policy_source": "explicit",
            "deferrals": [{"group_id": "abcd", "relation": "scope_extension"}]})
        calls = stub_phases(fail={"apply-code-fixes": 3})
        assert _full(plan, *EXPAND) == 3
        assert [mode for mode, _a in calls] == ["apply-code-fixes"]
        calls = stub_phases()
        calls.clear()
        _full(plan, "--resume")
        assert [mode for mode, _a in calls] == ["apply-code-fixes"], (
            "the live expand override re-offers the deferrals")

    def test_expand_offers_a_blocked_apply_again(
        self, tmp_path, stub_phases, generation, capsys
    ):
        plan = _full_plan(tmp_path)
        self._seed_completed(plan, {
            "review_policy": "preserve", "review_policy_source": "explicit",
            "recorded_at": T(0), "apply_policy": "preserve", "apply_policy_source": "explicit",
            "deferrals": [],
            "blockers": [{"group_id": "g1", "task_id": "apply-fix-batch-1",
                          "reason": "needs the new API"}],
            "completion": {"outcome": "blocked", "policy": "preserve", "deferred": 0,
                           "blockers": 1, "at": T(5)}})
        calls = stub_phases()
        _full(plan, *EXPAND)
        assert [mode for mode, _a in calls] == ["apply-code-fixes"]

    def test_force_is_a_new_run(self, tmp_path, stub_phases, generation, capsys):
        stub_phases(fail={"implement": 3})
        plan = _full_plan(tmp_path)
        assert _full(plan, *PRESERVE) == 3
        calls = stub_phases(fail={"implement": 3})
        calls.clear()
        assert _full(plan, "--force") == 3
        for mode, argv in calls:
            assert "--scope-policy" not in argv, (mode, argv)
        calls = stub_phases()
        assert _full(plan, "--resume") == 0
        for mode, argv in calls:
            assert "--scope-policy" not in argv, (mode, argv)

    def test_the_legacy_full_spelling_takes_it(self, tmp_path, capsys):
        """``--full`` is the user flag of mode ``full``; both reach one parser."""
        plan = _full_plan(tmp_path)
        args = fwo.parse_args(["--plan-file", str(plan), *PRESERVE])
        assert args.scope_policy == "preserve"
        assert mode_table.resolve("full").name == "full"
        assert multi_llm.main(["full", "missing.md", *PRESERVE]) == 1
        assert "Plan file not found" in capsys.readouterr().err
