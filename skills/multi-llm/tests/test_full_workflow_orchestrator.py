#!/usr/bin/env python3
"""`--full` as a resumable state machine (D1a/D1e), and Phase 3's closed loop.

The sequencing that used to live in ``instructions/full-workflow.md`` is Python
now, so these tests are about the *machine*: what order it runs, what it treats as
progress, where it resumes, and how it ends when a phase does not finish.

Most tests stub :meth:`FullWorkflow.run_mode` — the per-phase orchestrators have
their own suites, and re-running a fan-out here would test them a second time
while testing the loop not at all. The stub records the phase and (unless the test
wants a failure) records the completion a real phase would, which is exactly what
the loop reads.

The exception is **Phase 3**: it runs for real, because "task generation closes its
loop under `--full` too" is the Done-when row, and because Phase 3 is the one step
the workflow calls as a Python function rather than through the mode table — that is
where ``ensure_prerequisites``' ``apply_runner`` is injected (D6a rows 5/6).

Usage:
    uv run -- pytest tests/test_full_workflow_orchestrator.py -v
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
import full_workflow_orchestrator as fwo  # noqa: E402
import generate_tasks_orchestrator as gto  # noqa: E402
import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from harness import SkillRunner  # noqa: E402
from mocks.mock_llm import echo_subagent_answer  # noqa: E402
from utils import unattended as un  # noqa: E402
from utils.cli_common import parse_result_line, record_next  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_executor import simulate_native_invoker  # noqa: E402

PLAN_BODY = """# Full Workflow Plan

## Overview
Add a foundation module and wire it into the entry point.

## Step 1
Create `src/foundation.py`.

## Step 2
Call it from `src/main.py`.
"""

#: The phases the loop drives, in order, with the phase word each records.
EXPECTED_ORDER = [
    "review-plan",
    "apply-suggestions",
    "generate-tasks",
    "review-tasks",
    "apply-task-suggestions",
    "implement",
    "code-review",
    "apply-code-fixes",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan(tmp_path: Path, name: str = "wf-plan") -> Path:
    plan = tmp_path / f"{name}.md"
    plan.write_text(PLAN_BODY, encoding="utf-8")
    return plan


def _complete(plan: Path, *phases: str) -> None:
    state = StateManager(plan)
    for phase in phases:
        state.mark_phase_completed(phase)
    state.save()


def _phase_dir(plan: Path, phase: str) -> Path:
    return plan.parent / plan.stem / phase


def _tasks_md(plan: Path) -> Path:
    return plan.parent / plan.stem / "tasks" / "tasks.md"


def _valid_document() -> dict:
    return {
        "schema_version": "1.0",
        "plan_preamble": "Foundation first, then the wiring that consumes it.",
        "tasks": [
            {
                "id": "T001",
                "title": "Create the foundation module",
                "description": "Add src/foundation.py exporting the interface.",
                "depends_on": [],
                "files_to_create": ["src/foundation.py"],
                "files_to_modify": [],
                "acceptance_criteria": ["src/foundation.py exports the interface"],
                "estimated_complexity": "low",
                "executor_role": "agent",
                "subagent_type": None,
            },
            {
                "id": "T002",
                "title": "Wire it into the entry point",
                "description": "Call T001's interface from src/main.py.",
                "depends_on": ["T001"],
                "files_to_create": [],
                "files_to_modify": ["src/main.py"],
                "acceptance_criteria": ["src/main.py calls the interface"],
                "estimated_complexity": "medium",
                "executor_role": "agent",
                "subagent_type": None,
            },
        ],
    }


def _persist_generation(plan: Path, document) -> Path:
    """Play the native invoker for the materialized task-generation task."""
    phase_dir = _phase_dir(plan, "generate-tasks")
    task = json.loads(
        (phase_dir / gto.TASK_FILE_NAME).read_text(encoding="utf-8")
    )["tasks"][0]
    report = simulate_native_invoker(
        [task], phase_dir=str(phase_dir),
        respond=lambda _task: document, kind="task_generation",
    )
    assert report.ok, [(o.task_id, o.status, o.reason) for o in report.failed]
    return phase_dir / gto.RESULT_NAME


@pytest.fixture
def stub_phases(monkeypatch):
    """Stub every mode-dispatched phase; return the call log.

    ``configure(fail={...}, silent={...})`` shapes the stub:
    ``fail`` maps a mode word to the exit code it returns (nothing recorded), and
    ``silent`` names modes that return 0 *without* recording a completion — the
    native-invoker case a real fan-out produces when it stops at its own pending
    marker.
    """
    calls = []
    state = {"fail": {}, "silent": set()}

    def run_mode(self, mode_word, extra):
        calls.append((mode_word, list(extra)))
        if mode_word in state["fail"]:
            return state["fail"][mode_word]
        if mode_word in state["silent"]:
            record_next(f"<follow-up for {mode_word}>")
            return 0
        phase = mode_table.phase_for(mode_word) or mode_word
        manager = StateManager(self.plan_path)
        manager.mark_phase_completed(phase)
        manager.save()
        return 0

    monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)

    def configure(fail=None, silent=None):
        state["fail"] = dict(fail or {})
        state["silent"] = set(silent or ())
        return calls

    configure()
    return configure


def _modes_called(calls):
    return [mode for mode, _flags in calls]


def _driver_config(tmp_path: Path, skill_dir: Path, runner: str) -> Path:
    base = yaml.safe_load((skill_dir / "providers.yaml").read_text(encoding="utf-8"))
    base.setdefault("driver", {})["subagent_runner"] = runner
    path = tmp_path / "driver-providers.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def exec_env(tmp_path, mock_provider, monkeypatch):
    """The executor leg, in-process, against the mock provider binaries."""
    import utils.provider_registry as registry

    runner = SkillRunner(tmp_path / "harness", extra_env=mock_provider.get_env())
    env = runner._build_env()
    env["MULTI_LLM_PROVIDERS_CONFIG"] = str(
        _driver_config(tmp_path, runner.skill_dir, "codex:gpt-5.5")
    )
    env["MOCK_LLM_ECHO"] = "task_generation"
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    registry._config = None
    registry._config_key = None
    yield
    registry._config = None
    registry._config_key = None


# ---------------------------------------------------------------------------
# 1. The sequence
# ---------------------------------------------------------------------------

class TestSequence:
    def test_the_documented_order_is_the_executed_order(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        # Phase 3 is pre-recorded so this test is about the *order*, not about
        # task generation (which has its own tests below).
        _complete(plan, "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--yes"]) == 0
        capsys.readouterr()
        assert _modes_called(calls) == [
            "review-plan", "apply-suggestions", "review-tasks", "implement",
            "review-code",
        ], "the two conditional apply phases had nothing to apply"
        state = StateManager(plan)
        # Nothing is left pending: every phase is completed or *recorded* skipped.
        for phase in EXPECTED_ORDER:
            assert state.is_phase_completed(phase) or state.is_phase_skipped(phase), (
                phase
            )
        assert state.get(fwo.FullWorkflow.STATE_KEY)["current_phase"] == "complete"

    def test_the_fan_out_phases_share_one_model_resolution(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        assert multi_llm.main(
            ["full", str(plan), "--yes", "--models", "a:b", "c:d"]
        ) == 0
        capsys.readouterr()
        by_mode = dict(calls)
        for mode in ("review-plan", "review-tasks", "review-code"):
            assert by_mode[mode][:3] == ["--models", "a:b", "c:d"], mode
        assert "--models" not in by_mode["implement"]

    def test_apply_phases_get_the_unattended_answers(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--yes"]) == 0
        capsys.readouterr()
        flags = dict(calls)["apply-suggestions"]
        assert "--no-confirm" in flags and "--agent-decide" in flags
        assert "--claude-decide" not in flags
        assert "--yes" in flags

    def test_detach_and_internal_flags_are_never_forwarded(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--yes", "--exec-subagents"]) == 0
        capsys.readouterr()
        for mode, flags in calls:
            assert "--detach" not in flags, mode
            assert not any(f.startswith("--internal-") for f in flags), mode
            assert "--exec-subagents" in flags, mode

    def test_resume_skips_what_state_already_records(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "review-plan", "apply-suggestions", "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--yes", "--resume"]) == 0
        out = capsys.readouterr().out
        called = _modes_called(calls)
        assert "review-plan" not in called and "apply-suggestions" not in called
        assert called[0] == "review-tasks", called
        assert "already completed, skipping" in out
        # --resume reaches the phases that support it, so a partially processed
        # batch is not re-applied.
        assert "--resume" in dict(calls)["implement"]

    def test_force_re_runs_recorded_phases(self, tmp_path, stub_phases, capsys):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "review-plan", "apply-suggestions", "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--yes", "--force"]) == 0
        capsys.readouterr()
        called = _modes_called(calls)
        assert called[0] == "review-plan"
        assert "--force" not in dict(calls)["review-plan"], (
            "`full --force` re-runs phases; it never re-invokes the models"
        )


# ---------------------------------------------------------------------------
# 2. Row 14 — a failed phase is exit 3, never a question
# ---------------------------------------------------------------------------

class TestPhaseFailure:
    def test_a_nonzero_phase_is_exit_three_naming_resume(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases(fail={"apply-suggestions": 1})
        plan = _plan(tmp_path)
        code = multi_llm.main(["full", str(plan), "--yes"])
        captured = capsys.readouterr()
        assert code == 3
        assert "Phase 2: Apply Suggestions did not complete" in captured.err
        assert "--resume" in captured.err
        assert "full" in captured.err
        assert "never a question" in captured.err
        assert _modes_called(calls) == ["review-plan", "apply-suggestions"], (
            "no later phase ran"
        )
        result = parse_result_line(captured.out)
        assert result["exit"] == 3 and result["state"] == "needs-human"
        assert result["next"].endswith("--resume")
        assert StateManager(plan).get(fwo.FullWorkflow.STATE_KEY)["failed_phase"] == (
            "apply-suggestions"
        )

    def test_a_phase_that_recorded_nothing_is_not_progress(
        self, tmp_path, stub_phases, capsys
    ):
        """The native-invoker case: exit 0 at a pending marker is not a finished phase."""
        calls = stub_phases(silent={"review-plan"})
        plan = _plan(tmp_path)
        code = multi_llm.main(["full", str(plan), "--yes"])
        err = capsys.readouterr().err
        assert code == 3
        assert "state.json records no completion" in err
        assert "<follow-up for review-plan>" in err, (
            "the phase's own follow-up is handed back before the --resume line"
        )
        assert _modes_called(calls) == ["review-plan"]

    def test_the_stop_is_the_result_lines_error(self, tmp_path, stub_phases, capsys):
        """Fable review 2026-10-09: the stop printed its block to stderr and
        recorded no ``error``, so a detached close-out wrote no ``run.error``
        and both tool routes relayed no stop line."""
        stub_phases(fail={"apply-suggestions": 1})
        plan = _plan(tmp_path)
        assert multi_llm.main(["full", str(plan), "--yes"]) == 3
        result = parse_result_line(capsys.readouterr().out)
        assert result["error"] == (
            "Phase 2: Apply Suggestions did not complete (exited 1). The "
            "workflow stopped here; no later phase ran."
        )

    def test_a_long_detail_is_cut_and_the_stop_sentence_kept(
        self, tmp_path, stub_phases, capsys, monkeypatch
    ):
        from utils.cli_common import ERROR_MAX_CHARS

        stub_phases(fail={"apply-suggestions": 1})
        monkeypatch.setattr(
            fwo.FullWorkflow, "_verify",
            lambda self, phase, code: (fwo.FAILED, "word " * 600, code),
        )
        plan = _plan(tmp_path)
        assert multi_llm.main(["full", str(plan), "--yes"]) == 3
        error = parse_result_line(capsys.readouterr().out)["error"]
        assert len(error) <= ERROR_MAX_CHARS
        assert error.startswith("Phase 1: Review Plan did not complete (word word")
        assert error.endswith("…). The workflow stopped here; no later phase ran.")

    def test_a_phases_own_stop_line_is_kept(self, tmp_path, monkeypatch, capsys):
        """A phase that stopped with its own headline (an apply or implement
        stop names the remedy) keeps it: the workflow's line would only say
        `exited 3`."""
        from utils.cli_common import record_error

        def run_mode(self, mode_word, extra):
            if mode_word == "review-plan":
                record_error("review-plan stopped: batch 2 needs you (the remedy).")
                return 3
            pytest.fail(f"no phase after the stop: {mode_word}")

        monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)
        plan = _plan(tmp_path)
        assert multi_llm.main(["full", str(plan), "--yes"]) == 3
        result = parse_result_line(capsys.readouterr().out)
        assert result["error"] == "review-plan stopped: batch 2 needs you (the remedy)."

    def test_an_earlier_steps_error_line_is_not_the_stops(
        self, tmp_path, monkeypatch, capsys
    ):
        """An error a step recorded and then finished past is not the stop of
        the step that failed later."""
        from utils.cli_common import record_error

        def run_mode(self, mode_word, extra):
            phase = mode_table.phase_for(mode_word) or mode_word
            if mode_word == "apply-suggestions":
                return 1
            record_error(f"{mode_word}: a warning it recorded and finished past")
            _complete(self.plan_path, phase)
            return 0

        monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)
        plan = _plan(tmp_path)
        assert multi_llm.main(["full", str(plan), "--yes"]) == 3
        error = parse_result_line(capsys.readouterr().out)["error"]
        assert error.startswith("Phase 2: Apply Suggestions did not complete")

    def test_a_failed_run_with_no_error_recorded_prints_no_none(
        self, tmp_path, capsys
    ):
        """``status`` rendered a run record with no ``error`` (a worker that
        returned 1 with no crash and no stop line) as ``error:     None``."""
        from utils import status_file

        plan = _plan(tmp_path)
        run_dir = plan.parent / plan.stem / "full"
        run_dir.mkdir(parents=True)
        status_file.claim_run(str(run_dir), {
            "run_id": "run-full-0001", "state": "failed", "exit_code": 1,
            "error": None, "detached": True, "mode": "full",
        })
        assert multi_llm.main([
            "status", str(plan), "--view", "run", "--run-id", "run-full-0001",
        ]) in (1, 3)
        out = capsys.readouterr().out
        assert "exit_code: 1" in out
        assert "None" not in out.split("exit_code: 1", 1)[1].split("\n", 2)[1]
        assert "error:     None" not in out

    def test_attended_failure_is_also_exit_three(self, tmp_path, stub_phases, capsys):
        stub_phases(fail={"review-plan": 1})
        plan = _plan(tmp_path)
        assert multi_llm.main(["full", str(plan)]) == 3
        err = capsys.readouterr().err
        assert "--resume" in err
        assert "--yes --resume" not in err, (
            "the re-invocation echoes the run's own flags, not invented ones"
        )


# ---------------------------------------------------------------------------
# 3. Row 12 — review the tasks?
# ---------------------------------------------------------------------------

class TestReviewTasksContinuation:
    def test_yes_runs_it_automatically_and_records_the_resolution(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--yes"]) == 0
        out = capsys.readouterr().out
        assert "review-tasks" in _modes_called(calls)
        assert f"[AUTO_RESOLVED] {un.row(12).site_id}" in out

    def test_without_the_flag_the_phase_is_recorded_skipped(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        assert multi_llm.main(["full", str(plan)]) == 0
        capsys.readouterr()
        assert "review-tasks" not in _modes_called(calls)
        state = StateManager(plan)
        assert state.is_phase_skipped("review-tasks")
        assert "--review-tasks" in (
            state.get_phase_skip_reason("review-tasks") or ""
        )
        # (a) of the three-case conditional: the dependent phase is skipped too.
        assert state.is_phase_skipped("apply-task-suggestions")

    def test_the_explicit_flag_opts_in_without_yes(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--review-tasks"]) == 0
        capsys.readouterr()
        assert "review-tasks" in _modes_called(calls)


class TestConditionalApplyPhases:
    def test_findings_make_apply_task_suggestions_run(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        grouped = _phase_dir(plan, "review-tasks")
        grouped.mkdir(parents=True)
        (grouped / "grouped.json").write_text(json.dumps([
            {"group_id": "g1", "suggestions": [{"title": "x", "importance": "HIGH"}]},
        ]), encoding="utf-8")
        assert multi_llm.main(["full", str(plan), "--yes"]) == 0
        capsys.readouterr()
        assert "apply-task-suggestions" in _modes_called(calls)

    def test_code_review_findings_make_apply_code_fixes_run(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        grouped = _phase_dir(plan, "code-review")
        grouped.mkdir(parents=True)
        (grouped / "grouped.json").write_text(json.dumps({"groups": [
            {"group_id": "g1", "suggestions": [{"title": "y", "importance": "HIGH"}]},
        ]}), encoding="utf-8")
        assert multi_llm.main(["full", str(plan), "--yes"]) == 0
        capsys.readouterr()
        assert "apply-code-fixes" in _modes_called(calls)

    @pytest.mark.parametrize("review, apply, step, dirname, payload", [
        pytest.param(
            "review-tasks", "apply-task-suggestions", "step_apply_task_suggestions",
            "review-tasks",
            [{"group_id": "g1", "suggestions": [{"title": "x", "importance": "HIGH"}]}],
        ),
        ("code-review", "apply-code-fixes", "step_apply_code_fixes",
         "code-review",
         {"groups": [{"group_id": "g1",
                      "suggestions": [{"title": "y", "importance": "HIGH"}]}]}),
    ])
    def test_a_review_skipped_then_completed_does_not_skip_its_apply_phase(
        self, tmp_path, review, apply, step, dirname, payload
    ):
        """A completion wins over an earlier skip record, as in `status`.

        ``mark_phase_completed`` leaves the stale skip in place; `full` must
        read the effective status (``_recorded``), not the raw skip flag, or it
        skips the apply phase while `status` shows it pending.
        """
        plan = _plan(tmp_path)
        phase_dir = _phase_dir(plan, dirname)
        phase_dir.mkdir(parents=True)
        (phase_dir / "grouped.json").write_text(json.dumps(payload), encoding="utf-8")
        state = StateManager(plan)
        state.mark_phase_skipped(review, reason="earlier skip")
        state.mark_phase_completed(review)
        state.save()
        workflow = fwo.FullWorkflow(
            fwo.parse_args(["--plan-file", str(plan), "--yes"])
        )
        ran = []
        workflow.run_mode = lambda mode, extra: ran.append(mode) or 0  # type: ignore[assignment]
        workflow._verify = lambda *a, **k: ("ok", "stub", None)  # type: ignore[assignment]
        getattr(workflow, step)()
        assert ran == [apply]
        assert not StateManager(plan).is_phase_skipped(apply)

    def test_the_workflow_never_opts_into_the_standalone_high_gate(
        self, tmp_path, stub_phases, capsys
    ):
        """`full` must never ask `review-code` to fail on HIGH findings.

        A completed review is exit 0 whatever it found; ``--fail-on-high`` is the
        opt-in that turns HIGH findings into exit 3, and it exists for a
        standalone CI caller. Handing it to the workflow would re-create the bug
        it replaced: :meth:`_verify` stops the chain on any non-zero, so `full`
        would abort **at** review-code — one phase before `apply-code-fixes`, the
        phase whose entire job is to fix what the review just found.
        """
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--yes"]) == 0
        capsys.readouterr()
        assert "--fail-on-high" not in dict(calls)["review-code"]

    def test_no_code_review_findings_records_the_skip(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        _complete(plan, "generate-tasks")
        assert multi_llm.main(["full", str(plan), "--yes"]) == 0
        capsys.readouterr()
        assert "apply-code-fixes" not in _modes_called(calls)
        assert StateManager(plan).is_phase_skipped("apply-code-fixes")


# ---------------------------------------------------------------------------
# 4. Rows 5/6 — the bounded auto-run this module finally supplies
# ---------------------------------------------------------------------------

class TestBoundedAutoRun:
    def _gated_plan(self, tmp_path: Path) -> Path:
        """A plan whose review-plan phase left valid, never-applied suggestions."""
        plan = _plan(tmp_path, "gated")
        review = _phase_dir(plan, "review-plan")
        review.mkdir(parents=True)
        (review / "validation.json").write_text(json.dumps({
            "g1": {"status": "valid", "importance": "HIGH"},
            "g2": {"status": "valid", "importance": "MEDIUM"},
        }), encoding="utf-8")
        return plan

    def _workflow(self, plan: Path, **flags):
        args = fwo.parse_args(["--plan-file", str(plan), *flags.get("argv", ["--yes"])])
        return fwo.FullWorkflow(args)

    def test_the_workflow_is_the_runner_ensure_prerequisites_was_waiting_for(
        self, tmp_path, capsys
    ):
        plan = self._gated_plan(tmp_path)
        workflow = self._workflow(plan)
        calls = []

        def run_mode(mode_word, extra):
            calls.append(mode_word)
            state = StateManager(plan)
            state.mark_phase_completed(mode_word)
            state.save()
            return 0

        workflow.run_mode = run_mode  # type: ignore[assignment]
        assert workflow.gate("generate-tasks", un.row(5).site_id) is None
        capsys.readouterr()
        assert calls == ["apply-suggestions"], "auto-run once, then re-check"

    def test_the_auto_run_happens_at_most_once_across_both_gates(
        self, tmp_path, capsys
    ):
        """D6a: a second unmet re-check is exit 3, not a retry — across the run."""
        plan = self._gated_plan(tmp_path)
        workflow = self._workflow(plan)
        calls = []

        def run_mode(mode_word, extra):
            calls.append(mode_word)
            return 0  # exits 0 and resolves nothing

        workflow.run_mode = run_mode  # type: ignore[assignment]
        first = workflow.gate("generate-tasks", un.row(5).site_id)
        second = workflow.gate("implement", un.row(6).site_id)
        err = capsys.readouterr().err
        assert first and "re-check still reports" in first
        assert second, "the second gate is still unmet"
        assert calls == ["apply-suggestions"], (
            "at most once per phase per run, even across rows 5 and 6"
        )
        assert "already auto-run once in this run" in err
        assert workflow.auto_ran == ["apply-suggestions"]

    def test_attended_supplies_no_runner_so_nothing_proceeds_on_stale_input(
        self, tmp_path
    ):
        plan = self._gated_plan(tmp_path)
        workflow = self._workflow(plan, argv=[])
        assert workflow.apply_runner(un.row(5).site_id) is None
        blocked = workflow.gate("generate-tasks", un.row(5).site_id)
        assert blocked and "[PREREQUISITE_UNMET]" in blocked

    def test_the_implement_gate_refuses_before_the_phase_runs(self, tmp_path, capsys):
        """Row 6 is checked by the loop, so Phase 4 never starts on stale input.

        Exercised at the step level: in a linear run Phase 2 satisfies the gate
        (which is the design's point — "satisfied rather than skipped"), so the only
        way to observe the guard is to reach Phase 4 with apply-suggestions
        unrecorded, exactly as a hand-driven sequence could.
        """
        plan = self._gated_plan(tmp_path)
        workflow = self._workflow(plan)
        calls = []

        def run_mode(mode_word, extra):
            calls.append(mode_word)
            return 0  # runs, but resolves nothing the counter can see

        workflow.run_mode = run_mode  # type: ignore[assignment]
        outcome, detail, code = workflow.step_implement()
        err = capsys.readouterr().err
        assert outcome == fwo.FAILED
        assert code == 3 and "row 6" in detail
        assert "[PREREQUISITE_UNMET] implement:" in err
        assert calls == ["apply-suggestions"], (
            "the gate auto-ran the missing apply phase once and then stopped; "
            "implement itself was never dispatched"
        )


# ---------------------------------------------------------------------------
# 5. Phase 3 for real: task generation closes its loop under `--full` too
# ---------------------------------------------------------------------------

class TestPhaseThreeClosesItsLoop:
    def test_phase_three_materializes_then_resumes_into_the_consumer(
        self, tmp_path, stub_phases, capsys
    ):
        """The native-invoker path across two invocations, mid-sequence.

        Run 1 stops at Phase 3 with the task materialized and its own follow-up
        named. Run 2 (`--resume`) consumes the persisted document, closes the loop
        and carries on through the rest of the sequence.
        """
        calls = stub_phases()
        plan = _plan(tmp_path)
        code = multi_llm.main(["full", str(plan), "--yes"])
        captured = capsys.readouterr()
        assert code == 3, captured.err
        task_file = _phase_dir(plan, "generate-tasks") / gto.TASK_FILE_NAME
        assert task_file.exists(), "Phase 3 materialized the task_generation task"
        document = json.loads(task_file.read_text(encoding="utf-8"))
        assert document["kind"] == "task_generation"
        assert document["tasks"][0]["context_mode"] == "workspace-read"
        assert "generate-tasks" in captured.err and "--reaggregate" in captured.err
        assert "implement" not in _modes_called(calls), (
            "--full never proceeds past an unfinished task generation"
        )

        _persist_generation(plan, _valid_document())
        code = multi_llm.main(["full", str(plan), "--yes", "--resume"])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "<!-- TASKS_FILE:" in plan.read_text(encoding="utf-8")
        assert _tasks_md(plan).exists()
        called = _modes_called(calls)
        assert called[-2:] == ["review-code"] or "implement" in called
        assert StateManager(plan).is_phase_completed("generate-tasks")

    def test_phase_three_through_the_executor_matches_the_native_invoker(
        self, tmp_path, stub_phases, exec_env, capsys
    ):
        """One invocation, and the same bytes either invoker would have written."""
        stub_phases()
        plan = _plan(tmp_path)
        assert multi_llm.main(
            ["full", str(plan), "--yes", "--exec-subagents"]
        ) == 0, capsys.readouterr().err
        capsys.readouterr()
        executor_result = (
            _phase_dir(plan, "generate-tasks") / gto.RESULT_NAME
        ).read_bytes()
        assert "<!-- TASKS_FILE:" in plan.read_text(encoding="utf-8")
        assert _tasks_md(plan).exists()

        # The same prompt, answered by the same mock function, persisted by the
        # same leg: byte-identical, which is the whole D3a claim.
        #
        # A second plan is a second *process* in real life, and §9.3.b's run
        # anchor is set-once per interpreter (pinned above by the `full` run's
        # own pre-flights). Clear it the way tests/conftest.py clears it between
        # tests, so this test keeps asserting the bytes rather than the anchor
        # invariant — which tests/test_run_anchor_pinning.py owns.
        import utils.provider_registry as _registry
        _registry._RUN_ANCHOR = None

        native_plan = _plan(tmp_path, "native-plan")
        assert multi_llm.main(["generate-tasks", str(native_plan), "--yes"]) == 0
        capsys.readouterr()
        task = json.loads(
            (_phase_dir(native_plan, "generate-tasks") / gto.TASK_FILE_NAME)
            .read_text(encoding="utf-8")
        )["tasks"][0]
        answer = json.loads(echo_subagent_answer("task_generation", task["prompt"]))
        native_result = _persist_generation(native_plan, answer).read_bytes()
        assert executor_result == native_result

    def test_a_schema_invalid_generation_stops_the_workflow(
        self, tmp_path, stub_phases, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        assert multi_llm.main(["full", str(plan), "--yes"]) == 3
        capsys.readouterr()
        broken = _valid_document()
        broken["tasks"][0].pop("estimated_complexity")
        (_phase_dir(plan, "generate-tasks") / gto.RESULT_NAME).write_text(
            json.dumps(broken), encoding="utf-8"
        )
        before = plan.read_bytes()
        code = multi_llm.main(["full", str(plan), "--yes", "--resume"])
        err = capsys.readouterr().err
        assert code == 3
        assert "schema_invalid" in err
        assert plan.read_bytes() == before
        assert "implement" not in _modes_called(calls)
        assert not StateManager(plan).is_phase_completed("generate-tasks")


# ---------------------------------------------------------------------------
# 6. The mode surface
# ---------------------------------------------------------------------------

class TestModeSurface:
    def test_full_is_dispatchable_and_owns_a_run_dir(self):
        mode = mode_table.resolve("full")
        assert mode.unavailable is None
        assert mode.target_module == "full_workflow_orchestrator"
        assert mode.phase == "full", (
            "--detach needs a run dir, and `status --phase full` resolves through "
            "the same table"
        )
        assert mode.supports_detach

    def test_the_parser_accepts_every_row_it_owns(self, capsys):
        args = fwo.parse_args([
            "--plan-file", "p.md", "--yes", "--review-tasks", "--resume",
            "--consolidate", "--models", "a:b", "--skip-prerequisites",
        ])
        assert args.review_tasks and args.resume and args.consolidate
        assert args.models == ["a:b"] and args.skip_prerequisites

    def test_a_missing_plan_is_exit_one(self, tmp_path, capsys):
        assert multi_llm.main(["full", str(tmp_path / "nope.md")]) == 1
        assert "Plan file not found" in capsys.readouterr().err

    def test_the_gate_helper_is_the_shared_one(self):
        assert fwo.ensure_prerequisites is prereq.ensure_prerequisites, (
            "the gate is called, never reimplemented (D6a)"
        )


# ---------------------------------------------------------------------------
# 7. Ctrl-C inside an in-process phase (decision 11.9)
# ---------------------------------------------------------------------------

class TestAnInterruptedPhaseSweepsItsClis:
    """``run_mode`` runs an async phase under its own ``asyncio.run``, which joins
    every worker thread before a ``KeyboardInterrupt`` can reach the dispatcher.
    So the phase's CLIs are swept inside the cancellation, exactly as the
    dispatcher's own ``_call_target`` does it."""

    def _phase(self, monkeypatch, body):
        import types

        module = types.ModuleType("fake_async_phase")

        async def main(argv):
            return await body()

        module.main = main
        monkeypatch.setitem(sys.modules, "fake_async_phase", module)
        monkeypatch.setattr(
            fwo.mode_table, "resolve",
            lambda word: types.SimpleNamespace(
                name=word, target_module="fake_async_phase", target_attr="main",
            ),
        )

    def _registry(self, monkeypatch):
        from types import SimpleNamespace

        from utils import process_control

        swept = []

        def fake_sweep(*_args, **_kwargs):
            swept.append(True)
            return [(SimpleNamespace(pid=4242),
                     process_control.TreeSweep(cli="terminated", groups={77: "killed"}))]

        monkeypatch.setattr(process_control.live_cli_registry, "sweep", fake_sweep)
        return swept

    def test_the_main_task_cancelled_by_a_ctrl_c_sweeps_before_asyncio_run_returns(
        self, tmp_path, monkeypatch, capsys
    ):
        import asyncio

        async def interrupted():
            # What asyncio.run's runner does with the first Ctrl-C.
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        swept = self._registry(monkeypatch)
        self._phase(monkeypatch, interrupted)
        workflow = fwo.FullWorkflow(
            fwo.parse_args(["--plan-file", str(_plan(tmp_path)), "--yes"])
        )

        with pytest.raises(asyncio.CancelledError):
            workflow.run_mode("review-plan", [])
        assert swept == [True]
        assert (
            "[INTERRUPT_SWEEP] cli pid 4242: cli terminated; process groups: 77 killed"
            in capsys.readouterr().err
        )

    def test_a_phase_that_returns_sweeps_nothing(self, tmp_path, monkeypatch, capsys):
        async def finished():
            return 0

        swept = self._registry(monkeypatch)
        self._phase(monkeypatch, finished)
        workflow = fwo.FullWorkflow(
            fwo.parse_args(["--plan-file", str(_plan(tmp_path)), "--yes"])
        )

        assert workflow.run_mode("review-plan", []) == 0
        assert swept == []
        assert "[INTERRUPT_SWEEP]" not in capsys.readouterr().err


class TestANativeWritePhaseInsideFullIsNotProgress:
    """A write phase that exits 0 recorded *started* with its batch document
    written is the row-14 case: inside `full` nobody runs the loop. Before
    2026-10-05 the check keyed on the completion two phases stamped at emit
    time; the started record is the evidence now, and the message still
    names the pending document rather than the generic "records no
    completion"."""

    def test_started_with_pending_batches_stops_at_write_phase_not_executed(
        self, tmp_path, stub_phases, monkeypatch, capsys
    ):
        calls = stub_phases()
        plan = _plan(tmp_path)
        original = fwo.FullWorkflow.run_mode

        def run_mode(self, mode_word, extra):
            if mode_word != "apply-suggestions":
                return original(self, mode_word, extra)
            calls.append((mode_word, list(extra)))
            manager = StateManager(self.plan_path)
            manager.mark_phase_started("apply-suggestions")
            manager.save()
            document = _phase_dir(plan, "apply-suggestions") / "orchestrator_output.json"
            document.parent.mkdir(parents=True, exist_ok=True)
            document.write_text(json.dumps({
                "plan_file": str(plan),
                "batches": [{"prompt": "apply 1"}, {"prompt": "apply 2"}],
            }), encoding="utf-8")
            return 0

        monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)
        code = multi_llm.main(["full", str(plan), "--yes"])
        captured = capsys.readouterr()
        assert code == 3
        assert "[WRITE_PHASE_NOT_EXECUTED] apply-suggestions exited 0 after writing 2 batch(es)" in captured.err
        assert "recorded itself completed" not in captured.err
        assert "records no completion" not in captured.err
        assert _modes_called(calls) == ["review-plan", "apply-suggestions"]
        assert not StateManager(plan).is_phase_completed("apply-suggestions")


# ---------------------------------------------------------------------------
# 8. The hooks a composite subclasses, and `full`'s answers to them
# ---------------------------------------------------------------------------

def _workflow(plan: Path, *argv: str) -> "fwo.FullWorkflow":
    return fwo.FullWorkflow(fwo.parse_args(["--plan-file", str(plan), *argv]))


_COMMON = [
    "--host", "codex", "--yes", "--exec-subagents", "--dry-run-subagents",
    "--skip-prerequisites",
]

#: The argv each reused step hands its phase, as the steps built it inline
#: before ``step_argv`` existed: (the run's flags, {step word: argv}).
STEP_ARGV_CASES = [
    pytest.param(["--yes"], {
        "review-plan": ["--yes"],
        "consolidate": ["--consolidate", "--yes"],
        "apply-suggestions": ["--agent-decide", "--no-confirm", "--yes"],
        "review-code": ["--yes"],
        "apply-code-fixes": ["--agent-decide", "--no-confirm", "--yes"],
    }, id="unattended"),
    pytest.param(["--interactive", "--quick", "--resume"], {
        "review-plan": ["--quick", "--interactive"],
        "consolidate": ["--consolidate", "--interactive"],
        "apply-suggestions": ["--resume", "--interactive"],
        "review-code": ["--quick", "--interactive"],
        "apply-code-fixes": ["--resume", "--interactive"],
    }, id="attended"),
    pytest.param([
        "--yes", "--host", "codex", "--models", "a:b", "c:d", "--resume",
        "--exec-subagents", "--dry-run-subagents", "--skip-prerequisites",
    ], {
        "review-plan": ["--models", "a:b", "c:d", *_COMMON],
        "consolidate": ["--consolidate", *_COMMON],
        "apply-suggestions": ["--agent-decide", "--no-confirm", "--resume", *_COMMON],
        "review-code": ["--models", "a:b", "c:d", *_COMMON],
        "apply-code-fixes": ["--agent-decide", "--no-confirm", "--resume", *_COMMON],
    }, id="every-forwarded-flag"),
]

#: Step word -> (the step method, the mode word it dispatches).
REUSED_STEPS = {
    "review-plan": ("step_review_plan", "review-plan"),
    "consolidate": ("step_consolidate", "review-plan"),
    "apply-suggestions": ("step_apply_suggestions", "apply-suggestions"),
    "review-code": ("step_review_code", "review-code"),
    "apply-code-fixes": ("step_apply_code_fixes", "apply-code-fixes"),
}


def _run_reused_steps(workflow: "fwo.FullWorkflow") -> dict:
    """Run the five reused steps; return {step word: (mode, argv)} as dispatched.

    ``--consolidate`` is set so the pass runs whatever the review recommended,
    and a code review with a finding is on disk so apply-code-fixes is not
    recorded skipped before it dispatches.
    """
    grouped = workflow.plan_dir / "code-review"
    grouped.mkdir(parents=True, exist_ok=True)
    (grouped / "grouped.json").write_text(json.dumps({"groups": [
        {"group_id": "g1", "suggestions": [{"title": "y", "importance": "HIGH"}]},
    ]}), encoding="utf-8")
    workflow.args.consolidate = True
    dispatched = {}
    for word, (method, _mode) in REUSED_STEPS.items():
        calls = []
        workflow.run_mode = (  # type: ignore[assignment]
            lambda mode, extra, calls=calls: calls.append((mode, list(extra))) or 0
        )
        workflow._verify = lambda *a, **k: ("ok", "stub", None)  # type: ignore[assignment]
        getattr(workflow, method)()
        assert len(calls) == 1, (word, calls)
        dispatched[word] = calls[0]
    return dispatched


def _review_plan_dir(plan: Path, *, actionable: int, consolidated: bool) -> Path:
    """A review-plan dir with *actionable* valid/needs-human groups, one invalid."""
    phase_dir = _phase_dir(plan, "review-plan")
    phase_dir.mkdir(parents=True, exist_ok=True)
    statuses = ["valid", "needs-human-decision"]
    (phase_dir / "validation.json").write_text(json.dumps({"groups": [
        *({"group_index": i, "status": statuses[i % 2]} for i in range(actionable)),
        {"group_index": actionable, "status": "invalid"},
    ]}), encoding="utf-8")
    grouped = json.dumps({"groups": []})
    (phase_dir / "grouped.json").write_text(grouped, encoding="utf-8")
    if consolidated:
        import hashlib

        (phase_dir / "consolidated.json").write_text(json.dumps({"metadata": {
            "grouped_hash": hashlib.sha256(grouped.encode("utf-8")).hexdigest(),
        }}), encoding="utf-8")
    return plan.parent / plan.stem


class TestTheIdentityIsTheInstances:
    """A composite subclass must get its own run dir, progress record and
    row-14 command. Class attributes alone would not give it them while any read
    still bound a module global, so the globals are gone: a missed read is a
    ``NameError``, never `full`'s value."""

    def test_full_keeps_its_values_on_the_class_and_none_in_the_module(self):
        cls = fwo.FullWorkflow
        assert (cls.MODE_NAME, cls.RUN_DIR_PHASE, cls.STATE_KEY) == (
            "full", "full", "full_workflow",
        )
        assert cls.REPORT_TITLE == "Full Workflow Summary"
        for name in ("MODE_NAME", "RUN_DIR_PHASE", "STATE_KEY"):
            assert not hasattr(fwo, name), name

    def test_a_subclass_identity_reaches_every_read(
        self, tmp_path, stub_phases, capsys
    ):
        from utils import cli_common

        class Other(fwo.FullWorkflow):
            MODE_NAME = "other-mode"
            RUN_DIR_PHASE = "other-run"
            STATE_KEY = "other_workflow"

            def resume_flags(self):
                return ["--models", "a:b", "--resume"]

        stub_phases(fail={"review-plan": 1})
        plan = _plan(tmp_path)
        workflow = Other(fwo.parse_args(["--plan-file", str(plan), "--yes"]))
        assert workflow.run_dir == plan.parent / plan.stem / "other-run"
        ctx = cli_common.activate_result(cli_common.ResultContext(mode="full"))
        try:
            assert workflow.run() == 3
        finally:
            cli_common.deactivate_result()
        err = capsys.readouterr().err
        resume = cli_common.dispatcher_command(
            "other-mode", plan, "--models", "a:b", "--resume"
        )
        assert resume in err
        assert ctx.next == resume and ctx.mode == "other-mode"
        state = StateManager(plan)
        assert state.get("other_workflow")["failed_phase"] == "review-plan"
        assert state.get("full_workflow") is None


class TestStepArgv:
    @pytest.mark.parametrize("argv, expected", STEP_ARGV_CASES)
    def test_full_answers_with_the_lists_the_steps_built_inline(
        self, tmp_path, argv, expected
    ):
        workflow = _workflow(_plan(tmp_path), *argv)
        assert {word: workflow.step_argv(word) for word in expected} == expected

    @pytest.mark.parametrize("argv, expected", STEP_ARGV_CASES)
    def test_each_reused_step_dispatches_exactly_its_step_argv(
        self, tmp_path, argv, expected
    ):
        dispatched = _run_reused_steps(_workflow(_plan(tmp_path), *argv))
        assert dispatched == {
            word: (REUSED_STEPS[word][1], expected[word]) for word in REUSED_STEPS
        }

    def test_a_subclass_step_argv_reaches_every_reused_step(self, tmp_path):
        class Marked(fwo.FullWorkflow):
            def step_argv(self, step):
                return [f"--step={step}"]

        workflow = Marked(
            fwo.parse_args(["--plan-file", str(_plan(tmp_path)), "--yes"])
        )
        dispatched = _run_reused_steps(workflow)
        assert dispatched == {
            word: (mode, [f"--step={word}"])
            for word, (_method, mode) in REUSED_STEPS.items()
        }

    def test_the_steps_only_full_runs_are_not_its_words(self, tmp_path):
        workflow = _workflow(_plan(tmp_path), "--yes")
        for word in ("generate-tasks", "review-tasks", "apply-task-suggestions",
                     "implement", "code-review"):
            with pytest.raises(ValueError):
                workflow.step_argv(word)


class TestRunHooks:
    def test_resume_flags_are_the_runs_own_yes_then_resume(self, tmp_path):
        plan = _plan(tmp_path)
        assert _workflow(plan, "--yes").resume_flags() == ["--yes", "--resume"]
        assert _workflow(plan).resume_flags() == ["--resume"]
        assert _workflow(
            plan, "--yes", "--models", "a:b", "--force", "--resume"
        ).resume_flags() == ["--yes", "--resume"], "no selector is carried"

    def test_skip_reason_is_the_recorded_word(self, tmp_path):
        plan = _plan(tmp_path)
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        state.mark_phase_skipped("apply-suggestions", reason="x")
        state.mark_phase_skipped("code-review", reason="earlier skip")
        state.mark_phase_completed("code-review")
        state.save()
        workflow = _workflow(plan, "--yes")
        answers = {
            step.phase: workflow.skip_reason(step)
            for step in workflow.steps() if step.skippable
        }
        assert answers == {
            "review-plan": "completed",
            "apply-suggestions": "skipped",
            "generate-tasks": None,
            "review-tasks": None,
            "apply-task-suggestions": None,
            "implement": None,
            "code-review": "completed",
            "apply-code-fixes": None,
        }

    def test_consolidation_is_the_one_step_with_no_phase(self, tmp_path):
        steps = _workflow(_plan(tmp_path), "--yes").steps()
        assert [s.phase for s in steps if not s.skippable] == [""]
        assert [s.phase for s in steps if s.skippable] == EXPECTED_ORDER

    @pytest.mark.parametrize("force", [False, True])
    def test_run_asks_skip_reason_only_inside_its_guard(
        self, tmp_path, stub_phases, capsys, force
    ):
        asked = []

        class Asking(fwo.FullWorkflow):
            def steps(self):
                return super().steps()[:3]

            def skip_reason(self, step):
                asked.append(step.phase)
                return "completed"

        calls = stub_phases()
        plan = _plan(tmp_path)
        argv = ["--plan-file", str(plan), "--yes", *(["--force"] if force else [])]
        assert Asking(fwo.parse_args(argv)).run() == 0
        out = capsys.readouterr().out
        if force:
            assert asked == [], "never asked under --force"
            assert _modes_called(calls) == ["review-plan", "apply-suggestions"]
        else:
            assert asked == ["review-plan", "apply-suggestions"], (
                "never asked about the consolidation step"
            )
            assert _modes_called(calls) == []
            assert "Phase 1: Review Plan: already completed, skipping" in out

    def test_nothing_to_do_is_false_for_full_even_when_finished(self, tmp_path):
        plan = _plan(tmp_path)
        assert _workflow(plan, "--yes").nothing_to_do() is False
        _complete(plan, *EXPECTED_ORDER)
        assert _workflow(plan, "--yes").nothing_to_do() is False


class TestConsolidationSkipReason:
    """Row 4's unattended condition, one function a planner with only a plan
    dir asks too."""

    def test_a_pass_on_these_groups_already_ran(self, tmp_path):
        from utils.consolidation import CONSOLIDATION_RECOMMENDED_THRESHOLD

        plan_dir = _review_plan_dir(
            _plan(tmp_path), actionable=CONSOLIDATION_RECOMMENDED_THRESHOLD,
            consolidated=True,
        )
        assert fwo.consolidation_skip_reason(plan_dir) == (
            "already ran on these review groups"
        )

    @pytest.mark.parametrize("review", [True, False], ids=["below", "no-review"])
    def test_below_the_threshold_it_is_not_recommended(self, tmp_path, review):
        from utils.consolidation import CONSOLIDATION_RECOMMENDED_THRESHOLD

        plan = _plan(tmp_path)
        plan_dir = plan.parent / plan.stem
        if review:
            _review_plan_dir(
                plan, actionable=CONSOLIDATION_RECOMMENDED_THRESHOLD - 1,
                consolidated=False,
            )
        assert fwo.consolidation_skip_reason(str(plan_dir)) == (
            "not recommended by review-plan"
        )

    def test_at_the_threshold_and_not_yet_run_it_runs(self, tmp_path):
        from utils.consolidation import CONSOLIDATION_RECOMMENDED_THRESHOLD

        plan_dir = _review_plan_dir(
            _plan(tmp_path), actionable=CONSOLIDATION_RECOMMENDED_THRESHOLD,
            consolidated=False,
        )
        assert fwo.consolidation_skip_reason(plan_dir) is None

    @pytest.mark.parametrize("actionable, consolidated, reason", [
        (15, True, "already ran on these review groups"),
        (14, False, "not recommended by review-plan"),
    ])
    def test_the_unattended_step_skips_with_that_answer(
        self, tmp_path, actionable, consolidated, reason
    ):
        plan = _plan(tmp_path)
        _review_plan_dir(plan, actionable=actionable, consolidated=consolidated)
        workflow = _workflow(plan, "--yes")
        workflow.run_mode = lambda *a: pytest.fail("the pass ran")  # type: ignore[assignment]
        assert workflow.step_consolidate() == (fwo.SKIPPED, reason, None)


class TestTheReportIsTheInstances:
    def test_full_prints_its_own_title(self, tmp_path, capsys):
        _workflow(_plan(tmp_path), "--yes").print_report()
        assert "\n## Full Workflow Summary\n" in capsys.readouterr().out

    def test_a_subclass_names_its_title_and_artifacts(self, tmp_path, capsys):
        plan = _plan(tmp_path)

        class Other(fwo.FullWorkflow):
            REPORT_TITLE = "Other Summary"

            def report_artifacts(self):
                return [str(plan)]

        Other(fwo.parse_args(["--plan-file", str(plan), "--yes"])).print_report()
        out = capsys.readouterr().out
        assert "\n## Other Summary\n" in out
        assert f"### Output files\n\n- {plan}\n" in out
