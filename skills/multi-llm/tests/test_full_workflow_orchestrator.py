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
        assert state.get(fwo.STATE_KEY)["current_phase"] == "complete"

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
        assert "--no-confirm" in flags and "--claude-decide" in flags
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
        assert StateManager(plan).get(fwo.STATE_KEY)["failed_phase"] == (
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
