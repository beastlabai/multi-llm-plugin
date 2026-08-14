"""`--yes` end-to-end, with stdin closed and no TTY (D6's Done-when row).

"Zero prompts" is **observed here, not inferred**. Grepping the instruction files
for `AskUserQuestion` proves nothing about a run: the failure this plan exists to
remove is a *process* that stops and waits, and on a headless host that shows up
as a hang until the host's 2–10 min command cap kills it mid-phase. So every run
below:

* goes through the real entry point (``multi_llm.py <mode> <plan> --yes``) as a
  **subprocess**, so a hang is a test timeout rather than a passing assertion;
* runs with **stdin at /dev/null and stdout/stderr on pipes** — no TTY anywhere,
  which is exactly the `codex exec` / `opencode run` / `cursor-agent -p` shape;
* runs with a **tripwire on PATH**: `gum` and `fzf` (the two interactive pickers
  this skill can reach) are replaced by scripts that record the attempt and fail.
  A recorded attempt fails the test, so "no picker was reached" is a fact about
  the run and not a claim about the code.

Each run then has to end in one of D6's three endings — a recorded auto-selection,
exit 3 with the re-invocation named, or a flag the user typed — and to print
exactly one parseable ``MULTI_LLM_RESULT`` line either way.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import MockProvider, SkillRunner, PERF_SCALE  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent

PLAN_CONTENT = """# Unattended Plan

## Overview
A small plan used to prove that `--yes` never renders a question.

## Steps
1. Add the module.
2. Wire it up.
"""

TASKS_CONTENT = """# Tasks

## T001: Add the module
- Depends on: none
- Files: src/mod.py

Create the module.

### Acceptance Criteria
- It imports.

## T002: Wire it up
- Depends on: T001
- Files: src/app.py

Wire the module in.

### Acceptance Criteria
- It is reachable.
"""


# ---------------------------------------------------------------------------
# The harness: stdin closed, no TTY, and a tripwire where the pickers live
# ---------------------------------------------------------------------------

class PromptTripwire:
    """`gum`/`fzf` replaced by recorders, so a reached picker is provable."""

    def __init__(self, tmp_path: Path):
        self.dir = tmp_path / "tripwire-bin"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = tmp_path / "prompt-attempts.log"
        for name in ("gum", "fzf"):
            script = self.dir / name
            script.write_text(
                "#!/bin/sh\n"
                f'echo "{name} $*" >> "{self.log}"\n'
                "exit 1\n",
                encoding="utf-8",
            )
            script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                         | stat.S_IXOTH)

    @property
    def attempts(self):
        if not self.log.exists():
            return []
        return [line for line in self.log.read_text(encoding="utf-8").splitlines()
                if line.strip()]


def run_unattended(
    runner: SkillRunner,
    tripwire: PromptTripwire,
    *argv: str,
    timeout: int = 60,
):
    """Run ``multi_llm.py`` with no TTY and no stdin, and assert nothing prompted."""
    env = runner._build_env()
    env["PATH"] = f"{tripwire.dir}{os.pathsep}{env['PATH']}"
    command = [sys.executable, str(SKILL_ROOT / "multi_llm.py"), *argv]
    completed = subprocess.run(
        command,
        # No TTY: both streams are pipes. No stdin at all: a read blocks nothing,
        # it fails immediately, which is what makes a would-be prompt visible.
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout * PERF_SCALE,
        env=env,
        cwd=str(SKILL_ROOT),
    )
    assert tripwire.attempts == [], (
        "an interactive picker was launched under --yes: "
        f"{tripwire.attempts}\nstdout:\n{completed.stdout[-2000:]}"
    )
    # Belt and braces: the run must not have told anyone to ask, either.
    assert "AskUserQuestion" not in completed.stdout
    return completed


def assert_models_ran(runner: SkillRunner):
    """The fan-out really happened: the mock provider recorded a call.

    Without this a mode that exited early (a resume guard, a silent
    misconfiguration) would still "pass" the no-prompt assertion, which would make
    the whole file vacuous.
    """
    log = runner.call_log_path
    assert log.exists() and log.read_text(encoding="utf-8").strip(), (
        "the mock provider was never invoked, so this run proved nothing about a "
        "fan-out"
    )


def result_of(completed) -> dict:
    payload = parse_result_line(completed.stdout)
    assert payload is not None, (
        f"no MULTI_LLM_RESULT line in stdout:\n{completed.stdout[-2000:]}"
    )
    return payload


@pytest.fixture
def tripwire(tmp_path):
    return PromptTripwire(tmp_path)


def _configure(runner: SkillRunner, provider: MockProvider, scenario: str):
    provider.set_scenario(scenario)
    runner.extra_env.update(provider.get_env())


# ---------------------------------------------------------------------------
# Every standalone mode, unattended, on a fixture plan
# ---------------------------------------------------------------------------

class TestReviewPlanUnattended:
    def test_yes_resolves_models_from_defaults_and_never_prompts(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-review-plan", PLAN_CONTENT)
        completed = run_unattended(
            skill_runner, tripwire,
            "review-plan", str(plan.plan_path), "--yes",
            "--models", "cursor-agent", "--skip-validation",
        )
        assert completed.returncode == 0, completed.stdout[-3000:]
        assert (plan.output_dir / "review-plan" / "grouped.json").exists()
        assert result_of(completed)["exit"] == 0
        assert_models_ran(skill_runner)

    def test_defaults_are_auto_selected_and_recorded(
        self, skill_runner, fixture_manager, mock_provider, tripwire, tmp_path
    ):
        """Row 2's auto-select half, end to end: no --models, no question.

        The resolution is *announced* (`[AUTO_RESOLVED] model-selection: …`), which
        is what makes an auto-selection distinguishable from a silent degradation
        in a log nobody watched being written.
        """
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-defaults", PLAN_CONTENT)
        config = tmp_path / "defaults-providers.yaml"
        base = (SKILL_ROOT / "providers.yaml").read_text(encoding="utf-8")
        config.write_text(
            base + "\n\ndefaults:\n  models:\n    - cursor-agent:auto\n",
            encoding="utf-8",
        )
        skill_runner.extra_env["MULTI_LLM_PROVIDERS_CONFIG"] = str(config)
        completed = run_unattended(
            skill_runner, tripwire,
            "review-plan", str(plan.plan_path), "--yes", "--skip-validation",
        )
        assert completed.returncode == 0, completed.stdout[-3000:]
        assert "[AUTO_RESOLVED] model-selection: defaults.models: " \
            "cursor-agent:auto" in completed.stdout
        assert_models_ran(skill_runner)

    def test_no_models_configured_is_exit_three_not_a_picker(
        self, skill_runner, fixture_manager, mock_provider, tripwire, tmp_path
    ):
        """Row 2's fail-closed half: unresolvable models never render a gate."""
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-no-models", PLAN_CONTENT)
        # A config layer with no defaults and no quick models at all.
        empty = tmp_path / "empty-providers.yaml"
        empty.write_text("defaults:\n  models: []\n  quick_models: []\n",
                         encoding="utf-8")
        skill_runner.extra_env["MULTI_LLM_PROVIDERS_CONFIG"] = str(empty)
        completed = run_unattended(
            skill_runner, tripwire, "review-plan", str(plan.plan_path), "--yes",
        )
        assert completed.returncode == 3, completed.stdout[-3000:]
        combined = completed.stdout + completed.stderr
        assert "--models" in combined
        assert result_of(completed)["state"] == "needs-human"


class TestReviewTasksUnattended:
    def _plan_with_tasks(self, fixture_manager, name="unattended-review-tasks"):
        plan = fixture_manager.create_plan(name, PLAN_CONTENT)
        tasks_dir = plan.output_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "tasks.md").write_text(TASKS_CONTENT, encoding="utf-8")
        state = StateManager(plan.plan_path)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.mark_phase_completed("generate-tasks")
        state.save()
        return plan

    def test_yes_runs_to_completion_without_a_prompt(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = self._plan_with_tasks(fixture_manager)
        completed = run_unattended(
            skill_runner, tripwire,
            "review-tasks", str(plan.plan_path), "--yes",
            "--models", "cursor-agent", "--skip-validation",
        )
        assert completed.returncode == 0, completed.stdout[-3000:]
        assert (plan.output_dir / "review-tasks" / "grouped.json").exists()
        assert_models_ran(skill_runner)

    def test_row_thirteen_no_tasks_is_exit_three(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        """The gate `review-tasks` owns: no task list, so nothing to review."""
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-no-tasks", PLAN_CONTENT)
        completed = run_unattended(
            skill_runner, tripwire,
            "util", "prereq", "--plan-file", str(plan.plan_path),
            "--mode", "review-tasks", "--yes",
        )
        assert completed.returncode == 3, completed.stdout[-3000:]
        assert "[PREREQUISITE_UNMET] review-tasks:" in completed.stderr
        assert "generate-tasks" in completed.stderr
        # stdout stays pure JSON up to the dispatcher's result line.
        body = completed.stdout.split("MULTI_LLM_RESULT:")[0]
        assert json.loads(body)["prerequisites_met"] is False


class TestReviewCodeUnattended:
    def test_yes_runs_to_completion_without_a_prompt(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-review-code", PLAN_CONTENT)
        implement_dir = plan.ensure_phase_dir("implement")
        (implement_dir / "summary.json").write_text(json.dumps({
            "tasks_completed": 1, "tasks_total": 1,
            "files_modified": ["src/mod.py"], "status": "completed",
        }), encoding="utf-8")
        fixture_manager.create_state_file(
            plan,
            phases_completed=["review-plan", "implement"],
            extra_state={
                "head_at_start": "abc123def",
                "tracked_files": [{"path": "src/mod.py", "task_id": "T001"}],
            },
        )
        completed = run_unattended(
            skill_runner, tripwire,
            "review-code", str(plan.plan_path), "--yes",
            "--models", "cursor-agent", "--skip-validation",
        )
        assert completed.returncode == 0, completed.stdout[-3000:]
        assert (plan.output_dir / "code-review" / "grouped.json").exists()
        assert_models_ran(skill_runner)


class TestAskUnattended:
    def test_yes_answers_without_a_prompt(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-ask", PLAN_CONTENT)
        completed = run_unattended(
            skill_runner, tripwire,
            "ask", str(plan.plan_path), "Is the plan complete?", "--yes",
            "--models", "cursor-agent",
        )
        assert completed.returncode in (0, 3), completed.stdout[-3000:]
        payload = result_of(completed)
        # Either it answered (0/partial) or it failed closed (3) — never a prompt.
        assert payload["state"] in ("done", "partial", "needs-human")
        assert_models_ran(skill_runner)


class TestGenerateTasksGateUnattended:
    """Rows 5/6: the gate `--yes` must not skip, reached through the real CLI."""

    def _plan_with_unapplied(self, fixture_manager, name):
        plan = fixture_manager.create_plan(name, PLAN_CONTENT)
        review = plan.ensure_phase_dir("review-plan")
        (review / "validation.json").write_text(json.dumps({
            "g1": {"status": "valid", "importance": "HIGH"},
            "g2": {"status": "valid", "importance": "MEDIUM"},
        }), encoding="utf-8")
        return plan

    def test_gate_exits_three_and_names_both_fixes(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = self._plan_with_unapplied(fixture_manager, "unattended-gate")
        completed = run_unattended(
            skill_runner, tripwire,
            "util", "prereq", "--plan-file", str(plan.plan_path),
            "--mode", "generate-tasks", "--yes",
        )
        assert completed.returncode == 3, completed.stdout[-3000:]
        err = completed.stderr
        assert "[PREREQUISITE_UNMET] generate-tasks:" in err
        assert "1) Apply them first" in err
        assert "--skip-prerequisites" in err
        # Nothing was produced by the gate itself.
        assert not (plan.output_dir / "generate-tasks").exists()

    def test_skip_prerequisites_proceeds_and_records(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = self._plan_with_unapplied(fixture_manager, "unattended-gate-skip")
        completed = run_unattended(
            skill_runner, tripwire,
            "util", "prereq", "--plan-file", str(plan.plan_path),
            "--mode", "generate-tasks", "--yes", "--skip-prerequisites",
        )
        assert completed.returncode == 0, completed.stdout[-3000:]
        assert "[PREREQUISITE_SKIPPED]" in completed.stderr
        state = json.loads((plan.output_dir / "state.json").read_text(encoding="utf-8"))
        assert "apply-suggestions" in state.get("phases_skipped", {})

    def test_implement_gate_exits_three_through_the_dispatcher(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = self._plan_with_unapplied(fixture_manager, "unattended-impl-gate")
        completed = run_unattended(
            skill_runner, tripwire,
            "implement", str(plan.plan_path), "--yes",
            "--output", str(plan.output_dir / "tasks.json"),
        )
        assert completed.returncode == 3, completed.stdout[-3000:]
        assert "[PREREQUISITE_UNMET] implement:" in completed.stdout
        assert result_of(completed)["exit"] == 3


class TestGenerateTasksUnattended:
    """The mode itself, not just its gate: one turn, one follow-up, no prompt."""

    def test_yes_materializes_the_task_and_names_one_follow_up(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-generate", PLAN_CONTENT)
        completed = run_unattended(
            skill_runner, tripwire, "generate-tasks", str(plan.plan_path), "--yes",
        )
        assert completed.returncode == 0, completed.stdout[-3000:]
        task_file = (
            plan.output_dir / "generate-tasks" / "task_generation_tasks.json"
        )
        assert task_file.exists()
        document = json.loads(task_file.read_text(encoding="utf-8"))
        assert document["kind"] == "task_generation"
        assert len(document["tasks"]) == 1
        payload = result_of(completed)
        assert payload["exit"] == 0
        assert payload["next"].endswith("--reaggregate"), payload["next"]
        # The deleted channel stays deleted, even under the real CLI.
        assert not list(Path(plan.plan_path).parent.rglob("generated_tasks_*.json"))


class TestFullWorkflowUnattended:
    """`--full` under `--yes`: it runs, it prompts nothing, and it ends in D6's endings."""

    def test_yes_runs_phase_one_then_stops_with_resume_named(
        self, skill_runner, fixture_manager, mock_provider, tripwire
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-full", PLAN_CONTENT)
        completed = run_unattended(
            skill_runner, tripwire,
            "full", str(plan.plan_path), "--yes", "--models", "cursor-agent",
            timeout=180,
        )
        # Phase 1 fanned out for real and then stopped for its validation batch:
        # without `--exec-subagents` the phase does not record itself complete, so
        # the workflow ends at row 14 rather than marching on.
        assert_models_ran(skill_runner)
        assert completed.returncode == 3, completed.stdout[-3000:]
        assert "Phase 1: Review Plan did not complete" in completed.stderr
        assert "--resume" in completed.stderr
        payload = result_of(completed)
        assert payload["mode"] == "full" and payload["exit"] == 3
        assert payload["state"] == "needs-human"
        # `next` is the *next* command to run, which here is the phase's own
        # prerequisite: resuming before it re-materializes the same pending state
        # and points back at resume, so a host following `next` mechanically would
        # loop. The prose names both, in order.
        assert payload["next"].endswith("--reaggregate"), payload["next"]
        assert "review-plan" in payload["next"]


class TestContradictionsAndEmptyAnswers:
    """Fail-closed rows reached the same way a host would reach them."""

    @pytest.mark.parametrize("extra", [
        ["--interactive"],
        ["--models", ""],
        ["--force-strategy", "pause-and-ask"],
    ])
    def test_exit_three_before_anything_runs(
        self, skill_runner, fixture_manager, mock_provider, tripwire, extra
    ):
        _configure(skill_runner, mock_provider, "happy_path")
        plan = fixture_manager.create_plan("unattended-contradiction", PLAN_CONTENT)
        mode = "implement" if extra[0] == "--force-strategy" else "review-plan"
        completed = run_unattended(
            skill_runner, tripwire, mode, str(plan.plan_path), "--yes", *extra,
        )
        assert completed.returncode == 3, completed.stdout[-3000:]
        assert not (plan.output_dir / "review-plan").exists()
