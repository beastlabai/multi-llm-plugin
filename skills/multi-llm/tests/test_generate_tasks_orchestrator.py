#!/usr/bin/env python3
"""Task generation closes its loop (D2i/D3 — the Goal's one invalidating gap).

Standalone ``--generate-tasks`` used to produce *no task file at all*: the sequence
lived in markdown, so neither a single-shot host nor ``--exec-subagents`` could run
it. What this pins is the whole loop, end to end and under **both** invokers:

1. the phase materializes exactly one ``task_generation`` task, in the same
   envelope every other kind uses, as ``context_mode: workspace-read`` with the
   pinned repo root as the only path its prompt names;
2. dispatching it through the **executor** and through the **simulated native
   invoker** produces byte-identical ``task_generation_result.json`` files — the D3a
   print/persist contract, asserted rather than assumed;
3. the returned document is validated against
   ``schemas/generated_tasks.schema.json`` **plus** the semantic rules
   ``utils/task_decomposer.py`` owns, before a byte reaches the plan;
4. ``update_plan_tasks`` runs **in-process**, and the run succeeds only once the
   ``<!-- TASKS_FILE: … -->`` marker *and* the rendered tasks file both exist and
   agree with the document;
5. a schema-invalid document, a rule-invalid document, and a document that
   validates but leaves no marker each exit **3 with the plan file unmodified**;
6. a runner that cannot read the workspace, or cannot run read-only, is refused
   **before** anything is dispatched — the single worst outcome this phase can
   produce is a confidently invented task list, and a non-agentic runner does not
   fail on this prompt.

Usage:
    uv run -- pytest tests/test_generate_tasks_orchestrator.py -v
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import generate_tasks_orchestrator as gto  # noqa: E402
import multi_llm  # noqa: E402
from harness import SkillRunner  # noqa: E402
from mocks.mock_llm import echo_subagent_answer  # noqa: E402
from utils import task_decomposer  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402
from utils.subagent_executor import simulate_native_invoker  # noqa: E402

PLAN_BODY = """# Task Generation Plan

## Overview
Add a foundation module and wire it into the entry point.

## Step 1
Create `src/foundation.py`.

## Step 2
Call it from `src/main.py`.
"""

RESULT_NAME = gto.RESULT_NAME
TASK_FILE_NAME = gto.TASK_FILE_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan(tmp_path: Path, name: str = "tg-plan") -> Path:
    plan = tmp_path / f"{name}.md"
    plan.write_text(PLAN_BODY, encoding="utf-8")
    return plan


def _phase_dir(plan: Path) -> Path:
    return plan.parent / plan.stem / "generate-tasks"


def _tasks_md(plan: Path) -> Path:
    return plan.parent / plan.stem / "tasks" / "tasks.md"


def _valid_document() -> dict:
    """The document every leg of the loop is expected to accept."""
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


def _materialize(plan: Path) -> dict:
    """Run the phase up to (and including) the pending marker; return the task."""
    assert multi_llm.main(["generate-tasks", str(plan)]) == 0
    document = json.loads((_phase_dir(plan) / TASK_FILE_NAME).read_text(encoding="utf-8"))
    return document["tasks"][0]


def _persist(plan: Path, document) -> Path:
    """Play the native invoker: write what a subagent printed to output_path."""
    task = json.loads(
        (_phase_dir(plan) / TASK_FILE_NAME).read_text(encoding="utf-8")
    )["tasks"][0]
    report = simulate_native_invoker(
        [task], phase_dir=str(_phase_dir(plan)),
        respond=lambda _task: document, kind="task_generation",
    )
    assert report.ok, [(o.task_id, o.status, o.reason) for o in report.failed]
    return _phase_dir(plan) / RESULT_NAME


def _driver_config(tmp_path: Path, skill_dir: Path, runner: str) -> Path:
    """Base config re-asserted verbatim with one ``driver`` key changed.

    Same isolation the rest of the suite gets from
    ``MULTI_LLM_PROVIDERS_CONFIG=<base>``, one difference: the subagent runner.
    """
    base = yaml.safe_load((skill_dir / "providers.yaml").read_text(encoding="utf-8"))
    base.setdefault("driver", {})["subagent_runner"] = runner
    path = tmp_path / f"driver-{runner.replace(':', '-')}.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def exec_env(tmp_path, mock_provider, monkeypatch):
    """Run the executor leg in-process against the mock provider binaries.

    Returns a callable ``(runner_spec) -> None`` that pins the driver runner; the
    mock answers a task-generation prompt from ``MOCK_LLM_ECHO``, so the executor
    leg and the native leg derive the *same* document from the *same* prompt.
    """
    import utils.provider_registry as registry

    runner = SkillRunner(tmp_path / "harness", extra_env=mock_provider.get_env())
    env = runner._build_env()

    def configure(spec: str = "codex:gpt-5.5") -> None:
        env["MULTI_LLM_PROVIDERS_CONFIG"] = str(
            _driver_config(tmp_path, runner.skill_dir, spec)
        )
        env["MOCK_LLM_ECHO"] = "task_generation"
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        registry._config = None
        registry._config_key = None

    yield configure
    registry._config = None
    registry._config_key = None


# ---------------------------------------------------------------------------
# 1. Materialization
# ---------------------------------------------------------------------------

class TestMaterialization:
    def test_one_task_file_in_the_standard_envelope(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        assert multi_llm.main(["generate-tasks", str(plan)]) == 0
        capsys.readouterr()
        document = json.loads(
            (_phase_dir(plan) / TASK_FILE_NAME).read_text(encoding="utf-8")
        )
        assert document["tasks_schema_version"] == 2
        assert document["kind"] == "task_generation"
        assert document["inputs"]["plan_sha256"]
        assert len(document["tasks"]) == 1, "one task, so the executor needs no case"
        task = document["tasks"][0]
        assert task["task_id"] == "task_generation"
        assert task["output_name"] == RESULT_NAME
        assert task["result_channel"] == "file"
        assert task["capability_tier"] == "full"
        assert task["executor_role"] == "agent"

    def test_it_is_the_standing_workspace_read_consumer(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        task = _materialize(plan)
        capsys.readouterr()
        assert task["context_mode"] == "workspace-read", (
            "task generation is the one kind whose context is the repository"
        )
        assert task["repo_root"], "the pinned repo root is the one path it names"
        # The prompt embeds the plan and names exactly one path.
        prompt = task["prompt"]
        assert "## Plan" in prompt and "Create `src/foundation.py`" in prompt
        assert prompt.count(task["repo_root"]) == 1
        assert task["output_name"] not in prompt, (
            "output_path is invoker plumbing, never a subagent instruction (D3a)"
        )
        # Capability-neutral exploration: no Claude tool names.
        for tool in ("Glob", "Grep", "Read tool", "Write tool"):
            assert tool not in prompt, tool

    def test_the_pending_path_hands_back_exactly_one_follow_up(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        assert multi_llm.main(["generate-tasks", str(plan)]) == 0
        out = capsys.readouterr().out
        assert gto.PENDING_MARKER in out
        result = parse_result_line(out)
        assert result["next"] and "generate-tasks" in result["next"]
        assert result["next"].endswith("--reaggregate")
        assert "--internal-" not in result["next"]

    def test_the_temp_file_channel_is_gone(self, tmp_path, capsys):
        """`{PROJECT_ROOT}/.multi-llm/tmp/generated_tasks_*.json` is not a contract."""
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        assert multi_llm.main(["generate-tasks", str(plan), "--reaggregate"]) == 0
        capsys.readouterr()
        assert not list(tmp_path.rglob("generated_tasks_*.json"))
        assert not list(tmp_path.rglob(".multi-llm")), (
            "the temp dir and its .gitignore bootstrap are deleted, not ported"
        )


# ---------------------------------------------------------------------------
# 2. Both invokers, byte-identically (D3a)
# ---------------------------------------------------------------------------

class TestBothInvokers:
    def test_executor_and_native_invoker_agree_byte_for_byte(
        self, tmp_path, exec_env, capsys
    ):
        exec_env("codex:gpt-5.5")

        (tmp_path / "exec").mkdir(parents=True, exist_ok=True)
        (tmp_path / "native").mkdir(parents=True, exist_ok=True)
        executor_plan = _plan(tmp_path / "exec", "tg-plan")
        native_plan = _plan(tmp_path / "native", "tg-plan")

        # Executor leg: one invocation, marker to consumer, no second turn.
        assert multi_llm.main(
            ["generate-tasks", str(executor_plan), "--yes", "--exec-subagents"]
        ) == 0
        captured = capsys.readouterr().out
        assert "[exec-subagents]" in captured
        executor_result = (_phase_dir(executor_plan) / RESULT_NAME).read_bytes()

        # Native leg: the same prompt, answered by the same mock function, written
        # by the same persist leg.
        task = _materialize(native_plan)
        answer = json.loads(echo_subagent_answer("task_generation", task["prompt"]))
        _persist(native_plan, answer)
        assert multi_llm.main(
            ["generate-tasks", str(native_plan), "--yes", "--reaggregate"]
        ) == 0
        capsys.readouterr()
        native_result = (_phase_dir(native_plan) / RESULT_NAME).read_bytes()

        assert executor_result == native_result, (
            "whichever invoker ran, the persisted document must be identical"
        )
        # And both closed the loop.
        for plan in (executor_plan, native_plan):
            assert "<!-- TASKS_FILE:" in plan.read_text(encoding="utf-8")
            assert _tasks_md(plan).exists()

    def test_the_loop_closes_only_when_both_artifacts_exist(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        assert multi_llm.main(["generate-tasks", str(plan), "--reaggregate"]) == 0
        capsys.readouterr()
        plan_text = plan.read_text(encoding="utf-8")
        assert "<!-- TASKS_FILE: tg-plan/tasks/tasks.md -->" in plan_text
        tasks_md = _tasks_md(plan)
        assert tasks_md.exists()
        parsed = task_decomposer.TaskDecomposer().parse_plan(
            tasks_md.read_text(encoding="utf-8")
        )
        assert [t.id for t in parsed] == ["T001", "T002"]
        # The consumer ran in-process, so the phase is recorded complete here.
        from utils.state_manager import StateManager

        state = StateManager(plan)
        assert state.is_phase_completed("generate-tasks")
        assert state.get("plan_preamble")


# ---------------------------------------------------------------------------
# 3. Validation before anything touches the plan
# ---------------------------------------------------------------------------

class TestDocumentValidation:
    def _run_with(self, plan: Path, document) -> tuple:
        """Materialize, persist *document* as the reply, consume. Returns (code, plan)."""
        _materialize(plan)
        before = plan.read_bytes()
        path = _phase_dir(plan) / RESULT_NAME
        path.write_text(json.dumps(document), encoding="utf-8")
        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"])
        return code, before

    def test_schema_invalid_is_exit_three_and_the_plan_is_untouched(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        broken = _valid_document()
        broken["tasks"][0].pop("acceptance_criteria")
        code, before = self._run_with(plan, broken)
        err = capsys.readouterr().err
        assert code == 3
        assert "schema_invalid" in err and "NOT retried" in err
        assert plan.read_bytes() == before
        assert not _tasks_md(plan).exists()

    @pytest.mark.parametrize("mutate,needle", [
        (lambda d: d["tasks"][1].__setitem__("depends_on", ["T404"]),
         "never declares"),
        (lambda d: (d["tasks"][0].__setitem__("depends_on", ["T002"]),
                    d["tasks"][1].__setitem__("depends_on", ["T001"])),
         "cycle"),
        (lambda d: d["tasks"][0].__setitem__("depends_on", ["T002"]),
         "cycle"),
        (lambda d: d["tasks"][1].__setitem__("id", "T001"),
         "duplicate"),
    ])
    def test_rule_invalid_is_exit_three_and_the_plan_is_untouched(
        self, tmp_path, capsys, mutate, needle
    ):
        plan = _plan(tmp_path)
        document = _valid_document()
        mutate(document)
        code, before = self._run_with(plan, document)
        err = capsys.readouterr().err
        assert code == 3, err
        assert needle in err, err
        assert plan.read_bytes() == before
        assert not _tasks_md(plan).exists()

    def test_a_document_that_validates_but_leaves_no_marker_is_exit_three(
        self, tmp_path, monkeypatch, capsys
    ):
        """The outcome is verified, never the subagent's claim (D2i)."""
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        before = plan.read_bytes()

        import update_plan_tasks

        # The consumer "succeeds" and writes the plan back unchanged: no marker.
        monkeypatch.setattr(
            update_plan_tasks, "insert_tasks_file_reference",
            lambda content, tasks_file: content,
        )
        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"])
        err = capsys.readouterr().err
        assert code == 3
        assert "TASKS_FILE" in err
        assert plan.read_bytes() == before, "the plan is restored, not left half-done"
        from utils.state_manager import StateManager

        assert not StateManager(plan).is_phase_completed("generate-tasks"), (
            "a verification failure must not leave the phase recorded complete"
        )

    def test_a_count_mismatch_is_caught(self, tmp_path, monkeypatch, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        before = plan.read_bytes()

        import update_plan_tasks

        original = update_plan_tasks.generate_tasks_markdown
        monkeypatch.setattr(
            update_plan_tasks, "generate_tasks_markdown",
            lambda tasks, plan_name: original(tasks[:1], plan_name),
        )
        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"])
        err = capsys.readouterr().err
        assert code == 3
        assert "parses to 1 task(s)" in err and "declared 2" in err
        assert plan.read_bytes() == before

    def test_an_unreadable_result_is_exit_three(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        (_phase_dir(plan) / RESULT_NAME).write_text("not json at all", encoding="utf-8")
        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--reaggregate"]
        ) == 3
        assert "could not be read as JSON" in capsys.readouterr().err


class TestSemanticRules:
    """The four rules, as a unit — they live with the task graph, not the schema."""

    def test_a_valid_graph_has_no_problems(self):
        assert task_decomposer.validate_generated_tasks(
            _valid_document()["tasks"]
        ) == []

    def test_no_dependency_free_task_is_rejected(self):
        tasks = _valid_document()["tasks"]
        tasks[0]["depends_on"] = ["T002"]
        problems = task_decomposer.validate_generated_tasks(tasks)
        assert any("cycle" in p for p in problems)

    def test_a_single_task_depending_on_itself_is_rejected(self):
        tasks = [dict(_valid_document()["tasks"][0], depends_on=["T001"])]
        problems = task_decomposer.validate_generated_tasks(tasks)
        assert any("itself" in p for p in problems)

    def test_a_bad_id_shape_is_rejected(self):
        tasks = _valid_document()["tasks"]
        tasks[0]["id"] = "task-1"
        assert any(
            "T### id" in p
            for p in task_decomposer.validate_generated_tasks(tasks)
        )

    def test_an_empty_list_is_rejected(self):
        assert task_decomposer.validate_generated_tasks([]) == [
            "`tasks` must be a non-empty array"
        ]

    def test_a_long_chain_is_accepted(self):
        tasks = []
        for index in range(1, 8):
            task = dict(_valid_document()["tasks"][0])
            task["id"] = f"T{index:03d}"
            task["depends_on"] = [] if index == 1 else [f"T{index - 1:03d}"]
            tasks.append(task)
        assert task_decomposer.validate_generated_tasks(tasks) == []


# ---------------------------------------------------------------------------
# 4. The pre-dispatch runner refusal (D2c/D3e)
# ---------------------------------------------------------------------------

class TestRunnerRefusal:
    def test_a_runner_without_workspace_reads_is_refused_before_dispatch(
        self, tmp_path, exec_env, capsys
    ):
        exec_env("gemini:gemini-2.5-flash")
        plan = _plan(tmp_path)
        code = multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        )
        err = capsys.readouterr().err
        assert code == 3
        assert "workspace-read" in err and "gemini" in err
        assert "invents a task list" in err, "the reason is named, not just the code"
        assert not (_phase_dir(plan) / RESULT_NAME).exists()
        assert "<!-- TASKS_FILE:" not in plan.read_text(encoding="utf-8")

    def test_a_runner_without_the_read_only_profile_is_refused(
        self, tmp_path, exec_env, monkeypatch, capsys
    ):
        """The *other* half of the requirement, isolated.

        Every adapter that declares ``supports_workspace_reads`` today also
        implements ``agent-readonly``, so the profile branch is reached by granting
        a non-agentic adapter the capability and nothing else — which is exactly
        the configuration that must still be refused.
        """
        from utils.provider_registry import get_provider

        exec_env("gemini:gemini-2.5-flash")
        provider = get_provider("gemini")
        monkeypatch.setattr(type(provider), "supports_workspace_reads", True)
        plan = _plan(tmp_path)
        code = multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        )
        err = capsys.readouterr().err
        assert code == 3
        assert "agent-readonly" in err
        assert not (_phase_dir(plan) / RESULT_NAME).exists()

    def test_the_native_path_needs_no_runner_at_all(self, tmp_path, exec_env, capsys):
        """Without ``--exec-subagents`` the invoker is the host, so no runner is asked."""
        exec_env("gemini:gemini-2.5-flash")
        plan = _plan(tmp_path)
        assert multi_llm.main(["generate-tasks", str(plan), "--yes"]) == 0
        assert gto.PENDING_MARKER in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 5. The gate, resume, and the follow-up
# ---------------------------------------------------------------------------

class TestGate:
    def test_unapplied_suggestions_block_the_phase_and_produce_nothing(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        review = plan.parent / plan.stem / "review-plan"
        review.mkdir(parents=True)
        (review / "validation.json").write_text(json.dumps({
            "g1": {"status": "valid", "importance": "HIGH"},
        }), encoding="utf-8")
        code = multi_llm.main(["generate-tasks", str(plan), "--yes"])
        err = capsys.readouterr().err
        assert code == 3
        assert "[PREREQUISITE_UNMET] generate-tasks:" in err
        assert not _phase_dir(plan).exists(), "nothing is produced before the gate"

    def test_skip_prerequisites_is_loud_recorded_and_proceeds(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        review = plan.parent / plan.stem / "review-plan"
        review.mkdir(parents=True)
        (review / "validation.json").write_text(json.dumps({
            "g1": {"status": "valid", "importance": "HIGH"},
        }), encoding="utf-8")
        code = multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--skip-prerequisites"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "[PREREQUISITE_SKIPPED] generate-tasks:" in out
        assert (_phase_dir(plan) / TASK_FILE_NAME).exists()


class TestResume:
    def test_an_existing_result_is_consumed_instead_of_re_dispatched(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        assert multi_llm.main(["generate-tasks", str(plan)]) == 0
        out = capsys.readouterr().out
        assert "[resume]" in out
        assert _tasks_md(plan).exists()

    def test_force_re_dispatches_instead_of_consuming(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        assert multi_llm.main(["generate-tasks", str(plan), "--force"]) == 0
        out = capsys.readouterr().out
        assert "[resume]" not in out
        assert gto.PENDING_MARKER in out

    def test_reaggregate_without_a_persisted_result_is_exit_three(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        assert multi_llm.main(["generate-tasks", str(plan), "--reaggregate"]) == 3
        assert RESULT_NAME in capsys.readouterr().err

    def test_a_missing_plan_is_exit_one(self, tmp_path, capsys):
        assert multi_llm.main(["generate-tasks", str(tmp_path / "nope.md")]) == 1
        assert "Plan file not found" in capsys.readouterr().err
