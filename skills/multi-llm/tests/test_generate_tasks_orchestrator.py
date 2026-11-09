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
import os
import shutil
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
from utils import write_safety  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402
from tests.harness.native_invoker import simulate_native_invoker  # noqa: E402
from tests.support.generate_tasks_world import (
    PLAN_BODY,
    RESULT_NAME,
    TASK_FILE_NAME,
    _materialize,
    _phase_dir,
    _plan,
    _tasks_md,
    _valid_document,
)
from utils import task_result  # noqa: E402


#: The reserved ``progress`` keys, as ``utils/status_file.py``'s module docstring
#: enumerates them. A foreground run writes this object now (``report_progress``
#: falls back to ``status_file.update_progress`` when no run handle owns the phase
#: dir), so the foreground assertion below pins its *shape*: the keys a reader of
#: the checkpoint is allowed to meet, and never a ``run`` object beside them.
_RESERVED_PROGRESS_KEYS = {
    "step", "batch_index", "batches_total", "tasks_done", "tasks_total",
    "tasks_generated",
    "failed_task_ids", "models", "models_done", "models_succeeded",
    "models_total", "interval_s",
    # Written straight through ``update_status`` by the two executor mirrors,
    # so they already reach a foreground checkpoint.
    "subagents", "write_subagents", "sandbox_bypassed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        #
        # A second plan is a second *process* in real life, and §9.3.b's run
        # anchor is set-once per interpreter (pinned by the executor leg above).
        # Clear it the way tests/conftest.py clears it between tests, so this
        # test keeps asserting the bytes rather than the anchor invariant —
        # which tests/test_run_anchor_pinning.py owns.
        import utils.provider_registry as _registry
        _registry._RUN_ANCHOR = None

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
        # Issue #17: the log never carries a raw `uv run … review-tasks` line
        # beside the next-command block; only that block names the next step.
        assert "Next: uv run" not in captured
        assert "multi_llm.py review-tasks" not in captured
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
        assert "schema_invalid" in err and "nothing re-asked it" in err
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

    def test_the_consumer_is_called_with_the_document_not_through_argv(
        self, tmp_path, monkeypatch, capsys
    ):
        """`apply_tasks_document` gets the validated document; the CLI wrapper
        (and its second read of the file) is never involved."""
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())

        import update_plan_tasks

        def no_cli(argv=None):
            raise AssertionError("consume_document must not call main()")

        monkeypatch.setattr(update_plan_tasks, "main", no_cli)
        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"])
        assert code == 0, capsys.readouterr().err
        assert _tasks_md(plan).exists()

    def test_a_refused_document_restores_the_plan(self, tmp_path, monkeypatch, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        before = plan.read_bytes()

        import update_plan_tasks

        def refuse(plan_path, data, *, dry_run=False):
            update_plan_tasks.replace_file(Path(plan_path), "half-done\n")
            raise update_plan_tasks.TasksDocumentError("SOURCE_DATE_EPOCH is bad")

        monkeypatch.setattr(update_plan_tasks, "apply_tasks_document", refuse)
        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"])
        err = capsys.readouterr().err
        assert code == 3
        assert "refused the document (SOURCE_DATE_EPOCH is bad)" in err
        assert plan.read_bytes() == before

    def test_an_unreadable_result_is_exit_three(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        (_phase_dir(plan) / RESULT_NAME).write_text("not json at all", encoding="utf-8")
        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--reaggregate"]
        ) == 3
        assert "could not be read as JSON" in capsys.readouterr().err


class TestTheMarkerIsCheckedTheWayImplementFollowsIt:
    """Generation succeeds only on a plan implement can read its task list
    from: the writer rewrites every marker that counts (not one quoted as
    code), and `verify_outcome` resolves the marker with implement's own
    resolver (`tasks_file.resolve_tasks_marker`)."""

    def _generate(self, plan: Path) -> int:
        _materialize(plan)
        _persist(plan, _valid_document())
        return multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"])

    @pytest.mark.parametrize("stray", [
        "<!-- TASKS_FILE: tg-plan/../tg-plan/tasks/tasks.md -->",
        "<!--TASKS_FILE: stale/tasks.md-->",
        "<!--TASKS_FILE:-->",
        "<!-- TASKS_FILE:-->",
        "<!--TASKS_FILE:\n-->",
    ])
    def test_a_marker_under_the_title_is_rewritten_for_implement(
        self, tmp_path, capsys, stray
    ):
        """The `..` marker used to be left in place: generation reported
        success, and implement then refused the plan."""
        from utils.tasks_file import read_task_list, tasks_marker_spans

        plan = tmp_path / "tg-plan.md"
        plan.write_text(PLAN_BODY.replace("\n", f"\n{stray}\n", 1), encoding="utf-8")

        assert self._generate(plan) == 0, capsys.readouterr().err

        text = plan.read_text(encoding="utf-8")
        assert text.startswith(
            "# Task Generation Plan\n<!-- TASKS_FILE: tg-plan/tasks/tasks.md -->\n"
        )
        assert {raw for _, _, raw in tasks_marker_spans(text)} == {"tg-plan/tasks/tasks.md"}
        assert read_task_list(plan).path == _tasks_md(plan).resolve()

    def test_a_quoted_example_marker_is_left_alone(self, tmp_path, capsys):
        from utils.tasks_file import read_task_list

        example = "```\n<!-- TASKS_FILE: example.md -->\n```\n"
        plan = tmp_path / "tg-plan.md"
        plan.write_text(PLAN_BODY + "\n" + example, encoding="utf-8")

        assert self._generate(plan) == 0, capsys.readouterr().err

        assert example in plan.read_text(encoding="utf-8")
        assert read_task_list(plan).path == _tasks_md(plan).resolve()

    def test_a_marker_implement_refuses_fails_generation(
        self, tmp_path, monkeypatch, capsys
    ):
        """Even one that leads to the rendered file: `(plan.parent /
        marker).resolve()` accepted `tg-plan/../tg-plan/tasks/tasks.md`."""
        import update_plan_tasks

        plan = tmp_path / "tg-plan.md"
        plan.write_text(
            PLAN_BODY + "\n<!-- TASKS_FILE: tg-plan/../tg-plan/tasks/tasks.md -->\n",
            encoding="utf-8",
        )
        _materialize(plan)
        _persist(plan, _valid_document())
        before = plan.read_bytes()
        monkeypatch.setattr(
            update_plan_tasks, "insert_tasks_file_reference",
            lambda content, tasks_file: content,
        )

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"])

        err = capsys.readouterr().err
        assert code == 3
        assert "implement would refuse the plan's TASKS_FILE marker" in err
        assert "climbs out with .." in err
        assert plan.read_bytes() == before

    def test_verify_outcome_refuses_disagreeing_markers(self, tmp_path):
        plan = tmp_path / "tg-plan.md"
        _tasks_md(plan).parent.mkdir(parents=True)
        _tasks_md(plan).write_text("### T001: One\nBody.\n", encoding="utf-8")
        plan.write_text(
            PLAN_BODY + "\n<!-- TASKS_FILE: tg-plan/tasks/tasks.md -->\n"
            "<!-- TASKS_FILE: other.md -->\n",
            encoding="utf-8",
        )

        problems = gto.verify_outcome(plan, 1)

        assert len(problems) == 1
        assert "different task lists" in problems[0]


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
    def test_an_unavailable_session_model_uses_the_executor_fallback(
        self, tmp_path, exec_env, monkeypatch, capsys
    ):
        from utils import runner_selection
        from utils import subagent_executor as executor
        from utils.provider_registry import HOST_DEFAULT_LAYER, SESSION_LAYER

        exec_env("codex:gpt-5.5")
        session_spec = "codex:unavailable-session-model"
        fallback_spec = "codex:gpt-5.5"
        monkeypatch.setattr(
            runner_selection, "resolve_subagent_runner",
            lambda *args, skip_session=False, **kwargs: (
                (fallback_spec, HOST_DEFAULT_LAYER) if skip_session
                else (session_spec, SESSION_LAYER)
            ),
        )
        real_preflight = executor.preflight_runner

        def preflight(spec, **kwargs):
            if spec == session_spec:
                raise executor.RunnerRejected(
                    "session model is absent from the CLI catalog",
                    spec=spec, reason_code="invalid_model_catalog",
                )
            return real_preflight(spec, **kwargs)

        monkeypatch.setattr(executor, "preflight_runner", preflight)
        plan = _plan(tmp_path)
        code = multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        )
        output = capsys.readouterr()
        assert code == 0, output.out + output.err
        assert _tasks_md(plan).exists()
        report = json.loads(
            (_phase_dir(plan) / "subagent_batch_report.json").read_text(encoding="utf-8")
        )
        assert report["runner_fallback"]["from"] == session_spec
        assert report["runner_fallback"]["to"] == fallback_spec
        assert report["tasks"][0]["runner"] == fallback_spec

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

    def test_force_moves_the_stale_result_out_of_the_executors_way(
        self, tmp_path, capsys
    ):
        """Skipping the consume branch is not enough (D3i).

        The executor's resume rule is defined by the same artifact — a task is
        done iff its result file exists, parses and validates — so a schema-valid
        result left in place makes the re-dispatch a silent no-op and the
        consumer ingests exactly the stale document ``--force`` was invoked to
        replace. It is archived rather than deleted: a result that cost a
        dispatch is never thrown away silently.
        """
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        phase_dir = _phase_dir(plan)

        assert multi_llm.main(["generate-tasks", str(plan), "--force"]) == 0

        out = capsys.readouterr().out
        assert f"[force] archived the previous {RESULT_NAME}" in out
        assert not (phase_dir / RESULT_NAME).exists(), (
            "a surviving result file re-arms the executor's resume rule"
        )
        archived = [
            p for p in phase_dir.iterdir()
            if p.name.startswith(Path(RESULT_NAME).stem) and p.name != RESULT_NAME
        ]
        assert len(archived) == 1, f"expected one archived result, got {archived}"
        assert json.loads(archived[0].read_text(encoding="utf-8"))

    def test_dry_run_subagents_keeps_the_result_it_would_not_replace(
        self, tmp_path, capsys
    ):
        """D3l: the flag's promise is that nothing else changes.

        This run starts no child, so there is nothing to replace the old result
        with — archiving it would mutate the phase dir under a flag whose whole
        point is that it does not.
        """
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())

        assert multi_llm.main(
            ["generate-tasks", str(plan), "--force", "--dry-run-subagents"]
        ) == 0

        out = capsys.readouterr().out
        assert "[dry-run-subagents] keeping the existing" in out
        assert (_phase_dir(plan) / RESULT_NAME).exists()

    def test_reaggregate_without_a_persisted_result_is_exit_three(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        assert multi_llm.main(["generate-tasks", str(plan), "--reaggregate"]) == 3
        assert RESULT_NAME in capsys.readouterr().err

    def test_a_missing_plan_is_exit_one(self, tmp_path, capsys):
        assert multi_llm.main(["generate-tasks", str(tmp_path / "nope.md")]) == 1
        assert "Plan file not found" in capsys.readouterr().err


class TestOmittedListKeysOnTheNativePath:
    """The native host writes the reply itself, so the executor's completion of
    an omitted-but-empty list key runs on consume instead — and the completed
    document goes back to disk, because `update_plan_tasks` reads the file."""

    def test_a_reply_without_files_to_create_is_consumed(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        document = _valid_document()
        for task in document["tasks"]:
            task.pop("files_to_create")
        path = _phase_dir(plan) / RESULT_NAME
        path.write_text(json.dumps(document), encoding="utf-8")
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 0
        out = capsys.readouterr().out
        assert f"[generate-tasks] {RESULT_NAME} omitted 'files_to_create' on 2 task(s)" in out
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert all(t["files_to_create"] == [] for t in on_disk["tasks"])
        assert _tasks_md(plan).exists()

    def test_a_missing_substantive_field_is_still_schema_invalid(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        document = _valid_document()
        document["tasks"][0].pop("acceptance_criteria")
        document["tasks"][0].pop("files_to_create")
        path = _phase_dir(plan) / RESULT_NAME
        path.write_text(json.dumps(document), encoding="utf-8")
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 3
        err = capsys.readouterr().err
        assert "acceptance_criteria" in err and "files_to_create" not in err
        assert not _tasks_md(plan).exists()


# ---------------------------------------------------------------------------
# 9. The phase machine reaches `complete` on success
# ---------------------------------------------------------------------------

class TestPhaseStateOnSuccess:
    """`.status.json`'s top-level `state` is the phase machine, and `claim_run`
    deliberately leaves it alone (a native host's `task_generation_pending`
    must survive into the `--reaggregate` run). So a successful consume has to
    write its own terminal word, or a `done` run reads beside whatever the last
    failed attempt left — measured 2026-09-14: `phase state: subagents_failed`
    under `run.state: done`, shown to the session model on the retry that
    worked."""

    def test_a_successful_consume_records_complete(self, tmp_path, capsys):
        from utils import status_file

        plan = _plan(tmp_path)
        _materialize(plan)
        # What a failed executor dispatch leaves behind (review_orchestrator_base).
        status_file.update_status(str(_phase_dir(plan)), {"state": "subagents_failed"})
        _persist(plan, _valid_document())
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 0
        capsys.readouterr()
        assert status_file.read_status(str(_phase_dir(plan))).get("state") == "complete"

    def test_a_failed_consume_leaves_the_state_alone(self, tmp_path, capsys):
        from utils import status_file

        plan = _plan(tmp_path)
        _materialize(plan)
        status_file.update_status(str(_phase_dir(plan)), {"state": "subagents_failed"})
        document = _valid_document()
        document["tasks"][0].pop("acceptance_criteria")
        (_phase_dir(plan) / RESULT_NAME).write_text(json.dumps(document), encoding="utf-8")
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 3
        capsys.readouterr()
        assert status_file.read_status(str(_phase_dir(plan))).get("state") == "subagents_failed"


# ---------------------------------------------------------------------------
# 10. A detached run hands its artifacts back instead of being globbed for
# ---------------------------------------------------------------------------

class _FakeRunHandle:
    """Just enough of ``detach.RunHandle`` for the detached-only writes to fire.

    ``record_output_files`` and ``report_progress`` are no-ops unless a handle
    owns this phase dir — that is what keeps a foreground run's `.status.json`
    free of new keys — so the detached half of the contract needs one.
    """

    def __init__(self, phase_dir: Path):
        self.phase_dir = str(phase_dir)

    def progress(self, **updates):
        from utils import status_file

        status_file.update_progress(self.phase_dir, updates)


class TestTheHostIsToldWhatWasProduced:
    """`status --view run` reads `output_files` off `.status.json` and repeats
    it in `MULTI_LLM_RESULT.artifacts`; that poll is all a detached caller sees.
    With nothing recorded, the one file this phase exists to produce was unnamed
    — and it lands in the sibling `tasks/` dir, so globbing the phase dir could
    not find it either. Measured 2026-09-16: an opencode host listed the four
    intermediate JSON blobs under `generate-tasks/` and never mentioned
    `tasks.md`, the plan edit, or how many tasks had been generated."""

    def _consume_detached(self, tmp_path, monkeypatch, capsys):
        from utils import detach, status_file

        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        monkeypatch.setattr(detach, "_ACTIVE", _FakeRunHandle(_phase_dir(plan)))
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 0
        capsys.readouterr()
        return plan, status_file.read_status(str(_phase_dir(plan)))

    def test_the_rendered_task_list_and_the_plan_are_named(
        self, tmp_path, monkeypatch, capsys
    ):
        plan, status = self._consume_detached(tmp_path, monkeypatch, capsys)
        output_files = status.get("output_files") or {}
        assert output_files.get("tasks") == str(_tasks_md(plan))
        assert output_files.get("plan") == str(plan)
        assert all(Path(path).exists() for path in output_files.values())

    def test_the_progress_line_carries_the_task_count(
        self, tmp_path, monkeypatch, capsys
    ):
        """1/1 subagents is the least informative number this phase has.

        The count is what this run PRODUCED (`tasks_generated`), never a
        done/total pair: `apply-task-suggestions` later splits tasks in
        tasks.md, and `tasks 10/10` beside a 12-task list read as a live
        count gone stale (2026-10-04, Claude Code racing re-run)."""
        _plan, status = self._consume_detached(tmp_path, monkeypatch, capsys)
        progress = status.get("progress") or {}
        assert progress.get("step") == "complete"
        assert progress.get("tasks_generated") == 2
        assert "tasks_total" not in progress and "tasks_done" not in progress

    def test_a_foreground_run_records_progress_and_no_run(self, tmp_path, capsys):
        """The two halves of the same checkpoint now disagree, on purpose.

        ``record_output_files`` keeps its handle guard — artifacts are the
        *detached* hand-back, and a foreground host reads them off
        ``MULTI_LLM_RESULT.artifacts`` (the sibling class below pins that) — so
        ``output_files`` stays absent. ``report_progress`` grew a foreground
        write path, so ``progress`` is present and carries this phase's real
        task counts. What survives of "with `--detach` unset, behavior is
        unchanged" is the narrower invariant: a foreground ``.status.json``
        gains a ``progress`` object and still **no `run` object**, because only
        a detached worker's ``claim_run`` writes process state.
        """
        from utils import status_file

        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 0
        capsys.readouterr()
        status = status_file.read_status(str(_phase_dir(plan)))
        assert "output_files" not in status
        progress = status.get("progress") or {}
        assert progress.get("step") == "complete"
        assert progress.get("tasks_generated") == 2
        # Only keys the reserved object documents: a foreground checkpoint is
        # not a place for a writer to invent one.
        assert set(progress) <= _RESERVED_PROGRESS_KEYS, progress
        assert "run" not in status, (
            "a foreground run claims no run; `progress` is the only reserved "
            "object it writes"
        )


class TestAForegroundRunNamesItsArtifactsOnTheResultLine:
    """`record_output_files` is detached-only by design; the terminal
    `MULTI_LLM_RESULT.artifacts` list is what a foreground host reads, and
    `consume_document` records the task list and the plan there too."""

    def test_the_task_list_and_the_plan_are_on_the_result_line(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        _persist(plan, _valid_document())
        assert multi_llm.main(["generate-tasks", str(plan), "--yes", "--reaggregate"]) == 0
        result = parse_result_line(capsys.readouterr().out)
        assert result is not None and result["exit"] == 0
        assert str(_tasks_md(plan)) in result["artifacts"]
        assert str(plan) in result["artifacts"]


# ---------------------------------------------------------------------------
# The one schema-repair re-ask (2026-10-05)
# ---------------------------------------------------------------------------

#: What `opencode/big-pickle` returned on 2026-10-04, twice, for this phase: one
#: task's fields at the top level, none of the document's own keys.
WRONG_SHAPE = {
    "id": "T001",
    "title": "Build the game",
    "acceptance_criteria": ["the game loads"],
    "depends_on": [],
}


def _codex_stream(document) -> str:
    """A ``codex exec --json`` stream whose final agent message is *document*."""
    events = [
        {"type": "thread.started", "thread_id": "t-1"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {
            "id": "item_1", "type": "agent_message", "text": json.dumps(document),
        }},
        {"type": "turn.completed", "usage": {
            "input_tokens": 10, "cached_input_tokens": 0,
            "output_tokens": 12, "reasoning_output_tokens": 0,
        }},
    ]
    return "\n".join(json.dumps(event) for event in events)


@pytest.fixture
def scripted_runner(monkeypatch):
    """Replace the dispatch spawn with a runner that answers from a script.

    Each task-generation dispatch takes the next reply in order; anything else
    the executor spawns (a nested-exec preflight probe) gets the probe token.
    Returns the list of prompts the task-generation dispatches were sent.
    """
    from utils import subagent_executor as executor

    prompts = []
    script = []

    def fake_spawn(**kwargs):
        prompt = kwargs.get("prompt") or ""
        if "You are decomposing" not in prompt:
            return executor._Capture(
                stdout=_codex_stream(executor.NESTED_PROBE_MARKER),
                exit_code=0, duration=0.01, argv=["codex", "exec"],
            )
        prompts.append(prompt)
        reply = script[min(len(prompts), len(script)) - 1]
        if isinstance(reply, executor._Capture):
            # A scripted transport failure (a timeout, a crash) rather than a
            # reply.
            return reply
        return executor._Capture(
            stdout=_codex_stream(reply), stderr="", exit_code=0,
            duration=0.01, argv=["codex", "exec", "--json"],
        )

    monkeypatch.setattr(executor, "_spawn", fake_spawn)

    def play(*replies):
        script[:] = list(replies)
        return prompts

    return play


class TestSchemaRepairReAsk:
    """A task-generation reply in the wrong shape gets ONE re-ask with the
    validator's errors before the phase gives up (FINDINGS-opencode issue 2:
    a free model answered the wrong shape twice and generate-tasks dead-ended
    at exit 3, rung 6 never retried)."""

    def _report(self, plan: Path) -> dict:
        return json.loads(
            (_phase_dir(plan) / "subagent_batch_report.json").read_text(
                encoding="utf-8"
            )
        )

    def test_wrong_then_right_succeeds_with_both_attempts_recorded(
        self, tmp_path, exec_env, scripted_runner, capsys
    ):
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(WRONG_SHAPE, _valid_document())
        plan = _plan(tmp_path)

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--exec-subagents"])
        out = capsys.readouterr().out

        assert code == 0, out
        assert "<!-- TASKS_FILE:" in plan.read_text(encoding="utf-8")
        assert _tasks_md(plan).exists()
        assert "schema repair" in out

        # The re-ask is the original question plus the validator's own words
        # and the reply shape, sent to the same runner.
        assert len(prompts) == 2
        original, repair = prompts
        assert repair.startswith(original.rstrip())
        assert "## Your previous reply was rejected" in repair
        assert "missing required field 'tasks'" in repair
        assert "unexpected field 'acceptance_criteria'" in repair
        from utils.subagent_prompts import (
            SCHEMA_REPAIR_EXAMPLES,
            TASK_GENERATION_REPLY_SHAPE,
        )
        assert TASK_GENERATION_REPLY_SHAPE.rstrip() in repair
        # It closes on one complete, parseable document of the right shape.
        assert repair.rstrip().endswith(
            json.dumps(SCHEMA_REPAIR_EXAMPLES["task_generation"], indent=2)
        )

        # Both attempts are on record: the rejected one as a failure capture,
        # the re-ask in the batch report under its own name.
        capture = _phase_dir(plan) / "failed" / "task_generation.attempt1.json"
        assert json.loads(capture.read_text(encoding="utf-8"))["status"] == (
            "schema_invalid"
        )
        row = self._report(plan)["tasks"][0]
        assert row["status"] == "ok"
        assert row["attempt_count"] == 2
        # The row itself names the re-ask (the round-5 report's
        # `repair: schema_invalid`), not only its second attempt.
        assert row["repair"] == "schema_invalid"
        assert [(a["attempt"], a["status"], a.get("repair")) for a in row["attempts"]] == [
            (1, "schema_invalid", None),
            (2, "ok", "schema_invalid"),
        ]

    def test_wrong_twice_is_exit_three_with_the_plan_untouched(
        self, tmp_path, exec_env, scripted_runner, capsys
    ):
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(WRONG_SHAPE, WRONG_SHAPE)
        plan = _plan(tmp_path)
        before = plan.read_bytes()

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--exec-subagents"])
        capsys.readouterr()

        assert code == 3
        assert plan.read_bytes() == before
        assert not _tasks_md(plan).exists()
        assert len(prompts) == 2, "exactly one re-ask, never a second"
        failed = _phase_dir(plan) / "failed"
        second = json.loads(
            (failed / "task_generation.attempt2.json").read_text(encoding="utf-8")
        )
        assert second["status"] == "schema_invalid"
        assert second["repair"] == "schema_invalid"
        assert (failed / "task_generation.attempt1.json").exists()
        row = self._report(plan)["tasks"][0]
        assert row["status"] == "schema_invalid"
        assert row["attempt_count"] == 2
        assert row["repair"] == "schema_invalid"
        assert "after the one schema-repair re-ask" in row["reason"]

    def test_a_re_run_after_a_reply_shape_failure_takes_the_new_runner(
        self, tmp_path, exec_env, scripted_runner, capsys
    ):
        """2026-10-05, OpenCode run 3: generate-tasks failed twice on the
        runner's reply shape, the host switched model, and the re-run stayed on
        the failed runner because its batch-report row pinned it. A
        reply-quality failure is not a pin (fix round 8)."""
        exec_env("codex:gpt-5.5")
        scripted_runner(WRONG_SHAPE, WRONG_SHAPE, _valid_document())
        plan = _plan(tmp_path)
        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        ) == 3
        capsys.readouterr()

        exec_env("codex:gpt-5.4")
        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        ) == 0
        second = capsys.readouterr().out

        assert "task_generation -> codex:gpt-5.4 (version" in second
        assert "kept from subagent_batch_report.json" not in second

    def test_a_re_run_names_why_it_kept_the_earlier_runner(
        self, tmp_path, exec_env, scripted_runner, capsys
    ):
        """2026-10-05, OpenCode run 3: the dispatch line named a pinned runner
        but not why. A transport failure (here a timeout, twice) keeps the
        pin, and the line says where it came from."""
        from utils import subagent_executor as executor

        exec_env("codex:gpt-5.5")
        timed_out = executor._Capture(
            timed_out=True, exit_code=None, duration=0.01,
            argv=["codex", "exec", "--json"],
        )
        scripted_runner(timed_out, timed_out, _valid_document())
        plan = _plan(tmp_path)
        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        ) == 3
        first = capsys.readouterr().out
        assert "task_generation -> codex:gpt-5.5 (version" in first
        assert "kept from subagent_batch_report.json" not in first

        exec_env("codex:gpt-5.4")
        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        ) == 0
        second = capsys.readouterr().out

        assert "task_generation -> codex:gpt-5.5 (version" in second
        assert (
            "the runner this task's earlier attempt used, kept from "
            "subagent_batch_report.json so a re-run repeats it" in second
        )

    def test_a_reply_over_the_byte_cap_is_not_re_asked(
        self, tmp_path, exec_env, scripted_runner, capsys, monkeypatch
    ):
        """The cap is not a shape problem (fix round 6): ``persist_result``
        refuses an oversized but valid document, and a re-ask telling the
        runner its reply "does not match the result shape" would be false."""
        from utils import subagent_executor as executor

        monkeypatch.setattr(task_result, "MAX_RESULT_BYTES", 512)
        exec_env("codex:gpt-5.5")
        big = _valid_document()
        big["tasks"][0]["description"] = "x" * 2000
        prompts = scripted_runner(big, _valid_document())
        plan = _plan(tmp_path)

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--exec-subagents"])
        out = capsys.readouterr().out

        assert code == 3
        assert len(prompts) == 1, "no re-ask for a result over the byte cap"
        assert "schema repair" not in out
        assert "result shape" not in out
        row = self._report(plan)["tasks"][0]
        assert row["status"] == "schema_invalid"
        assert "over the 512-byte cap" in row["reason"]
        assert "schema-repair re-ask" not in row["reason"]
        assert [a.get("repair") for a in row["attempts"]] == [None]
        assert "repair" not in row

    @staticmethod
    def _slot_clock(monkeypatch, *, wall_below_timeout=None):
        """Drive the dispatch's ladder on a hand-advanced clock.

        Returns ``(clock, box)``: each task-generation spawn advances
        ``clock["t"]`` by ``clock["tick"]`` (a callable of the ladder's
        timeout), and ``box["ladder"]`` is the ladder the dispatch built. With
        *wall_below_timeout*, the runner declares a worst-case attempt that
        many seconds under the slot's timeout (§7.5 rule 4's envelope), so the
        first attempt is admitted and nothing after a long one is.
        """
        from types import SimpleNamespace

        from utils import subagent_executor as executor

        clock = {"t": 1000.0, "tick": lambda timeout: 0.0}
        box = {}
        real_ladder = executor.RetryLadder

        class Ladder(real_ladder):
            def __init__(self, *args, **kwargs):
                kwargs["now"] = lambda: clock["t"]
                super().__init__(*args, **kwargs)
                if wall_below_timeout is not None:
                    self.envelope = SimpleNamespace(
                        max_requests=4,
                        max_wall_s=self.timeout - wall_below_timeout,
                        retries_rate_limit=False,
                    )
                box["ladder"] = self

        monkeypatch.setattr(executor, "RetryLadder", Ladder)
        scripted = executor._spawn

        def ticking_spawn(**kwargs):
            capture = scripted(**kwargs)
            if "You are decomposing" in (kwargs.get("prompt") or ""):
                clock["t"] += clock["tick"](box["ladder"].timeout)
            return capture

        monkeypatch.setattr(executor, "_spawn", ticking_spawn)
        return clock, box

    def test_a_re_ask_the_envelope_cannot_hold_keeps_the_attempts_own_status(
        self, tmp_path, exec_env, scripted_runner, capsys, monkeypatch
    ):
        """Fix round 6: the re-ask used to be granted on rule 3 alone, then
        refused by rule 4 at the next start, and the slot was recorded
        ``budget_exhausted`` — "Nothing was dispatched" — after attempt 1 ran."""
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(WRONG_SHAPE, _valid_document())
        clock, _box = self._slot_clock(monkeypatch, wall_below_timeout=10.0)
        clock["tick"] = lambda timeout: 40.0
        plan = _plan(tmp_path)

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--exec-subagents"])
        capsys.readouterr()

        assert code == 3
        assert len(prompts) == 1
        row = self._report(plan)["tasks"][0]
        assert row["status"] == "schema_invalid", row["reason"]
        assert "Nothing was dispatched" not in row["reason"]
        assert "no schema-repair re-ask: only " in row["reason"]
        assert "§7.5 rule 4" in row["reason"]
        assert [a["status"] for a in row["attempts"]] == ["schema_invalid"]

    def test_a_re_ask_the_deadline_refuses_says_so(
        self, tmp_path, exec_env, scripted_runner, capsys, monkeypatch
    ):
        """The ladder's "(no schema-repair re-ask: …)" note used to be dropped,
        leaving a reason that read as if the re-ask had never been owed."""
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(WRONG_SHAPE, _valid_document())
        clock, _box = self._slot_clock(monkeypatch)
        clock["tick"] = lambda timeout: timeout - 0.5
        plan = _plan(tmp_path)

        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        ) == 3
        capsys.readouterr()

        assert len(prompts) == 1
        row = self._report(plan)["tasks"][0]
        assert row["status"] == "schema_invalid"
        assert "no schema-repair re-ask: the per-slot wall-clock budget" in row["reason"]

    def test_a_transport_retry_the_envelope_cannot_hold_keeps_its_status(
        self, tmp_path, exec_env, scripted_runner, capsys, monkeypatch
    ):
        """The same grant-then-refuse shape on rung 5: an unparseable reply's
        retry was granted, refused at the next start, and recorded as a slot
        that dispatched nothing."""
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner("no JSON here, just prose", _valid_document())
        clock, _box = self._slot_clock(monkeypatch, wall_below_timeout=10.0)
        clock["tick"] = lambda timeout: 40.0
        plan = _plan(tmp_path)

        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        ) == 3
        capsys.readouterr()

        assert len(prompts) == 1
        row = self._report(plan)["tasks"][0]
        assert row["status"] == "unparseable_result", row["reason"]
        assert "Nothing was dispatched" not in row["reason"]
        assert "(no retry: only " in row["reason"]
        assert "1 retry unused" in row["reason"]

    def test_a_right_first_reply_is_never_re_asked(
        self, tmp_path, exec_env, scripted_runner, capsys
    ):
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(_valid_document())
        plan = _plan(tmp_path)

        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--exec-subagents"]
        ) == 0
        assert "schema repair" not in capsys.readouterr().out
        assert len(prompts) == 1
        attempts = self._report(plan)["tasks"][0]["attempts"]
        assert [a.get("repair") for a in attempts] == [None]


# ---------------------------------------------------------------------------
# The semantic rules are part of the executor's verdict (fix round 6)
# ---------------------------------------------------------------------------

def _cyclic_document() -> dict:
    """Schema-valid, and every task depends on another: no task can start."""
    document = _valid_document()
    document["tasks"][0]["depends_on"] = ["T002"]
    document["tasks"][1]["depends_on"] = ["T001"]
    return document


class TestSemanticRulesOnTheExecutorLane:
    """``validate_generated_tasks`` used to run only in ``consume_document``,
    after the executor had persisted the document as ``ok``: a cyclic reply was
    exit 3 with no re-ask, and a plain re-invoke then re-consumed the same bad
    file through ``[resume]`` (only ``--force`` escaped)."""

    def _report(self, plan: Path) -> dict:
        return json.loads(
            (_phase_dir(plan) / "subagent_batch_report.json").read_text(
                encoding="utf-8"
            )
        )

    def test_validate_result_runs_the_graph_rules(self):
        from utils.subagent_executor import validate_result

        assert validate_result("task_generation", _valid_document()) == []
        problems = validate_result("task_generation", _cyclic_document())
        assert any("cycle" in problem for problem in problems), problems

    def test_a_cyclic_result_on_disk_is_not_complete(self, tmp_path):
        from utils.subagent_executor import task_is_complete

        plan = _plan(tmp_path)
        task = _materialize(plan)
        path = _phase_dir(plan) / RESULT_NAME
        path.write_text(json.dumps(_cyclic_document()), encoding="utf-8")
        assert task_is_complete(task, str(_phase_dir(plan))) is False
        path.write_text(json.dumps(_valid_document()), encoding="utf-8")
        assert task_is_complete(task, str(_phase_dir(plan))) is True

    def test_a_cyclic_reply_gets_the_re_ask(
        self, tmp_path, exec_env, scripted_runner, capsys
    ):
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(_cyclic_document(), _valid_document())
        plan = _plan(tmp_path)

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--exec-subagents"])
        out = capsys.readouterr().out

        assert code == 0, out
        assert _tasks_md(plan).exists()
        assert len(prompts) == 2
        assert "## Your previous reply was rejected" in prompts[1]
        assert "cycle" in prompts[1]
        row = self._report(plan)["tasks"][0]
        assert [(a["attempt"], a["status"], a.get("repair")) for a in row["attempts"]] == [
            (1, "schema_invalid", None),
            (2, "ok", "schema_invalid"),
        ]

    def test_a_cyclic_reply_twice_persists_nothing(
        self, tmp_path, exec_env, scripted_runner, capsys
    ):
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(_cyclic_document(), _cyclic_document())
        plan = _plan(tmp_path)
        before = plan.read_bytes()

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--exec-subagents"])
        capsys.readouterr()

        assert code == 3
        assert len(prompts) == 2
        assert plan.read_bytes() == before
        assert not (_phase_dir(plan) / RESULT_NAME).exists(), (
            "a rule-invalid document reached the result path"
        )
        row = self._report(plan)["tasks"][0]
        assert row["status"] == "schema_invalid"
        assert "cycle" in row["reason"]

    def test_a_stale_invalid_result_is_archived_and_re_dispatched(
        self, tmp_path, exec_env, scripted_runner, capsys
    ):
        """A plain re-invoke never resumes from a document that fails
        validation: it sets the file aside and dispatches again."""
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(_valid_document())
        plan = _plan(tmp_path)
        _materialize(plan)
        stale = _phase_dir(plan) / RESULT_NAME
        stale.write_text(json.dumps(_cyclic_document()), encoding="utf-8")
        capsys.readouterr()

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--exec-subagents"])
        out = capsys.readouterr().out

        assert code == 0, out
        assert "[resume] " in out and "is invalid" in out
        assert "consuming it instead of re-dispatching" not in out
        assert len(prompts) == 1, "the task was dispatched again"
        assert _tasks_md(plan).exists()
        archived = [
            p for p in _phase_dir(plan).iterdir()
            if p.name.startswith(Path(RESULT_NAME).stem) and p.name != RESULT_NAME
        ]
        assert len(archived) == 1, archived
        assert json.loads(archived[0].read_text(encoding="utf-8")) == _cyclic_document()

    def test_the_native_path_re_materializes_instead_of_consuming(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        _materialize(plan)
        (_phase_dir(plan) / RESULT_NAME).write_text(
            json.dumps(_cyclic_document()), encoding="utf-8"
        )
        capsys.readouterr()

        assert multi_llm.main(["generate-tasks", str(plan)]) == 0
        out = capsys.readouterr().out

        assert gto.PENDING_MARKER in out
        assert not (_phase_dir(plan) / RESULT_NAME).exists()
        assert not _tasks_md(plan).exists()

    def test_dry_run_subagents_keeps_the_invalid_result(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        stale = _phase_dir(plan) / RESULT_NAME
        stale.write_text(json.dumps(_cyclic_document()), encoding="utf-8")
        capsys.readouterr()

        assert multi_llm.main(
            ["generate-tasks", str(plan), "--dry-run-subagents"]
        ) == 0
        out = capsys.readouterr().out

        assert "[dry-run-subagents]" in out and "is invalid" in out
        assert stale.exists()
        assert not _tasks_md(plan).exists()

    def test_a_valid_result_on_disk_still_resumes(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        document = _valid_document()
        for task in document["tasks"]:
            task.pop("files_to_create")  # completed on consume, so still valid
        (_phase_dir(plan) / RESULT_NAME).write_text(
            json.dumps(document), encoding="utf-8"
        )
        assert multi_llm.main(["generate-tasks", str(plan)]) == 0
        out = capsys.readouterr().out
        assert "consuming it instead of re-dispatching" in out
        assert _tasks_md(plan).exists()

    def test_the_reaggregate_failure_says_what_each_lane_does(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        _materialize(plan)
        (_phase_dir(plan) / RESULT_NAME).write_text(
            json.dumps(_cyclic_document()), encoding="utf-8"
        )
        assert multi_llm.main(
            ["generate-tasks", str(plan), "--yes", "--reaggregate"]
        ) == 3
        err = capsys.readouterr().err
        assert "NOT retried" not in err
        assert "--exec-subagents" in err and "re-asks" in err
        assert "without --reaggregate" in err


# ---------------------------------------------------------------------------
# Every file this phase writes beside the plan stays in the plan's directory
# ---------------------------------------------------------------------------

def _tree(root: Path) -> dict:
    """Every path under *root*: a file's bytes, inode and mtime, a link's
    target text, ``None`` for a folder. Equal trees mean nothing was written
    in there, not even bytes put back afterwards."""
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            found[str(path)] = ("link", os.readlink(path))
        elif path.is_dir():
            found[str(path)] = None
        else:
            info = path.stat()
            found[str(path)] = (path.read_bytes(), info.st_ino, info.st_mtime_ns)
    return found


def _bytes_or_none(path: Path):
    return path.read_bytes() if path.exists() else None


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs privileges on Windows")
class TestTheOutputsStayInThePlansDirectory:
    """A repository symlink named like one of this phase's outputs never
    carries a write out of the plan's directory: the task list, the
    phase folder, the task file, the rewritten result document and the
    archived result are each contained before their first byte
    (docs/development/write-safety.md#plan-output-containment). The plan
    itself may still be linked in from anywhere."""

    @pytest.fixture
    def world(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "victim.md").write_text("precious\n", encoding="utf-8")
        return repo, outside

    @staticmethod
    def _link_the_task_list(plan: Path, target: Path) -> None:
        _tasks_md(plan).parent.mkdir(parents=True, exist_ok=True)
        _tasks_md(plan).symlink_to(target)

    def test_a_task_list_linked_out_is_refused_before_anything_is_produced(
            self, world, capsys):
        repo, outside = world
        plan = _plan(repo)
        self._link_the_task_list(plan, outside / "victim.md")
        outside_before = _tree(outside)

        assert multi_llm.main(["generate-tasks", str(plan)]) == 3

        err = capsys.readouterr().err
        assert "Refusing the task list" in err and "Nothing was dispatched" in err
        assert not _phase_dir(plan).exists(), "no phase folder, no task file"
        assert _tree(outside) == outside_before
        assert plan.read_text(encoding="utf-8") == PLAN_BODY

    @pytest.mark.parametrize("entry", ["the phase", "the consumer"])
    def test_a_task_list_linked_out_after_dispatch_is_refused_and_nothing_changes(
            self, world, capsys, entry):
        """The link planted between dispatch and consume: the plan, its state
        and the file outside are byte for byte what they were, whether the
        phase's preflight or the consumer's own check meets it."""
        repo, outside = world
        plan = _plan(repo)
        _materialize(plan)
        result = _persist(plan, _valid_document())
        self._link_the_task_list(plan, outside / "victim.md")
        state = plan.parent / plan.stem / "state.json"
        plan_before, state_before = plan.read_bytes(), _bytes_or_none(state)
        outside_before = _tree(outside)
        capsys.readouterr()

        if entry == "the phase":
            code = multi_llm.main(["generate-tasks", str(plan), "--reaggregate"])
        else:
            code = gto.consume_document(plan, str(result))

        assert code == 3
        assert "Refusing the task list" in capsys.readouterr().err
        assert _tree(outside) == outside_before, "the task list landed outside"
        assert plan.read_bytes() == plan_before
        assert _bytes_or_none(state) == state_before

    def test_a_task_file_linked_out_is_refused(self, world, capsys):
        repo, outside = world
        plan = _plan(repo)
        _phase_dir(plan).mkdir(parents=True)
        (_phase_dir(plan) / TASK_FILE_NAME).symlink_to(outside / "victim.md")
        outside_before = _tree(outside)

        assert multi_llm.main(["generate-tasks", str(plan)]) == 3

        assert "Refusing the task file" in capsys.readouterr().err
        assert _tree(outside) == outside_before

    def test_a_result_document_linked_out_is_not_rewritten(self, world, capsys):
        """The consumer writes a supplied ``schema_version`` back into the
        result before validating: never through a link that leads out."""
        repo, outside = world
        plan = _plan(repo)
        _materialize(plan)
        document = _valid_document()
        document.pop("schema_version")
        (outside / "result.json").write_text(json.dumps(document), encoding="utf-8")
        (_phase_dir(plan) / RESULT_NAME).symlink_to(outside / "result.json")
        outside_before = _tree(outside)
        capsys.readouterr()

        assert multi_llm.main(["generate-tasks", str(plan), "--reaggregate"]) == 3

        assert "Refusing the result document" in capsys.readouterr().err
        assert _tree(outside) == outside_before
        assert plan.read_text(encoding="utf-8") == PLAN_BODY

    def test_the_move_aside_is_contained(self, world, capsys):
        """``--force`` archives the old result by renaming it in its folder:
        a folder that leads out is refused, and nothing moves there."""
        repo, outside = world
        plan = _plan(repo)
        _materialize(plan)
        _persist(plan, _valid_document())
        moved = outside / "generate-tasks"
        shutil.move(str(_phase_dir(plan)), str(moved))
        _phase_dir(plan).symlink_to(moved, target_is_directory=True)
        outside_before = _tree(outside)
        capsys.readouterr()

        assert multi_llm.main(["generate-tasks", str(plan), "--force"]) == 3
        assert "Refusing the generate-tasks folder" in capsys.readouterr().err
        assert _tree(outside) == outside_before

        # The archive's own check, met without the phase's preflight.
        code = gto._move_result_aside(
            plan, str(_phase_dir(plan) / RESULT_NAME), False,
            kept="", failed=lambda exc: f"ERROR: {exc}", done=lambda path: "",
        )
        assert code == 3
        assert "Refusing the archived result" in capsys.readouterr().err
        assert _tree(outside) == outside_before, "nothing was renamed"

    def test_a_task_list_linked_inside_the_plans_directory_is_written_through(
            self, world, capsys):
        repo, _outside = world
        plan = _plan(repo)
        shared = repo / "shared" / "tasks.md"
        shared.parent.mkdir()
        self._link_the_task_list(plan, Path("..") / ".." / "shared" / "tasks.md")
        _materialize(plan)
        _persist(plan, _valid_document())

        assert multi_llm.main(["generate-tasks", str(plan), "--reaggregate"]) == 0

        capsys.readouterr()
        assert _tasks_md(plan).is_symlink()
        assert "### Task T001" in shared.read_text(encoding="utf-8")

    def test_a_plan_linked_in_from_outside_the_repository_still_completes(
            self, world, capsys):
        repo, outside = world
        (outside / "kept").mkdir()
        real = _plan(outside / "kept")
        link = repo / real.name
        link.symlink_to(real)
        _materialize(link)
        _persist(link, _valid_document())

        assert multi_llm.main(["generate-tasks", str(link), "--reaggregate"]) == 0

        capsys.readouterr()
        assert link.is_symlink() and link.resolve() == real.resolve()
        assert "<!-- TASKS_FILE: tg-plan/tasks/tasks.md -->" in real.read_text(
            encoding="utf-8")
        assert _tasks_md(link).is_file()

    def test_a_failed_consume_restores_a_linked_plans_own_state(
            self, world, capsys, monkeypatch):
        """``StateManager`` keeps a linked plan's state beside the file the
        link leads to, so that is the ``state.json`` the restore puts back."""
        repo, outside = world
        (outside / "kept").mkdir()
        real = _plan(outside / "kept")
        link = repo / real.name
        link.symlink_to(real)
        _materialize(link)
        result = _persist(link, _valid_document())
        state = real.parent / real.stem / "state.json"
        plan_before, state_before = real.read_bytes(), _bytes_or_none(state)
        monkeypatch.setattr(gto, "verify_outcome", lambda *_a, **_k: ["forced"])
        capsys.readouterr()

        assert gto.consume_document(link, str(result)) == 3

        capsys.readouterr()
        assert real.read_bytes() == plan_before
        assert _bytes_or_none(state) == state_before, (
            "generate-tasks must not stay recorded as completed")
        assert not _tasks_md(link).exists()


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs privileges on Windows")
class TestEveryWriteGoesThroughItsHeldFolder:
    """What a static check cannot see: a link swapped in after it, for one of
    the phase's files or a folder above one, and the folder the executor's
    failure captures go in (Codex's review of U86a, 2026-11-09). Every write
    of the flow goes through its folder held below the plan's directory, so a
    link met there is refused, never followed; every destination is checked
    before the first write, so a refusal changes nothing in the tree
    (docs/development/write-safety.md#plan-output-containment)."""

    @pytest.fixture
    def world(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "victim.json").write_text('{"precious": true}\n', encoding="utf-8")
        return tmp_path, repo, outside

    @staticmethod
    def _swap_after_check(monkeypatch, what, swap):
        """Run *swap* once ``contain_plan_output`` has checked *what*: the
        seam just before that file is written."""
        import update_plan_tasks

        real_check = update_plan_tasks.contain_plan_output
        wanted, swapped = what, []

        def check_then_swap(plan_path, path, *, what: str, **kwargs):
            checked = real_check(plan_path, path, what=what, **kwargs)
            if what == wanted and not swapped:
                swap()
                swapped.append(what)
            return checked

        monkeypatch.setattr(update_plan_tasks, "contain_plan_output", check_then_swap)
        return swapped

    @pytest.mark.skipif(not write_safety._PUBLISHES_BY_DESCRIPTOR,
                        reason="the held-folder write is Linux and macOS")
    def test_a_task_file_swapped_for_a_link_after_its_check_is_refused(
            self, world, monkeypatch):
        tmp_path, repo, outside = world
        plan = _plan(repo)
        phase = _phase_dir(plan)
        phase.mkdir(parents=True)
        (phase / TASK_FILE_NAME).write_text("{}", encoding="utf-8")

        def swap():
            (phase / TASK_FILE_NAME).unlink()
            (phase / TASK_FILE_NAME).symlink_to(outside / "victim.json")

        swapped = self._swap_after_check(monkeypatch, "task file", swap)
        before = _tree(outside)

        with pytest.raises(gto.TaskFileError, match="Refusing the task file"):
            gto.materialize_task_file(plan, str(phase))

        assert swapped
        assert _tree(outside) == before, "the task file landed outside"

    @pytest.mark.skipif(not write_safety._PUBLISHES_BY_DESCRIPTOR,
                        reason="the held-folder write is Linux and macOS")
    def test_a_result_swapped_for_a_link_after_its_check_is_not_rewritten(
            self, world, monkeypatch, capsys):
        """The consumer writes a supplied ``schema_version`` back into the
        result: never through a link swapped in for it after its check."""
        tmp_path, repo, outside = world
        plan = _plan(repo)
        _materialize(plan)
        result = _persist(plan, _valid_document())
        document = _valid_document()
        document.pop("schema_version")
        (outside / "victim.json").write_text(json.dumps(document), encoding="utf-8")

        def swap():
            result.unlink()
            result.symlink_to(outside / "victim.json")

        swapped = self._swap_after_check(monkeypatch, "result document", swap)
        before, plan_before = _tree(outside), plan.read_bytes()
        capsys.readouterr()

        assert gto.consume_document(plan, str(result)) == 3

        assert swapped
        assert "could not be rewritten" in capsys.readouterr().err
        assert _tree(outside) == before, "the result was rewritten outside"
        assert plan.read_bytes() == plan_before

    @pytest.mark.skipif(not write_safety._PUBLISHES_BY_DESCRIPTOR,
                        reason="the held-folder write is Linux and macOS")
    def test_a_phase_folder_swapped_for_a_link_after_its_check_moves_nothing(
            self, world, monkeypatch, capsys):
        """``--force`` moves the old result aside by a rename in its folder:
        never in a folder a link swapped in after the check leads to."""
        tmp_path, repo, outside = world
        plan = _plan(repo)
        _materialize(plan)
        _persist(plan, _valid_document())
        phase = _phase_dir(plan)
        copy = outside / "generate-tasks"
        shutil.copytree(phase, copy)

        def swap():
            phase.rename(phase.with_name("saved"))
            phase.symlink_to(copy, target_is_directory=True)

        swapped = self._swap_after_check(monkeypatch, "archived result", swap)
        before = _tree(outside)
        capsys.readouterr()

        code = gto._move_result_aside(
            plan, str(phase / RESULT_NAME), False,
            kept="", failed=lambda exc: f"ERROR: {exc}", done=lambda path: "",
        )

        assert swapped
        assert code == 3
        assert "without following a link" in capsys.readouterr().err
        assert _tree(outside) == before, "the result was moved aside outside"

    def test_a_failure_capture_folder_linked_out_is_never_written(self, world):
        """The executor's own write: a capture whose folder leads out is not
        written, and an earlier capture there is not retired either."""
        from utils import subagent_executor as executor

        tmp_path, repo, outside = world
        plan = _plan(repo)
        phase = _phase_dir(plan)
        phase.mkdir(parents=True)
        (phase / executor.FAILED_CAPTURE_DIRNAME).symlink_to(
            outside, target_is_directory=True)
        (outside / f"{gto.TASK_ID}.attempt1.json").write_text("{}", encoding="utf-8")
        before = _tree(outside)

        written = executor._write_failure_capture(
            phase_dir=str(phase), task={"task_id": gto.TASK_ID, "kind": gto.TASK_KIND},
            attempt_index=1, argv=["runner"], prompt="the prompt", stdout="",
            stderr="failed", exit_code=1, duration=0.01, status="execution_failed",
            error="failed",
        )

        assert written is None
        assert _tree(outside) == before, "a capture was written or retired outside"

    def test_a_failure_capture_folder_linked_out_is_refused_before_dispatch(
            self, world, exec_env, scripted_runner, capsys):
        """The same folder met by the phase's own check: nothing is dispatched,
        so no attempt can leave its capture outside."""
        tmp_path, repo, outside = world
        plan = _plan(repo)
        phase = _phase_dir(plan)
        phase.mkdir(parents=True)
        (phase / "failed").symlink_to(outside, target_is_directory=True)
        exec_env("codex:gpt-5.5")
        prompts = scripted_runner(WRONG_SHAPE, _valid_document())
        before = _tree(outside)

        code = multi_llm.main(["generate-tasks", str(plan), "--yes", "--exec-subagents"])

        err = capsys.readouterr().err
        assert code == 3
        assert "Refusing the failure-capture folder" in err
        assert "Nothing was dispatched" in err
        assert prompts == []
        assert _tree(outside) == before

    def test_a_task_file_linked_out_changes_nothing_in_the_tree(self, world, capsys):
        """Refused before the scope policy is recorded, before the prompt
        sidecar is cleaned up or written, before the task file: the whole
        tree, the existing ``state.json`` and prompt included, is byte for
        byte and inode for inode what it was."""
        from utils.state_manager import StateManager

        tmp_path, repo, outside = world
        plan = _plan(repo)
        StateManager(plan).save()
        phase = _phase_dir(plan)
        phase.mkdir(parents=True)
        (phase / gto.PROMPT_FILE_NAME).write_text("an earlier prompt\n", encoding="utf-8")
        (phase / TASK_FILE_NAME).symlink_to(outside / "victim.json")
        state = plan.parent / plan.stem / "state.json"
        assert state.is_file()
        before = _tree(tmp_path)
        capsys.readouterr()

        code = multi_llm.main(["generate-tasks", str(plan), "--scope-policy", "preserve"])

        assert code == 3
        assert "Refusing the task file" in capsys.readouterr().err
        assert _tree(tmp_path) == before, "something was written before the refusal"
