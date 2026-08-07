"""D2h consumer sweep: every emitter carries ``executor_role`` and nothing
branches on ``subagent_type`` any more.

``subagent_type`` holds a Claude Code agent name, and its two values mean
different things: ``general-purpose`` carries no information (it is pure Claude
Code vocabulary), while ``human`` is not an agent type at all but the marker that
no subagent may run this task. The field therefore splits rather than being
renamed — and survives one release as optional Claude-only metadata that no reader
consults.
"""

import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from update_plan_tasks import (  # noqa: E402
    generate_dependency_diagram,
    generate_tasks_markdown,
    task_executor_role,
)
from utils.code_fix_batcher import (  # noqa: E402
    CodeFixBatch,
    determine_executor_role,
    determine_subagent_type,
    batch_code_fixes,
    estimate_batch_processing_stats,
)
from utils.metrics import normalize_recorded_executor, record_metric  # noqa: E402
from utils.task_decomposer import Task  # noqa: E402


class TestCodeFixBatcher:
    def test_batches_carry_executor_role(self):
        fixes = [
            {"file": "a.py", "type": "bug", "description": "d", "importance": "MEDIUM",
             "title": "t", "line_range": [1, 2]},
        ]
        batches = batch_code_fixes(fixes)
        assert batches
        for batch in batches:
            assert batch.executor_role == "agent"
            assert batch.to_dict()["executor_role"] == "agent"
            # The legacy field survives for hosts whose Task tool needs it ...
            assert batch.to_dict()["subagent_type"] == "general-purpose"

    def test_the_two_fields_are_derived_from_one_decision(self):
        fix = {"file": "a.py", "type": "bug"}
        assert determine_executor_role(fix) == "agent"
        assert determine_subagent_type(fix) == "general-purpose"

    def test_stats_report_the_executor_distribution(self):
        fixes = [
            {"file": "a.py", "type": "bug", "description": "d", "importance": "HIGH",
             "title": "t", "line_range": [1, 2]},
            {"file": "b.py", "type": "style", "description": "d", "importance": "LOW",
             "title": "t", "line_range": [3, 4]},
        ]
        stats = estimate_batch_processing_stats(batch_code_fixes(fixes))
        assert stats["executor_distribution"] == {"agent": stats["total_batches"]}
        # ... and the historical key is still there for existing consumers.
        assert "subagent_distribution" in stats

    def test_explicit_role_survives_construction(self):
        batch = CodeFixBatch(executor_role="human", subagent_type="human")
        assert batch.to_dict()["executor_role"] == "human"


class TestTaskDecomposer:
    def test_default_task_is_an_agent_task(self):
        task = Task(id="T001", title="t", description="d")
        assert task.executor_role == "agent"
        assert task.to_dict()["executor_role"] == "agent"

    def test_legacy_human_subagent_type_becomes_a_human_role(self):
        """`human` is load-bearing and already host-neutral, so a task file
        written before the rename still routes correctly on resume."""
        task = Task.from_dict({"id": "T002", "title": "t", "subagent_type": "human"})
        assert task.executor_role == "human"

    def test_explicit_executor_role_is_authoritative(self):
        task = Task.from_dict({"id": "T003", "title": "t", "executor_role": "human"})
        assert task.executor_role == "human"
        assert task.subagent_type == "human"

    def test_the_pair_can_never_disagree(self):
        """The task schema rejects a disagreement, so the dataclass reconciles;
        `human` wins in either direction, since 'do not dispatch' is the only safe
        resolution."""
        assert Task(id="T1", title="t", description="d",
                    subagent_type="human").executor_role == "human"
        assert Task(id="T2", title="t", description="d",
                    executor_role="human").subagent_type == "human"

    def test_round_trip_through_dict(self):
        task = Task(id="T004", title="t", description="d", executor_role="human")
        assert Task.from_dict(task.to_dict()).executor_role == "human"


class TestRenderedTasksMarkdown:
    """`tasks.md` is read by every host and by humans, so it loses the agent
    type unconditionally."""

    def test_agent_task_renders_the_executor_line(self):
        markdown = generate_tasks_markdown(
            [{"id": "T001", "title": "Do it", "executor_role": "agent"}], "plan.md"
        )
        assert "**Executor**: agent" in markdown
        assert "**Subagent**" not in markdown
        assert "general-purpose" not in markdown

    def test_human_task_wording_is_unchanged(self):
        markdown = generate_tasks_markdown(
            [{"id": "T001", "title": "Make a key", "executor_role": "human"}], "plan.md"
        )
        assert "**Type**: Manual (user action required)" in markdown
        assert "**Executor**" not in markdown

    def test_legacy_records_still_render(self):
        markdown = generate_tasks_markdown(
            [{"id": "T001", "title": "x", "subagent_type": "human"},
             {"id": "T002", "title": "y", "subagent_type": "general-purpose"}],
            "plan.md",
        )
        assert "**Type**: Manual (user action required)" in markdown
        assert "**Executor**: agent" in markdown

    def test_mermaid_shapes_branch_on_the_role(self):
        diagram = generate_dependency_diagram([
            {"id": "T001", "title": "Setup", "depends_on": [],
             "estimated_complexity": "low", "executor_role": "agent"},
            {"id": "T002", "title": "Make a key", "depends_on": ["T001"],
             "estimated_complexity": "low", "executor_role": "human"},
        ])
        assert 'T001["T001: Setup"]' in diagram
        assert 'T002(["T002: Make a key"])' in diagram
        assert "fill:#87ceeb" in diagram

    @pytest.mark.parametrize(
        "task,expected",
        [
            ({"executor_role": "human"}, "human"),
            ({"executor_role": "agent"}, "agent"),
            ({"subagent_type": "human"}, "human"),
            ({"subagent_type": "general-purpose"}, "agent"),
            ({}, "agent"),
            ({"executor_role": "nonsense", "subagent_type": "human"}, "human"),
        ],
    )
    def test_task_executor_role_resolution(self, task, expected):
        assert task_executor_role(task) == expected


class TestMetrics:
    """The flag name and the JSON key stay put; only the recorded value and the
    rendered column header change."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("general-purpose", "agent"),
            ("human", "human"),
            ("claude-code:sonnet", "claude-code:sonnet"),
            ("agent", "agent"),
        ],
    )
    def test_recorded_value_normalization(self, value, expected):
        assert normalize_recorded_executor(value) == expected

    def test_record_metric_stores_the_executor_under_the_historical_key(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
        record_metric(
            state_path=state_file,
            phase="review-plan",
            label="Validation batch 1",
            subagent_type="general-purpose",
            print_eta=False,
        )
        entry = json.loads(state_file.read_text(encoding="utf-8"))["metrics"][
            "review-plan"
        ][0]
        assert entry["subagent_type"] == "agent"

    def test_executor_path_records_the_resolved_runner_spec(self, tmp_path):
        """Two invokers then read differently in --status output instead of both
        printing general-purpose."""
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
        record_metric(
            state_path=state_file,
            phase="review-plan",
            label="Validation batch 1",
            subagent_type="claude-code:sonnet",
            print_eta=False,
        )
        entry = json.loads(state_file.read_text(encoding="utf-8"))["metrics"][
            "review-plan"
        ][0]
        assert entry["subagent_type"] == "claude-code:sonnet"

    def test_the_column_header_names_the_executor(self, tmp_path):
        from utils.metrics import generate_phase_report

        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
        record_metric(
            state_path=state_file, phase="implement", label="T001",
            subagent_type="general-purpose", token_count=10, print_eta=False,
        )
        report = generate_phase_report(state_file, "implement")
        assert "| Task | Executor | Tokens | Tool Uses | Duration |" in report
        assert "Subagent Type" not in report


class TestPrintedFollowUps:
    """The stdout follow-up text stops teaching the field."""

    def test_code_review_follow_up_is_host_neutral(self):
        source = (SKILL_ROOT / "code_review_orchestrator.py").read_text(
            encoding="utf-8"
        )
        for taught in (
            "spawn a Task subagent using the 'subagent_type' field",
            "all batches use 'general-purpose' subagent_type",
        ):
            assert taught not in source
        assert "run task.prompt as a subagent" in source
        assert "write the" in source and "task.output_path" in source


class TestSalvageTaskEmission:
    def test_salvage_request_and_task_are_written_side_by_side(self, tmp_path):
        from utils.subagent_tasks import read_task_file, tasks_of

        request = {
            "model": "gemini:gemini-3-pro",
            "phase": "review_plan",
            "raw_output": "not json",
            "expected_type": "array",
            "expected_schema": {"fields": ["title"]},
            "output_path": str(tmp_path / "gemini_gemini-3-pro.json"),
            "timestamp": "2026-01-01T00:00:00",
        }
        request_path = tmp_path / "salvage_gemini_gemini-3-pro.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")

        from utils.subagent_tasks import write_salvage_task_file

        task_path = write_salvage_task_file(
            salvage_request=request,
            salvage_request_path=str(request_path),
            phase_dir=str(tmp_path),
        )
        document = read_task_file(task_path, phase_dir=str(tmp_path))
        task = tasks_of(document)[0]
        assert task["kind"] == "salvage"
        assert task["suggestion_id"] == "gemini_gemini-3-pro"
        assert task["output_name"] == "gemini_gemini-3-pro.json"
        assert task["executor_role"] == "agent"
        assert task["capability_tier"] == "full"
        assert task["inputs"]["salvage_request_sha256"]
        assert "not json" in task["prompt"]
        # The salvage task file name matches the reaggregation exclude patterns,
        # so it is never parsed as a model result.
        assert "salvage" in Path(task_path).name

    def test_the_request_hash_binds_the_prompt_to_the_request(self, tmp_path):
        from utils.subagent_tasks import (
            TaskFileError,
            build_inputs,
            read_task_file,
            write_salvage_task_file,
        )

        request = {
            "model": "m", "phase": "review_plan", "raw_output": "x",
            "expected_type": "array", "expected_schema": {},
            "output_path": str(tmp_path / "m.json"),
        }
        request_path = tmp_path / "salvage_m.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        task_path = write_salvage_task_file(
            salvage_request=request,
            salvage_request_path=str(request_path),
            phase_dir=str(tmp_path),
        )
        request_path.write_text(json.dumps({**request, "raw_output": "y"}),
                                encoding="utf-8")
        with pytest.raises(TaskFileError) as excinfo:
            read_task_file(
                task_path,
                phase_dir=str(tmp_path),
                current_inputs=build_inputs(salvage_request_file=str(request_path)),
            )
        assert "salvage_request_sha256" in str(excinfo.value)
