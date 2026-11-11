"""Tests for update_plan_tasks script."""

import json
import os
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import update_plan_tasks  # noqa: E402
from update_plan_tasks import (  # noqa: E402
    apply_tasks_document,
    generate_dependency_diagram,
    generate_tasks_markdown,
    generated_stamp,
    replace_file,
)
from utils import write_safety  # noqa: E402


def _document(tasks, preamble="Shared context for every task."):
    """A task-generation document the executor's validator accepts.

    The shared ``sample_tasks_json`` fixture predates the schema and leaves out
    keys it requires; they are filled here, never overwritten.
    """
    completed = []
    for task in tasks:
        full = {"acceptance_criteria": ["It works."], "executor_role": "agent"}
        full.update(task)
        completed.append(full)
    return {"plan_preamble": preamble, "tasks": completed}


def _write_inputs(temp_dir, document, plan_text="# My Plan\n\nSome content here.\n"):
    plan_path = temp_dir / "my-plan.md"
    plan_path.write_text(plan_text, encoding="utf-8")
    tasks_json_path = temp_dir / "tasks.json"
    tasks_json_path.write_text(json.dumps(document), encoding="utf-8")
    return plan_path, tasks_json_path


def _main(plan_path, tasks_json_path, *extra):
    return update_plan_tasks.main(
        ["--plan-file", str(plan_path), "--tasks-file", str(tasks_json_path), *extra]
    )


class TestTheCliValidatesLikeTheExecutor:
    """``util update-tasks`` applies the executor's verdict for the kind
    (``validate_result("task_generation", ...)``): the schema, then the graph
    rules. Each refusal names the rule and writes nothing."""

    @pytest.mark.parametrize("document,needle", [
        ({"items": []}, "missing required field 'tasks'"),
        ({"plan_preamble": "P.", "tasks": "not an array"}, "expected type array"),
        ({"plan_preamble": "P.", "tasks": []}, "fewer than minItems 1"),
        ([{"id": "T001", "title": "Test"}], "expected type object"),
    ])
    def test_a_malformed_document_is_refused_by_name(
            self, temp_dir, capsys, document, needle):
        plan_path, tasks_json_path = _write_inputs(temp_dir, document)

        assert _main(plan_path, tasks_json_path) == 1

        err = capsys.readouterr().err
        assert "Invalid tasks JSON" in err and needle in err, err
        assert plan_path.read_text(encoding="utf-8") == "# My Plan\n\nSome content here.\n"
        assert not (temp_dir / "my-plan").exists()

    @pytest.mark.parametrize("drop", ["id", "title", "executor_role"])
    def test_a_task_missing_a_required_key_is_refused(
            self, temp_dir, sample_tasks_json, capsys, drop):
        document = _document(sample_tasks_json)
        document["tasks"][0].pop(drop)
        plan_path, tasks_json_path = _write_inputs(temp_dir, document)

        assert _main(plan_path, tasks_json_path) == 1

        assert f"missing required field '{drop}'" in capsys.readouterr().err
        assert not (temp_dir / "my-plan").exists()

    def test_a_cyclic_list_never_reaches_the_plan(
            self, temp_dir, sample_tasks_json, capsys):
        """The weak id/title check let a cycle through to implement."""
        document = _document(sample_tasks_json)
        document["tasks"][0]["depends_on"] = ["T003"]
        plan_path, tasks_json_path = _write_inputs(temp_dir, document)

        assert _main(plan_path, tasks_json_path) == 1

        assert "cycle" in capsys.readouterr().err
        assert "TASKS_FILE" not in plan_path.read_text(encoding="utf-8")
        assert not (temp_dir / "my-plan").exists()

    def test_omitted_empty_values_are_read_as_empty_like_every_lane(
            self, temp_dir, sample_tasks_json):
        """Absent lists, ``subagent_type`` and ``schema_version`` mean the
        empty value (``normalize_result``/``supply_result_version``); the
        input file itself is not rewritten."""
        document = _document(sample_tasks_json)
        assert "schema_version" not in document
        assert "files_to_modify" not in document["tasks"][0]
        plan_path, tasks_json_path = _write_inputs(temp_dir, document)
        before = tasks_json_path.read_bytes()

        assert _main(plan_path, tasks_json_path) == 0

        assert tasks_json_path.read_bytes() == before
        assert (temp_dir / "my-plan" / "tasks" / "tasks.md").exists()

    def test_the_function_raises_where_main_returns_one(self, temp_dir):
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n", encoding="utf-8")

        with pytest.raises(update_plan_tasks.TasksDocumentInvalid) as raised:
            apply_tasks_document(plan_path, {"tasks": []})
        assert raised.value.errors
        with pytest.raises(update_plan_tasks.TasksDocumentError, match="not found"):
            apply_tasks_document(temp_dir / "missing.md", {"tasks": []})


class TestGenerateTasksMarkdown:
    """Tests for generate_tasks_markdown function."""

    def test_basic_task_generation(self, sample_tasks_json):
        """Test basic markdown generation."""
        result = generate_tasks_markdown(sample_tasks_json, "my-plan.md")

        assert "# Implementation Tasks for my-plan.md" in result
        assert "**Parent Plan**: [my-plan.md](../my-plan.md)" in result
        assert "### Task T001: Create directory structure" in result
        assert "### Task T002: Implement core module" in result

    def test_dependencies_displayed(self, sample_tasks_json):
        """Test that dependencies are properly shown."""
        result = generate_tasks_markdown(sample_tasks_json, "plan.md")

        # T001 has no dependencies
        assert "**Dependencies**: None" in result
        # T002 depends on T001
        assert "**Dependencies**: T001" in result

    def test_files_displayed(self, sample_tasks_json):
        """Test that files to create/modify are shown."""
        result = generate_tasks_markdown(sample_tasks_json, "plan.md")

        assert "**Files to create**: src/main.py" in result
        assert "**Files to modify**: src/main.py" in result

    def test_a_task_that_declares_no_files_still_renders_both_lines(self):
        """An empty file list is rendered as `none`, never as a missing line.

        `task_decomposer._explicit_files` reads "neither line present" as
        "hand-written task, guess from the narrative", so an omitted line is
        the one thing that hands a *generated* task to the keyword heuristic
        (audited 2026-09-16 on T014 of a 14-task list).
        """
        tasks = [{
            "id": "T001",
            "title": "Ask the user to create an API token",
            "description": "Manual step: no files change.",
            "depends_on": [],
            "files_to_create": [],
            "files_to_modify": [],
            "estimated_complexity": "low",
        }]

        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Files to modify**: none" in result
        assert "**Files to create**: none" in result

    def test_an_empty_file_list_round_trips_as_empty_not_as_a_guess(self):
        """The rendered `none` parses back to `[]` — the heuristic never runs."""
        from utils.task_decomposer import TaskDecomposer

        tasks = [{
            "id": "T001",
            "title": "Update the HUD interval",
            # Every word here the keyword heuristic would mine for a file.
            "description": "Create and update `CONFIG.hud.updateInterval` "
                           "and `hud.update` to add a new readout.",
            "depends_on": [],
            "files_to_create": [],
            "files_to_modify": [],
            "estimated_complexity": "low",
        }]

        parsed = TaskDecomposer().parse_plan(generate_tasks_markdown(tasks, "plan.md"))

        assert [t.files_to_create for t in parsed] == [[]]
        assert [t.files_to_modify for t in parsed] == [[]]

    def test_complexity_displayed(self, sample_tasks_json):
        """Test that complexity is shown."""
        result = generate_tasks_markdown(sample_tasks_json, "plan.md")

        assert "**Complexity**: low" in result
        assert "**Complexity**: medium" in result

    def test_the_declared_complexity_survives_the_round_trip(self, sample_tasks_json):
        """Render then re-parse: the decomposer reads the line it was given.

        It was the only field of the JSON → markdown → JSON round-trip that did
        not survive — all 14 tasks of an audited `tasks.md` came back `high`
        from the keyword heuristic, contradicting the colours of the mermaid
        graph rendered five lines above them.
        """
        from utils.task_decomposer import TaskDecomposer

        parsed = TaskDecomposer().parse_plan(
            generate_tasks_markdown(sample_tasks_json, "plan.md")
        )

        assert [t.estimated_complexity for t in parsed] == ["low", "medium", "medium"]

    def test_executor_role_displayed(self):
        """tasks.md names the host-neutral executor role, not a Claude agent type."""
        tasks = [{
            "id": "T001",
            "title": "Database migration",
            "executor_role": "agent"
        }]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Executor**: agent" in result
        assert "general-purpose" not in result

    def test_legacy_subagent_type_still_renders_as_executor(self):
        """A record written before the D2h rename renders through its legacy field."""
        tasks = [{
            "id": "T001",
            "title": "Database migration",
            "subagent_type": "general-purpose"
        }]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Executor**: agent" in result
        assert "**Subagent**" not in result

    def test_acceptance_criteria_displayed(self):
        """Test that acceptance criteria are shown as checkboxes."""
        tasks = [{
            "id": "T001",
            "title": "Test task",
            "acceptance_criteria": ["Criterion 1", "Criterion 2"]
        }]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Acceptance Criteria**:" in result
        assert "- [ ] Criterion 1" in result
        assert "- [ ] Criterion 2" in result

    def test_description_displayed(self):
        """Test that description is included."""
        tasks = [{
            "id": "T001",
            "title": "Test task",
            "description": "This is the task description."
        }]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "This is the task description." in result


class _FrozenNow(datetime):
    """A local wall clock that reads one known instant."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 3, 4, 5, 6, 7)


class TestGeneratedStamp:
    """The `*Generated: …*` line honours SOURCE_DATE_EPOCH, and only it.

    The stamp is hashed into every digest of tasks.md, so without the pin no two
    runs agree on one; the container rig's golden baselines depend on it.
    """

    @pytest.fixture
    def frozen(self, monkeypatch):
        monkeypatch.setattr(update_plan_tasks, "datetime", _FrozenNow)

    def test_unset_is_the_local_wall_clock_as_before(self, frozen, monkeypatch):
        monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
        assert generated_stamp() == "2026-03-04 05:06:07"
        lines = generate_tasks_markdown([], "plan.md").splitlines()
        assert lines[2] == "*Generated: 2026-03-04 05:06:07*"

    def test_empty_counts_as_unset(self, frozen, monkeypatch):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "")
        assert generated_stamp() == "2026-03-04 05:06:07"

    @pytest.mark.skipif(not hasattr(time, "tzset"), reason="POSIX-only tzset")
    def test_set_renders_that_instant_in_utc(self, frozen, monkeypatch):
        # A non-UTC local zone, so a local-time rendering would be caught.
        monkeypatch.setenv("TZ", "Asia/Tokyo")
        time.tzset()
        try:
            monkeypatch.setenv("SOURCE_DATE_EPOCH", "1767225600")
            assert generated_stamp() == "2026-01-01 00:00:00"
            one = generate_tasks_markdown([], "plan.md")
            assert one.splitlines()[2] == "*Generated: 2026-01-01 00:00:00*"
            assert generate_tasks_markdown([], "plan.md") == one
        finally:
            monkeypatch.undo()
            time.tzset()

    @pytest.mark.parametrize("raw", [
        "abc", "1.5", "-1", " 12", "1_000", "+5", "99999999999999999999",
    ])
    def test_a_malformed_value_raises_rather_than_falling_back(
            self, monkeypatch, raw):
        """Only something asking for determinism sets it; a silent wall-clock
        fallback would hand it a non-repeatable file and no reason why."""
        monkeypatch.setenv("SOURCE_DATE_EPOCH", raw)
        with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
            generated_stamp()

    def test_a_malformed_value_fails_the_script_before_any_write(
            self, monkeypatch, temp_dir, sample_tasks_json, capsys):
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n", encoding="utf-8")
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(
            json.dumps(_document(sample_tasks_json)), encoding="utf-8")
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "yesterday")

        code = update_plan_tasks.main(["--plan-file", str(plan_path),
                                       "--tasks-file", str(tasks_json_path)])

        assert code == 1
        assert "SOURCE_DATE_EPOCH" in capsys.readouterr().err
        assert not (temp_dir / "my-plan").exists()
        assert plan_path.read_text(encoding="utf-8") == "# My Plan\n"


class TestThePlanUpdateSaysWhatItWrote:
    """OpenCode racing run 4 (03): a bare "Updated plan file: <plan>" read as
    generate-tasks rewriting the plan. It adds (or refreshes) the plan's
    "Implementation Tasks" section and its TASKS_FILE marker, which is how
    every later phase finds the task list, and the line says so."""

    def test_the_line_names_the_section_and_the_task_list(
            self, temp_dir, sample_tasks_json, capsys):
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nBody.\n", encoding="utf-8")
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(
            json.dumps(_document(sample_tasks_json)), encoding="utf-8")

        code = update_plan_tasks.main(["--plan-file", str(plan_path),
                                       "--tasks-file", str(tasks_json_path)])

        assert code == 0
        out = capsys.readouterr().out
        assert (f"Updated plan file: {plan_path.resolve()} (its \"Implementation "
                "Tasks\" section now links my-plan/tasks/tasks.md, the task list "
                "later phases read)") in out
        text = plan_path.read_text(encoding="utf-8")
        assert text.startswith("# My Plan\n\nBody.\n")
        assert "<!-- TASKS_FILE: my-plan/tasks/tasks.md -->" in text


class TestGenerateDependencyDiagram:
    """Tests for generate_dependency_diagram function."""

    def test_nodes_and_edges(self, sample_tasks_json):
        """Test diagram has correct nodes and edges."""
        result = generate_dependency_diagram(sample_tasks_json)

        assert "```mermaid" in result
        assert "graph TD" in result
        assert 'T001["T001: Create directory structure"]' in result
        assert 'T002["T002: Implement core module"]' in result
        assert 'T003["T003: Add tests"]' in result
        assert "T001 --> T002" in result
        assert "T002 --> T003" in result

    def test_no_incoming_edges_for_root_tasks(self, sample_tasks_json):
        """Test that tasks with no dependencies have no incoming edges."""
        result = generate_dependency_diagram(sample_tasks_json)

        # T001 has no dependencies, so nothing should point to it
        assert "--> T001" not in result

    def test_title_truncation(self):
        """Test that long titles are truncated to ~40 chars."""
        tasks = [{
            "id": "T001",
            "title": "A" * 50,
            "depends_on": [],
            "estimated_complexity": "low"
        }]
        result = generate_dependency_diagram(tasks)

        assert 'T001["T001: ' + "A" * 37 + '..."]' in result

    def test_short_title_not_truncated(self):
        """Test that short titles are not truncated."""
        tasks = [{
            "id": "T001",
            "title": "Short title",
            "depends_on": [],
            "estimated_complexity": "low"
        }]
        result = generate_dependency_diagram(tasks)

        assert 'T001["T001: Short title"]' in result

    def test_complexity_styling(self, sample_tasks_json):
        """Test that nodes are styled by complexity."""
        result = generate_dependency_diagram(sample_tasks_json)

        assert "style T001 fill:#90ee90" in result  # low = green
        assert "style T002 fill:#f4a460" in result  # medium = orange
        assert "style T003 fill:#f4a460" in result  # medium = orange

    def test_high_complexity_styling(self):
        """Test high complexity gets red color."""
        tasks = [{
            "id": "T001",
            "title": "Complex task",
            "depends_on": [],
            "estimated_complexity": "high"
        }]
        result = generate_dependency_diagram(tasks)

        assert "style T001 fill:#cd5c5c" in result

    def test_empty_depends_on(self):
        """Test tasks with empty depends_on arrays produce no edges."""
        tasks = [
            {"id": "T001", "title": "Task A", "depends_on": [], "estimated_complexity": "low"},
            {"id": "T002", "title": "Task B", "depends_on": [], "estimated_complexity": "low"},
        ]
        result = generate_dependency_diagram(tasks)

        assert "-->" not in result

    def test_quotes_in_title_escaped(self):
        """Test that quotes in titles are escaped for Mermaid."""
        tasks = [{
            "id": "T001",
            "title": 'Use "quotes" here',
            "depends_on": [],
            "estimated_complexity": "low"
        }]
        result = generate_dependency_diagram(tasks)

        assert 'Use "quotes"' not in result
        assert "Use #quot;quotes#quot; here" in result

    def test_diagram_included_in_tasks_markdown(self, sample_tasks_json):
        """Test that the diagram is included in the full markdown output."""
        result = generate_tasks_markdown(sample_tasks_json, "plan.md")

        assert "## Dependency Graph" in result
        assert "```mermaid" in result
        assert "graph TD" in result

    def test_diagram_not_included_for_single_task(self):
        """Test that the diagram is omitted when there's only one task."""
        tasks = [{"id": "T001", "title": "Only task", "depends_on": []}]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "## Dependency Graph" not in result
        assert "```mermaid" not in result


class TestUpdatePlanTasksScript:
    """Integration tests for the update_plan_tasks.py script."""

    def test_script_creates_tasks_file(self, temp_dir, sample_tasks_json):
        """Test that the script creates the tasks file."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nSome content here.\n", encoding="utf-8")

        # Create tasks JSON
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps(_document(sample_tasks_json)), encoding="utf-8")

        # Run script
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check tasks file was created
        tasks_file = temp_dir / "my-plan" / "tasks" / "tasks.md"
        assert tasks_file.exists(), f"Tasks file not created at {tasks_file}"

        # Check content
        tasks_content = tasks_file.read_text(encoding="utf-8")
        assert "### Task T001" in tasks_content
        assert "### Task T002" in tasks_content

    def test_script_updates_plan_file(self, temp_dir, sample_tasks_json):
        """Test that the script updates the plan with tasks reference."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nSome content here.\n", encoding="utf-8")

        # Create tasks JSON
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps(_document(sample_tasks_json)), encoding="utf-8")

        # Run script
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check plan was updated
        plan_content = plan_path.read_text(encoding="utf-8")
        assert "<!-- TASKS_FILE:" in plan_content
        assert "## Implementation Tasks" in plan_content

    def test_script_dry_run(self, temp_dir, sample_tasks_json):
        """Test that dry-run mode doesn't modify files."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        original_content = "# My Plan\n\nSome content here.\n"
        plan_path.write_text(original_content, encoding="utf-8")

        # Create tasks JSON
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps(_document(sample_tasks_json)), encoding="utf-8")

        # Run script with --dry-run
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path),
             "--dry-run"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 0
        assert "Dry run" in result.stdout

        # Check nothing was modified
        assert plan_path.read_text(encoding="utf-8") == original_content
        tasks_file = temp_dir / "my-plan" / "tasks" / "tasks.md"
        assert not tasks_file.exists()

    def test_script_missing_plan_file(self, temp_dir, sample_tasks_json):
        """Test error handling when plan file doesn't exist."""
        plan_path = temp_dir / "nonexistent.md"
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps(_document(sample_tasks_json)), encoding="utf-8")

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 1
        assert "Plan file not found" in result.stderr

    def test_script_missing_tasks_file(self, temp_dir):
        """Test error handling when tasks file doesn't exist."""
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        tasks_json_path = temp_dir / "nonexistent.json"

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 1
        assert "Tasks JSON file not found" in result.stderr

    def test_script_invalid_json(self, temp_dir):
        """Test error handling with invalid JSON."""
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text("not valid json {", encoding="utf-8")

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 1
        assert "Failed to parse" in result.stderr

    def test_script_invalid_tasks_structure(self, temp_dir):
        """Test error handling with invalid tasks structure."""
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# Plan\n", encoding="utf-8")
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps({"tasks": []}), encoding="utf-8")

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 1
        assert "Invalid tasks JSON" in result.stderr


class TestThePluginsOwnPlanEdit:
    """The marker this script inserts is the plugin's edit, not a human's.

    `StateManager._load_or_create` reads any hash drift as "someone edited the
    plan mid-workflow", and this script used to construct its StateManager
    *after* rewriting the plan — so the run recorded `plan_changed: True` about
    itself. Audited 2026-09-16 on a clean state.json: the next
    `implement --resume` warns "Plan has changed since last session" for a
    change generate-tasks made one phase earlier.
    """

    def _run(self, temp_dir, sample_tasks_json):
        from utils.state_manager import StateManager

        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nSome content here.\n", encoding="utf-8")
        # An EXISTING state.json, stamped with the plan as it stands — what
        # every real generate-tasks run inherits from the review phases before
        # it, and the only condition under which `_load_or_create` compares
        # hashes at all.
        StateManager(plan_path).save()
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(
            json.dumps(_document(sample_tasks_json, preamble="P.")),
            encoding="utf-8",
        )
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        return plan_path

    def test_the_marker_insertion_is_not_recorded_as_a_human_edit(
        self, temp_dir, sample_tasks_json
    ):
        self._run(temp_dir, sample_tasks_json)

        state_data = json.loads(
            (temp_dir / "my-plan" / "state.json").read_text(encoding="utf-8")
        )
        assert state_data.get("plan_changed", False) is False

    def test_the_stamped_hash_matches_the_plan_as_written(
        self, temp_dir, sample_tasks_json
    ):
        """The other half: a stale hash would make the *next* reader flag it."""
        from utils.state_manager import StateManager

        plan_path = self._run(temp_dir, sample_tasks_json)

        state_data = json.loads(
            (temp_dir / "my-plan" / "state.json").read_text(encoding="utf-8")
        )
        # Constructed fresh, the way implement's resume path does it.
        assert state_data["plan_hash"] == StateManager(plan_path)._compute_plan_hash()
        assert StateManager(plan_path).has_plan_changed() is False


class TestPreambleStorage:
    """Tests for plan_preamble storage functionality."""

    def test_script_stores_preamble_in_state(self, temp_dir, sample_tasks_json):
        """Test that the script stores plan_preamble in state.json."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nSome content here.\n", encoding="utf-8")

        # Create tasks JSON with preamble
        tasks_data = _document(
            sample_tasks_json,
            preamble="This is a test preamble describing the plan goals and architecture.",
        )
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps(tasks_data), encoding="utf-8")

        # Run script
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Stored plan preamble" in result.stdout

        # Check state.json was created and contains preamble
        state_path = temp_dir / "my-plan" / "state.json"
        assert state_path.exists(), f"State file not created at {state_path}"

        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        assert "plan_preamble" in state_data
        assert state_data["plan_preamble"] == "This is a test preamble describing the plan goals and architecture."

    @pytest.mark.parametrize("preamble", [None, ""])
    def test_a_missing_or_empty_preamble_is_refused(
            self, temp_dir, sample_tasks_json, capsys, preamble):
        """The schema requires a non-empty preamble: without it every implement
        prompt silently loses its shared context. Generate-tasks never wrote
        one without it; ``util update-tasks`` now refuses it the same way."""
        document = _document(sample_tasks_json, preamble=preamble)
        if preamble is None:
            document.pop("plan_preamble")
        plan_path, tasks_json_path = _write_inputs(temp_dir, document)

        assert _main(plan_path, tasks_json_path) == 1

        assert "plan_preamble" in capsys.readouterr().err
        assert not (temp_dir / "my-plan").exists()


class TestThePlanWriteIsAtomic:
    """Both writes go through a sibling temp file and ``os.replace``: a write
    that fails part-way leaves the old bytes, never a truncated plan."""

    class _Torn:
        """A stream that writes half its data, then fails like a full disk."""

        def __init__(self, stream):
            self._stream = stream

        def write(self, data):
            self._stream.write(data[: len(data) // 2])
            self._stream.flush()
            raise OSError(28, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._stream.close()

        def __getattr__(self, name):
            return getattr(self._stream, name)

    def _tear_write(self, monkeypatch, which):
        """Tear the *which*-th (1-based) file write; earlier ones succeed."""
        real_fdopen = os.fdopen
        calls = []

        def fdopen(fd, *args, **kwargs):
            stream = real_fdopen(fd, *args, **kwargs)
            calls.append(fd)
            return self._Torn(stream) if len(calls) == which else stream

        monkeypatch.setattr(update_plan_tasks.os, "fdopen", fdopen)

    def test_a_failure_mid_write_leaves_the_plan_intact(
            self, temp_dir, sample_tasks_json, monkeypatch):
        plan_path, _ = _write_inputs(temp_dir, {})
        before = plan_path.read_bytes()
        # tasks.md first, then the plan: the plan's is the first stream where
        # tasks.md goes in by descriptor through its held folder.
        self._tear_write(
            monkeypatch, which=1 if write_safety._PUBLISHES_BY_DESCRIPTOR else 2)

        with pytest.raises(OSError, match="No space left"):
            apply_tasks_document(plan_path, _document(sample_tasks_json))
        monkeypatch.undo()

        assert plan_path.read_bytes() == before
        assert sorted(p.name for p in temp_dir.iterdir()) == [
            "my-plan", "my-plan.md", "tasks.json",
        ], "no temp file is left beside the plan"

    def test_replace_file_keeps_the_old_bytes_on_a_torn_write(
            self, temp_dir, monkeypatch):
        target = temp_dir / "plan.md"
        target.write_bytes(b"old bytes\n")
        self._tear_write(monkeypatch, which=1)

        with pytest.raises(OSError):
            replace_file(target, "new text that is never fully written\n")
        monkeypatch.undo()

        assert target.read_bytes() == b"old bytes\n"
        assert [p.name for p in temp_dir.iterdir()] == ["plan.md"]

    def test_text_is_written_like_write_text_and_bytes_exactly(self, temp_dir):
        target = temp_dir / "plan.md"
        replace_file(target, "a\nb\n")
        expected = temp_dir / "expected.md"
        expected.write_text("a\nb\n", encoding="utf-8")
        assert target.read_bytes() == expected.read_bytes()
        replace_file(target, b"a\r\nb")
        assert target.read_bytes() == b"a\r\nb"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits and symlinks")
    def test_mode_bits_and_a_symlinked_plan_survive(
            self, temp_dir, sample_tasks_json):
        real = temp_dir / "real" / "my-plan.md"
        real.parent.mkdir()
        real.write_text("# My Plan\n", encoding="utf-8")
        real.chmod(0o640)
        link = temp_dir / "my-plan.md"
        link.symlink_to(real)

        replace_file(link, "# My Plan\n\nEdited.\n")

        assert link.is_symlink() and os.readlink(link) == str(real)
        assert real.read_text(encoding="utf-8") == "# My Plan\n\nEdited.\n"
        assert stat.S_IMODE(real.stat().st_mode) == 0o640

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
    def test_a_private_file_is_never_readable_more_widely_while_replaced(
            self, temp_dir, monkeypatch):
        target = temp_dir / "plan.md"
        target.write_text("old\n", encoding="utf-8")
        target.chmod(0o600)
        seen = []
        real_fsync = os.fsync

        def watching_fsync(descriptor):
            info = os.fstat(descriptor)
            if stat.S_ISREG(info.st_mode):
                seen.append(stat.S_IMODE(info.st_mode))
            return real_fsync(descriptor)

        previous = os.umask(0o022)
        try:
            monkeypatch.setattr(os, "fsync", watching_fsync)
            replace_file(target, "private\n")
        finally:
            os.umask(previous)
        assert seen == [0o600]
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert target.read_text(encoding="utf-8") == "private\n"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync")
    def test_the_directory_is_flushed_after_the_rename(self, temp_dir, monkeypatch):
        target = temp_dir / "plan.md"
        target.write_text("old\n", encoding="utf-8")
        events = []
        real_fsync, real_replace = os.fsync, os.replace

        def recording_fsync(descriptor):
            kind = "dir" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
            events.append(kind)
            return real_fsync(descriptor)

        def recording_replace(source, destination):
            events.append("replace")
            return real_replace(source, destination)

        monkeypatch.setattr(os, "fsync", recording_fsync)
        monkeypatch.setattr(os, "replace", recording_replace)
        replace_file(target, "new\n")
        assert events == ["file", "replace", "dir"]


class TestHumanTaskRendering:
    """Tests for human task visual distinction in markdown and Mermaid."""

    def test_human_task_markdown_shows_manual_type(self):
        """Test that human tasks render as 'Type: Manual' instead of 'Subagent: human'."""
        tasks = [{
            "id": "T001",
            "title": "Create API key",
            "description": "Create an API key in the external dashboard.",
            "subagent_type": "human",
            "estimated_complexity": "low",
        }]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Type**: Manual (user action required)" in result
        assert "**Subagent**: human" not in result

    def test_non_human_task_shows_executor(self):
        """Test that non-human tasks render with the Executor label."""
        tasks = [{
            "id": "T001",
            "title": "Implement feature",
            "subagent_type": "general-purpose",
        }]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Executor**: agent" in result
        assert "**Type**: Manual" not in result

    def test_human_task_mermaid_stadium_shape(self):
        """Test that human tasks use stadium shape ([...]) in Mermaid diagram."""
        tasks = [
            {"id": "T001", "title": "Setup code", "depends_on": [], "estimated_complexity": "low", "subagent_type": "general-purpose"},
            {"id": "T002", "title": "Create API key", "depends_on": ["T001"], "estimated_complexity": "low", "subagent_type": "human"},
        ]
        result = generate_dependency_diagram(tasks)

        # T001 should use rectangle shape
        assert 'T001["T001: Setup code"]' in result
        # T002 should use stadium shape
        assert 'T002(["T002: Create API key"])' in result

    def test_human_task_mermaid_light_blue_color(self):
        """Test that human tasks use light blue color in Mermaid diagram."""
        tasks = [
            {"id": "T001", "title": "Code task", "depends_on": [], "estimated_complexity": "high", "subagent_type": "general-purpose"},
            {"id": "T002", "title": "Manual step", "depends_on": [], "estimated_complexity": "low", "subagent_type": "human"},
        ]
        result = generate_dependency_diagram(tasks)

        # T001 should use complexity color (high = red)
        assert "style T001 fill:#cd5c5c" in result
        # T002 should use light blue regardless of complexity
        assert "style T002 fill:#87ceeb" in result

    def test_mixed_human_and_code_tasks_markdown(self):
        """Test markdown output with both human and code tasks."""
        tasks = [
            {"id": "T001", "title": "Setup DB", "subagent_type": "general-purpose"},
            {"id": "T002", "title": "Create API key", "subagent_type": "human"},
            {"id": "T003", "title": "Implement feature", "subagent_type": "general-purpose"},
        ]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Executor**: agent" in result
        assert "**Type**: Manual (user action required)" in result


class TestInPlanTaskSections:
    """Issue 6: apply-suggestions wrote a task list into the plan, then generate-tasks
    wrote a second one. The script cannot merge them; it names the first."""

    def test_finds_the_headings_the_run_produced(self):
        from update_plan_tasks import find_in_plan_task_sections

        plan = (
            "# Plan\n\n## Design\n\ntext\n\n### Tasks\n\n1. a\n\n"
            "## Implementation tasks\n\n- b\n\n"
            "### Implementation tasks with dependencies\n\n- c\n"
        )
        assert find_in_plan_task_sections(plan) == [
            "### Tasks", "## Implementation tasks",
            "### Implementation tasks with dependencies",
        ]

    def test_its_own_pointer_section_is_not_a_task_list(self):
        from update_plan_tasks import find_in_plan_task_sections

        plan = (
            "# Plan\n\n## Implementation Tasks\n\nSee [t](t.md).\n\n"
            "<!-- TASKS_FILE: plans/p/tasks/tasks.md -->\n"
        )
        assert find_in_plan_task_sections(plan) == []

    def test_ordinary_headings_are_not_flagged(self):
        from update_plan_tasks import find_in_plan_task_sections

        plan = "# Plan\n\n## Subtasks of the design\n\n### Task 3: API\n\n## Testing\n"
        assert find_in_plan_task_sections(plan) == []


def _tree(root):
    """Every path under *root*: a file's bytes, inode and mtime, a link's
    target text, ``None`` for a folder. Equal trees mean nothing was written
    in there, not even bytes put back afterwards."""
    found = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            found[str(path)] = ("link", os.readlink(path))
        elif path.is_dir():
            found[str(path)] = None
        else:
            info = path.stat()
            found[str(path)] = (path.read_bytes(), info.st_ino, info.st_mtime_ns)
    return found


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs privileges on Windows")
class TestTheTaskListStaysInThePlansDirectory:
    """The task list is written through a symlink only while the link leads
    somewhere inside the plan's directory: a repository link named like the
    output (the list, its folder, the plan's output folder) that leads out is
    refused before anything is written, and the plan and its state stay as
    they were (docs/development/write-safety.md#plan-output-containment). The
    plan itself may still be linked in from anywhere."""

    @pytest.fixture
    def world(self, temp_dir):
        repo = temp_dir / "repo"
        (repo / ".git").mkdir(parents=True)
        outside = temp_dir / "outside"
        outside.mkdir()
        (outside / "victim.md").write_text("precious\n", encoding="utf-8")
        plan = repo / "my-plan.md"
        plan.write_text("# My Plan\n\nSome content here.\n", encoding="utf-8")
        return repo, outside, plan

    @staticmethod
    def _plant(repo, outside, where):
        output = repo / "my-plan"
        if where == "the list":
            (output / "tasks").mkdir(parents=True)
            (output / "tasks" / "tasks.md").symlink_to(outside / "victim.md")
        elif where == "a dangling list":
            (output / "tasks").mkdir(parents=True)
            (output / "tasks" / "tasks.md").symlink_to(outside / "new.md")
        elif where == "its folder":
            output.mkdir()
            (output / "tasks").symlink_to(outside, target_is_directory=True)
        else:  # the plan's output folder
            output.symlink_to(outside, target_is_directory=True)

    @pytest.mark.parametrize("where", [
        "the list", "a dangling list", "its folder", "the output folder",
    ])
    def test_a_link_out_of_the_repository_is_refused_and_nothing_is_written(
            self, world, sample_tasks_json, where, capsys):
        repo, outside, plan = world
        self._plant(repo, outside, where)
        tasks_json = repo.parent / "tasks.json"
        tasks_json.write_text(json.dumps(_document(sample_tasks_json)), encoding="utf-8")
        plan_before = plan.read_bytes()
        outside_before = _tree(outside)
        repo_before = _tree(repo)

        assert _main(plan, tasks_json) == 1

        err = capsys.readouterr().err
        assert "Refusing the task list" in err
        assert "nothing was written" in err
        assert _tree(outside) == outside_before, "a write landed outside the repository"
        assert plan.read_bytes() == plan_before
        assert _tree(repo) == repo_before, "no state.json, no folder, no marker"

    def test_a_dry_run_reports_the_same_refusal_and_creates_no_folder(
            self, world, sample_tasks_json):
        repo, outside, plan = world
        self._plant(repo, outside, "the output folder")
        outside_before = _tree(outside)

        for dry_run in (True, False):
            with pytest.raises(update_plan_tasks.TasksOutputEscape) as refused:
                apply_tasks_document(
                    plan, _document(sample_tasks_json), dry_run=dry_run)
            assert isinstance(refused.value, update_plan_tasks.TasksDocumentError)
        assert _tree(outside) == outside_before, "not even the tasks/ folder"

    def test_a_link_inside_the_plans_directory_is_written_through(
            self, world, sample_tasks_json):
        repo, _outside, plan = world
        shared = repo / "shared" / "my-tasks.md"
        shared.parent.mkdir()
        (repo / "my-plan" / "tasks").mkdir(parents=True)
        link = repo / "my-plan" / "tasks" / "tasks.md"
        link.symlink_to(Path("..") / ".." / "shared" / "my-tasks.md")
        tasks_json = repo.parent / "tasks.json"
        tasks_json.write_text(json.dumps(_document(sample_tasks_json)), encoding="utf-8")

        assert _main(plan, tasks_json) == 0

        assert link.is_symlink(), "the link survives"
        assert "### Task T001" in shared.read_text(encoding="utf-8")
        assert "<!-- TASKS_FILE: my-plan/tasks/tasks.md -->" in plan.read_text(
            encoding="utf-8")
        from utils.tasks_file import resolve_tasks_marker
        assert resolve_tasks_marker(plan) == shared.resolve(), (
            "implement reads the list the link leads to")

    def test_a_link_elsewhere_in_the_repository_is_refused_where_implement_would_refuse_it(
            self, temp_dir, sample_tasks_json):
        """The boundary is the plan's directory, the one implement reads the
        list from (``tasks_file.resolve_tasks_marker``): a list written
        elsewhere in the repository is one implement then refuses, and the
        link could name any tracked file."""
        repo = temp_dir / "repo"
        (repo / ".git").mkdir(parents=True)
        source = repo / "src" / "main.py"
        source.parent.mkdir()
        source.write_text("print('keep me')\n", encoding="utf-8")
        plan = repo / "plans" / "my-plan.md"
        plan.parent.mkdir()
        plan.write_text("# My Plan\n", encoding="utf-8")
        (plan.parent / "my-plan" / "tasks").mkdir(parents=True)
        (plan.parent / "my-plan" / "tasks" / "tasks.md").symlink_to(source)

        with pytest.raises(update_plan_tasks.TasksOutputEscape):
            apply_tasks_document(plan, _document(sample_tasks_json))

        assert source.read_text(encoding="utf-8") == "print('keep me')\n"
        from utils.tasks_file import TasksFileRefused, _resolve_one
        with pytest.raises(TasksFileRefused):
            _resolve_one(str(plan.parent), "my-plan/tasks/tasks.md")

    def test_a_plan_linked_in_from_outside_the_repository_still_works(
            self, world, sample_tasks_json):
        """The plan is the user's own file, written through its link wherever
        that leads; its outputs stay beside the name it is reached by."""
        repo, outside, _plan = world
        real = outside / "plans" / "linked-plan.md"
        real.parent.mkdir()
        real.write_text("# Linked\n\nBody.\n", encoding="utf-8")
        link = repo / "linked-plan.md"
        link.symlink_to(real)

        # In-process, as generate-tasks calls it: the link's own path.
        apply_tasks_document(link, _document(sample_tasks_json))
        assert link.is_symlink() and link.resolve() == real.resolve()
        assert "<!-- TASKS_FILE: linked-plan/tasks/tasks.md -->" in real.read_text(
            encoding="utf-8")
        assert (repo / "linked-plan" / "tasks" / "tasks.md").is_file()

        # `util update-tasks` resolves the plan first: the outputs go beside it.
        real.write_text("# Linked\n\nBody.\n", encoding="utf-8")
        tasks_json = repo.parent / "tasks.json"
        tasks_json.write_text(json.dumps(_document(sample_tasks_json)), encoding="utf-8")
        assert _main(link, tasks_json) == 0
        assert link.is_symlink()
        assert (real.parent / "linked-plan" / "tasks" / "tasks.md").is_file()
        assert "<!-- TASKS_FILE:" in real.read_text(encoding="utf-8")

    def test_a_looping_link_fails_the_write_alike_on_every_python(
            self, world, sample_tasks_json):
        """The check resolves the way the write does (``os.path.realpath``),
        so a link that leads nowhere is the write's own ``OSError`` on every
        Python, the plan untouched, never a ``RuntimeError`` from the check
        (``Path.resolve`` raised one on a loop before 3.13)."""
        repo, _outside, plan = world
        (repo / "my-plan" / "tasks").mkdir(parents=True)
        link = repo / "my-plan" / "tasks" / "tasks.md"
        link.symlink_to("tasks.md")
        plan_before = plan.read_bytes()

        with pytest.raises(OSError):
            apply_tasks_document(plan, _document(sample_tasks_json))

        assert link.is_symlink() and os.readlink(link) == "tasks.md"
        assert plan.read_bytes() == plan_before

    def test_a_colon_in_the_plans_directory_is_not_an_escape(
            self, temp_dir, sample_tasks_json):
        """The check is for paths the plugin computed, not ones a model
        supplied: a POSIX directory name may hold a colon, which the
        task-path check (``contain_under_root``) reads as an NTFS stream."""
        repo = temp_dir / "proj:2"
        (repo / ".git").mkdir(parents=True)
        plan = repo / "my-plan.md"
        plan.write_text("# My Plan\n", encoding="utf-8")

        apply_tasks_document(plan, _document(sample_tasks_json))

        assert (repo / "my-plan" / "tasks" / "tasks.md").is_file()


@pytest.mark.skipif(not write_safety._PUBLISHES_BY_DESCRIPTOR,
                    reason="the held-folder write is Linux and macOS; Windows writes by path")
class TestALinkSwappedInAfterTheCheckIsNeverWrittenThrough:
    """A check of the task list's path followed by a write by that path can be
    redirected in between: a link swapped in for the list, or for its folder,
    after the containment check and before the write (Codex's review of U86a,
    2026-11-09). The list is written through its folder, held below the
    plan's directory and reached without following a link, so the link is
    refused or replaced, never followed: the file outside, the plan and its
    state stay as they were
    (docs/development/write-safety.md#plan-output-containment)."""

    @pytest.fixture
    def world(self, temp_dir):
        repo = temp_dir / "repo"
        (repo / ".git").mkdir(parents=True)
        outside = temp_dir / "outside"
        outside.mkdir()
        (outside / "tasks.md").write_text("precious\n", encoding="utf-8")
        plan = repo / "my-plan.md"
        plan.write_text("# My Plan\n\nSome content here.\n", encoding="utf-8")
        folder = repo / "my-plan" / "tasks"
        folder.mkdir(parents=True)
        (folder / "tasks.md").write_text("the old list\n", encoding="utf-8")
        return repo, outside, plan, folder

    @staticmethod
    def _swap(folder, outside, swap):
        if swap == "the list":
            (folder / "tasks.md").unlink()
            (folder / "tasks.md").symlink_to(outside / "tasks.md")
        else:
            folder.rename(folder.with_name("saved"))
            folder.symlink_to(outside, target_is_directory=True)

    @pytest.mark.parametrize("swap", ["the list", "its folder"])
    def test_a_link_swapped_in_just_before_the_write_is_refused(
            self, world, sample_tasks_json, monkeypatch, swap):
        """The seam: the containment check returns, then the swap."""
        repo, outside, plan, folder = world
        real_check = update_plan_tasks.contain_plan_output

        def check_then_swap(*args, **kwargs):
            checked = real_check(*args, **kwargs)
            self._swap(folder, outside, swap)
            return checked

        monkeypatch.setattr(update_plan_tasks, "contain_plan_output", check_then_swap)
        outside_before, plan_before = _tree(outside), plan.read_bytes()

        with pytest.raises(OSError, match="nothing was written"):
            apply_tasks_document(plan, _document(sample_tasks_json))

        assert _tree(outside) == outside_before, "the list landed outside"
        assert plan.read_bytes() == plan_before
        assert not (repo / "my-plan" / "state.json").exists()

    @pytest.mark.parametrize("swap", ["the list", "its folder"])
    def test_a_link_swapped_in_while_the_list_is_staged_is_never_followed(
            self, world, sample_tasks_json, monkeypatch, swap):
        """The seam: the folder held and the list staged in it, just before
        the rename. A link at the list's name is replaced by the list, its
        target untouched; a folder replaced by a link is no longer the one
        held, so nothing is renamed into it."""
        repo, outside, plan, folder = world
        real_check = write_safety._check_staged
        swapped = []

        def swap_then_check(*args, **kwargs):
            if not swapped:
                self._swap(folder, outside, swap)
                swapped.append(swap)
            return real_check(*args, **kwargs)

        monkeypatch.setattr(write_safety, "_check_staged", swap_then_check)
        outside_before, plan_before = _tree(outside), plan.read_bytes()

        if swap == "the list":
            apply_tasks_document(plan, _document(sample_tasks_json))
            assert not (folder / "tasks.md").is_symlink()
            assert "### Task T001" in (folder / "tasks.md").read_text(encoding="utf-8")
        else:
            with pytest.raises(OSError, match="moved or replaced"):
                apply_tasks_document(plan, _document(sample_tasks_json))
            assert plan.read_bytes() == plan_before

        assert swapped, "the list went in through its held folder"
        assert _tree(outside) == outside_before, "the list landed outside"
