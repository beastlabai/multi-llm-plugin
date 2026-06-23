"""Tests for update_plan_tasks script."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from update_plan_tasks import (
    generate_dependency_diagram,
    generate_tasks_markdown,
    validate_tasks_json,
)


class TestValidateTasksJson:
    """Tests for validate_tasks_json function."""

    def test_valid_tasks(self, sample_tasks_json):
        """Test validation with valid tasks."""
        data = {"tasks": sample_tasks_json}
        is_valid, error = validate_tasks_json(data)
        assert is_valid is True
        assert error == ""

    def test_missing_tasks_field(self):
        """Test validation when tasks field is missing."""
        data = {"items": []}
        is_valid, error = validate_tasks_json(data)
        assert is_valid is False
        assert "Missing 'tasks' field" in error

    def test_tasks_not_array(self):
        """Test validation when tasks is not an array."""
        data = {"tasks": "not an array"}
        is_valid, error = validate_tasks_json(data)
        assert is_valid is False
        assert "'tasks' must be an array" in error

    def test_empty_tasks(self):
        """Test validation when tasks array is empty."""
        data = {"tasks": []}
        is_valid, error = validate_tasks_json(data)
        assert is_valid is False
        assert "empty" in error

    def test_task_missing_id(self):
        """Test validation when a task is missing id."""
        data = {"tasks": [{"title": "Test"}]}
        is_valid, error = validate_tasks_json(data)
        assert is_valid is False
        assert "missing 'id'" in error

    def test_task_missing_title(self):
        """Test validation when a task is missing title."""
        data = {"tasks": [{"id": "T001"}]}
        is_valid, error = validate_tasks_json(data)
        assert is_valid is False
        assert "missing 'title'" in error

    def test_root_not_object(self):
        """Test validation when root is not an object."""
        data = [{"id": "T001", "title": "Test"}]
        is_valid, error = validate_tasks_json(data)
        assert is_valid is False
        assert "Root must be an object" in error


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

    def test_complexity_displayed(self, sample_tasks_json):
        """Test that complexity is shown."""
        result = generate_tasks_markdown(sample_tasks_json, "plan.md")

        assert "**Complexity**: low" in result
        assert "**Complexity**: medium" in result

    def test_subagent_type_displayed(self):
        """Test that subagent type is shown."""
        tasks = [{
            "id": "T001",
            "title": "Database migration",
            "subagent_type": "general-purpose"
        }]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Subagent**: general-purpose" in result

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
        plan_path.write_text("# My Plan\n\nSome content here.\n")

        # Create tasks JSON
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps({"tasks": sample_tasks_json}))

        # Run script
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check tasks file was created
        tasks_file = temp_dir / "my-plan" / "tasks" / "tasks.md"
        assert tasks_file.exists(), f"Tasks file not created at {tasks_file}"

        # Check content
        tasks_content = tasks_file.read_text()
        assert "### Task T001" in tasks_content
        assert "### Task T002" in tasks_content

    def test_script_updates_plan_file(self, temp_dir, sample_tasks_json):
        """Test that the script updates the plan with tasks reference."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nSome content here.\n")

        # Create tasks JSON
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps({"tasks": sample_tasks_json}))

        # Run script
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check plan was updated
        plan_content = plan_path.read_text()
        assert "<!-- TASKS_FILE:" in plan_content
        assert "## Implementation Tasks" in plan_content

    def test_script_dry_run(self, temp_dir, sample_tasks_json):
        """Test that dry-run mode doesn't modify files."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        original_content = "# My Plan\n\nSome content here.\n"
        plan_path.write_text(original_content)

        # Create tasks JSON
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps({"tasks": sample_tasks_json}))

        # Run script with --dry-run
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path),
             "--dry-run"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "Dry run" in result.stdout

        # Check nothing was modified
        assert plan_path.read_text() == original_content
        tasks_file = temp_dir / "my-plan" / "tasks" / "tasks.md"
        assert not tasks_file.exists()

    def test_script_missing_plan_file(self, temp_dir, sample_tasks_json):
        """Test error handling when plan file doesn't exist."""
        plan_path = temp_dir / "nonexistent.md"
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps({"tasks": sample_tasks_json}))

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1
        assert "Plan file not found" in result.stderr

    def test_script_missing_tasks_file(self, temp_dir):
        """Test error handling when tasks file doesn't exist."""
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# Plan\n")
        tasks_json_path = temp_dir / "nonexistent.json"

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1
        assert "Tasks JSON file not found" in result.stderr

    def test_script_invalid_json(self, temp_dir):
        """Test error handling with invalid JSON."""
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# Plan\n")
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text("not valid json {")

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1
        assert "Failed to parse" in result.stderr

    def test_script_invalid_tasks_structure(self, temp_dir):
        """Test error handling with invalid tasks structure."""
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# Plan\n")
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps({"tasks": []}))

        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1
        assert "Invalid tasks JSON" in result.stderr


class TestPreambleStorage:
    """Tests for plan_preamble storage functionality."""

    def test_script_stores_preamble_in_state(self, temp_dir, sample_tasks_json):
        """Test that the script stores plan_preamble in state.json."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nSome content here.\n")

        # Create tasks JSON with preamble
        tasks_data = {
            "plan_preamble": "This is a test preamble describing the plan goals and architecture.",
            "tasks": sample_tasks_json
        }
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps(tasks_data))

        # Run script
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Stored plan preamble" in result.stdout

        # Check state.json was created and contains preamble
        state_path = temp_dir / "my-plan" / "state.json"
        assert state_path.exists(), f"State file not created at {state_path}"

        state_data = json.loads(state_path.read_text())
        assert "plan_preamble" in state_data
        assert state_data["plan_preamble"] == "This is a test preamble describing the plan goals and architecture."

    def test_script_handles_missing_preamble_gracefully(self, temp_dir, sample_tasks_json):
        """Test that the script works when plan_preamble is missing from JSON."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nSome content here.\n")

        # Create tasks JSON without preamble (legacy format)
        tasks_data = {"tasks": sample_tasks_json}
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps(tasks_data))

        # Run script
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        # Should not mention storing preamble since there was none
        assert "Stored plan preamble" not in result.stdout

    def test_script_handles_empty_preamble(self, temp_dir, sample_tasks_json):
        """Test that the script handles empty preamble string."""
        # Create plan file
        plan_path = temp_dir / "my-plan.md"
        plan_path.write_text("# My Plan\n\nSome content here.\n")

        # Create tasks JSON with empty preamble
        tasks_data = {
            "plan_preamble": "",
            "tasks": sample_tasks_json
        }
        tasks_json_path = temp_dir / "tasks.json"
        tasks_json_path.write_text(json.dumps(tasks_data))

        # Run script
        script_path = Path(__file__).parent.parent / "update_plan_tasks.py"
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--plan-file", str(plan_path),
             "--tasks-file", str(tasks_json_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        # Should not store empty preamble
        assert "Stored plan preamble" not in result.stdout


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

    def test_non_human_task_still_shows_subagent(self):
        """Test that non-human tasks still render with Subagent label."""
        tasks = [{
            "id": "T001",
            "title": "Implement feature",
            "subagent_type": "general-purpose",
        }]
        result = generate_tasks_markdown(tasks, "plan.md")

        assert "**Subagent**: general-purpose" in result
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

        assert "**Subagent**: general-purpose" in result
        assert "**Type**: Manual (user action required)" in result
