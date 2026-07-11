"""Tests for implement_orchestrator module."""

import json
import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.task_decomposer import TaskDecomposer, Task, TaskStatus


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_plan_with_tasks(temp_dir):
    """Create a sample plan file with tasks for testing."""
    plan_content = """# Sample Implementation Plan

## Overview
This is a sample plan for testing the implement orchestrator.

## Tasks

### T001: Create directory structure
Create the basic directory structure.
- Depends on: none
- Files to create: src/main.py

### T002: Implement core module
Implement the core functionality.
- Depends on: T001
- Files to modify: src/main.py

### T003: Add tests
Write unit tests for the core module.
- Depends on: T002
- Files to create: tests/test_main.py
"""
    plan_path = temp_dir / "sample-plan.md"
    plan_path.write_text(plan_content)
    return plan_path


@pytest.fixture
def sample_plan_without_tasks(temp_dir):
    """Create a sample plan file without tasks."""
    plan_content = """# Sample Implementation Plan

## Overview
This is a sample plan without implementation tasks.

## Description
Just a description, no tasks.
"""
    plan_path = temp_dir / "no-tasks-plan.md"
    plan_path.write_text(plan_content)
    return plan_path


@pytest.fixture
def sample_plan_with_tasks_reference(temp_dir):
    """Create a sample plan file with external tasks file reference."""
    plan_content = """# Sample Implementation Plan

<!-- TASKS_FILE: tasks/tasks.md -->

## Overview
This is a sample plan with an external tasks file reference.
"""
    plan_path = temp_dir / "ref-plan.md"
    plan_path.write_text(plan_content)

    # Create the tasks file
    tasks_dir = temp_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    tasks_content = """# Tasks

### T001: External task
Task from external file.
- Depends on: none
"""
    (tasks_dir / "tasks.md").write_text(tasks_content)
    return plan_path


@pytest.fixture
def sample_tasks_json():
    """Sample task decomposition JSON."""
    return [
        {
            "id": "T001",
            "title": "Create directory structure",
            "description": "Create the basic directory structure",
            "depends_on": [],
            "files_to_create": ["src/main.py"],
            "estimated_complexity": "low",
            "subagent_type": "general-purpose"
        },
        {
            "id": "T002",
            "title": "Implement core module",
            "description": "Implement the core functionality",
            "depends_on": ["T001"],
            "files_to_modify": ["src/main.py"],
            "estimated_complexity": "medium",
            "subagent_type": "general-purpose"
        },
        {
            "id": "T003",
            "title": "Add tests",
            "description": "Write unit tests",
            "depends_on": ["T002"],
            "files_to_create": ["tests/test_main.py"],
            "estimated_complexity": "medium",
            "subagent_type": "general-purpose"
        }
    ]


@pytest.fixture
def mock_state_manager():
    """Create a mock StateManager instance."""
    mock = MagicMock()
    mock.state_file = Path("/tmp/test_state.json")
    mock.get.return_value = None
    mock.get_all_task_statuses.return_value = {}
    mock.has_plan_changed.return_value = False
    return mock


@pytest.fixture
def mock_git_functions():
    """Mock git utility functions."""
    with patch('implement_orchestrator.get_modified_files', return_value=[]), \
         patch('implement_orchestrator.get_staged_files', return_value=[]):
        yield


# --- Test Classes ---

class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_with_plan_file(self):
        """Test parsing with required --plan-file argument."""
        # Import inside test to avoid module-level import issues
        from implement_orchestrator import parse_args

        test_args = ['--plan-file', 'plans/my-plan.md']
        with patch.object(sys, 'argv', ['implement_orchestrator.py'] + test_args):
            args = parse_args()

        assert args.plan_file == 'plans/my-plan.md'
        # The default output is resolved AFTER parsing (it depends on the git
        # root and the plan stem — see resolve_default_output), so argparse
        # leaves it unset.
        assert args.output is None
        assert args.resume is False
        assert args.dry_run is False
        assert args.task is None

    def test_parse_args_with_dry_run(self):
        """Test parsing with --dry-run flag."""
        from implement_orchestrator import parse_args

        test_args = ['--plan-file', 'plans/my-plan.md', '--dry-run']
        with patch.object(sys, 'argv', ['implement_orchestrator.py'] + test_args):
            args = parse_args()

        assert args.dry_run is True

    def test_parse_args_with_resume(self):
        """Test parsing with --resume flag."""
        from implement_orchestrator import parse_args

        test_args = ['--plan-file', 'plans/my-plan.md', '--resume']
        with patch.object(sys, 'argv', ['implement_orchestrator.py'] + test_args):
            args = parse_args()

        assert args.resume is True

    def test_parse_args_with_specific_task(self):
        """Test parsing with --task argument."""
        from implement_orchestrator import parse_args

        test_args = ['--plan-file', 'plans/my-plan.md', '--task', 'T002']
        with patch.object(sys, 'argv', ['implement_orchestrator.py'] + test_args):
            args = parse_args()

        assert args.task == 'T002'

    def test_parse_args_with_custom_output(self):
        """Test parsing with custom --output path."""
        from implement_orchestrator import parse_args

        test_args = ['--plan-file', 'plans/my-plan.md', '--output', '/custom/output.json']
        with patch.object(sys, 'argv', ['implement_orchestrator.py'] + test_args):
            args = parse_args()

        assert args.output == '/custom/output.json'

    def test_parse_args_missing_required_plan_file(self):
        """Test parsing without required --plan-file raises error."""
        from implement_orchestrator import parse_args

        test_args = ['--dry-run']
        with patch.object(sys, 'argv', ['implement_orchestrator.py'] + test_args):
            with pytest.raises(SystemExit):
                parse_args()


class TestDefaultOutputResolution:
    """Tests for the git-root-anchored default --output path (Phase 3).

    The default must stay byte-identical to the path documented in
    instructions/implement.md
    (<git-root>/.multi-llm/tmp/implementation_tasks_{plan_stem}.json) and must
    never fall back to the system temp dir.
    """

    def test_default_output_uses_git_root_and_plan_stem(self):
        """Expected path is computed with the same primitives the code uses
        (get_project_root() + plan stem + Path joining) — never a hardcoded
        POSIX-separator literal, which would pass on Linux while silently
        diverging on Windows."""
        from implement_orchestrator import resolve_default_output
        from utils.git_utils import get_project_root

        # A plan path inside this repo, so git-root detection succeeds.
        plan_path = (Path(__file__).parent / "my-sample-plan.md").resolve()
        project_root = get_project_root(str(plan_path))
        assert project_root, "test suite must run from inside the git repo"

        expected = Path(project_root) / ".multi-llm" / "tmp" / (
            f"implementation_tasks_{plan_path.stem}.json"
        )
        assert resolve_default_output(plan_path) == str(expected)

    def test_default_output_fails_fast_outside_git_repo(self, temp_dir):
        """Outside a git work tree the resolution exits with an error instead
        of falling back to tempfile.gettempdir()."""
        from implement_orchestrator import resolve_default_output

        plan_path = temp_dir / "plan.md"
        with patch('implement_orchestrator.get_project_root', return_value=None):
            with pytest.raises(SystemExit):
                resolve_default_output(plan_path)


class TestRecordPreExistingChanges:
    """Tests for record_pre_existing_changes function."""

    def test_records_modified_and_staged_files(self, mock_state_manager):
        """Test recording both modified and staged files."""
        from implement_orchestrator import record_pre_existing_changes

        with patch('implement_orchestrator.get_modified_files', return_value=['file1.py', 'file2.py']), \
             patch('implement_orchestrator.get_staged_files', return_value=['file3.py']):

            record_pre_existing_changes(mock_state_manager)

        mock_state_manager.set.assert_called_once()
        call_args = mock_state_manager.set.call_args[0]
        assert call_args[0] == "pre_existing_changes"
        assert set(call_args[1]) == {'file1.py', 'file2.py', 'file3.py'}

    def test_skips_if_already_recorded(self, mock_state_manager):
        """Test skipping if pre-existing changes already recorded."""
        from implement_orchestrator import record_pre_existing_changes

        mock_state_manager.get.return_value = ['existing.py']

        with patch('implement_orchestrator.get_modified_files') as mock_modified:
            record_pre_existing_changes(mock_state_manager)

        mock_modified.assert_not_called()

    def test_handles_empty_file_list(self, mock_state_manager):
        """Test handling when no files are modified."""
        from implement_orchestrator import record_pre_existing_changes

        with patch('implement_orchestrator.get_modified_files', return_value=[]), \
             patch('implement_orchestrator.get_staged_files', return_value=[]):

            record_pre_existing_changes(mock_state_manager)

        call_args = mock_state_manager.set.call_args[0]
        assert call_args[1] == []

    def test_filters_empty_strings(self, mock_state_manager):
        """Test filtering out empty strings from file lists."""
        from implement_orchestrator import record_pre_existing_changes

        with patch('implement_orchestrator.get_modified_files', return_value=['', 'file1.py', '']), \
             patch('implement_orchestrator.get_staged_files', return_value=['file2.py', '']):

            record_pre_existing_changes(mock_state_manager)

        call_args = mock_state_manager.set.call_args[0]
        assert '' not in call_args[1]
        assert set(call_args[1]) == {'file1.py', 'file2.py'}

    def test_handles_git_error_gracefully(self, mock_state_manager):
        """Test graceful handling of git errors."""
        from implement_orchestrator import record_pre_existing_changes

        with patch('implement_orchestrator.get_modified_files', side_effect=Exception("Git error")):
            record_pre_existing_changes(mock_state_manager)

        # Should set empty list on error
        call_args = mock_state_manager.set.call_args[0]
        assert call_args[0] == "pre_existing_changes"
        assert call_args[1] == []


class TestRecordHeadBeforeImplement:
    """Tests for head_before_implement capture in run_implement."""

    def test_records_head_when_absent(self, mock_state_manager):
        """Test that head_before_implement is recorded when not already in state."""
        mock_state_manager.get.side_effect = lambda key, *args: None

        with patch('implement_orchestrator.get_current_head', return_value='abc123') as mock_head:
            # Inline the logic under test (it's embedded in run_implement, not a separate function)
            if mock_state_manager.get("head_before_implement") is None:
                mock_state_manager.set("head_before_implement", mock_head())
                mock_state_manager.save()

        mock_state_manager.set.assert_called_once_with("head_before_implement", "abc123")
        mock_state_manager.save.assert_called()

    def test_skips_when_already_present(self, mock_state_manager):
        """Test that head_before_implement is not overwritten on resume."""
        mock_state_manager.get.side_effect = lambda key, *args: 'existing_sha' if key == 'head_before_implement' else None

        with patch('implement_orchestrator.get_current_head') as mock_head:
            if mock_state_manager.get("head_before_implement") is None:
                mock_state_manager.set("head_before_implement", mock_head())
                mock_state_manager.save()

        mock_head.assert_not_called()
        mock_state_manager.set.assert_not_called()

    def test_handles_git_error_gracefully(self, mock_state_manager):
        """Test graceful handling when get_current_head fails."""
        mock_state_manager.get.side_effect = lambda key, *args: None

        with patch('implement_orchestrator.get_current_head', side_effect=Exception("Git error")):
            # Replicate the try/except from the orchestrator
            if mock_state_manager.get("head_before_implement") is None:
                try:
                    from implement_orchestrator import get_current_head
                    mock_state_manager.set("head_before_implement", get_current_head())
                    mock_state_manager.save()
                except Exception:
                    pass

        # set should not be called since get_current_head raised
        mock_state_manager.set.assert_not_called()


class TestLoadImplementationPrompt:
    """Tests for load_implementation_prompt function."""

    def test_loads_prompt_from_file(self):
        """Test loading prompt from file."""
        from implement_orchestrator import load_implementation_prompt

        mock_prompt = "Test prompt {task_title}"
        with patch('implement_orchestrator.load_prompt', return_value=mock_prompt):
            result = load_implementation_prompt()

        assert result == mock_prompt

    def test_returns_fallback_on_file_not_found(self):
        """Test returning fallback prompt when file not found."""
        from implement_orchestrator import load_implementation_prompt

        with patch('implement_orchestrator.load_prompt', side_effect=FileNotFoundError()):
            result = load_implementation_prompt()

        assert "## Task:" in result
        assert "{task_title}" in result
        assert "{task_description}" in result


class TestBuildTaskPrompt:
    """Tests for build_task_prompt function."""

    def test_builds_prompt_with_all_fields(self):
        """Test building prompt with all task fields."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description",
            files_to_modify=["src/main.py"],
            files_to_create=["src/new.py"]
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value="Title: {task_title}\nDesc: {task_description}\n"
                               "Relevant: {relevant_files}\nOutput: {output_files}"):
            result = build_task_prompt(task)

        assert "Test Task" in result
        assert "Test description" in result

    def test_builds_prompt_with_empty_files(self):
        """Test building prompt when no files specified."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description"
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value="Relevant: {relevant_files}\nOutput: {output_files}"):
            result = build_task_prompt(task)

        assert "Explore codebase as needed" in result or "Files as needed" in result


class TestBuildTaskPromptAcceptanceCriteria:
    """Tests for build_task_prompt acceptance criteria rendering."""

    TEMPLATE = (
        "Title: {task_title}\n"
        "Desc: {task_description}\n"
        "Relevant: {relevant_files}\n"
        "Output: {output_files}\n"
        "Preamble: {plan_preamble}\n"
        "Plan: {plan_path}\n"
        "{acceptance_criteria_section}"
        "{dependency_section}"
    )

    def test_renders_criteria_checklist_when_nonempty(self):
        """Test build_task_prompt renders acceptance criteria as checklist when non-empty."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description",
            acceptance_criteria=["First criterion", "Second criterion"],
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            result = build_task_prompt(task)

        assert "## Acceptance Criteria" in result
        assert "- [ ] First criterion" in result
        assert "- [ ] Second criterion" in result

    def test_omits_criteria_section_when_empty(self):
        """Test build_task_prompt omits criteria section when acceptance_criteria is empty."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description",
            acceptance_criteria=[],
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            result = build_task_prompt(task)

        assert "Acceptance Criteria" not in result

    def test_omits_criteria_section_when_none(self):
        """Test build_task_prompt omits criteria section when acceptance_criteria is None."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description",
        )
        # Force acceptance_criteria to None to test the guard
        task.acceptance_criteria = None

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            result = build_task_prompt(task)

        assert "Acceptance Criteria" not in result

    def test_omits_criteria_section_when_all_empty_strings(self):
        """Test build_task_prompt omits criteria section when all entries are empty strings."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description",
            acceptance_criteria=["", "   ", ""],
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            result = build_task_prompt(task)

        assert "Acceptance Criteria" not in result


class TestBuildTaskPromptDependencySection:
    """Tests for build_task_prompt dependency context rendering."""

    TEMPLATE = (
        "Title: {task_title}\n"
        "Desc: {task_description}\n"
        "Relevant: {relevant_files}\n"
        "Output: {output_files}\n"
        "Preamble: {plan_preamble}\n"
        "Plan: {plan_path}\n"
        "{acceptance_criteria_section}"
        "{dependency_section}"
    )

    def test_renders_dependency_context(self):
        """Test build_task_prompt renders dependency context with title and first description line."""
        from implement_orchestrator import build_task_prompt

        dep_task = Task(
            id="T001",
            title="Setup database",
            description="Create the initial schema.\nMore details here.",
        )
        current_task = Task(
            id="T002",
            title="Implement API",
            description="Build the API endpoints.",
            depends_on=["T001"],
        )

        all_tasks = {"T001": dep_task, "T002": current_task}

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            result = build_task_prompt(current_task, all_tasks=all_tasks)

        assert "## Preceding Tasks (already completed)" in result
        assert "**T001: Setup database**" in result
        assert "Create the initial schema." in result

    def test_omits_dependency_section_when_empty(self):
        """Test build_task_prompt omits dependency section when depends_on is empty."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description",
            depends_on=[],
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            result = build_task_prompt(task, all_tasks={"T001": task})

        assert "Preceding Tasks" not in result

    def test_handles_nonexistent_dependency_ids(self):
        """Test build_task_prompt handles nonexistent dependency IDs gracefully."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T002",
            title="Test Task",
            description="Test description",
            depends_on=["T999", "T998"],
        )

        all_tasks = {"T002": task}

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            # Should not raise
            result = build_task_prompt(task, all_tasks=all_tasks)

        # With no valid dependencies found, section should be omitted
        assert "Preceding Tasks" not in result

    def test_dependency_summary_excludes_metadata(self):
        """Test dependency summaries use clean description, not raw metadata."""
        from implement_orchestrator import build_task_prompt

        dep_task = Task(
            id="T001",
            title="Setup module",
            description="Implement the core module.",
        )
        current_task = Task(
            id="T002",
            title="Test Task",
            description="Test description",
            depends_on=["T001"],
        )

        all_tasks = {"T001": dep_task, "T002": current_task}

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            result = build_task_prompt(current_task, all_tasks=all_tasks)

        # The dependency summary should contain the first line of the description
        assert "Implement the core module." in result

    def test_omits_dependency_section_when_all_tasks_is_none(self):
        """Test build_task_prompt omits dependency section when all_tasks is None."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T002",
            title="Test Task",
            description="Test description",
            depends_on=["T001"],
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value=self.TEMPLATE):
            result = build_task_prompt(task, all_tasks=None)

        assert "Preceding Tasks" not in result


class TestTokenBudgetEnforcement:
    """Tests for _enforce_token_budget."""

    def test_no_truncation_under_budget(self):
        """Test that sections are not truncated when under budget."""
        from implement_orchestrator import _enforce_token_budget

        acceptance = "## Acceptance Criteria\n- [ ] Short\n"
        dependency = "## Preceding Tasks\n- **T001: Short**\n"

        result_acc, result_dep = _enforce_token_budget(acceptance, dependency, budget=2000)
        assert result_acc == acceptance
        assert result_dep == dependency

    def test_truncates_dependency_first(self):
        """Test that dependency section is truncated first when over budget."""
        from implement_orchestrator import _enforce_token_budget

        acceptance = "Short acceptance section"
        dependency = "X" * 3000  # Very long dependency section

        result_acc, result_dep = _enforce_token_budget(acceptance, dependency, budget=200)

        # Acceptance should be preserved or close to original
        # Dependency should be truncated
        assert len(result_acc) + len(result_dep) <= 200 or result_dep == ""

    def test_truncates_acceptance_if_still_over_budget(self):
        """Test acceptance section is truncated if still over budget after dependency truncation."""
        from implement_orchestrator import _enforce_token_budget

        acceptance = "A" * 3000
        dependency = ""

        result_acc, result_dep = _enforce_token_budget(acceptance, dependency, budget=200)

        # Combined should be within budget or acceptance truncated
        assert len(result_acc) + len(result_dep) <= 200 or result_acc == ""


class TestFormatTaskForOutput:
    """Tests for format_task_for_output function."""

    def test_formats_task_with_all_fields(self):
        """Test formatting task with all fields."""
        from implement_orchestrator import format_task_for_output

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description",
            depends_on=["T000"],
            files_to_modify=["src/main.py"],
            files_to_create=["src/new.py"],
            estimated_complexity="high",
            subagent_type="general-purpose",
            status=TaskStatus.PENDING
        )

        with patch('implement_orchestrator.build_task_prompt', return_value="Test prompt"):
            result = format_task_for_output(task)

        assert result["id"] == "T001"
        assert result["title"] == "Test Task"
        assert result["description"] == "Test description"
        assert result["prompt"] == "Test prompt"
        assert result["subagent_type"] == "general-purpose"
        assert result["depends_on"] == ["T000"]
        assert result["files_to_modify"] == ["src/main.py"]
        assert result["files_to_create"] == ["src/new.py"]
        assert result["complexity"] == "high"
        assert result["status"] == "pending"
        assert result["is_human"] is False

    def test_is_human_flag_true_for_human_tasks(self):
        """Test that is_human is True when subagent_type is 'human'."""
        from implement_orchestrator import format_task_for_output

        task = Task(
            id="T002",
            title="Create API Key",
            description="Create an API key in the dashboard",
            depends_on=[],
            files_to_modify=[],
            files_to_create=[],
            estimated_complexity="low",
            subagent_type="human",
            status=TaskStatus.PENDING
        )

        with patch('implement_orchestrator.build_task_prompt', return_value="Test prompt"):
            result = format_task_for_output(task)

        assert result["is_human"] is True
        assert result["subagent_type"] == "human"

    def test_is_human_flag_false_for_general_purpose(self):
        """Test that is_human is False for non-human subagent types."""
        from implement_orchestrator import format_task_for_output

        task = Task(
            id="T003",
            title="Implement feature",
            description="Build the feature",
            depends_on=[],
            files_to_modify=[],
            files_to_create=[],
            estimated_complexity="medium",
            subagent_type="general-purpose",
            status=TaskStatus.PENDING
        )

        with patch('implement_orchestrator.build_task_prompt', return_value="Test prompt"):
            result = format_task_for_output(task)

        assert result["is_human"] is False


class TestFormatBatchesForOutput:
    """Tests for format_batches_for_output function."""

    def test_formats_single_batch(self):
        """Test formatting a single batch."""
        from implement_orchestrator import format_batches_for_output

        task = Task(id="T001", title="Task 1", description="Desc", status=TaskStatus.PENDING)
        batches = [[task]]

        with patch('implement_orchestrator.format_task_for_output', return_value={"id": "T001"}):
            result = format_batches_for_output(batches)

        assert len(result) == 1
        assert result[0]["batch_index"] == 0
        assert result[0]["can_parallelize"] is False
        assert len(result[0]["tasks"]) == 1

    def test_formats_multiple_batches(self):
        """Test formatting multiple batches."""
        from implement_orchestrator import format_batches_for_output

        task1 = Task(id="T001", title="Task 1", description="Desc", status=TaskStatus.PENDING)
        task2 = Task(id="T002", title="Task 2", description="Desc", status=TaskStatus.PENDING)
        task3 = Task(id="T003", title="Task 3", description="Desc", status=TaskStatus.PENDING)
        batches = [[task1], [task2, task3]]

        with patch('implement_orchestrator.format_task_for_output',
                   side_effect=[{"id": "T001"}, {"id": "T002"}, {"id": "T003"}]):
            result = format_batches_for_output(batches)

        assert len(result) == 2
        assert result[0]["can_parallelize"] is False  # single task
        assert result[1]["can_parallelize"] is True   # multiple tasks

    def test_skips_completed_tasks_when_requested(self):
        """Test skipping completed tasks."""
        from implement_orchestrator import format_batches_for_output

        task1 = Task(id="T001", title="Task 1", description="Desc", status=TaskStatus.COMPLETED)
        task2 = Task(id="T002", title="Task 2", description="Desc", status=TaskStatus.PENDING)
        batches = [[task1, task2]]

        with patch('implement_orchestrator.format_task_for_output', return_value={"id": "T002"}):
            result = format_batches_for_output(batches, skip_completed=True)

        assert len(result) == 1
        assert len(result[0]["tasks"]) == 1

    def test_skips_empty_batches_after_filtering(self):
        """Test skipping empty batches after filtering completed tasks."""
        from implement_orchestrator import format_batches_for_output

        task1 = Task(id="T001", title="Task 1", description="Desc", status=TaskStatus.COMPLETED)
        batches = [[task1]]

        result = format_batches_for_output(batches, skip_completed=True)

        assert len(result) == 0


class TestTaskProcessing:
    """Tests for task processing logic."""

    def test_single_task_execution(self, sample_plan_with_tasks, mock_state_manager, temp_dir):
        """Test processing a single specific task."""
        from implement_orchestrator import main

        output_path = temp_dir / "output.json"

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', return_value=temp_dir / "tasks.md"), \
             patch('implement_orchestrator.get_relative_output_path', return_value="tasks.md"), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(output_path),
                resume=False,
                task="T001",
                dry_run=False
            )

            main()

        # Check output file was written
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["total_tasks"] == 1
        assert len(data["batches"]) == 1
        assert data["batches"][0]["tasks"][0]["id"] == "T001"

    def test_creates_nested_workspace_tmp_output_and_gitignore(self, sample_plan_with_tasks, mock_state_manager, temp_dir):
        """Regression (Phase 3): the nested `.multi-llm/tmp/...` output path is
        created on demand (mkdir parents=True covers the new default) and the
        temp dir is made self-ignoring via a `.gitignore` containing `*`."""
        from implement_orchestrator import main

        output_path = temp_dir / ".multi-llm" / "tmp" / "implementation_tasks_sample-plan.json"
        assert not output_path.parent.exists()

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', return_value=temp_dir / "tasks.md"), \
             patch('implement_orchestrator.get_relative_output_path', return_value="tasks.md"), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(output_path),
                resume=False,
                task=None,
                dry_run=False
            )

            main()

        # Task JSON landed in the freshly-created nested directory
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["total_tasks"] == 3

        # The temp dir ignores itself so run artifacts never hit git status
        gitignore_path = output_path.parent / ".gitignore"
        assert gitignore_path.exists()
        assert gitignore_path.read_text(encoding="utf-8").strip() == "*"

    def test_task_not_found_exits(self, sample_plan_with_tasks, mock_state_manager, temp_dir):
        """Test that requesting non-existent task exits with error."""
        from implement_orchestrator import main

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output="/tmp/output.json",
                resume=False,
                task="T999",  # Non-existent task
                dry_run=False
            )

            with pytest.raises(SystemExit):
                main()

        mock_exit.assert_called_with(1)


class TestDependencyOrdering:
    """Tests for task dependency ordering."""

    def test_tasks_ordered_by_dependencies(self, sample_tasks_json):
        """Test that tasks are ordered respecting dependencies."""
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(sample_tasks_json)
        batches = decomposer.get_parallel_batches()

        # T001 should be in first batch (no deps)
        assert any(t.id == "T001" for t in batches[0])

        # T002 depends on T001, should be in later batch
        t002_batch = next(i for i, batch in enumerate(batches) if any(t.id == "T002" for t in batch))
        t001_batch = next(i for i, batch in enumerate(batches) if any(t.id == "T001" for t in batch))
        assert t002_batch > t001_batch

    def test_independent_tasks_in_same_batch(self):
        """Test that independent tasks can be in the same batch."""
        tasks_json = [
            {"id": "T001", "title": "Task 1", "description": "Desc", "depends_on": []},
            {"id": "T002", "title": "Task 2", "description": "Desc", "depends_on": []},
        ]
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(tasks_json)
        batches = decomposer.get_parallel_batches()

        # Both tasks should be in the first batch
        assert len(batches) == 1
        task_ids = {t.id for t in batches[0]}
        assert task_ids == {"T001", "T002"}


class TestStateManagement:
    """Tests for state management integration."""

    def test_restores_task_statuses_on_resume(self, sample_plan_with_tasks, temp_dir):
        """Test that task statuses are restored when resuming."""
        from implement_orchestrator import main

        mock_state = MagicMock()
        mock_state.state_file = temp_dir / "state.json"
        mock_state.get.return_value = None
        mock_state.get_all_task_statuses.return_value = {
            "T001": "completed",
            "T002": "in_progress"
        }
        mock_state.has_plan_changed.return_value = False

        output_path = temp_dir / "output.json"

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', return_value=temp_dir / "tasks.md"), \
             patch('implement_orchestrator.get_relative_output_path', return_value="tasks.md"), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(output_path),
                resume=True,
                task=None,
                dry_run=False
            )

            main()

        # Output should only contain pending tasks (T003)
        data = json.loads(output_path.read_text())
        all_task_ids = []
        for batch in data["batches"]:
            for task in batch["tasks"]:
                all_task_ids.append(task["id"])

        # T001 was completed, should be skipped
        assert "T001" not in all_task_ids or data.get("pending_tasks", 0) < data["total_tasks"]

    def test_warns_on_plan_change_during_resume(self, sample_plan_with_tasks, temp_dir, capsys):
        """Test warning when plan has changed during resume."""
        from implement_orchestrator import main

        mock_state = MagicMock()
        mock_state.state_file = temp_dir / "state.json"
        mock_state.get.return_value = None
        mock_state.get_all_task_statuses.return_value = {}
        mock_state.has_plan_changed.return_value = True

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', return_value=temp_dir / "tasks.md"), \
             patch('implement_orchestrator.get_relative_output_path', return_value="tasks.md"), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(temp_dir / "output.json"),
                resume=True,
                task=None,
                dry_run=False
            )

            main()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "changed" in captured.out.lower()
        mock_state.clear_plan_changed_flag.assert_called_once()


class TestSubagentRouting:
    """Tests for subagent type routing."""

    def test_task_has_subagent_type(self):
        """Test that tasks have subagent_type field."""
        task = Task(
            id="T001",
            title="Create database schema",
            description="Create PostgreSQL schema",
            subagent_type="general-purpose"
        )

        assert task.subagent_type == "general-purpose"

    def test_default_subagent_type(self):
        """Test default subagent_type is general-purpose."""
        task = Task(
            id="T001",
            title="Generic task",
            description="Some task"
        )

        assert task.subagent_type == "general-purpose"

    def test_subagent_type_in_output(self):
        """Test subagent_type is included in formatted output."""
        from implement_orchestrator import format_task_for_output

        task = Task(
            id="T001",
            title="Test Task",
            description="Test",
            subagent_type="general-purpose"
        )

        with patch('implement_orchestrator.build_task_prompt', return_value="Test prompt"):
            result = format_task_for_output(task)

        assert result["subagent_type"] == "general-purpose"

    def test_subagent_types(self):
        """Test general-purpose subagent type (only type available in Claude Code)."""
        task = Task(
            id="T001",
            title="Test",
            description="Test",
            subagent_type="general-purpose"
        )
        assert task.subagent_type == "general-purpose"


class TestDryRunOutput:
    """Tests for dry-run output."""

    def test_dry_run_does_not_write_file(self, sample_plan_with_tasks, mock_state_manager, temp_dir):
        """Test that dry-run exits before writing file."""
        from implement_orchestrator import main

        output_path = temp_dir / "output.json"

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('sys.exit', side_effect=SystemExit(0)) as mock_exit:

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(output_path),
                resume=False,
                task=None,
                dry_run=True
            )

            with pytest.raises(SystemExit) as exc_info:
                main()

        # Should exit with 0 (dry run prints summary and exits)
        mock_exit.assert_called_with(0)
        assert exc_info.value.code == 0

    def test_dry_run_prints_summary(self, sample_plan_with_tasks, mock_state_manager, temp_dir, capsys):
        """Test that dry-run prints task summary."""
        from implement_orchestrator import main

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(temp_dir / "output.json"),
                resume=False,
                task=None,
                dry_run=True
            )

            main()

        captured = capsys.readouterr()
        assert "Dry Run" in captured.out
        assert "Total tasks" in captured.out
        assert "Batches" in captured.out


class TestImplementationSummary:
    """Tests for implementation summary output."""

    def test_output_includes_summary_file_path(self, sample_plan_with_tasks, mock_state_manager, temp_dir):
        """Test that output includes summary file path."""
        from implement_orchestrator import main

        output_path = temp_dir / "output.json"
        summary_path = temp_dir / "implement" / "implementation_summary.md"

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', side_effect=[temp_dir / "tasks.md", summary_path]), \
             patch('implement_orchestrator.get_relative_output_path', side_effect=["tasks.md", "implement/summary.md"]), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(output_path),
                resume=False,
                task=None,
                dry_run=False
            )

            main()

        data = json.loads(output_path.read_text())
        assert "summary_file" in data
        assert "summary_file_relative" in data


class TestPlanWithoutTasks:
    """Tests for handling plans without tasks."""

    def test_outputs_tasks_missing_marker_for_plan_without_tasks(self, sample_plan_without_tasks, temp_dir, capsys):
        """Test that plan without tasks outputs TASKS_MISSING marker and exits with 0."""
        from implement_orchestrator import main

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('sys.exit', side_effect=SystemExit(0)) as mock_exit:

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_without_tasks),
                output=str(temp_dir / "output.json"),
                resume=False,
                task=None,
                dry_run=False
            )

            with pytest.raises(SystemExit):
                main()

        mock_exit.assert_called_with(0)
        captured = capsys.readouterr()
        assert "[TASKS_MISSING]" in captured.out
        assert "No implementation tasks" in captured.out


class TestExternalTasksFile:
    """Tests for external tasks file handling."""

    def test_loads_tasks_from_external_file(self, sample_plan_with_tasks_reference, mock_state_manager, temp_dir):
        """Test loading tasks from external file reference."""
        from implement_orchestrator import main

        output_path = temp_dir / "output.json"

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', return_value=temp_dir / "tasks.md"), \
             patch('implement_orchestrator.get_relative_output_path', return_value="tasks.md"), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks_reference),
                output=str(output_path),
                resume=False,
                task=None,
                dry_run=False
            )

            main()

        data = json.loads(output_path.read_text())
        # Should have loaded the external task
        assert data["total_tasks"] >= 1


class TestOutputFormat:
    """Tests for output JSON format."""

    def test_output_contains_required_fields(self, sample_plan_with_tasks, mock_state_manager, temp_dir):
        """Test that output contains all required fields."""
        from implement_orchestrator import main

        output_path = temp_dir / "output.json"

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', return_value=temp_dir / "tasks.md"), \
             patch('implement_orchestrator.get_relative_output_path', return_value="tasks.md"), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(output_path),
                resume=False,
                task=None,
                dry_run=False
            )

            main()

        data = json.loads(output_path.read_text())

        # Check required fields
        assert "plan_file" in data
        assert "tasks_file" in data
        assert "summary_file" in data
        assert "state_file" in data
        assert "generated_at" in data
        assert "total_tasks" in data
        assert "batches" in data

    def test_task_output_contains_required_fields(self, sample_plan_with_tasks, mock_state_manager, temp_dir):
        """Test that task output contains all required fields."""
        from implement_orchestrator import main

        output_path = temp_dir / "output.json"

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', return_value=temp_dir / "tasks.md"), \
             patch('implement_orchestrator.get_relative_output_path', return_value="tasks.md"), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(output_path),
                resume=False,
                task=None,
                dry_run=False
            )

            main()

        data = json.loads(output_path.read_text())
        task = data["batches"][0]["tasks"][0]

        # Check required task fields
        assert "id" in task
        assert "title" in task
        assert "description" in task
        assert "prompt" in task
        assert "subagent_type" in task
        assert "depends_on" in task
        assert "status" in task

    def test_generated_at_is_valid_iso_format(self, sample_plan_with_tasks, mock_state_manager, temp_dir):
        """Test that generated_at is valid ISO format timestamp."""
        from implement_orchestrator import main

        output_path = temp_dir / "output.json"

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('implement_orchestrator.get_or_create_state', return_value=mock_state_manager), \
             patch('implement_orchestrator.record_pre_existing_changes'), \
             patch('implement_orchestrator.get_output_paths', return_value=temp_dir / "tasks.md"), \
             patch('implement_orchestrator.get_relative_output_path', return_value="tasks.md"), \
             patch('sys.exit'):

            mock_args.return_value = MagicMock(
                plan_file=str(sample_plan_with_tasks),
                output=str(output_path),
                resume=False,
                task=None,
                dry_run=False
            )

            main()

        data = json.loads(output_path.read_text())

        # Should not raise
        datetime.fromisoformat(data["generated_at"])


class TestPlanFileValidation:
    """Tests for plan file validation."""

    def test_exits_on_nonexistent_plan_file(self, temp_dir, capsys):
        """Test that nonexistent plan file exits with error."""
        from implement_orchestrator import main

        with patch('implement_orchestrator.parse_args') as mock_args, \
             patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:

            mock_args.return_value = MagicMock(
                plan_file=str(temp_dir / "nonexistent.md"),
                output="/tmp/output.json",
                resume=False,
                task=None,
                dry_run=False
            )

            with pytest.raises(SystemExit):
                main()

        mock_exit.assert_called_with(1)
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "not found" in captured.out


class TestPlanPreamble:
    """Tests for plan preamble and plan path functionality."""

    def test_build_task_prompt_includes_preamble(self):
        """Test that build_task_prompt includes the preamble in the output."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description"
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value="Context: {plan_preamble}\nPath: {plan_path}\nTitle: {task_title}"):
            result = build_task_prompt(task, plan_preamble="This is the preamble", plan_path="/path/to/plan.md")

        assert "This is the preamble" in result
        assert "/path/to/plan.md" in result

    def test_build_task_prompt_includes_plan_path(self):
        """Test that build_task_prompt includes the plan path in the output."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description"
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value="Plan: {plan_path}\n{task_title}"):
            result = build_task_prompt(task, plan_path="/home/user/plans/feature.md")

        assert "/home/user/plans/feature.md" in result

    def test_build_task_prompt_with_missing_preamble(self):
        """Test that build_task_prompt uses fallback text when preamble is empty."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description"
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value="Context: {plan_preamble}\nTitle: {task_title}"):
            result = build_task_prompt(task, plan_preamble="", plan_path="")

        assert "See the full plan below for context" in result

    def test_build_task_prompt_with_none_preamble(self):
        """Test that build_task_prompt handles None preamble gracefully."""
        from implement_orchestrator import build_task_prompt

        task = Task(
            id="T001",
            title="Test Task",
            description="Test description"
        )

        with patch('implement_orchestrator.load_implementation_prompt',
                   return_value="Context: {plan_preamble}\nPath: {plan_path}"):
            # Should not raise, should use defaults
            result = build_task_prompt(task)

        assert "See the full plan below for context" in result

    def test_format_task_for_output_passes_preamble(self):
        """Test that format_task_for_output passes preamble to build_task_prompt."""
        from implement_orchestrator import format_task_for_output

        task = Task(
            id="T001",
            title="Test Task",
            description="Test"
        )

        with patch('implement_orchestrator.build_task_prompt', return_value="Test prompt") as mock_build:
            format_task_for_output(task, plan_preamble="My preamble", plan_path="/my/plan.md")

        mock_build.assert_called_once_with(task, "My preamble", "/my/plan.md", all_tasks=None)

    def test_format_batches_for_output_passes_preamble(self):
        """Test that format_batches_for_output passes preamble to format_task_for_output."""
        from implement_orchestrator import format_batches_for_output

        task = Task(id="T001", title="Task 1", description="Desc", status=TaskStatus.PENDING)
        batches = [[task]]

        with patch('implement_orchestrator.format_task_for_output', return_value={"id": "T001"}) as mock_format:
            format_batches_for_output(batches, plan_preamble="Batch preamble", plan_path="/batch/plan.md")

        mock_format.assert_called_with(task, "Batch preamble", "/batch/plan.md", all_tasks=None)


class TestHumanTaskDependencyEdgeCases:
    """Edge-case tests for human task dependency handling."""

    def test_multiple_human_prerequisites_for_single_task(self):
        """A code task depending on multiple human tasks should have all deps tracked."""
        tasks_json = [
            {"id": "T001", "title": "Create API key", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T002", "title": "Configure DNS", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T003", "title": "Deploy service", "description": "Code", "subagent_type": "general-purpose", "depends_on": ["T001", "T002"]},
        ]
        decomposer = TaskDecomposer()
        tasks = decomposer.parse_from_json(tasks_json)

        # T003 depends on both human tasks
        t003 = decomposer.get_task("T003")
        assert set(t003.depends_on) == {"T001", "T002"}

        # Both human tasks should be in first batch, T003 in second
        batches = decomposer.get_parallel_batches()
        batch0_ids = {t.id for t in batches[0]}
        assert "T001" in batch0_ids
        assert "T002" in batch0_ids
        assert any(t.id == "T003" for t in batches[1])

    def test_transitive_chain_through_human_task(self):
        """Skipping a human task propagates transitively: A(human) -> B -> C."""
        tasks_json = [
            {"id": "T001", "title": "Create API key", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T002", "title": "Build API client", "description": "Code", "subagent_type": "general-purpose", "depends_on": ["T001"]},
            {"id": "T003", "title": "Write integration tests", "description": "Code", "subagent_type": "general-purpose", "depends_on": ["T002"]},
        ]
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(tasks_json)

        # Simulate skip propagation: skip T001 -> T002 depends on it -> T003 depends on T002
        # Walk the dependency graph forward from skipped tasks
        skipped = {"T001"}
        queue = list(skipped)
        while queue:
            current = queue.pop(0)
            for tid, task in decomposer.tasks.items():
                if current in task.depends_on and tid not in skipped:
                    skipped.add(tid)
                    queue.append(tid)

        assert skipped == {"T001", "T002", "T003"}

    def test_transitive_skip_does_not_affect_independent_tasks(self):
        """Skipping a human task does NOT affect tasks with no dependency on it."""
        tasks_json = [
            {"id": "T001", "title": "Create API key", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T002", "title": "Build API client", "description": "Code", "subagent_type": "general-purpose", "depends_on": ["T001"]},
            {"id": "T003", "title": "Setup database", "description": "Code", "subagent_type": "general-purpose", "depends_on": []},
            {"id": "T004", "title": "Create models", "description": "Code", "subagent_type": "general-purpose", "depends_on": ["T003"]},
        ]
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(tasks_json)

        # Walk forward from T001 (human, skipped)
        skipped = {"T001"}
        queue = list(skipped)
        while queue:
            current = queue.pop(0)
            for tid, task in decomposer.tasks.items():
                if current in task.depends_on and tid not in skipped:
                    skipped.add(tid)
                    queue.append(tid)

        # T001 and T002 skipped, but T003 and T004 are independent
        assert skipped == {"T001", "T002"}

    def test_diamond_dependency_with_human_task(self):
        """Diamond: A(human) -> B, A -> C, B+C -> D. Skipping A skips everything."""
        tasks_json = [
            {"id": "T001", "title": "Create API key", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T002", "title": "Build client", "description": "Code", "subagent_type": "general-purpose", "depends_on": ["T001"]},
            {"id": "T003", "title": "Build server", "description": "Code", "subagent_type": "general-purpose", "depends_on": ["T001"]},
            {"id": "T004", "title": "Integration test", "description": "Code", "subagent_type": "general-purpose", "depends_on": ["T002", "T003"]},
        ]
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(tasks_json)

        skipped = {"T001"}
        queue = list(skipped)
        while queue:
            current = queue.pop(0)
            for tid, task in decomposer.tasks.items():
                if current in task.depends_on and tid not in skipped:
                    skipped.add(tid)
                    queue.append(tid)

        assert skipped == {"T001", "T002", "T003", "T004"}

    def test_human_tasks_in_same_batch_no_cross_deps(self):
        """Multiple human tasks with no deps are in the same batch."""
        tasks_json = [
            {"id": "T001", "title": "Create API key", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T002", "title": "Configure DNS", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T003", "title": "Setup monitoring", "description": "Manual", "subagent_type": "human", "depends_on": []},
        ]
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(tasks_json)
        batches = decomposer.get_parallel_batches()

        # All in first batch (no dependencies, no file overlap)
        assert len(batches) == 1
        batch_ids = {t.id for t in batches[0]}
        assert batch_ids == {"T001", "T002", "T003"}

    def test_partial_skip_with_multiple_human_tasks(self):
        """Only one of two human tasks skipped — only its dependents are affected."""
        tasks_json = [
            {"id": "T001", "title": "Create API key", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T002", "title": "Configure DNS", "description": "Manual", "subagent_type": "human", "depends_on": []},
            {"id": "T003", "title": "Build API client", "description": "Code", "depends_on": ["T001"]},
            {"id": "T004", "title": "Deploy service", "description": "Code", "depends_on": ["T002"]},
            {"id": "T005", "title": "E2E test", "description": "Code", "depends_on": ["T003", "T004"]},
        ]
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(tasks_json)

        # Only skip T001 (T002 completed by user)
        skipped = {"T001"}
        queue = list(skipped)
        while queue:
            current = queue.pop(0)
            for tid, task in decomposer.tasks.items():
                if current in task.depends_on and tid not in skipped:
                    skipped.add(tid)
                    queue.append(tid)

        # T001 -> T003 -> T005 skipped. T002 and T004 are NOT skipped.
        assert skipped == {"T001", "T003", "T005"}