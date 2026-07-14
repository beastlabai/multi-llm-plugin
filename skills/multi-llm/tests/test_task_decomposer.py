"""Tests for task decomposer module."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.task_decomposer import (
    TaskDecomposer,
    TaskStatus,
    Task,
)


class TestTaskDecomposer:
    """Tests for TaskDecomposer class."""

    def test_parse_plan_extracts_tasks(self, sample_plan):
        """Test that parse_plan extracts tasks from plan content."""
        decomposer = TaskDecomposer()
        plan_content = sample_plan.read_text(encoding="utf-8")
        tasks = decomposer.parse_plan(plan_content)

        assert len(tasks) == 4
        assert "T001" in decomposer.tasks
        assert "T002" in decomposer.tasks

    def test_parse_plan_extracts_dependencies(self, sample_plan):
        """Test that dependencies are extracted correctly."""
        decomposer = TaskDecomposer()
        plan_content = sample_plan.read_text(encoding="utf-8")
        decomposer.parse_plan(plan_content)

        t002 = decomposer.get_task("T002")
        assert "T001" in t002.depends_on

        t004 = decomposer.get_task("T004")
        assert "T002" in t004.depends_on or "T003" in t004.depends_on

    def test_parse_from_json(self, sample_tasks_json):
        """Test parsing tasks from JSON format."""
        decomposer = TaskDecomposer()
        tasks = decomposer.parse_from_json(sample_tasks_json)

        assert len(tasks) == 3
        assert decomposer.get_task("T001").title == "Create directory structure"

    def test_get_ready_tasks_initial(self, sample_tasks_json):
        """Test getting ready tasks when no tasks completed."""
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(sample_tasks_json)

        ready = decomposer.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "T001"

    def test_get_ready_tasks_after_completion(self, sample_tasks_json):
        """Test getting ready tasks after completing some."""
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(sample_tasks_json)

        decomposer.update_task_status("T001", TaskStatus.COMPLETED)
        ready = decomposer.get_ready_tasks()

        assert len(ready) == 1
        assert ready[0].id == "T002"

    def test_get_parallel_batches(self, sample_tasks_json):
        """Test organizing tasks into parallel batches."""
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(sample_tasks_json)

        batches = decomposer.get_parallel_batches()

        # First batch should have T001 (no deps)
        assert len(batches) >= 1
        assert any(t.id == "T001" for t in batches[0])

    def test_task_status_update(self, sample_tasks_json):
        """Test updating task status."""
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(sample_tasks_json)

        decomposer.update_task_status("T001", TaskStatus.IN_PROGRESS)
        assert decomposer.get_task("T001").status == TaskStatus.IN_PROGRESS

        decomposer.update_task_status("T001", TaskStatus.COMPLETED)
        assert decomposer.get_task("T001").status == TaskStatus.COMPLETED

    def test_progress_summary(self, sample_tasks_json):
        """Test getting progress summary."""
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(sample_tasks_json)

        decomposer.update_task_status("T001", TaskStatus.COMPLETED)

        summary = decomposer.get_progress_summary()
        assert summary["total"] == 3
        assert summary["by_status"]["completed"] == 1
        assert summary["by_status"]["pending"] == 2

    def test_to_json_serialization(self, sample_tasks_json):
        """Test JSON serialization."""
        decomposer = TaskDecomposer()
        decomposer.parse_from_json(sample_tasks_json)

        json_output = decomposer.to_json()
        assert "T001" in json_output
        assert "Create directory structure" in json_output


class TestTask:
    """Tests for Task dataclass."""

    def test_task_to_dict(self):
        """Test Task to_dict conversion."""
        task = Task(
            id="T001",
            title="Test task",
            description="A test task",
            depends_on=["T000"],
            files_to_create=["test.py"]
        )

        d = task.to_dict()
        assert d["id"] == "T001"
        assert d["title"] == "Test task"
        assert d["depends_on"] == ["T000"]

    def test_task_from_dict(self):
        """Test Task from_dict creation."""
        data = {
            "id": "T001",
            "title": "Test task",
            "description": "A test task",
            "status": "in_progress",
            "depends_on": ["T000"]
        }

        task = Task.from_dict(data)
        assert task.id == "T001"
        assert task.status == TaskStatus.IN_PROGRESS

    def test_task_roundtrip_with_acceptance_criteria(self):
        """Test from_dict(to_dict(task)) preserves acceptance_criteria."""
        task = Task(
            id="T001",
            title="Test task",
            description="A test task",
            acceptance_criteria=["Criterion A", "Criterion B"],
        )

        roundtripped = Task.from_dict(task.to_dict())
        assert roundtripped.acceptance_criteria == ["Criterion A", "Criterion B"]

    def test_task_roundtrip_without_acceptance_criteria(self):
        """Test from_dict(to_dict(task)) yields [] when acceptance_criteria is absent."""
        task = Task(
            id="T001",
            title="Test task",
            description="A test task",
        )

        roundtripped = Task.from_dict(task.to_dict())
        assert roundtripped.acceptance_criteria == []

    def test_from_dict_defaults_non_list_criteria_to_empty(self):
        """Test from_dict() defaults non-list acceptance_criteria to []."""
        data = {
            "id": "T001",
            "title": "Test task",
            "description": "A test task",
            "acceptance_criteria": "not a list",
        }
        task = Task.from_dict(data)
        assert task.acceptance_criteria == []

    def test_from_dict_defaults_none_criteria_to_empty(self):
        """Test from_dict() defaults None acceptance_criteria to []."""
        data = {
            "id": "T001",
            "title": "Test task",
            "description": "A test task",
            "acceptance_criteria": None,
        }
        task = Task.from_dict(data)
        assert task.acceptance_criteria == []

    def test_from_dict_rejects_list_with_any_invalid_entry(self):
        """Test from_dict() rejects the entire list if any entry is invalid."""
        data = {
            "id": "T001",
            "title": "Test task",
            "description": "A test task",
            "acceptance_criteria": ["valid", "", "   ", None, 42, "also valid"],
        }
        task = Task.from_dict(data)
        assert task.acceptance_criteria == []


class TestExtractAcceptanceCriteria:
    """Tests for TaskDecomposer._extract_acceptance_criteria()."""

    def test_extracts_bullet_list(self):
        """Test extraction from bullet list with - items and - [ ] items."""
        description = """Some narrative text.

**Acceptance Criteria**:
- First criterion
- [ ] Second criterion
- [x] Third criterion
"""
        decomposer = TaskDecomposer()
        result = decomposer._extract_acceptance_criteria(description)
        assert result == ["First criterion", "Second criterion", "Third criterion"]

    def test_extracts_numbered_list(self):
        """Test extraction from numbered list (1. item)."""
        description = """Some narrative text.

**Acceptance Criteria**:
1. First numbered criterion
2. Second numbered criterion
3. Third numbered criterion
"""
        decomposer = TaskDecomposer()
        result = decomposer._extract_acceptance_criteria(description)
        assert result == [
            "First numbered criterion",
            "Second numbered criterion",
            "Third numbered criterion",
        ]

    def test_returns_empty_when_section_missing(self):
        """Test returns [] when Acceptance Criteria section is missing."""
        description = """Some narrative text.

This description has no acceptance criteria section.
Just regular content.
"""
        decomposer = TaskDecomposer()
        result = decomposer._extract_acceptance_criteria(description)
        assert result == []

    def test_extracts_with_star_bullets(self):
        """Test extraction from * bullet items."""
        description = """Description.

**Acceptance Criteria**:
* Star bullet one
* Star bullet two
"""
        decomposer = TaskDecomposer()
        result = decomposer._extract_acceptance_criteria(description)
        assert result == ["Star bullet one", "Star bullet two"]

    def test_stops_at_next_bold_heading(self):
        """Test extraction stops at the next bold metadata heading."""
        description = """Description.

**Acceptance Criteria**:
- Criterion one
- Criterion two

**Dependencies**: T001, T002
"""
        decomposer = TaskDecomposer()
        result = decomposer._extract_acceptance_criteria(description)
        assert result == ["Criterion one", "Criterion two"]


class TestStripMetadataSections:
    """Tests for TaskDecomposer._strip_metadata_sections()."""

    def test_removes_metadata_preserves_narrative(self):
        """Test that metadata sections are removed while narrative text is preserved."""
        description = """This is the narrative description of the task.
It explains what needs to be done.

**Files to modify**: src/main.py, src/utils.py

**Dependencies**: T001, T002

**Acceptance Criteria**:
- First criterion
- Second criterion

**Complexity**: high
"""
        decomposer = TaskDecomposer()
        result = decomposer._strip_metadata_sections(description)

        # Narrative should be preserved
        assert "This is the narrative description" in result
        assert "explains what needs to be done" in result

        # Metadata should be removed
        assert "Files to modify" not in result
        assert "src/main.py" not in result
        assert "Dependencies" not in result
        assert "Acceptance Criteria" not in result
        assert "First criterion" not in result
        assert "Complexity" not in result

    def test_removes_metadata_first_preserves_narrative(self):
        """Test real-world format: metadata sections first, then narrative text.

        update_plan_tasks.py generates tasks with metadata at the top
        (Dependencies, Files to modify, Complexity, Subagent on consecutive
        lines), then a blank line, then narrative text, then Acceptance Criteria.
        This verifies _strip_metadata_sections handles that layout correctly.
        """
        description = (
            "**Dependencies**: T001, T003\n"
            "**Files to modify**: src/components/form.tsx, src/utils/validate.ts\n"
            "**Complexity**: high\n"
            "**Subagent type**: general-purpose\n"
            "\n"
            "Refactor the form validation logic to use a schema-based approach.\n"
            "The current inline validation is duplicated across multiple components\n"
            "and should be centralized into a shared utility.\n"
            "\n"
            "**Acceptance Criteria**:\n"
            "- All form components use the new schema validator\n"
            "- No inline validation remains in component files\n"
            "- Unit tests cover all validation rules"
        )
        decomposer = TaskDecomposer()
        result = decomposer._strip_metadata_sections(description)

        # Narrative should be preserved
        assert "Refactor the form validation logic" in result
        assert "schema-based approach" in result
        assert "centralized into a shared utility" in result

        # Metadata should be removed
        assert "Dependencies" not in result
        assert "T001" not in result
        assert "Files to modify" not in result
        assert "src/components/form.tsx" not in result
        assert "Complexity" not in result
        assert "Subagent type" not in result
        assert "general-purpose" not in result
        assert "Acceptance Criteria" not in result
        assert "schema validator" not in result

    def test_preserves_description_without_metadata(self):
        """Test that descriptions without metadata are returned unchanged."""
        description = "Just a plain description with no metadata."
        decomposer = TaskDecomposer()
        result = decomposer._strip_metadata_sections(description)
        assert result == description

    def test_removes_depends_on_section(self):
        """Test removal of Depends on section."""
        description = """Narrative text.

**Depends on**: T001
"""
        decomposer = TaskDecomposer()
        result = decomposer._strip_metadata_sections(description)
        assert "Depends on" not in result
        assert "Narrative text" in result

    def test_removes_subagent_type_section(self):
        """Test removal of Subagent type section."""
        description = """Narrative text.

**Subagent type**: general-purpose
"""
        decomposer = TaskDecomposer()
        result = decomposer._strip_metadata_sections(description)
        assert "Subagent" not in result
        assert "general-purpose" not in result
        assert "Narrative text" in result

    def test_removes_files_to_create_section(self):
        """Test removal of Files to create section."""
        description = """Narrative text.

**Files to create**: src/new.py
"""
        decomposer = TaskDecomposer()
        result = decomposer._strip_metadata_sections(description)
        assert "Files to create" not in result
        assert "Narrative text" in result


class TestFileOverlapDetection:
    """Tests for file-overlap detection in get_parallel_batches()."""

    def test_overlapping_files_to_modify_separate_batches(self):
        """Two independent tasks both modifying src/shared.py should be in separate batches."""
        decomposer = TaskDecomposer()
        t1 = Task(id="T001", title="Task 1", description="desc", files_to_modify=["src/shared.py"])
        t2 = Task(id="T002", title="Task 2", description="desc", files_to_modify=["src/shared.py"])
        decomposer.tasks = {"T001": t1, "T002": t2}
        decomposer.execution_order = ["T001", "T002"]

        batches = decomposer.get_parallel_batches()

        # They share a file, so they must NOT be in the same batch
        assert len(batches) >= 2
        batch_0_ids = {t.id for t in batches[0]}
        batch_1_ids = {t.id for t in batches[1]}
        assert "T001" in batch_0_ids or "T001" in batch_1_ids
        assert "T002" in batch_1_ids or "T002" in batch_0_ids
        # Crucially, they must be in different batches
        assert batch_0_ids & batch_1_ids == set()

    def test_non_overlapping_files_same_batch(self):
        """Two independent tasks modifying different files should be in the same batch."""
        decomposer = TaskDecomposer()
        t1 = Task(id="T001", title="Task 1", description="desc", files_to_modify=["src/a.py"])
        t2 = Task(id="T002", title="Task 2", description="desc", files_to_modify=["src/b.py"])
        decomposer.tasks = {"T001": t1, "T002": t2}
        decomposer.execution_order = ["T001", "T002"]

        batches = decomposer.get_parallel_batches()

        assert len(batches) == 1
        batch_0_ids = {t.id for t in batches[0]}
        assert batch_0_ids == {"T001", "T002"}

    def test_overlap_between_create_and_modify_separate_batches(self):
        """Task A creates src/new.py, Task B modifies src/new.py → separate batches."""
        decomposer = TaskDecomposer()
        t1 = Task(id="T001", title="Task 1", description="desc", files_to_create=["src/new.py"])
        t2 = Task(id="T002", title="Task 2", description="desc", files_to_modify=["src/new.py"])
        decomposer.tasks = {"T001": t1, "T002": t2}
        decomposer.execution_order = ["T001", "T002"]

        batches = decomposer.get_parallel_batches()

        assert len(batches) >= 2
        batch_0_ids = {t.id for t in batches[0]}
        batch_1_ids = {t.id for t in batches[1]}
        assert batch_0_ids & batch_1_ids == set()

    def test_no_files_listed_same_batch(self):
        """Two independent tasks with empty file lists should be in the same batch."""
        decomposer = TaskDecomposer()
        t1 = Task(id="T001", title="Task 1", description="desc")
        t2 = Task(id="T002", title="Task 2", description="desc")
        decomposer.tasks = {"T001": t1, "T002": t2}
        decomposer.execution_order = ["T001", "T002"]

        batches = decomposer.get_parallel_batches()

        assert len(batches) == 1
        batch_0_ids = {t.id for t in batches[0]}
        assert batch_0_ids == {"T001", "T002"}

    def test_three_tasks_two_overlapping_one_not(self):
        """T001 and T002 both modify src/a.py, T003 modifies src/b.py → T001+T003 in batch 0, T002 in batch 1."""
        decomposer = TaskDecomposer()
        t1 = Task(id="T001", title="Task 1", description="desc", files_to_modify=["src/a.py"])
        t2 = Task(id="T002", title="Task 2", description="desc", files_to_modify=["src/a.py"])
        t3 = Task(id="T003", title="Task 3", description="desc", files_to_modify=["src/b.py"])
        decomposer.tasks = {"T001": t1, "T002": t2, "T003": t3}
        decomposer.execution_order = ["T001", "T002", "T003"]

        batches = decomposer.get_parallel_batches()

        assert len(batches) == 2
        batch_0_ids = {t.id for t in batches[0]}
        batch_1_ids = {t.id for t in batches[1]}
        assert "T001" in batch_0_ids
        assert "T003" in batch_0_ids
        assert "T002" in batch_1_ids
