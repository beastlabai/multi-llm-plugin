"""Task decomposition module for breaking plans into executable tasks."""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pathlib import Path


class TaskStatus(Enum):
    """Status of a task in the decomposition."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a single task extracted from a plan."""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    files_to_modify: List[str] = field(default_factory=list)
    files_to_create: List[str] = field(default_factory=list)
    estimated_complexity: str = "medium"  # low, medium, high
    subagent_type: str = "general-purpose"
    acceptance_criteria: List[str] = field(default_factory=list)
    batch_group: Optional[int] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "files_to_modify": self.files_to_modify,
            "files_to_create": self.files_to_create,
            "estimated_complexity": self.estimated_complexity,
            "subagent_type": self.subagent_type,
            "acceptance_criteria": self.acceptance_criteria,
            "batch_group": self.batch_group,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        # Normalize acceptance_criteria: must be a list of non-empty strings.
        # Reject the entire list if any entry is invalid (non-string or empty).
        raw_criteria = data.get("acceptance_criteria", [])
        if (
            isinstance(raw_criteria, list)
            and raw_criteria
            and all(isinstance(entry, str) and entry.strip() for entry in raw_criteria)
        ):
            acceptance_criteria = raw_criteria
        else:
            acceptance_criteria = []

        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "pending")),
            depends_on=data.get("depends_on", []),
            files_to_modify=data.get("files_to_modify", []),
            files_to_create=data.get("files_to_create", []),
            estimated_complexity=data.get("estimated_complexity", "medium"),
            subagent_type=data.get("subagent_type", "general-purpose"),
            acceptance_criteria=acceptance_criteria,
            batch_group=data.get("batch_group"),
            error_message=data.get("error_message"),
        )


class TaskDecomposer:
    """Decomposes plans into executable tasks with dependency management."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []

    def parse_plan(self, plan_content: str) -> List[Task]:
        """
        Parse a plan document and extract tasks.

        Args:
            plan_content: Markdown content of the plan

        Returns:
            List of Task objects
        """
        tasks = []

        # Look for task sections with various formats:
        # - "### Step 1: Title" or "## Step 1: Title"
        # - "### Task 1: Title" or "## Task 1: Title"
        # - "### T1: Title" or "## 1: Title"
        task_pattern = r'(?:#{2,3})\s*(?:Step|Task)?\s*([T]?\d+):?\s*(.+?)(?=\n#{2,3}\s*(?:Step|Task)?\s*[T]?\d|\n## |\Z)'
        matches = re.findall(task_pattern, plan_content, re.DOTALL | re.IGNORECASE)

        for task_id, content in matches:
            task_id = task_id.strip()
            lines = content.strip().split('\n')
            title = lines[0].strip() if lines else f"Task {task_id}"
            description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

            # Normalize task ID
            normalized_id = f"T{task_id}" if not task_id.startswith('T') else task_id

            # Handle duplicate IDs by appending a suffix
            if normalized_id in self.tasks:
                suffix = 1
                while f"{normalized_id}_{suffix}" in self.tasks:
                    suffix += 1
                normalized_id = f"{normalized_id}_{suffix}"

            # Extract acceptance criteria before stripping metadata
            acceptance_criteria = self._extract_acceptance_criteria(description)

            # Extract structured metadata before stripping
            depends_on = self._extract_dependencies(description)
            files_to_modify = self._extract_files(description, "modify")
            files_to_create = self._extract_files(description, "create")
            estimated_complexity = self._estimate_complexity(description)

            # Strip metadata sections to keep only narrative content
            clean_description = self._strip_metadata_sections(description)

            task = Task(
                id=normalized_id,
                title=title,
                description=clean_description,
                depends_on=depends_on,
                files_to_modify=files_to_modify,
                files_to_create=files_to_create,
                estimated_complexity=estimated_complexity,
                acceptance_criteria=acceptance_criteria,
                # subagent_type defaults to "general-purpose"
                # Claude Code sets appropriate type during --generate-tasks
            )
            tasks.append(task)
            self.tasks[task.id] = task

        self._compute_execution_order()
        return tasks

    def parse_from_json(self, json_data: List[Dict[str, Any]]) -> List[Task]:
        """
        Parse tasks from JSON format (e.g., from LLM output).

        Args:
            json_data: List of task dictionaries

        Returns:
            List of Task objects
        """
        tasks = []
        for data in json_data:
            task = Task.from_dict(data)
            tasks.append(task)
            self.tasks[task.id] = task

        self._compute_execution_order()
        return tasks

    def _extract_dependencies(self, description: str) -> List[str]:
        """Extract task dependencies from description."""
        deps = []

        # Pattern: "depends on T001" or "**Depends on**: T001" or "after T002, T003"
        # The \*{0,2} handles optional markdown bold markers
        dep_patterns = [
            r'\*{0,2}depends?\*{0,2}\s+on\*{0,2}:?\s*([T\d,\s]+)',  # depends on / **depends on**
            r'\*{0,2}after\*{0,2}:?\s*([T\d,\s]+)',                  # after / **after**
            r'\*{0,2}requires?\*{0,2}:?\s*([T\d,\s]+)',              # requires / **requires**
            r'\*{0,2}dependencies?\*{0,2}:?\s*([T\d,\s]+)',          # dependencies / **dependencies**
        ]

        for pattern in dep_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                # Skip "none" or empty matches
                if match.strip().lower() in ('none', 'n/a', ''):
                    continue
                task_ids = re.findall(r'T?\d+', match)
                for tid in task_ids:
                    normalized = f"T{tid}" if not tid.startswith('T') else tid
                    if normalized not in deps:
                        deps.append(normalized)

        return deps

    def _extract_files(self, description: str, action: str) -> List[str]:
        """Extract file paths from description based on action type."""
        files = []

        # Look for file paths
        path_pattern = r'[`"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[`"]?'
        all_paths = re.findall(path_pattern, description)

        # Filter by action keywords near the path
        for path in all_paths:
            context_start = max(0, description.find(path) - 50)
            context = description[context_start:description.find(path) + len(path) + 50].lower()

            if action == "create" and any(w in context for w in ["create", "new", "add", "write"]):
                if path not in files:
                    files.append(path)
            elif action == "modify" and any(w in context for w in ["modify", "update", "edit", "change"]):
                if path not in files:
                    files.append(path)

        return files

    def _estimate_complexity(self, description: str) -> str:
        """Estimate task complexity based on description."""
        desc_lower = description.lower()

        high_indicators = ["complex", "refactor", "architecture", "multiple files", "integration"]
        low_indicators = ["simple", "minor", "typo", "rename", "comment"]

        for indicator in high_indicators:
            if indicator in desc_lower:
                return "high"

        for indicator in low_indicators:
            if indicator in desc_lower:
                return "low"

        return "medium"

    def _extract_acceptance_criteria(self, description: str) -> List[str]:
        """Extract acceptance criteria items from a task description.

        Looks for an **Acceptance Criteria** section (case-insensitive, with or
        without colon) and extracts bullet items (- item, - [ ] item) and
        numbered items (1. item).

        Args:
            description: Raw task description markdown

        Returns:
            List of acceptance criteria strings, empty if no section found.
        """
        # Match an Acceptance Criteria heading: **Acceptance Criteria**: or **Acceptance Criteria**
        # Case-insensitive, optional colon, optional bold markers
        heading_pattern = r'(?:^|\n)\s*\*{0,2}acceptance\s+criteria\*{0,2}\s*:?\s*\n'
        match = re.search(heading_pattern, description, re.IGNORECASE)
        if not match:
            return []

        # Extract the content after the heading until the next metadata section
        # heading or end of string
        rest = description[match.end():]
        # Stop at the next bold metadata heading or end of string
        next_section = re.search(r'\n\s*\*{2}[A-Za-z]', rest)
        if next_section:
            rest = rest[:next_section.start()]

        criteria = []
        for line in rest.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            # Match bullet items: - [ ] item, - [x] item, - item, * item
            bullet_match = re.match(r'^[-*]\s+(?:\[.\]\s+)?(.+)$', stripped)
            if bullet_match:
                item = bullet_match.group(1).strip()
                if item:
                    criteria.append(item)
                continue

            # Match numbered items: 1. item, 1) item
            numbered_match = re.match(r'^\d+[.)]\s+(.+)$', stripped)
            if numbered_match:
                item = numbered_match.group(1).strip()
                if item:
                    criteria.append(item)
                continue

        return criteria

    def _strip_metadata_sections(self, description: str) -> str:
        """Remove known metadata sections from a task description.

        Strips sections that start with **Section Name**: (or similar) on their
        own line, preserving only narrative text. Removed sections:
        - Files to modify
        - Files to create
        - Dependencies
        - Complexity
        - Subagent
        - Acceptance Criteria

        Args:
            description: Raw task description markdown

        Returns:
            Description with metadata sections removed and whitespace cleaned up.
        """
        section_names = [
            r'files?\s+to\s+modify',
            r'files?\s+to\s+create',
            r'dependenc(?:y|ies)',
            r'depends?\s+on',
            r'complexity',
            r'subagent(?:\s+type)?',
            r'acceptance\s+criteria',
        ]

        result = description
        for name in section_names:
            # Match the section heading and everything until the next bold
            # heading or end of string. The heading can be:
            # **Section Name**: content  (single line)
            # **Section Name**:\n- item\n- item  (multi-line with list items)
            # - Section Name: content (as a list item)
            pattern = (
                r'(?:(?<=\n)|(?:^))\s*'             # line start (zero-width)
                r'(?:\*{2}' + name + r'\*{2}'       # **Section Name**
                r'|[-*]\s+' + name + r')'           # or - Section Name
                r'\s*:?\s*'                         # optional colon
                r'[^\n]*'                           # rest of heading line
                r'(?:\n(?!\s*\n)(?!\s*\*{2}[A-Za-z])(?!\s*#{2,})[^\n]*)*'  # continuation lines
            )
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Clean up excessive blank lines left by removals
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    def _compute_execution_order(self) -> None:
        """Compute topological order for task execution."""
        # Kahn's algorithm for topological sort
        in_degree = {tid: 0 for tid in self.tasks}
        adj = {tid: [] for tid in self.tasks}

        for tid, task in self.tasks.items():
            for dep in task.depends_on:
                if dep in self.tasks:
                    adj[dep].append(tid)
                    in_degree[tid] += 1

        queue = [tid for tid, degree in in_degree.items() if degree == 0]
        self.execution_order = []

        while queue:
            # Sort queue to ensure deterministic order
            queue.sort()
            tid = queue.pop(0)
            self.execution_order.append(tid)

            for neighbor in adj[tid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Handle cycles (tasks not in execution_order)
        remaining = [tid for tid in self.tasks if tid not in self.execution_order]
        self.execution_order.extend(sorted(remaining))

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (no pending dependencies)."""
        ready = []
        completed_ids = {
            tid for tid, task in self.tasks.items()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        }

        for tid in self.execution_order:
            task = self.tasks[tid]
            if task.status != TaskStatus.PENDING:
                continue

            deps_satisfied = all(dep in completed_ids for dep in task.depends_on if dep in self.tasks)
            if deps_satisfied:
                ready.append(task)

        return ready

    def get_parallel_batches(self) -> List[List[Task]]:
        """
        Organize tasks into batches that can run in parallel.

        Uses a two-phase approach per iteration:
        1. Find all dependency-satisfied candidates
        2. Greedily assign candidates to the batch, skipping any whose
           files_to_modify + files_to_create overlap with already-batched tasks.
           Deferred candidates stay in remaining for a later batch.

        Tasks with empty file lists have no overlap and are always parallelizable.

        Returns:
            List of task batches, each batch can run in parallel
        """
        batches = []
        completed_ids: Set[str] = set()
        remaining = set(self.tasks.keys())

        batch_num = 0
        while remaining:
            # Phase 1: collect all dependency-satisfied candidates
            candidates = []
            for tid in sorted(remaining):
                task = self.tasks[tid]
                deps_satisfied = all(
                    dep in completed_ids
                    for dep in task.depends_on
                    if dep in self.tasks
                )
                if deps_satisfied:
                    candidates.append(task)

            if not candidates:
                # Remaining tasks have unsatisfied dependencies (cycle or missing)
                batch = []
                for tid in remaining:
                    task = self.tasks[tid]
                    task.batch_group = batch_num
                    batch.append(task)
                batches.append(batch)
                break

            # Phase 2: greedily assign candidates, checking file overlap
            batch = []
            batch_files: Set[str] = set()
            for task in candidates:
                task_files = set(task.files_to_modify + task.files_to_create)
                if task_files and task_files & batch_files:
                    # Overlapping files — defer to a later batch
                    continue
                task.batch_group = batch_num
                batch.append(task)
                batch_files.update(task_files)

            batches.append(batch)
            for task in batch:
                completed_ids.add(task.id)
                remaining.discard(task.id)
            batch_num += 1

        return batches

    def update_task_status(self, task_id: str, status: TaskStatus, error: Optional[str] = None) -> None:
        """Update the status of a task."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if error:
                self.tasks[task_id].error_message = error

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def to_json(self) -> str:
        """Export all tasks to JSON."""
        return json.dumps([task.to_dict() for task in self.tasks.values()], indent=2)

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of task progress."""
        status_counts = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            status_counts[task.status.value] += 1

        return {
            "total": len(self.tasks),
            "by_status": status_counts,
            "completion_rate": status_counts["completed"] / len(self.tasks) if self.tasks else 0,
        }


def decompose_plan_file(plan_path: Path) -> TaskDecomposer:
    """
    Convenience function to decompose a plan file.

    Args:
        plan_path: Path to the plan markdown file

    Returns:
        TaskDecomposer instance with parsed tasks
    """
    decomposer = TaskDecomposer()
    plan_content = plan_path.read_text(encoding="utf-8")
    decomposer.parse_plan(plan_content)
    return decomposer
