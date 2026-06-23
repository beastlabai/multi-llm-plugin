#!/usr/bin/env python3
"""Update plan file with generated tasks.

Usage:
    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/update_plan_tasks.py --plan-file <plan_path> --tasks-file <tasks_json_path>

This script:
1. Reads the tasks JSON file
2. Creates the tasks markdown file in {plan}/tasks/tasks.md
3. Updates the plan file with a reference to the tasks file
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.output_handler import get_output_paths, get_relative_output_path
from utils.plan_updater import insert_tasks_file_reference
from utils.state_manager import get_or_create_state


COMPLEXITY_COLORS = {
    "low": "#90ee90",
    "medium": "#f4a460",
    "high": "#cd5c5c",
}


def generate_dependency_diagram(tasks: List[Dict[str, Any]]) -> str:
    """
    Generate a Mermaid graph TD diagram showing task dependencies.

    Args:
        tasks: List of task dictionaries with id, title, depends_on, estimated_complexity

    Returns:
        Mermaid diagram wrapped in a fenced code block
    """
    lines = ["```mermaid", "graph TD"]

    # Node declarations
    for task in tasks:
        task_id = task.get("id", "T000")
        title = task.get("title", "Untitled")
        if len(title) > 40:
            title = title[:37] + "..."
        # Escape quotes in title for Mermaid
        title = title.replace('"', '#quot;')
        if task.get("subagent_type") == "human":
            lines.append(f'    {task_id}(["{task_id}: {title}"])')
        else:
            lines.append(f'    {task_id}["{task_id}: {title}"]')

    lines.append("")

    # Edges from depends_on
    has_edges = False
    for task in tasks:
        task_id = task.get("id", "T000")
        for dep in task.get("depends_on", []):
            lines.append(f"    {dep} --> {task_id}")
            has_edges = True

    if has_edges:
        lines.append("")

    # Style nodes by complexity
    for task in tasks:
        task_id = task.get("id", "T000")
        if task.get("subagent_type") == "human":
            color = "#87ceeb"  # Light blue for human/manual tasks
        else:
            complexity = task.get("estimated_complexity", "medium")
            color = COMPLEXITY_COLORS.get(complexity, COMPLEXITY_COLORS["medium"])
        lines.append(f"    style {task_id} fill:{color}")

    lines.append("```")
    return "\n".join(lines)


def generate_tasks_markdown(tasks: List[Dict[str, Any]], plan_name: str) -> str:
    """
    Generate markdown content for the tasks file.

    Args:
        tasks: List of task dictionaries
        plan_name: Name of the parent plan file

    Returns:
        Formatted markdown string
    """
    lines = [
        f"# Implementation Tasks for {plan_name}",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"**Parent Plan**: [{plan_name}](../{plan_name})",
        "",
        "---",
        "",
    ]

    # Dependency diagram
    if len(tasks) > 1:
        lines.append("## Dependency Graph")
        lines.append("")
        lines.append(generate_dependency_diagram(tasks))
        lines.append("")
        lines.append("---")
        lines.append("")

    for task in tasks:
        task_id = task.get("id", "T000")
        title = task.get("title", "Untitled Task")
        description = task.get("description", "")
        depends_on = task.get("depends_on", [])
        files_to_modify = task.get("files_to_modify", [])
        files_to_create = task.get("files_to_create", [])
        acceptance_criteria = task.get("acceptance_criteria", [])
        complexity = task.get("estimated_complexity", "medium")
        subagent_type = task.get("subagent_type", "general-purpose")

        # Task header
        lines.append(f"### Task {task_id}: {title}")

        # Dependencies
        if depends_on:
            deps_str = ", ".join(depends_on)
            lines.append(f"**Dependencies**: {deps_str}")
        else:
            lines.append("**Dependencies**: None")

        # Files
        if files_to_modify:
            lines.append(f"**Files to modify**: {', '.join(files_to_modify)}")
        if files_to_create:
            lines.append(f"**Files to create**: {', '.join(files_to_create)}")

        # Complexity and subagent
        lines.append(f"**Complexity**: {complexity}")
        if subagent_type == "human":
            lines.append("**Type**: Manual (user action required)")
        else:
            lines.append(f"**Subagent**: {subagent_type}")
        lines.append("")

        # Description
        if description:
            lines.append(description)
            lines.append("")

        # Acceptance criteria
        if acceptance_criteria:
            lines.append("**Acceptance Criteria**:")
            for criterion in acceptance_criteria:
                lines.append(f"- [ ] {criterion}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return '\n'.join(lines)


def validate_tasks_json(data: Any) -> tuple[bool, str]:
    """
    Validate the tasks JSON structure.

    Args:
        data: Parsed JSON data

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Root must be an object with 'tasks' array"

    if "tasks" not in data:
        return False, "Missing 'tasks' field"

    tasks = data["tasks"]
    if not isinstance(tasks, list):
        return False, "'tasks' must be an array"

    if len(tasks) == 0:
        return False, "'tasks' array is empty"

    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            return False, f"Task {i} is not an object"
        if "id" not in task:
            return False, f"Task {i} missing 'id' field"
        if "title" not in task:
            return False, f"Task {i} missing 'title' field"

    return True, ""


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update plan file with generated tasks"
    )
    parser.add_argument(
        "--plan-file",
        required=True,
        help="Path to the plan markdown file"
    )
    parser.add_argument(
        "--tasks-file",
        required=True,
        help="Path to the tasks JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files"
    )
    args = parser.parse_args()

    plan_path = Path(args.plan_file).resolve()
    tasks_json_path = Path(args.tasks_file).resolve()

    # Validate inputs
    if not plan_path.exists():
        print(f"ERROR: Plan file not found: {plan_path}", file=sys.stderr)
        return 1

    if not tasks_json_path.exists():
        print(f"ERROR: Tasks JSON file not found: {tasks_json_path}", file=sys.stderr)
        return 1

    # Read and parse tasks JSON
    try:
        tasks_data = json.loads(tasks_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse tasks JSON: {e}", file=sys.stderr)
        return 1

    # Validate structure
    is_valid, error_msg = validate_tasks_json(tasks_data)
    if not is_valid:
        print(f"ERROR: Invalid tasks JSON: {error_msg}", file=sys.stderr)
        return 1

    tasks = tasks_data["tasks"]
    preamble = tasks_data.get("plan_preamble", "")

    # Generate tasks markdown
    plan_name = plan_path.name
    tasks_markdown = generate_tasks_markdown(tasks, plan_name)

    # Get output path for tasks file
    tasks_output_path = get_output_paths(plan_path, "tasks", phase="tasks")
    tasks_relative_path = get_relative_output_path(plan_path, "tasks", phase="tasks")

    # Read plan content
    plan_content = plan_path.read_text(encoding="utf-8")

    # Update plan with tasks file reference
    updated_plan = insert_tasks_file_reference(plan_content, tasks_relative_path)

    if args.dry_run:
        print("Dry run - would perform these actions:")
        print(f"\n1. Create tasks file: {tasks_output_path}")
        print(f"   ({len(tasks)} tasks)")
        print(f"\n2. Update plan file: {plan_path}")
        print(f"   Add reference: <!-- TASKS_FILE: {tasks_relative_path} -->")
        return 0

    # Create tasks directory and file
    tasks_output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_output_path.write_text(tasks_markdown, encoding="utf-8")
    print(f"Created tasks file: {tasks_output_path}")

    # Update plan file
    plan_path.write_text(updated_plan, encoding="utf-8")
    print(f"Updated plan file: {plan_path}")

    # Mark generate-tasks phase as completed in state
    state = get_or_create_state(plan_path)
    state.mark_phase_completed("generate-tasks")
    state.save()

    # Store plan preamble for use during implementation
    if preamble:
        state.set("plan_preamble", preamble)
        state.save()
        print(f"Stored plan preamble ({len(preamble)} chars)")

    # Summary
    print(f"\nSuccessfully generated {len(tasks)} tasks:")
    for task in tasks:
        deps = task.get("depends_on", [])
        deps_str = f" (depends on: {', '.join(deps)})" if deps else ""
        print(f"  - {task['id']}: {task['title']}{deps_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
