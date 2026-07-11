#!/usr/bin/env python3
"""
Implementation orchestrator for preparing tasks for Claude Code execution.

This orchestrator takes a plan file, decomposes it into tasks, organizes them
into dependency-respecting batches, and outputs JSON for Claude Code to execute
using its native Task tool.

Usage:
    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/implement_orchestrator.py --plan-file plans/my-plan.md [--output tasks.json]
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    TaskDecomposer,
    TaskStatus,
    Task,
    StateManager,
    get_or_create_state,
    get_tasks_file_path,
    load_prompt,
    get_output_paths,
    get_relative_output_path,
    get_modified_files,
    get_staged_files,
    get_current_head,
)
from utils.stream_bootstrap import bootstrap_streams
from utils.output_handler import get_phase_dir, get_output_dir
from utils.git_utils import get_project_root
from check_workflow_prerequisites import check_apply_task_suggestions_prerequisite

logger = logging.getLogger(__name__)


def check_apply_suggestions_prerequisite(state, plan_dir: Path) -> dict:
    """Check if apply-suggestions phase should run before proceeding."""
    review_plan_dir = plan_dir / "review-plan"
    validation_path = review_plan_dir / "validation.json"

    # No review-plan run → no prerequisite
    if not validation_path.exists():
        return {"met": True, "reason": "No review-plan results"}

    # Check if apply-suggestions completed or skipped
    if state.is_phase_completed("apply-suggestions"):
        return {"met": True, "reason": "Apply-suggestions completed"}
    if state.is_phase_skipped("apply-suggestions"):
        return {"met": True, "reason": "Apply-suggestions skipped"}

    # Count valid suggestions
    try:
        with open(validation_path, 'r', encoding='utf-8') as f:
            validation = json.load(f)
        valid_count = sum(1 for v in validation.values() if v.get("status") == "valid")
        if valid_count == 0:
            return {"met": True, "reason": "No valid suggestions"}
        return {
            "met": False,
            "reason": f"{valid_count} unapplied suggestions",
            "valid_count": valid_count
        }
    except (json.JSONDecodeError, OSError):
        return {"met": True, "reason": "Could not read validation"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare implementation tasks from a plan file for Claude Code execution"
    )

    parser.add_argument(
        '--plan-file',
        required=True,
        help='Path to the implementation plan markdown file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for tasks JSON (default: '
             '<git-root>/.multi-llm/tmp/implementation_tasks_{plan_stem}.json, '
             'resolved from the plan file after argument parsing)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous session state (skip completed tasks)'
    )

    parser.add_argument(
        '--task',
        type=str,
        help='Output only a specific task ID'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show task summary without writing output file'
    )

    return parser.parse_args()


def resolve_default_output(plan_path: Path) -> str:
    """Resolve the default --output path anchored at the git project root.

    The default lives inside the repo workspace
    (``<git-root>/.multi-llm/tmp/implementation_tasks_{plan_stem}.json``) so
    the Bash side and harness Read/Write tools resolve the same file on every
    OS. This must stay byte-identical to the path documented in
    ``instructions/implement.md``. Never fall back to the system temp dir
    (``tempfile.gettempdir()``): on Git for Windows the MSYS ``/tmp`` and the
    native temp dir are different directories, which is exactly the boundary
    hazard this default avoids.

    Args:
        plan_path: Resolved path to the plan file (used both for git-root
            detection and for the ``{plan_stem}`` filename suffix).

    Returns:
        Absolute output path as a string.

    Exits:
        With status 1 when the plan file is not inside a git work tree.
    """
    project_root = get_project_root(str(plan_path))
    if not project_root:
        print("ERROR: multi-llm requires running inside a git repository")
        print("       (the default --output path is anchored at the git root; "
              "run from within a repository or pass --output explicitly)")
        sys.exit(1)
    return str(
        Path(project_root) / ".multi-llm" / "tmp"
        / f"implementation_tasks_{plan_path.stem}.json"
    )


def _write_temp_dir_gitignore(directory: Path) -> None:
    """Make ``.multi-llm/tmp/`` self-ignoring by writing ``.gitignore`` (``*``).

    Only applies when ``directory`` actually is a ``.multi-llm/tmp`` dir (a
    user-supplied ``--output`` elsewhere must not receive a stray ignore-all
    file). Keeps transient run artifacts out of ``git status`` regardless of
    whether the consuming repo ever ran ``--init --gitignore``.
    """
    if directory.name == "tmp" and directory.parent.name == ".multi-llm":
        gitignore_path = directory / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text("*\n", encoding="utf-8")


def record_pre_existing_changes(state: StateManager) -> None:
    """
    Record files already modified before implementation starts.

    This captures all uncommitted changes (staged and unstaged) present in the
    working tree at implementation startup. These are later subtracted from
    the final changed files to isolate only changes made during implementation.

    Args:
        state: StateManager instance to store the pre-existing changes
    """
    # Only record if not already recorded (avoid overwriting on --resume)
    if state.get("pre_existing_changes") is not None:
        print(f"Pre-existing changes already recorded ({len(state.get('pre_existing_changes', []))} files)")
        return

    # Get all modified files (staged and unstaged)
    # Filter empty strings before set union for robustness
    try:
        unstaged = set(f for f in get_modified_files() if f)
        staged = set(f for f in get_staged_files() if f)

        pre_existing = unstaged | staged

        state.set("pre_existing_changes", list(pre_existing))
        state.save()

        if pre_existing:
            print(f"Recorded {len(pre_existing)} pre-existing changed files")
        else:
            print("No pre-existing changes to record")
    except Exception as e:
        print(f"Warning: Could not record pre-existing changes: {e}")
        state.set("pre_existing_changes", [])
        state.save()


def load_implementation_prompt() -> str:
    """Load the implementation task prompt template."""
    try:
        return load_prompt("implementation_task.txt")
    except FileNotFoundError:
        # Fallback prompt template
        return """## Task: {task_title}

### Description
{task_description}

### Files to Consider
{relevant_files}

### Files to Create/Modify
{output_files}

### Instructions
1. Implement the task as described
2. Follow existing code patterns and conventions
3. Write clean, maintainable code
4. Add appropriate error handling
5. Do NOT make git commits - the orchestrator handles git state

### Output
After implementation, describe what you did and any issues encountered.
"""


def _build_acceptance_criteria_section(criteria: List[str]) -> str:
    """Build the acceptance criteria section from a list of criteria.

    Args:
        criteria: List of acceptance criteria strings.

    Returns:
        Formatted section string, or empty string if no valid criteria.
    """
    MAX_CRITERIA = 10
    MAX_CRITERION_CHARS = 200

    # Filter out empty/whitespace-only strings
    valid = [c.strip() for c in criteria if isinstance(c, str) and c.strip()]
    if not valid:
        return ""

    overflow = len(valid) - MAX_CRITERIA
    items = valid[:MAX_CRITERIA]

    lines = ["## Acceptance Criteria"]
    for item in items:
        if len(item) > MAX_CRITERION_CHARS:
            item = item[:MAX_CRITERION_CHARS - 3] + "..."
        lines.append(f"- [ ] {item}")

    if overflow > 0:
        lines.append(f"_(+{overflow} more criteria; see full plan)_")

    return "\n".join(lines) + "\n"


def _build_dependency_section(
    task: Task,
    all_tasks: Optional[Dict[str, Task]],
) -> str:
    """Build the preceding tasks section from dependency information.

    Args:
        task: The current task being rendered.
        all_tasks: Dictionary of all tasks keyed by ID, or None.

    Returns:
        Formatted section string, or empty string if no dependencies.
    """
    MAX_DEPS = 5
    MAX_SUMMARY_CHARS = 300

    if not task.depends_on or all_tasks is None:
        return ""

    visited: set = set()
    summaries: List[str] = []

    for dep_id in sorted(task.depends_on):
        # Circular dependency guard: skip if this task's own ID appears
        if dep_id == task.id:
            logger.warning(
                "Circular dependency: task %s depends on itself; skipping",
                task.id,
            )
            continue

        # Skip duplicates
        if dep_id in visited:
            logger.warning(
                "Duplicate dependency %s in task %s; skipping",
                dep_id,
                task.id,
            )
            continue
        visited.add(dep_id)

        dep_task = all_tasks.get(dep_id)
        if dep_task is None:
            logger.warning(
                "Dependency %s not found in all_tasks for task %s; skipping",
                dep_id,
                task.id,
            )
            continue

        # Build summary: title + first line of description
        summary = f"**{dep_id}: {dep_task.title}**"
        if dep_task.description:
            first_line = dep_task.description.split("\n")[0].strip()
            if first_line:
                summary += f" - {first_line}"

        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS - 3] + "..."

        summaries.append(summary)

        if len(summaries) >= MAX_DEPS:
            break

    if not summaries:
        return ""

    lines = ["## Preceding Tasks (already completed)"]
    for s in summaries:
        lines.append(f"- {s}")

    # Count overflow: total unique valid deps minus what we rendered
    total_valid = 0
    seen: set = set()
    for dep_id in task.depends_on:
        if dep_id == task.id or dep_id in seen:
            continue
        seen.add(dep_id)
        if all_tasks.get(dep_id) is not None:
            total_valid += 1
    overflow = total_valid - len(summaries)
    if overflow > 0:
        lines.append(f"_(+{overflow} more preceding tasks; see full plan)_")

    return "\n".join(lines) + "\n"


def _enforce_token_budget(
    acceptance_section: str,
    dependency_section: str,
    budget: int = 2000,
) -> tuple[str, str]:
    """Enforce a combined character budget on the two sections.

    Truncates dependency section first, then acceptance criteria section.

    Args:
        acceptance_section: Rendered acceptance criteria section.
        dependency_section: Rendered dependency section.
        budget: Maximum combined character count.

    Returns:
        Tuple of (acceptance_section, dependency_section) possibly truncated.
    """
    combined = len(acceptance_section) + len(dependency_section)
    if combined <= budget:
        return acceptance_section, dependency_section

    # Truncate dependency section first
    if dependency_section and len(dependency_section) > budget - len(acceptance_section):
        allowed = max(0, budget - len(acceptance_section))
        if allowed < 50:
            # Not enough room for any meaningful dependency info
            dependency_section = ""
        else:
            dependency_section = dependency_section[:allowed - 30] + "\n_(truncated)_\n"

    # If still over budget, truncate acceptance criteria
    combined = len(acceptance_section) + len(dependency_section)
    if combined > budget:
        allowed = max(0, budget - len(dependency_section))
        if allowed < 50:
            acceptance_section = ""
        else:
            acceptance_section = acceptance_section[:allowed - 30] + "\n_(truncated)_\n"

    return acceptance_section, dependency_section


def build_task_prompt(
    task: Task,
    plan_preamble: str = "",
    plan_path: str = "",
    all_tasks: Optional[Dict[str, Task]] = None,
) -> str:
    """Build the full prompt for a task.

    Args:
        task: The task to build a prompt for.
        plan_preamble: Plan context preamble text.
        plan_path: Path to the plan file.
        all_tasks: Optional dictionary of all tasks for dependency context.

    Returns:
        Formatted prompt string.
    """
    prompt_template = load_implementation_prompt()

    # Combine output files from task
    output_files = task.files_to_create + task.files_to_modify

    # Build acceptance criteria section
    acceptance_criteria_section = _build_acceptance_criteria_section(
        task.acceptance_criteria if task.acceptance_criteria else []
    )

    # Build dependency section
    dependency_section = _build_dependency_section(task, all_tasks)

    # Enforce combined budget
    acceptance_criteria_section, dependency_section = _enforce_token_budget(
        acceptance_criteria_section, dependency_section
    )

    # Use safe format that handles missing keys gracefully
    return prompt_template.format(
        task_title=task.title,
        task_description=task.description,
        relevant_files="\n".join(f"- {f}" for f in task.files_to_modify) if task.files_to_modify else "- Explore codebase as needed",
        output_files="\n".join(f"- {f}" for f in output_files) if output_files else "- Files as needed for the task",
        plan_preamble=plan_preamble if plan_preamble else "See the full plan below for context.",
        plan_path=plan_path if plan_path else "(plan path not available)",
        acceptance_criteria_section=acceptance_criteria_section,
        dependency_section=dependency_section,
    )


def format_task_for_output(
    task: Task,
    plan_preamble: str = "",
    plan_path: str = "",
    all_tasks: Optional[Dict[str, Task]] = None,
) -> Dict[str, Any]:
    """Format a task for JSON output."""
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "prompt": build_task_prompt(task, plan_preamble, plan_path, all_tasks=all_tasks),
        "subagent_type": task.subagent_type,
        "is_human": task.subagent_type == "human",
        "depends_on": task.depends_on,
        "files_to_modify": task.files_to_modify,
        "files_to_create": task.files_to_create,
        "complexity": task.estimated_complexity,
        "status": task.status.value,
    }


def format_batches_for_output(
    batches: List[List[Task]],
    skip_completed: bool = False,
    plan_preamble: str = "",
    plan_path: str = "",
    all_tasks: Optional[Dict[str, Task]] = None,
) -> List[Dict[str, Any]]:
    """Format task batches for JSON output."""
    output_batches = []

    for batch_idx, batch in enumerate(batches):
        # Filter tasks if skipping completed
        tasks_to_include = batch
        if skip_completed:
            tasks_to_include = [t for t in batch if t.status == TaskStatus.PENDING]

        if not tasks_to_include:
            continue

        output_batches.append({
            "batch_index": batch_idx,
            "can_parallelize": len(tasks_to_include) > 1,
            "tasks": [format_task_for_output(t, plan_preamble, plan_path, all_tasks=all_tasks) for t in tasks_to_include]
        })

    return output_batches


def main():
    """Main entry point."""
    bootstrap_streams()
    args = parse_args()

    # Validate plan file (resolve to absolute path first to handle
    # uv --directory which changes the working directory)
    plan_path = Path(args.plan_file).resolve()
    if not plan_path.exists():
        print(f"ERROR: Plan file not found: {args.plan_file}")
        print(f"       Resolved path: {plan_path}")
        sys.exit(1)

    # Resolve the default --output now that the plan argument is known (the
    # default depends on the git root and the plan stem, so it cannot be a
    # static argparse default).
    if args.output is None:
        args.output = resolve_default_output(plan_path)

    # Load plan
    plan_content = plan_path.read_text(encoding="utf-8")
    print(f"Loaded plan: {plan_path}")

    # Initialize state manager
    state = get_or_create_state(plan_path)
    print(f"State file: {state.state_file}")
    state.save()

    # Load plan preamble for task prompts
    plan_preamble = state.get("plan_preamble") or ""
    plan_path_str = str(plan_path.absolute())

    # Check prerequisites
    plan_dir = get_output_dir(plan_path)

    # Check for unapplied suggestions
    apply_check = check_apply_suggestions_prerequisite(state, plan_dir)
    if not apply_check["met"]:
        prereq_info = {
            "marker": "PREREQUISITE_CHECK",
            "phase": "apply-suggestions",
            "reason": apply_check["reason"],
            "valid_count": apply_check.get("valid_count", 0)
        }
        print(f"[PREREQUISITE_CHECK]")
        print(json.dumps(prereq_info, indent=2))
        sys.exit(0)

    # Check for unapplied task suggestions (soft advisory — does NOT block)
    task_suggestions_check = check_apply_task_suggestions_prerequisite(state, plan_dir)
    if not task_suggestions_check["met"]:
        advisory_info = {
            "marker": "TASK_SUGGESTIONS_ADVISORY",
            "phase": "apply-task-suggestions",
            "reason": task_suggestions_check["reason"],
            "actionable_count": task_suggestions_check.get("actionable_count", 0),
            "importance_breakdown": task_suggestions_check.get("importance_breakdown", {})
        }
        print(f"[TASK_SUGGESTIONS_ADVISORY]")
        print(json.dumps(advisory_info, indent=2))
        # NOTE: Does NOT exit — this is advisory only, implementation proceeds

    # Check if generate-tasks phase has been completed
    if not state.is_phase_completed("generate-tasks"):
        print("[TASKS_MISSING]")
        print(json.dumps({"marker": "TASKS_MISSING", "reason": "No implementation tasks found"}))
        sys.exit(0)

    # Check for external tasks file reference
    tasks_file_ref = get_tasks_file_path(plan_content)
    tasks_content = plan_content  # Default: parse from plan itself

    if tasks_file_ref:
        # Load tasks from external file
        tasks_file_path = plan_path.parent / tasks_file_ref
        if tasks_file_path.exists():
            tasks_content = tasks_file_path.read_text(encoding="utf-8")
            print(f"Loading tasks from: {tasks_file_path}")
        else:
            print(f"WARNING: Tasks file not found: {tasks_file_path}")
            print("Falling back to parsing plan content directly.")

    # Record pre-existing changes (before implementation modifies anything)
    record_pre_existing_changes(state)

    # Record HEAD before implementation so code-review diffs exclude pre-impl commits
    if state.get("head_before_implement") is None:
        try:
            state.set("head_before_implement", get_current_head())
            state.save()
        except Exception as e:
            print(f"Warning: Could not record head_before_implement: {e}")

    # Check for plan changes
    if state.has_plan_changed() and args.resume:
        print("WARNING: Plan has changed since last session")
        state.clear_plan_changed_flag()

    # Decompose tasks (from tasks file or plan content)
    decomposer = TaskDecomposer()
    tasks = decomposer.parse_plan(tasks_content)
    print(f"Found {len(tasks)} tasks")

    # Restore previous statuses if resuming
    if args.resume:
        for task_id, status in state.get_all_task_statuses().items():
            if task_id in decomposer.tasks:
                decomposer.update_task_status(task_id, TaskStatus(status))
        print("Restored previous session state")

    # If specific task requested
    if args.task:
        task = decomposer.get_task(args.task)
        if not task:
            print(f"ERROR: Task not found: {args.task}")
            sys.exit(1)

        # Compute related file paths (in phase-based subdirectories)
        tasks_file_path = get_output_paths(plan_path, "tasks", phase='tasks')
        summary_file_path = get_output_paths(plan_path, "implementation_summary", phase='implement')
        tasks_file_relative = get_relative_output_path(plan_path, "tasks", phase='tasks')
        summary_file_relative = get_relative_output_path(plan_path, "implementation_summary", phase='implement')

        output_data = {
            "plan_file": str(plan_path.absolute()),
            "tasks_file": str(tasks_file_path),
            "summary_file": str(summary_file_path),
            "tasks_file_relative": tasks_file_relative,
            "summary_file_relative": summary_file_relative,
            "state_file": str(state.state_file),
            "generated_at": datetime.now().isoformat(),
            "total_tasks": 1,
            "batches": [{
                "batch_index": 0,
                "can_parallelize": False,
                "tasks": [format_task_for_output(task, plan_preamble, plan_path_str, all_tasks=decomposer.tasks)]
            }]
        }
    else:
        # Get parallel batches
        batches = decomposer.get_parallel_batches()
        print(f"Organized into {len(batches)} execution batches")

        # Format for output
        output_batches = format_batches_for_output(batches, skip_completed=args.resume, plan_preamble=plan_preamble, plan_path=plan_path_str, all_tasks=decomposer.tasks)

        # Count pending tasks
        pending_count = sum(len(b["tasks"]) for b in output_batches)

        # Compute related file paths (in phase-based subdirectories)
        tasks_file_path = get_output_paths(plan_path, "tasks", phase='tasks')
        summary_file_path = get_output_paths(plan_path, "implementation_summary", phase='implement')
        tasks_file_relative = get_relative_output_path(plan_path, "tasks", phase='tasks')
        summary_file_relative = get_relative_output_path(plan_path, "implementation_summary", phase='implement')

        output_data = {
            "plan_file": str(plan_path.absolute()),
            "tasks_file": str(tasks_file_path),
            "summary_file": str(summary_file_path),
            "tasks_file_relative": tasks_file_relative,
            "summary_file_relative": summary_file_relative,
            "state_file": str(state.state_file),
            "generated_at": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "pending_tasks": pending_count,
            "batches": output_batches
        }

    # Dry run - just print summary
    if args.dry_run:
        print("\n" + "=" * 60)
        print("Task Summary (Dry Run)")
        print("=" * 60)
        summary = decomposer.get_progress_summary()
        print(f"Total tasks: {summary['total']}")
        print(f"Completed: {summary['by_status']['completed']}")
        print(f"Pending: {summary['by_status']['pending']}")
        print(f"Failed: {summary['by_status']['failed']}")
        print(f"\nBatches: {len(output_data['batches'])}")

        for batch in output_data["batches"]:
            print(f"\n  Batch {batch['batch_index']}:")
            for task in batch["tasks"]:
                print(f"    - {task['id']}: {task['title']} ({task['subagent_type']})")

        print(f"\nWould write to: {args.output}")
        sys.exit(0)

    # Write output JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_temp_dir_gitignore(output_path.parent)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    # Print summary
    print("\n" + "=" * 60)
    print("Tasks Prepared for Claude Code")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"State file: {state.state_file}")
    print(f"Total tasks: {output_data['total_tasks']}")
    print(f"Pending tasks: {output_data.get('pending_tasks', output_data['total_tasks'])}")
    print(f"Batches: {len(output_data['batches'])}")

    print("\nClaude Code should now:")
    print("  1. Read the output JSON file")
    print("  2. Execute each batch sequentially (batches have dependencies)")
    print("  3. Tasks within a batch can run in parallel")
    print("  4. Update the state file after each task completes")
    print("  5. After all tasks complete, generate the implementation summary")
    print(f"     Summary file: {output_data['summary_file']}")

    state.mark_phase_completed("implement")
    state.save()

    sys.exit(0)


if __name__ == "__main__":
    main()
