"""Utility for locating and validating task files associated with a plan.

Provides find_tasks_file() which handles TASKS_FILE comment parsing, fallback
path resolution, security checks, staleness warnings, and file validation.

Extracted from review_tasks_orchestrator.py for reuse across orchestrators
that need to locate the tasks file (review-tasks, apply-task-suggestions, etc.).
"""

import os
import re
import sys

from .output_handler import sanitize_prefix


def find_tasks_file(plan_path: str) -> str:
    """Locate the tasks file for the given plan.

    Discovery logic:
    1. Read the plan file and check for a <!-- TASKS_FILE: path/to/tasks.md --> comment.
       If found, extract and resolve the path relative to the plan directory.
    2. If no comment found, fall back to {plan_dir}/tasks/tasks.md (using the
       sanitized plan name directory structure).
    3. If neither location exists, raise FileNotFoundError.
    4. After resolving, validate the file is parseable (non-empty, has task headers).

    Path safety: Resolved path must stay within the plan workspace directory.

    Error handling for edge cases:
    - TASKS_FILE comment points to nonexistent file: specific error message
    - Both TASKS_FILE comment and default location exist but differ: prefer comment, log warning
    - Stale tasks file (older than plan): log warning but proceed

    Args:
        plan_path: Absolute path to the plan file

    Returns:
        Absolute path to the tasks file

    Raises:
        FileNotFoundError: If no tasks file can be found
        ValueError: If the tasks file is malformed
    """
    plan_path_abs = os.path.abspath(plan_path)
    plan_dir = os.path.dirname(plan_path_abs)

    # Determine the workspace root (the plan's output directory tree root)
    # The workspace is the plan file's parent directory
    workspace_root = os.path.realpath(plan_dir)

    # Read the plan file to look for TASKS_FILE comment
    with open(plan_path_abs, 'r', encoding='utf-8') as f:
        plan_content = f.read()

    # 1. Check for <!-- TASKS_FILE: path/to/tasks.md --> comment
    tasks_file_match = re.search(
        r'<!--\s*TASKS_FILE:\s*(.+?)\s*-->', plan_content
    )

    comment_path = None
    if tasks_file_match:
        raw_path = tasks_file_match.group(1).strip()

        # Security: reject absolute paths and .. traversal in the raw value
        if os.path.isabs(raw_path):
            print(
                f"ERROR: TASKS_FILE comment contains an absolute path: {raw_path}. "
                f"Only relative paths within the plan directory are allowed.",
                file=sys.stderr,
            )
            sys.exit(1)

        if '..' in raw_path.split(os.sep) or '..' in raw_path.split('/'):
            print(
                f"ERROR: TASKS_FILE comment contains path traversal (..): {raw_path}. "
                f"Only paths within the plan directory are allowed.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Resolve relative to plan directory
        comment_path = os.path.realpath(os.path.join(plan_dir, raw_path))

        # Verify it stays within workspace
        if not comment_path.startswith(workspace_root + os.sep) and comment_path != workspace_root:
            print(
                f"ERROR: TASKS_FILE path resolves outside the plan directory: {comment_path}. "
                f"Must be within: {workspace_root}",
                file=sys.stderr,
            )
            sys.exit(1)

    # 2. Determine default tasks file location
    # Uses the plan's output directory structure: {plan_dir}/{sanitized_name}/tasks/tasks.md
    prefix = sanitize_prefix(os.path.basename(plan_path_abs))
    default_tasks_dir = os.path.join(plan_dir, prefix, "tasks")
    default_path = os.path.realpath(os.path.join(default_tasks_dir, "tasks.md"))

    # 3. Resolve which file to use
    tasks_path = None

    if comment_path:
        # TASKS_FILE comment found
        if not os.path.isfile(comment_path):
            raise FileNotFoundError(
                f"Tasks file referenced in plan not found: {comment_path}. "
                f"Re-run '--generate-tasks' or correct the TASKS_FILE comment."
            )

        # Check if default also exists and differs
        if (os.path.isfile(default_path)
                and os.path.realpath(comment_path) != os.path.realpath(default_path)):
            print(
                f"Note: Using tasks file from TASKS_FILE comment ({comment_path}), "
                f"ignoring default tasks/tasks.md.",
                file=sys.stderr,
            )

        tasks_path = comment_path
    elif os.path.isfile(default_path):
        tasks_path = default_path
    else:
        raise FileNotFoundError(
            "No tasks file found. Run '--generate-tasks' first, then retry."
        )

    # 4. Check for staleness (tasks file older than plan)
    try:
        tasks_mtime = os.path.getmtime(tasks_path)
        plan_mtime = os.path.getmtime(plan_path_abs)
        if tasks_mtime < plan_mtime:
            print(
                "Warning: Tasks file is older than the plan file. "
                "Consider re-running '--generate-tasks' to pick up plan changes.",
                file=sys.stderr,
            )
    except OSError:
        pass  # If we can't stat, skip staleness check

    # 5. Validate the file is parseable
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, OSError) as e:
        raise ValueError(f"Cannot read tasks file {tasks_path}: {e}")

    if not content.strip():
        raise ValueError(
            f"Tasks file is empty: {tasks_path}. "
            f"Re-run '--generate-tasks' to generate tasks."
        )

    # Check for at least one task header (e.g., ## T001, ### Task T001:)
    task_header_pattern = re.compile(r'^#{2,3}\s+(?:Task\s+)?T\d+', re.MULTILINE)
    if not task_header_pattern.search(content):
        raise ValueError(
            f"Tasks file appears malformed (no task headers like '## T001' found): "
            f"{tasks_path}. Re-run '--generate-tasks' to regenerate."
        )

    return tasks_path
