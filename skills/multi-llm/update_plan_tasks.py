#!/usr/bin/env python3
"""Update plan file with generated tasks.

Usage:
    uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util update-tasks --plan-file <plan_path> --tasks-file <tasks_json_path>

This script:
1. Reads the tasks JSON file and validates it as a task-generation result
2. Creates the tasks markdown file in {plan}/tasks/tasks.md
3. Updates the plan file with a reference to the tasks file

``apply_tasks_document`` does steps 2-3 for a parsed document; generate-tasks
calls it in-process, and ``main`` is the CLI wrapper around it.
"""

import argparse
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.stream_bootstrap import bootstrap_streams
from utils.output_handler import get_output_paths, get_relative_output_path
from utils.plan_updater import insert_tasks_file_reference
from utils.state_manager import get_or_create_state
from utils.subagent_tasks import derive_executor_role


COMPLEXITY_COLORS = {
    "low": "#90ee90",
    "medium": "#f4a460",
    "high": "#cd5c5c",
}


#: The one format the tasks file's ``*Generated: …*`` line is rendered in.
GENERATED_STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def generated_stamp() -> str:
    """The instant the tasks file says it was generated, as rendered text.

    Local wall clock, exactly as it always was — unless ``SOURCE_DATE_EPOCH``
    is set, the reproducible-builds convention: then that instant, in UTC, so
    two runs pinned to the same epoch write byte-identical tasks files. The
    container rig pins it; without that the stamp made every digest of
    ``tasks.md`` a fresh number per run, and a golden baseline could not repeat.

    A malformed value raises rather than falling back: the variable is only ever
    set by something asking for determinism, and a silent wall-clock fallback
    would hand it a non-repeatable file with nothing to say why. Empty counts as
    unset, the way an env file spells "not configured".
    """
    raw = os.environ.get("SOURCE_DATE_EPOCH", "")
    if not raw:
        return datetime.now().strftime(GENERATED_STAMP_FORMAT)
    try:
        if not re.fullmatch(r"[0-9]+", raw):
            raise ValueError("not a non-negative integer")
        instant = datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except (ValueError, OverflowError, OSError) as exc:  # out of range, too
        raise ValueError(
            f"SOURCE_DATE_EPOCH must be a non-negative integer count of seconds "
            f"since the Unix epoch, got {raw!r} ({exc})") from exc
    return instant.strftime(GENERATED_STAMP_FORMAT)


def task_executor_role(task: Dict[str, Any]) -> str:
    """Return a task record's executor role (D2h).

    ``executor_role`` is authoritative; a record written before the rename is
    read through its legacy ``subagent_type``, so no reader branches on the
    Claude-Code-only field any more.
    """
    role = task.get("executor_role")
    if role in ("agent", "human"):
        return role
    return derive_executor_role(task.get("subagent_type"))


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
        if task_executor_role(task) == "human":
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
        if task_executor_role(task) == "human":
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
        f"*Generated: {generated_stamp()}*",
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
        executor_role = task_executor_role(task)

        # Task header
        lines.append(f"### Task {task_id}: {title}")

        # Dependencies
        if depends_on:
            deps_str = ", ".join(depends_on)
            lines.append(f"**Dependencies**: {deps_str}")
        else:
            lines.append("**Dependencies**: None")

        # Files. BOTH lines are always rendered, `none` when the task declares
        # no files: `task_decomposer._explicit_files` reads "neither line
        # present" as "hand-written task, guess the files from the narrative",
        # so omitting them for an empty list handed the keyword heuristic the
        # one kind of task it must never see — a generated one. The parser
        # already discards `none`/`n/a`/`-`, so an empty declaration round-trips
        # as empty instead of as absent.
        modify_text = ', '.join(files_to_modify) if files_to_modify else "none"
        create_text = ', '.join(files_to_create) if files_to_create else "none"
        lines.append(f"**Files to modify**: {modify_text}")
        lines.append(f"**Files to create**: {create_text}")

        # Complexity and executor. tasks.md is read by every host and by humans,
        # so it names the host-neutral executor role, never a Claude Code agent
        # type. The human branch's wording is unchanged — it was already
        # host-neutral and users recognize it.
        lines.append(f"**Complexity**: {complexity}")
        if executor_role == "human":
            lines.append("**Type**: Manual (user action required)")
        else:
            lines.append(f"**Executor**: {executor_role}")
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


#: A task list the plan already carries: a heading whose text is "Tasks" or
#: "<anything> tasks" (``### Tasks``, ``## Implementation tasks``, ``### Implementation
#: tasks with dependencies``). The pointer section this script itself writes is
#: told apart by the TASKS_FILE marker inside it.
_IN_PLAN_TASK_HEADING = re.compile(
    r"^#{2,4}[ \t]+(?:[\w-]+[ \t]+)?tasks\b[^\n]*$", re.IGNORECASE | re.MULTILINE
)


def find_in_plan_task_sections(plan_content: str) -> List[str]:
    """Headings of task lists the plan holds besides the generated pointer.

    The 2026-10-01 runs ended with two diverging task lists in one plan
    (`### Tasks`, `## Implementation tasks`, ... from apply-suggestions, then this
    script's pointer). Merging them is not this script's job, so it only names
    them: the caller prints a notice and the user decides which list stands.
    """
    found: List[str] = []
    for match in _IN_PLAN_TASK_HEADING.finditer(plan_content):
        rest = plan_content[match.end():]
        nxt = re.search(r"^#{1,4}[ \t]", rest, re.MULTILINE)
        body = rest[: nxt.start()] if nxt else rest
        if "TASKS_FILE" in body or "TASKS_FILE" in match.group(0):
            continue
        found.append(match.group(0).strip())
    return found


class TasksDocumentError(Exception):
    """A task-generation document was not applied.

    Raised before the first write for a missing plan, an invalid document or a
    malformed ``SOURCE_DATE_EPOCH``; an ``OSError`` from a write propagates as
    itself. Callers that must leave the plan untouched snapshot it first
    (``generate_tasks_orchestrator._Snapshot``).
    """


class TasksDocumentInvalid(TasksDocumentError):
    """The document fails ``validate_result("task_generation", ...)``: the
    schema, then ``task_decomposer.validate_generated_tasks``' graph rules."""

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__(
            "the document is not a valid task-generation result:\n"
            + "\n".join(f"  - {problem}" for problem in self.errors[:10])
        )


@dataclass(frozen=True)
class AppliedTasks:
    """What :func:`apply_tasks_document` wrote, or would write on a dry run."""

    tasks_path: Path
    tasks_relative_path: str
    task_count: int
    dry_run: bool


def replace_file(path: Path, data: Union[str, bytes]) -> None:
    """Write *data* to *path* through a sibling temp file and ``os.replace``.

    A crash or a failed write leaves the old bytes in place, never a truncated
    file. Text is written the way ``Path.write_text(encoding="utf-8")`` writes
    it (newline translation included); bytes land exactly. A symlink is
    written through to its target, so the link survives, and an existing
    file's permission bits carry over to the replacement; a new file gets the
    process umask, as ``write_text`` would.

    The temp file replacing an existing file is created owner-only and given
    the old file's mode before its bytes are flushed, so it is never readable
    more widely than the file it replaces. On POSIX the directory is flushed
    after the rename, so a returned call survives a power loss.
    """
    target = Path(os.path.realpath(path))
    try:
        mode: Optional[int] = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        mode = None
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    # O_BINARY: the CRT must not translate newlines a second time on Windows;
    # the text stream below does it, exactly as ``write_text`` would.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    handle = os.open(str(temporary), flags, 0o666 if mode is None else 0o600)
    try:
        if isinstance(data, bytes):
            stream = os.fdopen(handle, "wb")
        else:
            stream = os.fdopen(handle, "w", encoding="utf-8")
        with stream:
            stream.write(data)
            stream.flush()
            if mode is not None:
                if hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), mode)
                else:  # Windows before 3.13: only the read-only bit exists
                    os.chmod(temporary, mode)
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    _flush_directory(target.parent)


def _flush_directory(folder: Path) -> None:
    """Flush *folder*'s entries on POSIX; Windows can't open a directory for this."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(folder), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def apply_tasks_document(
    plan_path: Path, data: Any, *, dry_run: bool = False
) -> AppliedTasks:
    """Render *data*'s tasks into ``tasks.md``, point the plan at it, record the phase.

    *data* is a parsed task-generation document. It is read the way every lane
    reads one — an omitted empty list, ``subagent_type`` or ``schema_version``
    is supplied in place (``subagent_executor.normalize_result``,
    ``supply_result_version``) — then validated with the executor's own verdict
    for the kind, so nothing reaches the plan that the executor would have
    refused. Raises :class:`TasksDocumentError` (or an ``OSError`` from a
    write) on failure; prints what it did.
    """
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise TasksDocumentError(f"Plan file not found: {plan_path}")

    from utils.subagent_executor import (
        normalize_result,
        supply_result_version,
        validate_result,
    )

    normalize_result("task_generation", data)
    supply_result_version("task_generation", data)
    errors = validate_result("task_generation", data)
    if errors:
        raise TasksDocumentInvalid(errors)

    tasks = data["tasks"]
    preamble = data.get("plan_preamble", "")

    try:
        tasks_markdown = generate_tasks_markdown(tasks, plan_path.name)
    except ValueError as e:  # a malformed SOURCE_DATE_EPOCH; nothing written yet
        raise TasksDocumentError(str(e)) from e

    tasks_output_path = get_output_paths(plan_path, "tasks", phase="tasks")
    tasks_relative_path = get_relative_output_path(plan_path, "tasks", phase="tasks")
    applied = AppliedTasks(
        tasks_path=tasks_output_path,
        tasks_relative_path=tasks_relative_path,
        task_count=len(tasks),
        dry_run=dry_run,
    )

    plan_content = plan_path.read_text(encoding="utf-8")
    updated_plan = insert_tasks_file_reference(plan_content, tasks_relative_path)

    if dry_run:
        print("Dry run - would perform these actions:")
        print(f"\n1. Create tasks file: {tasks_output_path}")
        print(f"   ({len(tasks)} tasks)")
        print(f"\n2. Update plan file: {plan_path}")
        print(f"   Add reference: <!-- TASKS_FILE: {tasks_relative_path} -->")
        return applied

    tasks_output_path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(tasks_output_path, tasks_markdown)
    print(f"Created tasks file: {tasks_output_path}")

    replace_file(plan_path, updated_plan)
    # Said what was written: a bare "Updated plan file" read as generate-tasks
    # rewriting the plan (OpenCode racing run 4). The section and its
    # TASKS_FILE marker are how every later phase finds the task list.
    print(
        f"Updated plan file: {plan_path} (its \"Implementation Tasks\" section "
        f"now links {tasks_relative_path}, the task list later phases read)"
    )
    existing = find_in_plan_task_sections(plan_content)
    if existing:
        print(
            "NOTICE: the plan already holds its own task list under "
            + ", ".join(f"'{heading}'" for heading in existing)
            + f". {tasks_relative_path} is the list implement follows; the plan's "
            "list is not updated to match it and may disagree. Delete it, or keep "
            "it as background only."
        )

    # Mark generate-tasks phase as completed in state
    state = get_or_create_state(plan_path)
    state.mark_phase_completed("generate-tasks")
    state.save()

    # Store plan preamble for use during implementation
    if preamble:
        state.set("plan_preamble", preamble)
        state.save()
        print(f"Stored plan preamble ({len(preamble)} chars)")

    # Re-stamp the hash over the plan *this function just rewrote*. The
    # StateManager above is constructed after the write, so `_load_or_create`
    # sees the marker insertion as drift and sets `plan_changed: True` — the
    # plugin flagging its own edit as a human's. Measured 2026-09-16: the next
    # `implement --resume` then printed "WARNING: Plan has changed since last
    # session" for a change generate-tasks had made one phase earlier.
    state.refresh_plan_hash()
    state.save()

    # Summary
    print(f"\nSuccessfully generated {len(tasks)} tasks:")
    for task in tasks:
        deps = task.get("depends_on", [])
        deps_str = f" (depends on: {', '.join(deps)})" if deps else ""
        print(f"  - {task['id']}: {task['title']}{deps_str}")

    return applied


def main(argv=None) -> int:
    """``util update-tasks``: read a tasks JSON file and apply it to a plan."""
    bootstrap_streams()
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
    args = parser.parse_args(argv)

    plan_path = Path(args.plan_file).resolve()
    tasks_json_path = Path(args.tasks_file).resolve()

    if not plan_path.exists():
        print(f"ERROR: Plan file not found: {plan_path}", file=sys.stderr)
        return 1

    if not tasks_json_path.exists():
        print(f"ERROR: Tasks JSON file not found: {tasks_json_path}", file=sys.stderr)
        return 1

    try:
        tasks_data = json.loads(tasks_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse tasks JSON: {e}", file=sys.stderr)
        return 1

    try:
        apply_tasks_document(plan_path, tasks_data, dry_run=args.dry_run)
    except TasksDocumentInvalid as invalid:
        print(f"ERROR: Invalid tasks JSON: {invalid}", file=sys.stderr)
        return 1
    except TasksDocumentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
