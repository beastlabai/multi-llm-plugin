#!/usr/bin/env python3
"""
Implementation orchestrator for preparing tasks for Claude Code execution.

This orchestrator takes a plan file, decomposes it into tasks, organizes them
into dependency-respecting batches, and outputs JSON for Claude Code to execute
using its native Task tool.

Usage:
    uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" implement plans/my-plan.md [--output tasks.json]
"""

import argparse
import hashlib
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
from utils.cli_common import EXIT_NEEDS_HUMAN, build_parent_parser
from utils.output_handler import get_phase_dir, get_output_dir
from utils.git_utils import get_project_root
from utils import unattended as unattended_mod
from utils.unattended import (
    HUMAN_TASK_STRATEGIES,
    UNATTENDED_HUMAN_TASK_STRATEGY,
    NeedsHuman,
    parse_human_task_decisions,
)
from check_workflow_prerequisites import (
    SKIPPED_MARKER,
    SKIP_PREREQUISITES_REASON,
    build_unmet_message,
    check_apply_task_suggestions_prerequisite,
    record_prerequisite_skip,
    upstream_skip_notices,
)

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


def resolve_human_task_strategy(
    state,
    human_task_ids: List[str],
    *,
    force_strategy: Optional[str],
    unattended: bool,
    resume: bool,
) -> Optional[str]:
    """Resolve D6 row 8 — how tasks needing a human are handled.

    Resolution order, unchanged from ``instructions/implement.md`` except that the
    fourth rung is no longer a prompt:

    1. ``--force-strategy`` — an explicit flag always wins.
    2. ``--resume`` with a saved ``human_task_strategy`` in ``state.json``.
    3. ``--yes`` → :data:`UNATTENDED_HUMAN_TASK_STRATEGY` (``skip-dependents``).
       An unattended run cannot perform manual steps, so code tasks depending on
       them cannot be assumed valid — ``skip-continue`` would run them anyway.
    4. Neither → ``None``: the driving agent renders the gate (attended only).

    Whatever is resolved is persisted exactly as ``--force-strategy`` does, so
    ``--resume`` stays stable and never re-asks.
    """
    if not human_task_ids:
        return None
    strategy: Optional[str] = None
    if force_strategy:
        strategy = force_strategy
    elif resume and state.get("human_task_strategy"):
        strategy = state.get("human_task_strategy")
    elif unattended:
        strategy = UNATTENDED_HUMAN_TASK_STRATEGY
    if strategy is None:
        return None
    if state.get("human_task_strategy") != strategy:
        state.set("human_task_strategy", strategy)
        state.save()
    return strategy


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare implementation tasks from a plan file for Claude Code execution",
        parents=[build_parent_parser()],
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
             '<git-root>/.multi-llm/tmp/'
             'implementation_tasks_{plan_stem}_{hash8}.json, where {hash8} '
             'is a stable hash of the repo-relative plan path; resolved '
             'from the plan file after argument parsing)'
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

    # --- D5a decision flags (the non-interactive answer to D6 rows 7, 8, 9) ---
    # Documented in instructions/implement.md long before this parser accepted
    # them: the strategy prompt and the per-task pause were markdown-only, which
    # meant the *only* way to answer them was to be a human at a terminal. Every
    # gate now has a flag whose values enumerate exactly the options the question
    # rendered — `choices` validates them, and an empty/unknown value is exit 3.
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip the task-suggestion advisory prompt (D6 row 7): record '
             'apply-task-suggestions as skipped and proceed. Implied by --yes.'
    )
    parser.add_argument(
        '--force-strategy',
        choices=list(HUMAN_TASK_STRATEGIES),
        default=None,
        help='Answer the human-task strategy gate (D6 row 8) without a prompt '
             'and persist it to state.json, so --resume stays stable. Under '
             '--yes the default is skip-dependents; pause-and-ask together with '
             '--yes is exit 3.'
    )
    parser.add_argument(
        '--human-task',
        action='append',
        default=None,
        metavar='ID=done|skip',
        help='Answer one human task (D6 row 9), repeatable: --human-task '
             'T004=done. Turns the pause-and-ask pause into a process boundary '
             'instead of a blocked stdin.'
    )

    return parser.parse_args(argv)


def _plan_path_hash(plan_path: Path, project_root: str) -> str:
    """Stable 8-hex-char hash of the repo-relative plan path.

    Hashes the POSIX form of the plan path relative to ``project_root`` so
    the value is identical on every OS regardless of separator style. Falls
    back to the absolute POSIX path when the plan does not sit under the
    detected root (e.g. symlinked layouts) — still deterministic per plan
    location, which is all the filename needs.
    """
    try:
        key = plan_path.resolve().relative_to(
            Path(project_root).resolve()
        ).as_posix()
    except ValueError:
        key = plan_path.resolve().as_posix()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def resolve_default_output(plan_path: Path) -> str:
    """Resolve the default --output path anchored at the git project root.

    The default lives inside the repo workspace
    (``<git-root>/.multi-llm/tmp/implementation_tasks_{plan_stem}_{hash8}.json``)
    so the Bash side and harness Read/Write tools resolve the same file on
    every OS. ``{hash8}`` is :func:`_plan_path_hash` of the repo-relative
    plan path: the stem alone is not unique (``plans/api/index.md`` and
    ``plans/ui/index.md`` share ``index``), and without the hash concurrent
    runs on same-stem plans would clobber each other's task JSON. The
    orchestrator is the single source of truth for this name — it prints the
    resolved path on its ``Output file:`` summary line, and
    ``instructions/implement.md`` directs Claude to read that printed path
    rather than reconstructing it (a Bash-side reimplementation of the hash
    would be a byte-identical dual-computation hazard). Never fall back to
    the system temp dir (``tempfile.gettempdir()``): on Git for Windows the
    MSYS ``/tmp`` and the native temp dir are different directories, which is
    exactly the boundary hazard this default avoids.

    Args:
        plan_path: Resolved path to the plan file (used for git-root
            detection, the ``{plan_stem}`` filename part, and the
            ``{hash8}`` disambiguator).

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
    file_hash = _plan_path_hash(plan_path, project_root)
    return str(
        Path(project_root) / ".multi-llm" / "tmp"
        / f"implementation_tasks_{plan_path.stem}_{file_hash}.json"
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
        # D2h: `executor_role` is what every reader branches on; `subagent_type`
        # rides along as optional Claude-only metadata for hosts whose native
        # Task tool requires the parameter.
        "executor_role": task.executor_role,
        "subagent_type": task.subagent_type,
        "is_human": task.executor_role == "human",
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


def run_implement_write_phase(
    args: argparse.Namespace,
    *,
    plan_path: Path,
    state: StateManager,
    output_data: Dict[str, Any],
    human_task_strategy: Optional[str],
    human_task_decisions: Dict[str, str],
) -> Optional[int]:
    """Schedule the implement batches through the write executor, or decline (D3n).

    Returns ``None`` when neither ``--exec-subagents`` nor
    ``--dry-run-subagents`` was passed, so the caller falls through to today's
    "emit task JSON, let the driving agent run it" path unchanged. That is the
    additivity guarantee: with the flag off, nothing in this function runs.

    When it does run it owns the whole phase: ``batch_index`` ordering with
    verification gating the next dispatch, the resolved human-task strategy, the
    per-batch snapshot/classify/append/record cycle, and finalization
    (``mark_phase_completed`` + ``finalize_tracking`` + the implementation summary
    + the plan-side reference). Its exit code is the phase's.
    """
    if not (getattr(args, "exec_subagents", False)
            or getattr(args, "dry_run_subagents", False)):
        return None

    from utils.write_batch_scheduler import WriteBatchScheduler
    from utils.review_orchestrator_base import maybe_detach
    from utils.write_phase_bridge import (
        materialize_write_tasks,
        write_task_file_for_phase,
    )

    phase_dir = str(get_phase_dir(plan_path, "implement"))
    Path(phase_dir).mkdir(parents=True, exist_ok=True)
    plan_file = str(plan_path.absolute())

    # The detach boundary (D4): after every pre-flight — plan loaded, prerequisites
    # gated, tasks decomposed, human-task strategy resolved, phase dir created — and
    # before any child is dispatched, so a bad flag fails on the caller's terminal
    # with no child and no log.
    if getattr(args, "detach", False) or getattr(args, "internal_worker", False):
        detached = maybe_detach(
            args=args, mode="implement", phase_dir=phase_dir,
            plan_file=plan_file, phase="implement",
        )
        if detached is not None:
            return detached

    tasks = materialize_write_tasks(
        kind="implement",
        batches=output_data.get("batches") or [],
        phase_dir=phase_dir,
        plan_file=plan_file,
    )
    task_file = write_task_file_for_phase(
        kind="implement", tasks=tasks, phase_dir=phase_dir, plan_file=plan_file,
    )
    print(f"[write-subagents] materialized {len(tasks)} implement task(s): {task_file}")

    scheduler = WriteBatchScheduler(
        tasks=tasks,
        phase_dir=phase_dir,
        kind="implement",
        output_dir=str(get_output_dir(plan_path)),
        state=state,
        plan_file=plan_file,
        summary_file=output_data.get("summary_file"),
        summary_file_relative=output_data.get("summary_file_relative"),
        anchor=plan_file,
        resume=bool(getattr(args, "resume", False)),
        dry_run=bool(getattr(args, "dry_run_subagents", False)),
        human_task_strategy=human_task_strategy,
        human_task_decisions=human_task_decisions,
    )
    result = scheduler.run()
    for line in result.messages:
        print(line, file=sys.stderr if result.exit_code else sys.stdout)
    if result.report_path:
        print(f"[write-subagents] report: {result.report_path}")
    if result.summary_file:
        print(f"[write-subagents] implementation summary: {result.summary_file}")
    return result.exit_code


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    bootstrap_streams()
    args = parse_args(argv)

    # D6b: unattended is a *flag*, never a runtime guess — nothing here sniffs a
    # TTY, `CI` or a `--print` flag to decide it.
    unattended = bool(getattr(args, "yes", False))

    # D6 rows 9/15: validate the per-task answers before anything is read, so an
    # empty or unrecognized value is exit 3 at parse time rather than a silent
    # default halfway through a batch.
    try:
        human_task_decisions = parse_human_task_decisions(args.human_task)
    except NeedsHuman as exc:
        return unattended_mod.fail_closed(exc.message, site_id=exc.site)
    if unattended and args.force_strategy == "pause-and-ask":
        # The dispatcher already rejects this combination at parse time; this is
        # the same refusal for a direct invocation, so neither route can hang.
        return unattended_mod.fail_closed(
            "ERROR: --force-strategy pause-and-ask requires a human at the "
            "per-task gate, which --yes forbids. Pass a non-interactive strategy "
            f"(e.g. {UNATTENDED_HUMAN_TASK_STRATEGY}) or drop --yes.",
            site_id=unattended_mod.row(9).site_id,
        )

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

    # A recorded upstream `--skip-prerequisites` is reported once and never
    # re-litigated: consent was already given explicitly, and re-asking would make
    # the recorded skip meaningless (D6a).
    for notice in upstream_skip_notices(state):
        print(notice)

    # Check for unapplied suggestions (D6 row 6). Three endings, no fourth:
    #   --skip-prerequisites  -> loud, recorded, proceed
    #   --yes                 -> exit 3, one message naming both fixes
    #   neither               -> today's [PREREQUISITE_CHECK] payload, exit 0
    # Nothing has been produced at this point: no phase dir, no tasks, no
    # state.json mutation, so an unmet gate leaves no half-generated artifact.
    apply_check = check_apply_suggestions_prerequisite(state, plan_dir)
    if not apply_check["met"]:
        missing = [{
            "phase": "apply-suggestions",
            "reason": apply_check["reason"],
            "valid_count": apply_check.get("valid_count", 0),
            "importance_breakdown": apply_check.get("importance_breakdown", {}),
        }]
        if args.skip_prerequisites:
            record_prerequisite_skip(
                state, "apply-suggestions", reason=SKIP_PREREQUISITES_REASON
            )
            print(
                f"{SKIPPED_MARKER} implement: "
                f"{apply_check.get('valid_count', 0)} unapplied apply-suggestions "
                f"suggestions (reason: {SKIP_PREREQUISITES_REASON})"
            )
        elif unattended:
            print(build_unmet_message("implement", plan_path, plan_dir, missing))
            unattended_mod.mark_needs_human("implement")
            print(json.dumps({
                "marker": "PREREQUISITE_UNMET",
                "mode": "implement",
                "state": "needs-human",
                "exit": EXIT_NEEDS_HUMAN,
                "missing": missing,
            }, indent=2))
            sys.exit(EXIT_NEEDS_HUMAN)
        else:
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
        # D6 row 7: under --yes / --no-confirm the advisory resolves to "skip",
        # and the phase is *recorded* skipped here rather than left for the agent
        # to remember — an auto-selection nobody can see afterwards is
        # indistinguishable from a silent degradation.
        if unattended or args.no_confirm:
            record_prerequisite_skip(
                state,
                "apply-task-suggestions",
                reason="auto-skipped by --yes/--no-confirm (D6 row 7)",
            )
            unattended_mod.record_resolution(
                str(plan_dir / "implement"),
                unattended_mod.row(7).site_id,
                "skipped; apply-task-suggestions recorded as skipped",
            )

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

    # --- D6 rows 8 and 9: resolve and record, never prompt ------------------
    human_task_ids = [
        task.id for task in decomposer.tasks.values()
        if getattr(task, "executor_role", "") == "human"
    ]
    human_task_strategy = resolve_human_task_strategy(
        state,
        human_task_ids,
        force_strategy=args.force_strategy,
        unattended=unattended,
        resume=args.resume,
    )
    if human_task_ids and human_task_strategy and unattended and not args.force_strategy:
        unattended_mod.record_resolution(
            str(get_phase_dir(plan_path, "implement")),
            unattended_mod.row(8).site_id,
            f"{human_task_strategy} for {len(human_task_ids)} human task(s)",
        )
    if human_task_decisions:
        # Row 9's answer channel: a per-task decision arrives as a flag, is
        # recorded in state.json, and is handed to the driving agent in the task
        # JSON so the pause is a process boundary rather than a blocked stdin.
        unknown = [tid for tid in human_task_decisions if tid not in decomposer.tasks]
        if unknown:
            return unattended_mod.fail_closed(
                "ERROR: --human-task names task(s) that are not in this plan: "
                f"{', '.join(sorted(unknown))}.\n"
                f"       Human tasks in this plan: "
                f"{', '.join(human_task_ids) or 'none'}",
                site_id=unattended_mod.row(9).site_id,
            )
        not_human = [tid for tid in human_task_decisions if tid not in human_task_ids]
        if not_human:
            return unattended_mod.fail_closed(
                "ERROR: --human-task only answers tasks with executor_role "
                f"'human'; these are code tasks: {', '.join(sorted(not_human))}.",
                site_id=unattended_mod.row(9).site_id,
            )
        recorded = dict(state.get("human_task_decisions") or {})
        recorded.update(human_task_decisions)
        state.set("human_task_decisions", recorded)
        state.save()
        for task_id, action in sorted(human_task_decisions.items()):
            print(f"[HUMAN_TASK_ANSWERED] {task_id}={action}")

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

    # The resolved answers travel with the tasks, so the invoker copies a
    # decision instead of composing one (D5a).
    if human_task_ids:
        output_data["human_tasks"] = list(human_task_ids)
        output_data["human_task_strategy"] = human_task_strategy
        if human_task_decisions:
            output_data["human_task_decisions"] = dict(
                state.get("human_task_decisions") or {}
            )

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
                print(f"    - {task['id']}: {task['title']} ({task['executor_role']})")

        print(f"\nWould write to: {args.output}")
        sys.exit(0)

    # Write output JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_temp_dir_gitignore(output_path.parent)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    # --- the write executor (D3n): schedule the batches in this process --------
    # Strictly additive: with neither flag set this returns None and everything
    # below is byte-for-byte the native path that shipped before the executor.
    scheduled = run_implement_write_phase(
        args,
        plan_path=plan_path,
        state=state,
        output_data=output_data,
        human_task_strategy=human_task_strategy,
        human_task_decisions=dict(state.get("human_task_decisions") or {}),
    )
    if scheduled is not None:
        return scheduled

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

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
