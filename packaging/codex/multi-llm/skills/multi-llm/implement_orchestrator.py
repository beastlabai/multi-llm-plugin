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
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from utils.cli_common import (
    EXIT_NEEDS_HUMAN,
    build_parent_parser,
    dispatcher_command,
    record_artifacts,
    record_error,
    record_notice,
)
from utils.output_handler import get_phase_dir, get_output_dir
from utils.git_utils import get_project_root
from utils.provider_registry import set_run_anchor
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
    check_apply_suggestions_prerequisite as _shared_apply_suggestions_prerequisite,
    check_apply_task_suggestions_prerequisite,
    record_prerequisite_skip,
    upstream_skip_notices,
)

logger = logging.getLogger(__name__)


def check_apply_suggestions_prerequisite(state, plan_dir: Path) -> dict:
    """Row 6: whether apply-suggestions still owes work before implement runs.

    The gate is ``check_workflow_prerequisites``'s (D6a: the gates live there,
    never in a thin orchestrator copy), which reads every ``validation.json``
    shape and returns the ``valid_count``/``importance_breakdown`` the unmet
    block below renders. The copy this used to be read only the v1 flat dict
    and crashed on the v2 envelope (``validation.values()`` handed it an int)
    whenever the gate was actually reached (fix round 6, 2026-10-05).
    """
    return _shared_apply_suggestions_prerequisite(state, plan_dir)


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
        metavar='ID=done|skip|retry',
        help='Answer one human task (D6 row 9), repeatable: --human-task '
             'T004=done. Turns the pause-and-ask pause into a process boundary '
             'instead of a blocked stdin. With --resume it also answers a code '
             'task the last run recorded failed (a check that failed): done '
             'treats it as satisfied and runs the tasks after it, skip leaves '
             'it and its dependents undone, retry runs a check again even '
             'though nothing in the repository changed since it failed.'
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
    # Only record if not already recorded (avoid overwriting on --resume) —
    # by implement. `head_before_implement` is recorded right after this on
    # implement's first run, so a baseline without it was left by a planning
    # phase (apply-suggestions records one at its own start), before that
    # phase edited the plan: kept, it counted the plan edit as implement's
    # change (2026-10-05, OpenCode racing run 3: the plan listed `modified`
    # by implementation). Implement's first run takes its own.
    if state.get("pre_existing_changes") is not None:
        if state.get("head_before_implement") is not None:
            print(f"Pre-existing changes already recorded ({len(state.get('pre_existing_changes', []))} paths)")
            return
        print(
            f"Replacing the dirty-tree baseline an earlier phase recorded "
            f"({len(state.get('pre_existing_changes') or [])} paths) with "
            f"implement's own"
        )

    # Get all modified files (staged and unstaged)
    # Filter empty strings before set union for robustness
    try:
        unstaged = set(f for f in get_modified_files() if f)
        staged = set(f for f in get_staged_files() if f)

        pre_existing = unstaged | staged

        # Sorted, as the apply phases record theirs: `state.json` must not
        # change with the set's per-process order (17 bundles of one
        # container run held 17 orders of the same 5 paths).
        state.set("pre_existing_changes", sorted(pre_existing))
        state.save()

        if pre_existing:
            print(f"Recorded {len(pre_existing)} pre-existing changed paths")
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

### Changes Applied in Prior Batches
{{prior_changes_context}}

### Instructions
1. Implement the task as described
2. Follow existing code patterns and conventions
3. Write clean, maintainable code
4. Add appropriate error handling
5. Do NOT make git commits - the orchestrator handles git state

### Output
After implementation, describe what you did and any issues encountered.
"""


#: The one sentence that says whose job these criteria are. A plan's criteria are
#: routinely written as *manual* verification steps ("serving the repo root with
#: a local `http.server` and opening http://localhost:8000/ renders …"), and
#: pasted in unannotated a runner reads them as its own checklist: measured
#: 2026-09-16, a write subagent spent 10 of an 11-minute task driving headless
#: Chrome against a server it had started, left the server orphaned, and then
#: `killall -9 … python3`'d the detached orchestrator that dispatched it (F2).
#: Task-level manual work is already routed away from every dispatcher by
#: `executor_role == "human"`; criteria have no such marker, so they get a frame.
ACCEPTANCE_CRITERIA_FRAMING = (
    "How a human will verify the finished feature after this phase — implement "
    "so these would pass; do not attempt the browser or manual steps yourself."
)


def _checklist_bullets(criteria: List[str]) -> Tuple[List[str], int]:
    """Valid, length-capped ``- [ ] ...`` bullet lines, plus the overflow count.

    The shared core of the agent-facing acceptance-criteria section and the
    human-task checklist (N4) — both bound the same way, so the cap lives
    once.
    """
    MAX_CRITERIA = 10
    MAX_CRITERION_CHARS = 200

    # Filter out empty/whitespace-only strings
    valid = [c.strip() for c in criteria if isinstance(c, str) and c.strip()]
    overflow = max(len(valid) - MAX_CRITERIA, 0)
    items = valid[:MAX_CRITERIA]

    bullets = []
    for item in items:
        if len(item) > MAX_CRITERION_CHARS:
            item = item[:MAX_CRITERION_CHARS - 3] + "..."
        bullets.append(f"- [ ] {item}")
    return bullets, overflow


def _build_acceptance_criteria_section(criteria: List[str]) -> str:
    """Build the acceptance criteria section from a list of criteria.

    Args:
        criteria: List of acceptance criteria strings.

    Returns:
        Formatted section string, or empty string if no valid criteria.
    """
    bullets, overflow = _checklist_bullets(criteria)
    if not bullets:
        return ""

    lines = ["## Acceptance Criteria", ACCEPTANCE_CRITERIA_FRAMING, "", *bullets]

    if overflow > 0:
        lines.append(f"_(+{overflow} more criteria; see full plan)_")

    return "\n".join(lines) + "\n"


#: Rendered instead of the write-subagent template for a task whose
#: ``executor_role`` is ``"human"`` (N4, 2026-09-16). The standard template's
#: ``ACCEPTANCE_CRITERIA_FRAMING`` sentence ("do not attempt the browser or
#: manual steps yourself") and its terminal "Proceed with the implementation."
#: are both written for a write subagent, and directly contradict a task whose
#: whole job *is* the browser/manual work the framing forbids — a materialized
#: T014 once read both in the same file. (The process-safety rules the
#: template's ``## Constraints`` block once carried live only in
#: ``_with_result_contract`` now, appended per dispatch and never rendered
#: for a human task.)
#: Nothing dispatches a human task (``executor_role == "human"`` routes
#: around every invoker), so there is no runner to instruct: the rendered
#: entry exists only for the person who has to do the work by hand.
HUMAN_TASK_DEFERRAL_LINE = "This task is deferred to a human; nothing dispatches it."


def build_human_task_prompt(task: Task) -> str:
    """The task plus its checklist, for a task nothing ever dispatches.

    Deliberately not :func:`build_task_prompt`'s template — see
    ``HUMAN_TASK_DEFERRAL_LINE`` for why the omissions matter.
    """
    lines = ["## Task", f"Title: {task.title}", f"Description: {task.description}", ""]

    bullets, overflow = _checklist_bullets(task.acceptance_criteria or [])
    if bullets:
        lines.append("## Checklist")
        lines.extend(bullets)
        if overflow > 0:
            lines.append(f"_(+{overflow} more criteria; see full plan)_")
        lines.append("")

    lines.append(HUMAN_TASK_DEFERRAL_LINE)
    return "\n".join(lines)


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
    is_human = task.executor_role == "human"
    # A human task is never dispatched (N4): its entry is the checklist a
    # person reads, not a write-subagent prompt, so it skips the template
    # that would tell them not to do the exact work this task asks for.
    prompt = (
        build_human_task_prompt(task)
        if is_human
        else build_task_prompt(task, plan_preamble, plan_path, all_tasks=all_tasks)
    )
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "prompt": prompt,
        # D2h: `executor_role` is what every reader branches on; `subagent_type`
        # rides along as optional Claude-only metadata for hosts whose native
        # Task tool requires the parameter.
        "executor_role": task.executor_role,
        "subagent_type": task.subagent_type,
        "is_human": is_human,
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
    """Format task batches for JSON output.

    ``skip_completed`` (i.e. ``--resume``) keeps only work that is still
    outstanding — ``pending``, ``failed`` or ``in_progress``, never a task the
    last run recorded ``completed`` — **plus every human task that is not
    itself done**. A deferred
    human task is recorded ``skipped``, not ``pending``, so a plain PENDING
    filter dropped it from the resumed run entirely: the scheduler never saw it,
    ``resolve_human_task_plan`` had no human id to resolve, and the resumed
    phase's ``write_batch_report.json`` reported ``skipped_tasks: {}`` while its
    ``summary.md`` lost the ``## Manual Tasks (Deferred)`` section that is the
    only place the finished run tells the reader what is still owed by hand
    (measured 2026-09-16). A human task is never dispatched by either invoker,
    so carrying it through costs the resume nothing.

    A **skipped code task** rides along for the same reason (fix round 8): a
    dependent skipped with a deferred manual task, or a failed task answered
    ``--human-task <id>=skip`` and its dependents, is ``skipped`` too, and
    dropping it put it in no bucket of the resumed run's account — the
    summary read ``5 — 4 done, 1 deferred manual`` for a six-task plan, the
    stop block's "that is every one of the N task(s)" was false, and
    ``write_batch_report.json``'s ``skipped_tasks`` lost the chain. The
    scheduler re-resolves the same plan from the recorded strategy and
    answers, so the task is skipped again, never dispatched.
    """
    output_batches = []

    for batch_idx, batch in enumerate(batches):
        # Filter tasks if skipping completed
        tasks_to_include = batch
        if skip_completed:
            # PENDING alone dropped the task the last run recorded `failed`: a
            # clean failure or partial application stops the loop at exit 3
            # promising "--resume re-dispatches this batch", and the resumed
            # run then never materialized it, ran the rest and stamped the
            # phase completed with that task undone (2026-09-16).
            # IN_PROGRESS is a run killed mid-dispatch — the same debt.
            tasks_to_include = [
                t for t in batch
                if t.status in (TaskStatus.PENDING, TaskStatus.FAILED,
                                TaskStatus.IN_PROGRESS, TaskStatus.SKIPPED)
                or (t.executor_role == "human"
                    and t.status != TaskStatus.COMPLETED)
            ]

        if not tasks_to_include:
            continue

        output_batches.append({
            "batch_index": batch_idx,
            "can_parallelize": len(tasks_to_include) > 1,
            "tasks": [format_task_for_output(t, plan_preamble, plan_path, all_tasks=all_tasks) for t in tasks_to_include]
        })

    return output_batches


def maybe_detach_implement_phase(
    args: argparse.Namespace, *, plan_path: Path
) -> Optional[int]:
    """The one detach boundary ``implement`` crosses (D4), or ``None``.

    Deliberately *outside* the ``--exec-subagents`` gate, exactly as the apply
    phases separate ``maybe_detach_write_phase`` from ``run_write_executor``:
    ``modes.py`` registers ``implement`` with ``supports_detach=True``, so the
    dispatcher accepts ``--detach`` for every implement invocation, not only the
    executor-backed one. Leaving the check under the gate made
    ``implement <plan> --detach --yes`` run in the foreground with no worker, no
    ``.status.json`` and no ``[DETACHED_RUN]`` handshake — a flag that silently
    does nothing, which is what the dispatcher's own ``supports_detach`` refusal
    exists to prevent, and a host polling ``status`` would wait forever for a run
    that was never recorded.

    It sits after every pre-flight — plan loaded, prerequisites gated, tasks
    decomposed, human-task strategy resolved, task JSON written — and before any
    child is dispatched, so a bad flag fails on the caller's terminal with no
    child and no log.

    Returns ``None`` to keep running in this process (the ordinary foreground run,
    and also the detached worker once it owns the phase dir) or an exit code the
    caller must return immediately (the invoker, after forking; or a launch
    failure).
    """
    if not (getattr(args, "detach", False)
            or getattr(args, "internal_worker", False)):
        return None

    from utils.review_orchestrator_base import maybe_detach

    phase_dir = str(get_phase_dir(plan_path, "implement"))
    Path(phase_dir).mkdir(parents=True, exist_ok=True)
    return maybe_detach(
        args=args, mode="implement", phase_dir=phase_dir,
        plan_file=str(plan_path.absolute()), phase="implement",
    )


def _applicable_task_decisions(
    state: StateManager,
    human_task_ids: List[str],
    this_run: Dict[str, str],
) -> Dict[str, str]:
    """The ``--human-task`` answers the scheduler applies this run.

    Every recorded answer for a human task (they persist, so ``--resume``
    never re-asks), plus this run's ``retry`` answers (one-shot, never
    recorded) and a recorded ``done``/``skip`` for a code task only while
    the state still says that task ``failed`` — or, for ``skip``, ``skipped``:
    the answer that skipped it is what keeps it skipped on the next resume,
    now that a skipped code task is materialized again (fix round 8). Once
    it ran again or was accepted, a stale answer must not mark it done
    behind the user's back.
    """
    recorded = dict(state.get("human_task_decisions") or {})
    decisions: Dict[str, str] = {}
    for task_id, action in recorded.items():
        status = state.get_task_status(task_id)
        if task_id in human_task_ids:
            decisions[task_id] = action
        elif status == "failed" or (action == "skip" and status == "skipped"):
            decisions[task_id] = action
    for task_id, action in this_run.items():
        if action == "retry" and task_id not in human_task_ids:
            decisions[task_id] = action
    return decisions


def _stop_user_commands(args: argparse.Namespace) -> Dict[str, str]:
    """The way forward after a stop, in this host's own command spelling.

    Built from the host's ``next_command`` template the way the result line's
    ``user_next_step`` and ``resume_command`` are (``next_phase.phase_command``),
    with the plan as the user typed it (the run record's ``plan_arg``, else
    the ``--plan-file`` value). Empty when nothing can be spelled (no template
    for this host and route): the stop message then names the flags.
    """
    try:
        from utils import detach as _detach
        from utils import host_capabilities as _hosts
        from utils import next_phase as _next

        host_id = _hosts.resolve_host_id(explicit=getattr(args, "host", None))
        route = _next.resolve_route(host_id)
        handle = _detach.active_run()
        plan_arg = (
            (handle.plan_arg if handle is not None else None)
            or os.environ.get(_detach.PLAN_ARG_ENV)
            or str(args.plan_file)
        )

        def command(mode: str, *flags: str) -> str:
            return _next.phase_command(plan_arg, host_id, route, mode, *flags)

        return {
            "resume": command("implement", "--resume"),
            "review": command("review-code"),
            "fix": command("apply-code-fixes"),
            "accept": command("implement", "--resume", "--human-task", "{task_id}=done"),
            "skip": command("implement", "--resume", "--human-task", "{task_id}=skip"),
            "retry": command("implement", "--resume", "--human-task", "{task_id}=retry"),
        }
    except Exception:  # noqa: BLE001 - advisory: the flags are named instead
        return {}


def _seed_pending_task_statuses(state: StateManager, tasks: List[Dict[str, Any]]) -> None:
    """F5: give every materialized task a ``pending`` status before dispatch.

    ``WriteBatchScheduler`` only calls ``state.update_task_status`` when a task
    finishes or is skipped (``write_batch_scheduler.py:1864,2129``) — nothing
    seeds the map before the loop starts. On a 14-task plan that made
    ``check_workflow_prerequisites``' denominator (``len(task_status)``) read
    "tasks recorded so far" instead of "tasks in the plan": a workflow snapshot
    taken mid-run showed ``(1/2 tasks)`` and later ``(6/7 tasks)`` on the same
    14-task implement phase. Only a task with no status yet is touched — a
    ``--resume`` restores completed/failed/skipped statuses from disk before
    this runs, and re-marking one of those ``pending`` would erase it.
    """
    seeded = False
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if task_id and state.get_task_status(task_id) is None:
            state.update_task_status(task_id, TaskStatus.PENDING.value)
            seeded = True
    if seeded:
        state.save()


def _record_implement_write_artifacts(
    *,
    phase_dir: str,
    output_dir: str,
    task_file: str,
    plan_file: str,
    result: Any,
) -> None:
    """Hand back the files this phase produced (F7).

    ``implement`` never goes through ``ApplyOrchestratorBase`` — this function
    drives ``WriteBatchScheduler`` directly — so it never got that base's
    ``_record_write_artifacts``: a fully completed detached run's result line
    said ``"artifacts": []`` and ``status`` had no ``output_files:`` block,
    unlike every apply phase. This mirrors ``ApplyOrchestratorBase.
    _record_write_artifacts``'s shape (same two channels: ``record_output_files``
    feeds ``.status.json`` for ``status``, ``record_artifacts`` feeds the
    terminal ``MULTI_LLM_RESULT`` line) for the files implement itself writes,
    plus the file(s) the write executor actually verified changed
    (``files_touched`` on every ``ok`` outcome — e.g. ``index.html``), never the
    runner's own claim. Every path is existence-guarded, so a stopped run names
    only what it produced.

    ``files_touched`` entries are the pinned-repo-root-relative paths
    ``TaskSnapshot.changed()`` records (D3f/D3h), not paths relative to this
    process's own cwd — a host can invoke ``implement`` from anywhere, and
    checking existence against the wrong root would silently drop every touched
    file. Resolved against the same ``write_safety.resolve_repo_root`` anchor
    the scheduler itself pins the write path to, never the process cwd.
    """
    paths: Dict[str, str] = {"write_tasks": task_file}
    report_path = getattr(result, "report_path", None)
    paths["write_batch_report"] = report_path or os.path.join(
        phase_dir, "write_batch_report.json"
    )
    # The executor's own per-dispatch report (BatchReport.write(), F9/D3i) —
    # `implement` is not one of the kinds that shares a phase dir
    # (KINDS_SHARING_A_WRITE_PHASE_DIR), so its filename is the bare one.
    paths["subagent_batch_report"] = os.path.join(
        phase_dir, "subagent_batch_report.json"
    )
    paths["prior_changes"] = os.path.join(output_dir, "prior_changes.jsonl")
    summary_file = getattr(result, "summary_file", None)
    paths["summary"] = summary_file or os.path.join(phase_dir, "summary.md")

    from utils import write_safety as _write_safety

    repo_root = _write_safety.repo_root_for_report(plan_file)

    touched: List[str] = []
    seen: set = set()
    for outcome in getattr(result, "outcomes", None) or []:
        if not getattr(outcome, "ok", False):
            continue
        for touched_path in getattr(outcome, "files_touched", None) or []:
            if touched_path and touched_path not in seen:
                seen.add(touched_path)
                touched.append(touched_path)
    for index, touched_path in enumerate(touched, start=1):
        resolved = (
            touched_path if os.path.isabs(touched_path)
            else os.path.join(repo_root, touched_path)
        )
        paths[f"touched_file_{index}"] = resolved

    from utils import detach as _detach

    _detach.record_output_files(phase_dir, **paths)
    record_artifacts(path for path in paths.values() if path and os.path.exists(path))


def check_findings_notice(found: Dict[str, List[str]]) -> str:
    """The result line's ``notice`` for check tasks that reported problems.

    Bounded: one line a host relays, naming each task, how many problems its
    check reported and the first of them, and where the rest are.
    """
    parts = []
    for task_id, findings in found.items():
        first = findings[0] if findings else ""
        more = f" (+{len(findings) - 1} more)" if len(findings) > 1 else ""
        parts.append(f"{task_id}: {first[:160]}{more}")
    return (
        "check findings (implement continued; review-code is told about "
        "them; all listed in implement/summary.md): " + "; ".join(parts)
    )[:900]


def run_implement_write_phase(
    args: argparse.Namespace,
    *,
    plan_path: Path,
    state: StateManager,
    output_data: Dict[str, Any],
    human_task_strategy: Optional[str],
    human_task_decisions: Dict[str, str],
    tasks_file: Optional[Path] = None,
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

    The detach boundary is **not** here: it is
    :func:`maybe_detach_implement_phase`, which the caller crosses just above this
    call so ``--detach`` works on every implement invocation rather than only the
    executor-backed one.

    ``tasks_file`` (F15) is the external ``tasks/tasks.md`` this run's prompts
    were actually parsed from, when the plan carries a ``TASKS_FILE`` marker —
    ``None`` for the legacy embedded-task format. Threaded into the two
    materializer calls so ``write_tasks.json`` records ``inputs.tasks_sha256``
    the way ``review-tasks`` and ``apply-task-suggestions`` already do, and a
    hand-edited tasks file invalidates a stored ``write_tasks.json`` on
    ``--resume`` the same way an edited plan does.
    """
    if not (getattr(args, "exec_subagents", False)
            or getattr(args, "dry_run_subagents", False)):
        return None

    from utils.write_batch_scheduler import WriteBatchScheduler
    from utils.write_phase_bridge import (
        materialize_write_tasks,
        write_task_file_for_phase,
    )

    phase_dir = str(get_phase_dir(plan_path, "implement"))
    Path(phase_dir).mkdir(parents=True, exist_ok=True)
    plan_file = str(plan_path.absolute())
    output_dir = str(get_output_dir(plan_path))
    tasks_file_str = str(tasks_file) if tasks_file else None

    from utils import write_safety as _write_safety

    tasks = materialize_write_tasks(
        kind="implement",
        batches=output_data.get("batches") or [],
        phase_dir=phase_dir,
        plan_file=plan_file,
        tasks_file=tasks_file_str,
    )
    task_file = write_task_file_for_phase(
        kind="implement", tasks=tasks, phase_dir=phase_dir, plan_file=plan_file,
        tasks_file=tasks_file_str,
    )
    print(f"[write-subagents] materialized {len(tasks)} implement task(s): {task_file}")

    # F5: seed before the scheduler loop starts, not inside it — see the
    # function's own docstring.
    _seed_pending_task_statuses(state, tasks)

    try:
        scheduler = WriteBatchScheduler(
            tasks=tasks,
            phase_dir=phase_dir,
            kind="implement",
            output_dir=output_dir,
            state=state,
            plan_file=plan_file,
            summary_file=output_data.get("summary_file"),
            summary_file_relative=output_data.get("summary_file_relative"),
            anchor=plan_file,
            resume=bool(getattr(args, "resume", False)),
            dry_run=bool(getattr(args, "dry_run_subagents", False)),
            human_task_strategy=human_task_strategy,
            human_task_decisions=human_task_decisions,
            user_commands=_stop_user_commands(args),
        )
    except _write_safety.WriteSafetyError as exc:
        # A repository root that cannot be pinned (a planted `core.worktree`,
        # review fixes 2026-10-05) is exit 3 with the resolver's own words.
        print(f"ERROR: {exc.message}", file=sys.stderr)
        return int(getattr(exc, "exit_code", 3) or 3)
    result = scheduler.run()
    for line in result.messages:
        print(line, file=sys.stderr if result.exit_code else sys.stdout)
    stop = getattr(result, "stop", None) or {}
    if stop.get("headline"):
        # The result line's `error`: the one line the hosts relay as "the
        # last line it left", and the detached close-out's `run.error`.
        record_error(stop["headline"])
    found = result.check_findings() if hasattr(result, "check_findings") else {}
    if found:
        # Racing re-run 4 (2026-10-05): a check task's problems were on no
        # surface a host reads. The result line's `notice` carries them,
        # beside the `[CHECK_FINDINGS]` lines above.
        record_notice(check_findings_notice(found))
    if result.report_path:
        print(f"[write-subagents] report: {result.report_path}")
    if result.summary_file:
        print(f"[write-subagents] implementation summary: {result.summary_file}")
    _record_implement_write_artifacts(
        phase_dir=phase_dir, output_dir=output_dir, task_file=task_file,
        plan_file=plan_file, result=result,
    )
    return result.exit_code


def _materialize_native_write_tasks(
    plan_path: Path, output_data: Dict[str, Any], *, tasks_file: Optional[Path],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Write ``{phase}/write_tasks.json`` for the native leg; its path and tasks.

    The same two bridge calls :func:`run_implement_write_phase` makes, with
    the same inputs, so the file a native host's ``util write-prompt`` reads
    is the one the executor leg would have dispatched from.
    """
    from utils.write_phase_bridge import (
        materialize_write_tasks,
        write_task_file_for_phase,
    )

    phase_dir = str(get_phase_dir(plan_path, "implement"))
    plan_file = str(plan_path.absolute())
    tasks_file_str = str(tasks_file) if tasks_file else None
    tasks = materialize_write_tasks(
        kind="implement",
        batches=output_data.get("batches") or [],
        phase_dir=phase_dir,
        plan_file=plan_file,
        tasks_file=tasks_file_str,
    )
    path = write_task_file_for_phase(
        kind="implement", tasks=tasks, phase_dir=phase_dir, plan_file=plan_file,
        tasks_file=tasks_file_str,
    )
    return path, tasks


def _native_human_task_plan(
    output_data: Dict[str, Any],
    *,
    strategy: Optional[str],
    decisions: Dict[str, str],
    on_record: Optional[Any] = None,
    recorded_statuses: Optional[Dict[str, str]] = None,
):
    """Resolve the human-task plan over the native leg's batches; annotate them.

    The same ``resolve_human_task_plan`` the executor runs over the same
    tasks. Every task the plan skips (a deferred manual task, and under
    ``skip-dependents`` each task that depends on one) gets
    ``status: "skipped"`` and its ``skip_reason`` in the task JSON, a manual
    task answered ``done`` gets ``status: "completed"``, and the JSON gains a
    top-level ``skipped_tasks`` map: the host reads that it must not
    dispatch them instead of re-deriving the walk. With no strategy yet (an
    attended run, before the host asks) nothing is skipped.

    A failed code task the user accepted (``--human-task <id>=done`` on a
    resume) is **taken out of the batches**, exactly as a completed task is,
    and named in a top-level ``accepted_tasks`` map with its reason: a task
    left in the batches is one the host dispatches, and until fix round 8
    the native leg recorded the acceptance in ``state.json`` while handing
    the task out again as ``failed`` — whose ``util batch-done`` then
    overwrote the acceptance (reviewer repro, 2026-10-05).

    *on_record* is ``write_batch_scheduler.answered_done_on_record``'s
    ``(accepted, completed_manual)`` for the tasks this emit did not
    materialize: an answer an earlier run recorded keeps its place in
    ``accepted_tasks``, and a manual task answered ``done`` is named in a
    top-level ``completed_manual_tasks`` list, on every later resume too
    (fix round 9: both dropped out once the task was recorded completed).

    A task *recorded_statuses* (``state.json``'s ``task_status``) says
    ``skipped`` that this plan no longer skips — ``--resume --force-strategy
    skip-continue`` after a ``skip-dependents`` run — is handed out
    ``pending`` and recorded so (``write_batch_scheduler.mark_no_longer_skipped``),
    never left ``skipped`` for the host to pass over (fix round 9).
    """
    from utils.write_batch_scheduler import (
        mark_no_longer_skipped,
        resolve_human_task_plan,
    )

    batches = output_data.get("batches") or []
    entries = [task for batch in batches for task in batch.get("tasks") or []]
    plan = resolve_human_task_plan(
        [
            {
                "task_id": task.get("id"),
                "task_title": task.get("title"),
                "executor_role": task.get("executor_role"),
                "is_human": task.get("is_human"),
                "depends_on": list(task.get("depends_on") or []),
            }
            for task in entries
        ],
        strategy=strategy,
        decisions=decisions,
    )
    for task in entries:
        task_id = str(task.get("id") or "")
        if task_id in plan.skipped:
            task["status"] = TaskStatus.SKIPPED.value
            task["skip_reason"] = plan.skipped[task_id]
        elif task_id in plan.completed:
            task["status"] = TaskStatus.COMPLETED.value
    unskipped = set(mark_no_longer_skipped(
        plan,
        [str(task.get("id") or "") for task in entries],
        {
            **(recorded_statuses or {}),
            **{
                str(task.get("id") or ""): task.get("status")
                for task in entries if task.get("status") == TaskStatus.SKIPPED.value
            },
        },
    ))
    for task in entries:
        if str(task.get("id") or "") in unskipped:
            task["status"] = TaskStatus.PENDING.value
            task.pop("skip_reason", None)
    if on_record:
        accepted, completed_manual = on_record
        for task_id, reason in accepted.items():
            plan.accepted.setdefault(task_id, reason)
        for task_id in completed_manual:
            if task_id not in plan.completed:
                plan.completed.append(task_id)
    if plan.skipped:
        output_data["skipped_tasks"] = dict(sorted(plan.skipped.items()))
    if plan.completed:
        output_data["completed_manual_tasks"] = sorted(plan.completed)
    if plan.accepted:
        kept = []
        for batch in batches:
            batch["tasks"] = [
                task for task in batch.get("tasks") or []
                if str(task.get("id") or "") not in plan.accepted
            ]
            if batch["tasks"]:
                batch["can_parallelize"] = len(batch["tasks"]) > 1
                kept.append(batch)
        output_data["batches"] = kept
        output_data["accepted_tasks"] = dict(sorted(plan.accepted.items()))
        if "pending_tasks" in output_data:
            output_data["pending_tasks"] = sum(len(b["tasks"]) for b in kept)
    return plan


def _refuse_unrecorded_native_resume(
    tasks: List[Dict[str, Any]], *, plan_path: Path, state: StateManager, resume: bool,
) -> bool:
    """Print the refusal block and return True when a native ``--resume`` must stop.

    A task about to be handed out again whose id the implement log already
    holds, with no ``task_status`` record (``util batch-done``) saying it
    finished or failed, is one the host ran without recording: re-emitting
    it would run it twice. See :func:`utils.batch_done.unrecorded_logged_batches`.
    """
    if not resume or not tasks:
        return False
    from utils.batch_done import resume_refusal_lines, unrecorded_logged_batches

    output_dir = str(get_output_dir(plan_path))
    unrecorded = unrecorded_logged_batches(
        tasks, output_dir=output_dir, phase="implement", state=state.state,
    )
    if not unrecorded:
        return False
    phase_dir = str(get_phase_dir(plan_path, "implement"))
    for line in resume_refusal_lines(unrecorded, phase_dir=phase_dir, mode_word="implement"):
        print(line, file=sys.stderr)
    return True


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

    # §9.3.a/b: the anchor capture point — absolute, before anything can chdir
    # and before any provider read that decides routing.
    set_run_anchor(str(plan_path))

    # Resolve the default --output now that the plan argument is known (the
    # default depends on the git root and the plan stem, so it cannot be a
    # static argparse default).
    if args.output is None:
        args.output = resolve_default_output(plan_path)

    # Load plan
    plan_content = plan_path.read_text(encoding="utf-8")
    print(f"Loaded plan: {plan_path}")

    # Initialize state manager. Constructing it only *reads* (or synthesizes in
    # memory) the document — the first `save()` deliberately waits until the row-6
    # gate below has been cleared, so an unmet gate really does leave nothing
    # behind, exactly as its comment claims.
    state = get_or_create_state(plan_path)
    print(f"State file: {state.state_file}")

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
    # state.json mutation (the first `state.save()` is below this block, not
    # above it), so an unmet gate leaves no half-generated artifact.
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

    # The gate is cleared (or explicitly skipped, which recorded its own save):
    # only now does this run own a state.json on disk.
    state.save()

    # Preflight resolutions are replayed after the run has claimed its phase.
    pending_resolutions: List[Tuple[str, str]] = []
    task_skip_reason = "auto-skipped by --yes/--no-confirm (D6 row 7)"
    task_skip_decision = "skipped; apply-task-suggestions recorded as skipped"
    # The detached launcher may already have saved this choice. Its worker
    # then sees the prerequisite as met, but must still report the choice that
    # lets this run proceed. Explicit human skips keep their existing notice.
    if (unattended or args.no_confirm) and (
        state.get_phase_skip_reason("apply-task-suggestions") == task_skip_reason
    ):
        pending_resolutions.append((unattended_mod.row(7).site_id, task_skip_decision))

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
                reason=task_skip_reason,
            )
            unattended_mod.record_resolution(
                None,
                unattended_mod.row(7).site_id,
                task_skip_decision,
            )
            pending_resolutions.append((unattended_mod.row(7).site_id, task_skip_decision))

    # Check if generate-tasks phase has been completed. Two endings, decided by
    # the same rule the row-6 gate above uses:
    #   attended    -> today's [TASKS_MISSING] payload, exit 0 — the driving agent
    #                  reads the marker and runs generate-tasks itself.
    #   --yes       -> exit 3 naming the generate-tasks re-invocation. There is no
    #                  agent to read a marker on this path, and exit 0 would report
    #                  MULTI_LLM_RESULT state "done" for a run that implemented
    #                  nothing — a silent no-op reported as success.
    if not state.is_phase_completed("generate-tasks"):
        print("[TASKS_MISSING]")
        print(json.dumps({"marker": "TASKS_MISSING", "reason": "No implementation tasks found"}))
        if unattended:
            missing = [{
                "phase": "generate-tasks",
                "reason": "No implementation tasks found",
            }]
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
            # The plan names a task list that is not on disk (a moved or cleaned
            # `plans/<plan>/tasks/`, a plan copied without its sibling dir).
            # Parsing the plan instead does not fail — it *fabricates*: measured
            # 2026-09-16 against a real plan, the fallback returned 18 confident
            # tasks titled ". Overview" and ".1 Rendering", with zero acceptance
            # criteria and `camera.update` declared as a file to create. A
            # WARNING into a detached log is not a gate, and this is the same
            # rule `generate_tasks_orchestrator.verify_outcome` applies at the
            # other end: a task list nobody generated is worse than none. Only a
            # plan with NO marker at all (the legacy embedded-task format) still
            # parses its own content.
            print(
                f"ERROR: the plan's <!-- TASKS_FILE: {tasks_file_ref} --> marker "
                f"names a task list that does not exist:\n"
                f"       {tasks_file_path}\n"
                f"       Nothing was implemented. Regenerate the task list — "
                + dispatcher_command("generate-tasks", plan_path.absolute(), "--yes")
                + "\n       then re-run this command. (Parsing the plan itself "
                "instead would invent a task list from its prose, which every "
                "later phase would treat as ground truth.)"
            )
            unattended_mod.mark_needs_human("implement")
            print(json.dumps({
                "marker": "TASKS_FILE_MISSING",
                "mode": "implement",
                "state": "needs-human",
                "exit": EXIT_NEEDS_HUMAN,
                "tasks_file": str(tasks_file_path),
                "reason": "Plan names a TASKS_FILE that does not exist",
            }, indent=2))
            sys.exit(EXIT_NEEDS_HUMAN)

    # F15: the external tasks file this run's prompts are actually built from,
    # captured here (before `tasks_file_path` is reassigned below to the
    # phase-based `tasks_file` output path) so `run_implement_write_phase` can
    # stamp `write_tasks.json` with `inputs.tasks_sha256` — `None` for the
    # legacy embedded-task format, where there is no such file.
    implement_tasks_file: Optional[Path] = tasks_file_path if tasks_file_ref else None

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
    # Both legs pass here, so a task the run defers (a human task, or a
    # dependent `skip-dependents` skips with it) can be named by its title
    # once the workflow completes, not only by its id: `persist_human_task_plan`
    # (the scheduler's, and this emit's since fix round 7) records the
    # `skipped` status and its reason but no title, and `[MANUAL_TASK]` reads
    # the title from here (2026-10-04).
    # The manual tasks ride along (`implement_human_task_ids`), so a finished
    # implement whose manual task nobody answered still names it as owed.
    if state.record_task_titles(
        {task.id: task.title for task in tasks},
        human_ids=[
            task.id for task in tasks
            if getattr(task, "executor_role", "") == "human"
        ],
    ):
        state.save()

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
    # F8: this fires before `run()`'s equivalent — `maybe_detach_implement_phase`,
    # further down — crosses the detach boundary, so no run handle is claimed
    # yet: recording against the phase dir here is a silent no-op
    # (`detach.record_resolution` only writes when `_ACTIVE` names that phase
    # dir), and `.status.json.resolutions` stayed `[]` even though the log's
    # `[AUTO_RESOLVED]` line printed. Buffer it instead, the way
    # `ApplyOrchestratorBase._record_resolution`/`_flush_resolutions` do, and
    # replay it once this process owns the phase dir, just past the boundary.
    if human_task_ids and human_task_strategy and unattended and not args.force_strategy:
        decision = f"{human_task_strategy} for {len(human_task_ids)} human task(s)"
        unattended_mod.record_resolution(None, unattended_mod.row(8).site_id, decision)
        pending_resolutions.append((unattended_mod.row(8).site_id, decision))
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
        # A code task is answerable only once it FAILED and only on a resume
        # (racing re-run 3, 2026-10-05: a failed check blocked implement for
        # good; `done` accepts the failure, `skip` leaves it and its dependents
        # undone, `retry` re-runs a check the unchanged-tree guard refuses).
        failed_ids = {
            tid for tid in not_human if state.get_task_status(tid) == "failed"
        }
        if not_human and not args.resume:
            return unattended_mod.fail_closed(
                "ERROR: --human-task answers a code task only with --resume, "
                "after a run recorded it failed; these are code tasks: "
                f"{', '.join(sorted(not_human))}.",
                site_id=unattended_mod.row(9).site_id,
            )
        unanswerable = sorted(set(not_human) - failed_ids)
        if unanswerable:
            return unattended_mod.fail_closed(
                "ERROR: --human-task answers a human task, or a code task the "
                "last run recorded failed; these are code tasks that did not "
                f"fail: {', '.join(unanswerable)}.",
                site_id=unattended_mod.row(9).site_id,
            )
        retried_human = sorted(
            tid for tid, action in human_task_decisions.items()
            if action == "retry" and tid in human_task_ids
        )
        if retried_human:
            return unattended_mod.fail_closed(
                "ERROR: --human-task <id>=retry re-runs a failed code task; a "
                f"human task is answered done or skip: {', '.join(retried_human)}.",
                site_id=unattended_mod.row(9).site_id,
            )
        recorded = dict(state.get("human_task_decisions") or {})
        recorded.update({
            tid: action for tid, action in human_task_decisions.items()
            if action != "retry"  # one-shot: never carried into a later run
        })
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
        # Format for output
        output_batches = format_batches_for_output(batches, skip_completed=args.resume, plan_preamble=plan_preamble, plan_path=plan_path_str, all_tasks=decomposer.tasks)

        # The result line counts what dispatches ("0/13 batches"), and a manual
        # task is a graph node that dispatches nothing, so a bare "15 execution
        # batches" read as two more than the run would ever report (2026-10-01).
        manual_in_output = sum(
            1 for b in output_batches for t in b["tasks"]
            if t.get("executor_role") == "human"
        )
        dispatching = sum(len(b["tasks"]) for b in output_batches) - manual_in_output
        print(
            f"Organized into {len(batches)} execution batches: {dispatching} "
            f"task(s) to dispatch"
            + (f", {manual_in_output} manual task(s) it does not run"
               if manual_in_output else "")
        )

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

    # Write output JSON, for the one reader that needs it: the driving agent of
    # the native path, which `Output file:` hands it. The write executor reads
    # `write_tasks.json` in the phase dir instead, so an `--exec-subagents` run
    # that never needed `.multi-llm/tmp/` used to create it, `.gitignore`
    # included, even when it was refused at preflight and dispatched nothing
    # (2026-10-02). `--detach` without the executor still writes it: the worker
    # re-enters here and reaches this point itself.
    output_path = Path(args.output)

    def write_native_handoff() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_temp_dir_gitignore(output_path.parent)
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    executor_backed = bool(
        getattr(args, "exec_subagents", False)
        or getattr(args, "dry_run_subagents", False)
    )
    # The answers both legs apply this run: every human task's, and a code
    # task's only while it still applies (`_applicable_task_decisions`).
    applicable_decisions = _applicable_task_decisions(
        state, human_task_ids, human_task_decisions,
    )
    # The native leg resolves the human-task plan here, from the strategy and
    # the answers it already holds, exactly as the executor resolves it before
    # its first batch — so the skips are in the task JSON the host reads and
    # are written to state.json below, not left for the host to record. Also
    # with no manual task in the plan whenever an answer applies: a failed
    # code task's `done`/`skip` used to pass validation, print
    # `[HUMAN_TASK_ANSWERED]` and change nothing on this leg, because the
    # plan was only resolved for a plan holding a human task (fix round 8).
    native_human_plan = None
    answered_on_record = None
    if not executor_backed:
        from utils.write_batch_scheduler import answered_done_on_record

        answered_on_record = answered_done_on_record(
            state,
            exclude=[
                str(task.get("id") or "")
                for batch in output_data.get("batches") or []
                for task in batch.get("tasks") or []
            ],
        )
    if not executor_backed and (
        human_task_ids or applicable_decisions or any(answered_on_record)
    ):
        native_human_plan = _native_human_task_plan(
            output_data,
            strategy=human_task_strategy,
            decisions=applicable_decisions,
            on_record=answered_on_record,
            recorded_statuses=state.get_all_task_statuses(),
        )
    if not executor_backed:
        write_native_handoff()

    # --- the detach boundary (D4) ---------------------------------------------
    # Crossed here, before the write executor and independently of it: `implement`
    # is registered `supports_detach=True`, so `--detach` must produce a worker and
    # a `.status.json` whether or not `--exec-subagents` was passed.
    detached = maybe_detach_implement_phase(args, plan_path=plan_path)
    if detached is not None:
        return detached

    # F8: past the boundary this process owns the phase dir (the ordinary
    # foreground run, or the detached worker once it has claimed it), so the
    # D6 decisions buffered above can finally be written where `status`
    # reads it (`run.resolutions`). Quiet: the marker line already printed at
    # the decision site, and printing it twice would make one decision read as
    # two.
    if pending_resolutions:
        resolutions_phase_dir = str(get_phase_dir(plan_path, "implement"))
        for site_id, decision in pending_resolutions:
            unattended_mod.record_resolution(
                resolutions_phase_dir, site_id, decision, quiet=True
            )

    # --- the write executor (D3n): schedule the batches in this process --------
    # Strictly additive: with neither flag set this returns None and everything
    # below is byte-for-byte the native path that shipped before the executor.
    scheduled = run_implement_write_phase(
        args,
        plan_path=plan_path,
        state=state,
        output_data=output_data,
        human_task_strategy=human_task_strategy,
        human_task_decisions=applicable_decisions,
        tasks_file=implement_tasks_file,
    )
    if scheduled is not None:
        return scheduled

    # The native leg (2026-10-05): the same `write_tasks.json` the executor
    # leg materializes, so the host's per-task `util write-prompt` binds each
    # task's prior-changes context from the same seam instead of the host
    # fishing the prompt out of the handoff JSON and substituting the token
    # by hand (measured 2026-10-04: 3-6 tool calls per subagent, and one that
    # read tasks.md instead of its prompt).
    try:
        write_task_file, native_tasks = _materialize_native_write_tasks(
            plan_path, output_data, tasks_file=implement_tasks_file,
        )
    except Exception as exc:  # noqa: BLE001 - the handoff JSON is still usable
        print(f"WARNING: write_tasks.json not written: {exc}", file=sys.stderr)
    else:
        print(f"[WRITE_TASKS_FILE] {write_task_file}", file=sys.stderr)
        # A native `--resume` promises to skip what already ran: a task the
        # implement log already holds and no record says finished is refused
        # rather than handed out a second time (reviewer repro, 2026-10-05).
        if _refuse_unrecorded_native_resume(
            native_tasks, plan_path=plan_path, state=state, resume=bool(args.resume),
        ):
            return 3

    # Print summary
    print("\n" + "=" * 60)
    print("Tasks Prepared for Claude Code")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"State file: {state.state_file}")
    print(f"Total tasks: {output_data['total_tasks']}")
    print(f"Pending tasks: {output_data.get('pending_tasks', output_data['total_tasks'])}")
    print(f"Batches: {len(output_data['batches'])}")

    if native_human_plan is not None:
        titles = dict(state.get("task_titles") or {})

        def _named(task_id: str) -> str:
            title = titles.get(task_id)
            return f"{task_id}{' ' + title if title else ''}"

        deferred = native_human_plan.deferred()
        declined = native_human_plan.declined()
        if deferred:
            print(
                "Deferred tasks (recorded skipped in state.json; do not "
                "dispatch them, and list them in the summary's Manual Tasks "
                "(Deferred) section):"
            )
            for task_id, reason in sorted(deferred.items()):
                print(f"  {_named(task_id)}: {reason}")
        if declined:
            # Not owed: the workflow's `[MANUAL_TASK]` lines leave these out,
            # so a summary that filed them under Manual Tasks (Deferred)
            # disagreed with `[WORKFLOW_COMPLETE]` (fix round 8).
            print(
                "Skipped at your request (recorded skipped in state.json; do "
                "not dispatch them, and list them in the summary under a "
                "separate 'Skipped at your request' heading, not under Manual "
                "Tasks (Deferred) — they are not owed):"
            )
            for task_id, reason in sorted(declined.items()):
                print(f"  {_named(task_id)}: {reason}")
        if native_human_plan.accepted:
            print(
                "Accepted after failing (recorded completed in state.json; "
                "not in the batches, do not dispatch or record them; list "
                "them in the summary as accepted after failing):"
            )
            for task_id, reason in sorted(native_human_plan.accepted.items()):
                print(f"  {_named(task_id)}: {reason}")
        if native_human_plan.completed:
            print(
                "Completed by hand (manual tasks recorded completed in "
                "state.json; do not dispatch or record them; list them in the "
                "summary as completed by hand):"
            )
            for task_id in sorted(native_human_plan.completed):
                print(f"  {_named(task_id)}: answered --human-task {task_id}=done")

    print("\nClaude Code should now:")
    print("  1. Read the output JSON file")
    print("  2. Execute each batch sequentially (batches have dependencies)")
    print("  3. Tasks within a batch can run in parallel")
    print("  4. Update the state file after each task completes")
    print("  5. After all tasks complete, generate the implementation summary")
    print(f"     Summary file: {output_data['summary_file']}")
    print("  6. Close the phase with `util finalize-tracking … --mark-completed`")

    # The batches are now the host's to run, so the phase is recorded
    # *started*, never completed: completion is the instruction's closing
    # `util finalize-tracking --plan-file … --mark-completed`, which runs only
    # after every task, the verify step and the summary. Until 2026-10-05 this
    # site wrote `mark_phase_completed("implement")`, so `status` read the
    # phase as done minutes before its last task (racing re-run, 2026-10-04).
    if native_human_plan is not None:
        # What the executor's `_persist_human_plan` writes before its first
        # batch: the deferred manual tasks (and the dependents skipped with
        # them) as `skipped` with their reasons, a `done` answer as completed,
        # the skip-continue overrides. Until fix round 7 the native leg left
        # these writes to the host, which never made them, so `task_status`
        # had no T010 and the finished workflow named no manual task.
        from utils.write_batch_scheduler import persist_human_task_plan

        persist_human_task_plan(state, native_human_plan)
    state.mark_phase_started("implement")
    state.save()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
