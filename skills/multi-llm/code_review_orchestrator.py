#!/usr/bin/env python3
"""
Code review orchestrator for reviewing implementation changes.

This orchestrator reviews code changes made during implementation against
the original plan. It identifies issues, validates changes, and generates
a review report.

Supports multiple providers (cursor-agent, gemini, opencode) via the provider
registry. Models can be specified as 'provider:model' or bare 'model' names
(which use the default provider from providers.yaml).

Usage:
    # Use YAML defaults (no prompting if defaults.models is set)
    uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-code plans/my-plan.md

    # Override defaults with specific models
    uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-code plans/my-plan.md --models cursor-agent:auto gemini:gemini-2.5-flash

    # Force interactive selection even if defaults exist
    uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" review-code plans/my-plan.md --interactive
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_or_create_state,
    load_prompt,
    get_output_paths,
    validate_code_review_issues,
    get_files_changed_since_ref,
    group_similar_suggestions,
    export_groups_to_json,
    validate_groups,
    apply_validation_to_groups,
    save_validation_results,
    sanitize_prefix,
    intent_to_add_untracked,
    subtract_baseline,
    PlanUpdater,
)
from utils.stream_bootstrap import bootstrap_streams
from utils import cli_common
from utils.cli_common import (
    EXIT_NEEDS_HUMAN,
    EXIT_OK,
    build_parent_parser,
    dispatcher_prefix,
    reaggregate_command,
    record_next,
)
from utils.state_manager import (
    load_groups_payload,
    save_groups_payload,
    stamp_stable_ids,
    CURRENT_FORMAT_VERSION,
)
from utils.output_handler import get_phase_dir
from utils.detach import record_output_files, report_progress
from utils.review_orchestrator_base import (
    RETIRE_SUPERSEDED,
    _authoritative_validator_runner,
    dispatch_salvage_tasks,
    dispatch_subagent_tasks,
    exec_subagents_requested,
    maybe_detach,
    merge_validation_batches,
    resolve_result_files_by_model,
    retire_salvage_artifacts,
    retire_stale_validation,
    sandbox_refusal,
    update_status,
    write_status,
)
# The read side is named from the module that owns it rather than through the
# base's re-export. That re-export exists because every existing caller and test
# already spelled `write_status`/`update_status` through the base
# (`utils/review_orchestrator_base.py`, "Status file helpers"); there is no such
# legacy for `read_status`, and `utils/detach.py` reaches for `status_file`
# directly too. Its one use below is the read-back that turns a silently failed
# checkpoint write into a warning.
from utils.status_file import read_status
from utils.validation import (
    prepare_batched_validation_tasks,
    merge_batched_validation_results,
    resolved_reaggregate_command,
)
from utils.subagent_prompts import CODE_REVIEW_VALIDATION_GUIDANCE
from utils.subagent_tasks import (
    TaskFileError,
    build_inputs,
    read_task_file,
    write_salvage_task_file,
)
from utils.interactive import resolve_models
from utils import unattended as unattended_mod
from utils.unattended import NeedsHuman
from utils.provider_registry import (
    get_all_model_specs,
    parse_model_spec,
    get_provider_timeout,
    get_provider_max_concurrent,
    is_model_valid,
    set_run_anchor,
)
from utils.json_extractor import (
    build_unsanitize_map,
    generate_output_path,
    has_salvageable_json,
    read_json_from_file,
    sanitize_model_name,
)
from utils.llm_client import describe_output_file, invoke_with_provider, invoke_with_file_output
from utils.git_utils import get_project_root, validate_git_ref, _run_git
from utils.html_report_generator import generate_html_report, write_html_report, served_model_label, VALIDATION_ORDER, UNKNOWN_STATUS_RANK


def derive_prefix(plan_path: Path) -> str:
    """Derive output prefix from plan file path."""
    return sanitize_prefix(plan_path.name)


# ---------------------------------------------------------------------------
# One story about a phase dir whose checkpoint records no models
# ---------------------------------------------------------------------------
#
# Two moments can discover that `models_requested` is not on disk, and neither
# can repair it. The fan-out discovers it immediately after `write_status`,
# which is best-effort *by contract* — it wraps the temp write and the
# `os.replace` in a bare `except Exception: pass` (`utils/status_file.py`),
# returns `None`, and so cannot report a read-only or full filesystem, a phase
# dir made unwritable between the mkdir and the write, a checkpoint lock it
# could not take, or an `os.replace` across a boundary that refuses it.
# `--reaggregate` discovers it later, arriving at a dir whose checkpoint was
# lost that way, was never written (every dir whose fan-out predates the write),
# or was hand-merged without the field. Same defect from either end, so both
# sites print the same consequence and the same two recoveries instead of two
# paraphrases the operator has to reconcile into one.
#
# Warning, never refusal, at both. In the fan-out this sits below the detach
# boundary, after the models are resolved and immediately before dispatch:
# raising there would trade a checkpoint that could not be written for a fan-out
# that never ran — the expensive half of the run, on the machine where nobody is
# watching the terminal. At reaggregation a pre-checkpoint dir plus
# `--skip-validation` produces a report *today*, so failing closed would break a
# path that currently completes and would need an override flag to hand back.
# Fail-closed belongs at `resolve_models` (`utils/interactive.py`), where
# refusing costs a correct run nothing; it is wrong at both of these.

#: What is actually lost, in the operator's terms rather than the mechanism's.
#: `sanitize_model_name` (`utils/json_extractor.py`) is one
#: `re.sub(r'[^a-zA-Z0-9\-_]', '_', model)`, so it is lossy in one direction
#: only and the checkpoint is the sole record of the other.
_STEM_ATTRIBUTION_CONSEQUENCE = (
    "--reaggregate over this phase dir attributes issues by filename stem "
    "instead of by model spec: sanitize_model_name flattens every separator a "
    "spec carries, so 'cursor-agent_cursor-grok-4_6-xhigh' can never be turned "
    "back into 'cursor-agent:cursor-grok-4.6-xhigh'. The report then names a "
    "stem where a model belongs, and nothing downstream contradicts it."
)

#: Both recoveries, each with its cost named, because the choice is the
#: operator's: one costs money and reviews a diff that has moved on, the other
#: costs nothing but needs the specs from a source that still holds them.
_STEM_ATTRIBUTION_RECOVERIES = (
    "Recover either way: re-run the fan-out with --force, which writes the "
    "checkpoint on the way through, at the cost of a second round of model "
    "spend against whatever the diff is by then; or write .status.json in the "
    "phase dir by hand — build_unsanitize_map reads nothing but the canonical "
    "specs listed under models_requested / models_completed."
)


def warn_stem_attribution(
    headline: str, model_specs: Optional[List[str]] = None
) -> None:
    """Print the stem-attribution warning to stderr. Never raises, never exits.

    *headline* is the only part that differs between the two sites: it says
    which moment found the gap and in what tense. The consequence and the
    recoveries are shared verbatim so the fan-out's warning and the
    reaggregation's warning are recognisably the same story.

    *model_specs* is passed by the fan-out alone, where the specs are still in
    hand. Printing them makes this warning itself the canonical source the
    hand-written-checkpoint recovery needs — otherwise the operator has to
    reconstruct it from console scrollback, and a stem copied out of an old
    report is exactly the thing that cannot be inverted.

    stderr, not stdout: stdout is this phase's machine-readable channel (the
    `[VALIDATION_BATCHES_PENDING]`, `[PARTIAL_RESULTS: ...]` and
    `[REAGGREGATE_AFTER_SALVAGE]` markers a host parses), and a degradation
    nobody chose does not belong in it.
    """
    lines = [
        headline,
        f"  {_STEM_ATTRIBUTION_CONSEQUENCE}",
        f"  {_STEM_ATTRIBUTION_RECOVERIES}",
    ]
    if model_specs:
        lines.append(
            "  This run's canonical specs, for that second recovery: "
            + ", ".join(model_specs)
        )
    print("\n".join(lines), file=sys.stderr)


def save_model_result(
    prefix: str,
    model: str,
    success: bool,
    issues: Optional[List[Dict]],
    error: Optional[str],
    phase_dir: Path
) -> bool:
    """Save per-model code review result to file."""
    sanitized_model = sanitize_model_name(model)

    if success and issues is not None:
        result_path = phase_dir / f"{sanitized_model}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(issues, f, indent=2)
        print(f"[{model}] Saved {len(issues)} issues to: {result_path}")
        return True
    elif error:
        error_path = phase_dir / f"error_{sanitized_model}.log"
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error}\n")
        print(f"[{model}] Error logged to: {error_path}")
        return False
    return False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    # Get available models for help text
    available_models = get_all_model_specs()

    # Printed commands, so resolved from `__file__` and never a skill-dir
    # placeholder (see review_plan_orchestrator.parse_args for why).
    run = dispatcher_prefix()
    parser = argparse.ArgumentParser(
        description="Review implementation changes against a plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[build_parent_parser()],
        epilog=f"""
Available models (provider:model format):
  {', '.join(available_models[:10])}{'...' if len(available_models) > 10 else ''}

Model selection priority:
  1. --models flag       -> Use specified models
  2. --interactive flag  -> Force two-step interactive selection
  3. --quick flag        -> Use quick_models from providers.yaml (2 models)
  4. YAML defaults       -> Use configured defaults from providers.yaml
  5. Interactive         -> Two-step selection (fallback if no defaults)

Examples:
  # Use YAML defaults (no prompting)
  {run} review-code plans/my-plan.md

  # Specify models explicitly
  {run} review-code plans/my-plan.md --models cursor-agent:auto gemini:gemini-2.5-flash

  # Force interactive selection
  {run} review-code plans/my-plan.md --interactive
        """
    )

    parser.add_argument(
        '--plan-file',
        required=True,
        help='Path to the implementation plan markdown file'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='List of models in provider:model format (e.g., cursor-agent:auto gemini:gemini-2.5-flash). '
             'Bare model names use default provider from providers.yaml.'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Use quick_models from providers.yaml for lightweight reviews (2 models)'
    )

    parser.add_argument(
        '--base-ref',
        type=str,
        help='Git ref to compare against (default: from state file)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Override timeout per model in seconds (default: use per-provider timeout from providers.yaml)'
    )

    parser.add_argument(
        '--max-parallel',
        type=int,
        default=3,
        help='Maximum parallel model invocations (default: 3)'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip the validation step (faster but no false-positive filtering)'
    )

    parser.add_argument(
        '--validation-model',
        type=str,
        default='auto',
        help='Model to use for validation (default: auto)'
    )

    parser.add_argument(
        '--apply-fixes',
        action='store_true',
        help='Prepare fix tasks for Claude Code subagents to apply'
    )

    parser.add_argument(
        '--reaggregate',
        action='store_true',
        help='Re-aggregate existing model results (use after salvage operations complete)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Bypass the completed-phase and partial-completion guards and resume '
             'the phase (already-completed per-model results are kept; only missing '
             'models re-run). For a full re-run that discards existing results, also '
             'pass --rerun-all.'
    )

    parser.add_argument(
        '--rerun-all',
        action='store_true',
        help='Re-run every model from scratch, discarding any existing per-model '
             'result files (default: resume — skip models that already have results).'
    )

    parser.add_argument(
        '--report-style',
        choices=['pr', 'flat'],
        default='pr',
        help='HTML report template style: "pr" for PR-style contextual view, "flat" for classic card layout (default: pr)'
    )

    # Opt-in CI gate. A completed review is exit 0 whatever it found (see the
    # summary block in main()); this is for a standalone caller who wants HIGH
    # findings to stop a pipeline. Exit 3 — "needs a human" — never 1, because
    # the review did not error. `full` never passes it: the phase whose whole job
    # is to fix what the review found runs next.
    parser.add_argument(
        '--fail-on-high',
        action='store_true',
        help='Exit 3 ("needs a human") when the review finds HIGH-importance '
             'issues, instead of the default 0. For standalone CI gating; the '
             '[HIGH_PRIORITY_ISSUES] <n> marker is printed either way.'
    )

    return parser.parse_args(argv)


def load_code_review_prompt() -> str:
    """Load the code review prompt template."""
    return load_prompt("code_review.txt")


def format_tracked_files_for_prompt(tracked_files: List[Dict], max_chars: int = 5000) -> str:
    """
    Format tracked files list for the code review prompt.

    Args:
        tracked_files: List of tracked file entries from state
        max_chars: Maximum character length before truncation (default 5000)

    Returns:
        Formatted string for inclusion in the prompt
    """
    if not tracked_files:
        return "(No tracked files - using git diff to discover changes)"

    lines = []
    for entry in tracked_files:
        path = entry.get("path", "")
        task_id = entry.get("task_id", "unknown")
        lines.append(f"- `{path}` (from task: {task_id})")

    full_text = "\n".join(lines)

    if len(full_text) > max_chars:
        # Truncate to fit within threshold
        truncated_lines = []
        current_length = 0
        for line in lines:
            # +1 for the newline separator
            if current_length + len(line) + 1 > max_chars:
                break
            truncated_lines.append(line)
            current_length += len(line) + 1

        omitted = len(lines) - len(truncated_lines)
        notice = f"\n\n[... {omitted} more files not shown — list truncated to stay within context limits ...]"
        print(f"WARNING: File list truncated — {omitted} of {len(lines)} files omitted to stay within context limits")
        return "\n".join(truncated_lines) + notice

    return full_text


def prepare_fixes_for_subagents(
    validated_groups: List[Dict],
    plan_path: Path,
    out_dir: Path
) -> Optional[Path]:
    """
    Prepare validated issues for Claude Code subagents to fix.

    This outputs a JSON file with fix tasks that Claude will use to spawn
    Task subagents for each fix.

    Args:
        validated_groups: List of validated issue groups
        plan_path: Path to the plan file
        out_dir: Output directory

    Returns:
        Path to the fix tasks JSON file
    """
    # Filter to only valid issues
    valid_issues = [
        g for g in validated_groups
        if g.get("validation_status") == "valid"
    ]

    if not valid_issues:
        print("No valid issues to fix")
        return None

    # Build fix tasks for each valid issue
    fix_tasks = []
    for i, issue in enumerate(valid_issues, 1):
        # Build location hints from issue data with safe access
        location_hints = []
        if issue.get("file"):
            location_hints.append(f"File: {issue['file']}")

        line_range = issue.get("line_range")
        if line_range and isinstance(line_range, (list, tuple)) and len(line_range) >= 2:
            location_hints.append(f"Lines: {line_range[0]}-{line_range[1]}")

        if issue.get("anchor_text"):
            location_hints.append(f"Anchor text: {issue['anchor_text']}")

        # For grouped issues, include suggestions from all models (max 3)
        suggestions = issue.get("suggestions", [])
        if isinstance(suggestions, list):
            for sugg in suggestions[:3]:
                if isinstance(sugg, dict) and sugg.get("reference"):
                    location_hints.append(f"Reference: {sugg['reference']}")

        # Get issue type for subagent routing
        issue_type = issue.get("type", "unknown")

        fix_task = {
            "id": f"FIX-{i:03d}",
            "title": issue.get("title") or issue.get("theme", "Unknown issue"),
            "type": issue_type,
            "importance": issue.get("importance", "medium"),
            "description": issue.get("desc") or issue.get("description", "No description provided"),
            "location_hints": location_hints,
            "plan_path": str(plan_path.absolute()),
            # D2h: executor_role is the host-neutral routing field every reader
            # branches on; subagent_type rides along as optional Claude-only
            # metadata for hosts whose native Task tool requires the parameter.
            "executor_role": "agent",
            "subagent_type": "general-purpose",
            "status": "pending"
        }
        fix_tasks.append(fix_task)

    # Save fix tasks to JSON
    phase_dir = get_phase_dir(plan_path, 'code-review')
    fix_tasks_path = phase_dir / "fix_tasks.json"
    try:
        with open(fix_tasks_path, "w", encoding="utf-8") as f:
            json.dump({
                "format_version": CURRENT_FORMAT_VERSION,
                "plan_path": str(plan_path.absolute()),
                "total_issues": len(fix_tasks),
                "tasks": fix_tasks
            }, f, indent=2)
        print(f"\nPrepared {len(fix_tasks)} fix tasks for subagents: {fix_tasks_path}")
        return fix_tasks_path
    except (IOError, OSError) as e:
        print(f"ERROR: Failed to write fix tasks file: {e}")
        return None


def update_plan_with_fixes(
    plan_path: Path,
    fix_results: List[Dict]
) -> None:
    """
    Update the plan file to document what was fixed during review.

    Args:
        plan_path: Path to the plan file
        fix_results: List of fix results
    """
    if not fix_results:
        return

    # Validate input
    if not isinstance(fix_results, list):
        print(f"ERROR: fix_results must be a list, got {type(fix_results)}")
        return

    try:
        updater = PlanUpdater(plan_path)
    except Exception as e:
        print(f"ERROR: Failed to load plan file for update: {e}")
        return

    # Build the fixes section content
    fixed_count = sum(1 for r in fix_results if r.get("fixed"))
    failed_count = len(fix_results) - fixed_count

    lines = [
        "",
        "<!-- REVIEW_FIXES_START -->",
        "",
        "## Review Fixes Applied",
        "",
        f"*Applied: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"**Summary:** {fixed_count} fixed, {failed_count} could not be fixed",
        "",
    ]

    # List fixed issues
    fixed_items = [r for r in fix_results if r.get("fixed")]
    if fixed_items:
        lines.append("### Successfully Fixed")
        lines.append("")
        for item in fixed_items:
            lines.append(f"- **{item.get('issue_title', 'Unknown')}**")
            if item.get("changes_made"):
                lines.append(f"  - {item['changes_made']}")
            if item.get("files_modified"):
                for file_mod in item["files_modified"]:
                    path = file_mod.get("path", "unknown")
                    summary = file_mod.get("summary", "")
                    if summary:
                        lines.append(f"  - `{path}`: {summary}")
                    else:
                        lines.append(f"  - `{path}`")
        lines.append("")

    # List failed fixes
    failed_items = [r for r in fix_results if not r.get("fixed")]
    if failed_items:
        lines.append("### Could Not Fix")
        lines.append("")
        for item in failed_items:
            reason = (item.get("reason") or "Unknown reason")[:100]
            lines.append(f"- **{item.get('issue_title', 'Unknown')}**: {reason}")
        lines.append("")

    lines.append("<!-- REVIEW_FIXES_END -->")
    lines.append("")

    # Check if section already exists and replace, or append
    try:
        content = updater.get_content()
        start_marker = "<!-- REVIEW_FIXES_START -->"
        end_marker = "<!-- REVIEW_FIXES_END -->"

        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)

        if start_idx != -1 and end_idx != -1:
            # Replace existing section
            parts = [
                content[:start_idx],
                "\n".join(lines),
                content[end_idx + len(end_marker):]
            ]
            updater.content = "".join(parts)
        else:
            # Append new section at the end
            updater.content = content.rstrip() + "\n" + "\n".join(lines)

        updater.save()
        print(f"Updated plan with fix results: {plan_path}")
    except Exception as e:
        print(f"ERROR: Failed to update plan file: {e}")


def emit_salvage_task(
    salvage_request: Dict,
    salvage_path: Path,
    phase_dir: Path,
    plan_path: Path,
) -> None:
    """Materialize the salvage task beside a just-written salvage request.

    The prompt is assembled in Python (D2/D2i) rather than by a Claude reading
    ``references/salvage-handling.md``, so any invoker can run it. Best-effort:
    the parse error that produced the request must not be masked by a task-file
    problem, and a request with no task file is still recoverable by hand.
    """
    try:
        task_path = write_salvage_task_file(
            salvage_request=salvage_request,
            salvage_request_path=str(salvage_path),
            phase_dir=str(phase_dir),
            plan_file=str(plan_path),
        )
        print(f"[SALVAGE_TASK] {task_path}")
    except Exception as task_error:  # noqa: BLE001 - diagnostics only
        print(f"Warning: could not materialize salvage task: {task_error}")


async def run_single_review(
    model_spec: str,
    plan_path: Path,
    out_dir: Path,
    timeout: Optional[int] = None,
    tracked_files: Optional[List[Dict]] = None,
    base_ref: str = "HEAD~1",
    served_models: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
    """
    Run a single model code review using agent exploration with file-based output.

    Uses file-based JSON output where the LLM writes results to a specified file
    instead of stdout. Falls back to stdout parsing if file output fails.

    The agent will:
    1. Read the plan file to understand requirements
    2. Identify changed files from the tracked files list (or discover via git)
    3. Use git diff with base_ref to examine actual changes
    4. Write findings as JSON to the specified output file

    Args:
        model_spec: Model specification (e.g., 'cursor-agent:auto', 'gemini:gemini-2.5-flash')
        plan_path: Path to the implementation plan
        out_dir: Directory for output files
        timeout: Optional timeout override in seconds. Uses provider default if not specified.
        tracked_files: Optional list of tracked files from implementation phase
        base_ref: Git reference to diff against (commit at implementation start)
        served_models: Optional collector for §3.6's attribution.  When a
            configured ``--fallback-model`` list produced a served model, this
            slot's ``{model_spec: {"served_model", "fallback"}}`` entry is
            written into it, so the report can name the model that really
            answered.  The return tuple is deliberately unchanged — every caller
            of it destructures three values.

    Returns:
        Tuple of (success, issues_list, error)
    """
    import time
    start_time = time.time()

    # Detect project root so CLI providers (e.g. Claude Code) run in the
    # correct working directory and can access project files.
    project_root = get_project_root(str(plan_path))

    # Parse model spec to get provider info for display. Anchored on the plan
    # (§9.3.a): the value already exists here, it just used to be dropped.
    provider_name, model_name = parse_model_spec(model_spec, anchor=str(plan_path))
    display_name = f"{provider_name}:{model_name}"

    # D6 step 3, the gate `run_single_model` applies to the other review lanes,
    # asked here for the same reason and off the same once-per-run cache: this
    # slot is a `generate` dispatch, and a runner whose own sandbox cannot start
    # reads neither the plan nor the diff — it answers with a sentence about not
    # being able to open them, which scores as a completed review. Refused
    # before the phase dir is claimed and before anything an earlier attempt
    # left is retired, because a refused slot replaces nothing.
    refusal = await asyncio.to_thread(sandbox_refusal, provider_name, project_root)
    if refusal is not None:
        print(f"[{display_name}] {refusal}")
        return False, None, refusal

    # Generate log file path for debugging
    sanitized_model = sanitize_model_name(model_spec)
    phase_dir = get_phase_dir(plan_path, 'code-review')
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_file = phase_dir / f"log_{sanitized_model}.txt"

    # This model's attempt starts *here*, so any salvage pair an earlier attempt
    # left in the phase dir stops being outstanding work *here* — before the
    # provider is invoked, and whatever this attempt goes on to do.
    #
    # `get_phase_dir` is stable per plan (`utils/output_handler.py`), so every
    # run for this plan writes into the same directory, and `--force` clears
    # only the state file's phase-completion marker (see main()) — it deletes
    # nothing from disk. Without this call the exact P2 rerun still reproduces
    # P2: kilocode dies on `User not found.`, `has_salvageable_json` correctly
    # writes nothing, and the *previous* run's `salvage_task_<stem>.json` is
    # still dispatched by `dispatch_salvage_tasks`'s glob and still advertised
    # by main()'s end-of-run `salvage_*.json` scan. A salvage task's
    # `output_name` IS this model's result file, so what that repairs is a run
    # that no longer exists, over a slot this run recorded as failed — the
    # phantom participant the gate exists to prevent, arriving by the one route
    # the gate cannot see. Confirmed on disk: a real `code-review` phase dir
    # holds request/task pairs from three separate runs that are long over.
    #
    # **Retire-by-rename rather than run-stamping**, deliberately, and the
    # resumed run is what decides it. Retiring belongs to *the attempt that
    # would replace the artifact*, and this function is only ever reached by a
    # model that is actually re-running: `run_all_reviews` answers a resume
    # (`skip_existing`, i.e. no `--rerun-all`) from the existing
    # `<model>.json` and returns before calling here, so a skipped model keeps
    # its pair and a re-run model loses one that can only describe an answer it
    # is about to replace. That is the whole of the shared-phase-dir case a run
    # stamp was wanted for, without a run id to thread through two writers, two
    # globs and two reporting blocks — and without the question a stamp cannot
    # answer, which is what to do with the unstamped pairs already on disk.
    # What this shape does *not* cover is a model dropped from the selection
    # between runs: it begins no attempt, so nothing retires its pair. That is
    # the same residual as its stale `<model>.json`, which the fan-out also
    # leaves for reaggregation to pick up (see the `--rerun-all` note at the
    # salvage dispatch in main()), and it belongs to whatever eventually scopes
    # the phase dir by run rather than to the gate.
    #
    # Quiet: the fan-out already prints a line per model per phase, and one
    # `[SALVAGE_RETIRED]` per re-run model would bury them. The audit trail is
    # the renamed file, which is not deleted.
    retire_salvage_artifacts(
        str(phase_dir), sanitized_model, reason=RETIRE_SUPERSEDED, quiet=True
    )

    # Format tracked files for the prompt
    tracked_files_list = format_tracked_files_for_prompt(tracked_files or [])

    # Load prompt template
    prompt_template = load_code_review_prompt()

    # Derive prefix from plan path (same as in main())
    prefix = derive_prefix(plan_path)

    # Build prompt context for invoke_with_file_output
    prompt_context = {
        "plan_path": str(plan_path.absolute()),
        "tracked_files_list": tracked_files_list,
        "base_ref": base_ref,
        "prefix": prefix,
    }

    # Determine timeout: use provided override, or get provider-specific default
    effective_timeout = (
        timeout if timeout is not None
        else get_provider_timeout(provider_name, anchor=str(plan_path))
    )

    print(f"[{display_name}] Starting code review with file-based output (timeout: {effective_timeout}s)...")

    # Use asyncio.to_thread to run the synchronous invoke_with_file_output
    # in a thread pool, allowing concurrent execution
    result = await asyncio.to_thread(
        invoke_with_file_output,
        prompt_template=prompt_template,
        model_spec=model_spec,
        prompt_context=prompt_context,
        output_dir=out_dir,
        phase="code_review",
        timeout=int(effective_timeout),
        log_file=str(log_file),
        prefer_arrays=True,
        cwd=project_root,
        anchor=str(plan_path),
    )

    duration = time.time() - start_time
    output_file = result.get("output_file", "")
    source = result.get("source", "unknown")

    # §3.6: record which model served this slot before any of the branches
    # below return, so a fallback that fired is attributed even on the paths
    # that end in salvage. Only ever populated when a fallback is configured.
    details = result.get("details", {}) or {}
    if served_models is not None and details.get("served_model"):
        served_models[model_spec] = {
            "served_model": details["served_model"],
            "fallback": bool(details.get("served_model_is_fallback")),
        }

    print(f"[{display_name}] Log saved to: {log_file}")
    if output_file:
        print(f"[{display_name}] {describe_output_file(output_file, source)}")

    if not result["success"]:
        error_msg = result.get("error", "Unknown error")
        file_error = result.get("file_error", "")
        if file_error:
            print(f"[{display_name}] File error: {file_error}")
        print(f"[{display_name}] Agent failed ({duration:.1f}s): {error_msg}")

        # Write a salvage request only when the failure output still holds an
        # answer to repair. "stderr is non-empty" is not that test: a hard
        # provider failure — kilocode answering `Error: User not found.` on a
        # dead account, in ANSI colour — passes it with no JSON anywhere in the
        # stream, and the salvage prompt's own "if nothing recoverable is
        # present, print an empty array" then writes `[]` over this model's
        # result path, so a model that never ran is aggregated as one that
        # reviewed the diff and found nothing. It also spends the operator's
        # next move on a JSON repair when the fix is `kilocode` auth. With
        # nothing to salvage the model simply stays failed, which is the D6
        # row-3 partial-results path already taken by every other dead slot.
        # `has_salvageable_json` subsumes the old condition: a `stdout_error`
        # with empty stderr only ever produced a salvage request whose
        # `raw_output` was the empty string.
        #
        # WHICH CHANNEL THE ANSWER IS ON, and this branch used to look at only
        # one of them. `if not result["success"]` is not merely "the provider
        # never spoke": it also covers the provider that wrote a truncated or
        # fenced array and *then* exited non-zero — a kill at the deadline, a
        # crash on teardown — and that array is on **stdout**, never on stderr.
        # Gating on stderr alone therefore printed "Provider failed before
        # producing any JSON — nothing to salvage" about a run that had produced
        # JSON, and threw a recoverable review away without a word. The
        # JSON-erroring providers press the same point from the other side:
        # `opencode run --format json` puts its failure shape on stdout with
        # stderr empty, and a relayed `{"error": {"code": 401, ...}}` body is
        # indistinguishable from any HTTP-backed CLI — those reached the
        # predicate as `""`, so even the verdicts it reached correctly were
        # reached for the wrong reason. The pre-gate code was equally blind (its
        # `raw_output` was the empty string), so this closes a latent hole rather
        # than a regression; it is closed here because site A is the one place
        # the gate above could make a real loss *visible*.
        #
        # **Precedence, not concatenation.** The first non-empty channel wins and
        # the other is not appended, because `raw_output` is handed to a subagent
        # as *the* damaged answer to repair (`build_salvage_prompt` in
        # `utils/subagent_prompts.py`), and gluing a launch banner or an auth
        # notice onto a truncated array gives that subagent two documents and no
        # way to tell which one it was asked about — with the same empty-array
        # fallback waiting when it guesses wrong. stdout is the one that can hold
        # an answer, so stdout is the one that is asked first.
        #
        # `utils/llm_client.py`'s `_attach_stdout_excerpt` (:413) is what makes
        # the falsy test below sufficient: it attaches `stdout_excerpt` on all
        # three losing paths — non-zero exit, timeout kill, and the zero exit
        # whose output would not parse — already `_redact`ed and `_bound_stream`ed
        # in that order (§4.5/§4.4), and it **omits the key entirely** when
        # nothing survived rather than setting `""`. So absent and empty mean the
        # same thing here, no second emptiness check is owed, and stderr still
        # answers for the provider that died before printing a byte.
        #
        # Whatever this picks is also what gets persisted: the gate below judges
        # `raw_content` and the salvage request stores `raw_content`, so the text
        # a subagent is asked to repair is the text the decision was made on.
        # Gating on one string and writing another would let a request be written
        # for evidence nobody ever saw.
        raw_content = ""
        details = result.get("details", {}) or {}
        if details.get("stdout_excerpt"):
            raw_content = details["stdout_excerpt"]
        elif details.get("stderr"):
            raw_content = details["stderr"]

        if has_salvageable_json(raw_content):
            salvage_request = {
                "model": model_spec,
                "phase": "code_review",
                "raw_output": raw_content,
                "expected_type": "array",
                "expected_schema": {
                    "fields": ["title", "desc", "importance", "file", "line_range", "type", "anchor_text"],
                    "importance_values": ["high", "medium", "low"],
                    "type_values": ["bug", "missing", "improvement", "style", "scope"]
                },
                "output_path": str(phase_dir / f"{sanitized_model}.json"),
                "output_json_path": output_file,
                "timestamp": datetime.now().isoformat()
            }
            salvage_path = phase_dir / f"salvage_{sanitized_model}.json"
            with open(salvage_path, 'w', encoding='utf-8') as f:
                json.dump(salvage_request, f, indent=2)
            print(f"[SALVAGE_NEEDED] {salvage_path}")
            emit_salvage_task(salvage_request, salvage_path, phase_dir, plan_path)
        else:
            # Say which failure this was, in the same line-per-model channel the
            # markers use. Silence here would read as "salvage was not reached",
            # when what happened is that the provider failed before it produced
            # an answer — an auth, quota or launch failure whose fix is the
            # provider, not a JSON repair. The stream itself is in the log file
            # named above; the one-line summary belongs to the adapter's
            # `describe_failure` hook, which is why it is not re-printed here.
            print(
                f"[{display_name}] Provider failed before producing any JSON — "
                f"nothing to salvage, model recorded as failed"
            )

        return False, None, error_msg

    print(f"[{display_name}] Completed in {duration:.1f}s (source: {source})")

    # Get the parsed data from result
    output_data = result.get("data")

    if isinstance(output_data, list):
        # Validate issues format
        is_valid, errors = validate_code_review_issues(output_data)
        if not is_valid:
            print(f"[{display_name}] Warning: Validation errors: {errors}")
        print(f"[{display_name}] Found {len(output_data)} issues")
        return True, output_data, None
    elif isinstance(output_data, dict) and "raw" in output_data:
        # No valid JSON found - write salvage request for Claude Code to process
        raw_content = output_data["raw"]
        raw_preview = raw_content[:500] if raw_content else "(empty)"

        # Same rule as the hard-failure path above, for the same reason: a reply
        # with no JSON in it is not a parse failure, and salvage against it can
        # only print the empty array its prompt falls back to. This branch is
        # where §40's two write-refusal shapes land — cursor-agent's "I can't
        # write the requested file while Ask mode is active" and gemini's
        # "JSON written to …" confirmation for a write that never happened —
        # and a `[]` recorded for either would be an invented clean review.
        if not has_salvageable_json(raw_content):
            print(f"[{display_name}] Response carries no JSON — nothing to salvage")
            return False, None, f"Model produced no JSON output: {raw_preview}"

        salvage_request = {
            "model": model_spec,
            "phase": "code_review",
            "raw_output": raw_content,
            "expected_type": "array",
            "expected_schema": {
                "fields": ["title", "desc", "importance", "file", "line_range", "type", "anchor_text"],
                "importance_values": ["high", "medium", "low"],
                "type_values": ["bug", "missing", "improvement", "style", "scope"]
            },
            "output_path": str(phase_dir / f"{sanitized_model}.json"),
            "output_json_path": output_file,
            "timestamp": datetime.now().isoformat()
        }
        salvage_path = phase_dir / f"salvage_{sanitized_model}.json"
        with open(salvage_path, 'w', encoding='utf-8') as f:
            json.dump(salvage_request, f, indent=2)

        print(f"[SALVAGE_NEEDED] {salvage_path}")
        emit_salvage_task(salvage_request, salvage_path, phase_dir, plan_path)
        return False, None, f"Could not extract JSON from response: {raw_preview}"
    else:
        return False, None, f"Expected JSON array, got {type(output_data).__name__}"


# Delay between launching provider processes to avoid concurrent launch bugs
# Especially important for cursor-agent which has known issues with concurrent starts
PROVIDER_STAGGER_DELAY = 2.0  # seconds


async def run_all_reviews(
    model_specs: List[str],
    plan_path: Path,
    out_dir: Path,
    timeout: Optional[int],
    max_parallel: int,
    tracked_files: Optional[List[Dict]] = None,
    base_ref: str = "HEAD~1",
    skip_existing: bool = True,
    served_models: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Tuple[bool, Optional[List[Dict]], Optional[str]]]:
    """
    Run code review across all models using agent exploration.

    Supports multiple providers - each model spec can specify a different
    provider (e.g., 'cursor-agent:auto', 'gemini:gemini-2.5-flash').

    Each agent:
    - Receives the list of tracked files from implementation (if available)
    - Receives the base_ref to use for git diff commands
    - Reads the plan to understand requirements
    - Uses git diff to examine actual changes
    - Writes findings to a JSON file

    Args:
        model_specs: List of model specifications in provider:model format
        plan_path: Path to the implementation plan
        out_dir: Output directory for results
        timeout: Optional timeout override per model in seconds. If None, uses per-provider defaults.
        max_parallel: Maximum concurrent model invocations
        tracked_files: Optional list of tracked files from implementation phase
        base_ref: Git reference to diff against
        served_models: Optional §3.6 attribution collector, forwarded to each
            :func:`run_single_review` (see its docstring)

    Returns:
        Dict mapping model spec to tuple of (success, issues_list, error)
    """
    semaphore = asyncio.Semaphore(max_parallel)
    # Per-provider semaphores for concurrency limiting
    provider_semaphores: Dict[str, asyncio.Semaphore] = {}
    for spec in model_specs:
        prov, _ = parse_model_spec(spec, anchor=str(plan_path))
        if prov not in provider_semaphores:
            limit = get_provider_max_concurrent(prov, anchor=str(plan_path))
            if limit is not None:
                provider_semaphores[prov] = asyncio.Semaphore(limit)
    results: Dict[str, Tuple[bool, Optional[List[Dict]], Optional[str]]] = {}

    async def run_with_semaphore(model_spec: str, index: int) -> Tuple[str, Tuple[bool, Optional[List[Dict]], Optional[str]]]:
        provider_name, model_name = parse_model_spec(model_spec, anchor=str(plan_path))
        display_name = f"{provider_name}:{model_name}"

        # Skip models that already have results (resume after partial failure)
        if skip_existing:
            sanitized = sanitize_model_name(model_spec)
            phase_dir = get_phase_dir(plan_path, 'code-review')
            existing_path = phase_dir / f"{sanitized}.json"
            if existing_path.exists() and existing_path.stat().st_size > 0:
                try:
                    with open(existing_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        print(f"[SKIP] {display_name} - already has results ({len(existing_data)} issues)")
                        return (model_spec, (True, existing_data, None))
                except (json.JSONDecodeError, IOError):
                    print(f"[{display_name}] Existing result file corrupt, re-running...")

        # Stagger launches to avoid concurrent start bugs (especially with cursor-agent)
        # First model starts immediately, subsequent models wait
        if index > 0:
            stagger_delay = index * PROVIDER_STAGGER_DELAY
            print(f"[{display_name}] Waiting {stagger_delay:.1f}s before starting (staggered launch)...")
            await asyncio.sleep(stagger_delay)

        # Acquire provider semaphore first (if any), then global semaphore.
        # This order prevents capped providers from hogging global slots while
        # waiting for their provider-specific turn.
        prov_sem = provider_semaphores.get(provider_name)
        if prov_sem:
            async with prov_sem:
                async with semaphore:
                    result = await run_single_review(
                        model_spec, plan_path, out_dir, timeout, tracked_files,
                        base_ref, served_models=served_models,
                    )
                    return (model_spec, result)
        else:
            async with semaphore:
                result = await run_single_review(
                    model_spec, plan_path, out_dir, timeout, tracked_files,
                    base_ref, served_models=served_models,
                )
                return (model_spec, result)

    # Create tasks for all models with staggered starts
    tasks = [run_with_semaphore(spec, idx) for idx, spec in enumerate(model_specs)]

    # Calculate total timeout based on longest provider timeout
    if timeout is not None:
        base_timeout = timeout
    else:
        # Use maximum of all provider timeouts
        base_timeout = max(
            get_provider_timeout(
                parse_model_spec(spec, anchor=str(plan_path))[0], anchor=str(plan_path)
            )
            for spec in model_specs
        )

    stagger_overhead = len(model_specs) * PROVIDER_STAGGER_DELAY
    total_timeout = base_timeout * 2 + stagger_overhead

    try:
        completed = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=total_timeout
        )

        for item in completed:
            if isinstance(item, Exception):
                print(f"Review exception: {item}")
                continue
            model_spec, result = item
            results[model_spec] = result

    except asyncio.TimeoutError:
        print(f"Total timeout ({total_timeout}s) exceeded for all models")
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()

    return results


def generate_review_report(
    plan_path: Path,
    results: Dict[str, Tuple[bool, Optional[List[Dict]], Optional[str]]],
    changed_files: List[str],
    validated_groups: Optional[List[Dict]] = None,
    phase_dir: Optional[Path] = None,
    base_ref: Optional[str] = None,
    template_style: str = 'pr',
    served_models: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Generate a markdown review report.

    ``served_models`` is §3.6's attribution
    (``{model_spec: {"served_model", "fallback"}}``): every row that names a
    model — the ``**Models:**`` summary, each issue's ``**Model:**`` field and
    the HTML report's badges — names the model that actually served the turn
    instead of silently crediting the primary. Empty (the shipped default, since
    ``fallback_models`` ships empty) leaves the report byte-for-byte what it was.
    """
    lines = [
        "# Code Review Report",
        "",
        f"**Plan:** {plan_path.name}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Files Changed:** {len(changed_files)}",
        "",
    ]

    # Collect all issues
    all_issues = []
    successful_models = []
    failed_models = {}

    for model, (success, issues, error) in results.items():
        if success and issues:
            successful_models.append(model)
            for issue in issues:
                issue["model"] = model
                all_issues.append(issue)
        else:
            failed_models[model] = error or "No issues returned"

    lines.append(
        "**Models:** "
        + ", ".join(served_model_label(m, served_models) for m in successful_models)
    )
    lines.append(f"**Total Issues:** {len(all_issues)}")

    # Add validation summary if available
    if validated_groups:
        valid_count = sum(1 for g in validated_groups if g.get("validation_status") == "valid")
        invalid_count = sum(1 for g in validated_groups if g.get("validation_status") == "invalid")
        needs_human = sum(1 for g in validated_groups if g.get("validation_status") == "needs-human-decision")
        lines.append(f"**Validation:** {valid_count} valid, {invalid_count} invalid, {needs_human} needs human review")
        lines.append("")
    else:
        lines.append("")

    lines.append("> **Note:** An interactive HTML report is also available at `report.html`.")
    lines.append("> If you export selections from the HTML report (`user_selections.json`),")
    lines.append("> those selections will take precedence over checkboxes in this file.")
    lines.append("")

    # Build validation info lookup from validated_groups
    # Map (title, file) -> {status, reason} for correlation
    validation_lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    if validated_groups:
        for group in validated_groups:
            status = group.get("validation_status")
            reason = group.get("validation_reason", "")
            if status:
                info = {"status": status, "reason": reason}
                # Try to match by theme/title and file
                theme = group.get("theme", "")
                title = group.get("title", "")
                file_path = group.get("file", "")

                # Also look in suggestions for individual issue data
                suggestions = group.get("suggestions", [])
                for sugg in suggestions:
                    if isinstance(sugg, dict):
                        sugg_title = sugg.get("title", title or theme)
                        sugg_file = sugg.get("file", file_path)
                        if sugg_title:
                            validation_lookup[(sugg_title, sugg_file)] = info

                # Also store by theme/title directly
                if theme:
                    validation_lookup[(theme, file_path)] = info
                if title:
                    validation_lookup[(title, file_path)] = info

    def get_validation_info(issue: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Look up validation status and reason for an issue."""
        title = issue.get("title", "")
        file_path = issue.get("file", "")
        # Try exact match first
        if (title, file_path) in validation_lookup:
            info = validation_lookup[(title, file_path)]
            return info.get("status"), info.get("reason")
        # Try with empty file
        if (title, "") in validation_lookup:
            info = validation_lookup[(title, "")]
            return info.get("status"), info.get("reason")
        return None, None

    # Build issue-to-global-index mapping (matches grouped.json order)
    issue_global_index = {}
    for idx, issue in enumerate(all_issues, 1):
        issue_global_index[id(issue)] = idx

    # Group by importance
    high = [i for i in all_issues if i.get("importance") == "high"]
    medium = [i for i in all_issues if i.get("importance") == "medium"]
    low = [i for i in all_issues if i.get("importance") == "low"]

    def _issue_sort_key(issue):
        status, _ = get_validation_info(issue)
        return (
            VALIDATION_ORDER.get(status or "", UNKNOWN_STATUS_RANK),
            issue_global_index.get(id(issue), 0),
        )

    high.sort(key=_issue_sort_key)
    medium.sort(key=_issue_sort_key)
    low.sort(key=_issue_sort_key)

    # HIGH issues
    lines.append("## HIGH Priority")
    lines.append("")
    if high:
        for issue in high:
            status, reason = get_validation_info(issue)
            lines.extend(_format_issue(issue_global_index[id(issue)], issue, validation_status=status, validation_reason=reason, served_models=served_models))
    else:
        lines.append("_No high priority issues._")
        lines.append("")

    # MEDIUM issues
    lines.append("## MEDIUM Priority")
    lines.append("")
    if medium:
        for issue in medium:
            status, reason = get_validation_info(issue)
            lines.extend(_format_issue(issue_global_index[id(issue)], issue, validation_status=status, validation_reason=reason, served_models=served_models))
    else:
        lines.append("_No medium priority issues._")
        lines.append("")

    # LOW issues
    lines.append("## LOW Priority")
    lines.append("")
    if low:
        for issue in low:
            status, reason = get_validation_info(issue)
            lines.extend(_format_issue(issue_global_index[id(issue)], issue, validation_status=status, validation_reason=reason, served_models=served_models))
    else:
        lines.append("_No low priority issues._")
        lines.append("")

    # Changed files list
    lines.append("## Changed Files")
    lines.append("")
    for f in changed_files:
        lines.append(f"- `{f}`")
    lines.append("")

    # Failed models
    if failed_models:
        lines.append("## Failed Reviews")
        lines.append("")
        for model, error in failed_models.items():
            lines.append(f"- **{model}**: {error}")
        lines.append("")

    # Add section for items needing human review
    if validated_groups:
        needs_human = [g for g in validated_groups if g.get("validation_status") == "needs-human-decision"]
        if needs_human:
            lines.append("## Needs Human Review")
            lines.append("")
            lines.append("The following issues require human judgment:")
            lines.append("")
            for g in needs_human:
                theme = g.get("theme", "Unknown issue")
                reason = g.get("validation_reason", "No reason provided")
                lines.append(f"- **{theme}**: {reason}")
            lines.append("")

    # Generate HTML report if phase_dir provided
    if phase_dir:
        try:
            html_content = generate_html_report(
                groups=validated_groups or [],
                plan_path=plan_path,
                phase_dir=phase_dir,
                phase_type="code-review",
                models=successful_models,
                failed_models=failed_models,
                base_ref=base_ref,
                template_style=template_style,
                served_models=served_models,
            )
            write_html_report(html_content, phase_dir)
            print(f"HTML report written to: {phase_dir / 'report.html'}")
        except Exception as e:
            print(f"Warning: Failed to generate HTML report: {e}")

    return "\n".join(lines)


def _format_issue(
    index: int,
    issue: Dict,
    validation_status: Optional[str] = None,
    validation_reason: Optional[str] = None,
    served_models: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """Format a single issue for the report.

    Args:
        index: The issue number for display
        issue: The issue dict with title, desc, type, file, model, etc.
        validation_status: The validation status (valid, invalid, needs-human-decision, etc.)
        validation_reason: The reason for the validation decision (shown for invalid/needs-human-decision)
        served_models: Optional §3.6 attribution.  The ``**Model:**`` field is
            the one place this report credits a finding to a model, so a slot
            served by a configured fallback is named here too — otherwise the
            row reads as if the primary produced it.
    """
    file_ref = issue.get("file", "unknown")
    line_range = issue.get("line_range")
    if line_range:
        file_ref += f":{line_range[0]}-{line_range[1]}"

    # Build explicit validation status string
    validation_display = {
        "valid": "Valid",
        "invalid": "Invalid",
        "needs-human-decision": "? Needs Review",
        "validation_failed": "? Validation Failed",
    }
    validation_str = validation_display.get(validation_status, "? Unknown") if validation_status else "? Unknown"

    lines = [
        f"### {index}. {issue.get('title', 'Untitled')}",
        f"- [ ] Skip",  # Skip checkbox
    ]

    # Add override checkboxes for items needing human review
    if validation_status in ("needs-human-decision", "validation_failed"):
        lines.append("- [ ] Mark valid")
        lines.append("- [ ] Mark invalid")
    # "Let Claude decide" routes this issue to the per-item judge at apply time.
    # Scoped to needs-human-decision only.
    if validation_status == "needs-human-decision":
        lines.append("- [ ] Let Claude decide")

    lines.extend([
        f"**Validation:** {validation_str} | **File:** `{file_ref}` | **Type:** {issue.get('type', 'unknown')} | **Model:** {served_model_label(issue.get('model', 'unknown'), served_models)}",
        "",
    ])

    # Include validation reason for invalid or needs-human-decision issues
    if validation_reason and validation_status in ("invalid", "needs-human-decision", "validation_failed"):
        lines.append(f"> **Validation Reason:** {validation_reason}")
        lines.append("")

    lines.extend([
        issue.get("desc", "No description"),
        "",
        "---",
        "",
    ])
    return lines


async def reaggregate_from_existing_files(
    prefix: str,
    out_dir: Path,
    plan_path: Path,
    args: argparse.Namespace,
    template_style: str = 'pr',
    base_ref: Optional[str] = None,
) -> None:
    """Re-aggregate all model results after salvage operations complete.

    This function scans the output directory for model result files (including
    salvaged ones) and re-runs grouping, validation, and report generation.

    Args:
        prefix: Output file prefix
        out_dir: Output directory containing model results
        plan_path: Path to the plan file
        args: Command-line arguments for validation settings
        template_style: Template to use — ``'pr'`` or ``'flat'``.
            Defaults to ``'pr'``.
        base_ref: Optional explicit base git ref for diff context.
            When provided, takes priority over ``args.base_ref`` and
            the state's ``head_at_start``.
    """
    import glob as glob_module

    # Find all model result files in the code-review phase directory
    phase_dir = get_phase_dir(plan_path, 'code-review')
    pattern = str(phase_dir / "*.json")
    # Sort for a deterministic load order — glob() returns filesystem order,
    # which varies between machines/runs. Grouping itself is now order-
    # independent (see group_similar_suggestions), but a stable file order
    # keeps issues.json reproducible too.
    all_files = sorted(glob_module.glob(pattern))

    # Exclude non-model files
    # `subagent_batch_report` is executor bookkeeping (D3i), never a model
    # result — see the same note in review_orchestrator_base, which also explains
    # `subagent_dry_run` (D3l) and the write scheduler's two reports.
    exclude_patterns = ['grouped', 'validation', 'salvage', 'issues', 'fix_tasks', 'report_data', 'subagent_batch_report', 'subagent_dry_run', 'write_batch_report', 'write_tasks']
    result_files = [
        f for f in all_files
        if not any(ex in os.path.basename(f) for ex in exclude_patterns)
    ]

    if not result_files:
        print(f"No model result files found matching: {pattern}")
        return

    # Collect all issues and build synthetic results dict for report generation
    all_issues = []
    models_found = set()
    results: Dict[str, Tuple[bool, Optional[List[Dict]], Optional[str]]] = {}

    # Restore original model specs (with colons) from .status.json
    unsanitize = build_unsanitize_map(str(phase_dir))

    # An empty map is not "this dir has no models" — `result_files` is non-empty
    # or the function has already returned above. It is a phase dir whose
    # checkpoint records none, and from here on every group in the report is
    # attributed to a stem that no longer names a model. That outcome was
    # previously reached in complete silence, and on the `--skip-validation` leg
    # it still completes and still writes a report, so nothing downstream ever
    # contradicts it either.
    #
    # The condition is `build_unsanitize_map`'s *result*, deliberately, not
    # `.status.json`'s existence: the helper returns `{}` for a missing file, an
    # unreadable one, and one carrying neither `models_requested` nor
    # `models_completed` — and a hand-merged checkpoint holding `phase` and
    # `state` and nothing else inverts exactly as little as no file at all. Same
    # field-not-file rule as the fan-out's read-back below.
    if not unsanitize:
        warn_stem_attribution(
            f"WARNING: {phase_dir} holds {len(result_files)} model result "
            f"file(s), but its checkpoint records no models_requested — either "
            f"no .status.json at all, or one that never got the field. This "
            f"reaggregation continues."
        )

    # One file per logical model (W3): a phase dir that straddles the L1 anchor
    # upgrade holds both `<model>.json` and `<provider>_<model>.json` for the same
    # model, and reading both double-counts every issue. `results` matters as much
    # as the counts here — a duplicate entry renders the same model twice in the
    # report. The precedence (anchored wins, divergence WARNs, an ambiguous legacy
    # stem exits 3) lives in the shared helper.
    for model, _chosen_path, issues in resolve_result_files_by_model(
        str(phase_dir), result_files, unsanitize, noun="issues"
    ):
        models_found.add(model)

        for issue in issues:
            # `model` before `source_model`, which is the order the fan-out
            # stamps them in (`main` sets `model` as it flattens `results`, then
            # fills `source_model` in just before grouping). Insertion order is
            # not cosmetic here: `SuggestionGroup.to_dict` carries a suggestion's
            # dict through unreordered, `json.dump` writes keys in that order,
            # and `grouped_sha256` hashes the file's *bytes*. Stamped the other
            # way round, reaggregating a phase whose specs are already canonical
            # still moved grouped.json's bytes and still made
            # `check_task_file_inputs` refuse `validation_tasks.json` — the same
            # failure the missing checkpoint caused, from the other direction.
            issue["model"] = model            # report generation reads this one
            issue["source_model"] = model     # grouping attributes by this one
        all_issues.extend(issues)

        # Add to results dict for report generation
        results[model] = (True, issues, None)

        print(f"  Loaded {len(issues)} issues from {model}")

    print(f"\nRe-aggregating {len(all_issues)} issues from {len(models_found)} models")

    if not all_issues:
        print("No issues to aggregate.")
        return

    # Get changed files from state (or empty list if not available)
    state = get_or_create_state(plan_path)

    # Resolve base_ref for validation context.
    # Priority: explicit parameter > args.base_ref > state head_before_implement > head_at_start > fallback
    if base_ref is None:
        args_base_ref = getattr(args, 'base_ref', None)
        state_base_ref = state.get('head_before_implement') or state.get('head_at_start')
        if args_base_ref:
            base_ref = args_base_ref
        elif state_base_ref:
            base_ref = state_base_ref
        else:
            base_ref = 'HEAD~1'
            print(
                "Warning: head_at_start not found in state; falling back to HEAD~1. "
                "This may produce incorrect diffs if the branch has advanced since "
                "the original review.",
                file=sys.stderr,
            )
    base_ref = validate_git_ref(base_ref)

    tracked_files = state.get("tracked_files", [])
    if tracked_files:
        changed_files = list(set(entry.get("path", "") for entry in tracked_files if entry.get("path")))
    else:
        changed_files = []

    with intent_to_add_untracked(changed_files):
        # Save raw issues as JSON
        issues_path = get_output_paths(plan_path, "code_review_issues", phase='code-review')
        issues_path = issues_path.with_suffix(".json")
        with open(issues_path, "w", encoding="utf-8") as f:
            json.dump(all_issues, f, indent=2)
        print(f"Issues JSON saved to: {issues_path}")

        # Group similar issues — or, on the path where it matters most, do not.
        #
        # This function is the one that *checks* `grouped.json`'s bytes: a few
        # lines below it hands `build_inputs(..., grouped_file=grouped_path)` to
        # `read_task_file`, which compares the hash against the
        # `inputs.grouped_sha256` `validation_tasks.json` was materialized with
        # (`check_task_file_inputs`, `utils/subagent_tasks.py`). Regrouping
        # unconditionally meant it broke that binding and then audited it — the
        # file it hashes is the file it has just rewritten — so
        #
        #     ERROR: Refusing task file .../validation_tasks.json: the inputs it
        #     was materialized from have changed (grouped_sha256: ...)
        #
        # was this command refusing itself, and review-code's documented
        # validate-then-reaggregate flow could not complete at all.
        #
        # Making the round trip byte-identical (the canonical
        # `model`-then-`source_model` insertion order stamped above) fixes the
        # instance, not the class: it holds only while regrouping is perfectly
        # reproducible out of two separate dict literals in two functions. A
        # field added to an issue on one path, a reorder inside
        # `SuggestionGroup.to_dict` (`utils/suggestion_processor.py`), a changed
        # argument to this `json.dump`, or a grouping change that reorders the
        # array re-opens it — and none of those reads as a hash-binding change at
        # review time, which is how the original got in. So the bound file is not
        # rewritten at all, and byte-identity survives as the regression guard it
        # is good at being rather than as the thing the flow rests on.
        #
        # The branch is `utils/review_orchestrator_base.reaggregate_from_existing_files`'s,
        # ported — "preserves validation alignment", in its own printed words —
        # with each half of its guard meaning here what it means there:
        #
        #   * `validation.json` counts only when this run will *use* it, hence
        #     the `--skip-validation` half: a run told to ignore validation has
        #     no alignment to preserve. In this phase that disjunct is close to
        #     subsumed, since code-review only ever writes `validation.json`
        #     after merging batches and so `validation_tasks.json` was there
        #     first; it is kept because a hand-copied or partially cleaned dir
        #     carrying only the merged file is still a dir whose groups are
        #     spoken for.
        #   * `validation_tasks.json` counts *regardless* of the flag, and that
        #     asymmetry is deliberate: it is a binding on this file's bytes that
        #     outlives the invocation. A `--skip-validation` reaggregate that
        #     rewrote `grouped.json` would leave the refusal armed for the next
        #     run that does consume validation — the same defect, one command
        #     later, with nothing on screen to connect them.
        #
        # Both are ANDed with `grouped.json` actually existing, because
        # `--reaggregate` is also the path that completes a run interrupted
        # *before* grouping: `main` sends exactly that dir here ("Found N
        # per-model result file(s) but no grouped.json"). With nothing on disk
        # there is nothing bound and nothing to preserve, so that run groups and
        # writes as it always did.
        grouped_path = get_output_paths(plan_path, "code_review_grouped", phase='code-review')
        validation_path = get_output_paths(plan_path, "code_review_validation", phase='code-review')
        validation_tasks_path = phase_dir / "validation_tasks.json"

        has_existing_validation = (
            (validation_path.exists() and not args.skip_validation)
            or validation_tasks_path.exists()
        )

        if has_existing_validation and grouped_path.exists():
            print("Using existing grouped.json (preserves validation alignment)")
            with open(grouped_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            grouped = load_groups_payload(raw_data)
        else:
            raw_grouped = group_similar_suggestions(all_issues)
            # The two branches are made to agree on shape *here* rather than at
            # each consumer: `load_groups_payload` yields plain dicts (a v1 bare
            # array, or the v2 envelope's `groups`), so the regroup branch hands
            # on its stamped dict list too instead of the `SuggestionGroup`s it
            # used to. Everything below already tolerates dicts — `len(grouped)`,
            # the `[g.to_dict() if hasattr(g, 'to_dict') else g ...]` re-entries,
            # `merge_validation_batches` — and passing the stamped list moves no
            # group id, because `generate_group_id` reads `theme` and each
            # suggestion's `type`/`section`/`details` and never the stamped keys.
            grouped = [g.to_dict() if hasattr(g, 'to_dict') else g for g in raw_grouped]
            # Stamp stable content-based hashes on groups and suggestions
            stamp_stable_ids(grouped)
            # D7: a re-group orphans any verdicts on disk, so they leave with it.
            retire_stale_validation(str(phase_dir), reason="grouped.json rewritten by reaggregate")
            with open(grouped_path, 'w', encoding='utf-8') as f:
                json.dump(save_groups_payload(grouped), f, indent=2)
            print(f"Grouped issues saved to: {grouped_path}")

        validated_groups_for_report = None

        # Check if we're merging validation batch results
        if not args.skip_validation and validation_tasks_path.exists():
            print("\nMerging validation batch results...")
            # Same resume guard as the plan-review path: a v1 file (no prompts to
            # recover) or moved inputs both mean "re-materialize this phase", not
            # "interpret it anyway". Exit 3, nothing dispatched.
            try:
                batch_metadata = read_task_file(
                    str(validation_tasks_path),
                    phase_dir=str(phase_dir),
                    current_inputs=build_inputs(
                        plan_file=str(plan_path),
                        grouped_file=str(grouped_path),
                    ),
                )
            except TaskFileError as task_file_error:
                print(f"ERROR: {task_file_error}", file=sys.stderr)
                # Identical to the sibling guard in review_orchestrator_base: the
                # refusal has to reach the *process* exit code. main()'s
                # reaggregate branch discards this coroutine's return value, so a
                # `return 3` here would surface to a polling host as exit 0 —
                # "done" on a run that produced no report and needs --force.
                sys.exit(task_file_error.exit_code)

            validation_results = merge_batched_validation_results(
                output_dir=str(phase_dir),
                batch_metadata=batch_metadata,
                total_groups=len(grouped)
            )

            # Apply validation to groups
            validated_groups = apply_validation_to_groups(
                [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                validation_results
            )
            validated_groups_for_report = validated_groups

            # Save merged validation results. F3: this branch merges batches
            # already found on disk rather than calling the shared
            # `merge_validation_batches`, so it owes itself the same runner
            # attribution — the executor's own dispatch record (when
            # `--exec-subagents` produced this batch) outranks each batch
            # file's free-hand self-report.
            runner = _authoritative_validator_runner(str(phase_dir), batch_metadata)
            save_validation_results(
                validation_results, validation_path, model=runner, runner=runner
            )
            print(f"Validation results saved to: {validation_path}")
        elif not args.skip_validation:
            # No validation batches exist - need to prepare them
            print("\nPreparing batched validation for subagent execution...")
            plan_content = plan_path.read_text(encoding="utf-8")

            batched_tasks = prepare_batched_validation_tasks(
                groups=[g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                context=f"Plan file: {plan_path}\n\nPlan content:\n{plan_content[:30000]}",
                output_dir=str(phase_dir),
                plan_file=str(plan_path),
                model=args.validation_model,
                orchestrator="code_review_orchestrator.py",
                base_ref=base_ref,
                extra_instructions=CODE_REVIEW_VALIDATION_GUIDANCE,
            )

            # Save validation task instructions
            with open(validation_tasks_path, 'w', encoding='utf-8') as f:
                json.dump(batched_tasks, f, indent=2)

            total_batches = batched_tasks["total_batches"]
            stats = batched_tasks.get("batching_stats", {})

            if total_batches == 1:
                print(f"\n[VALIDATION_PENDING] {validation_tasks_path}")
            else:
                print(f"\n[VALIDATION_BATCHES_PENDING] {validation_tasks_path}")
                print(f"Batches: {total_batches}")

            print(f"\nValidation prepared for subagent execution.")
            print(f"Groups to validate: {stats.get('total_groups', len(grouped))}")

            # Resolved here, from the running skill dir, because the task file
            # stores only the durable mode + plan pair (D1c).
            reaggregate_cmd = resolved_reaggregate_command(batched_tasks)

            # D3b: with --exec-subagents this process dispatches the batch itself
            # and continues to the merge below instead of exiting at the marker.
            if dispatch_subagent_tasks(
                args,
                task_file=str(validation_tasks_path),
                phase_dir=str(phase_dir),
                phase="code-review",
                document=batched_tasks,
                plan_path=str(plan_path),
                label="validation",
                retry_command=reaggregate_cmd,
            ):
                validated_groups_for_report = merge_validation_batches(
                    str(phase_dir), batched_tasks, grouped, str(validation_path)
                )
            else:
                print("\nAfter validation completes, run:")
                print(f"  {reaggregate_cmd or ''}")
                record_next(reaggregate_cmd or None)
                return  # Exit early - Claude Code handles validation
        else:
            print("\nSkipping validation (--skip-validation flag set)")

        # Generate report
        report = generate_review_report(
            plan_path, results, changed_files, validated_groups_for_report,
            phase_dir=phase_dir, base_ref=base_ref, template_style=template_style,
        )

        # Save report
        report_path = get_output_paths(plan_path, "code_review", phase='code-review')
        report_path = report_path.with_suffix(".md")
        report_path.write_text(report, encoding="utf-8")

        print("\nReaggregation complete!")
        print(f"Report: {report_path}")

        # Mark phase complete after successful reaggregation
        state = get_or_create_state(plan_path)
        state.mark_phase_completed("code-review")
        state.save()
        update_status(str(phase_dir), {"state": "complete"})
        report_progress(str(phase_dir), step="complete")


async def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    # Line buffering + UTF-8/replace stream encoding (Windows-safe output);
    # see utils/stream_bootstrap.py for the full rationale.
    bootstrap_streams()

    args = parse_args(argv)

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

    # Derive prefix and output directory (needed for both normal and reaggregate modes)
    prefix = derive_prefix(plan_path)
    base_dir = plan_path.parent

    # Guard against double-nesting if parent already is the prefix
    if base_dir.name == prefix:
        out_dir = base_dir
    else:
        out_dir = base_dir / prefix

    # Handle --reaggregate mode: skip model invocation, just re-process existing results
    if args.reaggregate:
        print("=== REAGGREGATE MODE ===")
        print(f"Plan file: {plan_path}")
        print(f"Output directory: {out_dir}")
        print("")

        if not out_dir.is_dir():
            print(f"ERROR: Output directory does not exist: {out_dir}")
            sys.exit(1)

        # The detach boundary, crossed here as well as on the fan-out path
        # below: a `--reaggregate --detach` launch must print its run banner
        # and hand the re-dispatch to a worker, or the host that composed
        # `--detach` has nothing to poll (see the same branch in
        # `utils/review_orchestrator_base.main`).
        detach_outcome = maybe_detach(
            args=args,
            mode='review-code',
            phase_dir=str(get_phase_dir(plan_path, 'code-review')),
            plan_file=str(plan_path),
            phase='code-review',
        )
        if detach_outcome is not None:
            sys.exit(detach_outcome)

        await reaggregate_from_existing_files(
            prefix, out_dir, plan_path, args,
            template_style=args.report_style,
        )
        return

    # Guard against re-running already-completed phase (expensive LLM calls)
    state_guard = get_or_create_state(plan_path)
    if state_guard.is_phase_completed("code-review") and not args.force:
        phase_dir_check = get_phase_dir(plan_path, 'code-review')
        print("ERROR: Phase 'code-review' has already been completed for this plan.")
        print(f"Output directory: {phase_dir_check}")
        print("Use --force to re-run, or --reaggregate to reprocess existing results.")
        sys.exit(2)
    elif args.force and state_guard.is_phase_completed("code-review"):
        state_guard.state.get("phases_completed", {}).pop("code-review", None)
        state_guard.save()
        print("NOTE: Cleared previous phase completion (--force mode)")

    # Secondary guard: detect partial completion (validation_tasks.json, grouped.json, etc.)
    if not args.force:
        phase_dir_check = get_phase_dir(plan_path, 'code-review')
        validation_tasks_path = phase_dir_check / "validation_tasks.json"
        grouped_path = phase_dir_check / "grouped.json"
        validation_path = phase_dir_check / "validation.json"
        report_path_check = phase_dir_check / "report.md"

        if validation_tasks_path.exists():
            if validation_path.exists() and report_path_check.exists():
                # Auto-heal: all artifacts present, just mark complete
                state_guard.mark_phase_completed("code-review")
                state_guard.save()
                update_status(str(phase_dir_check), {"state": "complete"})
                report_progress(str(phase_dir_check), step="complete")
                print("Phase 'code-review' was complete but not marked. Auto-healed state.")
                print(f"Report: {report_path_check}")
                sys.exit(0)
            elif validation_path.exists():
                print("ERROR: Partial completion detected for 'code-review'.")
                print(f"  validation.json exists but report is missing.")
                print(f"  Run: --reaggregate to regenerate the report.")
                sys.exit(3)
            else:
                # Check for batch files
                batch_files = list(phase_dir_check.glob("validation_batch_*.json"))
                if batch_files:
                    print("ERROR: Partial completion detected for 'code-review'.")
                    print(f"  Found {len(batch_files)} validation batch file(s).")
                    print(f"  Check if all batches are complete, then run: --reaggregate")
                    sys.exit(3)
                else:
                    print("ERROR: Partial completion detected for 'code-review'.")
                    print(f"  validation_tasks.json exists but validation has not started.")
                    print(f"  Resume it: for each task in that file, run task.prompt as a")
                    print(f"  subagent and write the JSON it returns to task.output_path.")
                    sys.exit(3)
        elif grouped_path.exists():
            print("ERROR: Partial completion detected for 'code-review'.")
            print(f"  grouped.json exists but no validation_tasks.json.")
            print(f"  Run: --reaggregate to continue from grouping.")
            sys.exit(3)
        else:
            # Check for per-model result files
            import glob as glob_module
            model_files = [
                f for f in glob_module.glob(str(phase_dir_check / "*.json"))
                if not any(ex in os.path.basename(f) for ex in ['grouped', 'validation', 'salvage', 'issues', 'fix_tasks', 'state', '.status', 'report_data'])
            ]
            if model_files:
                print("ERROR: Partial completion detected for 'code-review'.")
                print(f"  Found {len(model_files)} per-model result file(s) but no grouped.json.")
                print(f"  Run: --reaggregate to aggregate results.")
                sys.exit(3)

    # Mutual exclusivity check: --quick and --interactive
    if args.quick and args.interactive:
        print("ERROR: --quick and --interactive are mutually exclusive.")
        sys.exit(1)

    # Handle model selection using the priority-based resolver
    # Priority: CLI --models > --interactive flag > --quick flag > YAML defaults > Interactive selection
    # D6 row 2: under --yes the ladder stops at defaults.models and an
    # unresolvable set is exit 3 — never a rendered multi-select.
    unattended = bool(getattr(args, "yes", False))
    try:
        model_specs = resolve_models(
            cli_models=args.models,
            interactive=args.interactive,
            quick=args.quick,
            mode='code-review',  # Mode-specific defaults from providers.yaml
            anchor=str(plan_path),  # per-project config discovery follows the plan-derived root
            unattended=unattended,
        )
    except NeedsHuman as e:
        sys.exit(unattended_mod.fail_closed(e.message, site_id=e.site))
    except RuntimeError as e:
        print(f"ERROR: {e}")
        available = get_all_model_specs(anchor=str(plan_path))
        print(f"Use --models flag. Available: {', '.join(available[:5])}...")
        sys.exit(1)

    if not model_specs:
        print("ERROR: No models selected.")
        sys.exit(1)

    # Display how models were selected
    if args.models:
        print(f"Using models from --models flag: {', '.join(model_specs)}")
    elif args.interactive:
        print(f"Using interactively selected models: {', '.join(model_specs)}")
    elif args.quick:
        print(f"Using quick models from providers.yaml: {', '.join(model_specs)}")
    else:
        print(f"Using default models from providers.yaml: {', '.join(model_specs)}")

    # Validate model specs against the registry
    invalid_models = [m for m in model_specs if not is_model_valid(m, anchor=str(plan_path))]
    if invalid_models:
        print(f"WARNING: Unknown models (proceeding anyway): {', '.join(invalid_models)}")
        # Anchored to match the rejection above (§9.3.c).
        available = get_all_model_specs(anchor=str(plan_path))
        print(f"Available models: {', '.join(available[:10])}...")

    print(f"Plan file: {plan_path}")
    print(f"Output directory: {out_dir}")
    print(f"Models: {', '.join(model_specs)}")
    if args.timeout:
        print(f"Timeout override: {args.timeout}s per model")
    else:
        print("Timeout: per-provider defaults from providers.yaml")
    print(f"Max parallel: {args.max_parallel}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Get tracked files and base ref from state (if available from implementation phase)
    state = get_or_create_state(plan_path)
    tracked_files = state.get("tracked_files", [])

    # Get the git base reference - this is the commit when implementation started
    # Used by agents to run: git diff <base_ref> -- <file>
    base_ref = args.base_ref or state.get("head_before_implement") or state.get("head_at_start") or "HEAD~1"
    base_ref = validate_git_ref(base_ref)
    print(f"Git base reference: {base_ref or '(empty - validation will lack diff context)'}")

    # Best-effort base-ref staleness visibility. Wrapped so it can NEVER fail the
    # run: surface the resolved commit, and warn if the base ref lags behind HEAD
    # (a common foot-gun when only the latest uncommitted work was meant to be
    # reviewed).
    if base_ref:
        try:
            log_out, _, log_rc = _run_git(
                "log", "-1", "--format=%h %s", base_ref, check=False
            )
            if log_rc == 0 and log_out.strip():
                print(f"Base ref commit: {log_out.strip()}")

            _, _, anc_rc = _run_git(
                "merge-base", "--is-ancestor", base_ref, "HEAD", check=False
            )
            if anc_rc == 0:
                count_out, _, count_rc = _run_git(
                    "rev-list", "--count", f"{base_ref}..HEAD", check=False
                )
                ahead = int(count_out.strip()) if count_rc == 0 and count_out.strip() else 0
                if ahead > 0:
                    print(
                        f"NOTE: base ref is {ahead} commit(s) behind HEAD. If you "
                        f"intend to review only the latest (uncommitted) work, pass "
                        f"--base-ref HEAD."
                    )
        except Exception:
            pass

    if tracked_files:
        print(f"Using {len(tracked_files)} tracked files from implementation phase")
        changed_files = list(set(entry["path"] for entry in tracked_files))
    else:
        # Fallback to git diff (legacy behavior or no implement phase run)
        if not base_ref:
            print(
                "Warning: No tracked files and no valid base_ref — cannot discover changed files via git diff. "
                "Code review will have no files to review.",
                file=sys.stderr,
            )
            changed_files = []
        else:
            print(f"No tracked files in state, falling back to git diff from {base_ref}")
            try:
                changed_files = get_files_changed_since_ref(base_ref)
            except Exception:
                changed_files = []

        # Filter out pre-existing changes that were present before implementation.
        # Through `subtract_baseline`, so a baseline entry that is an untracked
        # *directory* (`plans/`, one porcelain line) covers the files under it.
        pre_existing = list(state.get("pre_existing_changes", []))
        if pre_existing and changed_files:
            original_count = len(changed_files)
            changed_files = subtract_baseline(changed_files, pre_existing)
            filtered_count = original_count - len(changed_files)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} pre-existing files (not part of implementation)")

    print(f"Files to review: {len(changed_files)}")
    print("Running code review in agent exploration mode...")
    print("")

    # --- the detach boundary (see utils/review_orchestrator_base.maybe_detach) --
    # Late on purpose: everything above is pre-flight that must fail on the
    # caller's terminal — models resolved, base ref validated, changed files
    # discovered — and nothing below runs in the invoker.
    detach_phase_dir = get_phase_dir(plan_path, 'code-review')
    detach_outcome = maybe_detach(
        args=args,
        mode='review-code',
        phase_dir=str(detach_phase_dir),
        plan_file=str(plan_path),
        phase='code-review',
        model_specs=model_specs,
    )
    if detach_outcome is not None:
        sys.exit(detach_outcome)

    # The checkpoint this phase's *own* salvage path reads back (W3). Below the
    # boundary for the same reason the base fan-out writes it there — nothing
    # under that line runs in the invoker — and against `detach_phase_dir`,
    # which `get_phase_dir` has already created.
    #
    # `models_requested` is the only record anywhere of the canonical
    # `provider:model` specs this run fanned out to. `sanitize_model_name` is
    # lossy in one direction only, so `cursor-agent_cursor-grok-4_6-xhigh` can
    # never be turned back into `cursor-agent:cursor-grok-4.6-xhigh` from the
    # filename. Without this write `build_unsanitize_map` finds no `.status.json`
    # and returns `{}`, `resolve_result_files_by_model` falls through to "the
    # stem IS the identity", and `--reaggregate` re-attributes every group to the
    # stem — rewriting `grouped.json`'s `models` and `source_model`, and so its
    # bytes, and so the `grouped_sha256` that `check_task_file_inputs`
    # (`utils/subagent_tasks.py`) checks `validation_tasks.json` against moments
    # later in the same command. review-code's documented validate-then-
    # reaggregate path refused its own task file until this line existed.
    write_status(str(detach_phase_dir), {
        "phase": "code-review",
        "state": "models_running",
        "started_at": datetime.now().isoformat(),
        "models_requested": list(model_specs),
        "error": None,
    })
    # ...and read it straight back, because the write above cannot tell us it
    # failed. `write_status` is best-effort by contract (see the module note
    # beside `warn_stem_attribution`): no return value to check and no exception
    # to catch, so every way it can fail leaves this run proceeding *exactly* as
    # it did before the write existed, and leaves the operator with no signal at
    # all until a `--reaggregate` days later reports stems where models belong —
    # by which time the run that could have been re-done cheaply is gone.
    #
    # The probe is on `models_requested`, not on the file: that field alone is
    # what `build_unsanitize_map` needs to invert a stem, and a torn or
    # half-merged `.status.json` carrying `phase` and `state` without it is worth
    # exactly as little as no file. `read_status` returns `{}` for absent,
    # unreadable and torn alike and is documented never to raise; the try is
    # belt-and-braces so this diagnostic can never become the thing that stops
    # the dispatch it is warning about.
    #
    # Scoped to this one write. The `update_status` after the fan-out (below)
    # runs once the models have already finished, so losing it costs a resume
    # path some completion detail rather than the identity map, and does not earn
    # a second probe.
    try:
        recorded_specs = read_status(str(detach_phase_dir)).get("models_requested")
    except Exception:  # pragma: no cover - read_status is documented not to raise
        recorded_specs = None
    if not recorded_specs:
        warn_stem_attribution(
            f"WARNING: this run's checkpoint {detach_phase_dir / '.status.json'} "
            f"records no models_requested — the write is best-effort and cannot "
            f"report why. The fan-out below is unaffected and continues.",
            model_specs=list(model_specs),
        )
    if unattended and not args.models and not args.quick:
        # D6 row 2, recorded rather than merely applied (see review base): an
        # auto-selection nobody can see afterwards is indistinguishable from a
        # silent degradation. Only under `--yes`, where it replaced a question —
        # with `--yes` unset nothing about a run changes.
        unattended_mod.record_resolution(
            str(detach_phase_dir),
            unattended_mod.row(2).site_id,
            "defaults.models: " + ", ".join(model_specs),
        )
    report_progress(
        str(detach_phase_dir), step="fanout",
        models_total=len(model_specs), models_done=0,
    )

    fix_tasks_path = None
    # §3.6: filled per slot by run_single_review when a configured fallback
    # served the turn; empty under the shipped config, which is what keeps the
    # report identical.
    served_models: Dict[str, Dict[str, Any]] = {}
    with intent_to_add_untracked(changed_files):
        # Run reviews - agents receive tracked files list and base ref for git diff
        results = await run_all_reviews(
            model_specs,
            plan_path,
            out_dir,
            args.timeout,  # May be None - run_all_reviews will use per-provider defaults
            args.max_parallel,
            tracked_files,
            base_ref,
            skip_existing=not args.rerun_all,
            served_models=served_models,
        )

        # Get the phase directory for code review outputs
        phase_dir = get_phase_dir(plan_path, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Save per-model results
        failed_models: Dict[str, str] = {}
        for model_spec, (success, issues, error) in results.items():
            if not save_model_result(prefix, model_spec, success, issues, error, phase_dir):
                failed_models[model_spec] = error or "Unknown error"

        # D3b: a salvage task exists for every model whose output would not parse
        # (emit_salvage_task materialized it above). With --exec-subagents this
        # run repairs them itself, exactly as the fan-out path in
        # review_orchestrator_base does — otherwise review-code would print
        # [REAGGREGATE_AFTER_SALVAGE] and hand the host salvage subagents it may
        # not have. A recovered model's issues are folded back into `results` so
        # grouping, validation and the report all see the post-salvage truth
        # rather than the pre-salvage one. Deliberately non-fatal: a salvage that
        # fails leaves that model failed, which is the ordinary partial-results
        # outcome.
        salvaged_models = set()
        if failed_models and exec_subagents_requested(args):
            repaired_paths = {
                os.path.abspath(path)
                for path in dispatch_salvage_tasks(
                    args,
                    phase_dir=str(phase_dir),
                    phase='code-review',
                    plan_path=str(plan_path),
                )
            }
            for model_spec in list(failed_models):
                sanitized = sanitize_model_name(model_spec)
                recovered = phase_dir / f"{sanitized}.json"
                # The dispatcher's own outcome decides, not the mere presence of
                # the file: under --rerun-all a stale result from an earlier run
                # is still sitting there and must never be read as a repair.
                if os.path.abspath(str(recovered)) not in repaired_paths:
                    continue
                if not (recovered.exists() and recovered.stat().st_size > 0):
                    continue
                try:
                    repaired_issues = json.loads(recovered.read_text(encoding="utf-8"))
                except (OSError, ValueError) as read_error:
                    print(
                        f"[{model_spec}] salvage output unreadable, leaving failed: "
                        f"{read_error}",
                        file=sys.stderr,
                    )
                    continue
                if not isinstance(repaired_issues, list):
                    print(
                        f"[{model_spec}] salvage output is not a JSON array, "
                        f"leaving failed: {recovered}",
                        file=sys.stderr,
                    )
                    continue
                print(f"[{model_spec}] recovered by salvage: {recovered}")
                results[model_spec] = (True, repaired_issues, None)
                salvaged_models.add(sanitized)
                del failed_models[model_spec]

        # D6 row 3: partial results are usable output, so a run that lost *some*
        # models prints [PARTIAL_RESULTS: …], reports `state: "partial"` and
        # carries on to group, validate and report what exists — never a
        # question, never `--rerun-all`. A run where *no* model produced a result
        # has nothing to aggregate: exit 3 naming the retry command, taken here
        # (before the report is written and before mark_phase_completed) so a
        # total failure is never recorded as a completed phase. Same accounting
        # as review_orchestrator_base's fan-out, on the post-salvage truth.
        completed_models = [m for m in model_specs if m not in failed_models]
        if not completed_models:
            print("\nERROR: All models failed")
            for model_spec, reason in failed_models.items():
                print(f"  - {model_spec}: {reason}")
            sys.exit(unattended_mod.report_partial_results(
                completed=completed_models,
                failed=list(failed_models),
                phase_dir=str(phase_dir),
                # D2 row 5: shape 3 by construction (no reviewer produced a
                # result), so --force stays a command — with the caller's flags.
                retry_command=cli_common.retry_command(
                    "review-code", plan_path, "--force",
                    flags=cli_common.current_replay_flags(mode="review-code"),
                ),
            ))
        unattended_mod.report_partial_results(
            completed=completed_models,
            failed=list(failed_models),
            phase_dir=str(phase_dir),
            unattended=unattended,
        )

        # The base fan-out's completion fields, on the same post-salvage truth.
        # Two things read them and neither is optional here: `status --view run`
        # prints the top-level `state` on its own line, so a review-code run left
        # at `models_running` reports as still fanning out forever; and
        # `build_unsanitize_map` unions `models_requested` with
        # `models_completed`, so writing only the first would leave this phase's
        # checkpoint a different document from the base's for the one reader that
        # has to treat both alike.
        update_status(str(phase_dir), {
            "state": "models_complete",
            "models_completed": completed_models,
            "models_failed": failed_models,
            # Added only when something was attributed, so a default run's
            # `.status.json` gains no key.
            **({"models_served": served_models} if served_models else {}),
        })
        # The base fan-out's `models_done`, on the same truth: the fanout
        # step above wrote `models_done=0` and nothing here advanced it, so
        # `status --view run` printed `models 0/3` beside three completed
        # reviewers (measured 2026-09-16).
        report_progress(
            str(phase_dir), step="grouping",
            models_total=len(model_specs),
            models_done=len(model_specs) - len(failed_models),
        )

        # Save raw issues as JSON
        all_issues = []
        for model_spec, (success, issues, _) in results.items():
            if success and issues:
                for issue in issues:
                    issue["model"] = model_spec
                    all_issues.append(issue)

        if all_issues:
            issues_path = get_output_paths(plan_path, "code_review_issues", phase='code-review')
            issues_path = issues_path.with_suffix(".json")
            with open(issues_path, "w", encoding="utf-8") as f:
                json.dump(all_issues, f, indent=2)
            print(f"Issues JSON saved to: {issues_path}")

        # Group similar issues and validate
        validated_groups_for_report = None
        if all_issues:
            # Add source_model field for grouping
            for issue in all_issues:
                if "source_model" not in issue:
                    issue["source_model"] = issue.get("model", "unknown")

            grouped = group_similar_suggestions(all_issues)
            # Stamp stable content-based hashes on groups and suggestions
            grouped_list = [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped]
            stamp_stable_ids(grouped_list)
            grouped_path = get_output_paths(plan_path, "code_review_grouped", phase='code-review')
            # D7: the previous run's verdicts belong to the grouping this write
            # replaces; retired in the same step so no later reaggregate finds them.
            retire_stale_validation(str(grouped_path.parent), reason="grouped.json rewritten by a re-run")
            with open(grouped_path, 'w', encoding='utf-8') as f:
                json.dump(save_groups_payload(grouped_list), f, indent=2)
            print(f"Grouped issues saved to: {grouped_path}")

            # Validate grouped issues (unless skipped)
            if not args.skip_validation:
                print("\nPreparing batched validation for subagent execution...")
                # Read plan for validation context
                plan_content = plan_path.read_text(encoding="utf-8")

                batched_tasks = prepare_batched_validation_tasks(
                    groups=[g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                    context=f"Plan file: {plan_path}\n\nPlan content:\n{plan_content[:30000]}",
                    output_dir=str(phase_dir),
                    plan_file=str(plan_path),
                    model=args.validation_model,
                    orchestrator="code_review_orchestrator.py",
                    base_ref=base_ref,
                    extra_instructions=CODE_REVIEW_VALIDATION_GUIDANCE,
                )

                # Save validation task instructions
                validation_tasks_path = phase_dir / "validation_tasks.json"
                with open(validation_tasks_path, 'w', encoding='utf-8') as f:
                    json.dump(batched_tasks, f, indent=2)

                total_batches = batched_tasks["total_batches"]
                stats = batched_tasks.get("batching_stats", {})

                if total_batches == 1:
                    print(f"\n[VALIDATION_PENDING] {validation_tasks_path}")
                else:
                    print(f"\n[VALIDATION_BATCHES_PENDING] {validation_tasks_path}")
                    print(f"Batches: {total_batches}")

                print(f"\nValidation prepared for subagent execution.")
                print(f"Groups to validate: {stats.get('total_groups', len(grouped))}")
                print(f"HIGH priority: {stats.get('high_count', 0)}")
                print(f"Normal priority: {stats.get('normal_count', 0)}")
                # Fully resolved: a skill-dir placeholder no shell expands is
                # the bug. Still starts with `uv run`, so Claude Code's
                # Bash(uv:*) allow rule keeps matching.
                reaggregate_cmd = cli_common.retry_command(
                    'review-code', plan_path, '--reaggregate',
                    flags=cli_common.current_replay_flags(mode='review-code'),
                )

                # D3b: self-contained mode dispatches and merges here rather than
                # exiting for a host to drive a second turn.
                if dispatch_subagent_tasks(
                    args,
                    task_file=str(validation_tasks_path),
                    phase_dir=str(phase_dir),
                    phase="code-review",
                    document=batched_tasks,
                    plan_path=str(plan_path),
                    label="validation",
                    retry_command=reaggregate_cmd,
                ):
                    validation_path = get_output_paths(
                        plan_path, "code_review_validation", phase='code-review'
                    )
                    validated_groups_for_report = merge_validation_batches(
                        str(phase_dir), batched_tasks, grouped, str(validation_path)
                    )
                    validated_groups = validated_groups_for_report
                else:
                    print("\nAfter validation completes, run:")
                    print(f"  {reaggregate_cmd}")
                    record_next(reaggregate_cmd)
                    return  # Exit early - Claude Code handles validation
            else:
                print("\nSkipping validation (--skip-validation flag set)")
                validated_groups = [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped]

            # Prepare fix tasks for Claude Code subagents if requested
            if args.apply_fixes and validated_groups_for_report:
                fix_tasks_path = prepare_fixes_for_subagents(
                    validated_groups_for_report,
                    plan_path,
                    out_dir
                )

        # Generate report (after validation so we can include validation status)
        report = generate_review_report(
            plan_path, results, changed_files, validated_groups_for_report,
            phase_dir=phase_dir, base_ref=base_ref, template_style=args.report_style,
            served_models=served_models or None,
        )

        # Save report
        report_path = get_output_paths(plan_path, "code_review", phase='code-review')
        report_path = report_path.with_suffix(".md")
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {report_path}")

        # A detached run's caller never saw this stdout, so `status` hands the
        # artifacts back instead (no-op in the foreground).
        record_output_files(
            str(phase_dir),
            report=str(report_path),
            report_html=str(phase_dir / "report.html"),
            grouped=str(phase_dir / "grouped.json"),
        )

        # Mark phase as completed in state
        state.mark_phase_completed("code-review")
        state.save()
        # And the in-phase machine's terminal word, which `status` renders.
        update_status(str(phase_dir), {"state": "complete"})
        report_progress(str(phase_dir), step="complete")

    # Summary
    high_count = len([i for i in all_issues if i.get("importance") == "high"])
    print("\n=== Review Summary ===")
    print(f"Total issues: {len(all_issues)}")
    print(f"High priority: {high_count}")
    if high_count > 0:
        # The finding, in the channel this skill already uses for every other
        # "you must act on this" fact ([VALIDATION_PENDING], [PARTIAL_RESULTS: …],
        # [REAGGREGATE_AFTER_SALVAGE], [CONSOLIDATION_RECOMMENDED]). It carries the
        # count, which the exit code it replaces could not.
        print(f"[HIGH_PRIORITY_ISSUES] {high_count}")

    # Report fix tasks if prepared
    if args.apply_fixes and fix_tasks_path:
        valid_count = len([g for g in (validated_groups_for_report or []) if g.get("validation_status") == "valid"])
        print(f"\n{'='*60}")
        print("IMPORTANT: SUBAGENT DELEGATION REQUIRED")
        print(f"{'='*60}")
        print(f"\nValid issues to fix: {valid_count}")
        print(f"Fix tasks file: {fix_tasks_path}")
        # Host-neutral: the invoker runs task.prompt and persists what comes
        # back. No agent-type vocabulary, which is also what keeps the
        # forbidden-token lint green on printed text.
        print("\nThe invoker MUST NOW:")
        print("1. Read the fix tasks JSON file")
        print("2. For each task, run task.prompt as a subagent, then write the")
        print("   JSON it returns to task.output_path")
        print("3. DO NOT implement fixes manually - delegate every task")
        print("4. After all fixes complete, update the code review file with results")
        print("5. Update the original plan file to reference the code review")
        print(f"{'='*60}\n")

    # Check for salvage requests and print reaggregate command
    import glob as glob_module
    salvage_pattern = str(phase_dir / "salvage_*.json")
    # emit_salvage_task writes salvage_task_<model>.json beside each
    # salvage_<model>.json and the glob matches both, so one failed model would
    # be counted twice and the task file would yield a bogus model name
    # ("task_<model>") pointing the invoker at an output filename
    # reaggregation never reads. Only the request files are salvage requests.
    salvage_files = [
        path for path in glob_module.glob(salvage_pattern)
        if not os.path.basename(path).startswith("salvage_task_")
    ]

    if salvaged_models:
        # A model this run already repaired in-process is not pending salvage.
        # Its request file stays on disk as the audit trail, but it must not put
        # the host back on the salvage-then-reaggregate path.
        salvage_files = [
            path for path in salvage_files
            if os.path.basename(path) not in {
                f"salvage_{model}.json" for model in salvaged_models
            }
        ]

    if salvage_files:
        print("")
        print("=" * 60)
        print("SALVAGE NEEDED")
        print("=" * 60)
        print(f"Models requiring salvage: {len(salvage_files)}")
        for i, salvage_path in enumerate(salvage_files, 1):
            # Extract model name from filename
            basename = os.path.basename(salvage_path)
            model_name = basename.replace("salvage_", "").replace(".json", "")
            output_file = f"{model_name}.json"
            print(f"  {i}. {model_name} -> {basename}")
            print(f"     Output: {output_file}")

        # Print the reaggregate command for the host to use after salvage.
        # Fully resolved (see above) and still `uv run …`-prefixed.
        salvage_reaggregate_cmd = cli_common.retry_command(
            'review-code', plan_path, '--reaggregate',
            flags=cli_common.current_replay_flags(mode='review-code'),
        )
        print("")
        print(f"[REAGGREGATE_AFTER_SALVAGE] {salvage_reaggregate_cmd}")
        record_next(salvage_reaggregate_cmd)
        print("")
        print("Claude Code: After all salvage subagents complete, run the above command to regenerate grouped/validation/report files.")

    # A review that found HIGH-importance issues *worked*: the report is written
    # and `code-review` is recorded completed in state.json. Under the uniform
    # contract (0 done, 1 unexpected error, 2 nothing to do, 3 needs a human) that
    # is exit 0, and the 1 this shipped with claimed an error that never happened.
    # Two things read that claim. `full` routes every phase through
    # `_verify(phase, code)`, which stops the chain on any non-zero — so the
    # workflow aborted at review-code, one step before `apply-code-fixes`, the
    # phase that exists to fix exactly what the review found. And
    # `cli_common._STATE_BY_EXIT` maps 1 to "error", so the single
    # MULTI_LLM_RESULT line every host reads reported `state: "error"` for a
    # successful run. The finding itself is not lost: it is printed as
    # [HIGH_PRIORITY_ISSUES] <n> above, and `--fail-on-high` restores a non-zero
    # exit for a standalone CI gate. The sibling fan-outs (`review-plan`,
    # `review-tasks`) already return 0 regardless of what they found.
    if high_count > 0 and getattr(args, "fail_on_high", False):
        return EXIT_NEEDS_HUMAN
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
