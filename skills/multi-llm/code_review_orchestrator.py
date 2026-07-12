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
    uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/code_review_orchestrator.py" --plan-file plans/my-plan.md

    # Override defaults with specific models
    uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/code_review_orchestrator.py" --plan-file plans/my-plan.md --models cursor-agent:auto gemini:gemini-2.5-flash

    # Force interactive selection even if defaults exist
    uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/code_review_orchestrator.py" --plan-file plans/my-plan.md --interactive
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_or_create_state,
    parse_subagent_response,
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
    PlanUpdater,
)
from utils.stream_bootstrap import bootstrap_streams
from utils.state_manager import (
    load_groups_payload,
    save_groups_payload,
    stamp_stable_ids,
    CURRENT_FORMAT_VERSION,
)
from utils.output_handler import get_phase_dir
from utils.validation import (
    prepare_batched_validation_tasks,
    merge_batched_validation_results,
)
from utils.interactive import resolve_models
from utils.provider_registry import (
    get_all_model_specs,
    parse_model_spec,
    get_provider_timeout,
    get_provider_max_concurrent,
    is_model_valid,
)
from utils.json_extractor import generate_output_path, read_json_from_file, sanitize_model_name, build_unsanitize_map
from utils.llm_client import invoke_with_provider, invoke_with_file_output
from utils.git_utils import get_project_root, validate_git_ref, _run_git
from utils.html_report_generator import generate_html_report, write_html_report, VALIDATION_ORDER, UNKNOWN_STATUS_RANK


def derive_prefix(plan_path: Path) -> str:
    """Derive output prefix from plan file path."""
    return sanitize_prefix(plan_path.name)


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Get available models for help text
    available_models = get_all_model_specs()

    parser = argparse.ArgumentParser(
        description="Review implementation changes against a plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
  uv run --project "${{CLAUDE_SKILL_DIR}}" -- python "${{CLAUDE_SKILL_DIR}}/code_review_orchestrator.py" --plan-file plans/my-plan.md

  # Specify models explicitly
  uv run --project "${{CLAUDE_SKILL_DIR}}" -- python "${{CLAUDE_SKILL_DIR}}/code_review_orchestrator.py" --plan-file plans/my-plan.md --models cursor-agent:auto gemini:gemini-2.5-flash

  # Force interactive selection
  uv run --project "${{CLAUDE_SKILL_DIR}}" -- python "${{CLAUDE_SKILL_DIR}}/code_review_orchestrator.py" --plan-file plans/my-plan.md --interactive
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
        '--interactive', '-i',
        action='store_true',
        help='Force interactive model selection (ignores YAML defaults)'
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

    return parser.parse_args()


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
            # subagent_type defaults to general-purpose
            # Claude Code determines appropriate type based on issue context
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


async def run_single_review(
    model_spec: str,
    plan_path: Path,
    out_dir: Path,
    timeout: Optional[int] = None,
    tracked_files: Optional[List[Dict]] = None,
    base_ref: str = "HEAD~1"
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

    Returns:
        Tuple of (success, issues_list, error)
    """
    import time
    start_time = time.time()

    # Detect project root so CLI providers (e.g. Claude Code) run in the
    # correct working directory and can access project files.
    project_root = get_project_root(str(plan_path))

    # Parse model spec to get provider info for display
    provider_name, model_name = parse_model_spec(model_spec)
    display_name = f"{provider_name}:{model_name}"

    # Generate log file path for debugging
    sanitized_model = sanitize_model_name(model_spec)
    phase_dir = get_phase_dir(plan_path, 'code-review')
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_file = phase_dir / f"log_{sanitized_model}.txt"

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
    effective_timeout = timeout if timeout is not None else get_provider_timeout(provider_name)

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
        cwd=project_root
    )

    duration = time.time() - start_time
    output_file = result.get("output_file", "")
    source = result.get("source", "unknown")
    print(f"[{display_name}] Log saved to: {log_file}")
    if output_file:
        print(f"[{display_name}] Output file: {output_file} (source: {source})")

    if not result["success"]:
        error_msg = result.get("error", "Unknown error")
        file_error = result.get("file_error", "")
        if file_error:
            print(f"[{display_name}] File error: {file_error}")
        print(f"[{display_name}] Agent failed ({duration:.1f}s): {error_msg}")

        # Write salvage request if we have any raw output to salvage
        raw_content = ""
        details = result.get("details", {})
        if details.get("stderr"):
            raw_content = details["stderr"]

        if raw_content or result.get("stdout_error"):
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
    skip_existing: bool = True
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

    Returns:
        Dict mapping model spec to tuple of (success, issues_list, error)
    """
    semaphore = asyncio.Semaphore(max_parallel)
    # Per-provider semaphores for concurrency limiting
    provider_semaphores: Dict[str, asyncio.Semaphore] = {}
    for spec in model_specs:
        prov, _ = parse_model_spec(spec)
        if prov not in provider_semaphores:
            limit = get_provider_max_concurrent(prov)
            if limit is not None:
                provider_semaphores[prov] = asyncio.Semaphore(limit)
    results: Dict[str, Tuple[bool, Optional[List[Dict]], Optional[str]]] = {}

    async def run_with_semaphore(model_spec: str, index: int) -> Tuple[str, Tuple[bool, Optional[List[Dict]], Optional[str]]]:
        provider_name, model_name = parse_model_spec(model_spec)
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
                        model_spec, plan_path, out_dir, timeout, tracked_files, base_ref
                    )
                    return (model_spec, result)
        else:
            async with semaphore:
                result = await run_single_review(
                    model_spec, plan_path, out_dir, timeout, tracked_files, base_ref
                )
                return (model_spec, result)

    # Create tasks for all models with staggered starts
    tasks = [run_with_semaphore(spec, idx) for idx, spec in enumerate(model_specs)]

    # Calculate total timeout based on longest provider timeout
    if timeout is not None:
        base_timeout = timeout
    else:
        # Use maximum of all provider timeouts
        base_timeout = max(get_provider_timeout(parse_model_spec(spec)[0]) for spec in model_specs)

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
) -> str:
    """Generate a markdown review report."""
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

    lines.append(f"**Models:** {', '.join(successful_models)}")
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
            lines.extend(_format_issue(issue_global_index[id(issue)], issue, validation_status=status, validation_reason=reason))
    else:
        lines.append("_No high priority issues._")
        lines.append("")

    # MEDIUM issues
    lines.append("## MEDIUM Priority")
    lines.append("")
    if medium:
        for issue in medium:
            status, reason = get_validation_info(issue)
            lines.extend(_format_issue(issue_global_index[id(issue)], issue, validation_status=status, validation_reason=reason))
    else:
        lines.append("_No medium priority issues._")
        lines.append("")

    # LOW issues
    lines.append("## LOW Priority")
    lines.append("")
    if low:
        for issue in low:
            status, reason = get_validation_info(issue)
            lines.extend(_format_issue(issue_global_index[id(issue)], issue, validation_status=status, validation_reason=reason))
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
    validation_reason: Optional[str] = None
) -> List[str]:
    """Format a single issue for the report.

    Args:
        index: The issue number for display
        issue: The issue dict with title, desc, type, file, model, etc.
        validation_status: The validation status (valid, invalid, needs-human-decision, etc.)
        validation_reason: The reason for the validation decision (shown for invalid/needs-human-decision)
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
        f"**Validation:** {validation_str} | **File:** `{file_ref}` | **Type:** {issue.get('type', 'unknown')} | **Model:** {issue.get('model', 'unknown')}",
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
    exclude_patterns = ['grouped', 'validation', 'salvage', 'issues', 'fix_tasks', 'report_data']
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

    for result_path in result_files:
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                issues = json.load(f)
                # Extract model name from filename, restore original spec
                basename = os.path.basename(result_path)
                sanitized = basename.replace(".json", "")
                model = unsanitize.get(sanitized, sanitized)
                models_found.add(model)

                # Add source_model field for grouping
                for issue in issues:
                    issue["source_model"] = model
                    issue["model"] = model  # Also set model for report generation
                all_issues.extend(issues)

                # Add to results dict for report generation
                results[model] = (True, issues, None)

                print(f"  Loaded {len(issues)} issues from {model}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not read {result_path}: {e}")

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

        # Group similar issues
        grouped = group_similar_suggestions(all_issues)
        # Stamp stable content-based hashes on groups and suggestions
        grouped_list = [g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped]
        stamp_stable_ids(grouped_list)
        grouped_path = get_output_paths(plan_path, "code_review_grouped", phase='code-review')
        with open(grouped_path, 'w', encoding='utf-8') as f:
            json.dump(save_groups_payload(grouped_list), f, indent=2)
        print(f"Grouped issues saved to: {grouped_path}")

        validated_groups_for_report = None
        validation_path = get_output_paths(plan_path, "code_review_validation", phase='code-review')
        validation_tasks_path = phase_dir / "validation_tasks.json"

        # Check if we're merging validation batch results
        if not args.skip_validation and validation_tasks_path.exists():
            print("\nMerging validation batch results...")
            with open(validation_tasks_path, 'r', encoding='utf-8') as f:
                batch_metadata = json.load(f)

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

            # Save merged validation results
            save_validation_results(validation_results, validation_path)
            print(f"Validation results saved to: {validation_path}")
        elif not args.skip_validation:
            # No validation batches exist - need to prepare them
            print("\nPreparing batched validation for Claude Code subagent...")
            plan_content = plan_path.read_text(encoding="utf-8")

            batched_tasks = prepare_batched_validation_tasks(
                groups=[g.to_dict() if hasattr(g, 'to_dict') else g for g in grouped],
                context=f"Plan file: {plan_path}\n\nPlan content:\n{plan_content[:30000]}",
                output_dir=str(phase_dir),
                plan_file=str(plan_path),
                model=args.validation_model,
                orchestrator="code_review_orchestrator.py",
                base_ref=base_ref,
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

            print(f"\nValidation prepared for Claude Code subagent.")
            print(f"Groups to validate: {stats.get('total_groups', len(grouped))}")
            print("\nAfter validation completes, run:")
            print(f"  {batched_tasks.get('reaggregate_command', '')}")
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


async def main():
    """Main entry point."""
    # Line buffering + UTF-8/replace stream encoding (Windows-safe output);
    # see utils/stream_bootstrap.py for the full rationale.
    bootstrap_streams()

    args = parse_args()

    # Validate plan file (resolve to absolute path first to handle
    # uv --directory which changes the working directory)
    plan_path = Path(args.plan_file).resolve()
    if not plan_path.exists():
        print(f"ERROR: Plan file not found: {args.plan_file}")
        print(f"       Resolved path: {plan_path}")
        sys.exit(1)

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
                    print(f"  Resume validation from the instruction file steps.")
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
    try:
        model_specs = resolve_models(
            cli_models=args.models,
            interactive=args.interactive,
            quick=args.quick,
            mode='code-review',  # Mode-specific defaults from providers.yaml
            anchor=str(plan_path),  # per-project config discovery follows the plan-derived root
        )
    except RuntimeError as e:
        # TTY error from interactive selection
        print(f"ERROR: {e}")
        available = get_all_model_specs()
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
        available = get_all_model_specs()
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

        # Filter out pre-existing changes that were present before implementation
        pre_existing = set(state.get("pre_existing_changes", []))
        if pre_existing and changed_files:
            original_count = len(changed_files)
            changed_files = [f for f in changed_files if f not in pre_existing]
            filtered_count = original_count - len(changed_files)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} pre-existing files (not part of implementation)")

    print(f"Files to review: {len(changed_files)}")
    print("Running code review in agent exploration mode...")
    print("")

    fix_tasks_path = None
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
            skip_existing=not args.rerun_all
        )

        # Get the phase directory for code review outputs
        phase_dir = get_phase_dir(plan_path, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Save per-model results
        for model_spec, (success, issues, error) in results.items():
            save_model_result(prefix, model_spec, success, issues, error, phase_dir)

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
            with open(grouped_path, 'w', encoding='utf-8') as f:
                json.dump(save_groups_payload(grouped_list), f, indent=2)
            print(f"Grouped issues saved to: {grouped_path}")

            # Validate grouped issues (unless skipped)
            if not args.skip_validation:
                print("\nPreparing batched validation for Claude Code subagent...")
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

                print(f"\nValidation prepared for Claude Code subagent.")
                print(f"Groups to validate: {stats.get('total_groups', len(grouped))}")
                print(f"HIGH priority: {stats.get('high_count', 0)}")
                print(f"Normal priority: {stats.get('normal_count', 0)}")
                print("\nAfter validation completes, run:")
                reaggregate_cmd = (
                    f'uv run --project "${{CLAUDE_SKILL_DIR}}" -- python "${{CLAUDE_SKILL_DIR}}/code_review_orchestrator.py" '
                    f'--plan-file "{plan_path}" --reaggregate'
                )
                print(f"  {reaggregate_cmd}")
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
        )

        # Save report
        report_path = get_output_paths(plan_path, "code_review", phase='code-review')
        report_path = report_path.with_suffix(".md")
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {report_path}")

        # Mark phase as completed in state
        state.mark_phase_completed("code-review")
        state.save()

    # Summary
    high_count = len([i for i in all_issues if i.get("importance") == "high"])
    print("\n=== Review Summary ===")
    print(f"Total issues: {len(all_issues)}")
    print(f"High priority: {high_count}")

    # Report fix tasks if prepared
    if args.apply_fixes and fix_tasks_path:
        valid_count = len([g for g in (validated_groups_for_report or []) if g.get("validation_status") == "valid"])
        print(f"\n{'='*60}")
        print("IMPORTANT: SUBAGENT DELEGATION REQUIRED")
        print(f"{'='*60}")
        print(f"\nValid issues to fix: {valid_count}")
        print(f"Fix tasks file: {fix_tasks_path}")
        print("\nClaude Code MUST NOW:")
        print("1. Read the fix tasks JSON file")
        print("2. For EACH task, spawn a Task subagent using the 'subagent_type' field")
        print("3. DO NOT implement fixes manually - DELEGATE to subagents")
        print("4. After all fixes complete, update the code review file with results")
        print("5. Update the original plan file to reference the code review")
        print("\nSubagent routing: all batches use 'general-purpose' subagent_type")
        print(f"{'='*60}\n")

    # Check for salvage requests and print reaggregate command
    import glob as glob_module
    salvage_pattern = str(phase_dir / "salvage_*.json")
    salvage_files = glob_module.glob(salvage_pattern)

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

        # Print the reaggregate command for Claude Code to use after salvage
        print("")
        print(f"[REAGGREGATE_AFTER_SALVAGE] uv run --project \"${{CLAUDE_SKILL_DIR}}\" -- python \"${{CLAUDE_SKILL_DIR}}/code_review_orchestrator.py\" --plan-file \"{plan_path}\" --reaggregate")
        print("")
        print("Claude Code: After all salvage subagents complete, run the above command to regenerate grouped/validation/report files.")

    sys.exit(1 if high_count > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
