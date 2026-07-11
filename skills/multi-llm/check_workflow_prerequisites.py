#!/usr/bin/env python3
"""Check workflow prerequisites before proceeding to next phase."""

import argparse
import json
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.stream_bootstrap import bootstrap_streams
from utils.state_manager import StateManager
from utils.output_handler import get_output_dir

PHASE_PREREQUISITES = {
    "apply-suggestions": ["review-plan"],
    "generate-tasks": ["apply-suggestions"],
    "review-tasks": ["generate-tasks"],
    "apply-task-suggestions": ["review-tasks"],
    "implement": ["apply-suggestions", "generate-tasks"],
    "apply-code-fixes": ["code-review"],
}

OPTIONAL_PHASES = {"review-tasks", "apply-task-suggestions"}


def check_phase_prerequisite(state: StateManager, required_phase: str) -> dict:
    """Check if a required phase has been completed or skipped."""
    if state.is_phase_completed(required_phase):
        return {"met": True, "reason": f"{required_phase} completed"}
    if state.is_phase_skipped(required_phase):
        return {"met": True, "reason": f"{required_phase} skipped"}
    return {"met": False, "reason": f"{required_phase} not yet run"}


def check_apply_suggestions_prerequisite(state: StateManager, plan_dir: Path) -> dict:
    """
    Check if apply-suggestions phase should run before proceeding.

    Returns:
        Dict with 'met' boolean and details
    """
    review_plan_dir = plan_dir / "review-plan"
    validation_path = review_plan_dir / "validation.json"

    # No review-plan run → no prerequisite
    if not validation_path.exists():
        return {"met": True, "reason": "No review-plan results to apply"}

    # Check if apply-suggestions completed or skipped
    if state.is_phase_completed("apply-suggestions"):
        return {"met": True, "reason": "Apply-suggestions phase completed"}
    if state.is_phase_skipped("apply-suggestions"):
        reason = state.get_phase_skip_reason("apply-suggestions") or "Explicitly skipped"
        return {"met": True, "reason": f"Apply-suggestions skipped: {reason}"}

    # Count valid suggestions that would be applied
    try:
        with open(validation_path, 'r', encoding='utf-8') as f:
            validation = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"met": True, "reason": "Could not read validation results"}

    valid_count = 0
    importance_breakdown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    # Handle both v1 (flat dict keyed by group_id) and v2 (envelope with
    # metadata + groups list) validation formats.
    if isinstance(validation, dict) and "groups" in validation:
        # v2 envelope format
        validation_entries = validation["groups"]
    elif isinstance(validation, dict):
        # v1 flat dict format: convert values to a list
        validation_entries = list(validation.values())
    else:
        validation_entries = []

    for entry in validation_entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") == "valid":
            valid_count += 1
            importance = entry.get("importance", "MEDIUM")
            if importance in importance_breakdown:
                importance_breakdown[importance] += 1

    if valid_count == 0:
        return {"met": True, "reason": "No valid suggestions to apply"}

    # Prerequisite NOT met - has unapplied suggestions
    return {
        "met": False,
        "reason": f"Review-plan found {valid_count} valid suggestions not yet applied",
        "valid_count": valid_count,
        "importance_breakdown": importance_breakdown
    }


def check_apply_task_suggestions_prerequisite(state: StateManager, plan_dir: Path) -> dict:
    """
    Check if apply-task-suggestions phase should run before proceeding.

    Counts actionable task suggestions from review-tasks/grouped.json and
    surfaces them in the result. This enables the --implement soft prompt
    for unapplied task suggestions.

    Returns:
        Dict with 'met' boolean and details including actionable_count
    """
    review_tasks_dir = plan_dir / "review-tasks"
    grouped_path = review_tasks_dir / "grouped.json"
    validation_path = review_tasks_dir / "validation.json"

    # If review-tasks was skipped, apply-task-suggestions is skipped too
    if state.is_phase_skipped("review-tasks"):
        return {"met": True, "reason": "skipped (prerequisite skipped)"}

    # If review-tasks hasn't run yet (no grouped.json), nothing to do
    if not grouped_path.exists():
        return {"met": True, "reason": "No review-tasks results to apply"}

    # Check if apply-task-suggestions already completed or skipped
    if state.is_phase_completed("apply-task-suggestions"):
        return {"met": True, "reason": "Apply-task-suggestions phase completed"}
    if state.is_phase_skipped("apply-task-suggestions"):
        reason = state.get_phase_skip_reason("apply-task-suggestions") or "Explicitly skipped"
        return {"met": True, "reason": f"Apply-task-suggestions skipped: {reason}"}

    # Defensive fallback: check state["phases"] dict (written by orchestrator output
    # but not yet migrated to phases_completed by --mark-completed)
    phase_info = state.state.get("phases", {}).get("apply-task-suggestions", {})
    if phase_info.get("status") == "completed":
        return {"met": True, "reason": "Apply-task-suggestions phase completed (from phases metadata)"}

    # Load grouped.json and count actionable suggestions
    try:
        with open(grouped_path, 'r', encoding='utf-8') as f:
            grouped_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"met": True, "reason": "Could not read grouped results"}

    # Handle both v1 (bare list) and v2 (envelope with format_version) formats
    if isinstance(grouped_data, dict) and "groups" in grouped_data:
        groups = grouped_data["groups"]
    elif isinstance(grouped_data, list):
        groups = grouped_data
    else:
        return {"met": True, "reason": "No actionable task suggestions found"}

    # Count total suggestions across all groups
    total_suggestions = 0
    for group in groups:
        suggestions = group.get("suggestions", [])
        total_suggestions += len(suggestions)

    if total_suggestions == 0:
        return {"met": True, "reason": "skipped (no findings)", "actionable_count": 0}

    # Cross-reference with validation.json if available to count valid ones
    actionable_count = total_suggestions
    importance_breakdown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    if validation_path.exists():
        try:
            with open(validation_path, 'r', encoding='utf-8') as f:
                validation = json.load(f)

            # Handle both v1 (flat dict keyed by group_id) and v2 (envelope
            # with metadata + groups list) validation formats.
            if isinstance(validation, dict) and "groups" in validation:
                # v2 envelope format
                validation_groups = validation["groups"]
            elif isinstance(validation, dict):
                # v1 flat dict format: convert values to a list
                validation_groups = list(validation.values())
            else:
                validation_groups = []

            # Count valid suggestions from validation results
            valid_count = 0
            for entry in validation_groups:
                if not isinstance(entry, dict):
                    continue
                if entry.get("status") == "valid":
                    valid_count += 1
                    importance = entry.get("importance", "MEDIUM")
                    if importance in importance_breakdown:
                        importance_breakdown[importance] += 1

            if valid_count > 0:
                actionable_count = valid_count
            # If validation exists but nothing is valid, still count total
            # as actionable (user may override validation)
        except (json.JSONDecodeError, OSError):
            pass

    # Prerequisite NOT met - has unapplied task suggestions
    return {
        "met": False,
        "reason": f"review-tasks produced {actionable_count} unapplied task suggestions",
        "actionable_count": actionable_count,
        "importance_breakdown": importance_breakdown
    }


def build_prompt(mode: str, missing: list) -> dict:
    """Build prompt for user interaction based on missing prerequisites."""
    if not missing:
        return None

    # Handle apply-suggestions prerequisite
    for m in missing:
        if m.get("phase") == "apply-suggestions":
            valid_count = m.get("valid_count", 0)
            breakdown = m.get("importance_breakdown", {})

            details = []
            for level in ["HIGH", "MEDIUM", "LOW"]:
                if breakdown.get(level, 0) > 0:
                    details.append(f"{breakdown[level]} {level}")

            detail_str = ", ".join(details) if details else f"{valid_count} suggestions"

            return {
                "question": f"Review found {valid_count} suggestions ({detail_str}). Apply them before {mode}?",
                "options": [
                    {"label": "Apply suggestions first (Recommended)", "action": "run_apply_suggestions"},
                    {"label": "Skip suggestions, proceed", "action": "skip_and_continue"},
                    {"label": "Cancel", "action": "abort"}
                ]
            }

    # Handle apply-task-suggestions prerequisite
    for m in missing:
        if m.get("phase") == "apply-task-suggestions":
            actionable_count = m.get("actionable_count", 0)
            breakdown = m.get("importance_breakdown", {})

            details = []
            for level in ["HIGH", "MEDIUM", "LOW"]:
                if breakdown.get(level, 0) > 0:
                    details.append(f"{breakdown[level]} {level}")

            detail_str = ", ".join(details) if details else f"{actionable_count} task suggestions"

            return {
                "question": f"There are {actionable_count} unapplied task suggestions from review-tasks ({detail_str}). Apply them first?",
                "options": [
                    {"label": "Apply task suggestions first (Recommended)", "action": "run_apply_task_suggestions"},
                    {"label": "Skip task suggestions, proceed", "action": "skip_and_continue"},
                    {"label": "Cancel", "action": "abort"}
                ]
            }

    return None


def get_workflow_status(state: StateManager, plan_path: Path, plan_dir: Path) -> dict:
    """Get current workflow state for all phases."""
    ALL_PHASES = ["review-plan", "apply-suggestions", "generate-tasks",
                  "review-tasks", "apply-task-suggestions", "implement",
                  "code-review", "apply-code-fixes"]

    phases = {}
    suggested_next = None

    for phase in ALL_PHASES:
        if state.is_phase_completed(phase):
            timestamp = state.state.get("phases_completed", {}).get(phase, "")
            phases[phase] = {"status": "completed", "timestamp": timestamp}
        elif state.is_phase_skipped(phase):
            skip_info = state.state.get("phases_skipped", {}).get(phase, {})
            phases[phase] = {
                "status": "skipped",
                "reason": skip_info.get("reason", ""),
                "timestamp": skip_info.get("skipped_at", "")
            }
        else:
            phases[phase] = {"status": "pending"}
            if phase in OPTIONAL_PHASES:
                phases[phase]["optional"] = True
            if suggested_next is None and phase not in OPTIONAL_PHASES:
                suggested_next = phase

    # For apply-task-suggestions phase, determine skip reason from review-tasks state.
    # Cache the prerequisite check result to avoid redundant I/O (used again in notices below).
    cached_task_check = None
    if phases["apply-task-suggestions"]["status"] == "pending":
        if phases["review-tasks"]["status"] == "skipped":
            phases["apply-task-suggestions"]["status"] = "skipped"
            phases["apply-task-suggestions"]["reason"] = "skipped (prerequisite skipped)"
        elif phases["review-tasks"]["status"] == "completed":
            # Check if review-tasks produced actionable findings
            cached_task_check = check_apply_task_suggestions_prerequisite(state, plan_dir)
            if cached_task_check.get("reason") == "skipped (no findings)":
                phases["apply-task-suggestions"]["status"] = "skipped"
                phases["apply-task-suggestions"]["reason"] = "skipped (no findings)"

    # For implement phase, add task progress
    if phases["implement"]["status"] in ["pending", "in_progress"]:
        task_status = state.get_all_task_statuses()
        if task_status:
            tasks_done = sum(1 for s in task_status.values() if s == "completed")
            tasks_total = len(task_status)
            phases["implement"]["tasks_done"] = tasks_done
            phases["implement"]["tasks_total"] = tasks_total
            if tasks_done > 0 and tasks_done < tasks_total:
                phases["implement"]["status"] = "in_progress"

    notices = []

    # Check for unapplied task suggestions (reuse cached result from above —
    # cached_task_check is always set when review-tasks is completed, so no
    # redundant I/O call is needed here).
    if (phases["apply-task-suggestions"]["status"] == "pending"
            and phases["review-tasks"]["status"] == "completed"
            and cached_task_check is not None):
        actionable_count = cached_task_check.get("actionable_count", 0)
        if actionable_count > 0 and not cached_task_check["met"]:
            notices.append(
                f"review-tasks produced {actionable_count} unapplied task suggestions. "
                f"Run --apply-task-suggestions to apply them, or proceed to --implement to skip."
            )

    result = {
        "plan": str(plan_path),
        "phases": phases,
        "suggested_next": suggested_next,
        "hint": "Run /clear before starting the next phase for best performance"
    }

    if notices:
        result["notices"] = notices

    return result


def main():
    bootstrap_streams()
    parser = argparse.ArgumentParser(description="Check workflow prerequisites")
    parser.add_argument("--plan-file", required=True, help="Path to plan file")
    parser.add_argument("--status", action="store_true",
                        help="Show current workflow state")
    parser.add_argument("--mode", "--phase", required=False,
                        choices=["review-plan", "apply-suggestions", "generate-tasks",
                                 "review-tasks", "apply-task-suggestions", "implement",
                                 "code-review", "apply-code-fixes"],
                        help="Mode to check prerequisites for")
    parser.add_argument("--skip", action="store_true",
                        help="Mark the phase as skipped (requires --mode and --reason)")
    parser.add_argument("--reason", type=str, default="",
                        help="Reason for skipping the phase (used with --skip)")
    args = parser.parse_args()

    plan_path = Path(args.plan_file).resolve()

    # Handle --status mode
    if args.status:
        if not plan_path.exists():
            result = {"error": f"Plan file not found: {plan_path}"}
            print(json.dumps(result, indent=2))
            sys.exit(1)

        plan_dir = get_output_dir(plan_path)
        state = StateManager(plan_path)
        result = get_workflow_status(state, plan_path, plan_dir)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    # Validate --mode is provided when not using --status
    if not args.mode:
        print("ERROR: --mode is required unless using --status")
        sys.exit(1)

    # Handle --skip mode: mark a phase as skipped in state
    if args.skip:
        if not args.reason:
            print("ERROR: --reason is required when using --skip")
            sys.exit(1)
        if not plan_path.exists():
            print(f"ERROR: Plan file not found: {plan_path}")
            sys.exit(1)
        state = StateManager(plan_path)
        state.mark_phase_skipped(args.mode, reason=args.reason)
        state.save()
        result = {
            "skipped": True,
            "mode": args.mode,
            "reason": args.reason,
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)

    if not plan_path.exists():
        result = {
            "prerequisites_met": False,
            "mode": args.mode,
            "error": f"Plan file not found: {plan_path}"
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)

    plan_dir = get_output_dir(plan_path)
    state = StateManager(plan_path)

    missing = []

    # Get required phases for this mode
    required_phases = PHASE_PREREQUISITES.get(args.mode, [])

    for required_phase in required_phases:
        # Use specialized check for apply-suggestions (has detailed suggestion counting)
        if required_phase == "apply-suggestions":
            apply_check = check_apply_suggestions_prerequisite(state, plan_dir)
            if not apply_check["met"]:
                missing.append({
                    "phase": "apply-suggestions",
                    "reason": apply_check["reason"],
                    "valid_count": apply_check.get("valid_count", 0),
                    "importance_breakdown": apply_check.get("importance_breakdown", {})
                })
        else:
            # Use generic state-based check for other phases
            phase_check = check_phase_prerequisite(state, required_phase)
            if not phase_check["met"]:
                missing.append({
                    "phase": required_phase,
                    "reason": phase_check["reason"]
                })

    # Build result — compute prerequisites_met from hard prerequisites only
    result = {
        "prerequisites_met": len(missing) == 0,
        "mode": args.mode,
        "missing": missing
    }

    # For implement mode, also check for unapplied task suggestions (soft advisory)
    # apply-task-suggestions is NOT a hard prerequisite for implement, but we
    # surface it as an advisory prompt so users can apply task review findings first.
    # Advisories are appended AFTER prerequisites_met is computed so they don't
    # turn a soft prompt into a hard failure.
    advisories = []
    if args.mode == "implement":
        task_check = check_apply_task_suggestions_prerequisite(state, plan_dir)
        if not task_check["met"]:
            advisories.append({
                "phase": "apply-task-suggestions",
                "reason": task_check["reason"],
                "actionable_count": task_check.get("actionable_count", 0),
                "importance_breakdown": task_check.get("importance_breakdown", {})
            })

    if advisories:
        result["advisories"] = advisories

    # Add prompt if there are missing prerequisites that need user input.
    # Check hard prerequisites first; only fall back to advisory prompt if no
    # hard prerequisite prompt exists. Advisory prompts are tagged so callers
    # can distinguish them from blocking prerequisite prompts.
    prompt = build_prompt(args.mode, missing)
    if prompt:
        result["prompt"] = prompt
    elif advisories:
        advisory_prompt = build_prompt(args.mode, advisories)
        if advisory_prompt:
            advisory_prompt["advisory"] = True
            result["prompt"] = advisory_prompt

    print(json.dumps(result, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
