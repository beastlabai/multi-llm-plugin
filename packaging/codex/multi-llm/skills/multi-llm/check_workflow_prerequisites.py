#!/usr/bin/env python3
"""Check workflow prerequisites before proceeding to next phase.

This script owns D6 rows **5** (the generate-tasks gate), **6** (the implement
gate) and **13** (``review-tasks`` against a plan with no task list). It is the
only owner: the gate logic is not reimplemented in markdown and no thin
orchestrator wraps it, because the *check* was already a script.

D6a — consent to run unattended is not consent to stale input. Tasks generated
from, or an implementation run against, a plan whose HIGH-importance review
suggestions were never applied silently poisons every downstream phase, and a
warning printed into a headless log nobody reads *is* that failure mode. So
``--yes`` alone never skips these gates:

* **Inside the full workflow** — :func:`ensure_prerequisites` auto-runs the
  missing apply phase **once**, then re-checks. The gate is *satisfied*, not
  bypassed; a second unmet re-check is exit 3, never a retry loop.
* **Standalone** — exit 3 with one message naming both fixes (apply first, or
  re-run with ``--skip-prerequisites``), absolute paths, D5a-shaped commands.
* **``--skip-prerequisites``** — a separate, explicit flag ``--yes`` never
  implies. Loud but non-fatal: it prints ``[PREREQUISITE_SKIPPED]``, records the
  skip through the same ``--skip``/``--reason`` state path, and is therefore
  always attributable afterwards (``status``, the implementation summary and any
  later review can all see it).

**Nothing is produced before the gate**: rows 5/6 are preflight, so an unmet
prerequisite writes no tasks into the plan file, no phase dir and no
``state.json`` mutation — there is no "generated but not applied" middle state to
clean up.

Streams: **stdout stays pure JSON** in every path (its callers ``json.loads`` the
whole of it, and ``status --view workflow`` is byte-identical to
``--status --json``); the human-readable ``[PREREQUISITE_*]`` blocks go to stderr
and travel inside the JSON as ``message``. The orchestrators that already print
prose markers on stdout (``implement_orchestrator.py``) print theirs there.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.stream_bootstrap import bootstrap_streams
from utils.cli_common import (
    EXIT_NEEDS_HUMAN,
    EXIT_OK,
    build_parent_parser,
    dispatcher_command,
)

# One table owns the phase vocabulary: the dispatcher resolves a `--mode`/
# `--phase` word against the same rows (`modes.py`), so `review-code` (the mode
# word) and `code-review` (the phase name) can never drift apart per host.
from modes import PREREQUISITE_PHASES, phase_from_selector, resolve as resolve_mode
from utils import unattended as unattended_mod
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

    # The notice and validation both count groups. Without usable verdicts,
    # duplicate suggestions inside one group must not inflate that count.
    total_groups = sum(bool(group.get("suggestions")) for group in groups)

    if total_groups == 0:
        return {"met": True, "reason": "skipped (no findings)", "actionable_count": 0}

    # Cross-reference with validation.json if available to count valid ones
    actionable_count = total_groups
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
            # If validation exists but nothing is valid, still count all groups
            # as actionable (user may override validation)
        except (json.JSONDecodeError, OSError):
            pass

    # Prerequisite NOT met - has unapplied task suggestions
    # `actionable_count` counts valid groups, or all nonempty groups when no
    # usable valid verdict exists, never suggestions inside a group. A group
    # holds one or more suggestions, so "N unapplied task suggestions" reads as
    # a suggestion count when it is actually a group count — measured
    # 2026-09-16, a run with 17 valid groups (holding 26 suggestions between
    # them) reported "17 unapplied task suggestions".
    return {
        "met": False,
        "reason": f"review-tasks produced {actionable_count} unapplied task suggestion groups",
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


# ---------------------------------------------------------------------------
# D6a — the gates, and the three branches they resolve to
# ---------------------------------------------------------------------------

UNMET_MARKER = "[PREREQUISITE_UNMET]"
SKIPPED_MARKER = "[PREREQUISITE_SKIPPED]"
SKIPPED_UPSTREAM_MARKER = "[PREREQUISITE_SKIPPED_UPSTREAM"

#: The reason string recorded for an explicit ``--skip-prerequisites``. Quoted
#: verbatim by the downstream ``[PREREQUISITE_SKIPPED_UPSTREAM …]`` notice, so a
#: later phase can say *why* it is building on unapplied review output.
SKIP_PREREQUISITES_REASON = "explicit --skip-prerequisites"

#: The prerequisites ``--skip-prerequisites`` may skip: the ones whose output is
#: *advice* (review suggestions someone chose not to apply). A missing
#: ``generate-tasks`` is not skippable — a plan with no task list has nothing to
#: implement or review, which is row 13's "not a repair this mode may make".
SKIPPABLE_PREREQUISITES: Tuple[str, ...] = (
    "apply-suggestions", "apply-task-suggestions",
)

#: The prerequisites that are a *review* phase rather than an apply phase
#: (``apply-suggestions`` needs ``review-plan``, ``apply-code-fixes`` needs
#: ``code-review``, ``apply-task-suggestions`` needs ``review-tasks``). Nothing
#: has been *produced* yet in this case, so there are no unapplied suggestions to
#: name and no artifact to point at — and, being outside
#: ``SKIPPABLE_PREREQUISITES``, nothing ``--skip-prerequisites`` may skip either.
REVIEW_PREREQUISITES: Tuple[str, ...] = (
    "review-plan", "review-tasks", "code-review",
)

#: Mode -> the D6 row it is gated by, so one table owns the mapping.
GATE_ROW: Dict[str, int] = {
    "generate-tasks": 5,
    "implement": 6,
    "review-tasks": 13,
}


def collect_missing(state: StateManager, plan_dir: Path, mode: str) -> List[dict]:
    """Every unmet hard prerequisite of *mode*, in workflow order."""
    missing: List[dict] = []
    for required_phase in PHASE_PREREQUISITES.get(mode, []):
        # Use specialized check for apply-suggestions (has detailed suggestion counting)
        if required_phase == "apply-suggestions":
            apply_check = check_apply_suggestions_prerequisite(state, plan_dir)
            if not apply_check["met"]:
                missing.append({
                    "phase": "apply-suggestions",
                    "reason": apply_check["reason"],
                    "valid_count": apply_check.get("valid_count", 0),
                    "importance_breakdown": apply_check.get("importance_breakdown", {}),
                })
        else:
            # Use generic state-based check for other phases
            phase_check = check_phase_prerequisite(state, required_phase)
            if not phase_check["met"]:
                missing.append({
                    "phase": required_phase,
                    "reason": phase_check["reason"],
                })
    return missing


def _mode_word(phase: str) -> str:
    """The dispatcher word for *phase*.

    Mode word and phase name differ for exactly one row (mode ``review-code``,
    phase ``code-review``), so a command is rendered from the table rather than
    from the phase name a prerequisite entry happens to carry.
    """
    mode = resolve_mode(phase)
    return mode.name if mode is not None else phase


def _breakdown_text(entry: dict) -> str:
    """``7 HIGH / 3 MEDIUM`` from an importance breakdown, else a plain count."""
    breakdown = entry.get("importance_breakdown") or {}
    parts = [
        f"{breakdown[level]} {level}"
        for level in ("HIGH", "MEDIUM", "LOW")
        if breakdown.get(level, 0) > 0
    ]
    if parts:
        return " / ".join(parts)
    count = entry.get("valid_count") or entry.get("actionable_count") or 0
    return f"{count}" if count else "unapplied"


def build_unmet_message(
    mode: str, plan_path: Path, plan_dir: Path, missing: Sequence[dict]
) -> str:
    """The one blocking message, both fixes named, absolute paths, D5a commands.

    This is a *blocking* message, not a warning: a warning in a headless log
    nobody reads is the failure mode, not the mitigation.
    """
    plan_abs = str(Path(plan_path).resolve())
    lines: List[str] = []
    for entry in missing:
        phase = entry.get("phase", "")
        if phase == "generate-tasks":
            # Row 13 (and implement's tasks prerequisite): the answer is
            # out-of-band — generate the tasks, then re-run. There is no
            # `--skip-prerequisites` option here on purpose.
            lines += [
                f"{UNMET_MARKER} {mode}: no task list has been generated for "
                f"{plan_abs}.",
                "  1) Generate the tasks first — "
                + dispatcher_command("generate-tasks", plan_abs, "--yes"),
                "     then re-run              — "
                + dispatcher_command(mode, plan_abs, "--yes"),
            ]
            continue
        if phase in REVIEW_PREREQUISITES:
            # The unmet prerequisite is the *review itself*: nothing has been
            # produced, so there is no `validation.json` to cite and no
            # "suggestions were not applied" to count. There is no
            # `--skip-prerequisites` option here either — `ensure_prerequisites`
            # refuses it for anything outside SKIPPABLE_PREREQUISITES, so
            # printing it would be a command the agent copies and gets exit 3 on.
            lines += [
                f"{UNMET_MARKER} {mode}: {phase} has not been run for "
                f"{plan_abs}.",
                "  1) Run it first — "
                + dispatcher_command(_mode_word(phase), plan_abs, "--yes"),
                "     then re-run  — "
                + dispatcher_command(_mode_word(mode), plan_abs, "--yes"),
            ]
            continue
        source = str((Path(plan_dir) / "review-plan" / "validation.json").resolve()) \
            if phase == "apply-suggestions" \
            else str((Path(plan_dir) / "review-tasks" / "validation.json").resolve())
        lines += [
            f"{UNMET_MARKER} {mode}: {_breakdown_text(entry)} suggestions in "
            f"{source} have not been applied.",
            "  1) Apply them first — "
            + dispatcher_command(phase, plan_abs, "--yes"),
            "     then re-run      — "
            + dispatcher_command(mode, plan_abs, "--yes"),
            "  2) Proceed on unapplied review output (recorded) —",
            "     "
            + dispatcher_command(mode, plan_abs, "--yes", "--skip-prerequisites"),
        ]
    return "\n".join(lines)


def upstream_skip_notices(state: StateManager) -> List[str]:
    """``[PREREQUISITE_SKIPPED_UPSTREAM …]`` lines for skips already recorded.

    A mode gates on **its own** prerequisite only. It does not re-litigate how the
    artifact it consumes was produced: if an earlier phase ran under
    ``--skip-prerequisites``, consent was already given explicitly, and re-asking
    would make the recorded skip meaningless. So the recorded skip surfaces as a
    single line quoting the reason, and the run proceeds.
    """
    notices: List[str] = []
    for phase in SKIPPABLE_PREREQUISITES:
        if not state.is_phase_skipped(phase):
            continue
        reason = state.get_phase_skip_reason(phase) or ""
        if SKIP_PREREQUISITES_REASON not in reason:
            continue
        notices.append(f"{SKIPPED_UPSTREAM_MARKER} {phase}: {reason}]")
    return notices


def record_prerequisite_skip(
    state: StateManager, phase: str, *, reason: str
) -> None:
    """Mark *phase* skipped through the same state path ``--skip`` uses."""
    state.mark_phase_skipped(phase, reason=reason)
    state.save()


@dataclass
class PrereqOutcome:
    """What :func:`ensure_prerequisites` decided.

    Attributes:
        met: True when the mode may proceed.
        exit_code: 0 when it may proceed, 3 when a human is needed. Never 2 —
            "needs a human" and "already done" must stay distinguishable.
        missing: The unmet prerequisite entries (empty when met).
        message: The verbatim ``[PREREQUISITE_UNMET]`` block, when blocking.
        auto_ran: Phases whose apply step this call ran (at most one per phase
            per run).
        skipped: Phases recorded as skipped by ``--skip-prerequisites``.
        notices: ``[PREREQUISITE_SKIPPED*]`` lines the caller should print.
    """

    met: bool
    exit_code: int = EXIT_OK
    missing: List[dict] = field(default_factory=list)
    message: Optional[str] = None
    auto_ran: Tuple[str, ...] = ()
    skipped: Tuple[str, ...] = ()
    notices: List[str] = field(default_factory=list)


def ensure_prerequisites(
    plan_file,
    mode: str,
    *,
    unattended: bool = False,
    in_workflow: bool = False,
    skip_prerequisites: bool = False,
    apply_runner: Optional[Callable[[str], int]] = None,
    state: Optional[StateManager] = None,
) -> PrereqOutcome:
    """Resolve rows 5/6/13 for *mode* without ever rendering a question.

    The three D6a branches, in the order they are tried:

    1. ``--skip-prerequisites`` — record the skip through the existing ``--skip``
       path, emit ``[PREREQUISITE_SKIPPED]``, proceed with the mode's normal exit
       code. The only path on which an artifact may be built from unapplied
       review output, and always attributable afterwards.
    2. *Inside the workflow* (``in_workflow`` with an ``apply_runner``) — run the
       missing apply phase **once**, then re-check. If the apply phase exits
       non-zero, or exits 0 but the re-check still reports the prerequisite unmet
       (e.g. items left ``needs-human-decision`` that ``--agent-decide`` could
       not resolve), stop with exit 3 naming the unresolved items. The auto-run is
       attempted at most once per phase per run: a second unmet re-check is exit
       3, not a retry.
    3. Otherwise — exit 3 (unattended) or the legacy prompt payload (attended,
       where a rendered gate is still legal).

    ``apply_runner`` is injected because applying suggestions needs the driving
    agent's subagent channel (``apply_suggestions_orchestrator.py`` emits
    instructions rather than editing the plan itself). It is called as
    ``apply_runner(phase) -> exit_code``. ``full_workflow_orchestrator.py`` is the
    caller that supplies it (for row 5 through ``generate_tasks_orchestrator`` and
    for row 6 before dispatching ``implement``), and it bounds the runner to one
    auto-run per phase per *run* — this function bounds it to one per *call*. With
    no runner supplied, ``in_workflow`` behaves exactly like standalone, so nothing
    can silently proceed on stale input.

    Never mutates state on the blocking path (rows 5/6 are preflight), and never
    prints: the caller owns its own streams.
    """
    plan_path = Path(plan_file).resolve()
    plan_dir = get_output_dir(plan_path)
    state = state if state is not None else StateManager(plan_path)

    missing = collect_missing(state, plan_dir, mode)
    notices = upstream_skip_notices(state)
    if not missing:
        return PrereqOutcome(met=True, notices=notices)

    if skip_prerequisites:
        unskippable = [
            m["phase"] for m in missing
            if m.get("phase") not in SKIPPABLE_PREREQUISITES
        ]
        if unskippable:
            return PrereqOutcome(
                met=False,
                exit_code=EXIT_NEEDS_HUMAN,
                missing=missing,
                message=(
                    f"ERROR: --skip-prerequisites cannot skip "
                    f"{', '.join(unskippable)} for {mode}: that phase's output is "
                    "the input, not advice about it.\n"
                    + build_unmet_message(mode, plan_path, plan_dir, missing)
                ),
                notices=notices,
            )
        skipped: List[str] = []
        for entry in missing:
            phase = entry["phase"]
            record_prerequisite_skip(state, phase, reason=SKIP_PREREQUISITES_REASON)
            skipped.append(phase)
            notices.append(
                f"{SKIPPED_MARKER} {mode}: {_breakdown_text(entry)} unapplied "
                f"{phase} suggestions (reason: {SKIP_PREREQUISITES_REASON})"
            )
        return PrereqOutcome(
            met=True, missing=[], skipped=tuple(skipped), notices=notices
        )

    if in_workflow and apply_runner is not None:
        auto_ran: List[str] = []
        for entry in missing:
            phase = entry["phase"]
            if phase not in SKIPPABLE_PREREQUISITES:
                continue
            code = apply_runner(phase)
            auto_ran.append(phase)
            if code != EXIT_OK:
                return PrereqOutcome(
                    met=False,
                    exit_code=EXIT_NEEDS_HUMAN,
                    missing=missing,
                    message=(
                        f"{UNMET_MARKER} {mode}: the auto-run of {phase} exited "
                        f"{code}, so the prerequisite is still unmet and nothing "
                        "downstream ran.\n"
                        + build_unmet_message(mode, plan_path, plan_dir, missing)
                    ),
                    auto_ran=tuple(auto_ran),
                    notices=notices,
                )
        # One re-check, never a loop.
        state = StateManager(plan_path)
        remaining = collect_missing(state, plan_dir, mode)
        if not remaining:
            return PrereqOutcome(
                met=True, auto_ran=tuple(auto_ran),
                notices=upstream_skip_notices(state),
            )
        reasons = "; ".join(m.get("reason", "") for m in remaining)
        if auto_ran:
            headline = (
                f"{UNMET_MARKER} {mode}: the apply phase ran once and the "
                f"re-check still reports the prerequisite unmet ({reasons}). "
                "Stopping here rather than retrying."
            )
        else:
            # Nothing was auto-run: every missing prerequisite is outside
            # SKIPPABLE_PREREQUISITES, so the loop above skipped it. Claiming an
            # apply phase ran would send a reader debugging a phase that never
            # executed. Exit 3 is still the right answer — the gate is unmet and
            # the workflow may not repair it on its own.
            unrunnable = ", ".join(m.get("phase", "") for m in remaining)
            headline = (
                f"{UNMET_MARKER} {mode}: nothing was auto-run — {unrunnable} is "
                f"not a prerequisite the workflow may satisfy on its own "
                f"({reasons}). Stopping here."
            )
        return PrereqOutcome(
            met=False,
            exit_code=EXIT_NEEDS_HUMAN,
            missing=remaining,
            message=(
                headline + "\n"
                + build_unmet_message(mode, plan_path, plan_dir, remaining)
            ),
            auto_ran=tuple(auto_ran),
            notices=notices,
        )

    message = build_unmet_message(mode, plan_path, plan_dir, missing)
    return PrereqOutcome(
        met=False,
        exit_code=EXIT_NEEDS_HUMAN if unattended else EXIT_OK,
        missing=missing,
        message=message,
        notices=notices,
    )


def get_workflow_status(state: StateManager, plan_path: Path, plan_dir: Path) -> dict:
    """Get current workflow state for all phases."""
    ALL_PHASES = list(PREREQUISITE_PHASES)

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

    # For implement phase, add task progress. `implementation_task_statuses`
    # (not `get_all_task_statuses`) filters `task_status` to the T### ids the
    # task decomposer owns — the same flat map also carries apply-suggestions/
    # apply-code-fixes/apply-task-suggestions write-batch ids, which are not
    # implement tasks and must never inflate this count.
    if phases["implement"]["status"] in ["pending", "in_progress"]:
        task_status = state.implementation_task_statuses()
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
            # F9.5: `actionable_count` is a count of valid GROUPS (see the
            # matching comment on `check_apply_task_suggestions_prerequisite`),
            # so this notice must not call it "suggestions" either.
            notices.append(
                f"review-tasks produced {actionable_count} unapplied task suggestion groups. "
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


def main(argv=None) -> int:
    bootstrap_streams()
    parser = argparse.ArgumentParser(
        description="Check workflow prerequisites",
        parents=[build_parent_parser()],
    )
    parser.add_argument("--plan-file", required=True, help="Path to plan file")
    parser.add_argument("--status", action="store_true",
                        help="Show current workflow state")
    parser.add_argument("--mode", "--phase", required=False,
                        help="Mode word or phase name to check prerequisites "
                             "for (either vocabulary: e.g. review-code or "
                             "code-review). One of: "
                             + ", ".join(PREREQUISITE_PHASES))
    parser.add_argument("--skip", action="store_true",
                        help="Mark the phase as skipped (requires --mode and --reason)")
    parser.add_argument("--reason", type=str, default="",
                        help="Reason for skipping the phase (used with --skip)")
    args = parser.parse_args(argv)

    # `--mode`/`--phase` takes either vocabulary and resolves through the one
    # table that owns both (`modes.py`), so `review-code` (the mode word) and
    # `code-review` (the phase name) select the same gate and cannot drift. An
    # unresolvable word is exit **3** ("this run needs a human"), not argparse's
    # exit 2, which this skill reserves for "nothing to do / already complete".
    # The message goes to stderr so stdout stays pure JSON.
    if args.mode:
        resolved_phase = phase_from_selector(args.mode)
        if resolved_phase not in PREREQUISITE_PHASES:
            print(
                f"ERROR: unknown phase or mode '{args.mode}'. Valid phases: "
                + ", ".join(PREREQUISITE_PHASES),
                file=sys.stderr,
            )
            return EXIT_NEEDS_HUMAN
        args.mode = resolved_phase

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
        record_prerequisite_skip(state, args.mode, reason=args.reason)
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

    # D6a. `--yes` decides how an unmet gate *ends* (exit 3 instead of a rendered
    # question); `--skip-prerequisites` is the separate, explicit escape `--yes`
    # never implies. Attended runs keep today's payload byte-for-byte.
    unattended = bool(getattr(args, "yes", False))
    outcome = ensure_prerequisites(
        plan_path,
        args.mode,
        unattended=unattended,
        skip_prerequisites=bool(getattr(args, "skip_prerequisites", False)),
        state=state,
    )
    missing = outcome.missing

    # Build result — compute prerequisites_met from hard prerequisites only
    result = {
        "prerequisites_met": len(missing) == 0,
        "mode": args.mode,
        "missing": missing
    }
    if outcome.skipped:
        result["skipped_prerequisites"] = list(outcome.skipped)
        result["skip_reason"] = SKIP_PREREQUISITES_REASON
    if outcome.notices:
        result["notices"] = list(outcome.notices)
        for notice in outcome.notices:
            print(notice, file=sys.stderr)
    if outcome.message is not None:
        # stdout stays pure JSON for every caller that json.loads() it; the
        # verbatim block goes to stderr and travels here as `message`.
        result["message"] = outcome.message
        result["gate_row"] = GATE_ROW.get(args.mode)
        print(outcome.message, file=sys.stderr)
    if outcome.exit_code == EXIT_NEEDS_HUMAN:
        result["state"] = "needs-human"
        result["exit"] = EXIT_NEEDS_HUMAN
        unattended_mod.mark_needs_human(args.mode)
        print(json.dumps(result, indent=2))
        return EXIT_NEEDS_HUMAN

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
    #
    # D6b: under `--yes` **no prompt object is emitted at all** — not an empty
    # one, not a "resolved" one. A payload the caller could render is a question
    # waiting to happen, so the resolution travels as `resolution` instead. Row 7
    # (the task-suggestion advisory) auto-resolves to "skip, phase recorded
    # skipped"; the recording belongs to the phase that owns it
    # (`implement_orchestrator.py`), not to this read-only check.
    if unattended:
        resolutions = []
        if advisories:
            resolutions.append({
                "site": unattended_mod.row(7).site_id,
                "decision": "skip the task-suggestion advisory and proceed",
            })
        if resolutions:
            result["resolutions"] = resolutions
    else:
        prompt = build_prompt(args.mode, missing)
        if prompt:
            result["prompt"] = prompt
        elif advisories:
            advisory_prompt = build_prompt(args.mode, advisories)
            if advisory_prompt:
                advisory_prompt["advisory"] = True
                result["prompt"] = advisory_prompt

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
