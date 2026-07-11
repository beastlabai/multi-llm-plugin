"""Base class for apply orchestrators using the Template Method pattern.

Provides ``ApplyOrchestratorBase`` — a generic base class that owns the
shared 5-phase orchestration flow (setup, load, user-feedback, batching,
output).  Each concrete orchestrator (apply_suggestions, apply_code_fixes,
apply_task_suggestions) subclasses this and overrides ~8-11 abstract or
hook methods to supply phase-specific behavior.

Also provides ``build_common_arg_parser()`` for shared CLI flag construction,
and ``OrchestratorError`` for structured error propagation with exit codes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from .apply_output_helpers import (
    build_confirmation_needed_output,
    build_skipped_output,
    emit_json_output,
    write_and_emit_output,
)
from .apply_path_helpers import load_json_file
from .apply_selection_helpers import (
    merge_validation_with_groups,
    resolve_priority_args,
)
from .consolidation import load_merged_suggestions
from .filtering import (
    filter_items,
    filter_user_skipped_groups,
    resolve_bulk_option_conflicts,
    should_bypass_no_selection_confirmation,
    validate_claude_decide_items_honored,
)
from .importance import get_highest_importance
from .output_handler import derive_prefix, find_output_dir, get_phase_dir
from .report_parser import load_html_selections, merge_selections
from .state_manager import (
    CURRENT_FORMAT_VERSION,
    StateManager,
    generate_group_id,
)
from .suggestion_batcher import (
    SuggestionBatch,
    estimate_batch_processing_stats,
    group_suggestions_for_subagents,
)
from .validation import (
    ERROR_TYPE_AMBIGUOUS,
    prepare_batched_revalidation_tasks,
    revalidate_failed_items,
    save_validation_results,
)


# ---------------------------------------------------------------------------
# Type variables for generic base class
# ---------------------------------------------------------------------------

TItem = TypeVar("TItem")
TBatch = TypeVar("TBatch")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class OrchestratorError(Exception):
    """Raised for expected orchestrator failures with a specific exit code.

    Attributes:
        message: Human-readable error description including lifecycle phase
            and step context.
        exit_code: Process exit code to use when this error is not caught.
    """

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# Validation-override allowlist
# ---------------------------------------------------------------------------

# The only override values accepted when ingested report selections are turned
# into status/marker decisions. Anything else (a typo or a stale value left in
# user_selections.json / consolidated_user_selections.json / a hand-edited
# Markdown checkbox label) is warned-and-ignored at the apply boundary rather
# than written verbatim into ``validation_status`` (which would produce a bogus
# status that silently falls through ``filter_items``). ``claude_decide`` is a
# *routing marker*, not a status: an allowlisted ``claude_decide`` is handed to
# ``_route_claude_decide_marker`` and never assigned to ``validation_status``.
# ``needs-human-decision`` is accepted because the consolidated surface can emit
# it as an override value.
VALID_OVERRIDE_VALUES: frozenset = frozenset(
    {"valid", "invalid", "claude_decide", "needs-human-decision"}
)


def _format_override_target(value: str) -> str:
    """Render an override value for the dry-run/no-selection override print.

    ``claude_decide`` is a routing marker, not a validation status, so label it
    as such instead of printing a bare ``-> claude_decide`` (which misreads as a
    status change).
    """
    if value == "claude_decide":
        return "Let Claude decide (routing marker)"
    return value


# ---------------------------------------------------------------------------
# Argument parser builder
# ---------------------------------------------------------------------------


def build_common_arg_parser(
    description: str,
    epilog: str,
    *,
    include_revalidation: bool = False,
    include_output_format: bool = False,
    include_skip: bool = False,
    include_base_ref: bool = False,
    include_approve_validation_failed: bool = False,
    include_revalidate_all_human: bool = False,
    include_internal_revalidation: bool = False,
    include_mark_completed: bool = False,
) -> argparse.ArgumentParser:
    """Build an argument parser with common flags plus per-orchestrator extras.

    Always includes the full common flag set:
        --plan-file, --yes, --force, --dry-run, --fresh, --resume,
        --approve-all, --include-low, --min-priority, --verbose,
        --approve-all-low, --skip-all-human, --approve-importance,
        --claude-decide, --include-high, --no-batch, --max-batch-size,
        --batch-review-mode, --accept-stale-consolidation, --no-confirm,
        --skip-human-review

    Conditionally includes flags based on keyword toggles to match the
    per-orchestrator flag matrix.

    Args:
        description: Parser description text.
        epilog: Parser epilog text.
        include_revalidation: Add --revalidate and --revalidate-model.
        include_output_format: Add --output-format.
        include_skip: Add --skip.
        include_base_ref: Add --base-ref.
        include_approve_validation_failed: Add --approve-validation-failed.
        include_revalidate_all_human: Add --revalidate-all-human.
        include_internal_revalidation: Add --internal-revalidation.
        include_mark_completed: Add --mark-completed.

    Returns:
        A fully-constructed ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    # =================================================================
    # Common flags (always present)
    # =================================================================
    parser.add_argument(
        "--plan-file",
        required=True,
        help="Path to the implementation plan markdown file",
    )

    # --- Legacy/Basic Options ---
    parser.add_argument(
        "--skip-human-review",
        action="store_true",
        help="[DEPRECATED] Skip needs-human-decision items (use --skip-all-human instead)",
    )

    parser.add_argument(
        "--min-priority",
        choices=["low", "medium", "high"],
        default=None,
        help="Minimum importance level to include (default: low = all valid suggestions)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be applied without outputting for execution",
    )

    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable smart batching (process each suggestion separately)",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum suggestions per batch (default: 4)",
    )

    # --- Bulk Approval Options ---
    parser.add_argument(
        "--approve-all-low",
        action="store_true",
        help="Auto-approve all LOW importance items with needs-human-decision status",
    )

    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="Auto-approve ALL needs-human-decision items (use with caution, requires --yes or --force)",
    )

    parser.add_argument(
        "--skip-all-human",
        action="store_true",
        help="Skip all needs-human-decision and validation_failed items",
    )

    parser.add_argument(
        "--approve-importance",
        choices=["LOW", "MEDIUM", "HIGH"],
        nargs="+",
        help="Auto-approve needs-human-decision items at these importance levels",
    )

    parser.add_argument(
        "--claude-decide",
        "--let-claude-decide",
        dest="claude_decide",
        action="store_true",
        help=(
            "Let Claude evaluate each needs-human-decision item with its own "
            "judgment (no interactive prompt), instead of asking you. Equivalent "
            "to choosing 'Let Claude decide' in the interactive review. Unlike "
            "--approve-all / --skip-all-human, this is a per-item judgment, not a "
            "blanket rule: partially-valid items are salvaged (trimmed to their "
            "worthwhile core and applied) rather than skipped wholesale. Does not "
            "bypass the no-selection confirmation for valid items — combine with "
            "--no-confirm for fully unattended runs."
        ),
    )

    # --- Safety Guardrails ---
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Confirm bulk approval operations (required with --approve-all)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force bulk approval operations (alias for --yes)",
    )

    parser.add_argument(
        "--include-high",
        action="store_true",
        help="Allow --approve-all to include HIGH importance items (otherwise blocked)",
    )

    # --- Resume/State Options ---
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint, skipping already-processed items",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear previous progress and start fresh",
    )

    # --- Batch Review Mode ---
    parser.add_argument(
        "--batch-review-mode",
        choices=["individual", "by-importance", "summary-only"],
        default="by-importance",
        help="How to present items for human review (default: by-importance)",
    )

    parser.add_argument(
        "--accept-stale-consolidation",
        action="store_true",
        help="Accept consolidated decisions even if grouped.json has changed since consolidation",
    )

    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt when no user selections are found (for unattended operation)",
    )

    parser.add_argument(
        "--include-low",
        action="store_true",
        help="[DEPRECATED] No longer needed, low is now default. Use --min-priority to filter.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug output",
    )

    # =================================================================
    # Conditional flags (per-orchestrator toggles)
    # =================================================================

    if include_revalidation:
        parser.add_argument(
            "--revalidate",
            action="store_true",
            help="Re-run validation only on items with validation_failed status",
        )
        parser.add_argument(
            "--revalidate-model",
            type=str,
            default=None,
            help="Model to use for revalidation (e.g., cursor-agent:opus)",
        )

    if include_output_format:
        parser.add_argument(
            "--output-format",
            choices=["json", "text"],
            default="text",
            help="Output format (default: text)",
        )

    if include_skip:
        parser.add_argument(
            "--skip",
            action="store_true",
            help="Mark the phase as skipped without applying suggestions",
        )

    if include_base_ref:
        parser.add_argument(
            "--base-ref",
            type=str,
            help="Git ref to use for diffs (default: from state file or HEAD~1)",
        )

    if include_approve_validation_failed:
        parser.add_argument(
            "--approve-validation-failed",
            action="store_true",
            help="Auto-approve items that had recoverable validation failures (e.g., parsing/timeout)",
        )

    if include_revalidate_all_human:
        parser.add_argument(
            "--revalidate-all-human",
            action="store_true",
            help="Re-run validation on ALL needs-human-decision items",
        )

    if include_internal_revalidation:
        parser.add_argument(
            "--internal-revalidation",
            action="store_true",
            help="Run revalidation inside orchestrator (legacy mode) instead of delegating to Claude Code subagent",
        )

    if include_mark_completed:
        parser.add_argument(
            "--mark-completed",
            action="store_true",
            help="Mark the phase as completed (call after all batches are processed)",
        )

    return parser


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ApplyOrchestratorBase(Generic[TItem, TBatch]):
    """Template Method base class for the three apply orchestrators.

    Subclasses **must** set the following class-level attributes:

    * ``phase_name``  — e.g. ``"apply-suggestions"``
    * ``review_subdir`` — e.g. ``"review-plan"``
    * ``item_noun`` — e.g. ``"suggestion"``
    * ``supports_revalidation`` — ``True`` / ``False``
    * ``supports_skip_flag`` — ``True`` / ``False``
    * ``marks_phase_completed`` — ``True`` / ``False``

    And implement all 11 abstract methods (see the *Abstract methods*
    section below).

    The constructor accepts a pre-parsed ``argparse.Namespace`` — the
    caller (typically ``__main__``) is responsible for invoking
    ``SubClass.parse_args()`` and passing the result.
    """

    # ------------------------------------------------------------------
    # Class-level config (subclass must override)
    # ------------------------------------------------------------------

    phase_name: str = ""
    review_subdir: str = ""
    item_noun: str = ""
    supports_revalidation: bool = False
    supports_skip_flag: bool = False
    marks_phase_completed: bool = False
    # Override to emit a different phase identifier in confirmation_needed output.
    # When empty, defaults to phase_name.
    confirmation_phase_name: str = ""
    # Whether find_output_dir should guard against double-nesting when the
    # parent directory already matches the prefix.  The code-fixes orchestrator
    # needs this (True); the suggestions and task-suggestions orchestrators
    # originally did not have this guard, so they set it to False.
    guard_double_nesting: bool = True

    # ------------------------------------------------------------------
    # Subclass validation
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that subclasses set all required config attributes."""
        super().__init_subclass__(**kwargs)

        # Validate string config attributes are non-empty
        for attr in ("phase_name", "review_subdir", "item_noun"):
            value = getattr(cls, attr, "")
            if not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"{cls.__name__} must set '{attr}' to a non-empty string "
                    f"(got {value!r})"
                )

        # Validate boolean config attributes are explicitly set (not inherited defaults)
        for attr in ("supports_revalidation", "supports_skip_flag", "marks_phase_completed"):
            # Check that the attribute is defined directly on the subclass, not inherited
            if attr not in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} must explicitly set '{attr}' (bool). "
                    f"Do not rely on the base class default."
                )
            value = cls.__dict__[attr]
            if not isinstance(value, bool):
                raise TypeError(
                    f"{cls.__name__}.{attr} must be a bool (got {type(value).__name__})"
                )

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logger = logging.getLogger(f"orchestrator.{self.phase_name}")

        # State populated during lifecycle phases
        self.plan_path: str = ""
        self.prefix: str = ""
        self.out_dir: str = ""
        self.state: Optional[StateManager] = None

        # Data populated during _load_and_parse_inputs
        self.groups: List[Dict[str, Any]] = []
        self.validation: List[Dict[str, Any]] = []
        self.merged: List[Dict[str, Any]] = []

        # User feedback data
        self.edit_log: List[Dict[str, Any]] = []
        self.skipped_group_indices: Set[str] = set()
        self.skipped_suggestion_ids: Set[str] = set()
        self.old_skipped_ids: Set[str] = set()
        self.validation_overrides: Dict[str, str] = {}
        self.suggestion_validation_overrides: Dict[str, str] = {}
        # Group hashes / suggestion ids the reviewer pre-marked "Let Claude
        # decide" in the report. A routing marker, not a validation status —
        # marked items keep validation_status == "needs-human-decision" and
        # carry group["claude_decide"] = True so they reach the per-item judge.
        #
        # NOTE: write-only diagnostic mirror. Nothing in the production code
        # paths READS this set — all routing/filtering/formatting is driven by
        # group["claude_decide"] and the per-item decision_mode flag. It exists
        # purely as an observable record for tests/diagnostics and does NOT gate
        # any runtime behavior. (The _write_outputs() claude_decide audit keys
        # off build_human_review_config()["claude_decide_item_ids"], not this
        # set.) Do not assume populating it changes how items are routed.
        self.claude_decide_overrides: Set[str] = set()

        # Filtering results
        self.valid: List[Dict[str, Any]] = []
        self.needs_human: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, Any]] = []
        self.user_skipped_items: List[Dict[str, Any]] = []
        self.formatted_skipped: List[Dict[str, Any]] = []
        self.formatted_valid: List[Dict[str, Any]] = []
        self.formatted_human: List[Dict[str, Any]] = []

        # Batching results
        self.batches: List[Any] = []
        self.batching_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Template method: orchestrate the full apply flow.

        Calls the five lifecycle phases in order.  Catches
        ``OrchestratorError`` for consistent exit codes.

        Returns:
            Process exit code (0 on success).
        """
        try:
            self._setup()
            self._load_and_parse_inputs()
            self._apply_user_feedback()
            self._prepare_batches()
            return self._write_outputs()
        except OrchestratorError as e:
            self.logger.error("Orchestrator failed: %s", e)
            return e.exit_code
        except SystemExit as e:
            # Allow sys.exit() calls to propagate (used by early-exit paths)
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected error in %s: %s",
                self.phase_name,
                e,
                exc_info=True,
            )
            return 1

    # ==================================================================
    # Phase 1 — _setup
    # ==================================================================

    def _setup(self) -> None:
        """Phase 1: Environment and argument validation.

        Steps 1-10 from the lifecycle spec.
        """
        self.logger.debug("Entering _setup phase")

        # Step 1: Resolve plan path
        self.plan_path = os.path.abspath(self.args.plan_file)

        # Step 2: Handle --skip (if supports_skip_flag)
        if self.supports_skip_flag and getattr(self.args, "skip", False):
            state = StateManager(Path(self.plan_path))
            state.mark_phase_skipped(self.phase_name, "User chose to skip")
            state.save()
            print(
                json.dumps(
                    {
                        "status": "skipped",
                        "message": f"{self.phase_name.capitalize()} phase skipped by user request",
                    }
                )
            )
            sys.exit(0)

        # Handle --mark-completed (for task_suggestions)
        if getattr(self.args, "mark_completed", False):
            state = StateManager(Path(self.plan_path))
            state.mark_phase_completed(self.phase_name)
            state.save()
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "message": f"{self.phase_name} phase marked as completed",
                    }
                )
            )
            sys.exit(0)

        # Step 3: Validate plan file exists
        if not os.path.isfile(self.plan_path):
            print(
                f"ERROR: Plan file not found: {self.args.plan_file}",
                file=sys.stderr,
            )
            print(
                f"       Resolved path: {self.plan_path}",
                file=sys.stderr,
            )
            raise OrchestratorError(
                f"Phase _setup, step 3: Plan file not found: {self.plan_path}",
                exit_code=1,
            )

        # Step 4: Derive prefix + output dir
        self.prefix = derive_prefix(self.plan_path)
        self.out_dir = find_output_dir(
            self.plan_path, guard_double_nesting=self.guard_double_nesting
        )

        print(f"Plan: {self.plan_path}", file=sys.stderr)
        print(f"Output directory: {self.out_dir}", file=sys.stderr)
        print(f"Prefix: {self.prefix}", file=sys.stderr)

        # Step 5: pre_validate_hook (subclass-specific setup)
        try:
            self.pre_validate_hook()
        except OrchestratorError:
            raise
        except Exception as e:
            raise OrchestratorError(
                f"Phase _setup, step 5: pre_validate_hook failed: {e}",
                exit_code=1,
            ) from e

        # Step 6: Safety guardrails for --approve-all
        if getattr(self.args, "approve_all", False):
            if not (getattr(self.args, "yes", False) or getattr(self.args, "force", False)):
                print(
                    "ERROR: --approve-all requires --yes or --force flag for safety.",
                    file=sys.stderr,
                )
                print(
                    "       This prevents accidental bulk approval of potentially risky changes.",
                    file=sys.stderr,
                )
                raise OrchestratorError(
                    "Phase _setup, step 6: --approve-all requires --yes or --force",
                    exit_code=1,
                )

        # Step 7: Deprecation warnings
        if getattr(self.args, "skip_human_review", False):
            print(
                "DEPRECATION WARNING: --skip-human-review is deprecated.",
                file=sys.stderr,
            )
            print(
                "                     Please use --skip-all-human instead.",
                file=sys.stderr,
            )

        # Step 8: Warn on contradictory --claude-decide combinations.
        # --claude-decide only affects items that remain needs-human-decision;
        # bulk flags that drain that bucket leave it with nothing to judge.
        if getattr(self.args, "claude_decide", False):
            draining_flags = [
                name
                for name, flag in (
                    ("--skip-all-human", getattr(self.args, "skip_all_human", False)),
                    ("--skip-human-review", getattr(self.args, "skip_human_review", False)),
                    ("--approve-all", getattr(self.args, "approve_all", False)),
                )
                if flag
            ]
            if draining_flags:
                print(
                    "WARNING: --claude-decide has no effect alongside "
                    f"{', '.join(draining_flags)}; those flags resolve all "
                    "needs-human-decision items before Claude can judge them.",
                    file=sys.stderr,
                )

        # Step 9: Validate output dir exists
        if not os.path.isdir(self.out_dir):
            print(
                f"ERROR: Output directory not found: {self.out_dir}",
                file=sys.stderr,
            )
            raise OrchestratorError(
                f"Phase _setup, step 9: Output directory not found: {self.out_dir}",
                exit_code=1,
            )

        # Step 10: Initialize StateManager + handle --fresh
        self.state = StateManager(Path(self.plan_path))

        if getattr(self.args, "fresh", False):
            print(
                "[orchestrator] Fresh start requested, clearing previous progress...",
                file=sys.stderr,
            )
            self.state.clear_human_decisions(self.phase_name)
            self.state.clear_processed_items(self.phase_name)
            self.state.clear_processing_progress(self.phase_name)
            self.state.save()
            # Clear prior changes history for a fresh start
            prior_changes_file = Path(self.out_dir) / "prior_changes.jsonl"
            if prior_changes_file.exists():
                prior_changes_file.unlink()
                print(
                    "[orchestrator] Cleared prior changes history.",
                    file=sys.stderr,
                )

        self.logger.info(
            "Setup complete: plan=%s, out_dir=%s", self.plan_path, self.out_dir
        )

    # ==================================================================
    # Phase 2 — _load_and_parse_inputs
    # ==================================================================

    def _load_and_parse_inputs(self) -> None:
        """Phase 2: Data loading and report parsing.

        Steps 11-17 from the lifecycle spec.
        """
        self.logger.debug("Entering _load_and_parse_inputs phase")

        # Step 11: load_data() -> (groups, validation)
        try:
            self.groups, self.validation = self.load_data()
        except OrchestratorError:
            raise
        except Exception as e:
            raise OrchestratorError(
                f"Phase _load_and_parse_inputs, step 11: load_data failed: {e}",
                exit_code=1,
            ) from e

        self.logger.info(
            "Loaded %d groups, %d validation results",
            len(self.groups),
            len(self.validation) if self.validation else 0,
        )
        print(f"Loaded {len(self.groups)} {self.item_noun} groups", file=sys.stderr)
        if self.validation is not None:
            print(
                f"Loaded {len(self.validation)} validation results",
                file=sys.stderr,
            )

        # Step 12: post_load_hook
        try:
            self.post_load_hook()
        except OrchestratorError:
            raise
        except Exception as e:
            raise OrchestratorError(
                f"Phase _load_and_parse_inputs, step 12: post_load_hook failed: {e}",
                exit_code=1,
            ) from e

        # Step 13: Parse user edits
        review_dir = os.path.join(self.out_dir, self.review_subdir)
        report_path = os.path.join(review_dir, "report.md")

        try:
            edited_descriptions = self.parse_user_edits(report_path)
        except Exception as e:
            raise OrchestratorError(
                f"Phase _load_and_parse_inputs, step 13: parse_user_edits failed: {e}",
                exit_code=1,
            ) from e

        if edited_descriptions:
            try:
                self.groups, self.edit_log = self.merge_user_edits(
                    self.groups, edited_descriptions
                )
            except Exception as e:
                raise OrchestratorError(
                    f"Phase _load_and_parse_inputs, step 13: merge_user_edits failed: {e}",
                    exit_code=1,
                ) from e
            print(
                f"Merged {len(self.edit_log)} user-edited descriptions",
                file=sys.stderr,
            )

        # Step 14: Parse skips from report
        try:
            skips_result = self.parse_skips_from_report(report_path)
        except Exception as e:
            raise OrchestratorError(
                f"Phase _load_and_parse_inputs, step 14: parse_skips_from_report failed: {e}",
                exit_code=1,
            ) from e

        # parse_skips_from_report returns a tuple of
        # (skipped_group_indices, skipped_suggestion_ids, old_skipped_ids)
        if isinstance(skips_result, tuple) and len(skips_result) == 3:
            self.skipped_group_indices, self.skipped_suggestion_ids, self.old_skipped_ids = skips_result
        else:
            # Fallback: single set of skipped IDs treated as group indices
            self.skipped_group_indices = skips_result if isinstance(skips_result, set) else set()

        # Step 15: Parse validation overrides from report
        try:
            overrides_result = self.parse_validation_overrides_from_report(report_path)
        except Exception as e:
            raise OrchestratorError(
                f"Phase _load_and_parse_inputs, step 15: parse_validation_overrides_from_report failed: {e}",
                exit_code=1,
            ) from e

        if isinstance(overrides_result, tuple) and len(overrides_result) == 2:
            self.validation_overrides = dict(overrides_result[0])
            self.suggestion_validation_overrides = dict(overrides_result[1])
        else:
            self.validation_overrides = dict(overrides_result) if overrides_result else {}

        # Step 16: Load HTML selections + apply HTML-edited descriptions
        self.load_html_selections(review_dir, edited_descriptions)

        # Step 17: Load/apply consolidated (C-level) decisions
        self.load_and_apply_consolidated_decisions(review_dir)

        # Log skip/override summary
        if self.skipped_group_indices:
            print(
                f"User skipped {len(self.skipped_group_indices)} groups: "
                f"{sorted(self.skipped_group_indices)}",
                file=sys.stderr,
            )
        if self.skipped_suggestion_ids:
            print(
                f"User skipped {len(self.skipped_suggestion_ids)} individual suggestions: "
                f"{', '.join(sorted(self.skipped_suggestion_ids))}",
                file=sys.stderr,
            )
        if self.old_skipped_ids:
            print(
                f"User skipped {len(self.old_skipped_ids)} suggestions (old format): "
                f"{', '.join(sorted(self.old_skipped_ids))}",
                file=sys.stderr,
            )

        self.logger.info(
            "Input parsing complete: %d groups, %d skips, %d overrides",
            len(self.groups),
            len(self.skipped_group_indices) + len(self.skipped_suggestion_ids),
            len(self.validation_overrides),
        )

    # ==================================================================
    # Phase 3 — _apply_user_feedback
    # ==================================================================

    def _apply_user_feedback(self) -> None:
        """Phase 3: Validation, filtering, and user confirmation.

        Steps 18-27 from the lifecycle spec.
        """
        self.logger.debug("Entering _apply_user_feedback phase")

        # Step 18: Print preference summary to stderr
        has_any_preferences = (
            self.skipped_group_indices
            or self.skipped_suggestion_ids
            or self.old_skipped_ids
            or self.edit_log
            or self.validation_overrides
            or self.suggestion_validation_overrides
        )

        if has_any_preferences:
            parts: List[str] = []
            total_skips = (
                len(self.skipped_group_indices)
                + len(self.skipped_suggestion_ids)
                + len(self.old_skipped_ids)
            )
            if total_skips:
                parts.append(f"{total_skips} items will be skipped")
            if self.validation_overrides:
                parts.append(f"{len(self.validation_overrides)} validation overrides")
            if self.edit_log:
                parts.append(f"{len(self.edit_log)} edited descriptions")
            if self.suggestion_validation_overrides:
                parts.append(
                    f"{len(self.suggestion_validation_overrides)} per-suggestion overrides"
                )
            print(
                f"User preference summary: {', '.join(parts)}", file=sys.stderr
            )
            # Detail which groups are skipped by theme (hash-based lookup)
            if self.skipped_group_indices:
                _hash_to_info: Dict[str, Tuple[str, str]] = {}
                for group in self.groups:
                    ghash = group.get("group_hash", "")
                    if ghash:
                        _hash_to_info[ghash] = (
                            group.get("display_label", ghash[:8]),
                            group.get("theme", "Unknown"),
                        )
                for ghash in sorted(self.skipped_group_indices):
                    if ghash in _hash_to_info:
                        label, theme = _hash_to_info[ghash]
                        print(
                            f"  Skipping group {label}: {theme}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  Skipping group [{ghash[:8]}]",
                            file=sys.stderr,
                        )
            if self.skipped_suggestion_ids:
                print(
                    f"  Skipping individual suggestions: "
                    f"{', '.join(sorted(self.skipped_suggestion_ids))}",
                    file=sys.stderr,
                )
            # Detail validation overrides
            if self.validation_overrides:
                _hash_to_info_ovr: Dict[str, Tuple[str, str]] = {}
                for group in self.groups:
                    ghash = group.get("group_hash", "")
                    if ghash:
                        _hash_to_info_ovr[ghash] = (
                            group.get("display_label", ghash[:8]),
                            group.get("theme", "Unknown"),
                        )
                for ghash in sorted(self.validation_overrides):
                    new_status = _format_override_target(
                        self.validation_overrides[ghash]
                    )
                    if ghash in _hash_to_info_ovr:
                        label, theme = _hash_to_info_ovr[ghash]
                        print(
                            f"  Override {label} ({theme}): -> {new_status}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  Override [{str(ghash)[:8]}]: -> {new_status}",
                            file=sys.stderr,
                        )
            if self.suggestion_validation_overrides:
                for sid in sorted(self.suggestion_validation_overrides):
                    new_status = _format_override_target(
                        self.suggestion_validation_overrides[sid]
                    )
                    print(
                        f"  Override {sid}: -> {new_status}",
                        file=sys.stderr,
                    )
        else:
            # Step 19: No-selection confirmation prompt
            print(
                "No user preferences detected (checked user_selections.json, "
                "consolidated_user_selections.json, and report.md)",
                file=sys.stderr,
            )
            if not should_bypass_no_selection_confirmation(
                no_confirm=getattr(self.args, "no_confirm", False),
                yes=getattr(self.args, "yes", False),
                force=getattr(self.args, "force", False),
                approve_all=getattr(self.args, "approve_all", False),
                skip_all_human=getattr(self.args, "skip_all_human", False),
                approve_all_low=getattr(self.args, "approve_all_low", False),
                approve_importance=getattr(self.args, "approve_importance", None),
                approve_validation_failed=getattr(
                    self.args, "approve_validation_failed", False
                ),
                dry_run=getattr(self.args, "dry_run", False),
            ):
                output = build_confirmation_needed_output(
                    phase=self.confirmation_phase_name or self.phase_name,
                    message=(
                        "No user selections found in user_selections.json, "
                        "consolidated_user_selections.json, or report.md. "
                        "ALL valid items will be applied if you proceed. "
                        "To proceed, re-run with --no-confirm, --yes, or any bulk approval flag."
                    ),
                    item_count=len(self.groups),
                )
                print(json.dumps(output, indent=2))
                sys.exit(0)

        # Step 21: Handle revalidation (if supports_revalidation)
        self.handle_revalidation()

        # Step 22: Merge validation with groups
        self.merged = merge_validation_with_groups(self.groups, self.validation)

        # Step 23: Handle --resume
        if getattr(self.args, "resume", False) and self.state is not None:
            previous_decisions = self.state.get_all_human_decisions(self.phase_name)
            processed_items = self.state.get_processed_items(self.phase_name)

            for group in self.merged:
                group["_group_id"] = generate_group_id(group)

            unprocessed_merged = []
            for g in self.merged:
                gid = g["_group_id"]
                if gid in processed_items:
                    continue
                if gid in previous_decisions:
                    decision = previous_decisions[gid]
                    if decision["decision"] == "approved":
                        g["validation_status"] = "valid"
                        g["validation_reason"] = (
                            "Previously approved by human (resumed)"
                        )
                    elif decision["decision"] == "skipped":
                        g["validation_status"] = "invalid"
                        g["validation_reason"] = (
                            "Previously skipped by human (resumed)"
                        )
                unprocessed_merged.append(g)

            print(
                f"[resume] {len(processed_items)} already processed, "
                f"{len(unprocessed_merged)} remaining",
                file=sys.stderr,
            )
            self.merged = unprocessed_merged

        # Step 24: Apply validation overrides
        self.apply_group_validation_overrides()

        # Apply per-suggestion validation overrides
        self.apply_suggestion_validation_overrides()

        # Step 25: Filter user-skipped + by status/importance
        self.user_skipped_items = []
        if (
            self.skipped_group_indices
            or self.skipped_suggestion_ids
            or self.old_skipped_ids
        ):
            for group_idx, group in enumerate(self.merged, 1):
                theme = group.get("theme", "")
                suggestions = group.get("suggestions", [])
                primary_title = (
                    suggestions[0].get("title", theme) if suggestions else theme
                )
                importance = get_highest_importance(group)
                ghash = group.get("group_hash", "")
                if ghash and ghash in self.skipped_group_indices:
                    self.user_skipped_items.append(
                        {
                            "title": primary_title,
                            "theme": theme,
                            "importance": importance,
                        }
                    )
                else:
                    for sugg_idx, sugg in enumerate(suggestions, 1):
                        shash = sugg.get("suggestion_hash", "")
                        positional_id = f"G{group_idx}S{sugg_idx}"
                        old_id = sugg.get("id", "")
                        skipped_ids_upper = {
                            s.upper() for s in self.skipped_suggestion_ids
                        }
                        is_skipped = (
                            (shash and shash in self.skipped_suggestion_ids)
                            or positional_id.upper() in skipped_ids_upper
                            or old_id in self.old_skipped_ids
                        )
                        if is_skipped:
                            self.user_skipped_items.append(
                                {
                                    "title": sugg.get("title", theme),
                                    "theme": theme,
                                    "importance": sugg.get(
                                        "importance", importance
                                    ),
                                }
                            )
            self.merged, user_skipped_count = filter_user_skipped_groups(
                self.merged,
                self.skipped_group_indices,
                self.skipped_suggestion_ids,
                self.old_skipped_ids,
            )
            if user_skipped_count > 0:
                print(
                    f"Filtered {user_skipped_count} user-skipped items",
                    file=sys.stderr,
                )

        # Safety check for --approve-all with HIGH importance
        if getattr(self.args, "approve_all", False) and not getattr(
            self.args, "include_high", False
        ):
            high_importance_count = sum(
                1
                for g in self.merged
                if get_highest_importance(g) == "HIGH"
                and g.get("validation_status")
                in ("needs-human-decision", "validation_failed")
            )
            if high_importance_count > 0:
                print(
                    f"ERROR: --approve-all would approve {high_importance_count} HIGH importance items.",
                    file=sys.stderr,
                )
                print(
                    "       Use --include-high to explicitly allow approving HIGH importance items.",
                    file=sys.stderr,
                )
                if getattr(self.args, "approve_validation_failed", False):
                    print(
                        "       Or use --approve-validation-failed to only approve recoverable failures.",
                        file=sys.stderr,
                    )
                raise OrchestratorError(
                    f"Phase _apply_user_feedback, step 25: --approve-all blocked by HIGH importance items",
                    exit_code=1,
                )

        # Filter items by validation status and importance
        min_priority = resolve_priority_args(self.args)
        self.valid, self.needs_human, self.skipped, dry_run_report = filter_items(
            groups=self.merged,
            min_priority=min_priority,
            skip_human_review=getattr(self.args, "skip_human_review", False),
            approve_all_human=getattr(self.args, "approve_all", False),
            approve_low_human=getattr(self.args, "approve_all_low", False),
            approve_importance_levels=getattr(self.args, "approve_importance", None),
            approve_validation_failed=getattr(
                self.args, "approve_validation_failed", False
            ),
            skip_all_human=getattr(self.args, "skip_all_human", False),
            dry_run=getattr(self.args, "dry_run", False),
        )

        print(f"\nFiltered {self.item_noun}s:", file=sys.stderr)
        print(f"  Valid (to apply): {len(self.valid)}", file=sys.stderr)
        print(f"  Needs human review: {len(self.needs_human)}", file=sys.stderr)
        print(f"  Skipped: {len(self.skipped)}", file=sys.stderr)

        # Show dry-run bulk approval report
        if getattr(self.args, "dry_run", False) and dry_run_report.get(
            "would_auto_approve"
        ):
            print("\n[dry-run] Would auto-approve:", file=sys.stderr)
            for item in dry_run_report["would_auto_approve"][:5]:
                print(
                    f"  - [{item['importance']}] "
                    f"{item.get('theme', 'Group ' + str(item['group_index']))}: "
                    f"{item['reason']}",
                    file=sys.stderr,
                )
            if len(dry_run_report["would_auto_approve"]) > 5:
                print(
                    f"  ... and {len(dry_run_report['would_auto_approve']) - 5} more",
                    file=sys.stderr,
                )

        # Build formatted_skipped for output
        self.formatted_skipped = []
        for g in self.skipped:
            suggestions = g.get("suggestions", [])
            title = (
                suggestions[0].get("title", g.get("theme", "Unknown"))
                if suggestions
                else g.get("theme", "Unknown")
            )
            importance = get_highest_importance(g)
            status = g.get("validation_status", "unknown")
            reason = g.get("validation_reason", "")
            if status == "invalid":
                skip_status = "invalid"
            elif g.get("_bulk_skipped"):
                skip_status = "bulk_skipped"
            elif status == "valid":
                skip_status = "below_priority"
            else:
                skip_status = status
            self.formatted_skipped.append(
                {
                    "title": title,
                    "theme": g.get("theme", ""),
                    "importance": importance,
                    "status": skip_status,
                    "reason": reason,
                }
            )

        # Step 27: handle_no_items_early_exit
        if not self.valid and not self.needs_human:
            self.handle_no_items_early_exit()

        self.logger.info(
            "User feedback applied: %d valid, %d needs_human, %d skipped",
            len(self.valid),
            len(self.needs_human),
            len(self.skipped),
        )

    # ==================================================================
    # Phase 4 — _prepare_batches
    # ==================================================================

    def _prepare_batches(self) -> None:
        """Phase 4: Formatting, batching, and prompt generation.

        Steps 28-31 from the lifecycle spec.
        """
        self.logger.debug("Entering _prepare_batches phase")

        # Step 28: Format items for output
        try:
            self.formatted_valid = [
                self.format_item_for_output(g, i)
                for i, g in enumerate(self.valid)
            ]
            self.formatted_human = [
                self.format_item_for_output(g, i)
                for i, g in enumerate(self.needs_human)
            ]
        except Exception as e:
            raise OrchestratorError(
                f"Phase _prepare_batches, step 28: format_item_for_output failed: {e}",
                exit_code=1,
            ) from e

        # Step 29: Create batches
        try:
            self.batches = self.create_batches(self.formatted_valid)
        except Exception as e:
            raise OrchestratorError(
                f"Phase _prepare_batches, step 29: create_batches failed: {e}",
                exit_code=1,
            ) from e

        self.logger.info("Created %d batches", len(self.batches))

        # Step 30: Handle --dry-run
        if getattr(self.args, "dry_run", False):
            self._handle_dry_run()
            sys.exit(0)

        # Step 31: Generate prompts
        try:
            self.batches = self.generate_batch_prompts(self.batches)
        except Exception as e:
            raise OrchestratorError(
                f"Phase _prepare_batches, step 31: generate_batch_prompts failed: {e}",
                exit_code=1,
            ) from e

        self.logger.info("Batch preparation complete")

    # ==================================================================
    # Phase 5 — _write_outputs
    # ==================================================================

    def _write_outputs(self) -> int:
        """Phase 5: Output assembly and completion.

        Steps 32-35 from the lifecycle spec.

        Returns:
            Exit code (0 on success).
        """
        self.logger.debug("Entering _write_outputs phase")

        # Build resume_info
        resume_info: Optional[Dict[str, Any]] = None
        if self.state is not None:
            resume_info = {
                "previously_processed": list(
                    self.state.get_processed_items(self.phase_name).keys()
                ),
                "previous_decisions": self.state.get_all_human_decisions(
                    self.phase_name
                ),
                "can_resume": (
                    len(self.state.get_processed_items(self.phase_name)) > 0
                    or len(self.state.get_all_human_decisions(self.phase_name)) > 0
                ),
            }

        # Post-hoc audit: verify every report-pre-marked "Let Claude decide"
        # item actually reached the judge (decision_source == claude_*), rather
        # than being prompted interactively or dropped. This is the runtime
        # backstop for the per-item routing the apply instructions perform; it
        # surfaces warnings but never fails the run (a sibling of the
        # batch-mode-compliance audit). Recorded decisions only exist once at
        # least one batch has been processed (e.g. on --resume), so this is a
        # no-op on the initial pass when nothing has been recorded yet.
        if self.state is not None:
            recorded_decisions = self.state.get_all_human_decisions(
                self.phase_name
            )
            cd_ok, cd_warnings = validate_claude_decide_items_honored(
                self.build_human_review_config(),
                recorded_decisions,
            )
            if not cd_ok:
                for warning in cd_warnings:
                    self.logger.warning(warning)

        # Step 32: Build output JSON
        try:
            output = self.build_output_json(
                self.batches,
                resume_info=resume_info,
            )
        except Exception as e:
            raise OrchestratorError(
                f"Phase _write_outputs, step 32: build_output_json failed: {e}",
                exit_code=1,
            ) from e

        # Step 33: Write output + mark phase completed
        try:
            output_path = self.get_output_path()
        except Exception as e:
            raise OrchestratorError(
                f"Phase _write_outputs, step 33: get_output_path failed: {e}",
                exit_code=1,
            ) from e

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"[OUTPUT_FILE] {output_path}", file=sys.stderr)

        if self.marks_phase_completed and self.state is not None:
            self.state.mark_phase_completed(self.phase_name)
            self.state.save()

        # Step 34: Print summary
        try:
            self.print_text_summary(self.batches, output_path)
        except Exception as e:
            # Non-fatal: log but don't fail
            self.logger.warning(
                "print_text_summary failed: %s", e, exc_info=True
            )

        self.logger.info("Output written to %s", output_path)

        # Step 35: Return exit code 0
        return 0

    # ==================================================================
    # Abstract methods (subclass must implement)
    # ==================================================================

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load grouped items and validation results from the phase subdirectory.

        Returns:
            Tuple of (groups, validation) where groups is the list of
            grouped items and validation is the list of validation results.

        Raises:
            OrchestratorError: If required data files are missing or malformed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement load_data()"
        )

    def parse_user_edits(
        self, report_path: str
    ) -> Dict[str, Tuple[str, str]]:
        """Parse user-edited descriptions from the review report markdown.

        Args:
            report_path: Path to the report.md file.

        Returns:
            Dict mapping item ID to (original_description, edited_description).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_user_edits()"
        )

    def merge_user_edits(
        self,
        groups: List[Dict[str, Any]],
        edited_descriptions: Dict[str, Tuple[str, str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply user-edited descriptions to group items.

        Args:
            groups: List of group dicts (deep-copied internally).
            edited_descriptions: Mapping from parse_user_edits().

        Returns:
            Tuple of (updated_groups, edit_log).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement merge_user_edits()"
        )

    def parse_skips_from_report(
        self, report_path: str
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Parse user-skipped item identifiers from the review report.

        Args:
            report_path: Path to the report.md file.

        Returns:
            Tuple of (skipped_group_indices, skipped_suggestion_ids,
            old_skipped_ids). Each is a set of string identifiers.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_skips_from_report()"
        )

    def parse_validation_overrides_from_report(
        self, report_path: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Parse validation status overrides from the review report.

        Args:
            report_path: Path to the report.md file.

        Returns:
            Tuple of (group_overrides, suggestion_overrides). Each maps
            item ID to override status string.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_validation_overrides_from_report()"
        )

    def format_item_for_output(
        self, group: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """Format a single group for inclusion in output JSON.

        Args:
            group: A merged group dict (with validation fields).
            index: Zero-based index in the filtered list.

        Returns:
            Dict with phase-specific fields ready for JSON serialization.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement format_item_for_output()"
        )

    def create_batches(
        self, items: List[Dict[str, Any]]
    ) -> List[Any]:
        """Partition formatted items into SuggestionBatch batches.

        Default implementation uses ``SuggestionBatch`` and
        ``group_suggestions_for_subagents``.  Subclasses that need a
        different batch type (e.g. ``CodeFixBatch``) should override.

        Args:
            items: List of formatted item dicts.

        Returns:
            List of SuggestionBatch objects (or subclass-specific batch type).
        """
        if getattr(self.args, "no_batch", False):
            batches: List[Any] = [
                SuggestionBatch(
                    suggestions=[s],
                    section_key=s.get("reference", "unknown"),
                    batch_type=s.get("type", "modification"),
                    total_chars=len(s.get("description", "")),
                )
                for s in items
            ]
            self.batching_stats = {
                "total_suggestions": len(items),
                "total_batches": len(items),
                "subagent_calls_saved": 0,
                "efficiency_gain_percent": 0,
                "batching_enabled": False,
            }
        else:
            batches = group_suggestions_for_subagents(
                items,
                max_per_batch=getattr(self.args, "max_batch_size", 4),
            )
            self.batching_stats = estimate_batch_processing_stats(batches)
            self.batching_stats["batching_enabled"] = True

        print("\nBatching results:", file=sys.stderr)
        print(f"  Total suggestions: {self.batching_stats.get('total_suggestions', 0)}", file=sys.stderr)
        print(f"  Total batches: {self.batching_stats.get('total_batches', 0)}", file=sys.stderr)
        if self.batching_stats.get("batching_enabled"):
            print(f"  Subagent calls saved: {self.batching_stats.get('subagent_calls_saved', 0)}", file=sys.stderr)
            print(f"  Efficiency gain: {self.batching_stats.get('efficiency_gain_percent', 0)}%", file=sys.stderr)

        return batches

    def generate_batch_prompts(
        self, batches: List[Any]
    ) -> List[Any]:
        """Attach prompt text to each batch for subagent consumption.

        Args:
            batches: List of batch objects from create_batches().

        Returns:
            The same batch list with prompt fields populated.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement generate_batch_prompts()"
        )

    def build_output_json(
        self,
        batches: List[Any],
        *,
        resume_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble the final orchestrator_output.json payload.

        Args:
            batches: List of batch objects with prompts attached.
            resume_info: Optional resume state from a prior interrupted run.

        Returns:
            Dict representing the complete output JSON structure.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_output_json()"
        )

    def get_output_path(self) -> str:
        """Return the filesystem path for orchestrator_output.json.

        Returns:
            Absolute path where the output JSON should be written.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_output_path()"
        )

    def print_text_summary(
        self,
        batches: List[Any],
        output_path: str,
    ) -> None:
        """Print a human-readable summary of the orchestrator results.

        Args:
            batches: List of batch objects that were written.
            output_path: Path where orchestrator_output.json was saved.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement print_text_summary()"
        )

    # ==================================================================
    # Shared helpers for subclass use
    # ==================================================================

    def build_human_review_config(self) -> Dict[str, Any]:
        """Build the human_review_config dict from formatted_human items.

        This is identical across all three orchestrator subclasses, so it
        lives here to avoid triplication.
        """
        by_importance: Dict[str, List[Dict[str, Any]]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for item in self.formatted_human:
            imp = item.get("importance", "MEDIUM").upper()
            if imp in by_importance:
                by_importance[imp].append(item)

        mode = getattr(self.args, "batch_review_mode", "by-importance")
        claude_decide = bool(getattr(self.args, "claude_decide", False))
        return {
            "mode": mode,
            "batch_enabled": mode != "individual",
            # When "claude_auto_decide", Claude Code skips the interactive
            # batch-review prompt and evaluates each needs-human-decision item
            # itself ("Let Claude Decide" mode). See
            # references/human-decision-batch.md.
            "decision_mode": "claude_auto_decide" if claude_decide else "interactive",
            "prompt_strategy": "single_batch" if mode != "individual" else "per_item",
            "by_importance": by_importance,
            "total_count": len(self.formatted_human),
            "batch_prompt_template": "batch_approval_v1",
            # Convenience index of items the reviewer pre-marked "Let Claude
            # decide" in the report. The authoritative per-item signal is the
            # per-item ``decision_mode == "claude_auto_decide"`` flag the
            # formatters emit; this list lets the instructions find them without
            # scanning every item. The global ``decision_mode`` above stays
            # "interactive" unless --claude-decide was passed; per-item routing
            # is purely additive.
            "claude_decide_item_ids": [
                it.get("group_id", "")
                for it in self.formatted_human
                if it.get("decision_mode") == "claude_auto_decide"
            ],
        }

    # ==================================================================
    # Optional hooks (override if needed)
    # ==================================================================

    def pre_validate_hook(self) -> None:
        """Hook called during _setup before output directory validation.

        Subclasses can use this to discover additional files (e.g.,
        tasks.md), check prerequisites (e.g., prior phase completion),
        or set up instance state needed by later phases.

        The default implementation is a no-op.
        """
        pass

    def post_load_hook(self) -> None:
        """Hook called after load_data() in _load_and_parse_inputs.

        Subclasses can use this to build auxiliary data structures
        (e.g., issue-to-group map for code_fixes).

        The default implementation is a no-op.
        """
        pass

    def resolve_base_ref(self) -> str:
        """Return the git base ref used for revalidation and batch prompts.

        Called by ``handle_revalidation()`` and (in the code_fixes
        subclass) ``generate_batch_prompts()``.  The default returns an
        empty string, which is correct for plan-review orchestrators
        that have no git diff context.  The code_fixes subclass
        overrides this to resolve ``--base-ref`` / ``state.head_at_start``.

        Returns:
            A validated git ref string, or empty string.
        """
        return ""

    def _route_claude_decide_marker(
        self, group: Dict[str, Any], key: Any, value: str
    ) -> bool:
        """Handle a ``claude_decide`` override as a routing marker, not a status.

        Returns ``True`` if *value* was a ``claude_decide`` marker (the caller
        should skip the ``validation_status`` assignment), ``False`` otherwise.

        A ``claude_decide`` marker keeps the item in ``needs_human`` and tags it
        so the per-item "Let Claude Decide" judge picks it up at apply time; it
        must NEVER be written into ``validation_status`` (which is not a real
        status and would silently fall through ``filter_items``).
        """
        if value == "claude_decide":
            group["claude_decide"] = True
            # Diagnostics-only mirror; routing is driven by group["claude_decide"]
            # above, not by this set (see declaration note in __init__).
            self.claude_decide_overrides.add(key)
            # A surfaced reason so the routing is visible in logs/reports. Not a
            # setdefault no-op — the item already carries a needs-human reason.
            group["validation_reason"] = "Routed to Claude by reviewer"
            return True
        return False

    def apply_group_validation_overrides(self) -> None:
        """Apply group-level validation overrides to merged groups.

        Default implementation: hash-based string key matching.
        Keys in ``self.validation_overrides`` are matched against
        the ``group_hash`` field of each merged group.

        The ``apply_code_fixes`` subclass overrides this to match on
        integer issue-number keys instead.
        """
        if not self.validation_overrides:
            return

        override_count = 0
        for group in self.merged:
            ghash = group.get("group_hash", "")
            if ghash and ghash in self.validation_overrides:
                value = self.validation_overrides[ghash]
                if value not in VALID_OVERRIDE_VALUES:
                    print(
                        f"WARNING: ignoring unknown validation override "
                        f"{value!r} for group [{str(ghash)[:8]}]; leaving its "
                        f"underlying status unchanged.",
                        file=sys.stderr,
                    )
                    continue
                # Route "claude_decide" as a marker, never as a status.
                if self._route_claude_decide_marker(group, ghash, value):
                    override_count += 1
                    continue
                old_status = group.get("validation_status", "unknown")
                group["validation_status"] = value
                group["validation_reason"] = f"User override (was {old_status})"
                group["user_override"] = True
                override_count += 1
        if override_count:
            print(
                f"Applied {override_count} user validation overrides",
                file=sys.stderr,
            )

    def apply_suggestion_validation_overrides(self) -> None:
        """Apply per-suggestion validation overrides to ``self.merged``.

        ``invalid`` drops the suggestion; ``valid`` flags it (and promotes the
        group if *all* survivors are valid); ``claude_decide`` is a routing
        marker that keeps the suggestion and, after the loop, routes the whole
        containing group to the per-item judge (the apply-time review unit is a
        group, so the marker is propagated up — including the mixed-group case
        where only one suggestion is marked). Unknown values are warned and
        ignored.
        """
        if not self.suggestion_validation_overrides:
            return

        sugg_override_count = 0
        for idx, group in enumerate(self.merged, 1):
            suggestions = group.get("suggestions", [])
            modified = False
            for sugg_idx, sugg in enumerate(suggestions, 1):
                shash = sugg.get("suggestion_hash", "")
                positional_id = f"G{idx}S{sugg_idx}"
                matched_key = None
                if shash and shash in self.suggestion_validation_overrides:
                    matched_key = shash
                elif positional_id in self.suggestion_validation_overrides:
                    matched_key = positional_id

                if matched_key is not None:
                    override_status = self.suggestion_validation_overrides[
                        matched_key
                    ]
                    if override_status not in VALID_OVERRIDE_VALUES:
                        print(
                            f"WARNING: ignoring unknown per-suggestion "
                            f"validation override {override_status!r} for "
                            f"{positional_id}; leaving it unchanged.",
                            file=sys.stderr,
                        )
                        continue
                    if override_status == "invalid":
                        sugg["_user_override_invalid"] = True
                    elif override_status == "valid":
                        sugg["_user_override_valid"] = True
                    elif override_status == "claude_decide":
                        # Routing marker: keep the suggestion (do NOT drop or
                        # promote) and tag it. Propagated to the group below so
                        # the whole group routes to the per-item judge.
                        sugg["claude_decide"] = True
                    sugg_override_count += 1
                    modified = True

            if modified:
                remaining = [
                    s
                    for s in suggestions
                    if not s.get("_user_override_invalid")
                ]
                group["suggestions"] = remaining
                # A "Let Claude decide" marker (set group-level by
                # apply_group_validation_overrides, or per-suggestion in the
                # loop above) is an affirmative "route to the judge" request and
                # must win over a "valid" promotion. Without this guard, a group
                # marked claude_decide whose surviving suggestions are all
                # individually "valid" would be promoted to "valid" while still
                # carrying claude_decide=True; filter_items only honours
                # claude_decide in the needs-human-decision branch, so the group
                # would auto-apply and bypass the judge.
                marked_claude_decide = group.get("claude_decide") or any(
                    s.get("claude_decide") for s in remaining
                )
                if (
                    not marked_claude_decide
                    and remaining
                    and all(s.get("_user_override_valid") for s in remaining)
                    and group.get("validation_status")
                    in ("needs-human-decision", "validation_failed")
                ):
                    old_status = group.get("validation_status", "unknown")
                    group["validation_status"] = "valid"
                    group["validation_reason"] = (
                        f"All suggestions individually marked valid by user "
                        f"(was {old_status})"
                    )
                    group["user_override"] = True

                # Reconcile per-suggestion -> group granularity: if any
                # surviving suggestion is "claude_decide", route the whole
                # group to the judge (the apply-time review unit is a group,
                # and the formatters only inspect the group). This also covers
                # the mixed-group case (one claude_decide suggestion alongside
                # other untouched needs-human suggestions).
                if any(
                    s.get("claude_decide")
                    for s in group.get("suggestions", [])
                ):
                    group["claude_decide"] = True
                    # Diagnostics-only mirror; the group["claude_decide"] flag
                    # above is what actually routes the group to the judge (see
                    # declaration note in __init__). This set is never read.
                    self.claude_decide_overrides.add(
                        group.get("group_hash", "")
                    )

        if sugg_override_count:
            print(
                f"Applied {sugg_override_count} per-suggestion validation overrides",
                file=sys.stderr,
            )

    def handle_revalidation(self) -> None:
        """Handle --revalidate flow if supports_revalidation is True.

        When ``supports_revalidation`` is ``False`` (e.g., for
        task_suggestions), this returns immediately.

        Checks the --revalidate, --revalidate-model, --revalidate-all-human,
        --internal-revalidation, and --dry-run CLI flags to determine the
        revalidation mode:

        1. **dry-run**: Print which items *would* be revalidated, then return
           (the caller continues with the normal flow).
        2. **internal-revalidation** (legacy): Run revalidation in-process via
           ``asyncio.run(revalidate_failed_items(...))``, save updated
           validation results, and continue.
        3. **default** (subagent delegation): Prepare batched revalidation
           tasks, write them to ``revalidation_tasks.json``, emit the
           ``[REVALIDATION_PENDING]`` / ``[REVALIDATION_BATCHES_PENDING]``
           marker, and ``sys.exit(0)`` so that Claude Code picks up the
           task file.
        """
        if not self.supports_revalidation:
            return

        revalidate = getattr(self.args, "revalidate", False)
        revalidate_model = getattr(self.args, "revalidate_model", None)

        if not revalidate and not revalidate_model:
            return

        print("\n--- REVALIDATION MODE ---", file=sys.stderr)

        review_dir = os.path.join(self.out_dir, self.review_subdir)
        validation_path = Path(review_dir) / "validation.json"
        include_all_human = getattr(self.args, "revalidate_all_human", False)
        dry_run = getattr(self.args, "dry_run", False)

        # ----- dry-run: report what would be revalidated -----
        if dry_run:
            items_to_revalidate = []
            for i, val in enumerate(self.validation):
                status = val.get("status", "needs-human-decision")
                error_type = val.get("error_type", "unknown")
                would_revalidate = (
                    status == "validation_failed"
                    or (
                        include_all_human
                        and status == "needs-human-decision"
                        and error_type != ERROR_TYPE_AMBIGUOUS
                    )
                )
                if would_revalidate:
                    items_to_revalidate.append(
                        {
                            "group_index": i,
                            "current_status": status,
                            "error_type": error_type,
                            "reason": val.get("reason", ""),
                        }
                    )

            print(
                f"\n[dry-run] Would revalidate {len(items_to_revalidate)} items:",
                file=sys.stderr,
            )
            for item in items_to_revalidate[:5]:
                print(
                    f"  - Group {item['group_index']}: "
                    f"{item['current_status']} ({item['error_type']})",
                    file=sys.stderr,
                )
            if len(items_to_revalidate) > 5:
                print(
                    f"  ... and {len(items_to_revalidate) - 5} more",
                    file=sys.stderr,
                )
            # Return without exiting — caller proceeds with normal flow
            return

        # Read context for revalidation prompts
        context = self.get_revalidation_context()
        model = revalidate_model or "auto"

        # ----- internal-revalidation (legacy in-process mode) -----
        if getattr(self.args, "internal_revalidation", False):
            print("Running internal revalidation...", file=sys.stderr)

            self.validation = asyncio.run(
                revalidate_failed_items(
                    groups=self.groups,
                    validation_results=self.validation,
                    context=context,
                    model=model,
                    include_all_human=include_all_human,
                )
            )

            save_validation_results(
                self.validation, validation_path, model=model
            )
            print(
                f"Updated validation saved to: {validation_path}",
                file=sys.stderr,
            )
            # Continue with the updated validation (no exit)
            return

        # ----- default: prepare for Claude Code subagent revalidation -----
        print(
            "Preparing batched revalidation for Claude Code subagent...",
            file=sys.stderr,
        )

        # Resolve base_ref via hook (non-empty for code_fixes, empty for others)
        base_ref = self.resolve_base_ref()
        if base_ref:
            print(
                f"Revalidation git base reference: {base_ref}",
                file=sys.stderr,
            )

        revalidation_tasks = prepare_batched_revalidation_tasks(
            groups=self.groups,
            validation_results=self.validation,
            context=context,
            output_dir=review_dir,
            plan_file=str(self.plan_path),
            include_all_human=include_all_human,
            model=model,
            orchestrator=os.path.basename(sys.argv[0])
            if sys.argv
            else f"{self.phase_name}_orchestrator.py",
            base_ref=base_ref,
        )

        if revalidation_tasks["items_to_revalidate"] == 0:
            print("No items need revalidation.", file=sys.stderr)
            return

        # Save revalidation task instructions
        revalidation_task_path = os.path.join(
            review_dir, "revalidation_tasks.json"
        )
        with open(revalidation_task_path, "w", encoding="utf-8") as f:
            json.dump(revalidation_tasks, f, indent=2)

        total_batches = revalidation_tasks.get("total_batches", 1)
        if total_batches == 1:
            print(f"\n[REVALIDATION_PENDING] {revalidation_task_path}")
        else:
            print(
                f"\n[REVALIDATION_BATCHES_PENDING] {revalidation_task_path}"
            )
            print(f"Batches: {total_batches}")

        print("\nRevalidation prepared for Claude Code subagent.")
        print(
            f"Items to revalidate: {revalidation_tasks['items_to_revalidate']}"
        )
        print(
            "\nAfter revalidation completes, run this orchestrator again "
            "without --revalidate"
        )
        sys.exit(0)  # Exit — Claude Code handles revalidation

    def get_revalidation_context(self) -> str:
        """Return the context string used during revalidation.

        Default returns plan file contents. Subclasses override to provide
        alternative context (e.g. diff_context.txt for code_fixes).

        Only called when ``supports_revalidation=True``.

        Returns:
            Context string for revalidation prompts.
        """
        try:
            with open(self.plan_path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, IOError):
            self.logger.warning(
                "Could not read plan file for revalidation context: %s",
                self.plan_path,
            )
            return ""

    def handle_no_items_early_exit(self) -> None:
        """Handle the case where no actionable items remain after filtering.

        Called when both ``self.valid`` and ``self.needs_human`` are empty.
        The default implementation is a no-op (suggestions has no early exit).

        Subclasses override to write empty output, log messages, mark
        the phase as skipped, and call ``sys.exit(0)``.
        """
        pass

    # ==================================================================
    # Shared base-class logic (not abstract, not hooks)
    # ==================================================================

    def load_html_selections(
        self,
        review_dir: str,
        edited_descriptions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load HTML-based user selections and apply edits.

        Step 16 of ``_load_and_parse_inputs()``.  Identical across all
        three orchestrators.

        Args:
            review_dir: Path to the review phase directory.
            edited_descriptions: Previously parsed edited descriptions
                (from report.md), used as fallback when HTML selections
                are absent.
        """
        html_selections = load_html_selections(
            Path(review_dir),
            groups=self.groups,
            plan_path=self.plan_path,
        )
        if not html_selections:
            return

        print(
            f"Found HTML selections from: {review_dir}/user_selections.json",
            file=sys.stderr,
        )

        # Merge with HTML taking precedence
        (
            self.skipped_group_indices,
            self.skipped_suggestion_ids,
            html_edited,
        ) = merge_selections(
            html_selections,
            self.skipped_group_indices,
            self.skipped_suggestion_ids,
            edited_descriptions or {},
        )

        # Apply HTML-edited descriptions if any
        if html_edited:
            # Build lookup for original descriptions (by hash and positional ID)
            groups_lookup: Dict[str, str] = {}
            for group_idx, group in enumerate(self.groups, start=1):
                for sugg_idx, sugg in enumerate(
                    group.get("suggestions", []), start=1
                ):
                    positional_id = f"G{group_idx}S{sugg_idx}"
                    groups_lookup[positional_id] = sugg.get("desc", "")
                    shash = sugg.get("suggestion_hash", "")
                    if shash:
                        groups_lookup[shash] = sugg.get("desc", "")

            html_edit_dict = {
                k: (groups_lookup.get(k, ""), v) for k, v in html_edited.items()
            }
            if html_edit_dict:
                self.groups, extra_log = self.merge_user_edits(
                    self.groups, html_edit_dict
                )
                self.edit_log.extend(extra_log)
                print(
                    f"Applied {len(extra_log)} HTML-edited descriptions",
                    file=sys.stderr,
                )

        # Merge validation overrides from HTML.
        #
        # Keys can arrive in three shapes:
        #   - "G{N}S{M}"  -> a per-suggestion override
        #   - a group_hash (current report format)
        #   - a bare 0-based group index, e.g. "0" == the first group as
        #     displayed (legacy report format)
        #
        # A bare index must be resolved to its group_hash *here*, while the
        # groups are in scope, so that both orchestrator families match by
        # hash downstream. If it isn't: the code-fixes matcher enumerates
        # groups 1-based (so index "0" is off-by-one and never matches) and
        # the suggestion orchestrators match on hash strings only (so an int
        # key never matches at all). Either way the user's override was
        # silently dropped. Resolving to a hash also makes the override
        # order-independent, consistent with the rest of the join logic.
        #
        # A group_hash is normally non-numeric but can rarely be all digits,
        # so check the known-hash set first — a hash is never mistaken for an
        # index.
        known_group_hashes = {
            g.get("group_hash", "")
            for g in self.groups
            if g.get("group_hash")
        }
        html_overrides = html_selections.get("validation_overrides", {})
        for key, value in html_overrides.items():
            key_str = str(key)
            if re.match(r"G\d+S\d+", key_str, re.IGNORECASE):
                self.suggestion_validation_overrides[key_str.upper()] = value
            elif key_str in known_group_hashes:
                # Current report format: already keyed by group_hash.
                self.validation_overrides[key_str] = value
            elif key_str.isdigit() and int(key_str) < len(self.groups):
                # Legacy report format: 0-based group index -> group_hash.
                ghash = self.groups[int(key_str)].get("group_hash", "")
                if ghash:
                    self.validation_overrides[ghash] = value
                else:
                    print(
                        f"WARNING: HTML validation override for group index "
                        f"{key_str} has no group_hash; falling back to a "
                        f"positional key (may not apply).",
                        file=sys.stderr,
                    )
                    self.validation_overrides[int(key_str)] = value
            else:
                # Unrecognized: an out-of-range index, or a hash from a
                # different grouping. Preserve the original key (as int when
                # numeric, so the legacy code-fixes integer-key path keeps
                # working) rather than dropping the override outright.
                if key_str.isdigit():
                    self.validation_overrides[int(key_str)] = value
                else:
                    self.validation_overrides[key_str] = value

    def load_and_apply_consolidated_decisions(
        self, review_dir: str
    ) -> None:
        """Load and apply C-level consolidated decisions.

        Step 17 of ``_load_and_parse_inputs()``.  Identical across all
        three orchestrators.

        Args:
            review_dir: Path to the review phase directory.
        """
        c_level_skips, c_level_overrides = load_merged_suggestions(
            phase_dir=review_dir,
            groups=self.groups,
            plan_file=self.plan_path,
            accept_stale=getattr(
                self.args, "accept_stale_consolidation", False
            ),
        )

        if not c_level_skips and not c_level_overrides:
            return

        # Convert C-level 0-based indices to group hashes for merging
        for idx_0based in c_level_skips:
            if 0 <= idx_0based < len(self.groups):
                ghash = self.groups[idx_0based].get("group_hash", "")
                if ghash:
                    self.skipped_group_indices.add(ghash)
                else:
                    print(
                        f"WARNING: C-level skip for group index {idx_0based} has no group_hash",
                        file=sys.stderr,
                    )

        # C-level validation overrides
        for idx_0based, override_status in c_level_overrides.items():
            if 0 <= idx_0based < len(self.groups):
                ghash = self.groups[idx_0based].get("group_hash", "")
                if ghash:
                    if ghash not in self.validation_overrides:
                        self.validation_overrides[ghash] = override_status
                else:
                    print(
                        f"WARNING: C-level override for group index {idx_0based} has no group_hash",
                        file=sys.stderr,
                    )

        c_total = len(c_level_skips) + len(c_level_overrides)
        print(
            f"Applied {c_total} consolidated (C-level) decisions "
            f"({len(c_level_skips)} skips, {len(c_level_overrides)} overrides)",
            file=sys.stderr,
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _handle_dry_run(self) -> None:
        """Print dry-run batch summary and return.

        Called from ``_prepare_batches`` when ``--dry-run`` is active.
        """
        print("\n--- DRY RUN ---", file=sys.stderr)
        print("\nWould apply the following batches:", file=sys.stderr)
        for i, batch in enumerate(self.batches):
            # Try to get batch info generically
            section = getattr(batch, "section_key", None) or getattr(
                batch, "file_key", None
            ) or "various"
            count = getattr(batch, "size", 0)
            batch_type = getattr(batch, "batch_type", "unknown")
            print(
                f"  Batch {i + 1}: {count} {self.item_noun}(s) [{batch_type}] in {section}",
                file=sys.stderr,
            )
            items = getattr(batch, "suggestions", None) or getattr(
                batch, "fixes", None
            ) or []
            for s in items:
                print(
                    f"    - [{s.get('importance', 'MEDIUM')}] {s.get('title', 'Unknown')}",
                    file=sys.stderr,
                )

        if self.needs_human:
            print("\nWould ask for human review:", file=sys.stderr)
            for group in self.needs_human:
                theme = group.get("theme", group.get("title", "Unknown"))
                reason = group.get("validation_reason", "No reason")
                print(f"  - {theme}: {reason}", file=sys.stderr)
