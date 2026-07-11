#!/usr/bin/env python3
"""
Orchestrator for applying validated task suggestions to a tasks.md file.

This script reads validation results from a task review and outputs
the list of suggestions that should be applied. The actual application
is handled by Claude Code using Task subagents sequentially.

Usage:
    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file plans/my-plan.md [options]
"""

import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.stream_bootstrap import bootstrap_streams
from utils.apply_orchestrator_base import (
    ApplyOrchestratorBase,
    OrchestratorError,
    build_common_arg_parser,
)
from utils.apply_output_helpers import (
    build_skipped_output,
    emit_json_output,
    write_and_emit_output,
)
from utils.apply_path_helpers import load_json_file
from utils.apply_selection_helpers import merge_edited_descriptions
from utils.report_parser import (
    find_edited_descriptions,
    parse_skipped_groups,
    parse_skipped_group_suggestions,
    parse_skipped_suggestions,
    parse_suggestion_validation_overrides,
    parse_validation_overrides_groups,
)
from utils.state_manager import (
    CURRENT_FORMAT_VERSION,
    StateManager,
    load_groups_payload,
    stamp_stable_ids,
)
from utils.suggestion_batcher import (
    SuggestionBatch,
)
from utils.tasks_file import find_tasks_file


class ApplyTaskSuggestionsOrchestrator(ApplyOrchestratorBase[Dict[str, Any], SuggestionBatch]):
    """Thin subclass for the apply-task-suggestions phase."""

    phase_name = "apply-task-suggestions"
    review_subdir = "review-tasks"
    item_noun = "task suggestion"
    supports_revalidation = False
    supports_skip_flag = True
    marks_phase_completed = False
    guard_double_nesting = False

    # Instance state populated by pre_validate_hook
    tasks_file: str = ""
    tasks_content: str = ""

    # ----------------------------------------------------------------
    # Argument parser
    # ----------------------------------------------------------------

    @classmethod
    def parse_args(cls):
        """Build and parse CLI arguments for apply-task-suggestions."""
        parser = build_common_arg_parser(
            description="Prepare validated task suggestions for application to tasks.md",
            epilog="""
This orchestrator outputs a JSON list of task suggestions to apply.
The actual application is handled by Claude Code using Task subagents.

Example:
  uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file plans/my-plan.md

Bulk Approval Examples:
  # Auto-approve all LOW importance items
  uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file plans/my-plan.md --approve-all-low

  # Skip all items requiring human review
  uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/apply_task_suggestions_orchestrator.py --plan-file plans/my-plan.md --skip-all-human
        """,
            include_skip=True,
            include_output_format=True,
            include_mark_completed=True,
        )
        return parser.parse_args()

    # ----------------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------------

    def pre_validate_hook(self) -> None:
        """Discover tasks_file and check prerequisite phases."""
        # Find tasks file
        try:
            self.tasks_file = find_tasks_file(self.plan_path)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            raise OrchestratorError(str(e), exit_code=1) from e
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            raise OrchestratorError(str(e), exit_code=1) from e

        print(f"Tasks file: {self.tasks_file}", file=sys.stderr)

        # Read tasks content for prompt generation
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            self.tasks_content = f.read()

        # Prerequisite check: review-tasks must be completed or skipped
        state = StateManager(Path(self.plan_path))
        if not state.is_phase_completed("review-tasks") and not state.is_phase_skipped("review-tasks"):
            print("ERROR: review-tasks phase has not been completed.", file=sys.stderr)
            print("       Run --review-tasks first before applying task suggestions.", file=sys.stderr)
            raise OrchestratorError(
                "review-tasks phase has not been completed", exit_code=1
            )

    def handle_no_items_early_exit(self) -> None:
        """Write empty output and mark phase skipped when no items remain."""
        # Determine skip reason
        if self.user_skipped_items:
            skip_reason = "all suggestions skipped by user"
            skip_message = "All task suggestions were skipped by user selections"
        else:
            skip_reason = "no actionable findings from review-tasks"
            skip_message = "No actionable task suggestions remain after filtering"

        # Dry runs stay read-only
        if not getattr(self.args, "dry_run", False) and self.state is not None:
            self.state.mark_phase_skipped(self.phase_name, skip_reason)
            self.state.save()

        # Write empty orchestrator_output.json
        phase_dir = os.path.join(self.out_dir, self.phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        output = {
            "status": "skipped",
            "message": skip_message,
            "phase": self.phase_name,
            "batches": [],
            "to_apply": [],
            "needs_human_review": [],
            "skipped_items": self.formatted_skipped,
            "user_skipped_items": self.user_skipped_items,
            "skipped_count": len(self.skipped),
            "user_skipped_count": len(self.user_skipped_items),
            "summary": {
                "total_groups": len(self.groups),
                "valid_count": 0,
                "needs_human_count": 0,
                "skipped_count": len(self.skipped),
                "user_skipped_count": len(self.user_skipped_items),
                "batch_count": 0,
            },
        }
        write_and_emit_output(output, phase_dir)
        emit_json_output(output)
        sys.exit(0)

    # ----------------------------------------------------------------
    # Abstract method implementations
    # ----------------------------------------------------------------

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load grouped suggestions and validation from review-tasks subdir."""
        review_dir = os.path.join(self.out_dir, self.review_subdir)
        validation_path = Path(review_dir) / "validation.json"

        # Load grouped suggestions
        grouped_path = os.path.join(review_dir, "grouped.json")
        raw = load_json_file(grouped_path)
        groups: Optional[List[Dict[str, Any]]] = None
        if raw is not None:
            groups = load_groups_payload(raw)
            stamp_stable_ids(groups)

        # Handle zero-finding path
        if not validation_path.exists():
            if groups is None or len(groups) == 0:
                skip_reason = "no actionable findings from review-tasks"
                print(
                    "No actionable task suggestions found — skipping apply-task-suggestions phase.",
                    file=sys.stderr,
                )
                if self.state is not None:
                    self.state.mark_phase_skipped(self.phase_name, skip_reason)
                    self.state.save()
                # Write empty output
                phase_dir = os.path.join(self.out_dir, self.phase_name)
                output = build_skipped_output(
                    self.phase_name,
                    "No actionable task suggestions found — skipping apply-task-suggestions phase.",
                )
                write_and_emit_output(output, phase_dir)
                emit_json_output(output)
                sys.exit(0)
            else:
                raise OrchestratorError(
                    f"Validation results not found at {validation_path}. "
                    "Run --review-tasks first to generate validation results.",
                    exit_code=1,
                )

        # Load validation using v2 loader
        from utils.validation import load_validation_results as load_validation_v2

        validation = load_validation_v2(validation_path)

        if groups is None:
            raise OrchestratorError(
                f"Grouped suggestions not found in {self.out_dir}. "
                "Run --review-tasks first to generate grouped suggestions.",
                exit_code=1,
            )

        return groups, validation

    def parse_user_edits(self, report_path: str) -> Dict[str, Tuple[str, str]]:
        """Detect and return user-edited descriptions from report.md."""
        return find_edited_descriptions(report_path, self.groups)

    def merge_user_edits(
        self,
        groups: List[Dict[str, Any]],
        edited_descriptions: Dict[str, Tuple[str, str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Delegate to shared merge_edited_descriptions helper."""
        return merge_edited_descriptions(groups, edited_descriptions)

    def parse_skips_from_report(
        self, report_path: str
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Parse skipped groups and suggestions from report.md."""
        skipped_group_indices = parse_skipped_groups(report_path)
        skipped_suggestion_ids = parse_skipped_group_suggestions(report_path)
        old_skipped_ids = parse_skipped_suggestions(report_path)
        return skipped_group_indices, skipped_suggestion_ids, old_skipped_ids

    def parse_validation_overrides_from_report(
        self, report_path: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Parse validation overrides from report.md."""
        group_overrides = parse_validation_overrides_groups(report_path)
        suggestion_overrides = parse_suggestion_validation_overrides(report_path)
        return group_overrides, suggestion_overrides

    def format_item_for_output(self, group: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a suggestion group for output, including task_reference field."""
        suggestions = group.get("suggestions", [])

        if suggestions:
            primary = suggestions[0]
            title = primary.get("title", group.get("theme", "Unknown"))
            desc = primary.get("desc", "")
            suggestion_type = primary.get("type", "modification")
            reference = primary.get("reference", "")
            importance = primary.get("importance", "MEDIUM")
        else:
            title = group.get("theme", "Unknown")
            desc = ""
            suggestion_type = "modification"
            reference = ""
            importance = "MEDIUM"

        # Derive task_reference
        task_reference = ""
        if reference:
            task_match = re.search(r"(T\d+)", reference)
            if task_match:
                task_reference = f"Task {task_match.group(1)}"
            else:
                task_reference = reference

        # Combine descriptions if multiple suggestions
        if len(suggestions) > 1:
            all_descs = [s.get("desc", "") for s in suggestions if s.get("desc")]
            desc = "\n\n".join(all_descs)

        formatted = {
            "index": index,
            # Stable group identifier so judging subagents / human-decision
            # routing can reference pre-marked items.
            "group_id": group.get("group_hash", ""),
            "title": title,
            "description": desc,
            "type": suggestion_type,
            "reference": reference,
            "task_reference": task_reference,
            "importance": importance,
            "theme": group.get("theme", ""),
            "category": group.get("category", ""),
            "validation_status": group.get("validation_status", "valid"),
            "validation_reason": group.get("validation_reason", ""),
            "validation_confidence": group.get("validation_confidence", 0.0),
            "models": group.get("models", []),
            "suggestion_count": len(suggestions),
        }
        # Per-item routing flag: reviewer pre-marked this group "Let Claude
        # decide" in the report. Routed to the per-item judge without prompting.
        if group.get("claude_decide"):
            formatted["decision_mode"] = "claude_auto_decide"
        return formatted

    def generate_batch_prompts(self, batches: List[SuggestionBatch]) -> List[Dict[str, Any]]:
        """Attach task-specific prompts to each batch.

        Returns a list of dicts (batch.to_dict() + prompt) for output assembly.
        """
        result = []
        for batch in batches:
            prompt = self._generate_single_batch_prompt(batch)
            result.append({**batch.to_dict(), "prompt": prompt})
        return result

    def build_output_json(
        self,
        batches: List[Any],
        *,
        resume_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble the final orchestrator_output.json payload."""
        phase_dir = os.path.join(self.out_dir, self.phase_name)

        # Backup tasks.md before any modifications
        tasks_backup_path = os.path.join(phase_dir, "tasks-backup.md")
        os.makedirs(phase_dir, exist_ok=True)
        shutil.copy2(self.tasks_file, tasks_backup_path)
        print(f"Tasks file backup: {tasks_backup_path}", file=sys.stderr)

        return {
            "format_version": CURRENT_FORMAT_VERSION,
            "plan_file": self.plan_path,
            "tasks_file": self.tasks_file,
            "tasks_file_backup": tasks_backup_path,
            "prefix": self.prefix,
            "output_dir": self.out_dir,
            "apply_task_suggestions_dir": phase_dir,
            "timestamp": datetime.now().isoformat(),
            "batches": batches,
            "to_apply": self.formatted_valid,
            "needs_human_review": self.formatted_human,
            "skipped_items": self.formatted_skipped,
            "user_skipped_items": self.user_skipped_items,
            "skipped_count": len(self.skipped),
            "batching_stats": self.batching_stats,
            "human_review_config": self.build_human_review_config(),
            "resume_info": resume_info,
            "edited_descriptions": self.edit_log,
            "summary": {
                "total_groups": len(self.groups),
                "valid_count": len(self.valid),
                "needs_human_count": len(self.needs_human),
                "skipped_count": len(self.skipped),
                "user_skipped_count": len(self.user_skipped_items),
                "batch_count": len(batches),
                "validation_failed_count": sum(
                    1 for g in self.merged if g.get("validation_status") == "validation_failed"
                ),
                "auto_approved_count": sum(
                    1 for g in self.valid if g.get("auto_approved", False)
                ),
                "edited_description_count": len(self.edit_log),
            },
        }

    def get_output_path(self) -> str:
        """Return path for orchestrator_output.json."""
        phase_dir = os.path.join(self.out_dir, self.phase_name)
        return os.path.join(phase_dir, "orchestrator_output.json")

    def print_text_summary(self, batches: List[Any], output_path: str) -> None:
        """Print human-readable summary of results."""
        print(f"\n{'=' * 60}")
        print("TASK SUGGESTIONS TO APPLY (BATCHED)")
        print(f"{'=' * 60}")
        print(f"Plan: {self.plan_path}")
        print(f"Tasks file: {self.tasks_file}")
        print(f"Valid suggestions: {len(self.valid)}")
        print(f"Batches: {len(batches)}")
        print(f"Needs human review: {len(self.needs_human)}")
        print(f"Skipped: {len(self.skipped)}")

        if self.batching_stats.get("batching_enabled"):
            print("\nBatching efficiency:")
            print(f"  Subagent calls saved: {self.batching_stats.get('subagent_calls_saved', 0)}")
            print(f"  Efficiency gain: {self.batching_stats.get('efficiency_gain_percent', 0)}%")

        for i, batch in enumerate(batches):
            batch_suggestions = batch.get("suggestions", []) if isinstance(batch, dict) else batch.suggestions
            batch_section = batch.get("section_key", "") if isinstance(batch, dict) else batch.section_key
            batch_type = batch.get("batch_type", "") if isinstance(batch, dict) else batch.batch_type
            batch_size = batch.get("suggestion_count", len(batch_suggestions)) if isinstance(batch, dict) else batch.size
            batch_priority = batch.get("priority_score", 0) if isinstance(batch, dict) else batch.priority_score

            print(f"\n{'=' * 60}")
            print(f"BATCH {i + 1} ({batch_size} suggestion(s))")
            print(f"{'=' * 60}")
            print(f"Section: {batch_section}")
            print(f"Type: {batch_type}")
            print(f"Priority score: {batch_priority}")

            for j, s in enumerate(batch_suggestions):
                print(f"\n  --- Suggestion {j + 1} ---")
                print(f"  Title: {s['title']}")
                print(f"  Type: {s['type']}")
                print(f"  Importance: {s['importance']}")
                print(f"  Task reference: {s.get('task_reference', s['reference'])}")
                print(f"  Description:\n  {s['description'][:200]}...")

        if self.formatted_human:
            print(f"\n{'=' * 60}")
            print("NEEDS HUMAN REVIEW")
            print(f"{'=' * 60}")
            for s in self.formatted_human:
                print(f"\n- {s['title']}")
                print(f"  Reason: {s['validation_reason']}")

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _generate_single_batch_prompt(self, batch: SuggestionBatch) -> str:
        """Generate a task-specific prompt for a single batch."""
        suggestions = batch.suggestions
        if not suggestions:
            raise ValueError("Cannot format prompt for empty batch")

        type_instructions = """
## Task-Specific Edit Rules

For each suggestion type, follow these rules precisely:

### Addition
- Add new task(s) using `### Task T0XX:` format
- Assign the next available ID (scan existing IDs to determine the highest, then increment)
- New tasks MUST include ALL canonical fields:
  - `files_to_modify` (list of files to change)
  - `files_to_create` (list of new files, if any)
  - `estimated_complexity` (low/medium/high)
  - `subagent_type` (or manual type if applicable)
  - `depends_on` (list of prerequisite task IDs, or empty)
  - Acceptance criteria (bulleted list)
- After adding, regenerate the dependency graph and update header metadata (task count, complexity summary)

### Modification
- Update the specified task fields (description, dependencies, acceptance criteria, etc.)
- Modified tasks MUST preserve or explicitly update ALL canonical fields
  (`files_to_modify`, `files_to_create`, `estimated_complexity`, `subagent_type`,
  acceptance criteria) — never silently drop fields
- If changing `depends_on`, regenerate the dependency graph and update header metadata

### Deletion
- Remove the specified task entirely
- Clean up `depends_on` references to the deleted task ID in ALL other tasks
- Surviving tasks MUST NOT be renumbered — IDs are stable identifiers
- After deletion, regenerate the dependency graph and update header metadata (task count, complexity summary)

### Clarification
- Rewrite the task description and/or acceptance criteria for clarity
- ALL canonical fields must be preserved — do not remove or alter fields that are not being clarified
"""

        if len(suggestions) == 1:
            s = suggestions[0]
            return f"""Apply the following task suggestion to the tasks file.

**Target file (to edit)**: {self.tasks_file}
**Plan file (context only)**: {self.plan_path}

**Current tasks.md content** (for reference — read the actual file for the latest version):
```
{self.tasks_content[:3000]}{"..." if len(self.tasks_content) > 3000 else ""}
```

**Suggestion**: {s.get('title', 'Untitled')}
**Type**: {s.get('type', 'modification')}
**Task reference**: {s.get('reference', 'N/A')}
**Importance**: {s.get('importance', 'MEDIUM')}

**Details**:
{s.get('description', 'No description provided.')}

## Changes Applied in Prior Batches
(Review these to avoid redoing work — do NOT re-apply these changes)

{{prior_changes_context}}
{type_instructions}
## Instructions

1. Read the current tasks.md file
2. Locate the task mentioned in the reference (if applicable)
3. Apply the suggested change following the type-specific rules above
4. Ensure the change integrates smoothly with surrounding content
5. Check if the tasks file already contains a similar change (from a previously applied suggestion). If so, skip this suggestion or merge it with the existing content rather than duplicating
6. After any structural edit (addition, deletion, dependency change), regenerate the dependency graph and update header metadata
7. Do NOT make any other changes beyond this specific suggestion

Return a brief summary of what was changed."""

        # Multiple suggestions — batch format
        section_info = f" (Section: {batch.section_key})" if batch.section_key != "unknown" else ""

        prompt_parts = [
            f"Apply the following {len(suggestions)} related task suggestions to the tasks file{section_info}:",
            f"\n**Target file (to edit)**: {self.tasks_file}",
            f"**Plan file (context only)**: {self.plan_path}",
            f"**Batch type**: {batch.batch_type}",
            f"**Suggestions in this batch**: {len(suggestions)}",
            f"\n**Current tasks.md content** (for reference — read the actual file for the latest version):",
            "```",
            f"{self.tasks_content[:3000]}{'...' if len(self.tasks_content) > 3000 else ''}",
            "```\n",
        ]

        for i, s in enumerate(suggestions, 1):
            prompt_parts.append(f"""---
### Suggestion {i}: {s.get('title', 'Untitled')}
- **Type**: {s.get('type', 'modification')}
- **Task reference**: {s.get('reference', 'N/A')}
- **Importance**: {s.get('importance', 'MEDIUM')}

**Details**:
{s.get('description', 'No description provided.')}
""")

        prompt_parts.append("""## Changes Applied in Prior Batches
(Review these to avoid redoing work — do NOT re-apply these changes)

{prior_changes_context}
""")

        prompt_parts.append(f"---{type_instructions}")

        prompt_parts.append("""## Instructions

1. Read the current tasks.md file first
2. Apply ALL suggestions in this batch, processing them in order
3. For each suggestion:
   - Locate the referenced task (if applicable)
   - Apply the change according to its type (addition/modification/deletion/clarification)
   - Follow the type-specific rules above
   - Ensure changes integrate smoothly with surrounding content
4. After any structural edit (addition, deletion, dependency change), regenerate the dependency graph and update header metadata
5. Do NOT make any changes beyond these specific suggestions
6. If suggestions affect the same area, apply them intelligently to avoid conflicts
7. Before applying each suggestion, check if the tasks file already reflects a similar change (from a previously applied batch). If a suggestion is already addressed, note it as "already applied" in the summary and move on

## Return Format

Return a brief summary for EACH suggestion applied:
```
Suggestion 1: [What was changed]
Suggestion 2: [What was changed]
Suggestion N: Already addressed by prior changes - skipped
...
```""")

        return "\n".join(prompt_parts)


def main():
    """Main entry point."""
    bootstrap_streams()
    args = ApplyTaskSuggestionsOrchestrator.parse_args()
    orchestrator = ApplyTaskSuggestionsOrchestrator(args)
    exit_code = orchestrator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
