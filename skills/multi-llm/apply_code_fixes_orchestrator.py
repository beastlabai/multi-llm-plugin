#!/usr/bin/env python3
"""
Orchestrator for applying validated code review fixes to the codebase.

This script reads validation results from a code review and outputs
the list of fixes that should be applied. The actual application
is handled by Claude Code using Task subagents sequentially.

Usage:
    uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py" --plan-file plans/my-plan.md [options]
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.stream_bootstrap import bootstrap_streams
from utils.apply_orchestrator_base import (
    VALID_OVERRIDE_VALUES,
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
from utils.apply_selection_helpers import merge_validation_with_groups
from utils.code_fix_batcher import (
    CodeFixBatch,
    batch_code_fixes,
    determine_subagent_type,
    estimate_batch_processing_stats,
    format_fix_batch_prompt,
)
from utils.git_utils import validate_git_ref
from utils.importance import get_highest_importance
from utils.output_handler import derive_prefix, find_output_dir
from utils.report_parser import (
    load_html_selections,
    merge_selections,
    normalize_description,
    parse_issue_descriptions,
    parse_skipped_issues,
    parse_validation_overrides_issues,
)
from utils.state_manager import (
    CURRENT_FORMAT_VERSION,
    StateManager,
    generate_suggestion_id,
    load_groups_payload,
    stamp_stable_ids,
)
from utils.validation import (
    load_validation_results as load_validation_v2,
)


# ---------------------------------------------------------------------------
# Standalone helpers (code-fixes-specific functions used by the subclass)
# ---------------------------------------------------------------------------


def build_issue_to_group_map(
    issues_json_path: str,
    grouped: List[Dict],
) -> Optional[Dict[int, Tuple[int, int]]]:
    """Build mapping from 1-based issue number to (1-based group_index, 1-based suggestion_index).

    Primary strategy: Use `suggestion_hash` on suggestions in grouped.json matched against
    issues.json entries (hash-based matching).
    Secondary strategy: Use `issue_index` field on suggestions in grouped.json (if available).
    Fallback strategy: Match issues.json entries to grouped.json suggestions by (title, model) pair.

    Returns None if no mapping can be built (no issues.json, no issue_index fields).
    """
    # Build hash->location lookup from grouped.json
    hash_lookup: Dict[str, Tuple[int, int]] = {}  # suggestion_hash -> (group_idx, sugg_idx)
    for group_idx, group in enumerate(grouped, 1):
        for sugg_idx, sugg in enumerate(group.get("suggestions", []), 1):
            shash = sugg.get("suggestion_hash")
            if shash:
                hash_lookup[shash] = (group_idx, sugg_idx)

    # Try hash-based strategy: match issues.json by suggestion_hash
    issues = load_json_file(issues_json_path)
    mapping: Dict[int, Tuple[int, int]] = {}

    if issues and isinstance(issues, list) and hash_lookup:
        for issue_num, issue in enumerate(issues, 1):
            issue_hash = issue.get("suggestion_hash") or generate_suggestion_id(issue)
            if issue_hash in hash_lookup:
                mapping[issue_num] = hash_lookup[issue_hash]
        if mapping:
            return mapping

    # Secondary strategy: use issue_index fields on suggestions in grouped.json
    mapping = {}
    has_issue_index = False
    for group_idx, group in enumerate(grouped, 1):
        for sugg_idx, sugg in enumerate(group.get("suggestions", []), 1):
            if "issue_index" in sugg:
                has_issue_index = True
                mapping[sugg["issue_index"]] = (group_idx, sugg_idx)

    if has_issue_index and mapping:
        return mapping

    # Fallback strategy: match by (title, model) pair using issues.json
    if not issues or not isinstance(issues, list):
        return None

    mapping = {}
    # Build lookup from (title, model) -> (group_idx, sugg_idx) in grouped.json
    group_lookup: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for group_idx, group in enumerate(grouped, 1):
        for sugg_idx, sugg in enumerate(group.get("suggestions", []), 1):
            title = sugg.get("title", "")
            model = sugg.get("source_model", sugg.get("model", ""))
            key = (title, model)
            if key not in group_lookup:
                group_lookup[key] = (group_idx, sugg_idx)

    # Map each issue (1-based) to its group/suggestion
    for issue_num, issue in enumerate(issues, 1):
        title = issue.get("title", "")
        model = issue.get("source_model", issue.get("model", ""))
        key = (title, model)
        if key in group_lookup:
            mapping[issue_num] = group_lookup[key]

    return mapping if mapping else None


def find_edited_issue_descriptions(
    report_path: str,
    issues: List[Dict],
    issue_to_group_map: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, Tuple[str, str]]:
    """Compare report.md descriptions with issues, return edited items by index.

    Args:
        report_path: Path to code review report.md file
        issues: List of issue groups from grouped.json
        issue_to_group_map: Optional mapping from 1-based issue number to
                           (1-based group_idx, 1-based suggestion_idx).

    Returns:
        Dict mapping issue number (1-indexed) to (original_desc, edited_desc).
    """
    report_descriptions = parse_issue_descriptions(report_path)

    if not report_descriptions:
        return {}

    edited = {}

    if issue_to_group_map is not None:
        for issue_num, report_desc in report_descriptions.items():
            if issue_num not in issue_to_group_map:
                continue
            group_idx, sugg_idx = issue_to_group_map[issue_num]
            if group_idx - 1 >= len(issues):
                continue
            group = issues[group_idx - 1]
            suggestions = group.get('suggestions', [])
            if sugg_idx - 1 < len(suggestions):
                original_desc = suggestions[sugg_idx - 1].get('desc', suggestions[sugg_idx - 1].get('description', ''))
            else:
                original_desc = group.get('desc', group.get('description', ''))

            if normalize_description(original_desc) != normalize_description(report_desc):
                edited[issue_num] = (original_desc, report_desc)
    else:
        for issue_idx, issue in enumerate(issues, start=1):
            suggestions = issue.get('suggestions', [])
            if suggestions:
                original_desc = suggestions[0].get('desc', suggestions[0].get('description', ''))
            else:
                original_desc = issue.get('desc', issue.get('description', ''))

            if issue_idx in report_descriptions:
                report_desc = report_descriptions[issue_idx]
                if normalize_description(original_desc) != normalize_description(report_desc):
                    edited[issue_idx] = (original_desc, report_desc)

    return edited


def merge_edited_issue_descriptions(
    issues: List[Dict],
    edited_descriptions: Dict[int, Tuple[str, str]],
    issue_to_group_map: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Replace desc fields with user edits, return updated issues and edit log.

    Keys in *edited_descriptions* may be any of:
      - 1-based integer issue numbers (from report.md), resolved via
        *issue_to_group_map*;
      - suggestion hashes (from HTML user_selections.json);
      - ``G<g>S<s>`` positional IDs (from HTML user_selections.json).

    All three resolve to a (group, suggestion) location. Hash/G-format
    support is required because the HTML selection path keys edits by
    suggestion_hash; without it those edits were silently dropped
    ("Applied 0 HTML-edited descriptions"). Unknown keys are skipped.

    Args:
        issues: List of issue groups with 'suggestions' arrays
        edited_descriptions: Dict mapping key -> (original, edited) tuple.
        issue_to_group_map: Optional mapping from 1-based issue number to
                           (1-based group_idx, 1-based suggestion_idx).

    Returns:
        Tuple of (updated_issues, edit_log)
    """
    import copy

    updated_issues = copy.deepcopy(issues)
    edit_log = []

    # suggestion_hash -> (1-based group_num, 1-based sugg_num) for HTML keys
    hash_to_location: Dict[str, Tuple[int, int]] = {}
    for g_idx, group in enumerate(updated_issues, 1):
        for s_idx, sugg in enumerate(group.get("suggestions", []), 1):
            shash = sugg.get("suggestion_hash")
            if shash:
                hash_to_location[shash] = (g_idx, s_idx)

    def _resolve(key) -> Optional[Tuple[int, int]]:
        """Resolve an edit key to a 1-based (group_num, sugg_num)."""
        # Integer issue number via the issue map (report.md path)
        if issue_to_group_map is not None and key in issue_to_group_map:
            return issue_to_group_map[key]
        # Suggestion hash (HTML path)
        if isinstance(key, str) and key in hash_to_location:
            return hash_to_location[key]
        # G<g>S<s> positional id, or plain-int string (HTML path)
        if isinstance(key, str):
            return _parse_g_format_id(key)
        # Bare-int positional fallback when no issue map is available:
        # treat as 1-based group index, first suggestion (legacy behavior).
        if issue_to_group_map is None and isinstance(key, int):
            return (key, 1)
        return None

    for key, (original_desc, edited_desc) in edited_descriptions.items():
        location = _resolve(key)
        if location is None:
            continue
        group_idx, sugg_idx = location
        if group_idx - 1 < 0 or group_idx - 1 >= len(updated_issues):
            continue
        group = updated_issues[group_idx - 1]
        suggestions = group.get('suggestions', [])
        if 0 <= sugg_idx - 1 < len(suggestions):
            sugg = suggestions[sugg_idx - 1]
            sugg["_original_desc"] = sugg.get("desc", sugg.get("description", ""))
            sugg["desc"] = edited_desc
            sugg["_description_edited"] = True
            title = sugg.get("title", group.get("theme", "Unknown"))
        else:
            group["_original_desc"] = group.get("desc", group.get("description", ""))
            group["desc"] = edited_desc
            group["_description_edited"] = True
            title = group.get("theme", "Unknown")

        edit_log.append({
            "index": key,
            "title": title,
            "original_len": len(original_desc),
            "edited_len": len(edited_desc),
        })

    return updated_issues, edit_log


def format_fix_for_output(group: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Format a grouped issue for output to Claude Code."""
    suggestions = group.get("suggestions", [])

    if suggestions:
        primary = suggestions[0]
        title = group.get("theme") or primary.get("title", "Unknown")
        desc = primary.get("desc", primary.get("description", ""))
        fix_type = primary.get("type", "bug")
        file_path = primary.get("file", "")
        line_range = primary.get("line_range") or []
        anchor_text = primary.get("anchor_text", "")
        importance = primary.get("importance", "MEDIUM")
    else:
        title = group.get("theme", "Unknown")
        desc = group.get("desc", group.get("description", ""))
        fix_type = group.get("type", "bug")
        file_path = group.get("file", "")
        line_range = group.get("line_range") or []
        anchor_text = group.get("anchor_text", "")
        importance = group.get("importance", "MEDIUM")

    if isinstance(importance, str):
        importance = importance.upper()

    if len(suggestions) > 1:
        all_descs = [s.get("desc", s.get("description", "")) for s in suggestions if s.get("desc") or s.get("description")]
        if all_descs:
            desc = "\n\n".join(all_descs)

    formatted = {
        "index": index,
        # Stable group identifier so judging subagents / human-decision-batch.md
        # can reference pre-marked items for routing/recording/resume/overlay.
        "group_id": group.get("group_hash", ""),
        "title": title,
        "description": desc,
        "desc": desc,  # Keep both for compatibility
        "type": fix_type,
        "file": file_path,
        "line_range": line_range,
        "anchor_text": anchor_text,
        "importance": importance,
        "theme": group.get("theme", ""),
        "category": group.get("category", ""),
        "validation_status": group.get("validation_status", "valid"),
        "validation_reason": group.get("validation_reason", ""),
        "validation_confidence": group.get("validation_confidence", 0.0),
        "models": group.get("models", []),
        "suggestion_count": len(suggestions),
        "subagent_type": determine_subagent_type({
            "file": file_path,
            "type": fix_type,
            "description": desc,
            "title": title,
        }),
    }

    # Per-item routing flag: a reviewer pre-marked this group "Let Claude
    # decide" in the report. Routed to the per-item judge without prompting.
    if group.get("claude_decide"):
        formatted["decision_mode"] = "claude_auto_decide"

    return formatted


def _parse_g_format_id(gid: str) -> Optional[Tuple[int, int]]:
    """Parse a G-format ID like 'G3S1' into (group_num, suggestion_num).

    Supports two formats:
    - G-format: 'G3S1' -> (3, 1)  (case-insensitive)
    - Plain integer: '3' -> (3, 1) with a stderr warning

    Returns None for unparseable, zero, or negative inputs.
    """
    if not isinstance(gid, str) or not gid.strip():
        print(
            f"WARNING: Expected string ID, got {type(gid).__name__ if not isinstance(gid, str) else repr(gid)}. Ignoring.",
            file=sys.stderr,
        )
        return None

    gid = gid.strip()

    m = re.match(r'^G(\d+)S(\d+)$', gid, re.IGNORECASE)
    if m:
        group_num = int(m.group(1))
        sugg_num = int(m.group(2))
        if group_num < 1 or sugg_num < 1:
            print(
                f"WARNING: G-format ID '{gid}' has zero/negative index. Ignoring.",
                file=sys.stderr,
            )
            return None
        return (group_num, sugg_num)

    try:
        val = int(gid)
    except (ValueError, TypeError):
        print(
            f"WARNING: Cannot parse ID '{gid}' as G-format or integer. Ignoring.",
            file=sys.stderr,
        )
        return None

    if val < 1:
        print(
            f"WARNING: Integer ID '{gid}' is zero or negative. Ignoring.",
            file=sys.stderr,
        )
        return None

    print(
        f"WARNING: Treating plain integer ID '{gid}' as G{val}S1 "
        f"(first suggestion in group {val}).",
        file=sys.stderr,
    )
    return (val, 1)


def _apply_edited_descriptions_to_groups(
    grouped: List[Dict],
    edited_descs: Dict[str, str],
) -> Tuple[int, List[Dict]]:
    """Apply edited descriptions from HTML selections to grouped issues.

    Iterates over *edited_descs*, resolves each key (by suggestion_hash first,
    then by ``_parse_g_format_id``), converts to 0-based indices, validates
    bounds, and updates the target suggestion's ``desc`` field.

    Returns:
        Tuple of (count_applied, edit_log_entries).
    """
    # Build hash->(group_num, sugg_num) lookup (1-based)
    hash_to_location: Dict[str, Tuple[int, int]] = {}
    for g_idx, group in enumerate(grouped, 1):
        for s_idx, sugg in enumerate(group.get("suggestions", []), 1):
            shash = sugg.get("suggestion_hash")
            if shash:
                hash_to_location[shash] = (g_idx, s_idx)

    count_applied = 0
    edit_log: List[Dict] = []

    for gid, new_desc in edited_descs.items():
        # Try hash-based resolution first
        if gid in hash_to_location:
            group_num, sugg_num = hash_to_location[gid]
        else:
            parsed = _parse_g_format_id(gid)
            if parsed is None:
                continue
            group_num, sugg_num = parsed

        group_idx = group_num - 1  # 0-based
        sugg_idx = sugg_num - 1    # 0-based

        # Validate group bounds
        if group_idx < 0 or group_idx >= len(grouped):
            print(
                f"WARNING: Group index {group_num} (from ID '{gid}') is out of range "
                f"(valid: 1-{len(grouped)}). Skipping edit.",
                file=sys.stderr,
            )
            continue

        group = grouped[group_idx]
        suggestions = group.get("suggestions", [])

        # Validate suggestion bounds
        if sugg_idx < 0 or sugg_idx >= len(suggestions):
            if suggestions:
                print(
                    f"WARNING: Suggestion index {sugg_num} (from ID '{gid}') is out of range "
                    f"for group {group_num} (valid: 1-{len(suggestions)}). Skipping edit.",
                    file=sys.stderr,
                )
            else:
                # No suggestions list -- try updating at group level
                current_desc = group.get("desc", group.get("description", ""))
                if current_desc == new_desc:
                    continue  # Already identical, skip
                group["_original_desc"] = current_desc
                group["desc"] = new_desc
                group["_description_edited"] = True
                count_applied += 1
                edit_log.append({
                    "id": gid,
                    "group_index": group_num,
                    "sugg_index": sugg_num,
                    "title": group.get("theme", "Unknown"),
                    "original_len": len(current_desc),
                    "edited_len": len(new_desc),
                })
                print(f"HTML edit applied to group {group_num} (no suggestions, group-level desc)", file=sys.stderr)
            continue

        target_obj = suggestions[sugg_idx]
        current_desc = target_obj.get("desc", target_obj.get("description", ""))

        # Idempotency: skip if already identical
        if current_desc == new_desc:
            continue

        target_obj["_original_desc"] = current_desc
        target_obj["desc"] = new_desc
        target_obj["_description_edited"] = True
        count_applied += 1

        title = target_obj.get("title", group.get("theme", "Unknown"))
        edit_log.append({
            "id": gid,
            "group_index": group_num,
            "sugg_index": sugg_num,
            "title": title,
            "original_len": len(current_desc),
            "edited_len": len(new_desc),
        })
        print(f"HTML edit applied to {gid} (group {group_num}, suggestion {sugg_num})", file=sys.stderr)

    return count_applied, edit_log


# ===========================================================================
# Orchestrator subclass
# ===========================================================================


class ApplyCodeFixesOrchestrator(ApplyOrchestratorBase[Dict[str, Any], CodeFixBatch]):
    """Thin subclass for the apply-fixes phase (code review fixes)."""

    phase_name = "apply-fixes"
    confirmation_phase_name = "apply-code-fixes"
    review_subdir = "code-review"
    item_noun = "fix"
    supports_revalidation = True
    supports_skip_flag = False
    marks_phase_completed = False

    # Instance state populated by post_load_hook
    issue_to_group_map: Optional[Dict[int, Tuple[int, int]]] = None

    # ----------------------------------------------------------------
    # Argument parser
    # ----------------------------------------------------------------

    @classmethod
    def parse_args(cls):
        """Build and parse CLI arguments for apply-fixes."""
        parser = build_common_arg_parser(
            description="Prepare validated code review fixes for application",
            epilog="""
This orchestrator outputs a JSON list of fixes to apply.
The actual application is handled by Claude Code using Task subagents.

Example:
  uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py" --plan-file plans/my-plan.md

Bulk Approval Examples:
  # Auto-approve all LOW importance items
  uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py" --plan-file plans/my-plan.md --approve-all-low

  # Auto-approve items with validation failures (parsing errors, timeouts)
  uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py" --plan-file plans/my-plan.md --approve-validation-failed

  # Skip all items requiring human review
  uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py" --plan-file plans/my-plan.md --skip-all-human

  # Re-run validation on failed items with a different model
  uv run --project "${CLAUDE_SKILL_DIR}" -- python "${CLAUDE_SKILL_DIR}/apply_code_fixes_orchestrator.py" --plan-file plans/my-plan.md --revalidate --revalidate-model cursor-agent:opus
        """,
            include_revalidation=True,
            include_output_format=True,
            include_base_ref=True,
            include_approve_validation_failed=True,
            include_revalidate_all_human=True,
            include_internal_revalidation=True,
        )
        # Code fixes uses a smaller default batch size than the common parser
        parser.set_defaults(max_batch_size=3)
        return parser.parse_args()

    # ----------------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------------

    def post_load_hook(self) -> None:
        """Build issue_to_group_map after loading grouped data."""
        code_review_dir = os.path.join(self.out_dir, self.review_subdir)
        issues_json_path = os.path.join(code_review_dir, "issues.json")
        self.issue_to_group_map = build_issue_to_group_map(
            issues_json_path, self.groups
        )

    def handle_no_items_early_exit(self) -> None:
        """Write empty output and exit when no fixes remain."""
        print("\nNo fixes to apply.", file=sys.stderr)

        phase_dir = os.path.join(self.out_dir, self.phase_name)
        os.makedirs(phase_dir, exist_ok=True)

        output = {
            "format_version": CURRENT_FORMAT_VERSION,
            "plan_file": self.plan_path,
            "prefix": self.prefix,
            "output_dir": self.out_dir,
            "timestamp": datetime.now().isoformat(),
            "batches": [],
            "to_apply": [],
            "needs_human_review": [],
            "skipped_items": self.formatted_skipped,
            "user_skipped_items": self.user_skipped_items,
            "skipped_count": len(self.skipped),
            "batching_stats": {
                "total_fixes": 0,
                "total_batches": 0,
                "subagent_calls_saved": 0,
                "efficiency_gain_percent": 0,
            },
            "summary": {
                "total_issues": len(self.groups),
                "valid_count": 0,
                "needs_human_count": 0,
                "skipped_count": len(self.skipped),
                "user_skipped_count": len(self.user_skipped_items),
                "batch_count": 0,
            }
        }

        write_and_emit_output(output, phase_dir)
        emit_json_output(output)
        sys.exit(0)

    def apply_group_validation_overrides(self) -> None:
        """Apply validation overrides using integer issue-number keys.

        The code-fixes orchestrator uses 1-based integer group indices as
        override keys (from parse_validation_overrides_issues), unlike the
        hash-based keys used by suggestion orchestrators.
        """
        if not self.validation_overrides:
            return

        override_count = 0
        for idx, group in enumerate(self.merged, 1):  # 1-based group index
            # Try integer key match first (code-fixes pattern), then hash-based
            # match (from HTML/consolidated decisions).
            matched_key: Any = None
            if idx in self.validation_overrides:
                matched_key = idx
            else:
                ghash = group.get("group_hash", "")
                if ghash and ghash in self.validation_overrides:
                    matched_key = ghash

            if matched_key is None:
                continue

            value = self.validation_overrides[matched_key]
            if value not in VALID_OVERRIDE_VALUES:
                print(
                    f"WARNING: ignoring unknown validation override {value!r} "
                    f"for issue {matched_key}; leaving its underlying status "
                    f"unchanged.",
                    file=sys.stderr,
                )
                continue
            # Route "claude_decide" as a marker, never as a status.
            if self._route_claude_decide_marker(group, matched_key, value):
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

    def get_revalidation_context(self) -> str:
        """Return diff_context.txt contents for revalidation, falling back to plan."""
        code_review_dir = os.path.join(self.out_dir, self.review_subdir)
        context_path = os.path.join(code_review_dir, "diff_context.txt")
        if os.path.exists(context_path):
            with open(context_path, 'r', encoding='utf-8') as f:
                return f.read()
        # Fallback to plan content
        print("Warning: diff_context.txt not found, using plan content", file=sys.stderr)
        with open(self.plan_path, 'r', encoding='utf-8') as f:
            return f.read()

    def resolve_base_ref(self) -> str:
        """Resolve git base ref from --base-ref arg or state head_at_start.

        Used by the base class ``handle_revalidation()`` and by
        ``generate_batch_prompts()`` to provide diff context.

        Returns:
            A validated git ref string, or empty string.
        """
        base_ref = getattr(self.args, "base_ref", None)
        if not base_ref:
            base_ref = (
                (self.state.get("head_before_implement") or self.state.get("head_at_start"))
                if self.state else None
            ) or "HEAD~1"
        return validate_git_ref(base_ref)

    # ----------------------------------------------------------------
    # Abstract method implementations
    # ----------------------------------------------------------------

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load grouped issues and validation from code-review subdir."""
        code_review_dir = os.path.join(self.out_dir, self.review_subdir)
        grouped_path = os.path.join(code_review_dir, "grouped.json")
        validation_path = Path(code_review_dir) / "validation.json"

        raw_grouped = load_json_file(grouped_path)
        if raw_grouped is None:
            print(f"ERROR: Grouped issues not found at {grouped_path}", file=sys.stderr)
            print("Run --review-code first to generate code review results.", file=sys.stderr)
            raise OrchestratorError(
                f"Grouped issues not found at {grouped_path}. "
                "Run --review-code first to generate code review results.",
                exit_code=1,
            )
        groups = load_groups_payload(raw_grouped)
        stamp_stable_ids(groups)

        # Load validation results (using v2 loader with migration)
        if validation_path.exists():
            validation = load_validation_v2(validation_path)
        else:
            print("WARNING: Validation results not found, treating all as needs-human-decision", file=sys.stderr)
            validation = []

        print(f"Loaded {len(groups)} issue groups", file=sys.stderr)
        print(f"Loaded {len(validation)} validation results", file=sys.stderr)

        return groups, validation

    def parse_user_edits(self, report_path: str) -> Dict[int, Tuple[str, str]]:
        """Detect and return user-edited descriptions from report.md.

        Uses issue-number-based find_edited_issue_descriptions.
        """
        return find_edited_issue_descriptions(
            report_path, self.groups, self.issue_to_group_map
        )

    def merge_user_edits(
        self,
        groups: List[Dict[str, Any]],
        edited_descriptions,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Merge issue-number-based edited descriptions."""
        return merge_edited_issue_descriptions(
            groups, edited_descriptions, self.issue_to_group_map
        )

    def parse_skips_from_report(
        self, report_path: str
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Parse skipped issues from report.md.

        Returns integer-based skipped issue numbers converted to group_hash
        strings for compatibility with the base class filtering.
        """
        skipped_issue_numbers = parse_skipped_issues(report_path)
        skipped_suggestion_ids: Set[str] = set()

        # Convert integer-based skipped_group_indices to hash-based Set[str]
        skipped_group_hashes: Set[str] = set()
        for g_idx in skipped_issue_numbers:
            if isinstance(g_idx, int) and 1 <= g_idx <= len(self.groups):
                ghash = self.groups[g_idx - 1].get("group_hash", "")
                if ghash:
                    skipped_group_hashes.add(ghash)
                else:
                    print(f"WARNING: Skipped group index {g_idx} has no group_hash", file=sys.stderr)
            elif isinstance(g_idx, str):
                skipped_group_hashes.add(g_idx)

        return skipped_group_hashes, skipped_suggestion_ids, set()

    def parse_validation_overrides_from_report(
        self, report_path: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Parse validation overrides from report.md.

        Returns integer-key (as string) overrides for code-fixes.
        """
        md_validation_overrides = parse_validation_overrides_issues(report_path)
        # Return as group-level overrides (integer keys preserved), no per-suggestion overrides
        return md_validation_overrides, {}

    def format_item_for_output(self, group: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a grouped issue for output, including subagent_type routing."""
        return format_fix_for_output(group, index)

    def create_batches(self, items: List[Dict[str, Any]]) -> List[CodeFixBatch]:
        """Create batches from formatted fixes."""
        if getattr(self.args, "no_batch", False):
            batches = [
                CodeFixBatch(
                    fixes=[f],
                    file_key=f.get("file", "unknown"),
                    batch_type=f.get("type", "bug"),
                    subagent_type=f.get("subagent_type", "general-purpose"),
                    total_chars=len(f.get("description", ""))
                )
                for f in items
            ]
            self.batching_stats = {
                "total_fixes": len(items),
                "total_batches": len(items),
                "subagent_calls_saved": 0,
                "efficiency_gain_percent": 0,
                "batching_enabled": False,
            }
        else:
            batches = batch_code_fixes(
                items,
                max_per_batch=getattr(self.args, "max_batch_size", 3),
            )
            self.batching_stats = estimate_batch_processing_stats(batches)
            self.batching_stats["batching_enabled"] = True

        print("\nBatching results:", file=sys.stderr)
        print(f"  Total fixes: {self.batching_stats.get('total_fixes', 0)}", file=sys.stderr)
        print(f"  Total batches: {self.batching_stats.get('total_batches', 0)}", file=sys.stderr)
        if self.batching_stats.get("batching_enabled"):
            print(f"  Subagent calls saved: {self.batching_stats.get('subagent_calls_saved', 0)}", file=sys.stderr)
            print(f"  Efficiency gain: {self.batching_stats.get('efficiency_gain_percent', 0)}%", file=sys.stderr)
            if self.batching_stats.get("subagent_distribution"):
                print(f"  Subagent distribution: {self.batching_stats.get('subagent_distribution')}", file=sys.stderr)

        return batches

    def generate_batch_prompts(self, batches: List[CodeFixBatch]) -> List[Dict[str, Any]]:
        """Attach fix-specific prompts to each batch.

        Returns a list of dicts (batch.to_dict() + prompt) for output assembly.
        """
        # Resolve base_ref using the shared hook (consistent with handle_revalidation)
        base_ref = self.resolve_base_ref()
        print(f"Git base reference: {base_ref or '(empty - validation will lack diff context)'}", file=sys.stderr)

        # Store base_ref for use in build_output_json
        self._base_ref = base_ref

        result = []
        for batch in batches:
            prompt = format_fix_batch_prompt(batch, self.plan_path, base_ref)
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
        os.makedirs(phase_dir, exist_ok=True)

        state_file = os.path.join(self.out_dir, "state.json")
        base_ref = getattr(self, "_base_ref", "")

        return {
            "format_version": CURRENT_FORMAT_VERSION,
            "plan_file": self.plan_path,
            "prefix": self.prefix,
            "output_dir": self.out_dir,
            "state_file": state_file,
            "base_ref": base_ref,
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
                "total_issues": len(self.groups),
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
        """Return path for orchestrator_output.json.

        Uses self.out_dir (which includes the double-nesting guard from
        find_output_dir) rather than get_phase_dir (which uses the
        unguarded get_output_dir and can double-nest).
        """
        phase_dir = os.path.join(self.out_dir, self.phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        return os.path.join(phase_dir, "orchestrator_output.json")

    def print_text_summary(self, batches: List[Any], output_path: str) -> None:
        """Print human-readable summary or JSON output to stdout."""
        output_format = getattr(self.args, "output_format", "text")

        if output_format == "json":
            with open(output_path, "r", encoding="utf-8") as f:
                output = json.load(f)
            print(json.dumps(output, indent=2))
        else:
            base_ref = getattr(self, "_base_ref", "")

            print(f"\n{'='*60}")
            print("CODE FIXES TO APPLY (BATCHED)")
            print(f"{'='*60}")
            print(f"Plan: {self.plan_path}")
            print(f"Valid fixes: {len(self.valid)}")
            print(f"Batches: {len(batches)}")
            print(f"Needs human review: {len(self.needs_human)}")
            print(f"Skipped: {len(self.skipped)}")
            print(f"Base ref: {base_ref}")

            if self.batching_stats.get("batching_enabled"):
                print("\nBatching efficiency:")
                print(f"  Subagent calls saved: {self.batching_stats.get('subagent_calls_saved', 0)}")
                print(f"  Efficiency gain: {self.batching_stats.get('efficiency_gain_percent', 0)}%")

            for i, batch in enumerate(batches):
                batch_fixes = batch.get("fixes", []) if isinstance(batch, dict) else batch.fixes
                batch_file = batch.get("file_key", "") if isinstance(batch, dict) else batch.file_key
                batch_type = batch.get("batch_type", "") if isinstance(batch, dict) else batch.batch_type
                batch_subagent = batch.get("subagent_type", "") if isinstance(batch, dict) else batch.subagent_type
                batch_size = batch.get("fix_count", len(batch_fixes)) if isinstance(batch, dict) else batch.size
                batch_priority = batch.get("priority_score", 0) if isinstance(batch, dict) else batch.priority_score

                print(f"\n{'='*60}")
                print(f"BATCH {i+1} ({batch_size} fix(es))")
                print(f"{'='*60}")
                print(f"File: {batch_file}")
                print(f"Type: {batch_type}")
                print(f"Subagent: {batch_subagent}")
                print(f"Priority score: {batch_priority}")

                for j, f in enumerate(batch_fixes):
                    print(f"\n  --- Fix {j+1} ---")
                    print(f"  Title: {f['title']}")
                    print(f"  Type: {f['type']}")
                    print(f"  Importance: {f['importance']}")
                    print(f"  File: {f.get('file', 'unknown')}")
                    desc = f.get('description', f.get('desc', ''))[:200]
                    print(f"  Description:\n  {desc}...")

            if self.formatted_human:
                print(f"\n{'='*60}")
                print("NEEDS HUMAN REVIEW")
                print(f"{'='*60}")
                for f in self.formatted_human:
                    print(f"\n- {f['title']}")
                    print(f"  File: {f.get('file', 'unknown')}")
                    print(f"  Reason: {f['validation_reason']}")


def main():
    """Main entry point."""
    bootstrap_streams()
    args = ApplyCodeFixesOrchestrator.parse_args()
    orchestrator = ApplyCodeFixesOrchestrator(args)
    exit_code = orchestrator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
