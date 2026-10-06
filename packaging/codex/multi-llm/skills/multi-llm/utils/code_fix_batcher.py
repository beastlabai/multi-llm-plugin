"""
Code fix batcher module for grouping code review fixes into efficient subagent batches.

This module takes validated code review issues and groups them intelligently for processing
by fewer subagents, reducing API calls while maintaining edit safety.

Grouping Heuristics:
1. File Proximity - Group fixes targeting the same file
2. Type Safety - Security/HIGH fixes are always isolated
3. Batch Size Limits - Prevent context rot with max fixes per batch
4. Line Ordering - Process fixes in line order within each file
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    from .plan_updater import get_tasks_file_path
    from .prior_changes import entries as prior_changes_entries
    from .prior_changes import entry_id, stable_key
    from .subagent_tasks import subagent_type_for_role
except ImportError:
    from utils.plan_updater import get_tasks_file_path
    from utils.prior_changes import entries as prior_changes_entries
    from utils.prior_changes import entry_id, stable_key
    from utils.subagent_tasks import subagent_type_for_role


# Configuration constants
MAX_FIXES_PER_BATCH = 3  # Hard limit to prevent context overload
MAX_DESCRIPTION_CHARS = 3000  # Total description length limit per batch

# How many implement-phase prior_changes.jsonl entries a fix batch's
# "Implementation record" section shows, and the char budget for them (F3,
# 2026-09-16) — a smaller version of prior_changes.render's own DEFAULT_TAIL /
# MAX_CHARS, scoped to one phase's entries rather than the whole shared log.
MAX_IMPLEMENTATION_RECORD_ENTRIES = 5
MAX_IMPLEMENTATION_RECORD_CHARS = 1200


def fix_member_ids(fixes: List[Dict[str, Any]]) -> List[str]:
    """The batch's fixes as stable member ids, in order.

    The one spelling of "which fixes is this batch made of": it is what
    ``utils/write_phase_bridge`` puts in a write task's ``expected_ids`` and what
    both invokers digest into :func:`utils.prior_changes.entry_id`. Two batches on
    the same file are distinguished by nothing else, so this cannot be derived
    per-caller.

    A fix carrying none of ``id``/``group_id``/``title`` falls back to its
    position *within* its own batch, which is the one input that does not
    distinguish two same-sized batches on one file. Real fixes come from grouped
    review issues and carry a 16-char content-hash group id, so the fallback is a
    degenerate-input guard; widening it would change ``expected_ids``, which the
    D3g verification matches against what the runner reports.
    """
    return [
        stable_key([fix.get("id"), fix.get("group_id"), fix.get("title")], f"f{n}")
        for n, fix in enumerate(fixes)
    ]


@dataclass
class CodeFixBatch:
    """A batch of code fixes to be processed by a single subagent."""

    fixes: List[Dict[str, Any]] = field(default_factory=list)
    file_key: str = ""  # Primary file being fixed
    batch_type: str = "mixed"  # bug, security, improvement, style, mixed
    # Host-neutral routing field (D2h): "agent" means any invoker may dispatch
    # this batch, "human" means no invoker ever does. Every reader branches on
    # this; `subagent_type` below is optional Claude-only metadata derived from
    # it and is never consulted to choose a runner.
    executor_role: str = "agent"
    subagent_type: str = "general-purpose"
    total_chars: int = 0

    @property
    def size(self) -> int:
        return len(self.fixes)

    @property
    def fix_count(self) -> int:
        """Alias for size for JSON output compatibility."""
        return self.size

    @property
    def is_full(self) -> bool:
        return self.size >= MAX_FIXES_PER_BATCH

    @property
    def priority_score(self) -> float:
        """Sum of this batch's fix-importance weights.

        A SUM, not a rank — three LOW/MEDIUM fixes (1+2+2=5.0) outweighs one
        isolated HIGH fix (3.0). That is fine as a dispatch-order *tiebreak*
        (``batch_sort_key`` below), but sorting batches on this value alone
        undoes the isolation step that put the HIGH fix in its own batch for
        safety (F5, 2026-09-16) — kept as-is here for JSON-output
        compatibility (``to_dict()``'s ``priority_score`` field).
        """
        score = 0.0
        importance_weights = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
        for fix in self.fixes:
            importance = fix.get("importance", "MEDIUM")
            if isinstance(importance, str):
                importance = importance.upper()
            score += importance_weights.get(importance, 2.0)
        return score

    def can_add(self, fix: Dict[str, Any]) -> bool:
        """Check if a fix can be added to this batch."""
        if not isinstance(fix, dict):
            return False

        if self.is_full:
            return False

        desc_len = len(fix.get("description", fix.get("desc", "")))
        if self.total_chars + desc_len > MAX_DESCRIPTION_CHARS:
            return False

        return True

    def add(self, fix: Dict[str, Any]) -> None:
        """Add a fix to this batch."""
        self.fixes.append(fix)
        self.total_chars += len(fix.get("description", fix.get("desc", "")))

        # Update batch type
        types = set(f.get("type", "improvement") for f in self.fixes)
        if len(types) == 1:
            self.batch_type = types.pop()
        else:
            self.batch_type = "mixed"

    @property
    def prior_changes_id(self) -> str:
        """The stable, per-batch ``--id`` for the prior-changes append.

        NOT ``file_key``: the batcher emits many batches per file (three fixes a
        batch, plus one batch per security/HIGH fix), and ``prior_changes.append``
        keeps one entry per ``(id, phase)`` (the newest, since fix round 14) — so
        a file-keyed id keeps only the last batch on that file and silently drops
        every earlier one. Computed here so a
        native invoker reads it off the serialized batch instead of picking a key,
        and byte-identical to what ``utils/write_phase_bridge`` derives for the
        same batch.
        """
        return entry_id(
            self.file_key, fix_member_ids(self.fixes), fallback="batch"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary for JSON serialization."""
        return {
            "fixes": self.fixes,
            "file_key": self.file_key,
            "prior_changes_id": self.prior_changes_id,
            "batch_type": self.batch_type,
            "executor_role": self.executor_role,
            "subagent_type": self.subagent_type,
            "fix_count": self.size,
            "total_chars": self.total_chars,
            "priority_score": self.priority_score,
        }


def determine_executor_role(fix: Dict[str, Any]) -> str:
    """
    Determine the host-neutral executor role for a code fix (D2h).

    Code fixes are always agent work — a code review never emits a manual step —
    so this returns "agent" unconditionally. It exists as a function so the one
    place that decides is the one place to change.

    Args:
        fix: A code review fix dictionary

    Returns:
        "agent"
    """
    return "agent"


def determine_subagent_type(fix: Dict[str, Any]) -> str:
    """
    Determine the Claude-Code-only ``subagent_type`` for a code fix.

    Kept for the hosts whose native Task tool requires the parameter, derived
    from :func:`determine_executor_role` so the two can never disagree (the task
    schema rejects a disagreement). No reader branches on it.

    Args:
        fix: A code review fix dictionary

    Returns:
        "general-purpose"
    """
    return subagent_type_for_role(determine_executor_role(fix))


def get_line_start(fix: Dict[str, Any]) -> int:
    """Extract the starting line number from a fix for sorting."""
    line_range = fix.get("line_range")
    if isinstance(line_range, (list, tuple)) and len(line_range) >= 1:
        return int(line_range[0])
    return 0


def is_high_risk_fix(fix: Dict[str, Any]) -> bool:
    """
    Determine if a fix should be isolated (one per batch) for safety.

    Security fixes and HIGH importance fixes should always be isolated.
    """
    fix_type = (fix.get("type") or "").lower()
    importance = (fix.get("importance") or "").upper()

    # Security fixes are always isolated
    if fix_type == "security":
        return True

    # HIGH importance fixes are isolated
    if importance == "HIGH":
        return True

    return False


_IMPORTANCE_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def batch_sort_key(batch: "CodeFixBatch"):
    """Dispatch-order sort key: isolated high-risk first, then importance, then sum.

    ``instructions/apply-code-fixes.md`` promises "Security fixes are ALWAYS
    isolated … HIGH importance fixes are ALWAYS isolated" and "Batches are
    sorted by priority score (higher importance first)" — i.e. isolation for
    safety, *then* importance. Sorting on the summed ``priority_score`` alone
    breaks that: a batch of three LOW/MEDIUM fixes (5.0) always outranks an
    isolated HIGH fix (3.0), so the riskiest edit was dispatched LAST, landing
    on top of every other batch's edits instead of on known-good bytes (F5,
    2026-09-16). This key ranks isolation first (a batch containing any
    high-risk fix), then the batch's own peak importance, and only then the
    summed score as a tiebreak among batches that agree on both — the same
    key both sort sites in :func:`batch_code_fixes` use, so they cannot drift
    apart again.
    """
    is_isolated_high_risk = any(is_high_risk_fix(f) for f in batch.fixes)
    max_importance = max(
        (
            _IMPORTANCE_RANK.get(str(f.get("importance", "MEDIUM")).upper(), 2)
            for f in batch.fixes
        ),
        default=0,
    )
    return (not is_isolated_high_risk, -max_importance, -batch.priority_score)


def batch_code_fixes(
    fixes: List[Dict[str, Any]],
    max_per_batch: int = MAX_FIXES_PER_BATCH,
    max_chars: int = MAX_DESCRIPTION_CHARS,
) -> List[CodeFixBatch]:
    """
    Group validated code review fixes into batches for efficient subagent processing.

    Heuristics applied (in priority order):
    1. High-risk fixes (security, HIGH importance) -> isolated batches
    2. Same file -> group together
    3. Line order -> process in sequence
    4. Size limits -> split if too large

    Args:
        fixes: List of validated fix dicts from the orchestrator
        max_per_batch: Maximum fixes per batch (default: 3)
        max_chars: Maximum total description chars per batch (default: 3000)

    Returns:
        List of CodeFixBatch objects ready for subagent processing
    """
    if not fixes:
        return []

    batches: List[CodeFixBatch] = []

    # Step 1: Separate high-risk fixes - they always go in their own batches
    high_risk = [f for f in fixes if is_high_risk_fix(f)]
    normal = [f for f in fixes if not is_high_risk_fix(f)]

    # Process high-risk fixes first (each in its own batch for safety)
    for fix in high_risk:
        batch = CodeFixBatch(file_key=fix.get("file", "unknown"))
        batch.executor_role = determine_executor_role(fix)
        batch.subagent_type = determine_subagent_type(fix)
        batch.add(fix)
        batches.append(batch)

    if not normal:
        # Sort isolated-high-risk-first, then by importance (F5).
        batches.sort(key=batch_sort_key)
        return batches

    # Step 2: Group normal fixes by file
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fix in normal:
        file_path = fix.get("file", "unknown")
        by_file[file_path].append(fix)

    # Step 3: Within each file, create batches respecting limits
    for file_path, file_fixes in by_file.items():
        # Sort by line number for logical ordering
        file_fixes.sort(key=get_line_start)

        current_batch = CodeFixBatch(file_key=file_path)
        current_batch.executor_role = determine_executor_role(file_fixes[0])
        current_batch.subagent_type = determine_subagent_type(file_fixes[0])

        for fix in file_fixes:
            # Check if we can add to current batch
            if current_batch.can_add(fix):
                current_batch.add(fix)
            else:
                # Batch is full or would exceed limits - start new batch
                if current_batch.size > 0:
                    batches.append(current_batch)
                current_batch = CodeFixBatch(file_key=file_path)
                current_batch.executor_role = determine_executor_role(fix)
                current_batch.subagent_type = determine_subagent_type(fix)
                current_batch.add(fix)

        # Don't forget the last batch
        if current_batch.size > 0:
            batches.append(current_batch)

    # Step 4: Sort isolated-high-risk batches first, then by importance (F5) —
    # the same key Step 1's early return uses, so the two sort sites cannot
    # disagree about dispatch order.
    batches.sort(key=batch_sort_key)

    return batches


def _implementation_record_lines(
    output_dir: str, batch_files: Sequence[str]
) -> List[str]:
    """The implement phase's own record, for the files a fix batch is about to touch.

    A code-review finding is judged against the *plan*; nothing judged it
    against the implement phase's own ``prior_changes.jsonl`` — so a runner
    applying a fix batch could (and did) revert a deliberate, logged tune
    ("accel tuned 28->32 for <4s") that satisfied the plan's own goal, because
    it was never shown that entry (F3, 2026-09-16).

    Reads :func:`utils.prior_changes.entries` directly rather than ``.render``
    — ``render``'s ``phase=`` argument prefers the *dispatching* phase's own
    entries (apply-code-fixes' own prior batches), which is the opposite
    preference this section needs: apply-code-fixes always runs after
    implement has already finished, so it is implement's entries, not its
    own, that belong here. Entries whose summary names one of the batch's own
    files sort first, so a deliberate tune to the exact line a fix touches is
    never crowded out of the (small) budget by an unrelated implement note.
    """
    implement_entries = [
        e for e in prior_changes_entries(output_dir) if e.get("phase") == "implement"
    ]
    if not implement_entries:
        return []

    names = [os.path.basename(f) for f in batch_files if f]

    def mentions_a_batch_file(entry: Dict[str, Any]) -> bool:
        summary = entry.get("summary", "")
        return any(name and name in summary for name in names)

    ordered = sorted(
        implement_entries, key=lambda e: 0 if mentions_a_batch_file(e) else 1
    )[:MAX_IMPLEMENTATION_RECORD_ENTRIES]

    lines: List[str] = []
    total_chars = 0
    for entry in ordered:
        line = f"- ({entry.get('id', '?')}): {entry.get('summary', '')}"
        if lines and total_chars + len(line) > MAX_IMPLEMENTATION_RECORD_CHARS:
            break
        lines.append(line)
        total_chars += len(line)
    return lines


def _implementation_record_block(output_dir: Optional[str], batch_files: Sequence[str]) -> str:
    """The rendered "## Implementation record" section, or "" when there is none.

    Kept out of ``format_fix_batch_prompt`` itself so the section, and the
    warning that goes with it, has one place to change.
    """
    if not output_dir:
        return ""
    lines = _implementation_record_lines(output_dir, batch_files)
    if not lines:
        return ""
    return (
        "\n## Implementation record\n"
        "(What the implement phase deliberately did to these files — read this "
        "before treating a finding as ground truth)\n\n"
        + "\n".join(lines)
        + "\n\nIf a finding above contradicts a decision recorded here, do NOT "
        "silently revert it: leave the code as it is and say so in your "
        "returned summary.\n"
    )


def _tasks_file_named_by(plan_path: str) -> Optional[str]:
    """The task list the plan's ``<!-- TASKS_FILE: … -->`` marker names, or None.

    Spelled the way ``plan_path`` is (relative stays relative). Best-effort and
    silent: a plan that cannot be read, or carries no marker, gets the generic
    wording in :func:`code_only_rule`.
    """
    try:
        with open(plan_path, "r", encoding="utf-8") as handle:
            raw = get_tasks_file_path(handle.read())
    except (OSError, UnicodeDecodeError, TypeError):
        return None
    if not raw:
        return None
    return os.path.normpath(os.path.join(os.path.dirname(plan_path), raw))


def code_only_rule(plan_path: str, tasks_path: Optional[str] = None) -> str:
    """The rule appended to every fix-batch prompt's instructions.

    A code-fix phase leaves the plan alone. The 2026-10-02 Codex run's judge
    salvaged "bring plan and code back into agreement" into "record the
    look-ahead as an approved deviation in the plan and T009", and the batch
    then edited ``plans/racing-game.md`` and ``tasks.md`` where the reviewers
    had asked for a code fix. Blessing a deviation in the plan is a human call,
    so the default is to leave the plan alone and say so.

    It names THIS run's plan and task list rather than "the plan file or any
    task list": a review-code diff can carry markdown of its own (a README,
    docs, a repo's own tracked ``plans/*.md``), the judge treats every item
    with a ``file`` as a code fix, and under "change code only" a finding on
    such a file came back "not applied: needs a plan decision" (adversarial
    review, 2026-10-02).
    """
    protected = f"this run's plan (`{plan_path}`)"
    if tasks_path:
        protected += f" or its task list (`{tasks_path}`)"
    else:
        protected += " or the task list generated from it"
    return (
        f"Do NOT edit {protected} (not to record an \"approved deviation\", not "
        "to make the plan match the code). Every other file a fix names is "
        "yours to change, markdown included: a fix on a README, a doc or another "
        "plan in the diff is applied like any code fix. If a fix can only be "
        "satisfied by changing this run's plan or task list, or that plan should "
        "be what moves, make no edit for that fix and report it as "
        "\"not applied: needs a plan decision\"."
    )


def format_fix_batch_prompt(
    batch: CodeFixBatch,
    plan_path: str,
    base_ref: str = "HEAD~1",
    output_dir: Optional[str] = None,
    tasks_path: Optional[str] = None,
) -> str:
    """
    Format a batch of fixes into a prompt for a subagent.

    Returns a well-structured prompt that clearly describes all fixes
    in the batch and how to apply them.

    Args:
        batch: The CodeFixBatch to format
        plan_path: Path to the plan file for context
        base_ref: Git reference to diff against
        output_dir: The plan's output dir (``self.out_dir``), used to read the
            implement phase's ``prior_changes.jsonl`` for the "Implementation
            record" section (F3). Optional — with none given (or no matching
            entries) the section is omitted, exactly as before.
        tasks_path: This run's task list, named in :func:`code_only_rule`.
            Optional — read off the plan's ``TASKS_FILE`` marker when omitted.

    Raises:
        ValueError: If the batch contains no fixes.
    """
    fixes = batch.fixes

    if not fixes:
        raise ValueError("Cannot format prompt for empty batch")

    implementation_record = _implementation_record_block(
        output_dir, [f.get("file", "") for f in fixes]
    )
    rule = code_only_rule(plan_path, tasks_path or _tasks_file_named_by(plan_path))

    if len(fixes) == 1:
        # Single fix - use detailed format
        fix = fixes[0]
        file_path = fix.get("file", "unknown")
        line_range = fix.get("line_range") or []
        lines_str = f"{line_range[0]}-{line_range[1]}" if len(line_range) >= 2 else "unknown"
        anchor = fix.get("anchor_text", "N/A")
        desc = fix.get("description", fix.get("desc", "No description"))

        return f"""Fix the following code issue:

**Plan file**: {plan_path}
**File**: {file_path}
**Lines**: {lines_str}
**Issue**: {fix.get('title', 'Unknown issue')}
**Type**: {fix.get('type', 'unknown')}
**Importance**: {fix.get('importance', 'MEDIUM')}

**Description**:
{desc}

**Anchor text** (to help locate): `{anchor}`
{implementation_record}
## Changes Applied in Prior Batches
(Review these to avoid redoing work — do NOT re-apply these changes)

{{prior_changes_context}}

## Instructions
1. Use `git diff {base_ref} -- {file_path}` to see recent changes{'' if base_ref else ' (skip this step — no base_ref available)'}
2. Read the file and locate the issue using line numbers or anchor text
3. Make the necessary fix
4. Verify your fix doesn't break anything (run typecheck if applicable)
5. Do NOT make any other changes beyond this specific fix
6. {rule}

Return: Brief summary of what you changed."""

    # Multiple fixes - batch format
    prompt_parts = [
        f"Fix the following {len(fixes)} related issues in `{batch.file_key}`:",
        f"\n**Plan file**: {plan_path}",
        f"**Primary file**: {batch.file_key}",
        f"**Batch type**: {batch.batch_type}",
        f"**Fixes in this batch**: {len(fixes)}\n",
    ]

    for i, fix in enumerate(fixes, 1):
        line_range = fix.get("line_range") or []
        lines_str = f"{line_range[0]}-{line_range[1]}" if len(line_range) >= 2 else "unknown"
        desc = fix.get("description", fix.get("desc", "No description"))
        anchor = fix.get("anchor_text", "N/A")

        prompt_parts.append(f"""---
### Fix {i}: {fix.get('title', 'Unknown issue')}
- **Type**: {fix.get('type', 'unknown')}
- **Lines**: {lines_str}
- **Importance**: {fix.get('importance', 'MEDIUM')}
- **Anchor text**: `{anchor}`

**Description**:
{desc}
""")

    if implementation_record:
        prompt_parts.append(implementation_record)

    prompt_parts.append("""## Changes Applied in Prior Batches
(Review these to avoid redoing work — do NOT re-apply these changes)

{prior_changes_context}
""")

    prompt_parts.append(f"""---

## Instructions

1. Use `git diff {base_ref} -- {batch.file_key}` to see recent changes{'' if base_ref else ' (skip this step — no base_ref available)'}
2. Read the file first to understand the current state
3. Apply ALL fixes in this batch, processing them in line order (top to bottom)
4. For each fix:
   - Locate using line numbers or anchor text
   - Make the necessary change
   - Ensure it integrates smoothly with surrounding code
5. Do NOT make any changes beyond these specific fixes
6. Run typecheck if applicable to verify no regressions
7. {rule}

## Return Format

Return a brief summary for EACH fix applied:
```
Fix 1: [What was changed]
Fix 2: [What was changed]
...
```""")

    return "\n".join(prompt_parts)


def estimate_batch_processing_stats(batches: List[CodeFixBatch]) -> Dict[str, Any]:
    """
    Calculate statistics about the batching result.

    Useful for reporting efficiency gains.
    """
    total_fixes = sum(b.size for b in batches)
    total_batches = len(batches)

    # Without batching, each fix would be 1 subagent call
    subagent_calls_saved = total_fixes - total_batches
    efficiency_gain = (subagent_calls_saved / total_fixes * 100) if total_fixes > 0 else 0

    batch_sizes = [b.size for b in batches]
    avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0

    type_distribution: Dict[str, int] = defaultdict(int)
    subagent_distribution: Dict[str, int] = defaultdict(int)
    executor_distribution: Dict[str, int] = defaultdict(int)
    for b in batches:
        type_distribution[b.batch_type] += 1
        subagent_distribution[b.subagent_type] += 1
        executor_distribution[b.executor_role] += 1

    return {
        "total_fixes": total_fixes,
        "total_batches": total_batches,
        "subagent_calls_saved": subagent_calls_saved,
        "efficiency_gain_percent": round(efficiency_gain, 1),
        "average_batch_size": round(avg_batch_size, 2),
        "max_batch_size": max(batch_sizes) if batch_sizes else 0,
        "single_fix_batches": sum(1 for b in batches if b.size == 1),
        "multi_fix_batches": sum(1 for b in batches if b.size > 1),
        "batch_type_distribution": dict(type_distribution),
        "subagent_distribution": dict(subagent_distribution),
        "executor_distribution": dict(executor_distribution),
    }
