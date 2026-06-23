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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List


# Configuration constants
MAX_FIXES_PER_BATCH = 3  # Hard limit to prevent context overload
MAX_DESCRIPTION_CHARS = 3000  # Total description length limit per batch


@dataclass
class CodeFixBatch:
    """A batch of code fixes to be processed by a single subagent."""

    fixes: List[Dict[str, Any]] = field(default_factory=list)
    file_key: str = ""  # Primary file being fixed
    batch_type: str = "mixed"  # bug, security, improvement, style, mixed
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
        """Calculate batch priority based on contained fixes."""
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary for JSON serialization."""
        return {
            "fixes": self.fixes,
            "file_key": self.file_key,
            "batch_type": self.batch_type,
            "subagent_type": self.subagent_type,
            "fix_count": self.size,
            "total_chars": self.total_chars,
            "priority_score": self.priority_score,
        }


def determine_subagent_type(fix: Dict[str, Any]) -> str:
    """
    Determine the subagent type for a code fix.

    Claude Code only supports general-purpose subagents for implementation work,
    so this always returns "general-purpose".

    Args:
        fix: A code review fix dictionary

    Returns:
        "general-purpose"
    """
    return "general-purpose"


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
        batch.subagent_type = determine_subagent_type(fix)
        batch.add(fix)
        batches.append(batch)

    if not normal:
        # Sort by priority and return
        batches.sort(key=lambda b: -b.priority_score)
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
                current_batch.subagent_type = determine_subagent_type(fix)
                current_batch.add(fix)

        # Don't forget the last batch
        if current_batch.size > 0:
            batches.append(current_batch)

    # Step 4: Sort batches by priority (higher priority first)
    batches.sort(key=lambda b: -b.priority_score)

    return batches


def format_fix_batch_prompt(batch: CodeFixBatch, plan_path: str, base_ref: str = "HEAD~1") -> str:
    """
    Format a batch of fixes into a prompt for a subagent.

    Returns a well-structured prompt that clearly describes all fixes
    in the batch and how to apply them.

    Args:
        batch: The CodeFixBatch to format
        plan_path: Path to the plan file for context
        base_ref: Git reference to diff against

    Raises:
        ValueError: If the batch contains no fixes.
    """
    fixes = batch.fixes

    if not fixes:
        raise ValueError("Cannot format prompt for empty batch")

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

## Changes Applied in Prior Batches
(Review these to avoid redoing work — do NOT re-apply these changes)

{{prior_changes_context}}

## Instructions
1. Use `git diff {base_ref} -- {file_path}` to see recent changes{'' if base_ref else ' (skip this step — no base_ref available)'}
2. Read the file and locate the issue using line numbers or anchor text
3. Make the necessary fix
4. Verify your fix doesn't break anything (run typecheck if applicable)
5. Do NOT make any other changes beyond this specific fix

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
    for b in batches:
        type_distribution[b.batch_type] += 1
        subagent_distribution[b.subagent_type] += 1

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
    }
