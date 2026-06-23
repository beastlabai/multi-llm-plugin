"""Output file handling utilities for managing files on rerun."""

import hashlib
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Phase-to-subdirectory mapping (kebab-case for on-disk directories)
PHASE_DIRECTORIES = {
    'review-plan': 'review-plan',
    'apply-suggestions': 'apply-suggestions',
    'tasks': 'tasks',
    'review-tasks': 'review-tasks',
    'implement': 'implement',
    'code-review': 'code-review',
    'apply-fixes': 'apply-fixes',
    'apply-task-suggestions': 'apply-task-suggestions',
}

# Map output_type to phase for get_output_paths() backward compat
# NOTE: For ambiguous types (grouped, validation), callers in
# code_review_orchestrator.py MUST pass explicit phase='code-review'
OUTPUT_TYPE_TO_PHASE = {
    'reviews': 'review-plan',
    'grouped': 'review-plan',           # Ambiguous - prefer explicit phase
    'validation': 'review-plan',        # Ambiguous - prefer explicit phase
    'backup': 'review-plan',
    'changes': 'review-plan',
    'applied_suggestions': 'apply-suggestions',
    'tasks': 'tasks',
    'implementation_summary': 'implement',
    'code_review': 'code-review',
    'code_review_issues': 'code-review',
    'code_review_grouped': 'code-review',
    'code_review_validation': 'code-review',
    'task_review': 'review-tasks',
    'task_review_grouped': 'review-tasks',
    'task_review_validation': 'review-tasks',
    'applied_fixes': 'apply-fixes',
    'applied_task_suggestions': 'apply-task-suggestions',
}

# Map output_type to actual filename (directory provides context)
OUTPUT_TYPE_TO_FILENAME = {
    'reviews': 'report.md',
    'implementation_summary': 'summary.md',
    'code_review': 'report.md',
    'tasks': 'tasks.md',
    'applied_suggestions': 'results.json',
    'applied_fixes': 'summary.md',
    'backup': 'backup.md',
    'changes': 'changes.md',
    'grouped': 'grouped.json',
    'validation': 'validation.json',
    'code_review_issues': 'issues.json',
    'code_review_grouped': 'grouped.json',
    'code_review_validation': 'validation.json',
    'task_review': 'report.md',
    'task_review_grouped': 'grouped.json',
    'task_review_validation': 'validation.json',
    'applied_task_suggestions': 'results.json',
}

def archive_file(path: Path) -> Path:
    """Archive an existing file by renaming it with a timestamp."""
    if not path.exists():
        return path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = path.stem
    suffix = path.suffix
    archived_path = path.parent / f"{stem}_{timestamp}{suffix}"

    path.rename(archived_path)
    return archived_path

def prepare_output_file(path: Path, mode: str = "rename") -> Path:
    """
    Prepare output file path, handling existing files.

    Args:
        path: Target output file path
        mode: "rename" (archive old file), "append", or "keep_first"

    Returns:
        Path to write to
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if mode == "append":
        return path

    if mode == "keep_first":
        return path

    archived = archive_file(path)
    print(f"Archived existing file: {archived}")
    return path

def append_to_changelog(path: Path, content: str, section_title: Optional[str] = None) -> None:
    """Append content to a changelog file with timestamp header."""
    if section_title is None:
        section_title = "Changes Applied"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"""
---

## Run: {timestamp}

### {section_title}
{content}
"""

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'a', encoding='utf-8') as f:
        f.write(entry)

def sanitize_prefix(name: str) -> str:
    """Sanitize a plan name for use as a directory/file prefix.

    This matches the derive_prefix() logic in orchestrators to ensure consistency.
    - Removes .md extension if present
    - Replaces non-alphanumeric characters (except - and _) with underscores
    - Removes consecutive underscores
    - Strips leading/trailing underscores
    """
    # Remove .md extension
    if name.endswith('.md'):
        name = name[:-3]
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '_', name)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing underscores
    return sanitized.strip('_')


def slugify_question(question: str) -> str:
    """Slugify a free-text question into a stable, collision-resistant dir name.

    This is intentionally distinct from :func:`sanitize_prefix` (which keeps
    case, uses underscores, and never truncates). The ``ask`` mode needs a
    short, lowercase, hyphenated slug with a hash suffix so that two *different*
    questions can never map to the same directory.

    Algorithm:
    - lowercase the question;
    - replace every run of non-alphanumeric characters with a single ``-``;
    - strip leading/trailing ``-``;
    - truncate to 50 chars (then strip a trailing ``-`` if truncation left one);
    - append ``-<hash8>`` where ``<hash8>`` is the first 8 hex chars of
      ``sha1`` over the **full, original** question text (utf-8).

    The hash suffix is mandatory: it disambiguates two different questions that
    normalize/truncate to the same 50-char prefix (or differ only in
    punctuation), so distinct questions never silently share a directory.
    Identical question text deliberately produces an identical slug (the
    legitimate resume case).

    Args:
        question: The verbatim, original free-text question.

    Returns:
        A slug of the form ``<normalized-prefix>-<hash8>``.
    """
    hash8 = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    normalized = re.sub(r'[^a-z0-9]+', '-', question.lower()).strip('-')
    if len(normalized) > 50:
        normalized = normalized[:50].rstrip('-')
    if not normalized:
        # All-punctuation / empty-after-normalization questions still get a
        # readable, unique dir via the (distinct) hash suffix.
        normalized = "question"
    return f"{normalized}-{hash8}"


def derive_prefix(plan_file: str) -> str:
    """Derive output prefix from a plan file path.

    Convenience wrapper around :func:`sanitize_prefix` that extracts
    the basename first.  This is the canonical implementation —
    orchestrators should delegate here instead of reimplementing.

    Args:
        plan_file: Path (string) to the plan file.

    Returns:
        Sanitized prefix string suitable for directory names.
    """
    basename = os.path.basename(plan_file)
    return sanitize_prefix(basename)


def find_output_dir(plan_file: str, *, guard_double_nesting: bool = True) -> str:
    """Find the output directory for a plan file.

    Canonical implementation of the pattern used by all three apply
    orchestrators.

    Args:
        plan_file: Path (string) to the plan file.
        guard_double_nesting: When ``True`` (default), if the parent
            directory already matches the derived prefix the parent is
            returned directly, preventing double-nesting.  The
            apply-code-fixes orchestrator relies on this guard; the
            apply-suggestions and apply-task-suggestions orchestrators
            originally did **not** have it, so they pass ``False``.

    Returns:
        String path to the output directory.
    """
    prefix = derive_prefix(plan_file)
    base_dir = os.path.dirname(plan_file) or "."

    # Guard against double-nesting if parent already is the prefix
    if guard_double_nesting and os.path.basename(base_dir) == prefix:
        return base_dir

    out_dir = os.path.join(base_dir, prefix)
    return out_dir


def get_phase_dir(plan_path: Path, phase: str) -> Path:
    """Get the output directory for a specific phase.

    Args:
        plan_path: Path to the plan file
        phase: Phase name (review-plan, tasks, implement, etc.)

    Returns:
        Path to the phase subdirectory
    """
    plan_dir = get_output_dir(plan_path)  # e.g., plans/my-feature/
    phase_dir = plan_dir / PHASE_DIRECTORIES.get(phase, phase)
    phase_dir.mkdir(parents=True, exist_ok=True)
    return phase_dir


def get_output_paths(plan_path: Path, output_type: str, phase: Optional[str] = None) -> Path:
    """Get the output file path for a given plan and output type.

    Creates a phase subdirectory within the plan's output folder.
    Files use simplified names since the directory provides context.

    Args:
        plan_path: Path to the plan file (e.g., /path/to/my-plan.md)
        output_type: Type of output file (e.g., "backup", "grouped", "report")
        phase: Optional phase name. If not provided, inferred from output_type.

    Returns:
        Path to the output file in the phase subdirectory
        (e.g., /path/to/my-plan/review-plan/report.md)
    """
    if phase is None:
        phase = OUTPUT_TYPE_TO_PHASE.get(output_type, 'misc')

    phase_dir = get_phase_dir(plan_path, phase)

    # Use mapped filename if available, otherwise construct from output_type
    if output_type in OUTPUT_TYPE_TO_FILENAME:
        filename = OUTPUT_TYPE_TO_FILENAME[output_type]
    else:
        suffix_map = {
            "backup": ".md",
            "changes": ".md",
            "grouped": ".json",
            "validation": ".json",
            "reviews": ".md",
            "code_review": ".md",
            "code_review_issues": ".json",
            "code_review_grouped": ".json",
            "code_review_validation": ".json",
            "tasks": ".md",
            "implementation_summary": ".md",
            "applied_suggestions": ".json",
            "applied_fixes": ".md",
        }
        suffix = suffix_map.get(output_type, ".json")
        filename = f"{output_type}{suffix}"

    return phase_dir / filename

def get_relative_output_path(plan_path: Path, output_type: str, phase: Optional[str] = None) -> str:
    """Get the relative path from the plan file to an output file in the phase subfolder.

    This is used for references in the plan file that link to generated files.

    Args:
        plan_path: Path to the plan file (e.g., /path/to/my-plan.md)
        output_type: Type of output file (e.g., "tasks", "implementation_summary")
        phase: Optional phase name. If not provided, inferred from output_type.

    Returns:
        Relative path from plan to output file (e.g., "my-plan/tasks/tasks.md")
    """
    output_path = get_output_paths(plan_path, output_type, phase)
    # Return relative path from plan's parent directory to the output file
    return str(output_path.relative_to(plan_path.parent))


def get_output_dir(plan_path: Path) -> Path:
    """Get the output directory for a plan's generated files.

    Args:
        plan_path: Path to the plan file

    Returns:
        Path to the output subfolder
    """
    prefix = sanitize_prefix(plan_path.stem)
    output_dir = plan_path.parent / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def cleanup_old_archives(directory: Path, max_age_days: int = 7) -> int:
    """Remove archived files older than specified days."""
    removed = 0
    now = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    archive_pattern = re.compile(r'.*_\d{8}_\d{6}\.\w+$')

    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue

        if not archive_pattern.match(file_path.name):
            continue

        file_age = now - file_path.stat().st_mtime
        if file_age > max_age_seconds:
            try:
                file_path.unlink()
                removed += 1
            except Exception:
                pass

    return removed
