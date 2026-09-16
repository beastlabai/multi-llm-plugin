#!/usr/bin/env python3
"""Finalize file tracking after implementation completes.

Usage:
    uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util finalize-tracking --state-file <path_to_state.json>

This script:
1. Gets all currently modified files (using shared git utils)
2. Subtracts pre_existing_changes recorded at start
3. Stores the result in state.tracked_files

This approach is more reliable than parsing subagent output because:
- Git utilities always capture all changes consistently
- Subagents may not report all files they touch
- Files created then modified only appear once
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Set

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.stream_bootstrap import bootstrap_streams
from utils import get_modified_files as utils_get_modified_files
from utils import get_staged_files


def get_all_modified_files() -> Set[str]:
    """
    Get all currently modified files (staged and unstaged).

    Uses the same git utilities as implement_orchestrator.py for consistency.
    """
    modified = set()

    try:
        # Use consistent utils from git_utils.py
        unstaged = set(f for f in utils_get_modified_files() if f)
        staged = set(f for f in get_staged_files() if f)
        modified = unstaged | staged
    except Exception as e:
        print(f"Warning: Failed to get modified files: {e}", file=sys.stderr)

    return modified


def main(argv=None) -> int:
    """Main entry point."""
    bootstrap_streams()
    parser = argparse.ArgumentParser(
        description="Finalize file tracking after implementation"
    )
    parser.add_argument(
        "--state-file",
        required=True,
        help="Path to state JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tracked without updating state"
    )
    # The write executor verifies every batch by a pre-batch snapshot
    # comparison, so it knows which files a batch changed better than a git
    # diff does. A file that was already modified before the phase started
    # sits in pre_existing_changes and would be subtracted below along with
    # the genuinely unrelated ones — measured 2026-09-16, a nine-batch run that
    # edited only an already-dirty index.html tracked one stray file and the
    # code review that followed would have reviewed that file instead.
    parser.add_argument(
        "--touched",
        action="append",
        default=[],
        metavar="PATH",
        help="A repo-relative file the write executor verified it changed; "
             "kept in tracked_files even when it was modified before the "
             "phase started (repeatable)",
    )
    # The implementation summary's back-reference used to be an inline `python -c`
    # in instructions/implement.md that put the skill dir on sys.path by hand. That
    # made the instruction core carry a second skill-dir substitution site and a
    # block of Python for an agent to retype; the insertion is deterministic, so it
    # belongs behind the same command that finalizes tracking.
    parser.add_argument(
        "--plan-file",
        help="Plan file to insert the implementation-summary reference into "
             "(requires --summary-file-relative)"
    )
    parser.add_argument(
        "--summary-file-relative",
        help="Summary path, relative to the plan's parent directory, as printed "
             "by the orchestrator's summary_file_relative field"
    )
    args = parser.parse_args(argv)

    if bool(args.plan_file) != bool(args.summary_file_relative):
        print(
            "ERROR: --plan-file and --summary-file-relative are used together: "
            "one names the plan to update, the other the reference to insert.",
            file=sys.stderr,
        )
        return 1

    state_path = Path(args.state_file)

    if not state_path.exists():
        print(f"ERROR: State file not found: {state_path}", file=sys.stderr)
        return 1

    # Load state
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse state file: {e}", file=sys.stderr)
        return 1

    # Validate state structure
    if not isinstance(state, dict):
        print("ERROR: State file is not a JSON object", file=sys.stderr)
        return 1

    # Ensure pre_existing_changes is a list
    pre_existing_raw = state.get("pre_existing_changes")
    if pre_existing_raw is not None and not isinstance(pre_existing_raw, list):
        print("WARNING: pre_existing_changes is not a list, treating as empty", file=sys.stderr)
        state["pre_existing_changes"] = []

    # Get current modified files (using shared utilities for consistency)
    current_modified = get_all_modified_files()

    # Subtract pre-existing changes
    pre_existing = set(state.get("pre_existing_changes", []))
    implementation_changes = current_modified - pre_existing

    # ...except for the files the executor verified it changed: those are
    # implementation changes whether or not they were dirty at the start.
    verified = [f for f in args.touched if f and f.strip()]
    kept_pre_existing = sorted(
        f for f in verified if f in pre_existing and f not in implementation_changes
    )
    implementation_changes |= set(verified)

    # Build tracked_files list
    tracked_files = []
    for f in sorted(implementation_changes):
        tracked_files.append({
            "path": f,
            "action": "modified",
            "task_id": "implementation"
        })

    print(f"Current modified files: {len(current_modified)}")
    print(f"Pre-existing changes: {len(pre_existing)}")
    if verified:
        print(
            f"Verified by the write executor: {len(verified)}"
            + (
                f" ({len(kept_pre_existing)} kept although modified before the "
                f"phase started: {', '.join(kept_pre_existing)})"
                if kept_pre_existing else ""
            )
        )
    print(f"Implementation changes: {len(implementation_changes)}")

    if args.dry_run:
        print("\nDry run - would track these files:")
        for entry in tracked_files:
            print(f"  - {entry['path']}")
        return 0

    # Update state
    state["tracked_files"] = tracked_files
    state["updated_at"] = datetime.now().isoformat()

    # Write state back atomically
    try:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except IOError as e:
        print(f"ERROR: Failed to write state file: {e}", file=sys.stderr)
        return 1

    print(f"\nTracked {len(tracked_files)} implementation files:")
    for entry in tracked_files:
        print(f"  - {entry['path']}")

    print(f"\nState file updated: {state_path}")

    if args.plan_file:
        code = _insert_summary_reference(args.plan_file, args.summary_file_relative)
        if code:
            return code
    return 0


def _insert_summary_reference(plan_file: str, summary_relative: str) -> int:
    """Idempotently add the implementation-summary link to the plan."""
    from utils import insert_implementation_summary_reference

    plan_path = Path(plan_file)
    if not plan_path.exists():
        print(f"ERROR: Plan file not found: {plan_path}", file=sys.stderr)
        return 1
    content = plan_path.read_text(encoding="utf-8")
    updated = insert_implementation_summary_reference(content, summary_relative)
    if updated != content:
        plan_path.write_text(updated, encoding="utf-8")
        print(f"Updated {plan_path} with a reference to {summary_relative}")
    else:
        print(f"{plan_path} already references {summary_relative}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
