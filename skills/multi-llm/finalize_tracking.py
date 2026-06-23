#!/usr/bin/env python3
"""Finalize file tracking after implementation completes.

Usage:
    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/finalize_tracking.py --state-file <path_to_state.json>

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


def main() -> int:
    """Main entry point."""
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
    args = parser.parse_args()

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
