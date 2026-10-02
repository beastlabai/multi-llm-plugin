#!/usr/bin/env python3
"""Finalize file tracking after implementation completes.

Usage:
    uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" util finalize-tracking --state-file <path_to_state.json>

This script:
1. Gets all currently modified files (using shared git utils)
2. Subtracts pre_existing_changes recorded at start
3. Stores the result in state.tracked_files

With ``--only-touched`` it skips all of that and stores exactly the
``--touched`` paths: the route for an implementer that ran outside
``/multi-llm implement`` and so recorded no baseline to subtract.

This approach is more reliable than parsing subagent output because:
- Git utilities always capture all changes consistently
- Subagents may not report all files they touch
- Files created then modified only appear once
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.stream_bootstrap import bootstrap_streams
from utils import get_modified_files as utils_get_modified_files
from utils import (
    baseline_covers,
    get_staged_files,
    list_untracked_files,
    subtract_baseline,
)
from utils.git_utils import ignored_paths
from utils.state_manager import compute_plan_hash

# A tracked-file dump is a reference list, not a reading; state.json holds all
# of it. Measured 2026-09-16: a run that tracked 1 511 files spent 1 512 of its
# 1 585 log lines on this list, burying the batch outcomes either side of it.
MAX_LISTED_PATHS = 20


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


def _expand_uncovered_directories(entries: Set[str]) -> Set[str]:
    """Replace each surviving ``dir/`` entry with the files inside it.

    The census is deliberately directory-granular (see
    ``git_utils.get_modified_files``), which is what lets the stored baselines
    subtract a whole pre-existing tree. What survives that subtraction is a
    directory the *phase itself* created, and a directory path is useless to
    every consumer: `capture_diff_hunks` finds no per-file diff for it and
    skips it silently, so on the native leg (no ``--touched``) `review-code`
    reports the entry and reviews none of the new module's files (F2,
    2026-09-16).

    Only the survivors are expanded, so a baseline-covered ``plans/`` with
    1 511 files under it is never listed — that expansion is the regression
    6260c9e reverted, and it must not come back. An entry git can no longer
    list is kept as-is rather than silently dropped.
    """
    expanded: Set[str] = set()
    for entry in entries:
        if not entry.endswith("/"):
            expanded.add(entry)
            continue
        files = list_untracked_files(entry)
        expanded.update(files or {entry})
    return expanded


def _write_state_atomic(state_path: Path, state: dict) -> None:
    """Keep the previous state intact if either finalization write fails."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=str(state_path.parent), suffix=".tmp", prefix=".tracking_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        os.replace(temp_path, str(state_path))
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _restamp_plan_hash(state_path: Path, plan_path: Path) -> None:
    """Re-stamp ``plan_hash`` after this command's own edit to the plan.

    ``--plan-file`` makes this a plan writer, and it writes after the state
    file above. Left alone, the next ``StateManager`` built on this plan hashes
    content the stored ``plan_hash`` no longer matches, records
    ``plan_changed`` and ``implement --resume`` reports "Plan has changed since
    last session" about the plugin's own back-reference insertion. The executor
    leg re-stamps at exactly this point (``WriteBatchScheduler._finalize``,
    after ``_insert_plan_reference``); this is the native leg's counterpart.

    Best-effort and never fatal: a failure here costs a spurious warning, and
    the census — this command's actual job — is already on disk.
    """
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or "plan_hash" not in state:
            # No hash recorded means nothing reads one; do not invent a field.
            return
        state["plan_hash"] = compute_plan_hash(plan_path)
        state.pop("plan_changed", None)
        _write_state_atomic(state_path, state)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not re-stamp plan_hash: {e}", file=sys.stderr)


def _print_path_list(entries) -> None:
    """Print at most ``MAX_LISTED_PATHS`` of the tracked entries."""
    for entry in entries[:MAX_LISTED_PATHS]:
        print(f"  - {entry['path']}")
    remaining = len(entries) - MAX_LISTED_PATHS
    if remaining > 0:
        print(f"  … and {remaining} more (see state.json.tracked_files)")


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
        help="A repo-relative file the implementation changed; kept in "
             "tracked_files even when it was modified before the phase started "
             "(repeatable). It only ADDS paths and never removes a dirty one: "
             "without --only-touched the whole dirty census minus "
             "pre_existing_changes is still tracked beside it",
    )
    # The exact-scope route for an implementer that ran outside `/multi-llm
    # implement` and so recorded no `pre_existing_changes` baseline (review-scope
    # plan §2 D2.6). Plain `--touched` only unions its paths into the census, so
    # with no baseline every unrelated dirty or untracked path stays in — and a
    # non-empty `tracked_files` silences review-code's `[SCOPE_FALLBACK]` and is
    # labelled as the phase's own scope, so the over-wide list goes unnoticed.
    parser.add_argument(
        "--only-touched",
        action="store_true",
        help="Make tracked_files exactly the --touched paths (de-duplicated, "
             "sorted): the git census, the pre_existing_changes subtraction and "
             "the directory expansion are skipped, and pre_existing_changes is "
             "neither read nor written. Requires at least one --touched",
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

    args.touched = _drop_ignored(args.touched)
    explicit_scope = sorted({f.strip() for f in args.touched if f and f.strip()})
    if args.only_touched and not explicit_scope:
        # Refused before the state file is even read: an empty explicit list
        # would empty `tracked_files`, and review-code would then quietly fall
        # back to the whole `git diff` — the scope this flag exists to replace.
        print(
            "ERROR: --only-touched needs at least one --touched <path>: it makes "
            "tracked_files exactly those paths, and an empty list would send "
            "review-code back to its git-diff fallback.",
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

    if args.only_touched:
        tracked_files = _tracked_entries(explicit_scope)
        print(
            f"Explicit scope (--only-touched): {len(tracked_files)} files "
            f"(the git census and pre_existing_changes were not consulted)"
        )
    else:
        tracked_files = _census_tracked_files(state, args.touched)

    if args.dry_run:
        print("\nDry run - would track these files:")
        _print_path_list(tracked_files)
        return 0

    # Update state
    state["tracked_files"] = tracked_files
    state["updated_at"] = datetime.now().isoformat()

    try:
        _write_state_atomic(state_path, state)
    except OSError as e:
        print(f"ERROR: Failed to write state file: {e}", file=sys.stderr)
        return 1

    print(f"\nTracked {len(tracked_files)} implementation files:")
    _print_path_list(tracked_files)

    print(f"\nState file updated: {state_path}")

    if args.plan_file:
        code = _insert_summary_reference(args.plan_file, args.summary_file_relative)
        if code:
            return code
        _restamp_plan_hash(state_path, Path(args.plan_file))
    return 0


def _drop_ignored(paths) -> List[str]:
    """``paths`` without the ones git ignores (a transcript, a build cache).

    An ignored path is not a change to the repository, and ``tracked_files``
    becomes ``review-code``'s scope: the host's own ``logs/*.jsonl`` reached it
    through ``--touched`` (2026-10-02, Codex run). Said once, naming what went.
    """
    named = [p.strip() for p in paths if p and p.strip()]
    ignored = ignored_paths(named)
    if ignored:
        print(
            f"Left out {len(ignored)} gitignored path(s): "
            f"{', '.join(sorted(ignored)[:5])}"
            f"{', …' if len(ignored) > 5 else ''}"
        )
    return [p for p in paths if p.strip() not in ignored]


def _tracked_entries(paths) -> List[Dict[str, str]]:
    """``tracked_files`` entries for *paths*, in the order given."""
    return [
        {"path": f, "action": "modified", "task_id": "implementation"}
        for f in paths
    ]


def _census_tracked_files(state: dict, touched: List[str]) -> List[Dict[str, str]]:
    """The default scope: the dirty census, minus the baseline, plus ``--touched``.

    Prints the census counts. May rewrite a malformed ``pre_existing_changes``
    on *state* to ``[]`` (warned), which the caller then persists.
    """
    # Ensure pre_existing_changes is a list
    pre_existing_raw = state.get("pre_existing_changes")
    if pre_existing_raw is not None and not isinstance(pre_existing_raw, list):
        print("WARNING: pre_existing_changes is not a list, treating as empty", file=sys.stderr)
        state["pre_existing_changes"] = []

    # Get current modified files (using shared utilities for consistency)
    current_modified = get_all_modified_files()

    # Subtract pre-existing changes. Prefix-aware, because a baseline entry is
    # whatever `git status --porcelain` printed, and an untracked *directory* is
    # one `plans/` line standing for every path beneath it: a plain set
    # difference subtracts none of them.
    pre_existing = list(state.get("pre_existing_changes", []))
    implementation_changes = set(subtract_baseline(current_modified, pre_existing))

    # A directory that survived the subtraction is one this phase created:
    # expand it to the files inside it, which is what every consumer needs.
    implementation_changes = _expand_uncovered_directories(implementation_changes)

    # ...except for the files the executor verified it changed: those are
    # implementation changes whether or not they were dirty at the start.
    verified = [f for f in touched if f and f.strip()]
    kept_pre_existing = sorted(
        f for f in verified
        if baseline_covers(f, pre_existing) and f not in implementation_changes
    )
    implementation_changes |= set(verified)

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
    return _tracked_entries(sorted(implementation_changes))


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
