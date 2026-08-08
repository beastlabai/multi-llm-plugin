#!/usr/bin/env python3
"""``multi_llm.py util restore-batch`` — explicit, verified rollback (D3l).

D3h's per-task pre-batch snapshots become *usable* through this command. Three
properties are the whole design:

* **Never automatic.** A partial application stops the write loop and leaves the
  bytes in place; the exit-3 message *names* this command rather than silently
  reverting, because an automatic revert would destroy the half-applied state a
  human needs to inspect.
* **Verified.** It restores snapshot bytes for exactly the files the audit record
  lists as touched, and **refuses** a file whose current sha256 no longer matches
  what the audit recorded at completion — someone edited it since the run, and
  reverting their edit would be a second data-loss bug. ``--force`` overrides,
  deliberately explicitly.
* **Auditable.** It appends its own ``restored`` line to
  ``{phase}/subagent_audit.jsonl``, so a rollback is an event in the same log the
  dispatch is.

``git`` is never assumed as the rollback mechanism: the target file is frequently
untracked, which every ``git diff`` / ``git checkout`` variant handles wrongly for
this purpose (D3g).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:  # pragma: no cover - import shim mirrors the rest of utils/
    from .stream_bootstrap import bootstrap_streams
    from .write_safety import AuditLog, TaskSnapshot
except ImportError:  # pragma: no cover - direct-script invocation
    from stream_bootstrap import bootstrap_streams
    from write_safety import AuditLog, TaskSnapshot


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOTHING_TO_DO = 2
EXIT_NEEDS_HUMAN = 3


def _task_ids(phase_dir: str, args: argparse.Namespace) -> List[str]:
    if args.task_id:
        return list(args.task_id)
    audit = AuditLog(phase_dir)
    failed = audit.failed_task_ids()
    return failed


def restore(
    *,
    phase_dir: str,
    task_ids: Sequence[str],
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Restore each task's snapshot, verifying against the audit record.

    Returns ``{task_id: {relative_path: outcome}}``. An outcome beginning
    ``skipped:`` is a refusal, never a silent no-op.
    """
    audit = AuditLog(phase_dir)
    results: Dict[str, Dict[str, str]] = {}
    for task_id in task_ids:
        snapshot = TaskSnapshot.load(phase_dir, task_id)
        if snapshot is None:
            results[task_id] = {
                "*": (
                    "skipped: no pre-batch snapshot under "
                    f"{Path(phase_dir) / 'pre_batch' / task_id}"
                )
            }
            continue
        recorded = audit.completion_for(task_id)
        touched = list((recorded or {}).get("files_touched") or [])
        expected = audit.touched_with_hashes(task_id)
        if recorded is None:
            # No completion record: the run died before classifying. Restore
            # everything the snapshot covers, with no hash expectation to verify
            # against — and say so rather than implying verification happened.
            touched = [entry.relative for entry in snapshot.entries]
            expected = {}
        if not touched:
            results[task_id] = {
                "*": "skipped: the audit record lists no touched files for this task"
            }
            continue
        if dry_run:
            results[task_id] = {path: "would restore" for path in touched}
            continue
        report = snapshot.restore(touched, expected=expected or None, force=force)
        audit.restored(task_id=task_id, report=report, force=force)
        results[task_id] = report
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Exit 0 restored, 2 nothing to restore, 3 a refusal, 1 error."""
    bootstrap_streams()
    parser = argparse.ArgumentParser(
        prog="multi_llm.py util restore-batch",
        description=(
            "Restore a failed write batch from its pre-batch snapshot, refusing "
            "any file that changed after the run."
        ),
    )
    parser.add_argument(
        "--phase-dir", required=True,
        help="The write phase's directory (where pre_batch/ and "
             "subagent_audit.jsonl live).",
    )
    parser.add_argument(
        "--task-id", action="append", default=[],
        help="Task id to restore. Repeatable. Omit with --all-failed.",
    )
    parser.add_argument(
        "--all-failed", action="store_true",
        help="Restore every task the audit log records as failed, partially "
             "applied or policy-violating.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Restore even a file whose current sha256 no longer matches what the "
             "audit recorded at completion (i.e. one edited since the run). "
             "Never the default: that edit would be destroyed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be restored and change nothing.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the result map as JSON.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    phase_dir = str(Path(args.phase_dir).resolve())
    if not Path(phase_dir).is_dir():
        print(f"ERROR: phase dir not found: {phase_dir}", file=sys.stderr)
        return EXIT_ERROR
    if not args.task_id and not args.all_failed:
        print(
            "ERROR: pass --task-id <id> (repeatable) or --all-failed. Restoring "
            "everything by default would be exactly the automatic revert this "
            "command exists to avoid.",
            file=sys.stderr,
        )
        return EXIT_NEEDS_HUMAN

    task_ids = _task_ids(phase_dir, args)
    if not task_ids:
        print(
            f"Nothing to restore: {AuditLog(phase_dir).path} records no failed "
            f"write task.",
        )
        return EXIT_NOTHING_TO_DO

    results = restore(
        phase_dir=phase_dir, task_ids=task_ids, force=args.force,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for task_id, report in results.items():
            print(f"{task_id}:")
            for path, outcome in sorted(report.items()):
                print(f"  {path}: {outcome}")

    refused = [
        (task_id, path, outcome)
        for task_id, report in results.items()
        for path, outcome in report.items()
        if outcome.startswith("skipped:")
    ]
    if refused:
        print(
            f"\n{len(refused)} file(s) were NOT restored. A file that changed "
            f"after the run is refused on purpose — inspect it, then re-run with "
            f"--force if you really want the snapshot bytes back.",
            file=sys.stderr,
        )
        return EXIT_NEEDS_HUMAN
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
