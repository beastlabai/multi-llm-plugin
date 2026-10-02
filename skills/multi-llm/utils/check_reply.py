#!/usr/bin/env python3
"""``util check-reply``: is the reply a host persisted a well-formed batch result?

A native host (Claude Code, Codex, OpenCode driving subagents itself) runs each
validation batch as a subagent and writes the reply to the task's
``output_path``. Nothing checked that file until reaggregation, and a reply that
did not parse was handled by whatever the host improvised: the 2026-10-01 Claude
Code run lost a closing ``]`` on 3 of 14 Sonnet batches, and the host re-ran one,
hand-edited another and "used the corrected one the subagent included" for a
third (plans/2026-10-01-three-host-racing-run.md, issue 11). A hand-repaired
reply is a verdict nobody produced.

This is the check the host runs right after writing a batch, so a bad reply
costs one re-ask of the same subagent instead of a repair::

    util check-reply --tasks-file <phase>/validation_tasks.json [--task-id ID]

For every task it prints one ``[REPLY_OK]`` or ``[REPLY_INVALID]`` line and exits
1 when any is invalid. What it checks, per task:

* the ``output_path`` file exists, is not empty and parses as JSON;
* for a ``validation``/``revalidation`` task: the document is ``{"groups": [...]}``
  (a bare list is tolerated, as the merge tolerates it), every entry carries a
  ``status`` that is one of the three verdicts, and the entries cover the task's
  ``group_ids`` exactly — a missing, extra or foreign ``group_hash`` is reported
  by name, because the merge discards a foreign hash and marks a missing one
  ``validation_failed``.

Other kinds are only checked for being JSON: their shape is checked by their own
reaggregation. Advisory about nothing else: it never edits or deletes a file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

try:
    from .stream_bootstrap import bootstrap_streams
except ImportError:  # pragma: no cover - direct-script import shim
    from stream_bootstrap import bootstrap_streams

_VERDICTS = ("valid", "invalid", "needs-human-decision")
_GROUP_KINDS = ("validation", "revalidation")


def _entries(document: Any) -> Optional[List[Any]]:
    if isinstance(document, dict):
        groups = document.get("groups")
        return groups if isinstance(groups, list) else None
    return document if isinstance(document, list) else None


def check_task_reply(task: Dict[str, Any], phase_dir: str) -> List[str]:
    """Return the problems with one task's persisted reply; empty means fine."""
    path = os.path.join(phase_dir, task["output_name"]) if task.get("output_name") \
        else str(task.get("output_path") or "")
    if not path or not os.path.isfile(path):
        return [f"no reply file at {path or '(no output_path)'}"]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        return [f"reply file unreadable ({exc})"]
    if not text.strip():
        return ["reply file is empty"]
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"reply is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"]
    if task.get("kind") not in _GROUP_KINDS:
        return []

    entries = _entries(document)
    if entries is None:
        return ['reply has no "groups" list']
    problems: List[str] = []
    seen: List[str] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"groups[{position}] is not an object")
            continue
        if entry.get("status") not in _VERDICTS:
            problems.append(
                f"groups[{position}] status {entry.get('status')!r} is not one of "
                + ", ".join(_VERDICTS)
            )
        group_hash = entry.get("group_hash")
        if isinstance(group_hash, str) and group_hash:
            seen.append(group_hash)
        else:
            problems.append(f"groups[{position}] has no group_hash")
    expected = [str(item) for item in task.get("group_ids") or []]
    if expected:
        missing = [item for item in expected if item not in seen]
        foreign = sorted({item for item in seen if item not in expected})
        if missing:
            problems.append("missing group_hash: " + ", ".join(missing))
        if foreign:
            problems.append("foreign group_hash: " + ", ".join(foreign))
        if len(seen) != len(set(seen)):
            problems.append("a group_hash appears more than once")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    bootstrap_streams()
    parser = argparse.ArgumentParser(
        prog="multi_llm.py util check-reply",
        description="Check the replies persisted for a task file's batches.",
    )
    parser.add_argument("--tasks-file", required=True,
                        help="a materialized *_tasks.json (validation_tasks.json, ...)")
    parser.add_argument("--task-id", action="append", default=[],
                        help="check only this task (repeatable); default: every task")
    args = parser.parse_args(argv)

    try:
        from utils.subagent_tasks import tasks_of
        with open(args.tasks_file, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        tasks = tasks_of(document)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {args.tasks_file}: {exc}", file=sys.stderr)
        return 2
    phase_dir = os.path.dirname(os.path.abspath(args.tasks_file))
    if args.task_id:
        tasks = [task for task in tasks if task.get("task_id") in args.task_id]
        if not tasks:
            print(f"ERROR: no task matches {', '.join(args.task_id)}", file=sys.stderr)
            return 2

    bad = 0
    for task in tasks:
        problems = check_task_reply(task, phase_dir)
        if problems:
            bad += 1
            print(f"[REPLY_INVALID] {task.get('task_id')}: " + "; ".join(problems))
        else:
            print(f"[REPLY_OK] {task.get('task_id')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
