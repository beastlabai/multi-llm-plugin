#!/usr/bin/env python3
"""CLI utility for managing prior batch changes in a JSONL file.

Commands: read, append, clear.
Storage: {output-dir}/prior_changes.jsonl
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

try:
    from .stream_bootstrap import bootstrap_streams
except ImportError:
    # Direct script invocation (`python utils/prior_changes.py`): sys.path[0]
    # is the utils/ directory, so import the sibling module directly.
    from stream_bootstrap import bootstrap_streams

FILENAME = "prior_changes.jsonl"
DEFAULT_TAIL = 15
MAX_CHARS = 2000
DEFAULT_TEXT = "(none — this is the first batch)"


def _jsonl_path(output_dir: str) -> Path:
    return Path(output_dir) / FILENAME


def _load_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.warn(f"Skipping corrupted JSONL line: {line!r}", stacklevel=2)
            continue
    return entries


def render(output_dir: str) -> str:
    """The prior-batch context block, exactly as `read` prints it.

    Extracted from :func:`cmd_read` so the D3n write-batch scheduler can bind
    ``{prior_changes_context}`` **in process**, per dispatch, without shelling out
    to this script — and, more importantly, so both invokers substitute
    byte-identical text. The log is always **replayed from disk** rather than
    reconstructed from an in-memory list, which is what makes a resumed batch see
    the same context an uninterrupted run would have.
    """
    entries = _load_entries(_jsonl_path(output_dir))
    if not entries:
        return DEFAULT_TEXT

    total = len(entries)
    tail = min(DEFAULT_TAIL, total)
    output = ""
    while tail > 0:
        skipped = total - tail
        visible = entries[total - tail:]

        lines: list[str] = []
        if skipped > 0:
            lines.append(f"... and {skipped} earlier batches applied miscellaneous changes")

        for i, entry in enumerate(visible, start=skipped + 1):
            eid = entry.get("id", "?")
            summary = entry.get("summary", "")
            lines.append(f"{i}. **Batch {i}** ({eid}): {summary}")

        output = "\n".join(lines)
        if len(output) <= MAX_CHARS:
            break
        tail -= 1
    return output


def append(output_dir: str, entry_id: str, phase: str, summary: str) -> bool:
    """Append one entry, skipping a duplicate ``(id, phase)``. True when written.

    The in-process half of ``append``, for the same reason :func:`render` exists.
    Idempotent on ``(id, phase)`` — which is why the write task carries a stable
    ``prior_changes_id`` rather than a loop index: a re-dispatched batch must not
    double its own summary into the context every later batch reads.
    """
    path = _jsonl_path(output_dir)
    entries = _load_entries(path)
    for entry in entries:
        if entry.get("id") == entry_id and entry.get("phase") == phase:
            return False
    record = {
        "id": entry_id,
        "phase": phase,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return True


def entries(output_dir: str) -> list[dict]:
    """Every recorded entry, in order (resume replay, and the batch report)."""
    return _load_entries(_jsonl_path(output_dir))


def cmd_read(args: argparse.Namespace) -> None:
    entries = _load_entries(_jsonl_path(args.output_dir))
    if not entries:
        print(DEFAULT_TEXT)
        return

    total = len(entries)
    tail = min(DEFAULT_TAIL, total)

    # Build output by reducing visible entries until it fits within MAX_CHARS.
    # This ensures we never cut mid-line or truncate the most recent entries.
    while tail > 0:
        skipped = total - tail
        visible = entries[total - tail :]

        lines: list[str] = []
        if skipped > 0:
            lines.append(f"... and {skipped} earlier batches applied miscellaneous changes")

        for i, entry in enumerate(visible, start=skipped + 1):
            eid = entry.get("id", "?")
            summary = entry.get("summary", "")
            lines.append(f"{i}. **Batch {i}** ({eid}): {summary}")

        output = "\n".join(lines)
        if len(output) <= MAX_CHARS:
            break
        # Reduce visible entries to fit within budget
        tail -= 1

    print(output)


def cmd_append(args: argparse.Namespace) -> None:
    summary = args.summary if args.summary is not None else sys.stdin.read().strip()

    path = _jsonl_path(args.output_dir)
    entries = _load_entries(path)

    for entry in entries:
        if entry.get("id") == args.id and entry.get("phase") == args.phase:
            print(f"Notice: entry {args.id}/{args.phase} already exists, skipping", file=sys.stderr)
            return

    record = {
        "id": args.id,
        "phase": args.phase,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Appended entry {args.id}/{args.phase}", file=sys.stderr)


def cmd_clear(args: argparse.Namespace) -> None:
    path = _jsonl_path(args.output_dir)
    if path.exists():
        path.unlink()


def main(argv=None) -> int:
    bootstrap_streams()
    parser = argparse.ArgumentParser(description="Manage prior batch changes")
    sub = parser.add_subparsers(dest="command", required=True)

    read_p = sub.add_parser("read")
    read_p.add_argument("--output-dir", required=True)

    append_p = sub.add_parser("append")
    append_p.add_argument("--output-dir", required=True)
    append_p.add_argument("--id", required=True)
    append_p.add_argument("--phase", required=True)
    summary_grp = append_p.add_mutually_exclusive_group(required=True)
    summary_grp.add_argument("--summary", default=None)
    summary_grp.add_argument("--summary-stdin", action="store_true")

    clear_p = sub.add_parser("clear")
    clear_p.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)
    {"read": cmd_read, "append": cmd_append, "clear": cmd_clear}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
