#!/usr/bin/env python3
"""CLI utility for managing prior batch changes in a JSONL file.

Commands: read, append, clear.
Storage: {output-dir}/prior_changes.jsonl
"""

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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

#: Hex digits kept from the member-id digest in :func:`entry_id`. Eight is the
#: same width the state manager's group ids are truncated to for display, and it
#: is far wider than the handful of batches one phase ever produces.
ENTRY_ID_DIGEST_CHARS = 8

#: What :func:`entry_id` uses as the base when a batch carries no content key of
#: its own, so the two invokers (the Python bridge and a native agent reading the
#: serialized batch) still spell the same id.
ENTRY_ID_KEYLESS_BASE = "batch"


def stable_key(candidates: Sequence[Any], fallback: str) -> str:
    """The first non-empty candidate, else *fallback*. Never a loop index.

    The content-key primitive :func:`entry_id` is built on, and the one the write
    bridge and the batchers both spell their member ids with — it lives here so
    the id semantics have exactly one owner.
    """
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return fallback


def _short_digest(member_ids: Sequence[str]) -> str:
    """A stable short digest of an ordered member-id list."""
    payload = "\n".join(member_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:ENTRY_ID_DIGEST_CHARS]


def entry_id(key: str, member_ids: Sequence[str], *, fallback: str) -> str:
    """The ``--id`` for one batch: ``{key}#{digest of member_ids}``.

    The id handed to :func:`append` has to be **both** unique per batch and
    stable across a re-dispatch, and getting either half wrong fails silently:

    * a **colliding** id silently DROPS an entry. ``append`` is idempotent on
      ``(id, phase)``, so the second batch that presents an id an earlier batch
      already used is skipped with a `Notice:` on stderr and exit 0 — and every
      later batch then reads a prior-changes context missing its predecessors,
      which is the whole signal the step exists to provide. This is not
      hypothetical: the batchers emit **many** batches per file key and per
      section key (``MAX_FIXES_PER_BATCH``, the char caps, and the one-per-batch
      isolation of security/HIGH fixes), so keying on the bare file/section key
      records only the first batch of each.
    * an **unstable** id silently DOUBLES one — that is the failure ``append``'s
      idempotency exists to prevent, and it is why the digest is taken over a
      stable *content* key (the batch's own member ids) rather than over a loop
      index, which moves the moment a resumed run re-materializes a different
      subset of batches.

    Args:
        key: The batch's content key (``file_key``/``section_key``). May be empty.
        member_ids: The batch's member ids, in order — the same list the write
            task carries as ``expected_ids``, so both invokers digest the same
            bytes and derive the same id for the same batch.
        fallback: Used only when *member_ids* is empty, i.e. when there is no
            content to digest at all. The batchers never emit an empty batch, so
            this is a degenerate-input guard rather than a live path.

    Returns:
        ``"{key}#{digest}"``, or ``"batch#{digest}"`` for a batch with no key,
        or *fallback*/*key* when there are no member ids to digest.
    """
    ids = [text for text in (str(m or "").strip() for m in member_ids) if text]
    base = str(key or "").strip()
    if not ids:
        return base or fallback
    return f"{base or ENTRY_ID_KEYLESS_BASE}#{_short_digest(ids)}"


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
    double its own summary into the context every later batch reads. The other
    half of that contract is :func:`entry_id`, which is what makes the id unique
    per batch: idempotency on a key two different batches share is a *dropped*
    entry, silently, at exit 0.
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
        # Local naive clock, matching `state_manager.py`'s `updated_at` — a
        # UTC stamp here (F13, audited 2026-09-16) read two hours off from
        # `state.json` and `subagent_audit.jsonl` for the same instant, both
        # of which use `datetime.now()`.
        "timestamp": datetime.now().isoformat(),
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
        # See the in-process `append`'s matching comment (F13).
        "timestamp": datetime.now().isoformat(),
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
