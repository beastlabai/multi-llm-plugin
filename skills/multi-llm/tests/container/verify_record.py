#!/usr/bin/env python3
"""Re-hash a run bundle against the record that cited it (C10).

``make e2e-verify-record REC=<path>`` answers one question: is the
``runs/<utc>/<row-id>/`` directory still on this machine the one the committed
record was derived from?

Three outcomes, and the middle one is the point of the split:

* **match** — every file's sha256 and the aggregate agree; the evidence behind a
  capability promotion is intact and inspectable.
* **missing** — the bundle is gone. That is *not* a failure: ``runs/`` is
  gitignored, prunable, and ``make e2e-clean`` erases it by design. The record
  still stands on its own, and this tool's job is to say precisely what is no
  longer inspectable rather than to imply the claim evaporated with it.
* **mismatch** — the directory exists but its bytes are not what the record
  names. That is a real finding: something rewrote the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_bundle(root: Path) -> tuple[dict[str, str], str]:
    """Per-file digests plus one aggregate over the sorted list."""
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            # POSIX names: a record is portable, and its aggregate hashes them.
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    aggregate = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
        .encode("utf-8")
    ).hexdigest()
    return files, aggregate


def _resolve_root(raw: str, record_path: Path) -> tuple[Path | None, list[Path]]:
    """Find the bundle, trying every anchor a record's `root` might be written in.

    `bundle.root` is written relative to the container directory (so a record is
    portable between checkouts and carries no machine-specific absolute path), but
    a hand-written or older record may hold an absolute path or one relative to
    the repo root. Trying each and *naming what was tried* is the difference
    between "the bundle is gone" and "I looked in the wrong place" — and this tool
    reports MISSING as a non-failure, so guessing wrong would quietly excuse a
    bundle that is actually right there.

    An empty `raw` is rejected here rather than resolved: `base / Path("")` is
    `base`, so every anchor would collapse onto its own directory — the container
    dir is real, so the walk would succeed and hash the whole container tree
    (`runs/` included) into a spurious MISMATCH. "The record names no root" is a
    different fact from "the bytes moved", and only the caller can say so.
    """
    if not raw.strip():
        raise ValueError("the record's bundle block names no root")
    container_dir = record_path.resolve().parent.parent
    # container_dir is <repo>/skills/multi-llm/tests/container
    repo_root = container_dir.parent.parent.parent.parent
    candidate = Path(raw)
    tried: list[Path] = []
    for base in (None, container_dir, repo_root, Path.cwd()):
        resolved = candidate if base is None and candidate.is_absolute() \
            else (base / candidate if base is not None else None)
        if resolved is None:
            continue
        tried.append(resolved)
        if resolved.is_dir():
            return resolved, tried
    return None, tried


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", help="path to records/<utc>-<host>.json")
    parser.add_argument("--root", default=None,
                        help="override the bundle root named in the record")
    args = parser.parse_args(argv)

    record_path = Path(args.record)
    if not record_path.is_file():
        print(f"verify-record: no such record: {record_path}", file=sys.stderr)
        return 1
    record = json.loads(record_path.read_text(encoding="utf-8"))
    bundle = record.get("bundle") or {}
    if not bundle:
        print(f"verify-record: {record_path.name} carries no bundle block "
              f"— nothing to verify against", file=sys.stderr)
        return 1

    try:
        root, tried = _resolve_root(args.root or bundle.get("root") or "", record_path)
    except ValueError as exc:
        print(f"verify-record: {record_path.name}: {exc} "
              f"— pass --root <dir> to name one", file=sys.stderr)
        return 1
    if root is None:
        print("verify-record: MISSING — the bundle is not on this machine.")
        print("  looked in:")
        for candidate in tried:
            print(f"    {candidate}")
        print("  The record stands on its own; the raw bundle is an ephemeral")
        print("  diagnostic and may have been pruned (`make e2e-clean`).")
        if bundle.get("archive_url"):
            print(f"  archived at: {bundle['archive_url']}")
        return 0

    files, aggregate = hash_bundle(root)
    if aggregate == bundle.get("aggregate"):
        print(f"verify-record: MATCH — {len(files)} files, aggregate {aggregate[:16]}…")
        return 0

    print(f"verify-record: MISMATCH — {root}")
    recorded = bundle.get("files") or {}
    for name in sorted(set(recorded) | set(files)):
        was, now = recorded.get(name), files.get(name)
        if was == now:
            continue
        if was is None:
            print(f"  + {name} (not in the record)")
        elif now is None:
            print(f"  - {name} (recorded, now absent)")
        else:
            print(f"  ~ {name} ({was[:12]}… -> {now[:12]}…)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
