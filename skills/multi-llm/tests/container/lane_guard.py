#!/usr/bin/env python3
"""The credential guards of ``make e2e-container-latest-host``.

That target is the one lane that hands a live key to an image whose CLIs are
installed ``@latest``: unreviewed, unhashed binaries with outbound network. The
recipe runs two checks from this file, and both fail closed: anything but exit 0
(a refusal, a crash, a bad argument) stops make before the next line.

* ``unpinned-host``, before the image is built. The lane's key must be set (a
  keyless run skips every row, so it would build the image and prove nothing),
  and the opt-in must be exactly ``MLLM_ALLOW_UNPINNED_HOST_LANE=1``. A refusal
  names every vendor key it found exported; the recipe unsets those for the
  run, because none of them can become this lane's credential.
* ``rows-ran``, after a green pytest. At least one run record under
  ``records/`` must be newer than the stamp file the recipe made before pytest:
  a host-lane row that ran always writes one, so none means no row ran, and a
  lane that covered nothing is NOT-RUN rather than a pass.

The reasons behind each rule sit above the target in the Makefile; this module
holds the rules themselves, so ``tests/test_makefile_lanes.py`` can execute
them. Stdlib only: it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

#: The only credential this lane gives a container.
LANE_KEY = "OPENROUTER_API_KEY"
#: The deliberate opt-in. Only ``1`` opens the lane; it must never come from a
#: shell profile.
OPT_IN = "MLLM_ALLOW_UNPINNED_HOST_LANE"
#: Credentials that can never become this lane's, and that the recipe unsets
#: (``env -u``) for the run. The Makefile's list must equal this one.
VENDOR_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY")


def unpinned_host_refusal(env: Mapping[str, str]) -> Optional[List[str]]:
    """Why the unpinned host lane must not start under ``env``; ``None`` when it may."""
    if not env.get(LANE_KEY):
        return [
            f"make: this target IS the credentialed host lane, so it needs {LANE_KEY}",
            "      (a DEDICATED key carrying a key-level spend limit). Without one every row",
            "      skips, the lane covers nothing, and an @latest image proves nothing —",
            "      refusing here rather than building it first.",
            "      Credential-free instead: make e2e-container-host-discovery  (host lane, free tier)",
            "                               make e2e-container-latest          (provider lane @latest)",
        ]
    if env.get(OPT_IN) != "1":
        lines = [
            "make: refusing to run the host lane against an UNPINNED image with a credential present.",
            f"      found: {LANE_KEY}",
        ]
        lines += [f"      also exported: {name} (unset for the run below — no container is ever "
                  "given it)" for name in VENDOR_KEYS if env.get(name)]
        lines += [
            "      This target installs every CLI @latest — unreviewed, unhashed binaries — and would",
            "      hand them a live key plus outbound network. If that is what you want, set",
            f"      {OPT_IN}=1 deliberately; it must never come from a shell profile.",
        ]
        return lines
    return None


def rows_ran_refusal(records: Path, since: Path) -> Optional[List[str]]:
    """Why the run covered nothing: no ``*.json`` under ``records`` (at any
    depth) modified after ``since``. ``None`` when at least one was."""
    try:
        floor = Path(since).stat().st_mtime_ns
    except OSError as exc:
        return [f"make: NOT-RUN — the stamp {since} cannot be read ({exc}), so no record "
                "can be shown newer than it"]
    records = Path(records)
    fresh = [path for path in (records.rglob("*.json") if records.is_dir() else [])
             if path.is_file() and path.stat().st_mtime_ns > floor]
    if fresh:
        return None
    return [
        "make: NOT-RUN — pytest exited green but not one host-lane row executed:",
        f"      no {records}/<utc>-<row>.json was written by this invocation,",
        "      and a row that ran always writes one. A lane that covered nothing is a",
        "      distinct outcome from a pass, and it is reported as one.",
        f"      The named skip reasons are in the run above (-rs) and in {records.parent}/runs/.",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lane_guard.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("unpinned-host", help="refuse the unpinned host lane without its key and opt-in")
    ran = sub.add_parser("rows-ran", help="refuse a green run in which no row wrote a record")
    ran.add_argument("--records", type=Path, required=True)
    ran.add_argument("--since", type=Path, required=True, help="a file made before pytest ran")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "unpinned-host":
        refusal = unpinned_host_refusal(os.environ)
    else:
        refusal = rows_ran_refusal(args.records, args.since)
    if refusal:
        print("\n".join(refusal), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
