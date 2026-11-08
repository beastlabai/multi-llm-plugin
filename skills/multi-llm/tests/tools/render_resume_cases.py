#!/usr/bin/env python3
"""Render the case list of ``references/resume-detection.md`` from ``utils/resume_state.py``.

``util resume-state`` and the reference describe one table, ``CASES`` and
``ACTIONS``; this script writes the reference's block between its two markers
from them, so the prose a host reads and the answer the command prints cannot
say different things. ``tests/test_resume_state.py`` runs ``--check``.

Usage (from the repository root)::

    python skills/multi-llm/tests/tools/render_resume_cases.py           # rewrite the block
    python skills/multi-llm/tests/tools/render_resume_cases.py --check   # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SKILL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL_ROOT))

from utils import resume_state  # noqa: E402

REFERENCE = SKILL_ROOT / "references" / "resume-detection.md"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the reference's block differs from a fresh render")
    args = parser.parse_args(argv)
    current = REFERENCE.read_text(encoding="utf-8")
    try:
        fresh = resume_state.splice_cases(current)
    except ValueError:
        print(f"{REFERENCE} has lost its generated-block markers "
              f"({resume_state.DOC_BEGIN} … {resume_state.DOC_END}).", file=sys.stderr)
        return 1
    if args.check:
        if fresh != current:
            print(f"{REFERENCE} is stale: run "
                  "`python skills/multi-llm/tests/tools/render_resume_cases.py`.",
                  file=sys.stderr)
            return 1
        return 0
    if fresh != current:
        REFERENCE.write_text(fresh, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
