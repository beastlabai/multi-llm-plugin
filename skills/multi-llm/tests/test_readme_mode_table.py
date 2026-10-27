#!/usr/bin/env python3
"""Every user-facing mode has a row in the README's ``## 3. Modes`` table.

The mode table is how a user finds out a mode exists, and it is hand-written.
``modes.MODES`` is the one owner of the mode words (``argument-hint``, the
dispatcher, ``--help`` and the host shims all derive from it), but nothing tied
the README to it: ``--init`` was ``user_facing`` and had no row, and a new mode
could ship with a green suite and no front-page entry
(plans/review-commits.md §2.11, "A guard test"). This module ties the two, in
one direction: each ``user_facing`` row must appear as a ``--<name>`` row of
that table. The table may say more than the table of modes does (it is prose),
so a row with no mode behind it is left to review rather than refused here.

The README is located the way ``tests/test_docs_links.py`` locates it (three
levels up from this file), and the table is parsed rather than copied, so the
guard follows the README it checks. Only the first table under the ``Modes``
heading counts, outside code fences: the subsections below it carry tables of
their own (``Review and apply in one command``, ``Options``) that name the same
flags for other reasons.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import modes as mode_table  # noqa: E402

#: Repo root: tests/ -> multi-llm/ -> skills/ -> <repo>.
REPO_ROOT = Path(__file__).resolve().parents[3]
README = REPO_ROOT / "README.md"

#: An ATX heading; the README numbers its sections (``## 3. Modes``).
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+")

#: A fence opener or closer, indented or not.
FENCE_RE = re.compile(r"^\s*(```|~~~)")

#: The flag a mode-table row documents: its first cell is exactly `--<name>`.
ROW_FLAG_RE = re.compile(r"^`(--[a-z][a-z0-9-]*)`$")

SECTION_TITLE = "Modes"


def _outside_fences(text: str):
    """Yield ``(line_number, line)`` for every line not inside a code fence."""
    in_fence = False
    for number, line in enumerate(text.splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield number, line


def mode_table_rows(text: str) -> List[Tuple[int, str]]:
    """The ``(line_number, flag)`` rows of the first table under ``Modes``.

    The table ends at its first non-table line once it has started, or at the
    next heading. A header row and the ``| --- |`` separator are skipped; a
    body row whose first cell is not a single backticked ``--flag`` is not a
    mode row and is left out.
    """
    rows: List[Tuple[int, str]] = []
    in_section = in_table = in_body = False
    for number, line in _outside_fences(text):
        heading = HEADING_RE.match(line)
        if heading:
            if in_section:
                break
            title = SECTION_NUMBER_RE.sub("", heading.group(2).strip())
            in_section = title == SECTION_TITLE
            continue
        if not in_section:
            continue
        if not line.lstrip().startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        cell = line.strip().strip("|").split("|")[0].strip()
        if cell and set(cell) <= set("- :"):
            in_body = True
            continue
        if not in_body:
            continue
        match = ROW_FLAG_RE.match(cell)
        if match:
            rows.append((number, match.group(1)))
    return rows


def missing_mode_flags(text: str) -> List[str]:
    """The ``--<name>`` of every user-facing mode with no mode-table row."""
    listed = {flag for _, flag in mode_table_rows(text)}
    return [
        f"--{mode.name}"
        for mode in mode_table.user_facing_modes()
        if f"--{mode.name}" not in listed
    ]


def test_every_user_facing_mode_has_a_row_in_the_readme_mode_table():
    missing = missing_mode_flags(README.read_text(encoding="utf-8"))
    assert not missing, (
        f"user-facing modes with no row in README.md's '## 3. Modes' table: "
        f"{', '.join(missing)}. Every mode in modes.MODES with user_facing=True "
        f"needs a `| --<name> | ... |` row there, so a mode never ships "
        f"undocumented."
    )


def test_the_guard_reads_the_readme_mode_table_it_means():
    """Non-vacuity: a parser that found no table would pass the guard above.

    The table must be found, hold one row per flag, and include rows that
    exist only because of this guard (``--init``) and rows the subsections'
    own tables also name (``--improve-code``), so the parse is the mode
    table's and not one of theirs.
    """
    rows = mode_table_rows(README.read_text(encoding="utf-8"))
    flags = [flag for _, flag in rows]
    assert len(flags) >= len(mode_table.user_facing_modes()), flags
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    assert not duplicates, f"flags with more than one mode-table row: {duplicates}"
    assert "--init" in flags and "--review-plan" in flags, flags


def test_a_missing_row_is_named_and_a_table_elsewhere_does_not_count():
    """The failure names the missing flag, and only the Modes table counts.

    A synthetic README lists every user-facing mode but one in its mode
    table, and the one left out only in a later table and in a fenced
    sample: the guard must still name it.
    """
    flags = [f"--{mode.name}" for mode in mode_table.user_facing_modes()]
    left_out = flags[-1]
    body = "\n".join(f"| `{flag}` | does a thing |" for flag in flags[:-1])
    text = (
        "# Title\n\n## 2. Plans and outputs\n\n| Path | Written by |\n| --- | --- |\n"
        f"| `x` | `{left_out}` |\n\n"
        "## 3. Modes\n\nInvoke it.\n\n| Flag | What it does |\n| --- | --- |\n"
        f"{body}\n\nAfter the table.\n\n"
        f"```text\n| `{left_out}` | in a fence |\n```\n\n"
        f"| `{left_out}` | a second table in the same section |\n\n"
        f"### 3.1 More\n\n| Flag | Effect |\n| --- | --- |\n| `{left_out}` | x |\n"
    )
    assert missing_mode_flags(text) == [left_out]

    complete = text.replace(body, body + f"\n| `{left_out}` | does a thing |")
    assert missing_mode_flags(complete) == []


if __name__ == "__main__":  # pragma: no cover - parity with the sibling suites
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
