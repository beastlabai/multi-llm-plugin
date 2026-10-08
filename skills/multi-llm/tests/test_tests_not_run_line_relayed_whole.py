#!/usr/bin/env python3
"""apply-code-fixes' runbook hands the report the `[TESTS_NOT_RUN]` line
first, whole (fix round 28).

Live run 12 (2026-10-08), Claude Code racing 08: `--mark-completed` printed
"[TESTS_NOT_RUN] The batches ran the repository's known checks after their
edits: `npm test`: passed in apply-fix-batch-0. The plugin itself ran no tests:
…", and the report said only "The plugin ran no tests itself because
`driver.verify_command` isn't set. To have every code phase run them, add …":
the lead saying what the batches ran and the package.json clause were gone.
Step 7 asked the host, first, for "one plain line" of its own with the same
content, and named the printed line only after it. It now names the printed
line first, asks for its whole text word for word, and keeps the host's own
line for a run that printed none.
"""

from __future__ import annotations

from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent


def _step_7_paragraph() -> str:
    text = (SKILL_ROOT / "instructions" / "apply-code-fixes.md").read_text(encoding="utf-8")
    start = text.index("If no `verify_command` is configured, say so in the summary")
    end = text.index("\n\n", start)
    return " ".join(text[start:end].split())


def test_the_printed_line_comes_first_and_whole():
    paragraph = _step_7_paragraph()
    assert paragraph.index("[TESTS_NOT_RUN]") < paragraph.index("one plain line")
    assert ("Give the whole text after its tag, word for word, as plain text: never "
            "a shorter line of your own in its place or beside it.") in paragraph


def test_the_hosts_own_line_is_only_for_a_run_that_printed_none():
    paragraph = _step_7_paragraph()
    assert ("Only when step 10 printed no such line, write one plain line "
            "yourself") in paragraph
    assert "Say it in your report too, in one plain line" not in paragraph
