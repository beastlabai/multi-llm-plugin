"""A prompt shows each prior-changes entry once (the decline fixture, 2026-10-07).

An apply-code-fixes batch's first prompt quoted the implement phase's three
entries under `## Implementation record` (the code-fix batcher's own section,
with its "do not silently revert" rule) and then again as "Batch 1-3" under
`## Changes Applied in Prior Batches`, because a phase's first dispatch has no
entries of its own and the prior-changes block falls back to every phase's.
The binding (`bind_prior_changes`, both legs) now leaves out what the prompt
already quotes word for word and counts it on one line. Fix round 22 (live
run 9 #7): an apply-code-fixes batch's block lists only the earlier batches of
its own run, so implement's entries are shown in the record alone, and the
record names the ones past its five-entry budget by task id
(`test_code_fix_prior_batches.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils import prior_changes  # noqa: E402
from utils.code_fix_batcher import (  # noqa: E402
    MAX_IMPLEMENTATION_RECORD_ENTRIES, CodeFixBatch, format_fix_batch_prompt,
)
from utils.write_batch_scheduler import bind_prior_changes  # noqa: E402

HEADING = "## Changes Applied in Prior Batches"


def _log(tmp_path: Path, entries):
    for entry_id, phase, summary in entries:
        prior_changes.append(str(tmp_path), entry_id, phase, summary)


def _block(prompt: str) -> str:
    """The prior-changes section of a bound prompt."""
    tail = prompt.split(HEADING, 1)[1]
    return tail.split("\n---", 1)[0].split("\n## ", 1)[0]


def _fix_prompt(tmp_path: Path) -> str:
    batch = CodeFixBatch(
        fixes=[{"title": "Clamp opponent speed", "file": "src/opponent.js",
                "description": "d", "type": "bug", "importance": "MEDIUM"}],
        file_key="src/opponent.js", batch_type="bug",
    )
    return format_fix_batch_prompt(
        batch, "plans/kart-sim.md", output_dir=str(tmp_path)
    )


def test_the_first_fix_batch_shows_the_implement_entries_once(tmp_path):
    _log(tmp_path, [
        ("T001", "implement", "Created src/opponent.js; speed deliberately NOT clamped."),
        ("T002", "implement", "Created src/race.js with formatTime."),
    ])
    prompt = _fix_prompt(tmp_path)
    assert "## Implementation record" in prompt
    bound = bind_prior_changes(
        {"task_id": "apply-fix-batch-0", "prompt": prompt},
        output_dir=str(tmp_path), phase="apply-fixes",
    )
    assert bound.count("speed deliberately NOT clamped") == 1
    assert bound.count("Created src/race.js with formatTime") == 1
    block = _block(bound)
    assert "**Batch" not in block
    # Fix round 22: the first batch of the run has nothing of its own to list.
    assert prior_changes.DEFAULT_TEXT in block, block


def test_an_entry_the_record_does_not_quote_is_named_by_its_task_id(tmp_path):
    entries = [
        (f"T{n:03d}", "implement", f"Task {n} edited src/other{n}.js.")
        for n in range(1, MAX_IMPLEMENTATION_RECORD_ENTRIES + 3)
    ] + [("apply-batch-0#abc", "apply-suggestions", "Plan: clarified the lap rule.")]
    _log(tmp_path, entries)
    prompt = _fix_prompt(tmp_path)
    bound = bind_prior_changes(
        {"task_id": "apply-fix-batch-0", "prompt": prompt},
        output_dir=str(tmp_path), phase="apply-fixes",
    )
    shown = entries[:MAX_IMPLEMENTATION_RECORD_ENTRIES]
    for _id, _phase, summary in shown:
        assert bound.count(summary) == 1, summary
    left_out = [entry_id for entry_id, _phase, _summary in entries[
        MAX_IMPLEMENTATION_RECORD_ENTRIES:-1]]
    assert f"- and 2 more tasks: {', '.join(left_out)}" in bound
    for _id, _phase, summary in entries[MAX_IMPLEMENTATION_RECORD_ENTRIES:]:
        assert summary not in bound, summary
    assert prior_changes.DEFAULT_TEXT in _block(bound)


def test_a_prompt_that_quotes_nothing_renders_as_before(tmp_path):
    _log(tmp_path, [("T001", "implement", "Created src/a.js.")])
    assert prior_changes.quoted_in("no record here", str(tmp_path)) == []
    assert prior_changes.render(str(tmp_path), phase="apply-fixes", quoted=[]) == (
        prior_changes.render(str(tmp_path), phase="apply-fixes")
    ) == "- **T001**: Created src/a.js."


def test_quoted_in_reads_the_record_line_shape(tmp_path):
    _log(tmp_path, [("T001", "implement", "Created src/a.js."),
                    ("T002", "implement", "Created src/b.js.")])
    text = "## Implementation record\n\n- (T002): Created src/b.js.\n"
    assert prior_changes.quoted_in(text, str(tmp_path)) == [("T002", "implement")]
    assert prior_changes.render(
        str(tmp_path), phase="apply-fixes", quoted=[("T002", "implement")]
    ) == (
        "- **T001**: Created src/a.js.\n"
        "... and 1 earlier entry already quoted in this prompt"
    )
