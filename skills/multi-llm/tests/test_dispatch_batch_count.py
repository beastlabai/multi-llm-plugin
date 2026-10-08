"""The ``Organized into N execution batches`` count and the batches the log shows.

Fix round 28 (live run 12, the stop fixture on OpenCode and Codex): the first
run printed "Organized into 3 execution batches: 5 task(s) to dispatch" and
then batch lines for batches 0, 1, 2 and 3. Batch 3 was T005, refused before
any runner started: the count rightly left it out (nothing was dispatched for
it), but its line still said ``batch 3:``, so the log read as four batches.
The resumes that dispatched nothing said "Organized into 0 execution batches"
beside the same ``batch 3:`` line. A task no runner started (refused before
dispatch, a failed check not run again, a batch kept at the user's request)
now names no batch, so the batches shown are the ones counted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_check_gate_fixture import _first_run, repo  # noqa: E402,F401

BATCH_LINE = re.compile(r"^\[write-subagents\] batch (\d+): (T\d{3}) ", re.M)


def test_the_count_matches_the_batches_the_log_shows(repo, capsys):
    text = _first_run(repo, capsys)
    count = re.search(r"Organized into (\d+) execution batch(?:es)?:", text)
    assert count, text
    shown = {int(index) for index, _task in BATCH_LINE.findall(text)}
    assert len(shown) == int(count.group(1)) == 3, (sorted(shown), text)
    # The refused task is named, with no batch of its own.
    assert "[write-subagents] T005 refused before dispatch" in text
    assert not any(task == "T005" for _index, task in BATCH_LINE.findall(text))


def test_a_resume_that_dispatches_nothing_shows_no_batch(repo, capsys):
    import multi_llm

    _first_run(repo, capsys)
    assert multi_llm.main([
        "implement", str(repo.plan), "--host", "codex", "--yes",
        "--exec-subagents", "--resume",
    ]) == 3
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "Organized into 0 execution batches" in text
    assert not BATCH_LINE.findall(text), text
