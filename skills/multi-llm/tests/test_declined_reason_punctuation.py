"""A declined item's reason is recorded as the batch wrote it and printed
with one full stop (the decline fixture, 2026-10-07; live run 11 #10).

The live-run decline fixture's Claude Code leg recorded
`util batch-done --declined "<id>=… raise MAX_SPEED."` and the
`[BATCH_DECLINED]` line read "… — … raise MAX_SPEED.. List it in the
phase's summary.md": the line adds its own stop after the reason. Fix round
21 dropped one closing stop or semicolon from every reason as both legs read
it, so the records lost it too: live run 11's `processed_apply-fixes` read
"… needs a plan decision" while the reply, the log and the host's summary.md
read "… needs a plan decision." (fix round 26, #10). The record now keeps the
batch's words, closing stop included (whitespace collapsed, bounded), and
only a line that puts its own punctuation right after a reason drops the
reason's one there (`text_clip.without_closing_mark`), so the marker line,
the executor's stop text "It declined fix 1 (`id`) — <reason>. ", the notes
and `status` still read one full stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from tests.test_batch_done import (  # noqa: E402,F401
    _materialize, _run, _state, _suggestion_batches, plan_dir,
)
from utils import batch_done  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_batch_scheduler as scheduler  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.text_clip import without_closing_mark  # noqa: E402

REASON = "the plan says never to raise MAX_SPEED"


@pytest.mark.parametrize("given, recorded", [
    (REASON + ".", REASON + "."), (REASON + ". ", REASON + "."),
    (REASON + ";", REASON + ";"), ("  " + REASON + " .", REASON + " ."),
    (REASON, REASON), ("a  reason\nover   lines.", "a reason over lines."),
])
def test_the_executor_records_a_reason_as_written(given, recorded):
    read = executor.declined_items(
        {"ids": ["b"], "declined": [{"id": "a", "reason": given}]}, ["a", "b"],
    )
    assert read.items == {"a": recorded}
    assert read.problems == []


@pytest.mark.parametrize("given, shown", [
    (REASON + ".", REASON), (REASON + ";", REASON), (REASON + " .", REASON),
    (REASON, REASON), ("...", ".."), (".", "."), (" . ", "."), (";", ";"),
    (REASON + "..", REASON + "."),
])
def test_a_line_drops_one_closing_mark_and_only_when_text_remains(given, shown):
    """Review of fix round 20, #2: stripping every closing mark read a
    reason of "..." or "." as no reason; at most one goes, at render."""
    assert without_closing_mark(given) == shown


@pytest.mark.parametrize("given", ["...", ".", ";"])
def test_a_punctuation_only_reason_is_still_a_reason(given):
    read = executor.declined_items(
        {"ids": ["b"], "declined": [{"id": "a", "reason": given}]}, ["a", "b"],
    )
    assert read.items == {"a": given}
    assert read.problems == []


@pytest.mark.parametrize("given", ["", "   ", None])
def test_no_text_is_still_no_reason(given):
    read = executor.declined_items(
        {"ids": ["b"], "declined": [{"id": "a", "reason": given}]}, ["a", "b"],
    )
    assert read.items == {}
    assert read.problems == ["`declined` gives no reason for a"]


def test_the_native_leg_records_a_punctuation_only_reason(plan_dir, capsys):
    plan, out = plan_dir
    StateManager(plan).save()
    phase_dir, _tasks = _materialize(
        plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
    code, stdout, err, _result = _run([
        "util", "batch-done", "--phase-dir", str(phase_dir),
        "--task-id", "apply-batch-1", "--status", "completed",
        "--declined", "g-rate=...",
    ], capsys)
    assert code == 0, err
    assert "a decline without its reason" not in err
    assert f"{batch_done.BATCH_DECLINED_MARKER} apply-batch-1 g-rate:" in stdout
    processed = _state(out)["processed_apply-suggestions"]
    assert processed["g-rate"]["details"]["reason"] == "..."


def test_a_row_recorded_with_its_stop_reads_with_it():
    assert executor.row_declined(
        {"declined": [{"id": "a", "reason": REASON + "."}]}
    ) == {"a": REASON + "."}
    # A record from before fix round 26 has none and reads as it is.
    assert executor.row_declined(
        {"declined": [{"id": "a", "reason": REASON}]}
    ) == {"a": REASON}


def test_the_native_marker_line_has_one_full_stop_and_the_record_keeps_it(
    plan_dir, capsys,
):
    plan, out = plan_dir
    StateManager(plan).save()
    phase_dir, _tasks = _materialize(
        plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
    code, stdout, err, _result = _run([
        "util", "batch-done", "--phase-dir", str(phase_dir),
        "--task-id", "apply-batch-1", "--status", "completed",
        "--declined", f"g-rate={REASON}.",
    ], capsys)
    assert code == 0, err
    assert (f"{batch_done.BATCH_DECLINED_MARKER} apply-batch-1 g-rate: not "
            f"applied, the batch declined it — {REASON}. List it") in stdout
    assert ".." not in stdout
    processed = _state(out)["processed_apply-suggestions"]
    assert processed["g-rate"]["details"]["reason"] == REASON + "."


def test_the_executors_declined_words_have_one_full_stop():
    words = scheduler.declined_words(
        "apply_code_fixes", {"c8": ("apply-fix-batch-0", 1)}, {"c8": REASON + "."})
    assert words == f"fix 1 (`c8`) — {REASON}"
