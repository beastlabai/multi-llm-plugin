"""A declined item's reason is printed with one full stop (the decline fixture, 2026-10-07).

The live-run decline fixture's Claude Code leg recorded
`util batch-done --declined "<id>=… raise MAX_SPEED."` and the
`[BATCH_DECLINED]` line read "… — … raise MAX_SPEED.. List it in the
phase's summary.md": the line adds its own stop after the reason. Every
surface that quotes a reason puts punctuation after it (the marker line, the
executor's stop text "It declined fix 1 (`id`) — <reason>. ", the notice), so
both legs read a reason once, through `subagent_executor.declined_reason`,
which drops a closing stop or semicolon.
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
from utils.state_manager import StateManager  # noqa: E402

REASON = "the plan says never to raise MAX_SPEED"


@pytest.mark.parametrize("given", [
    REASON + ".", REASON + ". ", REASON + ";", "  " + REASON + " .", REASON,
])
def test_the_executor_reads_a_reason_without_its_closing_stop(given):
    read = executor.declined_items(
        {"ids": ["b"], "declined": [{"id": "a", "reason": given}]}, ["a", "b"],
    )
    assert read.items == {"a": REASON}
    assert read.problems == []


@pytest.mark.parametrize("given, kept", [
    ("...", ".."), (".", "."), (" . ", "."), (";", ";"), (REASON + "..", REASON + "."),
])
def test_one_closing_mark_goes_and_only_when_text_remains(given, kept):
    """Review of fix round 20, #2: stripping every closing mark read a
    reason of "..." or "." as no reason, and the decline as a partial
    application (exit 3) where round 18/19 recorded it declined."""
    read = executor.declined_items(
        {"ids": ["b"], "declined": [{"id": "a", "reason": given}]}, ["a", "b"],
    )
    assert read.items == {"a": kept}
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
    assert processed["g-rate"]["details"]["reason"] == ".."


def test_a_row_recorded_with_its_stop_reads_without_it():
    assert executor.row_declined(
        {"declined": [{"id": "a", "reason": REASON + "."}]}
    ) == {"a": REASON}


def test_the_native_marker_line_has_one_full_stop(plan_dir, capsys):
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
    assert processed["g-rate"]["details"]["reason"] == REASON
