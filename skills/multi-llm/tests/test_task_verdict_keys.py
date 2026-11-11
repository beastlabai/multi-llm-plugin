"""Every per-task key of ``state.json`` is either part of a task's verdict or
explicitly not one.

A run that hands an implement task out again (a ``--task`` rerun, a
start-over) clears the task's verdict through
``state_manager.clear_task_verdict``, which walks
``state_manager.TASK_VERDICT_KEYS``. A per-task key added without a place in
that list would keep an earlier run's verdict on the rerun task, and nothing
else fails: these tests make the omission a failure. Put a new key either in
``TASK_VERDICT_KEYS`` or in one of the not-a-verdict tables below, with the
reason.
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import state_manager  # noqa: E402
from utils.state_manager import (  # noqa: E402
    TASK_VERDICT_KEYS,
    clear_task_verdict,
    set_task_batch_check_accepted_in_state,
    set_task_batch_check_in_state,
    set_task_batch_check_repeats_in_state,
    set_task_check_findings_in_state,
    set_task_check_tree_in_state,
    set_task_own_check_note_in_state,
    set_task_own_check_ran_in_state,
    set_task_own_text_in_state,
)

#: Module constants naming a task-related key that is not a verdict, by
#: constant name, with why a rerun leaves it alone.
NOT_A_VERDICT_CONSTANTS = {
    # Written by the rerun itself (`mark_tasks_reset_in_state`), never cleared.
    "TASK_RESET_AT_KEY": "a rerun's own stamp",
    # Retired by the task's next status write (`_consume_answer`).
    "HUMAN_TASK_DECISIONS_KEY": "the user's answer, consumed by a status write",
    # The decomposition's graph, one unit for the whole plan.
    "IMPLEMENT_TASK_GRAPH_KEY": "the decomposition, not a task's record",
}

#: Per-task record keys (``state_manager._TASK_RECORD_MAPS`` and
#: ``_TASK_RECORD_LISTS``) that are not a verdict, by key, with why.
NOT_A_VERDICT_RECORD_KEYS = {
    "task_status": "rewritten by update_task_status on every record",
    "task_status_reasons": "goes with task_status",
    "implement_task_reset_at": "a rerun's own stamp",
    "human_task_decisions": "the user's answer, consumed by a status write",
    "dependency_overrides": "the user's skip-continue override",
    "apply_fix_batch_checks": "an apply-code-fixes batch's report, not implement's",
}

_TASK_CONSTANT = re.compile(r"_?[A-Z_]*TASK[A-Z_]*_KEY")


def _task_key_constants():
    return {
        name: value for name, value in vars(state_manager).items()
        if _TASK_CONSTANT.fullmatch(name) and isinstance(value, str)
    }


def test_every_task_key_constant_is_a_verdict_or_listed_as_not_one():
    constants = _task_key_constants()
    assert constants, "the scan found no TASK key constant"
    unplaced = sorted(
        name for name, value in constants.items()
        if value not in TASK_VERDICT_KEYS and name not in NOT_A_VERDICT_CONSTANTS
    )
    assert not unplaced, (
        f"{unplaced}: add each to state_manager.TASK_VERDICT_KEYS, or to "
        f"NOT_A_VERDICT_CONSTANTS here with the reason a rerun keeps it")
    both = sorted(
        name for name in NOT_A_VERDICT_CONSTANTS
        if constants.get(name) in TASK_VERDICT_KEYS
    )
    assert not both, f"{both} are listed as a verdict and as not one"
    stale = sorted(set(NOT_A_VERDICT_CONSTANTS) - set(constants))
    assert not stale, f"{stale} name no constant in state_manager"


def test_every_task_record_key_is_a_verdict_or_listed_as_not_one():
    record_keys = set(state_manager._TASK_RECORD_MAPS) | set(
        state_manager._TASK_RECORD_LISTS)
    unplaced = sorted(
        record_keys - set(TASK_VERDICT_KEYS) - set(NOT_A_VERDICT_RECORD_KEYS))
    assert not unplaced, (
        f"{unplaced}: add each to state_manager.TASK_VERDICT_KEYS, or to "
        f"NOT_A_VERDICT_RECORD_KEYS here with the reason a rerun keeps it")
    assert not set(TASK_VERDICT_KEYS) & set(NOT_A_VERDICT_RECORD_KEYS)
    # A verdict key is part of the task's record, so the save merges it with
    # the task (docs/development/write-scheduler.md, "Merge units").
    assert set(TASK_VERDICT_KEYS) <= record_keys
    assert len(set(TASK_VERDICT_KEYS)) == len(TASK_VERDICT_KEYS)


def _record_verdict(state, task_id):
    set_task_check_findings_in_state(state, task_id, ["check finding"])
    set_task_check_tree_in_state(state, task_id, "tree-1")
    set_task_batch_check_in_state(state, task_id, ["own check failed"])
    set_task_own_text_in_state(state, task_id, "Fix the release check.")
    set_task_batch_check_accepted_in_state(state, task_id, {"T009": ["accepted"]})
    set_task_batch_check_repeats_in_state(state, task_id, {"T010": ["repeat"]})
    set_task_own_check_ran_in_state(state, task_id, True)
    set_task_own_check_note_in_state(state, task_id, "ran npm test")


def test_clear_task_verdict_drops_every_verdict_key_of_one_task():
    state = {"task_status": {"T001": "completed", "T002": "completed"},
             "implement_task_reset_at": {"T001": "2026-11-08T00:00:00"}}
    _record_verdict(state, "T001")
    _record_verdict(state, "T002")
    assert set(TASK_VERDICT_KEYS) <= set(state), "a setter wrote nothing"
    other = {key: copy.deepcopy(state[key]) for key in TASK_VERDICT_KEYS}

    clear_task_verdict(state, "T001")

    for key in TASK_VERDICT_KEYS:
        recorded = state[key]
        assert "T001" not in recorded, key
        expected = other[key]
        if isinstance(expected, dict):
            expected = {k: v for k, v in expected.items() if k != "T001"}
        else:
            expected = [t for t in expected if t != "T001"]
        assert recorded == expected, key
    assert state["task_status"] == {"T001": "completed", "T002": "completed"}
    assert state["implement_task_reset_at"] == {"T001": "2026-11-08T00:00:00"}


def test_clear_task_verdict_drops_a_key_it_empties_and_matches_the_setters():
    state = {}
    _record_verdict(state, "T001")
    by_setters = copy.deepcopy(state)
    set_task_check_findings_in_state(by_setters, "T001", None)
    set_task_check_tree_in_state(by_setters, "T001", None)
    set_task_batch_check_in_state(by_setters, "T001", None)
    set_task_batch_check_accepted_in_state(by_setters, "T001", None)
    set_task_batch_check_repeats_in_state(by_setters, "T001", None)
    set_task_own_check_ran_in_state(by_setters, "T001", False)
    set_task_own_check_note_in_state(by_setters, "T001", None)

    clear_task_verdict(state, "T001")

    assert state == {} == by_setters


def test_clear_task_verdict_on_a_state_without_records_writes_nothing():
    state = {"task_status": {"T001": "pending"}}
    clear_task_verdict(state, "T001")
    assert state == {"task_status": {"T001": "pending"}}
