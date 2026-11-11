"""``task_ledger``: the rules both legs name a task's outcome by
(docs/development/status.md, "Status parity across execution paths").

The executor reads a skip chain from ``HumanTaskPlan.chain``, ``status``
from the ``skipped — depends on skipped task …`` reasons the same walk
recorded; both go through :func:`~utils.task_ledger.skip_bucket`, so one
scenario reads the same either way.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils import state_manager, task_ledger, write_batch_scheduler  # noqa: E402
from utils.task_ledger import (  # noqa: E402
    ANSWERED_SKIP_PREFIX,
    USER_DECLINED_REASON,
    chain_declined,
    skip_bucket,
    skip_chain_root,
)

DEPENDS = re.compile(r"^skipped — depends on skipped task (T[0-9]{3})\b")


def _depends(parent):
    return f"skipped — depends on skipped task {parent} (its title)"


#: A manual T001 deferred with T002 and T003 behind it, a T004 the user
#: declined with T005 behind it, and a T006 answered skip after failing.
REASONS = {
    "T001": "deferred — manual task",
    "T002": _depends("T001"),
    "T003": _depends("T002"),
    "T004": USER_DECLINED_REASON,
    "T005": _depends("T004"),
    "T006": f"{ANSWERED_SKIP_PREFIX}T006=skip after it failed",
}
EXPECTED = {
    "T001": ("deferred", None),
    "T002": ("deferred_dependents", "T001"),
    "T003": ("deferred_dependents", "T001"),
    "T004": ("declined", None),
    "T005": ("declined_dependents", None),
    "T006": ("declined", None),
}


def _parent_from_reason(task_id):
    match = DEPENDS.match(REASONS.get(task_id, ""))
    return match.group(1) if match else None


def test_the_state_reading_names_each_skip():
    for task_id, expected in EXPECTED.items():
        assert skip_bucket(
            task_id, REASONS[task_id],
            declined=lambda t=task_id: chain_declined(t, _parent_from_reason, REASONS.get),
            root=lambda t=task_id: skip_chain_root(t, _parent_from_reason),
        ) == expected


def test_the_executor_reading_names_each_skip_the_same_way():
    plan = write_batch_scheduler.HumanTaskPlan(
        skipped=dict(REASONS),
        chain={"T002": "T001", "T003": "T002", "T005": "T004"},
    )
    assert write_batch_scheduler._declined_split(plan) == {
        "declined_task_ids": [t for t, (b, _r) in EXPECTED.items() if b == "declined"],
        "declined_dependent_task_ids": [
            t for t, (b, _r) in EXPECTED.items() if b == "declined_dependents"],
    }
    assert write_batch_scheduler._deferred_split(plan) == {
        "deferred_waits_on": {
            t: root for t, (b, root) in EXPECTED.items() if b == "deferred_dependents"},
    }


@pytest.mark.parametrize("chain, root", [
    ({}, None),
    ({"T002": "T001"}, "T001"),
    ({"T002": "T001", "T001": "T002"}, "T002"),
])
def test_a_chain_ends_where_it_stops_or_closes(chain, root):
    assert skip_chain_root("T002", chain.get) == root


def test_a_cycle_is_not_a_decline():
    chain = {"T002": "T003", "T003": "T002"}
    assert chain_declined("T002", chain.get, lambda _t: "") is False


def test_the_state_manager_reads_the_ledgers_words():
    assert state_manager.StateManager._USER_DECLINED_REASON == USER_DECLINED_REASON
    assert state_manager.StateManager._ANSWERED_SKIP_PREFIX == ANSWERED_SKIP_PREFIX
    assert state_manager._ACCEPTED_REASON_PREFIX == task_ledger.ACCEPTED_REASON_PREFIX
    assert state_manager._COMPLETED_MANUAL_REASON == task_ledger.COMPLETED_MANUAL_REASON
    assert state_manager.StateManager._DEPENDS_ON_SKIPPED.pattern == DEPENDS.pattern


def test_the_tally_names_the_answered_tasks_as_the_kept_lists_do():
    labels = task_ledger.BUCKET_LABELS
    assert labels["accepted_findings"] == task_ledger.ACCEPTED_WITH_FINDINGS_LABEL
    assert labels["accepted_failed"] == task_ledger.ACCEPTED_AFTER_FAILING_LABEL
    assert labels["completed_manual"] == task_ledger.COMPLETED_BY_HAND_LABEL
