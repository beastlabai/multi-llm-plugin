"""A plan with every required phase on record and two manual implement tasks deferred.

``_complete_plan(root)`` builds it (with ``TITLES`` recorded unless
``titles=False``). No fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.state_manager import StateManager

from tests.support.next_phase_world import ALL_REQUIRED, CR_FINDING, S, _build


SKILL_ROOT = Path(__file__).parent.parent.parent


TITLES = {
    "T001": "Page shell",
    "T010": "Static validation of the inline module",
    "T011": "Manual playthrough over the static server",
    "T012": "Manual file:// load and blocked-CDN fallback check",
}
DEFERRED = {"T001": "completed", "T010": "completed",
            "T011": "skipped", "T012": "skipped"}


def _complete_plan(root: Path, *, titles=True, tasks=None) -> Path:
    """Every required phase on record, two implement tasks deferred."""
    here = os.getcwd()
    os.chdir(SKILL_ROOT)  # the first StateManager runs `git rev-parse HEAD`
    try:
        plan = _build(root, S(ALL_REQUIRED, tasks=tasks or DEFERRED, files=CR_FINDING))
    finally:
        os.chdir(here)
    if titles:
        state = StateManager(plan)
        state.record_task_titles(TITLES)
        state.save()
    return plan
