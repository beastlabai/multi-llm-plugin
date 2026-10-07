"""What an accepted check's re-run may hide, and where it is on record.

Fix round 22 took out of a released task's own-check findings those that only
repeat what the user accepted for a check (`split_accepted_findings`, live
run 9, OpenCode accept 02). The review of that round (fix round 23, records
#1-#3 and #6) found it hid more than that:

* a count after `exited with` / `returned` was read as an exit code and
  dropped, so "exited with 9 failures in tests/test_auth.py" matched an
  accepted "exited with 3 failures in tests/test_auth.py";
* it ran on every implement task's own check, also a task that does not
  depend on the accepted check (its prompt never said the problems were
  accepted) and a task whose own job is to fix them;
* with two accepted findings, a finding straddling both matched the two read
  one after the other, and any slice of three words from the middle of one
  matched it;
* the split findings lived only in the result row and the printed line:
  `state.json` and `status` showed the task as plainly `completed`.

Now numbers are words (only a whole exit clause is set aside), the split
runs only for a task whose prompt carried the accepted-checks section and
whose own text does not name the check's command or ask to fix its findings,
a finding matches one accepted finding (all of it, or a run of its words from
its start), and the split is on record in `state.json`'s
`task_batch_check_accepted` on both legs, which `status` names.
"""

from __future__ import annotations

import json

import pytest

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: F401 - `repo` is a fixture
    ALLOW,
    CHECK_GATE,
    DENY,
    NOTE,
    PLAN,
    _policy,
    _statuses,
)
from tests.test_implement_stop_account import _implement, make_repo  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.write_batch_scheduler import (  # noqa: E402
    BATCH_CHECK_ACCEPTED_MARKER,
    BATCH_CHECK_MARKER,
    split_accepted_findings,
)

#: T002's finding as OpenCode's check recorded it in run 9.
ACCEPTED = "node scripts/check.mjs: retries must be 3 (found 2)"
#: What a task's own check reported when it ran the same check again.
RERUN = "node scripts/check.mjs: check failed: retries must be 3 (found 2) (exit 1)"
#: The Claude Code host's wording of the same T002 finding (run 9).
NATIVE = (
    "node scripts/check.mjs exited 1 and printed 'check failed: retries must be "
    "3 (found 2)'. src/config.json has retries=2 (set by T001), but the check "
    "requires 3."
)


# -- #1: numbers are words ----------------------------------------------------


class TestNumbersAreWords:
    """Only a whole exit clause is set aside: never a count."""

    @pytest.mark.parametrize("accepted, rerun", [
        # The reviewer's scenario.
        ("pytest tests/: exited with 3 failures in tests/test_auth.py",
         "pytest tests/: exited with 9 failures in tests/test_auth.py"),
        ("npm test: returned 1 failure in auth.test.js",
         "npm test: returned 4 failures in auth.test.js"),
        ("pytest tests/: exited 1 with 3 failures in tests/test_auth.py",
         "pytest tests/: exited 1 with 9 failures in tests/test_auth.py"),
        ("npm test exited with 2 failing tests in auth.test.js",
         "npm test exited with 5 failing tests in auth.test.js"),
    ])
    def test_another_count_is_never_the_accepted_finding(self, accepted, rerun):
        own, matched = split_accepted_findings([rerun], {"T002": [accepted]})
        assert own == [rerun] and matched == {}

    @pytest.mark.parametrize("accepted, rerun", [
        (ACCEPTED, RERUN),
        (ACCEPTED, "node scripts/check.mjs exited 1: retries must be 3 (found 2)"),
        (ACCEPTED, "node scripts/check.mjs: retries must be 3 (found 2), exit code 1"),
        (ACCEPTED, "node scripts/check.mjs: retries must be 3 (found 2); exited with status 2"),
        (ACCEPTED, "node scripts/check.mjs: retries must be 3 (found 2) and exited 1"),
        (ACCEPTED, "node scripts/check.mjs returned 1: retries must be 3 (found 2)"),
        (NATIVE, RERUN),
        ("pytest tests/: exited with 3 failures in tests/test_auth.py",
         "pytest tests/: exited with 3 failures in tests/test_auth.py (exit code 1)"),
    ])
    def test_a_whole_exit_clause_is_still_set_aside(self, accepted, rerun):
        own, matched = split_accepted_findings([rerun], {"T002": [accepted]})
        assert own == [] and matched == {"T002": [rerun]}
