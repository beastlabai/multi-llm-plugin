"""A batch killed at its time limit after its edits landed says so plainly.

OpenCode racing run 6 (fix round 16): T006 ran past the 20-minute per-task
limit after its edit to ``index.html`` had landed, and was recorded a partial
application — correctly, and it was not retried. But the stop's one line, the
line the tool routes quote, read "partial application (bytes moved, the batch
did not finish) … To go on: inspect the half-applied state, restore or keep
it, fix the cause, then resume this phase": it named no time limit, did not say
the edits were kept, and "fix the cause" named no cause. The resume run then
found the work already there.
"""

from __future__ import annotations

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from tests.test_write_executor import _patch_edit, env  # noqa: E402,F401
from utils import subagent_executor as executor  # noqa: E402


def _edits_then_times_out(env, monkeypatch) -> None:
    """The runner makes its edit, then is killed at the time limit."""
    env.edits([_patch_edit(env.plan)], prose=True)
    real = executor._spawn

    def spawn(**kwargs):
        real(**kwargs)
        return executor._Capture(timed_out=True, argv=["codex"], duration=0.01)

    monkeypatch.setattr(executor, "_spawn", spawn)


class TestATimedOutPartialApplication:

    def test_the_stop_line_says_the_edits_are_kept_and_what_to_do(
        self, env, monkeypatch
    ):
        _edits_then_times_out(env, monkeypatch)
        result = env.scheduler([env.apply_task(0)]).run()

        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_PARTIAL, outcome.reason
        assert result.exit_code == 3
        head = result.stop["headline"]
        assert "ran past its time limit" in head, head
        assert "kept in place" in head, head
        assert "resume this phase" in head and "dispatched again on top of them" in head
        assert "driver.subagent_timeout_s" in head, head
        assert "fix the cause" not in head, head

    def test_the_stop_block_says_it_too(self, env, monkeypatch):
        _edits_then_times_out(env, monkeypatch)
        result = env.scheduler([env.apply_task(0)]).run()

        text = "\n".join(result.messages)
        # The existing words stay, and the time limit joins them.
        assert "partial application (bytes moved, the batch did not finish)" in text
        assert "its runner ran past its time limit and was stopped" in text, text
        assert "its edits are kept in place: resume this phase" in text, text
        assert "util restore-batch" in text
        assert "inspect the half-applied state" not in text, text

    def test_a_partial_that_did_not_time_out_reads_as_before(
        self, env, monkeypatch
    ):
        """A runner that crashed after editing: no time limit is named."""
        env.edits([_patch_edit(env.plan)], prose=True)
        real = executor._spawn

        def spawn(**kwargs):
            real(**kwargs)
            return executor._Capture(exit_code=1, stderr="boom", argv=["codex"],
                                     duration=0.01)

        monkeypatch.setattr(executor, "_spawn", spawn)
        result = env.scheduler([env.apply_task(0)]).run()

        assert result.outcomes[0].write_outcome == executor.WRITE_PARTIAL
        head = result.stop["headline"]
        assert "time limit" not in head, head
        assert "inspect the half-applied state" in head, head
