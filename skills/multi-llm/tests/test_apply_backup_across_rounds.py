"""The plan's `backup.md` across review rounds, through the real apply phase.

`apply-suggestions` snapshots the plan as `{phase}/backup.md` before its first
batch, and keeps an existing one only when the phase has recorded progress.
Round one's processed records, `write_batch_report.json` and batch results
outlive a review re-run in the same phase dir, and they used to count: round
two printed `Backup kept`, `backup.md` kept the plan from before round one, and
the plan as round two found it (round one's edits included) was saved nowhere.
Each signal is now dated against the review's latest completion, and a
re-stamp keeps the earlier backup aside as `backup.<sha8>.md`, as `--fresh`
does. A round two interrupted mid-loop is still a resume of round two.

The executor leg over the mock runner, the standalone `--apply-suggestions`
after a review that completed again (`_ApplyRig._review`); `full` and the
composites reach the same phase code.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_write_executor_orchestrators import (  # noqa: E402,F401
    APPLY_PLAN,
    _ApplyRig,
    repo,
)

#: Where `keep_backup_aside` moves round one's backup: the sha256 of the
#: bytes it holds, the plan from before round one.
ROUND_ONE_ASIDE = f"backup.{hashlib.sha256(APPLY_PLAN.encode()).hexdigest()[:8]}.md"


class TestTheBackupAcrossReviewRounds(_ApplyRig):

    def _round_one(self, repo):
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, sum(ids, []), content="round one\n") == 0
        assert (self._phase_dir(plan) / "backup.md").read_text(
            encoding="utf-8") == APPLY_PLAN
        assert "round one" in plan.read_text(encoding="utf-8")
        return plan

    def test_round_two_backs_up_the_plan_as_round_two_found_it(self, repo, capsys):
        self._round_one(repo)
        plan = self._review(repo, "two")
        ids = self._ids(plan)
        found = plan.read_bytes()
        capsys.readouterr()

        assert self._run(repo, plan, sum(ids, []), content="round two\n") == 0

        phase_dir = self._phase_dir(plan)
        assert (phase_dir / "backup.md").read_bytes() == found
        assert (phase_dir / ROUND_ONE_ASIDE).read_text(encoding="utf-8") == APPLY_PLAN
        err = capsys.readouterr().err
        assert "Backup kept" not in err
        assert f"Earlier backup kept at {phase_dir / ROUND_ONE_ASIDE}" in err
        assert "round two" in plan.read_text(encoding="utf-8")

    @pytest.mark.parametrize("flags", [("--resume",), ()], ids=["resume", "bare"])
    def test_round_two_interrupted_after_one_batch_keeps_its_own_backup(
        self, repo, capsys, flags
    ):
        """The re-run's progress signals are round two's own, newer than the
        review, so it is a resume of round two whether `--resume` is typed
        (or inferred by a composite) or not."""
        self._round_one(repo)
        plan = self._review(repo, "two")
        ids = self._ids(plan)
        found = plan.read_bytes()
        # Batch 0 attested and applied, batch 1 a partial application: stop.
        assert self._run(repo, plan, ids[0], content="round two\n") == 3
        phase_dir = self._phase_dir(plan)
        assert (phase_dir / "backup.md").read_bytes() == found
        capsys.readouterr()

        assert self._run(repo, plan, sum(ids, []), *flags, content="round two\n") == 0

        assert "Backup kept:" in capsys.readouterr().err
        assert (phase_dir / "backup.md").read_bytes() == found
        assert sorted(p.name for p in phase_dir.glob("backup.*.md")) == [
            ROUND_ONE_ASIDE]
