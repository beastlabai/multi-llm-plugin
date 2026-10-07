"""What a stop and a resume say, said only when it is what happened.

Fix round 14, from the fifth live run's stop fixture (2026-10-06):

* "The half-applied state is deliberately left in place … util restore-batch"
  followed stops where nothing was half-applied — a clean failure (accept 02:
  T007 said `failed` and changed no file), and in `status` a task refused
  before dispatch (fix 01b). It is said only when a failed batch moved bytes.
* `resume: kept 1 task(s) …; dispatched 1; not reached 2` counted the task
  refused before dispatch as dispatched, and kept only the resume skips: an
  implement resume materializes only the unfinished tasks, so OpenCode racing
  06b, which kept 14, printed no `resume: kept` line at all.
* `[CHECK_FINDINGS]` lines reached the log only at the end of a run, never on
  a stop, so the stop fixture's first run (stopped on T005) logged none for
  T002's findings.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_check_gate_fixture import (  # noqa: E402,F401
    ALLOW,
    PLAN,
    _first_run,
    _policy,
    repo,
)
from tests.test_implement_stop_account import _implement  # noqa: E402
from tests.test_write_executor import _patch_edit, env  # noqa: E402,F401
from utils import run_status  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402

HALF_APPLIED = "The half-applied state is deliberately left in place"


def _text(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


def _tally(text: str) -> str:
    lines = [l for l in text.splitlines() if "[write-subagents] resume: kept" in l]
    assert len(lines) == 1, text[-4000:]
    return lines[0]


# ===========================================================================
# [CHECK_FINDINGS] on a stop
# ===========================================================================

def test_a_stop_logs_the_check_findings_lines(repo, capsys):
    text = _first_run(repo, capsys)
    assert "[CHECK_FINDINGS] T002 1 problem(s): retries must be 3 (found 2)" in text
    assert "the tasks that depend on them are held (T007)" in text
    # Before the stop's own block, so the ERROR block stays in one piece.
    assert text.index("[CHECK_FINDINGS] T002") < text.index("ERROR: implement stopped")


# ===========================================================================
# The resume tally
# ===========================================================================

def test_a_plain_resume_counts_what_it_kept_and_does_not_call_a_refusal_dispatched(
    repo, capsys,
):
    """The stop fixture's 01b: nothing changed, T002's findings stand, T005
    is refused again before any runner starts."""
    _first_run(repo, capsys)
    assert _implement(repo, "--resume") == 3
    line = _tally(_text(capsys))
    # Fix round 20 (live run 8 #6): T002 is named as a check whose findings
    # stand, not listed as if it had passed.
    assert line.endswith(
        "resume: kept 4 task(s) an earlier run finished — T001, T003, T004, "
        "T002 (its check reported problems, not run again while the files are "
        "unchanged); dispatched 0; refused before dispatch 1 (T005); not reached 2"
    ), line


def test_an_implement_resume_names_the_tasks_it_did_not_materialize(repo, capsys):
    """OpenCode racing 06b: an implement resume materializes only the
    unfinished tasks, so its resume skips were empty and no line was
    printed although 14 tasks were kept."""
    _first_run(repo, capsys)
    _policy(repo, ALLOW)
    assert _implement(repo, "--resume", "--human-task", "T002=done") == 0
    line = _tally(_text(capsys))
    assert "an earlier run finished — T001, T003, T004" in line, line
    assert "dispatched 3" in line, line
    assert "refused before dispatch" not in line


def test_a_check_accepted_with_done_counts_as_kept_and_says_so(repo, capsys):
    """Run 6, Codex stop-path accept 02: after `--human-task T002=done
    --human-task T005=skip` the line read `kept 3 task(s) … T001, T003,
    T004; dispatched 1`, leaving the accepted check out."""
    _first_run(repo, capsys)
    assert _implement(
        repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
    ) == 0
    line = _tally(_text(capsys))
    assert line.endswith(
        "resume: kept 4 task(s) an earlier run finished — T001, T003, T004, "
        "T002 (accepted with its check findings); dispatched 1"
    ), line


def test_a_failed_task_accepted_with_done_counts_as_kept_and_says_so(repo, capsys):
    """The same for a code task refused before dispatch and answered
    `=done`: it is kept, accepted after failing, and never dispatched."""
    _first_run(repo, capsys)
    assert _implement(
        repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=done",
    ) == 0
    line = _tally(_text(capsys))
    assert (
        "kept 5 task(s) an earlier run finished — T001, T003, T004, "
        "T002 (accepted with its check findings), T005 (accepted after failing); "
    ) in line, line


# ===========================================================================
# "half-applied" only when bytes moved
# ===========================================================================

def test_a_clean_failure_stop_does_not_say_half_applied(repo, capsys):
    """Accept 02's shape: the task said `failed` and changed no file."""
    _policy(repo, ALLOW)
    repo.plan.write_text(PLAN.replace(
        "### T005: Edit: add the build stamp module",
        "### T005: Stamp: add the build stamp module",
    ), encoding="utf-8")
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Stamp:")
    text = _first_run(repo, capsys)
    assert "ERROR: implement stopped at T005" in text
    assert HALF_APPLIED not in text
    assert "inspect the half-applied state" not in text
    assert "Nothing was half-applied: the stopped task changed no file" in text
    report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
    assert report["stop"]["bytes_moved"] is False
    assert "half-applied" not in report["stop"]["headline"]


class TestAStopThatMovedNothing:

    def test_a_prose_reply_that_moved_nothing_offers_no_restore(self, env):
        env.edits([], prose=True)
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.exit_code == 3
        joined = "\n".join(result.messages)
        assert result.outcomes[0].files_touched == []
        assert "inspect the half-applied state" not in joined
        assert HALF_APPLIED not in joined
        # Fix round 18: said once in the block, by its closing line.
        assert joined.count("Nothing was half-applied: the stopped task changed no file") == 1
        assert "nothing to restore" not in joined
        assert "  - fix the cause the details name, then resume this phase" in joined
        assert "Nothing changed, so there is nothing to restore" in result.stop["headline"]

    def test_a_partial_application_still_says_it(self, env):
        env.edits([_patch_edit(env.plan)],
                  result={"status": "failed", "ids": [], "files_touched": [],
                          "summary": "gave up half way"})
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.outcomes[0].status == executor.STATUS_PARTIAL
        joined = "\n".join(result.messages)
        assert HALF_APPLIED in joined
        assert "inspect the half-applied state" in joined
        assert result.stop["bytes_moved"] is True


class TestTheStatusLine:

    @staticmethod
    def _view(files_touched):
        row = {"task_id": "T005", "status": "rejected"}
        if files_touched is not None:
            row["files_touched"] = files_touched
        return {
            "run": {"mode": "implement", "replay_flags": []},
            "phase": "implement",
            "progress": {"write_subagents": {
                "kind": "implement", "total": 7, "batches_dispatched": [0, 1],
                "tasks": [
                    {"task_id": "T001", "status": "ok", "files_touched": ["a"]},
                    row,
                ],
                "failed_task_ids": ["T005"],
            }},
        }

    def _cost(self, files_touched):
        workflow = {"phases": {"implement": {"status": "in_progress"}}}
        lines = run_status._resume_lines(
            self._view(files_touched), Path("plans/p.md"), workflow=workflow,
        )
        return lines[1]

    def test_a_failed_batch_that_moved_nothing_is_not_half_applied(self):
        """Fix round 18 (live run 7): the line names the cost alone. That
        nothing is half-applied is the stop block's closing line, which this
        view shows among the run's last lines; it said so a third time."""
        cost = self._cost([])
        assert cost == (
            "      (re-dispatches the 1 failed batch(es) and the ones never "
            "reached, keeps the 1 already applied)"
        ), cost
        assert "half-applied" not in cost
        assert "restore-batch" not in cost

    def test_a_failed_batch_that_moved_bytes_keeps_the_restore(self):
        cost = self._cost(["src/main.py"])
        assert "the half-applied state is left in place" in cost
        assert "restore-batch" in cost

    def test_an_older_mirror_without_the_key_keeps_the_restore(self):
        assert "restore-batch" in self._cost(None)


# ===========================================================================
# Fix round 15: "bytes moved" is what the sweep observed, not what it can
# restore. `files_touched` is the rollback set; a moved file the store kept no
# bytes for (over the per-file cap, past the budget, unreadable, or a path the
# write policy denies) was absent from it, and the stop announced "Nothing was
# half-applied … nothing to restore" over a file the batch had just changed.
# ===========================================================================

class TestAMoveWithNoRollbackPointIsStillAMove:

    @staticmethod
    def _implement_task(env, task_id="T009"):
        from utils import write_batch_scheduler as wbs

        return env.apply_task(
            0, kind="implement", task_id=task_id, expected_ids=[task_id],
            declared_files=[str(env.root / "src" / "a.py")],
            prompt=f"Do {task_id}\n\n## Changes Applied in Prior Batches\n\n"
                   f"{wbs.PRIOR_CHANGES_TOKEN}\n",
        )

    def test_a_partial_that_moved_only_an_unrestorable_file_says_so(self, env):
        from utils import write_batch_scheduler as wbs

        # The store keeps no bytes for a file over the per-file cap; a small
        # cap stands in for a 16 MB file.
        env.monkeypatch.setattr(wbs, "MAX_SNAPSHOT_BYTES", 64)
        big = env.root / "big.bin"
        big.write_bytes(b"x" * 100)
        env.edits([_patch_edit(big, "more\n")],
                  result={"status": "applied", "ids": [], "files_touched": ["big.bin"],
                          "summary": "wrote the wrong file"})
        result = env.scheduler(
            [self._implement_task(env)], kind="implement", finalize=False,
        ).run()

        outcome = result.outcomes[0]
        assert outcome.status == executor.STATUS_PARTIAL, outcome.reason
        assert outcome.files_touched == [], "nothing restorable joined the rollback set"
        assert outcome.unrestorable == ["big.bin"]
        assert result.stop["bytes_moved"] is True
        joined = "\n".join(result.messages)
        assert "Nothing was half-applied" not in joined
        assert "nothing to restore" not in joined
        assert "Nothing changed" not in result.stop["headline"]
        assert "big.bin" in joined and "cannot undo" in joined, joined
        row = result.to_dict()["tasks"][0]
        assert row["unrestorable"] == ["big.bin"]
        mirror = json.loads((env.phase_dir / ".status.json").read_text())
        mirror_row = mirror["progress"]["write_subagents"]["tasks"][0]
        assert mirror_row["unrestorable"] == ["big.bin"]

    def test_a_policy_violation_on_a_denied_file_is_not_nothing_half_applied(self, env):
        """`.env` is denied by default, so its bytes are never copied into the
        rollback store: the violation cannot restore it, and it stays changed."""
        secret = env.root / ".env"
        secret.write_text("TOKEN=old\n", encoding="utf-8")
        env.edits([_patch_edit(secret, "TOKEN=new\n")],
                  result={"status": "applied", "ids": ["T009"],
                          "files_touched": [".env"], "summary": "set the token"})
        result = env.scheduler(
            [self._implement_task(env)], kind="implement", finalize=False,
        ).run()

        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_POLICY_VIOLATION, outcome.reason
        assert ".env" not in outcome.files_touched
        assert outcome.unrestorable == [".env"]
        assert "TOKEN=new" in secret.read_text()
        assert result.stop["bytes_moved"] is True
        joined = "\n".join(result.messages)
        assert "Nothing was half-applied" not in joined
        assert "nothing to restore" not in joined

    def test_the_status_line_reads_an_unrestorable_move_as_a_move(self):
        view = TestTheStatusLine._view([])
        row = view["progress"]["write_subagents"]["tasks"][1]
        row["unrestorable"] = ["big.bin"]
        workflow = {"phases": {"implement": {"status": "in_progress"}}}
        cost = run_status._resume_lines(view, Path("plans/p.md"), workflow=workflow)[1]
        assert "nothing was half-applied" not in cost
        assert "the half-applied state is left in place" in cost
