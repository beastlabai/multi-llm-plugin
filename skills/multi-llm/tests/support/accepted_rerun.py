"""An accepted check failure that the check then reports again, on the check-gate world.

``_first_stop`` runs the check-gate fixture to its first stop with the check's
accepted failure, ``_own_check`` scripts the dependent task's own check, and
``_result_line`` / ``_row`` read the run's result line and a task's record.
``RESUME`` answers the stop (T002 done, T005 skipped). No fixtures.
"""

from __future__ import annotations

import json

from tests.support.gate_fixture import NOTE
from tests.support.implement_world import _implement


#: T002's finding as OpenCode's check recorded it in run 9.
ACCEPTED = "node scripts/check.mjs: retries must be 3 (found 2)"
#: What T007's own check reported when it ran the same check again (run 9).
RERUN = "node scripts/check.mjs: check failed: retries must be 3 (found 2) (exit 1)"
#: The Claude Code host's wording of the same T002 finding (run 9, native leg).
NATIVE = (
    "node scripts/check.mjs exited 1 and printed 'check failed: retries must be "
    "3 (found 2)'. src/config.json has retries=2 (set by T001), but the check "
    "requires 3."
)


RESUME = ("--resume", "--human-task", "T002=done", "--human-task", "T005=skip")


def _first_stop(repo, capsys) -> None:
    """Run 1: T002's check reports ACCEPTED (with its command), T005 is
    refused by the write policy, T007 waits on T002."""
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": [ACCEPTED]}))
    assert _implement(repo) == 3
    capsys.readouterr()
    repo.log.unlink(missing_ok=True)


def _own_check(repo, findings) -> None:
    """T007's reply carries its own `check` with *findings*."""
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
        {"passed": False, "findings": list(findings)}))
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", f"Title: {NOTE}\n")


def _result_line(text: str) -> dict:
    line = next(
        line for line in reversed(text.splitlines())
        if line.startswith("MULTI_LLM_RESULT: ")
    )
    return json.loads(line.split(": ", 1)[1])


def _row(repo, task_id: str) -> dict:
    report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
    return next(row for row in report["tasks"] if row["task_id"] == task_id)
