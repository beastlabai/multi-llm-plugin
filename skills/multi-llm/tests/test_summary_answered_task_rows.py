"""A task you answered on this run reads as your answer in summary.md's
Task Results (fix round 26, live run 11 #9).

OpenCode's fix 02 (`--resume --human-task T005=skip` after T005's write
policy refusal) wrote summary.md with T005 twice and two stories: as
"T005 (earlier run) | policy-violation" in `## Task Results` and under
`## Skipped at your request`. The accept path told T002's story twice the
same way: "T002 (earlier run) | check-findings" and "T002 | accepted with its
check findings". A task the user answered on this run now has one row, with
the answer as its outcome and the earlier run's outcome named as what the
answer was about; the `## Skipped at your request` list below it agrees.
"""

from __future__ import annotations

from typing import List, Tuple

from tests.test_check_gate_fixture import (  # noqa: F401  (the fixture)
    _check,
    _first_run,
    repo,
)
from tests.test_implement_stop_account import _implement

POLICY = (
    "Write policy violation (pre-dispatch): src/vendor/stamp.js "
    "(driver.write_paths.deny pattern 'src/vendor/**')"
)


def _task_rows(repo) -> Tuple[List[str], str]:
    summary = (repo.phase_dir / "summary.md").read_text(encoding="utf-8")
    section = summary.split("## Task Results", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if line.startswith("| T")], summary


def _rows_of(rows: List[str], task_id: str) -> List[str]:
    return [row for row in rows if row.startswith(f"| {task_id} ")]


def _skipped_section(summary: str) -> str:
    return summary.split("## Skipped at your request", 1)[1].split("\n## ", 1)[0]


class TestTheFixPath:

    def test_the_skipped_task_reads_skipped_at_your_request_once(self, repo, capsys):
        _first_run(repo, capsys)
        _check(repo, passed=True)
        assert _implement(
            repo, "--resume", "--human-task", "T002=retry", "--human-task", "T005=skip",
        ) == 0
        capsys.readouterr()
        rows, summary = _task_rows(repo)
        t005 = _rows_of(rows, "T005")
        assert len(t005) == 1, rows
        assert t005[0].startswith(
            "| T005 | skipped at your request | — | skipped — you answered "
            "--human-task T005=skip after it failed (earlier run: "
            f"policy-violation: {POLICY}"
        ), t005
        assert "T005 (earlier run)" not in summary
        # The list below tells the same story.
        skipped = _skipped_section(summary)
        assert "| T005 | skipped — you answered --human-task T005=skip" in skipped
        assert "| T006 | skipped — depends on skipped task T005" in skipped
        # The summary sentence counts only the rows still marked earlier.
        assert "carries 3 task(s) recorded by earlier run(s)" in summary, summary


class TestTheAcceptPath:

    def test_the_accepted_check_and_the_skipped_task_have_one_row_each(
        self, repo, capsys,
    ):
        _first_run(repo, capsys)
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        capsys.readouterr()
        rows, summary = _task_rows(repo)
        t002 = _rows_of(rows, "T002")
        assert len(t002) == 1, rows
        assert t002[0].startswith(
            "| T002 | accepted with its check findings | — | accepted — you "
            "answered --human-task T002=done"
        ), t002
        assert "(earlier run: check-findings: " in t002[0], t002
        assert len(_rows_of(rows, "T005")) == 1, rows
        assert "(earlier run)" not in "\n".join(_rows_of(rows, "T002") + _rows_of(rows, "T005"))
        # T007, run on the accepted findings, keeps its own row.
        assert len(_rows_of(rows, "T007")) == 1, rows
