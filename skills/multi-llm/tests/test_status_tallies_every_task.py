"""`status` tallies every task of implement, not only the last run's batches.

Fix round 16, from the sixth live run (2026-10-06, the check-gate stop
fixture on Codex and OpenCode): after a resume, `status`'s implement block
said "2/2 batches ok, 1/2 items applied" (fix path) and "1/1" (accept
path) — the counts of the run that resumed — and never added up to the
fixture's seven tasks, while summary.md and the stop's own account did.
On the OpenCode route the result's `progress_lines` (the only lines that
host shows) carried the same one line. And the accept path's line read
"accepted after failing T002" for a check accepted with its findings.

The implement block now carries an `all tasks:` line read from
`state.json` (every task of the decomposition, every run, each in one
bucket), a pinned poll's `progress_lines` carry it as `implement tasks:`,
`--json` carries it as `implement_tally`, and the run's own progress line
names a check accepted with its findings as such.
"""

from __future__ import annotations

import json

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: E402,F401
    _check,
    _first_run,
    repo,
)
from tests.test_implement_stop_account import _implement  # noqa: E402
from utils import run_status  # noqa: E402

#: Fix round 20 (run 8 #6): "accepted with its check findings", the words of
#: the `resume: kept` line and of the progress line's kept note.
ACCEPTED = (
    "5/7 done; completed: T001, T003, T004, T007; accepted with its check "
    "findings: T002; skipped at your request: T005; skipped (depend on a task "
    "you skipped): T006"
)


def _status(repo, capsys, *flags):
    code = multi_llm.main(["status", str(repo.plan), *flags])
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.startswith("MULTI_LLM_RESULT:")]
    result = json.loads(lines[-1].split(" ", 1)[1]) if lines else {}
    return code, captured.out, result


def _accept(repo, capsys):
    _first_run(repo, capsys)
    assert _implement(
        repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
    ) == 0
    capsys.readouterr()


class TestAfterAResume:

    def test_plain_status_tallies_all_seven_tasks(self, repo, capsys):
        _accept(repo, capsys)
        _code, out, _result = _status(repo, capsys)
        assert f"      all tasks:      {ACCEPTED}" in out, out
        # The run's own line still counts the run, and names the accepted
        # check as what it is.
        progress = [l for l in out.splitlines() if "progress:" in l and "implement" in l]
        assert progress and "1/1 tasks ok" in progress[0], out
        # Fix round 20 (OpenCode accept 02, run 8 #6): kept from the earlier
        # run, not dispatched, in the `resume: kept` line's words; fix round
        # 22: every task it kept, as that line names them.
        assert (
            "1/1 tasks ok (4 kept from an earlier run: T001, T003, T004, T002 "
            "(accepted with its check findings))"
        ) in progress[0], progress[0]
        # Fix round 24: an implement task is its own one item.
        assert "items applied" not in progress[0], progress[0]
        assert "accepted with their check findings" not in progress[0]
        assert "accepted after failing" not in progress[0]

    def test_the_run_view_tallies_them_too(self, repo, capsys):
        _accept(repo, capsys)
        _code, out, _result = _status(repo, capsys, "--view", "run")
        assert f"      all tasks:      {ACCEPTED}" in out, out

    def test_a_pinned_poll_carries_the_tally_in_its_progress_lines(self, repo, capsys):
        """The OpenCode route shows the result's `progress_lines` and
        nothing else."""
        _accept(repo, capsys)
        _code, _out, result = _status(
            repo, capsys, "--view", "run", "--phase", "implement")
        assert f"implement tasks: {ACCEPTED}" in result["progress_lines"], result

    def test_json_carries_the_tally(self, repo, capsys):
        _accept(repo, capsys)
        multi_llm.main(["status", str(repo.plan), "--view", "run", "--json"])
        out = capsys.readouterr().out
        payload = json.loads(out[: out.index("\nMULTI_LLM_RESULT:")])
        tally = payload["implement_tally"]
        assert tally["total"] == 7 and tally["done"] == 5
        assert tally["buckets"]["accepted_findings"] == ["T002"]


def test_at_the_stop_every_task_is_in_one_bucket(repo, capsys):
    _first_run(repo, capsys)
    _code, out, _result = _status(repo, capsys, "--view", "run")
    assert (
        "      all tasks:      4/7 done; completed: T001, T003, T004; "
        "completed, its check reported problems: T002; failed: T005; "
        "not run: T006, T007"
    ) in out, out


def test_no_decomposition_on_record_adds_no_line(repo, capsys):
    """A plan whose implement never ran has nothing to tally."""
    assert run_status.implement_task_tally(repo.plan) is None
