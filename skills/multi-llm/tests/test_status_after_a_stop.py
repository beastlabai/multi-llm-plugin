"""`status` after a run that stopped needing the user, a zero-dispatch
resume included (live run 9 #5).

After Codex 01b (`--resume` on unchanged files: nothing dispatched, the same
stop at T005), `status` read::

    implement  run 2843b4c3  failed
        run.state:      failed
        progress:       stopped, implement 0/0 batches ok (1 kept from an
                        earlier run: T002 (…)), 0/0 items applied, …

The run did not fail: it stopped, as the first run had, and needs the user.
It kept four tasks from the earlier run (its own `resume: kept 4 task(s)`
line), not one. A run that stopped needing the user reads so, its kept work
is counted whole, and a run that dispatched nothing says that instead of
"0/0". Implement's counts are tasks, as its `[ETA]` lines are.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: F401  (the fixture)
    _first_run,
    repo,
)
from tests.test_implement_stop_account import _implement
from utils import run_status, status_file

KEPT_FOUR = (
    "(4 kept from an earlier run: T001, T003, T004, T002 (its check reported "
    "problems, not run again while the files are unchanged))"
)


def _progress(repo, capsys) -> tuple:
    code = multi_llm.main([
        "status", str(repo.plan), "--view", "run", "--phase", "implement"])
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "progress:" in l and "implement" in l]
    assert lines, (code, out)
    result = json.loads(
        [l for l in out.splitlines() if l.startswith("MULTI_LLM_RESULT:")][-1]
        .split(" ", 1)[1])
    return lines[0], result.get("progress_lines") or []


class TestAZeroDispatchResume:

    def test_it_reads_as_nothing_dispatched_with_every_kept_task(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(repo, "--resume") == 3
        capsys.readouterr()
        line, polled = _progress(repo, capsys)
        assert f"stopped, implement: nothing dispatched {KEPT_FOUR}" in line, line
        assert "0/0" not in line, line
        # The OpenCode route shows the poll's own rendered lines.
        assert any(KEPT_FOUR in row and "0/0" not in row for row in polled), polled

    def test_a_plain_implement_that_continues_reads_the_same(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(repo) == 3
        capsys.readouterr()
        line, _polled = _progress(repo, capsys)
        assert f"stopped, implement: nothing dispatched {KEPT_FOUR}" in line, line


class TestAForegroundStop:

    def test_its_head_does_not_read_as_running(self, repo, capsys):
        """The write mirror's `progress.step` on a stop is `stopped`, which
        the run view did not count as terminal: a foreground executor run
        that stopped read "running in foreground"."""
        _first_run(repo, capsys)
        assert _implement(repo, "--resume") == 3
        capsys.readouterr()
        multi_llm.main(["status", str(repo.plan), "--view", "run"])
        out = capsys.readouterr().out
        head = [l for l in out.splitlines() if l.startswith("  implement  ")]
        assert head == [
            "  implement  finished in foreground (no detached run record)"
        ], out


class TestTheCountsAreTasks:

    def test_the_first_run_counts_tasks(self, repo, capsys):
        _first_run(repo, capsys)
        line, _polled = _progress(repo, capsys)
        assert "stopped, implement 4/4 tasks ok, 3/4 items applied" in line, line
        assert "batches ok" not in line, line

    def test_a_resume_that_ran_one_task_counts_what_it_kept_whole(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        capsys.readouterr()
        line, _polled = _progress(repo, capsys)
        assert (
            "implement 1/1 tasks ok (4 kept from an earlier run: T001, T003, "
            "T004, T002 (accepted with its check findings)), 1/1 items applied"
        ) in line, line


@pytest.fixture
def plan(tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "plans" / "p.md"
    path.parent.mkdir()
    path.write_text("# Plan\n\n## Step 1\n\nBody.\n", encoding="utf-8")
    for argv in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid",
                  "commit", "-q", "-m", "plan"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True,
                       capture_output=True)
    return path


def _detached_run(plan: Path, *, exit_code, state="failed") -> None:
    phase = plan.parent / plan.stem / "implement"
    phase.mkdir(parents=True, exist_ok=True)
    status_file.claim_run(str(phase), {
        "run_id": "2843b4c3-f684-4592-b7e2-4ab0382afe79", "state": state,
        "exit_code": exit_code, "error": "implement stopped at T005",
        "detached": True, "mode": "implement",
        "replay_flags": ["--host", "codex", "--detach", "--resume"],
        "plan_arg": "plans/p.md",
    })


def _run_block(capsys) -> str:
    run_status.main(["--plan-file", "plans/p.md", "--view", "run"])
    return capsys.readouterr().out


class TestTheHeadOfAStoppedRun:

    def test_exit_3_reads_as_stopped_and_needing_the_user(self, plan, capsys):
        _detached_run(plan, exit_code=3)
        out = _run_block(capsys)
        head = [l for l in out.splitlines() if l.startswith("  implement  run ")]
        assert head == ["  implement  run 2843b4c3  stopped and needs you (exit 3)"], out
        assert (
            "      run.state:      failed (exit 3: the run stopped and needs "
            "you; nothing crashed)"
        ) in out, out

    def test_a_real_failure_still_reads_failed(self, plan, capsys):
        _detached_run(plan, exit_code=1)
        out = _run_block(capsys)
        head = [l for l in out.splitlines() if l.startswith("  implement  run ")]
        assert head == ["  implement  run 2843b4c3  failed"], out
        assert "      run.state:      failed\n" in out, out
