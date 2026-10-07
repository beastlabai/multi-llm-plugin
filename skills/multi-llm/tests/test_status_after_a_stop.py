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

    def test_its_head_reads_as_stopped_and_needing_the_user(self, repo, capsys):
        """The write mirror's `progress.step` on a stop is `stopped`, which
        the run view did not count as terminal: a foreground executor run
        that stopped read "running in foreground". Fix round 23 (review of
        round 22, #2): then "finished in foreground", while the same ending
        detached heads "stopped and needs you (exit 3)"; it says so too."""
        _first_run(repo, capsys)
        assert _implement(repo, "--resume") == 3
        capsys.readouterr()
        multi_llm.main(["status", str(repo.plan), "--view", "run"])
        out = capsys.readouterr().out
        head = [l for l in out.splitlines() if l.startswith("  implement  ")]
        assert head == [
            "  implement  stopped in foreground (needs you; no detached run record)"
        ], out

    def test_a_finished_foreground_run_still_reads_finished(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        capsys.readouterr()
        multi_llm.main(["status", str(repo.plan), "--view", "run"])
        out = capsys.readouterr().out
        head = [l for l in out.splitlines() if l.startswith("  implement  ")]
        assert head == [
            "  implement  finished in foreground (no detached run record)"
        ], out


class TestTheMirrorNamesTheTasksWhoseRunnerFailed:
    """What the run block's gloss reads (fix round 23): a refusal before
    dispatch failed no runner, a runner that failed did."""

    @staticmethod
    def _write_mirror(repo) -> dict:
        status = json.loads((repo.phase_dir / ".status.json").read_text())
        return status["progress"]["write_subagents"]

    def test_a_refusal_fails_no_runner(self, repo, capsys):
        _first_run(repo, capsys)
        mirror = self._write_mirror(repo)
        assert mirror["failed_task_ids"] == ["T005"]
        assert mirror["runner_failed_task_ids"] == []

    def test_a_task_whose_runner_failed_is_named(self, repo, capsys, monkeypatch):
        from tests.test_check_gate_fixture import _check

        _check(repo, passed=False)
        monkeypatch.setenv(
            "MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", "Title: Edit: add the greeting module")
        monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_SUMMARY", "could not add the greeting")
        assert _implement(repo) == 3
        capsys.readouterr()
        assert self._write_mirror(repo)["runner_failed_task_ids"] == ["T003"]


class TestTheCountsAreTasks:

    def test_the_first_run_counts_tasks(self, repo, capsys):
        _first_run(repo, capsys)
        line, _polled = _progress(repo, capsys)
        assert "stopped, implement 4/4 tasks ok, " in line, line
        # Fix round 24 (live run 10 #7): tasks only, never "items".
        assert "batches ok" not in line and "items applied" not in line, line

    def test_a_resume_that_ran_one_task_counts_what_it_kept_whole(self, repo, capsys):
        _first_run(repo, capsys)
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        capsys.readouterr()
        line, _polled = _progress(repo, capsys)
        assert (
            "implement 1/1 tasks ok (4 kept from an earlier run: T001, T003, "
            "T004, T002 (accepted with its check findings))"
        ) in line, line
        assert "items applied" not in line, line


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

    def test_a_runner_failure_reads_stopped_without_nothing_crashed(self, plan, capsys):
        """Fix round 23 (review of round 22, #4): a stop on a task whose
        runner failed needs the user too, but something did fail."""
        _detached_run(plan, exit_code=3)
        status_file.update_status(str(plan.parent / plan.stem / "implement"), {
            "progress": {"step": "stopped", "write_subagents": {
                "kind": "implement", "total": 3, "ok": 2,
                "failed_task_ids": ["T003"], "runner_failed_task_ids": ["T003"],
            }},
        })
        out = _run_block(capsys)
        head = [l for l in out.splitlines() if l.startswith("  implement  run ")]
        assert head == ["  implement  run 2843b4c3  stopped and needs you (exit 3)"], out
        assert (
            "      run.state:      failed (exit 3: the run stopped and needs you)\n"
        ) in out, out
        assert "nothing crashed" not in out, out

    def test_an_older_mirror_with_a_failed_task_says_nothing_crashed_neither(
        self, plan, capsys,
    ):
        _detached_run(plan, exit_code=3)
        status_file.update_status(str(plan.parent / plan.stem / "implement"), {
            "progress": {"step": "stopped", "write_subagents": {
                "kind": "implement", "total": 3, "ok": 2, "failed_task_ids": ["T003"],
            }},
        })
        assert "nothing crashed" not in _run_block(capsys)

    def test_a_real_failure_still_reads_failed(self, plan, capsys):
        _detached_run(plan, exit_code=1)
        out = _run_block(capsys)
        head = [l for l in out.splitlines() if l.startswith("  implement  run ")]
        assert head == ["  implement  run 2843b4c3  failed"], out
        assert "      run.state:      failed\n" in out, out
