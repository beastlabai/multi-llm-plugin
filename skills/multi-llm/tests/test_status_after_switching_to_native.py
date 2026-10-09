"""`status` shows the run that is current after a phase goes on natively.

The commit review of 2026-10-08 ("Exclusions and limits"): "Stale executor
status after switching to native execution was reproduced, but its
suppression guards predate 4ca1f34." Its reproduction: implement stops on
the executor leg (the check-gate fixture: T002's check reports findings,
T005 is refused by the write policy), the user goes on with the same phase
on the Claude Code native leg (`--resume --human-task T002=done
--human-task T005=skip`), the host records T007 and closes the phase with
`--mark-completed`, and `status --view run --host claude-code` still headed
implement with the executor run's `.status.json`: "implement  stopped in
foreground (needs you; …)", `phase state: subagents_failed`, while the
native run had finished. `state_implement_view` (and `state_apply_views`)
stood back for any executor record in the phase folder, so the native block
never showed.

The native emit now stamps `write_tasks.json` (`leg: native`, the host),
which every run of the phase rewrites on either leg; `status` sets aside an
executor record whose phase folder the native emit wrote last (never a
detached run still going) and shows the native block, which names the
earlier run and says the native one is newer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import multi_llm  # noqa: E402
from tests.test_check_gate_fixture import (  # noqa: F401  (the fixture)
    _first_run,
    repo,
)
from tests.test_implement_stop_account import _implement
from utils import run_status, status_file
from utils.write_phase_bridge import WRITE_TASK_FILENAME

EARLIER = (
    "      earlier run:    a foreground run (its subagents dispatched by the "
    "plugin) stopped and needed you; this run, on the host (claude-code), is newer"
)


def _util(capsys, *argv) -> int:
    code = multi_llm.main(["util", *argv])
    capsys.readouterr()
    return code


def _go_on_natively(repo, capsys, *, finish: bool = True) -> None:
    """The review's reproduction: the same phase, continued on the native
    leg with the stop's answers; the host records T007 and closes it."""
    assert multi_llm.main([
        "implement", str(repo.plan), "--host", "claude-code", "--yes", "--resume",
        "--human-task", "T002=done", "--human-task", "T005=skip",
    ]) == 0
    capsys.readouterr()
    state = str(repo.phase_dir.parent / "state.json")
    assert _util(capsys, "metrics", "start", "--state-file", state,
                 "--phase", "implement", "--total-batches", "1") == 0
    assert _util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
                 "--task-id", "T007", "--status", "completed") == 0
    if finish:
        assert _util(capsys, "metrics", "finish", "--state-file", state,
                     "--phase", "implement") == 0
        assert _util(capsys, "finalize-tracking", "--state-file", state,
                     "--mark-completed") == 0


def _status(capsys, repo, *flags) -> str:
    capsys.readouterr()
    run_status.main(["--plan-file", str(repo.plan), "--host", "claude-code", *flags])
    return capsys.readouterr().out


def _implement_lines(out: str) -> list:
    """The run view's implement heads (not the workflow view's row)."""
    return [line for line in out.splitlines()
            if re.match(r"  implement  \S", line)]


class TestTheNativeRunIsTheOneShown:

    def test_the_reviews_reproduction(self, repo, capsys):
        _first_run(repo, capsys)
        _go_on_natively(repo, capsys)
        for flags in (("--view", "run"), ()):
            out = _status(capsys, repo, *flags)
            assert "stopped in foreground" not in out, out
            assert "subagents_failed" not in out, out
            assert _implement_lines(out) == [
                "  implement  done on the host (marked completed) — no run "
                "record, since the host ran these tasks itself; read from "
                "state.json"
            ], out
            lines = out.splitlines()
            head = lines.index(_implement_lines(out)[0])
            assert lines[head + 1] == EARLIER, out

    def test_json_names_the_earlier_run_beside_the_native_block(self, repo, capsys):
        _first_run(repo, capsys)
        _go_on_natively(repo, capsys)
        payload = json.loads(_status(capsys, repo, "--view", "run", "--json"))
        assert all(
            Path(view["phase_dir"]).name != "implement" for view in payload["runs"]
        ), payload["runs"]
        earlier = payload["implement_tasks"]["earlier_run"]
        assert earlier["leg"] == "executor"
        assert earlier["ended"] == "stopped and needed you"
        assert earlier["native_host"] == "claude-code"

    def test_a_native_run_in_progress_is_shown_too(self, repo, capsys):
        _first_run(repo, capsys)
        _go_on_natively(repo, capsys, finish=False)
        out = _status(capsys, repo, "--view", "run")
        heads = _implement_lines(out)
        assert len(heads) == 1 and "in progress on the host" in heads[0], out
        assert EARLIER in out.splitlines(), out

    def test_a_pinned_run_id_still_reads_that_run(self, repo, capsys):
        """`--run-id` names a run, so nothing is set aside."""
        _first_run(repo, capsys)
        _go_on_natively(repo, capsys)
        run_id = "2843b4c3-f684-4592-b7e2-4ab0382afe79"
        status_file.update_status(str(repo.phase_dir), {"run": {
            "run_id": run_id, "state": "failed", "exit_code": 3, "detached": True,
            "mode": "implement",
        }})
        out = _status(capsys, repo, "--view", "run", "--run-id", run_id)
        assert "run 2843b4c3  stopped and needs you (exit 3)" in out, out
        assert "      earlier run:    " not in out, out


class TestTheStampFollowsTheNewestRun:

    def test_the_native_emit_stamps_and_an_executor_run_drops_it(self, repo, capsys):
        _first_run(repo, capsys)
        tasks = repo.phase_dir / WRITE_TASK_FILENAME
        assert "leg" not in json.loads(tasks.read_text())
        assert multi_llm.main([
            "implement", str(repo.plan), "--host", "claude-code", "--yes", "--resume",
        ]) in (0, 3)
        capsys.readouterr()
        document = json.loads(tasks.read_text())
        assert (document.get("leg"), document.get("host")) == ("native", "claude-code")
        # The executor goes on again: its run is the one shown, as before.
        assert _implement(
            repo, "--resume", "--human-task", "T002=done", "--human-task", "T005=skip",
        ) == 0
        capsys.readouterr()
        assert "leg" not in json.loads(tasks.read_text())
        out = _status(capsys, repo, "--view", "run")
        assert _implement_lines(out) == [
            "  implement  finished in foreground (no detached run record)"
        ], out
        assert "      earlier run:    " not in out, out

    def test_a_detached_run_still_going_is_never_set_aside(self, tmp_path):
        phase = tmp_path / "plans" / "p" / "implement"
        phase.mkdir(parents=True)
        (tmp_path / "plans" / "p.md").write_text("# P\n", encoding="utf-8")
        (phase / WRITE_TASK_FILENAME).write_text(
            json.dumps({"leg": "native", "host": "claude-code"}), encoding="utf-8")
        view = {"phase_dir": str(phase), "verdict": "running", "run": {"run_id": "x"}}
        assert run_status.superseded_by_native(tmp_path / "plans" / "p.md", view) is None
        view["verdict"] = "failed"
        assert run_status.superseded_by_native(
            tmp_path / "plans" / "p.md", view) == {"leg": "native", "host": "claude-code"}

    def test_a_detached_run_started_after_the_native_emit_is_not_set_aside(
        self, tmp_path,
    ):
        """A detached executor attempt that failed or was aborted before it
        rewrote `write_tasks.json` leaves the native stamp behind: the stamp
        alone does not make its record the earlier run."""
        from datetime import datetime

        plan = tmp_path / "plans" / "p.md"
        phase = tmp_path / "plans" / "p" / "implement"
        phase.mkdir(parents=True)
        plan.write_text("# P\n", encoding="utf-8")
        tasks = phase / WRITE_TASK_FILENAME
        tasks.write_text(
            json.dumps({"leg": "native", "host": "claude-code"}), encoding="utf-8")
        emitted = tasks.stat().st_mtime
        stamp = {"leg": "native", "host": "claude-code"}
        for verdict in ("failed", "aborted"):
            view = {"phase_dir": str(phase), "verdict": verdict, "run": {
                "run_id": "y",
                "started_at": datetime.fromtimestamp(emitted + 60).isoformat(),
            }}
            assert run_status.superseded_by_native(plan, view) is None, verdict
            view["run"]["started_at"] = datetime.fromtimestamp(emitted - 60).isoformat()
            assert run_status.superseded_by_native(plan, view) == stamp, verdict
        # The apply emit's recorded `native_run.emitted_at` is the emit's time.
        tasks.write_text(json.dumps({**stamp, "native_run": {
            "emitted_at": datetime.fromtimestamp(emitted - 3600).isoformat(),
        }}), encoding="utf-8")
        view = {"phase_dir": str(phase), "verdict": "failed", "run": {
            "run_id": "y",
            "started_at": datetime.fromtimestamp(emitted - 60).isoformat(),
        }}
        assert run_status.superseded_by_native(plan, view) is None

    def test_status_shows_a_newer_detached_failure_over_the_native_block(
        self, repo, capsys,
    ):
        from datetime import datetime

        _first_run(repo, capsys)
        _go_on_natively(repo, capsys)
        emitted = (repo.phase_dir / WRITE_TASK_FILENAME).stat().st_mtime
        run_id = "5d1e7a2b-0c4f-4e8e-9a51-3f2b6c7d8e90"
        status_file.update_status(str(repo.phase_dir), {"run": {
            "run_id": run_id, "state": "failed", "exit_code": 1, "detached": True,
            "mode": "implement", "error": "worker crashed before dispatch",
            "started_at": datetime.fromtimestamp(emitted + 60).isoformat(),
        }})
        out = _status(capsys, repo, "--view", "run")
        assert "run 5d1e7a2b" in out, out
        assert "worker crashed before dispatch" in out, out
        assert EARLIER not in out.splitlines(), out


@pytest.fixture
def apply_plan(tmp_path, monkeypatch):
    import subprocess

    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "plans" / "p.md"
    path.parent.mkdir()
    path.write_text("# Plan\n\n## Step 1\n\nBody.\n", encoding="utf-8")
    for argv in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid",
                  "commit", "-q", "-m", "plan"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True, capture_output=True)
    return path


class TestAnApplyPhase:
    """The same hole on an apply phase: an executor apply run stopped, the
    host went on with the phase natively."""

    def _native_emit(self, plan: Path) -> Path:
        from utils.state_manager import StateManager
        from utils.write_phase_bridge import (
            materialize_write_tasks, write_task_file_for_phase,
        )

        out = plan.parent / plan.stem
        phase_dir = out / "apply-suggestions"
        grouped = out / "review-plan" / "grouped.json"
        grouped.parent.mkdir(parents=True, exist_ok=True)
        grouped.write_text(json.dumps([{"theme": "t"}]), encoding="utf-8")
        batches = [{"section_key": "s1", "prompt": "Apply.", "suggestions": [
            {"group_id": "g1", "title": "One"}]}]
        tasks = materialize_write_tasks(
            kind="apply_suggestions", batches=batches, phase_dir=str(phase_dir),
            plan_file=str(plan), grouped_file=str(grouped))
        write_task_file_for_phase(
            kind="apply_suggestions", tasks=tasks, phase_dir=str(phase_dir),
            plan_file=str(plan), grouped_file=str(grouped), native_host="claude-code")
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        state.mark_phase_started("apply-suggestions")
        state.save()
        return phase_dir

    def test_the_native_apply_block_replaces_the_executor_record(
        self, apply_plan, capsys,
    ):
        phase_dir = self._native_emit(apply_plan)
        # What the earlier executor run left: its mirror and its report.
        status_file.update_status(str(phase_dir), {
            "state": "subagents_failed",
            "progress": {"step": "stopped", "write_subagents": {
                "kind": "apply_suggestions", "total": 1, "ok": 0,
                "failed_task_ids": ["apply-batch-0"],
            }},
        })
        (phase_dir / "write_batch_report.json").write_text("{}", encoding="utf-8")
        assert multi_llm.main(["util", "batch-done", "--phase-dir", str(phase_dir),
                               "--task-id", "apply-batch-0", "--status",
                               "completed"]) == 0
        capsys.readouterr()
        run_status.main(["--plan-file", str(apply_plan), "--host", "claude-code",
                         "--view", "run"])
        out = capsys.readouterr().out
        heads = [line for line in out.splitlines()
                 if line.startswith("  apply-suggestions  ")]
        assert len(heads) == 1 and "on the host" in heads[0], out
        assert "stopped in foreground" not in out, out
        assert any(line.startswith("      earlier run:    a foreground run")
                   for line in out.splitlines()), out
