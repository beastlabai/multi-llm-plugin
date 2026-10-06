"""A check's tree stamp follows the run, on every way a native run closes
and across the two legs.

Fix round 15 (review of round 14, stop #2 and #3). A check with findings is
handed out again on a native ``--resume`` only when the files git sees moved
since its stamp (``state.json``'s ``task_check_trees``), and the native run's
closing ``util finalize-tracking`` re-stamps it with the tree the run left, so
the run's own edits after the check are not a change. Two holes:

* ``finalize-tracking --mark-completed`` refused by a held task (exit 3)
  returned before the re-stamp, so the stamp predated T003's edit and the next
  ``--resume`` handed the check out again on files only the run had changed —
  and the first emit's closing step told the host to pass ``--mark-completed``
  even when a ``[CHECK_BLOCKED]`` line had held a task;
* the executor stamps its check's verdict (``verdict.tree_fingerprint``) and
  writes no ``task_check_trees``, so an executor stop resumed natively handed
  the check out again on unchanged files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_check_findings_hold_dependents import (  # noqa: F401 - fixture
    FINDINGS, _check_finds, repo,
)
from tests.test_implement_stop_account import _implement

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402


def _emit(repo, capsys, *flags):
    assert implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--yes", *flags]
    ) == 0
    out = capsys.readouterr().out
    path = next(line.split(": ", 1)[1] for line in out.splitlines()
                if line.startswith("Output file: "))
    return json.loads(Path(path).read_text(encoding="utf-8")), out


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _done(repo, capsys, task_id, *extra):
    code, out, err = _util(
        capsys, "batch-done", "--phase-dir", str(repo.phase_dir), "--task-id",
        task_id, "--status", "completed", *extra,
    )
    assert code == 0, (out, err)
    return out


def _findings():
    return [arg for finding in FINDINGS for arg in ("--finding", finding)]


def _edit(repo, note):
    main = repo.root / "src" / "main.py"
    main.write_text(main.read_text() + f"# {note}\n", encoding="utf-8")


def _emitted(handoff):
    return {t["id"] for b in handoff["batches"] for t in b["tasks"]}


class TestARefusedCloseStillRestamps:

    def _run_to_the_close(self, repo, capsys):
        _emit(repo, capsys)
        _done(repo, capsys, "T001")
        out = _done(repo, capsys, "T002", "--check", "findings", *_findings())
        assert "[CHECK_BLOCKED] T004" in out
        _edit(repo, "T003's edit")
        _done(repo, capsys, "T003")
        return str(get_or_create_state(repo.plan).state_file)

    def test_a_refused_mark_completed_restamps_the_check_tree(self, repo, capsys):
        state_file = self._run_to_the_close(repo, capsys)
        code, _out, err = _util(
            capsys, "finalize-tracking", "--state-file", state_file, "--mark-completed",
        )
        assert code == 3, err
        assert "T004 (waits for T002) did not run" in err
        assert not get_or_create_state(repo.plan).is_phase_completed("implement")

        handoff, out = _emit(repo, capsys, "--resume")
        assert "T002" not in _emitted(handoff), (
            "the check was handed out again on a tree only the run changed")
        assert "T002" in handoff["not_rerun_checks"]
        assert "[CHECK_NOT_RERUN] T002" in out

    def test_the_refusal_says_what_it_wrote(self, repo, capsys):
        state_file = self._run_to_the_close(repo, capsys)
        _code, _out, err = _util(
            capsys, "finalize-tracking", "--state-file", state_file, "--mark-completed",
        )
        assert "Nothing was written" not in err
        assert "implement stays in progress" in err

    def test_the_first_emit_says_a_held_task_closes_without_the_flag(
        self, repo, capsys,
    ):
        _handoff, out = _emit(repo, capsys)
        step = next(line for line in out.splitlines() if line.startswith("  6. "))
        assert "--mark-completed" in step
        assert "[CHECK_BLOCKED]" in step and "without --mark-completed" in step, step
        core = (Path(implement_orchestrator.__file__).parent / "instructions"
                / "implement.md").read_text(encoding="utf-8")
        assert (
            "after a `[CHECK_BLOCKED]` line, a failed task or a stop, run it "
            "without `--mark-completed`"
        ) in core


class TestAnExecutorStopResumedNatively:

    def _executor_stop(self, repo, capsys):
        _check_finds(repo)
        assert _implement(repo) == 3
        capsys.readouterr()
        state = get_or_create_state(repo.plan)
        assert state.get("task_check_findings") == {"T002": FINDINGS}
        assert not state.get("task_check_trees")
        verdict = json.loads(
            (repo.phase_dir / "implement_batch_T002.json").read_text()
        )["verdict"]
        assert verdict.get("tree_fingerprint")

    def test_unchanged_files_keep_the_check_and_its_stamp_follows_the_run(
        self, repo, capsys,
    ):
        self._executor_stop(repo, capsys)

        handoff, out = _emit(repo, capsys, "--resume")
        assert "T002" not in _emitted(handoff), (
            "the native resume ignored the executor's stamp")
        assert "T002" in handoff["not_rerun_checks"]
        assert "T004" in handoff["held_tasks"]
        # Adopted, so the native close re-stamps it like its own.
        assert "T002" in (get_or_create_state(repo.plan).get("task_check_trees") or {})

        _edit(repo, "the native run's own edit")
        state_file = str(get_or_create_state(repo.plan).state_file)
        assert _util(capsys, "finalize-tracking", "--state-file", state_file)[0] == 0
        handoff, _out = _emit(repo, capsys, "--resume")
        assert "T002" not in _emitted(handoff)

    def test_a_changed_file_hands_the_check_out_again(self, repo, capsys):
        self._executor_stop(repo, capsys)
        _edit(repo, "fixed by hand")
        handoff, _out = _emit(repo, capsys, "--resume")
        assert {"T002", "T004"} <= _emitted(handoff)
        assert "not_rerun_checks" not in handoff
        assert "T002" not in (get_or_create_state(repo.plan).get("task_check_trees") or {})
