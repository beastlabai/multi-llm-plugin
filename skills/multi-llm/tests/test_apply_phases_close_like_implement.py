"""The apply phases' `[ETA]` lines close on what is true, on both legs.

Fix round 23 (review of round 22, progress #1): round 22 made a native
implement task recorded `failed` close its run on "run stopped", as the
executor's does, through `metrics_progress.<phase>.failed` — but `util
batch-done` wrote that key for implement only. On the native leg of the three
apply phases a clean failure of the last batch still read "batch 2/2 … ALL
BATCHES COMPLETE" and `util metrics finish` said "phase complete", while the
executor's same ending said "batch 2/2 done" and "run stopped after batch
2/2". A batch `util write-prompt` refused is never dispatched: the native
record takes it out of the run's total and the closing line names it, as on
implement, and the executor no longer counts it among the batches done
("run stopped after batch 2/3 · apply-batch-1 refused before dispatch").
The closing line's own words count batches on an apply phase, never tasks.
"""

from __future__ import annotations

import re

from tests.test_batch_done import PROMPT, _materialize, plan_dir  # noqa: F401
from tests.test_dispatch_counts import _util
from tests.test_write_executor import FakeState, _patch_edit, env  # noqa: F401
from utils import metrics
from utils.state_manager import StateManager

PHASE = "apply-code-fixes"


def _fix_batches(count: int) -> list:
    return [
        {
            "file_key": f"src/m{index}.py", "prompt": PROMPT, "batch_type": "general",
            "fixes": [{"group_id": f"f-{index}", "title": f"Fix {index}",
                       "file": f"src/m{index}.py"}],
        }
        for index in range(count)
    ]


def _eta(text: str) -> list:
    return [line for line in text.splitlines() if line.startswith(f"[ETA] {PHASE}:")]


class TestTheNativeLeg:
    """The host's loop (`instructions/apply-code-fixes.md` 6 and 9): one `util
    batch-done` then one `util metrics record` per batch it dispatched, and
    `util metrics finish` after the loop."""

    def _start(self, plan_dir, count: int) -> tuple:
        plan, out = plan_dir
        StateManager(plan).save()
        phase_dir, _tasks = _materialize(
            plan, out, "apply_code_fixes", _fix_batches(count), "apply-fixes")
        state_file = str(out / "state.json")
        metrics.start_phase(out / "state.json", PHASE, count)
        return phase_dir, state_file

    @staticmethod
    def _done(capsys, phase_dir, task_id, status, *extra) -> None:
        code, _out, err = _util(
            capsys, "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", task_id, "--status", status, *extra)
        assert code == 0, err

    @staticmethod
    def _record(capsys, state_file, index, total) -> str:
        code, _out, err = _util(
            capsys, "metrics", "record", "--state-file", state_file,
            "--phase", PHASE, "--label", f"Batch {index}",
            "--subagent-type", "agent", "--duration-ms", "1000",
            "--total-batches", str(total), "--batch-index", str(index))
        assert code == 0, err
        return err

    @staticmethod
    def _finish(capsys, state_file) -> list:
        code, _out, err = _util(
            capsys, "metrics", "finish", "--state-file", state_file, "--phase", PHASE)
        assert code == 0, err
        return _eta(err)

    def test_a_failed_last_batch_closes_on_run_stopped(self, plan_dir, capsys):
        phase_dir, state_file = self._start(plan_dir, 2)
        self._done(capsys, phase_dir, "apply-fix-batch-0", "completed")
        self._record(capsys, state_file, 1, 2)
        self._done(capsys, phase_dir, "apply-fix-batch-1", "failed",
                   "--reason", "subagent failed, files unchanged")
        last = _eta(self._record(capsys, state_file, 2, 2))
        # Fix round 25 (stops #2): its record names no batch (`Batch 2`, no
        # `--task-id`), and apply-fix-batch-1 is this run's one failed batch
        # with no record yet, so the line says it failed, as the executor's
        # does since fix round 24.
        assert last == [
            f"[ETA] {PHASE}: batch 2/2 failed (subagent failed, files "
            f"unchanged) after 0m 1s"], last
        closing = self._finish(capsys, state_file)
        assert len(closing) == 1, closing
        assert re.fullmatch(
            rf"\[ETA\] {PHASE}: run stopped after batch 2/2 in [^·]+", closing[0],
        ), closing

    def test_a_failed_batch_recorded_again_completed_completes(self, plan_dir, capsys):
        """The failure is the batch's latest record only: one recorded
        `completed` after it (the host's retry) takes it back."""
        phase_dir, state_file = self._start(plan_dir, 1)
        self._done(capsys, phase_dir, "apply-fix-batch-0", "failed", "--reason", "x")
        self._done(capsys, phase_dir, "apply-fix-batch-0", "completed")
        assert _eta(self._record(capsys, state_file, 1, 1))[0].endswith(
            "ALL BATCHES COMPLETE")
        assert "phase complete" in self._finish(capsys, state_file)[0]

    def test_a_batch_refused_before_dispatch_leaves_the_total_and_is_named(
        self, plan_dir, capsys,
    ):
        phase_dir, state_file = self._start(plan_dir, 2)
        self._done(capsys, phase_dir, "apply-fix-batch-0", "completed")
        self._record(capsys, state_file, 1, 2)
        # `[WRITE_PROMPT_REFUSED]`'s own record; the host dispatches nothing.
        self._done(
            capsys, phase_dir, "apply-fix-batch-1", "failed", "--reason",
            "refused before dispatch: the write policy (driver.write_paths) "
            "does not allow a file it declares")
        closing = self._finish(capsys, state_file)
        assert len(closing) == 1, closing
        assert re.fullmatch(
            rf"\[ETA\] {PHASE}: run stopped after batch 1/1 in [^·]+ · "
            r"apply-fix-batch-1 refused before dispatch",
            closing[0],
        ), closing


class TestTheExecutor:

    def test_a_refused_batch_is_not_counted_among_the_batches_done(self, env):
        (env.root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        env.edits([_patch_edit(env.plan)],
                  result={"status": "applied", "ids": ["s0"],
                          "files_touched": ["plans/p.md"], "summary": "S"})
        lines = []
        tasks = [
            env.apply_task(0),
            env.apply_task(1, declared_files=[str(env.root / ".env")]),
            env.apply_task(2),
        ]
        result = env.scheduler(
            tasks, state=FakeState(env.tmp_path / "state.json"), finalize=False,
            log=lines.append,
        ).run()
        assert result.exit_code == 3, result.messages
        eta = [l for l in lines if l.startswith("[ETA] apply-suggestions:")]
        assert eta[-1] == (
            "[ETA] apply-suggestions: run stopped after batch 1/2 · "
            "apply-batch-1 refused before dispatch; apply-batch-2 not run"
        ), eta


class TestTheWordsCountBatches:

    def test_the_tail_counts_batches_on_an_apply_phase(self):
        left = {f"apply-fix-batch-{n}": "" for n in range(3)}
        left.update({f"b{n}": metrics.LEFT_REFUSED for n in range(3)})
        left.update({"c0": metrics.left_out_check("x"), "d0": metrics.left_out_check("y")})
        left["e0"] = metrics.left_out_check("z")
        tail = metrics.left_out_tail(left, unit=metrics.eta_unit(PHASE))
        assert tail.endswith("more batch(es) not run"), tail
        assert "task" not in tail, tail
        assert metrics.left_out_tail(left).endswith("more task(s) not run")

    def test_a_stop_before_any_batch_finished_says_batches(self):
        assert metrics.closing_line(PHASE, done=0, total=3) == (
            f"[ETA] {PHASE}: run stopped before any of its 3 batch(es) finished"
        )
        assert metrics.closing_line("implement", done=0, total=3) == (
            "[ETA] implement: run stopped before any of its 3 task(s) finished"
        )
