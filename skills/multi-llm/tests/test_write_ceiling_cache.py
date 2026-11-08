"""How often the write scheduler resolves runners for its ceiling (code health WRITE-1).

``WriteBatchScheduler.write_ceiling()`` is the minimum of
``driver.max_concurrent_write_subagents``, the runner providers'
``max_concurrent`` and the host's ceiling. The provider leg resolves every
task's runner, and each resolution spawns git, so asking again on every batch
cost about two git processes per task per batch: quadratic over a phase, and
paid in full by a ``--resume`` that dispatches nothing.

Now the phase's ceiling is resolved once (``_ceiling_legs``) and kept as an
upper bound, and each batch resolves its own tasks' runners once
(``_resolve_batch_runners``) for its ceiling, its preflight and its dispatch
alike. A batch is admitted under the phase's number or any lower value the
configuration gives when it starts, never a higher one
(``docs/development/write-safety.md``).

These tests count git spawns and resolutions across several batches, pin that
an unchanged configuration gives every batch the phase's ceiling, and run
parallelisable batches through the real admission and thread pool (only the
task bodies are stand-ins) across a configuration or runner change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from tests.test_write_executor import WriteEnv, env  # noqa: E402,F401
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_batch_scheduler as scheduler_mod  # noqa: E402
from utils.providers.base import PROFILE_AGENT_WRITE  # noqa: E402
from utils.subagent_tasks import (  # noqa: E402
    PRIOR_CHANGES_TOKEN,
    build_inputs,
    make_write_task,
    write_output_name,
)
from utils.write_batch_scheduler import WriteBatchScheduler  # noqa: E402

#: The real project-config discovery, which is where the provider leg spawns
#: git; the suite's default fixture stubs it out.
pytestmark = pytest.mark.config_override

#: The frames that resolve runners for the ceilings.
_CEILING_FRAMES = {
    "write_ceiling", "provider_write_limits", "_ceiling_legs",
    "_provider_limits", "_batch_ceiling", "_resolve_batch_runners",
    "_ceiling_record",
}


class _GitSpawns:
    """Counts git processes, and the ones started under the ceiling."""

    def __init__(self, monkeypatch) -> None:
        self.total = 0
        self.from_ceiling = 0
        original = subprocess.Popen.__init__
        counter = self

        def popen(self, args, *rest, **kwargs):
            argv = list(args) if isinstance(args, (list, tuple)) else [args]
            if argv and os.path.basename(str(argv[0])) in ("git", "git.exe"):
                counter.total += 1
                names = {frame.name for frame in traceback.extract_stack()}
                if names & _CEILING_FRAMES:
                    counter.from_ceiling += 1
            return original(self, args, *rest, **kwargs)

        monkeypatch.setattr(subprocess.Popen, "__init__", popen)

    def reset(self) -> None:
        self.total = 0
        self.from_ceiling = 0


def _uncached(monkeypatch) -> None:
    """Resolve the phase's legs afresh on every call, as the scheduler did
    before."""
    original = WriteBatchScheduler._ceiling_legs

    def every_time(self):
        self._ceiling_legs_cache = None
        return original(self)

    monkeypatch.setattr(WriteBatchScheduler, "_ceiling_legs", every_time)


def _resolutions(monkeypatch) -> List[str]:
    """Every runner resolution the scheduler makes, by task id."""
    calls: List[str] = []
    original = scheduler_mod.resolve_task_runner

    def counting(task, **kwargs):
        calls.append(str(task.get("task_id", "")))
        return original(task, **kwargs)

    monkeypatch.setattr(scheduler_mod, "resolve_task_runner", counting)
    return calls


def _fix_task(env, index, path, *, model_hint=None):
    task = make_write_task(
        kind="apply_code_fixes",
        task_id=f"apply-fix-batch-{index}",
        prompt=f"Apply fix {index} to {env.root / path}\n\n{PRIOR_CHANGES_TOKEN}\n",
        inputs=build_inputs(plan_file=str(env.plan)),
        batch_index=index,
        prior_changes_id=f"{path}#{index:08d}",
        phase_dir=str(env.phase_dir),
        declared_files=[str(env.root / path)],
        expected_ids=[f"f{index}"],
        output_name=write_output_name("apply_code_fixes", index),
    )
    if model_hint:
        task["model_hint"] = model_hint
    return task


def _fix_tasks(env, batches):
    """One task per batch, each on its own file; one runs on claude-code."""
    tasks = []
    for index in range(batches):
        path = f"src/m{index}.py"
        (env.root / path).write_text(f"m = {index}\n", encoding="utf-8")
        hint = "claude-code:sonnet" if index == 1 else None
        tasks.append(_fix_task(env, index, path, model_hint=hint))
    return tasks


def _respond(env):
    """Each batch edits its file and adds a new one, so the tree moves."""
    def respond(task, prompt):
        index = int(task["batch_index"])
        path = f"src/m{index}.py"
        target = env.root / path
        target.write_text(target.read_text() + f"m += {index}\n", encoding="utf-8")
        (env.root / f"notes-{index}.txt").write_text("new\n", encoding="utf-8")
        return {"status": "applied", "ids": [f"f{index}"],
                "files_touched": [path], "summary": f"fix {index}"}
    return respond


def _run(env, tasks):
    return env.scheduler(
        tasks, kind="apply_code_fixes", respond=_respond(env), host_max=8,
        finalize=False,
    ).run()


def _ceilings_seen(monkeypatch):
    """Every ceiling the run hands the disjoint admission, in order."""
    seen = []
    original = scheduler_mod.admit_disjoint_groups

    def recording(tasks, *, ceiling, **kwargs):
        seen.append(ceiling)
        return original(tasks, ceiling=ceiling, **kwargs)

    monkeypatch.setattr(scheduler_mod, "admit_disjoint_groups", recording)
    return seen


def _second_env(tmp_path, monkeypatch):
    """A repo of its own for the uncached run, set up as the first one."""
    base = tmp_path / "uncached"
    base.mkdir()
    second = WriteEnv(base, monkeypatch)
    second.use("codex:gpt-5.5", max_concurrent_write_subagents=4)
    return second


class TestGitSpawnsAcrossBatches:

    BATCHES = 6

    def test_each_task_is_resolved_once_for_the_phase_and_once_for_its_batch(
        self, env, monkeypatch, tmp_path,
    ):
        env.use("codex:gpt-5.5", max_concurrent_write_subagents=4)
        tasks = _fix_tasks(env, self.BATCHES)
        spawns = _GitSpawns(monkeypatch)

        # One resolution pass over these tasks, on a scheduler of its own.
        env.scheduler(tasks, kind="apply_code_fixes", host_max=8)._ceiling_legs()
        one_pass = spawns.from_ceiling
        assert one_pass > 0, "the provider leg resolves runners through git"

        spawns.reset()
        calls = _resolutions(monkeypatch)
        result = _run(env, tasks)
        assert result.exit_code == 0, result.messages
        cached_total, cached_ceiling = spawns.total, spawns.from_ceiling
        # The phase's pass, then each batch's own tasks: linear in the tasks.
        assert sorted(calls) == sorted(
            [task["task_id"] for task in tasks] * 2), calls
        assert cached_ceiling <= 2 * one_pass, (cached_ceiling, one_pass)

        # The same run with the cache defeated: one pass per batch and more.
        second = _second_env(tmp_path, monkeypatch)
        second_tasks = _fix_tasks(second, self.BATCHES)
        _uncached(monkeypatch)
        spawns.reset()
        result = _run(second, second_tasks)
        assert result.exit_code == 0, result.messages
        assert spawns.from_ceiling >= self.BATCHES * one_pass, (
            spawns.from_ceiling, one_pass)
        assert spawns.total - spawns.from_ceiling <= cached_total - cached_ceiling + 4
        print(f"git spawns over {self.BATCHES} batches: cached {cached_total} "
              f"({cached_ceiling} from the ceiling), uncached {spawns.total} "
              f"({spawns.from_ceiling} from the ceiling)")

    def test_a_resume_that_dispatches_nothing_stays_linear(
        self, env, monkeypatch,
    ):
        env.use("codex:gpt-5.5", max_concurrent_write_subagents=4)
        tasks = _fix_tasks(env, self.BATCHES)
        assert _run(env, tasks).exit_code == 0
        spawns = _GitSpawns(monkeypatch)
        env.scheduler(tasks, kind="apply_code_fixes", host_max=8)._ceiling_legs()
        one_pass = spawns.from_ceiling

        spawns.reset()
        resumed = env.scheduler(
            tasks, kind="apply_code_fixes", respond=_respond(env), host_max=8,
            finalize=False, resume=True,
        ).run()
        assert resumed.exit_code == 0, resumed.messages
        assert all(outcome.skipped for outcome in resumed.outcomes)
        assert spawns.from_ceiling <= 2 * one_pass, (spawns.from_ceiling, one_pass)


class TestTheCeilingABatchSees:

    BATCHES = 4

    @pytest.mark.parametrize("hint_runner,expected", [
        # codex has no provider cap: driver 4 against host 8.
        (False, 4),
        # One task on claude-code binds the phase to its cap of 2.
        (True, 2),
    ])
    def test_it_is_the_same_with_and_without_the_cache(
        self, env, monkeypatch, tmp_path, hint_runner, expected,
    ):
        env.use("codex:gpt-5.5", max_concurrent_write_subagents=4)
        tasks = _fix_tasks(env, self.BATCHES)
        if not hint_runner:
            for task in tasks:
                task.pop("model_hint", None)

        seen = _ceilings_seen(monkeypatch)
        result = _run(env, tasks)
        assert result.exit_code == 0, result.messages
        cached = list(seen)
        assert len(cached) >= self.BATCHES, cached

        second = _second_env(tmp_path, monkeypatch)
        second_tasks = _fix_tasks(second, self.BATCHES)
        if not hint_runner:
            for task in second_tasks:
                task.pop("model_hint", None)
        _uncached(monkeypatch)
        seen.clear()
        result = _run(second, second_tasks)
        assert result.exit_code == 0, result.messages
        assert seen == cached
        assert set(cached) == {expected}, cached

    def test_the_report_records_the_number_the_batches_used(self, env, monkeypatch):
        env.use("codex:gpt-5.5", max_concurrent_write_subagents=4)
        tasks = _fix_tasks(env, self.BATCHES)
        seen = _ceilings_seen(monkeypatch)
        result = _run(env, tasks)
        assert result.exit_code == 0, result.messages
        report = json.loads(Path(executor.batch_report_path(
            str(env.phase_dir), "apply_code_fixes")).read_text())
        ceilings = report["ceilings"]
        assert set(seen) == {ceilings["effective"]} == {2}, (seen, ceilings)
        assert ceilings["driver_max_concurrent_write_subagents"] == 4
        assert ceilings["provider_max_concurrent"] == {"codex": None, "claude-code": 2}

    def test_the_provider_map_handed_out_is_a_copy(self, env):
        env.use("codex:gpt-5.5", max_concurrent_write_subagents=4)
        scheduler = env.scheduler(
            _fix_tasks(env, self.BATCHES), kind="apply_code_fixes", host_max=8,
        )
        limits = scheduler.provider_write_limits()
        limits["claude-code"] = 99
        assert scheduler.provider_write_limits()["claude-code"] == 2
        assert scheduler.write_ceiling() == 2


# -- a configuration or runner change between batches ------------------------

#: Tasks per batch: disjoint files, all parallelisable.
_PER_BATCH = 4


def _configure(env, runner="codex:gpt-5.5", *, driver_max, codex_max=None):
    """Point the run at a driver config, as a change the registry recognises."""
    lines = ["driver:", f"  subagent_runner: {runner}",
             f"  max_concurrent_write_subagents: {driver_max}"]
    if codex_max is not None:
        lines += ["providers:", "  codex:", f"    max_concurrent: {codex_max}"]
    path = env.tmp_path / "driver-config.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env.monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
    registry._config = None
    registry._config_key = None


def _parallel_tasks(env, batches=2):
    tasks = []
    for batch in range(batches):
        for slot in range(_PER_BATCH):
            number = batch * 10 + slot
            path = f"src/p{number}.py"
            (env.root / path).write_text(f"p = {number}\n", encoding="utf-8")
            tasks.append(make_write_task(
                kind="apply_code_fixes",
                task_id=f"apply-fix-{batch}-{slot}",
                prompt=(f"Apply fix {number} to {env.root / path}\n\n"
                        f"{PRIOR_CHANGES_TOKEN}\n"),
                inputs=build_inputs(plan_file=str(env.plan)),
                batch_index=batch,
                prior_changes_id=f"{path}#{number:08d}",
                phase_dir=str(env.phase_dir),
                declared_files=[str(env.root / path)],
                expected_ids=[f"f{number}"],
                output_name=write_output_name("apply_code_fixes", number),
                can_parallelize=True,
            ))
    return tasks


class _Bodies:
    """``dispatch_task`` with stand-in task bodies, behind the real admission.

    Each body waits until its batch's four tasks have all started or
    ``HOLD`` seconds pass, so ``peak[batch]`` is exactly the size of the
    largest group the scheduler admitted together. ``runners`` is the spec
    each task was dispatched on.
    """

    HOLD = 0.5

    def __init__(self, monkeypatch) -> None:
        self.cond = threading.Condition()
        self.in_flight = 0
        self.started: Dict[int, int] = {}
        self.peak: Dict[int, int] = {}
        self.runners: Dict[str, str] = {}
        monkeypatch.setattr(scheduler_mod, "dispatch_task", self.dispatch)
        monkeypatch.setattr(scheduler_mod, "preflight_runner",
                            lambda spec, **kwargs: {"version": "test"})
        monkeypatch.setattr(executor, "sandbox_rung", lambda *a, **kw: None)
        monkeypatch.setattr(WriteBatchScheduler, "_render_argv",
                            lambda *a, **kw: ["stand-in"])

    def dispatch(self, task, *, phase_dir, runner, cwd, before_attempt,
                 after_result, **kwargs):
        task_id = task["task_id"]
        batch = int(task["batch_index"])
        before_attempt({"task_id": task_id, "attempt": 1, "cwd": cwd})
        deadline = time.monotonic() + self.HOLD
        with self.cond:
            self.in_flight += 1
            self.runners[task_id] = runner
            self.started[batch] = self.started.get(batch, 0) + 1
            self.peak[batch] = max(self.peak.get(batch, 0), self.in_flight)
            self.cond.notify_all()
            while (self.started[batch] < _PER_BATCH
                   and time.monotonic() < deadline):
                self.cond.wait(timeout=0.05)
        target = Path(task["declared_files"][0])
        target.write_text(target.read_text() + "applied\n", encoding="utf-8")
        with self.cond:
            self.in_flight -= 1
        relative = os.path.relpath(target, cwd)
        data = {"status": "applied", "ids": task["expected_ids"],
                "files_touched": [relative], "summary": f"fixed {task_id}"}
        outcome = executor.TaskOutcome(
            task_id=task_id, kind="apply_code_fixes", runner=runner,
            status=executor.STATUS_OK, profile=PROFILE_AGENT_WRITE,
        )
        outcome.output_path = executor.persist_result(task, phase_dir, data)
        outcome.attempts.append(
            executor.Attempt(index=1, status=executor.STATUS_OK, exit_code=0))
        after_result({"result": data, "outcome": outcome, "attempt": 1,
                      "exit_code": 0, "duration_seconds": 0.1,
                      "sandbox_rung": None})
        return outcome


def _between_batches(monkeypatch, change) -> None:
    """Run ``change`` on the loop's thread once batch 0 has settled."""
    original = WriteBatchScheduler._run_batch

    def run_batch(self, batch_index, tasks, result):
        outcomes = original(self, batch_index, tasks, result)
        if batch_index == 0:
            change()
        return outcomes

    monkeypatch.setattr(WriteBatchScheduler, "_run_batch", run_batch)


class TestAChangeBetweenBatches:
    """Two batches of four disjoint parallelisable tasks, with the configuration
    or the runner changed once the first has settled. The phase's ceiling is
    every batch's upper bound; the second batch binds to any lower value and
    to the runners it really dispatches on."""

    def _run(self, env, monkeypatch, change, *, first_peak=4):
        tasks = _parallel_tasks(env)
        bodies = _Bodies(monkeypatch)
        seen = _ceilings_seen(monkeypatch)
        _between_batches(monkeypatch, change)
        result = env.scheduler(
            tasks, kind="apply_code_fixes", host_max=8, finalize=False,
        ).run()
        assert result.exit_code == 0, result.messages
        assert len(result.outcomes) == 2 * _PER_BATCH, result.outcomes
        # The control: the first batch's bodies did run together.
        assert bodies.peak[0] == first_peak, bodies.peak
        report = json.loads(Path(executor.batch_report_path(
            str(env.phase_dir), "apply_code_fixes")).read_text())
        return bodies, seen, report["ceilings"]

    def test_a_lowered_driver_cap_binds_the_next_batch(self, env, monkeypatch):
        _configure(env, driver_max=4)
        bodies, seen, ceilings = self._run(
            env, monkeypatch, lambda: _configure(env, driver_max=1))
        assert bodies.peak[1] == 1, bodies.peak
        assert seen == [4, 1], seen
        assert ceilings["effective"] == 4
        assert ceilings["driver_max_concurrent_write_subagents"] == 1

    def test_a_lowered_provider_cap_binds_the_next_batch(self, env, monkeypatch):
        _configure(env, driver_max=4)
        bodies, seen, ceilings = self._run(
            env, monkeypatch, lambda: _configure(env, driver_max=4, codex_max=1))
        assert bodies.peak[1] == 1, bodies.peak
        assert seen == [4, 1], seen
        assert ceilings["provider_max_concurrent"] == {"codex": 1}

    def test_a_provider_change_binds_the_next_batch_to_its_runner(
        self, env, monkeypatch,
    ):
        _configure(env, driver_max=4)
        bodies, seen, ceilings = self._run(
            env, monkeypatch,
            lambda: _configure(env, "claude-code:sonnet", driver_max=4))
        # claude-code ships `max_concurrent: 2`.
        assert bodies.peak[1] == 2, bodies.peak
        assert seen == [4, 2], seen
        by_batch = {
            batch: {spec for task_id, spec in bodies.runners.items()
                    if task_id.startswith(f"apply-fix-{batch}-")}
            for batch in (0, 1)
        }
        assert by_batch == {0: {"codex:gpt-5.5"}, 1: {"claude-code:sonnet"}}
        # The report names the provider the second batch ran on.
        assert ceilings["provider_max_concurrent"] == {
            "codex": None, "claude-code": 2}

    def test_a_runner_that_resolves_differently_binds_the_batch_it_runs_in(
        self, env, monkeypatch,
    ):
        """No config change at all: the runner a task resolves to changes, as
        when a CLI goes missing. The batch's ceiling and its dispatches read
        one resolution per task, so they agree."""
        _configure(env, driver_max=4)
        codex_gone = threading.Event()
        calls: List[str] = []

        def resolve(task, **kwargs):
            calls.append(task["task_id"])
            return "claude-code:sonnet" if codex_gone.is_set() else "codex:gpt-5.5"

        monkeypatch.setattr(scheduler_mod, "resolve_task_runner", resolve)
        bodies, seen, ceilings = self._run(env, monkeypatch, codex_gone.set)
        assert bodies.peak[1] == 2, bodies.peak
        assert seen == [4, 2], seen
        assert {spec for task_id, spec in bodies.runners.items()
                if task_id.startswith("apply-fix-1-")} == {"claude-code:sonnet"}
        assert ceilings["provider_max_concurrent"] == {
            "codex": None, "claude-code": 2}
        # The phase's pass, then each batch's own: no dispatch resolves again.
        assert len(calls) == 2 * 2 * _PER_BATCH, calls

    def test_a_raised_cap_stays_at_the_first_ceiling(self, env, monkeypatch):
        _configure(env, driver_max=2)
        bodies, seen, ceilings = self._run(
            env, monkeypatch, lambda: _configure(env, driver_max=4),
            first_peak=2)
        assert bodies.peak[1] == 2, bodies.peak
        assert seen == [2, 2], seen
        assert ceilings["effective"] == 2
