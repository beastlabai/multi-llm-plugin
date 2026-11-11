"""The fan-out's overall deadline outlasts the longest queue behind its gates.

A slot's budget starts once it holds its provider's ``max_concurrent``
semaphore and then the global ``max_parallel`` one, so the overall deadline
of ``review_orchestrator_base.run_models_concurrent`` must outlast both
waits added together: ``concurrency.overall_deadline``. kilocode's six default
models at ``max_concurrent: 2`` are three waves (code-health PD-7), and nine
codex plus nine kilocode slots at ``max_concurrent: 1`` behind a global gate
of three need more than either wait alone (F07 review, point 2).
See docs/development/providers.md#the-overall-deadline.

The schedules are run through the real gating code on a virtual clock: each
slot sleeps its budget in virtual time, and the clock moves to the next wake
only once every task is blocked, so a makespan is exact and no test depends
on wall-clock timing.
"""

import asyncio
import heapq
import itertools
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import concurrency  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.concurrency import overall_deadline  # noqa: E402


class VirtualClock:
    """Virtual time for coroutines that sleep through :meth:`sleep`."""

    def __init__(self):
        self.now = 0.0
        self._wakes = []
        self._order = itertools.count()

    def sleep(self, delay):
        wake = asyncio.get_running_loop().create_future()
        heapq.heappush(self._wakes, (self.now + delay, next(self._order), wake))
        return wake

    async def run(self, coro):
        task = asyncio.ensure_future(coro)
        while True:
            for _ in range(500):  # let every handoff between the gates settle
                if task.done():
                    return task.result()
                await asyncio.sleep(0)
            assert self._wakes, "every slot is blocked and none sleeps"
            when = self._wakes[0][0]
            self.now = when
            while self._wakes and self._wakes[0][0] == when:
                heapq.heappop(self._wakes)[2].set_result(None)


@pytest.fixture
def no_spacing(monkeypatch):
    monkeypatch.setattr(concurrency, "PROVIDER_STAGGER_DELAY", 0.0)


def _fan_out_makespan(monkeypatch, specs, budgets, limits, max_parallel):
    """Run *specs* through ``run_models_concurrent`` on a virtual clock, each
    slot taking its provider's whole budget; return the latest finish."""
    clock = VirtualClock()
    finished = {}
    monkeypatch.setattr(rob, "get_provider_max_concurrent", lambda name, anchor=None: limits.get(name))
    monkeypatch.setattr(rob, "get_provider_timeout", lambda name, anchor=None: budgets[name])

    async def slot(spec, _index):
        await clock.sleep(budgets[spec.split(":", 1)[0]])
        finished[spec] = clock.now
        return {"success": True, "source": "file", "error": None}

    results = asyncio.run(clock.run(rob.run_models_concurrent(specs, slot, None, max_parallel)))
    assert sorted(results) == sorted(specs)
    return max(finished.values())


def _core(specs, budgets, limits, max_parallel):
    """The deadline less its headroom (one largest budget) and stagger."""
    slots = [(spec.split(":", 1)[0], float(budgets[spec.split(":", 1)[0]])) for spec in specs]
    return overall_deadline(slots, max_parallel, limits, 0.0) - max(b for _, b in slots)


def _orders(specs):
    grouped = list(specs)
    by_provider = {}
    for spec in specs:
        by_provider.setdefault(spec.split(":", 1)[0], []).append(spec)
    interleaved = [
        spec for row in itertools.zip_longest(*by_provider.values()) for spec in row if spec
    ]
    shuffled = list(specs)
    random.Random(7).shuffle(shuffled)
    return {"grouped": grouped, "reversed": grouped[::-1], "interleaved": interleaved,
            "shuffled": shuffled}


class TestTheDeadline:
    def test_one_slot_is_twice_its_budget_plus_the_stagger(self):
        assert overall_deadline([("codex", 1800.0)], 5, {}, 4.0) == 3604.0

    def test_a_provider_gate_serialises_its_slots(self):
        """kilocode's six default models, two at a time."""
        assert overall_deadline([("kilocode", 1200.0)] * 6, 5, {"kilocode": 2}, 0.0) == (
            5 * 1200.0 / 2 + 1200.0 + 1200.0
        )

    def test_the_global_and_provider_waits_add_up(self):
        """Codex's case: nine codex and nine kilocode slots, kilocode one at a
        time, a global gate of three. Either wait alone is too short."""
        slots = [("codex", 1.0)] * 9 + [("kilocode", 1.0)] * 9
        core = overall_deadline(slots, 3, {"kilocode": 1}, 0.0) - 1.0
        assert core == 9 / 3 + 8 / 1 + 1

    def test_no_slot_is_the_stagger(self):
        assert overall_deadline([], 5, {}, 3.0) == 3.0


CODEX_CASE = [f"codex:m{i}" for i in range(9)] + [f"kilocode:m{i}" for i in range(9)]


@pytest.mark.parametrize("order", ["grouped", "reversed", "interleaved", "shuffled"])
def test_mixed_queues_finish_inside_the_deadline(monkeypatch, no_spacing, order):
    specs = _orders(CODEX_CASE)[order]
    budgets, limits = {"codex": 1.0, "kilocode": 1.0}, {"kilocode": 1}
    makespan = _fan_out_makespan(monkeypatch, specs, budgets, limits, 3)
    assert makespan <= _core(specs, budgets, limits, 3), (order, makespan)


def _random_case(seed):
    rng = random.Random(seed)
    names = rng.sample(["codex", "kilocode", "cursor-agent", "opencode", "gemini", "claude-code"],
                       rng.randint(1, 4))
    specs = [f"{name}:m{i}" for name in names for i in range(rng.randint(1, 8))]
    budgets = {name: float(rng.choice([1, 2, 3, 5])) for name in names}
    limits = {name: rng.choice([None, 1, 2, 3]) for name in names}
    return specs, budgets, limits, rng.randint(1, 6)


@pytest.mark.parametrize("seed", range(40))
def test_every_schedule_finishes_inside_the_deadline(monkeypatch, no_spacing, seed):
    specs, budgets, limits, max_parallel = _random_case(seed)
    core = _core(specs, budgets, limits, max_parallel)
    for order, ordered in _orders(specs).items():
        makespan = _fan_out_makespan(monkeypatch, ordered, budgets, limits, max_parallel)
        assert makespan <= core, (seed, order, makespan, core)


def test_the_executor_gather_counts_both_waits_without_a_batch_budget(monkeypatch, no_spacing):
    """``run_tasks_concurrent`` with no caller budget sizes its deadline the
    same way (``task_batch`` passes one task budget per task instead)."""
    seen = {}
    real = concurrency.overall_deadline

    def record(slots, max_parallel, limits, stagger):
        seen["value"] = real(slots, max_parallel, limits, stagger)
        return seen["value"]

    monkeypatch.setattr(concurrency, "overall_deadline", record)
    monkeypatch.setattr(concurrency, "get_provider_timeout", lambda name, anchor=None: 10)
    tasks = [{"task_id": f"t{i}"} for i in range(4)]

    async def per_task(task, runner, index):
        return task["task_id"]

    results = asyncio.run(concurrency.run_tasks_concurrent(
        tasks, per_task, runner_for_task=lambda task: "kilocode:m",
        max_parallel=4, provider_limits={"kilocode": 1},
    ))
    assert sorted(results) == ["t0", "t1", "t2", "t3"]
    assert seen["value"] == 3 * 10 + 10 + 10


def test_a_real_mixed_fan_out_completes(monkeypatch, no_spacing):
    """Codex's probe on the wall clock: every slot runs 95% of its budget."""
    monkeypatch.setattr(
        rob, "get_provider_max_concurrent",
        lambda name, anchor=None: 1 if name == "kilocode" else None,
    )

    async def slot(spec, _index):
        await asyncio.sleep(0.19)
        return {"success": True, "source": "file", "error": None}

    results = asyncio.run(rob.run_models_concurrent(CODEX_CASE, slot, 0.2, 3))
    assert sorted(results) == sorted(CODEX_CASE)
