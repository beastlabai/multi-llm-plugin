"""Launch spacing between dispatches of one provider (PERF-6).

Every timeline here runs on a virtual clock (`tests/launch_timeline.py`): the
fan-outs' real semaphores, tasks and deadlines, with the time each slot
actually dispatched recorded and asserted, not the sleeps it asked for. The
spacing that matters is between *dispatches*: a delay counted from a slot's
admission let two cursor-agent slots queued behind other providers dispatch
together, and let a retry dispatch 0.2 s beside another slot's first attempt.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ask_orchestrator  # noqa: E402
import code_review_orchestrator as cro  # noqa: E402
import review_plan_orchestrator as rpo  # noqa: E402
from utils import progress_tracker  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils import run_status, status_file  # noqa: E402
from utils.launch_limiter import LaunchLimiter  # noqa: E402
from utils.llm_client import FAILURE_CATEGORY_KEY  # noqa: E402
from utils.providers.cursor_agent import CursorAgentProvider  # noqa: E402
from utils.retry_ladder import STOP_DEADLINE, RetryLadder  # noqa: E402

from tests.launch_timeline import inline_threads, now, run_on_virtual_clock  # noqa: E402

STEP = CursorAgentProvider.LAUNCH_STAGGER_S

# The reviewer's first schedule: two other providers hold both global slots
# for 3 s while both cursor-agent slots queue behind them.
CONTENDED = ["gemini:gemini-3-flash", "claude-code:sonnet",
             "cursor-agent:auto", "cursor-agent:sonnet"]
HOLD = {"gemini:gemini-3-flash": 3.0, "claude-code:sonnet": 3.0}


@pytest.fixture(autouse=True)
def adapter_spacing(monkeypatch):
    """The adapters' own intervals: no uniform override, no fast backoff."""
    monkeypatch.delenv("MULTI_LLM_TEST_FAST_BACKOFF", raising=False)
    monkeypatch.setattr(base, "PROVIDER_STAGGER_DELAY", None)
    assert STEP == 2.0


def _cursor_times(dispatched):
    return sorted(t for spec, t in dispatched if spec.startswith("cursor-agent"))


def _assert_spaced(times, interval=STEP):
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(gap >= interval - 1e-9 for gap in gaps), (times, gaps)


# ---------------------------------------------------------------------------
# The limiter itself
# ---------------------------------------------------------------------------

class TestTheLimiter:

    def test_waiters_are_served_in_arrival_order_one_interval_apart(self):
        limiter = LaunchLimiter({"cursor-agent": STEP})
        claimed, announced = {}, {}

        async def waiter(name):
            await limiter.wait_turn(
                "cursor-agent", name,
                on_wait=lambda s: announced.setdefault(name, s),
            )
            claimed[name] = now()

        async def main():
            await asyncio.gather(*(waiter(n) for n in ("a", "b", "c")))

        run_on_virtual_clock(main())
        assert claimed == {"a": 0.0, "b": STEP, "c": 2 * STEP}
        # Each wait names the real turn, not one interval for everyone.
        assert announced == {"b": STEP, "c": 2 * STEP}

    def test_a_waiter_that_gave_up_took_no_turn(self):
        limiter = LaunchLimiter({"cursor-agent": STEP})
        claimed = {}

        async def main():
            assert await limiter.wait_turn("cursor-agent", "a")
            quitter = asyncio.ensure_future(limiter.wait_turn("cursor-agent", "b"))
            await asyncio.sleep(0.5)
            quitter.cancel()
            assert await limiter.wait_turn("cursor-agent", "c", budget=0.5) is False
            assert now() == 1.0
            await limiter.wait_turn("cursor-agent", "d")
            claimed["d"] = now()

        run_on_virtual_clock(main())
        # Neither the cancelled waiter nor the one whose budget ran out holds a
        # turn: the next arrival goes one interval after the last real claim.
        assert claimed == {"d": STEP}

    def test_an_unspaced_provider_never_waits_and_leaves_no_state(self):
        limiter = LaunchLimiter({"cursor-agent": STEP, "gemini": 0.0})
        times = []

        async def main():
            for name in ("a", "b", "c"):
                assert await limiter.wait_turn("gemini", name)
                times.append(now())

        run_on_virtual_clock(main())
        assert times == [0.0, 0.0, 0.0]
        assert not limiter.spaces("gemini")
        assert limiter._last == {} and limiter._waiting == {}

    def test_a_confirmed_turn_moves_to_the_dispatch(self):
        limiter = LaunchLimiter({"cursor-agent": STEP})
        claimed = {}

        async def late_dispatch():
            await limiter.wait_turn("cursor-agent", "a")
            await asyncio.sleep(1.5)  # its preparation
            assert limiter.confirm("cursor-agent", "a")

        async def next_one():
            await asyncio.sleep(0.1)
            await limiter.wait_turn("cursor-agent", "b")
            claimed["b"] = now()

        async def main():
            await asyncio.gather(late_dispatch(), next_one())

        run_on_virtual_clock(main())
        assert claimed == {"b": 1.5 + STEP}

    def test_a_turn_taken_by_someone_else_is_not_confirmed(self):
        limiter = LaunchLimiter({"cursor-agent": STEP})

        async def main():
            await limiter.wait_turn("cursor-agent", "a")
            await limiter.wait_turn("cursor-agent", "b")
            assert now() == STEP
            assert limiter.confirm("cursor-agent", "a") is False
            assert limiter.confirm("cursor-agent", "b") is True

        run_on_virtual_clock(main())

    def test_the_blocking_wait_uses_the_same_turns(self):
        clock = [0.0]
        slept = []

        def pause(_woken, seconds):
            slept.append(seconds)
            clock[0] += seconds

        limiter = LaunchLimiter({"cursor-agent": STEP}, clock=lambda: clock[0])
        assert limiter.wait_turn_blocking("cursor-agent", "a", pause=pause)
        assert limiter.wait_turn_blocking("cursor-agent", "b", pause=pause)
        assert (clock[0], slept) == (STEP, [STEP])
        assert limiter.wait_turn_blocking(
            "cursor-agent", "c", budget=1.0, pause=pause
        ) is False
        assert clock[0] == STEP + 1.0

    def test_no_turn_is_granted_once_the_deadline_has_passed(self):
        limiter = LaunchLimiter({"cursor-agent": STEP})

        async def main():
            assert await limiter.wait_turn("cursor-agent", "a")
            # b's deadline is the very moment the next turn opens: no turn.
            assert await limiter.wait_turn("cursor-agent", "b", budget=STEP) is False
            assert now() == STEP
            # So that turn is still free for the next arrival, at once.
            assert await limiter.wait_turn("cursor-agent", "c")
            assert now() == STEP
            # A spent budget is refused even with the turn free.
            await asyncio.sleep(STEP)
            assert await limiter.wait_turn("cursor-agent", "d", budget=0.0) is False
            assert limiter._last["cursor-agent"] == STEP

        run_on_virtual_clock(main())

    def test_no_blocking_turn_is_granted_once_the_deadline_has_passed(self):
        clock = [0.0]

        def pause(_woken, seconds):
            clock[0] += seconds

        limiter = LaunchLimiter({"cursor-agent": STEP}, clock=lambda: clock[0])
        assert limiter.wait_turn_blocking("cursor-agent", "a", pause=pause)
        assert limiter.wait_turn_blocking(
            "cursor-agent", "b", budget=STEP, pause=pause
        ) is False
        assert clock[0] == STEP
        assert limiter.wait_turn_blocking("cursor-agent", "c", pause=pause)
        assert clock[0] == STEP and limiter._last["cursor-agent"] == STEP

    def test_a_waiter_that_leaves_wakes_the_ones_behind_it(self):
        limiter = LaunchLimiter({"cursor-agent": STEP})
        claimed, waits = {}, {}

        async def main():
            assert await limiter.wait_turn("cursor-agent", "a")
            # b gives up at 1.0; c, behind it, first works out a turn at 4.0
            # (its place times the interval), with a deadline at 3.0.
            leaver = asyncio.ensure_future(
                limiter.wait_turn("cursor-agent", "b", budget=1.0))
            await asyncio.sleep(0)
            survivor = await limiter.wait_turn(
                "cursor-agent", "c", budget=3.0,
                on_wait=lambda s: waits.setdefault("c", []).append((now(), s)),
            )
            claimed["c"] = (survivor, now())
            assert await leaver is False

        run_on_virtual_clock(main())
        # Woken when b left, c moved up and took the turn at 2.0.
        assert waits["c"] == [(0.0, 3.0), (1.0, 1.0)]
        assert claimed == {"c": (True, STEP)}

    def test_a_blocking_waiter_is_woken_when_the_one_ahead_leaves(self):
        clock = [0.0]
        limiter = LaunchLimiter({"cursor-agent": STEP}, clock=lambda: clock[0])
        assert limiter.wait_turn_blocking("cursor-agent", "a")
        b_waiting, release_b, c_waiting = (threading.Event() for _ in range(3))
        c_waits, outcome = [], {}

        def pause_b(_woken, seconds):
            b_waiting.set()
            assert release_b.wait(5)
            clock[0] += seconds  # it slept to its deadline

        def pause_c(woken, seconds):
            c_waits.append((clock[0], seconds))
            if len(c_waits) == 1:
                c_waiting.set()
                # Real time: only a wake from b's departure ends this early.
                if woken.wait(5):
                    return
            clock[0] += seconds

        def wait(name, budget, pause):
            outcome[name] = limiter.wait_turn_blocking(
                "cursor-agent", name, budget=budget, pause=pause)
            outcome[name + "_at"] = clock[0]

        b = threading.Thread(target=wait, args=("b", 1.0, pause_b))
        b.start()
        assert b_waiting.wait(5)
        c = threading.Thread(target=wait, args=("c", 3.0, pause_c))
        c.start()
        assert c_waiting.wait(5)
        release_b.set()
        b.join(5)
        c.join(10)
        assert outcome["b"] is False
        assert c_waits == [(0.0, 3.0), (1.0, 1.0)]
        assert (outcome["c"], outcome["c_at"]) == (True, STEP)


# ---------------------------------------------------------------------------
# run_models_concurrent: review-plan, review-tasks and ask
# ---------------------------------------------------------------------------

def _holding_factory(dispatched, hold=HOLD, default=0.5):
    """A slot with no ladder (the ask lane's shape): it dispatches on entry."""
    async def factory(model_spec, index):
        dispatched.append((model_spec, now()))
        await asyncio.sleep(hold.get(model_spec, default))
        return {"success": True, "source": "file"}
    return factory


class TestTheBaseFanOut:

    def test_cursor_slots_queued_behind_other_providers_do_not_dispatch_together(self):
        dispatched = []
        run_on_virtual_clock(base.run_models_concurrent(
            list(CONTENDED), _holding_factory(dispatched), 600, 2,
        ))
        times = dict(dispatched)
        assert times["gemini:gemini-3-flash"] == 0.0
        assert times["claude-code:sonnet"] == 0.0
        # Both freed global slots at 3.0; the second cursor slot waits its turn.
        assert _cursor_times(dispatched) == [3.0, 3.0 + STEP]

    def test_a_slot_waits_for_its_turn_as_staggering_only_once_admitted(self, tmp_path):
        tracker = progress_tracker.FanoutProgress(
            str(tmp_path), list(CONTENDED), timeout=600, interval_s=0
        )
        seen = []
        real = tracker.staggering

        def spy(spec, **kwargs):
            seen.append((spec, now(), kwargs))
            real(spec, **kwargs)

        dispatched = []
        with patch.object(tracker, "staggering", spy):
            run_on_virtual_clock(base.run_models_concurrent(
                list(CONTENDED), _holding_factory(dispatched), 600, 2,
                tracker=tracker,
            ))
        # Parked on the global semaphore until 3.0, the slot read `queued`;
        # only then did it wait, for one interval, as `staggering`.
        assert seen == [("cursor-agent:sonnet", 3.0, {"delay_s": STEP})]
        assert all(e["state"] == "done" for e in tracker.snapshot().values())

    def test_a_cursor_retry_takes_its_own_turn(self):
        """The reviewer's second schedule, through the real ladder loop."""
        specs = ["cursor-agent:auto", "gemini:gemini-3-flash", "cursor-agent:sonnet"]
        dispatched = []

        async def factory(model_spec, index):
            ladder = RetryLadder(timeout=600, retries=1, now=now)
            attempts = []

            async def dispatch(attempt_timeout):
                dispatched.append((model_spec, now()))
                attempts.append(attempt_timeout)
                if model_spec == "cursor-agent:auto" and len(attempts) == 1:
                    await asyncio.sleep(1.8)
                    return {"success": False, "error": "boom",
                            "details": {FAILURE_CATEGORY_KEY: None}}
                await asyncio.sleep(5.0)
                return {"success": True, "details": {}}

            run = await base.dispatch_with_ladder(
                dispatch, ladder=ladder, model_spec=model_spec,
                display_name=model_spec, timeout=600,
                start_line=lambda attempt, timeout: f"{model_spec} {attempt}",
                jail_summary_key=None,
            )
            return {"success": bool(run.result and run.result["success"]),
                    "source": "file"}

        run_on_virtual_clock(base.run_models_concurrent(specs, factory, 600, 3))
        assert dict(dispatched)["gemini:gemini-3-flash"] == 0.0
        cursor = [(spec, t) for spec, t in dispatched if spec.startswith("cursor")]
        # The retry at 1.8 s queues behind the second slot's turn at 2.0 s.
        assert cursor == [("cursor-agent:auto", 0.0), ("cursor-agent:sonnet", STEP),
                          ("cursor-agent:auto", 2 * STEP)]

    def test_spacing_counts_from_the_dispatch_not_from_the_turn(self):
        """A slot that prepares after its turn pushes the next dispatch back."""
        specs = ["cursor-agent:auto", "cursor-agent:sonnet"]
        prep = {"cursor-agent:auto": 1.0}
        dispatched = []

        async def factory(model_spec, index):
            await asyncio.sleep(prep.get(model_spec, 0.0))

            async def dispatch(attempt_timeout):
                dispatched.append((model_spec, now()))
                return {"success": True, "details": {}}

            await base.dispatch_with_ladder(
                dispatch, ladder=RetryLadder(timeout=600, retries=0, now=now),
                model_spec=model_spec, display_name=model_spec, timeout=600,
                start_line=lambda attempt, timeout: "", jail_summary_key=None,
            )
            return {"success": True, "source": "file"}

        run_on_virtual_clock(base.run_models_concurrent(specs, factory, 600, 2))
        assert dispatched == [("cursor-agent:auto", 1.0),
                              ("cursor-agent:sonnet", 1.0 + STEP)]

    def test_a_retrys_wait_is_charged_to_the_slot_deadline(self):
        specs = ["cursor-agent:auto", "cursor-agent:sonnet"]
        budgets, runs = {}, {}

        def make_factory(timeout):
            async def factory(model_spec, index):
                ladder = RetryLadder(timeout=timeout, retries=1, now=now)

                async def dispatch(attempt_timeout):
                    budgets.setdefault(model_spec, []).append((now(), attempt_timeout))
                    if model_spec == "cursor-agent:auto":
                        await asyncio.sleep(0.1)
                        return {"success": False, "error": "boom",
                                "details": {FAILURE_CATEGORY_KEY: None}}
                    await asyncio.sleep(3.0)
                    return {"success": True, "details": {}}

                runs[model_spec] = await base.dispatch_with_ladder(
                    dispatch, ladder=ladder, model_spec=model_spec,
                    display_name=model_spec, timeout=timeout,
                    start_line=lambda attempt, t: "", jail_summary_key=None,
                )
                return {"success": False, "source": "file"}
            return factory

        run_on_virtual_clock(base.run_models_concurrent(specs, make_factory(10), 600, 2))
        # The retry waited until 4.0 for its turn, and runs on what is left.
        assert budgets["cursor-agent:auto"] == [(0.0, 10), (2 * STEP, 6)]

        budgets.clear()
        run_on_virtual_clock(base.run_models_concurrent(specs, make_factory(3), 600, 2))
        # A 3 s slot cannot reach a turn at 4.0: no second dispatch, and the
        # slot ends on its deadline.
        assert budgets["cursor-agent:auto"] == [(0.0, 3)]
        assert runs["cursor-agent:auto"].ladder.stop_reason == STOP_DEADLINE

    def test_a_surviving_retry_gets_the_turn_a_departed_waiter_freed(self):
        """The reviewer's third schedule, through the real ladder loop.

        A dispatches at 0, B at 2.0; B's retry queues at 2.1 and A's at 2.2.
        B's 1.5 s slot ends at 3.5 with no turn, which frees the 4.0 turn for
        A, whose 5.1 s slot still has 1.1 s: enough for one more attempt.
        """
        specs = ["cursor-agent:auto", "cursor-agent:sonnet"]
        shapes = {"cursor-agent:auto": (5.1, 2.2), "cursor-agent:sonnet": (1.5, 0.1)}
        dispatched, runs = [], {}

        async def factory(model_spec, index):
            timeout, fails_after = shapes[model_spec]
            ladder = RetryLadder(timeout=timeout, retries=1, now=now)

            async def dispatch(attempt_timeout):
                dispatched.append((model_spec, now()))
                if ladder.attempt_index == 1:
                    await asyncio.sleep(fails_after)
                    return {"success": False, "error": "boom",
                            "details": {FAILURE_CATEGORY_KEY: None}}
                await asyncio.sleep(0.5)
                return {"success": True, "details": {}}

            runs[model_spec] = await base.dispatch_with_ladder(
                dispatch, ladder=ladder, model_spec=model_spec,
                display_name=model_spec, timeout=timeout,
                start_line=lambda attempt, t: "", jail_summary_key=None,
            )
            return {"success": True, "source": "file"}

        run_on_virtual_clock(base.run_models_concurrent(specs, factory, 600, 2))
        assert [(spec, round(t, 6)) for spec, t in dispatched] == [
            ("cursor-agent:auto", 0.0), ("cursor-agent:sonnet", STEP),
            ("cursor-agent:auto", 2 * STEP),
        ]
        assert runs["cursor-agent:auto"].result["success"] is True
        assert runs["cursor-agent:sonnet"].ladder.stop_reason == STOP_DEADLINE

    def test_different_providers_still_start_together(self):
        specs = ["gemini:gemini-3-flash", "claude-code:sonnet", "codex:gpt-5.5",
                 "cursor-agent:auto", "gemini:gemini-3-pro", "codex:gpt-5.4"]
        dispatched = []
        run_on_virtual_clock(base.run_models_concurrent(
            specs, _holding_factory(dispatched, hold={}), 600, 6,
        ))
        assert sorted(t for _spec, t in dispatched) == [0.0] * len(specs)

    def test_the_deadline_margin_is_the_largest_providers(self):
        limiter, overhead = base.plan_launch_limiter(
            ["cursor-agent", "gemini", "cursor-agent", "cursor-agent"])
        assert overhead == 3 * STEP
        assert limiter.spaces("cursor-agent") and not limiter.spaces("gemini")
        assert base.plan_launch_limiter(["gemini", "codex", "gemini"])[1] == 0.0


# ---------------------------------------------------------------------------
# The other fan-outs that stagger: review-code, review-plan's own, tasks
# ---------------------------------------------------------------------------

class TestTheOtherFanOuts:

    def test_review_code(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        dispatched = []

        async def fake_review(model_spec, *args, **kwargs):
            dispatched.append((model_spec, now()))
            await asyncio.sleep(HOLD.get(model_spec, 0.5))
            kwargs["tracker"].done(model_spec, source="file")
            return True, [], None

        with patch.object(cro, "run_single_review", new=fake_review):
            run_on_virtual_clock(cro.run_all_reviews(
                list(CONTENDED), plan_path, tmp_path, 600, 2,
            ))
        assert dict(dispatched)["gemini:gemini-3-flash"] == 0.0
        assert _cursor_times(dispatched) == [3.0, 3.0 + STEP]

    def test_review_plans_own_loop(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        dispatched = []

        async def fake_model(model_spec, *args, **kwargs):
            dispatched.append((model_spec, now()))
            await asyncio.sleep(HOLD.get(model_spec, 0.5))
            return {"success": True}

        with patch.object(rpo, "run_single_model", new=fake_model):
            run_on_virtual_clock(rpo.run_all_models(
                list(CONTENDED), str(plan_path), 600, 2,
            ))
        assert dict(dispatched)["claude-code:sonnet"] == 0.0
        assert _cursor_times(dispatched) == [3.0, 3.0 + STEP]

    def test_the_task_variant(self):
        tasks = [{"task_id": f"t{i}"} for i in range(len(CONTENDED))]
        runners = {f"t{i}": spec for i, spec in enumerate(CONTENDED)}
        dispatched = []

        async def per_task(task, runner, index):
            dispatched.append((runner, now()))
            await asyncio.sleep(HOLD.get(runner, 0.5))
            return runner

        run_on_virtual_clock(base.run_tasks_concurrent(
            tasks, per_task, runner_for_task=lambda t: runners[t["task_id"]],
            timeout=600, max_parallel=2,
        ))
        assert dict(dispatched)["gemini:gemini-3-flash"] == 0.0
        assert _cursor_times(dispatched) == [3.0, 3.0 + STEP]


# ---------------------------------------------------------------------------
# The lanes with no ladder: ask, and review-plan without an output dir
# ---------------------------------------------------------------------------

#: The first slot's pre-dispatch preparation (its sandbox probe here; a large
#: repository's porcelain snapshot in the field) runs past the next turn.
SLOW_PREP = {"sandbox_refusal": [1.9]}


class TestTheLanesWithNoLadder:
    """One dispatch per slot, which takes the turn after the preparation."""

    def test_ask(self, tmp_path, monkeypatch):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\nBuild a widget.\n", encoding="utf-8")
        inline_threads(monkeypatch, SLOW_PREP)
        dispatched = []

        def invoker(prompt, model_spec, **_kwargs):
            dispatched.append((model_spec, now()))
            return {"success": True, "data": "An answer.",
                    "details": {"answer_text": "An answer."}}

        monkeypatch.setattr(ask_orchestrator, "invoke_with_provider", invoker)
        rc = run_on_virtual_clock(ask_orchestrator.main([
            "--plan-file", str(plan), "--question", "Is it sound?",
            "--models", "cursor-agent:auto", "cursor-agent:sonnet",
        ]))
        assert rc == 0
        # The first slot dispatched at 1.9, after its probe; the second waits
        # one interval from there, not from the first slot's turn at 0.
        assert [(spec, round(t, 6)) for spec, t in dispatched] == [
            ("cursor-agent:auto", 1.9), ("cursor-agent:sonnet", 1.9 + STEP),
        ]

    def test_review_plan_without_an_output_dir(self, tmp_path, monkeypatch):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\nBuild a widget.\n", encoding="utf-8")
        inline_threads(monkeypatch, SLOW_PREP)
        dispatched = []

        def invoker(prompt, model_spec, **_kwargs):
            dispatched.append((model_spec, now()))
            return {"success": True, "data": "[]", "details": {}}

        monkeypatch.setattr(base, "invoke_with_provider", invoker)
        results = run_on_virtual_clock(rpo.run_all_models(
            ["cursor-agent:auto", "cursor-agent:sonnet"], str(plan), 600, 2,
        ))
        assert all(r["success"] for r in results.values()), results
        assert [(spec, round(t, 6)) for spec, t in dispatched] == [
            ("cursor-agent:auto", 1.9), ("cursor-agent:sonnet", 1.9 + STEP),
        ]


# ---------------------------------------------------------------------------
# A retry waiting for its turn, as the progress readers see it
# ---------------------------------------------------------------------------

class _Clocks:
    """The tracker's monotonic and wall clocks, locked together, moved by hand."""

    WALL = 1_789_000_000.0

    def __init__(self):
        self.t = 0.0

    def install(self, monkeypatch):
        monkeypatch.setattr(progress_tracker, "_monotonic", lambda: self.t)
        monkeypatch.setattr(progress_tracker, "_wall_time", self.wall)
        monkeypatch.setattr(
            progress_tracker, "_now_iso",
            lambda: datetime.fromtimestamp(self.wall()).isoformat(),
        )

    def wall(self):
        return self.WALL + self.t


class TestARetryWaitingForItsTurn:
    """Admitted at 0, first dispatched at 2.0, failed, waiting for a launch
    turn as ``staggering`` and read at 12.0: 10 s run, 2 s queued."""

    SPEC = "cursor-agent:auto"

    def _retry_row(self, tmp_path, monkeypatch):
        clocks = _Clocks()
        clocks.install(monkeypatch)
        tracker = progress_tracker.FanoutProgress(
            str(tmp_path), [self.SPEC], timeout=600, interval_s=0)
        tracker.seed()
        clocks.t = 2.0
        tracker.running(self.SPEC, attempt=1)
        clocks.t = 5.0
        tracker.staggering(self.SPEC, delay_s=STEP)
        clocks.t = 12.0
        row = tracker.snapshot()[self.SPEC]
        assert row["state"] == progress_tracker.STATE_STAGGERING
        return row, clocks.wall()

    def test_its_run_time_and_its_wait(self, tmp_path, monkeypatch):
        row, reading = self._retry_row(tmp_path, monkeypatch)
        assert not progress_tracker.awaits_first_dispatch(row)
        assert progress_tracker.entry_elapsed_s(row, reading) == pytest.approx(10.0)
        assert progress_tracker.entry_queued_s(row, reading) == pytest.approx(2.0)

    def test_a_slot_waiting_for_its_first_turn_is_still_only_a_wait(
        self, tmp_path, monkeypatch
    ):
        clocks = _Clocks()
        clocks.install(monkeypatch)
        tracker = progress_tracker.FanoutProgress(
            str(tmp_path), [self.SPEC], timeout=600, interval_s=0)
        tracker.seed()
        clocks.t = 3.0
        tracker.staggering(self.SPEC, delay_s=STEP)
        clocks.t = 4.0
        row = tracker.snapshot()[self.SPEC]
        assert progress_tracker.awaits_first_dispatch(row)
        assert progress_tracker.entry_elapsed_s(row, clocks.wall()) is None
        assert progress_tracker.entry_queued_s(row, clocks.wall()) == pytest.approx(4.0)

    def test_the_status_table_renders_its_run_time(self):
        # Stamps relative to the real clock: the table reads that one.
        stamp = lambda ago: datetime.fromtimestamp(time.time() - ago).isoformat()
        row = {
            "state": progress_tracker.STATE_STAGGERING, "timeout_s": 600,
            "started_at": stamp(12.0), "dispatched_at": stamp(10.0),
            "queued_s": 2.0, "attempt": 2,
        }
        assert run_status._model_timing_text(row).startswith("0m1")
        assert "/10m" in run_status._model_timing_text(row)
        assert not run_status._model_timing_text(row).startswith("waiting")

    def test_an_abort_retires_it_with_its_run_time(self, tmp_path, monkeypatch):
        _row, reading = self._retry_row(tmp_path, monkeypatch)
        assert progress_tracker.retire_live_entries(tmp_path, now=reading) == 1
        retired = status_file.read_status(str(tmp_path))["progress"]["models"][self.SPEC]
        assert retired["state"] == progress_tracker.STATE_FAILED
        assert (retired["elapsed_s"], retired["queued_s"]) == (10.0, 2.0)
