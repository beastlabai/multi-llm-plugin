"""``utils/progress_tracker.py``: the per-model map, the two tickers, the sweep.

Everything here runs against a synthetic phase dir and a driven clock — no host,
no provider, no real ``asyncio.sleep``. That is deliberate and it is what makes
these the gate rather than the container rows: a host row costs a real turn and
can only ever sample one run at one interval, while the properties this feature
is made of are all decidable in milliseconds.

Five of them are not about "does it print":

* **Cadence, not existence.** A single ``[PROGRESS]`` line proves nothing — the
  teardown flush alone produces one. The ticker is driven over a 30 s scripted
  fan-out at ``interval_s: 5`` and what is asserted is the *count* and the
  *spacing*.
* **Lifetime.** After the fan-out ends, by either ending, the ticker must be
  finished: a line or a ``models`` write arriving afterwards is a leaked task,
  not a late tick, and a write landing after the sweep resurrects entries the
  sweep had just retired.
* **Every ending retires every slot**, with one of exactly two wordings.
* **Elapsed is computed, not stored.** A live row's elapsed advances with the
  clock while the file on disk does not change at all; a terminal row's is
  frozen and stays frozen.
* **The ending that runs no Python is covered too.** ``--abort`` reaches neither
  sweep — ``os._exit`` executes no ``finally`` — so the close-out that *does*
  run (``RunHandle.finish``, and the aborting process itself when the worker
  took the SIGKILL) retires the map on disk. Driven through the real handle,
  because a `task.cancel()` standing in for an abort is how this went untested
  in the first place (F1, 2026-09-20).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import detach
from utils import progress_tracker
from utils import status_file
from utils.progress_tracker import (
    CANCELLED_ERROR,
    DEADLINE_ERROR,
    PROGRESS_PREFIX,
    STATE_DONE,
    STATE_FAILED,
    STATE_QUEUED,
    STATE_RETRYING,
    STATE_RUNNING,
    STATE_STAGGERING,
    TERMINAL_STATES,
    FanoutProgress,
    PhaseProgress,
    entry_elapsed_s,
    entry_stagger_remaining_s,
    format_budget,
    format_elapsed,
    render_roster_line,
)

SPECS = ["codex:gpt-5.6-sol", "opencode:auto", "gemini:pro"]

#: Everything a slot may read while it is still alive — the module's own list,
#: not a second copy, so a state added there cannot go unasserted here.
LIVE_STATES = set(progress_tracker.LIVE_STATES)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock the *ticker* advances, which is what makes this exact.

    The ticker's only wait is ``progress_tracker._tick_sleep``; patched to
    :meth:`sleep`, every interval it would have spent moves the clock by exactly
    that interval and yields to the loop. A scripted fan-out that runs "for 30 s"
    is then a loop that yields until the clock says 30 — no real time passes, and
    the number of ticks is a property of the script rather than of the machine
    the suite runs on.

    It carries a **wall** clock beside the monotonic one, locked to it, because
    the module needs both and they answer different questions: ``_monotonic``
    freezes ``elapsed_s`` at a terminal transition, while a *live* row's elapsed
    is ``now - started_at`` on the wall clock. Left unpatched, the wall half is
    the machine's, so a rendered-line assertion like ``(0m00s/30m)`` passes only
    while the test body runs inside one second — which is a flake, not a test
    (F6, 2026-09-20). Patched, the fan-out's own script decides what every clock
    in the module reads.
    """

    def __init__(self, start: float = 10_000.0, wall: float = 1_789_000_000.0) -> None:
        self.now = float(start)
        self._origin = float(start)
        self._wall_origin = float(wall)

    def monotonic(self) -> float:
        return self.now

    def wall_time(self) -> float:
        return self._wall_origin + (self.now - self._origin)

    def now_iso(self) -> str:
        """What ``_now_iso`` writes into ``started_at``, on this clock."""
        from datetime import datetime

        return datetime.fromtimestamp(self.wall_time()).isoformat()

    async def sleep(self, delay: float) -> None:
        self.now += float(delay)
        await asyncio.sleep(0)


class Emissions:
    """Every ``[PROGRESS]`` line, with the clock reading it was printed at."""

    #: The roster's own words. Two different lines share the ``[PROGRESS]``
    #: prefix — the ticker's, and the one-per-fan-out roster ``seed()`` emits —
    #: and cadence and lifetime are properties of the *ticker* alone, so those
    #: assertions read :attr:`ticks` and a roster assertion reads
    #: :attr:`rosters`. Filtering here rather than widening a tolerance keeps
    #: "one tick per interval, no more and no fewer" exact.
    ROSTER_MARK = "fan-out over"

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.rows = []

    def __call__(self, line: str) -> None:
        self.rows.append((self._clock.now, line))

    @property
    def lines(self):
        return [line for _at, line in self.rows]

    @property
    def times(self):
        return [at for at, _line in self.rows]

    @property
    def ticks(self):
        """``(at, line)`` for the lines a *ticker* produced."""
        return [row for row in self.rows if self.ROSTER_MARK not in row[1]]

    @property
    def rosters(self):
        """The roster lines, in the order they were printed."""
        return [line for _at, line in self.rows if self.ROSTER_MARK in line]


@pytest.fixture
def clock(monkeypatch):
    """Every clock the module reads, driven by the test rather than the machine.

    All four move together on purpose: a stamp written by ``_now_iso`` is read
    back by ``_wall_time`` one line later (that is how a live row's elapsed is
    computed), so patching one and not the other would compare the fake clock
    against the real one and make every elapsed assertion a race.
    """
    fake = FakeClock()
    monkeypatch.setattr(progress_tracker, "_monotonic", fake.monotonic)
    monkeypatch.setattr(progress_tracker, "_wall_time", fake.wall_time)
    monkeypatch.setattr(progress_tracker, "_now_iso", fake.now_iso)
    monkeypatch.setattr(progress_tracker, "_tick_sleep", fake.sleep)
    return fake


@pytest.fixture
def lines(monkeypatch, clock):
    recorder = Emissions(clock)
    monkeypatch.setattr(progress_tracker, "_emit", recorder)
    return recorder


def read_progress(phase_dir) -> dict:
    """The checkpoint's reserved ``progress`` object, as a reader sees it."""
    status = json.loads(Path(phase_dir, ".status.json").read_text(encoding="utf-8"))
    return status.get("progress", {})


async def pump(turns: int = 20) -> None:
    """Give any still-live task *turns* chances to run. Nothing should."""
    for _ in range(turns):
        await asyncio.sleep(0)


async def run_until(clock: FakeClock, seconds: float, *, max_turns: int = 10_000) -> None:
    """Yield until the ticker has moved the clock *seconds* forward.

    Bounded so a ticker that never got scheduled fails the test instead of
    hanging the suite.
    """
    deadline = clock.now + seconds
    for _ in range(max_turns):
        if clock.now >= deadline:
            return
        await asyncio.sleep(0)
    raise AssertionError("the ticker never advanced the clock - it never ran")


# ---------------------------------------------------------------------------
# The map and its lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Every transition in the plan's table, and the frozen terminal state."""

    def test_the_map_advances_through_every_transition(self, tmp_path, clock):
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()

        seeded = read_progress(tmp_path)["models"]
        assert set(seeded) == set(SPECS), "the map is complete before any task exists"
        assert [e["state"] for e in seeded.values()] == [STATE_QUEUED] * 3
        assert all(e["timeout_s"] == 1800 for e in seeded.values())
        assert all(e["attempt"] == 0 for e in seeded.values())
        assert all(e["source"] is None for e in seeded.values())
        assert all("ended_at" not in e and "elapsed_s" not in e for e in seeded.values())
        assert all("error" not in e for e in seeded.values())

        spec = SPECS[0]
        tracker.staggering(spec, delay_s=4.0)
        entry = read_progress(tmp_path)["models"][spec]
        assert entry["state"] == STATE_STAGGERING
        # The row says *when* it dispatches, not only that it is waiting: the
        # stamp is the sleep's end on the same wall clock as `started_at`.
        assert entry_stagger_remaining_s(entry, now=clock.wall_time()) == pytest.approx(4.0)

        # The stagger sleep ends before the semaphores are acquired, so the
        # slot goes back to `queued` — admitted, not yet dispatched — for the
        # whole `max_parallel` wait rather than reading `staggering` through it.
        tracker.queued(spec)
        entry = read_progress(tmp_path)["models"][spec]
        assert entry["state"] == STATE_QUEUED
        assert "stagger_until" not in entry, "a spent countdown is not carried"

        tracker.running(spec, attempt=1)
        entry = read_progress(tmp_path)["models"][spec]
        assert entry["state"] == STATE_RUNNING and entry["attempt"] == 1
        assert "retry_after_s" not in entry

        clock.now += 20
        tracker.retrying(spec, attempt=1, retry_after_s=20)
        entry = read_progress(tmp_path)["models"][spec]
        assert entry["state"] == STATE_RETRYING and entry["retry_after_s"] == 20

        tracker.running(spec, attempt=2)
        entry = read_progress(tmp_path)["models"][spec]
        assert entry["state"] == STATE_RUNNING and entry["attempt"] == 2
        assert "retry_after_s" not in entry, "a running row advertises no backoff"

        clock.now += 100
        tracker.done(spec, source="file")
        entry = read_progress(tmp_path)["models"][spec]
        assert entry["state"] == STATE_DONE
        assert entry["source"] == "file"
        assert entry["ended_at"] and entry["elapsed_s"] == 120
        assert "error" not in entry, "`error` is absent on every done entry"
        assert entry["attempt"] == 2, "the ladder's counter rides out with the row"

        tracker.running(SPECS[1], attempt=1)
        tracker.failed(SPECS[1], error="exit 1: boom", source="failed")
        entry = read_progress(tmp_path)["models"][SPECS[1]]
        assert entry["state"] == STATE_FAILED
        assert entry["error"] == "exit 1: boom" and entry["source"] == "failed"
        assert entry["ended_at"] and "retry_after_s" not in entry

        # queued -> done directly: the resume skip answers from disk, no attempt.
        tracker.done(SPECS[2], source="existing_file")
        assert read_progress(tmp_path)["models"][SPECS[2]]["attempt"] == 0

    def test_queued_is_the_semaphore_wait_and_not_a_terminal_row(
        self, tmp_path, clock
    ):
        """The two guards on the transition the semaphore wait needs.

        It must reach a live row that has already staggered (that *is* the
        wait it was added for), and it must bounce off a terminal one exactly
        as every other non-terminal transition does — a resumed slot seeded
        `done` must not be dragged back to `queued` by a lane that transitions
        unconditionally.
        """
        tracker = FanoutProgress(tmp_path, SPECS[:2], interval_s=0, timeout=1800)
        tracker.seed()
        live, resumed = SPECS[0], SPECS[1]
        tracker.staggering(live)
        tracker.done(resumed, source="existing_file")
        frozen = dict(read_progress(tmp_path)["models"][resumed])

        tracker.queued(live)
        tracker.queued(resumed)

        models = read_progress(tmp_path)["models"]
        assert models[live]["state"] == STATE_QUEUED
        # Nothing about the wait is an ending, so the row still has none.
        assert "ended_at" not in models[live] and "error" not in models[live]
        assert models[resumed] == frozen, "a terminal row is frozen both ways"

    def test_a_terminal_entry_is_frozen_in_both_directions(self, tmp_path, clock):
        tracker = FanoutProgress(tmp_path, SPECS[:1], interval_s=0, timeout=1800)
        tracker.seed()
        spec = SPECS[0]
        tracker.running(spec, attempt=1)
        clock.now += 42
        tracker.done(spec, source="stdout")
        frozen = dict(read_progress(tmp_path)["models"][spec])

        # The sweep, F11's backfill and a second lane-level transition all aim at
        # the same entry; whoever recorded the real ending keeps it.
        tracker.failed(spec, error=DEADLINE_ERROR)
        tracker.set_sweep_reason(DEADLINE_ERROR)
        tracker.sweep()
        tracker.running(spec, attempt=3)
        clock.now += 600
        assert read_progress(tmp_path)["models"][spec] == frozen

    def test_record_result_carries_the_lane_s_own_source_and_error(self, tmp_path):
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=900)
        tracker.seed()
        tracker.record_result(SPECS[0], {"success": True, "source": "file_extracted"})
        tracker.record_result(
            SPECS[1], {"success": False, "error": "rate limited", "source": "failed"}
        )
        tracker.record_result(SPECS[2], None)

        models = read_progress(tmp_path)["models"]
        assert models[SPECS[0]]["state"] == STATE_DONE
        assert models[SPECS[0]]["source"] == "file_extracted"
        assert models[SPECS[1]]["error"] == "rate limited"
        assert models[SPECS[2]]["state"] == STATE_FAILED, "a lane that returned nothing still retires its slot"

    def test_a_resumed_slot_counts_on_both_numbers(self, tmp_path, clock):
        """D4a on the one lane where nothing dispatches at all."""
        tracker = FanoutProgress(tmp_path, SPECS + ["cursor-agent:auto"], interval_s=0,
                                 timeout=1800)
        tracker.seed()
        for spec in SPECS:
            tracker.done(spec, source="existing_file")

        progress = read_progress(tmp_path)
        assert (progress["models_done"], progress["models_succeeded"]) == (3, 3)
        assert progress["models_total"] == 4
        assert tracker.counts() == (3, 3, 4)
        skipped = progress["models"][SPECS[0]]
        assert skipped["state"] == STATE_DONE, "there is no separate `skipped` state"
        assert skipped["source"] == "existing_file", "the distinction lives in `source`"
        assert skipped["attempt"] == 0 and skipped["elapsed_s"] == 0

    def test_models_done_counts_finished_slots_and_never_walks_backwards(
        self, tmp_path
    ):
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        seen = [read_progress(tmp_path)["models_done"]]
        tracker.done(SPECS[0], source="file")
        seen.append(read_progress(tmp_path)["models_done"])
        tracker.failed(SPECS[1], error="boom", source="failed")
        seen.append(read_progress(tmp_path)["models_done"])
        tracker.done(SPECS[2], source="stdout")
        seen.append(read_progress(tmp_path)["models_done"])

        assert seen == sorted(seen), "a live count that walks backwards is unreadable"
        assert seen[-1] == read_progress(tmp_path)["models_total"] == 3
        # ...and the losses survive as the second number rather than as a
        # shrinking numerator.
        assert read_progress(tmp_path)["models_succeeded"] == 2

    def test_the_seed_fills_a_budget_from_the_provider_when_none_is_given(
        self, tmp_path
    ):
        """A queued row renders `0m00s/30m`, not a blank denominator."""
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0)
        tracker.seed()
        for entry in read_progress(tmp_path)["models"].values():
            assert isinstance(entry["timeout_s"], (int, float))
            assert entry["timeout_s"] > 0

    def test_the_cadence_reaches_the_file_as_a_round_number(self, tmp_path):
        """`_seconds` exists to keep `1800.0` out of a file a human reads.

        The cadence was written raw, so a `30` in `providers.yaml` landed as
        `30.0` beside a `timeout_s: 1800` that had gone through `_seconds`
        (F5, 2026-09-20).
        """
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=30, timeout=1800)
        tracker.seed()
        progress = read_progress(tmp_path)
        assert progress["interval_s"] == 30
        assert isinstance(progress["interval_s"], int)
        assert "30.0" not in json.dumps(progress)

    def test_a_map_write_keeps_the_progress_siblings_other_writers_own(
        self, tmp_path
    ):
        """The merge is one level deep: `models` is whole, its siblings survive."""
        status_file.update_progress(str(tmp_path), {
            "step": "models_running",
            "sandbox_bypassed": ["codex:gpt-5.6-sol"],
            "subagents": {"kind": "validation", "total": 8, "ok": 5},
        })
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.done(SPECS[0], source="file")

        progress = read_progress(tmp_path)
        assert progress["step"] == "models_running"
        assert progress["sandbox_bypassed"] == ["codex:gpt-5.6-sol"]
        assert progress["subagents"]["ok"] == 5
        assert set(progress["models"]) == set(SPECS), (
            "the map is written whole - a one-entry patch would drop the rest"
        )


# ---------------------------------------------------------------------------
# Elapsed: computed at read time, stored once
# ---------------------------------------------------------------------------


class TestElapsedIsComputed:
    def test_a_live_row_advances_with_the_clock_without_any_write(self, tmp_path):
        tracker = FanoutProgress(tmp_path, SPECS[:1], interval_s=0, timeout=1800)
        tracker.seed()
        tracker.running(SPECS[0], attempt=1)
        entry = read_progress(tmp_path)["models"][SPECS[0]]
        on_disk = Path(tmp_path, ".status.json").read_bytes()

        origin = entry_elapsed_s(entry, now=_posix_of(entry["started_at"]))
        later = entry_elapsed_s(entry, now=_posix_of(entry["started_at"]) + 130)
        assert origin == pytest.approx(0.0, abs=1.0)
        assert later == pytest.approx(130.0, abs=1.0)
        assert Path(tmp_path, ".status.json").read_bytes() == on_disk, (
            "nothing wrote: the number came from `started_at`, not from a cache"
        )

    def test_a_live_row_is_clamped_at_its_whole_budget(self):
        entry = {"state": STATE_RUNNING, "started_at": "2026-09-19T10:00:00",
                 "timeout_s": 60}
        at = _posix_of("2026-09-19T10:00:00")
        assert entry_elapsed_s(entry, now=at + 3600) == 60.0

    def test_an_aware_started_at_renders_instead_of_crashing_the_table(self):
        """A hand-edited or foreign stamp is read, or declined — never raised.

        Our writer is `status_file.now_iso` (naive), so an offset-bearing stamp
        means a hand-edited or foreign checkpoint — which `read_status`'s own
        contract says must not make `status` crash. `datetime.fromisoformat`
        returns an **aware** datetime for it, and `datetime.now() - <aware>`
        raises `TypeError`, which `detach._elapsed_since` (this function's one
        ISO reader) used to let straight out, having caught only `ValueError`:
        one such row anywhere in the map took the whole command down (F4,
        2026-09-20).
        """
        from datetime import datetime, timezone

        when = datetime(2026, 9, 19, 10, 14, 3, tzinfo=timezone.utc)
        entry = {"state": STATE_RUNNING, "started_at": when.isoformat(),
                 "timeout_s": 1800}
        assert entry_elapsed_s(entry, now=when.timestamp() + 90) == 90.0

        # The `Z` spelling is the plan's own schema example and what the JS
        # tests emit. 3.11+ parses it (90.0); 3.10 declines it and the reader
        # answers `None`. Both are fine; an exception is the one outcome that is
        # not, which is the whole point of pinning it here.
        z_form = dict(entry, started_at="2026-09-19T10:14:03Z")
        assert entry_elapsed_s(z_form, now=when.timestamp() + 90) in (90.0, None)

    def test_a_terminal_row_is_frozen_whatever_the_clock_says(self, tmp_path, clock):
        tracker = FanoutProgress(tmp_path, SPECS[:1], interval_s=0, timeout=1800)
        tracker.seed()
        clock.now += 130
        tracker.done(SPECS[0], source="file")
        entry = read_progress(tmp_path)["models"][SPECS[0]]
        assert entry["elapsed_s"] == 130
        # An hour later the table still reads 130 s, because `ended_at` is what
        # picks the formula.
        assert entry_elapsed_s(entry, now=_posix_of(entry["started_at"]) + 3600) == 130.0

    def test_elapsed_is_not_reset_by_a_retry(self, tmp_path, clock):
        """The backoff is charged to the same deadline, so the origin holds."""
        tracker = FanoutProgress(tmp_path, SPECS[:1], interval_s=0, timeout=1800)
        tracker.seed()
        tracker.running(SPECS[0], attempt=1)
        clock.now += 60
        tracker.retrying(SPECS[0], attempt=1, retry_after_s=20)
        clock.now += 20
        tracker.running(SPECS[0], attempt=2)
        clock.now += 40
        tracker.done(SPECS[0], source="file")
        assert read_progress(tmp_path)["models"][SPECS[0]]["elapsed_s"] == 120


def _posix_of(stamp: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(stamp).timestamp()


# ---------------------------------------------------------------------------
# The rendered line
# ---------------------------------------------------------------------------


class TestStaggerCountdown:
    """2026-09-21: `staggering` said the row was waiting and not for how long.

    A fan-out launches slot N `N * PROVIDER_STAGGER_DELAY` seconds in, and the
    launch-time `status` call — made at second zero on purpose, for the roster
    — met every row but the first reading `staggering`, which a user read as
    the CLIs being run one at a time. The word was true; it named no duration.
    `staggering(spec, delay_s=...)` now stamps `stagger_until`, and
    `entry_stagger_remaining_s` reads it back as the countdown the table
    renders as `starts in 4s`.
    """

    def test_the_stamp_is_the_sleep_end_on_the_checkpoints_own_clock(self, tmp_path, clock):
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.staggering(SPECS[1], delay_s=2.5)
        entry = read_progress(tmp_path)["models"][SPECS[1]]
        assert _posix_of(entry["stagger_until"]) == pytest.approx(clock.wall_time() + 2.5)
        # Read a second later, it counts down rather than re-reading the file.
        assert entry_stagger_remaining_s(entry, now=clock.wall_time() + 1.0) == pytest.approx(1.5)

    def test_a_caller_with_no_delay_to_name_writes_no_stamp(self, tmp_path, clock):
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.staggering(SPECS[0])
        entry = read_progress(tmp_path)["models"][SPECS[0]]
        assert entry["state"] == STATE_STAGGERING
        assert "stagger_until" not in entry
        assert entry_stagger_remaining_s(entry, now=clock.wall_time()) is None

    @pytest.mark.parametrize(
        "delay", [None, True, float("nan"), float("inf"), -1.0, "4"],
        ids=["none", "bool", "nan", "inf", "negative", "string"],
    )
    def test_a_delay_the_fanout_never_slept_is_not_stamped(self, tmp_path, clock, delay):
        """The same shapes `entry_budget` refuses, refused at the writer: a
        countdown built from a bool, NaN, an infinity or a negative is a number
        no sleep ever took."""
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.staggering(SPECS[0], delay_s=delay)
        assert "stagger_until" not in read_progress(tmp_path)["models"][SPECS[0]]

    def test_a_stamp_already_past_is_no_countdown(self, tmp_path, clock):
        """A poll that lands on the flip to `queued` renders no clause rather
        than `starts in 0s` — and a run that died mid-stagger does not advertise
        a dispatch that will never come."""
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.staggering(SPECS[0], delay_s=2.0)
        entry = read_progress(tmp_path)["models"][SPECS[0]]
        assert entry_stagger_remaining_s(entry, now=clock.wall_time() + 2.0) is None
        assert entry_stagger_remaining_s(entry, now=clock.wall_time() + 60.0) is None

    def test_only_a_staggering_row_counts_down(self):
        """A stale stamp on a row that moved on is not a countdown: the state
        word gates the read, so a hand-edited or racily written row cannot
        render a `starts in` clause beside `running`."""
        stamp = "2026-09-21T10:00:10"
        at = _posix_of("2026-09-21T10:00:00")
        assert entry_stagger_remaining_s(
            {"state": STATE_STAGGERING, "stagger_until": stamp}, now=at
        ) == pytest.approx(10.0)
        for state in (STATE_QUEUED, STATE_RUNNING, STATE_RETRYING, STATE_DONE):
            assert entry_stagger_remaining_s(
                {"state": state, "stagger_until": stamp}, now=at
            ) is None

    @pytest.mark.parametrize(
        "stamp", [None, "", "not a stamp", 42, {"at": 1}, "2026-13-45T99:00:00"],
        ids=["absent", "empty", "garbage", "int", "dict", "unparsable"],
    )
    def test_a_stamp_that_cannot_be_read_costs_the_clause_and_never_raises(self, stamp):
        assert entry_stagger_remaining_s(
            {"state": STATE_STAGGERING, "stagger_until": stamp}
        ) is None
        assert entry_stagger_remaining_s("not a row") is None

    def test_an_aware_stamp_reads_through_the_same_normalisation(self):
        """The one ISO reader (F4): a `Z` stamp is converted to this machine's
        local clock, so a foreign file counts down rather than raising."""
        from datetime import datetime, timezone

        when = datetime(2026, 9, 21, 10, 0, 10, tzinfo=timezone.utc)
        at = when.astimezone().replace(tzinfo=None).timestamp() - 3.0
        assert entry_stagger_remaining_s(
            {"state": STATE_STAGGERING, "stagger_until": "2026-09-21T10:00:10Z"},
            now=at,
        ) == pytest.approx(3.0)

    def test_dispatch_drops_the_stamp(self, tmp_path, clock):
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.staggering(SPECS[0], delay_s=2.0)
        tracker.running(SPECS[0], attempt=1)
        assert "stagger_until" not in read_progress(tmp_path)["models"][SPECS[0]]


class TestRenderedLine:
    def test_the_line_names_live_rows_and_counts_the_rest(self, tmp_path, clock):
        tracker = FanoutProgress(tmp_path, SPECS + ["cursor-agent:auto"],
                                 interval_s=0, timeout=1800)
        tracker.seed()
        tracker.running(SPECS[0], attempt=1)
        tracker.done(SPECS[1], source="file")
        tracker.staggering(SPECS[2])
        # Two minutes ten of the live slot's budget, on the driven clock. A live
        # row's elapsed is `now - started_at` on the **wall** clock, so this
        # assertion is only a real one with that seam patched: against the
        # machine's clock it read `(0m00s/30m)` and passed exactly as long as the
        # body ran inside one second (F6, 2026-09-20).
        clock.now += 130
        line = tracker.render_line()

        assert line.startswith(f"{PROGRESS_PREFIX} 1/4 done (1 ok, 0 failed)")
        assert f"running: {SPECS[0]} (2m10s/30m)" in line
        assert "staggering 1" in line and "queued 1" in line

    def test_the_attempt_has_no_denominator(self, tmp_path):
        tracker = FanoutProgress(tmp_path, SPECS[:1], interval_s=0, timeout=1800)
        tracker.seed()
        tracker.running(SPECS[0], attempt=2)
        line = tracker.render_line()
        assert "attempt 2" in line and "2/3" not in line

    def test_the_line_is_ascii_even_when_a_spec_is_not(self, tmp_path, clock):
        spec = "codex:gpt‑5"          # U+2011, out of a config file
        tracker = FanoutProgress(tmp_path, [spec], interval_s=0, timeout=1800)
        tracker.seed()
        tracker.running(spec, attempt=1)   # a *named* row, so the spec is in the line
        line = tracker.render_line()
        assert line.isascii(), "a redirected stream takes the locale codec (cp1252)"
        assert "·" not in line and "–" not in line
        # The scrub is `errors="replace"`, so the exotic byte degrades to `?` and
        # the row survives *named* rather than dropped or mangled...
        assert "running: codex:gpt?5 (0m00s/30m)" in line
        # ...and the separators are still there to read the line by. (This used
        # to assert `"|" in line or line.count(",") >= 1`, which is true of
        # essentially every line this function can produce — F6, 2026-09-20.)
        assert " | " in line

    @pytest.mark.parametrize("seconds,rendered", [
        (0, "0m00s"), (40, "0m40s"), (130, "2m10s"), (3850, "1h04m10s"),
    ])
    def test_format_elapsed(self, seconds, rendered):
        assert format_elapsed(seconds) == rendered

    @pytest.mark.parametrize("seconds,rendered", [
        (1800, "30m"), (1200, "20m"), (45, "45s"), (3600, "1h"), (5400, "1h30m"),
    ])
    def test_format_budget(self, seconds, rendered):
        assert format_budget(seconds) == rendered

# ---------------------------------------------------------------------------
# The roster: what the fan-out is over, said once at launch
# ---------------------------------------------------------------------------


class TestTheRosterAtLaunch:
    """One line, at ``seed()``, naming the CLI and the model of every slot.

    The gap it closes was watched on a real run: the log opened at whatever the
    first interval happened to catch (``models 1/5 done``) and nothing on
    screen, ever, said which five models the fan-out was over. A model spec
    already *is* the CLI and the model, so the roster derives nothing — it
    states the seeded map, in seed order, with each slot's whole budget.

    It is not a progress line, so none of the ticker's properties apply to it:
    it is emitted once, from the one method that runs once, and the cadence
    assertions above read :attr:`Emissions.ticks` so that stays exact.
    """

    #: The contract's own example, and a deliberately mixed one: two providers
    #: whose budget resolves and one whose does not.
    SPECS = [
        "codex:gpt-6-astra:xhigh",
        "pi:openrouter/deepseek/deepseek-v4-flash-0731",
        "opencode:auto",
    ]

    EXPECTED = (
        "[PROGRESS] fan-out over 3 models: "
        "codex:gpt-6-astra:xhigh (30m), "
        "pi:openrouter/deepseek/deepseek-v4-flash-0731 (20m), "
        "opencode:auto (?)"
    )

    @staticmethod
    def _budgets(monkeypatch, table):
        """Resolve a provider's budget from *table*; anything else raises.

        Hermetic on purpose — `providers.yaml` really does give codex 1800 and
        pi 1200 today, so the roster's example renders against the live config,
        but a test that asserts a rendered budget against an operator-editable
        file is asserting the file. The raise is `_budget_for`'s documented
        failure mode (a config it cannot read), which is the only way the key
        becomes `None` and the only way `(?)` is ever reached.
        """
        from utils import provider_registry

        def resolve(provider, anchor=None):
            if provider not in table:
                raise RuntimeError(f"no timeout configured for {provider}")
            return table[provider]

        monkeypatch.setattr(provider_registry, "get_provider_timeout", resolve)

    def test_it_names_every_spec_with_its_budget_in_seed_order(
        self, tmp_path, lines, monkeypatch
    ):
        self._budgets(monkeypatch, {"codex": 1800, "pi": 1200})
        tracker = FanoutProgress(tmp_path, self.SPECS, interval_s=30)
        tracker.seed()

        assert lines.lines == [self.EXPECTED]

    def test_the_roster_is_seed_order_and_not_the_map_s_luck(
        self, tmp_path, lines, monkeypatch
    ):
        """Dispatch order is the order the user typed; reversing it is a lie."""
        self._budgets(monkeypatch, {"codex": 1800, "pi": 1200, "opencode": 900})
        tracker = FanoutProgress(tmp_path, list(reversed(self.SPECS)), interval_s=30)
        tracker.seed()

        named = [part.split(" (")[0] for part in lines.rosters[0].split(": ", 1)[1].split(", ")]
        assert named == list(reversed(self.SPECS))

    def test_interval_zero_emits_nothing_at_all(self, tmp_path, lines, monkeypatch):
        """`progress.interval_s: 0` is "print no [PROGRESS] lines", not "no ticker".

        The roster is one of those lines, so the switch has to reach it too —
        and a reader loses nothing, because every entry it would have named is
        in the checkpoint the same `seed()` just wrote.
        """
        self._budgets(monkeypatch, {"codex": 1800, "pi": 1200})
        tracker = FanoutProgress(tmp_path, self.SPECS, interval_s=0)
        tracker.seed()

        assert lines.rows == []
        assert set(read_progress(tmp_path)["models"]) == set(self.SPECS)

    def test_an_unresolvable_budget_renders_a_question_mark(
        self, tmp_path, lines, monkeypatch
    ):
        """`_budget_for` leaves the key `None` rather than failing the fan-out."""
        self._budgets(monkeypatch, {})
        tracker = FanoutProgress(tmp_path, self.SPECS, interval_s=30)
        tracker.seed()

        assert lines.lines == [
            "[PROGRESS] fan-out over 3 models: "
            "codex:gpt-6-astra:xhigh (?), "
            "pi:openrouter/deepseek/deepseek-v4-flash-0731 (?), "
            "opencode:auto (?)"
        ]
        assert all(e["timeout_s"] is None for e in read_progress(tmp_path)["models"].values())

    def test_an_entry_with_no_budget_key_at_all_renders_too(self, tmp_path, clock):
        """The docstring says "absent **or** None", so the renderer reads both.

        A `_seed_one` entry always carries the key, but `retire_live_entries`
        copies a hand-edited or foreign row through untouched and the map is
        one object several writers reach.
        """
        tracker = FanoutProgress(tmp_path, ["codex:gpt-5.6-sol"], interval_s=30,
                                 timeout=1800)
        tracker.seed()
        tracker._models["codex:gpt-5.6-sol"].pop("timeout_s")

        assert tracker.render_roster() == (
            "[PROGRESS] fan-out over 1 model: codex:gpt-5.6-sol (?)"
        )

    def test_one_model_reads_one_model(self, tmp_path, lines):
        tracker = FanoutProgress(tmp_path, ["codex:gpt-5.6-sol"], interval_s=30,
                                 timeout=1800)
        tracker.seed()

        assert lines.lines == [
            "[PROGRESS] fan-out over 1 model: codex:gpt-5.6-sol (30m)"
        ]

    def test_an_empty_spec_list_says_nothing(self, tmp_path, lines):
        """Not "fan-out over 0 models:" — a line with nothing after its colon."""
        tracker = FanoutProgress(tmp_path, [], interval_s=30, timeout=1800)
        tracker.seed()

        assert lines.rows == []
        assert tracker.render_roster() == ""

    def test_a_non_ascii_spec_is_scrubbed_not_crashed(self, tmp_path, lines):
        """Same backstop as the ticker's: the stream takes the locale codec."""
        spec = "codex:gpt‑5"          # U+2011, out of a config file
        tracker = FanoutProgress(tmp_path, [spec], interval_s=30, timeout=1800)
        tracker.seed()

        assert lines.lines == [
            "[PROGRESS] fan-out over 1 model: codex:gpt?5 (30m)"
        ]
        assert lines.lines[0].isascii()

    def test_a_stream_that_refuses_the_line_does_not_lose_the_fan_out(
        self, tmp_path, monkeypatch
    ):
        """The print is inside the guard, not beside it.

        `seed()` runs *before* the tasks are created, so this is the one
        narration site where an exception costs the whole run rather than one
        line — and the likeliest raiser is the print itself, not the renderer:
        a closed stdout or a codec the scrub did not save raises at `_emit`.
        The guard covered `render_roster()` only until 2026-09-20, which left
        the docstring promising protection the code did not give. `_tick` has
        always wrapped both.
        """
        def _explode(_line):
            raise OSError("stdout is closed")

        monkeypatch.setattr(progress_tracker, "_emit", _explode)
        tracker = FanoutProgress(
            tmp_path, ["codex:gpt-5.6-sol"], interval_s=30, timeout=1800
        )

        tracker.seed()          # the assertion: it returns at all

        assert tracker.snapshot()["codex:gpt-5.6-sol"]["state"] == "queued"
        written = read_progress(tmp_path)["models"]
        assert written["codex:gpt-5.6-sol"]["timeout_s"] == 1800

    def test_the_map_on_disk_renders_the_same_roster(self, tmp_path, lines):
        """One spelling, two processes.

        The fan-out emits this line from its in-memory map; ``status`` renders
        it for the OpenCode package route out of ``progress.models`` read back
        off ``.status.json``, because that route has no agent in its poll loop
        and its plugin may compose no sentence of its own. Those are two
        processes and the same sentence, so the renderer is one module function
        both reach — asserted here against the bytes the tracker itself printed,
        which is the only way a second spelling could not creep back in.
        """
        tracker = FanoutProgress(
            tmp_path,
            ["codex:gpt-5.6-sol", "opencode:auto"],
            interval_s=30,
            timeout=1800,
        )
        tracker.seed()

        on_disk = read_progress(tmp_path)["models"]
        # Through JSON, because that is how the map reaches the reader: key
        # order is the seed order the line is read in, and it survives the trip.
        round_tripped = json.loads(json.dumps(on_disk))

        assert render_roster_line(round_tripped) == tracker.render_roster()
        assert render_roster_line(round_tripped) == lines.rosters[0]

    def test_a_map_the_renderer_cannot_trust_still_renders(self):
        """`status` is the command a poll loop runs, so the line must answer.

        The on-disk map is not guaranteed to hold what the tracker wrote — a
        hand-edited checkpoint, or a row `retire_live_entries` copied through
        untouched — and `format_budget` takes `float()` to its argument. Each
        unreadable budget costs its own `(?)` and nothing else; a map that is
        not a map at all renders nothing rather than raising.
        """
        assert render_roster_line({
            "codex:gpt-5.6-sol": {"timeout_s": "half an hour"},
            "opencode:auto": "not an entry at all",
            "gemini:pro": {"timeout_s": 1200},
        }) == (
            "[PROGRESS] fan-out over 3 models: codex:gpt-5.6-sol (?), "
            "opencode:auto (?), gemini:pro (20m)"
        )
        assert render_roster_line(None) == ""
        assert render_roster_line({}) == ""

    def test_it_does_not_re_emit_on_any_later_transition(
        self, tmp_path, clock, lines
    ):
        """Once per fan-out. A roster repeated per poll is the poll loop's job
        to *not* do, and the line itself must not do it either."""
        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=5, timeout=1800)
            tracker.seed()
            tracker.staggering(SPECS[0])
            tracker.running(SPECS[0], attempt=1)
            tracker.retrying(SPECS[0], attempt=1, retry_after_s=10)
            tracker.running(SPECS[0], attempt=2)
            tracker.start_ticker()
            await run_until(clock, 15)
            # A lane transitioning a slot `seed()` never saw goes through
            # `_seed_one`, which is not `seed()` and prints nothing.
            tracker.running("cursor-agent:auto", attempt=1)
            for spec in SPECS:
                tracker.done(spec, source="file")
            await tracker.stop_ticker()
            tracker.sweep()

        asyncio.run(scenario())

        assert len(lines.rosters) == 1, lines.lines
        assert lines.lines[0] == lines.rosters[0], "and it is the log's first line"
        assert lines.ticks, "the ticker still ran"


# ---------------------------------------------------------------------------
# The ticker: cadence, silence, lifetime
# ---------------------------------------------------------------------------


class TestTickerCadence:
    def test_one_line_per_interval_over_a_run_that_outlives_several(
        self, tmp_path, clock, lines
    ):
        """Cadence, not existence: a single line proves nothing."""
        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=5, timeout=1800)
            tracker.seed()
            tracker.running(SPECS[0], attempt=1)
            tracker.start_ticker()
            try:
                await run_until(clock, 30)
                for spec in SPECS:
                    tracker.done(spec, source="file")
            finally:
                await tracker.stop_ticker()
                tracker.sweep()
            return tracker

        asyncio.run(scenario())

        expected = 30 / 5
        # The roster `seed()` emits is not a tick — see `Emissions.ROSTER_MARK`.
        ticks = lines.ticks
        assert expected - 1 <= len(ticks) <= expected + 1, lines.lines
        at = sorted({moment for moment, _line in ticks})
        assert len(at) == expected, "one tick per interval, no more and no fewer"
        spacing = {round(b - a, 6) for a, b in zip(at, at[1:])}
        assert spacing == {5.0}, f"uneven cadence: {at}"
        assert all(line.startswith(PROGRESS_PREFIX) for line in lines.lines)

    def test_the_ticker_writes_nothing_on_the_interval(self, tmp_path, clock, lines):
        """The one write a ticker makes is its teardown sweep."""
        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=5, timeout=1800)
            tracker.seed()
            for spec in SPECS:
                tracker.done(spec, source="file")
            tracker.start_ticker()
            settled = Path(tmp_path, ".status.json").read_bytes()
            await run_until(clock, 30)
            during = Path(tmp_path, ".status.json").read_bytes()
            await tracker.stop_ticker()
            return settled, during

        settled, during = asyncio.run(scenario())
        assert len(lines.rows) >= 5, "it did tick"
        assert settled == during, (
            "the checkpoint stays event-driven: elapsed is computed, not stored"
        )

    def test_interval_zero_is_silent_and_still_records_every_transition(
        self, tmp_path, clock, lines
    ):
        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
            tracker.seed()
            assert tracker.start_ticker() is None
            tracker.running(SPECS[0], attempt=1)
            tracker.done(SPECS[0], source="file")
            tracker.failed(SPECS[1], error="boom", source="failed")
            await pump(5)
            await tracker.stop_ticker()
            tracker.sweep()

        asyncio.run(scenario())
        assert lines.rows == [], "`interval_s: 0` costs the log its lines..."
        models = read_progress(tmp_path)["models"]
        assert models[SPECS[0]]["state"] == STATE_DONE
        assert models[SPECS[1]]["error"] == "boom"
        assert models[SPECS[2]]["state"] == STATE_FAILED
        assert read_progress(tmp_path)["models_done"] == 3, "...and a reader nothing"


class TestTickerLifetime:
    def test_nothing_arrives_after_a_normal_completion(self, tmp_path, clock, lines):
        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=5, timeout=1800)
            tracker.seed()
            tracker.start_ticker()
            await run_until(clock, 15)
            for spec in SPECS:
                tracker.done(spec, source="file")
            await tracker.stop_ticker()
            tracker.sweep()
            settled = (len(lines.rows), Path(tmp_path, ".status.json").read_bytes())
            await pump()
            return settled, (len(lines.rows), Path(tmp_path, ".status.json").read_bytes())

        settled, after = asyncio.run(scenario())
        assert settled == after, "a later line from that task is a leak, not a late tick"

    def test_nothing_arrives_after_a_cancellation(self, tmp_path, clock, lines):
        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=5, timeout=1800)
            tracker.seed()
            tracker.running(SPECS[0], attempt=1)
            tracker.start_ticker()
            try:
                await run_until(clock, 15)
                raise asyncio.TimeoutError
            except asyncio.TimeoutError:
                tracker.set_sweep_reason(DEADLINE_ERROR)
            finally:
                # The fixed order: cancel and await, then sweep, then return.
                await tracker.stop_ticker()
                tracker.sweep()
            settled = (len(lines.rows), Path(tmp_path, ".status.json").read_bytes())
            await pump()
            return settled, (len(lines.rows), Path(tmp_path, ".status.json").read_bytes())

        settled, after = asyncio.run(scenario())
        assert settled == after
        models = read_progress(tmp_path)["models"]
        assert all(e["state"] in TERMINAL_STATES for e in models.values())
        assert models[SPECS[0]]["error"] == DEADLINE_ERROR

    def test_the_ticker_is_cancelled_and_awaited_when_the_owner_raises(
        self, tmp_path, clock, lines
    ):
        leftovers = []

        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=5, timeout=1800)
            tracker.seed()
            tracker.running(SPECS[0], attempt=1)
            tracker.start_ticker()
            try:
                await run_until(clock, 10)
                raise RuntimeError("a slot raised out of the gather")
            finally:
                await tracker.stop_ticker()
                tracker.sweep()
                leftovers.extend(
                    t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                )
                leftovers.append(len(lines.rows))

        with pytest.raises(RuntimeError, match="raised out of the gather"):
            asyncio.run(scenario())

        at_sweep = leftovers.pop()
        assert leftovers == [], "the ticker task outlived the fan-out that owned it"
        assert len(lines.rows) == at_sweep, "no line landed after the sweep"
        assert all(
            e["state"] in TERMINAL_STATES
            for e in read_progress(tmp_path)["models"].values()
        )

    def test_the_tickers_own_finally_sweeps_the_cancel_all_path(
        self, tmp_path, clock, lines
    ):
        """What the ticker's own teardown covers is a cancellation aimed at *it*.

        Renamed and re-described 2026-09-20. It used to be called
        `..._sweeps_the_abort_path` and its docstring claimed "`--abort` /
        KeyboardInterrupt bypass the fan-out's `finally` entirely", while the
        body drives a plain `task.cancel()` — which is neither of those. It is
        **asyncio shutdown's cancel-all**: a second Ctrl-C, or a loop tearing
        down every task at once, reaching this task directly rather than through
        `stop_ticker`. A *first* Ctrl-C is not this case (3.11's
        `Runner._on_sigint` cancels the main task, so the fan-out's own
        `finally` runs and calls `stop_ticker`), and `--abort` is not this case
        either: it runs no Python here at all, which is what
        `TestAbortRetiresTheMapOnDisk` below covers.
        """
        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=5, timeout=1800)
            tracker.seed()
            tracker.running(SPECS[0], attempt=1)
            task = tracker.start_ticker()
            await run_until(clock, 10)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())
        models = read_progress(tmp_path)["models"]
        assert all(e["state"] == STATE_FAILED for e in models.values())
        assert {e["error"] for e in models.values()} == {CANCELLED_ERROR}

    def test_a_write_failure_never_reaches_the_gather(
        self, tmp_path, clock, lines, monkeypatch, capsys
    ):
        from utils import detach

        def explode(*_args, **_kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(detach, "report_progress", explode)

        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=5, timeout=1800)
            tracker.seed()
            tracker.start_ticker()
            await run_until(clock, 15)
            await tracker.stop_ticker()
            tracker.sweep()

        asyncio.run(scenario())  # must not raise
        warnings = [
            line for line in capsys.readouterr().err.splitlines()
            if "progress write failed" in line
        ]
        assert len(warnings) == 1, "said once, not once per transition"


class TestEveryEndingRetiresEverySlot:
    """After the fan-out returns, nothing reads queued/staggering/running/retrying."""

    def _run(self, tmp_path, clock, *, reason=None, raise_with=None, runtime=0.0):
        async def scenario():
            tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
            tracker.seed()
            tracker.staggering(SPECS[0])
            tracker.running(SPECS[1], attempt=1)
            tracker.retrying(SPECS[2], attempt=1, retry_after_s=20)
            try:
                # How long the slots were live before the ending arrived, so the
                # `elapsed_s` the sweep freezes has a value to be right about.
                clock.now += runtime
                if raise_with is not None:
                    raise raise_with
            finally:
                if reason is not None:
                    tracker.set_sweep_reason(reason)
                await tracker.stop_ticker()
                tracker.sweep()
            return tracker

        return scenario()

    def test_the_deadline_path_uses_the_wording_the_report_uses(self, tmp_path, clock):
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(self._run(tmp_path, clock, reason=DEADLINE_ERROR,
                                  raise_with=asyncio.TimeoutError(), runtime=95))
        models = read_progress(tmp_path)["models"]
        assert not LIVE_STATES & {e["state"] for e in models.values()}
        assert {e["error"] for e in models.values()} == {DEADLINE_ERROR}
        # Each retired row carries the time it really spent live, frozen from the
        # monotonic origin stamped at *seed* — which is what a table rendered an
        # hour later shows. (This used to assert `elapsed_s is not None`, which
        # `_freeze` cannot produce: it writes a number or `0` — F6, 2026-09-20.)
        assert {e["elapsed_s"] for e in models.values()} == {95}
        assert all(e["source"] is None for e in models.values())
        assert all("retry_after_s" not in e for e in models.values()), (
            "a slot that will never wake from its backoff must not advertise one"
        )

    def test_the_abort_path_uses_the_other_one(self, tmp_path, clock):
        asyncio.run(self._run(tmp_path, clock))
        models = read_progress(tmp_path)["models"]
        assert not LIVE_STATES & {e["state"] for e in models.values()}
        assert {e["error"] for e in models.values()} == {CANCELLED_ERROR}

    def test_the_sweep_and_f11s_backfill_agree_on_the_failed_set(self, tmp_path, clock):
        """Both touch only non-terminal entries, so the order cannot matter.

        Driven **both** ways round, because agreement is the claim and one order
        cannot show it. The previous version ran the sweep and then called
        `backfill_deadline_losses` without `tracker=`: the backfill never
        touched a map at all, the assertion was satisfied by the sweep that had
        already run, and deleting the `tracker` parameter from the function under
        test left this green (F6, 2026-09-20).

        The two maps are compared whole, and that is exact rather than
        approximate because every clock the module reads is the driven one: the
        stamps and `elapsed_s` of two runs of the same script are identical.
        """
        from utils.review_orchestrator_base import backfill_deadline_losses

        def fan_out(directory, *, sweep):
            directory.mkdir(parents=True, exist_ok=True)
            tracker = FanoutProgress(directory, SPECS, interval_s=0, timeout=1800)
            tracker.seed()
            tracker.done(SPECS[0], source="file")
            tracker.failed(SPECS[1], error="exit 1", source="failed")
            if sweep:
                tracker.set_sweep_reason(DEADLINE_ERROR)
                tracker.sweep()
            failed_models = {SPECS[1]: "exit 1"}
            backfill_deadline_losses(
                SPECS, {SPECS[0]: {"success": True}}, failed_models, tracker=tracker
            )
            return failed_models

        swept_first = tmp_path / "swept-then-backfilled"
        backfill_only = tmp_path / "backfilled-with-no-sweep"
        first_failed = fan_out(swept_first, sweep=True)
        second_failed = fan_out(backfill_only, sweep=False)

        first = read_progress(swept_first)["models"]
        second = read_progress(backfill_only)["models"]
        assert first == second, (
            "the order of the two passes is visible in the map - one of them is "
            "not writing what the other does"
        )
        assert first_failed == second_failed == {
            SPECS[1]: "exit 1", SPECS[2]: DEADLINE_ERROR,
        }
        assert {s for s, e in first.items() if e["state"] == STATE_FAILED} == set(
            first_failed
        ), "the table's failed set is the report's failed set"
        assert first[SPECS[2]]["error"] == DEADLINE_ERROR
        assert first[SPECS[0]]["state"] == STATE_DONE, "a real ending is never overwritten"

    def test_the_sweep_rewrites_nothing_when_every_slot_already_ended(self, tmp_path):
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        for spec in SPECS:
            tracker.done(spec, source="file")
        settled = Path(tmp_path, ".status.json").read_bytes()
        tracker.sweep()
        assert Path(tmp_path, ".status.json").read_bytes() == settled


# ---------------------------------------------------------------------------
# The ending that runs no Python: `--abort`
# ---------------------------------------------------------------------------


class TestAbortRetiresTheMapOnDisk:
    """`status --abort` reaches no sweep at all, so `finish()` closes the map.

    The signal handler records the terminal ``run`` record and calls
    ``os._exit(3)``: no ``finally`` runs, no task is cancelled, the ticker gets
    no teardown. Nothing in this file's ticker section covers that — the test
    that claimed to drove a plain ``task.cancel()``, which is the cancel-all
    path — and on disk the result was ``run.state: aborted`` beside rows still
    reading ``running``/``staggering``, ``models_done`` frozen mid-count, and a
    poll an hour later rendering ``codex:gpt-5.6-sol  running  30m00s/30m``
    under a run that had been over since (F1, 2026-09-20).
    """

    def _claimed(self, tmp_path, monkeypatch, run_id="run-abort-0001"):
        """A real `RunHandle` owning a real phase dir, as a worker holds one.

        The signal *wiring* is stubbed out — this drives `_on_signal` directly,
        and installing real SIGTERM/SIGINT handlers would leave them installed
        for the rest of the session. What the wiring does is covered by the real
        forked workers in `tests/test_detach.py`; what is covered here is what
        the handler leaves behind.
        """
        from utils import process_control

        monkeypatch.setattr(
            process_control, "install_worker_signal_handlers", lambda _cb: None
        )
        # `start()` sets the module-global `_ACTIVE`, and a handle left there
        # leaks into every later test in the session. Registering the *current*
        # value (None) first is what gets it put back at teardown, whatever the
        # test does with the handle afterwards.
        monkeypatch.setattr(detach, "_ACTIVE", None)
        plan = tmp_path / "a plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase = tmp_path / "a_plan" / "review-plan"
        phase.mkdir(parents=True)
        handle = detach.RunHandle(
            str(phase), run_id=run_id, mode="review-plan", plan_file=str(plan),
            detached=False,
        )
        handle.start()
        return handle, phase

    def _mid_fan_out(self, phase, clock):
        """One slot finished, one dispatched, one waiting — and 5 minutes in."""
        tracker = FanoutProgress(phase, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.done(SPECS[0], source="file")
        tracker.running(SPECS[1], attempt=2)
        tracker.staggering(SPECS[2])
        clock.now += 300
        return tracker

    def test_an_abort_retires_every_live_row_and_restates_the_counters(
        self, tmp_path, clock, monkeypatch
    ):
        import signal as _signal

        handle, phase = self._claimed(tmp_path, monkeypatch)
        self._mid_fan_out(phase, clock)
        assert read_progress(phase)["models_done"] == 1, "mid-run, as the poll saw it"

        exits = []
        monkeypatch.setattr(detach.os, "_exit", exits.append)
        handle._on_signal(_signal.SIGTERM)

        assert exits == [3], "the handler still exits the way `--abort` requires"
        run = status_file.read_run(str(phase))
        assert run["state"] == detach.RUN_STATE_ABORTED

        models = read_progress(phase)["models"]
        assert not LIVE_STATES & {e["state"] for e in models.values()}, (
            "a row still reading `running` under an aborted run is the bug"
        )
        # The slot that really finished keeps its own ending; the two the abort
        # took keep the one wording that path owns.
        assert models[SPECS[0]]["state"] == STATE_DONE
        assert models[SPECS[0]]["source"] == "file"
        assert "error" not in models[SPECS[0]]
        for spec in SPECS[1:]:
            entry = models[spec]
            assert entry["state"] == STATE_FAILED
            assert entry["error"] == CANCELLED_ERROR
            assert entry["source"] is None
            assert entry["ended_at"]
            # Derived on the wall clock from `started_at`, by the same read-time
            # rule every renderer applies, because the monotonic origins live in
            # a tracker a signal handler must not touch.
            assert entry["elapsed_s"] == 300

        progress = read_progress(phase)
        assert (progress["models_done"], progress["models_succeeded"]) == (3, 1), (
            "the counters were frozen wherever the last transition left them"
        )
        assert progress["models_total"] == 3

    def test_a_crashed_worker_gets_the_same_close_out(
        self, tmp_path, clock, monkeypatch
    ):
        """`failed` is the other non-normal ending, and it strands rows too.

        The `_atexit` backstop reaches `finish()` by another route (a `sys.exit`
        deep in a target), with the same fan-out left mid-flight.
        """
        handle, phase = self._claimed(tmp_path, monkeypatch, run_id="run-crash-0001")
        self._mid_fan_out(phase, clock)

        handle._atexit()

        assert status_file.read_run(str(phase))["state"] == detach.RUN_STATE_FAILED
        models = read_progress(phase)["models"]
        assert not LIVE_STATES & {e["state"] for e in models.values()}
        assert {models[s]["error"] for s in SPECS[1:]} == {CANCELLED_ERROR}

    def test_a_clean_finish_leaves_the_map_exactly_as_the_fan_out_wrote_it(
        self, tmp_path, clock, monkeypatch
    ):
        """Only a *non-normal* close-out retires, and it is narrow on purpose.

        A `done` run finished its slots; anything still live in its map is the
        fan-out's business (its own sweep runs on that path), and rewriting it
        from here could only overwrite an honest ending with `cancelled`.
        """
        handle, phase = self._claimed(tmp_path, monkeypatch, run_id="run-clean-0001")
        self._mid_fan_out(phase, clock)
        before = read_progress(phase)["models"]

        handle.finish(0)

        assert status_file.read_run(str(phase))["state"] == detach.RUN_STATE_DONE
        assert read_progress(phase)["models"] == before
        assert read_progress(phase)["models"][SPECS[1]]["state"] == STATE_RUNNING

    def test_the_aborting_process_retires_a_worker_that_took_the_sigkill(
        self, tmp_path, clock, monkeypatch
    ):
        """`--abort` escalates to SIGKILL after 10 s, and SIGKILL catches nothing.

        `finish()`'s retirement covers the worker that got its SIGTERM; this
        covers the one that did not, where the only process left to record
        anything is the `status --abort` that killed it. Same rule from the other
        side: whoever writes the terminal record retires the map.
        """
        from utils import process_control

        handle, phase = self._claimed(tmp_path, monkeypatch, run_id="run-kill-0001")
        self._mid_fan_out(phase, clock)
        killed = []
        # The recorded pid is this test process's own — that is what makes the
        # identity probe say ALIVE — so the kill is stubbed rather than run.
        monkeypatch.setattr(
            process_control, "kill_tree",
            lambda pid, **_kw: (killed.append(pid), "killed")[1],
        )

        result = detach.abort_run(str(phase))

        assert result["result"] == "aborted" and killed == [os.getpid()]
        assert status_file.read_run(str(phase))["state"] == detach.RUN_STATE_ABORTED
        models = read_progress(phase)["models"]
        assert not LIVE_STATES & {e["state"] for e in models.values()}
        assert {models[spec]["error"] for spec in SPECS[1:]} == {CANCELLED_ERROR}
        assert read_progress(phase)["models_done"] == 3
        handle._closed = True   # the ending is recorded; no atexit re-entry

    def test_a_dead_workers_close_out_retires_the_map_too(
        self, tmp_path, clock, monkeypatch
    ):
        """`close_out_dead_run` — the heartbeat-timeout and never-started legs.

        The worker is gone, so nothing ran its `finally` either, and `status`
        (which is what notices) is the process that records the ending. Without
        this the table reads `running  30m00s/30m` under `run.state: failed`
        forever — the same lie by another route.
        """
        from utils import process_control

        monkeypatch.setattr(process_control, "kill_tree", lambda *_a, **_k: "killed")
        phase = tmp_path / "review-plan"
        phase.mkdir()
        status_file.claim_run(str(phase), {
            "run_id": "run-gone-0001", "mode": "review-plan", "state": "running",
            "pid": 2 ** 30, "process_start_time": "12345", "detached": True,
            "heartbeat_at": status_file.now_iso(), "exit_code": None,
        })
        self._mid_fan_out(phase, clock)

        assert detach.close_out_dead_run(str(phase), reason="heartbeat_timeout")

        run = status_file.read_run(str(phase))
        assert (run["state"], run["error"]) == (
            detach.RUN_STATE_FAILED, "heartbeat_timeout",
        )
        models = read_progress(phase)["models"]
        assert not LIVE_STATES & {e["state"] for e in models.values()}
        assert models[SPECS[0]]["state"] == STATE_DONE, "its own ending still stands"

    def test_a_successor_run_s_live_map_is_never_retired_by_its_predecessor(
        self, tmp_path, clock, monkeypatch
    ):
        """The compare-and-set gates the retirement, not just the `run` write.

        A worker paused past its heartbeat window is closed out and the dir taken
        over; when it wakes and calls `finish()`, its `run` write is refused —
        and the map on disk is now the **successor's**, seeded by that run's own
        `claim_run`. Retiring on a refused CAS would mark a live run's slots
        `cancelled` under a poller's nose.
        """
        handle, phase = self._claimed(tmp_path, monkeypatch, run_id="run-old-0001")
        status_file.claim_run(str(phase), {
            "run_id": "run-new-0002", "mode": "review-plan", "state": "running",
            "detached": True, "exit_code": None,
        })
        self._mid_fan_out(phase, clock)      # the successor's fan-out
        before = read_progress(phase)["models"]

        handle.finish(3)                     # the predecessor, waking up late

        assert status_file.read_run(str(phase))["run_id"] == "run-new-0002"
        assert read_progress(phase)["models"] == before
        assert read_progress(phase)["models"][SPECS[1]]["state"] == STATE_RUNNING

    @pytest.mark.parametrize("ending", ["heartbeat_timeout", "abort"])
    def test_a_successor_claiming_the_dir_mid_close_out_keeps_its_live_rows(
        self, tmp_path, clock, monkeypatch, ending
    ):
        """The terminal record and the retirement are **one** critical section.

        `retire_live_entries` takes a `checkpoint_lock` of its own and compares
        no run id, so a close-out that drops the lock between its accepted
        compare-and-set and the retirement leaves a real window: a successor
        claims the dir (which drops our `progress`), seeds its own map inside
        that window, and the predecessor's retirement then marks a *live* run's
        slots `failed: cancelled` — the same lie the CAS gate above prevents,
        arriving by the back door. Driven with a real thread, because what is
        under test is which writes the lock spans; `RunHandle.finish` already
        spans both, and these two legs must match it.
        """
        import threading

        from utils import process_control

        monkeypatch.setattr(process_control, "kill_tree", lambda *_a, **_kw: "killed")
        phase = tmp_path / "review-plan"
        phase.mkdir()
        aborting = ending == "abort"
        record = {
            "run_id": "run-old-0001", "mode": "review-plan", "state": "running",
            "detached": True, "heartbeat_at": status_file.now_iso(),
            "exit_code": None,
        }
        # `--abort` kills only a pid the identity probe calls ALIVE, so that leg
        # is handed this process's own identity; the heartbeat leg needs the
        # opposite, a pid the probe reports DEAD.
        record.update(
            process_control.identity() if aborting
            else {"pid": 2 ** 30, "process_start_time": "12345"}
        )
        status_file.claim_run(str(phase), record)
        self._mid_fan_out(phase, clock)          # the predecessor's fan-out

        at_the_door = threading.Event()
        through = threading.Event()

        def successor_worker():
            at_the_door.set()
            status_file.claim_run(str(phase), {
                "run_id": "run-new-0002", "mode": "review-plan",
                "state": "running", "detached": True, "exit_code": None,
            })
            tracker = FanoutProgress(phase, SPECS, interval_s=0, timeout=1800)
            tracker.seed()
            tracker.running(SPECS[0], attempt=1)
            through.set()

        successor = threading.Thread(target=successor_worker, daemon=True)
        real_update_run = status_file.update_run
        raced = []

        def update_run_then_hand_over(*args, **kwargs):
            wrote = real_update_run(*args, **kwargs)
            if not raced:
                # The instant the terminal `run` record is on disk: exactly the
                # window the bug left open. The successor gets in only if the
                # checkpoint lock was released with it, so the wait times out
                # once the two writes share one section.
                raced.append(True)
                successor.start()
                assert at_the_door.wait(5), "the successor thread never ran"
                through.wait(1)
            return wrote

        monkeypatch.setattr(status_file, "update_run", update_run_then_hand_over)

        if aborting:
            assert detach.abort_run(str(phase))["result"] == "aborted"
        else:
            assert detach.close_out_dead_run(str(phase), reason="heartbeat_timeout")
        successor.join(5)
        assert raced and not successor.is_alive()

        assert status_file.read_run(str(phase))["run_id"] == "run-new-0002"
        models = read_progress(phase)["models"]
        assert models[SPECS[0]]["state"] == STATE_RUNNING, (
            "the predecessor's close-out retired the successor's live slot"
        )
        assert {e["state"] for e in models.values()} == {STATE_RUNNING, STATE_QUEUED}
        assert not any(e.get("error") == CANCELLED_ERROR for e in models.values())

    def test_retiring_a_map_that_already_ended_rewrites_nothing(
        self, tmp_path, clock
    ):
        """Idempotent, and silent: the ordinary abort of a finished fan-out.

        By far the common case — the fan-out swept on its way out and only the
        `run` record is left to write — so it must not churn the file or restate
        counters a reader is already looking at.
        """
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        for spec in SPECS:
            tracker.done(spec, source="file")
        settled = Path(tmp_path, ".status.json").read_bytes()

        assert progress_tracker.retire_live_entries(tmp_path) == 0
        assert Path(tmp_path, ".status.json").read_bytes() == settled

    def test_retiring_a_checkpoint_with_no_map_is_a_no_op_not_a_crash(self, tmp_path):
        """Three shapes a close-out must survive: no file, no `progress`, no map.

        This runs inside a signal handler on a run that is already ending; a
        raise here would replace the abort's own exit path with a traceback.
        """
        assert progress_tracker.retire_live_entries(tmp_path / "never-created") == 0
        assert progress_tracker.retire_live_entries(tmp_path) == 0
        status_file.update_progress(str(tmp_path), {"step": "fanout"})
        assert progress_tracker.retire_live_entries(tmp_path) == 0
        assert read_progress(tmp_path) == {"step": "fanout"}

    @pytest.mark.parametrize(
        "reason", ["heartbeat_timeout", "worker_never_started"]
    )
    def test_a_dead_workers_rows_say_how_it_died_not_cancelled(
        self, tmp_path, clock, monkeypatch, reason
    ):
        """Both halves of one checkpoint tell one story about the same ending.

        `close_out_dead_run` already receives the word — `heartbeat_timeout`,
        `worker_never_started`, `aborted_by_request (process already gone)` —
        and stamps it on the `run` record, but it called `retire_live_entries`
        with the default `CANCELLED_ERROR`. A poller then read `failed:
        cancelled` in the models table and the `just finished:` line under
        `run.error: heartbeat_timeout`, i.e. "somebody cancelled these models"
        for a worker that died on its own.
        """
        from utils import process_control

        monkeypatch.setattr(process_control, "kill_tree", lambda *_a, **_k: "killed")
        phase = tmp_path / "review-plan"
        phase.mkdir()
        status_file.claim_run(str(phase), {
            "run_id": "run-gone-0002", "mode": "review-plan", "state": "running",
            "pid": 2 ** 30, "process_start_time": "12345", "detached": True,
            "heartbeat_at": status_file.now_iso(), "exit_code": None,
        })
        self._mid_fan_out(phase, clock)

        assert detach.close_out_dead_run(str(phase), reason=reason)

        assert status_file.read_run(str(phase))["error"] == reason
        models = read_progress(phase)["models"]
        assert {models[spec]["error"] for spec in SPECS[1:]} == {reason}, (
            "the table contradicts the `run` record about the same ending"
        )
        assert CANCELLED_ERROR not in {
            e.get("error") for e in models.values()
        }


# ---------------------------------------------------------------------------
# The mode whose ending is recorded where its map is not: `full`
# ---------------------------------------------------------------------------


class TestAFullRunRetiresTheMapsOfThePhasesItDrove:
    """`full` claims `{plan}/full/` and fills `{plan}/<phase>/`.

    It owns a directory for its lock, log and ``run`` record and no phase of its
    own; each phase it runs is called in process, and every one of those
    trackers is built for its own ``{plan}/<phase>/`` and writes there through
    the foreign-dir branch of ``report_progress`` (F3, 2026-09-20). So the
    close-out rule "whoever records the ending retires the map" swept
    ``{plan}/full/`` — a directory that holds no ``progress.models`` at all —
    and the phase that was actually fanning out kept rows reading
    ``codex:gpt-5.6-sol  running  30m00s/30m`` under **no** ``run`` record of
    its own, so nothing left in the system could ever retire them. That dir then
    read ``running in foreground (…)`` and kept ``_checkpoint_in_flight`` at
    ``True`` on every later poll, on the longest-running mode the tool has.

    All three close-outs carry the blind spot, so all three are driven here.
    """

    def _plan_dirs(self, tmp_path):
        """``{plan}/full/`` beside ``{plan}/review-plan/``, as the mode lays them out."""
        plan = tmp_path / "a plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        out = tmp_path / "a_plan"
        full = out / "full"
        phase = out / "review-plan"
        full.mkdir(parents=True)
        phase.mkdir(parents=True)
        return plan, full, phase

    def _driven_fan_out(self, phase, clock):
        """One phase mid-fan-out, with no ``run`` record of its own — as `full` leaves it."""
        tracker = FanoutProgress(phase, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.done(SPECS[0], source="file")
        tracker.running(SPECS[1], attempt=2)
        tracker.staggering(SPECS[2])
        clock.now += 300
        assert "run" not in status_file.read_status(str(phase)), (
            "the phase dirs `full` drives carry no run record; that is the bug"
        )
        return tracker

    def _handle(self, plan, full, monkeypatch, *, mode="full",
                run_id="run-full-0001"):
        from utils import process_control

        monkeypatch.setattr(
            process_control, "install_worker_signal_handlers", lambda _cb: None
        )
        monkeypatch.setattr(detach, "_ACTIVE", None)
        handle = detach.RunHandle(
            str(full), run_id=run_id, mode=mode, plan_file=str(plan),
            detached=False,
        )
        handle.start()
        return handle

    def _assert_retired(self, phase, error):
        models = read_progress(phase)["models"]
        assert not LIVE_STATES & {e["state"] for e in models.values()}, (
            "the phase `full` was driving still reads `running` under a run "
            "that is over, and nothing else will ever retire it"
        )
        assert models[SPECS[0]]["state"] == STATE_DONE, "its own ending still stands"
        assert {models[spec]["error"] for spec in SPECS[1:]} == {error}
        assert read_progress(phase)["models_done"] == 3

    def test_an_aborted_full_run_retires_the_phase_it_was_driving(
        self, tmp_path, clock, monkeypatch
    ):
        import signal as _signal

        plan, full, phase = self._plan_dirs(tmp_path)
        handle = self._handle(plan, full, monkeypatch)
        self._driven_fan_out(phase, clock)

        monkeypatch.setattr(detach.os, "_exit", [].append)
        handle._on_signal(_signal.SIGTERM)

        assert status_file.read_run(str(full))["state"] == detach.RUN_STATE_ABORTED
        self._assert_retired(phase, CANCELLED_ERROR)

    def test_a_clean_full_run_leaves_the_phase_maps_exactly_as_they_are(
        self, tmp_path, clock, monkeypatch
    ):
        """The same narrow gate the handle's own dir gets: only a non-`done` ending.

        A `full` run that finished really did finish its phases, and each one's
        own sweep ran in process; rewriting their maps from here could only
        overwrite an honest ending with `cancelled`.
        """
        plan, full, phase = self._plan_dirs(tmp_path)
        handle = self._handle(plan, full, monkeypatch, run_id="run-full-0002")
        self._driven_fan_out(phase, clock)
        before = read_progress(phase)["models"]

        handle.finish(0)

        assert status_file.read_run(str(full))["state"] == detach.RUN_STATE_DONE
        assert read_progress(phase)["models"] == before
        assert read_progress(phase)["models"][SPECS[1]]["state"] == STATE_RUNNING

    def test_only_full_reaches_into_its_neighbours(
        self, tmp_path, clock, monkeypatch
    ):
        """A phase run owns one dir and must never touch the dir next door.

        `review-plan --detach` ending is not a statement about `tasks/`, and the
        sweep is keyed on the mode word for exactly that reason.
        """
        plan, full, phase = self._plan_dirs(tmp_path)
        own = full.parent / "tasks"
        own.mkdir()
        handle = self._handle(
            plan, own, monkeypatch, mode="review-plan", run_id="run-solo-0001"
        )
        self._driven_fan_out(phase, clock)
        before = read_progress(phase)["models"]

        handle.finish(3)

        assert status_file.read_run(str(own))["state"] == detach.RUN_STATE_FAILED
        assert read_progress(phase)["models"] == before

    def test_a_neighbour_with_a_live_run_of_its_own_is_left_alone(
        self, tmp_path, clock, monkeypatch
    ):
        """A separately detached phase is not one this handle drove.

        `full`'s phases carry no `run` record — that is what strands their rows.
        A dir that *does* carry a live one belongs to another worker, and
        retiring its map would mark a live run's slots `cancelled`: the same lie
        the compare-and-set prevents for the handle's own dir, arriving through
        the new sweep.
        """
        plan, full, phase = self._plan_dirs(tmp_path)
        handle = self._handle(plan, full, monkeypatch, run_id="run-full-0003")
        self._driven_fan_out(phase, clock)
        # Someone ran `review-plan --detach` over the same plan, concurrently.
        status_file.claim_run(str(phase), {
            "run_id": "run-solo-0002", "mode": "review-plan", "state": "running",
            "detached": True, "exit_code": None,
        })
        tracker = FanoutProgress(phase, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        tracker.running(SPECS[0], attempt=1)
        before = read_progress(phase)["models"]

        handle.finish(3)

        assert read_progress(phase)["models"] == before
        assert read_progress(phase)["models"][SPECS[0]]["state"] == STATE_RUNNING

    def test_the_aborting_process_sweeps_the_phase_dirs_too(
        self, tmp_path, clock, monkeypatch
    ):
        """`abort_run`: the worker took the SIGKILL, so the killer records it all."""
        from utils import process_control

        plan, full, phase = self._plan_dirs(tmp_path)
        handle = self._handle(plan, full, monkeypatch, run_id="run-full-0004")
        self._driven_fan_out(phase, clock)
        monkeypatch.setattr(
            process_control, "kill_tree", lambda *_a, **_kw: "killed"
        )

        assert detach.abort_run(str(full))["result"] == "aborted"

        assert status_file.read_run(str(full))["state"] == detach.RUN_STATE_ABORTED
        self._assert_retired(phase, CANCELLED_ERROR)
        handle._closed = True   # the ending is recorded; no atexit re-entry

    def test_a_dead_full_workers_close_out_sweeps_them_with_its_own_reason(
        self, tmp_path, clock, monkeypatch
    ):
        """`close_out_dead_run`, and the reason it carries reaches the phase rows too."""
        from utils import process_control

        monkeypatch.setattr(process_control, "kill_tree", lambda *_a, **_k: "killed")
        plan, full, phase = self._plan_dirs(tmp_path)
        status_file.claim_run(str(full), {
            "run_id": "run-full-0005", "mode": "full", "state": "running",
            "pid": 2 ** 30, "process_start_time": "12345", "detached": True,
            "plan_file": str(plan),
            "heartbeat_at": status_file.now_iso(), "exit_code": None,
        })
        self._driven_fan_out(phase, clock)

        assert detach.close_out_dead_run(str(full), reason="heartbeat_timeout")

        assert status_file.read_run(str(full))["state"] == detach.RUN_STATE_FAILED
        self._assert_retired(phase, "heartbeat_timeout")


class TestTheTeardownKeepsItsCallersCancellation:
    """Whose `CancelledError` was that? `task.cancelled()` cannot say.

    Both tickers cancel their own task and then await it, so by the time the
    `except` runs `task.cancelled()` is `True` in **both** of the cases the old
    check claimed to distinguish. Reproduced: with the fan-out parked inside
    `stop_ticker`, a sibling cancelling it left `stop_ticker` returning normally
    and the fan-out walking on into grouping — the caller's cancellation eaten,
    which is the exact thing the check exists to prevent (F2, 2026-09-20).
    """

    TICKERS = [
        (lambda d: FanoutProgress(d, SPECS, interval_s=5, timeout=1800), "fanout"),
        (lambda d: PhaseProgress(d, interval_s=5), "phase"),
    ]

    @pytest.mark.parametrize("make", [t[0] for t in TICKERS],
                             ids=[t[1] for t in TICKERS])
    def test_our_own_cancellation_is_still_swallowed(self, tmp_path, clock, lines,
                                                     make):
        """The half that must not regress: a teardown never fails a good run."""
        async def scenario():
            tracker = make(tmp_path)
            tracker.start_ticker()
            await run_until(clock, 10)
            await tracker.stop_ticker()
            return "returned normally"

        assert asyncio.run(scenario()) == "returned normally"

    @pytest.mark.skipif(
        not hasattr(asyncio.Task, "cancelling"),
        reason="Task.cancelling() is 3.11+; below it the distinction cannot be made",
    )
    @pytest.mark.parametrize("make", [t[0] for t in TICKERS],
                             ids=[t[1] for t in TICKERS])
    def test_a_cancellation_arriving_during_the_teardown_is_re_raised(
        self, tmp_path, clock, lines, make
    ):
        walked_on = []

        async def scenario():
            tracker = make(tmp_path)
            main = asyncio.current_task()

            async def ticker_body():
                try:
                    await asyncio.Event().wait()
                finally:
                    # The real ticker's teardown sweeps and emits, and the caller
                    # is parked on `await task` for the whole of it. That is the
                    # window a sibling's cancellation lands in, so it is the one
                    # this drives - no sleep, no scheduling race.
                    main.cancel()
                    await asyncio.sleep(0)

            tracker._ticker = asyncio.ensure_future(ticker_body())
            await asyncio.sleep(0)      # let it park on the event
            await tracker.stop_ticker()
            walked_on.append("the phase carried on")

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(scenario())
        assert walked_on == [], (
            "stop_ticker swallowed its caller's cancellation and returned"
        )


# ---------------------------------------------------------------------------
# The phase-scoped second ticker
# ---------------------------------------------------------------------------


class TestPhaseProgress:
    def _checkpoint(self, tmp_path, **progress):
        status_file.update_progress(str(tmp_path), progress)

    def test_it_renders_the_step_its_elapsed_and_the_executor_s_own_counts(
        self, tmp_path, clock
    ):
        self._checkpoint(
            tmp_path,
            step="validation",
            subagents={"kind": "validation", "invoker": "claude", "total": 8,
                       "ok": 5, "counts": {"ok": 5}, "failed_task_ids": []},
        )
        phase = PhaseProgress(tmp_path, interval_s=5)
        clock.now += 150
        line = phase.render_line()

        assert line == f"{PROGRESS_PREFIX} step: validation | 2m30s | subagents 5/8 done (5 ok)"
        assert line.isascii()

    def test_it_reads_the_write_lane_s_mirror_too(self, tmp_path, clock):
        self._checkpoint(
            tmp_path,
            step="implement",
            write_subagents={"kind": "implement", "total": 4,
                             "counts": {"ok": 2, "failed": 1}, "ok": 2},
        )
        phase = PhaseProgress(tmp_path, interval_s=5)
        assert "write_subagents 3/4 done (2 ok)" in phase.render_line()

    def test_it_writes_nothing_at_all(self, tmp_path, clock, lines):
        """Its ticks print; the fan-out's terminal map stays frozen."""
        tracker = FanoutProgress(tmp_path, SPECS, interval_s=0, timeout=1800)
        tracker.seed()
        for spec in SPECS:
            tracker.done(spec, source="file")
        self._checkpoint(tmp_path, step="grouping")
        before = Path(tmp_path, ".status.json").read_bytes()
        before_map = read_progress(tmp_path)["models"]

        async def scenario():
            phase = PhaseProgress(tmp_path, interval_s=5)
            phase.start_ticker()
            await run_until(clock, 20)
            await phase.stop_ticker()

        asyncio.run(scenario())

        assert len(lines.rows) >= 3, "it did tick, through the whole phase window"
        assert Path(tmp_path, ".status.json").read_bytes() == before
        assert read_progress(tmp_path)["models"] == before_map
        assert "models" not in " ".join(lines.lines)
        assert "validation" not in read_progress(tmp_path), (
            "the counts are read, never recomputed into a second key"
        )

    def test_its_last_line_carries_the_terminal_step(self, tmp_path, clock, lines):
        self._checkpoint(tmp_path, step="grouping")

        async def scenario():
            phase = PhaseProgress(tmp_path, interval_s=5)
            phase.start_ticker()
            try:
                await run_until(clock, 10)
            finally:
                self._checkpoint(tmp_path, step="complete")
                await phase.stop_ticker()

        asyncio.run(scenario())
        assert "step: grouping" in lines.lines[0]
        assert lines.lines[-1].startswith(f"{PROGRESS_PREFIX} step: complete")

    def test_it_is_silent_at_interval_zero(self, tmp_path, clock, lines):
        async def scenario():
            phase = PhaseProgress(tmp_path, interval_s=0)
            assert phase.start_ticker() is None
            await pump(5)
            await phase.stop_ticker()

        asyncio.run(scenario())
        assert lines.rows == []

    def test_a_missing_step_renders_rather_than_raises(self, tmp_path, clock):
        phase = PhaseProgress(tmp_path, interval_s=5)
        assert phase.render_line().startswith(f"{PROGRESS_PREFIX} step: ")
