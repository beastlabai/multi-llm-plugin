"""The host lane's spend guard: a per-row ceiling, a lane total, and a readback that fails closed.

Split out of `test_container_host_lane.py` so it can be unit-tested with no
Docker, no key and no network (`tests/test_host_lane_budget.py`). Nothing in
here reads the environment's key except :func:`openrouter_usage`, the one real
usage source, and nothing in here ever formats the key into a string.

THREE LIMITS, AND ONLY ONE OF THEM IS OURS TO GUARANTEE:

* **The key's own `limit`** — enforced by OpenRouter, the party doing the
  billing. The only ceiling that holds against one oversized request. The
  entrypoint refuses a key that reports none.
* **The lane total, `MLLM_E2E_BUDGET_USD`.** Checked between rows by
  :meth:`Ledger.refusal`, and — since 2026-09-23 — also mid-row, as the
  remaining headroom a :class:`RowWatch` enforces.
* **The row ceiling, `cost_budget_usd × MLLM_E2E_ROW_BUDGET_HEADROOM`** (floored
  at :data:`MIN_CEILING_USD`). Read off
  the matrix row by :func:`row_ceiling` and enforced mid-row by a
  :class:`RowWatch`, which polls the key's `usage` and STOPS THE ROW'S CONTAINER
  when the row's delta crosses the lower of the two ceilings.

WHY THE MID-ROW STOP IS NOT A HARD CAP, stated so nobody reads it as one: spend
is observable only as the key's `usage`, which is polled, and which SETTLES
AFTER THE FACT — `host_matrix.yaml` measured $0.050279 and $0.234400 landing in
the gaps after two rows of `runs/20260922T191142Z`. So a row is stopped at the
first poll whose (lagging) reading crosses its ceiling, and the overshoot is
bounded by one poll interval of spend plus the settling lag, not by zero. That
is still categorically better than the pre-2026-09-23 behaviour, where one row
could overshoot by its whole `wall_clock_s` and nothing read `cost_budget_usd`
at all.

WHY `cost_budget_usd` IS MULTIPLIED BY A HEADROOM: the cells are measured
EXPECTED spend (`claude-code-phase-implement`'s $0.92 is its one sample, rounded),
not ceilings, and this rig has recorded a 3.5x spread across four runs of one row
on one pin ($0.479441 .. $1.700198 on `claude-code-apply-native`). Stopping a row
at its own expected figure would kill ordinary passing runs.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable

#: The lane total a credentialed invocation may spend, read once per session.
DEFAULT_LANE_BUDGET_USD = 2.00
#: The multiple of a row's expected `cost_budget_usd` at which it is stopped.
DEFAULT_ROW_HEADROOM = 2.0
#: No row ceiling is set below this, whatever its cell says. The endpoint's
#: settling lag lands the PREVIOUS row's tail inside the next row's bracket —
#: $0.234400 of it, measured on `runs/20260922T191142Z` — so a ceiling under
#: that would stop a cheap row for spend it never made.
MIN_CEILING_USD = 0.25
#: How often a running row's spend is read back.
DEFAULT_POLL_S = 20.0
#: A readback is retried this many times in total before it is declared failed.
READ_ATTEMPTS = 3
#: Seconds slept before retry n (1-based), for the transient-error case only.
READ_BACKOFF_S = (2.0, 4.0)

#: The outcome word a row stopped by the guard is filed under — distinct from
#: `fail` (the product broke) and from a wall-clock cap hit (the row was slow).
BUDGET_EXCEEDED = "budget-exceeded"
#: The outcome word for a row stopped because its spend could not be read.
SPEND_UNOBSERVABLE = "spend-unobservable"

KEY_ENV = "OPENROUTER_API_KEY"
USAGE_URL = "https://openrouter.ai/api/v1/key"


class SpendUnobservable(RuntimeError):
    """The key's `usage` could not be read, after the bounded retry.

    FAIL-CLOSED is the whole point of the type: a readback that returned `None`
    used to bracket nothing and credit the row $0, and the lane went on to start
    the next paid row having measured none of the last one.
    """


def lane_budget_usd() -> float:
    return float(os.environ.get("MLLM_E2E_BUDGET_USD", DEFAULT_LANE_BUDGET_USD))


def row_headroom() -> float:
    value = float(os.environ.get("MLLM_E2E_ROW_BUDGET_HEADROOM",
                                 DEFAULT_ROW_HEADROOM))
    if value < 1.0:
        raise ValueError(
            f"MLLM_E2E_ROW_BUDGET_HEADROOM={value} is below 1, which would stop a "
            f"row BEFORE it reached its own expected spend")
    return value


def poll_interval_s() -> float:
    return float(os.environ.get("MLLM_E2E_SPEND_POLL_S", DEFAULT_POLL_S))


def declared_budget(matrix_row: dict) -> float | None:
    """The row's numeric `cost_budget_usd`, or None when absent or not a number.

    `bool` is refused explicitly: YAML's `yes` is a `bool`, and `bool` is an
    `int` in Python, so `cost_budget_usd: yes` would otherwise read as $1.
    """
    value = matrix_row.get("cost_budget_usd")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def row_ceiling(matrix_row: dict, headroom: float | None = None) -> float:
    """The spend at which this row is stopped: `cost_budget_usd × headroom`,
    never below :data:`MIN_CEILING_USD`.

    Raises `ValueError` on a row that declares no usable budget — at run time the
    credentialed fixture turns that into a failure BEFORE the container starts,
    and `tests/test_host_lane_budget.py` catches it in plain CI first.
    """
    budget = declared_budget(matrix_row)
    if budget is None:
        raise ValueError(
            f"{matrix_row.get('id')!r} declares no positive numeric "
            f"`cost_budget_usd` (got {matrix_row.get('cost_budget_usd')!r}), so "
            f"nothing bounds its spend but the lane total")
    return max(budget * (row_headroom() if headroom is None else headroom),
               MIN_CEILING_USD)


def _redact(text: str) -> str:
    key = os.environ.get(KEY_ENV) or ""
    return text.replace(key, "<redacted>") if len(key) >= 8 else text


def openrouter_usage() -> float:
    """ONE read of the dedicated key's `usage`. Raises on any failure.

    The key goes into a header and nowhere else. `urllib`'s exceptions carry the
    URL and the status line, never request headers, and :func:`read_usage`
    redacts its message anyway.
    """
    import urllib.request

    request = urllib.request.Request(
        USAGE_URL, headers={"Authorization": f"Bearer {os.environ[KEY_ENV]}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        usage = json.load(response)["data"]["usage"]
    if isinstance(usage, bool) or not isinstance(usage, (int, float)):
        raise ValueError(f"`data.usage` is {type(usage).__name__}, not a number")
    return float(usage)


def read_usage(source: Callable[[], float] = openrouter_usage, *,
               attempts: int = READ_ATTEMPTS,
               backoff_s: tuple[float, ...] = READ_BACKOFF_S,
               sleep: Callable[[float], None] = time.sleep) -> float:
    """The key's `usage`, retried a bounded number of times, or `SpendUnobservable`.

    Every exception is a failed read — a network error, an HTTP 401/5xx, a body
    that is not JSON, a `usage` that is null. `KeyboardInterrupt` and
    `SystemExit` are not `Exception`s and pass straight through.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return float(source())
        except Exception as exc:  # noqa: BLE001 — any failure is a failed read
            last = exc
            if attempt < attempts:
                sleep(backoff_s[min(attempt, len(backoff_s)) - 1]
                      if backoff_s else 0.0)
    raise SpendUnobservable(_redact(
        f"the key's `usage` could not be read after {attempts} attempt(s): "
        f"{type(last).__name__}: {last}"))


class Ledger:
    """The lane's running total, and the reasons it refuses to start another row."""

    def __init__(self, budget_usd: float | None = None) -> None:
        self.budget_usd = lane_budget_usd() if budget_usd is None else budget_usd
        self.total = 0.0
        #: Set when a row's closing readback failed: that row's spend is unknown,
        #: so the lane total is unknown, so no further row may start.
        self.unobservable: str | None = None
        #: Set when a row was stopped mid-row on the LANE total (or went dark
        #: mid-row): the lane is spent, whatever the lagging total says.
        self.halted: str | None = None

    def remaining(self) -> float:
        return self.budget_usd - self.total

    def charge(self, before: float, after: float) -> float:
        delta = after - before
        self.total += delta
        return delta

    def refusal(self) -> str | None:
        """Why the next row must not start, or None when it may."""
        if self.unobservable:
            return (f"host lane: spend is UNOBSERVABLE — {self.unobservable}. The "
                    f"lane total is unknown, so it refuses to start another row "
                    f"rather than bracket the next one against a number nobody "
                    f"measured")
        if self.halted:
            return (f"host lane: HALTED — {self.halted}. Refusing to start "
                    f"another row")
        if self.total >= self.budget_usd:
            return (f"host lane: cumulative spend ${self.total:.4f} reached "
                    f"MLLM_E2E_BUDGET_USD=${self.budget_usd:.2f} — refusing to "
                    f"start another row")
        return None


class RowWatch:
    """Poll the key while a row runs; stop the row once it crosses its ceiling.

    `ceiling_usd` is already the lower of the row ceiling and the lane's
    remaining headroom — :meth:`for_row` computes it — and `bound` names which of
    the two it was, so a trip says whether the ROW overran or the LANE ran out.

    A poll that fails after the bounded retry also stops the row
    (`SPEND_UNOBSERVABLE`): a paid turn left running with no readback is exactly
    the fail-open this replaces.
    """

    def __init__(self, *, before: float, ceiling_usd: float, bound: str,
                 stop_row: Callable[[], None],
                 source: Callable[[], float] = openrouter_usage,
                 interval_s: float | None = None,
                 read: Callable[..., float] = read_usage) -> None:
        self.before = before
        self.ceiling_usd = ceiling_usd
        self.bound = bound
        self._stop_row = stop_row
        self._source = source
        self._read = read
        self.interval_s = poll_interval_s() if interval_s is None else interval_s
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        #: None until the guard stops the row; then the evidence of why.
        self.tripped: dict | None = None
        self.peak_usd = 0.0
        self.polls = 0

    @classmethod
    def for_row(cls, matrix_row: dict, ledger: Ledger, before: float,
                stop_row: Callable[[], None], **kwargs) -> "RowWatch":
        row_cap = row_ceiling(matrix_row)
        lane_cap = ledger.remaining()
        bound = "row" if row_cap <= lane_cap else "lane"
        return cls(before=before, ceiling_usd=min(row_cap, lane_cap), bound=bound,
                   stop_row=stop_row, **kwargs)

    def poll_once(self) -> bool:
        """One readback. True when the row was stopped by it."""
        self.polls += 1
        try:
            spent = self._read(self._source) - self.before
        except SpendUnobservable as exc:
            return self._trip(SPEND_UNOBSERVABLE, str(exc), spent=None)
        self.peak_usd = max(self.peak_usd, spent)
        if spent > self.ceiling_usd:
            return self._trip(
                BUDGET_EXCEEDED,
                f"the row's key-usage delta ${spent:.4f} crossed its "
                f"{self.bound} ceiling ${self.ceiling_usd:.4f}", spent=spent)
        return False

    def _trip(self, outcome: str, reason: str, spent: float | None) -> bool:
        self.tripped = {"outcome": outcome, "reason": reason, "bound": self.bound,
                        "ceiling_usd": round(self.ceiling_usd, 6),
                        "spent_at_trip_usd": (None if spent is None
                                              else round(spent, 6)),
                        "polls": self.polls}
        self._done.set()
        self._stop_row()
        return True

    def _loop(self) -> None:
        while not self._done.wait(self.interval_s):
            if self.poll_once():
                return

    def start(self) -> "RowWatch":
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="host-lane-spend-watch")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=120)

    def __enter__(self) -> "RowWatch":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
