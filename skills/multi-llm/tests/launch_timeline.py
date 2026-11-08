"""An asyncio event loop on a virtual clock, for launch-timeline tests.

The loop is a stock selector loop whose ``time()`` is a counter: whenever the
loop would block waiting for its next timer, the selector moves the counter to
that timer instead of sleeping. Tasks, semaphores, ``asyncio.sleep`` and
``asyncio.wait_for`` all behave as on a real loop, so a test asserts the times
at which slots really dispatched under a real schedule, in no wall time.

Nothing run on it may hand work to a thread: a thread finishes in real time,
which the virtual clock cannot see. A lane that does (``asyncio.to_thread``)
runs under :func:`inline_threads`, which keeps that work on the loop.
"""

from __future__ import annotations

import asyncio
import selectors
from typing import Any, Awaitable, Mapping, Optional, Sequence


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0


class _AdvancingSelector(selectors.DefaultSelector):
    """Moves the clock to the next timer instead of waiting for it."""

    def __init__(self, clock: _Clock) -> None:
        super().__init__()
        self._clock = clock

    def select(self, timeout=None):
        if timeout is None:
            # No timer is pending and nothing is ready: only I/O could wake the
            # loop, and a timeline test has none, so this would hang for ever.
            events = super().select(5.0)
            if not events:
                raise RuntimeError("the virtual-clock loop stalled with no timer")
            return events
        if timeout > 0:
            self._clock.now += timeout
        return super().select(0)


class VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        self._virtual = _Clock()
        super().__init__(selector=_AdvancingSelector(self._virtual))

    def time(self) -> float:
        return self._virtual.now


def run_on_virtual_clock(awaitable: Awaitable[Any]) -> Any:
    """``asyncio.run`` on a :class:`VirtualClockLoop` that starts at 0.0."""
    loop = VirtualClockLoop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(awaitable)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def now() -> float:
    """The running loop's clock: virtual on a :class:`VirtualClockLoop`."""
    return asyncio.get_running_loop().time()


def inline_threads(
    monkeypatch: Any, takes: Optional[Mapping[str, Sequence[float]]] = None
) -> None:
    """Run every ``asyncio.to_thread`` call on the loop, in the caller's context.

    *takes* maps a function's ``__name__`` to the virtual seconds its successive
    calls take (a slow probe or snapshot); a call past the end of its list, or
    of a function not named, takes none.
    """
    pending = {name: list(seconds) for name, seconds in (takes or {}).items()}

    async def to_thread(func, /, *args, **kwargs):
        seconds = pending.get(getattr(func, "__name__", ""))
        if seconds:
            await asyncio.sleep(seconds.pop(0))
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
