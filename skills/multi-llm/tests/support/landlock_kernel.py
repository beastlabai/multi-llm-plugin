"""Whether this kernel offers Landlock: ``require_landlock()`` skips a test unless the ABI is 3 or newer.

With ``MULTI_LLM_REQUIRE_LANDLOCK=1`` (``REQUIRE_ENV``) a missing Landlock fails
the test instead of skipping it. No fixtures.
"""

import errno
import functools
import os

import pytest

from utils import landlock_exec as lx


#: Set on a host that must run tier 2: a skip there fails instead.
REQUIRE_ENV = "MULTI_LLM_REQUIRE_LANDLOCK"


@functools.lru_cache(maxsize=None)
def _landlock_unusable_reason():
    """``None`` when this kernel can run the jail, else why not."""
    kernel = lx.Kernel()
    machine = kernel.machine()
    if machine not in lx.SYSCALLS:
        return f"machine {machine!r} is not in the wrapper's syscall table"
    try:
        abi = kernel.abi()
    except OSError as exc:
        name = errno.errorcode.get(exc.errno, str(exc.errno))
        return f"Landlock unavailable: {name} ({exc.strerror})"
    if abi < lx.ABI_FLOOR:
        return f"Landlock ABI {abi} found; the floor is {lx.ABI_FLOOR}"
    return None


def require_landlock():
    """Skip unless the ABI query returns >= 3; with ``MULTI_LLM_REQUIRE_LANDLOCK=1``
    the same condition fails instead, so a skip cannot hide a regression."""
    reason = _landlock_unusable_reason()
    if reason is None:
        return
    if os.environ.get(REQUIRE_ENV, "") not in ("", "0"):
        pytest.fail(f"{REQUIRE_ENV} is set but the tier-2 Landlock rows cannot run: {reason}")
    pytest.skip(reason)
