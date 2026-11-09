"""``utils/clock.py``: the three timestamp formats, and the writers that use them.

Each writer that once spelled its own ``datetime.now()`` helper now names one
of these; the identity checks below keep a copy from coming back.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import clock  # noqa: E402
from utils import (  # noqa: E402
    commit_apply, commit_front_door, scope_apply, scope_policy, scope_policy_runs,
    status_file, write_safety,
)

_LOCAL_SECONDS = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")
_LOCAL_MICROS = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{6})?")
_UTC_Z = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")


def _close_to(stamp: datetime, now: datetime) -> bool:
    return abs(stamp - now) < timedelta(seconds=5)


def test_local_iso_is_naive_local_time_to_the_microsecond():
    stamp = clock.local_iso()
    assert _LOCAL_MICROS.fullmatch(stamp), stamp
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is None and _close_to(parsed, datetime.now())


def test_local_iso_seconds_is_naive_local_time_to_the_second():
    stamp = clock.local_iso_seconds()
    assert _LOCAL_SECONDS.fullmatch(stamp), stamp
    assert _close_to(datetime.fromisoformat(stamp), datetime.now())


def test_utc_iso_z_is_utc_to_the_second():
    stamp = clock.utc_iso_z()
    assert _UTC_Z.fullmatch(stamp), stamp
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert _close_to(parsed, datetime.now(timezone.utc))


@pytest.mark.parametrize("name, function", [
    ("status_file.now_iso", status_file.now_iso),
    ("scope_policy_runs._now", scope_policy_runs._now),
], ids=lambda value: value if isinstance(value, str) else "")
def test_the_microsecond_writers_use_local_iso(name, function):
    assert function is clock.local_iso, name


@pytest.mark.parametrize("name, function", [
    ("scope_policy._now", scope_policy._now),
    ("scope_apply._now", scope_apply._now),
    ("write_safety._now", write_safety._now),
], ids=lambda value: value if isinstance(value, str) else "")
def test_the_seconds_writers_use_local_iso_seconds(name, function):
    assert function is clock.local_iso_seconds, name


@pytest.mark.parametrize("name, function", [
    ("commit_apply._now_utc", commit_apply._now_utc),
    ("commit_front_door._now_iso", commit_front_door._now_iso),
], ids=lambda value: value if isinstance(value, str) else "")
def test_the_utc_writers_use_utc_iso_z(name, function):
    assert function is clock.utc_iso_z, name
