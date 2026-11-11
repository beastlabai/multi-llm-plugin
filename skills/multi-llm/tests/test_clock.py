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


@pytest.mark.parametrize("text", [
    "2026-09-21T10:00:10Z",
    "2026-09-21T10:00:10z",
    "2026-09-21T10:00:10+00:00",
    "2026-09-21T10:00:10.250000Z",
])
def test_parse_iso_reads_utc_in_either_spelling_as_aware(text):
    parsed = clock.parse_iso(text)
    assert parsed.utcoffset() == timedelta(0)
    assert parsed.replace(microsecond=0) == datetime(2026, 9, 21, 10, 0, 10, tzinfo=timezone.utc)


def test_parse_iso_leaves_a_naive_stamp_naive():
    assert clock.parse_iso("2026-09-21T10:00:10") == datetime(2026, 9, 21, 10, 0, 10)
    assert clock.parse_iso("2026-09-21T10:00:10").tzinfo is None


@pytest.mark.parametrize("text", [
    "", "Z", "not a stamp",
    # A UTC suffix is read only after a time, once, and only as the last character.
    "2026-09-21Z", "2026-09-21z", "2026-09-21T10:00:10ZZ", "2026-09-21T10:00:10zz",
    "2026-09-21T10:00:10Zz", "2026-09-21T10:00:10+00:00Z", "2026-09-21Z10:00:10",
    "2026-09-21T10:00:10Z ", "2026-09-21T10:00:10Z+00:00",
])
def test_parse_iso_refuses_a_malformed_utc_stamp(text):
    with pytest.raises(ValueError):
        clock.parse_iso(text)


def test_a_utc_stamp_is_never_a_fabricated_naive_midnight():
    # ``2026-09-21+00:00`` reads as a date, a separator and a time of day.
    assert datetime.fromisoformat("2026-09-21+00:00").tzinfo is None
    with pytest.raises(ValueError):
        clock.parse_local("2026-09-21Z")


@pytest.mark.parametrize("text, expect_aware", [
    ("2026-09-21T10:00:10Z", False),
    ("2026-09-21T10:00:10+00:00", False),
    ("2026-09-21T10:00:10", False),
])
def test_the_checkpoint_stamp_reader_gives_naive_local_time_for_every_spelling(text, expect_aware):
    from utils import run_record

    got = run_record._parse_stamp(text)
    assert got is not None and (got.tzinfo is not None) is expect_aware
    if text.endswith(("Z", "+00:00")):
        want = datetime(2026, 9, 21, 10, 0, 10, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    else:
        want = datetime(2026, 9, 21, 10, 0, 10)
    assert got == want


# -- stamps compared with each other, and at the edge of the range --------------

_needs_tzset = pytest.mark.skipif(
    not hasattr(__import__("time"), "tzset"), reason="time.tzset is POSIX-only")


@pytest.fixture
def berlin(monkeypatch):
    """The process in a zone ahead of UTC, so a year-9999 UTC stamp overflows locally."""
    import time
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def new_york(monkeypatch):
    import time
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _unreadable_readers():
    from utils import prior_changes, run_record, state_manager
    return [
        ("clock.parse_local", clock.parse_local, ValueError),
        ("state_manager", state_manager._parse_completion_time, None),
        ("prior_changes", prior_changes.parse_stamp, None),
        ("run_record", run_record._parse_stamp, None),
        ("scope_policy_runs", scope_policy_runs._parse, None),
    ]


@_needs_tzset
@pytest.mark.parametrize("stamp, zone", [
    ("9999-12-31T23:59:59Z", "berlin"),
    ("9999-12-31T23:59:59+00:00", "berlin"),
    ("0001-01-01T00:00:00Z", "new_york"),
])
def test_a_boundary_stamp_is_unreadable_not_an_overflow(stamp, zone, request):
    request.getfixturevalue(zone)
    for name, reader, raises in _unreadable_readers():
        if raises is not None:
            with pytest.raises(raises):
                reader(stamp)
        else:
            assert reader(stamp) is None, name


@_needs_tzset
def test_a_stamp_inside_the_range_still_converts_to_local_time(berlin):
    from utils import prior_changes, state_manager
    want = datetime(2026, 9, 21, 12, 0, 10)  # CEST is UTC+2
    for stamp in ("2026-09-21T10:00:10Z", "2026-09-21T10:00:10z", "2026-09-21T12:00:10+02:00"):
        assert clock.parse_local(stamp) == want
        assert state_manager._parse_completion_time(stamp) == want
        assert prior_changes.parse_stamp(stamp) == want


def _spellings(instant: datetime) -> dict:
    """One instant as a naive local stamp, ``Z``, ``z`` and two offsets."""
    return {
        "naive": instant.astimezone().replace(tzinfo=None).isoformat(),
        "Z": instant.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "z": instant.strftime("%Y-%m-%dT%H:%M:%Sz"),
        "utc-offset": instant.isoformat(),
        "offset": instant.astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat(),
    }


_RECORD = datetime(2026, 9, 21, 10, 0, 0, tzinfo=timezone.utc)
_KINDS = ["naive", "Z", "z", "utc-offset", "offset"]


@pytest.mark.parametrize("completed_kind", _KINDS)
@pytest.mark.parametrize("record_kind", _KINDS)
@pytest.mark.parametrize("where", ["phases_completed", "phases_skipped"])
def test_a_continuation_compares_any_two_stamp_spellings(where, record_kind, completed_kind):
    """A scope-policy record and the phase stamp that may close it are two stamps
    that two writers spelled: each in any of the spellings, they compare as instants."""
    def state(offset_s: int) -> dict:
        stamp = _spellings(_RECORD + timedelta(seconds=offset_s))[completed_kind]
        phase = stamp if where == "phases_completed" else {"skipped_at": stamp}
        return {
            "scope_policy_runs": {"improve-code": {
                "policy": "preserve", "recorded_at": _spellings(_RECORD)[record_kind]}},
            where: {"apply-code-fixes": phase},
        }

    assert scope_policy_runs.resolve_run_choice(None, state(-10), "improve-code") == (
        "preserve", "continuation")
    assert scope_policy_runs.resolve_run_choice(None, state(10), "improve-code") is None


def test_the_ordered_comparison_does_not_depend_on_the_zone_of_the_machine(new_york):
    state = {
        "scope_policy_runs": {"improve-code": {
            "policy": "preserve", "recorded_at": "2026-09-21T10:00:00"}},
        "phases_completed": {"apply-code-fixes": "2026-09-21T10:00:10Z"},
    }
    # 10:00:10Z is 06:00:10 in New York, earlier than a naive 10:00:00 local record.
    assert scope_policy_runs.resolve_run_choice(None, state, "improve-code") == (
        "preserve", "continuation")


def test_no_skill_module_parses_a_stamp_with_a_bare_fromisoformat():
    """One stamp parser: every module reads an ISO stamp through
    ``clock.parse_iso`` or ``clock.parse_local``, which read the ``Z``/``z``
    spelling ``datetime.fromisoformat`` refuses on some Pythons (Python 3.10
    refuses ``Z``, every Python ``z``). A ``date.fromisoformat`` (a date, not
    a stamp) is not this rule's."""
    import ast

    skill = Path(__file__).resolve().parents[1]
    found = []
    for path in sorted(skill.rglob("*.py")):
        parts = path.relative_to(skill).parts
        if parts[0] in ("tests", ".venv") or "__pycache__" in parts:
            continue
        if path.relative_to(skill).as_posix() == "utils/clock.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Attribute) and node.attr == "fromisoformat"
                    and isinstance(node.value, ast.Name) and node.value.id == "datetime"):
                found.append(f"{path.relative_to(skill)}:{node.lineno}")
    assert found == []
