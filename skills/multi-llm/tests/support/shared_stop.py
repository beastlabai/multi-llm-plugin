"""A stop shared by the tests that only read it: one run per test module.

A first stop runs the real orchestrator and its mock runners (about half a
second on Linux, several on Windows), and many tests then read one line of it.
Those readers share one stop per test module. The fixtures that build it
(``tmp_path``, ``monkeypatch``, ``capsys`` and conftest's autouse isolation)
are function-scoped, so a module-scoped fixture cannot build it, and their
scope is not widened to let one. The stop is built lazily instead, inside the
isolation of whichever test asks first, and the module keeps only what is safe
to share: the finished output and snapshots, with the stop's folder left where
it ran. Its records (``state.json``, the backup manifests, the reports) name
that folder by absolute path, so a finished stop is never copied elsewhere.

Only a test that reads the stop may share it. A test that resumes, answers,
retries, repairs or writes anything there keeps its own fresh world; each
reader is checked, as it ends, to have changed no file of the stop
(:meth:`SharedStop.assert_untouched`), so a writer that slips in fails instead
of making the tests after it pass or fail by the order they ran in.

Not registered in ``tests/conftest.py``: this module imports ``utils.process_state``,
which a suite-wide fixture would put on the import map of every test that
names ``repo`` (``tests/MODULE_TESTS.md``). A test file imports the fixtures::

    from tests.support.shared_stop import (  # noqa: F401 - the fixtures
        gate_first_stop_fixture, module_stop_cache, shared_stop,
    )

Fixtures: ``stop_cache``, the module's stops by name (module-scoped);
``gate_first_stop``, the check-gate world's first stop
(``tests.support.gate_fixture._first_run``). :func:`shared_stop` shares a stop
of another world or another run through the same cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Tuple

import pytest

from utils import process_state
from utils import provider_registry as registry
from utils.state_manager import get_or_create_state

from tests.support.gate_fixture import _first_run_parts, _gate_world


#: How a stop's world is set up: ``(tmp_path, monkeypatch) -> Repo``.
World = Callable[..., object]
#: How it is run to its stop: ``(repo, capsys) -> (stdout, stderr)``.
Run = Callable[..., Tuple[str, str]]


def _fingerprint(base: Path) -> Dict[str, Tuple[int, str]]:
    """Every file under *base*: its size and sha256, by relative path."""
    found = {}
    for folder, _dirs, files in os.walk(base):
        for name in files:
            path = Path(folder) / name
            data = path.read_bytes()
            found[path.relative_to(base).as_posix()] = (
                len(data), hashlib.sha256(data).hexdigest())
    return found


class _WorldPatch:
    """The build's own patcher, remembering each variable it sets and where it
    changes directory, so a reader can put the world's environment back on its
    own function-scoped patcher (:meth:`SharedStop.enter`). Everything is undone
    when the build ends, never left to a later test."""

    def __init__(self, patch: pytest.MonkeyPatch):
        self._patch = patch
        self.env: Dict[str, Optional[str]] = {}
        self.cwd: Optional[Path] = None

    def setenv(self, name: str, value: str, prepend: Optional[str] = None) -> None:
        self._patch.setenv(name, value, prepend)
        self.env[name] = os.environ[name]

    def delenv(self, name: str, raising: bool = True) -> None:
        self._patch.delenv(name, raising=raising)
        self.env[name] = None

    def chdir(self, path) -> None:
        self._patch.chdir(path)
        self.cwd = Path(os.getcwd())

    def __getattr__(self, name):
        # setattr and the rest: undone with the build, and not replayed by
        # enter(), so a world that needs one there must record it here first.
        return getattr(self._patch, name)


@dataclass(frozen=True)
class SharedStop:
    """A finished stop, as the tests that share it may see it.

    The output and the task states are snapshots taken as the run ended. The
    folder (``base``, holding ``root``) is the one the run used. No ``Repo`` and
    no patcher of the test that built it is kept: that patcher was undone
    when the build ended.
    """

    out: str
    err: str
    base: Path
    root: Path
    plan: Path
    phase_dir: Path
    statuses: Mapping[str, str]
    reasons: Mapping[str, str]
    report_text: str
    env: Mapping[str, Optional[str]]
    cwd: Path
    fingerprint: Mapping[str, Tuple[int, str]]

    @property
    def text(self) -> str:
        """What ``_first_run`` returns: stdout, then stderr."""
        return self.out + self.err

    def report(self) -> dict:
        """``write_batch_report.json`` as the stop wrote it, parsed afresh for
        each reader, so no reader can change what the next one reads."""
        return json.loads(self.report_text)

    def enter(self, monkeypatch) -> None:
        """Set the world's environment and working directory on the reader's
        own *monkeypatch*, for a reader that runs a read-only command (such as
        ``status``) against the stop as the test that ran it would have."""
        for name, value in self.env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        monkeypatch.chdir(self.cwd)
        registry._config = None

    def assert_untouched(self) -> None:
        """Fail a reader that changed a file of the stop: the tests after it
        would read what it left."""
        now = _fingerprint(self.base)
        if now != self.fingerprint:
            changed = sorted(
                path for path in set(now) | set(self.fingerprint)
                if now.get(path) != self.fingerprint.get(path))
            pytest.fail(
                f"a test that shares the stop in {self.base} changed "
                f"{changed}: a test that writes there needs its own world",
                pytrace=False,
            )


def _build_stop(capsys, tmp_path_factory, key: str, world: World, run: Run) -> SharedStop:
    base = tmp_path_factory.mktemp(f"stop-{key}")
    try:
        with pytest.MonkeyPatch.context() as patch:
            recorder = _WorldPatch(patch)
            # Out-of-tree rollback evidence (`_isolate_state_home`) lives with
            # the world, inside what assert_untouched() watches, not in the tmp
            # dir of the test that happened to build it.
            recorder.setenv("XDG_STATE_HOME", str(base / "xdg-state"))
            obj = world(base, recorder)
            capsys.readouterr()  # what the setup printed is not the stop's
            out, err = run(obj, capsys)
            state = get_or_create_state(obj.plan)
            statuses = dict(state.implementation_task_statuses())
            reasons = {
                task_id: state.get_task_status_reason(task_id) or ""
                for task_id in statuses
            }
            report = obj.phase_dir / "write_batch_report.json"
            return SharedStop(
                out=out, err=err, base=base, root=obj.root, plan=obj.plan,
                phase_dir=obj.phase_dir,
                statuses=MappingProxyType(statuses),
                reasons=MappingProxyType(reasons),
                report_text=report.read_text(encoding="utf-8") if report.exists() else "",
                env=MappingProxyType(dict(recorder.env)),
                cwd=recorder.cwd or obj.root,
                fingerprint=MappingProxyType(_fingerprint(base)),
            )
    finally:
        # The build's patches are undone; empty what it loaded too, as the
        # next test's autouse fixtures would, so the test that built the stop
        # runs its body as one that found it built does.
        registry._config = None
        registry._RUN_ANCHOR = None
        process_state.reset_all()


def shared_stop(
    cache: dict, key: str, capsys, tmp_path_factory, world: World, run: Run,
) -> SharedStop:
    """The module's stop named *key*: *world* run to its stop by *run*, built on
    the first request (inside that test's own isolation) and reused after.

    *cache* is the module's ``stop_cache``. A fixture that hands the stop out
    calls :meth:`SharedStop.assert_untouched` as each reader ends. A build that
    fails caches nothing, so each reader reports the failure itself."""
    stop = cache.get(key)
    if stop is None:
        stop = cache[key] = _build_stop(capsys, tmp_path_factory, key, world, run)
    return stop


@pytest.fixture(scope="module", name="stop_cache")
def module_stop_cache():
    """The stops one test module shares, by name. It holds only finished
    snapshots and requests no function-scoped fixture: every stop is built by
    :func:`shared_stop` inside a test's own isolation."""
    stops: Dict[str, SharedStop] = {}
    yield stops
    stops.clear()


@pytest.fixture(name="gate_first_stop")
def gate_first_stop_fixture(stop_cache, capsys, tmp_path_factory):
    """The check-gate ``repo`` world's first stop (``_first_run``), shared by
    the tests of a module that only read it."""
    stop = shared_stop(
        stop_cache, "first", capsys, tmp_path_factory, _gate_world, _first_run_parts)
    yield stop
    stop.assert_untouched()
