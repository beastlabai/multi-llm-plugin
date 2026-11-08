"""Pytest fixtures for multi-llm skill tests."""

import glob
import hashlib
import json
import os
import pytest
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import (
    Dict, Generator, List, NamedTuple, Optional, Sequence, Set, Tuple,
)
from unittest.mock import MagicMock, patch

# Make the skill's own packages (utils.*) importable from conftest regardless of
# where pytest is launched, so the config-isolation fixture below can reach
# utils.provider_registry.
sys.path.insert(0, str(Path(__file__).parent.parent))

from .harness import (
    SkillRunner,
    FixtureManager,
    MockProvider,
    AssertionHelpers,
)


#: The uv this test session runs under, resolved once at import — before any
#: test narrows PATH to a mock bin directory.
_SESSION_UV = os.environ.get("MULTI_LLM_UV") or __import__("shutil").which("uv")


@pytest.fixture(autouse=True)
def _session_uv_stays_reachable(monkeypatch):
    """Keep the real uv reachable for the dispatcher's uv backstop.

    Many fixtures narrow PATH to a mock bin dir plus `/usr/bin:/bin`, and
    `unattended.require_uv()` runs before every mode. On a developer machine it
    then still found uv through its `~/.local/bin` home fallback, so those tests
    passed there and stopped at `[UV_MISSING]` (exit 3) on a CI runner, whose uv
    lives in the tool cache: measured 2026-10-23, 104 tests on the first push of
    this history to master. `$MULTI_LLM_UV` is the lookup's documented first
    rung; tests of the lookup itself `delenv` it, which still wins.
    """
    if _SESSION_UV and "MULTI_LLM_UV" not in os.environ:
        monkeypatch.setenv("MULTI_LLM_UV", _SESSION_UV)


@pytest.fixture(autouse=True)
def _clear_process_caches():
    """Start and end every test with every module-level cache empty.

    The caches are process-scoped on purpose (``utils/process_state.py``), so in
    one pytest worker a probe verdict, a version read or a warn-once entry would
    otherwise reach the next test. Every cache is registered there and emptied
    in place here; a test that needs a cold cache mid-test calls ``.clear()`` on
    it, never ``setattr`` (``tests/test_process_state.py``).
    """
    from utils import process_state

    process_state.reset_all()
    yield
    process_state.reset_all()


def pytest_addoption(parser):
    """The container host lane's empty-lane escape hatch (C6).

    A host-lane invocation that executed zero credentialed rows reports `not-run`
    and exits non-zero, because "everything skipped" and "everything passed" are
    the two outcomes that lane must never conflate. This flag is the only way to a
    green exit out of that state — it suppresses the failure, never the census —
    and it exists for the contributor deliberately confirming the rig is wired up
    without a key. It is never the default.
    """
    parser.addoption(
        "--allow-empty-host-lane",
        action="store_true",
        default=False,
        help="container host lane: accept an invocation that executed zero "
             "credentialed rows (still prints and stores the census) instead of "
             "failing it as not-run",
    )


#: The container modules that GRADE what other container rows produced instead of
#: driving rows of their own. `test_container_argv.py` reads the wrapper call logs
#: the provider lane and the full chain leave behind — and alphabetically it
#: collects before both of them, so in one `make e2e-container` it would run with
#: nothing on disk yet (a skip) or against the previous invocation's logs (a pass).
#: Neither graded the run in front of it. pytest gives no ordering guarantee
#: between modules and no ordering plugin is a dependency here, so the one hook
#: that does exist moves these items to the end of the session, keeping them
#: contiguous and in their collected order so their module-scoped fixtures still
#: set up and tear down exactly once.
_GRADES_OTHER_ROWS = ("test_container_argv.py",)


def pytest_collection_modifyitems(items):
    deferred = [item for item in items
                if Path(item.nodeid.split("::", 1)[0]).name in _GRADES_OTHER_ROWS]
    if not deferred or len(deferred) == len(items):
        return
    postponed = {id(item) for item in deferred}
    items[:] = [item for item in items if id(item) not in postponed] + deferred


# --------------------------------------------------------------------------- #
# The container host lane's SECOND verdict (C10b).
#
# A `hosts/capabilities.yaml` promotion is a claim that a capability was
# demonstrated, and the harness may only propose one for a row that passed twice:
# the container exited 0 **and** the assertions over the evidence that run left
# behind also passed. The driver knows the first when `run_row` returns; only
# pytest knows the second, and only after the record has already been written. So
# the driver writes the record provisionally (no promotions, by construction) and
# these two hooks carry the grader's verdict back to it.
#
# Both are no-ops for every non-container session: the ledger is empty unless a
# host row really ran, and the import is guarded so a checkout without the
# container package still collects.
# --------------------------------------------------------------------------- #

#: `None` until first asked; `False` when the container package is unavailable,
#: which is a legitimate state (the harness is optional) rather than an error.
_GRADE_LEDGER = None


def _grade_ledger():
    global _GRADE_LEDGER
    if _GRADE_LEDGER is None:
        try:
            from .container.harness import GRADES
        except Exception:                       # noqa: BLE001 — optional package
            _GRADE_LEDGER = False
        else:
            _GRADE_LEDGER = GRADES
    return _GRADE_LEDGER or None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """File each test's verdict against every container row the test read.

    Deliberately swallowing: this runs for every phase of every test in the suite,
    and a bookkeeping error here must never be what turns a green test red. A row
    that collects no verdict promotes nothing, so the failure mode of a silent
    miss is a withheld promotion — the safe direction.
    """
    outcome = yield
    ledger = _grade_ledger()
    if ledger is None or not ledger.rows:
        return
    try:
        ledger.note_report(item, outcome.get_result())
    except Exception:                           # noqa: BLE001 — see docstring
        pass


def pytest_sessionfinish(session, exitstatus):
    """Re-emit every host row's record with both verdicts, then print promotions.

    Session end rather than fixture teardown, because it is the one point every
    row has been graded by everything that was ever going to grade it — including
    the modules pytest reordered — and it still runs after `pytest.exit`, which is
    how the host lane reports an empty lane and a blown budget.
    """
    ledger = _grade_ledger()
    if ledger is None or not ledger.rows:
        return
    try:
        ledger.resolve_all()
        for line in ledger.summary():
            print(line)
    except Exception as exc:                    # noqa: BLE001 — reported, not hidden
        print(f"container harness: the run records could not be resolved: "
              f"{type(exc).__name__}: {exc}")
        session.exitstatus = session.exitstatus or 1
        return
    if ledger.errors:
        # A record that cannot be written is a driver bug, and C10's whole point
        # is that the record is the evidence. Reported as a failed
        # invocation rather than as a green run with a missing file.
        session.exitstatus = session.exitstatus or 1


#: The stdlib's own ``subprocess.run``, captured before any test can patch it.
_STDLIB_SUBPROCESS_RUN = subprocess.run

#: Signals whose handlers a test can leave behind in this process.
_RESTORED_SIGNALS = tuple(
    sig for sig in (getattr(signal, name, None)
                    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"))
    if sig is not None
)


@pytest.fixture(autouse=True)
def _signal_handlers_do_not_outlive_their_test():
    """Put back every cancellation handler a test replaced.

    A test that claims a detached run in-process installs the worker's abort
    handlers (``process_control.install_worker_signal_handlers``), which answer
    SIGINT/SIGTERM with ``os._exit(3)``, and nothing ever uninstalled them. That
    was harmless while no test signalled the suite's own process. The Ctrl-C
    sweep tests (plan: review-run-scope-salvage-and-failure-visibility, decision
    11.9) do exactly that, and under a leaked handler the whole pytest process
    vanished mid-run with exit 3 and no summary.
    """
    saved = {}
    for sig in _RESTORED_SIGNALS:
        try:
            saved[sig] = signal.getsignal(sig)
        except (OSError, ValueError):  # pragma: no cover
            pass
    yield
    for sig, handler in saved.items():
        if handler is None:
            continue
        try:
            if signal.getsignal(sig) is not handler:
                signal.signal(sig, handler)
        except (OSError, ValueError):  # pragma: no cover - non-main thread
            pass


@pytest.fixture(autouse=True)
def _lane_a_spawn_honours_a_faked_subprocess_run(monkeypatch):
    """Keep ``subprocess.run`` the suite's fake for the Lane A process boundary.

    The Lane A spawn used to BE ``subprocess.run``, so dozens of tests across the
    suite fake the provider process by patching ``utils.llm_client.subprocess.run``
    (which, being the stdlib module's attribute, patches it globally). The spawn
    is now ``llm_client._run_provider_cli`` — ``subprocess.run``'s own contract
    and arguments over a ``Popen``, so a timed-out CLI's detached children can be
    swept (plan: review-run-scope-salvage-and-failure-visibility D3.3). While a
    test has replaced ``subprocess.run``, the spawn is routed to that
    replacement with the identical call, so every existing fake still receives
    exactly the argv and keyword arguments the production call site passes.
    With ``subprocess.run`` untouched — every test that runs a real fake CLI —
    the real ``_run_provider_cli`` runs, sweep and all.
    """
    try:
        from utils import llm_client
    except Exception:
        yield
        return
    real_spawn = llm_client._run_provider_cli

    def spawn(cmd, **kwargs):
        if subprocess.run is not _STDLIB_SUBPROCESS_RUN:
            return subprocess.run(cmd, **kwargs)
        return real_spawn(cmd, **kwargs)

    monkeypatch.setattr(llm_client, "_run_provider_cli", spawn)
    yield


@pytest.fixture(autouse=True)
def _isolate_provider_config(request, monkeypatch):
    """Neutralize per-project config discovery for the default test suite.

    Now that ``load_config()`` reads from CWD (project-local discovery) and from
    ``MULTI_LLM_PROVIDERS_CONFIG``, the suite's many base-value assertions would
    otherwise become environment-dependent (breaking if a dev/CI env exports the
    var, or if pytest runs inside a dir that has a ``.multi-llm/providers.yaml``).

    Applied to every test that does NOT opt into override behavior via
    ``@pytest.mark.config_override``, this fixture:
      * unsets ``MULTI_LLM_PROVIDERS_CONFIG`` / ``MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE``,
      * forces project-local discovery to return ``None`` (deterministic;
        independent of where pytest is launched),
      * resets ``registry._config`` / ``registry._config_key`` before and after so
        cache state never leaks between tests.

    Tests that exercise override behavior add ``@pytest.mark.config_override`` and
    set up their own discovery/env within the test body.
    """
    try:
        import utils.provider_registry as registry
    except Exception:
        yield
        return

    def _reset_cache():
        registry._config = None
        if hasattr(registry, "_config_key"):
            registry._config_key = None
        # §9.3.b: the process-wide run anchor is set-once and immutable within a
        # process. A test that drives an orchestrator pre-flight pins one; the
        # next test's plan is a different path, which would (correctly) raise.
        if hasattr(registry, "_RUN_ANCHOR"):
            registry._RUN_ANCHOR = None

    _reset_cache()
    monkeypatch.delenv("MULTI_LLM_PROVIDERS_CONFIG", raising=False)
    monkeypatch.delenv("MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE", raising=False)
    if "config_override" not in request.keywords:
        monkeypatch.setattr(registry, "_find_project_config", lambda anchor=None: None)
    _reset_cache()
    try:
        yield
    finally:
        _reset_cache()


@pytest.fixture(autouse=True)
def _pin_opencode_model_catalog(request, monkeypatch):
    """Keep the default suite from spawning `opencode models` (D3 step 4).

    `preflight_runner` now asks the adapter's ``describe_unlisted_model`` for
    the CLI's own listing, and only ``OpenCodeProvider`` overrides the base's
    ``known=None`` default with a real subprocess. Any registry or executor
    test that preflights a real opencode spec would otherwise spawn the CLI —
    or, under the mock launcher, log a phantom dispatch. So the override is
    pinned to ``known=None`` (unknown: refuses nothing, exactly the base
    posture) for every test that does not opt out with
    ``@pytest.mark.model_catalog``. The stub providers other suites build
    inherit the base default and need nothing.
    """
    if "model_catalog" in request.keywords:
        yield
        return
    try:
        from utils.providers import opencode as oc
        from utils.providers.base import ModelVerdict
    except Exception:
        yield
        return
    monkeypatch.setattr(
        oc.OpenCodeProvider,
        "describe_unlisted_model",
        lambda self, model, *, anchor=None, profile="agent-readonly": ModelVerdict(
            known=None
        ),
    )
    yield


@pytest.fixture(autouse=True)
def _isolate_state_home(tmp_path, monkeypatch):
    """Keep every test's out-of-grant rollback evidence under its own tmp dir.

    A repo-wide write batch on the plugin's jail keeps its pre-batch
    snapshots, rollback store and audit log under ``$XDG_STATE_HOME/multi-llm``
    (``write_safety.claim_external_safety_dir``, review fixes 2026-10-05); with
    the variable unset that is the developer's own ``~/.local/state``.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    yield


@pytest.fixture(autouse=True)
def _isolate_cli_version_cache(tmp_path, monkeypatch):
    """Keep doctor's version cache (``utils/cli_version_cache.py``) under each
    test's tmp dir, never the developer's ``~/.cache/multi-llm``.

    ``XDG_CACHE_HOME`` carries the redirect into child interpreters (a
    subprocess doctor run); the module patch holds it in-process even for a
    test that resets that variable. The variable also moves uv's cache, so on
    POSIX ``UV_CACHE_DIR`` is pinned to the one uv would have used (uv's
    Windows cache ignores ``XDG_CACHE_HOME``). It sits beside ``tmp_path``,
    not in it, because tests that ``chdir`` there assert that doctor leaves
    the working directory alone; nothing is created until a doctor run writes.
    """
    from utils import cli_version_cache

    if os.name != "nt" and not os.environ.get("UV_CACHE_DIR"):
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        monkeypatch.setenv("UV_CACHE_DIR", os.path.join(base, "uv"))
    home = tmp_path.parent / f"{tmp_path.name}.xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(home))
    target = str(home / "multi-llm")
    monkeypatch.setattr(cli_version_cache, "cache_dir", lambda: target)
    yield


#: The test files that run with ``MULTI_LLM_STRICT_ADVISORY=1``: an exception an
#: advisory read of ``run_status`` (or a ``status_*`` module beside it),
#: ``detach`` or ``next_phase`` swallows (``utils/advisory.py``) fails the
#: test instead of vanishing. A test in them
#: that feeds a malformed input on purpose asserts the swallow with
#: :func:`lenient_advisory` (docs/development/runtime.md#advisory-guards).
STRICT_ADVISORY_FILES = frozenset({
    "test_detach.py",
    "test_next_command_emitters.py",
    "test_next_phase.py",
    "test_status_after_a_stop.py",
    "test_status_after_switching_to_native.py",
    "test_status_file.py",
    "test_status_native_stop_head.py",
    "test_status_phase_dirs.py",
    "test_status_recorded_route.py",
    "test_status_run_view_every_leg.py",
    "test_status_tallies_every_task.py",
    "test_status_why_punctuation.py",
})


@pytest.fixture(autouse=True)
def _advisory_mode(request, monkeypatch):
    """Strict advisory guards for :data:`STRICT_ADVISORY_FILES`, and the
    default (silent) everywhere else, whatever the developer's shell exports:
    a ``MULTI_LLM_DEBUG=1`` left set would put tracebacks on the stderr that
    tests compare."""
    monkeypatch.delenv("MULTI_LLM_DEBUG", raising=False)
    monkeypatch.delenv("MULTI_LLM_STRICT_ADVISORY", raising=False)
    if request.path.name in STRICT_ADVISORY_FILES:
        monkeypatch.setenv("MULTI_LLM_STRICT_ADVISORY", "1")
    yield


def stand_in(double):
    """Make *double* a stand-in for a production callable called with keywords.

    Production never inspects a double's signature: it passes every keyword
    it has, unconditionally (a hook that is off is ``None``). So every double
    takes ``**_`` — and this is the one place that rule lives. *double*
    receives the keywords its own signature names; the rest are accepted and
    dropped. A double that already takes ``**kwargs`` is returned as is::

        @stand_in
        def fake_dispatch(task, *, phase_dir, runner, after_attempt=None):
            ...

        monkeypatch.setattr(executor, "_run_probe", stand_in(lambda argv: done))
    """
    import functools
    import inspect

    params = inspect.signature(double).parameters.values()
    if any(param.kind is param.VAR_KEYWORD for param in params):
        return double
    named = {
        param.name for param in params
        if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
    }

    @functools.wraps(double)
    def accepting(*args, **kwargs):
        return double(*args, **{k: v for k, v in kwargs.items() if k in named})

    return accepting


@pytest.fixture
def lenient_advisory(monkeypatch):
    """The production (silent) guard inside a strict file, for a test that
    feeds a malformed input on purpose. Yields the labels of the sites that
    swallowed something, in order, so the test asserts the swallow it expects
    rather than merely tolerating any. In-process only: a child process's
    swallow is not recorded."""
    import importlib

    from utils import advisory

    monkeypatch.delenv("MULTI_LLM_STRICT_ADVISORY", raising=False)
    labels: List[str] = []
    real = advisory.note_swallowed

    def record(label, exc):
        labels.append(label)
        real(label, exc)

    for name in ("advisory", "run_status", "status_views", "status_commands",
                 "status_native", "status_text", "detach", "next_phase"):
        monkeypatch.setattr(importlib.import_module(f"utils.{name}"), "note_swallowed", record)
    yield labels


@pytest.fixture(autouse=True)
def _pin_opencode_debug_config(request, monkeypatch):
    """Keep the default suite from spawning `opencode debug config` (plan §8).

    The reader behind `_user_defines_provider` (T8.5), the auth leg's config
    source (T8.3) and doctor's `config-only` row (T8.6) spawns the real binary
    and would read the developer's own opencode config. Pinned to a failed read
    (``None``, i.e. "unknown": every caller keeps its pre-existing verdict, and
    `_user_defines_provider` falls back to its file read) for every test that
    does not opt out with ``@pytest.mark.debug_config``. The per-process memo
    starts every test empty (`_clear_process_caches`), so one read never leaks
    into the next.
    """
    try:
        from utils.providers import opencode as oc
    except Exception:
        yield
        return
    if "debug_config" not in request.keywords and hasattr(oc, "_spawn_debug_config"):
        monkeypatch.setattr(
            oc, "_spawn_debug_config", lambda argv, env, cwd, timeout: None
        )
    yield


@pytest.fixture(autouse=True)
def _no_real_codex_trust_bookkeeping(monkeypatch):
    """Keep the suite from arming the developer's real ``~/.codex`` for the
    I12 trust-entry restore (plan codex-sandbox-unavailable §5, option B):
    codex's adapter arms ``utils.codex_trust`` on every dispatch context, and
    ``multi_llm.main``'s ``finally`` restores what it armed. Off by default;
    ``tests/test_codex_trust.py`` turns it on against a temp home."""
    try:
        from utils import codex_trust
    except Exception:
        yield
        return
    monkeypatch.setattr(codex_trust, "ENABLED", False)
    codex_trust.reset()
    yield
    codex_trust.reset()


def _canonical_file(path: str) -> str:
    """``path`` with every symlink resolved, case-folded where the OS folds case
    (macOS ``/var`` → ``/private/var``, Windows 8.3 names)."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


class _WrapperIdentity(NamedTuple):
    """What makes a file a production Landlock wrapper, whatever its name or
    place: its canonical path, its inode (a hard link or symlink shares it) or
    its bytes (a copy, a generated package's copy share them)."""

    paths: frozenset
    inodes: frozenset
    digests: frozenset
    sizes: frozenset
    names: frozenset


def _production_wrapper_files() -> List[str]:
    """Every copy of the production wrapper this checkout holds.

    The copy beside ``landlock_dispatch.__file__`` (the one ``WRAPPER_PATHS``
    is built from), plus the source tree's and each generated package's
    (``packaging/<host>/…/skills/multi-llm/utils``), found by walking up from
    it: a run from either tree must not run the other's, stale or not.
    """
    try:
        from utils import landlock_dispatch
    except Exception:
        return []
    names = {os.path.basename(p) for p in landlock_dispatch.WRAPPER_PATHS.values()}
    here = os.path.dirname(os.path.abspath(landlock_dispatch.__file__))
    payload = os.path.join("skills", "multi-llm", "utils")
    dirs = [here]
    ancestor = here
    for _ in range(8):
        parent = os.path.dirname(ancestor)
        if parent == ancestor:
            break
        ancestor = parent
        dirs.append(os.path.join(ancestor, payload))
        packaging = os.path.join(glob.escape(ancestor), "packaging", "*")
        dirs.extend(glob.glob(os.path.join(packaging, payload)))
        dirs.extend(glob.glob(os.path.join(packaging, "multi-llm", payload)))
    return sorted({
        os.path.join(d, n) for d in dirs for n in names
        if os.path.isfile(os.path.join(d, n))
    })


def _production_wrapper_identity() -> _WrapperIdentity:
    paths, inodes, digests, sizes, names = set(), set(), set(), set(), set()
    for path in _production_wrapper_files():
        try:
            info = os.stat(path)
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        paths.add(_canonical_file(path))
        if info.st_ino:
            inodes.add((info.st_dev, info.st_ino))
        digests.add(hashlib.sha256(data).hexdigest())
        sizes.add(len(data))
        names.add(os.path.basename(path))
    return _WrapperIdentity(*(frozenset(x) for x in (paths, inodes, digests, sizes, names)))


#: Taken at import, before any test can patch ``WRAPPER_PATHS``: tests point
#: that table at their fakes with ``monkeypatch.setitem``, and a call-time read
#: would classify every fake as production.
_PRODUCTION_WRAPPERS = _production_wrapper_identity()

#: (canonical path, size, mtime) → whether that file's bytes are a production
#: copy, so a file is hashed once per version, not once per probe.
_WRAPPER_DIGESTS: Dict[Tuple[str, int, int], bool] = {}


def _is_production_wrapper(path: str) -> bool:
    """Whether the regular file at ``path`` (symlinks followed) is a production
    wrapper by inode, canonical path or bytes (size first, then a hash)."""
    identity = _PRODUCTION_WRAPPERS
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return False
    if not stat.S_ISREG(info.st_mode):
        return False
    if info.st_ino and (info.st_dev, info.st_ino) in identity.inodes:
        return True
    canonical = _canonical_file(path)
    if canonical in identity.paths:
        return True
    if info.st_size not in identity.sizes:
        return False
    key = (canonical, info.st_size, info.st_mtime_ns)
    known = _WRAPPER_DIGESTS.get(key)
    if known is None:
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            return False
        known = _WRAPPER_DIGESTS[key] = digest in identity.digests
    return known


def _under(path: str, root: str) -> bool:
    """Whether ``path`` lies under ``root`` (already canonical).

    The DIRECTORY is resolved, never the file: a mock launcher is often a
    symlink from ``tmp_path/bin`` into the harness, and the PATH entry is the
    fact that matters.
    """
    absolute = os.path.abspath(path)
    resolved = os.path.join(
        os.path.realpath(os.path.dirname(absolute)), os.path.basename(absolute)
    )
    return os.path.normcase(resolved).startswith(root + os.sep)


#: Programs whose code is text the guard cannot follow to a file
#: (``sh -c 'exec "$PY" "$WRAPPER"'``), in any form.
_SHELLS = frozenset({
    "sh", "bash", "dash", "zsh", "ksh", "mksh", "ash", "yash", "busybox",
    "fish", "csh", "tcsh", "cmd", "powershell", "pwsh",
})
#: Launchers that run another command from their argv (``env -S``, ``env -C``
#: and their own option values included): an indirection, treated as a shell.
_PREFIX_LAUNCHERS = frozenset({
    "env", "nice", "nohup", "setsid", "stdbuf", "timeout", "ionice", "taskset",
    "time", "xargs",
})
_PYTHON = re.compile(r"(?:python|pythonw|pypy)[0-9.]*t?|py")
_SHELL_VARIABLE = re.compile(r"\$(?:\{(\w+)\}|(\w+))")


def _program_names(*words: str) -> Set[str]:
    """The names ``words`` run as: each one's own and, for a symlink, its
    target's (``/bin/sh`` → ``dash``), lower-cased, without ``.exe``."""
    names: Set[str] = set()
    for word in words:
        if not word:
            continue
        for candidate in (word, os.path.realpath(word)):
            name = os.path.basename(candidate).lower()
            names.add(name[:-4] if name.endswith(".exe") else name)
    return names


def _python_runs_text(options: Sequence[str]) -> bool:
    """Whether a Python command line runs ``-c`` text or a ``-m`` module rather
    than a script file; its options end at the first word that is not one."""
    takes_value = False
    for word in options:
        if takes_value:
            takes_value = False
            continue
        if not word.startswith("-") or word in ("-", "--"):
            return False
        if word.startswith("--"):
            takes_value = word == "--check-hash-based-pycs"
            continue
        for position, letter in enumerate(word[1:], 1):
            if letter in "cm":
                return True
            if letter in "WX":
                takes_value = position == len(word) - 1
                break
    return False


def _shell_form(words: Sequence[str], launcher: Sequence[str]) -> bool:
    """Whether ``words`` launch program text or another command line rather
    than a file the guard can identify: a shell or a prefix launcher
    (``env``, ``timeout``, …) in any form, or Python ``-c``/``-m``."""
    names = _program_names(words[0], *launcher[:1])
    if names & (_SHELLS | _PREFIX_LAUNCHERS):
        return True
    return any(_PYTHON.fullmatch(name) for name in names) and _python_runs_text(words[1:])


def _shell_text_words(words: Sequence[str], env: Optional[Dict[str, str]]) -> List[str]:
    """The paths a shell-form probe reaches without naming them as a word:
    each word's shell tokens with ``$NAME``/``${NAME}`` expanded from the
    child's environment, and those expansions split again (``eval "$CMD"``)."""
    child_env = os.environ if env is None else env

    def expand(token: str) -> str:
        return _SHELL_VARIABLE.sub(
            lambda m: str(child_env.get(m.group(1) or m.group(2), m.group(0))), token
        )

    found: List[str] = []
    for word in words:
        try:
            tokens = shlex.split(word)
        except ValueError:
            tokens = word.split()
        for token in tokens:
            expanded = expand(token)
            found.append(expanded)
            if expanded != token:
                found.extend(expanded.split())
    return found


#: A Python module argument, bare or glued to its flag (``-mutils.x``).
_MODULE_WORD = re.compile(r"(?:-[A-Za-z]*m)?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")


def _mentions_wrapper(tokens: Sequence[str], names: Set[str]) -> bool:
    """Whether program text names a wrapper file (``…/landlock_exec.py``) or
    module (``-m utils.landlock_exec``) at all: text that builds the path out
    of pieces no expansion here resolves is still caught by the name."""
    stems = {os.path.splitext(name)[0] for name in names}
    for token in tokens:
        if any(name in token for name in names):
            return True
        match = _MODULE_WORD.fullmatch(token)
        if match and match.group(1).rsplit(".", 1)[-1] in stems:
            return True
    return False


def _current_wrapper_names() -> Set[str]:
    try:
        from utils import landlock_dispatch
    except Exception:
        return set()
    return {os.path.basename(str(p)) for p in landlock_dispatch.WRAPPER_PATHS.values()}


def _probe_guard_verdict(
    argv,
    *,
    cwd,
    temp_root: str,
    env: Optional[Dict[str, str]] = None,
    launcher: Sequence[str] = (),
    shell_ok: bool = False,
) -> Optional[str]:
    """What the probe guard does with one probe ``argv``, before its other rules.

    ``"wrapper"`` (answered exit 126, no usable Landlock): some word reaches a
    production wrapper by identity (any copy, hard link or symlink, any name,
    any position), or names a file called like a wrapper (the production names,
    or whatever ``WRAPPER_PATHS`` holds now) that does not resolve under
    ``temp_root``. Relative words resolve against the child's ``cwd``, as the
    child opens them. ``launcher`` is ``argv[0]`` as the spawn resolves it.
    A :func:`_shell_form` is checked through its shell tokens and the
    variables they expand too, and is ``"wrapper"`` when its text mentions a
    wrapper's file or module name at all. ``"shell"``: any other shell form in
    a test that did not opt in with ``@pytest.mark.shell_probe``. ``None``:
    the guard's other rules decide, and a test's own fake wrapper under
    ``tmp_path`` runs for real.
    """
    words = [os.fspath(a) if isinstance(a, os.PathLike) else str(a) for a in argv]
    if not words:
        return None
    base = os.fspath(cwd) if cwd is not None else os.getcwd()
    shell = _shell_form(words, [str(w) for w in launcher])
    text = _shell_text_words(words, env) if shell else []
    candidates = [*(str(w) for w in launcher), *words, *text]
    names = _PRODUCTION_WRAPPERS.names | _current_wrapper_names()
    for word in candidates:
        if not word:
            continue
        path = word if os.path.isabs(word) else os.path.join(base, word)
        if _is_production_wrapper(path):
            return "wrapper"
        named = os.path.basename(path) in names or (
            os.path.isfile(path) and os.path.basename(os.path.realpath(path)) in names
        )
        if named and not _canonical_file(path).startswith(temp_root + os.sep):
            return "wrapper"
    if shell and _mentions_wrapper([*words, *text], names):
        return "wrapper"
    if shell and not shell_ok:
        return "shell"
    return None


#: The shell-form probes the guard refused in the running test (stash key on
#: the test's node); a non-empty list fails the test at teardown.
PROBE_GUARD_REFUSALS = pytest.StashKey[List[List[str]]]()


@pytest.fixture(autouse=True)
def _pin_real_sandbox_probes(request, monkeypatch):
    """Keep the default suite from probing a developer's real provider CLI (D6).

    The review fan-out now asks each slot's adapter for its ``generate`` sandbox
    self-test before dispatching, and codex answers with its own
    ``codex sandbox …`` argv. A test that names ``codex:gpt-5.5`` with a mocked
    invoker never installs a launcher, so the probe would resolve to whatever
    ``codex`` the developer has on PATH — and its verdict would then decide the
    test (refused on a machine whose bubblewrap cannot start, dispatched on one
    where it can). ``runner_env`` already guarantees "this suite must never be
    able to reach a real provider CLI"; this extends that guarantee to the
    probe. A probe whose argv names a registered provider command that resolves
    OUTSIDE the temp dir (i.e. not a mock launcher) is answered ``None``, which
    ``sandbox_verdict`` reads as ``unknown`` — dispatch exactly as before the
    gate existed. A *fallback rung's* probe (landlock rung, 2026-09-22) is the
    one exception: ``None`` there would stop the ladder walk and refuse the
    slot, the opposite of "as before". It is answered as a CLI that does not
    know the rung (``unexpected argument``), which the walk reads as *this rung
    is absent* — so the ladder falls through to exactly the pre-ladder bypass,
    and a test that wants rung 1 fakes its probe itself. Mock launchers under ``tmp_path`` run for real.
    The production Landlock wrapper never runs (:func:`_probe_guard_verdict`):
    a probe reaching it in any launch form is answered exit 126. A probe whose
    program is shell text or another command line (``sh -c``, ``python -c``,
    ``env …``) is refused,
    and fails the test at teardown, unless the test opts in with
    ``@pytest.mark.shell_probe``, as the gate's own tests with fake ``sh``
    probes do. The shared per-process probe
    cache starts every test empty (`_clear_process_caches`), so one verdict
    never leaks into the next.
    Opt out with ``@pytest.mark.sandbox_probe``.
    """
    try:
        import utils.subagent_executor as executor
        import utils.review_orchestrator_base as rob
        from utils.llm_client import _resolve_executable
        from utils import provider_registry as registry
    except Exception:
        yield
        return
    # The probe cache both lanes share, its decisions, the version reads that
    # feed the walk's removed_at skip and the retirement lines start empty with
    # every other registered cache (`_clear_process_caches`).
    if "sandbox_probe" in request.keywords:
        yield
        return
    real = executor._run_sandbox_probe
    shell_ok = "shell_probe" in request.keywords
    refused: List[List[str]] = []
    request.node.stash[PROBE_GUARD_REFUSALS] = refused
    # `realpath`, because pytest's `tmp_path` is a resolved path: with `/var` a
    # symlink to `/private/var` (macOS) or a short 8.3 `RUNNER~1` user dir
    # (Windows) the unresolved temp dir is not a prefix of any mock's path.
    temp_root = os.path.normcase(os.path.realpath(tempfile.gettempdir()))

    def _in_temp(path: str) -> bool:
        return _under(path, temp_root)

    def _provider_commands() -> Set[str]:
        names: Set[str] = set()
        for provider in list(getattr(registry, "_PROVIDERS", {}).values()):
            command = getattr(provider, "command", "")
            if command:
                names.add(os.path.basename(str(command)))
        return names or {"codex"}

    def _rung_probes() -> Set[Tuple[str, ...]]:
        from utils.providers.base import (
            SANDBOX_FALLBACK_PROFILES, SANDBOX_WRITE_FALLBACK_PROFILES,
        )

        probes: Set[Tuple[str, ...]] = set()
        for provider in list(getattr(registry, "_PROVIDERS", {}).values()):
            hook = getattr(provider, "sandbox_fallback_rungs", None)
            if not callable(hook):
                continue
            for profile in (*SANDBOX_FALLBACK_PROFILES, *SANDBOX_WRITE_FALLBACK_PROFILES):
                for rung in hook(profile) or ():
                    if rung.probe_argv:
                        probes.add(tuple(rung.probe_argv))
        return probes

    def guarded(argv, **kwargs):
        if argv:
            try:
                launcher, _ = _resolve_executable(str(argv[0]))
            except Exception:
                launcher = []
            verdict = _probe_guard_verdict(
                argv, cwd=kwargs.get("cwd"), env=kwargs.get("env"),
                launcher=launcher, temp_root=temp_root, shell_ok=shell_ok,
            )
            if verdict == "wrapper":
                # The production wrapper (by identity, so a checkout under the
                # temp dir, a copy or a link is covered too), or any wrapper
                # outside the temp dir: answered as a kernel with no usable
                # Landlock (the wrapper's own exit 126), so the walk records
                # the rung `absent` and falls through exactly as before the
                # rung existed. A test that wants the rung points
                # `landlock_dispatch.WRAPPER_PATHS` at a fake under tmp_path,
                # which runs for real.
                return subprocess.CompletedProcess(
                    list(argv), 126, "",
                    "landlock_exec: unavailable: the default suite never runs "
                    "the production Landlock wrapper (plugin-landlock rung "
                    "absent)\n",
                )
            if verdict == "shell":
                refused.append([str(a) for a in argv])
                return subprocess.CompletedProcess(
                    list(argv), 126, "",
                    "probe guard: a shell-form probe runs only in a test marked "
                    "@pytest.mark.shell_probe\n",
                )
        head = os.path.basename(str(argv[0])) if argv else ""
        if head in _provider_commands():
            launcher, _ = _resolve_executable(str(argv[0]))
            # Any word: an npm-style shim (a Windows mock) launches as
            # `node <cli.js>`, whose second word is the one in the temp dir.
            if not any(_in_temp(str(part)) for part in launcher):
                if tuple(str(a) for a in argv) in _rung_probes():
                    return subprocess.CompletedProcess(
                        list(argv), 2, "",
                        "error: unexpected argument (the default suite never "
                        "probes a real CLI's fallback rung)\n",
                    )
                return None
        return real(argv, **kwargs)

    monkeypatch.setattr(executor, "_run_sandbox_probe", guarded)
    yield
    if refused:
        pytest.fail(
            f"the probe guard refused {len(refused)} shell-form sandbox "
            f"probe(s), first {refused[0]!r}: a test whose fake probes are "
            "shell text opts in with @pytest.mark.shell_probe",
            pytrace=False,
        )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_plan(temp_dir):
    """Create a sample plan file for testing."""
    plan_content = """# Sample Implementation Plan

## Overview
This is a sample plan for testing.

## Tasks

### T001: Create directory structure
Create the basic directory structure.
- Depends on: none

### T002: Implement core module
Implement the core functionality.
- Depends on: T001

### T003: Add tests
Write unit tests for the core module.
- Depends on: T002

### T004: Update documentation
Update README and docs.
- Depends on: T002, T003
"""
    plan_path = temp_dir / "sample-plan.md"
    plan_path.write_text(plan_content, encoding="utf-8")
    return plan_path


@pytest.fixture
def sample_suggestions():
    """Sample LLM suggestions for testing."""
    return [
        {
            "title": "Add error handling",
            "desc": "The plan lacks error handling. Add try-catch blocks.",
            "importance": "high",
            "type": "addition",
            "source_model": "model-a"
        },
        {
            "title": "Add error handling for edge cases",
            "desc": "Consider adding error handling for edge cases.",
            "importance": "medium",
            "type": "addition",
            "source_model": "model-b"
        },
        {
            "title": "Improve performance",
            "desc": "Consider caching the results.",
            "importance": "low",
            "type": "improvement",
            "source_model": "model-a"
        }
    ]


@pytest.fixture
def sample_tasks_json():
    """Sample task decomposition JSON."""
    return [
        {
            "id": "T001",
            "title": "Create directory structure",
            "description": "Create the basic directory structure",
            "depends_on": [],
            "files_to_create": ["src/main.py"],
            "estimated_complexity": "low"
        },
        {
            "id": "T002",
            "title": "Implement core module",
            "description": "Implement the core functionality",
            "depends_on": ["T001"],
            "files_to_modify": ["src/main.py"],
            "estimated_complexity": "medium"
        },
        {
            "id": "T003",
            "title": "Add tests",
            "description": "Write unit tests",
            "depends_on": ["T002"],
            "files_to_create": ["tests/test_main.py"],
            "estimated_complexity": "medium"
        }
    ]


@pytest.fixture
def sample_code_review_issues():
    """Sample code review issues."""
    return [
        {
            "title": "Missing error handling",
            "desc": "Function lacks try-catch block",
            "importance": "high",
            "file": "src/main.py",
            "line_range": [10, 15],
            "type": "bug"
        },
        {
            "title": "Unused import",
            "desc": "Import 'os' is not used",
            "importance": "low",
            "file": "src/main.py",
            "line_range": [1, 1],
            "type": "style"
        }
    ]


@pytest.fixture
def sample_state(temp_dir, sample_plan):
    """Create a sample state file."""
    state = {
        "schema_version": "1.0",
        "plan_path": str(sample_plan),
        "plan_hash": "abc123",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
        "head_at_start": "abc123def",
        "branch_name": "feature/test",
        "review_phase_completed": False,
        "tracked_files": [],
        "task_status": {}
    }
    state_path = temp_dir / "sample-plan_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path


class MockSubprocess:
    """Mock subprocess.run with configurable responses."""

    def __init__(self):
        self.calls_log = []
        self._responses = {}
        self._exceptions = {}
        self._default_response = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

    def set_response(self, cmd_pattern, returncode=0, stdout="", stderr=""):
        """Set response for commands matching pattern."""
        self._responses[cmd_pattern] = subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr
        )

    def set_exception(self, pattern, exception):
        """Set exception to raise for commands matching pattern."""
        self._exceptions[pattern] = exception

    def __call__(self, cmd, *args, **kwargs):
        """Mock subprocess.run call."""
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        self.calls_log.append({"cmd": cmd, "args": args, "kwargs": kwargs})

        # Check for exceptions first
        for pattern, exc in self._exceptions.items():
            if re.search(pattern, cmd_str):
                raise exc

        # Check for configured responses
        for pattern, response in self._responses.items():
            if re.search(pattern, cmd_str):
                return response

        return self._default_response


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run to prevent real git/LLM process execution."""
    mock = MockSubprocess()
    with patch("subprocess.run", mock):
        yield mock


class MockGitRepo:
    """Mock git repository with configurable state."""

    def __init__(self, mock_subprocess, tmp_path):
        self._mock_subprocess = mock_subprocess
        self._tmp_path = tmp_path
        self._branch = "main"
        self._dirty_files = []
        self._staged_files = []

        # Create basic git repo structure
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (tmp_path / ".git" / "config").write_text("[core]\n\trepositoryformatversion = 0\n", encoding="utf-8")

        # Set default git responses
        self._update_git_responses()

    def _update_git_responses(self):
        """Update mock responses based on current state."""
        # Status output
        status_lines = []
        for f in self._staged_files:
            status_lines.append(f"A  {f}")
        for f in self._dirty_files:
            status_lines.append(f" M {f}")
        status_output = "\n".join(status_lines) if status_lines else ""

        self._mock_subprocess.set_response(
            r"git\s+status", returncode=0, stdout=status_output
        )
        self._mock_subprocess.set_response(
            r"git\s+branch", returncode=0, stdout=f"* {self._branch}\n"
        )
        self._mock_subprocess.set_response(
            r"git\s+rev-parse", returncode=0, stdout="abc123def456\n"
        )

    def set_clean(self):
        """Set repository to clean state (no uncommitted changes)."""
        self._dirty_files = []
        self._staged_files = []
        self._update_git_responses()

    def set_dirty(self, files):
        """Set repository to have uncommitted changes."""
        self._dirty_files = list(files)
        self._update_git_responses()

    def set_staged(self, files):
        """Set repository to have staged files."""
        self._staged_files = list(files)
        self._update_git_responses()

    def set_branch(self, name):
        """Set current branch name."""
        self._branch = name
        self._update_git_responses()

    @property
    def path(self):
        """Return the repository path."""
        return self._tmp_path


@pytest.fixture
def mock_git_repo(mock_subprocess, tmp_path):
    """Provide a minimal mock git repository layout."""
    return MockGitRepo(mock_subprocess, tmp_path)


class MockProviderAvailable:
    """Mock provider availability with configurable providers."""

    def __init__(self, patcher):
        self._patcher = patcher
        self._available_providers = set()

    def set_available(self, providers):
        """Set which providers are available."""
        self._available_providers = set(providers)

    def which(self, name):
        """Mock shutil.which behavior."""
        if name in self._available_providers:
            return f"/usr/bin/{name}"
        return None


@pytest.fixture
def mock_provider_available():
    """Stub provider availability checks to return fake paths."""
    mock = MagicMock()
    available_providers = {"claude", "cursor-agent", "gemini", "openai"}

    def which_side_effect(name):
        if name in available_providers:
            return f"/usr/bin/{name}"
        return None

    mock.side_effect = which_side_effect

    controller = MockProviderAvailable(mock)
    controller.set_available(available_providers)

    def patched_which(name):
        return controller.which(name)

    with patch("shutil.which", patched_which):
        yield controller


@pytest.fixture
def mock_provider_unavailable():
    """Stub providers as unavailable (shutil.which returns None)."""
    with patch("shutil.which", return_value=None):
        yield


@pytest.fixture
def truncated_json():
    """Return dict of truncated JSON test cases."""
    return {
        "mid_object": '{"title": "Test", "desc": "Some description',
        "mid_array": '[{"title": "Test"}, {"title": "Another"',
        "mid_string": '{"title": "Test with truncated string...',
        "unbalanced_brackets": '{"items": [{"nested": {"deep": true}',
        "trailing_comma": '{"title": "Test", "items": [1, 2, 3,',
    }


@pytest.fixture
def invalid_enum_value():
    """Return dict of invalid enum value test cases."""
    return {
        "invalid_status": {
            "title": "Test",
            "desc": "Description",
            "importance": "high",
            "status": "maybe_valid",  # Invalid: should be valid/invalid/needs-human-decision
        },
        "invalid_importance": {
            "title": "Test",
            "desc": "Description",
            "importance": "critical",  # Invalid: should be high/medium/low
            "status": "valid",
        },
        "invalid_type": {
            "title": "Test",
            "desc": "Description",
            "importance": "high",
            "type": "enhancement",  # Invalid: should be addition/modification/removal/clarification/bug/style/security/performance
        },
    }


@pytest.fixture
def cursor_agent_empty_result():
    """Return cursor-agent empty result variants."""
    return {
        "null_result": {"result": None},
        "empty_string_result": {"result": ""},
        "whitespace_result": {"result": "   \n\t  "},
        "missing_result_field": {"status": "ok", "data": []},
        "wrong_type_field": {"result": 12345},
    }


# ==============================================================================
# E2E Integration Test Fixtures
# ==============================================================================


@pytest.fixture
def fixture_manager(tmp_path: Path) -> FixtureManager:
    """Create a FixtureManager for managing test fixtures in isolated tmp_path.

    Provides methods for:
    - Creating plan files in the test directory
    - Loading plans from fixtures/e2e/plans/
    - Loading response fixtures from fixtures/e2e/responses/
    - Creating pre-populated test states for specific phases

    Usage:
        def test_example(fixture_manager):
            plan = fixture_manager.create_plan("my-plan", "# Plan content")
            # or
            plan = fixture_manager.load_plan("auth-feature")
    """
    return FixtureManager(tmp_path)


@pytest.fixture
def mock_provider(tmp_path: Path) -> MockProvider:
    """Create a MockProvider for configuring mock LLM behavior.

    Provides methods for:
    - Setting scenarios for response matching
    - Setting direct responses or fixtures
    - Injecting failures or timeouts
    - Reading call logs

    Usage:
        def test_example(mock_provider, skill_runner):
            mock_provider.set_scenario("happy_path")
            # or
            mock_provider.set_response({"key": "value"})
    """
    return MockProvider(tmp_path)


@pytest.fixture
def skill_runner(tmp_path: Path, mock_provider: MockProvider) -> SkillRunner:
    """Create a SkillRunner with mock LLM binaries configured.

    The runner:
    - Creates mock binary symlinks in tmp_path/bin/
    - Prepends bin/ to PATH to intercept provider commands
    - Sets MULTI_LLM_TEST_MODE=1
    - Configures mock call logging

    Mock provider configuration can be updated via the mock_provider fixture
    before running orchestrators.

    Usage:
        def test_example(skill_runner, fixture_manager):
            plan = fixture_manager.create_plan("test", "# Content")
            result = skill_runner.run_orchestrator("review_plan", plan.plan_path)
            assert result.success
            assert result.mock_was_invoked()
    """
    # Create runner with mock provider's environment configuration
    runner = SkillRunner(
        tmp_path=tmp_path,
        extra_env=mock_provider.get_env(),
    )
    return runner


@pytest.fixture
def skill_runner_live(tmp_path: Path) -> SkillRunner:
    """Create a SkillRunner for live tests that call real LLM providers.

    This fixture is for optional live integration tests that require
    the Claude CLI and actual API access. Tests using this fixture
    should be marked with @pytest.mark.live.

    The fixture skips the test if MULTI_LLM_LIVE_MODE environment
    variable is not set to "1".

    Usage:
        @pytest.mark.live
        def test_live_example(skill_runner_live, fixture_manager):
            plan = fixture_manager.create_plan("test", "# Content")
            result = skill_runner_live.run_orchestrator("review_plan", plan.plan_path)
            assert result.success
    """
    if os.environ.get("MULTI_LLM_LIVE_MODE") != "1":
        pytest.skip(
            "Live mode tests are skipped by default. "
            "Set MULTI_LLM_LIVE_MODE=1 to run live tests."
        )

    # Create runner without mock binaries (real PATH)
    # Don't set up mock symlinks or inject mock environment
    runner = SkillRunner.__new__(SkillRunner)
    runner.tmp_path = Path(tmp_path)
    runner.bin_dir = runner.tmp_path / "bin"
    runner.call_log_path = runner.tmp_path / "live_calls.jsonl"
    runner.mock_llm_path = None
    runner.skill_dir = Path(__file__).parent.parent
    runner.scenario_path = None
    runner.extra_env = {}

    # Create bin_dir but don't set up mock symlinks
    runner.bin_dir.mkdir(parents=True, exist_ok=True)

    return runner


@pytest.fixture
def assertions() -> AssertionHelpers:
    """Create an AssertionHelpers instance for custom test assertions.

    Provides domain-specific assertion methods for verifying:
    - State phase completion/skipping
    - Output directory structure
    - Mock LLM call patterns
    - JSON file validity
    - Salvage file creation
    - stdout/stderr content
    - Exit codes

    Usage:
        def test_example(skill_runner, fixture_manager, assertions):
            plan = fixture_manager.create_plan("test", "# Content")
            result = skill_runner.run_orchestrator("review_plan", plan.plan_path)

            assertions.assert_state_phase_completed(result, "review-plan")
            assertions.assert_mock_was_invoked(result)
    """
    return AssertionHelpers()


class _WriteTracker:
    """Internal class to track file writes during tests."""

    def __init__(self, tmp_path: Path, allowed_paths: Optional[Set[Path]] = None):
        self.tmp_path = tmp_path
        self.allowed_paths = allowed_paths or set()
        self.written_paths: Set[Path] = set()
        self.violations: Set[Path] = set()
        self._original_open = None
        self._original_write_text = None
        self._original_write_bytes = None

    def _is_allowed_path(self, path: Path) -> bool:
        """Check if a path is within allowed directories."""
        try:
            resolved = Path(path).resolve()
            tmp_resolved = self.tmp_path.resolve()

            # Allow writes within tmp_path
            if str(resolved).startswith(str(tmp_resolved)):
                return True

            # Allow writes to /tmp (pytest creates tmp_path there)
            if str(resolved).startswith("/tmp"):
                return True

            # Allow writes to common pytest/coverage paths
            if ".pytest_cache" in str(resolved):
                return True
            if ".coverage" in str(resolved):
                return True

            # Allow writes to explicitly allowed paths
            for allowed in self.allowed_paths:
                allowed_resolved = allowed.resolve()
                if str(resolved).startswith(str(allowed_resolved)):
                    return True

            return False
        except (OSError, ValueError):
            # If we can't resolve the path, be conservative
            return False

    def track_write(self, path: Path) -> None:
        """Track a write operation and check for violations."""
        self.written_paths.add(path)
        if not self._is_allowed_path(path):
            self.violations.add(path)

    def get_violations(self) -> Set[Path]:
        """Return all write violations."""
        return self.violations

    def add_allowed_path(self, path: Path) -> None:
        """Add an allowed path for writes."""
        self.allowed_paths.add(path)


@pytest.fixture
def verify_no_external_writes(
    tmp_path: Path, request
) -> Generator[_WriteTracker, None, None]:
    """Fixture to verify tests don't write outside tmp_path.

    This fixture tracks file write operations during tests and fails
    if any writes occur outside the test's tmp_path directory.

    NOTE: This fixture is NOT autouse - it must be explicitly requested
    by tests that want this protection. E2E tests should use this fixture
    to ensure test isolation.

    Allowed write locations:
    - The test's tmp_path directory
    - /tmp (standard temp directory)
    - .pytest_cache directories
    - .coverage files
    - Any paths added via tracker.add_allowed_path()

    Note: This is a best-effort check using Path.write_text/write_bytes
    monitoring. Some low-level writes may not be detected.

    Usage:
        def test_isolated(verify_no_external_writes, tmp_path):
            # Test code here - will fail if writes occur outside tmp_path
            pass

        def test_with_exceptions(verify_no_external_writes, tmp_path):
            # Allow writes to a specific directory
            verify_no_external_writes.add_allowed_path(Path("/some/path"))
            # Test code here
            pass
    """
    # Skip enforcement for live tests (they may need external writes)
    if request.node.get_closest_marker("live"):
        yield _WriteTracker(tmp_path)
        return

    tracker = _WriteTracker(tmp_path)

    # Store original methods
    original_write_text = Path.write_text
    original_write_bytes = Path.write_bytes

    def tracked_write_text(self, data, *args, **kwargs):
        tracker.track_write(self)
        return original_write_text(self, data, *args, **kwargs)

    def tracked_write_bytes(self, data, *args, **kwargs):
        tracker.track_write(self)
        return original_write_bytes(self, data, *args, **kwargs)

    # Patch Path methods
    Path.write_text = tracked_write_text
    Path.write_bytes = tracked_write_bytes

    try:
        yield tracker
    finally:
        # Restore original methods
        Path.write_text = original_write_text
        Path.write_bytes = original_write_bytes

        # Check for violations
        violations = tracker.get_violations()
        if violations:
            violation_list = "\n  - ".join(str(p) for p in violations)
            pytest.fail(
                f"Test wrote files outside tmp_path:\n  - {violation_list}\n"
                f"Tests should only write to tmp_path: {tmp_path}"
            )
