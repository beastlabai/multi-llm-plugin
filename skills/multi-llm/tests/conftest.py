"""Pytest fixtures for multi-llm skill tests."""

import json
import os
import pytest
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Generator, Optional, Set, Tuple
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
def _pin_opencode_debug_config(request, monkeypatch):
    """Keep the default suite from spawning `opencode debug config` (plan §8).

    The reader behind `_user_defines_provider` (T8.5), the auth leg's config
    source (T8.3) and doctor's `config-only` row (T8.6) spawns the real binary
    and would read the developer's own opencode config. Pinned to a failed read
    (``None``, i.e. "unknown": every caller keeps its pre-existing verdict, and
    `_user_defines_provider` falls back to its file read) for every test that
    does not opt out with ``@pytest.mark.debug_config``. The per-process memo
    is cleared around every test so one read never leaks into the next.
    """
    try:
        from utils.providers import opencode as oc
    except Exception:
        yield
        return
    memo = getattr(oc, "_USER_CONFIG_MEMO", None)
    if memo is not None:
        memo.clear()
    if "debug_config" not in request.keywords and hasattr(oc, "_spawn_debug_config"):
        monkeypatch.setattr(
            oc, "_spawn_debug_config", lambda argv, env, cwd, timeout: None
        )
    yield
    if memo is not None:
        memo.clear()


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
    and a test that wants rung 1 fakes its probe itself. Mock launchers under ``tmp_path`` and the fake ``sh`` probes
    the gate's own tests build run for real. The shared per-process probe
    cache is reset around every test so one verdict never leaks into the next.
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
    # The one probe cache both lanes share (plan §3.6), with its decisions,
    # version reads and once-per-process retirement lines.
    for name in (
        "_SANDBOX_PROBE_CACHE", "_SANDBOX_PROBE_KEY_LOCKS", "_SANDBOX_DECISIONS",
        "_PROBE_VERSION_CACHE",
        # The version preflight's cache feeds the walk's removed_at skip
        # (plan §3.2): a version read by an earlier test (the mock answers
        # 0.156.1) must not skip rung 1 in a later one.
        "_VERSION_CACHE",
    ):
        if hasattr(executor, name):
            monkeypatch.setattr(executor, name, {})
    if hasattr(executor, "_RETIREMENTS_ANNOUNCED"):
        monkeypatch.setattr(executor, "_RETIREMENTS_ANNOUNCED", set())
    if "sandbox_probe" in request.keywords:
        yield
        return
    real = executor._run_sandbox_probe
    temp_root = os.path.abspath(tempfile.gettempdir())

    def _provider_commands() -> Set[str]:
        names: Set[str] = set()
        for provider in list(getattr(registry, "_PROVIDERS", {}).values()):
            command = getattr(provider, "command", "")
            if command:
                names.add(os.path.basename(str(command)))
        return names or {"codex"}

    def _rung_probes() -> Set[Tuple[str, ...]]:
        from utils.providers.base import SANDBOX_FALLBACK_PROFILES

        probes: Set[Tuple[str, ...]] = set()
        for provider in list(getattr(registry, "_PROVIDERS", {}).values()):
            hook = getattr(provider, "sandbox_fallback_rungs", None)
            if not callable(hook):
                continue
            for profile in SANDBOX_FALLBACK_PROFILES:
                for rung in hook(profile) or ():
                    if rung.probe_argv:
                        probes.add(tuple(rung.probe_argv))
        return probes

    def guarded(argv, **kwargs):
        head = os.path.basename(str(argv[0])) if argv else ""
        if head in _provider_commands():
            launcher, _ = _resolve_executable(str(argv[0]))
            # `abspath`, not `realpath`: a mock launcher is often a symlink
            # from `tmp_path/bin` into the harness, and the PATH entry is the
            # fact that matters.
            resolved = os.path.abspath(str(launcher[0])) if launcher else ""
            if not resolved.startswith(temp_root + os.sep):
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
