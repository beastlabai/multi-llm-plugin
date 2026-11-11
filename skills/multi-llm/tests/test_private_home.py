"""The private config roots adapters hand their CLIs (``utils/providers/private_home.py``).

copilot, kimi, qwen-code and vibe each create one ``0700`` root per spawn
through one :class:`PrivateHomes`, and remove only the roots their own
process created. The per-adapter isolation tests
(``test_*_home_isolation.py``) pin what each CLI sees; this file pins the
shared lifecycle, and the fork guard on opencode's shared config home.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import process_control  # noqa: E402
from utils.providers import copilot, kimi, opencode, private_home, qwen_code, vibe  # noqa: E402
from utils.providers.private_home import PrivateHomes  # noqa: E402

ADAPTERS = {"copilot": copilot, "kimi": kimi, "qwen_code": qwen_code, "vibe": vibe}


@pytest.fixture
def homes():
    instance = PrivateHomes("multi-llm-test-private-home-")
    yield instance
    instance.cleanup()


def test_create_makes_a_fresh_private_root_under_the_prefix(homes):
    first, second = homes.create(), homes.create()

    assert first != second
    for path in (first, second):
        assert Path(path).is_dir()
        assert Path(path).name.startswith(homes.prefix)
        assert Path(path).parent == Path(tempfile.gettempdir())
        if os.name != "nt":  # POSIX mode bits: Windows keeps only a read-only flag
            assert Path(path).stat().st_mode & 0o777 == 0o700
    assert homes.isolated_home_dirs() == (first, second)


def test_cleanup_removes_every_root_and_never_raises(homes):
    roots = [homes.create() for _ in range(3)]
    (Path(roots[0]) / "sessions").mkdir()
    (Path(roots[0]) / "sessions" / "s.jsonl").write_text("{}\n", encoding="utf-8")
    shutil.rmtree(roots[1])  # already gone: still no exception

    homes.cleanup()

    assert not any(Path(root).exists() for root in roots)
    assert homes.isolated_home_dirs() == ()
    homes.cleanup()  # idempotent


def test_the_atexit_sweep_is_registered_once_at_the_first_creation(homes, monkeypatch):
    registered = []
    monkeypatch.setattr(private_home.atexit, "register", registered.append)

    homes.create()
    homes.create()

    assert registered == [homes.cleanup]


def test_a_root_that_cannot_be_made_private_is_removed_and_raises(homes, monkeypatch):
    made = []
    real_mkdtemp = private_home.tempfile.mkdtemp

    def _mkdtemp(**kwargs):
        made.append(real_mkdtemp(**kwargs))
        return made[-1]

    def _chmod(_path, _mode):
        raise PermissionError("chmod refused")

    monkeypatch.setattr(private_home.tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setattr(private_home.os, "chmod", _chmod)

    with pytest.raises(OSError):
        homes.create()

    assert made and not Path(made[0]).exists()
    assert homes.isolated_home_dirs() == ()


def test_a_forked_child_never_removes_its_parents_roots(homes):
    parent_root = homes.create()
    homes._pid = os.getpid() + 1  # this process did not create what is tracked

    assert homes.isolated_home_dirs() == ()
    homes.cleanup()
    assert Path(parent_root).is_dir(), "a child removed its parent's root"

    child_root = homes.create()
    assert homes.isolated_home_dirs() == (child_root,)
    homes.cleanup()
    assert not Path(child_root).exists()
    shutil.rmtree(parent_root)


def test_reset_forgets_without_removing(homes):
    root = homes.create()
    homes.reset()
    assert homes.isolated_home_dirs() == ()
    assert Path(root).is_dir()
    shutil.rmtree(root)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_each_adapter_keeps_its_module_names_on_its_own_instance(name):
    module = ADAPTERS[name]
    assert module._HOMES.prefix == module._HOME_PREFIX
    assert module.isolated_home_dirs == module._HOMES.isolated_home_dirs
    assert module.cleanup_home_dirs == module._HOMES.cleanup
    assert module.reset_home_dirs == module._HOMES.reset


def test_the_adapters_share_no_instance_and_no_prefix():
    assert len({id(module._HOMES) for module in ADAPTERS.values()}) == len(ADAPTERS)
    assert len({module._HOME_PREFIX for module in ADAPTERS.values()}) == len(ADAPTERS)


class TestOpencodeSharedHomeForkGuard:
    """opencode's one shared config home per process: a forked child neither
    reuses nor removes the directory its parent made."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        opencode.sweep_config_home()
        yield
        opencode.sweep_config_home()

    def test_a_child_sweep_leaves_the_parents_directory(self, monkeypatch):
        home = opencode.acquire_config_home()
        monkeypatch.setattr(opencode, "_CONFIG_HOME_PID", os.getpid() + 1)

        opencode.sweep_config_home()

        assert Path(home).is_dir(), "a child removed its parent's config home"
        assert opencode._CONFIG_HOME is None
        shutil.rmtree(home)

    def test_a_child_acquire_makes_its_own_directory(self, monkeypatch):
        parent = opencode.acquire_config_home()
        monkeypatch.setattr(opencode, "_CONFIG_HOME_PID", os.getpid() + 1)

        child = opencode.acquire_config_home()

        assert child != parent
        assert opencode._CONFIG_HOME_USERS == 1
        if os.name != "nt":
            assert Path(child).stat().st_mode & 0o777 == 0o700
        opencode.release_config_home()
        assert not Path(child).exists()
        assert Path(parent).is_dir()
        shutil.rmtree(parent)

    def test_the_owning_process_still_sweeps_its_directory(self):
        home = opencode.acquire_config_home()
        assert opencode.acquire_config_home() == home
        opencode.release_config_home()
        assert Path(home).is_dir(), "removed while a slot still holds it"
        opencode.release_config_home()
        assert not Path(home).exists()


# ---------------------------------------------------------------------------
# Lifetime: a root lives as long as the dispatch that made it
# ---------------------------------------------------------------------------


def _owner_fields():
    pid = os.getpid()
    return (
        str(pid),
        private_home._start_token(process_control.sourced_start_time(pid)),
        private_home._namespace_token(process_control.pid_namespace()),
    )


def test_a_start_time_is_recorded_with_its_source(homes):
    source, value = process_control.sourced_start_time(os.getpid())
    assert process_control.process_start_time(os.getpid()) == value
    start = homes._name.match(Path(homes.create()).name).group(2)
    assert start == f"{source}.{private_home._token(value, 'unknown')}"


def test_a_root_is_named_after_its_owner(homes):
    root = Path(homes.create())
    match = homes._name.match(root.name)
    assert match, root.name
    assert match.groups() == _owner_fields()


def test_a_dispatch_releases_the_roots_it_made_and_no_others(homes):
    before = homes.create()
    with private_home.dispatch_homes():
        inside = homes.create()
        with private_home.dispatch_homes():
            nested = homes.create()
        assert not Path(nested).exists(), "the inner dispatch's root outlived it"
        assert Path(inside).is_dir(), "the inner dispatch removed the outer one's root"
    assert not Path(inside).exists()
    assert Path(before).is_dir(), "a root made outside any dispatch was removed"
    assert homes.isolated_home_dirs() == (before,)


def test_concurrent_dispatches_release_only_their_own(homes):
    """Slots run on threads: each dispatch's scope is its own context."""
    barrier = threading.Barrier(2, timeout=30)
    made, errors = {}, []

    def slot(name):
        with private_home.dispatch_homes():
            made[name] = homes.create()
            barrier.wait()  # both roots exist
            if name == "first":
                return
            barrier.wait()  # the first slot has left its dispatch
            if not Path(made[name]).is_dir():
                errors.append("another slot's dispatch removed this root")

    def first():
        slot("first")
        barrier.wait()

    def guarded(target):
        try:
            target()
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(repr(exc))

    threads = [threading.Thread(target=guarded, args=(first,)),
               threading.Thread(target=guarded, args=(lambda: slot("second"),))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert not Path(made["first"]).exists() and not Path(made["second"]).exists()
    assert homes.isolated_home_dirs() == ()


@pytest.mark.skipif(os.name == "nt" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="POSIX permission bits, and root ignores them")
def test_a_root_that_cannot_be_removed_stays_tracked_for_exit(homes):
    root = homes.create()
    locked = Path(root) / "locked"
    locked.mkdir()
    (locked / "state").write_text("x", encoding="utf-8")
    locked.chmod(0)
    try:
        homes.release(root)
        assert Path(root).is_dir()
        assert homes.isolated_home_dirs() == (root,), "the exit pass must try again"
    finally:
        locked.chmod(0o700)
    homes.cleanup()
    assert not Path(root).exists()


def test_release_never_touches_a_root_it_did_not_create(homes, tmp_path):
    other = tmp_path / f"{homes.prefix}1-2-3-abc"
    other.mkdir()
    homes.release(str(other))
    assert other.is_dir()


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_a_dispatch_context_releases_the_adapters_home(name, monkeypatch):
    """Both spawn sites enter ``provider_dispatch_context``; the slot's home is
    gone when it closes, and a home made outside one waits for exit."""
    from utils import llm_client
    from utils.provider_registry import get_provider
    from utils.providers.base import PROFILE_GENERATE

    module = ADAPTERS[name]
    for variable in ("VIBE_HOME", "COPILOT_HOME", "KIMI_CODE_HOME", "QWEN_HOME"):
        monkeypatch.delenv(variable, raising=False)  # a pinned root is never made here
    provider = get_provider(name.replace("_", "-"))
    module.cleanup_home_dirs()
    try:
        with llm_client.provider_dispatch_context(provider, "some-model", PROFILE_GENERATE):
            provider.resolve_env("some-model", PROFILE_GENERATE)
            inside = module.isolated_home_dirs()
            assert len(inside) == 1 and Path(inside[0]).is_dir()
        assert not Path(inside[0]).exists()
        assert module.isolated_home_dirs() == ()

        provider.resolve_env("some-model", PROFILE_GENERATE)
        outside = module.isolated_home_dirs()
        assert len(outside) == 1 and Path(outside[0]).is_dir()
    finally:
        module.cleanup_home_dirs()


# ---------------------------------------------------------------------------
# The sweep: an orphan goes only when its owner is provably dead
# ---------------------------------------------------------------------------


@pytest.fixture
def sweeper():
    return PrivateHomes("multi-llm-test-sweep-")


def _dead_owner():
    """A pid and start token whose process has exited (and been reaped)."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        start = process_control.sourced_start_time(child.pid)
    finally:
        child.kill()
        child.wait()
    return str(child.pid), private_home._start_token(start)


def test_the_sweep_removes_a_dead_owners_root_and_keeps_every_other(sweeper, tmp_path):
    pid, start, namespace = _owner_fields()
    source = start.partition(".")[0]
    other_source = "ps" if source != "ps" else "proc"
    # Another start time from this platform's source marks a reused pid only
    # where that source is clock-independent (Linux's /proc ticks, Windows'
    # FILETIME). macOS reads `ps` text, which follows the reader's time zone and
    # locale, so there a mismatch is inconclusive and the root is kept
    # (test_a_ps_start_time_that_reads_differently_never_sweeps).
    reused_kept = source == process_control.START_SOURCE_PS
    dead_pid, dead_start = _dead_owner()
    cases = {
        "dead": (f"{dead_pid}-{dead_start}-{namespace}-abc1", False),
        "reused pid": (f"{pid}-{source}.123-{namespace}-abc2", reused_kept),
        "alive": (f"{pid}-{start}-{namespace}-abc3", True),
        "unknown start": (f"{pid}-unknown-{namespace}-abc4", True),
        "start without a source": (f"{pid}-123-{namespace}-abc10", True),
        "start from another source": (f"{pid}-{other_source}.123-{namespace}-abc11", True),
        "other namespace": (f"{pid}-{start}-999-abc5", True),
        "dead in another namespace": (f"{dead_pid}-{dead_start}-999-abc6", True),
        "dead in an unknown namespace": (f"{dead_pid}-{dead_start}-unknown-abc12", True),
    }
    for _label, (fields, _kept) in cases.items():
        root = tmp_path / f"{sweeper.prefix}{fields}"
        root.mkdir()
        (root / "session.jsonl").write_text("{}\n", encoding="utf-8")
    legacy = tmp_path / f"{sweeper.prefix}o9jpbgw5"
    legacy.mkdir()
    foreign = tmp_path / f"multi-llm-other-{dead_pid}-{dead_start}-{namespace}-abc7"
    foreign.mkdir()
    a_file = tmp_path / f"{sweeper.prefix}{dead_pid}-{dead_start}-{namespace}-abc8"
    a_file.write_text("not a root", encoding="utf-8")

    removed = sweeper.sweep_orphans(str(tmp_path))

    for label, (fields, kept) in cases.items():
        assert (tmp_path / f"{sweeper.prefix}{fields}").exists() is kept, label
    assert sorted(removed) == sorted(
        str(tmp_path / f"{sweeper.prefix}{fields}") for fields, kept in cases.values() if not kept
    )
    assert legacy.is_dir() and foreign.is_dir() and a_file.is_file()


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_the_sweep_never_follows_a_link(sweeper, tmp_path):
    dead_pid, dead_start = _dead_owner()
    namespace = _owner_fields()[2]
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep").write_text("x", encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    (base / f"{sweeper.prefix}{dead_pid}-{dead_start}-{namespace}-abc9").symlink_to(target)

    assert sweeper.sweep_orphans(str(base)) == []
    assert (target / "keep").is_file()


def test_the_first_creation_in_a_process_sweeps_once(homes, monkeypatch):
    calls = []
    monkeypatch.setattr(homes, "sweep_orphans", lambda *a: calls.append(a) or [])
    homes.create()
    homes.create()
    assert calls == [()]
    homes._swept_pid = os.getpid() + 1  # a forked child sweeps for itself
    homes.create()
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# A root outlives a CLI whose stop is not confirmed
# ---------------------------------------------------------------------------

SKILL_ROOT = Path(__file__).resolve().parent.parent

#: Waits for a line on stdin, then writes into the home it was lent: the
#: session file a CLI still running after its dispatch writes last.
_LATE_WRITER = (
    "import pathlib, sys\n"
    "home = pathlib.Path(sys.argv[1])\n"
    "print('ready', flush=True)\n"
    "sys.stdin.readline()\n"
    "(home / 'session.json').write_text('completed answer')\n"
    "print('wrote', flush=True)\n"
)


@pytest.fixture
def spawned():
    """Children started through ``process_control.popen_tree``, the spawner
    both lanes use; any still running is killed afterwards."""
    children = []

    def spawn(args, **kwargs):
        proc, job = process_control.popen_tree(args, **kwargs)
        children.append((proc, job))
        return proc

    yield spawn
    for proc, job in children:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.communicate(timeout=30)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        if job is not None:
            job.close()


def test_a_cli_still_running_when_its_dispatch_ends_keeps_its_home(homes, spawned):
    with private_home.dispatch_homes():
        home = homes.create()
        cli = spawned([sys.executable, "-c", _LATE_WRITER, home],
                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert cli.stdout.readline().strip() == "ready"

    assert Path(home).is_dir(), "the home of a CLI still running was removed"
    if os.name != "nt":
        assert Path(home).stat().st_mode & 0o777 == 0o700
    assert homes.isolated_home_dirs() == (home,), "kept for the exit pass"
    out, _ = cli.communicate("\n", timeout=30)
    assert out.strip() == "wrote"
    assert (Path(home) / "session.json").read_text(encoding="utf-8") == "completed answer"
    homes.cleanup()
    assert not Path(home).exists()


def test_a_cli_seen_to_exit_releases_its_home_when_the_dispatch_ends(homes, spawned):
    with private_home.dispatch_homes():
        home = homes.create()
        spawned([sys.executable, "-c", "pass"]).wait(timeout=30)
        assert Path(home).is_dir()
    assert not Path(home).exists()
    assert homes.isolated_home_dirs() == ()


@pytest.mark.parametrize("stops, kept", [
    ((False,), True),
    ((False, True), True),  # an unconfirmed stop sticks
    ((True,), False),
])
def test_a_recorded_stop_outranks_the_cli_exit(homes, spawned, stops, kept):
    """A CLI can exit while a process group it started survives the sweep."""
    with private_home.dispatch_homes():
        home = homes.create()
        cli = spawned([sys.executable, "-c", "pass"])
        cli.wait(timeout=30)
        for confirmed in stops:
            process_control.note_stop(cli, confirmed)
    assert Path(home).is_dir() is kept
    assert homes.isolated_home_dirs() == ((home,) if kept else ())


def test_a_confirmed_stop_releases_before_the_cli_is_reaped(homes, spawned):
    """A kill the spawner confirmed (``killed``, ``gone``, its own zombie)."""
    with private_home.dispatch_homes():
        home = homes.create()
        cli = spawned([sys.executable, "-c", "import time; time.sleep(60)"])
        cli.kill()
        process_control.note_stop(cli, True)
    assert not Path(home).exists()


def test_a_child_attributed_to_one_slot_never_keeps_another_slots_home(homes):
    barrier = threading.Barrier(2, timeout=30)
    made, errors = {}, []

    def slot(name, survivor):
        try:
            with private_home.dispatch_homes():
                made[name] = homes.create()
                if survivor:
                    process_control.note_stop(object(), False)
                barrier.wait()  # both slots are inside their dispatch
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(repr(exc))

    threads = [threading.Thread(target=slot, args=("kept", True)),
               threading.Thread(target=slot, args=("released", False))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert Path(made["kept"]).is_dir() and not Path(made["released"]).exists()
    assert homes.isolated_home_dirs() == (made["kept"],)


_SLOW_CLI = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.mark.parametrize("cli_verdict, groups, job_failed, kept", [
    ("killed", {4242: "survived"}, False, True),
    ("killed", {4242: "unresolved"}, False, True),
    ("terminated", {4242: "gone", -4243: "skipped"}, False, True),
    ("survived", {}, False, True),
    ("terminated", {}, True, True),
    ("terminated", {4242: "gone", 4243: "killed"}, False, False),
    ("killed", {}, False, False),
])
def test_lane_a_keeps_the_home_of_a_cli_whose_stop_is_unconfirmed(
    homes, monkeypatch, cli_verdict, groups, job_failed, kept,
):
    """``llm_client._run_provider_cli``'s deadline stop, verdict by verdict
    (``process_control.unconfirmed_kill``'s rule for the groups)."""
    from utils import llm_client

    def sweep(proc, _snapshot, _grace_s, **_kwargs):
        proc.kill()
        proc.wait(timeout=30)
        return process_control.TreeSweep(
            cli=cli_verdict, groups=dict(groups),
            job=process_control.JOB_FAILED if job_failed else None,
        )

    monkeypatch.setattr(process_control, "terminate_timed_out", sweep)
    with private_home.dispatch_homes():
        home = homes.create()
        with pytest.raises(subprocess.TimeoutExpired):
            llm_client._run_provider_cli(_SLOW_CLI, capture_output=True, text=True,
                                         timeout=0.5, stdin=subprocess.DEVNULL)
    assert Path(home).is_dir() is kept
    assert homes.isolated_home_dirs() == ((home,) if kept else ())


def test_lane_a_releases_the_home_of_a_cli_that_exited(homes):
    from utils import llm_client

    with private_home.dispatch_homes():
        home = homes.create()
        done = llm_client._run_provider_cli([sys.executable, "-c", "pass"],
                                            capture_output=True, text=True, timeout=30,
                                            stdin=subprocess.DEVNULL)
    assert done.returncode == 0
    assert not Path(home).exists()


@pytest.mark.parametrize("verdict, own_zombie, kept", [
    ("survived", False, True),
    ("unresolved", False, True),
    ("skipped", False, True),
    (None, False, True),  # the kill raised
    ("survived", True, False),  # the child's own zombie: confirmed
    ("killed", False, False),
    ("gone", False, False),
])
def test_lane_b_keeps_the_home_of_a_runner_whose_kill_is_unconfirmed(
    homes, monkeypatch, capsys, verdict, own_zombie, kept,
):
    """``subagent_executor._kill_timed_out``, every timeout kill's one path."""
    import types

    from utils import subagent_executor as executor

    def kill_tree(_pid, **_kwargs):
        if verdict is None:
            raise OSError("the kill failed")
        return verdict

    monkeypatch.setattr(process_control, "kill_tree", kill_tree)
    monkeypatch.setattr(process_control, "signals_by_identity", lambda: False)
    monkeypatch.setattr(process_control, "own_child_stopped", lambda _p, _g: own_zombie)
    runner = types.SimpleNamespace(pid=4242, returncode=None, kill=lambda: None)
    with private_home.dispatch_homes():
        home = homes.create()
        executor._kill_timed_out(runner, None)
    capsys.readouterr()
    assert Path(home).is_dir() is kept
    assert homes.isolated_home_dirs() == ((home,) if kept else ())


# ---------------------------------------------------------------------------
# The sweep's owner identity: a PID namespace and a start time it can trust
# ---------------------------------------------------------------------------


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Roots made under *tmp_path*, and the first creation's sweep kept there."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def test_an_unreadable_pid_namespace_on_both_sides_never_sweeps(sweeper, in_tmp, monkeypatch):
    """Two sandboxes sharing a temp directory, neither able to read its PID
    namespace: the other's pid being absent here proves nothing."""
    monkeypatch.setattr(private_home, "_PID_NAMESPACES", True)
    monkeypatch.setattr(process_control, "pid_namespace", lambda: None)
    root = sweeper.create()
    sweeper.reset()  # another process's root now
    monkeypatch.setattr(process_control, "process_liveness", lambda _pid: process_control.DEAD)

    assert sweeper.sweep_orphans(str(in_tmp)) == []
    assert Path(root).is_dir()


def test_a_namespace_unreadable_here_never_sweeps_a_known_one(sweeper, in_tmp, monkeypatch):
    monkeypatch.setattr(private_home, "_PID_NAMESPACES", True)
    monkeypatch.setattr(process_control, "pid_namespace", lambda: "pid:[4026531836]")
    root = sweeper.create()
    sweeper.reset()
    monkeypatch.setattr(process_control, "pid_namespace", lambda: None)
    monkeypatch.setattr(process_control, "process_liveness", lambda _pid: process_control.DEAD)

    assert sweeper.sweep_orphans(str(in_tmp)) == []
    assert Path(root).is_dir()


def test_without_pid_namespaces_a_dead_owner_is_still_swept(sweeper, tmp_path, monkeypatch):
    """Off Linux there are no namespaces: every process shares one pid space."""
    monkeypatch.setattr(private_home, "_PID_NAMESPACES", False)
    monkeypatch.setattr(process_control, "pid_namespace", lambda: None)
    dead_pid, dead_start = _dead_owner()
    root = tmp_path / f"{sweeper.prefix}{dead_pid}-{dead_start}-na-abc1"
    root.mkdir()

    assert sweeper.sweep_orphans(str(tmp_path)) == [str(root)]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc start times")
def test_a_start_time_read_through_the_ps_fallback_never_sweeps(sweeper, in_tmp, monkeypatch):
    """Recorded from ``/proc`` ticks, read back as ``ps`` text: one live owner."""
    root = sweeper.create()
    sweeper.reset()
    monkeypatch.setattr(process_control, "_linux_start_time", lambda _pid: None)
    assert process_control.process_start_time(os.getpid()) is not None, "ps fallback"

    assert sweeper.sweep_orphans(str(in_tmp)) == []
    assert Path(root).is_dir()


def test_a_start_time_unreadable_now_never_sweeps(sweeper, in_tmp, monkeypatch):
    root = sweeper.create()
    sweeper.reset()
    monkeypatch.setattr(process_control, "sourced_start_time", lambda _pid: None)

    assert sweeper.sweep_orphans(str(in_tmp)) == []
    assert Path(root).is_dir()


def test_an_owner_whose_liveness_cannot_be_queried_is_never_swept(sweeper, in_tmp, monkeypatch):
    """A failed query (Windows' OpenProcess or GetExitCodeProcess refusing) is
    not a death certificate (Codex's re-check of U60b)."""
    root = sweeper.create()
    sweeper.reset()
    monkeypatch.setattr(process_control, "process_liveness", lambda _pid: process_control.UNKNOWN)

    assert sweeper.sweep_orphans(str(in_tmp)) == []
    assert Path(root).is_dir()


def test_a_ps_start_time_that_reads_differently_never_sweeps(sweeper, tmp_path, monkeypatch):
    """`ps` text follows the reader's time zone, so the same live owner can read
    differently from the dispatcher that named the root: inconclusive."""
    monkeypatch.setattr(private_home, "_PID_NAMESPACES", False)
    monkeypatch.setattr(process_control, "pid_namespace", lambda: None)
    recorded = private_home._start_token(("ps", "Sun Nov  8 23:00:00 2026"))
    root = tmp_path / f"{sweeper.prefix}{os.getpid()}-{recorded}-na-abc1"
    root.mkdir()
    monkeypatch.setattr(process_control, "sourced_start_time",
                        lambda _pid: ("ps", "Mon Nov  9 04:00:00 2026"))

    assert sweeper.sweep_orphans(str(tmp_path)) == []
    assert root.is_dir()


def test_a_proc_start_time_that_differs_still_marks_a_reused_pid(sweeper, tmp_path, monkeypatch):
    monkeypatch.setattr(private_home, "_PID_NAMESPACES", False)
    monkeypatch.setattr(process_control, "pid_namespace", lambda: None)
    recorded = private_home._start_token(("proc", "100"))
    root = tmp_path / f"{sweeper.prefix}{os.getpid()}-{recorded}-na-abc1"
    root.mkdir()
    monkeypatch.setattr(process_control, "sourced_start_time", lambda _pid: ("proc", "200"))

    assert sweeper.sweep_orphans(str(tmp_path)) == [str(root)]


@pytest.mark.skipif(os.name == "nt", reason="POSIX kill(0) probe")
def test_process_liveness_tells_gone_from_unanswerable(monkeypatch):
    dead_pid, _ = _dead_owner()
    assert process_control.process_liveness(int(dead_pid)) == process_control.DEAD
    assert process_control.process_liveness(os.getpid()) == process_control.ALIVE

    def refuse(_pid, _sig):
        raise OSError(22, "EINVAL")

    monkeypatch.setattr(os, "kill", refuse)
    assert process_control.process_liveness(12345) == process_control.UNKNOWN


#: Makes a root under argv[2] with prefix argv[3], prints it, and exits on a
#: line from stdin without its exit pass, as a SIGKILL would leave it.
_OWNER = (
    "import os, sys, tempfile\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "tempfile.tempdir = sys.argv[2]\n"
    "from utils.providers.private_home import PrivateHomes\n"
    "print(PrivateHomes(sys.argv[3]).create(), flush=True)\n"
    "sys.stdin.readline()\n"
    "os._exit(0)\n"
)


def test_a_root_is_swept_once_its_owner_is_confirmed_dead_and_not_before(sweeper, tmp_path):
    owner = subprocess.Popen(
        [sys.executable, "-c", _OWNER, str(SKILL_ROOT), str(tmp_path), sweeper.prefix],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        root = owner.stdout.readline().strip()
        assert Path(root).is_dir()
        assert sweeper.sweep_orphans(str(tmp_path)) == [], "a live owner's root was swept"
        owner.communicate("\n", timeout=30)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
    assert sweeper.sweep_orphans(str(tmp_path)) == [root]
    assert not Path(root).exists()
