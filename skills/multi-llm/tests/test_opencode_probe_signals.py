"""A SIGTERM to an opencode probe ends what the probe started.

``release_gate.py step`` bounds ``probe.py row7`` (and the registry-install
probe runs the same way) by signalling the probe's process group at its
budget. Every child the probe starts through ``probe.run_bounded`` leads a
session of its own, so that signal never reaches it: ``opencode run`` and the
``uv`` and Python under it are the probe's to sweep. With Python's default
SIGTERM the interpreter ended on the spot, no ``finally`` ran, and they
survived. Both probes' ``main`` now install ``probe.install_exit_handlers``:
SIGTERM and SIGINT raise ``SystemExit``, every sweep runs, and the moments a
child exists but could not yet be swept (its spawn, the sweep itself) hold the
signal until they end.

The stand-in for ``opencode run`` is what the probe meets: a session leader
(``run_bounded`` makes it one) that starts a child in a process group of its
own, the way the plugin starts ``uv``. Neither ignores any signal. Linux only:
the sweep reads a session's members from ``/proc``, and the gate runs on Linux.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO / "packaging" / "opencode"

linux_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason="the session sweep reads /proc, and the release gate runs on Linux")

STAND_IN = textwrap.dedent("""\
    import os, subprocess, sys, time
    # `opencode run` as the probe meets it: it leads the session run_bounded
    # gave it and starts a child in a process group of its own (the plugin's
    # uv). Nothing here ignores a signal.
    grand = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                             process_group=0)
    with open(sys.argv[1] + ".tmp", "w") as handle:
        handle.write(f"{os.getpid()} {grand.pid}")
    os.replace(sys.argv[1] + ".tmp", sys.argv[1])
    time.sleep(600)
    """)

#: Runs one probe's own ``main`` with its work replaced by one bounded child,
#: the stand-in; ``control`` runs that child without ``main`` (no handlers).
DRIVER = textwrap.dedent("""\
    import importlib.util, os, sys
    from pathlib import Path
    which, package_dir, stand_in, pids, work = sys.argv[1:6]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def bounded(*_args, **_kwargs):
        sys.modules["opencode_probe"].run_bounded(
            [sys.executable, stand_in, pids], cwd=Path(work), env=dict(os.environ),
            budget_s=600, stdout=Path(work) / "child.out", stderr=Path(work) / "child.err")
        return 0

    if which == "probe":
        probe = load("opencode_probe", f"{package_dir}/probe.py")
        probe.row7 = bounded
        sys.exit(probe.main(["row7", "--tarball", "t.tgz", "--opencode", "oc",
                             "--registry", "r", "--out", "o.json"]))
    elif which == "registry_check":
        rc = load("opencode_registry_check", f"{package_dir}/registry_check.py")
        rc._probe()
        rc.check = bounded
        sys.exit(rc.main(["--spec", "a@1.2.3", "--registry", "r", "--tag", "t",
                          "--environment", "vm", "--out", "o.json"]))
    else:
        load("opencode_probe", f"{package_dir}/probe.py")
        sys.exit(bounded())
    """)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _alive(pid: int) -> bool:
    """Running, as opposed to gone or a zombie waiting to be reaped."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return stat[stat.rfind(")") + 2] != "Z"


def _survivors(pids, timeout: float = 15.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_alive(pid) for pid in pids):
            return []
        time.sleep(0.1)
    return [pid for pid in pids if _alive(pid)]


def _wait_for(path: Path, proc: "subprocess.Popen", timeout: float = 30.0) -> list:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"the driver exited {proc.returncode} before its child started: {err}")
        if time.monotonic() > deadline:
            pytest.fail("the stand-in never started")
        time.sleep(0.05)
    return [int(pid) for pid in path.read_text().split()]


def _kill(pids) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _drive(tmp_path: Path, which: str):
    stand_in = tmp_path / "stand_in.py"
    stand_in.write_text(STAND_IN, encoding="utf-8")
    pids = tmp_path / "pids"
    # A session of its own, so cleanup can reach everything the driver left;
    # the signal under test goes to the probe's pid alone.
    driver = subprocess.Popen(
        [sys.executable, "-c", DRIVER, which, str(PACKAGE_DIR), str(stand_in), str(pids),
         str(tmp_path)], cwd=tmp_path, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return driver, pids


@linux_only
@pytest.mark.parametrize("which", ["probe", "registry_check"])
def test_sigterm_to_the_probe_leaves_no_child_of_its_sessions(tmp_path, which):
    driver, pids_file = _drive(tmp_path, which)
    children = []
    try:
        children = _wait_for(pids_file, driver)
        assert all(_alive(pid) for pid in children)
        os.kill(driver.pid, signal.SIGTERM)
        _out, err = driver.communicate(timeout=60)
        assert driver.returncode == 128 + signal.SIGTERM, err
        assert _survivors(children) == [], "the probe's child session outlived its SIGTERM"
    finally:
        try:
            os.killpg(driver.pid, signal.SIGKILL)
        except OSError:
            pass
        _kill(children)


@linux_only
def test_the_control_without_main_leaves_them_running(tmp_path):
    """What the handlers fix, observed: the same child under the library alone
    (Python's default SIGTERM) survives the probe. So the row above proves
    the handlers, not a child that would have died anyway."""
    driver, pids_file = _drive(tmp_path, "control")
    children = []
    try:
        children = _wait_for(pids_file, driver)
        os.kill(driver.pid, signal.SIGTERM)
        driver.communicate(timeout=60)
        assert driver.returncode == -signal.SIGTERM
        time.sleep(0.5)
        assert all(_alive(pid) for pid in children), "the control's children died on their own"
    finally:
        try:
            os.killpg(driver.pid, signal.SIGKILL)
        except OSError:
            pass
        _kill(children)


HOLD = textwrap.dedent("""\
    import importlib.util, os, signal, subprocess, sys, time
    spec = importlib.util.spec_from_file_location("opencode_probe", sys.argv[1])
    probe = importlib.util.module_from_spec(spec)
    sys.modules["opencode_probe"] = probe
    spec.loader.exec_module(probe)
    probe.install_exit_handlers()
    if sys.argv[2] == "hold":
        with probe.held_exit_signals():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.2)
            print("inside the hold, still running", flush=True)
        print("after the hold: never", flush=True)
    else:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        finally:
            os.kill(os.getpid(), signal.SIGTERM)   # a second one, mid-sweep
            time.sleep(0.2)
            inherited = subprocess.run(
                [sys.executable, "-c",
                 "import signal; print(signal.getsignal(signal.SIGTERM) == signal.SIG_IGN)"],
                capture_output=True, text=True).stdout.strip()
            print(f"the sweep finished; a child inherits SIG_IGN: {inherited}", flush=True)
    """)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals to self")
def test_a_held_signal_waits_for_the_block_then_exits(tmp_path):
    done = subprocess.run([sys.executable, "-c", HOLD, str(PACKAGE_DIR / "probe.py"), "hold"],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 128 + signal.SIGTERM, done.stderr
    assert "inside the hold, still running" in done.stdout
    assert "never" not in done.stdout


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals to self")
def test_a_second_signal_cannot_cut_the_sweep_short(tmp_path):
    done = subprocess.run([sys.executable, "-c", HOLD, str(PACKAGE_DIR / "probe.py"), "sweep"],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 128 + signal.SIGTERM, done.stderr
    assert "the sweep finished; a child inherits SIG_IGN: False" in done.stdout


# ---------------------------------------------------------------------------
# The long children of the registry-install probe go through run_bounded
# ---------------------------------------------------------------------------

@pytest.fixture()
def rc():
    return _load("opencode_registry_check_signals", PACKAGE_DIR / "registry_check.py")


@linux_only
def test_a_build_past_its_budget_takes_its_grandchildren_with_it(tmp_path, rc):
    """``_plain_run`` (the candidate's build, the temporary worktree) used
    ``subprocess.run(timeout=)``, which kills the direct child only: ``uv
    run``'s Python would have gone on building under the next step."""
    stand_in = tmp_path / "stand_in.py"
    stand_in.write_text(STAND_IN, encoding="utf-8")
    pids = tmp_path / "pids"
    log = tmp_path / "build.log"
    log.write_text("earlier output\n", encoding="utf-8")
    children = []
    try:
        result = rc._plain_run([sys.executable, str(stand_in), str(pids)], tmp_path,
                               dict(os.environ), 2, log)
        children = [int(pid) for pid in pids.read_text().split()]
        assert result["exit"] is None, "the budget did not end it"
        assert _survivors(children) == []
        text = log.read_text(encoding="utf-8")
        assert text.startswith("earlier output\n$ "), "the log is appended to, as before"
    finally:
        _kill(children)


def test_plain_run_reports_an_exit_and_a_missing_program(tmp_path, rc):
    log = tmp_path / "log"
    ok = rc._plain_run([sys.executable, "-c", "print('built')"], tmp_path, dict(os.environ), 60, log)
    assert ok["exit"] == 0 and "built" in log.read_text()
    missing = rc._plain_run([str(tmp_path / "no-such-program")], tmp_path, dict(os.environ), 60, log)
    assert missing["exit"] == 127 and "cannot run" in log.read_text()


def test_the_host_install_is_bounded_by_the_session_runner(tmp_path, rc, monkeypatch):
    seen = {}
    probe = rc._probe()

    def run_bounded(argv, *, cwd, env, budget_s, stdout, stderr, **_kwargs):
        seen.update(argv=argv, budget_s=budget_s, home=env["HOME"])
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("npm noise\n/x/node_modules/.bin/opencode\n")
        stderr.write_text("")
        return probe.Bounded(argv=tuple(argv), returncode=0, killed=False, reason=None,
                             elapsed_s=0.0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(probe, "run_bounded", run_bounded)
    assert rc._install_host_binary("https://registry.npmjs.org", tmp_path) == \
        "/x/node_modules/.bin/opencode"
    assert seen["budget_s"] == rc.HOST_INSTALL_BUDGET_S
    assert seen["argv"][1].endswith("bump.py") and seen["argv"][2] == "install"
    assert seen["home"] == str(tmp_path / "opencode-host-home")
