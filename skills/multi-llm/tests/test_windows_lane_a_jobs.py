"""Lane A on Windows: a stopped provider CLI takes its whole tree with it (2026-10-25).

``llm_client._run_provider_cli`` is the review fan-out's spawn of a provider CLI.
On POSIX a stop (the timeout, an interrupt, a stop from outside, the registry's
sweep) snapshots the CLI's descendants and sweeps every process group they lead.
Windows has no groups, so ``descendants`` is ``[]`` there and the stop used to be
the CLI's ``TerminateProcess`` alone: what the CLI had started kept running past
the timeout, holding the inherited pipes until the drain's 5 s cap. A detached
worker's own KILL_ON_JOB_CLOSE job hid that; a foreground run had nothing.

Now the CLI starts in a Job Object of its own (``process_control.popen_tree``),
the job travels with it into ``live_cli_registry``, and every stop ends the job
right after the CLI (``process_control.terminate_timed_out``, ``LiveCliRegistry
.sweep``). A CLI that exits on its own keeps its job untouched, and the job is
closed, never killed, once the call is done with it.

Three kinds of row:

* **stand-ins** for the Win32 calls (a recording job API, a recording CLI), with
  ``IS_WINDOWS`` pinned, for the order of every step and every degradation;
* **POSIX rows** where a process group plays the job (a SIGSTOP for
  ``CREATE_SUSPENDED``, a ``killpg`` for ``TerminateJobObject``), with real
  processes, so "is the grandchild gone, and did the call return in time?" is
  asked of real processes;
* **Windows-only rows** with the real calls, skipped here, for the Windows leg.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.harness import cli_tree  # noqa: E402
from utils import llm_client, process_control  # noqa: E402

WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="the real Win32 calls")
POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="a POSIX process group plays the job")

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_SUSPENDED = 0x00000004
STATUS_CONTROL_C_EXIT = 0xC000013A


# ===========================================================================
# Stand-ins: one shared event log, so the order of every step can be asserted
# ===========================================================================

class Events(list):
    """``(name, monotonic time)`` for every step a row takes, in order."""

    def add(self, name):
        self.append((name, time.monotonic()))

    def names(self):
        return [name for name, _ in self]

    def at(self, name):
        return next(when for what, when in self if what == name)


class RecordingJobApi:
    """``process_control._JobApi``'s calls, recorded; each answer configurable."""

    def __init__(self, events, *, create=101, terminate=True):
        self.events = events
        self._create, self._terminate = create, terminate
        self.terminated = []
        self.closed = []

    def create(self):
        self.events.add("job.create")
        return self._create

    def open_process(self, pid):
        return 202

    def assign(self, job, process):
        self.events.add("job.assign")
        return True

    def resume(self, process):
        self.events.add("cli.resume")
        return True

    def terminate(self, job, code):
        self.events.add("job.terminate")
        self.terminated.append(job)
        if isinstance(self._terminate, BaseException):
            raise self._terminate
        return self._terminate

    def close(self, handle):
        if handle == self._create:
            self.events.add("job.close")
            self.closed.append(handle)


class FakeCli:
    """A provider CLI that never runs: ``Popen``'s surface, every call recorded.

    *wait_raises* is what the first ``communicate`` raises (``None``: it returns
    and the CLI exits with *exit_code*). *obeys_terminate* says whether
    ``terminate`` ends it; ``kill`` always does. *during_wait* runs inside the
    first ``communicate``, while the call is waiting on the CLI.
    """

    spawned = []

    def __init__(self, args, *, events, exit_code=0, wait_raises=None,
                 obeys_terminate=True, during_wait=None, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.events = events
        self.pid = 4242
        self.returncode = None
        self.stdin = self.stdout = self.stderr = None
        self._exit_code = exit_code
        self._wait_raises = wait_raises
        self._obeys_terminate = obeys_terminate
        self._during_wait = during_wait
        self._waits = 0
        FakeCli.spawned.append(self)

    def communicate(self, input=None, timeout=None):
        self._waits += 1
        if self._waits == 1:
            self.events.add("cli.wait")
            if self._during_wait is not None:
                self._during_wait(self)
            if self._wait_raises is not None:
                raise self._wait_raises
            if self.returncode is None:
                self.returncode = self._exit_code
            return "out", "err"
        self.events.add("cli.drain")
        return "partial", ""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.events.add("cli.terminate")
        if self._obeys_terminate and self.returncode is None:
            self.returncode = 1  # TerminateProcess(handle, 1)

    def kill(self):
        self.events.add("cli.kill")
        if self.returncode is None:
            self.returncode = 1


@pytest.fixture
def lane_a(monkeypatch):
    """``_run_provider_cli`` as it runs on Windows, over the stand-ins.

    ``spawn(**behaviour)`` sets what the next CLI does. ``IS_WINDOWS`` is
    pinned: off it, ``descendants(4242)`` would walk whatever process really has
    that pid here. A fresh registry, so a row sees only its own CLI.
    """
    events = Events()
    api = RecordingJobApi(events)
    behaviour = {}
    FakeCli.spawned = []

    def popen(args, **kwargs):
        return FakeCli(args, events=events, **behaviour, **kwargs)

    registry = process_control.LiveCliRegistry(interval_s=3600)
    monkeypatch.setattr(process_control, "IS_WINDOWS", True)
    monkeypatch.setattr(process_control.subprocess, "Popen", popen)
    monkeypatch.setattr(process_control, "_REAL_POPEN", popen)
    monkeypatch.setattr(process_control, "_job_api", lambda: api)
    monkeypatch.setattr(process_control, "live_cli_registry", registry)
    monkeypatch.setattr(process_control, "taskkill_tree",
                        lambda *a, **k: pytest.fail("Lane A never runs taskkill"))
    # Short, so a CLI that ignores its stop costs a row a fraction of a second.
    monkeypatch.setattr(llm_client, "LANE_A_TERMINATE_GRACE_S", 0.3)

    class LaneA:
        def __init__(self):
            self.events, self.api, self.registry = events, api, registry

        def spawn(self, **kwargs):
            behaviour.clear()
            behaviour.update(kwargs)

        def run(self, **kwargs):
            kwargs.setdefault("capture_output", True)
            kwargs.setdefault("text", True)
            kwargs.setdefault("timeout", 30)
            kwargs.setdefault("stdin", subprocess.DEVNULL)
            return llm_client._run_provider_cli(["pi", "--mode", "json"], **kwargs)

        @property
        def cli(self):
            [cli] = FakeCli.spawned
            return cli

    return LaneA()


def _timeout():
    return subprocess.TimeoutExpired(["pi"], 30, output="so far", stderr="")


# ===========================================================================
# 1. The spawn: a Job Object, and no new process group
# ===========================================================================

class TestTheSpawn:
    def test_the_cli_starts_in_a_job_and_in_no_group_of_its_own(self, lane_a):
        """A terminal Ctrl-C must still reach the CLI: CREATE_NEW_PROCESS_GROUP
        would disable it for the CLI's whole group, and a job changes nothing
        about who the console delivers it to."""
        lane_a.spawn()
        completed = lane_a.run()

        assert completed.returncode == 0 and completed.stdout == "out"
        kwargs = lane_a.cli.kwargs
        assert kwargs["creationflags"] == CREATE_SUSPENDED
        assert not kwargs["creationflags"] & CREATE_NEW_PROCESS_GROUP
        for key in ("start_new_session", "process_group", "preexec_fn"):
            assert key not in kwargs, f"the CLI was given {key}"
        # Created suspended, put in its job, and only then resumed and waited on.
        assert lane_a.events.names()[:4] == ["job.create", "job.assign", "cli.resume",
                                              "cli.wait"]

    def test_the_job_travels_with_the_cli_into_the_registry(self, lane_a):
        seen = []
        lane_a.spawn(during_wait=lambda cli: seen.extend(lane_a.registry.entries()))
        lane_a.run()

        [entry] = seen
        assert entry.proc is lane_a.cli
        assert isinstance(entry.job, process_control.TreeJob)
        assert len(lane_a.registry) == 0, "unregistered once the call returned"

    @POSIX_ONLY
    def test_off_windows_it_is_the_plain_spawn_with_no_job(self, monkeypatch):
        monkeypatch.setattr(process_control, "IS_WINDOWS", False)
        monkeypatch.setattr(process_control, "_job_api",
                            lambda: pytest.fail("no job API off Windows"))
        seen = []
        real_register = process_control.live_cli_registry.register

        def register(proc, job=None):
            seen.append(job)
            return real_register(proc, job)

        monkeypatch.setattr(process_control.live_cli_registry, "register", register)
        completed = llm_client._run_provider_cli(
            [sys.executable, "-c", "print('ok')"], capture_output=True, text=True,
            timeout=30)
        assert completed.returncode == 0 and completed.stdout == "ok\n"
        assert seen == [None]


# ===========================================================================
# 2. The stop paths end the job right after the CLI; a normal exit does not
# ===========================================================================

class TestTheStopPaths:
    def test_a_timeout_ends_the_job_right_after_the_cli_and_closes_it(self, lane_a):
        """Not after the grace. On Windows ``terminate`` is ``TerminateProcess``,
        so the CLI runs no cleanup and the grace would only keep its children
        running (and holding the pipes) for longer."""
        lane_a.spawn(wait_raises=_timeout(), obeys_terminate=False)
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            lane_a.run()

        events = lane_a.events
        assert events.names()[3:] == ["cli.wait", "cli.terminate", "job.terminate",
                                      "cli.kill", "cli.drain", "job.close"]
        assert events.at("job.terminate") - events.at("cli.terminate") < 0.2
        # The CLI shrugged off its stop, so the grace ran in full before its
        # kill; the job had been ended before any of it.
        assert events.at("cli.kill") - events.at("job.terminate") >= 0.25
        sweep = caught.value.tree_sweep
        assert (sweep.cli, sweep.groups, sweep.job) == ("killed", {}, "terminated")
        assert caught.value.stdout == "partial", "the drain still collects the output"
        assert len(lane_a.registry) == 0

    def test_the_timeouts_log_line_names_the_job(self, lane_a):
        lane_a.spawn(wait_raises=_timeout())
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            lane_a.run()
        assert caught.value.tree_sweep.describe() == "cli terminated; job terminated"
        assert llm_client._describe_tree_sweep(caught.value) == (
            "; cli terminated; job terminated")

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_an_interrupt_ends_the_job_and_still_propagates(self, lane_a, interrupt):
        lane_a.spawn(wait_raises=interrupt())
        with pytest.raises(interrupt):
            lane_a.run()
        assert lane_a.events.names()[3:] == ["cli.wait", "cli.terminate",
                                             "job.terminate", "job.close"]
        assert len(lane_a.registry) == 0

    @pytest.mark.parametrize("exit_code", [0, 1, 2, 3, 0xC0000005])
    def test_a_normal_exit_leaves_the_job_alone_and_closes_it(self, lane_a, exit_code):
        """Parity with POSIX, whose groups a normal exit leaves alone: what a
        CLI leaves running after it exits on its own keeps running. A crash
        (0xC0000005) is not a stop from outside either."""
        lane_a.spawn(exit_code=exit_code)
        completed = lane_a.run()
        assert completed.returncode == exit_code
        assert "job.terminate" not in lane_a.events.names()
        assert "cli.terminate" not in lane_a.events.names()
        assert lane_a.events.names()[-1] == "job.close"
        assert lane_a.api.closed == [101]

    @pytest.mark.parametrize("exit_code", [
        STATUS_CONTROL_C_EXIT,  # the console's Ctrl-C reached the default handler
        130,                    # a CLI that handled the Ctrl-C and exited 128+2
        143,                    # ... or a SIGTERM, 128+15
    ], ids=["status-control-c-exit", "130", "143"])
    def test_a_cli_stopped_from_outside_on_windows_has_its_job_ended(
        self, lane_a, exit_code
    ):
        """Windows exit codes are never negative, so the ``-N`` half of
        ``_stopped_from_outside`` cannot fire there; Ctrl-C's own status and a
        handler's ``128 + N`` can. The CLI is dead already, so nothing is sent
        to it; its job, which still holds what ignored the Ctrl-C, is ended."""
        lane_a.spawn(exit_code=exit_code)
        completed = lane_a.run()
        assert completed.returncode == exit_code
        assert lane_a.events.names()[3:] == ["cli.wait", "job.terminate", "job.close"]

    @pytest.mark.parametrize("returncode, stopped", [
        (None, False), (0, False), (1, False), (2, False), (-9, False),
        (-2, True), (-15, True), (130, True), (143, True),
        (STATUS_CONTROL_C_EXIT, True), (0xC0000005, False),
    ])
    def test_which_exits_are_a_stop_from_outside(self, returncode, stopped):
        assert llm_client._stopped_from_outside(returncode) is stopped


# ===========================================================================
# 3. The registry: every live CLI's job, inside the sweep's one clock
# ===========================================================================

class _LiveCli(FakeCli):
    def __init__(self, events, pid, **kwargs):
        super().__init__(["pi"], events=events, **kwargs)
        self.pid = pid


class TestTheRegistrySweep:
    @pytest.fixture
    def registry(self, monkeypatch):
        monkeypatch.setattr(process_control, "IS_WINDOWS", True)
        return process_control.LiveCliRegistry(interval_s=3600)

    def _register(self, registry, events, pid, **behaviour):
        cli = _LiveCli(events, pid, **behaviour)
        api = RecordingJobApi(Events(), create=pid * 10)
        job = process_control.TreeJob(api, pid * 10)
        registry.register(cli, job)
        return cli, api

    def test_the_sweep_ends_every_live_clis_job(self, registry):
        events = Events()
        running, running_api = self._register(registry, events, 7001)
        stubborn, stubborn_api = self._register(registry, events, 7002,
                                                obeys_terminate=False)
        gone, gone_api = self._register(registry, events, 7003)
        gone.returncode = 0  # exited on its own, still registered

        started = time.monotonic()
        swept = registry.sweep(grace_s=0.3, poll_interval_s=0.02, budget_s=2.0)
        elapsed = time.monotonic() - started

        results = {entry.pid: result for entry, result in swept}
        assert {pid: (r.cli, r.job) for pid, r in results.items()} == {
            7001: ("terminated", "terminated"),
            7002: ("killed", "terminated"),
            7003: ("exited", "terminated"),
        }
        for api in (running_api, stubborn_api, gone_api):
            assert api.terminated == [api._create]
            assert api.closed == [], "the spawning call closes its job, not the sweep"
        # The jobs went out with the CLIs' stops, before the grace was waited.
        assert stubborn_api.events.at("job.terminate") < events.at("cli.kill")
        assert elapsed < 2.0 + 0.5, f"{elapsed:.2f}s: past the sweep's one budget"
        assert results[7002].describe() == "cli killed; job terminated"

    def test_a_job_is_ended_under_the_lock_so_no_unregister_can_close_it_meanwhile(
        self, registry
    ):
        events = Events()
        cli, api = self._register(registry, events, 7001)
        free = []

        def terminate(job, code):
            # Another thread (the slot, about to unregister) cannot get in.
            probe = threading.Thread(
                target=lambda: free.append(registry._lock.acquire(blocking=False)))
            probe.start()
            probe.join(10)
            if free and free[0]:
                registry._lock.release()
            return True

        api.terminate = terminate
        [(_, result)] = registry.sweep(grace_s=0.1, budget_s=1.0)
        assert free == [False]
        assert result.job == "terminated"

    def test_a_cli_whose_call_returned_mid_sweep_keeps_its_job_untouched(
        self, registry, monkeypatch
    ):
        """Its call unregistered it and closed the handle; Windows may already
        have handed that handle value to another slot's job."""
        events = Events()
        leaving, leaving_api = self._register(registry, events, 7001)
        staying, staying_api = self._register(registry, events, 7002)
        real_signal = process_control._signal_cli

        def signal_cli(proc):
            if proc is leaving:
                [entry] = [e for e in registry.entries() if e.proc is leaving]
                registry.unregister(leaving)
                entry.job.close()
            return real_signal(proc)

        monkeypatch.setattr(process_control, "_signal_cli", signal_cli)
        results = {entry.pid: result
                   for entry, result in registry.sweep(grace_s=0.1, budget_s=1.0)}

        assert leaving_api.terminated == []
        assert results[7001].job is None
        assert staying_api.terminated == [staying_api._create]
        assert results[7002].job == "terminated"

    def test_a_sweep_from_inside_a_held_lock_does_not_deadlock(self, registry):
        """A signal handler runs on the main thread, which may hold the lock."""
        events = Events()
        self._register(registry, events, 7001)
        done = []

        def sweep_while_holding():
            with registry._lock:
                done.append(registry.sweep(grace_s=0.1, budget_s=1.0))

        worker = threading.Thread(target=sweep_while_holding, daemon=True)
        worker.start()
        worker.join(10)
        assert done and done[0][0][1].job == "terminated"

    def test_the_interrupt_sweep_names_the_job_and_the_call_then_returns(
        self, lane_a, capsys
    ):
        """``sweep_live_clis`` while the slot waits on its CLI: the Ctrl-C's
        sweep ends the job, the CLI's wait returns, and the call closes it."""
        lane_a.spawn(during_wait=lambda cli: process_control.sweep_live_clis(
            "INTERRUPT_SWEEP"))
        completed = lane_a.run()

        err = capsys.readouterr().err
        assert "[INTERRUPT_SWEEP] cli pid 4242: cli terminated; job terminated" in err
        # TerminateProcess's exit code is an ordinary 1: not a second stop.
        assert completed.returncode == 1
        assert lane_a.events.names().count("job.terminate") == 1
        assert lane_a.events.names()[-1] == "job.close"

    def test_entries_without_a_job_sweep_as_before(self, registry):
        cli = _LiveCli(Events(), 7001)
        registry.register(cli)
        [(_, result)] = registry.sweep(grace_s=0.1, budget_s=1.0)
        assert result.job is None
        assert result.describe() == "cli terminated; no descendant process groups"


# ===========================================================================
# 4. Degradation: no job, or a job that will not end, is today's stop
# ===========================================================================

class TestDegradation:
    def test_a_job_that_fails_to_terminate_is_named_and_the_cli_still_stopped(
        self, lane_a
    ):
        lane_a.api._terminate = False
        lane_a.spawn(wait_raises=_timeout())
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            lane_a.run()
        sweep = caught.value.tree_sweep
        assert (sweep.cli, sweep.job) == ("terminated", "failed")
        assert sweep.describe() == (
            "cli terminated; job failed — JOB NOT ENDED: what the cli started may "
            "still be running")
        assert lane_a.api.closed == [101]

    def test_a_job_whose_terminate_raises_never_breaks_the_teardown(self, lane_a):
        lane_a.api._terminate = RuntimeError("handle gone")
        lane_a.spawn(wait_raises=_timeout())
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            lane_a.run()
        assert caught.value.tree_sweep.job == "failed"
        assert lane_a.api.closed == [101]

    def test_a_job_object_that_raises_is_failed_not_an_exception(self):
        class Raising:
            def terminate(self):
                raise RuntimeError("not a TreeJob's manners, but possible")

        assert process_control._end_job(Raising()) == "failed"
        assert process_control._end_job(None) is None

    def test_a_closed_job_reads_failed(self):
        api = RecordingJobApi(Events())
        job = process_control.TreeJob(api, 101)
        job.close()
        assert process_control._end_job(job) == "failed"
        assert api.terminated == [], "a closed handle is never used"

    def test_a_sweep_that_raises_falls_back_to_the_cli_then_its_job(
        self, lane_a, monkeypatch
    ):
        def broken(*_a, **_k):
            raise RuntimeError("process table unreadable")

        monkeypatch.setattr(process_control, "terminate_timed_out", broken)
        lane_a.spawn(wait_raises=_timeout())
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            lane_a.run()
        assert caught.value.tree_sweep is None
        names = lane_a.events.names()
        assert names.index("cli.kill") < names.index("job.terminate") < names.index(
            "job.close")

    def test_a_windows_cli_that_got_no_job_is_stopped_as_before(self, lane_a):
        lane_a.api._create = None
        lane_a.spawn(wait_raises=_timeout())
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            lane_a.run()
        assert "creationflags" not in lane_a.cli.kwargs, "no suspended spawn without a job"
        sweep = caught.value.tree_sweep
        assert sweep.job is None
        assert sweep.describe() == "cli terminated; no descendant process groups"
        assert "job.terminate" not in lane_a.events.names()

    def test_a_replaced_popen_is_called_as_it_is_and_gets_no_job(self, lane_a, monkeypatch):
        """The suite's own fakes of ``subprocess.Popen`` reach the Lane A spawn
        unchanged: there is no process to put in a job."""
        events = Events()

        def double(args, **kwargs):
            return FakeCli(args, events=events, **kwargs)

        monkeypatch.setattr(process_control.subprocess, "Popen", double)
        FakeCli.spawned = []
        completed = lane_a.run()
        assert completed.returncode == 0
        assert "creationflags" not in FakeCli.spawned[0].kwargs
        assert lane_a.events.names() == []


# ===========================================================================
# 5. What the log line says, and that a POSIX line has not moved
# ===========================================================================

def _describe_before_2026_10_25(cli, groups):
    """``TreeSweep.describe`` as it was, the reference a POSIX line must match."""
    survived = [pgid for pgid, verdict in groups.items() if verdict == "survived"]
    if not groups:
        return f"cli {cli}; no descendant process groups"
    verdicts = ", ".join(f"{pgid} {verdict}" for pgid, verdict in groups.items())
    line = f"cli {cli}; process groups: {verdicts}"
    if survived:
        line += " — SURVIVED: " + ", ".join(str(p) for p in survived) + " still running"
    return line


class TestDescribe:
    @pytest.mark.parametrize("cli, groups", [
        ("exited", {}),
        ("terminated", {}),
        ("killed", {77: "killed"}),
        ("terminated", {77: "killed", 78: "gone", 79: "skipped"}),
        ("survived", {77: "survived", 78: "killed", 79: "survived"}),
    ])
    def test_a_posix_line_is_word_for_word_what_it_was(self, cli, groups):
        assert process_control.TreeSweep(cli=cli, groups=dict(groups)).describe() == \
            _describe_before_2026_10_25(cli, groups)

    @pytest.mark.parametrize("cli, groups, job, line", [
        ("terminated", {}, "terminated", "cli terminated; job terminated"),
        ("exited", {}, "terminated", "cli exited; job terminated"),
        ("killed", {}, "failed",
         "cli killed; job failed — JOB NOT ENDED: what the cli started may still be "
         "running"),
        ("killed", {77: "survived"}, "terminated",
         "cli killed; job terminated; process groups: 77 survived — SURVIVED: 77 "
         "still running"),
    ])
    def test_a_job_verdict_is_named(self, cli, groups, job, line):
        assert process_control.TreeSweep(cli=cli, groups=groups, job=job).describe() == line

    @POSIX_ONLY
    def test_a_real_posix_timeout_says_what_it_always_said(self, monkeypatch):
        monkeypatch.setattr(process_control, "IS_WINDOWS", False)
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            llm_client._run_provider_cli(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                capture_output=True, text=True, timeout=0.5)
        sweep = caught.value.tree_sweep
        assert sweep.job is None
        assert sweep.describe() == "cli terminated; no descendant process groups"


# ===========================================================================
# 6. Real processes, with a POSIX process group playing the job
# ===========================================================================

#: The stand-in CLI: it starts a grandchild python that writes its pid to
#: argv[1] and sleeps, then sleeps itself — or, with argv[2] set, waits for the
#: grandchild's pid file and exits with that code (a CLI already stopped from
#: outside). argv[3] == "inherit" hands the grandchild the CLI's own stdout,
#: the drain's worst case. Plain Python on both platforms: no console, no
#: signal handler, no shell. The grandchild writes its own start time after
#: its pid, as ``process_control`` reads it, so the Windows rows' cleanup and
#: "gone" checks act on that very process and never on a recycled pid
#: (Fable review, 2026-10-25).
_GRANDCHILD = (
    "import os, sys, time\n"
    "path = sys.argv[1]\n"
    "sys.path.insert(0, " + repr(str(Path(__file__).parent.parent)) + ")\n"
    "try:\n"
    "    from utils import process_control\n"
    "    stamp = process_control.process_start_time(os.getpid()) or ''\n"
    "except Exception:\n"
    "    stamp = ''\n"
    "with open(path + '.tmp', 'w') as handle:\n"
    "    handle.write(str(os.getpid()) + ' ' + stamp)\n"
    "os.replace(path + '.tmp', path)\n"
    "time.sleep(60)\n"
)
_CLI = (
    "import os, subprocess, sys, time\n"
    "path, exit_code, stdout = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "subprocess.Popen(\n"
    "    [sys.executable, '-c', " + repr(_GRANDCHILD) + ", path],\n"
    "    stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
    "    stdout=sys.stdout if stdout == 'inherit' else subprocess.DEVNULL)\n"
    "if exit_code:\n"
    "    deadline = time.monotonic() + 30\n"
    "    while not os.path.exists(path) and time.monotonic() < deadline:\n"
    "        time.sleep(0.05)\n"
    "    os._exit(int(exit_code))\n"
    "time.sleep(60)\n"
)


def _cli_argv(marker, *, exit_code="", stdout="inherit"):
    return [sys.executable, "-c", _CLI, str(marker), str(exit_code), stdout]


def _grandchild(marker, timeout=30.0):
    """``(pid, start time)`` as the grandchild wrote them, read at least once;
    ``None`` if it never did."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            pid, _, stamp = Path(marker).read_text().partition(" ")
            return int(pid), stamp or None
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)


def _grandchild_pid(marker, timeout=30.0):
    """The pid the grandchild wrote, read at least once; ``None`` if it never did."""
    found = _grandchild(marker, timeout)
    return found[0] if found else None


class _GroupJobApi(RecordingJobApi):
    """A job off Windows: the CLI's own process group stands in for it, a
    SIGSTOP right after the spawn for ``CREATE_SUSPENDED``."""

    def __init__(self):
        super().__init__(Events())
        self.pid = None

    def open_process(self, pid):
        self.pid = pid
        return super().open_process(pid)

    def resume(self, process):
        os.killpg(self.pid, signal.SIGCONT)
        return super().resume(process)

    def terminate(self, job, code):
        super().terminate(job, code)
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True


@pytest.fixture
def group_job(monkeypatch):
    """``popen_tree`` with a process group as the job, on real processes.

    Off Windows nothing but the job could reach the grandchild here:
    ``IS_WINDOWS`` is pinned, so the snapshot is ``[]`` and no refresh thread
    runs, exactly as on Windows.
    """
    real = subprocess.Popen

    def suspended(args, **kwargs):
        flags = kwargs.pop("creationflags")
        assert flags == CREATE_SUSPENDED, "the CLI got a group of its own"
        proc = real(args, start_new_session=True, **kwargs)
        os.killpg(proc.pid, signal.SIGSTOP)
        return proc

    api = _GroupJobApi()
    monkeypatch.setattr(process_control, "IS_WINDOWS", True)
    monkeypatch.setattr(process_control.subprocess, "Popen", suspended)
    monkeypatch.setattr(process_control, "_REAL_POPEN", suspended)
    monkeypatch.setattr(process_control, "_job_api", lambda: api)
    monkeypatch.setattr(process_control, "live_cli_registry",
                        process_control.LiveCliRegistry(interval_s=3600))
    monkeypatch.setattr(process_control, "taskkill_tree",
                        lambda *a, **k: pytest.fail("Lane A never runs taskkill"))
    yield api
    if api.pid is not None:  # a failed row leaves nothing running
        try:
            os.killpg(api.pid, signal.SIGKILL)
        except OSError:
            pass


@POSIX_ONLY
class TestRealProcessesWithAGroupForTheJob:
    def test_a_timeout_ends_the_grandchild_and_returns_inside_the_drain(
        self, group_job, tmp_path
    ):
        """The gap itself: the grandchild holds the CLI's stdout, so a stop of
        the CLI alone left the drain waiting out its cap, and the grandchild
        running after it."""
        marker = tmp_path / "grandchild.pid"
        timeout = 1.5
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            llm_client._run_provider_cli(
                _cli_argv(marker), capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL)
        elapsed = time.monotonic() - started

        grandchild = _grandchild_pid(marker, timeout=0)
        assert grandchild, "the grandchild never started inside the deadline"
        assert not cli_tree.await_dead([grandchild], timeout=5), "the grandchild survived"
        assert elapsed - timeout < llm_client.LANE_A_DRAIN_S - 1.0, (
            f"{elapsed - timeout:.1f}s past the deadline: the drain waited on a survivor")
        assert caught.value.tree_sweep.job == "terminated"
        assert group_job.terminated == [101] and group_job.closed == [101]

    def test_a_cli_stopped_from_outside_has_its_grandchild_ended(
        self, group_job, tmp_path
    ):
        marker = tmp_path / "grandchild.pid"
        completed = llm_client._run_provider_cli(
            _cli_argv(marker, exit_code=130, stdout="devnull"),
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        assert completed.returncode == 130
        grandchild = _grandchild_pid(marker, timeout=0)
        assert grandchild
        assert not cli_tree.await_dead([grandchild], timeout=5)
        assert group_job.terminated == [101] and group_job.closed == [101]

    def test_a_normal_exit_leaves_the_grandchild_running(self, group_job, tmp_path):
        """As a POSIX CLI's groups are: only a stop ends what a CLI started."""
        marker = tmp_path / "grandchild.pid"
        completed = llm_client._run_provider_cli(
            _cli_argv(marker, exit_code=0, stdout="devnull"),
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        grandchild = _grandchild_pid(marker, timeout=0)
        try:
            assert completed.returncode == 0 and grandchild
            assert cli_tree.running(grandchild)
            assert group_job.terminated == [] and group_job.closed == [101]
        finally:
            if grandchild:
                os.kill(grandchild, signal.SIGKILL)


# ===========================================================================
# 7. The real Win32 calls, on the Windows leg
# ===========================================================================

def _still_running(identity):  # pragma: no cover - Windows
    """The very process *identity* (``(pid, start time)``) names is alive; a
    pid whose start time no longer matches is a recycled one, not ours."""
    pid, start = identity
    return bool(start) and process_control.probe(pid, start) == process_control.ALIVE


def _kill_if_alive(identity):  # pragma: no cover - Windows
    """Cleanup for a failed row, by identity, so a recycled pid is never
    killed. ``SIGTERM`` is ``TerminateProcess`` here; never 0 or 1, which
    ``os.kill`` sends as CTRL_C/CTRL_BREAK events."""
    if identity and _still_running(identity):
        try:
            os.kill(identity[0], signal.SIGTERM)
        except OSError:
            pass


def _await_gone(identity, timeout=10.0):  # pragma: no cover - Windows
    # Strict: with no start time the process could not be told from a
    # recycled pid, and "gone" would hold vacuously.
    assert identity[1], f"the grandchild wrote no start time: {identity}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _still_running(identity):
        time.sleep(0.1)
    return not _still_running(identity)


@pytest.fixture
def own_registry(monkeypatch):
    """A registry of the row's own, so a sweep reaches only the row's CLI."""
    registry = process_control.LiveCliRegistry(interval_s=3600)
    monkeypatch.setattr(process_control, "live_cli_registry", registry)
    return registry


@WINDOWS_ONLY
@pytest.mark.usefixtures("own_registry")
class TestTheRealCallsOnWindows:  # pragma: no cover - Windows
    def test_a_timeout_ends_the_grandchild_and_returns_inside_the_drain(self, tmp_path):
        marker = tmp_path / "grandchild.pid"
        # Long enough for two interpreters (and their venv launchers) to start on
        # a loaded runner; the CLI sleeps a minute, so the deadline always fires.
        timeout = 6.0
        grandchild = None
        started = time.monotonic()
        try:
            with pytest.raises(subprocess.TimeoutExpired) as caught:
                llm_client._run_provider_cli(
                    _cli_argv(marker), capture_output=True, text=True,
                    timeout=timeout, stdin=subprocess.DEVNULL)
            elapsed = time.monotonic() - started
            grandchild = _grandchild(marker, timeout=0)
            assert grandchild, "the grandchild never started inside the deadline"
            assert caught.value.tree_sweep.job == "terminated"
            assert _await_gone(grandchild), "the grandchild outlived its CLI's job"
            # Without the job the drain waited its whole cap on the grandchild.
            assert elapsed - timeout < llm_client.LANE_A_DRAIN_S - 1.0, (
                f"{elapsed - timeout:.1f}s past the deadline")
        finally:
            _kill_if_alive(grandchild or _grandchild(marker, timeout=0))

    def test_the_registry_sweep_ends_the_grandchild(self, tmp_path, own_registry):
        """The Ctrl-C and abort path: a sweep while the slot waits on its CLI.
        The grandchild holds the CLI's stdout, so the slot returns only once
        the job has ended it."""
        registry = own_registry
        marker = tmp_path / "grandchild.pid"
        returned = []
        slot = threading.Thread(
            target=lambda: returned.append(llm_client._run_provider_cli(
                _cli_argv(marker), capture_output=True, text=True, timeout=120,
                stdin=subprocess.DEVNULL)),
            daemon=True)
        grandchild = None
        clis = []
        slot.start()
        try:
            grandchild = _grandchild(marker)
            assert grandchild, "the grandchild never started"
            clis = [entry.proc for entry in registry.entries()]
            assert len(clis) == 1 and clis[0].poll() is None
            swept = registry.sweep(grace_s=1.0)
            [(entry, result)] = swept
            assert entry.job is not None, "the job did not travel with the CLI"
            assert result.job == "terminated", result.describe()
            assert _await_gone(grandchild), "the grandchild outlived the sweep"
            slot.join(15)
            assert not slot.is_alive(), "the slot still waits on its CLI's pipes"
            assert returned and returned[0].returncode != 0
            assert len(registry) == 0
        finally:
            _kill_if_alive(grandchild)
            for cli in clis:
                if cli.poll() is None:
                    cli.kill()

    def test_a_cli_stopped_from_outside_has_its_grandchild_ended(self, tmp_path):
        """130: a CLI that handled a Ctrl-C and exited, its grandchild running on."""
        marker = tmp_path / "grandchild.pid"
        grandchild = None
        try:
            completed = llm_client._run_provider_cli(
                _cli_argv(marker, exit_code=130, stdout="devnull"),
                capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
            grandchild = _grandchild(marker, timeout=0)
            assert completed.returncode == 130
            assert grandchild, "the CLI exited before its grandchild started"
            assert _await_gone(grandchild), "the grandchild outlived its CLI's job"
        finally:
            _kill_if_alive(grandchild or _grandchild(marker, timeout=0))
