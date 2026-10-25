"""A detached run's named Job Object (2026-10-25): the kills by a recorded pid.

Round 8 gave every child the executor spawns a Job Object of its own
(``test_windows_tree_jobs.py``), but a handle does not cross processes: the
kills made by whoever reads a run's ``.status.json`` or registry entry
(``status --abort``, a close-out, ``util runs --sweep``) and the launcher's
failed-handshake teardown all still ended in ``taskkill /T``, which walks
parent pids and can reach an unrelated orphan. So a run's job has a name,
``Local\\multi-llm-run-<run_id>``:

1. **The launcher** creates it and spawns the worker into it suspended, then
   resumes it (``process_control.spawn_detached_run``); a name that already
   exists is refused. Its handle lasts until the handshake resolves.
2. **The worker** opens it, or creates it, and holds it for life
   (``enter_detached_worker(log, run_id=...)``); an existing job it is not in
   is a squatter's and is let go of.
3. **An aborter** opens it by name for TERMINATE and QUERY only, trusts it only
   if a live recorded pid is in it, and kills through it
   (``open_run_job`` / ``kill_run_tree``). Every failure falls back to the
   ``taskkill`` it had, and off Windows nothing here touches a Win32 call.

The rows below run anywhere through a stand-in for the Win32 calls; the
``WINDOWS_ONLY`` rows make the real ones on the Windows leg. Every row pins
``IS_WINDOWS`` itself, so a session that flips it does not change what a row
tests.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import cli_common, detach, process_control, run_registry, status_file  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="the real Win32 calls")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_SUSPENDED = 0x00000004
JOB_OBJECT_QUERY = 0x0004
JOB_OBJECT_TERMINATE = 0x0008

RUN_ID = "0b0c5f8e-2d6a-4a51-9a43-6c0f1e9d7a21"
NAME = "Local\\multi-llm-run-" + RUN_ID

#: The stand-in's pseudo handle for "this process" (``GetCurrentProcess``).
ME = "process:me"


# ===========================================================================
# The stand-in for kernel32
# ===========================================================================

class FakeRunJobApi:
    """The run-job calls, recorded in order, over a small model of named jobs.

    Handles are ints from 101 up, process handles ``"process:<pid>"``; a job is
    its name, and ``members`` holds the pids in each (``"me"`` for the calling
    process). Any method named in ``fail`` reports failure the way the real
    call does (a NULL handle, FALSE, or a failed ``IsProcessInJob``).
    """

    def __init__(self):
        self.calls = []
        self.members = {}
        self.handles = {}
        self.fail = set()
        self.terminated = []
        self.on_terminate = None
        self._next = 100

    # -- the model -----------------------------------------------------------
    def existing(self, name, *pids):
        """A job by *name* already there, with *pids* in it."""
        self.members[name] = {str(p) for p in pids}

    def _open(self, name):
        self._next += 1
        self.handles[self._next] = name
        return self._next

    @staticmethod
    def _pid(process):
        return process.split(":", 1)[1]

    def names(self):
        return [call[0] for call in self.calls]

    # -- the calls ------------------------------------------------------------
    def create(self):
        self.calls.append(("create",))
        return self._open("<anonymous>")

    def create_named(self, name):
        self.calls.append(("create_named", name))
        if "create_named" in self.fail:
            return None, False
        existed = name in self.members
        self.members.setdefault(name, set())
        return self._open(name), existed

    def open_named(self, name, access):
        self.calls.append(("open_named", name, access))
        if "open_named" in self.fail or name not in self.members:
            return None
        return self._open(name)

    def open_process(self, pid):
        self.calls.append(("open_process", pid))
        return f"process:{pid}"

    def query_process(self, pid):
        self.calls.append(("query_process", pid))
        return None if "query_process" in self.fail else f"process:{pid}"

    def current_process(self):
        self.calls.append(("current_process",))
        return ME

    def assign(self, job, process):
        self.calls.append(("assign", job, process))
        if "assign" in self.fail:
            return False
        self.members[self.handles[job]].add(self._pid(process))
        return True

    def resume(self, process):
        self.calls.append(("resume", process))
        return "resume" not in self.fail

    def in_job(self, process, job):
        self.calls.append(("in_job", process, job))
        if "in_job" in self.fail:
            return None
        return self._pid(process) in self.members[self.handles[job]]

    def terminate(self, job, code):
        self.calls.append(("terminate", job, code))
        if "terminate" in self.fail:
            return False
        self.terminated.append(self.handles[job])
        if self.on_terminate is not None:
            self.on_terminate()
        return True

    def close(self, handle):
        self.calls.append(("close", handle))
        self.handles.pop(handle, None)


class SpawnRecorder:
    """``subprocess.Popen`` as the detached spawn sees it; starts nothing.

    Carries a ``_handle`` the way the real one does, so the route under test
    is the one Windows takes (Popen's own handle, no ``OpenProcess``).
    """

    spawned = []
    next_pid = 4242

    def __init__(self, args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.pid = SpawnRecorder.next_pid
        SpawnRecorder.next_pid += 1
        self._handle = f"process:{self.pid}"
        self.killed = self.waited = False
        self.stdin = self.stdout = self.stderr = None
        SpawnRecorder.spawned.append(self)

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        return 1


class Leader:
    """The recorded pid 4242: alive until the break, a taskkill or the job
    ends it, whichever the row arranges."""

    def __init__(self, pid=4242):
        self.pid = pid
        self.alive = True

    def process_alive(self, pid):
        return self.alive if pid and int(pid) == self.pid else False

    def end(self, *_args, **_kwargs):
        self.alive = False


@pytest.fixture
def windows(monkeypatch):
    """The run-job route as it runs on Windows, with the stand-ins installed.

    ``taskkill_tree`` and ``_send_ctrl_break`` only record: a real break could
    reach a console group on the Windows leg, and a real taskkill could reach a
    stranger whose pid is 4242.
    """
    SpawnRecorder.spawned = []
    SpawnRecorder.next_pid = 4242
    api = FakeRunJobApi()
    api.taskkills = []
    api.breaks = []
    monkeypatch.setattr(process_control, "IS_WINDOWS", True)
    monkeypatch.setattr(process_control.subprocess, "Popen", SpawnRecorder)
    monkeypatch.setattr(process_control, "_REAL_POPEN", SpawnRecorder)
    monkeypatch.setattr(process_control, "_job_api", lambda: api)
    monkeypatch.setattr(process_control, "_RUN_JOB_HANDLES", {})
    monkeypatch.setattr(process_control, "taskkill_tree",
                        lambda pid, timeout=30.0: api.taskkills.append(pid))
    monkeypatch.setattr(process_control, "_send_ctrl_break", api.breaks.append)
    return api


@pytest.fixture
def posix(monkeypatch):
    """Off Windows: no job call may be made at all."""
    monkeypatch.setattr(process_control, "IS_WINDOWS", False)
    monkeypatch.setattr(process_control, "_job_api",
                        lambda: pytest.fail("a Win32 job call off Windows"))


@pytest.fixture
def host_locks(monkeypatch):
    """The phase-dir locks of the OS the row runs on, whatever ``IS_WINDOWS``
    says: the flag would send them to ``msvcrt`` on Linux, and to ``fcntl`` on
    the Windows leg. Each lock call runs the real helper with the flag set to
    the host's own value for that call only."""
    host = os.name == "nt"

    def on_the_host(helper):
        def call(fd):
            pinned = process_control.IS_WINDOWS
            process_control.IS_WINDOWS = host
            try:
                return helper(fd)
            finally:
                process_control.IS_WINDOWS = pinned
        return call

    monkeypatch.setattr(process_control, "_take_fd_lock",
                        on_the_host(process_control._take_fd_lock))
    monkeypatch.setattr(process_control, "_drop_fd_lock",
                        on_the_host(process_control._drop_fd_lock))


# ===========================================================================
# 1. The launcher: the worker starts inside the run's job
# ===========================================================================

def _spawn(tmp_path, run_id=RUN_ID):
    return process_control.spawn_detached_run(
        ["uv", "run"], cwd=str(tmp_path), env={"A": "B"},
        log_path=str(tmp_path / "orchestrator-run.log"), run_id=run_id,
    )


class TestTheLauncher:
    def test_the_worker_starts_suspended_inside_the_runs_named_job(self, windows, tmp_path):
        pid, job = _spawn(tmp_path)
        [proc] = SpawnRecorder.spawned
        assert pid == proc.pid == 4242
        assert proc.kwargs["creationflags"] == (
            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED)
        # Otherwise the detached spawn it always was: one log handle as both
        # streams, nothing inherited, no stdin.
        assert proc.kwargs["stdout"] is proc.kwargs["stderr"]
        assert proc.kwargs["close_fds"] is True
        assert proc.kwargs["stdin"] is subprocess.DEVNULL
        assert proc.kwargs["cwd"] == str(tmp_path) and proc.kwargs["env"] == {"A": "B"}
        # The job is made, by the run's name, before the worker exists; the
        # worker joins it before it runs an instruction.
        assert windows.calls == [
            ("create_named", NAME), ("assign", 101, "process:4242"),
            ("resume", "process:4242"),
        ]
        assert windows.members[NAME] == {"4242"}
        # The launcher's handle is handed back, and closing it never kills.
        assert isinstance(job, process_control.TreeJob)
        job.close()
        assert windows.calls[-1] == ("close", 101)
        assert "terminate" not in windows.names() and not proc.killed

    def test_a_squatted_name_is_never_used(self, windows, tmp_path):
        windows.existing(NAME, 999)
        pid, job = _spawn(tmp_path)
        [proc] = SpawnRecorder.spawned
        assert job is None and pid == proc.pid
        assert windows.calls == [("create_named", NAME), ("close", 101)]
        assert proc.kwargs["creationflags"] == DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        assert windows.members[NAME] == {"999"}, "the worker joined a stranger's job"

    @pytest.mark.parametrize("run_id", ["", "a\\b", "x" * 200, "r\u00e9sum\u00e9"])
    def test_a_run_id_that_names_no_job_spawns_as_before(self, windows, tmp_path, run_id):
        _, job = _spawn(tmp_path, run_id=run_id)
        assert job is None and windows.calls == []
        [proc] = SpawnRecorder.spawned
        assert proc.kwargs["creationflags"] == DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    def test_a_job_that_cannot_be_made_spawns_as_before(self, windows, tmp_path):
        windows.fail.add("create_named")
        _, job = _spawn(tmp_path)
        assert job is None
        [proc] = SpawnRecorder.spawned
        assert proc.kwargs["creationflags"] == DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    def test_a_worker_that_cannot_join_runs_without_the_job(self, windows, tmp_path):
        windows.fail.add("assign")
        _, job = _spawn(tmp_path)
        assert job is None
        [proc] = SpawnRecorder.spawned
        # Resumed anyway, and the job let go of: the worker runs, taskkill
        # stays its kill, and the worker makes the job itself.
        assert ("resume", "process:4242") in windows.calls
        assert ("close", 101) in windows.calls and not proc.killed

    def test_a_worker_that_cannot_be_resumed_is_spawned_again_without_a_job(
        self, windows, tmp_path
    ):
        windows.fail.add("resume")
        pid, job = _spawn(tmp_path)
        first, second = SpawnRecorder.spawned
        assert first.killed and first.waited, "a suspended worker was left behind"
        assert ("close", 101) in windows.calls
        assert job is None and pid == second.pid
        assert second.kwargs["creationflags"] == DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    def test_a_spawn_that_fails_closes_the_job(self, windows, tmp_path, monkeypatch):
        def refuse(args, **kwargs):
            raise FileNotFoundError("no such uv")

        monkeypatch.setattr(process_control.subprocess, "Popen", refuse)
        monkeypatch.setattr(process_control, "_REAL_POPEN", refuse)
        with pytest.raises(FileNotFoundError):
            _spawn(tmp_path)
        assert windows.calls == [("create_named", NAME), ("close", 101)]

    def test_a_replaced_spawn_detached_is_called_as_it_is(self, windows, tmp_path,
                                                          monkeypatch):
        """The launch tests' doubles replace ``spawn_detached`` and start
        nothing, so there is nothing to put in a job."""
        seen = []

        def fake_spawn(command, *, cwd, env, log_path):
            seen.append((list(command), cwd, env, log_path))
            return 77

        monkeypatch.setattr(process_control, "spawn_detached", fake_spawn)
        assert _spawn(tmp_path) == (77, None)
        assert seen == [(["uv", "run"], str(tmp_path), {"A": "B"},
                         str(tmp_path / "orchestrator-run.log"))]
        assert windows.calls == [] and SpawnRecorder.spawned == []

    def test_spawn_detached_itself_makes_no_run_job(self, windows, tmp_path):
        pid = process_control.spawn_detached(
            ["uv", "run"], cwd=str(tmp_path), env={},
            log_path=str(tmp_path / "orchestrator-run.log"))
        assert pid == 4242 and windows.calls == []
        [proc] = SpawnRecorder.spawned
        assert proc.kwargs["creationflags"] == DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    def test_off_windows_it_is_spawn_detached(self, posix, tmp_path, monkeypatch):
        seen = []

        def spawn(command, *, cwd, env, log_path):
            seen.append((list(command), cwd, env, log_path))
            return 4343

        monkeypatch.setattr(process_control, "spawn_detached", spawn)
        assert _spawn(tmp_path) == (4343, None)
        assert len(seen) == 1


# ===========================================================================
# 2. The worker: it holds the run's job for life
# ===========================================================================

class TestTheWorker:
    @pytest.fixture
    def reapers(self, windows, monkeypatch):
        """The calls already made each time the anonymous reaper job forms."""
        made = []
        # On the log already, so no descriptor is reopened (the stdio half of
        # the entry is test_detach's TestWindowsSeam).
        monkeypatch.setattr(process_control, "_stdio_already_on", lambda _log: True)
        monkeypatch.setattr(process_control, "_assign_own_job_object",
                            lambda: made.append(list(windows.calls)))
        return made

    def test_the_worker_keeps_the_job_its_launcher_started_it_in(self, windows, reapers):
        windows.existing(NAME, "me")
        process_control.enter_detached_worker("C:\\nope\\run.log", run_id=RUN_ID)
        assert windows.calls == [
            ("create_named", NAME), ("current_process",), ("in_job", ME, 101),
        ]
        # Held for life: never closed, and recorded.
        assert process_control._RUN_JOB_HANDLES == {NAME: 101}
        # The reaper still forms, after the hold, so it nests inside the job.
        assert reapers == [windows.calls]

    def test_a_worker_with_no_job_of_its_name_makes_it_and_joins_it(self, windows, reapers):
        """A worker launched some other way, or whose launcher could not."""
        process_control.enter_detached_worker("C:\\nope\\run.log", run_id=RUN_ID)
        assert windows.calls == [
            ("create_named", NAME), ("current_process",), ("assign", 101, ME),
        ]
        assert windows.members[NAME] == {"me"}
        assert process_control._RUN_JOB_HANDLES == {NAME: 101}
        assert len(reapers) == 1

    def test_a_job_the_worker_is_not_in_is_a_squatters_and_let_go_of(
        self, windows, reapers
    ):
        windows.existing(NAME, 999)
        process_control.enter_detached_worker("C:\\nope\\run.log", run_id=RUN_ID)
        assert windows.calls[-1] == ("close", 101)
        assert "assign" not in windows.names(), "the worker joined a stranger's job"
        assert process_control._RUN_JOB_HANDLES == {}
        assert len(reapers) == 1, "the anonymous reaper is what the worker keeps"

    @pytest.mark.parametrize("failure", ["create_named", "in_job"])
    def test_a_job_that_cannot_be_held_leaves_the_worker_as_it_was(
        self, windows, reapers, failure
    ):
        windows.existing(NAME, "me")
        windows.fail.add(failure)
        process_control.enter_detached_worker("C:\\nope\\run.log", run_id=RUN_ID)
        assert process_control._RUN_JOB_HANDLES == {}
        assert len(reapers) == 1

    def test_a_second_entry_holds_the_job_once(self, windows, reapers):
        windows.existing(NAME, "me")
        process_control.enter_detached_worker("C:\\nope\\run.log", run_id=RUN_ID)
        process_control.enter_detached_worker("C:\\nope\\run.log", run_id=RUN_ID)
        assert windows.names().count("create_named") == 1
        assert len(reapers) == 2

    def test_no_run_id_is_the_worker_it_always_was(self, windows, reapers):
        process_control.enter_detached_worker("C:\\nope\\run.log")
        assert windows.calls == [] and reapers == [[]]

    def test_off_windows_the_run_id_reaches_no_job(self, posix, monkeypatch, tmp_path):
        monkeypatch.setattr(process_control, "_assign_own_job_object",
                            lambda: pytest.fail("a Job Object off Windows"))
        monkeypatch.setattr(process_control, "_stdio_already_on", lambda _log: True)
        process_control.enter_detached_worker(str(tmp_path / "run.log"), run_id=RUN_ID)

    def test_begin_worker_run_hands_its_run_id_over(self, monkeypatch, tmp_path):
        """The worker's run id is the launcher's ``--internal-run-id``."""
        entered = []

        class Handle:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def start(self):
                pass

        monkeypatch.setenv(detach.WORKER_ENV, "1")
        monkeypatch.delenv(detach.REPLAY_FLAGS_ENV, raising=False)
        monkeypatch.delenv(detach.PLAN_ARG_ENV, raising=False)
        monkeypatch.setattr(process_control, "enter_detached_worker",
                            lambda log, run_id=None: entered.append((log, run_id)))
        monkeypatch.setattr(detach, "RunHandle", Handle)
        phase = tmp_path / "review-plan"
        phase.mkdir()
        handle = detach.begin_worker_run(
            phase_dir=str(phase), mode="review-plan",
            plan_file=str(tmp_path / "plan.md"),
            args=argparse.Namespace(internal_run_id=RUN_ID, host=None),
        )
        assert entered == [(detach.log_path(str(phase)), RUN_ID)]
        assert handle.kwargs["run_id"] == RUN_ID


# ===========================================================================
# 3. The aborter: open by name, trust only a pid that is in it
# ===========================================================================

class TestOpenRunJob:
    def test_it_opens_by_name_for_terminate_and_query_only(self, windows, monkeypatch):
        windows.existing(NAME, 4242)
        monkeypatch.setattr(process_control, "process_alive", Leader().process_alive)
        job = process_control.open_run_job(RUN_ID, 4242)
        assert isinstance(job, process_control.TreeJob)
        assert windows.calls == [
            ("open_named", NAME, JOB_OBJECT_TERMINATE | JOB_OBJECT_QUERY),
            ("query_process", 4242), ("in_job", "process:4242", 101),
            ("close", "process:4242"),
        ]
        job.close()
        assert windows.calls[-1] == ("close", 101)

    def test_a_live_pid_that_is_not_in_the_job_is_not_trusted(self, windows, monkeypatch):
        """A recycled pid, or a squatter's job: either way, not the run's."""
        windows.existing(NAME, 999)
        monkeypatch.setattr(process_control, "process_alive", Leader().process_alive)
        assert process_control.open_run_job(RUN_ID, 4242) is None
        assert windows.calls[-1] == ("close", 101)

    @pytest.mark.parametrize("failure", ["query_process", "in_job"])
    def test_a_membership_that_cannot_be_read_is_not_trusted(
        self, windows, monkeypatch, failure
    ):
        windows.existing(NAME, 4242)
        windows.fail.add(failure)
        monkeypatch.setattr(process_control, "process_alive", Leader().process_alive)
        assert process_control.open_run_job(RUN_ID, 4242) is None
        assert windows.calls[-1] == ("close", 101)

    @pytest.mark.parametrize("pid", [4242, 0, None])
    def test_a_dead_or_unrecorded_pid_trusts_the_name(self, windows, monkeypatch, pid):
        windows.existing(NAME, 4343)
        monkeypatch.setattr(process_control, "process_alive", lambda _pid: False)
        job = process_control.open_run_job(RUN_ID, pid)
        assert job is not None
        assert windows.names() == ["open_named"]
        job.close()

    def test_a_pid_that_dies_during_the_membership_check_trusts_the_name(
        self, windows, monkeypatch
    ):
        """The worker exits between the liveness look and ``IsProcessInJob``:
        the "no" comes from its being gone, and a dead pid trusts the name, so
        what it left in the job is still reachable (Fable review, 2026-10-25)."""
        windows.existing(NAME, 999)
        looks = iter([True, False])
        monkeypatch.setattr(process_control, "process_alive", lambda _pid: next(looks))
        job = process_control.open_run_job(RUN_ID, 4242)
        assert job is not None
        job.close()

    def test_no_job_of_that_name_is_none(self, windows, monkeypatch):
        monkeypatch.setattr(process_control, "process_alive", Leader().process_alive)
        assert process_control.open_run_job(RUN_ID, 4242) is None
        assert windows.names() == ["open_named"]

    @pytest.mark.parametrize("run_id", [None, "", "a\\b"])
    def test_a_run_id_that_names_no_job_opens_nothing(self, windows, run_id):
        assert process_control.open_run_job(run_id, 4242) is None
        assert windows.calls == []

    def test_without_the_job_api_it_is_none(self, windows, monkeypatch):
        monkeypatch.setattr(process_control, "_job_api", lambda: None)
        assert process_control.open_run_job(RUN_ID, 4242) is None

    def test_off_windows_it_is_none_and_touches_nothing(self, posix):
        assert process_control.open_run_job(RUN_ID, 4242) is None
        assert process_control.run_job_name(RUN_ID) == NAME


class TestKillRunTree:
    def test_the_escalation_ends_the_job_never_taskkill(self, windows, monkeypatch):
        """The leader ignores the break; the escalation is the job's."""
        leader = Leader()
        windows.existing(NAME, 4242)
        windows.on_terminate = leader.end
        monkeypatch.setattr(process_control, "process_alive", leader.process_alive)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        verdict = process_control.kill_run_tree(
            RUN_ID, 4242, start_time="t0", escalate_after_s=0.1, poll_interval_s=0.01)
        assert verdict == "killed"
        assert windows.breaks == [4242], "the graceful break still comes first"
        assert windows.taskkills == []
        assert windows.names()[-2:] == ["terminate", "close"]
        assert windows.terminated == [NAME]

    def test_a_leader_that_took_the_break_still_has_its_tree_ended(
        self, windows, monkeypatch
    ):
        leader = Leader()
        windows.existing(NAME, 4242)
        monkeypatch.setattr(process_control, "process_alive", leader.process_alive)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        monkeypatch.setattr(process_control, "_send_ctrl_break", leader.end)
        assert process_control.kill_run_tree(RUN_ID, 4242, start_time="t0") == "killed"
        assert windows.terminated == [NAME] and windows.taskkills == []

    @pytest.mark.parametrize("why", ["pid-not-in-job", "job-cannot-open"])
    def test_without_a_trusted_job_it_is_taskkill_as_before(
        self, windows, monkeypatch, why
    ):
        leader = Leader()
        if why == "pid-not-in-job":
            windows.existing(NAME, 999)
        windows.on_terminate = lambda: pytest.fail("a job that was not trusted")
        monkeypatch.setattr(process_control, "taskkill_tree",
                            lambda pid, timeout=30.0: (windows.taskkills.append(pid),
                                                       leader.end()))
        monkeypatch.setattr(process_control, "process_alive", leader.process_alive)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        verdict = process_control.kill_run_tree(
            RUN_ID, 4242, start_time="t0", escalate_after_s=0.0, poll_interval_s=0.01)
        assert verdict == "killed"
        assert windows.taskkills == [4242]
        assert windows.terminated == []

    @pytest.mark.parametrize("pid", [0, None])
    def test_an_unrecorded_pid_still_has_its_job_ended(self, windows, monkeypatch, pid):
        """A record with no pid: nothing to probe or signal, but the name still
        finds the tree, and the job it opened is what ends it (Fable review,
        2026-10-25)."""
        windows.existing(NAME, 4343)
        monkeypatch.setattr(process_control, "process_alive", lambda _pid: False)
        assert process_control.kill_run_tree(RUN_ID, pid) == "gone"
        assert windows.terminated == [NAME]
        assert windows.taskkills == [] and windows.breaks == []
        assert windows.names()[-2:] == ["terminate", "close"]

    def test_a_job_route_that_raises_is_the_kill_it_was(self, windows, monkeypatch):
        seen = []

        def boom(*_a, **_k):
            raise RuntimeError("no job today")

        monkeypatch.setattr(process_control, "open_run_job", boom)
        monkeypatch.setattr(process_control, "kill_tree",
                            lambda pid, **kwargs: (seen.append((pid, kwargs)), "gone")[1])
        assert process_control.kill_run_tree(RUN_ID, 4242, pgid=7) == "gone"
        assert seen == [(4242, {"pgid": 7})]

    def test_off_windows_it_is_the_same_kill_tree_call(self, posix, monkeypatch):
        seen = []
        monkeypatch.setattr(process_control, "kill_tree",
                            lambda pid, **kwargs: (seen.append((pid, kwargs)), "killed")[1])
        assert process_control.kill_run_tree(
            RUN_ID, 4242, start_time="t0", pgid=7) == "killed"
        assert seen == [(4242, {"start_time": "t0", "pgid": 7})], "no job= off Windows"


# ===========================================================================
# 4. The four kill sites
# ===========================================================================

def _record(phase, plan, *, pid=4242, start="t0", **extra):
    status_file.claim_run(str(phase), {
        "run_id": RUN_ID, "state": "running", "pid": pid,
        "process_start_time": start, "pgid": None, "detached": True,
        "mode": "review-plan", "plan_file": str(plan),
        "heartbeat_at": status_file.now_iso(), "exit_code": None, **extra,
    })


@pytest.fixture
def run_dirs(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    phase = tmp_path / "plan" / "review-plan"
    phase.mkdir(parents=True)
    return plan, phase


class TestTheKillSites:
    def test_an_abort_kills_through_the_runs_job(self, windows, host_locks, run_dirs,
                                                 monkeypatch):
        plan, phase = run_dirs
        _record(phase, plan)
        leader = Leader()
        windows.existing(NAME, 4242)
        monkeypatch.setattr(process_control, "process_alive", leader.process_alive)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        monkeypatch.setattr(process_control, "_send_ctrl_break", leader.end)
        result = detach.abort_run(str(phase))
        assert result["result"] == "aborted"
        assert ("open_named", NAME, JOB_OBJECT_TERMINATE | JOB_OBJECT_QUERY) in windows.calls
        assert ("in_job", "process:4242", 101) in windows.calls
        assert windows.terminated == [NAME] and windows.taskkills == []
        assert windows.calls[-1] == ("close", 101), "the aborter's handle was kept"
        assert status_file.read_run(str(phase))["state"] == detach.RUN_STATE_ABORTED

    def test_an_abort_with_no_job_to_open_kills_as_before(
        self, windows, host_locks, run_dirs, monkeypatch
    ):
        """No job by the run's name (a worker that could not hold one): the
        abort is the job-less kill it was, and terminates nothing."""
        plan, phase = run_dirs
        _record(phase, plan)
        leader = Leader()
        monkeypatch.setattr(process_control, "process_alive", leader.process_alive)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        monkeypatch.setattr(process_control, "_send_ctrl_break", leader.end)
        assert detach.abort_run(str(phase))["result"] == "aborted"
        assert windows.names() == ["open_named"] and windows.terminated == []

    def test_a_close_out_ends_what_a_dead_leader_left_in_its_job(
        self, windows, host_locks, run_dirs, monkeypatch
    ):
        plan, phase = run_dirs
        _record(phase, plan)
        windows.existing(NAME, 4343)  # the leader is gone; a child is left
        monkeypatch.setattr(process_control, "process_alive", lambda _pid: False)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.DEAD)
        assert detach.close_out_dead_run(str(phase)) is True
        assert windows.names() == ["open_named", "terminate", "close"]
        assert windows.terminated == [NAME] and windows.taskkills == []
        assert status_file.read_run(str(phase))["state"] == detach.RUN_STATE_FAILED

    def test_a_sweep_prunes_an_old_entry_through_its_job(self, windows, host_locks,
                                                         tmp_path, monkeypatch):
        root = tmp_path / "project"
        phase = tmp_path / "phase"
        phase.mkdir()
        old = run_registry.write_entry(str(root), {
            "run_id": RUN_ID, "pid": 4242, "process_start_time": "t0", "pgid": None,
            "phase_dir": str(phase), "mode": "review-plan",
        })
        eight_days = time.time() - 8 * 86400
        os.utime(old, (eight_days, eight_days))
        leader = Leader()
        windows.existing(NAME, 4242)
        monkeypatch.setattr(process_control, "process_alive", leader.process_alive)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        monkeypatch.setattr(process_control, "_send_ctrl_break", leader.end)
        [row] = run_registry.sweep(str(root))
        assert (row["run_id"], row["action"]) == (RUN_ID, "killed")
        assert windows.terminated == [NAME] and windows.taskkills == []
        assert ("close", 101) in windows.calls

    @pytest.fixture
    def launch_ready(self, run_dirs, tmp_path, monkeypatch):
        """``detach.launch`` up to its spawn, with a handshake that never comes."""
        plan, phase = run_dirs
        uv = tmp_path / "fake-uv"
        uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        uv.chmod(0o755)
        monkeypatch.setenv(detach.UV_ENV, str(uv))
        monkeypatch.setenv(detach.HANDSHAKE_TIMEOUT_ENV, "0.1")
        monkeypatch.setattr(detach, "missing_provider_binaries", lambda *a, **k: {})
        monkeypatch.setattr(detach, "missing_runner_binaries", lambda *a, **k: {})
        cli_common.activate_result(cli_common.ResultContext(mode="review-plan"))

        def launch():
            return detach.launch(
                mode="review-plan",
                argv=["--plan-file", str(plan), "--detach", "--yes"],
                phase_dir=str(phase), plan_file=str(plan), phase="review-plan",
                args=argparse.Namespace(force=False, if_not_running=False),
            )

        try:
            yield launch
        finally:
            cli_common.deactivate_result()

    def test_a_failed_handshake_ends_the_tree_through_the_launchers_job(
        self, windows, host_locks, launch_ready, monkeypatch
    ):
        """The real ``spawn_detached_run``, on the stand-ins: the worker never
        publishes, so the launcher kills the job it started it in, and lets go."""
        leader = Leader()
        monkeypatch.setattr(process_control, "process_alive", leader.process_alive)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.UNKNOWN)
        monkeypatch.setattr(process_control, "_send_ctrl_break", leader.end)
        with pytest.raises(detach.DetachError, match="never published"):
            launch_ready()
        # The preflight's own `git` probes reach the recorder too; the worker
        # is the spawn that carries the run id.
        [proc] = [p for p in SpawnRecorder.spawned if "--internal-run-id" in p.args]
        assert proc.kwargs["creationflags"] & CREATE_SUSPENDED
        [name] = [call[1] for call in windows.calls if call[0] == "create_named"]
        assert name.startswith("Local\\multi-llm-run-")
        assert proc.args[proc.args.index("--internal-run-id") + 1] == name.split("-run-", 1)[1]
        assert windows.names()[-2:] == ["terminate", "close"]
        assert windows.taskkills == []

    def test_a_handshake_that_resolves_lets_the_launchers_job_go_unkilled(
        self, windows, host_locks, launch_ready, run_dirs, monkeypatch
    ):
        _plan, phase = run_dirs

        def published(phase_dir, run_id):
            return {"run_id": run_id, "state": "running", "pid": 4242}

        monkeypatch.setattr(detach, "_await_handshake", published)
        assert launch_ready() == 0
        assert windows.names()[-1] == "close" and windows.terminated == []

    def test_off_windows_a_failed_handshake_is_the_kill_it_was(
        self, posix, host_locks, launch_ready, monkeypatch
    ):
        seen = []
        monkeypatch.setattr(process_control, "spawn_detached",
                            lambda command, *, cwd, env, log_path: 4242)
        monkeypatch.setattr(process_control, "kill_tree",
                            lambda pid, **kwargs: (seen.append((pid, kwargs)), "gone")[1])
        with pytest.raises(detach.DetachError, match="never published"):
            launch_ready()
        assert seen == [(4242, {"escalate_after_s": 1.0})]

    @pytest.mark.parametrize("site", ["abort", "close-out", "sweep"])
    def test_off_windows_each_site_calls_kill_tree_as_it_did(
        self, posix, host_locks, run_dirs, tmp_path, monkeypatch, site
    ):
        """POSIX, byte for byte: the same pid and keywords, no ``job``."""
        plan, phase = run_dirs
        seen = []
        monkeypatch.setattr(process_control, "kill_tree",
                            lambda pid, **kwargs: (seen.append((pid, kwargs)), "killed")[1])
        if site == "sweep":
            root = tmp_path / "project"
            old = run_registry.write_entry(str(root), {
                "run_id": RUN_ID, "pid": 4242, "process_start_time": "t0", "pgid": 77,
                "phase_dir": str(tmp_path / "elsewhere"), "mode": "review-plan",
            })
            os.utime(old, (time.time() - 8 * 86400,) * 2)
            monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
            run_registry.sweep(str(root))
            assert seen == [(4242, {"start_time": "t0", "pgid": 77})]
            return
        _record(phase, plan, pgid=77)
        if site == "abort":
            monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
            detach.abort_run(str(phase))
            assert seen == [(4242, {"start_time": "t0", "pgid": 77})]
        else:
            detach.close_out_dead_run(str(phase))
            assert seen == [(4242, {"start_time": "t0", "pgid": 77,
                                    "escalate_after_s": 1.0})]


# ===========================================================================
# 5. The real calls (the Windows leg)
# ===========================================================================

#: A stand-in worker: it enters the way ``begin_worker_run`` does, starts a
#: grandchild, and writes its pids. The grandchild is started through
#: ``sys.executable``, which on Windows is the venv's launcher, so it sits
#: under that launcher's own SILENT_BREAKAWAY_OK job as a provider CLI does.
#: Each pid goes with its start time, read while the worker still holds the
#: process (a pid is not reused while a handle to it is open), so a row's
#: cleanup and its "gone" check never act on a recycled pid. ``in_job`` is the
#: worker's own raw ``IsProcessInJob`` against the run's job, so a Windows-leg
#: failure of ``held`` says in one line whether the nested-job membership
#: check is what said no (Fable review, 2026-10-25).
_STAND_IN_WORKER = r'''
import json, os, subprocess, sys, time
sys.path.insert(0, {skill!r})
from utils import process_control

RUN_ID, MARKER, LOG = {run_id!r}, {marker!r}, {log!r}
process_control.enter_detached_worker(LOG, run_id=RUN_ID)
held = process_control.run_job_name(RUN_ID) in process_control._RUN_JOB_HANDLES
in_job = None
api = process_control._job_api()
if api is not None:
    probe = api.open_named(process_control.run_job_name(RUN_ID),
                           process_control._JOB_OBJECT_QUERY)
    if probe is not None:
        in_job = api.in_job(api.current_process(), probe)
        api.close(probe)
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
tmp = MARKER + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump({{
        "pid": os.getpid(), "grandchild": grandchild.pid, "held": held,
        "in_job": in_job,
        "pid_start": process_control.process_start_time(os.getpid()),
        "grandchild_start": process_control.process_start_time(grandchild.pid),
    }}, f)
os.replace(tmp, MARKER)
time.sleep(120)
'''


def _launch_stand_in(tmp_path, run_id):
    marker = tmp_path / "worker.json"
    log = tmp_path / "orchestrator-run.log"
    script = tmp_path / "stand-in-worker.py"
    script.write_text(_STAND_IN_WORKER.format(
        skill=str(SKILL_ROOT), run_id=run_id, marker=str(marker), log=str(log)),
        encoding="utf-8")
    pid, job = process_control.spawn_detached_run(
        [sys.executable, str(script)], cwd=str(tmp_path), env=dict(os.environ),
        log_path=str(log), run_id=run_id)
    # The launcher's child waits on the worker, which sleeps two minutes, so
    # it is alive here and this is its start time, not a recycled pid's.
    return (pid, process_control.process_start_time(pid)), job, marker, log


def _await_marker(marker, log, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if marker.exists():
            try:
                return json.loads(marker.read_text(encoding="utf-8"))
            except ValueError:
                pass
        time.sleep(0.1)
    tail = log.read_text(encoding="utf-8", errors="replace")[-2000:] if log.exists() else ""
    pytest.fail(f"the stand-in worker never reported its pids; log:\n{tail}")


def _identities(launched, info):
    """Every (pid, start time) a row learned: the launcher's child, then the
    worker's and the grandchild's as the worker reported them."""
    found = [launched] if launched else []
    for key in ("pid", "grandchild"):
        if info.get(key):
            found.append((info[key], info.get(key + "_start")))
    return found


def _still_running(identity):
    """The very process *identity* names is alive. A pid whose start time no
    longer matches (or was never read) is a recycled one, or gone: not ours."""
    pid, start = identity
    return bool(start) and process_control.probe(pid, start) == process_control.ALIVE


def _assert_gone(identities, timeout=15.0):
    # Strict: an identity with no start time would count as gone vacuously.
    unread = [pid for pid, start in identities if not start]
    assert not unread, f"no start time was read for {unread}"
    deadline = time.time() + timeout
    while time.time() < deadline and any(_still_running(i) for i in identities):
        time.sleep(0.1)
    still = [pid for pid, start in identities if _still_running((pid, start))]
    assert not still, f"outlived the kill: {still}"


def _reap(job, identities):  # pragma: no cover - Windows cleanup
    """Leave nothing behind whatever failed: the job if there is one, then each
    process the row learned, alone (``TerminateProcess``, never a tree walk),
    and only while its start time still matches, so a recycled pid is never
    killed (Fable review, 2026-10-25)."""
    if job is not None:
        job.terminate()
        job.close()
    for identity in identities:
        if _still_running(identity):
            try:
                os.kill(identity[0], signal.SIGTERM)
            except OSError:
                pass


@pytest.fixture
def no_real_break(monkeypatch):
    """The stand-in has no console, and the test's own console group is not
    the stand-in's business: the graceful break is recorded, never sent."""
    breaks = []
    monkeypatch.setattr(process_control, "_send_ctrl_break", breaks.append)
    return breaks


@WINDOWS_ONLY
class TestTheRealCalls:  # pragma: no cover - Windows
    def test_a_failed_handshake_ends_the_whole_tree_through_the_launchers_job(
        self, tmp_path, no_real_break
    ):
        run_id = str(uuid.uuid4())
        launched, job, marker, log = _launch_stand_in(tmp_path, run_id)
        info = {}
        try:
            assert job is not None, "no named Job Object on Windows"
            info = _await_marker(marker, log)
            assert info["held"] is True, f"the worker did not hold its run's job: {info}"
            # The grandchild is in the run's job, under the venv launcher's
            # job and the worker's reaper: IsProcessInJob sees the whole chain.
            seen = process_control.open_run_job(run_id, info["grandchild"])
            assert seen is not None, (
                f"IsProcessInJob said no for the grandchild in nested jobs: {info}")
            seen.close()
            verdict = process_control.kill_tree(
                launched[0], start_time=None, escalate_after_s=0.5,
                poll_interval_s=0.05, job=job)
            assert verdict == "killed"
            _assert_gone(_identities(launched, info))
        finally:
            _reap(job, _identities(launched, info))

    def test_an_abort_opens_the_job_by_name_after_the_launcher_let_go(
        self, tmp_path, no_real_break, monkeypatch
    ):
        """The handshake's end: only the worker's own handle keeps the name."""
        taskkills = []
        monkeypatch.setattr(process_control, "taskkill_tree",
                            lambda pid, timeout=30.0: taskkills.append(pid))
        run_id = str(uuid.uuid4())
        launched, job, marker, log = _launch_stand_in(tmp_path, run_id)
        info = {}
        try:
            assert job is not None, "no named Job Object on Windows"
            info = _await_marker(marker, log)
            assert info["held"] is True, f"the worker did not hold its run's job: {info}"
            job.close()
            job = None
            phase = tmp_path / "plan" / "review-plan"
            phase.mkdir(parents=True)
            status_file.claim_run(str(phase), {
                "run_id": run_id, "state": "running", "pid": info["pid"],
                "process_start_time": process_control.process_start_time(info["pid"]),
                "pgid": None, "detached": True, "mode": "review-plan",
                "plan_file": str(tmp_path / "plan.md"),
                "heartbeat_at": status_file.now_iso(), "exit_code": None,
            })
            # The break is not sent, so this waits out the abort's 10 s grace.
            result = detach.abort_run(str(phase))
            assert result["result"] == "aborted", result
            assert taskkills == [], "the abort fell back to taskkill"
            _assert_gone(_identities(launched, info))
            assert status_file.read_run(str(phase))["state"] == detach.RUN_STATE_ABORTED
        finally:
            _reap(job, _identities(launched, info))

    def test_a_close_out_ends_what_is_left_in_a_dead_leaders_job(
        self, tmp_path, no_real_break, monkeypatch
    ):
        """The record names a leader that is gone; its job is still held open
        (here by the launcher's handle), so the close-out ends what is in it."""
        taskkills = []
        monkeypatch.setattr(process_control, "taskkill_tree",
                            lambda pid, timeout=30.0: taskkills.append(pid))
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait(timeout=60)
        run_id = str(uuid.uuid4())
        launched, job, marker, log = _launch_stand_in(tmp_path, run_id)
        info = {}
        try:
            assert job is not None, "no named Job Object on Windows"
            info = _await_marker(marker, log)
            phase = tmp_path / "plan" / "review-plan"
            phase.mkdir(parents=True)
            status_file.claim_run(str(phase), {
                "run_id": run_id, "state": "running", "pid": gone.pid,
                "process_start_time": "0", "pgid": None, "detached": True,
                "mode": "review-plan", "plan_file": str(tmp_path / "plan.md"),
                "heartbeat_at": status_file.now_iso(), "exit_code": None,
            })
            assert detach.close_out_dead_run(str(phase)) is True
            assert taskkills == []
            _assert_gone(_identities(launched, info))
        finally:
            _reap(job, _identities(launched, info))

    def test_a_squatted_name_is_refused_by_the_launcher_and_the_worker(
        self, tmp_path, no_real_break
    ):
        api = process_control._job_api()
        assert api is not None, "no job API on Windows"
        run_id = str(uuid.uuid4())
        name = process_control.run_job_name(run_id)
        squat, existed = api.create_named(name)
        assert squat is not None and existed is False
        info = {}
        launched, job = None, None
        try:
            # The real ERROR_ALREADY_EXISTS, read back through ctypes.
            again, existed = api.create_named(name)
            assert again is not None and existed is True
            api.close(again)
            launched, job, marker, log = _launch_stand_in(tmp_path, run_id)
            assert job is None, "the launcher used a job it did not make"
            info = _await_marker(marker, log)
            assert info["held"] is False, f"the worker held a job it was not in: {info}"
            assert process_control.open_run_job(run_id, info["pid"]) is None, (
                "an aborter trusted a squatted job")
        finally:
            api.close(squat)
            _reap(job, _identities(launched, info))
