"""Round-8 review of the Windows port (2026-10-24): the gaps ``taskkill /T`` left.

Three fixes, each runnable off Windows through a stand-in for the Win32 calls:

1. **A Job Object per spawned child** (``process_control.popen_tree``). The child
   is created suspended, put in a job of its own, then resumed; the timeout
   kills the job (``TerminateJobObject``), never a parent-pid walk that can reach
   an unrelated orphan, and never a ``taskkill`` that can time out with only the
   direct child dead. The real calls run on the Windows leg (the ``win32`` rows).
2. **The off-Linux sweep's ``ps`` cost** (``process_control._wait_targets``):
   waits that poll at the same time share one ``ps`` table, and each poll sleeps
   only what its look left of the interval.
3. **The Windows prompt remover** (``prompt_files.remove_held``): the phase dir
   and ``prompts/`` are held open while each file is deleted through its own
   handle, and a folder OneDrive or ProjFS tagged is walked, not taken for a link.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import implement_orchestrator  # noqa: E402
from utils import process_control, prompt_files  # noqa: E402

WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="the real Win32 calls")
POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups stand in")

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_SUSPENDED = 0x00000004


# ===========================================================================
# 1. popen_tree / TreeJob, through a recording stand-in for kernel32 + ntdll
# ===========================================================================

class FakeJobApi:
    """The calls ``popen_tree`` and ``TreeJob`` make, recorded in order."""

    def __init__(self, *, create=101, open_process=202, assign=True, resume=True,
                 terminate=True):
        self.calls = []
        self._create, self._open = create, open_process
        self._assign, self._resume, self._terminate = assign, resume, terminate

    def create(self):
        self.calls.append(("create",))
        return self._create

    def open_process(self, pid):
        self.calls.append(("open_process", pid))
        return self._open

    def assign(self, job, process):
        self.calls.append(("assign", job, process))
        return self._assign

    def resume(self, process):
        self.calls.append(("resume", process))
        return self._resume

    def terminate(self, job, code):
        self.calls.append(("terminate", job, code))
        return self._terminate

    def close(self, handle):
        self.calls.append(("close", handle))

    def names(self):
        return [call[0] for call in self.calls]


class FakePopen:
    """Records how it was called; never starts anything."""

    spawned = []

    def __init__(self, args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.pid = 4242
        self.killed = False
        self.waited = False
        self.stdin = self.stdout = self.stderr = None
        FakePopen.spawned.append(self)

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        return 1


@pytest.fixture
def windows(monkeypatch):
    """``popen_tree`` as it runs on Windows, with the stand-ins installed."""
    FakePopen.spawned = []
    api = FakeJobApi()
    monkeypatch.setattr(process_control, "IS_WINDOWS", True)
    monkeypatch.setattr(process_control.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(process_control, "_REAL_POPEN", FakePopen)
    monkeypatch.setattr(process_control, "_job_api", lambda: api)
    return api


class TestPopenTree:
    def test_the_child_is_created_suspended_put_in_its_job_then_resumed(self, windows):
        proc, job = process_control.popen_tree(
            ["runner"], stdout=subprocess.PIPE, creationflags=CREATE_NEW_PROCESS_GROUP)
        assert isinstance(job, process_control.TreeJob)
        # The caller's flags are kept, and the child starts suspended.
        assert proc.kwargs["creationflags"] == CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
        assert proc.kwargs["stdout"] == subprocess.PIPE
        # The job exists before the child, and the child joins it before it runs.
        assert windows.calls == [
            ("create",), ("open_process", 4242), ("assign", 101, 202),
            ("resume", 202), ("close", 202),
        ]
        # The job handle is the caller's to close; closing never kills.
        job.close()
        job.close()
        assert windows.calls[-1] == ("close", 101)
        assert "terminate" not in windows.names()
        assert not proc.killed

    def test_popens_own_handle_is_used_and_never_closed_here(self, windows, monkeypatch):
        """CreateProcess gave it every right; an OpenProcess that failed used to
        fail the spawn, where a failed assignment only degrades it."""
        class WithHandle(FakePopen):
            def __init__(self, args, **kwargs):
                super().__init__(args, **kwargs)
                self._handle = 999

        monkeypatch.setattr(process_control.subprocess, "Popen", WithHandle)
        monkeypatch.setattr(process_control, "_REAL_POPEN", WithHandle)
        _, job = process_control.popen_tree(["runner"])
        assert job is not None
        assert windows.calls == [("create",), ("assign", 101, 999), ("resume", 999)]

    def test_terminate_ends_the_job_and_reports_failure_for_a_fallback(self, windows):
        _, job = process_control.popen_tree(["runner"])
        assert job.terminate() is True
        assert windows.calls[-1] == ("terminate", 101, 1)
        windows._terminate = False
        assert job.terminate() is False
        job.close()
        assert job.terminate() is False, "a closed job kills nothing"

    def test_a_child_that_cannot_join_runs_without_a_job(self, windows):
        windows._assign = False
        proc, job = process_control.popen_tree(["runner"])
        assert job is None
        # Resumed anyway: it runs, and taskkill stays its kill.
        assert ("resume", 202) in windows.calls
        assert ("close", 101) in windows.calls
        assert not proc.killed

    @pytest.mark.parametrize("failure", ["resume", "open_process"])
    def test_a_child_that_cannot_be_resumed_is_killed_never_left_suspended(
        self, windows, failure
    ):
        if failure == "resume":
            windows._resume = False
        else:
            windows._open = None
        with pytest.raises(OSError, match="could not be resumed"):
            process_control.popen_tree(["runner"])
        [proc] = FakePopen.spawned
        assert proc.killed and proc.waited
        assert ("close", 101) in windows.calls

    def test_no_job_means_the_spawn_it_always_was(self, windows):
        windows._create = None
        proc, job = process_control.popen_tree(
            ["runner"], creationflags=CREATE_NEW_PROCESS_GROUP)
        assert job is None
        assert proc.kwargs["creationflags"] == CREATE_NEW_PROCESS_GROUP

    def test_a_spawn_that_fails_closes_the_job(self, windows, monkeypatch):
        def refuse(args, **kwargs):
            raise FileNotFoundError("no such runner")

        monkeypatch.setattr(process_control.subprocess, "Popen", refuse)
        monkeypatch.setattr(process_control, "_REAL_POPEN", refuse)
        with pytest.raises(FileNotFoundError):
            process_control.popen_tree(["runner"])
        assert windows.calls == [("create",), ("close", 101)]

    def test_a_replaced_popen_is_called_as_it_is(self, windows, monkeypatch):
        """A test double creates no process, so there is nothing to put in a job."""
        class Double(FakePopen):
            pass

        monkeypatch.setattr(process_control.subprocess, "Popen", Double)
        proc, job = process_control.popen_tree(["runner"], creationflags=7)
        assert job is None and isinstance(proc, Double)
        assert proc.kwargs["creationflags"] == 7
        assert windows.calls == []

    def test_off_windows_it_is_popen(self, monkeypatch):
        monkeypatch.setattr(process_control, "IS_WINDOWS", False)
        monkeypatch.setattr(process_control, "_job_api",
                            lambda: pytest.fail("no job API off Windows"))
        proc, job = process_control.popen_tree(
            [sys.executable, "-c", "print('hi')"], stdout=subprocess.PIPE, text=True)
        assert job is None
        assert proc.communicate(timeout=30)[0].strip() == "hi"


class TestEndTree:
    def test_the_job_is_the_kill_and_taskkill_only_its_fallback(self, monkeypatch):
        taskkilled = []
        monkeypatch.setattr(process_control, "taskkill_tree",
                            lambda pid, timeout=30.0: taskkilled.append((pid, timeout)))
        api = FakeJobApi()
        process_control.end_tree(7, process_control.TreeJob(api, 101), timeout=1.0)
        assert taskkilled == [] and api.names() == ["terminate"]
        api._terminate = False
        process_control.end_tree(7, process_control.TreeJob(api, 101), timeout=1.0)
        assert taskkilled == [(7, 1.0)]
        process_control.end_tree(8, None, timeout=2.0)
        assert taskkilled[-1] == (8, 2.0)

    @pytest.fixture(autouse=True)
    def no_real_break(self, monkeypatch):
        """Pid 4242 is nobody's here; on Windows a real break could reach a group."""
        self.breaks = []
        monkeypatch.setattr(process_control, "_send_ctrl_break", self.breaks.append)

    def test_kill_tree_escalates_to_the_job_on_windows(self, monkeypatch):
        """``_windows_kill_tree``: the break, the grace, then the job, not taskkill."""
        monkeypatch.setattr(process_control, "IS_WINDOWS", True)
        api = FakeJobApi()
        job = process_control.TreeJob(api, 101)
        alive = {"leader": True}

        def terminate(handle, code):
            api.calls.append(("terminate", handle, code))
            alive["leader"] = False
            return True

        api.terminate = terminate
        monkeypatch.setattr(process_control, "process_alive", lambda pid: alive["leader"])
        monkeypatch.setattr(process_control, "probe", lambda pid, start: process_control.UNKNOWN)
        monkeypatch.setattr(process_control, "taskkill_tree",
                            lambda *a, **k: pytest.fail("the job was there to kill"))
        verdict = process_control.kill_tree(
            4242, start_time=None, escalate_after_s=0.2, job=job)
        assert verdict == "killed"
        assert self.breaks == [4242], "the graceful break comes first"
        assert api.names() == ["terminate"]

    def test_a_leader_that_took_the_break_still_has_its_tree_ended(self, monkeypatch):
        """Whatever it left running is in its job, and only its tree is."""
        monkeypatch.setattr(process_control, "IS_WINDOWS", True)
        api = FakeJobApi()
        monkeypatch.setattr(process_control, "process_alive", lambda pid: False)
        monkeypatch.setattr(process_control, "probe", lambda pid, start: process_control.UNKNOWN)
        verdict = process_control.kill_tree(
            4242, start_time=None, escalate_after_s=5.0,
            job=process_control.TreeJob(api, 101))
        assert verdict == "killed"
        assert api.names() == ["terminate"]


class _GroupJobApi(FakeJobApi):
    """A job off Windows: the child's process group stands in for it, a
    SIGSTOP right after the spawn for ``CREATE_SUSPENDED``."""

    def __init__(self):
        super().__init__()
        self.pid = None

    def open_process(self, pid):
        self.pid = pid
        return super().open_process(pid)

    def resume(self, process):
        os.killpg(self.pid, signal.SIGCONT)
        return super().resume(process)

    def terminate(self, job, code):
        os.killpg(self.pid, signal.SIGKILL)
        return super().terminate(job, code)


@POSIX_ONLY
def test_the_windows_probe_kills_its_tree_through_the_job(tmp_path, monkeypatch):
    """The sandbox probe end to end: a grandchild holding the probe's pipes is
    gone within the budget, by the job, with no taskkill run at all."""
    from utils import subagent_executor as executor

    real = subprocess.Popen

    def suspended(args, **kwargs):
        assert kwargs.pop("creationflags") & CREATE_SUSPENDED
        proc = real(args, start_new_session=True, **kwargs)
        os.killpg(proc.pid, signal.SIGSTOP)
        return proc

    api = _GroupJobApi()
    monkeypatch.setattr(process_control, "IS_WINDOWS", True)
    monkeypatch.setattr(process_control.subprocess, "Popen", suspended)
    monkeypatch.setattr(process_control, "_REAL_POPEN", suspended)
    monkeypatch.setattr(process_control, "_job_api", lambda: api)
    monkeypatch.setattr(process_control, "taskkill_tree",
                        lambda *a, **k: pytest.fail("the job was there to kill"))
    marker = tmp_path / "grandchild.pid"
    started = time.monotonic()
    result = executor._run_sandbox_probe_tree(
        ["sh", "-c", f"sleep 30 & echo $! > {marker}; wait"],
        cwd=str(tmp_path), timeout=0.5)
    assert result is None
    assert time.monotonic() - started < 4.0
    assert api.names()[-2:] == ["terminate", "close"]
    grandchild = int(marker.read_text().strip())
    deadline = time.time() + 5
    while time.time() < deadline and process_control.process_alive(grandchild):
        time.sleep(0.05)
    assert not process_control.process_alive(grandchild), "the grandchild survived"

    # A probe that finishes is read in full, and its job let go of, never killed.
    api.calls.clear()
    done = executor._run_sandbox_probe_tree(
        ["sh", "-c", "echo out; echo err >&2; exit 3"], cwd=str(tmp_path), timeout=30)
    assert (done.returncode, done.stdout, done.stderr) == (3, "out\n", "err\n")
    assert "terminate" not in api.names() and api.names()[-1] == "close"


#: A child that starts a grandchild deaf to CTRL_BREAK, prints its pid and
#: waits: only the job can end that grandchild.
_TREE_SCRIPT = (
    "import subprocess, sys, time; "
    "p = subprocess.Popen([sys.executable, '-c', "
    "'import signal, time; signal.signal(signal.SIGBREAK, signal.SIG_IGN); time.sleep(60)']); "
    "print(p.pid, flush=True); time.sleep(60)"
)


@WINDOWS_ONLY
@pytest.mark.parametrize("shell", [False, True], ids=["runner", "verify-shell"])
def test_a_real_timeout_kill_ends_the_whole_tree_through_its_job(shell):  # pragma: no cover
    """The real calls, through ``kill_tree`` as a timeout reaches it: the
    break, then the job, which ends a grandchild the break cannot reach. The
    ``shell=True`` row is the verify command's spawn (``cmd.exe /c``)."""
    argv = [sys.executable, "-c", _TREE_SCRIPT]
    command = subprocess.list2cmdline(argv) if shell else argv
    proc, job = process_control.popen_tree(
        command, shell=shell, stdout=subprocess.PIPE, text=True,
        **process_control.new_process_group_kwargs())
    try:
        assert job is not None, "no Job Object on Windows"
        grandchild = int(proc.stdout.readline())
        assert process_control.process_alive(grandchild)
        verdict = process_control.kill_tree(
            proc.pid, start_time=None, escalate_after_s=0.5, poll_interval_s=0.05, job=job)
        assert verdict == "killed"
        proc.wait(timeout=10)
        deadline = time.time() + 10
        while time.time() < deadline and process_control.process_alive(grandchild):
            time.sleep(0.1)
        assert not process_control.process_alive(grandchild), "the grandchild survived"
        # The pipe's last writer is gone with it, so the read ends.
        assert proc.stdout.read() == ""
    finally:
        if job is not None:
            job.close()
        if proc.poll() is None:
            proc.kill()


# ===========================================================================
# 2. The off-Linux sweep: one shared ps table, the interval kept
# ===========================================================================

class TestWaitingLooks:
    @pytest.fixture
    def ps_route(self, monkeypatch):
        """The macOS route: no /proc, and a counting ``ps -A``."""
        reads = []

        def ps_table():
            reads.append(time.monotonic())
            return {1: process_control._ProcRow(1, 0, 1, None, "t", "S")}

        # On the Windows leg too: there the table is `{}` before any reader runs.
        monkeypatch.setattr(process_control, "IS_WINDOWS", False)
        monkeypatch.setattr(process_control, "_linux_process_table", lambda: None)
        monkeypatch.setattr(process_control, "_ps_process_table", ps_table)
        monkeypatch.setattr(process_control, "_SHARED_PS_TABLE", [])
        return reads

    def test_waits_polling_together_share_one_ps(self, ps_route):
        began = time.monotonic()
        looks = [process_control._Liveness(not_before=began, max_age_s=5.0)
                 for _ in range(3)]
        assert len(ps_route) == 1
        assert all(look.table == looks[0].table for look in looks)

    def test_a_table_older_than_the_wait_or_the_interval_is_read_again(self, ps_route):
        process_control._Liveness(not_before=time.monotonic(), max_age_s=5.0)
        # A wait that began after that read never sees a table from before it.
        process_control._Liveness(not_before=time.monotonic(), max_age_s=5.0)
        assert len(ps_route) == 2
        # Nor one older than its interval.
        began = time.monotonic()
        time.sleep(0.05)
        process_control._Liveness(not_before=began, max_age_s=0.01)
        assert len(ps_route) == 3

    def test_a_look_that_decides_a_signal_always_reads_afresh(self, ps_route):
        began = time.monotonic()
        process_control._Liveness(not_before=began, max_age_s=5.0)
        process_control._Liveness()
        process_control._Liveness()
        assert len(ps_route) == 3

    def test_concurrent_waits_wait_for_one_read_instead_of_spawning_their_own(
        self, ps_route, monkeypatch
    ):
        def slow_table():
            ps_route.append(time.monotonic())
            time.sleep(0.2)
            return {1: process_control._ProcRow(1, 0, 1, None, "t", "S")}

        monkeypatch.setattr(process_control, "_ps_process_table", slow_table)
        began = time.monotonic()
        threads = [threading.Thread(
            target=process_control._Liveness,
            kwargs={"not_before": began, "max_age_s": 5.0}) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert len(ps_route) == 1

    def test_a_ps_slower_than_the_interval_is_still_shared(self, ps_route, monkeypatch):
        """Its age counts from the read's end: dated by its start, it would be
        stale on arrival, and each wait queued behind it would spawn its own."""
        def slow_table():
            ps_route.append(time.monotonic())
            time.sleep(0.15)
            return {1: process_control._ProcRow(1, 0, 1, None, "t", "S")}

        monkeypatch.setattr(process_control, "_ps_process_table", slow_table)
        began = time.monotonic()
        threads = [threading.Thread(
            target=process_control._Liveness,
            kwargs={"not_before": began, "max_age_s": 0.1}) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert len(ps_route) == 1

    def test_linux_keeps_its_own_proc_pass(self, monkeypatch):
        if process_control._linux_process_table() is None:
            pytest.skip("no /proc here")
        monkeypatch.setattr(process_control, "IS_WINDOWS", False)
        monkeypatch.setattr(process_control, "_ps_process_table",
                            lambda: pytest.fail("Linux reads /proc"))
        look = process_control._Liveness(not_before=time.monotonic(), max_age_s=5.0)
        assert look.table and os.getpid() in look.table


class TestWaitCadence:
    """A poll sleeps what its look left of the interval, not a whole interval."""

    def _run(self, monkeypatch, look_cost, interval=0.1, polls=4):
        clock = {"now": 0.0}
        slept = []
        fake_time = SimpleNamespace(
            monotonic=lambda: clock["now"],
            sleep=lambda s: (slept.append(round(s, 6)), clock.__setitem__("now", clock["now"] + s)),
        )

        class Look:
            def __init__(self, not_before=None, max_age_s=0.0):
                clock["now"] += look_cost

        class Target:
            def __init__(self):
                self.looks = 0

            def running(self, look):
                self.looks += 1
                return self.looks < polls

        monkeypatch.setattr(process_control, "time", fake_time)
        monkeypatch.setattr(process_control, "_Liveness", Look)
        still = process_control._wait_targets([Target()], until=100.0, poll_interval_s=interval)
        assert still == []
        return slept

    def test_the_scan_time_comes_out_of_the_sleep(self, monkeypatch):
        assert self._run(monkeypatch, look_cost=0.03) == [0.07, 0.07, 0.07]

    def test_a_look_dearer_than_the_interval_still_sleeps_the_floor(self, monkeypatch):
        floor = round(0.1 * process_control._POLL_SLEEP_FLOOR_SHARE, 6)
        assert self._run(monkeypatch, look_cost=0.25) == [floor, floor, floor]

    def test_a_free_look_sleeps_the_whole_interval(self, monkeypatch):
        assert self._run(monkeypatch, look_cost=0.0) == [0.1, 0.1, 0.1]


# ===========================================================================
# 3. The Windows prompt remover: folders held, files deleted by handle
# ===========================================================================

IO_REPARSE_TAG_SYMLINK = 0xA000000C
IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003  # a junction
IO_REPARSE_TAG_CLOUD_6 = 0x9000601A       # OneDrive Files On-Demand
DIRECTORY, REPARSE = 0x10, 0x400


class FakeFiles:
    """``CreateFileW`` and friends over a POSIX tree.

    A symlink is a name-surrogate reparse point; *tags* gives any other path a
    reparse tag of its own (a junction, a cloud folder); *final* rewrites the
    final path a handle reports (a folder swapped behind the walk). A delete
    lands when its handle closes, as ``FILE_DISPOSITION_INFO`` does.
    """

    def __init__(self, tags=None, final=None, unopenable=(), no_final_path=()):
        self.tags = {os.path.normpath(k): v for k, v in (tags or {}).items()}
        self.final = {os.path.normpath(k): v for k, v in (final or {}).items()}
        self.unopenable = {os.path.normpath(p) for p in unopenable}
        self.no_final_path = {os.path.normpath(p) for p in no_final_path}
        self.open_handles = {}
        self.events = []
        self.doomed = set()
        self._next = 1

    def open(self, path, *, folder):
        if not os.path.lexists(path) or os.path.normpath(path) in self.unopenable:
            return None
        handle, self._next = self._next, self._next + 1
        self.open_handles[handle] = os.path.normpath(path)
        self.events.append(("open", os.path.basename(path), folder))
        return handle

    def attributes(self, handle):
        path = self.open_handles[handle]
        if path in self.tags:
            attributes = REPARSE | (DIRECTORY if os.path.isdir(path) else 0)
            return attributes, self.tags[path]
        if os.path.islink(path):
            return REPARSE, IO_REPARSE_TAG_SYMLINK
        return (DIRECTORY if os.path.isdir(path) else 0), 0

    def final_path(self, handle):
        path = self.open_handles[handle]
        if path in self.no_final_path:
            # ERROR_PATH_NOT_FOUND: a volume mounted with no drive letter.
            raise FileNotFoundError(3, "no DOS name for this volume", path)
        return self.final.get(path, path)

    def delete(self, handle):
        self.doomed.add(handle)
        return True

    def close(self, handle):
        path = self.open_handles.pop(handle)
        self.events.append(("close", os.path.basename(path)))
        if handle in self.doomed:
            os.unlink(path)
            self.events.append(("deleted", os.path.basename(path)))


def _wanted(name):
    return name.endswith(".prompt.txt")


@pytest.fixture
def phase(tmp_path):
    folder = tmp_path / "implement" / prompt_files.FOLDER
    folder.mkdir(parents=True)
    for name in ("T001.prompt.txt", "T001.attempt-2.prompt.txt", "T002.tree.json",
                 "notes.txt"):
        (folder / name).write_text(name, encoding="utf-8")
    return folder.parent


class TestRemoveHeld:
    def test_removes_the_prompt_files_and_nothing_else(self, phase):
        files = FakeFiles()
        assert prompt_files.remove_held(str(phase), _wanted, api=files) == 2
        assert sorted(p.name for p in (phase / "prompts").iterdir()) == [
            "T002.tree.json", "notes.txt"]
        assert files.open_handles == {}, "a handle was left open"

    def test_both_folders_are_held_until_every_file_is_gone(self, phase):
        files = FakeFiles()
        prompt_files.remove_held(str(phase), _wanted, api=files)
        events = files.events
        assert events[:2] == [("open", "implement", True), ("open", "prompts", True)]
        last_delete = max(i for i, e in enumerate(events) if e[0] == "deleted")
        assert events.index(("close", "prompts")) > last_delete
        assert events.index(("close", "implement")) > events.index(("close", "prompts"))

    @pytest.mark.parametrize("where", ["implement", "implement/prompts"])
    @pytest.mark.parametrize("tag", [IO_REPARSE_TAG_MOUNT_POINT, IO_REPARSE_TAG_SYMLINK])
    def test_a_folder_that_names_another_path_removes_nothing(self, phase, where, tag):
        target = phase.parent / where
        files = FakeFiles(tags={str(target): tag})
        assert prompt_files.remove_held(str(phase), _wanted, api=files) == 0
        assert (phase / "prompts" / "T001.prompt.txt").exists()
        assert files.open_handles == {}

    @pytest.mark.parametrize("where", ["implement", "implement/prompts"])
    def test_a_cloud_folder_is_the_folder_it_looks_like(self, phase, where):
        """OneDrive Files On-Demand tags the folder; ``lstat`` called it a link."""
        files = FakeFiles(tags={str(phase.parent / where): IO_REPARSE_TAG_CLOUD_6})
        assert prompt_files.remove_held(str(phase), _wanted, api=files) == 2

    def test_a_prompts_folder_that_is_not_the_phase_dirs_child_removes_nothing(
        self, phase, tmp_path
    ):
        elsewhere = tmp_path / "elsewhere" / "prompts"
        files = FakeFiles(final={str(phase / "prompts"): str(elsewhere)})
        assert prompt_files.remove_held(str(phase), _wanted, api=files) == 0
        assert (phase / "prompts" / "T001.prompt.txt").exists()

    def test_links_folders_and_strays_are_left(self, phase, tmp_path):
        prompts = phase / "prompts"
        victim = tmp_path / "victim.prompt.txt"
        victim.write_text("not the plugin's", encoding="utf-8")
        (prompts / "T006.prompt.txt").mkdir()
        os.symlink(victim, prompts / "T008.prompt.txt")
        (prompts / "T009.prompt.txt").write_text("x", encoding="utf-8")
        files = FakeFiles(final={str(prompts / "T009.prompt.txt"): str(victim)})
        assert prompt_files.remove_held(str(phase), _wanted, api=files) == 2
        assert victim.read_text(encoding="utf-8") == "not the plugin's"
        assert (prompts / "T006.prompt.txt").is_dir()
        assert (prompts / "T008.prompt.txt").is_symlink()
        assert (prompts / "T009.prompt.txt").exists(), "its handle named another file"

    def test_a_cloud_file_is_a_file(self, phase):
        target = phase / "prompts" / "T001.prompt.txt"
        files = FakeFiles(tags={str(target): IO_REPARSE_TAG_CLOUD_6})
        assert prompt_files.remove_held(str(phase), _wanted, api=files) == 2
        assert not target.exists()

    def test_a_missing_phase_dir_is_no_walk_and_the_path_walk_finds_nothing(
        self, tmp_path, monkeypatch
    ):
        assert prompt_files.remove_held(
            str(tmp_path / "absent"), _wanted, api=FakeFiles()) is None
        monkeypatch.setattr(prompt_files, "_windows_files", lambda: FakeFiles())
        assert implement_orchestrator._remove_kept_prompts_windows(tmp_path / "absent") == 0

    @pytest.mark.parametrize("failure", ["unopenable", "no_final_path"])
    @pytest.mark.parametrize("where", ["implement", "implement/prompts"])
    def test_a_walk_that_cannot_be_made_falls_back_to_the_path_walk(
        self, phase, monkeypatch, failure, where
    ):
        """Review of this round: an open or a final path that fails is no
        refusal. Read as one, a start-over kept every prompt again."""
        files = FakeFiles(**{failure: [str(phase.parent / where)]})
        assert prompt_files.remove_held(str(phase), _wanted, api=files) is None
        assert files.open_handles == {}
        monkeypatch.setattr(prompt_files, "_windows_files", lambda: files)
        assert implement_orchestrator._remove_kept_prompts_windows(phase) == 2

    def test_without_the_windows_calls_the_orchestrator_walks_by_path(
        self, phase, monkeypatch
    ):
        monkeypatch.setattr(prompt_files, "_windows_files", lambda: None)
        assert prompt_files.remove_held(str(phase), _wanted) is None
        assert implement_orchestrator._remove_kept_prompts_windows(phase) == 2
        assert sorted(p.name for p in (phase / "prompts").iterdir()) == [
            "T002.tree.json", "notes.txt"]

    @WINDOWS_ONLY
    def test_the_real_calls_remove_and_refuse_junctions(self, phase, tmp_path):  # pragma: no cover
        """``_WindowsFiles`` itself: CreateFileW, the tag, the final path, the
        delete by handle. A junction needs no privilege (``_winapi``)."""
        import _winapi

        prompts = phase / "prompts"
        (prompts / "T006.prompt.txt").mkdir()
        assert prompt_files.remove_held(str(phase), _wanted) == 2
        assert sorted(p.name for p in prompts.iterdir()) == [
            "T002.tree.json", "T006.prompt.txt", "notes.txt"]

        # A phase dir reached through a junction removes nothing.
        (prompts / "T010.prompt.txt").write_text("x", encoding="utf-8")
        linked = tmp_path / "linked"
        _winapi.CreateJunction(str(phase), str(linked))
        assert prompt_files.remove_held(str(linked), _wanted) == 0
        assert (prompts / "T010.prompt.txt").exists()
        assert prompt_files.remove_held(str(phase), _wanted) == 1

        # Nor does a `prompts/` swapped for a junction to somewhere else.
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "T009.prompt.txt"
        victim.write_text("not the plugin's", encoding="utf-8")
        prompts.rename(phase / "prompts.real")
        _winapi.CreateJunction(str(outside), str(prompts))
        assert prompt_files.remove_held(str(phase), _wanted) == 0
        assert victim.read_text(encoding="utf-8") == "not the plugin's"

    def test_the_orchestrator_takes_the_held_walk_when_it_can(self, phase, monkeypatch):
        files = FakeFiles()
        monkeypatch.setattr(prompt_files, "_windows_files", lambda: files)
        monkeypatch.setattr(implement_orchestrator, "_remove_kept_prompts_by_path",
                            lambda phase_dir: pytest.fail("the held walk was there"))
        assert implement_orchestrator._remove_kept_prompts_windows(phase) == 2
        assert files.events, "the held walk never ran"
