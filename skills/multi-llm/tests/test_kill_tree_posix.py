"""``process_control.kill_tree`` on POSIX for a child this process holds.

The timeout paths hand ``kill_tree`` their ``Popen`` as ``child``: a child this
process has not reaped yet. Once it dies of its SIGTERM it is a zombie until that
``Popen`` reaps it, which happens only after ``kill_tree`` returns. A zombie still
answers ``kill(pid, 0)`` and ``killpg(pgid, 0)``, so a liveness check that
trusted those sat out the whole grace and the SIGKILL window on a tree already
dead, then reported ``survived``. For an owned child the zombie counts as
exited. Until it is reaped, its pid and the number of the group it leads cannot
be recycled, under the lifecycle contract docs/development/runtime.md states
("An owned child's kill"). So the group always gets one SIGKILL by number before
the kill reports it gone: one look at the process table can miss a member
forked while it was read, and after the SIGKILL nothing in the group can fork.
That proof was checked for Linux, under glibc or musl on the machines whose
``struct sigaction`` layout was read, and the owned kill runs only there:
everywhere else, and for targets known only by number (no ``child``), the kill
is the one it was before.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import process_control  # noqa: E402

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")

#: Real processes get the owned kill only where it runs at all.
owned_kill_here = pytest.mark.skipif(
    process_control._sigaction_abi() is None,
    reason="the owned kill runs only on Linux, under glibc or musl, on a checked machine",
)

#: Ignores SIGTERM, says so, then hangs: only a SIGKILL ends it.
_DEAF = (
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)

#: Dies of SIGTERM (Python's default) after starting a deaf member in its own
#: process group, whose pid it writes to argv[1] once that member is ready.
_LEADER_WITH_A_DEAF_MEMBER = (
    "import os, subprocess, sys, time\n"
    f"member = subprocess.Popen([sys.executable, '-c', {_DEAF!r}],\n"
    "                          stdout=subprocess.PIPE, text=True)\n"
    "member.stdout.readline()\n"
    "with open(sys.argv[1] + '.tmp', 'w') as handle:\n"
    "    handle.write(str(member.pid))\n"
    "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
    "time.sleep(60)\n"
)

#: Its SIGTERM handler forks a child that ignores SIGTERM, writes that child's
#: pid to argv[1] and exits 23: a group member the kill could not have seen
#: before its SIGTERM.
_FORKS_A_DEAF_CHILD_ON_SIGTERM = (
    "import os, signal, sys, time\n"
    "def term(*_args):\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "    pid = os.fork()\n"
    "    if pid:\n"
    "        with open(sys.argv[1] + '.tmp', 'w') as handle:\n"
    "            handle.write(str(pid))\n"
    "        os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
    "        os._exit(23)\n"
    "    time.sleep(60)\n"
    "    os._exit(0)\n"
    "signal.signal(signal.SIGTERM, term)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)


#: Ignores SIGTERM, writes its pid to <argv[1]>/member, and once <argv[1]>/go
#: exists forks a child (deaf too, it inherits the SIG_IGN) and exits at once.
#: The child writes its pid to <argv[1]>/late and hangs.
_FORKS_AND_LEAVES_ON_CUE = (
    "import os, pathlib, signal, sys, time\n"
    "run = pathlib.Path(sys.argv[1])\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "(run / 'member.tmp').write_text(str(os.getpid()))\n"
    "os.replace(run / 'member.tmp', run / 'member')\n"
    "while not (run / 'go').exists():\n"
    "    time.sleep(0.001)\n"
    "if os.fork():\n"
    "    os._exit(0)\n"
    "(run / 'late.tmp').write_text(str(os.getpid()))\n"
    "os.replace(run / 'late.tmp', run / 'late')\n"
    "time.sleep(60)\n"
)

#: Dies of SIGTERM, after starting :data:`_FORKS_AND_LEAVES_ON_CUE` in its group.
_LEADER_OF_A_MEMBER_THAT_FORKS_ON_CUE = (
    "import pathlib, subprocess, sys, time\n"
    "run = pathlib.Path(sys.argv[1])\n"
    f"subprocess.Popen([sys.executable, '-c', {_FORKS_AND_LEAVES_ON_CUE!r}, str(run)])\n"
    "while not (run / 'member').exists():\n"
    "    time.sleep(0.001)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)


def _running(pid: int) -> bool:
    """Alive and not a zombie, read the way the kill reads it (``/proc`` or ``ps``)."""
    return process_control._Liveness().pid_running(pid)


def _await_file(path: Path, timeout: float = 15.0) -> int:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.02)
    return int(path.read_text().strip())


def _kill_if_running(*pids) -> None:
    """Clean-up after a failed assertion; a pid already reaped is left alone."""
    for pid in pids:
        if pid and _running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def _reap(proc) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)
    if proc.stdout is not None:
        proc.stdout.close()


@pytest.fixture(params=["proc", "ps"])
def process_table(request, monkeypatch):
    """Run each real-process test with ``/proc`` readable, and as Linux runs
    without it (``hidepid``, a container that does not mount it).

    ``ps`` hides ``/proc`` from the kill, so every look, state and start time
    comes from ``ps``; the child's own exit still comes from ``waitid``, as on
    any Linux. A stand-in for nothing else: the owned kill does not run off
    Linux, and these tests never ran on macOS or a BSD kernel.
    """
    if request.param == "ps":
        if shutil.which("ps") is None:
            pytest.skip("no ps on this host")
        monkeypatch.setattr(process_control, "_linux_process_table", lambda: None)
        monkeypatch.setattr(process_control, "_linux_row", lambda _pid: (False, None))
        # The identity kill needs /proc: without it the kill is by number.
        monkeypatch.setattr(process_control, "_pidfd_api", lambda: None)
        monkeypatch.setattr(process_control, "_linux_state", lambda _pid: None)
        monkeypatch.setattr(process_control, "_linux_start_time", lambda _pid: None)
    elif process_control._linux_process_table() is None:
        pytest.skip("no /proc on this host")
    return request.param


def _recording_os(record, *, send):
    """``os`` for the kill, recording every signal it sends as ``(call,
    target, signal name)``. With *send* false only the existence probes
    (signal 0) go out."""
    real_os = os

    class _RecordingOs:
        def __getattr__(self, name):
            return getattr(real_os, name)

        @staticmethod
        def kill(pid, sig):
            if sig:
                record.append(("kill", pid, signal.Signals(sig).name))
            if sig and not send:
                return None
            return real_os.kill(pid, sig)

        @staticmethod
        def killpg(pgid, sig):
            if sig:
                record.append(("killpg", pgid, signal.Signals(sig).name))
            if sig and not send:
                return None
            return real_os.killpg(pgid, sig)

    return _RecordingOs()


def _recording_pidfds(monkeypatch, record, *, send):
    """The identity kill's signals too, as ``("pidfd", pid, signal name)``."""
    real_send = process_control._IdentityUnit._send

    def recording(unit, api, target, sig):
        record.append(("pidfd", target.pid, signal.Signals(sig).name))
        if send:
            real_send(unit, api, target, sig)

    monkeypatch.setattr(process_control._IdentityUnit, "_send", recording)


@pytest.fixture
def sent(monkeypatch):
    """Every signal the kill sends; each still goes out."""
    record = []
    monkeypatch.setattr(process_control, "os", _recording_os(record, send=True))
    _recording_pidfds(monkeypatch, record, send=True)
    return record


@pytest.fixture
def held(monkeypatch):
    """Every signal the kill would send; none goes out."""
    record = []
    monkeypatch.setattr(process_control, "os", _recording_os(record, send=False))
    _recording_pidfds(monkeypatch, record, send=False)
    return record


def _kill_child(proc, **kwargs):
    began = time.monotonic()
    verdict = process_control.kill_tree(
        proc.pid, start_time=None, child=proc, poll_interval_s=0.02, **kwargs)
    return verdict, time.monotonic() - began


@owned_kill_here
@pytest.mark.usefixtures("process_table")
class TestAnOwnedZombieHasExited:
    def test_a_leader_that_dies_of_its_sigterm_ends_the_kill_at_once(self, sent):
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            verdict, took = _kill_child(proc, pgid=proc.pid, escalate_after_s=30.0)
            assert verdict == "killed"
            # The closing SIGKILL reaches only the zombie, and changes nothing.
            assert sent == [("killpg", proc.pid, "SIGTERM"), ("killpg", proc.pid, "SIGKILL")]
            # A wait on the zombie would sit out the 30 s grace.
            assert took < 30.0, took
            # Not reaped by the kill: its Popen still reads the real status.
            assert proc.wait(timeout=5) == -signal.SIGTERM
        finally:
            _reap(proc)

    def test_a_leader_in_this_processs_own_group_is_signalled_alone(self, sent):
        """No group of its own: only the pid is signalled, never our group."""
        proc = subprocess.Popen(["sleep", "60"])
        try:
            verdict, took = _kill_child(proc, escalate_after_s=30.0)
            assert verdict == "killed" and took < 30.0, (verdict, took)
            assert sent == [("kill", proc.pid, "SIGTERM"), ("kill", proc.pid, "SIGKILL")]
            assert proc.wait(timeout=5) == -signal.SIGTERM
        finally:
            _reap(proc)


@owned_kill_here
@pytest.mark.usefixtures("process_table")
class TestWhatStillRunsInTheGroupGetsTheSigkill:
    def test_a_leader_that_ignores_sigterm_is_escalated(self, sent):
        proc = subprocess.Popen([sys.executable, "-c", _DEAF], start_new_session=True,
                                stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout.readline().strip() == "ready"
            verdict, took = _kill_child(proc, pgid=proc.pid, escalate_after_s=0.3)
            assert verdict == "killed"
            assert sent == [("killpg", proc.pid, "SIGTERM"), ("killpg", proc.pid, "SIGKILL")]
            assert took >= 0.3, took
            assert proc.wait(timeout=5) == -signal.SIGKILL
        finally:
            _reap(proc)

    def test_a_member_that_outlives_its_zombie_leader_is_killed(self, tmp_path, sent):
        marker = tmp_path / "member.pid"
        proc = subprocess.Popen(
            [sys.executable, "-c", _LEADER_WITH_A_DEAF_MEMBER, str(marker)],
            start_new_session=True,
        )
        member = None
        try:
            member = _await_file(marker)
            verdict, _took = _kill_child(proc, pgid=proc.pid, escalate_after_s=0.3)
            assert verdict == "killed"
            assert not _running(member), "the member that ignored SIGTERM still runs"
            assert sent == [("killpg", proc.pid, "SIGTERM"), ("killpg", proc.pid, "SIGKILL")]
            # The leader died of the SIGTERM; the member kept the wait going.
            assert proc.wait(timeout=5) == -signal.SIGTERM
        finally:
            _kill_if_running(member)
            _reap(proc)

    def test_a_member_forked_after_the_sigterm_is_killed(self, tmp_path, sent):
        """No snapshot taken before the SIGTERM could have named it: the group
        does, by its number, which the unreaped leader keeps."""
        marker = tmp_path / "late.pid"
        proc = subprocess.Popen(
            [sys.executable, "-c", _FORKS_A_DEAF_CHILD_ON_SIGTERM, str(marker)],
            start_new_session=True, stdout=subprocess.PIPE, text=True,
        )
        late = None
        try:
            assert proc.stdout.readline().strip() == "ready"
            verdict, _took = _kill_child(proc, pgid=proc.pid, escalate_after_s=0.3)
            late = _await_file(marker)
            assert verdict == "killed"
            assert not _running(late), "the member forked on SIGTERM still runs"
            assert sent == [("killpg", proc.pid, "SIGTERM"), ("killpg", proc.pid, "SIGKILL")]
            # Its own exit status, read by its Popen after the kill.
            assert proc.wait(timeout=5) == 23
        finally:
            _kill_if_running(late)
            _reap(proc)

    def test_a_member_forked_while_a_look_reads_the_table_is_killed(
        self, tmp_path, sent, monkeypatch, process_table
    ):
        """A look is not atomic. Here the one look after the leader's death has
        listed its processes when the last running member forks and exits, so
        the look finds only zombies and never the new child. Nothing that look
        says ends the kill: the group's closing SIGKILL takes the child."""
        proc = subprocess.Popen(
            [sys.executable, "-c", _LEADER_OF_A_MEMBER_THAT_FORKS_ON_CUE, str(tmp_path)],
            start_new_session=True, stdout=subprocess.PIPE, text=True,
        )
        member = late = None
        armed = [True]
        missed = []

        def fork_unseen(member_left):
            """Once, right after the SIGTERM: the member forks and exits."""
            if not (armed[0] and any(entry[2] == "SIGTERM" for entry in sent)):
                return None
            armed[0] = False
            (tmp_path / "go").touch()
            child = _await_file(tmp_path / "late")
            deadline = time.monotonic() + 15
            while not member_left():
                assert time.monotonic() < deadline, "the member never exited"
                time.sleep(0.001)
            missed.append(child)
            return child

        if process_table == "proc":
            # The names are read before the fork, every row after it.
            recording_os = process_control.os

            class _ListsThenForks:
                def __getattr__(self, name):
                    return getattr(recording_os, name)

                @staticmethod
                def listdir(path):
                    names = recording_os.listdir(path)
                    if path == "/proc":
                        fork_unseen(
                            lambda: process_control._linux_state(member) in ("Z", None))
                    return names

            monkeypatch.setattr(process_control, "os", _ListsThenForks())
        else:
            # One ps read is one listing: the child forked behind it is absent.
            real_table = process_control._ps_process_table

            def member_left():
                row = real_table().get(member)
                return row is None or row.state == "Z"

            def ps_then_fork():
                table = real_table()
                leader = table.get(proc.pid)
                if leader is not None and leader.state == "Z":
                    child = fork_unseen(member_left)
                    if child is not None:
                        table = {pid: row for pid, row in real_table().items()
                                 if pid != child}
                return table

            monkeypatch.setattr(process_control, "_ps_process_table", ps_then_fork)
        try:
            assert proc.stdout.readline().strip() == "ready"
            member = _await_file(tmp_path / "member")
            verdict, _took = _kill_child(proc, pgid=proc.pid, escalate_after_s=0.3)
            assert missed, "no look ran between the leader's death and the end of the kill"
            late = missed[0]
            assert not _running(late), "the child forked behind the look still runs"
            assert verdict == "killed"
            assert sent == [("killpg", proc.pid, "SIGTERM"), ("killpg", proc.pid, "SIGKILL")]
            assert proc.wait(timeout=5) == -signal.SIGTERM
        finally:
            monkeypatch.undo()
            _kill_if_running(late, member)
            _reap(proc)


#: Starts ``sleep`` in a session of its own, prints its pid, reaps it once it
#: exits, then hangs: the sleep is this test's grandchild, never its child.
_PARENT_OF_A_SESSION_LEADER = (
    "import subprocess, time\n"
    "child = subprocess.Popen(['sleep', '60'], start_new_session=True)\n"
    "print(child.pid, flush=True)\n"
    "child.wait()\n"
    "time.sleep(60)\n"
)


@pytest.mark.usefixtures("process_table")
class TestOnlyAnUnreapedChildAnchors:
    """Which kill a ``child`` gets, read off what the kill does: the owned kill
    ends at once on a zombie and always closes with a SIGKILL; the kill by
    number does neither."""

    @owned_kill_here
    def test_a_child_that_is_already_a_zombie_still_anchors(self, sent):
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            os.kill(proc.pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while _running(proc.pid):
                assert time.monotonic() < deadline, "the child never exited"
                time.sleep(0.02)
            # By number, the zombie would be waited on for the grace and more.
            verdict, took = _kill_child(proc, pgid=proc.pid, escalate_after_s=3.0)
            assert verdict == "killed" and took < 2.5, (verdict, took)
            assert sent == [("killpg", proc.pid, "SIGTERM"), ("killpg", proc.pid, "SIGKILL")]
            assert proc.wait(timeout=5) == -signal.SIGKILL
        finally:
            _reap(proc)

    def test_a_pid_reaped_behind_its_popen_does_not(self, held):
        """The kernel says it is no child of ours: by number, the pid and its
        group are gone, and nothing is sent."""
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        pid = proc.pid
        _reap(proc)
        stale = types.SimpleNamespace(pid=pid, returncode=None)
        verdict = process_control.kill_tree(
            pid, start_time=None, pgid=pid, child=stale, escalate_after_s=0.3,
            poll_interval_s=0.02)
        assert verdict == "gone"
        assert held == []

    def test_a_process_that_is_not_this_ones_child_does_not(self, sent):
        """A grandchild leading its own group, with no start time recorded.
        Where the kill goes by identity nothing names it but its pid, so it is
        not signalled at all (``unresolved``); by number, a SIGTERM only, once
        its own parent has reaped it."""
        parent = subprocess.Popen([sys.executable, "-c", _PARENT_OF_A_SESSION_LEADER],
                                  start_new_session=True, stdout=subprocess.PIPE, text=True)
        grandchild = None
        try:
            grandchild = int(parent.stdout.readline())
            claimed = types.SimpleNamespace(pid=grandchild, returncode=None)
            verdict, _took = _kill_child(claimed, pgid=grandchild, escalate_after_s=3.0)
            if process_control.signals_by_identity():
                assert (verdict, sent) == ("unresolved", [])
                return
            assert verdict == "killed"
            assert sent == [("killpg", grandchild, "SIGTERM")]
        finally:
            _kill_if_running(grandchild)
            _reap(parent)


#: Sets SIGCHLD's disposition as argv[2] says, then kills a ``sleep`` it owns
#: and prints the verdict, the signals sent and the status ``wait`` returned.
_KILL_WITH_CHILDREN_REAPED_AT_EXIT = """
import ctypes, json, os, signal, subprocess, sys
sys.path.insert(0, sys.argv[1])
from utils import process_control

how = sys.argv[2]
if how == "signal module SIG_IGN":
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
else:
    class SigAction(ctypes.Structure):
        _fields_ = [("handler", ctypes.c_void_p),
                    ("mask", ctypes.c_ulong * (1024 // (8 * ctypes.sizeof(ctypes.c_ulong)))),
                    ("flags", ctypes.c_int), ("restorer", ctypes.c_void_p)]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sigaction.argtypes = [ctypes.c_int, ctypes.POINTER(SigAction), ctypes.POINTER(SigAction)]
    action = SigAction()
    if how == "native SIG_IGN":
        action.handler = int(signal.SIG_IGN)
    else:
        action.flags = 0x2  # SA_NOCLDWAIT
    assert libc.sigaction(signal.SIGCHLD, ctypes.byref(action), None) == 0
    assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, "Python saw it"

sent = []
real_os = os
class RecordingOs:
    def __getattr__(self, name):
        return getattr(real_os, name)
    def killpg(self, pgid, sig):
        if sig:
            sent.append(signal.Signals(sig).name)
        return real_os.killpg(pgid, sig)
    def kill(self, pid, sig):
        if sig:
            sent.append(signal.Signals(sig).name)
        return real_os.kill(pid, sig)
process_control.os = RecordingOs()
real_send = process_control._IdentityUnit._send
def recording_send(unit, api, target, sig):
    sent.append(signal.Signals(sig).name)
    return real_send(unit, api, target, sig)
# Where the kill by number goes by identity (Linux with pidfds), its signals.
process_control._IdentityUnit._send = recording_send
child = subprocess.Popen(["sleep", "60"], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    verdict = process_control.kill_tree(child.pid, start_time=None, pgid=child.pid,
                                        child=child, escalate_after_s=3.0,
                                        poll_interval_s=0.02)
    process_control.os = real_os
    if child.poll() is None:
        child.kill()  # not signalled by the kill: no identity, no signal
    print(json.dumps({"verdict": verdict, "sent": sent, "wait": child.wait(timeout=5),
                      "by_identity": process_control.signals_by_identity()}))
finally:
    if child.poll() is None:
        child.kill()
"""

#: Where the script above can fill in ``struct sigaction`` (glibc and musl on
#: a checked Linux machine: a handler, a 1024-bit mask, an ``int`` of flags),
#: and where the kill reads SIGCHLD's flags.
_SIGACTION_READ_HERE = process_control._sigaction_abi() is not None


class TestChildrenReapedAtExitAnchorNothing:
    """With SIGCHLD ignored or ``SA_NOCLDWAIT`` the kernel reaps a child at its
    exit, behind its ``Popen``: its pid is free at once and ``wait`` returns 0.
    The child anchors nothing, and with no start time recorded nothing else
    names it: where the kill goes by identity it is not signalled at all
    (``unresolved``; no identity, no signal), and elsewhere the kill is the one
    by number. Each case runs in a process of its own, so this one's
    disposition never changes."""

    @pytest.mark.parametrize("how", [
        "signal module SIG_IGN", "native SIG_IGN", "SA_NOCLDWAIT"])
    def test_the_kill_is_the_one_by_number(self, how):
        if how != "signal module SIG_IGN" and not _SIGACTION_READ_HERE:
            pytest.skip("set natively only where the kill reads the struct")
        result = subprocess.run(
            [sys.executable, "-c", _KILL_WITH_CHILDREN_REAPED_AT_EXIT,
             str(Path(__file__).parent.parent), how],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        outcome = json.loads(result.stdout)
        if outcome.pop("by_identity"):
            assert outcome == {"verdict": "unresolved", "sent": [], "wait": 0}, outcome
            return
        # By number: the SIGTERM ended it and the kernel reaped it at once. The
        # owned kill would have closed with a SIGKILL (refused, the anchor gone).
        assert outcome == {"verdict": "killed", "sent": ["SIGTERM"], "wait": 0}, outcome


# ---------------------------------------------------------------------------
# The same kill on a scripted host and a fake clock: nothing real is signalled.
# ---------------------------------------------------------------------------

#: :func:`process_control._sigchld_action` itself, before any test replaces it.
_REAL_SIGCHLD_ACTION = process_control._sigchld_action


class _FakeStruct:
    """Stands in for the built ``struct sigaction``: never filled in."""

    handler = 0
    flags = 0


#: A leader in a session of its own (its group's number is its pid) and a
#: provider CLI it started, in its group.
_GROUP = _LEADER = 4101
_MEMBER = 4103


class _Host:
    """A process table on a fake clock, for the kill to look at and signal.

    *procs* maps pid to ``(pgid, start time)``. A process dies of any signal but
    a SIGTERM it is *deaf* to (a SIGKILL too when it is *unkillable*). One in
    *owned* is this process's child: it dies into a zombie, kept until reaped,
    and ``waitid`` answers for it. Any other is reaped at once, by init.
    *after_term* scripts what happens once the first SIGTERM lands, as
    ``(delay, action)``, and *before_sigkill* what happens between the kill's
    last look and its SIGKILL. A group signal in *failing_group_signals* fails
    (EPERM), after running the action it maps to, if any. A process ``spawn`` starts is a bystander, one
    ``fork`` starts is a member of the tree, and every signal records the pids
    it reached.

    With *pidfd* (and ``/proc``: *fast_path*) the host has pidfds, and a kill
    of a target known by number goes by identity: each pidfd holds the process
    its pid named when it was opened (a pid taken again later is another
    *generation*), and a signal through it reaches that process or fails. Their
    numbers are never real descriptors: the host closes them itself. Without
    (the default) the kill is the one by number.
    """

    OWN_PGID = 9999

    def __init__(self, procs, *, deaf=(), fast_path=True, owned=(), unkillable=(),
                 pidfd=False):
        self.now = 0.0
        self.procs = dict(procs)
        self.deaf = set(deaf)
        self.owned = set(owned)
        self.unkillable = set(unkillable)
        self.zombies = set()
        self.pidfd = pidfd
        #: pid -> how many times it was handed out again; fd -> (pid, generation).
        self.generation = {}
        self.pidfds = {}
        self.open_pidfds = set()
        #: Sorted by delay; set once the host's actions are built.
        self.after_term = []
        self.before_sigkill = []
        self.failing_group_signals = {}
        self.fast_path = fast_path
        self.term_at = None
        self.bystanders = set()
        self.sent = []
        self.wrong = []

    # The scripted host.

    def _die(self, pid):
        if pid in self.owned:
            self.zombies.add(pid)
        else:
            self.procs.pop(pid, None)

    def exit(self, pid):
        return lambda: self._die(pid)

    def reap(self, pid):
        """Someone reaps an owned child behind its ``Popen``'s back."""
        def action():
            self.owned.discard(pid)
            self.zombies.discard(pid)
            self.procs.pop(pid, None)
        return action

    def spawn(self, pid, pgid, start):
        def action():
            assert pid not in self.procs, f"pid {pid} is still in use"
            self.procs[pid] = (pgid, start)
            self.generation[pid] = self.generation.get(pid, 0) + 1
            self.deaf.discard(pid)
            self.bystanders.add(pid)
        return action

    def fork(self, pid, pgid, *, deaf=False):
        def action():
            assert pid not in self.procs, f"pid {pid} is still in use"
            self.procs[pid] = (pgid, f"{pid}-start")
            self.generation[pid] = self.generation.get(pid, 0) + 1
            if deaf:
                self.deaf.add(pid)
        return action

    def setsid(self, pid):
        """*pid* leaves its group for a session and group of its own."""
        def action():
            self.procs[pid] = (pid, self.procs[pid][1])
        return action

    # The pidfd calls.

    def pidfd_open(self, pid):
        self._tick()
        if pid not in self.procs:
            raise ProcessLookupError(pid)
        fd = 1_000_000 + len(self.pidfds)
        self.pidfds[fd] = (pid, self.generation.get(pid, 0))
        self.open_pidfds.add(fd)
        return fd

    def pidfd_exited(self, fd):
        """A pidfd polls readable once its process has exited: a zombie, or
        reaped (its pid perhaps handed out again since)."""
        self._tick()
        pid, generation = self.pidfds[fd]
        return (pid not in self.procs or self.generation.get(pid, 0) != generation
                or pid in self.zombies)

    def pidfd_send_signal(self, fd, sig):
        self._tick()
        self._sigkill_lands(sig)
        pid, generation = self.pidfds[fd]
        if pid not in self.procs or self.generation.get(pid, 0) != generation:
            raise ProcessLookupError(pid)
        if sig:
            self._deliver([pid], sig, f"pidfd {pid}")

    def _tick(self):
        while (self.term_at is not None and self.after_term
               and self.term_at + self.after_term[0][0] <= self.now + 1e-9):
            self.after_term.pop(0)[1]()

    def _deliver(self, pids, sig, how):
        self.sent.append((round(self.now, 3), how, signal.Signals(sig).name,
                          sorted(pids)))
        self.wrong += [(how, pid) for pid in pids if pid in self.bystanders]
        for pid in pids:
            if pid in self.bystanders or pid in self.zombies:
                continue  # a bystander is recorded above; the test fails on it
            if sig == signal.SIGTERM and pid in self.deaf:
                continue
            if sig == signal.SIGKILL and pid in self.unkillable:
                continue
            self._die(pid)
        if sig == signal.SIGTERM and self.term_at is None:
            self.term_at = self.now

    def hit_bystanders(self):
        """Every (signal target, bystander pid) a signal reached."""
        return list(self.wrong)

    # The OS calls the kill makes.

    def _sigkill_lands(self, sig):
        while sig == signal.SIGKILL and self.before_sigkill:
            self.before_sigkill.pop(0)()

    def kill(self, pid, sig):
        self._tick()
        self._sigkill_lands(sig)
        if pid not in self.procs:
            raise ProcessLookupError(pid)
        if sig:
            self._deliver([pid], sig, f"pid {pid}")

    def killpg(self, pgid, sig):
        self._tick()
        self._sigkill_lands(sig)
        if sig in self.failing_group_signals:
            action = self.failing_group_signals[sig]
            if action is not None:
                action()
            raise PermissionError(pgid)
        members = [pid for pid, (group, _) in self.procs.items() if group == pgid]
        if not members:
            raise ProcessLookupError(pgid)
        if sig:
            self._deliver(members, sig, f"group {pgid}")

    def getpgid(self, pid):
        self._tick()
        if pid == 0:
            return self.OWN_PGID
        if pid not in self.procs:
            raise ProcessLookupError(pid)
        return self.procs[pid][0]

    def waitid(self, _idtype, pid, _options):
        self._tick()
        if pid not in self.owned or pid not in self.procs:
            raise ChildProcessError(pid)
        return types.SimpleNamespace(si_pid=pid) if pid in self.zombies else None

    def alive(self, pid):
        self._tick()
        return pid in self.procs

    def start_time(self, pid):
        self._tick()
        return self.procs[pid][1] if pid in self.procs else None

    def _state(self, pid):
        return "Z" if pid in self.zombies else "S"

    def state(self, pid):
        self._tick()
        return self._state(pid) if self.fast_path and pid in self.procs else None

    def table(self, *_args):
        self._tick()
        return {
            pid: process_control._ProcRow(
                pid, os.getpid() if pid in self.owned else 1, pgid, pgid, start,
                self._state(pid))
            for pid, (pgid, start) in self.procs.items()
        }

    def sleep(self, seconds):
        self.now += max(seconds, 0.0)

    def install(self, monkeypatch):
        real_os = os

        class _Os:
            kill = staticmethod(self.kill)
            killpg = staticmethod(self.killpg)
            getpgid = staticmethod(self.getpgid)

            def __getattr__(self, name):
                return getattr(real_os, name)

        clock = types.SimpleNamespace(
            time=lambda: self.now, monotonic=lambda: self.now, sleep=self.sleep)
        host = self
        pidfds = None
        if self.pidfd and hasattr(process_control, "_Pidfds"):
            class _HostPidfds(process_control._Pidfds):
                def close(self, fd):
                    host.open_pidfds.discard(fd)

                def exited(self, fd):
                    return host.pidfd_exited(fd)

            pidfds = _HostPidfds(self.pidfd_open, self.pidfd_send_signal)
        for name, value in {
            "os": _Os(), "time": clock,
            # A scripted Linux host under glibc or musl: SIGCHLD at its default.
            "_sigchld_action": lambda: (int(signal.SIG_DFL), 0),
            "process_alive": self.alive, "process_start_time": self.start_time,
            "_linux_state": self.state,
            "_linux_start_time": lambda pid: (
                self.start_time(pid) if self.fast_path else None),
            "_linux_process_table": lambda: self.table() if self.fast_path else None,
            "_linux_row": lambda pid: (
                (True, self.table().get(pid)) if self.fast_path else (False, None)),
            "_process_table": self.table, "_waiting_process_table": self.table,
            "_pidfd_api": lambda: pidfds if self.fast_path else None,
        }.items():
            # Not raising: the same scripted host runs against a kill without
            # pidfds or per-pid rereads (an older one), where the scenarios in
            # test_signal_identity_posix.py show what it signalled.
            monkeypatch.setattr(process_control, name, value,
                                raising=name not in ("_linux_row", "_pidfd_api"))
        # Linux answers an owned child's exit with waitid, /proc or not.
        monkeypatch.setattr(process_control, "_waitid", self.waitid)
        return self


@pytest.fixture(params=[True, False], ids=["proc", "ps"])
def fast_path(request):
    """A Linux host with ``/proc``, or one whose looks read only the shared
    ``ps`` table (``/proc`` unreadable). ``waitid`` answers on both."""
    return request.param


def _owned_host(fast_path, *, members=(), **kwargs):
    procs = {_LEADER: (_GROUP, "leader-start")}
    procs.update({pid: (_GROUP, f"{pid}-start") for pid in members})
    return _Host(procs, owned=[_LEADER], fast_path=fast_path, **kwargs)


def _kill_owned(host, *, child=True, grace=0.5, popen=None, pgid=_GROUP):
    if popen is None and child:
        popen = types.SimpleNamespace(pid=_LEADER, returncode=None)
    return process_control.kill_tree(
        _LEADER, start_time=None, pgid=pgid, escalate_after_s=grace,
        poll_interval_s=0.1, child=popen)


def _by_number(host, grace=0.5):
    """What the kill by number does to the owned leader's zombie: waits it
    out, SIGKILLs it, and reports it ``survived``."""
    return (host.now == pytest.approx(grace + process_control._KILL_CONFIRM_S, abs=0.11)
            and [entry[2] for entry in host.sent] == ["SIGTERM", "SIGKILL"])


class TestTheOwnedKillOnAFakeClock:
    def test_the_zombie_ends_the_kill_at_its_first_look(self, monkeypatch, fast_path):
        """The latency itself: no grace and no SIGKILL window sat out."""
        host = _owned_host(fast_path).install(monkeypatch)
        assert _kill_owned(host, grace=10.0) == "killed"
        # The closing SIGKILL reaches only the zombie.
        assert [entry[1:] for entry in host.sent] == [
            (f"group {_GROUP}", "SIGTERM", [_LEADER]), (f"group {_GROUP}", "SIGKILL", [_LEADER])]
        assert host.now == 0.0
        assert _LEADER in host.zombies, "the kill reaped the child"

    def test_without_a_child_the_kill_is_the_one_by_number(self, monkeypatch, fast_path):
        """Ownership is handed over, never inferred: a bare pid, even this
        process's child, gets the kill by number, whose zombie still counts as
        running (U77)."""
        host = _owned_host(fast_path).install(monkeypatch)
        assert _kill_owned(host, child=False, grace=0.5) == "survived"
        assert [entry[1:3] for entry in host.sent] == [
            (f"group {_GROUP}", "SIGTERM"), (f"group {_GROUP}", "SIGKILL")]
        assert host.now == pytest.approx(0.5 + process_control._KILL_CONFIRM_S, abs=0.11)

    def test_a_member_forked_after_the_sigterm_takes_the_sigkill(
        self, monkeypatch, fast_path
    ):
        host = _owned_host(fast_path, members=[_MEMBER], deaf=[_MEMBER])
        late = 4104
        host.after_term = [
            (0.05, host.fork(late, _GROUP, deaf=True)),
            # The member the kill saw first leaves before the SIGKILL.
            (0.06, host.exit(_MEMBER)),
        ]
        host.install(monkeypatch)
        assert _kill_owned(host, grace=0.3) == "killed"
        assert host.hit_bystanders() == []
        assert [entry[1:] for entry in host.sent] == [
            (f"group {_GROUP}", "SIGTERM", [_LEADER, _MEMBER]),
            (f"group {_GROUP}", "SIGKILL", [_LEADER, late]),
        ], host.sent
        assert late not in host.procs

    def test_the_sigkill_still_lands_when_the_last_member_leaves_first(
        self, monkeypatch, fast_path
    ):
        """Between the grace's last look and the SIGKILL the group may empty
        of all but the zombie: its number is still the leader's."""
        host = _owned_host(fast_path, members=[_MEMBER], deaf=[_MEMBER])
        host.before_sigkill = [host.exit(_MEMBER)]
        host.install(monkeypatch)
        assert _kill_owned(host, grace=0.3) == "killed"
        assert host.hit_bystanders() == []
        assert [entry[1:] for entry in host.sent][-1] == (
            f"group {_GROUP}", "SIGKILL", [_LEADER])

    def test_a_member_the_sigkill_cannot_end_is_reported(self, monkeypatch, fast_path):
        host = _owned_host(fast_path, members=[_MEMBER], deaf=[_MEMBER],
                           unkillable=[_MEMBER]).install(monkeypatch)
        assert _kill_owned(host, grace=0.3) == "survived"
        assert host.now == pytest.approx(0.3 + process_control._KILL_CONFIRM_S, abs=0.11)

    def test_a_child_reaped_behind_the_kill_gets_no_sigkill(self, monkeypatch, fast_path):
        """The anchor is gone with the zombie: the numbers may name others now."""
        host = _owned_host(fast_path, members=[_MEMBER], deaf=[_MEMBER])
        host.after_term = [(0.01, host.reap(_LEADER))]
        host.install(monkeypatch)
        assert _kill_owned(host, grace=0.3) == "survived"
        assert [entry[2] for entry in host.sent] == ["SIGTERM"], host.sent

    def test_a_member_forked_while_a_look_reads_the_table_takes_the_sigkill(
        self, monkeypatch, fast_path
    ):
        """The look after the SIGTERM misses the member's new child: it was
        forked after the table's names were read, and the member exited before
        its own row was."""
        host = _owned_host(fast_path, members=[_MEMBER], deaf=[_MEMBER])
        late = 4104
        real_table = host.table

        def table(*args):
            if host.term_at is not None and _MEMBER in host.procs:
                host.fork(late, _GROUP, deaf=True)()
                host.exit(_MEMBER)()
                return {pid: row for pid, row in real_table().items() if pid != late}
            return real_table()

        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_waiting_process_table", table)
        assert _kill_owned(host, grace=0.3) == "killed"
        assert late not in host.procs, "the child forked behind the look survived"
        assert [entry[1:] for entry in host.sent][-1] == (
            f"group {_GROUP}", "SIGKILL", [_LEADER, late])


class TestWhatAnchorsTheOwnedKillOnAFakeClock:
    """The owned kill ends on the zombie at once (``killed`` at 0 s); anything
    that does not anchor gets the kill by number (:func:`_by_number`)."""

    def test_the_anchored_child(self, monkeypatch, fast_path):
        host = _owned_host(fast_path).install(monkeypatch)
        assert _kill_owned(host) == "killed" and host.now == 0.0

    def test_a_popen_that_was_reaped(self, monkeypatch, fast_path):
        host = _owned_host(fast_path).install(monkeypatch)
        popen = types.SimpleNamespace(pid=_LEADER, returncode=0)
        assert _kill_owned(host, popen=popen) == "survived" and _by_number(host)

    def test_a_popen_for_another_pid(self, monkeypatch, fast_path):
        host = _owned_host(fast_path).install(monkeypatch)
        popen = types.SimpleNamespace(pid=_LEADER + 1, returncode=None)
        assert _kill_owned(host, popen=popen) == "survived" and _by_number(host)

    def test_a_group_the_child_does_not_lead(self, monkeypatch, fast_path):
        other = 4200
        host = _Host({_LEADER: (other, "leader-start")}, owned=[_LEADER],
                     fast_path=fast_path).install(monkeypatch)
        assert _kill_owned(host, pgid=other) == "survived" and _by_number(host)
        assert {entry[1] for entry in host.sent} == {f"group {other}"}

    @pytest.mark.parametrize("action, python_level", [
        ((int(signal.SIG_IGN), 0), signal.SIG_DFL),
        ((int(signal.SIG_DFL), 0x2), signal.SIG_DFL),  # Linux's SA_NOCLDWAIT
        # Python's own record refuses even where the kernel's answer is unread.
        (None, signal.SIG_IGN),
        # And never authorises: unread, its default handler proves nothing.
        (None, signal.SIG_DFL),
    ], ids=["SIG_IGN", "SA_NOCLDWAIT", "signal-module-SIG_IGN",
            "unread-with-python-default"])
    def test_children_reaped_at_exit(self, monkeypatch, fast_path, action, python_level):
        host = _owned_host(fast_path).install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", lambda: action)
        monkeypatch.setattr(process_control.signal, "getsignal",
                            lambda _signum: python_level)
        assert _kill_owned(host) == "survived" and _by_number(host)

    def test_sigchld_ignored_after_the_sigterm_stops_the_sigkill(
        self, monkeypatch, fast_path
    ):
        """The disposition is read again right before the SIGKILL."""
        host = _owned_host(fast_path, members=[_MEMBER], deaf=[_MEMBER])
        disposition = [(int(signal.SIG_DFL), 0)]
        host.after_term = [(0.01, lambda: disposition.__setitem__(0, (int(signal.SIG_IGN), 0)))]
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", lambda: disposition[0])
        assert _kill_owned(host, grace=0.3) == "survived"
        assert [entry[2] for entry in host.sent] == ["SIGTERM"], host.sent

    def test_a_failed_native_read_with_pythons_default_handler(self, monkeypatch, fast_path):
        """The ``sigaction`` call itself fails: no disposition is read, and
        Python's ``SIG_DFL`` does not stand in for it."""
        host = _owned_host(fast_path).install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", _REAL_SIGCHLD_ACTION)
        monkeypatch.setattr(process_control, "_SIGACTION_ABI", ["glibc"])
        calls = []

        def failing_sigaction(*args):
            calls.append(args)
            return -1

        monkeypatch.setattr(process_control, "_SIGACTION", [(failing_sigaction, _FakeStruct)])
        monkeypatch.setattr(process_control.signal, "getsignal",
                            lambda _signum: signal.SIG_DFL)
        assert _kill_owned(host) == "survived" and _by_number(host)
        assert calls, "the native read was never tried"

    def test_a_disposition_unread_before_the_sigterm_hands_the_kill_over(
        self, monkeypatch, fast_path
    ):
        """Read on entry, unread at the re-check: nothing owned goes out, and
        the whole kill is the one by number."""
        host = _owned_host(fast_path).install(monkeypatch)
        reads = []

        def disposition():
            reads.append(host.now)
            return (int(signal.SIG_DFL), 0) if len(reads) == 1 else None

        monkeypatch.setattr(process_control, "_sigchld_action", disposition)
        assert _kill_owned(host) == "survived" and _by_number(host)

    def test_a_disposition_unread_before_the_sigkill_hands_the_rest_over(
        self, monkeypatch, fast_path
    ):
        """Unread at the SIGKILL's re-check: the owned path sends no SIGKILL
        of its own; the kill by number finishes, its grace already spent."""
        host = _owned_host(fast_path, members=[_MEMBER], deaf=[_MEMBER])
        readable = [True]
        host.after_term = [(0.25, lambda: readable.__setitem__(0, False))]
        host.install(monkeypatch)
        monkeypatch.setattr(
            process_control, "_sigchld_action",
            lambda: (int(signal.SIG_DFL), 0) if readable[0] else None)
        assert _kill_owned(host, grace=0.3) == "survived"
        # The owned SIGTERM, then the kill by number from its start with no
        # grace: its SIGTERM, its SIGKILL, its SIGKILL window on the zombie.
        assert [entry[2] for entry in host.sent] == ["SIGTERM", "SIGTERM", "SIGKILL"]
        assert host.now == pytest.approx(0.3 + process_control._KILL_CONFIRM_S, abs=0.11)
        assert _MEMBER not in host.procs

    def test_a_failed_group_signal_falls_back_to_the_child_while_it_anchors(
        self, monkeypatch, fast_path
    ):
        host = _owned_host(fast_path)
        host.failing_group_signals = {signal.SIGKILL: None}
        host.install(monkeypatch)
        assert _kill_owned(host) == "killed"
        assert [entry[1:3] for entry in host.sent] == [
            (f"group {_GROUP}", "SIGTERM"), (f"pid {_LEADER}", "SIGKILL")]

    def test_a_failed_group_signal_sends_nothing_once_the_child_is_reaped(
        self, monkeypatch, fast_path
    ):
        """Reaped while the group signal failed, and its pid taken at once by
        another process: the fallback re-reads the anchor and stops."""
        host = _owned_host(fast_path)

        def reaped_and_reused():
            host.reap(_LEADER)()
            host.spawn(_LEADER, 7777, "replacement")()

        host.failing_group_signals = {signal.SIGKILL: reaped_and_reused}
        host.install(monkeypatch)
        assert _kill_owned(host) == "survived"
        assert host.hit_bystanders() == []
        assert [entry[2] for entry in host.sent] == ["SIGTERM"], host.sent


# ---------------------------------------------------------------------------
# Where the owned kill runs at all.
# ---------------------------------------------------------------------------


class _Sys:
    """``sys`` for :mod:`process_control`, with another platform's name, and
    Android's API-level call where *android* says so."""

    def __init__(self, platform, *, android=False):
        self.platform = platform
        if android:
            self.getandroidapilevel = lambda: 24

    def __getattr__(self, name):
        return getattr(sys, name)


def _platform(monkeypatch, platform, *, machine, glibc=None, interpreter=None,
              android=False):
    """Make :mod:`process_control` see *platform* on *machine*, with glibc
    naming itself as *glibc* and the executable naming *interpreter* as its
    loader, and forget what it worked out before."""
    real_os = process_control.os

    class _Os:
        @staticmethod
        def uname():
            return types.SimpleNamespace(machine=machine)

        def __getattr__(self, name):
            return getattr(real_os, name)

    monkeypatch.setattr(process_control, "sys", _Sys(platform, android=android))
    monkeypatch.setattr(process_control, "os", _Os())
    monkeypatch.setattr(process_control, "_glibc_version", lambda: glibc)
    monkeypatch.setattr(process_control, "_program_interpreter", lambda *_a: interpreter)
    monkeypatch.setattr(process_control, "_SIGACTION_ABI", [])
    monkeypatch.setattr(process_control, "_SIGACTION", [])


#: (platform, machine, glibc's name for itself, the loader, Android) per case.
_CHECKED = {
    "glibc-x86_64": ("linux", "x86_64", "glibc 2.39", "/lib64/ld-linux-x86-64.so.2", False),
    "glibc-i686": ("linux", "i686", "glibc 2.39", "/lib/ld-linux.so.2", False),
    "glibc-aarch64": ("linux", "aarch64", "glibc 2.39", "/lib/ld-linux-aarch64.so.1", False),
    "musl-x86_64": ("linux", "x86_64", None, "/lib/ld-musl-x86_64.so.1", False),
    "musl-i386-on-x86_64": ("linux", "x86_64", None, "/lib/ld-musl-i386.so.1", False),
    "musl-aarch64": ("linux", "aarch64", None, "/lib/ld-musl-aarch64.so.1", False),
}
_NOT_CHECKED = {
    # Python < 3.13 on Android says "linux"; Bionic has neither name.
    "bionic-as-linux": ("linux", "aarch64", None, "/system/bin/linker64", True),
    # Android's API call alone refuses, whatever else answers.
    "bionic-claiming-glibc": ("linux", "aarch64", "glibc 2.39", "/lib/ld-linux-aarch64.so.1",
                              True),
    "android-3.13": ("android", "aarch64", None, "/system/bin/linker64", False),
    "unknown-libc": ("linux", "x86_64", None, "/lib/ld-uClibc.so.0", False),
    "static-musl": ("linux", "x86_64", None, None, False),
    "static-glibc": ("linux", "x86_64", "glibc 2.39", None, False),
    "musl-x32": ("linux", "x86_64", None, "/lib/ld-musl-x32.so.1", False),
    "glibc-x32": ("linux", "x86_64", "glibc 2.39", "/libx32/ld-linux-x32.so.2", False),
    # A 32-bit Arm program on a 64-bit kernel: the machine is checked, its ABI is not.
    "glibc-armhf-on-aarch64": ("linux", "aarch64", "glibc 2.36", "/lib/ld-linux-armhf.so.3",
                               False),
    "musl-armhf-on-aarch64": ("linux", "aarch64", None, "/lib/ld-musl-armhf.so.1", False),
    # i386's loader name, on the machines that name theirs the same.
    "glibc-alpha": ("linux", "alpha", "glibc 2.39", "/lib/ld-linux.so.2", False),
    "glibc-sparc64": ("linux", "sparc64", "glibc 2.39", "/lib/ld-linux.so.2", False),
    # The library and its loader disagree.
    "glibc-loader-without-glibc": ("linux", "x86_64", None, "/lib64/ld-linux-x86-64.so.2",
                                   False),
    "musl-loader-claiming-glibc": ("linux", "x86_64", "glibc 2.39",
                                   "/lib/ld-musl-x86_64.so.1", False),
    "glibc-mips64": ("linux", "mips64", "glibc 2.39", "/lib64/ld.so.1", False),
    "musl-mips64": ("linux", "mips64", None, "/lib/ld-musl-mips64.so.1", False),
    "glibc-s390x": ("linux", "s390x", "glibc 2.39", "/lib/ld64.so.1", False),
    "macos": ("darwin", "arm64", None, None, False),
    "macos-intel": ("darwin", "x86_64", None, None, False),
    "freebsd": ("freebsd14", "amd64", None, "/libexec/ld-elf.so.1", False),
    # glibc on another kernel: the C library is known, the kernel's proof is not.
    "glibc-on-kfreebsd": ("gnukfreebsd10", "x86_64", "glibc 2.19",
                          "/lib64/ld-linux-x86-64.so.2", False),
    "glibc-on-hurd": ("gnu0", "x86_64", "glibc 2.38", "/lib/ld-x86-64.so.1", False),
}


class TestOnlyACheckedPlatformTakesTheOwnedKill:
    """The owned kill's proof was checked for Linux (a fork and a group signal
    take the same lock) under glibc or musl on the machines whose ``struct
    sigaction`` layout was read. Anywhere else the C library is not named,
    SIGCHLD's disposition is not read, and the kill is the one by number."""

    @pytest.mark.parametrize("case", sorted(_CHECKED))
    def test_a_checked_platform_is_named(self, monkeypatch, case):
        platform, machine, glibc, loader, android = _CHECKED[case]
        _platform(monkeypatch, platform, machine=machine, glibc=glibc,
                  interpreter=loader, android=android)
        assert process_control._sigaction_abi() == case.split("-")[0]

    @pytest.mark.parametrize("case", sorted(_NOT_CHECKED))
    def test_any_other_is_not(self, monkeypatch, case):
        platform, machine, glibc, loader, android = _NOT_CHECKED[case]
        _platform(monkeypatch, platform, machine=machine, glibc=glibc,
                  interpreter=loader, android=android)
        assert process_control._sigaction_abi() is None

    @pytest.mark.parametrize("case", sorted(_NOT_CHECKED))
    def test_any_other_reads_nothing_natively(self, monkeypatch, case):
        """No ``struct sigaction`` is laid over memory whose layout is unknown."""
        platform, machine, glibc, loader, android = _NOT_CHECKED[case]
        _platform(monkeypatch, platform, machine=machine, glibc=glibc,
                  interpreter=loader, android=android)
        import ctypes

        def no_native_call(*_args, **_kwargs):
            raise AssertionError("ctypes was reached on an unchecked platform")

        monkeypatch.setattr(ctypes, "CDLL", no_native_call)
        assert process_control._sigchld_action() is None
        assert process_control._children_become_zombies() is None
        assert process_control._SIGACTION == []

    @pytest.mark.parametrize("case", sorted(_NOT_CHECKED))
    def test_any_other_gets_the_kill_by_number(self, monkeypatch, case):
        """The scripted host would take the owned kill (``killed`` at once):
        on these platforms it gets the kill by number instead."""
        host = _owned_host(True).install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", _REAL_SIGCHLD_ACTION)
        platform, machine, glibc, loader, android = _NOT_CHECKED[case]
        _platform(monkeypatch, platform, machine=machine, glibc=glibc,
                  interpreter=loader, android=android)
        assert _kill_owned(host) == "survived" and _by_number(host)
        assert _LEADER in host.zombies, "the kill reaped the child"

    def test_a_checked_platform_gets_the_owned_kill(self, monkeypatch):
        """The control: the same host, named glibc on x86_64, and the native
        read answering SIGCHLD's default."""
        host = _owned_host(True).install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", _REAL_SIGCHLD_ACTION)
        _platform(monkeypatch, "linux", machine="x86_64", glibc="glibc 2.39",
                  interpreter="/lib64/ld-linux-x86-64.so.2")
        monkeypatch.setattr(process_control, "_SIGACTION",
                            [(lambda *_args: 0, _FakeStruct)])
        assert _kill_owned(host, grace=10.0) == "killed" and host.now == 0.0

    def test_without_waitid_the_kill_is_the_one_by_number(self, monkeypatch):
        host = _owned_host(True).install(monkeypatch)
        monkeypatch.setattr(process_control, "_waitid", None)
        assert _kill_owned(host) == "survived" and _by_number(host)

    @pytest.mark.parametrize("answer, named", [
        ("glibc 2.39", "glibc 2.39"),
        (OSError(22, "Invalid argument"), None),  # musl refuses the name
        (ValueError("unrecognized configuration name"), None),  # Bionic lacks it
        (None, None),
        ("uClibc-ng 1.0.45", None),
    ], ids=["glibc", "musl", "bionic", "empty", "other"])
    def test_only_glibc_names_itself_glibc(self, monkeypatch, answer, named):
        real_os = process_control.os

        class _Os:
            @staticmethod
            def confstr(name):
                assert name == "CS_GNU_LIBC_VERSION"
                if isinstance(answer, Exception):
                    raise answer
                return answer

            def __getattr__(self, name):
                return getattr(real_os, name)

        monkeypatch.setattr(process_control, "os", _Os())
        assert process_control._glibc_version() == named


def _elf(path, *, wide, little, interpreter):
    """A minimal ELF file whose program headers hold a ``PT_LOAD`` and, given
    an *interpreter*, a ``PT_INTERP`` naming it."""
    import struct

    order = "<" if little else ">"
    header_size, entry_size = (64, 56) if wide else (52, 32)
    entries = 2 if interpreter else 1
    name_at = header_size + entries * entry_size
    name = (interpreter or "").encode() + b"\0"
    ident = b"\x7fELF" + bytes([2 if wide else 1, 1 if little else 2, 1]) + bytes(9)
    if wide:
        header = ident + struct.pack(order + "HHIQQQIHHHHHH", 3, 62, 1, 0, header_size, 0,
                                     0, header_size, entry_size, entries, 0, 0, 0)
    else:
        header = ident + struct.pack(order + "HHIIIIIHHHHHH", 3, 3, 1, 0, header_size, 0,
                                     0, header_size, entry_size, entries, 0, 0, 0)

    def entry(kind, offset, size):
        if wide:
            return struct.pack(order + "IIQQQQQQ", kind, 4, offset, 0, 0, size, size, 1)
        return struct.pack(order + "IIIIIIII", kind, offset, 0, 0, size, size, 4, 1)

    table = entry(1, 0, name_at)
    if interpreter:
        table = entry(3, name_at, len(name)) + table
    path.write_bytes(header + table + (name if interpreter else b""))
    return path


class TestTheLoaderAnExecutableNames:
    @pytest.mark.parametrize("wide, little", [
        (True, True), (True, False), (False, True), (False, False)],
        ids=["64-le", "64-be", "32-le", "32-be"])
    def test_read_from_pt_interp(self, tmp_path, wide, little):
        exe = _elf(tmp_path / "exe", wide=wide, little=little,
                   interpreter="/lib/ld-musl-x86_64.so.1")
        assert process_control._program_interpreter(str(exe)) == "/lib/ld-musl-x86_64.so.1"

    def test_a_static_executable_names_none(self, tmp_path):
        exe = _elf(tmp_path / "exe", wide=True, little=True, interpreter=None)
        assert process_control._program_interpreter(str(exe)) is None

    @pytest.mark.parametrize("content", [
        b"", b"#!/bin/sh\n", b"\x7fELF" + bytes(10), b"\x7fELF\x03\x01" + bytes(60)],
        ids=["empty", "script", "truncated", "bad-class"])
    def test_anything_else_names_none(self, tmp_path, content):
        exe = tmp_path / "exe"
        exe.write_bytes(content)
        assert process_control._program_interpreter(str(exe)) is None

    def test_an_unreadable_file_names_none(self, tmp_path):
        assert process_control._program_interpreter(str(tmp_path / "missing")) is None


class TestTheVerifyKillSaysWhatItSent:
    """The verify command's timeout note: ``SIGTERM then SIGKILL`` only when the
    kill sent them; ``skipped`` sent nothing."""

    def _timed_out(self, monkeypatch, tmp_path, verdict):
        from utils import write_batch_scheduler as wbs

        real_kill = process_control.kill_tree

        def kill_then_report(pid, **kwargs):
            real_kill(pid, **kwargs)
            return verdict

        monkeypatch.setattr(wbs._process_control, "kill_tree", kill_then_report)
        scheduler = object.__new__(wbs.WriteBatchScheduler)
        scheduler.repo_root = tmp_path
        # An unconfirmed kill is logged as a warning naming the pid.
        self.logged = []
        scheduler.log = self.logged.append
        monkeypatch.setattr(scheduler, "_verify_env", lambda: None, raising=False)
        exit_code, tail = scheduler._run_verify_command("echo started; exec sleep 30", 1)
        assert exit_code == wbs.VERIFY_EXIT_TIMEOUT
        assert "started" in tail
        return tail

    def test_a_skipped_kill_says_termination_was_attempted(self, monkeypatch, tmp_path):
        tail = self._timed_out(monkeypatch, tmp_path, "skipped")
        assert "termination attempted" in tail and "no signal was sent" in tail
        assert "SIGTERM then SIGKILL" not in tail and "killed after" not in tail
        assert tail.rstrip().endswith("verdict skipped)")
        assert [line for line in self.logged if "kill verdict skipped" in line]

    def test_a_killed_kill_names_its_signals(self, monkeypatch, tmp_path):
        tail = self._timed_out(monkeypatch, tmp_path, "killed")
        assert "killed after 1s: SIGTERM then SIGKILL" in tail
        assert tail.rstrip().endswith("verdict killed)")
        assert self.logged == []
