"""No POSIX signal to a target this process does not own reaches a bystander.

A number names whoever holds it when the signal lands. A recorded pid can be
handed to another process once its own is reaped; a recorded process group's
number can be made again by anyone once the group empties, and a process can
leave a group and join a new one of the same number within its session. So on
Linux with pidfds and ``/proc`` the kill of a target known only by number
(``kill_tree`` without an owned ``child``, and the Lane A sweep of a CLI's
snapshotted groups) signals no number at all, and only processes identified
one by one before it: the target and its recorded *members* (a detached
worker's provider CLIs, a Lane A CLI's snapshotted descendants), each through a
pidfd verified to hold it. Groups are only read, afterwards: something
unidentified still running in one makes the verdict ``unresolved``, and the
callers keep their records of the run until a later kill confirms it.
docs/development/runtime.md, "A target known only by number".

Without pidfds (macOS, the BSDs, an old kernel, an ABI not checked) or
``/proc`` the kill is the one by number, exactly as it was: where a scenario
runs there too, it pins that kill's outcome, the documented residual risk
included.

The scenarios run on the scripted host of ``test_kill_tree_posix`` (a fake
clock; nothing real is signalled), each at the boundary a review found. The
record, abort, close-out and sweep tests check the verdicts and records the
callers keep; the real-process tests at the end use processes this test starts.
"""

import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_kill_tree_posix import _Host  # noqa: E402
from utils import detach, process_control, run_registry, run_status, status_file  # noqa: E402

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")

#: A detached worker's group: its leader (the intermediate that made the
#: session) is gone, the worker and a provider CLI are in it.
_GROUP = 4101
_WORKER = 4102
_MEMBER = 4103
_BYSTANDER = 4104
#: A session id, and a group number some unrelated program holds elsewhere.
_SESSION = 7777
_ELSEWHERE = 7778


@pytest.fixture(params=[True, False], ids=["proc", "ps"])
def fast_path(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["pidfd", "no-pidfd"])
def pidfd(request):
    return request.param


def _sweep(snapshot, grace=0.3):
    return process_control._sweep_groups(snapshot, grace, 0.1)


def _kill(pid, *, start_time, pgid, grace=0.3, **kwargs):
    return process_control.kill_tree(
        pid, start_time=start_time, pgid=pgid, escalate_after_s=grace,
        poll_interval_s=0.1, **kwargs)


def _record(pid, pgid, start):
    return process_control.Descendant(pid, pgid, pgid, start)


def _cli(pid, start, pgid=None):
    """A provider CLI as the worker's record names it."""
    return process_control.Descendant(pid, pgid, None, start)


def _signals(host):
    return [entry[1:3] for entry in host.sent]


def _outcome(host, verdict):
    """What a kill did: its verdict, each signal as (how, name), every
    bystander a signal reached."""
    return verdict, _signals(host), host.hit_bystanders()


def _by_identity(fast_path, pidfd):
    return fast_path and pidfd


def _racing_read(host, monkeypatch):
    """The first ``/proc`` pass after the worker was held: it reads the
    worker's row, then the worker exits and is reaped, a new session takes the
    group's number and a bystander joins it, all before the bystander's row
    is read."""
    real = host.table
    fired = []

    def scan(*_args):
        if fired or (host.pidfd and host.fast_path and not host.open_pidfds):
            return real()
        fired.append(host.now)
        old = process_control._ProcRow(_WORKER, 1, _GROUP, _GROUP, "worker", "S")
        host.procs.pop(_WORKER, None)
        host.spawn(_GROUP, _GROUP, "new-session")()
        host.procs[_BYSTANDER] = (_GROUP, "bystander")
        host.bystanders.add(_BYSTANDER)
        return {_WORKER: old, _BYSTANDER: real()[_BYSTANDER]}

    monkeypatch.setattr(process_control, "_linux_process_table",
                        lambda: scan() if host.fast_path else None)
    monkeypatch.setattr(process_control, "_process_table", scan)
    monkeypatch.setattr(process_control, "_waiting_process_table", lambda *_a: scan())
    return fired


# ---------------------------------------------------------------------------
# Round 2, point 1: an anchor leaves its group and rejoins one made again
# ---------------------------------------------------------------------------


class TestAProcessThatLeavesAndRejoinsARecycledGroup:
    """Group ``_GROUP`` is not its session's; the held process leaves it, the
    number is freed and taken by an unrelated session's new group, a
    bystander joins that group, and the held process rejoins it, all inside
    the kill's first ``/proc`` pass. Its pidfd holds it throughout, so nothing
    about the process tells the kill the group is not the one it was in."""

    @staticmethod
    def _leave_and_rejoin(host, monkeypatch):
        fired = []

        def rows():
            return {pid: process_control._ProcRow(pid, 1, pgid, _SESSION, start, "S")
                    for pid, (pgid, start) in host.procs.items()}

        def scan():
            if fired or not host.open_pidfds:
                return rows()
            fired.append(True)
            host.procs[_WORKER] = (_SESSION, "anchor")         # leaves; the number is free
            host.spawn(_GROUP, _GROUP, "new-group-leader")()   # an unrelated new group
            host.spawn(_BYSTANDER, _GROUP, "bystander")()
            result = rows()                                   # the bystander, in the new group
            host.procs[_BYSTANDER] = (_SESSION, "bystander")
            host.procs[_WORKER] = (_GROUP, "anchor")           # rejoins before any later read
            return result

        monkeypatch.setattr(process_control, "_linux_process_table", scan)
        monkeypatch.setattr(process_control, "_linux_row",
                            lambda pid: (True, rows().get(pid)))
        return fired

    @pytest.mark.parametrize("via", ["kill_tree", "lane_a"])
    def test_nothing_in_the_group_is_signalled_and_the_verdict_says_so(
        self, monkeypatch, via
    ):
        host = _Host({_WORKER: (_GROUP, "anchor")}, deaf=[_WORKER], pidfd=True)
        host.install(monkeypatch)
        fired = self._leave_and_rejoin(host, monkeypatch)
        if via == "kill_tree":
            verdict = _kill(_WORKER, start_time="anchor", pgid=_GROUP)
        else:
            verdict = _sweep([process_control.Descendant(_WORKER, _GROUP, _SESSION, "anchor")])
        assert fired
        assert host.hit_bystanders() == []
        # The held process itself is stopped, through its own pidfd.
        assert _signals(host) == [(f"pidfd {_WORKER}", "SIGTERM"),
                                  (f"pidfd {_WORKER}", "SIGKILL")]
        assert _WORKER not in host.procs
        # The group runs what nobody identified: not called killed.
        assert verdict == ("unresolved" if via == "kill_tree" else {_GROUP: "unresolved"})
        assert {_GROUP, _BYSTANDER} <= set(host.procs)
        assert host.open_pidfds == set()


# ---------------------------------------------------------------------------
# Round 1's interleavings, with pidfds and without
# ---------------------------------------------------------------------------


class TestAGroupMadeAgainDuringTheReadSignalsNothingInIt:
    """Round 1, point 1: the read of a group is not atomic."""

    def test_in_the_lane_a_sweep(self, monkeypatch, fast_path, pidfd):
        host = _Host({_WORKER: (_GROUP, "worker")}, deaf=[_WORKER],
                     fast_path=fast_path, pidfd=pidfd).install(monkeypatch)
        fired = _racing_read(host, monkeypatch)
        verdict = _sweep([_record(_WORKER, _GROUP, "worker")])
        assert fired
        if _by_identity(fast_path, pidfd):
            assert _outcome(host, verdict) == ({_GROUP: "unresolved"}, [], [])
            assert host.open_pidfds == set()
        else:
            # By number, as it always was: the documented residual risk.
            assert _outcome(host, verdict) == (
                {_GROUP: "survived"},
                [(f"group {_GROUP}", "SIGTERM"), (f"group {_GROUP}", "SIGKILL")],
                [(f"group {_GROUP}", _GROUP), (f"group {_GROUP}", _BYSTANDER)] * 2)

    def test_in_kill_tree(self, monkeypatch, fast_path, pidfd):
        host = _Host({_WORKER: (_GROUP, "worker")}, deaf=[_WORKER],
                     fast_path=fast_path, pidfd=pidfd).install(monkeypatch)
        fired = _racing_read(host, monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP)
        if _by_identity(fast_path, pidfd):
            assert fired
            assert _outcome(host, verdict) == ("unresolved", [], [])
            assert host.open_pidfds == set()
        else:
            # The kill by number never read the group: its race is elsewhere.
            assert _outcome(host, verdict) == (
                "killed", [(f"group {_GROUP}", "SIGTERM"), (f"group {_GROUP}", "SIGKILL")], [])


class TestAProcessReplacedAfterTheLastCheckGetsNothing:
    """Round 1, point 2: every member exits and is reaped, and its numbers are
    handed out again, after the kill's last check and before its SIGKILL
    (the scripted host's ``before_sigkill``)."""

    @staticmethod
    def _group_replaced(host):
        def replace():
            host.procs.clear()
            host.spawn(_GROUP, _GROUP, "bystander")()
        host.before_sigkill = [replace]

    def test_a_group_in_the_lane_a_sweep(self, monkeypatch, fast_path, pidfd):
        host = _Host({_WORKER: (_GROUP, "worker"), _MEMBER: (_GROUP, "member")},
                     deaf=[_WORKER, _MEMBER], fast_path=fast_path, pidfd=pidfd)
        self._group_replaced(host)
        host.install(monkeypatch)
        verdict = _sweep([_record(_WORKER, _GROUP, "worker"),
                          _record(_MEMBER, _GROUP, "member")])
        if _by_identity(fast_path, pidfd):
            assert _outcome(host, verdict) == (
                {_GROUP: "unresolved"},
                [(f"pidfd {_WORKER}", "SIGTERM"), (f"pidfd {_MEMBER}", "SIGTERM")], [])
        else:
            assert _outcome(host, verdict) == (
                {_GROUP: "survived"},
                [(f"group {_GROUP}", "SIGTERM"), (f"group {_GROUP}", "SIGKILL")],
                [(f"group {_GROUP}", _GROUP)])

    def test_a_group_in_kill_tree(self, monkeypatch, fast_path, pidfd):
        host = _Host({_WORKER: (_GROUP, "worker"), _MEMBER: (_GROUP, "member")},
                     deaf=[_WORKER, _MEMBER], fast_path=fast_path, pidfd=pidfd)
        self._group_replaced(host)
        host.install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "member")])
        if _by_identity(fast_path, pidfd):
            assert _outcome(host, verdict) == (
                "unresolved",
                [(f"pidfd {_WORKER}", "SIGTERM"), (f"pidfd {_MEMBER}", "SIGTERM")], [])
        else:
            assert _outcome(host, verdict) == (
                "killed", [(f"group {_GROUP}", "SIGTERM"), (f"group {_GROUP}", "SIGKILL")],
                [(f"group {_GROUP}", _GROUP)])

    @pytest.mark.parametrize("via", ["kill_tree", "lane_a"])
    def test_a_pid(self, monkeypatch, fast_path, pidfd, via):
        """A target in this process's own group, so only its pid is signalled."""
        own = _Host.OWN_PGID
        host = _Host({_WORKER: (own, "worker")}, deaf=[_WORKER],
                     fast_path=fast_path, pidfd=pidfd)

        def replace():
            host.procs.clear()
            host.spawn(_WORKER, _ELSEWHERE, "bystander")()

        host.before_sigkill = [replace]
        host.install(monkeypatch)
        if via == "kill_tree":
            verdict = _kill(_WORKER, start_time="worker", pgid=own)
        else:
            verdict = _sweep([_record(_WORKER, own, "worker")])
        if _by_identity(fast_path, pidfd):
            # The SIGKILL through its pidfd reaches nothing: its process is gone.
            assert _outcome(host, verdict) == (
                "killed" if via == "kill_tree" else {own: "killed"},
                [(f"pidfd {_WORKER}", "SIGTERM")], [])
        else:
            assert _outcome(host, verdict) == (
                "survived" if via == "kill_tree" else {own: "killed"},
                [(f"pid {_WORKER}", "SIGTERM"), (f"pid {_WORKER}", "SIGKILL")],
                [(f"pid {_WORKER}", _WORKER)])


class TestAPidfdIsNeverPairedWithItsReplacementsRow:
    """Round 1, point 3: the pidfd opens on the worker; it exits, and its pid
    and its group's number are handed out again before its row is read. By
    identity the worker is held by its recorded start time (without one it is
    not held at all: ``TestAnUndatedTargetIsNeverSignalled``); by number the
    kill is the unverified one it always was."""

    def test_the_target_is_retired_and_its_old_group_is_not_read(
        self, monkeypatch, fast_path, pidfd
    ):
        host = _Host({_WORKER: (_GROUP, "worker")}, fast_path=fast_path, pidfd=pidfd)
        host.install(monkeypatch)

        def replace():
            host.procs.clear()
            host.spawn(_GROUP, _GROUP, "new-session")()
            host.spawn(_WORKER, _GROUP, "replacement")()
            host.spawn(_BYSTANDER, _GROUP, "bystander")()

        if _by_identity(fast_path, pidfd):
            api = process_control._pidfd_api()
            real_open = api._open

            def opening(pid):
                fd = real_open(pid)
                replace()
                return fd

            monkeypatch.setattr(api, "_open", opening)
        else:
            # By number: the same replacement, right after the first read of the pid.
            done = []

            def reading(pid):
                if not done:
                    done.append(pid)
                    replace()
                return host.start_time(pid)

            monkeypatch.setattr(process_control, "process_start_time", reading)
        verdict = _kill(_WORKER, start_time="worker" if _by_identity(fast_path, pidfd) else None,
                        pgid=_GROUP)
        if _by_identity(fast_path, pidfd):
            # Nothing sent. The group's number is the pid of a live process
            # that is none of the target's, so the group is that process's
            # (a number in use by a group is never handed out as a pid).
            assert _outcome(host, verdict) == ("gone", [], [])
            assert host.open_pidfds == set()
        else:
            signalled = [_GROUP, _WORKER, _BYSTANDER]
            assert _outcome(host, verdict) == (
                "survived", [(f"group {_GROUP}", "SIGTERM"), (f"group {_GROUP}", "SIGKILL")],
                [(f"group {_GROUP}", pid) for pid in signalled] * 2)


class TestAnInterruptedKillLeaksNoPidfd:
    """Round 1, point 7: whatever raises once a pidfd is open."""

    @pytest.mark.parametrize("where", ["_linux_row", "_linux_process_table"])
    def test_none_is_left_open(self, monkeypatch, where):
        host = _Host({_WORKER: (_GROUP, "worker"), _MEMBER: (_GROUP, "member")},
                     fast_path=True, pidfd=True).install(monkeypatch)
        calls = []
        real = getattr(process_control, where)

        def interrupted(*args):
            calls.append(args)
            if host.open_pidfds:
                raise KeyboardInterrupt()
            return real(*args)

        monkeypatch.setattr(process_control, where, interrupted)
        for _ in range(3):
            with pytest.raises(KeyboardInterrupt):
                _kill(_WORKER, start_time="worker", pgid=_GROUP,
                      members=[_cli(_MEMBER, "member")])
        with pytest.raises(KeyboardInterrupt):
            _sweep([_record(_WORKER, _GROUP, "worker"), _record(_MEMBER, _GROUP, "member")])
        assert calls
        assert host.open_pidfds == set()


# ---------------------------------------------------------------------------
# Only what was identified before the kill is signalled
# ---------------------------------------------------------------------------


class TestOnlyIdentifiedProcessesAreSignalled:
    """With pidfds: the target and its recorded members, nothing found beside them."""

    def test_a_live_worker_and_its_recorded_cli_that_ignores_sigterm(self, monkeypatch):
        host = _Host({_WORKER: (_GROUP, "worker"), _MEMBER: (_GROUP, "member")},
                     deaf=[_MEMBER], pidfd=True).install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "member", _GROUP)])
        assert verdict == "killed"
        assert host.procs == {}
        assert _signals(host) == [
            (f"pidfd {_WORKER}", "SIGTERM"), (f"pidfd {_MEMBER}", "SIGTERM"),
            (f"pidfd {_MEMBER}", "SIGKILL")]
        assert host.open_pidfds == set()

    def test_a_member_nobody_recorded_is_reported_not_signalled(self, monkeypatch, pidfd):
        host = _Host({_WORKER: (_GROUP, "worker"), _MEMBER: (_GROUP, "member")},
                     deaf=[_MEMBER], pidfd=pidfd).install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP)
        if pidfd:
            assert verdict == "unresolved"
            assert _signals(host) == [(f"pidfd {_WORKER}", "SIGTERM")]
            assert _MEMBER in host.procs
        else:
            # By number, as it always was: the group's SIGTERM, then the
            # leader alone decides; what ignored the SIGTERM runs on.
            assert verdict == "killed"
            assert _signals(host) == [(f"group {_GROUP}", "SIGTERM")]

    def test_a_process_forked_during_the_grace_is_reported_not_signalled(
        self, monkeypatch
    ):
        late = 4105
        host = _Host({_WORKER: (_GROUP, "worker"), _MEMBER: (_GROUP, "member")},
                     deaf=[_MEMBER], pidfd=True)
        host.after_term = [(0.05, host.fork(late, _GROUP, deaf=True))]
        host.install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "member", _GROUP)])
        assert verdict == "unresolved"
        assert late in host.procs and _MEMBER not in host.procs
        assert all(entry[1] != f"pidfd {late}" for entry in host.sent)

    def test_what_exits_while_the_groups_are_read_is_waited_for(self, monkeypatch):
        """An unrecorded child the member's own SIGTERM handler stops on its
        way out: the read waits, within the confirmation window, for it."""
        child = 4105
        host = _Host({_WORKER: (_GROUP, "worker"), child: (_GROUP, "child")},
                     deaf=[child], pidfd=True)
        host.after_term = [(0.5, host.exit(child))]
        host.install(monkeypatch)
        assert _kill(_WORKER, start_time="worker", pgid=_GROUP) == "killed"
        assert host.procs == {}
        assert _signals(host) == [(f"pidfd {_WORKER}", "SIGTERM")]

    def test_a_member_that_left_by_setsid_is_followed_alone(self, monkeypatch):
        host = _Host({_WORKER: (_GROUP, "worker"), _MEMBER: (_GROUP, "member")},
                     deaf=[_WORKER, _MEMBER], pidfd=True)
        host.after_term = [(0.05, host.setsid(_MEMBER))]
        host.install(monkeypatch)
        verdict = _sweep([_record(_WORKER, _GROUP, "worker"),
                          _record(_MEMBER, _GROUP, "member")])
        assert verdict == {_GROUP: "killed"}
        assert host.procs == {}
        assert host.hit_bystanders() == []

    def test_a_snapshot_whose_pid_was_handed_out_again_is_skipped(self, monkeypatch):
        host = _Host({_MEMBER: (_ELSEWHERE, "someone-else")}, pidfd=True)
        host.bystanders.add(_MEMBER)
        host.install(monkeypatch)
        assert _sweep([_record(_MEMBER, _GROUP, "member")]) == {_GROUP: "skipped"}
        assert host.sent == []

    def test_a_target_whose_identity_cannot_be_read_gets_nothing(self, monkeypatch):
        host = _Host({_WORKER: (_GROUP, None)}, pidfd=True).install(monkeypatch)
        assert _kill(_WORKER, start_time="worker", pgid=_GROUP) == "skipped"
        assert host.sent == [] and host.open_pidfds == set()

    def test_a_recorded_cli_whose_pid_was_handed_out_again_is_left_alone(
        self, monkeypatch
    ):
        host = _Host({_MEMBER: (_GROUP, "someone-else")}, pidfd=True)
        host.bystanders.add(_MEMBER)
        host.install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "member")])
        # The worker's group still runs something: what, the kill cannot tell.
        assert _outcome(host, verdict) == ("unresolved", [], [])


class TestACrashedWorkersRecordedClis:
    """Round 1, point 4: the worker died without its handler; a provider CLI
    it started ignores SIGTERM and runs on in its group."""

    def test_are_stopped_by_their_identities(self, monkeypatch, fast_path, pidfd):
        host = _Host({_MEMBER: (_GROUP, "member")}, deaf=[_MEMBER],
                     fast_path=fast_path, pidfd=pidfd).install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "member", _GROUP)])
        assert host.hit_bystanders() == []
        assert _MEMBER not in host.procs, "the provider CLI was left running"
        assert verdict == "killed"
        assert _signals(host) == (
            [(f"pidfd {_MEMBER}", "SIGTERM"), (f"pidfd {_MEMBER}", "SIGKILL")]
            if _by_identity(fast_path, pidfd)
            else [(f"group {_GROUP}", "SIGTERM"), (f"group {_GROUP}", "SIGKILL")])

    def test_without_a_record_nothing_is_sent_and_it_says_so(
        self, monkeypatch, fast_path, pidfd
    ):
        host = _Host({_MEMBER: (_GROUP, "member")}, deaf=[_MEMBER],
                     fast_path=fast_path, pidfd=pidfd).install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP)
        if _by_identity(fast_path, pidfd):
            assert _outcome(host, verdict) == ("unresolved", [], [])
            assert _MEMBER in host.procs
        else:
            assert _outcome(host, verdict) == (
                "killed", [(f"group {_GROUP}", "SIGTERM"), (f"group {_GROUP}", "SIGKILL")], [])


# ---------------------------------------------------------------------------
# Round 2, point 3: a recorded CLI's own group is read too
# ---------------------------------------------------------------------------


class TestARecordedClisOwnGroup:
    """An executor CLI runs in a session of its own, so its tool children are
    in its group, not the worker's."""

    def test_its_unrecorded_child_there_is_reported_not_signalled(self, monkeypatch):
        tool = _BYSTANDER
        host = _Host({_MEMBER: (_MEMBER, "cli"), tool: (_MEMBER, "tool")},
                     deaf=[_MEMBER, tool], pidfd=True).install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "cli")])
        assert verdict == "unresolved"
        assert _MEMBER not in host.procs, "the recorded CLI was not stopped"
        assert tool in host.procs
        assert all(entry[1] != f"pidfd {tool}" for entry in host.sent)

    def test_read_by_its_recorded_number_once_the_cli_is_gone(self, monkeypatch):
        tool = _BYSTANDER
        host = _Host({tool: (_MEMBER, "tool")}, pidfd=True).install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "cli", _MEMBER)])
        assert _outcome(host, verdict) == ("unresolved", [], [])

    def test_a_recorded_number_now_another_process_s_group_is_not_read(self, monkeypatch):
        """The CLI is long gone and its pid now leads an unrelated group."""
        host = _Host({_MEMBER: (_MEMBER, "someone-else"),
                      _BYSTANDER: (_MEMBER, "their-child")}, pidfd=True)
        host.bystanders.update({_MEMBER, _BYSTANDER})
        host.install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "cli", _MEMBER)])
        assert _outcome(host, verdict) == ("gone", [], [])

    def test_a_group_it_moved_into_is_read(self, monkeypatch):
        moved_to = 4200
        tool = _BYSTANDER
        host = _Host({_MEMBER: (_GROUP, "cli"), tool: (moved_to, "tool")},
                     deaf=[_MEMBER, tool], pidfd=True)
        host.after_term = [(0.0, lambda: host.procs.__setitem__(_MEMBER, (moved_to, "cli")))]
        host.install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP,
                        members=[_cli(_MEMBER, "cli", _GROUP)])
        assert verdict == "unresolved"
        assert tool in host.procs and _MEMBER not in host.procs


# ---------------------------------------------------------------------------
# Round 2, point 6: what cannot be read is not taken as gone
# ---------------------------------------------------------------------------


class TestWhatCannotBeReadIsNotTakenAsGone:
    def test_rows_lost_after_the_hold(self, monkeypatch):
        """The held process's exit is its pidfd's answer; the groups cannot
        be read, so the verdict cannot be ``killed``."""
        host = _Host({_MEMBER: (_GROUP, "cli")}, deaf=[_MEMBER], pidfd=True)
        host.install(monkeypatch)
        calls = []
        real = process_control._linux_row

        def row(pid):
            calls.append(pid)
            return real(pid) if len(calls) == 1 else (False, None)

        monkeypatch.setattr(process_control, "_linux_row", row)
        monkeypatch.setattr(process_control, "_linux_process_table", lambda: None)
        verdict = _kill(_MEMBER, start_time="cli", pgid=_GROUP)
        assert verdict == "unresolved"
        assert _MEMBER not in host.procs, "the held process was taken as gone"
        assert _signals(host) == [(f"pidfd {_MEMBER}", "SIGTERM"),
                                  (f"pidfd {_MEMBER}", "SIGKILL")]

    def test_a_pidfd_that_cannot_say_and_no_row(self, monkeypatch):
        host = _Host({_MEMBER: (_MEMBER, "cli")}, deaf=[_MEMBER], pidfd=True)
        host.install(monkeypatch)
        api = process_control._pidfd_api()
        monkeypatch.setattr(api, "exited", lambda _fd: None)
        held = []
        real = process_control._linux_row

        def row(pid):
            held.append(pid)
            return real(pid) if len(held) == 1 else (False, None)

        monkeypatch.setattr(process_control, "_linux_row", row)
        monkeypatch.setattr(process_control, "_linux_process_table", lambda: None)
        verdict = _kill(_MEMBER, start_time="cli", pgid=None)
        # Still signalled through its pidfd (it reaches the process or nothing).
        assert _signals(host)[0] == (f"pidfd {_MEMBER}", "SIGTERM")
        assert verdict == "unresolved"

    def test_a_member_whose_identity_cannot_be_read(self, monkeypatch):
        host = _Host({_MEMBER: (_GROUP, None)}, pidfd=True).install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=None,
                        members=[_cli(_MEMBER, "cli")])
        assert _outcome(host, verdict) == ("unresolved", [], [])

    def test_a_record_that_may_miss_a_process(self, monkeypatch):
        host = _Host({}, pidfd=True).install(monkeypatch)
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP, members_complete=False)
        assert _outcome(host, verdict) == ("unresolved", [], [])

    def test_off_linux_s_pidfds_an_incomplete_record_changes_nothing(self, monkeypatch):
        host = _Host({}, pidfd=False).install(monkeypatch)
        assert _kill(_WORKER, start_time="worker", pgid=_GROUP,
                     members_complete=False) == "gone"


class TestThePidfdGate:
    def test_calls_that_do_not_work_here_leave_the_kill_by_number(self, monkeypatch):
        def refused(_pid):
            raise OSError(38, "ENOSYS")

        monkeypatch.setattr(process_control, "_PIDFDS", [])
        monkeypatch.setattr(process_control.sys, "platform", "linux")
        monkeypatch.setattr(process_control.os, "pidfd_open", refused, raising=False)
        monkeypatch.setattr(process_control.signal, "pidfd_send_signal",
                            lambda *_a: None, raising=False)
        assert process_control._pidfd_api() is None

    def test_python_s_own_calls_come_first(self, monkeypatch):
        opened = []

        def open_call(pid):
            opened.append(pid)
            return os.open(os.devnull, os.O_RDONLY)

        monkeypatch.setattr(process_control, "_PIDFDS", [])
        monkeypatch.setattr(process_control.os, "pidfd_open", open_call, raising=False)
        monkeypatch.setattr(process_control.signal, "pidfd_send_signal",
                            lambda *_a: None, raising=False)
        monkeypatch.setattr(process_control, "_syscall_pidfds",
                            lambda: pytest.fail("the ctypes route was taken"))
        if sys.platform != "linux" or not os.path.exists("/proc/self/stat"):
            assert process_control._pidfd_api() is None
            return
        api = process_control._pidfd_api()
        assert api is not None and api._open is open_call
        assert opened == [os.getpid()]


# ---------------------------------------------------------------------------
# Round 2, point 2: every launch is recorded, or says it may not be
# ---------------------------------------------------------------------------


#: Only where the identity kill runs is anything recorded.
records_here = pytest.mark.skipif(
    process_control._pidfd_api() is None,
    reason="launches are recorded on Linux with pidfds and /proc only")


#: The run the launch-record tests record for.
_RUN = "r"


@pytest.fixture
def recording(tmp_path, monkeypatch):
    """A detached worker's record, named for this test; the path."""
    if process_control._pidfd_api() is None:
        pytest.skip("launches are recorded on Linux with pidfds and /proc only")
    path = tmp_path / process_control.cli_record_name(_RUN)
    monkeypatch.setattr(process_control, "_CLI_RECORD", [])
    process_control.record_cli_identities(str(tmp_path), _RUN)
    yield path
    process_control.record_cli_identities(None)


def _read(path, run_id=_RUN):
    """The record at *path*, read as run *run_id*'s."""
    return process_control.read_cli_record(str(path.parent), run_id)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestTheLaunchRecord:
    def test_a_process_in_a_session_of_its_own_is_recorded_with_its_group(
        self, recording
    ):
        proc, _job = process_control.popen_tree(
            ["sleep", "30"], **process_control.new_process_group_kwargs())
        try:
            record = _read(recording)
            start = process_control.process_start_time(proc.pid)
        finally:
            proc.kill()
            proc.wait()
        assert record.complete
        assert record.members == [process_control.Descendant(proc.pid, proc.pid, None, start)]
        launch, cli = _lines(recording)
        assert cli["cli"] == launch["launch"]

    def test_lane_a_s_cli_is_recorded(self, recording):
        from utils import llm_client

        done = llm_client._run_provider_cli([sys.executable, "-c", "pass"],
                                            capture_output=True, timeout=30)
        assert done.returncode == 0
        record = _read(recording)
        assert record.complete and len(record.members) == 1
        # In this process's group: a terminal Ctrl-C must still reach it.
        assert record.members[0].pgid == os.getpgid(0)

    def test_the_executor_s_cli_is_recorded(self, recording, tmp_path, runner_env):
        """The mock ``codex`` the executor's own tests run (no provider is
        reached), launched in a session of its own."""
        from tests.test_subagent_executor import _validation_batch
        from utils import subagent_executor

        runner_env.use("codex:gpt-5.5")
        subagent_executor.dispatch_task(
            _validation_batch(tmp_path), phase_dir=str(tmp_path),
            runner="codex:gpt-5.5", sleep=lambda _s: None)
        record = _read(recording)
        assert record.complete and record.members
        assert all(member.pgid == member.pid for member in record.members)

    def test_a_spawn_that_fails_says_so(self, recording):
        with pytest.raises(OSError):
            process_control.popen_tree(["/nonexistent/multi-llm-test-binary"])
        assert _read(recording) == (
            process_control.CliRecord([], True))
        assert [sorted(line) for line in _lines(recording)] == [
            ["launch", "run"], ["failed", "run"]]

    @records_here
    def test_nothing_starts_when_the_record_cannot_be_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(tmp_path / "missing-dir"), _RUN)
        spawned = []
        monkeypatch.setattr(process_control.subprocess, "Popen",
                            lambda *a, **k: spawned.append(a))
        try:
            with pytest.raises(OSError, match="nothing was started"):
                process_control.popen_tree(["sleep", "30"])
        finally:
            process_control.record_cli_identities(None)
        assert spawned == []

    def test_a_launch_whose_identity_was_never_written_is_incomplete(
        self, recording, monkeypatch
    ):
        """The worker died between the spawn and the identity line, or the
        line could not be written: the record says it may miss a process."""
        monkeypatch.setattr(process_control, "_record_spawned", lambda *_a: None)
        proc, _job = process_control.popen_tree(["sleep", "30"])
        proc.kill()
        proc.wait()
        assert _read(recording) == (
            process_control.CliRecord([], False))

    def test_a_short_write_tears_one_line_and_no_other(self, recording, monkeypatch):
        real_write = os.write
        writes = []

        def short_once(fd, data):
            writes.append(data)
            if len(writes) == 2:  # the identity line
                return real_write(fd, data[:7])
            return real_write(fd, data)

        monkeypatch.setattr(process_control.os, "write", short_once)
        proc, _job = process_control.popen_tree(["sleep", "30"])
        proc.kill()
        proc.wait()
        monkeypatch.setattr(process_control.os, "write", real_write)
        proc2, _job = process_control.popen_tree(["sleep", "30"])
        start2 = process_control.process_start_time(proc2.pid)
        proc2.kill()
        proc2.wait()
        record = _read(recording)
        assert not record.complete
        assert record.members == [process_control.Descendant(
            proc2.pid, process_control.process_group_id(os.getpid()), None, start2)]

    @pytest.mark.parametrize("text", [
        b'{"run":"r","launch":"1.1"}\n{"run":"r","cli":"1.1","pid":12,"start":"9"',  # torn
        b'{"run":"r","launch":"1.1"}\nnot json\n'
        b'{"run":"r","cli":"1.1","pid":12,"start":"9"}\n',
        b'{"run":"r","launch":"1.1"}\n{"run":"r","cli":"1.1","pid":12,"start":null}\n',
        b'{"run":"r","launch":"1.1"}\n{"run":"r","launch":"1.2"}\n'
        b'{"run":"r","cli":"1.1","pid":12,"start":"9"}\n',
        b'{"launch":"1.1"}\n{"cli":"1.1","pid":12,"start":"9"}\n',  # no run named
    ])
    def test_what_reads_as_incomplete(self, tmp_path, text):
        path = tmp_path / process_control.cli_record_name(_RUN)
        path.write_bytes(text)
        assert not _read(path).complete

    def test_no_file_is_nothing_recorded(self, tmp_path):
        assert process_control.read_cli_record(str(tmp_path), "none") == (
            process_control.CliRecord([], True))

    def test_nothing_is_recorded_where_the_identity_kill_does_not_run(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        monkeypatch.setattr(process_control, "_pidfd_api", lambda: None)
        process_control.record_cli_identities(str(tmp_path), _RUN)
        assert process_control._CLI_RECORD == []

    def test_recording_stops(self, recording):
        process_control.record_cli_identities(None)
        assert process_control._CLI_RECORD == []
        proc, _job = process_control.popen_tree(["true"])
        proc.wait()
        assert not recording.exists()


# runner_env, from the executor's own tests: mock provider CLIs on PATH.
from tests.test_subagent_executor import runner_env  # noqa: E402,F401


# ---------------------------------------------------------------------------
# The callers' verdicts and records: abort, close-out, sweep, finish
# ---------------------------------------------------------------------------


def _plan(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    return plan


def _detached_run(tmp_path, run_id="run-u77", *, state="running"):
    plan = _plan(tmp_path)
    phase = tmp_path / "plan" / "review-plan"
    phase.mkdir(parents=True)
    record = {
        "run_id": run_id, "pid": _WORKER, "process_start_time": "worker", "pgid": _GROUP,
        "detached": True, "mode": "review-plan", "plan_file": str(plan),
        "log_path": str(phase / "orchestrator-run.log"), "state": state,
    }
    status_file.claim_run(str(phase), record)
    root = run_registry.resolve_project_root(str(plan))
    run_registry.write_entry(root, {
        "run_id": run_id, "pid": _WORKER, "process_start_time": "worker", "pgid": _GROUP,
        "plan_file": str(plan), "phase_dir": str(phase), "mode": "review-plan",
    })
    return plan, phase, root


def _write_record(phase, run_id="run-u77", *, complete=True):
    path = Path(detach.cli_record_path(str(phase), run_id))
    lines = [{"run": run_id, "launch": "1.1"},
             {"run": run_id, "cli": "1.1", "pid": _MEMBER, "start": "member",
              "pgid": _GROUP}]
    if not complete:
        lines.append({"run": run_id, "launch": "1.2"})
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _kills(monkeypatch, *verdicts, during=None):
    """``kill_run_tree`` answering *verdicts* in turn; the calls it got.
    *during* runs inside each call, before it answers."""
    calls = []
    answers = list(verdicts)

    def kill_run_tree(run_id, pid, **kwargs):
        calls.append(dict(kwargs, run_id=run_id, pid=pid))
        if during is not None:
            during()
        return answers.pop(0) if len(answers) > 1 else answers[0]

    monkeypatch.setattr(process_control, "kill_run_tree", kill_run_tree)
    return calls


def _entry(root, run_id="run-u77"):
    return next((e for e in run_registry.list_entries(root) if e.get("run_id") == run_id),
                None)


def _entry_kept(root, run_id="run-u77"):
    return _entry(root, run_id) is not None


def _old(root, run_id="run-u77"):
    path = run_registry.entry_path(root, run_id)
    week_ago = time.time() - 8 * 86400
    os.utime(path, (week_ago, week_ago))


class TestAnAbortThatCouldNotConfirmTheTeardown:
    """``skipped`` and ``unresolved`` are not ``aborted``."""

    @pytest.mark.parametrize("verdict", ["skipped", "unresolved"])
    def test_is_reported_incomplete_and_keeps_its_records(self, tmp_path, monkeypatch, verdict):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        calls = _kills(monkeypatch, verdict)
        result = detach.abort_run(str(phase))
        assert result["result"] == "incomplete" and result["verdict"] == verdict
        assert calls[0]["members"] == [_cli(_MEMBER, "member", _GROUP)]
        run = status_file.read_run(str(phase))
        assert run["state"] == "aborted"
        assert run["cleanup"]["state"] == "incomplete"
        assert run["cleanup"]["verdict"] == verdict
        assert _entry(root)["cleanup"]["state"] == "incomplete"
        assert clis.exists()

    def test_status_abort_says_so(self, tmp_path, monkeypatch, capsys):
        plan, phase, root = _detached_run(tmp_path)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        _kills(monkeypatch, "skipped")
        code = run_status.main(["--plan-file", str(plan), "--abort"])
        out = capsys.readouterr().out
        assert code == 3, out
        assert "INCOMPLETE" in out and "could be confirmed stopped" in out
        assert "process group terminated" not in out

    def test_a_confirmed_teardown_is_still_aborted(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        _kills(monkeypatch, "killed")
        assert detach.abort_run(str(phase))["result"] == "aborted"
        assert status_file.read_run(str(phase))["cleanup"]["state"] == "complete"
        assert not _entry_kept(root) and not clis.exists()


class TestTheWorkerFinishingDuringAnAbort:
    """Round 2, point 7: the worker's SIGTERM handler runs ``finish`` while
    the abort's kill is still under way."""

    @staticmethod
    def _worker(phase, plan):
        return detach.RunHandle(str(phase), run_id="run-u77", mode="review-plan",
                                plan_file=str(plan))

    @pytest.mark.parametrize("verdict", ["unresolved", "survived", "skipped"])
    def test_an_unconfirmed_teardown_keeps_both_records(self, tmp_path, monkeypatch, verdict):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        worker = self._worker(phase, plan)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        seen = []

        def finish():
            seen.append(status_file.read_run(str(phase))["cleanup"]["state"])
            worker.finish(exit_code=3, state=detach.RUN_STATE_ABORTED, error="signal SIGTERM")

        _kills(monkeypatch, verdict, during=finish)
        result = detach.abort_run(str(phase))
        assert seen == ["pending"]
        assert result["result"] in ("incomplete", "survived")
        run = status_file.read_run(str(phase))
        assert run["state"] == "aborted" and run["cleanup"]["state"] == "incomplete"
        assert _entry(root)["cleanup"]["state"] == "incomplete"
        assert clis.exists()

    def test_a_confirmed_teardown_retires_them_after_the_kill(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        worker = self._worker(phase, plan)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        kept = []

        def finish():
            worker.finish(exit_code=3, state=detach.RUN_STATE_ABORTED, error="signal SIGTERM")
            kept.append((_entry_kept(root), clis.exists()))

        _kills(monkeypatch, "killed", during=finish)
        assert detach.abort_run(str(phase))["result"] == "aborted"
        assert kept == [(True, True)], "finish retired them while the kill was under way"
        assert status_file.read_run(str(phase))["cleanup"]["state"] == "complete"
        assert not _entry_kept(root) and not clis.exists()

    def test_an_aborter_that_died_mid_kill_leaves_it_pending_and_a_sweep_retries(
        self, tmp_path, monkeypatch, capsys
    ):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        worker = self._worker(phase, plan)
        status_file.update_run(str(phase), {"cleanup": detach._cleanup_entry(
            None, status_file.read_run(str(phase)), state=detach.CLEANUP_PENDING)},
            expect_run_id="run-u77")
        worker.finish(exit_code=3, state=detach.RUN_STATE_ABORTED, error="signal SIGTERM")
        assert _entry_kept(root) and clis.exists()
        run_status.main(["--plan-file", str(plan), "--view", "run"])
        assert "cleanup:        PENDING" in capsys.readouterr().out
        monkeypatch.setattr(run_registry, "verdict_for", lambda _e: process_control.DEAD)
        calls = _kills(monkeypatch, "killed")
        [row] = run_registry.sweep(root)
        assert row["action"] == "closed" and calls[0]["pid"] == _WORKER
        assert status_file.read_run(str(phase))["cleanup"]["state"] == "complete"
        assert not _entry_kept(root) and not clis.exists()

    def test_a_finish_with_no_teardown_under_way_retires_them(self, tmp_path):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        self._worker(phase, plan).finish(exit_code=0)
        assert not _entry_kept(root) and not clis.exists()


class TestACloseOutThatCouldNotConfirmTheTeardown:
    """The run is closed, but not as cleanly closed."""

    def test_the_record_says_so_and_both_records_are_kept(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        calls = _kills(monkeypatch, "unresolved")
        assert detach.close_out_dead_run(str(phase)) is True
        # The processes the worker recorded starting are what is signalled now.
        assert calls[0]["members"] == [_cli(_MEMBER, "member", _GROUP)]
        assert "members_complete" not in calls[0]
        run = status_file.read_run(str(phase))
        assert run["state"] == "failed"
        assert run["cleanup"]["state"] == "incomplete"
        assert _entry(root)["cleanup"]["state"] == "incomplete" and clis.exists()

    def test_a_record_that_may_miss_a_process_is_passed_on(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        _write_record(phase, complete=False)
        calls = _kills(monkeypatch, "unresolved")
        detach.close_out_dead_run(str(phase))
        assert calls[0]["members_complete"] is False

    def test_status_shows_it_and_an_abort_retries_it(self, tmp_path, monkeypatch, capsys):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        _kills(monkeypatch, "unresolved", "unresolved", "killed")
        detach.close_out_dead_run(str(phase))
        run_status.main(["--plan-file", str(plan), "--view", "run"])
        shown = capsys.readouterr().out
        assert "cleanup:        INCOMPLETE" in shown, shown
        # Still not confirmed: exit 3, nothing dropped.
        assert run_status.main(["--plan-file", str(plan), "--abort"]) == 3
        assert "INCOMPLETE" in capsys.readouterr().out
        assert _entry_kept(root)
        # Confirmed now: the record says so, and its trackers go.
        result = detach.abort_run(str(phase))
        assert result["result"] == "not_running"
        assert status_file.read_run(str(phase))["cleanup"]["state"] == "complete"
        assert not _entry_kept(root) and not clis.exists()

    def test_an_abort_of_a_worker_already_gone_reports_it(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.DEAD)
        _kills(monkeypatch, "unresolved")
        result = detach.abort_run(str(phase))
        assert result["result"] == "incomplete"
        assert result["verdict"] == "unresolved"

    def test_a_confirmed_close_out_drops_both_records(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        _kills(monkeypatch, "killed")
        assert detach.close_out_dead_run(str(phase)) is True
        assert "cleanup" not in status_file.read_run(str(phase))
        assert not _entry_kept(root) and not clis.exists()


class TestTheSweepTargetsOnlyTheRecordItValidated:
    """Round 2, point 4: a new run claims the phase dir right after the
    record was read."""

    @staticmethod
    def _claimed_after_the_read(monkeypatch, phase):
        real = status_file.read_run
        seen = []

        def read(path):
            run = real(path)
            if not seen:
                seen.append(True)
                status_file.claim_run(str(phase), dict(
                    run, run_id="run-new", pid=5000, pgid=5000,
                    process_start_time="new"))
            return run

        monkeypatch.setattr(status_file, "read_run", read)
        return real

    def test_in_the_close_out(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        real = self._claimed_after_the_read(monkeypatch, phase)
        calls = _kills(monkeypatch, "killed")
        detach.close_out_dead_run(str(phase), expect_run_id="run-u77")
        assert [(c["run_id"], c["pid"], c["start_time"]) for c in calls] == [
            ("run-u77", _WORKER, "worker")]
        new = real(str(phase))
        assert (new["run_id"], new["state"]) == ("run-new", "running")

    def test_in_the_sweep(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        monkeypatch.setattr(run_registry, "verdict_for", lambda _e: process_control.DEAD)
        real = self._claimed_after_the_read(monkeypatch, phase)
        calls = _kills(monkeypatch, "killed")
        run_registry.sweep(root)
        assert {(c["run_id"], c["pid"]) for c in calls} == {("run-u77", _WORKER)}
        new = real(str(phase))
        assert (new["run_id"], new["state"]) == ("run-new", "running")

    def test_a_close_out_held_to_another_run_signals_nothing(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        calls = _kills(monkeypatch, "killed")
        assert detach.close_out_dead_run(str(phase), expect_run_id="run-other") is False
        assert calls == []
        assert status_file.read_run(str(phase))["state"] == "running"


class TestASweepThatCouldNotConfirmTheTeardown:
    """``unresolved`` and ``skipped`` are not ``killed`` (round 2, point 5)."""

    @pytest.mark.parametrize("verdict", ["unresolved", "skipped", "survived"])
    def test_an_old_entry_is_kept_and_needs_attention(
        self, tmp_path, monkeypatch, capsys, verdict
    ):
        plan, phase, root = _detached_run(tmp_path)
        _old(root)
        monkeypatch.setattr(run_registry, "verdict_for", lambda _e: process_control.UNKNOWN)
        # The phase dir belongs to a newer run by now.
        status_file.claim_run(str(phase), {"run_id": "run-new", "state": "running"})
        _kills(monkeypatch, verdict)
        rows = run_registry.sweep(root)
        assert [(row["action"], row["kill_verdict"]) for row in rows] == [
            ("needs_attention", verdict)]
        assert _entry(root)["cleanup"]["state"] == "incomplete"
        assert run_registry.main(["--project-root", root, "--sweep"]) == 3
        assert "needs_attention" in capsys.readouterr().out
        assert status_file.read_run(str(phase))["run_id"] == "run-new"

    def test_the_next_sweep_retries_from_the_entry_not_the_dir(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        clis = _write_record(phase)
        monkeypatch.setattr(run_registry, "verdict_for", lambda _e: process_control.DEAD)
        _kills(monkeypatch, "unresolved")
        assert [row["action"] for row in run_registry.sweep(root)] == ["needs_attention"]
        # A new run claims the dir; the old entry's retry is the old run's.
        status_file.claim_run(str(phase), {
            "run_id": "run-next", "pid": 4200, "process_start_time": "next",
            "pgid": 4200, "state": "running"})
        calls = _kills(monkeypatch, "killed")
        assert [row["action"] for row in run_registry.sweep(root)] == ["closed"]
        assert [(c["run_id"], c["pid"], c["start_time"], c["pgid"]) for c in calls] == [
            ("run-u77", _WORKER, "worker", _GROUP)]
        assert calls[0]["members"] == [_cli(_MEMBER, "member", _GROUP)]
        assert status_file.read_run(str(phase))["state"] == "running"
        assert not _entry_kept(root) and not clis.exists()

    def test_a_dead_entry_whose_close_out_was_incomplete_needs_attention(
        self, tmp_path, monkeypatch
    ):
        plan, phase, root = _detached_run(tmp_path)
        monkeypatch.setattr(run_registry, "verdict_for", lambda _e: process_control.DEAD)
        _kills(monkeypatch, "unresolved")
        rows = run_registry.sweep(root)
        assert [row["action"] for row in rows] == ["needs_attention"]
        assert _entry_kept(root)

    def test_a_confirmed_kill_is_still_killed(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        _old(root)
        monkeypatch.setattr(run_registry, "verdict_for", lambda _e: process_control.ALIVE)
        _kills(monkeypatch, "killed")
        assert [row["action"] for row in run_registry.sweep(root)] == ["killed"]
        assert not _entry_kept(root)


# ---------------------------------------------------------------------------
# Round 3, point 1: no identity, no signal
# ---------------------------------------------------------------------------

#: A provider CLI the worker starts while an abort is under way.
_LATE = 4105
#: A runner the executor started, its own group's leader.
_RUNNER = 4106


class TestAnUndatedTargetIsNeverSignalled:
    """A pid with no recorded start time names whoever has it now. By
    identity it is held only through an owned child that anchors it; else
    it gets nothing, and only what is identified is signalled."""

    def test_a_worker_whose_pid_was_handed_out_again(self, monkeypatch):
        host = _Host({_WORKER: (_GROUP, "stranger")}, deaf=[_WORKER], pidfd=True)
        host.install(monkeypatch)
        host.bystanders.add(_WORKER)
        verdict = _kill(_WORKER, start_time=None, pgid=_GROUP)
        assert _outcome(host, verdict) == ("unresolved", [], [])
        assert host.open_pidfds == set()

    def test_its_recorded_members_are_still_stopped(self, monkeypatch):
        host = _Host({_WORKER: (_GROUP, "stranger"), _MEMBER: (_GROUP, "member")},
                     pidfd=True).install(monkeypatch)
        host.bystanders.add(_WORKER)
        verdict = _kill(_WORKER, start_time=None, pgid=_GROUP,
                        members=[_cli(_MEMBER, "member", _GROUP)])
        assert _outcome(host, verdict) == (
            "unresolved", [(f"pidfd {_MEMBER}", "SIGTERM")], [])
        assert _MEMBER not in host.procs

    def test_a_pid_no_process_has_is_gone(self, monkeypatch):
        host = _Host({}, pidfd=True).install(monkeypatch)
        assert _outcome(host, _kill(_WORKER, start_time=None, pgid=_GROUP)) == (
            "gone", [], [])

    def test_an_owned_child_is_held_through_its_anchor(self, monkeypatch):
        """In a group it does not lead, so the owned kill does not take it:
        the identity kill holds it through its ``Popen``, an unreaped child."""
        host = _Host({_WORKER: (_ELSEWHERE, "child")}, owned=[_WORKER], pidfd=True)
        host.install(monkeypatch)
        popen = types.SimpleNamespace(pid=_WORKER, returncode=None)
        verdict = _kill(_WORKER, start_time=None, pgid=None, child=popen)
        assert _outcome(host, verdict) == ("killed", [(f"pidfd {_WORKER}", "SIGTERM")], [])

    def test_a_child_reaped_behind_its_popen_is_not_held(self, monkeypatch):
        """Its pid is no child of this process any more: another's now."""
        host = _Host({_WORKER: (_ELSEWHERE, "stranger")}, pidfd=True).install(monkeypatch)
        host.bystanders.add(_WORKER)
        popen = types.SimpleNamespace(pid=_WORKER, returncode=None)
        verdict = _kill(_WORKER, start_time=None, pgid=None, child=popen)
        assert _outcome(host, verdict) == ("unresolved", [], [])


class TestAForcedAbortKeepsTheRecordedStartTime:
    """``--force`` overrides a probe that could not read the worker; by
    identity the kill still goes by the start time the run recorded."""

    def test_the_kill_gets_it_by_identity_and_none_by_number(
        self, tmp_path, monkeypatch, pidfd
    ):
        plan, phase, root = _detached_run(tmp_path)
        _Host({_WORKER: (_GROUP, "worker")}, pidfd=pidfd).install(monkeypatch)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.UNKNOWN)
        calls = _kills(monkeypatch, "killed")
        assert detach.abort_run(str(phase), force=True)["result"] == "aborted"
        # By number, the unverified kill `--force` always requested.
        assert calls[0]["start_time"] == ("worker" if pidfd else None)

    def test_a_worker_replaced_before_the_kill_gets_nothing(self, tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        host = _Host({_WORKER: (_GROUP, "worker")}, deaf=[_WORKER], pidfd=True)
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.UNKNOWN)
        real = process_control.kill_run_tree

        def replaced(*args, **kwargs):
            host.exit(_WORKER)()
            host.spawn(_WORKER, _GROUP, "stranger")()
            return real(*args, **kwargs)

        monkeypatch.setattr(process_control, "kill_run_tree", replaced)
        result = detach.abort_run(str(phase), force=True)
        assert result["result"] == "incomplete"
        assert host.sent == [] and host.hit_bystanders() == []
        assert _entry_kept(root)
        assert status_file.read_run(str(phase))["cleanup"]["state"] == "incomplete"


class TestAFailedHandshakeKillsByTheStartTimeReadAtTheSpawn:
    """The launcher's teardown of a worker that never published: by the start
    time read while the intermediate spawn still held the worker's session."""

    def test_the_spawn_reads_it_only_from_the_intermediates_session(self, monkeypatch):
        intermediate = 4100
        rows = {}
        anchored = {intermediate}
        monkeypatch.setattr(process_control, "signals_by_identity", lambda: True)
        monkeypatch.setattr(process_control, "_owned_anchor", lambda pid: pid in anchored)
        monkeypatch.setattr(process_control, "_linux_row", lambda pid: (True, rows.get(pid)))
        worker = process_control._ProcRow(_WORKER, 1, intermediate, intermediate, "worker", "S")
        rows[_WORKER] = worker
        process_control._note_worker_start(str(_WORKER).encode(), intermediate)
        assert process_control.detached_worker_start_time(_WORKER) == "worker"
        # Another session's process has the pid: it was handed out again.
        rows[_WORKER] = process_control._ProcRow(_WORKER, 1, _SESSION, _SESSION, "x", "S")
        process_control._note_worker_start(str(_WORKER).encode(), intermediate)
        assert process_control.detached_worker_start_time(_WORKER) is None
        # The intermediate already reaped: nothing holds the session's number.
        rows[_WORKER] = worker
        anchored.clear()
        process_control._note_worker_start(str(_WORKER).encode(), intermediate)
        assert process_control.detached_worker_start_time(_WORKER) is None

    @staticmethod
    def _failed_launch(tmp_path, monkeypatch, start):
        from utils import cli_common

        plan = _plan(tmp_path)
        phase = tmp_path / "plan" / "review-plan"
        phase.mkdir(parents=True)
        uv = tmp_path / "fake-uv"
        uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        uv.chmod(0o755)
        monkeypatch.setenv(detach.UV_ENV, str(uv))
        monkeypatch.setenv(detach.HANDSHAKE_TIMEOUT_ENV, "0.1")
        monkeypatch.setattr(detach, "missing_provider_binaries", lambda *a, **k: {})
        monkeypatch.setattr(detach, "missing_runner_binaries", lambda *a, **k: {})

        def spawn(command, *, cwd, env, log_path):
            process_control._DETACHED_START.clear()
            if start is not None:
                process_control._DETACHED_START[_WORKER] = start
            return _WORKER

        monkeypatch.setattr(process_control, "spawn_detached", spawn)
        seen = []
        real = process_control.kill_tree

        def kill_tree(pid, **kwargs):
            seen.append(dict(kwargs))
            return real(pid, **kwargs)

        monkeypatch.setattr(process_control, "kill_tree", kill_tree)
        cli_common.activate_result(cli_common.ResultContext(mode="review-plan"))
        try:
            with pytest.raises(detach.DetachError, match="never published") as caught:
                detach.launch(
                    mode="review-plan",
                    argv=["--plan-file", str(plan), "--detach", "--yes"],
                    phase_dir=str(phase), plan_file=str(plan), phase="review-plan",
                    args=types.SimpleNamespace(force=False, if_not_running=False),
                )
        finally:
            cli_common.deactivate_result()
        return caught.value, seen

    @pytest.mark.parametrize("start, now, verdict, signals", [
        # No start time read at the spawn: the pid alone is not signalled,
        # even while it is still the worker's.
        (None, "worker", "unresolved", []),
        ("worker", "worker", "killed", [(f"pidfd {_WORKER}", "SIGTERM")]),
        # The worker is gone and a stranger has its pid: nothing reaches it.
        ("worker", "stranger", "gone", []),
    ])
    def test_the_launch_kills_by_it_and_says_what_it_could_not_confirm(
        self, tmp_path, monkeypatch, start, now, verdict, signals
    ):
        host = _Host({_WORKER: (_GROUP, now)}, pidfd=True).install(monkeypatch)
        if now != "worker":
            host.bystanders.add(_WORKER)
        error, seen = self._failed_launch(tmp_path, monkeypatch, start)
        assert seen == [dict({"escalate_after_s": 1.0},
                             **({"start_time": start} if start else {}))]
        assert (_signals(host), host.hit_bystanders()) == (signals, [])
        assert error.exit_code == 3
        if verdict == "unresolved":
            assert f"Not confirmed stopped (kill verdict {verdict})" in str(error)
            assert f"Check pid {_WORKER}" in str(error)
        else:
            assert "the worker was terminated" in str(error)

    @records_here
    def test_a_real_detached_worker(self, tmp_path):
        pid = process_control.spawn_detached(
            ["sleep", "30"], cwd=str(tmp_path), env=dict(os.environ),
            log_path=str(tmp_path / "worker.log"))
        try:
            start = process_control.detached_worker_start_time(pid)
            assert start is not None and start == process_control.process_start_time(pid)
            assert process_control.kill_tree(pid, start_time=start, escalate_after_s=0.5,
                                             poll_interval_s=0.05) == "killed"
            assert _gone(pid)
        finally:
            _kill_quietly(pid)


# ---------------------------------------------------------------------------
# Round 3, point 2: a record is its run's alone
# ---------------------------------------------------------------------------


def _two_runs(tmp_path):
    """Run A's phase dir, A's record path (not written), and run B's record
    naming B's live CLI."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pb = Path(detach.cli_record_path(str(b), "B"))
    pb.write_text(
        json.dumps({"run": "B", "launch": "1.1"}) + "\n"
        + json.dumps({"run": "B", "cli": "1.1", "pid": _BYSTANDER, "start": "B-cli",
                      "pgid": _BYSTANDER}) + "\n", encoding="utf-8")
    return a, Path(detach.cli_record_path(str(a), "A")), pb


class TestARecordIsItsRunsAlone:
    @records_here
    @pytest.mark.parametrize("link", ["symlink", "hard link"])
    def test_a_link_to_another_runs_record_is_neither_written_nor_read(
        self, tmp_path, monkeypatch, link
    ):
        a, pa, pb = _two_runs(tmp_path)
        before = pb.read_bytes()
        if link == "symlink":
            pa.symlink_to(pb)
        else:
            os.link(pb, pa)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(a), "A")
        spawned = []
        monkeypatch.setattr(process_control.subprocess, "Popen",
                            lambda *args, **kwargs: spawned.append(args))
        try:
            with pytest.raises(process_control.LaunchRefused, match="nothing was started"):
                process_control.popen_tree(["sleep", "30"])
        finally:
            process_control.record_cli_identities(None)
        assert spawned == [] and pb.read_bytes() == before
        assert process_control.read_cli_record(str(a), "A") == (
            process_control.CliRecord([], False))
        if link == "hard link":
            # Two names for one file: neither run's record any more.
            assert not process_control.read_cli_record(str(pb.parent), "B").complete
        # Run A's kill reaches nothing of run B's.
        host = _Host({_BYSTANDER: (_BYSTANDER, "B-cli")}, pidfd=True).install(monkeypatch)
        host.bystanders.add(_BYSTANDER)
        verdict = detach.kill_recorded_tree(
            str(a), {"run_id": "A", "pid": _WORKER, "pgid": _GROUP}, start_time="A-worker",
            escalate_after_s=0.3)
        assert _outcome(host, verdict) == ("unresolved", [], [])

    def test_a_phase_dir_that_is_a_link_is_refused(self, tmp_path):
        a, pa, pb = _two_runs(tmp_path)
        linked = tmp_path / "linked"
        linked.symlink_to(pb.parent)
        assert process_control.read_cli_record(str(linked), "B") == (
            process_control.CliRecord([], False))
        assert process_control.read_cli_record(str(pb.parent), "B").members == [
            _cli(_BYSTANDER, "B-cli", _BYSTANDER)]

    def test_a_record_another_user_owns_is_refused(self, tmp_path, monkeypatch):
        a, pa, pb = _two_runs(tmp_path)
        monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)
        assert process_control.read_cli_record(str(pb.parent), "B") == (
            process_control.CliRecord([], False))

    def test_another_runs_lines_are_ignored_and_reported(self, tmp_path, monkeypatch, capsys):
        a, pa, pb = _two_runs(tmp_path)
        pa.write_text("".join(json.dumps(line) + "\n" for line in [
            {"run": "A", "launch": "1.1"},
            {"run": "A", "cli": "1.1", "pid": _MEMBER, "start": "member", "pgid": _GROUP},
            {"run": "B", "cli": "9.9", "pid": _BYSTANDER, "start": "B-cli",
             "pgid": _BYSTANDER},
        ]), encoding="utf-8")
        assert process_control.read_cli_record(str(a), "A") == process_control.CliRecord(
            [_cli(_MEMBER, "member", _GROUP)], True, foreign=1)
        calls = _kills(monkeypatch, "killed")
        detach.kill_recorded_tree(str(a), {"run_id": "A", "pid": _WORKER, "pgid": _GROUP},
                                  start_time="worker")
        assert calls[0]["members"] == [_cli(_MEMBER, "member", _GROUP)]
        assert "1 line(s) recorded for another run were ignored" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Round 3, point 3: a launch racing an abort
# ---------------------------------------------------------------------------


class TestALaunchRacingAnAbort:
    """The abort closes the run's launch gate before its kill, under the
    record's lock: a launch is either in the abort's read or refused."""

    @staticmethod
    def _worker_recording(tmp_path, monkeypatch, **host_kwargs):
        plan, phase, root = _detached_run(tmp_path)
        host = _Host({_WORKER: (_GROUP, "worker")}, pidfd=True, **host_kwargs)
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(phase), "run-u77")
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        return phase, root, host

    def test_a_launch_after_the_gate_closed_starts_nothing(self, tmp_path, monkeypatch):
        phase, root, host = self._worker_recording(tmp_path, monkeypatch)
        spawned, refused = [], []
        real = process_control.kill_run_tree

        def kill(*args, **kwargs):
            # The worker, not yet stopped, starts a provider CLI.
            with pytest.MonkeyPatch.context() as spawn:
                spawn.setattr(process_control.subprocess, "Popen",
                              lambda *a, **k: spawned.append(a))
                try:
                    process_control.popen_tree(["codex", "exec", "-"])
                except process_control.LaunchRefused as exc:
                    refused.append(str(exc))
            return real(*args, **kwargs)

        monkeypatch.setattr(process_control, "kill_run_tree", kill)
        try:
            result = detach.abort_run(str(phase))
        finally:
            process_control.record_cli_identities(None)
        assert spawned == []
        assert refused and "being aborted" in refused[0] and "nothing was started" in refused[0]
        assert result["result"] == "aborted"
        assert _signals(host) == [(f"pidfd {_WORKER}", "SIGTERM")]
        assert not _entry_kept(root)

    def test_a_launch_past_the_gate_whose_cli_came_up_during_the_kill_is_stopped(
        self, tmp_path, monkeypatch
    ):
        phase, root, host = self._worker_recording(tmp_path, monkeypatch)
        token = process_control._record_launch()  # its intent got in first
        real = process_control.kill_run_tree
        members = []

        def kill(run_id, pid, **kwargs):
            members.append(kwargs.get("members", []))
            if len(members) == 1:
                # Its CLI comes up, in a session of its own, after the first
                # read: the worker's child (this process plays the worker).
                host.owned.add(_LATE)
                host.fork(_LATE, _LATE)()
                process_control._record_spawned(token, types.SimpleNamespace(pid=_LATE))
            return real(run_id, pid, **kwargs)

        monkeypatch.setattr(process_control, "kill_run_tree", kill)
        try:
            result = detach.abort_run(str(phase))
        finally:
            process_control.record_cli_identities(None)
        assert members == [[], [_cli(_LATE, f"{_LATE}-start", _LATE)]]
        assert result["result"] == "aborted", result
        assert (f"pidfd {_LATE}", "SIGTERM") in _signals(host)
        assert _LATE in host.zombies and host.hit_bystanders() == []
        assert not _entry_kept(root)
        assert not Path(detach.cli_record_path(str(phase), "run-u77")).exists()

    def test_a_launch_whose_identity_never_came_leaves_it_incomplete(
        self, tmp_path, monkeypatch
    ):
        """The worker was stopped between its intent and its identity line:
        it may have started a CLI that nothing names."""
        phase, root, host = self._worker_recording(tmp_path, monkeypatch)
        process_control._record_launch()
        try:
            result = detach.abort_run(str(phase))
        finally:
            process_control.record_cli_identities(None)
        assert result["result"] == "incomplete" and result["verdict"] == "unresolved"
        assert _entry_kept(root)
        assert Path(detach.cli_record_path(str(phase), "run-u77")).exists()

    def test_a_gate_that_cannot_be_closed_leaves_it_incomplete(self, tmp_path, monkeypatch):
        phase, root, host = self._worker_recording(tmp_path, monkeypatch)
        process_control.record_cli_identities(None)
        elsewhere = tmp_path / "elsewhere.clis"
        Path(detach.cli_record_path(str(phase), "run-u77")).symlink_to(elsewhere)
        result = detach.abort_run(str(phase))
        assert result["result"] == "incomplete"
        assert not elsewhere.exists()
        assert _entry_kept(root)


# ---------------------------------------------------------------------------
# Round 3, point 4: status --sweep says what it could not confirm
# ---------------------------------------------------------------------------


class TestStatusSweepSaysWhatItCouldNotConfirm:
    def test_an_unconfirmed_close_out_exits_3_with_the_way_to_retry(
        self, tmp_path, monkeypatch, capsys
    ):
        plan, phase, root = _detached_run(tmp_path)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.DEAD)
        _kills(monkeypatch, "unresolved")
        code = run_status.main(["--plan-file", str(plan), "--sweep"])
        out = capsys.readouterr().out
        assert code == 3, out
        assert "review-plan: running -> failed" in out
        assert "INCOMPLETE: review-plan run run-u77" in out
        assert "kill verdict unresolved" in out and "process group 4101" in out
        assert "--abort" in out and "run-u77" in out

    def test_its_result_line_needs_a_human(self, tmp_path, monkeypatch, capsys):
        import multi_llm

        plan, phase, root = _detached_run(tmp_path)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.DEAD)
        _kills(monkeypatch, "unresolved")
        assert multi_llm.main(["status", str(plan), "--sweep"]) == 3
        [line] = [l for l in capsys.readouterr().out.splitlines()
                  if l.startswith("MULTI_LLM_RESULT: ")]
        result = json.loads(line.split(": ", 1)[1])
        assert (result["state"], result["exit"]) == ("needs-human", 3)

    def test_a_run_closed_earlier_with_its_cleanup_unconfirmed_is_reported(
        self, tmp_path, monkeypatch, capsys
    ):
        plan, phase, root = _detached_run(tmp_path, state="failed")
        status_file.update_run(str(phase), {"state": "failed", "cleanup": {
            "state": "incomplete", "verdict": "skipped", "pid": _WORKER, "pgid": _GROUP,
            "checked_at": status_file.now_iso()}}, expect_run_id="run-u77")
        calls = _kills(monkeypatch, "killed")
        assert run_status.main(["--plan-file", str(plan), "--sweep"]) == 3
        out = capsys.readouterr().out
        assert "swept: nothing to close out" in out and "kill verdict skipped" in out
        assert calls == []

    def test_a_confirmed_close_out_is_still_exit_0(self, tmp_path, monkeypatch, capsys):
        plan, phase, root = _detached_run(tmp_path)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.DEAD)
        _kills(monkeypatch, "killed")
        assert run_status.main(["--plan-file", str(plan), "--sweep"]) == 0
        assert "INCOMPLETE" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Round 3, point 5: without pidfds the signals are the old ones, the verdicts
# honest
# ---------------------------------------------------------------------------


class TestWithoutPidfdsTheVerdictIsStillHonest:
    def test_an_abort_whose_teardown_cannot_read_the_worker_is_incomplete(
        self, tmp_path, monkeypatch
    ):
        """The first probe reads the worker; the kill's own probe cannot. The
        kill by number then sends nothing (``skipped``), as it always did; the
        abort says so, where it used to say ``aborted`` and drop its records."""
        plan, phase, root = _detached_run(tmp_path)
        host = _Host({_WORKER: (_GROUP, "worker")}, pidfd=False).install(monkeypatch)
        answers = [process_control.ALIVE]
        monkeypatch.setattr(process_control, "probe",
                            lambda *_a: answers.pop(0) if answers else process_control.UNKNOWN)
        result = detach.abort_run(str(phase))
        assert (result["result"], result["verdict"]) == ("incomplete", "skipped")
        assert host.sent == []
        assert status_file.read_run(str(phase))["cleanup"]["state"] == "incomplete"
        assert _entry(root)["cleanup"]["verdict"] == "skipped"
        # Nothing is recorded where the kill goes by number.
        assert not Path(detach.cli_record_path(str(phase), "run-u77")).exists()


# ---------------------------------------------------------------------------
# Round 4, point 1: a launch's identity is kept only for this process's child
# ---------------------------------------------------------------------------


class TestARecordedIdentityIsThisProcesssChild:
    """``popen_tree`` returns before anything waits on its child, so the pid
    it reports can name another process only once that child was reaped
    behind its ``Popen``: SIGCHLD ignored, the CLI exits at once, and its pid
    is handed to a bystander before the identity is read."""

    @staticmethod
    def _launch(phase, host, monkeypatch, *, owned, recycled):
        def spawn(_args, _kwargs):
            if owned:
                host.owned.add(_LATE)
            host.fork(_LATE, _LATE)()
            if recycled:
                host.exit(_LATE)()  # reaped by the kernel (SIGCHLD ignored)
                host.spawn(_LATE, _LATE, "bystander")()
            return types.SimpleNamespace(pid=_LATE, returncode=None), None

        monkeypatch.setattr(process_control, "_popen_tree", spawn)
        monkeypatch.setattr(process_control, "_sigchld_action",
                            lambda: (int(signal.SIG_IGN), 0))
        process_control.popen_tree(["codex", "exec", "-"])
        return process_control.read_cli_record(str(phase), "run-u77")

    @staticmethod
    def _worker_recording(tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        host = _Host({_WORKER: (_GROUP, "worker")}, pidfd=True).install(monkeypatch)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(phase), "run-u77")
        return phase, root, host

    def test_a_pid_handed_out_again_is_written_unverified_and_never_signalled(
        self, tmp_path, monkeypatch
    ):
        phase, root, host = self._worker_recording(tmp_path, monkeypatch)
        try:
            record = self._launch(phase, host, monkeypatch, owned=False, recycled=True)
            assert (record.members, record.complete) == ([], False)
            line = _lines(Path(detach.cli_record_path(str(phase), "run-u77")))[-1]
            assert (line["pid"], line["start"], line["unverified"]) == (_LATE, None, True)
            monkeypatch.setattr(process_control, "probe",
                                lambda *_a: process_control.ALIVE)
            result = detach.abort_run(str(phase))
        finally:
            process_control.record_cli_identities(None)
        assert (result["result"], result["verdict"]) == ("incomplete", "unresolved")
        assert host.hit_bystanders() == []
        assert _LATE in host.procs  # the bystander runs on, untouched
        assert _entry_kept(root)

    def test_this_process_s_child_is_recorded_by_its_own_row(self, tmp_path, monkeypatch):
        phase, root, host = self._worker_recording(tmp_path, monkeypatch)
        try:
            record = self._launch(phase, host, monkeypatch, owned=True, recycled=False)
        finally:
            process_control.record_cli_identities(None)
        assert record.complete
        assert record.members == [_cli(_LATE, f"{_LATE}-start", _LATE)]

    def test_a_row_that_cannot_be_read_is_written_unverified(self, tmp_path, monkeypatch):
        phase, root, host = self._worker_recording(tmp_path, monkeypatch)
        monkeypatch.setattr(process_control, "_linux_row", lambda _pid: (False, None))
        try:
            record = self._launch(phase, host, monkeypatch, owned=True, recycled=False)
        finally:
            process_control.record_cli_identities(None)
        assert (record.members, record.complete) == ([], False)


# ---------------------------------------------------------------------------
# Round 4, point 2: native pidfds do not depend on the sigaction ABI
# ---------------------------------------------------------------------------


on_linux_with_proc = pytest.mark.skipif(
    sys.platform != "linux" or not os.path.exists("/proc/self/stat"),
    reason="the identity kill runs on Linux with /proc only")


class TestNativePidfdsNeedNoSigactionAbi:
    """The ABI names ``struct sigaction``'s layout, which only the anchor of
    an undated owned child reads; Python's own pidfd calls do not need it."""

    @staticmethod
    def _no_abi(monkeypatch, *, native):
        monkeypatch.setattr(process_control, "_PIDFDS", [])
        monkeypatch.setattr(process_control, "_SIGACTION_ABI", [None])
        if native:
            monkeypatch.setattr(process_control.os, "pidfd_open",
                                lambda _pid: os.open(os.devnull, os.O_RDONLY),
                                raising=False)
            monkeypatch.setattr(process_control.signal, "pidfd_send_signal",
                                lambda *_a: None, raising=False)
        else:
            monkeypatch.delattr(process_control.os, "pidfd_open", raising=False)
            monkeypatch.delattr(process_control.signal, "pidfd_send_signal",
                                raising=False)

    @on_linux_with_proc
    def test_python_s_own_calls_run_the_identity_kill_in_any_abi(self, monkeypatch):
        self._no_abi(monkeypatch, native=True)
        assert process_control._pidfd_api() is not None
        assert process_control.signals_by_identity()

    def test_the_syscall_route_needs_the_abi_its_numbers_belong_to(self, monkeypatch):
        self._no_abi(monkeypatch, native=False)
        assert process_control._syscall_pidfds() is None
        assert process_control._pidfd_api() is None

    @on_linux_with_proc
    def test_a_dated_worker_s_group_made_again_gets_no_signal(self, monkeypatch):
        """Before the SIGKILL the worker is reaped and an unrelated program
        makes the group's number again: by number, it got the SIGKILL."""
        factory = process_control._pidfd_api
        host = _Host({_WORKER: (_GROUP, "worker")}, deaf=[_WORKER], pidfd=True)
        host.install(monkeypatch)
        self._no_abi(monkeypatch, native=True)
        monkeypatch.setattr(process_control, "_pidfd_api", factory)
        monkeypatch.setattr(process_control.os, "pidfd_open", host.pidfd_open,
                            raising=False)
        monkeypatch.setattr(process_control.signal, "pidfd_send_signal",
                            host.pidfd_send_signal, raising=False)
        monkeypatch.setattr(process_control, "_working_pidfds", lambda _api: True)

        def replace():
            host.procs.clear()
            host.spawn(_GROUP, _GROUP, "unrelated")()

        host.before_sigkill = [replace]
        verdict = _kill(_WORKER, start_time="worker", pgid=_GROUP)
        assert process_control.signals_by_identity()
        assert host.hit_bystanders() == [], _outcome(host, verdict)
        assert not [how for how, _sig in _signals(host) if how.startswith("group")]

    def test_an_undated_owned_child_there_is_unresolved_and_gets_nothing(
        self, monkeypatch
    ):
        """Its anchor reads SIGCHLD's disposition, which needs the ABI."""
        host = _Host({_RUNNER: (_RUNNER, "leader")}, owned=[_RUNNER], pidfd=True)
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_SIGACTION_ABI", [None])
        monkeypatch.setattr(process_control, "_sigchld_action", lambda: None)
        child = types.SimpleNamespace(pid=_RUNNER, returncode=None)
        verdict = _kill(_RUNNER, start_time=None, pgid=_RUNNER, child=child)
        assert (verdict, host.sent) == ("unresolved", [])


# ---------------------------------------------------------------------------
# Round 4, point 3: the worker refuses launches once its teardown began
# ---------------------------------------------------------------------------


class TestTheWorkerRefusesLaunchesOnceItsTeardownBegan:
    @staticmethod
    def _worker(tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path)
        host = _Host({_WORKER: (_GROUP, "worker")}, pidfd=True).install(monkeypatch)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        monkeypatch.setattr(process_control, "_LAUNCHES_REFUSED", [], raising=False)
        process_control.record_cli_identities(str(phase), "run-u77")
        handle = detach.RunHandle(str(phase), run_id="run-u77", mode="review-plan",
                                  plan_file=str(plan))
        spawned = []

        def spawn(_args, _kwargs):
            spawned.append(_LATE)
            host.owned.add(_LATE)
            host.fork(_LATE, _LATE)()
            return types.SimpleNamespace(pid=_LATE, returncode=None), None

        monkeypatch.setattr(process_control, "_popen_tree", spawn)
        return phase, root, host, handle, spawned

    def test_a_launch_between_the_handler_s_finish_and_its_exit_starts_nothing(
        self, tmp_path, monkeypatch
    ):
        """The abort's SIGTERM reached the worker: its handler finished the
        run; an executor thread launches before ``os._exit``."""
        phase, root, host, handle, spawned = self._worker(tmp_path, monkeypatch)
        monkeypatch.setattr(process_control, "probe", lambda *_a: process_control.ALIVE)
        real = process_control.kill_run_tree
        refused = []

        def kill(*args, **kwargs):
            handle.finish(3, state=detach.RUN_STATE_ABORTED, error="signal SIGTERM")
            try:
                process_control.popen_tree(["codex", "exec", "-"])
            except process_control.LaunchRefused as exc:
                refused.append(str(exc))
            return real(*args, **kwargs)

        monkeypatch.setattr(process_control, "kill_run_tree", kill)
        try:
            result = detach.abort_run(str(phase))
        finally:
            process_control.record_cli_identities(None)
        assert spawned == [] and refused and "nothing was started" in refused[0]
        assert result["result"] == "aborted", result
        assert _LATE not in host.procs and not _entry_kept(root)

    def test_a_launch_after_a_finish_starts_nothing_and_writes_no_record(
        self, tmp_path, monkeypatch
    ):
        phase, root, host, handle, spawned = self._worker(tmp_path, monkeypatch)
        record = Path(detach.cli_record_path(str(phase), "run-u77"))
        try:
            handle.finish(0)
            with pytest.raises(process_control.LaunchRefused, match="ending"):
                process_control.popen_tree(["codex", "exec", "-"])
        finally:
            process_control.record_cli_identities(None)
        assert spawned == [] and not record.exists()

    def test_a_launch_already_past_its_intent_still_records_its_identity(
        self, tmp_path, monkeypatch
    ):
        """An abort is under way (its kill pending): the worker's finish keeps
        the record, and a CLI whose intent got in first is still named there."""
        phase, root, host, handle, spawned = self._worker(tmp_path, monkeypatch)
        token = process_control._record_launch()
        status_file.update_run(str(phase), {"cleanup": {"state": "pending"}},
                               expect_run_id="run-u77")
        try:
            handle.finish(3, state=detach.RUN_STATE_ABORTED)
            host.owned.add(_LATE)
            host.fork(_LATE, _LATE)()
            process_control._record_spawned(token, types.SimpleNamespace(pid=_LATE))
            record = process_control.read_cli_record(str(phase), "run-u77")
        finally:
            process_control.record_cli_identities(None)
        assert record.complete and record.members == [_cli(_LATE, f"{_LATE}-start", _LATE)]


# ---------------------------------------------------------------------------
# Round 4, point 4: every cleanup runs the whole teardown, a retry's too
# ---------------------------------------------------------------------------


class TestARetryRunsTheWholeTeardown:
    """A first abort could not stop the worker (``skipped``): the run reads
    ``aborted`` with its cleanup incomplete, while its worker still runs and
    may launch. The retry closes the gate, kills, and reads the record again."""

    @staticmethod
    def _left_running(tmp_path, monkeypatch):
        plan, phase, root = _detached_run(tmp_path, state="aborted")
        status_file.update_run(str(phase), {"state": "aborted", "cleanup": {
            "state": "incomplete", "verdict": "skipped", "pid": _WORKER,
            "pgid": _GROUP, "checked_at": status_file.now_iso()}},
            expect_run_id="run-u77")
        run_registry.mark_cleanup(root, "run-u77", {"state": "incomplete",
                                                    "verdict": "skipped"},
                                  status_file.read_run(str(phase)))
        host = _Host({_WORKER: (_GROUP, "worker")}, pidfd=True).install(monkeypatch)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(phase), "run-u77")
        return phase, root, host

    @staticmethod
    def _cli_comes_up(host, token):
        host.owned.add(_LATE)
        host.fork(_LATE, _LATE)()
        process_control._record_spawned(token, types.SimpleNamespace(pid=_LATE))

    def test_a_launch_during_the_retry_starts_nothing(self, tmp_path, monkeypatch):
        phase, root, host = self._left_running(tmp_path, monkeypatch)
        real = process_control.kill_run_tree
        spawned = []

        def kill(*args, **kwargs):
            # The still-running worker launches while the retry kills.
            try:
                token = process_control._record_launch()
            except process_control.LaunchRefused:
                pass
            else:
                spawned.append(token)
                self._cli_comes_up(host, token)
            return real(*args, **kwargs)

        monkeypatch.setattr(process_control, "kill_run_tree", kill)
        try:
            result = detach.abort_run(str(phase))
        finally:
            process_control.record_cli_identities(None)
        assert spawned == []
        assert result["result"] == "not_running", result
        assert "now confirmed stopped" in result["detail"]
        assert _LATE not in host.procs and host.hit_bystanders() == []
        assert not _entry_kept(root)

    def test_a_cli_whose_intent_got_in_first_is_stopped_by_the_reread(
        self, tmp_path, monkeypatch
    ):
        phase, root, host = self._left_running(tmp_path, monkeypatch)
        token = process_control._record_launch()
        real = process_control.kill_run_tree
        calls = []

        def kill(*args, **kwargs):
            calls.append(kwargs.get("members", []))
            if len(calls) == 1:
                self._cli_comes_up(host, token)
            return real(*args, **kwargs)

        monkeypatch.setattr(process_control, "kill_run_tree", kill)
        try:
            result = detach.abort_run(str(phase))
        finally:
            process_control.record_cli_identities(None)
        assert calls == [[], [_cli(_LATE, f"{_LATE}-start", _LATE)]]
        assert (f"pidfd {_LATE}", "SIGTERM") in _signals(host)
        assert result["result"] == "not_running", result
        assert not _entry_kept(root)

    def test_the_sweep_s_prune_of_a_live_worker_closes_the_gate_too(
        self, tmp_path, monkeypatch
    ):
        plan, phase, root = _detached_run(tmp_path)
        _old(root)
        host = _Host({_WORKER: (_GROUP, "worker")}, pidfd=True).install(monkeypatch)
        monkeypatch.setattr(run_registry, "verdict_for", lambda _e: process_control.ALIVE)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(phase), "run-u77")
        real = process_control.kill_run_tree
        refused = []

        def kill(*args, **kwargs):
            try:
                process_control._record_launch()
            except process_control.LaunchRefused:
                refused.append(True)
            return real(*args, **kwargs)

        monkeypatch.setattr(process_control, "kill_run_tree", kill)
        try:
            rows = run_registry.sweep(root)
        finally:
            process_control.record_cli_identities(None)
        # The prune's kill closes the gate first; the close-out of the run's
        # still-open record takes its verdict rather than killing again.
        assert refused == [True]
        assert [(row["action"], row["kill_verdict"]) for row in rows] == [
            ("killed", "killed")]
        assert status_file.read_run(str(phase))["state"] == "failed"


# ---------------------------------------------------------------------------
# Round 4, point 5: an executor timeout that could not stop its runner
# ---------------------------------------------------------------------------


class TestAnExecutorTimeoutThatCouldNotStopItsRunner:
    """SIGCHLD ignored: the timed-out runner cannot be anchored as this
    process's child, so the kill sends it nothing (``unresolved``)."""

    @staticmethod
    def _runner(monkeypatch):
        host = _Host({_RUNNER: (_RUNNER, "provider")}, owned=[_RUNNER], pidfd=True)
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action",
                            lambda: (int(signal.SIG_IGN), 0))
        return host, types.SimpleNamespace(pid=_RUNNER, returncode=None)

    def test_the_verdict_comes_back_with_a_warning_naming_the_pid(
        self, monkeypatch, capsys
    ):
        from utils import subagent_executor as executor

        host, proc = self._runner(monkeypatch)
        verdict = executor._kill_timed_out(proc, _RUNNER)
        err = capsys.readouterr().err
        assert (verdict, host.sent) == ("unresolved", [])
        assert f"pid {_RUNNER}" in err and "could not be confirmed stopped" in err
        assert _RUNNER in host.procs

    @pytest.mark.parametrize("verdict, killed", [
        ("unresolved", False), ("skipped", False), ("survived", False), (None, False),
        ("some-other-word", False), ("killed", True), ("gone", True),
    ])
    def test_the_classification_claims_a_kill_only_when_there_was_one(
        self, verdict, killed
    ):
        from utils import subagent_executor as executor

        status, error = executor._classify_pre_parse(
            executor._Capture(timed_out=True, kill_verdict=verdict, kill_pid=_RUNNER))
        assert status == executor.STATUS_TIMEOUT
        assert ("was killed" in error) is killed
        if not killed:
            assert f"kill verdict {verdict}, pid {_RUNNER}" in error
            assert "may still be running" in error

    def test_the_spawn_carries_the_verdict_into_its_capture(self, monkeypatch, tmp_path):
        from utils import subagent_executor as executor
        from utils.providers.base import PROFILE_AGENT_READONLY
        from utils.providers.codex import CodexProvider

        provider = CodexProvider()
        monkeypatch.setattr(type(provider), "build_agent_command_for_transport",
                            lambda self, *a, **k: (["sleep", "30"], None))

        def unresolved(proc, _pgid, _job=None):
            proc.kill()  # the test's own child: stopped for real, said unresolved
            return "unresolved"

        monkeypatch.setattr(executor, "_kill_timed_out", unresolved)
        capture = executor._spawn_in_context(
            provider=provider, model="gpt-5.5", prompt="p", profile=PROFILE_AGENT_READONLY,
            transport="argv", timeout=0.3, cwd=str(tmp_path))
        assert capture.timed_out and capture.kill_verdict == "unresolved"
        assert capture.kill_pid
        assert "was killed" not in executor._classify_pre_parse(capture)[1]

    def test_a_detached_run_keeps_it_for_the_abort_that_stops_it(
        self, tmp_path, monkeypatch, capsys
    ):
        """The runner's identity is in the worker's record (verified at its
        launch); the note keeps that record and the registry entry past the
        run's finish, ``status --sweep`` reports it, and the abort that
        follows stops the runner by that identity."""
        from utils import subagent_executor as executor

        plan, phase, root = _detached_run(tmp_path)
        host, proc = self._runner(monkeypatch)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        monkeypatch.setattr(process_control, "_LAUNCHES_REFUSED", [], raising=False)
        process_control.record_cli_identities(str(phase), "run-u77")
        handle = detach.RunHandle(str(phase), run_id="run-u77", mode="review-plan",
                                  plan_file=str(plan))
        monkeypatch.setattr(detach, "_ACTIVE", handle)
        token = process_control._record_launch()
        process_control._record_spawned(token, proc)
        try:
            assert executor._kill_timed_out(proc, _RUNNER) == "unresolved"
            cleanup = status_file.read_run(str(phase))["cleanup"]
            assert (cleanup["state"], cleanup["verdict"], cleanup["pid"]) == (
                "incomplete", "unresolved", _RUNNER)
            host.procs.pop(_WORKER, None)
            handle.finish(0)
        finally:
            process_control.record_cli_identities(None)
        record = Path(detach.cli_record_path(str(phase), "run-u77"))
        assert record.exists() and _entry_kept(root)
        capsys.readouterr()
        assert run_status.main(["--plan-file", str(plan), "--sweep"]) == 3
        assert "kill verdict unresolved" in capsys.readouterr().out
        result = detach.abort_run(str(phase))
        assert result["result"] == "not_running", result
        assert (f"pidfd {_RUNNER}", "SIGTERM") in _signals(host)
        assert host.hit_bystanders() == []
        assert not record.exists() and not _entry_kept(root)

    def test_without_a_record_nothing_is_written(self, tmp_path, monkeypatch, capsys):
        """A foreground run records no launches: the warning is all there is."""
        from utils import subagent_executor as executor

        plan, phase, root = _detached_run(tmp_path)
        host, proc = self._runner(monkeypatch)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        handle = detach.RunHandle(str(phase), run_id="run-u77", mode="review-plan",
                                  plan_file=str(plan), detached=False)
        monkeypatch.setattr(detach, "_ACTIVE", handle)
        assert executor._kill_timed_out(proc, _RUNNER) == "unresolved"
        assert "cleanup" not in status_file.read_run(str(phase))
        assert f"pid {_RUNNER}" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Round 4, point 6: a FIFO in a record's place
# ---------------------------------------------------------------------------


def _no_wait(call, timeout=5.0):
    """*call*'s answer, made in a thread that must end within *timeout*."""
    import threading

    answer = []
    thread = threading.Thread(target=lambda: answer.append(call()), daemon=True)
    thread.start()
    thread.join(timeout)
    return thread.is_alive(), answer


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs")
class TestARecordThatIsAFifo:
    """Opened blocking, a FIFO waits for a writer before any check runs, and
    an abort or sweep hangs on it. Each open is non-blocking and refuses it."""

    @staticmethod
    def _fifo(tmp_path):
        path = tmp_path / process_control.cli_record_name(_RUN)
        os.mkfifo(path)
        return path

    @staticmethod
    def _unblock(path):
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass

    def test_reading_it_answers_at_once_as_incomplete(self, tmp_path):
        path = self._fifo(tmp_path)
        try:
            hung, answer = _no_wait(lambda: _read(path))
        finally:
            self._unblock(path)
        assert not hung, "read_cli_record waited on a FIFO"
        assert answer == [process_control.CliRecord([], False)]

    @records_here
    def test_a_launch_or_a_gate_refuses_it_at_once(self, tmp_path, monkeypatch):
        path = self._fifo(tmp_path)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(tmp_path), _RUN)
        spawned = []
        monkeypatch.setattr(process_control, "_popen_tree",
                            lambda *a: spawned.append(a))

        def launch():
            try:
                process_control.popen_tree(["codex"])
            except process_control.LaunchRefused as exc:
                return str(exc)

        try:
            hung, refused = _no_wait(launch)
            gate_hung, gate = _no_wait(
                lambda: process_control.close_launch_gate(str(tmp_path), _RUN))
        finally:
            process_control.record_cli_identities(None)
            self._unblock(path)
        assert not hung and not gate_hung
        assert refused[0] and "nothing was started" in refused[0] and spawned == []
        assert not gate[0].complete


# ---------------------------------------------------------------------------
# Real processes
# ---------------------------------------------------------------------------

_SLEEPER_IN_ITS_OWN_SESSION = (
    "import subprocess\n"
    "child = subprocess.Popen(['sleep', '60'], start_new_session=True,\n"
    "                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    "print(child.pid, flush=True)\n"
)

#: Makes a session (so its group's leader is gone once it exits), starts a
#: worker and a member that ignores SIGTERM in it, prints the group and both
#: pids and exits: a detached worker's shape.
_A_WORKERS_GROUP = (
    "import os, subprocess, sys\n"
    "os.setsid()\n"
    "quiet = dict(stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    "worker = subprocess.Popen(['sleep', '60'], stdout=subprocess.DEVNULL, **quiet)\n"
    "member = subprocess.Popen([sys.executable, '-c',\n"
    "    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '\n"
    "    'print(1, flush=True); time.sleep(60)'], stdout=subprocess.PIPE, text=True,\n"
    "    **quiet)\n"
    "member.stdout.readline()\n"
    "print(os.getpgid(0), worker.pid, member.pid, flush=True)\n"
    "os._exit(0)\n"
)


def _orphan(code):
    """Pids an exited parent printed: processes reparented to init."""
    spawned = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=30)
    return [int(word) for word in spawned.stdout.split()]


def _running(pid):
    return process_control._Liveness().pid_running(pid)


def _gone(*pids, timeout=5.0):
    deadline = time.monotonic() + timeout
    while any(_running(pid) for pid in pids):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _kill_quietly(*pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


identity_kill_here = pytest.mark.skipif(
    getattr(process_control, "_pidfd_api", lambda: None)() is None,
    reason="the identity kill runs on Linux with pidfds and /proc only")


@pytest.fixture
def bystander():
    [pid] = _orphan(_SLEEPER_IN_ITS_OWN_SESSION)
    yield pid
    _kill_quietly(pid)


@identity_kill_here
class TestRealProcesses:
    def test_a_recorded_group_now_another_process_s_is_neither_signalled_nor_read(
        self, bystander
    ):
        reaped = subprocess.Popen(["true"])
        reaped.wait()
        verdict = process_control.kill_tree(
            reaped.pid, start_time="not-its-start-time", pgid=bystander,
            escalate_after_s=0.3, poll_interval_s=0.05)
        assert verdict == "gone"
        time.sleep(0.2)
        assert _running(bystander), "the bystander's group was signalled"

    def test_a_live_worker_and_its_recorded_cli_are_stopped(self):
        group, worker, member = _orphan(_A_WORKERS_GROUP)
        try:
            verdict = process_control.kill_tree(
                worker, start_time=process_control.process_start_time(worker),
                pgid=group, escalate_after_s=0.3, poll_interval_s=0.05,
                members=[_cli(member, process_control.process_start_time(member), group)])
            assert verdict == "killed"
            assert _gone(worker, member), "the member that ignored SIGTERM still runs"
        finally:
            _kill_quietly(worker, member)

    def test_an_unrecorded_member_is_left_and_reported(self):
        group, worker, member = _orphan(_A_WORKERS_GROUP)
        try:
            verdict = process_control.kill_tree(
                worker, start_time=process_control.process_start_time(worker),
                pgid=group, escalate_after_s=0.2, poll_interval_s=0.05)
            assert verdict == "unresolved"
            assert _gone(worker)
            assert _running(member), "an unrecorded member was signalled"
        finally:
            _kill_quietly(worker, member)

    def test_a_crashed_worker_s_recorded_cli_is_stopped(self, tmp_path, monkeypatch):
        group, worker, member = _orphan(_A_WORKERS_GROUP)
        try:
            start = process_control.process_start_time(worker)
            record = tmp_path / process_control.cli_record_name(_RUN)
            record.write_text(json.dumps({"run": _RUN, "launch": "1.1"}) + "\n" + json.dumps({
                "run": _RUN, "cli": "1.1", "pid": member, "pgid": group,
                "start": process_control.process_start_time(member)}) + "\n",
                encoding="utf-8")
            os.kill(worker, signal.SIGKILL)  # the crash: no handler ran
            assert _gone(worker)
            # No record: the member is left, and the verdict says so.
            assert process_control.kill_tree(
                worker, start_time=start, pgid=group, escalate_after_s=0.2,
                poll_interval_s=0.05) == "unresolved"
            assert _running(member)
            assert process_control.kill_tree(
                worker, start_time=start, pgid=group, escalate_after_s=0.2,
                poll_interval_s=0.05,
                members=_read(record).members) == "killed"
            assert _gone(member)
        finally:
            _kill_quietly(worker, member)

    def test_a_cli_the_worker_started_in_its_own_session_is_recorded_and_stopped(
        self, tmp_path, monkeypatch
    ):
        """Round 2, point 2, end to end: an executor-style launch (a session
        of its own) through the one spawn helper, then the close-out's kill
        from the record once the worker is gone."""
        path = tmp_path / process_control.cli_record_name(_RUN)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(tmp_path), _RUN)
        try:
            proc, _job = process_control.popen_tree(
                [sys.executable, "-c",
                 "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                 "print(1, flush=True); time.sleep(60)"],
                stdout=subprocess.PIPE, text=True,
                **process_control.new_process_group_kwargs())
        finally:
            process_control.record_cli_identities(None)
        try:
            proc.stdout.readline()  # SIGTERM is ignored from here on
            record = _read(path)
            assert record.complete and [m.pid for m in record.members] == [proc.pid]
            gone_worker = subprocess.Popen(["true"])
            gone_worker.wait()
            assert process_control.kill_tree(
                gone_worker.pid, start_time="worker", pgid=None,
                escalate_after_s=0.2, poll_interval_s=0.05,
                members=record.members) == "killed"
            assert proc.wait(timeout=5) == -signal.SIGKILL
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()

    def test_a_sweep_never_signals_a_group_its_member_left(self, bystander):
        [member] = _orphan(_SLEEPER_IN_ITS_OWN_SESSION)
        try:
            record = process_control.Descendant(
                member, bystander, bystander, process_control.process_start_time(member))
            verdict = process_control._sweep_groups([record], 0.3, 0.05)
            assert _gone(member), "the member itself was not stopped"
            # The recorded group's number is the bystander's own pid: its group.
            assert verdict == {bystander: "killed"}
            assert _running(bystander), "the bystander's group was signalled"
        finally:
            _kill_quietly(member)

    def test_a_pidfd_answers_exit_and_reaches_its_process_or_nothing(self):
        api = process_control._pidfd_api()
        child = subprocess.Popen(["sleep", "60"])
        fd = api.open(child.pid)
        try:
            assert fd is not None and api.exists(fd)
            assert api.exited(fd) is False
            assert api.send(fd, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while not api.exited(fd) and time.monotonic() < deadline:
                time.sleep(0.01)
            # A zombie: exited, though its pid is not handed out yet.
            assert api.exited(fd) is True and api.exists(fd)
            assert child.wait(timeout=5) == -signal.SIGTERM
            # Reaped: whatever takes the pid now, the pidfd reaches none of it.
            assert api.exited(fd) is True and not api.exists(fd)
            assert not api.send(fd, signal.SIGKILL)
        finally:
            if fd is not None:
                api.close(fd)
            if child.poll() is None:
                child.kill()
                child.wait()


# ---------------------------------------------------------------------------
# Round 5, point 1: SIGCHLD at its default, and a Lane A root verified before
# anything under it is adopted
# ---------------------------------------------------------------------------

_SKILL_DIR = Path(__file__).resolve().parent.parent

#: Run in a child that inherited SIGCHLD ignored: one process entry, then the
#: disposition it left. ``cli`` is ``multi_llm.main``; any other word is an
#: orchestrator run directly, as its own ``__main__``.
_ENTRY_PROBE = """
import runpy, signal, sys
sys.path.insert(0, {skill!r})
print("inherited", signal.getsignal(signal.SIGCHLD) == signal.SIG_IGN, file=sys.stderr)
entry = {entry!r}
try:
    if entry == "cli":
        import multi_llm
        multi_llm.main(["no-such-mode"])
    else:
        sys.argv = [entry, "--help"]
        runpy.run_path({skill!r} + "/" + entry, run_name="__main__")
except SystemExit:
    pass
print("after", "SIG_DFL" if signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
      else "not-default", file=sys.stderr)
"""


class TestAProcessEntryRestoresSigchld:
    """Under an inherited ``SIG_IGN`` the kernel reaps every child at once, so
    a ``Popen`` with no exit status no longer reserves its pid."""

    @pytest.mark.parametrize("entry", [
        "cli", "review_plan_orchestrator.py", "implement_orchestrator.py"])
    def test_an_entry_resets_an_inherited_sig_ign(self, tmp_path, entry):
        probe = _ENTRY_PROBE.format(skill=str(_SKILL_DIR), entry=entry)
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=str(tmp_path), capture_output=True,
            text=True, timeout=120,
            preexec_fn=lambda: signal.signal(signal.SIGCHLD, signal.SIG_IGN))
        lines = result.stderr.splitlines()
        assert "inherited True" in lines, result.stderr
        assert lines[-1] == "after SIG_DFL", result.stderr

    def test_every_orchestrator_and_the_worker_reset_it_first(self):
        """Each orchestrator's ``__main__`` block, before it runs ``main``;
        the detached worker's entry into its run."""
        import ast
        import inspect
        import textwrap

        def calls(nodes):
            return [getattr(node.func, "attr", getattr(node.func, "id", None))
                    for stmt in nodes for node in ast.walk(stmt)
                    if isinstance(node, ast.Call)]

        missing = []
        for path in sorted(_SKILL_DIR.glob("*_orchestrator.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            [block] = [node for node in tree.body if isinstance(node, ast.If)
                       and "__main__" in ast.unparse(node.test)]
            named = calls(block.body)
            if "restore_sigchld_default" not in named or named.index(
                    "restore_sigchld_default") > named.index("exit"):
                missing.append(path.name)
        assert missing == []
        source = textwrap.dedent(inspect.getsource(process_control.enter_detached_worker))
        assert "restore_sigchld_default" in calls(ast.parse(source).body)

    @pytest.mark.parametrize("inherited, reset", [
        (signal.SIG_IGN, True), (signal.SIG_DFL, False), (lambda *_a: None, False)])
    def test_only_an_ignored_sigchld_is_changed(self, monkeypatch, inherited, reset):
        changed = []
        monkeypatch.setattr(process_control.signal, "getsignal", lambda _s: inherited)
        monkeypatch.setattr(process_control.signal, "signal",
                            lambda signum, handler: changed.append((signum, handler)))
        process_control.restore_sigchld_default()
        assert changed == ([(signal.SIGCHLD, signal.SIG_DFL)] if reset else [])


#: A Lane A CLI's pid, auto-reaped and taken by another process, and that
#: process's child.
_CLI_PID = 4120
_FOREIGN_CHILD = 4121


def _replaced_root(monkeypatch, *, pidfd, registered="original-cli", rows=True):
    """The CLI registered with start time *registered*; its pid now runs as
    another process (``reused-root``), whose child is a bystander. Its
    ``Popen`` has no exit status: SIGCHLD was ignored, and the kernel reaped
    it."""
    host = _Host({_CLI_PID: (_GROUP, "reused-root"),
                  _FOREIGN_CHILD: (_GROUP, "foreign-child")}, pidfd=pidfd)
    host.install(monkeypatch)
    host.bystanders.update([_CLI_PID, _FOREIGN_CHILD])
    monkeypatch.setattr(process_control, "_sigchld_action",
                        lambda: (int(signal.SIG_IGN), 0))

    def table(*_args):
        rows_now = host.table()
        rows_now[_FOREIGN_CHILD] = rows_now[_FOREIGN_CHILD]._replace(ppid=_CLI_PID)
        return rows_now if rows else {
            pid: row for pid, row in rows_now.items() if pid != _CLI_PID}

    monkeypatch.setattr(process_control, "_process_table", table)
    proc = types.SimpleNamespace(pid=_CLI_PID, returncode=None)
    registry = process_control.LiveCliRegistry()
    registry._entries[id(proc)] = process_control.LiveCli(_CLI_PID, registered, proc)
    return host, registry, proc


class TestALaneARootIsVerifiedBeforeItsChildrenAreAdopted:
    """``returncode is None`` proves nothing once the kernel reaps children
    itself: the snapshot adopts a root's children only when the root's row
    has the start time it was registered with."""

    def test_a_replacement_s_child_is_not_adopted_and_gets_nothing(
        self, monkeypatch, pidfd
    ):
        host, registry, proc = _replaced_root(monkeypatch, pidfd=pidfd)
        snapshot = registry.snapshot_for(proc)
        assert list(snapshot) == [] and snapshot.unverified_root == _CLI_PID
        groups = _sweep(snapshot)
        assert groups == {-_CLI_PID: "unresolved"}
        assert (host.sent, host.hit_bystanders()) == ([], [])
        described = process_control.TreeSweep(cli="terminated", groups=groups).describe()
        assert "UNRESOLVED" in described

    def test_a_root_with_no_row_adopts_nothing(self, monkeypatch, pidfd):
        _host, registry, proc = _replaced_root(monkeypatch, pidfd=pidfd, rows=False)
        snapshot = registry.snapshot_for(proc)
        assert list(snapshot) == [] and snapshot.unverified_root == _CLI_PID

    def test_a_root_registered_undated_adopts_nothing(self, monkeypatch, pidfd):
        _host, registry, proc = _replaced_root(monkeypatch, pidfd=pidfd, registered=None)
        assert registry.snapshot_for(proc).unverified_root == _CLI_PID

    def test_the_refresh_adopts_only_under_a_verified_root(self, monkeypatch):
        _host, registry, proc = _replaced_root(monkeypatch, pidfd=True)
        registry.refresh()
        assert registry.entries()[0].snapshot == []

    def test_the_root_registered_with_that_start_time_is_adopted(self, monkeypatch):
        """The control: the same table, the root verified."""
        _host, registry, proc = _replaced_root(monkeypatch, pidfd=True,
                                               registered="reused-root")
        snapshot = registry.snapshot_for(proc)
        assert [d.pid for d in snapshot] == [_FOREIGN_CHILD]
        assert snapshot.unverified_root is None
        registry.refresh()
        assert [d.pid for d in registry.entries()[0].snapshot] == [_FOREIGN_CHILD]


# ---------------------------------------------------------------------------
# Round 5, point 2: what one kill could not confirm outlives the next
# ---------------------------------------------------------------------------

#: A recorded CLI that moved to a group of its own, which its tool leads, and
#: a launch whose identity lands after the gate's read.
_MOVED_CLI = 4110
_TOOL = 4112
_LATE_CLI = 4113


def _moved_group(tmp_path, monkeypatch, *, tool_exits_in_second_kill=False):
    """Codex's round-5 model: the first kill reads the moved group and finds
    the tool running there unidentified (``unresolved``); the reread finds
    the late launch and kills again, and that kill no longer sees the moved
    group: the CLI that showed it has exited. Returns the abort's result,
    the kills' verdicts, the host and the project root."""
    plan, phase, root = _detached_run(tmp_path)
    host = _Host({_WORKER: (_GROUP, "worker"), _MOVED_CLI: (_GROUP, "cli")},
                 owned=[_MOVED_CLI], pidfd=True).install(monkeypatch)
    monkeypatch.setattr(process_control, "_CLI_RECORD", [])
    monkeypatch.setattr(process_control, "_LAUNCHES_REFUSED", [], raising=False)
    process_control.record_cli_identities(str(phase), "run-u77")
    try:
        token = process_control._record_launch()
        process_control._record_spawned(token, types.SimpleNamespace(pid=_MOVED_CLI))
        late = process_control._record_launch()
        # The CLI, not its group's leader, moves to the tool's group, in the
        # same session; during the teardown it is another process's.
        host.procs[_MOVED_CLI] = (_TOOL, "cli")
        host.procs[_TOOL] = (_TOOL, "tool")
        host.owned.clear()
        plain = host.table

        def table(*_args):
            return {pid: row._replace(sid=_GROUP if pid in (_WORKER, _MOVED_CLI, _TOOL)
                                      else row.sid)
                    for pid, row in plain().items()}

        host.table = table
        monkeypatch.setattr(process_control, "_process_table", table)
        monkeypatch.setattr(process_control, "_linux_process_table", table)
        real = process_control.kill_run_tree
        verdicts = []

        def kill(*args, **kwargs):
            if not verdicts:
                host.procs[_LATE_CLI] = (_LATE_CLI, "late")
                host.owned.add(_LATE_CLI)
                process_control._record_spawned(
                    late, types.SimpleNamespace(pid=_LATE_CLI))
                host.owned.discard(_LATE_CLI)
            elif tool_exits_in_second_kill:
                host.procs.pop(_TOOL, None)
            verdict = real(*args, **{**kwargs, "escalate_after_s": 0.1,
                                     "poll_interval_s": 0.05})
            verdicts.append(verdict)
            return verdict

        monkeypatch.setattr(process_control, "kill_run_tree", kill)
        result = detach.abort_run(str(phase))
    finally:
        process_control.record_cli_identities(None)
    return result, verdicts, host, root, phase


class TestAMovedGroupOutlivesTheRereadsKill:
    def test_the_first_kill_s_unresolved_group_stands(self, tmp_path, monkeypatch):
        result, verdicts, host, root, phase = _moved_group(tmp_path, monkeypatch)
        assert verdicts == ["unresolved", "killed"]
        assert (result["result"], result.get("verdict")) == ("incomplete", "unresolved")
        assert _TOOL in host.procs and not [
            how for how, _sig in _signals(host) if how.endswith(str(_TOOL))]
        assert Path(detach.cli_record_path(str(phase), "run-u77")).exists()
        assert _entry_kept(root)
        assert status_file.read_run(str(phase))["cleanup"]["state"] == "incomplete"

    def test_a_group_re_checked_empty_is_cleared(self, tmp_path, monkeypatch):
        """The control: the tool is gone by the time the second kill ends."""
        result, verdicts, host, root, phase = _moved_group(
            tmp_path, monkeypatch, tool_exits_in_second_kill=True)
        assert verdicts == ["unresolved", "killed"]
        assert result["result"] == "aborted", result
        assert not Path(detach.cli_record_path(str(phase), "run-u77")).exists()
        assert not _entry_kept(root)


class TestWhatAKillOwesIsClearedOnlyByAReCheck:
    @staticmethod
    def _table(monkeypatch, rows):
        table = {row.pid: row for row in rows}
        monkeypatch.setattr(process_control, "_linux_process_table", lambda: table)

    @staticmethod
    def _row(pid, pgid, start, state="S"):
        return process_control._ProcRow(pid, 1, pgid, pgid, start, state)

    def test_a_group_until_it_runs_nothing_but_zombies(self, monkeypatch):
        owed = process_control.KillObligations()
        owed.groups.add(_GROUP)
        self._table(monkeypatch, [self._row(_MEMBER, _GROUP, "m")])
        assert owed.outstanding() == ["unresolved"]
        self._table(monkeypatch, [self._row(_MEMBER, _GROUP, "m", "Z")])
        assert owed.outstanding() == []

    def test_an_identity_until_no_process_runs_as_it(self, monkeypatch):
        owed = process_control.KillObligations()
        owed.owe(_MEMBER, "m", "survived")
        owed.owe(_LATE, None, "unresolved")
        self._table(monkeypatch, [self._row(_MEMBER, _GROUP, "m"),
                                  self._row(_LATE, _LATE, "whoever")])
        assert sorted(owed.outstanding()) == ["survived", "unresolved"]
        # Another start time for the dated one; the undated one's pid free.
        self._table(monkeypatch, [self._row(_MEMBER, _GROUP, "someone-else")])
        assert owed.outstanding() == []

    def test_nothing_is_cleared_without_the_table(self, monkeypatch):
        owed = process_control.KillObligations()
        owed.groups.add(_GROUP)
        owed.owe(_MEMBER, "m", "survived")
        monkeypatch.setattr(process_control, "_linux_process_table", lambda: None)
        assert sorted(owed.outstanding()) == ["survived", "unresolved"]

    @pytest.mark.parametrize("verdict", [None, "survived", "some-other-word"])
    def test_a_verdict_worse_than_what_it_named_is_owed_as_it_stands(
        self, monkeypatch, verdict
    ):
        owed = process_control.KillObligations()
        owed.groups.add(_GROUP)
        monkeypatch.setattr(process_control, "_linux_process_table", lambda: {})
        expected = [verdict] if verdict != "some-other-word" else []
        assert detach._owed_by(verdict, owed) == expected
        assert detach._owed_by("unresolved", process_control.KillObligations()) == [
            "unresolved"]

    def test_the_worst_verdict_wins_and_killed_beats_gone(self):
        assert detach._worst_verdict(["killed", "unresolved"]) == "unresolved"
        assert detach._worst_verdict(["gone", "killed"]) == "killed"
        assert detach._worst_verdict(["killed", "some-other-word"]) == "unresolved"


# ---------------------------------------------------------------------------
# Round 5, point 3: the executor dates its own child
# ---------------------------------------------------------------------------


class TestTheExecutorDatesItsOwnChild:
    """Without a ``sigaction`` ABI the anchor cannot be read; the child's
    start time, read while its row names this process as its parent, holds it
    for the kill by identity instead."""

    def test_an_unnamed_abi_kills_its_own_child(self, monkeypatch, capsys):
        from utils import subagent_executor as executor

        host = _Host({_RUNNER: (_RUNNER, "provider")}, owned=[_RUNNER], pidfd=True)
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", lambda: None)
        proc = types.SimpleNamespace(pid=_RUNNER, returncode=None)
        assert process_control.dated_own_child(proc) == "provider"
        assert executor._kill_timed_out(proc, _RUNNER) == "killed"
        assert (f"pidfd {_RUNNER}", "SIGTERM") in _signals(host)
        assert _RUNNER in host.zombies and "WARNING" not in capsys.readouterr().err

    @pytest.mark.parametrize("case", ["another-parent", "reaped", "sigchld-ignored"])
    def test_ownership_not_established_sends_nothing(self, monkeypatch, capsys, case):
        from utils import subagent_executor as executor

        host = _Host({_RUNNER: (_RUNNER, "provider")},
                     owned=[] if case == "another-parent" else [_RUNNER], pidfd=True)
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", lambda: None)
        if case == "sigchld-ignored":
            monkeypatch.setattr(process_control.signal, "getsignal",
                                lambda _s: signal.SIG_IGN)
        proc = types.SimpleNamespace(pid=_RUNNER,
                                     returncode=0 if case == "reaped" else None)
        assert process_control.dated_own_child(proc) is None
        verdict = executor._kill_timed_out(proc, _RUNNER)
        if case == "reaped":
            # Its exit status is had: the child is gone, whatever has the pid.
            assert verdict == "gone"
        else:
            assert verdict == "unresolved"
            assert f"pid {_RUNNER}" in capsys.readouterr().err
        assert host.sent == []

    @on_linux_with_proc
    @identity_kill_here
    def test_a_real_harmless_child_under_an_unnamed_abi(self, monkeypatch):
        from utils import subagent_executor as executor

        api = process_control._pidfd_api()
        monkeypatch.setattr(process_control, "_PIDFDS", [api])
        monkeypatch.setattr(process_control, "_SIGACTION_ABI", [None])
        assert process_control._sigchld_action() is None
        child = subprocess.Popen(["sleep", "30"],
                                 **process_control.new_process_group_kwargs())
        try:
            assert executor._kill_timed_out(child, child.pid) == "killed"
            assert child.wait(timeout=5) == -signal.SIGTERM
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


# ---------------------------------------------------------------------------
# Round 5, point 4: every unconfirmed timeout kill is reported
# ---------------------------------------------------------------------------


class TestEveryUnconfirmedTimeoutKillIsReported:
    """Only ``killed`` and ``gone`` confirm the runner stopped: a warning
    naming the pid, a classification that does not say killed, and in a
    detached run an incomplete ``run.cleanup`` for everything else."""

    @staticmethod
    def _assert_reported(executor, verdict, err):
        assert f"pid {_RUNNER}" in err and "could not be confirmed stopped" in err
        error = executor._classify_pre_parse(
            executor._Capture(timed_out=True, kill_verdict=verdict, kill_pid=_RUNNER))[1]
        assert "was killed" not in error and "may still be running" in error

    def test_signals_denied_on_the_identity_kill(self, monkeypatch, capsys):
        from utils import subagent_executor as executor

        host = _Host({_RUNNER: (_GROUP, "provider")}, owned=[_RUNNER], pidfd=True)
        host.install(monkeypatch)
        api = process_control._pidfd_api()
        real = api._send

        def deny(fd, sig):
            if sig:
                raise PermissionError("signals denied")
            return real(fd, sig)

        monkeypatch.setattr(api, "_send", deny)
        proc = types.SimpleNamespace(pid=_RUNNER, returncode=None)
        assert executor._kill_timed_out(proc, _GROUP) == "survived"
        assert _RUNNER not in host.zombies
        self._assert_reported(executor, "survived", capsys.readouterr().err)

    def test_signals_denied_on_the_owned_kill(self, monkeypatch, capsys):
        from utils import subagent_executor as executor

        host = _Host({_RUNNER: (_RUNNER, "provider")}, owned=[_RUNNER], pidfd=True)
        host.install(monkeypatch)

        def denied(_pid, sig):
            if sig:
                raise PermissionError("host policy denied the signal")

        monkeypatch.setattr(process_control.os, "kill", denied)
        monkeypatch.setattr(process_control.os, "killpg", denied)
        proc = types.SimpleNamespace(pid=_RUNNER, returncode=None)
        assert executor._kill_timed_out(proc, _RUNNER) == "survived"
        self._assert_reported(executor, "survived", capsys.readouterr().err)

    @pytest.mark.parametrize("fallback_works", [True, False])
    def test_a_kill_that_raised(self, monkeypatch, capsys, fallback_works):
        from utils import subagent_executor as executor

        def broken(*_a, **_k):
            raise OSError("cleanup failed")

        def fallback():
            if not fallback_works:
                raise PermissionError("kill denied")

        monkeypatch.setattr(process_control, "kill_tree", broken)
        proc = types.SimpleNamespace(pid=_RUNNER, returncode=None, kill=fallback)
        assert executor._kill_timed_out(proc, _RUNNER) is None
        self._assert_reported(executor, None, capsys.readouterr().err)

    def test_an_unknown_verdict(self, monkeypatch, capsys):
        from utils import subagent_executor as executor

        monkeypatch.setattr(process_control, "kill_tree", lambda *_a, **_k: "maybe")
        proc = types.SimpleNamespace(pid=_RUNNER, returncode=None)
        assert executor._kill_timed_out(proc, _RUNNER) == "maybe"
        self._assert_reported(executor, "maybe", capsys.readouterr().err)

    @pytest.mark.parametrize("verdict", ["survived", None])
    def test_a_detached_run_keeps_it(self, tmp_path, monkeypatch, capsys, verdict):
        from utils import subagent_executor as executor

        plan, phase, _root = _detached_run(tmp_path)

        def kill_tree(*_a, **_k):
            if verdict is None:
                raise OSError("cleanup failed")
            return verdict

        monkeypatch.setattr(process_control, "kill_tree", kill_tree)
        monkeypatch.setattr(process_control, "_CLI_RECORD", [])
        process_control.record_cli_identities(str(phase), "run-u77")
        handle = detach.RunHandle(str(phase), run_id="run-u77", mode="review-plan",
                                  plan_file=str(plan))
        monkeypatch.setattr(detach, "_ACTIVE", handle)
        try:
            proc = types.SimpleNamespace(pid=_RUNNER, returncode=None, kill=lambda: None)
            assert executor._kill_timed_out(proc, _RUNNER) == verdict
        finally:
            process_control.record_cli_identities(None)
        cleanup = status_file.read_run(str(phase))["cleanup"]
        assert (cleanup["state"], cleanup["verdict"], cleanup["pid"]) == (
            "incomplete", verdict, _RUNNER)

    def test_the_kill_by_number_s_zombie_is_a_confirmed_exit(self, monkeypatch, capsys):
        """No pidfds, no named ABI: the kill by number waits out the child's
        own zombie and calls it ``survived``; the child has exited and its
        group runs nothing, so the timeout kill is ``killed``. Not reaped."""
        from utils import subagent_executor as executor

        host = _Host({_RUNNER: (_RUNNER, "provider")}, owned=[_RUNNER])
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", lambda: None)
        proc = types.SimpleNamespace(pid=_RUNNER, returncode=None)
        assert executor._kill_timed_out(proc, _RUNNER) == "killed"
        assert [entry[2] for entry in host.sent] == ["SIGTERM", "SIGKILL"]
        assert _RUNNER in host.zombies and "WARNING" not in capsys.readouterr().err

    def test_a_zombie_beside_a_survivor_in_its_group_is_not(self, monkeypatch, capsys):
        from utils import subagent_executor as executor

        host = _Host({_RUNNER: (_RUNNER, "provider"), _MEMBER: (_RUNNER, "tool")},
                     owned=[_RUNNER], deaf=[_MEMBER], unkillable=[_MEMBER])
        host.install(monkeypatch)
        monkeypatch.setattr(process_control, "_sigchld_action", lambda: None)
        proc = types.SimpleNamespace(pid=_RUNNER, returncode=None)
        assert executor._kill_timed_out(proc, _RUNNER) == "survived"
        self._assert_reported(executor, "survived", capsys.readouterr().err)
