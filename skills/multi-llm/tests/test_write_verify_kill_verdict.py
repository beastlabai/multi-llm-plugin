"""What a timed-out ``driver.verify_command`` reports about its kill.

``write_verify.VerifyCommand.kill_tree`` keeps the executor's rule for its
timed-out runner (``process_control.unconfirmed_kill``): only ``killed`` and
``gone`` confirm the command stopped, and a ``survived`` that is only the
command's own zombie becomes ``killed`` once its exit is confirmed. Every other
verdict (``survived``, ``unresolved``, ``skipped``, ``None`` for a kill that
raised, any other word) is logged as a warning naming the pid, is returned as
it is, and words the output's closing note without claiming a kill
(docs/development/write-scheduler.md, "When the verify command times out").
Signalling itself is ``process_control.kill_tree``'s; these tests fake it.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import process_control  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_batch_scheduler as wbs  # noqa: E402
from utils import write_verify  # noqa: E402
from tests.scheduler_support import make_scheduler  # noqa: E402

PID = 4242
PGID = 4242

#: Verdicts that leave the command possibly running, ``None`` (the kill
#: raised) included.
UNCONFIRMED = ["survived", "unresolved", "skipped", "maybe"]


class _Proc:
    """A ``Popen`` stand-in: a pid, a no-op reap, and a record of kills."""

    def __init__(self) -> None:
        self.pid = PID
        self.returncode = None
        self.killed = 0
        self.waited = 0

    def kill(self) -> None:
        self.killed += 1

    def wait(self, timeout=None):
        self.waited += 1
        return 0


@pytest.fixture
def scheduler(tmp_path):
    lines = []
    built = make_scheduler(tmp_path, kind="implement", log=lines.append)
    built.lines = lines
    return built


@pytest.fixture
def left_running(monkeypatch):
    """``detach.note_left_running``'s calls, answered ``False`` (no detached
    run here)."""
    from utils import detach
    calls = []

    def note(pid, pgid, verdict):
        calls.append((pid, pgid, verdict))
        return False

    monkeypatch.setattr(detach, "note_left_running", note)
    return calls


def _kill_answers(monkeypatch, verdict, *, zombie=False):
    calls = []

    def kill_tree(pid, **kwargs):
        calls.append((pid, kwargs))
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict

    monkeypatch.setattr(write_verify._process_control, "kill_tree", kill_tree)
    monkeypatch.setattr(write_verify._process_control, "own_child_stopped",
                        lambda child, pgid: zombie)
    return calls


def _warnings(scheduler):
    return [line for line in scheduler.lines if "warning:" in line]


class TestOnlyKilledAndGoneConfirmAStop:

    def test_the_rule_is_the_executors(self):
        """One rule for both timeout kills: the executor's."""
        assert process_control.CONFIRMED_KILLS == ("killed", "gone")
        for verdict in process_control.CONFIRMED_KILLS:
            assert process_control.unconfirmed_kill(verdict) is None
        for verdict in [*UNCONFIRMED, None]:
            assert process_control.unconfirmed_kill(verdict) is not None

    @pytest.mark.parametrize("verdict", ["killed", "gone"])
    def test_a_confirmed_stop_warns_nothing(self, scheduler, monkeypatch,
                                            left_running, verdict):
        _kill_answers(monkeypatch, verdict)
        proc = _Proc()

        assert scheduler._verify_command.kill_tree(proc, PGID) == verdict
        assert _warnings(scheduler) == []
        assert left_running == []
        assert proc.waited == 1 and proc.killed == 0

    def test_a_survived_over_its_own_zombie_is_killed(self, scheduler, monkeypatch,
                                                      left_running):
        """The kill by number reads the command's unreaped zombie as running;
        once its exit and an empty group are confirmed it was killed."""
        _kill_answers(monkeypatch, "survived", zombie=True)

        assert scheduler._verify_command.kill_tree(_Proc(), PGID) == "killed"
        assert _warnings(scheduler) == []
        assert left_running == []

    @pytest.mark.parametrize("verdict", UNCONFIRMED)
    def test_any_other_verdict_warns_with_the_pid(self, scheduler, monkeypatch,
                                                  left_running, verdict):
        _kill_answers(monkeypatch, verdict)

        assert scheduler._verify_command.kill_tree(_Proc(), PGID) == verdict
        [warning] = _warnings(scheduler)
        assert f"pid {PID} (process group {PGID})" in warning
        assert f"kill verdict {verdict}" in warning
        assert "may still be running" in warning
        assert process_control.unconfirmed_kill(verdict) in warning
        assert "killed" not in warning.replace(f"kill verdict {verdict}", "")
        assert left_running == [(PID, PGID, verdict)]

    def test_a_survived_whose_exit_is_not_confirmed_stays_survived(
            self, scheduler, monkeypatch, left_running):
        _kill_answers(monkeypatch, "survived", zombie=False)

        assert scheduler._verify_command.kill_tree(_Proc(), PGID) == "survived"
        assert len(_warnings(scheduler)) == 1

    def test_a_kill_that_raised_is_none_and_falls_back_unconfirmed(
            self, scheduler, monkeypatch, left_running):
        _kill_answers(monkeypatch, RuntimeError("no /proc"))
        proc = _Proc()

        assert scheduler._verify_command.kill_tree(proc, PGID) is None
        assert proc.killed == 1, "the fallback kill still goes to the command"
        [warning] = _warnings(scheduler)
        assert f"pid {PID}" in warning and "kill verdict None" in warning
        assert "the kill itself failed" in warning
        assert left_running == [(PID, PGID, None)]

    def test_an_unknown_group_is_named_as_such(self, scheduler, monkeypatch,
                                               left_running):
        _kill_answers(monkeypatch, "survived")

        scheduler._verify_command.kill_tree(_Proc(), None)
        [warning] = _warnings(scheduler)
        assert f"pid {PID} (process group ?)" in warning

    def test_a_detached_runs_record_keeping_it_is_said(self, scheduler, monkeypatch):
        from utils import detach
        monkeypatch.setattr(detach, "note_left_running", lambda *args: True)
        _kill_answers(monkeypatch, "unresolved")

        scheduler._verify_command.kill_tree(_Proc(), PGID)
        [warning] = _warnings(scheduler)
        assert "the run's record keeps it for status --abort" in warning

    def test_signalling_stays_kill_trees(self, scheduler, monkeypatch, left_running):
        """The request is unchanged: an owned child, no start time."""
        calls = _kill_answers(monkeypatch, "killed")
        proc = _Proc()

        scheduler._verify_command.kill_tree(proc, PGID, job=None)
        [(pid, kwargs)] = calls
        assert pid == PID
        assert kwargs["start_time"] is None and kwargs["child"] is proc
        assert kwargs["pgid"] == PGID and kwargs["job"] is None


class TestTheNoteClaimsOnlyWhatWasConfirmed:

    def test_killed_names_the_signals(self):
        note = wbs._verify_timeout_note(9, "killed", PID, PGID)
        assert "killed after 9s: SIGTERM then SIGKILL" in note
        assert note.rstrip().endswith("verdict killed)")

    def test_gone_says_it_had_stopped(self):
        note = wbs._verify_timeout_note(9, "gone", PID, PGID)
        assert "had already stopped" in note
        assert "killed" not in note and "SIGKILL" not in note
        assert note.rstrip().endswith("verdict gone)")

    @pytest.mark.parametrize("verdict", [*UNCONFIRMED, None])
    def test_an_unconfirmed_stop_names_the_pid_and_claims_no_kill(self, verdict):
        note = wbs._verify_timeout_note(9, verdict, PID, PGID)
        assert note.startswith("\n(timed out after 9s: ")
        assert "killed" not in note and "SIGTERM then SIGKILL" not in note
        assert f"pid {PID}, process group {PGID}" in note
        assert "may still be running" in note
        assert note.rstrip().endswith(f"verdict {verdict})")

    def test_an_unknown_group_is_left_out(self):
        note = wbs._verify_timeout_note(9, "survived", PID, None)
        assert f"(pid {PID})" in note and "process group" not in note


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
class TestThroughTheRunner:
    """The real runner and a real command; only the verdict is faked, after
    the real kill ran."""

    def _timed_out(self, scheduler, monkeypatch, verdict):
        real_kill = process_control.kill_tree

        def kill_then_report(pid, **kwargs):
            real_kill(pid, **kwargs)
            if isinstance(verdict, BaseException):
                raise verdict
            return verdict

        monkeypatch.setattr(write_verify._process_control, "kill_tree", kill_then_report)
        monkeypatch.setattr(scheduler._verify_command, "env", lambda: None)
        exit_code, tail = scheduler._verify_command.run("echo started; exec sleep 30", 1)
        assert exit_code == wbs.VERIFY_EXIT_TIMEOUT
        assert "started" in tail
        return tail

    def test_a_raised_kill_is_reported_unconfirmed(self, scheduler, monkeypatch,
                                                   left_running):
        tail = self._timed_out(scheduler, monkeypatch, RuntimeError("boom"))
        assert "killed after" not in tail and "SIGTERM then SIGKILL" not in tail
        assert "could not be confirmed stopped: the kill itself failed" in tail
        assert tail.rstrip().endswith("verdict None)")
        [warning] = _warnings(scheduler)
        assert "kill verdict None" in warning
        [(pid, _pgid, verdict)] = left_running
        assert verdict is None and isinstance(pid, int)

    def test_a_survived_over_the_commands_own_zombie_is_killed(
            self, scheduler, monkeypatch, left_running):
        """The real kill stopped the command; its unreaped zombie and an empty
        group confirm it (``process_control.own_child_stopped``, unfaked)."""
        tail = self._timed_out(scheduler, monkeypatch, "survived")
        assert "killed after 1s: SIGTERM then SIGKILL" in tail
        assert tail.rstrip().endswith("verdict killed)")
        assert _warnings(scheduler) == []
        assert left_running == []

    def test_the_command_goes_through_the_recorded_launcher(self, scheduler,
                                                            monkeypatch):
        """``popen_tree`` records a detached worker's launches, which is what
        lets ``detach.note_left_running`` keep the command for a later abort."""
        launched = []
        real = process_control.popen_tree

        def recorded(args, **kwargs):
            launched.append(args)
            return real(args, **kwargs)

        monkeypatch.setattr(write_verify._process_control, "popen_tree", recorded)
        monkeypatch.setattr(scheduler._verify_command, "env", lambda: None)
        assert scheduler._verify_command.run("exit 0", 5) == (0, "")
        assert launched == ["exit 0"]
