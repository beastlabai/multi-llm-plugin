#!/usr/bin/env python3
"""A signal handler never prints through a buffered stream it may interrupt.

Codex racing run 6 (2026-10-06): the tool server aborted a detached implement
run when its host left mid-call. ``status --abort`` signals the worker's whole
process group, which holds ``uv run`` *and* its python child, and ``uv run``
forwards the SIGTERM it receives to that child — so the worker gets the signal
twice, microseconds apart. The handler's ``print(..., file=sys.stderr)`` was
interrupted by the second copy inside ``BufferedWriter``'s flush (which runs
pending signal handlers after each raw write, with the buffer's lock held); the
nested handler printed through the same buffer and raised ``RuntimeError:
reentrant call inside <_io.BufferedWriter name='<stderr>'>``. The run ended in a
35-line traceback with ``MULTI_LLM_RESULT state: error`` instead of the
``aborted`` record that names the way forward. The same happens with a single
signal whenever the main thread is inside a stderr write when it lands.

Every test here drives the production handler in a really detached worker.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from utils import detach, process_control, status_file  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent
POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="POSIX signals and a real detached worker"
)

#: A worker that drives the production RunHandle; ``{prelude}`` runs once the
#: handle is started, then the worker waits to be signalled.
WORKER = '''
import io, os, signal, sys, time
sys.path.insert(0, {skill!r})
from utils import detach, process_control

PHASE = {phase!r}
process_control.enter_detached_worker(detach.log_path(PHASE))
handle = detach.RunHandle(
    PHASE, run_id={run_id!r}, mode="implement", plan_file={plan!r}, host="test-host",
)
handle.start()


class SignallingRaw(io.RawIOBase):
    """fd 2, plus a SIGTERM to this process when a write carries *trigger*.

    The signal is delivered while ``sys.stderr``'s BufferedWriter holds its
    lock (the write is inside its flush), which is where a real one lands
    when the main thread or the handler itself is printing.
    """

    def __init__(self, trigger):
        self.trigger = trigger

    def writable(self):
        return True

    def write(self, data):
        data = bytes(data)
        written = os.write(2, data)
        if self.trigger in data:
            self.trigger = b"\\0never\\0"
            os.kill(os.getpid(), signal.SIGTERM)
        return written


def signalling_stderr(trigger):
    sys.stderr = io.TextIOWrapper(
        io.BufferedWriter(SignallingRaw(trigger)), encoding="utf-8",
        line_buffering=True,
    )


print("worker up", flush=True)
{prelude}
deadline = time.time() + 30
while time.time() < deadline:
    time.sleep(0.05)
handle.finish(0)
sys.exit(0)
'''


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    return plan


def _phase(plan: Path) -> Path:
    phase = plan.parent / plan.stem / "implement"
    phase.mkdir(parents=True, exist_ok=True)
    return phase


def _spawn(tmp_path: Path, prelude: str, run_id: str) -> tuple:
    plan = _plan(tmp_path)
    phase = _phase(plan)
    script = tmp_path / f"worker-{run_id}.py"
    script.write_text(WORKER.format(
        skill=str(SKILL_ROOT), phase=str(phase), plan=str(plan), run_id=run_id,
        prelude=textwrap.dedent(prelude),
    ), encoding="utf-8")
    env = dict(os.environ)
    env[detach.WORKER_ENV] = "1"
    env["MULTI_LLM_HEARTBEAT_INTERVAL_S"] = "0.4"
    pid = process_control.spawn_detached(
        [sys.executable, str(script)], cwd=str(tmp_path), env=env,
        log_path=str(detach.log_path(str(phase))),
    )
    _await(lambda: status_file.read_run(str(phase)).get("state")
           not in (None, "starting"))
    return pid, phase


def _await(predicate, timeout: float = 20.0):
    deadline = time.time() + timeout
    value = predicate()
    while not value and time.time() < deadline:
        time.sleep(0.05)
        value = predicate()
    return value


def _terminal(phase: Path) -> dict:
    assert _await(lambda: status_file.read_run(str(phase)).get("state")
                  in status_file.TERMINAL_RUN_STATES), status_file.read_run(str(phase))
    return status_file.read_run(str(phase))


def _log(phase: Path) -> str:
    return Path(detach.log_path(str(phase))).read_text(encoding="utf-8", errors="replace")


@POSIX_ONLY
class TestTheWorkerAbortHandler:

    def test_a_signal_landing_inside_a_stderr_write_still_records_aborted(
        self, tmp_path
    ):
        """The main thread is printing to stderr when SIGTERM lands."""
        pid, phase = _spawn(tmp_path, '''
            signalling_stderr(b"TRIGGER")
            print("TRIGGER: the main thread is inside a stderr write", file=sys.stderr)
        ''', "run-main-write")
        run = _terminal(phase)
        log = _log(phase)
        assert "reentrant call" not in log, log
        assert "Traceback" not in log, log
        assert run["state"] == "aborted", run
        assert run["exit_code"] == 3
        assert "SIGTERM" in str(run["error"])
        assert log.count("Received SIGTERM: aborting this detached run.") == 1, log
        assert _await(lambda: not process_control.process_alive(pid))

    def test_the_copy_uv_forwards_during_the_announcement_is_ignored(self, tmp_path):
        """Run 6's exact shape: the second SIGTERM lands while the handler's
        own announcement is being written."""
        pid, phase = _spawn(tmp_path, '''
            signalling_stderr(b"Received SIGTERM")
        ''', "run-announce")
        os.kill(pid, signal.SIGTERM)
        run = _terminal(phase)
        log = _log(phase)
        assert "reentrant call" not in log, log
        assert "Traceback" not in log, log
        assert run["state"] == "aborted", run
        assert run["exit_code"] == 3
        assert _await(lambda: not process_control.process_alive(pid))

    def test_a_second_signal_while_the_abort_runs_is_ignored(self, tmp_path):
        """One abort, announced once, whatever else arrives meanwhile: the
        sweep is where a forwarded copy or a repeated ``--abort`` lands."""
        pid, phase = _spawn(tmp_path, '''
            _real_sweep = detach._sweep_live_clis

            def _sweep_with_a_second_signal():
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.3)
                _real_sweep()

            detach._sweep_live_clis = _sweep_with_a_second_signal
        ''', "run-second")
        os.kill(pid, signal.SIGTERM)
        run = _terminal(phase)
        assert _await(lambda: not process_control.process_alive(pid))
        log = _log(phase)
        assert run["state"] == "aborted", run
        assert run["exit_code"] == 3
        assert log.count("Received SIGTERM: aborting this detached run.") == 1, log
        assert "Traceback" not in log, log


@POSIX_ONLY
class TestTheForegroundSigtermHandler:
    """A foreground run turns SIGTERM into ``SystemExit`` so ``main``'s
    ``finally`` restores codex's config and prints the result line. The copy
    ``uv run`` forwards arrives a moment later and must not cut that
    ``finally`` short."""

    def test_a_second_sigterm_during_the_cleanup_is_ignored(self, tmp_path):
        script = tmp_path / "fg.py"
        script.write_text(textwrap.dedent(f'''
            import os, signal, sys
            sys.path.insert(0, {str(SKILL_ROOT)!r})
            import multi_llm
            multi_llm._install_foreground_sigterm()
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                import time
                time.sleep(5)
            finally:
                os.kill(os.getpid(), signal.SIGTERM)
                import time
                time.sleep(0.2)
                print("cleanup finished", flush=True)
        '''), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=30)
        assert "cleanup finished" in proc.stdout, proc.stderr
        assert proc.returncode == 128 + signal.SIGTERM

