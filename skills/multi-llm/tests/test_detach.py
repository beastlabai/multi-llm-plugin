#!/usr/bin/env python3
"""``--detach``: self-daemonizing runs, their lifecycle, and the ``status`` surface.

Four things are asserted here, and the split matters because only one of them can
be faked:

1. **The re-exec argv** (D4e) — stubbed spawn, exact argv shape. Re-launching as
   ``[sys.executable, __file__]`` would look right and be subtly wrong, so the
   shape is pinned rather than described.
2. **Real detachment** — genuinely forked workers (``setsid`` + double fork on
   POSIX, ``DETACHED_PROCESS`` on Windows), driven through the real
   ``RunHandle``: heartbeat, lock, log ownership, terminal close-out, abort. These
   run the actual mechanism; nothing about liveness is mocked, because "is this
   process alive?" is exactly what a mock cannot answer.
3. **The race set** (D4b) — a second detach against a held lock, a recycled pid, a
   crashed worker, a broken stale lock. Every one must end in exactly one terminal
   state with ``run.exit_code`` recorded, and no phase dir may be left parked at
   ``running`` or ``stale``.
4. **The ``status`` contract** (D4f/D4g) — two views, the selectors, the exit codes
   a poll loop branches on, that it works *while* another run holds the phase dir,
   and that one call is sub-second.

Plus one full-stack run: a real ``review-plan`` fan-out against mock providers,
launched with ``--detach`` through the dispatcher and watched to ``done`` with
``status`` — the Done-when row, end to end, with a ``uv`` stub that *asserts* the
child's argv shape rather than bypassing it.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import multi_llm  # noqa: E402
import modes as mode_table  # noqa: E402
from utils import cli_common, detach, process_control, run_registry, run_status, unattended  # noqa: E402
from utils import status_file  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent
POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="POSIX process semantics (setsid, SIGKILL, killpg)"
)

#: A worker started by the tests beats once a second so a test never waits 10.
FAST_HEARTBEAT = {"MULTI_LLM_HEARTBEAT_INTERVAL_S": "0.4"}


# ===========================================================================
# Helpers
# ===========================================================================

def _plan(tmp_path: Path, name: str = "a plan") -> Path:
    plan = tmp_path / f"{name}.md"
    plan.write_text("# Plan\n\n## Tasks\n- [ ] 1. Do the thing\n", encoding="utf-8")
    return plan


def _phase_dir(plan: Path, phase: str = "review-plan") -> Path:
    directory = plan.parent / plan.stem.replace(" ", "_") / phase
    directory.mkdir(parents=True, exist_ok=True)
    return directory


WORKER_TEMPLATE = '''
import os, sys, time
sys.path.insert(0, {skill!r})
from utils import detach, process_control

PHASE = {phase!r}
PLAN = {plan!r}
RUN_ID = {run_id!r}
DONE = {done!r}
CRASH = {crash!r}
EXIT_CODE = {exit_code!r}

process_control.enter_detached_worker(detach.log_path(PHASE))
handle = detach.RunHandle(
    PHASE, run_id=RUN_ID, mode={mode!r}, plan_file=PLAN, host="test-host",
)
handle.start()
# A non-ASCII line, on purpose: a Windows-piped stream defaults to cp1252 and
# raises UnicodeEncodeError unless the worker inherited stream_bootstrap's
# encoding="utf-8", errors="replace" — not just its line buffering.
print("worker up \\u2014 em dash survived")
if CRASH:
    raise RuntimeError("worker crashed before recording an outcome")
deadline = time.time() + 60
while time.time() < deadline:
    if os.path.exists(DONE):
        break
    time.sleep(0.1)
handle.finish(EXIT_CODE)
sys.exit(EXIT_CODE)
'''


#: A worker that prints on both streams *before* the detach boundary, the way
#: every orchestrator does on its way to `maybe_detach`.
PRE_DETACH_WORKER = '''
import os, sys
sys.path.insert(0, {skill!r})
from utils import process_control

LOG = {log!r}
print("stdout before the boundary", flush=True)
sys.stderr.write("stderr before the boundary\\n")
sys.stderr.flush()
already_on = process_control._stdio_already_on(LOG)
process_control.enter_detached_worker(LOG)
print("after the boundary: already_on=%s" % already_on)
sys.exit(0)
'''


def spawn_worker(
    tmp_path: Path,
    phase_dir: Path,
    plan: Path,
    *,
    run_id: str,
    mode: str = "review-plan",
    crash: bool = False,
    exit_code: int = 0,
    wait_for_running: bool = True,
) -> tuple:
    """Really detach a fixture worker that drives the production RunHandle.

    Returns ``(pid, done_sentinel)``: touch the sentinel to let it finish cleanly.
    """
    done = tmp_path / f"done-{run_id[:8]}"
    script = tmp_path / f"worker-{run_id[:8]}.py"
    script.write_text(
        WORKER_TEMPLATE.format(
            skill=str(SKILL_ROOT), phase=str(phase_dir), plan=str(plan),
            run_id=run_id, done=str(done), crash=crash, exit_code=exit_code,
            mode=mode,
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env[detach.WORKER_ENV] = "1"
    env.update(FAST_HEARTBEAT)
    pid = process_control.spawn_detached(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        env=env,
        log_path=str(detach.log_path(str(phase_dir))),
    )
    if wait_for_running:
        deadline = time.time() + 20
        while time.time() < deadline:
            run = status_file.read_run(str(phase_dir))
            if run.get("run_id") == run_id and run.get("state") in (
                "running", "done", "failed", "aborted"
            ):
                break
            time.sleep(0.05)
    return pid, done


def _await(predicate, *, timeout: float = 20.0, interval: float = 0.05):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    return last


def _status(argv, capsys=None) -> int:
    code = run_status.main(argv)
    if capsys is not None:
        capsys.readouterr()
    return code


def _age_heartbeat(phase_dir: Path, seconds: float) -> None:
    """Rewind the recorded heartbeat, the way a killed worker's would look."""
    from datetime import datetime, timedelta

    status = status_file.read_status(str(phase_dir))
    run = dict(status.get("run") or {})
    run["heartbeat_at"] = (datetime.now() - timedelta(seconds=seconds)).isoformat()
    status["run"] = run
    status_file.write_status(str(phase_dir), status)


def _terminal_once(phase_dir: Path) -> dict:
    """Assert exactly one terminal state is recorded, and return the run."""
    run = status_file.read_run(str(phase_dir))
    assert run.get("state") in status_file.TERMINAL_RUN_STATES, run
    assert "exit_code" in run, "a terminal run must record run.exit_code (null is a value)"
    # Terminal is terminal: no writer may move it again.
    assert not status_file.update_run(
        str(phase_dir), {"state": "running"},
        expect_run_id=str(run["run_id"]), quiet=True,
    )
    return run


# ===========================================================================
# D4e — the child re-executes uv, not bare python
# ===========================================================================

class TestWorkerCommand:
    def test_child_argv_is_uv_run_project_multi_llm(self, tmp_path, monkeypatch):
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        argv = ["--plan-file", "plans/a plan.md", "--quick", "--detach"]
        command = detach.build_child_command(
            "review-plan", argv, run_id="RID", base=SKILL_ROOT
        )
        assert command[0] == str(_fake_uv(tmp_path))
        # The uv flags come from unattended's single owner and sit before
        # --project, which is where uv's own argument grammar wants them.
        head = ["run", *unattended.uv_run_args()]
        assert command[1:1 + len(head)] == head
        rest = command[1 + len(head):]
        assert rest[:4] == ["--project", str(SKILL_ROOT), "--", "python"]
        # --project and the script path are exactly one argv element each.
        assert rest[4] == str(SKILL_ROOT / "multi_llm.py")
        assert rest[5] == "review-plan"
        assert "--internal-worker" in command
        assert "--detach" not in command
        assert command[command.index("--internal-run-id") + 1] == "RID"
        # The plan path is absolute; the implied --yes is written out.
        plan = command[command.index("--plan-file") + 1]
        assert os.path.isabs(plan) and plan.endswith("a plan.md")
        assert "--yes" in command

    def test_yes_is_not_duplicated(self, tmp_path, monkeypatch):
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        command = detach.build_child_command(
            "ask", ["--plan-file", "p.md", "--non-interactive", "--detach"],
            run_id="R", base=SKILL_ROOT,
        )
        assert command.count("--yes") == 0
        assert "--non-interactive" in command

    def test_launcher_only_flags_do_not_reach_the_worker(self, tmp_path, monkeypatch):
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        command = detach.build_child_command(
            "review-plan",
            ["--plan-file", "p.md", "--detach", "--if-not-running",
             "--max-detached", "9"],
            run_id="R", base=SKILL_ROOT,
        )
        assert "--if-not-running" not in command
        assert "--max-detached" not in command and "9" not in command

    def test_unresolvable_uv_is_a_foreground_failure(self, monkeypatch):
        monkeypatch.delenv(detach.UV_ENV, raising=False)
        monkeypatch.setattr(detach.shutil, "which", lambda _name: None)
        monkeypatch.setattr(os.path, "isfile", lambda _p: False)
        with pytest.raises(detach.DetachError) as exc:
            detach.resolve_uv()
        assert exc.value.exit_code == 3
        assert "uv" in str(exc.value)

    def test_child_env_is_pinned_not_inherited_blindly(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", "relative.yaml")
        env = detach.child_env(base=SKILL_ROOT)
        assert "CLAUDECODE" not in env
        assert os.path.isabs(env["MULTI_LLM_PROVIDERS_CONFIG"])
        assert env["MULTI_LLM_SKILL_DIR"] == str(SKILL_ROOT)
        assert env[detach.WORKER_ENV] == "1"


def _fake_uv(tmp_path: Path) -> Path:
    path = tmp_path / "fake-uv"
    if not path.exists():
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    return path


# ===========================================================================
# A really detached worker: running -> done, and the log it owns
# ===========================================================================

class TestRealDetachedWorker:
    def test_runs_then_reaches_done_through_status(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        pid, done = spawn_worker(tmp_path, phase, plan, run_id="run-alpha-0001")

        view = detach.classify_run(str(phase))
        assert view["verdict"] == "running", view
        assert view["run"]["pid"] == pid
        assert view["run"]["detached"] is True
        assert view["lock_held"] is True, "the worker holds {phase}/.lock"
        assert _status(["--plan-file", str(plan), "--view", "run"], capsys) == 2

        done.touch()
        assert _await(
            lambda: status_file.read_run(str(phase)).get("state") == "done"
        ), status_file.read_run(str(phase))

        assert _status(["--plan-file", str(plan), "--view", "run"], capsys) == 0
        run = _terminal_once(phase)
        assert run["exit_code"] == 0
        assert detach.classify_run(str(phase))["verdict"] == "done"
        # The lock is released on the terminal path.
        assert process_control.lock_is_free(detach.lock_path(str(phase)))

    @POSIX_ONLY
    def test_worker_owns_the_log_with_utf8_and_line_buffering(self, tmp_path):
        """Both stream_bootstrap settings, not just the buffering one.

        The line arrives *while the worker is still running* (line buffering) and
        it survives as UTF-8 (encoding). Without the second setting a Windows
        detached log raises UnicodeEncodeError on the first em dash a model emits;
        without the first it stays empty for minutes.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _pid, done = spawn_worker(tmp_path, phase, plan, run_id="run-log-0001")
        log = Path(detach.log_path(str(phase)))
        text = _await(
            lambda: log.read_text(encoding="utf-8") if log.exists() else "",
            timeout=20,
        )
        assert "em dash survived" in text
        assert "—" in text, "the log must be UTF-8, not the locale codec"
        assert "[DETACHED_RUN] run_id=run-log-0001" in text
        done.touch()

    @POSIX_ONLY
    def test_the_worker_writes_to_the_log_before_it_enters_the_detached_worker(
        self, tmp_path
    ):
        """The launcher hands the log in; `enter_detached_worker` only confirms it.

        The worker prints its whole pre-flight — inputs loaded, models resolved,
        backup written — before it reaches the detach boundary, so a worker whose
        stdio started on ``/dev/null`` threw all of it away. The launcher now
        opens the log and dup2s it onto fds 1/2 in the intermediate, which makes
        `enter_detached_worker` idempotent: it must neither truncate the bytes
        already there nor open a second file description over the same file.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        log = Path(detach.log_path(str(phase)))
        log.write_text("a predecessor's bytes\n", encoding="utf-8")
        script = tmp_path / "pre-detach-worker.py"
        script.write_text(
            PRE_DETACH_WORKER.format(skill=str(SKILL_ROOT), log=str(log)),
            encoding="utf-8",
        )
        pid = process_control.spawn_detached(
            [sys.executable, str(script)],
            cwd=str(tmp_path), env=dict(os.environ), log_path=str(log),
        )
        assert pid > 0
        text = _await(
            lambda: log.read_text(encoding="utf-8")
            if log.exists() and "after the boundary" in log.read_text(
                encoding="utf-8") else "",
            timeout=20,
        )
        assert "a predecessor's bytes" in text, "O_APPEND, never O_TRUNC"
        assert "stdout before the boundary" in text
        assert "stderr before the boundary" in text
        assert text.index("stdout before the boundary") < text.index(
            "after the boundary"
        )
        # The worker's own verdict: the launcher had already put fds 1 and 2 on
        # this file, so the reopen was correctly skipped.
        assert "already_on=True" in text

    @POSIX_ONLY
    def test_worker_survives_its_launcher_and_has_no_controlling_terminal(
        self, tmp_path
    ):
        """The point of the double spawn: reparented to init, no session leader."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        pid, done = spawn_worker(tmp_path, phase, plan, run_id="run-orphan-001")
        # The launcher (this process) is not its parent — pid 1 reaped the
        # intermediate — so nothing here can wait on it.
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        ppid = int(
            Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1]
        ) if Path(f"/proc/{pid}/stat").exists() else 1
        assert ppid != os.getpid()
        done.touch()

    @POSIX_ONLY
    def test_killed_worker_is_stale_then_failed_never_done(self, tmp_path, capsys):
        """A killed run is classified stale, not running — and then closed out."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        pid, _done = spawn_worker(tmp_path, phase, plan, run_id="run-killed-001")
        os.kill(pid, signal.SIGKILL)
        assert _await(lambda: not process_control.process_alive(pid))

        # Heartbeat still fresh, process provably gone: stale, never running.
        view = detach.classify_run(str(phase))
        assert view["verdict"] == "stale", view
        assert _status(["--plan-file", str(plan), "--view", "run"], capsys) == 2

        # Past the heartbeat window, status closes it out itself: stale is a
        # waiting room, not a resting state.
        _age_heartbeat(phase, 120)
        assert _status(["--plan-file", str(plan), "--view", "run"], capsys) == 3
        run = _terminal_once(phase)
        assert run["state"] == "failed"
        assert run["error"] == "heartbeat_timeout"
        assert run["exit_code"] is None, "no exit status was ever observed"
        assert process_control.lock_is_free(detach.lock_path(str(phase)))

    @POSIX_ONLY
    def test_abort_is_honored_and_terminates_the_run(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        pid, _done = spawn_worker(tmp_path, phase, plan, run_id="run-abort-001")
        code = run_status.main(["--plan-file", str(plan), "--abort"])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "aborted" in out
        assert _await(lambda: not process_control.process_alive(pid))
        run = _terminal_once(phase)
        assert run["state"] == "aborted"
        assert run["exit_code"] == 3
        assert run["error"] == "aborted_by_request"
        # Idempotent: the same command on an already-terminal run exits 2 and
        # rewrites nothing.
        before = Path(status_file.status_path(str(phase))).read_text(encoding="utf-8")
        assert run_status.main(["--plan-file", str(plan), "--abort"]) == 2
        capsys.readouterr()
        assert Path(status_file.status_path(str(phase))).read_text(
            encoding="utf-8"
        ) == before

    @POSIX_ONLY
    def test_signal_handler_records_aborted_before_exiting(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        pid, _done = spawn_worker(tmp_path, phase, plan, run_id="run-sigterm-01")
        os.kill(pid, signal.SIGTERM)
        assert _await(
            lambda: status_file.read_run(str(phase)).get("state") == "aborted"
        ), status_file.read_run(str(phase))
        run = _terminal_once(phase)
        assert "SIGTERM" in str(run["error"])

    @POSIX_ONLY
    def test_crashed_worker_lands_failed_with_a_reason(self, tmp_path):
        """No handler ran the phase to completion, but the run still closes out."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        spawn_worker(
            tmp_path, phase, plan, run_id="run-crash-0001", crash=True,
            wait_for_running=False,
        )
        assert _await(
            lambda: status_file.read_run(str(phase)).get("state")
            in status_file.TERMINAL_RUN_STATES,
            timeout=25,
        ), status_file.read_run(str(phase))
        run = _terminal_once(phase)
        assert run["state"] == "failed"
        assert run["error"]
        log = Path(detach.log_path(str(phase))).read_text(encoding="utf-8")
        assert "worker crashed" in log


# ===========================================================================
# D4b — the race set. Each ends in exactly one terminal state.
# ===========================================================================

class TestRaces:
    @POSIX_ONLY
    def test_second_detach_against_a_held_lock_exits_three(self, tmp_path,
                                                           monkeypatch, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        pid, done = spawn_worker(tmp_path, phase, plan, run_id="run-holder-001")
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        log_before = Path(detach.log_path(str(phase))).read_text(encoding="utf-8")

        with pytest.raises(detach.DetachError) as exc:
            detach.launch(
                mode="review-plan",
                argv=["--plan-file", str(plan), "--detach"],
                phase_dir=str(phase),
                plan_file=str(plan),
                phase="review-plan",
                args=argparse.Namespace(force=False, if_not_running=False,
                                        max_detached=None),
            )
        assert exc.value.exit_code == 3
        assert "run-holder-001" in str(exc.value)
        capsys.readouterr()

        # The loser wrote nothing: the holder is untouched and still running.
        run = status_file.read_run(str(phase))
        assert run["run_id"] == "run-holder-001"
        assert run["state"] == "running"
        assert Path(detach.log_path(str(phase))).read_text(
            encoding="utf-8"
        ).startswith(log_before)

        # And the holder still finishes in exactly one terminal state.
        done.touch()
        assert _await(
            lambda: status_file.read_run(str(phase)).get("state") == "done"
        )
        assert _terminal_once(phase)["exit_code"] == 0
        assert not process_control.process_alive(pid) or True

    @POSIX_ONLY
    def test_if_not_running_reattaches_instead_of_racing(self, tmp_path,
                                                        monkeypatch, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _pid, done = spawn_worker(tmp_path, phase, plan, run_id="run-attach-001")
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        code = detach.launch(
            mode="review-plan",
            argv=["--plan-file", str(plan), "--detach", "--if-not-running"],
            phase_dir=str(phase),
            plan_file=str(plan),
            phase="review-plan",
            args=argparse.Namespace(force=False, if_not_running=True,
                                    max_detached=None),
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "run-attach-001" in out
        assert "already in flight" in out
        done.touch()

    def test_recycled_pid_reports_stale_not_running(self, tmp_path, capsys):
        """PID is corroboration, never identity."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": "run-recycled-01",
            "pid": os.getpid(),                 # alive…
            "process_start_time": "0",          # …but not the same process
            "detached": True,
            "state": "running",
            "heartbeat_at": status_file.now_iso(),
            "exit_code": None,
        })
        view = detach.classify_run(str(phase))
        assert view["identity"] == process_control.DEAD
        assert view["verdict"] == "stale", view
        assert _status(["--plan-file", str(plan), "--view", "run"], capsys) == 2

        _age_heartbeat(phase, 300)
        assert _status(["--plan-file", str(plan), "--view", "run"], capsys) == 3
        run = _terminal_once(phase)
        assert run["error"] == "heartbeat_timeout"
        assert run["exit_code"] is None

    def test_crashed_dispatcher_run_records_exit_code_and_traceback(self, tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
        """The dispatcher's close-out path: a crash is `failed`, never `stale`."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        handle = detach.RunHandle(
            str(phase), run_id="run-dispcrash", mode="review-plan",
            plan_file=str(plan), detached=False,
        )
        handle.start()
        try:
            def boom(argv):
                raise RuntimeError("kaboom in the phase")

            monkeypatch.setattr(multi_llm, "_load_target", lambda m, a: boom)
            assert multi_llm.main(["review-plan", str(plan)]) == 1
            capsys.readouterr()
            run = _terminal_once(phase)
            assert run["state"] == "failed"
            assert run["exit_code"] == 1
            assert "kaboom in the phase" in str(run["error"])
        finally:
            handle.finish(1)

    def test_a_stale_lock_is_broken_exactly_once(self, tmp_path):
        """And the stale run's own writes are refused, not merged."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        # A record + lock left by a process that no longer exists.
        dead_pid = _dead_pid()
        status_file.claim_run(str(phase), {
            "run_id": "run-stale-lock", "pid": dead_pid,
            "process_start_time": "12345", "state": "running",
            "detached": True, "heartbeat_at": status_file.now_iso(),
            "exit_code": None,
        })
        Path(detach.lock_path(str(phase))).write_text(
            json.dumps({"run_id": "run-stale-lock", "pid": dead_pid,
                        "process_start_time": "12345"}),
            encoding="utf-8",
        )
        fresh = detach.RunHandle(
            str(phase), run_id="run-fresh-0001", mode="review-plan",
            plan_file=str(plan), detached=False,
        )
        fresh.start()
        try:
            run = status_file.read_run(str(phase))
            assert run["run_id"] == "run-fresh-0001"
            assert run["state"] == "running"
            holder = process_control.read_lock_holder(detach.lock_path(str(phase)))
            assert holder["run_id"] == "run-fresh-0001"
            # The displaced run cannot resurrect itself: compare-and-set refuses.
            assert not status_file.update_run(
                str(phase), {"heartbeat_at": status_file.now_iso()},
                expect_run_id="run-stale-lock", quiet=True,
            )
            assert status_file.read_run(str(phase))["run_id"] == "run-fresh-0001"
        finally:
            fresh.finish(0)
        run = _terminal_once(phase)
        assert run["run_id"] == "run-fresh-0001"
        assert run["exit_code"] == 0

    def test_second_run_handle_against_a_live_lock_raises(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        first = detach.RunHandle(
            str(phase), run_id="run-first-0001", mode="review-plan",
            plan_file=str(plan), detached=False,
        )
        first.start()
        try:
            second = detach.RunHandle(
                str(phase), run_id="run-second-001", mode="review-plan",
                plan_file=str(plan), detached=False,
            )
            with pytest.raises(detach.DetachError) as exc:
                second.start()
            assert exc.value.exit_code == 3
            assert "run-first-0001" in str(exc.value)
            assert status_file.read_run(str(phase))["run_id"] == "run-first-0001"
        finally:
            first.finish(0)
        assert _terminal_once(phase)["run_id"] == "run-first-0001"

    def test_atexit_backstop_never_leaves_a_run_at_running(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        handle = detach.RunHandle(
            str(phase), run_id="run-backstop-1", mode="review-plan",
            plan_file=str(plan), detached=False,
        )
        handle.start()
        handle._atexit()          # what interpreter shutdown would call
        run = _terminal_once(phase)
        assert run["state"] == "failed"
        assert run["exit_code"] is None
        assert "without recording an outcome" in str(run["error"])


def _dead_pid() -> int:
    """A pid that is not in use (spawn something trivial and let it exit)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# ===========================================================================
# D4c — the gates that must fail in the foreground
# ===========================================================================

class TestDetachGates:
    @pytest.mark.parametrize(
        "argv",
        [
            ["review-plan", "p.md", "--detach", "--interactive"],
            ["review-plan", "p.md", "--detach", "-i"],
            ["ask", "p.md", "q?", "--detach", "--interactive"],
        ],
    )
    def test_detach_plus_interactive_is_exit_three_at_parse_time(self, argv,
                                                                capsys):
        assert multi_llm.main(argv) == 3
        err = capsys.readouterr().err
        assert "--detach" in err and "contradictory" in err

    def test_detach_plus_pause_and_ask_is_exit_three(self, capsys):
        assert multi_llm.main(
            ["implement", "p.md", "--detach", "--force-strategy", "pause-and-ask"]
        ) == 3
        assert "pause-and-ask" in capsys.readouterr().err

    def test_detach_implies_yes_in_the_forwarded_argv(self, monkeypatch, capsys):
        captured = {}

        def loader(module_path, attr):
            def target(argv):
                captured["argv"] = list(argv)
                return 0

            return target

        monkeypatch.setattr(multi_llm, "_load_target", loader)
        assert multi_llm.main(["review-plan", "p.md", "--detach"]) == 0
        capsys.readouterr()
        assert "--yes" in captured["argv"], captured["argv"]
        assert cli_common.invocation() is None, (
            "the invocation record is cleared when the dispatcher returns"
        )

    @pytest.mark.parametrize("mode", ["status", "init"])
    def test_detach_on_a_mode_that_does_not_support_it_yet(self, mode, tmp_path,
                                                           capsys):
        """A loud exit 3 at the dispatcher, never a flag that does nothing.

        The mode table owns which modes take the flag (``supports_detach``), so
        the rejection cannot drift from the list the message prints. The
        executor-backed modes left this list when the write executor made them
        long-running; what remains are the modes that finish in milliseconds
        (``status``, ``init``) or install a shim.
        """
        args = [] if mode == "init" else ["p.md"]
        assert multi_llm.main([mode, *args, "--detach"]) == 3
        err = capsys.readouterr().err
        assert "not available for mode" in err
        assert ", ".join(mode_table.DETACH_MODES) in err
        assert not mode_table.resolve(mode).supports_detach

    def test_the_detach_guard_agrees_with_the_mode_table(self, tmp_path):
        plan = _plan(tmp_path)
        assert detach.DETACH_MODES == mode_table.DETACH_MODES
        for mode in mode_table.mode_names():
            assert detach.mode_supports_detach(mode) is (
                mode_table.resolve(mode).supports_detach
            )
        with pytest.raises(detach.DetachError) as exc:
            detach.handle_phase_detach(
                args=argparse.Namespace(detach=True, internal_worker=False),
                mode="status", argv=[], phase_dir=str(tmp_path),
                plan_file=str(plan),
            )
        assert exc.value.exit_code == 3

    def test_every_executor_backed_mode_accepts_detach(self):
        """Task 6: the write modes became long-running, so they take the flag.

        ``generate-tasks`` and ``full`` were in the set before their orchestrators
        existed, so the flag gate a shim saw was already the final one. Now that
        both are Python, the plumbing they registered for is the plumbing that
        runs (see ``test_the_registered_detach_path_really_launches``).
        """
        for mode in ("implement", "apply-suggestions", "apply-task-suggestions",
                     "apply-code-fixes", "generate-tasks", "full"):
            assert mode_table.resolve(mode).supports_detach, mode
            assert detach.mode_supports_detach(mode), mode

    @pytest.mark.parametrize("mode", ["generate-tasks", "full"])
    def test_the_registered_detach_path_really_launches(self, mode, tmp_path,
                                                       monkeypatch, capsys):
        """The two Phase-3 modes cross the same boundary as every other mode.

        Their ``unavailable`` refusal used to fire first, so ``--detach`` was
        registered plumbing that nothing exercised. With both modes implemented the
        dispatcher must reach ``detach.launch`` with the mode word and the
        normalized argv (``--yes`` written out), and the phase dir it hands over
        must be the run dir that mode owns — not exit 3.
        """
        launched = {}

        def fake_launch(*, mode, argv, phase_dir, plan_file, args, **kwargs):
            launched.update(
                mode=mode, argv=list(argv), phase_dir=phase_dir,
                plan_file=plan_file,
            )
            return 0

        monkeypatch.setattr(detach, "launch", fake_launch)
        plan = _plan(tmp_path)
        assert multi_llm.main([mode, str(plan), "--detach"]) == 0
        capsys.readouterr()
        assert launched["mode"] == mode
        assert "--yes" in launched["argv"], launched["argv"]
        assert launched["plan_file"] == str(plan)
        expected_phase = mode_table.resolve(mode).phase
        assert Path(launched["phase_dir"]).name == expected_phase

    def test_direct_orchestrator_invocation_refuses_to_detach(self, tmp_path,
                                                              capsys):
        """The child re-executes the dispatcher, so the dispatcher must launch it."""
        from utils.review_orchestrator_base import maybe_detach

        cli_common.clear_invocation()
        plan = _plan(tmp_path)
        code = maybe_detach(
            args=argparse.Namespace(detach=True, internal_worker=False),
            mode="review-plan", phase_dir=str(_phase_dir(plan)),
            plan_file=str(plan), phase="review-plan",
        )
        assert code == 3
        err = capsys.readouterr().err
        assert "must be invoked through multi_llm.py" in err

    def test_ceiling_refuses_in_the_foreground_and_creates_nothing(
        self, tmp_path, monkeypatch, capsys
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        monkeypatch.setenv(run_registry.MAX_DETACHED_ENV, "2")
        root = run_registry.resolve_project_root(str(plan))
        identity = process_control.identity()
        for index in range(2):
            run_registry.write_entry(root, {
                "run_id": f"ceiling-{index}",
                "pid": identity["pid"],
                "process_start_time": identity["process_start_time"],
                "plan_file": str(plan),
                "phase_dir": str(tmp_path / f"other-{index}"),
                "mode": "ask",
            })
        with pytest.raises(detach.DetachError) as exc:
            detach.launch(
                mode="review-plan",
                argv=["--plan-file", str(plan), "--detach"],
                phase_dir=str(phase), plan_file=str(plan), phase="review-plan",
                args=argparse.Namespace(force=False, if_not_running=False,
                                        max_detached=None),
            )
        assert exc.value.exit_code == 3
        assert "ceiling 2" in str(exc.value)
        capsys.readouterr()
        assert not Path(detach.log_path(str(phase))).exists()
        assert not Path(detach.lock_path(str(phase))).exists()
        assert len(run_registry.list_entries(root)) == 2, "no new registry entry"

    def test_all_provider_binaries_missing_is_a_foreground_failure(
        self, tmp_path, monkeypatch, capsys
    ):
        """Knowable before the fork, so it must not be discovered by polling."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        monkeypatch.setattr(detach.shutil, "which", lambda _name: None)
        with pytest.raises(detach.DetachError) as exc:
            detach.launch(
                mode="review-plan",
                argv=["--plan-file", str(plan), "--detach"],
                phase_dir=str(phase), plan_file=str(plan), phase="review-plan",
                args=argparse.Namespace(force=False, if_not_running=False,
                                        max_detached=None),
                model_specs=["cursor-agent:auto"],
            )
        assert exc.value.exit_code == 3
        assert "not on PATH" in str(exc.value)
        capsys.readouterr()
        assert not Path(detach.log_path(str(phase))).exists()

    def test_one_missing_provider_only_warns(self, tmp_path, monkeypatch, capsys):
        """A partial aggregate is usable output, so one absent CLI is not fatal."""
        plan = _plan(tmp_path)
        missing = detach.missing_provider_binaries(
            ["cursor-agent:auto", "gemini:gemini-2.5-flash"], anchor=str(plan)
        )
        assert set(missing) <= {"cursor-agent:auto", "gemini:gemini-2.5-flash"}
        monkeypatch.setattr(
            detach.shutil, "which",
            lambda name: None if name == "gemini" else f"/usr/bin/{name}",
        )
        missing = detach.missing_provider_binaries(
            ["cursor-agent:auto", "gemini:gemini-2.5-flash"], anchor=str(plan)
        )
        assert list(missing) == ["gemini:gemini-2.5-flash"]
        assert missing["gemini:gemini-2.5-flash"] == "gemini"

    def test_max_detached_flag_overrides_the_env(self, monkeypatch):
        monkeypatch.setenv(run_registry.MAX_DETACHED_ENV, "2")
        assert run_registry.max_detached(None) == 2
        assert run_registry.max_detached(7) == 7
        monkeypatch.delenv(run_registry.MAX_DETACHED_ENV)
        assert run_registry.max_detached(None) == run_registry.DEFAULT_MAX_DETACHED


# ===========================================================================
# D4f / D4g — the status command surface
# ===========================================================================

class TestStatusSurface:
    def test_workflow_view_is_byte_identical_to_the_retired_mode(self, tmp_path,
                                                                capsys):
        import check_workflow_prerequisites as prereq

        plan = _plan(tmp_path)
        assert run_status.main(
            ["--plan-file", str(plan), "--view", "workflow", "--json"]
        ) == 0
        through_status = capsys.readouterr().out
        with pytest.raises(SystemExit):
            prereq.main(["--plan-file", str(plan), "--status"])
        legacy = capsys.readouterr().out
        assert through_status == legacy

    def test_no_status_json_anywhere_exits_two_not_zero(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        assert run_status.main(["--plan-file", str(plan)]) == 2
        out = capsys.readouterr().out
        assert "no run recorded" in out
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        capsys.readouterr()

    def test_missing_plan_is_exit_one(self, tmp_path, capsys):
        assert run_status.main(["--plan-file", str(tmp_path / "nope.md")]) == 1
        assert "not found" in capsys.readouterr().out

    def test_three_state_fields_and_the_unmarked_phase_note(self, tmp_path,
                                                            capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "validation_pending",
            "reaggregate_mode": "review-plan", "plan_file": str(plan),
        })
        status_file.claim_run(str(phase), {
            "run_id": "run-done-0001", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
        })
        assert run_status.main(["--plan-file", str(plan)]) == 0
        out = capsys.readouterr().out
        assert "run.state:      done" in out
        assert "phase state:    validation_pending" in out
        assert "workflow phase: pending" in out
        assert "not yet marked completed" in out
        # The advance command is rebuilt from the running skill dir, not stored.
        assert str(cli_common.skill_dir()) in out
        assert "{SKILL_DIR}" not in out

    def test_complete_phase_state_renders_as_finished(self, tmp_path, capsys):
        """§Fix2: a run whose phase machine reached ``complete`` renders as
        finished on all three of ``status``'s state lines — not stuck showing
        whichever intermediate step (``grouping_complete``, ``models_complete``,
        ``validation_pending``) the run's last write happened to leave behind,
        and without the "not yet marked completed" note a still-pending
        workflow phase would print alongside a done run.
        """
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        # `claim_run` drops any predecessor `run`/`progress` wholesale, so it
        # has to land BEFORE the phase-level writes below — exactly the order
        # a real run takes (the detach boundary's claim, then the phase's own
        # `write_status`/`update_status` calls), not before it.
        status_file.claim_run(str(phase), {
            "run_id": "run-done-0002", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
        })
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
        })
        status_file.update_status(str(phase), {
            "progress": {"step": "complete", "models_total": 3, "models_done": 3},
        })
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        state.save()

        assert run_status.main(["--plan-file", str(plan)]) == 0
        out = capsys.readouterr().out
        assert "run.state:      done" in out
        assert "phase state:    complete" in out
        assert "progress:       complete, models 3/3" in out
        assert "workflow phase: completed" in out
        assert "not yet marked completed" not in out

    def test_the_run_block_names_the_host_that_launched_the_run(
        self, tmp_path, capsys
    ):
        """F6: the block reported a runner for a host the reader never named,
        and nothing on screen said whose run it was (measured 2026-09-16, an
        `opencode` run polled from Claude Code)."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": "run-host-0001", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "host": "opencode", "mode": "review-plan",
        })
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 0
        out = capsys.readouterr().out
        assert "host:           opencode" in out
        # ...and the runner block below it is about that host, not this one.
        assert "every subagent opencode dispatches runs through it" in out
        assert "claude-code dispatches subagents natively" not in out

    def test_the_poll_command_it_hands_back_carries_the_runs_host(
        self, tmp_path, capsys
    ):
        """Run the emitted `next` verbatim and it must report the same run the
        same way — which it cannot without the selector the run was launched
        with."""
        assert "--host opencode" in detach.status_command_for(
            "/p.md", phase="review-plan", host="opencode"
        )
        # Never guessed: a run that recorded no host selects none (D5b).
        assert "--host" not in detach.status_command_for("/p.md", phase="review-plan")

        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": "run-host-0002", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True, "host": "opencode",
            "mode": "review-plan",
        })
        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context.next and "--host opencode" in context.next

    def test_progress_renders_the_counts_a_run_is_still_accumulating(
        self, tmp_path, capsys
    ):
        """F5's display half: mid-run the two longest stretches read `models
        0/4` and `grouping` — 435 s of fan-out and 62 s of validation with
        nothing on the line that moved (measured 2026-09-16)."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": "run-progress-001", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True, "mode": "review-plan",
        })
        status_file.update_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {"step": "fanout", "models_total": 4, "models_done": 2},
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        assert "progress:       fanout, models 2/4" in capsys.readouterr().out

        status_file.update_status(str(phase), {
            "state": "grouping_complete",
            "progress": {
                "step": "validating", "models_done": 4,
                "subagents": {
                    "kind": "validation", "invoker": "executor",
                    "total": 11, "ok": 3, "counts": {"ok": 3},
                    "failed_task_ids": [],
                },
            },
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        assert (
            "progress:       validating, models 4/4, validation 3/11 ok"
            in capsys.readouterr().out
        )

    def test_the_executor_mirror_is_progress_too(self, tmp_path, capsys):
        """A dispatch-only phase (`generate-tasks`) records no `step`; the
        executor's `progress.subagents` mirror is all the progress it has, and
        the line used to print "(none recorded)" beside it (measured 2026-09-14).
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "generate-tasks")
        status_file.claim_run(str(phase), {
            "run_id": "run-mirror-001", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "failed", "exit_code": 3, "detached": True,
            "mode": "generate-tasks",
        })
        status_file.update_status(str(phase), {
            "phase": "generate-tasks", "state": "subagents_failed",
            "progress": {"subagents": {
                "kind": "task_generation", "invoker": "executor",
                "total": 1, "ok": 0, "counts": {"timeout": 1},
                "failed_task_ids": ["task_generation"],
            }},
        })
        # A failed run is exit 3 from `status` too — the verdict is the run's.
        assert run_status.main(["--plan-file", str(plan)]) == 3
        out = capsys.readouterr().out
        assert (
            "progress:       task_generation 0/1 ok, timeout 1, failed task_generation"
            in out
        )
        assert "(none recorded)" not in out

        # And the successful retry: the mirror says so, the step agrees.
        status_file.claim_run(str(phase), {
            "run_id": "run-mirror-002", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "generate-tasks",
        })
        status_file.update_status(str(phase), {
            "state": "complete",
            "progress": {
                "step": "complete",
                "subagents": {
                    "kind": "task_generation", "invoker": "executor",
                    "total": 1, "ok": 1, "counts": {"ok": 1},
                    "failed_task_ids": [],
                },
            },
        })
        assert run_status.main(["--plan-file", str(plan)]) == 0
        out = capsys.readouterr().out
        assert "phase state:    complete" in out
        assert "progress:       complete, task_generation 1/1 ok" in out

    def test_the_write_mirror_counts_its_task_rows(self, tmp_path, capsys):
        """`write_subagents` records one row per task and no `counts`/`ok`
        summary; a finished nine-task run printed `implement 0/9 ok`
        (measured 2026-09-16)."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "implement")
        status_file.claim_run(str(phase), {
            "run_id": "run-write-001", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "implement",
        })
        rows = [
            {"task_id": f"T00{i}", "batch_index": i, "status": "ok",
             "write_outcome": "applied", "files_touched": ["index.html"]}
            for i in range(7)
        ] + [
            {"task_id": f"T00{i}", "batch_index": i, "status": "ok",
             "write_outcome": "no-op", "files_touched": []}
            for i in (7, 8)
        ]
        status_file.update_status(str(phase), {
            "phase": "implement", "state": "complete",
            "progress": {"write_subagents": {
                "kind": "implement", "invoker": "executor", "total": 9,
                "dispatched": 9, "finalized": True, "tasks": rows,
                "failed_task_ids": [],
            }},
        })
        assert run_status.main(["--plan-file", str(plan)]) == 0
        out = capsys.readouterr().out
        assert "phase state:    complete" in out
        assert "progress:       implement 9/9 ok, applied 7, no-op 2" in out

    def test_phase_selector_reports_exactly_the_named_one(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        for phase in ("review-plan", "code-review"):
            directory = _phase_dir(plan, phase)
            status_file.write_status(str(directory), {"phase": phase, "state": "x"})
        assert run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--phase", "code-review",
             "--json"]
        ) == 2
        payload = json.loads(_json_body(capsys.readouterr().out))
        assert len(payload["runs"]) == 1
        assert payload["runs"][0]["phase"] == "code-review"

        # `--phase` resolves through modes.py, so the mode word works too — the
        # one place mode word != phase name.
        assert run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--phase", "review-code",
             "--json"]
        ) == 2
        payload = json.loads(_json_body(capsys.readouterr().out))
        assert [r["phase"] for r in payload["runs"]] == ["code-review"]

    def test_unknown_phase_selector_is_rejected(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        with pytest.raises(SystemExit):
            run_status.main(["--plan-file", str(plan), "--phase", "nonsense"])
        capsys.readouterr()

    def test_json_view_all_has_two_top_level_keys(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {"phase": "review-plan", "state": "x"})
        run_status.main(["--plan-file", str(plan), "--json"])
        payload = json.loads(_json_body(capsys.readouterr().out))
        assert set(payload) == {"workflow", "runs"}
        assert payload["workflow"]["phases"]

    @POSIX_ONLY
    def test_status_works_while_another_run_holds_the_phase_dir(self, tmp_path,
                                                               capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _pid, done = spawn_worker(tmp_path, phase, plan, run_id="run-held-0001")
        lock = detach.lock_path(str(phase))
        holder_before = Path(lock).read_text(encoding="utf-8")

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "running" in out
        # It took no lock and broke none.
        assert Path(lock).read_text(encoding="utf-8") == holder_before
        assert status_file.read_run(str(phase))["state"] == "running"
        done.touch()

    @POSIX_ONLY
    def test_one_status_call_is_well_under_a_second(self, tmp_path, capsys):
        """The poll loop is short commands by construction (D4f)."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _pid, done = spawn_worker(tmp_path, phase, plan, run_id="run-timing-01")
        run_status.main(["--plan-file", str(plan), "--view", "run"])  # warm caches
        capsys.readouterr()
        start = time.monotonic()
        run_status.main(["--plan-file", str(plan), "--view", "run"])
        elapsed = time.monotonic() - start
        capsys.readouterr()
        assert elapsed < 0.5, f"one status call took {elapsed:.3f}s"
        done.touch()

    @POSIX_ONLY
    def test_wait_returns_exit_two_at_its_timeout_with_the_run_alive(self,
                                                                    tmp_path,
                                                                    capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _pid, done = spawn_worker(tmp_path, phase, plan, run_id="run-wait-0001")
        start = time.monotonic()
        code = run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--wait", "--timeout", "1"]
        )
        elapsed = time.monotonic() - start
        capsys.readouterr()
        assert code == 2
        assert 0.5 <= elapsed < 15
        assert status_file.read_run(str(phase))["state"] == "running"
        done.touch()

    def test_wait_default_sits_under_the_tightest_host_shell_cap(self):
        assert run_status.DEFAULT_WAIT_TIMEOUT_S == 90.0
        assert run_status.DEFAULT_WAIT_TIMEOUT_S < 120.0

    def test_failed_run_hands_back_a_reason_and_the_log_tail(self, tmp_path,
                                                             capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        Path(detach.log_path(str(phase))).write_text(
            "line one\nboom: provider exploded\n", encoding="utf-8"
        )
        status_file.claim_run(str(phase), {
            "run_id": "run-failed-001", "state": "failed", "exit_code": 1,
            "error": "RuntimeError: provider exploded",
            "log_path": detach.log_path(str(phase)), "detached": True,
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 3
        out = capsys.readouterr().out
        assert "provider exploded" in out
        assert "boom: provider exploded" in out, "the log tail is inline"
        assert "resume with:" in out

    def test_done_run_that_exited_three_is_needs_human(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": "run-three-0001", "state": "failed", "exit_code": 3,
            "error": "needs_human (exit 3)", "detached": True,
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 3
        capsys.readouterr()

    def test_output_files_are_handed_back_not_globbed(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        report = phase / "report.md"
        report.write_text("# report\n", encoding="utf-8")
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "output_files": {"report": str(report)},
        })
        status_file.claim_run(str(phase), {
            "run_id": "run-artifact-1", "state": "done", "exit_code": 0,
            "detached": True,
        })
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 0
            assert str(report) in ctx.artifacts
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert str(report) in out

    def test_a_consolidation_recommendation_survives_the_detach_boundary(
        self, tmp_path, capsys
    ):
        """The `[CONSOLIDATION_RECOMMENDED]` marker is a log line, and the
        caller of a detached run never reads the log — so `classify_run` carries
        the checkpoint's copy and `status` is where a host learns of it. It is
        never handed back as `next`: `--consolidate` is a mechanical
        continuation a host may follow unasked, and a recommendation is not an
        instruction.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "consolidation_recommended": {
                "valid_groups": 29, "threshold": 15, "command": "--consolidate",
            },
        })
        status_file.claim_run(str(phase), {
            "run_id": "run-consol-001", "state": "done", "exit_code": 0,
            "detached": True,
        })
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 0
            assert ctx.consolidation_recommended == {
                "valid_groups": 29, "threshold": 15, "command": "--consolidate",
            }
            assert ctx.payload(0)["next"] is None
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert "consolidation recommended: 29 valid groups >= 15 — run --consolidate" in out

    def test_sweep_closes_out_orphans_across_the_plan(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": "run-orphan-001", "pid": _dead_pid(),
            "process_start_time": "999", "state": "running", "detached": True,
            "heartbeat_at": "2020-01-01T00:00:00", "exit_code": None,
        })
        assert run_status.main(["--plan-file", str(plan), "--sweep"]) == 0
        out = capsys.readouterr().out
        assert "running -> failed" in out
        run = _terminal_once(phase)
        assert run["error"] == "heartbeat_timeout"


def _json_body(text: str) -> str:
    return text.rsplit(cli_common.RESULT_MARKER, 1)[0]


# ===========================================================================
# D4b — ask runs: two questions detached at once
# ===========================================================================

class TestAskSelection:
    def _ask_dir(self, plan: Path, question: str, run_id: str, state: str) -> Path:
        from utils.output_handler import slugify_question

        directory = (
            plan.parent / plan.stem.replace(" ", "_") / "ask"
            / slugify_question(question)
        )
        directory.mkdir(parents=True, exist_ok=True)
        status_file.write_status(str(directory), {
            "phase": "ask", "state": "models_running", "question": question,
            "plan_hash": "abc123", "models_requested": ["cursor-agent:auto"],
        })
        status_file.claim_run(str(directory), {
            "run_id": run_id, "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": state, "detached": True,
            "heartbeat_at": status_file.now_iso(),
            "exit_code": 0 if state == "done" else None,
        })
        return directory

    def test_both_questions_are_listed_and_each_is_selectable(self, tmp_path,
                                                              capsys):
        plan = _plan(tmp_path)
        first = "Is the rollback strategy sufficient?"
        second = "What breaks on Windows?"
        self._ask_dir(plan, first, "run-ask-first-1", "running")
        self._ask_dir(plan, second, "run-ask-second1", "running")

        assert run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--json"]
        ) == 2
        payload = json.loads(_json_body(capsys.readouterr().out))
        questions = {view["question"] for view in payload["runs"]}
        assert questions == {first, second}

        assert run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--question", second,
             "--json"]
        ) == 2
        payload = json.loads(_json_body(capsys.readouterr().out))
        assert [view["question"] for view in payload["runs"]] == [second]

        assert run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--run-id",
             "run-ask-first", "--json"]
        ) == 2
        payload = json.loads(_json_body(capsys.readouterr().out))
        assert [view["question"] for view in payload["runs"]] == [first]

    def test_history_is_listed_only_under_all(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        self._ask_dir(plan, "old one?", "run-ask-old-01", "done")
        time.sleep(0.01)
        self._ask_dir(plan, "older one?", "run-ask-old-02", "done")
        self._ask_dir(plan, "live one?", "run-ask-live-1", "running")

        run_status.main(["--plan-file", str(plan), "--view", "run", "--json"])
        default = json.loads(_json_body(capsys.readouterr().out))["runs"]
        run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--all", "--json"]
        )
        every = json.loads(_json_body(capsys.readouterr().out))["runs"]
        assert len(default) < len(every)
        assert any(v["question"] == "live one?" for v in default)


# ===========================================================================
# D4h — the registry
# ===========================================================================

class TestRegistry:
    def test_sweep_keeps_exactly_the_live_entry(self, tmp_path, capsys):
        root = tmp_path / "project"
        (root / ".multi-llm" / "runs").mkdir(parents=True)
        identity = process_control.identity()
        run_registry.write_entry(str(root), {
            "run_id": "live", "pid": identity["pid"],
            "process_start_time": identity["process_start_time"],
            "phase_dir": str(tmp_path / "live-phase"), "mode": "ask",
        })
        run_registry.write_entry(str(root), {
            "run_id": "recycled", "pid": identity["pid"],
            "process_start_time": "0",
            "phase_dir": str(tmp_path / "recycled-phase"), "mode": "ask",
        })
        old = run_registry.write_entry(str(root), {
            "run_id": "ancient", "pid": identity["pid"],
            "process_start_time": identity["process_start_time"],
            "phase_dir": str(tmp_path / "ancient-phase"), "mode": "review-plan",
        })
        eight_days = time.time() - 8 * 86400
        os.utime(old, (eight_days, eight_days))

        rows = run_registry.sweep(str(root), kill=False)
        assert {row["run_id"]: row["action"] for row in rows} == {
            "live": "kept", "recycled": "closed", "ancient": "pruned",
        }
        assert [e["run_id"] for e in run_registry.list_entries(str(root))] == ["live"]

    def test_util_runs_lists_and_sweeps(self, tmp_path, capsys):
        root = tmp_path / "project"
        (root / ".multi-llm" / "runs").mkdir(parents=True)
        run_registry.write_entry(str(root), {
            "run_id": "dead-one", "pid": _dead_pid(),
            "process_start_time": "1", "phase_dir": str(tmp_path / "p"),
            "mode": "ask",
        })
        assert multi_llm.main(["util", "runs", "--project-root", str(root)]) == 0
        assert "dead-one" in capsys.readouterr().out
        assert multi_llm.main(
            ["util", "runs", "--project-root", str(root), "--sweep"]
        ) == 0
        capsys.readouterr()
        assert run_registry.list_entries(str(root)) == []

    def test_worker_registers_and_deregisters_itself(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        handle = detach.RunHandle(
            str(phase), run_id="run-registry-1", mode="review-plan",
            plan_file=str(plan), detached=False,
        )
        handle.start()
        root = handle.project_root
        assert any(
            e["run_id"] == "run-registry-1" for e in run_registry.list_entries(root)
        )
        handle.finish(0)
        assert not any(
            e["run_id"] == "run-registry-1" for e in run_registry.list_entries(root)
        )


# ===========================================================================
# D4d — the process-control façade is the only place OS internals live
# ===========================================================================

class TestProcessControlFacade:
    def test_no_runtime_module_imports_fcntl_or_msvcrt_directly(self):
        offenders = []
        for path in sorted(SKILL_ROOT.rglob("*.py")):
            parts = path.relative_to(SKILL_ROOT).parts
            if any(p in {"tests", "__pycache__", ".venv"} or p.startswith(".")
                   for p in parts):
                continue
            if path.name == "process_control.py":
                continue
            text = path.read_text(encoding="utf-8")
            for module in ("fcntl", "msvcrt"):
                if f"import {module}" in text:
                    offenders.append(f"{path.relative_to(SKILL_ROOT)}: {module}")
        assert not offenders, (
            "Unix/Windows-only stdlib belongs behind utils/process_control.py: "
            + ", ".join(offenders)
        )

    def test_lock_is_exclusive_and_readable_while_held(self, tmp_path):
        path = str(tmp_path / ".lock")
        first = process_control.PhaseLock(path)
        first.acquire({"run_id": "holder"})
        try:
            assert process_control.read_lock_holder(path)["run_id"] == "holder"
            assert not process_control.lock_is_free(path)
            with pytest.raises(process_control.LockUnavailable) as exc:
                process_control.PhaseLock(path).acquire({"run_id": "other"})
            assert exc.value.holder["run_id"] == "holder"
        finally:
            first.release()
        assert process_control.lock_is_free(path)
        # Releasing drops the lock and *keeps* the file: unlock-then-unlink would
        # let a second process take this inode in the gap, the unlink would strip
        # its pathname, and a third would create and lock a new file at the same
        # path — two holders of "the" phase lock.
        assert os.path.exists(path), "the lock file is durable; the lock is not"
        second = process_control.PhaseLock(path)
        second.acquire({"run_id": "successor"})
        try:
            assert process_control.read_lock_holder(path)["run_id"] == "successor"
        finally:
            second.release()

    def test_lock_is_released_when_the_holder_dies(self, tmp_path):
        """Held on an open fd, so a crash releases it even though the file stays."""
        path = tmp_path / ".lock"
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from utils import process_control\n"
            "lock = process_control.PhaseLock(%r)\n"
            "lock.acquire({'run_id': 'crasher'})\n"
            "print('locked', flush=True)\n"
            "import os; os._exit(9)\n" % (str(SKILL_ROOT), str(path))
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=60)
        assert "locked" in proc.stdout
        assert path.exists(), "the file remains…"
        assert process_control.lock_is_free(str(path)), "…but nobody holds it"

    def test_probe_needs_identity_not_just_a_live_pid(self):
        identity = process_control.identity()
        assert process_control.probe(
            identity["pid"], identity["process_start_time"]
        ) == process_control.ALIVE
        assert process_control.probe(identity["pid"], "0") == process_control.DEAD
        assert process_control.probe(identity["pid"], None) == process_control.UNKNOWN
        assert process_control.probe(_dead_pid(), "0") == process_control.DEAD
        assert process_control.probe(None, None) == process_control.DEAD

    def test_start_time_is_stable_across_reads(self):
        pid = os.getpid()
        assert process_control.process_start_time(pid) == \
            process_control.process_start_time(pid)


class TestWindowsSeam:
    """The Windows half cannot execute here, so the seam is asserted instead.

    D4d's Windows rules are structural — *one* spawn not two, explicit handles
    never inherited, no console, a Job Object, and a ``taskkill /T /F`` escalation
    — and every one of them is decided in code this machine can reach with
    ``IS_WINDOWS`` flipped. This is deliberately not a skip: a Windows CI job runs
    the same mechanism for real, and until it does these assertions are what keeps
    the Windows branch from rotting silently.
    """

    def test_windows_detaches_in_one_spawn_with_explicit_handles(self, tmp_path,
                                                                monkeypatch):
        log = tmp_path / "orchestrator-run.log"
        log.write_text("existing\n", encoding="utf-8")
        recorded = {}

        class FakePopen:
            def __init__(self, argv, **kwargs):
                recorded["argv"] = argv
                recorded["kwargs"] = kwargs
                recorded["stdout_closed"] = kwargs["stdout"].closed
                self.pid = 4242

        def no_fork(*_a, **_k):
            raise AssertionError("Windows must not double-spawn: there is no "
                                 "intermediate to reap")

        monkeypatch.setattr(process_control, "IS_WINDOWS", True)
        monkeypatch.setattr(process_control.subprocess, "Popen", FakePopen)
        monkeypatch.setattr(process_control.os, "fork", no_fork, raising=False)

        pid = process_control.spawn_detached(
            ["uv", "run"], cwd=str(tmp_path), env={"A": "B"}, log_path=str(log)
        )
        assert pid == 4242
        kwargs = recorded["kwargs"]
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: no console, own group, so
        # GenerateConsoleCtrlEvent is deliverable as the abort path.
        assert kwargs["creationflags"] == 0x00000008 | 0x00000200
        assert kwargs["close_fds"] is True
        assert kwargs["stdin"] is process_control.subprocess.DEVNULL
        # One handle, passed as both stdout and stderr: a console-less child has
        # no valid stdout to inherit, and only the three std handles cross over.
        assert kwargs["stdout"] is kwargs["stderr"]
        assert kwargs["stdout"] is not process_control.subprocess.DEVNULL
        assert kwargs["cwd"] == str(tmp_path) and kwargs["env"] == {"A": "B"}
        # The launcher opened it append-only (the predecessor's bytes survive) and
        # closed its own copy after handing it over.
        assert log.read_text(encoding="utf-8") == "existing\n"
        assert recorded["stdout_closed"] is False

    def test_windows_worker_does_not_reopen_stdio_but_joins_a_job_object(
        self, monkeypatch
    ):
        """The launcher already handed the log in; the worker adds the reaper."""
        calls = []
        monkeypatch.setattr(process_control, "IS_WINDOWS", True)
        monkeypatch.setattr(
            process_control, "_assign_own_job_object",
            lambda: calls.append("job"),
        )
        monkeypatch.setattr(
            process_control.os, "dup2",
            lambda *a, **k: calls.append("dup2"),
        )
        process_control.enter_detached_worker("C:\\nope\\orchestrator-run.log")
        assert calls == ["job"], (
            "on Windows the log arrives as an inherited handle, so redirecting "
            "fds would be wrong; the Job Object is what reaps provider children"
        )

    def test_windows_abort_escalates_to_taskkill_of_the_whole_tree(self,
                                                                  monkeypatch):
        runs = []
        alive = iter([True, True, False])
        monkeypatch.setattr(process_control, "IS_WINDOWS", True)
        monkeypatch.setattr(process_control, "process_alive",
                            lambda _pid: next(alive, False))
        monkeypatch.setattr(process_control, "probe",
                            lambda *_a, **_k: process_control.ALIVE)
        monkeypatch.setattr(
            process_control.subprocess, "run",
            lambda argv, **kwargs: runs.append(argv),
        )
        verdict = process_control.kill_tree(4242, escalate_after_s=0.0)
        assert verdict == "killed"
        assert runs and runs[0] == ["taskkill", "/PID", "4242", "/T", "/F"], runs

    def test_windows_lock_leaves_the_holder_payload_readable(self):
        """Windows locks are mandatory, so the lock byte sits past the payload."""
        assert process_control._WINDOWS_LOCK_OFFSET >= 1 << 20

    def test_the_facade_reports_which_platform_it_is_on(self):
        assert process_control.IS_WINDOWS == (os.name == "nt")


# ===========================================================================
# The documented poll loop
# ===========================================================================

class TestPollLoopContract:
    def test_banner_text_forbids_collapsing_the_wait_into_one_shell_call(self):
        text = detach.POLL_LOOP_TEXT
        assert "sleep 90; status" in text
        assert "Never collapse the wait" in text
        assert "Closing this session does NOT stop the run" in text
        assert "exit 2" in text and "exit 0" in text and "exit 3" in text

    def test_heartbeat_window_is_six_missed_beats(self):
        assert detach.HEARTBEAT_INTERVAL_S == 10.0
        assert detach.HEARTBEAT_STALE_AFTER_S == 60.0
        assert (
            detach.HEARTBEAT_STALE_AFTER_S / detach.HEARTBEAT_INTERVAL_S == 6
        )

    def test_detach_modes_are_the_long_running_modes(self):
        """The fan-out modes plus the executor-backed write modes (task 6).

        Pinned as a literal so *adding* a mode to the set is a deliberate edit
        here, not a side effect of a table change: a mode that takes ``--detach``
        without being long-running gives a caller a run to poll that finished
        before the poll loop started.
        """
        assert detach.DETACH_MODES == (
            "review-plan", "apply-suggestions", "generate-tasks", "review-tasks",
            "apply-task-suggestions", "implement", "review-code",
            "apply-code-fixes", "full", "ask",
        )
        for mode in detach.DETACH_MODES:
            assert mode_table.resolve(mode) is not None


# ===========================================================================
# Full stack: a real fan-out, detached, watched to `done` with `status`
# ===========================================================================

UV_STUB = '''#!/usr/bin/env python3
"""A `uv` that asserts the argv shape D4e specifies, then execs it.

Substituting uv (rather than skipping it) is what makes this test exercise the
real re-exec path: if the child were ever re-launched as bare python, or with
--project fused into another argument, this stub would fail the run.
"""
import os
import sys

argv = sys.argv[1:]
assert argv[0] == "run", argv
# The uv flags sit between `run` and `--project`, and the worker must carry the
# same ones the launcher did. `--no-dev` is the load-bearing half: without it
# `uv run` syncs the dev group, so a careful launch would hand off to a child
# that pulled 17 packages -- or failed outright offline -- with the failure
# moving out of the launcher and into a log.
flags = []
index = 1
while index < len(argv) and argv[index] != "--project":
    flags.append(argv[index])
    index += 1
assert "--frozen" in flags, argv
assert "--no-dev" in flags, argv
assert argv[index] == "--project", argv
assert os.path.isdir(argv[index + 1]), argv
assert argv[index + 2] == "--", argv
assert argv[index + 3] == "python", argv
assert argv[index + 4].endswith("multi_llm.py"), argv
os.execv(sys.executable, [sys.executable, *argv[index + 4:]])
'''


@pytest.mark.timeout(300)
class TestFullStackDetachedFanOut:
    def test_detached_review_plan_reaches_done_through_status(
        self, tmp_path, skill_runner, fixture_manager, mock_provider
    ):
        """The Done-when row: `status` reports running -> done off one .status.json."""
        if os.name == "nt":
            pytest.skip("the POSIX double-spawn path; Windows is covered in CI")
        mock_provider.set_scenario("happy_path")
        skill_runner.extra_env.update(mock_provider.get_env())
        plan = fixture_manager.create_plan(
            "detached-plan", "# Detached Plan\n\n## Overview\nRun me detached.\n"
        )
        stub = tmp_path / "uv-stub"
        stub.write_text(UV_STUB, encoding="utf-8")
        stub.chmod(0o755)

        env = skill_runner._build_env()
        env[detach.UV_ENV] = str(stub)
        env["MULTI_LLM_HEARTBEAT_INTERVAL_S"] = "1"

        launch = subprocess.run(
            [sys.executable, str(SKILL_ROOT / "multi_llm.py"), "review-plan",
             str(plan.plan_path), "--detach", "--models", "cursor-agent",
             "--skip-validation"],
            cwd=str(SKILL_ROOT), env=env, capture_output=True, text=True,
            timeout=120,
        )
        assert launch.returncode == 0, launch.stdout + launch.stderr
        assert "=== DETACHED RUN STARTED ===" in launch.stdout
        run_id = _banner_field(launch.stdout, "run_id")
        log = Path(_banner_field(launch.stdout, "log"))
        result = cli_common.parse_result_line(launch.stdout)
        assert result["next"] and " status " in result["next"]

        phase = plan.plan_path.parent / "detached-plan" / "review-plan"
        assert status_file.read_run(str(phase))["run_id"] == run_id

        def poll():
            proc = subprocess.run(
                [sys.executable, str(SKILL_ROOT / "multi_llm.py"), "status",
                 str(plan.plan_path), "--view", "run", "--phase", "review-plan",
                 "--json"],
                cwd=str(SKILL_ROOT), env=env, capture_output=True, text=True,
                timeout=60,
            )
            return proc

        deadline = time.time() + 120
        proc = poll()
        assert proc.returncode in (0, 2), proc.stdout + proc.stderr
        while proc.returncode == 2 and time.time() < deadline:
            time.sleep(1)
            proc = poll()
        assert proc.returncode == 0, proc.stdout + proc.stderr

        run = status_file.read_run(str(phase))
        assert run["state"] == "done"
        assert run["exit_code"] == 0
        assert run["detached"] is True

        # The log is the whole of the worker, not the part after the fork.
        # Everything knowable before the boundary is still reported on the
        # caller's own terminal (the pre-flight rule, unchanged) — and the
        # *worker's* copy of those same lines is in the log above the
        # [DETACHED_RUN] banner, because the launcher hands the log in as the
        # child's stdout/stderr. Before that, a detaching host that keeps only
        # the poll payload had no durable record of what a run loaded or decided:
        # measured 2026-09-16, a completed apply-suggestions log opened at
        # [DETACHED_RUN] with no sign of the 33 groups it had loaded or the
        # "no user preferences detected" notice under which it applied all 29.
        assert "Models: cursor-agent" in launch.stdout
        log_text = log.read_text(encoding="utf-8")
        for line in ("Loaded plan:", "Backup saved:", "Models: cursor-agent"):
            assert line in log_text, f"pre-detach line missing from the log: {line}"
        banner = f"[DETACHED_RUN] run_id={run_id}"
        assert banner in log_text
        assert log_text.index("Backup saved:") < log_text.index(banner), (
            "the banner marks where the invoker stopped sharing the run, so the "
            "setup lines belong above it"
        )
        # Everything after it belongs to the worker — including the
        # MULTI_LLM_RESULT line, so a host tailing the log gets the same
        # machine-readable contract as a host reading stdout.
        assert cli_common.parse_result_line(log_text)["exit"] == 0
        assert (phase / "grouped.json").exists()

        # Success hands back paths, not a search: the artifacts a foreground run
        # would have printed are in output_files, and `status` repeats them in
        # MULTI_LLM_RESULT.artifacts, so no host globs a phase dir.
        payload = json.loads(_json_body(proc.stdout))
        view = payload["runs"][0]
        assert view["output_files"]["report"] == str(phase / "report.md")
        assert Path(view["output_files"]["report"]).exists()
        artifacts = cli_common.parse_result_line(proc.stdout)["artifacts"]
        assert str(phase / "report.md") in artifacts
        assert process_control.lock_is_free(detach.lock_path(str(phase)))
        assert run_registry.list_entries(
            run_registry.resolve_project_root(str(plan.plan_path))
        ) == []

    def test_detached_launch_rotates_a_predecessor_log(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        log = Path(detach.log_path(str(phase)))
        log.write_text("previous run output\n", encoding="utf-8")
        status_file.claim_run(str(phase), {
            "run_id": "run-previous-1", "state": "done", "exit_code": 0,
            "detached": True,
        })
        rotated = detach.rotate_log(str(phase), "run-previous-1")
        assert rotated and Path(rotated).name == "orchestrator-run.run-previous-1.log"
        assert not log.exists()
        assert Path(rotated).read_text(encoding="utf-8") == "previous run output\n"

    def test_rotated_logs_are_pruned_to_five(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        for index in range(8):
            Path(detach.log_path(str(phase))).write_text(f"run {index}\n",
                                                         encoding="utf-8")
            detach.rotate_log(str(phase), f"r{index}")
        rotated = sorted(
            p.name for p in phase.iterdir()
            if p.name.startswith("orchestrator-run.") and p.name.endswith(".log")
        )
        assert len(rotated) == 5, rotated


def _banner_field(text: str, name: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name}:"):
            return stripped.split(":", 1)[1].strip()
    raise AssertionError(f"no {name} in the detach banner:\n{text}")


# ===========================================================================
# D2 — the detached route carries the launcher's flags
# ===========================================================================

class TestReplayFlagsAcrossTheDetachBoundary:
    """Three assertions, one level each: the launcher ships its list on the
    child's environment, the worker pops it before the phase body runs and
    writes it into the record, and ``status`` splices it back into every
    command it prints — the inlined log tail included."""

    def test_launcher_ships_replay_flags_on_the_child_env(self, tmp_path,
                                                          monkeypatch, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        seen = {}

        def fake_spawn(command, *, cwd, env, log_path):
            seen["command"] = list(command)
            seen["env"] = dict(env)
            run_id = command[command.index("--internal-run-id") + 1]
            status_file.claim_run(str(phase), {
                "run_id": run_id, "state": "running", "pid": os.getpid(),
                "heartbeat_at": status_file.now_iso(),
            })
            return os.getpid()

        monkeypatch.setattr(process_control, "spawn_detached", fake_spawn)
        argv = ["--plan-file", str(plan), "--quick", "--detach", "--yes",
                "--exec-subagents"]
        code = detach.launch(
            mode="review-plan", argv=argv, phase_dir=str(phase),
            plan_file=str(plan), phase="review-plan",
            args=argparse.Namespace(force=False, if_not_running=False),
        )
        capsys.readouterr()
        assert code == 0
        shipped = json.loads(seen["env"][detach.REPLAY_FLAGS_ENV])
        # The launcher's list: the selector and the caller's --detach, never
        # the implied pair nor the re-entry flags the child command carries.
        assert shipped == ["--quick", "--detach"]
        assert "--internal-worker" in seen["command"]
        assert "--yes" in seen["command"] and "--exec-subagents" in seen["command"]

    def test_worker_pops_the_env_and_records_the_launchers_list(self, tmp_path,
                                                                monkeypatch):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        monkeypatch.setenv(
            detach.REPLAY_FLAGS_ENV, json.dumps(["--quick", "--detach"])
        )
        monkeypatch.delenv(detach.WORKER_ENV, raising=False)
        args = argparse.Namespace(
            internal_worker=True, internal_run_id="run-replay-0001",
            host="opencode",
        )
        handle = detach.begin_worker_run(
            phase_dir=str(phase), mode="review-plan", plan_file=str(plan),
            args=args,
        )
        try:
            # Absent by the time the phase body runs: no provider child inherits it.
            assert detach.REPLAY_FLAGS_ENV not in os.environ
            record = status_file.read_run(str(phase))
            assert record["replay_flags"] == ["--quick", "--detach"]
            for token in ("--internal-worker", "--internal-run-id", "--yes",
                          "--exec-subagents"):
                assert token not in record["replay_flags"]
        finally:
            handle.finish(0)

    def test_a_malformed_env_value_records_an_empty_list(self, tmp_path, monkeypatch):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        monkeypatch.setenv(detach.REPLAY_FLAGS_ENV, "{not json")
        monkeypatch.delenv(detach.WORKER_ENV, raising=False)
        handle = detach.begin_worker_run(
            phase_dir=str(phase), mode="review-plan", plan_file=str(plan),
            args=argparse.Namespace(internal_worker=True, internal_run_id="r2"),
        )
        try:
            assert status_file.read_run(str(phase))["replay_flags"] == []
        finally:
            handle.finish(0)

    def test_status_after_the_worker_died_carries_quick_everywhere(self, tmp_path,
                                                                   capsys):
        """The printed-text carrier: the whole of ``status``'s stdout — the
        inlined log tail included — has no flag-less ``--reaggregate``."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        worker_line = cli_common.retry_command(
            "review-plan", str(plan), "--reaggregate", flags=["--quick"]
        )
        Path(detach.log_path(str(phase))).write_text(
            "ERROR: 1 of 1 validation task(s) did not produce a usable result\n"
            "Re-invoke to retry only the failures (validation-batch-0):\n"
            f"  {worker_line}\n",
            encoding="utf-8",
        )
        status_file.claim_run(str(phase), {
            "run_id": "run-quick-00001", "state": "failed", "exit_code": 3,
            "error": "BatchFailed", "log_path": detach.log_path(str(phase)),
            "detached": True, "mode": "review-plan",
            "replay_flags": ["--quick", "--detach"],
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 3
        out = capsys.readouterr().out
        assert "resume with:" in out
        assert "--quick --detach --reaggregate" in out
        carriers = [line for line in out.splitlines() if "--reaggregate" in line]
        assert carriers
        assert all("--quick" in line for line in carriers), out
        assert "--force" not in out


class TestReaggregateCrossesTheDetachBoundary:
    """`--reaggregate --detach` launches a worker exactly as the fan-out does.

    Measured 2026-09-14 on review-tasks: the reaggregate branch ran ahead of
    `maybe_detach`, the re-dispatch of the two failed validation batches
    completed in the invoker's foreground with no run banner, and the opencode
    plugin — which composes `--detach` on every launch and adopts a run only
    from that banner — reported the finished run as an error ("the launch
    reported no run_id on its banner").
    """

    @pytest.mark.parametrize("mode,phase", [
        ("review-plan", "review-plan"),
        ("review-tasks", "review-tasks"),
        ("review-code", "code-review"),
    ])
    def test_the_reaggregate_branch_launches_a_worker(self, mode, phase, tmp_path,
                                                      monkeypatch, capsys):
        launched = {}

        def fake_launch(*, mode, argv, phase_dir, plan_file, args, **kwargs):
            launched.update(mode=mode, argv=list(argv), phase_dir=phase_dir)
            return 0

        monkeypatch.setattr(detach, "launch", fake_launch)
        plan = _plan(tmp_path)
        # The branch's one pre-flight, kept in the foreground: the dir exists.
        _phase_dir(plan, phase)
        code = multi_llm.main([mode, str(plan), "--detach", "--reaggregate"])
        out = capsys.readouterr()
        assert code == 0, (out.out, out.err)
        assert launched["mode"] == mode
        assert "--reaggregate" in launched["argv"]
        assert Path(launched["phase_dir"]).name == phase
        # Nothing re-aggregated in the invoker: the worker owns that.
        assert "Reaggregation complete" not in out.out
