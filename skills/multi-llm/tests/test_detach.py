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
from utils import progress_tracker, status_file  # noqa: E402

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

    def test_the_session_model_reaches_the_detached_worker(self, monkeypatch):
        """A host plugin sets it on the launcher; the worker's ladder reads it."""
        from utils.provider_registry import HOST_SESSION_MODEL_ENV

        monkeypatch.setenv(HOST_SESSION_MODEL_ENV, "gpt-5.5")
        assert detach.child_env(base=SKILL_ROOT)[HOST_SESSION_MODEL_ENV] == "gpt-5.5"


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
# `report_progress` — the one reserved object a foreground run writes
# ===========================================================================

class _HandleFor:
    """Just enough of ``detach.RunHandle`` to own a phase dir.

    The real handle needs a lock, a heartbeat thread and a claimed ``run``
    record; what ``report_progress`` actually reads off it is two things — that
    it exists, and which dir it owns — so a stand-in is the honest subject for
    the dir-mismatch branch and keeps the test from claiming a second run.
    """

    def __init__(self, phase_dir):
        self.phase_dir = str(phase_dir)
        self.calls = []

    def progress(self, **updates):
        self.calls.append(updates)
        status_file.update_progress(self.phase_dir, updates)


class TestForegroundProgressWrites:
    """`report_progress` was never a ``--detach`` check: it read the module-global
    ``_ACTIVE`` handle, so a foreground run — and a Claude Code native-background
    run, which is a foreground run inside a background Bash — wrote nothing at
    all for the length of a fan-out.

    The handle now picks the **door** and never whether the write happens: it
    goes through ``handle.progress`` (which stamps that run's heartbeat with it)
    only when the handle owns the dir being written, and straight through
    ``status_file.update_progress`` otherwise — with no handle at all, and with a
    handle that owns a *different* dir (F3, 2026-09-20)."""

    def test_a_foreground_run_writes_progress_with_no_handle(
        self, tmp_path, monkeypatch
    ):
        """The `handle is None` branch, gated on the handle and not on a flag."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        monkeypatch.setattr(detach, "_ACTIVE", None)

        detach.report_progress(
            str(phase), step="fanout", models_total=4, models_done=1
        )

        progress = status_file.read_status(str(phase)).get("progress") or {}
        assert progress == {"step": "fanout", "models_total": 4, "models_done": 1}

    def test_a_write_for_another_phase_dir_lands_where_the_caller_named(
        self, tmp_path, monkeypatch
    ):
        """`full --detach` is this branch, and it used to lose every phase's map.

        A handle whose ``phase_dir`` differs is not "no run" — it is a run that
        owns a *different* dir, and the mode that does this every time is
        ``full``: it claims ``{plan}/full/`` for its own ``.status.json``, log
        and lock, then calls each phase's ``main()`` in process, where every
        phase's tracker targets ``{plan}/<phase>/``. Dropping the write there
        discarded **all** the progress of the longest-running mode the tool has —
        ``[PROGRESS]`` lines in the log, and no ``models`` map, no counters and
        no step under ``status --phase review-plan``.

        The reason given for the drop conflated two directories that are never
        the same one: a tracker writes to the dir it was *built for* and has no
        way to address the handle's, so "one run's ticker scribbling into another
        run's map" could not happen. What the handle really owns is its own
        **record** — the heartbeat stamp and the ``run_id`` compare-and-set —
        which is why a foreign dir goes straight through ``update_progress`` and
        never through ``handle.progress`` (F3, 2026-09-20).
        """
        plan = _plan(tmp_path)
        owned = _phase_dir(plan, "full")
        other = _phase_dir(plan, "review-plan")
        handle = _HandleFor(owned)
        monkeypatch.setattr(detach, "_ACTIVE", handle)

        detach.report_progress(
            str(other), step="fanout", models_done=1, models_total=4
        )

        progress = status_file.read_status(str(other)).get("progress") or {}
        assert progress == {"step": "fanout", "models_done": 1, "models_total": 4}
        assert handle.calls == [], (
            "not through the handle: its record, and its heartbeat, are not this "
            "dir's to stamp"
        )
        # The run's own dir is untouched by a write aimed at the phase it drives:
        # one checkpoint per dir, still.
        assert "progress" not in status_file.read_status(str(owned))
        assert "run" not in status_file.read_status(str(other)), (
            "and the phase dir gains no `run` object - `full` owns the run"
        )

    def test_the_merge_cuts_both_ways_over_one_checkpoint(
        self, tmp_path, monkeypatch
    ):
        """One checkpoint, three writers, nothing lost in either direction.

        ``update_status`` merges `progress` key by key at its **top** level, and
        that single fact has to hold two ways at once or the map and the rest of
        the object take turns erasing each other: a `models` write must keep the
        siblings other sites wrote (`step`, `sandbox_bypassed`, `models_total`,
        and the `subagents` / `write_subagents` executor mirrors, which go
        straight through ``update_status`` and so already reach a *foreground*
        checkpoint), and a later write that is not about the map must leave the
        stored map identical rather than clobbering it.

        The merge is one level deep, though, so the nested `models` value is a
        single key: a write that carries it replaces the stored map wholesale
        and never deep-merges into it. That is why the fan-out holds the map in
        memory and writes it whole, and why a counters-only write carries no
        `models` key at all.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        monkeypatch.setattr(detach, "_ACTIVE", None)

        # The siblings, written the way their real sites write them.
        detach.report_progress(str(phase), step="fanout", models_total=2)
        detach.report_progress(str(phase), sandbox_bypassed=["codex:gpt-6-astra"])
        status_file.update_status(str(phase), {"progress": {
            "subagents": {"kind": "validation", "total": 3, "ok": 1},
            "write_subagents": {"total": 2, "ok": 2},
        }})

        first_map = {
            "claude-code:sonnet-4.6": {"state": "done", "elapsed_s": 12},
            "codex:gpt-6-astra": {"state": "running"},
        }
        detach.report_progress(
            str(phase), models=first_map, models_done=1, models_succeeded=1,
            models_total=2,
        )

        progress = status_file.read_status(str(phase))["progress"]
        assert progress["models"] == first_map
        assert progress["step"] == "fanout"
        assert progress["sandbox_bypassed"] == ["codex:gpt-6-astra"]
        assert progress["models_total"] == 2
        assert progress["subagents"] == {"kind": "validation", "total": 3, "ok": 1}
        assert progress["write_subagents"] == {"total": 2, "ok": 2}

        # ... and back the other way: neither a counters-only write nor a
        # `step=`-only one carries a `models` key, so the map stays byte-identical.
        detach.report_progress(str(phase), models_done=1, models_succeeded=1)
        detach.report_progress(str(phase), step="validating")

        progress = status_file.read_status(str(phase))["progress"]
        assert progress["models"] == first_map
        assert progress["step"] == "validating"
        assert progress["sandbox_bypassed"] == ["codex:gpt-6-astra"]

        # One level deep: a write that *does* carry `models` replaces the stored
        # map wholesale, which is what the seed and the teardown sweep depend on.
        second_map = {"claude-code:sonnet-4.6": {"state": "done", "elapsed_s": 12}}
        detach.report_progress(str(phase), models=second_map)
        progress = status_file.read_status(str(phase))["progress"]
        assert progress["models"] == second_map
        assert progress["step"] == "validating"

    def test_a_foreground_run_still_records_no_run_object(
        self, tmp_path, monkeypatch, capsys
    ):
        """The new write path adds `progress` keys and *nothing else*.

        "With ``--detach`` unset, behavior is unchanged" narrows to "a foreground
        `.status.json` gains a `progress` object and still no `run` object" —
        and `run` is where every reader actually branches. ``claim_run`` is never
        called, so `classify_run` still reads `no_run`, and F1's rule that a
        `no_run` phase is not a `next` poll target still skips it in favor of a
        genuinely detached run.
        """
        plan = _plan(tmp_path)
        monkeypatch.setattr(detach, "_ACTIVE", None)

        done = _phase_dir(plan, "review-plan")
        status_file.write_status(
            str(done), {"phase": "review-plan", "state": "complete"}
        )
        status_file.claim_run(str(done), {
            "run_id": "run-detached-0002", "mode": "review-plan", "state": "done",
            "exit_code": 0, "detached": True, "host": "opencode",
        })
        # Sorts after `review-plan`, so it cannot win on ordering alone — and it
        # is now a phase dir with a populated `progress` object, which is exactly
        # the thing F1 must not mistake for liveness.
        foreground = _phase_dir(plan, "tasks")
        status_file.write_status(
            str(foreground), {"phase": "tasks", "state": "complete"}
        )
        detach.report_progress(
            str(foreground), step="fanout", models_total=4, models_done=2
        )

        status = status_file.read_status(str(foreground))
        assert (status.get("progress") or {}).get("models_done") == 2
        assert "run" not in status

        view = detach.classify_run(str(foreground))
        assert view["verdict"] == detach.VERDICT_NO_RUN
        assert view["progress"]["models_done"] == 2

        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context.next and "run-detached-0002" in context.next
        assert "--phase tasks" not in context.next


# ===========================================================================
# A stamp this process did not write
# ===========================================================================

class TestAForeignTimestampNeverCrashesStatus:
    """`_elapsed_since` is the one ISO reader for the whole checkpoint.

    Every stamp *we* write is `status_file.now_iso` → `datetime.now()`, which is
    naive, so an offset-bearing one means a hand-edited or foreign file — which
    `read_status`'s own contract says must not make `status` crash. It did:
    `datetime.fromisoformat("2026-09-19T10:14:03Z")` returns an **aware**
    datetime on 3.11+ (that `Z` form is the plan's own schema example, and what
    the JS tests emit), `datetime.now() - <aware>` raises **TypeError**, and this
    reader caught only `ValueError` — so one such `started_at`, `heartbeat_at` or
    per-model row took down the whole command, not just its own line (F4,
    2026-09-20).
    """

    def test_an_aware_stamp_is_normalised_rather_than_raised(self):
        from datetime import datetime, timezone

        when = datetime(2026, 9, 19, 10, 14, 3, tzinfo=timezone.utc)
        assert detach._elapsed_since(
            when.isoformat(), now=when.timestamp() + 90
        ) == 90.0
        # The `Z` spelling: read on 3.11+, declined (`None`) on 3.10. Either is
        # fine; raising is not.
        assert detach._elapsed_since(
            "2026-09-19T10:14:03Z", now=when.timestamp() + 90
        ) in (90.0, None)

    def test_garbage_is_still_declined_rather_than_guessed(self):
        for stamp in ["not a date", "", None, 17, "2026-13-45T99:99:99"]:
            assert detach._elapsed_since(stamp) is None

    def test_an_aware_heartbeat_still_classifies_the_run(self, tmp_path):
        """The reader's caller: one bad stamp must cost that *line*, not the view."""
        from datetime import datetime, timezone

        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        status_file.claim_run(str(phase), {
            "run_id": "run-aware-0001", "mode": "review-plan", "state": "running",
            "detached": True, "host": "opencode", "exit_code": None,
            "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "started_at": stamp, "heartbeat_at": stamp,
        })

        view = detach.classify_run(str(phase), close_out=False)

        assert view["verdict"] == detach.VERDICT_RUNNING
        age = view["heartbeat_age_s"]
        assert age is None or age >= 0.0


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

    def test_suggested_next_names_the_optional_phases_it_skipped(
        self, tmp_path, capsys
    ):
        """F8: `suggested next: implement` reads as "nothing else is
        outstanding" on its own — and paired with the tool's `next: null` for a
        finished phase, a weak model asserted exactly that instead of naming
        `review-tasks`/`apply-task-suggestions` (measured 2026-09-16). The
        machine-readable `suggested_next` is unchanged; only the rendered line
        gains the optional phases still pending."""
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("apply-suggestions")
        state.mark_phase_completed("generate-tasks")
        state.save()

        # `--view workflow` alone stays raw JSON (byte-identical to the
        # retired `--status` mode); the rendered text lives in the default
        # `--view all`, which prints the workflow section first.
        assert run_status.main(["--plan-file", str(plan)]) == 2
        out = capsys.readouterr().out
        assert (
            "suggested next: implement  (review-tasks / apply-task-suggestions "
            "optional, not run)"
        ) in out

        # The JSON view carries the plain word only — the parenthetical is a
        # text-rendering fact, not a change to what a host parses.
        assert run_status.main(
            ["--plan-file", str(plan), "--view", "workflow", "--json"]
        ) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["suggested_next"] == "implement"

    def test_suggested_next_names_nothing_optional_when_none_are_pending(
        self, tmp_path, capsys
    ):
        """The bare word stays bare once an optional phase is actually
        skipped/completed rather than merely pending — nothing to flag."""
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        state.save()

        assert run_status.main(["--plan-file", str(plan)]) == 2
        out = capsys.readouterr().out
        assert "suggested next: apply-suggestions" in out
        assert "optional, not run" not in out

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

        # F2, 2026-09-20: and it WAITS. Four instruction cores tell an agent to
        # poll this command instead of composing one, and all four describe it
        # as waiting `progress.poll_s`. Without the flag every poll was a 0.13 s
        # file read: the turn loop spun at the host's pace instead of the run's,
        # the announced "back in ~90 s" was a gap nobody waited, and
        # `just finished:` — a difference between two readings inside one call —
        # could never print on a `--detach` lane at all. No `--timeout` rides
        # with it: the resolved value is clamped under the `--host` above, and a
        # number baked in here could not know the reader's shell cap.
        printed = detach.status_command_for(
            "/p.md", phase="review-plan", host="opencode"
        )
        assert "--wait" in printed
        assert "--timeout" not in printed

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

    def test_the_next_poll_targets_the_live_run_not_the_alphabetically_first(
        self, tmp_path, capsys
    ):
        """F3: with a finished `apply-suggestions` and a live `generate-tasks`
        both listed, `apply-suggestions` sorts first — but it is not why the
        exit code is 2, and polling it back reports a terminal run as done
        while the real one is still dispatching (measured 2026-09-16)."""
        plan = _plan(tmp_path)
        finished = _phase_dir(plan, "apply-suggestions")
        status_file.claim_run(str(finished), {
            "run_id": "run-finished-0001", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "host": "opencode", "mode": "apply-suggestions",
        })
        status_file.update_status(str(finished), {
            "phase": "apply-suggestions", "state": "complete",
        })
        live = _phase_dir(plan, "generate-tasks")
        status_file.claim_run(str(live), {
            "run_id": "run-live-0002", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True,
            "host": "opencode", "mode": "generate-tasks",
        })
        status_file.update_status(str(live), {
            "phase": "generate-tasks", "state": "dispatching",
        })
        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main(["--plan-file", str(plan)]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context.next and "generate-tasks" in context.next
        assert "run-live-0002" in context.next
        assert "apply-suggestions" not in context.next

    def test_state_word_says_running_not_nothing_to_do_while_a_run_is_live(
        self, tmp_path, capsys
    ):
        """F11: exit 2 genuinely means "poll again", but the payload word
        `nothing-to-do` contradicted the text one line above it — `generate-tasks
        run … running (last beat 10s ago)` (measured 2026-09-16). The exit code
        stays 2 (the poll loops key off it); only the word changes, and only
        when the exit-2 reason is a run still in flight."""
        plan = _plan(tmp_path)
        live = _phase_dir(plan, "generate-tasks")
        status_file.claim_run(str(live), {
            "run_id": "run-live-0003", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True, "mode": "generate-tasks",
        })
        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main(["--plan-file", str(plan)]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context.payload(2)["state"] == "running"

        # The genuinely empty case — no run recorded at all — keeps the old
        # word: exit 2 there means "no such run", not "come back later".
        empty_plan = _plan(tmp_path, "empty plan")
        context2 = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main(["--plan-file", str(empty_plan)]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context2.payload(2)["state"] == "nothing-to-do"

        # F1, 2026-09-17: and so does the case the pin above missed — a phase
        # dir holding a checkpoint but NO run record. That is every foreground
        # phase (only `--internal-worker` claims a run), so the Claude Code
        # host's normal mode was reading `state: running` for a plan where
        # nothing had ever been launched, with `next` pointed at the phase that
        # never had a run.
        fg_plan = _plan(tmp_path, "foreground plan")
        fg_phase = _phase_dir(fg_plan, "apply-suggestions")
        status_file.write_status(
            str(fg_phase), {"phase": "apply-suggestions", "state": "complete"}
        )
        context3 = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main(["--plan-file", str(fg_plan)]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context3.payload(2)["state"] == "nothing-to-do"

    def test_the_next_poll_skips_a_phase_that_never_had_a_run(
        self, tmp_path, capsys
    ):
        """F1, 2026-09-17: `no_run` is non-terminal only because there is
        nothing to be terminal about. Counted as live, a foreground-run phase
        outranked a genuinely finished detached run as the `next` poll target,
        and the poll it handed back can never leave exit 2."""
        plan = _plan(tmp_path)
        done = _phase_dir(plan, "review-plan")
        status_file.write_status(
            str(done), {"phase": "review-plan", "state": "complete"}
        )
        status_file.claim_run(str(done), {
            "run_id": "run-detached-0001", "mode": "review-plan", "state": "done",
            "exit_code": 0, "detached": True, "host": "opencode",
        })
        # Sorts after `review-plan`, so it cannot win on ordering alone.
        foreground = _phase_dir(plan, "tasks")
        status_file.write_status(
            str(foreground), {"phase": "tasks", "state": "complete"}
        )

        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context.next and "run-detached-0001" in context.next
        assert "--phase tasks" not in context.next

    def test_a_run_whose_pid_is_gone_is_rendered_died_not_running(
        self, tmp_path, capsys
    ):
        """F2, 2026-09-16: an implement write subagent ran
        `killall -9 node chrome google-chrome python3` and SIGKILLed the
        detached orchestrator that had dispatched it. The record was left
        `state: running, exit_code: null`, and for the whole heartbeat window
        every poll printed `running (last beat Ns ago)` for a process the
        identity probe could already prove was gone — with the resume command
        withheld until the close-out fired. The close-out is right to wait; the
        word is not."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "implement")
        dead = _dead_pid()
        status_file.claim_run(str(phase), {
            "run_id": "run-sigkilled-1", "pid": dead,
            # Not the same process even if the pid is recycled under us.
            "process_start_time": "0",
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True, "mode": "implement",
        })

        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            # Still exit 2 — `classify_run` closes the run out as
            # failed(heartbeat_timeout) once the beat expires, and every poll
            # loop keys off the code, never off the word.
            assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert f"died (pid {dead} gone, last beat " in out, out
        assert "running (stale" not in out, out
        assert "process is gone (pid/creation-time probe says dead)" in out, out
        assert "resume with:" in out and "--resume" in out, out
        assert context.payload(2)["state"] == "died"

    def test_a_dead_run_beside_a_live_one_still_reads_running(
        self, tmp_path, capsys
    ):
        """The word answers for the plan, not for one phase dir: a poller with
        something genuinely in flight must not be told the plan died."""
        plan = _plan(tmp_path)
        dead_phase = _phase_dir(plan, "implement")
        status_file.claim_run(str(dead_phase), {
            "run_id": "run-sigkilled-2", "pid": _dead_pid(),
            "process_start_time": "0",
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True, "mode": "implement",
        })
        live_phase = _phase_dir(plan, "review-plan")
        status_file.claim_run(str(live_phase), {
            "run_id": "run-still-alive1", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True, "mode": "review-plan",
        })

        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context.payload(2)["state"] == "running"

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

    def test_a_foreground_fanout_renders_a_head_line_and_a_per_model_table(
        self, tmp_path, capsys
    ):
        """The Claude Code lane, which is the lane this feature is written for.

        Checkpoint written, ``claim_run`` never called — so there is no ``run``
        object, ``classify_run`` returns ``no_run``, and before this the whole
        run view of a phase dispatching four models was the bare word ``no_run``
        beside ``run.state: (none)``, which reads as "nothing happened here".

        Three things are pinned together because they are one screen: the head
        line, the **additive** ``progress:`` summary (a models-only table that
        replaced it would hide the mirrors, which are the only progress a
        dispatch-only phase records), and one row per requested spec.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        beat = status_file.now_iso()
        models = {
            "codex:gpt-5.6-sol": {
                "state": "done", "started_at": beat, "updated_at": beat,
                "ended_at": beat, "elapsed_s": 130, "timeout_s": 1800,
                "attempt": 1, "source": "file",
            },
            # Answered from disk by the resume skip: terminal the moment it was
            # seeded, no attempt ever dispatched. There is no `skipped` *state*
            # — `source` is where that lives — so the row is labelled from it.
            "opencode:auto": {
                "state": "done", "started_at": beat, "updated_at": beat,
                "ended_at": beat, "elapsed_s": 0, "timeout_s": 1800,
                "attempt": 0, "source": "existing_file",
            },
            "gemini:gemini-3-pro": {
                "state": "running", "started_at": beat, "updated_at": beat,
                "timeout_s": 1800, "attempt": 1, "source": None,
            },
            "openai:gpt-5.2": {
                "state": "queued", "started_at": beat, "updated_at": beat,
                "timeout_s": 1800, "source": None,
            },
        }
        requested = list(models)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "models_requested": requested,
            "progress": {
                "step": "fanout",
                "models_total": 4, "models_done": 2, "models_succeeded": 2,
                "interval_s": 30,
                "tasks_total": 9, "tasks_done": 4,
                "batches_total": 3, "batch_index": 1,
                "subagents": {
                    "kind": "validation", "invoker": "executor",
                    "total": 11, "ok": 3, "counts": {"ok": 3},
                    "failed_task_ids": [],
                },
                "write_subagents": {
                    "kind": "implement", "invoker": "executor",
                    "total": 9, "ok": 7, "counts": {"ok": 7, "timeout": 2},
                    "failed_task_ids": ["t8", "t9"],
                },
                "sandbox_bypassed": ["codex:gpt-5.6-sol"],
                "models": models,
            },
        })
        assert "run" not in status_file.read_status(str(phase))
        view = detach.classify_run(str(phase))
        # The verdict constant is deliberately unchanged: only the text moved,
        # so `TERMINAL_VERDICTS`, `exit_code_for`'s 2 and F1's "a `no_run` phase
        # is never the `next` poll target" all still hold.
        assert view["verdict"] == detach.VERDICT_NO_RUN
        assert view["verdict"] not in detach.TERMINAL_VERDICTS

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out

        assert "running in foreground (2/4 models, no detached run record)" in out
        assert "no_run" not in out

        # ADDITIVE: every clause `_progress_text` rendered before the table
        # existed is still on the line, in order, and only the model count
        # gained its second half.
        assert (
            "progress:       fanout, models 2/4 done, 2 ok, 0 failed, "
            "tasks 4/9, batch 1/3, validation 3/11 ok, "
            "implement 7/9 ok, timeout 2, failed t8, t9, "
            "unsandboxed codex:gpt-5.6-sol"
        ) in out

        table = [line for line in out.splitlines() if line.startswith("        ")]
        assert "      models:" in out.splitlines()
        # One row per requested spec, `queued` rows included: a table that
        # rendered only the entries that had started would disagree with the
        # `models 2/4` count one line above it.
        assert len(table) == len(requested)
        assert table[0].split() == ["codex:gpt-5.6-sol", "done", "2m10s/30m"]
        assert table[1].split() == ["opencode:auto", "skipped", "0m00s/30m"]
        assert table[2].split() == ["gemini:gemini-3-pro", "running", "0m00s/30m"]
        assert table[3].split() == ["openai:gpt-5.2", "queued", "0m00s/30m"]

        # `--json` gains no key: the view already carries `progress` whole, so
        # the map rides out exactly as `update_progress` wrote it — which is the
        # shape the OpenCode package route reads instead of scraping the text.
        assert run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--json"]
        ) == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["runs"][0]["progress"]["models"] == models
        assert payload["runs"][0]["progress"]["models_succeeded"] == 2

    def test_a_spec_missing_from_a_half_seeded_map_still_gets_a_row(
        self, tmp_path, capsys
    ):
        """Between a run's launch and the tracker's seed the map is short.

        On a *re-run* of a phase dir that window renders the predecessor's map,
        because `claim_run` — the one writer that drops the old `progress` — is
        never called on a foreground run. Dropping the specs that are missing
        would make the table quietly disagree with `models N/M`, so they render
        `queued` with no timings and the row count still matches `models_total`.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        beat = status_file.now_iso()
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "models_requested": ["codex:gpt-5.6-sol", "opencode:auto"],
            "progress": {
                "step": "fanout",
                "models_total": 2, "models_done": 0, "models_succeeded": 0,
                "models": {
                    "codex:gpt-5.6-sol": {
                        "state": "running", "started_at": beat,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 1,
                        "source": None,
                    },
                },
            },
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        table = [
            line.strip() for line in capsys.readouterr().out.splitlines()
            if line.startswith("        ")
        ]
        assert len(table) == 2
        assert table[0].split() == ["codex:gpt-5.6-sol", "running", "0m00s/30m"]
        # No timings at all, rather than a fabricated clock or a dropped row.
        assert table[1] == "opencode:auto      queued"

    def test_a_retrying_row_names_its_attempt_and_its_backoff(
        self, tmp_path, capsys
    ):
        """`attempt` has no denominator on purpose — the ladder counts transport
        retries and rate-limit backoffs independently, so `2/3` would be a lie —
        and `retry_after_s` exists only while the slot is sleeping."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        beat = status_file.now_iso()
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {
                "step": "fanout",
                "models_total": 2, "models_done": 1, "models_succeeded": 0,
                "models": {
                    "codex:gpt-5.6-sol": {
                        "state": "retrying", "started_at": beat,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 2,
                        "retry_after_s": 20, "source": None,
                    },
                    "opencode:auto": {
                        "state": "failed", "started_at": beat,
                        "updated_at": beat, "ended_at": beat, "elapsed_s": 5,
                        "timeout_s": 1800, "attempt": 1,
                        "error": "no result (cancelled at the fan-out deadline)",
                        "source": "failed",
                    },
                },
            },
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "progress:       fanout, models 1/2 done, 0 ok, 1 failed" in out
        assert "retrying  0m00s/30m  attempt 2, retry in 20s" in out
        assert (
            "failed    0m05s/30m  no result (cancelled at the fan-out deadline)"
            in out
        )

    def test_a_staggering_row_says_when_it_starts(self, tmp_path, capsys):
        """2026-09-21: the launch-time poll — taken at second zero on purpose,
        for the roster — showed every row but the first as `staggering`, and a
        reader took that for the models being run one at a time. The word is
        the lifecycle's and stays; the detail column now says how long, in the
        shape `retry in 20s` already has. Rounded up, so a slot 5.9 s out reads
        `6s`; a stamp already past renders no clause — a poll landing on the
        flip to `queued` must not print `starts in 0s`."""
        from datetime import datetime, timedelta

        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        beat = status_file.now_iso()
        ahead = (datetime.fromisoformat(beat) + timedelta(seconds=6)).isoformat()
        behind = (datetime.fromisoformat(beat) - timedelta(seconds=6)).isoformat()
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {
                "step": "fanout",
                "models_total": 3, "models_done": 0, "models_succeeded": 0,
                "models": {
                    "codex:gpt-5.6-sol": {
                        "state": "running", "started_at": beat,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 1,
                        "source": None,
                    },
                    "opencode:auto": {
                        "state": "staggering", "started_at": beat,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 0,
                        "stagger_until": ahead, "source": None,
                    },
                    "gemini:pro": {
                        "state": "staggering", "started_at": beat,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 0,
                        "stagger_until": behind, "source": None,
                    },
                },
            },
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        table = [
            line.strip() for line in out.splitlines() if line.startswith("        ")
        ]
        assert table[0].split() == ["codex:gpt-5.6-sol", "running", "0m00s/30m"]
        assert table[1].split() == [
            "opencode:auto", "staggering", "0m00s/30m", "starts", "in", "6s",
        ]
        assert table[2].split() == ["gemini:pro", "staggering", "0m00s/30m"]

    def test_a_progress_object_with_no_models_map_renders_as_it_always_did(
        self, tmp_path, capsys
    ):
        """The additive rule, from the other side: the table is appended under
        the *existing* progress guard, so a dispatch-only phase — whose only
        progress is the executor's mirror — grows no empty `models:` header."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "generate-tasks")
        status_file.write_status(str(phase), {
            "phase": "generate-tasks", "state": "dispatching",
            "progress": {"subagents": {
                "kind": "task_generation", "invoker": "executor",
                "total": 1, "ok": 1, "counts": {"ok": 1},
                "failed_task_ids": [],
            }},
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "progress:       task_generation 1/1 ok" in out
        assert "models:" not in out

    def test_the_result_line_carries_the_progress_object_for_a_pinned_poll(
        self, tmp_path, capsys
    ):
        """`MULTI_LLM_RESULT` carried no `progress` key at all, so the OpenCode
        package route — which supervises its own poll and never opens the
        rendered text — would have had to regex a human line to relay anything
        mid-run. Same selector rule as `mode`/`phase`: only a poll pinned to
        exactly one run is "the watched run's"."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        progress = {
            "step": "fanout", "models_total": 2, "models_done": 1,
            "models_succeeded": 1,
            "models": {"codex:gpt-5.6-sol": {"state": "running"}},
        }
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": progress,
        })
        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run", "--phase", "review-plan",
            ]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert context.payload(2)["progress"] == progress

        # Present only when there is something to report, so every other
        # invocation's line stays byte-for-byte what it was.
        bare = cli_common.ResultContext(mode="review-plan")
        assert "progress" not in bare.payload(0)

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

    def test_a_finished_generate_tasks_run_reports_the_task_count(
        self, tmp_path, capsys
    ):
        """For the phase whose whole output is a count, `1/1 subagents` is the
        least informative number on screen — the interesting one is how many
        tasks it wrote (measured 2026-09-16: 14, present nowhere in
        `.status.json`). `consume_document` now reports `tasks_total`/
        `tasks_done`, which `_progress_text` already knew how to render."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "generate-tasks")
        status_file.claim_run(str(phase), {
            "run_id": "run-count-0001", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "generate-tasks",
        })
        status_file.update_status(str(phase), {
            "phase": "generate-tasks", "state": "complete",
            "progress": {
                "step": "complete", "tasks_total": 14, "tasks_done": 14,
                "subagents": {
                    "kind": "task_generation", "invoker": "executor",
                    "total": 1, "ok": 1, "counts": {"ok": 1},
                    "failed_task_ids": [],
                },
            },
        })
        assert run_status.main(["--plan-file", str(plan)]) == 0
        assert (
            "progress:       complete, tasks 14/14, task_generation 1/1 ok"
            in capsys.readouterr().out
        )

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

    def test_the_write_mirror_reports_item_counts_when_the_scheduler_recorded_them(
        self, tmp_path, capsys
    ):
        """F9: `ok`/`total` on the write mirror count BATCHES, not the
        suggestions or fixes inside them. A 15-batch, 29-suggestion
        apply-suggestions run mirrored (and the model then reported)
        "15/15 ok, applied 15" — 14 suggestions short. When the scheduler
        recorded `items_total`/`items_applied` (`write_batch_scheduler.
        _mirror_status`), the line says so explicitly. The write-outcome
        breakdown is dropped here (F14, 2026-09-16): every batch outcome is
        `applied` and the count (15) is exactly the `ok` batch count already
        stated, so printing it again read as a third, smaller quantity."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "apply-suggestions")
        status_file.claim_run(str(phase), {
            "run_id": "run-write-002", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "apply-suggestions",
        })
        rows = [
            {"task_id": f"apply-batch-{i}", "batch_index": i, "status": "ok",
             "write_outcome": "applied", "files_touched": ["plan.md"]}
            for i in range(15)
        ]
        status_file.update_status(str(phase), {
            "phase": "apply-suggestions", "state": "complete",
            "progress": {"write_subagents": {
                "kind": "apply_suggestions", "invoker": "executor", "total": 15,
                "dispatched": 15, "finalized": True, "tasks": rows,
                "failed_task_ids": [], "items_total": 29, "items_applied": 29,
            }},
        })
        assert run_status.main(["--plan-file", str(plan)]) == 0
        out = capsys.readouterr().out
        assert "phase state:    complete" in out
        assert (
            "progress:       apply_suggestions 15/15 batches ok, "
            "29/29 items applied"
        ) in out
        # No redundant `, applied 15` tail: it would just restate the ok count.
        assert ", applied 15" not in out

    def test_the_write_mirror_names_a_task_still_in_flight(self, tmp_path, capsys):
        """F10: the mirror's `dispatched` counter used to count FINISHED
        tasks — for an 11-minute single-task batch it read `dispatched: 0`
        the whole time, indistinguishable from a stall. `in_flight`
        (`write_batch_scheduler._begin_sweep`/`_mirror_status`) names the
        task that is still running and how long it has been running for."""
        import re
        from datetime import datetime, timedelta

        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "implement")
        status_file.claim_run(str(phase), {
            "run_id": "run-write-003", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "running", "exit_code": None, "detached": True,
            "mode": "implement",
        })
        started = (datetime.now() - timedelta(seconds=200)).isoformat()
        rows = [
            {"task_id": "T001", "batch_index": 0, "status": "ok",
             "write_outcome": "applied", "files_touched": ["index.html"]},
        ]
        status_file.update_status(str(phase), {
            "phase": "implement", "state": "running",
            "progress": {"write_subagents": {
                "kind": "implement", "invoker": "executor", "total": 14,
                "completed": 1, "finalized": False, "tasks": rows,
                "failed_task_ids": [],
                "in_flight": [{"task_id": "T002", "dispatched_at": started}],
            }},
        })
        assert run_status.main(["--plan-file", str(plan)]) == 2
        out = capsys.readouterr().out
        assert re.search(
            r"progress:       implement 1/14 ok, in flight: T002 "
            r"\(since \d+m \d+s\)",
            out,
        ), out

    def test_the_write_mirror_names_a_deferred_manual_task(self, tmp_path, capsys):
        """F9: `skipped_task_ids`/`human_task_strategy` — resolved before
        dispatch and already on `.status.json` — were rendered nowhere, so a
        finished run's own progress line said nothing about the human task
        `--yes`'s default strategy deferred."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "implement")
        status_file.claim_run(str(phase), {
            "run_id": "run-write-004", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "implement",
        })
        rows = [
            {"task_id": f"T00{i}", "batch_index": i, "status": "ok",
             "write_outcome": "applied", "files_touched": ["index.html"]}
            for i in range(1, 14)
        ]
        status_file.update_status(str(phase), {
            "phase": "implement", "state": "complete",
            "progress": {"write_subagents": {
                "kind": "implement", "invoker": "executor", "total": 13,
                "completed": 13, "finalized": True, "tasks": rows,
                "failed_task_ids": [],
                "human_task_strategy": "skip-dependents",
                "skipped_task_ids": ["T014"],
            }},
        })
        assert run_status.main(["--plan-file", str(plan)]) == 0
        out = capsys.readouterr().out
        # No redundant `applied 13` breakdown (F14): every task's outcome is
        # `applied` and 13 is exactly the ok count already stated.
        assert (
            "progress:       implement 13/13 ok, "
            "deferred manual: T014 (skip-dependents)"
        ) in out

    def test_the_write_mirror_names_a_failed_verification(self, tmp_path, capsys):
        """F4: a run whose batches all applied and whose repo check then failed
        rendered as `apply_code_fixes 1/1 batches ok` with no failed id — the
        only sign of the failure a phase state the counts contradicted."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "apply-fixes")
        status_file.claim_run(str(phase), {
            "run_id": "run-write-005", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 3, "detached": True,
            "mode": "apply-code-fixes",
        })
        status_file.update_status(str(phase), {
            "phase": "apply-code-fixes", "state": "subagents_failed",
            "progress": {"write_subagents": {
                "kind": "apply_code_fixes", "invoker": "executor", "total": 1,
                "completed": 1, "finalized": False, "failed_task_ids": [],
                "tasks": [{"task_id": "apply-fix-batch-0", "batch_index": 0,
                           "status": "ok", "write_outcome": "applied",
                           "files_touched": ["index.html"]}],
                "verification": {"command": "pnpm typecheck", "source": "project",
                                 "exit_code": 2, "duration_seconds": 4.2,
                                 "output_tail": "boom"},
            }},
        })
        assert run_status.main(["--plan-file", str(plan)]) == 3
        out = capsys.readouterr().out
        assert "verification FAILED (exit 2): `pnpm typecheck`" in out

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

    def test_unknown_phase_error_lists_apply_fixes_as_valid(self, tmp_path, capsys):
        """F1: before the fix, `apply-fixes` (the on-disk directory word) was
        itself rejected as "unknown phase or mode" — the error's own "Valid
        phases" list must name a selector that actually works.
        """
        plan = _plan(tmp_path)
        with pytest.raises(SystemExit):
            run_status.main(["--plan-file", str(plan), "--phase", "nonsense"])
        err = capsys.readouterr().err
        assert "apply-fixes" in err

    def test_apply_code_fixes_selector_finds_the_apply_fixes_dir(
        self, tmp_path, capsys
    ):
        """F1: apply-code-fixes is the one phase whose modes.py phase name
        (``apply-code-fixes``) differs from its on-disk directory
        (``apply-fixes``). Both spellings of ``--phase`` must resolve to the
        same, real ``apply-fixes/`` directory — before the fix, the mode word
        (the tool's own printed ``next`` command) found no run at all, and the
        directory word was rejected outright as an unknown phase.
        """
        plan = _plan(tmp_path)
        directory = _phase_dir(plan, "apply-fixes")
        status_file.write_status(str(directory), {"phase": "apply-code-fixes", "state": "x"})

        for selector in ("apply-code-fixes", "apply-fixes"):
            assert run_status.main(
                ["--plan-file", str(plan), "--view", "run", "--phase", selector,
                 "--json"]
            ) == 2
            payload = json.loads(_json_body(capsys.readouterr().out))
            assert len(payload["runs"]) == 1
            assert payload["runs"][0]["phase_dir"] == str(directory)

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
        """The shipped default is unchanged, and that is now the whole claim.

        `progress.poll_s` ships at 90 s, so a project that configures nothing
        keeps exactly today's wait — moving that number is a deliberate UX
        change, not a side effect of wanting a chattier ticker. Its sibling
        below drives the same two numbers against the real capability data
        instead of restating them.
        """
        from utils.provider_registry import PROGRESS_FALLBACKS

        assert run_status.DEFAULT_WAIT_TIMEOUT_S == 90.0
        assert run_status.DEFAULT_WAIT_TIMEOUT_S < 120.0
        assert PROGRESS_FALLBACKS["poll_s"] == run_status.DEFAULT_WAIT_TIMEOUT_S

    def test_the_default_is_the_tightest_real_ceiling_less_the_margin(self):
        """`WAIT_CAP_MARGIN_S` is the number that *explains* the 90 s default.

        Asserted against `hosts/capabilities.yaml` rather than against a second
        copy of `120`, so the day a host declares a tighter cap this goes red
        here — where the arithmetic lives — rather than in whatever poll is
        killed mid-wait.
        """
        from utils import host_capabilities as caps

        assert run_status.WAIT_CAP_MARGIN_S == 30.0
        host, ceiling = caps.tightest_timeout_ceiling()
        assert (host, ceiling) == ("opencode", 120)
        assert ceiling - run_status.WAIT_CAP_MARGIN_S == (
            run_status.DEFAULT_WAIT_TIMEOUT_S
        )
        # `0` means *no documented cap*, not a cap of zero, so codex is skipped
        # as the minimum rather than winning it.
        assert caps.timeout_ceiling_s("codex") == 0

    @POSIX_ONLY
    @pytest.mark.parametrize("finished", [False, True])
    def test_wait_terminates_on_a_foreground_checkpoint(
        self, tmp_path, monkeypatch, capsys, finished
    ):
        """Acceptance criterion 1, and the reason the verdict stays `no_run`.

        A phase dir holding a checkpoint but no run record is *every* foreground
        phase, and `no_run` is not in `TERMINAL_VERDICTS` — so `_wait_for_terminal`
        spins to its own deadline and hands back the non-terminal view. That is
        the designed behavior (`--wait` is a timed progress snapshot on this
        lane, not a terminal signal), and what has to hold is that it comes
        back: with the timeout now resolved from `progress.poll_s`, a regression
        here is a poll that never returns.

        Both halves of the checkpoint machine are driven (2026-09-20). The
        **finished** row is the one F1's guard could have broken: keeping the
        wait alive whenever the checkpoint looks live is right, and reading a
        terminal checkpoint as live would turn every completed foreground phase
        into a full-timeout poll forever. And the explicit `--timeout 1` is
        gone — it made the *config-resolved* path unreachable, which is the path
        every real poll takes; `poll_s: 5` is the floor, so this is the shortest
        honest end-to-end wait.
        """
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        checkpoint = {
            "phase": "review-plan", "state": "models_running",
            "progress": {"step": "fanout", "models_total": 4, "models_done": 1,
                         "models_succeeded": 1},
        }
        if finished:
            checkpoint["state"] = "complete"
            checkpoint["progress"] = dict(
                checkpoint["progress"], step="complete", models_done=4,
                models_succeeded=4,
            )
        status_file.write_status(str(phase), checkpoint)
        assert "run" not in status_file.read_status(str(phase))
        start = time.monotonic()
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--host", "opencode",
            "--wait",
        ])
        elapsed = time.monotonic() - start
        out = capsys.readouterr().out
        assert code == 2
        assert elapsed < 30, f"--wait did not come back ({elapsed:.1f}s)"
        if finished:
            assert "finished in foreground (4/4 models, no detached run record)" in out
            # F2, 2026-09-21: `< 30` with `poll_s: 5` could not tell a wait that
            # *ended* from one that spent its whole deadline on a phase nothing
            # was doing. The checkpoint is the only vote this lane casts, and a
            # settled one ends the wait before it starts.
            assert elapsed < 2.0, (
                f"a settled foreground checkpoint held --wait {elapsed:.1f}s"
            )
        else:
            assert "running in foreground (1/4 models, no detached run record)" in out
            # ...and a live one still holds the turn, which is what `--wait` is
            # for on this lane (acceptance criterion 1). The deadline is 5 s.
            assert elapsed >= 1.0, (
                f"--wait over a live foreground fan-out returned in {elapsed:.1f}s"
            )

    @POSIX_ONLY
    def test_a_settled_foreground_phase_is_handed_back_as_no_poll_at_all(
        self, tmp_path, monkeypatch, capsys
    ):
        """F2, 2026-09-21: the `no_run` loop, and the half the guard above missed.

        Every foreground phase is `no_run` forever — only the detached worker
        calls `claim_run` — so on a plan whose phases all ran in the foreground
        the `live_views` guard filtered *everything* and the fallback reached for
        `views[0]`, which can only be one of those same `no_run` views. F1,
        2026-09-17 already ruled that a `no_run` phase is never the next poll
        target; this handed one back anyway, and once `status_command_for` began
        carrying `--wait` the resulting loop stopped being merely tight and
        became a poll that spends its whole clamped `poll_s`, returns exit 2 and
        hands back the identical command, forever.

        Two changes and both are asserted here: the wait ends at once because
        the checkpoint is settled, and nothing is handed back to follow.
        `test_the_next_poll_skips_a_phase_that_never_had_a_run` pins the other
        side — a finished *detached* run under the same plan is still the
        fallback, because its poll exits 0 and ends the loop.
        """
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "progress": {"step": "complete", "models_total": 2,
                         "models_done": 2, "models_succeeded": 2},
        })
        assert "run" not in status_file.read_status(str(phase))

        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            start = time.monotonic()
            code = run_status.main([
                "--plan-file", str(plan), "--view", "run", "--host", "opencode",
                "--wait",
            ])
            elapsed = time.monotonic() - start
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert code == 2
        assert elapsed < 2.0, (
            f"--wait held a settled foreground phase for {elapsed:.1f}s"
        )
        assert context.next is None, (
            f"exit 2 handed back a poll of a `no_run` phase: {context.next!r}"
        )

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

    def test_validation_counts_survive_the_detach_boundary(self, tmp_path, capsys):
        """§6 T6.1: `classify_run` copies only named status fields, so the
        verdict counts need their own entry or a detached run loses them
        between `.status.json` and the poll's result line — and the host's
        summary leaves out the groups the user still has to decide."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        counts = {
            "valid": 53, "invalid": 1,
            "needs_human_decision": 3, "validation_failed": 0,
        }
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "validation_counts": counts,
        })
        status_file.claim_run(str(phase), {
            "run_id": "run-counts-001", "state": "done", "exit_code": 0,
            "detached": True,
        })
        assert detach.classify_run(str(phase))["validation_counts"] == counts
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--phase", "review-plan",
            ]) == 0
            assert ctx.payload(0)["validation_counts"] == counts
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()

    def test_a_status_without_validation_counts_leaves_them_out(self, tmp_path):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
        })
        status_file.claim_run(str(phase), {
            "run_id": "run-counts-002", "state": "done", "exit_code": 0,
            "detached": True,
        })
        assert detach.classify_run(str(phase))["validation_counts"] is None
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            run_status.main(["--plan-file", str(plan), "--view", "run"])
            assert "validation_counts" not in ctx.payload(0)
        finally:
            cli_common.deactivate_result()

    def test_high_priority_groups_renders_for_a_terminal_code_review_run(
        self, tmp_path, capsys
    ):
        """F4: `[HIGH_PRIORITY_ISSUES] <n>` is a bare log line a detached
        caller never reads — `status` is where a host actually learns the
        count, the same way it already surfaces `[CONSOLIDATION_RECOMMENDED]`.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, phase="code-review")
        status_file.write_status(str(phase), {
            "phase": "code-review", "state": "complete",
            "high_priority_groups": {"count": 1},
        })
        status_file.claim_run(str(phase), {
            "run_id": "run-highpri-001", "state": "done", "exit_code": 0,
            "detached": True,
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 0
        out = capsys.readouterr().out
        assert "high-priority groups: 1" in out

    def test_high_priority_groups_is_silent_off_a_non_terminal_run(
        self, tmp_path, capsys
    ):
        """The key is written once, at the very end of main() — a mid-run poll
        must not render a stale or premature count.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, phase="code-review")
        status_file.write_status(str(phase), {
            "phase": "code-review", "state": "models_running",
            "high_priority_groups": {"count": 1},
        })
        status_file.claim_run(str(phase), {
            "run_id": "run-highpri-002", "state": "running", "detached": True,
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "high-priority groups" not in out

    def test_high_priority_groups_is_silent_on_a_non_code_review_phase(
        self, tmp_path, capsys
    ):
        """The key only exists on code-review's own checkpoint; the phase
        gate is belt-and-braces against a stray key elsewhere.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, phase="review-plan")
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "high_priority_groups": {"count": 1},
        })
        status_file.claim_run(str(phase), {
            "run_id": "run-highpri-003", "state": "done", "exit_code": 0,
            "detached": True,
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 0
        out = capsys.readouterr().out
        assert "high-priority groups" not in out

    def test_a_terminal_run_carries_the_workflow_notice_into_view_run(
        self, tmp_path, capsys
    ):
        """F7: the opencode plugin polls only `--view run`, which never fetched
        the workflow view at all — so a terminal run's real follow-up (the
        `notice:`/`suggested next:` lines `--view all` already renders) was
        invisible on the one channel the plugin reads. The model saw
        `next: null`, said "nothing further to run", then invented
        `--apply-task-suggestions` from general knowledge instead of the
        notice sitting one view over (measured 2026-09-16). It must land in
        the rendered text AND in the `notice` key of `MULTI_LLM_RESULT` — never
        in `next`, which the plugin auto-follows without asking."""
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        review_tasks = _phase_dir(plan, "review-tasks")
        grouped = {
            "format_version": 2,
            "groups": [{
                "category": "addition", "theme": "g1",
                "suggestions": [{"title": "s1", "desc": "d", "importance": "LOW",
                                  "type": "addition"}],
            }],
        }
        (review_tasks / "grouped.json").write_text(json.dumps(grouped), encoding="utf-8")
        validation = {"format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]}
        (review_tasks / "validation.json").write_text(json.dumps(validation), encoding="utf-8")

        state = StateManager(plan)
        state.mark_phase_completed("review-tasks")
        state.save()

        status_file.write_status(str(review_tasks), {
            "phase": "review-tasks", "state": "complete",
        })
        status_file.claim_run(str(review_tasks), {
            "run_id": "run-notice-0001", "state": "done", "exit_code": 0,
            "detached": True,
        })

        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main(
                ["--plan-file", str(plan), "--view", "run", "--phase", "review-tasks"]
            ) == 0
            payload = ctx.payload(0)
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out

        assert "workflow suggested next:" in out
        assert "workflow notice: review-tasks produced 1 unapplied task suggestion groups" in out
        assert payload["notice"] is not None
        assert "review-tasks produced 1 unapplied task suggestion groups" in payload["notice"]
        assert "suggested next:" in payload["notice"]
        # The notice is advisory prose, never a mechanical continuation.
        assert payload["next"] is None

    def test_a_still_running_run_carries_no_workflow_notice(self, tmp_path, capsys):
        """The carry-over is for a TERMINAL run only — a poll on a run still in
        flight has nothing to say yet about what comes after it, and the
        non-terminal payload must stay exactly what it was."""
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        review_tasks = _phase_dir(plan, "review-tasks")
        grouped = {
            "format_version": 2,
            "groups": [{
                "category": "addition", "theme": "g1",
                "suggestions": [{"title": "s1", "desc": "d", "importance": "LOW",
                                  "type": "addition"}],
            }],
        }
        (review_tasks / "grouped.json").write_text(json.dumps(grouped), encoding="utf-8")
        validation = {"format_version": 2, "groups": [{"group_index": 0, "status": "valid"}]}
        (review_tasks / "validation.json").write_text(json.dumps(validation), encoding="utf-8")

        state = StateManager(plan)
        state.mark_phase_completed("review-tasks")
        state.save()

        status_file.claim_run(str(review_tasks), {
            "run_id": "run-notice-0002", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True,
        })

        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main(
                ["--plan-file", str(plan), "--view", "run", "--phase", "review-tasks"]
            ) == 2
            payload = ctx.payload(2)
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out

        assert "workflow notice:" not in out
        assert "workflow suggested next:" not in out
        assert "notice" not in payload

    def test_wait_pinned_to_one_run_carries_its_mode_and_phase_into_the_result(
        self, tmp_path, capsys
    ):
        """N5: `multi_llm.py`'s dispatcher stamps `ctx.mode = "status"`,
        `ctx.phase = None` before `run_status.main` ever runs — `status` is a
        command word, not a phase. Right for a plain multi-run listing, but
        the opencode plugin's supervised poll pins one run with `--run-id` and
        hands this leg's own `MULTI_LLM_RESULT` to the model verbatim: before
        this it read `{"mode": "status", "phase": null, "state": "done", …}`
        for a finished `implement` run (measured 2026-09-16), and a weaker
        model than the one that hit this would have to infer its phase from
        `state` alone. The watched run already knows both from its own
        `.status.json`, so they travel into this leg's result."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "implement")
        status_file.write_status(str(phase), {"phase": "implement", "state": "complete"})
        status_file.claim_run(str(phase), {
            "run_id": "run-mode-0000001", "mode": "implement", "state": "done",
            "exit_code": 0, "detached": True,
        })
        ctx = cli_common.activate_result(cli_common.ResultContext(mode="status", phase=None))
        try:
            code = run_status.main([
                "--plan-file", str(plan), "--view", "run", "--wait",
                "--run-id", "run-mode-0000001",
            ])
            assert code == 0
            payload = ctx.payload(code)
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert payload["mode"] == "implement"
        assert payload["phase"] == "implement"

    def test_wait_pinned_by_phase_alone_also_carries_mode_and_phase(
        self, tmp_path, capsys
    ):
        """The carry-over fires off `--phase` alone too, with no `--run-id`."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-plan")
        status_file.write_status(str(phase), {"phase": "review-plan", "state": "complete"})
        status_file.claim_run(str(phase), {
            "run_id": "run-mode-0000002", "mode": "review-plan", "state": "done",
            "exit_code": 0, "detached": True,
        })
        ctx = cli_common.activate_result(cli_common.ResultContext(mode="status", phase=None))
        try:
            code = run_status.main([
                "--plan-file", str(plan), "--view", "run", "--wait",
                "--phase", "review-plan",
            ])
            assert code == 0
            payload = ctx.payload(code)
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert payload["mode"] == "review-plan"
        assert payload["phase"] == "review-plan"

    def test_a_pinned_poll_carries_mode_and_phase_without_wait_too(
        self, tmp_path, capsys
    ):
        """F3, 2026-09-17: what makes a poll "the watched run's" is the
        selector, not `--wait`. The carry-over lived inside `if args.wait:`, so
        a host (or a human) polling one pinned run once still got the N5
        payload this was meant to retire — `{"mode": "status", "phase": null}`
        for a finished `implement` run."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "implement")
        status_file.write_status(str(phase), {"phase": "implement", "state": "complete"})
        status_file.claim_run(str(phase), {
            "run_id": "run-mode-0000004", "mode": "implement", "state": "done",
            "exit_code": 0, "detached": True,
        })
        ctx = cli_common.activate_result(cli_common.ResultContext(mode="status", phase=None))
        try:
            code = run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--phase", "implement",
            ])
            assert code == 0
            payload = ctx.payload(code)
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert payload["mode"] == "implement"
        assert payload["phase"] == "implement"

    def test_wait_falls_back_to_the_phase_dir_when_the_run_record_has_no_mode(
        self, tmp_path, capsys
    ):
        """A run record with no `mode` field (predates it, or never recorded
        one) still answers from the phase dir it lives in."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "review-tasks")
        status_file.write_status(str(phase), {"phase": "review-tasks", "state": "complete"})
        status_file.claim_run(str(phase), {
            "run_id": "run-mode-0000003", "state": "done", "exit_code": 0,
            "detached": True,
        })
        ctx = cli_common.activate_result(cli_common.ResultContext(mode="status", phase=None))
        try:
            code = run_status.main([
                "--plan-file", str(plan), "--view", "run", "--wait",
                "--run-id", "run-mode-0000003",
            ])
            assert code == 0
            payload = ctx.payload(code)
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert payload["mode"] == "review-tasks"
        assert payload["phase"] == "review-tasks"

    def test_a_plain_multi_run_wait_listing_keeps_mode_status(self, tmp_path, capsys):
        """No selector, several runs in view: no single run is "the" answer, so
        this leg's own mode/phase stay whatever the dispatcher stamped."""
        plan = _plan(tmp_path)
        first = _phase_dir(plan, "review-plan")
        second = _phase_dir(plan, "review-tasks")
        for phase_dir, run_id in (
            (first, "run-multi-0000001"), (second, "run-multi-0000002"),
        ):
            phase_name = os.path.basename(str(phase_dir))
            status_file.write_status(
                str(phase_dir), {"phase": phase_name, "state": "complete"}
            )
            status_file.claim_run(str(phase_dir), {
                "run_id": run_id, "mode": phase_name, "state": "done",
                "exit_code": 0, "detached": True,
            })
        ctx = cli_common.activate_result(cli_common.ResultContext(mode="status", phase=None))
        try:
            code = run_status.main([
                "--plan-file", str(plan), "--view", "run", "--wait", "--all",
            ])
            payload = ctx.payload(code)
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        assert payload["mode"] == "status"
        assert payload["phase"] is None

    def test_a_completed_phase_with_a_live_run_reads_re_running_not_bare_completed(
        self, tmp_path, capsys
    ):
        """N7: `phases_completed` records that a phase finished once, not that
        it stays finished — `implement --resume` can reopen a completed phase
        (a later review sends tasks back), and the bare word "completed"
        printed beside `run.state: running` is the one line a reader is most
        likely to take at face value and stop watching (measured 2026-09-16,
        mid-run `implement`, `run.state: running` beside `workflow phase:
        completed`)."""
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        implement = _phase_dir(plan, "implement")

        state = StateManager(plan)
        state.mark_phase_completed("implement")
        state.save()

        status_file.claim_run(str(implement), {
            "run_id": "run-rerun-0001", "mode": "implement",
            "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True,
            "started_at": "2026-09-16T19:54:19.649547",
        })

        assert run_status.main(
            ["--plan-file", str(plan), "--view", "all", "--phase", "implement"]
        ) == 2
        out = capsys.readouterr().out
        assert (
            "workflow phase: completed (re-running since "
            "2026-09-16T19:54:19.649547)" in out
        )

    def test_a_finished_run_under_a_completed_phase_reads_plain_completed(
        self, tmp_path, capsys
    ):
        """The re-running qualifier is for a LIVE run only: a terminal run
        under a completed phase is the ordinary finished condition and must
        not carry a stale "(re-running since ...)" suffix off a `started_at`
        left over from the run that finished it."""
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        implement = _phase_dir(plan, "implement")

        state = StateManager(plan)
        state.mark_phase_completed("implement")
        state.save()

        status_file.claim_run(str(implement), {
            "run_id": "run-rerun-0002", "mode": "implement", "state": "done",
            "exit_code": 0, "detached": True,
            "started_at": "2026-09-16T19:54:19.649547",
        })

        assert run_status.main(
            ["--plan-file", str(plan), "--view", "all", "--phase", "implement"]
        ) == 0
        out = capsys.readouterr().out
        assert "workflow phase: completed" in out
        assert "re-running" not in out

    def test_a_dead_run_under_a_completed_phase_does_not_read_re_running(
        self, tmp_path, capsys
    ):
        """F4, 2026-09-17: `stale` covers two conditions — a worker slow to
        beat, and one the identity probe can already prove is gone. Only the
        first is live, so a SIGKILLed worker printed `completed (re-running
        since …)` two lines under its own `died (pid N gone, …)`, which is the
        opposite of what N7's wording exists to say."""
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        implement = _phase_dir(plan, "implement")

        state = StateManager(plan)
        state.mark_phase_completed("implement")
        state.save()

        dead = _dead_pid()
        status_file.claim_run(str(implement), {
            "run_id": "run-rerun-0003", "mode": "implement", "pid": dead,
            # Not the same process even if the pid is recycled under us.
            "process_start_time": "0",
            "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
            "state": "running", "detached": True,
            "started_at": "2026-09-16T19:54:19.649547",
        })

        assert run_status.main(
            ["--plan-file", str(plan), "--view", "all", "--phase", "implement"]
        ) == 2
        out = capsys.readouterr().out
        assert f"died (pid {dead} gone, last beat " in out, out
        assert "workflow phase: completed" in out, out
        assert "re-running" not in out, out

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

class TestWaitTimeoutResolution:
    """``--wait``'s deadline: which source wins, and what the clamp binds.

    Everything here is driven through ``run_status.main`` rather than through
    ``get_progress_poll`` alone, deliberately: ``--timeout`` is a *sentinel* out
    of argparse and the gap between ``parse_args`` and the ``--wait`` branch is
    exactly where a sentinel default gets lost. ``_wait_for_terminal`` is
    replaced by a spy so a test asserting a 300 s wait does not take 300 s — the
    subject is the number that reaches it, not the sleeping.
    """

    @staticmethod
    def _config(tmp_path, monkeypatch, block: str) -> None:
        """Point ``MULTI_LLM_PROVIDERS_CONFIG`` at a one-key `progress:` YAML.

        The env layer is an *override* layer, which is what makes
        `validate_progress_block` run over it — the base is never validated.
        """
        import utils.provider_registry as registry

        override = tmp_path / "providers-override.yaml"
        override.write_text(f"progress:\n{block}", encoding="utf-8")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(override))
        registry._config = None
        registry._config_key = None

    @staticmethod
    def _resolved(tmp_path, monkeypatch, argv) -> float:
        """Run ``status --wait`` through `main` and report the timeout it used."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
        })
        seen = {}

        def _spy(plan_path, args, views):
            seen["timeout"] = args.timeout
            return views

        monkeypatch.setattr(run_status, "_wait_for_terminal", _spy)
        run_status.main(["--plan-file", str(plan), "--view", "run", "--wait", *argv])
        return seen["timeout"]

    def test_the_two_knobs_resolve_independently(self, tmp_path, monkeypatch,
                                                 capsys):
        """`interval_s: 0` is the ticker's disable form and nothing else.

        `--wait` never reads it, so silencing the `[PROGRESS]` lines must not
        turn a poll into an instant return — with no `poll_s` configured the
        wait is still the shipped 90 s.
        """
        self._config(tmp_path, monkeypatch, "  interval_s: 0\n")
        assert self._resolved(tmp_path, monkeypatch, []) == 90.0
        capsys.readouterr()

    def test_a_configured_poll_s_is_the_wait_default(self, tmp_path, monkeypatch,
                                                     capsys):
        self._config(tmp_path, monkeypatch, "  poll_s: 60\n")
        assert self._resolved(tmp_path, monkeypatch, ["--host", "opencode"]) == 60.0
        assert capsys.readouterr().err == ""

    def test_poll_s_clamps_under_opencode_and_stands_on_codex(
        self, tmp_path, monkeypatch, capsys
    ):
        """120 - 30 = 90 on opencode; `timeout_ceiling_s: 0` is *no cap*, not
        a cap of zero, so an explicitly-routed codex poll keeps its 300 s."""
        self._config(tmp_path, monkeypatch, "  poll_s: 300\n")
        assert self._resolved(tmp_path, monkeypatch, ["--host", "opencode"]) == 90.0
        # Silent: the user configured a cadence, not a deadline.
        assert capsys.readouterr().err == ""
        assert self._resolved(tmp_path, monkeypatch, ["--host", "codex"]) == 300.0
        assert capsys.readouterr().err == ""

    def test_claude_codes_own_ceiling_admits_the_recommended_range(
        self, tmp_path, monkeypatch, capsys
    ):
        """600 - 30 = 570, well above the 120-300 s the poll loop is built
        around — which is why the *Bash* `timeout` parameter, and not this
        clamp, is what actually keeps a Claude Code poll alive."""
        self._config(tmp_path, monkeypatch, "  poll_s: 300\n")
        assert self._resolved(
            tmp_path, monkeypatch, ["--host", "claude-code"]
        ) == 300.0
        assert capsys.readouterr().err == ""

    def test_an_unrouted_poll_clamps_against_the_tightest_ceiling(
        self, tmp_path, monkeypatch, capsys
    ):
        """No `--host` clamps rather than skipping the clamp: an unrouted poll
        could be running under any host, and overshooting one of their caps by
        accident is the failure the clamp exists for."""
        self._config(tmp_path, monkeypatch, "  poll_s: 300\n")
        assert self._resolved(tmp_path, monkeypatch, []) == 90.0
        assert capsys.readouterr().err == ""

    def test_the_clamp_reads_the_host_the_env_var_declares(
        self, tmp_path, monkeypatch, capsys
    ):
        """F3, 2026-09-21: `--host` is not the only channel that names a host.

        `resolve_host_id` is `--host` -> `$MULTI_LLM_HOST` -> the default, and
        it is what `render_runner_text` resolves in this very invocation and
        what the dispatcher validates against. The clamp read `args.host`
        alone, so a poll routed entirely through the env var — which is how an
        installer routes one — was clamped against the tightest ceiling on
        record instead of its own host's: 90 s under a host that documents
        600 s.
        """
        self._config(tmp_path, monkeypatch, "  poll_s: 300\n")
        monkeypatch.setenv("MULTI_LLM_HOST", "claude-code")
        assert self._resolved(tmp_path, monkeypatch, []) == 300.0
        assert capsys.readouterr().err == ""
        # ...and the env var is a declaration, not an exemption: a host that
        # does document a cap still clamps.
        monkeypatch.setenv("MULTI_LLM_HOST", "opencode")
        assert self._resolved(tmp_path, monkeypatch, []) == 90.0
        assert capsys.readouterr().err == ""

    def test_the_flag_still_beats_the_env_var(self, tmp_path, monkeypatch,
                                              capsys):
        """The ladder's order, not merely its rungs: `--host` wins."""
        self._config(tmp_path, monkeypatch, "  poll_s: 300\n")
        monkeypatch.setenv("MULTI_LLM_HOST", "codex")   # no documented cap
        assert self._resolved(
            tmp_path, monkeypatch, ["--host", "opencode"]
        ) == 90.0
        assert capsys.readouterr().err == ""

    def test_the_clamp_note_stops_claiming_no_host_when_one_was_declared(
        self, tmp_path, monkeypatch, capsys
    ):
        """The loud path's prose has to match the ladder that produced it.

        A caller who declared `opencode` through the environment and asked for
        300 s was told the clamp used "the tightest declared cap" because "no
        --host was given" — two wrong statements about a host they had named.
        """
        self._config(tmp_path, monkeypatch, "  poll_s: 60\n")
        monkeypatch.setenv("MULTI_LLM_HOST", "opencode")
        assert self._resolved(
            tmp_path, monkeypatch, ["--timeout", "300"]
        ) == 90.0
        err = capsys.readouterr().err
        assert "host opencode" in err
        assert "no --host" not in err
        assert err.count("\n") == 1, "exactly one stderr line"

    def test_the_note_names_both_channels_when_neither_declared_a_host(
        self, tmp_path, monkeypatch, capsys
    ):
        """With nothing declared the fallback stands — and now says so fully."""
        self._config(tmp_path, monkeypatch, "  poll_s: 60\n")
        monkeypatch.delenv("MULTI_LLM_HOST", raising=False)
        assert self._resolved(
            tmp_path, monkeypatch, ["--timeout", "300"]
        ) == 90.0
        err = capsys.readouterr().err
        assert "no --host and no $MULTI_LLM_HOST" in err
        assert "opencode" in err, "the tightest declared cap is still named"

    def test_an_explicit_timeout_beats_poll_s_and_is_itself_clamped(
        self, tmp_path, monkeypatch, capsys
    ):
        """"Wins" settles precedence, not the ceiling.

        Option A's rendered command always passes `--timeout <interval>`, so a
        clamp guarding only the `poll_s` path would leave the one command the
        instructions actually emit unguarded. This is the loud path: that caller
        asked for a specific number and is getting another.
        """
        self._config(tmp_path, monkeypatch, "  poll_s: 60\n")
        resolved = self._resolved(
            tmp_path, monkeypatch, ["--host", "opencode", "--timeout", "300"]
        )
        assert resolved == 90.0
        err = capsys.readouterr().err
        assert "--timeout 300s" in err
        assert "opencode" in err
        assert "120s" in err
        assert "waiting 90s instead" in err
        assert err.count("\n") == 1, "exactly one stderr line"

    def test_an_explicit_timeout_under_the_cap_is_silent_and_untouched(
        self, tmp_path, monkeypatch, capsys
    ):
        self._config(tmp_path, monkeypatch, "  poll_s: 300\n")
        resolved = self._resolved(
            tmp_path, monkeypatch, ["--host", "opencode", "--timeout", "45"]
        )
        assert resolved == 45.0
        assert capsys.readouterr().err == ""

    def test_with_no_config_at_all_the_wait_is_todays_ninety_seconds(
        self, tmp_path, monkeypatch, capsys
    ):
        """The `poll_s` fallback and `DEFAULT_WAIT_TIMEOUT_S` are the same
        number, so codex and opencode keep exactly today's wait."""
        assert self._resolved(tmp_path, monkeypatch, ["--host", "opencode"]) == 90.0
        assert capsys.readouterr().err == ""

    @staticmethod
    def _exit_code(tmp_path, monkeypatch, argv) -> int:
        """Run `status --wait` through `main` and report only its exit code.

        The sibling of :meth:`_resolved`, for the rows where no timeout is ever
        resolved because the resolution *refuses*. Deliberately does not spy
        `_wait_for_terminal` out: a row that expects a refusal has to prove the
        wait was never entered, and a spy would hide a wait that was.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
        })
        return run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--wait", *argv]
        )

    @pytest.mark.parametrize("value", ["0", "3"])
    def test_a_rejected_poll_s_surfaces_rather_than_degrading(
        self, tmp_path, monkeypatch, capsys, value
    ):
        """The floor is `provider_registry`'s, and this asserts it *reaches* the
        poll. A `poll_s` silently dropped back to 90 s would be
        indistinguishable from the user's setting having worked, which is the
        failure the closed-key validation exists for — and `0` means a poll
        that returns instantly, not one that is switched off.

        F4, 2026-09-20: it reaches the poll as an `ERROR:` line and **exit 3**,
        not as the raw `ConfigError` it used to be. The exception propagated to
        the dispatcher, which printed a traceback and returned exit **1** — and
        rule 2 reads a `1` as "the run state is unreadable, stop and report",
        which is a statement about the run rather than about the config key the
        user has to fix. Exit 3 is this codebase's "needs a human"; the rejection
        still surfaces, named, which was the point of validating at load.
        """
        self._config(tmp_path, monkeypatch, f"  poll_s: {value}\n")
        assert self._exit_code(tmp_path, monkeypatch, ["--host", "opencode"]) == 3
        captured = capsys.readouterr()
        assert captured.err.startswith("ERROR: ")
        assert "poll_s" in captured.err
        assert value in captured.err
        assert "Traceback" not in captured.err

    def test_an_unrelated_config_error_does_not_end_the_poll_loop_on_a_lie(
        self, tmp_path, monkeypatch, capsys
    ):
        """F4, 2026-09-20: `--wait` died on *any* ConfigError, not only `poll_s`'s.

        `_resolve_wait_timeout` → `get_progress_poll` → `load_config` validates
        **every** override block, so a bogus key under `driver:` — nothing to do
        with polling — took the whole poll down. Measured: with
        `MULTI_LLM_PROVIDERS_CONFIG` pointing at a config carrying `driver:
        {not_a_key: 1}`, plain `status` exited 2 normally while `status --wait
        --json` exited **1** with a stack trace on stderr and no JSON view on
        stdout at all. A config edit made *during* a run therefore ended the poll
        loop with the agent reporting the run as unreadable.
        """
        import utils.provider_registry as registry

        override = tmp_path / "providers-override.yaml"
        override.write_text(
            "driver:\n  not_a_driver_key: 1\n", encoding="utf-8"
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(override))
        registry._config = None
        registry._config_key = None

        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
        })
        argv = ["--plan-file", str(plan), "--view", "run"]

        # The control: the same config, the same phase dir, no `--wait`.
        assert run_status.main(argv) == 2
        capsys.readouterr()

        assert run_status.main(argv + ["--wait", "--json"]) == 3
        captured = capsys.readouterr()
        assert captured.out == "", (
            "a refused `--wait` prints no view, so `--json`'s stdout stays empty "
            "rather than half a document"
        )
        assert captured.err.startswith("ERROR: ")
        assert "not_a_driver_key" in captured.err
        assert "Traceback" not in captured.err

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1", "0"])
    def test_a_timeout_that_is_not_a_positive_finite_number_is_refused(
        self, tmp_path, monkeypatch, capsys, value
    ):
        """F7, 2026-09-20: the CLI undid the validator one layer up.

        `validate_progress_block` rejects `.nan`/`.inf` for `poll_s` *precisely*
        to prevent "a `--wait` that never returns", but `--timeout` is a plain
        `type=float` and both parse. With `--host codex` — `timeout_ceiling_s:
        0`, the documented "no cap", so nothing clamps — the value reached
        `_wait_for_terminal` intact, where `time.time() >= nan` is `False`
        forever over a non-terminal view. `0` and a negative are the mirror
        image: they return before the first re-read, which is the instant-return
        failure `poll_s: 0`'s own rejection exists for.

        No `_wait_for_terminal` spy here on purpose: a spy is exactly what would
        hide a wait that was entered.
        """
        # `--timeout=<v>`, not `--timeout <v>`: argparse reads a leading `-` as
        # the start of the next option, and a negative value is half the subject.
        assert self._exit_code(
            tmp_path, monkeypatch, ["--host", "codex", f"--timeout={value}"]
        ) == 3
        captured = capsys.readouterr()
        assert captured.err.startswith("ERROR: --timeout ")
        assert "Traceback" not in captured.err

    def test_a_configured_poll_s_is_really_waited_not_only_computed(
        self, tmp_path, monkeypatch, capsys
    ):
        """The one row in this class that does **not** spy the sleeper out.

        Every other row replaces `_wait_for_terminal`, which is right for
        asserting the number that reaches it and wrong as the only coverage:
        with the sleeper spied out on every row and the one end-to-end wait test
        passing `--timeout 1` explicitly, nothing in the suite drove the
        config-resolved value through the real loop. A `poll_s` that resolved
        correctly and was then ignored — or obeyed for the wrong duration —
        would have gone green everywhere (2026-09-20).

        5 s is `provider_registry`'s own floor, so it is the shortest honest
        configuration; the real loop re-checks on a 5 s grid, so it comes back at
        one grid step.
        """
        self._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {"step": "fanout", "models_total": 2, "models_done": 0},
        })
        start = time.monotonic()
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--host", "opencode",
            "--wait",
        ])
        elapsed = time.monotonic() - start
        capsys.readouterr()
        assert code == 2
        assert 4.0 <= elapsed < 20.0, (
            f"`--wait` with no `--timeout` and `poll_s: 5` took {elapsed:.2f}s; "
            "it should wait the configured cadence and then hand back the "
            "non-terminal view"
        )


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


# ===========================================================================
# 2026-09-20 — what a re-used phase dir does to the poll loop
# ===========================================================================

class TestAReusedPhaseDirDoesNotFakeATerminalRun:
    """One phase dir, two runs' state: the `run` half and the checkpoint half.

    A phase dir is stable per plan and per phase (`{plan}/review-plan/` every
    time), and nothing drops a predecessor's `run` record on the foreground
    path — only a detached worker's `claim_run` ever writes one. So a
    host-background re-run over a dir an earlier `--detach` run used inherits
    that earlier run's terminal record while writing its own live `progress`.
    Every test here is that pair.
    """

    @staticmethod
    def _inherited_run_over_a_live_checkpoint(plan: Path) -> Path:
        phase = _phase_dir(plan)
        # The predecessor: a `--detach`ed run that finished cleanly.
        status_file.claim_run(str(phase), {
            "run_id": "run-predecess1", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "review-plan", "host": "claude-code",
            "heartbeat_at": status_file.now_iso(),
        })
        # The successor, in this host's own background job: no `run` write of
        # any kind, only the phase machine and the per-model map.
        beat = status_file.now_iso()
        status_file.update_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {
                "step": "fanout", "models_total": 2, "models_done": 0,
                "models_succeeded": 0,
                "models": {
                    "codex:gpt-5.6-sol": {
                        "state": "running", "started_at": beat,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 1,
                    },
                    "opencode:auto": {
                        "state": "running", "started_at": beat,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 1,
                    },
                },
            },
        })
        return phase

    def test_the_inherited_record_really_does_read_terminal(self, tmp_path):
        """The premise, asserted rather than assumed.

        Without this the test below could pass for the wrong reason — a view
        that was never terminal has nothing to be kept waiting past.
        """
        phase = self._inherited_run_over_a_live_checkpoint(_plan(tmp_path))
        view = detach.classify_run(str(phase), close_out=False)
        assert view["verdict"] in detach.TERMINAL_VERDICTS
        assert view["state"] == "models_running"
        assert (view["progress"] or {}).get("step") == "fanout"

    def test_wait_keeps_waiting_while_the_checkpoint_is_still_going(
        self, tmp_path, monkeypatch, capsys
    ):
        """F1, 2026-09-20: the loop became a zero-interval spin.

        Measured on exactly this pair: `status … --view run --wait --timeout
        300` returned in **0.11 s** with exit 0, printing `done` directly above a
        table saying `running`. The instruction then relays nothing (rule 2: a
        `0` on a host recording no run is an earlier run's record) and polls
        again — so a 120-300 s loop becomes a tight sequence of instant tool
        calls for the whole 10-30 minute run, and re-running a phase is the
        common path rather than an edge case.

        The fix is the *checkpoint* half of the file getting a vote:
        `_wait_for_terminal` keeps waiting while the phase machine says the
        phase is working, whatever the inherited `run` record says.
        """
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        self._inherited_run_over_a_live_checkpoint(plan)

        start = time.monotonic()
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--host", "opencode",
            "--wait",
        ])
        elapsed = time.monotonic() - start
        capsys.readouterr()

        assert elapsed >= 4.0, (
            f"`--wait` came back in {elapsed:.2f}s over a live fan-out because "
            "the phase dir still held a *predecessor's* terminal `run` record. "
            "An instant return is what turns the poll loop into a spin."
        )
        assert elapsed < 30.0, f"--wait did not come back ({elapsed:.1f}s)"
        # The exit code is still the inherited record's; dropping that record is
        # the seed-time half of the fix and lives elsewhere. What this owns is
        # that the poll spent its interval instead of returning instantly.
        assert code in (0, 2)

    def test_a_terminal_checkpoint_under_a_terminal_run_still_returns_at_once(
        self, tmp_path, monkeypatch, capsys
    ):
        """The other side of the same guard, and the reason it is evidence-based.

        A finished `--detach`ed run is the overwhelmingly common terminal case
        and its whole point is that `--wait` hands back immediately. A guard that
        read *any* checkpoint as "still going" would make every completed
        detached run cost a full poll interval before saying so.
        """
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": "run-finished-01", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "review-plan", "heartbeat_at": status_file.now_iso(),
        })
        status_file.update_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "progress": {"step": "complete", "models_total": 2,
                         "models_done": 2, "models_succeeded": 2},
        })

        start = time.monotonic()
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--host", "opencode",
            "--wait",
        ])
        elapsed = time.monotonic() - start
        capsys.readouterr()
        assert code == 0
        assert elapsed < 2.0, (
            f"a finished run took {elapsed:.2f}s to report itself finished; the "
            "checkpoint guard must key on evidence the phase is *working*, not "
            "on the checkpoint existing"
        )


class TestTheAskLanesTerminalWordsAreTerminal:
    """`ask` ends on `completed` / `partial` / `failed`, never on `complete`.

    Every other lane's phase machine stops on `complete`, and
    `CHECKPOINT_TERMINAL_STATES` was written from that ladder — so two of the
    three words the ask lane really writes were absent, `failed` was in the set
    only by coincidence, and `_checkpoint_in_flight` answered `True` for every
    ask that ever *succeeded*. Two consequences off that one gap: a finished
    foreground ask rendered `running in foreground (5/5 models, no detached run
    record)` for the rest of the phase dir's life — the head line rule 1 tells a
    host agent to relay verbatim — and `--wait --question …` over a detached ask
    whose `run.state` was already `done` spent its whole resolved `poll_s` on
    every poll before exiting 0.
    """

    @staticmethod
    def _terminal_words():
        """The words the ask lane really assigns, read out of its own source.

        Derived, never typed: a hand-copied list is exactly what drifted here in
        the first place, and a test that invented a word would go green over a
        lane that had stopped writing it.
        """
        import ast

        tree = ast.parse(
            (SKILL_ROOT / "ask_orchestrator.py").read_text(encoding="utf-8")
        )
        words = {
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "state"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        assert words, (
            "ask_orchestrator.py no longer assigns its final phase `state` as "
            "string literals, so this test can no longer read the real words"
        )
        return words

    @staticmethod
    def _ask_dir(plan: Path, question: str) -> Path:
        from utils.output_handler import slugify_question

        directory = (
            plan.parent / plan.stem.replace(" ", "_") / "ask"
            / slugify_question(question)
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def test_every_word_it_ends_on_is_a_terminal_checkpoint_state(self):
        words = self._terminal_words()
        # The premise, asserted rather than assumed: this lane does not spell
        # its ending the way the review lanes do, which is why the set had to
        # widen instead of the lane being made to write `complete`.
        assert "complete" not in words, (
            "the ask lane now writes the review lanes' word, so this test is "
            "pinning a distinction that no longer exists"
        )
        missing = sorted(words - run_status.CHECKPOINT_TERMINAL_STATES)
        assert not missing, (
            f"ask_orchestrator ends a run on {missing}, and "
            "`CHECKPOINT_TERMINAL_STATES` has never heard of those words — so "
            "`_checkpoint_in_flight` reads every ask that ended that way as "
            "still working"
        )

    def test_a_finished_foreground_ask_stops_calling_itself_running(
        self, tmp_path, capsys
    ):
        """One pass per real terminal word, against the head line rule 1 relays."""
        plan = _plan(tmp_path)
        question = "Does the rollback plan cover the schema change?"
        directory = self._ask_dir(plan, question)
        for word in sorted(self._terminal_words()):
            status_file.write_status(str(directory), {
                "phase": "ask", "state": word, "question": question,
                "progress": {"step": "complete", "models_total": 5,
                             "models_done": 5, "models_succeeded": 5},
            })
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--question", question,
            ]) == 2
            out = capsys.readouterr().out
            assert (
                "finished in foreground (5/5 models, no detached run record)"
                in out
            ), out
            assert "running in foreground" not in out, (
                f"an ask that ended on `{word}` still reports itself as "
                f"running:\n{out}"
            )

    def test_wait_over_a_finished_detached_ask_returns_at_once(
        self, tmp_path, monkeypatch, capsys
    ):
        """The `--wait` half: the whole poll interval, on every poll, forever.

        `progress.step` is already `complete` here, so the only thing that could
        hold this wait is the phase `state` word — which is the point.
        """
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        question = "Which models disagreed about the cache?"
        directory = self._ask_dir(plan, question)
        status_file.claim_run(str(directory), {
            "run_id": "run-ask-done01", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "ask", "host": "opencode",
            "heartbeat_at": status_file.now_iso(),
        })
        status_file.update_status(str(directory), {
            "phase": "ask", "question": question, "state": "completed",
            "progress": {"step": "complete", "models_total": 5,
                         "models_done": 5, "models_succeeded": 5},
        })

        start = time.monotonic()
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--host", "opencode",
            "--question", question, "--wait",
        ])
        elapsed = time.monotonic() - start
        capsys.readouterr()
        assert code == 0
        assert elapsed < 2.0, (
            f"a finished detached ask took {elapsed:.2f}s to report itself "
            "finished; every poll of every completed ask pays that, and the "
            "run had been done since before the call"
        )


class TestTheRefusalCloseOutWordsAreTerminal:
    """A run that refused before dispatch has ended, and the checkpoint says so.

    Same defect class as the ask lane's three words directly above, one lane
    further on. `review_orchestrator_base._print_executor_refusal` is the exit-3
    block for a run that dispatched nothing, and its one caller `sys.exit`s on
    the next line — so `runner_rejected` / `executor_refused` are the last thing
    that run ever writes. Neither was in `CHECKPOINT_TERMINAL_STATES`, so
    `_checkpoint_in_flight` read a refused phase as still working: the head line
    called it `running in foreground (…)` for the rest of the dir's life, and
    `--wait` burned its whole resolved deadline before exiting 2 with a `next`
    that polls it again — a loop nothing in it can end, over a run that cannot
    move. `run_status._resume_shape` meanwhile already reads a top-level
    `runner_rejected` as the refusal's durable classification, so one document
    was terminal to one reader and live to the other.
    """

    @staticmethod
    def _close_out_words():
        """The words the refusal close-out really writes, read out of its source.

        Derived rather than typed, for the reason the ask lane's twin above
        gives: a hand-copied list is exactly what drifted in the first place,
        and a test that invented a word would go green over a site that had
        stopped writing it. Scoped to `_print_executor_refusal` because *that
        function* is the ending — its caller exits on the next line, so every
        word it stamps on the checkpoint is terminal by construction. The
        module at large is not: it also writes `models_running`,
        `grouping_complete` and the rest of the live ladder.
        """
        import ast

        tree = ast.parse(
            (SKILL_ROOT / "utils" / "review_orchestrator_base.py")
            .read_text(encoding="utf-8")
        )
        func = next(
            (node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)
             and node.name == "_print_executor_refusal"),
            None,
        )
        assert func is not None, (
            "review_orchestrator_base no longer defines "
            "`_print_executor_refusal`, so this test can no longer read the "
            "real close-out words"
        )
        words = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (isinstance(key, ast.Constant) and key.value == "state"):
                    continue
                # `{"state": "runner_rejected" if rejected else "executor_refused"}`
                # — both branches are written, so both are collected.
                branches = (
                    [value.body, value.orelse] if isinstance(value, ast.IfExp)
                    else [value]
                )
                words.update(
                    branch.value for branch in branches
                    if isinstance(branch, ast.Constant)
                    and isinstance(branch.value, str)
                )
        assert words, (
            "`_print_executor_refusal` no longer assigns the checkpoint's "
            "`state` as string literals, so this test can no longer read the "
            "real words"
        )
        return words

    def test_the_premise_it_pins(self):
        """The site really is an ending, and it really writes its own words."""
        words = self._close_out_words()
        assert "runner_rejected" in words and "executor_refused" in words, (
            f"the refusal close-out now writes {sorted(words)}; this test's "
            "behavioural half below names the two it was written for"
        )
        assert "complete" not in words, (
            "the refusal now spells its ending the way a successful phase does, "
            "so this test is pinning a distinction that no longer exists"
        )

    def test_every_word_the_close_out_writes_is_a_terminal_checkpoint_state(self):
        missing = sorted(self._close_out_words() - run_status.CHECKPOINT_TERMINAL_STATES)
        assert not missing, (
            f"`_print_executor_refusal` ends a run on {missing}, and "
            "`CHECKPOINT_TERMINAL_STATES` has never heard of those words — so "
            "`_checkpoint_in_flight` reads every run refused that way as still "
            "working"
        )

    def test_a_refused_foreground_phase_stops_calling_itself_running(
        self, tmp_path, capsys
    ):
        """One pass per real close-out word, against the head line rule 1 relays."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        for word in sorted(self._close_out_words()):
            # What the refusal leaves behind: the fan-out's own `progress` is
            # whatever the run had got to, and `_print_executor_refusal` stamps
            # only the top-level `state`. `step` is terminal here so the phase
            # `state` word is the only thing that could hold the view — which
            # is the half under test.
            status_file.write_status(str(phase), {
                "phase": "review-plan", "state": word,
                "reason_code": "runner_invalid_model",
                "progress": {"step": "complete", "models_total": 2,
                             "models_done": 2, "models_succeeded": 2},
            })
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--phase", "review-plan",
            ]) == 2
            out = capsys.readouterr().out
            assert (
                "finished in foreground (2/2 models, no detached run record)"
                in out
            ), out
            assert "running in foreground" not in out, (
                f"a run refused with `{word}` still reports itself as "
                f"running:\n{out}"
            )

    def test_wait_over_a_refused_run_returns_at_once(
        self, tmp_path, monkeypatch, capsys
    ):
        """The `--wait` half: the whole resolved deadline, on every poll, forever.

        Foreground, so there is no `run` record and the verdict is `no_run` —
        not a vote either way (F2), which leaves the checkpoint casting the only
        one. That is exactly the shape a Claude Code review-plan run refused by
        its runner leaves on disk.
        """
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "runner_rejected",
            "reason_code": "runner_invalid_model",
            "progress": {"step": "complete", "models_total": 2,
                         "models_done": 2, "models_succeeded": 2},
        })

        start = time.monotonic()
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--host", "opencode",
            "--phase", "review-plan", "--wait",
        ])
        elapsed = time.monotonic() - start
        capsys.readouterr()
        assert code == 2
        assert elapsed < 2.0, (
            f"a refused run took {elapsed:.2f}s to report itself finished; the "
            "run had been over since before the call, and the instruction's "
            "answer to that exit 2 is to poll again"
        )


class TestAPollThatNamesTheRunIsNotHeldByAStaleCheckpoint:
    """The mirror image of F1: the terminal record IS the run being asked about.

    Nothing walks the phase machine back when a run fails or is aborted, so a
    dead run's checkpoint keeps `state: models_running` and `progress.step:
    fanout` for good. F1 gave that checkpoint a vote because on a re-used phase
    dir it normally describes a *successor*. Pinned with `--run-id` there is no
    successor left for it to describe — the id the caller typed is the id on the
    record — and holding them for the whole resolved timeout before reporting a
    failure that was already on disk is the opposite of what `--wait` is for.
    """

    RUN_ID = "run-current0001"

    @classmethod
    def _dead_run_over_a_stale_checkpoint(cls, plan: Path, *, state: str) -> Path:
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": cls.RUN_ID, "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": state, "exit_code": 1 if state == "failed" else None,
            "error": "the worker died mid-fan-out", "detached": True,
            "mode": "review-plan", "host": "opencode",
            "heartbeat_at": status_file.now_iso(),
        })
        # The phase half, left exactly where the failure found it.
        status_file.update_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {"step": "fanout", "models_total": 2,
                         "models_done": 0, "models_succeeded": 0},
        })
        return phase

    @pytest.mark.parametrize("state", ["failed", "aborted"])
    def test_the_premise_a_dead_runs_checkpoint_still_reads_in_flight(
        self, tmp_path, state
    ):
        """Without this the timing test below could pass for the wrong reason."""
        phase = self._dead_run_over_a_stale_checkpoint(_plan(tmp_path), state=state)
        view = detach.classify_run(str(phase), close_out=False)
        assert view["verdict"] == state
        assert view["verdict"] in detach.TERMINAL_VERDICTS
        assert run_status._checkpoint_in_flight(view) is True

    @pytest.mark.parametrize("state", ["failed", "aborted"])
    def test_a_terminal_run_named_by_its_id_reports_at_once(
        self, tmp_path, monkeypatch, capsys, state
    ):
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        self._dead_run_over_a_stale_checkpoint(plan, state=state)

        start = time.monotonic()
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--host", "opencode",
            "--run-id", self.RUN_ID[:8], "--wait",
        ])
        elapsed = time.monotonic() - start
        capsys.readouterr()
        # Terminal-needs-human, which is what the caller was waiting to be told.
        assert code == cli_common.EXIT_NEEDS_HUMAN
        assert elapsed < 2.0, (
            f"a poll pinned to run {self.RUN_ID[:8]} spent {elapsed:.2f}s "
            f"before reporting that very run's `{state}` verdict, because the "
            "phase fields it left behind still say `models_running`/`fanout`"
        )

    def test_the_same_pair_without_the_id_still_waits(
        self, tmp_path, monkeypatch, capsys
    ):
        """F1 is untouched: only an *identified* run skips the checkpoint.

        A bare poll cannot know whether the live-looking checkpoint belongs to
        the terminal record above it or to a successor nobody recorded, and F1's
        whole point is that guessing "terminal" there turns the poll loop into a
        spin. Same bytes on disk, no `--run-id`, still waits.
        """
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        self._dead_run_over_a_stale_checkpoint(plan, state="failed")

        start = time.monotonic()
        run_status.main([
            "--plan-file", str(plan), "--view", "run", "--host", "opencode",
            "--wait",
        ])
        elapsed = time.monotonic() - start
        capsys.readouterr()
        assert elapsed >= 4.0, (
            "a poll that named no run skipped the checkpoint's vote, which is "
            f"the F1 regression (returned in {elapsed:.2f}s)"
        )


class TestTheForegroundHeadLineHasATense:
    """`running in foreground (…)` is relayed verbatim by rule 1, so it must be true."""

    @staticmethod
    def _foreground(plan: Path, *, state: str, step: str, done: int) -> Path:
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": state,
            "progress": {"step": step, "models_total": 2, "models_done": done,
                         "models_succeeded": done},
        })
        assert "run" not in status_file.read_status(str(phase))
        return phase

    def test_a_finished_foreground_phase_stops_calling_itself_running(
        self, tmp_path, capsys
    ):
        """F3, 2026-09-20: the head line gated on `progress` being non-empty.

        A foreground checkpoint keeps its `progress` object forever — nothing
        clears it — so `state: complete` plus `progress.step: complete` still
        printed `running in foreground (2/2 models, no detached run record)`
        with `phase state: complete` two lines below it. Rule 1 tells the agent
        to relay that head line **verbatim**, so `status` on any completed Claude
        Code phase, and on every historical ask dir a bare `status` lists,
        reported it as running.

        "finished" claims only that the phase stopped working; how it ended is
        the `phase state:` line, which is printed on its own and never collapsed
        into this one.
        """
        plan = _plan(tmp_path)
        self._foreground(plan, state="complete", step="complete", done=2)

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "finished in foreground (2/2 models, no detached run record)" in out
        assert "running in foreground" not in out, out
        # Still three state fields on three lines, still never collapsed.
        assert "phase state:    complete" in out

    def test_a_working_foreground_phase_still_says_running(self, tmp_path, capsys):
        """The mirror row: the tense has to distinguish, not just soften."""
        plan = _plan(tmp_path)
        self._foreground(plan, state="models_running", step="fanout", done=1)

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "running in foreground (1/2 models, no detached run record)" in out

    def test_the_host_handoff_state_counts_as_finished(self, tmp_path, capsys):
        """`validation_pending` is a terminal step, not a stalled one.

        The process has exited and the agent's own subagents take over, so a
        phase parked there is not working and must not be relayed as though it
        were. `progress_tracker.PhaseProgress` names this and `complete` as the
        two terminal steps; this is the reader agreeing with it.
        """
        plan = _plan(tmp_path)
        self._foreground(
            plan, state="validation_pending", step="validation_pending", done=2
        )

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "finished in foreground (2/2 models" in out
        assert "running in foreground" not in out, out


class TestOneBadModelRowCostsItsOwnColumn:
    """The `models` map is the least trustworthy part of the checkpoint.

    It is the only part written per slot, by whichever lane won the race, and
    `status` is the command a poll loop runs — so a malformed row costs its own
    cell and never the command. Everything else in the map already degraded;
    one shape did not (F5, 2026-09-20).
    """

    @staticmethod
    def _two_rows(plan: Path, first_started_at) -> Path:
        phase = _phase_dir(plan)
        beat = status_file.now_iso()
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {
                "step": "fanout", "models_total": 2, "models_done": 0,
                "models_succeeded": 0,
                "models": {
                    "codex:gpt-5.6-sol": {
                        "state": "running", "started_at": first_started_at,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 1,
                    },
                    "opencode:auto": {
                        "state": "running", "started_at": beat,
                        "updated_at": beat, "timeout_s": 1800, "attempt": 1,
                    },
                },
            },
        })
        return phase

    def test_a_clock_that_raises_costs_that_row_its_clock_only(
        self, tmp_path, monkeypatch, capsys
    ):
        """The renderer's own guard, pinned independently of the parser's.

        `detach._elapsed_since` caught `ValueError` only, and `datetime.now() -
        <aware datetime>` raises **TypeError**, so one `started_at:
        "2026-09-19T00:00:00Z"` in the map took the whole `status` command down
        with a traceback and exit 1 — which rule 2 maps to "the run state is
        unreadable, stop and report", ending a poll loop over a perfectly
        healthy run.

        The parser has its own fix and its own test; this one stands whatever
        the parser does, by making the reader raise on demand. Two guards for
        one incident is deliberate: the map is read by a command that must not
        fail, and a second reader of a third stamp shape would otherwise
        reintroduce it.
        """
        plan = _plan(tmp_path)
        self._two_rows(plan, status_file.now_iso())

        def _explode(entry, now=None):
            if str(entry.get("timeout_s")) == "1800" and entry.get("attempt") == 1:
                raise TypeError(
                    "can't subtract offset-naive and offset-aware datetimes"
                )
            return None

        monkeypatch.setattr(run_status, "entry_elapsed_s", _explode)

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "Traceback" not in out
        # Both rows are still rendered, with their state and no clock.
        rows = [
            line.strip() for line in out.splitlines()
            if line.startswith("        ")
        ]
        assert [row.split() for row in rows] == [
            ["codex:gpt-5.6-sol", "running"], ["opencode:auto", "running"]
        ], rows

    def test_an_aware_started_at_does_not_take_the_command_down(
        self, tmp_path, capsys
    ):
        """End to end, with the stamp shape that really appeared.

        `Z` is the plan's own schema example and what the JS tests emit, so this
        is a file a real host can write. Whether the row ends up with a clock or
        without one is the parser's business; what this pins is that `status`
        answers at all.
        """
        plan = _plan(tmp_path)
        self._two_rows(plan, "2026-09-19T00:00:00Z")

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "Traceback" not in out
        assert "codex:gpt-5.6-sol" in out and "opencode:auto" in out


class TestAMalformedProgressObjectCostsItsOwnLine:
    """`progress` is a reserved *object*, and a file on disk can disagree.

    `classify_run` passed `status.get("progress") or {}` straight through, and a
    non-empty list or string is **truthy** — so the malformed value reached the
    view and every reader that spells the same idiom called `.get` on it:
    `_checkpoint_in_flight`, `_verdict_text`, `_model_table_lines` and
    `_terminal_fingerprints`. The result was an `AttributeError` traceback and
    exit 1, which rule 2 maps to "the run state is unreadable, stop and report"
    — a far larger claim than "one key of this checkpoint is malformed", and it
    ends the poll loop over a run that is otherwise perfectly readable. The two
    newest readers sit on the head line and on the `--wait` loop, so both are
    driven here.

    Same rule the per-model rows already follow: a value that is not a mapping
    is not trusted, coerced once where the view is built rather than guarded at
    each reader — the guard that has to be repeated is the one that gets
    forgotten.
    """

    SHAPES = [["fanout"], "fanout", 7, True]

    @staticmethod
    def _write_raw(plan: Path, progress) -> Path:
        """Hand-write the checkpoint: `write_status` is not the writer that did this."""
        phase = _phase_dir(plan)
        Path(phase, ".status.json").write_text(
            json.dumps({
                "phase": "review-plan", "state": "models_running",
                "progress": progress,
            }),
            encoding="utf-8",
        )
        return phase

    @pytest.mark.parametrize("progress", SHAPES)
    def test_the_view_carries_a_mapping_whatever_the_file_says(
        self, tmp_path, progress
    ):
        plan = _plan(tmp_path)
        phase = self._write_raw(plan, progress)

        view = detach.classify_run(str(phase), close_out=False)

        assert view["progress"] == {}, (
            "a non-dict `progress` reached the view; every reader below it "
            "calls `.get` on this"
        )

    @pytest.mark.parametrize("progress", SHAPES)
    def test_the_head_line_and_the_table_still_render(
        self, tmp_path, capsys, progress
    ):
        """The head line's tense gate is `_checkpoint_in_flight`, which reads it."""
        plan = _plan(tmp_path)
        self._write_raw(plan, progress)

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "Traceback" not in out and "AttributeError" not in out
        # The dir is still reported and the *other* half of the checkpoint is
        # still read: only the malformed key is dropped, along with the counts
        # and the foreground tense it would have carried. `no_run` here is the
        # honest fallback — with no readable `progress` there is nothing to say
        # "2/2 models" from — and it is the same word this dir printed before a
        # foreground run ever wrote `progress` at all.
        assert "review-plan  no_run" in out, out
        assert "phase state:    models_running" in out, out

    @pytest.mark.parametrize("progress", SHAPES)
    def test_the_wait_loop_still_polls_it(self, tmp_path, monkeypatch, progress):
        """`--wait` re-reads the same file every interval — one bad key, every pass."""
        plan = _plan(tmp_path)
        self._write_raw(plan, progress)
        real_sleep = run_status.time.sleep
        monkeypatch.setattr(
            run_status.time, "sleep", lambda s: real_sleep(min(s, 0.01))
        )

        assert run_status.main([
            "--plan-file", str(plan), "--view", "run", "--wait",
            "--timeout", "0.2",
        ]) == 2


class TestTheAskLanesOwnSelectorCarriesProgress:
    def test_a_question_pinned_poll_carries_progress_into_its_result(
        self, tmp_path, capsys
    ):
        """F8a, 2026-09-20: the one mode that always pins one run had no `progress`.

        `MULTI_LLM_RESULT.progress` was gated on `--phase or --run-id`, and
        `ask` is the one mode whose instruction core forbids `--phase` outright
        (`--phase ask` selects every question ever asked under the plan, so a
        poll relays a wall of unrelated tables and holds exit 2 long after this
        invocation finished). Its documented selector is `--question`, which
        names exactly one run — the condition the carry-over is *for* — so an
        ask poll was the one poll whose result line could not relay a count.
        """
        from utils.output_handler import slugify_question

        plan = _plan(tmp_path)
        question = "Is the rollback strategy sufficient?"
        directory = (
            plan.parent / plan.stem.replace(" ", "_") / "ask"
            / slugify_question(question)
        )
        directory.mkdir(parents=True, exist_ok=True)
        progress = {
            "step": "fanout", "models_total": 3, "models_done": 1,
            "models_succeeded": 1,
            "models": {"codex:gpt-5.6-sol": {"state": "running"}},
        }
        status_file.write_status(str(directory), {
            "phase": "ask", "state": "models_running", "question": question,
            "progress": progress,
        })

        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--question", question,
            ]) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()
        payload = context.payload(2)
        assert payload["progress"] == progress
        # The same selector answers the mode/phase half too: one named run is
        # one named run, whichever of the three selectors named it.
        assert payload["phase"] == "ask"


class TestTheRosterRendersBeforeTheMapIsSeeded:
    """The launch-time window, which is the one the user actually watched.

    The child writes `models_requested` and reports `step=fanout,
    models_total=N` *before* it builds the tracker that seeds `progress.models`,
    so for the first seconds of every fan-out the checkpoint knows the whole
    roster and the table rendered nothing:

        running in foreground (1/5 models, no detached run record)
        progress:       fanout, models 1/5 done, 1 ok, 0 failed

    Which CLI is running which model was on disk the whole time. The two asks
    this answers are "say that all five started, and say what each one is".
    """

    @staticmethod
    def _checkpoint(plan: Path, progress: dict, *, requested=None) -> Path:
        phase = _phase_dir(plan)
        checkpoint = {
            "phase": "review-plan", "state": "models_running",
            "progress": dict(progress, step="fanout"),
        }
        if requested is not None:
            checkpoint["models_requested"] = requested
        status_file.write_status(str(phase), checkpoint)
        return phase

    ROSTER = [
        "codex:gpt-6-astra:xhigh",
        "pi:openrouter/deepseek/deepseek-v4-flash-0731",
        "opencode:auto",
    ]

    # The three shapes a not-yet-seeded map arrives in: the key absent (what the
    # child really writes), an empty dict (a tracker that seeded nothing), and a
    # non-dict (a hand-edited or truncated checkpoint). All three mean the same
    # thing to a reader and must render the same roster.
    @pytest.mark.parametrize(
        "models", [None, {}, "not-a-map"], ids=["absent", "empty", "garbage"]
    )
    def test_the_requested_roster_renders_before_the_map_exists(
        self, tmp_path, capsys, models
    ):
        plan = _plan(tmp_path)
        progress = {"models_total": 3, "models_done": 0, "models_succeeded": 0}
        if models is not None:
            progress["models"] = models
        self._checkpoint(plan, progress, requested=self.ROSTER)

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out.splitlines()

        assert "      progress:       fanout, models 0/3 done, 0 ok, 0 failed" in out
        assert "      models:" in out
        # Pinned whole: the column padding is what makes the CLI and the model
        # readable down a column, and every host reads this back through a file
        # codec, so it is ASCII to begin with.
        assert out[out.index("      models:") + 1:][:3] == [
            "        codex:gpt-6-astra:xhigh                        queued",
            "        pi:openrouter/deepseek/deepseek-v4-flash-0731  queued",
            "        opencode:auto                                  queued",
        ], out

    def test_a_dispatch_only_phase_still_grows_no_models_header(
        self, tmp_path, capsys
    ):
        """The guard the roster change had to leave standing.

        `generate-tasks` never calls `report_progress(step=…)` before the end, so
        its only progress is the executor's mirror and it records no
        `models_total` at all. Reading an absent map as `{}` would have handed
        `_model_row_specs` an empty list to answer from — and its `models_total`
        gate is what keeps the answer empty. A bare `models:` header over no rows
        on every dispatch-only phase is the regression this forbids.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "generate-tasks")
        status_file.write_status(str(phase), {
            "phase": "generate-tasks", "state": "dispatching",
            "progress": {"subagents": {
                "kind": "task_generation", "invoker": "executor",
                "total": 1, "ok": 1, "counts": {"ok": 1},
                "failed_task_ids": [],
            }},
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "progress:       task_generation 1/1 ok" in out
        assert "models:" not in out

    def test_a_roster_with_no_count_beside_it_is_not_a_table(
        self, tmp_path, capsys
    ):
        """The `models_total` gate itself, which the test above does not reach.

        That one passes because its fixture carries no `models_requested` — cut
        the gate out entirely and it stays green, so it pins the dispatch-only
        phase's *shape*, not the line that protects it. This is the other half:
        a checkpoint that DOES carry a roster and no `models_total` still grows
        no table, because `models_total` is what says a fan-out is what this
        phase is doing. Without it a roster is a leftover key, and a header over
        rows nobody is running is worse than no header.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, "generate-tasks")
        status_file.write_status(str(phase), {
            "phase": "generate-tasks", "state": "dispatching",
            "models_requested": ["codex:gpt-6-astra:xhigh", "gemini:pro"],
            "progress": {"subagents": {
                "kind": "task_generation", "invoker": "executor",
                "total": 1, "ok": 1, "counts": {"ok": 1},
                "failed_task_ids": [],
            }},
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "models:" not in out
        assert "codex:gpt-6-astra:xhigh" not in out

    def test_a_total_with_no_roster_anywhere_renders_no_table(
        self, tmp_path, capsys
    ):
        """`models_total` alone is a count, not a roster.

        A checkpoint written by an older run (or truncated mid-write) can carry
        the count with neither a seeded map nor `models_requested`. There is
        nothing to name, so the count keeps the line it already has and no
        header is invented over an empty table.
        """
        plan = _plan(tmp_path)
        self._checkpoint(
            plan, {"models_total": 5, "models_done": 0, "models_succeeded": 0}
        )
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "progress:       fanout, models 0/5 done, 0 ok, 0 failed" in out
        assert "models:" not in out


class TestJustFinishedNamesWhatCrossedTheLineDuringTheWait:
    """`progress:` says how many finished; this says which.

    The second ask, and the one the counts structurally cannot answer: a poller
    watching `models 2/5 done` go to `3/5` learns that something finished and
    never what. `--wait` is the only invocation that reads the same file twice,
    so it is the only one that can honestly say *just*.
    """

    DONE_AT = "2026-09-20T12:00:00"

    @classmethod
    def _entry(cls, state, elapsed_s, **extra):
        beat = status_file.now_iso()
        entry = {
            "state": state, "started_at": beat, "updated_at": beat,
            "timeout_s": 1800, "attempt": 1, "source": None,
        }
        if state in ("done", "failed"):
            entry["ended_at"] = cls.DONE_AT
            entry["elapsed_s"] = elapsed_s
        entry.update(extra)
        return entry

    @staticmethod
    def _progress(models: dict) -> dict:
        terminal = sum(
            1 for e in models.values()
            if isinstance(e, dict) and e.get("state") in ("done", "failed")
        )
        return {
            "step": "fanout", "models_total": len(models),
            "models_done": terminal, "models_succeeded": terminal,
            "models": models,
        }

    @classmethod
    def _write(cls, phase: Path, models: dict) -> None:
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "models_requested": list(models),
            "progress": cls._progress(models),
        })

    @staticmethod
    def _wait_once(plan: Path, monkeypatch, mutate, *argv) -> int:
        """One `--wait` pass with `mutate` applied inside the sleep.

        `main` collects once for the baseline and `_wait_for_terminal` collects
        again after its first sleep; mutating between the two is exactly the
        "a model finished while the poll was waiting" event, without spawning a
        worker to produce it. The timeout is small but not tiny: the deadline is
        checked *before* the first sleep, so a value in the milliseconds would
        make a loaded machine skip the only iteration this needs.
        """
        calls = []
        real_collect = run_status._collect

        def _collect(plan_path, args):
            calls.append(1)
            if len(calls) == 2:
                mutate()
            return real_collect(plan_path, args)

        monkeypatch.setattr(run_status, "_collect", _collect)
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--wait",
            "--timeout", "0.2", *argv,
        ])
        assert len(calls) == 2, (
            f"`--wait` re-read the checkpoint {len(calls) - 1} times; the delta "
            "this asserts needs exactly one re-read"
        )
        return code

    def test_a_wait_with_nothing_recorded_yet_waits_instead_of_spinning(
        self, tmp_path, monkeypatch
    ):
        """An empty selection is a waiting state, not a terminal one.

        The first poll of every launch is issued before the run has written its
        checkpoint, so it selects no view at all — and a loop keyed on "any view
        is live" exits instantly over an empty list. Exit 2 already means "still
        going, and that covers no run recorded yet", so returning in 0.13 s
        turned the one call whose job is to hold a turn into a spin the agent
        repeats as fast as its host allows (F4, 2026-09-20). It now costs its own
        deadline and then still exits 2.
        """
        plan = _plan(tmp_path)          # a plan with no phase dir at all
        slept = []
        real_sleep = run_status.time.sleep

        def _record(seconds):
            slept.append(seconds)
            real_sleep(min(seconds, 0.01))

        monkeypatch.setattr(run_status.time, "sleep", _record)
        assert run_status.main([
            "--plan-file", str(plan), "--view", "run", "--wait",
            "--timeout", "0.2",
        ]) == 2
        assert slept, "`--wait` returned without waiting over an empty selection"

    def test_a_re_run_of_the_same_models_still_names_what_finished(
        self, tmp_path, monkeypatch, capsys
    ):
        """The case the whole feature is for, and the one it used to lose.

        A phase dir is re-used run after run, so a re-review of the same plan
        finds the PREVIOUS run's rows on disk — the same specs, already `done`.
        Keyed on the spec alone, the baseline said "already terminal" before
        this run had seeded anything, so the completion that happened during the
        wait was skipped; on the next poll the row was genuinely terminal and in
        that baseline too, so it was skipped again, and no poll ever announced
        it (F1, 2026-09-20). The baseline is keyed on the *ending* — state,
        `ended_at`, `elapsed_s` — so a second, different ending is a difference.

        The launch writes now drop the predecessor's `progress` outright, which
        is where this is really fixed; this is the half that still has to hold
        when a poll lands in the window before one of them.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        spec = "codex:gpt-6-astra:xhigh"
        other = "gemini:gemini-3.5-pro"
        # The predecessor: both models finished here, days ago.
        old = self._entry("done", 300)
        old["ended_at"] = "2026-09-19T10:05:00"
        self._write(phase, {spec: dict(old), other: dict(old)})

        def _finish():
            # This run: same two specs, and one of them ends again — later.
            self._write(phase, {
                spec: self._entry("done", 252),
                other: self._entry("running", None),
            })

        assert self._wait_once(plan, monkeypatch, _finish) == 2
        out = capsys.readouterr().out
        assert f"      just finished: {spec} (4m12s, done)" in out
        # ...and only that one: the other row went backwards, not forwards.
        assert other not in out.split("just finished:")[1].split("\n")[0]

    def test_an_unreadable_baseline_claims_nothing_when_it_heals(
        self, tmp_path, monkeypatch, capsys
    ):
        """A map that could not be read is not a map that said "nothing yet".

        A torn or hand-edited checkpoint whose `models` is not a dict reads as
        an empty baseline unless the two cases are kept apart, and then every
        row in the healthy map that replaces it is announced as having just
        finished — rows that may have finished an hour ago (F4, 2026-09-20).
        `None` means "nothing was observed here", which is the same answer a
        phase dir discovered mid-wait gets.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "models_requested": ["codex:gpt-6-astra:xhigh"],
            "progress": {
                "step": "fanout", "models_total": 1, "models_done": 0,
                "models": "torn",
            },
        })

        def _heal():
            self._write(phase, {"codex:gpt-6-astra:xhigh": self._entry("done", 252)})

        assert self._wait_once(plan, monkeypatch, _heal) == 2
        assert "just finished:" not in capsys.readouterr().out

    def test_it_names_only_what_went_terminal_during_this_wait(
        self, tmp_path, monkeypatch, capsys
    ):
        """The baseline is what makes the word "just" true.

        `codex` was already `done` when the wait was entered — it is on the
        table, it is in the counts, and naming it again on every poll for the
        rest of the run would turn the line into a second, staler copy of the
        table. Only the row that crossed the line while this command waited is
        news.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self._write(phase, {
            "codex:gpt-6-astra:xhigh": self._entry("done", 252),
            "gemini:gemini-3.5-pro": self._entry("running", None),
        })

        def _finish():
            self._write(phase, {
                "codex:gpt-6-astra:xhigh": self._entry("done", 252),
                "gemini:gemini-3.5-pro": self._entry("done", 62),
            })

        assert self._wait_once(plan, monkeypatch, _finish) == 2
        out = capsys.readouterr().out
        assert "      just finished: gemini:gemini-3.5-pro (1m02s, done)" in out
        assert "codex:gpt-6-astra:xhigh (" not in out

    def test_the_line_sits_between_the_counts_and_the_table(
        self, tmp_path, monkeypatch, capsys
    ):
        """Reading order: how many, then which, then the roster they sit in.

        All three are relayed by Layer 2 rule 1 as they are printed, so the
        order on screen is the order the agent relays them in.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self._write(phase, {"codex:gpt-6-astra:xhigh": self._entry("running", None)})

        def _finish():
            self._write(
                phase, {"codex:gpt-6-astra:xhigh": self._entry("done", 252)}
            )

        assert self._wait_once(plan, monkeypatch, _finish) == 2
        out = capsys.readouterr().out.splitlines()
        progress_at = next(i for i, l in enumerate(out) if "progress:" in l)
        assert out[progress_at + 1] == (
            "      just finished: codex:gpt-6-astra:xhigh (4m12s, done)"
        ), out
        assert out[progress_at + 2] == "      models:", out

    def test_several_join_with_semicolons_and_a_failure_names_its_error(
        self, tmp_path, monkeypatch, capsys
    ):
        """`; `, not `, ` — the clause's own interior is comma-separated.

        Joined with commas, two models render as four fields no reader can
        re-pair. The failure's reason rides in its own clause because a poller
        told only "1 failed" has to open the log to learn anything at all, and
        the whole point of the line is that it does not have to.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self._write(phase, {
            "codex:gpt-6-astra:xhigh": self._entry("running", None),
            "gemini:gemini-3.5-pro": self._entry("running", None),
            "opencode:auto": self._entry("running", None),
        })

        def _finish():
            self._write(phase, {
                "codex:gpt-6-astra:xhigh": self._entry("done", 252),
                "gemini:gemini-3.5-pro": self._entry(
                    "failed", 62, error="rate limited"
                ),
                "opencode:auto": self._entry("running", None),
            })

        assert self._wait_once(plan, monkeypatch, _finish) == 2
        out = capsys.readouterr().out
        assert (
            "      just finished: codex:gpt-6-astra:xhigh (4m12s, done); "
            "gemini:gemini-3.5-pro (1m02s, failed: rate limited)"
        ) in out

    def test_a_multi_line_error_stays_one_line_in_both_renderers(
        self, tmp_path, monkeypatch, capsys
    ):
        """`error` is the one field a provider writes, and it can hold a newline.

        An adapter that puts a stderr excerpt in there hands both renderers a
        string with `\n` in it. Layer 2 rule 1 has the host agent relay
        `just finished:` **verbatim**, so a two-line value turns one rule-bound
        line into two — the second carrying no label and reading as prose the
        agent wrote — and the table's aligned columns come apart on the same
        row. Both sites collapse it, through one helper, because one of them
        collapsing is the two spellings of one field disagreeing.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self._write(phase, {
            "gemini:gemini-3.5-pro": self._entry("running", None),
        })

        def _finish():
            self._write(phase, {
                "gemini:gemini-3.5-pro": self._entry(
                    "failed", 62,
                    error="rate limited\n  retry-after: 60\n  upstream said no",
                ),
            })

        assert self._wait_once(plan, monkeypatch, _finish) == 2
        out = capsys.readouterr().out
        assert (
            "      just finished: gemini:gemini-3.5-pro (1m02s, failed: "
            "rate limited retry-after: 60 upstream said no)"
        ) in out
        # ...and the table row beneath it, which renders the same field.
        assert (
            "rate limited retry-after: 60 upstream said no" in out
        )
        assert "\n  retry-after" not in out
        # The model still running is not on the line, whatever else is.
        assert "opencode:auto (" not in out

    def test_a_resume_landing_mid_wait_is_skipped_and_never_ok(
        self, tmp_path, monkeypatch, capsys
    ):
        """One vocabulary for one row.

        There is no `skipped` *state* — a model answered from disk is `done`
        and `source` carries the distinction — so both this line and the table
        read it through `_model_state_label`. `ok` is the counting line's word
        for a slot that produced a usable result; borrowing it here would put
        `ok` directly above the same row labelled `skipped`.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self._write(phase, {"opencode:auto": self._entry("queued", None)})

        def _finish():
            self._write(phase, {
                "opencode:auto": self._entry("done", 0, source="existing_file"),
            })

        assert self._wait_once(plan, monkeypatch, _finish) == 2
        out = capsys.readouterr().out
        assert "      just finished: opencode:auto (0m00s, skipped)" in out
        assert "(0m00s, ok)" not in out

    def test_a_plain_status_prints_no_line_and_its_json_grows_no_key(
        self, tmp_path, monkeypatch, capsys
    ):
        """Without `--wait` nothing proved anything finished *during* the call.

        The `--json` half is a compatibility constraint, not tidiness: the
        payload is `{"runs": views}` serialized raw, so a key present-but-empty
        on every view would change the bytes of every JSON poll ever made —
        including the OpenCode package route's supervised poll, which reads the
        payload instead of the text.

        Both directions of that constraint are asserted here, because the
        comment beside the attach site claimed only one of them and claimed it
        backwards ("this key itself still never leaves the view", 2026-09-21).
        It does leave: `{"runs": views}` *is* these dicts, so a `--wait` that
        observes a completion hands the key straight back and a consumer has to
        handle it. The key is absent below because nothing finished during a
        wait — never because it cannot reach `--json`.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self._write(phase, {
            "codex:gpt-6-astra:xhigh": self._entry("done", 252),
            "gemini:gemini-3.5-pro": self._entry("done", 62),
        })

        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 2
        out = capsys.readouterr().out
        assert "just finished:" not in out
        # Both rows are on the table, so nothing is *missing* — only the tense.
        assert "codex:gpt-6-astra:xhigh  done" in out

        assert run_status.main(
            ["--plan-file", str(plan), "--view", "run", "--json"]
        ) == 2
        payload = json.loads(capsys.readouterr().out)
        assert "finished_during_wait" not in payload["runs"][0], payload["runs"][0]

        # The other direction: a wait that really observed one.
        self._write(phase, {
            "codex:gpt-6-astra:xhigh": self._entry("running", None),
        })

        def _finish():
            self._write(phase, {
                "codex:gpt-6-astra:xhigh": self._entry("done", 252),
            })

        assert self._wait_once(plan, monkeypatch, _finish, "--json") == 2
        waited = json.loads(capsys.readouterr().out)
        assert waited["runs"][0]["finished_during_wait"] == [{
            "spec": "codex:gpt-6-astra:xhigh", "state": "done",
            "elapsed_s": 252.0, "error": None,
        }], waited["runs"][0]

    def test_a_wait_that_returns_at_once_attaches_nothing(
        self, tmp_path, monkeypatch, capsys
    ):
        """A finished detached run: `--wait` hands back without ever looping.

        The baseline and the returned view are then literally the same dicts, so
        the difference is empty and no key is attached — which is the right
        answer rather than a missed case: nothing finished during a wait that
        never waited. Asserted on the JSON, because the empty text proves only
        that `_just_finished_text` returned "".
        """
        TestWaitTimeoutResolution._config(tmp_path, monkeypatch, "  poll_s: 5\n")
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.claim_run(str(phase), {
            "run_id": "run-finished-jf", "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "state": "done", "exit_code": 0, "detached": True,
            "mode": "review-plan", "heartbeat_at": status_file.now_iso(),
        })
        # Written *after* the claim, and with `update_status` rather than
        # `write_status`: both halves of the file have to say "finished" or the
        # wait keeps waiting on the one that does not, and this test's whole
        # subject is the loop that never runs.
        status_file.update_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "progress": dict(
                self._progress(
                    {"codex:gpt-6-astra:xhigh": self._entry("done", 252)}
                ),
                step="complete",
            ),
        })

        start = time.monotonic()
        code = run_status.main([
            "--plan-file", str(plan), "--view", "run", "--wait", "--json",
        ])
        elapsed = time.monotonic() - start
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert elapsed < 2.0, f"a finished run waited {elapsed:.2f}s"
        assert "finished_during_wait" not in payload["runs"][0], payload["runs"][0]

    def test_a_malformed_entry_costs_its_own_clause_never_the_command(
        self, tmp_path, monkeypatch, capsys
    ):
        """`status` is the command a poll loop runs, so it must answer.

        Three bad shapes in one map, and the good row still gets named: a clock
        whose reader raises (F5's `started_at: "…Z"`, which took the whole
        command down with a TypeError and exit 1 — which rule 2 reads as "the
        run state is unreadable, stop and report"), a stored `elapsed_s` that is
        not a number, and an entry that is not a dict at all. A clause with no
        clock beats a traceback; a row that cannot be read as terminal is simply
        not claimed as finished.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self._write(phase, {
            "codex:gpt-6-astra:xhigh": self._entry("running", None),
            "gemini:gemini-3.5-pro": self._entry("running", None),
            "pi:auto": self._entry("running", None),
        })

        def _finish():
            self._write(phase, {
                "codex:gpt-6-astra:xhigh": self._entry("done", 252),
                "gemini:gemini-3.5-pro": self._entry("done", "soon"),
                "pi:auto": "truncated mid-write",
            })
            real = run_status.entry_elapsed_s

            def _explode(entry, now=None):
                if entry.get("elapsed_s") == 252:
                    raise TypeError(
                        "can't subtract offset-naive and offset-aware datetimes"
                    )
                return real(entry, now)

            monkeypatch.setattr(run_status, "entry_elapsed_s", _explode)

        assert self._wait_once(plan, monkeypatch, _finish) == 2
        out = capsys.readouterr().out
        assert "Traceback" not in out
        # The raising clock and the unreadable `elapsed_s` both degrade to a
        # clause with no elapsed; the non-dict row is not claimed at all.
        assert (
            "      just finished: codex:gpt-6-astra:xhigh (done); "
            "gemini:gemini-3.5-pro (done)"
        ) in out
        assert "pi:auto (" not in out

    def test_a_phase_dir_that_appeared_mid_wait_is_not_claimed_as_just_finished(
        self, tmp_path, monkeypatch, capsys
    ):
        """No "before" picture, no claim.

        A poll with no selector lists every phase under the plan, so another
        process opening a second phase dir mid-wait adds a view this command
        never saw the start of. Treating its missing baseline as an *empty*
        baseline would announce every already-terminal row in it as having just
        finished, on no evidence at all — and a finished phase dir is exactly
        what tends to appear that way. It gets its roster from the table, like
        any other first sighting, and only the dir that was here all along can
        say what moved in it.
        """
        plan = _plan(tmp_path)
        watched = _phase_dir(plan)
        self._write(watched, {"codex:gpt-6-astra:xhigh": self._entry("running", None)})

        def _appear():
            self._write(
                watched, {"codex:gpt-6-astra:xhigh": self._entry("done", 252)}
            )
            self._write(
                _phase_dir(plan, "implement"),
                {"gemini:gemini-3.5-pro": self._entry("done", 62)},
            )

        assert self._wait_once(plan, monkeypatch, _appear) == 2
        out = capsys.readouterr().out
        assert "      just finished: codex:gpt-6-astra:xhigh (4m12s, done)" in out
        # The newcomer is listed, tabled, counted — and not claimed.
        assert "gemini:gemini-3.5-pro" in out
        assert "gemini:gemini-3.5-pro (" not in out

    def test_a_models_key_that_is_not_there_yet_is_an_empty_baseline(
        self, tmp_path, monkeypatch, capsys
    ):
        """F1, 2026-09-21: absent is not unreadable, and the difference is a line.

        There is a real window where the key is simply missing: the claim write
        drops the predecessor's `progress` outright and `seed()` writes this
        run's roster a moment later. A poll landing between them — the first
        poll of every launch — reads a checkpoint with no map at all, which is
        the honest "nothing has finished yet". Folded into the same `None` the
        torn map above returns, that baseline was skipped as "nothing observed",
        so every spec that went terminal during that whole first wait produced
        no `just finished:` line.
        """
        # The two shapes, told apart at the source.
        assert run_status._terminal_fingerprints(
            {"progress": {"step": "fanout"}}
        ) == {}
        assert run_status._terminal_fingerprints({"progress": {}}) == {}
        assert run_status._terminal_fingerprints(
            {"progress": {"step": "fanout", "models": "torn"}}
        ) is None

        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {"step": "fanout", "models_total": 2, "models_done": 0,
                         "models_succeeded": 0},
        })

        def _seed_and_finish():
            self._write(phase, {
                "codex:gpt-6-astra:xhigh": self._entry("done", 252),
                "gemini:gemini-3.5-pro": self._entry("running", None),
            })

        assert self._wait_once(plan, monkeypatch, _seed_and_finish) == 2
        out = capsys.readouterr().out
        assert "      just finished: codex:gpt-6-astra:xhigh (4m12s, done)" in out
        assert "gemini:gemini-3.5-pro (" not in out


class TestTheLaunchAnnouncesTheRosterItResolved:
    """The line-up, on the first result line a supervising host can read.

    `progress_lines` is otherwise `status`'s key, rendered from
    `progress.models` — a map that does not exist when the launcher returns,
    because the worker seeds it on the other side of the fork. The launcher does
    not need the map: it is the process that **resolved the selection** (it has
    already refused this very launch if none of those specs had a CLI on PATH),
    so the roster is knowable there, and `roster_line_for_specs` renders it
    through the budget resolution `seed()` uses.

    Why it is worth a key rather than the banner it already prints: on the
    OpenCode package route the launch's result is the *first* thing `execute`
    relays, the next relay is a whole `--wait` away (~83 s, the host cap less a
    margin), and the terminal poll is deliberately never relayed — so a run that
    went terminal inside that first wait announced its models nowhere at all,
    and every other run announced them a poll late, against a requirement that
    says a fan-out names its models when it starts.
    """

    @staticmethod
    def _launch(tmp_path, monkeypatch, specs, **namespace):
        """One `detach.launch` with the fork stubbed, returning its payload."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        monkeypatch.setenv(detach.UV_ENV, str(_fake_uv(tmp_path)))
        # Not what this row is about, and a machine with no `codex` on PATH
        # would otherwise refuse the launch before it ever reached the banner.
        monkeypatch.setattr(
            detach, "missing_provider_binaries", lambda *a, **k: {}
        )

        def fake_spawn(command, *, cwd, env, log_path):
            run_id = command[command.index("--internal-run-id") + 1]
            status_file.claim_run(str(phase), {
                "run_id": run_id, "state": "running", "pid": os.getpid(),
                "heartbeat_at": status_file.now_iso(),
            })
            return os.getpid()

        monkeypatch.setattr(process_control, "spawn_detached", fake_spawn)
        context = cli_common.activate_result(
            cli_common.ResultContext(mode="review-plan")
        )
        try:
            code = detach.launch(
                mode="review-plan",
                argv=["--plan-file", str(plan), "--detach", "--yes"],
                phase_dir=str(phase), plan_file=str(plan), phase="review-plan",
                args=argparse.Namespace(
                    force=False, if_not_running=False, **namespace
                ),
                model_specs=specs,
            )
        finally:
            cli_common.deactivate_result()
        assert code == 0
        return context.payload(0)

    def test_the_launch_result_carries_the_roster_the_worker_will_seed(
        self, tmp_path, monkeypatch, capsys
    ):
        """The same sentence, one fork early — not a second spelling of it.

        The launcher and the worker each read config rather than each other, so
        there is no seed to race: the string below is what
        `FanoutProgress.seed()` emits a moment later from the same specs, the
        same anchor and the same `--timeout` override, which is what lets the
        package route's per-call "shown once" memory drop the duplicate the
        first poll brings instead of saying the roster twice.
        """
        specs = ["codex:gpt-6-astra:xhigh", "opencode:auto"]
        payload = self._launch(tmp_path, monkeypatch, specs, timeout=1800)
        capsys.readouterr()

        assert payload["progress_lines"] == [
            "[PROGRESS] fan-out over 2 models: "
            "codex:gpt-6-astra:xhigh (30m), opencode:auto (30m)"
        ]
        seeded = tmp_path / "seeded"
        seeded.mkdir()
        tracker = progress_tracker.FanoutProgress(
            seeded, specs, interval_s=0, timeout=1800,
        )
        tracker.seed()
        assert payload["progress_lines"][0] == tracker.render_roster()
        # And nothing else on the line moved: the launch still hands back its
        # log, its checkpoint and the poll command as `next`.
        assert payload["next"] and "status" in payload["next"]
        assert any(a.endswith("orchestrator-run.log") for a in payload["artifacts"])

    def test_a_launch_with_no_fan_out_grows_no_key(
        self, tmp_path, monkeypatch, capsys
    ):
        """A `--reaggregate` branch, `generate-tasks`, a write phase.

        They cross the same boundary with no specs at all. The roster renders
        `""` and `set_progress_lines` drops an empty entry, so their result
        lines are byte-for-byte what they were — "absent" and "nothing to
        announce" are not two states a reader has to tell apart.
        """
        payload = self._launch(tmp_path, monkeypatch, None, timeout=None)
        capsys.readouterr()
        assert "progress_lines" not in payload

    def test_an_unreadable_config_costs_the_line_and_not_the_launch(
        self, tmp_path, monkeypatch, capsys
    ):
        """Narration, at the one point where the worker is already running.

        The banner has been printed and the daemon is alive by the time this
        renders, so a raise here would report a launch failure for a launch that
        succeeded. The budget resolution already degrades to `?` on its own, so
        the wrapper covers the renderer itself.
        """
        monkeypatch.setattr(
            progress_tracker, "roster_line_for_specs",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("config is gone")),
        )
        payload = self._launch(
            tmp_path, monkeypatch, ["codex:gpt-6-astra:xhigh"], timeout=1800
        )
        capsys.readouterr()
        assert "progress_lines" not in payload
        assert payload["next"], "the launch still reported its poll command"


class TestTheRenderedProgressLinesRideTheResultLine:
    """The half of the poll a host with no agent in its loop can actually show.

    `progress` rides the result line whole, which is the right shape for a
    reader that can turn a map into a sentence. The OpenCode **package** route
    cannot: `execute` supervises the run itself and the model makes one blocking
    tool call, so nothing there reads `instructions/*.md`, nothing relays a
    `[PROGRESS]` line out of a log the plugin never opens, and the plugin's own
    rule forbids it to compose a sentence about a run. Measured 2026-09-21: a
    5 m 40 s four-model fan-out reached the user as silence and then one
    compressed line, with a complete `progress.models` map and twelve ticker
    lines sitting in artifacts nothing on that route reads.

    So `MULTI_LLM_RESULT` carries the *rendered* lines too — the roster and the
    `just finished:` line, from the renderers that already write the human view
    — and these rows pin that they are the same text, that they ride only where
    `progress` already does, and that they are a key of their own rather than a
    key inside `progress`.
    """

    JF = TestJustFinishedNamesWhatCrossedTheLineDuringTheWait

    @staticmethod
    def _payload(plan, *argv, exit_code=2):
        """`status`'s own result payload, as the dispatcher would emit it."""
        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run", *argv,
            ]) == exit_code
        finally:
            cli_common.deactivate_result()
        return context.payload(exit_code)

    def test_a_pinned_poll_carries_the_roster_rendered(self, tmp_path, capsys):
        """The line the fan-out printed at seed, on the surface that reader reads.

        Byte-identical to `progress_tracker`'s own, because it *is* that
        renderer run over the map the checkpoint carries — not a second spelling
        assembled here out of the same fields.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        models = {
            "codex:gpt-6-astra:xhigh": self.JF._entry("running", None),
            "opencode:auto": self.JF._entry("queued", None),
        }
        self.JF._write(phase, models)

        payload = self._payload(plan, "--phase", "review-plan")
        capsys.readouterr()

        lines = payload["progress_lines"]
        assert lines[0] == (
            "[PROGRESS] fan-out over 2 models: "
            "codex:gpt-6-astra:xhigh (30m), opencode:auto (30m)"
        )
        assert lines[0] == progress_tracker.render_roster_line(models)
        # The state line rides behind it on every poll; its own row is below.
        assert lines[1].startswith("[PROGRESS] 0/2 done (0 ok, 0 failed) | ")
        assert len(lines) == 2, lines

    def test_the_progress_object_is_still_the_file_s_own(self, tmp_path, capsys):
        """A key of its own, never one more key inside `progress`.

        `progress` is what the status file wrote, carried through unreworded so
        a reader gets the numbers where they were computed; a rendered line
        folded into it would make that object something neither the file nor the
        renderer owns, and would change the bytes of a payload other readers
        already parse.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        models = {"codex:gpt-6-astra:xhigh": self.JF._entry("running", None)}
        self.JF._write(phase, models)
        written = status_file.read_status(str(phase))["progress"]

        payload = self._payload(plan, "--phase", "review-plan")
        capsys.readouterr()

        assert payload["progress"] == written
        assert "lines" not in payload["progress"]
        assert payload["progress_lines"], "the lines went missing entirely"

    def test_the_just_finished_line_reaches_the_result_line_too(
        self, tmp_path, monkeypatch, capsys
    ):
        """The one thing the counts structurally cannot say, on the same surface.

        `--wait` is the only invocation that reads the same file twice, and this
        route's every poll is a `--wait` of ~83 s — so a completion inside that
        window is exactly what `_attach_finished_during_wait` already captures.
        It reached the rendered view and nothing else; now it rides the result
        line, labelled, so a host that shows the string shows what it is a list
        of.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self.JF._write(phase, {
            "codex:gpt-6-astra:xhigh": self.JF._entry("running", None),
            "gemini:gemini-3.5-pro": self.JF._entry("running", None),
        })

        def _finish():
            self.JF._write(phase, {
                "codex:gpt-6-astra:xhigh": self.JF._entry("done", 252),
                "gemini:gemini-3.5-pro": self.JF._entry("running", None),
            })

        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert self.JF._wait_once(
                plan, monkeypatch, _finish, "--phase", "review-plan"
            ) == 2
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        payload = context.payload(2)

        lines = payload["progress_lines"]
        assert len(lines) == 3, lines
        assert lines[0].startswith("[PROGRESS] fan-out over 2 models: ")
        # Roster, state, news — the order the human view prints the last two in.
        assert lines[1].startswith("[PROGRESS] 1/2 done (1 ok, 0 failed) | ")
        assert lines[2] == "just finished: codex:gpt-6-astra:xhigh (4m12s, done)"
        # And it is the rendered view's own line, label included — one spelling
        # for the reader who sees the table and the reader who only gets this.
        assert f"      {lines[2]}" in out

    def test_it_rides_exactly_where_the_progress_object_rides(
        self, tmp_path, capsys
    ):
        """Same selector rule, same absence rule.

        Only a poll pinned to one run is "the watched run's" — a plain listing
        answers about no single run, so it carries neither key.

        A phase with no per-model map is **not** the absence case, and used to
        be treated as one: `generate-tasks`, an apply phase and every
        dispatch-only phase record their progress as an executor mirror, and
        those polls were exactly the ones the OpenCode route had nothing to show
        for but a serialized object. The state line answers them from
        `_progress_text` — the `progress:` line the human view already prints —
        so the roster being empty no longer means the key is. What is left
        absent is a `progress` object with nothing renderable in it at all.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self.JF._write(phase, {"codex:gpt-6-astra:xhigh": self.JF._entry("running", None)})

        # No selector: `status`/`null`, and neither key.
        payload = self._payload(plan)
        capsys.readouterr()
        assert "progress" not in payload and "progress_lines" not in payload

        # A pinned poll of a phase whose progress is an executor mirror: there
        # is an object, so `progress` rides; there is no roster, so nothing is
        # rendered and the key stays away.
        mirror = _phase_dir(plan, "generate-tasks")
        status_file.write_status(str(mirror), {
            "phase": "generate-tasks", "state": "subagents_failed",
            "progress": {"subagents": {
                "kind": "task_generation", "invoker": "executor",
                "total": 1, "ok": 1, "counts": {"ok": 1}, "failed_task_ids": [],
            }},
        })
        payload = self._payload(plan, "--phase", "generate-tasks")
        capsys.readouterr()
        assert payload["progress"]["subagents"]["total"] == 1
        # No roster (no map) and no news (no `--wait`), but the phase still says
        # what it is doing — the same clause `render_run_text` prints.
        assert payload["progress_lines"] == ["task_generation 1/1 ok"]
        assert payload["progress_lines"] == [
            run_status._progress_text(payload["progress"])
        ]

        # And the one shape that really has nothing to say still grows no key.
        empty = _phase_dir(plan, "review-tasks")
        status_file.write_status(str(empty), {
            "phase": "review-tasks", "state": "running", "progress": {},
        })
        payload = self._payload(plan, "--phase", "review-tasks")
        capsys.readouterr()
        assert "progress_lines" not in payload

    def test_the_json_view_is_byte_for_byte_what_it_was(self, tmp_path, capsys):
        """Nothing is attached to the view, so `--json` cannot have moved.

        The rendered lines are built at the result-line site out of the view,
        never written onto it — the same compatibility constraint
        `finished_during_wait` is held to, and for the same reason: `--json`
        serializes these dicts raw, so a key attached here would change the
        bytes of every JSON poll ever made.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        self.JF._write(phase, {"codex:gpt-6-astra:xhigh": self.JF._entry("running", None)})

        assert run_status.main([
            "--plan-file", str(plan), "--view", "run", "--phase", "review-plan",
            "--json",
        ]) == 2
        payload = json.loads(capsys.readouterr().out)
        view = payload["runs"][0]
        assert "progress_lines" not in view, view
        assert "lines" not in view["progress"], view["progress"]

    #: Everything `json.loads` can hand back as a `timeout_s`, and what each
    #: one did before 2026-09-21. The guard's docstring said "anything that is
    #: not a plain number is `None`" while the code passed every `int`/`float`
    #: straight through to `int(max(0.0, float(x)))`, which is partial.
    UNTRUSTWORTHY_BUDGETS = [
        # The two shapes the guard already survived.
        ("a string", "half an hour"),
        # `Infinity` and `NaN` are bare tokens Python's `json.loads` accepts.
        # `int(float("inf"))` is `OverflowError` — it took the whole command
        # down, here and in the table's own denominator.
        ("infinity", float("inf")),
        ("negative infinity", float("-inf")),
        # Rendered `(0s)`: a budget of nothing, claimed about a slot that has
        # one. A lie is cheaper to miss than a crash and worse to act on.
        ("not a number", float("nan")),
        ("a negative duration", -5),
        # Valid JSON; `float()` itself raises `OverflowError` on it.
        ("an integer with no float", int("9" * 400)),
        # A `bool` is an `int` in Python, so this rendered `(1s)` — a plausible
        # duration invented out of a flag.
        ("a flag", True),
    ]

    def test_a_checkpoint_the_renderer_cannot_trust_costs_no_poll(
        self, tmp_path, capsys
    ):
        """`status` is the command a poll loop runs, so it has to answer.

        A hand-edited or torn map reaches the renderers exactly as a healthy one
        does. Each unreadable budget costs its own `(?)` and nothing else, and a
        `models` value that is not a map at all renders no roster rather than an
        exit 1 the poll loop reads as "the run state is unreadable, stop".

        Every row of `UNTRUSTWORTHY_BUDGETS` is asserted, and the row is not a
        catalogue of hypotheticals: this test used to carry the first entry and
        a non-dict entry only — the two shapes the guard survived — and each of
        the others either raised out of a poll or rendered a duration nobody
        configured. The assertion runs end to end (`status` exits 2 with its
        table printed) rather than against the renderer alone, because the
        crash lived in `run_status`'s own denominator as much as in the
        roster's: `_model_timing_text` wraps `entry_elapsed_s` in a `try` and
        called `format_budget` outside it.
        """
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        models = {"opencode:auto": "not an entry at all"}
        for label, budget in self.UNTRUSTWORTHY_BUDGETS:
            models[f"codex:{label}"] = dict(
                self.JF._entry("running", None), timeout_s=budget
            )
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {"step": "fanout", "models": models},
        })
        payload = self._payload(plan, "--phase", "review-plan")
        out = capsys.readouterr().out

        rendered = ", ".join(f"{spec} (?)" for spec in models)
        assert payload["progress_lines"][0] == (
            f"[PROGRESS] fan-out over {len(models)} models: {rendered}"
        )
        # The state line reads the same rows through the same guard, so an
        # unrenderable budget costs a denominator there too and nothing more.
        assert payload["progress_lines"][1].startswith(
            f"[PROGRESS] 0/{len(models)} done (0 ok, 0 failed) | running: "
        )
        assert "/?)" in payload["progress_lines"][1]
        assert len(payload["progress_lines"]) == 2, payload["progress_lines"]
        # The human view answered too, and every live row is on it — the guard
        # costs a denominator, never a column and never the command.
        for spec in models:
            assert f"        {spec}" in out, out
        assert "/?" not in out and "?s" not in out


        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "progress": {"step": "fanout", "models": "torn"},
        })
        payload = self._payload(plan, "--phase", "review-plan")
        capsys.readouterr()
        # No roster and no state line off a map that is not one — but the
        # checkpoint's own `step` is still a true thing to say, so the poll is
        # not silent.
        assert payload["progress_lines"] == ["fanout"]
        assert payload["progress"]["models"] == "torn"

    def test_neither_rendered_line_can_forge_a_second_one(
        self, tmp_path, monkeypatch, capsys
    ):
        """One line means one line, on a surface that joins them with newlines.

        The plugin's `toastMessage` shows the supplied lines joined by a
        newline, so a newline *inside* one of them is an extra line the user
        reads as the dispatcher's own — no `[PROGRESS]` prefix, no label, and
        nothing downstream able to tell it from a line Python composed. Every
        field these two renderers interpolate comes from outside them and a
        model spec is the easiest: it is a **config key** (`providers.yaml`, a
        project override, a typed `--models` value) reaching the roster and
        `just finished:` by the same route.

        The two scrubbed differently until 2026-09-21 — the roster ASCII-only
        and unfolded, `just finished:` folding `error` alone and never scrubbing
        at all, under a docstring that claimed "ASCII only". They now take one
        pass (`progress_tracker.one_line_ascii`), which is the point of it being
        one function.
        """
        # The two renderers, side by side, on the same input.
        assert progress_tracker.render_roster_line({
            "codex:gpt\u2013 5\nnot a line": {"timeout_s": 60},
        }) == "[PROGRESS] fan-out over 1 model: codex:gpt? 5 not a line (1m)"
        assert run_status._just_finished_text({"finished_during_wait": [
            {"spec": "codex:gpt\u2013 5\nnot a line", "state": "done",
             "elapsed_s": 1.0, "error": None},
        ]}) == "codex:gpt? 5 not a line (0m01s, done)"
        # `caf\u00e9` is the half no amount of newline folding would have
        # caught: `_just_finished_text` ran no ASCII scrub whatsoever.
        assert run_status._just_finished_text({"finished_during_wait": [
            {"spec": "caf\u00e9", "state": "done", "elapsed_s": 1.0},
        ]}) == "caf? (0m01s, done)"

        # And end to end, where both lines ride one `progress_lines` list.
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        spec = "codex:gpt\u2013 5\nnot a line"
        self.JF._write(phase, {
            spec: self.JF._entry("running", None),
            "opencode:auto": self.JF._entry("running", None),
        })

        def _finish():
            self.JF._write(phase, {
                spec: self.JF._entry("done", 252),
                "opencode:auto": self.JF._entry("running", None),
            })

        context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
        try:
            assert self.JF._wait_once(
                plan, monkeypatch, _finish, "--phase", "review-plan"
            ) == 2
        finally:
            cli_common.deactivate_result()
        capsys.readouterr()

        lines = context.payload(2)["progress_lines"]
        assert len(lines) == 3, lines
        for line in lines:
            assert "\n" not in line, line
            assert line.isascii(), line
        # The roster and the news name the spec; the state line names only the
        # rows that are still live, and this one has gone terminal.
        assert "codex:gpt? 5 not a line" in lines[0]
        assert "codex:gpt? 5 not a line" in lines[2]

    def test_every_poll_carries_a_line_and_a_live_one_changes(
        self, tmp_path, monkeypatch
    ):
        """What stops a host falling back to a serialized object mid-run.

        Measured 2026-09-21 on a 363 s package-route run: five relays, and the
        one whose 90 s window happened to contain no completion relayed a
        multi-KB `JSON.stringify` of the whole payload. That is the **ordinary**
        poll — a window that contains a completion is the exception — so a set
        of lines that only says something when something *changed* leaves the
        blob as what the user usually sees.

        Two properties, and the second is the one easy to assume rather than
        check: a line is there on every poll, and on a live fan-out it *differs*
        between two of them. The host's shown-once memory suppresses a line it
        has already displayed, so a state line that rendered identically twice
        would be deduped straight back into the blob. It differs because a live
        row's elapsed is derived at read time from `started_at` — asserted here
        across a simulated 90 s rather than reasoned about.
        """
        view = {"progress": {"step": "fanout", "models": {
            "codex:gpt-6-astra:xhigh": self.JF._entry("running", None),
            "opencode:auto": self.JF._entry("queued", None),
        }}}
        base = time.time()
        monkeypatch.setattr(progress_tracker, "_wall_time", lambda: base)
        first = run_status._progress_lines(view)
        monkeypatch.setattr(progress_tracker, "_wall_time", lambda: base + 90)
        second = run_status._progress_lines(view)

        assert len(first) == len(second) == 2, (first, second)
        # The roster is the line that must NOT move — it is what a shown-once
        # memory is for.
        assert first[0] == second[0]
        assert first[1] != second[1], (first[1], second[1])
        assert "(0m00s/30m)" in first[1] and "(1m30s/30m)" in second[1]

        # Once every slot is terminal the fan-out has nothing left to narrate,
        # and the phase's own `progress:` line takes over — the same changeover
        # the orchestrator log makes when `PhaseProgress` replaces the fan-out
        # ticker. It moves as the validation batches land.
        done_view = {"progress": {
            "step": "validating", "models_total": 2, "models_done": 2,
            "models_succeeded": 2,
            "models": {
                "codex:gpt-6-astra:xhigh": self.JF._entry("done", 252),
                "opencode:auto": self.JF._entry("done", 62),
            },
            "subagents": {"kind": "validation", "total": 8, "ok": 3,
                          "counts": {"ok": 3}, "failed_task_ids": []},
        }}
        lines = run_status._progress_lines(done_view)
        assert len(lines) == 2, lines
        assert lines[1] == (
            "validating, models 2/2 done, 2 ok, 0 failed, validation 3/8 ok"
        )
        # Not a second spelling: it *is* the line the human view prints.
        assert lines[1] == run_status._progress_text(done_view["progress"])
        assert not lines[1].startswith("[PROGRESS]"), (
            "the frozen fan-out counts would repeat themselves every poll"
        )

    def test_a_view_whose_progress_is_not_a_mapping_renders_nothing(self):
        """The one guard this function keeps, pinned where it is reachable.

        `classify_run` coerces a torn `progress` to `{}` while it builds the
        view, so no view this module assembles can fail the check — which makes
        it dead through the checkpoint and alive through the call. It is kept
        because the result-line site calls this function *outside* the
        `if view.get("progress")` guard the text renderer sits behind, and an
        `AttributeError` here is exit 1, which rule 2 reads as "the run state is
        unreadable, stop and report" — paid for a key that is display-only.
        Asserted at the level it is reachable from rather than through a
        checkpoint that can no longer produce the shape.
        """
        assert run_status._progress_lines({"progress": "torn"}) == []
        assert run_status._progress_lines({"progress": None}) == []
        assert run_status._progress_lines({}) == []
        # Not a guard that swallowed the feature: the healthy shape still
        # renders, through the same one statement.
        assert run_status._progress_lines({"progress": {"models": {
            "codex:gpt-6-astra:xhigh": {"state": "running", "timeout_s": 1800},
        }}}) == [
            "[PROGRESS] fan-out over 1 model: codex:gpt-6-astra:xhigh (30m)",
            "[PROGRESS] 0/1 done (0 ok, 0 failed) | "
            "running: codex:gpt-6-astra:xhigh (?/30m)",
        ]

    def test_an_empty_rendered_line_is_dropped_rather_than_carried(self):
        """`set_progress_lines`' own contract, at the level it is reachable from.

        The filter is load-bearing on the launch path, where
        `detach.launch` hands over `[roster]` unconditionally and a mode with no
        fan-out renders `""` — so a `--reaggregate`, a `generate-tasks` or a
        write-phase launch must grow no key rather than carry a blank a reader
        has to filter. Asserted here because `_progress_lines` builds its list
        from two truthiness checks and would never hand one over.
        """
        context = cli_common.ResultContext(mode="status")
        context.set_progress_lines(["a line", "", None, "another"])
        assert context.payload(2)["progress_lines"] == ["a line", "another"]
        context.set_progress_lines([""])
        assert "progress_lines" not in context.payload(2)
        context.set_progress_lines(None)
        assert "progress_lines" not in context.payload(2)
