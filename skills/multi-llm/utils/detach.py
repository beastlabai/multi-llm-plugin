#!/usr/bin/env python3
"""``--detach``: the orchestrator self-daemonizes, and the run stays findable.

Several host harnesses cap a single shell command at 2–10 minutes, and the
fan-out review/ask modes routinely run longer than any of those caps. Claude Code
has a native background Bash, so it keeps using it; every other host needs the
orchestrator to put *itself* in the background and hand back a way to watch and
stop the run. That is this module.

**Exactly one backgrounding mechanism per host — never both.** ``--detach`` inside
a host's own background primitive double-backgrounds: the launcher returns in
milliseconds, the host reports "finished", and the real work continues in a
detached grandchild whose log path and abort command were printed into a shell
nobody reads. A shim picks one mechanism; Claude Code's is its own.

The three halves, and where each lives:

* **The invoker** (:func:`launch`) runs entirely in the foreground: pre-flight,
  the concurrency ceiling, the lock pre-check, log rotation, the spawn, and a
  bounded handshake that waits until the worker has published its own identity.
  Every failure here exits non-zero *on the caller's terminal* with no child
  spawned and no log file written, so a failed launch is indistinguishable from a
  failed foreground run.
* **The worker** (:class:`RunHandle`) owns the phase dir for the run's lifetime:
  the lock, ``run`` in ``{phase}/.status.json``, the heartbeat, the signal
  handlers, and — because after the POSIX double-spawn *nothing* is its parent and
  no caller can ever observe its wait status — its own terminal record. Every run
  ends in exactly one of ``done`` / ``failed`` / ``aborted`` with
  ``run.exit_code`` recorded.
* **The reader** (:func:`classify_run`, and ``utils/run_status.py`` on top of it)
  answers "is this run alive?" from files alone, closes out runs whose heartbeat
  expired *and* whose process is provably gone, and never blocks or signals a
  process just to look.

The one gap the design accepts: a death that runs no handler (SIGKILL, OOM,
reboot). It is closed by heartbeat expiry rather than by guessing — ``status``
reports ``running (stale)`` while the liveness probe is inconclusive and
``failed(heartbeat_timeout)`` with ``run.exit_code: null`` once it is not, so "no
exit code was ever observed" is a defined value and never a fabricated ``0``.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import process_control, run_registry, status_file
from .cli_common import (
    EXIT_ERROR,
    EXIT_NEEDS_HUMAN,
    EXIT_NOTHING_TO_DO,
    EXIT_OK,
    dispatcher_command,
    modes_table,
    record_artifact,
    record_next,
    skill_dir,
)
from .status_file import (
    RUN_STATE_ABORTED,
    RUN_STATE_DONE,
    RUN_STATE_FAILED,
    RUN_STATE_RUNNING,
    RUN_STATE_STARTING,
    TERMINAL_RUN_STATES,
)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: The child refreshes ``run.heartbeat_at`` at most this often.
HEARTBEAT_INTERVAL_S = 10.0

#: A heartbeat older than this is expired — six missed beats, not one, so a
#: loaded machine or a slow filesystem cannot make a healthy run look dead.
HEARTBEAT_STALE_AFTER_S = 60.0

#: How long the invoker waits for the worker to publish ``run.state: running``.
HANDSHAKE_TIMEOUT_S = 10.0

#: Rotated logs kept per phase dir.
ROTATED_LOG_KEEP = 5

#: The current run's log. Existing behavior this inherits, not a new artifact:
#: today's ``host_background`` path already redirects here, the instruction corpus
#: already names it, and a poller that may come back after a re-detach reads
#: ``run.log_path`` instead of hardcoding it.
LOG_NAME = "orchestrator-run.log"

LOCK_NAME = ".lock"

#: Set in the child's environment by :func:`launch`. Its presence is what makes a
#: worker willing to take over the terminal (reopen stdio onto the log) and to
#: hard-exit when it discovers its record was closed out from under it — an
#: in-process test or a hand-run ``--internal-worker`` does neither.
WORKER_ENV = "MULTI_LLM_DETACHED_WORKER"

#: Env overrides, all read at call time so a test can set them per-case.
HEARTBEAT_INTERVAL_ENV = "MULTI_LLM_HEARTBEAT_INTERVAL_S"
HANDSHAKE_TIMEOUT_ENV = "MULTI_LLM_HANDSHAKE_TIMEOUT_S"

#: An explicit ``uv`` location, tried before the documented probe chain. A
#: detached process inherits no guarantee about the host's ``PATH`` edits, and a
#: host that installs ``uv`` somewhere unusual should not have to patch code.
UV_ENV = "MULTI_LLM_UV"

#: The modes ``--detach`` covers: the fan-out modes plus the executor-backed write
#: modes (``implement``, the three apply phases, ``generate-tasks``, ``full``),
#: which became long-running when the write executor landed — a phase that
#: schedules N batches through real agent CLIs outlives a host's shell cap exactly
#: as a fan-out does. A mode outside this set still gets a loud exit 3 rather than
#: a flag that silently does nothing.
#:
#: Read from the mode table, never re-listed: ``supports_detach`` is a column on
#: the row, so adding the flag to a mode stays a one-line change there.
DETACH_MODES: Tuple[str, ...] = modes_table().DETACH_MODES

#: Flags the launcher consumes itself and must not pass to the worker.
_LAUNCHER_ONLY_FLAGS: Tuple[str, ...] = ("--detach", "--if-not-running")
_LAUNCHER_ONLY_VALUE_FLAGS: Tuple[str, ...] = ("--max-detached",)

#: Flags whose value is a path the worker must see absolute: it starts in the
#: recorded launch cwd, but a run that outlives its launching shell must not
#: depend on that cwd still meaning the same thing.
_PATH_FLAGS: Tuple[str, ...] = (
    "--plan-file", "--question-file", "--out-dir", "--tasks-file",
)

#: Nested-session variables that must not leak into a detached run.
_STRIPPED_ENV: Tuple[str, ...] = (
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT",
    "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
)


def heartbeat_interval() -> float:
    raw = os.environ.get(HEARTBEAT_INTERVAL_ENV)
    if raw:
        try:
            return max(float(raw), 0.05)
        except ValueError:
            pass
    return HEARTBEAT_INTERVAL_S


def handshake_timeout() -> float:
    raw = os.environ.get(HANDSHAKE_TIMEOUT_ENV)
    if raw:
        try:
            return max(float(raw), 0.1)
        except ValueError:
            pass
    return HANDSHAKE_TIMEOUT_S


class DetachError(Exception):
    """A launch-time failure with a fixed exit code (3 = needs a human)."""

    def __init__(self, message: str, exit_code: int = EXIT_NEEDS_HUMAN):
        super().__init__(message)
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def log_path(phase_dir: str) -> str:
    return os.path.join(str(phase_dir), LOG_NAME)


def lock_path(phase_dir: str) -> str:
    return os.path.join(str(phase_dir), LOCK_NAME)


def rotated_log_path(phase_dir: str, run_id: str) -> str:
    return os.path.join(str(phase_dir), f"orchestrator-run.{run_id}.log")


def rotate_log(phase_dir: str, previous_run_id: Optional[str]) -> Optional[str]:
    """Move an existing log aside so ``orchestrator-run.log`` names *this* run.

    Rotation is scoped to the detach path: the ``host_background`` path keeps
    today's truncate-on-``>`` behavior and never rotates.
    """
    current = log_path(phase_dir)
    if not os.path.exists(current):
        return None
    label = previous_run_id or time.strftime("%Y%m%d-%H%M%S")
    target = rotated_log_path(phase_dir, label)
    try:
        if os.path.exists(target):
            os.unlink(target)
        os.replace(current, target)
    except OSError:
        return None
    _prune_rotated_logs(phase_dir)
    return target


def _prune_rotated_logs(phase_dir: str, keep: int = ROTATED_LOG_KEEP) -> None:
    try:
        rotated = [
            os.path.join(phase_dir, name)
            for name in os.listdir(phase_dir)
            if name.startswith("orchestrator-run.") and name.endswith(".log")
            and name != LOG_NAME
        ]
    except OSError:
        return
    rotated.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
    for path in rotated[:-keep] if len(rotated) > keep else []:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# The re-exec command (D4e): uv, rebuilt — never remembered, never bare python
# ---------------------------------------------------------------------------

def resolve_uv() -> str:
    """The absolute ``uv`` path, by the same probe the shim preflight uses.

    ``MULTI_LLM_UV`` first (an explicit override, also what the hermetic tests
    point at a stub), then ``PATH``, then the two documented install locations.
    An unresolvable ``uv`` is a foreground pre-flight failure, never a child that
    dies into a log.
    """
    override = os.environ.get(UV_ENV)
    if override and os.path.isfile(override):
        return os.path.abspath(override)
    found = shutil.which("uv")
    if found:
        return os.path.abspath(found)
    for candidate in (
        os.path.expanduser("~/.local/bin/uv"),
        os.path.expanduser("~/.cargo/bin/uv"),
    ):
        if os.path.isfile(candidate):
            return candidate
    raise DetachError(
        "ERROR: --detach needs `uv` on PATH and it could not be found.\n"
        "       Looked at: $MULTI_LLM_UV, PATH, ~/.local/bin/uv, ~/.cargo/bin/uv\n"
        "       Install it (https://docs.astral.sh/uv/) or run the phase in the "
        "foreground."
    )


def build_worker_argv(
    mode: str, argv: Sequence[str], *, run_id: str
) -> List[str]:
    """The worker's flags: post-parse, normalized, with implications explicit.

    Rebuilt rather than echoed, so the child is identical whether the invoker was
    reached through a rendered shim, a ``"$MULTI_LLM_SKILL_DIR"`` spelling or a
    direct call:

    * ``--detach`` is **replaced** by the reserved ``--internal-worker`` (plus the
      run id the invoker minted), so the child cannot detach recursively and the
      handshake has something to match on.
    * ``--yes`` is written out rather than left implied — a detached run has no one
      to ask, so the implication is part of the recorded argv.
    * path flags are made absolute.
    * launcher-only flags (the ceiling, ``--if-not-running``) are dropped.
    """
    out: List[str] = []
    index = 0
    tokens = list(argv)
    while index < len(tokens):
        token = tokens[index]
        head = token.split("=", 1)[0]
        if head in _LAUNCHER_ONLY_FLAGS:
            index += 1
            continue
        if head in _LAUNCHER_ONLY_VALUE_FLAGS:
            index += 2 if "=" not in token else 1
            continue
        if head in _PATH_FLAGS:
            if "=" in token:
                value = token.split("=", 1)[1]
                out.append(f"{head}={os.path.abspath(value)}")
                index += 1
            elif index + 1 < len(tokens):
                out += [head, os.path.abspath(tokens[index + 1])]
                index += 2
            else:  # a dangling flag; argparse will report it
                out.append(token)
                index += 1
            continue
        out.append(token)
        index += 1

    if not any(
        t.split("=", 1)[0] in ("--yes", "-y", "--non-interactive") for t in out
    ):
        out.append("--yes")
    out += ["--internal-worker", "--internal-run-id", run_id]
    return out


def build_child_command(
    mode: str,
    argv: Sequence[str],
    *,
    run_id: str,
    uv: Optional[str] = None,
    base: Optional[Path] = None,
) -> List[str]:
    """``[<abs uv>, run, --project, <SKILL_DIR>, --, python, <SKILL_DIR>/multi_llm.py, <mode>, …]``

    Reproducing the invoker's own wrapping is the point: re-launching as
    ``[sys.executable, __file__, *args]`` would look right and be subtly wrong —
    inside ``uv run``, ``sys.executable`` is the interpreter uv resolved, but the
    child would inherit it *without* uv, so no lockfile resolution, no re-sync of
    a stale environment, and no ``--project`` for a worker that may run for an hour
    from a different cwd.
    """
    base = Path(base) if base is not None else skill_dir()
    uv_bin = uv or resolve_uv()
    env_dir = os.environ.get("MULTI_LLM_SKILL_DIR")
    if env_dir and os.path.abspath(env_dir) != os.path.abspath(str(base)):
        print(
            f"WARNING: MULTI_LLM_SKILL_DIR ({env_dir}) disagrees with the running "
            f"skill dir ({base}); using the running one.",
            file=sys.stderr,
        )
    return [
        uv_bin, "run", "--project", str(base), "--", "python",
        str(base / "multi_llm.py"), mode,
        *build_worker_argv(mode, argv, run_id=run_id),
    ]


def child_env(*, base: Optional[Path] = None) -> Dict[str, str]:
    """The invoker's environment, pinned for a process that outlives its shell."""
    base = Path(base) if base is not None else skill_dir()
    env = dict(os.environ)
    for name in _STRIPPED_ENV:
        env.pop(name, None)
    env["MULTI_LLM_SKILL_DIR"] = str(base)
    env[WORKER_ENV] = "1"
    config = env.get("MULTI_LLM_PROVIDERS_CONFIG")
    if config and not os.path.isabs(config):
        env["MULTI_LLM_PROVIDERS_CONFIG"] = os.path.abspath(config)
    return env


def missing_provider_binaries(
    model_specs: Optional[Sequence[str]], *, anchor: Optional[str] = None
) -> Dict[str, str]:
    """``{model_spec: binary}`` for every requested model whose CLI is not on PATH.

    Part of the foreground pre-flight: a provider that is not installed is
    knowable *before* the fork, and discovering it by polling ``status`` and
    reading a log is exactly the experience ``--detach`` is supposed to avoid. The
    binary name comes from the adapter that would launch it, so this cannot drift
    from what the run actually runs.
    """
    missing: Dict[str, str] = {}
    if not model_specs:
        return missing
    try:
        from .provider_registry import get_provider, parse_model_spec
    except Exception:  # pragma: no cover - import guard
        return missing
    for spec in model_specs:
        try:
            provider_name, model = parse_model_spec(spec, anchor=anchor)
            provider = get_provider(provider_name)
            if provider is None:
                continue
            binary = provider.build_command("preflight", model)[0]
        except Exception:
            continue  # an adapter that cannot say is not evidence of absence
        if binary and shutil.which(binary) is None:
            missing[spec] = binary
    return missing


def launch_key(mode: str, plan_path: str, phase_dir: str) -> str:
    """``(mode, plan_path, phase[, ask slug])`` — what "the same run" means."""
    return "|".join([mode, os.path.abspath(str(plan_path)), os.path.abspath(str(phase_dir))])


# ---------------------------------------------------------------------------
# Reading a run: liveness, staleness, close-out
# ---------------------------------------------------------------------------

#: ``status``-level verdicts. ``stale`` is deliberately not a stored state.
VERDICT_NO_RUN = "no_run"
VERDICT_STARTING = "starting"
VERDICT_RUNNING = "running"
VERDICT_STALE = "stale"
VERDICT_DONE = RUN_STATE_DONE
VERDICT_FAILED = RUN_STATE_FAILED
VERDICT_ABORTED = RUN_STATE_ABORTED

TERMINAL_VERDICTS = (VERDICT_DONE, VERDICT_FAILED, VERDICT_ABORTED)


def heartbeat_age(run: Dict[str, Any], now: Optional[float] = None) -> Optional[float]:
    """Seconds since the last heartbeat, or ``None`` when none was recorded."""
    from datetime import datetime

    stamp = run.get("heartbeat_at")
    if not stamp:
        return None
    try:
        beat = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    reference = datetime.now() if now is None else datetime.fromtimestamp(now)
    return max((reference - beat).total_seconds(), 0.0)


def classify_run(
    phase_dir: str,
    *,
    close_out: bool = True,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """The whole run view of one phase dir. Reads files; takes no lock.

    Works *while another run holds the dir* — that is the point of a poll loop —
    and writes at most one thing: the terminal close-out of a run whose heartbeat
    expired **and** whose process is provably gone. ``stale`` is reported only
    while the probe is inconclusive (the pid is alive and its start time matches
    but it has stopped beating — a hung worker, which ``--abort`` resolves), so a
    phase dir never sits at ``running`` forever after a reboot.
    """
    status = status_file.read_status(phase_dir)
    run = status.get("run") if isinstance(status.get("run"), dict) else {}
    view: Dict[str, Any] = {
        "phase_dir": str(phase_dir),
        "phase": status.get("phase"),
        "question": status.get("question"),
        "state": status.get("state"),
        "updated_at": status.get("updated_at"),
        "output_files": status.get("output_files") or {},
        "progress": status.get("progress") or {},
        "run": dict(run),
        "run_state": run.get("state") if isinstance(run, dict) else None,
        "reaggregate_command": resolved_reaggregate_command(status),
        "lock_held": not process_control.lock_is_free(lock_path(phase_dir)),
    }
    if not status:
        view["verdict"] = VERDICT_NO_RUN
        view["reason"] = "no .status.json in this phase dir"
        return view
    if not run:
        view["verdict"] = VERDICT_NO_RUN
        view["reason"] = "checkpoint present but no detached run recorded"
        return view

    run_state = run.get("state")
    view["run_state"] = run_state
    if run_state in TERMINAL_RUN_STATES:
        view["verdict"] = run_state
        return view

    age = heartbeat_age(run, now)
    view["heartbeat_age_s"] = age
    identity = process_control.probe(run.get("pid"), run.get("process_start_time"))
    view["identity"] = identity

    if age is not None and age > HEARTBEAT_STALE_AFTER_S and identity == process_control.DEAD:
        if close_out:
            close_out_dead_run(phase_dir, reason="heartbeat_timeout")
            refreshed = status_file.read_run(phase_dir)
            view["run"] = dict(refreshed)
            view["run_state"] = refreshed.get("state")
            view["verdict"] = refreshed.get("state") or VERDICT_FAILED
            view["lock_held"] = not process_control.lock_is_free(lock_path(phase_dir))
        else:
            view["verdict"] = VERDICT_FAILED
        view["reason"] = "heartbeat_timeout"
        return view

    if identity == process_control.DEAD and run_state == RUN_STATE_STARTING:
        # It never came up and its process is gone: nothing will ever beat.
        if close_out:
            close_out_dead_run(phase_dir, reason="worker_never_started")
            refreshed = status_file.read_run(phase_dir)
            view["run"] = dict(refreshed)
            view["run_state"] = refreshed.get("state")
        view["verdict"] = VERDICT_FAILED
        view["reason"] = "worker_never_started"
        return view

    if age is not None and age > HEARTBEAT_STALE_AFTER_S:
        view["verdict"] = VERDICT_STALE
        view["reason"] = f"heartbeat {age:.0f}s old; process probe {identity}"
        return view

    if identity == process_control.DEAD:
        # The heartbeat is fresh but the process is provably gone (killed
        # between beats, or the pid was recycled and its creation time no longer
        # matches). Reporting `running` for a process we can prove is dead would
        # be a lie; closing it out on the spot would race a worker that is merely
        # slow to write. So: `stale` now, terminal once the heartbeat window
        # expires — which is the two bullets of D4b working exactly as written.
        view["verdict"] = VERDICT_STALE
        view["reason"] = (
            "process is gone (pid/creation-time probe says dead) but its "
            f"heartbeat is only {age:.0f}s old; it closes out as "
            f"failed(heartbeat_timeout) after "
            f"{HEARTBEAT_STALE_AFTER_S:.0f}s"
            if age is not None else
            "process is gone and no heartbeat was ever recorded"
        )
        return view

    view["verdict"] = VERDICT_RUNNING if run_state == RUN_STATE_RUNNING else VERDICT_STARTING
    return view


def close_out_dead_run(phase_dir: str, *, reason: str = "heartbeat_timeout") -> bool:
    """Write the terminal record for a run whose process is gone.

    Compare-and-set on ``run_id``, so a worker that was merely paused (SIGSTOP, a
    suspended laptop) and resumes after being closed out fails the check and exits
    rather than writing over the terminal record. Also breaks the lock, removes
    the registry entry, and kills any surviving member of the recorded process
    group whose creation time still matches — the crash case where no handler ran.
    """
    run = status_file.read_run(phase_dir)
    run_id = run.get("run_id")
    if not run_id or run.get("state") in TERMINAL_RUN_STATES:
        return False
    pid = run.get("pid")
    if pid:
        try:
            process_control.kill_tree(
                int(pid),
                start_time=run.get("process_start_time"),
                escalate_after_s=1.0,
            )
        except Exception:
            pass
    wrote = status_file.update_run(
        phase_dir,
        {
            "state": RUN_STATE_FAILED,
            "exit_code": None,      # no exit status was ever observed
            "error": reason,
            "closed_out_at": status_file.now_iso(),
        },
        expect_run_id=str(run_id),
        quiet=True,
    )
    if wrote:
        holder = process_control.read_lock_holder(lock_path(phase_dir))
        if holder is None or holder.get("run_id") == run_id:
            process_control.break_lock(lock_path(phase_dir))
        plan = run.get("plan_file") or phase_dir
        try:
            run_registry.remove_entry(
                run_registry.resolve_project_root(str(plan)), str(run_id)
            )
        except Exception:
            pass
    return wrote


def abort_run(phase_dir: str, *, force: bool = False) -> Dict[str, Any]:
    """The only way to stop a detached run. Identity probe → group kill → record.

    Returns ``{"result": …}`` where result is ``aborted`` (a live run was stopped),
    ``not_running`` (already terminal, or no record at all — so a cleanup step is
    safe to run unconditionally), ``survived`` (the pid outlived the SIGKILL
    window; a human gets pid, run_id and the log path) or ``needs_force`` (the
    identity probe was inconclusive).
    """
    run = status_file.read_run(phase_dir)
    if not run or run.get("state") in TERMINAL_RUN_STATES:
        return {"result": "not_running", "run": run, "phase_dir": str(phase_dir)}
    run_id = str(run.get("run_id"))
    pid = run.get("pid")
    verdict = process_control.probe(pid, run.get("process_start_time"))
    if verdict == process_control.DEAD:
        close_out_dead_run(phase_dir, reason="aborted_by_request (process already gone)")
        return {"result": "not_running", "run": run, "phase_dir": str(phase_dir),
                "detail": "process was already gone; record closed out"}
    if verdict == process_control.UNKNOWN and not force:
        return {"result": "needs_force", "run": run, "phase_dir": str(phase_dir)}

    kill_verdict = process_control.kill_tree(
        int(pid), start_time=None if force else run.get("process_start_time")
    )
    wrote = status_file.update_run(
        phase_dir,
        {
            "state": RUN_STATE_ABORTED,
            "exit_code": EXIT_NEEDS_HUMAN,
            "error": "aborted_by_request",
            "closed_out_at": status_file.now_iso(),
        },
        expect_run_id=run_id,
        quiet=True,
    )
    holder = process_control.read_lock_holder(lock_path(phase_dir))
    if holder is None or holder.get("run_id") == run_id:
        process_control.break_lock(lock_path(phase_dir))
    try:
        plan = run.get("plan_file") or phase_dir
        run_registry.remove_entry(
            run_registry.resolve_project_root(str(plan)), run_id
        )
    except Exception:
        pass
    if kill_verdict == "survived":
        return {"result": "survived", "run": run, "phase_dir": str(phase_dir)}
    return {"result": "aborted", "run": run, "phase_dir": str(phase_dir),
            "recorded": wrote}


# ---------------------------------------------------------------------------
# The durable, rebindable reaggregate command
# ---------------------------------------------------------------------------

def resolved_reaggregate_command(status: Dict[str, Any]) -> Optional[str]:
    """Rebuild the phase's follow-up command from the *running* skill dir.

    ``.status.json`` stores ``reaggregate_mode`` + ``plan_file`` (durable), never a
    resolved skill-dir-absolute command: the Claude plugin cache is keyed by
    commit hash, so the skill dir moves on every plugin update and a persisted
    absolute command silently points at a directory that no longer exists. Skill-dir
    movement is *rebindable*, so the reader resolves it — this function is that
    read-time resolution, and it is why ``status`` can hand back a command that
    works today for a checkpoint written a week ago.
    """
    mode = status.get("reaggregate_mode")
    plan = status.get("plan_file")
    if not mode or not plan:
        return None
    from .cli_common import reaggregate_command

    return reaggregate_command(str(mode), str(plan))


# ---------------------------------------------------------------------------
# The worker side
# ---------------------------------------------------------------------------

_ACTIVE: Optional["RunHandle"] = None


def active_run() -> Optional["RunHandle"]:
    """The run this process owns, if it is a detached worker."""
    return _ACTIVE


class RunHandle:
    """The worker's ownership of a phase dir: lock, record, heartbeat, close-out."""

    def __init__(
        self,
        phase_dir: str,
        *,
        run_id: str,
        mode: str,
        plan_file: str,
        host: Optional[str] = None,
        detached: bool = True,
    ):
        self.phase_dir = str(phase_dir)
        self.run_id = run_id
        self.mode = mode
        self.plan_file = str(plan_file)
        self.host = host
        self.detached = detached
        self.log_path = log_path(self.phase_dir)
        self.cwd = os.getcwd()
        self.project_root = run_registry.resolve_project_root(self.plan_file)
        self._lock = process_control.PhaseLock(lock_path(self.phase_dir))
        self._stop = threading.Event()
        self._beat: Optional[threading.Thread] = None
        self._closed = False
        self.interval = heartbeat_interval()
        self.hard_exit_on_lost = os.environ.get(WORKER_ENV) == "1"

    # -- startup --------------------------------------------------------
    def start(self) -> None:
        """Take the dir, publish identity, start beating. Raises on conflict."""
        global _ACTIVE
        self._acquire_lock()
        identity = process_control.identity()
        record = {
            "run_id": self.run_id,
            "pid": identity["pid"],
            "process_start_time": identity["process_start_time"],
            "detached": self.detached,
            "host": self.host,
            "mode": self.mode,
            "plan_file": self.plan_file,
            "cwd": self.cwd,
            "log_path": self.log_path,
            "started_at": status_file.now_iso(),
            "heartbeat_at": status_file.now_iso(),
            "heartbeat_interval_s": self.interval,
            "state": RUN_STATE_STARTING,
            "exit_code": None,
            "error": None,
            "resolutions": [],
        }
        status_file.claim_run(self.phase_dir, record)
        run_registry.write_entry(
            self.project_root,
            {
                "run_id": self.run_id,
                "pid": identity["pid"],
                "process_start_time": identity["process_start_time"],
                "plan_file": self.plan_file,
                "phase_dir": self.phase_dir,
                "mode": self.mode,
                "host": self.host,
                "launch_key": launch_key(self.mode, self.plan_file, self.phase_dir),
                "started_at": record["started_at"],
                "log_path": self.log_path,
            },
        )
        process_control.install_worker_signal_handlers(self._on_signal)
        atexit.register(self._atexit)
        self._beat = threading.Thread(
            target=self._beat_loop, name=f"multi-llm-heartbeat-{self.run_id[:8]}",
            daemon=True,
        )
        self._beat.start()
        status_file.update_run(
            self.phase_dir, {"state": RUN_STATE_RUNNING}, expect_run_id=self.run_id
        )
        _ACTIVE = self
        print(f"[DETACHED_RUN] run_id={self.run_id} pid={identity['pid']} "
              f"phase_dir={self.phase_dir}")

    def _acquire_lock(self) -> None:
        payload = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "process_start_time": process_control.process_start_time(os.getpid()),
            "host": self.host,
            "acquired_at": status_file.now_iso(),
        }
        try:
            self._lock.acquire(payload)
            return
        except process_control.LockUnavailable as exc:
            holder = exc.holder or {}
            verdict = process_control.probe(
                holder.get("pid"), holder.get("process_start_time")
            )
            if verdict == process_control.DEAD:
                print(
                    f"WARNING: breaking a stale {LOCK_NAME} left by run "
                    f"{holder.get('run_id')} (pid {holder.get('pid')} is gone).",
                    file=sys.stderr,
                )
                process_control.break_lock(lock_path(self.phase_dir))
                self._lock = process_control.PhaseLock(lock_path(self.phase_dir))
                try:
                    self._lock.acquire(payload)
                    return
                except process_control.LockUnavailable:
                    pass
            run = status_file.read_run(self.phase_dir)
            age = heartbeat_age(run)
            raise DetachError(
                "ERROR: another run owns this phase dir "
                f"({self.phase_dir}).\n"
                f"       holder run_id: {holder.get('run_id')}\n"
                f"       holder pid:    {holder.get('pid')} ({verdict})\n"
                f"       last heartbeat: "
                f"{'never' if age is None else f'{age:.0f}s ago'}\n"
                "       Wait for it, abort it (status <plan> --abort --run-id "
                f"{holder.get('run_id')}), or pass --force to break the lock."
            )

    # -- heartbeat ------------------------------------------------------
    def _beat_loop(self) -> None:
        while not self._stop.wait(self.interval):
            if self._closed:
                return
            ok = status_file.update_run(
                self.phase_dir,
                {"heartbeat_at": status_file.now_iso()},
                expect_run_id=self.run_id,
                quiet=True,
            )
            if ok:
                continue
            # Our record was closed out (or taken over) while we were away. Do
            # not resurrect it: exit and leave the terminal record standing.
            print(
                "ERROR: this run's record in .status.json no longer belongs to "
                f"run {self.run_id}; exiting without overwriting it.",
                file=sys.stderr,
            )
            self._stop.set()
            self._closed = True
            if self.hard_exit_on_lost:
                try:
                    sys.stderr.flush()
                    sys.stdout.flush()
                except (AttributeError, ValueError):
                    pass
                os._exit(EXIT_NEEDS_HUMAN)
            return

    # -- progress -------------------------------------------------------
    def progress(self, **updates: Any) -> None:
        status_file.update_progress(self.phase_dir, updates)

    def resolution(self, site: str, decision: str) -> None:
        status_file.append_resolution(
            self.phase_dir, run_id=self.run_id, site=site, decision=decision
        )
        print(f"[AUTO_RESOLVED] {site}: {decision}")

    # -- shutdown -------------------------------------------------------
    def finish(
        self,
        exit_code: Optional[int],
        *,
        state: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record the terminal state. Idempotent; the first call wins."""
        global _ACTIVE
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if state is None:
            state = state_for_exit(exit_code)
        if error is None and exit_code == EXIT_NEEDS_HUMAN and state == RUN_STATE_FAILED:
            error = "needs_human (exit 3)"
        status_file.update_run(
            self.phase_dir,
            {
                "state": state,
                "exit_code": exit_code,
                "error": error,
                "heartbeat_at": status_file.now_iso(),
                "finished_at": status_file.now_iso(),
            },
            expect_run_id=self.run_id,
            quiet=True,
        )
        run_registry.remove_entry(self.project_root, self.run_id)
        self._lock.release()
        if _ACTIVE is self:
            _ACTIVE = None

    def _atexit(self) -> None:
        """Backstop: no run may be left parked at ``running``.

        Reached when the process exits without anyone calling :meth:`finish` —
        a direct ``sys.exit`` deep in a target, say. The exit status is not
        observable from here, so it is recorded as ``null`` rather than guessed.
        """
        if self._closed:
            return
        self.finish(
            None,
            state=RUN_STATE_FAILED,
            error="worker exited without recording an outcome",
        )

    def _on_signal(self, signum: int) -> None:
        name = _signal_name(signum)
        print(f"\nReceived {name}: aborting this detached run.", file=sys.stderr)
        self.finish(
            EXIT_NEEDS_HUMAN,
            state=RUN_STATE_ABORTED,
            error=f"aborted_by_request ({name})",
        )
        os._exit(EXIT_NEEDS_HUMAN)


def _signal_name(signum: int) -> str:
    import signal as _signal

    try:
        return _signal.Signals(signum).name
    except (ValueError, AttributeError):  # pragma: no cover
        return f"signal {signum}"


def state_for_exit(exit_code: Optional[int]) -> str:
    """Map a process exit status onto the run's terminal state.

    ``0`` and ``2`` are both clean stops (``2`` is "nothing to do / already
    complete"); ``3`` is D6's needs-a-human and ``1`` an unexpected error, and both
    are ``failed`` so a poller's exit-3 branch fires.
    """
    if exit_code in (EXIT_OK, EXIT_NOTHING_TO_DO):
        return RUN_STATE_DONE
    return RUN_STATE_FAILED


def finish_active_run(
    exit_code: Optional[int], *, error: Optional[str] = None
) -> None:
    """Close out this process's run, if it owns one. Called by the dispatcher."""
    handle = _ACTIVE
    if handle is not None:
        handle.finish(exit_code, error=error)


def report_progress(phase_dir: str, **updates: Any) -> None:
    """Record ``progress`` — but only for a detached run.

    A foreground run's ``.status.json`` gains no new keys, which is what keeps
    "with ``--detach`` unset, behavior is unchanged" literally true.
    """
    handle = _ACTIVE
    if handle is None or os.path.abspath(handle.phase_dir) != os.path.abspath(str(phase_dir)):
        return
    handle.progress(**updates)


def record_output_files(phase_dir: str, **paths: Any) -> None:
    """Record this phase's artifacts in ``output_files`` — detached runs only.

    ``status`` on a ``done`` run prints them and repeats them in
    ``MULTI_LLM_RESULT.artifacts``, so a host never globs a phase dir to discover
    what a detached run produced and never has to read the log on the success
    path. Only paths that exist are recorded; a foreground run writes nothing.
    """
    handle = _ACTIVE
    if handle is None or os.path.abspath(handle.phase_dir) != os.path.abspath(str(phase_dir)):
        return
    existing = {
        key: str(value) for key, value in paths.items()
        if value and os.path.exists(str(value))
    }
    if existing:
        status_file.update_status(phase_dir, {"output_files": existing})


def record_resolution(phase_dir: str, site: str, decision: str) -> None:
    """Record one auto-resolved decision point, for a detached run only."""
    handle = _ACTIVE
    if handle is None or os.path.abspath(handle.phase_dir) != os.path.abspath(str(phase_dir)):
        return
    handle.resolution(site, decision)


# ---------------------------------------------------------------------------
# The invoker side
# ---------------------------------------------------------------------------

#: Printed verbatim by every ``--detach`` launch. The poll loop lives here, once,
#: so the shims quote it instead of each host inventing a cadence — and so the two
#: rules that make it safe travel with it: the wait happens *between* calls in the
#: host's own turn loop (never ``sleep <n>; status`` in one shell call, which
#: reintroduces the very shell-cap kill ``--detach`` exists to remove), and
#: closing the session does not stop the run.
POLL_LOOP_TEXT = (
    "How to watch it:\n"
    "  1. run the status command above; it is a file read plus a liveness probe "
    "(sub-second).\n"
    "  2. on exit 2 the run is still going: return to your turn loop and poll "
    "again in ~15s (~30s after the first two minutes — the heartbeat is 10s, so "
    "faster polling learns nothing).\n"
    "  3. stop on exit 0 (read output_files) or exit 3 (read run.error).\n"
    "  4. after this mode's documented ceiling, stop polling and abort rather "
    "than looping forever.\n"
    "  Never collapse the wait into one shell call (`sleep 90; status`): that is "
    "the shell-cap kill --detach exists to remove.\n"
    "  Closing this session does NOT stop the run — only the abort command does."
)


def _print_banner(
    *,
    run_id: str,
    pid: int,
    mode: str,
    phase_dir: str,
    log: str,
    status_cmd: str,
    abort_cmd: str,
) -> None:
    print("")
    print("=== DETACHED RUN STARTED ===")
    print(f"  run_id:  {run_id}")
    print(f"  pid:     {pid}")
    print(f"  mode:    {mode}")
    print(f"  phase:   {phase_dir}")
    print(f"  log:     {log}")
    print(f"  status:  {status_cmd}")
    print(f"  abort:   {abort_cmd}")
    print("")
    print(POLL_LOOP_TEXT)


def status_command_for(
    plan_file: str, *, phase: Optional[str] = None, run_id: Optional[str] = None
) -> str:
    args: List[str] = [plan_file, "--view", "run"]
    if phase:
        args += ["--phase", phase]
    if run_id:
        args += ["--run-id", run_id]
    return dispatcher_command("status", *args)


def abort_command_for(plan_file: str, *, run_id: str) -> str:
    return dispatcher_command("status", plan_file, "--abort", "--run-id", run_id)


def launch(
    *,
    mode: str,
    argv: Sequence[str],
    phase_dir: str,
    plan_file: str,
    phase: Optional[str] = None,
    args: Any = None,
    ask_slug: Optional[str] = None,
    model_specs: Optional[Sequence[str]] = None,
) -> int:
    """Fork the worker and return, printing the log/status/abort commands.

    Everything knowable before the fork is reported before it: an unresolvable
    ``uv``, an unwritable phase dir, the concurrency ceiling and a live lock holder
    all exit non-zero here, with no child spawned and no log file created.
    """
    plan_file = os.path.abspath(str(plan_file))
    phase_dir = os.path.abspath(str(phase_dir))
    force = bool(getattr(args, "force", False))
    if_not_running = bool(getattr(args, "if_not_running", False))
    uv = resolve_uv()

    if not os.path.isdir(phase_dir) or not os.access(phase_dir, os.W_OK):
        raise DetachError(
            f"ERROR: phase dir is not writable: {phase_dir}", exit_code=EXIT_ERROR
        )

    # -- provider binaries: knowable now, invisible once detached -------
    missing = missing_provider_binaries(model_specs, anchor=plan_file)
    if missing:
        listing = "\n".join(
            f"       {spec} -> {binary} (not on PATH)"
            for spec, binary in missing.items()
        )
        if len(missing) == len(model_specs or ()):
            raise DetachError(
                "ERROR: none of the requested models has its CLI installed, so a "
                "detached run could only ever produce failures:\n" + listing
                + "\n       Install one, or pass --models with a provider you have."
            )
        print(
            "WARNING: some requested models have no CLI on PATH and will fail; "
            "the run continues with the rest (a partial aggregate is usable "
            "output):\n" + listing,
            file=sys.stderr,
        )

    # -- re-attach beats relaunch --------------------------------------
    existing = classify_run(phase_dir, close_out=True)
    if existing["verdict"] in (VERDICT_RUNNING, VERDICT_STARTING, VERDICT_STALE):
        existing_id = str(existing["run"].get("run_id"))
        if if_not_running:
            print(f"A detached run is already in flight for {phase_dir}.")
            print(f"  run_id:  {existing_id}")
            print(f"  log:     {existing['run'].get('log_path')}")
            print(f"  status:  {status_command_for(plan_file, phase=phase, run_id=existing_id)}")
            record_artifact(existing["run"].get("log_path") or log_path(phase_dir))
            record_next(status_command_for(plan_file, phase=phase, run_id=existing_id))
            return EXIT_OK
        if force:
            print(
                f"WARNING: --force is taking {phase_dir} over from live run "
                f"{existing_id}. Its next heartbeat fails the run_id "
                "compare-and-set and it exits rather than writing over the new "
                "record — but it may still be mid-write for up to one heartbeat "
                f"interval ({heartbeat_interval():.0f}s).",
                file=sys.stderr,
            )
        else:
            age = existing.get("heartbeat_age_s")
            raise DetachError(
                f"ERROR: a detached run already owns {phase_dir}.\n"
                f"       run_id:        {existing_id}\n"
                f"       pid:           {existing['run'].get('pid')} "
                f"({existing.get('identity')})\n"
                f"       last heartbeat: "
                f"{'never' if age is None else f'{age:.0f}s ago'}\n"
                "       Pass --if-not-running to re-attach to it, abort it "
                f"({abort_command_for(plan_file, run_id=existing_id)}), or pass "
                "--force to take the dir over."
            )

    # -- lock pre-check (authoritative acquisition happens in the worker) --
    lock = lock_path(phase_dir)
    if not process_control.lock_is_free(lock):
        holder = process_control.read_lock_holder(lock) or {}
        verdict = process_control.probe(
            holder.get("pid"), holder.get("process_start_time")
        )
        if verdict == process_control.DEAD or force:
            print(
                f"WARNING: breaking a stale {LOCK_NAME} left by run "
                f"{holder.get('run_id')} (pid {holder.get('pid')}: {verdict}).",
                file=sys.stderr,
            )
            process_control.break_lock(lock)
        else:
            run = status_file.read_run(phase_dir)
            age = heartbeat_age(run)
            raise DetachError(
                f"ERROR: {lock} is held by run {holder.get('run_id')} "
                f"(pid {holder.get('pid')}, {verdict}, last heartbeat "
                f"{'never' if age is None else f'{age:.0f}s ago'}).\n"
                "       Two orchestrators must not interleave .status.json "
                "merges and result files in one phase dir.\n"
                "       Abort the holder or pass --force to break the lock."
            )

    # -- the ceiling ---------------------------------------------------
    project_root = run_registry.resolve_project_root(plan_file)
    ceiling = run_registry.max_detached(getattr(args, "max_detached", None))
    live = run_registry.live_entries(project_root)
    live = [e for e in live if os.path.abspath(str(e.get("phase_dir", ""))) != phase_dir]
    if len(live) >= ceiling:
        lines = [
            f"ERROR: {len(live)} detached runs are already in flight for this "
            f"project (ceiling {ceiling}).",
        ]
        for entry in live:
            run = status_file.read_run(str(entry.get("phase_dir") or ""))
            age = heartbeat_age(run)
            lines.append(
                f"       {str(entry.get('run_id'))[:8]}  {entry.get('mode')}  "
                f"{entry.get('phase_dir')}  last heartbeat "
                f"{'never' if age is None else f'{age:.0f}s ago'}"
            )
        lines.append(
            "       Abort one of them, wait, or raise the ceiling "
            f"(--max-detached N / {run_registry.MAX_DETACHED_ENV})."
        )
        raise DetachError("\n".join(lines))

    # -- fork ----------------------------------------------------------
    run_id = str(uuid.uuid4())
    previous = status_file.read_run(phase_dir).get("run_id")
    rotate_log(phase_dir, previous)
    argv = _snapshot_question_file(argv, phase_dir)
    command = build_child_command(mode, argv, run_id=run_id, uv=uv)
    env = child_env()
    log = log_path(phase_dir)
    try:
        worker_pid = process_control.spawn_detached(
            command, cwd=os.getcwd(), env=env, log_path=log
        )
    except (OSError, process_control.SpawnFailed) as exc:
        raise DetachError(
            f"ERROR: could not start the detached worker: {exc}", exit_code=EXIT_ERROR
        )

    observed = _await_handshake(phase_dir, run_id)
    if observed is None:
        try:
            process_control.kill_tree(worker_pid, escalate_after_s=1.0)
        except Exception:
            pass
        tail = _log_tail(log, 20)
        raise DetachError(
            f"ERROR: the detached worker (pid {worker_pid}) never published "
            f"run {run_id} within {handshake_timeout():.0f}s.\n"
            "       Nothing is left half-detached: the worker was terminated.\n"
            + (f"       Last log lines:\n{tail}" if tail else "")
        )

    status_cmd = status_command_for(
        plan_file, phase=phase, run_id=run_id if ask_slug else None
    )
    _print_banner(
        run_id=run_id,
        pid=int(observed.get("pid") or worker_pid),
        mode=mode,
        phase_dir=phase_dir,
        log=log,
        status_cmd=status_cmd,
        abort_cmd=abort_command_for(plan_file, run_id=run_id),
    )
    record_artifact(log)
    record_artifact(status_file.status_path(phase_dir))
    record_next(status_cmd)
    return EXIT_OK


def _snapshot_question_file(argv: Sequence[str], phase_dir: str) -> List[str]:
    """Copy a dispatcher-written ``--question-file`` into the phase dir.

    A question too long for argv is handed over as a temp file that the
    dispatcher owns for the length of *its* invocation — and the invoker returns
    (deleting it) while the worker is still starting. The handshake happens to
    order those two correctly today, but a detached run must not depend on the
    startup order of a process it cannot see, so the file becomes the run's own.
    Dot-prefixed, so it is invisible to anything that globs the dir.
    """
    out = list(argv)
    for index, token in enumerate(out):
        if token != "--question-file" or index + 1 >= len(out):
            continue
        source = out[index + 1]
        target = os.path.join(phase_dir, ".question.txt")
        try:
            with open(source, "r", encoding="utf-8") as src:
                text = src.read()
            with open(target, "w", encoding="utf-8") as dst:
                dst.write(text)
            out[index + 1] = target
        except (OSError, IOError, UnicodeDecodeError):
            pass  # leave the original path; the worker will report the failure
    return out


def _await_handshake(phase_dir: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Block (bounded) until the worker publishes its own identity.

    The POSIX double-spawn produces three pids and only the final worker is
    authoritative, so the invoker never guesses ``run.pid``: it waits for the
    worker's own record. If that never appears, the launch failed and says so.
    """
    deadline = time.time() + handshake_timeout()
    while time.time() < deadline:
        run = status_file.read_run(phase_dir)
        if run.get("run_id") == run_id and run.get("state") in (
            RUN_STATE_RUNNING, *TERMINAL_RUN_STATES
        ):
            return run
        time.sleep(0.05)
    run = status_file.read_run(phase_dir)
    return run if run.get("run_id") == run_id and run.get("pid") else None


def _log_tail(path: str, lines: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            tail = f.read().splitlines()[-lines:]
    except (OSError, IOError):
        return ""
    return "\n".join(f"         {line}" for line in tail)


# ---------------------------------------------------------------------------
# The one hook the orchestrators call
# ---------------------------------------------------------------------------

def mode_supports_detach(mode: Optional[str]) -> bool:
    return mode in DETACH_MODES


def handle_phase_detach(
    *,
    args: Any,
    mode: str,
    argv: Sequence[str],
    phase_dir: str,
    plan_file: str,
    phase: Optional[str] = None,
    ask_slug: Optional[str] = None,
    model_specs: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """``None`` to keep running in this process; an exit code to return instead.

    Called by each fan-out orchestrator at the point where pre-flight has passed
    and the phase dir exists — arguments checked, inputs read, models resolved,
    output dir created — and *before* any model runs. Both directions of the
    detach story pass through here:

    * ``--internal-worker``: this process *is* the worker. Take the lock, publish
      identity, start the heartbeat, then carry on with the phase in-process.
    * ``--detach``: this process is the invoker. Fork, hand back the log and the
      status/abort commands, and return.
    """
    if getattr(args, "internal_worker", False):
        begin_worker_run(
            phase_dir=phase_dir, mode=mode, plan_file=plan_file, args=args
        )
        return None
    if not getattr(args, "detach", False):
        return None
    if not mode_supports_detach(mode):
        raise DetachError(
            f"ERROR: --detach is not available for mode '{mode}' yet.\n"
            f"       It covers the long-running fan-out modes today: "
            f"{', '.join(DETACH_MODES)}.\n"
            "       Run this mode in the foreground (it emits task JSON and "
            "exits) or under your host's own background mechanism."
        )
    return launch(
        mode=mode,
        argv=argv,
        phase_dir=phase_dir,
        plan_file=plan_file,
        phase=phase,
        args=args,
        ask_slug=ask_slug,
        model_specs=model_specs,
    )


def begin_worker_run(
    *, phase_dir: str, mode: str, plan_file: str, args: Any
) -> "RunHandle":
    """Start this process's ownership of a phase dir as the detached worker."""
    run_id = getattr(args, "internal_run_id", None) or str(uuid.uuid4())
    detached = os.environ.get(WORKER_ENV) == "1"
    if detached:
        # POSIX: the invoker handed us /dev/null; we own the log. Windows: the
        # launcher already handed the log handle in. Either way both
        # stream_bootstrap settings (line buffering AND utf-8/replace) apply to
        # whatever we end up writing to.
        process_control.enter_detached_worker(log_path(phase_dir))
    handle = RunHandle(
        phase_dir,
        run_id=run_id,
        mode=mode,
        plan_file=plan_file,
        host=getattr(args, "host", None),
        detached=detached,
    )
    handle.start()
    return handle
