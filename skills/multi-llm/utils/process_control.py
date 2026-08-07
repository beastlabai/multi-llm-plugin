#!/usr/bin/env python3
"""The one import boundary for OS process control: locking, liveness, detach.

**Ownership note (cross-plan).** ``plans/windows-macos-portability.md`` owns this
file's POSIX/Windows selection and the "no Unix-only stdlib in runtime code"
invariant that keeps ``fcntl``/``msvcrt`` behind it. That plan had not landed when
``--detach`` shipped, and its sequencing rule says whichever plan lands first
creates the file and the second adopts it. So this module was typed here, by the
detach work, deliberately thin:

* Everything detach-specific — process-group / Job Object control,
  creation-time identity, the detached spawn — is contributed *back* to that
  plan and has no other caller.
* Everything generic — the POSIX/Windows split itself, the file lock, the
  liveness probe — is the seam that plan will take over. Signatures here are a
  cross-plan interface: changing one is a change to both plans.

No orchestrator imports ``fcntl`` or ``msvcrt``; they import this. Every
POSIX-only call below is guarded, because this project supports Windows today.

The three primitives and why each exists:

1. **Identity, not just liveness.** A ``.status.json`` left by a prior session can
   name a pid the OS has since recycled, so ``pid`` is corroboration and the
   *process creation time* is identity. ``probe()`` returns ``alive`` only when
   both agree; a mismatch is ``dead`` whatever the pid says.
2. **One writer per phase dir.** :class:`PhaseLock` is ``flock(LOCK_EX|LOCK_NB)``
   on POSIX and ``msvcrt.locking(LK_NBLCK)`` on Windows, held on an open fd for
   the run's lifetime — so a crashed process releases it even though the file
   remains. The holder's identity is written *into* the file so a second launcher
   can name who owns the dir.
3. **Detachment that actually detaches.** POSIX needs two spawns (cut the
   terminal and the fds; then cut the parent, so the worker reparents to init and
   the invoker has no child to reap). Windows needs one
   (``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` already removes the console),
   and the launcher must hand the log handle in because a console-less child has
   no valid stdout to inherit.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

IS_WINDOWS = os.name == "nt"

#: :func:`probe` verdicts.
ALIVE = "alive"
DEAD = "dead"
UNKNOWN = "unknown"

#: Byte offset the Windows lock is taken at. Windows file locks are *mandatory*,
#: so locking byte 0 would make the holder payload unreadable to anyone else —
#: including the ``status`` command, whose whole job is to report who holds the
#: dir. Locking a byte far past the payload keeps the file readable while the
#: lock stays exclusive.
_WINDOWS_LOCK_OFFSET = 1 << 30

#: How long :func:`kill_tree` waits between SIGTERM and SIGKILL.
DEFAULT_ESCALATE_AFTER_S = 10.0


# ---------------------------------------------------------------------------
# Identity + liveness
# ---------------------------------------------------------------------------

def process_start_time(pid: int) -> Optional[str]:
    """The process's creation time as a stable string, or ``None`` if unknown.

    Linux reads ``/proc/<pid>/stat`` field 22 (start time in clock ticks since
    boot); macOS/BSD shell out to ``ps -o lstart=``; Windows asks
    ``GetProcessTimes`` for the creation time. ``None`` means "this platform could
    not tell us", which callers must treat as *inconclusive*, never as a match.
    """
    if pid is None or pid <= 0:
        return None
    if IS_WINDOWS:
        return _windows_start_time(pid)
    linux = _linux_start_time(pid)
    if linux is not None:
        return linux
    return _ps_start_time(pid)


def _linux_start_time(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except (OSError, IOError):
        return None
    # comm (field 2) is parenthesised and may itself contain spaces and ')'.
    close = raw.rfind(")")
    if close == -1:
        return None
    fields = raw[close + 2:].split()
    # After comm the next field is 3, so field 22 is index 19.
    if len(fields) <= 19:
        return None
    return fields[19]


def _ps_start_time(pid: int) -> Optional[str]:
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _windows_start_time(pid: int) -> Optional[str]:  # pragma: no cover - Windows
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except Exception:
        # A locked-down Windows sandbox can deny the DLL or the API. Unknown is a
        # defined answer here (the caller renders it as `stale`); a crash is not.
        return None
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation), ctypes.byref(exit_time),
            ctypes.byref(kernel_time), ctypes.byref(user_time),
        )
        if not ok:
            return None
        return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)


def process_alive(pid: Optional[int]) -> bool:
    """Is *pid* a live process? ``os.kill(pid, 0)`` is not a probe on Windows."""
    if not pid or pid <= 0:
        return False
    if IS_WINDOWS:  # pragma: no cover - Windows
        return _windows_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive but owned by another user.
        return True
    except OSError:
        return False
    return True


def _windows_alive(pid: int) -> bool:  # pragma: no cover - Windows
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except Exception:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def probe(pid: Optional[int], start_time: Optional[str]) -> str:
    """``alive`` / ``dead`` / ``unknown`` for a recorded (pid, start time) pair.

    ``unknown`` is the honest answer when the pid is live but the platform could
    not corroborate identity: the caller renders that as ``stale`` and waits for a
    human (``--abort``) rather than closing the run out on a guess.
    """
    if not pid:
        return DEAD
    if not process_alive(pid):
        return DEAD
    observed = process_start_time(pid)
    if not start_time or not observed:
        return UNKNOWN
    return ALIVE if str(observed) == str(start_time) else DEAD


def identity() -> Dict[str, Any]:
    """This process's own ``(pid, process_start_time)`` pair."""
    pid = os.getpid()
    return {"pid": pid, "process_start_time": process_start_time(pid)}


# ---------------------------------------------------------------------------
# The phase-dir lock
# ---------------------------------------------------------------------------

class LockUnavailable(Exception):
    """Another process holds the lock. Carries the recorded holder, if any."""

    def __init__(self, path: str, holder: Optional[Dict[str, Any]]):
        super().__init__(f"lock held: {path}")
        self.path = path
        self.holder = holder or {}


class PhaseLock:
    """A non-blocking, advisory-on-POSIX / mandatory-on-Windows file lock.

    Held on an open fd for the whole run, so a crash releases it even though the
    file remains on disk — which is why "the file exists" is never treated as
    "someone holds it".
    """

    def __init__(self, path: str):
        self.path = str(path)
        self._fd: Optional[int] = None

    # -- acquisition ----------------------------------------------------
    def acquire(self, payload: Optional[Dict[str, Any]] = None) -> None:
        """Take the lock, recording *payload* in the file. Raises on conflict."""
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_BINARY"):  # pragma: no cover - Windows
            flags |= os.O_BINARY
        fd = os.open(self.path, flags, 0o644)
        try:
            self._lock_fd(fd)
        except (OSError, IOError):
            os.close(fd)
            raise LockUnavailable(self.path, read_lock_holder(self.path))
        self._fd = fd
        if payload is not None:
            self._write_payload(payload)

    def _lock_fd(self, fd: int) -> None:
        if IS_WINDOWS:  # pragma: no cover - Windows
            import msvcrt

            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            os.lseek(fd, 0, os.SEEK_SET)
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        assert self._fd is not None
        body = json.dumps(payload, indent=2).encode("utf-8")
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.truncate(self._fd, 0)
        os.write(self._fd, body)
        try:
            os.fsync(self._fd)
        except OSError:  # pragma: no cover - filesystem without fsync
            pass

    # -- release --------------------------------------------------------
    def release(self, *, remove: bool = True) -> None:
        """Unlock and (by default) delete the file. Idempotent."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if IS_WINDOWS:  # pragma: no cover - Windows
                import msvcrt

                os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        if remove:
            try:
                os.unlink(self.path)
            except OSError:
                pass

    @property
    def held(self) -> bool:
        return self._fd is not None

    def __enter__(self) -> "PhaseLock":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


def read_lock_holder(path: str) -> Optional[Dict[str, Any]]:
    """The identity recorded in a lock file, or ``None``.

    On Windows the payload sits before the locked byte range precisely so this
    read works while the lock is held.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, IOError, ValueError):
        return None


def lock_is_free(path: str) -> bool:
    """True when nobody currently holds *path* (probed, not inferred).

    Used by the ``status`` read path, which must not create the file: a missing
    lock file is free by definition, and an existing one is tested by trying to
    take it and immediately releasing it without deleting.
    """
    if not os.path.exists(path):
        return True
    probe_lock = PhaseLock(path)
    try:
        probe_lock.acquire()
    except LockUnavailable:
        return False
    except OSError:
        return False
    probe_lock.release(remove=False)
    return True


def break_lock(path: str) -> bool:
    """Delete a lock file whose holder is gone. ``True`` when it is gone now."""
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Process-tree teardown
# ---------------------------------------------------------------------------

def kill_tree(
    pid: int,
    *,
    start_time: Optional[str] = None,
    escalate_after_s: float = DEFAULT_ESCALATE_AFTER_S,
    poll_interval_s: float = 0.1,
) -> str:
    """Terminate *pid* and every process it spawned. Returns a verdict string.

    ``gone`` (was already dead), ``killed`` (died, gracefully or after the
    escalation), ``survived`` (still alive after the SIGKILL window) or
    ``skipped`` (identity mismatch — the pid was recycled, so killing it would
    hit an unrelated process).

    The worker is its own process-group leader and every provider CLI it spawns
    inherits that group, so a group signal takes the whole tree down and no
    provider is left burning quota against a run nobody is reading.
    """
    if not pid:
        return "gone"
    verdict = probe(pid, start_time)
    if verdict == DEAD:
        return "gone"
    if verdict == UNKNOWN and start_time:
        # Live pid, unverifiable identity. Callers reach this only via --force.
        pass
    if IS_WINDOWS:  # pragma: no cover - Windows
        return _windows_kill_tree(pid, escalate_after_s)
    return _posix_kill_tree(pid, escalate_after_s, poll_interval_s)


def _posix_kill_tree(pid: int, escalate_after_s: float, poll_interval_s: float) -> str:
    def signal_target(sig: int) -> None:
        sent = False
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = None
        if pgid is not None and pgid != os.getpgid(0):
            try:
                os.killpg(pgid, sig)
                sent = True
            except OSError:
                sent = False
        if not sent:
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    signal_target(signal.SIGTERM)
    deadline = time.time() + max(escalate_after_s, 0.0)
    while time.time() < deadline:
        if not process_alive(pid):
            return "killed"
        time.sleep(poll_interval_s)
    if not process_alive(pid):
        return "killed"
    signal_target(signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not process_alive(pid):
            return "killed"
        time.sleep(poll_interval_s)
    return "survived" if process_alive(pid) else "killed"


def _windows_kill_tree(pid: int, escalate_after_s: float) -> str:  # pragma: no cover
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        CTRL_BREAK_EVENT = 1
        kernel32.GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid)
    except Exception:
        pass
    deadline = time.time() + max(escalate_after_s, 0.0)
    while time.time() < deadline:
        if not process_alive(pid):
            return "killed"
        time.sleep(0.1)
    # A group-less kill would strand the provider CLIs, so escalate to the whole
    # child tree.
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not process_alive(pid):
            return "killed"
        time.sleep(0.1)
    return "survived" if process_alive(pid) else "killed"


# ---------------------------------------------------------------------------
# Detached spawn
# ---------------------------------------------------------------------------

class SpawnFailed(Exception):
    """The detached spawn itself failed (before the worker ran)."""


def spawn_detached(
    argv: Sequence[str],
    *,
    cwd: str,
    env: Dict[str, str],
    log_path: Optional[str] = None,
) -> int:
    """Start *argv* fully detached and return the worker's pid.

    POSIX: two spawns. The first cuts the controlling terminal
    (``setsid``) and every inherited descriptor (stdio to ``/dev/null``,
    everything else non-inheritable, so the host's stdout pipe sees EOF at once);
    the second cuts the parent, so the worker reparents to ``init`` and this
    process returns with no child to wait on and no zombie for the host to reap.
    The worker opens ``log_path`` itself (see :func:`enter_detached_worker`).

    Windows: one spawn. ``DETACHED_PROCESS`` already removes the console, so
    there is no intermediate and nothing to reap — but a console-less child has no
    valid stdout, so the launcher opens the log and hands that one handle in as
    both stdout and stderr.
    """
    argv = [str(a) for a in argv]
    if IS_WINDOWS:  # pragma: no cover - Windows
        return _spawn_detached_windows(argv, cwd=cwd, env=env, log_path=log_path)
    return _spawn_detached_posix(argv, cwd=cwd, env=env)


def _spawn_detached_posix(argv: List[str], *, cwd: str, env: Dict[str, str]) -> int:
    read_fd, write_fd = os.pipe()
    intermediate = os.fork()
    if intermediate == 0:
        # --- spawn one: new session, no controlling terminal, no host fds ---
        try:
            os.close(read_fd)
            os.setsid()
            devnull = os.open(os.devnull, os.O_RDWR)
            for target in (0, 1, 2):
                os.dup2(devnull, target)
            if devnull > 2:
                os.close(devnull)
            worker = os.fork()
            if worker == 0:
                # --- spawn two: the real worker, reparented to init on exit ---
                try:
                    os.close(write_fd)
                except OSError:
                    pass
                try:
                    os.chdir(cwd)
                except OSError:
                    pass
                try:
                    if os.path.isabs(argv[0]):
                        os.execve(argv[0], argv, env)
                    else:
                        os.execvpe(argv[0], argv, env)
                except BaseException:
                    os._exit(127)
                os._exit(127)  # unreachable
            os.write(write_fd, str(worker).encode("ascii"))
            os.close(write_fd)
        except BaseException:
            os._exit(127)
        os._exit(0)
    # --- invoker: read the worker pid, reap the intermediate, return ---
    os.close(write_fd)
    try:
        raw = os.read(read_fd, 32)
    finally:
        os.close(read_fd)
    try:
        os.waitpid(intermediate, 0)
    except OSError:
        pass
    try:
        return int(raw.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError):
        raise SpawnFailed(
            "the detach intermediate did not report a worker pid"
        )


def _spawn_detached_windows(  # pragma: no cover - Windows
    argv: List[str], *, cwd: str, env: Dict[str, str], log_path: Optional[str]
) -> int:
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    handle = None
    if log_path:
        handle = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle if handle is not None else subprocess.DEVNULL,
            stderr=handle if handle is not None else subprocess.DEVNULL,
            close_fds=True,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        )
    finally:
        if handle is not None:
            handle.close()
    return proc.pid


def enter_detached_worker(log_path: Optional[str]) -> None:
    """Worker-side setup: own the log, own the process group, own the tree.

    POSIX: reopen fds 1 and 2 onto *log_path* append-only, so no application
    output — the worker's own or any provider subprocess's — ever passes through a
    descriptor the host handed us, and then re-run the stream bootstrap so the
    log gets line-buffered UTF-8 with ``errors="replace"``. **Both** settings
    matter: without line buffering a detached log stays empty for minutes, and
    without the encoding a Windows-piped stream defaults to cp1252 and raises
    ``UnicodeEncodeError`` on the first em dash a model emits.

    Windows: the launcher already handed the log in as stdout/stderr, so there is
    nothing to reopen; instead assign this process to a Job Object with
    ``KILL_ON_JOB_CLOSE`` so the OS reaps provider children even if the worker
    crashes without running a handler.
    """
    if not IS_WINDOWS and log_path:
        try:
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError:
            fd = None
        if fd is not None:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except (AttributeError, ValueError):
                pass
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            if fd > 2:
                os.close(fd)
    if IS_WINDOWS:  # pragma: no cover - Windows
        _assign_own_job_object()
    from .stream_bootstrap import bootstrap_streams

    bootstrap_streams()


def _assign_own_job_object() -> None:  # pragma: no cover - Windows
    """Best-effort Job Object with KILL_ON_JOB_CLOSE around this process."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        )
        kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
        # The handle is deliberately leaked for the process's lifetime: the job
        # dies (killing the tree) when the last handle closes, i.e. on exit.
        globals()["_JOB_HANDLE"] = job
    except Exception:
        return


def install_worker_signal_handlers(on_abort: Callable[[int], None]) -> None:
    """Route cancellation signals to the abort path; ignore hangups.

    SIGHUP is explicitly *not* an abort trigger: the worker is a session leader
    with no controlling terminal so a host shell's exit cannot deliver one, and
    ignoring it means a host that kills its whole process tree on session
    teardown cannot take the run down silently either.
    """
    def handler(signum, _frame):
        on_abort(signum)

    names = ["SIGTERM", "SIGINT", "SIGBREAK"]
    for name in names:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):  # pragma: no cover - non-main thread
            pass
    sighup = getattr(signal, "SIGHUP", None)
    if sighup is not None:
        try:
            signal.signal(sighup, signal.SIG_IGN)
        except (OSError, ValueError):  # pragma: no cover
            pass
