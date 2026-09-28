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
   remains. **The file is durable; only the lock is ever released** (see
   :meth:`PhaseLock.release`): unlock-then-unlink would let a second process take
   the inode in the gap and a third create a new file at the same path, which is
   two holders of one phase lock. The holder's identity is written *into* the
   file so a second launcher
   can name who owns the dir. **The lock is the authority; the payload is only a
   label.** It is written after the lock is taken, so it can legitimately be
   empty (a brand-new file), half-written, or still naming a dead predecessor
   while a live process holds the lock — which is why an unreadable payload is
   :data:`UNKNOWN` (see :func:`holder_verdict`) and why :func:`break_lock` proves
   the lock is free by *taking* it rather than by trusting what it read.
3. **Detachment that actually detaches.** POSIX needs two spawns (cut the
   terminal and the fds; then cut the parent, so the worker reparents to init and
   the invoker has no child to reap). Windows needs one
   (``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` already removes the console),
   and the launcher must hand the log handle in because a console-less child has
   no valid stdout to inherit.

A fourth, for the Lane A fan-out: **a stopped provider CLI takes its detached
children with it.** :func:`descendants` snapshots a CLI's tree while it is alive,
:func:`terminate_timed_out` stops the CLI (SIGTERM first) and then every
snapshotted process group under :func:`kill_tree`'s identity rule, and
:data:`live_cli_registry` keeps a refreshed snapshot of every live CLI so an
abort or a Ctrl-C can do the same after the CLI itself has already died —
inside a budget that ends before the abort SIGKILLs the worker.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any, Awaitable, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence,
    Tuple, TypeVar,
)

_T = TypeVar("_T")

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


def process_group_id(pid: Optional[int] = None) -> Optional[int]:
    """The process group *pid* belongs to (this process by default), or ``None``.

    Recorded at worker start because it cannot be recovered later: after the
    POSIX double spawn the worker's group leader is the *intermediate*, which has
    already exited, so ``os.getpgid(worker_pid)`` is the only way to read it —
    and that call fails the moment the worker itself dies, which is exactly when
    the close-out needs the group to kill the provider children it left behind.
    ``None`` on Windows, where the tree is a Job Object (D4d) rather than a group.
    """
    if IS_WINDOWS:  # pragma: no cover - Windows
        return None
    try:
        return os.getpgid(os.getpid() if pid is None else int(pid))
    except (OSError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The phase-dir lock
# ---------------------------------------------------------------------------

def _take_fd_lock(fd: int) -> None:
    """Take the exclusive OS lock on *fd*, **non-blocking**. Raises on conflict.

    The one place the platform split for locking lives; both lock shapes below
    (ownership and mutex) are built out of it, so a Windows fix is one edit.
    """
    if IS_WINDOWS:  # pragma: no cover - Windows
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        os.lseek(fd, 0, os.SEEK_SET)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _drop_fd_lock(fd: int) -> None:
    """Release the exclusive OS lock on *fd*."""
    if IS_WINDOWS:  # pragma: no cover - Windows
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


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
        _take_fd_lock(fd)

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        # Write first, *then* truncate to the bytes just written: truncating to
        # zero first would open a window in which a concurrent reader sees an
        # empty file while this process already holds the lock. It cannot be
        # closed entirely (a reader can still catch a short read), which is why
        # no caller is allowed to read "no payload" as "no holder".
        assert self._fd is not None
        body = json.dumps(payload, indent=2).encode("utf-8")
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, body)
        os.truncate(self._fd, len(body))
        try:
            os.fsync(self._fd)
        except OSError:  # pragma: no cover - filesystem without fsync
            pass

    # -- release --------------------------------------------------------
    def release(self) -> None:
        """Drop the OS lock. The file itself is **never** deleted. Idempotent.

        Unlinking here would split the lock into two domains. Unlock-then-unlink
        cannot be made atomic: between them another process legitimately takes
        the lock on this inode, the unlink then removes that holder's *pathname*,
        and a third process ``O_CREAT``s a fresh file at the same path and locks
        that — two workers, two inodes, both believing they own the phase dir,
        interleaving ``.status.json`` merges. (The mirror image is just as bad:
        a holder displaced by ``break_lock(force=True)`` would, on release,
        unlink the path its *successor* now owns.) So the file is durable and the
        lock is what is held: an empty ``{phase}/.lock`` left behind means
        nothing, exactly as it already does after a crash — see
        :func:`lock_is_free`, which probes rather than stats.
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _drop_fd_lock(fd)
        finally:
            try:
                os.close(fd)
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
    read works while the lock is held. ``None`` means *the label is unreadable*
    — never *nobody holds this lock*; :func:`holder_verdict` is the rule.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, IOError, ValueError):
        return None


def holder_verdict(holder: Optional[Dict[str, Any]]) -> str:
    """The identity verdict for a *recorded* lock holder.

    A missing, empty or half-written payload is :data:`UNKNOWN`, never
    :data:`DEAD`. The payload is written after the lock is taken, so "I could not
    read it" is indistinguishable from "the holder is one instruction younger
    than my read" — and every caller treats ``DEAD`` as a licence to break the
    lock. Only a payload that was actually read *and* whose (pid, creation-time)
    pair probes dead earns that.
    """
    if not holder:
        return UNKNOWN
    return probe(holder.get("pid"), holder.get("process_start_time"))


def lock_is_free(path: str) -> bool:
    """True when nobody currently holds *path* (probed, not inferred).

    Used by the ``status`` read path, which must not create the file: a missing
    lock file is free by definition, and an existing one is tested by trying to
    take it and immediately releasing it. The file is never deleted, so "free"
    is always a probe result and never inferred from the file's absence.
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
    probe_lock.release()
    return True


def break_lock(path: str, *, force: bool = False) -> bool:
    """Free a lock whose holder is gone. ``True`` when nobody holds it now.

    The proof that the holder is gone is *the lock itself*: this takes it, so a
    file whose payload is unreadable or names a dead predecessor while a live
    process still holds it is **refused** (``False``) instead of broken.
    Breaking a held lock is the one failure the lock exists to prevent — it puts
    two runs on two inodes of "the" phase lock, interleaving ``.status.json``
    merges.

    A stale lock is therefore broken by *acquiring and releasing* it, never by
    unlinking: a crashed holder's lock is already free (the OS dropped it), and
    the leftover file is inert — the next acquirer overwrites its payload.

    ``force=True`` is the deliberate ``--force`` path, which the plan defines as
    taking the dir over from a **live** holder — the one case an unlink is the
    only available mechanism, since no process can drop another's lock. It still
    tries the non-destructive path first, so forcing a lock nobody holds does
    not needlessly unlink an inode a concurrent launcher may be acquiring.
    """
    if not os.path.exists(path):
        return True
    probe_lock = PhaseLock(path)
    try:
        probe_lock.acquire()
    except LockUnavailable:
        pass
    except OSError:
        # The file could not even be opened: inconclusive, unless it has since
        # vanished — in which case there is nothing left to hold.
        if not os.path.exists(path):
            return True
    else:
        probe_lock.release()
        return True
    if not force:
        return False
    # `--force` against a genuinely held lock. The displaced holder keeps its
    # (now unnamed) inode and cannot unlink the successor's file, because
    # `release` no longer unlinks anything.
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# The short-held mutex (serialization, not ownership)
# ---------------------------------------------------------------------------

#: How long :meth:`FileMutex.acquire` waits for its turn. Generous by three
#: orders of magnitude — a holder does a JSON read plus a temp+rename — because
#: the only thing this bound exists for is to keep a *stopped* holder from
#: hanging a heartbeat thread forever.
DEFAULT_MUTEX_TIMEOUT_S = 10.0


class FileMutex:
    """A **blocking**, briefly-held interprocess mutex over one path.

    :class:`PhaseLock` and this are the same OS primitive in two different
    shapes, kept as two classes because they mean opposite things to a caller:

    * A phase lock is **ownership** — taken once, held for the whole run, and a
      conflict is an error a human is told about (exit 3 naming the holder).
    * A mutex is **serialization** — taken for the microseconds of one
      read-modify-write, and a conflict is the *normal* case: the caller waits
      its turn. Using ``PhaseLock`` for this would turn every concurrent
      checkpoint write into a spurious "someone else owns this phase dir", and
      the run that owns the dir already holds the phase lock for its lifetime,
      so it could never take it a second time.

    The wait is bounded rather than infinite. A holder that *dies* releases the
    lock (the OS drops it with the fd), but one that is *stopped* — SIGSTOP, a
    suspended laptop — never does, and a worker's heartbeat thread blocked
    forever on a checkpoint write is a worse failure than the interleaving the
    lock prevents. :meth:`acquire` therefore reports ``False`` on timeout and
    lets its caller decide; it never raises.

    The file is created on demand and, like every lock file here, is **never
    unlinked**: unlock-then-unlink would put two processes on two inodes of
    "the" mutex (see :meth:`PhaseLock.release`). An existing file means nothing
    about whether anyone holds it.
    """

    def __init__(self, path: str):
        self.path = str(path)
        self._fd: Optional[int] = None

    def acquire(
        self,
        *,
        timeout_s: float = DEFAULT_MUTEX_TIMEOUT_S,
        poll_interval_s: float = 0.005,
    ) -> bool:
        """Wait for the lock; ``True`` when it was taken. Never raises.

        ``False`` means *proceed unserialized or give up* — the file could not
        be opened (a phase dir that does not exist) or the wait expired.
        """
        if self._fd is not None:
            return True
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_BINARY"):  # pragma: no cover - Windows
            flags |= os.O_BINARY
        try:
            fd = os.open(self.path, flags, 0o644)
        except (OSError, IOError, ValueError):
            return False
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            try:
                _take_fd_lock(fd)
            except (OSError, IOError):
                if time.monotonic() >= deadline:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    return False
                time.sleep(poll_interval_s)
                continue
            self._fd = fd
            return True

    def release(self) -> None:
        """Drop the lock, keeping the file. Idempotent, never raises."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _drop_fd_lock(fd)
        except (OSError, IOError):
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    @property
    def held(self) -> bool:
        return self._fd is not None


# ---------------------------------------------------------------------------
# Process-tree teardown
# ---------------------------------------------------------------------------

def new_process_group_kwargs() -> Dict[str, Any]:
    """``Popen`` kwargs that make a child the root of its own process group.

    The precondition for :func:`kill_tree`, and the reason it is a facade call
    rather than a literal at each spawn site: an agentic runner is not a leaf.
    It spawns tool children of its own (a shell, a nested provider CLI), which
    inherit the parent's group *and* the parent's stdout pipe — so a signal to
    the direct child alone leaves them running, still burning quota, still
    holding the pipe open, and an unbounded post-kill ``communicate()`` blocks
    forever on the writer that never went away.

    POSIX: ``start_new_session`` (a real ``setsid`` in the forked child, so it
    is safe to use from a thread), which makes the child a session and group
    leader — its group id *is* its pid, and ``os.killpg`` on it never reaches
    back into the caller's own group. Windows: ``CREATE_NEW_PROCESS_GROUP``,
    which is what makes the child a valid ``CTRL_BREAK_EVENT`` target; it does
    **not** break the child out of the worker's Job Object (D4d), so job
    teardown still covers the tree.
    """
    if IS_WINDOWS:  # pragma: no cover - Windows
        return {
            "creationflags": getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
            )
        }
    return {"start_new_session": True}


def kill_tree(
    pid: int,
    *,
    start_time: Optional[str] = None,
    pgid: Optional[int] = None,
    escalate_after_s: float = DEFAULT_ESCALATE_AFTER_S,
    poll_interval_s: float = 0.1,
) -> str:
    """Terminate *pid* and every process it spawned. Returns a verdict string.

    ``gone`` (was already dead, and nothing of its group survives), ``killed``
    (died, gracefully or after the escalation), ``survived`` (still alive after
    the SIGKILL window) or ``skipped`` (identity could not be corroborated — the
    pid may have been recycled, so signalling it would hit an unrelated process).

    Two identity rules, and they are the whole contract:

    * ``start_time`` supplied and the probe cannot corroborate it ⇒ ``skipped``,
      **no signal**. ``kill_tree`` is reached by routine paths (``util runs
      --sweep`` over week-old entries whose ``/proc`` read failed), and a
      recycled pid's *process group* is an unrelated group.
    * an unverified kill is *requested*, never inferred: pass ``start_time=None``
      — which is how ``--force`` and the failed-handshake teardown spell it.

    The worker is its own process-group leader and every provider CLI it spawns
    inherits that group, so a group signal takes the whole tree down and no
    provider is left burning quota against a run nobody is reading. That group
    outlives its leader, so a recorded *pgid* is still killed when the leader
    itself is gone (the crash case D4b's close-out has to clean up); it is only
    ever the recorded value, never one re-derived from a pid whose identity just
    failed.
    """
    if not pid and not pgid:
        return "gone"
    verdict = probe(pid, start_time) if pid else DEAD
    if verdict == UNKNOWN and start_time:
        return "skipped"
    if verdict == DEAD:
        if pgid and not IS_WINDOWS:
            return _posix_kill_group(pgid, escalate_after_s, poll_interval_s)
        return "gone"
    if IS_WINDOWS:  # pragma: no cover - Windows
        return _windows_kill_tree(pid, escalate_after_s)
    return _posix_kill_tree(pid, escalate_after_s, poll_interval_s, pgid=pgid)


def _resolve_pgid(recorded: Optional[int], pid: Optional[int]) -> Optional[int]:
    """The group to signal: the recorded one, else the live leader's own.

    Never this process's own group (a self-kill) and never init's.
    """
    candidate: Optional[int] = None
    if recorded:
        try:
            candidate = int(recorded)
        except (TypeError, ValueError):
            candidate = None
    if candidate is None and pid:
        try:
            candidate = os.getpgid(int(pid))
        except (OSError, TypeError, ValueError):
            candidate = None
    if candidate is None or candidate <= 1:
        return None
    try:
        if candidate == os.getpgid(0):
            return None
    except OSError:  # pragma: no cover - getpgid(0) does not fail in practice
        return None
    return candidate


def _group_has_members(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _posix_kill_group(pgid: int, escalate_after_s: float, poll_interval_s: float) -> str:
    """SIGTERM → SIGKILL a recorded process group whose leader is already gone."""
    group = _resolve_pgid(pgid, None)
    if group is None:
        return "skipped"
    if not _group_has_members(group):
        return "gone"

    def send(sig: int) -> None:
        try:
            os.killpg(group, sig)
        except OSError:
            pass

    send(signal.SIGTERM)
    deadline = time.time() + max(escalate_after_s, 0.0)
    while time.time() < deadline:
        if not _group_has_members(group):
            return "killed"
        time.sleep(poll_interval_s)
    if not _group_has_members(group):
        return "killed"
    send(signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _group_has_members(group):
            return "killed"
        time.sleep(poll_interval_s)
    return "survived" if _group_has_members(group) else "killed"


def _signal_tree(pid: Optional[int], pgid: Optional[int], sig: int) -> None:
    """*sig* to the group :func:`_resolve_pgid` picks, else to *pid* alone.

    A ``None`` *pid* means no pid is safe to signal (its identity was not just
    corroborated), so a group that cannot be signalled gets nothing at all.
    """
    group = _resolve_pgid(pgid, pid)
    if group is not None:
        try:
            os.killpg(group, sig)
            return
        except OSError:
            pass
    if pid:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _posix_kill_tree(
    pid: int,
    escalate_after_s: float,
    poll_interval_s: float,
    pgid: Optional[int] = None,
) -> str:
    def signal_target(sig: int) -> None:
        _signal_tree(pid, pgid, sig)

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
# A stopped CLI's detached descendants (plan: review-run-scope-salvage-and-
# failure-visibility D3.3)
# ---------------------------------------------------------------------------
#
# A Lane A provider CLI stays in the orchestrator's own process group — moving it
# into a session of its own would stop a terminal Ctrl-C from reaching it — so
# `kill_tree` on its group is not an option, and a signal to the CLI alone misses
# every child that left the group. pi's bash tool is the measured case: each
# shell is spawned `detached: true` (a session and group of its own), and GNU
# `timeout` inside it moves into yet another group. Once the CLI dies those
# children reparent to init and nothing links them to the slot any more, so the
# tree is SNAPSHOTTED (pid, pgid, sid, start time) while the CLI is still alive,
# and every snapshotted group is swept afterwards under `kill_tree`'s identity
# rule: a member's recorded start time is re-checked before anything is sent,
# so a recycled pid is never signalled. The groups are signalled together and
# share one grace and one SIGKILL window — a sweep's cost does not grow with
# the number of groups, which is what lets an abort bound it (below).

#: How long :func:`terminate_timed_out` waits between the CLI's SIGTERM and its
#: SIGKILL, and between each snapshotted group's. SIGTERM first is the point:
#: pi reaps its own tracked shells only from its SIGTERM/SIGHUP handler, which a
#: bare SIGKILL never runs. Short, because the slot has already spent its budget.
DEFAULT_TERMINATE_GRACE_S = 2.0

#: How long the CLI gets to die after its SIGKILL before it is reported
#: ``survived`` (a process in uninterruptible sleep can outlast a SIGKILL).
_KILL_CONFIRM_S = 5.0

#: The whole wall clock :meth:`LiveCliRegistry.sweep` may spend by default. The
#: abort's sweep runs inside the detached worker's SIGTERM handler while
#: ``detach.abort_run``'s :func:`kill_tree` counts :data:`DEFAULT_ESCALATE_AFTER_S`
#: down to the SIGKILL that ends the worker, and every signal the sweep still
#: owes a detached group dies with it — so the sweep has to be *finished*, not
#: merely started, well inside that window. What is left over is the worker's
#: own close-out (``RunHandle.finish``) after the sweep returns.
LIVE_SWEEP_BUDGET_S = DEFAULT_ESCALATE_AFTER_S - 3.0

#: The share of a bounded sweep's budget the SIGTERM graces may use. The SIGKILL
#: that follows goes out by then at the latest, leaving the rest of the budget
#: for it to land and be confirmed.
_SWEEP_TERM_SHARE = 0.75

#: How often :data:`live_cli_registry` re-snapshots every live CLI's tree. The
#: registry's snapshot is what an abort or a terminal Ctrl-C sweeps, and both of
#: those signal the CLI at the same moment they signal this process — so by the
#: time a handler here runs, the CLI may already be dead and its children
#: reparented. One ``/proc`` scan serves every live CLI (~10 ms for ~1300
#: processes, measured 2026-09-28); a child started less than one interval
#: before the signal is the residual gap.
LIVE_SNAPSHOT_INTERVAL_S = 1.0


class Descendant(NamedTuple):
    """One process below a CLI, as it was when the snapshot was taken."""

    pid: int
    pgid: Optional[int]
    sid: Optional[int]
    #: The same string :func:`process_start_time` returns, so :func:`probe` can
    #: corroborate it later; ``None`` when the platform could not say.
    start_time: Optional[str]


class _ProcRow(NamedTuple):
    pid: int
    ppid: int
    pgid: Optional[int]
    sid: Optional[int]
    start_time: Optional[str]
    #: ``/proc`` state letter; ``Z`` is a zombie (dead, not yet reaped).
    #: ``None`` where ``ps`` built the table.
    state: Optional[str] = None


def _linux_process_table() -> Optional[Dict[int, _ProcRow]]:
    """Every process from one pass over ``/proc/*/stat``, or ``None`` without /proc."""
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    if "self" not in names:
        return None
    table: Dict[int, _ProcRow] = {}
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as handle:
                raw = handle.read()
        except OSError:
            continue  # exited between listdir and open
        # comm (field 2) is parenthesised and may itself contain ')'.
        close = raw.rfind(b")")
        fields = raw[close + 2:].split() if close != -1 else []
        if len(fields) <= 19:
            continue
        try:
            # After comm: state(3) ppid(4) pgrp(5) session(6) … starttime(22).
            table[int(name)] = _ProcRow(
                pid=int(name),
                ppid=int(fields[1]),
                pgid=int(fields[2]),
                sid=int(fields[3]),
                start_time=fields[19].decode("ascii", "replace"),
                state=fields[0].decode("ascii", "replace"),
            )
        except ValueError:
            continue
    return table


#: The ``ps`` columns :func:`_ps_process_table` asks for, each as its own ``-o``
#: (POSIX reads everything after a ``name=`` as that column's header, commas
#: included). ``lstart`` is last because it is several words.
_PS_TABLE_COLUMNS = ("pid", "ppid", "pgid", "lstart")


def _ps_process_table() -> Dict[int, _ProcRow]:
    """The ``ps`` walk for POSIX without /proc (macOS, the BSDs).

    ONE ``ps -A`` gives the tree and every start time. :data:`live_cli_registry`
    re-walks this table every :data:`LIVE_SNAPSHOT_INTERVAL_S` for as long as a
    CLI runs, so nothing here may cost a spawn per process: the start time used
    to come from a ``ps -o lstart= -p <pid>`` per descendant, every second, for
    the whole of a fan-out. ``lstart`` is the column :func:`_ps_start_time`
    reads, so a row's start time is the string :func:`process_start_time`
    returns for that pid — its internal spacing kept, since ``%e`` pads a
    one-digit day — and :func:`probe` can corroborate it at sweep time. The
    session id is left to :func:`_complete_row` (``os.getsid``, a system call,
    not a spawn), because ``ps`` spells it differently on every platform. A
    ``ps`` that rejects ``lstart`` still yields the tree, undated.
    """
    for columns in (_PS_TABLE_COLUMNS, _PS_TABLE_COLUMNS[:3]):
        argv = ["ps", "-A"]
        for column in columns:
            argv += ["-o", f"{column}="]
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if result.returncode == 0:
            return _parse_ps_table(result.stdout)
    return {}


def _parse_ps_table(stdout: str) -> Dict[int, _ProcRow]:
    table: Dict[int, _ProcRow] = {}
    for line in stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid, ppid, pgid = (int(part) for part in parts[:3])
        except ValueError:
            continue
        start = parts[3].strip() if len(parts) > 3 else ""
        table[pid] = _ProcRow(pid, ppid, pgid, None, start or None)
    return table


def _process_table() -> Dict[int, _ProcRow]:
    if IS_WINDOWS:  # pragma: no cover - Windows
        return {}
    table = _linux_process_table()
    if table is None:
        table = _ps_process_table()
    return table


def _complete_row(row: _ProcRow) -> Descendant:
    # Both tables date their rows, so this per-pid start-time probe (a spawn off
    # Linux) is reached only for a row the table could not date.
    sid = row.sid
    if sid is None:
        try:
            sid = os.getsid(row.pid)
        except (OSError, AttributeError):
            sid = None
    start = row.start_time if row.start_time is not None else process_start_time(row.pid)
    return Descendant(row.pid, row.pgid, sid, start)


def _descendants_in(table: Dict[int, _ProcRow], pid: int) -> List[Descendant]:
    children: Dict[int, List[int]] = {}
    for row in table.values():
        children.setdefault(row.ppid, []).append(row.pid)
    found: List[Descendant] = []
    seen = {pid}
    queue = sorted(children.get(pid, ()))
    while queue:
        child = queue.pop(0)
        if child in seen or child not in table:
            continue
        seen.add(child)
        found.append(_complete_row(table[child]))
        queue.extend(sorted(children.get(child, ())))
    return found


def descendants(pid: int) -> List[Descendant]:
    """(pid, pgid, sid, start time) for every live descendant of *pid*.

    Linux reads ``/proc/*/stat`` once; other POSIX platforms run one ``ps -A``
    for pid, ppid, pgid and ``lstart``; Windows returns an empty list (its tree
    is a Job Object, torn down by the worker's own job, D4d). Breadth-first,
    *pid* itself excluded. The walk follows parent links, which is exactly why
    it must run **before** the CLI dies: afterwards its children belong to init.
    """
    if IS_WINDOWS or not pid or pid <= 0:  # pragma: no cover - Windows leg
        return []
    return _descendants_in(_process_table(), int(pid))


def _merge_snapshots(*snapshots: Sequence[Descendant]) -> List[Descendant]:
    """One entry per (pid, start time), first-seen order."""
    merged: List[Descendant] = []
    seen = set()
    for snapshot in snapshots:
        for entry in snapshot or ():
            key = (entry.pid, entry.start_time)
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    return merged


@dataclass
class TreeSweep:
    """What :func:`terminate_timed_out` did, for the log line that names it.

    ``cli`` is the CLI's own verdict: ``exited`` (already gone, nothing sent),
    ``terminated`` (exited within the grace after SIGTERM), ``killed`` (needed
    the SIGKILL) or ``survived``. ``groups`` maps each snapshotted process group
    to its verdict, in :func:`kill_tree`'s vocabulary (``gone``, ``killed``,
    ``survived``, ``skipped``), in the order the snapshot met them.
    """

    cli: str
    groups: Dict[int, str] = field(default_factory=dict)

    @property
    def survived(self) -> List[int]:
        return [pgid for pgid, verdict in self.groups.items() if verdict == "survived"]

    def describe(self) -> str:
        if not self.groups:
            return f"cli {self.cli}; no descendant process groups"
        verdicts = ", ".join(f"{pgid} {verdict}" for pgid, verdict in self.groups.items())
        line = f"cli {self.cli}; process groups: {verdicts}"
        if self.survived:
            line += (" — SURVIVED: " + ", ".join(str(pgid) for pgid in self.survived)
                     + " still running")
        return line


def _exited(proc: Any) -> bool:
    try:
        return proc.poll() is not None
    except OSError:  # pragma: no cover - reaped behind Popen's back
        return True


def _wait_exit(proc: Any, timeout_s: float, poll_interval_s: float) -> bool:
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        if _exited(proc):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval_s)


def _signal_cli(proc: Any) -> bool:
    """SIGTERM the CLI alone (never its group: that group is this process's).

    ``False`` when it had already exited, so nothing was sent.
    """
    try:
        if proc.poll() is not None:
            return False
        proc.terminate()
    except OSError:
        return False
    return True


def _await_cli(proc: Any, signalled: bool, grace_s: float, poll_interval_s: float) -> str:
    if not signalled:
        return "exited"
    if _wait_exit(proc, grace_s, poll_interval_s):
        return "terminated"
    try:
        proc.kill()
    except OSError:
        pass
    return "killed" if _wait_exit(proc, _KILL_CONFIRM_S, poll_interval_s) else "survived"


def _worst(verdicts: Sequence[str]) -> str:
    for verdict in ("survived", "killed", "skipped"):
        if verdict in verdicts:
            return verdict
    return "gone"


def _linux_state(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    close = raw.rfind(b")")
    fields = raw[close + 2:].split() if close != -1 else []
    return fields[0].decode("ascii", "replace") if fields else None


def _running(pid: int) -> bool:
    """Alive and not a zombie (a zombie keeps its pid and group, runs nothing)."""
    return process_alive(pid) and _linux_state(pid) != "Z"


def _group_running(pgid: int) -> bool:
    """Does any process that is not a zombie still belong to *pgid*?

    ``killpg(pgid, 0)`` succeeds for a group of zombies too, so on Linux the
    answer comes from ``/proc``; elsewhere membership is the best available.
    """
    if not _group_has_members(pgid):
        return False
    table = _linux_process_table()
    if table is None:  # pragma: no cover - non-Linux POSIX
        return True
    return any(row.pgid == pgid and row.state != "Z" for row in table.values())


class _SweepClock:
    """When a sweep's SIGTERM graces must end, and when the sweep itself must.

    Unbounded (``budget_s=None``) is the timeout path's clock: every grace runs
    in full and a SIGKILL gets :data:`_KILL_CONFIRM_S` to land. Bounded, every
    wait is clipped — the graces to :data:`_SWEEP_TERM_SHARE` of the budget, so
    the SIGKILLs still go out with time to land, and the confirmations to the
    budget itself.
    """

    def __init__(self, budget_s: Optional[float] = None):
        now = time.monotonic()
        self.term_by: Optional[float] = None
        self.deadline: Optional[float] = None
        if budget_s is not None:
            budget = max(float(budget_s), 0.0)
            self.term_by = now + budget * _SWEEP_TERM_SHARE
            self.deadline = now + budget

    def grace_until(self, grace_s: float) -> float:
        end = time.monotonic() + max(grace_s, 0.0)
        return end if self.term_by is None else min(end, self.term_by)

    def confirm_until(self) -> float:
        end = time.monotonic() + _KILL_CONFIRM_S
        return end if self.deadline is None else min(end, self.deadline)


class _Liveness:
    """One look at what still runs, shared by every group a poll checks.

    One ``/proc`` scan per poll however many groups are waited on; off Linux
    each question falls back to its own per-pid probe. "Running" never counts a
    zombie: an unreaped leader still answers ``kill(pid, 0)``, and a wait that
    trusted that would sit out its whole window on a process already dead.
    """

    def __init__(self) -> None:
        try:
            self.table: Optional[Dict[int, _ProcRow]] = _linux_process_table()
        except Exception:  # noqa: BLE001 - the per-pid probes still answer
            self.table = None
        self._groups = None if self.table is None else {
            row.pgid for row in self.table.values() if row.state != "Z"
        }

    def pid_running(self, pid: int) -> bool:
        """Running, whoever it is now."""
        if self.table is None:
            return _running(pid)
        row = self.table.get(pid)
        return row is not None and row.state != "Z"

    def corroborated(self, member: Descendant) -> bool:
        """Running, and still the process the snapshot recorded (start time)."""
        if self.table is None:
            return probe(member.pid, member.start_time) == ALIVE and _running(member.pid)
        row = self.table.get(member.pid)
        return (row is not None and row.state != "Z" and bool(member.start_time)
                and str(row.start_time) == str(member.start_time))

    def group_running(self, pgid: int) -> bool:
        if self._groups is None:
            return _group_running(pgid)
        return pgid in self._groups

    def pgid_of(self, pid: int) -> Optional[int]:
        """The group *pid* is in now, or ``None`` when that cannot be read."""
        if self.table is None:
            try:
                return os.getpgid(pid)
            except (OSError, AttributeError):
                return None
        row = self.table.get(pid)
        return row.pgid if row is not None else None


@dataclass
class _GroupTarget:
    """One snapshotted process group's part in :func:`_sweep_snapshots`.

    A member is *corroborated* when its pid is running (not a zombie) with the
    start time the snapshot recorded. None corroborated: ``skipped`` (no signal
    at all) if a pid is running but is not the process the snapshot saw — a
    recycled pid's group is an unrelated group — else ``gone``: every member is
    dead, or dead and not yet reaped, which is the same thing to a signal.

    A signalled group's corroborated members are also watched one by one
    (``watched``), because the group a member was snapshotted in need not be
    the group it is in now: GNU ``timeout`` calls ``setpgid`` right after exec,
    and the registry's snapshot that an abort or an outside kill sweeps can be
    up to :data:`LIVE_SNAPSHOT_INTERVAL_S` old. A signal to the recorded group
    never reaches such a member, so it is stopped by pid and start time the way
    :func:`kill_tree` stops a pid (its live group, never ours), and its verdict
    folds into the group's: a recorded group swept empty while a member runs on
    elsewhere is ``survived``, never ``killed``.
    """

    key: int
    members: List[Descendant]
    #: The group signalled whole, or ``None`` when each corroborated member is
    #: stopped on its own (``alone``): a child that stayed in the CLI's group
    #: stayed in OURS, which is never signalled (a self-kill) — and
    #: :func:`_resolve_pgid` resolves no group for it for the same reason.
    pgid: Optional[int] = None
    leader: Optional[Descendant] = None
    alone: List[Descendant] = field(default_factory=list)
    #: The corroborated members of a group signalled whole (see above).
    watched: List[Descendant] = field(default_factory=list)
    verdict: Optional[str] = None

    def terminate(self, look: _Liveness, own: Optional[int]) -> None:
        """Corroborate, then SIGTERM — or settle a verdict that owes no signal."""
        corroborated = [m for m in self.members if look.corroborated(m)]
        if not corroborated:
            running = any(look.pid_running(m.pid) for m in self.members)
            self.verdict = "skipped" if running else "gone"
            return
        if self.key > 1 and self.key != own:
            self.pgid = self.key
            self.watched = corroborated
            self.leader = next((m for m in corroborated if m.pid == self.key),
                               corroborated[0])
            _signal_tree(self.leader.pid, self.pgid, signal.SIGTERM)
            for member in self.watched:
                if look.pgid_of(member.pid) != self.pgid:
                    _signal_tree(member.pid, None, signal.SIGTERM)
            return
        self.alone = corroborated
        for member in self.alone:
            _signal_tree(member.pid, None, signal.SIGTERM)

    def _group_or_watched_running(self, look: _Liveness, pgid: int) -> bool:
        """The group, else any watched member that runs outside it.

        The members are asked only once the group is empty, so off Linux a
        group that still runs costs no per-pid probe on each poll.
        """
        if look.group_running(pgid):
            return True
        return any(look.corroborated(m) for m in self.watched)

    def running(self, look: _Liveness) -> bool:
        try:
            if self.pgid is not None:
                return self._group_or_watched_running(look, self.pgid)
            return any(look.corroborated(m) for m in self.alone)
        except Exception:  # noqa: BLE001 - unknown is treated as still running
            return True

    def kill(self, look: _Liveness) -> None:
        """SIGKILL whatever still runs, sending nothing to a pid that moved on.

        The whole group goes, not just the member the grace watched: a member
        that shrugged off the SIGTERM its leader died of still gets it. So does
        a watched member that runs outside the group, by pid and start time.
        """
        if self.pgid is not None:
            if look.group_running(self.pgid):
                leader = self.leader if self.leader and look.corroborated(self.leader) else None
                _signal_tree(leader.pid if leader else None, self.pgid, signal.SIGKILL)
            for member in self.watched:
                if look.pgid_of(member.pid) != self.pgid and look.corroborated(member):
                    _signal_tree(member.pid, None, signal.SIGKILL)
            return
        for member in self.alone:
            if look.corroborated(member):
                _signal_tree(member.pid, None, signal.SIGKILL)

    def settle(self, look: _Liveness) -> str:
        if self.verdict is None:
            if self.pgid is not None:
                self.verdict = ("survived" if self._group_or_watched_running(look, self.pgid)
                                else "killed")
            else:
                self.verdict = _worst(["survived" if look.corroborated(m) else "killed"
                                       for m in self.alone])
        return self.verdict

    def fallback(self) -> str:
        """The verdict when one of this group's own steps raised."""
        return "survived" if any(
            probe(m.pid, m.start_time) == ALIVE for m in self.members
        ) else "gone"


def _wait_targets(
    targets: Sequence[_GroupTarget], until: float, poll_interval_s: float
) -> List[_GroupTarget]:
    """Poll until none of *targets* runs or *until* passes; return those still running."""
    still = list(targets)
    while still:
        look = _Liveness()
        still = [target for target in still if target.running(look)]
        now = time.monotonic()
        if not still or now >= until:
            break
        time.sleep(max(0.0, min(poll_interval_s, until - now)))
    return still


def _sweep_snapshots(
    snapshots: Sequence[Sequence[Descendant]],
    grace_s: float,
    poll_interval_s: float,
    clock: _SweepClock,
) -> List[Dict[int, str]]:
    """Stop every group of every snapshot on one shared clock.

    Every group gets its SIGTERM before any is waited on; one grace covers them
    all; whatever still runs gets its SIGKILL in one pass; one confirmation
    window covers those. So a sweep costs one grace plus one confirmation however
    many groups there are. Swept one at a time, each group paid both — and a
    zombie leader could make one pay its confirmation in full — so an abort with
    enough detached groups ran past the SIGKILL that ends the worker and left the
    groups it had not reached yet running. Every wait is clipped by *clock*.
    Returns one ``{group: verdict}`` map per snapshot, in order.
    """
    try:
        own: Optional[int] = os.getpgid(0)
    except (OSError, AttributeError):  # pragma: no cover - Windows; never in practice
        own = None
    plans: List[List[_GroupTarget]] = []
    for snapshot in snapshots:
        by_group: Dict[int, List[Descendant]] = {}
        for entry in snapshot or ():
            key = entry.pgid if entry.pgid is not None else -entry.pid
            by_group.setdefault(key, []).append(entry)
        plans.append([_GroupTarget(key, members) for key, members in by_group.items()])
    targets = [target for plan in plans for target in plan]
    if targets:
        look = _Liveness()
        for target in targets:
            try:
                target.terminate(look, own)
            except Exception:  # noqa: BLE001 - one group must not stop the sweep
                target.verdict = target.fallback()
        live = [target for target in targets if target.verdict is None]
        live = _wait_targets(live, clock.grace_until(grace_s), poll_interval_s)
        if live:
            look = _Liveness()
            for target in live:
                try:
                    target.kill(look)
                except Exception:  # noqa: BLE001 - one group must not stop the sweep
                    pass
            _wait_targets(live, clock.confirm_until(), poll_interval_s)
        look = _Liveness()
        for target in targets:
            try:
                target.settle(look)
            except Exception:  # noqa: BLE001 - one group must not stop the sweep
                target.verdict = target.fallback()
    return [{target.key: str(target.verdict) for target in plan} for plan in plans]


def _sweep_groups(
    snapshot: Sequence[Descendant], grace_s: float, poll_interval_s: float
) -> Dict[int, str]:
    return _sweep_snapshots([snapshot], grace_s, poll_interval_s, _SweepClock())[0]


def terminate_timed_out(
    proc: Any,
    snapshot: Sequence[Descendant],
    grace_s: float = DEFAULT_TERMINATE_GRACE_S,
    *,
    poll_interval_s: float = 0.05,
) -> TreeSweep:
    """Stop a Lane A CLI and every process group its snapshot recorded.

    *proc* is the CLI's ``subprocess.Popen`` (``poll``/``terminate``/``kill``);
    *snapshot* is :func:`descendants` of it, taken **before** anything was
    signalled. SIGTERM to the CLI alone (it shares this process's group), the
    grace, SIGKILL if it is still alive; then every distinct snapshotted group
    together — SIGTERM, one shared *grace_s*, SIGKILL (:func:`_sweep_snapshots`)
    — under :func:`kill_tree`'s identity rule, each member's recorded start time
    re-checked, so a pid recycled since the snapshot is ``skipped`` and never
    signalled. Several groups, not one: pi's shell and a ``timeout`` inside it
    each lead their own. The Ctrl-C and abort paths run the same steps for every
    live CLI at once (:meth:`LiveCliRegistry.sweep`, decision 11.9). Never raises
    for a process that is already gone.
    """
    signalled = _signal_cli(proc)
    cli = _await_cli(proc, signalled, grace_s, poll_interval_s)
    return TreeSweep(cli=cli, groups=_sweep_groups(snapshot, grace_s, poll_interval_s))


@dataclass
class LiveCli:
    """One Lane A CLI this process is waiting on."""

    pid: int
    start_time: Optional[str]
    proc: Any
    snapshot: List[Descendant] = field(default_factory=list)


class LiveCliRegistry:
    """Every live Lane A CLI in this process, with its latest tree snapshot.

    Entries are added right before ``communicate()`` and removed in its
    ``finally``; a daemon thread re-snapshots every entry's descendants each
    :data:`LIVE_SNAPSHOT_INTERVAL_S` from ONE process-table scan, and exits when
    the registry is empty. The detached worker's SIGTERM handler calls
    :meth:`sweep` before it exits, so an abort takes the CLIs' detached children
    with it instead of leaving them to ``kill_tree`` on the worker's group, which
    never reaches a child in a session of its own.

    Safe to read from a signal handler: the lock is re-entrant (the handler runs
    on the main thread, which may already hold it) and never held across a scan.
    """

    def __init__(self, interval_s: float = LIVE_SNAPSHOT_INTERVAL_S):
        self.interval_s = interval_s
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._entries: Dict[int, LiveCli] = {}
        self._thread: Optional[threading.Thread] = None

    def register(self, proc: Any) -> LiveCli:
        entry = LiveCli(pid=int(proc.pid), start_time=process_start_time(proc.pid),
                        proc=proc)
        with self._lock:
            self._entries[id(proc)] = entry
            if not IS_WINDOWS and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(
                    target=self._refresh_loop, name="multi-llm-cli-tree-snapshot",
                    daemon=True,
                )
                self._thread.start()
        return entry

    def unregister(self, proc: Any) -> None:
        with self._lock:
            self._entries.pop(id(proc), None)
            self._wake.notify_all()

    @contextmanager
    def tracking(self, proc: Any) -> Iterator[LiveCli]:
        entry = self.register(proc)
        try:
            yield entry
        finally:
            self.unregister(proc)

    def entries(self) -> List[LiveCli]:
        with self._lock:
            return list(self._entries.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def refresh(self) -> None:
        """Re-snapshot every entry from one scan. Identity-checked per entry."""
        entries = self.entries()
        if not entries:
            return
        table = _process_table()
        for entry in entries:
            row = table.get(entry.pid)
            if row is None:
                continue  # already gone: keep the last snapshot it had
            if entry.start_time and row.start_time and row.start_time != entry.start_time:
                continue  # the pid names someone else now
            fresh = _descendants_in(table, entry.pid)
            with self._lock:
                if id(entry.proc) in self._entries:
                    entry.snapshot = fresh

    def snapshot_for(self, proc: Any) -> List[Descendant]:
        """The last refreshed snapshot of *proc*'s tree, merged with a fresh one.

        The fresh walk runs only while *proc* is un-reaped (``returncode`` still
        ``None``): until then its pid cannot be recycled, so its children are
        still its own. Afterwards only the stored snapshot, whose start times
        protect every later signal, is used.
        """
        with self._lock:
            entry = self._entries.get(id(proc))
            stored = list(entry.snapshot) if entry else []
        fresh: List[Descendant] = []
        if getattr(proc, "returncode", 0) is None:
            fresh = descendants(proc.pid)
        return _merge_snapshots(fresh, stored)

    def sweep(
        self,
        grace_s: float = DEFAULT_TERMINATE_GRACE_S,
        *,
        poll_interval_s: float = 0.05,
        budget_s: Optional[float] = LIVE_SWEEP_BUDGET_S,
    ) -> List[Tuple[LiveCli, TreeSweep]]:
        """Terminate every registered CLI and sweep its snapshotted groups, on one clock.

        The abort that calls this is itself counting down to the SIGKILL that
        ends the worker, and a signal this sweep has not sent by then is never
        sent — so nothing here is paid per CLI or per group. All CLIs are
        signalled first and share one grace; the survivors are SIGKILLed in one
        pass; then every group of every snapshot is signalled before any is
        waited on, and they share one grace and one SIGKILL window
        (:func:`_sweep_snapshots`). Every wait is clipped to *budget_s*
        (:data:`LIVE_SWEEP_BUDGET_S`, inside the abort's window), the graces
        early enough that the SIGKILLs still go out within it; ``None`` leaves
        the sweep unbounded.
        """
        clock = _SweepClock(budget_s)
        entries = self.entries()
        snapshots = [self.snapshot_for(entry.proc) for entry in entries]
        signalled = [_signal_cli(entry.proc) for entry in entries]
        waiting = [entry.proc for entry, sent in zip(entries, signalled) if sent]
        until = clock.grace_until(grace_s)
        for proc in waiting:
            _wait_exit(proc, until - time.monotonic(), poll_interval_s)
        killed: List[Any] = []
        for proc in waiting:
            if _exited(proc):
                continue
            try:
                proc.kill()
            except OSError:
                pass
            killed.append(proc)
        groups = _sweep_snapshots(snapshots, grace_s, poll_interval_s, clock)
        until = clock.confirm_until()
        for proc in killed:
            _wait_exit(proc, until - time.monotonic(), poll_interval_s)
        results: List[Tuple[LiveCli, TreeSweep]] = []
        for entry, sent, swept in zip(entries, signalled, groups):
            if not sent:
                cli = "exited"
            elif not any(proc is entry.proc for proc in killed):
                cli = "terminated"
            else:
                cli = "killed" if _exited(entry.proc) else "survived"
            results.append((entry, TreeSweep(cli=cli, groups=swept)))
        return results

    def _refresh_loop(self) -> None:
        while True:
            with self._lock:
                if not self._entries:
                    self._thread = None
                    return
                self._wake.wait(self.interval_s)
                if not self._entries:
                    self._thread = None
                    return
            try:
                self.refresh()
            except Exception:  # noqa: BLE001 - a snapshot is best-effort
                pass


#: The process-wide registry every Lane A call registers its CLI in.
live_cli_registry = LiveCliRegistry()


def sweep_live_clis(tag: str) -> None:
    """Sweep :data:`live_cli_registry`, one ``[<tag>] cli pid N: …`` stderr line per CLI.

    A no-op when nothing is registered. Never raises: the interrupt that asked
    for it must still end the run and its result line must still be written.
    """
    try:
        swept = live_cli_registry.sweep()
    except Exception as exc:  # noqa: BLE001 - see above
        print(f"[{tag}] failed: {exc}", file=sys.stderr)
        return
    for entry, result in swept:
        print(f"[{tag}] cli pid {entry.pid}: {result.describe()}", file=sys.stderr)


async def sweeping_live_clis_on_interrupt(awaitable: Awaitable[_T]) -> _T:
    """Await *awaitable*; if a cancellation or Ctrl-C ends it, sweep the registry first.

    What ``asyncio.run`` is handed wherever a dispatch target's fan-out runs
    (decision 11.9). A slot waits on its CLI from an ``asyncio.to_thread`` worker,
    and ``asyncio.run`` raises its ``KeyboardInterrupt`` only after the runner
    has shut the default executor down, which joins every such worker. So a
    sweep left to the caller's ``except KeyboardInterrupt`` waits for each CLI to
    reach its own deadline whenever the SIGINT reached the orchestrator alone,
    or a CLI handled it without exiting. Here it runs on the way out of the
    coroutine, while the runner is still cancelling the main task: the CLIs are
    stopped, their workers return, and the executor shutdown that follows has
    nothing left to wait for.

    Both endings are caught: the first Ctrl-C arrives as the main task's
    cancellation (3.11+'s runner), a later one — or any Ctrl-C on 3.10 — as a
    ``KeyboardInterrupt`` raised wherever the main thread is, after which the
    runner's cancel-all reaches this frame as a cancellation too.
    """
    try:
        return await awaitable
    except (asyncio.CancelledError, KeyboardInterrupt):
        sweep_live_clis("INTERRUPT_SWEEP")
        raise


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
    (``setsid``) and every inherited descriptor (stdin to ``/dev/null``,
    stdout/stderr to *log_path*, everything above them closed outright by
    :func:`_close_inherited_fds`, so the host's stdout pipe sees EOF at once);
    the second cuts the parent, so the worker reparents to ``init`` and this
    process returns with no child to wait on and no zombie for the host to reap.

    ``log_path`` is handed in as the child's stdout/stderr on **both** platforms,
    and that is the point: the worker prints its setup lines (inputs loaded,
    models resolved, backup written, "no user preferences detected") long before
    it reaches ``maybe_detach`` and calls :func:`enter_detached_worker`, so a
    worker that only reopened its own log there sent every one of them to
    ``/dev/null`` — and with them the only durable record of what a detached run
    decided and on what inputs. It also means a worker that dies *before* the
    handshake (a bad re-exec, a failed ``uv`` sync) leaves its reason in the log
    the launcher's failure message tails, rather than an empty file.

    Windows: one spawn. ``DETACHED_PROCESS`` already removes the console, so
    there is no intermediate and nothing to reap — but a console-less child has no
    valid stdout, so the launcher opens the log and hands that one handle in as
    both stdout and stderr.
    """
    argv = [str(a) for a in argv]
    if IS_WINDOWS:  # pragma: no cover - Windows
        return _spawn_detached_windows(argv, cwd=cwd, env=env, log_path=log_path)
    return _spawn_detached_posix(argv, cwd=cwd, env=env, log_path=log_path)


#: Bounds for :func:`_fd_sweep_limit`, which only runs where the kernel does not
#: publish the open descriptors as a directory (see :func:`_close_inherited_fds`).
_FD_SWEEP_FALLBACK = 4096
_FD_SWEEP_CEILING = 1 << 16


def _fd_sweep_limit() -> int:
    """Exclusive upper bound for a blind ``closerange`` sweep.

    The soft ``RLIMIT_NOFILE`` is the number the kernel enforced when every
    descriptor this process holds was opened, so nothing can sit above it;
    ``SC_OPEN_MAX`` is the fallback. Both are clamped, because "unlimited" is a
    legal answer (``RLIM_INFINITY``, and on Linux a soft limit of 2^20 is a
    common container default) and a sweep to infinity is not a sweep:
    :func:`os.closerange` is one ``close_range`` syscall where the platform has
    it, but degrades to a per-descriptor loop where it does not.
    """
    limit = 0
    try:  # POSIX-only, and absent from some locked-down builds.
        import resource

        for candidate in resource.getrlimit(resource.RLIMIT_NOFILE):
            # RLIM_INFINITY is -1 here, so "unlimited" fails the > 0 test too.
            if candidate is not None and candidate > 0:
                limit = int(candidate)
                break
    except Exception:
        limit = 0
    if limit <= 0:
        try:
            limit = int(os.sysconf("SC_OPEN_MAX"))
        except (AttributeError, ValueError, OSError):
            limit = 0
    if limit <= 0:
        limit = _FD_SWEEP_FALLBACK
    return min(max(limit, 3), _FD_SWEEP_CEILING)


def _close_inherited_fds() -> None:
    """Close every descriptor above stdio. Call in a forked child, pre-``exec``.

    D4's spawn recipe requires the worker to hold *nothing* belonging to the
    host: fds 0/1/2 are replaced, and everything above them has to go, or the
    host's pipe never sees EOF and a harness waiting on the invoking command
    blocks on a run that intends to outlive it. PEP 446 covers only descriptors
    *Python* opened — it makes those non-inheritable, so ``execve`` drops them —
    but a descriptor the invoker itself **inherited** from the host (a harness
    status pipe on fd 3, a shell's redirection) arrives already inheritable and
    would survive into the worker untouched. ``close_fds=True`` would cover it if
    this were a :class:`subprocess.Popen`, but the double spawn is a raw
    ``fork``/``execve``, so the sweep is ours to do.

    Where the kernel publishes the open set (``/proc/self/fd``, ``/dev/fd``) that
    is both exact and cheap; otherwise fall back to a blind
    :func:`os.closerange` over :func:`_fd_sweep_limit`. Errors are ignored
    throughout: a descriptor that cannot be closed is one already closed.
    """
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            # `listdir` reads the whole directory and closes its own descriptor
            # before returning, so nothing here is iterating over a handle it is
            # about to invalidate.
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for name in names:
            try:
                fd = int(name)
            except ValueError:  # pragma: no cover - non-numeric entry
                continue
            if fd < 3:
                continue
            try:
                os.close(fd)
            except OSError:
                pass
        return
    os.closerange(3, _fd_sweep_limit())  # pragma: no cover - no fd directory


def _open_worker_log(log_path: Optional[str]) -> Optional[int]:
    """Append-only descriptor for the worker's stdout/stderr, or ``None``.

    Append, never truncate: the launcher rotated any predecessor aside already,
    and a worker that re-enters :func:`enter_detached_worker` must find its own
    earlier bytes still there. Best effort — a log that cannot be opened falls
    back to ``/dev/null`` (the pre-2026-09 behaviour) rather than sinking a
    launch whose phase dir was already proved writable.
    """
    if not log_path:
        return None
    try:
        return os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    except OSError:
        return None


def _spawn_detached_posix(
    argv: List[str], *, cwd: str, env: Dict[str, str], log_path: Optional[str]
) -> int:
    # Opened here, in the invoker, so the descriptor exists before the fork and
    # the child has nothing to fail at: it only dup2s what it was handed.
    log_fd = _open_worker_log(log_path)
    read_fd, write_fd = os.pipe()
    try:
        intermediate = os.fork()
    except OSError:
        for spare in (log_fd, read_fd, write_fd):
            if spare is not None:
                try:
                    os.close(spare)
                except OSError:
                    pass
        raise
    if intermediate == 0:
        # --- spawn one: new session, no controlling terminal, no host fds ---
        try:
            os.close(read_fd)
            os.setsid()
            devnull = os.open(os.devnull, os.O_RDWR)
            # stdout/stderr onto the phase log from the first byte the worker
            # writes — not from `enter_detached_worker`, which the orchestrators
            # reach only after their setup output is already on the floor.
            # Duplicated before stdin, so that a log that landed on a low
            # descriptor (an invoker whose own stdin was closed) is copied up to
            # 1 and 2 before /dev/null overwrites 0.
            out_fd = devnull if log_fd is None else log_fd
            os.dup2(out_fd, 1)
            os.dup2(out_fd, 2)
            os.dup2(devnull, 0)
            for spare in (devnull, log_fd):
                if spare is not None and spare > 2:
                    try:
                        os.close(spare)
                    except OSError:
                        pass
            worker = os.fork()
            if worker == 0:
                # --- spawn two: the real worker, reparented to init on exit ---
                try:
                    os.close(write_fd)
                except OSError:
                    pass
                # Everything above the three fds replaced above goes with it —
                # including whatever the *invoker* inherited from the host, which
                # `execve` alone would carry into the worker and leave the host's
                # pipe waiting on an EOF that never comes.
                _close_inherited_fds()
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
    # The log belongs to the worker now; holding a second descriptor open here
    # would keep the invoker attached to a file it never writes.
    if log_fd is not None:
        try:
            os.close(log_fd)
        except OSError:
            pass
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


def _stdio_already_on(log_path: str) -> bool:
    """True when fds 1 **and** 2 already name the file at *log_path*."""
    try:
        target = os.stat(str(log_path))
    except OSError:
        return False
    for fd in (1, 2):
        try:
            current = os.fstat(fd)
        except OSError:
            return False
        if (current.st_dev, current.st_ino) != (target.st_dev, target.st_ino):
            return False
    return True


def enter_detached_worker(log_path: Optional[str]) -> None:
    """Worker-side setup: own the log, own the process group, own the tree.

    Idempotent with the launcher, and that is now the normal case: since the
    POSIX spawn hands the log in as the child's stdout/stderr (so the setup lines
    printed *before* the detach boundary are recorded too), fds 1 and 2 usually
    already name it and there is nothing to reopen. Reopening anyway would be
    worse than redundant — a second file description means a second buffer over
    one file — so the descriptors are compared by ``(st_dev, st_ino)`` first.
    Nothing here ever truncates: the open is ``O_APPEND``, so a second call
    cannot drop the bytes the first one wrote.

    Where the comparison fails (a worker launched some other way, or one whose
    log was rotated out from under it) POSIX reopens fds 1 and 2 onto *log_path*,
    so no application output — the worker's own or any provider subprocess's —
    passes through a descriptor the host handed us. Either way the stream
    bootstrap is re-run so the log gets line-buffered UTF-8 with
    ``errors="replace"``. **Both** settings matter: without line buffering a
    detached log stays empty for minutes, and without the encoding a
    Windows-piped stream defaults to cp1252 and raises ``UnicodeEncodeError`` on
    the first em dash a model emits.

    Windows: the launcher already handed the log in as stdout/stderr, so there is
    nothing to reopen; instead assign this process to a Job Object with
    ``KILL_ON_JOB_CLOSE`` so the OS reaps provider children even if the worker
    crashes without running a handler.
    """
    if not IS_WINDOWS and log_path and not _stdio_already_on(log_path):
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
