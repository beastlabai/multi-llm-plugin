#!/usr/bin/env python3
"""Ending a bounded child and the process group, or session, it leads.

The one kill routine of the packaging scripts: the release gates' step runner
(:func:`common.gate.run_step`) and the opencode probes' bounded children
(``packaging/opencode/probe.py``). Stdlib only, and importable on its own: the
probe also runs from the copied kit inside the registry-check container
(``registry_check.KIT_FILES``), with nothing installed.

**POSIX.** The child leads its own process group (both callers start it in a
new session), so one ``killpg`` reaches every descendant that stayed in it.
:func:`stop` sends SIGTERM, waits, then SIGKILLs. ``session=True`` also signals
every other member of the leader's session (Linux ``/proc``): a child that
moved into a group of its own inside the session, as the opencode plugin
starts ``uv``, is out of reach of the group signal alone. Two wait strategies,
because the callers need different things from the grace:

* ``until="group"`` (the gate's steps): wait up to ``grace`` for the whole group
  to empty, reaping the leader as it goes so it cannot alone keep the group
  alive, and SIGKILL only what is still there. A probe interrupted by a step's
  bound uses that time to sweep its own children; a SIGKILL that landed as soon
  as the probe's leader exited would kill the sweeper mid-sweep.
* ``until="leader"`` (the probe's children): wait up to ``grace`` for the
  leader alone, then SIGKILL the rest at once. The probe's sweep bound
  (``probe.SWEEP_BOUND_S``) counts one grace per child, and a container
  without an init leaves orphaned zombies that a group poll would wait on.

**Windows** has no groups: the child alone is terminated and reaped. The gates
and probes are POSIX maintainer tools; there this only ends the children the
suite's own cases start, which spawn nothing.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List

#: How often ``until="group"`` re-checks the group.
POLL_S = 0.05


def session_members(sid: int) -> List[int]:
    """Every live pid in session ``sid`` (Linux ``/proc``; empty elsewhere)."""
    members: List[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return members
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            fields = text[text.rfind(")") + 2:].split()
            if int(fields[3]) == sid:
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def signal_group(pid: int, sig: int, *, session: bool = False) -> None:
    """``sig`` to the group ``pid`` leads, and with ``session`` to every other
    member of its session. POSIX only; a target already gone is not an error."""
    targets = session_members(pid) if session else []
    try:
        os.killpg(pid, sig)
    except OSError:
        pass
    for member in targets:
        try:
            os.kill(member, sig)
        except OSError:
            pass


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # macOS answers EPERM, not ESRCH, while the group's last members have
        # exited but their new parent (launchd) has not reaped them yet. They
        # are gone in moments; a member that really refuses the signal meets
        # the SIGKILL at the deadline.
        return True
    return True


def _reap(proc: "subprocess.Popen", grace: float) -> None:
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass


def stop(proc: "subprocess.Popen", *, grace: float, session: bool = False,
         until: str = "group") -> None:
    """SIGTERM ``proc``'s group (and session), up to ``grace`` for it to go,
    SIGKILL, then reap ``proc``. ``until`` is ``"group"`` or ``"leader"``; the
    module docstring says which caller needs which."""
    if until not in ("group", "leader"):
        raise ValueError(f"until must be 'group' or 'leader', not {until!r}")
    if os.name != "posix":
        try:
            proc.terminate()
        except OSError:
            pass
        _reap(proc, grace)
        return
    signal_group(proc.pid, signal.SIGTERM, session=session)
    if until == "leader":
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        signal_group(proc.pid, signal.SIGKILL, session=session)
    else:
        deadline = time.monotonic() + grace
        while True:
            proc.poll()  # Reap the leader so it cannot alone keep the group alive.
            if not _group_alive(proc.pid) and not (session and session_members(proc.pid)):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                signal_group(proc.pid, signal.SIGKILL, session=session)
                break
            time.sleep(min(POLL_S, remaining))
    _reap(proc, grace)
