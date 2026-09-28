"""A fake provider CLI that leaves pi's orphan shape behind (plan D3.3).

pi's bash tool spawns each shell ``detached: true`` — a session and process
group of its own — and GNU ``timeout`` inside that shell moves into yet another
group. A signal to the CLI, or to the CLI's group, reaches none of them, which
is how a killed review's ``timeout 1700 python -m pytest`` could outlive it.

:func:`install_fake_cli` writes an executable that reproduces exactly that:

* it prints one partial stdout line, so a test can see the partial output
  survive the kill;
* it starts ``sh -c 'timeout 60 sleep 60 | cat'`` with ``start_new_session``
  (the detached shell) — reading the shell through a pipe of its own, as pi
  does, or with ``CLI_TREE_SHELL_STDOUT=inherit`` letting ``cat`` hold the
  CLI's own stdout pipe (the worst case for a drain);
* it records every pid of that tree — the shell, ``timeout``, ``sleep`` and
  ``cat`` — in a marker file, and its own pid beside it, then hangs.

``CLI_TREE_ON_TERM`` shapes its SIGTERM: unset dies at once (Python's default),
``exit`` writes ``<marker>.term`` and exits 0 (a CLI with pi's own handler),
``ignore`` shrugs it off (so only the SIGKILL ends it), ``reap-shell`` SIGTERMs
the detached shell's group only, writes ``<marker>.term`` and exits 143 (pi's
real handler: its shell goes, the ``timeout`` group inside it does not).
``CLI_TREE_SELF_SIGNAL`` names a signal it sends itself
``CLI_TREE_SELF_SIGNAL_AFTER_S`` seconds after it is ready (a CLI a terminal
Ctrl-C or an abort already killed); under ``reap-shell`` that signal reaches the
handler instead of the default disposition.

It answers ``--help`` with pi 0.87.1's ``--tools`` line and ``--version`` with
``0.87.1``, so it can stand in for ``pi`` on PATH.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

#: The line the fake prints before it hangs; a test finds it in the envelope.
PARTIAL_LINE = '{"type":"session","version":3,"id":"cli-tree-partial-line"}'

_TREE_SH = """#!/bin/sh
# The detached shell: its own session and group (the CLI started it with
# start_new_session). `timeout` then takes a group of its own inside it.
echo "sh $$" >> "$1"
timeout 60 sh -c 'echo "timeout $PPID" >> "$0"; echo "sleep $$" >> "$0"; exec sleep 60' "$1" \\
  | sh -c 'echo "cat $$" >> "$0"; exec cat' "$1"
"""

_FAKE_CLI = """#!{python}
import os, signal, subprocess, sys, time

args = sys.argv[1:]
if "--help" in args:
    print("  --tools, -t <tools>            Comma-separated allowlist of tool names to enable")
    sys.exit(0)
if "--version" in args:
    print("0.87.1")
    sys.exit(0)

marker = os.environ["CLI_TREE_MARKER"]
with open(marker + ".cli", "w") as handle:
    handle.write(str(os.getpid()))

if os.environ.get("CLI_TREE_SHELL_STDOUT") == "inherit":
    # The worst case: the tree holds the CALLER's stdout pipe (via cat), so a
    # drain that waits for EOF waits for the tree.
    shell = subprocess.Popen(["sh", {tree_sh!r}, marker], start_new_session=True)
else:
    # pi's own shape: the bash tool reads its shell through a pipe of its own.
    shell = subprocess.Popen(["sh", {tree_sh!r}, marker], start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Installed AFTER the tree is spawned: an ignored signal is inherited across
# fork and exec, and the tree must keep the default SIGTERM a real shell has.
on_term = os.environ.get("CLI_TREE_ON_TERM", "")
if on_term == "exit":
    def _exit_on_term(signum, frame):
        with open(marker + ".term", "w") as handle:
            handle.write("SIGTERM")
        os._exit(0)
    signal.signal(signal.SIGTERM, _exit_on_term)
elif on_term == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
elif on_term == "reap-shell":
    def _reap_shell_on_term(signum, frame):
        # The shell leads its own session, so its pid is its group: sh and cat
        # go, while `timeout` (a group of its own) and its sleep do not.
        try:
            os.killpg(shell.pid, signal.SIGTERM)
        except OSError:
            pass
        with open(marker + ".term", "w") as handle:
            handle.write("reap-shell")
        os._exit(143)
    signal.signal(signal.SIGTERM, _reap_shell_on_term)
deadline = time.time() + 10
while time.time() < deadline:
    try:
        with open(marker) as handle:
            if len(handle.read().splitlines()) >= 4:
                break
    except FileNotFoundError:
        pass
    time.sleep(0.02)

sys.stdout.write({partial!r} + "\\n")
sys.stdout.flush()
with open(marker + ".ready", "w") as handle:
    handle.write("ready")

self_signal = os.environ.get("CLI_TREE_SELF_SIGNAL")
if self_signal:
    time.sleep(float(os.environ.get("CLI_TREE_SELF_SIGNAL_AFTER_S", "1.5")))
    if on_term != "reap-shell":
        signal.signal(getattr(signal, self_signal), signal.SIG_DFL)
    os.kill(os.getpid(), getattr(signal, self_signal))
while True:
    time.sleep(1)
"""


def install_fake_cli(directory: Path, name: str = "pi") -> Path:
    """Write the fake CLI as ``directory/<name>`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    tree_sh = directory / "cli-tree.sh"
    tree_sh.write_text(_TREE_SH, encoding="utf-8")
    fake = directory / name
    fake.write_text(
        _FAKE_CLI.format(python=sys.executable, tree_sh=str(tree_sh),
                         partial=PARTIAL_LINE),
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def tree_roles(marker: Path) -> Dict[str, int]:
    """``{"sh": pid, "timeout": pid, "sleep": pid, "cat": pid}`` as recorded.

    ``sh`` leads the detached shell's group (``cat`` is in it too) and
    ``timeout`` leads its own (``sleep`` is in it): the two groups a sweep must
    reach besides the CLI.
    """
    roles: Dict[str, int] = {}
    try:
        text = marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        return roles
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            roles[parts[0]] = int(parts[1])
    return roles


def tree_pids(marker: Path) -> List[int]:
    """The shell, ``timeout``, ``sleep`` and ``cat`` pids the fake recorded."""
    return list(tree_roles(marker).values())


def cli_pid(marker: Path) -> Optional[int]:
    try:
        return int(Path(str(marker) + ".cli").read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def ready(marker: Path) -> bool:
    return Path(str(marker) + ".ready").exists()


def running(pid: int) -> bool:
    """Alive and not a zombie (a zombie holds its pid but runs nothing)."""
    stat = Path(f"/proc/{pid}/stat")
    if Path("/proc/self").exists():
        try:
            state = stat.read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            return False
        return state != "Z"
    try:  # pragma: no cover - non-Linux POSIX
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def await_dead(pids: Iterable[int], timeout: float = 10.0) -> List[int]:
    """Poll until none of *pids* is running; return the ones still running."""
    pids = list(pids)
    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = [pid for pid in pids if running(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if running(pid)]


def await_ready(marker: Path, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ready(marker) and len(tree_pids(marker)) >= 4:
            return True
        time.sleep(0.02)
    return False


def reap_leftovers(marker: Path) -> None:
    """Test cleanup: SIGKILL whatever a failing test left running."""
    pids = tree_pids(marker)
    cli = cli_pid(marker)
    if cli:
        pids.append(cli)
    for pid in pids:
        if running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
