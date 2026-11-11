"""Shell text that starts a command in a session of its own on every POSIX: ``SETSID``.

Several tests play a provider CLI that leaves a helper behind outside its run's
process group (a language server, a watcher), started from ``sh`` as
``setsid <command> &``. util-linux ships ``setsid(1)``; macOS does not, so on the
macOS runner the line failed (``sh: setsid: command not found``) and no helper
ever started, which those tests then read as the product's doing (the macOS CI
leg, 2026-11-11). ``SETSID`` does the same with the interpreter running the
suite, which every runner has. No fixtures.
"""

import shlex
import sys

#: ``setsid(2)``, then ``exec`` of the rest of the line: the command keeps the
#: pid the shell started (so ``$!`` names it) and leads a new session and group.
#: Unlike ``setsid(1)`` it never forks, so started as a group leader it fails
#: (``EPERM``) rather than leaving ``$!`` naming a parent that exits; a command
#: backgrounded by a non-interactive shell (no job control) never leads one.
_CODE = "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])"

#: Put before a command in shell text: ``f"{SETSID} sleep 30 &"``. ``-I -S``
#: keep the interpreter's start short and deaf to the environment: the session
#: is made a few milliseconds after the shell's ``&``, well inside the lanes'
#: drain, before anything of the run is stopped.
SETSID = f"{shlex.quote(sys.executable)} -I -S -c {shlex.quote(_CODE)}"
