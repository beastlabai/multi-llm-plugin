"""Private config roots that adapters hand a provider CLI, one per spawn.

An adapter whose CLI writes state under the user's home (sessions, caches,
logs) points the CLI at a directory made here instead, so a fan-out never
writes into the user's own CLI state and two concurrent slots never share one
writable root. Each directory is:

* private: ``0700``, as ``mkdtemp`` makes it, asserted with ``chmod``;
* findable: under ``TMPDIR`` with the adapter's fixed prefix, so one a SIGKILL
  leaked (``atexit`` never ran) can be swept by name;
* owned: tracked by the process that created it, and removed by that process
  only, at :meth:`PrivateHomes.cleanup` or at exit. A root the user pinned
  through the adapter's variable is never created here, so never removed; a
  forked child forgets the directories its parent created instead of removing
  them.

This module writes nothing into a root; what lands there is the CLI's own
state, removed with the root.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import threading
from typing import List, Tuple


class PrivateHomes:
    """The private config roots one adapter creates, one per :meth:`create`.

    The adapter keeps one instance at module level and exposes its methods
    under the module names callers use (``isolated_home_dirs``,
    ``cleanup_home_dirs``, ``reset_home_dirs``).
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._dirs: List[str] = []
        self._pid = os.getpid()
        self._atexit_registered = False

    def _owned(self) -> List[str]:
        """The tracked directories, called with the lock held. A forked child
        starts with an empty list: its parent's directories are not its own."""
        if self._pid != os.getpid():
            self._dirs, self._pid = [], os.getpid()
        return self._dirs

    def create(self) -> str:
        """A fresh 0700 config root, owned and tracked by this process.

        Raises ``OSError`` when it cannot be made; the adapter then runs the
        slot without one, since containment must never fail a run. The
        ``atexit`` sweep is registered at the first creation.
        """
        with self._lock:
            path = tempfile.mkdtemp(prefix=self.prefix)
            try:
                os.chmod(path, 0o700)  # mkdtemp already does this; asserted, not assumed
            except BaseException:
                shutil.rmtree(path, ignore_errors=True)
                raise
            self._owned().append(path)
            if not self._atexit_registered:
                atexit.register(self.cleanup)
                self._atexit_registered = True
            return path

    def isolated_home_dirs(self) -> Tuple[str, ...]:
        """Every config root this process created and has not yet removed."""
        with self._lock:
            return tuple(self._owned())

    def cleanup(self) -> None:
        """Remove every config root this process created. Never raises.

        Containment must never be the thing that fails a run, and this runs
        from ``atexit``, so a directory that cannot be removed is left behind
        under its known prefix rather than raising.
        """
        with self._lock:
            paths = list(self._owned())
            self._dirs = []
        for path in paths:
            shutil.rmtree(path, ignore_errors=True)

    def reset(self) -> None:
        """Forget the tracked roots without removing anything (tests)."""
        with self._lock:
            self._dirs = []
