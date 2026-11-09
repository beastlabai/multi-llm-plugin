"""Wall-clock timestamps: one function for each format the skill writes.

A writer names the format its readers expect instead of spelling
``datetime.now()`` again. Three formats exist, and each is kept as it is
because records already on disk and the readers of them compare these
strings (merging the formats would be a change of its own):

* :func:`local_iso`: local time, no offset, microseconds
  (``2026-11-09T14:03:07.123456``; ``isoformat`` leaves the fraction out at
  a whole second). ``.status.json`` (``status_file.now_iso``) and the scope
  policy's run records, which compare stamps made within the same second.
* :func:`local_iso_seconds`: local time, no offset, whole seconds
  (``2026-11-09T14:03:07``). The scope policy's round and apply records and
  the write-safety records.
* :func:`utc_iso_z`: UTC, whole seconds, ``Z`` (``2026-11-09T13:03:07Z``).
  The commit range's ``resolved_at``/``created_at`` and the commit guard.

Stdlib only; layer L0 (``tests/test_import_layers.py``).
"""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["local_iso", "local_iso_seconds", "utc_iso_z"]


def local_iso() -> str:
    """Now, local time without an offset, to the microsecond."""
    return datetime.now().isoformat()


def local_iso_seconds() -> str:
    """Now, local time without an offset, to the second."""
    return datetime.now().isoformat(timespec="seconds")


def utc_iso_z() -> str:
    """Now, in UTC, to the second, with a ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
