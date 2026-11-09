"""One atomic file write: a temp file beside the target, flushed, renamed in.

A reader of the target sees the old bytes or the new ones, never part of
either, and a failed write leaves the old bytes and no temp file. Every
replace-a-file-whole writer in the skill goes through here, so the guarantees
are the same everywhere and a call site states only how it differs (no
``fsync`` for a file rewritten many times a run, an explicit mode, writing
through a symlink):

* the temp file sits in the target's directory (the same volume, which is what
  makes ``os.replace`` atomic, on Windows too) and is created exclusively
  (``O_EXCL``: a file or symlink already at its name is never opened);
* while it holds bytes it is owner-only, and never readable more widely
  than its final mode: replacing an existing file, or with an explicit
  *mode*, it is created owner-only (0600); a new file with no explicit mode
  is created empty with the mode ``open()`` would give it (the umask's, read
  back with ``fstat``: the process umask is never set), narrowed to
  owner-only before its first byte (where the filesystem keeps modes);
  either way it gets its final mode only once written;
* the mode is applied before the ``fsync``, so data and mode reach the disk
  together, then ``os.replace``, then (POSIX) the directory is flushed so a
  returned call survives a power loss;
* a failure closes the descriptor, gives the temp file back its owner write
  bit if the final mode had taken it away (Windows deletes no read-only
  file), and removes it.

See docs/development/runtime.md#atomic-file-writes. Stdlib only; layer L0
(``tests/test_import_layers.py``).
"""

from __future__ import annotations

import os
import secrets
import stat
from typing import Optional, Tuple, Union

PathLike = Union[str, "os.PathLike[str]"]

#: The temp file's suffix. Folders the workflow owns treat a dot-file with it
#: as its own leftover (``output_handler._WORKFLOW_DOTFILE_SUFFIXES``).
TEMP_SUFFIX = ".tmp"

#: ``O_BINARY`` keeps the Windows CRT from translating newlines: the bytes
#: handed in are the bytes written.
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)

#: The temp file's bits while it holds bytes the final mode may not yet allow.
_OWNER_ONLY = 0o600


def temp_path_for(target: str, prefix: Optional[str] = None) -> str:
    """The temp file name used for *target*: ``<prefix><pid>.<random>.tmp``
    beside it, the prefix defaulting to ``.<target name>.``."""
    directory = os.path.dirname(target)
    if prefix is None:
        prefix = f".{os.path.basename(target)}."
    name = f"{prefix}{os.getpid()}.{secrets.token_hex(4)}{TEMP_SUFFIX}"
    return os.path.join(directory, name)


def flush_directory(directory: PathLike) -> None:
    """Flush *directory*'s entries (a rename into it) to disk, best-effort.

    POSIX only: Windows cannot open a directory for this, and NTFS journals
    the rename itself."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(os.fspath(directory) or os.curdir,
                             os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _existing_mode(target: str) -> Optional[int]:
    """The permission bits of the file a reader of *target* sees now (through
    a symlink), or ``None`` when there is none."""
    try:
        return stat.S_IMODE(os.stat(target).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _apply_mode(descriptor: int, temporary: str, mode: int) -> None:
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, mode)
    else:  # Windows before 3.13: only the read-only bit exists
        os.chmod(temporary, mode)


def _narrow(descriptor: int, temporary: str, mode: int) -> bool:
    """Narrow a new temp file to *mode* before its first byte; whether it
    was. A filesystem that keeps no modes (FAT, some network and FUSE
    mounts) refuses the change, and the file keeps the mode ``open()`` gave
    it: never wider than a plain write's."""
    try:
        _apply_mode(descriptor, temporary, mode)
    except OSError:
        return False
    return True


def _close_owned(descriptor: int, identity: Optional[Tuple[int, int]]) -> None:
    """Close this write's *descriptor*, which no stream owns. Before
    ``os.fdopen`` (*identity* ``None``) it is ours; after a failed one, only
    while it still names the file created (``(st_dev, st_ino)``):
    ``fdopen`` closes it itself when it fails after wrapping it, and by then
    the number may be another file's."""
    if identity is not None:
        try:
            now = os.fstat(descriptor)
        except OSError:
            return
        if (now.st_dev, now.st_ino) != identity:
            return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _discard(temporary: str, read_only: bool) -> None:
    """Remove this write's own temp file, best-effort. A read-only one gets
    its owner write bit back first: Windows refuses to delete a read-only
    file (a read-only target's mode was copied onto it)."""
    if read_only:
        try:
            os.chmod(temporary, _OWNER_ONLY)
        except OSError:
            pass
    try:
        os.unlink(temporary)
    except OSError:
        pass


def atomic_write_bytes(
    path: PathLike,
    data: bytes,
    *,
    keep_mode: bool = True,
    fsync: bool = True,
    mode: Optional[int] = None,
    follow_symlinks: bool = False,
    temp_prefix: Optional[str] = None,
) -> None:
    """Replace *path* with *data*, atomically.

    - *keep_mode*: an existing file's permission bits carry over to the
      replacement. Off, the replacement gets a new file's mode (the umask's).
    - *mode*: the replacement's exact permission bits, whatever the old file
      or the umask say (it overrides *keep_mode*).
    - *fsync*: flush the file before the rename and the directory after it.
      Off only for a file rewritten so often that the flush would cost more
      than a lost last write (``state.json``, under its lock).
    - *follow_symlinks*: a symlink at *path* is written through, so the link
      survives and its target gets the bytes. Off, the link itself is
      replaced by a regular file (its target is left alone; *keep_mode* reads
      the mode through the link).
    - *temp_prefix*: the temp file's name before ``<pid>.<random>.tmp``, for a
      caller whose leftovers something matches by name.

    The directory must exist. Raises ``OSError`` (``FileExistsError`` if the
    temp name is taken); on any exception the temp file is removed and the
    old file is untouched.
    """
    target = os.fspath(path)
    if follow_symlinks:
        target = os.path.realpath(target)
    final_mode = mode
    if final_mode is None and keep_mode:
        final_mode = _existing_mode(target)
    temporary = temp_path_for(target, temp_prefix)
    descriptor = os.open(temporary, _CREATE_FLAGS, 0o666 if final_mode is None else 0o600)
    # Ours from here: on any failure the descriptor is closed and the file
    # removed.
    identity: Optional[Tuple[int, int]] = None
    stream = None
    read_only = False
    apply_final = final_mode is not None
    try:
        created = os.fstat(descriptor)
        if final_mode is None:
            # The umask's mode, read off the empty file; the bytes go in
            # owner-only and the mode comes back before the fsync.
            final_mode = stat.S_IMODE(created.st_mode)
            if final_mode & ~_OWNER_ONLY:
                apply_final = _narrow(descriptor, temporary, final_mode & _OWNER_ONLY)
        identity = (created.st_dev, created.st_ino)
        stream = os.fdopen(descriptor, "wb")
        with stream:
            stream.write(data)
            stream.flush()
            if apply_final:
                _apply_mode(stream.fileno(), temporary, final_mode)
            read_only = not final_mode & stat.S_IWUSR
            if fsync:
                os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if stream is None:
            _close_owned(descriptor, identity)
        _discard(temporary, read_only)
        raise
    if fsync:
        flush_directory(os.path.dirname(target))


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    newline: str = "\n",
    keep_mode: bool = True,
    fsync: bool = True,
    mode: Optional[int] = None,
    follow_symlinks: bool = False,
    temp_prefix: Optional[str] = None,
) -> bytes:
    """:func:`atomic_write_bytes` of *text* as UTF-8; the bytes written.

    Each ``\\n`` is written as *newline*, ``\\n`` by default, so the bytes
    are the same on every platform. ``newline=os.linesep`` writes what a
    text-mode ``write_text`` would, for a caller that has always written the
    platform's newlines."""
    if newline != "\n":
        text = text.replace("\n", newline)
    data = text.encode("utf-8")
    atomic_write_bytes(
        path, data, keep_mode=keep_mode, fsync=fsync, mode=mode,
        follow_symlinks=follow_symlinks, temp_prefix=temp_prefix,
    )
    return data
