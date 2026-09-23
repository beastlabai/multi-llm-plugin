"""Shared stdout/stderr bootstrap for orchestrator entry points.

Every top-level script (orchestrators, display/apply helpers, init) should
call :func:`bootstrap_streams` at the start of ``main()`` so all entry points
get consistent stream behavior for free.
"""

import sys


def bootstrap_streams() -> None:
    """Reconfigure sys.stdout/sys.stderr for portable streaming output.

    - ``line_buffering=True``: backgrounded runs (stdout redirected to a
      file, i.e. non-TTY) stream progress/validation/salvage markers instead
      of block-buffering for minutes. Defense-in-depth alongside
      PYTHONUNBUFFERED.
    - ``encoding="utf-8", errors="replace"``: when piped or redirected,
      Python on Windows defaults the streams to the locale codec (cp1252),
      so printing non-cp1252 characters (provider/model output, em dashes,
      smart quotes) raises UnicodeEncodeError. ``replace`` is lossy for
      unencodable characters but guarantees output always succeeds — the
      standard choice for tool/LLM text output. Rejected alternatives:
      surrogateescape (lone surrogates blow up on downstream UTF-8
      re-encode), backslashreplace (noisier in model-visible output).

    Safe to call unconditionally: streams without ``reconfigure`` (tests or
    embedders replacing sys.stdout with non-TextIOWrapper objects) are left
    untouched.
    """
    for _stream in (sys.stdout, sys.stderr):
        if not hasattr(_stream, "reconfigure"):
            continue
        try:
            _stream.reconfigure(
                line_buffering=True, encoding="utf-8", errors="replace"
            )
        except (AttributeError, ValueError, OSError):
            # Non-reconfigurable stream — best effort only.
            pass
