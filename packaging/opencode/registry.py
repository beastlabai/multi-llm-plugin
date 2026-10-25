#!/usr/bin/env python3
"""The one npm registry the opencode release path consults.

The URL is defined once, as ``OPENCODE_REGISTRY`` in the repository's
``Makefile``, and this module is how Python reads it: every release-path
script takes the registry explicitly and refuses any other value, because npm's
registry is configurable (a user ``.npmrc``, ``npm_config_registry``, a scoped
registry) and an inherited setting could otherwise make a gate accept ``E404``
from a different server than the one publication targets.

Parsed rather than duplicated, so a second literal cannot drift from the first.
The parse is strict: exactly one ``OPENCODE_REGISTRY := <https url>`` line, or
:class:`RegistryError`.

Stdlib only, so the ``publish`` job's third-party code stays the builder's
locked runtime dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"

_DEFINITION = re.compile(r"^OPENCODE_REGISTRY\s*:=\s*(\S+)\s*$", re.M)
_URL = re.compile(r"https://[A-Za-z0-9.-]+(?::\d+)?/?")


class RegistryError(RuntimeError):
    """The registry definition is missing, duplicated or malformed, or a
    caller named a different registry."""


def single_registry(makefile: Optional[Path] = None) -> str:
    """The URL the Makefile defines, without a trailing slash."""
    path = Path(makefile) if makefile else MAKEFILE
    text = path.read_text(encoding="utf-8")
    found = _DEFINITION.findall(text)
    if len(found) != 1:
        raise RegistryError(
            f"{path}: expected exactly one `OPENCODE_REGISTRY := <url>` line, "
            f"found {len(found)}"
        )
    url = found[0]
    if not _URL.fullmatch(url):
        raise RegistryError(f"{path}: OPENCODE_REGISTRY {url!r} is not an https origin")
    return url.rstrip("/")


def normalize(url: str) -> str:
    """``url`` with any trailing slash removed — npm writes both spellings."""
    return str(url).strip().rstrip("/")


def require_single(url: str, makefile: Optional[Path] = None) -> str:
    """``url`` normalized, or :class:`RegistryError` when it is not the one
    registry. For validators and probes handed a registry on the command line."""
    expected = single_registry(makefile)
    if normalize(url) != expected:
        raise RegistryError(
            f"registry {url!r} is not the release path's single registry "
            f"{expected!r} (Makefile OPENCODE_REGISTRY)"
        )
    return expected


if __name__ == "__main__":
    try:
        print(single_registry())
    except RegistryError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        raise SystemExit(1)
