"""One reusable "is there a Docker daemon here?" probe (File changes, Phase 2).

A pytest marker deselects nothing by itself — it only gives a ``-m`` expression
something to exclude. The container lanes are therefore kept out of a Docker-less
run by three mechanisms, and this module is the third:

1. ``pyproject.toml`` registers the ``container`` marker and sets
   ``addopts = -m 'not container'``, so a bare ``pytest`` stays green.
2. ``.github/workflows/test.yml`` says ``-m "not live and not container"``, because
   an explicit ``-m`` on the command line *replaces* ``addopts`` rather than
   composing with it.
3. This probe, applied as a module-level ``skipif`` by every container-marked
   module. Marker expressions are one careless edit — or one contributor's
   ``-m ''`` — away from being widened, and a skip that names "no Docker daemon" is
   a far better failure than a connection error out of the CLI.

The result is cached for the process: the probe costs a subprocess, and the modules
that want it are collected together.
"""

from __future__ import annotations

import functools
import subprocess


@functools.lru_cache(maxsize=1)
def docker_skip_reason() -> str | None:
    """Return why the container lanes cannot run here, or ``None`` if they can.

    ``docker version`` rather than ``docker --version`` on purpose: the latter
    answers from the client binary alone and says nothing about whether a daemon
    is reachable, which is the thing that actually decides whether a row can run.
    The daemon must also run Linux containers: every lane builds a Linux image,
    and a Windows-container daemon (the GitHub Windows runner's) answers the
    probe and then refuses the build.
    """
    try:
        completed = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Os}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return "docker CLI not on PATH"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"docker probe failed: {exc}"
    if completed.returncode == 0:
        server_os = (completed.stdout or "").strip()
        if server_os and server_os != "linux":
            return f"docker daemon runs {server_os} containers, not Linux ones"
        return None
    detail = [
        line.strip()
        for line in (completed.stderr or completed.stdout or "").splitlines()
        if line.strip()
    ]
    return "docker daemon not reachable: " + (detail[-1] if detail else "unknown error")


def requires_docker():
    """A ``pytest.mark.skipif`` for a container-marked module.

    Usage, at module scope::

        pytestmark = [pytest.mark.container, requires_docker()]
    """
    import pytest

    reason = docker_skip_reason()
    return pytest.mark.skipif(reason is not None, reason=reason or "")
