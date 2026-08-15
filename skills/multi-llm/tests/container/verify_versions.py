#!/usr/bin/env python3
"""The build's final layer: what did we ACTUALLY install? (C9)

A pin is a request. `npm ci` against a lockfile, `--require-hashes`, a verified
sha256 -- each one asks the network for named bytes, and each one can still be
defeated by an installer that resolves something else, a postinstall step that
self-updates, or a lock row nobody regenerated after a hand edit. This script is
the evidence: it runs every `verify.command` in ``cli-versions.lock``, recovers a
version with that row's own `verify.pattern`, and exits non-zero on any mismatch.
The Dockerfile runs it as its last RUN, so a drifted install is a **red build**
rather than a green build that lies about its contents.

It checks three separate things, because they are three separate failures:

1. **Version identity.** Every enforced row's installed version equals its pin.
   A row declaring ``verify: none`` is recorded with its reason -- a CLI with no
   machine-readable version surface is a visible gap, never one assumed away.
2. **PATH shape.** Every CLI binary resolves inside ``/opt/mllm/bin`` (the
   wrapper) and not inside ``/opt/mllm/real`` (the genuine binary). C2's whole
   redirection mechanism is "the wrapper is first on PATH"; an image where that
   silently stopped being true would run every row against a real backend and
   only ever fail as a mysterious network error.
3. **Wrapper identity.** Each ``/opt/mllm/bin/<cli>`` really is ``wrapper.py``
   under that name, since the wrapper resolves which CLI it is from
   ``Path(sys.argv[0]).name``.

**Why the version calls go to ``/opt/mllm/real`` rather than through PATH.** The
wrapper deliberately rebuilds its child's environment from scratch and points
every config root at a fresh per-invocation directory -- which is correct at run
time and wrong here: at build time there is no ``redirect.yaml``, so a CLI whose
own root has to be named (``CODEX_HOME``) would be handed a ``$HOME`` with no
``.codex`` in it and exit 1 on a question that has nothing to do with its version.
Asking the genuine binary keeps this step about the pin, and check 2 above keeps
the wrapper tree asserted separately instead of implicitly.

``--mode latest`` inverts exactly the ``clis`` rows: they are recorded rather
than enforced, because that build asked for ``@latest`` on purpose. Everything
else -- node, npm, uv, and both PATH checks -- is enforced in both modes, which
is what makes a red drift run point at the CLI that moved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

BIN_DIR = Path(os.environ.get("MLLM_BIN_DIR", "/opt/mllm/bin"))
REAL_DIR = Path(os.environ.get("MLLM_REAL_DIR", "/opt/mllm/real"))
WRAPPER = Path(os.environ.get("MLLM_WRAPPER", "/opt/mllm/rig/wrapper.py"))

#: Some CLIs print a banner, a deprecation notice or an update nag before the
#: version, and at least one (`goose`) prints leading whitespace. The pattern is
#: matched against each stripped line rather than the whole blob so a row's
#: anchored `^...$` stays meaningful.
def _extract(pattern: str, output: str) -> str | None:
    compiled = re.compile(pattern)
    for line in output.splitlines():
        match = compiled.search(line.strip())
        if match:
            try:
                return match.group("version")
            except IndexError:  # pragma: no cover - a lock row without the group
                return match.group(0)
    return None


def _resolve(binary: str) -> str | None:
    """The genuine binary if we installed one, else whatever PATH offers.

    ``node``/``npm``/``python3``/``git`` come from the base image and have no
    entry under ``/opt/mllm/real``; every CLI row does.
    """
    candidate = REAL_DIR / binary
    if candidate.exists():
        return str(candidate)
    return shutil.which(binary)


def check_row(row: dict, env: dict, timeout: int) -> dict:
    """Run one row's version command and grade it."""
    row_id = row.get("id", row.get("binary", "?"))
    result = {
        "id": row_id,
        "binary": row.get("binary"),
        "install": row.get("install"),
        "expected": row.get("version"),
        "installed": None,
        "status": "unknown",
        "enforced": bool(row.get("enforce", True)),
    }

    verify = row.get("verify")
    if verify in (None, "none"):
        result["status"] = "not-verifiable"
        result["reason"] = row.get("reason", "no machine-readable version surface")
        return result

    binary = row.get("binary") or row_id
    resolved = _resolve(binary)
    if resolved is None:
        result["status"] = "missing"
        result["reason"] = f"{binary} is not installed"
        return result

    command = [resolved, *verify["command"][1:]]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["reason"] = f"{' '.join(verify['command'])} did not answer in {timeout}s"
        return result

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    result["exit"] = completed.returncode
    installed = _extract(verify["pattern"], output)
    result["installed"] = installed

    if installed is None:
        result["status"] = "unparseable"
        result["reason"] = (
            f"{' '.join(verify['command'])} (exit {completed.returncode}) printed "
            f"nothing matching {verify['pattern']!r}: {output.strip()[:400]!r}"
        )
        return result

    result["status"] = "ok" if installed == row.get("version") else "mismatch"
    return result


def check_path_shape(rows: list[dict]) -> list[dict]:
    """C2's load-bearing invariant, asserted rather than commented."""
    findings: list[dict] = []
    for row in rows:
        binary = row.get("binary") or row["id"]
        resolved = shutil.which(binary)
        entry = {"id": row.get("id"), "binary": binary, "path": resolved}
        if resolved is None:
            entry["status"] = "missing-from-path"
        elif Path(resolved).parent != BIN_DIR:
            entry["status"] = "not-wrapped"
            entry["reason"] = (
                f"{binary} resolves to {resolved}, not {BIN_DIR}/{binary} -- "
                "the wrapper is not first on PATH, so every row would run "
                "unredirected"
            )
        elif Path(resolved).resolve() != WRAPPER.resolve():
            entry["status"] = "not-the-wrapper"
            entry["reason"] = (
                f"{resolved} is not {WRAPPER}; the wrapper resolves its identity "
                "from Path(sys.argv[0]).name and cannot work under another file"
            )
        elif not (REAL_DIR / binary).exists():
            entry["status"] = "no-real-binary"
            entry["reason"] = f"{BIN_DIR/binary} wraps nothing: {REAL_DIR/binary} is absent"
        else:
            entry["status"] = "ok"
        findings.append(entry)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default="/opt/mllm/lock/cli-versions.lock")
    parser.add_argument("--mode", choices=("pinned", "latest"), default="pinned")
    parser.add_argument("--out", default="/opt/mllm/lock/cli-versions.installed.json")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--home", default=None,
                        help="a scratch $HOME for the version calls; several CLIs "
                             "crash outright when $HOME does not exist")
    args = parser.parse_args(argv)

    lock = yaml.safe_load(Path(args.lock).read_text(encoding="utf-8"))

    env = dict(os.environ)
    if args.home:
        home = Path(args.home)
        # `gemini` crashes when $HOME does not exist and `codex` exits 1 when its
        # own root does not, so both are created rather than merely named.
        for rel in (".", ".codex", ".config", ".cache", ".local/share", ".gemini"):
            (home / rel).mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["CODEX_HOME"] = str(home / ".codex")
        env["XDG_CONFIG_HOME"] = str(home / ".config")
        env["XDG_CACHE_HOME"] = str(home / ".cache")
        env["XDG_DATA_HOME"] = str(home / ".local" / "share")

    toolchain = list(lock.get("toolchain") or [])
    clis = list(lock.get("clis") or [])

    report = {
        "mode": args.mode,
        "platform": lock.get("platform"),
        "base_image": lock.get("base_image"),
        "toolchain": [check_row(r, env, args.timeout) for r in toolchain],
        "clis": [check_row(r, env, args.timeout) for r in clis],
        "path_shape": check_path_shape(clis),
    }

    failures: list[str] = []
    graded = ([(row, False) for row in report["toolchain"]]
              # The one group the pin mode inverts: a `latest` build asked for a
              # moving CLI layer on purpose, so its rows are recorded, never
              # enforced. node, npm, uv and both PATH checks stay enforced.
              + [(row, args.mode == "latest") for row in report["clis"]])
    for row, inverted in graded:
        enforced = row["enforced"] and not inverted
        line = (f"{row['id']:<14} {row['status']:<15} "
                f"expected={row['expected']!s:<14} installed={row['installed']!s}")
        print(line, flush=True)
        if row["status"] in ("ok", "not-verifiable"):
            continue
        message = f"{row['id']}: {row['status']} -- {row.get('reason', '')}".strip()
        if row["status"] == "mismatch":
            message = (f"{row['id']}: pinned {row['expected']!r} but installed "
                       f"{row['installed']!r}")
        if enforced:
            failures.append(message)
        else:
            print(f"  (recorded, not enforced) {message}", flush=True)

    for entry in report["path_shape"]:
        if entry["status"] != "ok":
            failures.append(f"{entry['binary']}: {entry['status']} -- "
                            f"{entry.get('reason', '')}".strip())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    if failures:
        print("\nVERSION VERIFICATION FAILED -- the image does not contain what "
              "cli-versions.lock says it does:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print("\nEither the lock row is wrong (fix it, in a reviewable commit) or "
              "the installer resolved something we did not name.", file=sys.stderr)
        return 1

    print(f"\nall pins verified ({args.mode} mode); wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
