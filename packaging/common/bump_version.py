#!/usr/bin/env python3
"""Move the shared plugin version: the owner, every mirror, and the generated trees.

``skills/multi-llm/pyproject.toml`` owns the version that Claude Code, Codex and
OpenCode all ship. ``packaging/claude/README.md`` ("Bump procedure") lists the
hand-kept mirrors and the regeneration that must follow, all in one commit. This
does that procedure as one step::

    make bump-version VERSION=0.1.7            # or DRY_RUN=1 to see the edits only

Every edit is planned and checked before any file is written. Each mirror must
already carry the current version where the procedure says it does, so a mirror that
drifted is reported rather than silently rewritten around. The new version must be
strict semver, newer than the current one and newer than every ``claude-v*``,
``codex-v*`` and ``opencode-v*`` tag in this clone. npm's own version list is
``make claude-check-bump``'s to read, since it needs the network.

It commits nothing. The source change that needs the release, these edits and both
regenerated trees go into one commit together.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "packaging") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packaging"))

from common import release_history  # noqa: E402
from common.payload import PayloadError  # noqa: E402

OWNER = "skills/multi-llm/pyproject.toml"
NPM_NAME = "opencode-multi-llm"

#: The regeneration, in the procedure's order: capability docs first, since both
#: trees carry the rendered files.
REGENERATE: Tuple[Tuple[str, ...], ...] = (
    ("uv", "run", "--project", "skills/multi-llm", "--", "python",
     "skills/multi-llm/hosts/render_capabilities.py"),
    ("make", "--no-print-directory", "codex-regen"),
    ("make", "--no-print-directory", "claude-regen"),
)


class BumpError(Exception):
    """The bump cannot be planned; nothing was written."""


@dataclass(frozen=True)
class Mirror:
    path: str
    #: What the procedure calls it, for messages.
    what: str
    #: ``(text, old, new) -> new text``; raises :class:`BumpError` when the old
    #: version is not where it should be.
    edit: Callable[[str, str, str], str]


def _replace_once(text: str, pattern: str, replacement: str, path: str, old: str) -> str:
    found = list(re.finditer(pattern, text, re.M))
    if len(found) != 1:
        raise BumpError(
            f"{path}: expected the version {old} exactly once where the bump procedure "
            f"says it is, found {len(found)}. Bring it in line with {OWNER} by hand, "
            "then bump."
        )
    match = found[0]
    return text[:match.start()] + match.expand(replacement) + text[match.end():]


def _pyproject(text: str, old: str, new: str) -> str:
    section = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not section:
        raise BumpError(f"{OWNER}: no [project] section")
    body = _replace_once(
        section.group(1), rf'^(version\s*=\s*"){re.escape(old)}("\s*)$', rf"\g<1>{new}\g<2>",
        OWNER, old,
    )
    return text[:section.start(1)] + body + text[section.end(1):]


def _uv_lock(text: str, old: str, new: str) -> str:
    return _replace_once(
        text, rf'^(name = "multi-llm"\nversion = "){re.escape(old)}(")$', rf"\g<1>{new}\g<2>",
        "skills/multi-llm/uv.lock", old,
    )


def _json_version(path: str) -> Callable[[str, str, str], str]:
    def edit(text: str, old: str, new: str) -> str:
        result = _replace_once(
            text, rf'^(\s*"version"\s*:\s*"){re.escape(old)}(")', rf"\g<1>{new}\g<2>", path, old,
        )
        try:
            declared = json.loads(result).get("version")
        except ValueError as exc:
            raise BumpError(f"{path}: does not parse after the edit: {exc}") from exc
        if declared != new:
            raise BumpError(f"{path}: the edited line is not the top-level version")
        return result
    return edit


def _capabilities(text: str, old: str, new: str) -> str:
    """The ``opencode`` entry's ``package.version``, never ``public_version``."""
    return _replace_once(
        text,
        rf'^(([ \t]+)name: {re.escape(NPM_NAME)}[ \t]*\n\2version: "){re.escape(old)}(")',
        rf"\g<1>{new}\g<3>",
        "skills/multi-llm/hosts/capabilities.yaml", old,
    )


#: ``opencode-multi-llm@<version>`` with the whole version token: ``0.1.60`` is not
#: ``0.1.6`` followed by something, and ``0.1.6.`` ending a sentence is ``0.1.6``.
_SPEC_RE = re.compile(
    rf"{re.escape(NPM_NAME)}@(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)"
    r"(?![0-9A-Za-z-]|\.\d)"
)


def _npm_readme(text: str, old: str, new: str) -> str:
    """Every pinned spec of the package, each of which must say *old*."""
    path = "packaging/opencode/README.md"
    found = list(_SPEC_RE.finditer(text))
    stale = sorted({match.group(1).rstrip(".") for match in found} - {old})
    if not found or stale:
        what = f"also pins {', '.join(stale)}" if stale else f"does not pin {NPM_NAME}@{old}"
        raise BumpError(
            f"{path}: {what}. Every pinned spec must say the current version {old}; bring "
            f"it in line with {OWNER} by hand, then bump."
        )
    pieces, last = [], 0
    for match in found:
        pieces += [text[last:match.start(1)], new]
        last = match.start(1) + len(old)
    return "".join(pieces) + text[last:]


MIRRORS: Tuple[Mirror, ...] = (
    Mirror(OWNER, "the version owner", _pyproject),
    Mirror("skills/multi-llm/uv.lock", "the lockfile's own entry", _uv_lock),
    Mirror(".claude-plugin/plugin.json", "the Claude Code manifest",
           _json_version(".claude-plugin/plugin.json")),
    Mirror("skills/multi-llm/hosts/capabilities.yaml", "the opencode package.version",
           _capabilities),
    Mirror("packaging/opencode/package.json", "the npm manifest",
           _json_version("packaging/opencode/package.json")),
    Mirror("packaging/opencode/README.md", "the npm page's pinned spec", _npm_readme),
)


def released_versions(root: Path) -> List[str]:
    """Every version a release tag of any family names in this clone."""
    versions = []
    for tag in release_history.release_tag_names(root):
        prefix = next(p for _, p in release_history.FAMILIES if tag.startswith(p))
        named = tag[len(prefix):]
        if release_history.is_version(named):
            versions.append(named)
    return versions


def plan(root: Path, new: str) -> Tuple[str, Dict[str, Tuple[str, str]]]:
    """``(old version, {path: (old text, new text)})``, or :class:`BumpError` listing
    every reason the bump cannot go ahead."""
    if not release_history.is_version(new):
        raise BumpError(f"{new!r} is not a strict semver version (x.y.z)")
    try:
        old = release_history.version_in((root / OWNER).read_text(encoding="utf-8"))
    except (OSError, PayloadError) as exc:
        raise BumpError(f"{OWNER}: cannot read the current version: {exc}") from exc

    def key(version: str):
        return release_history.version_key(release_history.release_version(version))

    problems: List[str] = []
    if key(new) <= key(old):
        problems.append(f"{new} is not newer than the current version {old}")
    released = [v for v in released_versions(root) if key(v) >= key(new)]
    if released:
        newest = max(released, key=key)
        problems.append(
            f"{new} is not newer than every released version: {newest} is tagged. "
            "Every family shares one version line."
        )
    edits: Dict[str, Tuple[str, str]] = {}
    for mirror in MIRRORS:
        path = root / mirror.path
        try:
            text = path.read_bytes().decode("utf-8")
        except OSError as exc:
            problems.append(f"{mirror.path}: cannot be read: {exc}")
            continue
        try:
            edits[mirror.path] = (text, mirror.edit(text, old, new))
        except BumpError as exc:
            problems.append(str(exc))
    if problems:
        raise BumpError("\n".join(problems))
    return old, edits


def changed_lines(before: str, after: str) -> List[Tuple[str, str]]:
    return [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]


def bump(
    root: Path, new: str, *, dry_run: bool = False, regenerate: bool = True,
    runner: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run,
) -> int:
    try:
        old, edits = plan(root, new)
    except BumpError as exc:
        for line in str(exc).splitlines():
            print(f"REFUSING: {line}", file=sys.stderr)
        print("bump-version: nothing was written.", file=sys.stderr)
        return 1

    print(f"bump-version: {old} -> {new}")
    for path, (before, after) in edits.items():
        print(f"  {path}")
        for was, now in changed_lines(before, after):
            print(f"    - {was.strip()}\n    + {now.strip()}")
    steps = [" ".join(argv) for argv in REGENERATE] if regenerate else []
    if dry_run:
        for step in steps:
            print(f"  then: {step}")
        print("bump-version: dry run, nothing written.")
        return 0

    for path, (_before, after) in edits.items():
        (root / path).write_bytes(after.encode("utf-8"))
    for argv in REGENERATE if regenerate else ():
        print(f"bump-version: {' '.join(argv)}", flush=True)
        done = runner(list(argv), cwd=str(root), check=False)
        if done.returncode != 0:
            print(
                f"REFUSING: `{' '.join(argv)}` failed (exit {done.returncode}). The mirrors "
                f"already say {new}; fix the failure, then run the remaining steps: "
                + "; ".join(" ".join(a) for a in REGENERATE[REGENERATE.index(argv):]),
                file=sys.stderr,
            )
            return 1
    print(
        f"bump-version: every mirror says {new}"
        + (" and both plugin trees are regenerated." if regenerate else
           "; the plugin trees are NOT regenerated (--no-regen).")
        + "\nNext: commit all of it in one commit with the change that needs the release, "
        "then run `make claude-publish` from master."
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bump_version.py",
        description="Set the shared plugin version in its owner and every mirror, then regenerate.",
    )
    parser.add_argument("version", help="the new version, strict semver (x.y.z)")
    parser.add_argument("--dry-run", action="store_true", help="print the edits, write nothing")
    parser.add_argument("--no-regen", action="store_true",
                        help="edit the mirrors only; skip render_capabilities and both regens")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return bump(args.repo.resolve(), args.version, dry_run=args.dry_run,
                regenerate=not args.no_regen)


if __name__ == "__main__":
    raise SystemExit(main())
