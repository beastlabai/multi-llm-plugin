#!/usr/bin/env python3
"""Stage the skill payload into a distribution tree, and sweep what came out.

**Why this is shared rather than per-host.** Two plans ship the same
``skills/multi-llm`` tree through two different distributions — an npm package
(``plans/opencode-native-plugin.md``) and a codex plugin
(``plans/codex-native-plugin.md``) — and each needs the same six staging rules
plus one or two of its own. Neither list subsumes the other and every omission
is unrecoverable in its own way, so the union lands once, here, and each host's
``build.py`` is a thin driver over it.

The rules, and the failure each one catches:

============================================  ===============================
rule                                          the failure it catches
============================================  ===============================
allowlist over ``git ls-files``               untracked scratch, ``.env``, a
                                              developer's
                                              ``.multi-llm/providers.yaml``
an unexpected file is an **error**            a fat tarball
**required-inventory** assertion              a **short** tarball — an ignore
                                              file inside the staged tree,
                                              which no absence list can see
**symlink refusal** (mode ``120000``)         an install copy that silently
                                              omits symlinks and leaves the
                                              parent empty at exit 0
**dirty/untracked-input refusal**             a build that ships uncommitted
                                              local state
sweep a **temp tree before the swap**         a half-written payload on a
                                              failed sweep
============================================  ===============================

**Why an allowlist and not a prune.** npm never reads an ignore file *above* the
package root, so this repository's own ``.gitignore`` is invisible to a
``npm pack`` run under ``packaging/``. Every locally-ignored artefact staged into
that tree is a publish candidate unless something names it — and the live
counter-example is ``skills/multi-llm/plans/``, ignored at ``.gitignore:19``,
invisible to ``git status``, and copied straight into a tarball by any denylist
that did not think to name it. Untracked scratch output, ``.env`` files, editor
backups and a developer's ``.multi-llm/providers.yaml`` are the same class, and
**none of them is on npm's default-ignore list at any depth**.

**And the rule has an opposite edge that is easy to get backwards.** An ignore
file *inside* a ``files``-listed tree **is** honoured, and ``files`` cannot
re-include against it — not even by naming the excluded path explicitly. So any
``.gitignore`` this module copied into the staged tree would silently *delete*
files from the tarball. The skill's own runtime writes ``.multi-llm/tmp/.gitignore``
into consuming repos, and the source tree carries ``.gitignore`` files under
``.venv/``, ``.pytest_cache/`` and ``.ruff_cache/``. Today the allowlist keeps
them out; the day the allowlist grows, the failure mode is a tarball that is
**short**, not one that is fat — which an absence list can never catch. That is
why :func:`stage` asserts a *required inventory* after staging rather than
merely that forbidden paths are absent.

**The allowlist is parsed, never typed.** It already exists in this repository
twice — at ``.dockerignore`` and at ``tests/container/context_manifest.txt``,
where ``test_build_context.py`` asserts the two equal in both directions. A third
hand-typed copy here would be exactly the single-source failure this project's
``AGENTS.md`` names, so :func:`runtime_allowlist` parses the manifest's
product-under-test block and a test guards the parse.
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import (
    Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
)

#: Repo-root-relative path of the manifest the allowlist is parsed out of.
MANIFEST_RELATIVE = "skills/multi-llm/tests/container/context_manifest.txt"

#: The manifest heading that opens the product-under-test block, and the one
#: that closes it. Matched on the heading *text*, not on a line number: the file
#: is a hand-maintained document and a row will be added to it.
_BLOCK_START = "--- the product under test ---"
_BLOCK_END = "--- the rig that drives it ---"

#: Everything the runtime reads at some point during a dispatch. A payload
#: missing any of these is an ``ENOENT`` raised from inside a tool ``execute``,
#: which on the npm route is a silent exit 0 — the exact failure this assertion
#: exists to convert into a build error.
#:
#: The last four were missing until the two packaging plans were reconciled, and
#: at least one omission is fatal rather than cosmetic:
#:
#: * ``providers.yaml`` — ``utils/provider_registry.py`` loads the base registry
#:   as ``Path(__file__).parent.parent / "providers.yaml"``;
#: * ``templates/config/providers.override.yaml`` — the ``--init`` contract;
#: * ``templates/consolidated_report_template.html`` — loaded by
#:   ``utils/consolidation.py``;
#: * ``references/`` — cited by ``instructions/*.md`` at run time;
#: * ``assets/icons/`` — inlined by the HTML report.
REQUIRED_INVENTORY: Tuple[str, ...] = (
    "multi_llm.py",
    "modes.py",
    "pyproject.toml",
    "uv.lock",
    "SKILL.md",
    "providers.yaml",
    "hosts/capabilities.yaml",
    "hosts/uv-missing.txt",
    "hosts/opencode/shim.md.tmpl",
    "hosts/codex/shim.md.tmpl",
    "templates/config/providers.override.yaml",
    "templates/consolidated_report_template.html",
)

#: Directories the payload must not be missing. Checked as "at least one file
#: under this prefix", because naming every file would make the assertion a
#: second copy of the tree.
REQUIRED_TREES: Tuple[str, ...] = (
    "utils/",
    "hosts/",
    "instructions/",
    "schemas/",
    "prompts/",
    "references/",
    "templates/",
    "assets/icons/",
)

#: Filenames and suffixes that must never reach a distribution tree. This is the
#: *secret* sweep, and it is mandatory rather than belt-and-braces: measured, a
#: ``files: ["skills"]`` fixture shipped ``.env``, ``creds/auth.json`` and
#: secret-bearing prose, while npm dropped only ``.npmrc``, ``node_modules/``,
#: ``.git/`` and ``package-lock.json``.
_SECRET_NAMES: Tuple[str, ...] = (
    ".env",
    ".npmrc",
    "auth.json",
    "e2e.env",
    "credentials.json",
)
#: ``id_*`` is the plan's literal glob and is deliberately kept literal:
#: enumerating ``id_rsa``/``id_ecdsa``/``id_ed25519`` lets ``id_dsa``, the
#: FIDO ``*_sk`` variants and whatever shape ssh-keygen grows next walk
#: straight through. Nothing tracked under ``skills/multi-llm`` begins with
#: ``id_``, so the width costs nothing today, and a future false positive is a
#: build error a developer sees rather than a key a user gets.
_SECRET_GLOBS: Tuple[str, ...] = (
    ".env.*",
    "*.pem",
    "*.key",
    "id_*",
    "*.tfstate",
    "*.credentials.json",
    "*.p12",
    "*.pfx",
)
#: Directory names whose *contents* are credentials by location, whatever the
#: filename. Matched on a single path segment, anywhere in the path.
_SECRET_DIRS: Tuple[str, ...] = ("creds", ".codex", ".multi-llm")

#: Credential directories whose trailing name is **not** distinctive on its own,
#: so they are matched as *consecutive* path segments instead.
#:
#: ``~/.config/multi-llm/`` is where this project's own live-lane key lives, and
#: the sweep owes it more than the two filenames it happens to know: ``e2e.env``
#: is in :data:`_SECRET_NAMES`, but a second credential dropped beside it under
#: an unscanned suffix would otherwise ship. ``multi-llm`` cannot simply join
#: :data:`_SECRET_DIRS` — the staged tree legitimately *is* ``skills/multi-llm``,
#: so a bare-name match would flag the whole payload.
_SECRET_DIR_PATHS: Tuple[Tuple[str, ...], ...] = (
    (".config", "multi-llm"),
)

#: Key-shaped strings. Deliberately anchored on vendor prefixes rather than on
#: entropy: an entropy heuristic over a tree that ships lockfile hashes and
#: base64 icons is all false positives, and the point of this sweep is that a hit
#: stops a release.
#:
#: **Order is load-bearing**: :func:`scan_tree` reports the first pattern that
#: hits and stops, and the generic ``sk-`` catch-all matches every ``sk-ant-``
#: and ``sk-or-v1-`` string too. Specific vendors therefore come first, so a
#: finding names the key a reader has to go and rotate.
_SECRET_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenRouter key", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}")),
    ("PEM private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)

#: Text suffixes worth scanning the *contents* of. A binary sweep would be all
#: noise, and the icons are the only sizeable binaries in the payload.
_SCANNED_SUFFIXES: Tuple[str, ...] = (
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".tmpl", ".mjs",
    ".js", ".html", ".cfg", ".ini", ".lock", ".sh",
)


class PayloadError(RuntimeError):
    """A staging or sweep refusal. Always fatal: a build that cannot account for
    what it is shipping does not produce an artifact."""


# ---------------------------------------------------------------------------
# The allowlist, parsed from the one file that already owns it
# ---------------------------------------------------------------------------

def repo_root(start: Optional[Path] = None) -> Path:
    """The checkout root, found by walking up for ``.git``.

    Not ``git rev-parse``: this must work inside a staged copy and inside a
    container layer where the binary may be absent, and the answer is a
    directory rather than a git fact.
    """
    here = Path(start).resolve() if start else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise PayloadError(
        f"not inside a git checkout: no .git found at or above {here}"
    )


def runtime_allowlist(root: Optional[Path] = None) -> Tuple[str, ...]:
    """The product-under-test globs, parsed from ``context_manifest.txt``.

    Repo-root-relative, in file order, with comments and ``# optional`` markers
    stripped. The rig block below the heading is deliberately *not* included:
    ``tests/`` is tracked but is not runtime payload.
    """
    base = Path(root) if root else repo_root()
    path = base / MANIFEST_RELATIVE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PayloadError(
            f"cannot read the payload allowlist at {path}: {exc}\n"
            "It is the single owner of what ships; a build cannot guess it."
        ) from exc

    patterns: List[str] = []
    inside = False
    for line in text.splitlines():
        if _BLOCK_START in line:
            inside = True
            continue
        if _BLOCK_END in line:
            break
        if not inside:
            continue
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            patterns.append(stripped)
    if not patterns:
        raise PayloadError(
            f"the product-under-test block of {path} parsed empty. The heading "
            f"{_BLOCK_START!r} or {_BLOCK_END!r} has moved; fix the parse rather "
            "than inlining a second copy of the list."
        )
    return tuple(patterns)


@functools.lru_cache(maxsize=None)
def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """Compile a manifest glob the way the manifest's own reader does.

    ``*`` and ``?`` stop at a separator; ``**`` spans them, and ``**/`` also
    matches zero directories so ``**/x`` catches the root copy too.

    **Hand-rolled rather than ``fnmatch``, and that is not a style choice.**
    ``fnmatch`` translates ``*`` to ``.*``, which makes
    ``skills/multi-llm/*.py`` match every ``.py`` in every subtree — so the
    allowlist would silently admit the whole ``tests/`` tree, and a payload
    staged with it would ship this repository's own test corpus. That is not a
    hypothetical: it is what this function was written in response to.

    Byte-for-byte the rule ``tests/test_build_context.py::_glob_regex`` applies
    to the same file, and ``test_payload_allowlist_matches_the_build_context``
    asserts the two agree on the real manifest rather than leaving that to this
    comment.
    """
    out: List[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            run = index
            while run < len(pattern) and pattern[run] == "*":
                run += 1
            if run - index >= 2:
                if pattern[run:run + 1] == "/":
                    out.append("(?:.*/)?")
                    index = run + 1
                else:
                    out.append(".*")
                    index = run
            else:
                out.append("[^/]*")
                index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(char))
            index += 1
    return re.compile("".join(out) + r"\Z")


def _matches(relative: str, pattern: str) -> bool:
    """Whether *relative* is admitted by *pattern*."""
    return _glob_regex(pattern).match(relative) is not None


# ---------------------------------------------------------------------------
# Inputs: tracked files only, and only from a clean tree
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        raise PayloadError("git is required to stage a payload") from exc
    except subprocess.CalledProcessError as exc:
        raise PayloadError(
            f"git {' '.join(args)} failed: {exc.stderr.strip()}"
        ) from exc
    return result.stdout


def tracked_files(root: Path, prefix: str) -> Tuple[str, ...]:
    """Tracked paths under *prefix*, repo-root-relative, sorted.

    ``git ls-files -z`` rather than a walk: untracked is never a candidate at
    all, so a new ignored tree added years from now cannot leak by default.
    """
    out = _git(root, "ls-files", "-z", "--", prefix)
    return tuple(sorted(p for p in out.split("\0") if p))


def _symlinks(root: Path, prefix: str) -> Tuple[str, ...]:
    """Tracked paths under *prefix* whose git mode is ``120000``.

    A symlink inside a distribution is a portability hazard in both directions:
    npm tarballs carry them unevenly across platforms, and codex's install copy
    silently omits them and leaves the parent directory empty at **exit 0**. So
    a symlink in the payload is refused rather than resolved — resolving one
    would ship a second copy of whatever it points at, which is its own
    single-source failure.
    """
    out = _git(root, "ls-files", "-s", "-z", "--", prefix)
    found: List[str] = []
    for record in out.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        if meta.split(" ", 1)[0] == "120000":
            found.append(path)
    return tuple(sorted(found))


def assert_clean(root: Path, prefix: str) -> None:
    """Refuse to stage from a tree with uncommitted or untracked changes.

    A build that ships local state is a build nobody can reproduce, and on npm
    the result is immutable the moment it is published. ``--allow-dirty`` exists
    for staging to scratch during development; it may never reach a publish.
    """
    out = _git(root, "status", "--porcelain", "--", prefix)
    dirty = [line for line in out.splitlines() if line.strip()]
    if dirty:
        listing = "\n".join(f"    {line}" for line in dirty[:20])
        more = "" if len(dirty) <= 20 else f"\n    … and {len(dirty) - 20} more"
        raise PayloadError(
            f"refusing to stage from a dirty tree — {len(dirty)} change(s) under "
            f"{prefix}:\n{listing}{more}\n"
            "Commit them, or pass --allow-dirty to stage into scratch for a "
            "local test. --allow-dirty never reaches a publish."
        )


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def selected_files(
    root: Path,
    prefix: str,
    allowlist: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """Tracked paths under *prefix* that the allowlist admits, sorted."""
    patterns = tuple(allowlist) if allowlist is not None else runtime_allowlist(root)
    return tuple(
        path for path in tracked_files(root, prefix)
        if any(_matches(path, pattern) for pattern in patterns)
    )


def expected_payload(
    root: Optional[Path] = None,
    *,
    prefix: str = "skills/multi-llm",
    strip: str = "skills/multi-llm",
    allowlist: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """The allowlist expansion, payload-relative: what a staged tree — or a
    tarball built from one — must contain **exactly**.

    :func:`stage` returns the same set as a fact about the copy it just made.
    This computes it from ``git ls-files`` and the manifest alone, so
    :func:`scan_tarball` grades what shipped against an independent answer
    rather than against the build's own bookkeeping.
    """
    base = Path(root) if root else repo_root()
    strip_prefix = strip.rstrip("/") + "/"
    return tuple(sorted(
        path[len(strip_prefix):]
        for path in selected_files(base, prefix, allowlist)
        if path.startswith(strip_prefix)
    ))


def _staging_parent(destination: Path) -> Path:
    """Where the temp staging tree is created: **one level above** the directory
    a distribution names wholesale.

    Deliberately not a sibling of *destination*. The staging directory is removed
    by :func:`stage`'s ``finally``, which a killed process never reaches — and
    ``package.json`` says ``files: ["skills"]``, so a leftover
    ``skills/multi-llm.staging.XXXX`` is gitignored (``.gitignore:95``), invisible
    to ``git status``, unseen by :func:`_assert_no_unexpected` (which grades the
    staged tree, never its neighbours) and **packed by the next ``npm pack``**.
    One level up it is outside every ``files`` entry, so it cannot ship, and it
    is untracked rather than ignored, so ``git status`` shows it. The tarball
    check catches the same class from the other end: a leftover beside the
    payload is a *surplus* entry in :func:`scan_tarball`'s comparison.

    Still on the same filesystem as *destination*, which is what keeps the swap
    a rename rather than a copy. The equality guard is for a destination like
    ``/payload``, where there is no level above the parent to move to.
    """
    parent = Path(destination).parent
    above = parent.parent
    return parent if above == parent else above


def stage(
    root: Path,
    destination: Path,
    *,
    prefix: str = "skills/multi-llm",
    strip: str = "skills/multi-llm",
    allow_dirty: bool = False,
    allowlist: Optional[Sequence[str]] = None,
    required: Sequence[str] = REQUIRED_INVENTORY,
    required_trees: Sequence[str] = REQUIRED_TREES,
    required_extra: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Copy the allowlisted payload into *destination*, then account for it.

    Returns the staged paths, relative to *destination*. Raises
    :class:`PayloadError` on a dirty tree, a symlink, an unexpected file or a
    missing required resource.

    *required_extra* is merged additively with *required* — it never replaces
    it. It exists because :data:`REQUIRED_INVENTORY` is a static, hand-typed
    list and cannot see a resource a caller only discovers by reading data this
    module deliberately never imports (see the module docstring: no skill
    module import here). ``modes.py``'s ``target_module`` rows are exactly
    that case — a dispatch table entry can name a module that ``git
    ls-files`` never selected (untracked), and the static list has no way to
    know the module exists at all. The caller derives that set and passes it
    here; this function only merges and checks it.

    The write is **atomic against the destination**: staging happens in a temp
    directory on the same filesystem (see :func:`_staging_parent` for *which*
    one, which is load-bearing) and the swap only occurs once every assertion
    has passed, so a failed sweep never leaves a half-written payload where a
    previous good one was.
    """
    root = Path(root)
    destination = Path(destination)

    if not allow_dirty:
        assert_clean(root, prefix)

    links = _symlinks(root, prefix)
    if links:
        listing = "\n".join(f"    {path}" for path in links)
        raise PayloadError(
            "refusing to stage a payload containing symlinks:\n"
            f"{listing}\n"
            "A symlink survives a tarball unevenly and is silently dropped by a "
            "plain install copy, which leaves the parent directory empty at exit "
            "0. Replace it with a real file."
        )

    chosen = selected_files(root, prefix, allowlist)
    if not chosen:
        raise PayloadError(
            f"the allowlist selected no files under {prefix}. Either the prefix "
            "is wrong or the manifest block failed to parse."
        )

    strip_prefix = strip.rstrip("/") + "/"
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(
        prefix=destination.name + ".staging.",
        dir=str(_staging_parent(destination)),
    ))
    try:
        staged: List[str] = []
        for path in chosen:
            if not path.startswith(strip_prefix):
                raise PayloadError(
                    f"selected path {path!r} is not under {strip_prefix!r}"
                )
            relative = path[len(strip_prefix):]
            target = scratch / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / path, target)
            staged.append(relative)

        _assert_no_unexpected(scratch, staged)
        merged_required = tuple(dict.fromkeys((*required, *required_extra)))
        _assert_inventory(scratch, staged, merged_required, required_trees)
        scan_tree(scratch)

        if destination.exists():
            shutil.rmtree(destination)
        scratch.replace(destination)
        scratch = None  # type: ignore[assignment]
    finally:
        if scratch is not None and scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

    return tuple(sorted(staged))


def _assert_no_unexpected(tree: Path, staged: Sequence[str]) -> None:
    """Every file on disk was one we meant to put there. A build that cannot
    account for a file does not produce an artifact."""
    expected = set(staged)
    actual = {
        str(path.relative_to(tree)).replace(os.sep, "/")
        for path in tree.rglob("*") if path.is_file()
    }
    surplus = sorted(actual - expected)
    if surplus:
        listing = "\n".join(f"    {path}" for path in surplus[:20])
        raise PayloadError(
            f"{len(surplus)} unexpected file(s) in the staged tree:\n{listing}\n"
            "Staging is an allowlist; anything it did not intend to ship is an "
            "error, not a silent copy."
        )


def _assert_inventory(
    tree: Path,
    staged: Sequence[str],
    required: Sequence[str],
    required_trees: Sequence[str],
) -> None:
    """The complete required inventory, not a spot check.

    This is the assertion that catches a **short** payload, which no absence
    list can see: an ignore file inside the staged tree silently deletes files
    from a tarball, and a too-aggressive allowlist looks exactly like a
    successful build.
    """
    present = set(staged)
    missing = [name for name in required if name not in present]
    empty_trees = [
        prefix for prefix in required_trees
        if not any(path.startswith(prefix) for path in present)
    ]
    if missing or empty_trees:
        lines = [f"    {name}" for name in missing]
        lines += [f"    {prefix} (no files)" for prefix in empty_trees]
        raise PayloadError(
            "the staged payload is missing resources the runtime reads:\n"
            + "\n".join(lines)
            + "\nA missing message or template file is an ENOENT raised from "
            "inside a dispatch, which on the npm route is a silent exit 0."
        )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def _credential_directory(segments: Sequence[str]) -> Optional[str]:
    """The credential directory *segments* sits inside, or ``None``.

    Two shapes, because one does not cover both cases. A bare name from
    :data:`_SECRET_DIRS` matches any single segment of the path; a path pair from
    :data:`_SECRET_DIR_PATHS` must match *consecutive* segments, because its
    trailing name is not distinctive enough to match alone.
    """
    for name in _SECRET_DIRS:
        if name in segments:
            return name + "/"
    for pair in _SECRET_DIR_PATHS:
        span = len(pair)
        for index in range(len(segments) - span + 1):
            if tuple(segments[index:index + span]) == pair:
                return "/".join(pair) + "/"
    return None


def scan_tree(tree: Path) -> None:
    """Refuse a directory that carries a credential-shaped file or string.

    The signature takes a **directory**, never a tarball: the npm route extracts
    its ``.tgz`` and calls this on the result, while the codex route calls it on
    the staging directory before the swap. A tarball-shaped API would be
    unusable on the second.
    """
    tree = Path(tree)
    findings: List[str] = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(tree)).replace(os.sep, "/")
        name = path.name
        segments = relative.split("/")[:-1]

        if name in _SECRET_NAMES:
            findings.append(f"{relative}: credential filename {name!r}")
            continue
        if any(fnmatch.fnmatchcase(name, glob) for glob in _SECRET_GLOBS):
            findings.append(f"{relative}: credential-shaped filename")
            continue
        located = _credential_directory(segments)
        if located is not None:
            findings.append(f"{relative}: inside credential directory {located}")
            continue
        if path.suffix.lower() not in _SCANNED_SUFFIXES:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for label, pattern in _SECRET_PATTERNS:
            match = pattern.search(body)
            if match:
                findings.append(
                    f"{relative}: {label} at offset {match.start()}"
                )
                break

    if findings:
        listing = "\n".join(f"    {line}" for line in findings)
        raise PayloadError(
            f"the sweep found {len(findings)} credential-shaped item(s):\n"
            f"{listing}\n"
            "Nothing here is publishable. This sweep is mandatory rather than "
            "belt-and-braces: npm's own default-ignore list drops none of these "
            "at any depth."
        )


def scan_tarball(
    archive: Path,
    *,
    payload_prefix: str = "package/skills/multi-llm",
    expected: Optional[Sequence[str]] = None,
    root: Optional[Path] = None,
    required: Sequence[str] = REQUIRED_INVENTORY,
    required_trees: Sequence[str] = REQUIRED_TREES,
) -> None:
    """Extract *archive* and hold it to every rule the staging tree already met.

    What ``npm pack`` writes is what ships, so this runs over the finished
    tarball rather than over the staging directory it was built from — a
    lifecycle hook that re-stages runs too early to see one. **That is exactly
    why the inventory assertions belong here as well and not only in**
    :func:`stage`: the staged tree passed them before npm ever looked at it, and
    npm applies its own ignore rules *afterwards*. An ignore file inside a
    ``files``-listed tree is honoured and ``files`` cannot re-include against it,
    so entries can vanish between staging and pack — and a tarball that is short
    for that reason looks precisely like a successful build. Inside
    :func:`stage`, by contrast, ``_assert_no_unexpected`` is near-tautological:
    it grades a directory this module created empty moments earlier.

    Three assertions, and the middle one is scoped to the *container* — the
    directory the distribution names wholesale, ``package/skills/`` — rather
    than to the payload directory itself:

    1. :func:`scan_tree`: no credential-shaped filename, directory or string;
    2. the container holds exactly ``multi-llm/`` plus the allowlist expansion,
       so a **short** tarball is a set difference rather than a silent success,
       and a leftover ``multi-llm.staging.XXXX`` from a killed build — which
       sits *beside* the payload, not inside it — is a surplus rather than an
       invisible passenger;
    3. :func:`_assert_inventory`: every resource the runtime reads is present,
       re-run over what actually shipped.

    *expected* overrides the comparison set for a caller that already knows it
    (a test staging from a fixture rather than from this checkout); by default
    it is :func:`expected_payload` over the checkout this module lives in, which
    is where ``make opencode-package`` calls it from.
    """
    archive = Path(archive)
    wanted = tuple(expected) if expected is not None else expected_payload(root)
    if not wanted:
        raise PayloadError(
            "the allowlist expansion is empty, so there is nothing to hold "
            f"{archive} to. Either the manifest block failed to parse or this "
            "is not a checkout; pass expected= if the payload is not this "
            "repository's."
        )

    with tempfile.TemporaryDirectory(prefix="payload-scan.") as scratch:
        with tarfile.open(archive, "r:gz") as handle:
            members = [
                member for member in handle.getmembers()
                if member.isfile() and not member.name.startswith(("/", ".."))
            ]
            if hasattr(tarfile, "data_filter"):
                handle.extractall(scratch, members=members, filter="data")
            else:  # pragma: no cover - Python < 3.11.4
                handle.extractall(scratch, members=members)

        tree = Path(scratch)
        scan_tree(tree)

        parts = [part for part in payload_prefix.strip("/").split("/") if part]
        payload_root = tree.joinpath(*parts)
        if not payload_root.is_dir():
            raise PayloadError(
                f"{archive} carries no {payload_prefix}/ — the payload did not "
                "ship at all, which on this route is a plugin that loads and "
                "then raises ENOENT from inside a tool call. Check `files` in "
                "package.json, and that the build ran before the pack."
            )

        container = payload_root.parent
        stem = payload_root.name + "/"
        shipped = {
            str(path.relative_to(container)).replace(os.sep, "/")
            for path in container.rglob("*") if path.is_file()
        }
        target = {stem + relative for relative in wanted}
        missing = sorted(target - shipped)
        surplus = sorted(shipped - target)
        if missing or surplus:
            lines = [f"    - {path}" for path in missing[:20]]
            if len(missing) > 20:
                lines.append(f"    … and {len(missing) - 20} more missing")
            lines += [f"    + {path}" for path in surplus[:20]]
            if len(surplus) > 20:
                lines.append(f"    … and {len(surplus) - 20} more unexpected")
            where = "/".join(parts[:-1]) or "the archive root"
            raise PayloadError(
                f"{archive} is not the staged payload — {len(missing)} "
                f"missing, {len(surplus)} unexpected under {where}:\n"
                + "\n".join(lines)
                + "\nA '-' line is a SHORT tarball: npm honours an ignore file "
                "inside a `files`-listed tree and `files` cannot re-include "
                "against it, so entries can disappear between staging and pack. "
                "A '+' line is something the build never staged — a leftover "
                "staging directory from a killed run is the usual cause."
            )

        _assert_inventory(
            payload_root,
            sorted(path[len(stem):] for path in shipped),
            required,
            required_trees,
        )


# ---------------------------------------------------------------------------
# The digest the publish gate is keyed on
# ---------------------------------------------------------------------------

def tree_digest(
    tree: Path,
    *,
    exclude: Optional[Callable[[str], bool]] = None,
) -> str:
    """A deterministic digest over every file in *tree*.

    ``sorted(path) + sha256(content)``, so it is free of tar metadata,
    mtimes and ownership — the thing being published is the staged *content*,
    and a git tree hash is the wrong key for it (it would miss generated files
    and uncommitted edits alike).

    *exclude* is called with each ``/``-separated tree-relative path and drops
    the ones it returns true for. A staging directory is rarely all-shipped —
    it holds the build's own inputs, the digest file itself, an editor's
    scratch, a ``__pycache__`` written by whichever interpreter last imported
    the builder — and a digest that attests to bytes the distribution never
    carries makes the publish gate's key depend on the environment rather than
    on the artifact. It is a parameter and not a fixed ignore list because
    *which* paths ship is the caller's manifest to know, not this module's.
    """
    tree = Path(tree)
    accumulator = hashlib.sha256()
    for path in sorted(
        (p for p in tree.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(tree)).replace(os.sep, "/"),
    ):
        relative = str(path.relative_to(tree)).replace(os.sep, "/")
        if exclude is not None and exclude(relative):
            continue
        accumulator.update(relative.encode("utf-8"))
        accumulator.update(b"\0")
        accumulator.update(hashlib.sha256(path.read_bytes()).digest())
    return accumulator.hexdigest()


def file_inventory(tree: Path) -> Dict[str, int]:
    """``{relative path: size}`` for every file, for a build's own report."""
    tree = Path(tree)
    return {
        str(path.relative_to(tree)).replace(os.sep, "/"): path.stat().st_size
        for path in sorted(tree.rglob("*")) if path.is_file()
    }
