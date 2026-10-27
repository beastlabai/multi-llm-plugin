#!/usr/bin/env python3
"""Build-context leak guard (C6b rule 1) — the repo-root ``.dockerignore``, proven.

The containerized E2E harness builds with the **repo root** as its context
(``docker build -f skills/multi-llm/tests/container/Dockerfile <repo-root>``),
because ``COPY`` cannot reach above the context and the product under test sits
above ``tests/container/``. Everything in that root — a developer's ``.env``, the
dogfood ``.multi-llm/providers.yaml``, whatever scratch credential file is lying
around — therefore rides along and can be baked into a layer by any ``COPY``.
The repo-root ``.dockerignore`` is what stops that, and it is written
allow-nothing-by-default, which means the *product* is out too until it is named.

An allowlist that has to name the product is only as good as the naming, and
every way of getting it wrong is silent: a trailing comment corrupts the pattern
it decorates, a trailing ``/`` copies an empty directory, a bare parent negation
turns the whole thing back into a denylist, and a widened ``*`` looks exactly
like a green build. So the manifest is *proven*, not reasoned about. These tests
export the real build context from a **clean ``git worktree`` checkout of HEAD**
— clean so no untracked file in the developer's tree can mask a manifest entry
that is missing — with a stub ``FROM scratch`` + ``COPY . /ctx`` Dockerfile and
``--output type=local``, then assert against
``tests/container/context_manifest.txt`` in all three directions:

1. **every named source is present** — a widened ``*``, a missing ``/**`` or a
   new runtime directory nobody added fails here, rather than as an
   ``ImportError`` three phases later;
2. **the planted secrets are absent** — a ``.env`` at the root, a
   ``.multi-llm/`` config, and a scratch ``creds.json`` planted *inside an
   allowlisted tree*, which is the only one of the three that the negations
   could plausibly let through;
3. **nothing outside the manifest is present** — the direction that keeps the
   allowlist from drifting into a denylist as those trees grow.

Docker is needed for the context-build step only, so the module skips itself when
no daemon answers. It deliberately carries **no** ``container`` marker: this is
an ordinary-CI leak guard, not part of the container lane, and a leak guard that
only runs where the container lane runs is a leak guard that does not run.

The working tree is never written to — the plants go into the throwaway worktree,
which is removed in fixture teardown on success and on failure alike.
"""

from __future__ import annotations

import functools
import glob as globlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from .container.docker_probe import docker_skip_reason

#: Repo root: tests/ -> multi-llm/ -> skills/ -> <repo>.
REPO_ROOT = Path(__file__).resolve().parents[3]

DOCKERIGNORE_REL = ".dockerignore"
MANIFEST_REL = "skills/multi-llm/tests/container/context_manifest.txt"
CONTAINER_DOCKERFILE_REL = "skills/multi-llm/tests/container/Dockerfile"

#: Written into every plant so a leak is greppable in the export, not just
#: inferrable from a path listing.
SENTINEL = "sk-planted-canary-abcdefghijklmnopqrstuvwxyz"

#: The three secrets of C6b rule 1, keyed by their repo-root-relative path. The
#: first two sit outside every allowlisted tree (so they prove `*` still holds);
#: the third sits *inside* one, and is the only plant a mis-ordered or missing
#: re-exclusion would actually let through.
PLANTED_SECRETS = {
    ".env": f"OPENAI_API_KEY={SENTINEL}\n",
    ".multi-llm/providers.yaml": f"providers:\n  openai:\n    api_key: {SENTINEL}\n",
    "skills/multi-llm/tests/container/creds.json": json.dumps({"api_key": SENTINEL}) + "\n",
}


# --------------------------------------------------------------------------- #
# Docker probe
# --------------------------------------------------------------------------- #


_SKIP_REASON = docker_skip_reason()

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# --------------------------------------------------------------------------- #
# Manifest parsing and glob matching
# --------------------------------------------------------------------------- #


class ManifestPattern:
    """One manifest line: the glob, its regex, and whether it may match nothing."""

    def __init__(self, pattern: str, optional: bool, lineno: int):
        self.pattern = pattern
        self.optional = optional
        self.lineno = lineno
        self.regex = _glob_regex(pattern)

    def matches(self, rel_path: str) -> bool:
        return self.regex.match(rel_path) is not None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{self.pattern}{' (optional)' if self.optional else ''} @ line {self.lineno}>"


@functools.lru_cache(maxsize=None)
def _glob_regex(pattern: str) -> re.Pattern:
    """Compile a manifest glob the way Docker's matcher reads one.

    ``*`` and ``?`` stop at a separator; ``**`` spans them, and ``**/`` also
    matches zero directories so ``**/creds.json`` catches the root copy too.
    Hand-rolled rather than ``fnmatch``: ``fnmatch`` translates ``*`` to ``.*``,
    which would make ``skills/multi-llm/*.py`` match every ``.py`` in every
    subtree and quietly render direction 3 toothless.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "*":
            run = i
            while run < len(pattern) and pattern[run] == "*":
                run += 1
            if run - i >= 2:
                if pattern[run : run + 1] == "/":
                    out.append("(?:.*/)?")
                    i = run + 1
                else:
                    out.append(".*")
                    i = run
            else:
                out.append("[^/]*")
                i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(char))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _load_manifest(path: Path) -> list[ManifestPattern]:
    """Parse the manifest.

    Unlike ``.dockerignore`` — where a trailing comment is part of the pattern —
    inline ``#`` comments are stripped here, because this file is read by this
    test and never by Docker. A ``# optional`` comment marks a pattern whose
    target does not exist yet (a later phase adds it) and exempts it from
    direction 1 only; it still counts for direction 3.
    """
    patterns: list[ManifestPattern] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        body, _, comment = raw.partition("#")
        body = body.strip()
        if not body:
            continue
        patterns.append(ManifestPattern(body, "optional" in comment.lower(), lineno))
    return patterns


MANIFEST = _load_manifest(REPO_ROOT / MANIFEST_REL)


# --------------------------------------------------------------------------- #
# The clean-checkout context export
# --------------------------------------------------------------------------- #


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
    )


def _export_context(context_dir: Path, dockerfile: Path, dest: Path) -> str:
    """Export the filtered build context to ``dest`` and return the build log.

    The ``local`` exporter is a BuildKit feature, so this runs ``docker buildx
    build`` and falls back to ``DOCKER_BUILDKIT=1 docker build``; both are known
    to work against Docker Engine 28.4, and having the fallback means a host
    without the buildx plugin still runs the guard rather than skipping it.
    """
    common = [
        "--no-cache",
        "--progress=plain",
        "--output",
        f"type=local,dest={dest}",
        "-f",
        str(dockerfile),
        str(context_dir),
    ]
    attempts = (
        (["docker", "buildx", "build", *common], {}),
        (["docker", "build", *common], {"DOCKER_BUILDKIT": "1"}),
    )
    failures = []
    for cmd, extra_env in attempts:
        env = {**os.environ, **extra_env}
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if proc.returncode == 0:
            return proc.stdout + proc.stderr
        failures.append(f"$ {shlex.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    raise AssertionError("could not export the build context:\n\n" + "\n\n".join(failures))


class ExportedContext:
    """The exported context: its root on disk plus every path in it."""

    def __init__(self, root: Path, log: str):
        self.root = root
        self.log = log
        self.paths = self._walk(root)

    @staticmethod
    def _walk(root: Path) -> list[str]:
        found = []
        for path in root.rglob("*"):
            # Symlinks are listed as entries, never followed: a dangling one is
            # still something that shipped.
            if path.is_dir() and not path.is_symlink():
                continue
            found.append(path.relative_to(root).as_posix())
        return sorted(found)


@pytest.fixture(scope="module")
def exported_context(tmp_path_factory) -> ExportedContext:
    """Build the context from a throwaway clean checkout of HEAD.

    ``git worktree add --detach`` rather than the working tree, so an untracked
    file cannot stand in for a manifest entry that is missing from the commit.
    The two files that *govern* the build are then copied in from the working
    tree: they are uncommitted while this phase is in flight, and a guard that
    only works once it is committed cannot be what proves it before the commit.
    """
    tmp = tmp_path_factory.mktemp("build-context")
    worktree = tmp / "checkout"

    added = _git("worktree", "add", "--detach", str(worktree), "HEAD")
    assert added.returncode == 0, f"git worktree add failed:\n{added.stdout}\n{added.stderr}"

    try:
        for rel in (DOCKERIGNORE_REL, MANIFEST_REL):
            src = REPO_ROOT / rel
            assert src.is_file(), f"{rel} is missing from the working tree"
            dst = worktree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

        for rel, body in PLANTED_SECRETS.items():
            plant = worktree / rel
            plant.parent.mkdir(parents=True, exist_ok=True)
            plant.write_text(body, encoding="utf-8")

        # The stub lives outside the context so it cannot itself become an
        # exported path the manifest would then have to know about.
        stub_dir = tmp / "stub"
        stub_dir.mkdir()
        dockerfile = stub_dir / "Dockerfile"
        dockerfile.write_text("FROM scratch\nCOPY . /ctx\n", encoding="utf-8")

        dest = tmp / "export"
        dest.mkdir()
        log = _export_context(worktree, dockerfile, dest)

        ctx_root = dest / "ctx"
        assert ctx_root.is_dir(), f"the export produced no /ctx directory:\n{log}"
        yield ExportedContext(ctx_root, log)
    finally:
        removed = _git("worktree", "remove", "--force", str(worktree))
        if removed.returncode != 0:
            shutil.rmtree(worktree, ignore_errors=True)
            _git("worktree", "prune")


# --------------------------------------------------------------------------- #
# The three directions
# --------------------------------------------------------------------------- #


def test_export_is_not_empty_and_carries_the_product(exported_context):
    """An anchor with a legible failure, ahead of the pattern-by-pattern checks.

    If the builder ever declines to descend into a directory that ``*`` excludes,
    everything below fails at once and in a hundred voices; this one says which
    thing broke. The remedy is written into ``.dockerignore``'s own header.
    """
    paths = set(exported_context.paths)
    assert paths, f"the exported context is empty:\n{exported_context.log}"

    assert "skills/multi-llm/multi_llm.py" in paths, (
        "the dispatcher is missing from the exported context — the builder did not "
        "descend into a directory excluded by `*`. Apply the ancestor-negation "
        "remedy documented in the repo-root .dockerignore, in full."
    )
    utils = [p for p in paths if p.startswith("skills/multi-llm/utils/")]
    assert len(utils) > 50, f"utils/ came through with only {len(utils)} files: {utils[:10]}"
    assert "skills/multi-llm/utils/providers/base.py" in paths, (
        "utils/ did not export recursively — `utils/**` matched only one level"
    )


@pytest.mark.parametrize(
    "entry",
    [p for p in MANIFEST if not p.optional],
    ids=[p.pattern for p in MANIFEST if not p.optional],
)
def test_every_manifest_pattern_is_present(entry, exported_context):
    """Direction 1: nothing the image needs is silently absent.

    This is the assertion a widened ``*``, a missing ``/**`` or a trailing
    ``.dockerignore`` comment dies on — at the commit that introduces it, rather
    than as a runtime ``ImportError`` several phases downstream.
    """
    assert any(entry.matches(path) for path in exported_context.paths), (
        f"manifest line {entry.lineno} ({entry.pattern!r}) matched nothing in the "
        f"exported context. Either the .dockerignore negation for it is missing or "
        f"malformed, or the path moved and both files need updating. "
        f"({len(exported_context.paths)} paths exported.)"
    )


def test_planted_secrets_are_absent(exported_context):
    """Direction 2: the whole reason the file is allow-nothing-by-default."""
    leaked = [rel for rel in PLANTED_SECRETS if rel in set(exported_context.paths)]
    assert not leaked, (
        f"planted secrets reached the build context: {leaked}. "
        "A `*`-excluded plant leaking means the allowlist has a bare parent "
        "negation; the in-tree creds.json leaking means the credential "
        "re-exclusions are missing or ordered above the negations they undo."
    )

    # Belt and braces: the plants carry a sentinel, so a leak that arrives under
    # a different name (a stray copy, a build artifact) is caught too.
    carriers = []
    for rel in exported_context.paths:
        path = exported_context.root / rel
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if SENTINEL in path.read_text(encoding="utf-8", errors="ignore"):
                carriers.append(rel)
        except OSError:  # pragma: no cover - unreadable export entry
            continue
    assert not carriers, f"the planted sentinel reached the build context via: {carriers}"


def test_no_exported_path_is_outside_the_manifest(exported_context):
    """Direction 3: what stops the allowlist drifting back into a denylist.

    Optional patterns count here — a file that exists is in scope whether or not
    the phase that owns it has landed.
    """
    unlisted = [
        rel
        for rel in exported_context.paths
        if not any(entry.matches(rel) for entry in MANIFEST)
    ]
    assert not unlisted, (
        f"{len(unlisted)} exported path(s) match no manifest pattern: {unlisted[:20]}. "
        "Either the .dockerignore is wider than the manifest (add the negation's "
        "twin here, and say why the image needs it) or it is wider than it should "
        "be (tighten the negation)."
    )


# --------------------------------------------------------------------------- #
# Belt and braces: the Dockerfile's own COPY sources
# --------------------------------------------------------------------------- #


def _dockerfile_copy_sources(text: str) -> list[str]:
    """Every context-relative ``COPY`` source in a Dockerfile.

    Line continuations are joined, comments dropped, both the shell and JSON
    argument forms handled, and ``COPY --from=<stage>`` skipped — that one reads
    from a build stage, not from the context this manifest governs.
    """
    logical: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1].rstrip() + " "
            continue
        logical.append(buffer + stripped)
        buffer = ""
    if buffer:
        logical.append(buffer)

    sources: list[str] = []
    for line in logical:
        if not re.match(r"copy\s", line, flags=re.IGNORECASE):
            continue
        rest = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
        flags = []
        while rest.startswith("--"):
            flag, _, rest = rest.partition(" ")
            flags.append(flag)
            rest = rest.lstrip()
        if any(f.startswith("--from=") for f in flags):
            continue
        args = json.loads(rest) if rest.startswith("[") else shlex.split(rest)
        if len(args) < 2:
            continue
        sources.extend(args[:-1])
    return sources


@pytest.mark.skipif(
    not (REPO_ROOT / CONTAINER_DOCKERFILE_REL).is_file(),
    reason=f"{CONTAINER_DOCKERFILE_REL} does not exist yet (a later phase adds it)",
)
def test_dockerfile_copy_sources_resolve_inside_the_manifest():
    """C6b rule 2: the Dockerfile never does ``COPY .``, and never reaches wider.

    Belt and braces with the ``.dockerignore``, which is one careless commit away
    from being widened. Needs no Docker — it reads the file — but shares the
    module's daemon guard, which costs nothing where the container lane runs.
    """
    text = (REPO_ROOT / CONTAINER_DOCKERFILE_REL).read_text(encoding="utf-8")
    sources = _dockerfile_copy_sources(text)
    assert sources, "the container Dockerfile declares no COPY source"

    for source in sources:
        assert source not in (".", "./"), (
            "the container Dockerfile does a bare `COPY .` — named paths only, "
            "drawn from tests/container/context_manifest.txt"
        )
        matched = globlib.glob(str(REPO_ROOT / source), recursive=True)
        if not matched and any(
            entry.optional and (entry.matches(source) or entry.matches(f"{source}/x"))
            for entry in MANIFEST
        ):
            # A build output the manifest marks `# optional` (the staged
            # opencode payload, `make opencode-stage`): absent from a clean
            # checkout, which is what a CI runner has. Its manifest entry is
            # the containment this rule asks for; there is nothing to walk.
            continue
        assert matched, f"COPY source {source!r} resolves to nothing in the repo"

        files: list[str] = []
        for hit in matched:
            hit_path = Path(hit)
            if hit_path.is_dir():
                files.extend(
                    str(child.relative_to(REPO_ROOT).as_posix())
                    for child in hit_path.rglob("*")
                    if child.is_file()
                )
            else:
                files.append(hit_path.relative_to(REPO_ROOT).as_posix())

        outside = [f for f in files if not any(e.matches(f) for e in MANIFEST)]
        assert not outside, (
            f"COPY source {source!r} reaches {len(outside)} path(s) the manifest "
            f"does not name: {outside[:20]}"
        )
