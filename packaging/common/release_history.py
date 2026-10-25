#!/usr/bin/env python3
"""The shared-version release-history check, run by both release families.

**Why one check and not two.** The version owner is
``skills/multi-llm/pyproject.toml``, and three distributions ship it: the Claude
Code route, the Codex plugin and the ``opencode-multi-llm`` npm package. Codex's
own ``--check-bump`` reads only ``codex-v*`` tags (``release_tags()`` in
``packaging/codex/build.py``), so it would pass a ``codex-v0.1.1`` cut from a
``master`` whose shared bytes differ from npm's 0.1.1, and an OpenCode check
reading only ``opencode-v*`` tags would pass an unbumped 0.1.0 over bytes
``codex-v0.1.0`` never shipped. The invariant is across families, so the
history it reads is too (plans/opencode-plugin-publish.md, step 2):

* **A shared version names one shared payload.** Every ``codex-v<V>`` tag,
  every ``opencode-v<V>`` tag and npm's ``opencode-multi-llm@<V>`` carry one
  shared-payload digest.
* **Versions are shared; releases are per family.** Either family may release a
  version the other has not, and may skip one.
* **Choosing a version.** A candidate's version either equals a released
  version, and then its shared-payload digest must equal that release's, or is
  newer than every version either family released. An unreleased version older
  than the newest release is refused.

**The shared-payload digest is read from one commit alone.** It is
:func:`payload.tree_digest`'s rule over the files :func:`payload.expected_payload`
admits under ``skills/multi-llm/``, with the allowlist *and* the bytes taken from
that commit (``git show``, ``git ls-tree``, ``git cat-file``): never from the
working tree, which is not what any tag carries, and never from a checked-in
digest file, which is a claim about bytes rather than the bytes. Each
distribution's own files (Codex's ``.codex-plugin/``, ``.mcp.json`` and
``mcp/``; OpenCode's ``dist/`` and ``package.json``) are outside it: Codex's
own-tree rule and npm's no-reuse rule already cover them. Blob bytes equal a
fresh checkout's on every platform because ``.gitattributes`` pins ``eol=lf``.

**A missing tag is never an empty history.** ``git tag --list`` answers from
whatever this clone fetched, and a shallow CI checkout fetches no tags at all:
read naively, it is a repository that never released anything, and every
version passes as "newer than every release". So the check refuses a shallow
clone outright, and refuses a clone without :data:`FIRST_RELEASE_TAG`, which is
released and therefore in every complete history. Codex's own rules still pass
that state vacuously (an empty ``release_tags()`` is "nothing to compare"); this
is the check that does not.

**It is out of the ``tests`` workflow on purpose.** After V is released,
``master`` may change shipped bytes while still saying V (the post-release
policy); that is unreleased development, and this check refuses a release from
it. Run on every push it would turn ``master`` red over a legal edit, so only
the release paths call it: ``packaging/codex/build.py --check-bump`` and
``packaging/opencode/build.py --check-bump`` (``make opencode-check-bump``).

It reads Git and the registry and writes nothing. Stdlib only, plus
:mod:`common.payload` and the single-registry module.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from common import payload
from common.payload import PayloadError

#: The checkout this module ships in: where the npm package name and the single
#: registry are read from, whatever repository the history is read out of.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The two release families and their tag prefixes. A third distribution that
#: tags releases joins here, or its versions are invisible to the invariant.
FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("codex", "codex-v"),
    ("opencode", "opencode-v"),
)

#: The first release of the shared version, ever. It is in every complete
#: history, so its absence means "this clone has no tags", never "nothing was
#: released" (the shallow-checkout trap the module docstring describes).
FIRST_RELEASE_TAG = "codex-v0.1.0"

#: The version owner, repo-relative (``test_version_source_is_singular``).
VERSION_PATH = "skills/multi-llm/pyproject.toml"

#: The shared payload's prefix: :func:`payload.expected_payload`'s default.
PAYLOAD_PREFIX = "skills/multi-llm"

#: Where the npm package's name is owned.
PACKAGE_JSON = REPO_ROOT / "packaging" / "opencode" / "package.json"

#: The single-registry module (step 0). Loaded by path: it lives beside the
#: OpenCode builder, which is not on the Codex builder's import path.
REGISTRY_MODULE = REPO_ROOT / "packaging" / "opencode" / "registry.py"

#: ``npm view``'s bounds. npm's own retry is bounded (two retries, backoff
#: capped at 5 s), ``--fetch-timeout`` bounds a slow response, and
#: :data:`NPM_VIEW_BUDGET_S` is the backstop for a connect that hangs, which
#: npm's timeout does not cover (``packaging/opencode/bump.py`` measured the
#: same). ``--prefer-online`` revalidates a cached packument: a version
#: published since the last ``npm view`` must not read as absent.
NPM_VIEW_FLAGS: Tuple[str, ...] = (
    "--json",
    "--prefer-online",
    "--fetch-retries=2",
    "--fetch-retry-mintimeout=1000",
    "--fetch-retry-maxtimeout=5000",
    "--fetch-timeout=20000",
)
NPM_VIEW_BUDGET_S = 90

#: semver.org's grammar, ``+build`` accepted. Tag names and npm's list are both
#: held to it; a name that is not a version cannot be compared with one.
SEMVER_RE = re.compile(
    r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)

#: What runs npm: ``(argv, timeout_s) -> CompletedProcess`` with text output.
NpmRunner = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]


class ReleaseHistoryError(PayloadError):
    """The history could not be read, so nothing about it can be concluded.

    A :class:`PayloadError` so both builders' existing ``except`` clauses turn
    it into a named ``REFUSING:`` line rather than a traceback.
    """


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def version_in(pyproject_text: str) -> str:
    """``[project] version`` out of a ``pyproject.toml`` text.

    By hand rather than ``tomllib``, which the skill's 3.10 floor lacks; the
    key is a one-line string by construction. The Codex builder's
    ``project_version`` reads through this too, so the build and the check can
    never parse one file two ways.
    """
    section = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", pyproject_text, re.M | re.S)
    match = section and re.search(r'^version\s*=\s*"([^"]+)"\s*$', section.group(1), re.M)
    if not match:
        raise PayloadError("pyproject.toml declares no [project] version")
    return match.group(1)


def is_version(text: str) -> bool:
    return SEMVER_RE.fullmatch(text) is not None


def release_version(version: str) -> str:
    """*version* without build metadata: the identity a release carries.

    semver ignores ``+build`` for precedence and npm drops it on publish, so
    ``0.1.2`` and ``0.1.2+rebuild`` are one version here, exactly as Codex's
    own history rule already treats them.
    """
    return version.split("+", 1)[0]


def version_key(version: str) -> Tuple[object, ...]:
    """semver precedence: numeric fields, then a release after its own
    pre-releases, pre-release identifiers numeric-before-alphanumeric."""
    match = SEMVER_RE.fullmatch(version)
    if not match:
        raise ReleaseHistoryError(f"{version!r} is not a strict semver version")
    numbers = tuple(int(match.group(index)) for index in (1, 2, 3))
    pre = match.group(4)
    if not pre:
        return numbers + ((1,),)
    identifiers = tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in pre.split(".")
    )
    return numbers + ((0, identifiers),)


# ---------------------------------------------------------------------------
# Git, read-only
# ---------------------------------------------------------------------------

def _git_bytes(root: Path, *args: str, stdin: Optional[bytes] = None) -> bytes:
    """``git -C root <args>``, bytes out. Not :func:`payload._git`, which
    decodes text with the locale's codec and universal newlines: blob bytes
    must reach the digest exactly as committed."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            input=stdin, capture_output=True, check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        raise ReleaseHistoryError("git is required to read the release history") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReleaseHistoryError(f"git {' '.join(args)} failed in {root}: {detail}")
    return result.stdout


def _git_text(root: Path, *args: str) -> str:
    return _git_bytes(root, *args).decode("utf-8")


def resolve_commit(root: Path, ref: str) -> str:
    """The full SHA *ref* names, peeled through an annotated tag."""
    try:
        return _git_text(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()
    except ReleaseHistoryError as exc:
        raise ReleaseHistoryError(f"{ref} does not resolve to a commit in {root}") from exc


def _show(root: Path, commit: str, path: str) -> str:
    try:
        return _git_bytes(root, "show", f"{commit}:{path}").decode("utf-8")
    except ReleaseHistoryError as exc:
        raise ReleaseHistoryError(
            f"commit {commit[:12]} carries no {path}, so its release cannot be read"
        ) from exc


def commit_version(root: Path, commit: str) -> str:
    """The version *commit*'s ``pyproject.toml`` declares."""
    try:
        return version_in(_show(root, commit, VERSION_PATH))
    except ReleaseHistoryError:
        raise
    except PayloadError as exc:
        raise ReleaseHistoryError(f"commit {commit[:12]}: {exc}") from exc


def commit_allowlist(root: Path, commit: str) -> Tuple[str, ...]:
    """The payload allowlist as *commit* defined it.

    Parsed by :func:`payload.runtime_allowlist` itself, over a scratch copy of
    that commit's ``context_manifest.txt``, so the parse has one owner and the
    list one source: a release whose manifest admitted a file the working tree's
    manifest no longer does still digests what it shipped.
    """
    text = _show(root, commit, payload.MANIFEST_RELATIVE)
    with tempfile.TemporaryDirectory(prefix="mllm-allowlist-") as scratch:
        target = Path(scratch) / payload.MANIFEST_RELATIVE
        target.parent.mkdir(parents=True)
        target.write_text(text, encoding="utf-8")
        try:
            return payload.runtime_allowlist(Path(scratch))
        except PayloadError as exc:
            raise ReleaseHistoryError(
                f"commit {commit[:12]}: the allowlist in {payload.MANIFEST_RELATIVE} "
                f"does not parse: {exc}"
            ) from exc


def _admitted_blobs(root: Path, commit: str) -> List[Tuple[str, str]]:
    """``(payload-relative path, blob id)`` for every file the commit's
    allowlist admits under :data:`PAYLOAD_PREFIX`, sorted.

    ``git ls-tree`` over the commit is :func:`payload.selected_files`'s
    ``git ls-files`` over a checkout: tracked paths only, then the allowlist.
    A symlink or submodule among them is refused, as :func:`payload.stage`
    refuses one, because no release could have staged it.
    """
    patterns = commit_allowlist(root, commit)
    listing = _git_bytes(
        root, "ls-tree", "-r", "-z", "--full-tree", commit, "--", PAYLOAD_PREFIX,
    ).decode("utf-8")
    strip = PAYLOAD_PREFIX + "/"
    admitted: List[Tuple[str, str]] = []
    odd: List[str] = []
    for record in listing.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        mode, kind, oid = meta.split(" ")
        if not path.startswith(strip):
            continue
        if not any(payload._matches(path, p) for p in patterns):  # noqa: SLF001
            continue
        if kind != "blob" or mode == "120000":
            odd.append(f"{path} (mode {mode})")
            continue
        admitted.append((path[len(strip):], oid))
    if odd:
        raise ReleaseHistoryError(
            f"commit {commit[:12]}: the payload allowlist admits entries no build "
            "can stage (symlinks or submodules): " + ", ".join(odd)
        )
    if not admitted:
        raise ReleaseHistoryError(
            f"commit {commit[:12]}: the allowlist admits no files under "
            f"{PAYLOAD_PREFIX}/, so there is no shared payload to digest"
        )
    return sorted(admitted)


def _read_blobs(root: Path, oids: Sequence[str]) -> List[bytes]:
    """Every blob in *oids*, in order, through one ``git cat-file --batch``."""
    out = _git_bytes(
        root, "cat-file", "--batch", stdin=("\n".join(oids) + "\n").encode("ascii"),
    )
    blobs: List[bytes] = []
    position = 0
    for oid in oids:
        end = out.index(b"\n", position)
        header = out[position:end].decode("ascii").split()
        if len(header) != 3 or header[1] != "blob":
            raise ReleaseHistoryError(f"git cat-file could not read blob {oid}: {' '.join(header)}")
        size = int(header[2])
        start = end + 1
        blobs.append(out[start:start + size])
        position = start + size + 1  # the LF after each object
    return blobs


def shared_payload_digest(root: Path, commit: str) -> str:
    """The shared-payload digest of *commit*: :func:`payload.tree_digest` over
    the admitted files, written to a scratch tree from that commit's blobs.

    Through a scratch tree rather than a second implementation of the rule, so
    the digest a release is held to and the digest a build computes cannot
    drift apart by construction.
    """
    entries = _admitted_blobs(root, commit)
    blobs = _read_blobs(root, [oid for _, oid in entries])
    with tempfile.TemporaryDirectory(prefix="mllm-shared-payload-") as scratch:
        tree = Path(scratch)
        for (relative, _), data in zip(entries, blobs):
            target = tree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return payload.tree_digest(tree)


# ---------------------------------------------------------------------------
# The two halves of the history: tags, and npm's list
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Release:
    """One release tag, read out of its commit."""

    tag: str
    family: str
    version: str   # the version the tag's name says
    commit: str
    declared: str  # the version the commit's pyproject.toml says
    digest: str


def release_tag_names(root: Path) -> Tuple[str, ...]:
    """Every ``codex-v*`` and ``opencode-v*`` tag this clone has, sorted."""
    patterns = [f"{prefix}*" for _, prefix in FAMILIES]
    out = _git_text(root, "tag", "--list", *patterns)
    return tuple(sorted(line.strip() for line in out.splitlines() if line.strip()))


def history_gap(root: Path, tags: Sequence[str]) -> Optional[str]:
    """Why this clone cannot answer for the history, or ``None``.

    Asked before any comparison, because every comparison would otherwise
    succeed against the empty history a tagless clone presents.
    """
    if _git_text(root, "rev-parse", "--is-shallow-repository").strip() == "true":
        return (
            f"{root} is a shallow clone: its tags and their commits may be missing, "
            "and an incomplete release history reads as one where every version is "
            "new. Check from a full clone with every tag (`git fetch --unshallow "
            "--tags`; actions/checkout `fetch-depth: 0`)."
        )
    if FIRST_RELEASE_TAG not in tags:
        found = ", ".join(tags) if tags else "none"
        return (
            f"{root} has no {FIRST_RELEASE_TAG} tag (release tags found: {found}). "
            f"{FIRST_RELEASE_TAG} is released, so its absence means this clone did "
            "not fetch the tags, not that nothing was released. Run `git fetch "
            "--tags` (actions/checkout `fetch-depth: 0`) and re-run."
        )
    return None


def read_releases(
    root: Path, tags: Sequence[str], digests: Dict[str, str],
) -> Tuple[Tuple[Release, ...], List[str]]:
    """Each tag as a :class:`Release`, oldest version first, plus the tags that
    could not be read (each a refusal: a release the check cannot see is a
    release it cannot hold anything to)."""
    releases: List[Release] = []
    problems: List[str] = []
    for tag in tags:
        family, prefix = next((f, p) for f, p in FAMILIES if tag.startswith(p))
        named = tag[len(prefix):]
        if not is_version(named):
            problems.append(f"tag {tag} does not name a strict semver version")
            continue
        try:
            commit = resolve_commit(root, f"refs/tags/{tag}")
            declared = commit_version(root, commit)
            if commit not in digests:
                digests[commit] = shared_payload_digest(root, commit)
        except ReleaseHistoryError as exc:
            problems.append(f"tag {tag} cannot be read: {exc}")
            continue
        releases.append(Release(tag, family, named, commit, declared, digests[commit]))
    releases.sort(key=lambda r: (version_key(r.version), r.tag))
    return tuple(releases), problems


@dataclass(frozen=True)
class RegistryAnswer:
    """npm's version list for the package, or its documented absence."""

    package: str
    registry: str
    versions: Tuple[str, ...]
    absent: bool


def _run_npm(argv: Sequence[str], timeout: float) -> "subprocess.CompletedProcess[str]":
    """Run npm. ``shutil.which`` first so Windows finds ``npm.cmd``. From an
    empty scratch directory, so no project ``.npmrc`` applies; ``--registry``
    on the command line already outranks a user or environment one."""
    binary = shutil.which(argv[0])
    if binary is None:
        raise ReleaseHistoryError(
            f"{argv[0]} is not on PATH, so npm's version list cannot be read and "
            "the history is incomplete"
        )
    with tempfile.TemporaryDirectory(prefix="mllm-npm-view-") as scratch:
        try:
            return subprocess.run(
                [binary, *argv[1:]], capture_output=True, text=True, encoding="utf-8",
                errors="replace", stdin=subprocess.DEVNULL, timeout=timeout, cwd=scratch,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReleaseHistoryError(
                f"`{' '.join(argv)}` did not answer within {timeout:g} s"
            ) from exc


def npm_view_argv(package: str, registry: str) -> Tuple[str, ...]:
    return ("npm", "view", package, "versions", "--registry", registry, *NPM_VIEW_FLAGS)


def npm_versions(
    package: str, registry: str, *, runner: Optional[NpmRunner] = None,
) -> RegistryAnswer:
    """The versions the registry lists for *package*.

    Three outcomes, never two. Exit 0 with a JSON list (or the bare string an
    older npm printed for one version) is the list. ``E404`` naming the package
    is the documented "no such package", so no versions. Everything else, a
    network error, ``E403``/``E429``/``E5xx``, a timeout, output that does not
    parse, is npm declining to answer, and it raises: read as "no versions" it
    would let a version npm already holds look unreleased.
    """
    argv = npm_view_argv(package, registry)
    run = runner if runner is not None else _run_npm
    done = run(argv, NPM_VIEW_BUDGET_S)
    stdout = (done.stdout or "").strip()
    try:
        parsed = json.loads(stdout) if stdout else None
    except ValueError:
        parsed = None
    if done.returncode == 0:
        listed = [parsed] if isinstance(parsed, str) else parsed
        if not isinstance(listed, list) or not all(isinstance(v, str) for v in listed):
            raise ReleaseHistoryError(
                f"`{' '.join(argv)}` exited 0 but printed no version list: {stdout[:200]!r}"
            )
        bad = [v for v in listed if not is_version(v)]
        if bad:
            raise ReleaseHistoryError(
                f"{registry} lists {package} versions that are not semver: {', '.join(bad)}"
            )
        return RegistryAnswer(package, registry, tuple(listed), absent=False)
    error = parsed.get("error") if isinstance(parsed, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if not code:
        stderr_code = re.search(r"npm (?:error|ERR!) code (\S+)", done.stderr or "")
        code = stderr_code.group(1) if stderr_code else None
    summary = " ".join(str((error or {}).get(key) or "") for key in ("summary", "detail")).strip()
    if code == "E404" and package in summary:
        return RegistryAnswer(package, registry, (), absent=True)
    raise ReleaseHistoryError(
        f"npm could not list {package} versions on {registry}: exit {done.returncode}, "
        f"code {code or '<none>'}"
        + (f" ({summary.splitlines()[0]})" if summary else "")
        + ". Only E404 on the package name means it has none; anything else is an "
        "unknown, not an empty list. Re-run once the registry answers."
    )


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _registry_module():
    spec = importlib.util.spec_from_file_location("mllm_single_registry", REGISTRY_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def resolve_registry(url: Optional[str] = None) -> str:
    """The single registry (``Makefile`` ``OPENCODE_REGISTRY``), or *url* when
    it is that registry; any other URL raises, naming both."""
    module = _registry_module()
    try:
        return module.require_single(url) if url is not None else module.single_registry()
    except module.RegistryError as exc:
        raise ReleaseHistoryError(str(exc)) from exc


def npm_package_name() -> str:
    """``packaging/opencode/package.json``'s ``name``: its one owner."""
    try:
        return json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))["name"]
    except (OSError, ValueError, KeyError) as exc:
        raise ReleaseHistoryError(f"cannot read the npm package name from {PACKAGE_JSON}: {exc}") from exc


@dataclass
class Report:
    """What the check read, and every refusal. Empty ``problems`` is a pass."""

    root: str
    candidate: str = ""
    lines: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)


def check(
    root: Optional[Path] = None,
    *,
    candidate: str = "HEAD",
    registry: Optional[str] = None,
    runner: Optional[NpmRunner] = None,
    package: Optional[str] = None,
) -> Report:
    """Run the release-history check over *root* for *candidate* (a commit).

    Refuses when two tags carrying one version have different shared-payload
    digests, or a tag's name differs from its commit's ``pyproject.toml``
    version; when npm lists a version no ``opencode-v*`` tag names; and when the
    candidate breaks the version-choice rule. Also when the history cannot be
    read: a shallow or tagless clone, an unreadable tag, a registry that does
    not answer. *runner* stands in for npm (the suite's stub registry).
    """
    checkout = Path(root) if root else REPO_ROOT
    url = resolve_registry(registry)
    name = package or npm_package_name()
    report = Report(root=str(checkout), candidate=candidate)

    tags = release_tag_names(checkout)
    gap = history_gap(checkout, tags)
    if gap:
        report.problems.append(gap)
        return report

    digests: Dict[str, str] = {}
    releases, unreadable = read_releases(checkout, tags, digests)
    report.problems.extend(unreadable)
    report.lines.append(
        "release history (every codex-v* and opencode-v* tag, and npm's "
        f"{name} versions on {url}):"
    )
    for item in releases:
        report.lines.append(
            f"  {item.tag:<20} {item.declared:<10} commit {item.commit[:12]}  "
            f"shared-payload digest {item.digest}"
        )

    # 1. One version, one shared payload; and every tag says what it carries.
    by_version: Dict[str, List[Release]] = {}
    for item in releases:
        by_version.setdefault(release_version(item.version), []).append(item)
        if release_version(item.declared) != release_version(item.version):
            report.problems.append(
                f"{item.tag} names {item.version}, but its commit {item.commit[:12]}'s "
                f"{VERSION_PATH} says {item.declared}"
            )
    for version, items in by_version.items():
        if len({item.digest for item in items}) > 1:
            listing = "; ".join(
                f"{item.tag} (commit {item.commit[:12]}) {item.digest}" for item in items
            )
            report.problems.append(
                f"version {version} names more than one shared payload: {listing}. A "
                "shared version names one shared payload; a later release of either "
                "family takes the next unused version instead."
            )

    # 2. npm's list: every version it holds was published from an opencode-v* tag.
    npm: Optional[RegistryAnswer] = None
    try:
        npm = npm_versions(name, url, runner=runner)
    except ReleaseHistoryError as exc:
        report.problems.append(str(exc))
    if npm is not None:
        if npm.absent:
            report.lines.append(f"  npm: {name} is not on {url} (E404): no versions")
        else:
            report.lines.append(f"  npm: {name} lists {', '.join(npm.versions) or 'none'}")
        tagged = {release_version(r.version) for r in releases if r.family == "opencode"}
        for version in npm.versions:
            if release_version(version) not in tagged:
                report.problems.append(
                    f"npm lists {name}@{version} on {url}, but no opencode-v{version} "
                    "tag names it, so its shared bytes cannot be read from Git. If the "
                    "tag exists upstream, `git fetch --tags`; otherwise it was published "
                    "from outside the release path and needs investigating before any "
                    "release."
                )

    # 3. The candidate against everything either family has released.
    try:
        commit = resolve_commit(checkout, candidate)
        version = commit_version(checkout, commit)
        digest = digests.get(commit) or shared_payload_digest(checkout, commit)
    except ReleaseHistoryError as exc:
        report.problems.append(f"the candidate {candidate} cannot be read: {exc}")
        return report
    report.lines.append(
        f"candidate {candidate} (commit {commit[:12]}): version {version}, "
        f"shared-payload digest {digest}"
    )
    if not is_version(version):
        report.problems.append(
            f"the candidate {candidate} declares {version!r} in {VERSION_PATH}, "
            "which is not a strict semver version"
        )
        return report

    released: Dict[str, List[Release]] = {key: list(items) for key, items in by_version.items()}
    for listed in (npm.versions if npm is not None else ()):
        released.setdefault(release_version(listed), [])
    own = release_version(version)
    if own in released:
        carriers = released[own]
        if not carriers:
            report.problems.append(
                f"the candidate carries {own}, which npm lists but no tag names, so "
                "its shared payload cannot be compared with the published one"
            )
        for item in carriers:
            if item.digest != digest:
                report.problems.append(
                    f"{item.tag} carries {own} with another shared-payload digest: "
                    f"{item.tag} (commit {item.commit[:12]}) has {item.digest}, the "
                    f"candidate {candidate} (commit {commit[:12]}) has {digest}. A "
                    "released version names one shared payload: bump "
                    f"{VERSION_PATH} past {_newest(released)}, the newest release."
                )
        if carriers and all(item.digest == digest for item in carriers):
            report.lines.append(
                f"  {own} is released ({', '.join(item.tag for item in carriers)}) "
                "with this same shared payload"
            )
    else:
        newest = _newest(released)
        if newest is not None and version_key(own) <= version_key(newest):
            holders = ", ".join(item.tag for item in released[newest]) or f"npm {name}@{newest}"
            report.problems.append(
                f"the candidate's version {own} is unreleased but older than the newest "
                f"release {newest} ({holders}). A new release takes a version newer "
                f"than every version either family released: bump {VERSION_PATH} past "
                f"{newest}."
            )
        else:
            report.lines.append(
                f"  {own} is newer than every released version (newest: {newest or 'none'})"
            )

    dirty = _git_text(checkout, "status", "--porcelain", "--", PAYLOAD_PREFIX).strip()
    if dirty and candidate == "HEAD":
        report.lines.append(
            f"  note: {PAYLOAD_PREFIX}/ has uncommitted changes; this check grades "
            "the commit, and no release ships uncommitted bytes"
        )
    return report


def _newest(released: Dict[str, List[Release]]) -> Optional[str]:
    return max(released, key=version_key) if released else None


def print_report(
    report: Report, *, extra: Sequence[str] = (), ok: str = "the shared-version check passes",
) -> int:
    """Print *report* (and a caller's own *extra* refusals, first) the way both
    builders' ``--check-bump`` answer: what was read on stdout, each refusal as
    a ``REFUSING:`` line on stderr. Exit status 0 only with no refusal at all."""
    for line in report.lines:
        print(line)
    problems = [*extra, *report.problems]
    for line in problems:
        print(f"REFUSING: {line}", file=sys.stderr)
    if problems:
        return 1
    print(ok)
    return 0
