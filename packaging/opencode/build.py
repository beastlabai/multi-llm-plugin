#!/usr/bin/env python3
"""Stage the ``opencode-multi-llm`` npm package. Deterministic, and never packs.

Run it as::

    uv run --project skills/multi-llm -- python packaging/opencode/build.py

**What it produces**, all of it gitignored:

* ``dist/index.mjs`` — a **verbatim copy** of ``src/index.mjs``. Not a bundle and
  not a render: there is nothing to bundle (the package declares no
  dependencies), and nothing authored is substituted into it. That copy is what
  ``main``/``exports`` resolve and what ships, which is why the suite's greps can
  read the committed ``src/`` file and still be grading the shipped bytes —
  ``test_dist_mirrors_src`` is what licenses that.
* ``dist/command.md`` — the slash-command body, rendered through
  ``host_install.render()`` with ``{SKILL_DIR}`` deliberately **intact** for the
  plugin to resolve at startup.
* ``dist/cli-surface.json`` — everything the plugin needs to know about the CLI,
  generated from ``modes.py`` and ``utils/cli_common.py``. The plugin holds no
  mode word, no alias, no summary and no flag name of its own.
* ``skills/multi-llm/`` — the runtime payload, staged by allowlist.
* ``README.md`` / ``LICENSE`` — copied, never symlinked.
* ``dist/tree.sha256`` — a digest over exactly the files above and nothing
  else, which is what the publish gate's stamp is keyed on. Not over the whole
  staging directory: ``src/index.mjs`` is an input rather than an artifact, and
  a ``__pycache__`` written by whichever interpreter last imported this module
  would make the gate's key depend on the machine instead of on the package.

**What it deliberately does not do: it never invokes ``npm``.** Packing and
publishing are the Makefile's, so there is exactly one place each happens and a
build cannot accidentally produce a second tarball nobody publishes. The tarball
secret sweep is *exposed* here (:func:`scan_tarball`) and called by
``make opencode-package`` **after** ``npm pack`` — same module, one step later in
the sequence, because what npm writes is what ships and a ``prepack`` hook runs
too early to see a tarball. The one npm call it does make is a read:
``--check-bump`` asks the registry for the package's version list
(:func:`check_bump`), and packs nothing.

**It is an importable module with a thin ``__main__``**, and that is a design
requirement rather than a test-time convenience: ``dist/`` is gitignored and CI
never builds, so the suite imports :func:`cli_surface` and :func:`command_body`
and renders in-process. A red row then can never mean "someone forgot to
rebuild".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "multi-llm"

# The skill tree is the library this build reads its single sources out of, and
# ``packaging/common`` is the shared staging module. Neither is installed.
for _path in (str(SKILL_DIR), str(REPO_ROOT / "packaging")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import payload, surface  # noqa: E402
from common import release_history  # noqa: E402
from common.payload import (  # noqa: F401  (re-exported for the Make targets)
    PayloadError, scan_tarball, scan_tree, target_module_paths, tree_digest,
)
from utils import host_install  # noqa: E402

#: The host this package is a distribution of. Not a new host id — the harness
#: capabilities are the harness's, however we got installed.
HOST_ID = "opencode"

#: The token the rendered command body carries in place of an absolute prefix.
#: The plugin resolves it from ``import.meta.dirname`` at startup, which is the
#: whole point of the route: the skill tree and the entry point ship in the same
#: directory, so there is no baked path to go stale.
SKILL_DIR_PLACEHOLDER = surface.SKILL_DIR_PLACEHOLDER

#: The digest file's own name inside ``dist/``. Written last, and it can never
#: attest to itself.
DIGEST_NAME = "tree.sha256"

#: Exactly what ``dist/`` holds after a build — asserted, never assumed.
#: ``package.json`` says ``files: ["dist"]``, so npm packs the directory
#: wholesale and a file this build cannot account for is a file that ships.
DIST_FILES: Tuple[str, ...] = (
    "cli-surface.json", "command.md", "index.mjs", DIGEST_NAME,
)

#: What ships, and therefore exactly what :data:`DIGEST_NAME` attests to:
#: ``package.json``'s ``files`` list, plus the manifest npm always packs. The
#: staging directory is a superset of it — when a build stages in place it *is*
#: ``packaging/opencode/``, holding this module, ``src/``, a ``__pycache__``,
#: possibly a ``node_modules/`` from the dev dependency and the previous
#: ``.tgz`` — and a digest over that superset attests to bytes npm never packs
#: while making the publish gate's key environment-dependent: a ``.pyc`` is
#: interpreter-version-specific, so a Python-minor switch between the gate and
#: ``make opencode-publish-first`` would produce a spurious "staged tree has changed".
SHIPPED_TREES: Tuple[str, ...] = ("dist/", "skills/")
SHIPPED_FILES: Tuple[str, ...] = ("LICENSE", "README.md", "package.json")

#: Everything **outside** ``skills/multi-llm`` that decides what this package
#: ships, and is therefore outside :func:`payload.stage`'s dirty-tree refusal.
#: Two kinds, and both matter: ``src/index.mjs`` is copied verbatim into
#: ``dist/`` while the manifest, the README and the LICENSE are packed as they
#: are — and ``build.py`` and ``common/payload.py`` are the modules whose bytes
#: *generate* ``dist/command.md``, ``dist/cli-surface.json`` and the staged
#: payload tree, so an uncommitted edit to either ships just as surely as an
#: uncommitted edit to the plugin source. Without them the refusal has a hole
#: the stamp cannot close — the stamp only catches an edit made *after* the
#: gate, so a change made before it runs would be staged, digested, stamped and
#: published with nothing refusing it.
#:
#: Named paths rather than the whole ``packaging/opencode`` prefix, and that is
#: a decision rather than an oversight: a developer who runs ``npm install``
#: there for the ``@opencode-ai/plugin`` types leaves an untracked
#: ``node_modules/`` and ``package-lock.json`` behind — neither is gitignored
#: and neither can ship (``files`` names ``dist``, ``skills``, ``README.md``
#: and ``LICENSE``, and npm excludes ``node_modules`` from every pack) — so a
#: prefix-wide refusal would block a publish over bytes the tarball cannot
#: carry. Everything it *can* carry is on this list; ``dist/`` and ``skills/``
#: are this build's own output, and ``.tgz``/``__pycache__``/the stamp are
#: gitignored.
PACKAGING_INPUTS: Tuple[str, ...] = (
    "packaging/opencode/src",
    "packaging/opencode/package.json",
    "packaging/opencode/README.md",
    "packaging/opencode/build.py",
    "packaging/common/payload.py",
    "packaging/common/surface.py",
    "LICENSE",
)

#: The surface's shape version. Owned by :mod:`common.surface`, where its
#: version history lives; re-exported so the suite and the plugin's lockstep
#: constant have one name to compare against.
SURFACE_SCHEMA_VERSION = surface.SURFACE_SCHEMA_VERSION

#: Opt-in that lets a staging run proceed from a dirty tree.
#:
#: An *environment* variable and not only a flag, because ``npm pack`` invokes
#: the ``prepack`` hook itself with a fixed command line — so a flag on the Make
#: recipe cannot reach it, and without this a developer's local pack refuses on
#: their own uncommitted work while the recipe that called it had already been
#: told to allow it. It is deliberately a named opt-in rather than a fallback:
#: ``make opencode-publish-first`` neither sets nor forwards it, so a dirty publish is
#: refused however it was reached.
#:
#: Read by *exact* spelling — ``=1`` and nothing else — rather than as "any
#: non-empty value", which is the spelling the Makefile writes
#: (``$(if $(ALLOW_DIRTY),1,)``) and the only reading under which turning the
#: opt-in off works: a developer who exports ``MULTI_LLM_ALLOW_DIRTY=0`` to
#: disable it must not thereby enable it, least of all on a variable npm's
#: ``prepack`` hook inherits.
ALLOW_DIRTY_ENV = "MULTI_LLM_ALLOW_DIRTY"
ALLOW_DIRTY_TRUE = "1"

#: Which template ``dist/command.md`` is rendered from. Through Phase 1 this was
#: the file route's shim; from Phase 2 it is the authored one-line delegation,
#: which is a change of *template* and never a post-render trim — that is what
#: keeps the artifact a faithful render of a single source across the switch.
#: ``hosts/opencode/shim.md.tmpl`` is untouched and stays the file route's shim.
COMMAND_TEMPLATE = "hosts/opencode/plugin-command.md.tmpl"


# ---------------------------------------------------------------------------
# dist/command.md
# ---------------------------------------------------------------------------

def command_body(base: Optional[Path] = None) -> str:
    """The slash-command body, ``{SKILL_DIR}`` intact.

    ``render()`` already takes a ``skill=`` parameter — the absolute prefix baked
    into the result, defaulting to ``base`` — so passing the token *as* the
    prefix makes the substitution an identity and no sentinel round-trip is
    needed. ``base`` stays the real skill dir, so the capability data and
    ``hosts/uv-missing.txt`` are still read from the true tree; only the baked
    prefix becomes the token. It also folds the other skill-dir spellings onto
    ``{SKILL_DIR}``, which is exactly what the plugin wants: one token to
    resolve.
    """
    root = Path(base) if base else SKILL_DIR
    template = (root / COMMAND_TEMPLATE).read_text(encoding="utf-8")
    return host_install.render(
        HOST_ID, base=str(root), skill=SKILL_DIR_PLACEHOLDER, template=template
    )


# ---------------------------------------------------------------------------
# dist/cli-surface.json — the emitter is packaging/common/surface.py
# ---------------------------------------------------------------------------

def cli_surface(base: Optional[Path] = None) -> Dict[str, Any]:
    """This package's surface: :func:`surface.cli_surface` for :data:`HOST_ID`.

    *base* is accepted for signature symmetry with :func:`command_body` and is
    deliberately unused — the surface is a function of the imported tables and
    the host id alone (D3e), which is what lets the codex plugin ship the same
    emitter.
    """
    return surface.cli_surface(HOST_ID)


def cli_surface_json(base: Optional[Path] = None) -> str:
    """:func:`cli_surface` serialized exactly as the build writes it."""
    return surface.cli_surface_json(HOST_ID)


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def build(
    *,
    destination: Optional[Path] = None,
    base: Optional[Path] = None,
    root: Optional[Path] = None,
    allow_dirty: bool = False,
) -> Dict[str, Any]:
    """Produce the staged package tree. Returns a small report.

    *destination* defaults to the package directory itself, which is what a
    publish packs; the suite points it at a ``tmp_path`` so a developer's stale
    ``dist/`` can neither hide a failure nor cause one.
    """
    out = Path(destination) if destination else PACKAGE_DIR
    skill_base = Path(base) if base else SKILL_DIR
    checkout = Path(root) if root else REPO_ROOT

    # 0. The inputs outside the payload prefix are clean too.
    #    ``payload.stage`` refuses a dirty ``skills/multi-llm`` at step 4, and
    #    these are the files outside it that decide what ships — the ones copied
    #    or packed verbatim, and the two modules that generate the rest.
    #    Asserted first, so a refusal lands before ``dist/`` is emptied and a
    #    developer's previous build survives it.
    if not allow_dirty:
        for path in PACKAGING_INPUTS:
            payload.assert_clean(checkout, path)

    dist = out / "dist"
    # Emptied first, and then accounted for at the end. The payload tree gets
    # both guarantees from ``payload.stage``'s temp-and-swap; ``dist/`` is
    # written in place, so it gets them from a clean slate plus an inventory
    # assertion. Without the clean slate an artifact an earlier build wrote
    # under a name this one no longer uses simply stays, and ``files: ["dist"]``
    # packs it — while step 6's digest is computed *over* it, so the stamp and
    # the check agree with each other and neither can see it.
    shutil.rmtree(dist, ignore_errors=True)
    dist.mkdir(parents=True, exist_ok=True)

    # 1. The plugin source, copied verbatim. No bundling, no transform.
    source = PACKAGE_DIR / "src" / "index.mjs"
    shutil.copy2(source, dist / "index.mjs")

    # 2. The command body, {SKILL_DIR} intact. ``newline="\n"`` on every file
    #    this function writes: text mode would turn each "\n" into "\r\n" on
    #    Windows, and the tree digest below is taken over bytes, so the same
    #    source would build a package with a different digest there.
    (dist / "command.md").write_text(
        command_body(skill_base), encoding="utf-8", newline="\n"
    )

    # 3. The generated CLI surface.
    (dist / "cli-surface.json").write_text(
        cli_surface_json(skill_base), encoding="utf-8", newline="\n"
    )

    # 4. The payload, by allowlist, with the sweep and the inventory assertion.
    #    ``required_extra`` is the dispatch table's own required set — every
    #    ``modes.py`` target_module, derived rather than hand-typed — so a mode
    #    row naming a module the static REQUIRED_INVENTORY never anticipated
    #    still fails the build if that module goes untracked.
    staged = payload.stage(
        checkout,
        out / "skills" / "multi-llm",
        allow_dirty=allow_dirty,
        required_extra=target_module_paths(skill_base),
    )

    # 5. LICENSE and README — copied, never symlinked. A symlink inside an npm
    #    tarball is a portability hazard, and ``files`` names both.
    shutil.copy2(checkout / "LICENSE", out / "LICENSE")
    if out != PACKAGE_DIR:
        shutil.copy2(PACKAGE_DIR / "README.md", out / "README.md")
        shutil.copy2(PACKAGE_DIR / "package.json", out / "package.json")

    # 6. The digest the publish gate is keyed on. Written last and computed
    #    over the shipped set alone, so it can neither include itself nor
    #    attest to an input. The stamp, the previous ``.tgz`` and any
    #    ``__pycache__`` fall out of it by the same rule rather than by name:
    #    npm does not pack them.
    accumulator = tree_digest(out, exclude=_not_shipped)
    (dist / DIGEST_NAME).write_text(
        accumulator + "\n", encoding="utf-8", newline="\n"
    )

    # 7. And dist/ is exactly what those steps wrote. The digest cannot stand in
    #    for this check, because a stale file is *inside* the digest.
    _assert_dist_inventory(dist)

    return {
        "destination": str(out),
        "payload_files": len(staged),
        "digest": accumulator,
        "command_bytes": (dist / "command.md").stat().st_size,
    }


def _assert_dist_inventory(dist: Path) -> None:
    """``dist/`` holds :data:`DIST_FILES` and nothing else, or raise.

    The same accounting :func:`payload._assert_no_unexpected` does for the
    payload tree, for the same reason and against the same failure: npm packs
    ``dist`` wholesale. Named in both directions — a surplus file ships, a
    missing one is a half-written package — because "dist is wrong" is not an
    actionable message.
    """
    actual = {
        str(path.relative_to(dist)).replace(os.sep, "/")
        for path in dist.rglob("*") if path.is_file()
    }
    expected = set(DIST_FILES)
    surplus = sorted(actual - expected)
    missing = sorted(expected - actual)
    if not surplus and not missing:
        return
    parts = []
    if surplus:
        parts.append("unexpected: " + ", ".join(surplus))
    if missing:
        parts.append("missing: " + ", ".join(missing))
    raise PayloadError(
        f"{dist} is not the tree this build produced ({'; '.join(parts)}). "
        "`files: [\"dist\"]` packs the directory wholesale, so an unaccounted "
        "file ships."
    )


def _not_shipped(relative: str) -> bool:
    """Is *relative* a staged path npm will not pack?

    Spelled as an exclusion because that is what :func:`payload.tree_digest`
    takes, but the rule is an allowlist: :data:`SHIPPED_TREES` and
    :data:`SHIPPED_FILES` are the manifest, and everything else — inputs,
    caches, the digest file itself — is out. A closed list is the point: a file
    that appears in the package directory years from now is excluded by
    default, so it can never quietly become part of the publish gate's key.
    """
    if relative == f"dist/{DIGEST_NAME}":
        return True
    return not (
        relative in SHIPPED_FILES or relative.startswith(SHIPPED_TREES)
    )


# ---------------------------------------------------------------------------
# A packed tarball, as the stamp and the publish preflight bind it
# ---------------------------------------------------------------------------

#: The prefix npm writes every member of a packed tarball under.
TARBALL_PREFIX = "package/"


def integrity_of(data: bytes) -> str:
    """npm's ``dist.integrity`` spelling of *data*: ``sha512-<base64>``.

    The registry computes this over the uploaded file's bytes, so a tarball
    published as a file (``npm publish <file>.tgz``) reads back with exactly the
    value this returns for the file.
    """
    import base64
    import hashlib

    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def tarball_facts(tarball: Path) -> Dict[str, Any]:
    """What a packed ``.tgz`` is: its integrity, its inventory, its tree digest.

    Returns ``{"path", "integrity", "sha256", "files", "digest",
    "recorded_digest"}``:

    * ``integrity`` — :func:`integrity_of` over the file, comparable with
      ``npm view <spec> dist.integrity`` as it stands;
    * ``files`` — the sorted member paths with npm's ``package/`` prefix
      removed, regular files only;
    * ``digest`` — this builder's own tree-digest rule (:func:`tree_digest`
      with :func:`_not_shipped`) over the extracted package, which equals
      ``dist/tree.sha256`` of the build that was packed;
    * ``recorded_digest`` — the ``dist/tree.sha256`` the tarball carries, or
      ``None`` when it carries none.

    Raises :class:`PayloadError` for a member outside ``package/``, an absolute
    or ``..`` path, a link or device member, or a file that is not a gzip tar —
    an archive this function cannot account for is never described as one.
    Extraction happens in a private temporary directory that is removed before
    returning.
    """
    import hashlib
    import tarfile
    import tempfile

    path = Path(tarball)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PayloadError(f"{path}: cannot read the tarball ({exc})") from exc
    files: List[str] = []
    with tempfile.TemporaryDirectory(prefix="mllm-tarball-") as scratch:
        root = Path(scratch)
        try:
            archive = tarfile.open(path, mode="r:gz")
        except (tarfile.TarError, OSError) as exc:
            raise PayloadError(f"{path}: not a gzip tarball ({exc})") from exc
        with archive:
            for member in archive.getmembers():
                name = member.name
                parts = name.split("/")
                if name.startswith("/") or ".." in parts or not name.startswith(TARBALL_PREFIX):
                    raise PayloadError(
                        f"{path}: member {name!r} is not under {TARBALL_PREFIX!r}"
                    )
                if member.isdir():
                    continue
                if not member.isfile():
                    raise PayloadError(
                        f"{path}: member {name!r} is a link or device, which no "
                        "build of this package produces"
                    )
                relative = name[len(TARBALL_PREFIX):]
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise PayloadError(f"{path}: member {name!r} has no content")
                with source:
                    target.write_bytes(source.read())
                files.append(relative)
        digest = tree_digest(root, exclude=_not_shipped)
        recorded_path = root / "dist" / DIGEST_NAME
        recorded = (recorded_path.read_text(encoding="utf-8").strip()
                    if recorded_path.is_file() else None)
    return {
        "path": str(path),
        "integrity": integrity_of(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "files": sorted(files),
        "digest": digest,
        "recorded_digest": recorded,
    }


# ---------------------------------------------------------------------------
# The shared-version bump check (plans/opencode-plugin-publish.md, step 2)
# ---------------------------------------------------------------------------

def check_bump(
    *, repo: Optional[Path] = None, candidate: str = "HEAD",
    registry: Optional[str] = None,
) -> "release_history.Report":
    """OpenCode's entry point for the release-history check both families run.

    The version is ``pyproject.toml``'s, shared with the Claude and Codex
    distributions, so "is this version new?" has no OpenCode-only answer: the
    check reads every ``codex-v*`` and ``opencode-v*`` tag and npm's list for
    this package, and holds *candidate*'s shared-payload digest to theirs
    (:mod:`common.release_history`). This package's own bytes outside the
    payload (``dist/``, ``package.json``) are npm's no-reuse rule's to police.

    *repo* is the clone whose history is read (default: this one), which is how
    the suite points the real CLI at a fixture history. It never packs, writes
    or fetches; a shallow or tagless clone is refused rather than read as one
    with no releases.
    """
    return release_history.check(
        repo or REPO_ROOT, candidate=candidate, registry=registry,
    )


# ---------------------------------------------------------------------------
# The publish gate's stamp
# ---------------------------------------------------------------------------

#: Where the full gate records that it passed. Gitignored, and **it never
#: travels**: it is not committed and is never uploaded as a CI artifact, because
#: a stamp from another machine attests to that machine's rows rather than to
#: this one's. CI consumes no stamps at all: its jobs re-run the rows they can
#: (1-7 and 9), and row 8 — which needs a Docker daemon the runners do not have
#: — reaches CI only through the release-gate record exported from the same
#: session that authorized this stamp (``release_gate.py export-record``, into
#: the gitignored ``.gate/export/``). It is copied into
#: ``tests/container/records/`` and committed in the default branch's own commit
#: after the fast-forward, never in the tag's tree; the tag's commit is then
#: checked out again, clean, before ``make opencode-publish-first`` reads this
#: stamp.
STAMP_NAME = ".package-check.stamp"

#: The tier a stamp must carry to authorize a publish. A `-ci` run writes no
#: stamp at all, and a stamp claiming any other tier is refused by name.
FULL_TIER = "full"

#: The stamp's shape. Schema 1 was ``{tier, digest, version, stamped_at}``: a
#: digest-only stamp ``--write-stamp`` wrote without reading any gate evidence,
#: so recipe ordering was all that stood between it and rows nobody ran.
#: Schema 2 embeds the gate session's checked summary and binds the candidate
#: commit, the host, the image, the registry and the gated tarball; a schema-1
#: stamp is refused by name.
STAMP_SCHEMA = 2

#: The bindings a schema-2 stamp must carry, each re-checked by
#: :func:`check_stamp` against what is true now.
STAMP_BINDINGS: Tuple[str, ...] = (
    "digest", "version", "session", "candidate_sha", "host_version", "image_id",
    "registry", "tarball", "rows",
)
TARBALL_BINDINGS: Tuple[str, ...] = ("path", "integrity", "sha256", "files", "digest")


def _gate():
    """``packaging/opencode/release_gate.py``, loaded by path as
    ``opencode_release_gate``: the codex route has a ``release_gate`` too, and
    one pytest session imports both under their bare names."""
    import importlib.util

    name = "opencode_release_gate"
    path = (PACKAGE_DIR / "release_gate.py").resolve()
    main = sys.modules.get("__main__")
    if getattr(main, "__file__", None) and Path(main.__file__).resolve() == path:
        return main
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _manifest_at(out: Path) -> Dict[str, Any]:
    source = out / "package.json"
    if not source.is_file():
        source = PACKAGE_DIR / "package.json"
    return json.loads(source.read_text(encoding="utf-8"))


def write_stamp(
    destination: Optional[Path] = None, *, timestamp: str = "",
    allow_dirty: bool = False, evidence_dir: Optional[Path] = None,
    tarball: Optional[Path] = None,
) -> Path:
    """Record that the full gate passed **on these exact bytes**, in one session.

    Keyed on ``dist/tree.sha256`` — a digest over the staged content — rather
    than on a git tree hash, which is the wrong key twice over: the thing being
    published is the *staged* tree, which includes generated files ``.gitignore``
    excludes, and a git hash would miss uncommitted edits entirely.

    **It reads the gate's evidence before it writes anything**, and refuses
    unless the running session (``release_gate.py begin``) holds a passing row
    for every row of the plan's table — 1-6 and 5b graded node by node, row 7's
    probe on the pinned host and the gated tarball, row 8's five scenarios on
    the image the session built and bound, row 9's ``E404`` from the single
    registry — over this digest, this version and the commit checked out now.
    A rehearsal session (``ALLOW_DIRTY=1``, or a dirty tree at ``begin``) and a
    missing session refuse first, by name, before the rebuild. Calling it
    directly, with no gate run, therefore refuses: the stamp is the gate's
    record, so it cannot exist for rows that never ran.

    ``tarball`` is the gated tarball (default: the ``<name>-<version>.tgz``
    ``make opencode-package`` packed). The stamp binds its path, integrity,
    sha256, inventory and tree digest, so a later pack that overwrites it, or
    an upload of any other file, fails :func:`check_stamp`.

    ``allow_dirty`` is a parameter and **not** read from the environment here,
    unlike ``main``'s flag. The environment variable exists for one reason — npm
    invokes ``prepack`` with a fixed command line, so a flag cannot reach it —
    and this function is not on that path. Reading it here would mean a developer
    who exported it once could stamp, and therefore publish, from a dirty tree.
    It relaxes the build's clean-input check and the whole-tree one, for the
    suite's temp-directory stamps; it never makes a rehearsal session count.
    """
    gate = _gate()
    directory = Path(evidence_dir) if evidence_dir else gate.EVIDENCE_DIR
    evidence = gate.load_evidence(directory)
    early = gate.session_problems(evidence.get("session"))
    if early:
        raise PayloadError("\n".join(early))
    out = Path(destination) if destination else PACKAGE_DIR
    report = build(destination=out, allow_dirty=allow_dirty)
    version = _manifest_at(out)["version"]
    gated = Path(tarball) if tarball else gate.gated_tarball()
    facts = tarball_facts(gated) if gated.is_file() else None
    try:
        problems = gate.evidence_problems(
            evidence, digest=report["digest"], version=version,
            head=gate.head_sha(), host_version=gate.host_pin(),
            registry=gate.single_registry(), tarball=facts,
            dirty=() if allow_dirty else gate.dirty_paths(),
        )
    except gate.GateError as exc:
        problems = [str(exc)]
    if problems:
        raise PayloadError(
            "the release gate's evidence does not authorize a stamp over these "
            "bytes:\n" + "\n".join(f"    {line}" for line in problems)
            + "\nRun `make opencode-package-check`; it writes this evidence and "
            "the stamp."
        )
    summary = gate.stamp_summary(evidence)
    stamp = out / STAMP_NAME
    stamp.write_text(
        json.dumps({
            "schema": STAMP_SCHEMA,
            "tier": FULL_TIER,
            "digest": report["digest"],
            "version": version,
            "stamped_at": timestamp,
            "session": summary["session"],
            "candidate_sha": summary["candidate_sha"],
            "host_version": evidence["rows"]["7"]["opencode_version"],
            "image_id": summary["image"]["id"],
            "registry": evidence["rows"]["9"]["registry"],
            "tarball": {key: facts[key] for key in TARBALL_BINDINGS},
            "rows": summary["rows"],
            "gate": summary,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return stamp


def check_stamp(
    destination: Optional[Path] = None, *, allow_dirty: bool = False,
) -> Optional[str]:
    """``None`` if a publish may proceed, else the reason it may not.

    Every condition is **named separately**, so "the gate is red" is never the
    same message as "you edited a file since". A gate is worth nothing if the
    publish command can skip it, and a refusal nobody can act on is nearly as
    bad. In order: no stamp; unreadable; not the full tier; a schema-1
    (digest-only) stamp; a row the stamp does not record as passed, or a
    binding it lacks; the clean rebuild's digest; the version; the candidate
    commit (HEAD now); the pinned host; the single registry; and the gated
    tarball — present, with the stamped integrity, sha256, inventory and tree
    digest recomputed now.
    """
    out = Path(destination) if destination else PACKAGE_DIR
    stamp_path = out / STAMP_NAME
    if not stamp_path.is_file():
        return (
            f"no {stamp_path}. Run `make opencode-package-check` first — the "
            "full gate is what writes it."
        )
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return f"{stamp_path} is not readable JSON ({exc}). Re-run the full gate."
    if not isinstance(stamp, dict):
        return f"{stamp_path} is not a JSON object. Re-run the full gate."
    if stamp.get("tier") != FULL_TIER:
        return (
            f"the stamp says tier={stamp.get('tier')!r}. Only a full "
            "`make opencode-package-check` authorizes a publish; the CI tier "
            "runs rows 1-6 and 5b and deliberately writes no stamp."
        )
    if stamp.get("schema") != STAMP_SCHEMA:
        return (
            f"the stamp is schema {stamp.get('schema', 1)!r}, a digest-only stamp "
            "written before the gate read any evidence; it names no session, "
            "rows, host, image or tarball. Re-run `make opencode-package-check`."
        )
    gate = _gate()
    rows = stamp.get("rows") if isinstance(stamp.get("rows"), dict) else {}
    missing_rows = [row for row in gate.ROW_IDS if rows.get(row) != "pass"]
    if missing_rows:
        return (
            f"the stamp does not record rows {', '.join(missing_rows)} as passed; "
            "it is partial or was written by hand. Re-run "
            "`make opencode-package-check`."
        )
    tarball = stamp.get("tarball") if isinstance(stamp.get("tarball"), dict) else {}
    missing = [key for key in STAMP_BINDINGS if not stamp.get(key)] + [
        f"tarball.{key}" for key in TARBALL_BINDINGS if not tarball.get(key)
    ]
    if missing:
        return (
            f"the stamp lacks {', '.join(missing)}; it was not written by the "
            "evidence-bound gate. Re-run `make opencode-package-check`."
        )
    fresh = build(destination=out, allow_dirty=allow_dirty)["digest"]
    if stamp.get("digest") != fresh:
        return (
            f"the staged tree has changed since the gate ran (stamp "
            f"{stamp.get('digest')}, now {fresh}). Re-run "
            "`make opencode-package-check`."
        )
    version = _manifest_at(out)["version"]
    if stamp.get("version") != version:
        return (
            f"the stamp is for version {stamp.get('version')!r} but package.json "
            f"says {version!r}."
        )
    try:
        head = gate.head_sha()
        host = gate.host_pin()
        registry = gate.single_registry()
    except gate.GateError as exc:
        return str(exc)
    if stamp.get("candidate_sha") != head:
        return (
            f"the stamp is for commit {stamp.get('candidate_sha')}, HEAD is "
            f"{head}: the gate ran on another candidate. Re-run it here."
        )
    if stamp.get("host_version") != host:
        return (
            f"the stamp's rows ran opencode {stamp.get('host_version')!r}, the "
            f"pinned host is now {host}. Re-run the gate on the new pin."
        )
    if stamp.get("registry") != registry:
        return (
            f"the stamp's row 9 asked {stamp.get('registry')!r}, the single "
            f"registry is {registry}."
        )
    gated = Path(str(tarball["path"]))
    if not gated.is_file():
        return (
            f"the gated tarball {gated} is missing: a publish uploads the file "
            "the gate packed, scanned and loaded, and no other."
        )
    try:
        now = tarball_facts(gated)
    except PayloadError as exc:
        return f"the gated tarball {gated} is unreadable: {exc}"
    for key, label in (("integrity", "integrity"), ("sha256", "sha256"),
                       ("files", "file inventory"), ("digest", "tree digest")):
        if now[key] != tarball.get(key):
            return (
                f"the gated tarball {gated} no longer has the stamped {label} "
                "(it was repacked or replaced since the gate ran). Re-run "
                "`make opencode-package-check`."
            )
    if tarball.get("digest") != stamp.get("digest"):
        return (
            f"the stamped tarball's tree digest {tarball.get('digest')} is not "
            f"the stamped candidate digest {stamp.get('digest')}."
        )
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Stage the opencode-multi-llm npm package. Never packs.",
    )
    parser.add_argument(
        "--destination", type=Path, default=None,
        help="where to stage (default: the package directory itself)",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        default=os.environ.get(ALLOW_DIRTY_ENV) == ALLOW_DIRTY_TRUE,
        help=(
            "stage from a tree with uncommitted changes. For a local test only "
            f"— a publish refuses a dirty build. Also settable as "
            f"${ALLOW_DIRTY_ENV}={ALLOW_DIRTY_TRUE} (that exact spelling, so "
            "`=0` turns it off), which is how npm's own `prepack` hook — "
            "invoked by npm rather than by us — inherits the intent."
        ),
    )
    parser.add_argument(
        "--write-stamp", metavar="TIMESTAMP", default=None,
        help=(
            "record that the full gate passed on the freshly staged bytes; "
            "refuses unless this gate session's evidence covers every row"
        ),
    )
    parser.add_argument(
        "--check-stamp", action="store_true",
        help=(
            "exit non-zero unless the full gate passed on these exact bytes, "
            "this commit, host, registry and gated tarball, naming which "
            "condition failed"
        ),
    )
    parser.add_argument(
        "--scan-tarball", type=Path, default=None,
        help=(
            "sweep a packed .tgz for credentials and exit. Run after `npm pack`, "
            "because what npm writes is what ships."
        ),
    )
    parser.add_argument(
        "--check-bump", action="store_true",
        help=(
            "the shared-version release-history check (`make opencode-check-bump`): "
            "exit non-zero unless the candidate's version is released with this "
            "same shared payload or newer than every codex-v*/opencode-v* tag and "
            "npm version. Reads git and the registry; writes nothing"
        ),
    )
    parser.add_argument(
        "--repo", type=Path, default=None,
        help="with --check-bump: the clone whose history is read (default: this one)",
    )
    parser.add_argument(
        "--candidate", default="HEAD",
        help="with --check-bump: the commit to check (default: HEAD)",
    )
    parser.add_argument(
        "--registry", default=None,
        help="with --check-bump: must be the single registry (default: it)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_bump:
        try:
            report = check_bump(
                repo=args.repo, candidate=args.candidate, registry=args.registry,
            )
        except PayloadError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 1
        return release_history.print_report(report)

    # Both stamp branches call ``build()`` and therefore ``payload.stage()``,
    # whose dirty-tree refusal is a ``PayloadError``. Caught here for the same
    # reason the ``build`` branch catches it: a refusal is this tool's most
    # likely outcome on the publish path — `make opencode-publish-first` from a dirty
    # skill tree hits it, and so does the gate's final ``--write-stamp``, which
    # ignores $MULTI_LLM_ALLOW_DIRTY by design — and a gate that answers with a
    # traceback instead of a named line is a gate nobody can act on. The
    # evidence refusals (no session, a rehearsal, a row missing) are raised as
    # the same ``PayloadError`` for the same reason.
    if args.check_stamp:
        try:
            reason = check_stamp(args.destination)
        except PayloadError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 1
        if reason:
            print(f"REFUSING: {reason}", file=sys.stderr)
            return 1
        print("the full gate passed on these exact bytes")
        return 0

    if args.write_stamp is not None:
        try:
            stamp = write_stamp(args.destination, timestamp=args.write_stamp)
        except PayloadError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {stamp}")
        return 0

    if args.scan_tarball is not None:
        try:
            # Same derived requirement `build()` holds the staging tree to
            # (see `target_module_paths`), re-checked here because npm applies
            # its own ignore rules *after* staging — a tarball can go short
            # for a reason the staged tree never saw. See `scan_tarball`'s own
            # docstring for why the inventory assertions belong at both ends.
            scan_tarball(
                args.scan_tarball,
                required=tuple(payload.REQUIRED_INVENTORY)
                + target_module_paths(),
            )
        except PayloadError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"sweep clean: {args.scan_tarball}")
        return 0

    try:
        report = build(
            destination=args.destination, allow_dirty=args.allow_dirty
        )
    except PayloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"staged {report['payload_files']} payload files into "
        f"{report['destination']}\n  digest {report['digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
