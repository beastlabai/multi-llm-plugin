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
too early to see a tarball.

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
SKILL_DIR_TOKEN = surface.SKILL_DIR_TOKEN

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
#: ``make opencode-publish`` would produce a spurious "staged tree has changed".
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
#: ``make opencode-publish`` neither sets nor forwards it, so a dirty publish is
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
        HOST_ID, base=str(root), skill=SKILL_DIR_TOKEN, template=template
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

    # 2. The command body, {SKILL_DIR} intact.
    (dist / "command.md").write_text(command_body(skill_base), encoding="utf-8")

    # 3. The generated CLI surface.
    (dist / "cli-surface.json").write_text(
        cli_surface_json(skill_base), encoding="utf-8"
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
    (dist / DIGEST_NAME).write_text(accumulator + "\n", encoding="utf-8")

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
# The publish gate's stamp
# ---------------------------------------------------------------------------

#: Where the full gate records that it passed. Gitignored, and **it never
#: travels**: it is not committed and is never uploaded as a CI artifact, because
#: a stamp from another machine attests to that machine's rows rather than to
#: this one's. CI consumes no stamps at all — the publishing job re-runs the gate
#: inline, which is why the two mechanisms are spelled out separately rather than
#: one being made to serve both.
STAMP_NAME = ".package-check.stamp"

#: The tier a stamp must carry to authorize a publish. A `-ci` run writes no
#: stamp at all, and a stamp claiming any other tier is refused by name.
FULL_TIER = "full"


def write_stamp(
    destination: Optional[Path] = None, *, timestamp: str = "",
    allow_dirty: bool = False,
) -> Path:
    """Record that the full gate passed **on these exact bytes**.

    Keyed on ``dist/tree.sha256`` — a digest over the staged content — rather
    than on a git tree hash, which is the wrong key twice over: the thing being
    published is the *staged* tree, which includes generated files ``.gitignore``
    excludes, and a git hash would miss uncommitted edits entirely.

    ``allow_dirty`` is a parameter and **not** read from the environment here,
    unlike ``main``'s flag. The environment variable exists for one reason — npm
    invokes ``prepack`` with a fixed command line, so a flag cannot reach it —
    and this function is not on that path. Reading it here would mean a developer
    who exported it once could stamp, and therefore publish, from a dirty tree.
    """
    out = Path(destination) if destination else PACKAGE_DIR
    report = build(destination=out, allow_dirty=allow_dirty)
    manifest = json.loads((out / "package.json").read_text(encoding="utf-8")) \
        if (out / "package.json").is_file() \
        else json.loads((PACKAGE_DIR / "package.json").read_text(encoding="utf-8"))
    stamp = out / STAMP_NAME
    stamp.write_text(
        json.dumps({
            "tier": FULL_TIER,
            "digest": report["digest"],
            "version": manifest["version"],
            "stamped_at": timestamp,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return stamp


def check_stamp(
    destination: Optional[Path] = None, *, allow_dirty: bool = False,
) -> Optional[str]:
    """``None`` if a publish may proceed, else the reason it may not.

    Four conditions, **each named separately**, so "the gate is red" is never the
    same message as "you edited a file since". A gate is worth nothing if the
    publish command can skip it, and a refusal nobody can act on is nearly as
    bad.
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
    if stamp.get("tier") != FULL_TIER:
        return (
            f"the stamp says tier={stamp.get('tier')!r}. Only a full "
            "`make opencode-package-check` authorizes a publish; the CI tier "
            "runs rows 1-6 and deliberately writes no stamp."
        )
    fresh = build(destination=out, allow_dirty=allow_dirty)["digest"]
    if stamp.get("digest") != fresh:
        return (
            f"the staged tree has changed since the gate ran (stamp "
            f"{stamp.get('digest')}, now {fresh}). Re-run "
            "`make opencode-package-check`."
        )
    manifest = json.loads((out / "package.json").read_text(encoding="utf-8")) \
        if (out / "package.json").is_file() \
        else json.loads((PACKAGE_DIR / "package.json").read_text(encoding="utf-8"))
    if stamp.get("version") != manifest["version"]:
        return (
            f"the stamp is for version {stamp.get('version')!r} but package.json "
            f"says {manifest['version']!r}."
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
        help="record that the full gate passed on the freshly staged bytes",
    )
    parser.add_argument(
        "--check-stamp", action="store_true",
        help=(
            "exit non-zero unless the full gate passed on these exact bytes, "
            "naming which of the four conditions failed"
        ),
    )
    parser.add_argument(
        "--scan-tarball", type=Path, default=None,
        help=(
            "sweep a packed .tgz for credentials and exit. Run after `npm pack`, "
            "because what npm writes is what ships."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Both stamp branches call ``build()`` and therefore ``payload.stage()``,
    # whose dirty-tree refusal is a ``PayloadError``. Caught here for the same
    # reason the ``build`` branch catches it: a refusal is this tool's most
    # likely outcome on the publish path — `make opencode-publish` from a dirty
    # skill tree hits it, and so does the gate's final ``--write-stamp``, which
    # ignores $MULTI_LLM_ALLOW_DIRTY by design — and a gate that answers with a
    # traceback instead of a named line is a gate nobody can act on.
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
