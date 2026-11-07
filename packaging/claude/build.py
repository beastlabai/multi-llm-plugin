#!/usr/bin/env python3
"""Build the Claude Code plugin tree ``packaging/claude/multi-llm/``. Checked in, byte-compared.

Run it as::

    uv run --project skills/multi-llm -- python packaging/claude/build.py [--allow-dirty]

or, as a developer would, ``make claude-regen`` (dirty inputs allowed, writes the
tracked tree) and ``make claude-package`` (clean inputs only, and the rebuilt tree
must equal the committed one).

**What it writes.** The destination is always the source re-rooted under
``skills/multi-llm/`` by the shared allowlist, plus two explicit additions:

====================================  =========================================
staged                                from
====================================  =========================================
``.claude-plugin/plugin.json``        the root manifest, verbatim
``hooks/hooks.json``                  the root hook configuration, verbatim
``hooks/doctor.sh``                   the root hook script, verbatim
``hooks/report-check.sh``             the root hook script, verbatim
``hooks/wakeup-check.sh``             the root hook script, verbatim
``LICENSE``                           the repo root's, verbatim
``skills/multi-llm/legacy/…``         :data:`LEGACY_FILES`, Claude Code only
====================================  =========================================

**Why the tree is committed.** A Claude Code marketplace entry names a directory
in the repository, and installing it copies that directory with no build on the
user's machine. So every commit touching a shipped file owes a regenerated tree
in the same commit, and ``test_claude_package.py`` is the row that says so.

**The root stays the authored source.** ``.claude-plugin/plugin.json``, ``hooks/``
and ``skills/multi-llm/`` are edited where they are, and loading the repository
directly (``claude --plugin-dir <checkout>``) keeps working during development.
Nothing here renders them: the shipped ``SKILL.md`` is the authored one.

**Staging is** :func:`payload.stage` **plus the whole-plugin rules.** Its subtree
swap and sweep cover ``skills/multi-llm/`` alone, which is not enough for a tree
whose manifest and hooks live outside that subtree. So the complete plugin root is
assembled in a scratch directory, swept and validated as one tree, and only then
swapped into place: a failure at any step leaves the previous complete plugin
intact.

**The legacy files are named, not globbed.** ``MULTI_LLM_LEGACY_INSTRUCTIONS=1``
serves ``legacy/instructions/<mode>.md``, and a missing frozen copy silently falls
back to the current core rather than failing. A non-empty directory therefore
proves nothing, so the build requires exactly :data:`LEGACY_FILES`, both ways.
``legacy/README.md`` ships as the operator documentation the runtime's stderr
notice points at, the one exception to the rule that development documentation
stays out of a payload. None of this widens the shared allowlist: the Codex and
OpenCode builders pass no extension and carry none of these files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "multi-llm"

# Not this directory: this builder imports nothing from it, and on ``sys.path`` its
# ``build.py`` would answer a later ``import build`` meant for the OpenCode builder.
for _path in (str(SKILL_DIR), str(REPO_ROOT / "packaging")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import payload, release_history  # noqa: E402
from common.payload import (  # noqa: E402
    PayloadError, scan_tree, target_module_paths, tree_digest,
)

#: The host this plugin distributes.
HOST_ID = "claude-code"

#: The checked-in plugin directory: the marketplace entry's ``source``.
PLUGIN_DIR = PACKAGE_DIR / "multi-llm"

#: Repo-relative spelling of :data:`PLUGIN_DIR`, for git and for messages.
PLUGIN_RELATIVE = "packaging/claude/multi-llm"

#: The payload's place inside the plugin. Hooks resolve it from
#: ``${CLAUDE_PLUGIN_ROOT}`` at this exact path.
PAYLOAD_RELATIVE = "skills/multi-llm"

#: Every file the plugin carries outside ``skills/multi-llm/``, as
#: ``plugin-relative -> repo-relative``. Asserted exactly, both directions, because
#: the install copies the directory wholesale.
PLUGIN_FILES: Mapping[str, str] = {
    ".claude-plugin/plugin.json": ".claude-plugin/plugin.json",
    "LICENSE": "LICENSE",
    "hooks/hooks.json": "hooks/hooks.json",
    "hooks/doctor.sh": "hooks/doctor.sh",
    "hooks/report-check.sh": "hooks/report-check.sh",
    "hooks/wakeup-check.sh": "hooks/wakeup-check.sh",
}

#: The hook scripts, which must stay executable: ``hooks.json`` runs each by path.
HOOK_SCRIPTS: Tuple[str, ...] = (
    "hooks/doctor.sh", "hooks/report-check.sh", "hooks/wakeup-check.sh",
)

#: The complete frozen core, relative to ``skills/multi-llm/legacy/``. Exactly
#: these files, or the build fails: see the module docstring.
LEGACY_FILES: Tuple[str, ...] = (
    "README.md",
    "instructions/apply-code-fixes.md",
    "instructions/apply-suggestions.md",
    "instructions/apply-task-suggestions.md",
    "instructions/ask.md",
    "instructions/full-workflow.md",
    "instructions/generate-tasks.md",
    "instructions/implement.md",
    "instructions/init-config.md",
    "instructions/review-code.md",
    "instructions/review-plan.md",
    "instructions/review-tasks.md",
    "references/human-decision-batch.md",
    "references/resume-detection.md",
    "references/salvage-handling.md",
    "references/uv-check.md",
    "references/wave-batching.md",
)

#: Skill-relative legacy paths, as ``stage``'s ``required_extra`` wants them.
LEGACY_REQUIRED: Tuple[str, ...] = tuple(f"legacy/{name}" for name in LEGACY_FILES)

#: Repo-relative legacy paths, as ``stage``'s ``extra_allowlist`` wants them. Each
#: is a literal path with no glob character, so a new file under ``legacy/`` is
#: not admitted by accident; the exact-inventory check then names it.
LEGACY_ALLOWLIST: Tuple[str, ...] = tuple(
    f"skills/multi-llm/legacy/{name}" for name in LEGACY_FILES
)

#: Everything outside ``skills/multi-llm`` that decides what this plugin ships, and
#: is therefore outside ``payload.stage``'s own dirty-tree refusal. The generated
#: tree itself is not an input: it is this build's output.
PACKAGING_INPUTS: Tuple[str, ...] = (
    "packaging/claude/build.py",
    "packaging/common/payload.py",
    ".claude-plugin/plugin.json",
    "hooks/hooks.json",
    "hooks/doctor.sh",
    "hooks/report-check.sh",
    "hooks/wakeup-check.sh",
    "LICENSE",
)

#: Same opt-in, same exact spelling, as the other two drivers.
ALLOW_DIRTY_ENV = "MULTI_LLM_ALLOW_DIRTY"
ALLOW_DIRTY_TRUE = "1"

#: Untracked by-products a developer's run can leave inside the checked-in tree
#: (running the plugin builds a ``.venv`` and ``__pycache__``). Gitignored, so the
#: Git route never carries them, and excluded from every check for that reason.
_BY_PRODUCT_SEGMENTS = ("__pycache__", ".venv", ".pytest_cache", ".ruff_cache")

#: Tag prefix of a Claude release: ``claude-v<version>``.
TAG_PREFIX = "claude-v"

VERSION_PATH = "skills/multi-llm/pyproject.toml"
MANIFEST_RELATIVE = ".claude-plugin/plugin.json"


def allow_dirty_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    """``--allow-dirty``'s default: ``$MULTI_LLM_ALLOW_DIRTY`` spelled exactly ``1``."""
    source = os.environ if environ is None else environ
    return source.get(ALLOW_DIRTY_ENV) == ALLOW_DIRTY_TRUE


def project_version(base: Optional[Path] = None) -> str:
    """``[project] version`` from the payload's ``pyproject.toml``, the version owner."""
    root = Path(base) if base else SKILL_DIR
    return release_history.version_in((root / "pyproject.toml").read_text(encoding="utf-8"))


def _is_by_product(relative: str) -> bool:
    return any(segment in _BY_PRODUCT_SEGMENTS for segment in relative.split("/"))


def digest_of(tree: Path) -> str:
    """The digest of a finished plugin tree: every shipped file, by-products aside."""
    return tree_digest(Path(tree), exclude=_is_by_product)


def shipped_files(tree: Path) -> Dict[str, int]:
    """``{plugin-relative path: size}`` for every shipped file in *tree*."""
    return {
        relative: size
        for relative, size in payload.file_inventory(Path(tree)).items()
        if not _is_by_product(relative)
    }


# ---------------------------------------------------------------------------
# What the build asserts over the finished tree
# ---------------------------------------------------------------------------

def _inventory_problems(tree: Path, expected_payload: Sequence[str]) -> List[str]:
    """The finished tree holds exactly the plugin files plus the payload."""
    present = set(shipped_files(tree))
    expected = set(PLUGIN_FILES) | {
        f"{PAYLOAD_RELATIVE}/{name}" for name in expected_payload
    }
    problems = [f"missing: {path}" for path in sorted(expected - present)]
    problems += [f"unexpected: {path}" for path in sorted(present - expected)]
    return problems


def _legacy_problems(root: Path) -> List[str]:
    """Tracked ``legacy/`` equals :data:`LEGACY_FILES`, both ways."""
    prefix = "skills/multi-llm/legacy"
    tracked = {
        path[len(prefix) + 1:] for path in payload.tracked_files(root, prefix)
    }
    wanted = set(LEGACY_FILES)
    # On disk as well as in the index: --allow-dirty reads the working tree, where a
    # tracked file can already be gone.
    present = {name for name in tracked if (root / prefix / name).is_file()}
    problems = [f"missing: legacy/{name}" for name in sorted(wanted - present)]
    problems += [f"not in LEGACY_FILES: legacy/{name}" for name in sorted(tracked - wanted)]
    return problems


def _hook_commands(configuration: Mapping[str, Any]) -> List[str]:
    """Every command string in a ``hooks.json``."""
    commands: List[str] = []
    for groups in (configuration.get("hooks") or {}).values():
        for group in groups:
            for hook in group.get("hooks") or []:
                if hook.get("type") == "command":
                    commands.append(str(hook.get("command", "")))
    return commands


def component_problems(tree: Path, version: str) -> List[str]:
    """The installed-plugin rules: the manifest and every path it reaches.

    Each is something an install copies without complaint and a session then
    fails on, so the build refuses it instead of a user's first run.
    """
    tree = Path(tree)
    problems: List[str] = []
    manifest_path = tree / MANIFEST_RELATIVE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{MANIFEST_RELATIVE} does not parse: {exc}"]
    if manifest.get("name") != "multi-llm":
        problems.append(f"{MANIFEST_RELATIVE} names {manifest.get('name')!r}, not 'multi-llm'")
    if manifest.get("version") != version:
        problems.append(
            f"{MANIFEST_RELATIVE} says version {manifest.get('version')!r}, "
            f"{VERSION_PATH} says {version!r}. The root manifest is the Claude Code "
            "end of the shared version chain: update it, then run `make claude-regen`."
        )
    hooks_rel = manifest.get("hooks")
    if hooks_rel != "./hooks/hooks.json":
        problems.append(
            f"{MANIFEST_RELATIVE} points hooks at {hooks_rel!r}, not ./hooks/hooks.json"
        )
    shipped = shipped_files(tree)
    configuration_path = tree / "hooks" / "hooks.json"
    try:
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return problems + [f"hooks/hooks.json does not parse: {exc}"]
    commands = _hook_commands(configuration)
    if not commands:
        problems.append("hooks/hooks.json registers no command hook")
    for command in commands:
        target = command.strip().strip('"')
        prefix = "${CLAUDE_PLUGIN_ROOT}/"
        if not target.startswith(prefix):
            problems.append(
                f"hook command {command!r} does not resolve from ${{CLAUDE_PLUGIN_ROOT}}, "
                "so it depends on something outside the installed plugin"
            )
            continue
        relative = target[len(prefix):]
        # Against the shipped set rather than the disk: `../x` can resolve to a file
        # beside the tree here and to nothing in an install.
        if relative not in shipped:
            problems.append(
                f"hook command {command!r} names a file the plugin does not ship: {relative}"
            )
    for script in HOOK_SCRIPTS:
        path = tree / script
        if not path.is_file():
            problems.append(f"{script} is missing")
        elif os.name != "nt" and not path.stat().st_mode & stat.S_IXUSR:
            problems.append(f"{script} is not executable; hooks.json runs it by path")
    skill = tree / PAYLOAD_RELATIVE / "SKILL.md"
    if not skill.is_file():
        problems.append(f"{PAYLOAD_RELATIVE}/SKILL.md is missing")
    return problems


def _catalog_entry_problems(checkout: Path) -> List[str]:
    """The catalog entry names this tree and this plugin's name."""
    catalog = checkout / ".claude-plugin" / "marketplace.json"
    try:
        entries = json.loads(catalog.read_text(encoding="utf-8")).get("plugins") or []
    except (OSError, ValueError) as exc:
        return [f".claude-plugin/marketplace.json does not parse: {exc}"]
    wanted = f"./{PLUGIN_RELATIVE}"
    matching = [entry for entry in entries if entry.get("name") == "multi-llm"]
    if len(matching) != 1 or matching[0].get("source") != wanted:
        return [
            f".claude-plugin/marketplace.json's multi-llm entry must have source "
            f"{wanted!r}; it has {[entry.get('source') for entry in matching]!r}"
        ]
    return []


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def _copy_executable_faithfully(source: Path, target: Path, *, executable: bool) -> None:
    """Copy bytes, then set the mode from the git index rather than from disk.

    A checkout's mode bits follow its umask. The tree is byte-compared against a
    clean rebuild, so the only mode that may reach it is the one git records.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if os.name != "nt":
        os.chmod(target, 0o755 if executable else 0o644)


def _tracked_modes(checkout: Path, paths: Sequence[str]) -> Dict[str, bool]:
    """``{repo-relative path: is 100755 in the index}`` for each of *paths*."""
    out = payload._git(  # noqa: SLF001 — the one shared git runner
        checkout, "ls-files", "-s", "-z", "--", *paths,
    )
    modes: Dict[str, bool] = {}
    for record in out.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        modes[path] = meta.split(" ", 1)[0] == "100755"
    return modes


def _normalise_payload_modes(checkout: Path, payload_root: Path) -> None:
    """Every staged payload file gets the mode git records for it."""
    if os.name == "nt":
        return
    modes = _tracked_modes(checkout, [PAYLOAD_RELATIVE])
    strip = PAYLOAD_RELATIVE + "/"
    for path, executable in modes.items():
        target = payload_root / path[len(strip):]
        if target.is_file():
            os.chmod(target, 0o755 if executable else 0o644)


def _refuse_symlinked_inputs(checkout: Path) -> None:
    """A plugin input outside the payload is a real file, in the index and on disk.

    :func:`payload.stage` refuses symlinks under ``skills/multi-llm`` only. These
    inputs are copied through ``shutil.copyfile``, which follows a link, so without
    this the tree would ship whatever the link points at, possibly outside the
    repository, and a clean rebuild elsewhere could not reproduce it.
    """
    linked = sorted(
        source for source in PLUGIN_FILES.values()
        if payload.symlinked_component(checkout, source) is not None
        or payload._symlinks(checkout, source)  # noqa: SLF001 — the shared mode reader
    )
    if linked:
        raise PayloadError(
            "refusing to build a plugin from symlinked inputs:\n"
            + "\n".join(f"    {path}" for path in linked)
            + "\nReplace each with a real file."
        )


def _refuse_untracked_inputs(checkout: Path) -> None:
    """``--allow-dirty`` tolerates *modified* inputs, never an untracked one."""
    patterns = (*payload.runtime_allowlist(checkout), *LEGACY_ALLOWLIST)
    untracked = list(payload.untracked_admitted(checkout, "skills/multi-llm", patterns))
    legacy_prefix = "skills/multi-llm/legacy"
    out = payload._git(  # noqa: SLF001 — the one shared git runner
        checkout, "ls-files", "--others", "--exclude-standard", "-z", "--", legacy_prefix,
    )
    untracked += [path for path in out.split("\0") if path and path not in untracked]
    for source in PLUGIN_FILES.values():
        probe = payload._git(  # noqa: SLF001
            checkout, "ls-files", "--others", "--exclude-standard", "--", source,
        )
        untracked += [line for line in probe.splitlines() if line.strip()]
    if untracked:
        untracked = sorted(set(untracked))
        listing = "\n".join(f"    {path}" for path in untracked[:20])
        more = "" if len(untracked) <= 20 else f"\n    … and {len(untracked) - 20} more"
        raise PayloadError(
            f"refusing to build: {len(untracked)} untracked input file(s):\n"
            f"{listing}{more}\n"
            "Staging selects from `git ls-files`, so --allow-dirty would build a tree "
            "without them, and a clean build of the commit could not reproduce it "
            "once they are added. `git add` them (or delete them), then rebuild."
        )


def build(
    *,
    destination: Optional[Path] = None,
    base: Optional[Path] = None,
    root: Optional[Path] = None,
    allow_dirty: bool = False,
    check_catalog: bool = True,
) -> Dict[str, Any]:
    """Build the plugin tree into *destination* (default: the checked-in one).

    Assembled in a sibling temp directory and swapped in only after every check
    has passed. Returns a small report. *check_catalog* is off only for a fixture
    repository that carries no marketplace catalog.
    """
    out = Path(destination) if destination else PLUGIN_DIR
    checkout = Path(root) if root else REPO_ROOT
    # Follows *root*: a caller naming another checkout must not read this one's version.
    skill_base = Path(base) if base else checkout / PAYLOAD_RELATIVE

    if not allow_dirty:
        for path in PACKAGING_INPUTS:
            payload.assert_clean(checkout, path)
    else:
        _refuse_untracked_inputs(checkout)

    legacy_gaps = _legacy_problems(checkout)
    if legacy_gaps:
        raise PayloadError(
            "the legacy inventory is not the one this plugin ships:\n"
            + "\n".join(f"    {line}" for line in legacy_gaps)
            + "\nA missing frozen copy silently falls back to the current core when "
            "MULTI_LLM_LEGACY_INSTRUCTIONS=1 is set, so the build requires the exact "
            "list. Restore the file, or change LEGACY_FILES on purpose."
        )

    _refuse_symlinked_inputs(checkout)

    version = project_version(skill_base)
    modes = _tracked_modes(checkout, list(PLUGIN_FILES.values()))
    missing_sources = [
        source for source in PLUGIN_FILES.values()
        if not (checkout / source).is_file() or source not in modes
    ]
    if missing_sources:
        raise PayloadError(
            "a tracked plugin input is missing: " + ", ".join(sorted(missing_sources))
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{out.name}.build.", dir=str(out.parent)))
    try:
        # mkdtemp is 0700; the installed plugin is an ordinary directory.
        os.chmod(scratch, 0o755)

        # 1. The payload: the shared allowlist plus the Claude-only legacy files.
        staged = payload.stage(
            checkout,
            scratch / PAYLOAD_RELATIVE,
            allow_dirty=allow_dirty,
            required_extra=(*target_module_paths(skill_base), *LEGACY_REQUIRED),
            extra_allowlist=LEGACY_ALLOWLIST,
        )
        os.chmod(scratch / PAYLOAD_RELATIVE, 0o755)
        _normalise_payload_modes(checkout, scratch / PAYLOAD_RELATIVE)

        # 2. The manifest, the hooks and the licence, from their authored sources.
        for target, source in PLUGIN_FILES.items():
            _copy_executable_faithfully(
                checkout / source, scratch / target, executable=modes[source],
            )

        # 3. The sweep over the complete plugin root: the files above are not the
        #    payload stage() swept.
        scan_tree(scratch)

        # 4. Every assertion, before anything moves.
        problems = _inventory_problems(scratch, staged)
        problems += component_problems(scratch, version)
        if check_catalog:
            problems += _catalog_entry_problems(checkout)
        if problems:
            raise PayloadError(
                "the built plugin tree cannot ship:\n"
                + "\n".join(f"    {line}" for line in problems)
                + "\nThe install copies this directory wholesale, so an unaccounted "
                "file ships and a missing one is a plugin that installs and cannot run."
            )
        files = shipped_files(scratch)
        payload.check_budget(HOST_ID, len(files), sum(files.values()))
        digest = digest_of(scratch)

        # 5. The swap.
        payload.swap_tree(scratch, out)
        scratch = None  # type: ignore[assignment]
    finally:
        if scratch is not None and scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

    return {
        "destination": str(out),
        "payload_files": len(staged),
        "files": len(files),
        "bytes": sum(files.values()),
        "digest": digest,
        "version": version,
    }


def committed_drift(root: Optional[Path] = None) -> Tuple[str, ...]:
    """``git status --porcelain`` lines under the checked-in tree, by-products
    aside. Empty means the working tree's plugin equals the committed one."""
    checkout = Path(root) if root else REPO_ROOT
    out = payload._git(  # noqa: SLF001 — the one shared git runner
        checkout, "status", "--porcelain", "--untracked-files=all", "--",
        PLUGIN_RELATIVE,
    )
    return tuple(
        line for line in out.splitlines()
        if line.strip() and not _is_by_product(line[3:])
    )


# ---------------------------------------------------------------------------
# Claude follows master: a shipped-byte change is a release
# ---------------------------------------------------------------------------

def release_tags(root: Optional[Path] = None) -> Tuple[str, ...]:
    """Every ``claude-v*`` tag, oldest version first."""
    checkout = Path(root) if root else REPO_ROOT
    out = payload._git(checkout, "tag", "--list", f"{TAG_PREFIX}*")  # noqa: SLF001
    named = [line.strip() for line in out.splitlines() if line.strip()]
    parsed = []
    for tag in named:
        version = tag[len(TAG_PREFIX):]
        if not release_history.is_version(version):
            raise PayloadError(f"tag {tag} does not name a strict semver version")
        parsed.append((release_history.version_key(version), tag))
    return tuple(tag for _, tag in sorted(parsed))


def _tree_id(checkout: Path, ref: str) -> Optional[str]:
    """The git tree id of the generated plugin at *ref*, or ``None`` if absent."""
    try:
        return payload._git(  # noqa: SLF001
            checkout, "rev-parse", "--verify", "--quiet",
            f"{ref}:{PLUGIN_RELATIVE}",
        ).strip() or None
    except PayloadError:
        return None


def _version_at(checkout: Path, ref: str) -> str:
    text = payload._git(checkout, "show", f"{ref}:{VERSION_PATH}")  # noqa: SLF001
    return release_history.version_in(text)


def release_problems(
    root: Optional[Path] = None, *, candidate: str = "HEAD", base: Optional[str] = None,
) -> Tuple[str, ...]:
    """Refusals for *candidate* against the last Claude release, and against *base*.

    Claude Code installs follow ``master`` and cache by manifest version, so a
    shipped byte that moves under an unmoved version reaches nobody who already
    has that version. The rule is over the **whole generated tree**, hooks and
    legacy files included, which the shared-payload digest deliberately does not
    cover. The git tree id is the comparison: it covers bytes and the executable
    bit, and it needs no recorded digest file in the plugin.

    *base* is the pull request's target. It catches the same edit before any
    ``claude-v*`` tag exists, and on a branch whose release tag has not been cut.
    A base without the tree is the commit that adds it, which has nothing to be
    compared with. A base that does not resolve raises instead: read as that same
    "no tree", a misspelt or unfetched target would pass every edit. A newest tag
    without the tree and a shallow clone, which holds no older tag, are refused for
    the same reason: each would otherwise read as nothing to compare.
    """
    checkout = Path(root) if root else REPO_ROOT
    shallow = release_history.shallow_gap(checkout)
    if shallow:
        return (shallow,)
    problems: List[str] = []
    current_tree = _tree_id(checkout, candidate)
    version = _version_at(checkout, candidate)
    if current_tree is None:
        return (f"{candidate} carries no {PLUGIN_RELATIVE}/: run `make claude-regen`.",)

    def compare(ref: str, label: str) -> None:
        previous_tree = _tree_id(checkout, ref)
        if previous_tree is None or previous_tree == current_tree:
            return
        previous_version = _version_at(checkout, ref)
        previous_key = release_history.version_key(
            release_history.release_version(previous_version))
        current_key = release_history.version_key(release_history.release_version(version))
        if current_key <= previous_key:
            problems.append(
                f"{PLUGIN_RELATIVE}/ moved since {label} but the version is {version}, "
                f"not newer than {previous_version}. Claude Code caches by version, so "
                f"installed users would never receive these bytes. Bump {VERSION_PATH} "
                "and its mirrors, run `make claude-regen`, and cut a new "
                f"{TAG_PREFIX}<version> tag (packaging/claude/README.md)."
            )

    tags = release_tags(checkout)
    if tags:
        newest = f"refs/tags/{tags[-1]}"
        if _tree_id(checkout, newest) is None:
            problems.append(
                f"{tags[-1]} carries no {PLUGIN_RELATIVE}/, so it cannot be a Claude "
                "release and the moved-tree rule has nothing to hold this commit to. "
                f"A {TAG_PREFIX}<version> tag records the commit that shipped the tree "
                "(packaging/claude/README.md); find out how this one was cut."
            )
        else:
            compare(newest, tags[-1])
    if base:
        release_history.resolve_commit(checkout, base)
        compare(base, base)
    return tuple(problems)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Build the Claude Code plugin tree packaging/claude/multi-llm/.",
    )
    parser.add_argument(
        "--destination", type=Path, default=None,
        help="where to build (default: the checked-in packaging/claude/multi-llm)",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", default=allow_dirty_from_env(),
        help=(
            "build from inputs with uncommitted changes: what `make claude-regen` "
            "does, since the regenerated tree lands in the same commit as the edit "
            f"that implies it. Also ${ALLOW_DIRTY_ENV}={ALLOW_DIRTY_TRUE} (that exact "
            "spelling)."
        ),
    )
    parser.add_argument(
        "--require-no-diff", action="store_true",
        help=(
            "after building into the checked-in tree, fail unless it equals the "
            "committed one (what `make claude-package` asserts)"
        ),
    )
    parser.add_argument(
        "--check-release", action="store_true",
        help=(
            "exit non-zero if the generated tree moved since the newest claude-v* tag "
            "(and since --base) under a version that did not. Reads git only"
        ),
    )
    parser.add_argument(
        "--base", default=None,
        help="with --check-release: also compare against this ref (a PR's target)",
    )
    parser.add_argument(
        "--check-bump", action="store_true",
        help=(
            "--check-release plus the shared-version release-history check over every "
            "claude-v*/codex-v*/opencode-v* tag and npm's version list. Reads the registry"
        ),
    )
    parser.add_argument(
        "--repo", type=Path, default=None,
        help="with --check-release/--check-bump: the clone to read (default: this one)",
    )
    parser.add_argument(
        "--registry", default=None,
        help="with --check-bump: must be the single registry (default: it)",
    )
    parser.add_argument(
        "--print-version", action="store_true",
        help="print the version the plugin ships (pyproject.toml's) and exit",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.print_version:
            print(project_version())
            return 0
        if args.check_release or args.check_bump:
            problems = list(release_problems(args.repo, base=args.base))
            if args.check_bump:
                shared = release_history.check(args.repo or REPO_ROOT, registry=args.registry)
                return release_history.print_report(
                    shared, extra=problems, ok="the Claude release rules hold",
                )
            for line in problems:
                print(f"REFUSING: {line}", file=sys.stderr)
            if problems:
                return 1
            print("the Claude release rules hold")
            return 0
        report = build(destination=args.destination, allow_dirty=args.allow_dirty)
        if args.require_no_diff:
            if args.destination is not None:
                print("ERROR: --require-no-diff grades the checked-in tree only",
                      file=sys.stderr)
                return 2
            drift = committed_drift()
            if drift:
                print(
                    "ERROR: the rebuilt plugin differs from the committed tree:\n"
                    + "\n".join(f"    {line}" for line in drift[:20])
                    + "\nRun `make claude-regen` and commit the result with the "
                    "source edit that implies it.",
                    file=sys.stderr,
                )
                return 1
    except PayloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"built {report['files']} files ({report['bytes']} bytes) into "
        f"{report['destination']}\n  version {report['version']}\n"
        f"  digest {report['digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
