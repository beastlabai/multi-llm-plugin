#!/usr/bin/env python3
"""Build the codex plugin tree ``packaging/codex/multi-llm/``. Checked in, byte-compared.

Run it as::

    uv run --project skills/multi-llm -- python packaging/codex/build.py [--allow-dirty]

or, as a developer would, ``make codex-regen`` (dirty inputs allowed, writes the
tracked tree) and ``make codex-package`` (clean inputs only, and the rebuilt tree
must equal the committed one).

**What it writes** (plans/codex-native-plugin.md, D1's mapping table). The
destination is always the source re-rooted under ``skills/multi-llm/``, with six
exceptions:

====================================  =========================================
staged                                from
====================================  =========================================
``.codex-plugin/plugin.json``         ``plugin.json.tmpl``, ``version`` filled from
                                      ``skills/multi-llm/pyproject.toml`` (D6)
``.mcp.json``                         ``mcp.json``, verbatim (A2b)
``mcp/server.py``                     ``mcp_server.py``, verbatim (D3)
``dist/cli-surface.json``             ``packaging/common/surface.py`` for ``codex``
``LICENSE``                           the repo root's, verbatim
``skills/multi-llm/SKILL.md``         **overwritten** with ``host_install.render``
                                      of ``hosts/codex/plugin-skill.md.tmpl`` (D7)
====================================  =========================================

plus ``dist/tree.sha256``, the digest over everything above, written last.

**Why the tree is committed** when the opencode one is gitignored: the Git
marketplace route serves the checked-in copy with no build on the user's
machine (R19), so what ``codex plugin add`` copies is exactly these bytes.
Every commit touching ``skills/multi-llm/`` therefore owes a regenerated tree in
the same commit, and ``test_checked_in_plugin_matches_build`` is the row that
says so.

**Staging is** :func:`payload.stage` **and nothing else** — the allowlist, the
required inventory (plus every ``modes.py`` target module), the symlink and
dirty-input refusals and the secret sweep all come from there. This driver adds
the six files above, re-runs the sweep and the ingestion rules over the whole
tree, and only then swaps it into place, so a failed build never leaves a
half-written plugin where a good one was.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "multi-llm"

for _path in (str(SKILL_DIR), str(REPO_ROOT / "packaging"), str(PACKAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import payload, surface  # noqa: E402
from common.payload import (  # noqa: E402,F401
    PayloadError, scan_tree, target_module_paths, tree_digest,
)
from utils import host_install  # noqa: E402

import ingestion_rules  # noqa: E402
import release_gate as gate  # noqa: E402

#: The host this plugin distributes. Not a new host id.
HOST_ID = "codex"

#: The checked-in plugin directory — the marketplace entry's ``source.path``.
PLUGIN_DIR = PACKAGE_DIR / "multi-llm"

#: Repo-relative spelling of :data:`PLUGIN_DIR`, for git and for messages.
PLUGIN_RELATIVE = "packaging/codex/multi-llm"

#: The payload's place inside the plugin (must normalize to ``skills``, R6).
PAYLOAD_RELATIVE = "skills/multi-llm"

#: The thin skill's template, skill-relative (D3, D7).
SKILL_TEMPLATE = "hosts/codex/plugin-skill.md.tmpl"

#: The token passed as ``render``'s baked prefix. The template carries no slot
#: for it, so this is belt-and-braces: nothing absolute can be baked (D2).
SKILL_DIR_TOKEN = surface.SKILL_DIR_TOKEN

#: The hand-written sources of the six non-payload files.
MANIFEST_TEMPLATE = PACKAGE_DIR / "plugin.json.tmpl"
MCP_SOURCE = PACKAGE_DIR / "mcp.json"
SERVER_SOURCE = PACKAGE_DIR / "mcp_server.py"
VERSION_SLOT = "{VERSION}"

DIGEST_NAME = "tree.sha256"
DIGEST_RELATIVE = f"dist/{DIGEST_NAME}"

#: Every file the plugin carries outside ``skills/multi-llm/`` — asserted
#: exactly, both directions, because ``source.path`` copies the directory
#: wholesale and an unaccounted file is a file that installs.
PLUGIN_FILES: Tuple[str, ...] = (
    ".codex-plugin/plugin.json",
    ".mcp.json",
    "LICENSE",
    "mcp/server.py",
    "dist/cli-surface.json",
    DIGEST_RELATIVE,
)

#: A4 row 11's runnable-payload list, under ``skills/multi-llm/``. The static
#: inventory ``payload.stage`` already asserts is wider; this is the gate's own
#: named set, re-checked over the finished tree rather than the staged half.
RUNNABLE_PAYLOAD: Tuple[str, ...] = (
    "multi_llm.py", "pyproject.toml", "uv.lock", "SKILL.md",
)
RUNNABLE_TREES: Tuple[str, ...] = ("instructions/", "hosts/")

#: Everything outside ``skills/multi-llm`` that decides what this plugin ships,
#: and is therefore outside ``payload.stage``'s own dirty-tree refusal. The
#: generated tree itself is *not* an input — it is this build's output.
PACKAGING_INPUTS: Tuple[str, ...] = (
    "packaging/codex/build.py",
    "packaging/codex/ingestion_rules.py",
    "packaging/codex/mcp_server.py",
    "packaging/codex/mcp.json",
    "packaging/codex/plugin.json.tmpl",
    "packaging/common/payload.py",
    "packaging/common/surface.py",
    "LICENSE",
)

#: Same opt-in, same exact spelling, as the opencode driver: ``=1`` and
#: nothing else, so exporting ``=0`` turns it off rather than on.
ALLOW_DIRTY_ENV = "MULTI_LLM_ALLOW_DIRTY"
ALLOW_DIRTY_TRUE = "1"


def allow_dirty_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    """``--allow-dirty``'s default: ``$MULTI_LLM_ALLOW_DIRTY`` spelled exactly ``1``."""
    source = os.environ if environ is None else environ
    return source.get(ALLOW_DIRTY_ENV) == ALLOW_DIRTY_TRUE

#: The full gate's record, gitignored (``packaging/*/.package-check.stamp``).
STAMP_NAME = ".package-check.stamp"
FULL_TIER = "full"

#: Untracked by-products a developer's run can leave inside the checked-in tree
#: (running the server builds a ``.venv`` and ``__pycache__``). Gitignored, so
#: the Git route never carries them; excluded from the digest for that reason.
_BY_PRODUCT_SEGMENTS = ("__pycache__", ".venv", ".pytest_cache", ".ruff_cache")


# ---------------------------------------------------------------------------
# The generated pieces
# ---------------------------------------------------------------------------

def project_version(base: Optional[Path] = None) -> str:
    """``[project] version`` from the payload's ``pyproject.toml`` (D6's owner).

    Parsed by hand rather than with ``tomllib``, which the skill's 3.10 floor
    does not have; the key is a one-line string by construction.
    """
    root = Path(base) if base else SKILL_DIR
    return _version_in((root / "pyproject.toml").read_text(encoding="utf-8"))


def manifest_text(version: Optional[str] = None) -> str:
    """``.codex-plugin/plugin.json``: the template with ``version`` filled."""
    template = MANIFEST_TEMPLATE.read_text(encoding="utf-8")
    if template.count(VERSION_SLOT) != 1:
        raise PayloadError(
            f"{MANIFEST_TEMPLATE} must carry exactly one {VERSION_SLOT} slot"
        )
    text = template.replace(VERSION_SLOT, version or project_version())
    json.loads(text)  # a template that stops being JSON fails here, not at install
    return text


#: The skill template's one codex-only slot: ``surface.long_call_rule``,
#: filled from the same ``tool_timeout_sec`` the server enforces (fix round 16).
LONG_CALL_SLOT = "{LONG_CALL_RULE}"


def skill_md(base: Optional[Path] = None) -> str:
    """The plugin's ``SKILL.md`` — D1's mapping-table call, spelled once."""
    root = Path(base) if base else SKILL_DIR
    template = (root / SKILL_TEMPLATE).read_text(encoding="utf-8")
    if template.count(LONG_CALL_SLOT) != 1:
        raise PayloadError(
            f"{SKILL_TEMPLATE} must carry exactly one {LONG_CALL_SLOT} slot"
        )
    template = template.replace(LONG_CALL_SLOT, surface.long_call_rule(HOST_ID))
    return host_install.render(
        HOST_ID, base=str(root), skill=SKILL_DIR_TOKEN, template=template
    )


def cli_surface_json() -> str:
    """This plugin's ``dist/cli-surface.json``."""
    return surface.cli_surface_json(HOST_ID)


def _is_by_product(relative: str) -> bool:
    return any(segment in _BY_PRODUCT_SEGMENTS for segment in relative.split("/"))


def _not_digested(relative: str) -> bool:
    return relative == DIGEST_RELATIVE or _is_by_product(relative)


def digest_of(tree: Path) -> str:
    """The digest ``dist/tree.sha256`` records: every shipped file but itself."""
    return tree_digest(Path(tree), exclude=_not_digested)


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def _inventory(tree: Path) -> None:
    """The finished tree holds exactly the plugin files plus the payload."""
    outside = {
        str(p.relative_to(tree)).replace(os.sep, "/")
        for p in tree.rglob("*")
        if p.is_file() and not str(p.relative_to(tree)).replace(
            os.sep, "/").startswith(PAYLOAD_RELATIVE + "/")
    }
    expected = set(PLUGIN_FILES)
    surplus = sorted(outside - expected)
    missing = sorted(expected - outside)
    payload_root = tree / PAYLOAD_RELATIVE
    missing += [
        f"{PAYLOAD_RELATIVE}/{name}" for name in RUNNABLE_PAYLOAD
        if not (payload_root / name).is_file()
    ]
    missing += [
        f"{PAYLOAD_RELATIVE}/{prefix} (no files)" for prefix in RUNNABLE_TREES
        if not any(p.is_file() for p in (payload_root / prefix).rglob("*"))
    ]
    if surplus or missing:
        parts = []
        if surplus:
            parts.append("unexpected: " + ", ".join(surplus))
        if missing:
            parts.append("missing: " + ", ".join(missing))
        raise PayloadError(
            f"the built plugin tree is not what this build produces "
            f"({'; '.join(parts)}). `source.path` installs the directory "
            "wholesale, so an unaccounted file ships and a missing one is a "
            "plugin that installs and cannot run."
        )


def untracked_admitted(root: Optional[Path] = None) -> Tuple[str, ...]:
    """Untracked, non-ignored files under the payload prefix the allowlist admits.

    ``payload.stage`` selects from ``git ls-files``, so an untracked file is
    never a candidate — and ``--allow-dirty`` switches off the only refusal
    (``assert_clean``) that would have named it. The result is a build that
    silently omits an input the working tree plainly has: the checked-in tree
    then attests (``dist/tree.sha256``) to bytes no clean build of the commit
    reproduces, once the file is finally added. Same lesson as ``doctor.py``:
    a dirty-tolerant mode must still see what it cannot ship.
    """
    checkout = Path(root) if root else REPO_ROOT
    prefix = "skills/multi-llm"
    out = payload._git(  # noqa: SLF001 — the one shared git runner
        checkout, "ls-files", "--others", "--exclude-standard", "-z", "--", prefix,
    )
    patterns = payload.runtime_allowlist(checkout)
    return tuple(sorted(
        path for path in out.split("\0")
        if path and any(payload._matches(path, p) for p in patterns)  # noqa: SLF001
    ))


def build(
    *,
    destination: Optional[Path] = None,
    base: Optional[Path] = None,
    root: Optional[Path] = None,
    allow_dirty: bool = False,
    server: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the plugin tree into *destination* (default: the checked-in one).

    Staged into a sibling temp directory first and swapped in only after every
    check has passed. Returns a small report. *server* overrides the server
    source for a test that grades the build around a stand-in.
    """
    out = Path(destination) if destination else PLUGIN_DIR
    skill_base = Path(base) if base else SKILL_DIR
    checkout = Path(root) if root else REPO_ROOT
    server_source = Path(server) if server else SERVER_SOURCE

    if not server_source.is_file():
        raise PayloadError(
            f"{server_source} does not exist. It is the plugin's stdio server "
            "source, copied to mcp/server.py; `.mcp.json` names that file, so a "
            "tree without it installs a tool that can never start."
        )
    if not allow_dirty:
        for path in PACKAGING_INPUTS:
            payload.assert_clean(checkout, path)
    else:
        # --allow-dirty tolerates *modified* tracked inputs, which stage()
        # copies from disk. It cannot tolerate an untracked one: stage() reads
        # `git ls-files`, so the file would be silently absent from the tree.
        untracked = untracked_admitted(checkout)
        if untracked:
            listing = "\n".join(f"    {path}" for path in untracked[:20])
            more = ("" if len(untracked) <= 20
                    else f"\n    … and {len(untracked) - 20} more")
            raise PayloadError(
                f"refusing to build: {len(untracked)} untracked file(s) the "
                f"payload allowlist admits:\n{listing}{more}\n"
                "Staging selects from `git ls-files`, so --allow-dirty would "
                "build a tree without them, and the digest would attest to "
                "bytes a clean build cannot reproduce once they are added. "
                "`git add` them (or delete them), then rebuild."
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{out.name}.build.", dir=str(out.parent)))
    try:
        # mkdtemp is 0700; the installed plugin is an ordinary directory.
        os.chmod(scratch, 0o755)
        # 1. The payload, by allowlist — stage() does the sweep and inventory.
        staged = payload.stage(
            checkout,
            scratch / PAYLOAD_RELATIVE,
            allow_dirty=allow_dirty,
            required_extra=target_module_paths(skill_base),
        )

        os.chmod(scratch / PAYLOAD_RELATIVE, 0o755)

        # 2. The thin skill, over the staged core SKILL.md (D1, D7).
        (scratch / PAYLOAD_RELATIVE / "SKILL.md").write_text(
            skill_md(skill_base), encoding="utf-8"
        )

        # 3. The generated surface (D3e).
        (scratch / "dist").mkdir(parents=True, exist_ok=True)
        (scratch / "dist" / "cli-surface.json").write_text(
            cli_surface_json(), encoding="utf-8"
        )

        # 4. The server, the server declaration, the manifest, the licence.
        (scratch / "mcp").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(server_source, scratch / "mcp" / "server.py")
        shutil.copyfile(MCP_SOURCE, scratch / ".mcp.json")
        (scratch / ".codex-plugin").mkdir(parents=True, exist_ok=True)
        version = project_version(skill_base)
        (scratch / ".codex-plugin" / "plugin.json").write_text(
            manifest_text(version), encoding="utf-8"
        )
        shutil.copyfile(checkout / "LICENSE", scratch / "LICENSE")

        # 5. The restated ingestion rules over what will install.
        errors = ingestion_rules.validate_manifest(
            json.loads((scratch / ".codex-plugin" / "plugin.json").read_text("utf-8")),
            scratch,
        ) + ingestion_rules.validate_layout(scratch)
        if errors:
            raise PayloadError(
                "the built plugin breaks the restated ingestion rules:\n"
                + "\n".join(f"    {line}" for line in errors)
            )

        # 6. The sweep again, over the whole tree this time (the six files
        #    above are not the payload stage() swept), then the digest, then
        #    the exact inventory — the digest cannot stand in for it, because a
        #    stale file is *inside* a digest.
        scan_tree(scratch)
        digest = digest_of(scratch)
        (scratch / DIGEST_RELATIVE).write_text(digest + "\n", encoding="utf-8")
        _inventory(scratch)

        # 7. The swap.
        if out.exists():
            retired = Path(tempfile.mkdtemp(
                prefix=f".{out.name}.old.", dir=str(out.parent)
            ))
            shutil.rmtree(retired)
            out.replace(retired)
            try:
                scratch.replace(out)
            except OSError:
                retired.replace(out)
                raise
            shutil.rmtree(retired, ignore_errors=True)
        else:
            scratch.replace(out)
        scratch = None  # type: ignore[assignment]
    finally:
        if scratch is not None and scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

    return {
        "destination": str(out),
        "payload_files": len(staged),
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
# The publish gate's stamp
# ---------------------------------------------------------------------------

def _stamp_path(stamp: Optional[Path]) -> Path:
    return Path(stamp) if stamp else PACKAGE_DIR / STAMP_NAME


def _stamped_tiers(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """What the stamp keeps of each tier's evidence: enough to say which codex
    builds the gate ran, on which image, and what row 1 actually found."""
    kept = {}
    for tier in gate.TIERS:
        item = evidence[tier]
        kept[tier] = {
            "session": item.get("session"),
            "run": item.get("run"),
            "codex_version": item.get("codex_version"),
            "image": item.get("image"),
            "image_digest": item.get("image_digest"),
            "rows": sorted(item.get("rows") or {}),
            "upstream_validator": (item.get("upstream_validator") or {}).get("status"),
            "features_diff": (item.get("features") or {}).get("diff") or {},
            "checked_at": item.get("checked_at"),
        }
    return kept


def write_stamp(
    *, timestamp: str = "", stamp: Optional[Path] = None,
    root: Optional[Path] = None, evidence_dir: Optional[Path] = None,
) -> Path:
    """Record that the full gate passed **on the committed bytes**.

    Rebuilds from clean inputs into the checked-in tree and refuses unless that
    produced no diff against the commit: a stamp over bytes nobody committed
    would authorize a tag over a tree the tag does not contain. Takes no
    ``allow_dirty`` at all — the environment opt-in is for ``codex-regen``.

    Then refuses unless ``release_gate.py`` left passing evidence for every tier
    (pinned, the declared minimum, npm's newest) over this exact digest and
    version: the stamp is the gate's record, so it cannot be written for rows
    that never ran (A4; plans/codex-plugin-publish.md).
    """
    report = build(root=root)
    drift = committed_drift(root)
    if drift:
        raise PayloadError(
            "the clean rebuild differs from the committed plugin tree:\n"
            + "\n".join(f"    {line}" for line in drift[:20])
            + "\nRun `make codex-regen`, commit, and re-run the gate."
        )
    directory = Path(evidence_dir) if evidence_dir else gate.EVIDENCE_DIR
    evidence = gate.load_evidence(directory)
    problems = gate.evidence_problems(evidence, report["digest"], report["version"],
                                      session=gate.current_session(directory))
    if problems:
        raise PayloadError(
            "the release gate's container rows did not all pass on these bytes:\n"
            + "\n".join(f"    {line}" for line in problems)
            + "\nRun `make codex-package-check`; it writes this evidence and the stamp."
        )
    path = _stamp_path(stamp)
    path.write_text(json.dumps({
        "tier": FULL_TIER,
        "digest": report["digest"],
        "version": report["version"],
        "stamped_at": timestamp,
        "gate": _stamped_tiers(evidence),
    }, indent=2) + "\n", encoding="utf-8")
    return path


def check_stamp(
    *, stamp: Optional[Path] = None, root: Optional[Path] = None,
    tree: Optional[Path] = None,
) -> Optional[str]:
    """``None`` if a publish may proceed, else the reason it may not.

    Reads the **committed** tree: its recorded digest must be what its bytes
    hash to, it must carry no uncommitted change, and the stamp must name that
    digest and the owner's version. Each condition is its own message.
    """
    path = _stamp_path(stamp)
    plugin = Path(tree) if tree else PLUGIN_DIR
    if not path.is_file():
        return (
            f"no {path}. Run `make codex-package-check` first — the full gate "
            "is what writes it."
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return f"{path} is not readable JSON ({exc}). Re-run the full gate."
    if record.get("tier") != FULL_TIER:
        return (
            f"the stamp says tier={record.get('tier')!r}. Only a full "
            "`make codex-package-check` authorizes a publish; the CI tier writes "
            "no stamp."
        )
    tiers = record.get("gate")
    if not isinstance(tiers, Mapping) or any(
            not (tiers.get(tier) or {}).get("rows") for tier in gate.TIERS):
        return (
            f"the stamp names no passing container rows for every gate tier "
            f"({', '.join(gate.TIERS)}); it predates the wired gate or was written "
            "by hand. Re-run `make codex-package-check`."
        )
    if tree is None:
        drift = committed_drift(root)
        if drift:
            return (
                "the plugin tree has uncommitted changes — a tag would not "
                "contain them:\n" + "\n".join(f"    {line}" for line in drift[:20])
            )
    recorded_file = plugin / DIGEST_RELATIVE
    if not recorded_file.is_file():
        return f"{recorded_file} is missing; the tree was never built."
    recorded = recorded_file.read_text(encoding="utf-8").strip()
    actual = digest_of(plugin)
    if recorded != actual:
        return (
            f"{recorded_file} says {recorded} but the tree hashes to {actual}: "
            "the checked-in tree was edited by hand. Run `make codex-regen`."
        )
    if record.get("digest") != actual:
        return (
            f"the tree has changed since the gate ran (stamp "
            f"{record.get('digest')}, now {actual}). Re-run "
            "`make codex-package-check`."
        )
    owner = project_version()
    shipped = json.loads(
        (plugin / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    ).get("version")
    if record.get("version") != owner or shipped != owner:
        return (
            f"version mismatch: stamp {record.get('version')!r}, plugin.json "
            f"{shipped!r}, pyproject.toml {owner!r}."
        )
    return None


# ---------------------------------------------------------------------------
# Bump discipline (D6, A4 row 10): digest moved => version moved, release to
# release, read out of git rather than out of a recorded copy
# ---------------------------------------------------------------------------

TAG_PREFIX = "codex-v"
RELEASE_REF = "release"
DIGEST_PATH = f"{PLUGIN_RELATIVE}/{DIGEST_RELATIVE}"
VERSION_PATH = "skills/multi-llm/pyproject.toml"


def _version_key(version: str) -> Tuple[Any, ...]:
    core, _, pre = version.partition("+")[0].partition("-")
    numbers = tuple(int(part) for part in core.split("."))
    # A release sorts after its own pre-releases.
    return numbers + ((1, "") if not pre else (0, pre),)


def release_tags(root: Optional[Path] = None) -> Tuple[str, ...]:
    """Every ``codex-v*`` tag, oldest version first."""
    checkout = Path(root) if root else REPO_ROOT
    out = payload._git(checkout, "tag", "--list", f"{TAG_PREFIX}*")  # noqa: SLF001
    tags = [line.strip() for line in out.splitlines() if line.strip()]
    parsed = []
    for tag in tags:
        version = tag[len(TAG_PREFIX):]
        if not ingestion_rules.SEMVER_RE.fullmatch(version):
            raise PayloadError(f"tag {tag} does not name a strict semver version")
        parsed.append((_version_key(version), tag))
    return tuple(tag for _, tag in sorted(parsed))


def _show(checkout: Path, ref: str, path: str) -> str:
    try:
        return payload._git(checkout, "show", f"{ref}:{path}")  # noqa: SLF001
    except PayloadError as exc:
        raise PayloadError(f"`git show {ref}:{path}` did not resolve: {exc}") from exc


def _version_in(pyproject_text: str) -> str:
    section = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", pyproject_text, re.M | re.S)
    match = section and re.search(r'^version\s*=\s*"([^"]+)"\s*$', section.group(1), re.M)
    if not match:
        raise PayloadError("pyproject.toml declares no [project] version")
    return match.group(1)


def release_record(ref: str, root: Optional[Path] = None) -> Tuple[str, str]:
    """``(digest, version)`` as committed at *ref*."""
    checkout = Path(root) if root else REPO_ROOT
    return (
        _show(checkout, ref, DIGEST_PATH).strip(),
        _version_in(_show(checkout, ref, VERSION_PATH)),
    )


def bump_history_problems(root: Optional[Path] = None) -> Tuple[str, ...]:
    """The history half: adjacent ``codex-v*`` tags whose digests differ carry
    different versions, and no version is carried by two tags."""
    tags = release_tags(root)
    problems = []
    records = [(tag, *release_record(tag, root)) for tag in tags]
    seen: Dict[str, str] = {}
    for tag, digest, version in records:
        if version in seen:
            problems.append(f"{tag} and {seen[version]} both carry version {version}")
        seen.setdefault(version, tag)
        if tag[len(TAG_PREFIX):].split("+")[0] != version.split("+")[0]:
            problems.append(f"{tag} names {tag[len(TAG_PREFIX):]} but its pyproject says {version}")
    for (prev, d0, v0), (cur, d1, v1) in zip(records, records[1:]):
        if d0 != d1 and v0 == v1:
            problems.append(
                f"{prev} -> {cur}: the payload digest moved but the version "
                f"stayed {v0}"
            )
    return tuple(problems)


def bump_ahead_problem(
    root: Optional[Path] = None, tree: Optional[Path] = None,
) -> Optional[str]:
    """The publish half: the working tree against the newest tag."""
    tags = release_tags(root)
    if not tags:
        return None
    newest = tags[-1]
    digest, version = release_record(newest, root)
    plugin = Path(tree) if tree else PLUGIN_DIR
    current = (plugin / DIGEST_RELATIVE).read_text(encoding="utf-8").strip()
    owner = project_version()
    if current != digest and owner == version:
        return (
            f"the plugin digest moved since {newest} ({digest} -> {current}) but "
            f"the version is still {owner}. Bump skills/multi-llm/pyproject.toml "
            "(and its mirrors, A6 step 1), then `make codex-regen` and commit."
        )
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Build the codex plugin tree packaging/codex/multi-llm/.",
    )
    parser.add_argument(
        "--destination", type=Path, default=None,
        help="where to build (default: the checked-in packaging/codex/multi-llm)",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        default=allow_dirty_from_env(),
        help=(
            "build from inputs with uncommitted changes — what `make codex-regen` "
            "does, since the regenerated tree lands in the same commit as the "
            f"edit that implies it. Also ${ALLOW_DIRTY_ENV}={ALLOW_DIRTY_TRUE} "
            "(that exact spelling). Never reaches a stamp or a publish."
        ),
    )
    parser.add_argument(
        "--require-no-diff", action="store_true",
        help=(
            "after building into the checked-in tree, fail unless it equals the "
            "committed one (what `make codex-package` asserts)"
        ),
    )
    parser.add_argument(
        "--write-stamp", metavar="TIMESTAMP", default=None,
        help="record that the full gate passed on the committed bytes",
    )
    parser.add_argument(
        "--check-stamp", action="store_true",
        help="exit non-zero unless the full gate passed on these exact bytes",
    )
    parser.add_argument(
        "--check-bump", action="store_true",
        help=(
            "exit non-zero if the committed digest moved since the newest "
            "codex-v* tag under an unmoved version (A4 row 10)"
        ),
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
        if args.check_bump:
            problems = list(bump_history_problems())
            ahead = bump_ahead_problem()
            if ahead:
                problems.append(ahead)
            if problems:
                for line in problems:
                    print(f"REFUSING: {line}", file=sys.stderr)
                return 1
            print("bump discipline holds")
            return 0
        if args.check_stamp:
            reason = check_stamp()
            if reason:
                print(f"REFUSING: {reason}", file=sys.stderr)
                return 1
            print("the full gate passed on these exact bytes")
            return 0
        if args.write_stamp is not None:
            print(f"wrote {write_stamp(timestamp=args.write_stamp)}")
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
                    + "\nRun `make codex-regen` and commit the result with the "
                    "source edit that implies it.",
                    file=sys.stderr,
                )
                return 1
    except PayloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"built {report['payload_files']} payload files into "
        f"{report['destination']}\n  version {report['version']}\n"
        f"  digest {report['digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
