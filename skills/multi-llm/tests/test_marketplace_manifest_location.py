"""The two marketplace catalogs this repository ships, and the trap between them.

Codex resolves a marketplace from **four** locations, and reads them in a fixed
precedence: `.agents/plugins/marketplace.json`, `.agents/plugins/api_marketplace.json`,
`.claude-plugin/marketplace.json`, `.cursor-plugin/marketplace.json`. That table
lives once in the binary (`core-plugins/src/marketplace.rs`) and serves BOTH the
local-path route and the Git route — the Git route clones first and then resolves
the same four paths against the clone root — which is why one file guards both.

The trap is that this repo has shipped `.claude-plugin/marketplace.json` since long
before anyone thought about codex, and its single entry reads `"source": "./"`
because for Claude Code the plugin *is* the repository. Codex reads that file too.
Measured on codex-cli 0.149.1 against this checkout:

    codex plugin marketplace add <repo root>   ->  Added marketplace `beastlabai`
                                                  (from .claude-plugin/marketplace.json)
    codex plugin add multi-llm@beastlabai      ->  264,636,084 bytes copied into
                                                  plugins/cache/beastlabai/multi-llm/local/

`.git`, `.venv`, the gitignored `plans/` scratch and the E2E evidence bundles, all
of it — because **nothing that copies a working tree consults `.gitignore`**, and
codex honours no ignore file of any kind (no `.codexignore`, no `.agentsignore`; the
only ignore machinery in the binary belongs to its code-search walker). The version
segment is the literal string `local`, which is what an ingested manifest carrying
no `version` key produces — and `.claude-plugin/plugin.json` carries none, because
Claude Code does not need one.

`.agents/plugins/marketplace.json` outranks it, so shipping that file is what
decides which of those two things happens. This module asserts the properties that
make it safe, in both the state it is in today (an empty catalog, so the command
fails cleanly instead of succeeding wrongly) and the state Phase 1 leaves it in
(one entry pointing at `packaging/codex/multi-llm`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS_CATALOG = REPO_ROOT / ".agents" / "plugins" / "marketplace.json"
CLAUDE_CATALOG = REPO_ROOT / ".claude-plugin" / "marketplace.json"

#: `policy.authentication` is an enum of exactly these two. A plausible-looking
#: `"NONE"` is not a rejected entry — it is a parse error at `marketplace add`
#: that takes the whole catalog down with it.
AUTH_POLICIES = {"ON_INSTALL", "ON_USE"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_source_path(entry: dict) -> str | None:
    """The relative path an entry installs from, in either accepted spelling.

    `source` is a bare relative-path string OR an object
    `{"source": "local", "path": "./x"}`. Both install identically.
    """
    source = entry.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, dict):
        return source.get("path")
    return None


def test_the_agents_catalog_exists_and_outranks_the_claude_one() -> None:
    """Deleting this file re-arms the whole-repo copy. That is the whole point."""
    assert AGENTS_CATALOG.is_file(), (
        f"{AGENTS_CATALOG.relative_to(REPO_ROOT)} is missing. Without it codex "
        "falls through to .claude-plugin/marketplace.json, whose entry is "
        "'source': './' — and `codex plugin add` copies the entire working tree."
    )


def test_both_catalogs_present_one_identity() -> None:
    """A user types one marketplace name; two files must not disagree on it."""
    assert _load(AGENTS_CATALOG)["name"] == _load(CLAUDE_CATALOG)["name"]


@pytest.mark.parametrize("catalog", [AGENTS_CATALOG, CLAUDE_CATALOG],
                         ids=["agents", "claude"])
def test_a_catalog_parses_and_is_shaped_like_a_catalog(catalog: Path) -> None:
    data = _load(catalog)
    assert isinstance(data.get("name"), str) and data["name"]
    assert isinstance(data.get("plugins"), list)


def test_no_entry_installs_from_the_repository_root() -> None:
    """`./` is the footgun in one character. It is also the Claude Code spelling.

    So this is asserted on the codex catalog only: `.claude-plugin`'s `"./"` is
    correct for the host that owns that file, and is exactly what the file above
    exists to shadow.
    """
    for entry in _load(AGENTS_CATALOG)["plugins"]:
        source = _entry_source_path(entry)
        assert source is not None, f"entry {entry.get('name')!r} has no source path"
        normalised = source.rstrip("/") or "."
        assert normalised not in {".", "./", ""}, (
            f"entry {entry.get('name')!r} installs from the repository root; "
            "that copies the whole working tree, .git and gitignored files "
            "included, because codex honours no ignore file."
        )
        assert not source.startswith(("/", "~")), (
            f"entry {entry.get('name')!r} has a non-relative source {source!r}"
        )
        assert ".." not in Path(source).parts, (
            f"entry {entry.get('name')!r} escapes the repository root; codex "
            "skips such an entry silently at manifest-load time, and the failure "
            "surfaces much later as 'plugin was not found in marketplace'."
        )


def test_every_entry_resolves_to_a_real_installable_plugin() -> None:
    """Vacuous today (the catalog is empty); load-bearing the moment it is not.

    Each assertion here is a measured failure mode, not a style rule:
    a missing `.codex-plugin/plugin.json` is `Error: missing plugin.json`; a name
    mismatch is a refused install naming both names; and a manifest with no
    `version` installs to a cache segment of the literal string `local`, which no
    version-pinned resolution can match.
    """
    for entry in _load(AGENTS_CATALOG)["plugins"]:
        name = entry.get("name")
        source = _entry_source_path(entry)
        plugin_dir = (REPO_ROOT / source).resolve()
        assert plugin_dir.is_dir(), f"{name!r}: {source} is not a directory"

        manifest_path = plugin_dir / ".codex-plugin" / "plugin.json"
        assert manifest_path.is_file(), (
            f"{name!r}: no .codex-plugin/plugin.json — `codex plugin add` fails "
            "with 'Error: missing plugin.json' and caches nothing"
        )
        manifest = _load(manifest_path)
        assert manifest.get("name") == name, (
            f"{name!r}: plugin.json name {manifest.get('name')!r} does not match "
            "the marketplace entry name; codex refuses the install"
        )
        assert manifest.get("version"), (
            f"{name!r}: plugin.json carries no version, so the install lands in a "
            "cache directory whose version segment is the literal 'local'"
        )


def test_no_entry_carries_an_unparseable_auth_policy() -> None:
    """One bad enum value is not one bad entry — it is a dead catalog."""
    for entry in _load(AGENTS_CATALOG)["plugins"]:
        policy = entry.get("policy") or {}
        if "authentication" in policy:
            assert policy["authentication"] in AUTH_POLICIES, (
                f"{entry.get('name')!r}: policy.authentication "
                f"{policy['authentication']!r} is not one of {sorted(AUTH_POLICIES)}; "
                "codex fails to parse the whole marketplace file, not just this entry"
            )


def test_the_payload_carries_no_symlink_the_install_would_drop() -> None:
    """Measured, and worse than the dangling link anyone would have predicted.

    `codex plugin add` neither follows nor preserves a symlink: it **omits** it,
    creating the parent directory empty, exiting 0 with no diagnostic. A packaging
    tree whose `skills/` is a link back at the source tree therefore installs as an
    empty `skills/` that reports success. Both routes lose them in the same place —
    the Git *fetch* preserves symlinks, the copy into the version-keyed cache is
    where they disappear — so this cannot be worked around by choosing a route.
    """
    for entry in _load(AGENTS_CATALOG)["plugins"]:
        plugin_dir = (REPO_ROOT / _entry_source_path(entry)).resolve()
        links = sorted(p for p in plugin_dir.rglob("*") if p.is_symlink())
        assert not links, (
            f"{entry.get('name')!r}: symlinks under the packaged tree are silently "
            f"dropped by the install: {[str(p.relative_to(REPO_ROOT)) for p in links]}"
        )
