"""The two marketplace catalogs this repository ships, and what keeps them apart.

Each host installs its own small generated tree. `.claude-plugin/marketplace.json`
selects `packaging/claude/multi-llm`, and `.agents/plugins/marketplace.json` selects
`packaging/codex/multi-llm`. The registration commands in `README.md` and
`docs/installation.md` are a contract with the first of those: their `--sparse` paths
have to reach the catalog and the directory it names, so a source move that leaves
the documents behind fails here rather than on a user's machine.

Codex resolves a marketplace from **four** locations, and reads them in a fixed
precedence: `.agents/plugins/marketplace.json`, `.agents/plugins/api_marketplace.json`,
`.claude-plugin/marketplace.json`, `.cursor-plugin/marketplace.json`. That table
lives once in the binary (`core-plugins/src/marketplace.rs`) and serves BOTH the
local-path route and the Git route — the Git route clones first and then resolves
the same four paths against the clone root — which is why one file guards both.

So codex reads Claude's catalog too, whenever a checkout has no `.agents` catalog,
and it would install that entry. Measured on codex-cli 0.167.0 (2026-11-06) with only
the Claude catalog and its source in the checkout, a plain entry gives:

    codex plugin marketplace add <repo>     ->  Added marketplace `beastlabai`
    codex plugin add multi-llm@beastlabai   ->  exit 0, 234 files in
                                                plugins/cache/beastlabai/multi-llm/<version>/

Codex takes `.claude-plugin/plugin.json` as the plugin manifest, and `codex debug
prompt-input` then shows the `multi-llm:multi-llm` skill with no `multi_llm` tool,
because only the Codex tree carries `.mcp.json`. A missing `.codex-plugin/plugin.json`
does not stop it. So the Claude entry carries `policy.installation: NOT_AVAILABLE`, which
Codex refuses with `plugin `multi-llm` is not available for install`, leaving no
payload, and which Claude Code ignores. The `.agents` file has to exist, outrank the
other one, and never turn into a copy of it.

The repository-root guards come from an older failure of the same shape. The Claude
catalog's entry used to read `"source": "./"`, and measured on codex-cli 0.149.1 the
same fallback copied 264,636,084 bytes into `plugins/cache/beastlabai/multi-llm/local/`:
`.git`, `.venv`, the gitignored `plans/` scratch and the E2E evidence bundles, because
**nothing that copies a working tree consults `.gitignore`**, and codex honours no
ignore file of any kind. Claude Code itself copied all 2,363 tracked files
(53,128,937 bytes). One character in either catalog would bring that back.

The checks are helpers that take a catalog and a root, so every failure they exist to
catch runs here against a fixture.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, List, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS_CATALOG = REPO_ROOT / ".agents" / "plugins" / "marketplace.json"
CLAUDE_CATALOG = REPO_ROOT / ".claude-plugin" / "marketplace.json"

_SKILL_ROOT = str(Path(__file__).resolve().parent.parent)
if _SKILL_ROOT not in sys.path:
    sys.path.insert(0, _SKILL_ROOT)

from tests.generated_distributions import CLAUDE, CODEX, DISTRIBUTIONS, Distribution  # noqa: E402

#: `policy.authentication` is an enum of exactly these two. A plausible-looking
#: `"NONE"` is not a rejected entry — it is a parse error at `marketplace add`
#: that takes the whole catalog down with it.
AUTH_POLICIES = {"ON_INSTALL", "ON_USE"}

PLUGIN_NAME = "multi-llm"
REPOSITORY = "beastlabai/multi-llm-plugin"
CLAUDE_SOURCE = f"./{CLAUDE.root.rstrip('/')}"
CODEX_SOURCE = f"./{CODEX.root.rstrip('/')}"

#: The documents whose fenced commands are a contract with the Claude catalog.
REGISTRATION_DOCUMENTS = ("README.md", "docs/installation.md")

#: The directory holding the catalog. A checkout without it has no marketplace.
CLAUDE_CATALOG_DIR = ".claude-plugin"

#: Every file name codex or Claude Code would read as a catalog.
CATALOG_FILE_NAMES = ("marketplace.json", "api_marketplace.json")


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


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def location_problems(catalog: dict, root: Path) -> List[str]:
    """Every entry in *catalog* that points at the repository root or out of it.

    `./` is one character, and it installs everything: the whole working tree for
    codex, which honours no ignore file, and every tracked file for Claude Code. A
    path that leaves the root is worse to debug: codex skips such an entry silently at
    manifest-load time, and the failure surfaces much later as 'plugin was not found
    in marketplace'.
    """
    problems = []
    top = root.resolve()
    for entry in catalog.get("plugins") or []:
        name = entry.get("name")
        source = _entry_source_path(entry)
        if source is None:
            problems.append(f"entry {name!r} has no source path")
            continue
        if source.startswith(("/", "~")):
            problems.append(f"entry {name!r} has a non-relative source {source!r}")
            continue
        if ".." in PurePosixPath(source).parts:
            problems.append(f"entry {name!r} escapes the repository root: {source!r}")
            continue
        resolved = (root / source).resolve()
        if resolved == top:
            problems.append(
                f"entry {name!r} installs from the repository root ({source!r}); that "
                "copies the whole repository into the plugin cache"
            )
        elif top not in resolved.parents:
            problems.append(f"entry {name!r} escapes the repository root: {source!r}")
    return problems


def entry_problems(catalog: dict, root: Path, distribution: Distribution) -> List[str]:
    """Why *catalog*'s plugin entry does not install *distribution*'s generated tree."""
    expected = f"./{distribution.root.rstrip('/')}"
    entries = [e for e in catalog.get("plugins") or [] if e.get("name") == PLUGIN_NAME]
    if len(entries) != 1:
        return [f"expected one {PLUGIN_NAME!r} entry, found {len(entries)}"]
    source = _entry_source_path(entries[0])
    problems = []
    if source != expected:
        problems.append(f"source is {source!r}; it must be {expected!r}")
    if not isinstance(source, str):
        return problems
    wanted = distribution.root.rstrip("/")
    plugin_dir = (root / source).resolve()
    if plugin_dir != (root / wanted).resolve():
        problems.append(f"source {source!r} resolves to {plugin_dir}, not {wanted}")
    manifest_path = plugin_dir / distribution.version_manifest
    if not manifest_path.is_file():
        problems.append(f"{source!r} has no {distribution.version_manifest}")
        return problems
    try:
        manifest_name = _load(manifest_path).get("name")
    except ValueError as exc:
        problems.append(f"{distribution.version_manifest} does not parse: {exc}")
        return problems
    if manifest_name != PLUGIN_NAME:
        problems.append(
            f"{distribution.version_manifest} names {manifest_name!r}, but the catalog "
            f"entry names {PLUGIN_NAME!r}"
        )
    return problems


def manifest_boundary_problems(root: Path) -> List[str]:
    """A generated tree carrying the other host's manifest.

    The Codex tree must not offer a Claude manifest, and the Claude tree must not
    offer a Codex one. Adding a Codex manifest to the Claude tree would make the
    fallback look deliberate without giving it the tool. `.mcp.json` is held to the
    same line because `docs/installation.md` tells users the Claude package has no
    `multi_llm` tool.
    """
    problems = []
    claude_tree, codex_tree = root / CLAUDE.root, root / CODEX.root
    for tree, label, foreign in (
        (claude_tree, "Claude", CODEX.version_manifest),
        (claude_tree, "Claude", ".mcp.json"),
        (codex_tree, "Codex", CLAUDE.version_manifest),
    ):
        if (tree / foreign).exists():
            problems.append(f"the {label} tree carries {foreign}")
    return problems


def catalogs_inside(tree: Path) -> List[str]:
    """Every marketplace catalog inside a generated plugin *tree*, relative to it."""
    return sorted(
        path.relative_to(tree).as_posix()
        for name in CATALOG_FILE_NAMES
        for path in tree.rglob(name)
    )


def stale_copy_problems(agents: dict, claude: dict, root: Path) -> List[str]:
    """Signs that the `.agents` catalog was copied from, or still describes, Claude's."""
    problems = []
    claude_dirs = {
        (root / _entry_source_path(entry)).resolve()
        for entry in claude.get("plugins") or []
        if _entry_source_path(entry) is not None
    }
    for entry in agents.get("plugins") or []:
        source = _entry_source_path(entry)
        if source is not None and (root / source).resolve() in claude_dirs:
            problems.append(
                f".agents entry {entry.get('name')!r} points at the Claude source {source!r}"
            )
    description = agents.get("description") or ""
    if "source './'" in description or 'source "./"' in description:
        problems.append("the .agents description still calls the Claude source './'")
    for entry in claude.get("plugins") or []:
        source = _entry_source_path(entry)
        if source is not None and source not in description:
            problems.append(
                f"the .agents description does not name the Claude source {source!r}"
            )
    return problems


# ---------------------------------------------------------------------------
# Documentation helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Registration:
    """One `<tool> plugin marketplace add` command read out of a fenced block."""

    line: int
    source: Optional[str]
    sparse: Tuple[str, ...]


def _fenced_blocks(text: str) -> Iterator[List[Tuple[int, str]]]:
    """Each fenced code block in *text* as `(line_number, line)` pairs.

    Prose and inline code are skipped on purpose: a failing form quoted in a sentence
    is not a recommendation. A fence may be indented or inside a blockquote.
    """
    fence = None
    block: List[Tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip("> \t")
        if fence is None:
            if stripped.startswith(("```", "~~~")):
                fence = stripped[:3]
                block = []
            continue
        if stripped.startswith(fence):
            yield block
            fence = None
            continue
        block.append((number, stripped))


def _logical_lines(block: Sequence[Tuple[int, str]]) -> Iterator[Tuple[int, str]]:
    """Join `\\` continuations, keeping the number of the first physical line."""
    pending = ""
    first = None
    for number, line in block:
        first = number if first is None else first
        trimmed = line.rstrip()
        if trimmed.endswith("\\"):
            pending += trimmed[:-1] + " "
            continue
        yield first, pending + trimmed
        pending, first = "", None
    if pending:
        yield first, pending


def _simple_commands(line: str) -> Iterator[List[str]]:
    """The argv of each command on a shell line, split at `&&`, `;`, `|` and friends."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens: List[str] = []
    for token in lexer:
        if token and set(token) <= set("();<>|&"):
            if tokens:
                yield tokens
            tokens = []
        else:
            tokens.append(token)
    if tokens:
        yield tokens


def _parse_add(number: int, arguments: Sequence[str]) -> Registration:
    """Read a source and `--sparse` paths the way the CLI's option parser does.

    Claude's `--sparse` is variadic and takes every value up to the next option, so
    a source written after it is swallowed. Measured: `error: missing required
    argument 'source'`. Codex takes one path per flag, and this reads both spellings.
    """
    source = None
    sparse: List[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        index += 1
        if argument == "--sparse":
            while index < len(arguments) and not arguments[index].startswith("-"):
                sparse.append(arguments[index])
                index += 1
        elif argument.startswith("--sparse="):
            sparse.append(argument.split("=", 1)[1])
        elif argument in ("--scope", "--ref"):
            index += 1
        elif not argument.startswith("-") and source is None:
            source = argument
    return Registration(number, source, tuple(sparse))


def registrations(text: str, tool: str = "claude") -> List[Registration]:
    """Every `<tool> plugin marketplace add` command in *text*'s fenced blocks.

    The in-session `/plugin marketplace add` does not match. It takes no sparse paths,
    and the documents offer it as the shorter alternative.
    """
    words = [tool, "plugin", "marketplace", "add"]
    pattern = re.compile(r"\b" + r"\s+".join(words) + r"\b")
    found = []
    for block in _fenced_blocks(text):
        for number, line in _logical_lines(block):
            if not pattern.search(line):
                continue
            for argv in _simple_commands(line):
                if argv and argv[0] == "!":
                    argv = argv[1:]
                if argv[:4] == words:
                    found.append(_parse_add(number, argv[4:]))
    return found


def _sparse_is_usable(path: str) -> bool:
    """False for a path naming the repository root, or one outside it."""
    pure = PurePosixPath(path)
    return bool(pure.parts) and not pure.is_absolute() and ".." not in pure.parts


def sparse_covers(sparse_path: str, required: str) -> bool:
    """Whether a sparse-checkout path includes *required*, component by component.

    A string prefix would let `packaging/claudeX` pass for `packaging/claude/multi-llm`.
    """
    if not _sparse_is_usable(sparse_path):
        return False
    have = PurePosixPath(sparse_path).parts
    return PurePosixPath(required).parts[: len(have)] == have


def registration_problems(text: str, document: str, catalog_source: str) -> List[str]:
    """Why *document*'s Claude registrations would not fetch what the catalog selects."""
    found = registrations(text)
    if not found:
        return [f"{document}: no `claude plugin marketplace add` command in a fenced block"]
    problems = []
    source_parts = PurePosixPath(catalog_source).parts
    required = [CLAUDE_CATALOG_DIR]
    if source_parts:
        required.append(PurePosixPath(catalog_source).as_posix())
    else:
        problems.append(
            f"{document}: the catalog source {catalog_source!r} is the repository root, "
            "and no sparse path can select it without fetching everything"
        )
    for registration in found:
        where = f"{document}:{registration.line}"
        if registration.source != REPOSITORY:
            problems.append(
                f"{where}: source is {registration.source!r}, not {REPOSITORY!r}; "
                "--sparse must come after the source"
            )
        if not registration.sparse:
            problems.append(f"{where}: no --sparse paths")
        for path in registration.sparse:
            if not _sparse_is_usable(path):
                problems.append(
                    f"{where}: sparse path {path!r} selects the repository root or "
                    "leaves it, which fetches the development files it should exclude"
                )
        for path in required:
            if not any(sparse_covers(have, path) for have in registration.sparse):
                problems.append(
                    f"{where}: --sparse {list(registration.sparse)} does not cover {path!r}"
                )
    return problems


def _claude_source(catalog: dict) -> str:
    entries = [e for e in catalog["plugins"] if e.get("name") == PLUGIN_NAME]
    assert len(entries) == 1, entries
    return _entry_source_path(entries[0])


# ---------------------------------------------------------------------------
# A small repository to break
# ---------------------------------------------------------------------------

def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _claude_catalog(source: object = CLAUDE_SOURCE) -> dict:
    return {"name": "beastlabai", "plugins": [{"name": PLUGIN_NAME, "source": source}]}


def _agents_catalog(path: str = CODEX_SOURCE, description: Optional[str] = None) -> dict:
    return {
        "name": "beastlabai",
        "description": description or f"It shadows the catalog whose source is {CLAUDE_SOURCE}.",
        "plugins": [{"name": PLUGIN_NAME, "source": {"source": "local", "path": path}}],
    }


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Both generated trees, each with only its own manifest."""
    for distribution in DISTRIBUTIONS:
        _write_json(
            tmp_path / distribution.root / distribution.version_manifest,
            {"name": PLUGIN_NAME, "version": "1.2.3"},
        )
    return tmp_path


# ---------------------------------------------------------------------------
# The real catalogs
# ---------------------------------------------------------------------------

def test_the_agents_catalog_exists_and_outranks_the_claude_one() -> None:
    """Deleting this file makes codex install the wrong package, and report success."""
    assert AGENTS_CATALOG.is_file(), (
        f"{AGENTS_CATALOG.relative_to(REPO_ROOT)} is missing. Without it codex falls "
        "through to .claude-plugin/marketplace.json and `codex plugin add` installs "
        "the Claude Code package, which has the skill but no multi_llm tool."
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


@pytest.mark.parametrize("catalog", [AGENTS_CATALOG, CLAUDE_CATALOG],
                         ids=["agents", "claude"])
def test_no_entry_installs_from_the_repository_root_or_outside_it(catalog: Path) -> None:
    """The guard covers both files, since Claude Code installs from the second one."""
    data = _load(catalog)
    assert data["plugins"], f"{catalog} has no entries, so this row proves nothing"
    assert location_problems(data, REPO_ROOT) == []


def test_the_claude_entry_selects_the_generated_claude_tree() -> None:
    """The source is what limits the installed copy, whichever way a user registers."""
    assert CLAUDE_SOURCE == "./packaging/claude/multi-llm"
    catalog = _load(CLAUDE_CATALOG)
    assert entry_problems(catalog, REPO_ROOT, CLAUDE) == []
    assert (REPO_ROOT / _claude_source(catalog)).resolve() == (
        REPO_ROOT / "packaging" / "claude" / "multi-llm"
    ).resolve()


def test_the_agents_entry_selects_the_generated_codex_tree() -> None:
    """Catalog precedence only helps if the winner names the Codex package."""
    assert CODEX_SOURCE == "./packaging/codex/multi-llm"
    catalog = _load(AGENTS_CATALOG)
    assert entry_problems(catalog, REPO_ROOT, CODEX) == []
    entry = next(e for e in catalog["plugins"] if e["name"] == PLUGIN_NAME)
    assert (REPO_ROOT / _entry_source_path(entry) / ".codex-plugin" / "plugin.json").is_file()


def test_each_generated_tree_carries_only_its_own_manifest() -> None:
    assert (REPO_ROOT / CLAUDE.root / ".claude-plugin" / "plugin.json").is_file()
    assert (REPO_ROOT / CODEX.root / ".codex-plugin" / "plugin.json").is_file()
    assert manifest_boundary_problems(REPO_ROOT) == []


@pytest.mark.parametrize("distribution", DISTRIBUTIONS, ids=lambda d: d.root)
def test_no_generated_tree_contains_a_marketplace_catalog(distribution: Distribution) -> None:
    """A catalog inside a plugin would make that tree a marketplace of its own."""
    tree = REPO_ROOT / distribution.root
    assert tree.is_dir(), f"{distribution.root} is missing, so this row proves nothing"
    assert catalogs_inside(tree) == []


def test_the_agents_catalog_is_not_a_stale_copy_of_the_claude_one() -> None:
    assert stale_copy_problems(_load(AGENTS_CATALOG), _load(CLAUDE_CATALOG), REPO_ROOT) == []


def test_every_entry_resolves_to_a_real_installable_plugin() -> None:
    """Each assertion here is a measured way for a codex install to go wrong.

    A name mismatch is a refused install naming both names. A source with no manifest
    at all still installs on codex 0.167.0, under a manifest codex writes itself, and
    a manifest with no `version` lands in a cache segment of the literal string
    `local`, which no version-pinned resolution can match.
    """
    entries = _load(AGENTS_CATALOG)["plugins"]
    assert entries, "the .agents catalog has no entries"
    for entry in entries:
        name = entry.get("name")
        source = _entry_source_path(entry)
        plugin_dir = (REPO_ROOT / source).resolve()
        assert plugin_dir.is_dir(), f"{name!r}: {source} is not a directory"

        manifest_path = plugin_dir / ".codex-plugin" / "plugin.json"
        assert manifest_path.is_file(), (
            f"{name!r}: no .codex-plugin/plugin.json; codex would install it anyway, "
            "under a manifest it writes itself, at a cache segment of 'local'"
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


#: `policy.installation`'s enum, as codex 0.164.0 names it when it refuses a
#: catalog over a bad value (measured 2026-10-23: `unknown variant `BOGUS`,
#: expected one of `NOT_AVAILABLE`, `AVAILABLE`, `INSTALLED_BY_DEFAULT``).
INSTALL_POLICIES = {"NOT_AVAILABLE", "AVAILABLE", "INSTALLED_BY_DEFAULT"}


def test_codex_is_refused_the_claude_package() -> None:
    """Without this key, a checkout with no `.agents` catalog installs the Claude tree
    into Codex at exit 0, as a skill with no tool. Claude Code ignores the key."""
    for entry in _load(CLAUDE_CATALOG)["plugins"]:
        assert (entry.get("policy") or {}).get("installation") == "NOT_AVAILABLE", (
            f"{entry.get('name')!r}: the Claude catalog entry must carry "
            "policy.installation NOT_AVAILABLE, or Codex's fallback installs it"
        )


def test_the_codex_entry_stays_installable() -> None:
    """The same key on the catalog Codex is meant to use would refuse every install."""
    for entry in _load(AGENTS_CATALOG)["plugins"]:
        assert (entry.get("policy") or {}).get("installation") == "AVAILABLE", (
            f"{entry.get('name')!r}: the .agents entry must stay installable"
        )


@pytest.mark.parametrize("catalog", [AGENTS_CATALOG, CLAUDE_CATALOG],
                         ids=["agents", "claude"])
def test_no_entry_carries_an_unparseable_install_policy(catalog: Path) -> None:
    """The same dead-catalog failure, on the other policy key. Both catalogs, because
    Codex parses whichever one it falls back to."""
    for entry in _load(catalog)["plugins"]:
        policy = entry.get("policy") or {}
        if "installation" in policy:
            assert policy["installation"] in INSTALL_POLICIES, (
                f"{entry.get('name')!r}: policy.installation "
                f"{policy['installation']!r} is not one of {sorted(INSTALL_POLICIES)}; "
                "codex refuses the whole marketplace file"
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


# ---------------------------------------------------------------------------
# The real documents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("document", REGISTRATION_DOCUMENTS)
def test_the_documented_registrations_fetch_the_catalog_and_the_plugin(document: str) -> None:
    """Each document has at least one command, and each command reaches both paths."""
    text = (REPO_ROOT / document).read_text(encoding="utf-8")
    source = _claude_source(_load(CLAUDE_CATALOG))
    assert registration_problems(text, document, source) == []
    assert registrations(text), f"{document}: nothing was parsed"


@pytest.mark.parametrize("document", REGISTRATION_DOCUMENTS)
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("./", "is the repository root"),
        ("./packaging/claude-next/multi-llm", "does not cover 'packaging/claude-next/multi-llm'"),
    ],
    ids=["source-reverted-to-root", "source-moved"],
)
def test_the_real_documents_go_red_when_the_source_changes(
    document: str, source: str, expected: str
) -> None:
    """The documented paths are tied to the catalog rather than to a copy of its value."""
    text = (REPO_ROOT / document).read_text(encoding="utf-8")
    assert any(expected in problem for problem in registration_problems(text, document, source))


def test_the_installation_guide_shows_codex_only_the_complete_registration() -> None:
    """Its failing forms sit in prose. Any command a reader could paste has both paths."""
    text = (REPO_ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    found = registrations(text, tool="codex")
    assert found, "docs/installation.md has no codex registration in a fenced block"
    for registration in found:
        assert registration.sparse == (".agents", "packaging/codex"), registration


# ---------------------------------------------------------------------------
# The catalog helpers, on fixtures
# ---------------------------------------------------------------------------

def test_a_correct_fixture_passes_every_catalog_helper(repo: Path) -> None:
    """Without this row, helpers that always complain would pass the rows below."""
    assert location_problems(_claude_catalog(), repo) == []
    assert location_problems(_agents_catalog(), repo) == []
    assert entry_problems(_claude_catalog(), repo, CLAUDE) == []
    assert entry_problems(_agents_catalog(), repo, CODEX) == []
    assert manifest_boundary_problems(repo) == []
    assert stale_copy_problems(_agents_catalog(), _claude_catalog(), repo) == []
    for distribution in DISTRIBUTIONS:
        assert catalogs_inside(repo / distribution.root) == []


@pytest.mark.parametrize(
    "source",
    ["./packaging/claude/multi-lm", "./packaging/claude/multi-llm-old", "packaging/claude/multi-llm"],
    ids=["typo", "sibling", "no-dot-slash"],
)
def test_a_wrong_claude_source_fails(repo: Path, source: str) -> None:
    problems = entry_problems(_claude_catalog(source), repo, CLAUDE)
    assert any("must be './packaging/claude/multi-llm'" in p for p in problems), problems


def test_a_source_that_resolves_elsewhere_is_named(repo: Path) -> None:
    problems = entry_problems(_claude_catalog("./packaging/claude/multi-llm/.."), repo, CLAUDE)
    assert any("resolves to" in p and "not packaging/claude/multi-llm" in p for p in problems), problems


def test_a_missing_generated_manifest_fails(repo: Path) -> None:
    (repo / CLAUDE.root / CLAUDE.version_manifest).unlink()
    problems = entry_problems(_claude_catalog(), repo, CLAUDE)
    assert any("has no .claude-plugin/plugin.json" in p for p in problems), problems


def test_a_manifest_name_that_differs_from_the_entry_fails(repo: Path) -> None:
    _write_json(repo / CLAUDE.root / CLAUDE.version_manifest, {"name": "multi-llm-next"})
    problems = entry_problems(_claude_catalog(), repo, CLAUDE)
    assert any("names 'multi-llm-next'" in p for p in problems), problems


def test_a_manifest_that_does_not_parse_fails(repo: Path) -> None:
    (repo / CLAUDE.root / CLAUDE.version_manifest).write_text("{", encoding="utf-8")
    problems = entry_problems(_claude_catalog(), repo, CLAUDE)
    assert any("does not parse" in p for p in problems), problems


@pytest.mark.parametrize(
    "plugins",
    [[], [{"name": PLUGIN_NAME, "source": CLAUDE_SOURCE}] * 2],
    ids=["no-entry", "two-entries"],
)
def test_the_entry_has_to_be_there_exactly_once(repo: Path, plugins: list) -> None:
    problems = entry_problems({"name": "beastlabai", "plugins": plugins}, repo, CLAUDE)
    assert any("expected one 'multi-llm' entry" in p for p in problems), problems


@pytest.mark.parametrize(
    "source",
    ["./", ".", "", ".//", "./."],
    ids=["dot-slash", "dot", "empty", "dot-double-slash", "dot-slash-dot"],
)
@pytest.mark.parametrize(
    "shape",
    [_claude_catalog, _agents_catalog],
    ids=["claude-spelling", "agents-spelling"],
)
def test_a_root_source_fails_in_either_catalog(repo: Path, shape, source: str) -> None:
    problems = location_problems(shape(source), repo)
    assert any("installs from the repository root" in p for p in problems), problems


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("../elsewhere", "escapes the repository root"),
        ("./packaging/../../elsewhere", "escapes the repository root"),
        ("./packaging/..", "escapes the repository root"),
        ("/opt/plugin", "non-relative source"),
        ("~/plugin", "non-relative source"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [_claude_catalog, _agents_catalog],
    ids=["claude-spelling", "agents-spelling"],
)
def test_a_source_outside_the_root_fails_in_either_catalog(
    repo: Path, shape, source: str, expected: str
) -> None:
    problems = location_problems(shape(source), repo)
    assert any(expected in p for p in problems), problems


def test_an_entry_with_no_local_source_path_fails(repo: Path) -> None:
    catalog = _claude_catalog({"source": "github", "repo": REPOSITORY})
    assert location_problems(catalog, repo) == ["entry 'multi-llm' has no source path"]


def test_the_codex_entry_must_name_the_codex_tree(repo: Path) -> None:
    """Copying the Claude path into `.agents` would install the package with no tool."""
    problems = entry_problems(_agents_catalog(CLAUDE_SOURCE), repo, CODEX)
    assert any("must be './packaging/codex/multi-llm'" in p for p in problems), problems


@pytest.mark.parametrize(
    ("tree", "foreign", "expected"),
    [
        (CLAUDE.root, CODEX.version_manifest, "the Claude tree carries .codex-plugin/plugin.json"),
        (CLAUDE.root, ".mcp.json", "the Claude tree carries .mcp.json"),
        (CODEX.root, CLAUDE.version_manifest, "the Codex tree carries .claude-plugin/plugin.json"),
    ],
    ids=["codex-manifest-in-claude", "mcp-in-claude", "claude-manifest-in-codex"],
)
def test_a_foreign_manifest_in_a_generated_tree_fails(
    repo: Path, tree: str, foreign: str, expected: str
) -> None:
    _write_json(repo / tree / foreign, {"name": PLUGIN_NAME})
    assert manifest_boundary_problems(repo) == [expected]


@pytest.mark.parametrize(
    "relative",
    [
        ".claude-plugin/marketplace.json",
        ".agents/plugins/marketplace.json",
        ".agents/plugins/api_marketplace.json",
        "skills/multi-llm/marketplace.json",
    ],
)
def test_a_catalog_inside_a_generated_tree_is_found(repo: Path, relative: str) -> None:
    _write_json(repo / CLAUDE.root / relative, {"name": "beastlabai", "plugins": []})
    assert catalogs_inside(repo / CLAUDE.root) == [relative]


def test_a_copy_of_the_claude_catalog_in_agents_is_a_stale_copy(repo: Path) -> None:
    problems = stale_copy_problems(_claude_catalog(), _claude_catalog(), repo)
    assert any("points at the Claude source" in p for p in problems), problems


def test_the_old_description_is_a_stale_copy(repo: Path) -> None:
    catalog = _agents_catalog(description="whose one entry has source './': with only that file")
    problems = stale_copy_problems(catalog, _claude_catalog(), repo)
    assert any("still calls the Claude source" in p for p in problems), problems


def test_a_description_left_behind_by_a_source_move_is_stale(repo: Path) -> None:
    moved = _claude_catalog("./packaging/claude-next/multi-llm")
    problems = stale_copy_problems(_agents_catalog(), moved, repo)
    assert any("does not name the Claude source" in p for p in problems), problems


# ---------------------------------------------------------------------------
# The documentation helpers, on strings
# ---------------------------------------------------------------------------

def _doc(*lines: str, fence: str = "```bash") -> str:
    closing = fence.strip()[:3]
    return "\n".join(["Intro.", "", fence, *lines, closing, "", "Outro."]) + "\n"


GOOD_COMMAND = (
    "claude plugin marketplace add beastlabai/multi-llm-plugin "
    "--sparse .claude-plugin packaging/claude/multi-llm"
)


@pytest.mark.parametrize(
    "text",
    [
        _doc(GOOD_COMMAND),
        _doc(
            "claude plugin marketplace add beastlabai/multi-llm-plugin \\",
            "  --sparse .claude-plugin packaging/claude/multi-llm",
            "claude plugin install multi-llm@beastlabai",
        ),
        _doc(
            "claude plugin marketplace add beastlabai/multi-llm-plugin \\",
            "  --sparse .claude-plugin \\",
            "  packaging/claude/multi-llm",
        ),
        _doc(f"{GOOD_COMMAND} --scope project   # or --scope local"),
        _doc(f"cd /path/to/project && {GOOD_COMMAND} --scope local"),
        _doc(f"! {GOOD_COMMAND}"),
        _doc(
            "claude plugin marketplace add beastlabai/multi-llm-plugin "
            "--sparse .claude-plugin --sparse packaging/claude/multi-llm"
        ),
        _doc(
            "claude plugin marketplace add beastlabai/multi-llm-plugin "
            "--sparse=.claude-plugin --sparse=packaging/claude/multi-llm"
        ),
        _doc(
            "claude plugin marketplace add beastlabai/multi-llm-plugin "
            "--sparse ./.claude-plugin/ ./packaging/claude/multi-llm/"
        ),
        _doc(
            "claude plugin marketplace add beastlabai/multi-llm-plugin "
            "--sparse .claude-plugin packaging/claude"
        ),
        "1. Register:\n\n   ```bash\n   " + GOOD_COMMAND + "\n   ```\n",
        "> Register:\n>\n> ```bash\n> " + GOOD_COMMAND + "\n> ```\n",
        _doc(GOOD_COMMAND, fence="~~~bash"),
    ],
    ids=[
        "one-line", "continuation", "two-continuations", "trailing-scope-and-comment",
        "after-cd", "in-session-bang", "repeated-flag", "equals-form", "dot-slash-spelling",
        "ancestor-path", "indented-fence", "blockquote-fence", "tilde-fence",
    ],
)
def test_a_registration_that_fetches_both_paths_passes(text: str) -> None:
    assert registration_problems(text, "doc.md", CLAUDE_SOURCE) == []


def test_the_parser_reads_the_source_paths_and_first_line() -> None:
    text = _doc(
        "# a comment",
        "claude plugin marketplace add beastlabai/multi-llm-plugin \\",
        "  --sparse .claude-plugin packaging/claude/multi-llm --scope project",
    )
    assert registrations(text) == [
        Registration(5, REPOSITORY, (".claude-plugin", "packaging/claude/multi-llm"))
    ]


def test_commands_outside_a_fence_or_for_another_tool_are_not_registrations() -> None:
    text = "\n".join([
        f"Prose mentions `{GOOD_COMMAND}` inline.",
        "```text",
        "/plugin marketplace add beastlabai/multi-llm-plugin",
        "```",
        "```bash",
        f"# {GOOD_COMMAND}",
        "codex plugin marketplace add beastlabai/multi-llm-plugin --sparse .agents",
        "```",
    ])
    assert registrations(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_doc("echo nothing here"), "no `claude plugin marketplace add` command"),
        ("", "no `claude plugin marketplace add` command"),
        (
            _doc("/plugin marketplace add beastlabai/multi-llm-plugin"),
            "no `claude plugin marketplace add` command",
        ),
        (
            _doc("claude plugin marketplace add beastlabai/multi-llm-plugin"),
            "no --sparse paths",
        ),
        (
            _doc(
                "claude plugin marketplace add beastlabai/multi-llm-plugin "
                "--sparse packaging/claude/multi-llm"
            ),
            "does not cover '.claude-plugin'",
        ),
        (
            _doc(
                "claude plugin marketplace add beastlabai/multi-llm-plugin "
                "--sparse .claude-plugin"
            ),
            "does not cover 'packaging/claude/multi-llm'",
        ),
        (
            _doc(
                "claude plugin marketplace add beastlabai/multi-llm-plugin "
                "--sparse .claude-plugin packaging/claudeX"
            ),
            "does not cover 'packaging/claude/multi-llm'",
        ),
        (
            _doc(
                "claude plugin marketplace add beastlabai/multi-llm-plugin "
                "--sparse .claude-plugin packaging/claude/multi-llm/skills"
            ),
            "does not cover 'packaging/claude/multi-llm'",
        ),
        (
            _doc(
                "claude plugin marketplace add beastlabai/multi-llm-plugin "
                "--sparse .claude-plugin/marketplace.json packaging/claude/multi-llm"
            ),
            "does not cover '.claude-plugin'",
        ),
        (
            _doc(
                "claude plugin marketplace add --sparse .claude-plugin "
                "packaging/claude/multi-llm beastlabai/multi-llm-plugin"
            ),
            "--sparse must come after the source",
        ),
        (_doc(GOOD_COMMAND.replace("beastlabai/multi-llm-plugin", "other/repo")), "not 'beastlabai"),
        # One good command does not excuse a stale one in the same document.
        (
            _doc(
                GOOD_COMMAND,
                "claude plugin marketplace add beastlabai/multi-llm-plugin --sparse .claude-plugin",
            ),
            "does not cover 'packaging/claude/multi-llm'",
        ),
    ],
    ids=[
        "no-command", "empty-document", "interactive-only", "no-sparse", "catalog-dir-omitted",
        "source-omitted", "sibling-is-not-a-prefix", "deeper-than-the-source",
        "file-instead-of-catalog-dir", "sparse-before-source", "another-repository",
        "second-command-stale",
    ],
)
def test_a_registration_that_misses_something_fails(text: str, expected: str) -> None:
    problems = registration_problems(text, "doc.md", CLAUDE_SOURCE)
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("root_path", [".", "./", "''", "/", "./.", "../up"],
                         ids=["dot", "dot-slash", "empty", "slash", "dot-slash-dot", "parent"])
def test_a_sparse_path_at_the_repository_root_is_rejected(root_path: str) -> None:
    """A root path covers everything, so it also fetches everything the source excludes."""
    text = _doc(
        "claude plugin marketplace add beastlabai/multi-llm-plugin "
        f"--sparse .claude-plugin packaging/claude/multi-llm {root_path}"
    )
    problems = registration_problems(text, "doc.md", CLAUDE_SOURCE)
    assert any("selects the repository root or leaves it" in p for p in problems), problems


def test_a_root_sparse_path_alone_does_not_count_as_coverage() -> None:
    text = _doc("claude plugin marketplace add beastlabai/multi-llm-plugin --sparse .")
    problems = registration_problems(text, "doc.md", CLAUDE_SOURCE)
    assert any("does not cover '.claude-plugin'" in p for p in problems), problems
    assert any("does not cover 'packaging/claude/multi-llm'" in p for p in problems), problems


@pytest.mark.parametrize("source", ["./", ".", ""], ids=["dot-slash", "dot", "empty"])
def test_a_catalog_source_reverted_to_the_root_fails_the_documents(source: str) -> None:
    problems = registration_problems(_doc(GOOD_COMMAND), "doc.md", source)
    assert any("is the repository root" in p for p in problems), problems


def test_a_moved_source_with_stale_documented_paths_fails() -> None:
    problems = registration_problems(_doc(GOOD_COMMAND), "doc.md", "./packaging/claude/renamed")
    assert problems == [
        "doc.md:4: --sparse ['.claude-plugin', 'packaging/claude/multi-llm'] does not "
        "cover 'packaging/claude/renamed'"
    ]


@pytest.mark.parametrize(
    ("sparse_path", "required", "expected"),
    [
        ("packaging/claude/multi-llm", "packaging/claude/multi-llm", True),
        ("packaging/claude/multi-llm/", "packaging/claude/multi-llm", True),
        ("./packaging/claude", "packaging/claude/multi-llm", True),
        ("packaging", "packaging/claude/multi-llm", True),
        ("packaging/claudeX", "packaging/claude/multi-llm", False),
        ("packaging/clau", "packaging/claude/multi-llm", False),
        ("packaging/claude/multi-llm-old", "packaging/claude/multi-llm", False),
        ("packaging/claude/multi-llm/skills", "packaging/claude/multi-llm", False),
        (".claude-plugin", ".claude-plugin", True),
        (".claude", ".claude-plugin", False),
        (".", "packaging/claude/multi-llm", False),
        ("", "packaging/claude/multi-llm", False),
        ("/", "packaging/claude/multi-llm", False),
        ("../packaging/claude/multi-llm", "packaging/claude/multi-llm", False),
    ],
)
def test_coverage_is_by_path_component(sparse_path: str, required: str, expected: bool) -> None:
    assert sparse_covers(sparse_path, required) is expected
