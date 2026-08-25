#!/usr/bin/env python3
"""Every link in ``README.md`` and ``docs/*.md`` resolves -- file and anchor.

The README was trimmed to a front page and its prose moved into five ``docs/``
pages, which turned a single document's internal structure into a web of
cross-file links and heading anchors. That web was checked **once**, from a
shell, off a script that was never committed. A guard that ran once is a guard
that has already stopped running: the next heading rename or ``docs/`` move
404s the front page with a green suite. This module is that script, committed,
so the check follows the prose it checks.

Four properties, each with its own way of rotting quietly:

1. **Every relative link target exists.** ``](docs/hosts.md)``,
   ``](../README.md)``, ``](providers.md)``, ``](LICENSE)`` and the directory
   pointers are resolved on disk. Absence is a failure, never a skip -- a link
   into a page this tree does not have is exactly the breakage the split can
   cause.
2. **Every in-page anchor matches a heading.** ``](#codex-and-opencode)`` is
   checked against the ids GitHub generates from that file's own ``#`` lines. A
   retitled heading strands its links silently; nothing renders a 404 for a bad
   fragment, the browser just sits where it was.
3. **Every cross-file anchor matches a heading in the file it names.**
   ``](hosts.md#per-host-quickstart)`` has two ways to break, and the file half
   being fine says nothing about the fragment half.
4. **The slug algorithm itself is asserted.** A walker whose slugger is wrong
   passes vacuously or fails noisily on correct links, so ``_slug`` is pinned
   against GitHub's rules (lowercase, ASCII punctuation dropped except ``-`` and
   ``_``, spaces to hyphens, repeats suffixed) rather than trusted.

Beside them, follow-up 18: the README's ``## Documentation`` table is the front
page's only map of the split and is hand-written, so a seventh ``docs/`` page --
or a rename -- leaves it silently incomplete. It is tied to the directory it
indexes here rather than maintained by hand.

Deliberate scoping decisions:

* **Fenced code blocks are not markdown.** Headings are collected outside
  fences only (``README.md``'s install recipes are full of ``# comment`` lines
  that are not headings), and links are collected outside fences too, so a
  ``](…)`` printed inside a sample command is sample text, not a promise.
* **Image links are checked like any other link.** ``![alt](target)`` shares the
  ``](…)`` shape; a local image that has moved is as broken as a local page that
  has moved. The one image in the tree today is an external URL and is skipped
  with the rest of them.
* **Bare ``<a id=…>`` anchors are not honoured.** There are none, and honouring
  hand-written HTML ids would let a heading rename be "fixed" by an invisible
  anchor rather than by the link. If one is ever added, this module fails and
  the decision gets made explicitly.
* **External schemes are skipped** (``http://``, ``https://``, ``mailto:``):
  reaching them needs the network, and this suite is pure file reads.
* **Fragments are compared exactly**, not re-slugged. GitHub's heading ids are
  lowercase and the browser matches them case-sensitively, so ``#Prerequisites``
  really is a dead link even though ``#prerequisites`` resolves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Repo root: tests/ -> multi-llm/ -> skills/ -> <repo>.
REPO_ROOT = Path(__file__).resolve().parents[3]
README = REPO_ROOT / "README.md"
DOCS_DIR = REPO_ROOT / "docs"

#: Link schemes this module cannot follow without a network.
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:")

#: The ``## Documentation`` table's one row that is not a ``docs/`` page, named
#: rather than pattern-matched so a second such row is a decision, not a drift.
#: The table's omissions are choices too, recorded here so they read as such:
#: ``TODO.md`` is reached from ``## Roadmap`` instead, and
#: ``skills/multi-llm/SKILL.md`` / ``skills/multi-llm/CLAUDE.md`` are
#: skill-internal entry points, linked from ``## How it works`` and
#: ``## Your project repository`` where they mean something.
TABLE_NON_DOCS_ROWS = ("skills/multi-llm/AGENTS.md",)

#: A markdown inline link or image target: the ``](target)`` half, with an
#: optional ``<…>`` wrapper and an optional title after the target.
LINK_RE = re.compile(r"\]\(\s*<?([^)<>\s]+)>?[^)]*\)")

#: An ATX heading outside a fence. Setext headings are not used in this tree.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

#: A fence opener/closer, indented or not.
FENCE_RE = re.compile(r"^\s*(```|~~~)")

#: What GitHub's slugger drops: ASCII punctuation except ``-`` and ``_``, plus
#: control characters and the general-punctuation blocks (em dash, bullets).
SLUG_STRIP_RE = re.compile(
    "["
    r"\x00-\x1f"         # control characters
    r"!-,./:-@\[-^`{-~"   # ASCII punctuation, minus '-' (0x2d) and '_' (0x5f)
    r"\u2000-\u206f"     # general punctuation: em dash, bullets, primes
    r"\u2e00-\u2e7f"     # supplemental punctuation
    "]"
)

#: ``[text](url)`` inside a heading: GitHub slugs the rendered text, not the
#: link syntax, so the target is dropped before the punctuation strip.
HEADING_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _slug(text: str) -> str:
    """The id GitHub gives a heading with this text.

    Mirrors ``github-slugger``: render-level markup removed, lowercased, the
    punctuation set above dropped, spaces to hyphens. Repeat handling belongs to
    the caller, which is the only place that knows what came before.
    """
    text = HEADING_LINK_RE.sub(r"\1", text)
    return SLUG_STRIP_RE.sub("", text.strip().lower()).replace(" ", "-")


def _outside_fences(text: str):
    """Yield ``(line_number, line)`` for every line not inside a code fence."""
    in_fence = False
    for number, line in enumerate(text.splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield number, line


def _slugs(path: Path) -> set[str]:
    """Every heading id in ``path``, repeats suffixed the way GitHub suffixes."""
    seen: dict[str, int] = {}
    ids: set[str] = set()
    for _, line in _outside_fences(path.read_text(encoding="utf-8")):
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = _slug(match.group(2))
        count = seen.get(base, 0)
        seen[base] = count + 1
        ids.add(base if count == 0 else f"{base}-{count}")
    return ids


def _links(path: Path) -> list[tuple[int, str]]:
    """Every non-external link target in ``path``, as ``(line_number, target)``."""
    found = []
    for number, line in _outside_fences(path.read_text(encoding="utf-8")):
        for target in LINK_RE.findall(line):
            if not target.startswith(EXTERNAL_SCHEMES):
                found.append((number, target))
    return found


def _documented_files() -> list[Path]:
    """The README plus every ``docs/`` page, in a stable order."""
    return [README] + sorted(DOCS_DIR.glob("*.md"))


def _where(path: Path, number: int) -> str:
    """``README.md:15`` -- the file/line half of every failure message here."""
    return f"{path.relative_to(REPO_ROOT).as_posix()}:{number}"


def _documentation_table_rows() -> list[tuple[int, str, str]]:
    """The ``## Documentation`` table, as ``(line_number, target, text)`` rows.

    Parsed rather than hard-coded, because a hard-coded copy of the table is the
    hand-maintenance this guard exists to remove.
    """
    rows: list[tuple[int, str, str]] = []
    in_section = False
    in_body = False
    for number, line in _outside_fences(README.read_text(encoding="utf-8")):
        heading = HEADING_RE.match(line)
        if heading:
            in_section = heading.group(2).strip() == "Documentation"
            in_body = False
            continue
        if not in_section or not line.lstrip().startswith("|"):
            continue
        cell = line.strip().strip("|").split("|")[0].strip()
        # The `| --- |` separator opens the body; everything above it is the
        # header row, which names columns rather than documents.
        if cell and set(cell) <= set("- :"):
            in_body = True
            continue
        if not in_body or not cell:
            continue
        match = re.search(r"\[([^\]]*)\]\(\s*<?([^)<>\s]+)>?[^)]*\)", cell)
        if match:
            rows.append((number, match.group(2), match.group(1)))
        else:
            rows.append((number, "", cell))
    return rows


# ---------------------------------------------------------------------------
# 1. The link-and-anchor walker (follow-up 8)
# ---------------------------------------------------------------------------

class TestEveryDocumentedLinkResolves:
    def test_the_slug_algorithm_matches_the_ids_github_generates(self):
        """The walker is only as good as its slugger, so pin the slugger.

        Every case here is one this tree either has or is one heading away from
        having: an ``&``/``/`` between words leaves *two* hyphens because the
        punctuation goes before the spaces do, backticks vanish entirely, and a
        hyphen or underscore already in the text survives.
        """
        cases = {
            "Codex and OpenCode": "codex-and-opencode",
            "Updating the Claude Code plugin": "updating-the-claude-code-plugin",
            "Letting Claude decide ambiguous findings":
                "letting-claude-decide-ambiguous-findings",
            "Pointing a provider CLI at a different endpoint "
            "(GLM / Z.ai, and the general case)":
                "pointing-a-provider-cli-at-a-different-endpoint-glm--zai-"
                "and-the-general-case",
            "Tests & fixtures": "tests--fixtures",
            "The `providers.yaml` file": "the-providersyaml-file",
            "multi-llm - Harness the harnesses": "multi-llm---harness-the-harnesses",
            "snake_case names kept": "snake_case-names-kept",
            "  Padded  ": "padded",
            "[A link](target.md) in a heading": "a-link-in-a-heading",
        }
        for text, expected in cases.items():
            assert _slug(text) == expected, text

    def test_a_repeated_heading_gets_the_suffixed_id_github_gives_it(self, tmp_path):
        page = tmp_path / "repeats.md"
        page.write_text(
            "# Notes\n\n## Setup\n\n## Setup\n\n## Setup\n", encoding="utf-8"
        )
        assert _slugs(page) == {"notes", "setup", "setup-1", "setup-2"}

    def test_a_hash_inside_a_code_fence_is_not_a_heading(self, tmp_path):
        page = tmp_path / "fenced.md"
        page.write_text(
            "# Real\n\n```bash\n# not a heading\n```\n\n## Also real\n",
            encoding="utf-8",
        )
        assert _slugs(page) == {"real", "also-real"}

    def test_every_relative_link_target_exists_on_disk(self):
        """A link into a page this tree does not have is a failure, not a skip."""
        missing = []
        for path in _documented_files():
            for number, target in _links(path):
                relative, _, _fragment = target.partition("#")
                if not relative:
                    continue
                if not (path.parent / relative).exists():
                    missing.append(f"{_where(path, number)}: {target} — no such file")
        assert not missing, (
            "relative link targets that do not resolve on disk:\n  "
            + "\n  ".join(missing)
        )

    def test_every_in_page_anchor_matches_a_heading_in_its_own_file(self):
        """``](#slug)`` is checked against the file's own heading ids."""
        broken = []
        for path in _documented_files():
            ids = _slugs(path)
            for number, target in _links(path):
                relative, hashed, fragment = target.partition("#")
                if relative or not hashed:
                    continue
                if fragment not in ids:
                    broken.append(
                        f"{_where(path, number)}: {target} — no heading with that "
                        f"id in {path.name}"
                    )
        assert not broken, (
            "in-page anchors with no matching heading (a heading was renamed "
            "without its links):\n  " + "\n  ".join(broken)
        )

    def test_every_cross_file_anchor_matches_a_heading_in_the_file_it_names(self):
        """``](other.md#slug)`` breaks in two places; both are checked here."""
        broken = []
        for path in _documented_files():
            for number, target in _links(path):
                relative, hashed, fragment = target.partition("#")
                if not relative or not hashed:
                    continue
                destination = (path.parent / relative).resolve()
                if not destination.is_file():
                    broken.append(
                        f"{_where(path, number)}: {target} — no such file"
                    )
                elif fragment not in _slugs(destination):
                    broken.append(
                        f"{_where(path, number)}: {target} — no heading with that "
                        f"id in {relative}"
                    )
        assert not broken, (
            "cross-file anchors that do not resolve:\n  " + "\n  ".join(broken)
        )

    def test_the_walker_has_the_split_documents_and_their_anchors_to_walk(self):
        """Non-vacuity: three green assertions over an empty list prove nothing.

        The split is what created these links, so the walk is asserted to cover
        the README, every ``docs/`` page, and a plural number of anchors -- a
        ``docs/`` directory that vanished, or a link regex that stopped matching,
        fails here rather than passing quietly above.
        """
        files = _documented_files()
        assert README.is_file(), README
        assert DOCS_DIR.is_dir(), DOCS_DIR
        assert len(files) >= 6, [path.name for path in files]

        targets = [target for path in files for _, target in _links(path)]
        anchors = [target for target in targets if "#" in target]
        assert len(targets) >= 20, targets
        assert len(anchors) >= 4, anchors


# ---------------------------------------------------------------------------
# 2. The README's ``## Documentation`` table indexes ``docs/`` (follow-up 18)
# ---------------------------------------------------------------------------

class TestTheDocumentationTableIndexesTheDocsDirectory:
    def test_every_docs_page_has_exactly_one_row_in_the_documentation_table(self):
        """The table is the front page's only map of the split.

        Hand-written, so a seventh ``docs/`` page -- or a rename -- leaves it
        incomplete with nothing to notice, the same hand-maintenance failure D4
        refuses to accept for the host list.
        """
        rows = _documentation_table_rows()
        listed = [target for _, target, _ in rows if target.startswith("docs/")]
        on_disk = sorted(
            path.relative_to(REPO_ROOT).as_posix() for path in DOCS_DIR.glob("*.md")
        )

        duplicates = sorted({t for t in listed if listed.count(t) > 1})
        assert not duplicates, f"listed more than once in the table: {duplicates}"

        missing = sorted(set(on_disk) - set(listed))
        assert not missing, (
            "docs/ pages with no row in the README's ## Documentation table: "
            f"{missing} — the table is the front page's only map of the split, "
            "so add a row rather than leaving the page unreachable"
        )

        stale = sorted(set(listed) - set(on_disk))
        assert not stale, (
            f"## Documentation rows naming docs/ pages that do not exist: {stale}"
        )

    def test_every_documentation_table_row_points_at_a_file_that_exists(self):
        rows = _documentation_table_rows()
        assert len(rows) >= len(list(DOCS_DIR.glob("*.md"))), rows
        for number, target, text in rows:
            assert target, (
                f"{_where(README, number)}: table row {text!r} has no link — every "
                "row in ## Documentation is a pointer into the docs it indexes"
            )
            assert (REPO_ROOT / target).exists(), (
                f"{_where(README, number)}: {target} — no such file"
            )

    def test_the_one_row_outside_docs_is_the_allowlisted_one(self):
        """``skills/multi-llm/AGENTS.md`` is in the table on purpose.

        Allowlisted by name so a second non-``docs/`` row has to be a decision
        recorded here, not a row that quietly widens what the table indexes.
        """
        outside = sorted(
            target
            for _, target, _ in _documentation_table_rows()
            if not target.startswith("docs/")
        )
        assert outside == sorted(TABLE_NON_DOCS_ROWS), (
            f"## Documentation rows outside docs/: {outside} — expected exactly "
            f"{sorted(TABLE_NON_DOCS_ROWS)}. A new one is a decision about what "
            "the table indexes: record it in TABLE_NON_DOCS_ROWS with the reason."
        )


if __name__ == "__main__":  # pragma: no cover - parity with the sibling suites
    raise SystemExit(pytest.main([__file__, "-q"]))
