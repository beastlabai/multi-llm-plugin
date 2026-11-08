"""The checked-in generated plugin trees, named once for every repository-wide guard.

Several guards walk the whole repository and meet these trees on the way: the one
that counts ``SKILL.md`` manifests, the uv-candidate discovery scan, the placeholder
manifest, the ``.gitattributes`` rule set and the version chain. Each needs the same
two answers: which trees exist, and which of their files are the authored file
again rather than something new. A guard that keeps its own copy of those answers
goes wrong in one of two ways when a tree is added. It fails on every copy, or
someone widens it until it stops failing on anything. So the answers live here and
each guard derives what it needs.

**A copy may be graded as its source only because something proves the bytes
match.** The build-parity rows in ``test_codex_package.py`` and
``test_claude_package.py`` compare each whole tree to a fresh build. A file the
renderer writes differs from any source by design, so it is listed under
``rendered`` and is never treated as a copy: a guard has to account for it under
its own path.

**The list is deliberately not a glob.** ``packaging/*/multi-llm/`` would admit a
third tree nobody decided to add, and that third tree would then be excused from
every guard that derives from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

#: Where the authored skill lives in the repository, and also where each plugin
#: carries its copy of it, relative to the plugin root.
PAYLOAD = "skills/multi-llm/"

#: The one ``SKILL.md`` a developer edits.
AUTHORED_MANIFEST = f"{PAYLOAD}SKILL.md"


@dataclass(frozen=True)
class Distribution:
    """One generated tree. Every path in a mapping key is relative to :attr:`root`."""

    #: Repository-relative, with the trailing slash, so a prefix test cannot match
    #: a sibling such as ``packaging/claude/multi-llm-old/``.
    root: str
    #: The generated file that carries this plugin's version.
    version_manifest: str
    #: The command that rewrites this tree, which a stale-tree failure must name.
    regen: str
    #: Byte copies from outside the payload: ``plugin-relative -> repo-relative``.
    copied: Mapping[str, str]
    #: Files this host's builder writes itself, which therefore differ from any
    #: authored file: ``plugin-relative -> what they are rendered from``.
    rendered: Mapping[str, str]

    @property
    def manifest(self) -> str:
        """This tree's ``SKILL.md``, repository-relative."""
        return f"{self.root}{PAYLOAD}SKILL.md"


CODEX = Distribution(
    root="packaging/codex/multi-llm/",
    version_manifest=".codex-plugin/plugin.json",
    regen="make codex-regen",
    copied={
        ".mcp.json": "packaging/codex/mcp.json",
        "LICENSE": "LICENSE",
        "assets/icon.svg": "packaging/codex/assets/icon.svg",
        "mcp/server.py": "packaging/codex/mcp_server.py",
    },
    rendered={
        ".codex-plugin/plugin.json": "packaging/codex/plugin.json.tmpl",
        "dist/cli-surface.json": "packaging/common/surface.py",
        "dist/tree.sha256": "packaging/codex/build.py",
        f"{PAYLOAD}SKILL.md": f"{PAYLOAD}hosts/codex/plugin-skill.md.tmpl",
    },
)

CLAUDE = Distribution(
    root="packaging/claude/multi-llm/",
    version_manifest=".claude-plugin/plugin.json",
    regen="make claude-regen",
    copied={
        ".claude-plugin/plugin.json": ".claude-plugin/plugin.json",
        "LICENSE": "LICENSE",
        "hooks/hooks.json": "hooks/hooks.json",
        "hooks/doctor.sh": "hooks/doctor.sh",
        "hooks/report-check.sh": "hooks/report-check.sh",
        "hooks/wakeup-check.sh": "hooks/wakeup-check.sh",
    },
    rendered={},
)

#: Every checked-in generated tree. Order is the ``.gitattributes`` rule order.
DISTRIBUTIONS: Tuple[Distribution, ...] = (CODEX, CLAUDE)


def authored_source(path: str) -> Optional[str]:
    """The authored file that repository-relative *path* is a byte copy of.

    ``None`` means it is not a copy of anything: outside every generated tree,
    rendered by a host's builder, or simply unknown. A caller must then account for
    the file under its own path, which is what keeps an unexpected file under a
    generated root a failure rather than a silent pass.
    """
    for distribution in DISTRIBUTIONS:
        if not path.startswith(distribution.root):
            continue
        relative = path[len(distribution.root):]
        if relative in distribution.rendered:
            return None
        if relative in distribution.copied:
            return distribution.copied[relative]
        if not relative.startswith(PAYLOAD):
            return None
        return relative
    return None
