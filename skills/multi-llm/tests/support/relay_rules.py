"""The next-step relay rules as each host surface states them.

``RELAY_SURFACES`` lists the files that carry the rule, ``_flat`` joins a text's
lines for a phrase search, and ``MARKDOWN_RULE`` is the rule's wording. No
fixtures.
"""

from __future__ import annotations

from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent.parent


def _flat(path: str) -> str:
    return " ".join((SKILL_ROOT / path).read_text(encoding="utf-8").split())


#: Every route's relay rule for `user_next_step`: the tool description both
#: package hosts carry, their two plugin instructions, the two file shims and
#: Claude Code's SKILL.md.
RELAY_SURFACES = (
    "hosts/codex/plugin-skill.md.tmpl",
    "hosts/opencode/plugin-command.md.tmpl",
    "hosts/codex/shim.md.tmpl",
    "hosts/opencode/shim.md.tmpl",
    "SKILL.md",
)


#: The rule of every route that copies `user_next_step.markdown`, the
#: section already written (fix round 22: Codex 04 left both commands
#: unfenced, and the hosts folded the manual close command into prose).
#: Fix round 24: Claude Code copies it too (the lookup prints it between
#: `[NEXT_STEP_MARKDOWN]` lines; racing 06 put the manual tasks first), in
#: its words and order (OpenCode 03 reordered it under captions of its own).
MARKDOWN_RULE = (
    "copied as it is (its words and order, each command alone in its fenced "
    "block, each why and note plain text, the manual tasks it lists included, "
    "no caption of your own)")
