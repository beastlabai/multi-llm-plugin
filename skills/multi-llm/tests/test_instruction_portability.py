#!/usr/bin/env python3
"""
Windows/macOS portability regression guards for the instruction layer.

Phase 1 of the portability plan swept every shell-argument occurrence of
``${CLAUDE_SKILL_DIR}`` in the instruction/reference markdown into the
double-quoted form (Windows install paths routinely contain spaces, e.g.
``C:\\Users\\John Smith\\...``, so an unquoted expansion word-splits and the
command dies before Python starts). These tests encode the plan's two machine
greps plus the allowed-tools glob-syntax audit so a future edit cannot silently
reintroduce the bugs.

Known blind spot (accepted by the plan): the unquoted-path-argument pattern
stops scanning a line at its first double quote, so a mixed line with one
quoted and one later-unquoted occurrence can pass. That residual case is a
manual-review concern; these tests block the shapes that actually occurred.
"""

import re
from pathlib import Path

SKILL_ROOT = Path(__file__).parent.parent


def _scanned_files():
    files = sorted(SKILL_ROOT.glob("instructions/*.md"))
    files += sorted(SKILL_ROOT.glob("references/*.md"))
    files.append(SKILL_ROOT / "SKILL.md")
    return files


def _matches(pattern):
    """Return (file, line_no, line) for every regex match across the docs."""
    hits = []
    for path in _scanned_files():
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(pattern, line):
                hits.append((path.name, i, line.strip()))
    return hits


class TestSkillDirQuoting:
    def test_no_unquoted_project_flag(self):
        """``--project ${CLAUDE_SKILL_DIR}`` must always be quoted:
        ``--project "${CLAUDE_SKILL_DIR}"``."""
        hits = _matches(r"--project \$\{CLAUDE_SKILL_DIR\}")
        assert not hits, (
            "Unquoted `--project ${CLAUDE_SKILL_DIR}` (must be "
            '`--project "${CLAUDE_SKILL_DIR}"`) found at:\n'
            + "\n".join(f"  {f}:{ln}: {txt}" for f, ln, txt in hits)
        )

    def test_no_unquoted_skill_dir_command_argument(self):
        """No ``uv run``/``python`` command line may pass an unquoted
        ``${CLAUDE_SKILL_DIR}`` argument (e.g. ``python
        ${CLAUDE_SKILL_DIR}/script.py``)."""
        hits = _matches(r'(uv run|python)[^"]*[ =]\$\{CLAUDE_SKILL_DIR\}')
        assert not hits, (
            "Unquoted ${CLAUDE_SKILL_DIR} in shell-argument position "
            '(must be wrapped in double quotes, e.g. '
            '`"${CLAUDE_SKILL_DIR}/script.py"`) found at:\n'
            + "\n".join(f"  {f}:{ln}: {txt}" for f, ln, txt in hits)
        )


class TestAllowedToolsGlobSyntax:
    def test_no_regex_syntax_in_bash_rules(self):
        """Claude Code `allowed-tools` Bash patterns are glob-style prefix
        rules, not regexes: a rule like ``Bash(grep:.*)`` is inert (it matches
        the literal prefix ``grep:.``) and silently reintroduces permission
        prompts. Only ``:*`` glob prefixes or exact strings are valid."""
        skill_md = (SKILL_ROOT / "SKILL.md").read_text()
        offending = [
            line.strip()
            for line in skill_md.splitlines()
            if "Bash(" in line and re.search(r"\.\*|\.\+", line)
        ]
        assert not offending, (
            "Regex syntax inside Bash(...) allowed-tools rules (glob-style "
            "`:*` prefix or exact string only):\n"
            + "\n".join(f"  {line}" for line in offending)
        )
