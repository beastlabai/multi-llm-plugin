"""Comments name code that exists: the two facades' names resolve.

``utils/write_batch_scheduler.py`` and ``utils/subagent_executor.py`` are facades
over modules their code moved into, so a comment or docstring elsewhere that
spells ``write_batch_scheduler.<name>`` can outlive its target and point a reader
at nothing. Every such spelling in a non-test module must resolve, attribute by
attribute, on the imported facade (a method as ``<module>.<Class>.<method>``);
a name that moved is spelled by its owner instead
(skills/multi-llm/AGENTS.md#comments-and-docstrings).
"""

from __future__ import annotations

import importlib
import io
import re
import sys
import tokenize
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

FACADES = ("write_batch_scheduler", "subagent_executor")

SPELLED = re.compile(
    r"\b(" + "|".join(FACADES) + r")\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")

#: A one-letter capital is a placeholder in prose (``write_batch_scheduler.X``),
#: and ``py`` is the file name, never an attribute.
PLACEHOLDER = re.compile(r"^(?:[A-Z]|py)$")

_SKIPPED_PARTS = {"tests", "__pycache__", ".venv"}


def _spellings():
    for path in sorted(SKILL_ROOT.rglob("*.py")):
        if _SKIPPED_PARTS & set(path.relative_to(SKILL_ROOT).parts):
            continue
        source = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            for match in SPELLED.finditer(token.string):
                line = token.start[0] + token.string.count("\n", 0, match.start())
                yield path.relative_to(SKILL_ROOT).as_posix(), line, match


def _resolves(facade: str, dotted: str) -> bool:
    target = importlib.import_module(f"utils.{facade}")
    for part in dotted.split("."):
        if PLACEHOLDER.match(part):
            return True
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


def test_every_facade_name_a_comment_spells_resolves():
    unresolved = [f"{where}:{line}: {match.group(0)}"
                  for where, line, match in _spellings()
                  if not _resolves(match.group(1), match.group(2))]
    assert unresolved == [], (
        "comments name facade attributes that do not exist; spell the owner "
        "(`write_reports.WriteReports.mirror_status`, "
        "`write_batch_scheduler.WriteBatchScheduler._finalize`):\n  "
        + "\n  ".join(unresolved))


@pytest.mark.parametrize("dotted, expected", [
    ("WriteBatchScheduler._finalize", True),
    ("_mirror_status", False),
    ("X", True),
])
def test_the_resolver(dotted, expected):
    assert _resolves("write_batch_scheduler", dotted) is expected
