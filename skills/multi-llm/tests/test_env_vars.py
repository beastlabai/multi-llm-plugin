"""``utils/env_vars.py`` is the one inventory of ``MULTI_LLM_*`` names.

The package test (``test_codex_package.py``) holds the table to the shipped tree
and to the Codex ``.mcp.json``; this file holds the table to itself, to the repo
outside the shipped tree, and to ``docs/configuration.md``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import env_vars  # noqa: E402

_SKIPPED_PARTS = {".venv", "__pycache__", "node_modules"}


def _repo_sources():
    """Hand-written source outside the generated trees (``packaging/*/multi-llm``)."""
    roots = [SKILL_ROOT, REPO_ROOT / "packaging"]
    for root in roots:
        for path in root.rglob("*.py"):
            parts = set(path.relative_to(REPO_ROOT).parts)
            if parts & _SKIPPED_PARTS:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            if re.match(r"packaging/[^/]+/multi-llm/", relative):
                continue
            if path == Path(env_vars.__file__).resolve():
                continue
            yield path
    yield REPO_ROOT / "Makefile"


#: A ``MULTI_LLM_*`` spelling: the family prefix plus the run of name
#: characters after it. The bare prefix (``MULTI_LLM_*``,
#: ``startswith("MULTI_LLM_")``, ``MULTI_LLM_{host}_...``) names no variable.
_SPELLING = re.compile(r"(?<![A-Za-z0-9_])MULTI_LLM_[A-Z0-9_]*")
_BARE_PREFIX = "MULTI_LLM_"

_SCAN_TESTS = "skills/multi-llm/tests/test_codex_package.py"
_SYNTHETIC = ((_SCAN_TESTS,), "a made-up name the environment-scan tests read in memory")

#: Spellings in tracked files that are not variables, each with the paths (or
#: path prefixes) it may appear under and why. This file may spell them too.
NOT_A_VARIABLE = {
    "MULTI_LLM_X": _SYNTHETIC,
    "MULTI_LLM_Y": _SYNTHETIC,
    "MULTI_LLM_Z": _SYNTHETIC,
    "MULTI_LLM_COMMENT": _SYNTHETIC,
    "MULTI_LLM_UNLISTED_REVIEW": _SYNTHETIC,
    "MULTI_LLM_REPLAY_": (
        ("skills/multi-llm/tests/container/records/",),
        "a fixture fragment: a recorded `codex mcp get` stdout cut off mid-name",
    ),
}

_THIS_FILE = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()


def _tracked_spellings() -> dict:
    """``{spelling: {path, ...}}`` over every tracked text file outside the
    generated trees (``packaging/*/multi-llm/`` mirror the payload)."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True, capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    found: dict = {}
    for relative in filter(None, listed):
        if re.match(r"packaging/[^/]+/multi-llm/", relative):
            continue
        try:
            data = (REPO_ROOT / relative).read_bytes()
        except OSError:
            continue  # deleted in the working tree, not yet in the index
        if b"\0" in data or b"MULTI_LLM_" not in data:
            continue
        for spelling in _SPELLING.findall(data.decode("utf-8", errors="replace")):
            if spelling != _BARE_PREFIX:
                found.setdefault(spelling, set()).add(relative)
    return found


class TestTable:
    def test_the_table_is_sound(self):
        assert env_vars.table_problems() == []

    def test_every_constant_matches_its_row(self):
        constants = {
            value for key, value in vars(env_vars).items()
            if key.isupper() and isinstance(value, str) and value.startswith("MULTI_LLM_")
            and not key.startswith("DOC_")
        }
        assert constants == set(env_vars.by_name())

    def test_forwarded_rows_are_real_environment_reads(self):
        for row in env_vars.ENV_VARS:
            if row.forwarded_to_codex:
                assert row.kind not in env_vars.NOT_READ_KINDS + (env_vars.BUILD, env_vars.TEST)

    def test_no_row_names_a_variable_nothing_mentions(self):
        """A row whose name no source file, test or Makefile spells has gone
        stale. A reserved row is exempt: its name sits in a comment or nowhere."""
        text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in _repo_sources() if path.is_file()
        )
        stale = [
            row.name for row in env_vars.ENV_VARS
            if row.kind != env_vars.RESERVED and row.name not in text
        ]
        assert not stale, f"rows with no remaining mention (drop or fix them): {stale}"

    def test_every_name_the_repository_spells_has_a_row(self):
        """The reverse of the row-mention check: a ``MULTI_LLM_*`` name any
        tracked file spells (tests, packaging, the Makefile, docs) has a row,
        unless :data:`NOT_A_VARIABLE` says why not, for the paths it names."""
        rows = set(env_vars.by_name())
        unlisted = {}
        for spelling, paths in _tracked_spellings().items():
            if spelling in rows:
                continue
            allowed = NOT_A_VARIABLE.get(spelling, ((),))[0] + (_THIS_FILE,)
            stray = sorted(p for p in paths if not p.startswith(allowed))
            if stray:
                unlisted[spelling] = stray
        assert not unlisted, (
            f"MULTI_LLM_* names with no row in utils/env_vars.py: {unlisted}. "
            "Add a row (a test seam is kind `test`); a spelling that is not a "
            "variable goes in NOT_A_VARIABLE with its paths and reason"
        )

    def test_every_exclusion_is_still_spelled(self):
        spelled = _tracked_spellings()
        for spelling, (paths, _reason) in NOT_A_VARIABLE.items():
            assert spelling not in env_vars.by_name(), f"{spelling} has a row now"
            assert any(p.startswith(paths) for p in spelled.get(spelling, ())), (
                f"{spelling} no longer appears under {paths}; drop it from NOT_A_VARIABLE"
            )

    def test_kind_counts_cover_every_row(self):
        counts = env_vars.counts_by_kind()
        assert sum(counts.values()) == len(env_vars.ENV_VARS)

    def test_the_advisory_switches_are_read_by_the_advisory_guard(self):
        """``MULTI_LLM_DEBUG`` is a setting the Codex tool server does not
        forward (run the command in a shell to use it); the strict switch is a
        test seam. Both are read by ``utils/advisory.py``."""
        from utils import advisory

        rows = env_vars.by_name()
        debug, strict = rows[env_vars.DEBUG], rows[env_vars.STRICT_ADVISORY]
        assert (debug.kind, debug.owner, debug.forwarded_to_codex) == (
            env_vars.SETTING, "utils/advisory.py", False)
        assert (strict.kind, strict.owner, strict.forwarded_to_codex) == (
            env_vars.TEST, "utils/advisory.py", False)
        assert (advisory.DEBUG_ENV, advisory.STRICT_ENV) == (
            env_vars.DEBUG, env_vars.STRICT_ADVISORY)


class TestDocsTable:
    def test_the_checked_in_table_is_current(self):
        assert env_vars.doc_problem() is None, (
            "docs/configuration.md is stale: run "
            "`python skills/multi-llm/utils/env_vars.py` and commit the result"
        )

    def test_every_setting_is_documented(self):
        body = (REPO_ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
        for row in env_vars.settings():
            assert f"`{row.name}`" in body

    def test_only_settings_reach_the_docs(self):
        table = env_vars.render_table()
        for row in env_vars.ENV_VARS:
            assert (f"`{row.name}`" in table) == (row.kind == env_vars.SETTING)

    def test_a_stale_table_is_reported_and_fixed(self, tmp_path, capsys, monkeypatch):
        doc = tmp_path / "configuration.md"
        fresh = f"intro\n\n{env_vars.DOC_BEGIN}\n{env_vars.DOC_END}\n\noutro\n"
        doc.write_text(fresh, encoding="utf-8")
        monkeypatch.setattr(env_vars, "docs_path", lambda: doc)

        assert "stale" in env_vars.doc_problem(doc)
        assert env_vars.main(["--check"]) == 1
        assert env_vars.main([]) == 0
        assert env_vars.main(["--check"]) == 0
        written = doc.read_text(encoding="utf-8")
        assert written.startswith("intro\n\n") and written.endswith("\n\noutro\n")
        assert "`MULTI_LLM_MAX_DETACHED`" in written

    def test_a_hand_edit_inside_the_markers_fails_the_check(self, tmp_path, monkeypatch):
        doc = tmp_path / "configuration.md"
        doc.write_text(f"{env_vars.DOC_BEGIN}\n{env_vars.DOC_END}\n", encoding="utf-8")
        monkeypatch.setattr(env_vars, "docs_path", lambda: doc)
        assert env_vars.main([]) == 0
        doc.write_text(doc.read_text(encoding="utf-8").replace("forwarded", "FORWARDED", 1),
                       encoding="utf-8")
        assert env_vars.main(["--check"]) == 1

    def test_missing_markers_are_stale_not_created(self, tmp_path, monkeypatch, capsys):
        doc = tmp_path / "configuration.md"
        doc.write_text("no markers\n", encoding="utf-8")
        monkeypatch.setattr(env_vars, "docs_path", lambda: doc)
        assert env_vars.main(["--check"]) == 1
        assert env_vars.main([]) == 1
        assert doc.read_text(encoding="utf-8") == "no markers\n"
        assert "lacks the markers" in capsys.readouterr().err
