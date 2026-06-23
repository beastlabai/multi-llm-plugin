#!/usr/bin/env python3
"""
Regression tests ensuring uv commands use --project (not --directory)
and do not contain duplicated path segments.

Prevents reintroduction of the path-duplication bug where --directory
changes cwd, causing full relative paths to double up:
  ${CLAUDE_SKILL_DIR}/${CLAUDE_SKILL_DIR}/script.py
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validation import prepare_batched_validation_tasks

SKILL_DIR = "${CLAUDE_SKILL_DIR}"
DOUBLED_PATH = f"{SKILL_DIR}/{SKILL_DIR}/"


class TestReaggregateCommandUsesProject:
    """Verify reaggregate_command in prepare_batched_validation_tasks uses --project."""

    def _make_groups(self):
        return [
            {
                "group_id": "g1",
                "importance": "HIGH",
                "suggestions": [{"id": "s1", "description": "test", "section": "s"}],
            }
        ]

    def test_reaggregate_uses_project_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=self._make_groups(),
                context="test context",
                output_dir=tmpdir,
                plan_file="/tmp/plan.md",
                orchestrator="review_plan_orchestrator.py",
            )
            cmd = result["reaggregate_command"]
            assert "--project" in cmd, f"Expected --project in: {cmd}"
            assert "--directory" not in cmd, f"Found --directory in: {cmd}"

    def test_reaggregate_no_doubled_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=self._make_groups(),
                context="test context",
                output_dir=tmpdir,
                plan_file="/tmp/plan.md",
                orchestrator="review_plan_orchestrator.py",
            )
            cmd = result["reaggregate_command"]
            assert DOUBLED_PATH not in cmd, f"Doubled path found in: {cmd}"

    def test_reaggregate_contains_full_script_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=self._make_groups(),
                context="test context",
                output_dir=tmpdir,
                plan_file="/tmp/plan.md",
                orchestrator="review_plan_orchestrator.py",
            )
            cmd = result["reaggregate_command"]
            assert f"{SKILL_DIR}/review_plan_orchestrator.py" in cmd


class TestNoDirectoryInSourceFiles:
    """Scan Python source and instruction files for --directory remnants."""

    SKILL_ROOT = Path(__file__).parent.parent

    def _get_files(self, pattern):
        return list(self.SKILL_ROOT.glob(pattern))

    def test_no_directory_in_instructions(self):
        for md_file in self._get_files("instructions/*.md"):
            content = md_file.read_text()
            matches = [
                (i + 1, line)
                for i, line in enumerate(content.splitlines())
                if "uv run --directory" in line
            ]
            assert not matches, (
                f"{md_file.name} still uses --directory at lines: "
                + ", ".join(f"{ln}: {txt.strip()}" for ln, txt in matches)
            )

    def test_no_directory_in_orchestrators(self):
        for py_file in self._get_files("*_orchestrator.py"):
            content = py_file.read_text()
            matches = [
                (i + 1, line)
                for i, line in enumerate(content.splitlines())
                if "uv run --directory" in line
            ]
            assert not matches, (
                f"{py_file.name} still uses --directory at lines: "
                + ", ".join(f"{ln}: {txt.strip()}" for ln, txt in matches)
            )

    def test_no_directory_in_validation(self):
        val_file = self.SKILL_ROOT / "utils" / "validation.py"
        content = val_file.read_text()
        matches = [
            (i + 1, line)
            for i, line in enumerate(content.splitlines())
            if "uv run --directory" in line
        ]
        assert not matches, (
            f"validation.py still uses --directory at lines: "
            + ", ".join(f"{ln}: {txt.strip()}" for ln, txt in matches)
        )

    def test_no_doubled_paths_in_source(self):
        for py_file in list(self._get_files("*.py")) + list(
            self._get_files("utils/*.py")
        ):
            content = py_file.read_text()
            if DOUBLED_PATH in content:
                lines = [
                    (i + 1, line)
                    for i, line in enumerate(content.splitlines())
                    if DOUBLED_PATH in line
                ]
                pytest.fail(
                    f"{py_file.name} contains doubled path at lines: "
                    + ", ".join(f"{ln}" for ln, _ in lines)
                )
