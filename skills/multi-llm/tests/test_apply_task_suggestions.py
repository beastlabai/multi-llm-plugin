"""Unit tests for apply_task_suggestions_orchestrator.py.

Tests cover:
- find_tasks_file() via the shared utility (TASKS_FILE marker and fallback paths)
- _generate_single_batch_prompt() — task-specific instructions per suggestion type
- filter_items() — honoring user selections (skip/approve)
- load_and_merge_user_selections() — correctly merging HTML and MD selections
- format_item_for_output() — includes task_reference field
- CLI argument parsing — required args, flag combinations
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from apply_task_suggestions_orchestrator import ApplyTaskSuggestionsOrchestrator
from utils.apply_path_helpers import load_json_file
from utils.apply_selection_helpers import (
    merge_edited_descriptions,
    merge_validation_with_groups,
    resolve_priority_args,
)
from utils.filtering import filter_items
from utils.output_handler import derive_prefix, find_output_dir
from utils.suggestion_batcher import SuggestionBatch
from utils.tasks_file import find_tasks_file
from utils.report_parser import merge_selections


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_with_tasks_file_marker(tmp_path):
    """Create a plan file with a TASKS_FILE comment and a valid tasks file."""
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()

    plan_content = """\
# My Plan
<!-- TASKS_FILE: my-plan/tasks/tasks.md -->

## Overview
Some plan content.
"""
    plan_path = plan_dir / "my-plan.md"
    plan_path.write_text(plan_content, encoding="utf-8")

    # Create the tasks file at the referenced location
    tasks_dir = plan_dir / "my-plan" / "tasks"
    tasks_dir.mkdir(parents=True)
    tasks_file = tasks_dir / "tasks.md"
    tasks_file.write_text("# Tasks\n\n## T001: First task\nDo something.\n", encoding="utf-8")

    return plan_path, tasks_file


@pytest.fixture
def plan_with_default_tasks(tmp_path):
    """Create a plan file without TASKS_FILE comment using default location."""
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()

    plan_content = "# Test Plan\n\n## Overview\nA test plan.\n"
    plan_path = plan_dir / "test-plan.md"
    plan_path.write_text(plan_content, encoding="utf-8")

    # Create tasks file at the default location: {plan_dir}/{prefix}/tasks/tasks.md
    prefix = derive_prefix(str(plan_path))
    tasks_dir = plan_dir / prefix / "tasks"
    tasks_dir.mkdir(parents=True)
    tasks_file = tasks_dir / "tasks.md"
    tasks_file.write_text("# Tasks\n\n## T001: Setup project\nInitialize.\n", encoding="utf-8")

    return plan_path, tasks_file


@pytest.fixture
def sample_groups():
    """Create sample suggestion groups for testing."""
    return [
        {
            "theme": "Add input validation",
            "category": "security",
            "models": ["claude-sonnet", "gpt-4"],
            "suggestions": [{
                "title": "Add email validation",
                "desc": "Add regex validation for email.",
                "type": "addition",
                "reference": "### Task T001: Create user model",
                "importance": "HIGH",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Improve error messages",
            "category": "usability",
            "models": ["gpt-4"],
            "suggestions": [{
                "title": "Improve error descriptions",
                "desc": "Make error messages more descriptive.",
                "type": "modification",
                "reference": "### Task T003: Error handling",
                "importance": "MEDIUM",
                "source_model": "gpt-4"
            }]
        },
        {
            "theme": "Remove deprecated endpoint",
            "category": "cleanup",
            "models": ["claude-sonnet"],
            "suggestions": [{
                "title": "Remove /api/v1/old endpoint",
                "desc": "Remove the deprecated endpoint.",
                "type": "deletion",
                "reference": "### Task T005: API routes",
                "importance": "LOW",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Clarify caching strategy",
            "category": "architecture",
            "models": ["gemini-pro"],
            "suggestions": [{
                "title": "Clarify caching approach",
                "desc": "The task description is unclear about caching strategy.",
                "type": "clarification",
                "reference": "T002",
                "importance": "MEDIUM",
                "source_model": "gemini-pro"
            }]
        },
    ]


@pytest.fixture
def sample_validation():
    """Create sample validation results matching the sample groups."""
    return [
        {"group_index": 0, "status": "valid", "reason": "Clear suggestion.", "confidence": 0.95},
        {"group_index": 1, "status": "needs-human-decision", "reason": "Ambiguous scope.", "confidence": 0.55},
        {"group_index": 2, "status": "valid", "reason": "Straightforward deletion.", "confidence": 0.90},
        {"group_index": 3, "status": "validation_failed", "reason": "Parse error.", "confidence": 0.0,
         "error_type": "parsing_error", "recoverable": True},
    ]


@pytest.fixture
def merged_groups(sample_groups, sample_validation):
    """Return groups merged with validation results."""
    return merge_validation_with_groups(sample_groups, sample_validation)


# ---------------------------------------------------------------------------
# Tests: find_tasks_file()
# ---------------------------------------------------------------------------


class TestFindTasksFile:
    """Tests for find_tasks_file() via the shared utility."""

    def test_resolves_tasks_file_marker(self, plan_with_tasks_file_marker):
        """TASKS_FILE marker in plan is resolved correctly."""
        plan_path, expected_tasks_file = plan_with_tasks_file_marker
        result = find_tasks_file(str(plan_path))
        assert os.path.realpath(result) == os.path.realpath(str(expected_tasks_file))

    def test_fallback_to_default_path(self, plan_with_default_tasks):
        """Falls back to {prefix}/tasks/tasks.md when no marker present."""
        plan_path, expected_tasks_file = plan_with_default_tasks
        result = find_tasks_file(str(plan_path))
        assert os.path.realpath(result) == os.path.realpath(str(expected_tasks_file))

    def test_raises_on_missing_tasks_file(self, tmp_path):
        """Raises FileNotFoundError when no tasks file exists."""
        plan_path = tmp_path / "orphan-plan.md"
        plan_path.write_text("# No tasks\n\nJust a plan.\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="No tasks file found"):
            find_tasks_file(str(plan_path))

    def test_raises_on_marker_pointing_to_nonexistent(self, tmp_path):
        """Raises FileNotFoundError when TASKS_FILE marker points to missing file."""
        plan_dir = tmp_path / "plans"
        plan_dir.mkdir()
        plan_content = "# Plan\n<!-- TASKS_FILE: nonexistent/tasks.md -->\n"
        plan_path = plan_dir / "broken.md"
        plan_path.write_text(plan_content, encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="not found"):
            find_tasks_file(str(plan_path))

    def test_raises_on_empty_tasks_file(self, tmp_path):
        """Raises ValueError when tasks file is empty."""
        plan_dir = tmp_path / "plans"
        plan_dir.mkdir()
        plan_content = "# Plan\n<!-- TASKS_FILE: empty-plan/tasks/tasks.md -->\n"
        plan_path = plan_dir / "empty-plan.md"
        plan_path.write_text(plan_content, encoding="utf-8")

        tasks_dir = plan_dir / "empty-plan" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "tasks.md").write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="empty"):
            find_tasks_file(str(plan_path))

    def test_raises_on_malformed_tasks_file(self, tmp_path):
        """Raises ValueError when tasks file has no task headers."""
        plan_dir = tmp_path / "plans"
        plan_dir.mkdir()
        plan_content = "# Plan\n<!-- TASKS_FILE: bad/tasks/tasks.md -->\n"
        plan_path = plan_dir / "bad.md"
        plan_path.write_text(plan_content, encoding="utf-8")

        tasks_dir = plan_dir / "bad" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "tasks.md").write_text("# Tasks\n\nNo actual task headers here.\n", encoding="utf-8")

        with pytest.raises(ValueError, match="malformed"):
            find_tasks_file(str(plan_path))

    def test_rejects_absolute_path_in_marker(self, tmp_path):
        """Rejects absolute path in TASKS_FILE marker (security check)."""
        plan_dir = tmp_path / "plans"
        plan_dir.mkdir()
        plan_content = "# Plan\n<!-- TASKS_FILE: /etc/passwd -->\n"
        plan_path = plan_dir / "evil.md"
        plan_path.write_text(plan_content, encoding="utf-8")

        # find_tasks_file calls sys.exit(1) for absolute paths
        with pytest.raises(SystemExit):
            find_tasks_file(str(plan_path))

    def test_rejects_path_traversal_in_marker(self, tmp_path):
        """Rejects .. path traversal in TASKS_FILE marker."""
        plan_dir = tmp_path / "plans"
        plan_dir.mkdir()
        plan_content = "# Plan\n<!-- TASKS_FILE: ../../etc/passwd -->\n"
        plan_path = plan_dir / "traversal.md"
        plan_path.write_text(plan_content, encoding="utf-8")

        with pytest.raises(SystemExit):
            find_tasks_file(str(plan_path))


# ---------------------------------------------------------------------------
# Tests: _generate_single_batch_prompt() (instance method)
# ---------------------------------------------------------------------------


class TestGenerateBatchPrompt:
    """Tests for _generate_single_batch_prompt() — task-specific instructions per type."""

    @pytest.fixture
    def tasks_content(self):
        """Sample tasks.md content for prompt generation."""
        return "# Tasks\n\n## T001: Setup\nSetup the project.\n\n## T002: Core\nBuild core logic.\n"

    def _make_batch(self, suggestions, section_key="unknown", batch_type="mixed"):
        """Helper to create a SuggestionBatch."""
        return SuggestionBatch(
            suggestions=suggestions,
            section_key=section_key,
            batch_type=batch_type,
            total_chars=sum(len(s.get("description", "")) for s in suggestions),
        )

    def _make_orchestrator(self, plan_path="/plans/p.md", tasks_file="/plans/p/tasks/tasks.md", tasks_content=""):
        """Helper to create an ApplyTaskSuggestionsOrchestrator with preset state."""
        args = argparse.Namespace(plan_file=plan_path)
        orch = ApplyTaskSuggestionsOrchestrator.__new__(ApplyTaskSuggestionsOrchestrator)
        # Initialize only the fields needed for prompt generation
        orch.args = args
        orch.plan_path = plan_path
        orch.tasks_file = tasks_file
        orch.tasks_content = tasks_content
        return orch

    def test_single_addition_prompt(self, tasks_content):
        """Single addition suggestion produces correct task-specific instructions."""
        batch = self._make_batch([{
            "title": "Add caching layer",
            "type": "addition",
            "reference": "### Task T002: Core",
            "importance": "HIGH",
            "description": "Add a caching layer.",
        }])
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "Add caching layer" in prompt
        assert "addition" in prompt.lower() or "Addition" in prompt
        assert "### Task T002: Core" in prompt
        assert "Target file (to edit)" in prompt
        assert "Plan file (context only)" in prompt
        # Must include type-specific edit rules
        assert "### Addition" in prompt
        assert "### Modification" in prompt
        assert "### Deletion" in prompt
        assert "### Clarification" in prompt

    def test_single_modification_prompt(self, tasks_content):
        """Single modification suggestion includes correct instructions."""
        batch = self._make_batch([{
            "title": "Update error handling",
            "type": "modification",
            "reference": "### Task T001: Setup",
            "importance": "MEDIUM",
            "description": "Update error handling approach.",
        }])
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "modification" in prompt.lower()
        assert "Update error handling" in prompt

    def test_single_deletion_prompt(self, tasks_content):
        """Single deletion suggestion includes cleanup instructions."""
        batch = self._make_batch([{
            "title": "Remove deprecated task",
            "type": "deletion",
            "reference": "### Task T005: Old code",
            "importance": "LOW",
            "description": "Remove the deprecated task.",
        }])
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "deletion" in prompt.lower()
        assert "Remove the deprecated task" in prompt
        # Check deletion-specific instructions are present
        assert "Remove the specified task entirely" in prompt
        assert "depends_on" in prompt

    def test_single_clarification_prompt(self, tasks_content):
        """Single clarification suggestion includes rewrite instructions."""
        batch = self._make_batch([{
            "title": "Clarify deployment steps",
            "type": "clarification",
            "reference": "T003",
            "importance": "MEDIUM",
            "description": "The task description is vague about deployment steps.",
        }])
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "clarification" in prompt.lower()
        assert "Rewrite the task description" in prompt

    def test_multi_suggestion_batch_prompt(self, tasks_content):
        """Multiple suggestions produce a batch prompt with all suggestions listed."""
        suggestions = [
            {
                "title": "Add validation",
                "type": "addition",
                "reference": "### Task T001: Setup",
                "importance": "HIGH",
                "description": "Add input validation.",
            },
            {
                "title": "Fix error handling",
                "type": "modification",
                "reference": "### Task T001: Setup",
                "importance": "MEDIUM",
                "description": "Fix error handling in setup.",
            },
        ]
        batch = self._make_batch(suggestions, section_key="### Task T001: Setup")
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "2 related task suggestions" in prompt
        assert "Suggestion 1:" in prompt
        assert "Suggestion 2:" in prompt
        assert "Add validation" in prompt
        assert "Fix error handling" in prompt
        assert "Section: ### Task T001: Setup" in prompt
        assert "Apply ALL suggestions in this batch" in prompt

    def test_empty_batch_raises_error(self, tasks_content):
        """Empty batch raises ValueError."""
        batch = self._make_batch([])
        orch = self._make_orchestrator(tasks_content=tasks_content)
        with pytest.raises(ValueError, match="empty batch"):
            orch._generate_single_batch_prompt(batch)

    def test_prompt_includes_tasks_content(self, tasks_content):
        """Prompt includes current tasks.md content for context."""
        batch = self._make_batch([{
            "title": "Test",
            "type": "addition",
            "reference": "T001",
            "importance": "LOW",
            "description": "Test suggestion.",
        }])
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "## T001: Setup" in prompt
        assert "## T002: Core" in prompt

    def test_long_tasks_content_is_truncated(self):
        """Tasks content longer than 3000 chars is truncated with ellipsis."""
        long_content = "x" * 5000
        batch = self._make_batch([{
            "title": "Test",
            "type": "addition",
            "reference": "T001",
            "importance": "LOW",
            "description": "Test.",
        }])
        orch = self._make_orchestrator(tasks_content=long_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "..." in prompt
        # The full 5000 chars should NOT be in the prompt
        assert "x" * 5000 not in prompt

    def test_type_instructions_present_for_all_types(self, tasks_content):
        """All four type-specific instruction blocks appear in prompt."""
        batch = self._make_batch([{
            "title": "Test",
            "type": "addition",
            "reference": "T001",
            "importance": "LOW",
            "description": "Test.",
        }])
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "### Addition" in prompt
        assert "### Modification" in prompt
        assert "### Deletion" in prompt
        assert "### Clarification" in prompt
        # Check key instructions per type
        assert "Add new task(s)" in prompt
        assert "Update the specified task fields" in prompt
        assert "Remove the specified task entirely" in prompt
        assert "Rewrite the task description" in prompt

    def test_single_suggestion_includes_prior_changes_placeholder(self, tasks_content):
        """Single suggestion format includes {prior_changes_context} placeholder."""
        batch = self._make_batch([{
            "title": "Test",
            "type": "addition",
            "reference": "T001",
            "importance": "LOW",
            "description": "Test suggestion.",
        }])
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "{prior_changes_context}" in prompt
        assert "## Changes Applied in Prior Batches" in prompt

    def test_multi_suggestion_includes_prior_changes_placeholder(self, tasks_content):
        """Multiple suggestion format includes {prior_changes_context} placeholder."""
        suggestions = [
            {
                "title": "Add validation",
                "type": "addition",
                "reference": "T001",
                "importance": "HIGH",
                "description": "Add input validation.",
            },
            {
                "title": "Fix error handling",
                "type": "modification",
                "reference": "T001",
                "importance": "MEDIUM",
                "description": "Fix error handling.",
            },
        ]
        batch = self._make_batch(suggestions, section_key="### Task T001: Setup")
        orch = self._make_orchestrator(tasks_content=tasks_content)
        prompt = orch._generate_single_batch_prompt(batch)

        assert "{prior_changes_context}" in prompt
        assert "## Changes Applied in Prior Batches" in prompt


# ---------------------------------------------------------------------------
# Tests: filter_items() (from utils.filtering)
# ---------------------------------------------------------------------------


class TestFilterSuggestions:
    """Tests for filter_items() — honoring user selections."""

    def test_valid_items_pass_through(self, merged_groups):
        """Valid items go to the valid list."""
        valid, needs_human, skipped, report = filter_items(merged_groups)

        # Groups 0 and 2 are valid
        assert len(valid) == 2
        assert all(g["validation_status"] == "valid" for g in valid)

    def test_needs_human_items_identified(self, merged_groups):
        """Needs-human-decision items go to needs_human list."""
        valid, needs_human, skipped, report = filter_items(merged_groups)

        # Group 1 (needs-human-decision) + Group 3 (validation_failed)
        assert len(needs_human) == 2

    def test_skip_all_human_skips_human_items(self, merged_groups):
        """skip_all_human=True skips all human-review items."""
        valid, needs_human, skipped, report = filter_items(
            merged_groups, skip_all_human=True
        )

        assert len(needs_human) == 0
        assert len(skipped) == 2  # groups 1 and 3

    def test_approve_all_human_approves_all(self, merged_groups):
        """approve_all_human=True auto-approves all human-review items."""
        valid, needs_human, skipped, report = filter_items(
            merged_groups, approve_all_human=True
        )

        assert len(valid) == 4
        assert len(needs_human) == 0
        assert len(skipped) == 0

    def test_approve_importance_levels_selective(self, merged_groups):
        """approve_importance_levels only approves matching importance levels."""
        valid, needs_human, skipped, report = filter_items(
            merged_groups, approve_importance_levels=["MEDIUM"]
        )

        # Group 1 (MEDIUM, needs-human) should be auto-approved
        # Group 3 (MEDIUM, validation_failed) should also be auto-approved by importance
        medium_approved = [g for g in valid if g.get("auto_approved")]
        assert len(medium_approved) >= 1

    def test_min_priority_medium_skips_low(self, merged_groups):
        """min_priority='medium' skips LOW importance valid items."""
        valid, needs_human, skipped, report = filter_items(
            merged_groups, min_priority="medium"
        )

        # Group 2 (LOW, valid) should be skipped
        skipped_importances = [g.get("_importance", "") for g in skipped]
        assert "LOW" in skipped_importances

    def test_min_priority_high_only_keeps_high(self, merged_groups):
        """min_priority='high' only keeps HIGH importance items."""
        valid, needs_human, skipped, report = filter_items(
            merged_groups, min_priority="high"
        )

        for g in valid:
            assert g.get("_importance") == "HIGH"

    def test_dry_run_does_not_modify_groups(self, merged_groups):
        """dry_run=True does not modify group data."""
        valid, needs_human, skipped, report = filter_items(
            merged_groups, approve_all_human=True, dry_run=True
        )

        # In dry run, items stay in needs_human
        assert len(valid) == 2  # only originally valid
        assert len(needs_human) == 2
        assert "would_auto_approve" in report
        assert len(report["would_auto_approve"]) == 2

    def test_skip_human_review_legacy(self, merged_groups):
        """Legacy skip_human_review flag skips needs-human-decision items.

        Note: unlike skip_all_human, the legacy flag does NOT skip
        validation_failed items — only needs-human-decision.
        """
        valid, needs_human, skipped, report = filter_items(
            merged_groups, skip_human_review=True
        )

        # Group 1 (needs-human-decision) should be skipped
        # Group 3 (validation_failed) stays in needs_human with legacy flag
        skipped_statuses = {g.get("validation_status") for g in skipped}
        assert "needs-human-decision" in skipped_statuses
        assert len(skipped) >= 1


# ---------------------------------------------------------------------------
# Tests: merge_selections() (load_and_merge_user_selections)
# ---------------------------------------------------------------------------


class TestLoadAndMergeUserSelections:
    """Tests for merge_selections() — merging HTML and MD selections."""

    def test_html_none_returns_md_values(self):
        """When HTML selections are None, MD values are returned."""
        md_groups = {"hash_g1", "hash_g2"}
        md_suggestions = {"hash_s1"}
        md_edited = {"hash_s2": ("old desc", "new desc")}

        merged_groups, merged_suggestions, merged_edited = merge_selections(
            html_selections=None,
            md_skipped_groups=md_groups,
            md_skipped_suggestions=md_suggestions,
            md_edited=md_edited,
        )

        assert merged_groups == md_groups
        assert merged_suggestions == md_suggestions
        assert merged_edited == {"hash_s2": "new desc"}

    def test_html_skips_are_unioned_with_md(self):
        """HTML skips are added to MD skips (union)."""
        md_groups = {"hash_g1"}
        md_suggestions = {"hash_s1"}
        html_selections = {
            "skipped_groups": ["hash_g2", "hash_g3"],
            "skipped_suggestions": ["hash_s2"],
            "edited_descriptions": {},
        }

        merged_groups, merged_suggestions, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=md_groups,
            md_skipped_suggestions=md_suggestions,
            md_edited={},
        )

        assert merged_groups == {"hash_g1", "hash_g2", "hash_g3"}
        assert merged_suggestions == {"hash_s1", "hash_s2"}

    def test_empty_html_skips_do_not_erase_md(self):
        """Empty HTML skip lists do not erase MD skips."""
        md_groups = {"hash_g1"}
        md_suggestions = {"hash_s1"}
        html_selections = {
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {},
        }

        merged_groups, merged_suggestions, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=md_groups,
            md_skipped_suggestions=md_suggestions,
            md_edited={},
        )

        assert merged_groups == {"hash_g1"}
        assert merged_suggestions == {"hash_s1"}

    def test_html_edited_descriptions_win_per_key(self):
        """HTML edited descriptions override MD for the same key."""
        md_edited = {"hash_s1": ("orig", "md_edit")}
        html_selections = {
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {"hash_s1": "html_edit"},
        }

        _, _, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=set(),
            md_skipped_suggestions=set(),
            md_edited=md_edited,
        )

        # HTML edit should win
        assert merged_edited["hash_s1"] == "html_edit"

    def test_md_edits_preserved_when_html_has_different_keys(self):
        """MD edited descriptions are preserved when HTML has different keys."""
        md_edited = {"hash_s1": ("orig1", "md_edit1")}
        html_selections = {
            "skipped_groups": [],
            "skipped_suggestions": [],
            "edited_descriptions": {"hash_s2": "html_edit2"},
        }

        _, _, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=set(),
            md_skipped_suggestions=set(),
            md_edited=md_edited,
        )

        assert merged_edited["hash_s1"] == "md_edit1"
        assert merged_edited["hash_s2"] == "html_edit2"

    def test_full_merge_scenario(self):
        """Full merge with both MD and HTML having various entries."""
        md_groups = {"g_hash_1"}
        md_suggestions = {"s_hash_1", "s_hash_2"}
        md_edited = {
            "s_hash_1": ("old1", "md_new1"),
            "s_hash_3": ("old3", "md_new3"),
        }
        html_selections = {
            "skipped_groups": ["g_hash_2"],
            "skipped_suggestions": ["s_hash_3"],
            "edited_descriptions": {"s_hash_1": "html_new1"},
        }

        merged_groups, merged_suggestions, merged_edited = merge_selections(
            html_selections=html_selections,
            md_skipped_groups=md_groups,
            md_skipped_suggestions=md_suggestions,
            md_edited=md_edited,
        )

        assert merged_groups == {"g_hash_1", "g_hash_2"}
        assert merged_suggestions == {"s_hash_1", "s_hash_2", "s_hash_3"}
        # HTML edit wins for s_hash_1
        assert merged_edited["s_hash_1"] == "html_new1"
        # MD edit preserved for s_hash_3
        assert merged_edited["s_hash_3"] == "md_new3"


# ---------------------------------------------------------------------------
# Tests: format_item_for_output() (instance method)
# ---------------------------------------------------------------------------


class TestFormatSuggestionForOutput:
    """Tests for format_item_for_output() — includes task_reference field."""

    def _make_orchestrator(self):
        """Helper to create a minimal ApplyTaskSuggestionsOrchestrator instance."""
        orch = ApplyTaskSuggestionsOrchestrator.__new__(ApplyTaskSuggestionsOrchestrator)
        orch.args = argparse.Namespace(plan_file="p.md")
        return orch

    def test_includes_task_reference_from_header(self):
        """task_reference is derived from '### Task T001: ...' reference."""
        group = {
            "theme": "Add validation",
            "category": "security",
            "models": ["claude-sonnet"],
            "validation_status": "valid",
            "validation_reason": "Clear.",
            "validation_confidence": 0.95,
            "suggestions": [{
                "title": "Add email validation",
                "desc": "Validate emails.",
                "type": "addition",
                "reference": "### Task T001: Create user model",
                "importance": "HIGH",
            }],
        }
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(group, 0)

        assert result["task_reference"] == "Task T001"
        assert result["reference"] == "### Task T001: Create user model"

    def test_task_reference_from_bare_id(self):
        """task_reference is derived from bare 'T003' reference."""
        group = {
            "theme": "Clarify something",
            "suggestions": [{
                "title": "Clarify caching",
                "desc": "Unclear.",
                "type": "clarification",
                "reference": "T003",
                "importance": "MEDIUM",
            }],
        }
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(group, 1)

        assert result["task_reference"] == "Task T003"

    def test_task_reference_fallback_to_raw_reference(self):
        """task_reference falls back to raw reference when no T-ID found."""
        group = {
            "theme": "Misc",
            "suggestions": [{
                "title": "Some suggestion",
                "desc": "Details.",
                "type": "modification",
                "reference": "Section 5: Deployment",
                "importance": "LOW",
            }],
        }
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(group, 2)

        assert result["task_reference"] == "Section 5: Deployment"

    def test_task_reference_empty_when_no_reference(self):
        """task_reference is empty string when suggestion has no reference."""
        group = {
            "theme": "General",
            "suggestions": [{
                "title": "Generic improvement",
                "desc": "Improve things.",
                "type": "addition",
                "reference": "",
                "importance": "LOW",
            }],
        }
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(group, 3)

        assert result["task_reference"] == ""

    def test_output_contains_all_expected_fields(self):
        """Output dict contains all expected fields."""
        group = {
            "theme": "Test theme",
            "category": "testing",
            "models": ["model-a"],
            "validation_status": "valid",
            "validation_reason": "Ok.",
            "validation_confidence": 0.9,
            "suggestions": [{
                "title": "Test title",
                "desc": "Test desc.",
                "type": "addition",
                "reference": "### Task T001: Setup",
                "importance": "HIGH",
            }],
        }
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(group, 0)

        expected_keys = {
            "index", "group_id", "title", "description", "type", "reference",
            "task_reference", "importance", "theme", "category",
            "validation_status", "validation_reason",
            "validation_confidence", "models", "suggestion_count",
        }
        assert set(result.keys()) == expected_keys

    def test_multiple_suggestions_combined_description(self):
        """Multiple suggestions in group produce combined description."""
        group = {
            "theme": "Error handling",
            "category": "robustness",
            "models": ["model-a", "model-b"],
            "validation_status": "valid",
            "validation_reason": "Agreed.",
            "validation_confidence": 0.88,
            "suggestions": [
                {
                    "title": "Add try-catch",
                    "desc": "Add error handling blocks.",
                    "type": "addition",
                    "reference": "### Task T002: Core",
                    "importance": "HIGH",
                    "source_model": "model-a",
                },
                {
                    "title": "Add retry logic",
                    "desc": "Add retry for transient failures.",
                    "type": "addition",
                    "reference": "### Task T002: Core",
                    "importance": "HIGH",
                    "source_model": "model-b",
                },
            ],
        }
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(group, 0)

        assert result["suggestion_count"] == 2
        assert "Add error handling blocks." in result["description"]
        assert "Add retry for transient failures." in result["description"]
        assert "Agreed by models:" not in result["description"]

    def test_empty_suggestions_uses_theme_defaults(self):
        """Group with no suggestions uses theme as title and default values."""
        group = {
            "theme": "Fallback theme",
            "suggestions": [],
        }
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(group, 5)

        assert result["title"] == "Fallback theme"
        assert result["type"] == "modification"
        assert result["importance"] == "MEDIUM"
        assert result["task_reference"] == ""
        assert result["suggestion_count"] == 0


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    """Tests for ApplyTaskSuggestionsOrchestrator.parse_args() — required args, flag combinations."""

    def test_plan_file_required(self):
        """--plan-file is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["prog"]):
                ApplyTaskSuggestionsOrchestrator.parse_args()

    def test_plan_file_accepted(self):
        """--plan-file is parsed correctly."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/my-plan.md"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.plan_file == "plans/my-plan.md"

    def test_dry_run_flag(self):
        """--dry-run flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--dry-run"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.dry_run is True

    def test_skip_all_human_flag(self):
        """--skip-all-human flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--skip-all-human"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.skip_all_human is True

    def test_approve_all_flag(self):
        """--approve-all flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--approve-all"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.approve_all is True

    def test_approve_all_low_flag(self):
        """--approve-all-low flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--approve-all-low"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.approve_all_low is True

    def test_yes_flag_short(self):
        """-y short flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "-y"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.yes is True

    def test_force_flag_short(self):
        """-f short flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "-f"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.force is True

    def test_include_high_flag(self):
        """--include-high flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--include-high"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.include_high is True

    def test_resume_and_fresh_flags(self):
        """--resume and --fresh flags are parsed independently."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--resume"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.resume is True
        assert args.fresh is False

        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--fresh"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.resume is False
        assert args.fresh is True

    def test_no_batch_flag(self):
        """--no-batch flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--no-batch"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.no_batch is True

    def test_max_batch_size_default(self):
        """--max-batch-size default is 4."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.max_batch_size == 4

    def test_max_batch_size_custom(self):
        """--max-batch-size custom value is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--max-batch-size", "8"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.max_batch_size == 8

    def test_min_priority_choices(self):
        """--min-priority accepts low, medium, high."""
        for level in ["low", "medium", "high"]:
            with patch("sys.argv", ["prog", "--plan-file", "p.md", "--min-priority", level]):
                args = ApplyTaskSuggestionsOrchestrator.parse_args()
            assert args.min_priority == level

    def test_min_priority_invalid_rejected(self):
        """--min-priority rejects invalid values."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["prog", "--plan-file", "p.md", "--min-priority", "critical"]):
                ApplyTaskSuggestionsOrchestrator.parse_args()

    def test_approve_importance_multiple_levels(self):
        """--approve-importance accepts multiple levels."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md",
                                 "--approve-importance", "LOW", "MEDIUM"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.approve_importance == ["LOW", "MEDIUM"]

    def test_batch_review_mode_choices(self):
        """--batch-review-mode accepts valid choices."""
        for mode in ["individual", "by-importance", "summary-only"]:
            with patch("sys.argv", ["prog", "--plan-file", "p.md", "--batch-review-mode", mode]):
                args = ApplyTaskSuggestionsOrchestrator.parse_args()
            assert args.batch_review_mode == mode

    def test_batch_review_mode_default(self):
        """--batch-review-mode default is by-importance."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.batch_review_mode == "by-importance"

    def test_skip_flag(self):
        """--skip flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--skip"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.skip is True

    def test_no_confirm_flag(self):
        """--no-confirm flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--no-confirm"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.no_confirm is True

    def test_accept_stale_consolidation_flag(self):
        """--accept-stale-consolidation flag is parsed."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--accept-stale-consolidation"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.accept_stale_consolidation is True

    def test_combined_flags(self):
        """Multiple flags can be combined."""
        with patch("sys.argv", [
            "prog", "--plan-file", "p.md",
            "--approve-all", "--yes", "--include-high",
            "--dry-run", "--no-batch",
        ]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert args.approve_all is True
        assert args.yes is True
        assert args.include_high is True
        assert args.dry_run is True
        assert args.no_batch is True


# ---------------------------------------------------------------------------
# Tests: resolve_priority_args()
# ---------------------------------------------------------------------------


class TestResolvePriorityArgs:
    """Tests for resolve_priority_args()."""

    def test_default_is_low(self):
        """Default (no --min-priority) resolves to 'low'."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert resolve_priority_args(args) == "low"

    def test_explicit_medium(self):
        """Explicit --min-priority medium returns 'medium'."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--min-priority", "medium"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert resolve_priority_args(args) == "medium"

    def test_explicit_high(self):
        """Explicit --min-priority high returns 'high'."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--min-priority", "high"]):
            args = ApplyTaskSuggestionsOrchestrator.parse_args()
        assert resolve_priority_args(args) == "high"


# ---------------------------------------------------------------------------
# Tests: merge_validation_with_groups()
# ---------------------------------------------------------------------------


class TestMergeValidationWithGroups:
    """Tests for merge_validation_with_groups()."""

    def test_basic_merge(self, sample_groups, sample_validation):
        """Validation results are merged into groups."""
        merged = merge_validation_with_groups(sample_groups, sample_validation)

        assert len(merged) == 4
        assert merged[0]["validation_status"] == "valid"
        assert merged[1]["validation_status"] == "needs-human-decision"
        assert merged[2]["validation_status"] == "valid"
        assert merged[3]["validation_status"] == "validation_failed"

    def test_error_type_copied(self, sample_groups, sample_validation):
        """error_type and recoverable fields are copied from validation."""
        merged = merge_validation_with_groups(sample_groups, sample_validation)

        assert merged[3]["validation_error_type"] == "parsing_error"
        assert merged[3]["validation_recoverable"] is True

    def test_missing_validation_defaults_to_needs_human(self, sample_groups):
        """Groups without validation default to needs-human-decision."""
        partial_validation = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9},
        ]
        merged = merge_validation_with_groups(sample_groups, partial_validation)

        assert merged[0]["validation_status"] == "valid"
        assert merged[1]["validation_status"] == "needs-human-decision"
        assert merged[2]["validation_status"] == "needs-human-decision"
        assert merged[3]["validation_status"] == "needs-human-decision"

    def test_group_index_tracked(self, sample_groups, sample_validation):
        """group_index is set on each merged group."""
        merged = merge_validation_with_groups(sample_groups, sample_validation)

        for i, group in enumerate(merged):
            assert group["group_index"] == i


# ---------------------------------------------------------------------------
# Tests: merge_edited_descriptions()
# ---------------------------------------------------------------------------


class TestMergeEditedDescriptions:
    """Tests for merge_edited_descriptions()."""

    def test_positional_id_matching(self):
        """Descriptions matched by positional ID (G1S1) are updated."""
        groups = [
            {
                "theme": "Test",
                "suggestions": [
                    {"title": "First", "desc": "Original desc."},
                ],
            },
        ]
        edited = {"G1S1": ("Original desc.", "Edited desc.")}
        updated, log = merge_edited_descriptions(groups, edited)

        assert updated[0]["suggestions"][0]["desc"] == "Edited desc."
        assert updated[0]["suggestions"][0]["_description_edited"] is True
        assert updated[0]["suggestions"][0]["_original_desc"] == "Original desc."
        assert len(log) == 1
        assert log[0]["title"] == "First"

    def test_hash_based_matching(self):
        """Descriptions matched by suggestion_hash are updated."""
        groups = [
            {
                "theme": "Test",
                "suggestions": [
                    {"title": "Hashed", "desc": "Old.", "suggestion_hash": "abc123def456"},
                ],
            },
        ]
        edited = {"abc123def456": ("Old.", "New.")}
        updated, log = merge_edited_descriptions(groups, edited)

        assert updated[0]["suggestions"][0]["desc"] == "New."
        assert len(log) == 1
        assert log[0]["id"] == "abc123def456"

    def test_hash_takes_precedence_over_positional(self):
        """Hash-based match is preferred over positional match."""
        groups = [
            {
                "theme": "Test",
                "suggestions": [
                    {"title": "Both", "desc": "Original.", "suggestion_hash": "hash123"},
                ],
            },
        ]
        # Both hash and positional keys present
        edited = {
            "hash123": ("Original.", "Hash edit."),
            "G1S1": ("Original.", "Positional edit."),
        }
        updated, log = merge_edited_descriptions(groups, edited)

        # Hash should win
        assert updated[0]["suggestions"][0]["desc"] == "Hash edit."

    def test_no_edits_returns_unchanged(self):
        """No matching edits returns groups unchanged."""
        groups = [
            {
                "theme": "Test",
                "suggestions": [
                    {"title": "Untouched", "desc": "Same."},
                ],
            },
        ]
        updated, log = merge_edited_descriptions(groups, {})

        assert updated[0]["suggestions"][0]["desc"] == "Same."
        assert len(log) == 0

    def test_original_groups_not_mutated(self):
        """Original groups list is not mutated."""
        groups = [
            {
                "theme": "Test",
                "suggestions": [
                    {"title": "Orig", "desc": "Unchanged."},
                ],
            },
        ]
        edited = {"G1S1": ("Unchanged.", "Changed.")}
        updated, log = merge_edited_descriptions(groups, edited)

        # Original should be untouched
        assert groups[0]["suggestions"][0]["desc"] == "Unchanged."
        # Updated copy should be changed
        assert updated[0]["suggestions"][0]["desc"] == "Changed."


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for derive_prefix, find_output_dir, load_json_file."""

    def test_derive_prefix_strips_md(self):
        """derive_prefix strips .md extension."""
        assert derive_prefix("plans/my-plan.md") == "my-plan"

    def test_derive_prefix_sanitizes_special_chars(self):
        """derive_prefix sanitizes special characters."""
        result = derive_prefix("plans/my plan (v2).md")
        assert " " not in result
        assert "(" not in result

    def test_find_output_dir(self):
        """find_output_dir returns correct path.

        Compared through Path() because find_output_dir uses os.path.join,
        which yields an OS-native separator (backslash on Windows).
        """
        result = find_output_dir("plans/my-plan.md")
        assert Path(result) == Path("plans/my-plan")

    def test_load_json_file_valid(self, tmp_path):
        """load_json_file loads valid JSON."""
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        result = load_json_file(str(f))
        assert result == {"key": "value"}

    def test_load_json_file_missing(self, tmp_path):
        """load_json_file returns None for missing file."""
        result = load_json_file(str(tmp_path / "missing.json"))
        assert result is None

    def test_load_json_file_invalid(self, tmp_path):
        """load_json_file returns None for invalid JSON."""
        f = tmp_path / "bad.json"
        f.write_text("not json{", encoding="utf-8")
        result = load_json_file(str(f))
        assert result is None


class TestFormatItemClaudeDecide:
    """format_item_for_output emits group_id and per-item decision_mode."""

    def _make_orchestrator(self):
        orch = ApplyTaskSuggestionsOrchestrator.__new__(ApplyTaskSuggestionsOrchestrator)
        orch.args = argparse.Namespace(plan_file="p.md")
        return orch

    def _group(self, **extra):
        g = {
            "group_hash": "task_g_hash",
            "theme": "Theme",
            "validation_status": "needs-human-decision",
            "suggestions": [{"title": "T", "desc": "d", "reference": "T001"}],
        }
        g.update(extra)
        return g

    def test_group_id_emitted_for_every_item(self):
        """group_id == group_hash on every human-review item (not just marked)."""
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(self._group(), 0)
        assert result["group_id"] == "task_g_hash"
        # Unmarked group carries no decision_mode.
        assert "decision_mode" not in result

    def test_decision_mode_for_claude_decide_group(self):
        """A claude_decide-marked group emits decision_mode == claude_auto_decide."""
        orch = self._make_orchestrator()
        result = orch.format_item_for_output(self._group(claude_decide=True), 0)
        assert result["group_id"] == "task_g_hash"
        assert result["decision_mode"] == "claude_auto_decide"


class TestOrchestratorOutputClaudeDecideShape:
    """The serialized orchestrator_output.json carries the new routing fields."""

    def _orch(self, tmp_path):
        orch = ApplyTaskSuggestionsOrchestrator.__new__(ApplyTaskSuggestionsOrchestrator)
        orch.args = argparse.Namespace(
            plan_file="p.md", batch_review_mode="by-importance", claude_decide=False
        )
        orch.out_dir = str(tmp_path)
        orch.plan_path = "p.md"
        orch.prefix = "feat"
        # build_output_json backs up the tasks file, so it must exist on disk.
        tasks_file = tmp_path / "tasks.md"
        tasks_file.write_text("# Tasks\n\n## T001: First task\nDo something.\n", encoding="utf-8")
        orch.tasks_file = str(tasks_file)
        orch.formatted_valid = []
        orch.formatted_human = []
        orch.formatted_skipped = []
        orch.user_skipped_items = []
        orch.skipped = []
        orch.valid = []
        orch.needs_human = []
        orch.merged = []
        orch.groups = []
        orch.batching_stats = {}
        orch.edit_log = []
        return orch

    def test_written_output_has_group_id_and_decision_mode(self, tmp_path):
        orch = self._orch(tmp_path)
        marked = orch.format_item_for_output(
            {"group_hash": "h_marked", "theme": "M",
             "validation_status": "needs-human-decision",
             "claude_decide": True,
             "suggestions": [{"title": "A", "desc": "a", "reference": "T001",
                              "importance": "HIGH"}]},
            0,
        )
        plain = orch.format_item_for_output(
            {"group_hash": "h_plain", "theme": "P",
             "validation_status": "needs-human-decision",
             "suggestions": [{"title": "B", "desc": "b", "reference": "T002",
                              "importance": "MEDIUM"}]},
            1,
        )
        orch.formatted_human = [marked, plain]

        output = orch.build_output_json([])
        # Serialize to disk and read back to assert the on-disk shape.
        out_path = tmp_path / "orchestrator_output.json"
        out_path.write_text(json.dumps(output), encoding="utf-8")
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))

        nhr = on_disk["needs_human_review"]
        # Every human-review item carries group_id == source group_hash.
        assert {it["group_id"] for it in nhr} == {"h_marked", "h_plain"}
        # Only the marked item carries the per-item routing flag.
        marked_items = [it for it in nhr if it.get("decision_mode") == "claude_auto_decide"]
        assert len(marked_items) == 1
        assert marked_items[0]["group_id"] == "h_marked"

        cfg = on_disk["human_review_config"]
        assert cfg["claude_decide_item_ids"] == ["h_marked"]
        # Per-item routing does not flip the global mode.
        assert cfg["decision_mode"] == "interactive"
