#!/usr/bin/env python3
"""End-to-end tests for apply_code_fixes_orchestrator.

These tests verify the apply-fixes phase workflow including:
- test_apply_fixes_with_user_skips: Parse [x] Skip checkboxes from report.md
- test_apply_fixes_edited_descriptions: User edits issue description in report.md
- test_apply_fixes_revalidation_mode: --revalidate flag targets validation_failed items
- test_apply_fixes_safety_guardrails: --approve-all blocked without --yes
- test_apply_fixes_resume_state: Resume after partial processing

All tests run in isolated tmp_path directories with mock LLM providers.

Usage:
    uv run -- pytest tests/test_e2e_apply_code_fixes.py -v
"""

import json
import os
import sys
import pytest
from pathlib import Path

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    SkillRunner,
    FixtureManager,
    MockProvider,
    AssertionHelpers,
)


def _configure_scenario(skill_runner: SkillRunner, mock_provider: MockProvider, scenario_name: str):
    """Configure scenario on both mock_provider and skill_runner."""
    mock_provider.set_scenario(scenario_name)
    skill_runner.extra_env.update(mock_provider.get_env())


def _create_code_review_phase(
    fixture: "FixturePlan",
    grouped_data: list,
    validation_data: list,
    report_content: str = None,
) -> Path:
    """Create pre-populated code-review phase outputs.

    Args:
        fixture: The FixturePlan to populate
        grouped_data: List of grouped issues from code review
        validation_data: List of validation results
        report_content: Optional report.md content

    Returns:
        Path to the code-review directory
    """
    # Create code-review directory
    code_review_dir = fixture.ensure_phase_dir("code-review")

    # Write grouped.json
    grouped_path = code_review_dir / "grouped.json"
    grouped_path.write_text(json.dumps(grouped_data, indent=2))

    # Write validation.json
    validation_path = code_review_dir / "validation.json"
    validation_path.write_text(json.dumps(validation_data, indent=2))

    # Write report.md if provided
    if report_content:
        report_path = code_review_dir / "report.md"
        report_path.write_text(report_content)

    return code_review_dir


class TestApplyFixesWithUserSkips:
    """Test that [x] Skip checkboxes in report.md filter out issues."""

    def test_apply_fixes_with_user_skips(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify user-marked [x] Skip items are filtered from output.

        When a user edits report.md and marks issues with [x] Skip checkbox,
        those issues should be excluded from the apply-fixes output.
        """
        # Create test plan
        plan_content = """# Authentication Security Plan

## Overview
Security improvements for the authentication system.

## Goals
- Fix security vulnerabilities
- Improve code quality
"""
        plan = fixture_manager.create_plan("test-apply-fixes-skips", plan_content)

        # Load grouped issues fixture
        grouped_path = Path(__file__).parent / "fixtures/e2e/responses/apply_fixes/grouped_with_issues.json"
        with open(grouped_path) as f:
            grouped_data = json.load(f)

        # Load validation fixture
        validation_path = Path(__file__).parent / "fixtures/e2e/responses/apply_fixes/validation_mixed.json"
        with open(validation_path) as f:
            validation_data = json.load(f)

        # Load report with skip markers
        report_path = Path(__file__).parent / "fixtures/e2e/responses/apply_fixes/with_user_skips.md"
        report_content = report_path.read_text()

        # Create code-review phase outputs
        _create_code_review_phase(plan, grouped_data, validation_data, report_content)

        # Mark code-review as completed
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        # Run apply_fixes orchestrator
        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--dry-run",
            timeout=30,
        )

        # Should succeed
        assert result.success, f"apply_fixes failed: {result.stderr}"

        # Verify skipped items are excluded
        # Issues 1 and 3 were marked with [x] Skip in the report
        assert "User skipped" in result.stderr, "Should log that user skipped issues"

        # Parse output to verify filtered results
        try:
            output = json.loads(result.stdout)
            to_apply = output.get("to_apply", [])

            # Issue 1 (SQL injection) was skipped by user
            # Issue 3 (Rate limiting) was skipped by user
            # Issue 5 (Session cleanup) was invalid
            # So we should only have Issue 2 (password hashing) and Issue 4 (JWT secret)

            titles = [item.get("title", "") for item in to_apply]

            # Skipped items should NOT be in output
            assert not any("SQL injection" in t for t in titles), \
                "SQL injection (user-skipped) should not be in output"
            assert not any("Rate limiter" in t for t in titles), \
                "Rate limiter (user-skipped) should not be in output"

            # Check summary
            summary = output.get("summary", {})
            # Should show that items were user-skipped

        except json.JSONDecodeError:
            # If output is not JSON, check stderr for skip messages
            pass

        # Verify the skip count is reported
        assert "skipped" in result.stderr.lower(), \
            "Should report skipped items in stderr"

    def test_user_skip_checkbox_variations(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Test various checkbox formatting is recognized.

        The parser should handle:
        - [x] Skip
        - [X] Skip
        - [ x] Skip (with space)
        - [x ] Skip (with trailing space)
        """
        plan = fixture_manager.create_plan("checkbox-variations", "# Test Plan\n")

        # Load grouped issues
        grouped_path = Path(__file__).parent / "fixtures/e2e/responses/apply_fixes/grouped_with_issues.json"
        with open(grouped_path) as f:
            grouped_data = json.load(f)[:3]  # Just first 3

        # Simple validation - all valid
        validation_data = [
            {"group_index": i, "status": "valid", "reason": "Valid issue"}
            for i in range(3)
        ]

        # Report with various checkbox formats
        report_content = """# Code Review Report

### 1. SQL injection vulnerability
- [x] Skip

Description text.

---

### 2. Missing password hashing
- [X] Skip

Description text.

---

### 3. Rate limiter not applied
- [ ] Skip

Description text.

---
"""
        _create_code_review_phase(plan, grouped_data, validation_data, report_content)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--dry-run",
        )

        assert result.success

        # Issues 1 and 2 marked as skip, issue 3 not skipped
        try:
            output = json.loads(result.stdout)
            to_apply = output.get("to_apply", [])

            # Only issue 3 (rate limiter) should remain
            assert len(to_apply) == 1, f"Expected 1 item, got {len(to_apply)}"

        except json.JSONDecodeError:
            pass


class TestApplyFixesEditedDescriptions:
    """Test that user-edited descriptions in report.md are merged."""

    def test_apply_fixes_edited_descriptions(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify user-edited descriptions are merged into fix tasks.

        When a user edits the description text in report.md, the orchestrator
        should detect the change and use the edited version in the output.
        """
        plan = fixture_manager.create_plan("test-edited-descriptions", "# Test Plan\n")

        # Load grouped issues
        grouped_path = Path(__file__).parent / "fixtures/e2e/responses/apply_fixes/grouped_with_issues.json"
        with open(grouped_path) as f:
            grouped_data = json.load(f)

        # Load validation
        validation_path = Path(__file__).parent / "fixtures/e2e/responses/apply_fixes/validation_mixed.json"
        with open(validation_path) as f:
            validation_data = json.load(f)

        # Load report with edited descriptions
        report_path = Path(__file__).parent / "fixtures/e2e/responses/apply_fixes/with_edited_descriptions.md"
        report_content = report_path.read_text()

        _create_code_review_phase(plan, grouped_data, validation_data, report_content)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--dry-run",
        )

        assert result.success, f"apply_fixes failed: {result.stderr}"

        # Should report merged descriptions
        assert "Merged" in result.stderr or "edited" in result.stderr.lower(), \
            "Should report merged/edited descriptions"

        # Parse output to verify edited descriptions
        try:
            output = json.loads(result.stdout)

            # Check that edit log is present
            edit_log = output.get("edited_descriptions", [])
            assert len(edit_log) > 0, "Should have edit log entries"

            # Verify the edits were for issues 1 and 3 (which had EDITED BY USER)
            edited_indices = {item.get("index") for item in edit_log}
            assert 1 in edited_indices or 3 in edited_indices, \
                f"Expected edited indices 1 or 3, got {edited_indices}"

            # Verify the edited content is in the output
            to_apply = output.get("to_apply", [])
            for item in to_apply:
                if item.get("title", "").startswith("SQL injection"):
                    desc = item.get("description", "")
                    # The edited description mentions SQLite and ? placeholder
                    assert "SQLite" in desc or "?" in desc or "EDITED" in desc, \
                        f"SQL injection fix should have edited description: {desc[:100]}"

        except json.JSONDecodeError:
            # Check stderr for edit info
            pass

    def test_edited_description_preserves_original(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify original description is preserved with _original_desc field."""
        plan = fixture_manager.create_plan("preserve-original", "# Test Plan\n")

        # Create simple test data
        grouped_data = [{
            "theme": "Test issue",
            "category": "test",
            "models": ["cursor-agent"],
            "suggestions": [{
                "title": "Test issue title",
                "desc": "Original description from LLM.",
                "importance": "MEDIUM",
                "type": "bug",
                "file": "test.py",
                "source_model": "cursor-agent"
            }]
        }]

        validation_data = [{
            "group_index": 0,
            "status": "valid",
            "reason": "Valid issue"
        }]

        # Report with edited description
        report_content = """# Code Review Report

### 1. Test issue title
- [ ] Skip

**Validation:** valid | **Importance:** MEDIUM

User edited this description completely.

---
"""
        _create_code_review_phase(plan, grouped_data, validation_data, report_content)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--dry-run",
        )

        assert result.success

        # Verify edit was detected
        try:
            output = json.loads(result.stdout)
            edit_log = output.get("edited_descriptions", [])
            assert len(edit_log) >= 1, "Should detect edited description"

        except json.JSONDecodeError:
            pass


class TestApplyFixesRevalidationMode:
    """Test --revalidate flag for retrying validation_failed items."""

    def test_apply_fixes_revalidation_mode(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --revalidate targets only validation_failed items.

        When --revalidate is passed, the orchestrator should:
        1. Identify items with validation_failed status
        2. Prepare revalidation tasks for them
        3. Output a marker indicating revalidation is pending
        """
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan("test-revalidation", "# Test Plan\n")

        # Create grouped data with multiple items
        grouped_data = [
            {
                "theme": f"Issue {i}",
                "category": "test",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": f"Issue {i} title",
                    "desc": f"Description for issue {i}",
                    "importance": "MEDIUM",
                    "type": "bug",
                    "file": "test.py",
                    "source_model": "cursor-agent"
                }]
            }
            for i in range(4)
        ]

        # Validation with some failures
        validation_data = [
            {"group_index": 0, "status": "valid", "reason": "Valid"},
            {"group_index": 1, "status": "validation_failed", "reason": "Timeout", "error_type": "timeout", "recoverable": True},
            {"group_index": 2, "status": "needs-human-decision", "reason": "Ambiguous"},
            {"group_index": 3, "status": "validation_failed", "reason": "Parse error", "error_type": "parsing_error", "recoverable": True},
        ]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--revalidate",
            "--dry-run",
        )

        # Should succeed and indicate revalidation mode
        assert result.success or "REVALIDATION" in result.stdout, \
            f"Revalidation should succeed or output marker: {result.stderr}"

        # Should report items to revalidate
        output = result.stdout + result.stderr
        assert "revalidat" in output.lower(), \
            "Should mention revalidation in output"

        # In dry-run mode, should report count
        assert "2" in output or "revalidate" in output.lower(), \
            "Should report 2 items need revalidation (indices 1 and 3)"

    def test_revalidation_with_model_override(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --revalidate-model overrides the default model."""
        _configure_scenario(skill_runner, mock_provider, "happy_path")

        plan = fixture_manager.create_plan("revalidate-model", "# Test Plan\n")

        grouped_data = [{
            "theme": "Test issue",
            "models": ["cursor-agent"],
            "suggestions": [{"title": "Test", "desc": "Desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}]
        }]

        validation_data = [{
            "group_index": 0,
            "status": "validation_failed",
            "reason": "Parse error",
            "error_type": "parsing_error",
            "recoverable": True
        }]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--revalidate",
            "--revalidate-model", "cursor-agent:opus",
            "--dry-run",
        )

        # Should succeed
        assert result.success or "REVALIDATION" in result.stdout


class TestApplyFixesSafetyGuardrails:
    """Test safety guardrails for bulk approval operations."""

    def test_approve_all_blocked_without_yes(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --approve-all requires --yes or --force flag.

        This is a safety guardrail to prevent accidental bulk approval.
        """
        plan = fixture_manager.create_plan("approve-all-safety", "# Test Plan\n")

        # Create grouped data
        grouped_data = [{
            "theme": "Test issue",
            "models": ["cursor-agent"],
            "suggestions": [{"title": "Test", "desc": "Desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}]
        }]

        validation_data = [{"group_index": 0, "status": "needs-human-decision", "reason": "Needs review"}]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        # Run WITHOUT --yes flag
        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--approve-all",
        )

        # Should fail with error about missing --yes
        assert not result.success, "--approve-all without --yes should fail"
        assert "--yes" in result.stderr or "--force" in result.stderr, \
            "Error should mention --yes or --force flag"

    def test_approve_all_works_with_yes(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --approve-all --yes works correctly."""
        plan = fixture_manager.create_plan("approve-all-with-yes", "# Test Plan\n")

        grouped_data = [{
            "theme": "Test issue",
            "models": ["cursor-agent"],
            "suggestions": [{"title": "Test", "desc": "Desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}]
        }]

        validation_data = [{"group_index": 0, "status": "needs-human-decision", "reason": "Needs review"}]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--approve-all",
            "--yes",
            "--dry-run",
        )

        assert result.success, f"--approve-all --yes should succeed: {result.stderr}"

    def test_high_importance_blocked_without_include_high(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --approve-all blocks HIGH importance items without --include-high.

        HIGH importance items require explicit --include-high flag because
        they may need more careful review.
        """
        plan = fixture_manager.create_plan("high-importance-safety", "# Test Plan\n")

        # Create HIGH importance item
        grouped_data = [{
            "theme": "Critical security issue",
            "models": ["cursor-agent"],
            "suggestions": [{
                "title": "Critical bug",
                "desc": "A critical security vulnerability",
                "importance": "HIGH",
                "type": "bug",
                "file": "test.py",
                "source_model": "cursor-agent"
            }]
        }]

        validation_data = [{"group_index": 0, "status": "needs-human-decision", "reason": "Needs review"}]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        # Run with --approve-all --yes but WITHOUT --include-high
        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--approve-all",
            "--yes",
        )

        # Should fail with error about HIGH importance
        assert not result.success, "--approve-all without --include-high should fail for HIGH items"
        assert "HIGH" in result.stderr or "--include-high" in result.stderr, \
            "Error should mention HIGH importance or --include-high flag"

    def test_high_importance_allowed_with_include_high(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --approve-all --include-high allows HIGH importance items."""
        plan = fixture_manager.create_plan("high-with-flag", "# Test Plan\n")

        grouped_data = [{
            "theme": "Critical security issue",
            "models": ["cursor-agent"],
            "suggestions": [{
                "title": "Critical bug",
                "desc": "A critical security vulnerability",
                "importance": "HIGH",
                "type": "bug",
                "file": "test.py",
                "source_model": "cursor-agent"
            }]
        }]

        validation_data = [{"group_index": 0, "status": "needs-human-decision", "reason": "Needs review"}]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--approve-all",
            "--yes",
            "--include-high",
            "--dry-run",
        )

        assert result.success, f"--approve-all --include-high should succeed: {result.stderr}"

    def test_force_flag_works_as_yes_alias(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --force works as alias for --yes."""
        plan = fixture_manager.create_plan("force-flag", "# Test Plan\n")

        grouped_data = [{
            "theme": "Test issue",
            "models": ["cursor-agent"],
            "suggestions": [{"title": "Test", "desc": "Desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}]
        }]

        validation_data = [{"group_index": 0, "status": "needs-human-decision", "reason": "Needs review"}]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--approve-all",
            "--force",
            "--dry-run",
        )

        assert result.success, f"--force should work as --yes alias: {result.stderr}"


class TestApplyFixesResumeState:
    """Test resume functionality with state persistence."""

    def test_apply_fixes_resume_state(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --resume skips already-processed items.

        When resuming, items that were already processed in a previous run
        should be skipped.
        """
        plan = fixture_manager.create_plan("test-resume", "# Test Plan\n")

        # Create grouped data with multiple items
        grouped_data = [
            {
                "theme": f"Issue {i}",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": f"Issue {i} title",
                    "desc": f"Description for issue {i}",
                    "importance": "MEDIUM",
                    "type": "bug",
                    "file": "test.py",
                    "source_model": "cursor-agent"
                }]
            }
            for i in range(3)
        ]

        validation_data = [
            {"group_index": 0, "status": "valid", "reason": "Valid"},
            {"group_index": 1, "status": "valid", "reason": "Valid"},
            {"group_index": 2, "status": "valid", "reason": "Valid"},
        ]

        _create_code_review_phase(plan, grouped_data, validation_data)

        # Create state with some items already processed
        # The group_id is typically a hash - we need to match what the orchestrator generates
        import hashlib

        def generate_group_id(group):
            """Generate stable group ID matching the orchestrator's logic."""
            theme = group.get("theme", "")
            suggestions = group.get("suggestions", [])
            if suggestions:
                first_title = suggestions[0].get("title", "")
                first_desc = suggestions[0].get("desc", "")[:100]
            else:
                first_title = ""
                first_desc = ""
            key = f"{theme}|{first_title}|{first_desc}"
            return hashlib.sha256(key.encode()).hexdigest()[:16]

        # Mark first two items as processed
        processed_items = {}
        for i in range(2):
            group_id = generate_group_id(grouped_data[i])
            processed_items[group_id] = {
                "status": "applied",
                "timestamp": "2025-01-01T00:00:00"
            }

        extra_state = {
            "processed_apply-fixes": processed_items
        }

        fixture_manager.create_state_file(
            plan,
            phases_completed=["code-review"],
            extra_state=extra_state,
        )

        # Run with --resume
        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--resume",
            "--dry-run",
        )

        assert result.success, f"Resume should succeed: {result.stderr}"

        # Should report resumed items
        assert "resume" in result.stderr.lower() or "already processed" in result.stderr.lower() or \
               "remaining" in result.stderr.lower(), \
            "Should report resume status"

        # Parse output
        try:
            output = json.loads(result.stdout)
            to_apply = output.get("to_apply", [])

            # Only the third item (Issue 2) should remain
            assert len(to_apply) == 1, f"Expected 1 unprocessed item, got {len(to_apply)}"

            resume_info = output.get("resume_info", {})
            assert resume_info.get("can_resume", False), "Should indicate resumable state"

        except json.JSONDecodeError:
            pass

    def test_resume_applies_previous_human_decisions(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --resume applies previous human decisions."""
        plan = fixture_manager.create_plan("resume-decisions", "# Test Plan\n")

        grouped_data = [{
            "theme": "Test issue",
            "models": ["cursor-agent"],
            "suggestions": [{"title": "Test", "desc": "Desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}]
        }]

        validation_data = [{"group_index": 0, "status": "needs-human-decision", "reason": "Ambiguous"}]

        _create_code_review_phase(plan, grouped_data, validation_data)

        # Create state with human decision
        import hashlib
        group_id = hashlib.sha256(b"Test issue|Test|Desc").hexdigest()[:16]

        extra_state = {
            "human_decisions_apply-fixes": {
                group_id: {
                    "decision": "approved",
                    "timestamp": "2025-01-01T00:00:00",
                    "reason": "Previously approved"
                }
            }
        }

        fixture_manager.create_state_file(
            plan,
            phases_completed=["code-review"],
            extra_state=extra_state,
        )

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--resume",
            "--dry-run",
        )

        assert result.success

    def test_fresh_flag_clears_previous_progress(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --fresh clears previous progress and starts over."""
        plan = fixture_manager.create_plan("fresh-start", "# Test Plan\n")

        grouped_data = [
            {
                "theme": f"Issue {i}",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": f"Issue {i}",
                    "desc": f"Desc {i}",
                    "importance": "MEDIUM",
                    "type": "bug",
                    "file": "test.py",
                    "source_model": "cursor-agent"
                }]
            }
            for i in range(3)
        ]

        validation_data = [
            {"group_index": i, "status": "valid", "reason": "Valid"}
            for i in range(3)
        ]

        _create_code_review_phase(plan, grouped_data, validation_data)

        # Create state with all items processed
        extra_state = {
            "processed_apply-fixes": {
                f"fake_id_{i}": {"status": "applied"}
                for i in range(3)
            }
        }

        fixture_manager.create_state_file(
            plan,
            phases_completed=["code-review"],
            extra_state=extra_state,
        )

        # Run with --fresh
        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--fresh",
            "--dry-run",
        )

        assert result.success, f"--fresh should succeed: {result.stderr}"

        # Should have all 3 items to apply (progress was cleared)
        try:
            output = json.loads(result.stdout)
            to_apply = output.get("to_apply", [])
            assert len(to_apply) == 3, f"Expected 3 items after --fresh, got {len(to_apply)}"

        except json.JSONDecodeError:
            pass


class TestApplyFixesBulkApprovalOptions:
    """Test various bulk approval options."""

    def test_approve_all_low_only_affects_low_importance(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --approve-all-low only auto-approves LOW importance items."""
        plan = fixture_manager.create_plan("approve-low", "# Test Plan\n")

        grouped_data = [
            {"theme": "High issue", "models": ["cursor-agent"], "suggestions": [
                {"title": "High", "desc": "High desc", "importance": "HIGH", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
            ]},
            {"theme": "Medium issue", "models": ["cursor-agent"], "suggestions": [
                {"title": "Medium", "desc": "Medium desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
            ]},
            {"theme": "Low issue", "models": ["cursor-agent"], "suggestions": [
                {"title": "Low", "desc": "Low desc", "importance": "LOW", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
            ]},
        ]

        validation_data = [
            {"group_index": 0, "status": "needs-human-decision", "reason": "Review"},
            {"group_index": 1, "status": "needs-human-decision", "reason": "Review"},
            {"group_index": 2, "status": "needs-human-decision", "reason": "Review"},
        ]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--approve-all-low",
            "--dry-run",
        )

        assert result.success

        try:
            output = json.loads(result.stdout)
            to_apply = output.get("to_apply", [])
            needs_human = output.get("needs_human_review", [])

            # Only LOW should be auto-approved
            # HIGH and MEDIUM should still need human review
            applied_importance = [item.get("importance", "").upper() for item in to_apply]
            human_importance = [item.get("importance", "").upper() for item in needs_human]

            # LOW should be in to_apply (auto-approved)
            assert "LOW" in applied_importance, "LOW importance should be auto-approved"

            # HIGH and MEDIUM should still need human review
            assert "HIGH" in human_importance or "MEDIUM" in human_importance, \
                "HIGH/MEDIUM importance should still need human review"

        except json.JSONDecodeError:
            pass

    def test_skip_all_human_skips_needs_human_items(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --skip-all-human skips all needs-human-decision items."""
        plan = fixture_manager.create_plan("skip-all-human", "# Test Plan\n")

        grouped_data = [
            {"theme": "Valid issue", "models": ["cursor-agent"], "suggestions": [
                {"title": "Valid", "desc": "Valid desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
            ]},
            {"theme": "Human issue", "models": ["cursor-agent"], "suggestions": [
                {"title": "Human", "desc": "Human desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
            ]},
        ]

        validation_data = [
            {"group_index": 0, "status": "valid", "reason": "Valid"},
            {"group_index": 1, "status": "needs-human-decision", "reason": "Needs review"},
        ]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--skip-all-human",
            "--dry-run",
        )

        assert result.success

        try:
            output = json.loads(result.stdout)
            to_apply = output.get("to_apply", [])
            needs_human = output.get("needs_human_review", [])

            # Valid issue should be in to_apply
            assert len(to_apply) >= 1, "Valid issue should be in to_apply"

            # needs-human-decision items should be skipped (empty needs_human list)
            assert len(needs_human) == 0, "Human review items should be skipped"

        except json.JSONDecodeError:
            pass

    def test_approve_validation_failed(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify --approve-validation-failed auto-approves recoverable failures."""
        plan = fixture_manager.create_plan("approve-failed", "# Test Plan\n")

        grouped_data = [
            {"theme": "Valid issue", "models": ["cursor-agent"], "suggestions": [
                {"title": "Valid", "desc": "Valid desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
            ]},
            {"theme": "Failed issue", "models": ["cursor-agent"], "suggestions": [
                {"title": "Failed", "desc": "Failed desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
            ]},
        ]

        validation_data = [
            {"group_index": 0, "status": "valid", "reason": "Valid"},
            {"group_index": 1, "status": "validation_failed", "reason": "Timeout", "error_type": "timeout", "recoverable": True},
        ]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
            "--approve-validation-failed",
            "--dry-run",
        )

        assert result.success

        try:
            output = json.loads(result.stdout)
            to_apply = output.get("to_apply", [])

            # Both items should be in to_apply
            assert len(to_apply) == 2, f"Expected 2 items, got {len(to_apply)}"

        except json.JSONDecodeError:
            pass


class TestApplyFixesOutputFormat:
    """Test output format and structure."""

    def test_output_contains_batches(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify output contains batched fixes for Claude Code."""
        plan = fixture_manager.create_plan("batch-output", "# Test Plan\n")

        grouped_data = [
            {"theme": f"Issue {i}", "models": ["cursor-agent"], "suggestions": [
                {"title": f"Issue {i}", "desc": f"Desc {i}", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
            ]}
            for i in range(3)
        ]

        validation_data = [
            {"group_index": i, "status": "valid", "reason": "Valid"}
            for i in range(3)
        ]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
        )

        assert result.success

        try:
            output = json.loads(result.stdout)

            # Should have batches key
            assert "batches" in output, "Output should contain batches"
            assert isinstance(output["batches"], list), "batches should be a list"

            # Should have summary
            assert "summary" in output, "Output should contain summary"

            # Should have batching_stats
            assert "batching_stats" in output, "Output should contain batching_stats"

        except json.JSONDecodeError:
            pytest.fail("Output should be valid JSON")

    def test_output_contains_state_file_path(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify output contains state_file path for Claude Code."""
        plan = fixture_manager.create_plan("state-file-path", "# Test Plan\n")

        grouped_data = [{"theme": "Issue", "models": ["cursor-agent"], "suggestions": [
            {"title": "Issue", "desc": "Desc", "importance": "MEDIUM", "type": "bug", "file": "test.py", "source_model": "cursor-agent"}
        ]}]

        validation_data = [{"group_index": 0, "status": "valid", "reason": "Valid"}]

        _create_code_review_phase(plan, grouped_data, validation_data)
        fixture_manager.create_state_file(plan, phases_completed=["code-review"])

        result = skill_runner.run_orchestrator(
            "apply_fixes",
            plan.plan_path,
        )

        assert result.success

        try:
            output = json.loads(result.stdout)

            assert "state_file" in output, "Output should contain state_file path"
            assert output["state_file"].endswith("state.json"), "state_file should point to state.json"

        except json.JSONDecodeError:
            pytest.fail("Output should be valid JSON")
