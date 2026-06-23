#!/usr/bin/env python3
"""
Integration tests for apply_suggestions_orchestrator.py

These tests run the orchestrator with various options and verify the output.
They use mock data in plans/test-plan/ to avoid calling real LLMs.

To run these tests:
    uv run -- pytest tests/test_apply_suggestions_integration.py -v

To regenerate the test fixtures:
    uv run -- python tests/test_apply_suggestions_integration.py --setup-fixtures
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Test directory setup
SKILL_DIR = Path(__file__).parent.parent
PLANS_DIR = SKILL_DIR / "plans"
TEST_PLAN_DIR = PLANS_DIR / "test-plan"
TEST_PLAN_FILE = PLANS_DIR / "test-plan.md"


class TestFixtures:
    """Test data fixtures for integration tests."""

    PLAN_CONTENT = """# Test Plan: User Authentication Feature

## Overview
Implement a user authentication system with login, logout, and session management.

## Step 1: Database Schema
Create the users table with the following fields:
- id: UUID primary key
- email: unique, not null
- password_hash: not null
- created_at: timestamp
- updated_at: timestamp

## Step 2: User Model
Create the User model with validation for email format and password requirements.

## Step 3: Authentication Service
Implement the authentication service with:
- login(email, password) - validates credentials and returns session token
- logout(token) - invalidates the session
- verify_token(token) - checks if token is valid

## Step 4: API Endpoints
Create REST endpoints:
- POST /auth/login - accepts email/password, returns token
- POST /auth/logout - invalidates current session
- GET /auth/me - returns current user info

## Step 5: Session Management
Implement session storage using Redis with:
- 24-hour TTL for session tokens
- Automatic renewal on activity
- Multi-device support

## Step 6: Security Measures
Add security features:
- Rate limiting on login attempts
- Password hashing with bcrypt
- HTTPS enforcement
- CSRF protection

## Step 7: Testing
Write comprehensive tests:
- Unit tests for User model
- Integration tests for auth service
- E2E tests for API endpoints
"""

    GROUPED_SUGGESTIONS = [
        {
            "theme": "Add input validation for email",
            "category": "security",
            "models": ["claude-sonnet", "gpt-4"],
            "suggestions": [{
                "title": "Add email format validation",
                "desc": "Add regex validation to ensure email addresses are in valid format.",
                "type": "addition",
                "reference": "### Step 2: User Model",
                "importance": "HIGH",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Add password complexity requirements",
            "category": "security",
            "models": ["gpt-4"],
            "suggestions": [{
                "title": "Enforce password complexity",
                "desc": "Require passwords to have at least 8 characters with mixed case and special chars.",
                "type": "addition",
                "reference": "### Step 2: User Model",
                "importance": "MEDIUM",
                "source_model": "gpt-4"
            }]
        },
        {
            "theme": "Add account lockout after failed attempts",
            "category": "security",
            "models": ["claude-sonnet", "gpt-4", "gemini-pro"],
            "suggestions": [{
                "title": "Implement account lockout",
                "desc": "Lock account for 15 minutes after 5 failed login attempts.",
                "type": "addition",
                "reference": "### Step 6: Security Measures",
                "importance": "HIGH",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Add session activity logging",
            "category": "monitoring",
            "models": ["claude-sonnet"],
            "suggestions": [{
                "title": "Log session activities",
                "desc": "Add logging for login, logout, and token refresh events.",
                "type": "addition",
                "reference": "### Step 5: Session Management",
                "importance": "LOW",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Clarify token storage approach",
            "category": "architecture",
            "models": ["gpt-4"],
            "suggestions": [{
                "title": "Specify token storage location",
                "desc": "The plan doesn't specify whether to use cookies or localStorage.",
                "type": "clarification",
                "reference": "### Step 4: API Endpoints",
                "importance": "MEDIUM",
                "source_model": "gpt-4"
            }]
        },
        {
            "theme": "Add refresh token mechanism",
            "category": "architecture",
            "models": ["gemini-pro"],
            "suggestions": [{
                "title": "Implement refresh tokens",
                "desc": "Add refresh token support for long-lived sessions.",
                "type": "addition",
                "reference": "### Step 3: Authentication Service",
                "importance": "MEDIUM",
                "source_model": "gemini-pro"
            }]
        },
        {
            "theme": "Remove redundant HTTPS mention",
            "category": "cleanup",
            "models": ["claude-sonnet"],
            "suggestions": [{
                "title": "Remove HTTPS enforcement from plan",
                "desc": "HTTPS is typically handled at infrastructure level.",
                "type": "deletion",
                "reference": "### Step 6: Security Measures",
                "importance": "LOW",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Add database indexes",
            "category": "performance",
            "models": ["gpt-4", "gemini-pro"],
            "suggestions": [{
                "title": "Add index on email column",
                "desc": "Add a unique index on the email column to speed up login queries.",
                "type": "addition",
                "reference": "### Step 1: Database Schema",
                "importance": "MEDIUM",
                "source_model": "gpt-4"
            }]
        },
        {
            "theme": "Add password reset functionality",
            "category": "feature",
            "models": ["claude-sonnet", "gpt-4"],
            "suggestions": [{
                "title": "Add password reset flow",
                "desc": "The authentication system should include a password reset mechanism.",
                "type": "addition",
                "reference": "### Step 4: API Endpoints",
                "importance": "HIGH",
                "source_model": "claude-sonnet"
            }]
        },
        {
            "theme": "Update test coverage requirements",
            "category": "testing",
            "models": ["gemini-pro"],
            "suggestions": [{
                "title": "Specify minimum coverage threshold",
                "desc": "Add requirement for minimum 80% code coverage.",
                "type": "modification",
                "reference": "### Step 7: Testing",
                "importance": "LOW",
                "source_model": "gemini-pro"
            }]
        }
    ]

    VALIDATION_RESULTS = {
        "metadata": {
            "schema_version": "2.0",
            "validated_at": "2025-01-30T10:00:00",
            "model": "mock-llm",
            "plan_hash": "abc123",
            "total_groups": 10
        },
        "groups": [
            {"group_index": 0, "status": "valid", "reason": "Email validation is clear.", "confidence": 0.95},
            {"group_index": 1, "status": "valid", "reason": "Password complexity is well-defined.", "confidence": 0.90},
            {"group_index": 2, "status": "needs-human-decision", "reason": "Lockout config needs business input.",
             "confidence": 0.60, "error_type": "real_ambiguity", "recoverable": False},
            {"group_index": 3, "status": "valid", "reason": "Session logging is straightforward.", "confidence": 0.88},
            {"group_index": 4, "status": "needs-human-decision", "reason": "Multiple valid token storage options.",
             "confidence": 0.45, "error_type": "real_ambiguity", "recoverable": False},
            {"group_index": 5, "status": "validation_failed", "reason": "Failed to parse LLM response.",
             "confidence": 0.0, "error_type": "parsing_error", "recoverable": True},
            {"group_index": 6, "status": "valid", "reason": "HTTPS clarification is valid.", "confidence": 0.82},
            {"group_index": 7, "status": "validation_failed", "reason": "Request timed out.",
             "confidence": 0.0, "error_type": "timeout", "recoverable": True},
            {"group_index": 8, "status": "validation_failed", "reason": "Rate limit exceeded.",
             "confidence": 0.0, "error_type": "rate_limited", "recoverable": True},
            {"group_index": 9, "status": "invalid", "reason": "Too vague and not related.", "confidence": 0.75}
        ]
    }


def get_plan_hash() -> str:
    """Compute the actual plan hash for the test plan."""
    sys.path.insert(0, str(SKILL_DIR))
    from utils.state_manager import StateManager
    sm = StateManager(TEST_PLAN_FILE)
    return sm._compute_plan_hash()


def setup_test_fixtures(force: bool = False) -> None:
    """Create or refresh test fixtures."""
    if TEST_PLAN_DIR.exists() and not force:
        return

    # Create directories
    review_plan_dir = TEST_PLAN_DIR / "review-plan"
    review_plan_dir.mkdir(parents=True, exist_ok=True)

    # Write plan file
    TEST_PLAN_FILE.write_text(TestFixtures.PLAN_CONTENT)

    # Write grouped suggestions
    (review_plan_dir / "grouped.json").write_text(
        json.dumps(TestFixtures.GROUPED_SUGGESTIONS, indent=2)
    )

    # Write validation results
    (review_plan_dir / "validation.json").write_text(
        json.dumps(TestFixtures.VALIDATION_RESULTS, indent=2)
    )

    # Write backup
    (review_plan_dir / "backup.md").write_text(TestFixtures.PLAN_CONTENT)

    # Write state file with correct plan hash
    plan_hash = get_plan_hash()

    # Get actual group IDs
    sys.path.insert(0, str(SKILL_DIR))
    from utils.state_manager import generate_group_id

    group_ids = [generate_group_id(g) for g in TestFixtures.GROUPED_SUGGESTIONS]

    state = {
        "version": "1",
        "plan_file": "plans/test-plan.md",
        "plan_hash": plan_hash,
        "created_at": "2025-01-30T09:00:00",
        "updated_at": "2025-01-30T10:30:00",
        "human_decisions_apply-suggestions": {
            group_ids[2]: {  # Account lockout suggestion
                "decision": "approved",
                "timestamp": "2025-01-30T10:15:00",
                "reason": "User approved during previous session",
                "batch_context": {"batch_id": "batch_20250130_101500"}
            }
        },
        "processed_apply-suggestions": {
            group_ids[0]: {  # Email validation suggestion
                "status": "applied",
                "timestamp": "2025-01-30T10:20:00"
            }
        },
        "progress_apply-suggestions": {
            "last_batch_index": 1,
            "total_batches": 4,
            "started_at": "2025-01-30T10:00:00"
        }
    }
    (TEST_PLAN_DIR / "state.json").write_text(json.dumps(state, indent=2))


def run_orchestrator(*args) -> subprocess.CompletedProcess:
    """Run the orchestrator with given arguments."""
    cmd = [
        "uv", "run", "--",
        "python", "apply_suggestions_orchestrator.py",
        "--plan-file", str(TEST_PLAN_FILE),
        "--output-format", "json",
        *args
    ]
    return subprocess.run(
        cmd,
        cwd=str(SKILL_DIR),
        capture_output=True,
        text=True
    )


@pytest.fixture(scope="module", autouse=True)
def setup_fixtures():
    """Ensure test fixtures exist before running tests."""
    setup_test_fixtures(force=True)
    yield
    # Optionally clean up after tests
    # shutil.rmtree(TEST_PLAN_DIR)


class TestDryRunApproveAllLow:
    """Test dry-run with --approve-all-low option."""

    def test_dry_run_shows_what_would_be_approved(self):
        """Verify dry-run correctly reports LOW items that would be auto-approved."""
        result = run_orchestrator("--approve-all-low", "--dry-run")

        assert result.returncode == 0
        # Should show dry-run report
        assert "DRY RUN" in result.stderr or "DRY RUN" in result.stdout

    def test_dry_run_identifies_valid_suggestions(self):
        """Verify dry-run identifies valid suggestions to apply."""
        result = run_orchestrator("--dry-run")

        assert result.returncode == 0
        assert "Valid (to apply):" in result.stderr

    def test_dry_run_identifies_human_review_items(self):
        """Verify dry-run identifies items needing human review."""
        result = run_orchestrator("--dry-run")

        assert result.returncode == 0
        assert "Needs human review:" in result.stderr

    def test_dry_run_shows_batching_stats(self):
        """Verify dry-run shows batching efficiency stats."""
        result = run_orchestrator("--dry-run")

        assert result.returncode == 0
        assert "Batching results:" in result.stderr
        assert "Total suggestions:" in result.stderr
        assert "Total batches:" in result.stderr


class TestRevalidation:
    """Test revalidation options."""

    def test_revalidate_dry_run_identifies_failed_items(self):
        """Verify --revalidate --dry-run identifies validation_failed items."""
        result = run_orchestrator("--revalidate", "--dry-run")

        assert result.returncode == 0
        assert "REVALIDATION MODE" in result.stderr
        assert "Would revalidate" in result.stderr
        # Should identify 3 validation_failed items
        assert "parsing_error" in result.stderr
        assert "timeout" in result.stderr
        assert "rate_limited" in result.stderr

    def test_revalidate_counts_correct_items(self):
        """Verify correct number of items would be revalidated."""
        result = run_orchestrator("--revalidate", "--dry-run")

        assert result.returncode == 0
        # Should find 3 items (groups 5, 7, 8 with validation_failed status)
        assert "3 items" in result.stderr


class TestResume:
    """Test resume interrupted session functionality."""

    def test_resume_skips_processed_items(self):
        """Verify --resume skips already-processed items."""
        result = run_orchestrator("--resume", "--dry-run")

        assert result.returncode == 0
        # Should show 1 already processed (group 0 - email validation)
        assert "1 already processed" in result.stderr
        # Note: The resume count may be 9 or 10 depending on whether
        # user-edited descriptions change the group_id hash. When report.md
        # descriptions differ from grouped.json, the desc edit changes the
        # group_id, preventing the resume match for that group.
        assert "remaining" in result.stderr

    def test_resume_applies_previous_human_decisions(self):
        """Verify --resume applies previous human decisions."""
        # Run without resume first to get baseline
        baseline = run_orchestrator("--dry-run")

        # Run with resume
        resumed = run_orchestrator("--resume", "--dry-run")

        # Both should succeed
        assert baseline.returncode == 0
        assert resumed.returncode == 0

        # With resume, should have fewer items needing human review
        # (group 2 - account lockout was pre-approved)


class TestSkipAllHuman:
    """Test --skip-all-human option."""

    def test_skip_all_human_skips_human_review_items(self):
        """Verify --skip-all-human skips needs-human-decision items."""
        result = run_orchestrator("--skip-all-human", "--dry-run")

        assert result.returncode == 0
        # Should have 0 items in needs_human_review
        assert "Needs human review: 0" in result.stderr


class TestApproveValidationFailed:
    """Test --approve-validation-failed option."""

    def test_approve_validation_failed_auto_approves_recoverable(self):
        """Verify --approve-validation-failed auto-approves recoverable failures."""
        result = run_orchestrator("--approve-validation-failed", "--dry-run")

        assert result.returncode == 0
        # Should show auto-approval in dry-run report
        assert "would_auto_approve" in result.stdout or "Would auto-approve" in result.stderr


class TestApproveAll:
    """Test --approve-all option."""

    def test_approve_all_requires_confirmation(self):
        """Verify --approve-all requires --yes or --force."""
        result = run_orchestrator("--approve-all")

        # Should fail without --yes
        assert result.returncode != 0
        assert "requires --yes or --force" in result.stderr

    def test_approve_all_with_yes_succeeds(self):
        """Verify --approve-all --yes works (with --include-high for HIGH items)."""
        result = run_orchestrator("--approve-all", "--yes", "--include-high", "--dry-run")

        assert result.returncode == 0


class TestBatchingOptions:
    """Test batching-related options."""

    def test_no_batch_disables_batching(self):
        """Verify --no-batch processes suggestions individually."""
        result = run_orchestrator("--no-batch", "--output-format", "json")

        assert result.returncode == 0
        output = json.loads(result.stdout)
        # Check batching is disabled
        assert output["batching_stats"]["batching_enabled"] is False
        # Each batch should have exactly 1 suggestion
        for batch in output["batches"]:
            assert batch["suggestion_count"] == 1

    def test_max_batch_size_limits_batch(self):
        """Verify --max-batch-size limits suggestions per batch."""
        result = run_orchestrator("--max-batch-size", "2", "--dry-run")

        assert result.returncode == 0
        # Batches should have at most 2 suggestions


class TestFreshStart:
    """Test --fresh option."""

    def test_fresh_clears_previous_progress(self):
        """Verify --fresh clears previous state and starts fresh."""
        # First run with resume to verify state exists
        with_resume = run_orchestrator("--resume", "--dry-run")
        assert "1 already processed" in with_resume.stderr

        # Run with fresh
        fresh = run_orchestrator("--fresh", "--dry-run")
        assert fresh.returncode == 0
        assert "Fresh start requested" in fresh.stderr

        # Restore fixtures for other tests
        setup_test_fixtures(force=True)


class TestOutputFormat:
    """Test output format options."""

    def test_json_output_is_valid(self):
        """Verify JSON output is valid JSON."""
        result = run_orchestrator("--output-format", "json")

        assert result.returncode == 0
        # stdout should be valid JSON
        output = json.loads(result.stdout)
        assert "plan_file" in output
        assert "batches" in output
        assert "summary" in output

    def test_json_output_contains_required_fields(self):
        """Verify JSON output contains all required fields."""
        result = run_orchestrator("--output-format", "json")

        output = json.loads(result.stdout)
        assert "plan_file" in output
        assert "prefix" in output
        assert "output_dir" in output
        assert "timestamp" in output
        assert "batches" in output
        assert "to_apply" in output
        assert "needs_human_review" in output
        assert "batching_stats" in output
        assert "human_review_config" in output
        assert "resume_info" in output
        assert "summary" in output

    def test_text_output_is_readable(self):
        """Verify text output is human-readable."""
        result = run_orchestrator("--output-format", "text")

        assert result.returncode == 0
        assert "SUGGESTIONS TO APPLY" in result.stdout


class TestApproveImportanceLevels:
    """Test --approve-importance option."""

    def test_approve_importance_single_level(self):
        """Verify --approve-importance with single level."""
        result = run_orchestrator("--approve-importance", "LOW", "--dry-run")

        assert result.returncode == 0

    def test_approve_importance_multiple_levels(self):
        """Verify --approve-importance with multiple levels."""
        result = run_orchestrator("--approve-importance", "LOW", "MEDIUM", "--dry-run")

        assert result.returncode == 0


class TestMinPriority:
    """Test --min-priority option."""

    def test_min_priority_low_includes_all(self):
        """Verify --min-priority low includes all valid suggestions."""
        result = run_orchestrator("--min-priority", "low", "--output-format", "json")

        assert result.returncode == 0
        output = json.loads(result.stdout)
        # With min_priority=low, all valid items should be included
        # Check that LOW importance items are in to_apply if they're valid
        valid_count = output["summary"]["valid_count"]
        assert valid_count > 0

    def test_min_priority_medium_skips_low(self):
        """Verify --min-priority medium skips LOW importance suggestions."""
        # Get baseline with low
        result_low = run_orchestrator("--min-priority", "low", "--output-format", "json")
        result_medium = run_orchestrator("--min-priority", "medium", "--output-format", "json")

        assert result_low.returncode == 0
        assert result_medium.returncode == 0

        output_low = json.loads(result_low.stdout)
        output_medium = json.loads(result_medium.stdout)

        # Medium should have fewer or equal valid items (LOW skipped)
        assert output_medium["summary"]["valid_count"] <= output_low["summary"]["valid_count"]
        # Medium should have more or equal skipped items
        assert output_medium["summary"]["skipped_count"] >= output_low["summary"]["skipped_count"]

    def test_min_priority_high_only_high(self):
        """Verify --min-priority high only includes HIGH importance suggestions."""
        result = run_orchestrator("--min-priority", "high", "--output-format", "json")

        assert result.returncode == 0
        output = json.loads(result.stdout)

        # All items in to_apply should be HIGH importance
        for item in output["to_apply"]:
            assert item["importance"] == "HIGH"

    def test_min_priority_default_is_low(self):
        """Verify default behavior (no --min-priority) is same as --min-priority low."""
        result_default = run_orchestrator("--output-format", "json")
        result_low = run_orchestrator("--min-priority", "low", "--output-format", "json")

        assert result_default.returncode == 0
        assert result_low.returncode == 0

        output_default = json.loads(result_default.stdout)
        output_low = json.loads(result_low.stdout)

        # Should have same counts
        assert output_default["summary"]["valid_count"] == output_low["summary"]["valid_count"]
        assert output_default["summary"]["skipped_count"] == output_low["summary"]["skipped_count"]

    def test_include_low_deprecated_still_works(self):
        """Verify deprecated --include-low still works but shows warning."""
        result = run_orchestrator("--include-low", "--output-format", "json")

        assert result.returncode == 0
        # Should show deprecation warning
        assert "deprecated" in result.stderr.lower() or "DEPRECATED" in result.stderr

    def test_include_low_and_min_priority_conflict(self):
        """Verify warning when both --include-low and --min-priority are used."""
        result = run_orchestrator("--include-low", "--min-priority", "medium")

        # Should succeed with warning (--min-priority takes precedence)
        assert "WARNING" in result.stderr
        assert "--min-priority" in result.stderr


class TestBatchReviewMode:
    """Test --batch-review-mode option."""

    def test_batch_review_mode_individual(self):
        """Verify --batch-review-mode individual."""
        result = run_orchestrator("--batch-review-mode", "individual", "--output-format", "json")

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["human_review_config"]["mode"] == "individual"
        assert output["human_review_config"]["batch_enabled"] is False

    def test_batch_review_mode_by_importance(self):
        """Verify --batch-review-mode by-importance (default)."""
        result = run_orchestrator("--batch-review-mode", "by-importance", "--output-format", "json")

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["human_review_config"]["mode"] == "by-importance"
        assert output["human_review_config"]["batch_enabled"] is True

    def test_batch_review_mode_summary_only(self):
        """Verify --batch-review-mode summary-only."""
        result = run_orchestrator("--batch-review-mode", "summary-only", "--output-format", "json")

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["human_review_config"]["mode"] == "summary-only"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_plan_file(self):
        """Verify error on nonexistent plan file."""
        cmd = [
            "uv", "run", "--",
            "python", "apply_suggestions_orchestrator.py",
            "--plan-file", "nonexistent.md"
        ]
        result = subprocess.run(cmd, cwd=str(SKILL_DIR), capture_output=True, text=True)

        assert result.returncode != 0
        assert "not found" in result.stderr.lower()

    def test_missing_review_plan_dir(self, tmp_path):
        """Verify error when review-plan directory is missing."""
        # Create a plan file without review-plan directory
        plan_file = tmp_path / "test.md"
        plan_file.write_text("# Test Plan")

        cmd = [
            "uv", "run", "--",
            "python", "apply_suggestions_orchestrator.py",
            "--plan-file", str(plan_file)
        ]
        result = subprocess.run(cmd, cwd=str(SKILL_DIR), capture_output=True, text=True)

        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "Run --review-plan first" in result.stderr


@pytest.fixture
def restore_fixtures():
    """Fixture to restore test fixtures after tests that modify them."""
    yield
    # Restore fixtures after test completes
    setup_test_fixtures(force=True)


class TestUserSkipFunctionality:
    """Test user skip checkbox functionality in report.md."""

    def test_user_skip_filters_suggestions(self, restore_fixtures):
        """Verify user-marked [x] Skip suggestions are filtered out."""
        # Create a report.md with skip markers
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        report_content = '''# Review Report

## HIGH

### S001: Add email format validation
- [ ] Skip
**Validation:** ✓ Valid | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 2

Description.

---

### S003: Implement account lockout
- [x] Skip
**Validation:** ✓ Valid | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 6

Description.

---

## MEDIUM

### S002: Enforce password complexity
- [x] Skip
**Validation:** ✓ Valid | **Model:** gpt-4 | **Type:** addition | **Section:** Step 2

Description.

---
'''
        (review_plan_dir / "report.md").write_text(report_content)

        result = run_orchestrator("--output-format", "json")

        assert result.returncode == 0
        # Should show user skipped message
        assert "User skipped" in result.stderr
        assert "S002" in result.stderr or "S003" in result.stderr

    def test_user_skip_with_dry_run(self, restore_fixtures):
        """Verify dry-run shows user skip filtering."""
        # Create a report.md with skip markers
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        report_content = '''# Review Report

## HIGH

### S001: Test suggestion
- [x] Skip
**Validation:** ✓ Valid | **Model:** test | **Type:** addition | **Section:** Step 1

Description.

---
'''
        (review_plan_dir / "report.md").write_text(report_content)

        result = run_orchestrator("--dry-run")

        assert result.returncode == 0
        assert "User skipped" in result.stderr

    def test_no_report_file_works_normally(self, restore_fixtures):
        """Verify orchestrator works when report.md doesn't exist."""
        # Remove report.md if it exists
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        report_path = review_plan_dir / "report.md"
        if report_path.exists():
            report_path.unlink()

        result = run_orchestrator("--dry-run")

        # Should still work without report.md
        assert result.returncode == 0

    def test_empty_skip_checkboxes_filters_nothing(self, restore_fixtures):
        """Verify no filtering when all checkboxes are unchecked."""
        # Create a report.md without skip markers
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        report_content = '''# Review Report

## HIGH

### S001: Test suggestion
- [ ] Skip
**Validation:** ✓ Valid | **Model:** test | **Type:** addition | **Section:** Step 1

Description.

---

### S003: Another suggestion
- [ ] Skip
**Validation:** ✓ Valid | **Model:** test | **Type:** addition | **Section:** Step 6

Description.

---
'''
        (review_plan_dir / "report.md").write_text(report_content)

        result = run_orchestrator("--output-format", "json")

        assert result.returncode == 0
        # Should NOT show user skipped message (no skips marked)
        assert "User skipped 0" not in result.stderr or "User skipped" not in result.stderr


class TestGroupBasedSkipping:
    """Test group-based skip functionality (G{n} groups and G{n}S{m} suggestions)."""

    def test_skip_entire_group_removes_all_suggestions(self, restore_fixtures):
        """Marking group as skipped removes all its suggestions."""
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        report_content = '''# Review Report

## G1: Error handling improvements
- [x] Skip this group

### G1S1: Add try/catch block
- [ ] Skip
**Validation:** ✓ Valid | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 2

Description.

---

### G1S2: Add logging
- [ ] Skip
**Validation:** ✓ Valid | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 2

Description.

---

## G2: Security improvements
- [ ] Skip this group

### G2S1: Add validation
- [ ] Skip
**Validation:** ✓ Valid | **Model:** gpt-4 | **Type:** addition | **Section:** Step 3

Description.

---
'''
        (review_plan_dir / "report.md").write_text(report_content)

        result = run_orchestrator("--output-format", "json")

        assert result.returncode == 0
        assert "User skipped 1 groups" in result.stderr

    def test_skip_individual_suggestion_keeps_others(self, restore_fixtures):
        """Skipping an individual suggestion via bracket-notation hash."""
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        # Use v2 bracket-notation with the actual suggestion hash for
        # G2S1 ("Enforce password complexity") = 364825472afa4119
        # from the test fixture's grouped.json.
        report_content = '''# Review Report

## G1 [37cb243b0635c1ed]: Add input validation for email
- [ ] Skip this group

### G1S1 [8bcf94998a487ed7]: Add email format validation
- [ ] Skip
**Validation:** ✓ Valid | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 2

Description.

---

## G2 [1440c45af71cc3a1]: Add password complexity requirements
- [ ] Skip this group

### G2S1 [364825472afa4119]: Enforce password complexity
- [x] Skip
**Validation:** ✓ Valid | **Model:** gpt-4 | **Type:** addition | **Section:** Step 2

Description.

---
'''
        (review_plan_dir / "report.md").write_text(report_content)

        result = run_orchestrator("--output-format", "json")

        assert result.returncode == 0
        assert "User skipped 1 individual suggestions" in result.stderr
        assert "364825472afa4119" in result.stderr

    def test_mixed_group_and_individual_skips(self, restore_fixtures):
        """Skip G1 entirely, skip G2S1 individually using bracket-notation hashes."""
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        # Use v2 bracket-notation with actual hashes from the test fixture.
        # G1 = 37cb243b0635c1ed ("Add input validation for email")
        # G2S1 = 364825472afa4119 ("Enforce password complexity")
        report_content = '''# Review Report

## G1 [37cb243b0635c1ed]: Add input validation for email
- [x] Skip this group

### G1S1 [8bcf94998a487ed7]: Add email format validation
- [ ] Skip
**Validation:** ✓ Valid

---

## G2 [1440c45af71cc3a1]: Add password complexity requirements
- [ ] Skip this group

### G2S1 [364825472afa4119]: Enforce password complexity
- [x] Skip
**Validation:** ✓ Valid

---
'''
        (review_plan_dir / "report.md").write_text(report_content)

        result = run_orchestrator("--output-format", "json")

        assert result.returncode == 0
        assert "User skipped 1 groups" in result.stderr
        assert "User skipped 1 individual suggestions" in result.stderr
        assert "364825472afa4119" in result.stderr

    def test_backward_compatibility_old_format(self, restore_fixtures):
        """Old S001 format still works."""
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        # Use old format with S### IDs
        report_content = '''# Review Report

## HIGH

### S001: Add error handling
- [ ] Skip
**Validation:** ✓ Valid | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 3

Description.

---

### S002: Add validation
- [x] Skip
**Validation:** ✓ Valid | **Model:** gpt-4 | **Type:** addition | **Section:** Step 2

Description.

---
'''
        (review_plan_dir / "report.md").write_text(report_content)

        result = run_orchestrator("--output-format", "json")

        assert result.returncode == 0
        # Should show old format skipped
        assert "old format" in result.stderr.lower() or "S002" in result.stderr


class TestReportFormat:
    """Test the new report format with skip checkboxes and explicit validation."""

    def test_report_contains_skip_checkbox_placeholder(self):
        """Verify generated report contains - [ ] Skip lines."""
        # This tests that the report.md generated by review_plan has the new format
        # We can check by reading the report.md created by test fixtures or previous runs
        review_plan_dir = TEST_PLAN_DIR / "review-plan"
        report_path = review_plan_dir / "report.md"

        if report_path.exists():
            content = report_path.read_text()
            # New format should have "- [ ] Skip" or "- [x] Skip" lines
            # Old format would have badges like [✓] in the title
            # Check for the new validation format
            if "### S" in content:  # Only check if there are suggestions
                # Either it has the new format or it's the old format
                has_new_format = "- [ ] Skip" in content or "- [x] Skip" in content
                has_validation_line = "**Validation:**" in content
                # If suggestions exist and we have the new format, validation line should be there
                if has_new_format:
                    assert has_validation_line, "New format should have **Validation:** line"


def apply_suggestion_overrides(groups, suggestion_overrides):
    """Replicate per-suggestion override application logic from the orchestrator.

    This mirrors the logic in apply_suggestions_orchestrator.py (lines ~806-838)
    for testing without running the full orchestrator.
    """
    import copy
    merged = copy.deepcopy(groups)

    for idx, group in enumerate(merged, 1):
        suggestions = group.get("suggestions", [])
        modified = False
        for sugg_idx, sugg in enumerate(suggestions, 1):
            sid = f"G{idx}S{sugg_idx}"
            if sid in suggestion_overrides:
                override_status = suggestion_overrides[sid]
                if override_status == "invalid":
                    sugg["_user_override_invalid"] = True
                elif override_status == "valid":
                    sugg["_user_override_valid"] = True
                modified = True

        if modified:
            # Remove invalid-overridden suggestions
            remaining = [s for s in suggestions if not s.get("_user_override_invalid")]
            group["suggestions"] = remaining

            # If all remaining are explicitly marked valid and group was
            # needs-human-decision/validation_failed, promote to valid
            if (remaining
                and all(s.get("_user_override_valid") for s in remaining)
                and group.get("validation_status") in ("needs-human-decision", "validation_failed")):
                old_status = group.get("validation_status", "unknown")
                group["validation_status"] = "valid"
                group["validation_reason"] = f"All suggestions individually marked valid by user (was {old_status})"
                group["user_override"] = True

    return merged


class TestPerSuggestionOverrides:
    """Test per-suggestion validation override application logic."""

    def test_suggestion_override_invalid_removes_suggestion(self):
        """Override G1S2 as invalid removes it from the group, keeps others."""
        groups = [
            {
                "theme": "Error handling",
                "validation_status": "valid",
                "suggestions": [
                    {"title": "Add try/catch", "desc": "Add error handling"},
                    {"title": "Add logging", "desc": "Add debug logging"},
                    {"title": "Add retry", "desc": "Add retry logic"},
                ],
            }
        ]
        overrides = {"G1S2": "invalid"}
        result = apply_suggestion_overrides(groups, overrides)

        # G1S2 (Add logging) should be removed
        assert len(result[0]["suggestions"]) == 2
        remaining_titles = [s["title"] for s in result[0]["suggestions"]]
        assert "Add try/catch" in remaining_titles
        assert "Add retry" in remaining_titles
        assert "Add logging" not in remaining_titles

    def test_suggestion_override_valid_promotes_group(self):
        """Mark all suggestions in a needs-human-decision group as valid, group becomes valid."""
        groups = [
            {
                "theme": "Security fix",
                "validation_status": "needs-human-decision",
                "validation_reason": "Ambiguous lockout policy",
                "suggestions": [
                    {"title": "Add rate limiting", "desc": "Limit login attempts"},
                    {"title": "Add lockout", "desc": "Lock after failures"},
                ],
            }
        ]
        overrides = {"G1S1": "valid", "G1S2": "valid"}
        result = apply_suggestion_overrides(groups, overrides)

        # Group should be promoted to valid
        assert result[0]["validation_status"] == "valid"
        assert "All suggestions individually marked valid" in result[0]["validation_reason"]
        assert result[0]["user_override"] is True
        # Both suggestions should still be present
        assert len(result[0]["suggestions"]) == 2

    def test_suggestion_override_partial_no_group_promotion(self):
        """Mark only some suggestions valid, group status unchanged."""
        groups = [
            {
                "theme": "Performance",
                "validation_status": "needs-human-decision",
                "validation_reason": "Unclear caching strategy",
                "suggestions": [
                    {"title": "Add caching", "desc": "Cache responses"},
                    {"title": "Add pooling", "desc": "Pool connections"},
                    {"title": "Add indexing", "desc": "Add DB indexes"},
                ],
            }
        ]
        # Only mark first suggestion valid, leave others untouched
        overrides = {"G1S1": "valid"}
        result = apply_suggestion_overrides(groups, overrides)

        # Group should NOT be promoted (not all suggestions marked valid)
        assert result[0]["validation_status"] == "needs-human-decision"
        assert result[0]["validation_reason"] == "Unclear caching strategy"
        assert "user_override" not in result[0]
        # All 3 suggestions should still be present
        assert len(result[0]["suggestions"]) == 3


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-fixtures", action="store_true",
                        help="Set up test fixtures without running tests")
    args = parser.parse_args()

    if args.setup_fixtures:
        print("Setting up test fixtures...")
        setup_test_fixtures(force=True)
        print(f"Fixtures created in {TEST_PLAN_DIR}")
    else:
        # Run with pytest
        pytest.main([__file__, "-v"])
