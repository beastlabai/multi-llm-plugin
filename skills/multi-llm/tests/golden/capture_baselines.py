#!/usr/bin/env python3
"""
Capture pre-refactor golden baseline snapshots for all three apply orchestrators.

This script creates temporary fixture data, runs each orchestrator in various
modes, and saves the outputs as golden reference files for parity validation
during and after the refactor.

Usage:
    uv run --project skills/multi-llm -- python \
        skills/multi-llm/tests/golden/capture_baselines.py

The script is idempotent — re-running it will overwrite existing golden files.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Paths
SKILL_DIR = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent
TESTS_DIR = SKILL_DIR / "tests"

sys.path.insert(0, str(SKILL_DIR))

from utils.output_handler import sanitize_prefix, get_phase_dir
from utils.state_manager import (
    StateManager,
    generate_group_id,
    generate_suggestion_id,
    stamp_stable_ids,
    CURRENT_FORMAT_VERSION,
)


# ===========================================================================
# Fixture Data
# ===========================================================================

PLAN_CONTENT = """\
# Golden Baseline Test Plan

## Overview
A sample plan for capturing golden baseline orchestrator outputs.

## Step 1: Database Schema
Create users table with id, email, password_hash, created_at.

## Step 2: User Model
Create User model with email validation and password requirements.

## Step 3: Authentication Service
Implement login(email, password), logout(token), verify_token(token).

## Step 4: API Endpoints
Create POST /auth/login, POST /auth/logout, GET /auth/me.

## Step 5: Session Management
Session storage using Redis with 24-hour TTL and multi-device support.

## Step 6: Security Measures
Rate limiting, password hashing with bcrypt, CSRF protection.

## Step 7: Testing
Unit tests, integration tests, E2E tests.
"""

SUGGESTIONS_GROUPS = [
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
]

SUGGESTIONS_VALIDATION = {
    "metadata": {
        "schema_version": "2.0",
        "validated_at": "2026-03-14T10:00:00",
        "model": "mock-llm",
        "plan_hash": "golden123",
        "total_groups": 5
    },
    "groups": [
        {"group_index": 0, "status": "valid", "reason": "Email validation is clear.", "confidence": 0.95},
        {"group_index": 1, "status": "valid", "reason": "Password complexity is well-defined.", "confidence": 0.90},
        {"group_index": 2, "status": "needs-human-decision", "reason": "Lockout config needs business input.",
         "confidence": 0.60, "error_type": "real_ambiguity", "recoverable": False},
        {"group_index": 3, "status": "valid", "reason": "Session logging is straightforward.", "confidence": 0.88},
        {"group_index": 4, "status": "validation_failed", "reason": "Failed to parse LLM response.",
         "confidence": 0.0, "error_type": "parsing_error", "recoverable": True},
    ]
}


CODE_FIXES_GROUPS = [
    {
        "theme": "Missing null check",
        "category": "bug",
        "models": ["gpt-4", "claude-3"],
        "suggestions": [{
            "title": "Add null check",
            "desc": "Add null check before accessing user.name",
            "type": "bug",
            "importance": "HIGH",
            "file": "src/auth.py",
            "line_range": [42, 50],
            "anchor_text": "user.name",
            "source_model": "gpt-4"
        }]
    },
    {
        "theme": "Improve error handling",
        "category": "improvement",
        "models": ["claude-3"],
        "suggestions": [{
            "title": "Add try-catch",
            "desc": "Wrap database call in try-catch",
            "type": "improvement",
            "importance": "MEDIUM",
            "file": "src/db.py",
            "line_range": [100, 120],
            "anchor_text": "db.query()",
            "source_model": "claude-3"
        }]
    },
    {
        "theme": "Security fix",
        "category": "security",
        "models": ["gpt-4"],
        "suggestions": [{
            "title": "Sanitize input",
            "desc": "Sanitize user input to prevent SQL injection",
            "type": "security",
            "importance": "HIGH",
            "file": "src/api.py",
            "line_range": [200, 210],
            "anchor_text": "request.params",
            "source_model": "gpt-4"
        }]
    },
    {
        "theme": "Code cleanup",
        "category": "style",
        "models": ["gpt-4"],
        "suggestions": [{
            "title": "Remove unused import",
            "desc": "Remove unused import statement",
            "type": "style",
            "importance": "LOW",
            "file": "src/utils.py",
            "line_range": [1, 5],
            "anchor_text": "import os",
            "source_model": "gpt-4"
        }]
    },
]

CODE_FIXES_VALIDATION = {
    "metadata": {
        "schema_version": "2.0",
        "validated_at": "2026-03-14T10:00:00",
        "model": "mock-llm",
        "plan_hash": "golden_cf123",
        "total_groups": 4
    },
    "groups": [
        {"group_index": 0, "status": "valid", "reason": "Real issue confirmed", "confidence": 0.95},
        {"group_index": 1, "status": "needs-human-decision", "reason": "Ambiguous whether this is needed",
         "confidence": 0.5, "error_type": "real_ambiguity", "recoverable": False},
        {"group_index": 2, "status": "valid", "reason": "Security issue confirmed", "confidence": 0.99},
        {"group_index": 3, "status": "invalid", "reason": "Import is actually used", "confidence": 0.8},
    ]
}


TASK_SUGGESTIONS_GROUPS = [
    {
        "theme": "Missing rate limiting task",
        "category": "coverage",
        "models": ["cursor-agent"],
        "suggestions": [{
            "title": "Add rate limiting task",
            "desc": "Plan requires rate limiting but no task covers it.",
            "type": "addition",
            "reference": "Plan Coverage",
            "importance": "HIGH",
            "source_model": "cursor-agent",
        }]
    },
    {
        "theme": "T003 missing dependency on T001",
        "category": "dependency",
        "models": ["cursor-agent"],
        "suggestions": [{
            "title": "Fix T003 dependency",
            "desc": "T003 uses the schema created by T001 but doesn't list T001 in depends_on.",
            "type": "modification",
            "reference": "T003",
            "importance": "HIGH",
            "source_model": "cursor-agent",
        }]
    },
    {
        "theme": "T005 description lacks detail",
        "category": "clarity",
        "models": ["cursor-agent", "gpt-4"],
        "suggestions": [{
            "title": "Clarify T005 description",
            "desc": "T005 says 'Create API endpoints' but doesn't specify request/response formats.",
            "type": "clarification",
            "reference": "T005",
            "importance": "MEDIUM",
            "source_model": "cursor-agent",
        }]
    },
    {
        "theme": "T002 acceptance criteria vague",
        "category": "clarity",
        "models": ["gpt-4"],
        "suggestions": [{
            "title": "Improve T002 acceptance criteria",
            "desc": "T002 acceptance criteria should list specific interfaces.",
            "type": "clarification",
            "reference": "T002",
            "importance": "LOW",
            "source_model": "gpt-4",
        }]
    },
]

TASK_SUGGESTIONS_VALIDATION = {
    "metadata": {
        "schema_version": "2.0",
        "validated_at": "2026-03-14T10:00:00",
        "model": "mock-llm",
        "plan_hash": "golden_ts123",
        "total_groups": 4,
    },
    "groups": [
        {"group_index": 0, "status": "valid", "reason": "Clear addition needed.", "confidence": 0.95},
        {"group_index": 1, "status": "valid", "reason": "Dependency fix is straightforward.", "confidence": 0.90},
        {"group_index": 2, "status": "needs-human-decision", "reason": "Ambiguous scope of clarification.",
         "confidence": 0.55, "error_type": "real_ambiguity", "recoverable": False},
        {"group_index": 3, "status": "valid", "reason": "Minor wording improvement.", "confidence": 0.80},
    ],
}

TASKS_CONTENT = """\
# Implementation Tasks

## T001: Create notification data model
Set up the database schema for notifications.
- Depends on: none
- Complexity: low

## T002: Implement notification service
Core service layer for creating and querying notifications.
- Depends on: T001
- Complexity: medium

## T003: Add email channel adapter
Integrate with email provider for transactional emails.
- Depends on: T002
- Complexity: medium

## T004: Add WebSocket delivery
Real-time in-app notification delivery via WebSocket.
- Depends on: T002
- Complexity: high

## T005: Create API endpoints
REST endpoints for sending and managing notifications.
- Depends on: T002
- Complexity: medium

## T006: Write tests
Unit, integration, and E2E tests for the notification system.
- Depends on: T003, T004, T005
- Complexity: medium
"""

REPORT_MD_SUGGESTIONS = """\
# Review Report

## HIGH

### G1S1 [{g1s1_hash}]: Add email format validation
- [ ] Skip
**Validation:** Valid | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 2

Add regex validation to ensure email addresses are in valid format.

---

### G3S1 [{g3s1_hash}]: Implement account lockout
- [x] Skip
**Validation:** Needs human decision | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 6

Lock account for 15 minutes after 5 failed login attempts.

---

## MEDIUM

### G2S1 [{g2s1_hash}]: Enforce password complexity
- [ ] Skip
**Validation:** Valid | **Model:** gpt-4 | **Type:** addition | **Section:** Step 2

Require passwords to have at least 8 characters with mixed case and special chars.

---

### G5S1 [{g5s1_hash}]: Specify token storage location
- [ ] Skip
**Validation:** Validation failed | **Model:** gpt-4 | **Type:** clarification | **Section:** Step 4

The plan doesn't specify whether to use cookies or localStorage.

---

## LOW

### G4S1 [{g4s1_hash}]: Log session activities
- [ ] Skip
**Validation:** Valid | **Model:** claude-sonnet | **Type:** addition | **Section:** Step 5

Add logging for login, logout, and token refresh events.

---
"""

REPORT_MD_CODE_FIXES = """\
# Code Review Report

## HIGH

### G1S1 [{g1s1_hash}]: Add null check
- [ ] Skip
**Validation:** Valid | **Model:** gpt-4 | **Type:** bug | **File:** src/auth.py

Add null check before accessing user.name

---

### G3S1 [{g3s1_hash}]: Sanitize input
- [x] Skip
**Validation:** Valid | **Model:** gpt-4 | **Type:** security | **File:** src/api.py

Sanitize user input to prevent SQL injection

---

## MEDIUM

### G2S1 [{g2s1_hash}]: Add try-catch
- [ ] Skip
**Validation:** Needs human decision | **Model:** claude-3 | **Type:** improvement | **File:** src/db.py

Wrap database call in try-catch

---

## LOW

### G4S1 [{g4s1_hash}]: Remove unused import
- [ ] Skip
**Validation:** Invalid | **Model:** gpt-4 | **Type:** style | **File:** src/utils.py

Remove unused import statement

---
"""

REPORT_MD_TASK_SUGGESTIONS = """\
# Review Report

## HIGH

### G1S1 [{g1s1_hash}]: Add rate limiting task
- [ ] Skip
**Validation:** Valid | **Model:** cursor-agent | **Type:** addition | **Reference:** Plan Coverage

Plan requires rate limiting but no task covers it.

---

### G2S1 [{g2s1_hash}]: Fix T003 dependency
- [ ] Skip
**Validation:** Valid | **Model:** cursor-agent | **Type:** modification | **Reference:** T003

T003 uses the schema created by T001 but doesn't list T001 in depends_on.

---

## MEDIUM

### G3S1 [{g3s1_hash}]: Clarify T005 description
- [x] Skip
**Validation:** Needs human decision | **Model:** cursor-agent | **Type:** clarification | **Reference:** T005

T005 says 'Create API endpoints' but doesn't specify request/response formats.

---

## LOW

### G4S1 [{g4s1_hash}]: Improve T002 acceptance criteria
- [ ] Skip
**Validation:** Valid | **Model:** gpt-4 | **Type:** clarification | **Reference:** T002

T002 acceptance criteria should list specific interfaces.

---
"""


# ===========================================================================
# Helper functions
# ===========================================================================

def compute_suggestion_hashes(groups):
    """Compute stable suggestion hashes for a set of groups."""
    hashes = {}
    for g_idx, group in enumerate(groups, 1):
        g_id = generate_group_id(group)
        for s_idx, sugg in enumerate(group.get("suggestions", []), 1):
            s_id = generate_suggestion_id(sugg)
            hashes[f"g{g_idx}s{s_idx}_hash"] = s_id
    return hashes


def setup_apply_suggestions_fixtures(tmp_dir):
    """Set up fixtures for apply_suggestions_orchestrator."""
    plan_file = tmp_dir / "golden-plan.md"
    plan_file.write_text(PLAN_CONTENT, encoding="utf-8")

    prefix = sanitize_prefix("golden-plan")
    output_dir = tmp_dir / prefix
    review_plan_dir = output_dir / "review-plan"
    review_plan_dir.mkdir(parents=True, exist_ok=True)

    # Create groups with stable IDs
    groups = copy.deepcopy(SUGGESTIONS_GROUPS)
    stamp_stable_ids(groups)

    # Save grouped.json
    (review_plan_dir / "grouped.json").write_text(
        json.dumps({"format_version": 2, "groups": groups}, indent=2),
        encoding="utf-8",
    )

    # Save validation.json
    (review_plan_dir / "validation.json").write_text(
        json.dumps(SUGGESTIONS_VALIDATION, indent=2),
        encoding="utf-8",
    )

    # Save backup.md
    (review_plan_dir / "backup.md").write_text(PLAN_CONTENT, encoding="utf-8")

    # Compute hashes for report template
    hashes = compute_suggestion_hashes(SUGGESTIONS_GROUPS)
    report_content = REPORT_MD_SUGGESTIONS.format(**hashes)
    (review_plan_dir / "report.md").write_text(report_content, encoding="utf-8")

    # Create state
    state = StateManager(plan_file)
    state.save()

    return plan_file


def setup_apply_code_fixes_fixtures(tmp_dir):
    """Set up fixtures for apply_code_fixes_orchestrator."""
    plan_file = tmp_dir / "golden-plan.md"
    plan_file.write_text(PLAN_CONTENT, encoding="utf-8")

    prefix = sanitize_prefix("golden-plan")
    output_dir = tmp_dir / prefix
    code_review_dir = output_dir / "code-review"
    code_review_dir.mkdir(parents=True, exist_ok=True)

    # Create groups with stable IDs
    groups = copy.deepcopy(CODE_FIXES_GROUPS)
    stamp_stable_ids(groups)

    # Save grouped.json (code review uses flat list format)
    (code_review_dir / "grouped.json").write_text(
        json.dumps(groups, indent=2),
        encoding="utf-8",
    )

    # Save validation.json
    (code_review_dir / "validation.json").write_text(
        json.dumps(CODE_FIXES_VALIDATION, indent=2),
        encoding="utf-8",
    )

    # Compute hashes for report template
    hashes = compute_suggestion_hashes(CODE_FIXES_GROUPS)
    report_content = REPORT_MD_CODE_FIXES.format(**hashes)
    (code_review_dir / "report.md").write_text(report_content, encoding="utf-8")

    # Create state with code-review phase completed
    state = StateManager(plan_file)
    state.mark_phase_completed("code-review")
    state.save()

    return plan_file


def setup_apply_task_suggestions_fixtures(tmp_dir):
    """Set up fixtures for apply_task_suggestions_orchestrator."""
    plan_file = tmp_dir / "golden-plan.md"
    plan_file.write_text(PLAN_CONTENT, encoding="utf-8")

    prefix = sanitize_prefix("golden-plan")
    output_dir = tmp_dir / prefix

    # Create tasks directory
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "tasks.md").write_text(TASKS_CONTENT, encoding="utf-8")

    # Create review-tasks directory
    review_tasks_dir = output_dir / "review-tasks"
    review_tasks_dir.mkdir(parents=True, exist_ok=True)

    # Create groups with stable IDs
    groups = copy.deepcopy(TASK_SUGGESTIONS_GROUPS)
    stamp_stable_ids(groups)

    (review_tasks_dir / "grouped.json").write_text(
        json.dumps({"format_version": 2, "groups": groups}, indent=2),
        encoding="utf-8",
    )

    (review_tasks_dir / "validation.json").write_text(
        json.dumps(TASK_SUGGESTIONS_VALIDATION, indent=2),
        encoding="utf-8",
    )

    # Compute hashes for report template
    hashes = compute_suggestion_hashes(TASK_SUGGESTIONS_GROUPS)
    report_content = REPORT_MD_TASK_SUGGESTIONS.format(**hashes)
    (review_tasks_dir / "report.md").write_text(report_content, encoding="utf-8")

    # Create state with prerequisite phases completed
    state = StateManager(plan_file)
    state.mark_phase_completed("review-plan")
    state.mark_phase_completed("apply-suggestions")
    state.mark_phase_completed("generate-tasks")
    state.mark_phase_completed("review-tasks")
    state.save()

    return plan_file


def run_orchestrator(script_name, plan_file, *extra_args, timeout=30):
    """Run an orchestrator and return the completed process."""
    cmd = [
        sys.executable,
        str(SKILL_DIR / script_name),
        "--plan-file", str(plan_file),
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        cwd=str(SKILL_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
    )


def save_golden(name, content, is_json=False):
    """Save content to a golden file."""
    filepath = GOLDEN_DIR / name
    if is_json:
        if isinstance(content, str):
            # Try to parse and re-format JSON
            try:
                parsed = json.loads(content)
                content = json.dumps(parsed, indent=2, sort_keys=False)
            except json.JSONDecodeError:
                pass  # Keep raw content if not valid JSON
    filepath.write_text(content, encoding="utf-8")
    print(f"  Saved: {filepath.name}")


def capture_report_parsing_intermediates(groups, phase_dir, orchestrator_type):
    """Capture report-parsing intermediate results as golden snapshots.

    Parses report.md for each orchestrator to capture:
    - Skipped suggestions (user [x] Skip checkboxes)
    - Skipped groups (group-level [x] Skip)
    - Validation overrides (user override of validation status)
    - Edited descriptions (user edits to suggestion descriptions)

    All report_parser functions take a file path (str), not content.
    """
    from utils.report_parser import (
        parse_skipped_suggestions,
        parse_skipped_groups,
        parse_skipped_group_suggestions,
        parse_validation_overrides_groups,
        parse_suggestion_validation_overrides,
        find_edited_descriptions,
    )

    report_path = phase_dir / "report.md"
    if not report_path.exists():
        return {}

    report_path_str = str(report_path)

    intermediates = {
        "report_file": str(report_path.name),
        "orchestrator_type": orchestrator_type,
    }

    # Parse skipped suggestions (old S### format)
    try:
        skipped_old = parse_skipped_suggestions(report_path_str)
        intermediates["skipped_suggestions_old_format"] = sorted(list(skipped_old)) if isinstance(skipped_old, set) else skipped_old
    except Exception as e:
        intermediates["skipped_suggestions_old_format"] = f"parse_error: {e}"

    # Parse skipped groups (group-level skip)
    try:
        skipped_groups = parse_skipped_groups(report_path_str)
        intermediates["skipped_groups"] = sorted(list(skipped_groups)) if isinstance(skipped_groups, set) else skipped_groups
    except Exception as e:
        intermediates["skipped_groups"] = f"parse_error: {e}"

    # Parse skipped individual suggestions within groups (GxSy format)
    try:
        skipped_group_suggs = parse_skipped_group_suggestions(report_path_str)
        intermediates["skipped_group_suggestions"] = sorted(list(skipped_group_suggs)) if isinstance(skipped_group_suggs, set) else skipped_group_suggs
    except Exception as e:
        intermediates["skipped_group_suggestions"] = f"parse_error: {e}"

    # Parse validation overrides for groups
    try:
        val_overrides = parse_validation_overrides_groups(report_path_str)
        intermediates["validation_overrides_groups"] = val_overrides
    except Exception as e:
        intermediates["validation_overrides_groups"] = f"parse_error: {e}"

    # Parse per-suggestion validation overrides
    try:
        sugg_overrides = parse_suggestion_validation_overrides(report_path_str)
        intermediates["suggestion_validation_overrides"] = sugg_overrides
    except Exception as e:
        intermediates["suggestion_validation_overrides"] = f"parse_error: {e}"

    # Parse edited descriptions
    try:
        edited = find_edited_descriptions(report_path_str, groups)
        intermediates["edited_descriptions"] = edited
    except Exception as e:
        intermediates["edited_descriptions"] = f"parse_error: {e}"

    return intermediates


# ===========================================================================
# Main capture routine
# ===========================================================================

def main():
    print("=" * 70)
    print("Capturing golden baseline snapshots for apply orchestrators")
    print("=" * 70)

    commit_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(SKILL_DIR),
        encoding="utf-8",
    ).stdout.strip()

    capture_date = datetime.now().isoformat()

    # -------------------------------------------------------------------
    # 1. apply_suggestions_orchestrator
    # -------------------------------------------------------------------
    print("\n--- apply_suggestions_orchestrator ---")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        plan_file = setup_apply_suggestions_fixtures(tmp_dir)
        prefix = sanitize_prefix("golden-plan")

        # Dry-run
        result = run_orchestrator(
            "apply_suggestions_orchestrator.py", plan_file,
            "--dry-run", "--no-confirm"
        )
        save_golden("apply_suggestions_dry_run_stdout.txt", result.stdout)
        save_golden("apply_suggestions_dry_run_stderr.txt", result.stderr)
        save_golden("apply_suggestions_dry_run_exit_code.txt", str(result.returncode))

        # JSON output
        result = run_orchestrator(
            "apply_suggestions_orchestrator.py", plan_file,
            "--output-format", "json", "--no-confirm"
        )
        save_golden("apply_suggestions_output.json", result.stdout, is_json=True)
        save_golden("apply_suggestions_json_stderr.txt", result.stderr)
        save_golden("apply_suggestions_json_exit_code.txt", str(result.returncode))

        # Capture orchestrator_output.json from disk
        output_file = tmp_dir / prefix / "apply-suggestions" / "orchestrator_output.json"
        if output_file.exists():
            save_golden("apply_suggestions_orchestrator_output.json",
                       output_file.read_text(encoding="utf-8"), is_json=True)
        else:
            print(f"  WARNING: {output_file} not found")

        # Report-parsing intermediates
        review_plan_dir = tmp_dir / prefix / "review-plan"
        groups = copy.deepcopy(SUGGESTIONS_GROUPS)
        stamp_stable_ids(groups)
        intermediates = capture_report_parsing_intermediates(
            groups, review_plan_dir, "apply_suggestions"
        )
        save_golden("apply_suggestions_report_parsing.json",
                   json.dumps(intermediates, indent=2, default=str), is_json=True)

    # -------------------------------------------------------------------
    # 2. apply_code_fixes_orchestrator
    # -------------------------------------------------------------------
    print("\n--- apply_code_fixes_orchestrator ---")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        plan_file = setup_apply_code_fixes_fixtures(tmp_dir)
        prefix = sanitize_prefix("golden-plan")

        # Dry-run
        result = run_orchestrator(
            "apply_code_fixes_orchestrator.py", plan_file,
            "--dry-run", "--no-confirm"
        )
        save_golden("apply_code_fixes_dry_run_stdout.txt", result.stdout)
        save_golden("apply_code_fixes_dry_run_stderr.txt", result.stderr)
        save_golden("apply_code_fixes_dry_run_exit_code.txt", str(result.returncode))

        # JSON output
        result = run_orchestrator(
            "apply_code_fixes_orchestrator.py", plan_file,
            "--output-format", "json", "--no-confirm"
        )
        save_golden("apply_code_fixes_output.json", result.stdout, is_json=True)
        save_golden("apply_code_fixes_json_stderr.txt", result.stderr)
        save_golden("apply_code_fixes_json_exit_code.txt", str(result.returncode))

        # Capture orchestrator_output.json from disk
        output_file = tmp_dir / prefix / "apply-fixes" / "orchestrator_output.json"
        if output_file.exists():
            save_golden("apply_code_fixes_orchestrator_output.json",
                       output_file.read_text(encoding="utf-8"), is_json=True)
        else:
            print(f"  WARNING: {output_file} not found")

        # Report-parsing intermediates
        code_review_dir = tmp_dir / prefix / "code-review"
        groups = copy.deepcopy(CODE_FIXES_GROUPS)
        stamp_stable_ids(groups)
        intermediates = capture_report_parsing_intermediates(
            groups, code_review_dir, "apply_code_fixes"
        )
        save_golden("apply_code_fixes_report_parsing.json",
                   json.dumps(intermediates, indent=2, default=str), is_json=True)

    # -------------------------------------------------------------------
    # 3. apply_task_suggestions_orchestrator
    # -------------------------------------------------------------------
    print("\n--- apply_task_suggestions_orchestrator ---")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        plan_file = setup_apply_task_suggestions_fixtures(tmp_dir)
        prefix = sanitize_prefix("golden-plan")

        # Dry-run
        result = run_orchestrator(
            "apply_task_suggestions_orchestrator.py", plan_file,
            "--dry-run", "--no-confirm"
        )
        save_golden("apply_task_suggestions_dry_run_stdout.txt", result.stdout)
        save_golden("apply_task_suggestions_dry_run_stderr.txt", result.stderr)
        save_golden("apply_task_suggestions_dry_run_exit_code.txt", str(result.returncode))

        # Normal run (produces orchestrator_output.json)
        result = run_orchestrator(
            "apply_task_suggestions_orchestrator.py", plan_file,
            "--no-confirm"
        )
        save_golden("apply_task_suggestions_stdout.txt", result.stdout)
        save_golden("apply_task_suggestions_stderr.txt", result.stderr)
        save_golden("apply_task_suggestions_exit_code.txt", str(result.returncode))

        # Capture orchestrator_output.json from disk
        output_file = tmp_dir / prefix / "apply-task-suggestions" / "orchestrator_output.json"
        if output_file.exists():
            save_golden("apply_task_suggestions_orchestrator_output.json",
                       output_file.read_text(encoding="utf-8"), is_json=True)
        else:
            print(f"  WARNING: {output_file} not found")

        # Report-parsing intermediates
        review_tasks_dir = tmp_dir / prefix / "review-tasks"
        groups = copy.deepcopy(TASK_SUGGESTIONS_GROUPS)
        stamp_stable_ids(groups)
        intermediates = capture_report_parsing_intermediates(
            groups, review_tasks_dir, "apply_task_suggestions"
        )
        save_golden("apply_task_suggestions_report_parsing.json",
                   json.dumps(intermediates, indent=2, default=str), is_json=True)

    # -------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------
    metadata = {
        "capture_date": capture_date,
        "commit_hash": commit_hash,
        "python_version": sys.version,
        "orchestrators": [
            "apply_suggestions_orchestrator.py",
            "apply_code_fixes_orchestrator.py",
            "apply_task_suggestions_orchestrator.py",
        ],
        "artifacts": {
            "help_files": [
                "apply_suggestions_help.txt",
                "apply_code_fixes_help.txt",
                "apply_task_suggestions_help.txt",
            ],
            "dry_run_files": [
                "apply_suggestions_dry_run_stdout.txt",
                "apply_suggestions_dry_run_stderr.txt",
                "apply_code_fixes_dry_run_stdout.txt",
                "apply_code_fixes_dry_run_stderr.txt",
                "apply_task_suggestions_dry_run_stdout.txt",
                "apply_task_suggestions_dry_run_stderr.txt",
            ],
            "json_output_files": [
                "apply_suggestions_output.json",
                "apply_code_fixes_output.json",
                "apply_task_suggestions_orchestrator_output.json",
            ],
            "orchestrator_output_files": [
                "apply_suggestions_orchestrator_output.json",
                "apply_code_fixes_orchestrator_output.json",
                "apply_task_suggestions_orchestrator_output.json",
            ],
            "report_parsing_files": [
                "apply_suggestions_report_parsing.json",
                "apply_code_fixes_report_parsing.json",
                "apply_task_suggestions_report_parsing.json",
            ],
        },
    }
    save_golden("capture_metadata.json", json.dumps(metadata, indent=2), is_json=True)

    print("\n" + "=" * 70)
    print("Golden baseline capture complete!")
    print(f"Files saved to: {GOLDEN_DIR}")
    print(f"Commit: {commit_hash}")
    print("=" * 70)


if __name__ == "__main__":
    main()
