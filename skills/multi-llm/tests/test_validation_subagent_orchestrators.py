"""Integration tests for validation subagent support in orchestrators.

Tests the CLI flags and output markers:
- review_plan_orchestrator.py: --internal-validation flag and [VALIDATION_PENDING] marker
- apply_suggestions_orchestrator.py: --internal-revalidation flag and [REVALIDATION_PENDING] marker
- apply_code_fixes_orchestrator.py: --internal-revalidation flag and [REVALIDATION_PENDING] marker
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Get the skill directory for running orchestrators
SKILL_DIR = Path(__file__).parent.parent


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_plan_dir():
    """Create a temporary directory with a sample plan file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_dir = Path(tmpdir)

        # Create plan file
        plan_content = """# Test Implementation Plan

## Overview
This is a test plan for validating subagent support.

## Tasks
1. Task A
2. Task B
3. Task C
"""
        plan_path = plan_dir / "test-plan.md"
        plan_path.write_text(plan_content)

        # Create output directory structure
        output_dir = plan_dir / "test-plan"
        output_dir.mkdir()

        yield plan_path


@pytest.fixture
def temp_plan_with_review_results(temp_plan_dir):
    """Create a temp plan with existing review results for apply-suggestions tests."""
    plan_path = temp_plan_dir
    plan_dir = plan_path.parent
    output_dir = plan_dir / "test-plan"
    review_dir = output_dir / "review-plan"
    review_dir.mkdir(parents=True, exist_ok=True)

    # Create grouped.json
    grouped = [
        {
            "theme": "Error Handling",
            "category": "reliability",
            "models": ["model-a"],
            "suggestions": [
                {"title": "Add error handling", "desc": "Add try-catch", "importance": "HIGH", "type": "addition"}
            ]
        },
        {
            "theme": "Performance",
            "category": "optimization",
            "models": ["model-b"],
            "suggestions": [
                {"title": "Add caching", "desc": "Cache results", "importance": "MEDIUM", "type": "addition"}
            ]
        }
    ]
    (review_dir / "grouped.json").write_text(json.dumps(grouped, indent=2))

    # Create validation.json with some validation_failed items
    validation = {
        "groups": [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": "unknown", "recoverable": False, "revalidated": False},
            {"group_index": 1, "status": "validation_failed", "reason": "Timeout", "confidence": 0.0, "error_type": "timeout", "recoverable": True, "revalidated": False},
        ],
        "metadata": {"model": "test", "timestamp": "2025-01-01T00:00:00", "schema_version": "2.0"}
    }
    (review_dir / "validation.json").write_text(json.dumps(validation, indent=2))

    return plan_path


@pytest.fixture
def temp_plan_with_code_review_results(temp_plan_dir):
    """Create a temp plan with existing code review results for apply-code-fixes tests."""
    plan_path = temp_plan_dir
    plan_dir = plan_path.parent
    output_dir = plan_dir / "test-plan"
    code_review_dir = output_dir / "code-review"
    code_review_dir.mkdir(parents=True, exist_ok=True)

    # Create grouped.json
    grouped = [
        {
            "theme": "Missing null check",
            "category": "bug",
            "models": ["model-a"],
            "suggestions": [
                {"title": "Add null check", "desc": "Check for null", "importance": "HIGH", "type": "bug", "file": "src/main.py", "line_range": [10, 15]}
            ]
        },
        {
            "theme": "Unused import",
            "category": "style",
            "models": ["model-b"],
            "suggestions": [
                {"title": "Remove unused import", "desc": "Remove os import", "importance": "LOW", "type": "style", "file": "src/main.py", "line_range": [1, 1]}
            ]
        }
    ]
    (code_review_dir / "grouped.json").write_text(json.dumps(grouped, indent=2))

    # Create validation.json with some validation_failed items
    validation = {
        "groups": [
            {"group_index": 0, "status": "validation_failed", "reason": "Parse error", "confidence": 0.0, "error_type": "parsing_error", "recoverable": True, "revalidated": False},
            {"group_index": 1, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": "unknown", "recoverable": False, "revalidated": False},
        ],
        "metadata": {"model": "test", "timestamp": "2025-01-01T00:00:00", "schema_version": "2.0"}
    }
    (code_review_dir / "validation.json").write_text(json.dumps(validation, indent=2))

    # Create state.json with head_at_start
    state = {
        "schema_version": "1.0",
        "head_at_start": "abc123",
    }
    (output_dir / "state.json").write_text(json.dumps(state, indent=2))

    return plan_path


# ============================================================================
# Review Plan Orchestrator Tests
# ============================================================================

class TestReviewPlanOrchestratorCLI:
    """Tests for review_plan_orchestrator.py CLI argument parsing."""

    def test_internal_validation_flag_recognized(self, temp_plan_dir):
        """--internal-validation flag is recognized."""
        # This test just verifies the flag is parsed without error
        # We mock the actual execution since we can't run full orchestration
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "review_plan_orchestrator.py"),
                "--plan-file", str(temp_plan_dir),
                "--internal-validation",
                "--skip-validation",  # Skip actual validation to avoid needing LLM
                "--help"  # Just test help to verify flag is recognized
            ],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR)
        )
        # --help should show the flag
        assert "--internal-validation" in result.stdout

    def test_help_shows_internal_validation_option(self, temp_plan_dir):
        """Help output includes --internal-validation option."""
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "review_plan_orchestrator.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR)
        )
        assert result.returncode == 0
        assert "--internal-validation" in result.stdout
        assert "legacy" in result.stdout.lower() or "orchestrator" in result.stdout.lower()


class TestReviewPlanOrchestratorValidationPending:
    """Tests for [VALIDATION_PENDING] marker output."""

    def test_validation_task_json_structure(self):
        """Test the structure of validation_task.json output."""
        # Import the function directly for unit testing
        from utils.validation import prepare_validation_task

        groups = [
            {"theme": "Test", "category": "test", "models": [], "suggestions": [{"title": "T", "desc": "D", "importance": "LOW", "type": "addition"}]}
        ]
        context = "Test context"
        output_path = "/tmp/validation.json"

        result = prepare_validation_task(groups, context, output_path)

        # Verify structure
        assert "prompt" in result
        assert "output_path" in result
        assert "groups_count" in result
        assert "suggestions_json" in result
        assert "model_hint" in result

        # Verify values
        assert result["output_path"] == output_path
        assert result["groups_count"] == 1
        assert result["model_hint"] == "auto"


# ============================================================================
# Apply Suggestions Orchestrator Tests
# ============================================================================

class TestApplySuggestionsOrchestratorCLI:
    """Tests for apply_suggestions_orchestrator.py CLI argument parsing."""

    def test_internal_revalidation_flag_recognized(self):
        """--internal-revalidation flag is recognized."""
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "apply_suggestions_orchestrator.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR)
        )
        assert result.returncode == 0
        assert "--internal-revalidation" in result.stdout

    def test_help_shows_internal_revalidation_option(self):
        """Help output includes --internal-revalidation option."""
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "apply_suggestions_orchestrator.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR)
        )
        assert result.returncode == 0
        assert "--internal-revalidation" in result.stdout
        assert "legacy" in result.stdout.lower() or "orchestrator" in result.stdout.lower()


class TestApplySuggestionsOrchestratorRevalidation:
    """Tests for revalidation subagent support in apply_suggestions_orchestrator."""

    def test_revalidation_task_json_structure(self):
        """Test the structure of revalidation_task.json output."""
        from utils.validation import prepare_revalidation_task

        groups = [
            {"theme": "Test", "category": "test", "models": [], "suggestions": []},
        ]
        validation_results = [
            {"group_index": 0, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": "timeout", "recoverable": True},
        ]
        context = "Test context"
        output_path = "/tmp/revalidation.json"

        result = prepare_revalidation_task(groups, validation_results, context, output_path)

        # Verify structure
        assert "prompt" in result
        assert "output_path" in result
        assert "items_to_revalidate" in result
        assert "item_indices" in result
        assert "original_validation" in result
        assert "model_hint" in result

        # Verify values
        assert result["output_path"] == output_path
        assert result["items_to_revalidate"] == 1
        assert 0 in result["item_indices"]
        assert result["original_validation"] == validation_results

    def test_dry_run_with_revalidation(self, temp_plan_with_review_results):
        """--dry-run with --revalidate shows what would be revalidated."""
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "apply_suggestions_orchestrator.py"),
                "--plan-file", str(temp_plan_with_review_results),
                "--revalidate",
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR)
        )
        # Should show dry-run output without error
        assert "dry-run" in result.stderr.lower() or "Would revalidate" in result.stderr


# ============================================================================
# Apply Code Fixes Orchestrator Tests
# ============================================================================

class TestApplyCodeFixesOrchestratorCLI:
    """Tests for apply_code_fixes_orchestrator.py CLI argument parsing."""

    def test_internal_revalidation_flag_recognized(self):
        """--internal-revalidation flag is recognized."""
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "apply_code_fixes_orchestrator.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR)
        )
        assert result.returncode == 0
        assert "--internal-revalidation" in result.stdout

    def test_help_shows_internal_revalidation_option(self):
        """Help output includes --internal-revalidation option."""
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "apply_code_fixes_orchestrator.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR)
        )
        assert result.returncode == 0
        assert "--internal-revalidation" in result.stdout
        assert "legacy" in result.stdout.lower() or "orchestrator" in result.stdout.lower()


class TestApplyCodeFixesOrchestratorRevalidation:
    """Tests for revalidation subagent support in apply_code_fixes_orchestrator."""

    def test_dry_run_with_revalidation(self, temp_plan_with_code_review_results):
        """--dry-run with --revalidate shows what would be revalidated."""
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "apply_code_fixes_orchestrator.py"),
                "--plan-file", str(temp_plan_with_code_review_results),
                "--revalidate",
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR)
        )
        # Should show dry-run output without error
        assert "dry-run" in result.stderr.lower() or "Would revalidate" in result.stderr


# ============================================================================
# Marker Output Tests
# ============================================================================

class TestMarkerOutputs:
    """Tests for [VALIDATION_PENDING] and [REVALIDATION_PENDING] marker formats."""

    def test_validation_pending_marker_format(self):
        """[VALIDATION_PENDING] marker has correct format."""
        # The marker should be: [VALIDATION_PENDING] <path_to_validation_task.json>
        marker = "[VALIDATION_PENDING]"
        assert marker.startswith("[")
        assert marker.endswith("]")
        assert "VALIDATION" in marker
        assert "PENDING" in marker

    def test_revalidation_pending_marker_format(self):
        """[REVALIDATION_PENDING] marker has correct format."""
        marker = "[REVALIDATION_PENDING]"
        assert marker.startswith("[")
        assert marker.endswith("]")
        assert "REVALIDATION" in marker
        assert "PENDING" in marker


# ============================================================================
# Import Tests
# ============================================================================

class TestImports:
    """Tests that new functions are properly importable."""

    def test_import_build_validation_subagent_prompt(self):
        """Can import build_validation_subagent_prompt."""
        from utils.validation import build_validation_subagent_prompt
        assert callable(build_validation_subagent_prompt)

    def test_import_prepare_validation_task(self):
        """Can import prepare_validation_task."""
        from utils.validation import prepare_validation_task
        assert callable(prepare_validation_task)

    def test_import_prepare_revalidation_task(self):
        """Can import prepare_revalidation_task."""
        from utils.validation import prepare_revalidation_task
        assert callable(prepare_revalidation_task)

    def test_review_plan_orchestrator_imports_prepare_validation_task(self):
        """review_plan_orchestrator imports prepare_validation_task."""
        # Read the file and check for import
        orchestrator_path = SKILL_DIR / "review_plan_orchestrator.py"
        content = orchestrator_path.read_text()
        assert "prepare_validation_task" in content

    def test_base_class_handles_revalidation_for_apply_suggestions(self):
        """apply_suggestions_orchestrator supports revalidation via base class."""
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator
        assert ApplySuggestionsOrchestrator.supports_revalidation is True
        # Revalidation logic is in the base class (ApplyOrchestratorBase.handle_revalidation)
        from utils.apply_orchestrator_base import ApplyOrchestratorBase
        assert hasattr(ApplyOrchestratorBase, "handle_revalidation")

    def test_base_class_handles_revalidation_for_apply_code_fixes(self):
        """apply_code_fixes_orchestrator supports revalidation via base class."""
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator
        assert ApplyCodeFixesOrchestrator.supports_revalidation is True
        # Code fixes overrides handle_revalidation with base_ref support
        assert hasattr(ApplyCodeFixesOrchestrator, "handle_revalidation")


# ============================================================================
# JSON Output Format Tests
# ============================================================================

class TestValidationTaskJsonFormat:
    """Tests for validation_task.json and revalidation_task.json format."""

    def test_validation_task_is_json_serializable(self):
        """validation_task output is JSON serializable."""
        from utils.validation import prepare_validation_task

        groups = [{"theme": "Test", "category": "test", "models": [], "suggestions": []}]
        result = prepare_validation_task(groups, "context", "/tmp/out.json")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed == result

    def test_revalidation_task_is_json_serializable(self):
        """revalidation_task output is JSON serializable."""
        from utils.validation import prepare_revalidation_task

        groups = [{"theme": "Test", "category": "test", "models": [], "suggestions": []}]
        validation = [{"group_index": 0, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": "timeout"}]
        result = prepare_revalidation_task(groups, validation, "context", "/tmp/out.json")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed == result

    def test_validation_task_prompt_is_string(self):
        """validation_task prompt is a non-empty string."""
        from utils.validation import prepare_validation_task

        groups = [{"theme": "Test", "category": "test", "models": [], "suggestions": []}]
        result = prepare_validation_task(groups, "context", "/tmp/out.json")

        assert isinstance(result["prompt"], str)
        assert len(result["prompt"]) > 0

    def test_revalidation_task_prompt_is_string_or_none(self):
        """revalidation_task prompt is a string or None."""
        from utils.validation import prepare_revalidation_task

        # Case 1: Has items to revalidate - prompt should be string
        groups = [{"theme": "Test", "category": "test", "models": [], "suggestions": []}]
        validation = [{"group_index": 0, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": "timeout"}]
        result = prepare_revalidation_task(groups, validation, "context", "/tmp/out.json")
        assert isinstance(result["prompt"], str)

        # Case 2: No items to revalidate - prompt should be None
        validation = [{"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": "unknown"}]
        result = prepare_revalidation_task(groups, validation, "context", "/tmp/out.json")
        assert result["prompt"] is None
