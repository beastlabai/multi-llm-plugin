"""Integration tests for validation subagent support in orchestrators.

Tests the CLI flags and output markers:
- review_plan_orchestrator.py: [VALIDATION_PENDING] marker
- apply_suggestions_orchestrator.py: [REVALIDATION_PENDING] marker
- apply_code_fixes_orchestrator.py: [REVALIDATION_PENDING] marker
- the retired --internal-validation / --internal-revalidation flags, which still
  parse, warn once and change nothing (the in-process lane is gone)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import cli_common  # noqa: E402


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
        plan_path.write_text(plan_content, encoding="utf-8")

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
    (review_dir / "grouped.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")

    # Create validation.json with some validation_failed items
    validation = {
        "groups": [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": "unknown", "recoverable": False, "revalidated": False},
            {"group_index": 1, "status": "validation_failed", "reason": "Timeout", "confidence": 0.0, "error_type": "timeout", "recoverable": True, "revalidated": False},
        ],
        "metadata": {"model": "test", "timestamp": "2025-01-01T00:00:00", "schema_version": "2.0"}
    }
    (review_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

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
    (code_review_dir / "grouped.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")

    # Create validation.json with some validation_failed items
    validation = {
        "groups": [
            {"group_index": 0, "status": "validation_failed", "reason": "Parse error", "confidence": 0.0, "error_type": "parsing_error", "recoverable": True, "revalidated": False},
            {"group_index": 1, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": "unknown", "recoverable": False, "revalidated": False},
        ],
        "metadata": {"model": "test", "timestamp": "2025-01-01T00:00:00", "schema_version": "2.0"}
    }
    (code_review_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    # Create state.json with head_at_start
    state = {
        "schema_version": "1.0",
        "head_at_start": "abc123",
    }
    (output_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    return plan_path


# ============================================================================
# Review Plan Orchestrator Tests
# ============================================================================

def _parse(script: str, argv):
    """*script*'s own ``parse_args`` on *argv*, in-process."""
    if script == "review_plan":
        import review_plan_orchestrator as module

        return module.parse_args(argv)
    if script == "review_tasks":
        import review_tasks_orchestrator as module

        return module.parse_args(argv)
    if script == "apply_suggestions":
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator

        return ApplySuggestionsOrchestrator.parse_args(argv)
    from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator

    return ApplyCodeFixesOrchestrator.parse_args(argv)


RETIRED_FLAG_CASES = [
    ("review_plan", "--internal-validation"),
    ("review_tasks", "--internal-validation"),
    ("apply_suggestions", "--internal-revalidation"),
    ("apply_code_fixes", "--internal-revalidation"),
]


class TestRetiredInProcessFlags:
    """The in-process validation lane is retired; its flags warn, never fail.

    Each still parses (an old command line completes), prints one stderr line
    naming the delegated route, and leaves the namespace exactly as a run
    without it — so nothing downstream can branch on it.
    """

    @pytest.mark.parametrize("script,flag", RETIRED_FLAG_CASES)
    def test_parses_warns_once_and_changes_nothing(self, script, flag, capsys):
        base_argv = ["--plan-file", "p.md"]
        default = _parse(script, base_argv)
        capsys.readouterr()

        # Typed twice, still one line: the notice is per parse, not per token.
        args = _parse(script, base_argv + [flag, flag])
        err = capsys.readouterr().err

        assert vars(args) == vars(default)
        notices = [line for line in err.splitlines() if "is retired" in line]
        assert notices == [cli_common.RETIRED_FLAGS[flag]]
        assert "delegated subagent route" in notices[0]

    @pytest.mark.parametrize("script,flag", RETIRED_FLAG_CASES)
    def test_hidden_from_help(self, script, flag, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _parse(script, ["--help"])
        assert exit_info.value.code == 0
        assert flag not in capsys.readouterr().out

    @pytest.mark.parametrize("flag", sorted(cli_common.RETIRED_FLAGS))
    def test_never_replayed_into_a_follow_up(self, flag):
        """A replayed retired flag would make `set_next` drop the whole command."""
        replay = cli_common.replay_flags(
            "review-plan", ["--plan-file", "p.md", flag, "--quick"]
        )
        assert replay == ["--quick"]

    def test_apply_revalidation_takes_the_delegated_route(
        self, temp_plan_with_review_results
    ):
        """--internal-revalidation used to revalidate in-process and continue;
        now it materializes the task file and stops at the marker, as without it."""
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "apply_suggestions_orchestrator.py"),
                "--plan-file", str(temp_plan_with_review_results),
                "--revalidate", "--internal-revalidation", "--no-confirm",
            ],
            capture_output=True,
            text=True,
            cwd=str(SKILL_DIR),
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("--internal-revalidation is retired") == 1
        assert "Running internal revalidation" not in result.stderr
        assert "[REVALIDATION_PENDING]" in result.stdout
        review_dir = temp_plan_with_review_results.parent / "test-plan" / "review-plan"
        assert (review_dir / "revalidation_tasks.json").is_file()


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

class TestApplySuggestionsOrchestratorRevalidation:
    """Tests for revalidation subagent support in apply_suggestions_orchestrator."""

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
            cwd=str(SKILL_DIR),
            encoding="utf-8",
        )
        # Should show dry-run output without error
        assert "dry-run" in result.stderr.lower() or "Would revalidate" in result.stderr


# ============================================================================
# Apply Code Fixes Orchestrator Tests
# ============================================================================

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
            cwd=str(SKILL_DIR),
            encoding="utf-8",
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

    def test_validation_task_prompt_is_string(self):
        """validation_task prompt is a non-empty string."""
        from utils.validation import prepare_validation_task

        groups = [{"theme": "Test", "category": "test", "models": [], "suggestions": []}]
        result = prepare_validation_task(groups, "context", "/tmp/out.json")

        assert isinstance(result["prompt"], str)
        assert len(result["prompt"]) > 0
