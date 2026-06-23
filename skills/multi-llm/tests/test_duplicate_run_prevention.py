"""Tests for duplicate run prevention: --force flag, phase guards, and model-level skip.

Tests the changes from coding_planning/prevent-duplicate-review-runs.md:
1. --force flag in parse_args()
2. Phase completion guard in main()
3. Model-level skip (skip_existing) in run_with_semaphore()
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

SKILL_DIR = Path(__file__).parent.parent

from utils.state_manager import StateManager, get_or_create_state
from utils.output_handler import get_phase_dir
from utils.json_extractor import sanitize_model_name


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_plan():
    """Create a temporary plan file with output directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_dir = Path(tmpdir)
        plan_path = plan_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n\n## Tasks\n1. Task A\n")

        # Create output directory
        output_dir = plan_dir / "test-plan"
        output_dir.mkdir()

        yield plan_path


@pytest.fixture
def temp_plan_completed_review(temp_plan):
    """Create a temp plan with review-plan phase marked as completed."""
    state = StateManager(temp_plan)
    state.mark_phase_completed("review-plan")
    state.save()
    return temp_plan


@pytest.fixture
def temp_plan_completed_code_review(temp_plan):
    """Create a temp plan with code-review phase marked as completed."""
    state = StateManager(temp_plan)
    state.mark_phase_completed("code-review")
    state.save()
    return temp_plan


@pytest.fixture
def temp_plan_with_model_results(temp_plan):
    """Create a temp plan with some existing per-model result files.

    Uses sanitize_model_name() to match what the orchestrator looks for.
    Model specs "test:model-a" and "test:model-b" sanitize to "test_model-a" and "test_model-b".
    """
    phase_dir = get_phase_dir(temp_plan, 'review-plan')
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Write valid model results using sanitized filenames
    model_a_results = [
        {"title": "Add error handling", "desc": "Try-catch needed", "importance": "HIGH", "type": "addition"}
    ]
    (phase_dir / f"{sanitize_model_name('test:model-a')}.json").write_text(json.dumps(model_a_results))

    # Write another model's results
    model_b_results = [
        {"title": "Add caching", "desc": "Cache results", "importance": "MEDIUM", "type": "addition"},
        {"title": "Fix typo", "desc": "Typo in docs", "importance": "LOW", "type": "modification"}
    ]
    (phase_dir / f"{sanitize_model_name('test:model-b')}.json").write_text(json.dumps(model_b_results))

    return temp_plan


@pytest.fixture
def temp_plan_with_code_review_results(temp_plan):
    """Create a temp plan with some existing per-model code review result files."""
    phase_dir = get_phase_dir(temp_plan, 'code-review')
    phase_dir.mkdir(parents=True, exist_ok=True)

    model_results = [
        {"title": "Missing null check", "desc": "Add null check", "importance": "HIGH",
         "type": "bug", "file": "src/main.py", "line_range": [10, 15]}
    ]
    (phase_dir / f"{sanitize_model_name('test:model-a')}.json").write_text(json.dumps(model_results))

    return temp_plan


@pytest.fixture
def temp_plan_with_corrupt_result(temp_plan):
    """Create a temp plan with a corrupt model result file."""
    phase_dir = get_phase_dir(temp_plan, 'review-plan')
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Valid result for model-a
    model_a_results = [{"title": "Good result", "desc": "OK", "importance": "LOW", "type": "addition"}]
    (phase_dir / f"{sanitize_model_name('test:model-a')}.json").write_text(json.dumps(model_a_results))

    # Corrupt result for model-b
    (phase_dir / f"{sanitize_model_name('test:model-b')}.json").write_text("{invalid json!!! [")

    return temp_plan


# ============================================================================
# 1. --force flag in parse_args()
# ============================================================================

class TestForceFlag:
    """Tests for --force CLI flag in both orchestrators."""

    def test_review_plan_help_shows_force(self):
        """--force flag appears in review_plan_orchestrator help."""
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "review_plan_orchestrator.py"), "--help"],
            capture_output=True, text=True, cwd=str(SKILL_DIR)
        )
        assert "--force" in result.stdout
        assert "Force re-run" in result.stdout

    def test_code_review_help_shows_force(self):
        """--force flag appears in code_review_orchestrator help."""
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"), "--help"],
            capture_output=True, text=True, cwd=str(SKILL_DIR)
        )
        assert "--force" in result.stdout
        assert "Force re-run" in result.stdout

    def test_review_plan_parse_args_force_default_false(self):
        """--force defaults to False in review_plan_orchestrator."""
        from review_plan_orchestrator import parse_args
        with patch('sys.argv', ['prog', '--plan-file', 'test.md', '--models', 'test:model']):
            args = parse_args()
            assert args.force is False

    def test_review_plan_parse_args_force_true(self):
        """--force is True when provided in review_plan_orchestrator."""
        from review_plan_orchestrator import parse_args
        with patch('sys.argv', ['prog', '--plan-file', 'test.md', '--models', 'test:model', '--force']):
            args = parse_args()
            assert args.force is True

    def test_code_review_parse_args_force_default_false(self):
        """--force defaults to False in code_review_orchestrator."""
        from code_review_orchestrator import parse_args
        with patch('sys.argv', ['prog', '--plan-file', 'test.md', '--models', 'test:model']):
            args = parse_args()
            assert args.force is False

    def test_code_review_parse_args_force_true(self):
        """--force is True when provided in code_review_orchestrator."""
        from code_review_orchestrator import parse_args
        with patch('sys.argv', ['prog', '--plan-file', 'test.md', '--models', 'test:model', '--force']):
            args = parse_args()
            assert args.force is True


# ============================================================================
# 2. Phase completion guard
# ============================================================================

class TestPhaseCompletionGuard:
    """Tests for phase completion guard in main()."""

    def test_review_plan_exits_code_2_when_completed(self, temp_plan_completed_review):
        """review_plan_orchestrator exits with code 2 when phase already completed."""
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "review_plan_orchestrator.py"),
                "--plan-file", str(temp_plan_completed_review),
                "--models", "test:model"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR)
        )
        assert result.returncode == 2
        assert "already been completed" in result.stdout
        assert "--force" in result.stdout

    def test_code_review_exits_code_2_when_completed(self, temp_plan_completed_code_review):
        """code_review_orchestrator exits with code 2 when phase already completed."""
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"),
                "--plan-file", str(temp_plan_completed_code_review),
                "--models", "test:model"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR)
        )
        assert result.returncode == 2
        assert "already been completed" in result.stdout
        assert "--force" in result.stdout

    def test_review_plan_force_clears_completion(self, temp_plan_completed_review):
        """--force bypasses the phase guard for review-plan."""
        # Run with --force -- it will proceed past the guard but fail later
        # (no actual LLM available). We just need to verify it doesn't exit with code 2.
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "review_plan_orchestrator.py"),
                "--plan-file", str(temp_plan_completed_review),
                "--models", "test:model",
                "--force"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        # Should NOT exit with code 2 (it will fail later due to no LLM, but that's OK)
        assert result.returncode != 2
        assert "Cleared previous phase completion" in result.stdout

    def test_code_review_force_clears_completion(self, temp_plan_completed_code_review):
        """--force bypasses the phase guard for code-review."""
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"),
                "--plan-file", str(temp_plan_completed_code_review),
                "--models", "test:model",
                "--force"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        assert result.returncode != 2
        assert "Cleared previous phase completion" in result.stdout

    def test_review_plan_no_guard_when_not_completed(self, temp_plan):
        """review_plan_orchestrator does not trigger guard when phase not completed."""
        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "review_plan_orchestrator.py"),
                "--plan-file", str(temp_plan),
                "--models", "test:model"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        # Should NOT mention "already been completed"
        assert "already been completed" not in result.stdout

    def test_reaggregate_bypasses_guard(self, temp_plan_completed_review):
        """--reaggregate mode should not be blocked by the phase guard."""
        # Create the phase directory so --reaggregate doesn't fail immediately
        phase_dir = get_phase_dir(temp_plan_completed_review, 'review-plan')
        phase_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "review_plan_orchestrator.py"),
                "--plan-file", str(temp_plan_completed_review),
                "--reaggregate"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        # Should not exit with code 2 - reaggregate runs before the guard
        assert result.returncode != 2
        assert "already been completed" not in result.stdout


# ============================================================================
# 3. Model-level skip (skip_existing)
# ============================================================================

class TestModelLevelSkipReviewPlan:
    """Tests for model-level skip in review_plan_orchestrator's run_all_models()."""

    def test_run_all_models_has_skip_existing_param(self):
        """run_all_models() accepts skip_existing parameter."""
        import inspect
        from review_plan_orchestrator import run_all_models
        sig = inspect.signature(run_all_models)
        assert "skip_existing" in sig.parameters
        assert sig.parameters["skip_existing"].default is True

    def test_skips_model_with_existing_results(self, temp_plan_with_model_results):
        """Models with existing valid JSON results are skipped."""
        from review_plan_orchestrator import run_all_models

        plan_path = str(temp_plan_with_model_results)
        phase_dir = get_phase_dir(temp_plan_with_model_results, 'review-plan')
        out_dir = str(temp_plan_with_model_results.parent / temp_plan_with_model_results.stem)

        # model-a and model-b have results; model-c does not
        # We mock run_single_model so only model-c would actually run
        with patch('review_plan_orchestrator.run_single_model', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "success": True,
                "output": json.dumps([{"title": "New finding", "desc": "test", "importance": "LOW", "type": "addition"}]),
                "error": None,
                "prompt": "test prompt",
                "stdout": "", "stderr": "",
                "duration_seconds": 1.0,
                "model_spec": "test:model-c"
            }

            results = asyncio.run(run_all_models(
                ["test:model-a", "test:model-b", "test:model-c"],
                plan_path,
                timeout=30,
                max_parallel=3,
                out_dir=out_dir,
                prefix="test-plan",
                skip_existing=True
            ))

        # model-a and model-b should be skipped (existing results)
        assert results["test:model-a"]["source"] == "existing_file"
        assert results["test:model-b"]["source"] == "existing_file"
        # model-c should have been actually run
        mock_run.assert_called_once()  # Only model-c triggered run_single_model

    def test_does_not_skip_when_force(self, temp_plan_with_model_results):
        """Models are NOT skipped when skip_existing=False (--force)."""
        from review_plan_orchestrator import run_all_models

        plan_path = str(temp_plan_with_model_results)
        out_dir = str(temp_plan_with_model_results.parent / temp_plan_with_model_results.stem)

        with patch('review_plan_orchestrator.run_single_model', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "success": True,
                "output": json.dumps([]),
                "error": None,
                "prompt": "test",
                "stdout": "", "stderr": "",
                "duration_seconds": 1.0,
                "model_spec": "test:model"
            }

            results = asyncio.run(run_all_models(
                ["test:model-a", "test:model-b"],
                plan_path,
                timeout=30,
                max_parallel=3,
                out_dir=out_dir,
                prefix="test-plan",
                skip_existing=False
            ))

        # Both models should have been run (not skipped)
        assert mock_run.call_count == 2
        for spec in ["test:model-a", "test:model-b"]:
            assert results[spec].get("source") != "existing_file"

    def test_corrupt_file_triggers_rerun(self, temp_plan_with_corrupt_result, capsys):
        """Corrupt JSON files cause the model to be re-run instead of skipped."""
        from review_plan_orchestrator import run_all_models

        plan_path = str(temp_plan_with_corrupt_result)
        out_dir = str(temp_plan_with_corrupt_result.parent / temp_plan_with_corrupt_result.stem)

        with patch('review_plan_orchestrator.run_single_model', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "success": True,
                "output": json.dumps([]),
                "error": None,
                "prompt": "test",
                "stdout": "", "stderr": "",
                "duration_seconds": 1.0,
                "model_spec": "test:model-b"
            }

            results = asyncio.run(run_all_models(
                ["test:model-a", "test:model-b"],
                plan_path,
                timeout=30,
                max_parallel=3,
                out_dir=out_dir,
                prefix="test-plan",
                skip_existing=True
            ))

        # model-a (valid) should be skipped
        assert results["test:model-a"]["source"] == "existing_file"
        # model-b (corrupt) should trigger re-run
        assert mock_run.call_count == 1

        captured = capsys.readouterr()
        assert "corrupt, re-running" in captured.out

    def test_empty_file_not_skipped(self, temp_plan):
        """Empty result files (0 bytes) are not skipped."""
        from review_plan_orchestrator import run_all_models

        phase_dir = get_phase_dir(temp_plan, 'review-plan')
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / f"{sanitize_model_name('test:model-a')}.json").write_text("")  # Empty file

        plan_path = str(temp_plan)
        out_dir = str(temp_plan.parent / temp_plan.stem)

        with patch('review_plan_orchestrator.run_single_model', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "success": True, "output": json.dumps([]),
                "error": None, "prompt": "test",
                "stdout": "", "stderr": "",
                "duration_seconds": 1.0, "model_spec": "test:model-a"
            }

            results = asyncio.run(run_all_models(
                ["test:model-a"],
                plan_path,
                timeout=30,
                max_parallel=3,
                out_dir=out_dir,
                prefix="test-plan",
                skip_existing=True
            ))

        # Empty file should not be skipped
        mock_run.assert_called_once()

    def test_non_list_json_not_skipped(self, temp_plan):
        """JSON files containing non-list data (e.g., a dict) are not skipped."""
        from review_plan_orchestrator import run_all_models

        phase_dir = get_phase_dir(temp_plan, 'review-plan')
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / f"{sanitize_model_name('test:model-a')}.json").write_text(json.dumps({"error": "not a list"}))

        plan_path = str(temp_plan)
        out_dir = str(temp_plan.parent / temp_plan.stem)

        with patch('review_plan_orchestrator.run_single_model', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "success": True, "output": json.dumps([]),
                "error": None, "prompt": "test",
                "stdout": "", "stderr": "",
                "duration_seconds": 1.0, "model_spec": "test:model-a"
            }

            results = asyncio.run(run_all_models(
                ["test:model-a"],
                plan_path,
                timeout=30,
                max_parallel=3,
                out_dir=out_dir,
                prefix="test-plan",
                skip_existing=True
            ))

        # Non-list JSON should not be skipped
        mock_run.assert_called_once()

    def test_skipped_result_contains_correct_data(self, temp_plan_with_model_results):
        """Skipped model results contain expected fields and data."""
        from review_plan_orchestrator import run_all_models

        plan_path = str(temp_plan_with_model_results)
        out_dir = str(temp_plan_with_model_results.parent / temp_plan_with_model_results.stem)

        results = asyncio.run(run_all_models(
            ["test:model-a"],
            plan_path,
            timeout=30,
            max_parallel=1,
            out_dir=out_dir,
            prefix="test-plan",
            skip_existing=True
        ))

        result = results["test:model-a"]
        assert result["success"] is True
        assert result["error"] is None
        assert result["source"] == "existing_file"
        assert result["duration_seconds"] == 0.0
        assert result["model_spec"] == "test:model-a"
        # Verify the output contains the original data
        output_data = json.loads(result["output"])
        assert len(output_data) == 1
        assert output_data[0]["title"] == "Add error handling"


class TestModelLevelSkipCodeReview:
    """Tests for model-level skip in code_review_orchestrator's run_all_reviews()."""

    def test_run_all_reviews_has_skip_existing_param(self):
        """run_all_reviews() accepts skip_existing parameter."""
        import inspect
        from code_review_orchestrator import run_all_reviews
        sig = inspect.signature(run_all_reviews)
        assert "skip_existing" in sig.parameters
        assert sig.parameters["skip_existing"].default is True

    def test_skips_model_with_existing_results(self, temp_plan_with_code_review_results):
        """Models with existing valid JSON results are skipped in code review."""
        from code_review_orchestrator import run_all_reviews

        plan_path = temp_plan_with_code_review_results
        out_dir = plan_path.parent / plan_path.stem

        with patch('code_review_orchestrator.run_single_review', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (True, [], None)

            results = asyncio.run(run_all_reviews(
                ["test:model-a", "test:model-c"],
                plan_path,
                out_dir,
                timeout=30,
                max_parallel=3,
                tracked_files=None,
                base_ref="HEAD~1",
                skip_existing=True
            ))

        # model-a should be skipped (has results)
        success, issues, error = results["test:model-a"]
        assert success is True
        assert len(issues) == 1
        assert issues[0]["title"] == "Missing null check"

        # model-c should have been run
        mock_run.assert_called_once()

    def test_does_not_skip_when_force(self, temp_plan_with_code_review_results):
        """Models are NOT skipped when skip_existing=False (--force) in code review."""
        from code_review_orchestrator import run_all_reviews

        plan_path = temp_plan_with_code_review_results
        out_dir = plan_path.parent / plan_path.stem

        with patch('code_review_orchestrator.run_single_review', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (True, [], None)

            results = asyncio.run(run_all_reviews(
                ["test:model-a"],
                plan_path,
                out_dir,
                timeout=30,
                max_parallel=3,
                tracked_files=None,
                base_ref="HEAD~1",
                skip_existing=False
            ))

        # model-a should have been run despite existing results
        mock_run.assert_called_once()

    def test_corrupt_file_triggers_rerun_code_review(self, temp_plan, capsys):
        """Corrupt JSON files cause re-run in code review."""
        from code_review_orchestrator import run_all_reviews

        phase_dir = get_phase_dir(temp_plan, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / f"{sanitize_model_name('test:model-a')}.json").write_text("not valid json {{{")

        out_dir = temp_plan.parent / temp_plan.stem

        with patch('code_review_orchestrator.run_single_review', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (True, [], None)

            results = asyncio.run(run_all_reviews(
                ["test:model-a"],
                temp_plan,
                out_dir,
                timeout=30,
                max_parallel=3,
                tracked_files=None,
                base_ref="HEAD~1",
                skip_existing=True
            ))

        mock_run.assert_called_once()
        captured = capsys.readouterr()
        assert "corrupt, re-running" in captured.out


# ============================================================================
# 4. Partial completion guard (secondary guard)
# ============================================================================

class TestPartialCompletionGuard:
    """Tests for secondary partial-completion guard in code_review_orchestrator main()."""

    def test_exit_3_when_validation_tasks_exist_no_batches(self, temp_plan):
        """Exit code 3 when validation_tasks.json exists but no batch files."""
        phase_dir = get_phase_dir(temp_plan, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "validation_tasks.json").write_text(json.dumps({"batches": []}))

        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"),
                "--plan-file", str(temp_plan),
                "--models", "test:model"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        assert result.returncode == 3
        assert "Partial completion detected" in result.stdout

    def test_exit_3_when_grouped_exists_no_validation_tasks(self, temp_plan):
        """Exit code 3 when grouped.json exists but no validation_tasks.json."""
        phase_dir = get_phase_dir(temp_plan, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "grouped.json").write_text(json.dumps([{"id": "g1"}]))

        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"),
                "--plan-file", str(temp_plan),
                "--models", "test:model"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        assert result.returncode == 3
        assert "Partial completion detected" in result.stdout

    def test_exit_3_when_model_results_exist_no_grouped(self, temp_plan):
        """Exit code 3 when per-model results exist but no grouped.json."""
        phase_dir = get_phase_dir(temp_plan, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "test_model-a.json").write_text(json.dumps([{"title": "Issue"}]))

        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"),
                "--plan-file", str(temp_plan),
                "--models", "test:model"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        assert result.returncode == 3
        assert "Partial completion detected" in result.stdout

    def test_force_bypasses_partial_guard(self, temp_plan):
        """--force bypasses the secondary partial-completion guard."""
        phase_dir = get_phase_dir(temp_plan, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "validation_tasks.json").write_text(json.dumps({"batches": []}))

        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"),
                "--plan-file", str(temp_plan),
                "--models", "test:model",
                "--force"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        # Should NOT exit with code 3 (--force bypasses partial guard)
        assert result.returncode != 3
        assert "Partial completion detected" not in result.stdout

    def test_auto_heals_when_report_and_validation_exist(self, temp_plan):
        """Auto-heals legacy state when report + validation exist but phase not marked."""
        phase_dir = get_phase_dir(temp_plan, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "validation_tasks.json").write_text(json.dumps({"batches": []}))
        (phase_dir / "validation.json").write_text(json.dumps({"groups": []}))
        (phase_dir / "report.md").write_text("# Report\n\nNo issues found.")

        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"),
                "--plan-file", str(temp_plan),
                "--models", "test:model"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=15
        )
        assert result.returncode == 0
        assert "Auto-healed" in result.stdout

        # Verify state was actually marked complete
        state = StateManager(temp_plan)
        assert state.is_phase_completed("code-review")


# ============================================================================
# 5. Reaggregate marks phase complete
# ============================================================================

class TestReaggregateMarksPhaseComplete:
    """Tests that reaggregate marks phase as complete after success."""

    def test_code_review_reaggregate_marks_complete(self, temp_plan):
        """code_review_orchestrator --reaggregate marks code-review phase complete."""
        phase_dir = get_phase_dir(temp_plan, 'code-review')
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal model result file so reaggregate has something to process
        model_results = [
            {"title": "Test issue", "desc": "Test", "importance": "LOW",
             "type": "bug", "file": "test.py", "line_range": [1, 5]}
        ]
        (phase_dir / "test_model-a.json").write_text(json.dumps(model_results))

        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "code_review_orchestrator.py"),
                "--plan-file", str(temp_plan),
                "--reaggregate", "--skip-validation"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=30
        )
        # Reaggregate should succeed (exit 0)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "Reaggregation complete" in result.stdout

        # Verify phase was marked complete in state
        state = StateManager(temp_plan)
        assert state.is_phase_completed("code-review"), \
            f"Phase 'code-review' was not marked complete. State: {state.state}"

    def test_review_plan_reaggregate_marks_complete(self, temp_plan):
        """review_plan_orchestrator --reaggregate marks review-plan phase complete."""
        phase_dir = get_phase_dir(temp_plan, 'review-plan')
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal model result file
        model_results = [
            {"title": "Add tests", "desc": "Test coverage needed",
             "importance": "MEDIUM", "type": "addition"}
        ]
        (phase_dir / "test_model-a.json").write_text(json.dumps(model_results))

        result = subprocess.run(
            [
                sys.executable, str(SKILL_DIR / "review_plan_orchestrator.py"),
                "--plan-file", str(temp_plan),
                "--reaggregate", "--skip-validation"
            ],
            capture_output=True, text=True, cwd=str(SKILL_DIR),
            timeout=30
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "Reaggregation complete" in result.stdout

        # Verify phase was marked complete in state
        state = StateManager(temp_plan)
        assert state.is_phase_completed("review-plan"), \
            f"Phase 'review-plan' was not marked complete. State: {state.state}"
