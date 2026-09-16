#!/usr/bin/env python3
"""Tests for review_tasks_orchestrator module.

Covers:
- Argument parsing (parse_args)
- Tasks file discovery (find_tasks_file) with various edge cases
- Suggestion validation via review_orchestrator_base.validate_suggestion
- Result saving via review_orchestrator_base.save_model_result
- Aggregation via review_orchestrator_base.aggregate_results
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from review_tasks_orchestrator import (
    find_tasks_file,
    parse_args,
    load_task_review_prompt_template,
    PHASE_NAME,
    INVOKE_PHASE,
    SALVAGE_PHASE_NAME,
    TASK_REVIEW_PROMPT_FILE,
)
from utils.review_orchestrator_base import (
    validate_suggestion,
    save_model_result,
    aggregate_results,
    extract_json_array,
    REQUIRED_FIELDS,
    VALID_IMPORTANCE,
    VALID_TYPES,
)
from utils.output_handler import sanitize_prefix
from utils import cli_common


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def plan_with_tasks(temp_dir):
    """Create a plan file with a corresponding tasks file in default location."""
    plan_content = "# My Feature Plan\n\n## Overview\nBuild a widget.\n"
    plan_path = temp_dir / "my-feature.md"
    plan_path.write_text(plan_content, encoding="utf-8")

    # Create the default tasks file location
    prefix = sanitize_prefix("my-feature.md")
    tasks_dir = temp_dir / prefix / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    tasks_content = (
        "# Implementation Tasks\n\n"
        "## T001: Create database schema\n"
        "Create the initial schema for the widget feature.\n\n"
        "## T002: Implement API endpoints\n"
        "Add REST endpoints for CRUD operations.\n"
    )
    tasks_path = tasks_dir / "tasks.md"
    tasks_path.write_text(tasks_content, encoding="utf-8")

    return plan_path, tasks_path


@pytest.fixture
def plan_with_comment_path(temp_dir):
    """Create a plan file with a TASKS_FILE comment pointing to a custom tasks file."""
    custom_tasks_dir = temp_dir / "custom"
    custom_tasks_dir.mkdir(parents=True, exist_ok=True)
    custom_tasks_path = custom_tasks_dir / "my-tasks.md"
    custom_tasks_path.write_text(
        "# Tasks\n\n## T001: Setup\nInitialize the project.\n",
        encoding="utf-8",
    )

    plan_content = (
        "# Feature Plan\n\n"
        "<!-- TASKS_FILE: custom/my-tasks.md -->\n\n"
        "## Overview\nDo the thing.\n"
    )
    plan_path = temp_dir / "feature.md"
    plan_path.write_text(plan_content, encoding="utf-8")

    return plan_path, custom_tasks_path


@pytest.fixture
def phase_dir(temp_dir):
    """Create a phase output directory for review-tasks."""
    prefix = sanitize_prefix("test-plan.md")
    pdir = temp_dir / prefix / "review-tasks"
    pdir.mkdir(parents=True, exist_ok=True)
    return str(pdir)


# ---------------------------------------------------------------------------
# parse_args tests
# ---------------------------------------------------------------------------

class TestParseArgs:
    """Tests for argument parsing."""

    def test_parse_args_plan_file_required(self):
        """--plan-file is required."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["prog"]):
                parse_args()

    def test_parse_args_minimal(self):
        """Minimal invocation with only --plan-file."""
        with patch("sys.argv", ["prog", "--plan-file", "plans/test.md"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=["m:a"]):
                args = parse_args()
        assert args.plan_file == "plans/test.md"
        assert args.models is None
        assert args.interactive is False
        assert args.quick is False
        assert args.skip_validation is False
        assert args.reaggregate is False
        assert args.force is False
        assert args.max_parallel == 5

    def test_parse_args_with_models(self):
        """--models accepts multiple model specs."""
        with patch("sys.argv", [
            "prog", "--plan-file", "plans/test.md",
            "--models", "cursor-agent:auto", "gemini:gemini-2.5-flash",
        ]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=["m:a"]):
                args = parse_args()
        assert args.models == ["cursor-agent:auto", "gemini:gemini-2.5-flash"]

    def test_parse_args_quick_flag(self):
        """--quick / -q flag is recognized."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "-q"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.quick is True

    def test_parse_args_interactive_flag(self):
        """--interactive / -i flag is recognized."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--interactive"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.interactive is True

    def test_parse_args_timeout(self):
        """--timeout accepts an integer."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--timeout", "120"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.timeout == 120

    def test_parse_args_max_parallel(self):
        """--max-parallel accepts an integer."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--max-parallel", "3"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.max_parallel == 3

    def test_parse_args_skip_validation(self):
        """--skip-validation flag is recognized."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--skip-validation"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.skip_validation is True

    def test_parse_args_reaggregate(self):
        """--reaggregate flag is recognized."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--reaggregate"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.reaggregate is True

    def test_parse_args_force(self):
        """--force flag is recognized."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--force"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.force is True

    def test_parse_args_validation_model(self):
        """--validation-model accepts a string."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--validation-model", "gemini:flash"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.validation_model == "gemini:flash"

    def test_parse_args_internal_validation(self):
        """--internal-validation flag is recognized."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md", "--internal-validation"]):
            with patch("review_tasks_orchestrator.get_all_model_specs", return_value=[]):
                args = parse_args()
        assert args.internal_validation is True

    def test_parse_args_providers_yaml_not_found(self):
        """Argument parsing succeeds even when providers.yaml is missing."""
        with patch("sys.argv", ["prog", "--plan-file", "p.md"]):
            with patch(
                "review_tasks_orchestrator.get_all_model_specs",
                side_effect=FileNotFoundError("not found"),
            ):
                args = parse_args()
        assert args.plan_file == "p.md"


# ---------------------------------------------------------------------------
# find_tasks_file tests
# ---------------------------------------------------------------------------

class TestFindTasksFile:
    """Tests for tasks file discovery."""

    def test_find_tasks_file_default_location(self, plan_with_tasks):
        """Discovers tasks file at the default {prefix}/tasks/tasks.md location."""
        plan_path, expected_tasks_path = plan_with_tasks
        result = find_tasks_file(str(plan_path))
        assert os.path.realpath(result) == os.path.realpath(str(expected_tasks_path))

    def test_find_tasks_file_from_comment(self, plan_with_comment_path):
        """Discovers tasks file from a <!-- TASKS_FILE: ... --> comment in the plan."""
        plan_path, expected_tasks_path = plan_with_comment_path
        result = find_tasks_file(str(plan_path))
        assert os.path.realpath(result) == os.path.realpath(str(expected_tasks_path))

    def test_find_tasks_file_comment_overrides_default(self, temp_dir):
        """When both comment path and default exist, comment path wins."""
        # Create default tasks file
        prefix = sanitize_prefix("dual.md")
        default_dir = temp_dir / prefix / "tasks"
        default_dir.mkdir(parents=True, exist_ok=True)
        (default_dir / "tasks.md").write_text(
            "# Default\n\n## T001: Default task\nDefault.\n",
            encoding="utf-8",
        )

        # Create custom tasks file
        custom_dir = temp_dir / "alt"
        custom_dir.mkdir(parents=True, exist_ok=True)
        custom_path = custom_dir / "custom-tasks.md"
        custom_path.write_text("# Custom\n\n## T001: Custom task\nCustom.\n", encoding="utf-8")

        # Plan with TASKS_FILE comment
        plan = temp_dir / "dual.md"
        plan.write_text(
            "# Plan\n\n<!-- TASKS_FILE: alt/custom-tasks.md -->\n\n## Overview\nStuff.\n",
            encoding="utf-8",
        )

        result = find_tasks_file(str(plan))
        assert os.path.realpath(result) == os.path.realpath(str(custom_path))

    def test_find_tasks_file_not_found_raises_error(self, temp_dir):
        """Raises FileNotFoundError when no tasks file exists."""
        plan_path = temp_dir / "lonely-plan.md"
        plan_path.write_text("# Plan\n\n## Overview\nNo tasks generated.\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="No tasks file found"):
            find_tasks_file(str(plan_path))

    def test_find_tasks_file_comment_points_to_nonexistent(self, temp_dir):
        """Raises FileNotFoundError when TASKS_FILE comment path doesn't exist."""
        plan_path = temp_dir / "bad-ref.md"
        plan_path.write_text(
            "# Plan\n\n<!-- TASKS_FILE: missing/tasks.md -->\n\n## Overview\n",
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="Tasks file referenced in plan not found"):
            find_tasks_file(str(plan_path))

    def test_find_tasks_file_malformed_raises_error(self, temp_dir):
        """Raises ValueError when tasks file is empty."""
        plan_path = temp_dir / "empty-tasks.md"
        plan_path.write_text("# Plan\n\n## Overview\nSomething.\n", encoding="utf-8")

        prefix = sanitize_prefix("empty-tasks.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "tasks.md").write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Tasks file is empty"):
            find_tasks_file(str(plan_path))

    def test_find_tasks_file_malformed_markdown(self, temp_dir):
        """Raises ValueError when tasks file has content but no task headers."""
        plan_path = temp_dir / "garbled.md"
        plan_path.write_text("# Plan\n\n## Overview\nBuild it.\n", encoding="utf-8")

        prefix = sanitize_prefix("garbled.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "tasks.md").write_text(
            "This is some random text without any task headers.\n"
            "Just paragraphs and nothing structured.\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="no task headers"):
            find_tasks_file(str(plan_path))

    def test_find_tasks_file_missing_required_sections(self, temp_dir):
        """Raises ValueError when tasks file has markdown but no T-numbered headings."""
        plan_path = temp_dir / "no-t-sections.md"
        plan_path.write_text("# Plan\n\n## Overview\nRedesign.\n", encoding="utf-8")

        prefix = sanitize_prefix("no-t-sections.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        # Valid markdown, but headings don't match '## T001' pattern
        (tasks_dir / "tasks.md").write_text(
            "# Implementation Tasks\n\n"
            "## Overview\nThese are tasks.\n\n"
            "## Step One\nDo something.\n\n"
            "## Step Two\nDo something else.\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="no task headers.*Re-run"):
            find_tasks_file(str(plan_path))

    def test_find_tasks_file_nonexistent_plan(self, temp_dir):
        """Raises FileNotFoundError when the plan file itself does not exist."""
        nonexistent_plan = str(temp_dir / "does-not-exist.md")

        with pytest.raises((FileNotFoundError, OSError)):
            find_tasks_file(nonexistent_plan)

    def test_find_tasks_file_absolute_path_in_comment_rejected(self, temp_dir):
        """Absolute path in TASKS_FILE comment triggers sys.exit(1)."""
        plan_path = temp_dir / "abs-path.md"
        plan_path.write_text(
            "# Plan\n\n<!-- TASKS_FILE: /etc/passwd -->\n\n## Overview\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc_info:
            find_tasks_file(str(plan_path))
        assert exc_info.value.code == 1

    def test_find_tasks_file_path_traversal_rejected(self, temp_dir):
        """Path traversal (..) in TASKS_FILE comment triggers sys.exit(1)."""
        plan_path = temp_dir / "traversal.md"
        plan_path.write_text(
            "# Plan\n\n<!-- TASKS_FILE: ../../../etc/passwd -->\n\n## Overview\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc_info:
            find_tasks_file(str(plan_path))
        assert exc_info.value.code == 1

    def test_find_tasks_file_staleness_warning(self, temp_dir, capsys):
        """Warns when the plan is genuinely newer than the tasks file and no
        `generate-tasks` completion record explains the gap (F3's fallback
        path — a hand-written tasks.md with no state.json at all)."""
        import time

        plan_path = temp_dir / "stale.md"
        prefix = sanitize_prefix("stale.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = tasks_dir / "tasks.md"

        tasks_path.write_text(
            "# Tasks\n\n## T001: Init\nSetup initial config.\n",
            encoding="utf-8",
        )
        plan_path.write_text(
            "# Plan\n\n## Overview\nUpdated plan content.\n",
            encoding="utf-8",
        )
        # A gap well past the F3 tolerance (seconds, not the microseconds a
        # single write burst produces), set directly rather than slept so the
        # test stays fast and deterministic.
        now = time.time()
        os.utime(tasks_path, (now - 5, now - 5))
        os.utime(plan_path, (now, now))

        find_tasks_file(str(plan_path))
        captured = capsys.readouterr()
        assert "older than the plan" in captured.err

    def test_find_tasks_file_no_false_warning_for_generate_tasks_own_writes(
        self, temp_dir, capsys
    ):
        """F3: `update_plan_tasks.py` writes tasks.md and then the plan,
        microseconds apart, and stamps
        `state.json.phases_completed["generate-tasks"]` right after — that
        record, not raw mtime order, is what the checker prefers, so the
        phase's own writes never trigger the warning about themselves."""
        from utils.state_manager import StateManager

        plan_path = temp_dir / "fresh.md"
        prefix = sanitize_prefix("fresh.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = tasks_dir / "tasks.md"

        # Same order update_plan_tasks.py writes in: tasks.md, then the plan.
        tasks_path.write_text(
            "# Tasks\n\n## T001: Init\nSetup initial config.\n",
            encoding="utf-8",
        )
        plan_path.write_text(
            "# Plan\n\n## Overview\nUpdated plan content.\n",
            encoding="utf-8",
        )
        # ...then, in the same run, the phase records its own completion.
        state = StateManager(plan_path)
        state.mark_phase_completed("generate-tasks")
        state.save()

        find_tasks_file(str(plan_path))
        captured = capsys.readouterr()
        assert "older than the plan" not in captured.err

    def test_find_tasks_file_warns_for_a_real_edit_after_generate_tasks(
        self, temp_dir, capsys
    ):
        """The other half of F3: a plan edited well after `generate-tasks`
        completed is a genuine human edit, and must still warn even though a
        completion record exists."""
        import time

        from utils.state_manager import StateManager

        plan_path = temp_dir / "edited-later.md"
        prefix = sanitize_prefix("edited-later.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = tasks_dir / "tasks.md"

        tasks_path.write_text(
            "# Tasks\n\n## T001: Init\nSetup initial config.\n",
            encoding="utf-8",
        )
        plan_path.write_text(
            "# Plan\n\n## Overview\nOriginal content.\n",
            encoding="utf-8",
        )
        state = StateManager(plan_path)
        state.mark_phase_completed("generate-tasks")
        state.save()

        # A human edits the plan long after generate-tasks finished.
        plan_path.write_text(
            "# Plan\n\n## Overview\nHuman edit after tasks were generated.\n",
            encoding="utf-8",
        )
        now = time.time()
        os.utime(plan_path, (now + 60, now + 60))

        find_tasks_file(str(plan_path))
        captured = capsys.readouterr()
        assert "older than the plan" in captured.err

    def test_find_tasks_file_whitespace_only(self, temp_dir):
        """Raises ValueError when tasks file contains only whitespace."""
        plan_path = temp_dir / "ws-only.md"
        plan_path.write_text("# Plan\n\n## Overview\nEmpty tasks.\n", encoding="utf-8")

        prefix = sanitize_prefix("ws-only.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "tasks.md").write_text("   \n\t\n   ", encoding="utf-8")

        with pytest.raises(ValueError, match="Tasks file is empty"):
            find_tasks_file(str(plan_path))


# ---------------------------------------------------------------------------
# validate_suggestion tests
# ---------------------------------------------------------------------------

class TestValidateSuggestion:
    """Tests for suggestion field validation."""

    def test_validate_suggestion_valid(self):
        """Valid suggestion passes validation."""
        suggestion = {
            "title": "Missing coverage for auth module",
            "desc": "No task covers the authentication requirements.",
            "importance": "high",
            "reference": "Plan Coverage",
            "type": "addition",
        }
        is_valid, error = validate_suggestion(suggestion)
        assert is_valid is True
        assert error is None

    def test_validate_suggestion_missing_field(self):
        """Missing required field fails validation."""
        for field in REQUIRED_FIELDS:
            suggestion = {f: f"value_{f}" for f in REQUIRED_FIELDS}
            del suggestion[field]
            is_valid, error = validate_suggestion(suggestion)
            assert is_valid is False
            assert f"Missing required field: {field}" in error

    def test_validate_suggestion_invalid_importance(self):
        """Invalid importance value fails validation."""
        suggestion = {
            "title": "Test",
            "desc": "Description",
            "importance": "critical",
            "reference": "Section A",
            "type": "addition",
        }
        is_valid, error = validate_suggestion(suggestion)
        assert is_valid is False
        assert "Invalid importance" in error

    def test_validate_suggestion_invalid_type(self):
        """Invalid type value fails validation."""
        suggestion = {
            "title": "Test",
            "desc": "Description",
            "importance": "high",
            "reference": "Section A",
            "type": "enhancement",
        }
        is_valid, error = validate_suggestion(suggestion)
        assert is_valid is False
        assert "Invalid type" in error

    def test_validate_suggestion_case_insensitive_importance(self):
        """Importance validation is case-insensitive."""
        suggestion = {
            "title": "Test",
            "desc": "Desc",
            "importance": "HIGH",
            "reference": "Ref",
            "type": "addition",
        }
        is_valid, _ = validate_suggestion(suggestion)
        assert is_valid is True

    def test_validate_suggestion_case_insensitive_type(self):
        """Type validation is case-insensitive."""
        suggestion = {
            "title": "Test",
            "desc": "Desc",
            "importance": "medium",
            "reference": "Ref",
            "type": "Modification",
        }
        is_valid, _ = validate_suggestion(suggestion)
        assert is_valid is True

    def test_validate_suggestion_all_valid_types(self):
        """All valid type values pass validation."""
        for t in VALID_TYPES:
            suggestion = {
                "title": "T",
                "desc": "D",
                "importance": "low",
                "reference": "R",
                "type": t,
            }
            is_valid, _ = validate_suggestion(suggestion)
            assert is_valid is True, f"Type '{t}' should be valid"

    def test_validate_suggestion_all_valid_importances(self):
        """All valid importance values pass validation."""
        for imp in VALID_IMPORTANCE:
            suggestion = {
                "title": "T",
                "desc": "D",
                "importance": imp,
                "reference": "R",
                "type": "addition",
            }
            is_valid, _ = validate_suggestion(suggestion)
            assert is_valid is True, f"Importance '{imp}' should be valid"


# ---------------------------------------------------------------------------
# save_model_result tests
# ---------------------------------------------------------------------------

class TestSaveModelResult:
    """Tests for result saving and salvage creation."""

    def _make_valid_suggestions(self, count=2):
        """Build a list of valid suggestion dicts."""
        suggestions = []
        for i in range(count):
            suggestions.append({
                "title": f"Finding {i+1}",
                "desc": f"Description for finding {i+1}",
                "importance": "high",
                "reference": "Plan Coverage",
                "type": "addition",
            })
        return suggestions

    def test_save_model_result_success(self, phase_dir, temp_dir):
        """Successful model result saves a JSON file with validated suggestions."""
        suggestions = self._make_valid_suggestions(3)
        output = json.dumps(suggestions)

        ok = save_model_result(
            prefix="test-plan",
            model="gemini:gemini-2.5-flash",
            success=True,
            output=output,
            error=None,
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review_tasks",
        )
        assert ok is True

        # Verify file was created
        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name("gemini:gemini-2.5-flash")
        result_path = os.path.join(phase_dir, f"{sanitized}.json")
        assert os.path.isfile(result_path)

        with open(result_path, 'r', encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == 3
        # Importance should be uppercased
        assert all(s["importance"] == "HIGH" for s in saved)

    def test_save_model_result_failure_logs_error(self, phase_dir, temp_dir):
        """Failed model result creates an error log file."""
        ok = save_model_result(
            prefix="test-plan",
            model="cursor-agent:auto",
            success=False,
            output=None,
            error="Timeout exceeded",
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review_tasks",
        )
        assert ok is False

        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name("cursor-agent:auto")
        error_path = os.path.join(phase_dir, f"error_{sanitized}.log")
        assert os.path.isfile(error_path)
        content = Path(error_path).read_text(encoding="utf-8")
        assert "Timeout exceeded" in content

    def test_save_model_result_invalid_json_creates_salvage(self, phase_dir, temp_dir):
        """Unparseable output creates a salvage file for later recovery.

        The payload is an answer the model really produced and then damaged —
        a fence around an array truncated mid-object. That is what salvage
        repairs, and it is the half of this branch that never moved; the case
        this test used to carry (a bare sentence, with no JSON anywhere in it)
        is the other half, and it is now
        ``test_save_model_result_without_any_json_skips_salvage``.
        """
        ok = save_model_result(
            prefix="test-plan",
            model="opencode:gpt-4",
            success=True,
            output='```json\n[{"title": "Finding 1", "desc": "Descripti',
            error=None,
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review_tasks",
        )
        assert ok is False

        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name("opencode:gpt-4")
        salvage_path = os.path.join(phase_dir, f"salvage_{sanitized}.json")
        assert os.path.isfile(salvage_path)

        with open(salvage_path, 'r', encoding="utf-8") as f:
            salvage = json.load(f)
        assert salvage["model"] == "opencode:gpt-4"
        assert salvage["phase"] == "review_tasks"
        assert "raw_output" in salvage

    def test_save_model_result_without_any_json_skips_salvage(self, phase_dir, temp_dir):
        """Output with no JSON in it is a failed model, not a salvage request.

        Salvage can only answer such a request with the empty array its prompt
        falls back to, which reaggregation would then read as a model that
        reviewed the document and found nothing. The error log is still written
        — the failure is recorded, just not as a parse failure.
        """
        ok = save_model_result(
            prefix="test-plan",
            model="opencode:gpt-4",
            success=True,
            output="This is not JSON at all",
            error=None,
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review_tasks",
        )
        assert ok is False

        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name("opencode:gpt-4")
        assert not os.path.isfile(
            os.path.join(phase_dir, f"salvage_{sanitized}.json")
        )
        assert not os.path.isfile(
            os.path.join(phase_dir, f"salvage_task_{sanitized}.json")
        )

        error_path = os.path.join(phase_dir, f"error_{sanitized}.log")
        content = Path(error_path).read_text(encoding="utf-8")
        assert "No JSON in response" in content
        assert "This is not JSON at all" in content

    def test_save_model_result_normalizes_fields(self, phase_dir, temp_dir):
        """Importance is uppercased and type is lowercased during save."""
        suggestions = [{
            "title": "Title",
            "desc": "Description",
            "importance": "medium",
            "reference": "Section A",
            "type": "Modification",
        }]
        output = json.dumps(suggestions)

        save_model_result(
            prefix="test-plan",
            model="test-model",
            success=True,
            output=output,
            error=None,
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review_tasks",
        )

        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name("test-model")
        result_path = os.path.join(phase_dir, f"{sanitized}.json")
        with open(result_path, 'r', encoding="utf-8") as f:
            saved = json.load(f)
        assert saved[0]["importance"] == "MEDIUM"
        assert saved[0]["type"] == "modification"

    def test_save_model_result_filters_invalid_suggestions(self, phase_dir, temp_dir):
        """Invalid suggestions are filtered out during save."""
        suggestions = [
            {
                "title": "Valid finding",
                "desc": "Proper description",
                "importance": "high",
                "reference": "Plan Coverage",
                "type": "addition",
            },
            {
                "title": "Invalid -- missing type",
                "desc": "Description",
                "importance": "high",
                "reference": "Plan Coverage",
                # Missing 'type' field
            },
        ]
        output = json.dumps(suggestions)

        save_model_result(
            prefix="test-plan",
            model="filter-test",
            success=True,
            output=output,
            error=None,
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review_tasks",
        )

        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name("filter-test")
        result_path = os.path.join(phase_dir, f"{sanitized}.json")
        with open(result_path, 'r', encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == 1
        assert saved[0]["title"] == "Valid finding"

    def test_save_model_result_empty_array(self, phase_dir, temp_dir):
        """Empty suggestions array saves successfully (model found no issues)."""
        output = json.dumps([])

        ok = save_model_result(
            prefix="test-plan",
            model="clean-model",
            success=True,
            output=output,
            error=None,
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review_tasks",
        )
        assert ok is True

        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name("clean-model")
        result_path = os.path.join(phase_dir, f"{sanitized}.json")
        with open(result_path, 'r', encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == []

    def test_save_model_result_json_in_code_block(self, phase_dir, temp_dir):
        """JSON wrapped in markdown code blocks is extracted correctly."""
        suggestions = self._make_valid_suggestions(1)
        output = f"Here are the results:\n```json\n{json.dumps(suggestions)}\n```\n"

        ok = save_model_result(
            prefix="test-plan",
            model="code-block-model",
            success=True,
            output=output,
            error=None,
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review_tasks",
        )
        assert ok is True

        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name("code-block-model")
        result_path = os.path.join(phase_dir, f"{sanitized}.json")
        with open(result_path, 'r', encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == 1


# ---------------------------------------------------------------------------
# aggregate_results tests
# ---------------------------------------------------------------------------

class TestAggregateResults:
    """Tests for grouping and report generation."""

    def _write_model_result(self, phase_dir, model_name, suggestions):
        """Helper: write a model result JSON file."""
        from utils.json_extractor import sanitize_model_name
        sanitized = sanitize_model_name(model_name)
        path = os.path.join(phase_dir, f"{sanitized}.json")
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(suggestions, f)
        return path

    def test_aggregate_results_basic(self, phase_dir, temp_dir):
        """Aggregation generates markdown and HTML reports."""
        suggestions_a = [
            {
                "title": "Missing coverage for auth",
                "desc": "Auth module not covered",
                "importance": "HIGH",
                "reference": "Plan Coverage",
                "type": "addition",
                "source_model": "model-a",
            }
        ]
        suggestions_b = [
            {
                "title": "Task ordering issue",
                "desc": "T003 depends on T002 but T002 is missing",
                "importance": "MEDIUM",
                "reference": "Task Dependencies",
                "type": "modification",
                "source_model": "model-b",
            }
        ]
        self._write_model_result(phase_dir, "model-a", suggestions_a)
        self._write_model_result(phase_dir, "model-b", suggestions_b)

        # Create a minimal plan file for HTML report generation
        plan_path = temp_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n", encoding="utf-8")

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review-tasks",
            models=["model-a", "model-b"],
            failed_models={},
            plan_path=str(plan_path),
        )

        assert os.path.isfile(report_path)
        report_content = Path(report_path).read_text(encoding="utf-8")
        assert "test-plan" in report_content

        # HTML report should also exist
        html_path = os.path.join(phase_dir, "report.html")
        assert os.path.isfile(html_path)

    def test_aggregate_results_with_failed_models(self, phase_dir, temp_dir):
        """Report includes a 'Models Failed' section when models fail."""
        plan_path = temp_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n", encoding="utf-8")

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review-tasks",
            models=["model-a", "model-b"],
            failed_models={"model-b": "Rate limit exceeded (429)"},
            plan_path=str(plan_path),
        )

        report_content = Path(report_path).read_text(encoding="utf-8")
        assert "Models Failed" in report_content
        assert "model-b" in report_content
        assert "Rate limit" in report_content

    def test_aggregate_results_no_suggestions(self, phase_dir, temp_dir):
        """Aggregation handles zero suggestions gracefully."""
        plan_path = temp_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n", encoding="utf-8")

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review-tasks",
            models=["model-a"],
            failed_models={"model-a": "timeout"},
            plan_path=str(plan_path),
        )

        report_content = Path(report_path).read_text(encoding="utf-8")
        assert "No suggestions found" in report_content

    def test_aggregate_results_with_validated_groups(self, phase_dir, temp_dir):
        """Aggregation accepts pre-computed validated groups."""
        plan_path = temp_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n", encoding="utf-8")

        validated_groups = [
            {
                "theme": "Missing Auth Coverage",
                "category": "addition",
                "validation_status": "valid",
                "validation_reason": None,
                "priority_score": 10,
                "models": ["model-a"],
                "suggestions": [
                    {
                        "title": "Auth module missing",
                        "desc": "No task covers authentication",
                        "importance": "HIGH",
                        "type": "addition",
                        "reference": "Plan Coverage",
                        "source_model": "model-a",
                    }
                ],
            }
        ]

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=str(temp_dir),
            phase_dir=phase_dir,
            phase_name="review-tasks",
            models=["model-a"],
            failed_models={},
            validated_groups=validated_groups,
            plan_path=str(plan_path),
        )

        report_content = Path(report_path).read_text(encoding="utf-8")
        assert "Missing Auth Coverage" in report_content
        assert "1 HIGH" in report_content

    def test_aggregate_results_records_artifacts_for_a_dispatched_run(
        self, phase_dir, temp_dir
    ):
        """F2: a detached run's caller never sees stdout, only the terminal
        MULTI_LLM_RESULT.artifacts line -- which used to stay [] on a fully
        successful review because nothing on this path called record_artifact.
        With no validation.json on disk, only the three report artifacts are
        recorded."""
        plan_path = temp_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n", encoding="utf-8")

        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            report_path = aggregate_results(
                prefix="test-plan",
                out_dir=str(temp_dir),
                phase_dir=phase_dir,
                phase_name="review-tasks",
                models=["model-a"],
                failed_models={},
                plan_path=str(plan_path),
            )
        finally:
            cli_common.deactivate_result()

        assert str(report_path) in ctx.artifacts
        assert os.path.join(phase_dir, "report.html") in ctx.artifacts
        assert os.path.join(phase_dir, "grouped.json") in ctx.artifacts
        assert os.path.join(phase_dir, "validation.json") not in ctx.artifacts

    def test_aggregate_results_records_validation_json_when_present(
        self, phase_dir, temp_dir
    ):
        """F12: validation.json is the only machine-readable record of the
        validation verdicts the report describes, and it was never handed to
        the host -- when the file exists beside the others, it belongs in
        MULTI_LLM_RESULT.artifacts too."""
        plan_path = temp_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n", encoding="utf-8")
        validation_path = os.path.join(phase_dir, "validation.json")
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump({"format_version": 2, "groups": []}, f)

        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            aggregate_results(
                prefix="test-plan",
                out_dir=str(temp_dir),
                phase_dir=phase_dir,
                phase_name="review-tasks",
                models=["model-a"],
                failed_models={},
                plan_path=str(plan_path),
            )
        finally:
            cli_common.deactivate_result()

        assert validation_path in ctx.artifacts


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for specific scenarios."""

    def test_empty_tasks_list(self, temp_dir):
        """Tasks file with valid markdown and task headers but tasks that are
        just headers with no content -- still parseable and find_tasks_file
        succeeds."""
        plan_path = temp_dir / "minimal.md"
        plan_path.write_text("# Plan\n\n## Overview\nMinimal.\n", encoding="utf-8")

        prefix = sanitize_prefix("minimal.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        # Tasks with just headers (bare minimum that satisfies the header pattern)
        (tasks_dir / "tasks.md").write_text(
            "# Tasks\n\n## T001: Placeholder\n\n## T002: Another Placeholder\n",
            encoding="utf-8",
        )

        # Should succeed -- the file is valid
        result = find_tasks_file(str(plan_path))
        assert os.path.isfile(result)

    def test_cross_reference_failure(self, temp_dir):
        """Tasks file with valid tasks that have zero overlap with plan content.

        The orchestrator should process this without crashing. The LLMs will
        identify the coverage gaps as findings during the actual review.
        find_tasks_file just validates structure, not semantic alignment.
        """
        plan_path = temp_dir / "xref.md"
        plan_path.write_text(
            "# Plan\n\n## Authentication\nBuild OAuth2 login flow.\n"
            "## Database\nCreate PostgreSQL schema.\n",
            encoding="utf-8",
        )

        prefix = sanitize_prefix("xref.md")
        tasks_dir = temp_dir / prefix / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        # Tasks about completely unrelated topics
        (tasks_dir / "tasks.md").write_text(
            "# Tasks\n\n"
            "## T001: Configure CI pipeline\n"
            "Set up GitHub Actions for the project.\n\n"
            "## T002: Write changelog\n"
            "Document changes for the release.\n",
            encoding="utf-8",
        )

        # find_tasks_file only validates structure, not semantic alignment
        result = find_tasks_file(str(plan_path))
        assert os.path.isfile(result)

    def test_extract_json_array_from_wrapper(self):
        """extract_json_array handles cursor-agent wrapper format."""
        wrapper = json.dumps({
            "result": json.dumps([{"title": "T", "desc": "D"}])
        })
        result = extract_json_array(wrapper)
        assert result is not None
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "T"

    def test_extract_json_array_no_array(self):
        """extract_json_array returns None when no array found."""
        result = extract_json_array("just some plain text")
        assert result is None

    def test_phase_constants(self):
        """Phase configuration constants are set correctly."""
        assert PHASE_NAME == "review-tasks"
        assert INVOKE_PHASE == "task_review"
        assert SALVAGE_PHASE_NAME == "review_tasks"
        assert TASK_REVIEW_PROMPT_FILE == "task_review.txt"


class TestLoadPromptTemplate:
    """Tests for prompt template loading."""

    def test_load_task_review_prompt_template(self):
        """Prompt template loads successfully and contains expected placeholders."""
        template = load_task_review_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
        # Template should have placeholders for plan and tasks paths
        assert "{plan_path}" in template
        assert "{tasks_path}" in template


# ---------------------------------------------------------------------------
# The inline context the phase's validators are handed
# ---------------------------------------------------------------------------

class TestValidationContextKwargs:
    """`review-tasks` validates claims *about a task list*, so the task list has
    to be in the prompt. Measured 2026-09-16 on a real run: 28 of 30 suggestions
    named a bare `T###`, `'T001' in prompt` was False for all eight batches, and
    two verdicts came back quoting text no validator had been given."""

    def test_review_tasks_embeds_both_documents_each_labelled(self, plan_with_tasks):
        from utils.review_orchestrator_base import validation_context_kwargs

        _plan_path, tasks_path = plan_with_tasks
        kwargs = validation_context_kwargs(
            "review-tasks", "PLAN TEXT", {"tasks_file": str(tasks_path)}
        )
        assert kwargs["context"] == "PLAN TEXT"
        assert "T001: Create database schema" in kwargs["secondary_context"]
        assert "Plan" in kwargs["context_label"]
        assert "artifact under review" in kwargs["secondary_context_label"]

    def test_every_other_phase_keeps_the_single_context_default(self):
        """review-plan's prompts must not move: the defaults are the contract."""
        from utils.review_orchestrator_base import validation_context_kwargs

        assert validation_context_kwargs("review-plan", "PLAN TEXT", None) == {
            "context": "PLAN TEXT"
        }
        assert validation_context_kwargs(
            "review-plan", "PLAN TEXT", {"tasks_file": "/somewhere/tasks.md"}
        ) == {"context": "PLAN TEXT"}

    def test_an_unreadable_tasks_file_degrades_to_the_plan_and_says_so(
        self, capsys, temp_dir
    ):
        """A completed fan-out is not thrown away over a missing scratch file;
        the batch still validates, against less."""
        from utils.review_orchestrator_base import validation_context_kwargs

        kwargs = validation_context_kwargs(
            "review-tasks", "PLAN TEXT", {"tasks_file": str(temp_dir / "gone.md")}
        )
        assert kwargs == {"context": "PLAN TEXT"}
        assert "validating against the plan alone" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# What a detached run hands back
# ---------------------------------------------------------------------------

class _FakeRunHandle:
    """Enough of ``detach.RunHandle`` for the detached-only writes to fire."""

    def __init__(self, phase_dir):
        self.phase_dir = str(phase_dir)


class TestValidationJsonIsNamedOnBothChannels:
    """The host reads the *poll* result, whose artifacts are rebuilt from
    `.status.json.output_files`. validation.json was recorded only in the
    terminal `MULTI_LLM_RESULT` a detached run prints into a log nobody reads —
    and `apply_task_suggestions_orchestrator.load_data` hard-fails without it."""

    def _aggregate(self, phase_dir, temp_dir, with_validation: bool):
        from utils import detach, status_file
        from utils.json_extractor import sanitize_model_name

        suggestions = [{
            "title": "T003 missing dependency",
            "desc": "T003 uses T001's schema but does not list it.",
            "importance": "HIGH",
            "reference": "T003",
            "type": "modification",
            "source_model": "model-a",
        }]
        path = os.path.join(phase_dir, f"{sanitize_model_name('model-a')}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(suggestions, handle)
        # Written upstream by the phase, not by aggregate_results; the recorder
        # only ever names paths that exist, so it has to be on disk here.
        with open(os.path.join(phase_dir, "grouped.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"groups": []}, handle)
        if with_validation:
            with open(os.path.join(phase_dir, "validation.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"results": []}, handle)

        plan_path = temp_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n", encoding="utf-8")

        saved = detach._ACTIVE
        detach._ACTIVE = _FakeRunHandle(phase_dir)
        try:
            aggregate_results(
                prefix="test-plan",
                out_dir=str(temp_dir),
                phase_dir=phase_dir,
                phase_name="review-tasks",
                models=["model-a"],
                failed_models={},
                plan_path=str(plan_path),
            )
        finally:
            detach._ACTIVE = saved
        return status_file.read_status(phase_dir).get("output_files") or {}

    def test_validation_json_is_recorded_as_an_output_file(self, phase_dir, temp_dir):
        output_files = self._aggregate(phase_dir, temp_dir, with_validation=True)
        assert output_files.get("validation") == os.path.join(
            phase_dir, "validation.json"
        )
        assert set(output_files) >= {"report", "report_html", "grouped", "validation"}

    def test_a_run_that_wrote_none_names_none(self, phase_dir, temp_dir):
        """`--skip-validation` writes no validation.json, and the recorder only
        ever names paths that exist."""
        output_files = self._aggregate(phase_dir, temp_dir, with_validation=False)
        assert "validation" not in output_files
        assert set(output_files) >= {"report", "report_html", "grouped"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTheHighPriorityGroupCountIsRecordedForEveryLane:
    """review-code recorded its distinct-HIGH-group count in `.status.json`
    and on the result line; review-plan and review-tasks printed
    `HIGH priority: N` (suggestions, during validation prep only) and
    recorded nothing, although their suggestions carry the same `importance`.
    `aggregate_results` records it through the base's one helper now."""

    def _aggregate(self, phase_dir, temp_dir, groups):
        from utils import status_file

        with open(os.path.join(phase_dir, "grouped.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"groups": groups}, handle)
        plan_path = temp_dir / "test-plan.md"
        plan_path.write_text("# Test Plan\n\n## Overview\nTest.\n", encoding="utf-8")
        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            aggregate_results(
                prefix="test-plan",
                out_dir=str(temp_dir),
                phase_dir=phase_dir,
                phase_name="review-tasks",
                models=["model-a"],
                failed_models={},
                validated_groups=groups,
                plan_path=str(plan_path),
            )
        finally:
            cli_common.deactivate_result()
        return status_file.read_status(phase_dir), context.payload(0)

    @staticmethod
    def _group(title, *importances):
        return {
            "theme": title,
            "suggestions": [
                {"title": title, "desc": f"{title} body", "importance": imp,
                 "reference": "T001", "type": "modification",
                 "source_model": f"model-{n}"}
                for n, imp in enumerate(importances)
            ],
            "models": [f"model-{n}" for n in range(len(importances))],
        }

    def test_distinct_high_groups_reach_the_checkpoint_and_the_result_line(
        self, phase_dir, temp_dir
    ):
        # Two models on one HIGH finding is ONE group; a MEDIUM group is none.
        groups = [
            self._group("T003 missing dependency", "HIGH", "HIGH"),
            self._group("Ordering", "MEDIUM"),
            self._group("Missing rollback", "high"),
        ]
        status, payload = self._aggregate(phase_dir, temp_dir, groups)
        assert status.get("high_priority_groups") == {"count": 2}
        assert payload["high_priority_groups"] == {"count": 2}

    def test_no_high_group_records_nothing(self, phase_dir, temp_dir):
        """Absent, not zero: the key is optional, like `consolidation_recommended`."""
        status, payload = self._aggregate(
            phase_dir, temp_dir, [self._group("Ordering", "MEDIUM", "LOW")]
        )
        assert "high_priority_groups" not in status
        assert "high_priority_groups" not in payload
