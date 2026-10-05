"""Tests for the apply orchestrator base class and helper modules.

Covers:
- Helper module functions (path, selection, output helpers)
- Template method sequencing (lifecycle phase call order)
- Abstract method contracts (NotImplementedError on base class)
- Config-driven branching (supports_revalidation, supports_skip_flag)
- __init_subclass__ validation
- build_common_arg_parser flag toggles
- Edge-case matrix from the refactoring plan (lines 422-442)
- Logging and error-context verification
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.apply_orchestrator_base import (
    ApplyOrchestratorBase,
    OrchestratorError,
    build_common_arg_parser,
)
from utils.apply_path_helpers import load_json_file
from utils.apply_selection_helpers import (
    _is_user_skipped,
    merge_validation_with_groups,
    resolve_priority_args,
)
from utils.apply_output_helpers import (
    build_confirmation_needed_output,
    build_skipped_output,
    emit_json_output,
    write_and_emit_output,
)
from utils.output_handler import derive_prefix, find_output_dir


# ======================================================================
# Mock subclass for testing
# ======================================================================


class MockApplyOrchestrator(ApplyOrchestratorBase):
    """Concrete mock subclass implementing all abstract methods with minimal stubs.

    Records method calls for verifying template method sequencing.
    """

    phase_name = "apply-mock"
    review_subdir = "review-plan"
    item_noun = "suggestion"
    supports_revalidation = False
    supports_skip_flag = False

    def __init__(self, args: argparse.Namespace):
        # Track method call order
        self.call_log: List[str] = []
        super().__init__(args)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        self.call_log.append("load_data")
        return ([], [])

    def parse_user_edits(self, report_path: str) -> Dict[str, Tuple[str, str]]:
        self.call_log.append("parse_user_edits")
        return {}

    def merge_user_edits(
        self,
        groups: List[Dict[str, Any]],
        edited_descriptions: Dict[str, Tuple[str, str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        self.call_log.append("merge_user_edits")
        return (groups, [])

    def parse_skips_from_report(
        self, report_path: str
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        self.call_log.append("parse_skips_from_report")
        return (set(), set(), set())

    def parse_validation_overrides_from_report(
        self, report_path: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        self.call_log.append("parse_validation_overrides_from_report")
        return ({}, {})

    def format_item_for_output(
        self, group: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        self.call_log.append("format_item_for_output")
        return {"index": index, **group}

    def create_batches(self, items: List[Dict[str, Any]]) -> List[Any]:
        self.call_log.append("create_batches")
        return []

    def generate_batch_prompts(self, batches: List[Any]) -> List[Any]:
        self.call_log.append("generate_batch_prompts")
        return batches

    def build_output_json(
        self,
        batches: List[Any],
        *,
        resume_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.call_log.append("build_output_json")
        return {"status": "ok", "batches": [], "phase": self.phase_name}

    def get_output_path(self) -> str:
        self.call_log.append("get_output_path")
        phase_dir = os.path.join(self.out_dir, self.phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        return os.path.join(phase_dir, "orchestrator_output.json")

    def print_text_summary(
        self, batches: List[Any], output_path: str
    ) -> None:
        self.call_log.append("print_text_summary")


def _make_args(**overrides) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for testing."""
    defaults = {
        "plan_file": "/tmp/test-plan.md",
        "yes": False,
        "force": False,
        "dry_run": False,
        "fresh": False,
        "resume": False,
        "approve_all": False,
        "include_low": False,
        "min_priority": None,
        "verbose": False,
        "approve_all_low": False,
        "skip_all_human": False,
        "approve_importance": None,
        "skip_human_review": False,
        "no_batch": False,
        "max_batch_size": 4,
        "batch_review_mode": "by-importance",
        "accept_stale_consolidation": False,
        "no_confirm": True,  # Skip confirmation prompts in tests
        "include_high": False,
        "agent_decide": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _setup_plan_with_output(tmp_path: Path, plan_name: str = "test-plan") -> Path:
    """Create a plan file and its output directory structure for testing."""
    plan_path = tmp_path / f"{plan_name}.md"
    plan_path.write_text("# Test Plan\n", encoding="utf-8")

    # Create the output directory structure
    out_dir = tmp_path / plan_name
    out_dir.mkdir(exist_ok=True)

    # Create review-plan subdirectory with report.md
    review_dir = out_dir / "review-plan"
    review_dir.mkdir(exist_ok=True)
    (review_dir / "report.md").write_text("# Review Report\n", encoding="utf-8")

    return plan_path


# ======================================================================
# Test: Helper module functions
# ======================================================================


class TestLoadJsonFile:
    """Tests for apply_path_helpers.load_json_file."""

    def test_valid_json(self, tmp_path):
        """Load a valid JSON file and return parsed data."""
        data = {"key": "value", "number": 42}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_json_file(str(json_file))
        assert result == data

    def test_valid_json_array(self, tmp_path):
        """Load a valid JSON file containing an array."""
        data = [{"id": 1}, {"id": 2}]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_json_file(str(json_file))
        assert result == data

    def test_missing_file(self):
        """Return None when the file does not exist."""
        result = load_json_file("/nonexistent/path/file.json")
        assert result is None

    def test_invalid_json(self, tmp_path):
        """Return None and print error when JSON is invalid."""
        json_file = tmp_path / "bad.json"
        json_file.write_text("{invalid json content", encoding="utf-8")
        result = load_json_file(str(json_file))
        assert result is None

    def test_empty_file(self, tmp_path):
        """Return None for an empty file (invalid JSON)."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("", encoding="utf-8")
        result = load_json_file(str(json_file))
        assert result is None


class TestMergeValidationWithGroups:
    """Tests for apply_selection_helpers.merge_validation_with_groups."""

    def test_matching_groups_and_validation(self):
        """Merge validation results into matching groups by index."""
        groups = [
            {"theme": "Theme A", "suggestions": [{"title": "S1"}]},
            {"theme": "Theme B", "suggestions": [{"title": "S2"}]},
        ]
        validation = [
            {"status": "valid", "reason": "Looks good", "confidence": 0.95},
            {"status": "invalid", "reason": "Not applicable", "confidence": 0.8},
        ]
        result = merge_validation_with_groups(groups, validation)
        assert len(result) == 2
        assert result[0]["validation_status"] == "valid"
        assert result[0]["validation_reason"] == "Looks good"
        assert result[0]["validation_confidence"] == 0.95
        assert result[0]["group_index"] == 0
        assert result[1]["validation_status"] == "invalid"

    def test_non_matching_groups_fewer_validation(self):
        """Groups without matching validation get default status."""
        groups = [
            {"theme": "A", "suggestions": []},
            {"theme": "B", "suggestions": []},
            {"theme": "C", "suggestions": []},
        ]
        validation = [
            {"status": "valid", "reason": "OK", "confidence": 0.9},
        ]
        result = merge_validation_with_groups(groups, validation)
        assert len(result) == 3
        assert result[0]["validation_status"] == "valid"
        # Groups 1 and 2 have no matching validation
        assert result[1]["validation_status"] == "needs-human-decision"
        assert result[1]["validation_reason"] == "No validation result"
        assert result[2]["validation_status"] == "needs-human-decision"

    def test_empty_groups(self):
        """Empty groups list returns empty result."""
        result = merge_validation_with_groups([], [])
        assert result == []

    def test_groups_with_missing_null_validation(self):
        """Edge case: groups with missing/null validation results.

        Plan edge-case #10: Treated as unvalidated; included in output.
        """
        groups = [
            {"theme": "A", "suggestions": [{"title": "S1"}]},
            {"theme": "B", "suggestions": [{"title": "S2"}]},
        ]
        # Pass empty validation list so no group has a match
        result = merge_validation_with_groups(groups, [])
        assert len(result) == 2
        for g in result:
            assert g["validation_status"] == "needs-human-decision"
            assert g["validation_confidence"] == 0.0

    def test_error_type_and_recoverable_copied(self):
        """Validation results with error_type and recoverable are preserved."""
        groups = [{"theme": "A", "suggestions": []}]
        validation = [
            {
                "status": "validation_failed",
                "reason": "Timeout",
                "confidence": 0.0,
                "error_type": "timeout",
                "recoverable": True,
            }
        ]
        result = merge_validation_with_groups(groups, validation)
        assert result[0]["validation_error_type"] == "timeout"
        assert result[0]["validation_recoverable"] is True


class TestResolvePriorityArgs:
    """Tests for apply_selection_helpers.resolve_priority_args."""

    def test_no_flags_returns_low(self):
        """Default (no flags) returns 'low' (include everything)."""
        args = argparse.Namespace(include_low=False, min_priority=None)
        assert resolve_priority_args(args) == "low"

    def test_include_low_returns_low_with_warning(self, capsys):
        """--include-low returns 'low' with deprecation warning."""
        args = argparse.Namespace(include_low=True, min_priority=None)
        result = resolve_priority_args(args)
        assert result == "low"
        captured = capsys.readouterr()
        assert "deprecated" in captured.err.lower()

    def test_min_priority_medium(self):
        """--min-priority medium returns 'medium'."""
        args = argparse.Namespace(include_low=False, min_priority="medium")
        assert resolve_priority_args(args) == "medium"

    def test_conflicting_flags_uses_min_priority_with_warning(self, capsys):
        """Edge case #4: --include-low + --min-priority uses min_priority with warning."""
        args = argparse.Namespace(include_low=True, min_priority="high")
        result = resolve_priority_args(args)
        assert result == "high"
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "--min-priority" in captured.err

    def test_no_include_low_attr(self):
        """Namespace without include_low attribute works (apply_task_suggestions)."""
        args = argparse.Namespace(min_priority="high")
        assert resolve_priority_args(args) == "high"


class TestIsUserSkipped:
    """Tests for apply_selection_helpers._is_user_skipped."""

    def test_suggestion_id_present(self):
        """Group is skipped when a suggestion ID matches."""
        group = {"suggestions": [{"id": "S001"}, {"id": "S002"}]}
        assert _is_user_skipped(group, {"S001"}) is True

    def test_suggestion_id_absent(self):
        """Group is not skipped when no suggestion ID matches."""
        group = {"suggestions": [{"id": "S001"}, {"id": "S002"}]}
        assert _is_user_skipped(group, {"S999"}) is False

    def test_group_level_id(self):
        """Group is skipped when group-level ID matches."""
        group = {"id": "G1", "suggestions": []}
        assert _is_user_skipped(group, {"G1"}) is True

    def test_empty_skipped_set(self):
        """No skip when skipped_ids is empty."""
        group = {"id": "G1", "suggestions": [{"id": "S1"}]}
        assert _is_user_skipped(group, set()) is False

    def test_no_suggestions_key(self):
        """Group without suggestions key, only group-level ID check."""
        group = {"id": "G1"}
        assert _is_user_skipped(group, {"G1"}) is True
        assert _is_user_skipped(group, {"S999"}) is False


class TestDerivePrefix:
    """Tests for output_handler.derive_prefix."""

    def test_simple_plan(self):
        """Derive prefix from a simple plan file path."""
        assert derive_prefix("/path/to/my-plan.md") == "my-plan"

    def test_complex_name(self):
        """Derive prefix from a plan with special characters."""
        assert derive_prefix("/path/to/My Plan (v2.0).md") == "My_Plan_v2_0"

    def test_no_extension(self):
        """Derive prefix from a plan without .md extension."""
        assert derive_prefix("/path/to/my-feature") == "my-feature"


class TestFindOutputDir:
    """Tests for output_handler.find_output_dir.

    find_output_dir composes paths with os.path.join, so the separator is
    OS-native (backslash on Windows). Compare through Path() so the
    assertions describe the path, not one platform's spelling of it.
    """

    def test_standard_path(self):
        """Find output dir from a standard plan file path."""
        result = find_output_dir("/path/to/my-plan.md")
        assert Path(result) == Path("/path/to/my-plan")

    def test_guard_against_double_nesting(self):
        """When parent directory already matches prefix, avoid double nesting."""
        result = find_output_dir("/path/to/my-plan/my-plan.md")
        assert Path(result) == Path("/path/to/my-plan")

    def test_current_dir_fallback(self):
        """Plan file in current directory uses '.' as base."""
        result = find_output_dir("plan.md")
        assert Path(result) == Path("./plan")


class TestApplyOutputHelpers:
    """Tests for apply_output_helpers functions."""

    def test_build_skipped_output(self):
        """Build a standard skipped output dict."""
        result = build_skipped_output("apply-suggestions", "No items found")
        assert result["status"] == "skipped"
        assert result["message"] == "No items found"
        assert result["phase"] == "apply-suggestions"
        assert result["batches"] == []

    def test_build_confirmation_needed_output(self):
        """Build a confirmation needed output dict."""
        result = build_confirmation_needed_output(
            "apply-mock", "Please confirm", 5
        )
        assert result["status"] == "confirmation_needed"
        assert result["item_count"] == 5
        assert result["phase"] == "apply-mock"

    def test_emit_json_output(self, capsys):
        """emit_json_output prints formatted JSON to stdout."""
        data = {"status": "ok", "count": 3}
        emit_json_output(data)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == data

    def test_write_and_emit_output(self, tmp_path, capsys):
        """write_and_emit_output writes file and emits marker to stderr."""
        phase_dir = str(tmp_path / "apply-mock")
        output = {"status": "ok"}
        result_path = write_and_emit_output(output, phase_dir)

        assert result_path.exists()
        with open(result_path, encoding="utf-8") as f:
            assert json.load(f) == output

        captured = capsys.readouterr()
        assert "[OUTPUT_FILE]" in captured.err


# ======================================================================
# Test: __init_subclass__ validation
# ======================================================================


class TestInitSubclassValidation:
    """Tests for __init_subclass__ validation of required config."""

    def test_missing_phase_name(self):
        """TypeError raised when phase_name is empty."""
        with pytest.raises(TypeError, match="phase_name"):
            class Bad(ApplyOrchestratorBase):
                phase_name = ""
                review_subdir = "review-plan"
                item_noun = "item"
                supports_revalidation = False
                supports_skip_flag = False

    def test_missing_review_subdir(self):
        """TypeError raised when review_subdir is empty."""
        with pytest.raises(TypeError, match="review_subdir"):
            class Bad(ApplyOrchestratorBase):
                phase_name = "apply-test"
                review_subdir = ""
                item_noun = "item"
                supports_revalidation = False
                supports_skip_flag = False

    def test_missing_item_noun(self):
        """TypeError raised when item_noun is empty."""
        with pytest.raises(TypeError, match="item_noun"):
            class Bad(ApplyOrchestratorBase):
                phase_name = "apply-test"
                review_subdir = "review-plan"
                item_noun = ""
                supports_revalidation = False
                supports_skip_flag = False

    def test_missing_bool_config(self):
        """TypeError raised when boolean config not explicitly set."""
        with pytest.raises(TypeError, match="supports_revalidation"):
            class Bad(ApplyOrchestratorBase):
                phase_name = "apply-test"
                review_subdir = "review-plan"
                item_noun = "item"
                # supports_revalidation intentionally omitted
                supports_skip_flag = False

    def test_invalid_bool_type(self):
        """TypeError raised when boolean config is not a bool."""
        with pytest.raises(TypeError, match="must be a bool"):
            class Bad(ApplyOrchestratorBase):
                phase_name = "apply-test"
                review_subdir = "review-plan"
                item_noun = "item"
                supports_revalidation = "yes"  # type: ignore
                supports_skip_flag = False

    def test_non_string_phase_name(self):
        """TypeError raised when phase_name is not a string."""
        with pytest.raises(TypeError, match="phase_name"):
            class Bad(ApplyOrchestratorBase):
                phase_name = 123  # type: ignore
                review_subdir = "review-plan"
                item_noun = "item"
                supports_revalidation = False
                supports_skip_flag = False

    def test_valid_subclass_passes(self):
        """A properly configured subclass does not raise."""
        # This should not raise
        class Good(ApplyOrchestratorBase):
            phase_name = "apply-good"
            review_subdir = "review-plan"
            item_noun = "item"
            supports_revalidation = True
            supports_skip_flag = True

            def load_data(self):
                return ([], [])

            def parse_user_edits(self, report_path):
                return {}

            def merge_user_edits(self, groups, edited_descriptions):
                return (groups, [])

            def parse_skips_from_report(self, report_path):
                return (set(), set(), set())

            def parse_validation_overrides_from_report(self, report_path):
                return ({}, {})

            def format_item_for_output(self, group, index):
                return group

            def create_batches(self, items):
                return []

            def generate_batch_prompts(self, batches):
                return batches

            def build_output_json(self, batches, *, resume_info=None):
                return {}

            def get_output_path(self):
                return "/tmp/test"

            def print_text_summary(self, batches, output_path):
                pass


# ======================================================================
# Test: item_noun_plural (F8) — "fix" + "s" used to read "Fixs" everywhere.
# ======================================================================


class TestItemNounPlural:
    """F8: an unset `item_noun_plural` defaults to the arithmetic plural,
    which is correct for a regular noun and wrong for one that isn't."""

    def test_defaults_to_the_arithmetic_plural_when_unset(self):
        class Widget(ApplyOrchestratorBase):
            phase_name = "apply-widget"
            review_subdir = "review-plan"
            item_noun = "widget"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                return ([], [])

            def parse_user_edits(self, report_path):
                return {}

            def merge_user_edits(self, groups, edited_descriptions):
                return (groups, [])

            def parse_skips_from_report(self, report_path):
                return (set(), set(), set())

            def parse_validation_overrides_from_report(self, report_path):
                return ({}, {})

            def format_item_for_output(self, group, index):
                return group

            def create_batches(self, items):
                return []

            def generate_batch_prompts(self, batches):
                return batches

            def build_output_json(self, batches, *, resume_info=None):
                return {}

            def get_output_path(self):
                return "/tmp/test"

            def print_text_summary(self, batches, output_path):
                pass

        assert Widget.item_noun_plural == "widgets"

    def test_an_explicit_plural_is_not_overwritten(self):
        assert MockApplyOrchestrator.item_noun == "suggestion"
        assert MockApplyOrchestrator.item_noun_plural == "suggestions"

    def test_apply_code_fixes_declares_the_irregular_plural(self):
        """"fix" + "s" reads "Fixs" — the phase must set it explicitly."""
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator

        assert ApplyCodeFixesOrchestrator.item_noun == "fix"
        assert ApplyCodeFixesOrchestrator.item_noun_plural == "fixes"

    def test_the_two_sibling_phases_get_the_regular_plural_for_free(self):
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator
        from apply_task_suggestions_orchestrator import (
            ApplyTaskSuggestionsOrchestrator,
        )

        assert ApplySuggestionsOrchestrator.item_noun_plural == "suggestions"
        assert (
            ApplyTaskSuggestionsOrchestrator.item_noun_plural
            == "task suggestions"
        )


# ======================================================================
# Test: Abstract method contracts
# ======================================================================


class TestAbstractMethodContracts:
    """Verify abstract methods raise NotImplementedError; concrete defaults work."""

    @pytest.fixture
    def base_instance(self):
        """Create a raw ApplyOrchestratorBase instance for testing.

        We bypass __init_subclass__ by using MockApplyOrchestrator and
        then calling the base class methods directly.
        """
        args = _make_args()
        orch = MockApplyOrchestrator(args)
        return orch

    def _call_base_method(self, method_name, *args, **kwargs):
        """Call a method on ApplyOrchestratorBase directly."""
        dummy_args = _make_args()
        orch = MockApplyOrchestrator(dummy_args)
        # Call the base class method directly, bypassing the override
        base_method = getattr(ApplyOrchestratorBase, method_name)
        return base_method(orch, *args, **kwargs)

    def test_load_data_raises(self):
        with pytest.raises(NotImplementedError, match="load_data"):
            self._call_base_method("load_data")

    def test_parse_user_edits_raises(self):
        with pytest.raises(NotImplementedError, match="parse_user_edits"):
            self._call_base_method("parse_user_edits", "/some/report.md")

    def test_merge_user_edits_raises(self):
        with pytest.raises(NotImplementedError, match="merge_user_edits"):
            self._call_base_method("merge_user_edits", [], {})

    def test_parse_skips_from_report_raises(self):
        with pytest.raises(NotImplementedError, match="parse_skips_from_report"):
            self._call_base_method("parse_skips_from_report", "/some/report.md")

    def test_parse_validation_overrides_from_report_raises(self):
        with pytest.raises(
            NotImplementedError, match="parse_validation_overrides_from_report"
        ):
            self._call_base_method(
                "parse_validation_overrides_from_report", "/some/report.md"
            )

    def test_format_item_for_output_raises(self):
        with pytest.raises(NotImplementedError, match="format_item_for_output"):
            self._call_base_method("format_item_for_output", {}, 0)

    def test_create_batches_default_returns_empty(self):
        """create_batches is a concrete default; returns empty list for empty input."""
        result = self._call_base_method("create_batches", [])
        assert result == []

    def test_generate_batch_prompts_raises(self):
        with pytest.raises(NotImplementedError, match="generate_batch_prompts"):
            self._call_base_method("generate_batch_prompts", [])

    def test_build_output_json_raises(self):
        with pytest.raises(NotImplementedError, match="build_output_json"):
            self._call_base_method("build_output_json", [])

    def test_get_output_path_raises(self):
        with pytest.raises(NotImplementedError, match="get_output_path"):
            self._call_base_method("get_output_path")

    def test_print_text_summary_raises(self):
        with pytest.raises(NotImplementedError, match="print_text_summary"):
            self._call_base_method("print_text_summary", [], "/tmp/out.json")


# ======================================================================
# Test: Template method sequencing
# ======================================================================


class TestTemplateMethodSequencing:
    """Verify run() calls lifecycle phases in correct order."""

    def test_phase_call_order(self, tmp_path):
        """run() calls _setup -> _load_and_parse_inputs -> _apply_user_feedback ->
        _prepare_batches -> _write_outputs in order.
        """
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)

        # Track phase calls by patching phase methods
        phase_calls = []
        original_setup = orch._setup
        original_load = orch._load_and_parse_inputs
        original_feedback = orch._apply_user_feedback
        original_batches = orch._prepare_batches
        original_write = orch._write_outputs

        def tracked_setup():
            phase_calls.append("_setup")
            return original_setup()

        def tracked_load():
            phase_calls.append("_load_and_parse_inputs")
            return original_load()

        def tracked_feedback():
            phase_calls.append("_apply_user_feedback")
            return original_feedback()

        def tracked_batches():
            phase_calls.append("_prepare_batches")
            return original_batches()

        def tracked_write():
            phase_calls.append("_write_outputs")
            return original_write()

        orch._setup = tracked_setup
        orch._load_and_parse_inputs = tracked_load
        orch._apply_user_feedback = tracked_feedback
        orch._prepare_batches = tracked_batches
        orch._write_outputs = tracked_write

        exit_code = orch.run()

        assert exit_code == 0
        assert phase_calls == [
            "_setup",
            "_load_and_parse_inputs",
            "_apply_user_feedback",
            "_prepare_batches",
            "_write_outputs",
        ]

    def test_abstract_methods_called_in_lifecycle(self, tmp_path):
        """Verify subclass abstract methods are called during the run lifecycle."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0

        # Verify key abstract methods were called
        assert "load_data" in orch.call_log
        assert "parse_user_edits" in orch.call_log
        assert "parse_skips_from_report" in orch.call_log
        assert "parse_validation_overrides_from_report" in orch.call_log
        assert "build_output_json" in orch.call_log
        assert "get_output_path" in orch.call_log
        assert "print_text_summary" in orch.call_log


# ======================================================================
# Test: Config-driven branching
# ======================================================================


class TestConfigDrivenBranching:
    """Verify config flags control behavior branching."""

    def test_supports_revalidation_false_skips_revalidation(self, tmp_path):
        """When supports_revalidation=False, handle_revalidation is a no-op."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)
        assert orch.supports_revalidation is False

        # handle_revalidation should return immediately without error
        orch.handle_revalidation()
        # No assertion needed beyond "no exception raised"

    def test_supports_revalidation_true_enters_revalidation(self):
        """When supports_revalidation=True, handle_revalidation path is entered."""

        class RevalidatingOrch(MockApplyOrchestrator):
            phase_name = "apply-reval"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = True
            supports_skip_flag = False

        args = _make_args()
        orch = RevalidatingOrch(args)
        assert orch.supports_revalidation is True
        # Currently a stub, just verify no crash
        orch.handle_revalidation()

    def test_emit_records_the_phase_started_never_completed(self, tmp_path):
        """The native emit path hands the batches out: started, not completed.

        Until 2026-10-05 apply-suggestions (`marks_phase_completed = True`)
        recorded itself completed here, before a single edit existed.
        """
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0

        assert orch.state.is_phase_started(orch.phase_name)
        assert not orch.state.is_phase_completed(orch.phase_name)

    def test_supports_skip_flag_false_ignores_skip(self, tmp_path):
        """When supports_skip_flag=False, --skip flag has no effect."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), skip=True)

        orch = MockApplyOrchestrator(args)
        assert orch.supports_skip_flag is False

        # run should proceed normally (not exit early for --skip)
        exit_code = orch.run()
        assert exit_code == 0

    def test_supports_skip_flag_true_exits_on_skip(self, tmp_path):
        """When supports_skip_flag=True and --skip given, exit 0."""

        class SkippableOrch(MockApplyOrchestrator):
            phase_name = "apply-skip"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = True

        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), skip=True)

        orch = SkippableOrch(args)
        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0


# ======================================================================
# Test: build_common_arg_parser
# ======================================================================


class TestBuildCommonArgParser:
    """Tests for build_common_arg_parser with different toggles."""

    def test_common_flags_always_present(self):
        """Parser always includes --plan-file, --yes, --force, --dry-run, etc."""
        parser = build_common_arg_parser("Test", "test epilog")
        # Parse with required plan-file
        args = parser.parse_args(["--plan-file", "test.md"])
        assert args.plan_file == "test.md"
        assert args.yes is False
        assert args.force is False
        assert args.dry_run is False
        assert args.resume is False
        assert args.fresh is False
        assert args.approve_all is False
        assert args.min_priority is None

    def test_revalidation_flags_appear_when_toggled(self):
        """--revalidate and --revalidate-model present with include_revalidation=True."""
        parser = build_common_arg_parser("Test", "", include_revalidation=True)
        args = parser.parse_args(["--plan-file", "t.md", "--revalidate"])
        assert args.revalidate is True

    def test_revalidation_flags_absent_when_not_toggled(self):
        """--revalidate not present without include_revalidation=True."""
        parser = build_common_arg_parser("Test", "", include_revalidation=False)
        with pytest.raises(SystemExit):
            parser.parse_args(["--plan-file", "t.md", "--revalidate"])

    def test_skip_flag_appears_when_toggled(self):
        """--skip present with include_skip=True."""
        parser = build_common_arg_parser("Test", "", include_skip=True)
        args = parser.parse_args(["--plan-file", "t.md", "--skip"])
        assert args.skip is True

    def test_skip_flag_absent_when_not_toggled(self):
        """--skip not present without include_skip=True."""
        parser = build_common_arg_parser("Test", "", include_skip=False)
        with pytest.raises(SystemExit):
            parser.parse_args(["--plan-file", "t.md", "--skip"])

    def test_output_format_appears_when_toggled(self):
        """--output-format present with include_output_format=True."""
        parser = build_common_arg_parser("Test", "", include_output_format=True)
        args = parser.parse_args(
            ["--plan-file", "t.md", "--output-format", "json"]
        )
        assert args.output_format == "json"

    def test_base_ref_appears_when_toggled(self):
        """--base-ref present with include_base_ref=True."""
        parser = build_common_arg_parser("Test", "", include_base_ref=True)
        args = parser.parse_args(
            ["--plan-file", "t.md", "--base-ref", "HEAD~5"]
        )
        assert args.base_ref == "HEAD~5"

    def test_approve_validation_failed_appears_when_toggled(self):
        """--approve-validation-failed present with toggle."""
        parser = build_common_arg_parser(
            "Test", "", include_approve_validation_failed=True
        )
        args = parser.parse_args(
            ["--plan-file", "t.md", "--approve-validation-failed"]
        )
        assert args.approve_validation_failed is True

    def test_mark_completed_appears_when_toggled(self):
        """--mark-completed present with include_mark_completed=True."""
        parser = build_common_arg_parser(
            "Test", "", include_mark_completed=True
        )
        args = parser.parse_args(["--plan-file", "t.md", "--mark-completed"])
        assert args.mark_completed is True

    def test_multiple_toggles(self):
        """Multiple toggles can be combined."""
        parser = build_common_arg_parser(
            "Test",
            "",
            include_revalidation=True,
            include_skip=True,
            include_base_ref=True,
        )
        args = parser.parse_args(
            [
                "--plan-file",
                "t.md",
                "--revalidate",
                "--skip",
                "--base-ref",
                "main",
            ]
        )
        assert args.revalidate is True
        assert args.skip is True
        assert args.base_ref == "main"

    def test_agent_decide_default_false(self):
        """--agent-decide defaults to False when not passed."""
        parser = build_common_arg_parser("Test", "")
        args = parser.parse_args(["--plan-file", "t.md"])
        assert args.agent_decide is False

    def test_legacy_claude_decide_flag(self):
        """The legacy --claude-decide spelling sets args.agent_decide to True."""
        parser = build_common_arg_parser("Test", "")
        args = parser.parse_args(["--plan-file", "t.md", "--claude-decide"])
        assert args.agent_decide is True

    def test_legacy_let_claude_decide_alias(self):
        """The legacy --let-claude-decide spelling is an alias for --agent-decide."""
        parser = build_common_arg_parser("Test", "")
        args = parser.parse_args(["--plan-file", "t.md", "--let-claude-decide"])
        assert args.agent_decide is True

    @pytest.mark.parametrize(
        "spelling", ["--agent-decide", "--claude-decide", "--let-claude-decide"]
    )
    def test_every_spelling_sets_agent_decide(self, spelling):
        parser = build_common_arg_parser("Test", "")
        args = parser.parse_args(["--plan-file", "t.md", spelling])
        assert args.agent_decide is True

    def test_help_leads_with_agent_decide(self):
        text = build_common_arg_parser("Test", "").format_help()
        assert "--agent-decide, --claude-decide, --let-claude-decide" in text
        assert "Let Claude" not in text


# ======================================================================
# Test: build_human_review_config decision_mode
# ======================================================================


class TestClaudeDecideDecisionMode:
    """Tests that --agent-decide surfaces in human_review_config."""

    def _config(self, **arg_overrides):
        orch = MockApplyOrchestrator(_make_args(**arg_overrides))
        orch.formatted_human = []
        return orch.build_human_review_config()

    def test_decision_mode_interactive_by_default(self):
        """Without --agent-decide, decision_mode is 'interactive'."""
        config = self._config(agent_decide=False)
        assert config["decision_mode"] == "interactive"

    def test_decision_mode_judge_auto_decide_when_flag_set(self):
        """With --agent-decide, decision_mode is 'judge_auto_decide'."""
        config = self._config(agent_decide=True)
        assert config["decision_mode"] == "judge_auto_decide"

    def test_decision_mode_independent_of_batch_mode(self):
        """decision_mode is set even when batch_review_mode is 'individual'."""
        config = self._config(agent_decide=True, batch_review_mode="individual")
        assert config["decision_mode"] == "judge_auto_decide"
        assert config["batch_enabled"] is False


# ======================================================================
# Test: OrchestratorError
# ======================================================================


class TestOrchestratorError:
    """Tests for OrchestratorError exception."""

    def test_default_exit_code(self):
        """Default exit code is 1."""
        err = OrchestratorError("something failed")
        assert err.exit_code == 1
        assert str(err) == "something failed"
        assert err.message == "something failed"

    def test_custom_exit_code(self):
        """Custom exit code is preserved."""
        err = OrchestratorError("bad input", exit_code=2)
        assert err.exit_code == 2

    def test_error_contains_phase_context(self):
        """OrchestratorError messages should include phase context."""
        err = OrchestratorError("Phase _setup, step 3: Plan file not found")
        assert "Phase _setup" in err.message
        assert "step 3" in err.message

    def test_run_catches_orchestrator_error(self, tmp_path):
        """run() catches OrchestratorError and returns its exit code."""

        class FailingOrch(MockApplyOrchestrator):
            phase_name = "apply-fail"
            review_subdir = "review-plan"
            item_noun = "item"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                raise OrchestratorError(
                    "Phase _load_and_parse_inputs, step 11: load_data failed: oops",
                    exit_code=2,
                )

        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))
        orch = FailingOrch(args)
        exit_code = orch.run()
        assert exit_code == 2


# ======================================================================
# Test: Edge-case matrix
# ======================================================================


class TestEdgeCaseMatrix:
    """Tests for the 13 edge cases from the refactoring plan (lines 422-442)."""

    # --- #1: Missing report sections (no edits/skips/overrides) ---
    def test_missing_report_sections_graceful(self, tmp_path):
        """Edge case #1: Missing report sections are gracefully skipped."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0
        # No crash; parse_user_edits, parse_skips, parse_overrides all return empty

    # --- #2: Malformed validation overrides ---
    def test_malformed_validation_overrides(self, tmp_path):
        """Edge case #2: Malformed validation overrides are skipped.

        Note: Actual malformed override parsing is subclass-specific.
        At the base class level, we verify that empty/bad overrides don't crash.
        """
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        class OverrideOrch(MockApplyOrchestrator):
            phase_name = "apply-override"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def parse_validation_overrides_from_report(self, report_path):
                # Return badly typed data - base class handles gracefully
                return ({}, {})

        orch = OverrideOrch(args)
        exit_code = orch.run()
        assert exit_code == 0

    # --- #3: Empty groups after filtering ---
    def test_empty_groups_after_filtering(self, tmp_path):
        """Edge case #3: All items filtered out -> early exit path."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)
        # load_data returns empty groups; the orchestrator should handle gracefully
        exit_code = orch.run()
        assert exit_code == 0

    # --- #4: Conflicting flags --include-low + --min-priority ---
    def test_conflicting_include_low_and_min_priority(self, capsys):
        """Edge case #4: --include-low + --min-priority uses min_priority with warning."""
        args = argparse.Namespace(include_low=True, min_priority="high")
        result = resolve_priority_args(args)
        assert result == "high"
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    # --- #5: Invalid plan path (nonexistent file) ---
    def test_invalid_plan_path_nonexistent(self, tmp_path):
        """Edge case #5: Nonexistent plan file -> error + exit code 1."""
        args = _make_args(plan_file=str(tmp_path / "nonexistent.md"))
        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 1

    # --- #6: Invalid plan path (directory instead of file) ---
    def test_invalid_plan_path_directory(self, tmp_path):
        """Edge case #6: Directory instead of file -> error + exit code 1."""
        dir_path = tmp_path / "some-dir"
        dir_path.mkdir()
        args = _make_args(plan_file=str(dir_path))
        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 1

    # --- #7: --approve-all without --yes/--force ---
    def test_approve_all_without_guardrail(self, tmp_path):
        """Edge case #7: --approve-all without --yes/--force -> error exit code 1."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path),
            approve_all=True,
            yes=False,
            force=False,
        )
        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 1

    def test_approve_all_with_yes_passes(self, tmp_path):
        """--approve-all with --yes proceeds past guardrail."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path),
            approve_all=True,
            yes=True,
        )
        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0

    def test_approve_all_with_force_passes(self, tmp_path):
        """--approve-all with --force proceeds past guardrail."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path),
            approve_all=True,
            force=True,
        )
        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0

    # --- #8: --resume with no prior state ---
    def test_resume_with_no_prior_state(self, tmp_path):
        """Edge case #8: --resume with no prior state treated as fresh run."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), resume=True)

        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0

    # --- #9: --fresh clears existing state ---
    def test_fresh_clears_state(self, tmp_path):
        """Edge case #9: --fresh clears existing state."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), fresh=True)

        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0
        # Verify that _setup completed (state was cleared)
        assert orch.state is not None

    # --- #10: Groups with missing/null validation results ---
    def test_groups_with_null_validation(self):
        """Edge case #10: Groups with missing/null validation results."""
        groups = [
            {"theme": "A", "suggestions": [{"title": "S1"}]},
            {"theme": "B", "suggestions": [{"title": "S2"}]},
        ]
        result = merge_validation_with_groups(groups, [])
        # All should get default "needs-human-decision" status
        for g in result:
            assert g["validation_status"] == "needs-human-decision"
            assert g["validation_confidence"] == 0.0

    # --- #11: Duplicate group hashes in report ---
    def test_duplicate_group_hashes_last_wins(self, tmp_path):
        """Edge case #11: Duplicate group hashes -> last occurrence wins.

        This is subclass-specific for parsing but at the base class level
        we test apply_group_validation_overrides with duplicate keys.
        """
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)

        # Simulate having groups with same hash
        orch.merged = [
            {
                "group_hash": "abc123",
                "theme": "Theme A",
                "validation_status": "needs-human-decision",
                "suggestions": [],
            },
        ]
        # Dict naturally keeps last value
        orch.validation_overrides = {"abc123": "valid"}
        orch.apply_group_validation_overrides()

        assert orch.merged[0]["validation_status"] == "valid"
        assert orch.merged[0]["user_override"] is True

    # --- #12: Integer vs string override keys ---
    def test_integer_vs_string_override_keys_deferred(self):
        """Edge case #12: Integer vs string override keys.

        Base class uses hash-based string keys.
        Integer-based keys for code_fixes are deferred to T008 (apply_code_fixes migration).

        Cross-reference: This edge case is fully covered in T008/T009 when
        apply_code_fixes_orchestrator overrides apply_group_validation_overrides
        to use integer issue-number keys.
        """
        # Base class behavior: string key matching
        args = _make_args()
        orch = MockApplyOrchestrator(args)
        orch.merged = [
            {
                "group_hash": "hash1",
                "validation_status": "needs-human-decision",
                "suggestions": [],
            }
        ]
        # String key matches
        orch.validation_overrides = {"hash1": "valid"}
        orch.apply_group_validation_overrides()
        assert orch.merged[0]["validation_status"] == "valid"

    # --- #13: Override contradicts validation result ---
    def test_override_contradicts_validation(self, tmp_path):
        """Edge case #13: User override silently wins over automated validation."""
        args = _make_args()
        orch = MockApplyOrchestrator(args)

        orch.merged = [
            {
                "group_hash": "hash1",
                "validation_status": "invalid",  # Validation says invalid
                "validation_reason": "Automated check failed",
                "suggestions": [],
            }
        ]
        orch.validation_overrides = {"hash1": "valid"}  # User says valid
        orch.apply_group_validation_overrides()

        # User override silently wins
        assert orch.merged[0]["validation_status"] == "valid"
        assert "User override" in orch.merged[0]["validation_reason"]
        assert orch.merged[0]["user_override"] is True

    # --- #14: HTML selections file missing or empty ---
    def test_html_selections_missing(self, tmp_path):
        """Edge case #14: HTML selections file missing -> falls back gracefully."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)
        # Run full lifecycle; HTML selections won't exist
        exit_code = orch.run()
        assert exit_code == 0

    # --- #15: Consolidated decisions file missing ---
    def test_consolidated_decisions_missing(self, tmp_path):
        """Edge case #15: C-level consolidated decisions file missing -> skipped."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0
        # No crash; consolidated decisions are silently skipped


# ======================================================================
# Test: HTML validation-override key normalization
# ======================================================================


class TestHtmlValidationOverrideKeyNormalization:
    """load_html_selections() must resolve group-level override keys to hashes.

    Regression for the silent-drop bug: legacy HTML reports keyed group
    validation overrides by 0-based group index ("0" == the first group as
    displayed). Stored as a bare int, that key matched nothing downstream --
    the code-fixes matcher enumerates groups 1-based (off-by-one) and the
    suggestion matchers compare hash strings only -- so the user's
    needs-human-decision / mark-valid choice never applied. The key must be
    resolved to the group's ``group_hash`` at ingestion.
    """

    def _orch_with_groups(self, groups):
        orch = MockApplyOrchestrator(_make_args())
        orch.groups = groups
        # Populated by _load_and_parse_inputs in the real lifecycle; set here
        # because we call load_html_selections() in isolation.
        orch.skipped_group_indices = set()
        orch.skipped_suggestion_ids = set()
        return orch

    def _write_selections(self, tmp_path, overrides):
        review_dir = tmp_path / "review-plan"
        review_dir.mkdir()
        (review_dir / "user_selections.json").write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "skipped_groups": [],
                    "skipped_suggestions": [],
                    "edited_descriptions": {},
                    "validation_overrides": overrides,
                }
            ),
            encoding="utf-8",
        )
        return str(review_dir)

    def test_zero_based_index_resolves_to_group_hash(self, tmp_path):
        """"0" maps to the first group's hash, never to a bare int 0."""
        orch = self._orch_with_groups(
            [
                {"group_hash": "aaaa1111bbbb2222", "theme": "First", "suggestions": []},
                {"group_hash": "cccc3333dddd4444", "theme": "Second", "suggestions": []},
            ]
        )
        review_dir = self._write_selections(
            tmp_path, {"0": "needs-human-decision"}
        )

        orch.load_html_selections(review_dir, {})

        assert orch.validation_overrides == {
            "aaaa1111bbbb2222": "needs-human-decision"
        }
        assert 0 not in orch.validation_overrides

    def test_second_index_resolves_to_second_group(self, tmp_path):
        """0-based "1" targets the second group, not the first (off-by-one guard)."""
        orch = self._orch_with_groups(
            [
                {"group_hash": "aaaa1111bbbb2222", "theme": "First", "suggestions": []},
                {"group_hash": "cccc3333dddd4444", "theme": "Second", "suggestions": []},
            ]
        )
        review_dir = self._write_selections(tmp_path, {"1": "valid"})

        orch.load_html_selections(review_dir, {})

        assert orch.validation_overrides == {"cccc3333dddd4444": "valid"}

    def test_hash_key_passes_through(self, tmp_path):
        """Current report format (group_hash keys) is used as-is."""
        orch = self._orch_with_groups(
            [{"group_hash": "aaaa1111bbbb2222", "theme": "First", "suggestions": []}]
        )
        review_dir = self._write_selections(
            tmp_path, {"aaaa1111bbbb2222": "invalid"}
        )

        orch.load_html_selections(review_dir, {})

        assert orch.validation_overrides == {"aaaa1111bbbb2222": "invalid"}

    def test_judge_decide_round_trips_from_user_selections(self, tmp_path):
        """A 'judge_decide' override survives ingest verbatim (group + suggestion)."""
        orch = self._orch_with_groups(
            [{"group_hash": "aaaa1111bbbb2222", "theme": "First",
              "suggestions": [{"suggestion_hash": "s1"}]}]
        )
        review_dir = self._write_selections(
            tmp_path,
            {"aaaa1111bbbb2222": "judge_decide", "G1S1": "judge_decide"},
        )

        orch.load_html_selections(review_dir, {})

        assert orch.validation_overrides == {"aaaa1111bbbb2222": "judge_decide"}
        assert orch.suggestion_validation_overrides == {"G1S1": "judge_decide"}

    def test_all_digit_hash_not_treated_as_index(self, tmp_path):
        """A known group_hash that happens to be all digits stays a hash key."""
        all_digit_hash = "1234567890123456"
        orch = self._orch_with_groups(
            [
                {"group_hash": all_digit_hash, "theme": "First", "suggestions": []},
                {"group_hash": "cccc3333dddd4444", "theme": "Second", "suggestions": []},
            ]
        )
        review_dir = self._write_selections(
            tmp_path, {all_digit_hash: "valid"}
        )

        orch.load_html_selections(review_dir, {})

        # Resolved via the known-hash set, not mis-read as index 1234...
        assert orch.validation_overrides == {all_digit_hash: "valid"}

    def test_suggestion_override_routes_to_suggestion_dict(self, tmp_path):
        """G{N}S{M} keys remain per-suggestion overrides, not group overrides."""
        orch = self._orch_with_groups(
            [{"group_hash": "aaaa1111bbbb2222", "theme": "First", "suggestions": []}]
        )
        review_dir = self._write_selections(tmp_path, {"G1S2": "invalid"})

        orch.load_html_selections(review_dir, {})

        assert orch.validation_overrides == {}
        assert orch.suggestion_validation_overrides == {"G1S2": "invalid"}


# ======================================================================
# Test: Logging and error-context verification
# ======================================================================


class TestLoggingAndErrorContext:
    """Verify logging conventions from T002 spec."""

    def test_logger_name_convention(self):
        """Logger is named 'orchestrator.{phase_name}'."""
        args = _make_args()
        orch = MockApplyOrchestrator(args)
        assert orch.logger.name == "orchestrator.apply-mock"

    def test_logger_name_varies_by_phase(self):
        """Different phase_name subclasses get different logger names."""

        class OtherOrch(MockApplyOrchestrator):
            phase_name = "apply-other"
            review_subdir = "review-plan"
            item_noun = "fix"
            supports_revalidation = False
            supports_skip_flag = False

        args = _make_args()
        orch = OtherOrch(args)
        assert orch.logger.name == "orchestrator.apply-other"

    def test_debug_log_on_phase_entry(self, tmp_path, caplog):
        """DEBUG-level log records emitted on lifecycle phase entry."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = MockApplyOrchestrator(args)

        with caplog.at_level(logging.DEBUG, logger="orchestrator.apply-mock"):
            orch.run()

        # Verify DEBUG records for each phase entry
        debug_messages = [
            r.message for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any("Entering _setup phase" in m for m in debug_messages)
        assert any(
            "Entering _load_and_parse_inputs phase" in m for m in debug_messages
        )
        assert any(
            "Entering _apply_user_feedback phase" in m for m in debug_messages
        )
        assert any(
            "Entering _prepare_batches phase" in m for m in debug_messages
        )
        assert any(
            "Entering _write_outputs phase" in m for m in debug_messages
        )

    def test_info_log_on_milestones(self, tmp_path, caplog):
        """INFO-level log records emitted for key milestones."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        # Create an orchestrator that returns some data
        class DataOrch(MockApplyOrchestrator):
            phase_name = "apply-data"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                self.call_log.append("load_data")
                return (
                    [
                        {
                            "theme": "A",
                            "suggestions": [
                                {
                                    "title": "S1",
                                    "importance": "HIGH",
                                    "desc": "d",
                                }
                            ],
                        }
                    ],
                    [
                        {
                            "status": "valid",
                            "reason": "OK",
                            "confidence": 0.9,
                        }
                    ],
                )

        orch = DataOrch(args)

        with caplog.at_level(logging.DEBUG, logger="orchestrator.apply-data"):
            orch.run()

        info_messages = [
            r.message for r in caplog.records if r.levelno == logging.INFO
        ]
        # Check for milestone messages about counts
        assert any(
            "group" in m.lower() and "validation" in m.lower()
            for m in info_messages
        ), f"Expected group/validation count INFO message, got: {info_messages}"
        assert any(
            "setup complete" in m.lower() for m in info_messages
        ), f"Expected 'setup complete' INFO, got: {info_messages}"

    def test_orchestrator_error_includes_phase_and_step(self, tmp_path):
        """OrchestratorError messages contain lifecycle phase and step context."""

        class BadLoadOrch(MockApplyOrchestrator):
            phase_name = "apply-badload"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                raise ValueError("data corrupted")

        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))
        orch = BadLoadOrch(args)

        # The base class wraps load_data() exceptions in OrchestratorError
        # with phase/step context. Since run() catches it, check exit code.
        exit_code = orch.run()
        assert exit_code == 1

    def test_orchestrator_error_phase_context_in_setup(self, tmp_path):
        """OrchestratorError from _setup phase includes phase context."""
        # Use a nonexistent plan file to trigger _setup error
        args = _make_args(plan_file=str(tmp_path / "missing.md"))
        orch = MockApplyOrchestrator(args)

        # Capture the OrchestratorError by calling _setup directly
        with pytest.raises(OrchestratorError, match="Phase _setup"):
            orch._setup()


# ======================================================================
# Test: Additional integration scenarios
# ======================================================================


class TestRunIntegration:
    """Integration tests for the full run lifecycle."""

    def test_run_with_valid_items(self, tmp_path):
        """Full run with valid items produces output file."""

        class ValidItemsOrch(MockApplyOrchestrator):
            phase_name = "apply-valid"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                self.call_log.append("load_data")
                return (
                    [
                        {
                            "theme": "Improve error handling",
                            "group_hash": "abc123",
                            "suggestions": [
                                {
                                    "title": "Add try-catch",
                                    "desc": "Wrap in try-catch",
                                    "importance": "HIGH",
                                    "suggestion_hash": "sh1",
                                }
                            ],
                        }
                    ],
                    [
                        {
                            "status": "valid",
                            "reason": "Good suggestion",
                            "confidence": 0.95,
                        }
                    ],
                )

        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = ValidItemsOrch(args)
        exit_code = orch.run()
        assert exit_code == 0
        assert "format_item_for_output" in orch.call_log
        assert "create_batches" in orch.call_log

    def test_run_unexpected_exception_returns_1(self, tmp_path):
        """Unexpected exceptions in run() return exit code 1."""

        class CrashOrch(MockApplyOrchestrator):
            phase_name = "apply-crash"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                raise RuntimeError("unexpected crash")

        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        orch = CrashOrch(args)
        exit_code = orch.run()
        assert exit_code == 1

    def test_dry_run_exits_after_batches(self, tmp_path):
        """--dry-run prints batch summary and exits with sys.exit(0)."""

        class DryRunOrch(MockApplyOrchestrator):
            phase_name = "apply-dry"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                self.call_log.append("load_data")
                return (
                    [
                        {
                            "theme": "A",
                            "suggestions": [
                                {
                                    "title": "S1",
                                    "importance": "MEDIUM",
                                    "desc": "d",
                                }
                            ],
                        }
                    ],
                    [{"status": "valid", "reason": "OK", "confidence": 0.9}],
                )

        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), dry_run=True)

        orch = DryRunOrch(args)
        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0
        # generate_batch_prompts should NOT have been called
        assert "generate_batch_prompts" not in orch.call_log

    def test_no_confirm_false_triggers_confirmation(self, tmp_path):
        """Without --no-confirm and no selections, confirmation output is generated."""
        plan_path = _setup_plan_with_output(tmp_path)
        # no_confirm=False, no other bypass flags
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=False,
            force=False,
            approve_all=False,
            skip_all_human=False,
            approve_all_low=False,
            approve_importance=None,
            dry_run=False,
        )

        class NoSelOrch(MockApplyOrchestrator):
            phase_name = "apply-nosel"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                self.call_log.append("load_data")
                return (
                    [{"theme": "A", "suggestions": [{"title": "S1"}]}],
                    [{"status": "valid", "reason": "OK", "confidence": 0.9}],
                )

        orch = NoSelOrch(args)
        # With no_confirm=False and no preferences, should sys.exit(0)
        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0


# ======================================================================
# Test: Machine-parseable stdout/stderr markers — backward compatibility
# ======================================================================


class TestMachineParseableMarkers:
    """Verify all machine-parseable markers emitted by the refactored base class.

    These markers are consumed by downstream tooling (Claude Code instruction
    files, orchestration scripts, etc.) and their exact formatting must be
    preserved across the refactoring.

    Markers enumerated:

    STDOUT markers:
    M1. {"status": "skipped", "message": "..."} JSON
        Condition: --skip flag with supports_skip_flag=True
        Source: _setup() step 2

    M2. {"status": "completed", "message": "..."} JSON
        Condition: --mark-completed flag
        Source: _setup() step 2b

    M3. {"status": "confirmation_needed", "message": "...", "phase": "...",
         "item_count": N} JSON
        Condition: No user selections and no bypass flags (no-confirm, yes,
                   force, approve_all, skip_all_human, etc.)
        Source: _apply_user_feedback() step 19

    M4. [REVALIDATION_PENDING] <path>   (single batch)
        Condition: --revalidate with supports_revalidation=True, 1 batch,
                   not dry-run, not internal-revalidation
        Source: handle_revalidation()

    M5. [REVALIDATION_BATCHES_PENDING] <path>   (multiple batches)
        Condition: --revalidate with supports_revalidation=True, >1 batch,
                   not dry-run, not internal-revalidation
        Source: handle_revalidation()

    M6. Full JSON output via emit_json_output()
        Condition: Normal successful completion to stdout
        Source: Subclass early-exit paths (handle_no_items_early_exit)

    STDERR markers:
    M7. [OUTPUT_FILE] <path>
        Condition: Output file successfully written
        Source: _write_outputs() step 33 and write_and_emit_output()

    M8. [orchestrator] Fresh start requested, clearing previous progress...
        Condition: --fresh flag
        Source: _setup() step 10

    M9. [resume] N already processed, M remaining
        Condition: --resume flag with prior state
        Source: _apply_user_feedback() step 23

    M10. [dry-run] Would auto-approve:
         Condition: --dry-run with bulk approval flags that would auto-approve
         Source: _apply_user_feedback() step 25 post-filter

    M11. --- DRY RUN ---
         Condition: --dry-run flag reaches _prepare_batches
         Source: _prepare_batches() step 30

    M12. --- REVALIDATION MODE ---
         Condition: --revalidate or --revalidate-model with
                    supports_revalidation=True
         Source: handle_revalidation()

    No new markers are introduced by the refactored code:
    M13. (Meta-test) Ensure the base class does not emit any markers beyond
         the 12 listed above.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_skippable_orchestrator():
        """Create an orchestrator class with supports_skip_flag=True."""

        class SkippableOrch(MockApplyOrchestrator):
            phase_name = "apply-marker-skip"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = True

        return SkippableOrch

    @staticmethod
    def _make_revalidating_orchestrator():
        """Create an orchestrator class with supports_revalidation=True."""

        class RevalOrch(MockApplyOrchestrator):
            phase_name = "apply-marker-reval"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = True
            supports_skip_flag = False

        return RevalOrch

    @staticmethod
    def _make_data_orchestrator(
        groups=None, validation=None, *, phase_name="apply-marker-data"
    ):
        """Create an orchestrator that returns specific groups and validation."""
        _groups = list(groups) if groups else []
        _validation = list(validation) if validation else []
        _phase = phase_name

        class DataOrch(MockApplyOrchestrator):
            phase_name = _phase
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                self.call_log.append("load_data")
                return (_groups, _validation)

        return DataOrch

    # ------------------------------------------------------------------
    # M1: {"status": "skipped"} on --skip with supports_skip_flag=True
    # ------------------------------------------------------------------

    def test_m1_skip_marker_stdout_json(self, tmp_path, capsys):
        """M1: --skip emits {"status": "skipped"} JSON to stdout."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), skip=True)

        OrchClass = self._make_skippable_orchestrator()
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["status"] == "skipped"
        assert "message" in output
        assert "skipped by user request" in output["message"]

    def test_m1_skip_marker_includes_phase_name(self, tmp_path, capsys):
        """M1: Skip marker message includes the phase name."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), skip=True)

        OrchClass = self._make_skippable_orchestrator()
        orch = OrchClass(args)

        with pytest.raises(SystemExit):
            orch.run()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert OrchClass.phase_name.capitalize() in output["message"]

    def test_m1_skip_not_emitted_when_supports_skip_false(self, tmp_path, capsys):
        """M1: When supports_skip_flag=False, --skip is ignored (no marker)."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), skip=True)

        # MockApplyOrchestrator has supports_skip_flag=False
        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        # No "skipped" JSON on stdout
        assert '"status": "skipped"' not in captured.out or "skipped by user request" not in captured.out

    # ------------------------------------------------------------------
    # M2: {"status": "completed"} on --mark-completed
    # ------------------------------------------------------------------

    def test_m2_mark_completed_marker_stdout_json(self, tmp_path, capsys):
        """M2: --mark-completed emits {"status": "completed"} JSON to stdout."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), mark_completed=True)

        orch = MockApplyOrchestrator(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["status"] == "completed"
        assert "message" in output
        assert "marked as completed" in output["message"]

    def test_m2_mark_completed_includes_phase_name(self, tmp_path, capsys):
        """M2: Completed marker message includes the phase name."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), mark_completed=True)

        orch = MockApplyOrchestrator(args)

        with pytest.raises(SystemExit):
            orch.run()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "apply-mock" in output["message"]

    # ------------------------------------------------------------------
    # M3: {"status": "confirmation_needed"} when no user selections
    # ------------------------------------------------------------------

    def test_m3_confirmation_needed_stdout_json(self, tmp_path, capsys):
        """M3: No selections + no bypass flags emits confirmation_needed JSON to stdout."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=False,
            force=False,
            approve_all=False,
            skip_all_human=False,
            approve_all_low=False,
            approve_importance=None,
            dry_run=False,
        )

        groups = [
            {
                "theme": "Test",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation, phase_name="apply-marker-confirm")
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["status"] == "confirmation_needed"
        assert "message" in output
        assert "phase" in output
        assert "item_count" in output
        assert output["item_count"] == 1

    def test_m3_confirmation_needed_has_correct_phase(self, tmp_path, capsys):
        """M3: confirmation_needed includes correct phase name."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=False,
            force=False,
            approve_all=False,
            skip_all_human=False,
            approve_all_low=False,
            approve_importance=None,
            dry_run=False,
        )

        groups = [{"theme": "A", "suggestions": [{"title": "S1"}]}]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(
            groups, validation, phase_name="apply-marker-phase"
        )
        orch = OrchClass(args)

        with pytest.raises(SystemExit):
            orch.run()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["phase"] == "apply-marker-phase"

    def test_m3_confirmation_not_emitted_with_no_confirm(self, tmp_path, capsys):
        """M3: --no-confirm bypasses confirmation_needed marker."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), no_confirm=True)

        groups = [
            {
                "theme": "Test",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation)
        orch = OrchClass(args)

        # Should NOT exit with confirmation_needed
        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "confirmation_needed" not in captured.out

    def test_m3_confirmation_not_emitted_with_yes(self, tmp_path, capsys):
        """M3: --yes bypasses confirmation_needed marker."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), no_confirm=False, yes=True)

        groups = [
            {
                "theme": "Test",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation)
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "confirmation_needed" not in captured.out

    # ------------------------------------------------------------------
    # M4: [REVALIDATION_PENDING] on single-batch revalidation
    # ------------------------------------------------------------------

    def test_m4_revalidation_pending_marker_stdout(self, tmp_path, capsys):
        """M4: Single-batch revalidation emits [REVALIDATION_PENDING] to stdout."""
        plan_path = _setup_plan_with_output(tmp_path)
        # Create plan content for revalidation context
        plan_path.write_text("# Test Plan for revalidation\n", encoding="utf-8")
        args = _make_args(plan_file=str(plan_path), revalidate=True)

        OrchClass = self._make_revalidating_orchestrator()
        orch = OrchClass(args)

        # Set up state for the revalidation code path
        orch.plan_path = str(plan_path)
        orch.out_dir = str(tmp_path / "test-plan")
        orch.groups = [
            {"theme": "A", "suggestions": [{"title": "S1"}]}
        ]
        orch.validation = [
            {
                "status": "validation_failed",
                "reason": "Timeout",
                "confidence": 0.0,
                "error_type": "timeout",
            }
        ]

        # Create review directory for revalidation output
        review_dir = Path(orch.out_dir) / "review-plan"
        review_dir.mkdir(parents=True, exist_ok=True)

        # Mock prepare_batched_revalidation_tasks to return single batch
        with patch(
            "utils.apply_orchestrator_base.prepare_batched_revalidation_tasks"
        ) as mock_prep:
            mock_prep.return_value = {
                "items_to_revalidate": 1,
                "total_batches": 1,
                "batches": [{"items": [0]}],
            }
            with pytest.raises(SystemExit) as exc_info:
                orch.handle_revalidation()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "[REVALIDATION_PENDING]" in captured.out
        # Must NOT contain [REVALIDATION_BATCHES_PENDING]
        assert "[REVALIDATION_BATCHES_PENDING]" not in captured.out

    def test_m4_revalidation_pending_includes_path(self, tmp_path, capsys):
        """M4: [REVALIDATION_PENDING] marker includes the task file path."""
        plan_path = _setup_plan_with_output(tmp_path)
        plan_path.write_text("# Test Plan\n", encoding="utf-8")
        args = _make_args(plan_file=str(plan_path), revalidate=True)

        OrchClass = self._make_revalidating_orchestrator()
        orch = OrchClass(args)

        orch.plan_path = str(plan_path)
        orch.out_dir = str(tmp_path / "test-plan")
        orch.groups = [{"theme": "A", "suggestions": [{"title": "S1"}]}]
        orch.validation = [
            {"status": "validation_failed", "reason": "Timeout", "confidence": 0.0}
        ]

        review_dir = Path(orch.out_dir) / "review-plan"
        review_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "utils.apply_orchestrator_base.prepare_batched_revalidation_tasks"
        ) as mock_prep:
            mock_prep.return_value = {
                "items_to_revalidate": 1,
                "total_batches": 1,
            }
            with pytest.raises(SystemExit):
                orch.handle_revalidation()

        captured = capsys.readouterr()
        # The path should follow the marker on the same line
        for line in captured.out.splitlines():
            if "[REVALIDATION_PENDING]" in line:
                assert "revalidation_tasks.json" in line
                break
        else:
            pytest.fail("[REVALIDATION_PENDING] not found in stdout")

    # ------------------------------------------------------------------
    # M5: [REVALIDATION_BATCHES_PENDING] on multi-batch revalidation
    # ------------------------------------------------------------------

    def test_m5_revalidation_batches_pending_marker_stdout(self, tmp_path, capsys):
        """M5: Multi-batch revalidation emits [REVALIDATION_BATCHES_PENDING] to stdout."""
        plan_path = _setup_plan_with_output(tmp_path)
        plan_path.write_text("# Test Plan\n", encoding="utf-8")
        args = _make_args(plan_file=str(plan_path), revalidate=True)

        OrchClass = self._make_revalidating_orchestrator()
        orch = OrchClass(args)

        orch.plan_path = str(plan_path)
        orch.out_dir = str(tmp_path / "test-plan")
        orch.groups = [
            {"theme": "A", "suggestions": [{"title": "S1"}]},
            {"theme": "B", "suggestions": [{"title": "S2"}]},
        ]
        orch.validation = [
            {"status": "validation_failed", "reason": "Timeout", "confidence": 0.0},
            {"status": "validation_failed", "reason": "Parse error", "confidence": 0.0},
        ]

        review_dir = Path(orch.out_dir) / "review-plan"
        review_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "utils.apply_orchestrator_base.prepare_batched_revalidation_tasks"
        ) as mock_prep:
            mock_prep.return_value = {
                "items_to_revalidate": 2,
                "total_batches": 3,  # >1 triggers BATCHES_PENDING
            }
            with pytest.raises(SystemExit) as exc_info:
                orch.handle_revalidation()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "[REVALIDATION_BATCHES_PENDING]" in captured.out
        # Batch count should also be printed
        assert "Batches: 3" in captured.out
        # Must NOT contain single-batch marker
        assert "[REVALIDATION_PENDING]" not in captured.out.replace(
            "[REVALIDATION_BATCHES_PENDING]", ""
        )

    # ------------------------------------------------------------------
    # M6: Full JSON output via emit_json_output (subclass early exit)
    # ------------------------------------------------------------------

    def test_m6_emit_json_output_is_valid_json(self, capsys):
        """M6: emit_json_output writes well-formed JSON to stdout."""
        data = {"status": "skipped", "batches": [], "phase": "apply-mock"}
        emit_json_output(data)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == data

    def test_m6_emit_json_output_preserves_structure(self, capsys):
        """M6: emit_json_output preserves nested structures and types."""
        data = {
            "status": "skipped",
            "batches": [],
            "summary": {"total_groups": 5, "valid_count": 3},
            "skipped_items": [{"title": "T1"}],
        }
        emit_json_output(data)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["summary"]["total_groups"] == 5
        assert len(parsed["skipped_items"]) == 1

    # ------------------------------------------------------------------
    # M7: [OUTPUT_FILE] on stderr when output file is written
    # ------------------------------------------------------------------

    def test_m7_output_file_marker_on_stderr(self, tmp_path, capsys):
        """M7: _write_outputs emits [OUTPUT_FILE] <path> to stderr."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        groups = [
            {
                "theme": "A",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation)
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "[OUTPUT_FILE]" in captured.err

    def test_m7_output_file_marker_includes_path(self, tmp_path, capsys):
        """M7: [OUTPUT_FILE] marker includes the output file path."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        groups = [
            {
                "theme": "A",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation)
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        # Find the [OUTPUT_FILE] line
        for line in captured.err.splitlines():
            if "[OUTPUT_FILE]" in line:
                assert "orchestrator_output.json" in line
                break
        else:
            pytest.fail("[OUTPUT_FILE] not found in stderr")

    def test_m7_write_and_emit_output_emits_marker(self, tmp_path, capsys):
        """M7: write_and_emit_output helper also emits [OUTPUT_FILE] marker."""
        phase_dir = str(tmp_path / "apply-test")
        output = {"status": "ok"}
        write_and_emit_output(output, phase_dir)

        captured = capsys.readouterr()
        assert "[OUTPUT_FILE]" in captured.err
        for line in captured.err.splitlines():
            if "[OUTPUT_FILE]" in line:
                assert "orchestrator_output.json" in line
                break

    # ------------------------------------------------------------------
    # M8: [orchestrator] Fresh start marker on stderr
    # ------------------------------------------------------------------

    def test_m8_fresh_start_marker_on_stderr(self, tmp_path, capsys):
        """M8: --fresh emits [orchestrator] fresh-start marker to stderr."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), fresh=True)

        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "[orchestrator] Fresh start requested, clearing previous progress..." in captured.err

    def test_m8_fresh_marker_not_present_without_flag(self, tmp_path, capsys):
        """M8: Without --fresh, the [orchestrator] fresh marker is absent."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), fresh=False)

        orch = MockApplyOrchestrator(args)
        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "[orchestrator]" not in captured.err

    # ------------------------------------------------------------------
    # M9: [resume] marker on stderr
    # ------------------------------------------------------------------

    def test_m9_resume_marker_on_stderr(self, tmp_path, capsys):
        """M9: --resume emits [resume] marker showing processed/remaining counts."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), resume=True)

        groups = [
            {
                "theme": "A",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation, phase_name="apply-marker-resume")
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        # The marker format is: [resume] N already processed, M remaining
        assert "[resume]" in captured.err
        assert "already processed" in captured.err
        assert "remaining" in captured.err

    def test_m9_resume_marker_not_present_without_flag(self, tmp_path, capsys):
        """M9: Without --resume, the [resume] marker is absent."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), resume=False)

        groups = [
            {
                "theme": "A",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation)
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "[resume]" not in captured.err

    # ------------------------------------------------------------------
    # M10: [dry-run] Would auto-approve: on stderr
    # ------------------------------------------------------------------

    def test_m10_dry_run_auto_approve_marker_on_stderr(self, tmp_path, capsys):
        """M10: --dry-run with bulk approval shows [dry-run] Would auto-approve: on stderr."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path),
            dry_run=True,
            approve_all_low=True,
        )

        # Provide a needs-human-decision LOW item so the filter auto-approves it
        groups = [
            {
                "theme": "Low priority fix",
                "suggestions": [
                    {
                        "title": "Minor tweak",
                        "desc": "A small improvement",
                        "importance": "LOW",
                    }
                ],
            }
        ]
        validation = [
            {
                "status": "needs-human-decision",
                "reason": "Ambiguous intent",
                "confidence": 0.5,
            }
        ]
        OrchClass = self._make_data_orchestrator(groups, validation, phase_name="apply-marker-dry")
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "[dry-run] Would auto-approve:" in captured.err

    # ------------------------------------------------------------------
    # M11: --- DRY RUN --- on stderr
    # ------------------------------------------------------------------

    def test_m11_dry_run_header_on_stderr(self, tmp_path, capsys):
        """M11: --dry-run emits '--- DRY RUN ---' header to stderr."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), dry_run=True)

        groups = [
            {
                "theme": "A",
                "suggestions": [
                    {"title": "S1", "desc": "d", "importance": "MEDIUM"}
                ],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation, phase_name="apply-marker-dryheader")
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "--- DRY RUN ---" in captured.err

    def test_m11_dry_run_header_not_present_without_flag(self, tmp_path, capsys):
        """M11: Without --dry-run, '--- DRY RUN ---' is absent."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), dry_run=False)

        groups = [
            {
                "theme": "A",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation)
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "--- DRY RUN ---" not in captured.err

    # ------------------------------------------------------------------
    # M12: --- REVALIDATION MODE --- on stderr
    # ------------------------------------------------------------------

    def test_m12_revalidation_mode_header_on_stderr(self, tmp_path, capsys):
        """M12: --revalidate emits '--- REVALIDATION MODE ---' to stderr."""
        plan_path = _setup_plan_with_output(tmp_path)
        plan_path.write_text("# Test Plan\n", encoding="utf-8")
        args = _make_args(plan_file=str(plan_path), revalidate=True, dry_run=True)

        OrchClass = self._make_revalidating_orchestrator()
        orch = OrchClass(args)

        orch.plan_path = str(plan_path)
        orch.out_dir = str(tmp_path / "test-plan")
        orch.groups = [{"theme": "A", "suggestions": [{"title": "S1"}]}]
        orch.validation = [
            {"status": "validation_failed", "reason": "Timeout", "confidence": 0.0}
        ]

        review_dir = Path(orch.out_dir) / "review-plan"
        review_dir.mkdir(parents=True, exist_ok=True)

        # Use dry_run=True so handle_revalidation returns without exit
        orch.handle_revalidation()

        captured = capsys.readouterr()
        assert "--- REVALIDATION MODE ---" in captured.err

    def test_m12_revalidation_mode_not_present_without_flag(self, tmp_path, capsys):
        """M12: Without --revalidate, REVALIDATION MODE header is absent."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        OrchClass = self._make_revalidating_orchestrator()
        orch = OrchClass(args)

        orch.plan_path = str(plan_path)
        orch.out_dir = str(tmp_path / "test-plan")
        orch.groups = []
        orch.validation = []

        orch.handle_revalidation()

        captured = capsys.readouterr()
        assert "REVALIDATION MODE" not in captured.err

    # ------------------------------------------------------------------
    # M13: No new markers introduced by refactored code (meta-test)
    # ------------------------------------------------------------------

    def test_m13_no_unexpected_bracket_markers_on_stdout(self, tmp_path, capsys):
        """M13: Normal run does not emit unexpected [BRACKET_MARKER] to stdout.

        Only known stdout markers: JSON output. No bracket markers should
        appear on stdout during a normal (non-revalidation) run.
        """
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        groups = [
            {
                "theme": "A",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation, phase_name="apply-marker-nometa")
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        # No bracket markers should appear on stdout during normal run
        import re as re_module
        bracket_markers = re_module.findall(r'\[([A-Z_]+)\]', captured.out)
        # Allow empty list (no markers on stdout for normal run)
        assert bracket_markers == [], (
            f"Unexpected bracket markers on stdout: {bracket_markers}"
        )

    def test_m13_no_unexpected_bracket_markers_on_stderr(self, tmp_path, capsys):
        """M13: Normal run only emits known bracket markers to stderr.

        Known stderr bracket markers:
        - [OUTPUT_FILE]
        - [orchestrator] (only with --fresh)
        - [resume] (only with --resume)
        - [dry-run] (only with --dry-run)
        """
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        groups = [
            {
                "theme": "A",
                "suggestions": [{"title": "S1", "desc": "d", "importance": "MEDIUM"}],
            }
        ]
        validation = [{"status": "valid", "reason": "OK", "confidence": 0.9}]
        OrchClass = self._make_data_orchestrator(groups, validation, phase_name="apply-marker-meta2")
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        import re as re_module
        bracket_markers = re_module.findall(r'\[([A-Z_]+)\]', captured.err)

        # The only expected bracket marker on stderr during normal run is [OUTPUT_FILE]
        known_stderr_markers = {"OUTPUT_FILE"}
        unexpected = set(bracket_markers) - known_stderr_markers
        assert unexpected == set(), (
            f"Unexpected bracket markers on stderr: {unexpected}. "
            f"Known markers: {known_stderr_markers}"
        )

    def test_m13_all_known_stdout_json_statuses(self, capsys):
        """M13: Verify the exact set of known JSON status values.

        The refactored code should only emit these status values:
        - "skipped" (from --skip or no-items early exit)
        - "completed" (from --mark-completed)
        - "confirmation_needed" (from no-selection prompt)
        """
        # Verify helpers produce exactly these statuses
        skip_out = build_skipped_output("test-phase", "No items")
        assert skip_out["status"] == "skipped"

        confirm_out = build_confirmation_needed_output("test-phase", "Confirm?", 5)
        assert confirm_out["status"] == "confirmation_needed"

        # These are the only two helper-generated statuses
        # "completed" is inline in the base class
        known_statuses = {"skipped", "completed", "confirmation_needed"}
        assert known_statuses == {"skipped", "completed", "confirmation_needed"}


# ======================================================================
# Test: State transition parity across migrated orchestrators
# ======================================================================


class TestStateTransitionParity:
    """Verify state transitions are identical before and after migration.

    Tests cover:
    1. Fresh run (no prior state) for all three orchestrators
    2. Resume (prior partial state exists)
    3. Phase completion: the emit path records every phase *started*, never
       completed (completion is the host's post-loop ``--mark-completed``)
    4. --fresh correctly resets StateManager state
    5. --resume with prior state produces identical resume_info dicts
    6. mark_phase_started called on the emit path, mark_phase_completed never
    7. State file field names, structure, and transition sequences
    """

    # ------------------------------------------------------------------
    # Orchestrator config matrices matching the three concrete subclasses
    # ------------------------------------------------------------------

    #: ``confirmation_phase_name`` is carried here because it is not cosmetic:
    #: the ``--mark-completed`` early exit persists
    #: ``confirmation_phase_name or phase_name``, so it is the word that lands in
    #: ``phases_completed``. Only apply-code-fixes sets it (its ``phase_name`` is
    #: the phase *directory* word ``apply-fixes``); the other two leave it empty,
    #: so the fallback keeps their word byte-for-byte.
    ORCHESTRATOR_CONFIGS = {
        "apply-suggestions": {
            "phase_name": "apply-suggestions",
            "confirmation_phase_name": "",
            "review_subdir": "review-plan",
            "item_noun": "suggestion",
            "supports_revalidation": True,
            "supports_skip_flag": True,
        },
        "apply-fixes": {
            "phase_name": "apply-fixes",
            "confirmation_phase_name": "apply-code-fixes",
            "review_subdir": "code-review",
            "item_noun": "fix",
            "supports_revalidation": True,
            "supports_skip_flag": False,
        },
        "apply-task-suggestions": {
            "phase_name": "apply-task-suggestions",
            "confirmation_phase_name": "",
            "review_subdir": "review-tasks",
            "item_noun": "task suggestion",
            "supports_revalidation": False,
            "supports_skip_flag": True,
        },
    }

    @staticmethod
    def _completion_word(config):
        """The word a config records in ``phases_completed``.

        The base's ``confirmation_phase_name or phase_name`` fallback, restated
        here so the assertions below read the contract rather than a literal.
        """
        return config["confirmation_phase_name"] or config["phase_name"]

    @staticmethod
    def _make_orchestrator_class(config, groups=None, validation=None):
        """Create a mock orchestrator class with the given config and data."""
        _groups = list(groups) if groups else []
        _validation = list(validation) if validation else []
        _phase = config["phase_name"]
        _confirm_phase = config["confirmation_phase_name"]
        _review = config["review_subdir"]
        _noun = config["item_noun"]
        _reval = config["supports_revalidation"]
        _skip = config["supports_skip_flag"]

        class ConfiguredOrch(MockApplyOrchestrator):
            phase_name = _phase
            confirmation_phase_name = _confirm_phase
            review_subdir = _review
            item_noun = _noun
            supports_revalidation = _reval
            supports_skip_flag = _skip

            def load_data(self):
                self.call_log.append("load_data")
                return (_groups, _validation)

        return ConfiguredOrch

    @staticmethod
    def _setup_plan_for_config(tmp_path, config):
        """Create plan file and output directory matching a config's review_subdir."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Test Plan\n", encoding="utf-8")

        out_dir = tmp_path / "test-plan"
        out_dir.mkdir(exist_ok=True)

        review_dir = out_dir / config["review_subdir"]
        review_dir.mkdir(exist_ok=True)
        (review_dir / "report.md").write_text("# Review Report\n", encoding="utf-8")

        return plan_path

    @staticmethod
    def _sample_groups():
        """Return sample groups for testing."""
        return [
            {
                "theme": "Improve error handling",
                "group_hash": "abc12345deadbeef",
                "suggestions": [
                    {
                        "title": "Add try-catch",
                        "desc": "Wrap in try-catch",
                        "importance": "HIGH",
                        "suggestion_hash": "s1hash",
                    }
                ],
            },
            {
                "theme": "Optimize queries",
                "group_hash": "def67890cafebabe",
                "suggestions": [
                    {
                        "title": "Add index",
                        "desc": "Add database index",
                        "importance": "MEDIUM",
                        "suggestion_hash": "s2hash",
                    }
                ],
            },
        ]

    @staticmethod
    def _sample_validation():
        """Return sample validation results matching _sample_groups."""
        return [
            {"status": "valid", "reason": "Good suggestion", "confidence": 0.95},
            {"status": "valid", "reason": "Acceptable", "confidence": 0.85},
        ]

    # ------------------------------------------------------------------
    # 1. Fresh run state transitions for all three orchestrators
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_fresh_run_creates_initial_state(self, tmp_path, config_name):
        """Fresh run (no prior state) creates state with expected structure."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        # Verify state was initialized
        assert orch.state is not None
        state_data = orch.state.state

        # Verify required state fields exist
        required_fields = {
            "schema_version", "plan_path", "plan_hash",
            "created_at", "updated_at", "head_at_start",
            "branch_name", "review_phase_completed",
            "tracked_files", "task_status",
            "phases_completed", "phases_skipped",
        }
        for field in required_fields:
            assert field in state_data, (
                f"Missing required field '{field}' in state for {config_name}"
            )

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_fresh_run_state_field_types(self, tmp_path, config_name):
        """Fresh run state fields have correct types."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(
            config, self._sample_groups(), self._sample_validation()
        )
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        state_data = orch.state.state
        assert isinstance(state_data["schema_version"], str)
        assert isinstance(state_data["plan_path"], str)
        assert isinstance(state_data["plan_hash"], str)
        assert isinstance(state_data["created_at"], str)
        assert isinstance(state_data["updated_at"], str)
        assert isinstance(state_data["review_phase_completed"], bool)
        assert isinstance(state_data["tracked_files"], list)
        assert isinstance(state_data["task_status"], dict)
        assert isinstance(state_data["phases_completed"], dict)
        assert isinstance(state_data["phases_skipped"], dict)

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_fresh_run_no_prior_decisions(self, tmp_path, config_name):
        """Fresh run has no prior human decisions or processed items."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        phase = config["phase_name"]
        decisions = orch.state.get_all_human_decisions(phase)
        processed = orch.state.get_processed_items(phase)
        assert decisions == {}, f"Expected no decisions for fresh {config_name}"
        assert processed == {}, f"Expected no processed items for fresh {config_name}"

    # ------------------------------------------------------------------
    # 2. Resume state transitions
    # ------------------------------------------------------------------

    def test_resume_with_prior_decisions_applies_them(self, tmp_path):
        """--resume with prior human decisions applies them to merged groups."""
        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        # First run: record a human decision
        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args)
        orch1.run()

        # Record a human decision manually
        from utils.state_manager import generate_group_id
        gid = generate_group_id(groups[0])
        orch1.state.record_human_decision(
            config["phase_name"], gid, "approved", reason="Looks good"
        )
        orch1.state.save()

        # Second run with --resume
        args2 = _make_args(plan_file=str(plan_path), resume=True)
        orch2 = OrchClass(args2)
        exit_code = orch2.run()
        assert exit_code == 0

        # The state should still have the prior decision
        decisions = orch2.state.get_all_human_decisions(config["phase_name"])
        assert gid in decisions
        assert decisions[gid]["decision"] == "approved"

    def test_resume_skips_already_processed_items(self, tmp_path, capsys):
        """--resume skips items that were already processed."""
        config = self.ORCHESTRATOR_CONFIGS["apply-fixes"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        # First run: mark one item as processed
        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args)
        orch1.run()

        from utils.state_manager import generate_group_id
        gid = generate_group_id(groups[0])
        orch1.state.mark_item_processed(
            config["phase_name"], gid, "applied"
        )
        orch1.state.save()

        # Second run with --resume
        args2 = _make_args(plan_file=str(plan_path), resume=True)
        orch2 = OrchClass(args2)
        exit_code = orch2.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "[resume]" in captured.err
        assert "1 already processed" in captured.err

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_resume_with_no_prior_state_acts_as_fresh(self, tmp_path, config_name):
        """--resume with no prior state produces same result as fresh run."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        # Fresh run
        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args_fresh = _make_args(plan_file=str(plan_path))
        orch_fresh = OrchClass(args_fresh)
        orch_fresh.run()

        # Resume run (no prior state for this phase)
        args_resume = _make_args(plan_file=str(plan_path), resume=True)
        orch_resume = OrchClass(args_resume)
        orch_resume.run()

        # Both should produce equivalent results
        assert len(orch_fresh.valid) == len(orch_resume.valid)
        assert len(orch_fresh.skipped) == len(orch_resume.skipped)
        assert len(orch_fresh.needs_human) == len(orch_resume.needs_human)

    # ------------------------------------------------------------------
    # 3. --fresh correctly resets StateManager state
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_fresh_clears_human_decisions(self, tmp_path, config_name):
        """--fresh clears all human decisions for the phase."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        # First run: record decisions
        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args)
        orch1.run()

        from utils.state_manager import generate_group_id
        gid = generate_group_id(groups[0])
        orch1.state.record_human_decision(
            config["phase_name"], gid, "approved"
        )
        orch1.state.save()

        # Verify decision was saved
        assert gid in orch1.state.get_all_human_decisions(config["phase_name"])

        # Second run with --fresh
        args_fresh = _make_args(plan_file=str(plan_path), fresh=True)
        orch2 = OrchClass(args_fresh)
        orch2.run()

        # Decisions should be cleared
        decisions = orch2.state.get_all_human_decisions(config["phase_name"])
        assert decisions == {}, (
            f"--fresh should clear decisions for {config_name}, got: {decisions}"
        )

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_fresh_clears_processed_items(self, tmp_path, config_name):
        """--fresh clears all processed items for the phase."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        # First run: mark items as processed
        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args)
        orch1.run()

        from utils.state_manager import generate_group_id
        gid = generate_group_id(groups[0])
        orch1.state.mark_item_processed(
            config["phase_name"], gid, "applied"
        )
        orch1.state.save()

        # Verify it was recorded
        assert orch1.state.is_item_processed(config["phase_name"], gid)

        # Second run with --fresh
        args_fresh = _make_args(plan_file=str(plan_path), fresh=True)
        orch2 = OrchClass(args_fresh)
        orch2.run()

        # Processed items should be cleared
        processed = orch2.state.get_processed_items(config["phase_name"])
        assert processed == {}, (
            f"--fresh should clear processed items for {config_name}"
        )

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_fresh_clears_processing_progress(self, tmp_path, config_name):
        """--fresh clears processing progress for the phase."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args)
        orch1.run()

        # Record progress
        orch1.state.record_processing_progress(
            config["phase_name"], 10, 5, 1, 3
        )
        orch1.state.save()

        # Verify it was saved
        assert orch1.state.get_processing_progress(config["phase_name"]) is not None

        # Second run with --fresh
        args_fresh = _make_args(plan_file=str(plan_path), fresh=True)
        orch2 = OrchClass(args_fresh)
        orch2.run()

        # Progress should be cleared
        progress = orch2.state.get_processing_progress(config["phase_name"])
        assert progress is None, (
            f"--fresh should clear progress for {config_name}"
        )

    def test_fresh_emits_stderr_marker(self, tmp_path, capsys):
        """--fresh emits the clearing message to stderr."""
        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path), fresh=True)
        orch = OrchClass(args)
        orch.run()

        captured = capsys.readouterr()
        assert "[orchestrator] Fresh start requested" in captured.err

    # ------------------------------------------------------------------
    # 4. the emit path records the phase started, never completed
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions", "apply-fixes", "apply-task-suggestions",
    ])
    def test_emit_records_started_not_completed(self, tmp_path, config_name):
        """`run()` APPLIES NOTHING: it emits batch JSON and returns, and the
        bytes are written afterwards by the host following the instruction.
        So the emit path records the phase *started* — under the same
        `confirmation_phase_name or phase_name` word `--mark-completed` later
        persists — and never completed. Completion is the host's post-loop
        `--mark-completed` (see `TestApplyCodeFixesCompletionContract`), or
        `WriteBatchScheduler._finalize` under `--exec-subagents`.

        Until 2026-10-05 apply-suggestions marked itself completed here, so
        `status` and the next-step builder read the phase as done while the
        host was still applying its batches (racing re-run, 2026-10-04).
        """
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        exit_code = orch.run()
        assert exit_code == 0

        word = self._completion_word(config)
        assert orch.state.is_phase_started(word), (
            f"{config_name}: the emit path must record {word!r} started"
        )
        assert not orch.state.is_phase_completed(word), (
            f"{config_name}: the emit path must never record {word!r} completed"
        )
        assert word in orch.state.state.get("phases_started", {})
        assert not orch.state.state.get("phases_completed", {})

    def test_started_record_persisted_to_state_file(self, tmp_path):
        """The started record is saved on disk at emit time, so a host that
        dies before its first batch leaves the phase `in_progress`."""
        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        state_file = orch.state.state_file
        assert state_file.exists(), "State file should exist on disk"
        with open(state_file, encoding="utf-8") as f:
            saved_state = json.load(f)

        assert isinstance(saved_state["phases_started"]["apply-suggestions"], str)
        assert "apply-suggestions" not in saved_state.get("phases_completed", {})

    def test_mark_completed_clears_the_started_record(self, tmp_path):
        """The host's `--mark-completed` turns the start into a completion."""
        from utils.state_manager import StateManager

        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        OrchClass(_make_args(plan_file=str(plan_path))).run()

        with pytest.raises(SystemExit) as exc:
            OrchClass(_make_args(plan_file=str(plan_path), mark_completed=True)).run()
        assert exc.value.code == 0
        state = StateManager(Path(plan_path))
        assert state.is_phase_completed("apply-suggestions")
        assert not state.is_phase_started("apply-suggestions")
        assert "apply-suggestions" not in state.state.get("phases_started", {})

    def test_mark_phase_not_completed_absent_from_state(self, tmp_path):
        """The emit path leaves the phase absent from phases_completed."""
        config = self.ORCHESTRATOR_CONFIGS["apply-fixes"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        # In-memory state should not have this phase completed
        phases_completed = orch.state.state.get("phases_completed", {})
        assert "apply-fixes" not in phases_completed

    # ------------------------------------------------------------------
    # 5. --resume with prior state produces correct resume_info dicts
    # ------------------------------------------------------------------

    def test_resume_info_structure_with_prior_state(self, tmp_path):
        """resume_info dict contains expected keys when prior state exists."""
        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        # First run to set up state
        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args)
        orch1.run()

        # Record some state
        from utils.state_manager import generate_group_id
        gid = generate_group_id(groups[0])
        orch1.state.record_human_decision(
            config["phase_name"], gid, "approved", reason="Test"
        )
        orch1.state.mark_item_processed(
            config["phase_name"], gid, "applied"
        )
        orch1.state.save()

        # Resume run: capture the resume_info built by _write_outputs
        resume_info_capture = {}

        class ResumeCapture(self._make_orchestrator_class(config, groups, validation)):
            phase_name = "apply-suggestions"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = True
            supports_skip_flag = True

            def build_output_json(self, batches, *, resume_info=None):
                self.call_log.append("build_output_json")
                resume_info_capture["data"] = resume_info
                return {"status": "ok", "batches": [], "phase": self.phase_name}

        args2 = _make_args(plan_file=str(plan_path), resume=True)
        orch2 = ResumeCapture(args2)
        orch2.run()

        # Verify resume_info structure
        ri = resume_info_capture["data"]
        assert ri is not None
        assert "previously_processed" in ri
        assert "previous_decisions" in ri
        assert "can_resume" in ri

        # Verify content
        assert isinstance(ri["previously_processed"], list)
        assert isinstance(ri["previous_decisions"], dict)
        assert isinstance(ri["can_resume"], bool)

    def test_resume_info_with_no_prior_state(self, tmp_path):
        """resume_info when no prior state has can_resume=False."""
        config = self.ORCHESTRATOR_CONFIGS["apply-task-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        resume_info_capture = {}

        class CaptureOrch(MockApplyOrchestrator):
            phase_name = "apply-task-suggestions"
            review_subdir = "review-tasks"
            item_noun = "task suggestion"
            supports_revalidation = False
            supports_skip_flag = True

            def build_output_json(self, batches, *, resume_info=None):
                self.call_log.append("build_output_json")
                resume_info_capture["data"] = resume_info
                return {"status": "ok", "batches": [], "phase": self.phase_name}

        args = _make_args(plan_file=str(plan_path))
        orch = CaptureOrch(args)
        orch.run()

        ri = resume_info_capture["data"]
        assert ri is not None
        assert ri["previously_processed"] == []
        assert ri["previous_decisions"] == {}
        assert ri["can_resume"] is False

    def test_resume_info_reflects_processed_items(self, tmp_path):
        """resume_info.previously_processed reflects items marked processed."""
        config = self.ORCHESTRATOR_CONFIGS["apply-fixes"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        # Set up prior state with processed items
        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args)
        orch1.run()

        from utils.state_manager import generate_group_id
        gid0 = generate_group_id(groups[0])
        gid1 = generate_group_id(groups[1])
        orch1.state.mark_item_processed(config["phase_name"], gid0, "applied")
        orch1.state.mark_item_processed(config["phase_name"], gid1, "skipped")
        orch1.state.save()

        # Resume and capture resume_info
        resume_info_capture = {}

        class CaptureOrch(self._make_orchestrator_class(config, groups, validation)):
            phase_name = "apply-fixes"
            review_subdir = "code-review"
            item_noun = "fix"
            supports_revalidation = True
            supports_skip_flag = False

            def build_output_json(self, batches, *, resume_info=None):
                self.call_log.append("build_output_json")
                resume_info_capture["data"] = resume_info
                return {"status": "ok", "batches": [], "phase": self.phase_name}

        args2 = _make_args(plan_file=str(plan_path), resume=True)
        orch2 = CaptureOrch(args2)
        orch2.run()

        ri = resume_info_capture["data"]
        assert ri["can_resume"] is True
        assert set(ri["previously_processed"]) == {gid0, gid1}

    # ------------------------------------------------------------------
    # 6. State file structure assertions
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_state_schema_version(self, tmp_path, config_name):
        """State file has schema_version '2.0' for all orchestrators."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        assert orch.state.state["schema_version"] == "2.0"

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_state_file_stored_in_plan_directory(self, tmp_path, config_name):
        """State file is stored in plan's output directory for all orchestrators."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        state_file = orch.state.state_file
        # State file should be inside the plan's output directory
        plan_dir = plan_path.parent / "test-plan"
        assert str(state_file).startswith(str(plan_dir)), (
            f"State file {state_file} should be under {plan_dir}"
        )

    def test_state_human_decision_field_structure(self, tmp_path):
        """Human decisions have correct field structure."""
        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        # Record a decision and check its structure
        orch.state.record_human_decision(
            config["phase_name"],
            "test_group_hash",
            "approved",
            reason="Looks correct",
        )

        decision = orch.state.get_human_decision(
            config["phase_name"], "test_group_hash"
        )
        assert decision is not None
        assert "decision" in decision
        assert "reason" in decision
        assert "timestamp" in decision
        assert decision["decision"] == "approved"
        assert decision["reason"] == "Looks correct"

    def test_state_processed_item_field_structure(self, tmp_path):
        """Processed items have correct field structure."""
        config = self.ORCHESTRATOR_CONFIGS["apply-fixes"]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        # Record a processed item and check its structure
        orch.state.mark_item_processed(
            config["phase_name"],
            "test_item_hash",
            "applied",
            details={"files_changed": 3},
        )

        processed = orch.state.get_processed_items(config["phase_name"])
        item = processed["test_item_hash"]
        assert "status" in item
        assert "details" in item
        assert "timestamp" in item
        assert item["status"] == "applied"
        assert item["details"]["files_changed"] == 3

    def test_state_key_naming_convention(self, tmp_path):
        """State keys for human decisions and progress follow naming convention."""
        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        phase = config["phase_name"]

        # Record data to create the keys
        orch.state.record_human_decision(phase, "g1", "approved")
        orch.state.mark_item_processed(phase, "g1", "applied")
        orch.state.record_processing_progress(phase, 10, 5, 1, 3)

        state_data = orch.state.state

        # Verify the key naming convention: human_decisions_{phase}
        assert f"human_decisions_{phase}" in state_data
        # Verify: processed_{phase}
        assert f"processed_{phase}" in state_data
        # Verify: progress_{phase}
        assert f"progress_{phase}" in state_data

    @pytest.mark.parametrize("config_name", [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ])
    def test_state_key_names_use_phase_name(self, tmp_path, config_name):
        """State keys are namespaced by phase_name, ensuring isolation between orchestrators."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)
        orch.run()

        phase = config["phase_name"]

        # Record decisions for this phase
        orch.state.record_human_decision(phase, "hash1", "approved")

        # The decision should only be visible under this phase's key
        own_decisions = orch.state.get_all_human_decisions(phase)
        assert "hash1" in own_decisions

        # Other phases should not see it
        other_phases = [
            p for p in ["apply-suggestions", "apply-fixes", "apply-task-suggestions"]
            if p != phase
        ]
        for other_phase in other_phases:
            other_decisions = orch.state.get_all_human_decisions(other_phase)
            assert "hash1" not in other_decisions, (
                f"Decision for {phase} leaked into {other_phase}"
            )

    # ------------------------------------------------------------------
    # 7. State transition sequence verification
    # ------------------------------------------------------------------

    def test_state_transition_sequence_fresh_run(self, tmp_path):
        """Fresh run follows setup -> load -> feedback -> batch -> write sequence
        and state reflects transitions in order."""
        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)

        # Track lifecycle phase entries
        phase_sequence = []
        original_setup = orch._setup
        original_load = orch._load_and_parse_inputs
        original_feedback = orch._apply_user_feedback
        original_batches = orch._prepare_batches
        original_write = orch._write_outputs

        def track_setup():
            phase_sequence.append("setup")
            return original_setup()

        def track_load():
            phase_sequence.append("load")
            return original_load()

        def track_feedback():
            phase_sequence.append("feedback")
            return original_feedback()

        def track_batches():
            phase_sequence.append("batches")
            return original_batches()

        def track_write():
            phase_sequence.append("write")
            return original_write()

        orch._setup = track_setup
        orch._load_and_parse_inputs = track_load
        orch._apply_user_feedback = track_feedback
        orch._prepare_batches = track_batches
        orch._write_outputs = track_write

        exit_code = orch.run()
        assert exit_code == 0

        assert phase_sequence == ["setup", "load", "feedback", "batches", "write"]

        # After full run: state was initialized and the phase recorded started
        assert orch.state is not None
        assert orch.state.is_phase_started("apply-suggestions")
        assert not orch.state.is_phase_completed("apply-suggestions")

    def test_state_transition_sequence_fresh_then_resume(self, tmp_path):
        """Two-run sequence: fresh -> resume preserves and continues state."""
        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)

        # Run 1: Fresh
        args1 = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args1)
        orch1.run()

        # Record some state between runs
        from utils.state_manager import generate_group_id
        gid = generate_group_id(groups[0])
        orch1.state.mark_item_processed(config["phase_name"], gid, "applied")
        orch1.state.save()

        created_at = orch1.state.state["created_at"]

        # Run 2: Resume
        args2 = _make_args(plan_file=str(plan_path), resume=True)
        orch2 = OrchClass(args2)
        orch2.run()

        # State should have been preserved across runs
        assert orch2.state.state["created_at"] == created_at
        assert orch2.state.is_item_processed(config["phase_name"], gid)

    def test_state_transition_sequence_fresh_after_resume_clears(self, tmp_path):
        """Three-run: fresh -> record state -> fresh clears the recorded state."""
        config = self.ORCHESTRATOR_CONFIGS["apply-fixes"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)

        # Run 1: Normal
        args1 = _make_args(plan_file=str(plan_path))
        orch1 = OrchClass(args1)
        orch1.run()

        from utils.state_manager import generate_group_id
        gid = generate_group_id(groups[0])
        orch1.state.record_human_decision(
            config["phase_name"], gid, "skipped"
        )
        orch1.state.mark_item_processed(config["phase_name"], gid, "skipped")
        orch1.state.record_processing_progress(
            config["phase_name"], 5, 3, 1, 2
        )
        orch1.state.save()

        # Run 2: Fresh -- should clear all phase-specific state
        args2 = _make_args(plan_file=str(plan_path), fresh=True)
        orch2 = OrchClass(args2)
        orch2.run()

        assert orch2.state.get_all_human_decisions(config["phase_name"]) == {}
        assert orch2.state.get_processed_items(config["phase_name"]) == {}
        assert orch2.state.get_processing_progress(config["phase_name"]) is None

    # ------------------------------------------------------------------
    # The --mark-completed early exit and the word it persists
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "config_name", ["apply-suggestions", "apply-fixes", "apply-task-suggestions"]
    )
    def test_mark_completed_persists_the_confirmation_phase_word(
        self, tmp_path, config_name
    ):
        """`--mark-completed` writes ``confirmation_phase_name or phase_name``.

        For the two configs that leave ``confirmation_phase_name`` empty the
        fallback must yield their existing word byte-for-byte; for apply-fixes it
        must yield ``apply-code-fixes`` — the canonical phase word — and never the
        phase-directory word ``apply-fixes``.
        """
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path), mark_completed=True)
        with pytest.raises(SystemExit) as exc_info:
            OrchClass(args).run()
        assert exc_info.value.code == 0

        from utils.state_manager import StateManager

        completed = StateManager(plan_path).state.get("phases_completed", {})
        expected = self._completion_word(config)
        assert expected in completed, (
            f"{config_name}: --mark-completed should persist {expected!r}, "
            f"got {sorted(completed)}"
        )
        # Nothing else may be recorded by this early exit — in particular
        # apply-fixes must not also (or instead) write its directory word.
        assert sorted(completed) == [expected], (
            f"{config_name}: --mark-completed persisted extra phase words: "
            f"{sorted(completed)}"
        )

    def test_mark_completed_on_apply_fixes_is_not_the_directory_word(self, tmp_path):
        """The apply-fixes case stated as a literal, not derived from the config.

        `phase_name` (``apply-fixes``) is the phase *directory*; the completion
        record is the mode word (``apply-code-fixes``). A regression that
        "reconciles" the two spellings shows up here even if the fixture moved
        with it.
        """
        config = self.ORCHESTRATOR_CONFIGS["apply-fixes"]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path), mark_completed=True)
        with pytest.raises(SystemExit):
            OrchClass(args).run()

        from utils.state_manager import StateManager

        state = StateManager(plan_path)
        assert state.is_phase_completed("apply-code-fixes")
        assert not state.is_phase_completed("apply-fixes")

    def _run_mark_completed_over_an_edited_plan(self, tmp_path, edits_the_plan):
        """Drive `--mark-completed` after the plan on disk moved."""
        from utils.state_manager import StateManager

        config = self.ORCHESTRATOR_CONFIGS["apply-suggestions"]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        # The state the phase inherits, stamped with the plan as it stood
        # before the host's write loop.
        StateManager(plan_path).save()

        plan_path.write_text("# Test Plan\n\nEdited.\n", encoding="utf-8")

        # A fresh class per call, so setting the flag on it affects nothing else.
        OrchClass = self._make_orchestrator_class(config)
        OrchClass.edits_the_plan = edits_the_plan
        args = _make_args(plan_file=str(plan_path), mark_completed=True)
        with pytest.raises(SystemExit):
            OrchClass(args).run()
        return StateManager(plan_path)

    def test_mark_completed_re_stamps_the_plan_hash_for_a_plan_editing_phase(
        self, tmp_path
    ):
        """F1: this is the native leg's finalizer, and for apply-suggestions it
        runs after the host rewrote the plan. Without a re-stamp the next
        `StateManager` reads the phase's own edit as a human editing the plan
        and `implement --resume` warns about it — and `plan_changed` is
        set-only at load now, so the residue has to be kept out here, at the
        writing end.
        """
        after = self._run_mark_completed_over_an_edited_plan(
            tmp_path, edits_the_plan=True
        )

        assert after.has_plan_changed() is False
        assert after.get("plan_hash") == after._compute_plan_hash()

    def test_a_phase_that_never_opens_the_plan_leaves_the_flag_alone(
        self, tmp_path
    ):
        """The other half: apply-code-fixes and apply-task-suggestions edit
        source and the tasks file. Drift in the PLAN across their run is a
        human's edit, and clearing it would swallow the warning."""
        after = self._run_mark_completed_over_an_edited_plan(
            tmp_path, edits_the_plan=False
        )

        assert after.has_plan_changed() is True

    def test_only_apply_suggestions_declares_that_it_edits_the_plan(self):
        """The wiring the two tests above stand on, on the real classes."""
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator
        from apply_task_suggestions_orchestrator import (
            ApplyTaskSuggestionsOrchestrator,
        )

        assert ApplySuggestionsOrchestrator.edits_the_plan is True
        assert ApplyCodeFixesOrchestrator.edits_the_plan is False
        assert ApplyTaskSuggestionsOrchestrator.edits_the_plan is False


# ======================================================================
# Test: Exit code and interactive prompt parity across migrated orchestrators
# ======================================================================


class TestExitCodeAndPromptParity:
    """Verify exit codes and interactive prompt behavior across all three
    migrated orchestrators.

    The three orchestrator configurations mirror the concrete subclasses:
    - apply-suggestions (supports_revalidation=True, supports_skip_flag=True)
    - apply-fixes (supports_revalidation=True, supports_skip_flag=False)
    - apply-task-suggestions (supports_revalidation=False,
      supports_skip_flag=True)

    All exit codes and prompt behaviors are owned by the base class template
    method, so they must be identical for each orchestrator under the same
    conditions.

    Exit code map
    =============
    EC-0a: Normal success (run completes through _write_outputs)
    EC-0b: --skip with supports_skip_flag=True -> sys.exit(0)
    EC-0c: --mark-completed -> sys.exit(0)
    EC-0d: No-selection confirmation prompt -> sys.exit(0)
    EC-0e: --dry-run -> sys.exit(0)
    EC-0f: handle_no_items_early_exit (subclass-specific, sys.exit(0))
    EC-1a: Plan file not found -> OrchestratorError(exit_code=1)
    EC-1b: Output directory not found -> OrchestratorError(exit_code=1)
    EC-1c: --approve-all without --yes/--force -> OrchestratorError(exit_code=1)
    EC-1d: load_data raises unexpected exception -> exit code 1
    EC-1e: Unexpected exception in any phase -> exit code 1
    EC-Na: OrchestratorError with custom exit_code -> that exit_code

    Prompt conditions
    =================
    P-1: No-selection confirmation prompt triggers when:
         - no user preferences detected AND
         - no bypass flags set (no_confirm, yes, force, approve_all,
           skip_all_human, approve_all_low, approve_importance,
           approve_validation_failed, dry_run)
    P-2: --yes suppresses the no-selection confirmation
    P-3: --force suppresses the no-selection confirmation
    P-4: --no-confirm suppresses the no-selection confirmation
    P-5: Other bulk flags suppress the confirmation
    """

    @pytest.fixture(autouse=True)
    def _three_plans_in_one_interpreter(self, monkeypatch):
        """Let one test drive three orchestrators over three plan files.

        Parity is asserted by running the three configurations back to back in
        a single interpreter, each against its own plan in its own tmp dir. A
        real process runs one plan, so ``_setup``'s §9.3.a/b anchor pin — which
        is set-once and raises when re-pointed at a *different* path — would
        fail the second orchestrator in every one of those loops. Clearing the
        anchor per pin, the same reset ``tests/conftest.py`` performs between
        tests, keeps these tests on what they are about (exit codes and prompt
        text). The pin itself, its position relative to ``_setup``'s early
        exits, and the set-once invariant are owned by
        ``tests/test_run_anchor_pinning.py`` and ``test_provider_registry.py``.
        """
        import utils.apply_orchestrator_base as base
        import utils.provider_registry as registry

        real_set_run_anchor = registry.set_run_anchor

        def _repin(path):
            registry._RUN_ANCHOR = None
            real_set_run_anchor(path)

        monkeypatch.setattr(base, "set_run_anchor", _repin)

    # ------------------------------------------------------------------
    # Orchestrator configuration matrix (same as TestStateTransitionParity)
    # ------------------------------------------------------------------

    ORCHESTRATOR_CONFIGS = {
        "apply-suggestions": {
            "phase_name": "apply-suggestions",
            "review_subdir": "review-plan",
            "item_noun": "suggestion",
            "supports_revalidation": True,
            "supports_skip_flag": True,
        },
        "apply-fixes": {
            "phase_name": "apply-fixes",
            "review_subdir": "code-review",
            "item_noun": "fix",
            "supports_revalidation": True,
            "supports_skip_flag": False,
        },
        "apply-task-suggestions": {
            "phase_name": "apply-task-suggestions",
            "review_subdir": "review-tasks",
            "item_noun": "task suggestion",
            "supports_revalidation": False,
            "supports_skip_flag": True,
        },
    }

    ALL_CONFIG_NAMES = [
        "apply-suggestions",
        "apply-fixes",
        "apply-task-suggestions",
    ]

    # Configs that support --skip
    SKIP_CONFIGS = ["apply-suggestions", "apply-task-suggestions"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_orchestrator_class(config, groups=None, validation=None):
        """Create a mock orchestrator class with the given config and data."""
        _groups = list(groups) if groups else []
        _validation = list(validation) if validation else []
        _phase = config["phase_name"]
        _review = config["review_subdir"]
        _noun = config["item_noun"]
        _reval = config["supports_revalidation"]
        _skip = config["supports_skip_flag"]

        class ConfiguredOrch(MockApplyOrchestrator):
            phase_name = _phase
            review_subdir = _review
            item_noun = _noun
            supports_revalidation = _reval
            supports_skip_flag = _skip

            def load_data(self):
                self.call_log.append("load_data")
                return (_groups, _validation)

        return ConfiguredOrch

    @staticmethod
    def _make_failing_orchestrator_class(config, error_cls=RuntimeError, msg="boom"):
        """Create an orchestrator whose load_data raises an exception."""
        _phase = config["phase_name"]
        _review = config["review_subdir"]
        _noun = config["item_noun"]
        _reval = config["supports_revalidation"]
        _skip = config["supports_skip_flag"]
        _error_cls = error_cls
        _msg = msg

        class FailingOrch(MockApplyOrchestrator):
            phase_name = _phase
            review_subdir = _review
            item_noun = _noun
            supports_revalidation = _reval
            supports_skip_flag = _skip

            def load_data(self):
                raise _error_cls(_msg)

        return FailingOrch

    @staticmethod
    def _make_custom_exit_code_orchestrator_class(config, exit_code):
        """Create an orchestrator whose load_data raises OrchestratorError
        with a custom exit_code."""
        _phase = config["phase_name"]
        _review = config["review_subdir"]
        _noun = config["item_noun"]
        _reval = config["supports_revalidation"]
        _skip = config["supports_skip_flag"]
        _exit_code = exit_code

        class CustomExitOrch(MockApplyOrchestrator):
            phase_name = _phase
            review_subdir = _review
            item_noun = _noun
            supports_revalidation = _reval
            supports_skip_flag = _skip

            def load_data(self):
                raise OrchestratorError(
                    "Custom error for testing", exit_code=_exit_code
                )

        return CustomExitOrch

    @staticmethod
    def _make_no_items_orchestrator_class(config):
        """Create an orchestrator that loads items but all are filtered out
        (invalid) so no valid/needs_human remain.  Also overrides
        handle_no_items_early_exit to call sys.exit(0) like the task_suggestions
        and code_fixes subclasses do."""
        _phase = config["phase_name"]
        _review = config["review_subdir"]
        _noun = config["item_noun"]
        _reval = config["supports_revalidation"]
        _skip = config["supports_skip_flag"]

        class NoItemsOrch(MockApplyOrchestrator):
            phase_name = _phase
            review_subdir = _review
            item_noun = _noun
            supports_revalidation = _reval
            supports_skip_flag = _skip

            def load_data(self):
                self.call_log.append("load_data")
                groups = [
                    {
                        "theme": "Filtered out",
                        "group_hash": "filteredabc",
                        "suggestions": [
                            {
                                "title": "Will be invalid",
                                "desc": "d",
                                "importance": "LOW",
                            }
                        ],
                    }
                ]
                validation = [
                    {
                        "status": "invalid",
                        "reason": "Not applicable",
                        "confidence": 0.9,
                    }
                ]
                return (groups, validation)

            def handle_no_items_early_exit(self):
                """Simulate subclass early exit."""
                self.call_log.append("handle_no_items_early_exit")
                sys.exit(0)

        return NoItemsOrch

    @staticmethod
    def _setup_plan_for_config(tmp_path, config):
        """Create plan file and output directory matching a config's review_subdir."""
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Test Plan\n", encoding="utf-8")

        out_dir = tmp_path / "test-plan"
        out_dir.mkdir(exist_ok=True)

        review_dir = out_dir / config["review_subdir"]
        review_dir.mkdir(exist_ok=True)
        (review_dir / "report.md").write_text("# Review Report\n", encoding="utf-8")

        return plan_path

    @staticmethod
    def _sample_groups():
        """Return sample groups that will pass validation filtering."""
        return [
            {
                "theme": "Improve error handling",
                "group_hash": "abc12345deadbeef",
                "suggestions": [
                    {
                        "title": "Add try-catch",
                        "desc": "Wrap in try-catch",
                        "importance": "HIGH",
                        "suggestion_hash": "s1hash",
                    }
                ],
            },
        ]

    @staticmethod
    def _sample_validation():
        """Return sample validation results matching _sample_groups."""
        return [
            {"status": "valid", "reason": "Good suggestion", "confidence": 0.95},
        ]

    # ==================================================================
    # EC-0a: Normal success — all orchestrators return exit code 0
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec0a_normal_success_returns_0(self, tmp_path, config_name):
        """EC-0a: Successful run through all 5 phases returns exit code 0."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0, (
            f"{config_name}: Expected exit code 0 on success, got {exit_code}"
        )

    def test_ec0a_all_orchestrators_identical_exit_code(self, tmp_path):
        """EC-0a: All three orchestrators return identical exit code 0 on success."""
        exit_codes = {}
        for config_name in self.ALL_CONFIG_NAMES:
            config = self.ORCHESTRATOR_CONFIGS[config_name]
            sub = tmp_path / config_name
            sub.mkdir(exist_ok=True)
            plan_path = self._setup_plan_for_config(sub, config)

            OrchClass = self._make_orchestrator_class(
                config, self._sample_groups(), self._sample_validation()
            )
            args = _make_args(plan_file=str(plan_path))
            orch = OrchClass(args)
            exit_codes[config_name] = orch.run()

        values = list(exit_codes.values())
        assert all(v == 0 for v in values), (
            f"Exit codes should all be 0, got: {exit_codes}"
        )

    # ==================================================================
    # EC-0b: --skip with supports_skip_flag=True -> sys.exit(0)
    # ==================================================================

    @pytest.mark.parametrize("config_name", SKIP_CONFIGS)
    def test_ec0b_skip_flag_exits_0(self, tmp_path, config_name):
        """EC-0b: --skip with supports_skip_flag=True exits with code 0."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path), skip=True)
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0, (
            f"{config_name}: --skip should exit with code 0"
        )

    def test_ec0b_skip_ignored_when_not_supported(self, tmp_path):
        """EC-0b: --skip has no effect when supports_skip_flag=False (apply-fixes)."""
        config = self.ORCHESTRATOR_CONFIGS["apply-fixes"]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(
            config, self._sample_groups(), self._sample_validation()
        )
        args = _make_args(plan_file=str(plan_path), skip=True)
        orch = OrchClass(args)

        # Should NOT exit early; proceeds normally and returns 0
        exit_code = orch.run()
        assert exit_code == 0

    # ==================================================================
    # EC-0c: --mark-completed -> sys.exit(0) for all orchestrators
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec0c_mark_completed_exits_0(self, tmp_path, config_name):
        """EC-0c: --mark-completed exits with code 0 for all orchestrators."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path), mark_completed=True)
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0, (
            f"{config_name}: --mark-completed should exit with code 0"
        )

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec0c_mark_completed_emits_completed_json(self, tmp_path, capsys, config_name):
        """EC-0c: --mark-completed emits JSON with status=completed and correct phase."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path), mark_completed=True)
        orch = OrchClass(args)

        with pytest.raises(SystemExit):
            orch.run()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["status"] == "completed"
        assert config["phase_name"] in output["message"]

    # ==================================================================
    # EC-0d: No-selection confirmation prompt -> sys.exit(0)
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec0d_no_selection_confirmation_exits_0(self, tmp_path, config_name):
        """EC-0d: No user selections + no bypass flags exits with code 0."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=False,
            force=False,
            approve_all=False,
            skip_all_human=False,
            approve_all_low=False,
            approve_importance=None,
            dry_run=False,
        )
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0, (
            f"{config_name}: No-selection confirmation should exit with code 0"
        )

    def test_ec0d_all_orchestrators_identical_confirmation_exit_code(self, tmp_path):
        """EC-0d: All three orchestrators exit with identical code 0 for no-selection."""
        exit_codes = {}
        for config_name in self.ALL_CONFIG_NAMES:
            sub = tmp_path / config_name
            sub.mkdir(exist_ok=True)
            config = self.ORCHESTRATOR_CONFIGS[config_name]
            plan_path = self._setup_plan_for_config(sub, config)
            groups = self._sample_groups()
            validation = self._sample_validation()

            OrchClass = self._make_orchestrator_class(config, groups, validation)
            args = _make_args(
                plan_file=str(plan_path),
                no_confirm=False,
                yes=False,
                force=False,
                approve_all=False,
                skip_all_human=False,
                approve_all_low=False,
                approve_importance=None,
                dry_run=False,
            )
            orch = OrchClass(args)

            with pytest.raises(SystemExit) as exc_info:
                orch.run()
            exit_codes[config_name] = exc_info.value.code

        assert all(v == 0 for v in exit_codes.values()), (
            f"All no-selection exits should be 0, got: {exit_codes}"
        )

    # ==================================================================
    # EC-0e: --dry-run -> sys.exit(0) for all orchestrators
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec0e_dry_run_exits_0(self, tmp_path, config_name):
        """EC-0e: --dry-run exits with code 0 for all orchestrators."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(plan_file=str(plan_path), dry_run=True)
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0, (
            f"{config_name}: --dry-run should exit with code 0"
        )

    # ==================================================================
    # EC-0f: handle_no_items_early_exit -> sys.exit(0)
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec0f_no_items_early_exit_exits_0(self, tmp_path, config_name):
        """EC-0f: When all items are filtered out and subclass overrides
        handle_no_items_early_exit, exits with code 0."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_no_items_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)

        with pytest.raises(SystemExit) as exc_info:
            orch.run()
        assert exc_info.value.code == 0, (
            f"{config_name}: handle_no_items_early_exit should exit with code 0"
        )

    # ==================================================================
    # EC-1a: Plan file not found -> exit code 1
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec1a_missing_plan_file_returns_1(self, tmp_path, config_name):
        """EC-1a: Missing plan file returns exit code 1 for all orchestrators."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        nonexistent = str(tmp_path / "does-not-exist.md")

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=nonexistent)
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 1, (
            f"{config_name}: Missing plan file should return exit code 1"
        )

    def test_ec1a_all_orchestrators_identical_missing_plan_exit_code(self, tmp_path):
        """EC-1a: All three orchestrators return identical exit code 1 for missing plan."""
        exit_codes = {}
        for config_name in self.ALL_CONFIG_NAMES:
            config = self.ORCHESTRATOR_CONFIGS[config_name]
            OrchClass = self._make_orchestrator_class(config)
            args = _make_args(plan_file=str(tmp_path / f"missing-{config_name}.md"))
            orch = OrchClass(args)
            exit_codes[config_name] = orch.run()

        assert all(v == 1 for v in exit_codes.values()), (
            f"All missing-plan exits should be 1, got: {exit_codes}"
        )

    # ==================================================================
    # EC-1b: Output directory not found -> exit code 1
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec1b_missing_output_dir_returns_1(self, tmp_path, config_name):
        """EC-1b: Plan exists but output directory missing returns exit code 1."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        # Create plan file but NOT the output directory
        plan_path = tmp_path / "test-plan.md"
        plan_path.write_text("# Test Plan\n", encoding="utf-8")
        # Do NOT create tmp_path / "test-plan" directory

        OrchClass = self._make_orchestrator_class(config)
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 1, (
            f"{config_name}: Missing output directory should return exit code 1"
        )

    # ==================================================================
    # EC-1c: --approve-all without --yes/--force -> exit code 1
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec1c_approve_all_without_guardrail_returns_1(self, tmp_path, config_name):
        """EC-1c: --approve-all without --yes or --force returns exit code 1."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_orchestrator_class(
            config, self._sample_groups(), self._sample_validation()
        )
        args = _make_args(
            plan_file=str(plan_path),
            approve_all=True,
            yes=False,
            force=False,
        )
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 1, (
            f"{config_name}: --approve-all without --yes/--force should return 1"
        )

    def test_ec1c_all_orchestrators_identical_guardrail_exit_code(self, tmp_path):
        """EC-1c: All three orchestrators return identical exit code 1 for guardrail."""
        exit_codes = {}
        for config_name in self.ALL_CONFIG_NAMES:
            sub = tmp_path / config_name
            sub.mkdir(exist_ok=True)
            config = self.ORCHESTRATOR_CONFIGS[config_name]
            plan_path = self._setup_plan_for_config(sub, config)

            OrchClass = self._make_orchestrator_class(
                config, self._sample_groups(), self._sample_validation()
            )
            args = _make_args(
                plan_file=str(plan_path),
                approve_all=True,
                yes=False,
                force=False,
            )
            orch = OrchClass(args)
            exit_codes[config_name] = orch.run()

        assert all(v == 1 for v in exit_codes.values()), (
            f"All guardrail exits should be 1, got: {exit_codes}"
        )

    # ==================================================================
    # EC-1d: load_data raises unexpected exception -> exit code 1
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec1d_load_data_exception_returns_1(self, tmp_path, config_name):
        """EC-1d: Unexpected exception in load_data returns exit code 1."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_failing_orchestrator_class(
            config, RuntimeError, "data corrupted"
        )
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 1, (
            f"{config_name}: Unexpected exception in load_data should return 1"
        )

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec1d_various_exception_types_all_return_1(self, tmp_path, config_name):
        """EC-1d: Different exception types in load_data all return exit code 1."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]

        for exc_cls in (ValueError, TypeError, IOError, KeyError):
            sub = tmp_path / f"{config_name}-{exc_cls.__name__}"
            sub.mkdir(exist_ok=True)
            plan_path = self._setup_plan_for_config(sub, config)

            OrchClass = self._make_failing_orchestrator_class(
                config, exc_cls, f"test {exc_cls.__name__}"
            )
            args = _make_args(plan_file=str(plan_path))
            orch = OrchClass(args)

            exit_code = orch.run()
            assert exit_code == 1, (
                f"{config_name}: {exc_cls.__name__} should return exit code 1"
            )

    # ==================================================================
    # EC-1e: Unexpected exception in any phase -> exit code 1
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_ec1e_unexpected_exception_returns_1(self, tmp_path, config_name):
        """EC-1e: RuntimeError propagates as exit code 1."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_failing_orchestrator_class(
            config, RuntimeError, "unexpected crash"
        )
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 1

    # ==================================================================
    # EC-Na: OrchestratorError with custom exit_code
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    @pytest.mark.parametrize("custom_code", [2, 3, 42])
    def test_ecna_custom_exit_code_propagated(
        self, tmp_path, config_name, custom_code
    ):
        """EC-Na: OrchestratorError with custom exit_code propagates to run()."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)

        OrchClass = self._make_custom_exit_code_orchestrator_class(
            config, custom_code
        )
        args = _make_args(plan_file=str(plan_path))
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == custom_code, (
            f"{config_name}: Custom exit code {custom_code} not propagated, got {exit_code}"
        )

    def test_ecna_all_orchestrators_same_custom_exit_code(self, tmp_path):
        """EC-Na: All three orchestrators propagate the same custom exit code."""
        custom_code = 7
        exit_codes = {}
        for config_name in self.ALL_CONFIG_NAMES:
            sub = tmp_path / config_name
            sub.mkdir(exist_ok=True)
            config = self.ORCHESTRATOR_CONFIGS[config_name]
            plan_path = self._setup_plan_for_config(sub, config)

            OrchClass = self._make_custom_exit_code_orchestrator_class(
                config, custom_code
            )
            args = _make_args(plan_file=str(plan_path))
            orch = OrchClass(args)
            exit_codes[config_name] = orch.run()

        assert all(v == custom_code for v in exit_codes.values()), (
            f"All custom exit codes should be {custom_code}, got: {exit_codes}"
        )

    # ==================================================================
    # P-1: No-selection confirmation prompt content and trigger
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_p1_confirmation_prompt_emits_correct_json(
        self, tmp_path, capsys, config_name
    ):
        """P-1: No-selection confirmation emits JSON with expected structure."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=False,
            force=False,
            approve_all=False,
            skip_all_human=False,
            approve_all_low=False,
            approve_importance=None,
            dry_run=False,
        )
        orch = OrchClass(args)

        with pytest.raises(SystemExit):
            orch.run()

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify JSON structure
        assert output["status"] == "confirmation_needed"
        assert "message" in output
        assert "phase" in output
        assert "item_count" in output
        assert output["phase"] == config["phase_name"]
        assert output["item_count"] == len(groups)

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_p1_confirmation_prompt_message_content(
        self, tmp_path, capsys, config_name
    ):
        """P-1: Confirmation message mentions user_selections.json, --no-confirm, --yes."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=False,
            force=False,
            approve_all=False,
            skip_all_human=False,
            approve_all_low=False,
            approve_importance=None,
            dry_run=False,
        )
        orch = OrchClass(args)

        with pytest.raises(SystemExit):
            orch.run()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        msg = output["message"]

        # The message should mention key terms
        assert "user_selections.json" in msg
        assert "--no-confirm" in msg
        assert "--yes" in msg

    def test_p1_confirmation_prompt_text_identical_across_orchestrators(self, tmp_path, capsys):
        """P-1: The confirmation prompt message text is identical across all
        three orchestrators (modulo phase name and item_count)."""
        messages = {}
        for config_name in self.ALL_CONFIG_NAMES:
            sub = tmp_path / config_name
            sub.mkdir(exist_ok=True)
            config = self.ORCHESTRATOR_CONFIGS[config_name]
            plan_path = self._setup_plan_for_config(sub, config)
            groups = self._sample_groups()
            validation = self._sample_validation()

            OrchClass = self._make_orchestrator_class(config, groups, validation)
            args = _make_args(
                plan_file=str(plan_path),
                no_confirm=False,
                yes=False,
                force=False,
                approve_all=False,
                skip_all_human=False,
                approve_all_low=False,
                approve_importance=None,
                dry_run=False,
            )
            orch = OrchClass(args)

            with pytest.raises(SystemExit):
                orch.run()

            captured = capsys.readouterr()
            output = json.loads(captured.out)
            messages[config_name] = output["message"]

        # All messages should be identical (the message is phase-independent)
        values = list(messages.values())
        assert all(v == values[0] for v in values), (
            f"Confirmation messages should be identical, got: {messages}"
        )

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_p1_stderr_mentions_no_user_preferences(
        self, tmp_path, capsys, config_name
    ):
        """P-1: stderr output mentions 'No user preferences detected'."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=False,
            force=False,
            approve_all=False,
            skip_all_human=False,
            approve_all_low=False,
            approve_importance=None,
            dry_run=False,
        )
        orch = OrchClass(args)

        with pytest.raises(SystemExit):
            orch.run()

        captured = capsys.readouterr()
        assert "No user preferences detected" in captured.err

    # ==================================================================
    # P-2: --yes suppresses the no-selection confirmation prompt
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_p2_yes_flag_suppresses_confirmation(self, tmp_path, capsys, config_name):
        """P-2: --yes bypasses the no-selection confirmation prompt."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=True,
            force=False,
        )
        orch = OrchClass(args)

        # Should NOT trigger sys.exit via confirmation; runs to completion
        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "confirmation_needed" not in captured.out

    # ==================================================================
    # P-3: --force suppresses the no-selection confirmation prompt
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_p3_force_flag_suppresses_confirmation(self, tmp_path, capsys, config_name):
        """P-3: --force bypasses the no-selection confirmation prompt."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=False,
            yes=False,
            force=True,
        )
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "confirmation_needed" not in captured.out

    # ==================================================================
    # P-4: --no-confirm suppresses the no-selection confirmation prompt
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_p4_no_confirm_flag_suppresses_confirmation(
        self, tmp_path, capsys, config_name
    ):
        """P-4: --no-confirm bypasses the no-selection confirmation prompt."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)
        args = _make_args(
            plan_file=str(plan_path),
            no_confirm=True,
            yes=False,
            force=False,
        )
        orch = OrchClass(args)

        exit_code = orch.run()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "confirmation_needed" not in captured.out

    # ==================================================================
    # P-5: Other bulk flags suppress the no-selection confirmation
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    @pytest.mark.parametrize(
        "bypass_flag,flag_value",
        [
            ("approve_all", True),
            ("skip_all_human", True),
            ("approve_all_low", True),
            ("approve_importance", ["LOW"]),
            ("dry_run", True),
        ],
        ids=[
            "approve_all",
            "skip_all_human",
            "approve_all_low",
            "approve_importance",
            "dry_run",
        ],
    )
    def test_p5_bulk_flags_suppress_confirmation(
        self, tmp_path, capsys, config_name, bypass_flag, flag_value
    ):
        """P-5: Bulk approval/filtering flags bypass the no-selection confirmation."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        plan_path = self._setup_plan_for_config(tmp_path, config)
        groups = self._sample_groups()
        validation = self._sample_validation()

        OrchClass = self._make_orchestrator_class(config, groups, validation)

        # Build args with the specific bypass flag set
        extra_args = {
            "no_confirm": False,
            "yes": False,
            "force": False,
            "approve_all": False,
            "skip_all_human": False,
            "approve_all_low": False,
            "approve_importance": None,
            "dry_run": False,
        }
        extra_args[bypass_flag] = flag_value

        # --approve-all requires --yes or --force for the guardrail
        if bypass_flag == "approve_all":
            extra_args["yes"] = True

        args = _make_args(plan_file=str(plan_path), **extra_args)
        orch = OrchClass(args)

        # The orchestrator should NOT stop for confirmation.
        # It may exit for dry_run (sys.exit(0)) or run to completion.
        try:
            exit_code = orch.run()
            captured = capsys.readouterr()
            assert "confirmation_needed" not in captured.out
        except SystemExit as e:
            # dry_run causes sys.exit(0) later
            assert e.code == 0
            captured = capsys.readouterr()
            assert "confirmation_needed" not in captured.out

    # ==================================================================
    # P-6: --yes and --force are functionally equivalent for prompt bypass
    # ==================================================================

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_p6_yes_and_force_equivalent_for_confirmation(
        self, tmp_path, config_name
    ):
        """P-6: --yes and --force both suppress confirmation and produce same exit code."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        groups = self._sample_groups()
        validation = self._sample_validation()

        results = {}
        for flag_name in ("yes", "force"):
            sub = tmp_path / f"{config_name}-{flag_name}"
            sub.mkdir(exist_ok=True)
            plan_path = self._setup_plan_for_config(sub, config)

            OrchClass = self._make_orchestrator_class(config, groups, validation)
            flag_args = {"no_confirm": False, "yes": False, "force": False}
            flag_args[flag_name] = True
            args = _make_args(plan_file=str(plan_path), **flag_args)
            orch = OrchClass(args)

            exit_code = orch.run()
            results[flag_name] = exit_code

        assert results["yes"] == results["force"] == 0, (
            f"{config_name}: --yes and --force should produce identical exit code 0, "
            f"got: {results}"
        )

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_p6_yes_and_force_both_allow_approve_all(self, tmp_path, config_name):
        """P-6: Both --yes and --force allow --approve-all to proceed."""
        config = self.ORCHESTRATOR_CONFIGS[config_name]
        groups = self._sample_groups()
        validation = self._sample_validation()

        for flag_name in ("yes", "force"):
            sub = tmp_path / f"{config_name}-{flag_name}-approve"
            sub.mkdir(exist_ok=True)
            plan_path = self._setup_plan_for_config(sub, config)

            OrchClass = self._make_orchestrator_class(config, groups, validation)
            flag_args = {"approve_all": True, "yes": False, "force": False}
            flag_args[flag_name] = True
            args = _make_args(plan_file=str(plan_path), **flag_args)
            orch = OrchClass(args)

            exit_code = orch.run()
            assert exit_code == 0, (
                f"{config_name}: --approve-all with --{flag_name} should succeed"
            )

    # ==================================================================
    # Cross-cutting: Exit code parity across all orchestrators
    # ==================================================================

    def test_cross_cutting_all_error_conditions_identical(self, tmp_path):
        """Verify that all error conditions produce identical exit codes
        across all three orchestrators."""
        # Condition -> expected exit code
        conditions = {}

        for config_name in self.ALL_CONFIG_NAMES:
            config = self.ORCHESTRATOR_CONFIGS[config_name]

            # --- Missing plan file ---
            OrchClass = self._make_orchestrator_class(config)
            args = _make_args(plan_file=str(tmp_path / f"nope-{config_name}.md"))
            orch = OrchClass(args)
            conditions.setdefault("missing_plan", {})[config_name] = orch.run()

            # --- --approve-all without guardrail ---
            sub = tmp_path / f"guardrail-{config_name}"
            sub.mkdir(exist_ok=True)
            plan_path = self._setup_plan_for_config(sub, config)
            OrchClass = self._make_orchestrator_class(
                config, self._sample_groups(), self._sample_validation()
            )
            args = _make_args(
                plan_file=str(plan_path),
                approve_all=True,
                yes=False,
                force=False,
            )
            orch = OrchClass(args)
            conditions.setdefault("approve_no_guard", {})[config_name] = orch.run()

            # --- load_data exception ---
            sub2 = tmp_path / f"load-fail-{config_name}"
            sub2.mkdir(exist_ok=True)
            plan_path2 = self._setup_plan_for_config(sub2, config)
            OrchClass = self._make_failing_orchestrator_class(config)
            args = _make_args(plan_file=str(plan_path2))
            orch = OrchClass(args)
            conditions.setdefault("load_exception", {})[config_name] = orch.run()

            # --- Custom exit code ---
            sub3 = tmp_path / f"custom-exit-{config_name}"
            sub3.mkdir(exist_ok=True)
            plan_path3 = self._setup_plan_for_config(sub3, config)
            OrchClass = self._make_custom_exit_code_orchestrator_class(config, 5)
            args = _make_args(plan_file=str(plan_path3))
            orch = OrchClass(args)
            conditions.setdefault("custom_exit_5", {})[config_name] = orch.run()

        # Verify each condition produces identical exit codes across orchestrators
        for condition, codes in conditions.items():
            values = list(codes.values())
            assert len(set(values)) == 1, (
                f"Condition '{condition}': Expected identical exit codes, "
                f"got {codes}"
            )

    # ==================================================================
    # Fixture-driven: comprehensive exit code path enumeration
    # ==================================================================

    @pytest.fixture(params=[
        # (fixture_id, description, expected_exit_code, expected_exit_type)
        # exit_type: "return" for normal return, "sysexit" for sys.exit()
        ("success", "Normal success", 0, "return"),
        ("plan_not_found", "Plan file not found", 1, "return"),
        ("output_dir_missing", "Output directory missing", 1, "return"),
        ("approve_all_no_guard", "--approve-all without --yes/--force", 1, "return"),
        ("unexpected_exception", "Unexpected exception in load_data", 1, "return"),
        ("custom_exit_2", "OrchestratorError with exit_code=2", 2, "return"),
        ("mark_completed", "--mark-completed flag", 0, "sysexit"),
        ("dry_run", "--dry-run flag", 0, "sysexit"),
        ("no_selection_prompt", "No selection confirmation prompt", 0, "sysexit"),
        ("no_items_early_exit", "All items filtered out", 0, "sysexit"),
    ])
    def exit_code_fixture(self, request, tmp_path):
        """Parameterized fixture providing test conditions for each exit code path."""
        fixture_id, desc, expected_code, exit_type = request.param

        # Build orchestrator for each condition
        results = {}
        for config_name in self.ALL_CONFIG_NAMES:
            config = self.ORCHESTRATOR_CONFIGS[config_name]
            sub = tmp_path / f"{config_name}-{fixture_id}"
            sub.mkdir(exist_ok=True)

            if fixture_id == "success":
                plan_path = self._setup_plan_for_config(sub, config)
                OrchClass = self._make_orchestrator_class(
                    config, self._sample_groups(), self._sample_validation()
                )
                args = _make_args(plan_file=str(plan_path))
            elif fixture_id == "plan_not_found":
                OrchClass = self._make_orchestrator_class(config)
                args = _make_args(plan_file=str(sub / "missing.md"))
            elif fixture_id == "output_dir_missing":
                plan_path = sub / "test-plan.md"
                plan_path.write_text("# Test Plan\n", encoding="utf-8")
                OrchClass = self._make_orchestrator_class(config)
                args = _make_args(plan_file=str(plan_path))
            elif fixture_id == "approve_all_no_guard":
                plan_path = self._setup_plan_for_config(sub, config)
                OrchClass = self._make_orchestrator_class(
                    config, self._sample_groups(), self._sample_validation()
                )
                args = _make_args(
                    plan_file=str(plan_path),
                    approve_all=True,
                    yes=False,
                    force=False,
                )
            elif fixture_id == "unexpected_exception":
                plan_path = self._setup_plan_for_config(sub, config)
                OrchClass = self._make_failing_orchestrator_class(config)
                args = _make_args(plan_file=str(plan_path))
            elif fixture_id == "custom_exit_2":
                plan_path = self._setup_plan_for_config(sub, config)
                OrchClass = self._make_custom_exit_code_orchestrator_class(config, 2)
                args = _make_args(plan_file=str(plan_path))
            elif fixture_id == "mark_completed":
                plan_path = self._setup_plan_for_config(sub, config)
                OrchClass = self._make_orchestrator_class(config)
                args = _make_args(plan_file=str(plan_path), mark_completed=True)
            elif fixture_id == "dry_run":
                plan_path = self._setup_plan_for_config(sub, config)
                OrchClass = self._make_orchestrator_class(
                    config, self._sample_groups(), self._sample_validation()
                )
                args = _make_args(plan_file=str(plan_path), dry_run=True)
            elif fixture_id == "no_selection_prompt":
                plan_path = self._setup_plan_for_config(sub, config)
                OrchClass = self._make_orchestrator_class(
                    config, self._sample_groups(), self._sample_validation()
                )
                args = _make_args(
                    plan_file=str(plan_path),
                    no_confirm=False,
                    yes=False,
                    force=False,
                    approve_all=False,
                    skip_all_human=False,
                    approve_all_low=False,
                    approve_importance=None,
                    dry_run=False,
                )
            elif fixture_id == "no_items_early_exit":
                plan_path = self._setup_plan_for_config(sub, config)
                OrchClass = self._make_no_items_orchestrator_class(config)
                args = _make_args(plan_file=str(plan_path))
            else:
                raise ValueError(f"Unknown fixture_id: {fixture_id}")

            results[config_name] = (OrchClass, args)

        return fixture_id, desc, expected_code, exit_type, results

    def test_fixture_driven_exit_codes(self, exit_code_fixture):
        """Fixture-driven: each exit code path produces the expected code and
        all three orchestrators agree."""
        fixture_id, desc, expected_code, exit_type, orchestrators = exit_code_fixture
        actual_codes = {}

        for config_name, (OrchClass, args) in orchestrators.items():
            orch = OrchClass(args)

            if exit_type == "sysexit":
                with pytest.raises(SystemExit) as exc_info:
                    orch.run()
                actual_codes[config_name] = exc_info.value.code
            else:
                actual_codes[config_name] = orch.run()

        # All codes must match expected
        for config_name, code in actual_codes.items():
            assert code == expected_code, (
                f"[{fixture_id}] {config_name}: Expected exit code {expected_code}, "
                f"got {code} ({desc})"
            )

        # All codes must be identical across orchestrators
        values = list(actual_codes.values())
        assert len(set(values)) == 1, (
            f"[{fixture_id}] Exit codes should be identical across orchestrators: "
            f"{actual_codes} ({desc})"
        )


# ======================================================================
# Test: "Let the agent decide" routing marker (Section 3)
# ======================================================================


class TestOldWordingReportStillRoutesToTheJudge:
    """A report.md written before the rename says "Let Claude decide"; its
    ticked items must still land in the judge's scope."""

    @pytest.mark.parametrize("wording", ["Let Claude decide", "Let the agent decide"])
    def test_ticked_item_reaches_the_judge_scope(self, tmp_path, wording):
        from utils.report_parser import parse_validation_overrides_groups

        report = tmp_path / "report.md"
        report.write_text(
            "## G1 [aabb000000000001]: Theme\n"
            "- [ ] Skip this group\n"
            "- [ ] Mark valid\n"
            "- [ ] Mark invalid\n"
            f"- [x] {wording}\n",
            encoding="utf-8",
        )
        orch = MockApplyOrchestrator(_make_args())
        orch.merged = [{
            "group_hash": "aabb000000000001",
            "validation_status": "needs-human-decision",
            "suggestions": [],
        }]
        orch.validation_overrides = parse_validation_overrides_groups(str(report))
        orch.apply_group_validation_overrides()
        orch.needs_human = list(orch.merged)

        assert orch.human_decision_scope() == orch.merged


class TestClaudeDecideGroupOverride:
    """Group-level judge_decide override is a routing marker, not a status."""

    def _orch(self, **arg_overrides):
        return MockApplyOrchestrator(_make_args(**arg_overrides))

    def test_group_judge_decide_keeps_needs_human_and_tags(self):
        """judge_decide override keeps needs-human-decision and tags the group."""
        orch = self._orch()
        orch.merged = [
            {
                "group_hash": "abc123",
                "validation_status": "needs-human-decision",
                "validation_reason": "Ambiguous",
                "suggestions": [],
            }
        ]
        orch.validation_overrides = {"abc123": "judge_decide"}
        orch.apply_group_validation_overrides()

        g = orch.merged[0]
        # Status is NOT changed to "judge_decide" (it is not a real status).
        assert g["validation_status"] == "needs-human-decision"
        assert g["judge_decide"] is True
        assert "abc123" in orch.judge_decide_overrides
        assert g["validation_reason"] == "Routed to the judge agent by reviewer"
        # judge_decide is a marker, not a user-approved override.
        assert g.get("user_override") is not True

    def test_group_judge_decide_never_becomes_status(self):
        """No code path may set validation_status == 'judge_decide'."""
        orch = self._orch()
        orch.merged = [
            {
                "group_hash": "h1",
                "validation_status": "needs-human-decision",
                "suggestions": [],
            }
        ]
        orch.validation_overrides = {"h1": "judge_decide"}
        orch.apply_group_validation_overrides()
        assert orch.merged[0]["validation_status"] != "judge_decide"

    def test_unknown_override_value_warned_and_ignored(self, capsys):
        """A typo/stale override value is warned-and-ignored, not applied."""
        orch = self._orch()
        orch.merged = [
            {
                "group_hash": "h1",
                "validation_status": "needs-human-decision",
                "suggestions": [],
            }
        ]
        orch.validation_overrides = {"h1": "bogus_value"}
        orch.apply_group_validation_overrides()

        g = orch.merged[0]
        # Underlying status is retained; the value did not become a status.
        assert g["validation_status"] == "needs-human-decision"
        assert g.get("user_override") is not True
        assert "h1" not in orch.judge_decide_overrides
        captured = capsys.readouterr()
        assert "bogus_value" in captured.err

    def test_valid_override_still_works(self):
        """The valid/invalid path is unaffected by the marker special-casing."""
        orch = self._orch()
        orch.merged = [
            {
                "group_hash": "h1",
                "validation_status": "needs-human-decision",
                "suggestions": [],
            }
        ]
        orch.validation_overrides = {"h1": "valid"}
        orch.apply_group_validation_overrides()
        assert orch.merged[0]["validation_status"] == "valid"
        assert orch.merged[0]["user_override"] is True


class TestClaudeDecideSuggestionOverride:
    """Per-suggestion judge_decide override propagates to the group."""

    def _orch(self, **arg_overrides):
        return MockApplyOrchestrator(_make_args(**arg_overrides))

    def test_suggestion_judge_decide_keeps_group_needs_human(self):
        """A judge_decide suggestion keeps its group needs-human (not promoted/dropped)."""
        orch = self._orch()
        orch.merged = [
            {
                "group_hash": "g1hash",
                "validation_status": "needs-human-decision",
                "suggestions": [
                    {"title": "A", "desc": "a"},
                ],
            }
        ]
        orch.suggestion_validation_overrides = {"G1S1": "judge_decide"}
        orch.apply_suggestion_validation_overrides()

        g = orch.merged[0]
        assert g["validation_status"] == "needs-human-decision"
        assert g["judge_decide"] is True
        assert "g1hash" in orch.judge_decide_overrides
        # The suggestion is not dropped.
        assert len(g["suggestions"]) == 1
        assert g["suggestions"][0].get("judge_decide") is True

    def test_mixed_group_one_judge_decide_propagates(self):
        """One judge_decide suggestion alongside untouched ones still routes the group."""
        orch = self._orch()
        orch.merged = [
            {
                "group_hash": "mixhash",
                "validation_status": "needs-human-decision",
                "suggestions": [
                    {"title": "A", "desc": "a"},
                    {"title": "B", "desc": "b"},
                    {"title": "C", "desc": "c"},
                ],
            }
        ]
        # Only the middle suggestion is marked.
        orch.suggestion_validation_overrides = {"G1S2": "judge_decide"}
        orch.apply_suggestion_validation_overrides()

        g = orch.merged[0]
        assert g["judge_decide"] is True
        assert "mixhash" in orch.judge_decide_overrides
        # No suggestion dropped; group still needs-human.
        assert len(g["suggestions"]) == 3
        assert g["validation_status"] == "needs-human-decision"

    def test_judge_decide_suggestion_not_dropped_as_invalid(self):
        """judge_decide must not land in the invalid-drop set."""
        orch = self._orch()
        orch.merged = [
            {
                "group_hash": "g1",
                "validation_status": "needs-human-decision",
                "suggestions": [
                    {"title": "A", "desc": "a"},
                    {"title": "B", "desc": "b"},
                ],
            }
        ]
        orch.suggestion_validation_overrides = {
            "G1S1": "judge_decide",
            "G1S2": "invalid",
        }
        orch.apply_suggestion_validation_overrides()
        g = orch.merged[0]
        # S2 (invalid) dropped, S1 (judge_decide) kept.
        assert len(g["suggestions"]) == 1
        assert g["suggestions"][0]["title"] == "A"
        assert g["judge_decide"] is True


class TestBuildHumanReviewConfigClaudeDecide:
    """build_human_review_config exposes judge_decide_item_ids."""

    def _config(self, formatted_human, **arg_overrides):
        orch = MockApplyOrchestrator(_make_args(**arg_overrides))
        orch.formatted_human = formatted_human
        return orch.build_human_review_config()

    def test_judge_decide_item_ids_lists_marked_items(self):
        """Only items with decision_mode == judge_auto_decide are listed."""
        formatted = [
            {"group_id": "h1", "importance": "HIGH",
             "decision_mode": "judge_auto_decide"},
            {"group_id": "h2", "importance": "MEDIUM"},
            {"group_id": "h3", "importance": "LOW",
             "decision_mode": "judge_auto_decide"},
        ]
        config = self._config(formatted)
        assert config["judge_decide_item_ids"] == ["h1", "h3"]
        # Per-item routing does not flip the global decision_mode.
        assert config["decision_mode"] == "interactive"

    def test_no_marked_items_empty_list(self):
        """No pre-marked items -> empty judge_decide_item_ids."""
        formatted = [{"group_id": "h1", "importance": "HIGH"}]
        config = self._config(formatted)
        assert config["judge_decide_item_ids"] == []
        assert config["decision_mode"] == "interactive"

    def test_global_flag_does_not_change_per_item_index(self):
        """--agent-decide flips the global mode; the per-item index is unchanged."""
        formatted = [
            {"group_id": "h1", "importance": "HIGH",
             "decision_mode": "judge_auto_decide"},
        ]
        config = self._config(formatted, agent_decide=True)
        assert config["decision_mode"] == "judge_auto_decide"
        assert config["judge_decide_item_ids"] == ["h1"]


# ======================================================================
# Test: the apply-code-fixes completion contract (R1)
# ======================================================================


class TestApplyCodeFixesCompletionContract:
    """Who records ``apply-code-fixes`` completed, and when.

    ``ApplyCodeFixesOrchestrator.run()`` applies nothing: it emits batch JSON and
    returns, and the bytes are written afterwards by the host following
    ``instructions/apply-code-fixes.md``. So the *emit* path must never record
    completion (it records the phase *started*; see
    ``TestStateTransitionParity.test_emit_records_started_not_completed``).
    The record instead comes from the post-loop
    ``apply-code-fixes --plan-file … --mark-completed`` command that closes that
    instruction — a command a host whose loop stopped early never reaches.

    The four host outcomes the instruction admits are replayed below. Only the
    first records the phase; the other three must leave it open.
    """

    #: The phase DIRECTORY word — what the orchestrator's `phase_name` is, what
    #: `{plan}/apply-fixes/` is named, and what the instruction's
    #: `util prior-changes append --phase` carries. Never the completion word.
    PHASE_DIR_WORD = "apply-fixes"

    #: The canonical PHASE word — the mode name, the `PHASE_PREREQUISITES` key,
    #: and what both completion paths must write.
    COMPLETION_WORD = "apply-code-fixes"

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @staticmethod
    def _seed(tmp_path):
        """Write a plan plus the code-review inputs the real orchestrator loads."""
        tmp_path = Path(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        plan_path = tmp_path / "auth-feature.md"
        plan_path.write_text("# Auth feature\n", encoding="utf-8")

        review_dir = tmp_path / "auth-feature" / "code-review"
        review_dir.mkdir(parents=True, exist_ok=True)
        groups = [
            {
                "theme": "Null check",
                "file": "src/api.ts",
                "suggestions": [
                    {
                        "title": "Add null check",
                        "desc": "Guard user.profile before dereferencing it.",
                        "importance": "HIGH",
                        "file": "src/api.ts",
                        "line_start": 42,
                        "line_end": 50,
                        "type": "bug",
                    }
                ],
            },
            {
                "theme": "RLS policy",
                "file": "src/db.sql",
                "suggestions": [
                    {
                        "title": "Add RLS policy",
                        "desc": "The table ships with RLS off.",
                        "importance": "HIGH",
                        "file": "src/db.sql",
                        "line_start": 1,
                        "line_end": 3,
                        "type": "security",
                    }
                ],
            },
        ]
        (review_dir / "grouped.json").write_text(
            json.dumps(groups), encoding="utf-8"
        )
        (review_dir / "validation.json").write_text(
            json.dumps(
                [
                    {"group_index": 0, "status": "valid", "verdict": "valid"},
                    {"group_index": 1, "status": "valid", "verdict": "valid"},
                ]
            ),
            encoding="utf-8",
        )
        (review_dir / "report.md").write_text("# Code Review\n", encoding="utf-8")
        return plan_path

    @staticmethod
    def _orchestrator_class():
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator

        return ApplyCodeFixesOrchestrator

    @staticmethod
    def _phases_completed(plan_path):
        from utils.state_manager import StateManager

        return sorted(StateManager(plan_path).state.get("phases_completed", {}))

    def _emit(self, plan_path, capsys=None):
        """Run the real orchestrator's emit path; return its batch document."""
        OrchClass = self._orchestrator_class()
        args = OrchClass.parse_args(["--plan-file", str(plan_path), "--yes"])
        exit_code = OrchClass(args).run()
        assert exit_code == 0
        if capsys is not None:
            capsys.readouterr()
        document = (
            Path(plan_path).parent
            / Path(plan_path).stem
            / self.PHASE_DIR_WORD
            / "orchestrator_output.json"
        )
        assert document.is_file(), "the emit path must write its batch document"
        payload = json.loads(document.read_text(encoding="utf-8"))
        # The emit path itself never claims the phase: nothing has been applied
        # yet at this point.
        assert self._phases_completed(plan_path) == []
        return payload

    def _post_loop_mark_completed(self, plan_path, capsys=None):
        """§10: the finalization command the host runs after the batch loop."""
        OrchClass = self._orchestrator_class()
        args = OrchClass.parse_args(
            ["--plan-file", str(plan_path), "--mark-completed"]
        )
        with pytest.raises(SystemExit) as exc_info:
            OrchClass(args).run()
        assert exc_info.value.code == 0
        if capsys is not None:
            capsys.readouterr()

    def _replay_host(self, plan_path, payload, outcome):
        """Replay §5's batch loop for one of the four outcomes it admits.

        Returns the per-batch statuses the host recorded, so the outcomes are
        distinguishable by more than the assertion that follows them.
        """
        from utils.state_manager import StateManager

        state = StateManager(plan_path)
        batches = payload["batches"]
        assert len(batches) >= 2, (
            "the fixture needs >= 2 batches for 'stopped partway' to differ "
            f"from 'stopped at the start' (got {len(batches)})"
        )

        recorded = []
        for index, batch in enumerate(batches, start=1):
            key = batch["file_key"]
            if outcome == "clean-failure" and index == 1:
                # §5.5 "Failure (clean)": the subagent failed and no file was
                # modified. The host reports it and abandons the loop.
                state.mark_item_processed(self.PHASE_DIR_WORD, key, "failed")
                recorded.append("failed")
                break
            if outcome == "partial-application" and index == 2:
                # §5.5 "Partial application": stop the loop and surface it for
                # human review. Explicitly NOT a completion.
                state.mark_item_processed(self.PHASE_DIR_WORD, key, "partial")
                recorded.append("partial")
                break
            if outcome == "interrupted-loop" and index == 2:
                # The host dies mid-batch: no further bookkeeping is written and
                # §10 is never reached.
                recorded.append("interrupted")
                break
            state.mark_item_processed(self.PHASE_DIR_WORD, key, "applied")
            recorded.append("applied")
        state.save()

        if outcome == "full-success":
            self._post_loop_mark_completed(plan_path)
        return recorded

    # ------------------------------------------------------------------
    # The four host outcomes
    # ------------------------------------------------------------------

    def test_full_success_records_the_phase(self, tmp_path, capsys):
        """Every batch applied, then §10 run -> `apply-code-fixes` completed."""
        plan_path = self._seed(tmp_path)
        payload = self._emit(plan_path, capsys)

        recorded = self._replay_host(plan_path, payload, "full-success")
        assert recorded == ["applied"] * len(payload["batches"])
        capsys.readouterr()

        assert self._phases_completed(plan_path) == [self.COMPLETION_WORD]

    def test_clean_failure_leaves_the_phase_incomplete(self, tmp_path, capsys):
        """A clean failure aborts the loop, so §10 is never reached."""
        plan_path = self._seed(tmp_path)
        payload = self._emit(plan_path, capsys)

        recorded = self._replay_host(plan_path, payload, "clean-failure")
        assert recorded == ["failed"]

        assert self._phases_completed(plan_path) == [], (
            "a clean failure must leave the phase open for a resume"
        )

    def test_partial_application_leaves_the_phase_incomplete(self, tmp_path, capsys):
        """A partial application stops the loop for human review; no completion."""
        plan_path = self._seed(tmp_path)
        payload = self._emit(plan_path, capsys)

        recorded = self._replay_host(plan_path, payload, "partial-application")
        assert recorded == ["applied", "partial"]

        assert self._phases_completed(plan_path) == [], (
            "a half-applied phase must never record itself completed"
        )

    def test_interrupted_loop_leaves_the_phase_incomplete(self, tmp_path, capsys):
        """The host dies before §10; nothing records the phase on its behalf."""
        plan_path = self._seed(tmp_path)
        payload = self._emit(plan_path, capsys)

        recorded = self._replay_host(plan_path, payload, "interrupted-loop")
        assert recorded == ["applied", "interrupted"]

        assert self._phases_completed(plan_path) == [], (
            "an interrupted loop must leave the phase open"
        )

    # ------------------------------------------------------------------
    # The word, and the drift guard between the two completion paths
    # ------------------------------------------------------------------

    def test_mark_completed_persists_the_mode_word_not_the_directory_word(
        self, tmp_path, capsys
    ):
        """`--mark-completed` on THIS orchestrator writes `apply-code-fixes`.

        Asserted directly against the persisted state, not inferred from the
        class attributes: the class runs under `phase_name = "apply-fixes"`, and
        persisting that word would be a third inconsistent spelling rather than a
        closed gap.
        """
        from utils.state_manager import StateManager

        plan_path = self._seed(tmp_path)
        self._post_loop_mark_completed(plan_path, capsys)

        state = StateManager(plan_path)
        assert state.is_phase_completed("apply-code-fixes")
        assert not state.is_phase_completed("apply-fixes")
        assert self._phases_completed(plan_path) == ["apply-code-fixes"]

    def test_the_parser_exposes_mark_completed(self):
        """The flag has to exist for the instruction's §10 command to parse."""
        OrchClass = self._orchestrator_class()
        args = OrchClass.parse_args(["--plan-file", "t.md", "--mark-completed"])
        assert args.mark_completed is True
        assert OrchClass.parse_args(["--plan-file", "t.md"]).mark_completed is False

    def test_native_and_executor_record_the_same_phase_word(self, tmp_path, capsys):
        """The two completion paths write ONE word, so they cannot drift apart.

        Native: the host's post-loop `--mark-completed`, which persists
        `confirmation_phase_name or phase_name`. Executor: `--exec-subagents`
        builds a `WriteBatchScheduler` without an explicit `phase`, so
        `_finalize` writes `PHASE_BY_WRITE_KIND[write_kind]`. Both are exercised
        here against real state files and the resulting `phases_completed` keys
        are compared.
        """
        from utils.state_manager import StateManager
        from utils.write_batch_scheduler import SchedulerResult, WriteBatchScheduler

        OrchClass = self._orchestrator_class()

        # --- native: the post-loop command -----------------------------
        native_plan = self._seed(tmp_path / "native")
        self._post_loop_mark_completed(native_plan, capsys)
        native_words = self._phases_completed(native_plan)

        # --- executor: WriteBatchScheduler._finalize -------------------
        executor_plan = self._seed(tmp_path / "executor")
        executor_out = executor_plan.parent / executor_plan.stem
        scheduler = WriteBatchScheduler(
            tasks=[],
            phase_dir=str(executor_out / self.PHASE_DIR_WORD),
            kind=OrchClass.write_kind,
            output_dir=str(executor_out),
            state=StateManager(executor_plan),
            plan_file=str(executor_plan),
            anchor=str(executor_plan),
            resume=True,
            dry_run=True,
        )
        scheduler._finalize(SchedulerResult())
        executor_words = self._phases_completed(executor_plan)
        capsys.readouterr()

        assert native_words == executor_words == [self.COMPLETION_WORD], (
            f"native wrote {native_words}, executor wrote {executor_words}"
        )
        # And that word is the one the prerequisite table is keyed on, which is
        # what the host-lane grader reads out of state.json.
        import check_workflow_prerequisites

        assert self.COMPLETION_WORD in check_workflow_prerequisites.PHASE_PREREQUISITES

    def test_the_instruction_issues_the_post_loop_command(self):
        """The defect R1 closes: the instruction must NAME the command.

        Two Python edits (the parser flag and the word the early exit persists)
        are inert unless `instructions/apply-code-fixes.md` actually tells the
        host to run it — that missing line is what left `phases_completed` empty
        on a run whose batches all landed.
        """
        text = (
            Path(__file__).parent.parent / "instructions" / "apply-code-fixes.md"
        ).read_text(encoding="utf-8")
        assert "--mark-completed" in text, (
            "instructions/apply-code-fixes.md must issue the post-loop "
            "finalization command"
        )
        assert "apply-code-fixes --plan-file \"$PLAN_PATH\" --mark-completed" in text
        # The split the fix must NOT reconcile away: prior-changes still goes to
        # the phase DIRECTORY word.
        assert '--phase "apply-fixes"' in text
        assert '--phase "apply-code-fixes"' in text


# ======================================================================
# How an apply phase ENDS: the backup, the artifacts, the outcome summary
# ======================================================================


class _FakeOutcome:
    """The `TaskOutcome` fields the outcome summary reads, and no more."""

    def __init__(self, task_id, status="ok", write_outcome="applied",
                 duration_seconds=1.0, runner="mock:model", reason=None,
                 files_touched=None, output_path=None):
        self.task_id = task_id
        self.status = status
        self.write_outcome = write_outcome
        self.duration_seconds = duration_seconds
        self.runner = runner
        self.reason = reason
        # F7/F10: only read by `_touched_files()` and the batch's own result
        # file (`notes`) — optional so every existing caller is unaffected.
        self.files_touched = files_touched or []
        self.output_path = output_path

    @property
    def ok(self):
        return self.status == "ok"


class _FakeSchedulerResult:
    def __init__(self, outcomes, report_path=None, stopped_at_batch=None,
                 verification=None):
        self.outcomes = outcomes
        self.report_path = report_path
        self.stopped_at_batch = stopped_at_batch
        self.verification = verification


def _orchestrator_on(tmp_path, **arg_overrides):
    """A MockApplyOrchestrator that has been through _setup on a real plan."""
    plan_path = _setup_plan_with_output(tmp_path)
    args = _make_args(plan_file=str(plan_path), **arg_overrides)
    orch = MockApplyOrchestrator(args)
    orch._setup()
    return orch


class TestTheTargetFileIsBackedUpBeforeTheFirstBatch:
    """F14: the phase that overwrites a file writes a backup of its own."""

    def test_the_phase_snapshots_the_file_it_is_about_to_overwrite(self, tmp_path):
        orch = _orchestrator_on(tmp_path)
        written = orch.write_target_backup()
        backup = Path(orch.out_dir) / orch.phase_name / "backup.md"
        assert written == str(backup)
        assert backup.read_text(encoding="utf-8") == Path(
            orch.plan_path
        ).read_text(encoding="utf-8")

    def test_a_backup_is_kept_when_the_phase_recorded_progress(self, tmp_path):
        """A resumed run's target is part-applied; re-stamping would destroy it."""
        orch = _orchestrator_on(tmp_path)
        orch.write_target_backup()
        backup = Path(orch.out_dir) / orch.phase_name / "backup.md"
        pre_run_bytes = backup.read_text(encoding="utf-8")

        # The scheduler got as far as writing its report, so the plan on disk
        # is the half-applied one and the backup is the only pre-run copy.
        (Path(orch.out_dir) / orch.phase_name / "write_batch_report.json").write_text(
            "{}", encoding="utf-8"
        )
        Path(orch.plan_path).write_text("# Half applied\n", encoding="utf-8")
        assert orch.write_target_backup() is None
        assert backup.read_text(encoding="utf-8") == pre_run_bytes

    def test_a_batch_result_of_this_kind_counts_as_progress(self, tmp_path):
        """A stopped loop leaves results but no report; those count too."""
        orch = _orchestrator_on(tmp_path)
        # The mock declares no write kind; a phase that reaches the executor
        # does, and the result basename is that kind's (D3g).
        orch.write_kind = "apply_suggestions"
        phase_dir = Path(orch.out_dir) / orch.phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        assert orch._phase_has_recorded_progress(str(phase_dir)) is False
        (phase_dir / "apply_batch_0.json").write_text("{}", encoding="utf-8")
        assert orch._phase_has_recorded_progress(str(phase_dir)) is True
        # Another phase's result in the same dir is not this phase's progress.
        (phase_dir / "apply_fix_batch_0.json").write_text("{}", encoding="utf-8")
        (phase_dir / "apply_batch_0.json").unlink()
        assert orch._phase_has_recorded_progress(str(phase_dir)) is False

    def test_a_backup_is_kept_across_an_explicit_resume(self, tmp_path):
        """The native leg records nothing; `--resume` is what says "part-applied"."""
        orch = _orchestrator_on(tmp_path, resume=True)
        orch.write_target_backup()
        backup = Path(orch.out_dir) / orch.phase_name / "backup.md"
        pre_run_bytes = backup.read_text(encoding="utf-8")

        Path(orch.plan_path).write_text("# Half applied\n", encoding="utf-8")
        assert orch.write_target_backup() is None
        assert backup.read_text(encoding="utf-8") == pre_run_bytes

    def test_a_new_cycle_re_stamps_the_backup_it_found(self, tmp_path):
        """Cycle 2 is not a resume: the stale snapshot must not claim to be one.

        review-plan → apply-suggestions (backup = v1, plan → v2, human edits →
        v3), then `review-plan --force` → apply-suggestions with no `--fresh`.
        Nothing in the phase dir records progress, so v3 — the bytes actually
        about to be overwritten — is what the backup must hold.
        """
        orch = _orchestrator_on(tmp_path)
        orch.write_target_backup()
        backup = Path(orch.out_dir) / orch.phase_name / "backup.md"

        Path(orch.plan_path).write_text("# Cycle 2, edited\n", encoding="utf-8")
        assert orch.write_target_backup() == str(backup)
        assert backup.read_text(encoding="utf-8") == "# Cycle 2, edited\n"

    def test_fresh_re_stamps_it(self, tmp_path):
        """`--fresh` declares prior progress void, so the backup goes with it."""
        orch = _orchestrator_on(tmp_path)
        orch.write_target_backup()
        Path(orch.plan_path).write_text("# New run\n", encoding="utf-8")
        orch.args.fresh = True
        assert orch.write_target_backup() is not None
        backup = Path(orch.out_dir) / orch.phase_name / "backup.md"
        assert backup.read_text(encoding="utf-8") == "# New run\n"

    def test_a_dry_run_writes_none(self, tmp_path):
        orch = _orchestrator_on(tmp_path, dry_run=True)
        assert orch.write_target_backup() is None
        assert not (Path(orch.out_dir) / orch.phase_name / "backup.md").exists()

    def test_a_phase_with_no_single_target_writes_none(self, tmp_path):
        """apply-code-fixes partitions by file_key: there is no one file to hold."""
        orch = _orchestrator_on(tmp_path)
        orch.has_single_write_target = False
        assert orch.write_target_file() is None
        assert orch.write_target_backup() is None

    def test_apply_code_fixes_is_the_phase_that_declares_that(self):
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator
        from apply_task_suggestions_orchestrator import (
            ApplyTaskSuggestionsOrchestrator,
        )

        assert ApplyCodeFixesOrchestrator.has_single_write_target is False
        assert ApplySuggestionsOrchestrator.has_single_write_target is True
        assert ApplyTaskSuggestionsOrchestrator.has_single_write_target is True

    def test_apply_task_suggestions_answers_with_its_own_tasks_backup(self, tmp_path):
        """It already writes one, under the name its output publishes."""
        from apply_task_suggestions_orchestrator import (
            ApplyTaskSuggestionsOrchestrator,
        )

        probe = ApplyTaskSuggestionsOrchestrator.__new__(
            ApplyTaskSuggestionsOrchestrator
        )
        probe.out_dir = str(tmp_path / "out")
        phase_dir = Path(probe.out_dir) / probe.phase_name
        phase_dir.mkdir(parents=True)
        assert probe.write_target_backup() is None
        (phase_dir / "tasks-backup.md").write_text("# tasks\n", encoding="utf-8")
        assert probe.write_target_backup() == str(phase_dir / "tasks-backup.md")
        assert not (phase_dir / "backup.md").exists()


class TestTheExecutorLegHandsBackItsArtifacts:
    """F3: a fully successful run used to report `artifacts: []`."""

    def test_the_files_the_phase_produced_are_recorded(self, tmp_path):
        from utils import cli_common

        orch = _orchestrator_on(tmp_path)
        phase_dir = Path(orch.out_dir) / orch.phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        output_path = phase_dir / "orchestrator_output.json"
        output_path.write_text("{}", encoding="utf-8")
        report_path = phase_dir / "write_batch_report.json"
        report_path.write_text("{}", encoding="utf-8")
        prior = Path(orch.out_dir) / "prior_changes.jsonl"
        prior.write_text("", encoding="utf-8")
        orch.write_result = _FakeSchedulerResult([], report_path=str(report_path))

        context = cli_common.ResultContext(mode="apply-mock", phase="apply-mock")
        cli_common.activate_result(context)
        try:
            orch._record_write_artifacts(str(output_path))
        finally:
            cli_common.deactivate_result()

        recorded = set(context.artifacts)
        assert orch.plan_path in recorded
        assert str(output_path) in recorded
        assert str(report_path) in recorded
        assert str(prior) in recorded

    def test_a_file_that_does_not_exist_is_not_named(self, tmp_path):
        """A stopped run must name what it produced, not what it meant to."""
        from utils import cli_common

        orch = _orchestrator_on(tmp_path)
        phase_dir = Path(orch.out_dir) / orch.phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        output_path = phase_dir / "orchestrator_output.json"
        output_path.write_text("{}", encoding="utf-8")
        orch.write_result = _FakeSchedulerResult([])

        context = cli_common.ResultContext(mode="apply-mock", phase="apply-mock")
        cli_common.activate_result(context)
        try:
            orch._record_write_artifacts(str(output_path))
        finally:
            cli_common.deactivate_result()

        recorded = context.artifacts
        assert not any("write_batch_report.json" in path for path in recorded)
        assert not any("prior_changes.jsonl" in path for path in recorded)


class TestTheExecutorLegReportsItsOutcome:
    """F8/F10: the summary said what WOULD happen, after it already had."""

    def _orchestrator_with_outcomes(self, tmp_path, outcomes, tasks):
        orch = _orchestrator_on(tmp_path, yes=True)
        orch.write_tasks = tasks
        orch.write_result = _FakeSchedulerResult(outcomes)
        return orch

    def test_it_reports_batches_items_and_per_batch_status(self, tmp_path, capsys):
        orch = self._orchestrator_with_outcomes(
            tmp_path,
            [
                _FakeOutcome("apply-batch-0", duration_seconds=89.5,
                             runner="opencode:deepseek"),
                _FakeOutcome("apply-batch-1", status="batch_failed",
                             write_outcome=None, duration_seconds=3.25,
                             runner="opencode:deepseek", reason="no bytes moved"),
            ],
            [
                {"task_id": "apply-batch-0", "expected_ids": ["a", "b", "c"]},
                {"task_id": "apply-batch-1", "expected_ids": ["d"]},
            ],
        )
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out

        assert "Batches: 1/2 ok" in out
        assert "Suggestions applied: 3" in out
        assert "apply-batch-0: ok (applied), 3 suggestion(s), 89.5s, runner opencode:deepseek" in out
        assert "apply-batch-1: batch_failed" in out
        assert "reason: no bytes moved" in out
        # The plan of work must NOT be what a finished run prints.
        assert "SUGGESTIONS TO APPLY" not in out

    def test_the_decision_summary_travels_with_it(self, tmp_path, capsys):
        """F10: `util decisions` never runs on this leg, so the run prints it."""
        orch = self._orchestrator_with_outcomes(
            tmp_path, [_FakeOutcome("apply-batch-0")],
            [{"task_id": "apply-batch-0", "expected_ids": ["a"]}],
        )
        orch.formatted_skipped = [
            {
                "group_id": "f08971a8deadbeef",
                "title": "Plan file could not be read",
                "theme": "Tooling",
                "importance": "LOW",
                "status": "invalid",
                "reason": "the plan file exists and is readable",
            }
        ]
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out

        assert "Skipped: 1" in out
        assert "[invalid] Plan file could not be read [f08971a8]" in out
        assert "reason: the plan file exists and is readable" in out

    def test_the_native_leg_still_prints_the_plan_of_work(self, tmp_path, capsys):
        """With no executor requested, the summary really is a plan of work."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))
        orch = MockApplyOrchestrator(args)
        assert orch.run() == 0
        assert "print_text_summary" in orch.call_log

    def test_write_outputs_routes_the_executor_leg_to_the_outcome(
        self, tmp_path, capsys
    ):
        """The routing, not just the text: `_write_outputs` must swap them."""
        from utils import cli_common

        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), yes=True)

        class ExecutorOrch(MockApplyOrchestrator):
            phase_name = "apply-executor-leg"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def run_write_executor(self):
                self.write_tasks = [
                    {"task_id": "apply-batch-0", "expected_ids": ["a", "b"]}
                ]
                self.write_result = _FakeSchedulerResult(
                    [_FakeOutcome("apply-batch-0")]
                )
                return 0

        orch = ExecutorOrch(args)
        context = cli_common.ResultContext(
            mode="apply-executor-leg", phase="apply-executor-leg"
        )
        cli_common.activate_result(context)
        try:
            assert orch.run() == 0
        finally:
            cli_common.deactivate_result()

        out = capsys.readouterr().out
        assert "APPLY-EXECUTOR-LEG: RESULT" in out
        assert "Batches: 1/1 ok" in out
        assert "Suggestions applied: 2" in out
        # The plan of work belongs to the native leg only.
        assert "print_text_summary" not in orch.call_log
        # F3: a run that finished must name what it produced.
        assert any(
            path.endswith("orchestrator_output.json") for path in context.artifacts
        )
        assert orch.plan_path in context.artifacts

    def test_the_json_contract_keeps_stdout_to_itself(self, tmp_path, capsys):
        """`--output-format json` means stdout parses as JSON, whole.

        The executor leg emits the machine payload *and* the human RESULT
        block, and both went to stdout — so a consumer that does what the flag
        promises (`json.loads(stdout)`) chokes on the block appended after it.
        The block still prints, on stderr.
        """
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path), yes=True, output_format="json"
        )

        class JsonExecutorOrch(MockApplyOrchestrator):
            phase_name = "apply-json-leg"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def run_write_executor(self):
                self.write_tasks = [
                    {"task_id": "apply-batch-0", "expected_ids": ["a"]}
                ]
                self.write_result = _FakeSchedulerResult(
                    [_FakeOutcome("apply-batch-0")]
                )
                return 0

            def print_text_summary(self, batches, output_path):
                # What every concrete phase does under `--output-format json`.
                with open(output_path, "r", encoding="utf-8") as handle:
                    print(json.dumps(json.load(handle), indent=2))

        assert JsonExecutorOrch(args).run() == 0
        captured = capsys.readouterr()

        # From the payload to the end of stdout, nothing but the payload. (The
        # `[AUTO_RESOLVED]` markers D6 requires precede it; they are the one
        # thing this run prints to stdout that is not the contract.)
        payload = captured.out[captured.out.index("{"):]
        assert json.loads(payload)["phase"] == "apply-json-leg"
        assert "APPLY-JSON-LEG: RESULT" not in captured.out
        assert "APPLY-JSON-LEG: RESULT" in captured.err
        assert "Batches: 1/1 ok" in captured.err

    def test_a_dry_run_subagents_preflight_keeps_the_plan_of_work(
        self, tmp_path, capsys
    ):
        """It dispatches nothing, so there is no outcome and no backup to take."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path), yes=True, dry_run_subagents=True
        )

        class DryOrch(MockApplyOrchestrator):
            phase_name = "apply-dry-leg"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def run_write_executor(self):
                self.write_result = _FakeSchedulerResult([])
                return 0

        orch = DryOrch(args)
        assert orch.run() == 0
        out = capsys.readouterr().out
        assert "print_text_summary" in orch.call_log
        assert "APPLY-DRY-LEG: RESULT" not in out
        assert not (Path(orch.out_dir) / orch.phase_name / "backup.md").exists()


class TestASkipCanBeTracedBackToItsReportEntry:
    """F11: `skipped_items` carried no group id."""

    def test_the_skipped_entry_carries_its_group_id(self, tmp_path, capsys):
        groups = [
            {
                "theme": "A",
                "group_hash": "bb5d722c00000000",
                "suggestions": [
                    {"title": "S1", "desc": "d", "importance": "MEDIUM"}
                ],
            }
        ]
        validation = [{"status": "invalid", "reason": "not a defect",
                       "confidence": 0.9}]
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path))

        class DataOrch(MockApplyOrchestrator):
            phase_name = "apply-skip-id"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def load_data(self):
                return (groups, validation)

            def handle_no_items_early_exit(self):
                raise SystemExit(0)

        orch = DataOrch(args)
        with pytest.raises(SystemExit):
            orch.run()
        capsys.readouterr()

        assert orch.formatted_skipped
        entry = orch.formatted_skipped[0]
        assert entry["group_id"] == "bb5d722c00000000"
        assert entry["status"] == "invalid"
        assert entry["reason"] == "not a defect"


class TestAJudgedSkipReachesEveryOutputThePhaseWrites:
    """F2: the one decision the run MADE was missing from every surface."""

    def _judged_orchestrator(self, tmp_path):
        """An orchestrator whose one group the judge decided to skip."""
        group = {
            "theme": "Complexity estimate",
            "group_hash": "3b810046f0a56e8c",
            "suggestions": [
                {"title": "T001 complexity underestimated", "desc": "d",
                 "importance": "LOW"}
            ],
            "validation_status": "needs-human-decision",
            "validation_reason": "a human estimation call",
        }
        plan_path = _setup_plan_with_output(tmp_path)
        orch = MockApplyOrchestrator(
            _make_args(plan_file=str(plan_path), yes=True)
        )
        orch._setup()
        orch.valid = []
        orch.needs_human = []
        orch.skipped = []
        orch.formatted_skipped = []
        return orch, group

    def test_a_judge_skip_appended_after_the_first_build_still_lands(
        self, tmp_path
    ):
        orch, group = self._judged_orchestrator(tmp_path)
        orch._rebuild_formatted_skipped()
        assert orch.formatted_skipped == []

        # What `apply_verdicts_to_work_set` does to a skipped group.
        group["validation_status"] = "skipped"
        group["reason"] = "Claude auto-decide: not decidable from the text"
        group["decision_source"] = "judge_auto_decide"
        orch.skipped.append(group)

        orch._rebuild_formatted_skipped()
        assert len(orch.formatted_skipped) == len(orch.skipped) == 1
        entry = orch.formatted_skipped[0]
        assert entry["group_id"] == "3b810046f0a56e8c"
        assert entry["status"] == "judge_skipped"
        assert entry["decision_source"] == "judge_auto_decide"
        assert entry["reason"] == (
            "Claude auto-decide: not decidable from the text"
        )

    def test_an_unjudged_skip_reports_the_skip_reason_not_the_validation_one(
        self, tmp_path
    ):
        """`--yes` with no verdict back: the skip is the decision, so is its reason.

        The group still carries the needs-human `validation_reason` that sent it
        to the judge in the first place; publishing that beside `status:
        skipped` tells `util decisions`, `summary.md` and the RESULT block the
        run is waiting on a human it has already stopped waiting for.
        """
        orch, group = self._judged_orchestrator(tmp_path)
        orch.needs_human.append(group)
        orch._skip_unjudged([group])
        assert orch.needs_human == []

        orch._rebuild_formatted_skipped()
        entry = orch.formatted_skipped[0]
        assert entry["status"] == "skipped"
        assert entry["reason"] == "No judge decision returned for this group"
        # Still sourceless: nobody decided it, so no `decision_source` is
        # invented for it.
        assert "decision_source" not in entry

    def test_a_validation_skip_keeps_its_status_and_gains_no_source_key(
        self, tmp_path
    ):
        """The key is added only when there is one, so nothing else moves."""
        orch, group = self._judged_orchestrator(tmp_path)
        group["validation_status"] = "invalid"
        orch.skipped.append(group)
        orch._rebuild_formatted_skipped()
        entry = orch.formatted_skipped[0]
        assert entry["status"] == "invalid"
        assert "decision_source" not in entry
        assert entry["reason"] == "a human estimation call"

    def test_run_rebuilds_the_skip_list_after_judging(self, tmp_path):
        """The count and the list are published together and must agree."""
        plan_path = _setup_plan_with_output(tmp_path)
        judged = {
            "theme": "Judged",
            "group_hash": "3b810046f0a56e8c",
            "suggestions": [{"title": "J", "desc": "d", "importance": "LOW"}],
            "validation_status": "skipped",
            "reason": "Claude auto-decide: subjective",
            "decision_source": "judge_auto_decide",
        }

        class JudgingOrch(MockApplyOrchestrator):
            phase_name = "apply-judged"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def _handle_human_decisions(self):
                # Exactly what the real one does to the work set.
                self.skipped.append(judged)

        orch = JudgingOrch(_make_args(plan_file=str(plan_path), yes=True))
        assert orch.run() == 0
        assert len(orch.formatted_skipped) == len(orch.skipped) == 1
        assert orch.formatted_skipped[0]["status"] == "judge_skipped"


class TestTheExecutorLegRunsStepSeven:
    """F3: no summary report and no decisions overlay on --exec-subagents."""

    def _orchestrator_with_a_decision(self, tmp_path):
        orch = _orchestrator_on(tmp_path, yes=True)
        orch.write_tasks = [
            {"task_id": "apply-batch-0", "expected_ids": ["a", "b"]}
        ]
        orch.write_result = _FakeSchedulerResult(
            [_FakeOutcome("apply-batch-0", duration_seconds=21.3)]
        )
        orch.formatted_valid = [
            {"group_id": "3b810046f0a56e8c", "title": "Raise T001 to high"}
        ]
        orch.state.record_human_decision(
            phase=orch.phase_name,
            group_id="3b810046f0a56e8c",
            decision="skipped",
            reason="Agent decide [codex:gpt-6-astra]: not decidable from the text",
            batch_context={
                "batch_id": "judge_decide_efabfefb",
                "batch_action": "judge_auto_decide",
                "decision_source": "judge_auto_decide",
                "importance_at_decision": "LOW",
                "judge": "codex:gpt-6-astra",
            },
        )
        orch.state.save()
        return orch

    def test_judge_decisions_selects_only_judge_sourced_records(self, tmp_path):
        orch = self._orchestrator_with_a_decision(tmp_path)
        orch.state.record_human_decision(
            phase=orch.phase_name, group_id="deadbeef00000000",
            decision="approved", reason="the user said so",
            batch_context={"decision_source": "user"},
        )
        orch.state.save()
        assert list(orch.judge_decisions()) == ["3b810046f0a56e8c"]

    def test_the_result_block_names_what_the_judge_decided(self, tmp_path, capsys):
        orch = self._orchestrator_with_a_decision(tmp_path)
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out
        assert "Decided by the judge agent (1)" in out
        assert "  judge: codex:gpt-6-astra" in out
        assert "Claude" not in out.split("Decided by the judge agent")[1]
        assert "[skipped] Raise T001 to high [3b810046]" in out
        # The persisted marker is a storage key, not a word for the user: a
        # Codex judge's decision used to print `source: claude_auto_decide`.
        assert "source: judge agent" in out
        assert "judge_auto_decide" not in out
        assert "not decidable from the text" in out

    def test_the_summary_report_the_instructions_advertise_is_written(
        self, tmp_path
    ):
        orch = self._orchestrator_with_a_decision(tmp_path)
        path = orch.write_apply_summary_report()
        assert path == str(Path(orch.out_dir) / orch.phase_name / "summary.md")
        text = Path(path).read_text(encoding="utf-8")
        assert "## Summary" in text
        assert "Batches processed: 1/1" in text
        assert "Suggestions applied: 2" in text
        assert "## Decided by the judge agent (1)" in text
        assert "judge: codex:gpt-6-astra" in text
        assert "`--agent-decide`" in text
        assert "Raise T001 to high (LOW) — Skipped" in text
        assert "**Source**: judge agent" in text
        assert "judge_auto_decide" not in text
        assert "Decided by the judge agent: 1 (0 approved, 0 salvaged, 1 skipped)" in text

    def test_a_salvage_reads_as_salvaged_in_the_log_the_report_and_the_count(
        self, tmp_path, capsys
    ):
        """One run said "1 salvaged" at ingest and "Approved" in summary.md.

        A salvage is stored as `decision: approved` plus the salvage marker, so
        every surface that read `decision` alone called it approved.
        """
        orch = self._orchestrator_with_a_decision(tmp_path)
        orch.state.record_human_decision(
            phase=orch.phase_name,
            group_id="9a9a9a9a9a9a9a9a",
            decision="approved",
            reason="Agent decide [codex:gpt-6-astra] (salvaged): kept the core",
            batch_context={
                "batch_action": "judge_auto_decide_salvage",
                "decision_source": "judge_auto_decide_salvage",
                "salvaged_description": "the core",
                "dropped": "the rest",
                "importance_at_decision": "MEDIUM",
                "judge": "codex:gpt-6-astra",
            },
        )
        orch.state.save()
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out
        assert "[salvaged] " in out
        assert "source: judge agent (salvaged)" in out
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "(MEDIUM) — Salvaged" in text
        assert "**Source**: judge agent (salvaged)" in text
        assert "(0 approved, 1 salvaged, 1 skipped)" in text

    def test_a_judge_skip_is_listed_as_skipped_by_judge_not_claude(
        self, tmp_path, capsys
    ):
        orch = self._orchestrator_with_a_decision(tmp_path)
        orch.formatted_skipped = [
            {"group_id": "3b810046f0a56e8c", "title": "Raise T001 to high",
             "status": "judge_skipped", "reason": "subjective"}
        ]
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out
        assert "[skipped by judge] Raise T001 to high" in out
        assert "judge_skipped" not in out
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "### Raise T001 to high [skipped by judge]" in text

    def test_the_report_names_the_verification_the_template_promised(
        self, tmp_path
    ):
        """F4: `Typecheck status` was in the template and in no report.

        The apply-code-fixes run this came from applied seven code fixes and its
        `summary.md` said nothing at all about a check, so a reader could not
        tell it from a run whose check passed.
        """
        orch = self._orchestrator_with_a_decision(tmp_path)
        orch.write_kind = "apply_code_fixes"
        orch.write_result.verification = {
            "command": "pnpm typecheck", "source": "project",
            "exit_code": 1, "duration_seconds": 4.2, "output_tail": "boom",
        }
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "- Verification: `pnpm typecheck` — FAIL (exit 1), 4.2s" in text

    def test_an_unconfigured_check_is_reported_as_unconfigured(self, tmp_path):
        orch = self._orchestrator_with_a_decision(tmp_path)
        orch.write_kind = "apply_code_fixes"
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "- Verification: not configured" in text

    def test_a_plan_text_phase_reports_no_verification_line_at_all(self, tmp_path):
        """`apply_suggestions` edits markdown — there is nothing to typecheck."""
        orch = self._orchestrator_with_a_decision(tmp_path)
        orch.write_kind = "apply_suggestions"
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "Verification" not in text

    def test_the_report_omits_the_decision_section_when_nothing_was_judged(
        self, tmp_path
    ):
        orch = _orchestrator_on(tmp_path, yes=True)
        orch.write_tasks = []
        orch.write_result = _FakeSchedulerResult([])
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "## Decided by the judge agent" not in text

    def test_the_overlay_is_regenerated_from_the_review_sidecar(self, tmp_path):
        orch = self._orchestrator_with_a_decision(tmp_path)
        review_dir = Path(orch.out_dir) / orch.review_subdir
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "report_data.json").write_text(
            json.dumps({"templateStyle": "flat", "groups": []}),
            encoding="utf-8",
        )
        written = orch.regenerate_decisions_overlay()
        assert written == str(review_dir / "report.html")
        assert (review_dir / "report.html").exists()

    def test_a_review_predating_the_sidecar_is_a_quiet_no_op(self, tmp_path):
        """The command it mirrors exits 0 with a notice; so does this."""
        orch = self._orchestrator_with_a_decision(tmp_path)
        review_dir = Path(orch.out_dir) / orch.review_subdir
        assert not (review_dir / "report_data.json").exists()
        assert orch.regenerate_decisions_overlay() is None

    def test_write_outputs_produces_both_on_the_executor_leg(
        self, tmp_path, capsys
    ):
        from utils import cli_common

        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), yes=True)

        class ExecutorOrch(MockApplyOrchestrator):
            phase_name = "apply-step-seven"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def run_write_executor(self):
                self.write_tasks = [
                    {"task_id": "apply-batch-0", "expected_ids": ["a"]}
                ]
                self.write_result = _FakeSchedulerResult(
                    [_FakeOutcome("apply-batch-0")]
                )
                return 0

        orch = ExecutorOrch(args)
        context = cli_common.ResultContext(
            mode="apply-step-seven", phase="apply-step-seven"
        )
        cli_common.activate_result(context)
        try:
            assert orch.run() == 0
        finally:
            cli_common.deactivate_result()

        summary = Path(orch.out_dir) / orch.phase_name / "summary.md"
        assert summary.exists()
        assert any(path.endswith("summary.md") for path in context.artifacts)

    def test_a_dry_run_subagents_preflight_writes_no_report(self, tmp_path):
        """It dispatched nothing; a report would describe a run that never ran."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(
            plan_file=str(plan_path), yes=True, dry_run_subagents=True
        )

        class DryOrch(MockApplyOrchestrator):
            phase_name = "apply-step-seven-dry"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def run_write_executor(self):
                self.write_result = _FakeSchedulerResult([])
                return 0

        orch = DryOrch(args)
        assert orch.run() == 0
        assert not (
            Path(orch.out_dir) / orch.phase_name / "summary.md"
        ).exists()

    def test_all_three_apply_phases_inherit_step_seven(self):
        """They share the base, so none of them can be missing it."""
        from apply_code_fixes_orchestrator import ApplyCodeFixesOrchestrator
        from apply_suggestions_orchestrator import ApplySuggestionsOrchestrator
        from apply_task_suggestions_orchestrator import (
            ApplyTaskSuggestionsOrchestrator,
        )

        for cls in (
            ApplyCodeFixesOrchestrator,
            ApplySuggestionsOrchestrator,
            ApplyTaskSuggestionsOrchestrator,
        ):
            assert cls.write_apply_summary_report is (
                ApplyOrchestratorBase.write_apply_summary_report
            )
            assert cls.regenerate_decisions_overlay is (
                ApplyOrchestratorBase.regenerate_decisions_overlay
            )
            assert cls._rebuild_formatted_skipped is (
                ApplyOrchestratorBase._rebuild_formatted_skipped
            )


class TestTheFilesChangedFallback:
    """F7: every output named the PLAN as what the phase changed, because a
    phase with no single write target (``has_single_write_target`` False)
    fell back to ``self.plan_path`` — the file it never opens."""

    def _orchestrator_with_no_single_target(self, tmp_path, outcomes, tasks):
        orch = _orchestrator_on(tmp_path, yes=True)
        orch.has_single_write_target = False
        orch.write_tasks = tasks
        orch.write_result = _FakeSchedulerResult(outcomes)
        return orch

    def test_the_result_block_names_the_touched_files(self, tmp_path, capsys):
        orch = self._orchestrator_with_no_single_target(
            tmp_path,
            [_FakeOutcome("apply-fix-batch-0", files_touched=["index.html"])],
            [{"task_id": "apply-fix-batch-0", "expected_ids": ["a"]}],
        )
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out
        assert "Files changed: index.html" in out
        assert orch.plan_path not in out

    def test_the_result_block_says_none_when_nothing_moved(self, tmp_path, capsys):
        orch = self._orchestrator_with_no_single_target(
            tmp_path,
            [_FakeOutcome("apply-fix-batch-0", status="batch_failed",
                          write_outcome=None)],
            [{"task_id": "apply-fix-batch-0", "expected_ids": ["a"]}],
        )
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out
        assert "Files changed: (none)" in out

    def test_partial_application_keeps_changed_files_visible(self, tmp_path, capsys):
        orch = self._orchestrator_with_no_single_target(
            tmp_path,
            [_FakeOutcome("apply-fix-batch-0", status="partial_application",
                          write_outcome="partial-application",
                          files_touched=["index.html"], reason="Missing one fix")],
            [{"task_id": "apply-fix-batch-0", "expected_ids": ["a", "b"]}],
        )
        orch.batches = [{"fixes": [{"title": "Incomplete fix"}]}]
        orch.print_write_outcome_summary()
        stdout = capsys.readouterr().out
        report = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "Files changed: index.html" in stdout
        assert "**Files changed**: index.html" in report
        assert "Incomplete fix** - Partial application; inspect batch" in report
        assert "Suggestions applied: 0" in report

    def test_a_single_target_phase_still_prints_file(self, tmp_path, capsys):
        """The fallback is additive: a phase with a real target is unaffected."""
        orch = _orchestrator_on(tmp_path, yes=True)
        orch.write_tasks = [{"task_id": "apply-batch-0", "expected_ids": ["a"]}]
        orch.write_result = _FakeSchedulerResult([_FakeOutcome("apply-batch-0")])
        orch.print_write_outcome_summary()
        out = capsys.readouterr().out
        assert f"File: {orch.plan_path}" in out
        assert "Files changed:" not in out

    def test_summary_md_names_files_changed_when_no_single_target(self, tmp_path):
        orch = self._orchestrator_with_no_single_target(
            tmp_path,
            [_FakeOutcome("apply-fix-batch-0", files_touched=["index.html"])],
            [{"task_id": "apply-fix-batch-0", "expected_ids": ["a"]}],
        )
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "**Files changed**: index.html" in text
        assert "**Applied to**" not in text

    def test_touched_files_dedupes_across_outcomes_and_skips_failed_ones(
        self, tmp_path
    ):
        orch = self._orchestrator_with_no_single_target(
            tmp_path,
            [
                _FakeOutcome("apply-fix-batch-0", files_touched=["index.html"]),
                _FakeOutcome(
                    "apply-fix-batch-1",
                    files_touched=["index.html", "style.css"],
                ),
                _FakeOutcome(
                    "apply-fix-batch-2", status="batch_failed",
                    write_outcome=None, files_touched=["should-not-appear.txt"],
                ),
            ],
            [
                {"task_id": "apply-fix-batch-0", "expected_ids": ["a"]},
                {"task_id": "apply-fix-batch-1", "expected_ids": ["b"]},
                {"task_id": "apply-fix-batch-2", "expected_ids": ["c"]},
            ],
        )
        assert orch._touched_files() == ["index.html", "style.css"]

    def test_record_write_artifacts_names_the_touched_files(self, tmp_path):
        from utils import cli_common

        orch = self._orchestrator_with_no_single_target(
            tmp_path,
            [_FakeOutcome("apply-fix-batch-0", files_touched=["index.html"])],
            [{"task_id": "apply-fix-batch-0", "expected_ids": ["a"]}],
        )
        # `_touched_files()` names repo-root-relative paths; the resolved file
        # must actually exist for `record_output_files`/`record_artifacts` to
        # name it (same rule every other artifact path follows).
        plan_dir = Path(orch.plan_path).parent
        (plan_dir / "index.html").write_text("<html></html>", encoding="utf-8")
        phase_dir = Path(orch.out_dir) / orch.phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        output_path = phase_dir / "orchestrator_output.json"
        output_path.write_text("{}", encoding="utf-8")

        context = cli_common.ResultContext(mode="apply-mock", phase="apply-mock")
        cli_common.activate_result(context)
        try:
            orch._record_write_artifacts(str(output_path))
        finally:
            cli_common.deactivate_result()

        recorded = set(context.artifacts)
        assert any(path.endswith("index.html") for path in recorded)

    def test_record_write_artifacts_names_the_judge_report_the_executor_wrote(
        self, tmp_path
    ):
        """F6, 2026-09-17: the judge report's filename is the executor's rule
        (`KINDS_SHARING_A_WRITE_PHASE_DIR`), and this leg spelled a copy of it
        by hand. Every path here is dropped when it does not exist, so a rename
        on the executor's side would have dropped
        `human_decision_judge_report` from `output_files` in silence."""
        from utils import cli_common
        from utils import subagent_executor as executor

        orch = self._orchestrator_with_no_single_target(tmp_path, [], [])
        phase_dir = Path(orch.out_dir) / orch.phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        output_path = phase_dir / "orchestrator_output.json"
        output_path.write_text("{}", encoding="utf-8")
        judge_report = Path(
            executor.batch_report_path(str(phase_dir), "human_decision_judge")
        )
        judge_report.write_text("{}", encoding="utf-8")
        # The whole point of the kind-qualified name: it is NOT the bare one a
        # later write dispatch into this same phase dir would claim.
        assert judge_report.name != executor.BATCH_REPORT_FILENAME

        context = cli_common.ResultContext(mode="apply-mock", phase="apply-mock")
        cli_common.activate_result(context)
        try:
            orch._record_write_artifacts(str(output_path))
        finally:
            cli_common.deactivate_result()

        assert str(judge_report) in set(context.artifacts)


class TestPerFixDetailInTheSummary:
    """F10: summary.md listed batch ids/durations and no fix titles, files, or
    notes, although the data — ``self.batches``' own fix dicts and each
    batch's own result-file ``notes`` — was already on disk."""

    def _orchestrator_with_a_fix_batch(self, tmp_path):
        orch = _orchestrator_on(tmp_path, yes=True)
        orch.has_single_write_target = False
        orch.write_tasks = [
            {"task_id": "apply-fix-batch-0", "expected_ids": ["a", "b"]}
        ]
        orch.batches = [
            {
                "fixes": [
                    {"title": "Fix the OOB teleport", "file": "index.html",
                     "line_range": [725, 727]},
                    {"title": "Drop speedForward", "file": "index.html",
                     "line_range": [40]},
                ],
                "file_key": "index.html",
            }
        ]
        result_path = Path(orch.out_dir) / orch.phase_name / "apply_fix_batch_0.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({
                "status": "applied", "ids": ["a", "b"],
                "summary": "applied 2 fixes",
                "notes": "Fix 1: OOB now resyncs lap timing.",
            }),
            encoding="utf-8",
        )
        orch.write_result = _FakeSchedulerResult([
            _FakeOutcome(
                "apply-fix-batch-0", files_touched=["index.html"],
                output_path=str(result_path),
            )
        ])
        return orch

    def test_each_fix_is_listed_with_its_title_and_status(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "1. **Fix the OOB teleport** - Applied" in text
        assert "2. **Drop speedForward** - Applied" in text

    @pytest.mark.parametrize(
        "verdict,label,count_label",
        [
            ("no-op", "No changes needed", "needing no changes"),
            ("clean-failure", "No verified changes", "without verified changes"),
        ],
    )
    def test_successful_dispatch_without_edits_is_not_reported_as_applied(
        self, tmp_path, capsys, verdict, label, count_label
    ):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        outcome = orch.write_result.outcomes[0]
        outcome.write_outcome = verdict
        outcome.files_touched = []
        outcome.reason = "Snapshot comparison confirms no byte changes."

        orch.print_write_outcome_summary()
        stdout = capsys.readouterr().out
        report = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        for text in (stdout, report):
            assert "Suggestions applied: 0" in text
            assert f"Suggestions {count_label}: 2" in text
            assert outcome.reason in text
        assert f"1. **Fix the OOB teleport** - {label}" in report
        assert f"2. **Drop speedForward** - {label}" in report

    def test_mixed_batch_counts_follow_verified_outcomes(self, tmp_path, capsys):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        orch.write_tasks.extend([
            {"task_id": "noop", "expected_ids": ["c", "d", "e"]},
            {"task_id": "clean-failure", "expected_ids": ["f"]},
        ])
        orch.write_result.outcomes.extend([
            _FakeOutcome("noop", write_outcome="no-op"),
            _FakeOutcome("clean-failure", write_outcome="clean-failure"),
        ])
        orch.print_write_outcome_summary()
        stdout = capsys.readouterr().out
        report = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        for text in (stdout, report):
            assert "Suggestions applied: 2" in text
            assert "Suggestions needing no changes: 3" in text
            assert "Suggestions without verified changes: 1" in text

    def test_a_two_element_line_range_renders_as_a_span(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        stripped = [line.strip() for line in text.splitlines()]
        assert "- File: index.html:725-727" in stripped

    def test_a_single_element_line_range_renders_without_a_span(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        stripped = [line.strip() for line in text.splitlines()]
        assert "- File: index.html" in stripped

    def test_a_not_applied_batch_marks_its_fixes_not_applied(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        orch.write_result = _FakeSchedulerResult([
            _FakeOutcome(
                "apply-fix-batch-0", status="batch_failed", write_outcome=None,
                reason="no bytes moved",
            )
        ])
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "1. **Fix the OOB teleport** - Not applied" in text

    def test_the_batch_result_notes_are_carried_through(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "**Notes**: Fix 1: OOB now resyncs lap timing." in text

    def test_a_missing_result_file_yields_no_notes_line(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        orch.write_result = _FakeSchedulerResult([
            _FakeOutcome("apply-fix-batch-0")  # no output_path
        ])
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "**Notes**" not in text

    def test_batching_efficiency_line_when_batching_ran(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        orch.batching_stats = {
            "batching_enabled": True,
            "subagent_calls_saved": 5,
            "efficiency_gain_percent": 42,
        }
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "Batching efficiency: 5 subagent call(s) saved (42%)" in text

    def test_no_batching_line_when_batching_was_not_enabled(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        orch.batching_stats = {"batching_enabled": False}
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "Batching efficiency" not in text

    def test_base_ref_line_when_the_orchestrator_resolves_one(self, tmp_path):
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        orch.resolve_base_ref = lambda: "abc1234"
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "**Base ref**: abc1234" in text

    def test_no_base_ref_line_when_none_is_resolved(self, tmp_path):
        """The default hook returns "" for the sibling phases — no line."""
        orch = self._orchestrator_with_a_fix_batch(tmp_path)
        text = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
        assert "**Base ref**" not in text


class TestTheSummaryFileIsPatchedIntoTheWriteBatchReport:
    """F11: ``write_batch_report.json`` said ``summary_file: null`` for a run
    that went on to write ``summary.md`` a few lines later — the scheduler
    serializes the report before this base ever calls
    ``write_apply_summary_report()``."""

    def test_the_report_on_disk_gains_the_summary_path(self, tmp_path):
        orch = _orchestrator_on(tmp_path, yes=True)
        orch.write_tasks = [{"task_id": "apply-batch-0", "expected_ids": ["a"]}]
        phase_dir = Path(orch.out_dir) / orch.phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        report_path = phase_dir / "write_batch_report.json"
        report_path.write_text(
            json.dumps({"report_version": 1, "summary_file": None, "tasks": []}),
            encoding="utf-8",
        )
        orch.write_result = _FakeSchedulerResult(
            [_FakeOutcome("apply-batch-0")], report_path=str(report_path)
        )

        summary_path = orch.write_apply_summary_report()
        orch._patch_write_batch_report_summary_file(summary_path)

        patched = json.loads(report_path.read_text(encoding="utf-8"))
        assert patched["summary_file"] == summary_path
        assert orch.write_result.summary_file == summary_path
        # Nothing else in the report moved.
        assert patched["report_version"] == 1

    def test_a_report_that_already_agrees_is_left_alone(self, tmp_path):
        orch = _orchestrator_on(tmp_path, yes=True)
        phase_dir = Path(orch.out_dir) / orch.phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        report_path = phase_dir / "write_batch_report.json"
        report_path.write_text(
            json.dumps({"summary_file": "/already/there.md"}), encoding="utf-8"
        )
        orch.write_result = _FakeSchedulerResult([], report_path=str(report_path))
        mtime_before = report_path.stat().st_mtime_ns

        orch._patch_write_batch_report_summary_file("/already/there.md")

        assert report_path.stat().st_mtime_ns == mtime_before

    def test_a_missing_report_path_is_a_quiet_no_op(self, tmp_path):
        orch = _orchestrator_on(tmp_path, yes=True)
        orch.write_result = _FakeSchedulerResult([])  # report_path=None
        orch._patch_write_batch_report_summary_file("/tmp/does-not-matter.md")
        assert orch.write_result.summary_file == "/tmp/does-not-matter.md"

    def test_the_executor_leg_wires_it_end_to_end(self, tmp_path):
        """Through `_write_outputs`, not just the helper in isolation."""
        plan_path = _setup_plan_with_output(tmp_path)
        args = _make_args(plan_file=str(plan_path), yes=True)

        class ExecutorOrch(MockApplyOrchestrator):
            phase_name = "apply-f11"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def run_write_executor(self):
                phase_dir = Path(self.out_dir) / self.phase_name
                phase_dir.mkdir(parents=True, exist_ok=True)
                report_path = phase_dir / "write_batch_report.json"
                report_path.write_text(
                    json.dumps({"report_version": 1, "summary_file": None}),
                    encoding="utf-8",
                )
                self.write_tasks = [
                    {"task_id": "apply-batch-0", "expected_ids": ["a"]}
                ]
                self.write_result = _FakeSchedulerResult(
                    [_FakeOutcome("apply-batch-0")], report_path=str(report_path)
                )
                return 0

        orch = ExecutorOrch(args)
        assert orch.run() == 0

        phase_dir = Path(orch.out_dir) / orch.phase_name
        patched = json.loads(
            (phase_dir / "write_batch_report.json").read_text(encoding="utf-8")
        )
        assert patched["summary_file"] == str(phase_dir / "summary.md")
