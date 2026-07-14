#!/usr/bin/env python3
"""End-to-end tests for smart batching edge cases.

These tests verify the batching logic for code fixes and suggestions, including:
- test_smart_batching_file_proximity: Items in same file grouped together
- test_smart_batching_high_isolation: HIGH importance items get their own batch
- test_smart_batching_security_isolation: Security-related items isolated
- test_batching_stats_accuracy: Verify savings percentage calculation

All tests run in isolated tmp_path directories with mock LLM providers.

Usage:
    uv run -- pytest tests/test_e2e_batching.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    SkillRunner,
    FixtureManager,
    MockProvider,
    AssertionHelpers,
)

# Also import the batcher modules directly for unit-style e2e tests
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.code_fix_batcher import (
    batch_code_fixes,
    estimate_batch_processing_stats as estimate_fix_stats,
    is_high_risk_fix,
    CodeFixBatch,
)
from utils.suggestion_batcher import (
    group_suggestions_for_subagents,
    estimate_batch_processing_stats as estimate_suggestion_stats,
    SuggestionBatch,
)


def _create_code_review_phase(
    fixture: "FixturePlan",
    grouped_data: list,
    validation_data: list,
) -> Path:
    """Create pre-populated code-review phase outputs.

    Args:
        fixture: The FixturePlan to populate
        grouped_data: List of grouped issues from code review
        validation_data: List of validation results

    Returns:
        Path to the code-review directory
    """
    # Create code-review directory
    code_review_dir = fixture.ensure_phase_dir("code-review")

    # Write grouped.json
    grouped_path = code_review_dir / "grouped.json"
    grouped_path.write_text(json.dumps(grouped_data, indent=2), encoding="utf-8")

    # Write validation.json
    validation_path = code_review_dir / "validation.json"
    validation_path.write_text(json.dumps(validation_data, indent=2), encoding="utf-8")

    return code_review_dir


def _make_code_fix(
    file: str = "src/main.py",
    description: str = "Fix something",
    importance: str = "MEDIUM",
    fix_type: str = "bug",
    line_range: tuple = (10, 15),
    title: str = "Fix issue",
    anchor_text: str = "some code",
) -> dict:
    """Helper to create a test code fix."""
    return {
        "file": file,
        "description": description,
        "importance": importance,
        "type": fix_type,
        "line_range": line_range,
        "title": title,
        "anchor_text": anchor_text,
    }


def _make_suggestion(
    title: str = "Test suggestion",
    description: str = "Test description",
    importance: str = "MEDIUM",
    suggestion_type: str = "modification",
    reference: str = "Section 1",
) -> dict:
    """Helper to create a test suggestion."""
    return {
        "title": title,
        "description": description,
        "importance": importance,
        "type": suggestion_type,
        "reference": reference,
    }


class TestSmartBatchingFileProximity:
    """Test that items in the same file are grouped together."""

    def test_code_fixes_same_file_grouped(self):
        """Verify code fixes targeting the same file are batched together.

        The batcher should group fixes by file path as the primary grouping
        heuristic, keeping related changes together for atomic commits.
        """
        # Create fixes across multiple files
        fixes = [
            _make_code_fix(file="src/api/handler.py", line_range=(10, 15), title="Fix 1", importance="MEDIUM"),
            _make_code_fix(file="src/api/handler.py", line_range=(20, 25), title="Fix 2", importance="MEDIUM"),
            _make_code_fix(file="src/api/handler.py", line_range=(30, 35), title="Fix 3", importance="MEDIUM"),
            _make_code_fix(file="src/utils/helpers.py", line_range=(5, 10), title="Fix 4", importance="MEDIUM"),
            _make_code_fix(file="src/utils/helpers.py", line_range=(15, 20), title="Fix 5", importance="MEDIUM"),
            _make_code_fix(file="src/models/user.py", line_range=(1, 5), title="Fix 6", importance="MEDIUM"),
        ]

        batches = batch_code_fixes(fixes)

        # Extract file keys from batches
        file_keys = [b.file_key for b in batches]

        # All 3 handler.py fixes should be in one batch (unless it exceeds limits)
        handler_batches = [b for b in batches if b.file_key == "src/api/handler.py"]
        assert len(handler_batches) == 1, "Handler fixes should be in one batch"
        assert handler_batches[0].size == 3, "Handler batch should have 3 fixes"

        # Both helpers.py fixes should be in one batch
        helper_batches = [b for b in batches if b.file_key == "src/utils/helpers.py"]
        assert len(helper_batches) == 1, "Helper fixes should be in one batch"
        assert helper_batches[0].size == 2, "Helper batch should have 2 fixes"

        # user.py fix should be in its own batch
        user_batches = [b for b in batches if b.file_key == "src/models/user.py"]
        assert len(user_batches) == 1, "User model fix should be in one batch"
        assert user_batches[0].size == 1, "User model batch should have 1 fix"

    def test_code_fixes_different_files_separate_batches(self):
        """Verify fixes in different files get separate batches."""
        fixes = [
            _make_code_fix(file="src/file_a.py", title="Fix A", importance="MEDIUM"),
            _make_code_fix(file="src/file_b.py", title="Fix B", importance="MEDIUM"),
            _make_code_fix(file="src/file_c.py", title="Fix C", importance="MEDIUM"),
        ]

        batches = batch_code_fixes(fixes)

        # Should have 3 batches, one for each file
        assert len(batches) == 3, f"Expected 3 batches for 3 files, got {len(batches)}"

        file_keys = {b.file_key for b in batches}
        assert file_keys == {"src/file_a.py", "src/file_b.py", "src/file_c.py"}

    def test_code_fixes_line_ordering_within_file(self):
        """Verify fixes within a file are ordered by line number."""
        fixes = [
            _make_code_fix(file="src/main.py", line_range=(100, 105), title="Fix 100", importance="MEDIUM"),
            _make_code_fix(file="src/main.py", line_range=(10, 15), title="Fix 10", importance="MEDIUM"),
            _make_code_fix(file="src/main.py", line_range=(50, 55), title="Fix 50", importance="MEDIUM"),
        ]

        batches = batch_code_fixes(fixes)

        assert len(batches) == 1, "All fixes should be in one batch"

        # Check line order within the batch
        batch_lines = [f.get("line_range", [0])[0] for f in batches[0].fixes]
        assert batch_lines == [10, 50, 100], f"Fixes should be in line order, got {batch_lines}"

    def test_file_proximity_respects_batch_limits(self):
        """Verify file grouping respects max batch size limits."""
        # Create 7 fixes for same file (more than MAX_FIXES_PER_BATCH=3)
        fixes = [
            _make_code_fix(
                file="src/large_file.py",
                line_range=(i * 10, i * 10 + 5),
                title=f"Fix {i}",
                importance="MEDIUM",
            )
            for i in range(7)
        ]

        batches = batch_code_fixes(fixes)

        # 7 fixes with max 3 per batch = 3 batches (3 + 3 + 1)
        assert len(batches) == 3, f"Expected 3 batches for 7 fixes, got {len(batches)}"

        # All batches should be for the same file
        for batch in batches:
            assert batch.file_key == "src/large_file.py"

        # Verify batch sizes
        sizes = sorted([b.size for b in batches], reverse=True)
        assert sizes == [3, 3, 1], f"Expected sizes [3, 3, 1], got {sizes}"


class TestSmartBatchingHighIsolation:
    """Test that HIGH importance items get their own batch (not grouped)."""

    def test_high_importance_fixes_isolated(self):
        """Verify HIGH importance fixes are always in their own batches.

        HIGH importance fixes should never be grouped with other fixes,
        even if they target the same file, to ensure careful review.
        """
        fixes = [
            _make_code_fix(file="src/main.py", importance="HIGH", title="Critical fix 1"),
            _make_code_fix(file="src/main.py", importance="HIGH", title="Critical fix 2"),
            _make_code_fix(file="src/main.py", importance="MEDIUM", title="Normal fix"),
        ]

        batches = batch_code_fixes(fixes)

        # Should have 3 batches: 2 HIGH (isolated) + 1 MEDIUM
        assert len(batches) == 3, f"Expected 3 batches (2 HIGH isolated + 1 MEDIUM), got {len(batches)}"

        # Count batches with HIGH importance fixes
        high_batches = [
            b for b in batches
            if any(f.get("importance", "").upper() == "HIGH" for f in b.fixes)
        ]
        assert len(high_batches) == 2, "Each HIGH fix should have its own batch"

        # Each HIGH batch should have exactly 1 fix
        for batch in high_batches:
            assert batch.size == 1, f"HIGH batch should have exactly 1 fix, got {batch.size}"

    def test_is_high_risk_fix_detection(self):
        """Verify is_high_risk_fix correctly identifies HIGH importance."""
        high_fix = _make_code_fix(importance="HIGH", fix_type="bug")
        medium_fix = _make_code_fix(importance="MEDIUM", fix_type="bug")
        low_fix = _make_code_fix(importance="LOW", fix_type="bug")

        assert is_high_risk_fix(high_fix) is True, "HIGH importance should be high risk"
        assert is_high_risk_fix(medium_fix) is False, "MEDIUM importance should not be high risk"
        assert is_high_risk_fix(low_fix) is False, "LOW importance should not be high risk"

    def test_high_importance_case_insensitive(self):
        """Verify HIGH importance detection is case-insensitive."""
        fixes = [
            _make_code_fix(importance="high", title="Lowercase high"),
            _make_code_fix(importance="HIGH", title="Uppercase HIGH"),
            _make_code_fix(importance="High", title="Mixed case High"),
        ]

        batches = batch_code_fixes(fixes)

        # All 3 should be isolated
        assert len(batches) == 3, "All HIGH fixes should be isolated regardless of case"

        for batch in batches:
            assert batch.size == 1, "Each HIGH fix should be alone in its batch"

    def test_high_mixed_with_medium_low(self):
        """Verify HIGH is isolated while MEDIUM/LOW can be grouped."""
        fixes = [
            _make_code_fix(file="src/main.py", importance="HIGH", title="Critical"),
            _make_code_fix(file="src/main.py", importance="MEDIUM", title="Normal 1"),
            _make_code_fix(file="src/main.py", importance="MEDIUM", title="Normal 2"),
            _make_code_fix(file="src/main.py", importance="LOW", title="Minor"),
        ]

        batches = batch_code_fixes(fixes)

        # HIGH should be isolated, MEDIUM+LOW can be grouped
        high_batch = None
        other_batch = None
        for batch in batches:
            if any(f.get("importance", "").upper() == "HIGH" for f in batch.fixes):
                high_batch = batch
            else:
                other_batch = batch

        assert high_batch is not None, "Should have a HIGH batch"
        assert high_batch.size == 1, "HIGH batch should have exactly 1 fix"

        assert other_batch is not None, "Should have a non-HIGH batch"
        assert other_batch.size == 3, "MEDIUM+LOW should be grouped (3 fixes)"


class TestSmartBatchingSecurityIsolation:
    """Test that security-related items are isolated in their own batch."""

    def test_security_fixes_isolated(self):
        """Verify security type fixes are always in their own batches.

        Security fixes should never be grouped with other fixes to ensure
        each security issue gets individual attention and review.
        """
        fixes = [
            _make_code_fix(file="src/auth.py", fix_type="security", importance="LOW", title="SQL injection fix"),
            _make_code_fix(file="src/auth.py", fix_type="bug", importance="MEDIUM", title="Normal bug fix"),
            _make_code_fix(file="src/auth.py", fix_type="security", importance="MEDIUM", title="XSS fix"),
        ]

        batches = batch_code_fixes(fixes)

        # Should have 3 batches: 2 security (isolated) + 1 bug
        assert len(batches) == 3, f"Expected 3 batches (2 security isolated + 1 bug), got {len(batches)}"

        # Count security batches
        security_batches = [
            b for b in batches
            if any(f.get("type", "").lower() == "security" for f in b.fixes)
        ]
        assert len(security_batches) == 2, "Each security fix should have its own batch"

        for batch in security_batches:
            assert batch.size == 1, "Security batch should have exactly 1 fix"

    def test_is_high_risk_fix_detects_security(self):
        """Verify is_high_risk_fix correctly identifies security type."""
        security_fix = _make_code_fix(fix_type="security", importance="LOW")
        bug_fix = _make_code_fix(fix_type="bug", importance="MEDIUM")

        assert is_high_risk_fix(security_fix) is True, "Security type should be high risk"
        assert is_high_risk_fix(bug_fix) is False, "Bug type should not be high risk by default"

    def test_security_type_case_insensitive(self):
        """Verify security detection is case-insensitive."""
        fixes = [
            _make_code_fix(fix_type="security", title="lowercase"),
            _make_code_fix(fix_type="SECURITY", title="uppercase"),
            _make_code_fix(fix_type="Security", title="mixed"),
        ]

        batches = batch_code_fixes(fixes)

        # All 3 should be isolated
        assert len(batches) == 3, "All security fixes should be isolated regardless of case"

    def test_security_overrides_file_grouping(self):
        """Verify security isolation takes precedence over file grouping."""
        fixes = [
            _make_code_fix(file="src/main.py", fix_type="security", title="Security 1"),
            _make_code_fix(file="src/main.py", fix_type="security", title="Security 2"),
            _make_code_fix(file="src/main.py", fix_type="bug", title="Bug"),
            _make_code_fix(file="src/main.py", fix_type="style", title="Style"),
        ]

        batches = batch_code_fixes(fixes)

        # 2 security batches (isolated) + 1 batch for bug+style
        assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

        # Verify security fixes are isolated
        for batch in batches:
            if any(f.get("type", "").lower() == "security" for f in batch.fixes):
                assert batch.size == 1, "Security batch should have exactly 1 fix"


class TestBatchingStatsAccuracy:
    """Test that batching statistics are calculated accurately."""

    def test_code_fix_stats_basic(self):
        """Verify basic code fix batching stats calculation."""
        # Create batches manually for controlled testing
        batch1 = CodeFixBatch()
        batch1.add(_make_code_fix(fix_type="bug"))
        batch1.add(_make_code_fix(fix_type="bug"))
        batch1.add(_make_code_fix(fix_type="bug"))

        batch2 = CodeFixBatch()
        batch2.add(_make_code_fix(fix_type="improvement"))
        batch2.add(_make_code_fix(fix_type="improvement"))

        batches = [batch1, batch2]
        stats = estimate_fix_stats(batches)

        # 5 fixes in 2 batches = 3 calls saved
        assert stats["total_fixes"] == 5
        assert stats["total_batches"] == 2
        assert stats["subagent_calls_saved"] == 3

        # Efficiency: (5 - 2) / 5 * 100 = 60%
        expected_efficiency = 60.0
        assert stats["efficiency_gain_percent"] == expected_efficiency, \
            f"Expected {expected_efficiency}%, got {stats['efficiency_gain_percent']}%"

    def test_code_fix_stats_no_savings(self):
        """Verify stats when no batching savings occur (all isolated)."""
        # Create 3 batches with 1 fix each (e.g., all HIGH importance)
        batches = []
        for i in range(3):
            batch = CodeFixBatch()
            batch.add(_make_code_fix(importance="HIGH", title=f"Fix {i}"))
            batches.append(batch)

        stats = estimate_fix_stats(batches)

        # 3 fixes in 3 batches = 0 calls saved
        assert stats["total_fixes"] == 3
        assert stats["total_batches"] == 3
        assert stats["subagent_calls_saved"] == 0
        assert stats["efficiency_gain_percent"] == 0.0

    def test_code_fix_stats_single_vs_multi_batch_counts(self):
        """Verify single and multi-fix batch counts are accurate."""
        batch1 = CodeFixBatch()
        batch1.add(_make_code_fix())  # 1 fix

        batch2 = CodeFixBatch()
        batch2.add(_make_code_fix())
        batch2.add(_make_code_fix())  # 2 fixes

        batch3 = CodeFixBatch()
        batch3.add(_make_code_fix())
        batch3.add(_make_code_fix())
        batch3.add(_make_code_fix())  # 3 fixes

        stats = estimate_fix_stats([batch1, batch2, batch3])

        assert stats["single_fix_batches"] == 1
        assert stats["multi_fix_batches"] == 2

    def test_code_fix_stats_average_batch_size(self):
        """Verify average batch size calculation."""
        batch1 = CodeFixBatch()
        batch1.add(_make_code_fix())  # 1

        batch2 = CodeFixBatch()
        batch2.add(_make_code_fix())
        batch2.add(_make_code_fix())  # 2

        batch3 = CodeFixBatch()
        batch3.add(_make_code_fix())
        batch3.add(_make_code_fix())
        batch3.add(_make_code_fix())  # 3

        stats = estimate_fix_stats([batch1, batch2, batch3])

        # (1 + 2 + 3) / 3 = 2.0
        assert stats["average_batch_size"] == 2.0

    def test_code_fix_stats_max_batch_size(self):
        """Verify max batch size tracking."""
        batch1 = CodeFixBatch()
        batch1.add(_make_code_fix())

        batch2 = CodeFixBatch()
        for _ in range(3):
            batch2.add(_make_code_fix())

        stats = estimate_fix_stats([batch1, batch2])

        assert stats["max_batch_size"] == 3

    def test_suggestion_stats_basic(self):
        """Verify basic suggestion batching stats calculation."""
        batch1 = SuggestionBatch()
        batch1.add(_make_suggestion())
        batch1.add(_make_suggestion())
        batch1.add(_make_suggestion())
        batch1.add(_make_suggestion())  # 4 suggestions

        batch2 = SuggestionBatch()
        batch2.add(_make_suggestion())
        batch2.add(_make_suggestion())  # 2 suggestions

        batches = [batch1, batch2]
        stats = estimate_suggestion_stats(batches)

        # 6 suggestions in 2 batches = 4 calls saved
        assert stats["total_suggestions"] == 6
        assert stats["total_batches"] == 2
        assert stats["subagent_calls_saved"] == 4

        # Efficiency: (6 - 2) / 6 * 100 = 66.7%
        expected_efficiency = 66.7
        assert stats["efficiency_gain_percent"] == expected_efficiency, \
            f"Expected {expected_efficiency}%, got {stats['efficiency_gain_percent']}%"

    def test_suggestion_stats_empty_batches(self):
        """Verify stats for empty batch list."""
        stats = estimate_suggestion_stats([])

        assert stats["total_suggestions"] == 0
        assert stats["total_batches"] == 0
        assert stats["subagent_calls_saved"] == 0
        assert stats["efficiency_gain_percent"] == 0
        assert stats["average_batch_size"] == 0
        assert stats["max_batch_size"] == 0

    def test_stats_via_actual_batching(self):
        """Verify stats from actual batching operation match expected values."""
        # Create fixes that should result in specific batching behavior
        fixes = [
            # Group 1: same file, should batch together (3 fixes)
            _make_code_fix(file="src/a.py", line_range=(10, 15), importance="MEDIUM"),
            _make_code_fix(file="src/a.py", line_range=(20, 25), importance="MEDIUM"),
            _make_code_fix(file="src/a.py", line_range=(30, 35), importance="MEDIUM"),
            # Group 2: different file (1 fix)
            _make_code_fix(file="src/b.py", importance="MEDIUM"),
            # Group 3: HIGH importance (isolated, 1 fix)
            _make_code_fix(file="src/c.py", importance="HIGH"),
        ]

        batches = batch_code_fixes(fixes)
        stats = estimate_fix_stats(batches)

        # Should have 3 batches: 1 for a.py (3 fixes), 1 for b.py, 1 for HIGH
        assert stats["total_fixes"] == 5
        assert stats["total_batches"] == 3

        # Calls saved: 5 - 3 = 2
        assert stats["subagent_calls_saved"] == 2

        # Efficiency: (5 - 3) / 5 * 100 = 40%
        assert stats["efficiency_gain_percent"] == 40.0


class TestBatchingViaOrchestrator:
    """Test batching behavior through the apply_fixes orchestrator."""

    def test_batching_output_structure(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify the orchestrator outputs proper batching structure and stats.

        This test runs the apply_fixes orchestrator and verifies that:
        1. The output contains a batches array
        2. The output contains batching_stats
        3. The stats match the expected calculation
        """
        plan = fixture_manager.create_plan("batch-structure-test", "# Test Plan\n")

        # Create grouped data with fixes that will result in known batching
        grouped_data = [
            {
                "theme": f"Issue {i}",
                "category": "test",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": f"Issue {i} in file_a.py",
                    "desc": f"Fix for issue {i}",
                    "importance": "MEDIUM",
                    "type": "bug",
                    "file": "src/file_a.py",  # All same file - should batch
                    "line_range": [i * 10, i * 10 + 5],
                    "source_model": "cursor-agent",
                }]
            }
            for i in range(3)
        ]

        # All valid
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

        assert result.success, f"apply_fixes should succeed: {result.stderr}"

        try:
            output = json.loads(result.stdout)

            # Verify batching structure
            assert "batches" in output, "Output should contain batches"
            assert isinstance(output["batches"], list), "batches should be a list"

            # Verify batching_stats
            assert "batching_stats" in output, "Output should contain batching_stats"
            stats = output["batching_stats"]

            # 3 MEDIUM fixes in same file should result in 1 batch
            # (assuming they fit under MAX_FIXES_PER_BATCH)
            assert stats["total_fixes"] == 3 or stats.get("original_count") == 3

        except json.JSONDecodeError:
            pytest.fail("Output should be valid JSON")

    def test_high_importance_isolation_via_orchestrator(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """Verify HIGH importance items are isolated in orchestrator output."""
        plan = fixture_manager.create_plan("high-isolation-test", "# Test Plan\n")

        # Create fixes with mixed importance
        grouped_data = [
            {
                "theme": "High priority fix",
                "category": "security",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": "Critical security issue",
                    "desc": "Fix this immediately",
                    "importance": "HIGH",
                    "type": "security",
                    "file": "src/auth.py",
                    "source_model": "cursor-agent",
                }]
            },
            {
                "theme": "Normal fix 1",
                "category": "bug",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": "Normal bug",
                    "desc": "Regular bug fix",
                    "importance": "MEDIUM",
                    "type": "bug",
                    "file": "src/auth.py",  # Same file as HIGH
                    "source_model": "cursor-agent",
                }]
            },
            {
                "theme": "Normal fix 2",
                "category": "bug",
                "models": ["cursor-agent"],
                "suggestions": [{
                    "title": "Another normal bug",
                    "desc": "Another regular bug fix",
                    "importance": "MEDIUM",
                    "type": "bug",
                    "file": "src/auth.py",  # Same file
                    "source_model": "cursor-agent",
                }]
            },
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

        assert result.success, f"apply_fixes should succeed: {result.stderr}"

        try:
            output = json.loads(result.stdout)
            batches = output.get("batches", [])

            # Should have at least 2 batches:
            # 1 for HIGH (isolated) + 1 for the MEDIUM fixes
            assert len(batches) >= 2, \
                f"Expected at least 2 batches (HIGH isolated), got {len(batches)}"

            # Find the HIGH importance batch
            high_batch = None
            for batch in batches:
                fixes = batch.get("fixes", [])
                if any(f.get("importance", "").upper() == "HIGH" for f in fixes):
                    high_batch = batch
                    break

            if high_batch:
                # HIGH batch should have exactly 1 fix
                assert len(high_batch.get("fixes", [])) == 1, \
                    "HIGH importance batch should have exactly 1 fix"

        except json.JSONDecodeError:
            pytest.fail("Output should be valid JSON")


class TestSuggestionBatchingProximity:
    """Test suggestion batching by section/reference proximity."""

    def test_same_section_grouped(self):
        """Verify suggestions targeting the same section are grouped."""
        suggestions = [
            _make_suggestion(reference="Step 1: Setup", title="Add 1", suggestion_type="addition"),
            _make_suggestion(reference="Step 1: Setup", title="Add 2", suggestion_type="addition"),
            _make_suggestion(reference="Step 2: Implement", title="Add 3", suggestion_type="addition"),
        ]

        batches = group_suggestions_for_subagents(suggestions)

        # Step 1 suggestions should be in one batch, Step 2 in another
        step1_batch = None
        step2_batch = None
        for batch in batches:
            if "step_1" in batch.section_key:
                step1_batch = batch
            elif "step_2" in batch.section_key:
                step2_batch = batch

        assert step1_batch is not None, "Should have a Step 1 batch"
        assert step1_batch.size == 2, "Step 1 batch should have 2 suggestions"

        assert step2_batch is not None, "Should have a Step 2 batch"
        assert step2_batch.size == 1, "Step 2 batch should have 1 suggestion"

    def test_deletions_always_isolated(self):
        """Verify deletion suggestions are always in their own batches."""
        suggestions = [
            _make_suggestion(suggestion_type="deletion", title="Delete 1", reference="Section A"),
            _make_suggestion(suggestion_type="deletion", title="Delete 2", reference="Section A"),
            _make_suggestion(suggestion_type="addition", title="Add 1", reference="Section A"),
        ]

        batches = group_suggestions_for_subagents(suggestions)

        # Each deletion should be isolated
        deletion_batches = [b for b in batches if b.batch_type == "deletion"]
        assert len(deletion_batches) == 2, "Each deletion should have its own batch"

        for batch in deletion_batches:
            assert batch.size == 1, "Deletion batch should have exactly 1 suggestion"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
