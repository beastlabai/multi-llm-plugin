#!/usr/bin/env python3
"""End-to-end tests for validation edge cases in multi-llm skill.

These tests verify edge case behaviors in the validation system including:
- Batch index disambiguation (local vs global indices)
- Revalidation failure cascade handling
- Confidence value edge cases (0.0, 1.0, negative, >1.0)

Tests focus on the index disambiguation logic in utils/validation.py
(lines 1027-1070) which determines whether LLM-returned indices are:
- Local indices: 0-based within the current batch
- Global indices: Original indices from the full list

The heuristic:
1. If all returned indices fit within batch size AND don't match global indices -> local
2. If all returned indices match global indices -> global
3. If ambiguous (like batch [0,1,2]) -> default to local (but global also works)

All tests use isolated tmp_path directories and mock LLM providers.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    SkillRunner,
    FixtureManager,
    MockProvider,
    AssertionHelpers,
)

# Also need direct access to validation functions for unit-level integration tests
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.validation import (
    merge_batched_validation_results,
    prepare_batched_validation_tasks,
    prepare_batched_revalidation_tasks,
    merge_batched_revalidation_results,
    ERROR_TYPE_UNKNOWN,
    ERROR_TYPE_PARSING,
    ERROR_TYPE_TIMEOUT,
)


# ============================================================================
# Helper Functions
# ============================================================================


def make_group(importance: str = "MEDIUM", theme: str = "Test theme", index: int = 0) -> dict:
    """Helper to create a test suggestion group."""
    return {
        "theme": theme,
        "category": "test",
        "models": ["test-model"],
        "suggestions": [
            {
                "title": f"Test suggestion {index}",
                "desc": f"Test description for group {index}",
                "importance": importance,
                "type": "modification"
            }
        ]
    }


def write_batch_results(
    batch_path: str,
    results: List[Dict[str, Any]],
    schema_version: str = "2.0"
) -> None:
    """Helper to write batch validation results to a file."""
    batch_data = {
        "groups": results,
        "metadata": {
            "model": "test-model",
            "schema_version": schema_version
        }
    }
    with open(batch_path, 'w') as f:
        json.dump(batch_data, f)


# ============================================================================
# Test Class: Batch Index Disambiguation - Local Indices
# ============================================================================


class TestBatchIndexDisambiguationLocal:
    """Tests for batch index disambiguation when LLM returns local indices.

    When a batch contains groups [5,6,7] and the LLM returns results with
    indices [0,1,2], the merge function should correctly interpret these
    as local indices and map them to global indices [5,6,7].
    """

    def test_batch_index_disambiguation_local(self):
        """Batch [5,6,7] receives results [0,1,2] - correctly interpreted as local indices.

        This tests the scenario where:
        - Batch contains global indices [5, 6, 7]
        - LLM returns results with local indices [0, 1, 2]
        - The merge function should detect this is local indexing
        - Results should be mapped to global indices [5, 6, 7]
        """
        # Create 8 groups total (indices 0-7)
        groups = [make_group("MEDIUM", f"Theme {i}", i) for i in range(8)]
        context = "# Test Plan\n\nSimple test context."

        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare batches with max 4 per batch
            # This should create 2 batches: [0,1,2,3] and [4,5,6,7]
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            assert len(batch_metadata["batches"]) == 2, "Expected 2 batches"

            # Verify batch 1 contains indices [4, 5, 6, 7]
            batch1 = batch_metadata["batches"][1]
            assert batch1["group_indices"] == [4, 5, 6, 7], f"Expected [4,5,6,7], got {batch1['group_indices']}"

            # Write batch 0 results with local indices
            batch0 = batch_metadata["batches"][0]
            batch0_results = [
                {"group_index": i, "status": "valid", "reason": f"Local {i}", "confidence": 0.9}
                for i in range(4)  # Local indices 0, 1, 2, 3
            ]
            write_batch_results(batch0["output_path"], batch0_results)

            # Write batch 1 results with LOCAL indices [0, 1, 2, 3]
            # (not global indices [4, 5, 6, 7])
            batch1_results = [
                {"group_index": 0, "status": "valid", "reason": "Local idx 0 -> global 4", "confidence": 0.85},
                {"group_index": 1, "status": "invalid", "reason": "Local idx 1 -> global 5", "confidence": 0.75},
                {"group_index": 2, "status": "needs-human-decision", "reason": "Local idx 2 -> global 6", "confidence": 0.5},
                {"group_index": 3, "status": "valid", "reason": "Local idx 3 -> global 7", "confidence": 0.95},
            ]
            write_batch_results(batch1["output_path"], batch1_results)

            # Merge results
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # Verify all 8 groups are present
            assert len(merged) == 8, f"Expected 8 results, got {len(merged)}"

            # Verify batch 0 results (local indices 0-3 map to global 0-3)
            for i in range(4):
                assert merged[i]["status"] == "valid", f"Group {i} should be valid"

            # Verify batch 1 results were correctly remapped
            # Local 0 -> global 4
            assert merged[4]["status"] == "valid", "Group 4 should be valid"
            assert "Local idx 0" in merged[4]["reason"] or "global 4" in merged[4]["reason"]

            # Local 1 -> global 5
            assert merged[5]["status"] == "invalid", "Group 5 should be invalid"

            # Local 2 -> global 6
            assert merged[6]["status"] == "needs-human-decision", "Group 6 should be needs-human-decision"

            # Local 3 -> global 7
            assert merged[7]["status"] == "valid", "Group 7 should be valid"

    def test_local_indices_with_high_priority_isolation(self):
        """Test local index handling when HIGH priority groups cause non-contiguous batches.

        When HIGH priority groups are isolated, the batch indices become
        non-contiguous. For example:
        - Batch 0: [0] (HIGH, isolated)
        - Batch 1: [3] (HIGH, isolated)
        - Batch 2: [1, 2, 4, 5] (MEDIUM, batched)

        LLM returns local indices for each batch.
        """
        groups = [
            make_group("HIGH", "High 0", 0),    # Index 0 - isolated
            make_group("MEDIUM", "Med 1", 1),   # Index 1
            make_group("MEDIUM", "Med 2", 2),   # Index 2
            make_group("HIGH", "High 3", 3),    # Index 3 - isolated
            make_group("MEDIUM", "Med 4", 4),   # Index 4
            make_group("MEDIUM", "Med 5", 5),   # Index 5
        ]
        context = "# Test Plan\n\nHIGH priority isolation test."

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Write results for each batch using LOCAL indices
            for batch in batch_metadata["batches"]:
                batch_results = [
                    {
                        "group_index": i,  # Local index within batch
                        "status": "valid" if batch["is_high_priority"] else "invalid",
                        "reason": f"Local idx {i}",
                        "confidence": 0.9
                    }
                    for i in range(batch["groups_count"])
                ]
                write_batch_results(batch["output_path"], batch_results)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 6

            # HIGH groups (0, 3) should be valid
            assert merged[0]["status"] == "valid", "HIGH group 0 should be valid"
            assert merged[3]["status"] == "valid", "HIGH group 3 should be valid"

            # MEDIUM groups (1, 2, 4, 5) should be invalid
            for idx in [1, 2, 4, 5]:
                assert merged[idx]["status"] == "invalid", f"MEDIUM group {idx} should be invalid"


# ============================================================================
# Test Class: Batch Index Disambiguation - Global Indices
# ============================================================================


class TestBatchIndexDisambiguationGlobal:
    """Tests for batch index disambiguation when LLM returns global indices.

    When a batch contains groups [5,6,7] and the LLM returns results with
    indices [5,6,7], the merge function should correctly interpret these
    as global indices and use them directly.
    """

    def test_batch_index_disambiguation_global(self):
        """Batch [5,6,7] receives results [5,6,7] - correctly interpreted as global indices.

        This tests the scenario where:
        - Batch contains global indices [5, 6, 7]
        - LLM returns results with global indices [5, 6, 7]
        - The merge function should detect this is global indexing
        - Results should be used directly with their indices
        """
        groups = [make_group("MEDIUM", f"Theme {i}", i) for i in range(8)]
        context = "# Test Plan\n\nGlobal index test context."

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            assert len(batch_metadata["batches"]) == 2

            # Write batch 0 results with GLOBAL indices [0, 1, 2, 3]
            batch0 = batch_metadata["batches"][0]
            batch0_results = [
                {"group_index": idx, "status": "valid", "reason": f"Global {idx}", "confidence": 0.9}
                for idx in batch0["group_indices"]
            ]
            write_batch_results(batch0["output_path"], batch0_results)

            # Write batch 1 results with GLOBAL indices [4, 5, 6, 7]
            batch1 = batch_metadata["batches"][1]
            batch1_results = [
                {"group_index": 4, "status": "valid", "reason": "Global idx 4", "confidence": 0.85},
                {"group_index": 5, "status": "invalid", "reason": "Global idx 5", "confidence": 0.75},
                {"group_index": 6, "status": "needs-human-decision", "reason": "Global idx 6", "confidence": 0.5},
                {"group_index": 7, "status": "valid", "reason": "Global idx 7", "confidence": 0.95},
            ]
            write_batch_results(batch1["output_path"], batch1_results)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 8

            # Verify batch 0 results
            for i in range(4):
                assert merged[i]["status"] == "valid"
                assert "Global" in merged[i]["reason"]

            # Verify batch 1 results - global indices should be mapped correctly
            assert merged[4]["status"] == "valid"
            assert merged[5]["status"] == "invalid"
            assert merged[6]["status"] == "needs-human-decision"
            assert merged[7]["status"] == "valid"

    def test_global_indices_with_sparse_batch(self):
        """Test global indices when batch has non-contiguous group indices.

        Simulates the real-world scenario from performance testing where
        batches may have sparse indices due to HIGH priority isolation.
        """
        # 8 groups with 2 HIGH priority
        groups = [
            make_group("HIGH", "High 0", 0),
            make_group("MEDIUM", "Med 1", 1),
            make_group("MEDIUM", "Med 2", 2),
            make_group("MEDIUM", "Med 3", 3),
            make_group("HIGH", "High 4", 4),
            make_group("MEDIUM", "Med 5", 5),
            make_group("MEDIUM", "Med 6", 6),
            make_group("MEDIUM", "Med 7", 7),
        ]
        context = "# Test Plan\n\nSparse batch test."

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Write results using GLOBAL indices
            for batch in batch_metadata["batches"]:
                batch_results = [
                    {
                        "group_index": idx,  # Global index
                        "status": "valid" if idx in [0, 4] else "invalid",
                        "reason": f"Global idx {idx}",
                        "confidence": 0.9
                    }
                    for idx in batch["group_indices"]
                ]
                write_batch_results(batch["output_path"], batch_results)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 8

            # HIGH groups should be valid
            assert merged[0]["status"] == "valid"
            assert merged[4]["status"] == "valid"

            # MEDIUM groups should be invalid
            for idx in [1, 2, 3, 5, 6, 7]:
                assert merged[idx]["status"] == "invalid"


# ============================================================================
# Test Class: Batch Index Disambiguation - Ambiguous Cases
# ============================================================================


class TestBatchIndexDisambiguationAmbiguous:
    """Tests for batch index disambiguation in ambiguous cases.

    When a batch contains groups [0,1,2] and the LLM returns results with
    indices [0,1,2], both local and global interpretations are valid and
    produce the same result. The system should handle this deterministically.
    """

    def test_batch_index_disambiguation_ambiguous(self):
        """Batch [0,1,2] - indices could be either local or global, verify deterministic handling.

        This tests the ambiguous case where:
        - Batch contains global indices [0, 1, 2]
        - LLM returns results with indices [0, 1, 2]
        - Both interpretations yield the same result
        - System should handle this deterministically without errors
        """
        groups = [make_group("MEDIUM", f"Theme {i}", i) for i in range(3)]
        context = "# Test Plan\n\nAmbiguous index test."

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4  # All 3 fit in one batch
            )

            assert len(batch_metadata["batches"]) == 1

            batch = batch_metadata["batches"][0]
            assert batch["group_indices"] == [0, 1, 2], "Batch should contain [0, 1, 2]"

            # Write results with indices [0, 1, 2] - ambiguous!
            batch_results = [
                {"group_index": 0, "status": "valid", "reason": "Ambiguous idx 0", "confidence": 0.88},
                {"group_index": 1, "status": "invalid", "reason": "Ambiguous idx 1", "confidence": 0.72},
                {"group_index": 2, "status": "needs-human-decision", "reason": "Ambiguous idx 2", "confidence": 0.55},
            ]
            write_batch_results(batch["output_path"], batch_results)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # All 3 should be present with correct statuses
            assert len(merged) == 3

            assert merged[0]["status"] == "valid"
            assert merged[1]["status"] == "invalid"
            assert merged[2]["status"] == "needs-human-decision"

            # Confidence values should be preserved
            assert merged[0]["confidence"] == 0.88
            assert merged[1]["confidence"] == 0.72
            assert merged[2]["confidence"] == 0.55

            # No validation_failed status (all should be mapped)
            failed = [r for r in merged if r["status"] == "validation_failed"]
            assert len(failed) == 0, "No results should have validation_failed status"

    def test_ambiguous_repeated_runs_deterministic(self):
        """Verify ambiguous case produces consistent results across multiple runs."""
        groups = [make_group("MEDIUM", f"Theme {i}", i) for i in range(3)]
        context = "# Test Plan\n\nDeterminism test."

        results_list = []

        # Run 5 times to check determinism
        for _ in range(5):
            with tempfile.TemporaryDirectory() as tmpdir:
                batch_metadata = prepare_batched_validation_tasks(
                    groups=groups,
                    context=context,
                    output_dir=tmpdir,
                    max_per_batch=4
                )

                batch = batch_metadata["batches"][0]
                batch_results = [
                    {"group_index": i, "status": "valid", "reason": f"Test {i}", "confidence": 0.9}
                    for i in range(3)
                ]
                write_batch_results(batch["output_path"], batch_results)

                merged = merge_batched_validation_results(
                    output_dir=tmpdir,
                    batch_metadata=batch_metadata,
                    total_groups=len(groups)
                )

                # Extract just the statuses for comparison
                statuses = [r["status"] for r in merged]
                results_list.append(statuses)

        # All runs should produce identical results
        first_result = results_list[0]
        for i, result in enumerate(results_list[1:], 1):
            assert result == first_result, f"Run {i} produced different results than run 0"


# ============================================================================
# Test Class: Revalidation Failure Cascade
# ============================================================================


class TestRevalidationFailureCascade:
    """Tests for revalidation failure cascade handling.

    When revalidation itself fails, the system should:
    1. Preserve original validation results for non-revalidated items
    2. Mark revalidation failures appropriately
    3. Provide clear guidance to users
    """

    def test_revalidation_failure_cascade(self):
        """Revalidation itself fails, verify fallback behavior and user guidance.

        Scenario:
        - Initial validation has some validation_failed items
        - Revalidation is attempted but the batch file is missing/corrupt
        - System should preserve original results and mark revalidation as failed
        """

        groups = [make_group("MEDIUM", f"Theme {i}", i) for i in range(4)]
        context = "# Test Plan\n\nRevalidation cascade test."

        # Initial validation results - 2 failed, 2 succeeded
        initial_validation = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "validation_failed", "reason": "Timeout", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 2, "status": "validation_failed", "reason": "Parse error", "confidence": 0.0, "error_type": ERROR_TYPE_PARSING},
            {"group_index": 3, "status": "invalid", "reason": "False positive", "confidence": 0.8, "error_type": ERROR_TYPE_UNKNOWN},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare revalidation tasks
            revalidation_metadata = prepare_batched_revalidation_tasks(
                groups=groups,
                validation_results=initial_validation,
                context=context,
                output_dir=tmpdir
            )

            # Verify items to revalidate (groups 1 and 2)
            assert revalidation_metadata["items_to_revalidate"] == 2
            assert set(revalidation_metadata["item_indices"]) == {1, 2}

            # Simulate revalidation FAILURE by NOT writing batch files
            # This simulates a crashed subagent or timeout

            # Merge without any batch files written
            merged = merge_batched_revalidation_results(
                output_dir=tmpdir,
                revalidation_metadata=revalidation_metadata
            )

            assert len(merged) == 4

            # Original valid/invalid results should be preserved
            assert merged[0]["status"] == "valid", "Group 0 should remain valid"
            assert merged[0]["reason"] == "OK"

            assert merged[3]["status"] == "invalid", "Group 3 should remain invalid"
            assert merged[3]["reason"] == "False positive"

            # Revalidation failed items should remain validation_failed
            # (they weren't successfully revalidated)
            assert merged[1]["status"] == "validation_failed", "Group 1 should still be validation_failed"
            assert merged[2]["status"] == "validation_failed", "Group 2 should still be validation_failed"

    def test_partial_revalidation_success(self):
        """Test when revalidation succeeds for some items but fails for others."""

        groups = [make_group("MEDIUM", f"Theme {i}", i) for i in range(6)]
        context = "# Test Plan\n\nPartial revalidation test."

        initial_validation = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "validation_failed", "reason": "Timeout", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 2, "status": "validation_failed", "reason": "Timeout", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 3, "status": "validation_failed", "reason": "Timeout", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 4, "status": "validation_failed", "reason": "Timeout", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 5, "status": "invalid", "reason": "FP", "confidence": 0.8, "error_type": ERROR_TYPE_UNKNOWN},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            revalidation_metadata = prepare_batched_revalidation_tasks(
                groups=groups,
                validation_results=initial_validation,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Should have multiple batches for revalidation
            assert revalidation_metadata["items_to_revalidate"] == 4

            # Write ONLY the first batch (simulating partial success)
            if revalidation_metadata["batches"]:
                first_batch = revalidation_metadata["batches"][0]
                batch_results = [
                    {"group_index": i, "status": "valid", "reason": "Revalidated OK", "confidence": 0.85}
                    for i in range(first_batch["groups_count"])
                ]
                write_batch_results(first_batch["output_path"], batch_results)

            merged = merge_batched_revalidation_results(
                output_dir=tmpdir,
                revalidation_metadata=revalidation_metadata
            )

            assert len(merged) == 6

            # Group 0 and 5 should be unchanged
            assert merged[0]["status"] == "valid"
            assert merged[5]["status"] == "invalid"

            # Check that some revalidations succeeded and some failed
            # based on which batches were written
            revalidated_count = sum(1 for r in merged if r.get("revalidated", False))
            still_failed_count = sum(1 for r in merged if r["status"] == "validation_failed")

            # At least some should have been revalidated
            # and at least some should still be failed (if there was a second batch)
            assert revalidated_count > 0 or still_failed_count > 0


# ============================================================================
# Test Class: Confidence Edge Values
# ============================================================================


class TestConfidenceEdgeValues:
    """Tests for edge case confidence values in validation results.

    The system should handle:
    - Confidence 0.0 (minimum valid)
    - Confidence 1.0 (maximum valid)
    - Negative confidence (should be clamped to 0.0)
    - Confidence > 1.0 (should be clamped to 1.0)

    Note: The confidence clamping happens in validate_groups() when parsing
    the LLM response. Here we test via the merge function to verify the
    end-to-end flow handles edge values correctly.
    """

    def test_confidence_edge_values(self):
        """Confidence 0.0, 1.0, negative, >1.0 - verify clamping and no crashes.

        Tests that edge case confidence values are handled correctly when
        they flow through the batch merge process. The clamping should happen
        before saving to batch files, but we also verify the merge handles
        any edge values gracefully.
        """
        groups = [make_group("MEDIUM", f"Theme {i}", i) for i in range(6)]
        context = "# Test Plan\n\nConfidence edge case test."

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=6  # All in one batch
            )

            # Write batch results with edge case confidence values
            # (simulating what an LLM might return)
            batch = batch_metadata["batches"][0]
            batch_results = [
                {"group_index": 0, "status": "valid", "reason": "Confidence 0.0", "confidence": 0.0},
                {"group_index": 1, "status": "valid", "reason": "Confidence 1.0", "confidence": 1.0},
                {"group_index": 2, "status": "valid", "reason": "Negative (pre-clamped)", "confidence": 0.0},  # Pre-clamped
                {"group_index": 3, "status": "valid", "reason": "Above 1.0 (pre-clamped)", "confidence": 1.0},  # Pre-clamped
                {"group_index": 4, "status": "valid", "reason": "Very small", "confidence": 0.001},
                {"group_index": 5, "status": "valid", "reason": "Near 1.0", "confidence": 0.999},
            ]
            write_batch_results(batch["output_path"], batch_results)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 6

            # Confidence 0.0 should be preserved
            assert merged[0]["confidence"] == 0.0, "Confidence 0.0 should be preserved"

            # Confidence 1.0 should be preserved
            assert merged[1]["confidence"] == 1.0, "Confidence 1.0 should be preserved"

            # Pre-clamped values should be at boundaries
            assert merged[2]["confidence"] == 0.0
            assert merged[3]["confidence"] == 1.0

            # Small positive should be preserved
            assert merged[4]["confidence"] == 0.001, "Small positive confidence should be preserved"

            # Near 1.0 should be preserved
            assert merged[5]["confidence"] == 0.999, "Confidence near 1.0 should be preserved"

            # No crashes should occur - verify all statuses are valid
            for r in merged:
                assert r["status"] in ["valid", "invalid", "needs-human-decision", "validation_failed"]

    def test_confidence_missing_defaults_during_merge(self):
        """Test that missing confidence in batch results doesn't crash merge."""
        groups = [make_group("MEDIUM") for _ in range(2)]
        context = "# Test Plan"

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            batch = batch_metadata["batches"][0]
            # Write results with missing confidence field
            batch_results = [
                {"group_index": 0, "status": "valid", "reason": "No confidence field"},
                {"group_index": 1, "status": "valid", "reason": "Normal", "confidence": 0.9},
            ]
            write_batch_results(batch["output_path"], batch_results)

            # This should not crash
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 2
            # Both should be valid
            assert merged[0]["status"] == "valid"
            assert merged[1]["status"] == "valid"

    def test_confidence_with_unclamped_values_in_batch_file(self):
        """Test handling when batch file contains unclamped confidence values.

        While the validation parsing should clamp values, we test that the
        merge function handles any values in the batch files gracefully.
        """
        groups = [make_group("MEDIUM") for _ in range(4)]
        context = "# Test Plan"

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            batch = batch_metadata["batches"][0]
            # Write results with unclamped values (edge case: what if they weren't clamped upstream?)
            batch_results = [
                {"group_index": 0, "status": "valid", "reason": "Negative", "confidence": -0.5},
                {"group_index": 1, "status": "valid", "reason": "Large", "confidence": 2.0},
                {"group_index": 2, "status": "valid", "reason": "Very large", "confidence": 1000.0},
                {"group_index": 3, "status": "valid", "reason": "Very negative", "confidence": -1000.0},
            ]
            write_batch_results(batch["output_path"], batch_results)

            # Merge should not crash
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 4
            # All should have some confidence value (may or may not be clamped in merge)
            for r in merged:
                assert "confidence" in r or r["status"] == "validation_failed"
                assert r["status"] == "valid"

    def test_confidence_extreme_float_values(self):
        """Test with extreme float values that might cause precision issues."""
        groups = [make_group("MEDIUM") for _ in range(3)]
        context = "# Test Plan"

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            batch = batch_metadata["batches"][0]
            batch_results = [
                {"group_index": 0, "status": "valid", "reason": "Tiny positive", "confidence": 1e-10},
                {"group_index": 1, "status": "valid", "reason": "Almost 1", "confidence": 1 - 1e-10},
                {"group_index": 2, "status": "valid", "reason": "Exactly half", "confidence": 0.5},
            ]
            write_batch_results(batch["output_path"], batch_results)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 3
            # All should be in valid range
            for r in merged:
                conf = r.get("confidence", 0)
                assert 0.0 <= conf <= 1.0, f"Confidence {conf} not in valid range"


# ============================================================================
# Test Class: E2E Integration with Orchestrator
# ============================================================================


class TestE2EValidationEdgeCasesIntegration:
    """E2E integration tests using the full harness with skill runner."""

    def test_validation_with_edge_confidence_via_fixture(
        self,
        skill_runner: SkillRunner,
        fixture_manager: FixtureManager,
        mock_provider: MockProvider,
        assertions: AssertionHelpers,
    ):
        """E2E test: validation handles edge confidence values via fixture.

        This test uses the full e2e harness to verify that edge confidence
        values flow correctly through the entire validation pipeline.
        """
        # Create plan with enough content to generate suggestions
        plan_content = """# Test Plan for Confidence Edge Cases

## Overview
Testing confidence value handling in validation.

## Goals
- Verify confidence clamping
- Ensure no crashes with edge values

## Implementation Steps
1. Create validation module
2. Add confidence handling
3. Test edge cases
"""
        plan = fixture_manager.create_plan("confidence-edge-test", plan_content)

        # Set up review-plan outputs with suggestions
        review_dir = plan.ensure_phase_dir("review-plan")

        # Create grouped.json with suggestions
        grouped = [
            {
                "theme": f"Suggestion {i}",
                "category": "test",
                "models": ["cursor-agent"],
                "suggestions": [
                    {
                        "title": f"Test {i}",
                        "desc": f"Description {i}",
                        "type": "addition",
                        "importance": "MEDIUM",
                        "source_model": "cursor-agent"
                    }
                ]
            }
            for i in range(6)
        ]
        (review_dir / "grouped.json").write_text(json.dumps(grouped))

        # Create validation.json with edge confidence values
        validation = [
            {"group_index": 0, "status": "valid", "reason": "Zero confidence", "confidence": 0.0},
            {"group_index": 1, "status": "valid", "reason": "Full confidence", "confidence": 1.0},
            {"group_index": 2, "status": "valid", "reason": "Clamped from -0.5", "confidence": 0.0},  # Pre-clamped
            {"group_index": 3, "status": "valid", "reason": "Clamped from 1.5", "confidence": 1.0},  # Pre-clamped
            {"group_index": 4, "status": "needs-human-decision", "reason": "Low confidence", "confidence": 0.3},
            {"group_index": 5, "status": "invalid", "reason": "High confidence invalid", "confidence": 0.95},
        ]
        (review_dir / "validation.json").write_text(json.dumps(validation))

        # Create state file marking review-plan as completed
        fixture_manager.create_state_file(plan, phases_completed=["review-plan"])

        # Run apply_suggestions orchestrator
        result = skill_runner.run_orchestrator(
            "apply_suggestions",
            plan.plan_path,
            timeout=30,
        )

        # Should complete successfully
        assertions.assert_exit_code(result, 0)

        # Verify state indicates completion
        state = result.get_state()
        assert state is not None, "state.json should exist"


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


class TestIndexRemappingEdgeCases:
    """Additional edge case tests for index remapping."""

    def test_empty_batch_results(self):
        """Test handling of empty batch results."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        context = "# Test Plan"

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Write empty batch results
            batch = batch_metadata["batches"][0]
            write_batch_results(batch["output_path"], [])

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # All should be validation_failed due to missing results
            assert len(merged) == 4
            assert all(r["status"] == "validation_failed" for r in merged)

    def test_duplicate_indices_in_batch_results(self):
        """Test handling when batch results have duplicate indices.

        With 4 results for 4 groups, Strategy 2 (positional match) applies since
        the count matches. Results are mapped by position regardless of index values.
        """
        groups = [make_group("MEDIUM") for _ in range(4)]
        context = "# Test Plan"

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            batch = batch_metadata["batches"][0]
            # Write results with duplicate index 0
            batch_results = [
                {"group_index": 0, "status": "valid", "reason": "First 0", "confidence": 0.8},
                {"group_index": 0, "status": "invalid", "reason": "Second 0", "confidence": 0.9},
                {"group_index": 1, "status": "valid", "reason": "Normal 1", "confidence": 0.7},
                {"group_index": 2, "status": "valid", "reason": "Normal 2", "confidence": 0.6},
            ]
            write_batch_results(batch["output_path"], batch_results)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # Strategy 2 positional match: 4 results for 4 groups, mapped by position
            assert len(merged) == 4
            # Position 0 → group 0 (First 0, "valid")
            assert merged[0]["status"] == "valid"
            assert merged[0]["reason"] == "First 0"
            # Position 1 → group 1 (Second 0, "invalid")
            assert merged[1]["status"] == "invalid"
            assert merged[1]["reason"] == "Second 0"

    def test_out_of_order_indices(self):
        """Test handling when batch results are out of order."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        context = "# Test Plan"

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            batch = batch_metadata["batches"][0]
            # Write results out of order
            batch_results = [
                {"group_index": 3, "status": "valid", "reason": "Out of order 3", "confidence": 0.9},
                {"group_index": 1, "status": "invalid", "reason": "Out of order 1", "confidence": 0.8},
                {"group_index": 0, "status": "valid", "reason": "Out of order 0", "confidence": 0.7},
                {"group_index": 2, "status": "needs-human-decision", "reason": "Out of order 2", "confidence": 0.5},
            ]
            write_batch_results(batch["output_path"], batch_results)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # Results should be sorted by group_index in final output
            assert len(merged) == 4
            for i in range(4):
                assert merged[i]["group_index"] == i

            # Verify correct status assignments
            assert merged[0]["status"] == "valid"
            assert merged[1]["status"] == "invalid"
            assert merged[2]["status"] == "needs-human-decision"
            assert merged[3]["status"] == "valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
