#!/usr/bin/env python3
"""
Tests for batched validation functions in utils/validation.py

Tests the following functions:
- prepare_batched_validation_tasks()
- merge_batched_validation_results()
- prepare_batched_revalidation_tasks()
- merge_batched_revalidation_results()
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validation import (
    prepare_batched_validation_tasks,
    merge_batched_validation_results,
    prepare_batched_revalidation_tasks,
    merge_batched_revalidation_results,
    ERROR_TYPE_AMBIGUOUS,
    ERROR_TYPE_PARSING,
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_UNKNOWN,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_groups() -> List[Dict[str, Any]]:
    """Sample suggestion groups for testing."""
    return [
        {
            "theme": "Error Handling",
            "category": "reliability",
            "models": ["model-a", "model-b"],
            "suggestions": [
                {
                    "title": "Add error handling",
                    "desc": "Add try-catch blocks for network calls",
                    "importance": "HIGH",
                    "type": "addition"
                }
            ]
        },
        {
            "theme": "Performance",
            "category": "optimization",
            "models": ["model-a"],
            "suggestions": [
                {
                    "title": "Add caching",
                    "desc": "Cache database queries",
                    "importance": "MEDIUM",
                    "type": "addition"
                },
                {
                    "title": "Optimize loops",
                    "desc": "Use more efficient loop patterns",
                    "importance": "LOW",
                    "type": "modification"
                }
            ]
        },
        {
            "theme": "Security",
            "category": "security",
            "models": ["model-b"],
            "suggestions": [
                {
                    "title": "Input validation",
                    "desc": "Validate all user inputs",
                    "importance": "HIGH",
                    "type": "addition"
                }
            ]
        },
        {
            "theme": "Documentation",
            "category": "docs",
            "models": ["model-a"],
            "suggestions": [
                {
                    "title": "Add docstrings",
                    "desc": "Document public functions",
                    "importance": "LOW",
                    "type": "addition"
                }
            ]
        }
    ]


@pytest.fixture
def sample_context() -> str:
    """Sample plan context for validation."""
    return """# Implementation Plan

## Overview
Build a REST API with user authentication.

## Tasks
1. Set up database
2. Create user model
3. Implement authentication
4. Add API endpoints

## Technical Details
- Use PostgreSQL for database
- JWT for authentication tokens
"""


@pytest.fixture
def sample_validation_results() -> List[Dict[str, Any]]:
    """Sample validation results with various statuses."""
    return [
        {
            "group_index": 0,
            "status": "valid",
            "reason": "Issue is legitimate",
            "confidence": 0.9,
            "error_type": ERROR_TYPE_UNKNOWN,
            "recoverable": False,
        },
        {
            "group_index": 1,
            "status": "validation_failed",
            "reason": "Timeout during validation",
            "confidence": 0.0,
            "error_type": ERROR_TYPE_TIMEOUT,
            "recoverable": True,
        },
        {
            "group_index": 2,
            "status": "needs-human-decision",
            "reason": "Requires human judgment",
            "confidence": 0.5,
            "error_type": ERROR_TYPE_AMBIGUOUS,
            "recoverable": False,
        },
        {
            "group_index": 3,
            "status": "invalid",
            "reason": "False positive",
            "confidence": 0.8,
            "error_type": ERROR_TYPE_UNKNOWN,
            "recoverable": False,
        },
    ]


def make_group(importance: str = "MEDIUM", theme: str = "Test theme") -> dict:
    """Helper to create a test group."""
    return {
        "theme": theme,
        "category": "test",
        "models": ["test-model"],
        "suggestions": [
            {
                "title": "Test suggestion",
                "desc": "Test description",
                "importance": importance,
                "type": "modification"
            }
        ]
    }


# ============================================================================
# Tests for prepare_batched_validation_tasks()
# ============================================================================

class TestPrepareBatchedValidationTasks:
    """Tests for the prepare_batched_validation_tasks function."""

    def test_returns_dict(self, sample_groups, sample_context):
        """Function returns a dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir
            )
            assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_groups, sample_context):
        """Result contains all required keys including reference-based fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir,
                plan_file="/path/to/plan.md"
            )
            # Reference-based format: includes grouped_file, plan_file, phase_dir, reaggregate_command
            required_keys = [
                "batches", "total_batches", "batching_stats", "model_hint",
                "grouped_file", "plan_file", "phase_dir", "reaggregate_command"
            ]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

    def test_single_batch_for_small_groups(self, sample_context):
        """Single batch when groups <= max_per_batch."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )
            # 4 groups with max 4 per batch = 1 batch
            assert result["total_batches"] == 1
            assert len(result["batches"]) == 1
            assert result["batches"][0]["groups_count"] == 4

    def test_multiple_batches_for_large_groups(self, sample_context):
        """Multiple batches when groups > max_per_batch."""
        groups = [make_group("MEDIUM") for _ in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )
            # 10 groups with max 4 per batch = 3 batches (4+4+2)
            assert result["total_batches"] == 3
            assert len(result["batches"]) == 3
            assert result["batches"][0]["groups_count"] == 4
            assert result["batches"][1]["groups_count"] == 4
            assert result["batches"][2]["groups_count"] == 2

    def test_high_priority_isolation(self, sample_context):
        """HIGH priority groups are isolated in separate batches."""
        groups = [
            make_group("HIGH", "High 1"),
            make_group("HIGH", "High 2"),
            make_group("MEDIUM", "Med 1"),
            make_group("MEDIUM", "Med 2"),
            make_group("MEDIUM", "Med 3"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )
            # 2 HIGH (isolated, 1 each) + 3 MEDIUM (batched together) = 3 batches
            high_batches = [b for b in result["batches"] if b["is_high_priority"]]
            normal_batches = [b for b in result["batches"] if not b["is_high_priority"]]

            assert len(high_batches) == 2
            assert len(normal_batches) == 1
            for batch in high_batches:
                assert batch["groups_count"] == 1

    def test_empty_groups(self, sample_context):
        """Empty groups return empty batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=[],
                context=sample_context,
                output_dir=tmpdir
            )
            assert result["batches"] == []
            assert result["total_batches"] == 0

    def test_batch_structure(self, sample_groups, sample_context):
        """Each batch has correct structure (reference-based, no embedded prompts)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir
            )
            for batch in result["batches"]:
                # Reference-based format: minimal fields per batch
                assert "batch_index" in batch
                assert "group_indices" in batch
                assert "groups_count" in batch
                assert "is_high_priority" in batch
                assert "output_path" in batch
                # No longer includes prompt or suggestions_json (reference-based)
                assert "prompt" not in batch, "Prompt should not be embedded in reference-based format"
                assert "suggestions_json" not in batch, "suggestions_json should not be embedded"

    def test_batch_output_paths(self, sample_context):
        """Batch output paths are correctly formatted."""
        groups = [make_group("MEDIUM") for _ in range(6)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )
            for i, batch in enumerate(result["batches"]):
                expected_path = str(Path(tmpdir) / f"validation_batch_{i}.json")
                assert batch["output_path"] == expected_path

    def test_batch_indices_sequential(self, sample_context):
        """Batch indices are sequential starting from 0."""
        groups = [make_group("MEDIUM") for _ in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=3
            )
            indices = [b["batch_index"] for b in result["batches"]]
            assert indices == list(range(len(result["batches"])))

    def test_group_indices_cover_all(self, sample_groups, sample_context):
        """All original group indices are present across batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir
            )
            all_indices = []
            for batch in result["batches"]:
                all_indices.extend(batch["group_indices"])
            assert sorted(all_indices) == list(range(len(sample_groups)))

    def test_batching_stats_accurate(self, sample_context):
        """Batching stats are accurate."""
        # 8 MEDIUM groups with max 4 per batch = 2 batches
        groups = [make_group("MEDIUM") for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )
            stats = result["batching_stats"]
            assert stats["total_groups"] == 8
            assert stats["high_count"] == 0
            assert stats["normal_count"] == 8
            assert stats["estimated_batches"] == 2
            assert stats["subagent_calls_saved"] == 6  # 8 - 2
            assert stats["efficiency_gain_percent"] == 75.0

    def test_batching_stats_with_high_priority(self, sample_context):
        """Batching stats account for HIGH priority items."""
        groups = [
            make_group("HIGH"),
            make_group("HIGH"),
            make_group("MEDIUM"),
            make_group("MEDIUM"),
            make_group("MEDIUM"),
            make_group("MEDIUM"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )
            stats = result["batching_stats"]
            assert stats["high_count"] == 2
            assert stats["normal_count"] == 4

    def test_reference_fields_present(self, sample_groups, sample_context):
        """Reference-based format includes grouped_file and plan_file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir,
                plan_file="/path/to/test-plan.md"
            )
            # Check reference fields are present
            assert result["grouped_file"] == str(Path(tmpdir) / "grouped.json")
            assert result["plan_file"] == "/path/to/test-plan.md"
            assert result["phase_dir"] == tmpdir

    def test_reaggregate_command_generated(self, sample_groups, sample_context):
        """Reaggregate command is generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir,
                plan_file="/path/to/plan.md",
                orchestrator="review_plan_orchestrator.py"
            )
            cmd = result["reaggregate_command"]
            assert "review_plan_orchestrator.py" in cmd
            assert "--plan-file" in cmd
            assert "/path/to/plan.md" in cmd
            assert "--reaggregate" in cmd

    def test_custom_orchestrator_in_reaggregate_command(self, sample_groups, sample_context):
        """Custom orchestrator name is used in reaggregate command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir,
                plan_file="/path/to/plan.md",
                orchestrator="code_review_orchestrator.py"
            )
            cmd = result["reaggregate_command"]
            assert "code_review_orchestrator.py" in cmd
            assert "review_plan_orchestrator.py" not in cmd

    def test_model_hint_preserved(self, sample_groups, sample_context):
        """Model hint is preserved in result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir,
                model="custom-model"
            )
            assert result["model_hint"] == "custom-model"

    def test_base_ref_passed_through(self, sample_groups, sample_context):
        """Passing base_ref='abc123' includes it in the returned dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir,
                base_ref="abc123"
            )
            assert result["base_ref"] == "abc123"

    def test_base_ref_defaults_to_empty_string(self, sample_groups, sample_context):
        """Omitting base_ref defaults to an empty string in the returned dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=sample_groups,
                context=sample_context,
                output_dir=tmpdir
            )
            assert result["base_ref"] == ""

    def test_base_ref_in_empty_groups_early_return(self, sample_context):
        """base_ref is present in the empty-groups early return dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=[],
                context=sample_context,
                output_dir=tmpdir,
                base_ref="deadbeef"
            )
            assert result["base_ref"] == "deadbeef"
            assert result["batches"] == []
            assert result["total_batches"] == 0

    def test_validation_tasks_file_size_small(self, sample_context):
        """Validation tasks JSON stays small regardless of batch count."""
        # Create many groups that would generate a large file if prompts were embedded
        groups = [make_group("MEDIUM", f"Theme {i}") for i in range(50)]
        long_context = "x" * 10000  # Long context that would bloat file if embedded

        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=long_context,
                output_dir=tmpdir,
                plan_file="/path/to/plan.md",
                max_per_batch=4
            )

            # Serialize to JSON and check size
            json_str = json.dumps(result, indent=2)

            # With reference-based format, file should be small (< 10KB)
            # because prompts and suggestions_json are not embedded
            assert len(json_str) < 10000, f"File too large: {len(json_str)} bytes"

            # Verify no prompt or suggestions_json in batches
            for batch in result["batches"]:
                assert "prompt" not in batch
                assert "suggestions_json" not in batch


# ============================================================================
# Tests for merge_batched_validation_results()
# ============================================================================

class TestMergeBatchedValidationResults:
    """Tests for the merge_batched_validation_results function."""

    def test_all_batches_succeed(self, sample_context):
        """All batches succeed and results are merged correctly."""
        groups = [make_group("MEDIUM") for _ in range(6)]
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare batches
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Write batch result files
            for batch in batch_metadata["batches"]:
                batch_results = {
                    "groups": [
                        {
                            "group_index": i,
                            "status": "valid",
                            "reason": "OK",
                            "confidence": 0.9
                        }
                        for i in range(batch["groups_count"])
                    ],
                    "metadata": {"model": "test", "schema_version": "2.0"}
                }
                with open(batch["output_path"], 'w', encoding="utf-8") as f:
                    json.dump(batch_results, f)

            # Merge results
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 6
            assert all(r["status"] == "valid" for r in merged)

    def test_partial_failure(self, sample_context):
        """Handles partial failure when some batches are missing."""
        groups = [make_group("MEDIUM") for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Only write first batch (skip second)
            first_batch = batch_metadata["batches"][0]
            batch_results = {
                "groups": [
                    {"group_index": i, "status": "valid", "reason": "OK", "confidence": 0.9}
                    for i in range(first_batch["groups_count"])
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(first_batch["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch_results, f)

            # Merge results
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 8
            # First 4 should be valid
            for i in range(4):
                assert merged[i]["status"] == "valid"
            # Last 4 should be validation_failed (missing batch)
            for i in range(4, 8):
                assert merged[i]["status"] == "validation_failed"
                assert "not found" in merged[i]["reason"].lower()

    def test_missing_batch_file_handling(self, sample_context):
        """Handles missing batch files gracefully."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Don't write any batch files
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 4
            assert all(r["status"] == "validation_failed" for r in merged)

    def test_index_remapping(self, sample_context):
        """Batch-local indices are correctly remapped to original indices."""
        # Create groups with HIGH items that get isolated
        groups = [
            make_group("HIGH", "High 0"),
            make_group("MEDIUM", "Med 1"),
            make_group("MEDIUM", "Med 2"),
            make_group("HIGH", "High 3"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Write batch results
            for batch in batch_metadata["batches"]:
                batch_results = {
                    "groups": [
                        {
                            "group_index": i,  # Batch-local index
                            "status": "valid",
                            "reason": f"OK for batch {batch['batch_index']}",
                            "confidence": 0.9
                        }
                        for i in range(batch["groups_count"])
                    ],
                    "metadata": {"model": "test", "schema_version": "2.0"}
                }
                with open(batch["output_path"], 'w', encoding="utf-8") as f:
                    json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # All original indices should be present
            indices = [r["group_index"] for r in merged]
            assert sorted(indices) == [0, 1, 2, 3]

    def test_handles_direct_list_format(self, sample_context):
        """Handles batch files with direct list format (no wrapper)."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Write batch result as direct list (no "groups" wrapper)
            batch = batch_metadata["batches"][0]
            batch_results = [
                {"group_index": i, "status": "valid", "reason": "OK", "confidence": 0.9}
                for i in range(batch["groups_count"])
            ]
            with open(batch["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 4
            assert all(r["status"] == "valid" for r in merged)

    def test_handles_json_decode_error(self, sample_context):
        """Handles malformed JSON in batch files."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Write invalid JSON
            batch = batch_metadata["batches"][0]
            with open(batch["output_path"], 'w', encoding="utf-8") as f:
                f.write("{ invalid json }")

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 4
            # All should be validation_failed due to JSON error
            assert all(r["status"] == "validation_failed" for r in merged)

    def test_results_sorted_by_group_index(self, sample_context):
        """Merged results are sorted by group_index."""
        groups = [make_group("MEDIUM") for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Write batch files in reverse order
            for batch in reversed(batch_metadata["batches"]):
                batch_results = {
                    "groups": [
                        {"group_index": i, "status": "valid", "reason": "OK", "confidence": 0.9}
                        for i in range(batch["groups_count"])
                    ],
                    "metadata": {"model": "test", "schema_version": "2.0"}
                }
                with open(batch["output_path"], 'w', encoding="utf-8") as f:
                    json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            indices = [r["group_index"] for r in merged]
            assert indices == sorted(indices)

    def test_handles_global_indices_in_batch_output(self, sample_context):
        """Handles batch files that use global group indices instead of batch-local.

        When subagents validate groups from grouped.json, they may output the original
        global group indices (e.g., 2, 3 for batch with group_indices=[2, 3]) instead
        of batch-local indices (0, 1). The merge function should handle both formats.
        """
        # Create 6 groups - will create 2 batches
        groups = [make_group("MEDIUM") for _ in range(6)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Verify we have 2 batches
            assert len(batch_metadata["batches"]) == 2
            batch0 = batch_metadata["batches"][0]
            batch1 = batch_metadata["batches"][1]

            # Write batch 0 with GLOBAL indices (not batch-local)
            # batch0 has group_indices [0, 1, 2, 3] so we use those directly
            batch0_results = {
                "groups": [
                    {"group_index": idx, "status": "valid", "reason": "OK", "confidence": 0.9}
                    for idx in batch0["group_indices"]  # Global indices!
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch0["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch0_results, f)

            # Write batch 1 with GLOBAL indices
            # batch1 has group_indices [4, 5] so we use those directly
            batch1_results = {
                "groups": [
                    {"group_index": idx, "status": "invalid", "reason": "Not OK", "confidence": 0.8}
                    for idx in batch1["group_indices"]  # Global indices!
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch1["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch1_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # All 6 groups should be merged correctly
            assert len(merged) == 6
            # First 4 should be valid (from batch 0)
            for i in range(4):
                assert merged[i]["status"] == "valid", f"Group {i} should be valid"
            # Last 2 should be invalid (from batch 1)
            for i in range(4, 6):
                assert merged[i]["status"] == "invalid", f"Group {i} should be invalid"

    def test_handles_global_indices_with_high_priority_isolation(self, sample_context):
        """Handles global indices when HIGH priority groups are isolated into single batches.

        When HIGH priority groups are isolated, the batches have non-contiguous
        group_indices (e.g., [0], [3], [1, 2, 4, 5]). Subagents may use global indices.
        """
        groups = [
            make_group("HIGH", "High 0"),    # Index 0 - isolated
            make_group("MEDIUM", "Med 1"),   # Index 1
            make_group("MEDIUM", "Med 2"),   # Index 2
            make_group("HIGH", "High 3"),    # Index 3 - isolated
            make_group("MEDIUM", "Med 4"),   # Index 4
            make_group("MEDIUM", "Med 5"),   # Index 5
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # HIGH priority batches come first (index 0 and 3)
            # Then remaining MEDIUM groups
            for batch in batch_metadata["batches"]:
                # Write using GLOBAL indices
                batch_results = {
                    "groups": [
                        {
                            "group_index": idx,  # Global index!
                            "status": "valid" if idx in [0, 3] else "invalid",
                            "reason": f"Validated group {idx}",
                            "confidence": 0.9
                        }
                        for idx in batch["group_indices"]
                    ],
                    "metadata": {"model": "test", "schema_version": "2.0"}
                }
                with open(batch["output_path"], 'w', encoding="utf-8") as f:
                    json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # All 6 groups should be present
            assert len(merged) == 6

            # HIGH groups (0, 3) should be valid
            assert merged[0]["status"] == "valid"
            assert merged[3]["status"] == "valid"

            # MEDIUM groups (1, 2, 4, 5) should be invalid
            assert merged[1]["status"] == "invalid"
            assert merged[2]["status"] == "invalid"
            assert merged[4]["status"] == "invalid"
            assert merged[5]["status"] == "invalid"

    def test_handles_mixed_local_and_global_indices(self, sample_context):
        """Handles case where some batches use local indices and some use global.

        In practice, all batches in a run likely use the same convention, but
        the merge function should handle mixed formats gracefully.
        """
        groups = [make_group("MEDIUM") for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            assert len(batch_metadata["batches"]) == 2
            batch0 = batch_metadata["batches"][0]
            batch1 = batch_metadata["batches"][1]

            # Batch 0 uses LOCAL indices (0, 1, 2, 3)
            batch0_results = {
                "groups": [
                    {"group_index": i, "status": "valid", "reason": "Local idx", "confidence": 0.9}
                    for i in range(batch0["groups_count"])
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch0["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch0_results, f)

            # Batch 1 uses GLOBAL indices (4, 5, 6, 7)
            batch1_results = {
                "groups": [
                    {"group_index": idx, "status": "invalid", "reason": "Global idx", "confidence": 0.8}
                    for idx in batch1["group_indices"]
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch1["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch1_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 8
            # First 4 (local indices) should be valid
            for i in range(4):
                assert merged[i]["status"] == "valid"
            # Last 4 (global indices) should be invalid
            for i in range(4, 8):
                assert merged[i]["status"] == "invalid"

    def test_handles_out_of_range_indices(self, sample_context):
        """Handles indices that are neither valid local nor global.

        When count matches (4 results for 4 groups), Strategy 2 (positional match)
        maps them positionally — this is correct because the LLM likely output the
        right results but with wrong index numbers.
        """
        groups = [make_group("MEDIUM") for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            batch = batch_metadata["batches"][0]
            # Write with invalid indices (100, 101, 102, 103) - neither local nor global
            # but count matches, so Strategy 2 (positional) maps them
            batch_results = {
                "groups": [
                    {"group_index": 100 + i, "status": "valid", "reason": "OK", "confidence": 0.9}
                    for i in range(batch["groups_count"])
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # Strategy 2 maps positionally when count matches
            assert len(merged) == 4
            assert all(r["status"] == "valid" for r in merged)

    def test_handles_partial_global_indices(self, sample_context):
        """Handles case where batch output has fewer results than expected."""
        groups = [make_group("MEDIUM") for _ in range(6)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            batch0 = batch_metadata["batches"][0]
            # Only output 2 of 4 expected results, using global indices
            batch0_results = {
                "groups": [
                    {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9},
                    {"group_index": 2, "status": "valid", "reason": "OK", "confidence": 0.9},
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch0["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch0_results, f)

            # Don't write batch1 at all
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert len(merged) == 6
            # Groups 0 and 2 should be valid
            assert merged[0]["status"] == "valid"
            assert merged[2]["status"] == "valid"
            # Groups 1, 3, 4, 5 should be validation_failed
            assert merged[1]["status"] == "validation_failed"
            assert merged[3]["status"] == "validation_failed"
            assert merged[4]["status"] == "validation_failed"
            assert merged[5]["status"] == "validation_failed"

    def test_merge_noncontiguous_with_extras(self):
        """Non-contiguous batch [2, 5] where LLM fills gaps returning [2, 3, 4, 5].

        Strategy 1 should extract only indices 2 and 5, ignoring extras.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [2, 5],
                        "group_ids": ["g2", "g5"],
                        "groups_count": 2,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            # LLM fills the gap, returning results for indices 2, 3, 4, 5
            batch_results = {
                "groups": [
                    {"group_index": 2, "status": "valid", "reason": "OK", "confidence": 0.9},
                    {"group_index": 3, "status": "valid", "reason": "Extra", "confidence": 0.8},
                    {"group_index": 4, "status": "valid", "reason": "Extra", "confidence": 0.8},
                    {"group_index": 5, "status": "invalid", "reason": "False positive", "confidence": 0.85},
                ],
                "metadata": {"model": "test", "schema_version": "2.0"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=6,
            )

            assert len(merged) == 6
            # Group 2 and 5 should have real results
            assert merged[2]["status"] == "valid"
            assert merged[5]["status"] == "invalid"
            assert merged[2].get("group_id") == "g2"
            assert merged[5].get("group_id") == "g5"
            # Other groups should remain validation_failed (untouched)
            assert merged[0]["status"] == "validation_failed"
            assert merged[1]["status"] == "validation_failed"
            assert merged[3]["status"] == "validation_failed"
            assert merged[4]["status"] == "validation_failed"

    def test_merge_positional_fallback(self):
        """Batch [6, 8] where LLM returns [0, 1] — Strategy 2 maps positionally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [6, 8],
                        "group_ids": ["g6", "g8"],
                        "groups_count": 2,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            # LLM uses local indices 0, 1 instead of global 6, 8
            batch_results = {
                "groups": [
                    {"group_index": 0, "status": "valid", "reason": "Issue confirmed", "confidence": 0.9},
                    {"group_index": 1, "status": "invalid", "reason": "Not an issue", "confidence": 0.7},
                ],
                "metadata": {"model": "test", "schema_version": "2.0"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=10,
            )

            assert len(merged) == 10
            # Index 6 should get result from position 0
            assert merged[6]["status"] == "valid"
            assert merged[6]["group_index"] == 6
            assert merged[6].get("group_id") == "g6"
            # Index 8 should get result from position 1
            assert merged[8]["status"] == "invalid"
            assert merged[8]["group_index"] == 8
            assert merged[8].get("group_id") == "g8"

    def test_merge_total_groups_smaller_than_batch_indices(self):
        """Regression: reaggregation produces fewer groups than original validation.

        When reaggregation re-groups issues and gets M < N groups, but batch
        metadata still references indices up to N-1, merged_results must be
        large enough to hold all referenced indices.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Batch metadata references group indices 0, 2, 4 (from original 5-group run)
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [0, 2, 4],
                        "group_ids": ["g0", "g2", "g4"],
                        "groups_count": 3,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            batch_results = {
                "groups": [
                    {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9},
                    {"group_index": 2, "status": "invalid", "reason": "FP", "confidence": 0.8},
                    {"group_index": 4, "status": "valid", "reason": "Real", "confidence": 0.7},
                ],
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            # Reaggregation produced only 4 groups, but batch references index 4
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=4,  # Less than max index (4) + 1
            )

            # Should expand to 5 entries (indices 0-4)
            assert len(merged) == 5
            assert merged[0]["status"] == "valid"
            assert merged[2]["status"] == "invalid"
            assert merged[4]["status"] == "valid"
            # Unmatched indices should have default failure status
            assert merged[1]["status"] == "validation_failed"
            assert merged[3]["status"] == "validation_failed"

    def test_hash_matching_corrects_off_by_one(self):
        """Batch [8, 9] where LLM returns correct indices but wrong content.

        LLM reports group_index=8 but group_hash matches group 9, and vice versa.
        Strategy 0 should use hash to place results at correct indices.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [8, 9],
                        "group_ids": ["hash_for_8", "hash_for_9"],
                        "groups_count": 2,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            # LLM returns results with swapped hashes (off-by-one content)
            # group_index=8 has hash_for_9's content, group_index=9 has hash_for_8's content
            batch_results = {
                "groups": [
                    {"group_index": 8, "group_hash": "hash_for_9", "status": "valid", "reason": "Content about group 9", "confidence": 0.9},
                    {"group_index": 9, "group_hash": "hash_for_8", "status": "invalid", "reason": "Content about group 8", "confidence": 0.85},
                ],
                "metadata": {"model": "test", "schema_version": "2.1"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=10,
            )

            assert len(merged) == 10
            # Hash-based matching should correct the swap:
            # hash_for_8 -> index 8, hash_for_9 -> index 9
            assert merged[8]["status"] == "invalid"
            assert merged[8]["reason"] == "Content about group 8"
            assert merged[8].get("group_id") == "hash_for_8"
            assert merged[9]["status"] == "valid"
            assert merged[9]["reason"] == "Content about group 9"
            assert merged[9].get("group_id") == "hash_for_9"

    def test_hash_matching_all_correct(self):
        """Batch [2, 5] where hashes and indices both agree — happy path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [2, 5],
                        "group_ids": ["hash_2", "hash_5"],
                        "groups_count": 2,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            batch_results = {
                "groups": [
                    {"group_index": 2, "group_hash": "hash_2", "status": "valid", "reason": "OK", "confidence": 0.9},
                    {"group_index": 5, "group_hash": "hash_5", "status": "invalid", "reason": "Not real", "confidence": 0.85},
                ],
                "metadata": {"model": "test", "schema_version": "2.1"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=6,
            )

            assert len(merged) == 6
            assert merged[2]["status"] == "valid"
            assert merged[2].get("group_id") == "hash_2"
            assert merged[5]["status"] == "invalid"
            assert merged[5].get("group_id") == "hash_5"

    def test_no_hash_falls_through_to_index_strategies(self):
        """Results without group_hash fall through to existing index strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [2, 5],
                        "group_ids": ["hash_2", "hash_5"],
                        "groups_count": 2,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            # No group_hash in results — should fall through to Strategy 1
            batch_results = {
                "groups": [
                    {"group_index": 2, "status": "valid", "reason": "OK", "confidence": 0.9},
                    {"group_index": 5, "status": "invalid", "reason": "Not real", "confidence": 0.85},
                ],
                "metadata": {"model": "test", "schema_version": "2.0"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=6,
            )

            assert len(merged) == 6
            assert merged[2]["status"] == "valid"
            assert merged[2].get("group_id") == "hash_2"
            assert merged[5]["status"] == "invalid"
            assert merged[5].get("group_id") == "hash_5"

    def test_partial_hash_falls_through(self):
        """Only some results have group_hash — Strategy 0 skips, index strategies handle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [2, 5],
                        "group_ids": ["hash_2", "hash_5"],
                        "groups_count": 2,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            # Only first result has group_hash
            batch_results = {
                "groups": [
                    {"group_index": 2, "group_hash": "hash_2", "status": "valid", "reason": "OK", "confidence": 0.9},
                    {"group_index": 5, "status": "invalid", "reason": "Not real", "confidence": 0.85},
                ],
                "metadata": {"model": "test", "schema_version": "2.1"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=6,
            )

            assert len(merged) == 6
            # Should fall through to Strategy 1 (global index match) and still work
            assert merged[2]["status"] == "valid"
            assert merged[5]["status"] == "invalid"

    def test_foreign_hash_is_not_misattributed_by_index(self):
        """Single-group batch where the subagent validated the WRONG group.

        Reproduces the validation-merge misattribution: batch 6 expects only G16
        (hash_for_16) but the subagent returns a result whose group_index is the
        correct slot (15) while its group_hash + reasoning belong to a different
        group (hash_for_15). Trusting the index would stamp G15's reasoning onto
        G16. The merge must discard the foreign-hash result and leave the target
        as validation_failed (so it gets revalidated), NOT silently misattribute.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [15],
                        "group_ids": ["hash_for_16"],
                        "groups_count": 1,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            # Correct-looking index (15) but the hash + reasoning are for a
            # different group (hash_for_15).
            batch_results = {
                "groups": [
                    {
                        "group_index": 15,
                        "group_hash": "hash_for_15",
                        "status": "needs-human-decision",
                        "reason": "Reasoning about group 15's topic",
                        "confidence": 0.8,
                    }
                ],
                "metadata": {"model": "test", "schema_version": "2.1"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=16,
            )

            assert len(merged) == 16
            # G16 (index 15) must NOT carry the foreign group's reasoning.
            assert merged[15]["status"] == "validation_failed"
            assert merged[15].get("group_hash") != "hash_for_15"
            assert "group 15" not in merged[15].get("reason", "")

    def test_foreign_hash_dropped_while_valid_hash_routes(self):
        """Multi-group batch: one result has the right hash, one is foreign.

        Batch expects G7 (hash_7) and G8 (hash_8). The subagent validates G7
        correctly but, instead of G8, re-validates G7's neighbor and stamps a
        foreign hash on a slot-looking index. The valid result must route by
        hash; the foreign one must be discarded; G8 stays validation_failed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [7, 8],
                        "group_ids": ["hash_7", "hash_8"],
                        "groups_count": 2,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            batch_results = {
                "groups": [
                    {"group_index": 7, "group_hash": "hash_7", "status": "valid", "reason": "Real G7 issue", "confidence": 0.9},
                    {"group_index": 8, "group_hash": "hash_elsewhere", "status": "invalid", "reason": "Wrong group reasoning", "confidence": 0.7},
                ],
                "metadata": {"model": "test", "schema_version": "2.1"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=9,
            )

            assert len(merged) == 9
            # G7 routed correctly by hash.
            assert merged[7]["status"] == "valid"
            assert merged[7].get("group_id") == "hash_7"
            # G8 was never actually validated — must not inherit foreign reasoning.
            assert merged[8]["status"] == "validation_failed"
            assert merged[8].get("reason") != "Wrong group reasoning"

    def test_merge_partial_salvage(self):
        """Batch [2, 5, 9] where LLM returns [2, 3] — Strategy 3 saves group 2.

        Count mismatch (2 results vs 3 expected) prevents Strategy 2.
        Only group 2 matches a global index, so Strategy 3 salvages it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = {
                "total_batches": 1,
                "batches": [
                    {
                        "batch_index": 0,
                        "group_indices": [2, 5, 9],
                        "group_ids": ["g2", "g5", "g9"],
                        "groups_count": 3,
                        "output_path": os.path.join(tmpdir, "validation_batch_0.json"),
                    }
                ],
            }

            # LLM returns only 2 results (count mismatch with 3 expected)
            # Only group 2 matches a global index; group 3 is extra
            batch_results = {
                "groups": [
                    {"group_index": 2, "status": "valid", "reason": "Real issue", "confidence": 0.9},
                    {"group_index": 3, "status": "valid", "reason": "Wrong group", "confidence": 0.8},
                ],
                "metadata": {"model": "test", "schema_version": "2.0"},
            }
            with open(batch_metadata["batches"][0]["output_path"], "w", encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=10,
            )

            assert len(merged) == 10
            # Group 2 should be salvaged
            assert merged[2]["status"] == "valid"
            assert merged[2].get("group_id") == "g2"
            # Groups 5 and 9 should remain validation_failed (not matched)
            assert merged[5]["status"] == "validation_failed"
            assert merged[9]["status"] == "validation_failed"


# ============================================================================
# Tests for prepare_batched_revalidation_tasks()
# ============================================================================

class TestPrepareBatchedRevalidationTasks:
    """Tests for the prepare_batched_revalidation_tasks function."""

    def test_items_to_revalidate(self, sample_groups, sample_validation_results, sample_context):
        """Correctly identifies items needing revalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_revalidation_tasks(
                groups=sample_groups,
                validation_results=sample_validation_results,
                context=sample_context,
                output_dir=tmpdir
            )
            # Only group 1 has validation_failed status
            assert result["items_to_revalidate"] == 1
            assert 1 in result["item_indices"]
            assert 0 not in result["item_indices"]  # valid
            assert 2 not in result["item_indices"]  # ambiguous needs-human-decision
            assert 3 not in result["item_indices"]  # invalid

    def test_no_items_case(self, sample_groups, sample_context):
        """Returns empty batches when no items need revalidation."""
        validation_results = [
            {"group_index": i, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN}
            for i in range(len(sample_groups))
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_revalidation_tasks(
                groups=sample_groups,
                validation_results=validation_results,
                context=sample_context,
                output_dir=tmpdir
            )
            assert result["batches"] == []
            assert result["total_batches"] == 0
            assert result["items_to_revalidate"] == 0
            assert result["item_indices"] == []

    def test_include_human_decision_flag(self, sample_groups, sample_context):
        """Includes needs-human-decision items when flag is set."""
        validation_results = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "needs-human-decision", "reason": "Parse error", "confidence": 0.0, "error_type": ERROR_TYPE_PARSING},
            {"group_index": 2, "status": "needs-human-decision", "reason": "Ambiguous", "confidence": 0.5, "error_type": ERROR_TYPE_AMBIGUOUS},
            {"group_index": 3, "status": "invalid", "reason": "False positive", "confidence": 0.8, "error_type": ERROR_TYPE_UNKNOWN},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_revalidation_tasks(
                groups=sample_groups,
                validation_results=validation_results,
                context=sample_context,
                output_dir=tmpdir,
                include_all_human=True
            )
            # Group 1 (parsing error) should be included, Group 2 (ambiguous) should NOT
            assert 1 in result["item_indices"]
            assert 2 not in result["item_indices"]

    def test_preserves_original_validation(self, sample_groups, sample_validation_results, sample_context):
        """Preserves original validation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_revalidation_tasks(
                groups=sample_groups,
                validation_results=sample_validation_results,
                context=sample_context,
                output_dir=tmpdir
            )
            assert result["original_validation"] == sample_validation_results

    def test_batching_stats_for_revalidation(self, sample_groups, sample_context):
        """Batching stats reflect only items to revalidate."""
        validation_results = [
            {"group_index": 0, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 1, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_PARSING},
            {"group_index": 2, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 3, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_revalidation_tasks(
                groups=sample_groups,
                validation_results=validation_results,
                context=sample_context,
                output_dir=tmpdir
            )
            # Only 2 items need revalidation
            assert result["items_to_revalidate"] == 2
            stats = result["batching_stats"]
            assert stats["total_groups"] == 2

    def test_model_hint_preserved(self, sample_groups, sample_validation_results, sample_context):
        """Model hint is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_revalidation_tasks(
                groups=sample_groups,
                validation_results=sample_validation_results,
                context=sample_context,
                output_dir=tmpdir,
                model="revalidation-model"
            )
            assert result["model_hint"] == "revalidation-model"

    def test_base_ref_forwarded(self, sample_groups, sample_validation_results, sample_context):
        """base_ref='abc123' is correctly forwarded to the inner prepare_batched_validation_tasks() call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_revalidation_tasks(
                groups=sample_groups,
                validation_results=sample_validation_results,
                context=sample_context,
                output_dir=tmpdir,
                base_ref="abc123"
            )
            assert result["base_ref"] == "abc123"

    def test_base_ref_in_empty_items_early_return(self, sample_groups, sample_context):
        """base_ref is present in the empty-items early return dict."""
        # All valid results means nothing to revalidate
        validation_results = [
            {"group_index": i, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN}
            for i in range(len(sample_groups))
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_revalidation_tasks(
                groups=sample_groups,
                validation_results=validation_results,
                context=sample_context,
                output_dir=tmpdir,
                base_ref="deadbeef"
            )
            assert result["base_ref"] == "deadbeef"
            assert result["batches"] == []
            assert result["total_batches"] == 0
            assert result["items_to_revalidate"] == 0


# ============================================================================
# Tests for merge_batched_revalidation_results()
# ============================================================================

class TestMergeBatchedRevalidationResults:
    """Tests for the merge_batched_revalidation_results function."""

    def test_all_batches_complete(self, sample_context):
        """All revalidation batches complete and merge correctly."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        validation_results = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 2, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_PARSING},
            {"group_index": 3, "status": "invalid", "reason": "False", "confidence": 0.8, "error_type": ERROR_TYPE_UNKNOWN},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            revalidation_metadata = prepare_batched_revalidation_tasks(
                groups=groups,
                validation_results=validation_results,
                context=sample_context,
                output_dir=tmpdir
            )

            # Write revalidation results
            for batch in revalidation_metadata["batches"]:
                batch_results = {
                    "groups": [
                        {"group_index": i, "status": "valid", "reason": "Now valid", "confidence": 0.85}
                        for i in range(batch["groups_count"])
                    ],
                    "metadata": {"model": "test", "schema_version": "2.0"}
                }
                with open(batch["output_path"], 'w', encoding="utf-8") as f:
                    json.dump(batch_results, f)

            merged = merge_batched_revalidation_results(
                output_dir=tmpdir,
                revalidation_metadata=revalidation_metadata
            )

            # Group 0 and 3 should be unchanged
            assert merged[0]["status"] == "valid"
            assert merged[3]["status"] == "invalid"
            # Groups 1 and 2 should be revalidated
            assert merged[1]["status"] == "valid"
            assert merged[1]["revalidated"] == True
            assert merged[2]["status"] == "valid"
            assert merged[2]["revalidated"] == True

    def test_partial_results(self, sample_context):
        """Handles partial revalidation results."""
        groups = [make_group("MEDIUM") for _ in range(6)]
        validation_results = [
            {"group_index": i, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT}
            for i in range(6)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            revalidation_metadata = prepare_batched_revalidation_tasks(
                groups=groups,
                validation_results=validation_results,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Only write first batch
            first_batch = revalidation_metadata["batches"][0]
            batch_results = {
                "groups": [
                    {"group_index": i, "status": "valid", "reason": "OK", "confidence": 0.9}
                    for i in range(first_batch["groups_count"])
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(first_batch["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_revalidation_results(
                output_dir=tmpdir,
                revalidation_metadata=revalidation_metadata
            )

            # First 4 should be valid
            for i in range(4):
                assert merged[i]["status"] == "valid"
            # Last 2 should still be validation_failed
            for i in range(4, 6):
                assert merged[i]["status"] == "validation_failed"

    def test_original_results_preserved_for_non_revalidated(self, sample_context):
        """Original results preserved for items not revalidated."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        validation_results = [
            {"group_index": 0, "status": "valid", "reason": "Good", "confidence": 0.95, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "invalid", "reason": "Bad", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 2, "status": "validation_failed", "reason": "Error", "confidence": 0.0, "error_type": ERROR_TYPE_TIMEOUT},
            {"group_index": 3, "status": "needs-human-decision", "reason": "Ambiguous", "confidence": 0.5, "error_type": ERROR_TYPE_AMBIGUOUS},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            revalidation_metadata = prepare_batched_revalidation_tasks(
                groups=groups,
                validation_results=validation_results,
                context=sample_context,
                output_dir=tmpdir
            )

            # Write revalidation results for group 2
            for batch in revalidation_metadata["batches"]:
                batch_results = {
                    "groups": [
                        {"group_index": 0, "status": "valid", "reason": "Now valid", "confidence": 0.85}
                    ],
                    "metadata": {"model": "test", "schema_version": "2.0"}
                }
                with open(batch["output_path"], 'w', encoding="utf-8") as f:
                    json.dump(batch_results, f)

            merged = merge_batched_revalidation_results(
                output_dir=tmpdir,
                revalidation_metadata=revalidation_metadata
            )

            # Original results preserved for non-revalidated items
            assert merged[0]["reason"] == "Good"
            assert merged[1]["reason"] == "Bad"
            assert merged[3]["reason"] == "Ambiguous"

    def test_empty_item_indices(self, sample_context):
        """Handles case with no items to revalidate."""
        groups = [make_group("MEDIUM") for _ in range(2)]
        validation_results = [
            {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9, "error_type": ERROR_TYPE_UNKNOWN},
            {"group_index": 1, "status": "invalid", "reason": "Bad", "confidence": 0.8, "error_type": ERROR_TYPE_UNKNOWN},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            revalidation_metadata = prepare_batched_revalidation_tasks(
                groups=groups,
                validation_results=validation_results,
                context=sample_context,
                output_dir=tmpdir
            )

            merged = merge_batched_revalidation_results(
                output_dir=tmpdir,
                revalidation_metadata=revalidation_metadata
            )

            # Should return original results unchanged
            assert merged == validation_results


# ============================================================================
# Tests for Batch File I/O
# ============================================================================

class TestBatchFileIO:
    """Tests for batch file I/O operations."""

    def test_write_and_read_batch_results(self, sample_context):
        """Write and read batch results correctly."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir
            )

            expected_results = [
                {"group_index": 0, "status": "valid", "reason": "Test 0", "confidence": 0.9},
                {"group_index": 1, "status": "invalid", "reason": "Test 1", "confidence": 0.8},
                {"group_index": 2, "status": "needs-human-decision", "reason": "Test 2", "confidence": 0.5},
                {"group_index": 3, "status": "valid", "reason": "Test 3", "confidence": 0.95},
            ]

            # Write batch file
            batch = batch_metadata["batches"][0]
            batch_data = {
                "groups": expected_results,
                "metadata": {"model": "test", "timestamp": "2025-01-01T00:00:00", "schema_version": "2.0"}
            }
            with open(batch["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch_data, f)

            # Read via merge
            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # Verify results match
            for i, expected in enumerate(expected_results):
                assert merged[i]["status"] == expected["status"]
                assert merged[i]["reason"] == expected["reason"]

    def test_batch_file_overwrite(self, sample_context):
        """Batch files can be overwritten."""
        groups = [make_group("MEDIUM") for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir
            )

            batch = batch_metadata["batches"][0]

            # Write initial results
            initial = {"groups": [{"group_index": 0, "status": "valid", "reason": "First", "confidence": 0.9}]}
            with open(batch["output_path"], 'w', encoding="utf-8") as f:
                json.dump(initial, f)

            # Overwrite
            updated = {"groups": [{"group_index": 0, "status": "invalid", "reason": "Second", "confidence": 0.1}]}
            with open(batch["output_path"], 'w', encoding="utf-8") as f:
                json.dump(updated, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            assert merged[0]["status"] == "invalid"
            assert merged[0]["reason"] == "Second"


# ============================================================================
# Tests for Marker Output
# ============================================================================

class TestMarkerOutput:
    """Tests for validation marker output (pending markers)."""

    def test_single_batch_structure(self, sample_context):
        """Single batch has proper structure for orchestrator (reference-based)."""
        groups = [make_group("MEDIUM") for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                plan_file="/path/to/plan.md",
                max_per_batch=4
            )

            # Single batch case
            assert result["total_batches"] == 1
            # Orchestrator would output [VALIDATION_PENDING] for single batch
            # Verify reference-based structure supports this
            assert len(result["batches"]) == 1
            assert result["batches"][0]["output_path"] is not None
            assert result["batches"][0]["group_indices"] is not None
            # Reference-based: grouped_file and plan_file are at top level
            assert result["grouped_file"] is not None
            assert result["plan_file"] == "/path/to/plan.md"

    def test_multiple_batches_structure(self, sample_context):
        """Multiple batches have proper structure for orchestrator (reference-based)."""
        groups = [make_group("MEDIUM") for _ in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                plan_file="/path/to/plan.md",
                max_per_batch=4
            )

            # Multiple batches case
            assert result["total_batches"] == 3
            # Orchestrator would output [VALIDATION_BATCHES_PENDING] for multiple batches
            # Verify reference-based structure supports iteration
            for batch in result["batches"]:
                assert batch["output_path"] is not None
                assert batch["batch_index"] is not None
                assert batch["group_indices"] is not None
                # No embedded prompts in reference-based format
                assert "prompt" not in batch

    def test_empty_returns_zero_batches(self, sample_context):
        """Empty groups returns zero batches (no pending marker needed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=[],
                context=sample_context,
                output_dir=tmpdir
            )

            assert result["total_batches"] == 0
            assert result["batches"] == []


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests for batched validation."""

    def test_single_group(self, sample_context):
        """Single group creates single batch."""
        groups = [make_group("MEDIUM")]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir
            )
            assert result["total_batches"] == 1
            assert result["batches"][0]["groups_count"] == 1

    def test_single_high_group(self, sample_context):
        """Single HIGH group creates single isolated batch."""
        groups = [make_group("HIGH")]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir
            )
            assert result["total_batches"] == 1
            assert result["batches"][0]["is_high_priority"] == True

    def test_max_per_batch_one(self, sample_context):
        """max_per_batch=1 creates one batch per group."""
        groups = [make_group("MEDIUM") for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=1
            )
            assert result["total_batches"] == 3
            for batch in result["batches"]:
                assert batch["groups_count"] == 1

    def test_large_number_of_groups(self, sample_context):
        """Handles large number of groups efficiently."""
        groups = [make_group("MEDIUM") for _ in range(100)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )
            # 100 / 4 = 25 batches
            assert result["total_batches"] == 25

            # All indices covered
            all_indices = []
            for batch in result["batches"]:
                all_indices.extend(batch["group_indices"])
            assert sorted(all_indices) == list(range(100))

    def test_many_high_priority_groups(self, sample_context):
        """Many HIGH groups are paired to reduce batch count."""
        # > 5 HIGH groups should be paired
        groups = [make_group("HIGH") for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir
            )
            # 8 HIGH groups paired = 4 batches
            assert result["total_batches"] == 4
            for batch in result["batches"]:
                assert batch["is_high_priority"] == True
                assert batch["groups_count"] == 2

    def test_unicode_in_groups(self, sample_context):
        """Handles Unicode in groups - reference-based format handles Unicode correctly."""
        groups = [
            {
                "theme": "Internationalization 国际化",
                "category": "i18n",
                "models": ["model-a"],
                "suggestions": [
                    {"title": "Add 中文 support", "desc": "Test", "importance": "MEDIUM", "type": "addition"}
                ]
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                plan_file="/path/to/plan.md"
            )
            # In reference-based format, no suggestions_json embedded
            # Just verify the structure is created correctly
            assert result["total_batches"] == 1
            assert result["batches"][0]["group_indices"] == [0]
            assert result["grouped_file"] is not None
            # Unicode in file path is handled correctly
            json_result = json.dumps(result, ensure_ascii=False)
            assert "grouped.json" in json_result


# ============================================================================
# Integration Tests - Real-world scenarios
# ============================================================================

class TestRealWorldScenarios:
    """Integration tests simulating real-world validation scenarios."""

    def test_performance_testing_scenario(self, sample_context):
        """Simulates the scenario from a real run with 41 groups across 13 batches.

        This reproduces the issue where subagents used global indices (2, 3, 4, ...)
        instead of batch-local indices (0, 1, 2, ...) in their output files.
        """
        # Create 41 groups (8 HIGH, 33 MEDIUM/LOW) like the real scenario
        groups = []
        high_indices = [0, 3, 4, 12, 13, 23, 27, 28]  # 8 HIGH priority
        for i in range(41):
            importance = "HIGH" if i in high_indices else "MEDIUM"
            groups.append(make_group(importance, f"Theme {i}"))

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Should have multiple batches due to HIGH isolation
            assert batch_metadata["total_batches"] > 5

            # Write batch results using GLOBAL indices (like real subagents did)
            for batch in batch_metadata["batches"]:
                batch_results = {
                    "groups": [
                        {
                            "group_index": idx,  # Global index!
                            "status": "valid" if idx in high_indices else "needs-human-decision",
                            "reason": f"Validated group {idx}",
                            "confidence": 0.9 if idx in high_indices else 0.7
                        }
                        for idx in batch["group_indices"]
                    ],
                    "metadata": {
                        "model": "claude-haiku-4-5-20251001",
                        "timestamp": "2026-02-01T00:00:00Z",
                        "schema_version": "2.0"
                    }
                }
                with open(batch["output_path"], 'w', encoding="utf-8") as f:
                    json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # All 41 groups should be merged correctly
            assert len(merged) == 41

            # No validation_failed items (all should be mapped correctly)
            failed_count = sum(1 for r in merged if r["status"] == "validation_failed")
            assert failed_count == 0, f"Expected 0 validation_failed, got {failed_count}"

            # HIGH groups should be valid
            for idx in high_indices:
                assert merged[idx]["status"] == "valid", f"Group {idx} should be valid (HIGH)"

            # Other groups should be needs-human-decision
            for idx in range(41):
                if idx not in high_indices:
                    assert merged[idx]["status"] == "needs-human-decision", f"Group {idx} should be needs-human-decision"

    def test_mixed_batch_with_ambiguous_indices(self, sample_context):
        """Test batch where indices could be interpreted as either local or global.

        For batch with group_indices [0, 1, 2, 3] and results with indices 0, 1, 2, 3,
        both interpretations are valid and produce the same result.
        """
        groups = [make_group("MEDIUM") for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            # Single batch with group_indices [0, 1, 2, 3]
            assert len(batch_metadata["batches"]) == 1
            batch = batch_metadata["batches"][0]
            assert batch["group_indices"] == [0, 1, 2, 3]

            # Write with indices 0, 1, 2, 3 - could be local OR global (they match)
            batch_results = {
                "groups": [
                    {"group_index": i, "status": "valid", "reason": "OK", "confidence": 0.9}
                    for i in range(4)
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # All 4 should be valid regardless of interpretation
            assert len(merged) == 4
            assert all(r["status"] == "valid" for r in merged)

    def test_recovery_from_subagent_index_mismatch(self, sample_context):
        """Test recovery when subagent outputs only some indices.

        Simulates a subagent that crashed mid-validation and only output
        results for some groups.
        """
        groups = [make_group("MEDIUM") for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_metadata = prepare_batched_validation_tasks(
                groups=groups,
                context=sample_context,
                output_dir=tmpdir,
                max_per_batch=4
            )

            assert len(batch_metadata["batches"]) == 2

            # Batch 0: only output first 2 of 4 groups (using global indices)
            batch0 = batch_metadata["batches"][0]
            batch0_results = {
                "groups": [
                    {"group_index": 0, "status": "valid", "reason": "OK", "confidence": 0.9},
                    {"group_index": 1, "status": "valid", "reason": "OK", "confidence": 0.9},
                    # Missing: 2, 3
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch0["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch0_results, f)

            # Batch 1: complete (using global indices)
            batch1 = batch_metadata["batches"][1]
            batch1_results = {
                "groups": [
                    {"group_index": idx, "status": "invalid", "reason": "Not OK", "confidence": 0.8}
                    for idx in batch1["group_indices"]
                ],
                "metadata": {"model": "test", "schema_version": "2.0"}
            }
            with open(batch1["output_path"], 'w', encoding="utf-8") as f:
                json.dump(batch1_results, f)

            merged = merge_batched_validation_results(
                output_dir=tmpdir,
                batch_metadata=batch_metadata,
                total_groups=len(groups)
            )

            # Groups 0, 1 should be valid
            assert merged[0]["status"] == "valid"
            assert merged[1]["status"] == "valid"
            # Groups 2, 3 should be validation_failed (missing from output)
            assert merged[2]["status"] == "validation_failed"
            assert merged[3]["status"] == "validation_failed"
            # Groups 4, 5, 6, 7 should be invalid
            for i in range(4, 8):
                assert merged[i]["status"] == "invalid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
