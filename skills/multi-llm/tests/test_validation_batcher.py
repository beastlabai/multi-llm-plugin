#!/usr/bin/env python3
"""
Tests for validation_batcher.py

Tests batching logic for minimizing context rot in validation subagents.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validation_batcher import (
    batch_validation_groups,
    estimate_validation_batching_stats,
    ValidationBatch,
    MAX_GROUPS_PER_BATCH,
    HIGH_THRESHOLD_FOR_PAIRING,
)


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


class TestBatchValidationGroups:
    """Tests for batch_validation_groups function."""

    def test_empty_groups(self):
        """Empty input returns empty batches."""
        result = batch_validation_groups([])
        assert result == []

    def test_single_group(self):
        """Single group goes into one batch."""
        groups = [make_group("MEDIUM")]
        result = batch_validation_groups(groups)

        assert len(result) == 1
        assert result[0].size == 1
        assert result[0].group_indices == [0]
        assert result[0].is_high_priority == False

    def test_single_high_group(self):
        """Single HIGH group is isolated."""
        groups = [make_group("HIGH")]
        result = batch_validation_groups(groups)

        assert len(result) == 1
        assert result[0].size == 1
        assert result[0].is_high_priority == True

    def test_multiple_medium_groups_batched(self):
        """Multiple MEDIUM groups are batched together up to max."""
        groups = [make_group("MEDIUM") for _ in range(6)]
        result = batch_validation_groups(groups, max_per_batch=4)

        # 6 groups with max 4 per batch = 2 batches
        assert len(result) == 2
        assert result[0].size == 4
        assert result[1].size == 2
        assert result[0].is_high_priority == False
        assert result[1].is_high_priority == False

    def test_high_groups_isolated(self):
        """HIGH groups are isolated (1 per batch) when <= threshold."""
        groups = [make_group("HIGH") for _ in range(3)]
        result = batch_validation_groups(groups)

        # 3 HIGH groups = 3 batches (one each)
        assert len(result) == 3
        for batch in result:
            assert batch.size == 1
            assert batch.is_high_priority == True

    def test_many_high_groups_paired(self):
        """When >5 HIGH groups, they are paired (2 per batch)."""
        # Create 8 HIGH groups (> HIGH_THRESHOLD_FOR_PAIRING = 5)
        groups = [make_group("HIGH") for _ in range(8)]
        result = batch_validation_groups(groups)

        # 8 HIGH groups paired = 4 batches
        assert len(result) == 4
        for batch in result:
            assert batch.size == 2
            assert batch.is_high_priority == True

    def test_mixed_importance_groups(self):
        """Mixed HIGH and MEDIUM groups are properly separated."""
        groups = [
            make_group("HIGH"),
            make_group("MEDIUM"),
            make_group("HIGH"),
            make_group("MEDIUM"),
            make_group("MEDIUM"),
            make_group("MEDIUM"),
            make_group("LOW"),
        ]
        result = batch_validation_groups(groups, max_per_batch=4)

        # 2 HIGH groups (isolated) + 5 normal groups (batched in 2 batches of 4+1)
        # HIGH batches come first
        high_batches = [b for b in result if b.is_high_priority]
        normal_batches = [b for b in result if not b.is_high_priority]

        assert len(high_batches) == 2  # 2 isolated HIGH
        assert len(normal_batches) == 2  # 5 normal in 2 batches

        # Check HIGH batches are first and isolated
        for batch in high_batches:
            assert batch.size == 1

    def test_group_indices_preserved(self):
        """Original group indices are correctly tracked."""
        groups = [
            make_group("MEDIUM", "Group 0"),
            make_group("HIGH", "Group 1"),
            make_group("MEDIUM", "Group 2"),
            make_group("MEDIUM", "Group 3"),
        ]
        result = batch_validation_groups(groups)

        # Collect all indices from all batches
        all_indices = []
        for batch in result:
            all_indices.extend(batch.group_indices)

        # Should have all original indices
        assert sorted(all_indices) == [0, 1, 2, 3]

    def test_batch_index_sequential(self):
        """Batch indices are sequential starting from 0."""
        groups = [make_group("MEDIUM") for _ in range(10)]
        result = batch_validation_groups(groups, max_per_batch=3)

        expected_indices = list(range(len(result)))
        actual_indices = [b.batch_index for b in result]
        assert actual_indices == expected_indices

    def test_isolate_high_disabled(self):
        """With isolate_high=False, HIGH groups are batched normally."""
        groups = [make_group("HIGH") for _ in range(3)]
        result = batch_validation_groups(groups, max_per_batch=4, isolate_high=False)

        # All 3 should be in one batch
        assert len(result) == 1
        assert result[0].size == 3

    def test_to_dict(self):
        """ValidationBatch.to_dict() produces correct output."""
        groups = [make_group("HIGH")]
        result = batch_validation_groups(groups)
        batch_dict = result[0].to_dict()

        assert "groups" in batch_dict
        assert "group_indices" in batch_dict
        assert "batch_index" in batch_dict
        assert "is_high_priority" in batch_dict
        assert "size" in batch_dict
        assert batch_dict["size"] == 1


class TestEstimateBatchingStats:
    """Tests for estimate_validation_batching_stats function."""

    def test_empty_groups(self):
        """Empty groups return zero stats."""
        stats = estimate_validation_batching_stats([])

        assert stats["total_groups"] == 0
        assert stats["high_count"] == 0
        assert stats["normal_count"] == 0
        assert stats["estimated_batches"] == 0
        assert stats["subagent_calls_saved"] == 0
        assert stats["efficiency_gain_percent"] == 0.0

    def test_single_group(self):
        """Single group produces correct stats."""
        stats = estimate_validation_batching_stats([make_group("MEDIUM")])

        assert stats["total_groups"] == 1
        assert stats["high_count"] == 0
        assert stats["normal_count"] == 1
        assert stats["estimated_batches"] == 1
        assert stats["subagent_calls_saved"] == 0

    def test_efficiency_calculation(self):
        """Efficiency is calculated correctly."""
        # 8 MEDIUM groups with max 4 per batch = 2 batches
        # Saved: 8 - 2 = 6 calls, efficiency = 6/8 = 75%
        groups = [make_group("MEDIUM") for _ in range(8)]
        stats = estimate_validation_batching_stats(groups, max_per_batch=4)

        assert stats["total_groups"] == 8
        assert stats["estimated_batches"] == 2
        assert stats["subagent_calls_saved"] == 6
        assert stats["efficiency_gain_percent"] == 75.0

    def test_high_groups_counted(self):
        """HIGH groups are counted separately."""
        groups = [
            make_group("HIGH"),
            make_group("HIGH"),
            make_group("MEDIUM"),
            make_group("LOW"),
        ]
        stats = estimate_validation_batching_stats(groups)

        assert stats["high_count"] == 2
        assert stats["normal_count"] == 2

    def test_many_high_groups_paired(self):
        """When >5 HIGH, estimate accounts for pairing."""
        # 8 HIGH groups paired = 4 batches
        groups = [make_group("HIGH") for _ in range(8)]
        stats = estimate_validation_batching_stats(groups)

        assert stats["high_count"] == 8
        assert stats["estimated_batches"] == 4  # 8 / 2 = 4 paired batches


class TestImportanceExtraction:
    """Tests for importance level extraction from groups."""

    def test_group_level_importance(self):
        """Importance at group level is used."""
        group = {
            "importance": "HIGH",
            "theme": "Test",
            "suggestions": []
        }
        batches = batch_validation_groups([group])
        assert batches[0].is_high_priority == True

    def test_suggestion_level_importance(self):
        """Highest importance from suggestions is used."""
        group = {
            "theme": "Test",
            "suggestions": [
                {"importance": "LOW", "title": "Low"},
                {"importance": "HIGH", "title": "High"},
                {"importance": "MEDIUM", "title": "Medium"},
            ]
        }
        batches = batch_validation_groups([group])
        assert batches[0].is_high_priority == True

    def test_case_insensitive_importance(self):
        """Importance matching is case-insensitive."""
        group = make_group("high")  # lowercase
        batches = batch_validation_groups([group])
        assert batches[0].is_high_priority == True

    def test_missing_importance_defaults_medium(self):
        """Missing importance defaults to MEDIUM."""
        group = {"theme": "Test", "suggestions": []}
        batches = batch_validation_groups([group])
        assert batches[0].is_high_priority == False


class TestEdgeCases:
    """Edge case tests."""

    def test_exactly_max_per_batch_groups(self):
        """Exactly max_per_batch groups = 1 batch."""
        groups = [make_group("MEDIUM") for _ in range(4)]
        result = batch_validation_groups(groups, max_per_batch=4)

        assert len(result) == 1
        assert result[0].size == 4

    def test_max_per_batch_plus_one(self):
        """max_per_batch + 1 groups = 2 batches."""
        groups = [make_group("MEDIUM") for _ in range(5)]
        result = batch_validation_groups(groups, max_per_batch=4)

        assert len(result) == 2
        assert result[0].size == 4
        assert result[1].size == 1

    def test_exactly_high_threshold(self):
        """Exactly HIGH_THRESHOLD_FOR_PAIRING (5) HIGH groups are isolated."""
        groups = [make_group("HIGH") for _ in range(HIGH_THRESHOLD_FOR_PAIRING)]
        result = batch_validation_groups(groups)

        assert len(result) == HIGH_THRESHOLD_FOR_PAIRING
        for batch in result:
            assert batch.size == 1

    def test_high_threshold_plus_one(self):
        """HIGH_THRESHOLD_FOR_PAIRING + 1 (6) HIGH groups are paired."""
        groups = [make_group("HIGH") for _ in range(HIGH_THRESHOLD_FOR_PAIRING + 1)]
        result = batch_validation_groups(groups)

        # 6 HIGH groups paired = 3 batches
        assert len(result) == 3
        for batch in result:
            assert batch.size == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
