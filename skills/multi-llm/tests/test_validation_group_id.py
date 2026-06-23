#!/usr/bin/env python3
"""
Tests for group_id-based validation matching functionality.

Tests the following behaviors:
- apply_validation_to_groups() uses group_id for matching when available
- save_validation_results() includes group_id when groups are provided
- prepare_batched_validation_tasks() includes group_ids in batch metadata
- merge_batched_validation_results() uses group_id from metadata
- Reaggregation scenarios preserve validation across reordering
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validation import (
    apply_validation_to_groups,
    save_validation_results,
    prepare_batched_validation_tasks,
    merge_batched_validation_results,
    load_validation_results,
)
from utils.state_manager import generate_group_id


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_groups() -> List[Dict[str, Any]]:
    """Sample suggestion groups for testing."""
    return [
        {
            "theme": "Security Vulnerability",
            "category": "security",
            "models": ["model-a", "model-b"],
            "suggestions": [
                {
                    "title": "SQL Injection Risk",
                    "desc": "Sanitize user inputs in database queries",
                    "importance": "HIGH",
                    "type": "fix"
                }
            ]
        },
        {
            "theme": "Documentation",
            "category": "docs",
            "models": ["model-a"],
            "suggestions": [
                {
                    "title": "Add README",
                    "desc": "Document API endpoints",
                    "importance": "LOW",
                    "type": "addition"
                }
            ]
        },
        {
            "theme": "Performance",
            "category": "optimization",
            "models": ["model-b"],
            "suggestions": [
                {
                    "title": "Cache Results",
                    "desc": "Add caching layer for expensive queries",
                    "importance": "MEDIUM",
                    "type": "addition"
                }
            ]
        }
    ]


def make_group(importance: str = "MEDIUM", theme: str = "Test Theme", desc: str = "Test description") -> dict:
    """Helper to create a test group."""
    return {
        "theme": theme,
        "category": "test",
        "models": ["test-model"],
        "suggestions": [
            {
                "title": "Test suggestion",
                "desc": desc,
                "importance": importance,
                "type": "modification"
            }
        ]
    }


# ============================================================================
# Tests for apply_validation_to_groups() with group_id
# ============================================================================

class TestApplyValidationWithGroupId:
    """Unit tests for group_id-based validation matching in apply_validation_to_groups()."""

    def test_match_by_group_id_when_available(self):
        """Validation results with group_id should match by content hash, not index.

        This test simulates reaggregation where groups are reordered but validation
        results have group_ids that should still match correctly.
        """
        # Create two groups
        security_group = make_group("HIGH", "Security Issue", "Fix authentication bypass")
        docs_group = make_group("LOW", "Documentation", "Add API docs")

        # Original order: [security, docs]
        original_groups = [security_group, docs_group]

        # Compute group_ids
        security_gid = generate_group_id(security_group)
        docs_gid = generate_group_id(docs_group)

        # Validation results in ORIGINAL order with group_ids
        validation_results = [
            {
                "group_index": 0,
                "group_id": security_gid,
                "status": "valid",
                "reason": "Real security issue",
                "confidence": 0.95
            },
            {
                "group_index": 1,
                "group_id": docs_gid,
                "status": "invalid",
                "reason": "Not needed for MVP",
                "confidence": 0.8
            }
        ]

        # NEW order after reaggregation: [docs, security] (reversed)
        reordered_groups = [docs_group, security_group]

        # Apply validation to reordered groups
        result = apply_validation_to_groups(reordered_groups, validation_results)

        # Verify: docs (now at index 0) should get "invalid" status (its original validation)
        assert result[0]["validation_status"] == "invalid"
        assert result[0]["validation_reason"] == "Not needed for MVP"

        # Verify: security (now at index 1) should get "valid" status (its original validation)
        assert result[1]["validation_status"] == "valid"
        assert result[1]["validation_reason"] == "Real security issue"

    def test_fallback_to_index_when_no_group_id(self):
        """Backward compat - validation results without group_id should use index matching."""
        groups = [
            make_group("HIGH", "Group A"),
            make_group("LOW", "Group B")
        ]

        # Validation results WITHOUT group_id (old format)
        validation_results = [
            {
                "group_index": 0,
                "status": "valid",
                "reason": "Good suggestion",
                "confidence": 0.9
            },
            {
                "group_index": 1,
                "status": "invalid",
                "reason": "False positive",
                "confidence": 0.85
            }
        ]

        result = apply_validation_to_groups(groups, validation_results)

        # Should match by index
        assert result[0]["validation_status"] == "valid"
        assert result[1]["validation_status"] == "invalid"

    def test_partial_group_id_coverage(self):
        """Some results have group_id, some don't - verify each uses appropriate matching method."""
        group_a = make_group("HIGH", "Group A", "Description A")
        group_b = make_group("MEDIUM", "Group B", "Description B")
        group_c = make_group("LOW", "Group C", "Description C")

        groups = [group_a, group_b, group_c]

        gid_a = generate_group_id(group_a)
        gid_c = generate_group_id(group_c)

        # Mixed: group_a and group_c have group_id, group_b does not
        validation_results = [
            {
                "group_index": 0,
                "group_id": gid_a,  # Has group_id
                "status": "valid",
                "reason": "Matched by group_id",
                "confidence": 0.9
            },
            {
                "group_index": 1,
                # No group_id - will match by index
                "status": "invalid",
                "reason": "Matched by index",
                "confidence": 0.8
            },
            {
                "group_index": 2,
                "group_id": gid_c,  # Has group_id
                "status": "needs-human-decision",
                "reason": "Also matched by group_id",
                "confidence": 0.5
            }
        ]

        result = apply_validation_to_groups(groups, validation_results)

        # All should be matched correctly
        assert result[0]["validation_status"] == "valid"
        assert result[0]["validation_reason"] == "Matched by group_id"

        assert result[1]["validation_status"] == "invalid"
        assert result[1]["validation_reason"] == "Matched by index"

        assert result[2]["validation_status"] == "needs-human-decision"
        assert result[2]["validation_reason"] == "Also matched by group_id"

    def test_missing_validation_gets_default(self):
        """Groups without matching validation (neither by group_id nor index) should get needs-human-decision."""
        groups = [
            make_group("HIGH", "Group A"),
            make_group("MEDIUM", "Group B"),
            make_group("LOW", "Group C")  # No validation for this one
        ]

        # Only validate first two groups
        gid_a = generate_group_id(groups[0])
        gid_b = generate_group_id(groups[1])

        validation_results = [
            {
                "group_index": 0,
                "group_id": gid_a,
                "status": "valid",
                "reason": "Good",
                "confidence": 0.9
            },
            {
                "group_index": 1,
                "group_id": gid_b,
                "status": "invalid",
                "reason": "Bad",
                "confidence": 0.8
            }
            # Note: no entry for group C at index 2
        ]

        result = apply_validation_to_groups(groups, validation_results)

        # First two should have their validation
        assert result[0]["validation_status"] == "valid"
        assert result[1]["validation_status"] == "invalid"

        # Third should get default status
        assert result[2]["validation_status"] == "needs-human-decision"
        assert result[2]["validation_reason"] == "No validation result"


# ============================================================================
# Tests for save_validation_results() with group_id
# ============================================================================

class TestSaveValidationWithGroupId:
    """Unit tests for saving validation results with group_id."""

    def test_save_includes_group_id_when_groups_provided(self, temp_dir):
        """When groups parameter is provided, group_id should be computed and saved."""
        groups = [
            make_group("HIGH", "Security"),
            make_group("LOW", "Docs")
        ]

        validation_results = [
            {"group_index": 0, "status": "valid", "reason": "Good", "confidence": 0.9},
            {"group_index": 1, "status": "invalid", "reason": "Bad", "confidence": 0.8}
        ]

        output_path = Path(temp_dir) / "validation.json"

        # Save with groups parameter
        save_validation_results(
            validation_results=validation_results,
            output_path=output_path,
            model="test-model",
            groups=groups
        )

        # Load and verify group_ids were saved
        with open(output_path, 'r') as f:
            saved_data = json.load(f)

        saved_groups = saved_data["groups"]

        # Verify each saved result has group_id matching the computed value
        expected_gid_0 = generate_group_id(groups[0])
        expected_gid_1 = generate_group_id(groups[1])

        assert saved_groups[0]["group_id"] == expected_gid_0
        assert saved_groups[1]["group_id"] == expected_gid_1

    def test_save_without_groups_preserves_existing_group_id(self, temp_dir):
        """When groups not provided, existing group_id in validation results should be preserved."""
        existing_gid = "abc123def456"

        validation_results = [
            {
                "group_index": 0,
                "group_id": existing_gid,  # Pre-existing group_id
                "status": "valid",
                "reason": "Already has group_id",
                "confidence": 0.9
            }
        ]

        output_path = Path(temp_dir) / "validation.json"

        # Save WITHOUT groups parameter
        save_validation_results(
            validation_results=validation_results,
            output_path=output_path,
            model="test-model"
            # groups not provided
        )

        # Load and verify existing group_id was preserved
        with open(output_path, 'r') as f:
            saved_data = json.load(f)

        assert saved_data["groups"][0]["group_id"] == existing_gid


# ============================================================================
# Tests for prepare_batched_validation_tasks() with group_ids
# ============================================================================

class TestBatchPreparationWithGroupId:
    """Unit tests for batch preparation including group_ids."""

    def test_batch_metadata_includes_group_ids(self, temp_dir):
        """Batch metadata should include group_ids list parallel to group_indices.

        Verify each batch has group_ids matching the groups at those indices.
        """
        groups = [
            make_group("MEDIUM", f"Group {i}", f"Description {i}")
            for i in range(6)
        ]

        # Pre-compute expected group_ids
        expected_gids = [generate_group_id(g) for g in groups]

        result = prepare_batched_validation_tasks(
            groups=groups,
            context="Test context",
            output_dir=temp_dir,
            plan_file="/path/to/plan.md",
            max_per_batch=4
        )

        # Should have 2 batches (4 + 2)
        assert len(result["batches"]) == 2

        # Verify each batch has group_ids parallel to group_indices
        for batch in result["batches"]:
            assert "group_ids" in batch
            assert len(batch["group_ids"]) == len(batch["group_indices"])

            # Verify each group_id matches the expected value for that index
            for i, group_idx in enumerate(batch["group_indices"]):
                assert batch["group_ids"][i] == expected_gids[group_idx]


# ============================================================================
# Tests for merge_batched_validation_results() with group_id
# ============================================================================

class TestMergeBatchedWithGroupId:
    """Unit tests for merging batched results with group_id."""

    def test_merge_uses_group_id_from_metadata(self, temp_dir):
        """Merged results should include group_id from batch metadata."""
        groups = [make_group("MEDIUM", f"Group {i}") for i in range(4)]

        batch_metadata = prepare_batched_validation_tasks(
            groups=groups,
            context="Test context",
            output_dir=temp_dir,
            plan_file="/path/to/plan.md",
            max_per_batch=4
        )

        # Write batch result file (single batch for 4 groups)
        batch = batch_metadata["batches"][0]
        batch_results = {
            "groups": [
                {"group_index": i, "status": "valid", "reason": "OK", "confidence": 0.9}
                for i in range(4)
            ],
            "metadata": {"model": "test", "schema_version": "2.1"}
        }
        with open(batch["output_path"], 'w') as f:
            json.dump(batch_results, f)

        # Merge results
        merged = merge_batched_validation_results(
            output_dir=temp_dir,
            batch_metadata=batch_metadata,
            total_groups=len(groups)
        )

        # Verify each merged result has group_id from metadata
        expected_gids = batch["group_ids"]
        for i, result in enumerate(merged):
            assert "group_id" in result
            assert result["group_id"] == expected_gids[i]


# ============================================================================
# Integration Tests for Reaggregation Preserving Validation
# ============================================================================

class TestValidationGroupIdIntegration:
    """Integration test for reaggregation preserving validation."""

    def test_reaggregation_preserves_validation_across_reordering(self, temp_dir):
        """Simulate reaggregation where existing groups are reordered.

        This test demonstrates that group_id matching preserves validation
        when existing groups change position in the list.

        Scenario:
        1. Initial groups: [Security (HIGH), Docs (LOW)]
        2. Validation marks Security as "valid", Docs as "invalid"
        3. Reaggregation reverses order: [Docs, Security]
        4. Validation should be preserved via group_id matching
        """
        # Step 1: Create initial groups
        security_group = {
            "theme": "Security Vulnerability",
            "category": "security",
            "models": ["model-a"],
            "suggestions": [
                {
                    "title": "Fix Auth Bypass",
                    "desc": "Prevent unauthorized access",
                    "importance": "HIGH",
                    "type": "fix"
                }
            ]
        }

        docs_group = {
            "theme": "Documentation Missing",
            "category": "docs",
            "models": ["model-b"],
            "suggestions": [
                {
                    "title": "Add API docs",
                    "desc": "Document REST endpoints",
                    "importance": "LOW",
                    "type": "addition"
                }
            ]
        }

        # Initial order: [security (HIGH), docs (LOW)]
        initial_groups = [security_group, docs_group]

        # Step 2: Create validation results with group_ids
        security_gid = generate_group_id(security_group)
        docs_gid = generate_group_id(docs_group)

        initial_validation = [
            {
                "group_index": 0,
                "group_id": security_gid,
                "status": "valid",
                "reason": "Real security issue",
                "confidence": 0.95
            },
            {
                "group_index": 1,
                "group_id": docs_gid,
                "status": "invalid",
                "reason": "Docs not needed for MVP",
                "confidence": 0.85
            }
        ]

        # Step 3: Save validation with groups parameter
        validation_path = Path(temp_dir) / "validation.json"
        save_validation_results(
            validation_results=initial_validation,
            output_path=validation_path,
            model="test-model",
            groups=initial_groups
        )

        # Step 4: Simulate reaggregation - reverse the order
        # New order: [docs (was index 1), security (was index 0)]
        reaggregated_groups = [docs_group, security_group]

        # Step 5: Load saved validation
        loaded_validation = load_validation_results(validation_path)

        # Step 6: Apply validation to reaggregated groups
        result = apply_validation_to_groups(reaggregated_groups, loaded_validation)

        # Step 7: Verify results - validation follows by group_id, not index

        # Docs (now at index 0) should get its original "invalid" status
        assert result[0]["validation_status"] == "invalid"
        assert result[0]["validation_reason"] == "Docs not needed for MVP"
        assert result[0]["validation_confidence"] == 0.85

        # Security (now at index 1) should get its original "valid" status
        assert result[1]["validation_status"] == "valid"
        assert result[1]["validation_reason"] == "Real security issue"
        assert result[1]["validation_confidence"] == 0.95

    def test_new_group_gets_index_fallback_when_at_existing_index(self, temp_dir):
        """When a new group is added at an existing index, it uses index fallback.

        This test documents the current behavior: the index fallback is always
        checked when group_id doesn't match, regardless of whether validation
        results have group_ids.

        In practice, this means new groups at existing indices will inherit
        the validation from whatever was at that index before. This is a
        known limitation that may be addressed in future versions.
        """
        # Create initial groups
        security_group = {
            "theme": "Security",
            "category": "security",
            "models": ["model-a"],
            "suggestions": [{"title": "Fix", "desc": "Security fix", "importance": "HIGH", "type": "fix"}]
        }

        # Initial validation
        security_gid = generate_group_id(security_group)
        validation_results = [
            {
                "group_index": 0,
                "group_id": security_gid,
                "status": "valid",
                "reason": "Security issue",
                "confidence": 0.9
            }
        ]

        # New group with different content
        new_group = {
            "theme": "Performance",
            "category": "perf",
            "models": ["model-b"],
            "suggestions": [{"title": "Cache", "desc": "Add caching", "importance": "MEDIUM", "type": "addition"}]
        }

        # Apply validation to just the new group at index 0
        result = apply_validation_to_groups([new_group], validation_results)

        # Current behavior: new group at index 0 falls back to index matching
        # and inherits the validation from the security group that was at index 0
        # (because its group_id doesn't match, but there's a validation at index 0)
        assert result[0]["validation_status"] == "valid"
        assert result[0]["validation_reason"] == "Security issue"

    def test_batch_validation_with_reaggregation(self, temp_dir):
        """Test batched validation workflow handles reaggregation correctly.

        Simulates a more complex scenario with batched validation results
        being merged and then reapplied after group reordering.
        """
        # Create 6 groups that will be batched
        groups = [
            make_group("HIGH", "Security", "Fix SQL injection"),
            make_group("MEDIUM", "Performance", "Add caching"),
            make_group("MEDIUM", "Reliability", "Add error handling"),
            make_group("MEDIUM", "Testing", "Add unit tests"),
            make_group("LOW", "Style", "Fix formatting"),
            make_group("LOW", "Docs", "Add comments"),
        ]

        # Prepare batched tasks
        batch_metadata = prepare_batched_validation_tasks(
            groups=groups,
            context="Test plan context",
            output_dir=temp_dir,
            plan_file="/test/plan.md",
            max_per_batch=4
        )

        # Simulate subagent writing batch results
        for batch in batch_metadata["batches"]:
            # Use global indices as subagents typically do
            batch_results = {
                "groups": [
                    {
                        "group_index": idx,
                        "status": "valid" if idx == 0 else "needs-human-decision",
                        "reason": f"Validated group {idx}",
                        "confidence": 0.9 if idx == 0 else 0.6
                    }
                    for idx in batch["group_indices"]
                ],
                "metadata": {"model": "test", "schema_version": "2.1"}
            }
            with open(batch["output_path"], 'w') as f:
                json.dump(batch_results, f)

        # Merge batch results
        merged = merge_batched_validation_results(
            output_dir=temp_dir,
            batch_metadata=batch_metadata,
            total_groups=len(groups)
        )

        # Verify all results have group_ids
        for i, result in enumerate(merged):
            assert "group_id" in result, f"Result {i} missing group_id"

        # Now simulate reaggregation - reverse the order
        reordered_groups = list(reversed(groups))

        # Apply merged validation to reordered groups
        final_result = apply_validation_to_groups(reordered_groups, merged)

        # Find where the Security group ended up (should be last after reversal)
        security_index = 5  # Was index 0, now at index 5 after reversal

        # Verify Security group (HIGH priority, originally index 0) still has "valid"
        assert final_result[security_index]["validation_status"] == "valid"
        assert "Validated group 0" in final_result[security_index]["validation_reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
