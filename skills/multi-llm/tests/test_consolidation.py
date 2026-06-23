"""Tests for the consolidation utility module (utils/consolidation.py)."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.consolidation import (
    generate_group_id,
    generate_consolidated_id,
    normalize_reference,
    pre_group_by_section,
    prepare_consolidation_tasks,
    merge_consolidation_results,
    generate_consolidated_json,
    generate_consolidated_report,
    load_merged_suggestions,
    _word_overlap_ratio,
    _highest_importance,
    _resolve_type_for_group_ids,
    _make_singleton_consolidated,
    CONSOLIDATION_SPLIT_THRESHOLD,
    MAX_GROUPS_PER_CONSOLIDATION_BATCH,
    CONSOLIDATION_CHAR_BUDGET,
    TYPE_PRIORITY,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_group(
    theme="Test theme",
    suggestions=None,
    reference="Task 1",
    models=None,
    category="modification",
    importance="MEDIUM",
):
    """Helper to create a test group dict."""
    if suggestions is None:
        suggestions = [
            {
                "title": "Test suggestion",
                "desc": "Test description",
                "type": "modification",
                "importance": "MEDIUM",
            }
        ]
    return {
        "theme": theme,
        "suggestions": suggestions,
        "reference": reference,
        "models": models or ["model-a"],
        "category": category,
        "importance": importance,
    }


def _make_validation_entry(group_index, status="valid", reason=""):
    """Helper to create a validation entry."""
    return {"group_index": group_index, "status": status, "reason": reason}


def _build_valid_metadata(
    total_original=10,
    total_consolidated=5,
    merged_count=3,
    singleton_count=2,
    skipped_report=False,
):
    """Helper to build a valid metadata dict for consolidated.json."""
    return {
        "schema_version": "1.0",
        "total_original_groups": total_original,
        "total_consolidated": total_consolidated,
        "merged_count": merged_count,
        "singleton_count": singleton_count,
        "consolidation_ratio": total_original / max(total_consolidated, 1),
        "timestamp": "2026-01-01T00:00:00",
        "plan_hash": "abc123",
        "grouped_hash": "def456",
        "plan_hash_algorithm": "sha256",
        "skipped_report": skipped_report,
        "sections_processed": 3,
    }


# ===========================================================================
# 1. generate_group_id
# ===========================================================================


class TestGenerateGroupId:
    """Tests for generate_group_id()."""

    def test_stable_12_char_hex(self):
        """ID is a stable 12-char hex string from theme + sorted titles."""
        group = _make_group(
            theme="Error handling",
            suggestions=[
                {"title": "Add try-catch"},
                {"title": "Add validation"},
            ],
        )
        gid = generate_group_id(group)
        assert len(gid) == 12
        assert all(c in "0123456789abcdef" for c in gid)

    def test_deterministic_across_calls(self):
        """Same group always produces the same ID."""
        group = _make_group(theme="Performance", suggestions=[{"title": "Cache"}])
        assert generate_group_id(group) == generate_group_id(group)

    def test_different_groups_produce_different_ids(self):
        """Different themes/titles produce different IDs."""
        g1 = _make_group(theme="Alpha", suggestions=[{"title": "A"}])
        g2 = _make_group(theme="Beta", suggestions=[{"title": "B"}])
        assert generate_group_id(g1) != generate_group_id(g2)

    def test_titles_sorted_for_stability(self):
        """Order of suggestions does not affect ID (titles are sorted)."""
        g1 = _make_group(
            theme="T",
            suggestions=[{"title": "Z"}, {"title": "A"}],
        )
        g2 = _make_group(
            theme="T",
            suggestions=[{"title": "A"}, {"title": "Z"}],
        )
        assert generate_group_id(g1) == generate_group_id(g2)

    def test_empty_theme_and_titles(self):
        """Empty theme/titles still produce a valid 12-char hex."""
        group = _make_group(theme="", suggestions=[{"title": ""}])
        gid = generate_group_id(group)
        assert len(gid) == 12
        assert all(c in "0123456789abcdef" for c in gid)

    def test_missing_theme_and_suggestions(self):
        """Group with no theme/suggestions keys still works."""
        gid = generate_group_id({})
        assert len(gid) == 12
        assert all(c in "0123456789abcdef" for c in gid)


# ===========================================================================
# 2. generate_consolidated_id
# ===========================================================================


class TestGenerateConsolidatedId:
    """Tests for generate_consolidated_id()."""

    def test_singleton_returns_underlying_id_directly(self):
        """Single underlying group ID is returned as-is (no hashing)."""
        assert generate_consolidated_id(["abc123def456"]) == "abc123def456"

    def test_multiple_ids_produce_12_char_hex(self):
        """Multiple IDs produce a stable 12-char hex hash."""
        cid = generate_consolidated_id(["aaa", "bbb", "ccc"])
        assert len(cid) == 12
        assert all(c in "0123456789abcdef" for c in cid)

    def test_stable_hash_from_sorted_ids(self):
        """Same set of IDs always produces the same hash."""
        ids = ["id_c", "id_a", "id_b"]
        assert generate_consolidated_id(ids) == generate_consolidated_id(ids)

    def test_order_independent(self):
        """Order of input IDs does not affect the output (sorted internally)."""
        assert generate_consolidated_id(["x", "y"]) == generate_consolidated_id(
            ["y", "x"]
        )

    def test_different_sets_produce_different_ids(self):
        """Different ID sets produce different consolidated IDs."""
        cid1 = generate_consolidated_id(["a", "b"])
        cid2 = generate_consolidated_id(["c", "d"])
        assert cid1 != cid2


# ===========================================================================
# 3. normalize_reference
# ===========================================================================


class TestNormalizeReference:
    """Tests for normalize_reference()."""

    def test_strip_leading_hashes(self):
        """Leading '#' characters and whitespace are stripped."""
        assert normalize_reference("## Task 3") == "task 3"

    def test_lowercase(self):
        """Reference is lowercased."""
        assert normalize_reference("Task 3") == "task 3"

    def test_strip_trailing_colon_description(self):
        """Description after colon is removed."""
        assert normalize_reference("task 3: api integration") == "task 3"

    def test_normalize_leading_zeros(self):
        """Leading zeros in numbers are stripped."""
        assert normalize_reference("task 01") == "task 1"
        assert normalize_reference("section 007") == "section 7"

    def test_collapse_whitespace(self):
        """Multiple whitespace chars are collapsed to single space."""
        assert normalize_reference("task   3") == "task 3"

    def test_none_returns_uncategorized(self):
        """None input returns '_uncategorized'."""
        assert normalize_reference(None) == "_uncategorized"

    def test_empty_string_returns_uncategorized(self):
        """Empty string returns '_uncategorized'."""
        assert normalize_reference("") == "_uncategorized"

    def test_whitespace_only_returns_uncategorized(self):
        """Whitespace-only returns '_uncategorized'."""
        assert normalize_reference("   ") == "_uncategorized"

    def test_hashes_only_returns_uncategorized(self):
        """Only '#' chars returns '_uncategorized'."""
        assert normalize_reference("###") == "_uncategorized"

    def test_combined_normalization(self):
        """All normalization rules apply together."""
        assert normalize_reference("  ## Task  01: Description ") == "task 1"


# ===========================================================================
# 4. pre_group_by_section
# ===========================================================================


class TestPreGroupBySection:
    """Tests for pre_group_by_section()."""

    def test_groups_by_normalized_reference(self):
        """Groups are grouped by their normalized reference."""
        groups = [
            _make_group(reference="Task 1"),
            _make_group(reference="## Task 1"),
            _make_group(reference="Task 2"),
        ]
        validation = [
            _make_validation_entry(0, "valid"),
            _make_validation_entry(1, "valid"),
            _make_validation_entry(2, "valid"),
        ]
        result = pre_group_by_section(groups, validation)
        assert sorted(result["task 1"]) == [0, 1]
        assert result["task 2"] == [2]

    def test_filters_to_valid_and_needs_human_decision(self):
        """Only 'valid' and 'needs-human-decision' statuses are accepted."""
        groups = [
            _make_group(reference="Task 1"),
            _make_group(reference="Task 1"),
            _make_group(reference="Task 1"),
        ]
        validation = [
            _make_validation_entry(0, "valid"),
            _make_validation_entry(1, "invalid"),
            _make_validation_entry(2, "needs-human-decision"),
        ]
        result = pre_group_by_section(groups, validation)
        assert sorted(result["task 1"]) == [0, 2]

    def test_omits_all_invalid_sections(self):
        """Sections where all groups are invalid are excluded."""
        groups = [
            _make_group(reference="Task 1"),
            _make_group(reference="Task 2"),
        ]
        validation = [
            _make_validation_entry(0, "invalid"),
            _make_validation_entry(1, "valid"),
        ]
        result = pre_group_by_section(groups, validation)
        assert "task 1" not in result
        assert "task 2" in result

    def test_handles_missing_reference_field(self):
        """Groups without a reference field get '_uncategorized'."""
        groups = [{"theme": "Theme A", "suggestions": []}]
        validation = [_make_validation_entry(0, "valid")]
        result = pre_group_by_section(groups, validation)
        assert "_uncategorized" in result


# ===========================================================================
# 5. prepare_consolidation_tasks
# ===========================================================================


class TestPrepareConsolidationTasks:
    """Tests for prepare_consolidation_tasks()."""

    def test_singleton_sections_in_output(self, tmp_path):
        """Sections with 1 group are placed in singleton_sections."""
        groups = [_make_group(reference="Task 1")]
        section_groups = {"task 1": [0]}
        result = prepare_consolidation_tasks(
            groups, section_groups, str(tmp_path), "plan.md"
        )
        assert "task 1" in result["singleton_sections"]
        assert result["total_batches"] == 0

    def test_multi_group_sections_become_batches(self, tmp_path):
        """Sections with 2+ groups create consolidation batches."""
        groups = [
            _make_group(theme=f"Theme {i}", reference="Task 1") for i in range(3)
        ]
        section_groups = {"task 1": [0, 1, 2]}
        result = prepare_consolidation_tasks(
            groups, section_groups, str(tmp_path), "plan.md"
        )
        assert result["total_batches"] >= 1
        assert len(result["batches"]) >= 1

    def test_respects_split_threshold(self, tmp_path):
        """Sections larger than CONSOLIDATION_SPLIT_THRESHOLD are split."""
        n = CONSOLIDATION_SPLIT_THRESHOLD + 3
        groups = [_make_group(theme=f"T{i}", reference="Task 1") for i in range(n)]
        section_groups = {"task 1": list(range(n))}
        result = prepare_consolidation_tasks(
            groups, section_groups, str(tmp_path), "plan.md"
        )
        # Should have more than 1 batch
        assert result["total_batches"] >= 2
        for batch in result["batches"]:
            assert batch["groups_count"] <= MAX_GROUPS_PER_CONSOLIDATION_BATCH

    def test_char_budget_enforcement(self, tmp_path):
        """Batches exceeding CONSOLIDATION_CHAR_BUDGET are split further."""
        # Create groups with large descriptions that exceed budget
        big_desc = "x" * (CONSOLIDATION_CHAR_BUDGET // 2 + 100)
        groups = [
            _make_group(
                theme=f"T{i}",
                suggestions=[{"title": "S", "description": big_desc}],
                reference="Task 1",
            )
            for i in range(4)
        ]
        section_groups = {"task 1": [0, 1, 2, 3]}
        result = prepare_consolidation_tasks(
            groups, section_groups, str(tmp_path), "plan.md"
        )
        # Each batch should have at most 2 groups (budget limits)
        for batch in result["batches"]:
            assert batch["groups_count"] <= 2

    def test_global_sequential_batch_numbering(self, tmp_path):
        """Batch numbering is global and sequential across sections."""
        groups = [
            _make_group(theme=f"A{i}", reference="Task 1") for i in range(3)
        ] + [_make_group(theme=f"B{i}", reference="Task 2") for i in range(2)]
        section_groups = {"task 1": [0, 1, 2], "task 2": [3, 4]}
        result = prepare_consolidation_tasks(
            groups, section_groups, str(tmp_path), "plan.md"
        )
        indices = [b["batch_index"] for b in result["batches"]]
        assert indices == list(range(len(indices)))

    def test_output_contains_reaggregate_command(self, tmp_path):
        """Output includes the reaggregate_command field."""
        groups = [_make_group(reference="Task 1") for _ in range(2)]
        section_groups = {"task 1": [0, 1]}
        result = prepare_consolidation_tasks(
            groups, section_groups, str(tmp_path), "plan.md"
        )
        assert result["reaggregate_command"] == "--reaggregate-consolidation"

    def test_writes_consolidation_tasks_json(self, tmp_path):
        """consolidation_tasks.json is written to phase_dir."""
        groups = [_make_group(reference="Task 1") for _ in range(2)]
        section_groups = {"task 1": [0, 1]}
        prepare_consolidation_tasks(
            groups, section_groups, str(tmp_path), "plan.md"
        )
        tasks_path = tmp_path / "consolidation_tasks.json"
        assert tasks_path.exists()
        data = json.loads(tasks_path.read_text())
        assert "batches" in data
        assert "singleton_sections" in data


# ===========================================================================
# 6. merge_consolidation_results
# ===========================================================================


class TestMergeConsolidationResults:
    """Tests for merge_consolidation_results()."""

    def _setup_merge(self, tmp_path, groups, batch_data_list, singleton_sections=None):
        """Helper to set up phase_dir, write batch files, and build tasks_metadata."""
        batches = []
        for i, (batch_indices, batch_content) in enumerate(batch_data_list):
            group_ids = [generate_group_id(groups[idx]) for idx in batch_indices]
            batch_file = tmp_path / f"consolidation_batch_{i}.json"
            if batch_content is not None:
                batch_file.write_text(json.dumps(batch_content))
            batches.append(
                {
                    "batch_index": i,
                    "section_key": "task 1",
                    "group_indices": batch_indices,
                    "group_ids": group_ids,
                    "groups_count": len(batch_indices),
                    "output_path": str(batch_file),
                }
            )
        tasks_metadata = {
            "batches": batches,
            "singleton_sections": singleton_sections or {},
            "total_batches": len(batches),
        }
        return tasks_metadata

    def test_successful_merge_with_singletons(self, tmp_path):
        """Clusters and singletons from batches are merged correctly."""
        groups = [
            _make_group(theme=f"Theme {i}", reference="Task 1") for i in range(3)
        ]
        g_ids = [generate_group_id(g) for g in groups]
        batch_content = {
            "clusters": [
                {
                    "title": "Merged cluster",
                    "description": "Combined",
                    "importance": "HIGH",
                    "type": "modification",
                    "underlying_group_ids": [g_ids[0], g_ids[1]],
                    "underlying_group_indices": [0, 1],
                    "reasoning": "Related themes",
                }
            ],
            "singletons": [
                {"group_id": g_ids[2], "group_index": 2, "reasoning": "Standalone"}
            ],
        }
        tasks_metadata = self._setup_merge(
            tmp_path, groups, [([0, 1, 2], batch_content)]
        )
        result, pf = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        assert len(result) == 2  # 1 cluster + 1 singleton
        assert pf["count"] == 0

    def test_missing_batch_file_falls_back_to_singletons(self, tmp_path):
        """Missing batch file causes all groups to be treated as singletons."""
        groups = [_make_group(theme=f"T{i}", reference="Task 1") for i in range(2)]
        # Pass None so file is not written
        tasks_metadata = self._setup_merge(
            tmp_path, groups, [([0, 1], None)]
        )
        result, pf = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        assert pf["count"] == 1
        assert all(cg["is_singleton"] for cg in result)

    def test_invalid_json_batch_file(self, tmp_path):
        """Invalid JSON in batch file triggers singleton fallback."""
        groups = [_make_group(theme="T0", reference="Task 1")]
        g_ids = [generate_group_id(g) for g in groups]
        batch_file = tmp_path / "consolidation_batch_0.json"
        batch_file.write_text("NOT VALID JSON {{{")
        tasks_metadata = {
            "batches": [
                {
                    "batch_index": 0,
                    "section_key": "task 1",
                    "group_indices": [0],
                    "group_ids": g_ids,
                    "groups_count": 1,
                    "output_path": str(batch_file),
                }
            ],
            "singleton_sections": {},
            "total_batches": 1,
        }
        result, pf = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        assert pf["count"] == 1
        assert 0 in pf["batches"]

    def test_error_batch_file(self, tmp_path):
        """Batch with 'error' key triggers singleton fallback."""
        groups = [_make_group(theme="T0", reference="Task 1")]
        g_ids = [generate_group_id(g) for g in groups]
        batch_content = {"error": "timeout", "groups": g_ids}
        tasks_metadata = self._setup_merge(
            tmp_path, groups, [([0], batch_content)]
        )
        result, pf = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        assert pf["count"] == 1
        assert all(cg["is_singleton"] for cg in result)

    def test_duplicate_group_id_first_cluster_wins(self, tmp_path):
        """Duplicate group_id across clusters: first-cluster-wins."""
        groups = [_make_group(theme=f"T{i}", reference="Task 1") for i in range(3)]
        g_ids = [generate_group_id(g) for g in groups]
        # Both clusters claim g_ids[1]
        batch_content = {
            "clusters": [
                {
                    "title": "Cluster A",
                    "description": "A",
                    "importance": "HIGH",
                    "type": "modification",
                    "underlying_group_ids": [g_ids[0], g_ids[1]],
                    "underlying_group_indices": [0, 1],
                    "reasoning": "A",
                },
                {
                    "title": "Cluster B",
                    "description": "B",
                    "importance": "LOW",
                    "type": "clarification",
                    "underlying_group_ids": [g_ids[1], g_ids[2]],
                    "underlying_group_indices": [1, 2],
                    "reasoning": "B",
                },
            ],
            "singletons": [],
        }
        tasks_metadata = self._setup_merge(
            tmp_path, groups, [([0, 1, 2], batch_content)]
        )
        result, pf = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        # Cluster A has g_ids[0] and g_ids[1]
        # Cluster B had g_ids[1] removed (dup) leaving only g_ids[2] -> becomes singleton
        cluster_a = [c for c in result if not c["is_singleton"]]
        assert len(cluster_a) == 1
        assert g_ids[0] in cluster_a[0]["underlying_group_ids"]
        assert g_ids[1] in cluster_a[0]["underlying_group_ids"]

    def test_missing_groups_added_as_singletons(self, tmp_path):
        """Groups missing from batch output are added as singletons."""
        groups = [_make_group(theme=f"T{i}", reference="Task 1") for i in range(3)]
        g_ids = [generate_group_id(g) for g in groups]
        # Batch only mentions first group
        batch_content = {
            "clusters": [],
            "singletons": [
                {"group_id": g_ids[0], "group_index": 0, "reasoning": "OK"}
            ],
        }
        tasks_metadata = self._setup_merge(
            tmp_path, groups, [([0, 1, 2], batch_content)]
        )
        result, pf = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        # All 3 groups should appear (index 0 from batch, 1 and 2 as missing singletons)
        all_underlying = set()
        for cg in result:
            for gid in cg["underlying_group_ids"]:
                all_underlying.add(gid)
        assert all_underlying == set(g_ids)

    def test_cross_batch_dedup_merges_overlapping_titles(self, tmp_path):
        """Cross-batch dedup merges clusters with >=0.8 word overlap."""
        groups = [_make_group(theme=f"T{i}", reference="Task 1") for i in range(4)]
        g_ids = [generate_group_id(g) for g in groups]

        # Two batches in same section with highly overlapping cluster titles
        batch0 = {
            "clusters": [
                {
                    "title": "Add error handling for API calls",
                    "description": "Desc A",
                    "importance": "HIGH",
                    "type": "addition",
                    "underlying_group_ids": [g_ids[0], g_ids[1]],
                    "underlying_group_indices": [0, 1],
                    "reasoning": "Reason A",
                }
            ],
            "singletons": [],
        }
        batch1 = {
            "clusters": [
                {
                    "title": "Add error handling for API calls",
                    "description": "Desc B",
                    "importance": "LOW",
                    "type": "clarification",
                    "underlying_group_ids": [g_ids[2], g_ids[3]],
                    "underlying_group_indices": [2, 3],
                    "reasoning": "Reason B",
                }
            ],
            "singletons": [],
        }

        batches_meta = []
        for i, (indices, content) in enumerate([([0, 1], batch0), ([2, 3], batch1)]):
            batch_gids = [g_ids[idx] for idx in indices]
            batch_file = tmp_path / f"consolidation_batch_{i}.json"
            batch_file.write_text(json.dumps(content))
            batches_meta.append(
                {
                    "batch_index": i,
                    "section_key": "task 1",
                    "group_indices": indices,
                    "group_ids": batch_gids,
                    "groups_count": len(indices),
                    "output_path": str(batch_file),
                }
            )

        tasks_metadata = {
            "batches": batches_meta,
            "singleton_sections": {},
            "total_batches": 2,
        }
        result, pf = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        # The two clusters should be merged into 1 due to identical titles
        non_singleton = [c for c in result if not c["is_singleton"]]
        assert len(non_singleton) == 1
        assert len(non_singleton[0]["underlying_group_ids"]) == 4

    def test_type_field_validation_override(self, tmp_path):
        """Type is upgraded per TYPE_PRIORITY when underlying groups have higher type."""
        groups = [
            _make_group(theme="T0", reference="Task 1", category="addition"),
            _make_group(theme="T1", reference="Task 1", category="clarification"),
        ]
        g_ids = [generate_group_id(g) for g in groups]
        batch_content = {
            "clusters": [
                {
                    "title": "Combined",
                    "description": "D",
                    "importance": "MEDIUM",
                    "type": "clarification",  # LLM said clarification
                    "underlying_group_ids": g_ids,
                    "underlying_group_indices": [0, 1],
                    "reasoning": "R",
                }
            ],
            "singletons": [],
        }
        tasks_metadata = self._setup_merge(
            tmp_path, groups, [([0, 1], batch_content)]
        )
        result, pf = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        non_singleton = [c for c in result if not c["is_singleton"]]
        assert len(non_singleton) == 1
        # Should be upgraded to "addition" (higher priority)
        assert non_singleton[0]["type"] == "addition"

    def test_partial_failures_metadata(self, tmp_path):
        """partial_failures tracks failed batch count and indices."""
        groups = [_make_group(theme="T0")]
        # Missing file = failure
        tasks_metadata = self._setup_merge(tmp_path, groups, [([0], None)])
        _result, pf = merge_consolidation_results(
            str(tmp_path), tasks_metadata, groups
        )
        assert pf["count"] == 1
        assert pf["batches"] == [0]
        assert pf["fallback"] == "all_singletons"

    def test_display_index_sequential(self, tmp_path):
        """Consolidated groups get sequential 1-based display_index."""
        groups = [_make_group(theme=f"T{i}", reference="Task 1") for i in range(3)]
        g_ids = [generate_group_id(g) for g in groups]
        batch_content = {
            "clusters": [],
            "singletons": [
                {"group_id": gid, "group_index": i, "reasoning": "OK"}
                for i, gid in enumerate(g_ids)
            ],
        }
        tasks_metadata = self._setup_merge(
            tmp_path, groups, [([0, 1, 2], batch_content)]
        )
        result, _ = merge_consolidation_results(str(tmp_path), tasks_metadata, groups)
        display_indices = [cg["display_index"] for cg in result]
        assert display_indices == list(range(1, len(result) + 1))


# ===========================================================================
# 7. generate_consolidated_report
# ===========================================================================


class TestGenerateConsolidatedReport:
    """Tests for generate_consolidated_report()."""

    def _make_consolidated_group(
        self,
        cid="abc123def456",
        display_index=1,
        title="Test Group",
        underlying_indices=None,
        underlying_ids=None,
        is_singleton=False,
    ):
        return {
            "consolidated_id": cid,
            "display_index": display_index,
            "title": title,
            "description": "Test description",
            "importance": "HIGH",
            "reference": "Task 1",
            "type": "modification",
            "underlying_group_indices": underlying_indices or [0],
            "underlying_group_ids": underlying_ids or ["aaa"],
            "model_count": 2,
            "original_suggestion_count": 3,
            "is_singleton": is_singleton,
            "reasoning": "Test reasoning",
        }

    def test_produces_markdown_with_cg_headings(self, tmp_path):
        """Report has CG headings with bracketed stable IDs."""
        cg = self._make_consolidated_group()
        groups = [_make_group()]
        path = generate_consolidated_report([cg], groups, str(tmp_path), "my-feature")
        content = Path(path).read_text()
        assert "## CG1 [abc123def456]: Test Group" in content

    def test_includes_skip_checkbox(self, tmp_path):
        """Report includes skip checkbox."""
        cg = self._make_consolidated_group()
        groups = [_make_group()]
        path = generate_consolidated_report([cg], groups, str(tmp_path), "prefix")
        content = Path(path).read_text()
        assert "- [ ] Skip this group" in content

    def test_includes_3_state_validation_checkboxes(self, tmp_path):
        """Report includes valid/invalid/needs-human checkboxes."""
        cg = self._make_consolidated_group()
        groups = [_make_group()]
        path = generate_consolidated_report([cg], groups, str(tmp_path), "prefix")
        content = Path(path).read_text()
        assert "- [ ] Mark valid" in content
        assert "- [ ] Mark invalid" in content
        assert "- [ ] Needs human attention" in content

    def test_includes_details_blocks(self, tmp_path):
        """Report includes <details> blocks with original suggestions."""
        cg = self._make_consolidated_group(underlying_indices=[0])
        groups = [_make_group(theme="Original Theme")]
        path = generate_consolidated_report([cg], groups, str(tmp_path), "prefix")
        content = Path(path).read_text()
        assert "<details>" in content
        assert "Original Theme" in content
        assert "</details>" in content

    def test_report_header_contains_stats(self, tmp_path):
        """Report header contains consolidation statistics."""
        cg = self._make_consolidated_group()
        groups = [_make_group()]
        path = generate_consolidated_report([cg], groups, str(tmp_path), "my-feature")
        content = Path(path).read_text()
        assert "# Consolidated Plan Review Report: my-feature" in content
        assert "Consolidation:" in content


# ===========================================================================
# 8. generate_consolidated_json
# ===========================================================================


class TestGenerateConsolidatedJson:
    """Tests for generate_consolidated_json()."""

    def _make_valid_cg(self, cid="aabbccddeeff", display_index=1):
        return {
            "consolidated_id": cid,
            "display_index": display_index,
            "title": "Test",
            "description": "Desc",
            "importance": "HIGH",
            "reference": "Task 1",
            "type": "modification",
            "underlying_group_indices": [0],
            "underlying_group_ids": ["abc123def456"],
            "model_count": 1,
            "original_suggestion_count": 1,
            "is_singleton": True,
            "reasoning": "Singleton",
        }

    def test_validates_against_schema(self, tmp_path):
        """Output passes schema validation (no ValueError raised)."""
        cg = self._make_valid_cg()
        metadata = _build_valid_metadata(
            total_original=10, total_consolidated=5, skipped_report=False
        )
        path = generate_consolidated_json([cg], metadata, str(tmp_path))
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert "consolidated_groups" in data
        assert "metadata" in data

    def test_less_than_10pct_reduction_sets_skipped_report(self, tmp_path):
        """<10% reduction sets skipped_report=True and removes old reports."""
        # Create stale report files
        (tmp_path / "consolidated-report.md").write_text("old report")
        (tmp_path / "consolidated-report.html").write_text("old html")

        cg = self._make_valid_cg()
        metadata = _build_valid_metadata(
            total_original=10,
            total_consolidated=10,  # 0% reduction
            skipped_report=False,
        )
        path = generate_consolidated_json([cg], metadata, str(tmp_path))
        data = json.loads(Path(path).read_text())
        assert data["metadata"]["skipped_report"] is True
        assert not (tmp_path / "consolidated-report.md").exists()
        assert not (tmp_path / "consolidated-report.html").exists()

    def test_exactly_10pct_reduction_does_not_skip(self, tmp_path):
        """Exactly 10% reduction (boundary) should NOT trigger skip."""
        cg = self._make_valid_cg()
        metadata = _build_valid_metadata(
            total_original=10,
            total_consolidated=9,  # exactly 10% reduction
            skipped_report=False,
        )
        path = generate_consolidated_json([cg], metadata, str(tmp_path))
        data = json.loads(Path(path).read_text())
        assert data["metadata"]["skipped_report"] is False

    def test_above_10pct_reduction_no_skip(self, tmp_path):
        """Above 10% reduction does not set skipped_report."""
        cg = self._make_valid_cg()
        metadata = _build_valid_metadata(
            total_original=10,
            total_consolidated=8,  # 20% reduction
            skipped_report=False,
        )
        path = generate_consolidated_json([cg], metadata, str(tmp_path))
        data = json.loads(Path(path).read_text())
        assert data["metadata"]["skipped_report"] is False


# ===========================================================================
# 9. load_merged_suggestions
# ===========================================================================


class TestLoadMergedSuggestions:
    """Tests for load_merged_suggestions()."""

    def _write_consolidated_json(self, phase_dir, groups, metadata_overrides=None,
                                  consolidated_groups=None):
        """Helper to write a consolidated.json file for testing."""
        phase_path = Path(phase_dir)
        phase_path.mkdir(parents=True, exist_ok=True)

        g_ids = [generate_group_id(g) for g in groups]

        if consolidated_groups is None:
            # Default: each group is a singleton CG
            consolidated_groups = []
            for i, (g, gid) in enumerate(zip(groups, g_ids)):
                consolidated_groups.append({
                    "consolidated_id": gid,
                    "display_index": i + 1,
                    "title": g.get("theme", ""),
                    "description": "",
                    "importance": "MEDIUM",
                    "reference": g.get("reference", ""),
                    "type": "modification",
                    "underlying_group_indices": [i],
                    "underlying_group_ids": [gid],
                    "model_count": 1,
                    "original_suggestion_count": 1,
                    "is_singleton": True,
                    "reasoning": "test",
                })

        # Compute actual hashes
        plan_file = phase_path / "plan.md"
        plan_file.write_text("# Plan")
        plan_hash = hashlib.sha256("# Plan".encode()).hexdigest()

        grouped_json = phase_path / "grouped.json"
        grouped_content = json.dumps(groups)
        grouped_json.write_text(grouped_content)
        grouped_hash = hashlib.sha256(grouped_content.encode()).hexdigest()

        metadata = {
            "schema_version": "1.0",
            "total_original_groups": len(groups),
            "total_consolidated": len(consolidated_groups),
            "merged_count": 0,
            "singleton_count": len(consolidated_groups),
            "consolidation_ratio": 1.0,
            "timestamp": "2026-01-01T00:00:00",
            "plan_hash": plan_hash,
            "grouped_hash": grouped_hash,
            "plan_hash_algorithm": "sha256",
            "skipped_report": False,
            "sections_processed": 1,
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)

        payload = {
            "consolidated_groups": consolidated_groups,
            "metadata": metadata,
        }
        (phase_path / "consolidated.json").write_text(json.dumps(payload))
        return str(plan_file), g_ids

    def test_returns_empty_when_no_consolidated_json(self, tmp_path):
        """Returns empty sets when consolidated.json does not exist."""
        skipped, overrides = load_merged_suggestions(
            str(tmp_path), [], "nonexistent.md"
        )
        assert skipped == set()
        assert overrides == {}

    def test_returns_empty_when_skipped_report_true(self, tmp_path):
        """Returns empty when metadata.skipped_report is True."""
        groups = [_make_group()]
        plan_file, _ = self._write_consolidated_json(
            str(tmp_path), groups, {"skipped_report": True}
        )
        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, plan_file
        )
        assert skipped == set()
        assert overrides == {}

    @patch("utils.consolidation.parse_consolidated_skipped_groups", return_value=set())
    @patch("utils.consolidation.parse_consolidated_validation_overrides", return_value={})
    @patch("utils.consolidation.load_consolidated_html_selections", return_value=None)
    @patch("utils.consolidation.merge_consolidated_selections", return_value=(set(), {}))
    def test_fails_closed_on_grouped_hash_mismatch(
        self, mock_merge, mock_html, mock_overrides, mock_skipped, tmp_path
    ):
        """grouped_hash mismatch with accept_stale=False returns empty."""
        groups = [_make_group()]
        plan_file, _ = self._write_consolidated_json(
            str(tmp_path), groups, {"grouped_hash": "wrong_hash"}
        )
        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, plan_file, accept_stale=False
        )
        assert skipped == set()
        assert overrides == {}

    @patch("utils.consolidation.parse_consolidated_skipped_groups")
    @patch("utils.consolidation.parse_consolidated_validation_overrides")
    @patch("utils.consolidation.load_consolidated_html_selections", return_value=None)
    @patch("utils.consolidation.merge_consolidated_selections")
    def test_accept_stale_overrides_grouped_hash_fail(
        self, mock_merge, mock_html, mock_overrides, mock_skipped, tmp_path
    ):
        """accept_stale=True allows grouped_hash mismatch to proceed."""
        groups = [_make_group(theme="G0")]
        plan_file, g_ids = self._write_consolidated_json(
            str(tmp_path), groups, {"grouped_hash": "wrong_hash"}
        )
        # Set up mocks to return a C-level skip for the singleton CG
        mock_skipped.return_value = {g_ids[0]}
        mock_overrides.return_value = {}
        mock_merge.return_value = ({g_ids[0]}, {})

        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, plan_file, accept_stale=True
        )
        # Should have group index 0 skipped since accept_stale=True
        assert 0 in skipped

    @patch("utils.consolidation.parse_consolidated_skipped_groups", return_value=set())
    @patch("utils.consolidation.parse_consolidated_validation_overrides", return_value={})
    @patch("utils.consolidation.load_consolidated_html_selections", return_value=None)
    @patch("utils.consolidation.merge_consolidated_selections", return_value=(set(), {}))
    def test_warns_on_plan_hash_mismatch_only(
        self, mock_merge, mock_html, mock_overrides, mock_skipped, tmp_path
    ):
        """Plan hash mismatch warns but still applies (does not fail closed)."""
        groups = [_make_group()]
        plan_file, _ = self._write_consolidated_json(
            str(tmp_path), groups, {"plan_hash": "wrong_plan_hash"}
        )
        # Should not fail; returns empty because mock_merge returns empty
        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, plan_file, accept_stale=False
        )
        assert skipped == set()
        assert overrides == {}

    @patch("utils.consolidation.parse_consolidated_skipped_groups", return_value=set())
    @patch("utils.consolidation.parse_consolidated_validation_overrides", return_value={})
    @patch("utils.consolidation.load_consolidated_html_selections", return_value=None)
    @patch("utils.consolidation.merge_consolidated_selections", return_value=(set(), {}))
    def test_handles_missing_hash_fields_backward_compat(
        self, mock_merge, mock_html, mock_overrides, mock_skipped, tmp_path
    ):
        """Missing hash fields trigger warning but continue (backward compat)."""
        groups = [_make_group()]
        plan_file, _ = self._write_consolidated_json(
            str(tmp_path),
            groups,
            {"plan_hash": None, "grouped_hash": None},
        )
        # Should not raise; backward compat path
        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, plan_file
        )
        assert skipped == set()

    @patch("utils.consolidation.parse_consolidated_skipped_groups")
    @patch("utils.consolidation.parse_consolidated_validation_overrides")
    @patch("utils.consolidation.load_consolidated_html_selections", return_value=None)
    @patch("utils.consolidation.merge_consolidated_selections")
    def test_maps_c_level_skips_to_g_level_indices(
        self, mock_merge, mock_html, mock_overrides, mock_skipped, tmp_path
    ):
        """C-level skip decisions map to correct G-level (0-based) indices."""
        groups = [
            _make_group(theme="G0", reference="Task 1"),
            _make_group(theme="G1", reference="Task 1"),
        ]
        plan_file, g_ids = self._write_consolidated_json(str(tmp_path), groups)

        # Skip the CG for group 1 (whose consolidated_id == g_ids[1])
        mock_skipped.return_value = {g_ids[1]}
        mock_overrides.return_value = {}
        mock_merge.return_value = ({g_ids[1]}, {})

        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, plan_file
        )
        assert 1 in skipped
        assert 0 not in skipped

    @patch("utils.consolidation.parse_consolidated_skipped_groups")
    @patch("utils.consolidation.parse_consolidated_validation_overrides")
    @patch("utils.consolidation.load_consolidated_html_selections", return_value=None)
    @patch("utils.consolidation.merge_consolidated_selections")
    def test_maps_c_level_validation_overrides_to_g_level(
        self, mock_merge, mock_html, mock_overrides, mock_skipped, tmp_path
    ):
        """C-level validation overrides map to correct G-level indices."""
        groups = [
            _make_group(theme="G0", reference="Task 1"),
            _make_group(theme="G1", reference="Task 1"),
        ]
        plan_file, g_ids = self._write_consolidated_json(str(tmp_path), groups)

        mock_skipped.return_value = set()
        mock_overrides.return_value = {}
        mock_merge.return_value = (set(), {g_ids[0]: "invalid"})

        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, plan_file
        )
        assert overrides.get(0) == "invalid"

    @patch("utils.consolidation.parse_consolidated_skipped_groups")
    @patch("utils.consolidation.parse_consolidated_validation_overrides")
    @patch("utils.consolidation.load_consolidated_html_selections", return_value=None)
    @patch("utils.consolidation.merge_consolidated_selections")
    def test_logs_warnings_for_unresolvable_group_ids(
        self, mock_merge, mock_html, mock_overrides, mock_skipped, tmp_path, caplog
    ):
        """Unresolvable group_ids in consolidated_groups log warnings."""
        groups = [_make_group(theme="G0")]
        plan_file, g_ids = self._write_consolidated_json(str(tmp_path), groups)

        # Inject a CG with an unresolvable underlying group_id
        consolidated_path = tmp_path / "consolidated.json"
        data = json.loads(consolidated_path.read_text())
        data["consolidated_groups"].append({
            "consolidated_id": "fake_cid_12ab",
            "display_index": 2,
            "title": "Ghost",
            "description": "",
            "importance": "LOW",
            "reference": "",
            "type": "clarification",
            "underlying_group_indices": [99],
            "underlying_group_ids": ["nonexistent_id"],
            "model_count": 1,
            "original_suggestion_count": 1,
            "is_singleton": True,
            "reasoning": "",
        })
        consolidated_path.write_text(json.dumps(data))

        mock_skipped.return_value = {"fake_cid_12ab"}
        mock_overrides.return_value = {}
        mock_merge.return_value = ({"fake_cid_12ab"}, {})

        import logging
        with caplog.at_level(logging.WARNING):
            skipped, overrides = load_merged_suggestions(
                str(tmp_path), groups, plan_file
            )

        # The unresolvable CID skip should log a warning and not crash
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("could not be resolved" in m for m in warning_messages)


# ===========================================================================
# 10. Helper functions
# ===========================================================================


class TestWordOverlapRatio:
    """Tests for _word_overlap_ratio()."""

    def test_no_overlap(self):
        """Completely different words return 0."""
        assert _word_overlap_ratio("foo bar", "baz qux") == 0.0

    def test_identical_strings(self):
        """Identical strings return 1.0."""
        assert _word_overlap_ratio("hello world", "hello world") == 1.0

    def test_empty_strings(self):
        """Two empty strings return 0.0."""
        assert _word_overlap_ratio("", "") == 0.0

    def test_partial_overlap(self):
        """Partial overlap returns correct ratio."""
        # {"add", "error"} & {"add", "validation"} = {"add"}
        # Union = {"add", "error", "validation"} = 3
        # Ratio = 1/3
        ratio = _word_overlap_ratio("add error", "add validation")
        assert abs(ratio - 1 / 3) < 0.01

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert _word_overlap_ratio("Hello World", "hello world") == 1.0


class TestHighestImportance:
    """Tests for _highest_importance()."""

    def test_high_wins(self):
        assert _highest_importance("LOW", "HIGH", "MEDIUM") == "HIGH"

    def test_medium_over_low(self):
        assert _highest_importance("LOW", "MEDIUM") == "MEDIUM"

    def test_single_value(self):
        assert _highest_importance("MEDIUM") == "MEDIUM"

    def test_unrecognized_treated_as_lowest(self):
        assert _highest_importance("UNKNOWN", "LOW") == "LOW"

    def test_case_normalization(self):
        """lowercase input is normalized to uppercase."""
        assert _highest_importance("high") == "HIGH"


class TestResolveTypeForGroupIds:
    """Tests for _resolve_type_for_group_ids()."""

    def test_returns_highest_priority_type(self):
        """Returns the most actionable type across group IDs."""
        groups = [
            _make_group(theme="T0", category="clarification"),
            _make_group(theme="T1", category="addition"),
        ]
        gid_map = {}
        for i, g in enumerate(groups):
            gid_map[generate_group_id(g)] = i
        result = _resolve_type_for_group_ids(
            list(gid_map.keys()), groups, gid_map
        )
        assert result == "addition"

    def test_fallback_to_clarification(self):
        """Returns 'clarification' when no group IDs resolve."""
        assert (
            _resolve_type_for_group_ids(["nonexistent"], [], {}) == "clarification"
        )

    def test_checks_suggestion_level_types(self):
        """Suggestion-level type fields are also considered."""
        groups = [
            _make_group(
                theme="T0",
                category="clarification",
                suggestions=[{"title": "S", "type": "addition"}],
            )
        ]
        gid = generate_group_id(groups[0])
        result = _resolve_type_for_group_ids(
            [gid], groups, {gid: 0}
        )
        assert result == "addition"


class TestMakeSingletonConsolidated:
    """Tests for _make_singleton_consolidated()."""

    def test_creates_correct_structure(self):
        """Singleton entry has all expected fields."""
        group = _make_group(theme="My Theme", category="modification")
        gid = generate_group_id(group)
        result = _make_singleton_consolidated(gid, 0, group)
        assert result["title"] == "My Theme"
        assert result["is_singleton"] is True
        assert result["underlying_group_ids"] == [gid]
        assert result["underlying_group_indices"] == [0]
        assert result["type"] == "modification"
        assert "Singleton" in result["reasoning"]

    def test_custom_reasoning(self):
        """Custom reasoning is used when provided."""
        group = _make_group()
        gid = generate_group_id(group)
        result = _make_singleton_consolidated(
            gid, 0, group, reasoning="Custom reason"
        )
        assert result["reasoning"] == "Custom reason"

    def test_unknown_type_falls_back_to_clarification(self):
        """Unknown type/category falls back to 'clarification'."""
        group = _make_group(category="unknown_type")
        group.pop("type", None)
        gid = generate_group_id(group)
        result = _make_singleton_consolidated(gid, 0, group)
        assert result["type"] == "clarification"

    def test_importance_normalized(self):
        """Importance is uppercased and validated."""
        group = _make_group(importance="high")
        gid = generate_group_id(group)
        result = _make_singleton_consolidated(gid, 0, group)
        assert result["importance"] == "HIGH"

    def test_model_count_from_models_list(self):
        """model_count derived from group's models list."""
        group = _make_group(models=["model-a", "model-b", "model-c"])
        gid = generate_group_id(group)
        result = _make_singleton_consolidated(gid, 0, group)
        assert result["model_count"] == 3
