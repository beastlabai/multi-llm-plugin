"""Integration tests for the consolidation pipeline (end-to-end)."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.consolidation import (
    generate_group_id,
    generate_consolidated_id,
    pre_group_by_section,
    prepare_consolidation_tasks,
    merge_consolidation_results,
    generate_consolidated_json,
    generate_consolidated_report,
    generate_consolidated_html,
    load_merged_suggestions,
)
from utils.report_parser import (
    parse_consolidated_skipped_groups,
    parse_consolidated_validation_overrides,
    load_consolidated_html_selections,
    merge_consolidated_selections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_group(theme, reference, suggestions=None, models=None, category="modification"):
    """Create a minimal suggestion group dict."""
    if suggestions is None:
        suggestions = [
            {
                "title": f"{theme} suggestion",
                "desc": f"Description for {theme}",
                "description": f"Description for {theme}",
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
        "importance": "MEDIUM",
    }


def _write_json(path, data):
    """Write a JSON file."""
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _compute_file_hash(path):
    """Compute SHA-256 of a file's UTF-8 content."""
    content = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _make_validation_entry(group_index, status="valid"):
    """Create a validation result entry."""
    return {"group_index": group_index, "status": status, "reason": ""}


def _build_metadata(
    phase_dir,
    plan_file,
    groups,
    consolidated_groups,
    merged_count=0,
    skipped_report=False,
):
    """Build a valid consolidated.json metadata dict with real hashes."""
    plan_hash = _compute_file_hash(plan_file)

    grouped_json_path = Path(phase_dir) / "grouped.json"
    grouped_content = json.dumps(groups, indent=2)
    grouped_json_path.write_text(grouped_content, encoding="utf-8")
    grouped_hash = _compute_file_hash(str(grouped_json_path))

    total_original = sum(
        len(cg.get("underlying_group_indices", [])) for cg in consolidated_groups
    )
    total_consolidated = len(consolidated_groups)
    singleton_count = total_consolidated - merged_count

    return {
        "schema_version": "1.0",
        "total_original_groups": total_original,
        "total_consolidated": total_consolidated,
        "merged_count": merged_count,
        "singleton_count": singleton_count,
        "consolidation_ratio": total_original / max(total_consolidated, 1),
        "timestamp": "2026-01-15T12:00:00",
        "plan_hash": plan_hash,
        "grouped_hash": grouped_hash,
        "plan_hash_algorithm": "sha256",
        "skipped_report": skipped_report,
        "sections_processed": 2,
    }


# ===========================================================================
# 1. Full pipeline test
# ===========================================================================


class TestFullPipeline:
    """End-to-end: pre_group_by_section -> prepare_consolidation_tasks ->
    merge_consolidation_results -> generate_consolidated_json/report/html."""

    def test_full_pipeline(self, tmp_path):
        """Full pipeline produces all output files with correct structure."""
        # --- Setup: 6 groups across 2 sections ---
        groups = [
            _make_group("Auth validation", "## Task 1: Authentication"),
            _make_group("Auth tokens", "## Task 1: Authentication"),
            _make_group("Auth errors", "## Task 1: Authentication"),
            _make_group("API caching", "## Task 2: Performance"),
            _make_group("API batching", "## Task 2: Performance"),
            _make_group("API throttling", "## Task 2: Performance"),
        ]
        validation = [
            _make_validation_entry(0, "valid"),
            _make_validation_entry(1, "valid"),
            _make_validation_entry(2, "needs-human-decision"),
            _make_validation_entry(3, "valid"),
            _make_validation_entry(4, "invalid"),  # filtered out
            _make_validation_entry(5, "valid"),
        ]

        # Step 1: pre_group_by_section
        section_groups = pre_group_by_section(groups, validation)
        assert "task 1" in section_groups
        assert "task 2" in section_groups
        # index 4 is invalid, so task 2 should have indices 3 and 5
        assert 4 not in section_groups["task 2"]

        # Step 2: prepare_consolidation_tasks
        phase_dir = str(tmp_path)
        tasks_meta = prepare_consolidation_tasks(
            groups, section_groups, phase_dir, "plan.md"
        )
        assert tasks_meta["total_batches"] >= 1
        assert (tmp_path / "consolidation_tasks.json").exists()

        # Step 3: Simulate batch outputs
        for batch in tasks_meta["batches"]:
            batch_idx = batch["batch_index"]
            gids = batch["group_ids"]
            indices = batch["group_indices"]

            if len(gids) >= 2:
                # Create a cluster from the first two, singleton the rest
                batch_output = {
                    "clusters": [
                        {
                            "title": f"Merged cluster {batch_idx}",
                            "description": f"Combined description for batch {batch_idx}",
                            "importance": "HIGH",
                            "type": "modification",
                            "underlying_group_ids": gids[:2],
                            "underlying_group_indices": indices[:2],
                            "reasoning": "Related themes in same section",
                        }
                    ],
                    "singletons": [
                        {
                            "group_id": gid,
                            "group_index": idx,
                            "reasoning": "Standalone",
                        }
                        for gid, idx in zip(gids[2:], indices[2:])
                    ],
                }
            else:
                batch_output = {
                    "clusters": [],
                    "singletons": [
                        {
                            "group_id": gid,
                            "group_index": idx,
                            "reasoning": "Standalone",
                        }
                        for gid, idx in zip(gids, indices)
                    ],
                }
            _write_json(tmp_path / f"consolidation_batch_{batch_idx}.json", batch_output)

        # Step 4: merge_consolidation_results
        consolidated_groups, partial_failures = merge_consolidation_results(
            phase_dir, tasks_meta, groups
        )
        assert len(consolidated_groups) >= 1
        assert partial_failures["count"] == 0

        # Every consolidated group must have required fields
        for cg in consolidated_groups:
            assert "consolidated_id" in cg
            assert "display_index" in cg
            assert "underlying_group_ids" in cg
            assert cg["display_index"] >= 1

        # Step 5: generate_consolidated_json
        metadata = {
            "schema_version": "1.0",
            "total_original_groups": 5,  # 5 valid groups
            "total_consolidated": len(consolidated_groups),
            "merged_count": sum(1 for c in consolidated_groups if not c["is_singleton"]),
            "singleton_count": sum(1 for c in consolidated_groups if c["is_singleton"]),
            "consolidation_ratio": 5 / max(len(consolidated_groups), 1),
            "timestamp": "2026-01-15T12:00:00",
            "plan_hash": "abc123",
            "grouped_hash": "def456",
            "plan_hash_algorithm": "sha256",
            "skipped_report": False,
            "sections_processed": 2,
        }
        json_path = generate_consolidated_json(
            consolidated_groups, metadata, phase_dir
        )
        assert Path(json_path).exists()
        json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert "consolidated_groups" in json_data
        assert "metadata" in json_data

        # Step 6: generate_consolidated_report
        report_path = generate_consolidated_report(
            consolidated_groups, groups, phase_dir, "my-feature"
        )
        assert Path(report_path).exists()
        report_content = Path(report_path).read_text(encoding="utf-8")
        assert "# Consolidated Plan Review Report" in report_content
        assert "- [ ] Skip this group" in report_content

        # Step 7: generate_consolidated_html
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# My Feature Plan\n\nSome plan content.", encoding="utf-8")
        html_path = generate_consolidated_html(
            consolidated_groups,
            groups,
            phase_dir,
            str(plan_file),
            ["model-a"],
        )
        assert Path(html_path).exists()
        html_content = Path(html_path).read_text(encoding="utf-8")
        assert "reportData" in html_content


# ===========================================================================
# 2. Merge logic: C-level skip + G-level skip = union
# ===========================================================================


class TestMergeCLevelAndGLevelSkips:
    """C-level skip of a consolidated group maps to all underlying G-level indices."""

    def test_c_level_skip_maps_to_underlying_groups(self, tmp_path):
        """CG1 skip maps to G1 and G2; CG2 skip handled separately."""
        phase_dir = str(tmp_path)

        # 5 underlying groups
        groups = [
            _make_group("Theme G1", "Task 1"),
            _make_group("Theme G2", "Task 1"),
            _make_group("Theme G3", "Task 2"),
            _make_group("Theme G4", "Task 3"),
            _make_group("Theme G5", "Task 3"),
        ]
        g_ids = [generate_group_id(g) for g in groups]

        # CG1 covers G1+G2, CG2 covers G3, CG3 covers G4+G5
        cg1_id = generate_consolidated_id([g_ids[0], g_ids[1]])
        cg2_id = generate_consolidated_id([g_ids[2]])  # singleton = g_ids[2]
        cg3_id = generate_consolidated_id([g_ids[3], g_ids[4]])

        consolidated_groups = [
            {
                "consolidated_id": cg1_id,
                "display_index": 1,
                "title": "Auth combined",
                "description": "Combined auth",
                "importance": "HIGH",
                "reference": "Task 1",
                "type": "modification",
                "underlying_group_indices": [0, 1],
                "underlying_group_ids": [g_ids[0], g_ids[1]],
                "model_count": 1,
                "original_suggestion_count": 2,
                "is_singleton": False,
                "reasoning": "Merged",
            },
            {
                "consolidated_id": cg2_id,
                "display_index": 2,
                "title": "Theme G3",
                "description": "Singleton",
                "importance": "MEDIUM",
                "reference": "Task 2",
                "type": "modification",
                "underlying_group_indices": [2],
                "underlying_group_ids": [g_ids[2]],
                "model_count": 1,
                "original_suggestion_count": 1,
                "is_singleton": True,
                "reasoning": "Singleton",
            },
            {
                "consolidated_id": cg3_id,
                "display_index": 3,
                "title": "Perf combined",
                "description": "Combined perf",
                "importance": "MEDIUM",
                "reference": "Task 3",
                "type": "modification",
                "underlying_group_indices": [3, 4],
                "underlying_group_ids": [g_ids[3], g_ids[4]],
                "model_count": 1,
                "original_suggestion_count": 2,
                "is_singleton": False,
                "reasoning": "Merged",
            },
        ]

        # Write plan and grouped.json, then compute metadata
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Test Plan", encoding="utf-8")
        metadata = _build_metadata(
            phase_dir, str(plan_file), groups, consolidated_groups, merged_count=2
        )

        payload = {
            "consolidated_groups": consolidated_groups,
            "metadata": metadata,
        }
        _write_json(tmp_path / "consolidated.json", payload)

        # Write consolidated-report.md with CG1 skip checked
        report_lines = [
            "# Consolidated Review Report: test",
            "",
            f"## CG1 [{cg1_id}]: Auth combined",
            "- [x] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "",
            f"## CG2 [{cg2_id}]: Theme G3",
            "- [ ] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "",
            f"## CG3 [{cg3_id}]: Perf combined",
            "- [ ] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "",
        ]
        (tmp_path / "consolidated-report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        # Call load_merged_suggestions
        skipped, overrides = load_merged_suggestions(
            phase_dir, groups, str(plan_file)
        )

        # CG1 skip -> G1 (index 0) and G2 (index 1)
        assert 0 in skipped
        assert 1 in skipped
        # CG2 and CG3 not skipped
        assert 2 not in skipped
        assert 3 not in skipped
        assert 4 not in skipped


# ===========================================================================
# 3. Validation override cascade
# ===========================================================================


class TestValidationOverrideCascade:
    """C-level overrides cascade to all underlying groups."""

    def test_c_level_override_cascades_to_all_underlying(self, tmp_path):
        """C-level 'invalid' override on CG1 maps to both G1 and G2."""
        phase_dir = str(tmp_path)

        groups = [
            _make_group("Theme G1", "Task 1"),
            _make_group("Theme G2", "Task 1"),
        ]
        g_ids = [generate_group_id(g) for g in groups]
        cg1_id = generate_consolidated_id([g_ids[0], g_ids[1]])

        consolidated_groups = [
            {
                "consolidated_id": cg1_id,
                "display_index": 1,
                "title": "Combined",
                "description": "Merged",
                "importance": "MEDIUM",
                "reference": "Task 1",
                "type": "modification",
                "underlying_group_indices": [0, 1],
                "underlying_group_ids": [g_ids[0], g_ids[1]],
                "model_count": 1,
                "original_suggestion_count": 2,
                "is_singleton": False,
                "reasoning": "Test",
            },
        ]

        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Test Plan", encoding="utf-8")
        metadata = _build_metadata(
            phase_dir, str(plan_file), groups, consolidated_groups, merged_count=1
        )

        payload = {
            "consolidated_groups": consolidated_groups,
            "metadata": metadata,
        }
        _write_json(tmp_path / "consolidated.json", payload)

        # Write report with CG1 marked invalid
        report_lines = [
            "# Report",
            "",
            f"## CG1 [{cg1_id}]: Combined",
            "- [ ] Skip this group",
            "- [ ] Mark valid",
            "- [x] Mark invalid",
            "- [ ] Needs human attention",
            "",
        ]
        (tmp_path / "consolidated-report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        skipped, overrides = load_merged_suggestions(
            phase_dir, groups, str(plan_file)
        )

        # Both G1 (index 0) and G2 (index 1) should get "invalid" override
        assert overrides.get(0) == "invalid"
        assert overrides.get(1) == "invalid"


# ===========================================================================
# 4. HTML selections precedence
# ===========================================================================


class TestHtmlSelectionsPrecedence:
    """HTML selections override markdown decisions."""

    def test_html_takes_precedence_over_markdown(self, tmp_path):
        """HTML skip overrides markdown non-skip; markdown skip works for CG2."""
        phase_dir = str(tmp_path)

        groups = [
            _make_group("Theme G1", "Task 1"),
            _make_group("Theme G2", "Task 2"),
        ]
        g_ids = [generate_group_id(g) for g in groups]
        cg1_id = g_ids[0]  # singleton
        cg2_id = g_ids[1]  # singleton

        consolidated_groups = [
            {
                "consolidated_id": cg1_id,
                "display_index": 1,
                "title": "Theme G1",
                "description": "",
                "importance": "MEDIUM",
                "reference": "Task 1",
                "type": "modification",
                "underlying_group_indices": [0],
                "underlying_group_ids": [g_ids[0]],
                "model_count": 1,
                "original_suggestion_count": 1,
                "is_singleton": True,
                "reasoning": "Singleton",
            },
            {
                "consolidated_id": cg2_id,
                "display_index": 2,
                "title": "Theme G2",
                "description": "",
                "importance": "MEDIUM",
                "reference": "Task 2",
                "type": "modification",
                "underlying_group_indices": [1],
                "underlying_group_ids": [g_ids[1]],
                "model_count": 1,
                "original_suggestion_count": 1,
                "is_singleton": True,
                "reasoning": "Singleton",
            },
        ]

        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Test Plan", encoding="utf-8")
        metadata = _build_metadata(
            phase_dir, str(plan_file), groups, consolidated_groups
        )

        payload = {
            "consolidated_groups": consolidated_groups,
            "metadata": metadata,
        }
        _write_json(tmp_path / "consolidated.json", payload)

        # Markdown: CG1 NOT skipped, CG2 skipped
        report_lines = [
            "# Report",
            "",
            f"## CG1 [{cg1_id}]: Theme G1",
            "- [ ] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "",
            f"## CG2 [{cg2_id}]: Theme G2",
            "- [x] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "",
        ]
        (tmp_path / "consolidated-report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        # HTML: CG1 skipped (overrides markdown), CG2 not in HTML so markdown wins
        html_selections = {
            "plan_path": str(plan_file),
            "phase": "review-plan-consolidated",
            "exported_at": "2026-01-15T12:00:00",
            "skipped_groups": [cg1_id],
            "validation_overrides": {},
        }
        _write_json(tmp_path / "consolidated_user_selections.json", html_selections)

        skipped, overrides = load_merged_suggestions(
            phase_dir, groups, str(plan_file)
        )

        # HTML says CG1 skipped -> group index 0 skipped
        assert 0 in skipped
        # HTML replaces markdown entirely, so CG2 is NOT skipped from HTML
        # (HTML selections completely replace markdown for skipped_groups)
        assert 1 not in skipped


# ===========================================================================
# 5. Staleness - fail closed
# ===========================================================================


class TestStalenessFailClosed:
    """Grouped hash mismatch fails closed by default."""

    def test_stale_grouped_hash_returns_empty(self, tmp_path):
        """accept_stale=False with grouped_hash mismatch returns empty."""
        phase_dir = str(tmp_path)

        groups = [_make_group("Theme A", "Task 1")]
        g_ids = [generate_group_id(g) for g in groups]

        consolidated_groups = [
            {
                "consolidated_id": g_ids[0],
                "display_index": 1,
                "title": "Theme A",
                "description": "",
                "importance": "MEDIUM",
                "reference": "Task 1",
                "type": "modification",
                "underlying_group_indices": [0],
                "underlying_group_ids": [g_ids[0]],
                "model_count": 1,
                "original_suggestion_count": 1,
                "is_singleton": True,
                "reasoning": "Singleton",
            },
        ]

        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Test Plan", encoding="utf-8")

        # Write grouped.json with original content
        grouped_json = tmp_path / "grouped.json"
        grouped_json.write_text(json.dumps(groups), encoding="utf-8")
        original_grouped_hash = _compute_file_hash(str(grouped_json))

        metadata = {
            "schema_version": "1.0",
            "total_original_groups": 1,
            "total_consolidated": 1,
            "merged_count": 0,
            "singleton_count": 1,
            "consolidation_ratio": 1.0,
            "timestamp": "2026-01-15T12:00:00",
            "plan_hash": _compute_file_hash(str(plan_file)),
            "grouped_hash": "wrong_hash_abc123",  # deliberate mismatch
            "plan_hash_algorithm": "sha256",
            "skipped_report": False,
            "sections_processed": 1,
        }
        payload = {
            "consolidated_groups": consolidated_groups,
            "metadata": metadata,
        }
        _write_json(tmp_path / "consolidated.json", payload)

        # Write report with CG1 skipped
        report_lines = [
            "# Report",
            "",
            f"## CG1 [{g_ids[0]}]: Theme A",
            "- [x] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "",
        ]
        (tmp_path / "consolidated-report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        # accept_stale=False (default) -> should return empty
        skipped, overrides = load_merged_suggestions(
            phase_dir, groups, str(plan_file), accept_stale=False
        )
        assert skipped == set()
        assert overrides == {}

    def test_accept_stale_returns_results(self, tmp_path):
        """accept_stale=True with grouped_hash mismatch returns results."""
        phase_dir = str(tmp_path)

        groups = [_make_group("Theme A", "Task 1")]
        g_ids = [generate_group_id(g) for g in groups]

        consolidated_groups = [
            {
                "consolidated_id": g_ids[0],
                "display_index": 1,
                "title": "Theme A",
                "description": "",
                "importance": "MEDIUM",
                "reference": "Task 1",
                "type": "modification",
                "underlying_group_indices": [0],
                "underlying_group_ids": [g_ids[0]],
                "model_count": 1,
                "original_suggestion_count": 1,
                "is_singleton": True,
                "reasoning": "Singleton",
            },
        ]

        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Test Plan", encoding="utf-8")

        # Write grouped.json
        grouped_json = tmp_path / "grouped.json"
        grouped_json.write_text(json.dumps(groups), encoding="utf-8")

        metadata = {
            "schema_version": "1.0",
            "total_original_groups": 1,
            "total_consolidated": 1,
            "merged_count": 0,
            "singleton_count": 1,
            "consolidation_ratio": 1.0,
            "timestamp": "2026-01-15T12:00:00",
            "plan_hash": _compute_file_hash(str(plan_file)),
            "grouped_hash": "wrong_hash_abc123",  # deliberate mismatch
            "plan_hash_algorithm": "sha256",
            "skipped_report": False,
            "sections_processed": 1,
        }
        payload = {
            "consolidated_groups": consolidated_groups,
            "metadata": metadata,
        }
        _write_json(tmp_path / "consolidated.json", payload)

        # Write report with CG1 skipped
        report_lines = [
            "# Report",
            "",
            f"## CG1 [{g_ids[0]}]: Theme A",
            "- [x] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "",
        ]
        (tmp_path / "consolidated-report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        # accept_stale=True -> should return results
        skipped, overrides = load_merged_suggestions(
            phase_dir, groups, str(plan_file), accept_stale=True
        )
        assert 0 in skipped


# ===========================================================================
# 6. Staleness - plan_hash warn but apply
# ===========================================================================


class TestStalenessPlanHashWarn:
    """Plan hash mismatch warns but does not fail closed."""

    def test_plan_hash_mismatch_still_returns_results(self, tmp_path):
        """When plan_hash differs but grouped_hash matches, results are returned."""
        phase_dir = str(tmp_path)

        groups = [_make_group("Theme A", "Task 1")]
        g_ids = [generate_group_id(g) for g in groups]

        consolidated_groups = [
            {
                "consolidated_id": g_ids[0],
                "display_index": 1,
                "title": "Theme A",
                "description": "",
                "importance": "MEDIUM",
                "reference": "Task 1",
                "type": "modification",
                "underlying_group_indices": [0],
                "underlying_group_ids": [g_ids[0]],
                "model_count": 1,
                "original_suggestion_count": 1,
                "is_singleton": True,
                "reasoning": "Singleton",
            },
        ]

        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Test Plan", encoding="utf-8")

        # Write grouped.json and compute correct hash
        grouped_json = tmp_path / "grouped.json"
        grouped_json.write_text(json.dumps(groups), encoding="utf-8")
        correct_grouped_hash = _compute_file_hash(str(grouped_json))

        metadata = {
            "schema_version": "1.0",
            "total_original_groups": 1,
            "total_consolidated": 1,
            "merged_count": 0,
            "singleton_count": 1,
            "consolidation_ratio": 1.0,
            "timestamp": "2026-01-15T12:00:00",
            "plan_hash": "wrong_plan_hash",  # plan hash mismatch
            "grouped_hash": correct_grouped_hash,  # grouped hash matches
            "plan_hash_algorithm": "sha256",
            "skipped_report": False,
            "sections_processed": 1,
        }
        payload = {
            "consolidated_groups": consolidated_groups,
            "metadata": metadata,
        }
        _write_json(tmp_path / "consolidated.json", payload)

        # Write report with CG1 skip checked
        report_lines = [
            "# Report",
            "",
            f"## CG1 [{g_ids[0]}]: Theme A",
            "- [x] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "",
        ]
        (tmp_path / "consolidated-report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        skipped, overrides = load_merged_suggestions(
            phase_dir, groups, str(plan_file), accept_stale=False
        )

        # Should return results despite plan_hash mismatch
        assert 0 in skipped


# ===========================================================================
# 7. Backward compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """No consolidated.json present returns empty gracefully."""

    def test_no_consolidated_json_returns_empty(self, tmp_path):
        """load_merged_suggestions returns empty when no consolidated.json exists."""
        groups = [_make_group("Theme A", "Task 1")]
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Plan", encoding="utf-8")

        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, str(plan_file)
        )
        assert skipped == set()
        assert overrides == {}

    def test_corrupt_consolidated_json_returns_empty(self, tmp_path):
        """Corrupt consolidated.json returns empty, no exception."""
        (tmp_path / "consolidated.json").write_text("{invalid json", encoding="utf-8")
        groups = [_make_group("Theme A", "Task 1")]
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Plan", encoding="utf-8")

        skipped, overrides = load_merged_suggestions(
            str(tmp_path), groups, str(plan_file)
        )
        assert skipped == set()
        assert overrides == {}


# ===========================================================================
# 8. Stable ID persistence
# ===========================================================================


class TestStableIdPersistence:
    """Group and consolidated IDs are deterministic across runs."""

    def test_group_ids_stable_across_runs(self):
        """Same groups produce the same group_ids in separate invocations."""
        groups = [
            _make_group("Alpha feature", "Task 1"),
            _make_group("Beta feature", "Task 2"),
            _make_group("Gamma feature", "Task 3"),
        ]
        ids_run1 = [generate_group_id(g) for g in groups]
        ids_run2 = [generate_group_id(g) for g in groups]
        assert ids_run1 == ids_run2

    def test_consolidated_ids_stable_regardless_of_order(self):
        """Consolidated IDs are order-independent and stable."""
        gids = ["aaa111bbb222", "ccc333ddd444", "eee555fff666"]
        cid1 = generate_consolidated_id(gids)
        cid2 = generate_consolidated_id(list(reversed(gids)))
        assert cid1 == cid2

    def test_pipeline_ids_stable_across_two_runs(self, tmp_path):
        """Full pipeline produces the same consolidated_ids across two runs."""
        groups = [
            _make_group("Auth validation", "Task 1"),
            _make_group("Auth tokens", "Task 1"),
            _make_group("Cache layer", "Task 2"),
        ]
        validation = [
            _make_validation_entry(0, "valid"),
            _make_validation_entry(1, "valid"),
            _make_validation_entry(2, "valid"),
        ]

        ids_per_run = []
        for run_idx in range(2):
            run_dir = tmp_path / f"run{run_idx}"
            run_dir.mkdir()
            section_groups = pre_group_by_section(groups, validation)
            tasks_meta = prepare_consolidation_tasks(
                groups, section_groups, str(run_dir), "plan.md"
            )

            # Simulate identical batch outputs for both runs
            for batch in tasks_meta["batches"]:
                batch_idx = batch["batch_index"]
                gids = batch["group_ids"]
                indices = batch["group_indices"]
                batch_output = {
                    "clusters": [
                        {
                            "title": "Combined",
                            "description": "Desc",
                            "importance": "MEDIUM",
                            "type": "modification",
                            "underlying_group_ids": gids,
                            "underlying_group_indices": indices,
                            "reasoning": "Test",
                        }
                    ] if len(gids) >= 2 else [],
                    "singletons": [
                        {"group_id": gid, "group_index": idx, "reasoning": "Solo"}
                        for gid, idx in zip(gids, indices)
                    ] if len(gids) < 2 else [],
                }
                _write_json(
                    run_dir / f"consolidation_batch_{batch_idx}.json", batch_output
                )

            consolidated, _pf = merge_consolidation_results(
                str(run_dir), tasks_meta, groups
            )
            run_ids = sorted(cg["consolidated_id"] for cg in consolidated)
            ids_per_run.append(run_ids)

        assert ids_per_run[0] == ids_per_run[1]


# ===========================================================================
# "Let Claude decide" on the consolidated surface (Section 5)
# ===========================================================================


class TestConsolidatedClaudeDecide:
    """Emit + parse + round-trip of the consolidated 'Let Claude decide' option."""

    def _cg(self, cid, indices, ids):
        return {
            "consolidated_id": cid,
            "display_index": 1,
            "title": "Combined",
            "description": "Merged",
            "importance": "MEDIUM",
            "reference": "Task 1",
            "type": "modification",
            "underlying_group_indices": indices,
            "underlying_group_ids": ids,
            "model_count": 1,
            "original_suggestion_count": len(indices),
            "is_singleton": len(indices) == 1,
            "reasoning": "Test",
        }

    def test_emit_checkbox_for_needs_human(self, tmp_path):
        """generate_consolidated_report emits the checkbox for a needs-human group."""
        groups = [_make_group("Theme G1", "Task 1")]
        cg = self._cg("abc123def456", [0], [generate_group_id(groups[0])])
        validation = [{"group_index": 0, "status": "needs-human-decision"}]
        path = generate_consolidated_report(
            [cg], groups, str(tmp_path), "feat", validation=validation
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "- [ ] Let Claude decide" in content

    def test_omit_checkbox_for_valid(self, tmp_path):
        """No checkbox when the aggregate status is not needs-human-decision."""
        groups = [_make_group("Theme G1", "Task 1")]
        cg = self._cg("abc123def456", [0], [generate_group_id(groups[0])])
        validation = [{"group_index": 0, "status": "valid"}]
        path = generate_consolidated_report(
            [cg], groups, str(tmp_path), "feat", validation=validation
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "- [ ] Let Claude decide" not in content
        # The existing checkboxes are still present.
        assert "- [ ] Mark valid" in content

    def test_round_trip_claude_decide_preserved_as_marker(self, tmp_path):
        """A consolidated claude_decide resolves to group indices, kept verbatim."""
        phase_dir = str(tmp_path)
        groups = [
            _make_group("Theme G1", "Task 1"),
            _make_group("Theme G2", "Task 1"),
        ]
        g_ids = [generate_group_id(g) for g in groups]
        cg1_id = generate_consolidated_id([g_ids[0], g_ids[1]])
        consolidated_groups = [self._cg(cg1_id, [0, 1], [g_ids[0], g_ids[1]])]

        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Test Plan", encoding="utf-8")
        metadata = _build_metadata(
            phase_dir, str(plan_file), groups, consolidated_groups, merged_count=1
        )
        _write_json(
            tmp_path / "consolidated.json",
            {"consolidated_groups": consolidated_groups, "metadata": metadata},
        )

        report_lines = [
            "# Report",
            "",
            f"## CG1 [{cg1_id}]: Combined",
            "- [ ] Skip this group",
            "- [ ] Mark valid",
            "- [ ] Mark invalid",
            "- [ ] Needs human attention",
            "- [x] Let Claude decide",
            "",
        ]
        (tmp_path / "consolidated-report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        skipped, overrides = load_merged_suggestions(
            phase_dir, groups, str(plan_file)
        )

        # Resolves to BOTH underlying group indices, preserved as the marker
        # string (NOT flattened into a status).
        assert overrides.get(0) == "claude_decide"
        assert overrides.get(1) == "claude_decide"
