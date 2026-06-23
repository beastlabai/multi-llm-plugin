"""Integration tests for priority sorting across HTML report generation and markdown output.

Tests verify that sorting is correctly applied end-to-end:
- HTML report embeds sorted groups and sortConfig in the reportData JSON
- Markdown report from aggregate_results() orders sections by priority
- sortConfig values match Python constants exactly
- originalIndex stability through the full pipeline
"""

import json
import re
import pytest
from pathlib import Path

from utils.html_report_generator import (
    generate_html_report,
    VALIDATION_ORDER,
    IMPORTANCE_ORDER,
    build_sort_config,
)
from utils.review_orchestrator_base import aggregate_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_report_data(html_content: str) -> dict:
    """Extract the embedded reportData JSON from HTML content.

    Uses brace counting to properly extract the complete JSON object,
    handling nested structures and strings with special characters.
    """
    start_marker = "const reportData = "
    start_idx = html_content.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find reportData in HTML")

    json_start = start_idx + len(start_marker)

    depth = 0
    in_string = False
    escape_next = False
    end_idx = json_start

    for i, char in enumerate(html_content[json_start:], json_start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    json_str = html_content[json_start:end_idx]
    return json.loads(json_str)


def _make_groups_with_mixed_priority():
    """Return a list of groups with varied validation_status and importance."""
    return [
        {
            "theme": "Low priority group",
            "category": "style",
            "models": ["model-a"],
            "validation_status": "invalid",
            "suggestions": [
                {"title": "Trivial", "desc": "Trivial issue", "importance": "LOW", "source_model": "model-a"},
            ],
        },
        {
            "theme": "High priority group",
            "category": "bug",
            "models": ["model-b"],
            "validation_status": "needs-human-decision",
            "suggestions": [
                {"title": "Critical", "desc": "Critical issue", "importance": "HIGH", "source_model": "model-b"},
            ],
        },
        {
            "theme": "Medium priority group",
            "category": "improvement",
            "models": ["model-a"],
            "validation_status": "valid",
            "suggestions": [
                {"title": "Important", "desc": "Important issue", "importance": "MEDIUM", "source_model": "model-a"},
            ],
        },
        {
            "theme": "Valid HIGH group",
            "category": "security",
            "models": ["model-b"],
            "validation_status": "valid",
            "suggestions": [
                {"title": "Security fix", "desc": "Security fix needed", "importance": "HIGH", "source_model": "model-b"},
            ],
        },
        {
            "theme": "Pending group",
            "category": "performance",
            "models": ["model-a"],
            "validation_status": "pending",
            "suggestions": [
                {"title": "Pending item", "desc": "Not yet validated", "importance": "MEDIUM", "source_model": "model-a"},
            ],
        },
    ]


# ============================================================================
# HTML Report Sorting Integration
# ============================================================================


class TestHtmlReportSorting:
    """Verify that generate_html_report() applies sorting and embeds sortConfig."""

    @pytest.fixture
    def plan_path(self, tmp_path):
        """Create a minimal plan file."""
        p = tmp_path / "test-plan.md"
        p.write_text("# Test Plan\n\nSome content here.\n")
        return p

    @pytest.fixture
    def phase_dir(self, tmp_path):
        """Create a phase directory."""
        d = tmp_path / "review-plan"
        d.mkdir()
        return d

    def test_groups_sorted_by_priority_in_html(self, plan_path, phase_dir):
        """Groups embedded in HTML reportData are sorted by priority."""
        groups = _make_groups_with_mixed_priority()

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["model-a", "model-b"],
        )

        data = extract_report_data(html)
        report_groups = data["groups"]

        # Expected order: needs-human-decision(0), valid+HIGH(1), valid+MEDIUM(1),
        #                 invalid(3), pending(4)
        statuses = [g["validationStatus"] for g in report_groups]
        assert statuses[0] == "needs-human-decision"
        # The two valid groups should come next
        assert statuses[1] == "valid"
        assert statuses[2] == "valid"
        # valid+HIGH before valid+MEDIUM
        assert report_groups[1]["maxImportance"] == "HIGH"
        assert report_groups[2]["maxImportance"] == "MEDIUM"
        # invalid before pending
        assert statuses[3] == "invalid"
        assert statuses[4] == "pending"

    def test_sort_config_present_in_html(self, plan_path, phase_dir):
        """sortConfig is embedded in the HTML report data."""
        groups = [
            {
                "theme": "Simple group",
                "category": "test",
                "models": ["model-a"],
                "suggestions": [
                    {"title": "Test", "desc": "Desc", "importance": "HIGH", "source_model": "model-a"},
                ],
            },
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["model-a"],
        )

        data = extract_report_data(html)
        assert "sortConfig" in data

    def test_original_index_stability_in_html(self, plan_path, phase_dir):
        """originalIndex in HTML report reflects pre-sort position, not post-sort position."""
        groups = _make_groups_with_mixed_priority()

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["model-a", "model-b"],
        )

        data = extract_report_data(html)
        report_groups = data["groups"]

        # The "needs-human-decision" group was at input index 1 (0-based),
        # so its originalIndex should be 2 (1-based) -- it was the 2nd group in input.
        nhd_group = next(
            g for g in report_groups if g["validationStatus"] == "needs-human-decision"
        )
        assert nhd_group["originalIndex"] == 2

        # The "invalid" group was at input index 0 (0-based),
        # so originalIndex should be 1
        invalid_group = next(
            g for g in report_groups if g["validationStatus"] == "invalid"
        )
        assert invalid_group["originalIndex"] == 1

        # The "pending" group was at input index 4 (0-based),
        # so originalIndex should be 5
        pending_group = next(
            g for g in report_groups if g["validationStatus"] == "pending"
        )
        assert pending_group["originalIndex"] == 5


# ============================================================================
# Markdown Report Ordering Integration
# ============================================================================


class TestMarkdownReportOrdering:
    """Verify that aggregate_results() orders markdown sections by priority."""

    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """Set up directory structure needed by aggregate_results."""
        out_dir = tmp_path / "plans"
        out_dir.mkdir()
        phase_dir = tmp_path / "plans" / "test-plan" / "review-plan"
        phase_dir.mkdir(parents=True)

        plan_path = tmp_path / "plans" / "test-plan.md"
        plan_path.write_text("# Test Plan\n\nContent.\n")

        return {
            "out_dir": str(out_dir),
            "phase_dir": str(phase_dir),
            "plan_path": str(plan_path),
        }

    def test_markdown_sections_in_priority_order(self, setup_dirs):
        """Markdown report sections should appear in priority order."""
        groups = _make_groups_with_mixed_priority()

        report_path = aggregate_results(
            prefix="test-plan",
            out_dir=setup_dirs["out_dir"],
            phase_dir=setup_dirs["phase_dir"],
            phase_name="review-plan",
            models=["model-a", "model-b"],
            failed_models={},
            validated_groups=groups,
            plan_path=setup_dirs["plan_path"],
        )

        report_content = Path(report_path).read_text(encoding="utf-8")

        # Extract group headers with their themes
        # Pattern: ## G<n>: <theme>
        group_headers = re.findall(r"## G\d+: (.+)", report_content)

        assert len(group_headers) == 5

        # needs-human-decision group should be first
        assert group_headers[0] == "High priority group"

        # The two valid groups next (HIGH before MEDIUM)
        assert group_headers[1] == "Valid HIGH group"
        assert group_headers[2] == "Medium priority group"

        # invalid group
        assert group_headers[3] == "Low priority group"

        # pending group last
        assert group_headers[4] == "Pending group"


# ============================================================================
# sortConfig Parity
# ============================================================================


class TestSortConfigParity:
    """Verify sortConfig in HTML matches Python constants exactly."""

    @pytest.fixture
    def plan_path(self, tmp_path):
        p = tmp_path / "parity-plan.md"
        p.write_text("# Parity Plan\n\nContent.\n")
        return p

    @pytest.fixture
    def phase_dir(self, tmp_path):
        d = tmp_path / "review-plan"
        d.mkdir()
        return d

    def test_sort_config_matches_python_constants(self, plan_path, phase_dir):
        """sortConfig embedded in HTML exactly matches VALIDATION_ORDER and IMPORTANCE_ORDER."""
        groups = [
            {
                "theme": "Test group",
                "category": "test",
                "models": ["model-a"],
                "suggestions": [
                    {"title": "Test", "desc": "Desc", "importance": "HIGH", "source_model": "model-a"},
                ],
            },
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["model-a"],
        )

        data = extract_report_data(html)
        sort_config = data["sortConfig"]

        # Validation order parity
        # JSON keys are strings, Python dict has string keys too
        assert sort_config["validationOrder"] == VALIDATION_ORDER

        # Importance order parity
        assert sort_config["importanceOrder"] == IMPORTANCE_ORDER

        # Unknown rank parity
        assert sort_config["unknownStatusRank"] == 5
        assert sort_config["unknownImportanceRank"] == 3

    def test_sort_config_matches_build_sort_config(self, plan_path, phase_dir):
        """sortConfig in HTML matches build_sort_config() exactly."""
        groups = [
            {
                "theme": "Test group",
                "category": "test",
                "models": ["model-a"],
                "suggestions": [
                    {"title": "Test", "desc": "Desc", "importance": "MEDIUM", "source_model": "model-a"},
                ],
            },
        ]

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["model-a"],
        )

        data = extract_report_data(html)
        expected = build_sort_config()

        assert data["sortConfig"] == expected


# ============================================================================
# originalIndex Stability through Pipeline
# ============================================================================


class TestOriginalIndexPipelineStability:
    """Verify originalIndex values survive the full generate -> embed -> parse pipeline."""

    @pytest.fixture
    def plan_path(self, tmp_path):
        p = tmp_path / "stability-plan.md"
        p.write_text("# Stability Plan\n\nContent.\n")
        return p

    @pytest.fixture
    def phase_dir(self, tmp_path):
        d = tmp_path / "review-plan"
        d.mkdir()
        return d

    def test_original_index_values_match_pre_sort_positions(self, plan_path, phase_dir):
        """After sorting, originalIndex should match 1-based position in the original input."""
        groups = _make_groups_with_mixed_priority()

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["model-a", "model-b"],
        )

        data = extract_report_data(html)
        report_groups = data["groups"]

        # Build a map from theme to expected originalIndex
        # Input order: index 0="Low priority", 1="High priority", 2="Medium priority",
        #              3="Valid HIGH", 4="Pending"
        expected_indices = {
            "Low priority group": 1,
            "High priority group": 2,
            "Medium priority group": 3,
            "Valid HIGH group": 4,
            "Pending group": 5,
        }

        for group in report_groups:
            theme = group["theme"]
            assert group["originalIndex"] == expected_indices[theme], (
                f"Group '{theme}' has originalIndex={group['originalIndex']}, "
                f"expected {expected_indices[theme]}"
            )

    def test_all_original_indices_unique(self, plan_path, phase_dir):
        """Every group in the report should have a unique originalIndex."""
        groups = _make_groups_with_mixed_priority()

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["model-a", "model-b"],
        )

        data = extract_report_data(html)
        indices = [g["originalIndex"] for g in data["groups"]]

        assert len(indices) == len(set(indices)), (
            f"Duplicate originalIndex values found: {indices}"
        )

    def test_original_indices_contiguous_1_based(self, plan_path, phase_dir):
        """originalIndex values should form a contiguous 1-based range."""
        groups = _make_groups_with_mixed_priority()

        html = generate_html_report(
            groups=groups,
            plan_path=plan_path,
            phase_dir=phase_dir,
            phase_type="review-plan",
            models=["model-a", "model-b"],
        )

        data = extract_report_data(html)
        indices = sorted(g["originalIndex"] for g in data["groups"])

        assert indices == list(range(1, len(groups) + 1)), (
            f"originalIndex values are not contiguous 1-based: {indices}"
        )
