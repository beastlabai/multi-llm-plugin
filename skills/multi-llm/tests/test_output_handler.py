"""Tests for output_handler utilities."""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.output_handler import sanitize_prefix, get_output_paths


class TestSanitizePrefix:
    """Tests for the sanitize_prefix() function."""

    def test_simple_name(self):
        """Test simple name passes through."""
        assert sanitize_prefix("my-feature") == "my-feature"

    def test_with_md_extension(self):
        """Test .md extension is removed."""
        assert sanitize_prefix("my-feature.md") == "my-feature"

    def test_spaces_replaced(self):
        """Test spaces are replaced with underscores."""
        assert sanitize_prefix("My Feature Plan") == "My_Feature_Plan"

    def test_special_chars_replaced(self):
        """Test special characters are replaced with underscores."""
        assert sanitize_prefix("feature@v1.0!") == "feature_v1_0"

    def test_consecutive_underscores_collapsed(self):
        """Test consecutive underscores are collapsed."""
        assert sanitize_prefix("my__feature") == "my_feature"

    def test_leading_trailing_underscores_stripped(self):
        """Test leading/trailing underscores are stripped."""
        assert sanitize_prefix("_feature_") == "feature"

    def test_hyphen_preserved(self):
        """Test hyphens are preserved."""
        assert sanitize_prefix("my-feature-plan") == "my-feature-plan"

    def test_underscore_preserved(self):
        """Test underscores are preserved."""
        assert sanitize_prefix("my_feature_plan") == "my_feature_plan"

    def test_complex_name(self):
        """Test complex name with multiple issues."""
        assert sanitize_prefix("My Plan (v2.0).md") == "My_Plan_v2_0"

    def test_numbers_preserved(self):
        """Test numbers are preserved."""
        assert sanitize_prefix("feature123") == "feature123"


class TestGetOutputPaths:
    """Tests for the get_output_paths() function."""

    def test_creates_subfolder(self, tmp_path):
        """Test that get_output_paths creates the plan subfolder and phase subfolder."""
        plan_file = tmp_path / "test-plan.md"
        plan_file.write_text("# Test Plan")

        output_path = get_output_paths(plan_file, "reviews")

        # Verify plan subfolder was created
        expected_plan_dir = tmp_path / "test-plan"
        assert expected_plan_dir.exists()
        assert expected_plan_dir.is_dir()

        # Verify phase subfolder was created (reviews -> review-plan)
        expected_phase_dir = expected_plan_dir / "review-plan"
        assert expected_phase_dir.exists()
        assert expected_phase_dir.is_dir()

        # Verify returned path is in phase subfolder with simplified name
        assert output_path.parent == expected_phase_dir
        assert output_path.name == "report.md"

    def test_output_path_format(self, tmp_path):
        """Test that output paths have correct format with simplified names."""
        plan_file = tmp_path / "my-feature.md"
        plan_file.write_text("# Test Plan")

        # Test various output types - now uses simplified filenames
        assert get_output_paths(plan_file, "reviews").name == "report.md"
        assert get_output_paths(plan_file, "grouped").name == "grouped.json"
        assert get_output_paths(plan_file, "validation").name == "validation.json"
        assert get_output_paths(plan_file, "backup").name == "backup.md"

        # Test that files are in correct phase subdirectories
        assert get_output_paths(plan_file, "reviews").parent.name == "review-plan"
        assert get_output_paths(plan_file, "grouped").parent.name == "review-plan"
        assert get_output_paths(plan_file, "validation").parent.name == "review-plan"
        assert get_output_paths(plan_file, "backup").parent.name == "review-plan"

    def test_existing_subfolder_no_error(self, tmp_path):
        """Test that existing subfolder doesn't cause errors."""
        plan_file = tmp_path / "test-plan.md"
        plan_file.write_text("# Test Plan")

        # Pre-create the plan subfolder and phase subfolder
        plan_subfolder = tmp_path / "test-plan"
        plan_subfolder.mkdir()
        phase_subfolder = plan_subfolder / "review-plan"
        phase_subfolder.mkdir()

        # Should not raise
        output_path = get_output_paths(plan_file, "reviews")
        assert output_path.parent == phase_subfolder

    def test_special_chars_sanitized_in_folder_name(self, tmp_path):
        """Test that special chars in plan name are sanitized for folder."""
        plan_file = tmp_path / "My Feature Plan.md"
        plan_file.write_text("# Test Plan")

        output_path = get_output_paths(plan_file, "reviews")

        # Plan folder should be sanitized
        expected_plan_dir = tmp_path / "My_Feature_Plan"
        assert expected_plan_dir.exists()

        # Phase folder should be inside the sanitized plan folder
        expected_phase_dir = expected_plan_dir / "review-plan"
        assert output_path.parent == expected_phase_dir
        assert expected_phase_dir.exists()

    def test_original_plan_location_unchanged(self, tmp_path):
        """Test that original plan file stays in place."""
        plan_file = tmp_path / "test-plan.md"
        plan_file.write_text("# Test Plan")

        get_output_paths(plan_file, "reviews")

        # Original plan should still exist at original location
        assert plan_file.exists()
        assert plan_file.parent == tmp_path  # Not in subfolder

    def test_code_review_output_types(self, tmp_path):
        """Test code review specific output types."""
        plan_file = tmp_path / "test-plan.md"
        plan_file.write_text("# Test Plan")

        assert get_output_paths(plan_file, "code_review").suffix == ".md"
        assert get_output_paths(plan_file, "code_review_issues").suffix == ".json"
        assert get_output_paths(plan_file, "code_review_grouped").suffix == ".json"
        assert get_output_paths(plan_file, "code_review_validation").suffix == ".json"
