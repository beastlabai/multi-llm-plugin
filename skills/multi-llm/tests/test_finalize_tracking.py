"""Unit tests for finalize_tracking.py module."""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from finalize_tracking import get_all_modified_files, main


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_state(temp_dir):
    """Create a valid state file with pre_existing_changes."""
    state = {
        "pre_existing_changes": ["existing_file1.py", "existing_file2.py"],
        "created_at": "2025-01-01T00:00:00",
        "plan_path": "/path/to/plan.md"
    }
    state_path = temp_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state_path


@pytest.fixture
def empty_pre_existing_state(temp_dir):
    """Create a valid state file with empty pre_existing_changes."""
    state = {
        "pre_existing_changes": [],
        "created_at": "2025-01-01T00:00:00"
    }
    state_path = temp_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state_path


@pytest.fixture
def state_without_pre_existing(temp_dir):
    """Create a state file without pre_existing_changes field."""
    state = {
        "created_at": "2025-01-01T00:00:00"
    }
    state_path = temp_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state_path


@pytest.fixture
def invalid_json_state(temp_dir):
    """Create a state file with invalid JSON."""
    state_path = temp_dir / "state.json"
    state_path.write_text("{ invalid json content", encoding="utf-8")
    return state_path


@pytest.fixture
def non_object_state(temp_dir):
    """Create a state file that is not a JSON object."""
    state_path = temp_dir / "state.json"
    state_path.write_text('["array", "not", "object"]', encoding="utf-8")
    return state_path


@pytest.fixture
def invalid_pre_existing_type_state(temp_dir):
    """Create a state file where pre_existing_changes is not a list."""
    state = {
        "pre_existing_changes": "not_a_list",
        "created_at": "2025-01-01T00:00:00"
    }
    state_path = temp_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state_path


# --- TestGetAllModifiedFiles ---

class TestGetAllModifiedFiles:
    """Tests for get_all_modified_files function."""

    @patch("finalize_tracking.utils_get_modified_files")
    @patch("finalize_tracking.get_staged_files")
    def test_returns_empty_set_when_no_changes(self, mock_staged, mock_modified):
        """Returns empty set when there are no modified or staged files."""
        mock_modified.return_value = []
        mock_staged.return_value = []

        result = get_all_modified_files()

        assert result == set()

    @patch("finalize_tracking.utils_get_modified_files")
    @patch("finalize_tracking.get_staged_files")
    def test_returns_modified_files_only(self, mock_staged, mock_modified):
        """Returns only modified files when there are no staged files."""
        mock_modified.return_value = ["file1.py", "file2.py"]
        mock_staged.return_value = []

        result = get_all_modified_files()

        assert result == {"file1.py", "file2.py"}

    @patch("finalize_tracking.utils_get_modified_files")
    @patch("finalize_tracking.get_staged_files")
    def test_returns_staged_files_only(self, mock_staged, mock_modified):
        """Returns only staged files when there are no modified files."""
        mock_modified.return_value = []
        mock_staged.return_value = ["staged1.py", "staged2.py"]

        result = get_all_modified_files()

        assert result == {"staged1.py", "staged2.py"}

    @patch("finalize_tracking.utils_get_modified_files")
    @patch("finalize_tracking.get_staged_files")
    def test_returns_union_of_modified_and_staged(self, mock_staged, mock_modified):
        """Returns union of modified and staged files."""
        mock_modified.return_value = ["modified.py", "both.py"]
        mock_staged.return_value = ["staged.py", "both.py"]

        result = get_all_modified_files()

        assert result == {"modified.py", "staged.py", "both.py"}

    @patch("finalize_tracking.utils_get_modified_files")
    @patch("finalize_tracking.get_staged_files")
    def test_filters_empty_strings(self, mock_staged, mock_modified):
        """Filters out empty strings from results."""
        mock_modified.return_value = ["file1.py", "", "file2.py"]
        mock_staged.return_value = ["", "staged.py", ""]

        result = get_all_modified_files()

        assert result == {"file1.py", "file2.py", "staged.py"}
        assert "" not in result

    @patch("finalize_tracking.utils_get_modified_files")
    @patch("finalize_tracking.get_staged_files")
    def test_handles_exception_returns_empty_set(self, mock_staged, mock_modified):
        """Handles exceptions gracefully and returns empty set."""
        mock_modified.side_effect = Exception("Git error")

        result = get_all_modified_files()

        assert result == set()


# --- TestMainEntryPoint ---

class TestMainEntryPoint:
    """Tests for main() entry point function."""

    def test_missing_state_file_returns_error(self, temp_dir):
        """Returns error code when state file does not exist."""
        nonexistent_path = temp_dir / "nonexistent.json"

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(nonexistent_path)]):
            result = main()

        assert result == 1

    def test_invalid_json_returns_error(self, invalid_json_state):
        """Returns error code when state file contains invalid JSON."""
        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(invalid_json_state)]):
            result = main()

        assert result == 1

    def test_non_object_state_returns_error(self, non_object_state):
        """Returns error code when state file is not a JSON object."""
        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(non_object_state)]):
            result = main()

        assert result == 1

    @patch("finalize_tracking.get_all_modified_files")
    def test_normal_completion_returns_zero(self, mock_get_files, valid_state):
        """Returns zero on normal completion."""
        mock_get_files.return_value = {"new_file.py", "another.py", "existing_file1.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            result = main()

        assert result == 0

    @patch("finalize_tracking.get_all_modified_files")
    def test_dry_run_does_not_modify_state(self, mock_get_files, valid_state):
        """Dry run mode does not modify the state file."""
        mock_get_files.return_value = {"new_file.py", "another.py"}

        # Get original content
        original_content = valid_state.read_text(encoding="utf-8")

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state), "--dry-run"]):
            result = main()

        assert result == 0
        # State file should not be modified
        assert valid_state.read_text(encoding="utf-8") == original_content

    @patch("finalize_tracking.get_all_modified_files")
    def test_dry_run_returns_zero(self, mock_get_files, valid_state):
        """Dry run mode returns zero."""
        mock_get_files.return_value = {"new_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state), "--dry-run"]):
            result = main()

        assert result == 0


# --- TestStateFileUpdates ---

class TestStateFileUpdates:
    """Tests for state file update functionality."""

    @patch("finalize_tracking.get_all_modified_files")
    def test_tracked_files_written_correctly(self, mock_get_files, valid_state):
        """tracked_files list is written correctly to state."""
        mock_get_files.return_value = {"new_file.py", "another.py", "existing_file1.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            main()

        state = json.loads(valid_state.read_text(encoding="utf-8"))
        tracked_files = state["tracked_files"]

        # Should only include new files, not pre-existing ones
        tracked_paths = [f["path"] for f in tracked_files]
        assert "new_file.py" in tracked_paths
        assert "another.py" in tracked_paths
        # existing_file1.py was in pre_existing_changes, so should NOT be tracked
        assert "existing_file1.py" not in tracked_paths

    @patch("finalize_tracking.get_all_modified_files")
    def test_tracked_files_have_correct_structure(self, mock_get_files, valid_state):
        """Each tracked file entry has correct structure."""
        mock_get_files.return_value = {"new_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            main()

        state = json.loads(valid_state.read_text(encoding="utf-8"))
        tracked_files = state["tracked_files"]

        assert len(tracked_files) == 1
        entry = tracked_files[0]
        assert entry["path"] == "new_file.py"
        assert entry["action"] == "modified"
        assert entry["task_id"] == "implementation"

    @patch("finalize_tracking.get_all_modified_files")
    def test_updated_at_field_added(self, mock_get_files, valid_state):
        """updated_at field is added to state."""
        mock_get_files.return_value = {"new_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            main()

        state = json.loads(valid_state.read_text(encoding="utf-8"))

        assert "updated_at" in state
        # Should be a valid ISO format datetime
        datetime.fromisoformat(state["updated_at"])

    @patch("finalize_tracking.get_all_modified_files")
    def test_existing_fields_preserved(self, mock_get_files, valid_state):
        """Existing state fields are preserved after update."""
        mock_get_files.return_value = {"new_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            main()

        state = json.loads(valid_state.read_text(encoding="utf-8"))

        # Original fields should still be present
        assert state["created_at"] == "2025-01-01T00:00:00"
        assert state["plan_path"] == "/path/to/plan.md"
        assert "existing_file1.py" in state["pre_existing_changes"]

    @patch("finalize_tracking.get_all_modified_files")
    def test_handles_missing_pre_existing_changes(self, mock_get_files, state_without_pre_existing):
        """Handles state file without pre_existing_changes field."""
        mock_get_files.return_value = {"new_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(state_without_pre_existing)]):
            result = main()

        assert result == 0
        state = json.loads(state_without_pre_existing.read_text(encoding="utf-8"))
        tracked_paths = [f["path"] for f in state["tracked_files"]]
        assert "new_file.py" in tracked_paths

    @patch("finalize_tracking.get_all_modified_files")
    def test_handles_empty_pre_existing_changes(self, mock_get_files, empty_pre_existing_state):
        """Handles state file with empty pre_existing_changes list."""
        mock_get_files.return_value = {"file1.py", "file2.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(empty_pre_existing_state)]):
            result = main()

        assert result == 0
        state = json.loads(empty_pre_existing_state.read_text(encoding="utf-8"))
        tracked_paths = [f["path"] for f in state["tracked_files"]]
        assert "file1.py" in tracked_paths
        assert "file2.py" in tracked_paths

    @patch("finalize_tracking.get_all_modified_files")
    def test_handles_invalid_pre_existing_type_with_warning(self, mock_get_files, invalid_pre_existing_type_state, capsys):
        """Handles state where pre_existing_changes is not a list."""
        mock_get_files.return_value = {"new_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(invalid_pre_existing_type_state)]):
            result = main()

        # Should succeed with warning
        assert result == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "pre_existing_changes is not a list" in captured.err

    @patch("finalize_tracking.get_all_modified_files")
    def test_no_new_files_creates_empty_tracked_files(self, mock_get_files, valid_state):
        """When all modified files are pre-existing, tracked_files is empty."""
        # Only return files that are in pre_existing_changes
        mock_get_files.return_value = {"existing_file1.py", "existing_file2.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            result = main()

        assert result == 0
        state = json.loads(valid_state.read_text(encoding="utf-8"))
        assert state["tracked_files"] == []

    @patch("finalize_tracking.get_all_modified_files")
    def test_tracked_files_sorted_alphabetically(self, mock_get_files, valid_state):
        """Tracked files are sorted alphabetically by path."""
        mock_get_files.return_value = {"zebra.py", "alpha.py", "middle.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            main()

        state = json.loads(valid_state.read_text(encoding="utf-8"))
        tracked_paths = [f["path"] for f in state["tracked_files"]]

        assert tracked_paths == ["alpha.py", "middle.py", "zebra.py"]

    @patch("finalize_tracking.get_all_modified_files")
    def test_write_failure_returns_error(self, mock_get_files, valid_state):
        """Returns error when state file cannot be written."""
        mock_get_files.return_value = {"new_file.py"}

        # Make the file read-only to cause write failure
        valid_state.chmod(0o444)

        try:
            with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
                result = main()

            assert result == 1
        finally:
            # Restore permissions for cleanup
            valid_state.chmod(0o644)


# --- TestOutputMessages ---

class TestOutputMessages:
    """Tests for output messages from main function."""

    @patch("finalize_tracking.get_all_modified_files")
    def test_outputs_file_counts(self, mock_get_files, valid_state, capsys):
        """Outputs correct file counts."""
        mock_get_files.return_value = {"new1.py", "new2.py", "existing_file1.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            main()

        captured = capsys.readouterr()
        assert "Current modified files: 3" in captured.out
        assert "Pre-existing changes: 2" in captured.out
        assert "Implementation changes: 2" in captured.out

    @patch("finalize_tracking.get_all_modified_files")
    def test_outputs_tracked_files_list(self, mock_get_files, valid_state, capsys):
        """Outputs list of tracked files."""
        mock_get_files.return_value = {"new_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            main()

        captured = capsys.readouterr()
        assert "Tracked 1 implementation files:" in captured.out
        assert "new_file.py" in captured.out

    @patch("finalize_tracking.get_all_modified_files")
    def test_dry_run_outputs_would_track_message(self, mock_get_files, valid_state, capsys):
        """Dry run outputs 'would track' message."""
        mock_get_files.return_value = {"new_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state), "--dry-run"]):
            main()

        captured = capsys.readouterr()
        assert "Dry run - would track these files:" in captured.out
        assert "new_file.py" in captured.out

    def test_missing_state_file_outputs_error(self, temp_dir, capsys):
        """Outputs error message when state file is missing."""
        nonexistent_path = temp_dir / "nonexistent.json"

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(nonexistent_path)]):
            main()

        captured = capsys.readouterr()
        assert "ERROR: State file not found" in captured.err

    def test_invalid_json_outputs_error(self, invalid_json_state, capsys):
        """Outputs error message for invalid JSON."""
        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(invalid_json_state)]):
            main()

        captured = capsys.readouterr()
        assert "ERROR: Failed to parse state file" in captured.err

    def test_non_object_state_outputs_error(self, non_object_state, capsys):
        """Outputs error message when state is not an object."""
        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(non_object_state)]):
            main()

        captured = capsys.readouterr()
        assert "ERROR: State file is not a JSON object" in captured.err


# --- TestEdgeCases ---

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch("finalize_tracking.get_all_modified_files")
    def test_handles_no_modified_files(self, mock_get_files, valid_state):
        """Handles case when there are no modified files at all."""
        mock_get_files.return_value = set()

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            result = main()

        assert result == 0
        state = json.loads(valid_state.read_text(encoding="utf-8"))
        assert state["tracked_files"] == []

    @patch("finalize_tracking.get_all_modified_files")
    def test_handles_files_with_special_characters(self, mock_get_files, valid_state):
        """Handles files with special characters in paths."""
        mock_get_files.return_value = {"path/to/file with spaces.py", "file-with-dashes.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            result = main()

        assert result == 0
        state = json.loads(valid_state.read_text(encoding="utf-8"))
        tracked_paths = [f["path"] for f in state["tracked_files"]]
        assert "path/to/file with spaces.py" in tracked_paths
        assert "file-with-dashes.py" in tracked_paths

    @patch("finalize_tracking.get_all_modified_files")
    def test_handles_deeply_nested_paths(self, mock_get_files, valid_state):
        """Handles deeply nested file paths."""
        mock_get_files.return_value = {"a/b/c/d/e/f/deep_file.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            result = main()

        assert result == 0
        state = json.loads(valid_state.read_text(encoding="utf-8"))
        tracked_paths = [f["path"] for f in state["tracked_files"]]
        assert "a/b/c/d/e/f/deep_file.py" in tracked_paths

    @patch("finalize_tracking.get_all_modified_files")
    def test_handles_large_number_of_files(self, mock_get_files, valid_state):
        """Handles large number of modified files."""
        # Generate many file names
        many_files = {f"file_{i}.py" for i in range(100)}
        mock_get_files.return_value = many_files

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
            result = main()

        assert result == 0
        state = json.loads(valid_state.read_text(encoding="utf-8"))
        assert len(state["tracked_files"]) == 100

    @patch("finalize_tracking.get_all_modified_files")
    def test_pre_existing_changes_exactly_matches_modified(self, mock_get_files, temp_dir):
        """When pre_existing_changes exactly matches modified files."""
        state = {
            "pre_existing_changes": ["file1.py", "file2.py", "file3.py"],
            "created_at": "2025-01-01T00:00:00"
        }
        state_path = temp_dir / "state.json"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        mock_get_files.return_value = {"file1.py", "file2.py", "file3.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(state_path)]):
            result = main()

        assert result == 0
        updated_state = json.loads(state_path.read_text(encoding="utf-8"))
        assert updated_state["tracked_files"] == []

    @patch("finalize_tracking.get_all_modified_files")
    def test_state_with_existing_tracked_files_overwrites(self, mock_get_files, temp_dir):
        """Existing tracked_files in state are overwritten."""
        state = {
            "pre_existing_changes": [],
            "tracked_files": [{"path": "old.py", "action": "modified", "task_id": "old"}],
            "created_at": "2025-01-01T00:00:00"
        }
        state_path = temp_dir / "state.json"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        mock_get_files.return_value = {"new.py"}

        with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(state_path)]):
            result = main()

        assert result == 0
        updated_state = json.loads(state_path.read_text(encoding="utf-8"))
        tracked_paths = [f["path"] for f in updated_state["tracked_files"]]
        assert tracked_paths == ["new.py"]
        assert "old.py" not in tracked_paths
