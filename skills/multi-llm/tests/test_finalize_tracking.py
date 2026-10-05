"""Unit tests for finalize_tracking.py module."""

import json
import subprocess
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

@pytest.fixture(autouse=True)
def _no_ignore_rules(monkeypatch):
    """`finalize_tracking` drops gitignored `--touched` paths by asking git in
    the test process's cwd, whose own `.gitignore` covers `plans/`. These tests
    are about the census arithmetic, so the ignore lookup answers "none"; the
    ignore rule itself is covered where a throwaway repo carries one
    (`TestTheSweepLeavesGitignoredPathsOut`)."""
    import finalize_tracking
    monkeypatch.setattr(finalize_tracking, "ignored_paths",
                        lambda paths, cwd=None: set())


class TestOnlyTouchedWithEveryPathIgnored:
    """W7 (adversarial review, 2026-10-02): `_drop_ignored` runs before the
    `--only-touched` check, so a call whose every `--touched` path is
    gitignored used to fail with "needs at least one --touched <path>" —
    against a command line that passed three. Still a refusal (an empty
    explicit list would send review-code to its git-diff fallback), but it
    says what happened."""

    def test_the_refusal_names_the_ignored_paths(self, valid_state, monkeypatch,
                                                 capsys):
        import finalize_tracking
        monkeypatch.setattr(finalize_tracking, "ignored_paths",
                            lambda paths, cwd=None: set(paths))
        code = main([
            "--state-file", str(valid_state), "--only-touched",
            "--touched", "logs/a.jsonl", "--touched", "logs/b.jsonl",
        ])
        assert code == 1
        err = capsys.readouterr().err
        assert "every one is gitignored" in err
        assert "logs/a.jsonl" in err and "logs/b.jsonl" in err
        assert "needs at least one --touched" not in err

    def test_with_no_path_at_all_the_old_message_stands(self, valid_state, capsys):
        code = main(["--state-file", str(valid_state), "--only-touched"])
        assert code == 1
        assert "needs at least one --touched" in capsys.readouterr().err


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
        """Returns error when the state file cannot be written.

        The *directory* is what has to be unwritable now: the write is temp file
        + ``os.replace`` (the shape the comment above it always claimed and
        ``StateManager.save()`` has always used), and a rename over a read-only
        file succeeds as long as its directory is writable. The failure this
        pins is the real one — no room, or no permission, to land the new file.
        """
        mock_get_files.return_value = {"new_file.py"}

        directory = valid_state.parent
        original = directory.stat().st_mode
        directory.chmod(0o555)

        try:
            with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
                result = main()

            assert result == 1
            # ...and the state file the run could not update is still intact.
            assert json.loads(valid_state.read_text(encoding="utf-8"))
        finally:
            directory.chmod(original)

    @patch("finalize_tracking.get_all_modified_files")
    def test_an_interrupted_write_never_truncates_the_state_file(
        self, mock_get_files, valid_state
    ):
        """The census lands whole or not at all (F4).

        ``write_text`` truncates first and serializes second, so a crash between
        the two left the phase's own state file empty — and this is the one
        file-level writer that runs *inside* a write phase. With temp +
        ``os.replace`` a serialization that blows up leaves the original bytes
        exactly where they were.
        """
        mock_get_files.return_value = {"new_file.py"}
        before = valid_state.read_text(encoding="utf-8")

        with patch("finalize_tracking.json.dump", side_effect=OSError("disk full")):
            with patch("sys.argv", ["finalize_tracking.py", "--state-file", str(valid_state)]):
                result = main()

        assert result == 1
        assert valid_state.read_text(encoding="utf-8") == before
        # No temp file is left lying beside it either.
        assert not list(valid_state.parent.glob(".tracking_*"))


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


class TestTouchedFilesSurvivePreExistingSubtraction:
    """`--touched` names a file the write executor verified it changed; it is
    tracked even when it was already dirty when the phase started."""

    @pytest.fixture(autouse=True)
    def _outside_a_repository(self, tmp_path, monkeypatch):
        """These feed made-up census and `--touched` paths. Inside this
        checkout a `--touched` path that is on no disk and in no base commit
        is a file created and then removed, and is left out (fix round 8);
        outside any repository nothing can be read off a tree, so every
        path keeps the old `modified` and the arithmetic is all that is
        tested."""
        monkeypatch.chdir(tmp_path)

    @patch("finalize_tracking.get_all_modified_files")
    def test_a_pre_existing_file_the_run_touched_is_tracked(self, mock_get_files, valid_state, capsys):
        mock_get_files.return_value = {"existing_file1.py", "stray.md"}
        result = main(["--state-file", str(valid_state), "--touched", "existing_file1.py"])
        assert result == 0
        state = json.loads(valid_state.read_text(encoding="utf-8"))
        paths = [e["path"] for e in state["tracked_files"]]
        assert paths == ["existing_file1.py", "stray.md"]
        out = capsys.readouterr().out
        assert "Verified by the write executor: 1 (1 kept although modified before the phase started: existing_file1.py)" in out

    @patch("finalize_tracking.get_all_modified_files")
    def test_touched_is_repeatable_and_deduplicated(self, mock_get_files, valid_state):
        mock_get_files.return_value = {"new.py"}
        result = main([
            "--state-file", str(valid_state),
            "--touched", "new.py", "--touched", "existing_file2.py",
            "--touched", "existing_file2.py",
        ])
        assert result == 0
        state = json.loads(valid_state.read_text(encoding="utf-8"))
        assert [e["path"] for e in state["tracked_files"]] == ["existing_file2.py", "new.py"]

    @patch("finalize_tracking.get_all_modified_files")
    def test_without_touched_the_subtraction_is_unchanged(self, mock_get_files, valid_state, capsys):
        mock_get_files.return_value = {"existing_file1.py", "new.py"}
        assert main(["--state-file", str(valid_state)]) == 0
        state = json.loads(valid_state.read_text(encoding="utf-8"))
        assert [e["path"] for e in state["tracked_files"]] == ["new.py"]
        assert "Verified by the write executor" not in capsys.readouterr().out


class TestADirectoryBaselineCoversThePathsUnderIt:
    """A baseline entry is whatever ``git status --porcelain`` printed, and an
    untracked *directory* is one line standing for everything beneath it."""

    @pytest.fixture(autouse=True)
    def _outside_a_repository(self, tmp_path, monkeypatch):
        """These feed made-up census and `--touched` paths. Inside this
        checkout a `--touched` path that is on no disk and in no base commit
        is a file created and then removed, and is left out (fix round 8);
        outside any repository nothing can be read off a tree, so every
        path keeps the old `modified` and the arithmetic is all that is
        tested."""
        monkeypatch.chdir(tmp_path)

    @pytest.fixture
    def legacy_directory_baseline_state(self, temp_dir):
        """The shape every real state file carries: four untracked directories."""
        state = {
            "pre_existing_changes": [".claude/", ".multi-llm/", "plans/", "saved/"],
            "created_at": "2025-01-01T00:00:00",
        }
        state_path = temp_dir / "state.json"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state_path

    @patch("finalize_tracking.get_all_modified_files")
    def test_only_the_verified_files_survive_a_file_granular_census(
        self, mock_get_files, legacy_directory_baseline_state
    ):
        """The regression: a plain set difference subtracted none of these."""
        mock_get_files.return_value = {
            ".claude/video/thumb.dat",
            "saved/session.json",
            "plans/game.md",
            "plans/game/tasks/tasks.md",
        }

        result = main([
            "--state-file", str(legacy_directory_baseline_state),
            "--touched", "plans/game.md",
            "--touched", "plans/game/tasks/tasks.md",
        ])

        assert result == 0
        state = json.loads(legacy_directory_baseline_state.read_text(encoding="utf-8"))
        assert [e["path"] for e in state["tracked_files"]] == [
            "plans/game.md",
            "plans/game/tasks/tasks.md",
        ]

    @patch("finalize_tracking.get_all_modified_files")
    def test_a_touched_file_under_a_directory_baseline_is_reported_as_kept(
        self, mock_get_files, legacy_directory_baseline_state, capsys
    ):
        """The "kept although modified before the phase started" note reads the
        baseline the same prefix-aware way the subtraction does."""
        mock_get_files.return_value = {"plans/game.md"}

        main([
            "--state-file", str(legacy_directory_baseline_state),
            "--touched", "plans/game.md",
        ])

        out = capsys.readouterr().out
        assert "1 kept although modified before the phase started: plans/game.md" in out

    @patch("finalize_tracking.get_all_modified_files")
    def test_a_rerun_shrinks_a_bloated_tracked_files_list(
        self, mock_get_files, temp_dir
    ):
        """``tracked_files`` is replaced, never unioned, so the repair command
        (`util finalize-tracking --touched ...`) can undo a bad census."""
        state = {
            "pre_existing_changes": [".claude/", "plans/", "saved/"],
            "tracked_files": [
                {"path": f"saved/junk_{i}.json", "action": "modified",
                 "task_id": "implementation"}
                for i in range(1511)
            ],
            "created_at": "2025-01-01T00:00:00",
        }
        state_path = temp_dir / "state.json"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        mock_get_files.return_value = {"saved/junk_0.json", "plans/game.md"}

        result = main([
            "--state-file", str(state_path),
            "--touched", "plans/game.md",
        ])

        assert result == 0
        updated = json.loads(state_path.read_text(encoding="utf-8"))
        assert [e["path"] for e in updated["tracked_files"]] == ["plans/game.md"]


class TestADirectoryThePhaseCreatedIsExpandedToItsFiles:
    """F2: the census reports a new directory as one ``newdir/`` entry. Stored
    as-is, `capture_diff_hunks` finds no per-file diff for it and `review-code`
    silently reviews none of the new module — which is what the native leg
    (no ``--touched``) always gets."""

    @patch("finalize_tracking.list_untracked_files")
    @patch("finalize_tracking.get_all_modified_files")
    def test_an_uncovered_directory_becomes_the_files_inside_it(
        self, mock_get_files, mock_list, empty_pre_existing_state
    ):
        mock_get_files.return_value = {"src/physics/", "index.html"}
        mock_list.return_value = [
            "src/physics/body.js", "src/physics/world.js",
        ]

        assert main(["--state-file", str(empty_pre_existing_state)]) == 0

        state = json.loads(empty_pre_existing_state.read_text(encoding="utf-8"))
        assert [e["path"] for e in state["tracked_files"]] == [
            "index.html", "src/physics/body.js", "src/physics/world.js",
        ]
        mock_list.assert_called_once_with("src/physics/")

    @patch("finalize_tracking.list_untracked_files")
    @patch("finalize_tracking.get_all_modified_files")
    def test_a_baseline_covered_directory_is_never_expanded(
        self, mock_get_files, mock_list, temp_dir
    ):
        """The regression 4b3d14c reverted: expanding the baseline's own
        directories wrote all 1 511 untracked files in the repo into
        `tracked_files`. Only what SURVIVES the subtraction is expanded, so
        git is never even asked about `plans/`."""
        state = {
            "pre_existing_changes": [".claude/", "plans/", "saved/"],
            "created_at": "2025-01-01T00:00:00",
        }
        state_path = temp_dir / "state.json"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        mock_get_files.return_value = {"plans/", "saved/", "index.html"}

        assert main(["--state-file", str(state_path)]) == 0

        mock_list.assert_not_called()
        updated = json.loads(state_path.read_text(encoding="utf-8"))
        assert [e["path"] for e in updated["tracked_files"]] == ["index.html"]

    @patch("finalize_tracking.list_untracked_files")
    @patch("finalize_tracking.get_all_modified_files")
    def test_a_directory_git_cannot_list_is_kept_rather_than_dropped(
        self, mock_get_files, mock_list, empty_pre_existing_state
    ):
        """Losing the evidence entirely is worse than an unusable entry."""
        mock_get_files.return_value = {"vanished/"}
        mock_list.return_value = []

        assert main(["--state-file", str(empty_pre_existing_state)]) == 0

        state = json.loads(empty_pre_existing_state.read_text(encoding="utf-8"))
        assert [e["path"] for e in state["tracked_files"]] == ["vanished/"]


class TestTheTrackedFileDumpIsTruncated:
    """The run log is the only durable record of a detached run; a 1 500-line
    path dump buries everything else in it (state.json keeps the full list)."""

    @patch("finalize_tracking.get_all_modified_files")
    def test_only_the_first_twenty_paths_are_printed(
        self, mock_get_files, empty_pre_existing_state, capsys
    ):
        mock_get_files.return_value = {f"file_{i:03d}.py" for i in range(50)}

        main(["--state-file", str(empty_pre_existing_state)])

        out = capsys.readouterr().out
        assert "Tracked 50 implementation files:" in out
        assert "  - file_019.py" in out
        assert "  - file_020.py" not in out
        assert "  … and 30 more (see state.json.tracked_files)" in out
        state = json.loads(empty_pre_existing_state.read_text(encoding="utf-8"))
        assert len(state["tracked_files"]) == 50

    @patch("finalize_tracking.get_all_modified_files")
    def test_the_dry_run_branch_is_truncated_too(
        self, mock_get_files, empty_pre_existing_state, capsys
    ):
        mock_get_files.return_value = {f"file_{i:03d}.py" for i in range(25)}

        main(["--state-file", str(empty_pre_existing_state), "--dry-run"])

        out = capsys.readouterr().out
        assert "  - file_019.py" in out
        assert "  - file_020.py" not in out
        assert "  … and 5 more (see state.json.tracked_files)" in out

    @patch("finalize_tracking.get_all_modified_files")
    def test_a_short_list_is_printed_whole_without_a_tail(
        self, mock_get_files, empty_pre_existing_state, capsys
    ):
        mock_get_files.return_value = {"a.py", "b.py"}

        main(["--state-file", str(empty_pre_existing_state)])

        out = capsys.readouterr().out
        assert "  - a.py" in out and "  - b.py" in out
        assert "more (see state.json.tracked_files)" not in out


class TestTheSummaryBackReferenceIsNotReadAsAHumanEdit:
    """``--plan-file`` makes this command a *plan* writer, and it is the
    native leg's counterpart of `WriteBatchScheduler._finalize`'s re-stamp.
    Without it the next `StateManager` hashes the plugin's own back-reference
    insertion as drift and `implement --resume` warns about a change nobody
    made (F1, 2026-09-16)."""

    @pytest.fixture
    def plan_and_state(self, temp_dir):
        from utils.state_manager import compute_plan_hash

        plan = temp_dir / "plan.md"
        plan.write_text("# Plan\n\nSome content.\n", encoding="utf-8")
        state_path = temp_dir / "state.json"
        state_path.write_text(
            json.dumps({
                "pre_existing_changes": [],
                "plan_hash": compute_plan_hash(plan),
                "created_at": "2025-01-01T00:00:00",
            }),
            encoding="utf-8",
        )
        return plan, state_path

    @patch("finalize_tracking.get_all_modified_files")
    def test_the_stamped_hash_matches_the_plan_as_written(
        self, mock_get_files, plan_and_state
    ):
        from utils.state_manager import StateManager

        plan, state_path = plan_and_state
        mock_get_files.return_value = {"index.html"}

        code = main([
            "--state-file", str(state_path),
            "--plan-file", str(plan),
            "--summary-file-relative", "plan/implement/summary.md",
        ])

        assert code == 0
        assert "summary.md" in plan.read_text(encoding="utf-8")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["plan_hash"] == StateManager(plan)._compute_plan_hash()
        # The census this command exists for survives the second write.
        assert [e["path"] for e in state["tracked_files"]] == ["index.html"]

    @patch("finalize_tracking.get_all_modified_files")
    def test_a_state_file_that_records_no_hash_does_not_gain_one(
        self, mock_get_files, temp_dir
    ):
        """Nothing reads a field this state file never had; do not invent it."""
        plan = temp_dir / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        state_path = temp_dir / "state.json"
        state_path.write_text(
            json.dumps({"pre_existing_changes": []}), encoding="utf-8"
        )
        mock_get_files.return_value = set()

        code = main([
            "--state-file", str(state_path),
            "--plan-file", str(plan),
            "--summary-file-relative", "plan/implement/summary.md",
        ])

        assert code == 0
        assert "plan_hash" not in json.loads(state_path.read_text(encoding="utf-8"))


# --- `--only-touched`: the exact-scope route (review-scope plan §2 D2.6) ---
#
# These run the REAL census in a real temp git repo (`monkeypatch.chdir`), not a
# mocked `get_all_modified_files`: the thing under test is what plain `--touched`
# does with the unrelated dirty paths a no-baseline repository carries, and a
# clean repo or a mocked census cannot show that union.


def _git(repo, *args):
    import subprocess

    subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True,
    )


@pytest.fixture
def dirty_repo_without_baseline(tmp_path, monkeypatch):
    """A repo holding the intended change and three unrelated dirty paths.

    ``app.py`` is the intended change. ``unrelated.py`` is a committed file
    modified outside it, ``notes.txt`` an unrelated untracked file and
    ``scratch/`` an untracked directory. The state file records NO
    ``pre_existing_changes`` — the external-implementer case — and lives outside
    the repo so it is not itself a dirty path.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "unrelated.py").write_text("y = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py", "unrelated.py")
    _git(repo, "commit", "--quiet", "-m", "init")

    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "unrelated.py").write_text("y = 2\n", encoding="utf-8")
    (repo / "notes.txt").write_text("scratch notes\n", encoding="utf-8")
    (repo / "scratch").mkdir()
    (repo / "scratch" / "tmp.log").write_text("log\n", encoding="utf-8")

    state_dir = tmp_path / "plan-out"
    state_dir.mkdir()
    state_path = state_dir / "state.json"
    state_path.write_text(
        json.dumps({"created_at": "2026-09-28T00:00:00"}, indent=2), encoding="utf-8"
    )
    monkeypatch.chdir(repo)
    return repo, state_path


def _tracked(state_path):
    return [e["path"] for e in json.loads(state_path.read_text(encoding="utf-8"))["tracked_files"]]


@pytest.fixture
def racing_repo(tmp_path, monkeypatch):
    """The Codex racing run 3 tree after a partial implement (2026-10-05).

    ``plans/racing-game.md`` is committed and then edited by apply-suggestions
    (it carries the ``TASKS_FILE`` marker generate-tasks wrote); the output
    folder ``plans/racing-game/`` holds ``state.json`` and the task list; and
    implement created ``index.html``. The baseline is the one apply-suggestions
    recorded at its own start, before it edited the plan: the output folder.
    """
    repo = tmp_path / "repo"
    (repo / "plans").mkdir(parents=True)
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    plan = repo / "plans" / "racing-game.md"
    plan.write_text("# Racing game\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("speed = 1\n", encoding="utf-8")
    _git(repo, "add", "plans/racing-game.md", "src/app.py")
    _git(repo, "commit", "--quiet", "-m", "plan")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    plan.write_text(
        "# Racing game\n\nApplied suggestion.\n\n"
        "<!-- TASKS_FILE: racing-game/tasks/tasks.md -->\n",
        encoding="utf-8",
    )
    tasks = repo / "plans" / "racing-game" / "tasks" / "tasks.md"
    tasks.parent.mkdir(parents=True)
    tasks.write_text("## T001: Scene\n", encoding="utf-8")
    (repo / "index.html").write_text("<canvas></canvas>\n", encoding="utf-8")
    state_path = repo / "plans" / "racing-game" / "state.json"
    state_path.write_text(json.dumps({
        "plan_path": str(plan),
        "pre_existing_changes": ["plans/racing-game/"],
        "head_before_implement": head,
    }, indent=2), encoding="utf-8")
    monkeypatch.chdir(repo)
    return repo, state_path


def _entries(state_path):
    return json.loads(state_path.read_text(encoding="utf-8"))["tracked_files"]


class TestTheWorkflowsOwnFilesAreNeverTracked:
    """2026-10-05, Codex racing run 3: ``tracked_files`` held the plan and
    ``tasks.md`` (apply-task-suggestions' census) and review-code reviewed
    ``tasks.md`` as code; on OpenCode the plan an earlier phase edited was
    recorded ``modified`` by implementation."""

    def test_the_plan_and_the_task_list_are_left_out(self, racing_repo, capsys):
        _repo, state_path = racing_repo

        code = main([
            "--state-file", str(state_path),
            "--touched", "index.html",
            "--touched", "plans/racing-game/tasks/tasks.md",
        ])

        assert code == 0
        assert _entries(state_path) == [
            {"path": "index.html", "action": "created", "task_id": "implementation"},
        ]
        out = capsys.readouterr().out
        assert "Left out 2 of the workflow's own file(s)" in out
        assert "plans/racing-game.md" in out

    def test_a_plan_path_from_a_moved_repository_still_finds_the_plan(
        self, racing_repo
    ):
        """`plan_path` is stored absolute; in a copied or moved repository it
        names a plan elsewhere, and the plan beside the output folder is the
        one that must stay out."""
        _repo, state_path = racing_repo
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["plan_path"] = "/elsewhere/plans/racing-game.md"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        assert main(["--state-file", str(state_path), "--touched", "index.html"]) == 0
        assert [e["path"] for e in _entries(state_path)] == ["index.html"]

    def test_only_touched_given_only_workflow_files_is_refused(
        self, racing_repo, capsys
    ):
        _repo, state_path = racing_repo
        before = state_path.read_text(encoding="utf-8")

        code = main([
            "--state-file", str(state_path), "--only-touched",
            "--touched", "plans/racing-game.md",
        ])

        assert code == 1
        assert "only the workflow's own files" in capsys.readouterr().err
        assert state_path.read_text(encoding="utf-8") == before


class TestEachCodePhaseKeepsItsOwnEntries:
    """``tracked_files`` names what each code phase changed: apply-code-fixes
    never re-labels a file implement created, and a resumed implement never
    takes over one apply-code-fixes created."""

    def _record(self, state_path, entries):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["tracked_files"] = entries
        state_path.write_text(json.dumps(state), encoding="utf-8")

    def test_apply_code_fixes_adds_its_files_beside_implements(self, racing_repo):
        repo, state_path = racing_repo
        self._record(state_path, [
            {"path": "index.html", "action": "created", "task_id": "implementation"},
        ])
        (repo / "index.html").write_text("<canvas id=c></canvas>\n", encoding="utf-8")
        (repo / "fix.js").write_text("// fix\n", encoding="utf-8")

        code = main([
            "--state-file", str(state_path), "--phase", "apply-code-fixes",
            "--touched", "index.html", "--touched", "fix.js",
        ])

        assert code == 0
        assert _entries(state_path) == [
            {"path": "fix.js", "action": "created", "task_id": "apply-code-fixes"},
            {"path": "index.html", "action": "created", "task_id": "implementation"},
        ]

    def test_a_resumed_implement_keeps_apply_code_fixes_entries(self, racing_repo):
        repo, state_path = racing_repo
        (repo / "fix.js").write_text("// fix\n", encoding="utf-8")
        self._record(state_path, [
            {"path": "fix.js", "action": "created", "task_id": "apply-code-fixes"},
            {"path": "index.html", "action": "created", "task_id": "implementation"},
            # An older run's planning-phase entry: replaced, never kept.
            {"path": "plans/racing-game.md", "action": "modified",
             "task_id": "implementation"},
        ])

        code = main(["--state-file", str(state_path), "--touched", "index.html"])

        assert code == 0
        assert _entries(state_path) == [
            {"path": "fix.js", "action": "created", "task_id": "apply-code-fixes"},
            {"path": "index.html", "action": "created", "task_id": "implementation"},
        ]


class TestAKeptEntryFollowsTheTree:
    """Fix round 8 (review of round 7): the other code phase's entries were
    copied verbatim, so after apply-code-fixes deleted the ``index.html``
    implement created, ``tracked_files`` still read ``index.html created
    implementation`` and review-code was sent to a file that no longer
    exists. A kept entry keeps the label of the phase that recorded it, and
    its action is read off the tree again; a file that is neither on disk nor
    in the base commit has nothing left to review and is left out, said once."""

    def _record(self, state_path, entries):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["tracked_files"] = entries
        state_path.write_text(json.dumps(state), encoding="utf-8")

    def test_a_created_file_a_later_phase_deleted_leaves_the_list(
        self, racing_repo, capsys
    ):
        repo, state_path = racing_repo
        self._record(state_path, [
            {"path": "index.html", "action": "created", "task_id": "implementation"},
        ])
        (repo / "index.html").unlink()
        (repo / "fix.js").write_text("// fix\n", encoding="utf-8")

        code = main([
            "--state-file", str(state_path), "--phase", "apply-code-fixes",
            "--touched", "fix.js",
        ])

        assert code == 0
        assert _entries(state_path) == [
            {"path": "fix.js", "action": "created", "task_id": "apply-code-fixes"},
        ]
        out = capsys.readouterr().out
        assert (
            "Left out 1 file(s) that no longer exist and are not in the base "
            "commit" in out
        )
        assert "index.html (implementation)" in out

    def test_a_kept_entry_says_what_the_tree_shows_now(self, racing_repo):
        repo, state_path = racing_repo
        self._record(state_path, [
            {"path": "src/app.py", "action": "modified", "task_id": "implementation"},
        ])
        (repo / "src" / "app.py").unlink()
        (repo / "index.html").unlink()

        code = main(["--state-file", str(state_path), "--phase", "apply-code-fixes"])

        assert code == 0
        assert _entries(state_path) == [
            {"path": "src/app.py", "action": "deleted", "task_id": "implementation"},
        ]

    def test_a_touched_file_the_phase_created_and_removed_is_left_out(
        self, racing_repo, capsys
    ):
        repo, state_path = racing_repo
        # A batch created scratch.js and a later batch of the same phase
        # removed it: the executor's --touched set still names it.
        code = main([
            "--state-file", str(state_path),
            "--touched", "index.html", "--touched", "scratch.js",
        ])

        assert code == 0
        assert [e["path"] for e in _entries(state_path)] == ["index.html"]
        assert "scratch.js (implementation)" in capsys.readouterr().out


class TestOnlyTouchedIsTheExactScope:
    def test_only_the_touched_paths_are_tracked(self, dirty_repo_without_baseline, capsys):
        _repo, state_path = dirty_repo_without_baseline

        code = main([
            "--state-file", str(state_path), "--only-touched", "--touched", "app.py",
        ])

        assert code == 0
        tracked = _tracked(state_path)
        assert tracked == ["app.py"]
        for unrelated in ("unrelated.py", "notes.txt", "scratch/", "scratch/tmp.log"):
            assert unrelated not in tracked
        out = capsys.readouterr().out
        assert "Explicit scope (--only-touched): 1 files" in out
        # The census was not consulted, so its lines are not printed either.
        assert "Current modified files" not in out
        # ...and it wrote no baseline of its own.
        assert "pre_existing_changes" not in json.loads(
            state_path.read_text(encoding="utf-8")
        )

    def test_plain_touched_still_unions_the_unrelated_dirty_paths(
        self, dirty_repo_without_baseline
    ):
        """Pins the documented "adds, never narrows" behaviour of plain
        `--touched`, so a later change to it is a deliberate one."""
        _repo, state_path = dirty_repo_without_baseline

        code = main(["--state-file", str(state_path), "--touched", "app.py"])

        assert code == 0
        tracked = _tracked(state_path)
        assert "app.py" in tracked
        assert "unrelated.py" in tracked
        assert "notes.txt" in tracked
        # The untracked directory the (baseline-less) census reports is expanded
        # to its file, as for any directory the subtraction did not cover.
        assert "scratch/tmp.log" in tracked

    def test_the_touched_paths_are_deduplicated_and_sorted(
        self, dirty_repo_without_baseline
    ):
        _repo, state_path = dirty_repo_without_baseline

        code = main([
            "--state-file", str(state_path), "--only-touched",
            "--touched", "b/z.py", "--touched", "app.py", "--touched", "b/z.py",
        ])

        assert code == 0
        assert _tracked(state_path) == ["app.py", "b/z.py"]

    def test_only_touched_without_touched_exits_1_and_leaves_the_state_alone(
        self, dirty_repo_without_baseline, capsys
    ):
        _repo, state_path = dirty_repo_without_baseline
        before = state_path.read_bytes()

        code = main(["--state-file", str(state_path), "--only-touched"])

        assert code == 1
        assert state_path.read_bytes() == before
        assert "--only-touched needs at least one --touched" in capsys.readouterr().err

    def test_an_existing_baseline_is_ignored_and_left_untouched(
        self, dirty_repo_without_baseline
    ):
        """A `--touched` path the baseline covers is still tracked, and the
        baseline — even a malformed one the census path would rewrite — is
        neither read nor written."""
        _repo, state_path = dirty_repo_without_baseline
        for baseline in (["app.py", "unrelated.py"], "not-a-list"):
            state_path.write_text(
                json.dumps({"pre_existing_changes": baseline}), encoding="utf-8"
            )

            code = main([
                "--state-file", str(state_path), "--only-touched",
                "--touched", "app.py",
            ])

            assert code == 0
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert [e["path"] for e in state["tracked_files"]] == ["app.py"]
            assert state["pre_existing_changes"] == baseline

    def test_the_touched_help_says_it_never_narrows(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        help_text = " ".join(capsys.readouterr().out.split())
        assert "never removes a dirty one" in help_text
        assert "--only-touched" in help_text


class TestTrackedFilesSayWhatHappenedToEachFile:
    """2026-10-04 (Claude Code racing re-run): `implement` created
    `index.html` and `tracked_files` recorded it as `modified` — every entry
    was `modified`, whatever the run did. The action is now read off the
    implement baseline: a path absent from `head_before_implement`'s tree and
    not in `pre_existing_changes` is `created`, a path in that tree and gone
    from disk `deleted`, everything else `modified`."""

    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "--quiet")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test")
        _git(repo, "config", "commit.gpgsign", "false")
        for name in ("kept.py", "gone.py"):
            (repo / name).write_text("x = 1\n", encoding="utf-8")
        _git(repo, "add", "kept.py", "gone.py")
        _git(repo, "commit", "--quiet", "-m", "init")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        # Dirty before the phase started: an untracked file it later edits.
        (repo / "notes.txt").write_text("before\n", encoding="utf-8")
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "head_before_implement": head,
            "pre_existing_changes": ["notes.txt"],
        }), encoding="utf-8")
        monkeypatch.chdir(repo)
        # What the phase did.
        (repo / "index.html").write_text("<p>new</p>\n", encoding="utf-8")
        (repo / "kept.py").write_text("x = 2\n", encoding="utf-8")
        (repo / "gone.py").unlink()
        (repo / "notes.txt").write_text("after\n", encoding="utf-8")
        (repo / "staged_new.py").write_text("z = 1\n", encoding="utf-8")
        _git(repo, "add", "staged_new.py")
        return repo, state_path

    @staticmethod
    def _actions(state_path):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return {entry["path"]: entry["action"] for entry in state["tracked_files"]}

    def test_the_census_records_created_modified_and_deleted(self, repo):
        _repo, state_path = repo
        assert main([
            "--state-file", str(state_path), "--touched", "notes.txt",
        ]) == 0
        assert self._actions(state_path) == {
            "index.html": "created",
            "staged_new.py": "created",
            "kept.py": "modified",
            "gone.py": "deleted",
            # Untracked before the phase began: it existed, so an edit.
            "notes.txt": "modified",
        }

    def test_a_removed_file_the_baseline_names_is_deleted(self, repo):
        """`notes.txt` was untracked before the phase began (the baseline
        names it) and the phase removed it: a deletion, not an edit."""
        repo_dir, state_path = repo
        (repo_dir / "notes.txt").unlink()
        assert main([
            "--state-file", str(state_path), "--touched", "notes.txt",
        ]) == 0
        assert self._actions(state_path)["notes.txt"] == "deleted"

    def test_only_touched_records_a_new_file_as_created(self, repo):
        _repo, state_path = repo
        assert main([
            "--state-file", str(state_path), "--only-touched",
            "--touched", "index.html", "--touched", "kept.py",
        ]) == 0
        assert self._actions(state_path) == {
            "index.html": "created", "kept.py": "modified",
        }

    def test_without_head_before_implement_head_is_the_base(self, repo):
        _repo, state_path = repo
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.pop("head_before_implement")
        state_path.write_text(json.dumps(state), encoding="utf-8")
        assert main(["--state-file", str(state_path)]) == 0
        actions = self._actions(state_path)
        assert actions["index.html"] == "created"
        assert actions["kept.py"] == "modified"


class TestMarkCompletedIsTheNativeLegsClosingWrite:
    """`--mark-completed` records implement completed and clears its start.

    The orchestrator records the phase *started* when it emits the task JSON
    (2026-10-05); this flag on the instruction's closing call is the only
    place the native path records it completed, the counterpart of
    `WriteBatchScheduler._finalize`.
    """

    def _state(self, temp_dir, **extra):
        state = {
            "pre_existing_changes": [],
            "created_at": "2025-01-01T00:00:00",
            "plan_path": "/path/to/plan.md",
            "phases_started": {"implement": "2026-10-05T10:43:30"},
            **extra,
        }
        path = temp_dir / "state.json"
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return path

    def test_without_the_flag_the_phase_stays_started(self, temp_dir, capsys):
        path = self._state(temp_dir)
        with patch("finalize_tracking.get_all_modified_files", return_value=[]):
            assert main(["--state-file", str(path)]) == 0
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert "implement" in saved["phases_started"]
        assert "implement" not in saved.get("phases_completed", {})
        assert "marked completed" not in capsys.readouterr().out

    def test_with_the_flag_the_start_becomes_a_completion(self, temp_dir, capsys):
        path = self._state(temp_dir)
        with patch("finalize_tracking.get_all_modified_files", return_value=[]):
            assert main(["--state-file", str(path), "--mark-completed"]) == 0
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(saved["phases_completed"]["implement"], str)
        assert "implement" not in saved["phases_started"]
        assert "Phase implement marked completed" in capsys.readouterr().out

    def test_the_flag_reads_back_through_the_state_manager(self, temp_dir):
        """`is_phase_completed` sees the raw write, so `status`, the next-step
        builder and the completed-phase guard agree with it."""
        from utils.state_manager import StateManager

        plan = temp_dir / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        state = StateManager(plan)
        state.mark_phase_started("implement")
        state.save()
        with patch("finalize_tracking.get_all_modified_files", return_value=[]):
            assert main(["--state-file", str(state.state_file), "--mark-completed"]) == 0
        reloaded = StateManager(plan)
        assert reloaded.is_phase_completed("implement")
        assert not reloaded.is_phase_started("implement")

    def test_a_dry_run_never_marks(self, temp_dir):
        path = self._state(temp_dir)
        with patch("finalize_tracking.get_all_modified_files", return_value=[]):
            assert main(["--state-file", str(path), "--mark-completed", "--dry-run"]) == 0
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert "implement" not in saved.get("phases_completed", {})
